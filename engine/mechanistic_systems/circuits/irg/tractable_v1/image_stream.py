"""Bounded requested-ID image decoding; no sampling or production authority."""
from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import threading
import tempfile
from typing import Iterator, Sequence

import numpy as np
from PIL import Image, __version__ as pillow_version
import torch
import torchvision
from torchvision import transforms

from mechanistic_systems.contracts.content_address import content_hash
from .alexnet_factors import IMAGENET_MEAN, IMAGENET_STD
from .data import StratifiedPopulation, _sha256_lines

IMAGE_SHAPE = (3, 224, 224)
UINT8_IMAGE_BYTES = 3 * 224 * 224
FLOAT32_IMAGE_BYTES = 4 * UINT8_IMAGE_BYTES


def _file_stamp(path: str) -> tuple[int, int, int, int]:
    value = Path(path).stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def ordered_context_id(stratum: str, listing_index: int) -> str:
    """String coordinate meaningful only with the bound population identity."""
    if stratum not in ('target', 'complement') or type(listing_index) is not int or listing_index < 0:
        raise ValueError('invalid image context coordinate')
    return f'{stratum}:{listing_index}'


@dataclass(frozen=True)
class ImageBatch:
    stratum: str
    context_ids: tuple[int, ...]
    images: torch.Tensor
    source_identity_hash: str
    source_batch_receipt_payload: dict
    source_batch_receipt_hash: str

    @property
    def ordered_context_ids(self) -> tuple[str, ...]:
        return tuple(ordered_context_id(self.stratum, i) for i in self.context_ids)

    @property
    def labels(self) -> torch.Tensor:
        return torch.full((len(self.context_ids),), float(self.stratum == 'target'), dtype=torch.float64)


@dataclass(frozen=True)
class _CacheEntry:
    pixels: torch.Tensor
    file_stamp: tuple[int, int, int, int]


class BoundedImageSource:
    """Immutable listing/permutation snapshot with bounded uint8 image payloads.

    ``iter_indices`` streams caller-selected IDs. The compatibility callable
    materializes one bounded output request. Neither API selects draws or
    establishes role independence. Source factors, decoder buffers, metadata,
    thread stacks and caller-retained output batches are separate memory owners.
    """
    def __init__(self, population: StratifiedPopulation, *, device: torch.device | str = 'cpu',
                 workers: int = 8, cache_bytes: int = 256 << 20, prefetch_images: int = 16,
                 batch_size: int = 64, max_materialized_bytes: int = 1 << 30) -> None:
        for name, value in dict(workers=workers, cache_bytes=cache_bytes, prefetch_images=prefetch_images).items():
            if type(value) is not int or value < 0: raise ValueError(f'{name} must be a nonnegative integer')
        for name, value in dict(batch_size=batch_size, max_materialized_bytes=max_materialized_bytes).items():
            if type(value) is not int or value <= 0: raise ValueError(f'{name} must be a positive integer')
        if batch_size * FLOAT32_IMAGE_BYTES > max_materialized_bytes:
            raise ValueError('one configured image batch exceeds the materialized output budget')
        self.device = torch.device(device)
        self.workers, self.cache_bytes, self.prefetch_images = workers, cache_bytes, prefetch_images
        self.batch_size, self.max_materialized_bytes = batch_size, max_materialized_bytes
        identity = population.identity()
        self._paths = {}; self._permutations = {}
        for stratum, count in zip(('target', 'complement'), population.counts, strict=True):
            self._paths[stratum] = tuple(population.paths_for_indices(stratum, np.arange(count, dtype=np.int64)))
            # Validate the raw owner before any integer cast can hide malformed IDs.
            raw = np.asarray(population._stratum_axis(stratum)[0])
            if raw.ndim != 1 or raw.dtype.kind not in 'iu' or len(raw) != count:
                raise ValueError('population permutation must be a full integer bijection')
            frozen = np.array(raw, copy=True)
            if count and (np.any(frozen < 0) or np.any(frozen >= count)):
                raise ValueError('population permutation contains an out-of-range ID')
            frozen = frozen.astype(np.int64, copy=False)
            if count and not np.all(np.bincount(frozen, minlength=count) == 1):
                raise ValueError('population permutation must be a full integer bijection')
            frozen.flags.writeable = False
            self._permutations[stratum] = frozen
            if _sha256_lines(self._paths[stratum]) != identity[f'{stratum}_listing_sha256']:
                raise ValueError('population listing changed while constructing image source')
        if population.identity() != identity: raise ValueError('population changed while constructing image source')
        for stratum, frozen in self._permutations.items():
            live = np.asarray(population._stratum_axis(stratum)[0])
            if live.dtype.kind not in 'iu' or not np.array_equal(live, frozen):
                raise ValueError('population permutation changed while constructing image source')
        self._transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.PILToTensor()])
        self._identity = dict(schema='tractable_v1_bounded_image_source_identity_v1', population=identity,
            permutations_sha256={s: hashlib.sha256(p.tobytes()).hexdigest() for s, p in self._permutations.items()},
            preprocessing=dict(rgb=True, transform=repr(self._transform), normalization='separate_fp32_div255_submean_divstd',
                               mean=IMAGENET_MEAN, std=IMAGENET_STD, torch=torch.__version__,
                               torchvision=torchvision.__version__, pillow=pillow_version),
            dataset_bytes_certified=False, sampling_authority=False, production_authority=False)
        self.source_identity_hash = content_hash(self._identity, prefix='tractable-v1-image-source-')
        self._mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self._cache: OrderedDict[tuple[str, int], _CacheEntry] = OrderedDict()
        self._query_lock = threading.Lock(); self._closed = False
        self._content_manifest = None
        self._content_manifest_lease = None
        self._stats = dict(cache_bytes=0, cache_peak_bytes=0, cache_hits=0, decode_requests_submitted=0,
                           evictions=0, prefetch_peak_images=0, batch_uint8_peak_bytes=0, batches_yielded=0)

    def identity(self) -> dict:
        return json.loads(json.dumps(self._identity))

    def freeze_content_manifest(self) -> str:
        """Hash every registered file before the first decoded observation.

        This is one-time source staging, with a 1 MiB copy buffer and O(population)
        immutable digest metadata. It neither samples contexts nor decodes images.
        A fresh source must repeat staging before checkpoint reconstruction.
        """
        if not self._query_lock.acquire(blocking=False):
            raise RuntimeError('image source has an active iterator')
        try:
            if self._closed:
                raise RuntimeError('image source is closed')
            if self._content_manifest is not None:
                return self.content_manifest_hash
            if self._stats['decode_requests_submitted'] or self._stats['batches_yielded']:
                raise ValueError('content manifest must be frozen before image outcomes')
            records=[]
            for stratum in ('target','complement'):
                row=[]
                for path in self._paths[stratum]:
                    before=_file_stamp(path)
                    digest=hashlib.sha256()
                    with open(path,'rb') as source:
                        while block:=source.read(1 << 20):
                            digest.update(block)
                    if _file_stamp(path)!=before:
                        raise ValueError('image file changed during content staging')
                    row.append((before[2],digest.hexdigest()))
                records.append(tuple(row))
            records=tuple(records)
            digest=content_hash(dict(schema='v1_image_file_content_manifest_v1',
                source_identity_hash=self.source_identity_hash,records=records),prefix='v1-image-content-')
            self._content_manifest=(self.source_identity_hash,records,digest)
            self._content_manifest_lease=self._content_manifest
            return digest
        finally:
            self._query_lock.release()

    @property
    def content_manifest_hash(self) -> str | None:
        # Tuples contain only immutable scalar values: guard replacement in O(1)
        # instead of rehashing population metadata on every inference batch.
        manifest=self._content_manifest
        if manifest is not self._content_manifest_lease:
            raise ValueError('image content manifest owner changed')
        if manifest is None:
            return None
        if manifest[0]!=self.source_identity_hash:
            raise ValueError('image content manifest source changed')
        return manifest[2]

    def frozen_content_record(self, stratum: str, index: int) -> tuple[int, str]:
        """Return one immutable (byte count, SHA256) record without decoding.

        The source listing owns the index; statistical sampling roles are not
        assigned by this metadata accessor. No population-sized copy is made.
        """
        if (stratum not in self._paths or type(index) is not int
                or not 0 <= index < len(self._paths[stratum])):
            raise ValueError('content record requires a valid exact listing index')
        if self.content_manifest_hash is None:
            raise ValueError('content records require a frozen file manifest')
        return self._content_manifest[1][int(stratum == 'complement')][index]

    def configuration(self) -> dict:
        return dict(device=str(self.device), workers=self.workers, cache_bytes=self.cache_bytes,
                    prefetch_images=self.prefetch_images, batch_size=self.batch_size,
                    max_materialized_bytes=self.max_materialized_bytes)

    def stats(self) -> dict:
        return dict(self._stats, cache_limit_bytes=self.cache_bytes,
                    prefetch_limit_images=max(1, self.prefetch_images) if self.workers else 1,
                    image_payload_scope='cache_and_requested_prefetch_excluding_decoder_metadata_and_caller_outputs',
                    source_identity_hash=self.source_identity_hash, closed=self._closed)

    def close(self) -> None:
        if not self._query_lock.acquire(blocking=False): raise RuntimeError('image source has an active iterator')
        try:
            self._cache.clear(); self._stats['cache_bytes'] = 0; self._closed = True
        finally: self._query_lock.release()

    def _validate_ids(self, stratum: str, ids: Sequence[int] | np.ndarray) -> np.ndarray:
        if stratum not in self._paths: raise ValueError('unknown image stratum')
        value = np.asarray(ids)
        if value.ndim != 1 or (value.size and value.dtype.kind not in 'iu'):
            raise ValueError('image IDs must be a one-dimensional integer sequence')
        if value.size and (np.any(value < 0) or np.any(value >= len(self._paths[stratum]))):
            raise ValueError('image ID outside the sealed listing')
        result = np.array(value, dtype=np.int64, copy=True); result.flags.writeable = False
        return result

    def _decode(self, key: tuple[str, int]) -> _CacheEntry:
        path = self._paths[key[0]][key[1]]; before = _file_stamp(path)
        if self.content_manifest_hash is None:
            with Image.open(path) as image:
                pixels = self._transform(image.convert('RGB')).contiguous()
        else:
            expected_size,expected_digest=self._content_manifest[1][int(key[0]=='complement')][key[1]]
            if before[2]!=expected_size:
                raise ValueError('image file differs from frozen content manifest')
            # Decode the same bytes that were hashed. Disk-backed scratch avoids
            # retaining an arbitrarily large compressed image in Python memory.
            with tempfile.TemporaryFile() as snapshot:
                digest=hashlib.sha256();size=0
                with open(path,'rb') as source:
                    while block:=source.read(1 << 20):
                        size+=len(block)
                        if size>expected_size:
                            raise ValueError('image file grew beyond frozen content manifest')
                        digest.update(block);snapshot.write(block)
                if size!=expected_size or digest.hexdigest()!=expected_digest:
                    raise ValueError('image file differs from frozen content manifest')
                snapshot.seek(0)
                with Image.open(snapshot) as image:
                    pixels=self._transform(image.convert('RGB')).contiguous()
        if pixels.dtype != torch.uint8 or tuple(pixels.shape) != IMAGE_SHAPE:
            raise ValueError('decoded image has unexpected dtype or shape')
        if _file_stamp(path) != before: raise ValueError('image file changed during decoding')
        return _CacheEntry(pixels, before)

    def _remember(self, key: tuple[str, int], value: _CacheEntry) -> None:
        if key in self._cache:
            self._cache.pop(key); self._stats['cache_bytes'] -= UINT8_IMAGE_BYTES
        if self.cache_bytes < UINT8_IMAGE_BYTES: return
        while self._stats['cache_bytes'] + UINT8_IMAGE_BYTES > self.cache_bytes:
            self._cache.popitem(last=False); self._stats['cache_bytes'] -= UINT8_IMAGE_BYTES
            self._stats['evictions'] += 1
        self._cache[key] = value; self._stats['cache_bytes'] += UINT8_IMAGE_BYTES
        self._stats['cache_peak_bytes'] = max(self._stats['cache_peak_bytes'], self._stats['cache_bytes'])

    def iter_indices(self, stratum: str, ids: Sequence[int] | np.ndarray,
                     *, batch_size: int | None = None) -> Iterator[ImageBatch]:
        """Snapshot IDs before iteration; yield repeats and ordering exactly as requested."""
        values = self._validate_ids(stratum, ids)
        size = self.batch_size if batch_size is None else batch_size
        if type(size) is not int or size <= 0 or size * FLOAT32_IMAGE_BYTES > self.max_materialized_bytes:
            raise ValueError('requested image batch exceeds the materialized output budget')
        return self._iterate(stratum, values, size)

    def _iterate(self, stratum: str, ids: np.ndarray, size: int) -> Iterator[ImageBatch]:
        if not self._query_lock.acquire(blocking=False): raise RuntimeError('image source has an active iterator')
        executor = None; pending = deque()
        try:
            if self._closed: raise RuntimeError('image source is closed')
            capacity = max(1, self.prefetch_images) if self.workers else 1
            if self.workers and self.prefetch_images:
                executor = ThreadPoolExecutor(max_workers=min(self.workers, capacity), thread_name_prefix='v1-image-decode')
            else: capacity = 1
            next_submit = 0
            def fill():
                nonlocal next_submit
                while next_submit < len(ids) and len(pending) < capacity:
                    key = (stratum, int(ids[next_submit])); next_submit += 1
                    if key in self._cache:
                        value = self._cache[key]; hit = True
                    else:
                        self._stats['decode_requests_submitted'] += 1; hit = False
                        value = executor.submit(self._decode, key) if executor else self._decode(key)
                    pending.append((key, value, hit))
                    self._stats['prefetch_peak_images'] = max(self._stats['prefetch_peak_images'], len(pending))
            fill()
            for start in range(0, len(ids), size):
                stop = min(start + size, len(ids))
                u8 = torch.empty((stop - start, *IMAGE_SHAPE), dtype=torch.uint8)
                self._stats['batch_uint8_peak_bytes'] = max(self._stats['batch_uint8_peak_bytes'], u8.numel())
                for offset in range(stop - start):
                    key, pending_value, hit = pending.popleft()
                    value = pending_value.result() if isinstance(pending_value, Future) else pending_value
                    if _file_stamp(self._paths[key[0]][key[1]]) != value.file_stamp:
                        raise ValueError('image file changed since decoding or caching')
                    if hit: self._stats['cache_hits'] += 1
                    self._remember(key, value); u8[offset].copy_(value.pixels)
                    del pending_value, value
                    fill()
                normalized = u8.to(device=self.device, dtype=torch.float32)
                normalized.div_(255.0).sub_(self._mean).div_(self._std)
                context_ids = tuple(int(v) for v in ids[start:stop])
                payload = dict(schema='tractable_v1_requested_image_batch_v1', source_identity_hash=self.source_identity_hash,
                    image_content_manifest_hash=self.content_manifest_hash,
                    stratum=stratum, listing_indices=context_ids,
                    ordered_context_ids=tuple(ordered_context_id(stratum, v) for v in context_ids),
                    labels=tuple(float(stratum == 'target') for _ in context_ids),
                    decoded_uint8_shape=list(u8.shape), decoded_uint8_sha256=hashlib.sha256(memoryview(u8.numpy())).hexdigest(),
                    normalized_shape=list(normalized.shape), normalized_dtype=str(normalized.dtype), normalized_device=str(normalized.device),
                    normalization_law='separate_fp32_div255_submean_divstd',
                    byte_scope='actual_decoded_uint8_bytes_and_normalization_provenance_not_normalized_tensor_digest',
                    sampling_authority=False, model_execution_certified=False, production_authority=False)
                batch_hash = content_hash(payload, prefix='tractable-v1-image-batch-')
                self._stats['batches_yielded'] += 1
                yield ImageBatch(stratum, context_ids, normalized, self.source_identity_hash, payload, batch_hash)
                del normalized, u8, payload
        finally:
            for _, value, _ in pending:
                if isinstance(value, Future): value.cancel()
            if executor is not None: executor.shutdown(wait=True, cancel_futures=True)
            pending.clear(); self._query_lock.release()

    def _validate_sealed_slice(self, stratum: str, start: int, count: int) -> None:
        if stratum not in self._paths: raise ValueError('unknown image stratum')
        if type(start) is not int or type(count) is not int or start < 0 or count < 0:
            raise ValueError('sealed image slice needs nonnegative integer start/count')
        if start + count > len(self._paths[stratum]): raise ValueError('sealed image slice exceeds population; no wrapping')

    def iter_sealed(self, stratum: str, start: int, count: int,
                    *, batch_size: int | None = None) -> Iterator[ImageBatch]:
        """Strict frozen-position slice yielding the existing exact-ID byte receipts."""
        self._validate_sealed_slice(stratum, start, count)
        return self.iter_indices(stratum, self._permutations[stratum][start:start + count], batch_size=batch_size)

    def __call__(self, stratum: str, start: int, count: int) -> torch.Tensor:
        self._validate_sealed_slice(stratum, start, count)
        if count * FLOAT32_IMAGE_BYTES > self.max_materialized_bytes:
            raise ValueError('requested images exceed materialized output budget; consume iter_indices batches')
        iterator = self.iter_sealed(stratum, start, count)
        try:
            first = next(iterator, None)
            if first is None: return torch.empty((0, *IMAGE_SHAPE), dtype=torch.float32, device=self.device)
            result = torch.empty((count, *IMAGE_SHAPE), dtype=torch.float32, device=self.device)
            offset = len(first.context_ids); result[:offset].copy_(first.images); del first
            for batch in iterator:
                stop = offset + len(batch.context_ids); result[offset:stop].copy_(batch.images); offset = stop
                del batch
            return result
        finally: iterator.close()
