"""Bounded terminal image transport without a development sampling law."""
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from types import MappingProxyType

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.measurement.contexts import ContextAxis
from .modern_v1_operator_data import _BoundedContextPanelTransport
from .tractable_v1.image_stream import BoundedImageSource
from .tractable_v1.genuine_issuer_bridge import (
    ContextTensorBatch, _bounded_source_identity, _context_axis_matches_bounded_preprocessing, _record_path,
)


@dataclass(frozen=True, slots=True)
class OperatorTerminalContextSource:
    image_source: BoundedImageSource
    context_axis: ContextAxis
    image_root: Path
    target_class_id: int
    max_batch_rows: int = 64
    source_identity_hash: str = field(init=False)
    actual_source_binding_hash: str = field(init=False)
    _content_manifest: str = field(init=False, repr=False)
    _indices: object = field(init=False, repr=False)
    _source_objects: tuple = field(init=False, repr=False)

    def __post_init__(self):
        if type(self.image_source) is not BoundedImageSource or type(self.context_axis) is not ContextAxis:
            raise TypeError('terminal transport requires exact image source and context axis')
        if type(self.max_batch_rows) is not int or self.max_batch_rows <= 0:
            raise ValueError('terminal source capacity must be positive')
        if type(self.target_class_id) is not int or self.target_class_id < 0:
            raise ValueError('terminal source target class must be a nonnegative integer')
        self.context_axis.validate()
        if self.context_axis.split not in ('val', 'validation'):
            raise ValueError('terminal source requires a declared validation split')
        if self.image_source.stats()['decode_requests_submitted']:
            raise ValueError('terminal source binding must precede decoded outcomes')
        identity, payload = _bounded_source_identity(self.image_source)
        _context_axis_matches_bounded_preprocessing(self.context_axis, payload)
        indices = {}
        all_paths = set()
        for stratum in ('target', 'complement'):
            paths = tuple(str(Path(p).resolve()) for p in self.image_source._paths[stratum])
            if len(set(paths)) != len(paths) or all_paths.intersection(paths):
                raise ValueError('terminal listing repeats image paths')
            all_paths.update(paths)
            lookup = {path:index for index,path in enumerate(paths)}
            records = tuple(r for r in self.context_axis.records
                if (r.label == self.target_class_id) == (stratum == 'target'))
            declared = tuple(_record_path(Path(self.image_root).resolve(), r.relative_path) for r in records)
            if not records or len(set(declared)) != len(declared) or set(declared) != set(paths):
                raise ValueError('terminal context axis differs from complete declared stratum listing')
            indices[stratum] = MappingProxyType({r.context_id:lookup[path] for r,path in zip(records,declared,strict=True)})
        object.__setattr__(self, '_indices', MappingProxyType(indices))
        object.__setattr__(self, 'source_identity_hash', identity)
        manifest = self.image_source.freeze_content_manifest()
        for record in self.context_axis.records:
            stratum = 'target' if record.label == self.target_class_id else 'complement'
            size, _ = self.image_source.frozen_content_record(stratum, indices[stratum][record.context_id])
            if size != record.source_size_bytes:
                raise ValueError('terminal file size differs from its registered context record')
        object.__setattr__(self, '_content_manifest', manifest)
        object.__setattr__(self, '_source_objects', (self.image_source, self.context_axis, self._indices))
        object.__setattr__(self, 'actual_source_binding_hash', content_hash(dict(
            schema='v1_operator_terminal_context_source_v1', source_identity_hash=identity,
            image_content_manifest_hash=manifest, context_axis_hash=self.context_axis.content_hash,
            target_class_id=self.target_class_id, max_batch_rows=self.max_batch_rows,
            sampling_authority=False, reveal_authority=False), prefix='v1-operator-terminal-source-'))

    @property
    def source_context_axis_hash(self): return self.context_axis.content_hash

    @property
    def source_label_map_hash(self): return self.context_axis.label_map_hash

    @property
    def preprocessing_hash(self): return self.context_axis.preprocessing.content_hash

    def assert_live(self):
        if any(a is not b for a,b in zip(self._source_objects,
                (self.image_source,self.context_axis,self._indices),strict=True)):
            raise ValueError('terminal source custody changed')
        identity, _ = _bounded_source_identity(self.image_source)
        if identity != self.source_identity_hash or self.image_source.content_manifest_hash != self._content_manifest:
            raise ValueError('terminal source content identity changed')

    def __call__(self, stratum, requested):
        self.assert_live()
        if isinstance(requested, (str,bytes)) or not isinstance(requested, Sequence):
            raise TypeError('terminal source needs an ordered context sequence')
        contexts = tuple(requested)
        if (stratum not in self._indices or not contexts or len(contexts) > self.max_batch_rows
                or len(set(contexts)) != len(contexts) or any(c not in self._indices[stratum] for c in contexts)):
            raise ValueError('terminal source request is outside its declared context/row bounds')
        indices = tuple(self._indices[stratum][c] for c in contexts)
        iterator = self.image_source.iter_indices(stratum, indices, batch_size=len(indices))
        try:
            batch = next(iterator)
            self.assert_live()
            if (batch.context_ids != indices or batch.source_identity_hash != self.source_identity_hash
                    or batch.source_batch_receipt_payload['image_content_manifest_hash'] != self._content_manifest
                    or content_hash(batch.source_batch_receipt_payload, prefix='tractable-v1-image-batch-') != batch.source_batch_receipt_hash):
                raise ValueError('terminal image receipt differs from the requested source/contexts')
            try:
                next(iterator)
            except StopIteration:
                pass
            else:
                raise ValueError('terminal source split a bounded single-batch request')
        finally:
            iterator.close()
        digest = content_hash(dict(binding=self.actual_source_binding_hash, context_ids=contexts,
            image_receipt_hash=batch.source_batch_receipt_hash, sampling_authority=False,
            reveal_authority=False), prefix='v1-operator-terminal-batch-')
        return ContextTensorBatch(context_ids=contexts, images=batch.images, labels=batch.labels,
            source_batch_receipt_hash=digest)


@dataclass(frozen=True, slots=True)
class OperatorTerminalImageDataPlane(_BoundedContextPanelTransport):
    context_source: OperatorTerminalContextSource
    batch_rows: int = 64
    source_role: str = field(default='terminal', init=False)
    binding_hash: str = field(init=False)
    _labels: object = field(init=False, repr=False)
    _source_object: object = field(init=False, repr=False)
    _lock: object = field(init=False, repr=False)
    _stats: dict = field(init=False, repr=False)

    def __post_init__(self):
        if type(self.context_source) is not OperatorTerminalContextSource:
            raise TypeError('terminal data plane requires its exact terminal source')
        if type(self.batch_rows) is not int or not 0 < self.batch_rows <= self.context_source.max_batch_rows:
            raise ValueError('terminal data plane exceeds source capacity')
        self.context_source.assert_live()
        object.__setattr__(self, '_source_object', self.context_source)
        object.__setattr__(self, '_labels', MappingProxyType({r.context_id:r.label for r in self.context_axis.records}))
        object.__setattr__(self, '_lock', Lock())
        object.__setattr__(self, '_stats', dict(panels_started=0, batches_yielded=0,
            rows_yielded=0, adapter_normalized_peak_bytes=0))
        object.__setattr__(self, 'binding_hash', content_hash(dict(
            source=self.context_source.actual_source_binding_hash, batch_rows=self.batch_rows,
            source_role='terminal', sampling_authority=False), prefix='v1-operator-terminal-plane-'))

    @property
    def context_axis(self): return self.context_source.context_axis

    @property
    def target_class_id(self): return self.context_source.target_class_id

    def assert_source(self, source):
        if source is not self.context_source or source is not self._source_object:
            raise ValueError('terminal data plane source custody changed')
        source.assert_live()
