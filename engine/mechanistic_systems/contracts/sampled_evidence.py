"""Capability-neutral sampled-panel and reference-evidence contracts.

These objects bind sample identity, cross-fit allocation, losses, and complete
hard-mask hashes. They do not select, accept, or commit a pruning action.

Panel axis chain
----------------
A backend that extends one logical panel look after look may attach an
incremental identity chain to ``allocation_metadata[PANEL_AXIS_CHAIN_KEY]``::

    {"parent_hash": <chain hash of the prior panel or None>,
     "hash": <chain hash of this panel>,
     "event": "extend" | "census_supersede"}

``assert_nested_extension`` accepts a chained panel in O(1) when its
``parent_hash`` equals the prior panel's chain ``hash`` and the event is
``"extend"``: the backend attests that the prior axis is a prefix of the new
one.  ``"census_supersede"`` (a stratum reached its finite census and its
logical rows were replaced by the exact census rows) is accepted only when
every dropped context belongs to a stratum listed in
``allocation_metadata["oracle_exhausted_strata"]`` and every retained context
keeps its fold. Panels without a chain compare their context and fold axes
directly.
"""

from __future__ import annotations

import binascii
import hashlib
import json
import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class PeriodicFolds(Sequence):
    """Exact i modulo fold_count assignments without a per-row tuple."""
    size: int
    fold_count: int

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(i % self.fold_count for i in range(*index.indices(self.size)))
        index = index+self.size if index < 0 else index
        if index < 0 or index >= self.size:
            raise IndexError(index)
        return index % self.fold_count


PANEL_AXIS_CHAIN_KEY = "panel_axis_chain"
PANEL_AXIS_CHAIN_EVENTS = ("extend", "census_supersede")
ORACLE_EXHAUSTED_STRATA_KEY = "oracle_exhausted_strata"
ORACLE_STRATA_KEY = "oracle_strata"


def _canonical_hash_payload(value: Any) -> Any:
    # Exact builtin types first; each reaches the same branch as
    # the chain below (ndarray.tolist() leaves are exactly these types).
    kind = type(value)
    if kind is str or kind is int or kind is bool or value is None:
        return value
    if kind is float:
        return value if math.isfinite(value) else None
    if kind is dict:
        return {
            str(key): _canonical_hash_payload(value[key])
            for key in sorted(value, key=str)
        }
    if kind is list or kind is tuple:
        return [_canonical_hash_payload(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_hash_payload(value[key])
            for key in sorted(value, key=str)
        }
    if isinstance(value, np.ndarray):
        return _canonical_hash_payload(value.tolist())
    if isinstance(value, np.generic):
        return _canonical_hash_payload(value.item())
    if isinstance(value, (list, tuple)):
        return [_canonical_hash_payload(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sampled_path_digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _canonical_hash_payload(payload),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


# A decision may request a mask digest several times. The cache keys contain
# complete ordered layer headers and packed little-endian bits, so mutation
# changes the key and equal copies share the digest. Each entry holds ~R/8
# bytes (7.6 MB for AlexNet's 61M routes).
_MASK_HASH_CACHE_ENTRIES = 4
_MASK_HASH_HEX_CHUNK_BYTES = 1 << 20
# Most recent last: (ordered layer headers, packed bits per layer, digest).
_MASK_HASH_CACHE: list[tuple[tuple[bytes, ...], tuple[bytes, ...], str]] = []
_MASK_HASH_LOCK = threading.Lock()


def _mask_hash_layers(
    masks: Mapping[str, np.ndarray],
) -> tuple[tuple[bytes, ...], tuple[bytes, ...]]:
    headers: list[bytes] = []
    packed: list[bytes] = []
    for layer_id in sorted(masks):
        mask = np.asarray(masks[layer_id], dtype=bool)
        headers.append(
            json.dumps(layer_id).encode("utf-8")
            + b",["
            + ",".join(str(int(dim)) for dim in mask.shape).encode("ascii")
        )
        packed.append(np.packbits(mask, bitorder="little").tobytes())
    return tuple(headers), tuple(packed)


def _mask_hash_from_layers(
    headers: tuple[bytes, ...], packed: tuple[bytes, ...]
) -> str:
    digest = hashlib.sha256()
    digest.update(b"[")
    for position, (header, bits) in enumerate(zip(headers, packed, strict=True)):
        if position:
            digest.update(b",")
        digest.update(b"[")
        digest.update(header)
        digest.update(b'],"')
        view = memoryview(bits)
        for start in range(0, len(view), _MASK_HASH_HEX_CHUNK_BYTES):
            # hexlify(b) == b.hex().encode("ascii"), chunk by chunk.
            digest.update(
                binascii.hexlify(view[start : start + _MASK_HASH_HEX_CHUNK_BYTES])
            )
        digest.update(b'"]')
    digest.update(b"]")
    return digest.hexdigest()


def sampled_path_mask_hash(masks: Mapping[str, np.ndarray]) -> str:
    """SHA-256 of the canonical JSON of [(layer, shape, packed-hex), ...].

    Layer IDs are sorted, and packed-mask hex text is streamed in bounded
    chunks. Cached digests are keyed by the complete layer headers and packed
    masks. Incremental mask views carry the digest of their current state.
    """

    # Committed states and candidate views update their digest from cached
    # per-layer bytes and SHA states; plain mappings are packed here.
    incremental = getattr(masks, "_v6_sampled_path_mask_hash", None)
    if incremental is not None:
        return incremental()
    headers, packed = _mask_hash_layers(masks)
    with _MASK_HASH_LOCK:
        for index in range(len(_MASK_HASH_CACHE) - 1, -1, -1):
            cached_headers, cached_packed, cached_digest = _MASK_HASH_CACHE[index]
            if cached_headers == headers and cached_packed == packed:
                _MASK_HASH_CACHE.append(_MASK_HASH_CACHE.pop(index))
                return cached_digest
    digest = _mask_hash_from_layers(headers, packed)
    with _MASK_HASH_LOCK:
        _MASK_HASH_CACHE.append((headers, packed, digest))
        del _MASK_HASH_CACHE[:-_MASK_HASH_CACHE_ENTRIES]
    return digest


def _panel_axis_chain(panel: "CrossfitPanel") -> Mapping[str, Any] | None:
    chain = panel.allocation_metadata.get(PANEL_AXIS_CHAIN_KEY)
    if not isinstance(chain, Mapping):
        return None
    if (
        not chain.get("hash")
        or chain.get("event") not in PANEL_AXIS_CHAIN_EVENTS
    ):
        raise ValueError("sampled panel axis chain is malformed")
    return chain


@dataclass(frozen=True)
class CrossfitPanel:
    context_ids: tuple[str, ...]
    fold_ids: tuple[int, ...]
    sampling_weights: tuple[float, ...] = tuple()
    labels: tuple[int, ...] = tuple()
    draw_id: str = ""
    stage_index: int = 0
    allocation_metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, *, fold_count: int) -> None:
        """Validate the panel's context, fold, weight and label axes."""

        size = len(self.context_ids)
        unique = bool(self.__dict__.get("_v6_unique_contexts_verified", False))
        if size < fold_count or (
            not unique and len(set(self.context_ids)) != size
        ):
            raise ValueError(
                "sampled panel requires unique contexts covering every fold"
            )
        folds_valid = (
            self.fold_ids.fold_count == fold_count
            if isinstance(self.fold_ids, PeriodicFolds)
            else set(self.fold_ids) == set(range(fold_count))
        )
        if len(self.fold_ids) != size or not folds_valid:
            raise ValueError(
                "sampled panel fold IDs must cover the declared cross-fit axis"
            )
        if self.sampling_weights:
            if len(self.sampling_weights) != size:
                raise ValueError("sampled panel weights must be finite and positive")
            try:
                weights = np.asarray(self.sampling_weights, dtype=np.float64)
            except (TypeError, ValueError):
                weights = None
            if weights is None or weights.shape != (size,):
                if any(
                    not math.isfinite(value) or value <= 0.0
                    for value in self.sampling_weights
                ):
                    raise ValueError(
                        "sampled panel weights must be finite and positive"
                    )
            elif not bool(np.all(np.isfinite(weights) & (weights > 0.0))):
                raise ValueError("sampled panel weights must be finite and positive")
        if self.labels and len(self.labels) != size:
            raise ValueError("sampled panel labels must follow the context axis")
        if self.stage_index < 0:
            raise ValueError("sampled panel stage index must be nonnegative")

    @cached_property
    def panel_hash(self) -> str:
        """Content hash of the complete panel, computed once per instance.

        Invariant: a panel is immutable after construction (frozen dataclass;
        ``allocation_metadata`` must not be mutated by its producer once the
        panel is handed out), so the memoised value equals the value a fresh
        computation would return.  The cache lives in the instance ``__dict__``
        and is not a dataclass field, so equality, ``asdict`` and ``replace``
        are unaffected.
        """

        return sampled_path_digest(
            {
                "context_ids": self.context_ids,
                "fold_ids": self.fold_ids,
                "sampling_weights": self.sampling_weights,
                "labels": self.labels,
                "draw_id": self.draw_id,
                "stage_index": self.stage_index,
                "allocation_metadata": dict(self.allocation_metadata),
            }
        )

    @property
    def panel_axis_chain(self) -> Mapping[str, Any] | None:
        return _panel_axis_chain(self)

    def assert_nested_extension(self, prior: "CrossfitPanel") -> None:
        """Require ``prior`` to be a retained prefix of this panel.

        Fast path: a well-formed ``panel_axis_chain`` whose ``parent_hash``
        equals the prior panel's chain ``hash``.  ``"extend"`` is accepted in
        O(1); ``"census_supersede"`` is accepted after checking that every
        dropped context sits in an oracle-exhausted stratum and every retained
        context keeps its fold.  Any other situation (no chain on either
        panel, a chain whose parent does not match, a chain with a different
        event) uses the full dictionary comparison unchanged.
        """

        if self.stage_index <= prior.stage_index:
            raise ValueError("sampled panel stages must advance monotonically")
        chain = _panel_axis_chain(self)
        prior_chain = _panel_axis_chain(prior) if chain is not None else None
        if (
            chain is not None
            and prior_chain is not None
            and chain.get("parent_hash") == prior_chain["hash"]
        ):
            if chain["event"] == "extend":
                return
            self._assert_census_supersede(prior)
            return
        self._assert_retained_prefix(prior)

    def _assert_retained_prefix(self, prior: "CrossfitPanel") -> None:
        current = dict(zip(self.context_ids, self.fold_ids, strict=True))
        for context_id, fold in zip(prior.context_ids, prior.fold_ids, strict=True):
            if current.get(context_id) != fold:
                raise ValueError(
                    "sampled panels must retain contexts and fold assignments"
                )

    def _assert_census_supersede(self, prior: "CrossfitPanel") -> None:
        exhausted = {
            str(value)
            for value in self.allocation_metadata.get(
                ORACLE_EXHAUSTED_STRATA_KEY, ()
            )
        }
        prior_strata = tuple(
            str(value) for value in prior.allocation_metadata.get(ORACLE_STRATA_KEY, ())
        )
        if len(prior_strata) != len(prior.context_ids):
            raise ValueError(
                "census supersession requires the prior panel's per-row strata"
            )
        current = dict(zip(self.context_ids, self.fold_ids, strict=True))
        for context_id, fold, stratum in zip(
            prior.context_ids, prior.fold_ids, prior_strata, strict=True
        ):
            retained_fold = current.get(context_id)
            if retained_fold is None:
                if stratum not in exhausted:
                    raise ValueError(
                        "sampled panels must retain contexts and fold assignments"
                    )
            elif retained_fold != fold:
                raise ValueError(
                    "sampled panels must retain contexts and fold assignments"
                )

    def calibration_indices(self, heldout_fold: int) -> np.ndarray:
        return np.flatnonzero(np.asarray(self.fold_ids) != int(heldout_fold))

    def extraction_indices(self, heldout_fold: int) -> np.ndarray:
        return np.flatnonzero(np.asarray(self.fold_ids) == int(heldout_fold))


@dataclass(frozen=True)
class SampledReferenceEvidence:
    """Cross-fitted loss for the committed state at one layer-group boundary."""

    query_id: str
    panel: CrossfitPanel
    losses: np.ndarray
    calibration_by_fold: tuple[Mapping[str, Any], ...]
    mask_hash: str

    def validate(self, *, fold_count: int) -> None:
        self.panel.validate(fold_count=fold_count)
        values = np.asarray(self.losses, dtype=np.float64).reshape(-1)
        if values.size != len(self.panel.context_ids) or not np.isfinite(values).all():
            raise ValueError("sampled-reference losses must cover the complete panel")
        if np.any(values < 0.0):
            raise ValueError("sampled-reference BCE losses must be nonnegative")
        if len(self.calibration_by_fold) != int(fold_count):
            raise ValueError(
                "sampled-reference evidence requires one fit per held-out fold"
            )
        if not self.query_id or len(self.mask_hash) != 64:
            raise ValueError("sampled-reference evidence identity is invalid")

    @property
    def canonical_loss(self) -> float:
        return float(np.asarray(self.losses, dtype=np.float64).reshape(-1).mean())

    @property
    def content_hash(self) -> str:
        return sampled_path_digest(
            {
                "query_id": self.query_id,
                "panel_hash": self.panel.panel_hash,
                "losses_float64": np.asarray(self.losses, dtype=np.float64).reshape(-1),
                "calibration_by_fold": self.calibration_by_fold,
                "mask_hash": self.mask_hash,
            }
        )


__all__ = [
    "CrossfitPanel",
    "ORACLE_EXHAUSTED_STRATA_KEY",
    "ORACLE_STRATA_KEY",
    "PANEL_AXIS_CHAIN_EVENTS",
    "PANEL_AXIS_CHAIN_KEY",
    "SampledReferenceEvidence",
    "sampled_path_digest",
    "sampled_path_mask_hash",
]
