"""Exact score-to-mask lowering for the frozen sampled-path frontier.

The historical frontier applies a retention density independently to each
declared layer and sign branch.  This module owns that small but scientifically
important contract without importing the legacy pruning package.
"""

from __future__ import annotations

import binascii
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from .sampled_path import sampled_path_mask_hash


HISTORICAL_MASK_LOWERING_SCHEMA = "mechanistic_historical_mask_lowering_v1"
HISTORICAL_DENSITY_SCHEDULE_SCHEMA = (
    "historical_family_specific_multiplicative_decay_v1"
)

# Stable sort placement: auto, numpy (CPU tensors), or a Torch device.
V6_SA_SORT_DEVICE_ENV = "IRG_V6_SA_SORT_DEVICE"


def _array(value: Any, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=dtype)
    if result.dtype == object:
        raise TypeError("route score and sign arrays must be numeric")
    return result


def historical_density_axis(
    route_group_kind: str,
    *,
    conv_decay: float = 0.9,
    conv_target_final_density: float = 0.05,
    linear_decay: float = 0.9,
    linear_target_final_density: float = 0.0001,
) -> tuple[float, ...]:
    """Return the exact family-specific axis used by the v1 frontier.

    UNIFIED DECAY: both families step at 0.9. A 0.85 linear decay would
    make the two families' schedules incommensurable step-for-step -- a
    linear step would remove 15% of surviving routes where a conv step
    removes 10%, so a "same density" comparison across families would not
    be a same-schedule comparison.

    The TARGETS remain deliberately distinct (conv 0.05, linear 1e-4): they
    encode how far each family can be pruned, which is a measured property
    rather than a schedule convention. At the unified decay the linear axis
    therefore has 87 interior cuts (56 at a 0.85 decay), because reaching
    1e-4 in 0.9 steps takes more steps than in 0.85 steps.
    """

    kind = str(route_group_kind).strip().lower()
    if kind in {"conv", "conv_like", "convolution"}:
        decay = float(conv_decay)
        target = float(conv_target_final_density)
    elif kind in {"linear", "linear_like", "matrix"}:
        decay = float(linear_decay)
        target = float(linear_target_final_density)
    else:
        raise ValueError("route_group_kind must be conv_like or linear_like")
    if not 0.0 < decay < 1.0:
        raise ValueError("historical density decay must lie in (0, 1)")
    if not 0.0 <= target < 1.0:
        raise ValueError("historical target density must lie in [0, 1)")
    bins = 1 if target == 0.0 else max(1, int(math.log(target) / math.log(decay)))
    axis = [1.0]
    axis.extend(float(np.power(decay, index)) for index in range(1, bins + 1))
    if abs(axis[-1]) > 1e-12:
        axis.append(0.0)
    return tuple(axis)


def _stable_keep_indices(
    scores: np.ndarray,
    eligible: np.ndarray,
    retained_count: int,
    *,
    precomputed_order: np.ndarray | None = None,
) -> np.ndarray:
    if retained_count <= 0 or eligible.size == 0:
        return np.empty(0, dtype=np.int64)
    if retained_count >= eligible.size:
        # Keeping every eligible index needs no order.  Callers only
        # assign ``mask[indices] = True``, so the returned SET is what matters
        # and it is exactly ``eligible`` (a dense density-1 group is all True).
        return eligible
    # Stable descending order reproduces torch.argsort(..., stable=True): ties
    # retain the original flattened route-coordinate order.
    order = (
        np.argsort(-scores[eligible], kind="stable")
        if precomputed_order is None
        else precomputed_order
    )
    return eligible[order[:retained_count]]


@dataclass(frozen=True)
class _PreparedPartitionedLayer:
    score: np.ndarray
    original_shape: tuple[int, ...]
    positive: np.ndarray
    negative: np.ndarray
    positive_order: np.ndarray | None = None
    negative_order: np.ndarray | None = None


def _prepare_partitioned_layer(
    raw_scores: Any,
    raw_labels: Any,
    *,
    layer: str,
    precompute_orders: bool,
) -> _PreparedPartitionedLayer:
    original = _array(raw_scores)
    score = original.reshape(-1)
    raw_label_array = _array(raw_labels)
    labels = raw_label_array.reshape(-1)
    if score.shape != labels.shape:
        raise ValueError(
            f"score/partition-label shape mismatch for layer {layer!r}"
        )
    if not np.isfinite(score).all():
        raise ValueError(f"score arrays must be finite for layer {layer!r}")
    if labels.dtype.kind not in {"i", "u", "b"} or not np.isin(
        labels, (0, 1)
    ).all():
        raise ValueError(
            f"partition labels must be integer zero/one for layer {layer!r}"
        )
    positive = np.flatnonzero(labels == 0).astype(np.int64, copy=False)
    negative = np.flatnonzero(labels == 1).astype(np.int64, copy=False)
    order_of = (
        _v6_stable_descending_order
        if precompute_orders
        else (lambda values: np.argsort(-values, kind="stable"))
    )
    return _PreparedPartitionedLayer(
        score=score,
        original_shape=original.shape,
        positive=positive,
        negative=negative,
        positive_order=(order_of(score[positive]) if precompute_orders else None),
        negative_order=(order_of(score[negative]) if precompute_orders else None),
    )


def _v6_sort_device() -> str:
    """The torch device for the group's stable route ordering."""

    raw = os.environ.get(V6_SA_SORT_DEVICE_ENV, "auto").strip().lower() or "auto"
    if raw == "numpy":
        return "cpu"
    if raw == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return raw


# Receipt-free counters of the v6 group-setup sorts (runtime telemetry only).
V6_SORT_TELEMETRY: dict[str, int] = {
    "device_sorts": 0,
}


def _v6_stable_descending_order(values: np.ndarray) -> np.ndarray:
    """Stable descending scores; equal scores keep their original route order."""

    import torch

    keys = -np.asarray(values)
    canonical = np.where(keys == 0, keys.dtype.type(0), keys)
    tensor = torch.from_numpy(np.ascontiguousarray(canonical)).to(_v6_sort_device())
    order = torch.argsort(tensor, stable=True).to("cpu").numpy()
    V6_SORT_TELEMETRY["device_sorts"] += 1
    return order


def _prepare_partitioned_layers(
    scores: Mapping[str, Any],
    partition_labels: Mapping[str, Any],
    *,
    precompute_orders: bool,
) -> dict[str, _PreparedPartitionedLayer]:
    if not scores:
        raise ValueError("mask lowering requires at least one scored layer")
    score_keys = tuple(str(key) for key in scores)
    label_keys = tuple(str(key) for key in partition_labels)
    if set(score_keys) != set(label_keys):
        raise ValueError("score and partition-label layers must match exactly")
    normalized_labels = {
        str(layer_id): value for layer_id, value in partition_labels.items()
    }
    return {
        str(layer_id): _prepare_partitioned_layer(
            raw_scores,
            normalized_labels[str(layer_id)],
            layer=str(layer_id),
            precompute_orders=precompute_orders,
        )
        for layer_id, raw_scores in scores.items()
    }


@dataclass(frozen=True)
class HistoricalMaskLoweringResult:
    masks: Mapping[str, np.ndarray]
    positive_route_count: int
    negative_route_count: int
    retained_positive_count: int
    retained_negative_count: int
    schema_name: str = HISTORICAL_MASK_LOWERING_SCHEMA

    @property
    def total_route_count(self) -> int:
        return int(self.positive_route_count + self.negative_route_count)

    @property
    def retained_route_count(self) -> int:
        return int(self.retained_positive_count + self.retained_negative_count)

    @property
    def mask_hash(self) -> str:
        return sampled_path_mask_hash(self.masks)


def lower_historical_partitioned_masks(
    scores: Mapping[str, Any],
    partition_labels: Mapping[str, Any],
    *,
    positive_density: float,
    negative_density: float,
    precomputed_orders: Mapping[str, _PreparedPartitionedLayer] | None = None,
) -> HistoricalMaskLoweringResult:
    """Lower route scores using exact v1 structural partition labels.

    Densities are branch-relative for retained-count rounding, while frontier
    sensitivity continues to use the complete layer route count separately.
    Label zero is the positive branch and label one is the negative branch.
    Requiring typed labels prevents attribution signs from being mistaken for
    the structural route polarity used by ``sampled_path_frontier_v1``.
    """

    positive_density = float(positive_density)
    negative_density = float(negative_density)
    if not 0.0 <= positive_density <= 1.0 or not 0.0 <= negative_density <= 1.0:
        raise ValueError("branch retention densities must lie in [0, 1]")
    prepared_layers = (
        _prepare_partitioned_layers(
            scores,
            partition_labels,
            precompute_orders=False,
        )
        if precomputed_orders is None
        else precomputed_orders
    )

    masks: dict[str, np.ndarray] = {}
    positive_count = negative_count = 0
    retained_positive = retained_negative = 0
    for layer_id, raw_scores in scores.items():
        layer = str(layer_id)
        prepared = prepared_layers[layer]
        score = prepared.score
        positive = prepared.positive
        negative = prepared.negative
        keep_positive = min(
            positive.size,
            int(math.ceil(positive.size * positive_density)),
        )
        keep_negative = min(
            negative.size,
            int(math.ceil(negative.size * negative_density)),
        )
        mask = np.zeros(score.size, dtype=bool)
        mask[
            _stable_keep_indices(
                score,
                positive,
                keep_positive,
                precomputed_order=prepared.positive_order,
            )
        ] = True
        mask[
            _stable_keep_indices(
                score,
                negative,
                keep_negative,
                precomputed_order=prepared.negative_order,
            )
        ] = True
        masks[layer] = mask.reshape(prepared.original_shape)
        positive_count += int(positive.size)
        negative_count += int(negative.size)
        retained_positive += int(keep_positive)
        retained_negative += int(keep_negative)
    return HistoricalMaskLoweringResult(
        masks=masks,
        positive_route_count=positive_count,
        negative_route_count=negative_count,
        retained_positive_count=retained_positive,
        retained_negative_count=retained_negative,
    )


def lower_historical_sign_partitioned_masks(
    scores: Mapping[str, Any],
    signs: Mapping[str, Any],
    *,
    positive_density: float,
    negative_density: float,
) -> HistoricalMaskLoweringResult:
    """Compatibility oracle for historical floating-point sign dictionaries."""

    labels: dict[str, np.ndarray] = {}
    for layer_id, value in signs.items():
        sign = _array(value)
        if not np.isfinite(sign).all():
            raise ValueError(f"sign arrays must be finite for layer {layer_id!r}")
        labels[str(layer_id)] = np.where(sign >= 0, 0, 1).astype(np.int8)
    return lower_historical_partitioned_masks(
        scores,
        labels,
        positive_density=positive_density,
        negative_density=negative_density,
    )


def lower_historical_unpartitioned_masks(
    scores: Mapping[str, Any],
    *,
    density: float,
) -> HistoricalMaskLoweringResult:
    """Lower unpartitioned scores with the same per-layer ceil contract."""

    density = float(density)
    if not 0.0 <= density <= 1.0:
        raise ValueError("retention density must lie in [0, 1]")
    if not scores:
        raise ValueError("mask lowering requires at least one scored layer")
    masks: dict[str, np.ndarray] = {}
    total = retained = 0
    for layer_id, raw_scores in scores.items():
        layer = str(layer_id)
        original = _array(raw_scores)
        score = original.reshape(-1)
        if not np.isfinite(score).all():
            raise ValueError(f"score arrays must be finite for layer {layer!r}")
        eligible = np.arange(score.size, dtype=np.int64)
        keep = min(score.size, int(math.ceil(score.size * density)))
        mask = np.zeros(score.size, dtype=bool)
        mask[_stable_keep_indices(score, eligible, keep)] = True
        masks[layer] = mask.reshape(original.shape)
        total += int(score.size)
        retained += int(keep)
    return HistoricalMaskLoweringResult(
        masks=masks,
        positive_route_count=total,
        negative_route_count=0,
        retained_positive_count=retained,
        retained_negative_count=0,
    )


def compose_committed_masks(
    candidate_masks: Mapping[str, Any],
    previous_masks: Mapping[str, Any] | None,
) -> dict[str, np.ndarray]:
    """Intersect a current group candidate with committed earlier groups."""

    merged = {
        str(layer): _array(mask, dtype=np.bool_).copy()
        for layer, mask in (previous_masks or {}).items()
    }
    for layer, raw_mask in candidate_masks.items():
        key = str(layer)
        mask = _array(raw_mask, dtype=np.bool_)
        prior = merged.get(key)
        if prior is not None and prior.shape != mask.shape:
            raise ValueError(f"candidate/prior mask shape mismatch for layer {key!r}")
        merged[key] = mask.copy() if prior is None else np.logical_and(prior, mask)
    return merged


def build_historical_candidate_lowerer(
    scores: Mapping[str, Any],
    signs: Mapping[str, Any],
    *,
    previous_group_masks: Mapping[str, Any] | None = None,
):
    """Return the callback expected by the canonical sampled-path tracer."""

    labels = {
        str(layer_id): np.where(_array(value) >= 0, 0, 1).astype(np.int8)
        for layer_id, value in signs.items()
    }
    return build_historical_partitioned_candidate_lowerer(
        scores,
        labels,
        previous_group_masks=previous_group_masks,
    )





def build_historical_partitioned_candidate_lowerer(
    scores: Mapping[str, Any],
    partition_labels: Mapping[str, Any],
    *,
    previous_group_masks: Mapping[str, Any] | None = None,
):
    """Build candidates from explicit structural partition labels."""

    return HistoricalPartitionedCandidateLowerer(
        scores,
        partition_labels,
        previous_group_masks=previous_group_masks,
    )


# ---------------------------------------------------------------------------
# Committed state, candidate views, order-slice deltas.
#
# Within one route group the committed mask is always
#     lowered(d_positive, d_negative) AND previous_group_masks
# (``build_candidate`` ignores its committed argument and intersects the
# group lowering with the FIXED earlier-group masks), so a candidate at the
# turn partition's new density ``d'`` differs from its parent exactly by
#     S = eligible[order[k_new:k_old]]  intersected with the parent's kept bits,
# a contiguous slice of the precomputed stable order.  Every byte and digest
# below is computed from that slice; the payloads and hash functions are
# those of the full-mask lowering, and only the bytes fed are produced
# incrementally.
# ---------------------------------------------------------------------------

_U32_MAX = int(np.iinfo(np.uint32).max)
_INT32_MAX = int(np.iinfo(np.int32).max)
_HEX_CHUNK_BYTES = 1 << 20
_PARTITIONS = ("positive", "negative")


def _keep_count(size: int, density: float) -> int:
    # Exactly the retained-count rule of lower_historical_partitioned_masks.
    return min(int(size), int(math.ceil(int(size) * float(density))))


@dataclass(frozen=True)
class MaskDelta:
    """One (layer, partition) order slice removed by a candidate."""

    layer: str
    partition: str
    k_old: int
    k_new: int
    removed_count: int


def _normalized_layout(
    sizes: Mapping[str, int], layout: Sequence[Sequence[Any]]
) -> tuple[tuple[str, int, int], ...] | None:
    """Validate a flat-axis layout ``((layer, start, stop), ...)`` or return None."""

    try:
        key = tuple((str(layer), int(start), int(stop)) for layer, start, stop in layout)
    except (TypeError, ValueError):
        return None
    if len(key) != len(sizes) or {layer for layer, _, _ in key} != set(sizes):
        return None
    cursor = 0
    for layer, start, stop in key:
        if start != cursor or stop - start != int(sizes[layer]):
            return None
        cursor = stop
    return key


def _layout_anchors(key: tuple[tuple[str, int, int], ...]) -> tuple[int, ...]:
    # SHA states are cached at the start BYTE of every layer.  Layer
    # boundaries need not be byte-aligned: the state before the boundary
    # byte is kept, and a resumed digest re-feeds that shared byte.
    return tuple(sorted({start // 8 for _, start, _ in key}))


def _feed_with_anchors(
    digest: Any,
    data: bytes,
    start: int,
    anchors: Sequence[int],
) -> dict[int, Any]:
    """Feed ``data[start:]``, snapshotting the state at each anchor > start."""

    view = memoryview(data)
    snapshots: dict[int, Any] = {}
    cursor = int(start)
    for anchor in anchors:
        if anchor <= cursor:
            continue
        digest.update(view[cursor:anchor])
        snapshots[int(anchor)] = digest.copy()
        cursor = int(anchor)
    digest.update(view[cursor:])
    return snapshots


class _PackedLayout:
    """Big-endian packed bytes of one mask on one flat-axis layout."""

    __slots__ = ("key", "route_count", "packed", "states", "digest")

    def __init__(
        self,
        key: tuple[tuple[str, int, int], ...],
        packed: bytes,
        states: dict[int, Any],
        digest: str,
    ) -> None:
        self.key = key
        self.route_count = int(key[-1][2]) if key else 0
        self.packed = packed
        self.states = states
        self.digest = digest


def _pack_arrays(
    arrays: Mapping[str, np.ndarray], key: tuple[tuple[str, int, int], ...]
) -> _PackedLayout:
    flat = np.empty(int(key[-1][2]), dtype=bool)
    for layer, start, stop in key:
        flat[start:stop] = np.asarray(arrays[layer], dtype=bool).reshape(-1)
    packed = np.packbits(flat, bitorder="big").tobytes()
    del flat
    digest = hashlib.sha256()
    anchors = _layout_anchors(key)
    states: dict[int, Any] = {0: digest.copy()} if 0 in anchors else {}
    states.update(_feed_with_anchors(digest, packed, 0, anchors))
    return _PackedLayout(key, packed, states, digest.hexdigest())


def _sampled_path_header(layer_id: str, shape: tuple[int, ...]) -> bytes:
    # The exact per-layer header of contracts.sampled_evidence._mask_hash_layers.
    return (
        json.dumps(layer_id).encode("utf-8")
        + b",["
        + ",".join(str(int(dim)) for dim in shape).encode("ascii")
    )


def _feed_sampled_path_layer(
    digest: Any, position: int, header: bytes, little_bits: bytes
) -> None:
    if position:
        digest.update(b",")
    digest.update(b"[")
    digest.update(header)
    digest.update(b'],"')
    view = memoryview(little_bits)
    for start in range(0, len(view), _HEX_CHUNK_BYTES):
        digest.update(binascii.hexlify(view[start : start + _HEX_CHUNK_BYTES]))
    digest.update(b'"]')


def _seeded_hard_mask(route_axis_hash: str, layout: _PackedLayout) -> Any:
    from mechanistic_systems.contracts.hard_mask_actions import HardMask

    mask = HardMask(
        route_axis_hash=str(route_axis_hash),
        route_count=int(layout.route_count),
        packed_bits=layout.packed,
    )
    # ``mask_hash`` is a cached_property: the value lives in the instance
    # ``__dict__``.  Seed it with the identical payload computed from the
    # streamed digest of the same packed bytes.
    mask.__dict__["mask_hash"] = HardMask.mask_hash_from_packed_digest(
        route_axis_hash=str(route_axis_hash),
        route_count=int(layout.route_count),
        packed_sha256=layout.digest,
    )
    mask.__dict__["_v6_packed_sha256"] = layout.digest
    return mask


class _LayerMaskMapping(Mapping):
    """Read-only ``Mapping[str, np.ndarray]`` over a declared layer order."""

    _layer_ids: tuple[str, ...]

    def __iter__(self) -> Iterator[str]:
        return iter(self._layer_ids)

    def __len__(self) -> int:
        return len(self._layer_ids)

    def __contains__(self, key: object) -> bool:
        return key in self._sizes  # type: ignore[attr-defined]


def _immutable_array(value: np.ndarray, *, dtype: np.dtype | type) -> np.ndarray:
    """Own array contents in bytes so no NumPy base can restore writeability."""

    array = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _bytes_backed(value: np.ndarray) -> bool:
    owner: Any = value
    while isinstance(owner, np.ndarray):
        owner = owner.base
    return isinstance(owner, bytes)


def _public_array(value: np.ndarray) -> np.ndarray:
    """Build a fresh public header over immutable storage, never an internal header."""

    owner: Any = value
    while isinstance(owner, np.ndarray):
        owner = owner.base
    if not isinstance(owner, bytes):
        raise TypeError("control-plane array has no immutable byte owner")
    return np.frombuffer(owner, dtype=value.dtype).reshape(value.shape)


class CommittedMaskState(_LayerMaskMapping):
    """The committed masks of one route group, with cached packed digests.

    It is a ``Mapping`` of views of bytes-backed per-layer bool arrays.
    Public NumPy views and their base chains cannot restore writeability.
    ``torch.from_numpy`` and ``torch.as_tensor`` may
    write through a read-only NumPy array; that unsafe aliasing is outside this
    contract. Torch callers must copy the public array before conversion.
    Per flat-axis layout it caches the
    big-endian packed bytes, ``hashlib`` states at every layer start byte and
    the packed SHA-256; per ``route_axis_hash`` the seeded ``HardMask``; and,
    lazily, the per-layer little-endian bytes and prefix states of
    ``sampled_path_mask_hash``.  ``kept_counts`` is the exact kind-1 identity:
    within a group the mask is a pure function of it.
    """

    def __init__(
        self,
        *,
        arrays: Mapping[str, np.ndarray],
        layer_ids: tuple[str, ...],
        group_layers: tuple[str, ...],
        densities: Mapping[str, float],
        kept_counts: Mapping[tuple[str, str], int],
        _owned_arrays: bool = False,
        _lazy: Mapping[str, tuple[np.ndarray, tuple[Any, ...]]] | None = None,
    ) -> None:
        # Accept reuses immutable state/view arrays. Direct callers receive
        # one bytes-backed copy per layer at construction.
        self._arrays = (
            dict(arrays)
            if _owned_arrays
            else {key: _immutable_array(value, dtype=bool) for key, value in arrays.items()}
        )
        if _owned_arrays:
            self._arrays = {
                key: value if _bytes_backed(value) else _immutable_array(value, dtype=bool)
                for key, value in self._arrays.items()
            }
        # An accepted layer is (base array, removed-index holders)
        # until a consumer reads it; the walk reads packed bytes, not arrays.
        self._lazy: dict[str, tuple[np.ndarray, tuple[Any, ...]]] = dict(_lazy or {})
        self._layer_ids = tuple(layer_ids)
        self._sizes = {layer: int(self._base(layer).size) for layer in layer_ids}
        self._shapes = {
            layer: tuple(int(dim) for dim in self._base(layer).shape)
            for layer in layer_ids
        }
        # The device plane (resolved on first use) and this state's
        # resident packed mask on it, per flat-axis layout.
        self._plane: Any = None
        self._device_packed: dict[tuple[tuple[str, int, int], ...], Any] = {}
        self.group_layers = tuple(group_layers)
        self._densities = {str(key): float(value) for key, value in densities.items()}
        self._kept_counts = dict(kept_counts)
        self._layouts: dict[tuple[tuple[str, int, int], ...], _PackedLayout] = {}
        self._hard_masks: dict[tuple[str, tuple[tuple[str, int, int], ...]], Any] = {}
        self._last_hard_mask: Any = None
        self._sp_little: dict[str, bytes] = {}
        self._sp_states: dict[int, Any] = {}
        self._sp_digest: str | None = None

    # -- lazily materialized layers -------------------------------------------
    def _base(self, layer: str) -> np.ndarray:
        array = self._arrays.get(layer)
        return self._lazy[layer][0] if array is None else array

    def _array(self, layer: str) -> np.ndarray:
        """The immutable bool array of one layer, built on its first read."""

        array = self._arrays.get(layer)
        if array is None:
            base, holders = self._lazy[layer]
            from .device_mask_plane import fetch_removed

            fetch_removed(holders)
            values = np.array(base, dtype=bool, copy=True, order="C")
            flat = values.reshape(-1)
            for holder in holders:
                flat[holder.host()] = False
            array = _immutable_array(values, dtype=bool)
            self._arrays[layer] = array
            self._lazy.pop(layer, None)
        return array

    def _all_arrays(self) -> dict[str, np.ndarray]:
        return {layer: self._array(layer) for layer in self._layer_ids}

    # -- Mapping ------------------------------------------------------------
    def __getitem__(self, layer: str) -> np.ndarray:
        return _public_array(self._array(str(layer)))

    @property
    def densities(self) -> dict[str, float]:
        return dict(self._densities)

    @property
    def kept_counts(self) -> dict[tuple[str, str], int]:
        return dict(self._kept_counts)

    def as_dict(self) -> dict[str, np.ndarray]:
        return {layer: self[layer] for layer in self._layer_ids}

    @property
    def kept_key(self) -> tuple[tuple[str, str, int], ...]:
        return tuple(
            sorted((layer, part, int(count)) for (layer, part), count in self._kept_counts.items())
        )

    @property
    def retained_count(self) -> int:
        return int(sum(int(np.count_nonzero(self._array(layer))) for layer in self._layer_ids))

    # -- packed layouts and HardMask -----------------------------------------
    def _layout(self, key: tuple[tuple[str, int, int], ...]) -> _PackedLayout:
        layout = self._layouts.get(key)
        if layout is None:
            layout = _pack_arrays(self._all_arrays(), key)
            self._layouts[key] = layout
        return layout

    def _device_plane(self) -> Any:
        """The walk's mask plane (CUDA when present, else CPU tensors)."""

        if self._plane is None:
            from .device_mask_plane import plane_for_walk

            self._plane = plane_for_walk()
        return self._plane

    def _device_packed_for(self, key: tuple[tuple[str, int, int], ...]) -> Any:
        """The resident packed mask on ``key`` (uploaded once per walk anchor)."""

        tensor = self._device_packed.get(key)
        if tensor is None:
            tensor = self._device_plane().upload_packed(self._layout(key).packed)
            self._device_packed[key] = tensor
        return tensor

    def _v6_hard_mask_for_layout(
        self, *, route_axis_hash: str, layout: Sequence[Sequence[Any]]
    ) -> Any:
        key = _normalized_layout(self._sizes, layout)
        if key is None:
            return None
        cache_key = (str(route_axis_hash), key)
        mask = self._hard_masks.get(cache_key)
        if mask is None:
            mask = _seeded_hard_mask(route_axis_hash, self._layout(key))
            self._hard_masks[cache_key] = mask
        self._last_hard_mask = mask
        return mask

    def cached_mask_hash(self) -> str | None:
        mask = self._last_hard_mask
        return None if mask is None else str(mask.mask_hash)

    def fork(self) -> "CommittedMaskState":
        """Share immutable arrays but give host planning private memo tables."""

        from copy import copy

        state = copy(self)
        for name in ("_arrays", "_lazy", "_sizes", "_shapes", "_densities", "_kept_counts",
                     "_layouts", "_hard_masks", "_sp_little", "_device_packed"):
            setattr(state, name, dict(getattr(self, name)))
        state._sp_states = {key: value.copy() for key, value in self._sp_states.items()}
        return state

    # -- sampled_path_mask_hash (exact, incremental) ---------------------------
    def _sp_order(self) -> tuple[str, ...]:
        return tuple(sorted(self._layer_ids))

    def _sp_little_bits(self, layer: str) -> bytes:
        bits = self._sp_little.get(layer)
        if bits is None:
            bits = np.packbits(
                np.asarray(self._array(layer), dtype=bool).reshape(-1),
                bitorder="little",
            ).tobytes()
            self._sp_little[layer] = bits
        return bits

    def _sp_state_before(self, position: int) -> Any:
        order = self._sp_order()
        if not self._sp_states:
            start = hashlib.sha256()
            start.update(b"[")
            self._sp_states[0] = start
        known = max(value for value in self._sp_states if value <= position)
        digest = self._sp_states[known].copy()
        for index in range(known, position):
            layer = order[index]
            _feed_sampled_path_layer(
                digest,
                index,
                _sampled_path_header(layer, self._shapes[layer]),
                self._sp_little_bits(layer),
            )
            self._sp_states[index + 1] = digest.copy()
        return self._sp_states[position]

    def _v6_sampled_path_mask_hash(self) -> str:
        if self._sp_digest is None:
            digest = self._sp_state_before(len(self._layer_ids)).copy()
            digest.update(b"]")
            self._sp_digest = digest.hexdigest()
        return self._sp_digest

    # -- accept ----------------------------------------------------------------
    def accept(self, view: "CandidateView") -> "CommittedMaskState":
        """The committed state after accepting ``view`` (no full-mask copy).

        A changed layer is not materialized here.  The new state holds
        it as (base array, removed-index holders) and builds it on its first
        read; the walk reads the packed layout and the device packed mask the
        view already formed.
        """

        if view.parent is not self:
            raise ValueError("candidate view belongs to another committed state")
        arrays = dict(self._arrays)
        lazy = dict(self._lazy)
        for layer in view.changed_layers:
            materialized = view._materialized.get(layer)
            if materialized is not None:
                arrays[layer] = materialized
                lazy.pop(layer, None)
                continue
            base, holders = lazy.pop(layer) if layer in lazy else (arrays.pop(layer), ())
            lazy[layer] = (base, holders + view._holders(layer))
        state = CommittedMaskState(
            arrays=arrays,
            layer_ids=self._layer_ids,
            group_layers=self.group_layers,
            densities=view._densities,
            kept_counts=view._kept_counts,
            _owned_arrays=True,
            _lazy=lazy,
        )
        state._plane = self._plane
        state._device_packed = dict(view._device_packed)
        state._layouts = dict(view._layouts)
        state._hard_masks = dict(view._hard_masks)
        state._last_hard_mask = view._last_hard_mask
        state._sp_little = {
            layer: bits
            for layer, bits in self._sp_little.items()
            if layer not in view.changed_layers
        }
        state._sp_little.update(view._sp_little)
        first = view._sp_first_changed_position()
        state._sp_states = {
            position: digest
            for position, digest in self._sp_states.items()
            if position <= first
        }
        state._sp_digest = view._sp_digest
        return state


class _HostRemoved:
    """One slice's sorted removed local indices on the host (immutable bytes)."""

    __slots__ = ("count", "_values")
    fetched = True

    def __init__(self, values: np.ndarray) -> None:
        owned = np.ascontiguousarray(values, dtype=np.int64).reshape(-1)
        self._values = np.frombuffer(owned.tobytes(order="C"), dtype=np.int64)
        self.count = int(owned.size)

    def host(self) -> np.ndarray:
        return self._values


@dataclass(frozen=True, init=False, eq=False)
class _LayerSlice:
    layer: str
    partition: str
    k_old: int
    k_new: int
    # Sorted unique local flat indices: a host holder, or a device
    # holder fetched once.  Copies of a slice share the holder.
    _removed: Any

    def __init__(
        self, layer: str, partition: str, k_old: int, k_new: int,
        removed_local: np.ndarray | None = None, *, removed: Any = None,
    ) -> None:
        if (removed_local is None) == (removed is None):
            raise ValueError("a layer slice needs exactly one removed-index source")
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "partition", partition)
        object.__setattr__(self, "k_old", k_old)
        object.__setattr__(self, "k_new", k_new)
        object.__setattr__(
            self, "_removed", _HostRemoved(removed_local) if removed is None else removed
        )

    @property
    def removed_count(self) -> int:
        return int(self._removed.count)

    @property
    def removed_local(self) -> np.ndarray:
        return _public_array(self._removed.host())

    def _public_copy(self) -> "_LayerSlice":
        result = object.__new__(_LayerSlice)
        for field in ("layer", "partition", "k_old", "k_new", "_removed"):
            object.__setattr__(result, field, getattr(self, field))
        return result


class CandidateView(_LayerMaskMapping):
    """One candidate as its parent plus order-slice deltas.

    ``Mapping`` access materializes a changed layer lazily (parent copy with
    the slice cleared); all public arrays have immutable bytes-backed
    storage. ``torch.from_numpy`` and ``torch.as_tensor`` aliasing is outside
    this contract; Torch
    callers must copy the public array first. The
    seed-bearing digests (``HardMask.mask_hash`` and the removed-index hash)
    are computed once per layout from the parent's cached bytes and SHA
    states, and cached here.

    When the lowerer formed the slices on a device plane, the removed
    indices and the packed candidate live on the device.  One transfer per
    layout brings back the bytes the digests hash; removed counts are exact
    integers from the slices and never need the device.
    """

    def __init__(
        self,
        *,
        parent: CommittedMaskState,
        partition_id: str,
        densities: Mapping[str, float],
        kept_counts: Mapping[tuple[str, str], int],
        slices: tuple[_LayerSlice, ...],
    ) -> None:
        self.parent = parent
        self.partition_id = str(partition_id)
        self._densities = {str(key): float(value) for key, value in densities.items()}
        self._kept_counts = dict(kept_counts)
        self._slices = tuple(slices)
        self._layer_ids = parent._layer_ids
        self._sizes = parent._sizes
        self._shapes = parent._shapes
        self._removed_counts: dict[str, int] = {}
        for item in self._slices:
            if item.removed_count:
                self._removed_counts[item.layer] = (
                    self._removed_counts.get(item.layer, 0) + item.removed_count
                )
        self._removed_cache: dict[str, np.ndarray] | None = None
        self.changed_layers = tuple(
            layer for layer in self._layer_ids if layer in self._removed_counts
        )
        self.removed_count = int(sum(self._removed_counts.values()))
        self._materialized: dict[str, np.ndarray] = {}
        self._layouts: dict[tuple[tuple[str, int, int], ...], _PackedLayout] = {}
        self._hard_masks: dict[tuple[str, tuple[tuple[str, int, int], ...]], Any] = {}
        self._removed_sets: dict[tuple[tuple[str, int, int], ...], Any] = {}
        # Per layout, the global removed indices the device encoded.
        self._removed_encoded: dict[tuple[tuple[str, int, int], ...], np.ndarray] = {}
        self._device_packed: dict[tuple[tuple[str, int, int], ...], Any] = {}
        self._last_hard_mask: Any = None
        self._sp_little: dict[str, bytes] = {}
        self._sp_digest: str | None = None

    def fork(self, *, parent: CommittedMaskState | None = None) -> "CandidateView":
        """A branch may fill only its own materialization and digest memos."""

        from copy import copy

        view = copy(self)
        view.parent = self.parent.fork() if parent is None else parent
        for name in ("_sizes", "_shapes", "_densities", "_kept_counts",
                     "_removed_counts", "_materialized", "_layouts",
                     "_hard_masks", "_removed_sets", "_removed_encoded",
                     "_sp_little", "_device_packed"):
            setattr(view, name, dict(getattr(self, name)))
        if self._removed_cache is not None:
            view._removed_cache = dict(self._removed_cache)
        return view

    # -- removed indices (fetched once, on first host read) -------------------
    def _holders(self, layer: str) -> tuple[Any, ...]:
        return tuple(
            item._removed for item in self._slices
            if item.layer == layer and item.removed_count
        )

    @property
    def _removed_by_layer(self) -> dict[str, np.ndarray]:
        if self._removed_cache is None:
            from .device_mask_plane import fetch_removed

            fetch_removed(item._removed for item in self._slices)
            removed: dict[str, np.ndarray] = {}
            for item in self._slices:
                if not item.removed_count:
                    continue
                values = item._removed.host()
                previous = removed.get(item.layer)
                removed[item.layer] = (
                    values
                    if previous is None
                    else _immutable_array(np.union1d(previous, values), dtype=np.int64)
                )
            self._removed_cache = removed
        return self._removed_cache

    def removed_counts_by_layer(self) -> dict[str, int]:
        """Exact removed counts per changed layer (no device read)."""

        return dict(self._removed_counts)

    # -- Mapping ------------------------------------------------------------
    @property
    def densities(self) -> dict[str, float]:
        return dict(self._densities)

    @property
    def kept_counts(self) -> dict[tuple[str, str], int]:
        return dict(self._kept_counts)

    @property
    def slices(self) -> tuple[_LayerSlice, ...]:
        return tuple(item._public_copy() for item in self._slices)

    def __getitem__(self, layer: str) -> np.ndarray:
        key = str(layer)
        if key not in self._removed_counts:
            return self.parent[key]
        array = self._materialized.get(key)
        if array is None:
            removed = self._removed_by_layer[key]
            array = np.array(self.parent._array(key), dtype=bool, copy=True, order="C")
            array.reshape(-1)[removed] = False
            array = _immutable_array(array, dtype=bool)
            self._materialized[key] = array
        return _public_array(array)

    def masks(self) -> dict[str, np.ndarray]:
        """Materialize every layer as a complete mask."""

        return {layer: self[layer] for layer in self._layer_ids}

    def removed_local_by_layer(self) -> dict[str, np.ndarray]:
        return {layer: _public_array(values) for layer, values in self._removed_by_layer.items()}

    def execution_order_slices(self) -> tuple:
        """Immutable orders and int32 device locators for native execution.

        Pooled callers retain exact route-index deltas. Their floors and
        pooled count ladder remain owned by PooledCandidateLowerer.
        """
        orders = getattr(self, "_execution_orders", {})
        return tuple((item.layer, item.partition, orders[(item.layer, item.partition)],
                      item.k_new, item.k_old) for item in self._slices
                     if (item.layer, item.partition) in orders)

    def mask_deltas(self) -> tuple[MaskDelta, ...]:
        return tuple(
            MaskDelta(
                layer=item.layer,
                partition=item.partition,
                k_old=int(item.k_old),
                k_new=int(item.k_new),
                removed_count=int(item.removed_count),
            )
            for item in self._slices
        )

    @property
    def kept_key(self) -> tuple[tuple[str, str, int], ...]:
        return tuple(
            sorted((layer, part, int(count)) for (layer, part), count in self._kept_counts.items())
        )

    # -- packed layouts, HardMask and the removed-index set --------------------
    def _layout(self, key: tuple[tuple[str, int, int], ...]) -> _PackedLayout:
        layout = self._layouts.get(key)
        if layout is None:
            parent = self.parent._layout(key)
            layout = self._device_layout(key, parent) if self.changed_layers else parent
            self._layouts[key] = layout
        return layout

    def _device_layout(
        self, key: tuple[tuple[str, int, int], ...], parent: _PackedLayout
    ) -> _PackedLayout:
        """The candidate's packed mask formed on the device plane.

        The parent's packed mask is resident on the plane; the candidate is a
        copy with its ascending removed positions cleared.  One transfer
        returns the changed byte span, the encoded removed set, the changed
        layers' little-endian bits (the ``sampled_path_mask_hash`` payload)
        and the fault flag.  SHA-256 resumes from the parent's cached state at
        the first changed layer's start byte, which yields the host path's
        digest and states (the bytes before it are the parent's).
        """

        plane = self.parent._device_plane()
        bounds = {layer: (int(start), int(stop)) for layer, start, stop in key}
        changed = [layer for layer, _, _ in key if layer in self._removed_counts]
        holders = []
        for layer in changed:
            layer_holders = self._holders(layer)
            if len(layer_holders) != 1:
                raise ValueError("a device candidate view changes one slice per layer")
            holders.append(layer_holders[0])
        route_count = int(key[-1][2])
        positions = plane.positions(
            [
                (
                    holder.tensor() if hasattr(holder, "tensor")
                    else plane.upload_indices(holder.host()),
                    bounds[layer][0],
                )
                for layer, holder in zip(changed, holders)
            ],
            wide=route_count > _INT32_MAX,
        )
        low = bounds[changed[0]][0] >> 3
        high = (bounds[changed[-1]][1] + 7) >> 3
        candidate, fault = plane.clear(self.parent._device_packed_for(key), positions, low, high)
        little_layers = [layer for layer in changed if layer not in self._sp_little]
        little = [plane.little_bits(candidate, *bounds[layer]) for layer in little_layers]
        span, flag, encoded, *little_host = plane.fetch(
            candidate[low:high], fault, positions, *little
        )
        if int(flag[0]):
            raise ValueError("candidate view clears a route its parent does not retain")
        cursor = 0
        for layer, holder in zip(changed, holders):
            count = int(holder.count)
            if not holder.fetched:
                holder.set_host(encoded[cursor : cursor + count].astype(np.int64) - bounds[layer][0])
            cursor += count
        if cursor != int(encoded.size):
            raise ValueError("device candidate changed its removed count")
        self._removed_encoded[key] = encoded
        for layer, bits in zip(little_layers, little_host, strict=True):
            self._sp_little[layer] = bits.tobytes()
        view = memoryview(parent.packed)
        packed = b"".join((view[:low], span, view[high:]))
        if len(packed) != len(parent.packed):
            raise ValueError("device candidate changed the packed axis length")
        anchors = _layout_anchors(key)
        digest = parent.states[low].copy()
        states = {anchor: state for anchor, state in parent.states.items() if anchor <= low}
        states.update(_feed_with_anchors(digest, packed, low, anchors))
        self._device_packed[key] = candidate
        return _PackedLayout(key, packed, states, digest.hexdigest())

    def _v6_hard_mask_for_layout(
        self, *, route_axis_hash: str, layout: Sequence[Sequence[Any]]
    ) -> Any:
        key = _normalized_layout(self._sizes, layout)
        if key is None:
            return None
        cache_key = (str(route_axis_hash), key)
        mask = self._hard_masks.get(cache_key)
        if mask is None:
            mask = _seeded_hard_mask(route_axis_hash, self._layout(key))
            self._hard_masks[cache_key] = mask
        self._last_hard_mask = mask
        return mask

    def cached_mask_hash(self) -> str | None:
        mask = self._last_hard_mask
        return None if mask is None else str(mask.mask_hash)

    def _removed_set(self, key: tuple[tuple[str, int, int], ...]) -> Any:
        cached = self._removed_sets.get(key)
        if cached is not None:
            return cached
        from mechanistic_systems.contracts.hard_mask_actions import (
            RemovedRouteIndexSet,
        )

        route_count = int(key[-1][2])
        code = "u32" if route_count <= _U32_MAX else "u64"
        dtype = np.dtype("<u4" if code == "u32" else "<u8")
        flat = self._removed_encoded.get(key)
        if flat is None:
            parts = [
                start + self._removed_by_layer[layer].astype(np.int64)
                for layer, start, _ in key
                if layer in self._removed_by_layer
            ]
            flat = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
        encoded = np.ascontiguousarray(flat).astype(dtype, copy=False).tobytes(order="C")
        removed = RemovedRouteIndexSet(
            route_count=route_count,
            count=int(flat.size),
            dtype_code=code,  # type: ignore[arg-type]
            encoded=encoded,
        )
        removed.__dict__["index_hash"] = RemovedRouteIndexSet.index_hash_from_encoded_digest(
            route_count=route_count,
            count=int(flat.size),
            dtype_code=code,  # type: ignore[arg-type]
            encoded_sha256=hashlib.sha256(encoded).hexdigest(),
        )
        self._removed_sets[key] = removed
        return removed

    def _v6_removed_index_set_for(self, candidate_mask: Any, parent_mask: Any) -> Any:
        """The removed set on the layout that built BOTH hard masks, or None."""

        for cache_key, mask in self._hard_masks.items():
            if mask is candidate_mask and self.parent._hard_masks.get(cache_key) is parent_mask:
                return self._removed_set(cache_key[1])
        return None

    # -- sampled_path_mask_hash (exact, incremental) ---------------------------
    def _sp_first_changed_position(self) -> int:
        order = self.parent._sp_order()
        changed = set(self.changed_layers)
        for index, layer in enumerate(order):
            if layer in changed:
                return index
        return len(order)

    def _sp_little_bits(self, layer: str) -> bytes:
        if layer not in self._removed_counts:
            return self.parent._sp_little_bits(layer)
        bits = self._sp_little.get(layer)
        if bits is None:
            # The device layout's one transfer carries these bytes. Form it on
            # a layout the parent already packed (the walk's route axis), or
            # else on the declared layer order.
            key = next(iter(self.parent._device_packed or self.parent._layouts), None)
            self._layout(key if key is not None else self._declared_layout())
            bits = self._sp_little[layer]
        return bits

    def _declared_layout(self) -> tuple[tuple[str, int, int], ...]:
        cursor, key = 0, []
        for layer in self._layer_ids:
            key.append((layer, cursor, cursor + int(self._sizes[layer])))
            cursor += int(self._sizes[layer])
        return tuple(key)

    def _v6_sampled_path_mask_hash(self) -> str:
        if self._sp_digest is None:
            order = self.parent._sp_order()
            first = self._sp_first_changed_position()
            digest = self.parent._sp_state_before(first).copy()
            for index in range(first, len(order)):
                layer = order[index]
                _feed_sampled_path_layer(
                    digest,
                    index,
                    _sampled_path_header(layer, self._shapes[layer]),
                    self._sp_little_bits(layer),
                )
            digest.update(b"]")
            self._sp_digest = digest.hexdigest()
        return self._sp_digest


class HistoricalPartitionedCandidateLowerer:
    """Group mask lowering with device candidate views."""

    def __init__(
        self,
        scores: Mapping[str, Any],
        partition_labels: Mapping[str, Any],
        *,
        previous_group_masks: Mapping[str, Any] | None = None,
    ) -> None:
        self.scores = scores
        self.partition_labels = partition_labels
        self.previous_group_masks = previous_group_masks
        self.prepared = _prepare_partitioned_layers(
            scores,
            partition_labels,
            precompute_orders=True,
        )
        self.group_layers = tuple(str(layer) for layer in scores)
        # When the earlier groups' committed masks keep every route of this
        # group's own layers (each layer is walked once, deep-first), the
        # parent's kept set of a partition is exactly order[:k_old], so a
        # slice needs no retained-bit gather.  Computed once per group.
        self._prior_dense = all(
            previous_group_masks is None
            or layer not in previous_group_masks
            or bool(np.all(np.asarray(previous_group_masks[layer], dtype=bool)))
            for layer in self.group_layers
        )
        # The route orders resident on the walk's device plane.
        from .device_mask_plane import OrderCache

        self._device_orders = OrderCache()
        self._order_values: dict[tuple[str, str], np.ndarray] = {}

    def _execution_order(self, layer: str, partition: str) -> np.ndarray:
        """``eligible[order]`` as immutable int32 local indices (built once)."""

        key = (layer, partition)
        values = self._order_values.get(key)
        if values is None:
            eligible, order = self._eligible(layer, partition)
            values = self._order_values[key] = _immutable_array(eligible[order], dtype=np.int32)
        return values

    def build_candidate(
        self,
        _committed_masks: Mapping[str, np.ndarray],
        *,
        partition_id: str,
        positive_density: float,
        negative_density: float,
    ) -> dict[str, np.ndarray]:
        """Materialize the candidate as complete masks."""

        if partition_id not in {"positive", "negative"}:
            raise ValueError("partition_id must be positive or negative")
        lowered = lower_historical_partitioned_masks(
            self.scores,
            self.partition_labels,
            positive_density=positive_density,
            negative_density=negative_density,
            precomputed_orders=self.prepared,
        )
        return compose_committed_masks(lowered.masks, self.previous_group_masks)

    __call__ = build_candidate

    def dense_group(self) -> HistoricalMaskLoweringResult:
        """The density-1 group lowering from the prepared layers (no sort)."""

        return lower_historical_partitioned_masks(
            self.scores,
            self.partition_labels,
            positive_density=1.0,
            negative_density=1.0,
            precomputed_orders=self.prepared,
        )

    def _eligible(self, layer: str, partition: str) -> tuple[np.ndarray, np.ndarray]:
        prepared = self.prepared[layer]
        if partition == "positive":
            return prepared.positive, prepared.positive_order  # type: ignore[return-value]
        return prepared.negative, prepared.negative_order  # type: ignore[return-value]

    def initial_state(
        self,
        committed_masks: Mapping[str, Any],
        *,
        positive_density: float = 1.0,
        negative_density: float = 1.0,
    ) -> CommittedMaskState:
        """Construct the committed state at the supplied densities."""

        arrays = {
            str(key): _immutable_array(value, dtype=bool)
            for key, value in committed_masks.items()
        }
        densities = {"positive": float(positive_density), "negative": float(negative_density)}
        kept = {
            (layer, partition): _keep_count(
                self._eligible(layer, partition)[0].size, densities[partition]
            )
            for layer in self.group_layers
            for partition in _PARTITIONS
        }
        return CommittedMaskState(
            arrays=arrays,
            layer_ids=tuple(arrays),
            group_layers=self.group_layers,
            densities=densities,
            kept_counts=kept,
            _owned_arrays=True,
        )

    def build_candidate_view(
        self,
        state: CommittedMaskState,
        *,
        partition_id: str,
        positive_density: float,
        negative_density: float,
    ) -> CandidateView:
        if partition_id not in {"positive", "negative"}:
            raise ValueError("partition_id must be positive or negative")
        if state.group_layers != self.group_layers:
            raise ValueError("committed state belongs to another route group")
        densities = {"positive": float(positive_density), "negative": float(negative_density)}
        kept: dict[tuple[str, str], int] = {}
        slices: list[_LayerSlice] = []
        from .device_mask_plane import DeviceRemoved

        plane = state._device_plane()
        for layer in self.group_layers:
            for partition in _PARTITIONS:
                eligible, _order = self._eligible(layer, partition)
                k_old = int(state._kept_counts[(layer, partition)])
                k_new = _keep_count(eligible.size, densities[partition])
                kept[(layer, partition)] = k_new
                if k_new > k_old:
                    raise ValueError(
                        "a candidate view may only lower a partition's kept count"
                    )
                if k_new == k_old:
                    continue
                if partition != partition_id:
                    raise ValueError(
                        "a candidate view may only change the turn partition"
                    )
                if self._prior_dense:
                    # The parent keeps exactly order[:k_old], so the slice is
                    # order[k_new:k_old] of the resident order, sorted on the
                    # device; the host gathers and sorts nothing.
                    resident = self._device_orders.get(
                        plane, (layer, partition),
                        lambda plane, layer=layer, partition=partition: plane.gathered_order(
                            *self._eligible(layer, partition)
                        ),
                    )
                    removed, count = plane.sorted_slice(resident, k_new, k_old), k_old - k_new
                else:
                    # Earlier groups may already have removed part of the slice.
                    chosen = self._execution_order(layer, partition)[k_new:k_old]
                    chosen = chosen[state._array(layer).reshape(-1)[chosen]]
                    count = int(chosen.size)
                    removed = plane.sorted_slice(plane.upload_indices(chosen), 0, count)
                slices.append(
                    _LayerSlice(
                        layer=layer,
                        partition=partition,
                        k_old=k_old,
                        k_new=k_new,
                        removed=DeviceRemoved(plane, removed, count),
                    )
                )
        view = CandidateView(
            parent=state,
            partition_id=partition_id,
            densities=densities,
            kept_counts=kept,
            slices=tuple(slices),
        )
        if os.environ.get("IRG_RESIDENT_PARENT_MASK", "0") == "1" and self._prior_dense:
            if not hasattr(self, "_execution_orders"):
                self._execution_orders = {
                    (layer, partition): self._execution_order(layer, partition)
                    for layer in self.group_layers for partition in _PARTITIONS
                }
            view._execution_orders = self._execution_orders
        return view


__all__ = [
    "CandidateView",
    "CommittedMaskState",
    "HISTORICAL_DENSITY_SCHEDULE_SCHEMA",
    "HISTORICAL_MASK_LOWERING_SCHEMA",
    "HistoricalMaskLoweringResult",
    "HistoricalPartitionedCandidateLowerer",
    "MaskDelta",
    "V6_SA_ENV",
    "build_historical_candidate_lowerer",
    "build_historical_partitioned_candidate_lowerer",
    "compose_committed_masks",
    "historical_density_axis",
    "lower_historical_partitioned_masks",
    "lower_historical_sign_partitioned_masks",
    "lower_historical_unpartitioned_masks",
        ]
