"""Exact nested bottom-k score-order selection for modern-v1.

This module is intentionally standalone.  It accelerates the public
``deterministic_bottom_k_action`` composition rule without changing that rule:
scores are ordered by ``(score, physical_route_index)``, ties at the boundary
are admitted by smallest physical route id, and each action exposed to mask
code remains a physical-sorted set.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
import os
import time
from types import MappingProxyType
from typing import Any, Final, Literal

import numpy as np

from mechanistic_systems.circuits.irg.modern_v1_external_state import (
    V1ExternalArrayPolicy,
    V1ExternalStateDescriptor,
    V1ExternalStateView,
    open_external_state,
    publish_external_state,
)
from mechanistic_systems.circuits.irg.modern_v1_proposal_statistics import (
    EXACT_QUOTA_PARTITION_MATROID_ACTION_FAMILY,
    GLOBAL_FIXED_CARDINALITY_ACTION_FAMILY,
    V1CorrectedAttributionActionFamily,
)
from mechanistic_systems.contracts.content_address import content_hash

try:  # pragma: no cover - exercised by CUDA-capable environments.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


NESTED_BOTTOM_K_REQUEST_SCHEMA: Final = (
    "partition_free_v1_exact_nested_bottom_k_score_order_request_v1"
)
NESTED_BOTTOM_K_BOUNDARY_WITNESS_SCHEMA: Final = (
    "partition_free_v1_exact_nested_bottom_k_boundary_witness_v1"
)
NESTED_BOTTOM_K_RUNG_SCHEMA: Final = (
    "partition_free_v1_exact_nested_bottom_k_action_rung_v1"
)
NESTED_BOTTOM_K_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_exact_nested_bottom_k_score_order_receipt_v1"
)
NESTED_BOTTOM_K_RESULT_SCHEMA: Final = (
    "partition_free_v1_exact_nested_bottom_k_score_order_result_v1"
)
NESTED_BOTTOM_K_PREFIX_EXTERNAL_STATE_SCHEMA: Final = (
    "partition_free_v1_exact_nested_bottom_k_ordered_prefix_external_state_v1"
)
NESTED_BOTTOM_K_PREFIX_EXTERNAL_DESCRIPTOR_SCHEMA: Final = (
    "partition_free_v1_exact_nested_bottom_k_ordered_prefix_external_descriptor_v1"
)
NESTED_BOTTOM_K_PREFIX_EXTERNAL_NAMESPACE: Final = (
    "modern_v1_exact_nested_bottom_k_ordered_prefix"
)
NESTED_BOTTOM_K_ORDERED_PREFIX_LOGICAL_NAME: Final = "ordered_kmax_prefix"
NESTED_BOTTOM_K_SELECTOR_ID: Final = (
    "exact_nested_bottom_k_min_score_then_physical_prefix_v1"
)
NESTED_BOTTOM_K_TIE_RULE: Final = "minimum_score_then_ascending_physical_route_id_v1"
NESTED_BOTTOM_K_ACTION_SET_RULE: Final = (
    "score_physical_ordered_prefix_materialized_as_physical_sorted_action_sets_v1"
)
NESTED_BOTTOM_K_VALIDATION_RULE: Final = (
    "single_synchronous_confidence_family_live_count_deep_validation_v1"
)
NESTED_BOTTOM_K_CUDA_PEAK_TARGET_BYTES: Final = 8 * 1024**3

_LE_F64: Final = np.dtype("<f8")
_LE_U64: Final = np.dtype("<u8")
_NEGATIVE_ZERO_U64: Final = np.uint64(1 << 63)
_VALIDATION_CHUNK_ELEMENTS: Final = 8 * 1024 * 1024
_REQUEST_PREFIX: Final = "partition-free-v1-exact-nested-bottom-k-request-"
_ORDER_PREFIX: Final = "partition-free-v1-exact-nested-bottom-k-ordered-prefix-"
_ACTION_PREFIX: Final = "partition-free-v1-exact-nested-bottom-k-action-axis-"
_WITNESS_PREFIX: Final = "partition-free-v1-exact-nested-bottom-k-witness-"
_RECEIPT_PREFIX: Final = "partition-free-v1-exact-nested-bottom-k-receipt-"
_RESULT_PREFIX: Final = "partition-free-v1-exact-nested-bottom-k-result-"
_RECEIPT_DIAGNOSTIC_FIELDS: Final = frozenset(
    {"timing_seconds", "cuda_peak_memory_bytes", "cuda_peak_under_target"}
)
_PREFIX_EXTERNAL_DESCRIPTOR_PREFIX: Final = (
    "partition-free-v1-exact-nested-bottom-k-prefix-external-descriptor-"
)

SelectorBackend = Literal["auto", "numpy", "torch"]


class V1NestedBottomKTorchUnavailable(RuntimeError):
    """Typed refusal for exact NumPy fallback eligibility."""


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be one nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be one nonnegative integer")
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result <= 0:
        raise ValueError(f"{name} must be one positive integer")
    return result


def _readonly_le_f64_vector(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, np.ndarray) and value.dtype == _LE_F64:
        candidate = value.reshape(-1) if value.flags.c_contiguous else None
        array = (
            candidate
            if candidate is not None and _readonly_aliasable(candidate)
            else None
        )
    else:
        array = None
    copied = array is None
    if copied:
        try:
            array = np.array(value, dtype=_LE_F64, order="C", copy=True).reshape(-1)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must be one finite float64 vector") from error
    assert array is not None
    negative_zero_seen = False
    for start in range(0, int(array.size), _VALIDATION_CHUNK_ELEMENTS):
        chunk = array[start : start + _VALIDATION_CHUNK_ELEMENTS]
        if not np.isfinite(chunk).all():
            raise ValueError(f"{name} must be one finite float64 vector")
        bits = chunk.view(np.uint64)
        if bool(np.any(bits == _NEGATIVE_ZERO_U64)):
            negative_zero_seen = True
            if copied:
                bits[bits == _NEGATIVE_ZERO_U64] = np.uint64(0)
    if negative_zero_seen and not copied:
        array = np.array(array, dtype=_LE_F64, order="C", copy=True)
        bits = array.view(np.uint64)
        bits[bits == _NEGATIVE_ZERO_U64] = np.uint64(0)
    result = array.view()
    result.setflags(write=False)
    return result


def _readonly_le_u64_vector(
    value: Any,
    *,
    name: str,
    size: int | None = None,
) -> np.ndarray:
    if isinstance(value, np.ndarray) and value.dtype == _LE_U64:
        candidate = value.reshape(-1) if value.flags.c_contiguous else None
        array = (
            candidate
            if candidate is not None and _readonly_aliasable(candidate)
            else None
        )
    else:
        array = None
    if array is None:
        try:
            array = np.array(value, dtype=_LE_U64, order="C", copy=True).reshape(-1)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must be one uint64 vector") from error
    assert array is not None
    if size is not None and int(array.size) != int(size):
        raise ValueError(f"{name} must contain exactly {int(size)} values")
    result = array.view()
    result.setflags(write=False)
    return result


def _count_tuple(value: Sequence[int], *, live_count: int) -> tuple[int, ...]:
    counts = tuple(_positive_int(item, name="new removal count") for item in value)
    if not counts:
        raise ValueError("new removal counts must be nonempty")
    if len(set(counts)) != len(counts):
        raise ValueError("new removal counts must be unique")
    if max(counts) > int(live_count):
        raise ValueError("new removal count exceeds live physical routes")
    return counts


def _backend(value: object) -> SelectorBackend:
    if value not in ("auto", "numpy", "torch"):
        raise ValueError("selector backend must be 'auto', 'numpy', or 'torch'")
    return value  # type: ignore[return-value]


def _mapping_proxy(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(payload))


def _exact_mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def _json_exact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_exact(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_exact(item) for item in value]
    if isinstance(value, list):
        return [_json_exact(item) for item in value]
    return value


def _readonly_aliasable(array: np.ndarray) -> bool:
    """Return true only when an ndarray view chain is provably read-only."""

    current: object | None = array
    seen: set[int] = set()
    while isinstance(current, np.ndarray):
        if id(current) in seen:
            return False
        seen.add(id(current))
        if current.flags.writeable:
            return False
        current = current.base
    if current is None:
        return True
    if isinstance(current, bytes):
        return True
    if isinstance(current, memoryview):
        return bool(current.readonly)
    return bool(getattr(current, "readonly", False))


def _array_hash(array: np.ndarray, *, prefix: str) -> str:
    return content_hash(array, prefix=prefix)


def count_ladder_from_live_fractions(
    live_count: int,
    fractions: Sequence[float],
    *,
    maximum_action_size: int | None = None,
) -> tuple[int, ...]:
    """Map live-axis fractions to counts while preserving request order."""

    live = _positive_int(live_count, name="live count")
    maximum = live if maximum_action_size is None else int(maximum_action_size)
    maximum = _positive_int(maximum, name="maximum action size")
    if maximum > live:
        raise ValueError("maximum action size exceeds live count")
    counts: list[int] = []
    seen: set[int] = set()
    for value in fractions:
        fraction = float(value)
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("fractions must lie in (0, 1]")
        count = min(maximum, max(1, int(math.floor(float(live) * fraction))))
        if count not in seen:
            counts.append(count)
            seen.add(count)
    if not counts:
        raise ValueError("fractions produced no counts")
    return tuple(counts)


@dataclass(frozen=True, slots=True)
class V1NestedBottomKScoreOrderRequest:
    """Typed request for one nested exact bottom-k score-order batch."""

    scores: np.ndarray = field(repr=False, compare=False)
    live_physical_indices: np.ndarray = field(repr=False, compare=False)
    new_removal_counts: tuple[int, ...]
    confidence_box_hash: str
    route_axis_hash: str
    ordered_route_axis_hash: str | None = None
    prevalidated_action_family_hashes: tuple[str, ...] = ()
    action_family_registry_hash: str | None = None
    cumulative_action_sizes: tuple[int, ...] = ()
    action_families: tuple[V1CorrectedAttributionActionFamily, ...] = field(
        default_factory=tuple,
        repr=False,
        compare=False,
    )
    already_removed_count: int = 0
    selector_backend: SelectorBackend = "auto"
    torch_device: str = "cuda"
    schema: str = NESTED_BOTTOM_K_REQUEST_SCHEMA
    _request_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )
    _score_array_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )
    _live_physical_indices_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )
    _family_hashes: tuple[str, ...] = field(
        init=False,
        default=(),
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.schema != NESTED_BOTTOM_K_REQUEST_SCHEMA:
            raise ValueError("unsupported nested bottom-k request schema")
        scores = _readonly_le_f64_vector(self.scores, name="score vector")
        if not int(scores.size):
            raise ValueError("score vector must be nonempty")
        live = _readonly_le_u64_vector(
            self.live_physical_indices,
            name="live physical indices",
        )
        if not int(live.size):
            raise ValueError("live physical indices must be nonempty")
        if int(live[-1]) >= int(scores.size):
            raise ValueError("live physical index leaves the score route axis")
        if int(live[-1]) > int(np.iinfo(np.int64).max):
            raise ValueError("live physical index exceeds CUDA int64 addressing")
        if live.size > 1 and bool(np.any(live[1:] <= live[:-1])):
            raise ValueError("live physical indices must be sorted and unique")
        counts = _count_tuple(self.new_removal_counts, live_count=int(live.size))
        already_removed = _nonnegative_int(
            self.already_removed_count,
            name="already removed count",
        )
        confidence_box_hash = _identity(
            self.confidence_box_hash,
            name="confidence box hash",
        )
        route_axis_hash = _identity(self.route_axis_hash, name="route axis hash")
        ordered_route_axis_hash = (
            None
            if self.ordered_route_axis_hash is None
            else _identity(
                self.ordered_route_axis_hash,
                name="ordered route axis hash",
            )
        )
        backend = _backend(self.selector_backend)
        device = _identity(self.torch_device, name="torch device")
        families = tuple(self.action_families)
        prevalidated_hashes = tuple(
            _identity(item, name="prevalidated action family hash")
            for item in self.prevalidated_action_family_hashes
        )
        registry_hash = (
            None
            if self.action_family_registry_hash is None
            else _identity(
                self.action_family_registry_hash,
                name="action family registry hash",
            )
        )
        cumulative_sizes = tuple(
            _positive_int(item, name="cumulative action size")
            for item in self.cumulative_action_sizes
        )
        if cumulative_sizes and len(cumulative_sizes) != len(counts):
            raise ValueError("cumulative action sizes must align with counts")
        if cumulative_sizes and cumulative_sizes != tuple(
            already_removed + int(count) for count in counts
        ):
            raise ValueError("cumulative action sizes differ from requested counts")
        if families:
            if prevalidated_hashes and len(prevalidated_hashes) != len(families):
                raise ValueError("prevalidated family hashes must align with families")
            if len(families) != len(counts):
                raise ValueError("action families must align with removal counts")
            family_hash_values: list[str] = []
            for count, family in zip(counts, families, strict=True):
                if type(family) is not V1CorrectedAttributionActionFamily:
                    raise TypeError(
                        "action families must be V1CorrectedAttributionActionFamily"
                    )
                family_hash = family.assert_integrity()
                family_hash_values.append(family_hash)
                if not family_hash:
                    raise ValueError("action family hash is empty")
                if family.family_kind not in {
                    GLOBAL_FIXED_CARDINALITY_ACTION_FAMILY,
                    EXACT_QUOTA_PARTITION_MATROID_ACTION_FAMILY,
                }:
                    raise PermissionError(
                        "nested bottom-k only accepts fixed-cardinality or "
                        "prior-removed/live exact-quota families"
                    )
                if family.route_count != int(scores.size):
                    raise ValueError("action family route count differs")
                if family.route_axis_hash not in {
                    route_axis_hash,
                    ordered_route_axis_hash,
                }:
                    raise ValueError("action family route axis differs")
                if family.action_size != already_removed + int(count):
                    raise ValueError(
                        "action family cumulative size differs from requested count"
                    )
                if family.family_kind == GLOBAL_FIXED_CARDINALITY_ACTION_FAMILY:
                    if (
                        family.partition_names
                        or int(family.partition_offsets.size)
                        or int(family.partition_route_indices.size)
                        or int(family.partition_exact_quotas.size)
                    ):
                        raise PermissionError(
                            "global fixed-cardinality family carried partition metadata"
                        )
                elif (
                    tuple(family.partition_names)
                    != ("already_removed", "currently_live")
                    or tuple(int(value) for value in family.partition_exact_quotas)
                    != (already_removed, int(count))
                ):
                    raise PermissionError(
                        "exact-quota family must only freeze prior removed routes "
                        "and choose the requested live-route count"
                    )
            if prevalidated_hashes and tuple(family_hash_values) != prevalidated_hashes:
                raise ValueError("prevalidated family hashes do not match families")
            family_hashes = tuple(family_hash_values)
        else:
            if prevalidated_hashes:
                if len(prevalidated_hashes) != len(counts):
                    raise ValueError("prevalidated family hashes must align with counts")
                if registry_hash is None:
                    raise ValueError(
                        "prevalidated family hashes require a registry hash"
                    )
            family_hashes = prevalidated_hashes
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "live_physical_indices", live)
        object.__setattr__(self, "new_removal_counts", counts)
        object.__setattr__(self, "confidence_box_hash", confidence_box_hash)
        object.__setattr__(self, "route_axis_hash", route_axis_hash)
        object.__setattr__(self, "ordered_route_axis_hash", ordered_route_axis_hash)
        object.__setattr__(
            self,
            "prevalidated_action_family_hashes",
            prevalidated_hashes,
        )
        object.__setattr__(self, "action_family_registry_hash", registry_hash)
        object.__setattr__(self, "cumulative_action_sizes", cumulative_sizes)
        object.__setattr__(self, "already_removed_count", already_removed)
        object.__setattr__(self, "selector_backend", backend)
        object.__setattr__(self, "torch_device", device)
        object.__setattr__(self, "action_families", families)
        object.__setattr__(
            self,
            "_score_array_hash",
            _array_hash(scores, prefix=_REQUEST_PREFIX),
        )
        object.__setattr__(
            self,
            "_live_physical_indices_hash",
            _array_hash(live, prefix=_REQUEST_PREFIX),
        )
        object.__setattr__(self, "_family_hashes", family_hashes)
        object.__setattr__(self, "_request_hash", self._compute_request_hash())

    @property
    def route_count(self) -> int:
        return int(self.scores.size)

    @property
    def live_count(self) -> int:
        return int(self.live_physical_indices.size)

    @property
    def kmax(self) -> int:
        return max(self.new_removal_counts)

    @property
    def score_array_hash(self) -> str:
        if self._score_array_hash is None:
            raise RuntimeError("nested bottom-k request has not been admitted")
        return self._score_array_hash

    @property
    def live_physical_indices_hash(self) -> str:
        if self._live_physical_indices_hash is None:
            raise RuntimeError("nested bottom-k request has not been admitted")
        return self._live_physical_indices_hash

    @property
    def family_hashes(self) -> tuple[str, ...]:
        return self._family_hashes

    def _hash_payload(
        self,
        *,
        score_array_hash: str | None = None,
        live_physical_indices_hash: str | None = None,
        action_family_hashes: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "selector_id": NESTED_BOTTOM_K_SELECTOR_ID,
            "route_count": self.route_count,
            "live_count": self.live_count,
            "kmax": self.kmax,
            "score_array_hash": (
                self.score_array_hash
                if score_array_hash is None
                else _identity(score_array_hash, name="score array hash")
            ),
            "live_physical_indices_hash": (
                self.live_physical_indices_hash
                if live_physical_indices_hash is None
                else _identity(
                    live_physical_indices_hash,
                    name="live physical indices hash",
                )
            ),
            "new_removal_counts": self.new_removal_counts,
            "confidence_box_hash": self.confidence_box_hash,
            "route_axis_hash": self.route_axis_hash,
            "ordered_route_axis_hash": self.ordered_route_axis_hash,
            "action_family_hashes": (
                self.family_hashes
                if action_family_hashes is None
                else tuple(
                    _identity(item, name="action family hash")
                    for item in action_family_hashes
                )
            ),
            "action_family_registry_hash": self.action_family_registry_hash,
            "cumulative_action_sizes": self.cumulative_action_sizes,
            "already_removed_count": self.already_removed_count,
            "selector_backend": self.selector_backend,
            "torch_device": self.torch_device,
            "validation_rule": NESTED_BOTTOM_K_VALIDATION_RULE,
        }

    def _compute_request_hash(self) -> str:
        return content_hash(self._hash_payload(), prefix=_REQUEST_PREFIX)

    @property
    def cached_request_hash(self) -> str:
        if self._request_hash is None:
            raise RuntimeError("nested bottom-k request has not been admitted")
        return self._request_hash

    @property
    def request_hash(self) -> str:
        return self.cached_request_hash

    def assert_integrity(
        self,
        expected_hash: str | None = None,
        *,
        deep: bool = True,
    ) -> str:
        if deep:
            score_hash = _array_hash(self.scores, prefix=_REQUEST_PREFIX)
            live_hash = _array_hash(
                self.live_physical_indices,
                prefix=_REQUEST_PREFIX,
            )
            if score_hash != self.score_array_hash or live_hash != (
                self.live_physical_indices_hash
            ):
                raise ValueError("nested bottom-k request was tampered")
            if self.action_families:
                family_hashes = tuple(
                    family.assert_integrity() for family in self.action_families
                )
                if family_hashes != self.family_hashes:
                    raise ValueError("nested bottom-k request family was tampered")
            else:
                family_hashes = self.family_hashes
            observed = content_hash(
                self._hash_payload(
                    score_array_hash=score_hash,
                    live_physical_indices_hash=live_hash,
                    action_family_hashes=family_hashes,
                ),
                prefix=_REQUEST_PREFIX,
            )
        else:
            observed = self.cached_request_hash
        if self._request_hash is not None and observed != self._request_hash:
            raise ValueError("nested bottom-k request was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("nested bottom-k request hash does not match")
        object.__setattr__(self, "_request_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1NestedBottomKBoundaryWitness:
    """Adjacent selected/outside witness for one nested prefix boundary."""

    new_removal_count: int
    selected_upper_physical_index: int
    selected_upper_score: float
    outside_lower_physical_index: int | None
    outside_lower_score: float | None
    selected_upper_prefix_position: int
    outside_lower_prefix_position: int | None
    selected_upper_leq_outside_lower: bool | None
    schema: str = NESTED_BOTTOM_K_BOUNDARY_WITNESS_SCHEMA
    _witness_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.schema != NESTED_BOTTOM_K_BOUNDARY_WITNESS_SCHEMA:
            raise ValueError("unsupported nested bottom-k witness schema")
        count = _positive_int(self.new_removal_count, name="new removal count")
        selected = _nonnegative_int(
            self.selected_upper_physical_index,
            name="selected upper physical index",
        )
        selected_position = _nonnegative_int(
            self.selected_upper_prefix_position,
            name="selected upper prefix position",
        )
        if selected_position != count - 1:
            raise ValueError("selected upper position must equal count - 1")
        selected_score = float(self.selected_upper_score)
        if not math.isfinite(selected_score):
            raise ValueError("selected upper score must be finite")
        outside_index = (
            None
            if self.outside_lower_physical_index is None
            else _nonnegative_int(
                self.outside_lower_physical_index,
                name="outside lower physical index",
            )
        )
        outside_score = (
            None
            if self.outside_lower_score is None
            else float(self.outside_lower_score)
        )
        outside_position = (
            None
            if self.outside_lower_prefix_position is None
            else _nonnegative_int(
                self.outside_lower_prefix_position,
                name="outside lower prefix position",
            )
        )
        if (outside_index is None) != (outside_score is None):
            raise ValueError("outside lower witness is partially specified")
        if (outside_index is None) != (outside_position is None):
            raise ValueError("outside lower position is partially specified")
        if outside_score is not None and not math.isfinite(outside_score):
            raise ValueError("outside lower score must be finite")
        if outside_position is not None and outside_position != count:
            raise ValueError("outside lower position must equal count")
        if outside_index is None:
            relation: bool | None = None
        else:
            if outside_score is None:
                raise ValueError("outside lower score is missing")
            relation = (selected_score, selected) <= (outside_score, outside_index)
            if relation is not True:
                raise ValueError("boundary witness violates score/physical order")
        if (
            self.selected_upper_leq_outside_lower is not None
            and self.selected_upper_leq_outside_lower is not relation
        ):
            raise ValueError("boundary witness relation flag is inconsistent")
        object.__setattr__(self, "new_removal_count", count)
        object.__setattr__(self, "selected_upper_physical_index", selected)
        object.__setattr__(self, "selected_upper_score", selected_score)
        object.__setattr__(self, "outside_lower_physical_index", outside_index)
        object.__setattr__(self, "outside_lower_score", outside_score)
        object.__setattr__(self, "selected_upper_prefix_position", selected_position)
        object.__setattr__(self, "outside_lower_prefix_position", outside_position)
        object.__setattr__(self, "selected_upper_leq_outside_lower", relation)
        self.assert_integrity()

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "new_removal_count": self.new_removal_count,
            "selected_upper_physical_index": self.selected_upper_physical_index,
            "selected_upper_score": self.selected_upper_score,
            "outside_lower_physical_index": self.outside_lower_physical_index,
            "outside_lower_score": self.outside_lower_score,
            "selected_upper_prefix_position": self.selected_upper_prefix_position,
            "outside_lower_prefix_position": self.outside_lower_prefix_position,
            "selected_upper_leq_outside_lower": (
                self.selected_upper_leq_outside_lower
            ),
            "tie_rule": NESTED_BOTTOM_K_TIE_RULE,
        }

    def _compute_witness_hash(self) -> str:
        return content_hash(self._hash_payload(), prefix=_WITNESS_PREFIX)

    @property
    def witness_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_witness_hash()
        if self._witness_hash is not None and observed != self._witness_hash:
            raise ValueError("nested bottom-k boundary witness was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("nested bottom-k boundary witness hash does not match")
        object.__setattr__(self, "_witness_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1NestedBottomKActionRung:
    """One requested action set from the shared ordered prefix."""

    rung_index: int
    new_removal_count: int
    cumulative_action_size: int
    physical_action_indices: np.ndarray = field(repr=False, compare=False)
    ordered_prefix_index_hash: str
    boundary_witness: V1NestedBottomKBoundaryWitness
    family_hash: str | None = None
    schema: str = NESTED_BOTTOM_K_RUNG_SCHEMA
    _rung_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.schema != NESTED_BOTTOM_K_RUNG_SCHEMA:
            raise ValueError("unsupported nested bottom-k action rung schema")
        rung_index = _nonnegative_int(self.rung_index, name="rung index")
        count = _positive_int(self.new_removal_count, name="new removal count")
        cumulative = _positive_int(
            self.cumulative_action_size,
            name="cumulative action size",
        )
        physical = _readonly_le_u64_vector(
            self.physical_action_indices,
            name="physical action indices",
            size=count,
        )
        if physical.size > 1 and bool(np.any(physical[1:] <= physical[:-1])):
            raise ValueError("physical action indices must be sorted and unique")
        prefix_hash = _identity(
            self.ordered_prefix_index_hash,
            name="ordered prefix index hash",
        )
        if type(self.boundary_witness) is not V1NestedBottomKBoundaryWitness:
            raise TypeError("boundary witness must be typed")
        self.boundary_witness.assert_integrity()
        family_hash = (
            None
            if self.family_hash is None
            else _identity(self.family_hash, name="family hash")
        )
        object.__setattr__(self, "rung_index", rung_index)
        object.__setattr__(self, "new_removal_count", count)
        object.__setattr__(self, "cumulative_action_size", cumulative)
        object.__setattr__(self, "physical_action_indices", physical)
        object.__setattr__(self, "ordered_prefix_index_hash", prefix_hash)
        object.__setattr__(self, "family_hash", family_hash)
        self.assert_integrity()

    @property
    def physical_action_index_hash(self) -> str:
        return _array_hash(self.physical_action_indices, prefix=_ACTION_PREFIX)

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "rung_index": self.rung_index,
            "new_removal_count": self.new_removal_count,
            "cumulative_action_size": self.cumulative_action_size,
            "physical_action_indices": self.physical_action_indices,
            "physical_action_index_hash": self.physical_action_index_hash,
            "ordered_prefix_index_hash": self.ordered_prefix_index_hash,
            "boundary_witness_hash": self.boundary_witness.witness_hash,
            "family_hash": self.family_hash,
            "action_set_rule": NESTED_BOTTOM_K_ACTION_SET_RULE,
        }

    def _compute_rung_hash(self) -> str:
        return content_hash(self._hash_payload(), prefix=_ACTION_PREFIX)

    @property
    def rung_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_rung_hash()
        if self._rung_hash is not None and observed != self._rung_hash:
            raise ValueError("nested bottom-k action rung was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("nested bottom-k action rung hash does not match")
        object.__setattr__(self, "_rung_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1NestedBottomKScoreOrderReceipt:
    """Content-addressed receipt for exact nested bottom-k selection."""

    request_hash: str
    confidence_box_hash: str
    route_axis_hash: str
    route_count: int
    live_count: int
    kmax: int
    witness_prefix_count: int
    already_removed_count: int
    counts_in_request_order: tuple[int, ...]
    action_family_hashes: tuple[str, ...]
    action_family_registry_hash: str | None
    cumulative_action_sizes: tuple[int, ...]
    ordered_route_axis_hash: str | None
    backend: str
    requested_backend: str
    torch_device: str
    fallback_reason: str | None
    score_array_hash: str
    live_physical_indices_hash: str
    ordered_prefix_index_hash: str
    ordered_witness_prefix_index_hash: str
    rung_hashes: tuple[str, ...]
    physical_action_index_hashes: tuple[str, ...]
    boundary_witness_hashes: tuple[str, ...]
    timing_seconds: Mapping[str, float]
    cuda_peak_memory_bytes: int | None
    cuda_peak_target_bytes: int = NESTED_BOTTOM_K_CUDA_PEAK_TARGET_BYTES
    schema: str = NESTED_BOTTOM_K_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.schema != NESTED_BOTTOM_K_RECEIPT_SCHEMA:
            raise ValueError("unsupported nested bottom-k receipt schema")
        for name in (
            "request_hash",
            "confidence_box_hash",
            "route_axis_hash",
            "backend",
            "requested_backend",
            "torch_device",
            "score_array_hash",
            "live_physical_indices_hash",
            "ordered_prefix_index_hash",
            "ordered_witness_prefix_index_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        route_count = _positive_int(self.route_count, name="route count")
        live_count = _positive_int(self.live_count, name="live count")
        if live_count > route_count:
            raise ValueError("live count exceeds route count")
        kmax = _positive_int(self.kmax, name="kmax")
        witness_prefix_count = _positive_int(
            self.witness_prefix_count,
            name="witness prefix count",
        )
        if kmax > live_count or witness_prefix_count not in {
            kmax,
            min(kmax + 1, live_count),
        }:
            raise ValueError("witness prefix count is not adjacent to kmax")
        counts = _count_tuple(self.counts_in_request_order, live_count=live_count)
        if max(counts) != kmax:
            raise ValueError("kmax does not match requested counts")
        already_removed = _nonnegative_int(
            self.already_removed_count,
            name="already removed count",
        )
        family_hashes = tuple(
            _identity(item, name="action family hash")
            for item in self.action_family_hashes
        )
        if family_hashes and len(family_hashes) != len(counts):
            raise ValueError("action family hashes must align with counts")
        registry_hash = (
            None
            if self.action_family_registry_hash is None
            else _identity(
                self.action_family_registry_hash,
                name="action family registry hash",
            )
        )
        cumulative_sizes = tuple(
            _positive_int(item, name="cumulative action size")
            for item in self.cumulative_action_sizes
        )
        if cumulative_sizes and len(cumulative_sizes) != len(counts):
            raise ValueError("cumulative action sizes must align with counts")
        if cumulative_sizes and cumulative_sizes != tuple(
            already_removed + int(count) for count in counts
        ):
            raise ValueError("cumulative action sizes differ from requested counts")
        ordered_route_axis_hash = (
            None
            if self.ordered_route_axis_hash is None
            else _identity(
                self.ordered_route_axis_hash,
                name="ordered route axis hash",
            )
        )
        if self.backend not in {"numpy", "torch"}:
            raise ValueError("receipt backend must be numpy or torch")
        if self.requested_backend not in {"auto", "numpy", "torch"}:
            raise ValueError("requested backend is invalid")
        fallback = (
            None
            if self.fallback_reason is None
            else _identity(self.fallback_reason, name="fallback reason")
        )
        rung_hashes = tuple(_identity(item, name="rung hash") for item in self.rung_hashes)
        action_hashes = tuple(
            _identity(item, name="physical action index hash")
            for item in self.physical_action_index_hashes
        )
        witness_hashes = tuple(
            _identity(item, name="boundary witness hash")
            for item in self.boundary_witness_hashes
        )
        if not (
            len(rung_hashes) == len(action_hashes) == len(witness_hashes) == len(counts)
        ):
            raise ValueError("rung/action/witness hashes must align with counts")
        timing: dict[str, float] = {}
        for key, value in self.timing_seconds.items():
            metric = _identity(str(key), name="timing key")
            amount = float(value)
            if not math.isfinite(amount) or amount < 0.0:
                raise ValueError("timing values must be finite nonnegative seconds")
            timing[metric] = amount
        peak = (
            None
            if self.cuda_peak_memory_bytes is None
            else _nonnegative_int(
                self.cuda_peak_memory_bytes,
                name="CUDA peak memory bytes",
            )
        )
        target = _positive_int(
            self.cuda_peak_target_bytes,
            name="CUDA peak target bytes",
        )
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "live_count", live_count)
        object.__setattr__(self, "kmax", kmax)
        object.__setattr__(self, "witness_prefix_count", witness_prefix_count)
        object.__setattr__(self, "already_removed_count", already_removed)
        object.__setattr__(self, "counts_in_request_order", counts)
        object.__setattr__(self, "action_family_hashes", family_hashes)
        object.__setattr__(self, "action_family_registry_hash", registry_hash)
        object.__setattr__(self, "cumulative_action_sizes", cumulative_sizes)
        object.__setattr__(self, "ordered_route_axis_hash", ordered_route_axis_hash)
        object.__setattr__(self, "fallback_reason", fallback)
        object.__setattr__(self, "rung_hashes", rung_hashes)
        object.__setattr__(self, "physical_action_index_hashes", action_hashes)
        object.__setattr__(self, "boundary_witness_hashes", witness_hashes)
        object.__setattr__(self, "timing_seconds", _mapping_proxy(timing))
        object.__setattr__(self, "cuda_peak_memory_bytes", peak)
        object.__setattr__(self, "cuda_peak_target_bytes", target)
        self.assert_integrity()

    @property
    def cuda_peak_under_target(self) -> bool | None:
        if self.cuda_peak_memory_bytes is None:
            return None
        return self.cuda_peak_memory_bytes <= self.cuda_peak_target_bytes

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "selector_id": NESTED_BOTTOM_K_SELECTOR_ID,
            "request_hash": self.request_hash,
            "confidence_box_hash": self.confidence_box_hash,
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "live_count": self.live_count,
            "kmax": self.kmax,
            "witness_prefix_count": self.witness_prefix_count,
            "already_removed_count": self.already_removed_count,
            "counts_in_request_order": self.counts_in_request_order,
            "action_family_hashes": self.action_family_hashes,
            "action_family_registry_hash": self.action_family_registry_hash,
            "cumulative_action_sizes": self.cumulative_action_sizes,
            "ordered_route_axis_hash": self.ordered_route_axis_hash,
            "backend": self.backend,
            "requested_backend": self.requested_backend,
            "torch_device": self.torch_device,
            "fallback_reason": self.fallback_reason,
            "score_array_hash": self.score_array_hash,
            "live_physical_indices_hash": self.live_physical_indices_hash,
            "ordered_prefix_index_hash": self.ordered_prefix_index_hash,
            "ordered_witness_prefix_index_hash": self.ordered_witness_prefix_index_hash,
            "rung_hashes": self.rung_hashes,
            "physical_action_index_hashes": self.physical_action_index_hashes,
            "boundary_witness_hashes": self.boundary_witness_hashes,
            "cuda_peak_target_bytes": self.cuda_peak_target_bytes,
            "tie_rule": NESTED_BOTTOM_K_TIE_RULE,
            "action_set_rule": NESTED_BOTTOM_K_ACTION_SET_RULE,
            "validation_rule": NESTED_BOTTOM_K_VALIDATION_RULE,
            "exact_numpy_fallback_available": True,
            "public_integrity_checks": (
                "deterministic_selection_identity_timing_is_diagnostic_v1"
            ),
        }

    def _compute_receipt_hash(self) -> str:
        return content_hash(self._hash_payload(), prefix=_RECEIPT_PREFIX)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            **self._hash_payload(),
            "timing_seconds": dict(self.timing_seconds),
            "cuda_peak_memory_bytes": self.cuda_peak_memory_bytes,
            "cuda_peak_under_target": self.cuda_peak_under_target,
            "receipt_hash": self.receipt_hash,
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_receipt_hash()
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("nested bottom-k receipt was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("nested bottom-k receipt hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1NestedBottomKScoreOrderResult:
    """Exact nested prefix plus physical-sorted action rungs."""

    request_hash: str
    ordered_prefix_physical_indices: np.ndarray = field(repr=False, compare=False)
    rungs: tuple[V1NestedBottomKActionRung, ...]
    receipt: V1NestedBottomKScoreOrderReceipt
    schema: str = NESTED_BOTTOM_K_RESULT_SCHEMA
    _result_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.schema != NESTED_BOTTOM_K_RESULT_SCHEMA:
            raise ValueError("unsupported nested bottom-k result schema")
        request_hash = _identity(self.request_hash, name="request hash")
        ordered = _readonly_le_u64_vector(
            self.ordered_prefix_physical_indices,
            name="ordered prefix physical indices",
            size=self.receipt.kmax,
        )
        rungs = tuple(self.rungs)
        if not rungs:
            raise ValueError("nested bottom-k result requires rungs")
        if type(self.receipt) is not V1NestedBottomKScoreOrderReceipt:
            raise TypeError("nested bottom-k receipt must be typed")
        self.receipt.assert_integrity()
        if self.receipt.request_hash != request_hash:
            raise ValueError("result request hash differs from receipt")
        if _array_hash(ordered, prefix=_ORDER_PREFIX) != self.receipt.ordered_prefix_index_hash:
            raise ValueError("ordered prefix hash differs from receipt")
        for rung in rungs:
            if type(rung) is not V1NestedBottomKActionRung:
                raise TypeError("result rungs must be typed")
            rung.assert_integrity()
        if tuple(rung.rung_hash for rung in rungs) != self.receipt.rung_hashes:
            raise ValueError("result rung hashes differ from receipt")
        object.__setattr__(self, "request_hash", request_hash)
        object.__setattr__(self, "ordered_prefix_physical_indices", ordered)
        object.__setattr__(self, "rungs", rungs)
        self.assert_integrity()

    def action_for_count(self, count: int) -> np.ndarray:
        target = _positive_int(count, name="new removal count")
        for rung in self.rungs:
            if rung.new_removal_count == target:
                return rung.physical_action_indices
        raise KeyError(f"no nested bottom-k action for count {target}")

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "selector_id": NESTED_BOTTOM_K_SELECTOR_ID,
            "request_hash": self.request_hash,
            "ordered_prefix_physical_indices": self.ordered_prefix_physical_indices,
            "ordered_prefix_index_hash": self.receipt.ordered_prefix_index_hash,
            "rung_hashes": tuple(rung.rung_hash for rung in self.rungs),
            "receipt_hash": self.receipt.receipt_hash,
        }

    def _compute_result_hash(self) -> str:
        return content_hash(self._hash_payload(), prefix=_RESULT_PREFIX)

    @property
    def result_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_result_hash()
        if self._result_hash is not None and observed != self._result_hash:
            raise ValueError("nested bottom-k result was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("nested bottom-k result hash does not match")
        object.__setattr__(self, "_result_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class _SelectionPayload:
    witness_order: np.ndarray
    backend: str
    fallback_reason: str | None
    timing_seconds: Mapping[str, float]
    cuda_peak_memory_bytes: int | None


def _select_witness_order_numpy(
    scores: np.ndarray,
    live: np.ndarray,
    *,
    stop_count: int,
) -> np.ndarray:
    if stop_count <= 0 or stop_count > int(live.size):
        raise ValueError("witness stop count is outside the live axis")
    live_scores = scores[live]
    if stop_count == int(live.size):
        order = np.lexsort((live, live_scores))
        result = live[order]
        return _readonly_le_u64_vector(result, name="witness order")
    threshold = float(np.partition(live_scores, stop_count - 1)[stop_count - 1])
    strict_positions = np.flatnonzero(live_scores < threshold)
    tie_positions = np.flatnonzero(live_scores == threshold)
    needed_from_tie = stop_count - int(strict_positions.size)
    if needed_from_tie < 0 or needed_from_tie > int(tie_positions.size):
        raise RuntimeError("bottom-k threshold failed to bracket the witness prefix")
    if needed_from_tie:
        tie_order = np.argsort(live[tie_positions], kind="stable")
        chosen_ties = tie_positions[tie_order[:needed_from_tie]]
        selected_positions = np.concatenate((strict_positions, chosen_ties))
    else:
        selected_positions = strict_positions
    order = np.lexsort((live[selected_positions], live_scores[selected_positions]))
    result = live[selected_positions][order]
    return _readonly_le_u64_vector(result, name="witness order", size=stop_count)


def _contiguous_live_span(live: np.ndarray) -> tuple[int, int] | None:
    if not int(live.size):
        return None
    start = int(live[0])
    stop = int(live[-1]) + 1
    if stop - start == int(live.size):
        return start, stop
    return None


def _torch_tensor_from_numpy_safe(
    array: np.ndarray,
    *,
    dtype: Any,
    device: Any,
) -> Any:
    """Transfer NumPy storage without exposing a read-only alias to PyTorch."""

    if torch is None:
        raise V1NestedBottomKTorchUnavailable("PyTorch is unavailable")
    constructor = torch.as_tensor if array.flags.writeable else torch.tensor
    return constructor(array, dtype=dtype, device=device)


def _select_witness_order_torch(
    scores: np.ndarray,
    live: np.ndarray,
    *,
    stop_count: int,
    device: str,
) -> tuple[np.ndarray, Mapping[str, float], int | None]:
    if torch is None:
        raise V1NestedBottomKTorchUnavailable("PyTorch is unavailable")
    runtime_device = torch.device(device)
    if runtime_device.type == "cuda" and not torch.cuda.is_available():
        raise V1NestedBottomKTorchUnavailable("CUDA is unavailable")
    if stop_count <= 0 or stop_count > int(live.size):
        raise ValueError("witness stop count is outside the live axis")
    timings: dict[str, float] = {}
    peak: int | None = None
    if runtime_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runtime_device)
        torch.cuda.synchronize(runtime_device)
    host_started = time.perf_counter()
    live_span = _contiguous_live_span(live)
    if live_span is None:
        live_scores_np = np.ascontiguousarray(scores[live], dtype=np.float64)
        live_i64_np: np.ndarray | None = live.view(np.int64)
    else:
        start, stop = live_span
        live_scores_np = scores[start:stop]
        live_i64_np = None
    timings["host_materialization_seconds"] = time.perf_counter() - host_started

    transfer_started = time.perf_counter()
    live_scores = _torch_tensor_from_numpy_safe(
        live_scores_np,
        dtype=torch.float64,
        device=runtime_device,
    )
    if live_span is None:
        assert live_i64_np is not None
        live_physical = _torch_tensor_from_numpy_safe(
            live_i64_np,
            dtype=torch.long,
            device=runtime_device,
        )
    else:
        start, stop = live_span
        live_physical = torch.arange(
            start,
            stop,
            dtype=torch.long,
            device=runtime_device,
        )
    if runtime_device.type == "cuda":
        torch.cuda.synchronize(runtime_device)
    timings["h2d_transfer_seconds"] = time.perf_counter() - transfer_started

    selection_started = time.perf_counter()
    if stop_count == int(live.size):
        selected_physical = live_physical
        selected_scores = live_scores
    else:
        # ``torch.kthvalue`` has no deterministic CUDA kernel; the k-th
        # smallest value is taken from a full sort instead (deterministic
        # under ``use_deterministic_algorithms(True)``; identical value).
        kth = torch.sort(live_scores).values[int(stop_count) - 1]
        threshold = kth.item()
        strict_mask = live_scores < kth
        tie_mask = live_scores == kth
        strict_positions = torch.nonzero(strict_mask, as_tuple=False).reshape(-1)
        tie_positions = torch.nonzero(tie_mask, as_tuple=False).reshape(-1)
        needed_from_tie = int(stop_count) - int(strict_positions.numel())
        if needed_from_tie < 0 or needed_from_tie > int(tie_positions.numel()):
            raise RuntimeError("CUDA bottom-k threshold failed to bracket prefix")
        strict_physical = live_physical.index_select(0, strict_positions)
        strict_scores = live_scores.index_select(0, strict_positions)
        if needed_from_tie:
            tie_physical = live_physical.index_select(0, tie_positions)
            chosen_tie_physical = torch.sort(tie_physical).values[:needed_from_tie]
            chosen_tie_scores = torch.full(
                (needed_from_tie,),
                float(threshold),
                dtype=torch.float64,
                device=runtime_device,
            )
            selected_physical = torch.cat(
                (strict_physical, chosen_tie_physical),
                dim=0,
            )
            selected_scores = torch.cat((strict_scores, chosen_tie_scores), dim=0)
        else:
            selected_physical = strict_physical
            selected_scores = strict_scores
    if runtime_device.type == "cuda":
        torch.cuda.synchronize(runtime_device)
    timings["selection_seconds"] = time.perf_counter() - selection_started

    materialization_started = time.perf_counter()
    selected_physical_cpu = np.asarray(
        selected_physical.detach().cpu().numpy(),
        dtype=np.uint64,
    )
    selected_scores_cpu = np.asarray(
        selected_scores.detach().cpu().numpy(),
        dtype=np.float64,
    )
    if runtime_device.type == "cuda":
        torch.cuda.synchronize(runtime_device)
    timings["d2h_transfer_seconds"] = time.perf_counter() - materialization_started

    canonical_started = time.perf_counter()
    order = np.lexsort((selected_physical_cpu, selected_scores_cpu))
    witness_order = _readonly_le_u64_vector(
        selected_physical_cpu[order],
        name="CUDA witness order",
        size=stop_count,
    )
    timings["materialization_seconds"] = time.perf_counter() - canonical_started
    if runtime_device.type == "cuda":
        peak = int(torch.cuda.max_memory_allocated(runtime_device))
    return witness_order, _mapping_proxy(timings), peak


def _select_witness_order(
    request: V1NestedBottomKScoreOrderRequest,
    *,
    stop_count: int,
) -> _SelectionPayload:
    requested_backend = request.selector_backend
    if requested_backend in {"auto", "torch"}:
        try:
            witness_order, timings, peak = _select_witness_order_torch(
                request.scores,
                request.live_physical_indices,
                stop_count=stop_count,
                device=request.torch_device,
            )
            return _SelectionPayload(
                witness_order=witness_order,
                backend="torch",
                fallback_reason=None,
                timing_seconds=timings,
                cuda_peak_memory_bytes=peak,
            )
        except V1NestedBottomKTorchUnavailable as error:
            if requested_backend == "torch":
                raise
            fallback_reason = f"torch_unavailable_or_failed:{type(error).__name__}"
    else:
        fallback_reason = None

    started = time.perf_counter()
    witness_order = _select_witness_order_numpy(
        request.scores,
        request.live_physical_indices,
        stop_count=stop_count,
    )
    timings = {"selection_seconds": time.perf_counter() - started}
    return _SelectionPayload(
        witness_order=witness_order,
        backend="numpy",
        fallback_reason=fallback_reason,
        timing_seconds=_mapping_proxy(timings),
        cuda_peak_memory_bytes=None,
    )


def _boundary_witness(
    *,
    scores: np.ndarray,
    witness_order: np.ndarray,
    count: int,
) -> V1NestedBottomKBoundaryWitness:
    selected = int(witness_order[count - 1])
    outside: int | None
    outside_score: float | None
    outside_position: int | None
    if count < int(witness_order.size):
        outside = int(witness_order[count])
        outside_score = float(scores[outside])
        outside_position = count
    else:
        outside = None
        outside_score = None
        outside_position = None
    return V1NestedBottomKBoundaryWitness(
        new_removal_count=count,
        selected_upper_physical_index=selected,
        selected_upper_score=float(scores[selected]),
        outside_lower_physical_index=outside,
        outside_lower_score=outside_score,
        selected_upper_prefix_position=count - 1,
        outside_lower_prefix_position=outside_position,
        selected_upper_leq_outside_lower=None,
    )


def select_nested_bottom_k_score_order(
    request: V1NestedBottomKScoreOrderRequest,
) -> V1NestedBottomKScoreOrderResult:
    """Select every requested nested bottom-k action from one exact prefix."""

    if type(request) is not V1NestedBottomKScoreOrderRequest:
        raise TypeError("request must be V1NestedBottomKScoreOrderRequest")
    request_hash = request.cached_request_hash
    total_started = time.perf_counter()
    witness_stop = min(request.kmax + 1, request.live_count)
    selection = _select_witness_order(request, stop_count=witness_stop)
    prefix = _readonly_le_u64_vector(
        selection.witness_order[: request.kmax],
        name="ordered kmax prefix",
        size=request.kmax,
    )
    ordered_prefix_hash = _array_hash(prefix, prefix=_ORDER_PREFIX)
    ordered_witness_hash = _array_hash(selection.witness_order, prefix=_ORDER_PREFIX)

    rungs: list[V1NestedBottomKActionRung] = []
    for rung_index, count in enumerate(request.new_removal_counts):
        ordered_rung_prefix = prefix[:count]
        physical_action = np.array(ordered_rung_prefix, dtype=_LE_U64, copy=True)
        physical_action.sort(kind="stable")
        physical_action = _readonly_le_u64_vector(
            physical_action,
            name="physical action indices",
            size=count,
        )
        family_hash = (
            None
            if not request.family_hashes
            else request.family_hashes[rung_index]
        )
        rungs.append(
            V1NestedBottomKActionRung(
                rung_index=rung_index,
                new_removal_count=count,
                cumulative_action_size=request.already_removed_count + count,
                physical_action_indices=physical_action,
                ordered_prefix_index_hash=_array_hash(
                    ordered_rung_prefix,
                    prefix=_ORDER_PREFIX,
                ),
                boundary_witness=_boundary_witness(
                    scores=request.scores,
                    witness_order=selection.witness_order,
                    count=count,
                ),
                family_hash=family_hash,
            )
        )

    hashing_started = time.perf_counter()
    rung_hashes = tuple(rung.rung_hash for rung in rungs)
    action_hashes = tuple(rung.physical_action_index_hash for rung in rungs)
    witness_hashes = tuple(rung.boundary_witness.witness_hash for rung in rungs)
    hashing_seconds = time.perf_counter() - hashing_started
    timings = dict(selection.timing_seconds)
    timings["hashing_seconds"] = float(hashing_seconds)
    timings["total_seconds"] = float(time.perf_counter() - total_started)

    receipt = V1NestedBottomKScoreOrderReceipt(
        request_hash=request_hash,
        confidence_box_hash=request.confidence_box_hash,
        route_axis_hash=request.route_axis_hash,
        route_count=request.route_count,
        live_count=request.live_count,
        kmax=request.kmax,
        witness_prefix_count=int(selection.witness_order.size),
        already_removed_count=request.already_removed_count,
        counts_in_request_order=request.new_removal_counts,
        action_family_hashes=request.family_hashes,
        action_family_registry_hash=request.action_family_registry_hash,
        cumulative_action_sizes=request.cumulative_action_sizes,
        ordered_route_axis_hash=request.ordered_route_axis_hash,
        backend=selection.backend,
        requested_backend=request.selector_backend,
        torch_device=request.torch_device,
        fallback_reason=selection.fallback_reason,
        score_array_hash=request.score_array_hash,
        live_physical_indices_hash=request.live_physical_indices_hash,
        ordered_prefix_index_hash=ordered_prefix_hash,
        ordered_witness_prefix_index_hash=ordered_witness_hash,
        rung_hashes=rung_hashes,
        physical_action_index_hashes=action_hashes,
        boundary_witness_hashes=witness_hashes,
        timing_seconds=_mapping_proxy(timings),
        cuda_peak_memory_bytes=selection.cuda_peak_memory_bytes,
    )
    return V1NestedBottomKScoreOrderResult(
        request_hash=request_hash,
        ordered_prefix_physical_indices=prefix,
        rungs=tuple(rungs),
        receipt=receipt,
    )


def select_nested_bottom_k_actions(
    scores: np.ndarray,
    *,
    live_physical_indices: np.ndarray,
    new_removal_counts: Sequence[int],
    confidence_box_hash: str,
    route_axis_hash: str,
    action_families: Sequence[V1CorrectedAttributionActionFamily] = (),
    already_removed_count: int = 0,
    selector_backend: SelectorBackend = "auto",
    torch_device: str = "cuda",
) -> V1NestedBottomKScoreOrderResult:
    """Convenience wrapper around the typed request/result API."""

    request = V1NestedBottomKScoreOrderRequest(
        scores=scores,
        live_physical_indices=live_physical_indices,
        new_removal_counts=tuple(int(count) for count in new_removal_counts),
        confidence_box_hash=confidence_box_hash,
        route_axis_hash=route_axis_hash,
        action_families=tuple(action_families),
        already_removed_count=already_removed_count,
        selector_backend=selector_backend,
        torch_device=torch_device,
    )
    return select_nested_bottom_k_score_order(request)


def validate_nested_bottom_k_receipt_payload(
    payload: Mapping[str, Any],
    *,
    expected_hash: str | None = None,
) -> Mapping[str, Any]:
    """Validate a JSON-compatible selector receipt payload and return it.

    The receipt includes timing fields, so response-exactification replay must
    verify the recorded payload's content address instead of rebuilding a new
    timing-bearing receipt object and expecting byte equality.
    """

    raw = _exact_mapping(payload, name="nested bottom-k receipt payload")
    receipt_hash = _identity(raw.pop("receipt_hash", None), name="receipt hash")
    if raw.get("schema") != NESTED_BOTTOM_K_RECEIPT_SCHEMA:
        raise ValueError("nested bottom-k receipt payload has another schema")
    if raw.get("selector_id") != NESTED_BOTTOM_K_SELECTOR_ID:
        raise ValueError("nested bottom-k receipt payload names another selector")
    identity_raw = {
        key: value
        for key, value in raw.items()
        if key not in _RECEIPT_DIAGNOSTIC_FIELDS
    }
    observed = content_hash(identity_raw, prefix=_RECEIPT_PREFIX)
    if observed != receipt_hash:
        raise ValueError("nested bottom-k receipt payload was tampered")
    if expected_hash is not None and observed != expected_hash:
        raise ValueError("nested bottom-k receipt hash does not match")
    return _mapping_proxy({**raw, "receipt_hash": receipt_hash})


def nested_bottom_k_ordered_prefix_hash(indices: np.ndarray) -> str:
    """Hash a score/tie-ordered physical prefix with the selector identity."""

    ordered = np.asarray(indices)
    if ordered.dtype != _LE_U64 or ordered.ndim != 1:
        raise ValueError("ordered prefix indices must be one canonical uint64 vector")
    return _array_hash(ordered, prefix=_ORDER_PREFIX)


def nested_bottom_k_physical_action_hash(indices: np.ndarray) -> str:
    """Hash a physical-sorted action set with the selector identity."""

    physical = np.asarray(indices)
    if physical.dtype != _LE_U64 or physical.ndim != 1:
        raise ValueError("physical action indices must be one canonical uint64 vector")
    return _array_hash(physical, prefix=_ACTION_PREFIX)


def _prefix_external_descriptor_payload(
    *,
    selector_result: V1NestedBottomKScoreOrderResult | None = None,
    request_hash: str | None = None,
    selector_receipt_hash: str | None = None,
    ordered_prefix_index_hash: str | None = None,
    route_count: int | None = None,
    kmax: int | None = None,
    external_state_descriptor: Mapping[str, Any] | None = None,
    external_state_descriptor_hash: str | None = None,
) -> dict[str, Any]:
    if selector_result is not None:
        selector_result.assert_integrity()
        request_hash = selector_result.request_hash
        selector_receipt_hash = selector_result.receipt.receipt_hash
        ordered_prefix_index_hash = selector_result.receipt.ordered_prefix_index_hash
        route_count = selector_result.receipt.route_count
        kmax = selector_result.receipt.kmax
    return {
        "schema": NESTED_BOTTOM_K_PREFIX_EXTERNAL_DESCRIPTOR_SCHEMA,
        "selector_id": NESTED_BOTTOM_K_SELECTOR_ID,
        "state_schema": NESTED_BOTTOM_K_PREFIX_EXTERNAL_STATE_SCHEMA,
        "namespace": NESTED_BOTTOM_K_PREFIX_EXTERNAL_NAMESPACE,
        "logical_name": NESTED_BOTTOM_K_ORDERED_PREFIX_LOGICAL_NAME,
        "dtype": _LE_U64.str,
        "shape": [int(kmax) if kmax is not None else None],
        "request_hash": _identity(request_hash, name="request hash"),
        "selector_receipt_hash": _identity(
            selector_receipt_hash,
            name="selector receipt hash",
        ),
        "ordered_prefix_index_hash": _identity(
            ordered_prefix_index_hash,
            name="ordered prefix index hash",
        ),
        "route_count": _positive_int(route_count, name="route count"),
        "kmax": _positive_int(kmax, name="kmax"),
        "external_state_descriptor_hash": _identity(
            external_state_descriptor_hash,
            name="external state descriptor hash",
        ),
        "external_state_descriptor": _exact_mapping(
            external_state_descriptor,
            name="external state descriptor",
        ),
    }


def publish_nested_bottom_k_ordered_prefix(
    root: str | os.PathLike[str],
    result: V1NestedBottomKScoreOrderResult,
) -> Mapping[str, Any]:
    """Publish the ordered Kmax prefix as immutable external replay state."""

    if type(result) is not V1NestedBottomKScoreOrderResult:
        raise TypeError("nested bottom-k prefix publication requires a typed result")
    result.assert_integrity()
    ordered = _readonly_le_u64_vector(
        result.ordered_prefix_physical_indices,
        name="ordered kmax prefix",
        size=result.receipt.kmax,
    )
    if _array_hash(ordered, prefix=_ORDER_PREFIX) != (
        result.receipt.ordered_prefix_index_hash
    ):
        raise ValueError("ordered prefix differs from selector receipt")
    descriptor = publish_external_state(
        root,
        namespace=NESTED_BOTTOM_K_PREFIX_EXTERNAL_NAMESPACE,
        state_schema=NESTED_BOTTOM_K_PREFIX_EXTERNAL_STATE_SCHEMA,
        arrays={NESTED_BOTTOM_K_ORDERED_PREFIX_LOGICAL_NAME: ordered},
        policies={
            NESTED_BOTTOM_K_ORDERED_PREFIX_LOGICAL_NAME: V1ExternalArrayPolicy(
                logical_name=NESTED_BOTTOM_K_ORDERED_PREFIX_LOGICAL_NAME,
                expected_dtype=_LE_U64.str,
                expected_shape=(int(result.receipt.kmax),),
            )
        },
    )
    payload = _prefix_external_descriptor_payload(
        selector_result=result,
        external_state_descriptor=descriptor.to_json_dict(),
        external_state_descriptor_hash=descriptor.descriptor_hash,
    )
    descriptor_hash = content_hash(
        payload,
        prefix=_PREFIX_EXTERNAL_DESCRIPTOR_PREFIX,
    )
    return _mapping_proxy({**payload, "prefix_descriptor_hash": descriptor_hash})


def validate_nested_bottom_k_ordered_prefix_descriptor(
    payload: Mapping[str, Any],
    *,
    expected_hash: str | None = None,
) -> Mapping[str, Any]:
    """Validate the root-free external ordered-prefix descriptor payload."""

    raw = _exact_mapping(payload, name="ordered prefix external descriptor")
    descriptor_hash = _identity(
        raw.pop("prefix_descriptor_hash", None),
        name="ordered prefix external descriptor hash",
    )
    expected = {
        "schema",
        "selector_id",
        "state_schema",
        "namespace",
        "logical_name",
        "dtype",
        "shape",
        "request_hash",
        "selector_receipt_hash",
        "ordered_prefix_index_hash",
        "route_count",
        "kmax",
        "external_state_descriptor_hash",
        "external_state_descriptor",
    }
    if set(raw) != expected:
        raise ValueError("ordered prefix external descriptor fields are not exact")
    if raw["schema"] != NESTED_BOTTOM_K_PREFIX_EXTERNAL_DESCRIPTOR_SCHEMA:
        raise ValueError("ordered prefix external descriptor has another schema")
    if raw["selector_id"] != NESTED_BOTTOM_K_SELECTOR_ID:
        raise ValueError("ordered prefix external descriptor names another selector")
    if raw["state_schema"] != NESTED_BOTTOM_K_PREFIX_EXTERNAL_STATE_SCHEMA:
        raise ValueError("ordered prefix external descriptor names another state schema")
    if raw["namespace"] != NESTED_BOTTOM_K_PREFIX_EXTERNAL_NAMESPACE:
        raise ValueError("ordered prefix external descriptor names another namespace")
    if raw["logical_name"] != NESTED_BOTTOM_K_ORDERED_PREFIX_LOGICAL_NAME:
        raise ValueError("ordered prefix external descriptor names another array")
    if raw["dtype"] != _LE_U64.str:
        raise ValueError("ordered prefix external descriptor names another dtype")
    shape = raw["shape"]
    if type(shape) not in {list, tuple} or len(shape) != 1:
        raise ValueError("ordered prefix external descriptor shape is not exact")
    kmax = _positive_int(raw["kmax"], name="kmax")
    if int(shape[0]) != kmax:
        raise ValueError("ordered prefix external descriptor shape differs from kmax")
    _positive_int(raw["route_count"], name="route count")
    for name in (
        "request_hash",
        "selector_receipt_hash",
        "ordered_prefix_index_hash",
        "external_state_descriptor_hash",
    ):
        _identity(raw[name], name=name)
    external_descriptor_payload = _exact_mapping(
        raw["external_state_descriptor"],
        name="external state descriptor",
    )
    external_descriptor_payload = _json_exact(external_descriptor_payload)
    raw["external_state_descriptor"] = external_descriptor_payload
    external_descriptor = V1ExternalStateDescriptor.from_json_dict(
        external_descriptor_payload
    )
    if external_descriptor.descriptor_hash != raw["external_state_descriptor_hash"]:
        raise ValueError("ordered prefix external descriptor hash does not replay")
    if external_descriptor.namespace != NESTED_BOTTOM_K_PREFIX_EXTERNAL_NAMESPACE:
        raise ValueError("ordered prefix external state namespace differs")
    if external_descriptor.state_schema != NESTED_BOTTOM_K_PREFIX_EXTERNAL_STATE_SCHEMA:
        raise ValueError("ordered prefix external state schema differs")
    if len(external_descriptor.arrays) != 1:
        raise ValueError("ordered prefix external state must carry one array")
    identity = external_descriptor.arrays[0]
    if (
        identity.logical_name != NESTED_BOTTOM_K_ORDERED_PREFIX_LOGICAL_NAME
        or identity.dtype != _LE_U64.str
        or identity.shape != (kmax,)
    ):
        raise ValueError("ordered prefix external array identity differs")
    observed = content_hash(raw, prefix=_PREFIX_EXTERNAL_DESCRIPTOR_PREFIX)
    if observed != descriptor_hash:
        raise ValueError("ordered prefix external descriptor was tampered")
    if expected_hash is not None and observed != expected_hash:
        raise ValueError("ordered prefix external descriptor hash does not match")
    return _mapping_proxy({**raw, "prefix_descriptor_hash": descriptor_hash})


def open_nested_bottom_k_ordered_prefix(
    root: str | os.PathLike[str],
    payload: Mapping[str, Any],
    *,
    expected_hash: str | None = None,
) -> V1ExternalStateView:
    """Cold-open and verify the durable ordered Kmax prefix external state."""

    validated = validate_nested_bottom_k_ordered_prefix_descriptor(
        payload,
        expected_hash=expected_hash,
    )
    descriptor = V1ExternalStateDescriptor.from_json_dict(
        validated["external_state_descriptor"]
    )
    view = open_external_state(root, descriptor)
    try:
        ordered = np.asarray(view.arrays[NESTED_BOTTOM_K_ORDERED_PREFIX_LOGICAL_NAME])
        if (
            ordered.dtype != _LE_U64
            or ordered.ndim != 1
            or int(ordered.size) != int(validated["kmax"])
            or ordered.flags.writeable
        ):
            raise ValueError("opened ordered prefix array is not canonical read-only")
        if _array_hash(ordered, prefix=_ORDER_PREFIX) != (
            validated["ordered_prefix_index_hash"]
        ):
            raise ValueError("opened ordered prefix hash does not match descriptor")
    except BaseException:
        view.close()
        raise
    return view


__all__ = [
    "NESTED_BOTTOM_K_ACTION_SET_RULE",
    "NESTED_BOTTOM_K_BOUNDARY_WITNESS_SCHEMA",
    "NESTED_BOTTOM_K_CUDA_PEAK_TARGET_BYTES",
    "NESTED_BOTTOM_K_ORDERED_PREFIX_LOGICAL_NAME",
    "NESTED_BOTTOM_K_PREFIX_EXTERNAL_DESCRIPTOR_SCHEMA",
    "NESTED_BOTTOM_K_PREFIX_EXTERNAL_NAMESPACE",
    "NESTED_BOTTOM_K_PREFIX_EXTERNAL_STATE_SCHEMA",
    "NESTED_BOTTOM_K_RECEIPT_SCHEMA",
    "NESTED_BOTTOM_K_REQUEST_SCHEMA",
    "NESTED_BOTTOM_K_RESULT_SCHEMA",
    "NESTED_BOTTOM_K_RUNG_SCHEMA",
    "NESTED_BOTTOM_K_SELECTOR_ID",
    "NESTED_BOTTOM_K_TIE_RULE",
    "NESTED_BOTTOM_K_VALIDATION_RULE",
    "V1NestedBottomKActionRung",
    "V1NestedBottomKBoundaryWitness",
    "V1NestedBottomKScoreOrderReceipt",
    "V1NestedBottomKScoreOrderRequest",
    "V1NestedBottomKScoreOrderResult",
    "V1NestedBottomKTorchUnavailable",
    "count_ladder_from_live_fractions",
    "nested_bottom_k_ordered_prefix_hash",
    "nested_bottom_k_physical_action_hash",
    "open_nested_bottom_k_ordered_prefix",
    "publish_nested_bottom_k_ordered_prefix",
    "select_nested_bottom_k_actions",
    "select_nested_bottom_k_score_order",
    "validate_nested_bottom_k_ordered_prefix_descriptor",
    "validate_nested_bottom_k_receipt_payload",
]
