"""Batched exact zero-regret certificates for nested score-order prefixes.

The theorem here is the zero-budget specialization of the legacy
one-exchange corrected-attribution certificate for the nested prefix shape:
for live routes ``L`` and a point-order prefix ``P_k``, zero robust regret is
certified exactly when

``max_{i in P_k} upper_i <= min_{j in L \\ P_k} lower_j``.

The already-removed exact-quota partition is fixed at full quota and is
therefore vacuous.  The currently-live partition is the only nontrivial
component.  All rungs are certified from one request admission, one scan of
the live axis outside ``Kmax``, and prefix/suffix reductions over the ordered
Kmax prefix.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Final

import numpy as np

from mechanistic_systems.circuits.irg.modern_v1_proposal_statistics import (
    CORRECTED_ATTRIBUTION_ACTION_REGRET_CURRENCY,
    CORRECTED_ATTRIBUTION_ZERO_REGRET_EXCHANGE_THEOREM,
    EXACT_QUOTA_PARTITION_MATROID_ACTION_FAMILY,
    GLOBAL_FIXED_CARDINALITY_ACTION_FAMILY,
    V1CorrectedAttributionActionFamily,
    V1CorrectedAttributionConfidenceBoxReceipt,
    V1CorrectedAttributionConfidenceBoxV2Receipt,
)
from mechanistic_systems.circuits.irg.modern_v1_score_order_gpu import (
    nested_bottom_k_ordered_prefix_hash,
)
from mechanistic_systems.contracts.content_address import content_hash


NESTED_PREFIX_ZERO_REGRET_REQUEST_SCHEMA: Final = (
    "partition_free_v1_nested_prefix_zero_regret_batch_request_v1"
)
NESTED_PREFIX_ZERO_REGRET_RUNG_SCHEMA: Final = (
    "partition_free_v1_nested_prefix_zero_regret_rung_certificate_v1"
)
NESTED_PREFIX_ZERO_REGRET_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_nested_prefix_zero_regret_batch_receipt_v1"
)
NESTED_PREFIX_ZERO_REGRET_THEOREM_ID: Final = (
    "nested_prefix_corrected_attribution_zero_regret_one_exchange_v1"
)
NESTED_PREFIX_ZERO_REGRET_SELECTOR_SHAPE: Final = (
    "score_physical_ordered_nested_prefix_over_currently_live_routes_v1"
)
NESTED_PREFIX_ZERO_REGRET_TIE_RULE: Final = (
    "boundary_value_then_smallest_physical_route_id_v1"
)
NESTED_PREFIX_ZERO_REGRET_SCAN_RULE: Final = (
    "one_live_scan_plus_ordered_prefix_segment_reductions_v1"
)

_LE_F64: Final = np.dtype("<f8")
_LE_U64: Final = np.dtype("<u8")
_REQUEST_PREFIX: Final = "partition-free-v1-nested-prefix-zero-regret-request-"
_RUNG_PREFIX: Final = "partition-free-v1-nested-prefix-zero-regret-rung-"
_RECEIPT_PREFIX: Final = "partition-free-v1-nested-prefix-zero-regret-receipt-"
_PROPOSAL_IDENTITY_PREFIX: Final = (
    "partition-free-v1-nested-prefix-zero-regret-proposal-identity-"
)
_PHYSICAL_DELTA_HASH_PREFIX: Final = (
    "partition-free-v1-nested-prefix-zero-regret-physical-delta-axis-"
)
_LIVE_HASH_PREFIX: Final = "partition-free-v1-nested-prefix-zero-regret-live-axis-"
_REMOVED_HASH_PREFIX: Final = (
    "partition-free-v1-nested-prefix-zero-regret-already-removed-axis-"
)
_SCAN_CHUNK: Final = 8 * 1024 * 1024


ConfidenceBox = (
    V1CorrectedAttributionConfidenceBoxReceipt
    | V1CorrectedAttributionConfidenceBoxV2Receipt
)


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


def _finite_float(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _optional_finite_float(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, name=name)


def _readonly_le_u64_vector(
    value: Any,
    *,
    name: str,
    size: int | None = None,
) -> np.ndarray:
    try:
        source = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be one exact integer vector") from error
    if source.ndim != 1 or (source.size and source.dtype.kind not in ("i", "u")):
        raise ValueError(f"{name} must be one exact integer vector")
    if source.dtype.kind == "i" and bool(np.any(source < 0)):
        raise ValueError(f"{name} cannot contain negative coordinates")
    try:
        array = np.ascontiguousarray(source, dtype=_LE_U64).reshape(-1)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must fit the unsigned 64-bit route axis") from error
    if size is not None and int(array.size) != int(size):
        raise ValueError(f"{name} must contain exactly {int(size)} values")
    result = np.frombuffer(array.tobytes(order="C"), dtype=_LE_U64)
    result.setflags(write=False)
    return result


def _readonly_le_f64_vector(
    value: Any,
    *,
    name: str,
    size: int | None = None,
) -> np.ndarray:
    try:
        array = np.array(value, dtype=_LE_F64, order="C", copy=True).reshape(-1)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be one finite float64 vector") from error
    if size is not None and int(array.size) != int(size):
        raise ValueError(f"{name} must contain exactly {int(size)} values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    array[array == 0.0] = 0.0
    result = np.frombuffer(array.tobytes(order="C"), dtype=_LE_F64)
    result.setflags(write=False)
    return result


def _count_tuple(value: Sequence[int], *, live_count: int) -> tuple[int, ...]:
    counts = tuple(_positive_int(item, name="new removal count") for item in value)
    if not counts:
        raise ValueError("new removal counts must be nonempty")
    if len(set(counts)) != len(counts):
        raise ValueError("new removal counts must be unique")
    if max(counts) > int(live_count):
        raise ValueError("new removal count exceeds the live axis")
    return counts


def _array_hash(array: np.ndarray, *, prefix: str) -> str:
    return content_hash(np.asarray(array), prefix=prefix)


def nested_prefix_zero_regret_ordered_prefix_hash(indices: np.ndarray) -> str:
    """Hash an ordered score prefix under the canonical selector namespace."""

    return nested_bottom_k_ordered_prefix_hash(
        _readonly_le_u64_vector(indices, name="ordered prefix hash input")
    )


def nested_prefix_zero_regret_physical_delta_hash(indices: np.ndarray) -> str:
    """Hash one physical-sorted new-removal delta under the batch namespace."""

    return _array_hash(
        _readonly_le_u64_vector(indices, name="physical delta hash input"),
        prefix=_PHYSICAL_DELTA_HASH_PREFIX,
    )


def nested_prefix_zero_regret_live_hash(indices: np.ndarray) -> str:
    """Hash the currently-live physical route axis for batch replay."""

    return _array_hash(
        _readonly_le_u64_vector(indices, name="live hash input"),
        prefix=_LIVE_HASH_PREFIX,
    )


def nested_prefix_zero_regret_already_removed_hash(indices: np.ndarray) -> str:
    """Hash the already-removed physical route axis for batch replay."""

    return _array_hash(
        _readonly_le_u64_vector(indices, name="already-removed hash input"),
        prefix=_REMOVED_HASH_PREFIX,
    )


def _mapping_proxy(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(payload))


def _exact_mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def _same_axis(a: np.ndarray, b: np.ndarray) -> bool:
    return int(a.size) == int(b.size) and bool(np.array_equal(a, b))


def _family_partition_slice(
    family: V1CorrectedAttributionActionFamily,
    partition: int,
) -> np.ndarray:
    start = int(family.partition_offsets[partition])
    stop = int(family.partition_offsets[partition + 1])
    return family.partition_route_indices[start:stop]


def _coordinate_pairs_match(
    family: V1CorrectedAttributionActionFamily,
    box: ConfidenceBox,
) -> bool:
    return (
        family.registered_design_hash == box.design_hash
        and family.campaign_contract_hash == box.campaign_contract_hash
        and family.model_identity_hash == box.model_identity_hash
        and family.query_contract_hash == box.query_contract_hash
        and family.preprocessing_hash == box.preprocessing_hash
        and family.current_mask_hash == box.current_mask_hash
        and family.proposal_generation == box.proposal_generation
        and family.route_count == box.route_count
        and family.route_axis_hash == box.route_axis_hash
    )


@dataclass(frozen=True, slots=True)
class V1NestedPrefixZeroRegretBatchRequest:
    """Typed request for one batched nested-prefix zero-regret proof."""

    confidence_box: ConfidenceBox = field(repr=False, compare=False)
    ordered_prefix_physical_indices: np.ndarray = field(repr=False, compare=False)
    live_physical_indices: np.ndarray = field(repr=False, compare=False)
    new_removal_counts: tuple[int, ...]
    action_families: tuple[V1CorrectedAttributionActionFamily, ...] = field(
        default_factory=tuple,
        repr=False,
        compare=False,
    )
    prevalidated_action_family_hashes: tuple[str, ...] = ()
    action_family_registry_hash: str | None = None
    cumulative_action_sizes: tuple[int, ...] = ()
    already_removed_physical_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=_LE_U64),
        repr=False,
        compare=False,
    )
    selector_receipt_hash: str | None = None
    selector_result_hash: str | None = None
    selector_ordered_prefix_hash: str | None = None
    batch_source_hash: str | None = None
    registered_maximum_action_regret: float = 0.0
    schema: str = NESTED_PREFIX_ZERO_REGRET_REQUEST_SCHEMA
    _request_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )
    _confidence_box_hash: str | None = field(
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
    _ordered_prefix_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )
    _live_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )
    _already_removed_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.schema != NESTED_PREFIX_ZERO_REGRET_REQUEST_SCHEMA:
            raise ValueError("unsupported nested-prefix zero-regret request schema")
        if type(self.confidence_box) not in (
            V1CorrectedAttributionConfidenceBoxReceipt,
            V1CorrectedAttributionConfidenceBoxV2Receipt,
        ):
            raise TypeError("request requires an exact confidence-box receipt")
        confidence_hash = self.confidence_box.receipt_hash
        route_count = int(self.confidence_box.route_count)
        live = _readonly_le_u64_vector(
            self.live_physical_indices,
            name="live physical indices",
        )
        if not int(live.size):
            raise ValueError("live physical indices must be nonempty")
        if int(live[-1]) >= route_count:
            raise ValueError("live physical index leaves the route axis")
        if live.size > 1 and bool(np.any(live[1:] <= live[:-1])):
            raise ValueError("live physical indices must be sorted and unique")
        removed = _readonly_le_u64_vector(
            self.already_removed_physical_indices,
            name="already-removed physical indices",
        )
        if removed.size and int(removed[-1]) >= route_count:
            raise ValueError("already-removed route index leaves the route axis")
        if removed.size > 1 and bool(np.any(removed[1:] <= removed[:-1])):
            raise ValueError("already-removed indices must be sorted and unique")
        if removed.size and live.size:
            positions = np.searchsorted(live, removed)
            in_bounds = positions < int(live.size)
            overlaps = np.zeros(int(removed.size), dtype=np.bool_)
            overlaps[in_bounds] = live[positions[in_bounds]] == removed[in_bounds]
            if bool(np.any(overlaps)):
                raise ValueError("already-removed and live axes overlap")
        counts = _count_tuple(self.new_removal_counts, live_count=int(live.size))
        prefix = _readonly_le_u64_vector(
            self.ordered_prefix_physical_indices,
            name="ordered Kmax prefix",
            size=max(counts),
        )
        if prefix.size > 1:
            sorted_prefix = np.array(prefix, dtype=_LE_U64, copy=True)
            sorted_prefix.sort(kind="stable")
            if bool(np.any(sorted_prefix[1:] <= sorted_prefix[:-1])):
                raise ValueError("ordered Kmax prefix contains duplicate routes")
        live_membership = np.zeros(route_count, dtype=np.bool_)
        live_membership[live] = True
        if not bool(np.all(live_membership[prefix])):
            raise ValueError("ordered Kmax prefix leaves the live axis")
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
            int(removed.size) + int(count) for count in counts
        ):
            raise ValueError("cumulative action sizes differ from requested counts")
        family_hashes: list[str] = []
        if families:
            if len(families) != len(counts):
                raise ValueError("action families must align with requested counts")
            if prevalidated_hashes and len(prevalidated_hashes) != len(families):
                raise ValueError("prevalidated family hashes must align with families")
            for count, family in zip(counts, families, strict=True):
                if type(family) is not V1CorrectedAttributionActionFamily:
                    raise TypeError("action families must be exact typed families")
                family_hash = getattr(family, "_family_hash", None)
                if family_hash is None:
                    family_hash = family.assert_integrity()
                family_hash = _identity(family_hash, name="action family hash")
                family_hashes.append(family_hash)
                if family.maximum_action_regret != 0.0:
                    raise ValueError("nested-prefix theorem requires zero regret budget")
                if not _coordinate_pairs_match(family, self.confidence_box):
                    raise ValueError(
                        "action family and confidence box coordinates differ"
                    )
                if family.family_kind == GLOBAL_FIXED_CARDINALITY_ACTION_FAMILY:
                    if int(removed.size):
                        raise PermissionError(
                            "global generation-0 family cannot freeze removed routes"
                        )
                    if family.action_size != int(count):
                        raise ValueError("global action size differs from prefix count")
                elif family.family_kind == EXACT_QUOTA_PARTITION_MATROID_ACTION_FAMILY:
                    if tuple(family.partition_names) != (
                        "already_removed",
                        "currently_live",
                    ):
                        raise PermissionError(
                            "exact-quota family must split prior removed/currently live"
                        )
                    quotas = tuple(int(value) for value in family.partition_exact_quotas)
                    if quotas != (int(removed.size), int(count)):
                        raise ValueError("exact-quota family quotas differ from prefix")
                    if family.action_size != int(removed.size) + int(count):
                        raise ValueError("exact-quota family action size differs")
                    if not _same_axis(_family_partition_slice(family, 0), removed):
                        raise ValueError("already-removed partition differs")
                    if not _same_axis(_family_partition_slice(family, 1), live):
                        raise ValueError("currently-live partition differs")
                else:
                    raise PermissionError("unsupported action-family kind")
            if prevalidated_hashes and tuple(family_hashes) != prevalidated_hashes:
                raise ValueError("prevalidated family hashes do not match families")
        else:
            if len(prevalidated_hashes) != len(counts):
                raise ValueError("prevalidated family hashes must align with counts")
            if registry_hash is None:
                raise ValueError("prevalidated family hashes require a registry hash")
            if not cumulative_sizes:
                raise ValueError("prevalidated family hashes require cumulative sizes")
            family_hashes = list(prevalidated_hashes)
        selector_hash = (
            None
            if self.selector_receipt_hash is None
            else _identity(self.selector_receipt_hash, name="selector receipt hash")
        )
        selector_result_hash = (
            None
            if self.selector_result_hash is None
            else _identity(self.selector_result_hash, name="selector result hash")
        )
        selector_ordered_prefix_hash = (
            None
            if self.selector_ordered_prefix_hash is None
            else _identity(
                self.selector_ordered_prefix_hash,
                name="selector ordered prefix hash",
            )
        )
        source_hash = (
            None
            if self.batch_source_hash is None
            else _identity(self.batch_source_hash, name="batch source hash")
        )
        registered_regret = _finite_float(
            self.registered_maximum_action_regret,
            name="registered maximum action regret",
        )
        if registered_regret != 0.0:
            raise ValueError("nested-prefix theorem requires zero registered regret")
        ordered_prefix_hash = nested_prefix_zero_regret_ordered_prefix_hash(prefix)
        if (
            selector_ordered_prefix_hash is not None
            and selector_ordered_prefix_hash != ordered_prefix_hash
        ):
            raise ValueError("selector ordered prefix hash differs from Kmax prefix")
        object.__setattr__(self, "live_physical_indices", live)
        object.__setattr__(self, "already_removed_physical_indices", removed)
        object.__setattr__(self, "ordered_prefix_physical_indices", prefix)
        object.__setattr__(self, "new_removal_counts", counts)
        object.__setattr__(self, "action_families", families)
        object.__setattr__(
            self,
            "prevalidated_action_family_hashes",
            prevalidated_hashes,
        )
        object.__setattr__(self, "action_family_registry_hash", registry_hash)
        object.__setattr__(self, "cumulative_action_sizes", cumulative_sizes)
        object.__setattr__(self, "selector_receipt_hash", selector_hash)
        object.__setattr__(self, "selector_result_hash", selector_result_hash)
        object.__setattr__(
            self,
            "selector_ordered_prefix_hash",
            selector_ordered_prefix_hash,
        )
        object.__setattr__(self, "batch_source_hash", source_hash)
        object.__setattr__(self, "registered_maximum_action_regret", registered_regret)
        object.__setattr__(self, "_confidence_box_hash", confidence_hash)
        object.__setattr__(self, "_family_hashes", tuple(family_hashes))
        object.__setattr__(
            self,
            "_ordered_prefix_hash",
            ordered_prefix_hash,
        )
        object.__setattr__(self, "_live_hash", _array_hash(live, prefix=_LIVE_HASH_PREFIX))
        object.__setattr__(
            self,
            "_already_removed_hash",
            _array_hash(removed, prefix=_REMOVED_HASH_PREFIX),
        )
        object.__setattr__(self, "_request_hash", self._compute_request_hash())

    @property
    def route_count(self) -> int:
        return int(self.confidence_box.route_count)

    @property
    def live_count(self) -> int:
        return int(self.live_physical_indices.size)

    @property
    def already_removed_count(self) -> int:
        return int(self.already_removed_physical_indices.size)

    @property
    def kmax(self) -> int:
        return max(self.new_removal_counts)

    @property
    def confidence_box_hash(self) -> str:
        if self._confidence_box_hash is None:
            raise RuntimeError("request is not admitted")
        return self._confidence_box_hash

    @property
    def action_family_hashes(self) -> tuple[str, ...]:
        return self._family_hashes

    @property
    def ordered_prefix_hash(self) -> str:
        if self._ordered_prefix_hash is None:
            raise RuntimeError("request is not admitted")
        return self._ordered_prefix_hash

    @property
    def live_hash(self) -> str:
        if self._live_hash is None:
            raise RuntimeError("request is not admitted")
        return self._live_hash

    @property
    def already_removed_hash(self) -> str:
        if self._already_removed_hash is None:
            raise RuntimeError("request is not admitted")
        return self._already_removed_hash

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "theorem_id": NESTED_PREFIX_ZERO_REGRET_THEOREM_ID,
            "legacy_theorem_id": CORRECTED_ATTRIBUTION_ZERO_REGRET_EXCHANGE_THEOREM,
            "confidence_box_hash": self.confidence_box_hash,
            "confidence_box_schema": self.confidence_box.schema,
            "route_axis_hash": self.confidence_box.route_axis_hash,
            "route_count": self.route_count,
            "live_count": self.live_count,
            "already_removed_count": self.already_removed_count,
            "kmax": self.kmax,
            "new_removal_counts": self.new_removal_counts,
            "action_family_hashes": self.action_family_hashes,
            "action_family_registry_hash": self.action_family_registry_hash,
            "cumulative_action_sizes": self.cumulative_action_sizes,
            "ordered_prefix_hash": self.ordered_prefix_hash,
            "live_hash": self.live_hash,
            "already_removed_hash": self.already_removed_hash,
            "selector_receipt_hash": self.selector_receipt_hash,
            "selector_result_hash": self.selector_result_hash,
            "selector_ordered_prefix_hash": self.selector_ordered_prefix_hash,
            "batch_source_hash": self.batch_source_hash,
            "registered_maximum_action_regret": self.registered_maximum_action_regret,
            "selector_shape": NESTED_PREFIX_ZERO_REGRET_SELECTOR_SHAPE,
            "scan_rule": NESTED_PREFIX_ZERO_REGRET_SCAN_RULE,
            "tie_rule": NESTED_PREFIX_ZERO_REGRET_TIE_RULE,
        }

    def _compute_request_hash(self) -> str:
        return content_hash(self._hash_payload(), prefix=_REQUEST_PREFIX)

    @property
    def request_hash(self) -> str:
        if self._request_hash is None:
            raise RuntimeError("request is not admitted")
        return self._request_hash

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_request_hash()
        if self._request_hash is not None and observed != self._request_hash:
            raise ValueError("nested-prefix zero-regret request was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("nested-prefix zero-regret request hash does not match")
        object.__setattr__(self, "_request_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1NestedPrefixZeroRegretRungCertificate:
    """Compact certificate for one nested prefix rung."""

    batch_request_hash: str
    rung_index: int
    new_removal_count: int
    cumulative_action_size: int
    action_family_hash: str
    proposal_identity_hash: str
    physical_delta_hash: str
    ordered_prefix_hash: str
    selected_upper_route_index: int | None
    selected_upper: float | None
    outside_lower_route_index: int | None
    outside_lower: float | None
    raw_exchange_advantage: float | None
    maximum_feasible_one_exchange_regret: float
    action_regret_resolved: bool
    strict_action_set_optimality_certified: bool
    vacuous_full_live_quota: bool
    schema: str = NESTED_PREFIX_ZERO_REGRET_RUNG_SCHEMA
    _certificate_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.schema != NESTED_PREFIX_ZERO_REGRET_RUNG_SCHEMA:
            raise ValueError("unsupported nested-prefix rung certificate schema")
        object.__setattr__(
            self,
            "batch_request_hash",
            _identity(self.batch_request_hash, name="batch request hash"),
        )
        object.__setattr__(self, "rung_index", _nonnegative_int(self.rung_index, name="rung index"))
        object.__setattr__(
            self,
            "new_removal_count",
            _positive_int(self.new_removal_count, name="new removal count"),
        )
        object.__setattr__(
            self,
            "cumulative_action_size",
            _positive_int(self.cumulative_action_size, name="cumulative action size"),
        )
        for name in (
            "action_family_hash",
            "proposal_identity_hash",
            "physical_delta_hash",
            "ordered_prefix_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        selected_route = (
            None
            if self.selected_upper_route_index is None
            else _nonnegative_int(
                self.selected_upper_route_index,
                name="selected upper route index",
            )
        )
        outside_route = (
            None
            if self.outside_lower_route_index is None
            else _nonnegative_int(
                self.outside_lower_route_index,
                name="outside lower route index",
            )
        )
        selected_upper = _optional_finite_float(
            self.selected_upper,
            name="selected upper",
        )
        outside_lower = _optional_finite_float(self.outside_lower, name="outside lower")
        raw_advantage = _optional_finite_float(
            self.raw_exchange_advantage,
            name="raw exchange advantage",
        )
        regret = _finite_float(
            self.maximum_feasible_one_exchange_regret,
            name="maximum feasible one-exchange regret",
        )
        if regret < 0.0:
            raise ValueError("maximum feasible one-exchange regret must be nonnegative")
        if type(self.action_regret_resolved) is not bool:
            raise TypeError("action_regret_resolved must be an exact bool")
        if type(self.strict_action_set_optimality_certified) is not bool:
            raise TypeError("strict optimality flag must be an exact bool")
        if type(self.vacuous_full_live_quota) is not bool:
            raise TypeError("vacuous full-live quota flag must be an exact bool")
        if self.vacuous_full_live_quota:
            if (
                selected_route is not None
                or selected_upper is not None
                or raw_advantage is not None
                or outside_route is not None
                or outside_lower is not None
                or self.action_regret_resolved is not True
                or self.strict_action_set_optimality_certified is not True
                or regret != 0.0
            ):
                raise ValueError("vacuous full-live rung carries nonvacuous data")
        else:
            if (
                selected_route is None
                or selected_upper is None
                or outside_route is None
                or outside_lower is None
                or raw_advantage is None
            ):
                raise ValueError("nonvacuous rung has incomplete boundary witnesses")
            expected_advantage = float(selected_upper - outside_lower)
            if raw_advantage != expected_advantage:
                raise ValueError("raw exchange advantage does not match witnesses")
            expected_resolved = raw_advantage <= 0.0
            expected_regret = 0.0 if expected_resolved else raw_advantage
            expected_strict = raw_advantage < 0.0
            if (
                self.action_regret_resolved is not expected_resolved
                or regret != expected_regret
                or self.strict_action_set_optimality_certified is not expected_strict
            ):
                raise ValueError("rung zero-regret flags do not match witnesses")
        object.__setattr__(self, "selected_upper_route_index", selected_route)
        object.__setattr__(self, "outside_lower_route_index", outside_route)
        object.__setattr__(self, "selected_upper", selected_upper)
        object.__setattr__(self, "outside_lower", outside_lower)
        object.__setattr__(self, "raw_exchange_advantage", raw_advantage)
        object.__setattr__(self, "maximum_feasible_one_exchange_regret", regret)
        object.__setattr__(self, "_certificate_hash", self._compute_certificate_hash())

    @property
    def certified_maximum_action_regret(self) -> float | None:
        return 0.0 if self.action_regret_resolved else None

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "theorem_id": NESTED_PREFIX_ZERO_REGRET_THEOREM_ID,
            "legacy_theorem_id": CORRECTED_ATTRIBUTION_ZERO_REGRET_EXCHANGE_THEOREM,
            "batch_request_hash": self.batch_request_hash,
            "rung_index": self.rung_index,
            "new_removal_count": self.new_removal_count,
            "cumulative_action_size": self.cumulative_action_size,
            "action_family_hash": self.action_family_hash,
            "proposal_identity_hash": self.proposal_identity_hash,
            "physical_delta_hash": self.physical_delta_hash,
            "ordered_prefix_hash": self.ordered_prefix_hash,
            "selected_upper_route_index": self.selected_upper_route_index,
            "selected_upper": self.selected_upper,
            "outside_lower_route_index": self.outside_lower_route_index,
            "outside_lower": self.outside_lower,
            "raw_exchange_advantage": self.raw_exchange_advantage,
            "maximum_feasible_one_exchange_regret": (
                self.maximum_feasible_one_exchange_regret
            ),
            "certified_maximum_action_regret": self.certified_maximum_action_regret,
            "action_regret_resolved": self.action_regret_resolved,
            "strict_action_set_optimality_certified": (
                self.strict_action_set_optimality_certified
            ),
            "vacuous_full_live_quota": self.vacuous_full_live_quota,
            "action_regret_currency": CORRECTED_ATTRIBUTION_ACTION_REGRET_CURRENCY,
            "selector_shape": NESTED_PREFIX_ZERO_REGRET_SELECTOR_SHAPE,
            "tie_rule": NESTED_PREFIX_ZERO_REGRET_TIE_RULE,
            "calibration_or_oracle_outcomes_observed": False,
            "terminal_validation_observed": False,
        }

    def _compute_certificate_hash(self) -> str:
        return content_hash(self._hash_payload(), prefix=_RUNG_PREFIX)

    @property
    def certificate_hash(self) -> str:
        return self.assert_integrity()

    def to_json_dict(self) -> dict[str, Any]:
        return {**self._hash_payload(), "certificate_hash": self.certificate_hash}

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_certificate_hash()
        if self._certificate_hash is not None and observed != self._certificate_hash:
            raise ValueError("nested-prefix rung certificate was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("nested-prefix rung certificate hash does not match")
        object.__setattr__(self, "_certificate_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1NestedPrefixZeroRegretBatchReceipt:
    """Deterministic receipt for the batched nested-prefix theorem."""

    request_hash: str
    confidence_box_hash: str
    route_axis_hash: str
    route_count: int
    live_count: int
    already_removed_count: int
    kmax: int
    new_removal_counts: tuple[int, ...]
    action_family_hashes: tuple[str, ...]
    action_family_registry_hash: str | None
    cumulative_action_sizes: tuple[int, ...]
    ordered_prefix_hash: str
    live_hash: str
    already_removed_hash: str
    selector_receipt_hash: str | None
    selector_result_hash: str | None
    selector_ordered_prefix_hash: str | None
    batch_source_hash: str | None
    registered_maximum_action_regret: float
    rung_certificates: tuple[V1NestedPrefixZeroRegretRungCertificate, ...]
    live_routes_scanned: int
    ordered_prefix_routes_reduced: int
    confidence_deep_admission_count: int = 1
    family_deep_admission_count: int = 0
    legacy_single_action_certifier_calls: int = 0
    schema: str = NESTED_PREFIX_ZERO_REGRET_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.schema != NESTED_PREFIX_ZERO_REGRET_RECEIPT_SCHEMA:
            raise ValueError("unsupported nested-prefix batch receipt schema")
        for name in (
            "request_hash",
            "confidence_box_hash",
            "route_axis_hash",
            "ordered_prefix_hash",
            "live_hash",
            "already_removed_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        selector_hash = (
            None
            if self.selector_receipt_hash is None
            else _identity(self.selector_receipt_hash, name="selector receipt hash")
        )
        selector_result_hash = (
            None
            if self.selector_result_hash is None
            else _identity(self.selector_result_hash, name="selector result hash")
        )
        selector_prefix_hash = (
            None
            if self.selector_ordered_prefix_hash is None
            else _identity(
                self.selector_ordered_prefix_hash,
                name="selector ordered prefix hash",
            )
        )
        batch_source_hash = (
            None
            if self.batch_source_hash is None
            else _identity(self.batch_source_hash, name="batch source hash")
        )
        registered_regret = _finite_float(
            self.registered_maximum_action_regret,
            name="registered maximum action regret",
        )
        if registered_regret != 0.0:
            raise ValueError("batch receipt requires zero registered regret")
        route_count = _positive_int(self.route_count, name="route count")
        live_count = _positive_int(self.live_count, name="live count")
        removed_count = _nonnegative_int(
            self.already_removed_count,
            name="already removed count",
        )
        kmax = _positive_int(self.kmax, name="kmax")
        counts = _count_tuple(self.new_removal_counts, live_count=live_count)
        if max(counts) != kmax:
            raise ValueError("kmax differs from requested counts")
        family_hashes = tuple(
            _identity(item, name="action family hash")
            for item in self.action_family_hashes
        )
        if len(family_hashes) != len(counts):
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
        if cumulative_sizes and cumulative_sizes != tuple(
            removed_count + int(count) for count in counts
        ):
            raise ValueError("cumulative action sizes differ from requested counts")
        rungs = tuple(self.rung_certificates)
        if len(rungs) != len(counts):
            raise ValueError("rung certificates must align with counts")
        for index, (count, family_hash, rung) in enumerate(
            zip(counts, family_hashes, rungs, strict=True)
        ):
            if type(rung) is not V1NestedPrefixZeroRegretRungCertificate:
                raise TypeError("rung certificates must be exact typed certificates")
            rung.assert_integrity()
            if (
                rung.batch_request_hash != self.request_hash
                or rung.rung_index != index
                or rung.new_removal_count != count
                or rung.action_family_hash != family_hash
                or rung.cumulative_action_size != removed_count + count
            ):
                raise ValueError("rung certificate is not bound to the batch receipt")
        live_scanned = _positive_int(self.live_routes_scanned, name="live routes scanned")
        if live_scanned != live_count:
            raise ValueError("batch theorem receipt did not scan the live axis")
        reduced = _positive_int(
            self.ordered_prefix_routes_reduced,
            name="ordered prefix routes reduced",
        )
        if reduced != kmax:
            raise ValueError("batch theorem receipt did not reduce the full Kmax prefix")
        confidence_count = _positive_int(
            self.confidence_deep_admission_count,
            name="confidence deep admission count",
        )
        family_count = _nonnegative_int(
            self.family_deep_admission_count,
            name="family deep admission count",
        )
        legacy_calls = _nonnegative_int(
            self.legacy_single_action_certifier_calls,
            name="legacy single-action certifier calls",
        )
        if confidence_count != 1 or family_count not in {0, 1} or legacy_calls != 0:
            raise ValueError("batch theorem admission/call counts are invalid")
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "live_count", live_count)
        object.__setattr__(self, "already_removed_count", removed_count)
        object.__setattr__(self, "kmax", kmax)
        object.__setattr__(self, "new_removal_counts", counts)
        object.__setattr__(self, "action_family_hashes", family_hashes)
        object.__setattr__(self, "action_family_registry_hash", registry_hash)
        object.__setattr__(self, "cumulative_action_sizes", cumulative_sizes)
        object.__setattr__(self, "rung_certificates", rungs)
        object.__setattr__(self, "live_routes_scanned", live_scanned)
        object.__setattr__(self, "ordered_prefix_routes_reduced", reduced)
        object.__setattr__(self, "confidence_deep_admission_count", confidence_count)
        object.__setattr__(self, "family_deep_admission_count", family_count)
        object.__setattr__(self, "legacy_single_action_certifier_calls", legacy_calls)
        object.__setattr__(self, "selector_receipt_hash", selector_hash)
        object.__setattr__(self, "selector_result_hash", selector_result_hash)
        object.__setattr__(self, "selector_ordered_prefix_hash", selector_prefix_hash)
        object.__setattr__(self, "batch_source_hash", batch_source_hash)
        object.__setattr__(
            self,
            "registered_maximum_action_regret",
            registered_regret,
        )
        object.__setattr__(self, "_receipt_hash", self._compute_receipt_hash())

    @property
    def every_rung_resolved(self) -> bool:
        return all(rung.action_regret_resolved for rung in self.rung_certificates)

    @property
    def resolved_new_removal_counts(self) -> tuple[int, ...]:
        return tuple(
            rung.new_removal_count
            for rung in self.rung_certificates
            if rung.action_regret_resolved
        )

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "theorem_id": NESTED_PREFIX_ZERO_REGRET_THEOREM_ID,
            "legacy_theorem_id": CORRECTED_ATTRIBUTION_ZERO_REGRET_EXCHANGE_THEOREM,
            "request_hash": self.request_hash,
            "confidence_box_hash": self.confidence_box_hash,
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "live_count": self.live_count,
            "already_removed_count": self.already_removed_count,
            "kmax": self.kmax,
            "new_removal_counts": self.new_removal_counts,
            "action_family_hashes": self.action_family_hashes,
            "action_family_registry_hash": self.action_family_registry_hash,
            "cumulative_action_sizes": self.cumulative_action_sizes,
            "ordered_prefix_hash": self.ordered_prefix_hash,
            "live_hash": self.live_hash,
            "already_removed_hash": self.already_removed_hash,
            "selector_receipt_hash": self.selector_receipt_hash,
            "selector_result_hash": self.selector_result_hash,
            "selector_ordered_prefix_hash": self.selector_ordered_prefix_hash,
            "batch_source_hash": self.batch_source_hash,
            "registered_maximum_action_regret": self.registered_maximum_action_regret,
            "rung_certificates": tuple(
                rung.to_json_dict() for rung in self.rung_certificates
            ),
            "every_rung_resolved": self.every_rung_resolved,
            "resolved_new_removal_counts": self.resolved_new_removal_counts,
            "live_routes_scanned": self.live_routes_scanned,
            "ordered_prefix_routes_reduced": self.ordered_prefix_routes_reduced,
            "confidence_deep_admission_count": self.confidence_deep_admission_count,
            "family_deep_admission_count": self.family_deep_admission_count,
            "legacy_single_action_certifier_calls": (
                self.legacy_single_action_certifier_calls
            ),
            "scan_rule": NESTED_PREFIX_ZERO_REGRET_SCAN_RULE,
            "tie_rule": NESTED_PREFIX_ZERO_REGRET_TIE_RULE,
            "action_regret_currency": CORRECTED_ATTRIBUTION_ACTION_REGRET_CURRENCY,
            "calibration_or_oracle_outcomes_observed": False,
            "terminal_validation_observed": False,
        }

    def _compute_receipt_hash(self) -> str:
        return content_hash(self._hash_payload(), prefix=_RECEIPT_PREFIX)

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def to_json_dict(self) -> dict[str, Any]:
        return {**self._hash_payload(), "receipt_hash": self.receipt_hash}

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_receipt_hash()
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("nested-prefix zero-regret receipt was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("nested-prefix zero-regret receipt hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


def _segment_max_upper(
    upper: np.ndarray,
    routes: np.ndarray,
) -> tuple[int | None, float | None]:
    if not int(routes.size):
        return None, None
    values = upper[routes]
    best_value = float(np.max(values))
    best_route = int(np.min(routes[values == best_value]))
    return best_route, best_value


def _segment_min_lower(
    lower: np.ndarray,
    routes: np.ndarray,
) -> tuple[int | None, float | None]:
    if not int(routes.size):
        return None, None
    values = lower[routes]
    best_value = float(np.min(values))
    best_route = int(np.min(routes[values == best_value]))
    return best_route, best_value


def _better_max(
    current_route: int | None,
    current_value: float | None,
    candidate_route: int | None,
    candidate_value: float | None,
) -> tuple[int | None, float | None]:
    if candidate_route is None or candidate_value is None:
        return current_route, current_value
    if current_route is None or current_value is None:
        return candidate_route, candidate_value
    if candidate_value > current_value or (
        candidate_value == current_value and candidate_route < current_route
    ):
        return candidate_route, candidate_value
    return current_route, current_value


def _better_min(
    current_route: int | None,
    current_value: float | None,
    candidate_route: int | None,
    candidate_value: float | None,
) -> tuple[int | None, float | None]:
    if candidate_route is None or candidate_value is None:
        return current_route, current_value
    if current_route is None or current_value is None:
        return candidate_route, candidate_value
    if candidate_value < current_value or (
        candidate_value == current_value and candidate_route < current_route
    ):
        return candidate_route, candidate_value
    return current_route, current_value


def _outside_kmax_min(
    *,
    lower: np.ndarray,
    live: np.ndarray,
    selected_kmax: np.ndarray,
    route_count: int,
) -> tuple[int | None, float | None]:
    selected = np.zeros(route_count, dtype=np.bool_)
    selected[selected_kmax] = True
    best_route: int | None = None
    best_value: float | None = None
    for start in range(0, int(live.size), _SCAN_CHUNK):
        chunk = live[start : start + _SCAN_CHUNK]
        outside = chunk[~selected[chunk]]
        if not int(outside.size):
            continue
        route, value = _segment_min_lower(lower, outside)
        best_route, best_value = _better_min(best_route, best_value, route, value)
    return best_route, best_value


def _proposal_identity_hash(
    *,
    already_removed_hash: str,
    ordered_prefix: np.ndarray,
    count: int,
    route_count: int,
    route_axis_hash: str,
) -> tuple[str, str, str]:
    delta = np.array(ordered_prefix[:count], dtype=_LE_U64, copy=True)
    delta.sort(kind="stable")
    delta.setflags(write=False)
    physical_delta_hash = nested_prefix_zero_regret_physical_delta_hash(delta)
    ordered_hash = nested_prefix_zero_regret_ordered_prefix_hash(
        ordered_prefix[:count]
    )
    return (
        content_hash(
            {
                "route_count": int(route_count),
                "route_axis_hash": route_axis_hash,
                "already_removed_hash": already_removed_hash,
                "physical_delta_hash": physical_delta_hash,
                "ordered_prefix_hash": ordered_hash,
                "new_removal_count": int(count),
                "composition_rule": (
                    "already_removed_axis_plus_physical_delta_without_"
                    "materialized_cumulative_action_v1"
                ),
            },
            prefix=_PROPOSAL_IDENTITY_PREFIX,
        ),
        physical_delta_hash,
        ordered_hash,
    )


def certify_nested_prefix_zero_regret(
    request: V1NestedPrefixZeroRegretBatchRequest,
    *,
    expected_request_hash: str | None = None,
) -> V1NestedPrefixZeroRegretBatchReceipt:
    """Certify every nested prefix rung with one batched zero-regret theorem."""

    if type(request) is not V1NestedPrefixZeroRegretBatchRequest:
        raise TypeError("nested-prefix zero-regret certification requires its request")
    request.assert_integrity(expected_request_hash)
    request.confidence_box.assert_integrity(request.confidence_box_hash)
    if request.action_families:
        for family, family_hash in zip(
            request.action_families,
            request.action_family_hashes,
            strict=True,
        ):
            family.assert_integrity(family_hash)
    lower = np.asarray(request.confidence_box.class_balanced_absolute_lower)
    upper = np.asarray(request.confidence_box.class_balanced_absolute_upper)
    prefix = request.ordered_prefix_physical_indices
    counts_sorted = tuple(sorted(request.new_removal_counts))

    selected_by_count: dict[int, tuple[int, float]] = {}
    best_selected_route: int | None = None
    best_selected_value: float | None = None
    previous = 0
    for count in counts_sorted:
        route, value = _segment_max_upper(upper, prefix[previous:count])
        best_selected_route, best_selected_value = _better_max(
            best_selected_route,
            best_selected_value,
            route,
            value,
        )
        if best_selected_route is None or best_selected_value is None:
            raise RuntimeError("positive prefix count has no selected boundary")
        selected_by_count[count] = (best_selected_route, best_selected_value)
        previous = count

    outside_route, outside_value = _outside_kmax_min(
        lower=lower,
        live=request.live_physical_indices,
        selected_kmax=prefix,
        route_count=request.route_count,
    )
    outside_by_count: dict[int, tuple[int | None, float | None]] = {}
    previous = request.kmax
    for count in reversed(counts_sorted):
        route, value = _segment_min_lower(lower, prefix[count:previous])
        outside_route, outside_value = _better_min(
            outside_route,
            outside_value,
            route,
            value,
        )
        outside_by_count[count] = (outside_route, outside_value)
        previous = count

    rungs: list[V1NestedPrefixZeroRegretRungCertificate] = []
    for rung_index, (count, family_hash) in enumerate(
        zip(request.new_removal_counts, request.action_family_hashes, strict=True)
    ):
        selected_route, selected_upper = selected_by_count[count]
        lower_route, lower_value = outside_by_count[count]
        proposal_hash, physical_delta_hash, ordered_hash = _proposal_identity_hash(
            already_removed_hash=request.already_removed_hash,
            ordered_prefix=prefix,
            count=count,
            route_count=request.route_count,
            route_axis_hash=request.confidence_box.route_axis_hash,
        )
        if lower_route is None or lower_value is None:
            certificate_selected_route: int | None = None
            certificate_selected_upper: float | None = None
            raw_advantage = None
            regret = 0.0
            resolved = True
            strict = True
            vacuous = True
        else:
            certificate_selected_route = selected_route
            certificate_selected_upper = selected_upper
            raw_advantage = float(selected_upper - lower_value)
            resolved = raw_advantage <= 0.0
            regret = 0.0 if resolved else raw_advantage
            strict = raw_advantage < 0.0
            vacuous = False
        rungs.append(
            V1NestedPrefixZeroRegretRungCertificate(
                batch_request_hash=request.request_hash,
                rung_index=rung_index,
                new_removal_count=count,
                cumulative_action_size=request.already_removed_count + count,
                action_family_hash=family_hash,
                proposal_identity_hash=proposal_hash,
                physical_delta_hash=physical_delta_hash,
                ordered_prefix_hash=ordered_hash,
                selected_upper_route_index=certificate_selected_route,
                selected_upper=certificate_selected_upper,
                outside_lower_route_index=lower_route,
                outside_lower=lower_value,
                raw_exchange_advantage=raw_advantage,
                maximum_feasible_one_exchange_regret=regret,
                action_regret_resolved=resolved,
                strict_action_set_optimality_certified=strict,
                vacuous_full_live_quota=vacuous,
            )
        )

    return V1NestedPrefixZeroRegretBatchReceipt(
        request_hash=request.request_hash,
        confidence_box_hash=request.confidence_box_hash,
        route_axis_hash=request.confidence_box.route_axis_hash,
        route_count=request.route_count,
        live_count=request.live_count,
        already_removed_count=request.already_removed_count,
        kmax=request.kmax,
        new_removal_counts=request.new_removal_counts,
        action_family_hashes=request.action_family_hashes,
        action_family_registry_hash=request.action_family_registry_hash,
        cumulative_action_sizes=(
            request.cumulative_action_sizes
            if request.cumulative_action_sizes
            else tuple(
                request.already_removed_count + int(count)
                for count in request.new_removal_counts
            )
        ),
        ordered_prefix_hash=request.ordered_prefix_hash,
        live_hash=request.live_hash,
        already_removed_hash=request.already_removed_hash,
        selector_receipt_hash=request.selector_receipt_hash,
        selector_result_hash=request.selector_result_hash,
        selector_ordered_prefix_hash=request.selector_ordered_prefix_hash,
        batch_source_hash=request.batch_source_hash,
        registered_maximum_action_regret=request.registered_maximum_action_regret,
        rung_certificates=tuple(rungs),
        live_routes_scanned=request.live_count,
        ordered_prefix_routes_reduced=request.kmax,
        family_deep_admission_count=1 if request.action_families else 0,
    )


def _validate_rung_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = _exact_mapping(payload, name="nested-prefix rung certificate")
    certificate_hash = _identity(
        raw.pop("certificate_hash", None),
        name="rung certificate hash",
    )
    if raw.get("schema") != NESTED_PREFIX_ZERO_REGRET_RUNG_SCHEMA:
        raise ValueError("nested-prefix rung certificate has another schema")
    observed = content_hash(raw, prefix=_RUNG_PREFIX)
    if observed != certificate_hash:
        raise ValueError("nested-prefix rung certificate was tampered")
    return _mapping_proxy({**raw, "certificate_hash": certificate_hash})


def validate_nested_prefix_zero_regret_receipt_payload(
    payload: Mapping[str, Any],
    *,
    expected_hash: str | None = None,
) -> Mapping[str, Any]:
    """Validate a JSON-compatible batched zero-regret receipt payload."""

    raw = _exact_mapping(payload, name="nested-prefix zero-regret receipt")
    receipt_hash = _identity(raw.pop("receipt_hash", None), name="receipt hash")
    if raw.get("schema") != NESTED_PREFIX_ZERO_REGRET_RECEIPT_SCHEMA:
        raise ValueError("nested-prefix zero-regret receipt has another schema")
    rung_payloads = tuple(
        _validate_rung_payload(item)
        for item in raw.get("rung_certificates", ())
    )
    raw["rung_certificates"] = tuple(dict(item) for item in rung_payloads)
    observed = content_hash(raw, prefix=_RECEIPT_PREFIX)
    if observed != receipt_hash:
        raise ValueError("nested-prefix zero-regret receipt was tampered")
    if expected_hash is not None and observed != expected_hash:
        raise ValueError("nested-prefix zero-regret receipt hash does not match")
    return _mapping_proxy({**raw, "receipt_hash": receipt_hash})


__all__ = [
    "NESTED_PREFIX_ZERO_REGRET_RECEIPT_SCHEMA",
    "NESTED_PREFIX_ZERO_REGRET_REQUEST_SCHEMA",
    "NESTED_PREFIX_ZERO_REGRET_RUNG_SCHEMA",
    "NESTED_PREFIX_ZERO_REGRET_SCAN_RULE",
    "NESTED_PREFIX_ZERO_REGRET_SELECTOR_SHAPE",
    "NESTED_PREFIX_ZERO_REGRET_THEOREM_ID",
    "NESTED_PREFIX_ZERO_REGRET_TIE_RULE",
    "V1NestedPrefixZeroRegretBatchReceipt",
    "V1NestedPrefixZeroRegretBatchRequest",
    "V1NestedPrefixZeroRegretRungCertificate",
    "certify_nested_prefix_zero_regret",
    "nested_prefix_zero_regret_already_removed_hash",
    "nested_prefix_zero_regret_live_hash",
    "nested_prefix_zero_regret_ordered_prefix_hash",
    "nested_prefix_zero_regret_physical_delta_hash",
    "validate_nested_prefix_zero_regret_receipt_payload",
]
