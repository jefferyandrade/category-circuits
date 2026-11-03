"""Atomic custody of the conditioned ``D1`` and sampled-residual sources.

The full train-population witness is a conditional finite-population object:

```
mu = (1/N) * sum(D1) + sum_h (R_h / (N n_h)) * sum(D_W2,h).
```

The two terms deliberately remain in their native compact representations.
This module binds their identities and exact rational measure, and supplies a
single process-local lease which locks both owners while a downstream
consumer compacts the terms independently.  It never concatenates tensors,
evaluates an action, constructs a confidence statement, or opens C/K/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
import math
from numbers import Integral
from typing import Final, NoReturn, Sequence

from .contracts import content_hash
from .modern_v1_sampled_conditioned_d1_compact_source import (
    V1SampledConditionedD1CompactSourceResult,
)
from .modern_v1_sampled_residual_population_measure import (
    V1SampledResidualPopulationMeasureDesign,
    V1SampledResidualPopulationWeightReceipt,
)
from .modern_v1_sampled_v7_common_kkt_coordinate_bridge import (
    V1SampledV7CommonKKTCoordinateBridgeReceipt,
)
from .modern_v1_sampled_variable_shell_packet_registration import (
    SAMPLED_VARIABLE_SHELL_PACKET_REGISTRATION_RECEIPT_SCHEMA,
    V1SampledVariableShellPacketRegistrationReceipt,
)
from .modern_v1_signed_work_variable_shell_source_append import (
    V1SignedWorkVariableShellSourceAppendSegmentLease,
    V1SignedWorkVariableShellSourceAppendSegmentedHandle,
    V1SignedWorkVariableShellSourceAppendRefusal,
)
from .modern_v1_signed_work_variable_shell_source_cuda import (
    V1SignedWorkVariableShellSourceRefusal,
    V1SignedWorkVariableShellSourceTensorLease,
)


SAMPLED_FULL_POPULATION_COMPACT_SOURCE_SCHEMA: Final = (
    "partition_free_v1_sampled_full_population_compact_source_receipt_v1"
)
SAMPLED_FULL_POPULATION_COMPACT_SOURCE_LEASE_SCHEMA: Final = (
    "partition_free_v1_sampled_full_population_compact_source_lease_v1"
)
SAMPLED_ANCHOR_INFLUENCING_AXIS_SCHEMA: Final = (
    "partition_free_v1_sampled_anchor_influencing_axis_receipt_v1"
)
SAMPLED_FULL_POPULATION_COMPACT_SOURCE_ALGORITHM: Final = (
    "conditioned_D1_plus_stratum_resolved_residual_exact_rational_measure_"
    "atomic_two_owner_borrow_without_tensor_merge_v1"
)
SAMPLED_FULL_POPULATION_COMPACT_SOURCE_AUTHORITY: Final = (
    "full_population_compact_source_custody_and_exact_measure_only_no_action_"
    "confidence_C_K_O_commit_terminal_or_pareto_authority_v1"
)
SAMPLED_ANCHOR_INFLUENCING_AXIS_AUTHORITY: Final = (
    "exact_ordered_D_fixed_proposal_influencing_axis_custody_only_no_rows_"
    "statistics_action_C_K_O_commit_terminal_or_pareto_authority_v1"
)
SAMPLED_FULL_POPULATION_CANONICAL_STRATA: Final = ("target", "complement")

_PREFIX: Final = "partition-free-v1-sampled-full-population-compact-source-"
_APPEND_PREFIX: Final = "partition-free-v1-variable-shell-source-append-"
_D1_PREFIX: Final = "partition-free-v1-sampled-conditioned-D1-compact-source-"
_DW2_SUFFIX_PREFIX: Final = "partition-free-v1-sampled-variable-shell-D-W2-suffix-"
_SOURCE_AXIS_PREFIX: Final = "partition-free-v1-variable-shell-source-axis-"
_ISSUER_TOKEN: Final = object()


class V1SampledFullPopulationCompactSourceRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_IDENTITY = "stale_identity"
    AXIS_MISMATCH = "axis_mismatch"
    ORDER_MISMATCH = "order_mismatch"
    WEIGHT_MISMATCH = "weight_mismatch"
    CUSTODY_MISMATCH = "custody_mismatch"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SampledFullPopulationCompactSourceRefusal(RuntimeError):
    """Typed fail-closed boundary for the two-component source snapshot."""

    def __init__(
        self,
        code: V1SampledFullPopulationCompactSourceRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledFullPopulationCompactSourceRefusalCode:
            raise TypeError("full-population source refusal code has the wrong type")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledFullPopulationCompactSourceRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledFullPopulationCompactSourceRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledFullPopulationCompactSourceRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SampledFullPopulationCompactSourceRefusalCode.INVALID_TYPE,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SampledFullPopulationCompactSourceRefusalCode.AXIS_MISMATCH,
            f"{name} must be nonnegative",
        )
    return result


def _identity_axis(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not tuple or not value:
        _refuse(
            V1SampledFullPopulationCompactSourceRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty immutable string axis",
        )
    result = tuple(_identity(item, name=f"{name} item") for item in value)
    return result


def _ordered_unique_axis(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) not in {tuple, list}:
        _refuse(
            V1SampledFullPopulationCompactSourceRefusalCode.INVALID_TYPE,
            f"{name} must be one ordered string sequence",
        )
    result = tuple(_identity(item, name=f"{name} item") for item in value)
    if not result or len(result) != len(set(result)):
        _refuse(
            V1SampledFullPopulationCompactSourceRefusalCode.AXIS_MISMATCH,
            f"{name} must be nonempty and unique",
        )
    return result


def _label_axis(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not tuple or not value:
        _refuse(
            V1SampledFullPopulationCompactSourceRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty immutable string axis",
        )
    return tuple(_identity(item, name=f"{name} item") for item in value)


def _conditioned_d1_axis_hash(role: str, values: Sequence[str]) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_sampled_conditioned_D1_axis_v1",
            "role": role,
            "values": tuple(values),
        },
        prefix=_D1_PREFIX + role + "-axis-",
    )


def _conditioned_d1_pair_axis_hash(
    *,
    target: Sequence[str],
    complement: Sequence[str],
) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_sampled_variable_shell_D_W2_pair_axis_v1",
            "role": "D1",
            "target": tuple(target),
            "complement": tuple(complement),
        },
        prefix=_DW2_SUFFIX_PREFIX + "D1-pair-axis-",
    )


def _registered_axis_hash(role: str, values: Sequence[str]) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_variable_shell_registered_axis_v1",
            "role": role,
            "values": tuple(values),
        },
        prefix=_SOURCE_AXIS_PREFIX,
    )


def _influencing_axis_hash(role: str, values: Sequence[str]) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_sampled_anchor_influencing_axis_v1",
            "role": role,
            "values": tuple(values),
        },
        prefix=_PREFIX + "anchor-influencing-" + role + "-axis-",
    )


def _count_pair(value: object, *, name: str) -> tuple[int, int]:
    if type(value) is not tuple or len(value) != 2:
        _refuse(
            V1SampledFullPopulationCompactSourceRefusalCode.INVALID_TYPE,
            f"{name} must use the target/complement pair axis",
        )
    return tuple(
        _nonnegative_int(item, name=f"{name} item") for item in value
    )  # type: ignore[return-value]


def _fraction_pair(value: object, *, name: str) -> tuple[int, int]:
    if (
        type(value) is not tuple
        or len(value) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, Integral) for item in value
        )
        or int(value[0]) < 0
        or int(value[1]) <= 0
    ):
        _refuse(
            V1SampledFullPopulationCompactSourceRefusalCode.WEIGHT_MISMATCH,
            f"{name} must be one exact nonnegative rational pair",
        )
    fraction = Fraction(int(value[0]), int(value[1]))
    return fraction.numerator, fraction.denominator


def _canonical_center_radius(value: Fraction) -> tuple[float, float]:
    """Nearest binary64 center plus a strict outward representation radius."""

    center = float(value)
    if not math.isfinite(center):
        _refuse(
            V1SampledFullPopulationCompactSourceRefusalCode.WEIGHT_MISMATCH,
            "exact full-population weight overflowed binary64",
        )
    exact_error = abs(Fraction.from_float(center) - value)
    if exact_error == 0:
        return center, 0.0
    radius = float(exact_error)
    if Fraction.from_float(radius) < exact_error:
        radius = math.nextafter(radius, math.inf)
    radius = math.nextafter(radius, math.inf)
    if not math.isfinite(radius) or Fraction.from_float(radius) < exact_error:
        _refuse(
            V1SampledFullPopulationCompactSourceRefusalCode.WEIGHT_MISMATCH,
            "binary64 full-population weight radius is not outward",
        )
    return center, radius


def _validate_source_enclosure(
    *,
    value: Fraction,
    enclosure: object,
    name: str,
) -> None:
    if (
        type(enclosure) is not tuple
        or len(enclosure) != 3
        or any(type(item) is not float or not math.isfinite(item) for item in enclosure)
    ):
        _refuse(
            V1SampledFullPopulationCompactSourceRefusalCode.WEIGHT_MISMATCH,
            f"{name} must be one finite binary64 enclosure",
        )
    lower, center, upper = enclosure
    if not (
        lower <= center <= upper
        and center == float(value)
        and Fraction.from_float(lower) <= value <= Fraction.from_float(upper)
    ):
        _refuse(
            V1SampledFullPopulationCompactSourceRefusalCode.WEIGHT_MISMATCH,
            f"{name} does not enclose its exact rational weight",
        )


def _logical_stratum_axis_hash(
    receipts: Sequence[V1SampledVariableShellPacketRegistrationReceipt],
) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_sampled_full_population_logical_stratum_axis_v1",
            "canonical_stratum_order": SAMPLED_FULL_POPULATION_CANONICAL_STRATA,
            "packet_logical_stratum_axes": tuple(
                item.registered_stratum_ids for item in receipts
            ),
            "packet_append_mapping_hashes": tuple(
                item.append_logical_stratum_mapping_hash for item in receipts
            ),
        },
        prefix=_PREFIX + "logical-stratum-axis-",
    )


def _append_packet_axis_hash(
    *,
    lineage_hash: str,
    receipts: Sequence[V1SampledVariableShellPacketRegistrationReceipt],
) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_variable_shell_append_packet_axis_v1",
            "lineage_hash": lineage_hash,
            "packet_registration_hashes": tuple(
                item.append_packet_registration_hash for item in receipts
            ),
        },
        prefix=_APPEND_PREFIX + "packet-axis-",
    )


@dataclass(frozen=True, slots=True)
class V1SampledFullPopulationCompactSourceReceipt:
    conditioned_d1_source_receipt_hash: str
    conditioned_d1_device_source_receipt_hash: str
    residual_segmented_source_receipt_hash: str
    residual_append_receipt_hash: str
    residual_source_chain_hash: str
    packet_registration_receipt_hashes: tuple[str, ...]
    append_packet_registration_hashes: tuple[str, ...]
    append_packet_registration_axis_hash: str
    residual_measure_design_hash: str
    final_residual_weight_receipt_hash: str
    suffix_registration_hash: str
    generation_frame_hash: str
    discovery_stream_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    shell_freeze_receipt_hash: str
    shell_indices: tuple[int, ...]
    shell_route_ids: tuple[str, ...]
    shell_native_fiber_indices: tuple[int, ...]
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    conditioned_d1_registered_prefix_hash: str
    d1_pair_prefix_axis_hash: str
    final_planned_dw2_prefix_hash: str
    residual_registered_dw2_prefix_hash: str
    logical_stratum_axis_hash: str
    append_logical_stratum_mapping_hashes: tuple[str, ...]
    corrected_evidence_stream_source_hash: str
    literal_j_stream_source_hash: str
    conditioned_d1_same_capture_vjp_source_hash: str
    residual_same_capture_vjp_source_hash: str
    source_replay_hash: str
    kkt_branch_id: str
    kkt_branch_source_receipt_hash: str
    resource_axis_hash: str
    resource_source_receipt_hash: str
    witness_stream_source_hash: str
    same_capture_vjp_source_hash: str
    alpha_ledger_id: str
    alpha_coordinate_id: str
    alpha_lineage_hash: str
    design_hash: str
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    population_context_axis_hash: str
    population_draw_axis_hash: str
    context_axis_hash: str
    draw_axis_hash: str
    measurement_axis_hash: str
    capture_axis_hash: str
    source_packet_axis_hash: str
    population_stratum_counts: tuple[int, int]
    d1_stratum_counts: tuple[int, int]
    residual_capacity_stratum_counts: tuple[int, int]
    final_dw2_stratum_counts: tuple[int, int]
    population_count: int
    context_count: int
    d1_context_count: int
    residual_context_count: int
    route_count: int
    fiber_count: int
    cut_count: int
    segment_count: int
    d1_context_weight_fraction: tuple[int, int]
    residual_context_weight_fractions: tuple[tuple[int, int], tuple[int, int]]
    d1_context_weight_binary64_center: float
    d1_context_weight_binary64_radius: float
    residual_context_weight_binary64_centers: tuple[float, float]
    residual_context_weight_binary64_radii: tuple[float, float]
    common_measure_denominator: int
    d1_context_measure_quantum: int
    residual_context_measure_quanta: tuple[int, int]
    total_measure_quanta: int
    common_kkt_coordinate_bridge_receipt_hash: str | None = None
    canonical_stratum_order: tuple[str, str] = SAMPLED_FULL_POPULATION_CANONICAL_STRATA
    exact_full_population_measure_closed: bool = True
    exact_packet_quanta_closed: bool = True
    final_cumulative_counts_closed: bool = True
    component_source_owners_retained_without_tensor_merge: bool = True
    atomic_two_owner_borrow_protocol: bool = True
    residual_prefix_snapshot_bound: bool = True
    finite_census_axis_custody_complete: bool = False
    finite_census_complete: bool = False
    context_by_route_rows_retained: bool = False
    component_tensors_merged: bool = False
    action_authority: bool = False
    confidence_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    algorithm: str = SAMPLED_FULL_POPULATION_COMPACT_SOURCE_ALGORITHM
    authority_scope: str = SAMPLED_FULL_POPULATION_COMPACT_SOURCE_AUTHORITY
    schema: str = SAMPLED_FULL_POPULATION_COMPACT_SOURCE_SCHEMA
    _issuance_token: object = field(default=None, repr=False, compare=False)
    _bound_receipt_hash: str | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self._issuance_token is not _ISSUER_TOKEN:
            _refuse(
                V1SampledFullPopulationCompactSourceRefusalCode.INVALID_TYPE,
                "full-population source receipts must be compositor-issued",
            )
        identity_names = (
            "conditioned_d1_source_receipt_hash",
            "conditioned_d1_device_source_receipt_hash",
            "residual_segmented_source_receipt_hash",
            "residual_append_receipt_hash",
            "residual_source_chain_hash",
            "append_packet_registration_axis_hash",
            "residual_measure_design_hash",
            "final_residual_weight_receipt_hash",
            "suffix_registration_hash",
            "generation_frame_hash",
            "discovery_stream_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "shell_freeze_receipt_hash",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
            "conditioned_d1_registered_prefix_hash",
            "d1_pair_prefix_axis_hash",
            "final_planned_dw2_prefix_hash",
            "residual_registered_dw2_prefix_hash",
            "logical_stratum_axis_hash",
            "corrected_evidence_stream_source_hash",
            "literal_j_stream_source_hash",
            "conditioned_d1_same_capture_vjp_source_hash",
            "residual_same_capture_vjp_source_hash",
            "source_replay_hash",
            "kkt_branch_id",
            "kkt_branch_source_receipt_hash",
            "resource_axis_hash",
            "resource_source_receipt_hash",
            "witness_stream_source_hash",
            "same_capture_vjp_source_hash",
            "alpha_ledger_id",
            "alpha_coordinate_id",
            "alpha_lineage_hash",
            "design_hash",
            "registered_d_prefix_hash",
            "registered_d_w2_prefix_hash",
            "population_context_axis_hash",
            "population_draw_axis_hash",
            "context_axis_hash",
            "draw_axis_hash",
            "measurement_axis_hash",
            "capture_axis_hash",
            "source_packet_axis_hash",
        )
        for name in identity_names:
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if self.common_kkt_coordinate_bridge_receipt_hash is not None:
            object.__setattr__(
                self,
                "common_kkt_coordinate_bridge_receipt_hash",
                _identity(
                    self.common_kkt_coordinate_bridge_receipt_hash,
                    name="common KKT coordinate bridge receipt hash",
                ),
            )
        for name in (
            "packet_registration_receipt_hashes",
            "append_packet_registration_hashes",
            "append_logical_stratum_mapping_hashes",
        ):
            object.__setattr__(
                self,
                name,
                _identity_axis(getattr(self, name), name=name),
            )
        if not (
            len(self.packet_registration_receipt_hashes)
            == len(self.append_packet_registration_hashes)
            == len(self.append_logical_stratum_mapping_hashes)
            == self.segment_count
        ):
            _refuse(
                V1SampledFullPopulationCompactSourceRefusalCode.AXIS_MISMATCH,
                "packet, segment, and logical-stratum identity axes do not align",
            )
        for name in (
            "proposal_generation",
            "population_count",
            "context_count",
            "d1_context_count",
            "residual_context_count",
            "route_count",
            "fiber_count",
            "cut_count",
            "segment_count",
            "common_measure_denominator",
            "d1_context_measure_quantum",
            "total_measure_quanta",
        ):
            value = _nonnegative_int(getattr(self, name), name=name)
            if name not in {"proposal_generation"} and value <= 0:
                _refuse(
                    V1SampledFullPopulationCompactSourceRefusalCode.AXIS_MISMATCH,
                    f"{name} must be positive",
                )
            object.__setattr__(self, name, value)
        populations = _count_pair(
            self.population_stratum_counts,
            name="population stratum counts",
        )
        d1 = _count_pair(self.d1_stratum_counts, name="D1 stratum counts")
        residual_capacity = _count_pair(
            self.residual_capacity_stratum_counts,
            name="residual-capacity stratum counts",
        )
        final = _count_pair(
            self.final_dw2_stratum_counts,
            name="final D_W2 stratum counts",
        )
        residual_quanta = _count_pair(
            self.residual_context_measure_quanta,
            name="residual context measure quanta",
        )
        if (
            self.population_count != sum(populations)
            or self.context_count != self.d1_context_count + self.residual_context_count
            or self.d1_context_count != sum(d1)
            or self.residual_context_count != sum(final)
            or any(
                d1[index] + residual_capacity[index] != populations[index]
                for index in (0, 1)
            )
            or any(not 0 < final[index] <= residual_capacity[index] for index in (0, 1))
        ):
            _refuse(
                V1SampledFullPopulationCompactSourceRefusalCode.AXIS_MISMATCH,
                "D1, sampled residual, and complete population counts do not close",
            )
        d1_weight_pair = _fraction_pair(
            self.d1_context_weight_fraction,
            name="D1 context weight",
        )
        residual_weight_pairs = tuple(
            _fraction_pair(value, name="residual context weight")
            for value in self.residual_context_weight_fractions
        )
        d1_weight = Fraction(*d1_weight_pair)
        residual_weights = tuple(Fraction(*value) for value in residual_weight_pairs)
        if (
            d1_weight != Fraction(1, self.population_count)
            or residual_weights
            != tuple(
                Fraction(
                    residual_capacity[index],
                    self.population_count * final[index],
                )
                for index in (0, 1)
            )
            or Fraction(
                self.d1_context_measure_quantum, self.common_measure_denominator
            )
            != d1_weight
            or any(
                Fraction(residual_quanta[index], self.common_measure_denominator)
                != residual_weights[index]
                for index in (0, 1)
            )
            or self.total_measure_quanta != self.common_measure_denominator
            or self.d1_context_count * self.d1_context_measure_quantum
            + sum(final[index] * residual_quanta[index] for index in (0, 1))
            != self.total_measure_quanta
        ):
            _refuse(
                V1SampledFullPopulationCompactSourceRefusalCode.WEIGHT_MISMATCH,
                "exact D1-plus-residual integer measure does not close to one",
            )
        centers = (self.d1_context_weight_binary64_center,) + tuple(
            self.residual_context_weight_binary64_centers
        )
        radii = (self.d1_context_weight_binary64_radius,) + tuple(
            self.residual_context_weight_binary64_radii
        )
        exact_weights = (d1_weight,) + residual_weights
        if (
            len(centers) != 3
            or len(radii) != 3
            or any(
                type(item) is not float or not math.isfinite(item)
                for item in centers + radii
            )
            or any(item < 0.0 for item in radii)
            or any(
                (center, radius) != _canonical_center_radius(exact)
                for center, radius, exact in zip(
                    centers, radii, exact_weights, strict=True
                )
            )
        ):
            _refuse(
                V1SampledFullPopulationCompactSourceRefusalCode.WEIGHT_MISMATCH,
                "binary64 weight centers/radii are not the canonical rational enclosures",
            )
        required_true = (
            self.exact_full_population_measure_closed,
            self.exact_packet_quanta_closed,
            self.final_cumulative_counts_closed,
            self.component_source_owners_retained_without_tensor_merge,
            self.atomic_two_owner_borrow_protocol,
            self.residual_prefix_snapshot_bound,
        )
        required_false = (
            self.context_by_route_rows_retained,
            self.component_tensors_merged,
            self.action_authority,
            self.confidence_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        census_flags = (
            self.finite_census_axis_custody_complete,
            self.finite_census_complete,
        )
        if (
            self.canonical_stratum_order != SAMPLED_FULL_POPULATION_CANONICAL_STRATA
            or any(
                type(item) is not bool
                for item in required_true + required_false + census_flags
            )
            or any(item is not True for item in required_true)
            or any(item is not False for item in required_false)
            or self.algorithm != SAMPLED_FULL_POPULATION_COMPACT_SOURCE_ALGORITHM
            or self.authority_scope != SAMPLED_FULL_POPULATION_COMPACT_SOURCE_AUTHORITY
            or self.schema != SAMPLED_FULL_POPULATION_COMPACT_SOURCE_SCHEMA
        ):
            _refuse(
                V1SampledFullPopulationCompactSourceRefusalCode.AUTHORITY_VIOLATION,
                "full-population compact receipt crossed its custody-only boundary",
            )
        census_counts_complete = final == residual_capacity
        if (
            self.finite_census_axis_custody_complete is not census_counts_complete
            or self.finite_census_complete is not census_counts_complete
        ):
            _refuse(
                V1SampledFullPopulationCompactSourceRefusalCode.AUTHORITY_VIOLATION,
                "finite-census status does not replay residual-capacity closure",
            )
        object.__setattr__(self, "population_stratum_counts", populations)
        object.__setattr__(self, "d1_stratum_counts", d1)
        object.__setattr__(self, "residual_capacity_stratum_counts", residual_capacity)
        object.__setattr__(self, "final_dw2_stratum_counts", final)
        object.__setattr__(self, "residual_context_measure_quanta", residual_quanta)
        object.__setattr__(self, "d1_context_weight_fraction", d1_weight_pair)
        object.__setattr__(
            self, "residual_context_weight_fractions", residual_weight_pairs
        )
        observed = self._compute_receipt_hash()
        if (
            self._bound_receipt_hash is not None
            and observed != self._bound_receipt_hash
        ):
            _refuse(
                V1SampledFullPopulationCompactSourceRefusalCode.STALE_IDENTITY,
                "full-population compact receipt changed after issuance",
            )
        object.__setattr__(self, "_bound_receipt_hash", observed)

    def _compute_receipt_hash(self) -> str:
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
        }
        # Preserve byte-identical legacy receipt identities when the optional
        # V7 historical-coordinate bridge is absent.
        if self.common_kkt_coordinate_bridge_receipt_hash is None:
            payload.pop("common_kkt_coordinate_bridge_receipt_hash")
        return content_hash(
            payload,
            prefix=_PREFIX + "receipt-",
        )

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_receipt_hash()
        if observed != self._bound_receipt_hash:
            _refuse(
                V1SampledFullPopulationCompactSourceRefusalCode.STALE_IDENTITY,
                "full-population compact receipt failed its bound hash",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SampledFullPopulationCompactSourceRefusalCode.STALE_IDENTITY,
                "full-population compact receipt differs from the expected source",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def exact_d1_context_weight(self) -> Fraction:
        return Fraction(*self.d1_context_weight_fraction)

    def exact_residual_context_weight(self, logical_stratum: str) -> Fraction:
        try:
            index = self.canonical_stratum_order.index(logical_stratum)
        except ValueError:
            _refuse(
                V1SampledFullPopulationCompactSourceRefusalCode.AXIS_MISMATCH,
                "logical stratum is not target or complement",
            )
        return Fraction(*self.residual_context_weight_fractions[index])


@dataclass(frozen=True, slots=True)
class V1SampledAnchorInfluencingAxisReceipt:
    """Exact ordered identities which influenced one frozen D-side anchor.

    The receipt is intentionally array-free.  The five supplied D1 axes are
    replayed against the hashes emitted while the transient raw shards were
    still live; the residual suffix is recovered only from the retained v4
    packet registrations.  Consequently the receipt exposes source identity,
    never corrected-A or literal-J row payloads.
    """

    full_population_compact_source_receipt_hash: str
    conditioned_d1_source_receipt_hash: str
    conditioned_d1_device_source_receipt_hash: str
    residual_segmented_source_receipt_hash: str
    packet_registration_receipt_hashes: tuple[str, ...]
    append_packet_registration_axis_hash: str
    residual_measure_design_hash: str
    final_residual_weight_receipt_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    alpha_ledger_id: str
    alpha_coordinate_id: str
    alpha_lineage_hash: str
    conditioned_d1_registered_prefix_hash: str
    registered_d_prefix_hash: str
    d1_pair_prefix_axis_hash: str
    residual_registered_dw2_prefix_hash: str
    final_planned_dw2_prefix_hash: str
    conditioned_d1_context_axis_hash: str
    conditioned_d1_draw_axis_hash: str
    conditioned_d1_measurement_axis_hash: str
    conditioned_d1_capture_axis_hash: str
    conditioned_d1_source_packet_axis_hash: str
    residual_context_axis_hash: str
    residual_draw_axis_hash: str
    residual_measurement_axis_hash: str
    residual_capture_axis_hash: str
    residual_source_packet_axis_hash: str
    context_axis_hash: str
    draw_axis_hash: str
    measurement_axis_hash: str
    capture_axis_hash: str
    source_packet_axis_hash: str
    context_ids: tuple[str, ...]
    draw_ids: tuple[str, ...]
    measurement_ids: tuple[str, ...]
    capture_ids: tuple[str, ...]
    source_packet_hashes: tuple[str, ...]
    stratum_ids: tuple[str, ...]
    source_component_ids: tuple[str, ...]
    packet_context_counts: tuple[int, ...]
    packet_stratum_counts: tuple[tuple[int, int], ...]
    d1_stratum_counts: tuple[int, int]
    residual_fit_stratum_counts: tuple[int, int]
    fit_stratum_counts: tuple[int, int]
    d1_context_count: int
    residual_fit_context_count: int
    context_count: int
    packet_count: int
    canonical_stratum_order: tuple[str, str] = SAMPLED_FULL_POPULATION_CANONICAL_STRATA
    conditioned_d1_target_then_complement_order_replayed: bool = True
    residual_packets_in_registration_order_replayed: bool = True
    conditioned_d1_precedes_residual_fit: bool = True
    all_identity_axes_unique: bool = True
    conditioned_d1_and_residual_fit_axes_disjoint: bool = True
    packet_lineage_bound: bool = True
    proposal_anchor_influencing_axis_complete: bool = True
    later_certification_axis_must_be_disjoint: bool = True
    array_free: bool = True
    tensor_payload_present: bool = False
    raw_corrected_or_literal_rows_exposed: bool = False
    outcome_payload_present: bool = False
    statistical_authority: bool = False
    confidence_authority: bool = False
    scoring_authority: bool = False
    action_authority: bool = False
    solution_sensitivity_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SAMPLED_ANCHOR_INFLUENCING_AXIS_AUTHORITY
    schema: str = SAMPLED_ANCHOR_INFLUENCING_AXIS_SCHEMA
    _issuance_token: object = field(default=None, repr=False, compare=False)
    _bound_receipt_hash: str | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self._issuance_token is not _ISSUER_TOKEN:
            _refuse(
                V1SampledFullPopulationCompactSourceRefusalCode.INVALID_TYPE,
                "anchor-influencing axes must be issued by the live composite handle",
            )
        identity_names = (
            "full_population_compact_source_receipt_hash",
            "conditioned_d1_source_receipt_hash",
            "conditioned_d1_device_source_receipt_hash",
            "residual_segmented_source_receipt_hash",
            "append_packet_registration_axis_hash",
            "residual_measure_design_hash",
            "final_residual_weight_receipt_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "alpha_ledger_id",
            "alpha_coordinate_id",
            "alpha_lineage_hash",
            "conditioned_d1_registered_prefix_hash",
            "registered_d_prefix_hash",
            "d1_pair_prefix_axis_hash",
            "residual_registered_dw2_prefix_hash",
            "final_planned_dw2_prefix_hash",
            "conditioned_d1_context_axis_hash",
            "conditioned_d1_draw_axis_hash",
            "conditioned_d1_measurement_axis_hash",
            "conditioned_d1_capture_axis_hash",
            "conditioned_d1_source_packet_axis_hash",
            "residual_context_axis_hash",
            "residual_draw_axis_hash",
            "residual_measurement_axis_hash",
            "residual_capture_axis_hash",
            "residual_source_packet_axis_hash",
            "context_axis_hash",
            "draw_axis_hash",
            "measurement_axis_hash",
            "capture_axis_hash",
            "source_packet_axis_hash",
        )
        for name in identity_names:
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        packet_hashes = _identity_axis(
            self.packet_registration_receipt_hashes,
            name="packet registration receipt hashes",
        )
        object.__setattr__(self, "packet_registration_receipt_hashes", packet_hashes)
        axes = tuple(
            _ordered_unique_axis(getattr(self, name), name=name)
            for name in (
                "context_ids",
                "draw_ids",
                "measurement_ids",
                "capture_ids",
                "source_packet_hashes",
            )
        )
        for name, axis in zip(
            (
                "context_ids",
                "draw_ids",
                "measurement_ids",
                "capture_ids",
                "source_packet_hashes",
            ),
            axes,
            strict=True,
        ):
            object.__setattr__(self, name, axis)
        strata = _label_axis(self.stratum_ids, name="stratum IDs")
        components = _label_axis(
            self.source_component_ids,
            name="source component IDs",
        )
        object.__setattr__(self, "stratum_ids", strata)
        object.__setattr__(self, "source_component_ids", components)
        proposal_generation = _nonnegative_int(
            self.proposal_generation,
            name="proposal generation",
        )
        object.__setattr__(self, "proposal_generation", proposal_generation)
        for name in (
            "d1_context_count",
            "residual_fit_context_count",
            "context_count",
            "packet_count",
        ):
            value = _nonnegative_int(getattr(self, name), name=name)
            if value <= 0:
                _refuse(
                    V1SampledFullPopulationCompactSourceRefusalCode.AXIS_MISMATCH,
                    f"{name} must be positive",
                )
            object.__setattr__(self, name, value)
        d1_counts = _count_pair(self.d1_stratum_counts, name="D1 stratum counts")
        residual_counts = _count_pair(
            self.residual_fit_stratum_counts,
            name="residual-fit stratum counts",
        )
        fit_counts = _count_pair(self.fit_stratum_counts, name="fit stratum counts")
        if type(self.packet_context_counts) is not tuple:
            _refuse(
                V1SampledFullPopulationCompactSourceRefusalCode.INVALID_TYPE,
                "packet context counts must be immutable",
            )
        packet_context_counts = tuple(
            _nonnegative_int(item, name="packet context count")
            for item in self.packet_context_counts
        )
        if type(self.packet_stratum_counts) is not tuple:
            _refuse(
                V1SampledFullPopulationCompactSourceRefusalCode.INVALID_TYPE,
                "packet stratum counts must be immutable",
            )
        packet_stratum_counts = tuple(
            _count_pair(item, name="packet stratum counts")
            for item in self.packet_stratum_counts
        )
        if any(item <= 0 for item in packet_context_counts) or any(
            min(item) <= 0 for item in packet_stratum_counts
        ):
            _refuse(
                V1SampledFullPopulationCompactSourceRefusalCode.AXIS_MISMATCH,
                "every registered residual-fit packet must cover both strata",
            )
        object.__setattr__(self, "packet_context_counts", packet_context_counts)
        object.__setattr__(self, "packet_stratum_counts", packet_stratum_counts)
        if (
            len({len(item) for item in axes}) != 1
            or len(axes[0]) != len(strata)
            or len(strata) != len(components)
            or self.context_count != len(strata)
            or self.d1_context_count != sum(d1_counts)
            or self.residual_fit_context_count != sum(residual_counts)
            or self.context_count
            != self.d1_context_count + self.residual_fit_context_count
            or fit_counts
            != tuple(d1_counts[index] + residual_counts[index] for index in (0, 1))
            or self.packet_count != len(packet_hashes)
            or self.packet_count != len(packet_context_counts)
            or self.packet_count != len(packet_stratum_counts)
            or sum(packet_context_counts) != self.residual_fit_context_count
            or tuple(
                sum(item[index] for item in packet_stratum_counts) for index in (0, 1)
            )
            != residual_counts
            or any(
                packet_context_counts[index] != sum(packet_stratum_counts[index])
                for index in range(self.packet_count)
            )
        ):
            _refuse(
                V1SampledFullPopulationCompactSourceRefusalCode.AXIS_MISMATCH,
                "D1, packet, stratum, and influencing-axis counts do not close",
            )
        expected_strata = (
            ("target",) * d1_counts[0]
            + ("complement",) * d1_counts[1]
            + tuple(
                stratum
                for counts in packet_stratum_counts
                for stratum in (("target",) * counts[0] + ("complement",) * counts[1])
            )
        )
        expected_components = ("conditioned_D1",) * self.d1_context_count + (
            "residual_fit",
        ) * self.residual_fit_context_count
        if strata != expected_strata or components != expected_components:
            _refuse(
                V1SampledFullPopulationCompactSourceRefusalCode.ORDER_MISMATCH,
                "influencing contexts are not D1 target/complement then residual packet order",
            )
        d1_stop = self.d1_context_count
        role_fields = (
            (
                "context",
                self.conditioned_d1_context_axis_hash,
                self.residual_context_axis_hash,
                self.context_axis_hash,
            ),
            (
                "draw",
                self.conditioned_d1_draw_axis_hash,
                self.residual_draw_axis_hash,
                self.draw_axis_hash,
            ),
            (
                "measurement",
                self.conditioned_d1_measurement_axis_hash,
                self.residual_measurement_axis_hash,
                self.measurement_axis_hash,
            ),
            (
                "capture",
                self.conditioned_d1_capture_axis_hash,
                self.residual_capture_axis_hash,
                self.capture_axis_hash,
            ),
            (
                "source_packet",
                self.conditioned_d1_source_packet_axis_hash,
                self.residual_source_packet_axis_hash,
                self.source_packet_axis_hash,
            ),
        )
        for axis, (role, d1_hash, residual_hash, combined_hash) in zip(
            axes,
            role_fields,
            strict=True,
        ):
            if (
                _conditioned_d1_axis_hash(role, axis[:d1_stop]) != d1_hash
                or _registered_axis_hash(role, axis[d1_stop:]) != residual_hash
                or _influencing_axis_hash(role, axis) != combined_hash
            ):
                _refuse(
                    V1SampledFullPopulationCompactSourceRefusalCode.STALE_IDENTITY,
                    f"{role} influencing axis does not replay its source hashes",
                )
        if (
            _conditioned_d1_pair_axis_hash(
                target=self.context_ids[: d1_counts[0]],
                complement=self.context_ids[d1_counts[0] : d1_stop],
            )
            != self.d1_pair_prefix_axis_hash
        ):
            _refuse(
                V1SampledFullPopulationCompactSourceRefusalCode.ORDER_MISMATCH,
                "D1 context order does not replay the source target/complement prefix",
            )
        required_true = (
            self.conditioned_d1_target_then_complement_order_replayed,
            self.residual_packets_in_registration_order_replayed,
            self.conditioned_d1_precedes_residual_fit,
            self.all_identity_axes_unique,
            self.conditioned_d1_and_residual_fit_axes_disjoint,
            self.packet_lineage_bound,
            self.proposal_anchor_influencing_axis_complete,
            self.later_certification_axis_must_be_disjoint,
            self.array_free,
        )
        required_false = (
            self.tensor_payload_present,
            self.raw_corrected_or_literal_rows_exposed,
            self.outcome_payload_present,
            self.statistical_authority,
            self.confidence_authority,
            self.scoring_authority,
            self.action_authority,
            self.solution_sensitivity_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if (
            self.canonical_stratum_order != SAMPLED_FULL_POPULATION_CANONICAL_STRATA
            or any(type(item) is not bool for item in required_true + required_false)
            or any(item is not True for item in required_true)
            or any(item is not False for item in required_false)
            or self.authority_scope != SAMPLED_ANCHOR_INFLUENCING_AXIS_AUTHORITY
            or self.schema != SAMPLED_ANCHOR_INFLUENCING_AXIS_SCHEMA
        ):
            _refuse(
                V1SampledFullPopulationCompactSourceRefusalCode.AUTHORITY_VIOLATION,
                "anchor-influencing receipt crossed its array-free custody boundary",
            )
        object.__setattr__(self, "d1_stratum_counts", d1_counts)
        object.__setattr__(self, "residual_fit_stratum_counts", residual_counts)
        object.__setattr__(self, "fit_stratum_counts", fit_counts)
        observed = self._compute_receipt_hash()
        if (
            self._bound_receipt_hash is not None
            and observed != self._bound_receipt_hash
        ):
            _refuse(
                V1SampledFullPopulationCompactSourceRefusalCode.STALE_IDENTITY,
                "anchor-influencing receipt changed after issuance",
            )
        object.__setattr__(self, "_bound_receipt_hash", observed)

    def _compute_receipt_hash(self) -> str:
        return content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            },
            prefix=_PREFIX + "anchor-influencing-receipt-",
        )

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_receipt_hash()
        if observed != self._bound_receipt_hash or (
            expected_hash is not None and observed != expected_hash
        ):
            _refuse(
                V1SampledFullPopulationCompactSourceRefusalCode.STALE_IDENTITY,
                "anchor-influencing receipt failed its bound source hash",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


class V1SampledFullPopulationCompactSourceLease:
    """One locked borrow spanning both compact source owners."""

    __slots__ = (
        "_parent",
        "_d1_lease",
        "_residual_lease",
        "_closed",
        "source_receipt_hash",
        "borrow_purpose_hash",
        "lease_hash",
        "source_custody_only",
        "action_authority",
        "confidence_authority",
        "c_stream_authority",
        "calibration_authority",
        "oracle_authority",
        "commit_authority",
        "terminal_authority",
        "pareto_authority",
        "schema",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        parent: "V1SampledFullPopulationCompactSourceHandle",
        d1_lease: V1SignedWorkVariableShellSourceTensorLease,
        residual_lease: V1SignedWorkVariableShellSourceAppendSegmentLease,
        borrow_purpose_hash: str,
        _issuance_token: object,
    ) -> None:
        if (
            _issuance_token is not _ISSUER_TOKEN
            or type(parent) is not V1SampledFullPopulationCompactSourceHandle
            or type(d1_lease) is not V1SignedWorkVariableShellSourceTensorLease
            or type(residual_lease)
            is not V1SignedWorkVariableShellSourceAppendSegmentLease
        ):
            _refuse(
                V1SampledFullPopulationCompactSourceRefusalCode.INVALID_TYPE,
                "full-population leases must be atomically owner-issued",
            )
        purpose = _identity(borrow_purpose_hash, name="borrow purpose hash")
        self._parent = parent
        self._d1_lease = d1_lease
        self._residual_lease = residual_lease
        self._closed = False
        self.source_receipt_hash = parent.receipt.receipt_hash
        self.borrow_purpose_hash = purpose
        self.source_custody_only = True
        self.action_authority = False
        self.confidence_authority = False
        self.c_stream_authority = False
        self.calibration_authority = False
        self.oracle_authority = False
        self.commit_authority = False
        self.terminal_authority = False
        self.pareto_authority = False
        self.schema = SAMPLED_FULL_POPULATION_COMPACT_SOURCE_LEASE_SCHEMA
        self.lease_hash = content_hash(
            {
                "schema": self.schema,
                "source_receipt_hash": self.source_receipt_hash,
                "borrow_purpose_hash": purpose,
                "d1_tensor_lease_hash": d1_lease.lease_hash,
                "residual_segment_lease_hash": residual_lease.lease_hash,
                "component_tensors_merged": False,
                "all_downstream_authority": False,
            },
            prefix=_PREFIX + "lease-",
        )

    @property
    def d1_tensor_lease(self) -> V1SignedWorkVariableShellSourceTensorLease:
        self._require_live()
        return self._d1_lease

    @property
    def residual_segment_lease(
        self,
    ) -> V1SignedWorkVariableShellSourceAppendSegmentLease:
        self._require_live()
        return self._residual_lease

    def _require_live(self) -> None:
        if self._closed:
            _refuse(
                V1SampledFullPopulationCompactSourceRefusalCode.CUSTODY_MISMATCH,
                "full-population source lease was closed",
            )
        self._parent.receipt.assert_integrity(self.source_receipt_hash)

    def d1_compact_tensors(self) -> tuple[object, ...]:
        self._require_live()
        return self._d1_lease.compact_tensors()

    def residual_compact_segments(self) -> tuple[object, ...]:
        self._require_live()
        return self._residual_lease.compact_segments()

    def close(self) -> None:
        if self._closed:
            return
        failure: Exception | None = None
        try:
            self._d1_lease.close()
        except Exception as error:  # retain the first custody failure
            failure = error
        try:
            self._residual_lease.close()
        except Exception as error:
            if failure is None:
                failure = error
        self._closed = True
        if failure is not None:
            raise failure

    def __enter__(self) -> "V1SampledFullPopulationCompactSourceLease":
        self.d1_compact_tensors()
        self.residual_compact_segments()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        del exc_type, exc, traceback
        self.close()
        return False

    def __reduce__(self) -> NoReturn:
        raise TypeError("full-population compact source leases are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("full-population compact source leases are nonserializable")


class V1SampledFullPopulationCompactSourceHandle:
    """Nonserializable snapshot retaining the two native compact sources."""

    __slots__ = (
        "_conditioned_d1",
        "_residual",
        "_packet_receipts",
        "_measure_design",
        "_weight_receipt",
        "_common_kkt_bridge",
        "_receipt",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        conditioned_d1: V1SampledConditionedD1CompactSourceResult,
        residual: V1SignedWorkVariableShellSourceAppendSegmentedHandle,
        packet_receipts: tuple[V1SampledVariableShellPacketRegistrationReceipt, ...],
        measure_design: V1SampledResidualPopulationMeasureDesign,
        weight_receipt: V1SampledResidualPopulationWeightReceipt,
        common_kkt_bridge: V1SampledV7CommonKKTCoordinateBridgeReceipt | None,
        receipt: V1SampledFullPopulationCompactSourceReceipt,
        _issuance_token: object,
    ) -> None:
        if (
            _issuance_token is not _ISSUER_TOKEN
            or type(conditioned_d1) is not V1SampledConditionedD1CompactSourceResult
            or type(residual)
            is not V1SignedWorkVariableShellSourceAppendSegmentedHandle
            or type(receipt) is not V1SampledFullPopulationCompactSourceReceipt
        ):
            _refuse(
                V1SampledFullPopulationCompactSourceRefusalCode.INVALID_TYPE,
                "full-population handles must be compositor-issued",
            )
        self._conditioned_d1 = conditioned_d1
        self._residual = residual
        self._packet_receipts = packet_receipts
        self._measure_design = measure_design
        self._weight_receipt = weight_receipt
        self._common_kkt_bridge = common_kkt_bridge
        self._receipt = receipt
        self._validate_live_components()

    def _validate_live_components(self) -> None:
        d1 = self._conditioned_d1
        residual = self._residual
        receipt = self._receipt
        bridge = self._common_kkt_bridge
        try:
            d1_device_receipt_hash = d1.source_handle.receipt.receipt_hash
            residual_receipt_hash = residual.receipt.receipt_hash
        except (
            V1SignedWorkVariableShellSourceAppendRefusal,
            V1SignedWorkVariableShellSourceRefusal,
        ) as error:
            raise V1SampledFullPopulationCompactSourceRefusal(
                V1SampledFullPopulationCompactSourceRefusalCode.STALE_IDENTITY,
                "one retained compact owner was evicted, mutated, or extended",
            ) from error
        if (
            d1.receipt.receipt_hash != receipt.conditioned_d1_source_receipt_hash
            or d1_device_receipt_hash
            != receipt.conditioned_d1_device_source_receipt_hash
            or residual_receipt_hash != receipt.residual_segmented_source_receipt_hash
            or self._measure_design.design_hash != receipt.residual_measure_design_hash
            or self._weight_receipt.receipt_hash
            != receipt.final_residual_weight_receipt_hash
            or tuple(item.receipt_hash for item in self._packet_receipts)
            != receipt.packet_registration_receipt_hashes
            or (
                bridge is None
                and receipt.common_kkt_coordinate_bridge_receipt_hash is not None
            )
            or (
                bridge is not None
                and (
                    type(bridge) is not V1SampledV7CommonKKTCoordinateBridgeReceipt
                    or bridge.assert_composition(
                        conditioned_d1=d1,
                        residual_segmented_source=residual,
                    )
                    != receipt.common_kkt_coordinate_bridge_receipt_hash
                )
            )
        ):
            _refuse(
                V1SampledFullPopulationCompactSourceRefusalCode.STALE_IDENTITY,
                "one retained component changed after the full-population snapshot",
            )
        receipt.assert_integrity()

    @property
    def receipt(self) -> V1SampledFullPopulationCompactSourceReceipt:
        self._validate_live_components()
        return self._receipt

    @property
    def conditioned_d1_source(self) -> V1SampledConditionedD1CompactSourceResult:
        self._validate_live_components()
        return self._conditioned_d1

    @property
    def residual_segmented_source(
        self,
    ) -> V1SignedWorkVariableShellSourceAppendSegmentedHandle:
        self._validate_live_components()
        return self._residual

    def issue_anchor_influencing_axis_receipt(
        self,
        *,
        conditioned_d1_context_ids: Sequence[str],
        conditioned_d1_draw_ids: Sequence[str],
        conditioned_d1_measurement_ids: Sequence[str],
        conditioned_d1_capture_ids: Sequence[str],
        conditioned_d1_source_packet_hashes: Sequence[str],
    ) -> V1SampledAnchorInfluencingAxisReceipt:
        """Issue the exact ordered ``D_fixed`` fit/influencing identity axis.

        Conditioned-D1 raw shards are intentionally transient, so their five
        source axes are supplied by the source owner and replayed against the
        durable D1 hashes.  The residual axes are not caller-supplied: they are
        recovered from every retained v4 packet receipt in registration order.
        """

        try:
            self._validate_live_components()
        except V1SampledFullPopulationCompactSourceRefusal:
            raise
        except RuntimeError as error:
            raise V1SampledFullPopulationCompactSourceRefusal(
                V1SampledFullPopulationCompactSourceRefusalCode.STALE_IDENTITY,
                "a retained packet lineage changed before influencing-axis issuance",
            ) from error
        composite = self._receipt
        d1 = self._conditioned_d1.receipt
        residual = self._residual.receipt
        d1_axes = tuple(
            _ordered_unique_axis(value, name=name)
            for value, name in (
                (conditioned_d1_context_ids, "conditioned-D1 context IDs"),
                (conditioned_d1_draw_ids, "conditioned-D1 draw IDs"),
                (
                    conditioned_d1_measurement_ids,
                    "conditioned-D1 measurement IDs",
                ),
                (conditioned_d1_capture_ids, "conditioned-D1 capture IDs"),
                (
                    conditioned_d1_source_packet_hashes,
                    "conditioned-D1 source-packet hashes",
                ),
            )
        )
        if len({len(item) for item in d1_axes}) != 1 or len(d1_axes[0]) != (
            composite.d1_context_count
        ):
            _refuse(
                V1SampledFullPopulationCompactSourceRefusalCode.AXIS_MISMATCH,
                "conditioned-D1 influencing axes do not cover the exact D1 prefix",
            )
        packets = self._packet_receipts
        for ordinal, packet in enumerate(packets):
            try:
                packet.assert_integrity()
            except RuntimeError as error:
                raise V1SampledFullPopulationCompactSourceRefusal(
                    V1SampledFullPopulationCompactSourceRefusalCode.STALE_IDENTITY,
                    "one retained v4 packet registration changed before axis issuance",
                ) from error
            if (
                packet.packet_ordinal != ordinal
                or packet.receipt_hash
                != composite.packet_registration_receipt_hashes[ordinal]
            ):
                _refuse(
                    V1SampledFullPopulationCompactSourceRefusalCode.ORDER_MISMATCH,
                    "v4 packet lineage is omitted, reordered, or substituted",
                )
        residual_axes = tuple(
            tuple(value for packet in packets for value in getattr(packet, field_name))
            for field_name in (
                "registered_context_ids",
                "registered_draw_ids",
                "registered_measurement_ids",
                "registered_capture_ids",
                "registered_source_packet_hashes",
            )
        )
        residual_hashes = (
            residual.context_axis_hash,
            residual.draw_axis_hash,
            residual.measurement_axis_hash,
            residual.capture_axis_hash,
            residual.source_packet_axis_hash,
        )
        for role, axis, expected_hash in zip(
            ("context", "draw", "measurement", "capture", "source_packet"),
            residual_axes,
            residual_hashes,
            strict=True,
        ):
            if (
                len(axis) != composite.residual_context_count
                or len(axis) != len(set(axis))
                or _registered_axis_hash(role, axis) != expected_hash
            ):
                _refuse(
                    V1SampledFullPopulationCompactSourceRefusalCode.ORDER_MISMATCH,
                    f"residual {role} axis omits, reorders, duplicates, or substitutes a v4 row",
                )
        combined_axes = tuple(
            d1_axis + residual_axis
            for d1_axis, residual_axis in zip(d1_axes, residual_axes, strict=True)
        )
        packet_stratum_counts = tuple(
            packet.packet_stratum_counts for packet in packets
        )
        residual_counts = tuple(
            sum(item[index] for item in packet_stratum_counts) for index in (0, 1)
        )
        if residual_counts != composite.final_dw2_stratum_counts:
            _refuse(
                V1SampledFullPopulationCompactSourceRefusalCode.ORDER_MISMATCH,
                "v4 packet strata do not replay the composite residual-fit prefix",
            )
        d1_strata = ("target",) * composite.d1_stratum_counts[0] + (
            "complement",
        ) * composite.d1_stratum_counts[1]
        residual_strata = tuple(
            value for packet in packets for value in packet.registered_stratum_ids
        )
        return V1SampledAnchorInfluencingAxisReceipt(
            full_population_compact_source_receipt_hash=composite.receipt_hash,
            conditioned_d1_source_receipt_hash=(
                composite.conditioned_d1_source_receipt_hash
            ),
            conditioned_d1_device_source_receipt_hash=(
                composite.conditioned_d1_device_source_receipt_hash
            ),
            residual_segmented_source_receipt_hash=(
                composite.residual_segmented_source_receipt_hash
            ),
            packet_registration_receipt_hashes=(
                composite.packet_registration_receipt_hashes
            ),
            append_packet_registration_axis_hash=(
                composite.append_packet_registration_axis_hash
            ),
            residual_measure_design_hash=composite.residual_measure_design_hash,
            final_residual_weight_receipt_hash=(
                composite.final_residual_weight_receipt_hash
            ),
            model_identity_hash=composite.model_identity_hash,
            chart_identity_hash=composite.chart_identity_hash,
            current_mask_hash=composite.current_mask_hash,
            proposal_generation=composite.proposal_generation,
            alpha_ledger_id=composite.alpha_ledger_id,
            alpha_coordinate_id=composite.alpha_coordinate_id,
            alpha_lineage_hash=composite.alpha_lineage_hash,
            conditioned_d1_registered_prefix_hash=(
                composite.conditioned_d1_registered_prefix_hash
            ),
            registered_d_prefix_hash=composite.registered_d_prefix_hash,
            d1_pair_prefix_axis_hash=composite.d1_pair_prefix_axis_hash,
            residual_registered_dw2_prefix_hash=(
                composite.residual_registered_dw2_prefix_hash
            ),
            final_planned_dw2_prefix_hash=(composite.final_planned_dw2_prefix_hash),
            conditioned_d1_context_axis_hash=d1.canonical_context_axis_hash,
            conditioned_d1_draw_axis_hash=d1.canonical_draw_axis_hash,
            conditioned_d1_measurement_axis_hash=(d1.canonical_measurement_axis_hash),
            conditioned_d1_capture_axis_hash=d1.canonical_capture_axis_hash,
            conditioned_d1_source_packet_axis_hash=(
                d1.canonical_source_packet_axis_hash
            ),
            residual_context_axis_hash=residual.context_axis_hash,
            residual_draw_axis_hash=residual.draw_axis_hash,
            residual_measurement_axis_hash=residual.measurement_axis_hash,
            residual_capture_axis_hash=residual.capture_axis_hash,
            residual_source_packet_axis_hash=residual.source_packet_axis_hash,
            context_axis_hash=_influencing_axis_hash("context", combined_axes[0]),
            draw_axis_hash=_influencing_axis_hash("draw", combined_axes[1]),
            measurement_axis_hash=_influencing_axis_hash(
                "measurement",
                combined_axes[2],
            ),
            capture_axis_hash=_influencing_axis_hash("capture", combined_axes[3]),
            source_packet_axis_hash=_influencing_axis_hash(
                "source_packet",
                combined_axes[4],
            ),
            context_ids=combined_axes[0],
            draw_ids=combined_axes[1],
            measurement_ids=combined_axes[2],
            capture_ids=combined_axes[3],
            source_packet_hashes=combined_axes[4],
            stratum_ids=d1_strata + residual_strata,
            source_component_ids=("conditioned_D1",) * composite.d1_context_count
            + ("residual_fit",) * composite.residual_context_count,
            packet_context_counts=tuple(
                len(packet.registered_context_ids) for packet in packets
            ),
            packet_stratum_counts=packet_stratum_counts,
            d1_stratum_counts=composite.d1_stratum_counts,
            residual_fit_stratum_counts=residual_counts,  # type: ignore[arg-type]
            fit_stratum_counts=tuple(
                composite.d1_stratum_counts[index] + residual_counts[index]
                for index in (0, 1)
            ),  # type: ignore[arg-type]
            d1_context_count=composite.d1_context_count,
            residual_fit_context_count=composite.residual_context_count,
            context_count=composite.context_count,
            packet_count=len(packets),
            _issuance_token=_ISSUER_TOKEN,
        )

    def borrow_components(
        self,
        *,
        borrow_purpose_hash: str,
    ) -> V1SampledFullPopulationCompactSourceLease:
        """Lock both component owners or leave neither owner borrowed."""

        purpose = _identity(borrow_purpose_hash, name="borrow purpose hash")
        receipt = self.receipt
        residual_purpose = content_hash(
            {
                "full_population_source_receipt_hash": receipt.receipt_hash,
                "borrow_purpose_hash": purpose,
                "component": "sampled_residual",
            },
            prefix=_PREFIX + "component-purpose-",
        )
        d1_purpose = content_hash(
            {
                "full_population_source_receipt_hash": receipt.receipt_hash,
                "borrow_purpose_hash": purpose,
                "component": "conditioned_D1",
            },
            prefix=_PREFIX + "component-purpose-",
        )
        residual_lease = self._residual.borrow_compact_segments(
            borrow_purpose_hash=residual_purpose
        )
        try:
            d1_lease = self._conditioned_d1.source_handle.borrow_compact_tensors(
                borrow_purpose_hash=d1_purpose
            )
        except Exception:
            residual_lease.close()
            raise
        try:
            return V1SampledFullPopulationCompactSourceLease(
                parent=self,
                d1_lease=d1_lease,
                residual_lease=residual_lease,
                borrow_purpose_hash=purpose,
                _issuance_token=_ISSUER_TOKEN,
            )
        except Exception:
            try:
                d1_lease.close()
            finally:
                residual_lease.close()
            raise

    def __reduce__(self) -> NoReturn:
        raise TypeError("full-population compact source handles are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("full-population compact source handles are nonserializable")


def compose_v1_sampled_full_population_compact_source(
    *,
    conditioned_d1: V1SampledConditionedD1CompactSourceResult,
    residual_segmented_source: V1SignedWorkVariableShellSourceAppendSegmentedHandle,
    packet_registration_receipts: Sequence[
        V1SampledVariableShellPacketRegistrationReceipt
    ],
    residual_measure_design: V1SampledResidualPopulationMeasureDesign,
    final_weight_receipt: V1SampledResidualPopulationWeightReceipt,
    common_kkt_coordinate_bridge: (
        V1SampledV7CommonKKTCoordinateBridgeReceipt | None
    ) = None,
) -> V1SampledFullPopulationCompactSourceHandle:
    """Bind one immutable D1-plus-residual source snapshot and exact measure."""

    if type(conditioned_d1) is not V1SampledConditionedD1CompactSourceResult:
        raise TypeError("full-population composition requires the exact D1 result")
    if (
        type(residual_segmented_source)
        is not V1SignedWorkVariableShellSourceAppendSegmentedHandle
    ):
        raise TypeError("full-population composition requires the segmented source")
    if type(residual_measure_design) is not V1SampledResidualPopulationMeasureDesign:
        raise TypeError("full-population composition requires the exact measure design")
    if type(final_weight_receipt) is not V1SampledResidualPopulationWeightReceipt:
        raise TypeError("full-population composition requires the exact weight receipt")
    if (
        common_kkt_coordinate_bridge is not None
        and type(common_kkt_coordinate_bridge)
        is not V1SampledV7CommonKKTCoordinateBridgeReceipt
    ):
        raise TypeError("common KKT coordinate bridge must be exact when supplied")
    if type(packet_registration_receipts) not in {tuple, list}:
        raise TypeError("packet receipts must be one ordered sequence")
    packets = tuple(packet_registration_receipts)
    if not packets or any(
        type(item) is not V1SampledVariableShellPacketRegistrationReceipt
        for item in packets
    ):
        _refuse(
            V1SampledFullPopulationCompactSourceRefusalCode.INVALID_TYPE,
            "the final prefix needs one or more exact v4 packet receipts",
        )
    if any(
        item.schema != SAMPLED_VARIABLE_SHELL_PACKET_REGISTRATION_RECEIPT_SCHEMA
        for item in packets
    ):
        _refuse(
            V1SampledFullPopulationCompactSourceRefusalCode.INVALID_TYPE,
            "packet receipts do not use the canonical v4 registration schema",
        )
    for item in packets:
        item.assert_integrity()
    residual_measure_design.assert_integrity()
    final_weight_receipt.assert_integrity()
    d1_receipt = conditioned_d1.receipt
    d1_receipt.assert_integrity()
    d1_device = conditioned_d1.source_handle.receipt
    residual = residual_segmented_source.receipt
    if common_kkt_coordinate_bridge is not None:
        common_kkt_coordinate_bridge.assert_composition(
            conditioned_d1=conditioned_d1,
            residual_segmented_source=residual_segmented_source,
        )

    lineage_hash = packets[0].append_lineage_hash
    expected_packet_axis_hash = _append_packet_axis_hash(
        lineage_hash=lineage_hash,
        receipts=packets,
    )
    previous_counts = (0, 0)
    seen_contexts: set[str] = set()
    for ordinal, packet in enumerate(packets):
        expected_counts = tuple(
            previous_counts[index] + packet.packet_stratum_counts[index]
            for index in (0, 1)
        )
        if (
            packet.packet_ordinal != ordinal
            or packet.append_lineage_hash != lineage_hash
            or packet.after_suffix_stratum_counts != expected_counts
            or any(item in seen_contexts for item in packet.registered_context_ids)
        ):
            _refuse(
                V1SampledFullPopulationCompactSourceRefusalCode.ORDER_MISMATCH,
                "packet receipts are reordered, overlapping, or skip a cumulative prefix",
            )
        previous_counts = expected_counts
        seen_contexts.update(packet.registered_context_ids)

    final_packet = packets[-1]
    final_counts = final_weight_receipt.cumulative_dw2_stratum_counts
    shared_packet_identity_names = (
        "suffix_registration_hash",
        "generation_frame_hash",
        "discovery_stream_hash",
        "model_identity_hash",
        "chart_identity_hash",
        "current_mask_hash",
        "proposal_generation",
        "route_axis_hash",
        "route_id_axis_hash",
        "native_fiber_axis_receipt_hash",
        "native_fiber_measure_hash",
        "shell_union_receipt_hash",
        "same_capture_vjp_source_hash",
        "corrected_evidence_stream_source_hash",
        "literal_j_stream_source_hash",
        "full_population_stratum_counts",
        "consumed_d1_stratum_counts",
        "residual_stratum_capacity_counts",
        "full_population_count",
        "residual_population_count",
        "residual_population_measure_design_hash",
    )
    if any(
        getattr(packet, name) != getattr(final_packet, name)
        for packet in packets
        for name in shared_packet_identity_names
    ):
        _refuse(
            V1SampledFullPopulationCompactSourceRefusalCode.STALE_IDENTITY,
            "packet receipts do not share one chart, shell, geometry, and measure graph",
        )
    component_shared = (
        (d1_device.model_identity_hash, residual.model_identity_hash),
        (d1_receipt.chart_identity_hash, residual.chart_identity_hash),
        (d1_receipt.current_mask_hash, residual.current_mask_hash),
        (d1_receipt.proposal_generation, residual.proposal_generation),
        (d1_receipt.route_axis_hash, residual.route_axis_hash),
        (d1_receipt.route_id_axis_hash, residual.route_id_axis_hash),
        (d1_device.shell_freeze_receipt_hash, residual.shell_freeze_receipt_hash),
        (d1_device.shell_indices, residual.shell_indices),
        (d1_device.shell_route_ids, residual.shell_route_ids),
        (d1_device.shell_native_fiber_indices, residual.shell_native_fiber_indices),
        (
            d1_receipt.native_fiber_axis_receipt_hash,
            residual.native_fiber_axis_receipt_hash,
        ),
        (d1_receipt.native_fiber_measure_hash, residual.native_fiber_measure_hash),
        (
            d1_receipt.corrected_evidence_stream_source_hash,
            residual.corrected_evidence_stream_source_hash,
        ),
        (
            d1_receipt.literal_j_stream_source_hash,
            residual.literal_j_stream_source_hash,
        ),
        (d1_device.route_count, residual.route_count),
        (d1_device.fiber_count, residual.fiber_count),
        (d1_device.cut_count, residual.cut_count),
        (
            d1_device.kkt_branch_source_receipt_hash,
            residual.kkt_branch_source_receipt_hash,
        ),
        (d1_device.resource_axis_hash, residual.resource_axis_hash),
        (
            d1_device.resource_source_receipt_hash,
            residual.resource_source_receipt_hash,
        ),
        (d1_device.alpha_ledger_id, residual.alpha_ledger_id),
        (d1_device.alpha_coordinate_id, residual.alpha_coordinate_id),
        (d1_device.alpha_lineage_hash, residual.alpha_lineage_hash),
    )
    packet_to_residual = (
        (final_packet.model_identity_hash, residual.model_identity_hash),
        (final_packet.chart_identity_hash, residual.chart_identity_hash),
        (final_packet.current_mask_hash, residual.current_mask_hash),
        (final_packet.proposal_generation, residual.proposal_generation),
        (final_packet.route_axis_hash, residual.route_axis_hash),
        (final_packet.route_id_axis_hash, residual.route_id_axis_hash),
        (
            final_packet.native_fiber_axis_receipt_hash,
            residual.native_fiber_axis_receipt_hash,
        ),
        (final_packet.native_fiber_measure_hash, residual.native_fiber_measure_hash),
        (final_packet.shell_union_receipt_hash, residual.shell_freeze_receipt_hash),
        (
            final_packet.same_capture_vjp_source_hash,
            residual.same_capture_vjp_source_hash,
        ),
        (
            final_packet.corrected_evidence_stream_source_hash,
            residual.corrected_evidence_stream_source_hash,
        ),
        (
            final_packet.literal_j_stream_source_hash,
            residual.literal_j_stream_source_hash,
        ),
    )
    kkt_identity_closed = d1_device.kkt_branch_id == residual.kkt_branch_id
    if not kkt_identity_closed and common_kkt_coordinate_bridge is not None:
        common_kkt_coordinate_bridge.assert_composition(
            conditioned_d1=conditioned_d1,
            residual_segmented_source=residual_segmented_source,
        )
        kkt_identity_closed = True
    if (
        not kkt_identity_closed
        or any(left != right for left, right in component_shared + packet_to_residual)
    ):
        _refuse(
            V1SampledFullPopulationCompactSourceRefusalCode.CUSTODY_MISMATCH,
            "D1 and residual components do not share one route/shell/native-fiber dual chart",
        )

    design = residual_measure_design
    weight = final_weight_receipt
    count_and_measure_checks = (
        d1_receipt.residual_measure_design_hash == design.design_hash,
        d1_receipt.residual_weight_receipt_hash == weight.receipt_hash,
        d1_receipt.d1_pair_prefix_axis_hash == design.d1_pair_prefix_axis_hash,
        d1_receipt.population_stratum_counts == design.population_stratum_counts,
        d1_receipt.d1_stratum_counts == design.d1_stratum_counts,
        d1_receipt.population_count == design.population_count,
        d1_device.registered_d_w2_prefix_hash == design.d1_pair_prefix_axis_hash,
        residual.registered_d_prefix_hash == design.d1_pair_prefix_axis_hash,
        residual.population_count == sum(design.residual_stratum_counts),
        residual.context_count == sum(final_counts),
        residual.segment_count == len(packets),
        residual.packet_registration_axis_hash == expected_packet_axis_hash,
        final_counts == previous_counts,
        final_counts == final_packet.after_suffix_stratum_counts,
        weight.measure_design_hash == design.design_hash,
        weight.suffix_registration_hash == design.suffix_registration_hash,
        weight.packet_hash == final_packet.packet_hash,
        weight.packet_ordinal == final_packet.packet_ordinal,
        final_packet.residual_population_measure_design_hash == design.design_hash,
        final_packet.residual_population_weight_receipt_hash == weight.receipt_hash,
        final_packet.full_population_stratum_counts == design.population_stratum_counts,
        final_packet.consumed_d1_stratum_counts == design.d1_stratum_counts,
        final_packet.residual_stratum_capacity_counts == design.residual_stratum_counts,
        final_packet.full_population_count == design.population_count,
        final_packet.d1_context_weight_fraction == weight.d1_context_weight_fraction,
        final_packet.cumulative_dw2_context_weight_fractions
        == weight.dw2_context_weight_fractions,
        final_packet.full_population_measure_common_denominator
        == weight.common_denominator,
        final_packet.d1_context_measure_quantum == weight.d1_context_quantum,
        final_packet.cumulative_dw2_context_measure_quanta == weight.dw2_context_quanta,
        final_packet.full_population_total_measure_quanta == weight.total_weight_quanta,
    )
    if not all(count_and_measure_checks):
        _refuse(
            V1SampledFullPopulationCompactSourceRefusalCode.WEIGHT_MISMATCH,
            "component counts, final prefix, or exact rational measure do not close",
        )
    _validate_source_enclosure(
        value=Fraction(*weight.d1_context_weight_fraction),
        enclosure=weight.d1_context_weight_binary64_enclosure,
        name="D1 context weight enclosure",
    )
    for exact, enclosure in zip(
        weight.dw2_context_weight_fractions,
        weight.dw2_context_weight_binary64_enclosures,
        strict=True,
    ):
        _validate_source_enclosure(
            value=Fraction(*exact),
            enclosure=enclosure,
            name="residual context weight enclosure",
        )

    d1_center, d1_radius = _canonical_center_radius(
        Fraction(*weight.d1_context_weight_fraction)
    )
    residual_center_radii = tuple(
        _canonical_center_radius(Fraction(*item))
        for item in weight.dw2_context_weight_fractions
    )
    logical_hash = _logical_stratum_axis_hash(packets)
    composite_same_capture_hash = content_hash(
        {
            "schema": "partition_free_v1_sampled_full_population_same_capture_dual_source_v1",
            "conditioned_d1_same_capture_vjp_source_hash": (
                d1_device.same_capture_vjp_source_hash
            ),
            "residual_same_capture_vjp_source_hash": (
                residual.same_capture_vjp_source_hash
            ),
            "corrected_evidence_stream_source_hash": (
                residual.corrected_evidence_stream_source_hash
            ),
            "literal_j_stream_source_hash": residual.literal_j_stream_source_hash,
            "same_capture_is_component_local_not_cross_component": True,
        },
        prefix=_PREFIX + "same-capture-dual-source-",
    )
    composite_witness_stream_hash = content_hash(
        {
            "schema": "partition_free_v1_sampled_full_population_witness_stream_v1",
            "conditioned_d1_witness_stream_source_hash": (
                d1_device.witness_stream_source_hash
            ),
            "residual_witness_stream_source_hash": residual.witness_stream_source_hash,
            "residual_measure_design_hash": design.design_hash,
            "final_weight_receipt_hash": weight.receipt_hash,
        },
        prefix=_PREFIX + "witness-stream-",
    )
    composite_design_hash = content_hash(
        {
            "schema": "partition_free_v1_sampled_full_population_compact_design_v1",
            "conditioned_d1_design_hash": d1_device.design_hash,
            "residual_design_hash": residual.design_hash,
            "two_component_compaction_without_tensor_merge": True,
        },
        prefix=_PREFIX + "design-",
    )
    composite_registered_d_prefix_hash = content_hash(
        {
            "schema": "partition_free_v1_sampled_full_population_registered_D_prefix_v1",
            "conditioned_d1_registered_prefix_hash": (
                d1_receipt.registered_d_prefix_hash
            ),
            "d1_pair_prefix_axis_hash": design.d1_pair_prefix_axis_hash,
            "final_planned_dw2_prefix_hash": final_packet.planned_d_w2_prefix_hash,
            "residual_registered_dw2_prefix_hash": (
                residual.registered_d_w2_prefix_hash
            ),
            "final_dw2_stratum_counts": final_counts,
        },
        prefix=_PREFIX + "registered-D-prefix-",
    )
    composite_population_context_axis_hash = content_hash(
        {
            "schema": "partition_free_v1_sampled_full_population_context_axis_v1",
            "conditioned_d1_context_axis_hash": (
                d1_receipt.canonical_context_axis_hash
            ),
            "sampled_residual_context_axis_hash": residual.context_axis_hash,
            "logical_stratum_axis_hash": logical_hash,
            "final_dw2_stratum_counts": final_counts,
        },
        prefix=_PREFIX + "population-context-axis-",
    )
    composite_population_draw_axis_hash = content_hash(
        {
            "schema": "partition_free_v1_sampled_full_population_draw_axis_v1",
            "conditioned_d1_draw_axis_hash": d1_receipt.canonical_draw_axis_hash,
            "sampled_residual_draw_axis_hash": residual.draw_axis_hash,
            "logical_stratum_axis_hash": logical_hash,
            "final_dw2_stratum_counts": final_counts,
        },
        prefix=_PREFIX + "population-draw-axis-",
    )
    composite_measurement_axis_hash = content_hash(
        {
            "schema": "partition_free_v1_sampled_full_population_measurement_axis_v1",
            "conditioned_d1_measurement_axis_hash": (
                d1_receipt.canonical_measurement_axis_hash
            ),
            "sampled_residual_measurement_axis_hash": residual.measurement_axis_hash,
            "logical_stratum_axis_hash": logical_hash,
        },
        prefix=_PREFIX + "measurement-axis-",
    )
    composite_capture_axis_hash = content_hash(
        {
            "schema": "partition_free_v1_sampled_full_population_capture_axis_v1",
            "conditioned_d1_capture_axis_hash": d1_receipt.canonical_capture_axis_hash,
            "sampled_residual_capture_axis_hash": residual.capture_axis_hash,
            "logical_stratum_axis_hash": logical_hash,
        },
        prefix=_PREFIX + "capture-axis-",
    )
    composite_source_packet_axis_hash = content_hash(
        {
            "schema": "partition_free_v1_sampled_full_population_source_packet_axis_v1",
            "conditioned_d1_source_packet_axis_hash": (
                d1_receipt.canonical_source_packet_axis_hash
            ),
            "sampled_residual_source_packet_axis_hash": (
                residual.source_packet_axis_hash
            ),
            "logical_stratum_axis_hash": logical_hash,
        },
        prefix=_PREFIX + "source-packet-axis-",
    )
    source_replay_payload = {
        "schema": "partition_free_v1_sampled_full_population_source_replay_v1",
        "conditioned_d1_source_replay_hash": d1_device.source_replay_hash,
        "residual_source_replay_hash": residual.source_replay_hash,
        "packet_registration_axis_hash": expected_packet_axis_hash,
        "residual_measure_design_hash": design.design_hash,
        "final_weight_receipt_hash": weight.receipt_hash,
        "composite_registered_d_prefix_hash": composite_registered_d_prefix_hash,
    }
    if common_kkt_coordinate_bridge is not None:
        source_replay_payload["common_kkt_coordinate_bridge_receipt_hash"] = (
            common_kkt_coordinate_bridge.receipt_hash
        )
    composite_source_replay_hash = content_hash(
        source_replay_payload,
        prefix=_PREFIX + "source-replay-",
    )
    receipt = V1SampledFullPopulationCompactSourceReceipt(
        conditioned_d1_source_receipt_hash=d1_receipt.receipt_hash,
        conditioned_d1_device_source_receipt_hash=d1_device.receipt_hash,
        residual_segmented_source_receipt_hash=residual.receipt_hash,
        residual_append_receipt_hash=residual.append_receipt_hash,
        residual_source_chain_hash=residual.source_chain_hash,
        packet_registration_receipt_hashes=tuple(item.receipt_hash for item in packets),
        append_packet_registration_hashes=tuple(
            item.append_packet_registration_hash for item in packets
        ),
        append_packet_registration_axis_hash=expected_packet_axis_hash,
        residual_measure_design_hash=design.design_hash,
        final_residual_weight_receipt_hash=weight.receipt_hash,
        suffix_registration_hash=design.suffix_registration_hash,
        generation_frame_hash=design.generation_frame_hash,
        discovery_stream_hash=design.discovery_stream_hash,
        model_identity_hash=residual.model_identity_hash,
        chart_identity_hash=residual.chart_identity_hash,
        current_mask_hash=residual.current_mask_hash,
        proposal_generation=residual.proposal_generation,
        route_axis_hash=residual.route_axis_hash,
        route_id_axis_hash=residual.route_id_axis_hash,
        shell_freeze_receipt_hash=residual.shell_freeze_receipt_hash,
        shell_indices=residual.shell_indices,
        shell_route_ids=residual.shell_route_ids,
        shell_native_fiber_indices=residual.shell_native_fiber_indices,
        native_fiber_axis_receipt_hash=residual.native_fiber_axis_receipt_hash,
        native_fiber_measure_hash=residual.native_fiber_measure_hash,
        conditioned_d1_registered_prefix_hash=d1_receipt.registered_d_prefix_hash,
        d1_pair_prefix_axis_hash=design.d1_pair_prefix_axis_hash,
        final_planned_dw2_prefix_hash=final_packet.planned_d_w2_prefix_hash,
        residual_registered_dw2_prefix_hash=residual.registered_d_w2_prefix_hash,
        logical_stratum_axis_hash=logical_hash,
        append_logical_stratum_mapping_hashes=tuple(
            item.append_logical_stratum_mapping_hash for item in packets
        ),
        corrected_evidence_stream_source_hash=(
            residual.corrected_evidence_stream_source_hash
        ),
        literal_j_stream_source_hash=residual.literal_j_stream_source_hash,
        conditioned_d1_same_capture_vjp_source_hash=(
            d1_device.same_capture_vjp_source_hash
        ),
        residual_same_capture_vjp_source_hash=residual.same_capture_vjp_source_hash,
        source_replay_hash=composite_source_replay_hash,
        kkt_branch_id=residual.kkt_branch_id,
        kkt_branch_source_receipt_hash=residual.kkt_branch_source_receipt_hash,
        resource_axis_hash=residual.resource_axis_hash,
        resource_source_receipt_hash=residual.resource_source_receipt_hash,
        witness_stream_source_hash=composite_witness_stream_hash,
        same_capture_vjp_source_hash=composite_same_capture_hash,
        alpha_ledger_id=residual.alpha_ledger_id,
        alpha_coordinate_id=residual.alpha_coordinate_id,
        alpha_lineage_hash=residual.alpha_lineage_hash,
        design_hash=composite_design_hash,
        registered_d_prefix_hash=composite_registered_d_prefix_hash,
        registered_d_w2_prefix_hash=residual.registered_d_w2_prefix_hash,
        population_context_axis_hash=composite_population_context_axis_hash,
        population_draw_axis_hash=composite_population_draw_axis_hash,
        context_axis_hash=composite_population_context_axis_hash,
        draw_axis_hash=composite_population_draw_axis_hash,
        measurement_axis_hash=composite_measurement_axis_hash,
        capture_axis_hash=composite_capture_axis_hash,
        source_packet_axis_hash=composite_source_packet_axis_hash,
        population_stratum_counts=design.population_stratum_counts,
        d1_stratum_counts=design.d1_stratum_counts,
        residual_capacity_stratum_counts=design.residual_stratum_counts,
        final_dw2_stratum_counts=final_counts,
        population_count=design.population_count,
        context_count=d1_receipt.context_count + residual.context_count,
        d1_context_count=d1_receipt.context_count,
        residual_context_count=residual.context_count,
        route_count=residual.route_count,
        fiber_count=residual.fiber_count,
        cut_count=residual.cut_count,
        segment_count=residual.segment_count,
        d1_context_weight_fraction=weight.d1_context_weight_fraction,
        residual_context_weight_fractions=weight.dw2_context_weight_fractions,
        d1_context_weight_binary64_center=d1_center,
        d1_context_weight_binary64_radius=d1_radius,
        residual_context_weight_binary64_centers=tuple(
            item[0] for item in residual_center_radii
        ),  # type: ignore[arg-type]
        residual_context_weight_binary64_radii=tuple(
            item[1] for item in residual_center_radii
        ),  # type: ignore[arg-type]
        common_measure_denominator=weight.common_denominator,
        d1_context_measure_quantum=weight.d1_context_quantum,
        residual_context_measure_quanta=weight.dw2_context_quanta,
        total_measure_quanta=weight.total_weight_quanta,
        common_kkt_coordinate_bridge_receipt_hash=(
            None
            if common_kkt_coordinate_bridge is None
            else common_kkt_coordinate_bridge.receipt_hash
        ),
        finite_census_axis_custody_complete=(
            final_counts == design.residual_stratum_counts
            and residual.finite_census_axis_custody_complete
        ),
        finite_census_complete=(
            final_counts == design.residual_stratum_counts
            and residual.finite_census_complete
        ),
        _issuance_token=_ISSUER_TOKEN,
    )
    return V1SampledFullPopulationCompactSourceHandle(
        conditioned_d1=conditioned_d1,
        residual=residual_segmented_source,
        packet_receipts=packets,
        measure_design=design,
        weight_receipt=weight,
        common_kkt_bridge=common_kkt_coordinate_bridge,
        receipt=receipt,
        _issuance_token=_ISSUER_TOKEN,
    )


__all__ = [
    "SAMPLED_ANCHOR_INFLUENCING_AXIS_AUTHORITY",
    "SAMPLED_ANCHOR_INFLUENCING_AXIS_SCHEMA",
    "SAMPLED_FULL_POPULATION_CANONICAL_STRATA",
    "SAMPLED_FULL_POPULATION_COMPACT_SOURCE_ALGORITHM",
    "SAMPLED_FULL_POPULATION_COMPACT_SOURCE_AUTHORITY",
    "SAMPLED_FULL_POPULATION_COMPACT_SOURCE_LEASE_SCHEMA",
    "SAMPLED_FULL_POPULATION_COMPACT_SOURCE_SCHEMA",
    "V1SampledFullPopulationCompactSourceHandle",
    "V1SampledFullPopulationCompactSourceLease",
    "V1SampledFullPopulationCompactSourceReceipt",
    "V1SampledFullPopulationCompactSourceRefusal",
    "V1SampledFullPopulationCompactSourceRefusalCode",
    "V1SampledAnchorInfluencingAxisReceipt",
    "compose_v1_sampled_full_population_compact_source",
]
