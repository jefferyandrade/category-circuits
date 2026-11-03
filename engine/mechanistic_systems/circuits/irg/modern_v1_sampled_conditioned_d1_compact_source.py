"""Zero-VJP compact source for the conditioned discovery prefix ``D1``.

The discovery scorer has already paid for, and sealed, corrected Aumann--
Shapley and literal relaxed-gate Jacobian rows.  This module replays those two
immutable roles together and lowers them directly into the bounded
variable-shell source ABI.  It never calls a model, captures another context,
or runs another VJP.

``D1`` is a fixed conditional term in the finite-population estimand.  Its
contexts therefore carry the exact full-population coefficient ``1 / N``;
the binary64 coefficient used by the compact CUDA representation is only the
registered center of the source-issued outward enclosure.  The receipt keeps
that distinction explicit.  Neither this source nor its receipt can select an
action or open C/K/O.  A later exact D1-plus-residual compositor must consume
both this source and the residual-population measure receipt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
import math
from numbers import Integral
from typing import Final, NoReturn, Sequence

import numpy as np

from .contracts import content_hash
from .modern_v1_neutral_runtime import DeclaredRouteAxis
from .modern_v1_sampled_dual_chart_row_custody import (
    V1SampledDualChartMaterializedRows,
    V1SampledDualChartRowRole,
    V1SampledDualChartRowShardAttachment,
)
from .modern_v1_sampled_literal_j_fd_rank_replay import (
    V1SampledLiteralJOriginalCaptureBinding,
)
from .modern_v1_sampled_residual_population_measure import (
    V1SampledResidualPopulationMeasureDesign,
    V1SampledResidualPopulationWeightReceipt,
)
from .modern_v1_sampled_variable_shell_source_bridge import (
    V1SampledVariableShellBoundChunkSinkBinding,
    V1SampledVariableShellSourceBridge,
    V1SampledVariableShellSourceBridgeResult,
)
from .modern_v1_signed_work_variable_shell_source_cuda import (
    V1SignedWorkVariableShellSourceCUDAAccumulator,
    V1SignedWorkVariableShellSourceDeviceHandle,
)

try:  # pragma: no cover - availability depends on the runtime image.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


SAMPLED_CONDITIONED_D1_COMPACT_SOURCE_SCHEMA: Final = (
    "partition_free_v1_sampled_conditioned_D1_compact_source_receipt_v1"
)
SAMPLED_CONDITIONED_D1_COMPACT_SOURCE_AUTHORITY: Final = (
    "conditioned_D1_exact_measure_compact_F_Q_source_only_no_standalone_"
    "action_confidence_C_K_O_commit_terminal_or_pareto_authority_v1"
)
SAMPLED_CONDITIONED_D1_COMPACT_SOURCE_ALGORITHM: Final = (
    "immutable_dual_role_shard_replay_target_then_complement_bounded_native_"
    "fiber_lowering_zero_new_VJP_v1"
)
SAMPLED_CONDITIONED_D1_CANONICAL_STRATA: Final = ("target", "complement")

_PREFIX: Final = "partition-free-v1-sampled-conditioned-D1-compact-source-"
_DW2_SUFFIX_PREFIX: Final = "partition-free-v1-sampled-variable-shell-D-W2-suffix-"
_FLOAT64_BYTES: Final = 8
_ISSUER_TOKEN: Final = object()


class V1SampledConditionedD1CompactSourceRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_IDENTITY = "stale_identity"
    AXIS_MISMATCH = "axis_mismatch"
    ROLE_MISMATCH = "role_mismatch"
    WEIGHT_MISMATCH = "weight_mismatch"
    SOURCE_CUSTODY_MISMATCH = "source_custody_mismatch"
    NONFINITE_SOURCE = "nonfinite_source"
    EXECUTION_REFUSAL = "execution_refusal"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SampledConditionedD1CompactSourceRefusal(RuntimeError):
    """Typed fail-closed boundary for conditioned-D1 compact replay."""

    def __init__(
        self,
        code: V1SampledConditionedD1CompactSourceRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledConditionedD1CompactSourceRefusalCode:
            raise TypeError("conditioned-D1 refusal code must have exact enum type")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledConditionedD1CompactSourceRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledConditionedD1CompactSourceRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledConditionedD1CompactSourceRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SampledConditionedD1CompactSourceRefusalCode.INVALID_TYPE,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SampledConditionedD1CompactSourceRefusalCode.INVALID_TYPE,
            f"{name} must be nonnegative",
        )
    return result


def _axis_hash(role: str, values: Sequence[str]) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_sampled_conditioned_D1_axis_v1",
            "role": role,
            "values": tuple(values),
        },
        prefix=_PREFIX + role + "-axis-",
    )


def _context_owner_token(
    *,
    role: str,
    source_owner_hash: str,
    context_id: str,
    measurement_id: str,
) -> str:
    """Derive the compact ABI's unique per-context source-owner token."""

    return content_hash(
        {
            "schema": "partition_free_v1_sampled_conditioned_D1_context_owner_v1",
            "role": role,
            "source_owner_hash": source_owner_hash,
            "context_id": context_id,
            "measurement_id": measurement_id,
        },
        prefix=_PREFIX + role + "-context-owner-",
    )


def _fraction_enclosure(value: Fraction) -> tuple[float, float, float]:
    center = float(value)
    if not math.isfinite(center):
        _refuse(
            V1SampledConditionedD1CompactSourceRefusalCode.WEIGHT_MISMATCH,
            "exact source conversion overflowed binary64",
        )
    lower = center
    upper = center
    if Fraction.from_float(center) > value:
        lower = math.nextafter(lower, -math.inf)
    elif Fraction.from_float(center) < value:
        upper = math.nextafter(upper, math.inf)
    return (
        math.nextafter(lower, -math.inf),
        center,
        math.nextafter(upper, math.inf),
    )


def _fraction_down(value: Fraction) -> float:
    result = float(value)
    if Fraction.from_float(result) > value:
        result = math.nextafter(result, -math.inf)
    return math.nextafter(result, -math.inf)


def _fraction_up(value: Fraction) -> float:
    result = float(value)
    if Fraction.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    return math.nextafter(result, math.inf)


def _d1_pair_axis_hash(
    *,
    target: Sequence[str],
    complement: Sequence[str],
) -> str:
    """Replay the source D-suffix owner's public content address."""

    role = "D1"
    return content_hash(
        {
            "schema": "partition_free_v1_sampled_variable_shell_D_W2_pair_axis_v1",
            "role": role,
            "target": tuple(target),
            "complement": tuple(complement),
        },
        prefix=_DW2_SUFFIX_PREFIX + role + "-pair-axis-",
    )


@dataclass(frozen=True, slots=True)
class V1SampledConditionedD1CompactSourceReceipt:
    materialization_hash: str
    row_prefix_manifest_hash: str
    original_capture_binding_hash: str
    residual_measure_design_hash: str
    residual_weight_receipt_hash: str
    d1_pair_prefix_axis_hash: str
    compact_source_receipt_hash: str
    compact_bridge_receipt_hash: str
    route_axis_hash: str
    route_id_axis_hash: str
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    registered_d_prefix_hash: str
    corrected_evidence_stream_source_hash: str
    literal_j_stream_source_hash: str
    canonical_context_axis_hash: str
    canonical_draw_axis_hash: str
    canonical_measurement_axis_hash: str
    canonical_capture_axis_hash: str
    canonical_source_packet_axis_hash: str
    corrected_a_shard_receipt_hashes: tuple[str, ...]
    literal_j_shard_receipt_hashes: tuple[str, ...]
    corrected_a_content_sha256s: tuple[str, ...]
    literal_j_content_sha256s: tuple[str, ...]
    d1_stratum_counts: tuple[int, int]
    population_stratum_counts: tuple[int, int]
    context_count: int
    population_count: int
    route_count: int
    exact_d1_context_weight_fraction: tuple[int, int]
    d1_context_weight_binary64_enclosure: tuple[float, float, float]
    compact_internal_context_weight_fraction: tuple[int, int]
    compact_internal_context_weight_binary64_enclosure: tuple[float, float, float]
    compact_to_full_population_scale_fraction: tuple[int, int]
    compact_to_full_population_scale_binary64_enclosure: tuple[float, float, float]
    d1_context_quantum: int
    common_measure_denominator: int
    source_descriptor_pass_count: int
    source_route_window_read_count: int
    source_payload_read_bytes: int
    provider_chunk_pair_count: int
    maximum_owned_host_staging_bytes: int
    maximum_native_fiber_carry_routes_per_context: int
    canonical_stratum_order: tuple[str, str] = SAMPLED_CONDITIONED_D1_CANONICAL_STRATA
    raw_corrected_and_literal_shards_reused: bool = True
    exact_d1_population_coefficient_bound: bool = True
    binary64_weight_is_enclosed_center_not_exact_rational_claim: bool = True
    target_then_complement_source_order: bool = True
    complete_d1_context_cover: bool = True
    complete_route_cover_per_context: bool = True
    same_original_capture_and_vjp_bound: bool = True
    additional_model_forward_count: int = 0
    additional_vjp_count: int = 0
    context_by_route_rows_retained: bool = False
    route_scale_device_to_host_transfer: bool = False
    exact_d1_plus_residual_join_required: bool = True
    raw_shards_are_transient_source_custody: bool = True
    raw_shards_window_streamed_into_compact_lowering: bool = True
    durable_output_is_compact_receipt_and_handle_only: bool = True
    owner_managed_terminal_source_eviction_required: bool = True
    source_shards_physically_evicted_by_this_adapter: bool = False
    standalone_action_authority: bool = False
    confidence_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    algorithm: str = SAMPLED_CONDITIONED_D1_COMPACT_SOURCE_ALGORITHM
    authority_scope: str = SAMPLED_CONDITIONED_D1_COMPACT_SOURCE_AUTHORITY
    schema: str = SAMPLED_CONDITIONED_D1_COMPACT_SOURCE_SCHEMA
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
                V1SampledConditionedD1CompactSourceRefusalCode.INVALID_TYPE,
                "conditioned-D1 receipts are source-issued after compact close",
            )
        for name in (
            "materialization_hash",
            "row_prefix_manifest_hash",
            "original_capture_binding_hash",
            "residual_measure_design_hash",
            "residual_weight_receipt_hash",
            "d1_pair_prefix_axis_hash",
            "compact_source_receipt_hash",
            "compact_bridge_receipt_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "registered_d_prefix_hash",
            "corrected_evidence_stream_source_hash",
            "literal_j_stream_source_hash",
            "canonical_context_axis_hash",
            "canonical_draw_axis_hash",
            "canonical_measurement_axis_hash",
            "canonical_capture_axis_hash",
            "canonical_source_packet_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        integer_names = (
            "proposal_generation",
            "context_count",
            "population_count",
            "route_count",
            "d1_context_quantum",
            "common_measure_denominator",
            "source_descriptor_pass_count",
            "source_route_window_read_count",
            "source_payload_read_bytes",
            "provider_chunk_pair_count",
            "maximum_owned_host_staging_bytes",
            "maximum_native_fiber_carry_routes_per_context",
            "additional_model_forward_count",
            "additional_vjp_count",
        )
        for name in integer_names:
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name),
            )
        if (
            type(self.d1_stratum_counts) is not tuple
            or len(self.d1_stratum_counts) != 2
            or type(self.population_stratum_counts) is not tuple
            or len(self.population_stratum_counts) != 2
        ):
            _refuse(
                V1SampledConditionedD1CompactSourceRefusalCode.INVALID_TYPE,
                "population counts must use the target/complement pair axis",
            )
        d1 = tuple(
            _nonnegative_int(item, name="D1 stratum count")
            for item in self.d1_stratum_counts
        )
        population = tuple(
            _nonnegative_int(item, name="population stratum count")
            for item in self.population_stratum_counts
        )
        if (
            self.context_count != sum(d1)
            or self.population_count != sum(population)
            or self.context_count <= 0
            or self.population_count <= self.context_count
            or any(d1[index] > population[index] for index in (0, 1))
        ):
            _refuse(
                V1SampledConditionedD1CompactSourceRefusalCode.AXIS_MISMATCH,
                "D1 counts do not form a strict prefix of the population",
            )
        if self.exact_d1_context_weight_fraction != (1, self.population_count):
            _refuse(
                V1SampledConditionedD1CompactSourceRefusalCode.WEIGHT_MISMATCH,
                "conditioned D1 must use the exact uniform population weight 1/N",
            )
        enclosures = (
            self.d1_context_weight_binary64_enclosure,
            self.compact_internal_context_weight_binary64_enclosure,
            self.compact_to_full_population_scale_binary64_enclosure,
        )
        if any(
            type(enclosure) is not tuple
            or len(enclosure) != 3
            or any(
                type(item) is not float or not np.isfinite(item) for item in enclosure
            )
            for enclosure in enclosures
        ):
            _refuse(
                V1SampledConditionedD1CompactSourceRefusalCode.WEIGHT_MISMATCH,
                "source weight enclosures must contain three finite floats",
            )
        enclosure = self.d1_context_weight_binary64_enclosure
        exact_weight = Fraction(1, self.population_count)
        if not (
            enclosure[0] <= enclosure[1] <= enclosure[2]
            and Fraction.from_float(enclosure[0]) <= exact_weight
            and exact_weight <= Fraction.from_float(enclosure[2])
        ):
            _refuse(
                V1SampledConditionedD1CompactSourceRefusalCode.WEIGHT_MISMATCH,
                "D1 binary64 weight enclosure does not contain 1/N",
            )
        internal_exact = Fraction(1, self.context_count)
        internal_enclosure = self.compact_internal_context_weight_binary64_enclosure
        if self.compact_internal_context_weight_fraction != (
            internal_exact.numerator,
            internal_exact.denominator,
        ) or not (
            Fraction.from_float(internal_enclosure[0]) <= internal_exact
            and internal_exact <= Fraction.from_float(internal_enclosure[2])
        ):
            _refuse(
                V1SampledConditionedD1CompactSourceRefusalCode.WEIGHT_MISMATCH,
                "compact D1 source must carry the normalized uniform conditional measure",
            )
        internal_center = Fraction.from_float(internal_enclosure[1])
        exact_scale = exact_weight / internal_center
        scale_enclosure = self.compact_to_full_population_scale_binary64_enclosure
        if self.compact_to_full_population_scale_fraction != (
            exact_scale.numerator,
            exact_scale.denominator,
        ) or not (
            Fraction.from_float(scale_enclosure[0]) <= exact_scale
            and exact_scale <= Fraction.from_float(scale_enclosure[2])
        ):
            _refuse(
                V1SampledConditionedD1CompactSourceRefusalCode.WEIGHT_MISMATCH,
                "compact-to-population scale does not exactly correct the binary64 conditional measure",
            )
        if (
            self.d1_context_quantum <= 0
            or self.common_measure_denominator <= 0
            or Fraction(
                self.d1_context_quantum,
                self.common_measure_denominator,
            )
            != exact_weight
        ):
            _refuse(
                V1SampledConditionedD1CompactSourceRefusalCode.WEIGHT_MISMATCH,
                "D1 integer measure quantum does not represent 1/N",
            )
        tuple_axes = (
            self.corrected_a_shard_receipt_hashes,
            self.literal_j_shard_receipt_hashes,
            self.corrected_a_content_sha256s,
            self.literal_j_content_sha256s,
        )
        if (
            any(type(axis) is not tuple or not axis for axis in tuple_axes)
            or len({len(axis) for axis in tuple_axes}) != 1
            or any(
                type(item) is not str or not item
                for axis in tuple_axes
                for item in axis
            )
        ):
            _refuse(
                V1SampledConditionedD1CompactSourceRefusalCode.SOURCE_CUSTODY_MISMATCH,
                "corrected/J shard receipt and content axes must align",
            )
        required_true = (
            self.raw_corrected_and_literal_shards_reused,
            self.exact_d1_population_coefficient_bound,
            self.binary64_weight_is_enclosed_center_not_exact_rational_claim,
            self.target_then_complement_source_order,
            self.complete_d1_context_cover,
            self.complete_route_cover_per_context,
            self.same_original_capture_and_vjp_bound,
            self.exact_d1_plus_residual_join_required,
            self.raw_shards_are_transient_source_custody,
            self.raw_shards_window_streamed_into_compact_lowering,
            self.durable_output_is_compact_receipt_and_handle_only,
            self.owner_managed_terminal_source_eviction_required,
        )
        required_false = (
            self.context_by_route_rows_retained,
            self.route_scale_device_to_host_transfer,
            self.source_shards_physically_evicted_by_this_adapter,
            self.standalone_action_authority,
            self.confidence_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if (
            self.canonical_stratum_order != SAMPLED_CONDITIONED_D1_CANONICAL_STRATA
            or any(type(item) is not bool for item in required_true + required_false)
            or any(item is not True for item in required_true)
            or any(item is not False for item in required_false)
            or self.additional_model_forward_count != 0
            or self.additional_vjp_count != 0
            or self.algorithm != SAMPLED_CONDITIONED_D1_COMPACT_SOURCE_ALGORITHM
            or self.authority_scope != SAMPLED_CONDITIONED_D1_COMPACT_SOURCE_AUTHORITY
            or self.schema != SAMPLED_CONDITIONED_D1_COMPACT_SOURCE_SCHEMA
        ):
            _refuse(
                V1SampledConditionedD1CompactSourceRefusalCode.AUTHORITY_VIOLATION,
                "conditioned-D1 receipt crossed its source-only authority boundary",
            )
        object.__setattr__(self, "d1_stratum_counts", d1)
        object.__setattr__(self, "population_stratum_counts", population)
        observed = self._compute_receipt_hash()
        if (
            self._bound_receipt_hash is not None
            and observed != self._bound_receipt_hash
        ):
            _refuse(
                V1SampledConditionedD1CompactSourceRefusalCode.STALE_IDENTITY,
                "conditioned-D1 receipt changed after source issuance",
            )
        object.__setattr__(self, "_bound_receipt_hash", observed)

    def _compute_receipt_hash(self) -> str:
        return content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            },
            prefix=_PREFIX + "receipt-",
        )

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_receipt_hash()
        if observed != self._bound_receipt_hash:
            _refuse(
                V1SampledConditionedD1CompactSourceRefusalCode.STALE_IDENTITY,
                "conditioned-D1 receipt failed its source-bound hash",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SampledConditionedD1CompactSourceRefusalCode.STALE_IDENTITY,
                "conditioned-D1 receipt differs from the expected source",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1SampledConditionedD1CompactSourceResult:
    source_handle: V1SignedWorkVariableShellSourceDeviceHandle = field(
        repr=False,
        compare=False,
    )
    bridge_result: V1SampledVariableShellSourceBridgeResult = field(
        repr=False,
        compare=False,
    )
    receipt: V1SampledConditionedD1CompactSourceReceipt

    def __post_init__(self) -> None:
        if type(self.source_handle) is not V1SignedWorkVariableShellSourceDeviceHandle:
            _refuse(
                V1SampledConditionedD1CompactSourceRefusalCode.INVALID_TYPE,
                "conditioned-D1 result requires the exact compact source ABI",
            )
        if type(self.receipt) is not V1SampledConditionedD1CompactSourceReceipt:
            _refuse(
                V1SampledConditionedD1CompactSourceRefusalCode.INVALID_TYPE,
                "conditioned-D1 result requires the exact source receipt",
            )
        if type(self.bridge_result) is not V1SampledVariableShellSourceBridgeResult:
            _refuse(
                V1SampledConditionedD1CompactSourceRefusalCode.INVALID_TYPE,
                "conditioned-D1 result requires the exact live compact bridge result",
            )
        bridge_receipt = self.bridge_result.receipt
        if (
            self.bridge_result.source_handle is not self.source_handle
            or self.source_handle.receipt.receipt_hash
            != self.receipt.compact_source_receipt_hash
            or bridge_receipt.compact_source_receipt_hash
            != self.receipt.compact_source_receipt_hash
            or bridge_receipt.receipt_hash
            != self.receipt.compact_bridge_receipt_hash
            or bridge_receipt.route_axis_hash != self.receipt.route_axis_hash
            or bridge_receipt.route_id_axis_hash != self.receipt.route_id_axis_hash
            or bridge_receipt.corrected_evidence_stream_source_hash
            != self.receipt.corrected_evidence_stream_source_hash
            or bridge_receipt.literal_j_stream_source_hash
            != self.receipt.literal_j_stream_source_hash
            or bridge_receipt.context_count != self.receipt.context_count
        ):
            _refuse(
                V1SampledConditionedD1CompactSourceRefusalCode.STALE_IDENTITY,
                "conditioned-D1 result changed its exact compact bridge ownership",
            )
        self.receipt.assert_integrity()

    def evaluate_full_population_contribution(
        self,
        shell_gates: object,
    ) -> "V1SampledConditionedD1ContributionReceipt":
        """Evaluate D1 and exactly convert the compact conditional measure."""

        compact = self.source_handle.evaluate(shell_gates)
        scale = Fraction(*self.receipt.compact_to_full_population_scale_fraction)
        return V1SampledConditionedD1ContributionReceipt(
            conditioned_d1_source_receipt_hash=self.receipt.receipt_hash,
            compact_evaluation_receipt_hash=compact.receipt_hash,
            shell_gate_hash=compact.shell_gate_hash,
            shell_gates=compact.shell_gates,
            f_lower=_fraction_down(Fraction.from_float(compact.f_lower) * scale),
            f_center=float(Fraction.from_float(compact.f_center) * scale),
            f_upper=_fraction_up(Fraction.from_float(compact.f_upper) * scale),
            q_lower=_fraction_down(Fraction.from_float(compact.q_lower) * scale),
            q_center=float(Fraction.from_float(compact.q_center) * scale),
            q_upper=_fraction_up(Fraction.from_float(compact.q_upper) * scale),
            compact_to_full_population_scale_fraction=(
                scale.numerator,
                scale.denominator,
            ),
            _issuance_token=_ISSUER_TOKEN,
        )

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live conditioned-D1 compact sources are nonserializable")


@dataclass(frozen=True, slots=True)
class V1SampledConditionedD1ContributionReceipt:
    """Authority-inert full-population D1 F/Q contribution enclosure."""

    conditioned_d1_source_receipt_hash: str
    compact_evaluation_receipt_hash: str
    shell_gate_hash: str
    shell_gates: tuple[float, ...]
    f_lower: float
    f_center: float
    f_upper: float
    q_lower: float
    q_center: float
    q_upper: float
    compact_to_full_population_scale_fraction: tuple[int, int]
    exact_binary64_measure_correction_applied: bool = True
    d1_plus_residual_join_required: bool = True
    standalone_action_authority: bool = False
    confidence_authority: bool = False
    c_stream_authority: bool = False
    commit_authority: bool = False
    schema: str = "partition_free_v1_sampled_conditioned_D1_contribution_v1"
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
                V1SampledConditionedD1CompactSourceRefusalCode.INVALID_TYPE,
                "D1 contribution receipts are issued only by the live compact source",
            )
        for name in (
            "conditioned_d1_source_receipt_hash",
            "compact_evaluation_receipt_hash",
            "shell_gate_hash",
        ):
            _identity(getattr(self, name), name=name)
        if (
            type(self.shell_gates) is not tuple
            or not self.shell_gates
            or any(
                type(item) is not float or not math.isfinite(item)
                for item in self.shell_gates
            )
        ):
            _refuse(
                V1SampledConditionedD1CompactSourceRefusalCode.INVALID_TYPE,
                "D1 contribution shell gates must be finite immutable floats",
            )
        values = (
            self.f_lower,
            self.f_center,
            self.f_upper,
            self.q_lower,
            self.q_center,
            self.q_upper,
        )
        scale = self.compact_to_full_population_scale_fraction
        if (
            any(type(item) is not float or not math.isfinite(item) for item in values)
            or not (0.0 <= self.f_lower <= self.f_center <= self.f_upper)
            or not (0.0 <= self.q_lower <= self.q_center <= self.q_upper)
            or type(scale) is not tuple
            or len(scale) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, Integral)
                for item in scale
            )
            or int(scale[0]) <= 0
            or int(scale[1]) <= 0
            or self.exact_binary64_measure_correction_applied is not True
            or self.d1_plus_residual_join_required is not True
            or self.standalone_action_authority is not False
            or self.confidence_authority is not False
            or self.c_stream_authority is not False
            or self.commit_authority is not False
            or self.schema != "partition_free_v1_sampled_conditioned_D1_contribution_v1"
        ):
            _refuse(
                V1SampledConditionedD1CompactSourceRefusalCode.AUTHORITY_VIOLATION,
                "D1 contribution receipt changed its interval or authority",
            )
        object.__setattr__(self, "_bound_receipt_hash", self._compute_receipt_hash())

    def _compute_receipt_hash(self) -> str:
        return content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            },
            prefix=_PREFIX + "contribution-",
        )

    @property
    def receipt_hash(self) -> str:
        observed = self._compute_receipt_hash()
        if observed != self._bound_receipt_hash:
            _refuse(
                V1SampledConditionedD1CompactSourceRefusalCode.STALE_IDENTITY,
                "D1 contribution receipt changed after issuance",
            )
        return observed


@dataclass(frozen=True, slots=True)
class _CanonicalContextRecord:
    context_id: str
    draw_id: str
    measurement_id: str
    capture_id: str
    source_packet_hash: str
    stratum: str
    shard_index: int
    row_index: int


def _validate_and_order_sources(
    *,
    materialized_rows: V1SampledDualChartMaterializedRows,
    original_capture_binding: V1SampledLiteralJOriginalCaptureBinding,
    measure_design: V1SampledResidualPopulationMeasureDesign,
    weight_receipt: V1SampledResidualPopulationWeightReceipt,
    accumulator: V1SignedWorkVariableShellSourceCUDAAccumulator,
    compact_route_axis: DeclaredRouteAxis,
) -> tuple[
    tuple[_CanonicalContextRecord, ...],
    tuple[V1SampledDualChartRowShardAttachment, ...],
    tuple[V1SampledDualChartRowShardAttachment, ...],
]:
    exact_types = (
        (materialized_rows, V1SampledDualChartMaterializedRows),
        (original_capture_binding, V1SampledLiteralJOriginalCaptureBinding),
        (measure_design, V1SampledResidualPopulationMeasureDesign),
        (weight_receipt, V1SampledResidualPopulationWeightReceipt),
        (accumulator, V1SignedWorkVariableShellSourceCUDAAccumulator),
        (compact_route_axis, DeclaredRouteAxis),
    )
    if any(type(value) is not expected for value, expected in exact_types):
        _refuse(
            V1SampledConditionedD1CompactSourceRefusalCode.INVALID_TYPE,
            "conditioned-D1 assembly requires exact source-owner types",
        )
    materialized_rows.assert_integrity()
    original_capture_binding.assert_integrity()
    measure_design.assert_integrity()
    weight_receipt.assert_integrity()
    attachment = materialized_rows.attachment
    attachment.verify_live_witnesses()
    manifest = attachment.manifest
    corrected = tuple(attachment.corrected_a_attachments)
    literal = tuple(attachment.literal_j_attachments)
    looks = manifest.looks
    if (
        len(corrected) != len(literal)
        or len(corrected) != len(looks)
        or len(corrected) != len(original_capture_binding.packet_plan_hashes)
        or len({item.receipt.receipt_hash for item in corrected}) != len(corrected)
        or len({item.receipt.receipt_hash for item in literal}) != len(literal)
    ):
        _refuse(
            V1SampledConditionedD1CompactSourceRefusalCode.SOURCE_CUSTODY_MISMATCH,
            "raw dual-role shard axes are incomplete, duplicated, or misaligned",
        )
    shared = (
        (materialized_rows.manifest_hash, manifest.manifest_hash),
        (original_capture_binding.row_prefix_manifest_hash, manifest.manifest_hash),
        (
            original_capture_binding.registered_d_prefix_hash,
            manifest.registered_d_prefix_hash,
        ),
        (original_capture_binding.chart_identity_hash, manifest.chart_identity_hash),
        (original_capture_binding.current_mask_hash, manifest.current_mask_hash),
        (original_capture_binding.proposal_generation, manifest.proposal_generation),
        (original_capture_binding.route_axis_hash, manifest.route_axis_hash),
        (original_capture_binding.route_id_axis_hash, manifest.route_id_axis_hash),
        (original_capture_binding.route_count, manifest.route_count),
        (compact_route_axis.route_axis_hash, manifest.route_axis_hash),
        (compact_route_axis.route_id_axis_hash, manifest.route_id_axis_hash),
        (compact_route_axis.route_count, manifest.route_count),
        (measure_design.current_mask_hash, manifest.current_mask_hash),
        (measure_design.generation_index, manifest.proposal_generation),
        (weight_receipt.measure_design_hash, measure_design.design_hash),
        (weight_receipt.d1_stratum_counts, measure_design.d1_stratum_counts),
        (
            weight_receipt.population_stratum_counts,
            measure_design.population_stratum_counts,
        ),
        (
            weight_receipt.d1_context_weight_fraction,
            (1, measure_design.population_count),
        ),
    )
    if any(left != right for left, right in shared):
        _refuse(
            V1SampledConditionedD1CompactSourceRefusalCode.STALE_IDENTITY,
            "D1 rows, original capture, route axis, and population measure do not share one frame",
        )

    records: list[_CanonicalContextRecord] = []
    for shard_index, (a_shard, j_shard, look) in enumerate(
        zip(corrected, literal, looks, strict=True)
    ):
        a_receipt = a_shard.receipt
        j_receipt = j_shard.receipt
        expected_contexts = original_capture_binding.packet_context_ids[shard_index]
        expected_draws = original_capture_binding.packet_draw_ids[shard_index]
        expected_strata = original_capture_binding.packet_stratum_labels[shard_index]
        measurements = original_capture_binding.packet_fd_source_measurement_hashes[
            shard_index
        ]
        capture = original_capture_binding.packet_source_capture_identity_hashes[
            shard_index
        ]
        source_packet = original_capture_binding.packet_plan_hashes[shard_index]
        if (
            a_receipt.role is not V1SampledDualChartRowRole.CORRECTED_AUMANN_SHAPLEY
            or j_receipt.role
            is not V1SampledDualChartRowRole.LITERAL_RELAXED_GATE_JACOBIAN
            or a_receipt.receipt_hash != look.corrected_a_shard.receipt_hash
            or j_receipt.receipt_hash != look.literal_j_shard.receipt_hash
            or a_receipt.context_ids != j_receipt.context_ids
            or a_receipt.draw_ids != j_receipt.draw_ids
            or a_receipt.stratum_labels != j_receipt.stratum_labels
            or a_receipt.route_windows != j_receipt.route_windows
            or a_receipt.context_ids != expected_contexts
            or a_receipt.draw_ids != expected_draws
            or a_receipt.stratum_labels != expected_strata
            or a_receipt.source_capture_identity_hash != capture
            or j_receipt.source_capture_identity_hash != capture
            or len(measurements) != len(expected_contexts)
        ):
            _refuse(
                V1SampledConditionedD1CompactSourceRefusalCode.ROLE_MISMATCH,
                "one corrected/J shard pair differs from its original same-VJP capture",
            )
        a_shard.verify_live_witnesses(
            expected_role=V1SampledDualChartRowRole.CORRECTED_AUMANN_SHAPLEY,
            expected_chart_identity_hash=manifest.chart_identity_hash,
            expected_current_mask_hash=manifest.current_mask_hash,
            expected_proposal_generation=manifest.proposal_generation,
        )
        j_shard.verify_live_witnesses(
            expected_role=V1SampledDualChartRowRole.LITERAL_RELAXED_GATE_JACOBIAN,
            expected_chart_identity_hash=manifest.chart_identity_hash,
            expected_current_mask_hash=manifest.current_mask_hash,
            expected_proposal_generation=manifest.proposal_generation,
        )
        for row_index, (context, draw, measurement, stratum) in enumerate(
            zip(
                expected_contexts,
                expected_draws,
                measurements,
                expected_strata,
                strict=True,
            )
        ):
            if stratum not in SAMPLED_CONDITIONED_D1_CANONICAL_STRATA:
                _refuse(
                    V1SampledConditionedD1CompactSourceRefusalCode.AXIS_MISMATCH,
                    "D1 raw rows must carry target/complement source labels",
                )
            records.append(
                _CanonicalContextRecord(
                    context_id=context,
                    draw_id=draw,
                    measurement_id=measurement,
                    capture_id=_context_owner_token(
                        role="capture",
                        source_owner_hash=capture,
                        context_id=context,
                        measurement_id=measurement,
                    ),
                    source_packet_hash=_context_owner_token(
                        role="source_packet",
                        source_owner_hash=source_packet,
                        context_id=context,
                        measurement_id=measurement,
                    ),
                    stratum=stratum,
                    shard_index=shard_index,
                    row_index=row_index,
                )
            )

    canonical = tuple(
        record
        for stratum in SAMPLED_CONDITIONED_D1_CANONICAL_STRATA
        for record in records
        if record.stratum == stratum
    )
    target = tuple(
        record.context_id for record in canonical if record.stratum == "target"
    )
    complement = tuple(
        record.context_id for record in canonical if record.stratum == "complement"
    )
    if (
        (len(target), len(complement)) != measure_design.d1_stratum_counts
        or len(canonical) != len(materialized_rows.context_ids)
        or len({record.context_id for record in canonical}) != len(canonical)
        or _d1_pair_axis_hash(target=target, complement=complement)
        != measure_design.d1_pair_prefix_axis_hash
    ):
        _refuse(
            V1SampledConditionedD1CompactSourceRefusalCode.AXIS_MISMATCH,
            "raw D1 rows are not the exact target/complement discovery prefix",
        )

    registration = accumulator.registration
    canonical_axes = (
        tuple(record.context_id for record in canonical),
        tuple(record.draw_id for record in canonical),
        tuple(record.measurement_id for record in canonical),
        tuple(record.capture_id for record in canonical),
        tuple(record.source_packet_hash for record in canonical),
    )
    registration_axes = (
        registration.registered_context_ids,
        registration.registered_draw_ids,
        registration.registered_measurement_ids,
        registration.registered_capture_ids,
        registration.registered_source_packet_hashes,
    )
    internal_weight = Fraction(1, len(canonical))
    expected_center = float(internal_weight)
    accumulator_shared = (
        (registration.chart_identity_hash, manifest.chart_identity_hash),
        (registration.current_mask_hash, manifest.current_mask_hash),
        (registration.proposal_generation, manifest.proposal_generation),
        (registration.route_axis_hash, manifest.route_axis_hash),
        (registration.route_id_axis_hash, manifest.route_id_axis_hash),
        (registration.registered_d_prefix_hash, manifest.registered_d_prefix_hash),
        (
            registration.registered_d_w2_prefix_hash,
            measure_design.d1_pair_prefix_axis_hash,
        ),
        (
            registration.parent_d_source_receipt_hash,
            original_capture_binding.binding_hash,
        ),
        (
            registration.same_capture_vjp_source_hash,
            original_capture_binding.binding_hash,
        ),
        (
            registration.corrected_evidence_stream_source_hash,
            manifest.corrected_a_stream_identity_hash,
        ),
        (
            registration.literal_j_stream_source_hash,
            manifest.literal_j_stream_identity_hash,
        ),
        (registration.population_count, measure_design.population_count),
        (
            registration.native_fiber_axis_receipt_hash,
            accumulator.native_fiber_axis_receipt.receipt_hash,
        ),
        (
            registration.native_fiber_measure_hash,
            accumulator.native_fiber_axis_receipt.measure_hash,
        ),
    )
    if (
        any(left != right for left, right in accumulator_shared)
        or registration_axes != canonical_axes
        or registration.context_weights
        != tuple(expected_center for _ in range(len(canonical)))
    ):
        _refuse(
            V1SampledConditionedD1CompactSourceRefusalCode.STALE_IDENTITY,
            "compact accumulator is not registered to the exact canonical D1 source and normalized conditional center",
        )
    return canonical, corrected, literal


def assemble_v1_sampled_conditioned_d1_compact_source(
    *,
    materialized_rows: V1SampledDualChartMaterializedRows,
    original_capture_binding: V1SampledLiteralJOriginalCaptureBinding,
    residual_measure_design: V1SampledResidualPopulationMeasureDesign,
    residual_weight_receipt: V1SampledResidualPopulationWeightReceipt,
    accumulator: V1SignedWorkVariableShellSourceCUDAAccumulator,
    compact_route_axis: DeclaredRouteAxis,
    bound_chunk_sinks: Sequence[V1SampledVariableShellBoundChunkSinkBinding] = (),
) -> V1SampledConditionedD1CompactSourceResult:
    """Replay sealed D1 dual rows into one compact source without a VJP."""

    if torch is None:
        _refuse(
            V1SampledConditionedD1CompactSourceRefusalCode.EXECUTION_REFUSAL,
            "conditioned-D1 compact replay requires PyTorch",
        )
    canonical, corrected, literal = _validate_and_order_sources(
        materialized_rows=materialized_rows,
        original_capture_binding=original_capture_binding,
        measure_design=residual_measure_design,
        weight_receipt=residual_weight_receipt,
        accumulator=accumulator,
        compact_route_axis=compact_route_axis,
    )
    destination_transport_hash = content_hash(
        {
            "schema": "partition_free_v1_sampled_conditioned_D1_transport_v1",
            "materialization_hash": materialized_rows.materialization_hash,
            "original_capture_binding_hash": original_capture_binding.binding_hash,
            "measure_design_hash": residual_measure_design.design_hash,
            "weight_receipt_hash": residual_weight_receipt.receipt_hash,
            "compact_registration_hash": accumulator.registration.registration_hash,
            "zero_new_vjp": True,
        },
        prefix=_PREFIX + "transport-",
    )
    bridge = V1SampledVariableShellSourceBridge(
        accumulator=accumulator,
        route_ids=compact_route_axis.route_ids,
        route_axis_hash=compact_route_axis.route_axis_hash,
        route_id_axis_hash=compact_route_axis.route_id_axis_hash,
        destination_transport_hash=destination_transport_hash,
        bound_chunk_sinks=bound_chunk_sinks,
    )
    consumer = bridge.producer_consumer
    physical_chunk = accumulator.design.route_chunk_size
    consumer.evidence_consumer.bind_route_chunk_size(physical_chunk)
    consumer.jacobian_consumer.bind_route_chunk_size(physical_chunk)

    descriptor_passes = 0
    route_window_reads = 0
    payload_read_bytes = 0
    maximum_host_staging = 0
    registration = accumulator.registration
    canonical_cursor = 0
    for stratum in SAMPLED_CONDITIONED_D1_CANONICAL_STRATA:
        for shard_index, (a_shard, j_shard) in enumerate(
            zip(corrected, literal, strict=True)
        ):
            records = tuple(
                record
                for record in canonical
                if record.stratum == stratum and record.shard_index == shard_index
            )
            for batch_start in range(
                0, len(records), accumulator.design.maximum_batch_size
            ):
                batch = records[
                    batch_start : batch_start + accumulator.design.maximum_batch_size
                ]
                if not batch:
                    continue
                expected = canonical[canonical_cursor : canonical_cursor + len(batch)]
                if batch != expected:
                    _refuse(
                        V1SampledConditionedD1CompactSourceRefusalCode.AXIS_MISMATCH,
                        "bounded shard replay departed from canonical D1 order",
                    )
                contexts = tuple(record.context_id for record in batch)
                evidence = consumer.evidence_consumer
                jacobian = consumer.jacobian_consumer
                evidence.begin_packet(
                    contexts,
                    route_ids=compact_route_axis.route_ids,
                    destination_transport_hash=destination_transport_hash,
                    loss_pullback_hash=(
                        registration.corrected_evidence_stream_source_hash
                    ),
                )
                jacobian.begin_packet(
                    contexts,
                    route_ids=compact_route_axis.route_ids,
                    destination_transport_hash=destination_transport_hash,
                    loss_pullback_hash=registration.literal_j_stream_source_hash,
                )
                row_indices = tuple(record.row_index for record in batch)
                a_windows = a_shard.iter_route_windows(verify=True)
                j_windows = j_shard.iter_route_windows(verify=True)
                descriptor_passes += 2
                for a_window, j_window in zip(a_windows, j_windows, strict=True):
                    a_start, a_stop, a_rows = a_window
                    j_start, j_stop, j_rows = j_window
                    if (a_start, a_stop) != (j_start, j_stop):
                        _refuse(
                            V1SampledConditionedD1CompactSourceRefusalCode.ROLE_MISMATCH,
                            "corrected/J route windows are not congruent",
                        )
                    width = a_stop - a_start
                    route_window_reads += 2
                    payload_read_bytes += int(a_rows.nbytes + j_rows.nbytes)
                    maximum_host_staging = max(
                        maximum_host_staging,
                        int(a_rows.nbytes + j_rows.nbytes),
                    )
                    for local_start in range(0, width, physical_chunk):
                        local_stop = min(width, local_start + physical_chunk)
                        start = a_start + local_start
                        stop = a_start + local_stop
                        a_selected = np.array(
                            a_rows[row_indices, local_start:local_stop],
                            dtype=np.float64,
                            order="C",
                            copy=True,
                        )
                        j_selected = np.array(
                            j_rows[row_indices, local_start:local_stop],
                            dtype=np.float64,
                            order="C",
                            copy=True,
                        )
                        if not (
                            bool(np.isfinite(a_selected).all())
                            and bool(np.isfinite(j_selected).all())
                        ):
                            _refuse(
                                V1SampledConditionedD1CompactSourceRefusalCode.NONFINITE_SOURCE,
                                "sealed D1 source window contains a nonfinite value",
                            )
                        maximum_host_staging = max(
                            maximum_host_staging,
                            int(
                                a_rows.nbytes
                                + j_rows.nbytes
                                + a_selected.nbytes
                                + j_selected.nbytes
                            ),
                        )
                        a_tensor = torch.from_numpy(a_selected).to(
                            device=accumulator.device,
                            dtype=torch.float64,
                        )
                        j_tensor = torch.from_numpy(j_selected).to(
                            device=accumulator.device,
                            dtype=torch.float64,
                        )
                        evidence.consume_positive_removal_chunk(
                            start,
                            stop,
                            a_tensor.contiguous(),
                        )
                        jacobian.consume_positive_removal_chunk(
                            start,
                            stop,
                            j_tensor.contiguous(),
                        )
                evidence.end_packet()
                jacobian.end_packet()
                canonical_cursor += len(batch)
    if canonical_cursor != len(canonical):
        _refuse(
            V1SampledConditionedD1CompactSourceRefusalCode.AXIS_MISMATCH,
            "conditioned-D1 replay did not consume every canonical context",
        )
    try:
        compact = bridge.seal()
    except (RuntimeError, TypeError, ValueError, PermissionError) as error:
        raise V1SampledConditionedD1CompactSourceRefusal(
            V1SampledConditionedD1CompactSourceRefusalCode.EXECUTION_REFUSAL,
            "compact same-VJP source bridge refused conditioned-D1 closure",
        ) from error

    manifest = materialized_rows.attachment.manifest
    internal_weight = Fraction(1, len(canonical))
    receipt = V1SampledConditionedD1CompactSourceReceipt(
        materialization_hash=materialized_rows.materialization_hash,
        row_prefix_manifest_hash=manifest.manifest_hash,
        original_capture_binding_hash=original_capture_binding.binding_hash,
        residual_measure_design_hash=residual_measure_design.design_hash,
        residual_weight_receipt_hash=residual_weight_receipt.receipt_hash,
        d1_pair_prefix_axis_hash=residual_measure_design.d1_pair_prefix_axis_hash,
        compact_source_receipt_hash=compact.source_handle.receipt.receipt_hash,
        compact_bridge_receipt_hash=compact.receipt.receipt_hash,
        route_axis_hash=manifest.route_axis_hash,
        route_id_axis_hash=manifest.route_id_axis_hash,
        native_fiber_axis_receipt_hash=(
            accumulator.native_fiber_axis_receipt.receipt_hash
        ),
        native_fiber_measure_hash=accumulator.native_fiber_axis_receipt.measure_hash,
        chart_identity_hash=manifest.chart_identity_hash,
        current_mask_hash=manifest.current_mask_hash,
        proposal_generation=manifest.proposal_generation,
        registered_d_prefix_hash=manifest.registered_d_prefix_hash,
        corrected_evidence_stream_source_hash=(
            manifest.corrected_a_stream_identity_hash
        ),
        literal_j_stream_source_hash=manifest.literal_j_stream_identity_hash,
        canonical_context_axis_hash=_axis_hash(
            "context", tuple(record.context_id for record in canonical)
        ),
        canonical_draw_axis_hash=_axis_hash(
            "draw", tuple(record.draw_id for record in canonical)
        ),
        canonical_measurement_axis_hash=_axis_hash(
            "measurement", tuple(record.measurement_id for record in canonical)
        ),
        canonical_capture_axis_hash=_axis_hash(
            "capture", tuple(record.capture_id for record in canonical)
        ),
        canonical_source_packet_axis_hash=_axis_hash(
            "source_packet", tuple(record.source_packet_hash for record in canonical)
        ),
        corrected_a_shard_receipt_hashes=tuple(
            item.receipt.receipt_hash for item in corrected
        ),
        literal_j_shard_receipt_hashes=tuple(
            item.receipt.receipt_hash for item in literal
        ),
        corrected_a_content_sha256s=tuple(
            item.receipt.content_sha256 for item in corrected
        ),
        literal_j_content_sha256s=tuple(
            item.receipt.content_sha256 for item in literal
        ),
        d1_stratum_counts=residual_measure_design.d1_stratum_counts,
        population_stratum_counts=residual_measure_design.population_stratum_counts,
        context_count=len(canonical),
        population_count=residual_measure_design.population_count,
        route_count=manifest.route_count,
        exact_d1_context_weight_fraction=(1, residual_measure_design.population_count),
        d1_context_weight_binary64_enclosure=(
            residual_weight_receipt.d1_context_weight_binary64_enclosure
        ),
        compact_internal_context_weight_fraction=(
            internal_weight.numerator,
            internal_weight.denominator,
        ),
        compact_internal_context_weight_binary64_enclosure=_fraction_enclosure(
            internal_weight
        ),
        compact_to_full_population_scale_fraction=(
            (
                Fraction(1, residual_measure_design.population_count)
                / Fraction.from_float(float(internal_weight))
            ).numerator,
            (
                Fraction(1, residual_measure_design.population_count)
                / Fraction.from_float(float(internal_weight))
            ).denominator,
        ),
        compact_to_full_population_scale_binary64_enclosure=_fraction_enclosure(
            Fraction(1, residual_measure_design.population_count)
            / Fraction.from_float(float(internal_weight))
        ),
        d1_context_quantum=residual_weight_receipt.d1_context_quantum,
        common_measure_denominator=residual_weight_receipt.common_denominator,
        source_descriptor_pass_count=descriptor_passes,
        source_route_window_read_count=route_window_reads,
        source_payload_read_bytes=payload_read_bytes,
        provider_chunk_pair_count=compact.receipt.paired_route_chunk_count,
        maximum_owned_host_staging_bytes=maximum_host_staging,
        maximum_native_fiber_carry_routes_per_context=(
            compact.receipt.maximum_native_fiber_carry_routes_per_context
        ),
        _issuance_token=_ISSUER_TOKEN,
    )
    return V1SampledConditionedD1CompactSourceResult(
        source_handle=compact.source_handle,
        bridge_result=compact,
        receipt=receipt,
    )


__all__ = [
    "SAMPLED_CONDITIONED_D1_CANONICAL_STRATA",
    "SAMPLED_CONDITIONED_D1_COMPACT_SOURCE_ALGORITHM",
    "SAMPLED_CONDITIONED_D1_COMPACT_SOURCE_AUTHORITY",
    "SAMPLED_CONDITIONED_D1_COMPACT_SOURCE_SCHEMA",
    "V1SampledConditionedD1CompactSourceReceipt",
    "V1SampledConditionedD1CompactSourceRefusal",
    "V1SampledConditionedD1CompactSourceRefusalCode",
    "V1SampledConditionedD1CompactSourceResult",
    "V1SampledConditionedD1ContributionReceipt",
    "assemble_v1_sampled_conditioned_d1_compact_source",
]
