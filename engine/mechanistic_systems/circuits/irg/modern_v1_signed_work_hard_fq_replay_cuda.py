"""Complete hard signed-F/Q replay for a frozen two-branch device shell.

The relaxed signed-work solver and deterministic shell rounder are proposal
mechanisms.  This module measures every frozen hard endpoint again, from the
resident first-inclusion labels, before any C-stream outcome is visible.

The implementation keeps the canonical separation of roles:

* signed ``F`` is the native-destination-fiber expectation produced by the
  signed-work operator; singleton additive work is never substituted;
* ``Q`` is replayed from the literal retained Jacobian factor and both
  deterministic FD residual envelopes;
* the spectral and diagonal convex branches are both retained; and
* no F budget or downstream authority is invented here.

Production execution is CUDA-only and forms endpoint membership only inside
bounded route/native-fiber tiles.  The private CPU path is an explicit,
route-capped mathematical reference.  Receipts carry compact scalar
enclosures and exact frozen-action identities only.
"""

from __future__ import annotations

from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

from dataclasses import InitVar, dataclass
from enum import Enum
import hashlib
import math
from numbers import Integral, Real
import os
from typing import Any, Final, NoReturn, Sequence

import numpy as np

from mechanistic_systems.contracts.content_address import canonical_json, content_hash

from .modern_v1_device_shell_ladder import (
    DEVICE_SHELL_ELIGIBILITY_COMMITMENT_DOMAIN,
    DEVICE_SHELL_LABEL_COMMITMENT_DOMAIN,
    V1DeviceShellBorrowOperation,
    V1DeviceShellFrozenLadderReceipt,
    V1DeviceShellRefusal,
)
from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch
from .modern_v1_signed_work_device_shell_cuda import (
    V1SignedWorkDeviceShellDeviceBundleV2,
    V1SignedWorkDeviceShellDeviceFamilyV2,
    V1SignedWorkDeviceShellExecutionBundle,
    V1SignedWorkDeviceShellFamilyBundle,
    V1SignedWorkDeviceShellRefusal,
    V1SignedWorkDeviceShellTensorOperationV2,
)
from .modern_v1_signed_work_kkt_cuda import (
    V1SignedWorkLeastWorkCUDAPolicy,
    V1SignedWorkLeastWorkCUDADeviceProblemV2,
    V1SignedWorkLeastWorkCUDAExecutionBundle,
    V1SignedWorkLeastWorkCUDAProblem,
    V1SignedWorkLeastWorkCUDARefusal,
    V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle,
)
from .modern_v1_signed_work_operator_cuda import V1SignedWorkOperatorCUDARefusal

try:  # pragma: no cover - availability is environment dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


SIGNED_WORK_HARD_FQ_REPLAY_DESIGN_SCHEMA: Final = (
    "partition_free_v1_signed_work_hard_fq_replay_design_v1"
)
SIGNED_WORK_HARD_FQ_HBM_SCHEMA: Final = (
    "partition_free_v1_signed_work_hard_fq_replay_hbm_v1"
)
SIGNED_WORK_HARD_FQ_ENDPOINT_SCHEMA: Final = (
    "partition_free_v1_signed_work_hard_fq_endpoint_receipt_v2"
)
SIGNED_WORK_HARD_FQ_BRANCH_SCHEMA: Final = (
    "partition_free_v1_signed_work_hard_fq_branch_receipt_v1"
)
SIGNED_WORK_HARD_FQ_FAMILY_SCHEMA: Final = (
    "partition_free_v1_signed_work_hard_fq_family_receipt_v1"
)
SIGNED_WORK_HARD_FQ_BUNDLE_SCHEMA: Final = (
    "partition_free_v1_signed_work_hard_fq_family_bundle_v1"
)
SIGNED_WORK_HARD_FQ_SINGLE_BRANCH_BUNDLE_SCHEMA: Final = (
    "partition_free_v1_signed_work_hard_fq_single_branch_bundle_v1"
)
SIGNED_WORK_HARD_FQ_ALGORITHM_ID: Final = (
    "device_first_inclusion_native_fiber_signed_F_and_tiled_literal_J_Q_"
    "complete_two_branch_fp64_v1"
)
SIGNED_WORK_HARD_FQ_OUTWARD_ARITHMETIC_ID: Final = (
    "binary64_gamma_n_absolute_contraction_envelope_nextafter_v1"
)
SIGNED_WORK_HARD_FQ_AUTHORITY: Final = (
    "pre_C_hard_signed_F_Q_measurement_only_no_R_C_K_O_calibration_oracle_"
    "commit_terminal_or_pareto_authority_v1"
)
SIGNED_WORK_HARD_FQ_UNIT_RESOURCE_ID: Final = (
    "unit_scalar_route_count_resource_bound_to_uint8_endpoint_counts_v1"
)

_DESIGN_PREFIX: Final = "partition-free-v1-signed-work-hard-fq-design-"
_HBM_PREFIX: Final = "partition-free-v1-signed-work-hard-fq-hbm-"
_ENDPOINT_PREFIX: Final = "partition-free-v1-signed-work-hard-fq-endpoint-"
_BRANCH_PREFIX: Final = "partition-free-v1-signed-work-hard-fq-branch-"
_FAMILY_PREFIX: Final = "partition-free-v1-signed-work-hard-fq-family-"
_F64_BYTES: Final = np.dtype(np.float64).itemsize
_U8_BYTES: Final = np.dtype(np.uint8).itemsize
_BOOL_BYTES: Final = np.dtype(np.bool_).itemsize
_UNIT_ROUNDOFF: Final = float(np.finfo(np.float64).eps) / 2.0
_MIN_NORMAL: Final = float(np.finfo(np.float64).tiny)
_MINIMUM_ALLOCATOR_GUARD_BYTES: Final = 64 * 1024 * 1024
_ALLOCATOR_GUARD_FRACTION: Final = 0.10


class V1SignedWorkHardFQReplayRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_IDENTITY = "stale_identity"
    LABEL_IDENTITY_MISMATCH = "label_identity_mismatch"
    SOURCE_CUSTODY_UNAVAILABLE = "source_custody_unavailable"
    CUDA_REQUIRED = "cuda_required"
    DETERMINISM_REQUIRED = "determinism_required"
    CPU_REFERENCE_SCALE_EXCEEDED = "cpu_reference_scale_exceeded"
    NONUNIT_RESOURCE_ENDPOINT_PLAN = "nonunit_resource_endpoint_plan"
    HBM_FORECAST_EXCEEDED = "hbm_forecast_exceeded"
    NUMERICAL_REFUSAL = "numerical_refusal"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SignedWorkHardFQReplayRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SignedWorkHardFQReplayRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SignedWorkHardFQReplayRefusalCode:
            raise TypeError("hard-F/Q replay refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SignedWorkHardFQReplayRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SignedWorkHardFQReplayRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.INVALID_TYPE,
            f"{name} must be one positive integer",
        )
    result = int(value)
    if result <= 0:
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.INVALID_TYPE,
            f"{name} must be nonnegative",
        )
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative finite real",
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.NUMERICAL_REFUSAL,
            f"{name} must be nonnegative and finite",
        )
    return result


def _next_down(value: float) -> float:
    result = math.nextafter(float(value), -math.inf)
    if not math.isfinite(result):
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.NUMERICAL_REFUSAL,
            "outward lower endpoint left finite binary64",
        )
    return result


def _next_up(value: float) -> float:
    result = math.nextafter(float(value), math.inf)
    if not math.isfinite(result):
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.NUMERICAL_REFUSAL,
            "outward upper endpoint left finite binary64",
        )
    return result


def _up_add(left: float, right: float) -> float:
    if left == 0.0:
        return float(right)
    if right == 0.0:
        return float(left)
    return _next_up(float(left) + float(right))


def _down_add(left: float, right: float) -> float:
    if left == 0.0:
        return float(right)
    if right == 0.0:
        return float(left)
    return _next_down(float(left) + float(right))


def _up_multiply(left: float, right: float) -> float:
    if left == 0.0 or right == 0.0:
        return 0.0
    return _next_up(float(left) * float(right))


def _down_multiply_nonnegative(left: float, right: float) -> float:
    if left == 0.0 or right == 0.0:
        return 0.0
    return max(0.0, _next_down(float(left) * float(right)))


def _gamma_n(operation_count: int) -> float:
    count = _nonnegative_int(operation_count, name="rounding operation count")
    if count == 0:
        return 0.0
    product = _up_multiply(float(count), _UNIT_ROUNDOFF)
    if product >= 0.5:
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.NUMERICAL_REFUSAL,
            "registered axis is too large for certified binary64 accumulation",
        )
    denominator = _next_down(1.0 - product)
    if denominator <= 0.0:
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.NUMERICAL_REFUSAL,
            "gamma_n denominator is nonpositive",
        )
    return _next_up(product / denominator)


def _signed_sum_interval(
    *, nominal: float, absolute_envelope: float, operation_count: int
) -> tuple[float, float]:
    center = float(nominal)
    envelope = _finite_nonnegative(absolute_envelope, name="absolute envelope")
    if not math.isfinite(center):
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.NUMERICAL_REFUSAL,
            "signed nominal is nonfinite",
        )
    gamma = _gamma_n(operation_count)
    inflated = (
        envelope
        if gamma == 0.0
        else _next_up(envelope / _next_down(1.0 - gamma))
    )
    roundoff = _up_multiply(gamma, inflated)
    underflow = _up_multiply(float(operation_count), _MIN_NORMAL)
    radius = _up_add(roundoff, underflow)
    return _next_down(center - radius), _next_up(center + radius)


def _nonnegative_interval_from_nominal(
    *, nominal: float, absolute_envelope: float, operation_count: int
) -> tuple[float, float]:
    lower, upper = _signed_sum_interval(
        nominal=nominal,
        absolute_envelope=absolute_envelope,
        operation_count=operation_count,
    )
    return max(0.0, lower), max(0.0, upper)


def _square_interval(lower: float, upper: float) -> tuple[float, float]:
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.NUMERICAL_REFUSAL,
            "invalid signed interval before squaring",
        )
    if lower <= 0.0 <= upper:
        minimum = 0.0
    else:
        minimum = _down_multiply_nonnegative(
            min(abs(lower), abs(upper)), min(abs(lower), abs(upper))
        )
    maximum_abs = max(abs(lower), abs(upper))
    return minimum, _up_multiply(maximum_abs, maximum_abs)


def _sum_nonnegative_intervals(
    intervals: Sequence[tuple[float, float]],
) -> tuple[float, float]:
    lower = 0.0
    upper = 0.0
    for item_lower, item_upper in intervals:
        lower = max(0.0, _down_add(lower, item_lower))
        upper = _up_add(upper, item_upper)
    return lower, upper


def _divide_nonnegative_interval(
    lower: float, upper: float, denominator: int
) -> tuple[float, float]:
    divisor = float(_positive_int(denominator, name="interval denominator"))
    return (
        max(0.0, _next_down(lower / divisor)) if lower else 0.0,
        _next_up(upper / divisor) if upper else 0.0,
    )


def _new_stream_digest(*, domain: str, metadata: object) -> "hashlib._Hash":
    encoded_domain = domain.encode("ascii")
    encoded_metadata = canonical_json(metadata).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(encoded_domain).to_bytes(8, "big", signed=False))
    digest.update(encoded_domain)
    digest.update(len(encoded_metadata).to_bytes(8, "big", signed=False))
    digest.update(encoded_metadata)
    return digest


class V1SignedWorkHardFQEndpointStatus(str, Enum):
    Q_SAFE_FEASIBLE = "q_safe_feasible"
    Q_ROBUST_INFEASIBLE = "q_robust_infeasible"
    Q_ENVELOPE_AMBIGUOUS = "q_envelope_ambiguous"


@dataclass(frozen=True, slots=True)
class V1SignedWorkHardFQReplayDesign:
    endpoint_lane_count: int = 8
    q_route_tile_size: int = 131_072
    label_verification_chunk_size: int = 1 << 20
    hbm_ceiling_fraction: float = 0.85
    production_cuda_required: bool = True
    allow_private_cpu_reference: bool = False
    private_reference_route_limit: int = 4096
    require_deterministic_algorithms: bool = True
    require_cublas_deterministic_workspace: bool = True
    replay_every_registered_endpoint: bool = True
    allow_nonmonotone_q_early_stop: bool = False
    resource_coordinate_id: str = SIGNED_WORK_HARD_FQ_UNIT_RESOURCE_ID
    algorithm_id: str = SIGNED_WORK_HARD_FQ_ALGORITHM_ID
    outward_arithmetic_id: str = SIGNED_WORK_HARD_FQ_OUTWARD_ARITHMETIC_ID
    schema: str = SIGNED_WORK_HARD_FQ_REPLAY_DESIGN_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "endpoint_lane_count",
            "q_route_tile_size",
            "label_verification_chunk_size",
        ):
            object.__setattr__(
                self, name, _positive_int(getattr(self, name), name=name)
            )
        object.__setattr__(
            self,
            "private_reference_route_limit",
            _nonnegative_int(
                self.private_reference_route_limit,
                name="private reference route limit",
            ),
        )
        ceiling = _finite_nonnegative(
            self.hbm_ceiling_fraction, name="HBM ceiling fraction"
        )
        if ceiling != 0.85:
            raise ValueError("canonical hard-F/Q HBM ceiling is frozen at 0.85")
        object.__setattr__(self, "hbm_ceiling_fraction", ceiling)
        flags = (
            self.production_cuda_required,
            self.allow_private_cpu_reference,
            self.require_deterministic_algorithms,
            self.require_cublas_deterministic_workspace,
            self.replay_every_registered_endpoint,
            self.allow_nonmonotone_q_early_stop,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("hard-F/Q design flags must be exact booleans")
        if self.production_cuda_required and self.allow_private_cpu_reference:
            raise ValueError("production replay cannot enable a CPU fallback")
        if not self.replay_every_registered_endpoint or self.allow_nonmonotone_q_early_stop:
            raise PermissionError("hard-F/Q replay must measure every frozen endpoint")
        if (
            self.resource_coordinate_id != SIGNED_WORK_HARD_FQ_UNIT_RESOURCE_ID
            or self.algorithm_id != SIGNED_WORK_HARD_FQ_ALGORITHM_ID
            or self.outward_arithmetic_id
            != SIGNED_WORK_HARD_FQ_OUTWARD_ARITHMETIC_ID
            or self.schema != SIGNED_WORK_HARD_FQ_REPLAY_DESIGN_SCHEMA
        ):
            raise ValueError("hard-F/Q replay semantics changed")

    @property
    def design_hash(self) -> str:
        return content_hash(self, prefix=_DESIGN_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkHardFQHBMForecast:
    problem_hash: str
    policy_hash: str
    least_work_execution_hash: str
    shell_family_hash: str
    design_hash: str
    device: str
    route_count: int
    context_count: int
    fiber_count: int
    response_rank: int
    endpoint_count: int
    lane_count: int
    q_route_tile_size: int
    resident_two_branch_label_bytes: int
    f_signed_and_envelope_cell_bytes: int
    f_operator_tile_workspace_upper_bytes: int
    q_membership_tile_bytes: int
    q_absolute_factor_tile_bytes: int
    q_compact_projection_bytes: int
    compact_lane_summary_bytes: int
    f_peak_workspace_bytes: int
    q_peak_workspace_bytes: int
    allocator_guard_bytes: int
    forecast_incremental_bytes: int
    current_reserved_bytes: int
    forecast_peak_reserved_bytes: int
    device_total_memory_bytes: int
    hbm_ceiling_bytes: int
    admitted: bool
    route_by_endpoint_action_matrix_materialized: bool = False
    schema: str = SIGNED_WORK_HARD_FQ_HBM_SCHEMA
    _admission_memory: InitVar[tuple[int, int] | None] = None

    def __post_init__(self, _admission_memory: tuple[int, int] | None) -> None:
        available, allocated = _admission_memory or (self.device_total_memory_bytes, self.current_reserved_bytes)
        for name in (
            "problem_hash",
            "policy_hash",
            "least_work_execution_hash",
            "shell_family_hash",
            "design_hash",
            "device",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "route_count",
            "context_count",
            "fiber_count",
            "endpoint_count",
            "lane_count",
            "q_route_tile_size",
        ):
            object.__setattr__(
                self, name, _positive_int(getattr(self, name), name=name)
            )
        object.__setattr__(
            self,
            "response_rank",
            _nonnegative_int(self.response_rank, name="response rank"),
        )
        byte_names = (
            "resident_two_branch_label_bytes",
            "f_signed_and_envelope_cell_bytes",
            "f_operator_tile_workspace_upper_bytes",
            "q_membership_tile_bytes",
            "q_absolute_factor_tile_bytes",
            "q_compact_projection_bytes",
            "compact_lane_summary_bytes",
            "f_peak_workspace_bytes",
            "q_peak_workspace_bytes",
            "allocator_guard_bytes",
            "forecast_incremental_bytes",
            "current_reserved_bytes",
            "forecast_peak_reserved_bytes",
            "device_total_memory_bytes",
            "hbm_ceiling_bytes",
        )
        for name in byte_names:
            object.__setattr__(
                self, name, _nonnegative_int(getattr(self, name), name=name)
            )
        n = self.route_count
        c = self.context_count
        f = self.fiber_count
        r = self.response_rank
        l = self.lane_count
        tile = min(n, self.q_route_tile_size)
        expected_labels = 2 * n * _U8_BYTES
        expected_cells = 2 * c * f * l * _F64_BYTES
        # ``(label > 0) & (label <= endpoint)`` may keep both comparison
        # results plus the combined mask live before the FP64 conversion.
        expected_q_membership = tile * l * (3 * _BOOL_BYTES + _F64_BYTES)
        expected_q_factor = r * tile * _F64_BYTES
        expected_q_compact = (2 * r * l + 5 * l) * _F64_BYTES
        expected_summary = 32 * l * _F64_BYTES
        expected_f_peak = (
            expected_cells
            + self.f_operator_tile_workspace_upper_bytes
            + expected_summary
        )
        expected_q_peak = (
            expected_q_membership
            + expected_q_factor
            + expected_q_compact
            + expected_summary
        )
        working = max(expected_f_peak, expected_q_peak)
        expected_guard = max(
            _MINIMUM_ALLOCATOR_GUARD_BYTES,
            math.ceil(_ALLOCATOR_GUARD_FRACTION * working),
        )
        expected_incremental = working + expected_guard
        expected_peak = self.current_reserved_bytes + expected_incremental
        expected_ceiling = math.floor(
            self.device_total_memory_bytes * 0.85
        )
        expected_admitted = allocated + expected_incremental <= expected_ceiling and expected_incremental <= available
        observed = (
            self.resident_two_branch_label_bytes,
            self.f_signed_and_envelope_cell_bytes,
            self.q_membership_tile_bytes,
            self.q_absolute_factor_tile_bytes,
            self.q_compact_projection_bytes,
            self.compact_lane_summary_bytes,
            self.f_peak_workspace_bytes,
            self.q_peak_workspace_bytes,
            self.allocator_guard_bytes,
            self.forecast_incremental_bytes,
            self.forecast_peak_reserved_bytes,
            self.hbm_ceiling_bytes,
            self.admitted,
        )
        expected = (
            expected_labels,
            expected_cells,
            expected_q_membership,
            expected_q_factor,
            expected_q_compact,
            expected_summary,
            expected_f_peak,
            expected_q_peak,
            expected_guard,
            expected_incremental,
            expected_peak,
            expected_ceiling,
            expected_admitted,
        )
        if observed != expected:
            raise ValueError("hard-F/Q HBM forecast does not replay its formula")
        if type(self.route_by_endpoint_action_matrix_materialized) is not bool or (
            self.route_by_endpoint_action_matrix_materialized
        ):
            raise PermissionError("hard-F/Q replay may not allocate route×endpoint z")
        if type(self.admitted) is not bool:
            raise TypeError("hard-F/Q HBM admission must be exact")
        if self.schema != SIGNED_WORK_HARD_FQ_HBM_SCHEMA:
            raise ValueError("unsupported hard-F/Q HBM schema")

    @property
    def forecast_hash(self) -> str:
        return content_hash(self, prefix=_HBM_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkHardFQEndpointReceipt:
    branch: V1RelaxedKKTBranch
    endpoint_label: int
    cumulative_removed_count: int
    frozen_action_receipt_hash: str
    packed_action_hash: str
    resulting_mask_hash: str
    removed_route_index_hash: str
    resource_nominal: float
    resource_lower: float
    resource_upper: float
    signed_f_nominal: float
    signed_f_lower: float
    signed_f_upper: float
    low_rank_q_nominal: float
    low_rank_q_lower: float
    low_rank_q_upper: float
    spectral_residual_q_nominal: float
    spectral_residual_q_lower: float
    spectral_residual_q_upper: float
    diagonal_residual_q_nominal: float
    diagonal_residual_q_lower: float
    diagonal_residual_q_upper: float
    spectral_q_lower: float
    spectral_q_upper: float
    diagonal_q_lower: float
    diagonal_q_upper: float
    branch_q_lower: float
    branch_q_upper: float
    canonical_min_q_lower: float
    canonical_min_q_upper: float
    quadratic_budget: float
    origin_branch_status: V1SignedWorkHardFQEndpointStatus
    status: V1SignedWorkHardFQEndpointStatus
    signed_f_is_objective_not_constraint: bool = True
    f_budget_used: bool = False
    replayed_without_q_early_stop: bool = True
    outward_arithmetic_id: str = SIGNED_WORK_HARD_FQ_OUTWARD_ARITHMETIC_ID
    authority_scope: str = SIGNED_WORK_HARD_FQ_AUTHORITY
    c_stream_authority: bool = False
    nonlinear_remainder_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = SIGNED_WORK_HARD_FQ_ENDPOINT_SCHEMA

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("hard-F/Q endpoint branch must be exact")
        object.__setattr__(
            self,
            "endpoint_label",
            _positive_int(self.endpoint_label, name="endpoint label"),
        )
        object.__setattr__(
            self,
            "cumulative_removed_count",
            _positive_int(
                self.cumulative_removed_count,
                name="cumulative removed count",
            ),
        )
        for name in (
            "frozen_action_receipt_hash",
            "packed_action_hash",
            "resulting_mask_hash",
            "removed_route_index_hash",
            "outward_arithmetic_id",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        scalar_names = (
            "resource_nominal",
            "resource_lower",
            "resource_upper",
            "signed_f_nominal",
            "signed_f_lower",
            "signed_f_upper",
            "low_rank_q_nominal",
            "low_rank_q_lower",
            "low_rank_q_upper",
            "spectral_residual_q_nominal",
            "spectral_residual_q_lower",
            "spectral_residual_q_upper",
            "diagonal_residual_q_nominal",
            "diagonal_residual_q_lower",
            "diagonal_residual_q_upper",
            "spectral_q_lower",
            "spectral_q_upper",
            "diagonal_q_lower",
            "diagonal_q_upper",
            "branch_q_lower",
            "branch_q_upper",
            "canonical_min_q_lower",
            "canonical_min_q_upper",
            "quadratic_budget",
        )
        for name in scalar_names:
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name=name),
            )
        interval_triplets = (
            (self.resource_lower, self.resource_nominal, self.resource_upper),
            (self.signed_f_lower, self.signed_f_nominal, self.signed_f_upper),
            (self.low_rank_q_lower, self.low_rank_q_nominal, self.low_rank_q_upper),
            (
                self.spectral_residual_q_lower,
                self.spectral_residual_q_nominal,
                self.spectral_residual_q_upper,
            ),
            (
                self.diagonal_residual_q_lower,
                self.diagonal_residual_q_nominal,
                self.diagonal_residual_q_upper,
            ),
        )
        if any(not (lower <= nominal <= upper) for lower, nominal, upper in interval_triplets):
            raise ValueError("hard-F/Q nominal escaped an outward enclosure")
        expected_spectral = (
            max(0.0, _down_add(self.low_rank_q_lower, self.spectral_residual_q_lower)),
            _up_add(self.low_rank_q_upper, self.spectral_residual_q_upper),
        )
        expected_diagonal = (
            max(0.0, _down_add(self.low_rank_q_lower, self.diagonal_residual_q_lower)),
            _up_add(self.low_rank_q_upper, self.diagonal_residual_q_upper),
        )
        expected_branch = (
            expected_spectral
            if self.branch is V1RelaxedKKTBranch.SPECTRAL
            else expected_diagonal
        )
        expected_canonical = (
            min(expected_spectral[0], expected_diagonal[0]),
            min(expected_spectral[1], expected_diagonal[1]),
        )
        if (
            (self.spectral_q_lower, self.spectral_q_upper) != expected_spectral
            or (self.diagonal_q_lower, self.diagonal_q_upper) != expected_diagonal
            or (self.branch_q_lower, self.branch_q_upper) != expected_branch
            or (
                self.canonical_min_q_lower,
                self.canonical_min_q_upper,
            )
            != expected_canonical
        ):
            raise ValueError("hard-F/Q endpoint Q envelope does not replay")
        expected_origin_status = (
            V1SignedWorkHardFQEndpointStatus.Q_SAFE_FEASIBLE
            if self.branch_q_upper < self.quadratic_budget
            else (
                V1SignedWorkHardFQEndpointStatus.Q_ROBUST_INFEASIBLE
                if self.branch_q_lower > self.quadratic_budget
                else V1SignedWorkHardFQEndpointStatus.Q_ENVELOPE_AMBIGUOUS
            )
        )
        expected_status = (
            V1SignedWorkHardFQEndpointStatus.Q_SAFE_FEASIBLE
            if self.canonical_min_q_upper < self.quadratic_budget
            else (
                V1SignedWorkHardFQEndpointStatus.Q_ROBUST_INFEASIBLE
                if self.canonical_min_q_lower > self.quadratic_budget
                else V1SignedWorkHardFQEndpointStatus.Q_ENVELOPE_AMBIGUOUS
            )
        )
        if (
            type(self.origin_branch_status) is not V1SignedWorkHardFQEndpointStatus
            or self.origin_branch_status is not expected_origin_status
            or
            type(self.status) is not V1SignedWorkHardFQEndpointStatus
            or self.status is not expected_status
        ):
            raise ValueError("hard-F/Q endpoint branch/union status does not replay")
        closed_flags = (
            self.signed_f_is_objective_not_constraint,
            self.f_budget_used,
            self.replayed_without_q_early_stop,
            self.c_stream_authority,
            self.nonlinear_remainder_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in closed_flags) or closed_flags != (
            True,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ):
            _refuse(
                V1SignedWorkHardFQReplayRefusalCode.AUTHORITY_VIOLATION,
                "hard-F/Q endpoint crossed its proposal-measurement boundary",
            )
        if (
            self.outward_arithmetic_id
            != SIGNED_WORK_HARD_FQ_OUTWARD_ARITHMETIC_ID
            or self.authority_scope != SIGNED_WORK_HARD_FQ_AUTHORITY
            or self.schema != SIGNED_WORK_HARD_FQ_ENDPOINT_SCHEMA
        ):
            raise ValueError("hard-F/Q endpoint semantics changed")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_ENDPOINT_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkHardFQBranchReceipt:
    branch: V1RelaxedKKTBranch
    problem_hash: str
    policy_hash: str
    least_work_execution_hash: str
    state_identity_hash: str
    shell_family_hash: str
    shell_execution_hash: str
    shell_execution_receipt_hash: str
    branch_identity_receipt_hash: str
    frozen_ladder_receipt_hash: str
    label_commitment: str
    operator_receipt_hash: str
    design_hash: str
    hbm_forecast_hash: str
    endpoints: tuple[V1SignedWorkHardFQEndpointReceipt, ...]
    label_identity_fully_verified: bool
    every_registered_endpoint_replayed: bool
    production_cuda_execution: bool
    private_cpu_reference_execution: bool
    route_by_endpoint_action_matrix_materialized: bool = False
    signed_f_is_objective_not_constraint: bool = True
    authority_scope: str = SIGNED_WORK_HARD_FQ_AUTHORITY
    c_stream_authority: bool = False
    nonlinear_remainder_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = SIGNED_WORK_HARD_FQ_BRANCH_SCHEMA

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("hard-F/Q branch must be exact")
        for name in (
            "problem_hash",
            "policy_hash",
            "least_work_execution_hash",
            "state_identity_hash",
            "shell_family_hash",
            "shell_execution_hash",
            "shell_execution_receipt_hash",
            "branch_identity_receipt_hash",
            "frozen_ladder_receipt_hash",
            "label_commitment",
            "operator_receipt_hash",
            "design_hash",
            "hbm_forecast_hash",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        endpoints = tuple(self.endpoints)
        if not endpoints or any(
            type(item) is not V1SignedWorkHardFQEndpointReceipt
            or item.branch is not self.branch
            for item in endpoints
        ):
            raise TypeError("hard-F/Q branch requires exact same-branch endpoints")
        if tuple(item.endpoint_label for item in endpoints) != tuple(
            range(1, len(endpoints) + 1)
        ):
            raise ValueError("hard-F/Q branch endpoints are incomplete or reordered")
        object.__setattr__(self, "endpoints", endpoints)
        flags = (
            self.label_identity_fully_verified,
            self.every_registered_endpoint_replayed,
            self.production_cuda_execution,
            self.private_cpu_reference_execution,
            self.route_by_endpoint_action_matrix_materialized,
            self.signed_f_is_objective_not_constraint,
            self.c_stream_authority,
            self.nonlinear_remainder_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("hard-F/Q branch flags must be exact booleans")
        if flags[:2] != (True, True) or flags[2] == flags[3] or flags[4:] != (
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ):
            _refuse(
                V1SignedWorkHardFQReplayRefusalCode.AUTHORITY_VIOLATION,
                "hard-F/Q branch receipt weakened completeness or authority",
            )
        if (
            self.authority_scope != SIGNED_WORK_HARD_FQ_AUTHORITY
            or self.schema != SIGNED_WORK_HARD_FQ_BRANCH_SCHEMA
        ):
            raise ValueError("hard-F/Q branch semantics changed")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_BRANCH_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkHardFQFamilyReceipt:
    problem_hash: str
    policy_hash: str
    least_work_execution_hash: str
    state_identity_hash: str
    shell_family_hash: str
    design_hash: str
    hbm_forecast_hash: str
    spectral: V1SignedWorkHardFQBranchReceipt
    diagonal_l1: V1SignedWorkHardFQBranchReceipt
    common_endpoint_counts: tuple[int, ...]
    both_q_branches_replayed: bool = True
    every_registered_endpoint_replayed: bool = True
    validation_data_seen: bool = False
    authority_scope: str = SIGNED_WORK_HARD_FQ_AUTHORITY
    c_stream_authority: bool = False
    nonlinear_remainder_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = SIGNED_WORK_HARD_FQ_FAMILY_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "problem_hash",
            "policy_hash",
            "least_work_execution_hash",
            "state_identity_hash",
            "shell_family_hash",
            "design_hash",
            "hbm_forecast_hash",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if (
            type(self.spectral) is not V1SignedWorkHardFQBranchReceipt
            or type(self.diagonal_l1) is not V1SignedWorkHardFQBranchReceipt
            or self.spectral.branch is not V1RelaxedKKTBranch.SPECTRAL
            or self.diagonal_l1.branch is not V1RelaxedKKTBranch.DIAGONAL_L1
        ):
            raise TypeError("hard-F/Q family requires both exact Q branches")
        shared = (
            self.problem_hash,
            self.policy_hash,
            self.least_work_execution_hash,
            self.state_identity_hash,
            self.shell_family_hash,
            self.design_hash,
            self.hbm_forecast_hash,
        )
        if any(
            (
                branch.problem_hash,
                branch.policy_hash,
                branch.least_work_execution_hash,
                branch.state_identity_hash,
                branch.shell_family_hash,
                branch.design_hash,
                branch.hbm_forecast_hash,
            )
            != shared
            for branch in (self.spectral, self.diagonal_l1)
        ):
            raise ValueError("hard-F/Q family and branch source identities differ")
        counts = tuple(
            _positive_int(value, name="common endpoint count")
            for value in self.common_endpoint_counts
        )
        if tuple(sorted(set(counts))) != counts:
            raise ValueError("hard-F/Q common endpoints must increase")
        if counts != tuple(
            endpoint.cumulative_removed_count for endpoint in self.spectral.endpoints
        ) or counts != tuple(
            endpoint.cumulative_removed_count for endpoint in self.diagonal_l1.endpoints
        ):
            raise ValueError("hard-F/Q branches do not cover one common endpoint axis")
        object.__setattr__(self, "common_endpoint_counts", counts)
        flags = (
            self.both_q_branches_replayed,
            self.every_registered_endpoint_replayed,
            self.validation_data_seen,
            self.c_stream_authority,
            self.nonlinear_remainder_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in flags) or flags != (
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ):
            _refuse(
                V1SignedWorkHardFQReplayRefusalCode.AUTHORITY_VIOLATION,
                "hard-F/Q family crossed its pre-C measurement boundary",
            )
        if (
            self.authority_scope != SIGNED_WORK_HARD_FQ_AUTHORITY
            or self.schema != SIGNED_WORK_HARD_FQ_FAMILY_SCHEMA
        ):
            raise ValueError("hard-F/Q family semantics changed")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_FAMILY_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkHardFQFamilyBundle:
    receipt: V1SignedWorkHardFQFamilyReceipt
    hbm_forecast: V1SignedWorkHardFQHBMForecast
    schema: str = SIGNED_WORK_HARD_FQ_BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        if (
            type(self.receipt) is not V1SignedWorkHardFQFamilyReceipt
            or type(self.hbm_forecast) is not V1SignedWorkHardFQHBMForecast
        ):
            raise TypeError("hard-F/Q bundle requires exact receipt and forecast")
        if self.receipt.hbm_forecast_hash != self.hbm_forecast.forecast_hash:
            raise ValueError("hard-F/Q receipt and forecast differ")
        if self.schema != SIGNED_WORK_HARD_FQ_BUNDLE_SCHEMA:
            raise ValueError("unsupported hard-F/Q bundle schema")

    @property
    def bundle_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "receipt_hash": self.receipt.receipt_hash,
                "forecast_hash": self.hbm_forecast.forecast_hash,
            },
            prefix=_FAMILY_PREFIX + "bundle-",
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkHardFQSingleBranchBundle:
    """Complete hard F/Q replay for one Phase-A-supported origin branch."""

    receipt: V1SignedWorkHardFQBranchReceipt
    hbm_forecast: V1SignedWorkHardFQHBMForecast
    branch_membership_precedes_replay: bool = True
    nonrequested_branch_replay_claimed: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = SIGNED_WORK_HARD_FQ_SINGLE_BRANCH_BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        if type(self.receipt) is not V1SignedWorkHardFQBranchReceipt:
            raise TypeError("single-branch hard-F/Q bundle requires exact receipt")
        if type(self.hbm_forecast) is not V1SignedWorkHardFQHBMForecast:
            raise TypeError("single-branch hard-F/Q bundle requires exact forecast")
        if self.receipt.hbm_forecast_hash != self.hbm_forecast.forecast_hash:
            raise ValueError("single-branch replay receipt and forecast differ")
        flags = (
            self.branch_membership_precedes_replay,
            self.nonrequested_branch_replay_claimed,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in flags) or flags != (
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ):
            _refuse(
                V1SignedWorkHardFQReplayRefusalCode.AUTHORITY_VIOLATION,
                "single-branch hard replay fabricated branch or downstream authority",
            )
        if self.schema != SIGNED_WORK_HARD_FQ_SINGLE_BRANCH_BUNDLE_SCHEMA:
            raise ValueError("unsupported single-branch hard-F/Q bundle schema")

    @property
    def bundle_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "receipt_hash": self.receipt.receipt_hash,
                "forecast_hash": self.hbm_forecast.forecast_hash,
                "branch_membership_precedes_replay": (
                    self.branch_membership_precedes_replay
                ),
                "nonrequested_branch_replay_claimed": (
                    self.nonrequested_branch_replay_claimed
                ),
                "c_stream_authority": self.c_stream_authority,
                "calibration_authority": self.calibration_authority,
                "oracle_authority": self.oracle_authority,
                "mask_commit_authority": self.mask_commit_authority,
                "terminal_authority": self.terminal_authority,
                "pareto_authority": self.pareto_authority,
            },
            prefix=_BRANCH_PREFIX + "bundle-",
        )


def _validate_identities(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    execution: V1SignedWorkLeastWorkCUDAExecutionBundle,
    shell_family: V1SignedWorkDeviceShellFamilyBundle,
    design: V1SignedWorkHardFQReplayDesign,
) -> None:
    if type(problem) is not V1SignedWorkLeastWorkCUDAProblem:
        raise TypeError("hard-F/Q replay requires an exact signed-work problem")
    if type(policy) is not V1SignedWorkLeastWorkCUDAPolicy:
        raise TypeError("hard-F/Q replay requires an exact signed-work policy")
    if type(execution) is not V1SignedWorkLeastWorkCUDAExecutionBundle:
        raise TypeError("hard-F/Q replay requires an exact least-work execution")
    if type(shell_family) is not V1SignedWorkDeviceShellFamilyBundle:
        raise TypeError("hard-F/Q replay requires an exact two-branch shell family")
    if type(design) is not V1SignedWorkHardFQReplayDesign:
        raise TypeError("hard-F/Q replay requires an exact replay design")
    problem.assert_integrity()
    source = problem.source_problem
    family = shell_family.receipt
    if (
        execution.result.problem_hash != problem.problem_hash
        or execution.result.policy_hash != policy.policy_hash
        or family.problem_hash != problem.problem_hash
        or family.policy_hash != policy.policy_hash
        or family.least_work_execution_hash != execution.execution_hash
        or shell_family.spectral.execution_receipt.problem_hash != problem.problem_hash
        or shell_family.diagonal_l1.execution_receipt.problem_hash != problem.problem_hash
        or shell_family.spectral.execution_receipt.policy_hash != policy.policy_hash
        or shell_family.diagonal_l1.execution_receipt.policy_hash != policy.policy_hash
        or shell_family.spectral.execution_receipt.least_work_execution_hash
        != execution.execution_hash
        or shell_family.diagonal_l1.execution_receipt.least_work_execution_hash
        != execution.execution_hash
    ):
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.STALE_IDENTITY,
            "problem, policy, solver execution, and shell family are not one chart",
        )
    branch_bundles = (shell_family.spectral, shell_family.diagonal_l1)
    solver_receipts = {
        receipt.branch: receipt.receipt_hash for receipt in execution.branch_receipts
    }
    if any(
        receipt.operator_receipt_hash
        != problem.signed_work_operator.receipt_hash
        for receipt in execution.branch_receipts
    ):
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.STALE_IDENTITY,
            "solver branch receipt and signed-F operator identity differ",
        )
    if (
        shell_family.spectral.frozen_ladder.cumulative_counts
        != shell_family.diagonal_l1.frozen_ladder.cumulative_counts
        or shell_family.spectral.frozen_ladder.route_axis_hash
        != source.route_axis_hash
        or shell_family.diagonal_l1.frozen_ladder.route_axis_hash
        != source.route_axis_hash
        or any(
            branch.frozen_ladder.route_count != source.route_count
            or branch.execution_receipt.source_geometry_hash
            != source.source_geometry_hash
            or branch.execution_receipt.source_registration_hash
            != problem.source_registration_hash
            or branch.execution_receipt.registered_d_coordinate_hash
            != problem.registered_d_coordinate_hash
            or branch.execution_receipt.state_identity_hash
            != execution.state_identity_hash
            or branch.execution_receipt.endpoint_plan_hash
            != family.endpoint_plan_hash
            or branch.execution_receipt.design_hash != family.design_hash
            or branch.execution_receipt.parent_mask_hash != family.parent_mask_hash
            or branch.branch_identity.signed_work_branch_receipt_hash
            != solver_receipts.get(branch.execution_receipt.selected_branch)
            for branch in branch_bundles
        )
    ):
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.STALE_IDENTITY,
            "frozen branch axes or D/source identities differ",
        )
    if not problem.source_bundle_authoritative or execution.state_handle is None:
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.SOURCE_CUSTODY_UNAVAILABLE,
            "hard-F/Q replay requires retained authoritative theorem state",
        )
    forbidden = (
        problem.c_stream_outcomes_seen,
        problem.calibration_authority,
        problem.bce_oracle_authority,
        problem.mask_commit_authority,
        problem.terminal_authority,
        problem.pareto_claim_authority,
        execution.result.c_stream_authority,
        execution.result.calibration_authority,
        execution.result.bce_oracle_authority,
        execution.result.mask_commit_authority,
        execution.result.terminal_authority,
        execution.result.pareto_claim_authority,
        family.c_stream_outcomes_seen,
        family.fqr_authority,
        family.calibration_authority,
        family.oracle_authority,
        family.mask_commit_authority,
        family.terminal_authority,
        family.pareto_authority,
    )
    if any(forbidden):
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.AUTHORITY_VIOLATION,
            "hard-F/Q replay input crossed the pre-C proposal boundary",
        )
    if not np.array_equal(source.resource, np.ones(source.route_count, dtype=np.float64)):
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.NONUNIT_RESOURCE_ENDPOINT_PLAN,
            "count-valued shell endpoints require exact unit route resource",
        )
    operator = problem.signed_work_operator
    if design.endpoint_lane_count > operator.design.maximum_lanes:
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.INVALID_TYPE,
            "replay endpoint lanes exceed the signed-work operator design",
        )
    if design.production_cuda_required:
        if (
            not execution.production_cuda_execution
            or not operator.receipt.production_cuda_execution
            or shell_family.spectral.label_handle.backend_kind != "torch_cuda"
            or shell_family.diagonal_l1.label_handle.backend_kind != "torch_cuda"
        ):
            _refuse(
                V1SignedWorkHardFQReplayRefusalCode.CUDA_REQUIRED,
                "production hard-F/Q replay requires CUDA solver/operator/labels",
            )
    else:
        if not design.allow_private_cpu_reference:
            _refuse(
                V1SignedWorkHardFQReplayRefusalCode.CUDA_REQUIRED,
                "CPU replay was not explicitly enabled",
            )
        if (
            source.route_count > design.private_reference_route_limit
            or not execution.private_cpu_reference_execution
            or not operator.receipt.private_cpu_reference_execution
            or shell_family.spectral.label_handle.backend_kind != "numpy_reference"
            or shell_family.diagonal_l1.label_handle.backend_kind != "numpy_reference"
        ):
            _refuse(
                V1SignedWorkHardFQReplayRefusalCode.CPU_REFERENCE_SCALE_EXCEEDED,
                "private CPU replay is restricted to one tiny all-CPU fixture",
            )
    if (
        design.require_deterministic_algorithms
        and execution.production_cuda_execution
        and (torch is None or not torch.are_deterministic_algorithms_enabled())
    ):
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.DETERMINISM_REQUIRED,
            "CUDA hard-F/Q replay requires deterministic torch algorithms",
        )
    if (
        design.require_cublas_deterministic_workspace
        and execution.production_cuda_execution
        and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}
    ):
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.DETERMINISM_REQUIRED,
            "CUDA hard-F/Q replay requires a deterministic cuBLAS workspace",
        )


def estimate_v1_signed_work_hard_fq_replay_hbm(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    execution: V1SignedWorkLeastWorkCUDAExecutionBundle,
    shell_family: V1SignedWorkDeviceShellFamilyBundle,
    design: V1SignedWorkHardFQReplayDesign = V1SignedWorkHardFQReplayDesign(),
    current_reserved_bytes: int | None = None,
    device_total_memory_bytes: int | None = None,
) -> V1SignedWorkHardFQHBMForecast:
    """Forecast incremental replay HBM without allocating route×endpoint z."""

    try:
        _validate_identities(
            problem=problem,
            policy=policy,
            execution=execution,
            shell_family=shell_family,
            design=design,
        )
    except V1DeviceShellRefusal as error:
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.LABEL_IDENTITY_MISMATCH,
            f"device-shell label custody refused the HBM forecast: {error}",
        )
    state = execution.state_handle
    if state is None:  # pragma: no cover - static validation already closes this.
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.SOURCE_CUSTODY_UNAVAILABLE,
            "hard-F/Q state disappeared before HBM forecast",
        )
    source = problem.source_problem
    operator = problem.signed_work_operator
    endpoint_count = shell_family.spectral.frozen_ladder.endpoint_count
    lanes = min(endpoint_count, design.endpoint_lane_count)
    route_count = source.route_count
    rank = source.rank
    q_tile = min(route_count, design.q_route_tile_size)
    labels_bytes = 2 * route_count * _U8_BYTES
    cells = 2 * operator.context_count * operator.fiber_count * lanes * _F64_BYTES
    # The label-native operator computes signed and absolute contractions in
    # one tile.  Twice its one-stream upper bound safely covers the paired
    # rank/fiber intermediates without introducing a route-scale action lane.
    operator_tile = 2 * operator.forecast.temporary_tile_bytes_upper
    q_membership = q_tile * lanes * (3 * _BOOL_BYTES + _F64_BYTES)
    q_factor = rank * q_tile * _F64_BYTES
    q_compact = (2 * rank * lanes + 5 * lanes) * _F64_BYTES
    summary = 32 * lanes * _F64_BYTES
    f_peak = cells + operator_tile + summary
    q_peak = q_membership + q_factor + q_compact + summary
    working = max(f_peak, q_peak)
    guard = max(
        _MINIMUM_ALLOCATOR_GUARD_BYTES,
        math.ceil(_ALLOCATOR_GUARD_FRACTION * working),
    )
    incremental = working + guard
    admission_memory = None
    if current_reserved_bytes is None or device_total_memory_bytes is None:
        if execution.production_cuda_execution:
            if torch is None or not torch.cuda.is_available():
                _refuse(
                    V1SignedWorkHardFQReplayRefusalCode.CUDA_REQUIRED,
                    "CUDA memory forecast requested without a CUDA runtime",
                )
            device = torch.device(state.device)
            admission_memory = (cuda_reusable_memory_bytes(device, framework=torch), int(torch.cuda.memory_allocated(device)))
            if current_reserved_bytes is None:
                current_reserved_bytes = int(torch.cuda.memory_reserved(device))
            if device_total_memory_bytes is None:
                device_total_memory_bytes = int(
                    torch.cuda.get_device_properties(device).total_memory
                )
        else:
            if current_reserved_bytes is None:
                current_reserved_bytes = 0
            if device_total_memory_bytes is None:
                device_total_memory_bytes = max(1 << 30, 4 * incremental)
    current = _nonnegative_int(current_reserved_bytes, name="current reserved bytes")
    total = _positive_int(device_total_memory_bytes, name="device total memory bytes")
    ceiling = math.floor(design.hbm_ceiling_fraction * total)
    peak = current + incremental
    available, allocated = admission_memory or (total, current)
    return V1SignedWorkHardFQHBMForecast(
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        least_work_execution_hash=execution.execution_hash,
        shell_family_hash=shell_family.family_hash,
        design_hash=design.design_hash,
        device=state.device,
        route_count=route_count,
        context_count=operator.context_count,
        fiber_count=operator.fiber_count,
        response_rank=rank,
        endpoint_count=endpoint_count,
        lane_count=lanes,
        q_route_tile_size=design.q_route_tile_size,
        resident_two_branch_label_bytes=labels_bytes,
        f_signed_and_envelope_cell_bytes=cells,
        f_operator_tile_workspace_upper_bytes=operator_tile,
        q_membership_tile_bytes=q_membership,
        q_absolute_factor_tile_bytes=q_factor,
        q_compact_projection_bytes=q_compact,
        compact_lane_summary_bytes=summary,
        f_peak_workspace_bytes=f_peak,
        q_peak_workspace_bytes=q_peak,
        allocator_guard_bytes=guard,
        forecast_incremental_bytes=incremental,
        current_reserved_bytes=current,
        forecast_peak_reserved_bytes=peak,
        device_total_memory_bytes=total,
        hbm_ceiling_bytes=ceiling,
        admitted=allocated + incremental <= ceiling and incremental <= available,
        _admission_memory=admission_memory,
    )


def _eligibility_commitment(problem: V1SignedWorkLeastWorkCUDAProblem) -> str:
    source = problem.source_problem
    digest = _new_stream_digest(
        domain=DEVICE_SHELL_ELIGIBILITY_COMMITMENT_DOMAIN,
        metadata={
            "route_axis_hash": source.route_axis_hash,
            "route_count": source.route_count,
        },
    )
    packed = np.packbits(source.eligible_mask, bitorder="big")
    digest.update(memoryview(packed).cast("B"))
    return "partition-free-v1-device-shell-eligibility-" + digest.hexdigest()


def _verify_label_identity(
    *,
    labels: object,
    branch_bundle: V1SignedWorkDeviceShellExecutionBundle,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    chunk_size: int,
) -> None:
    ladder = branch_bundle.frozen_ladder
    if type(ladder) is not V1DeviceShellFrozenLadderReceipt:
        raise TypeError("hard-F/Q replay requires an exact frozen ladder receipt")
    digest = _new_stream_digest(
        domain=DEVICE_SHELL_LABEL_COMMITMENT_DOMAIN,
        metadata={
            "route_axis_hash": ladder.route_axis_hash,
            "route_count": ladder.route_count,
            "endpoint_count": ladder.endpoint_count,
        },
    )
    shell_counts = np.zeros(ladder.endpoint_count, dtype=np.int64)
    block_shell_counts = np.zeros(
        (len(ladder.block_capacities), ladder.endpoint_count), dtype=np.int64
    )
    for start in range(0, ladder.route_count, chunk_size):
        stop = min(ladder.route_count, start + chunk_size)
        if isinstance(labels, np.ndarray):
            slab = np.asarray(labels[start:stop], dtype=np.uint8)
        else:
            slab = (
                labels[start:stop]
                .detach()
                .to(device="cpu", non_blocking=False)
                .contiguous()
                .numpy()
            )
        if slab.shape != (stop - start,) or slab.dtype != np.dtype(np.uint8):
            _refuse(
                V1SignedWorkHardFQReplayRefusalCode.LABEL_IDENTITY_MISMATCH,
                "resident label slab is malformed",
            )
        if slab.size and int(np.max(slab)) > ladder.endpoint_count:
            _refuse(
                V1SignedWorkHardFQReplayRefusalCode.LABEL_IDENTITY_MISMATCH,
                "resident label exceeds the registered endpoint count",
            )
        digest.update(memoryview(np.ascontiguousarray(slab)).cast("B"))
        histogram = np.bincount(slab, minlength=ladder.endpoint_count + 1)
        shell_counts += histogram[1 : ladder.endpoint_count + 1]
        for block_index, block in enumerate(ladder.block_capacities):
            overlap_start = max(start, block.start)
            overlap_stop = min(stop, block.stop)
            if overlap_start >= overlap_stop:
                continue
            block_slab = slab[overlap_start - start : overlap_stop - start]
            block_histogram = np.bincount(
                block_slab, minlength=ladder.endpoint_count + 1
            )
            block_shell_counts[block_index] += block_histogram[
                1 : ladder.endpoint_count + 1
            ]
    observed_commitment = (
        "partition-free-v1-device-shell-labels-" + digest.hexdigest()
    )
    observed_shells = tuple(int(value) for value in shell_counts)
    observed_prefixes = tuple(
        tuple(int(value) for value in np.cumsum(row))
        for row in block_shell_counts
    )
    if (
        observed_commitment != ladder.label_commitment
        or observed_shells != ladder.shell_counts
        or tuple(int(value) for value in np.cumsum(shell_counts))
        != ladder.cumulative_counts
        or observed_prefixes != ladder.block_prefix_counts
        or ladder.eligibility_commitment != _eligibility_commitment(problem)
        or branch_bundle.label_handle.route_count != ladder.route_count
        or branch_bundle.label_handle.backend_kind != ladder.handle_kind
        or branch_bundle.label_handle.device_identity != ladder.device_identity
    ):
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.LABEL_IDENTITY_MISMATCH,
            "resident labels do not replay their complete frozen receipt",
        )


@dataclass(frozen=True, slots=True)
class _LaneSummary:
    resource: np.ndarray
    signed_f_cells: np.ndarray
    signed_f_envelope: np.ndarray
    factor_projection: np.ndarray
    factor_absolute_projection: np.ndarray
    diagonal_projection: np.ndarray
    residual_linear: np.ndarray


def _tiled_singleton_rung_sums(
    *,
    labels: object,
    endpoint_labels: tuple[int, ...],
    singleton: Any,
    route_count: int,
    route_tile_size: int,
) -> np.ndarray:
    """Bounded endpoint query: sum the frozen singleton work over each rung.

    ``singleton`` is the geometry lease's corrected singleton-work tensor
    (nonnegative by custody).  The reduction mirrors the lane summary's
    membership matmul and transfers only ``len(endpoint_labels)`` scalars.
    """

    if (
        not torch.is_tensor(singleton)
        or singleton.dtype != torch.float64
        or tuple(singleton.shape) != (route_count,)
    ):
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.STALE_IDENTITY,
            "singleton-work source is not one exact FP64 route vector",
        )
    device = singleton.device
    if bool(torch.any(singleton < 0.0).item()):
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.STALE_IDENTITY,
            "singleton-work source lost its nonnegativity custody invariant",
        )
    endpoint_t = torch.tensor(endpoint_labels, dtype=torch.uint8, device=device)
    resident_labels = torch.as_tensor(labels, device=device)
    endpoints = endpoint_t.reshape(1, -1)
    projection = torch.zeros(
        len(endpoint_labels), dtype=torch.float64, device=device
    )
    for start in range(0, route_count, route_tile_size):
        stop = min(route_count, start + route_tile_size)
        label_tile = resident_labels[start:stop].reshape(-1, 1)
        membership = ((label_tile > 0) & (label_tile <= endpoints)).to(
            dtype=torch.float64
        )
        projection += torch.matmul(singleton[start:stop], membership)
        del label_tile, membership
    values = projection.detach().cpu().numpy().astype(np.float64, copy=True)
    if not np.isfinite(values).all():
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.NUMERICAL_REFUSAL,
            "singleton-work rung sums contain a nonfinite value",
        )
    return values


def _source_tensors_for_cpu(problem: V1SignedWorkLeastWorkCUDAProblem) -> dict[str, Any]:
    if torch is None:
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.CUDA_REQUIRED,
            "PyTorch is required by the private operator reference",
        )
    source = problem.source_problem
    return {
        "resource": torch.tensor(source.resource, dtype=torch.float64),
        "factor": torch.tensor(source.low_rank_factor, dtype=torch.float64),
        "sqrt_residual": torch.tensor(
            np.sqrt(source.residual_diagonal), dtype=torch.float64
        ),
    }


def _tiled_lane_summary(
    *,
    labels: object,
    endpoint_labels: tuple[int, ...],
    problem: V1SignedWorkLeastWorkCUDAProblem,
    source_tensors: dict[str, Any],
    route_tile_size: int,
) -> _LaneSummary:
    operator = problem.signed_work_operator
    endpoint_t = torch.tensor(
        endpoint_labels,
        dtype=torch.uint8,
        device=source_tensors["resource"].device,
    )
    signed_cells, absolute_cells = operator.forward_from_first_inclusion_labels(
        labels, endpoint_t
    )
    context_weights = operator.context_weights.reshape(-1, 1, 1)
    signed_f_cells_t = torch.sum(
        context_weights * torch.abs(signed_cells), dim=(0, 1)
    )
    signed_f_envelope_t = torch.sum(
        context_weights * absolute_cells, dim=(0, 1)
    )
    del signed_cells, absolute_cells
    factor = source_tensors["factor"]
    resource = source_tensors["resource"]
    sqrt_residual = source_tensors["sqrt_residual"]
    device = resource.device
    lanes = len(endpoint_labels)
    rank = int(factor.shape[0])
    factor_projection = torch.zeros((rank, lanes), dtype=torch.float64, device=device)
    factor_absolute = torch.zeros_like(factor_projection)
    resource_projection = torch.zeros(lanes, dtype=torch.float64, device=device)
    diagonal_projection = torch.zeros(lanes, dtype=torch.float64, device=device)
    residual_projection = torch.zeros(lanes, dtype=torch.float64, device=device)
    residual_linear = operator.residual_linear
    resident_labels = torch.as_tensor(labels, device=device)
    endpoints = endpoint_t.reshape(1, -1)
    for start in range(0, problem.source_problem.route_count, route_tile_size):
        stop = min(problem.source_problem.route_count, start + route_tile_size)
        label_tile = resident_labels[start:stop].reshape(-1, 1)
        membership = ((label_tile > 0) & (label_tile <= endpoints)).to(
            dtype=torch.float64
        )
        resource_projection += torch.matmul(resource[start:stop], membership)
        diagonal_projection += torch.matmul(
            sqrt_residual[start:stop], membership
        )
        residual_projection += torch.matmul(residual_linear[start:stop], membership)
        if rank:
            factor_tile = factor[:, start:stop]
            factor_projection += torch.matmul(factor_tile, membership)
            factor_absolute += torch.matmul(torch.abs(factor_tile), membership)
        del label_tile, membership
    arrays = tuple(
        value.detach().cpu().numpy().astype(np.float64, copy=True)
        for value in (
            resource_projection,
            signed_f_cells_t,
            signed_f_envelope_t,
            factor_projection,
            factor_absolute,
            diagonal_projection,
            residual_projection,
        )
    )
    if any(not np.isfinite(value).all() for value in arrays):
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.NUMERICAL_REFUSAL,
            "hard-F/Q device summary contains a nonfinite value",
        )
    return _LaneSummary(*arrays)


def _endpoint_from_summary(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    branch: V1RelaxedKKTBranch,
    ladder: V1DeviceShellFrozenLadderReceipt,
    endpoint_index: int,
    summary: _LaneSummary,
    lane_index: int,
) -> V1SignedWorkHardFQEndpointReceipt:
    source = problem.source_problem
    action = ladder.actions[endpoint_index]
    count = action.cumulative_removed_count
    # A fused multiply-add needs one rounding per term; using twice the route
    # axis also covers non-FMA multiply-plus-add implementations and bounded
    # tile accumulator merges without depending on a backend contraction tree.
    route_ops = max(
        1,
        2 * source.route_count + 2 * source.rank + 5,
    )
    outer_ops = max(
        1,
        4 * source.route_count
        + 2
        * problem.signed_work_operator.context_count
        * problem.signed_work_operator.fiber_count
        + 2 * source.rank
        + 8,
    )
    resource_nominal = float(summary.resource[lane_index])
    resource_lower, resource_upper = _nonnegative_interval_from_nominal(
        nominal=resource_nominal,
        absolute_envelope=resource_nominal,
        operation_count=route_ops,
    )
    cells_nominal = float(summary.signed_f_cells[lane_index])
    cells_envelope = float(summary.signed_f_envelope[lane_index])
    residual_nominal = float(summary.residual_linear[lane_index])
    raw_nominal = cells_nominal + residual_nominal
    raw_lower, raw_upper = _nonnegative_interval_from_nominal(
        nominal=raw_nominal,
        absolute_envelope=cells_envelope + residual_nominal,
        operation_count=outer_ops,
    )
    cut_count = problem.signed_work_operator.native_fiber_axis_receipt.cut_count
    f_lower, f_upper = _divide_nonnegative_interval(
        raw_lower, raw_upper, cut_count
    )
    f_nominal = raw_nominal / float(cut_count)

    low_nominal_values = summary.factor_projection[:, lane_index]
    low_absolute_values = summary.factor_absolute_projection[:, lane_index]
    low_intervals = tuple(
        _signed_sum_interval(
            nominal=float(value),
            absolute_envelope=float(envelope),
            operation_count=route_ops,
        )
        for value, envelope in zip(
            low_nominal_values, low_absolute_values, strict=True
        )
    )
    low_square_intervals = tuple(_square_interval(*item) for item in low_intervals)
    low_lower, low_upper = _sum_nonnegative_intervals(low_square_intervals)
    low_nominal = float(np.dot(low_nominal_values, low_nominal_values))
    low_lower = min(low_lower, low_nominal)
    low_upper = max(low_upper, low_nominal)

    rho = source.residual_spectral_bound
    spectral_nominal = float(rho * count)
    spectral_lower = _down_multiply_nonnegative(rho, float(count))
    spectral_upper = _up_multiply(rho, float(count))

    diagonal_projection = float(summary.diagonal_projection[lane_index])
    diagonal_projection_lower, diagonal_projection_upper = (
        _nonnegative_interval_from_nominal(
            nominal=diagonal_projection,
            absolute_envelope=diagonal_projection,
            operation_count=route_ops,
        )
    )
    diagonal_lower, diagonal_upper = _square_interval(
        diagonal_projection_lower, diagonal_projection_upper
    )
    diagonal_nominal = diagonal_projection * diagonal_projection
    diagonal_lower = min(diagonal_lower, diagonal_nominal)
    diagonal_upper = max(diagonal_upper, diagonal_nominal)

    spectral_q_lower = max(0.0, _down_add(low_lower, spectral_lower))
    spectral_q_upper = _up_add(low_upper, spectral_upper)
    diagonal_q_lower = max(0.0, _down_add(low_lower, diagonal_lower))
    diagonal_q_upper = _up_add(low_upper, diagonal_upper)
    branch_lower, branch_upper = (
        (spectral_q_lower, spectral_q_upper)
        if branch is V1RelaxedKKTBranch.SPECTRAL
        else (diagonal_q_lower, diagonal_q_upper)
    )
    canonical_lower = min(spectral_q_lower, diagonal_q_lower)
    canonical_upper = min(spectral_q_upper, diagonal_q_upper)
    origin_branch_status = (
        V1SignedWorkHardFQEndpointStatus.Q_SAFE_FEASIBLE
        if branch_upper < source.quadratic_budget
        else (
            V1SignedWorkHardFQEndpointStatus.Q_ROBUST_INFEASIBLE
            if branch_lower > source.quadratic_budget
            else V1SignedWorkHardFQEndpointStatus.Q_ENVELOPE_AMBIGUOUS
        )
    )
    status = (
        V1SignedWorkHardFQEndpointStatus.Q_SAFE_FEASIBLE
        if canonical_upper < source.quadratic_budget
        else (
            V1SignedWorkHardFQEndpointStatus.Q_ROBUST_INFEASIBLE
            if canonical_lower > source.quadratic_budget
            else V1SignedWorkHardFQEndpointStatus.Q_ENVELOPE_AMBIGUOUS
        )
    )
    return V1SignedWorkHardFQEndpointReceipt(
        branch=branch,
        endpoint_label=endpoint_index + 1,
        cumulative_removed_count=count,
        frozen_action_receipt_hash=action.receipt_hash,
        packed_action_hash=action.packed_action_hash,
        resulting_mask_hash=action.resulting_mask_hash,
        removed_route_index_hash=action.removed_route_index_hash,
        resource_nominal=resource_nominal,
        resource_lower=resource_lower,
        resource_upper=resource_upper,
        signed_f_nominal=f_nominal,
        signed_f_lower=min(f_lower, f_nominal),
        signed_f_upper=max(f_upper, f_nominal),
        low_rank_q_nominal=low_nominal,
        low_rank_q_lower=low_lower,
        low_rank_q_upper=low_upper,
        spectral_residual_q_nominal=spectral_nominal,
        spectral_residual_q_lower=spectral_lower,
        spectral_residual_q_upper=spectral_upper,
        diagonal_residual_q_nominal=diagonal_nominal,
        diagonal_residual_q_lower=diagonal_lower,
        diagonal_residual_q_upper=diagonal_upper,
        spectral_q_lower=spectral_q_lower,
        spectral_q_upper=spectral_q_upper,
        diagonal_q_lower=diagonal_q_lower,
        diagonal_q_upper=diagonal_q_upper,
        branch_q_lower=branch_lower,
        branch_q_upper=branch_upper,
        canonical_min_q_lower=canonical_lower,
        canonical_min_q_upper=canonical_upper,
        quadratic_budget=source.quadratic_budget,
        origin_branch_status=origin_branch_status,
        status=status,
    )


def _replay_branch(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    execution: V1SignedWorkLeastWorkCUDAExecutionBundle
    | V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle,
    shell_source_hash: str,
    branch_bundle: V1SignedWorkDeviceShellExecutionBundle,
    source_tensors: dict[str, Any],
    design: V1SignedWorkHardFQReplayDesign,
    forecast: V1SignedWorkHardFQHBMForecast,
) -> V1SignedWorkHardFQBranchReceipt:
    branch = branch_bundle.execution_receipt.selected_branch
    ladder = branch_bundle.frozen_ladder
    endpoints: list[V1SignedWorkHardFQEndpointReceipt] = []
    with branch_bundle.label_handle.borrow_for_registered_operation(
        operation=V1DeviceShellBorrowOperation.FQ_HARD_REPLAY
    ) as label_lease:
        labels = label_lease._labels_for_registered_operation(
            operation=V1DeviceShellBorrowOperation.FQ_HARD_REPLAY
        )
        _verify_label_identity(
            labels=labels,
            branch_bundle=branch_bundle,
            problem=problem,
            chunk_size=design.label_verification_chunk_size,
        )
        # PyTorch warns when aliasing the intentionally read-only NumPy label
        # commitment.  The private reference is explicitly tiny, so give its
        # arithmetic path one writable copy while preserving the committed
        # object for verification and lease-lifetime checks.  CUDA remains a
        # zero-copy resident borrow.
        labels_for_math = (
            np.array(labels, dtype=np.uint8, order="C", copy=True)
            if isinstance(labels, np.ndarray)
            else labels
        )
        if not isinstance(labels, np.ndarray):
            if (
                labels.dtype != torch.uint8
                or not labels.is_contiguous()
                or labels.device != source_tensors["resource"].device
            ):
                _refuse(
                    V1SignedWorkHardFQReplayRefusalCode.STALE_IDENTITY,
                    "CUDA labels and retained Q source tensors differ in device custody",
                )
        for start in range(0, ladder.endpoint_count, design.endpoint_lane_count):
            stop = min(ladder.endpoint_count, start + design.endpoint_lane_count)
            endpoint_labels = tuple(range(start + 1, stop + 1))
            summary = _tiled_lane_summary(
                labels=labels_for_math,
                endpoint_labels=endpoint_labels,
                problem=problem,
                source_tensors=source_tensors,
                route_tile_size=design.q_route_tile_size,
            )
            for lane_index, endpoint_index in enumerate(range(start, stop)):
                endpoints.append(
                    _endpoint_from_summary(
                        problem=problem,
                        branch=branch,
                        ladder=ladder,
                        endpoint_index=endpoint_index,
                        summary=summary,
                        lane_index=lane_index,
                    )
                )
    state_hash = execution.state_identity_hash
    if state_hash is None:  # pragma: no cover - static validation already closes this.
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.SOURCE_CUSTODY_UNAVAILABLE,
            "least-work state identity disappeared during replay",
        )
    return V1SignedWorkHardFQBranchReceipt(
        branch=branch,
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        least_work_execution_hash=execution.execution_hash,
        state_identity_hash=state_hash,
        shell_family_hash=_identity(shell_source_hash, name="shell source hash"),
        shell_execution_hash=branch_bundle.execution_hash,
        shell_execution_receipt_hash=branch_bundle.execution_receipt.receipt_hash,
        branch_identity_receipt_hash=branch_bundle.branch_identity.receipt_hash,
        frozen_ladder_receipt_hash=ladder.receipt_hash,
        label_commitment=ladder.label_commitment,
        operator_receipt_hash=problem.signed_work_operator.receipt_hash,
        design_hash=design.design_hash,
        hbm_forecast_hash=forecast.forecast_hash,
        endpoints=tuple(endpoints),
        label_identity_fully_verified=True,
        every_registered_endpoint_replayed=(len(endpoints) == ladder.endpoint_count),
        production_cuda_execution=execution.production_cuda_execution,
        private_cpu_reference_execution=execution.private_cpu_reference_execution,
    )


def _validate_single_branch_identities(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    execution: V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle,
    shell: V1SignedWorkDeviceShellExecutionBundle,
    design: V1SignedWorkHardFQReplayDesign,
) -> None:
    if type(problem) is not V1SignedWorkLeastWorkCUDAProblem:
        raise TypeError("single-branch hard replay requires exact problem")
    if type(policy) is not V1SignedWorkLeastWorkCUDAPolicy:
        raise TypeError("single-branch hard replay requires exact policy")
    if type(execution) is not V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle:
        raise TypeError("single-branch hard replay requires exact execution")
    if type(shell) is not V1SignedWorkDeviceShellExecutionBundle:
        raise TypeError("single-branch hard replay requires exact shell")
    if type(design) is not V1SignedWorkHardFQReplayDesign:
        raise TypeError("single-branch hard replay requires exact design")
    problem.assert_integrity()
    shell_receipt = shell.execution_receipt
    if (
        execution.result.problem_hash != problem.problem_hash
        or execution.result.policy_hash != policy.policy_hash
        or not execution.result.branch_converged
        or execution.state_handle is None
        or shell_receipt.problem_hash != problem.problem_hash
        or shell_receipt.policy_hash != policy.policy_hash
        or shell_receipt.least_work_execution_hash != execution.execution_hash
        or shell_receipt.state_identity_hash != execution.state_identity_hash
        or shell_receipt.selected_branch is not execution.branch
        or shell.branch_identity.signed_work_branch_receipt_hash
        != execution.branch_receipt.receipt_hash
        or shell_receipt.source_geometry_hash
        != problem.source_problem.source_geometry_hash
        or shell_receipt.source_registration_hash != problem.source_registration_hash
        or shell_receipt.registered_d_coordinate_hash
        != problem.registered_d_coordinate_hash
    ):
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.STALE_IDENTITY,
            "single-branch problem, solver, and shell do not share one chart/branch",
        )
    forbidden = (
        execution.result.nonrequested_branch_authority,
        execution.result.c_stream_authority,
        execution.result.calibration_authority,
        execution.result.bce_oracle_authority,
        execution.result.mask_commit_authority,
        execution.result.terminal_authority,
        execution.result.pareto_claim_authority,
        shell_receipt.fqr_authority,
        shell_receipt.c_stream_authority,
        shell_receipt.calibration_authority,
        shell_receipt.oracle_authority,
        shell_receipt.mask_commit_authority,
        shell_receipt.terminal_authority,
        shell_receipt.pareto_authority,
    )
    if any(forbidden):
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.AUTHORITY_VIOLATION,
            "single-branch replay input crossed pre-C authority",
        )
    source = problem.source_problem
    if not problem.source_bundle_authoritative or not np.array_equal(
        source.resource,
        np.ones(source.route_count, dtype=np.float64),
    ):
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.NONUNIT_RESOURCE_ENDPOINT_PLAN,
            "single-branch hard replay requires authoritative unit route resource",
        )
    operator = problem.signed_work_operator
    if design.endpoint_lane_count > operator.design.maximum_lanes:
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.INVALID_TYPE,
            "replay endpoint lanes exceed the signed-work operator design",
        )
    if design.production_cuda_required:
        if (
            not execution.production_cuda_execution
            or not operator.receipt.production_cuda_execution
            or shell.label_handle.backend_kind != "torch_cuda"
        ):
            _refuse(
                V1SignedWorkHardFQReplayRefusalCode.CUDA_REQUIRED,
                "production single-branch hard replay requires CUDA custody",
            )
    elif (
        not design.allow_private_cpu_reference
        or source.route_count > design.private_reference_route_limit
        or not execution.private_cpu_reference_execution
        or not operator.receipt.private_cpu_reference_execution
        or shell.label_handle.backend_kind != "numpy_reference"
    ):
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.CPU_REFERENCE_SCALE_EXCEEDED,
            "private single-branch replay is restricted to one tiny CPU fixture",
        )
    if (
        design.require_deterministic_algorithms
        and execution.production_cuda_execution
        and (torch is None or not torch.are_deterministic_algorithms_enabled())
    ):
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.DETERMINISM_REQUIRED,
            "single-branch CUDA replay requires deterministic torch algorithms",
        )
    if (
        design.require_cublas_deterministic_workspace
        and execution.production_cuda_execution
        and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}
    ):
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.DETERMINISM_REQUIRED,
            "single-branch CUDA replay requires deterministic cuBLAS workspace",
        )


def estimate_v1_signed_work_hard_fq_single_branch_hbm(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    execution: V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle,
    shell: V1SignedWorkDeviceShellExecutionBundle,
    design: V1SignedWorkHardFQReplayDesign = V1SignedWorkHardFQReplayDesign(),
    current_reserved_bytes: int | None = None,
    device_total_memory_bytes: int | None = None,
) -> V1SignedWorkHardFQHBMForecast:
    """Conservative bounded forecast for one supported origin branch."""

    try:
        _validate_single_branch_identities(
            problem=problem,
            policy=policy,
            execution=execution,
            shell=shell,
            design=design,
        )
    except V1DeviceShellRefusal as error:
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.LABEL_IDENTITY_MISMATCH,
            f"single-branch label custody refused forecast: {error}",
        )
    state = execution.state_handle
    if state is None:  # pragma: no cover - validation closes this.
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.SOURCE_CUSTODY_UNAVAILABLE,
            "single-branch state disappeared before forecast",
        )
    source = problem.source_problem
    operator = problem.signed_work_operator
    endpoint_count = shell.frozen_ladder.endpoint_count
    lanes = min(endpoint_count, design.endpoint_lane_count)
    route_count = source.route_count
    rank = source.rank
    q_tile = min(route_count, design.q_route_tile_size)
    # Reuse the established v1 forecast schema and its conservative two-label
    # envelope.  A single branch actually owns one label vector, so this is an
    # overestimate, never an unreceipted memory relaxation.
    labels_bytes = 2 * route_count * _U8_BYTES
    cells = 2 * operator.context_count * operator.fiber_count * lanes * _F64_BYTES
    operator_tile = 2 * operator.forecast.temporary_tile_bytes_upper
    q_membership = q_tile * lanes * (3 * _BOOL_BYTES + _F64_BYTES)
    q_factor = rank * q_tile * _F64_BYTES
    q_compact = (2 * rank * lanes + 5 * lanes) * _F64_BYTES
    summary = 32 * lanes * _F64_BYTES
    f_peak = cells + operator_tile + summary
    q_peak = q_membership + q_factor + q_compact + summary
    working = max(f_peak, q_peak)
    guard = max(
        _MINIMUM_ALLOCATOR_GUARD_BYTES,
        math.ceil(_ALLOCATOR_GUARD_FRACTION * working),
    )
    incremental = working + guard
    admission_memory = None
    if current_reserved_bytes is None or device_total_memory_bytes is None:
        if execution.production_cuda_execution:
            if torch is None or not torch.cuda.is_available():
                _refuse(
                    V1SignedWorkHardFQReplayRefusalCode.CUDA_REQUIRED,
                    "CUDA forecast requested without CUDA runtime",
                )
            device = torch.device(state.device)
            admission_memory = (cuda_reusable_memory_bytes(device, framework=torch), int(torch.cuda.memory_allocated(device)))
            if current_reserved_bytes is None:
                current_reserved_bytes = int(torch.cuda.memory_reserved(device))
            if device_total_memory_bytes is None:
                device_total_memory_bytes = int(
                    torch.cuda.get_device_properties(device).total_memory
                )
        else:
            if current_reserved_bytes is None:
                current_reserved_bytes = 0
            if device_total_memory_bytes is None:
                device_total_memory_bytes = max(1 << 30, 4 * incremental)
    current = _nonnegative_int(current_reserved_bytes, name="current reserved bytes")
    total = _positive_int(device_total_memory_bytes, name="device total memory bytes")
    ceiling = math.floor(design.hbm_ceiling_fraction * total)
    peak = current + incremental
    available, allocated = admission_memory or (total, current)
    return V1SignedWorkHardFQHBMForecast(
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        least_work_execution_hash=execution.execution_hash,
        shell_family_hash=shell.execution_hash,
        design_hash=design.design_hash,
        device=state.device,
        route_count=route_count,
        context_count=operator.context_count,
        fiber_count=operator.fiber_count,
        response_rank=rank,
        endpoint_count=endpoint_count,
        lane_count=lanes,
        q_route_tile_size=design.q_route_tile_size,
        resident_two_branch_label_bytes=labels_bytes,
        f_signed_and_envelope_cell_bytes=cells,
        f_operator_tile_workspace_upper_bytes=operator_tile,
        q_membership_tile_bytes=q_membership,
        q_absolute_factor_tile_bytes=q_factor,
        q_compact_projection_bytes=q_compact,
        compact_lane_summary_bytes=summary,
        f_peak_workspace_bytes=f_peak,
        q_peak_workspace_bytes=q_peak,
        allocator_guard_bytes=guard,
        forecast_incremental_bytes=incremental,
        current_reserved_bytes=current,
        forecast_peak_reserved_bytes=peak,
        device_total_memory_bytes=total,
        hbm_ceiling_bytes=ceiling,
        admitted=allocated + incremental <= ceiling and incremental <= available,
        _admission_memory=admission_memory,
    )


def replay_v1_signed_work_hard_fq_single_branch_cuda(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    execution: V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle,
    shell: V1SignedWorkDeviceShellExecutionBundle,
    design: V1SignedWorkHardFQReplayDesign = V1SignedWorkHardFQReplayDesign(),
    current_reserved_bytes: int | None = None,
    device_total_memory_bytes: int | None = None,
) -> V1SignedWorkHardFQSingleBranchBundle:
    """Replay complete signed F and both Q envelopes for one origin branch."""

    forecast = estimate_v1_signed_work_hard_fq_single_branch_hbm(
        problem=problem,
        policy=policy,
        execution=execution,
        shell=shell,
        design=design,
        current_reserved_bytes=current_reserved_bytes,
        device_total_memory_bytes=device_total_memory_bytes,
    )
    if not forecast.admitted:
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.HBM_FORECAST_EXCEEDED,
            "single-branch hard replay exceeds the HBM ceiling",
        )
    state = execution.state_handle
    if state is None:  # pragma: no cover - forecast validates this.
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.SOURCE_CUSTODY_UNAVAILABLE,
            "single-branch theorem state disappeared",
        )
    try:
        with state.borrow_for_hard_fq_replay(
            expected_problem_hash=problem.problem_hash,
            expected_policy_hash=policy.policy_hash,
            expected_branch_receipt_hashes=(
                execution.result.solver_state_binding_hashes
            ),
        ) as source_lease:
            if execution.production_cuda_execution:
                source_tensors = {
                    name: source_lease.source_tensor(name)
                    for name in ("resource", "factor", "sqrt_residual")
                }
            else:
                source_tensors = _source_tensors_for_cpu(problem)
            receipt = _replay_branch(
                problem=problem,
                policy=policy,
                execution=execution,
                shell_source_hash=shell.execution_hash,
                branch_bundle=shell,
                source_tensors=source_tensors,
                design=design,
                forecast=forecast,
            )
    except V1SignedWorkHardFQReplayRefusal:
        raise
    except V1SignedWorkLeastWorkCUDARefusal as error:
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.STALE_IDENTITY,
            f"single-branch theorem-state custody refused hard replay: {error}",
        )
    except V1DeviceShellRefusal as error:
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.LABEL_IDENTITY_MISMATCH,
            f"single-branch label custody refused hard replay: {error}",
        )
    except V1SignedWorkOperatorCUDARefusal as error:
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.NUMERICAL_REFUSAL,
            f"single-branch signed-F operator refused hard replay: {error}",
        )
    return V1SignedWorkHardFQSingleBranchBundle(
        receipt=receipt,
        hbm_forecast=forecast,
    )


def replay_v1_signed_work_hard_fq_family_cuda(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    execution: V1SignedWorkLeastWorkCUDAExecutionBundle,
    shell_family: V1SignedWorkDeviceShellFamilyBundle,
    design: V1SignedWorkHardFQReplayDesign = V1SignedWorkHardFQReplayDesign(),
    current_reserved_bytes: int | None = None,
    device_total_memory_bytes: int | None = None,
) -> V1SignedWorkHardFQFamilyBundle:
    """Replay complete signed F and both Q envelopes for every frozen rung."""

    forecast = estimate_v1_signed_work_hard_fq_replay_hbm(
        problem=problem,
        policy=policy,
        execution=execution,
        shell_family=shell_family,
        design=design,
        current_reserved_bytes=current_reserved_bytes,
        device_total_memory_bytes=device_total_memory_bytes,
    )
    if not forecast.admitted:
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.HBM_FORECAST_EXCEEDED,
            "hard-F/Q replay exceeds the registered 85% HBM ceiling",
        )
    if execution.production_cuda_execution:
        device = torch.device(forecast.device)
        incremental = forecast.forecast_incremental_bytes
        if current_reserved_bytes is None or device_total_memory_bytes is None:
            admitted = (
                int(torch.cuda.memory_allocated(device)) + incremental <= forecast.hbm_ceiling_bytes
                and incremental <= cuda_reusable_memory_bytes(device, framework=torch)
            )
        else:
            admitted = int(torch.cuda.memory_reserved(device)) <= forecast.current_reserved_bytes
        if not admitted:
            _refuse(
                V1SignedWorkHardFQReplayRefusalCode.HBM_FORECAST_EXCEEDED,
                "live CUDA allocation exceeds the frozen pre-replay memory budget",
            )
    state = execution.state_handle
    if state is None:  # pragma: no cover - already validated.
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.SOURCE_CUSTODY_UNAVAILABLE,
            "hard-F/Q replay lost theorem-state custody",
        )
    expected_branch_receipts = tuple(
        receipt.receipt_hash for receipt in execution.branch_receipts
    )
    try:
        with state.borrow_for_hard_fq_replay(
            expected_problem_hash=problem.problem_hash,
            expected_policy_hash=policy.policy_hash,
            expected_branch_receipt_hashes=expected_branch_receipts,  # type: ignore[arg-type]
        ) as source_lease:
            if execution.production_cuda_execution:
                source_tensors = {
                    name: source_lease.source_tensor(name)
                    for name in ("resource", "factor", "sqrt_residual")
                }
                expected_device = source_tensors["resource"].device
                if (
                    source_tensors["resource"].dtype != torch.float64
                    or source_tensors["factor"].dtype != torch.float64
                    or source_tensors["sqrt_residual"].dtype != torch.float64
                    or any(
                        tensor.device != expected_device
                        for tensor in source_tensors.values()
                    )
                    or problem.signed_work_operator.context_weights.device
                    != expected_device
                    or tuple(source_tensors["resource"].shape)
                    != (problem.source_problem.route_count,)
                    or tuple(source_tensors["factor"].shape)
                    != (
                        problem.source_problem.rank,
                        problem.source_problem.route_count,
                    )
                    or tuple(source_tensors["sqrt_residual"].shape)
                    != (problem.source_problem.route_count,)
                ):
                    _refuse(
                        V1SignedWorkHardFQReplayRefusalCode.STALE_IDENTITY,
                        "retained F/Q source tensors do not share one FP64 device",
                    )
            else:
                # The source-state borrow still guards chart lifetime and exact
                # problem/policy/branch identities.  Tiny reference arithmetic
                # uses immutable source arrays because manually constructed
                # focused state handles intentionally cannot lend production
                # source-tensor authority.
                source_tensors = _source_tensors_for_cpu(problem)
            spectral = _replay_branch(
                problem=problem,
                policy=policy,
                execution=execution,
                shell_source_hash=shell_family.family_hash,
                branch_bundle=shell_family.spectral,
                source_tensors=source_tensors,
                design=design,
                forecast=forecast,
            )
            diagonal = _replay_branch(
                problem=problem,
                policy=policy,
                execution=execution,
                shell_source_hash=shell_family.family_hash,
                branch_bundle=shell_family.diagonal_l1,
                source_tensors=source_tensors,
                design=design,
                forecast=forecast,
            )
    except V1SignedWorkHardFQReplayRefusal:
        raise
    except V1SignedWorkLeastWorkCUDARefusal as error:
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.STALE_IDENTITY,
            f"least-work source custody refused hard-F/Q replay: {error}",
        )
    except V1DeviceShellRefusal as error:
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.LABEL_IDENTITY_MISMATCH,
            f"device-shell label custody refused hard-F/Q replay: {error}",
        )
    except V1SignedWorkOperatorCUDARefusal as error:
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.NUMERICAL_REFUSAL,
            f"signed-F operator refused hard replay: {error}",
        )
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            _refuse(
                V1SignedWorkHardFQReplayRefusalCode.HBM_FORECAST_EXCEEDED,
                "measured replay allocation exceeded available device memory",
            )
        raise
    state_hash = execution.state_identity_hash
    if state_hash is None:  # pragma: no cover
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.SOURCE_CUSTODY_UNAVAILABLE,
            "hard-F/Q state identity disappeared after replay",
        )
    receipt = V1SignedWorkHardFQFamilyReceipt(
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        least_work_execution_hash=execution.execution_hash,
        state_identity_hash=state_hash,
        shell_family_hash=shell_family.family_hash,
        design_hash=design.design_hash,
        hbm_forecast_hash=forecast.forecast_hash,
        spectral=spectral,
        diagonal_l1=diagonal,
        common_endpoint_counts=shell_family.spectral.frozen_ladder.cumulative_counts,
    )
    return V1SignedWorkHardFQFamilyBundle(
        receipt=receipt,
        hbm_forecast=forecast,
    )


# ---------------------------------------------------------------------------
# Zero-copy DeviceProblemV2 replay
# ---------------------------------------------------------------------------

SIGNED_WORK_HARD_FQ_DEVICE_ENDPOINT_SCHEMA_V2: Final = (
    "partition_free_v1_signed_work_hard_fq_device_endpoint_v2_1"
)
SIGNED_WORK_HARD_FQ_DEVICE_BRANCH_SCHEMA_V2: Final = (
    "partition_free_v1_signed_work_hard_fq_device_branch_v2"
)
SIGNED_WORK_HARD_FQ_DEVICE_FAMILY_SCHEMA_V2: Final = (
    "partition_free_v1_signed_work_hard_fq_device_family_v2"
)
SIGNED_WORK_HARD_FQ_DEVICE_BUNDLE_SCHEMA_V2: Final = (
    "partition_free_v1_signed_work_hard_fq_device_bundle_v2"
)
_DEVICE_V2_PREFIX: Final = "partition-free-v1-signed-work-hard-fq-device-v2-"


@dataclass(frozen=True, slots=True)
class V1SignedWorkHardFQDeviceEndpointReceiptV2:
    """One quantitative replay bound under a live device action commitment.

    ``device_action_commitment_hash`` is intentionally not a packed-mask,
    resulting-mask, or removed-index hash.  Those durable identities require
    a later bounded query for the selected endpoint and cannot be fabricated
    by a no-D2H proposal replay.
    """

    branch: V1RelaxedKKTBranch
    endpoint_label: int
    cumulative_removed_count: int
    device_action_commitment_hash: str
    resource_nominal: float
    resource_lower: float
    resource_upper: float
    signed_f_nominal: float
    signed_f_lower: float
    signed_f_upper: float
    low_rank_q_nominal: float
    low_rank_q_lower: float
    low_rank_q_upper: float
    spectral_residual_q_nominal: float
    spectral_residual_q_lower: float
    spectral_residual_q_upper: float
    diagonal_residual_q_nominal: float
    diagonal_residual_q_lower: float
    diagonal_residual_q_upper: float
    spectral_q_lower: float
    spectral_q_upper: float
    diagonal_q_lower: float
    diagonal_q_upper: float
    branch_q_lower: float
    branch_q_upper: float
    canonical_min_q_lower: float
    canonical_min_q_upper: float
    quadratic_budget: float
    # Bounded endpoint query: the singleton-work screen summed over this
    # rung's label set.  It is the denominator of the preregistered
    # cancellation ratio F(1_S) / (A^T 1_S) and carries no proposal or
    # commit authority; F itself is never substituted by it.
    singleton_work_sum_nominal: float
    singleton_work_sum_lower: float
    singleton_work_sum_upper: float
    origin_branch_status: V1SignedWorkHardFQEndpointStatus
    status: V1SignedWorkHardFQEndpointStatus
    singleton_work_sum_query_operation: str = "bounded_endpoint_query"
    singleton_work_sum_semantics: str = (
        "stratum_weighted_absolute_corrected_evidence_mean_over_rung_routes"
    )
    legacy_packed_action_identity_available: bool = False
    bounded_endpoint_query_required_for_physical_identity: bool = True
    route_scale_d2h_bytes: int = 0
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_HARD_FQ_AUTHORITY
    schema: str = SIGNED_WORK_HARD_FQ_DEVICE_ENDPOINT_SCHEMA_V2

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("v2 hard-F/Q endpoint branch must be exact")
        object.__setattr__(
            self, "endpoint_label", _positive_int(self.endpoint_label, name="endpoint label")
        )
        object.__setattr__(
            self,
            "cumulative_removed_count",
            _positive_int(self.cumulative_removed_count, name="cumulative removed count"),
        )
        object.__setattr__(
            self,
            "device_action_commitment_hash",
            _identity(
                self.device_action_commitment_hash,
                name="device action commitment hash",
            ),
        )
        scalar_names = (
            "resource_nominal",
            "resource_lower",
            "resource_upper",
            "signed_f_nominal",
            "signed_f_lower",
            "signed_f_upper",
            "low_rank_q_nominal",
            "low_rank_q_lower",
            "low_rank_q_upper",
            "spectral_residual_q_nominal",
            "spectral_residual_q_lower",
            "spectral_residual_q_upper",
            "diagonal_residual_q_nominal",
            "diagonal_residual_q_lower",
            "diagonal_residual_q_upper",
            "spectral_q_lower",
            "spectral_q_upper",
            "diagonal_q_lower",
            "diagonal_q_upper",
            "branch_q_lower",
            "branch_q_upper",
            "canonical_min_q_lower",
            "canonical_min_q_upper",
            "quadratic_budget",
            "singleton_work_sum_nominal",
            "singleton_work_sum_lower",
            "singleton_work_sum_upper",
        )
        for name in scalar_names:
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name=name),
            )
        if (
            self.singleton_work_sum_query_operation != "bounded_endpoint_query"
            or type(self.singleton_work_sum_semantics) is not str
            or not self.singleton_work_sum_semantics
        ):
            raise ValueError("v2 hard-F/Q singleton-work sum lost its query binding")
        if any(
            not (lower <= nominal <= upper)
            for lower, nominal, upper in (
                (self.resource_lower, self.resource_nominal, self.resource_upper),
                (self.signed_f_lower, self.signed_f_nominal, self.signed_f_upper),
                (
                    self.singleton_work_sum_lower,
                    self.singleton_work_sum_nominal,
                    self.singleton_work_sum_upper,
                ),
                (self.low_rank_q_lower, self.low_rank_q_nominal, self.low_rank_q_upper),
                (
                    self.spectral_residual_q_lower,
                    self.spectral_residual_q_nominal,
                    self.spectral_residual_q_upper,
                ),
                (
                    self.diagonal_residual_q_lower,
                    self.diagonal_residual_q_nominal,
                    self.diagonal_residual_q_upper,
                ),
            )
        ):
            raise ValueError("v2 hard-F/Q nominal escaped an outward enclosure")
        if (
            type(self.origin_branch_status) is not V1SignedWorkHardFQEndpointStatus
            or type(self.status) is not V1SignedWorkHardFQEndpointStatus
        ):
            raise TypeError("v2 hard-F/Q endpoint status must be exact")
        closed = (
            self.legacy_packed_action_identity_available,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if (
            any(type(item) is not bool for item in closed)
            or any(closed)
            or self.bounded_endpoint_query_required_for_physical_identity is not True
            or self.route_scale_d2h_bytes != 0
            or self.authority_scope != SIGNED_WORK_HARD_FQ_AUTHORITY
            or self.schema != SIGNED_WORK_HARD_FQ_DEVICE_ENDPOINT_SCHEMA_V2
        ):
            _refuse(
                V1SignedWorkHardFQReplayRefusalCode.AUTHORITY_VIOLATION,
                "v2 hard-F/Q endpoint impersonated a packed action or downstream authority",
            )

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_DEVICE_V2_PREFIX + "endpoint-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkHardFQDeviceBranchReceiptV2:
    branch: V1RelaxedKKTBranch
    problem_hash: str
    policy_hash: str
    least_work_execution_hash: str
    state_identity_hash: str
    shell_family_hash: str
    shell_receipt_hash: str
    operator_receipt_hash: str
    design_hash: str
    hbm_forecast_hash: str
    endpoints: tuple[V1SignedWorkHardFQDeviceEndpointReceiptV2, ...]
    production_cuda_execution: bool
    private_cpu_reference_execution: bool
    live_label_storage_identity_verified: bool = True
    legacy_full_axis_label_digest_replayed: bool = False
    every_registered_endpoint_replayed: bool = True
    route_by_endpoint_action_matrix_materialized: bool = False
    route_scale_d2h_bytes: int = 0
    c_stream_authority: bool = False
    nonlinear_remainder_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_HARD_FQ_AUTHORITY
    schema: str = SIGNED_WORK_HARD_FQ_DEVICE_BRANCH_SCHEMA_V2

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("v2 hard-F/Q branch must be exact")
        for name in (
            "problem_hash",
            "policy_hash",
            "least_work_execution_hash",
            "state_identity_hash",
            "shell_family_hash",
            "shell_receipt_hash",
            "operator_receipt_hash",
            "design_hash",
            "hbm_forecast_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        endpoints = tuple(self.endpoints)
        if (
            not endpoints
            or any(
                type(item) is not V1SignedWorkHardFQDeviceEndpointReceiptV2
                or item.branch is not self.branch
                for item in endpoints
            )
            or tuple(item.endpoint_label for item in endpoints)
            != tuple(range(1, len(endpoints) + 1))
        ):
            raise TypeError("v2 hard-F/Q branch endpoints are incomplete or reordered")
        object.__setattr__(self, "endpoints", endpoints)
        flags = (
            self.production_cuda_execution,
            self.private_cpu_reference_execution,
            self.live_label_storage_identity_verified,
            self.legacy_full_axis_label_digest_replayed,
            self.every_registered_endpoint_replayed,
            self.route_by_endpoint_action_matrix_materialized,
            self.c_stream_authority,
            self.nonlinear_remainder_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if (
            any(type(item) is not bool for item in flags)
            or flags[0] == flags[1]
            or flags[2:] != (
                True,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            )
            or self.route_scale_d2h_bytes != 0
            or self.authority_scope != SIGNED_WORK_HARD_FQ_AUTHORITY
            or self.schema != SIGNED_WORK_HARD_FQ_DEVICE_BRANCH_SCHEMA_V2
        ):
            _refuse(
                V1SignedWorkHardFQReplayRefusalCode.AUTHORITY_VIOLATION,
                "v2 hard-F/Q branch weakened no-D2H proposal authority",
            )

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_DEVICE_V2_PREFIX + "branch-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkHardFQDeviceFamilyBundleV2:
    problem_hash: str
    policy_hash: str
    least_work_execution_hash: str
    state_identity_hash: str
    shell_family_hash: str
    design_hash: str
    hbm_forecast: V1SignedWorkHardFQHBMForecast
    spectral: V1SignedWorkHardFQDeviceBranchReceiptV2
    diagonal_l1: V1SignedWorkHardFQDeviceBranchReceiptV2
    common_endpoint_counts: tuple[int, ...]
    c_stream_authority: bool = False
    mask_commit_authority: bool = False
    schema: str = SIGNED_WORK_HARD_FQ_DEVICE_BUNDLE_SCHEMA_V2

    def __post_init__(self) -> None:
        for name in (
            "problem_hash",
            "policy_hash",
            "least_work_execution_hash",
            "state_identity_hash",
            "shell_family_hash",
            "design_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if type(self.hbm_forecast) is not V1SignedWorkHardFQHBMForecast:
            raise TypeError("v2 hard-F/Q family requires exact HBM forecast")
        if (
            type(self.spectral) is not V1SignedWorkHardFQDeviceBranchReceiptV2
            or type(self.diagonal_l1) is not V1SignedWorkHardFQDeviceBranchReceiptV2
            or self.spectral.branch is not V1RelaxedKKTBranch.SPECTRAL
            or self.diagonal_l1.branch is not V1RelaxedKKTBranch.DIAGONAL_L1
        ):
            raise TypeError("v2 hard-F/Q family requires both exact branches")
        counts = tuple(
            _positive_int(item, name="common endpoint count")
            for item in self.common_endpoint_counts
        )
        shared = (
            self.problem_hash,
            self.policy_hash,
            self.least_work_execution_hash,
            self.state_identity_hash,
            self.shell_family_hash,
            self.design_hash,
            self.hbm_forecast.forecast_hash,
        )
        if (
            tuple(item.cumulative_removed_count for item in self.spectral.endpoints)
            != counts
            or tuple(item.cumulative_removed_count for item in self.diagonal_l1.endpoints)
            != counts
            or any(
                (
                    branch.problem_hash,
                    branch.policy_hash,
                    branch.least_work_execution_hash,
                    branch.state_identity_hash,
                    branch.shell_family_hash,
                    branch.design_hash,
                    branch.hbm_forecast_hash,
                )
                != shared
                for branch in (self.spectral, self.diagonal_l1)
            )
            or self.c_stream_authority
            or self.mask_commit_authority
            or self.schema != SIGNED_WORK_HARD_FQ_DEVICE_BUNDLE_SCHEMA_V2
        ):
            raise ValueError("v2 hard-F/Q family identity or authority differs")
        object.__setattr__(self, "common_endpoint_counts", counts)

    @property
    def bundle_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "problem_hash": self.problem_hash,
                "policy_hash": self.policy_hash,
                "least_work_execution_hash": self.least_work_execution_hash,
                "state_identity_hash": self.state_identity_hash,
                "shell_family_hash": self.shell_family_hash,
                "design_hash": self.design_hash,
                "hbm_forecast_hash": self.hbm_forecast.forecast_hash,
                "spectral_receipt_hash": self.spectral.receipt_hash,
                "diagonal_l1_receipt_hash": self.diagonal_l1.receipt_hash,
                "common_endpoint_counts": self.common_endpoint_counts,
                "no_route_scale_d2h": True,
            },
            prefix=_DEVICE_V2_PREFIX + "bundle-",
        )


def _validate_device_v2_replay_inputs(
    *,
    problem: V1SignedWorkLeastWorkCUDADeviceProblemV2,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    execution: V1SignedWorkLeastWorkCUDAExecutionBundle,
    shell_family: V1SignedWorkDeviceShellDeviceFamilyV2,
    design: V1SignedWorkHardFQReplayDesign,
) -> None:
    if type(problem) is not V1SignedWorkLeastWorkCUDADeviceProblemV2:
        raise TypeError("v2 hard-F/Q replay requires exact DeviceProblemV2")
    if type(policy) is not V1SignedWorkLeastWorkCUDAPolicy:
        raise TypeError("v2 hard-F/Q replay requires exact policy")
    if type(execution) is not V1SignedWorkLeastWorkCUDAExecutionBundle:
        raise TypeError("v2 hard-F/Q replay requires exact execution")
    if type(shell_family) is not V1SignedWorkDeviceShellDeviceFamilyV2:
        raise TypeError("v2 hard-F/Q replay requires exact live shell family")
    if type(design) is not V1SignedWorkHardFQReplayDesign:
        raise TypeError("v2 hard-F/Q replay requires exact design")
    state = execution.state_handle
    if (
        state is None
        or execution.result.problem_hash != problem.problem_hash
        or execution.result.policy_hash != policy.policy_hash
        or state.problem_hash != problem.problem_hash
        or state.policy_hash != policy.policy_hash
        or shell_family.problem_hash != problem.problem_hash
        or shell_family.policy_hash != policy.policy_hash
        or shell_family.least_work_execution_hash != execution.execution_hash
        or shell_family.spectral.receipt.state_identity_hash != state.identity_hash
        or shell_family.diagonal_l1.receipt.state_identity_hash != state.identity_hash
        or shell_family.spectral.receipt.source_geometry_hash
        != problem.source_geometry_hash
        or shell_family.diagonal_l1.receipt.source_geometry_hash
        != problem.source_geometry_hash
        or shell_family.spectral.receipt.source_registration_hash
        != problem.source_registration_hash
        or shell_family.diagonal_l1.receipt.source_registration_hash
        != problem.source_registration_hash
        or shell_family.spectral.receipt.registered_d_coordinate_hash
        != problem.registered_d_coordinate_hash
        or shell_family.diagonal_l1.receipt.registered_d_coordinate_hash
        != problem.registered_d_coordinate_hash
        or shell_family.spectral.receipt.route_axis_hash != problem.route_axis_hash
        or shell_family.diagonal_l1.receipt.route_axis_hash != problem.route_axis_hash
        or shell_family.spectral.receipt.route_count != problem.route_count
        or shell_family.diagonal_l1.receipt.route_count != problem.route_count
        or state.eligible_mask_identity_hash != problem.eligible_mask_identity_hash
        or state.required_mask_identity_hash != problem.required_mask_identity_hash
    ):
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.STALE_IDENTITY,
            "v2 problem, theorem state, and shell family do not share one chart",
        )
    if (
        not problem.source_bundle_authoritative
        or execution.result.hard_mask_global_optimality_claimed
        or execution.result.c_stream_authority
        or execution.result.calibration_authority
        or execution.result.bce_oracle_authority
        or execution.result.mask_commit_authority
        or execution.result.terminal_authority
        or execution.result.pareto_claim_authority
        or shell_family.c_stream_authority
        or shell_family.mask_commit_authority
    ):
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.AUTHORITY_VIOLATION,
            "v2 hard-F/Q input crossed proposal-only authority",
        )
    operator = problem.signed_work_operator
    if design.endpoint_lane_count > operator.design.maximum_lanes:
        raise ValueError("v2 hard-F/Q lanes exceed signed-F operator capacity")
    if design.production_cuda_required:
        if (
            not execution.production_cuda_execution
            or not operator.receipt.production_cuda_execution
            or not shell_family.spectral.receipt.device.startswith("cuda:")
            or not shell_family.diagonal_l1.receipt.device.startswith("cuda:")
        ):
            _refuse(
                V1SignedWorkHardFQReplayRefusalCode.CUDA_REQUIRED,
                "production v2 hard-F/Q replay requires CUDA custody",
            )
    elif (
        not design.allow_private_cpu_reference
        or not execution.private_cpu_reference_execution
        or not operator.receipt.private_cpu_reference_execution
        or problem.route_count > design.private_reference_route_limit
        or shell_family.spectral.receipt.device != "cpu"
        or shell_family.diagonal_l1.receipt.device != "cpu"
    ):
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.CPU_REFERENCE_SCALE_EXCEEDED,
            "v2 CPU hard-F/Q replay is restricted to a tiny explicit reference",
        )
    if (
        design.require_deterministic_algorithms
        and execution.production_cuda_execution
        and (torch is None or not torch.are_deterministic_algorithms_enabled())
    ):
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.DETERMINISM_REQUIRED,
            "v2 CUDA hard-F/Q replay requires deterministic torch algorithms",
        )
    if (
        design.require_cublas_deterministic_workspace
        and execution.production_cuda_execution
        and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}
    ):
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.DETERMINISM_REQUIRED,
            "v2 CUDA hard-F/Q replay requires deterministic cuBLAS workspace",
        )


def estimate_v1_signed_work_hard_fq_device_hbm_v2(
    *,
    problem: V1SignedWorkLeastWorkCUDADeviceProblemV2,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    execution: V1SignedWorkLeastWorkCUDAExecutionBundle,
    shell_family: V1SignedWorkDeviceShellDeviceFamilyV2,
    design: V1SignedWorkHardFQReplayDesign = V1SignedWorkHardFQReplayDesign(),
    current_reserved_bytes: int | None = None,
    device_total_memory_bytes: int | None = None,
) -> V1SignedWorkHardFQHBMForecast:
    """Forecast replay allocations only; shell/source storage is borrowed."""

    _validate_device_v2_replay_inputs(
        problem=problem,
        policy=policy,
        execution=execution,
        shell_family=shell_family,
        design=design,
    )
    state = execution.state_handle
    if state is None:  # pragma: no cover - validation closes this.
        raise RuntimeError("v2 hard-F/Q state disappeared")
    operator = problem.signed_work_operator
    endpoint_count = len(shell_family.spectral.receipt.endpoint_counts)
    lanes = min(endpoint_count, design.endpoint_lane_count)
    routes = problem.route_count
    rank = problem.rank
    q_tile = min(routes, design.q_route_tile_size)
    labels_bytes = 2 * routes * _U8_BYTES
    cells = 2 * operator.context_count * operator.fiber_count * lanes * _F64_BYTES
    operator_tile = 2 * operator.forecast.temporary_tile_bytes_upper
    q_membership = q_tile * lanes * (3 * _BOOL_BYTES + _F64_BYTES)
    q_factor = rank * q_tile * _F64_BYTES
    q_compact = (2 * rank * lanes + 5 * lanes) * _F64_BYTES
    summary = 32 * lanes * _F64_BYTES
    f_peak = cells + operator_tile + summary
    q_peak = q_membership + q_factor + q_compact + summary
    working = max(f_peak, q_peak)
    guard = max(
        _MINIMUM_ALLOCATOR_GUARD_BYTES,
        math.ceil(_ALLOCATOR_GUARD_FRACTION * working),
    )
    incremental = working + guard
    device = state.device
    admission_memory = None
    if current_reserved_bytes is None or device_total_memory_bytes is None:
        if device.startswith("cuda:"):
            if torch is None or not torch.cuda.is_available():
                _refuse(
                    V1SignedWorkHardFQReplayRefusalCode.CUDA_REQUIRED,
                    "v2 CUDA replay forecast requires an available runtime",
                )
            torch_device = torch.device(device)
            admission_memory = (cuda_reusable_memory_bytes(torch_device, framework=torch), int(torch.cuda.memory_allocated(torch_device)))
            if current_reserved_bytes is None:
                current_reserved_bytes = int(torch.cuda.memory_reserved(torch_device))
            if device_total_memory_bytes is None:
                device_total_memory_bytes = int(
                    torch.cuda.get_device_properties(torch_device).total_memory
                )
        else:
            if current_reserved_bytes is None:
                current_reserved_bytes = 0
            if device_total_memory_bytes is None:
                device_total_memory_bytes = max(1 << 30, 4 * incremental)
    current = _nonnegative_int(current_reserved_bytes, name="current reserved bytes")
    total = _positive_int(device_total_memory_bytes, name="device total memory")
    ceiling = math.floor(0.85 * total)
    available, allocated = admission_memory or (total, current)
    return V1SignedWorkHardFQHBMForecast(
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        least_work_execution_hash=execution.execution_hash,
        shell_family_hash=shell_family.family_hash,
        design_hash=design.design_hash,
        device=device,
        route_count=routes,
        context_count=operator.context_count,
        fiber_count=operator.fiber_count,
        response_rank=rank,
        endpoint_count=endpoint_count,
        lane_count=lanes,
        q_route_tile_size=design.q_route_tile_size,
        resident_two_branch_label_bytes=labels_bytes,
        f_signed_and_envelope_cell_bytes=cells,
        f_operator_tile_workspace_upper_bytes=operator_tile,
        q_membership_tile_bytes=q_membership,
        q_absolute_factor_tile_bytes=q_factor,
        q_compact_projection_bytes=q_compact,
        compact_lane_summary_bytes=summary,
        f_peak_workspace_bytes=f_peak,
        q_peak_workspace_bytes=q_peak,
        allocator_guard_bytes=guard,
        forecast_incremental_bytes=incremental,
        current_reserved_bytes=current,
        forecast_peak_reserved_bytes=current + incremental,
        device_total_memory_bytes=total,
        hbm_ceiling_bytes=ceiling,
        admitted=allocated + incremental <= ceiling and incremental <= available,
        _admission_memory=admission_memory,
    )


def _device_endpoint_from_summary_v2(
    *,
    problem: V1SignedWorkLeastWorkCUDADeviceProblemV2,
    branch: V1RelaxedKKTBranch,
    shell_receipt: Any,
    endpoint_index: int,
    summary: _LaneSummary,
    lane_index: int,
    singleton_rung_sum: float,
) -> V1SignedWorkHardFQDeviceEndpointReceiptV2:
    count = shell_receipt.endpoint_counts[endpoint_index]
    route_ops = max(1, 2 * problem.route_count + 2 * problem.rank + 5)
    singleton_nominal = float(singleton_rung_sum)
    singleton_lower, singleton_upper = _nonnegative_interval_from_nominal(
        nominal=singleton_nominal,
        absolute_envelope=singleton_nominal,
        operation_count=route_ops,
    )
    outer_ops = max(
        1,
        4 * problem.route_count
        + 2
        * problem.signed_work_operator.context_count
        * problem.signed_work_operator.fiber_count
        + 2 * problem.rank
        + 8,
    )
    resource_nominal = float(summary.resource[lane_index])
    resource_lower, resource_upper = _nonnegative_interval_from_nominal(
        nominal=resource_nominal,
        absolute_envelope=resource_nominal,
        operation_count=route_ops,
    )
    cells_nominal = float(summary.signed_f_cells[lane_index])
    cells_envelope = float(summary.signed_f_envelope[lane_index])
    residual_nominal = float(summary.residual_linear[lane_index])
    raw_nominal = cells_nominal + residual_nominal
    raw_lower, raw_upper = _nonnegative_interval_from_nominal(
        nominal=raw_nominal,
        absolute_envelope=cells_envelope + residual_nominal,
        operation_count=outer_ops,
    )
    cut_count = problem.signed_work_operator.native_fiber_axis_receipt.cut_count
    f_lower, f_upper = _divide_nonnegative_interval(raw_lower, raw_upper, cut_count)
    f_nominal = raw_nominal / float(cut_count)
    low_values = summary.factor_projection[:, lane_index]
    low_absolute = summary.factor_absolute_projection[:, lane_index]
    low_intervals = tuple(
        _signed_sum_interval(
            nominal=float(value),
            absolute_envelope=float(envelope),
            operation_count=route_ops,
        )
        for value, envelope in zip(low_values, low_absolute, strict=True)
    )
    low_lower, low_upper = _sum_nonnegative_intervals(
        tuple(_square_interval(*item) for item in low_intervals)
    )
    low_nominal = float(np.dot(low_values, low_values))
    low_lower = min(low_lower, low_nominal)
    low_upper = max(low_upper, low_nominal)
    rho = problem.residual_spectral_bound
    spectral_nominal = float(rho * count)
    spectral_lower = _down_multiply_nonnegative(rho, float(count))
    spectral_upper = _up_multiply(rho, float(count))
    diagonal_projection = float(summary.diagonal_projection[lane_index])
    diagonal_projection_lower, diagonal_projection_upper = (
        _nonnegative_interval_from_nominal(
            nominal=diagonal_projection,
            absolute_envelope=diagonal_projection,
            operation_count=route_ops,
        )
    )
    diagonal_lower, diagonal_upper = _square_interval(
        diagonal_projection_lower, diagonal_projection_upper
    )
    diagonal_nominal = diagonal_projection * diagonal_projection
    diagonal_lower = min(diagonal_lower, diagonal_nominal)
    diagonal_upper = max(diagonal_upper, diagonal_nominal)
    spectral_q_lower = max(0.0, _down_add(low_lower, spectral_lower))
    spectral_q_upper = _up_add(low_upper, spectral_upper)
    diagonal_q_lower = max(0.0, _down_add(low_lower, diagonal_lower))
    diagonal_q_upper = _up_add(low_upper, diagonal_upper)
    branch_lower, branch_upper = (
        (spectral_q_lower, spectral_q_upper)
        if branch is V1RelaxedKKTBranch.SPECTRAL
        else (diagonal_q_lower, diagonal_q_upper)
    )
    canonical_lower = min(spectral_q_lower, diagonal_q_lower)
    canonical_upper = min(spectral_q_upper, diagonal_q_upper)
    origin_status = (
        V1SignedWorkHardFQEndpointStatus.Q_SAFE_FEASIBLE
        if branch_upper < problem.quadratic_budget
        else (
            V1SignedWorkHardFQEndpointStatus.Q_ROBUST_INFEASIBLE
            if branch_lower > problem.quadratic_budget
            else V1SignedWorkHardFQEndpointStatus.Q_ENVELOPE_AMBIGUOUS
        )
    )
    status = (
        V1SignedWorkHardFQEndpointStatus.Q_SAFE_FEASIBLE
        if canonical_upper < problem.quadratic_budget
        else (
            V1SignedWorkHardFQEndpointStatus.Q_ROBUST_INFEASIBLE
            if canonical_lower > problem.quadratic_budget
            else V1SignedWorkHardFQEndpointStatus.Q_ENVELOPE_AMBIGUOUS
        )
    )
    return V1SignedWorkHardFQDeviceEndpointReceiptV2(
        branch=branch,
        endpoint_label=endpoint_index + 1,
        cumulative_removed_count=count,
        device_action_commitment_hash=(
            shell_receipt.device_action_commitment_hashes[endpoint_index]
        ),
        resource_nominal=resource_nominal,
        resource_lower=resource_lower,
        resource_upper=resource_upper,
        signed_f_nominal=f_nominal,
        signed_f_lower=min(f_lower, f_nominal),
        signed_f_upper=max(f_upper, f_nominal),
        low_rank_q_nominal=low_nominal,
        low_rank_q_lower=low_lower,
        low_rank_q_upper=low_upper,
        spectral_residual_q_nominal=spectral_nominal,
        spectral_residual_q_lower=spectral_lower,
        spectral_residual_q_upper=spectral_upper,
        diagonal_residual_q_nominal=diagonal_nominal,
        diagonal_residual_q_lower=diagonal_lower,
        diagonal_residual_q_upper=diagonal_upper,
        spectral_q_lower=spectral_q_lower,
        spectral_q_upper=spectral_q_upper,
        diagonal_q_lower=diagonal_q_lower,
        diagonal_q_upper=diagonal_q_upper,
        branch_q_lower=branch_lower,
        branch_q_upper=branch_upper,
        canonical_min_q_lower=canonical_lower,
        canonical_min_q_upper=canonical_upper,
        quadratic_budget=problem.quadratic_budget,
        singleton_work_sum_nominal=singleton_nominal,
        singleton_work_sum_lower=min(singleton_lower, singleton_nominal),
        singleton_work_sum_upper=max(singleton_upper, singleton_nominal),
        origin_branch_status=origin_status,
        status=status,
    )


def _replay_device_branch_v2(
    *,
    problem: V1SignedWorkLeastWorkCUDADeviceProblemV2,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    execution: V1SignedWorkLeastWorkCUDAExecutionBundle,
    shell_family_hash: str,
    shell: V1SignedWorkDeviceShellDeviceBundleV2,
    source_tensors: dict[str, Any],
    design: V1SignedWorkHardFQReplayDesign,
    forecast: V1SignedWorkHardFQHBMForecast,
) -> V1SignedWorkHardFQDeviceBranchReceiptV2:
    receipt = shell.receipt
    endpoints: list[V1SignedWorkHardFQDeviceEndpointReceiptV2] = []
    endpoint_count = len(receipt.endpoint_counts)
    # Bounded endpoint query first (sequential borrow, closed before the
    # replay borrow opens): the singleton-work screen summed over every
    # registered rung.  Transfers ``endpoint_count`` scalars, no route rows.
    with shell.label_handle.borrow(
        operation=V1SignedWorkDeviceShellTensorOperationV2.BOUNDED_ENDPOINT_QUERY
    ) as query_lease:
        query_labels = query_lease.tensor
        if (
            query_labels.dtype != torch.uint8
            or query_labels.device != source_tensors["singleton_work"].device
            or tuple(query_labels.shape) != (problem.route_count,)
        ):
            _refuse(
                V1SignedWorkHardFQReplayRefusalCode.STALE_IDENTITY,
                "v2 shell labels and singleton-work source differ",
            )
        singleton_rung_sums = np.zeros(endpoint_count, dtype=np.float64)
        for start in range(0, endpoint_count, design.endpoint_lane_count):
            stop = min(endpoint_count, start + design.endpoint_lane_count)
            singleton_rung_sums[start:stop] = _tiled_singleton_rung_sums(
                labels=query_labels,
                endpoint_labels=tuple(range(start + 1, stop + 1)),
                singleton=source_tensors["singleton_work"],
                route_count=problem.route_count,
                route_tile_size=design.q_route_tile_size,
            )
    with shell.label_handle.borrow(
        operation=V1SignedWorkDeviceShellTensorOperationV2.HARD_FQ_REPLAY
    ) as label_lease:
        labels = label_lease.tensor
        if (
            labels.dtype != torch.uint8
            or labels.device != source_tensors["resource"].device
            or tuple(labels.shape) != (problem.route_count,)
        ):
            _refuse(
                V1SignedWorkHardFQReplayRefusalCode.STALE_IDENTITY,
                "v2 shell labels and F/Q source tensors differ",
            )
        for start in range(0, len(receipt.endpoint_counts), design.endpoint_lane_count):
            stop = min(
                len(receipt.endpoint_counts),
                start + design.endpoint_lane_count,
            )
            summary = _tiled_lane_summary(
                labels=labels,
                endpoint_labels=tuple(range(start + 1, stop + 1)),
                problem=problem,  # type: ignore[arg-type]
                source_tensors=source_tensors,
                route_tile_size=design.q_route_tile_size,
            )
            for lane_index, endpoint_index in enumerate(range(start, stop)):
                endpoints.append(
                    _device_endpoint_from_summary_v2(
                        problem=problem,
                        branch=receipt.branch,
                        shell_receipt=receipt,
                        endpoint_index=endpoint_index,
                        summary=summary,
                        lane_index=lane_index,
                        singleton_rung_sum=float(singleton_rung_sums[endpoint_index]),
                    )
                )
    state_hash = execution.state_identity_hash
    if state_hash is None:  # pragma: no cover - validation closes this.
        raise RuntimeError("v2 hard-F/Q state identity disappeared")
    return V1SignedWorkHardFQDeviceBranchReceiptV2(
        branch=receipt.branch,
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        least_work_execution_hash=execution.execution_hash,
        state_identity_hash=state_hash,
        shell_family_hash=shell_family_hash,
        shell_receipt_hash=receipt.receipt_hash,
        operator_receipt_hash=problem.signed_work_operator.receipt_hash,
        design_hash=design.design_hash,
        hbm_forecast_hash=forecast.forecast_hash,
        endpoints=tuple(endpoints),
        production_cuda_execution=execution.production_cuda_execution,
        private_cpu_reference_execution=execution.private_cpu_reference_execution,
    )


def replay_v1_signed_work_hard_fq_device_family_v2(
    *,
    problem: V1SignedWorkLeastWorkCUDADeviceProblemV2,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    execution: V1SignedWorkLeastWorkCUDAExecutionBundle,
    shell_family: V1SignedWorkDeviceShellDeviceFamilyV2,
    design: V1SignedWorkHardFQReplayDesign = V1SignedWorkHardFQReplayDesign(),
    current_reserved_bytes: int | None = None,
    device_total_memory_bytes: int | None = None,
) -> V1SignedWorkHardFQDeviceFamilyBundleV2:
    """Replay signed F and both Q envelopes with no route-scale D2H."""

    forecast = estimate_v1_signed_work_hard_fq_device_hbm_v2(
        problem=problem,
        policy=policy,
        execution=execution,
        shell_family=shell_family,
        design=design,
        current_reserved_bytes=current_reserved_bytes,
        device_total_memory_bytes=device_total_memory_bytes,
    )
    if not forecast.admitted:
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.HBM_FORECAST_EXCEEDED,
            "v2 hard-F/Q replay exceeds the registered HBM ceiling",
        )
    state = execution.state_handle
    if state is None:  # pragma: no cover - forecast validates this.
        raise RuntimeError("v2 hard-F/Q theorem state disappeared")
    expected_receipts = tuple(item.receipt_hash for item in execution.branch_receipts)
    try:
        with state.borrow_for_hard_fq_replay(
            expected_problem_hash=problem.problem_hash,
            expected_policy_hash=policy.policy_hash,
            expected_branch_receipt_hashes=expected_receipts,
        ) as lease:
            source_tensors = {
                name: lease.source_tensor(name)
                for name in ("resource", "factor", "sqrt_residual", "singleton_work")
            }
            device = source_tensors["resource"].device
            if (
                any(item.dtype != torch.float64 for item in source_tensors.values())
                or any(item.device != device for item in source_tensors.values())
                or tuple(source_tensors["resource"].shape) != (problem.route_count,)
                or tuple(source_tensors["factor"].shape)
                != (problem.rank, problem.route_count)
                or tuple(source_tensors["sqrt_residual"].shape)
                != (problem.route_count,)
                or tuple(source_tensors["singleton_work"].shape)
                != (problem.route_count,)
            ):
                _refuse(
                    V1SignedWorkHardFQReplayRefusalCode.STALE_IDENTITY,
                    "v2 replay source tensors differ in dtype, device, or route axis",
                )
            spectral = _replay_device_branch_v2(
                problem=problem,
                policy=policy,
                execution=execution,
                shell_family_hash=shell_family.family_hash,
                shell=shell_family.spectral,
                source_tensors=source_tensors,
                design=design,
                forecast=forecast,
            )
            diagonal = _replay_device_branch_v2(
                problem=problem,
                policy=policy,
                execution=execution,
                shell_family_hash=shell_family.family_hash,
                shell=shell_family.diagonal_l1,
                source_tensors=source_tensors,
                design=design,
                forecast=forecast,
            )
    except V1SignedWorkHardFQReplayRefusal:
        raise
    except V1SignedWorkDeviceShellRefusal as error:
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.LABEL_IDENTITY_MISMATCH,
            f"v2 shell label custody refused hard replay: {error}",
        )
    except V1SignedWorkLeastWorkCUDARefusal as error:
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.STALE_IDENTITY,
            f"v2 theorem-state custody refused hard replay: {error}",
        )
    except V1SignedWorkOperatorCUDARefusal as error:
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.NUMERICAL_REFUSAL,
            f"v2 signed-F operator refused hard replay: {error}",
        )
    state_hash = execution.state_identity_hash
    if state_hash is None:  # pragma: no cover
        raise RuntimeError("v2 state identity disappeared after replay")
    return V1SignedWorkHardFQDeviceFamilyBundleV2(
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        least_work_execution_hash=execution.execution_hash,
        state_identity_hash=state_hash,
        shell_family_hash=shell_family.family_hash,
        design_hash=design.design_hash,
        hbm_forecast=forecast,
        spectral=spectral,
        diagonal_l1=diagonal,
        common_endpoint_counts=shell_family.spectral.receipt.endpoint_counts,
    )


# ---------------------------------------------------------------------------
# Fiber-shuffle null lane
# ---------------------------------------------------------------------------

SIGNED_WORK_HARD_FQ_FIBER_SHUFFLE_NULL_DESIGN_SCHEMA: Final = (
    "partition_free_v1_signed_work_hard_fq_fiber_shuffle_null_design_v1"
)
SIGNED_WORK_HARD_FQ_FIBER_SHUFFLE_NULL_BRANCH_SCHEMA_V2: Final = (
    "partition_free_v1_signed_work_hard_fq_fiber_shuffle_null_branch_v2"
)
SIGNED_WORK_HARD_FQ_FIBER_SHUFFLE_NULL_FAMILY_SCHEMA_V2: Final = (
    "partition_free_v1_signed_work_hard_fq_fiber_shuffle_null_family_v2"
)
SIGNED_WORK_HARD_FQ_FIBER_SHUFFLE_NULL_AUTHORITY: Final = (
    "null_distribution_only_no_certificate_c_stream_or_commit_authority"
)
SIGNED_WORK_HARD_FQ_FIBER_SHUFFLE_PERMUTATION_SCOPE: Final = (
    "within_cut_route_to_fiber_reassignment_membership_and_singletons_fixed"
)
_NULL_DESIGN_PREFIX: Final = "irg-v1-signed-work-hard-fq-fiber-shuffle-null-design-"
_NULL_BRANCH_PREFIX: Final = "irg-v1-signed-work-hard-fq-fiber-shuffle-null-branch-"
_NULL_FAMILY_PREFIX: Final = "irg-v1-signed-work-hard-fq-fiber-shuffle-null-family-"


@dataclass(frozen=True, slots=True)
class V1SignedWorkHardFQFiberShuffleNullDesign:
    """Sibling design keyed by the (unchanged) hard-F/Q replay design hash.

    The replay design is a hashed family consumed by landed receipts, so the
    null lane's numerals live here rather than as new fields on it.
    """

    shuffle_count: int
    seed_hash: str
    replay_design_hash: str
    permutation_scope: str = SIGNED_WORK_HARD_FQ_FIBER_SHUFFLE_PERMUTATION_SCOPE
    schema: str = SIGNED_WORK_HARD_FQ_FIBER_SHUFFLE_NULL_DESIGN_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "shuffle_count",
            _positive_int(self.shuffle_count, name="fiber shuffle count"),
        )
        for name in ("seed_hash", "replay_design_hash"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if (
            self.permutation_scope != SIGNED_WORK_HARD_FQ_FIBER_SHUFFLE_PERMUTATION_SCOPE
            or self.schema != SIGNED_WORK_HARD_FQ_FIBER_SHUFFLE_NULL_DESIGN_SCHEMA
        ):
            raise ValueError("fiber-shuffle null semantics changed")

    @property
    def design_hash(self) -> str:
        return content_hash(self, prefix=_NULL_DESIGN_PREFIX)


def _finite_matrix(rows: object, *, row_count: int, column_count: int, name: str):
    matrix = tuple(tuple(float(value) for value in row) for row in rows)
    if len(matrix) != row_count or any(len(row) != column_count for row in matrix):
        raise ValueError(f"{name} must be {row_count}x{column_count}")
    if any(not math.isfinite(value) or value < 0.0 for row in matrix for value in row):
        raise ValueError(f"{name} must be finite and nonnegative")
    return matrix


@dataclass(frozen=True, slots=True)
class V1SignedWorkHardFQFiberShuffleNullBranchReceiptV2:
    branch: V1RelaxedKKTBranch
    problem_hash: str
    shell_receipt_hash: str
    replay_branch_receipt_hash: str
    operator_receipt_hash: str
    null_design_hash: str
    endpoint_labels: tuple[int, ...]
    nominal_signed_f: tuple[float, ...]
    singleton_work_sum_nominal: tuple[float, ...]
    signed_f_by_endpoint_by_shuffle: tuple[tuple[float, ...], ...]
    scalar_d2h_count: int
    route_scale_d2h_bytes: int = 0
    permutation_temporary_bytes: int = 0
    c_stream_authority: bool = False
    nonlinear_remainder_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_HARD_FQ_FIBER_SHUFFLE_NULL_AUTHORITY
    schema: str = SIGNED_WORK_HARD_FQ_FIBER_SHUFFLE_NULL_BRANCH_SCHEMA_V2

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("fiber-shuffle null branch must be exact")
        for name in (
            "problem_hash",
            "shell_receipt_hash",
            "replay_branch_receipt_hash",
            "operator_receipt_hash",
            "null_design_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        labels = tuple(int(value) for value in self.endpoint_labels)
        if not labels or labels != tuple(range(1, len(labels) + 1)):
            raise ValueError("fiber-shuffle null endpoints must be the complete ladder")
        object.__setattr__(self, "endpoint_labels", labels)
        for name in ("nominal_signed_f", "singleton_work_sum_nominal"):
            values = tuple(float(value) for value in getattr(self, name))
            if len(values) != len(labels) or any(
                not math.isfinite(value) or value < 0.0 for value in values
            ):
                raise ValueError(f"{name} must be one finite nonnegative scalar per rung")
            object.__setattr__(self, name, values)
        matrix = _finite_matrix(
            self.signed_f_by_endpoint_by_shuffle,
            row_count=len(labels),
            column_count=len(self.signed_f_by_endpoint_by_shuffle[0])
            if self.signed_f_by_endpoint_by_shuffle
            else 0,
            name="signed_f_by_endpoint_by_shuffle",
        )
        if not matrix or not matrix[0]:
            raise ValueError("fiber-shuffle null requires at least one shuffle")
        object.__setattr__(self, "signed_f_by_endpoint_by_shuffle", matrix)
        expected_scalars = len(labels) * len(matrix[0])
        if int(self.scalar_d2h_count) != expected_scalars:
            raise ValueError("fiber-shuffle null scalar D2H count disagrees with its matrix")
        object.__setattr__(self, "scalar_d2h_count", expected_scalars)
        if int(self.route_scale_d2h_bytes) != 0:
            raise ValueError("fiber-shuffle null must not transfer route-scale rows")
        object.__setattr__(
            self,
            "permutation_temporary_bytes",
            _nonnegative_int(self.permutation_temporary_bytes, name="permutation bytes"),
        )
        flags = (
            self.c_stream_authority,
            self.nonlinear_remainder_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in flags) or any(flags):
            raise ValueError("fiber-shuffle null receipts carry no authority")
        if (
            self.authority_scope != SIGNED_WORK_HARD_FQ_FIBER_SHUFFLE_NULL_AUTHORITY
            or self.schema != SIGNED_WORK_HARD_FQ_FIBER_SHUFFLE_NULL_BRANCH_SCHEMA_V2
        ):
            raise ValueError("fiber-shuffle null branch semantics changed")

    @property
    def shuffle_count(self) -> int:
        return len(self.signed_f_by_endpoint_by_shuffle[0])

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_NULL_BRANCH_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkHardFQFiberShuffleNullFamilyReceiptV2:
    problem_hash: str
    policy_hash: str
    shell_family_hash: str
    replay_bundle_hash: str
    null_design_hash: str
    spectral: V1SignedWorkHardFQFiberShuffleNullBranchReceiptV2
    diagonal_l1: V1SignedWorkHardFQFiberShuffleNullBranchReceiptV2
    forecast_incremental_bytes: int
    hbm_forecast_hash: str
    c_stream_authority: bool = False
    mask_commit_authority: bool = False
    authority_scope: str = SIGNED_WORK_HARD_FQ_FIBER_SHUFFLE_NULL_AUTHORITY
    schema: str = SIGNED_WORK_HARD_FQ_FIBER_SHUFFLE_NULL_FAMILY_SCHEMA_V2

    def __post_init__(self) -> None:
        for name in (
            "problem_hash",
            "policy_hash",
            "shell_family_hash",
            "replay_bundle_hash",
            "null_design_hash",
            "hbm_forecast_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if (
            type(self.spectral) is not V1SignedWorkHardFQFiberShuffleNullBranchReceiptV2
            or type(self.diagonal_l1) is not V1SignedWorkHardFQFiberShuffleNullBranchReceiptV2
            or self.spectral.branch is not V1RelaxedKKTBranch.SPECTRAL
            or self.diagonal_l1.branch is not V1RelaxedKKTBranch.DIAGONAL_L1
            or self.spectral.problem_hash != self.problem_hash
            or self.diagonal_l1.problem_hash != self.problem_hash
            or self.spectral.null_design_hash != self.null_design_hash
            or self.diagonal_l1.null_design_hash != self.null_design_hash
        ):
            raise ValueError("fiber-shuffle null family branches are inconsistent")
        object.__setattr__(
            self,
            "forecast_incremental_bytes",
            _nonnegative_int(self.forecast_incremental_bytes, name="forecast bytes"),
        )
        if (
            type(self.c_stream_authority) is not bool
            or type(self.mask_commit_authority) is not bool
            or self.c_stream_authority
            or self.mask_commit_authority
            or self.authority_scope != SIGNED_WORK_HARD_FQ_FIBER_SHUFFLE_NULL_AUTHORITY
            or self.schema != SIGNED_WORK_HARD_FQ_FIBER_SHUFFLE_NULL_FAMILY_SCHEMA_V2
        ):
            raise ValueError("fiber-shuffle null family carries no authority")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_NULL_FAMILY_PREFIX)


def _shuffled_signed_f_lanes(
    *,
    labels: object,
    endpoint_labels: tuple[int, ...],
    operator: object,
    route_permutation: object,
    device: object,
    residual_projection: np.ndarray,
    cut_count: int,
) -> np.ndarray:
    """Signed F per lane on the nominal scale: ``(cells + residual) / cut_count``.

    The cell sum is the same reduction ``_tiled_lane_summary`` performs; the
    residual projection is permutation-invariant (membership is fixed) and is
    folded in exactly as ``_device_endpoint_from_summary_v2`` folds it, so the
    identity permutation reproduces ``signed_f_nominal``.
    """

    endpoint_t = torch.tensor(endpoint_labels, dtype=torch.uint8, device=device)
    signed_cells, _absolute = operator.forward_from_first_inclusion_labels(
        labels, endpoint_t, route_permutation=route_permutation
    )
    context_weights = operator.context_weights.reshape(-1, 1, 1)
    lanes = torch.sum(context_weights * torch.abs(signed_cells), dim=(0, 1))
    del signed_cells, _absolute
    cells = lanes.detach().cpu().numpy().astype(np.float64, copy=False)
    if cells.shape != (len(endpoint_labels),) or not np.all(np.isfinite(cells)):
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.NUMERICAL_REFUSAL,
            "fiber-shuffle null produced a nonfinite or misshaped lane",
        )
    host = np.empty_like(cells)
    for index in range(len(endpoint_labels)):
        raw_nominal = float(cells[index]) + float(residual_projection[index])
        host[index] = raw_nominal / float(cut_count)
    return host


def _null_branch_v2(
    *,
    problem: V1SignedWorkLeastWorkCUDADeviceProblemV2,
    shell: V1SignedWorkDeviceShellDeviceBundleV2,
    replay_branch: V1SignedWorkHardFQDeviceBranchReceiptV2,
    null_design: V1SignedWorkHardFQFiberShuffleNullDesign,
    design: V1SignedWorkHardFQReplayDesign,
    device: object,
) -> V1SignedWorkHardFQFiberShuffleNullBranchReceiptV2:
    receipt = shell.receipt
    if replay_branch.shell_receipt_hash != receipt.receipt_hash:
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.STALE_IDENTITY,
            "fiber-shuffle null replay branch and shell receipt differ",
        )
    operator = problem.signed_work_operator
    endpoint_count = len(receipt.endpoint_counts)
    matrix = np.zeros((endpoint_count, null_design.shuffle_count), dtype=np.float64)
    permutation_bytes = 0
    cut_count = int(operator.native_fiber_axis_receipt.cut_count)
    with shell.label_handle.borrow(
        operation=V1SignedWorkDeviceShellTensorOperationV2.HARD_FQ_REPLAY
    ) as label_lease:
        labels = label_lease.tensor
        if (
            labels.dtype != torch.uint8
            or labels.device != device
            or tuple(labels.shape) != (problem.route_count,)
        ):
            _refuse(
                V1SignedWorkHardFQReplayRefusalCode.STALE_IDENTITY,
                "fiber-shuffle null labels differ from the replay source",
            )
        # Permutation-invariant residual projection epsilon^T 1_S per rung,
        # computed once with the same tiled membership as the replay.
        residual_projection = np.zeros(endpoint_count, dtype=np.float64)
        for start in range(0, endpoint_count, design.endpoint_lane_count):
            stop = min(endpoint_count, start + design.endpoint_lane_count)
            residual_projection[start:stop] = _tiled_singleton_rung_sums(
                labels=labels,
                endpoint_labels=tuple(range(start + 1, stop + 1)),
                singleton=operator.residual_linear,
                route_count=problem.route_count,
                route_tile_size=design.q_route_tile_size,
            )
        for shuffle_index in range(null_design.shuffle_count):
            seed = (
                f"{null_design.seed_hash}|{receipt.branch.value}|{shuffle_index}"
            ).encode("ascii")
            permutation = operator.within_cut_route_permutation(seed=seed)
            permutation_bytes = max(
                permutation_bytes, int(permutation.numel()) * 8
            )
            for start in range(0, endpoint_count, design.endpoint_lane_count):
                stop = min(endpoint_count, start + design.endpoint_lane_count)
                matrix[start:stop, shuffle_index] = _shuffled_signed_f_lanes(
                    labels=labels,
                    endpoint_labels=tuple(range(start + 1, stop + 1)),
                    operator=operator,
                    route_permutation=permutation,
                    device=device,
                    residual_projection=residual_projection[start:stop],
                    cut_count=cut_count,
                )
            del permutation
    return V1SignedWorkHardFQFiberShuffleNullBranchReceiptV2(
        branch=receipt.branch,
        problem_hash=problem.problem_hash,
        shell_receipt_hash=receipt.receipt_hash,
        replay_branch_receipt_hash=replay_branch.receipt_hash,
        operator_receipt_hash=operator.receipt_hash,
        null_design_hash=null_design.design_hash,
        endpoint_labels=tuple(range(1, endpoint_count + 1)),
        nominal_signed_f=tuple(
            float(item.signed_f_nominal) for item in replay_branch.endpoints
        ),
        singleton_work_sum_nominal=tuple(
            float(item.singleton_work_sum_nominal) for item in replay_branch.endpoints
        ),
        signed_f_by_endpoint_by_shuffle=tuple(
            tuple(float(value) for value in row) for row in matrix
        ),
        scalar_d2h_count=endpoint_count * null_design.shuffle_count,
        permutation_temporary_bytes=permutation_bytes,
    )


def replay_v1_signed_work_hard_fq_fiber_shuffle_null_v2(
    *,
    problem: V1SignedWorkLeastWorkCUDADeviceProblemV2,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    execution: V1SignedWorkLeastWorkCUDAExecutionBundle,
    shell_family: V1SignedWorkDeviceShellDeviceFamilyV2,
    replay: V1SignedWorkHardFQDeviceFamilyBundleV2,
    null_design: V1SignedWorkHardFQFiberShuffleNullDesign,
    design: V1SignedWorkHardFQReplayDesign = V1SignedWorkHardFQReplayDesign(),
    current_reserved_bytes: int | None = None,
    device_total_memory_bytes: int | None = None,
) -> V1SignedWorkHardFQFiberShuffleNullFamilyReceiptV2:
    """Signed F of every registered rung under within-cut fiber shuffles.

    The null keeps every rung's membership (and therefore its singleton-work
    sum) fixed and reassigns routes to fibers of equal size inside each cut,
    so only the fiber structure that signed cancellation exploits is
    destroyed.  Output is ``endpoints x shuffles`` scalars per branch; no
    route-scale row leaves the device.  Nothing here is a certificate.
    """

    if type(null_design) is not V1SignedWorkHardFQFiberShuffleNullDesign:
        raise TypeError("fiber-shuffle null requires its exact design")
    if null_design.replay_design_hash != design.design_hash:
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.STALE_IDENTITY,
            "fiber-shuffle null design is keyed to another replay design",
        )
    if (
        replay.problem_hash != problem.problem_hash
        or replay.policy_hash != policy.policy_hash
        or replay.shell_family_hash != shell_family.family_hash
        or replay.least_work_execution_hash != execution.execution_hash
    ):
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.STALE_IDENTITY,
            "fiber-shuffle null replay bundle is bound to other parents",
        )
    forecast = estimate_v1_signed_work_hard_fq_device_hbm_v2(
        problem=problem,
        policy=policy,
        execution=execution,
        shell_family=shell_family,
        design=design,
        current_reserved_bytes=current_reserved_bytes,
        device_total_memory_bytes=device_total_memory_bytes,
    )
    # One route-length int64 permutation plus one gathered right-factor tile
    # (bounded by the operator's own tile workspace) beyond the F replay.
    incremental = 8 * int(problem.route_count) + int(
        forecast.f_operator_tile_workspace_upper_bytes
    )
    total_incremental = forecast.forecast_incremental_bytes + incremental
    if execution.production_cuda_execution and (current_reserved_bytes is None or device_total_memory_bytes is None):
        device = torch.device(forecast.device)
        admitted = (int(torch.cuda.memory_allocated(device)) + total_incremental <= forecast.hbm_ceiling_bytes
                    and total_incremental <= cuda_reusable_memory_bytes(device, framework=torch))
    else:
        admitted = forecast.forecast_peak_reserved_bytes + incremental <= forecast.hbm_ceiling_bytes
    if not forecast.admitted or not admitted:
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.HBM_FORECAST_EXCEEDED,
            "fiber-shuffle null exceeds the registered HBM ceiling",
        )
    state = execution.state_handle
    if state is None:  # pragma: no cover - forecast validates this.
        raise RuntimeError("fiber-shuffle null theorem state disappeared")
    expected_receipts = tuple(item.receipt_hash for item in execution.branch_receipts)
    try:
        with state.borrow_for_hard_fq_replay(
            expected_problem_hash=problem.problem_hash,
            expected_policy_hash=policy.policy_hash,
            expected_branch_receipt_hashes=expected_receipts,
        ) as lease:
            device = lease.source_tensor("resource").device
            spectral = _null_branch_v2(
                problem=problem,
                shell=shell_family.spectral,
                replay_branch=replay.spectral,
                null_design=null_design,
                design=design,
                device=device,
            )
            diagonal = _null_branch_v2(
                problem=problem,
                shell=shell_family.diagonal_l1,
                replay_branch=replay.diagonal_l1,
                null_design=null_design,
                design=design,
                device=device,
            )
    except V1SignedWorkHardFQReplayRefusal:
        raise
    except V1SignedWorkDeviceShellRefusal as error:
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.LABEL_IDENTITY_MISMATCH,
            f"v2 shell label custody refused the fiber-shuffle null: {error}",
        )
    except V1SignedWorkLeastWorkCUDARefusal as error:
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.STALE_IDENTITY,
            f"v2 theorem-state custody refused the fiber-shuffle null: {error}",
        )
    except V1SignedWorkOperatorCUDARefusal as error:
        _refuse(
            V1SignedWorkHardFQReplayRefusalCode.NUMERICAL_REFUSAL,
            f"v2 signed-F operator refused the fiber-shuffle null: {error}",
        )
    return V1SignedWorkHardFQFiberShuffleNullFamilyReceiptV2(
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        shell_family_hash=shell_family.family_hash,
        replay_bundle_hash=replay.bundle_hash,
        null_design_hash=null_design.design_hash,
        spectral=spectral,
        diagonal_l1=diagonal,
        forecast_incremental_bytes=incremental,
        hbm_forecast_hash=forecast.forecast_hash,
    )


__all__ = [
    "SIGNED_WORK_HARD_FQ_ALGORITHM_ID",
    "SIGNED_WORK_HARD_FQ_AUTHORITY",
    "SIGNED_WORK_HARD_FQ_BUNDLE_SCHEMA",
    "SIGNED_WORK_HARD_FQ_BRANCH_SCHEMA",
    "SIGNED_WORK_HARD_FQ_ENDPOINT_SCHEMA",
    "SIGNED_WORK_HARD_FQ_FAMILY_SCHEMA",
    "SIGNED_WORK_HARD_FQ_HBM_SCHEMA",
    "SIGNED_WORK_HARD_FQ_OUTWARD_ARITHMETIC_ID",
    "SIGNED_WORK_HARD_FQ_REPLAY_DESIGN_SCHEMA",
    "SIGNED_WORK_HARD_FQ_SINGLE_BRANCH_BUNDLE_SCHEMA",
    "SIGNED_WORK_HARD_FQ_UNIT_RESOURCE_ID",
    "V1SignedWorkHardFQBranchReceipt",
    "V1SignedWorkHardFQEndpointReceipt",
    "V1SignedWorkHardFQEndpointStatus",
    "V1SignedWorkHardFQFamilyBundle",
    "V1SignedWorkHardFQFamilyReceipt",
    "V1SignedWorkHardFQHBMForecast",
    "V1SignedWorkHardFQReplayDesign",
    "V1SignedWorkHardFQReplayRefusal",
    "V1SignedWorkHardFQReplayRefusalCode",
    "V1SignedWorkHardFQSingleBranchBundle",
    "SIGNED_WORK_HARD_FQ_DEVICE_BRANCH_SCHEMA_V2",
    "SIGNED_WORK_HARD_FQ_DEVICE_BUNDLE_SCHEMA_V2",
    "SIGNED_WORK_HARD_FQ_DEVICE_ENDPOINT_SCHEMA_V2",
    "SIGNED_WORK_HARD_FQ_DEVICE_FAMILY_SCHEMA_V2",
    "V1SignedWorkHardFQDeviceBranchReceiptV2",
    "V1SignedWorkHardFQDeviceEndpointReceiptV2",
    "V1SignedWorkHardFQDeviceFamilyBundleV2",
    "estimate_v1_signed_work_hard_fq_device_hbm_v2",
    "estimate_v1_signed_work_hard_fq_replay_hbm",
    "estimate_v1_signed_work_hard_fq_single_branch_hbm",
    "replay_v1_signed_work_hard_fq_device_family_v2",
    "SIGNED_WORK_HARD_FQ_FIBER_SHUFFLE_NULL_AUTHORITY",
    "SIGNED_WORK_HARD_FQ_FIBER_SHUFFLE_NULL_BRANCH_SCHEMA_V2",
    "SIGNED_WORK_HARD_FQ_FIBER_SHUFFLE_NULL_DESIGN_SCHEMA",
    "SIGNED_WORK_HARD_FQ_FIBER_SHUFFLE_NULL_FAMILY_SCHEMA_V2",
    "SIGNED_WORK_HARD_FQ_FIBER_SHUFFLE_PERMUTATION_SCOPE",
    "V1SignedWorkHardFQFiberShuffleNullBranchReceiptV2",
    "V1SignedWorkHardFQFiberShuffleNullDesign",
    "V1SignedWorkHardFQFiberShuffleNullFamilyReceiptV2",
    "replay_v1_signed_work_hard_fq_fiber_shuffle_null_v2",
    "replay_v1_signed_work_hard_fq_family_cuda",
    "replay_v1_signed_work_hard_fq_single_branch_cuda",
]
