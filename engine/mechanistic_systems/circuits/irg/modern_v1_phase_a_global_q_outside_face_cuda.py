"""Bounded true-Q outer-LP and outside-face route scan.

This joint consumes the exact support-coefficient stream, one route tile at a
time, after a compact nonnegative dual proposal has been frozen.  It evaluates
the registered outer-LP dual objective and the incumbent-aligned reduced-sign
conditions while retaining only the compact data needed by the later exact
outside-fixed shell-face LP.  In particular, it never retains a routewise
coefficient, membership vector, or direction bank.

The scan is evidence, not the no-entry theorem.  Complete compact hard-shell
enumeration and the strict ``dual gap < outside bit margin`` implication remain
a later joint, so this result grants no D-stability, C/K/O, commit, runtime,
terminal, or Pareto authority.
"""

from __future__ import annotations

from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

from contextlib import contextmanager
from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
import hashlib
import math
from types import MappingProxyType
from typing import Any, Final, Iterator, Mapping, NoReturn

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask

from .modern_v1_literal_j_joint_affine_shell_family_cuda import (
    V1LiteralJJointAffineShellPreoutcomeRegistration,
)
from .modern_v1_literal_j_joint_population_mjv_stream_cuda import (
    V1LiteralJJointD1PopulationMvStreamCUDAHandle,
    V1LiteralJJointPopulationMvStreamCUDAHandle,
)
from .modern_v1_phase_a_global_q_outer_lp_dual_proposal import (
    V1PhaseAGlobalQOuterLPDualProposalResult,
)
from .modern_v1_phase_a_global_q_compact_outer_lp_dual_solver import (
    V1PhaseAGlobalQCompactOuterLPCUDASource,
)
from .modern_v1_phase_a_global_q_support_family import (
    V1PhaseAGlobalQD1SupportFamilyReceipt,
    V1PhaseAGlobalQD1SupportFamilyResult,
    V1PhaseAGlobalQSupportFamilyHandle,
    V1PhaseAGlobalQSupportFamilyReceipt,
    V1PhaseAGlobalQSupportFamilyResult,
)
from .modern_v1_phase_a_global_q_support_plane_scan import (
    V1PhaseAGlobalQSupportPlaneScanHandle,
    V1PhaseAGlobalQSupportPlaneScanReceipt,
)
from .modern_v1_relaxed_kkt import V1RelaxedKKTProblem
from .modern_v1_sampled_v7_action_pair_alias_source_join import (
    V1SampledV7ActionPairAliasSourceJoinReceipt,
)
from .modern_v1_sampled_v7_phase_a_robust_shell_decision import (
    V1SampledV7PhaseARobustDecisionStatus,
    V1SampledV7PhaseARobustShellDecisionReceipt,
)

try:  # pragma: no cover - runtime-image dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


PHASE_A_GLOBAL_Q_OUTSIDE_FACE_DESIGN_SCHEMA: Final = (
    "partition_free_v1_phase_a_global_true_q_outside_face_scan_design_v1"
)
PHASE_A_GLOBAL_Q_OUTSIDE_FACE_FORECAST_SCHEMA: Final = (
    "partition_free_v1_phase_a_global_true_q_outside_face_scan_hbm_forecast_v1"
)
PHASE_A_GLOBAL_Q_OUTSIDE_FACE_SCHEMA: Final = (
    "partition_free_v1_phase_a_global_true_q_outside_face_scan_v3"
)
PHASE_A_GLOBAL_Q_OUTSIDE_FACE_ALGORITHM: Final = (
    "exact_support_bound_route_scan_frozen_dual_outer_LP_enclosure_and_compact_"
    "outside_fixed_hard_shell_margin_shared_cuda_source_v3"
)
PHASE_A_GLOBAL_Q_OUTSIDE_FACE_AUTHORITY: Final = (
    "compact_true_q_outer_dual_and_outside_sign_scan_only_hard_face_margin_pending_"
    "no_D_C_K_O_commit_runtime_terminal_or_pareto_authority_v3"
)

_PREFIX: Final = "partition-free-v1-phase-a-global-q-outside-face-scan-"
_FORECAST_TOKEN: Final = object()
_RECEIPT_TOKEN: Final = object()
_HANDLE_TOKEN: Final = object()
_F64_BYTES: Final = 8
_MIN_SUBNORMAL: Final = float.fromhex("0x0.0000000000001p-1022")
_LEGACY_PROJECTION_SUPPORT_SOURCE_MODE: Final = "legacy_projection_v1"
_FULL_POPULATION_D1_SUPPORT_SOURCE_MODE: Final = "full_population_d1_v1"


@contextmanager
def _owned_support_plane_tile(tile: Any) -> Iterator[Any]:
    """Evict one acquired route tile without replacing a scan failure.

    The population-Mv stream permits exactly one live tile.  Every path after
    ``next_tile`` therefore owns a mandatory eviction, including numerical or
    source refusals raised halfway through the route calculation.  A cleanup
    failure is attached to the primary exception as a note; it must not erase
    the refusal that explains why the scan stopped.
    """

    try:
        yield tile
    except BaseException as primary:
        try:
            tile.evict()
        except BaseException as cleanup_error:
            primary.add_note(
                "support-plane tile cleanup also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        raise
    else:
        tile.evict()


class V1PhaseAGlobalQOutsideFaceRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_SOURCE = "stale_source"
    SOURCE_MISMATCH = "source_mismatch"
    NO_STABLE_PHYSICAL_ACTION = "no_stable_physical_action"
    STREAM_ALREADY_OPENED = "stream_already_opened"
    ROUTE_COVERAGE_MISMATCH = "route_coverage_mismatch"
    ACTION_MEMBERSHIP_MISMATCH = "action_membership_mismatch"
    BLOCK_CAPACITY_MISMATCH = "block_capacity_mismatch"
    DEVICE_REQUIRED = "device_required"
    CPU_PRODUCTION_FALLBACK = "cpu_production_fallback"
    DETERMINISM_DISABLED = "determinism_disabled"
    HBM_FORECAST_EXCEEDED = "hbm_forecast_exceeded"
    NUMERICAL_REFUSAL = "numerical_refusal"
    ACTIVE_LEASE = "active_lease"
    MUTATED_CUSTODY = "mutated_custody"
    EVICTED = "evicted"
    AUTHORITY_VIOLATION = "authority_violation"


class V1PhaseAGlobalQOutsideFaceRefusal(RuntimeError):
    def __init__(
        self,
        code: V1PhaseAGlobalQOutsideFaceRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1PhaseAGlobalQOutsideFaceRefusalCode:
            raise TypeError("outside-face refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(code: V1PhaseAGlobalQOutsideFaceRefusalCode, message: str) -> NoReturn:
    raise V1PhaseAGlobalQOutsideFaceRefusal(code, message)


def _support_family_source_mode(support_family: object) -> str:
    """Classify exact support custody without relabelling a measure as a projection."""

    if (
        type(support_family) is V1PhaseAGlobalQSupportFamilyResult
        and type(support_family.receipt) is V1PhaseAGlobalQSupportFamilyReceipt
        and type(support_family.device_handle) is V1PhaseAGlobalQSupportFamilyHandle
        and support_family.device_handle.receipt is support_family.receipt
    ):
        return _LEGACY_PROJECTION_SUPPORT_SOURCE_MODE
    if (
        type(support_family) is V1PhaseAGlobalQD1SupportFamilyResult
        and type(support_family.receipt) is V1PhaseAGlobalQD1SupportFamilyReceipt
        and type(support_family.device_handle) is V1PhaseAGlobalQSupportFamilyHandle
        and support_family.device_handle.receipt is support_family.receipt
    ):
        return _FULL_POPULATION_D1_SUPPORT_SOURCE_MODE
    _refuse(
        V1PhaseAGlobalQOutsideFaceRefusalCode.INVALID_TYPE,
        "outside-face execution requires one exact legacy-projection or "
        "full-population-D1 support owner",
    )


class V1PhaseAGlobalQOutsideSignStatus(str, Enum):
    STRICT_INCUMBENT_ALIGNED = "strict_incumbent_aligned"
    AMBIGUOUS = "ambiguous"
    MISALIGNED = "misaligned"


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1PhaseAGlobalQOutsideFaceRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty exact string",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value <= 0:
        _refuse(
            V1PhaseAGlobalQOutsideFaceRefusalCode.INVALID_TYPE,
            f"{name} must be one positive exact integer",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        _refuse(
            V1PhaseAGlobalQOutsideFaceRefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative exact integer",
        )
    return value


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _refuse(
            V1PhaseAGlobalQOutsideFaceRefusalCode.INVALID_TYPE,
            f"{name} must be one real scalar",
        )
    result = float(value)
    if not math.isfinite(result):
        _refuse(
            V1PhaseAGlobalQOutsideFaceRefusalCode.NUMERICAL_REFUSAL,
            f"{name} must be finite",
        )
    return result


def _payload_hash(payload: Mapping[str, object], *, role: str) -> str:
    return content_hash(dict(payload), prefix=_PREFIX + role + "-")


def _tensor_invariant(value: Any) -> tuple[object, ...]:
    return (
        id(value),
        int(value.data_ptr()),
        tuple(int(item) for item in value.shape),
        tuple(int(item) for item in value.stride()),
        str(value.dtype),
        str(value.device),
        int(value.storage_offset()),
        int(value._version),
        bool(value.requires_grad),
    )


def _compact_tensor_digest(value: Any, *, role: str) -> str:
    array = value.detach().to(device="cpu").contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(role.encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(repr(tuple(int(item) for item in array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return _payload_hash(
        {
            "role": role,
            "shape": tuple(int(item) for item in array.shape),
            "payload_sha256": digest.hexdigest(),
        },
        role="compact-tensor",
    )


def _down(value: Any) -> Any:
    return torch.nextafter(value, torch.full_like(value, -math.inf))


def _up(value: Any) -> Any:
    return torch.nextafter(value, torch.full_like(value, math.inf))


def _add_down(left: Any, right: Any) -> Any:
    raw = left + right
    return torch.where(right == 0.0, left, torch.where(left == 0.0, right, _down(raw)))


def _add_up(left: Any, right: Any) -> Any:
    raw = left + right
    return torch.where(right == 0.0, left, torch.where(left == 0.0, right, _up(raw)))


def _mul_down(left: Any, right: Any) -> Any:
    raw = left * right
    return torch.where((left == 0.0) | (right == 0.0), torch.zeros_like(raw), _down(raw))


def _mul_up(left: Any, right: Any) -> Any:
    raw = left * right
    return torch.where((left == 0.0) | (right == 0.0), torch.zeros_like(raw), _up(raw))


def _gamma(operation_count: int) -> float:
    if operation_count <= 0:
        return 0.0
    product = operation_count * np.finfo(np.float64).eps
    if product >= 1.0:
        _refuse(
            V1PhaseAGlobalQOutsideFaceRefusalCode.NUMERICAL_REFUSAL,
            "route-tile summation gamma is not finite below one",
        )
    return math.nextafter(product / (1.0 - product), math.inf)


def _tile_sum_error_upper(values: Any) -> tuple[Any, Any]:
    """Return ``fl(sum(values))`` and a sound absolute-error upper bound.

    The usual ``gamma_n * sum(abs(x))`` expression is not sound when the
    absolute sum on its right-hand side is itself evaluated in floating point.
    If ``m_hat = fl(sum(abs(x)))``, then

    ``sum(abs(x)) <= (m_hat + delta) / (1 - gamma_n)``.

    Substitution yields ``error <= (gamma_n*m_hat + delta)/(1-gamma_n)``.
    Every arithmetic operation used for that upper endpoint is rounded
    outward.  This remains a conservative sequential-operation bound for a
    deterministic parallel reduction because it permits up to ``n-1`` rounded
    additions independent of their tree ordering.
    """

    count = int(values.shape[-1])
    center = torch.sum(values, dim=-1, dtype=torch.float64)
    if count <= 1:
        return center, torch.zeros_like(center)
    mass = torch.sum(torch.abs(values), dim=-1, dtype=torch.float64)
    if not bool(torch.isfinite(center).all().item()) or not bool(
        torch.isfinite(mass).all().item()
    ):
        _refuse(
            V1PhaseAGlobalQOutsideFaceRefusalCode.NUMERICAL_REFUSAL,
            "route-tile reduction overflowed before outward enclosure",
        )
    gamma_value = _gamma(count - 1)
    denominator_lower = math.nextafter(1.0 - gamma_value, -math.inf)
    if denominator_lower <= 0.0:
        _refuse(
            V1PhaseAGlobalQOutsideFaceRefusalCode.NUMERICAL_REFUSAL,
            "route-tile reduction error denominator is nonpositive",
        )
    gamma_tensor = torch.full_like(mass, gamma_value)
    delta_value = math.nextafter((count - 1) * _MIN_SUBNORMAL, math.inf)
    delta_tensor = torch.full_like(mass, delta_value)
    numerator_upper = _add_up(_mul_up(gamma_tensor, mass), delta_tensor)
    error_upper = _up(numerator_upper / denominator_lower)
    return center, error_upper


def _tile_sum_lower(values: Any) -> Any:
    if int(values.shape[-1]) == 0:
        return torch.zeros(values.shape[:-1], dtype=torch.float64, device=values.device)
    center, error_upper = _tile_sum_error_upper(values)
    if int(values.shape[-1]) == 1:
        return center
    return _down(center - error_upper)


def _tile_sum_upper(values: Any) -> Any:
    if int(values.shape[-1]) == 0:
        return torch.zeros(values.shape[:-1], dtype=torch.float64, device=values.device)
    center, error_upper = _tile_sum_error_upper(values)
    if int(values.shape[-1]) == 1:
        return center
    return _up(center + error_upper)


def _dot_bounds_nonnegative(left: Any, right: Any) -> tuple[Any, Any]:
    lower = torch.zeros((), dtype=torch.float64, device=left.device)
    upper = torch.zeros_like(lower)
    for index in range(int(left.numel())):
        lower = _add_down(lower, _mul_down(left[index], right[index]))
        upper = _add_up(upper, _mul_up(left[index], right[index]))
    return lower, upper


def _packed_live_slice(mask: HardMask, start: int, stop: int) -> np.ndarray:
    indices = np.arange(start, stop, dtype=np.int64)
    packed = np.frombuffer(mask.packed_bits, dtype=np.uint8)
    result = (
        (packed[np.floor_divide(indices, 8)] >> (7 - np.remainder(indices, 8))) & 1
    ).astype(np.bool_, copy=False)
    result.setflags(write=False)
    return result


def _block_ordinals(
    problem: V1RelaxedKKTProblem,
    start: int,
    stop: int,
) -> np.ndarray:
    result = np.empty(stop - start, dtype=np.int64)
    filled = 0
    for ordinal, block in enumerate(problem.block_capacities):
        left = max(start, block.start)
        right = min(stop, block.stop)
        if left < right:
            result[left - start : right - start] = ordinal
            filled += right - left
    if filled != stop - start:
        _refuse(
            V1PhaseAGlobalQOutsideFaceRefusalCode.BLOCK_CAPACITY_MISMATCH,
            "compact block capacities do not cover one route tile",
        )
    return result


@dataclass(frozen=True, slots=True)
class V1PhaseAGlobalQOutsideFaceDesign:
    design_id: str = "canonical-global-q-outside-face-v1"
    hbm_ceiling_fraction: float = 0.85
    cpu_reference_route_limit: int = 4096
    private_reference_device_total_memory_bytes: int = 1 << 30
    require_deterministic_algorithms: bool = True
    schema: str = PHASE_A_GLOBAL_Q_OUTSIDE_FACE_DESIGN_SCHEMA

    def __post_init__(self) -> None:
        _identity(self.design_id, name="design_id")
        if self.hbm_ceiling_fraction != 0.85:
            raise ValueError("canonical outside-face HBM ceiling is exactly 85%")
        _positive_int(self.cpu_reference_route_limit, name="cpu_reference_route_limit")
        _positive_int(
            self.private_reference_device_total_memory_bytes,
            name="private_reference_device_total_memory_bytes",
        )
        if self.require_deterministic_algorithms is not True:
            raise ValueError("outside-face execution requires deterministic algorithms")
        if self.schema != PHASE_A_GLOBAL_Q_OUTSIDE_FACE_DESIGN_SCHEMA:
            raise ValueError("unsupported outside-face design schema")

    @property
    def design_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "design-")


@dataclass(frozen=True, slots=True)
class V1PhaseAGlobalQOutsideFaceHBMForecast:
    design_hash: str
    support_family_receipt_hash: str
    population_mjv_forecast_hash: str
    dual_proposal_receipt_hash: str
    measurement_device_identity: str
    route_count: int
    direction_count: int
    shell_size: int
    block_count: int
    source_route_tile_size: int
    component_bytes: tuple[tuple[str, int], ...]
    projected_incremental_peak_bytes: int
    current_reserved_device_bytes: int
    projected_reserved_device_bytes: int
    device_total_memory_bytes: int
    device_free_memory_bytes: int
    hbm_ceiling_fraction: float
    hbm_ceiling_bytes: int
    admitted_by_ceiling: bool
    admitted_by_free_memory: bool
    admitted: bool
    production_cuda_execution: bool
    private_cpu_reference_execution: bool
    source_support_tile_storage_excluded: bool = True
    route_scale_d2h_forecasted: bool = False
    schema: str = PHASE_A_GLOBAL_Q_OUTSIDE_FACE_FORECAST_SCHEMA
    _admission_memory: InitVar[tuple[int, int] | None] = None
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _admission_memory: tuple[int, int] | None, _issuance_token: object | None) -> None:
        available, allocated = _admission_memory or (self.device_free_memory_bytes, self.current_reserved_device_bytes)
        if _issuance_token is not _FORECAST_TOKEN:
            raise PermissionError("outside-face forecasts require their issuer")
        for name in (
            "design_hash",
            "support_family_receipt_hash",
            "population_mjv_forecast_hash",
            "dual_proposal_receipt_hash",
            "measurement_device_identity",
        ):
            _identity(getattr(self, name), name=name)
        for name in (
            "route_count",
            "direction_count",
            "shell_size",
            "source_route_tile_size",
            "device_total_memory_bytes",
        ):
            _positive_int(getattr(self, name), name=name)
        for name in (
            "block_count",
            "projected_incremental_peak_bytes",
            "current_reserved_device_bytes",
            "projected_reserved_device_bytes",
            "device_free_memory_bytes",
            "hbm_ceiling_bytes",
        ):
            _nonnegative_int(getattr(self, name), name=name)
        if (
            type(self.component_bytes) is not tuple
            or any(
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not str
                or not item[0]
                or type(item[1]) is not int
                or item[1] < 0
                for item in self.component_bytes
            )
            or self.projected_incremental_peak_bytes
            != sum(value for _name, value in self.component_bytes)
            or self.projected_reserved_device_bytes
            != self.current_reserved_device_bytes + self.projected_incremental_peak_bytes
            or self.hbm_ceiling_fraction != 0.85
            or self.hbm_ceiling_bytes != math.floor(0.85 * self.device_total_memory_bytes)
            or self.admitted_by_ceiling
            != (allocated + self.projected_incremental_peak_bytes <= self.hbm_ceiling_bytes)
            or self.admitted_by_free_memory
            != (self.projected_incremental_peak_bytes <= available)
            or self.admitted != (self.admitted_by_ceiling and self.admitted_by_free_memory)
            or self.production_cuda_execution == self.private_cpu_reference_execution
            or self.source_support_tile_storage_excluded is not True
            or self.route_scale_d2h_forecasted is not False
            or self.schema != PHASE_A_GLOBAL_Q_OUTSIDE_FACE_FORECAST_SCHEMA
        ):
            _refuse(
                V1PhaseAGlobalQOutsideFaceRefusalCode.AUTHORITY_VIOLATION,
                "outside-face forecast crossed its bounded-workspace contract",
            )
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                item.name: getattr(self, item.name)
                for item in fields(self)
                if not item.name.startswith("_")
            }
        )

    def _compute_hash(self) -> str:
        return _payload_hash(self._payload(), role="forecast")

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1PhaseAGlobalQOutsideFaceRefusalCode.MUTATED_CUSTODY,
                "outside-face HBM forecast no longer replays",
            )
        return observed

    @property
    def forecast_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1PhaseAGlobalQOutsideFaceReceipt:
    design_hash: str
    forecast_hash: str
    joint_preoutcome_registration_hash: str
    support_family_receipt_hash: str
    population_mjv_forecast_hash: str
    population_mjv_execution_receipt_hash: str
    support_plane_scan_receipt_hash: str
    dual_proposal_receipt_hash: str
    relaxed_kkt_problem_hash: str
    action_pair_alias_source_join_receipt_hash: str
    robust_shell_decision_receipt_hash: str
    action_pair_hash: str
    parent_mask_hash: str
    selected_physical_action_identity_hash: str
    selected_packed_action_hash: str
    selected_resulting_mask_hash: str
    route_axis_hash: str
    route_id_axis_hash: str
    joint_direction_axis_hash: str
    shell_axis_hash: str
    block_axis_hash: str
    route_count: int
    scanned_route_count: int
    scanned_tile_count: int
    direction_count: int
    shell_size: int
    block_count: int
    selected_total_count: int
    selected_outside_count: int
    selected_shell_count: int
    excluded_outside_eligible_count: int
    selected_outside_block_counts: tuple[int, ...]
    block_capacities: tuple[int, ...]
    remaining_shell_block_capacities: tuple[int, ...]
    support_coefficient_lower_digest: str
    support_coefficient_upper_digest: str
    outside_support_lower_sum_digest: str
    outside_support_upper_sum_digest: str
    support_rhs_upper_digest: str
    shell_resource_digest: str
    support_multiplier_digest: str
    block_multiplier_digest: str
    outside_resource_bounds_digest: str
    full_dual_bounds_digest: str
    full_dual_lower: float
    full_dual_upper: float
    outside_resource_lower: float
    outside_resource_upper: float
    selected_reduced_advantage_lower_min: float | None
    selected_reduced_advantage_upper_min: float | None
    excluded_reduced_advantage_lower_max: float | None
    excluded_reduced_advantage_upper_max: float | None
    outside_sign_status: V1PhaseAGlobalQOutsideSignStatus
    support_plane_tile_receipt_hashes: tuple[str, ...]
    route_ranges: tuple[tuple[int, int], ...]
    host_to_device_payload_bytes: int
    compact_device_to_host_scalar_count: int
    compact_device_to_host_payload_bytes: int
    projected_incremental_peak_bytes: int
    measurement_device_identity: str
    production_cuda_execution: bool
    private_cpu_reference_execution: bool
    exact_direct_factor_two_endpoint_bounds_consumed: bool = True
    registered_lower_support_rows_used_for_outer_lp: bool = True
    frozen_multipliers_used: bool = True
    gamma_bounded_route_tile_reductions: bool = True
    membership_derived_from_parent_and_selected_child_packed_bits: bool = True
    contiguous_block_cursor_used: bool = True
    one_live_support_tile_enforced: bool = True
    complete_route_cover: bool = True
    array_free_receipt: bool = True
    route_scale_d2h: bool = False
    route_scale_membership_retained: bool = False
    route_scale_coefficient_bank_retained: bool = False
    direction_route_bank_retained: bool = False
    compact_face_equality_authority: bool = False
    hard_face_margin_authority: bool = False
    d_stability_authority: bool = False
    c_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    runtime_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    algorithm: str = PHASE_A_GLOBAL_Q_OUTSIDE_FACE_ALGORITHM
    authority_scope: str = PHASE_A_GLOBAL_Q_OUTSIDE_FACE_AUTHORITY
    schema: str = PHASE_A_GLOBAL_Q_OUTSIDE_FACE_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _RECEIPT_TOKEN:
            raise PermissionError("outside-face receipts require their issuer")
        self._validate()
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _validate(self) -> None:
        for name in (
            "design_hash",
            "forecast_hash",
            "joint_preoutcome_registration_hash",
            "support_family_receipt_hash",
            "population_mjv_forecast_hash",
            "population_mjv_execution_receipt_hash",
            "support_plane_scan_receipt_hash",
            "dual_proposal_receipt_hash",
            "relaxed_kkt_problem_hash",
            "action_pair_alias_source_join_receipt_hash",
            "robust_shell_decision_receipt_hash",
            "action_pair_hash",
            "parent_mask_hash",
            "selected_physical_action_identity_hash",
            "selected_packed_action_hash",
            "selected_resulting_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "joint_direction_axis_hash",
            "shell_axis_hash",
            "block_axis_hash",
            "support_coefficient_lower_digest",
            "support_coefficient_upper_digest",
            "outside_support_lower_sum_digest",
            "outside_support_upper_sum_digest",
            "support_rhs_upper_digest",
            "shell_resource_digest",
            "support_multiplier_digest",
            "block_multiplier_digest",
            "outside_resource_bounds_digest",
            "full_dual_bounds_digest",
            "measurement_device_identity",
        ):
            _identity(getattr(self, name), name=name)
        for name in (
            "route_count",
            "scanned_route_count",
            "scanned_tile_count",
            "direction_count",
            "shell_size",
        ):
            _positive_int(getattr(self, name), name=name)
        for name in (
            "block_count",
            "selected_total_count",
            "selected_outside_count",
            "selected_shell_count",
            "excluded_outside_eligible_count",
            "host_to_device_payload_bytes",
            "compact_device_to_host_scalar_count",
            "compact_device_to_host_payload_bytes",
            "projected_incremental_peak_bytes",
        ):
            _nonnegative_int(getattr(self, name), name=name)
        for name in (
            "full_dual_lower",
            "full_dual_upper",
            "outside_resource_lower",
            "outside_resource_upper",
        ):
            _finite(getattr(self, name), name=name)
        for name in (
            "selected_reduced_advantage_lower_min",
            "selected_reduced_advantage_upper_min",
            "excluded_reduced_advantage_lower_max",
            "excluded_reduced_advantage_upper_max",
        ):
            value = getattr(self, name)
            if value is not None:
                _finite(value, name=name)
        if type(self.outside_sign_status) is not V1PhaseAGlobalQOutsideSignStatus:
            raise TypeError("outside sign status must be exact")
        if (
            self.scanned_route_count != self.route_count
            or self.selected_total_count
            != self.selected_outside_count + self.selected_shell_count
            or len(self.selected_outside_block_counts) != self.block_count
            or len(self.block_capacities) != self.block_count
            or len(self.remaining_shell_block_capacities) != self.block_count
            or any(
                cap - used != remaining or remaining < 0
                for used, cap, remaining in zip(
                    self.selected_outside_block_counts,
                    self.block_capacities,
                    self.remaining_shell_block_capacities,
                    strict=True,
                )
            )
            or self.full_dual_lower > self.full_dual_upper
            or self.outside_resource_lower > self.outside_resource_upper
            or self.compact_device_to_host_payload_bytes
            != self.compact_device_to_host_scalar_count * _F64_BYTES
            or len(self.support_plane_tile_receipt_hashes) != self.scanned_tile_count
            or len(self.route_ranges) != self.scanned_tile_count
            or not self.route_ranges
            or self.route_ranges[0][0] != 0
            or self.route_ranges[-1][1] != self.route_count
            or any(
                self.route_ranges[index][1] != self.route_ranges[index + 1][0]
                for index in range(len(self.route_ranges) - 1)
            )
            or self.production_cuda_execution == self.private_cpu_reference_execution
        ):
            _refuse(
                V1PhaseAGlobalQOutsideFaceRefusalCode.SOURCE_MISMATCH,
                "outside-face compact counts, bounds, or route cover diverged",
            )
        positive = (
            self.exact_direct_factor_two_endpoint_bounds_consumed,
            self.registered_lower_support_rows_used_for_outer_lp,
            self.frozen_multipliers_used,
            self.gamma_bounded_route_tile_reductions,
            self.membership_derived_from_parent_and_selected_child_packed_bits,
            self.contiguous_block_cursor_used,
            self.one_live_support_tile_enforced,
            self.complete_route_cover,
            self.array_free_receipt,
        )
        closed = (
            self.route_scale_d2h,
            self.route_scale_membership_retained,
            self.route_scale_coefficient_bank_retained,
            self.direction_route_bank_retained,
            self.compact_face_equality_authority,
            self.hard_face_margin_authority,
            self.d_stability_authority,
            self.c_measurement_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.runtime_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if (
            any(type(value) is not bool or value is not True for value in positive)
            or any(type(value) is not bool or value is not False for value in closed)
            or self.algorithm != PHASE_A_GLOBAL_Q_OUTSIDE_FACE_ALGORITHM
            or self.authority_scope != PHASE_A_GLOBAL_Q_OUTSIDE_FACE_AUTHORITY
            or self.schema != PHASE_A_GLOBAL_Q_OUTSIDE_FACE_SCHEMA
        ):
            _refuse(
                V1PhaseAGlobalQOutsideFaceRefusalCode.AUTHORITY_VIOLATION,
                "outside-face receipt crossed its scan-only authority",
            )

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                item.name: getattr(self, item.name)
                for item in fields(self)
                if not item.name.startswith("_")
            }
        )

    def _compute_hash(self) -> str:
        return _payload_hash(self._payload(), role="receipt")

    def assert_integrity(self) -> str:
        self._validate()
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1PhaseAGlobalQOutsideFaceRefusalCode.MUTATED_CUSTODY,
                "outside-face receipt no longer replays",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


_COMPACT_NAMES: Final = (
    "support_coefficient_lower",
    "support_coefficient_upper",
    "outside_support_lower_sum",
    "outside_support_upper_sum",
    "support_rhs_upper",
    "shell_resource",
    "support_multipliers",
    "block_multipliers",
    "outside_resource_bounds",
    "full_dual_bounds",
)


class V1PhaseAGlobalQOutsideFaceLease:
    __slots__ = ("_owner", "_token", "_closed") + _COMPACT_NAMES

    def __init__(self, owner: "V1PhaseAGlobalQOutsideFaceHandle", token: object, values: tuple[Any, ...]) -> None:
        self._owner = owner
        self._token = token
        self._closed = False
        for name, value in zip(_COMPACT_NAMES, values, strict=True):
            setattr(self, name, value)

    def __enter__(self) -> "V1PhaseAGlobalQOutsideFaceLease":
        self._owner.assert_live()
        return self

    def close(self) -> None:
        if not self._closed:
            self._owner._close(self._token)
            self._closed = True

    def __exit__(self, *_: object) -> bool:
        self.close()
        return False

    def __reduce__(self) -> NoReturn:
        raise TypeError("outside-face leases are nonserializable")


class V1PhaseAGlobalQOutsideFaceHandle:
    __slots__ = ("receipt", "_values", "_guards", "_invariants", "_active_token", "_evicted")

    def __init__(
        self,
        *,
        receipt: V1PhaseAGlobalQOutsideFaceReceipt,
        values: tuple[Any, ...],
        _issuance_token: object | None = None,
    ) -> None:
        if _issuance_token is not _HANDLE_TOKEN:
            raise PermissionError("outside-face handles require their issuer")
        if type(receipt) is not V1PhaseAGlobalQOutsideFaceReceipt or len(values) != len(_COMPACT_NAMES):
            raise TypeError("outside-face handle requires its exact compact result")
        expected_shapes = (
            (receipt.direction_count, receipt.shell_size),
            (receipt.direction_count, receipt.shell_size),
            (receipt.direction_count,),
            (receipt.direction_count,),
            (receipt.direction_count,),
            (receipt.shell_size,),
            (receipt.direction_count,),
            (receipt.block_count,),
            (2,),
            (2,),
        )
        if torch is None or any(
            type(value) is not torch.Tensor
            or value.dtype is not torch.float64
            or tuple(int(item) for item in value.shape) != shape
            or not value.is_contiguous()
            or value.requires_grad
            or str(value.device) != receipt.measurement_device_identity
            or not bool(torch.isfinite(value).all().item())
            for value, shape in zip(values, expected_shapes, strict=True)
        ):
            _refuse(
                V1PhaseAGlobalQOutsideFaceRefusalCode.MUTATED_CUSTODY,
                "outside-face compact tensors do not match their receipt",
            )
        self.receipt = receipt
        self._values = values
        self._guards = tuple(value.detach().clone() for value in values)
        self._invariants = tuple(_tensor_invariant(value) for value in values)
        self._active_token: object | None = None
        self._evicted = False
        self.assert_live()

    @property
    def is_evicted(self) -> bool:
        return self._evicted

    def assert_live(self) -> None:
        if self._evicted:
            _refuse(V1PhaseAGlobalQOutsideFaceRefusalCode.EVICTED, "outside-face handle was evicted")
        self.receipt.assert_integrity()
        if any(
            _tensor_invariant(value) != invariant or not bool(torch.equal(value, guard))
            for value, invariant, guard in zip(self._values, self._invariants, self._guards, strict=True)
        ):
            _refuse(
                V1PhaseAGlobalQOutsideFaceRefusalCode.MUTATED_CUSTODY,
                "outside-face compact tensor custody mutated",
            )

    def borrow(self) -> V1PhaseAGlobalQOutsideFaceLease:
        self.assert_live()
        if self._active_token is not None:
            _refuse(V1PhaseAGlobalQOutsideFaceRefusalCode.ACTIVE_LEASE, "only one outside-face lease may be live")
        token = object()
        self._active_token = token
        return V1PhaseAGlobalQOutsideFaceLease(self, token, self._values)

    def _close(self, token: object) -> None:
        if token is not self._active_token:
            _refuse(V1PhaseAGlobalQOutsideFaceRefusalCode.ACTIVE_LEASE, "outside-face lease token is stale")
        try:
            self.assert_live()
        finally:
            self._active_token = None

    def evict(self) -> None:
        self.assert_live()
        if self._active_token is not None:
            _refuse(V1PhaseAGlobalQOutsideFaceRefusalCode.ACTIVE_LEASE, "cannot evict under an active lease")
        self._values = ()
        self._guards = ()
        self._invariants = ()
        self._evicted = True

    def __reduce__(self) -> NoReturn:
        raise TypeError("outside-face handles are nonserializable")


@dataclass(frozen=True, slots=True)
class V1PhaseAGlobalQOutsideFaceResult:
    receipt: V1PhaseAGlobalQOutsideFaceReceipt
    compact_handle: V1PhaseAGlobalQOutsideFaceHandle = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.receipt) is not V1PhaseAGlobalQOutsideFaceReceipt or type(self.compact_handle) is not V1PhaseAGlobalQOutsideFaceHandle:
            raise TypeError("outside-face result requires exact components")
        self.compact_handle.assert_live()
        if self.compact_handle.receipt.receipt_hash != self.receipt.receipt_hash:
            raise ValueError("outside-face result receipt and compact handle diverged")


def _forecast(
    *,
    design: V1PhaseAGlobalQOutsideFaceDesign,
    support: Any,
    population_mjv_stream: V1LiteralJJointPopulationMvStreamCUDAHandle,
    proposal: Any,
    block_count: int,
    production: bool,
) -> V1PhaseAGlobalQOutsideFaceHBMForecast:
    return _forecast_from_source_metadata(
        design=design,
        support=support,
        population_mjv_forecast_hash=population_mjv_stream.forecast.forecast_hash,
        source_route_tile_size=population_mjv_stream.forecast.route_tile_size,
        proposal=proposal,
        block_count=block_count,
        production=production,
    )


def _forecast_from_source_metadata(
    *,
    design: V1PhaseAGlobalQOutsideFaceDesign,
    support: Any,
    population_mjv_forecast_hash: str,
    source_route_tile_size: int,
    proposal: Any,
    block_count: int,
    production: bool,
) -> V1PhaseAGlobalQOutsideFaceHBMForecast:
    route_tile = _positive_int(source_route_tile_size, name="source_route_tile_size")
    directions = support.anchor_count
    shell = support.shell_size
    compact_scalars = 2 * directions * shell + 4 * directions + shell + block_count + 4
    components = (
        ("support_interval_workspace", 4 * directions * route_tile * _F64_BYTES),
        ("route_scalar_workspace", 7 * route_tile * _F64_BYTES),
        ("route_boolean_workspace", 3 * route_tile),
        ("block_ordinal_workspace", route_tile * np.dtype(np.int64).itemsize),
        ("compact_results_and_guards", 2 * compact_scalars * _F64_BYTES),
    )
    incremental = sum(value for _name, value in components)
    if production:
        device = torch.device(proposal.measurement_device_identity)
        properties = torch.cuda.get_device_properties(device)
        total = int(properties.total_memory)
        reserved = int(torch.cuda.memory_reserved(device))
        free, _observed_total = torch.cuda.mem_get_info(device)
        free = int(free)
        available = cuda_reusable_memory_bytes(device, framework=torch, free_bytes=free)
        allocated = int(torch.cuda.memory_allocated(device))
    else:
        total = design.private_reference_device_total_memory_bytes
        reserved = 0
        free = total
        available, allocated = free, reserved
    peak = reserved + incremental
    ceiling = math.floor(total * 0.85)
    return V1PhaseAGlobalQOutsideFaceHBMForecast(
        design_hash=design.design_hash,
        support_family_receipt_hash=support.receipt_hash,
        population_mjv_forecast_hash=population_mjv_forecast_hash,
        dual_proposal_receipt_hash=proposal.receipt_hash,
        measurement_device_identity=proposal.measurement_device_identity,
        route_count=support.route_count,
        direction_count=directions,
        shell_size=shell,
        block_count=block_count,
        source_route_tile_size=route_tile,
        component_bytes=components,
        projected_incremental_peak_bytes=incremental,
        current_reserved_device_bytes=reserved,
        projected_reserved_device_bytes=peak,
        device_total_memory_bytes=total,
        device_free_memory_bytes=free,
        hbm_ceiling_fraction=0.85,
        hbm_ceiling_bytes=ceiling,
        admitted_by_ceiling=allocated + incremental <= ceiling,
        admitted_by_free_memory=incremental <= available,
        admitted=allocated + incremental <= ceiling and incremental <= available,
        production_cuda_execution=production,
        private_cpu_reference_execution=not production,
        _admission_memory=(available, allocated),
        _issuance_token=_FORECAST_TOKEN,
    )


def _execute(
    *,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    support_family: (
        V1PhaseAGlobalQSupportFamilyResult
        | V1PhaseAGlobalQD1SupportFamilyResult
    ),
    population_mjv_stream: (
        V1LiteralJJointPopulationMvStreamCUDAHandle
        | V1LiteralJJointD1PopulationMvStreamCUDAHandle
        | None
    ),
    support_plane_scan: V1PhaseAGlobalQSupportPlaneScanHandle | None,
    compact_dual_source: V1PhaseAGlobalQCompactOuterLPCUDASource | None = None,
    dual_proposal: V1PhaseAGlobalQOuterLPDualProposalResult,
    action_pair_alias_source_join: V1SampledV7ActionPairAliasSourceJoinReceipt,
    robust_shell_decision: V1SampledV7PhaseARobustShellDecisionReceipt,
    parent_mask: HardMask,
    relaxed_kkt_problem: V1RelaxedKKTProblem,
    design: V1PhaseAGlobalQOutsideFaceDesign,
    private_cpu_reference: bool,
) -> V1PhaseAGlobalQOutsideFaceResult:
    compact_source_mode = compact_dual_source is not None
    if compact_source_mode == (
        population_mjv_stream is not None or support_plane_scan is not None
    ):
        _refuse(
            V1PhaseAGlobalQOutsideFaceRefusalCode.INVALID_TYPE,
            "outside-face execution requires exactly one route coefficient source",
        )
    exact_types = (
        (joint_preoutcome_registration, V1LiteralJJointAffineShellPreoutcomeRegistration),
        (dual_proposal, V1PhaseAGlobalQOuterLPDualProposalResult),
        (action_pair_alias_source_join, V1SampledV7ActionPairAliasSourceJoinReceipt),
        (robust_shell_decision, V1SampledV7PhaseARobustShellDecisionReceipt),
        (parent_mask, HardMask),
        (relaxed_kkt_problem, V1RelaxedKKTProblem),
        (design, V1PhaseAGlobalQOutsideFaceDesign),
    )
    if torch is None or any(type(value) is not expected for value, expected in exact_types):
        _refuse(V1PhaseAGlobalQOutsideFaceRefusalCode.INVALID_TYPE, "outside-face execution requires exact source owners")
    support_source_mode = _support_family_source_mode(support_family)
    if compact_source_mode:
        if type(compact_dual_source) is not V1PhaseAGlobalQCompactOuterLPCUDASource:
            _refuse(V1PhaseAGlobalQOutsideFaceRefusalCode.INVALID_TYPE, "outside-face compact source mode requires exact CUDA source")
    else:
        expected_stream_type = (
            V1LiteralJJointPopulationMvStreamCUDAHandle
            if support_source_mode == _LEGACY_PROJECTION_SUPPORT_SOURCE_MODE
            else V1LiteralJJointD1PopulationMvStreamCUDAHandle
        )
        if (
            type(population_mjv_stream) is not expected_stream_type
            or type(support_plane_scan) is not V1PhaseAGlobalQSupportPlaneScanHandle
        ):
            _refuse(
                V1PhaseAGlobalQOutsideFaceRefusalCode.INVALID_TYPE,
                "outside-face scan mode requires exact source-mode-matched stream "
                "and scan owners",
            )
        if (
            getattr(support_plane_scan, "_source_mode", None) != support_source_mode
            or getattr(support_plane_scan, "_support", None) is not support_family
            or getattr(support_plane_scan, "_stream", None) is not population_mjv_stream
        ):
            _refuse(
                V1PhaseAGlobalQOutsideFaceRefusalCode.SOURCE_MISMATCH,
                "outside-face scan crossed the legacy/D1 source mode or live parent custody",
            )
    try:
        prereg_hash = joint_preoutcome_registration.assert_integrity()
        support_family.device_handle.assert_live()
        support_hash = support_family.receipt.assert_integrity()
        if compact_source_mode:
            assert compact_dual_source is not None
            compact_source_hash = compact_dual_source.assert_live()
        else:
            assert population_mjv_stream is not None
            assert support_plane_scan is not None
            population_mjv_stream.assert_live()
            support_plane_scan.assert_live()
            compact_source_hash = ""
        dual_proposal.device_handle.assert_live()
        proposal_hash = dual_proposal.receipt.assert_integrity()
        source_join_hash = action_pair_alias_source_join.assert_integrity()
        robust_hash = robust_shell_decision.assert_integrity()
        pair = action_pair_alias_source_join.action_pair
        pair_hash = pair.assert_integrity()
    except V1PhaseAGlobalQOutsideFaceRefusal:
        raise
    except Exception as error:
        _refuse(
            V1PhaseAGlobalQOutsideFaceRefusalCode.STALE_SOURCE,
            f"one outside-face source failed replay: {type(error).__name__}: {error}",
        )
    support = support_family.receipt
    proposal = dual_proposal.receipt
    robust = robust_shell_decision
    if robust.status is not V1SampledV7PhaseARobustDecisionStatus.STABLE_PHYSICAL_ACTION_PREMISE:
        _refuse(V1PhaseAGlobalQOutsideFaceRefusalCode.NO_STABLE_PHYSICAL_ACTION, "outside-face scan requires one stable physical action")
    if not compact_source_mode:
        assert support_plane_scan is not None
        assert population_mjv_stream is not None
        if support_source_mode == _LEGACY_PROJECTION_SUPPORT_SOURCE_MODE:
            stream_opened = bool(
                population_mjv_stream._source_rows_consumed
                or population_mjv_stream._route_ranges
            )
        else:
            stream_opened = bool(
                population_mjv_stream._d1_source_consumed
                or population_mjv_stream._residual_source_consumed
                or population_mjv_stream._route_ranges
            )
        if (
            support_plane_scan._support is not support_family
            or support_plane_scan._stream is not population_mjv_stream
            or support_plane_scan._source_mode != support_source_mode
            or support_plane_scan._active_tile_hash is not None
            or support_plane_scan._route_ranges
            or support_plane_scan._exhausted
            or support_plane_scan._final_receipt is not None
            or stream_opened
        ):
            _refuse(V1PhaseAGlobalQOutsideFaceRefusalCode.STREAM_ALREADY_OPENED, "support/Mv scan must be unopened at outside-face admission")
    physical = tuple(pair.physical_actions)
    matches = tuple(
        item for item in physical
        if item.physical_action_identity_hash == robust.selected_physical_action_identity_hash
    )
    if len(matches) != 1:
        _refuse(V1PhaseAGlobalQOutsideFaceRefusalCode.NO_STABLE_PHYSICAL_ACTION, "stable identity has no unique source child")
    selected = matches[0]
    selected.assert_integrity()
    child = selected.packed_action.resulting_mask
    try:
        removed_count, removed_hash = child.removed_index_count_and_hash_from(parent_mask)
    except Exception as error:
        _refuse(V1PhaseAGlobalQOutsideFaceRefusalCode.ACTION_MEMBERSHIP_MISMATCH, f"selected child is not irreversible: {error}")
    blocks = tuple(relaxed_kkt_problem.block_capacities)
    if not blocks:
        _refuse(V1PhaseAGlobalQOutsideFaceRefusalCode.BLOCK_CAPACITY_MISMATCH, "canonical outside-face scan requires compact block capacities")
    if compact_source_mode:
        assert compact_dual_source is not None
        compact_source_receipt = compact_dual_source.receipt
        population_mjv_forecast_hash = compact_source_receipt.population_mjv_forecast_hash
        population_mjv_execution_receipt_hash = (
            compact_source_receipt.population_mjv_execution_receipt_hash
        )
        source_route_tile_size = max(
            stop - start for start, stop in compact_source_receipt.route_ranges
        )
        source_route_count = compact_source_receipt.route_count
        source_direction_count = compact_source_receipt.support_plane_count
        source_production_cuda = compact_source_receipt.production_cuda_authority
        source_private_cpu = compact_source_receipt.private_cpu_reference
        source_joint_prereg_hash = compact_source_receipt.joint_preoutcome_registration_hash
    else:
        assert population_mjv_stream is not None
        compact_source_receipt = None
        forecast_source = population_mjv_stream.forecast
        population_mjv_forecast_hash = forecast_source.forecast_hash
        population_mjv_execution_receipt_hash = ""
        source_route_tile_size = (
            forecast_source.route_tile_size
            if support_source_mode == _LEGACY_PROJECTION_SUPPORT_SOURCE_MODE
            else forecast_source.maximum_output_route_tile_size
        )
        source_route_count = forecast_source.route_count
        source_direction_count = forecast_source.direction_count
        source_production_cuda = forecast_source.production_cuda_execution
        source_private_cpu = forecast_source.private_cpu_reference_execution
        source_joint_prereg_hash = forecast_source.joint_preoutcome_registration_hash
    if (
        proposal_hash != dual_proposal.device_handle.receipt.receipt_hash
        or proposal.support_family_receipt_hash != support_hash
        or proposal.relaxed_kkt_problem_hash != relaxed_kkt_problem.problem_hash
        or proposal.action_pair_alias_source_join_receipt_hash != source_join_hash
        or proposal.robust_shell_decision_receipt_hash != robust_hash
        or proposal.action_pair_hash != pair_hash
        or proposal.parent_mask_hash != parent_mask.mask_hash
        or proposal.selected_physical_action_identity_hash != selected.physical_action_identity_hash
        or proposal.selected_packed_action_hash != selected.packed_action_hash
        or proposal.selected_resulting_mask_hash != child.mask_hash
        or proposal.selected_removed_route_index_hash != removed_hash
        or proposal.selected_removed_count != removed_count
        or proposal.shell_indices != tuple(joint_preoutcome_registration.shell_indices)
        or proposal.shell_indices != tuple(robust.shell_indices)
        or proposal.anchor_direction_ids != support.anchor_direction_ids
        or proposal.block_ids != tuple(item.block_id for item in blocks)
        or proposal.block_hashes != tuple(item.block_hash for item in blocks)
        or support.joint_preoutcome_registration_hash != prereg_hash
        or support.relaxed_kkt_problem_hash != relaxed_kkt_problem.problem_hash
        or support.route_axis_hash != relaxed_kkt_problem.route_axis_hash
        or support.route_axis_hash != parent_mask.route_axis_hash
        or support.route_count != relaxed_kkt_problem.route_count
        or support.route_count != parent_mask.route_count
        or support.current_mask_hash != parent_mask.mask_hash
        or joint_preoutcome_registration.current_mask_hash != parent_mask.mask_hash
        or joint_preoutcome_registration.joint_direction_axis_hash != support.joint_direction_axis_hash
        or source_joint_prereg_hash != prereg_hash
        or source_route_count != support.route_count
        or source_direction_count != support.anchor_count
        or (
            compact_source_mode
            and (
                compact_source_hash != compact_dual_source.receipt.receipt_hash
                or compact_source_receipt.support_family_receipt_hash != support_hash
                or compact_source_receipt.relaxed_kkt_problem_hash != relaxed_kkt_problem.problem_hash
                or compact_source_receipt.route_axis_hash != support.route_axis_hash
                or compact_source_receipt.route_id_axis_hash != support.route_id_axis_hash
                or compact_source_receipt.current_mask_hash != parent_mask.mask_hash
                or compact_source_receipt.joint_direction_axis_hash != support.joint_direction_axis_hash
                or compact_source_receipt.direction_ids != support.anchor_direction_ids
                or compact_source_receipt.support_plane_count != support.anchor_count
                or compact_source_receipt.block_count != len(blocks)
            )
        )
    ):
        _refuse(V1PhaseAGlobalQOutsideFaceRefusalCode.SOURCE_MISMATCH, "outside-face source graph or compact axes diverged")
    production = bool(support.production_source_authority)
    if production == private_cpu_reference:
        _refuse(
            V1PhaseAGlobalQOutsideFaceRefusalCode.CPU_PRODUCTION_FALLBACK if production else V1PhaseAGlobalQOutsideFaceRefusalCode.DEVICE_REQUIRED,
            "outside-face production/private execution mode does not match source authority",
        )
    if (
        proposal.production_source_authority != production
        or source_production_cuda != production
        or support.private_cpu_reference == production
        or proposal.private_cpu_reference == production
        or source_private_cpu == production
    ):
        _refuse(V1PhaseAGlobalQOutsideFaceRefusalCode.SOURCE_MISMATCH, "outside-face source modes disagree")
    if production:
        if not proposal.measurement_device_identity.startswith("cuda:"):
            _refuse(V1PhaseAGlobalQOutsideFaceRefusalCode.DEVICE_REQUIRED, "production outside-face scan requires CUDA")
        if not torch.are_deterministic_algorithms_enabled():
            _refuse(V1PhaseAGlobalQOutsideFaceRefusalCode.DETERMINISM_DISABLED, "production outside-face scan requires deterministic torch algorithms")
    elif support.route_count > design.cpu_reference_route_limit:
        _refuse(V1PhaseAGlobalQOutsideFaceRefusalCode.CPU_PRODUCTION_FALLBACK, "private CPU reference route cap exceeded")
    forecast = _forecast_from_source_metadata(
        design=design,
        support=support,
        population_mjv_forecast_hash=population_mjv_forecast_hash,
        source_route_tile_size=source_route_tile_size,
        proposal=proposal,
        block_count=len(blocks),
        production=production,
    )
    if not forecast.admitted:
        _refuse(V1PhaseAGlobalQOutsideFaceRefusalCode.HBM_FORECAST_EXCEEDED, "outside-face workspace exceeds the pre-outcome 85% HBM ceiling")
    device = torch.device(proposal.measurement_device_identity)
    directions = support.anchor_count
    shell_indices = tuple(proposal.shell_indices)
    shell_offset = {physical: index for index, physical in enumerate(shell_indices)}
    shell_lower = torch.full((directions, len(shell_indices)), math.nan, dtype=torch.float64, device=device)
    shell_upper = torch.full_like(shell_lower, math.nan)
    shell_resource = torch.full((len(shell_indices),), math.nan, dtype=torch.float64, device=device)
    outside_support_lower = torch.zeros((directions,), dtype=torch.float64, device=device)
    outside_support_upper = torch.zeros_like(outside_support_lower)
    outside_resource_lower = torch.zeros((), dtype=torch.float64, device=device)
    outside_resource_upper = torch.zeros_like(outside_resource_lower)
    positive_advantage_lower = torch.zeros_like(outside_resource_lower)
    positive_advantage_upper = torch.zeros_like(outside_resource_lower)
    selected_lower_min = torch.full((), math.inf, dtype=torch.float64, device=device)
    selected_upper_min = torch.full_like(selected_lower_min, math.inf)
    excluded_lower_max = torch.full_like(selected_lower_min, -math.inf)
    excluded_upper_max = torch.full_like(selected_lower_min, -math.inf)
    selected_outside_blocks = [0] * len(blocks)
    selected_outside_count = selected_shell_count = excluded_outside_count = 0
    selected_total_count = 0
    shell_seen = [False] * len(shell_indices)
    tile_hashes: list[str] = []
    route_ranges: list[tuple[int, int]] = []
    rhs_upper: Any | None = None
    with dual_proposal.device_handle.borrow() as multiplier_lease:
        support_multipliers = multiplier_lease.support_multipliers.clone().contiguous()
        block_multipliers = multiplier_lease.block_multipliers.clone().contiguous()
    compact_bank: tuple[Any, Any, Any, Any, Any, Any, Any] | None = None
    if compact_source_mode:
        assert compact_dual_source is not None
        compact_bank = compact_dual_source.tensors()
        # The shared source receipt is bound to the exact relaxed-KKT problem
        # hash and its live tensors are custody-checked by invariant/version.
        # Rechecking eligible/block/capacity vectors route-by-route here would
        # add a hidden 9 B/route H2D validation path.  Outside-face therefore
        # trusts the source lineage and transfers only the selected/excluded
        # action masks needed by this theorem consumer.

    def consume_route_tile(
        *,
        start: int,
        stop: int,
        tile_hash: str,
        lower: Any,
        upper: Any,
        tile_rhs: Any,
        resource: Any,
        eligible: Any,
        block_axis: Any,
    ) -> None:
        nonlocal rhs_upper
        nonlocal outside_support_lower, outside_support_upper
        nonlocal outside_resource_lower, outside_resource_upper
        nonlocal positive_advantage_lower, positive_advantage_upper
        nonlocal selected_lower_min, selected_upper_min
        nonlocal excluded_lower_max, excluded_upper_max
        nonlocal selected_total_count, selected_outside_count
        nonlocal selected_shell_count, excluded_outside_count
        if route_ranges and route_ranges[-1][1] != start:
            _refuse(V1PhaseAGlobalQOutsideFaceRefusalCode.ROUTE_COVERAGE_MISMATCH, "support tiles are not contiguous")
        parent_live = _packed_live_slice(parent_mask, start, stop)
        child_live = _packed_live_slice(child, start, stop)
        selected_cpu = parent_live & ~child_live
        eligible_cpu = np.asarray(relaxed_kkt_problem.eligible_mask[start:stop], dtype=np.bool_)
        block_cpu = _block_ordinals(relaxed_kkt_problem, start, stop)
        if bool(np.any(eligible_cpu & ~parent_live)) or bool(np.any(selected_cpu & ~eligible_cpu)):
            _refuse(V1PhaseAGlobalQOutsideFaceRefusalCode.ACTION_MEMBERSHIP_MISMATCH, "eligibility/live/action membership axes disagree")
        shell_cpu = np.zeros(stop - start, dtype=np.bool_)
        for physical, offset in shell_offset.items():
            if start <= physical < stop:
                shell_cpu[physical - start] = True
                shell_seen[offset] = True
        outside_selected_cpu = selected_cpu & ~shell_cpu
        outside_excluded_cpu = eligible_cpu & ~selected_cpu & ~shell_cpu
        selected_total_count += int(np.count_nonzero(selected_cpu))
        selected_outside_count += int(np.count_nonzero(outside_selected_cpu))
        selected_shell_count += int(np.count_nonzero(selected_cpu & shell_cpu))
        excluded_outside_count += int(np.count_nonzero(outside_excluded_cpu))
        if bool(np.any(outside_selected_cpu)):
            counts = np.bincount(block_cpu[outside_selected_cpu], minlength=len(blocks))
            for ordinal, increment in enumerate(counts):
                selected_outside_blocks[ordinal] += int(increment)
        selected_mask = torch.tensor(outside_selected_cpu, dtype=torch.bool, device=device)
        excluded_mask = torch.tensor(outside_excluded_cpu, dtype=torch.bool, device=device)
        lower = lower.contiguous()
        upper = upper.contiguous()
        resource = resource.contiguous()
        eligible = eligible.contiguous()
        block_axis = block_axis.contiguous()
        if rhs_upper is None:
            rhs_upper = tile_rhs.clone().contiguous()
        elif not bool(torch.equal(rhs_upper, tile_rhs)):
            _refuse(V1PhaseAGlobalQOutsideFaceRefusalCode.SOURCE_MISMATCH, "compact support RHS changed between route tiles")
        for physical, offset in shell_offset.items():
            if start <= physical < stop:
                local = physical - start
                shell_lower[:, offset] = lower[:, local]
                shell_upper[:, offset] = upper[:, local]
                shell_resource[offset] = resource[local]
        if bool(np.any(outside_selected_cpu)):
            tile_lower = _tile_sum_lower(lower[:, selected_mask])
            # The outside-face interval encloses the registered LP row
            # ``coefficient_lower`` on both endpoints.  The population upper
            # endpoint remains a containment diagnostic only; using it here
            # would silently replace the outer LP.
            tile_upper = _tile_sum_upper(lower[:, selected_mask])
            outside_support_lower = _add_down(outside_support_lower, tile_lower)
            outside_support_upper = _add_up(outside_support_upper, tile_upper)
            resource_tile = resource[selected_mask].reshape(1, -1)
            outside_resource_lower = _add_down(outside_resource_lower, _tile_sum_lower(resource_tile).reshape(()))
            outside_resource_upper = _add_up(outside_resource_upper, _tile_sum_upper(resource_tile).reshape(()))
        weighted_lower = torch.zeros((stop - start,), dtype=torch.float64, device=device)
        weighted_upper = torch.zeros_like(weighted_lower)
        for direction in range(directions):
            lam = support_multipliers[direction]
            weighted_lower = _add_down(weighted_lower, _mul_down(lam, lower[direction]))
            weighted_upper = _add_up(weighted_upper, _mul_up(lam, lower[direction]))
        beta = block_multipliers[block_axis]
        weighted_lower = _add_down(weighted_lower, beta)
        weighted_upper = _add_up(weighted_upper, beta)
        advantage_lower = _down(resource - weighted_upper)
        advantage_upper = _up(resource - weighted_lower)
        if bool(np.any(eligible_cpu)):
            positive_lower = torch.clamp(advantage_lower[eligible], min=0.0).reshape(1, -1)
            positive_upper = torch.clamp(advantage_upper[eligible], min=0.0).reshape(1, -1)
            positive_advantage_lower = _add_down(positive_advantage_lower, _tile_sum_lower(positive_lower).reshape(()))
            positive_advantage_upper = _add_up(positive_advantage_upper, _tile_sum_upper(positive_upper).reshape(()))
        if bool(np.any(outside_selected_cpu)):
            selected_lower_min = torch.minimum(selected_lower_min, torch.amin(advantage_lower[selected_mask]))
            selected_upper_min = torch.minimum(selected_upper_min, torch.amin(advantage_upper[selected_mask]))
        if bool(np.any(outside_excluded_cpu)):
            excluded_lower_max = torch.maximum(excluded_lower_max, torch.amax(advantage_lower[excluded_mask]))
            excluded_upper_max = torch.maximum(excluded_upper_max, torch.amax(advantage_upper[excluded_mask]))
        tile_hashes.append(tile_hash)
        route_ranges.append((start, stop))

    if compact_source_mode:
        assert compact_bank is not None
        assert compact_source_receipt is not None
        bank_lower, bank_upper, bank_resource, bank_eligible, bank_blocks, bank_rhs, _bank_capacities = compact_bank
        for index, (start, stop) in enumerate(compact_source_receipt.route_ranges):
            consume_route_tile(
                start=start,
                stop=stop,
                tile_hash=compact_source_receipt.tile_receipt_hashes[index],
                lower=bank_lower[:, start:stop],
                upper=bank_upper[:, start:stop],
                tile_rhs=bank_rhs,
                resource=bank_resource[start:stop],
                eligible=bank_eligible[start:stop],
                block_axis=bank_blocks[start:stop],
            )
        scan_receipt = None
    else:
        assert support_plane_scan is not None
        while True:
            tile = support_plane_scan.next_tile()
            if tile is None:
                break
            with _owned_support_plane_tile(tile):
                receipt = tile.receipt
                start, stop = receipt.route_start, receipt.route_stop
                resource = torch.tensor(
                    np.array(relaxed_kkt_problem.resource[start:stop], dtype=np.float64, copy=True),
                    dtype=torch.float64,
                    device=device,
                )
                eligible_cpu = np.asarray(relaxed_kkt_problem.eligible_mask[start:stop], dtype=np.bool_)
                block_cpu = _block_ordinals(relaxed_kkt_problem, start, stop)
                eligible = torch.tensor(eligible_cpu, dtype=torch.bool, device=device)
                block_axis = torch.tensor(block_cpu, dtype=torch.int64, device=device)
                with tile.borrow() as lease:
                    lower, upper, tile_rhs = lease.coefficient_bounds()
                    consume_route_tile(
                        start=start,
                        stop=stop,
                        tile_hash=receipt.receipt_hash,
                        lower=lower,
                        upper=upper,
                        tile_rhs=tile_rhs,
                        resource=resource,
                        eligible=eligible,
                        block_axis=block_axis,
                    )
        scan_receipt = support_plane_scan.finalize()
    if compact_source_mode:
        assert compact_source_receipt is not None
        if (
            compact_source_receipt.support_family_receipt_hash != support_hash
            or compact_source_receipt.population_mjv_forecast_hash != population_mjv_forecast_hash
            or compact_source_receipt.joint_preoutcome_registration_hash != prereg_hash
            or compact_source_receipt.joint_direction_axis_hash != support.joint_direction_axis_hash
            or compact_source_receipt.direction_ids != support.anchor_direction_ids
            or compact_source_receipt.route_axis_hash != support.route_axis_hash
            or compact_source_receipt.route_id_axis_hash != support.route_id_axis_hash
            or compact_source_receipt.current_mask_hash != parent_mask.mask_hash
            or compact_source_receipt.route_count != support.route_count
            or compact_source_receipt.support_plane_count != support.anchor_count
            or compact_source_receipt.tile_receipt_hashes != tuple(tile_hashes)
            or compact_source_receipt.route_ranges != tuple(route_ranges)
            or compact_source_receipt.production_cuda_authority != production
            or compact_source_receipt.private_cpu_reference == production
        ):
            _refuse(
                V1PhaseAGlobalQOutsideFaceRefusalCode.SOURCE_MISMATCH,
                "shared compact CUDA source does not replay the consumed route bank",
            )
        support_plane_scan_receipt_hash = compact_source_receipt.support_plane_scan_receipt_hash
        population_execution_hash = population_mjv_execution_receipt_hash
    else:
        assert scan_receipt is not None
        if (
            type(scan_receipt) is not V1PhaseAGlobalQSupportPlaneScanReceipt
            or scan_receipt.support_family_receipt_hash != support_hash
            or scan_receipt.population_mjv_forecast_hash != population_mjv_forecast_hash
            or scan_receipt.joint_preoutcome_registration_hash != prereg_hash
            or scan_receipt.joint_direction_axis_hash != support.joint_direction_axis_hash
            or scan_receipt.direction_ids != support.anchor_direction_ids
            or scan_receipt.route_axis_hash != support.route_axis_hash
            or scan_receipt.route_id_axis_hash != support.route_id_axis_hash
            or scan_receipt.current_mask_hash != parent_mask.mask_hash
            or scan_receipt.route_count != support.route_count
            or scan_receipt.action_count != support.action_count
            or scan_receipt.shell_size != support.shell_size
            or scan_receipt.direction_count != support.anchor_count
            or scan_receipt.scanned_route_count != support.route_count
            or scan_receipt.scanned_tile_count != len(route_ranges)
            or scan_receipt.tile_receipt_hashes != tuple(tile_hashes)
            or scan_receipt.route_ranges != tuple(route_ranges)
            or scan_receipt.production_source_authority != production
            or scan_receipt.private_cpu_reference == production
            or scan_receipt.complete_route_axis_scanned is not True
            or scan_receipt.one_live_tile_enforced is not True
            or scan_receipt.no_route_scale_d2h is not True
            or scan_receipt.no_direction_route_bank_retained is not True
        ):
            _refuse(
                V1PhaseAGlobalQOutsideFaceRefusalCode.SOURCE_MISMATCH,
                "final support-plane/Mv receipt does not replay the consumed route scan",
            )
        support_plane_scan_receipt_hash = scan_receipt.receipt_hash
        population_execution_hash = scan_receipt.population_mjv_execution_receipt_hash
    if rhs_upper is None or not all(shell_seen) or not route_ranges or route_ranges[0][0] != 0 or route_ranges[-1][1] != support.route_count:
        _refuse(V1PhaseAGlobalQOutsideFaceRefusalCode.ROUTE_COVERAGE_MISMATCH, "outside-face scan did not cover every route and shell coordinate")
    if selected_total_count != removed_count:
        _refuse(V1PhaseAGlobalQOutsideFaceRefusalCode.ACTION_MEMBERSHIP_MISMATCH, "streamed selected membership differs from the source packed action")
    capacities = tuple(item.removal_capacity for item in blocks)
    remaining = tuple(capacity - count for capacity, count in zip(capacities, selected_outside_blocks, strict=True))
    if any(value < 0 for value in remaining):
        _refuse(V1PhaseAGlobalQOutsideFaceRefusalCode.BLOCK_CAPACITY_MISMATCH, "fixed outside incumbent exceeds a source block capacity")
    rhs_lower, rhs_upper_dot = _dot_bounds_nonnegative(support_multipliers, rhs_upper)
    capacity_tensor = torch.tensor(capacities, dtype=torch.float64, device=device)
    cap_lower, cap_upper = _dot_bounds_nonnegative(block_multipliers, capacity_tensor)
    full_lower = _add_down(_add_down(rhs_lower, cap_lower), positive_advantage_lower)
    full_upper = _add_up(_add_up(rhs_upper_dot, cap_upper), positive_advantage_upper)
    full_dual_bounds = torch.stack((full_lower, full_upper)).contiguous()
    outside_resource_bounds = torch.stack((outside_resource_lower, outside_resource_upper)).contiguous()
    margin_tensor = torch.stack((selected_lower_min, selected_upper_min, excluded_lower_max, excluded_upper_max))
    margin_values = tuple(float(value) for value in margin_tensor.detach().cpu().tolist())
    selected_lower_value = None if selected_outside_count == 0 else margin_values[0]
    selected_upper_value = None if selected_outside_count == 0 else margin_values[1]
    excluded_lower_value = None if excluded_outside_count == 0 else margin_values[2]
    excluded_upper_value = None if excluded_outside_count == 0 else margin_values[3]
    misaligned = (
        (selected_upper_value is not None and selected_upper_value < 0.0)
        or (excluded_lower_value is not None and excluded_lower_value > 0.0)
    )
    strict = (
        (selected_lower_value is None or selected_lower_value > 0.0)
        and (excluded_upper_value is None or excluded_upper_value < 0.0)
    )
    status = (
        V1PhaseAGlobalQOutsideSignStatus.MISALIGNED
        if misaligned
        else V1PhaseAGlobalQOutsideSignStatus.STRICT_INCUMBENT_ALIGNED
        if strict
        else V1PhaseAGlobalQOutsideSignStatus.AMBIGUOUS
    )
    values = (
        shell_lower.contiguous(),
        shell_upper.contiguous(),
        outside_support_lower.contiguous(),
        outside_support_upper.contiguous(),
        rhs_upper.contiguous(),
        shell_resource.contiguous(),
        support_multipliers.contiguous(),
        block_multipliers.contiguous(),
        outside_resource_bounds,
        full_dual_bounds,
    )
    digests = tuple(_compact_tensor_digest(value, role=name) for name, value in zip(_COMPACT_NAMES, values, strict=True))
    compact_scalars = sum(int(value.numel()) for value in values) + 4
    h2d_per_route = _F64_BYTES + 3 + np.dtype(np.int64).itemsize
    receipt = V1PhaseAGlobalQOutsideFaceReceipt(
        design_hash=design.design_hash,
        forecast_hash=forecast.forecast_hash,
        joint_preoutcome_registration_hash=prereg_hash,
        support_family_receipt_hash=support_hash,
        population_mjv_forecast_hash=population_mjv_forecast_hash,
        population_mjv_execution_receipt_hash=population_execution_hash,
        support_plane_scan_receipt_hash=support_plane_scan_receipt_hash,
        dual_proposal_receipt_hash=proposal_hash,
        relaxed_kkt_problem_hash=relaxed_kkt_problem.problem_hash,
        action_pair_alias_source_join_receipt_hash=source_join_hash,
        robust_shell_decision_receipt_hash=robust_hash,
        action_pair_hash=pair_hash,
        parent_mask_hash=parent_mask.mask_hash,
        selected_physical_action_identity_hash=selected.physical_action_identity_hash,
        selected_packed_action_hash=selected.packed_action_hash,
        selected_resulting_mask_hash=child.mask_hash,
        route_axis_hash=support.route_axis_hash,
        route_id_axis_hash=support.route_id_axis_hash,
        joint_direction_axis_hash=support.joint_direction_axis_hash,
        shell_axis_hash=proposal.shell_axis_hash,
        block_axis_hash=proposal.block_axis_hash,
        route_count=support.route_count,
        scanned_route_count=support.route_count,
        scanned_tile_count=len(route_ranges),
        direction_count=directions,
        shell_size=len(shell_indices),
        block_count=len(blocks),
        selected_total_count=selected_total_count,
        selected_outside_count=selected_outside_count,
        selected_shell_count=selected_shell_count,
        excluded_outside_eligible_count=excluded_outside_count,
        selected_outside_block_counts=tuple(selected_outside_blocks),
        block_capacities=capacities,
        remaining_shell_block_capacities=remaining,
        support_coefficient_lower_digest=digests[0],
        support_coefficient_upper_digest=digests[1],
        outside_support_lower_sum_digest=digests[2],
        outside_support_upper_sum_digest=digests[3],
        support_rhs_upper_digest=digests[4],
        shell_resource_digest=digests[5],
        support_multiplier_digest=digests[6],
        block_multiplier_digest=digests[7],
        outside_resource_bounds_digest=digests[8],
        full_dual_bounds_digest=digests[9],
        full_dual_lower=float(full_dual_bounds[0].item()),
        full_dual_upper=float(full_dual_bounds[1].item()),
        outside_resource_lower=float(outside_resource_bounds[0].item()),
        outside_resource_upper=float(outside_resource_bounds[1].item()),
        selected_reduced_advantage_lower_min=selected_lower_value,
        selected_reduced_advantage_upper_min=selected_upper_value,
        excluded_reduced_advantage_lower_max=excluded_lower_value,
        excluded_reduced_advantage_upper_max=excluded_upper_value,
        outside_sign_status=status,
        support_plane_tile_receipt_hashes=tuple(tile_hashes),
        route_ranges=tuple(route_ranges),
        host_to_device_payload_bytes=(
            (2 * support.route_count if compact_source_mode else h2d_per_route * support.route_count)
            if production
            else 0
        ),
        compact_device_to_host_scalar_count=compact_scalars if production else 0,
        compact_device_to_host_payload_bytes=compact_scalars * _F64_BYTES if production else 0,
        projected_incremental_peak_bytes=forecast.projected_incremental_peak_bytes,
        measurement_device_identity=proposal.measurement_device_identity,
        production_cuda_execution=production,
        private_cpu_reference_execution=not production,
        _issuance_token=_RECEIPT_TOKEN,
    )
    handle = V1PhaseAGlobalQOutsideFaceHandle(receipt=receipt, values=values, _issuance_token=_HANDLE_TOKEN)
    return V1PhaseAGlobalQOutsideFaceResult(receipt=receipt, compact_handle=handle)


def execute_v1_phase_a_global_q_outside_face_cuda(
    *,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    support_family: (
        V1PhaseAGlobalQSupportFamilyResult
        | V1PhaseAGlobalQD1SupportFamilyResult
    ),
    population_mjv_stream: (
        V1LiteralJJointPopulationMvStreamCUDAHandle
        | V1LiteralJJointD1PopulationMvStreamCUDAHandle
    ),
    support_plane_scan: V1PhaseAGlobalQSupportPlaneScanHandle,
    dual_proposal: V1PhaseAGlobalQOuterLPDualProposalResult,
    action_pair_alias_source_join: V1SampledV7ActionPairAliasSourceJoinReceipt,
    robust_shell_decision: V1SampledV7PhaseARobustShellDecisionReceipt,
    parent_mask: HardMask,
    relaxed_kkt_problem: V1RelaxedKKTProblem,
    design: V1PhaseAGlobalQOutsideFaceDesign = V1PhaseAGlobalQOutsideFaceDesign(),
) -> V1PhaseAGlobalQOutsideFaceResult:
    return _execute(
        joint_preoutcome_registration=joint_preoutcome_registration,
        support_family=support_family,
        population_mjv_stream=population_mjv_stream,
        support_plane_scan=support_plane_scan,
        compact_dual_source=None,
        dual_proposal=dual_proposal,
        action_pair_alias_source_join=action_pair_alias_source_join,
        robust_shell_decision=robust_shell_decision,
        parent_mask=parent_mask,
        relaxed_kkt_problem=relaxed_kkt_problem,
        design=design,
        private_cpu_reference=False,
    )


def execute_v1_phase_a_global_q_outside_face_from_compact_source_cuda(
    *,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    support_family: (
        V1PhaseAGlobalQSupportFamilyResult
        | V1PhaseAGlobalQD1SupportFamilyResult
    ),
    compact_dual_source: V1PhaseAGlobalQCompactOuterLPCUDASource,
    dual_proposal: V1PhaseAGlobalQOuterLPDualProposalResult,
    action_pair_alias_source_join: V1SampledV7ActionPairAliasSourceJoinReceipt,
    robust_shell_decision: V1SampledV7PhaseARobustShellDecisionReceipt,
    parent_mask: HardMask,
    relaxed_kkt_problem: V1RelaxedKKTProblem,
    design: V1PhaseAGlobalQOutsideFaceDesign = V1PhaseAGlobalQOutsideFaceDesign(),
) -> V1PhaseAGlobalQOutsideFaceResult:
    return _execute(
        joint_preoutcome_registration=joint_preoutcome_registration,
        support_family=support_family,
        population_mjv_stream=None,
        support_plane_scan=None,
        compact_dual_source=compact_dual_source,
        dual_proposal=dual_proposal,
        action_pair_alias_source_join=action_pair_alias_source_join,
        robust_shell_decision=robust_shell_decision,
        parent_mask=parent_mask,
        relaxed_kkt_problem=relaxed_kkt_problem,
        design=design,
        private_cpu_reference=False,
    )


def _execute_v1_phase_a_global_q_outside_face_cpu_reference(
    **kwargs: Any,
) -> V1PhaseAGlobalQOutsideFaceResult:
    kwargs.setdefault("compact_dual_source", None)
    return _execute(**kwargs, private_cpu_reference=True)


__all__ = (
    "PHASE_A_GLOBAL_Q_OUTSIDE_FACE_DESIGN_SCHEMA",
    "PHASE_A_GLOBAL_Q_OUTSIDE_FACE_FORECAST_SCHEMA",
    "PHASE_A_GLOBAL_Q_OUTSIDE_FACE_SCHEMA",
    "PHASE_A_GLOBAL_Q_OUTSIDE_FACE_ALGORITHM",
    "PHASE_A_GLOBAL_Q_OUTSIDE_FACE_AUTHORITY",
    "V1PhaseAGlobalQOutsideFaceRefusalCode",
    "V1PhaseAGlobalQOutsideFaceRefusal",
    "V1PhaseAGlobalQOutsideSignStatus",
    "V1PhaseAGlobalQOutsideFaceDesign",
    "V1PhaseAGlobalQOutsideFaceHBMForecast",
    "V1PhaseAGlobalQOutsideFaceReceipt",
    "V1PhaseAGlobalQOutsideFaceLease",
    "V1PhaseAGlobalQOutsideFaceHandle",
    "V1PhaseAGlobalQOutsideFaceResult",
    "execute_v1_phase_a_global_q_outside_face_cuda",
    "execute_v1_phase_a_global_q_outside_face_from_compact_source_cuda",
)
