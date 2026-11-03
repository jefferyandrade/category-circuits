"""Fixed-primal normal-cone and current-tensor rounding diagnostic.

This module is the deliberately narrow successor to the source-issued
fixed-primal projection joint.  It does **not** solve a sensitivity problem.
For one already registered resource coordinate, convex Q branch, relaxed
primal ``z``, and one frozen hard endpoint it:

* intersects the simultaneous literal-J gradient intervals with the
  deterministic FD residual envelope;
* combines the resulting Q interval and the signed-F subgradient interval
  with the solver-issued resource/Q/block dual custody;
* scans the complete route axis on device for normal-cone, zero-cell, and
  rounding-boundary ambiguity;
* transfers at most one pre-registered, bounded shell to a tiny exact
  enumerator;
* replays complete hard signed-F and both Q envelopes for every byte-distinct
  candidate, without a Q-monotonic early stop; and
* reports the physical hard actions reachable by this fixed-z interval
  construction.

This is not an action-invariance theorem.  Normal information at one fixed z
does not rule out another relaxed optimum or epsilon-near optimum moving the
bounded shell coordinates and rounding differently.  The diagnostic grants
no normal-cone/action-stability, unique-optimizer, full-sensitivity, global
hard-mask, path, C/K/O, commit, or Pareto authority.  The former issuer name
remains as a fail-closed seam and always refuses until strict outside-coordinate
screening plus a robust variable-z bounded-shell solve is implemented.

Production is CUDA-only.  Route-scale values, keys, and indices never leave
the device.  A private, route-capped CPU path exists solely for high-precision
tests and uses the same source/lease contracts.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
import itertools
import math
from numbers import Integral, Real
import os
from typing import Any, Final, NoReturn

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes
from mechanistic_systems.contracts.hard_mask_actions import (
    HardMask,
    RemovedRouteIndexSet,
)

from .modern_v1_device_shell_ladder import V1DeviceShellBorrowOperation
from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch
from .modern_v1_signed_work_device_shell_cuda import (
    V1SignedWorkDeviceShellExecutionBundle,
)
from .modern_v1_signed_work_fixed_primal_projection_cuda import (
    V1SignedWorkFixedPrimalProjectionCUDAExecution,
)
from .modern_v1_signed_work_hard_fq_replay_cuda import (
    _divide_nonnegative_interval,
    _down_add,
    _down_multiply_nonnegative,
    _gamma_n,
    _nonnegative_interval_from_nominal,
    _signed_sum_interval,
    _square_interval,
    _sum_nonnegative_intervals,
    _tiled_lane_summary,
    _up_add,
    _up_multiply,
)
from .modern_v1_signed_work_kkt_cuda import (
    V1SignedWorkLeastWorkCUDAPolicy,
    V1SignedWorkLeastWorkCUDAExecutionBundle,
    V1SignedWorkLeastWorkCUDAProblem,
    V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle,
    _canonical_fused_scales,
)

try:  # pragma: no cover - availability depends on the execution host.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


SIGNED_WORK_NORMAL_CONE_ACTION_DESIGN_SCHEMA: Final = (
    "partition_free_v1_signed_work_normal_cone_action_design_v1"
)
SIGNED_WORK_NORMAL_CONE_ACTION_HBM_SCHEMA: Final = (
    "partition_free_v1_signed_work_normal_cone_action_hbm_v1"
)
SIGNED_WORK_NORMAL_CONE_OUTSIDE_SCAN_SCHEMA: Final = (
    "partition_free_v1_signed_work_fixed_tensor_outside_scan_v3"
)
SIGNED_WORK_NORMAL_CONE_SHELL_SCHEMA: Final = (
    "partition_free_v1_signed_work_normal_cone_shell_enumeration_v1"
)
SIGNED_WORK_NORMAL_CONE_HARD_REPLAY_SCHEMA: Final = (
    "partition_free_v1_signed_work_normal_cone_hard_replay_v1"
)
SIGNED_WORK_NORMAL_CONE_ACTION_CERTIFICATE_SCHEMA: Final = (
    "partition_free_v1_signed_work_fixed_tensor_rounding_diagnostic_v2"
)
SIGNED_WORK_NORMAL_CONE_ACTION_RESULT_SCHEMA: Final = (
    "partition_free_v1_signed_work_fixed_tensor_rounding_result_v3"
)
SIGNED_WORK_FIXED_PRIMAL_BOUNDED_SHELL_HANDOFF_SCHEMA: Final = (
    "partition_free_v1_signed_work_fixed_primal_bounded_shell_handoff_v1"
)
SIGNED_WORK_FIXED_PRIMAL_BOUNDED_SHELL_HANDOFF_AUTHORITY: Final = (
    "provisional_variable_shell_initializer_only_fixed_primal_diagnostic_"
    "does_not_resolve_alternative_solution_sampling_FD_rank_numerical_"
    "action_invariance_C_K_O_commit_terminal_or_pareto_authority_v1"
)
SIGNED_WORK_NORMAL_CONE_ACTION_ALGORITHM: Final = (
    "fixed_primal_source_projection_intersect_literal_J_FD_normal_cone_"
    "bounded_shell_complete_face_enumeration_and_full_hard_FQ_replay_v1"
)
SIGNED_WORK_NORMAL_CONE_ACTION_AUTHORITY: Final = (
    "fixed_tensor_rounding_diagnostic_only_alternative_solution_sensitivity_"
    "unresolved_no_normal_cone_action_stability_C_K_O_commit_terminal_or_"
    "pareto_authority_v2"
)
SIGNED_WORK_NORMAL_CONE_FACE_SCOPE: Final = (
    "fixed_primal_independent_outward_interval_cartesian_face_superset_"
    "zero_cell_ambiguity_widens_source_F_subgradient_and_enters_shell_only_"
    "through_normal_or_rounding_boundary_may_pair_nonjoint_faces_and_"
    "overrefuse_but_never_understates_the_union_of_admissible_fixed_"
    "rounding_physical_actions_v2"
)
SIGNED_WORK_NORMAL_CONE_EPSILON_SCOPE: Final = (
    "registered_FD_convex_branch_actual_SOC_dual_adjoint_fixed_tensor_gap_"
    "diagnostic_only_not_an_alternative_solution_or_action_certificate_v2"
)
SIGNED_WORK_NORMAL_CONE_BYTE_DEDUP_SCOPE: Final = (
    "same_parent_same_outside_membership_bounded_shell_subset_is_exact_"
    "physical_mask_byte_identity_v1"
)

_PREFIX: Final = "partition-free-v1-signed-work-normal-cone-action-"
_F64_BYTES: Final = 8
_I64_BYTES: Final = 8
_BOOL_BYTES: Final = 1
_U8_BYTES: Final = 1
_MINIMUM_ALLOCATOR_GUARD_BYTES: Final = 64 * 1024 * 1024
_ALLOCATOR_GUARD_FRACTION: Final = 0.10
_DUAL_NONNEGATIVE_ROUNDOFF_ULPS: Final = 256
_ISSUER_TOKEN: Final = object()


class V1SignedWorkNormalConeActionInvarianceCUDARefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_IDENTITY = "stale_identity"
    SOURCE_CUSTODY_UNAVAILABLE = "source_custody_unavailable"
    CUDA_REQUIRED = "cuda_required"
    DETERMINISM_REQUIRED = "determinism_required"
    CPU_REFERENCE_SCALE_EXCEEDED = "cpu_reference_scale_exceeded"
    MULTI_ENDPOINT_UNSUPPORTED = "multi_endpoint_unsupported"
    FIXED_PRIMAL_MISMATCH = "fixed_primal_mismatch"
    SOURCE_FD_DISJOINT = "source_fd_disjoint"
    FIXED_PRIMAL_INFEASIBLE = "fixed_primal_infeasible"
    STALE_ROUNDING = "stale_rounding"
    SHELL_OVERSIZED = "shell_oversized"
    FACE_ENUMERATION_EXCEEDED = "face_enumeration_exceeded"
    NO_ADMISSIBLE_FACE = "no_admissible_face"
    ALTERNATIVE_SOLUTION_SENSITIVITY_UNRESOLVED = (
        "alternative_solution_sensitivity_unresolved"
    )
    HARD_Q_UNRESOLVED = "hard_q_unresolved"
    HBM_FORECAST_EXCEEDED = "hbm_forecast_exceeded"
    NUMERICAL_REFUSAL = "numerical_refusal"
    MUTATED_CUSTODY = "mutated_custody"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SignedWorkNormalConeActionInvarianceCUDARefusal(RuntimeError):
    """Typed, fail-closed refusal from the fixed-tensor diagnostic joint."""

    def __init__(
        self,
        code: V1SignedWorkNormalConeActionInvarianceCUDARefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SignedWorkNormalConeActionInvarianceCUDARefusalCode:
            raise TypeError("normal-cone refusal code must have exact enum type")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SignedWorkNormalConeActionInvarianceCUDARefusalCode,
    message: str,
) -> NoReturn:
    raise V1SignedWorkNormalConeActionInvarianceCUDARefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one positive integer",
        )
    result = int(value)
    if result <= 0:
        _refuse(
            V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.INVALID_TYPE,
            f"{name} must be nonnegative",
        )
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative finite real",
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        _refuse(
            V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.NUMERICAL_REFUSAL,
            f"{name} must be nonnegative and finite",
        )
    return result


def _next_down(value: float) -> float:
    result = math.nextafter(float(value), -math.inf)
    if not math.isfinite(result):
        _refuse(
            V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.NUMERICAL_REFUSAL,
            "outward lower endpoint left finite binary64",
        )
    return result


def _next_up(value: float) -> float:
    result = math.nextafter(float(value), math.inf)
    if not math.isfinite(result):
        _refuse(
            V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.NUMERICAL_REFUSAL,
            "outward upper endpoint left finite binary64",
        )
    return result


def _tensor_next_down(value: Any) -> Any:
    return torch.nextafter(value, torch.full_like(value, -math.inf))


def _tensor_next_up(value: Any) -> Any:
    return torch.nextafter(value, torch.full_like(value, math.inf))


@dataclass(frozen=True, slots=True)
class V1SignedWorkNormalConeActionInvarianceCUDADesign:
    route_tile_size: int = 131_072
    hard_replay_route_tile_size: int = 131_072
    maximum_shell_routes: int = 8
    maximum_shell_faces: int = 65_536
    private_reference_route_limit: int = 4096
    hbm_ceiling_fraction: float = 0.85
    production_cuda_required: bool = True
    allow_private_cpu_reference: bool = False
    require_deterministic_algorithms: bool = True
    require_cublas_deterministic_workspace: bool = True
    forbid_route_scale_device_to_host: bool = True
    require_one_frozen_endpoint: bool = True
    algorithm_id: str = SIGNED_WORK_NORMAL_CONE_ACTION_ALGORITHM
    schema: str = SIGNED_WORK_NORMAL_CONE_ACTION_DESIGN_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "route_tile_size",
            "hard_replay_route_tile_size",
            "maximum_shell_routes",
            "maximum_shell_faces",
            "private_reference_route_limit",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name=name))
        ceiling = _finite_nonnegative(self.hbm_ceiling_fraction, name="HBM ceiling fraction")
        if ceiling != 0.85:
            raise ValueError("canonical fixed-tensor diagnostic HBM ceiling is frozen at 0.85")
        object.__setattr__(self, "hbm_ceiling_fraction", ceiling)
        flags = (
            self.production_cuda_required,
            self.allow_private_cpu_reference,
            self.require_deterministic_algorithms,
            self.require_cublas_deterministic_workspace,
            self.forbid_route_scale_device_to_host,
            self.require_one_frozen_endpoint,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("fixed-tensor diagnostic design flags must be exact booleans")
        if self.production_cuda_required and self.allow_private_cpu_reference:
            raise ValueError("production fixed-tensor diagnostics cannot enable CPU fallback")
        if not self.forbid_route_scale_device_to_host or not self.require_one_frozen_endpoint:
            raise PermissionError("route-scale D2H and multi-endpoint diagnostics are forbidden")
        if self.algorithm_id != SIGNED_WORK_NORMAL_CONE_ACTION_ALGORITHM or self.schema != SIGNED_WORK_NORMAL_CONE_ACTION_DESIGN_SCHEMA:
            raise ValueError("unsupported fixed-tensor diagnostic design semantics")

    @property
    def design_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "design-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkNormalConeActionInvarianceCUDAHBMForecast:
    problem_hash: str
    policy_hash: str
    least_work_execution_hash: str
    projection_receipt_hash: str
    shell_execution_hash: str
    design_hash: str
    device: str
    route_count: int
    response_rank: int
    route_tile_size: int
    maximum_shell_routes: int
    borrowed_route_vectors_bytes: int
    parent_live_bytes: int
    candidate_label_clone_bytes: int
    q_dual_adjoint_workspace_bytes_upper: int
    scan_tile_bytes_upper: int
    hard_replay_workspace_bytes_upper: int
    compact_shell_bytes_upper: int
    allocator_guard_bytes: int
    forecast_incremental_bytes: int
    current_reserved_bytes: int
    forecast_peak_reserved_bytes: int
    device_total_memory_bytes: int
    hbm_ceiling_bytes: int
    admitted: bool
    route_scale_device_to_host_bytes: int = 0
    schema: str = SIGNED_WORK_NORMAL_CONE_ACTION_HBM_SCHEMA
    _admission_memory: InitVar[tuple[int, int] | None] = None

    def __post_init__(self, _admission_memory: tuple[int, int] | None) -> None:
        available, allocated = _admission_memory or (self.hbm_ceiling_bytes, self.current_reserved_bytes)
        for name in (
            "problem_hash", "policy_hash", "least_work_execution_hash",
            "projection_receipt_hash", "shell_execution_hash", "design_hash", "device",
        ):
            _identity(getattr(self, name), name=name)
        for name in (
            "route_count", "route_tile_size", "maximum_shell_routes",
            "device_total_memory_bytes", "hbm_ceiling_bytes",
        ):
            _positive_int(getattr(self, name), name=name)
        for name in (
            "response_rank", "borrowed_route_vectors_bytes", "parent_live_bytes",
            "candidate_label_clone_bytes", "q_dual_adjoint_workspace_bytes_upper",
            "scan_tile_bytes_upper",
            "hard_replay_workspace_bytes_upper", "compact_shell_bytes_upper",
            "allocator_guard_bytes", "forecast_incremental_bytes",
            "current_reserved_bytes", "forecast_peak_reserved_bytes",
            "route_scale_device_to_host_bytes",
        ):
            _nonnegative_int(getattr(self, name), name=name)
        if self.route_scale_device_to_host_bytes != 0:
            raise PermissionError("fixed-tensor diagnostic forecast cannot admit route-scale D2H")
        if self.forecast_peak_reserved_bytes != self.current_reserved_bytes + self.forecast_incremental_bytes:
            raise ValueError("fixed-tensor diagnostic HBM peak does not replay")
        if self.admitted is not (
            allocated + self.forecast_incremental_bytes <= self.hbm_ceiling_bytes
            and self.forecast_incremental_bytes <= available
        ):
            raise ValueError("fixed-tensor diagnostic HBM admission flag is inconsistent")
        if self.schema != SIGNED_WORK_NORMAL_CONE_ACTION_HBM_SCHEMA:
            raise ValueError("unsupported fixed-tensor diagnostic HBM schema")

    @property
    def forecast_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "hbm-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkNormalConeOutsideShellScanReceipt:
    problem_hash: str
    projection_receipt_hash: str
    shell_execution_hash: str
    fixed_primal_source_receipt_hash: str
    branch: V1RelaxedKKTBranch
    route_count: int
    scanned_route_count: int
    shell_route_count: int
    normal_boundary_route_count: int
    rounding_boundary_route_count: int
    zero_cell_ambiguity_route_count: int
    shell_index_hash: str
    outside_membership_fingerprint: str
    fixed_primal_q_lower: float
    fixed_primal_q_upper: float
    fd_q_lower: float
    fd_q_upper: float
    q_gradient_intersection_nonempty_every_route: bool
    resource_lower: float
    resource_upper: float
    q_dual_adjoint_fingerprint: str
    q_dual_adjoint_maximum_numerical_radius: float
    soc_dual_norm_upper: float
    soc_response_norm_lower: float
    soc_response_norm_upper: float
    soc_nonradial_residual_norm_upper: float
    soc_response_near_zero: bool
    resource_multiplier: float
    residual_linear_center_fingerprint: str
    residual_linear_center_max_abs: float
    block_multiplier_hash: str
    normal_gap_upper: float
    complementarity_gap_upper: float
    solver_stationarity_gap: float
    solver_absolute_duality_gap: float
    fixed_branch_epsilon_upper: float
    scalar_device_to_host_transfers: int
    bounded_shell_device_to_host_values: int
    route_scale_device_to_host_transfers: int = 0
    complete_tiled_scan: bool = True
    fd_uncertainty_is_deterministic_no_alpha_spend: bool = True
    normal_uses_source_f_subgradient_not_solver_work_dual: bool = True
    normal_includes_solver_residual_linear_center: bool = True
    normal_uses_actual_registered_soc_dual_adjoint: bool = True
    literal_j_q_gradient_used_for_normal_cone: bool = False
    zero_cell_ambiguity_enters_shell_only_through_normal_or_rounding: bool = True
    alternative_solution_sensitivity_resolved: bool = False
    schema: str = SIGNED_WORK_NORMAL_CONE_OUTSIDE_SCAN_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "problem_hash", "projection_receipt_hash", "shell_execution_hash",
            "fixed_primal_source_receipt_hash", "shell_index_hash",
            "outside_membership_fingerprint", "block_multiplier_hash",
            "q_dual_adjoint_fingerprint", "residual_linear_center_fingerprint",
        ):
            _identity(getattr(self, name), name=name)
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("outside-shell receipt requires an exact Q branch")
        for name in (
            "route_count", "scanned_route_count", "shell_route_count",
            "normal_boundary_route_count", "rounding_boundary_route_count",
            "zero_cell_ambiguity_route_count", "scalar_device_to_host_transfers",
            "bounded_shell_device_to_host_values", "route_scale_device_to_host_transfers",
        ):
            _nonnegative_int(getattr(self, name), name=name)
        if self.route_count <= 0 or self.scanned_route_count != self.route_count:
            raise ValueError("outside-shell scan did not cover the complete route axis")
        for name in (
            "fixed_primal_q_lower", "fixed_primal_q_upper", "fd_q_lower", "fd_q_upper",
            "resource_lower", "resource_upper",
            "q_dual_adjoint_maximum_numerical_radius", "soc_dual_norm_upper",
            "soc_response_norm_lower", "soc_response_norm_upper",
            "soc_nonradial_residual_norm_upper", "resource_multiplier",
            "residual_linear_center_max_abs",
            "normal_gap_upper", "complementarity_gap_upper", "solver_stationarity_gap",
            "solver_absolute_duality_gap", "fixed_branch_epsilon_upper",
        ):
            _finite_nonnegative(getattr(self, name), name=name)
        if self.fixed_primal_q_lower > self.fixed_primal_q_upper or self.fd_q_lower > self.fd_q_upper or self.resource_lower > self.resource_upper:
            raise ValueError("outside-shell scalar interval is reversed")
        if self.soc_response_norm_lower > self.soc_response_norm_upper:
            raise ValueError("SOC response-norm interval is reversed")
        if (
            self.q_gradient_intersection_nonempty_every_route is not True
            or self.complete_tiled_scan is not True
            or self.fd_uncertainty_is_deterministic_no_alpha_spend is not True
            or self.normal_uses_source_f_subgradient_not_solver_work_dual is not True
            or self.normal_includes_solver_residual_linear_center is not True
            or self.route_scale_device_to_host_transfers != 0
            or self.normal_uses_actual_registered_soc_dual_adjoint is not True
            or self.literal_j_q_gradient_used_for_normal_cone is not False
            or self.zero_cell_ambiguity_enters_shell_only_through_normal_or_rounding is not True
            or self.alternative_solution_sensitivity_resolved is not False
        ):
            raise PermissionError("outside-shell receipt overstates its fixed-tensor diagnostic scope")
        if type(self.soc_response_near_zero) is not bool:
            raise TypeError("SOC near-zero flag must be exact bool")
        if self.schema != SIGNED_WORK_NORMAL_CONE_OUTSIDE_SCAN_SCHEMA:
            raise ValueError("unsupported outside-shell scan schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "outside-scan-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkNormalConeShellEnumerationReceipt:
    outside_scan_receipt_hash: str
    bounded_shell_route_count: int
    selectable_rounding_shell_route_count: int
    normal_face_count: int
    enumerated_face_superset_count: int
    byte_distinct_physical_action_count: int
    shell_face_hash: str
    action_subset_hash: str
    baseline_subset_present: bool
    block_capacity_enforced: bool
    eligibility_enforced: bool
    every_interval_boundary_routed_to_shell: bool
    face_scope: str = SIGNED_WORK_NORMAL_CONE_FACE_SCOPE
    schema: str = SIGNED_WORK_NORMAL_CONE_SHELL_SCHEMA

    def __post_init__(self) -> None:
        for name in ("outside_scan_receipt_hash", "shell_face_hash", "action_subset_hash"):
            _identity(getattr(self, name), name=name)
        for name in (
            "bounded_shell_route_count", "selectable_rounding_shell_route_count",
            "normal_face_count", "enumerated_face_superset_count",
            "byte_distinct_physical_action_count",
        ):
            _nonnegative_int(getattr(self, name), name=name)
        if self.enumerated_face_superset_count <= 0 or self.byte_distinct_physical_action_count <= 0:
            raise ValueError("shell enumeration requires at least one face and action")
        flags = (
            self.baseline_subset_present, self.block_capacity_enforced,
            self.eligibility_enforced, self.every_interval_boundary_routed_to_shell,
        )
        if any(value is not True for value in flags):
            raise PermissionError("shell enumeration lost a completeness condition")
        if self.face_scope != SIGNED_WORK_NORMAL_CONE_FACE_SCOPE or self.schema != SIGNED_WORK_NORMAL_CONE_SHELL_SCHEMA:
            raise ValueError("unsupported shell enumeration semantics")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "shell-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkNormalConeHardCandidateReplayReceipt:
    candidate_ordinal: int
    physical_action_equivalence_hash: str
    removed_count: int
    resource_lower: float
    resource_upper: float
    signed_f_lower: float
    signed_f_upper: float
    spectral_q_lower: float
    spectral_q_upper: float
    diagonal_q_lower: float
    diagonal_q_upper: float
    canonical_q_lower: float
    canonical_q_upper: float
    quadratic_budget: float
    canonical_q_robustly_feasible: bool
    complete_f_replay: bool = True
    complete_two_branch_q_replay: bool = True
    q_monotonic_early_stop_used: bool = False
    schema: str = SIGNED_WORK_NORMAL_CONE_HARD_REPLAY_SCHEMA

    def __post_init__(self) -> None:
        _nonnegative_int(self.candidate_ordinal, name="candidate ordinal")
        _identity(self.physical_action_equivalence_hash, name="physical action equivalence hash")
        _positive_int(self.removed_count, name="removed count")
        for name in (
            "resource_lower", "resource_upper", "signed_f_lower", "signed_f_upper",
            "spectral_q_lower", "spectral_q_upper", "diagonal_q_lower",
            "diagonal_q_upper", "canonical_q_lower", "canonical_q_upper",
            "quadratic_budget",
        ):
            _finite_nonnegative(getattr(self, name), name=name)
        for lower, upper in (
            (self.resource_lower, self.resource_upper),
            (self.signed_f_lower, self.signed_f_upper),
            (self.spectral_q_lower, self.spectral_q_upper),
            (self.diagonal_q_lower, self.diagonal_q_upper),
            (self.canonical_q_lower, self.canonical_q_upper),
        ):
            if lower > upper:
                raise ValueError("hard replay interval is reversed")
        if self.canonical_q_robustly_feasible is not (self.canonical_q_upper <= self.quadratic_budget):
            raise ValueError("hard replay Q-feasibility flag is inconsistent")
        if not self.complete_f_replay or not self.complete_two_branch_q_replay or self.q_monotonic_early_stop_used:
            raise PermissionError("hard replay must cover F and both Q branches completely")
        if self.schema != SIGNED_WORK_NORMAL_CONE_HARD_REPLAY_SCHEMA:
            raise ValueError("unsupported hard candidate replay schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "hard-replay-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkFixedPrimalRoundingDiagnosticReceipt:
    problem_hash: str
    policy_hash: str
    least_work_execution_hash: str
    least_work_state_identity_hash: str
    branch_receipt_hash: str
    projection_receipt_hash: str
    projection_live_identity_hash: str
    shell_execution_hash: str
    frozen_ladder_receipt_hash: str
    outside_scan_receipt_hash: str
    shell_enumeration_receipt_hash: str
    hard_replay_receipt_hashes: tuple[str, ...]
    chart_identity_hash: str
    current_mask_hash: str
    route_axis_hash: str
    native_fiber_axis_receipt_hash: str
    fixed_primal_descriptor_hash: str
    fixed_primal_source_receipt_hash: str
    registered_resource_coordinate: float
    branch: V1RelaxedKKTBranch
    fixed_tensor_gap_upper: float
    observed_frozen_packed_action_hash: str
    observed_frozen_resulting_mask_hash: str
    observed_frozen_removed_route_index_hash: str
    observed_frozen_removed_count: int
    source_familywise_alpha: float | None
    source_alpha_ledger_id: str
    source_alpha_coordinate_id: str
    source_look_ordinal: int
    source_observed_counts: tuple[int, int]
    source_population_counts: tuple[int, int]
    source_exact_fitting_census: bool = False
    source_sampling_alpha_spent: bool = True
    source_full_population_decision_stability_authority: bool = False
    fixed_tensor_rounding_diagnostic: bool = True
    fixed_registered_coordinate_only: bool = True
    alternative_solution_sensitivity_resolved: bool = False
    normal_cone_authority: bool = False
    action_stability_authority: bool = False
    epsilon_optimality_claimed: bool = False
    byte_physical_action_invariance_claimed: bool = False
    unique_relaxed_optimizer_claimed: bool = False
    full_solution_sensitivity_claimed: bool = False
    hard_global_optimality_claimed: bool = False
    path_optimality_claimed: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    epsilon_scope: str = SIGNED_WORK_NORMAL_CONE_EPSILON_SCOPE
    byte_dedup_scope: str = SIGNED_WORK_NORMAL_CONE_BYTE_DEDUP_SCOPE
    authority_scope: str = SIGNED_WORK_NORMAL_CONE_ACTION_AUTHORITY
    schema: str = SIGNED_WORK_NORMAL_CONE_ACTION_CERTIFICATE_SCHEMA
    _issuance_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuance_token is not _ISSUER_TOKEN:
            _refuse(
                V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.INVALID_TYPE,
                "fixed-tensor diagnostic receipts are source-issued only",
            )
        for name in (
            "problem_hash", "policy_hash", "least_work_execution_hash",
            "least_work_state_identity_hash", "branch_receipt_hash",
            "projection_receipt_hash", "projection_live_identity_hash",
            "shell_execution_hash", "frozen_ladder_receipt_hash",
            "outside_scan_receipt_hash", "shell_enumeration_receipt_hash",
            "chart_identity_hash", "current_mask_hash", "route_axis_hash",
            "native_fiber_axis_receipt_hash", "fixed_primal_descriptor_hash",
            "fixed_primal_source_receipt_hash", "observed_frozen_packed_action_hash",
            "observed_frozen_resulting_mask_hash",
            "observed_frozen_removed_route_index_hash",
            "source_alpha_ledger_id", "source_alpha_coordinate_id",
        ):
            _identity(getattr(self, name), name=name)
        hashes = tuple(_identity(value, name="hard replay receipt hash") for value in self.hard_replay_receipt_hashes)
        if not hashes:
            raise ValueError("diagnostic requires at least one complete hard replay")
        object.__setattr__(self, "hard_replay_receipt_hashes", hashes)
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("diagnostic requires one exact convex Q branch")
        _finite_nonnegative(self.registered_resource_coordinate, name="registered resource coordinate")
        _finite_nonnegative(self.fixed_tensor_gap_upper, name="fixed-tensor gap upper")
        _positive_int(self.observed_frozen_removed_count, name="observed removed count")
        _positive_int(self.source_look_ordinal, name="source look ordinal")
        if len(self.source_observed_counts) != 2 or len(self.source_population_counts) != 2:
            raise ValueError("diagnostic source count axes must have two entries")
        if any(value <= 0 for value in self.source_observed_counts + self.source_population_counts):
            raise ValueError("diagnostic source counts must be positive")
        if (
            type(self.source_exact_fitting_census) is not bool
            or type(self.source_sampling_alpha_spent) is not bool
            or type(self.source_full_population_decision_stability_authority) is not bool
        ):
            raise TypeError("diagnostic source census/alpha flags must be exact booleans")
        if self.source_exact_fitting_census:
            if (
                self.source_familywise_alpha is not None
                or self.source_sampling_alpha_spent is not False
                or self.source_observed_counts != self.source_population_counts
            ):
                _refuse(
                    V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.AUTHORITY_VIOLATION,
                    "an alpha-free source census must exactly cover only its frozen fitting set",
                )
        else:
            _finite_nonnegative(
                self.source_familywise_alpha,
                name="source familywise alpha",
            )
            if self.source_sampling_alpha_spent is not True:
                _refuse(
                    V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.AUTHORITY_VIOLATION,
                    "a sampled diagnostic source must retain its alpha spend",
                )
        if self.source_full_population_decision_stability_authority is not False:
            _refuse(
                V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.AUTHORITY_VIOLATION,
                "fixed-primal diagnostics cannot promote a fitting-set census to full-population stability",
            )
        open_flags = (
            self.fixed_tensor_rounding_diagnostic,
            self.fixed_registered_coordinate_only,
        )
        closed_flags = (
            self.alternative_solution_sensitivity_resolved,
            self.normal_cone_authority,
            self.action_stability_authority,
            self.epsilon_optimality_claimed,
            self.byte_physical_action_invariance_claimed,
            self.unique_relaxed_optimizer_claimed,
            self.full_solution_sensitivity_claimed,
            self.hard_global_optimality_claimed,
            self.path_optimality_claimed,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(value is not True for value in open_flags) or any(value is not False for value in closed_flags):
            _refuse(
                V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.AUTHORITY_VIOLATION,
                "fixed-tensor diagnostic broadened its claim ceiling",
            )
        if (
            self.epsilon_scope != SIGNED_WORK_NORMAL_CONE_EPSILON_SCOPE
            or self.byte_dedup_scope != SIGNED_WORK_NORMAL_CONE_BYTE_DEDUP_SCOPE
            or self.authority_scope != SIGNED_WORK_NORMAL_CONE_ACTION_AUTHORITY
            or self.schema != SIGNED_WORK_NORMAL_CONE_ACTION_CERTIFICATE_SCHEMA
        ):
            raise ValueError("unsupported fixed-tensor diagnostic semantics")

    @property
    def diagnostic_hash(self) -> str:
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
        }
        return content_hash(payload, prefix=_PREFIX + "diagnostic-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkFixedPrimalBoundedShellHandoff:
    """Exact bounded indices from one fixed-primal diagnostic scan.

    The handoff prevents a second route-scale scan merely to rediscover the
    provisional active set.  It is deliberately weaker than a variable-z
    decision witness: downstream code may use these coordinates only to
    initialize a bounded robust solve whose complete outside-coordinate
    separation still owns action-stability authority.
    """

    diagnostic_hash: str
    outside_scan_receipt_hash: str
    problem_hash: str
    projection_receipt_hash: str
    shell_execution_hash: str
    fixed_primal_source_receipt_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    route_axis_hash: str
    native_fiber_axis_receipt_hash: str
    branch: V1RelaxedKKTBranch
    route_count: int
    maximum_shell_routes: int
    shell_indices: tuple[int, ...]
    shell_index_hash: str
    canonical_removed_route_index_hash: str
    complete_route_scan: bool = True
    bounded_device_to_host_only: bool = True
    provisional_initializer_only: bool = True
    alternative_solution_sensitivity_resolved: bool = False
    action_stability_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_FIXED_PRIMAL_BOUNDED_SHELL_HANDOFF_AUTHORITY
    schema: str = SIGNED_WORK_FIXED_PRIMAL_BOUNDED_SHELL_HANDOFF_SCHEMA
    _issuance_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuance_token is not _ISSUER_TOKEN:
            _refuse(
                V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.INVALID_TYPE,
                "fixed-primal bounded-shell handoffs are source-issued only",
            )
        for name in (
            "diagnostic_hash",
            "outside_scan_receipt_hash",
            "problem_hash",
            "projection_receipt_hash",
            "shell_execution_hash",
            "fixed_primal_source_receipt_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "native_fiber_axis_receipt_hash",
            "shell_index_hash",
            "canonical_removed_route_index_hash",
        ):
            _identity(getattr(self, name), name=name)
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("bounded-shell handoff requires one exact convex Q branch")
        route_count = _positive_int(self.route_count, name="route count")
        maximum = _positive_int(
            self.maximum_shell_routes,
            name="maximum shell routes",
        )
        indices = tuple(int(index) for index in self.shell_indices)
        if len(indices) > maximum:
            _refuse(
                V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.SHELL_OVERSIZED,
                "bounded-shell handoff exceeds its pre-registered route cap",
            )
        if any(index < 0 or index >= route_count for index in indices):
            raise ValueError("bounded-shell handoff index is outside the route axis")
        if any(right <= left for left, right in zip(indices, indices[1:])):
            raise ValueError("bounded-shell handoff indices must be strictly increasing")
        object.__setattr__(self, "shell_indices", indices)
        canonical = RemovedRouteIndexSet.from_indices(
            indices,
            route_count=route_count,
        )
        expected_shell_hash = content_hash(
            {
                "route_axis_hash": self.route_axis_hash,
                "bounded_indices": indices,
            },
            prefix=_PREFIX + "bounded-shell-indices-",
        )
        if (
            expected_shell_hash != self.shell_index_hash
            or canonical.index_hash != self.canonical_removed_route_index_hash
        ):
            raise ValueError("bounded-shell handoff index identity does not replay")
        open_flags = (
            self.complete_route_scan,
            self.bounded_device_to_host_only,
            self.provisional_initializer_only,
        )
        closed_flags = (
            self.alternative_solution_sensitivity_resolved,
            self.action_stability_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(flag is not True for flag in open_flags) or any(
            flag is not False for flag in closed_flags
        ):
            _refuse(
                V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.AUTHORITY_VIOLATION,
                "bounded-shell handoff broadened its provisional claim ceiling",
            )
        if (
            self.authority_scope
            != SIGNED_WORK_FIXED_PRIMAL_BOUNDED_SHELL_HANDOFF_AUTHORITY
            or self.schema != SIGNED_WORK_FIXED_PRIMAL_BOUNDED_SHELL_HANDOFF_SCHEMA
        ):
            raise ValueError("unsupported bounded-shell handoff semantics")

    @property
    def handoff_hash(self) -> str:
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
        }
        return content_hash(payload, prefix=_PREFIX + "bounded-shell-handoff-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkFixedPrimalRoundingDiagnosticCUDAResult:
    diagnostic: V1SignedWorkFixedPrimalRoundingDiagnosticReceipt
    forecast: V1SignedWorkNormalConeActionInvarianceCUDAHBMForecast
    outside_scan: V1SignedWorkNormalConeOutsideShellScanReceipt
    shell_enumeration: V1SignedWorkNormalConeShellEnumerationReceipt
    hard_replays: tuple[V1SignedWorkNormalConeHardCandidateReplayReceipt, ...]
    bounded_shell_handoff: V1SignedWorkFixedPrimalBoundedShellHandoff
    schema: str = SIGNED_WORK_NORMAL_CONE_ACTION_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if type(self.diagnostic) is not V1SignedWorkFixedPrimalRoundingDiagnosticReceipt:
            raise TypeError("fixed-tensor result requires exact diagnostic")
        if type(self.forecast) is not V1SignedWorkNormalConeActionInvarianceCUDAHBMForecast:
            raise TypeError("action result requires exact HBM forecast")
        if type(self.outside_scan) is not V1SignedWorkNormalConeOutsideShellScanReceipt:
            raise TypeError("action result requires exact outside scan")
        if type(self.shell_enumeration) is not V1SignedWorkNormalConeShellEnumerationReceipt:
            raise TypeError("action result requires exact shell enumeration")
        if type(self.bounded_shell_handoff) is not V1SignedWorkFixedPrimalBoundedShellHandoff:
            raise TypeError("action result requires an exact bounded-shell handoff")
        replays = tuple(self.hard_replays)
        if not replays or any(type(item) is not V1SignedWorkNormalConeHardCandidateReplayReceipt for item in replays):
            raise TypeError("action result requires exact hard replay receipts")
        object.__setattr__(self, "hard_replays", replays)
        if (
            self.diagnostic.outside_scan_receipt_hash != self.outside_scan.receipt_hash
            or self.diagnostic.shell_enumeration_receipt_hash != self.shell_enumeration.receipt_hash
            or self.diagnostic.hard_replay_receipt_hashes != tuple(item.receipt_hash for item in replays)
            or self.forecast.problem_hash != self.diagnostic.problem_hash
            or self.bounded_shell_handoff.diagnostic_hash
            != self.diagnostic.diagnostic_hash
            or self.bounded_shell_handoff.outside_scan_receipt_hash
            != self.outside_scan.receipt_hash
            or self.bounded_shell_handoff.shell_index_hash
            != self.outside_scan.shell_index_hash
            or len(self.bounded_shell_handoff.shell_indices)
            != self.outside_scan.shell_route_count
            or self.schema != SIGNED_WORK_NORMAL_CONE_ACTION_RESULT_SCHEMA
        ):
            raise ValueError("action result component identities do not close")


def _execution_branch_and_receipt(
    execution: V1SignedWorkLeastWorkCUDAExecutionBundle
    | V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle,
) -> tuple[V1RelaxedKKTBranch, Any, tuple[str, ...]]:
    if type(execution) is V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle:
        return (
            execution.branch,
            execution.branch_receipt,
            execution.result.solver_state_binding_hashes,
        )
    if type(execution) is V1SignedWorkLeastWorkCUDAExecutionBundle:
        branch = execution.selected_branch
        if branch is None:
            _refuse(
                V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.SOURCE_CUSTODY_UNAVAILABLE,
                "two-branch execution has no certified selected branch",
            )
        receipt = next(item for item in execution.branch_receipts if item.branch is branch)
        return branch, receipt, tuple(item.receipt_hash for item in execution.branch_receipts)
    raise TypeError("fixed-tensor diagnostic requires an exact least-work execution bundle")


def _validate_identities(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    execution: V1SignedWorkLeastWorkCUDAExecutionBundle
    | V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle,
    projection: V1SignedWorkFixedPrimalProjectionCUDAExecution,
    shell: V1SignedWorkDeviceShellExecutionBundle,
    parent_mask: HardMask,
    design: V1SignedWorkNormalConeActionInvarianceCUDADesign,
) -> tuple[V1RelaxedKKTBranch, Any, tuple[str, ...]]:
    if type(problem) is not V1SignedWorkLeastWorkCUDAProblem:
        raise TypeError("fixed-tensor diagnostic requires exact signed-work problem")
    if type(policy) is not V1SignedWorkLeastWorkCUDAPolicy:
        raise TypeError("fixed-tensor diagnostic requires exact signed-work policy")
    if type(projection) is not V1SignedWorkFixedPrimalProjectionCUDAExecution:
        raise TypeError("fixed-tensor diagnostic requires exact source projection execution")
    if type(shell) is not V1SignedWorkDeviceShellExecutionBundle:
        raise TypeError("fixed-tensor diagnostic requires exact device-shell execution")
    if type(parent_mask) is not HardMask:
        raise TypeError("fixed-tensor diagnostic requires exact parent HardMask")
    if type(design) is not V1SignedWorkNormalConeActionInvarianceCUDADesign:
        raise TypeError("fixed-tensor diagnostic requires exact design")
    problem.assert_integrity()
    projection.receipt.assert_integrity()
    projection.device_handle.assert_live_identity()
    branch, branch_receipt, branch_hashes = _execution_branch_and_receipt(execution)
    state = execution.state_handle
    shell_receipt = shell.execution_receipt
    source = problem.source_problem
    operator = problem.signed_work_operator
    projection_receipt = projection.receipt
    if state is None or not branch_receipt.converged or not state.signed_work_decision_dual_custody_available:
        _refuse(
            V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.SOURCE_CUSTODY_UNAVAILABLE,
            "fixed-tensor diagnostic requires converged solver state and final dual custody",
        )
    if shell.frozen_ladder.endpoint_count != 1:
        _refuse(
            V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.MULTI_ENDPOINT_UNSUPPORTED,
            "the fixed-tensor diagnostic joint binds exactly one frozen endpoint",
        )
    if (
        execution.result.problem_hash != problem.problem_hash
        or execution.result.policy_hash != policy.policy_hash
        or shell_receipt.problem_hash != problem.problem_hash
        or shell_receipt.policy_hash != policy.policy_hash
        or shell_receipt.least_work_execution_hash != execution.execution_hash
        or shell_receipt.state_identity_hash != execution.state_identity_hash
        or shell_receipt.selected_branch is not branch
        or shell.branch_identity.signed_work_branch_receipt_hash != branch_receipt.receipt_hash
        or shell_receipt.source_geometry_hash != source.source_geometry_hash
        or shell_receipt.source_registration_hash != problem.source_registration_hash
        or shell_receipt.registered_d_coordinate_hash != problem.registered_d_coordinate_hash
        or shell_receipt.parent_mask_hash != parent_mask.mask_hash
        or shell.frozen_ladder.parent_mask_hash != parent_mask.mask_hash
        or parent_mask.route_axis_hash != source.route_axis_hash
        or parent_mask.route_count != source.route_count
        or projection_receipt.route_axis_hash != source.route_axis_hash
        or projection_receipt.route_id_axis_hash != operator.receipt.route_id_axis_hash
        or projection_receipt.native_fiber_axis_receipt_hash != operator.native_fiber_axis_receipt.receipt_hash
        or projection_receipt.route_count != source.route_count
        or projection_receipt.current_mask_hash != parent_mask.mask_hash
        or projection_receipt.fixed_primal_source_receipt_hash != branch_receipt.receipt_hash
        or projection_receipt.fixed_primal_source_receipt_hash not in branch_hashes
    ):
        _refuse(
            V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.STALE_IDENTITY,
            "problem, solver, projection, shell, and parent do not bind one fixed chart/action coordinate",
        )
    forbidden = (
        execution.result.c_stream_authority,
        execution.result.calibration_authority,
        execution.result.bce_oracle_authority,
        execution.result.mask_commit_authority,
        execution.result.terminal_authority,
        execution.result.pareto_claim_authority,
        projection_receipt.normal_cone_authority,
        projection_receipt.action_stability_authority,
        projection_receipt.c_stream_authority,
        projection_receipt.commit_authority,
        shell_receipt.fqr_authority,
        shell_receipt.c_stream_authority,
        shell_receipt.mask_commit_authority,
    )
    if any(forbidden):
        _refuse(
            V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.AUTHORITY_VIOLATION,
            "an input crossed the pre-C fixed-primal authority boundary",
        )
    if design.production_cuda_required:
        if (
            not execution.production_cuda_execution
            or not projection_receipt.production_cuda_execution
            or not operator.receipt.production_cuda_execution
            or shell.label_handle.backend_kind != "torch_cuda"
            or not str(state.device).startswith("cuda")
        ):
            _refuse(
                V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.CUDA_REQUIRED,
                "production fixed-tensor diagnostics require one all-CUDA custody chain",
            )
    elif (
        not design.allow_private_cpu_reference
        or source.route_count > design.private_reference_route_limit
        or not execution.private_cpu_reference_execution
        or not projection_receipt.private_cpu_reference_execution
        or not operator.receipt.private_cpu_reference_execution
        or shell.label_handle.backend_kind != "numpy_reference"
        or not str(state.device).startswith("cpu")
    ):
        _refuse(
            V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.CPU_REFERENCE_SCALE_EXCEEDED,
            "CPU fixed-tensor diagnostics are restricted to one tiny all-reference fixture",
        )
    if (
        design.require_deterministic_algorithms
        and execution.production_cuda_execution
        and (torch is None or not torch.are_deterministic_algorithms_enabled())
    ):
        _refuse(
            V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.DETERMINISM_REQUIRED,
            "production fixed-tensor diagnostics require deterministic torch algorithms",
        )
    if (
        design.require_cublas_deterministic_workspace
        and execution.production_cuda_execution
        and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}
    ):
        _refuse(
            V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.DETERMINISM_REQUIRED,
            "production fixed-tensor diagnostics require deterministic cuBLAS workspace",
        )
    return branch, branch_receipt, branch_hashes


def estimate_v1_signed_work_normal_cone_action_invariance_cuda_hbm(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    execution: V1SignedWorkLeastWorkCUDAExecutionBundle
    | V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle,
    projection: V1SignedWorkFixedPrimalProjectionCUDAExecution,
    shell: V1SignedWorkDeviceShellExecutionBundle,
    parent_mask: HardMask,
    design: V1SignedWorkNormalConeActionInvarianceCUDADesign = V1SignedWorkNormalConeActionInvarianceCUDADesign(),
    current_reserved_bytes: int | None = None,
    device_total_memory_bytes: int | None = None,
) -> V1SignedWorkNormalConeActionInvarianceCUDAHBMForecast:
    """Forecast the bounded scan/replay peak before allocating diagnostic state."""

    _validate_identities(
        problem=problem, policy=policy, execution=execution, projection=projection,
        shell=shell, parent_mask=parent_mask, design=design,
    )
    state = execution.state_handle
    if state is None:  # pragma: no cover - validation closes this path.
        raise RuntimeError("least-work state disappeared after validation")
    source = problem.source_problem
    operator = problem.signed_work_operator
    n = source.route_count
    tile = min(n, design.route_tile_size)
    # Borrowed vectors are charged for visibility even though they are not
    # incrementally allocated here.  The peak increment charges only objects
    # newly owned by this diagnostic.
    borrowed = n * (10 * _F64_BYTES + 4 * _BOOL_BYTES + _U8_BYTES)
    parent_bytes = n * _BOOL_BYTES
    candidate_labels = n * _U8_BYTES
    # The actual SOC-dual adjoint deliberately avoids an unjustified scalar-Q
    # multiplier.  Its source matrix-vector products transiently coexist with
    # low/tail centers and envelopes plus the retained center/radius.  Charge
    # ten FP64 route vectors conservatively before entering the CUDA scan.
    q_dual_workspace = 10 * n * _F64_BYTES
    scan_tile = tile * (14 * _F64_BYTES + 10 * _BOOL_BYTES)
    compact_shell = design.maximum_shell_routes * (7 * _F64_BYTES + 4 * _I64_BYTES + 4 * _BOOL_BYTES)
    hard_replay = max(
        2 * operator.context_count * operator.fiber_count * _F64_BYTES
        + 2 * operator.forecast.temporary_tile_bytes_upper,
        min(n, design.hard_replay_route_tile_size)
        * (5 * _F64_BYTES + 3 * _BOOL_BYTES)
        + source.rank * min(n, design.hard_replay_route_tile_size) * _F64_BYTES,
    )
    working = (
        parent_bytes
        + candidate_labels
        + q_dual_workspace
        + scan_tile
        + compact_shell
        + hard_replay
    )
    guard = max(_MINIMUM_ALLOCATOR_GUARD_BYTES, math.ceil(_ALLOCATOR_GUARD_FRACTION * working))
    incremental = working + guard
    live_admission = (
        execution.production_cuda_execution
        and (current_reserved_bytes is None or device_total_memory_bytes is None)
    )
    if current_reserved_bytes is None or device_total_memory_bytes is None:
        if execution.production_cuda_execution:
            if torch is None or not torch.cuda.is_available():
                _refuse(
                    V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.CUDA_REQUIRED,
                    "CUDA HBM forecast requested without CUDA runtime",
                )
            device = torch.device(state.device)
            if current_reserved_bytes is None:
                current_reserved_bytes = int(torch.cuda.memory_reserved(device))
            if device_total_memory_bytes is None:
                device_total_memory_bytes = int(torch.cuda.get_device_properties(device).total_memory)
        else:
            current_reserved_bytes = 0 if current_reserved_bytes is None else current_reserved_bytes
            device_total_memory_bytes = max(1 << 30, 4 * incremental) if device_total_memory_bytes is None else device_total_memory_bytes
    current = _nonnegative_int(current_reserved_bytes, name="current reserved bytes")
    total = _positive_int(device_total_memory_bytes, name="device total memory bytes")
    ceiling = math.floor(design.hbm_ceiling_fraction * total)
    peak = current + incremental
    admission_memory = (
        cuda_reusable_memory_bytes(device, framework=torch),
        int(torch.cuda.memory_allocated(device)),
    ) if live_admission else None
    available, allocated = admission_memory or (ceiling, current)
    return V1SignedWorkNormalConeActionInvarianceCUDAHBMForecast(
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        least_work_execution_hash=execution.execution_hash,
        projection_receipt_hash=projection.receipt.receipt_hash,
        shell_execution_hash=shell.execution_hash,
        design_hash=design.design_hash,
        device=state.device,
        route_count=n,
        response_rank=source.rank,
        route_tile_size=design.route_tile_size,
        maximum_shell_routes=design.maximum_shell_routes,
        borrowed_route_vectors_bytes=borrowed,
        parent_live_bytes=parent_bytes,
        candidate_label_clone_bytes=candidate_labels,
        q_dual_adjoint_workspace_bytes_upper=q_dual_workspace,
        scan_tile_bytes_upper=scan_tile,
        hard_replay_workspace_bytes_upper=hard_replay,
        compact_shell_bytes_upper=compact_shell,
        allocator_guard_bytes=guard,
        forecast_incremental_bytes=incremental,
        current_reserved_bytes=current,
        forecast_peak_reserved_bytes=peak,
        device_total_memory_bytes=total,
        hbm_ceiling_bytes=ceiling,
        admitted=allocated + incremental <= ceiling and incremental <= available,
        _admission_memory=admission_memory,
    )


def _parent_live_tensor(parent_mask: HardMask, *, device: Any) -> Any:
    # Host-to-device transfer is permitted; the prohibition is route-scale
    # result/key/index D2H.  This array is the exact caller-bound parent input.
    source = np.ascontiguousarray(parent_mask.to_bool_array())
    return torch.tensor(source, dtype=torch.bool, device=device).contiguous()


def _block_dual_native(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    scales: Any,
    block_dual: Any,
) -> tuple[float, ...]:
    values = block_dual.detach().cpu().numpy().astype(np.float64, copy=True)
    if values.shape != (len(problem.source_problem.block_capacities),):
        _refuse(
            V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.STALE_IDENTITY,
            "solver block-dual axis differs from the registered compact blocks",
        )
    if not np.isfinite(values).all():
        _refuse(
            V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.NUMERICAL_REFUSAL,
            "solver block dual contains a nonfinite value",
        )
    negative_roundoff = math.nextafter(
        _DUAL_NONNEGATIVE_ROUNDOFF_ULPS * np.finfo(np.float64).eps,
        math.inf,
    )
    if bool(np.any(values < -negative_roundoff)):
        _refuse(
            V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.NUMERICAL_REFUSAL,
            "solver block dual is materially negative beyond binary64 roundoff",
        )
    native: list[float] = []
    for value, scale in zip(values, scales.block, strict=True):
        clipped = 0.0 if float(value) < 0.0 else float(value)
        native.append(
            _next_up(clipped * scales.work / scale) if clipped != 0.0 else 0.0
        )
    return tuple(native)


def _downward_nonnegative_device_sum(values: Any) -> float:
    """Outward lower endpoint for one nonnegative device-vector sum."""

    nominal = float(torch.sum(values).item())
    envelope = float(torch.sum(torch.abs(values)).item())
    lower, _upper = _nonnegative_interval_from_nominal(
        nominal=nominal,
        absolute_envelope=envelope,
        operation_count=max(1, 2 * int(values.numel()) + 3),
    )
    return lower


def _block_complementarity_slack_upper(
    *,
    values: Any,
    removal_capacity: int,
    native_multiplier: float,
) -> float:
    """Outward upper block complementarity from a downward block sum."""

    block_lower = _downward_nonnegative_device_sum(values)
    slack_upper = max(
        0.0,
        _next_up(float(removal_capacity) - block_lower),
    )
    return _up_multiply(native_multiplier, slack_upper)


@dataclass(slots=True)
class _QDualAdjointState:
    center: Any
    radius: Any
    fingerprint: str
    maximum_numerical_radius: float
    soc_dual_norm_upper: float
    soc_response_norm_lower: float
    soc_response_norm_upper: float
    soc_nonradial_residual_norm_upper: float
    soc_response_near_zero: bool
    q_complementarity_upper: float


def _sqrt_interval(lower: float, upper: float) -> tuple[float, float]:
    if lower < 0.0 or lower > upper or not math.isfinite(upper):
        _refuse(
            V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.NUMERICAL_REFUSAL,
            "cannot take an outward square root of the supplied interval",
        )
    lower_root = math.sqrt(lower)
    upper_root = math.sqrt(upper)
    return (
        0.0 if lower_root == 0.0 else max(0.0, _next_down(lower_root)),
        0.0 if upper_root == 0.0 else _next_up(upper_root),
    )


def _device_pair_fingerprint(center: Any, radius: Any, *, role: str) -> str:
    """Compact device reduction fingerprint; never transfers route values."""

    values: list[float] = []
    for tensor in (center, radius):
        values.extend(
            (
                float(torch.sum(tensor).item()),
                float(torch.sum(torch.abs(tensor)).item()),
                float(torch.dot(tensor, tensor).item()),
                float(torch.min(tensor).item()),
                float(torch.max(tensor).item()),
            )
        )
    if not all(math.isfinite(value) for value in values):
        _refuse(
            V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.NUMERICAL_REFUSAL,
            "actual Q-dual adjoint fingerprint contains a nonfinite reduction",
        )
    return content_hash(
        {
            "schema": "fixed-tensor-actual-q-dual-adjoint-fingerprint-v1",
            "role": role,
            "shape": tuple(int(item) for item in center.shape),
            "dtype": str(center.dtype),
            "device": str(center.device),
            "scalar_reductions": tuple(values),
        },
        prefix=_PREFIX + "q-dual-adjoint-",
    )


def _actual_q_dual_adjoint_state(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    branch: V1RelaxedKKTBranch,
    primal: Any,
    factor: Any,
    sqrt_residual: Any,
    scales: Any,
    low_rank_dual: Any,
    q_tail_dual: Any,
) -> _QDualAdjointState:
    """Return the actual routewise SOC-dual adjoint with outward roundoff.

    No scalar Q multiplier is inferred.  This is the exact dual coordinate of
    the registered FD convex branch.  A nonradial SOC dual remains lawful for
    a primal-dual gap certificate; its nonradial residual is measured rather
    than silently discarded.  Consequently this object cannot certify the
    literal-J population Q normal cone or alternative-solution stability.
    """

    source = problem.source_problem
    q_scale = (
        scales.q_spectral
        if branch is V1RelaxedKKTBranch.SPECTRAL
        else scales.q_diagonal
    )
    low_response = factor @ primal
    low_dual = low_rank_dual.reshape(-1)
    if low_dual.shape != low_response.shape:
        _refuse(
            V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.STALE_IDENTITY,
            "low-rank SOC dual and registered response axes differ",
        )
    low_center = factor.T @ low_dual
    low_envelope = torch.abs(factor).T @ torch.abs(low_dual)
    if branch is V1RelaxedKKTBranch.SPECTRAL:
        tail_dual = q_tail_dual.reshape(-1)
        if tuple(tail_dual.shape) != tuple(primal.shape):
            _refuse(
                V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.STALE_IDENTITY,
                "spectral SOC tail dual differs from the route axis",
            )
        sqrt_rho = math.sqrt(source.residual_spectral_bound)
        tail_center = sqrt_rho * tail_dual
        tail_envelope = abs(sqrt_rho) * torch.abs(tail_dual)
        response_tail = sqrt_rho * primal
    else:
        tail_dual = q_tail_dual.reshape(-1)
        if int(tail_dual.numel()) != 1:
            _refuse(
                V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.STALE_IDENTITY,
                "diagonal SOC tail dual must be one scalar",
            )
        tail_center = sqrt_residual * tail_dual[0]
        tail_envelope = torch.abs(sqrt_residual) * torch.abs(tail_dual[0])
        response_tail = torch.dot(sqrt_residual, primal).reshape(1)
    native_scale = scales.work / q_scale
    center = ((low_center + tail_center) * native_scale).contiguous()
    envelope = ((low_envelope + tail_envelope) * abs(native_scale)).contiguous()
    operation_count = max(1, 2 * source.rank + 9)
    gamma = _gamma_n(operation_count)
    radius = _tensor_next_up(
        envelope * gamma
        + torch.full_like(envelope, operation_count * float(np.finfo(np.float64).tiny))
    ).contiguous()
    if bool(
        (
            (~torch.isfinite(center))
            | (~torch.isfinite(radius))
            | (radius < 0.0)
        ).any().item()
    ):
        _refuse(
            V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.NUMERICAL_REFUSAL,
            "actual SOC-dual adjoint or its outward radius is invalid",
        )

    dual_norm_sq_nominal = float(
        (torch.dot(low_dual, low_dual) + torch.dot(tail_dual, tail_dual)).item()
    )
    dual_sq_lower, dual_sq_upper = _nonnegative_interval_from_nominal(
        nominal=dual_norm_sq_nominal,
        absolute_envelope=dual_norm_sq_nominal,
        operation_count=max(1, 2 * (int(low_dual.numel()) + int(tail_dual.numel())) + 3),
    )
    _dual_norm_lower, dual_norm_upper = _sqrt_interval(
        dual_sq_lower, dual_sq_upper
    )
    response_norm_sq_nominal = float(
        (torch.dot(low_response, low_response) + torch.dot(response_tail, response_tail)).item()
    )
    response_sq_lower, response_sq_upper = _nonnegative_interval_from_nominal(
        nominal=response_norm_sq_nominal,
        absolute_envelope=response_norm_sq_nominal,
        operation_count=max(
            1,
            2 * (int(low_response.numel()) + int(response_tail.numel())) + 3,
        ),
    )
    response_norm_lower, response_norm_upper = _sqrt_interval(
        response_sq_lower, response_sq_upper
    )

    pairing_nominal = float(
        (torch.dot(low_dual, low_response) + torch.dot(tail_dual, response_tail)).item()
    )
    pairing_envelope = float(
        (
            torch.dot(torch.abs(low_dual), torch.abs(low_response))
            + torch.dot(torch.abs(tail_dual), torch.abs(response_tail))
        ).item()
    )
    pairing_lower, _pairing_upper = _signed_sum_interval(
        nominal=pairing_nominal,
        absolute_envelope=pairing_envelope,
        operation_count=max(
            1,
            2 * (int(low_dual.numel()) + int(tail_dual.numel())) + 5,
        ),
    )
    q_radius_scaled = math.sqrt(source.quadratic_budget) / q_scale
    complementarity_scaled_upper = max(
        0.0,
        _next_up(q_radius_scaled * dual_norm_upper) - pairing_lower / q_scale,
    )
    q_complementarity_upper = _up_multiply(
        scales.work, complementarity_scaled_upper
    )

    # Nonradiality is diagnostic only.  Use the certified response lower bound
    # to avoid dividing by a near-zero response.  At/near zero, the complete
    # dual norm is the outward residual bound; no colinearity is claimed.
    near_zero = response_norm_lower == 0.0
    if near_zero:
        nonradial_upper = dual_norm_upper
    else:
        beta = pairing_nominal / response_norm_sq_nominal
        residual_low = low_dual - beta * low_response
        residual_tail = tail_dual - beta * response_tail
        residual_sq = float(
            (torch.dot(residual_low, residual_low) + torch.dot(residual_tail, residual_tail)).item()
        )
        _, residual_sq_upper = _nonnegative_interval_from_nominal(
            nominal=residual_sq,
            absolute_envelope=float(
                (
                    torch.dot(torch.abs(residual_low), torch.abs(residual_low))
                    + torch.dot(torch.abs(residual_tail), torch.abs(residual_tail))
                ).item()
            ),
            operation_count=max(
                1,
                2 * (int(residual_low.numel()) + int(residual_tail.numel())) + 7,
            ),
        )
        nonradial_upper = _sqrt_interval(0.0, residual_sq_upper)[1]
    maximum_radius = float(torch.max(radius).item())
    return _QDualAdjointState(
        center=center,
        radius=radius,
        fingerprint=_device_pair_fingerprint(
            center, radius, role=f"actual_{branch.value}_SOC_dual_adjoint"
        ),
        maximum_numerical_radius=maximum_radius,
        soc_dual_norm_upper=dual_norm_upper,
        soc_response_norm_lower=response_norm_lower,
        soc_response_norm_upper=response_norm_upper,
        soc_nonradial_residual_norm_upper=nonradial_upper,
        soc_response_near_zero=near_zero,
        q_complementarity_upper=q_complementarity_upper,
    )


@dataclass(slots=True)
class _ScanState:
    shell_indices: tuple[int, ...]
    shell_z: tuple[float, ...]
    shell_adv_lower: tuple[float, ...]
    shell_adv_upper: tuple[float, ...]
    shell_labels: tuple[int, ...]
    shell_eligible: tuple[bool, ...]
    shell_required: tuple[bool, ...]
    shell_parent_live: tuple[bool, ...]
    shell_rounding: tuple[bool, ...]
    shell_normal: tuple[bool, ...]
    normal_face_options: tuple[tuple[int, ...], ...]
    outside_selected_count: int
    outside_block_selected_counts: tuple[int, ...]
    target_removed_count: int
    q_lower: float
    q_upper: float
    fd_q_lower: float
    fd_q_upper: float
    resource_lower: float
    resource_upper: float
    normal_gap_upper: float
    complementarity_upper: float
    epsilon_upper: float
    q_dual_adjoint_fingerprint: str
    q_dual_adjoint_maximum_numerical_radius: float
    soc_dual_norm_upper: float
    soc_response_norm_lower: float
    soc_response_norm_upper: float
    soc_nonradial_residual_norm_upper: float
    soc_response_near_zero: bool
    resource_multiplier: float
    residual_linear_center_fingerprint: str
    residual_linear_center_max_abs: float
    block_multipliers: tuple[float, ...]
    normal_count: int
    rounding_count: int
    zero_count: int
    scalar_transfers: int


def _scan_fixed_primal_shell(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    branch: V1RelaxedKKTBranch,
    branch_receipt: Any,
    projection: V1SignedWorkFixedPrimalProjectionCUDAExecution,
    shell: V1SignedWorkDeviceShellExecutionBundle,
    parent_live: Any,
    labels: Any,
    primal: Any,
    source_tensors: dict[str, Any],
    duals: dict[str, Any],
    design: V1SignedWorkNormalConeActionInvarianceCUDADesign,
) -> _ScanState:
    source = problem.source_problem
    n = source.route_count
    device = primal.device
    if primal.dtype != torch.float64 or primal.ndim != 1 or int(primal.numel()) != n:
        _refuse(V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.STALE_IDENTITY, "solver primal is malformed")
    fixed = projection.device_handle
    with fixed.borrow() as projected:
        projected_z = projected.fixed_primal
        if projected_z.device != device or projected_z.dtype != torch.float64 or int(projected_z.numel()) != n:
            _refuse(V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.STALE_IDENTITY, "projection primal custody differs from solver device/axis")
        mismatch = bool(torch.any(projected_z != primal).item())
        if mismatch:
            _refuse(V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.FIXED_PRIMAL_MISMATCH, "projection was not measured at the exact solver-issued primal")
        f_center = projected.signed_f_subgradient_center
        f_radius = projected.signed_f_subgradient_radius
        q_center = projected.literal_j_q_gradient_center
        q_radius = projected.literal_j_q_gradient_radius
        zero_bitmap = projected.zero_cell_ambiguity_bitmap

        resource = source_tensors["resource"]
        factor = source_tensors["factor"]
        sqrt_residual = source_tensors["sqrt_residual"]
        eligible = source_tensors["eligible"]
        required = source_tensors["required"]
        cut_count = _positive_int(
            problem.signed_work_operator.native_fiber_axis_receipt.cut_count,
            name="native-fiber cut count",
        )
        residual_linear = problem.signed_work_operator.residual_linear
        if any(value.device != device for value in (resource, factor, sqrt_residual, eligible, required)):
            _refuse(V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.STALE_IDENTITY, "least-work source tensors span multiple devices")
        if residual_linear.device != device:
            _refuse(
                V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.STALE_IDENTITY,
                "signed-work residual-linear center is not on the solver device",
            )
        if residual_linear.dtype != torch.float64 or residual_linear.ndim != 1 or int(residual_linear.numel()) != n:
            _refuse(
                V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.STALE_IDENTITY,
                "signed-work residual-linear center is malformed",
            )
        if bool((~torch.isfinite(residual_linear)).any().item()) or bool((residual_linear < 0.0).any().item()):
            _refuse(
                V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.NUMERICAL_REFUSAL,
                "signed-work residual-linear center must be finite and nonnegative",
            )
        residual_linear_center_max_abs = _next_up(
            float(torch.max(torch.abs(residual_linear)).item()) / float(cut_count)
        )
        residual_linear_center_fingerprint = content_hash(
            {
                "schema": "fixed-tensor-normal-residual-linear-center-v1",
                "operator_receipt_hash": problem.signed_work_operator.receipt_hash,
                "residual_linear_digest": (
                    problem.signed_work_operator.receipt.residual_linear_digest
                ),
                "logical_zero_residual": (
                    problem.signed_work_operator.residual_is_logical_zero
                ),
                "cut_count": cut_count,
                "normal_center_scale": (1.0 / float(cut_count)).hex(),
                "maximum_center_abs_upper": residual_linear_center_max_abs.hex(),
                "route_scale_host_values_transferred": 0,
            },
            prefix=_PREFIX + "residual-normal-center-",
        )
        if bool(torch.any(primal < 0.0).item()) or bool(torch.any(primal > 1.0).item()):
            _refuse(V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.FIXED_PRIMAL_INFEASIBLE, "fixed primal leaves the exact relaxed box")
        if bool(torch.any(required & (primal != 1.0)).item()) or bool(torch.any((~eligible) & (primal != 0.0)).item()):
            _refuse(V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.FIXED_PRIMAL_INFEASIBLE, "fixed primal violates required/ineligible equalities")

        scales = _canonical_fused_scales(problem, active_branch=branch)
        low = factor @ primal
        low_q = float(torch.dot(low, low).item())
        z_norm_sq = float(torch.dot(primal, primal).item())
        diagonal_projection = float(torch.dot(sqrt_residual, primal).item())
        spectral_q = max(0.0, low_q + source.residual_spectral_bound * z_norm_sq)
        diagonal_q = max(0.0, low_q + diagonal_projection * diagonal_projection)
        del spectral_q, diagonal_q
        q_dual_state = _actual_q_dual_adjoint_state(
            problem=problem,
            branch=branch,
            primal=primal,
            factor=factor,
            sqrt_residual=sqrt_residual,
            scales=scales,
            low_rank_dual=duals["low_rank_dual"],
            q_tail_dual=duals["q_tail_dual"],
        )
        resource_multiplier_raw = float(duals["resource_dual"].item())
        if resource_multiplier_raw < 0.0:
            _refuse(V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.NUMERICAL_REFUSAL, "resource dual is negative")
        resource_multiplier = (
            _next_up(resource_multiplier_raw * scales.work / scales.resource)
            if resource_multiplier_raw else 0.0
        )
        block_multipliers = _block_dual_native(
            problem=problem, scales=scales, block_dual=duals["block_dual"]
        )

        # Literal-J source Q interval, contracted with the same frozen z.
        q_nominal_half = 0.0
        q_radius_half = 0.0
        resource_nominal = 0.0
        resource_abs = 0.0
        for start in range(0, n, design.route_tile_size):
            stop = min(n, start + design.route_tile_size)
            z_tile = primal[start:stop]
            q_nominal_half += 0.5 * float(torch.dot(z_tile, q_center[start:stop]).item())
            q_radius_half += 0.5 * float(torch.dot(torch.abs(z_tile), q_radius[start:stop]).item())
            resource_nominal += float(torch.dot(z_tile, resource[start:stop]).item())
            resource_abs += float(torch.dot(torch.abs(z_tile), torch.abs(resource[start:stop])).item())
        q_lower = max(0.0, _next_down(q_nominal_half - q_radius_half))
        q_upper = _next_up(q_nominal_half + q_radius_half)
        route_ops = max(1, 2 * n + 2 * source.rank + 8)
        resource_lower, resource_upper = _nonnegative_interval_from_nominal(
            nominal=resource_nominal,
            absolute_envelope=resource_abs,
            operation_count=route_ops,
        )
        # Deterministic FD value enclosure for the selected convex branch.
        low_lower, low_upper = _nonnegative_interval_from_nominal(
            nominal=low_q,
            absolute_envelope=low_q,
            operation_count=route_ops,
        )
        if branch is V1RelaxedKKTBranch.SPECTRAL:
            tail_lower = _down_multiply_nonnegative(source.residual_spectral_bound, z_norm_sq)
            tail_upper = _up_multiply(source.residual_spectral_bound, z_norm_sq)
        else:
            diag_lower, diag_upper = _nonnegative_interval_from_nominal(
                nominal=diagonal_projection,
                absolute_envelope=diagonal_projection,
                operation_count=route_ops,
            )
            tail_lower, tail_upper = _square_interval(diag_lower, diag_upper)
        fd_q_lower = max(0.0, _down_add(low_lower, tail_lower))
        fd_q_upper = _up_add(low_upper, tail_upper)
        q_lower = max(q_lower, fd_q_lower)
        q_upper = min(q_upper, fd_q_upper)
        if q_lower > q_upper:
            _refuse(
                V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.SOURCE_FD_DISJOINT,
                "literal-J Q value interval and deterministic FD envelope are disjoint",
            )
        if q_upper > source.quadratic_budget or resource_lower < problem.resource_floor:
            _refuse(
                V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.FIXED_PRIMAL_INFEASIBLE,
                "fixed primal is not robustly feasible under source/FD outward intervals",
            )

        # Baseline selected/unselected gate boundary is found by scalar device
        # reductions only.  Required routes are fixed and not a tie decision.
        candidate = eligible & parent_live & ~required
        selected = (labels == 1) & candidate
        unselected = (labels == 0) & candidate
        selected_count = int(torch.count_nonzero((labels == 1) & parent_live & eligible).item())
        target_count = shell.frozen_ladder.actions[0].cumulative_removed_count
        if selected_count != target_count:
            _refuse(V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.STALE_ROUNDING, "resident one-endpoint labels do not match the frozen action count")
        selected_min = float(torch.min(primal[selected]).item()) if bool(torch.any(selected).item()) else math.inf
        unselected_max = float(torch.max(primal[unselected]).item()) if bool(torch.any(unselected).item()) else -math.inf
        if selected_min < unselected_max:
            _refuse(V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.STALE_ROUNDING, "frozen labels contradict the primary relaxed-gate key")
        cutoff = selected_min if selected_min == unselected_max else math.nan

        shell_indices: list[int] = []
        shell_values: list[tuple[float, float, float, int, bool, bool, bool, bool, bool, tuple[int, ...]]] = []
        normal_count = 0
        rounding_count = 0
        zero_count = 0
        normal_gap_upper = 0.0
        q_intersection_complete = True
        block_cursor = 0
        blocks = source.block_capacities
        for start in range(0, n, design.route_tile_size):
            stop = min(n, start + design.route_tile_size)
            z_t = primal[start:stop]
            low_grad = 2.0 * (factor[:, start:stop].T @ low)
            if branch is V1RelaxedKKTBranch.SPECTRAL:
                fd_radius = torch.full_like(
                    z_t,
                    2.0 * source.residual_spectral_bound * math.sqrt(z_norm_sq),
                )
            else:
                fd_radius = 2.0 * sqrt_residual[start:stop] * abs(diagonal_projection)
            fd_lo = _tensor_next_down(low_grad - fd_radius)
            fd_hi = _tensor_next_up(low_grad + fd_radius)
            source_lo = _tensor_next_down(q_center[start:stop] - q_radius[start:stop])
            source_hi = _tensor_next_up(q_center[start:stop] + q_radius[start:stop])
            q_lo_t = torch.maximum(fd_lo, source_lo)
            q_hi_t = torch.minimum(fd_hi, source_hi)
            disjoint = q_lo_t > q_hi_t
            if bool(torch.any(disjoint).item()):
                q_intersection_complete = False
                _refuse(
                    V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.SOURCE_FD_DISJOINT,
                    "literal-J gradient interval and deterministic FD envelope are disjoint",
                )
            f_lo_t = _tensor_next_down(f_center[start:stop] - f_radius[start:stop])
            f_hi_t = _tensor_next_up(f_center[start:stop] + f_radius[start:stop])
            residual_center_t = residual_linear[start:stop] / float(cut_count)
            residual_lo_t = torch.where(
                residual_center_t == 0.0,
                residual_center_t,
                _tensor_next_down(residual_center_t),
            )
            residual_hi_t = torch.where(
                residual_center_t == 0.0,
                residual_center_t,
                _tensor_next_up(residual_center_t),
            )
            # The source literal-J/FD intersection above is a compatibility
            # and primal-feasibility diagnostic.  It is not substituted for
            # the registered branch normal.  The fixed-tensor normal combines
            # the source-projected signed-F subgradient with the deterministic
            # factored-operator residual-linear center, then uses the actual
            # solver-issued SOC dual adjoint with its deterministic routewise
            # numerical enclosure, avoiding any radial scalarization.
            del q_lo_t, q_hi_t
            q_adjoint_center_t = q_dual_state.center[start:stop]
            q_adjoint_radius_t = q_dual_state.radius[start:stop]
            normal_lo = (
                f_lo_t
                + residual_lo_t
                + q_adjoint_center_t
                - q_adjoint_radius_t
                - resource_multiplier * resource[start:stop]
            )
            normal_hi = (
                f_hi_t
                + residual_hi_t
                + q_adjoint_center_t
                + q_adjoint_radius_t
                - resource_multiplier * resource[start:stop]
            )
            while block_cursor < len(blocks) and blocks[block_cursor].stop <= start:
                block_cursor += 1
            local_block = block_cursor
            while local_block < len(blocks) and blocks[local_block].start < stop:
                block = blocks[local_block]
                overlap_start = max(start, block.start)
                overlap_stop = min(stop, block.stop)
                if overlap_start < overlap_stop:
                    offset_start = overlap_start - start
                    offset_stop = overlap_stop - start
                    normal_lo[offset_start:offset_stop].add_(block_multipliers[local_block])
                    normal_hi[offset_start:offset_stop].add_(block_multipliers[local_block])
                local_block += 1
            normal_lo = _tensor_next_down(normal_lo)
            normal_hi = _tensor_next_up(normal_hi)
            free_t = candidate[start:stop]
            lower_face = (z_t == 0.0) & (normal_lo <= 0.0)
            upper_face = (z_t == 1.0) & (normal_hi >= 0.0)
            interior_face = (z_t > 0.0) & (z_t < 1.0) & (normal_lo <= 0.0) & (normal_hi >= 0.0)
            normal_boundary = free_t & (lower_face | upper_face | interior_face)
            rounding_boundary = free_t & (z_t == cutoff) if math.isfinite(cutoff) else torch.zeros_like(free_t)
            zero_t = zero_bitmap[start:stop]
            # Zero-cell ambiguity already widens the source-F interval above.
            # It changes physical actions only when that widened force interval
            # reaches a normal boundary or when the route is on the primary
            # relaxed-gate rounding threshold.  Treating every zero cell as an
            # independent shell coordinate is safe but needlessly over-refuses.
            shell_t = normal_boundary | rounding_boundary
            # Worst source-normal residual over every interval realization.
            violation = torch.zeros_like(z_t)
            lower_box = z_t == 0.0
            upper_box = z_t == 1.0
            interior_box = (z_t > 0.0) & (z_t < 1.0)
            violation[lower_box & free_t] = torch.clamp(-normal_lo[lower_box & free_t], min=0.0)
            violation[upper_box & free_t] = torch.clamp(normal_hi[upper_box & free_t], min=0.0)
            violation[interior_box & free_t] = torch.maximum(
                torch.abs(normal_lo[interior_box & free_t]),
                torch.abs(normal_hi[interior_box & free_t]),
            )
            normal_gap_upper = _up_add(normal_gap_upper, float(torch.sum(violation).item()))
            tile_shell_count = int(torch.count_nonzero(shell_t).item())
            tile_normal_count = int(torch.count_nonzero(normal_boundary).item())
            tile_rounding_count = int(torch.count_nonzero(rounding_boundary).item())
            tile_zero_count = int(torch.count_nonzero(zero_t).item())
            normal_count += tile_normal_count
            rounding_count += tile_rounding_count
            zero_count += tile_zero_count
            if len(shell_indices) + tile_shell_count > design.maximum_shell_routes:
                _refuse(
                    V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.SHELL_OVERSIZED,
                    "complete device scan found more ambiguous routes than the registered bounded shell",
                )
            if tile_shell_count:
                local_indices = torch.nonzero(shell_t, as_tuple=False).reshape(-1)
                # This D2H is bounded by maximum_shell_routes by construction.
                physical = (local_indices + start).detach().cpu().numpy().astype(np.int64, copy=True)
                bounded = torch.stack(
                    (
                        z_t[local_indices],
                        -normal_hi[local_indices],
                        -normal_lo[local_indices],
                        labels[start:stop][local_indices].to(dtype=torch.float64),
                        eligible[start:stop][local_indices].to(dtype=torch.float64),
                        required[start:stop][local_indices].to(dtype=torch.float64),
                        parent_live[start:stop][local_indices].to(dtype=torch.float64),
                        rounding_boundary[local_indices].to(dtype=torch.float64),
                        normal_boundary[local_indices].to(dtype=torch.float64),
                    ),
                    dim=1,
                ).detach().cpu().numpy()
                lo_values = normal_lo[local_indices].detach().cpu().numpy()
                hi_values = normal_hi[local_indices].detach().cpu().numpy()
                for row, index, lo_value, hi_value in zip(bounded, physical, lo_values, hi_values, strict=True):
                    options = tuple(
                        sign
                        for sign, admitted in (
                            (-1, float(lo_value) < 0.0),
                            (0, float(lo_value) <= 0.0 <= float(hi_value)),
                            (1, float(hi_value) > 0.0),
                        )
                        if admitted
                    )
                    shell_indices.append(int(index))
                    shell_values.append(
                        (
                            float(row[0]), float(row[1]), float(row[2]), int(row[3]),
                            bool(row[4]), bool(row[5]), bool(row[6]), bool(row[7]),
                            bool(row[8]), options or (0,),
                        )
                    )

        outside_selected = selected_count - sum(1 for value in shell_values if value[3] == 1)
        outside_blocks: list[int] = []
        shell_index_tensor = torch.tensor(shell_indices, dtype=torch.int64, device=device) if shell_indices else torch.empty(0, dtype=torch.int64, device=device)
        shell_marker = torch.zeros(n, dtype=torch.bool, device=device)
        if shell_indices:
            shell_marker[shell_index_tensor] = True
        for block in blocks:
            outside_blocks.append(
                int(torch.count_nonzero((labels[block.start:block.stop] == 1) & ~shell_marker[block.start:block.stop]).item())
            )
        del shell_marker, shell_index_tensor
        block_slack = 0.0
        for multiplier, block in zip(block_multipliers, blocks, strict=True):
            block_values = primal[block.start:block.stop]
            # Complementarity uses capacity - block_value.  Its worst upper
            # endpoint therefore requires the downward-enclosed block sum.
            block_slack = _up_add(
                block_slack,
                _block_complementarity_slack_upper(
                    values=block_values,
                    removal_capacity=block.removal_capacity,
                    native_multiplier=multiplier,
                ),
            )
        resource_slack = _up_multiply(resource_multiplier, max(0.0, resource_upper - problem.resource_floor))
        complementarity = _up_add(
            _up_add(q_dual_state.q_complementarity_upper, resource_slack),
            block_slack,
        )
        epsilon = _up_add(
            _up_add(normal_gap_upper, complementarity),
            _up_add(branch_receipt.stationarity_gap, branch_receipt.absolute_duality_gap),
        )
        return _ScanState(
            shell_indices=tuple(shell_indices),
            shell_z=tuple(item[0] for item in shell_values),
            shell_adv_lower=tuple(item[1] for item in shell_values),
            shell_adv_upper=tuple(item[2] for item in shell_values),
            shell_labels=tuple(item[3] for item in shell_values),
            shell_eligible=tuple(item[4] for item in shell_values),
            shell_required=tuple(item[5] for item in shell_values),
            shell_parent_live=tuple(item[6] for item in shell_values),
            shell_rounding=tuple(item[7] for item in shell_values),
            shell_normal=tuple(item[8] for item in shell_values),
            normal_face_options=tuple(item[9] for item in shell_values),
            outside_selected_count=outside_selected,
            outside_block_selected_counts=tuple(outside_blocks),
            target_removed_count=target_count,
            q_lower=q_lower, q_upper=q_upper, fd_q_lower=fd_q_lower, fd_q_upper=fd_q_upper,
            resource_lower=resource_lower, resource_upper=resource_upper,
            normal_gap_upper=normal_gap_upper, complementarity_upper=complementarity,
            epsilon_upper=epsilon,
            q_dual_adjoint_fingerprint=q_dual_state.fingerprint,
            q_dual_adjoint_maximum_numerical_radius=(
                q_dual_state.maximum_numerical_radius
            ),
            soc_dual_norm_upper=q_dual_state.soc_dual_norm_upper,
            soc_response_norm_lower=q_dual_state.soc_response_norm_lower,
            soc_response_norm_upper=q_dual_state.soc_response_norm_upper,
            soc_nonradial_residual_norm_upper=(
                q_dual_state.soc_nonradial_residual_norm_upper
            ),
            soc_response_near_zero=q_dual_state.soc_response_near_zero,
            resource_multiplier=resource_multiplier,
            residual_linear_center_fingerprint=residual_linear_center_fingerprint,
            residual_linear_center_max_abs=residual_linear_center_max_abs,
            block_multipliers=block_multipliers,
            normal_count=normal_count, rounding_count=rounding_count, zero_count=zero_count,
            scalar_transfers=20 + 5 * math.ceil(n / design.route_tile_size) + len(blocks),
        )


def _enumerate_action_subsets(
    *,
    scan: _ScanState,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    design: V1SignedWorkNormalConeActionInvarianceCUDADesign,
) -> tuple[tuple[tuple[int, ...], ...], int, int]:
    shell_positions = tuple(range(len(scan.shell_indices)))
    fixed_selected = tuple(
        position for position in shell_positions
        if scan.shell_labels[position] == 1 and not scan.shell_rounding[position]
    )
    variable = tuple(
        position for position in shell_positions
        if scan.shell_rounding[position]
        and scan.shell_eligible[position]
        and scan.shell_parent_live[position]
        and not scan.shell_required[position]
    )
    required_selected = tuple(
        position for position in shell_positions
        if scan.shell_required[position] and scan.shell_parent_live[position]
    )
    fixed_selected_set = set(fixed_selected) | set(required_selected)
    need = scan.target_removed_count - scan.outside_selected_count - len(fixed_selected_set)
    if need < 0 or need > len(variable):
        _refuse(
            V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.NO_ADMISSIBLE_FACE,
            "bounded shell cannot reproduce the frozen endpoint cardinality",
        )
    baseline = tuple(
        position for position in shell_positions if scan.shell_labels[position] == 1
    )
    actions: set[tuple[int, ...]] = set()
    for choice in itertools.combinations(variable, need):
        selected = tuple(sorted(fixed_selected_set | set(choice)))
        selected_set = set(selected)
        # Exact compact block capacity replay.
        legal = True
        for block_index, block in enumerate(problem.source_problem.block_capacities):
            shell_count = sum(
                1 for position in selected
                if block.start <= scan.shell_indices[position] < block.stop
            )
            if scan.outside_block_selected_counts[block_index] + shell_count > block.removal_capacity:
                legal = False
                break
        if not legal:
            continue
        # Existential lexicographic interval feasibility.  Strict z dominates;
        # equal-z secondary intervals use one common threshold.  Ratio/index
        # ties are deliberately admitted as a superset, which can over-refuse
        # but cannot omit a possible physical action.
        selected_variable = tuple(position for position in variable if position in selected_set)
        excluded_variable = tuple(position for position in variable if position not in selected_set)
        feasible = True
        for left in selected_variable:
            for right in excluded_variable:
                if scan.shell_z[left] < scan.shell_z[right]:
                    feasible = False
                    break
                if (
                    scan.shell_z[left] == scan.shell_z[right]
                    and scan.shell_adv_upper[left] < scan.shell_adv_lower[right]
                ):
                    feasible = False
                    break
            if not feasible:
                break
        if feasible:
            actions.add(selected)
    if not actions or baseline not in actions:
        _refuse(
            V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.NO_ADMISSIBLE_FACE,
            "no admissible shell face contains the frozen physical action",
        )
    normal_face_count = math.prod(len(options) for options in scan.normal_face_options)
    face_count = 0
    # Actually enumerate the bounded Cartesian face *superset*.  The fixed-z
    # physical action depends on (a) exact primary z and (b) the union of
    # secondary-advantage values admitted by each route's outward interval.
    # ``actions`` above is exactly that union: strict-z membership is fixed,
    # while equal-z membership is admitted only when a common interval
    # threshold can rank the selected routes above the excluded routes.
    #
    # A particular normal-sign face may correlate coordinates and eliminate
    # some action from that face.  We intentionally do not infer such
    # unregistered cross-coordinate coupling.  Pairing every enumerated sign
    # face with the complete action union therefore overcounts possible joint
    # faces, but cannot omit an admissible action for this one fixed tensor.
    # A singleton byte-action set is only a fixed-tensor rounding diagnostic:
    # it does not rule out another optimal/epsilon-near tensor and therefore
    # establishes neither action invariance nor solution sensitivity.
    for _normal_face in itertools.product(*scan.normal_face_options):
        for _action in sorted(actions):
            face_count += 1
            if face_count > design.maximum_shell_faces:
                _refuse(
                    V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.FACE_ENUMERATION_EXCEEDED,
                    "bounded shell induces more faces than the registered exact enumerator",
                )
    return tuple(sorted(actions)), normal_face_count, face_count


def _physical_action_equivalence_hash(
    *,
    route_axis_hash: str,
    parent_mask_hash: str,
    outside_membership_fingerprint: str,
    shell_indices: tuple[int, ...],
    selected_shell_positions: tuple[int, ...],
    removed_count: int,
) -> str:
    """Bind an exact fixed-outside diagnostic action without mask bytes.

    The problem identity is deliberately absent.  Under the fixed-outside
    diagnostic construction, the physical child bytes are determined by the
    route axis, the exact parent-mask bytes, the compact outside-membership
    fingerprint, and the bounded selected shell subset.  Binding a solver
    problem hash in the parent coordinate would allow two different current
    masks under an otherwise identical problem identity to alias.
    """

    return content_hash(
        {
            "schema": "bounded-shell-physical-action-equivalence-v2",
            "route_axis_hash": _identity(route_axis_hash, name="route axis hash"),
            "parent_mask_hash": _identity(parent_mask_hash, name="parent mask hash"),
            "outside_membership_fingerprint": _identity(
                outside_membership_fingerprint,
                name="outside membership fingerprint",
            ),
            "selected_shell_physical_indices": tuple(
                shell_indices[position] for position in selected_shell_positions
            ),
            "removed_count": _positive_int(removed_count, name="removed count"),
        },
        prefix=_PREFIX + "physical-action-equivalence-",
    )


def _hard_replay_candidate(
    *,
    ordinal: int,
    labels: Any,
    selected_shell_positions: tuple[int, ...],
    scan: _ScanState,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    parent_mask_hash: str,
    outside_membership_fingerprint: str,
    source_tensors: dict[str, Any],
    design: V1SignedWorkNormalConeActionInvarianceCUDADesign,
) -> V1SignedWorkNormalConeHardCandidateReplayReceipt:
    candidate_labels = labels.clone() if type(labels) is torch.Tensor else torch.tensor(np.array(labels, copy=True), dtype=torch.uint8)
    if scan.shell_indices:
        shell_indices = torch.tensor(scan.shell_indices, dtype=torch.int64, device=candidate_labels.device)
        candidate_labels[shell_indices] = 0
        selected_indices = tuple(scan.shell_indices[position] for position in selected_shell_positions)
        if selected_indices:
            selected_t = torch.tensor(selected_indices, dtype=torch.int64, device=candidate_labels.device)
            candidate_labels[selected_t] = 1
    summary = _tiled_lane_summary(
        labels=candidate_labels,
        endpoint_labels=(1,),
        problem=problem,
        source_tensors=source_tensors,
        route_tile_size=design.hard_replay_route_tile_size,
    )
    source = problem.source_problem
    count = int(torch.count_nonzero(candidate_labels == 1).item())
    route_ops = max(1, 2 * source.route_count + 2 * source.rank + 5)
    outer_ops = max(
        1,
        4 * source.route_count
        + 2 * problem.signed_work_operator.context_count * problem.signed_work_operator.fiber_count
        + 2 * source.rank + 8,
    )
    resource_nominal = float(summary.resource[0])
    resource_lower, resource_upper = _nonnegative_interval_from_nominal(
        nominal=resource_nominal, absolute_envelope=resource_nominal, operation_count=route_ops,
    )
    raw_nominal = float(summary.signed_f_cells[0] + summary.residual_linear[0])
    raw_envelope = float(summary.signed_f_envelope[0] + summary.residual_linear[0])
    raw_lower, raw_upper = _nonnegative_interval_from_nominal(
        nominal=raw_nominal, absolute_envelope=raw_envelope, operation_count=outer_ops,
    )
    cut_count = problem.signed_work_operator.native_fiber_axis_receipt.cut_count
    f_lower, f_upper = _divide_nonnegative_interval(raw_lower, raw_upper, cut_count)
    low_intervals = tuple(
        _signed_sum_interval(
            nominal=float(value), absolute_envelope=float(envelope), operation_count=route_ops,
        )
        for value, envelope in zip(summary.factor_projection[:, 0], summary.factor_absolute_projection[:, 0], strict=True)
    )
    low_lower, low_upper = _sum_nonnegative_intervals(tuple(_square_interval(*item) for item in low_intervals))
    rho = source.residual_spectral_bound
    spectral_tail_lower = _down_multiply_nonnegative(rho, float(count))
    spectral_tail_upper = _up_multiply(rho, float(count))
    diagonal_nominal = float(summary.diagonal_projection[0])
    diagonal_projection_lower, diagonal_projection_upper = _nonnegative_interval_from_nominal(
        nominal=diagonal_nominal, absolute_envelope=diagonal_nominal, operation_count=route_ops,
    )
    diagonal_tail_lower, diagonal_tail_upper = _square_interval(diagonal_projection_lower, diagonal_projection_upper)
    spectral_lower = max(0.0, _down_add(low_lower, spectral_tail_lower))
    spectral_upper = _up_add(low_upper, spectral_tail_upper)
    diagonal_lower = max(0.0, _down_add(low_lower, diagonal_tail_lower))
    diagonal_upper = _up_add(low_upper, diagonal_tail_upper)
    canonical_lower = min(spectral_lower, diagonal_lower)
    canonical_upper = min(spectral_upper, diagonal_upper)
    equivalence_hash = _physical_action_equivalence_hash(
        route_axis_hash=source.route_axis_hash,
        parent_mask_hash=parent_mask_hash,
        outside_membership_fingerprint=outside_membership_fingerprint,
        shell_indices=scan.shell_indices,
        selected_shell_positions=selected_shell_positions,
        removed_count=count,
    )
    return V1SignedWorkNormalConeHardCandidateReplayReceipt(
        candidate_ordinal=ordinal,
        physical_action_equivalence_hash=equivalence_hash,
        removed_count=count,
        resource_lower=resource_lower,
        resource_upper=resource_upper,
        signed_f_lower=f_lower,
        signed_f_upper=f_upper,
        spectral_q_lower=spectral_lower,
        spectral_q_upper=spectral_upper,
        diagonal_q_lower=diagonal_lower,
        diagonal_q_upper=diagonal_upper,
        canonical_q_lower=canonical_lower,
        canonical_q_upper=canonical_upper,
        quadratic_budget=source.quadratic_budget,
        canonical_q_robustly_feasible=(canonical_upper <= source.quadratic_budget),
    )


def diagnose_v1_signed_work_fixed_primal_rounding_cuda(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    execution: V1SignedWorkLeastWorkCUDAExecutionBundle
    | V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle,
    projection: V1SignedWorkFixedPrimalProjectionCUDAExecution,
    shell: V1SignedWorkDeviceShellExecutionBundle,
    parent_mask: HardMask,
    design: V1SignedWorkNormalConeActionInvarianceCUDADesign = V1SignedWorkNormalConeActionInvarianceCUDADesign(),
    current_reserved_bytes: int | None = None,
    device_total_memory_bytes: int | None = None,
) -> V1SignedWorkFixedPrimalRoundingDiagnosticCUDAResult:
    """Measure one fixed tensor and enumerate its current-z rounding shell.

    This diagnostic deliberately leaves the alternative-solution quantifier
    open.  It cannot authorize C or establish physical-action invariance.
    """

    branch, branch_receipt, branch_hashes = _validate_identities(
        problem=problem, policy=policy, execution=execution, projection=projection,
        shell=shell, parent_mask=parent_mask, design=design,
    )
    forecast = estimate_v1_signed_work_normal_cone_action_invariance_cuda_hbm(
        problem=problem, policy=policy, execution=execution, projection=projection,
        shell=shell, parent_mask=parent_mask, design=design,
        current_reserved_bytes=current_reserved_bytes,
        device_total_memory_bytes=device_total_memory_bytes,
    )
    if not forecast.admitted:
        _refuse(
            V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.HBM_FORECAST_EXCEEDED,
            "pre-allocation fixed-tensor diagnostic HBM forecast exceeds the 85% ceiling",
        )
    state = execution.state_handle
    if state is None:  # pragma: no cover - validation closes this path.
        raise RuntimeError("least-work state disappeared after HBM admission")
    projection_live_hash = projection.device_handle.assert_live_identity()
    try:
        state_lease_cm = state.borrow_for_signed_work_decision_witness(
            expected_problem_hash=problem.problem_hash,
            expected_policy_hash=policy.policy_hash,
            expected_branch_receipt_hashes=branch_hashes,
        )
        label_lease_cm = shell.label_handle.borrow_for_registered_operation(
            operation=V1DeviceShellBorrowOperation.FQ_HARD_REPLAY
        )
        with state_lease_cm as state_lease, label_lease_cm as label_lease:
            primal = state_lease.primal(branch)
            source_tensors = {
                name: state_lease.source_tensor(name)
                for name in ("resource", "factor", "sqrt_residual", "eligible", "required")
            }
            duals = {
                name: state_lease.decision_dual_tensor(name, branch)
                for name in ("work_dual", "low_rank_dual", "q_tail_dual", "resource_dual", "block_dual")
            }
            # Merely touching work-dual custody is intentional: the source
            # projection replaces its sampled F subgradient, while final
            # solver custody and its receipt gap remain bound to epsilon.
            if not bool(torch.isfinite(duals["work_dual"]).all().item()):
                _refuse(V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.NUMERICAL_REFUSAL, "solver work dual is nonfinite")
            labels_raw = label_lease._labels_for_registered_operation(
                operation=V1DeviceShellBorrowOperation.FQ_HARD_REPLAY
            )
            labels = (
                labels_raw
                if type(labels_raw) is torch.Tensor
                else torch.tensor(np.array(labels_raw, copy=True), dtype=torch.uint8)
            )
            if labels.dtype != torch.uint8 or labels.ndim != 1 or int(labels.numel()) != problem.source_problem.route_count:
                _refuse(V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.STALE_IDENTITY, "resident shell labels are malformed")
            parent_live = _parent_live_tensor(parent_mask, device=primal.device)
            scan = _scan_fixed_primal_shell(
                problem=problem, branch=branch, branch_receipt=branch_receipt,
                projection=projection, shell=shell, parent_live=parent_live,
                labels=labels, primal=primal, source_tensors=source_tensors,
                duals=duals, design=design,
            )
            shell_index_hash = content_hash(
                {
                    "route_axis_hash": problem.source_problem.route_axis_hash,
                    "bounded_indices": scan.shell_indices,
                },
                prefix=_PREFIX + "bounded-shell-indices-",
            )
            outside_membership_fingerprint = content_hash(
                {
                    "schema": "fixed-outside-membership-from-frozen-labels-v1",
                    "route_axis_hash": problem.source_problem.route_axis_hash,
                    "route_count": problem.source_problem.route_count,
                    "endpoint_label": 1,
                    "complete_baseline_label_commitment": (
                        shell.frozen_ladder.label_commitment
                    ),
                    "bounded_shell_index_hash": shell_index_hash,
                    "outside_membership_is_baseline_label_equal_one": True,
                },
                prefix=_PREFIX + "outside-membership-",
            )
            action_subsets, normal_face_count, face_count = _enumerate_action_subsets(
                scan=scan, problem=problem, design=design,
            )
            replays = tuple(
                _hard_replay_candidate(
                    ordinal=ordinal,
                    labels=labels,
                    selected_shell_positions=subset,
                    scan=scan,
                    problem=problem,
                    parent_mask_hash=projection.receipt.current_mask_hash,
                    outside_membership_fingerprint=(
                        outside_membership_fingerprint
                    ),
                    source_tensors=source_tensors,
                    design=design,
                )
                for ordinal, subset in enumerate(action_subsets)
            )
    except V1SignedWorkNormalConeActionInvarianceCUDARefusal:
        raise
    except Exception as error:
        _refuse(
            V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.MUTATED_CUSTODY,
            f"source/solver/shell lease refused or mutated during diagnostic issuance: {error}",
        )
    if any(not replay.canonical_q_robustly_feasible for replay in replays):
        _refuse(
            V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.HARD_Q_UNRESOLVED,
            "at least one admissible physical action is not robustly feasible under complete two-branch Q replay",
        )
    block_multiplier_hash = content_hash(scan.block_multipliers, prefix=_PREFIX + "block-multipliers-")
    outside = V1SignedWorkNormalConeOutsideShellScanReceipt(
        problem_hash=problem.problem_hash,
        projection_receipt_hash=projection.receipt.receipt_hash,
        shell_execution_hash=shell.execution_hash,
        fixed_primal_source_receipt_hash=branch_receipt.receipt_hash,
        branch=branch,
        route_count=problem.source_problem.route_count,
        scanned_route_count=problem.source_problem.route_count,
        shell_route_count=len(scan.shell_indices),
        normal_boundary_route_count=scan.normal_count,
        rounding_boundary_route_count=scan.rounding_count,
        zero_cell_ambiguity_route_count=scan.zero_count,
        shell_index_hash=shell_index_hash,
        outside_membership_fingerprint=outside_membership_fingerprint,
        fixed_primal_q_lower=scan.q_lower,
        fixed_primal_q_upper=scan.q_upper,
        fd_q_lower=scan.fd_q_lower,
        fd_q_upper=scan.fd_q_upper,
        q_gradient_intersection_nonempty_every_route=True,
        resource_lower=scan.resource_lower,
        resource_upper=scan.resource_upper,
        q_dual_adjoint_fingerprint=scan.q_dual_adjoint_fingerprint,
        q_dual_adjoint_maximum_numerical_radius=(
            scan.q_dual_adjoint_maximum_numerical_radius
        ),
        soc_dual_norm_upper=scan.soc_dual_norm_upper,
        soc_response_norm_lower=scan.soc_response_norm_lower,
        soc_response_norm_upper=scan.soc_response_norm_upper,
        soc_nonradial_residual_norm_upper=(
            scan.soc_nonradial_residual_norm_upper
        ),
        soc_response_near_zero=scan.soc_response_near_zero,
        resource_multiplier=scan.resource_multiplier,
        residual_linear_center_fingerprint=(
            scan.residual_linear_center_fingerprint
        ),
        residual_linear_center_max_abs=scan.residual_linear_center_max_abs,
        block_multiplier_hash=block_multiplier_hash,
        normal_gap_upper=scan.normal_gap_upper,
        complementarity_gap_upper=scan.complementarity_upper,
        solver_stationarity_gap=branch_receipt.stationarity_gap,
        solver_absolute_duality_gap=branch_receipt.absolute_duality_gap,
        fixed_branch_epsilon_upper=scan.epsilon_upper,
        scalar_device_to_host_transfers=scan.scalar_transfers,
        bounded_shell_device_to_host_values=len(scan.shell_indices) * 12,
    )
    action_subset_hash = content_hash(
        tuple(tuple(scan.shell_indices[position] for position in subset) for subset in action_subsets),
        prefix=_PREFIX + "action-subsets-",
    )
    shell_face_hash = content_hash(
        {
            "normal_face_options": scan.normal_face_options,
            "actions": action_subset_hash,
            "face_count": face_count,
        },
        prefix=_PREFIX + "faces-",
    )
    baseline_subset = tuple(position for position, label in enumerate(scan.shell_labels) if label == 1)
    enumeration = V1SignedWorkNormalConeShellEnumerationReceipt(
        outside_scan_receipt_hash=outside.receipt_hash,
        bounded_shell_route_count=len(scan.shell_indices),
        selectable_rounding_shell_route_count=sum(scan.shell_rounding),
        normal_face_count=normal_face_count,
        enumerated_face_superset_count=face_count,
        byte_distinct_physical_action_count=len(action_subsets),
        shell_face_hash=shell_face_hash,
        action_subset_hash=action_subset_hash,
        baseline_subset_present=(baseline_subset in action_subsets),
        block_capacity_enforced=True,
        eligibility_enforced=True,
        every_interval_boundary_routed_to_shell=True,
    )
    action = shell.frozen_ladder.actions[0]
    diagnostic = V1SignedWorkFixedPrimalRoundingDiagnosticReceipt(
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        least_work_execution_hash=execution.execution_hash,
        least_work_state_identity_hash=execution.state_identity_hash,
        branch_receipt_hash=branch_receipt.receipt_hash,
        projection_receipt_hash=projection.receipt.receipt_hash,
        projection_live_identity_hash=projection_live_hash,
        shell_execution_hash=shell.execution_hash,
        frozen_ladder_receipt_hash=shell.frozen_ladder.receipt_hash,
        outside_scan_receipt_hash=outside.receipt_hash,
        shell_enumeration_receipt_hash=enumeration.receipt_hash,
        hard_replay_receipt_hashes=tuple(item.receipt_hash for item in replays),
        chart_identity_hash=projection.receipt.chart_identity_hash,
        current_mask_hash=projection.receipt.current_mask_hash,
        route_axis_hash=problem.source_problem.route_axis_hash,
        native_fiber_axis_receipt_hash=projection.receipt.native_fiber_axis_receipt_hash,
        fixed_primal_descriptor_hash=projection.receipt.fixed_primal_descriptor_hash,
        fixed_primal_source_receipt_hash=branch_receipt.receipt_hash,
        registered_resource_coordinate=problem.resource_floor,
        branch=branch,
        fixed_tensor_gap_upper=scan.epsilon_upper,
        observed_frozen_packed_action_hash=action.packed_action_hash,
        observed_frozen_resulting_mask_hash=action.resulting_mask_hash,
        observed_frozen_removed_route_index_hash=action.removed_route_index_hash,
        observed_frozen_removed_count=action.cumulative_removed_count,
        source_familywise_alpha=projection.receipt.familywise_alpha,
        source_alpha_ledger_id=projection.receipt.alpha_ledger_id,
        source_alpha_coordinate_id=projection.receipt.alpha_coordinate_id,
        source_look_ordinal=projection.receipt.look_ordinal,
        source_observed_counts=projection.receipt.observed_counts,
        source_population_counts=projection.receipt.population_counts,
        source_exact_fitting_census=projection.receipt.exact_census_mode,
        source_sampling_alpha_spent=projection.receipt.sampling_alpha_spent,
        source_full_population_decision_stability_authority=False,
        _issuance_token=_ISSUER_TOKEN,
    )
    canonical_shell_indices = RemovedRouteIndexSet.from_indices(
        scan.shell_indices,
        route_count=problem.source_problem.route_count,
    )
    bounded_shell_handoff = V1SignedWorkFixedPrimalBoundedShellHandoff(
        diagnostic_hash=diagnostic.diagnostic_hash,
        outside_scan_receipt_hash=outside.receipt_hash,
        problem_hash=problem.problem_hash,
        projection_receipt_hash=projection.receipt.receipt_hash,
        shell_execution_hash=shell.execution_hash,
        fixed_primal_source_receipt_hash=branch_receipt.receipt_hash,
        chart_identity_hash=projection.receipt.chart_identity_hash,
        current_mask_hash=projection.receipt.current_mask_hash,
        route_axis_hash=problem.source_problem.route_axis_hash,
        native_fiber_axis_receipt_hash=(
            projection.receipt.native_fiber_axis_receipt_hash
        ),
        branch=branch,
        route_count=problem.source_problem.route_count,
        maximum_shell_routes=design.maximum_shell_routes,
        shell_indices=scan.shell_indices,
        shell_index_hash=outside.shell_index_hash,
        canonical_removed_route_index_hash=canonical_shell_indices.index_hash,
        _issuance_token=_ISSUER_TOKEN,
    )
    return V1SignedWorkFixedPrimalRoundingDiagnosticCUDAResult(
        diagnostic=diagnostic,
        forecast=forecast,
        outside_scan=outside,
        shell_enumeration=enumeration,
        hard_replays=replays,
        bounded_shell_handoff=bounded_shell_handoff,
    )


def issue_v1_signed_work_normal_cone_action_invariance_cuda(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    execution: V1SignedWorkLeastWorkCUDAExecutionBundle
    | V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle,
    projection: V1SignedWorkFixedPrimalProjectionCUDAExecution,
    shell: V1SignedWorkDeviceShellExecutionBundle,
    parent_mask: HardMask,
    design: V1SignedWorkNormalConeActionInvarianceCUDADesign = V1SignedWorkNormalConeActionInvarianceCUDADesign(),
    current_reserved_bytes: int | None = None,
    device_total_memory_bytes: int | None = None,
) -> NoReturn:
    """Fail closed: fixed-z evidence cannot prove alternative-action invariance."""

    # This seam is intentionally noncomputational.  Running the bounded
    # fixed-tensor diagnostic first would spend a full device scan while still
    # being unable to discharge the missing quantifier over alternative
    # optimal/epsilon-near primals.  Callers that need descriptive fixed-z
    # evidence must opt into ``diagnose_*`` explicitly.
    del (
        problem,
        policy,
        execution,
        projection,
        shell,
        parent_mask,
        design,
        current_reserved_bytes,
        device_total_memory_bytes,
    )
    _refuse(
        V1SignedWorkNormalConeActionInvarianceCUDARefusalCode.ALTERNATIVE_SOLUTION_SENSITIVITY_UNRESOLVED,
        "fixed-z normal intervals and cutoff-face enumeration do not rule out "
        "a different admissible relaxed optimum or epsilon-near optimum; a "
        "strict outside-coordinate screen plus robust variable-z shell solve "
        "is required before physical-action invariance can be issued",
    )


__all__ = (
    "SIGNED_WORK_FIXED_PRIMAL_BOUNDED_SHELL_HANDOFF_AUTHORITY",
    "SIGNED_WORK_FIXED_PRIMAL_BOUNDED_SHELL_HANDOFF_SCHEMA",
    "SIGNED_WORK_NORMAL_CONE_ACTION_AUTHORITY",
    "V1SignedWorkFixedPrimalBoundedShellHandoff",
    "V1SignedWorkFixedPrimalRoundingDiagnosticReceipt",
    "V1SignedWorkNormalConeActionInvarianceCUDARefusal",
    "V1SignedWorkNormalConeActionInvarianceCUDARefusalCode",
    "V1SignedWorkNormalConeActionInvarianceCUDADesign",
    "V1SignedWorkNormalConeActionInvarianceCUDAHBMForecast",
    "V1SignedWorkFixedPrimalRoundingDiagnosticCUDAResult",
    "V1SignedWorkNormalConeHardCandidateReplayReceipt",
    "V1SignedWorkNormalConeOutsideShellScanReceipt",
    "V1SignedWorkNormalConeShellEnumerationReceipt",
    "diagnose_v1_signed_work_fixed_primal_rounding_cuda",
    "estimate_v1_signed_work_normal_cone_action_invariance_cuda_hbm",
    "issue_v1_signed_work_normal_cone_action_invariance_cuda",
)
