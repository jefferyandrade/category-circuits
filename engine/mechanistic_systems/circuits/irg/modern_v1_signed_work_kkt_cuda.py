"""CUDA-native fixed-resource least-work solver for modern-v1 signed ``F``.

This is the production sibling of the small-domain least-work reference.  It
solves, independently for the two certified FD response envelopes,

``min F_U(z)`` subject to ``resource(z) >= rho``, ``Q_b(z) <= tau_Q``,
eligibility, compact live-survivor capacities, and ``0 <= z <= 1``.

``F_U`` is the native-fiber signed operator plus its certified nonnegative
route residual.  It is *not* the singleton triangle vector and it has no
loss-derived ``tau_F``.  Because signed work is the objective rather than an
L1-ball constraint, the Chambolle--Pock work-dual proximal map is a
coordinatewise box clamp.  No per-iteration sorting or water filling occurs.

The public entry point requires CUDA.  The exact same tensor kernel has an
explicit private CPU-reference mode used only by focused tests.  A converged
receipt certifies one registered relaxed convex branch within numerical
tolerances.  It does not claim hard-mask optimality and grants no C, K, O,
mask-commit, terminal, or Pareto authority.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from enum import Enum
import math
from numbers import Integral, Real
from threading import RLock
from typing import Any, Final, NoReturn

import numpy as np

from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

from .contracts import content_hash
from .modern_v1_relaxed_kkt import (
    V1RelaxedKKTBranch,
    V1RelaxedKKTBlockCapacity,
    V1RelaxedKKTProblem,
)
from .modern_v1_sampled_device_resident_d_session import (
    V1SampledDeviceResidentDKKTGeometryDeviceHandle,
    V1SampledDeviceResidentDKKTGeometryLease,
)
from .modern_v1_signed_work_operator_cuda import V1SignedWorkOperatorCUDA

try:  # pragma: no cover - availability is environment dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


SIGNED_WORK_LEAST_WORK_CUDA_POLICY_SCHEMA: Final = (
    "partition_free_v1_signed_native_fiber_least_work_cuda_policy_v1"
)
SIGNED_WORK_LEAST_WORK_CUDA_PROBLEM_SCHEMA: Final = (
    "partition_free_v1_signed_native_fiber_least_work_cuda_problem_v1"
)
SIGNED_WORK_LEAST_WORK_CUDA_DEVICE_PROBLEM_SCHEMA_V2: Final = (
    "partition_free_v1_signed_native_fiber_least_work_cuda_device_problem_v2"
)
SIGNED_WORK_LEAST_WORK_CUDA_DEVICE_STRUCTURAL_RECEIPT_SEMANTICS_V2: Final = (
    "pre_outcome_exact_same_device_resource_legality_receipt_certifies_finite_"
    "nonnegative_resource_positive_on_eligible_required_subset_block_legal_"
    "and_registered_legal_and_required_resource_scalars_v2"
)
SIGNED_WORK_LEAST_WORK_ELIGIBLE_MASK_IDENTITY_PREFIX: Final = (
    "partition-free-v1-signed-work-kkt-eligible-mask-array-v1-"
)
SIGNED_WORK_LEAST_WORK_REQUIRED_MASK_IDENTITY_PREFIX: Final = (
    "partition-free-v1-signed-work-kkt-required-mask-array-v1-"
)
SIGNED_WORK_LEAST_WORK_CUDA_FORECAST_SCHEMA: Final = (
    "partition_free_v1_signed_native_fiber_least_work_cuda_hbm_forecast_v1"
)
SIGNED_WORK_LEAST_WORK_CUDA_DEVICE_FORECAST_SCHEMA_V2: Final = (
    "partition_free_v1_signed_native_fiber_least_work_cuda_device_hbm_forecast_v2"
)
SIGNED_WORK_LEAST_WORK_CUDA_SINGLE_BRANCH_FORECAST_SCHEMA: Final = (
    "partition_free_v1_signed_native_fiber_least_work_cuda_single_branch_"
    "hbm_forecast_v1"
)
SIGNED_WORK_LEAST_WORK_CUDA_BRANCH_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_signed_native_fiber_least_work_cuda_branch_receipt_v1"
)
SIGNED_WORK_LEAST_WORK_CUDA_RESULT_SCHEMA: Final = (
    "partition_free_v1_signed_native_fiber_least_work_cuda_result_v1"
)
SIGNED_WORK_LEAST_WORK_CUDA_STATE_SCHEMA: Final = (
    "partition_free_v1_signed_native_fiber_least_work_cuda_state_v2"
)
SIGNED_WORK_LEAST_WORK_CUDA_EXECUTION_SCHEMA: Final = (
    "partition_free_v1_signed_native_fiber_least_work_cuda_execution_v1"
)
SIGNED_WORK_LEAST_WORK_CUDA_SINGLE_BRANCH_RESULT_SCHEMA: Final = (
    "partition_free_v1_signed_native_fiber_least_work_cuda_single_branch_result_v2"
)
SIGNED_WORK_LEAST_WORK_CUDA_SINGLE_BRANCH_EXECUTION_SCHEMA: Final = (
    "partition_free_v1_signed_native_fiber_least_work_cuda_single_branch_execution_v2"
)
SIGNED_WORK_LEAST_WORK_CUDA_ALGORITHM_ID: Final = (
    "float64_native_fiber_signed_F_fixed_resource_fused_two_Q_branch_pdhg_v1"
)
SIGNED_WORK_LEAST_WORK_CUDA_AUTHORITY: Final = (
    "registered_fixed_resource_relaxed_signed_F_Q_branch_numerical_proposal_"
    "authority_only_no_hard_optimality_C_K_O_commit_terminal_or_pareto_v1"
)
SIGNED_WORK_LEAST_WORK_EQUATION: Final = (
    "minimize_signed_native_fiber_F_upper_subject_to_resource_at_least_rho_"
    "and_one_certified_Q_branch_and_irreversible_legality_v1"
)
SIGNED_WORK_LEAST_WORK_NO_TAU_F_ID: Final = (
    "no_loss_derived_tau_F_signed_F_is_the_resource_indexed_objective_v1"
)
SIGNED_WORK_LEAST_WORK_BRANCH_ORDER: Final = (
    V1RelaxedKKTBranch.SPECTRAL,
    V1RelaxedKKTBranch.DIAGONAL_L1,
)

_POLICY_PREFIX: Final = "partition-free-v1-signed-least-work-cuda-policy-"
_PROBLEM_PREFIX: Final = "partition-free-v1-signed-least-work-cuda-problem-"
_DEVICE_PROBLEM_PREFIX_V2: Final = (
    "partition-free-v1-signed-least-work-cuda-device-problem-v2-"
)
_FORECAST_PREFIX: Final = "partition-free-v1-signed-least-work-cuda-forecast-"
_DEVICE_FORECAST_PREFIX_V2: Final = (
    "partition-free-v1-signed-least-work-cuda-device-forecast-v2-"
)
_SINGLE_BRANCH_FORECAST_PREFIX: Final = (
    "partition-free-v1-signed-least-work-cuda-single-branch-forecast-"
)
_BRANCH_PREFIX: Final = "partition-free-v1-signed-least-work-cuda-branch-"
_RESULT_PREFIX: Final = "partition-free-v1-signed-least-work-cuda-result-"
_STATE_PREFIX: Final = "partition-free-v1-signed-least-work-cuda-state-"
_EXECUTION_PREFIX: Final = "partition-free-v1-signed-least-work-cuda-execution-"
_STATE_EXCLUSIVE_TRANSFER_TOKEN: Final = object()
_OUTSIDE_PAIR_SOURCE_LEASE_TOKEN: Final = object()
_F64_BYTES: Final = np.dtype(np.float64).itemsize
_BOOL_BYTES: Final = np.dtype(np.bool_).itemsize
_EPS: Final = np.finfo(np.float64).eps


class V1SignedWorkLeastWorkCUDARefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    AXIS_MISMATCH = "axis_mismatch"
    STALE_SOURCE = "stale_source"
    INFEASIBLE_RESOURCE = "infeasible_resource"
    CUDA_REQUIRED = "cuda_required"
    DETERMINISM_REQUIRED = "determinism_required"
    HBM_FORECAST_EXCEEDED = "hbm_forecast_exceeded"
    NONFINITE = "nonfinite"
    NONCONVERGENCE = "nonconvergence"
    HANDLE_EVICTED = "handle_evicted"
    ACTIVE_LEASE = "active_lease"
    CUSTODY_MUTATED = "custody_mutated"
    AUTHORITY_VIOLATION = "authority_violation"
    HASH_MISMATCH = "hash_mismatch"


class V1SignedWorkLeastWorkCUDARefusal(RuntimeError):
    def __init__(
        self,
        code: V1SignedWorkLeastWorkCUDARefusalCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SignedWorkLeastWorkCUDARefusalCode,
    message: str,
) -> NoReturn:
    raise V1SignedWorkLeastWorkCUDARefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SignedWorkLeastWorkCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SignedWorkLeastWorkCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one positive integer",
        )
    result = int(value)
    if result <= 0:
        _refuse(
            V1SignedWorkLeastWorkCUDARefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SignedWorkLeastWorkCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SignedWorkLeastWorkCUDARefusalCode.INVALID_TYPE,
            f"{name} must be nonnegative",
        )
    return result


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1SignedWorkLeastWorkCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one finite real",
        )
    result = float(value)
    if not math.isfinite(result):
        _refuse(
            V1SignedWorkLeastWorkCUDARefusalCode.NONFINITE,
            f"{name} must be finite",
        )
    return result


def _nonnegative(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if result < 0.0:
        _refuse(
            V1SignedWorkLeastWorkCUDARefusalCode.INVALID_TYPE,
            f"{name} must be nonnegative",
        )
    return result


def _positive(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if result <= 0.0:
        _refuse(
            V1SignedWorkLeastWorkCUDARefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _extended_real(value: object, *, name: str) -> float:
    """Return a real receipt field while rejecting NaN.

    The fused solver has one explicit nonconvergence receipt whose unresolved
    optimality fields are represented by signed infinities.  Receipt
    validation therefore cannot globally require finiteness, but NaN is never
    a meaningful certificate or sentinel.
    """

    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1SignedWorkLeastWorkCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one real",
        )
    result = float(value)
    if math.isnan(result):
        _refuse(
            V1SignedWorkLeastWorkCUDARefusalCode.NONFINITE,
            f"{name} must not be NaN",
        )
    return result


def _receipt_values_agree(left: float, right: float) -> bool:
    """ULP-scaled equality for independently recomputed FP64 receipt fields."""

    if not (math.isfinite(left) and math.isfinite(right)):
        return False
    return abs(left - right) <= 512.0 * _EPS * max(
        1.0,
        abs(left),
        abs(right),
    )


def _readonly_bool_vector(value: object, *, size: int, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.bool_).reshape(-1)
    except (TypeError, ValueError, OverflowError) as error:
        raise V1SignedWorkLeastWorkCUDARefusal(
            V1SignedWorkLeastWorkCUDARefusalCode.INVALID_TYPE,
            f"{name} must be a boolean vector",
        ) from error
    if result.shape != (size,):
        _refuse(
            V1SignedWorkLeastWorkCUDARefusalCode.AXIS_MISMATCH,
            f"{name} must have shape ({size},)",
        )
    frozen = np.frombuffer(result.tobytes(order="C"), dtype=np.bool_)
    frozen.setflags(write=False)
    return frozen


@dataclass(frozen=True, slots=True)
class V1SignedWorkLeastWorkCUDAPolicy:
    policy_id: str = "modern-v1-signed-least-work-cuda-default"
    torch_device: str = "cuda"
    max_iterations: int = 4096
    minimum_iterations: int = 64
    check_interval: int = 16
    over_relaxation: float = 1.0
    primal_feasibility_tolerance: float = 1e-9
    normalized_stationarity_tolerance: float = 1e-6
    normalized_complementarity_tolerance: float = 1e-6
    absolute_gap_tolerance: float = 1e-9
    relative_gap_tolerance: float = 1e-6
    hbm_ceiling_fraction: float = 0.85
    materialize_route_limit: int = 4096
    production_cuda_required: bool = True
    allow_private_cpu_reference: bool = False
    require_deterministic_algorithms: bool = True
    schema: str = SIGNED_WORK_LEAST_WORK_CUDA_POLICY_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _identity(self.policy_id, name="policy ID"))
        object.__setattr__(
            self,
            "torch_device",
            _identity(self.torch_device, name="torch device"),
        )
        for name in ("max_iterations", "minimum_iterations", "check_interval"):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name),
            )
        if self.minimum_iterations > self.max_iterations:
            raise ValueError("minimum_iterations exceeds max_iterations")
        if self.check_interval > self.max_iterations:
            raise ValueError("check_interval exceeds max_iterations")
        theta = _nonnegative(self.over_relaxation, name="over relaxation")
        if theta > 1.0:
            raise ValueError("over_relaxation must not exceed one")
        object.__setattr__(self, "over_relaxation", theta)
        for name in (
            "primal_feasibility_tolerance",
            "normalized_stationarity_tolerance",
            "normalized_complementarity_tolerance",
            "absolute_gap_tolerance",
            "relative_gap_tolerance",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative(getattr(self, name), name=name),
            )
        ceiling = _positive(self.hbm_ceiling_fraction, name="HBM ceiling fraction")
        if ceiling > 0.85:
            raise ValueError("canonical HBM ceiling cannot exceed 0.85")
        object.__setattr__(self, "hbm_ceiling_fraction", ceiling)
        object.__setattr__(
            self,
            "materialize_route_limit",
            _nonnegative_int(self.materialize_route_limit, name="materialize route limit"),
        )
        for name in (
            "production_cuda_required",
            "allow_private_cpu_reference",
            "require_deterministic_algorithms",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact bool")
        if self.production_cuda_required and self.allow_private_cpu_reference:
            raise ValueError("production CUDA policy cannot enable CPU reference mode")
        if self.schema != SIGNED_WORK_LEAST_WORK_CUDA_POLICY_SCHEMA:
            raise ValueError("unsupported signed least-work CUDA policy schema")

    @property
    def policy_hash(self) -> str:
        return content_hash(self, prefix=_POLICY_PREFIX)


def _maximum_resource_under_registered_blocks(problem: V1RelaxedKKTProblem) -> float:
    resource = problem.resource
    eligible = problem.eligible_mask
    if not problem.block_capacities:
        return float(np.sum(resource[eligible], dtype=np.float64))
    total = 0.0
    for block in problem.block_capacities:
        values = resource[block.start : block.stop][eligible[block.start : block.stop]]
        capacity = min(block.removal_capacity, int(values.size))
        if capacity <= 0:
            continue
        if capacity == int(values.size):
            total += float(np.sum(values, dtype=np.float64))
        else:
            split = int(values.size) - capacity
            total += float(np.sum(np.partition(values, split)[split:], dtype=np.float64))
    return total


@dataclass(frozen=True, slots=True)
class V1SignedWorkLeastWorkCUDAProblem:
    source_problem: V1RelaxedKKTProblem = field(repr=False, compare=False)
    signed_work_operator: V1SignedWorkOperatorCUDA = field(repr=False, compare=False)
    resource_floor: float
    source_registration_hash: str
    registered_d_coordinate_hash: str
    source_bundle_authoritative: bool
    required_hard_action: np.ndarray | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    objective_equation: str = SIGNED_WORK_LEAST_WORK_EQUATION
    no_tau_f_policy: str = SIGNED_WORK_LEAST_WORK_NO_TAU_F_ID
    c_stream_outcomes_seen: bool = False
    calibration_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_claim_authority: bool = False
    authority_scope: str = SIGNED_WORK_LEAST_WORK_CUDA_AUTHORITY
    schema: str = SIGNED_WORK_LEAST_WORK_CUDA_PROBLEM_SCHEMA
    _problem_hash: str | None = field(init=False, default=None, repr=False)
    _eligible_mask_identity_hash: str = field(init=False, repr=False)
    _required_mask_identity_hash: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.source_problem) is not V1RelaxedKKTProblem:
            raise TypeError("least-work CUDA problem requires exact relaxed source")
        if type(self.signed_work_operator) is not V1SignedWorkOperatorCUDA:
            raise TypeError("least-work CUDA problem requires exact signed operator")
        base = self.source_problem
        operator = self.signed_work_operator
        operator.receipt.assert_integrity()
        floor = _positive(self.resource_floor, name="resource floor")
        if (
            operator.route_count != base.route_count
            or operator.receipt.route_axis_hash != base.route_axis_hash
            or operator.receipt.source_geometry_hash != base.source_geometry_hash
        ):
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.STALE_SOURCE,
                "signed-F operator and Q/resource source do not share one chart",
            )
        maximum = _maximum_resource_under_registered_blocks(base)
        tolerance = 64.0 * _EPS * max(1.0, floor, maximum)
        if floor > maximum + tolerance:
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.INFEASIBLE_RESOURCE,
                "resource floor exceeds the registered box/block maximum",
            )
        required: np.ndarray | None = None
        if self.required_hard_action is not None:
            required = _readonly_bool_vector(
                self.required_hard_action,
                size=base.route_count,
                name="required hard action",
            )
            if bool(np.any(required & ~base.eligible_mask)):
                _refuse(
                    V1SignedWorkLeastWorkCUDARefusalCode.AXIS_MISMATCH,
                    "required hard action contains an ineligible route",
                )
            required_resource = float(np.dot(base.resource, required.astype(np.float64)))
            if required_resource > floor + tolerance:
                _refuse(
                    V1SignedWorkLeastWorkCUDARefusalCode.INFEASIBLE_RESOURCE,
                    "required prior hard rung already exceeds the next resource coordinate",
                )
            for block in base.block_capacities:
                if int(np.count_nonzero(required[block.start : block.stop])) > block.removal_capacity:
                    _refuse(
                        V1SignedWorkLeastWorkCUDARefusalCode.INFEASIBLE_RESOURCE,
                        "required prior hard rung violates a live-derived block capacity",
                    )
        for name in ("source_registration_hash", "registered_d_coordinate_hash"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if type(self.source_bundle_authoritative) is not bool:
            raise TypeError("source_bundle_authoritative must be an exact bool")
        forbidden = (
            self.c_stream_outcomes_seen,
            self.calibration_authority,
            self.bce_oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_claim_authority,
        )
        if any(type(value) is not bool for value in forbidden) or any(forbidden):
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.AUTHORITY_VIOLATION,
                "least-work proposal problem crossed its pre-C authority boundary",
            )
        if (
            self.objective_equation != SIGNED_WORK_LEAST_WORK_EQUATION
            or self.no_tau_f_policy != SIGNED_WORK_LEAST_WORK_NO_TAU_F_ID
            or self.authority_scope != SIGNED_WORK_LEAST_WORK_CUDA_AUTHORITY
            or self.schema != SIGNED_WORK_LEAST_WORK_CUDA_PROBLEM_SCHEMA
        ):
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.AUTHORITY_VIOLATION,
                "least-work problem semantics changed",
            )
        object.__setattr__(self, "resource_floor", floor)
        object.__setattr__(self, "required_hard_action", required)
        effective_required = (
            np.zeros(base.route_count, dtype=np.bool_)
            if required is None
            else required
        )
        object.__setattr__(
            self,
            "_eligible_mask_identity_hash",
            content_hash(
                np.asarray(base.eligible_mask, dtype=np.bool_),
                prefix=SIGNED_WORK_LEAST_WORK_ELIGIBLE_MASK_IDENTITY_PREFIX,
            ),
        )
        object.__setattr__(
            self,
            "_required_mask_identity_hash",
            content_hash(
                np.asarray(effective_required, dtype=np.bool_),
                prefix=SIGNED_WORK_LEAST_WORK_REQUIRED_MASK_IDENTITY_PREFIX,
            ),
        )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "source_problem_hash": self.source_problem.problem_hash,
            "signed_work_operator_receipt_hash": self.signed_work_operator.receipt_hash,
            "resource_floor": self.resource_floor,
            "source_registration_hash": self.source_registration_hash,
            "registered_d_coordinate_hash": self.registered_d_coordinate_hash,
            "source_bundle_authoritative": self.source_bundle_authoritative,
            "required_hard_action_hash": (
                None
                if self.required_hard_action is None
                else content_hash(self.required_hard_action)
            ),
            "eligible_mask_identity_hash": self._eligible_mask_identity_hash,
            "required_mask_identity_hash": self._required_mask_identity_hash,
            "objective_equation": self.objective_equation,
            "no_tau_f_policy": self.no_tau_f_policy,
            "c_stream_outcomes_seen": self.c_stream_outcomes_seen,
            "calibration_authority": self.calibration_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_claim_authority": self.pareto_claim_authority,
            "authority_scope": self.authority_scope,
        }

    @property
    def problem_hash(self) -> str:
        return self.assert_integrity()

    @property
    def eligible_mask_identity_hash(self) -> str:
        self.assert_integrity()
        return self._eligible_mask_identity_hash

    @property
    def required_mask_identity_hash(self) -> str:
        self.assert_integrity()
        return self._required_mask_identity_hash

    def assert_integrity(self) -> str:
        self.signed_work_operator.receipt.assert_integrity()
        observed = content_hash(self._payload(), prefix=_PROBLEM_PREFIX)
        if self._problem_hash is not None and observed != self._problem_hash:
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.HASH_MISMATCH,
                "least-work CUDA problem was tampered",
            )
        object.__setattr__(self, "_problem_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1SignedWorkLeastWorkCUDAHBMForecast:
    problem_hash: str
    policy_hash: str
    operator_forecast_hash: str
    device: str
    route_count: int
    response_rank: int
    context_count: int
    fiber_count: int
    block_count: int
    persistent_source_bytes: int
    solver_state_bytes: int
    work_dual_bytes: int
    transient_workspace_bytes: int
    forecast_incremental_bytes: int
    current_reserved_bytes: int
    forecast_peak_reserved_bytes: int
    device_total_memory_bytes: int
    hbm_ceiling_bytes: int
    admitted: bool
    schema: str = SIGNED_WORK_LEAST_WORK_CUDA_FORECAST_SCHEMA

    @property
    def forecast_hash(self) -> str:
        return content_hash(self, prefix=_FORECAST_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkLeastWorkCUDASingleBranchHBMForecast:
    """Pre-allocation HBM bound for one Phase-A-supported branch.

    The complete source geometry remains resident because downstream hard
    replay certifies both Q envelopes for each frozen hard action.  Solver
    iterates and signed-work dual cells, however, have exactly one branch
    lane.  Keeping this receipt distinct from the fused two-branch forecast
    prevents an excluded branch from consuming admission capacity or being
    represented in the Phase-B numerical authority.
    """

    problem_hash: str
    policy_hash: str
    operator_forecast_hash: str
    branch: V1RelaxedKKTBranch
    device: str
    route_count: int
    response_rank: int
    context_count: int
    fiber_count: int
    block_count: int
    persistent_source_bytes: int
    solver_state_bytes: int
    work_dual_bytes: int
    transient_workspace_bytes: int
    forecast_incremental_bytes: int
    current_reserved_bytes: int
    forecast_peak_reserved_bytes: int
    device_total_memory_bytes: int
    hbm_ceiling_bytes: int
    admitted: bool
    branch_lane_count: int = 1
    nonrequested_branch_solver_bytes: int = 0
    complete_hard_q_replay_source_retained: bool = True
    schema: str = SIGNED_WORK_LEAST_WORK_CUDA_SINGLE_BRANCH_FORECAST_SCHEMA

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("single-branch forecast requires one exact Q branch")
        if (
            self.branch_lane_count != 1
            or self.nonrequested_branch_solver_bytes != 0
            or self.complete_hard_q_replay_source_retained is not True
            or self.schema
            != SIGNED_WORK_LEAST_WORK_CUDA_SINGLE_BRANCH_FORECAST_SCHEMA
        ):
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.AUTHORITY_VIOLATION,
                "single-branch forecast represented an excluded solver branch",
            )

    @property
    def forecast_hash(self) -> str:
        return content_hash(self, prefix=_SINGLE_BRANCH_FORECAST_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkLeastWorkCUDABranchReceipt:
    branch: V1RelaxedKKTBranch
    problem_hash: str
    policy_hash: str
    operator_receipt_hash: str
    converged: bool
    finite_iterates: bool
    iterations: int
    termination_reason: str
    resource_value: float
    resource_floor: float
    signed_work_upper: float
    quadratic_value: float
    maximum_box_violation: float
    maximum_block_violation: float
    normalized_primal_violation: float
    dual_work_lower: float
    absolute_duality_gap: float
    relative_duality_gap: float
    stationarity_gap: float
    normalized_stationarity_gap: float
    work_fenchel_gap: float
    resource_complementarity_gap: float
    quadratic_complementarity_gap: float
    block_complementarity_gap: float
    normalized_complementarity_gap: float
    zero_work_cell_count: int
    maximum_work_dual_box_violation: float
    primal_step_size: float
    dual_step_size: float
    scaled_operator_norm_squared_upper_bound: float
    device: str
    production_cuda_execution: bool
    private_cpu_reference_execution: bool
    loss_derived_tau_f_used: bool = False
    hard_mask_global_optimality_claimed: bool = False
    c_stream_authority: bool = False
    mask_commit_authority: bool = False
    algorithm_id: str = SIGNED_WORK_LEAST_WORK_CUDA_ALGORITHM_ID
    authority_scope: str = SIGNED_WORK_LEAST_WORK_CUDA_AUTHORITY
    schema: str = SIGNED_WORK_LEAST_WORK_CUDA_BRANCH_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("least-work branch receipt requires an exact Q branch")
        for name in ("problem_hash", "policy_hash", "operator_receipt_hash"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "converged",
            "finite_iterates",
            "production_cuda_execution",
            "private_cpu_reference_execution",
            "loss_derived_tau_f_used",
            "hard_mask_global_optimality_claimed",
            "c_stream_authority",
            "mask_commit_authority",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact bool")
        if self.production_cuda_execution == self.private_cpu_reference_execution:
            raise ValueError("exactly one branch execution mode must be active")
        object.__setattr__(
            self,
            "iterations",
            _nonnegative_int(self.iterations, name="iterations"),
        )
        object.__setattr__(
            self,
            "termination_reason",
            _identity(self.termination_reason, name="termination reason"),
        )
        object.__setattr__(
            self,
            "resource_floor",
            _positive(self.resource_floor, name="resource floor"),
        )
        object.__setattr__(
            self,
            "resource_value",
            _nonnegative(self.resource_value, name="resource value"),
        )
        object.__setattr__(
            self,
            "zero_work_cell_count",
            _nonnegative_int(self.zero_work_cell_count, name="zero-work cell count"),
        )
        for name in (
            "primal_step_size",
            "dual_step_size",
            "scaled_operator_norm_squared_upper_bound",
        ):
            object.__setattr__(
                self,
                name,
                _positive(getattr(self, name), name=name),
            )
        stability_product = (
            self.primal_step_size
            * self.dual_step_size
            * self.scaled_operator_norm_squared_upper_bound
        )
        if not math.isfinite(stability_product) or stability_product >= 1.0:
            raise ValueError("PDHG receipt violates its strict operator boundary")
        object.__setattr__(self, "device", _identity(self.device, name="device"))
        if self.production_cuda_execution and not self.device.startswith("cuda"):
            raise ValueError("production branch receipt does not name a CUDA device")
        if self.private_cpu_reference_execution and not self.device.startswith("cpu"):
            raise ValueError("private reference branch receipt does not name a CPU device")

        optimality_names = (
            "signed_work_upper",
            "quadratic_value",
            "maximum_box_violation",
            "maximum_block_violation",
            "normalized_primal_violation",
            "dual_work_lower",
            "absolute_duality_gap",
            "relative_duality_gap",
            "stationarity_gap",
            "normalized_stationarity_gap",
            "work_fenchel_gap",
            "resource_complementarity_gap",
            "quadratic_complementarity_gap",
            "block_complementarity_gap",
            "normalized_complementarity_gap",
            "maximum_work_dual_box_violation",
        )
        for name in optimality_names:
            object.__setattr__(
                self,
                name,
                _extended_real(getattr(self, name), name=name),
            )
        finite_optimality = all(
            math.isfinite(getattr(self, name)) for name in optimality_names
        )
        if finite_optimality:
            nonnegative_names = tuple(
                name for name in optimality_names if name != "dual_work_lower"
            )
            if any(getattr(self, name) < 0.0 for name in nonnegative_names):
                raise ValueError("branch optimality diagnostics must be nonnegative")
            denominator = max(
                1.0,
                abs(self.signed_work_upper),
                abs(self.dual_work_lower),
            )
            expected_absolute_gap = max(
                0.0,
                self.signed_work_upper - self.dual_work_lower,
            )
            expected_relative_gap = expected_absolute_gap / denominator
            expected_stationarity = self.stationarity_gap / denominator
            complementarity = (
                self.work_fenchel_gap
                + self.resource_complementarity_gap
                + self.quadratic_complementarity_gap
                + self.block_complementarity_gap
            )
            expected_complementarity = complementarity / denominator
            for observed, expected, owner in (
                (self.absolute_duality_gap, expected_absolute_gap, "absolute gap"),
                (self.relative_duality_gap, expected_relative_gap, "relative gap"),
                (
                    self.normalized_stationarity_gap,
                    expected_stationarity,
                    "normalized stationarity gap",
                ),
                (
                    self.normalized_complementarity_gap,
                    expected_complementarity,
                    "normalized complementarity gap",
                ),
            ):
                if not _receipt_values_agree(observed, expected):
                    raise ValueError(f"{owner} is inconsistent with its raw receipt fields")
            resource_shortfall = max(
                0.0,
                self.resource_floor - self.resource_value,
            ) / max(1.0, self.resource_floor)
            if self.normalized_primal_violation + 512.0 * _EPS < resource_shortfall:
                raise ValueError("primal violation understates the resource shortfall")
        else:
            positive_infinity_names = tuple(
                name for name in optimality_names if name != "dual_work_lower"
            )
            exact_unresolved_sentinel = (
                self.signed_work_upper == math.inf
                and self.quadratic_value == math.inf
                and self.dual_work_lower == -math.inf
                and all(
                    getattr(self, name) == math.inf
                    for name in positive_infinity_names
                    if name not in ("signed_work_upper", "quadratic_value")
                )
            )
            if not exact_unresolved_sentinel:
                _refuse(
                    V1SignedWorkLeastWorkCUDARefusalCode.NONFINITE,
                    "branch optimality fields must be finite or the exact unresolved sentinel",
                )
        if self.converged and (not self.finite_iterates or not finite_optimality):
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.NONFINITE,
                "a converged branch requires finite iterates and finite optimality fields",
            )
        if self.converged and self.iterations == 0:
            raise ValueError("a converged branch requires a completed solver iteration")
        if (
            self.loss_derived_tau_f_used
            or self.hard_mask_global_optimality_claimed
            or self.c_stream_authority
            or self.mask_commit_authority
            or self.algorithm_id != SIGNED_WORK_LEAST_WORK_CUDA_ALGORITHM_ID
            or self.authority_scope != SIGNED_WORK_LEAST_WORK_CUDA_AUTHORITY
            or self.schema != SIGNED_WORK_LEAST_WORK_CUDA_BRANCH_RECEIPT_SCHEMA
        ):
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.AUTHORITY_VIOLATION,
                "least-work branch receipt crossed proposal-only authority",
            )

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_BRANCH_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkLeastWorkCUDAResult:
    problem_hash: str
    policy_hash: str
    forecast_hash: str
    branch_receipts: tuple[V1SignedWorkLeastWorkCUDABranchReceipt, ...]
    selected_branch: V1RelaxedKKTBranch | None
    branch_selection_certified_within_tolerance: bool
    relaxed_state_available: bool
    state_identity_hash: str | None
    all_branches_converged: bool
    relaxed_state_handle: Any | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    product_branch_union_is_not_one_convex_problem: bool = True
    deterministic_rounding_still_required: bool = True
    hard_mask_global_optimality_claimed: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_claim_authority: bool = False
    authority_scope: str = SIGNED_WORK_LEAST_WORK_CUDA_AUTHORITY
    schema: str = SIGNED_WORK_LEAST_WORK_CUDA_RESULT_SCHEMA

    def __post_init__(self) -> None:
        for name in ("problem_hash", "policy_hash", "forecast_hash"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        receipts = tuple(self.branch_receipts)
        if any(
            type(receipt) is not V1SignedWorkLeastWorkCUDABranchReceipt
            for receipt in receipts
        ):
            raise TypeError("least-work result requires exact branch receipts")
        if (
            tuple(receipt.branch for receipt in receipts)
            != SIGNED_WORK_LEAST_WORK_BRANCH_ORDER
        ):
            raise ValueError("least-work result does not contain the frozen two-branch family")
        if any(
            receipt.problem_hash != self.problem_hash
            or receipt.policy_hash != self.policy_hash
            for receipt in receipts
        ):
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.STALE_SOURCE,
                "least-work branch receipts do not share the result source",
            )
        if len({receipt.operator_receipt_hash for receipt in receipts}) != 1:
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.STALE_SOURCE,
                "least-work branch receipts do not share one signed-F operator",
            )
        if any(
            (
                receipt.production_cuda_execution,
                receipt.private_cpu_reference_execution,
            )
            != (
                receipts[0].production_cuda_execution,
                receipts[0].private_cpu_reference_execution,
            )
            for receipt in receipts[1:]
        ):
            raise ValueError("least-work branch receipts mix execution modes")
        bool_names = (
            "branch_selection_certified_within_tolerance",
            "relaxed_state_available",
            "all_branches_converged",
            "product_branch_union_is_not_one_convex_problem",
            "deterministic_rounding_still_required",
            "hard_mask_global_optimality_claimed",
            "c_stream_authority",
            "calibration_authority",
            "bce_oracle_authority",
            "mask_commit_authority",
            "terminal_authority",
            "pareto_claim_authority",
        )
        for name in bool_names:
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact bool")
        observed_all_converged = all(receipt.converged for receipt in receipts)
        if self.all_branches_converged is not observed_all_converged:
            raise ValueError("aggregate convergence differs from branch receipts")
        if self.selected_branch is not None and (
            type(self.selected_branch) is not V1RelaxedKKTBranch
        ):
            raise TypeError("selected_branch must be an exact registered Q branch or None")
        if self.branch_selection_certified_within_tolerance is not (
            self.selected_branch is not None
        ):
            raise ValueError("branch-selection certification differs from branch selection")
        if self.selected_branch is not None:
            if not self.all_branches_converged:
                raise ValueError("a selected branch requires convergence of both branches")
            selected_receipt = next(
                receipt for receipt in receipts if receipt.branch is self.selected_branch
            )
            smallest_primal = min(receipt.signed_work_upper for receipt in receipts)
            if selected_receipt.signed_work_upper != smallest_primal:
                raise ValueError("selected branch is not a least-work primal minimizer")
        if self.relaxed_state_available is not (
            self.branch_selection_certified_within_tolerance
            and self.all_branches_converged
        ):
            raise ValueError("relaxed-state availability differs from certified closure")
        if self.relaxed_state_available:
            object.__setattr__(
                self,
                "state_identity_hash",
                _identity(self.state_identity_hash, name="state identity hash"),
            )
        elif self.state_identity_hash is not None:
            raise ValueError("unavailable relaxed state cannot carry an identity")
        if self.relaxed_state_handle is not None:
            if not self.relaxed_state_available or (
                getattr(self.relaxed_state_handle, "identity_hash", None)
                != self.state_identity_hash
            ):
                raise ValueError("embedded relaxed-state handle differs from result identity")
        if (
            self.product_branch_union_is_not_one_convex_problem is not True
            or self.deterministic_rounding_still_required is not True
            or self.hard_mask_global_optimality_claimed
            or self.c_stream_authority
            or self.calibration_authority
            or self.bce_oracle_authority
            or self.mask_commit_authority
            or self.terminal_authority
            or self.pareto_claim_authority
            or self.authority_scope != SIGNED_WORK_LEAST_WORK_CUDA_AUTHORITY
            or self.schema != SIGNED_WORK_LEAST_WORK_CUDA_RESULT_SCHEMA
        ):
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.AUTHORITY_VIOLATION,
                "least-work result crossed proposal-only authority",
            )
        object.__setattr__(self, "branch_receipts", receipts)

    @property
    def result_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "problem_hash": self.problem_hash,
                "policy_hash": self.policy_hash,
                "forecast_hash": self.forecast_hash,
                "branch_receipts": self.branch_receipts,
                "selected_branch": self.selected_branch,
                "branch_selection_certified_within_tolerance": (
                    self.branch_selection_certified_within_tolerance
                ),
                "relaxed_state_available": self.relaxed_state_available,
                "state_identity_hash": self.state_identity_hash,
                "all_branches_converged": self.all_branches_converged,
                "product_branch_union_is_not_one_convex_problem": (
                    self.product_branch_union_is_not_one_convex_problem
                ),
                "deterministic_rounding_still_required": (
                    self.deterministic_rounding_still_required
                ),
                "hard_mask_global_optimality_claimed": (
                    self.hard_mask_global_optimality_claimed
                ),
                "c_stream_authority": self.c_stream_authority,
                "calibration_authority": self.calibration_authority,
                "bce_oracle_authority": self.bce_oracle_authority,
                "mask_commit_authority": self.mask_commit_authority,
                "terminal_authority": self.terminal_authority,
                "pareto_claim_authority": self.pareto_claim_authority,
                "authority_scope": self.authority_scope,
            },
            prefix=_RESULT_PREFIX,
        )


def _stable_l2_norm(value: np.ndarray, *, chunk_size: int = 1 << 20) -> float:
    """Compute a finite route-vector norm with bounded host workspace."""

    flat = np.asarray(value, dtype=np.float64).reshape(-1)
    scale = 0.0
    scaled_sum = 1.0
    for start in range(0, int(flat.size), chunk_size):
        chunk = np.abs(flat[start : start + chunk_size])
        if not chunk.size:
            continue
        chunk_scale = float(np.max(chunk))
        if chunk_scale == 0.0:
            continue
        normalized = chunk / chunk_scale
        chunk_sum = float(np.dot(normalized, normalized))
        if scale < chunk_scale:
            scaled_sum = scaled_sum * (scale / chunk_scale) ** 2 + chunk_sum
            scale = chunk_scale
        else:
            scaled_sum += chunk_sum * (chunk_scale / scale) ** 2
    result = scale * math.sqrt(scaled_sum) if scale > 0.0 else 0.0
    if not math.isfinite(result):
        _refuse(
            V1SignedWorkLeastWorkCUDARefusalCode.NONFINITE,
            "route-vector norm is not representable in float64",
        )
    return result


def _operator_device_object(operator: V1SignedWorkOperatorCUDA):
    operator.receipt.assert_integrity()
    device = getattr(operator, "_device_obj", None)
    if device is None:
        _refuse(
            V1SignedWorkLeastWorkCUDARefusalCode.STALE_SOURCE,
            "signed-work operator has no live device custody",
        )
    return device


def build_v1_signed_work_least_work_cuda_hbm_forecast(
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    *,
    device_total_memory_bytes: int | None = None,
    current_reserved_bytes: int | None = None,
) -> V1SignedWorkLeastWorkCUDAHBMForecast:
    """Issue the allocation-free solver HBM forecast.

    Operator custody already exists when this forecast is issued, so its
    factor bytes are represented by ``current_reserved_bytes`` rather than
    double-counted as a new allocation.  The incremental envelope covers the
    Q/resource source, both fused branch states, retained theorem state, work
    dual cells, and the largest simultaneous operator workspace.
    """

    if type(problem) is not V1SignedWorkLeastWorkCUDAProblem:
        raise TypeError("forecast requires the exact least-work CUDA problem")
    if type(policy) is not V1SignedWorkLeastWorkCUDAPolicy:
        raise TypeError("forecast requires the exact least-work CUDA policy")
    problem.assert_integrity()
    base = problem.source_problem
    operator = problem.signed_work_operator
    device = _operator_device_object(operator)
    route_count = base.route_count
    rank = base.rank
    contexts = operator.context_count
    fibers = operator.fiber_count
    blocks = len(base.block_capacities)

    # Resource, singleton-work tie breaker, H, residual diagonal and its
    # precomputed square root, eligible, and required coordinates.  Both
    # residual tensors coexist throughout the fused solve; omitting the square
    # root would understate HBM by one FP64 route vector.
    persistent_source = (
        (4 * route_count + rank * route_count) * _F64_BYTES
        + 2 * route_count * _BOOL_BYTES
    )
    # z, z_bar, ergodic average, best, reduced advantage, and retained best;
    # every route object has two branch lanes.  One spectral response-tail
    # dual is route sized.  Low-rank, scalar, and block duals are explicit.
    solver_state = (
        6 * route_count * 2 * _F64_BYTES
        + route_count * _F64_BYTES
        + (rank * 2 + 2 + blocks * 2) * _F64_BYTES
    )
    work_dual = contexts * fibers * 2 * _F64_BYTES
    # Candidate batching uses four lanes at receipt boundaries.  A certificate
    # can retain route-scale adjoints, gradients, reduced advantages, masked
    # box/KKT scratch, and a newly evaluated iterate while the signed-work
    # operator owns its tile workspace.  These allocations therefore coexist;
    # taking ``max(operator, receipt)`` would understate peak HBM.  Sixteen
    # FP64 route lanes is a deliberately closed upper envelope for the fused
    # implementation's live certificate/iteration scratch.  It is kept as
    # arithmetic metadata rather than discovered by an outcome-conditioned
    # allocator probe.
    operator_forecast = operator.forecast
    certificate_route_scratch = 16 * route_count * _F64_BYTES
    certificate_cell_scratch = contexts * fibers * 4 * _F64_BYTES
    certificate_factor_scratch = rank * 4 * _F64_BYTES
    transient = (
        operator_forecast.temporary_tile_bytes_upper
        + certificate_route_scratch
        + certificate_cell_scratch
        + certificate_factor_scratch
    )
    incremental = persistent_source + solver_state + work_dual + transient

    live_admission = (
        getattr(device, "type", None) == "cuda"
        and (device_total_memory_bytes is None or current_reserved_bytes is None)
    )
    if device_total_memory_bytes is None or current_reserved_bytes is None:
        if getattr(device, "type", None) == "cuda":
            if torch is None or not torch.cuda.is_available():
                _refuse(
                    V1SignedWorkLeastWorkCUDARefusalCode.CUDA_REQUIRED,
                    "CUDA forecast requested without an available CUDA runtime",
                )
            if device_total_memory_bytes is None:
                device_total_memory_bytes = int(
                    torch.cuda.get_device_properties(device).total_memory
                )
            if current_reserved_bytes is None:
                current_reserved_bytes = int(torch.cuda.memory_reserved(device))
        else:
            if device_total_memory_bytes is None:
                device_total_memory_bytes = max(1 << 30, incremental * 4)
            if current_reserved_bytes is None:
                current_reserved_bytes = 0
    total = _positive_int(device_total_memory_bytes, name="device total memory")
    current = _nonnegative_int(current_reserved_bytes, name="current reserved bytes")
    ceiling = int(math.floor(policy.hbm_ceiling_fraction * total))
    peak = current + incremental
    return V1SignedWorkLeastWorkCUDAHBMForecast(
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        operator_forecast_hash=operator_forecast.forecast_hash,
        device=str(device),
        route_count=route_count,
        response_rank=rank,
        context_count=contexts,
        fiber_count=fibers,
        block_count=blocks,
        persistent_source_bytes=persistent_source,
        solver_state_bytes=solver_state,
        work_dual_bytes=work_dual,
        transient_workspace_bytes=transient,
        forecast_incremental_bytes=incremental,
        current_reserved_bytes=current,
        forecast_peak_reserved_bytes=peak,
        device_total_memory_bytes=total,
        hbm_ceiling_bytes=ceiling,
        admitted=(
            int(torch.cuda.memory_allocated(device)) + incremental <= ceiling
            and incremental <= cuda_reusable_memory_bytes(device, framework=torch)
        ) if live_admission else peak <= ceiling,
    )


def build_v1_signed_work_least_work_single_branch_cuda_hbm_forecast(
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    *,
    branch: V1RelaxedKKTBranch,
    device_total_memory_bytes: int | None = None,
    current_reserved_bytes: int | None = None,
) -> V1SignedWorkLeastWorkCUDASingleBranchHBMForecast:
    """Issue the allocation-free HBM forecast for one exact Q branch.

    This is a branch-membership forecast, not a discounted copy of the fused
    receipt.  Source tensors needed by later complete hard-Q replay remain in
    the bound, while all iterative state and signed-work duals have one lane.
    The transient certificate envelope deliberately remains the fused upper
    bound until a separately measured one-lane bound is promoted; this makes
    the admission reduction conservative rather than aspirational.
    """

    if type(problem) is not V1SignedWorkLeastWorkCUDAProblem:
        raise TypeError("forecast requires the exact least-work CUDA problem")
    if type(policy) is not V1SignedWorkLeastWorkCUDAPolicy:
        raise TypeError("forecast requires the exact least-work CUDA policy")
    if type(branch) is not V1RelaxedKKTBranch:
        raise TypeError("single-branch forecast requires one exact Q branch")
    problem.assert_integrity()
    base = problem.source_problem
    operator = problem.signed_work_operator
    device = _operator_device_object(operator)
    route_count = base.route_count
    rank = base.rank
    contexts = operator.context_count
    fibers = operator.fiber_count
    blocks = len(base.block_capacities)

    # Hard F/Q replay needs the factor plus both deterministic residual
    # representations even though the convex solver owns one Q branch.
    persistent_source = (
        (4 * route_count + rank * route_count) * _F64_BYTES
        + 2 * route_count * _BOOL_BYTES
    )
    # Six conservative route lanes cover the one-column primal/ergodic/best/
    # reduced state.  Only the spectral branch owns a route-sized tail dual;
    # the diagonal-L1 tail dual is scalar.  Low-rank, resource, and compact
    # block duals each have one lane.
    spectral_tail = (
        route_count if branch is V1RelaxedKKTBranch.SPECTRAL else 0
    )
    solver_state = (
        (6 * route_count + spectral_tail) * _F64_BYTES
        + (rank + 1 + blocks) * _F64_BYTES
    )
    work_dual = contexts * fibers * _F64_BYTES
    operator_forecast = operator.forecast
    transient = (
        operator_forecast.temporary_tile_bytes_upper
        + 16 * route_count * _F64_BYTES
        + contexts * fibers * 4 * _F64_BYTES
        + rank * 4 * _F64_BYTES
    )
    incremental = persistent_source + solver_state + work_dual + transient

    live_admission = (
        getattr(device, "type", None) == "cuda"
        and (device_total_memory_bytes is None or current_reserved_bytes is None)
    )
    if device_total_memory_bytes is None or current_reserved_bytes is None:
        if getattr(device, "type", None) == "cuda":
            if torch is None or not torch.cuda.is_available():
                _refuse(
                    V1SignedWorkLeastWorkCUDARefusalCode.CUDA_REQUIRED,
                    "CUDA forecast requested without an available CUDA runtime",
                )
            if device_total_memory_bytes is None:
                device_total_memory_bytes = int(
                    torch.cuda.get_device_properties(device).total_memory
                )
            if current_reserved_bytes is None:
                current_reserved_bytes = int(torch.cuda.memory_reserved(device))
        else:
            if device_total_memory_bytes is None:
                device_total_memory_bytes = max(1 << 30, incremental * 4)
            if current_reserved_bytes is None:
                current_reserved_bytes = 0
    total = _positive_int(device_total_memory_bytes, name="device total memory")
    current = _nonnegative_int(current_reserved_bytes, name="current reserved bytes")
    ceiling = int(math.floor(policy.hbm_ceiling_fraction * total))
    peak = current + incremental
    return V1SignedWorkLeastWorkCUDASingleBranchHBMForecast(
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        operator_forecast_hash=operator_forecast.forecast_hash,
        branch=branch,
        device=str(device),
        route_count=route_count,
        response_rank=rank,
        context_count=contexts,
        fiber_count=fibers,
        block_count=blocks,
        persistent_source_bytes=persistent_source,
        solver_state_bytes=solver_state,
        work_dual_bytes=work_dual,
        transient_workspace_bytes=transient,
        forecast_incremental_bytes=incremental,
        current_reserved_bytes=current,
        forecast_peak_reserved_bytes=peak,
        device_total_memory_bytes=total,
        hbm_ceiling_bytes=ceiling,
        admitted=(
            int(torch.cuda.memory_allocated(device)) + incremental <= ceiling
            and incremental <= cuda_reusable_memory_bytes(device, framework=torch)
        ) if live_admission else peak <= ceiling,
    )


class V1SignedWorkLeastWorkCUDAStateBorrowOperation(str, Enum):
    """Closed set of lawful process-local theorem-state borrows."""

    DEVICE_SHELL_ROUNDING = "device_shell_rounding"
    HARD_FQ_REPLAY = "hard_fq_replay"
    SIGNED_WORK_DECISION_WITNESS = "signed_work_decision_witness"
    BLOCK_SUPPORT_OUTSIDE_PAIR = "block_support_outside_pair"
    REFERENCE_INSPECTION = "reference_inspection"
    BINDING_CONSTRAINT_DIAGNOSTIC = "binding_constraint_diagnostic"


def _state_tensor_storage_invariant(value: Any) -> tuple[object, ...]:
    storage = value.untyped_storage()
    return (
        str(value.device),
        str(value.dtype),
        tuple(int(item) for item in value.shape),
        tuple(int(item) for item in value.stride()),
        int(value.storage_offset()),
        int(value.data_ptr()),
        int(storage.data_ptr()),
        int(value._version),
        bool(value.requires_grad),
        bool(value.is_contiguous()),
    )


def _exact_same_device_tensor(
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: Any,
    device: Any,
    name: str,
) -> Any:
    """Admit one already-resident tensor without conversion or copying."""

    if torch is None or not torch.is_tensor(value):
        _refuse(
            V1SignedWorkLeastWorkCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one exact device tensor",
        )
    scalar_broadcast = (
        value.ndim == 1
        and tuple(int(item) for item in value.stride()) == (0,)
        and int(value.storage_offset()) == 0
        and int(value.untyped_storage().nbytes()) <= int(value.element_size())
    )
    if (
        type(value) is not torch.Tensor
        or tuple(int(item) for item in value.shape) != shape
        or value.dtype != dtype
        or value.device != device
        or not (value.is_contiguous() or scalar_broadcast)
        or value.requires_grad
    ):
        _refuse(
            V1SignedWorkLeastWorkCUDARefusalCode.AXIS_MISMATCH,
            f"{name} differs from its registered shape/dtype/device custody",
        )
    return value


def _is_intrinsic_scalar_broadcast(
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: Any,
    device: Any,
    scalar_value: float | bool,
) -> bool:
    if torch is None or type(value) is not torch.Tensor:
        return False
    return bool(
        tuple(int(item) for item in value.shape) == shape
        and value.dtype == dtype
        and value.device == device
        and value.ndim == 1
        and tuple(int(item) for item in value.stride()) == (0,)
        and int(value.storage_offset()) == 0
        and int(value.untyped_storage().nbytes()) <= int(value.element_size())
        and value[0].item() == scalar_value
    )


class V1SignedWorkLeastWorkCUDADeviceProblemV2:
    """Live zero-copy sibling of :class:`V1SignedWorkLeastWorkCUDAProblem`.

    The portable problem remains the replay/reference boundary and therefore
    owns immutable NumPy ``A/H/d`` arrays.  This versioned production sibling
    instead borrows ``singleton_work``, the literal-J FD factor, and its
    residual diagonal directly from the row-free sampled-D device handle.
    The borrow is deliberately long lived: it is transferred into converged
    theorem-state custody, so retained ``H/d`` aliases can never outlive the
    source lease that makes them meaningful.

    Route-scale structural tensors are admitted only as exact same-device
    tensors with sealed source identities.  Canonical unit resource, all-
    eligible, and empty-required coordinates may be constructed intrinsically
    on the device.  No route tensor is converted to NumPy, hashed by content,
    or copied to the host.
    """

    __slots__ = (
        "problem_id",
        "signed_work_operator",
        "resource_floor",
        "quadratic_budget",
        "block_capacities",
        "source_registration_hash",
        "registered_d_coordinate_hash",
        "structural_source_receipt_hash",
        "structural_source_receipt_semantics",
        "source_bundle_authoritative",
        "route_axis_hash",
        "source_geometry_hash",
        "work_budget",
        "corrected_work_envelope_id",
        "schema",
        "_geometry_handle",
        "_geometry_lease",
        "_geometry_receipt_hash",
        "_resource",
        "_eligible",
        "_required",
        "_resource_identity_hash",
        "_eligible_mask_identity_hash",
        "_required_mask_identity_hash",
        "_resource_is_intrinsic_unit",
        "_eligible_is_intrinsic_all",
        "_required_is_intrinsic_empty",
        "_registered_legal_resource_maximum",
        "_registered_required_resource_value",
        "_has_ineligible",
        "_has_required",
        "_factor_frobenius_norm",
        "_residual_sqrt_l2_norm",
        "_residual_spectral_bound",
        "_route_count",
        "_rank",
        "_structural_invariants",
        "_geometry_invariants",
        "_problem_hash",
        "_lease_transferred",
        "_closed",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        problem_id: str,
        geometry_handle: V1SampledDeviceResidentDKKTGeometryDeviceHandle,
        signed_work_operator: V1SignedWorkOperatorCUDA,
        resource_floor: float,
        quadratic_budget: float,
        block_capacities: tuple[V1RelaxedKKTBlockCapacity, ...],
        source_registration_hash: str,
        registered_d_coordinate_hash: str,
        structural_source_receipt_hash: str,
        source_bundle_authoritative: bool,
        expected_geometry_receipt_hash: str,
        expected_session_design_hash: str,
        expected_row_free_finalization_receipt_hash: str,
        expected_materialization_rank_receipt_hash: str,
        expected_registered_proposal_fd_rank: int,
        expected_route_axis_hash: str,
        resource_tensor: Any | None = None,
        eligible_mask_tensor: Any | None = None,
        required_hard_action_tensor: Any | None = None,
        resource_tensor_identity_hash: str | None = None,
        eligible_mask_identity_hash: str | None = None,
        required_mask_identity_hash: str | None = None,
        registered_legal_resource_maximum: float | None = None,
        registered_required_resource_value: float | None = None,
        structural_source_receipt_semantics: str = (
            SIGNED_WORK_LEAST_WORK_CUDA_DEVICE_STRUCTURAL_RECEIPT_SEMANTICS_V2
        ),
    ) -> None:
        if torch is None:
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.CUDA_REQUIRED,
                "PyTorch is required for live device KKT custody",
            )
        if type(geometry_handle) is not V1SampledDeviceResidentDKKTGeometryDeviceHandle:
            raise TypeError("device problem requires the exact live D-geometry handle")
        if type(signed_work_operator) is not V1SignedWorkOperatorCUDA:
            raise TypeError("device problem requires the exact signed-work operator")
        if type(source_bundle_authoritative) is not bool:
            raise TypeError("source_bundle_authoritative must be an exact bool")
        signed_work_operator.receipt.assert_integrity()
        operator_device = _operator_device_object(signed_work_operator)
        route_axis_hash = _identity(
            expected_route_axis_hash,
            name="expected route axis hash",
        )
        lease = geometry_handle.borrow_for_kkt(
            expected_geometry_receipt_hash=expected_geometry_receipt_hash,
            expected_session_design_hash=expected_session_design_hash,
            expected_row_free_finalization_receipt_hash=(
                expected_row_free_finalization_receipt_hash
            ),
            expected_materialization_rank_receipt_hash=(
                expected_materialization_rank_receipt_hash
            ),
            expected_registered_proposal_fd_rank=(
                expected_registered_proposal_fd_rank
            ),
            expected_route_axis_hash=route_axis_hash,
            expected_device=operator_device,
        )
        try:
            receipt = lease.receipt
            routes = receipt.route_count
            rank = int(receipt.factor_shape[0])
            if (
                receipt.receipt_hash != expected_geometry_receipt_hash
                or receipt.route_axis_hash != route_axis_hash
                or signed_work_operator.route_count != routes
                # The signed-F operator lives on the physical block axis; it
                # joins the route-ID proposal plane through its ordered
                # route-ID identity, never through the physical axis hash.
                or signed_work_operator.receipt.route_id_axis_hash != route_axis_hash
                or receipt.device_identity != str(operator_device)
            ):
                _refuse(
                    V1SignedWorkLeastWorkCUDARefusalCode.STALE_SOURCE,
                    "signed-F and literal-J device sources do not share one route chart",
                )
            singleton = lease.singleton_work
            factor = lease.factor
            residual = lease.residual_diagonal
            device = singleton.device

            resource_intrinsic = (
                resource_tensor is None
                or _is_intrinsic_scalar_broadcast(
                    resource_tensor,
                    shape=(routes,),
                    dtype=torch.float64,
                    device=device,
                    scalar_value=1.0,
                )
            )
            eligible_intrinsic = (
                eligible_mask_tensor is None
                or _is_intrinsic_scalar_broadcast(
                    eligible_mask_tensor,
                    shape=(routes,),
                    dtype=torch.bool,
                    device=device,
                    scalar_value=True,
                )
            )
            required_intrinsic = (
                required_hard_action_tensor is None
                or _is_intrinsic_scalar_broadcast(
                    required_hard_action_tensor,
                    shape=(routes,),
                    dtype=torch.bool,
                    device=device,
                    scalar_value=False,
                )
            )
            resource = (
                torch.ones((), dtype=torch.float64, device=device).expand(routes)
                if resource_tensor is None
                else _exact_same_device_tensor(
                    resource_tensor,
                    shape=(routes,),
                    dtype=torch.float64,
                    device=device,
                    name="resource tensor",
                )
            )
            eligible = (
                torch.ones((), dtype=torch.bool, device=device).expand(routes)
                if eligible_mask_tensor is None
                else _exact_same_device_tensor(
                    eligible_mask_tensor,
                    shape=(routes,),
                    dtype=torch.bool,
                    device=device,
                    name="eligible mask tensor",
                )
            )
            required = (
                torch.zeros((), dtype=torch.bool, device=device).expand(routes)
                if required_hard_action_tensor is None
                else _exact_same_device_tensor(
                    required_hard_action_tensor,
                    shape=(routes,),
                    dtype=torch.bool,
                    device=device,
                    name="required hard-action tensor",
                )
            )
            structural_semantics = _identity(
                structural_source_receipt_semantics,
                name="structural source receipt semantics",
            )
            if (
                structural_semantics
                != SIGNED_WORK_LEAST_WORK_CUDA_DEVICE_STRUCTURAL_RECEIPT_SEMANTICS_V2
            ):
                _refuse(
                    V1SignedWorkLeastWorkCUDARefusalCode.AUTHORITY_VIOLATION,
                    "device structural receipt does not certify the required semantics",
                )

            blocks = tuple(block_capacities)
            if any(type(item) is not V1RelaxedKKTBlockCapacity for item in blocks):
                raise TypeError("device problem requires exact compact block capacities")
            if blocks:
                previous = 0
                for block in blocks:
                    if block.start != previous or block.stop > routes:
                        _refuse(
                            V1SignedWorkLeastWorkCUDARefusalCode.AXIS_MISMATCH,
                            "device compact blocks do not form one ordered route cover",
                        )
                    previous = block.stop
                if previous != routes or len({block.block_id for block in blocks}) != len(blocks):
                    _refuse(
                        V1SignedWorkLeastWorkCUDARefusalCode.AXIS_MISMATCH,
                        "device compact blocks must cover the route axis exactly once",
                    )

            floor = _positive(resource_floor, name="resource floor")
            q_budget = _nonnegative(quadratic_budget, name="quadratic budget")
            if resource_intrinsic:
                required_resource = (
                    0.0
                    if required_intrinsic
                    else float(torch.count_nonzero(required).item())
                )
                maximum_resource = 0.0
                if blocks:
                    for block in blocks:
                        eligible_count = (
                            block.stop - block.start
                            if eligible_intrinsic
                            else int(
                                torch.count_nonzero(
                                    eligible[block.start : block.stop]
                                ).item()
                            )
                        )
                        capacity = min(block.removal_capacity, eligible_count)
                        required_count = (
                            0
                            if required_intrinsic
                            else int(
                                torch.count_nonzero(
                                    required[block.start : block.stop]
                                ).item()
                            )
                        )
                        if required_count > block.removal_capacity:
                            _refuse(
                                V1SignedWorkLeastWorkCUDARefusalCode.INFEASIBLE_RESOURCE,
                                "required hard action exceeds a live block capacity",
                            )
                        maximum_resource += float(capacity)
                else:
                    maximum_resource = float(
                        routes
                        if eligible_intrinsic
                        else torch.count_nonzero(eligible).item()
                    )
            else:
                if (
                    registered_legal_resource_maximum is None
                    or registered_required_resource_value is None
                ):
                    _refuse(
                        V1SignedWorkLeastWorkCUDARefusalCode.AUTHORITY_VIOLATION,
                        "nonintrinsic resource requires pre-outcome registered scalar bounds",
                    )
                maximum_resource = _nonnegative(
                    registered_legal_resource_maximum,
                    name="registered legal resource maximum",
                )
                required_resource = _nonnegative(
                    registered_required_resource_value,
                    name="registered required resource value",
                )
                if required_resource > maximum_resource:
                    _refuse(
                        V1SignedWorkLeastWorkCUDARefusalCode.INFEASIBLE_RESOURCE,
                        "registered required resource exceeds the registered legal maximum",
                    )
            tolerance = 128.0 * _EPS * max(1.0, floor, maximum_resource)
            if floor > maximum_resource + tolerance:
                _refuse(
                    V1SignedWorkLeastWorkCUDARefusalCode.INFEASIBLE_RESOURCE,
                    "resource floor exceeds the device-registered legal maximum",
                )
            if required_resource > floor + tolerance:
                _refuse(
                    V1SignedWorkLeastWorkCUDARefusalCode.INFEASIBLE_RESOURCE,
                    "required hard rung exceeds the next resource coordinate",
                )

            structural_hash = _identity(
                structural_source_receipt_hash,
                name="structural source receipt hash",
            )
            resource_identity = (
                content_hash(
                    {
                        "kind": "intrinsic_unit_resource_v1",
                        "route_axis_hash": route_axis_hash,
                        "route_count": routes,
                        "structural_source_receipt_hash": structural_hash,
                    },
                    prefix=_DEVICE_PROBLEM_PREFIX_V2 + "resource-",
                )
                if resource_intrinsic
                else _identity(
                    resource_tensor_identity_hash,
                    name="resource tensor identity hash",
                )
            )
            eligible_identity = (
                content_hash(
                    {
                        "kind": "intrinsic_all_eligible_v1",
                        "route_axis_hash": route_axis_hash,
                        "route_count": routes,
                        "structural_source_receipt_hash": structural_hash,
                    },
                    prefix=SIGNED_WORK_LEAST_WORK_ELIGIBLE_MASK_IDENTITY_PREFIX,
                )
                if eligible_intrinsic
                else _identity(
                    eligible_mask_identity_hash,
                    name="eligible mask identity hash",
                )
            )
            required_identity = (
                content_hash(
                    {
                        "kind": "intrinsic_empty_required_v1",
                        "route_axis_hash": route_axis_hash,
                        "route_count": routes,
                        "structural_source_receipt_hash": structural_hash,
                    },
                    prefix=SIGNED_WORK_LEAST_WORK_REQUIRED_MASK_IDENTITY_PREFIX,
                )
                if required_intrinsic
                else _identity(
                    required_mask_identity_hash,
                    name="required mask identity hash",
                )
            )

            self.problem_id = _identity(problem_id, name="device problem ID")
            self.signed_work_operator = signed_work_operator
            self.resource_floor = floor
            self.quadratic_budget = q_budget
            self.block_capacities = blocks
            self.source_registration_hash = _identity(
                source_registration_hash,
                name="source registration hash",
            )
            self.registered_d_coordinate_hash = _identity(
                registered_d_coordinate_hash,
                name="registered D coordinate hash",
            )
            self.structural_source_receipt_hash = structural_hash
            self.structural_source_receipt_semantics = structural_semantics
            self.source_bundle_authoritative = source_bundle_authoritative
            self.route_axis_hash = route_axis_hash
            self.source_geometry_hash = signed_work_operator.receipt.source_geometry_hash
            self.work_budget = 0.0
            self.corrected_work_envelope_id = (
                "live_sampled_D_singleton_triangle_tie_breaker_only_v2"
            )
            self.schema = SIGNED_WORK_LEAST_WORK_CUDA_DEVICE_PROBLEM_SCHEMA_V2
            self._geometry_handle = geometry_handle
            self._geometry_lease = lease
            self._geometry_receipt_hash = receipt.receipt_hash
            self._resource = resource
            self._eligible = eligible
            self._required = required
            self._resource_identity_hash = resource_identity
            self._eligible_mask_identity_hash = eligible_identity
            self._required_mask_identity_hash = required_identity
            self._resource_is_intrinsic_unit = resource_intrinsic
            self._eligible_is_intrinsic_all = eligible_intrinsic
            self._required_is_intrinsic_empty = required_intrinsic
            self._registered_legal_resource_maximum = maximum_resource
            self._registered_required_resource_value = required_resource
            # Do not form ``~eligible`` here: even for the intrinsic scalar-
            # broadcast view that expression materializes an unforecasted
            # route-length Boolean temporary.  The intrinsic coordinates are
            # known exactly from their issuer; nonintrinsic coordinates need
            # only a scalar reduction over their already-resident storage.
            self._has_ineligible = (
                False
                if eligible_intrinsic
                else not bool(torch.all(eligible).item())
            )
            self._has_required = (
                False
                if required_intrinsic
                else bool(torch.any(required).item())
            )
            self._factor_frobenius_norm = receipt.factor_frobenius_norm
            self._residual_sqrt_l2_norm = receipt.residual_sqrt_l2_norm
            self._residual_spectral_bound = receipt.residual_spectral_bound
            self._route_count = routes
            self._rank = rank
            self._structural_invariants = tuple(
                _state_tensor_storage_invariant(item)
                for item in (resource, eligible, required)
            )
            self._geometry_invariants = tuple(
                _state_tensor_storage_invariant(item)
                for item in (singleton, factor, residual)
            )
            self._lease_transferred = False
            self._closed = False
            self._problem_hash = content_hash(
                {
                    "schema": self.schema,
                    "problem_id": self.problem_id,
                    "geometry_receipt_hash": self._geometry_receipt_hash,
                    "signed_work_operator_receipt_hash": (
                        signed_work_operator.receipt_hash
                    ),
                    "resource_floor": self.resource_floor,
                    "quadratic_budget": self.quadratic_budget,
                    "block_capacities": self.block_capacities,
                    "source_registration_hash": self.source_registration_hash,
                    "registered_d_coordinate_hash": (
                        self.registered_d_coordinate_hash
                    ),
                    "structural_source_receipt_hash": structural_hash,
                    "structural_source_receipt_semantics": structural_semantics,
                    "registered_legal_resource_maximum": maximum_resource,
                    "registered_required_resource_value": required_resource,
                    "source_bundle_authoritative": source_bundle_authoritative,
                    "route_axis_hash": route_axis_hash,
                    "resource_tensor_identity_hash": resource_identity,
                    "eligible_mask_identity_hash": eligible_identity,
                    "required_mask_identity_hash": required_identity,
                    "resource_is_intrinsic_unit": resource_intrinsic,
                    "eligible_is_intrinsic_all": eligible_intrinsic,
                    "required_is_intrinsic_empty": required_intrinsic,
                    "route_scale_d2h_bytes": 0,
                    "route_scale_numpy_arrays_constructed": 0,
                    "proposal_authority_only": True,
                },
                prefix=_DEVICE_PROBLEM_PREFIX_V2,
            )
            self.assert_integrity()
        except BaseException:
            lease.close()
            raise

    @property
    def source_problem(self) -> "V1SignedWorkLeastWorkCUDADeviceProblemV2":
        return self

    @property
    def route_count(self) -> int:
        return self._route_count

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def resource(self) -> Any:
        self.assert_integrity()
        return self._resource

    @property
    def corrected_work(self) -> Any:
        return self._geometry_tensor("singleton_work")

    @property
    def low_rank_factor(self) -> Any:
        return self._geometry_tensor("factor")

    @property
    def residual_diagonal(self) -> Any:
        return self._geometry_tensor("residual_diagonal")

    @property
    def residual_spectral_bound(self) -> float:
        return self._residual_spectral_bound

    @property
    def eligible_mask(self) -> Any:
        self.assert_integrity()
        return self._eligible

    @property
    def required_hard_action(self) -> Any:
        self.assert_integrity()
        return self._required

    @property
    def has_ineligible(self) -> bool:
        self.assert_integrity()
        return self._has_ineligible

    @property
    def has_required(self) -> bool:
        self.assert_integrity()
        return self._has_required

    @property
    def problem_hash(self) -> str:
        # The serializable identity remains usable after exclusive lease
        # transfer; route-tensor reads still fail closed through the properties.
        return self._problem_hash

    @property
    def geometry_receipt_hash(self) -> str:
        return self._geometry_receipt_hash

    @property
    def eligible_mask_identity_hash(self) -> str:
        return self._eligible_mask_identity_hash

    @property
    def required_mask_identity_hash(self) -> str:
        return self._required_mask_identity_hash

    @property
    def resource_tensor_identity_hash(self) -> str:
        return self._resource_identity_hash

    @property
    def registered_legal_resource_maximum(self) -> float:
        return self._registered_legal_resource_maximum

    @property
    def registered_required_resource_value(self) -> float:
        return self._registered_required_resource_value

    @property
    def route_scale_d2h_bytes(self) -> int:
        return 0

    @property
    def route_scale_numpy_arrays_constructed(self) -> int:
        return 0

    def _geometry_tensor(self, role: str) -> Any:
        self.assert_integrity()
        lease = self._geometry_lease
        if lease is None:
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.HANDLE_EVICTED,
                "device problem no longer owns its D-geometry lease",
            )
        if role == "singleton_work":
            return lease.singleton_work
        if role == "factor":
            return lease.factor
        if role == "residual_diagonal":
            return lease.residual_diagonal
        raise ValueError("unknown device KKT geometry tensor role")

    def assert_integrity(self) -> str:
        if self._closed or self._lease_transferred or self._geometry_lease is None:
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.HANDLE_EVICTED,
                "device problem has closed or transferred its geometry lease",
            )
        self.signed_work_operator.receipt.assert_integrity()
        receipt = self._geometry_lease.receipt
        if receipt.receipt_hash != self._geometry_receipt_hash:
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.STALE_SOURCE,
                "device problem geometry receipt changed",
            )
        structural = tuple(
            _state_tensor_storage_invariant(item)
            for item in (self._resource, self._eligible, self._required)
        )
        geometry = tuple(
            _state_tensor_storage_invariant(item)
            for item in (
                self._geometry_lease.singleton_work,
                self._geometry_lease.factor,
                self._geometry_lease.residual_diagonal,
            )
        )
        if (
            structural != self._structural_invariants
            or geometry != self._geometry_invariants
        ):
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.CUSTODY_MUTATED,
                "device KKT source storage or tensor version changed",
            )
        return self._problem_hash

    def _take_geometry_lease_for_state(
        self,
    ) -> V1SampledDeviceResidentDKKTGeometryLease:
        self.assert_integrity()
        lease = self._geometry_lease
        if lease is None:  # pragma: no cover - closed by assert_integrity.
            raise RuntimeError("device geometry lease disappeared")
        self._geometry_lease = None
        self._lease_transferred = True
        return lease

    def spawn_resource_floor_sibling_v2(
        self,
        *,
        problem_id: str,
        resource_floor: float,
    ) -> "V1SignedWorkLeastWorkCUDADeviceProblemV2":
        """Issue another floor on the identical live source without copying.

        Phase A may discover an endogenous resource coordinate before the
        signed-F solve.  This constructor preserves the exact A/H/d pointers,
        structural tensors, compact blocks, Q budget, and source identities
        while changing only the registered floor and problem ID.  The sibling
        owns an independent handle lease and must itself be closed or transfer
        that lease to exactly one theorem-state owner.
        """

        self.assert_integrity()
        lease = self._geometry_lease
        if lease is None:  # pragma: no cover - closed by assert_integrity.
            raise RuntimeError("device geometry lease disappeared")
        receipt = lease.receipt
        return V1SignedWorkLeastWorkCUDADeviceProblemV2(
            problem_id=problem_id,
            geometry_handle=self._geometry_handle,
            signed_work_operator=self.signed_work_operator,
            resource_floor=resource_floor,
            quadratic_budget=self.quadratic_budget,
            block_capacities=self.block_capacities,
            source_registration_hash=self.source_registration_hash,
            registered_d_coordinate_hash=self.registered_d_coordinate_hash,
            structural_source_receipt_hash=self.structural_source_receipt_hash,
            source_bundle_authoritative=self.source_bundle_authoritative,
            expected_geometry_receipt_hash=receipt.receipt_hash,
            expected_session_design_hash=receipt.session_design_hash,
            expected_row_free_finalization_receipt_hash=(
                receipt.row_free_finalization_receipt_hash
            ),
            expected_materialization_rank_receipt_hash=(
                receipt.materialization_rank_receipt_hash
            ),
            expected_registered_proposal_fd_rank=(
                receipt.registered_proposal_fd_rank
            ),
            expected_route_axis_hash=self.route_axis_hash,
            resource_tensor=self._resource,
            eligible_mask_tensor=self._eligible,
            required_hard_action_tensor=self._required,
            resource_tensor_identity_hash=self._resource_identity_hash,
            eligible_mask_identity_hash=self._eligible_mask_identity_hash,
            required_mask_identity_hash=self._required_mask_identity_hash,
            registered_legal_resource_maximum=(
                None
                if self._resource_is_intrinsic_unit
                else self._registered_legal_resource_maximum
            ),
            registered_required_resource_value=(
                None
                if self._resource_is_intrinsic_unit
                else self._registered_required_resource_value
            ),
            structural_source_receipt_semantics=(
                self.structural_source_receipt_semantics
            ),
        )

    def close(self) -> None:
        if self._closed or self._lease_transferred:
            return
        lease = self._geometry_lease
        self._geometry_lease = None
        self._closed = True
        if lease is not None:
            lease.close()

    def __enter__(self) -> "V1SignedWorkLeastWorkCUDADeviceProblemV2":
        self.assert_integrity()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __reduce__(self) -> NoReturn:
        raise TypeError("device least-work problems are nonserializable")

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("device least-work problems are nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("device least-work problems cannot be copied")

    def __deepcopy__(self, _memo: dict[int, object]) -> NoReturn:
        raise TypeError("device least-work problems cannot be copied")


@dataclass(frozen=True, slots=True)
class V1SignedWorkLeastWorkCUDADeviceHBMForecastV2:
    """Incremental HBM bound with live A/H/d and row-bank custody excluded."""

    problem_hash: str
    policy_hash: str
    operator_forecast_hash: str
    geometry_receipt_hash: str
    device: str
    route_count: int
    response_rank: int
    context_count: int
    fiber_count: int
    block_count: int
    already_resident_geometry_bytes: int
    already_resident_structural_bytes: int
    newly_allocated_source_bytes: int
    persistent_source_bytes: int
    solver_state_bytes: int
    work_dual_bytes: int
    transient_workspace_bytes: int
    forecast_incremental_bytes: int
    current_reserved_bytes: int
    forecast_peak_reserved_bytes: int
    device_total_memory_bytes: int
    hbm_ceiling_bytes: int
    admitted: bool
    route_scale_d2h_bytes: int = 0
    route_scale_numpy_arrays_constructed: int = 0
    borrowed_geometry_counted_as_new_allocation: bool = False
    borrowed_signed_work_row_bank_counted_as_new_allocation: bool = False
    schema: str = SIGNED_WORK_LEAST_WORK_CUDA_DEVICE_FORECAST_SCHEMA_V2

    def __post_init__(self) -> None:
        for name in (
            "problem_hash",
            "policy_hash",
            "operator_forecast_hash",
            "geometry_receipt_hash",
            "device",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "route_count",
            "context_count",
            "fiber_count",
            "already_resident_geometry_bytes",
            "already_resident_structural_bytes",
            "newly_allocated_source_bytes",
            "persistent_source_bytes",
            "solver_state_bytes",
            "work_dual_bytes",
            "transient_workspace_bytes",
            "forecast_incremental_bytes",
            "current_reserved_bytes",
            "forecast_peak_reserved_bytes",
            "device_total_memory_bytes",
            "hbm_ceiling_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "response_rank",
            _nonnegative_int(self.response_rank, name="response rank"),
        )
        object.__setattr__(
            self,
            "block_count",
            _nonnegative_int(self.block_count, name="block count"),
        )
        if type(self.admitted) is not bool:
            raise TypeError("device HBM admission must be an exact bool")
        if (
            self.route_scale_d2h_bytes != 0
            or self.route_scale_numpy_arrays_constructed != 0
            or self.borrowed_geometry_counted_as_new_allocation
            or self.borrowed_signed_work_row_bank_counted_as_new_allocation
            or self.newly_allocated_source_bytes != self.persistent_source_bytes
            or self.forecast_peak_reserved_bytes
            != self.current_reserved_bytes + self.forecast_incremental_bytes
            or self.schema != SIGNED_WORK_LEAST_WORK_CUDA_DEVICE_FORECAST_SCHEMA_V2
        ):
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.AUTHORITY_VIOLATION,
                "device HBM forecast double-counted resident sources or route-host work",
            )

    @property
    def forecast_hash(self) -> str:
        return content_hash(self, prefix=_DEVICE_FORECAST_PREFIX_V2)


def build_v1_signed_work_least_work_cuda_device_hbm_forecast_v2(
    problem: V1SignedWorkLeastWorkCUDADeviceProblemV2,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    *,
    device_total_memory_bytes: int | None = None,
    current_reserved_bytes: int | None = None,
) -> V1SignedWorkLeastWorkCUDADeviceHBMForecastV2:
    """Forecast only allocations made *after* live source custody exists."""

    if type(problem) is not V1SignedWorkLeastWorkCUDADeviceProblemV2:
        raise TypeError("device forecast requires the exact v2 device problem")
    if type(policy) is not V1SignedWorkLeastWorkCUDAPolicy:
        raise TypeError("device forecast requires the exact least-work policy")
    problem.assert_integrity()
    operator = problem.signed_work_operator
    device = _operator_device_object(operator)
    routes = problem.route_count
    rank = problem.rank
    contexts = operator.context_count
    fibers = operator.fiber_count
    blocks = len(problem.block_capacities)

    # singleton_work, H, and d are aliases owned by the sampled-D lease.  The
    # resource/legality tensors already exist when the device problem is
    # issued.  Only sqrt(d) and compact block bounds are new persistent source
    # allocations made by the solve frame.
    resident_geometry = (2 * routes + rank * routes) * _F64_BYTES
    resident_structural = (
        (_F64_BYTES if problem._resource_is_intrinsic_unit else routes * _F64_BYTES)
        + (_BOOL_BYTES if problem._eligible_is_intrinsic_all else routes * _BOOL_BYTES)
        + (_BOOL_BYTES if problem._required_is_intrinsic_empty else routes * _BOOL_BYTES)
    )
    new_source = routes * _F64_BYTES + blocks * _F64_BYTES
    solver_state = (
        6 * routes * 2 * _F64_BYTES
        + routes * _F64_BYTES
        + (rank * 2 + 2 + blocks * 2) * _F64_BYTES
    )
    work_dual = contexts * fibers * 2 * _F64_BYTES
    transient = (
        operator.forecast.temporary_tile_bytes_upper
        + 16 * routes * _F64_BYTES
        + contexts * fibers * 4 * _F64_BYTES
        + rank * 4 * _F64_BYTES
    )
    incremental = new_source + solver_state + work_dual + transient

    live_admission = (
        getattr(device, "type", None) == "cuda"
        and (device_total_memory_bytes is None or current_reserved_bytes is None)
    )
    if device_total_memory_bytes is None or current_reserved_bytes is None:
        if getattr(device, "type", None) == "cuda":
            if torch is None or not torch.cuda.is_available():
                _refuse(
                    V1SignedWorkLeastWorkCUDARefusalCode.CUDA_REQUIRED,
                    "CUDA device forecast requested without an available runtime",
                )
            if device_total_memory_bytes is None:
                device_total_memory_bytes = int(
                    torch.cuda.get_device_properties(device).total_memory
                )
            if current_reserved_bytes is None:
                current_reserved_bytes = int(torch.cuda.memory_reserved(device))
        else:
            if device_total_memory_bytes is None:
                device_total_memory_bytes = max(1 << 30, incremental * 4)
            if current_reserved_bytes is None:
                current_reserved_bytes = 0
    total = _positive_int(device_total_memory_bytes, name="device total memory")
    current = _nonnegative_int(current_reserved_bytes, name="current reserved bytes")
    ceiling = int(math.floor(policy.hbm_ceiling_fraction * total))
    peak = current + incremental
    return V1SignedWorkLeastWorkCUDADeviceHBMForecastV2(
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        operator_forecast_hash=operator.forecast.forecast_hash,
        geometry_receipt_hash=problem.geometry_receipt_hash,
        device=str(device),
        route_count=routes,
        response_rank=rank,
        context_count=contexts,
        fiber_count=fibers,
        block_count=blocks,
        already_resident_geometry_bytes=resident_geometry,
        already_resident_structural_bytes=resident_structural,
        newly_allocated_source_bytes=new_source,
        persistent_source_bytes=new_source,
        solver_state_bytes=solver_state,
        work_dual_bytes=work_dual,
        transient_workspace_bytes=transient,
        forecast_incremental_bytes=incremental,
        current_reserved_bytes=current,
        forecast_peak_reserved_bytes=peak,
        device_total_memory_bytes=total,
        hbm_ceiling_bytes=ceiling,
        admitted=(
            int(torch.cuda.memory_allocated(device)) + incremental <= ceiling
            and incremental <= cuda_reusable_memory_bytes(device, framework=torch)
        ) if live_admission else peak <= ceiling,
    )


_STATE_SOURCE_TENSOR_NAMES: Final = (
    "resource",
    "singleton_work",
    "factor",
    "residual_diagonal",
    "sqrt_residual",
    "eligible",
    "required",
)


@dataclass(frozen=True, slots=True)
class _SignedWorkDualStateTensors:
    """Final PDHG duals retained for source-bound signed-work witnessing."""

    branch_order: tuple[V1RelaxedKKTBranch, ...]
    work_dual: Any
    low_rank_dual: Any
    q_tail_duals: tuple[Any, ...]
    resource_dual: Any
    block_dual: Any


def _dual_state_tensor_items(
    value: _SignedWorkDualStateTensors,
) -> tuple[tuple[str, Any], ...]:
    return (
        ("work_dual", value.work_dual),
        ("low_rank_dual", value.low_rank_dual),
        *tuple(
            (f"q_tail_dual_{index}", tensor)
            for index, tensor in enumerate(value.q_tail_duals)
        ),
        ("resource_dual", value.resource_dual),
        ("block_dual", value.block_dual),
    )


class V1SignedWorkLeastWorkCUDAStateLease(AbstractContextManager):
    """Nonserializable short-lived access to retained device theorem state."""

    def __init__(
        self,
        owner: "V1SignedWorkLeastWorkCUDAStateHandle",
        operation: V1SignedWorkLeastWorkCUDAStateBorrowOperation,
    ) -> None:
        self._owner = owner
        self._operation = operation
        self._closed = False
        self._storage_invariants = owner._acquire(operation)

    @property
    def operation(self) -> V1SignedWorkLeastWorkCUDAStateBorrowOperation:
        return self._operation

    def primal(self, branch: V1RelaxedKKTBranch):
        if self._closed:
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.HANDLE_EVICTED,
                "state lease is closed",
            )
        return self._owner._primal(branch, self._storage_invariants)

    def reduced_advantage(self, branch: V1RelaxedKKTBranch):
        if self._closed:
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.HANDLE_EVICTED,
                "state lease is closed",
            )
        return self._owner._reduced(branch, self._storage_invariants)

    def source_tensor(self, name: str):
        if self._closed:
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.HANDLE_EVICTED,
                "state lease is closed",
            )
        if self._operation not in {
            V1SignedWorkLeastWorkCUDAStateBorrowOperation.DEVICE_SHELL_ROUNDING,
            V1SignedWorkLeastWorkCUDAStateBorrowOperation.HARD_FQ_REPLAY,
            V1SignedWorkLeastWorkCUDAStateBorrowOperation.SIGNED_WORK_DECISION_WITNESS,
            V1SignedWorkLeastWorkCUDAStateBorrowOperation.BLOCK_SUPPORT_OUTSIDE_PAIR,
        }:
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.AUTHORITY_VIOLATION,
                "reference inspection cannot borrow route-scale source tensors",
            )
        return self._owner._source_tensor(name, self._storage_invariants)

    def decision_dual_summary(
        self,
        name: str,
        branch: V1RelaxedKKTBranch,
    ) -> tuple[float, float, int, int]:
        """Bounded scalar read of one final dual coordinate.

        Returns ``(l1_norm, max_value, argmax_index, positive_count)`` reduced
        on device under the ``BINDING_CONSTRAINT_DIAGNOSTIC`` lease.  No dual
        tensor is exposed; the read carries diagnostic authority only.
        """

        if self._closed:
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.HANDLE_EVICTED,
                "state lease is closed",
            )
        if (
            self._operation
            is not V1SignedWorkLeastWorkCUDAStateBorrowOperation.BINDING_CONSTRAINT_DIAGNOSTIC
        ):
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.AUTHORITY_VIOLATION,
                "dual summaries require a binding-constraint diagnostic lease",
            )
        tensor = self._owner._decision_dual_tensor(
            name,
            branch,
            self._storage_invariants,
        )
        flat = tensor.reshape(-1).to(dtype=torch.float64)
        if int(flat.numel()) == 0:
            return (0.0, 0.0, -1, 0)
        summary = torch.stack(
            (
                torch.sum(torch.abs(flat)),
                torch.max(flat),
                torch.argmax(flat).to(dtype=torch.float64),
                torch.count_nonzero(flat > 0.0).to(dtype=torch.float64),
            )
        ).detach().cpu().tolist()
        return (
            float(summary[0]),
            float(summary[1]),
            int(round(summary[2])),
            int(round(summary[3])),
        )

    def decision_dual_tensor(
        self,
        name: str,
        branch: V1RelaxedKKTBranch,
    ):
        """Borrow one exact final dual coordinate for the signed-work witness."""

        if self._closed:
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.HANDLE_EVICTED,
                "state lease is closed",
            )
        if (
            self._operation
            is not V1SignedWorkLeastWorkCUDAStateBorrowOperation.SIGNED_WORK_DECISION_WITNESS
        ):
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.AUTHORITY_VIOLATION,
                "decision duals require a signed-work witness lease",
            )
        return self._owner._decision_dual_tensor(
            name,
            branch,
            self._storage_invariants,
        )

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        if not self._closed:
            self._closed = True
            self._owner._release(self._operation, self._storage_invariants)

    def __reduce__(self) -> NoReturn:
        raise TypeError("least-work state leases are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("least-work state leases are nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("least-work state leases cannot be copied")

    def __deepcopy__(self, memo: dict[int, Any]) -> NoReturn:
        del memo
        raise TypeError("least-work state leases cannot be copied")


class V1SignedWorkLeastWorkCUDAOutsidePairSourceLease(AbstractContextManager):
    """Purpose-bound zero-copy view of the outside-pair KKT source tensors.

    The physical alternate-domain provider owns preferred/fixed membership.
    This lease contributes only the exact Phase-A resource, eligibility, and
    required-rung tensors already retained by the KKT state.  It intentionally
    exposes neither arbitrary source tensors nor any host materialization or
    content-digest surface.
    """

    __slots__ = (
        "_lease",
        "_closed",
        "problem_hash",
        "policy_hash",
        "branch_receipt_hashes",
        "state_identity_hash",
        "eligible_mask_identity_hash",
        "required_mask_identity_hash",
        "route_count",
        "device",
        "route_scale_d2h_bytes",
        "route_scale_content_digest_bytes",
        "downstream_authority",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        owner: "V1SignedWorkLeastWorkCUDAStateHandle",
        _issuer_token: object,
    ) -> None:
        if _issuer_token is not _OUTSIDE_PAIR_SOURCE_LEASE_TOKEN:
            raise TypeError("outside-pair KKT source leases are handle-issued")
        self._lease = V1SignedWorkLeastWorkCUDAStateLease(
            owner,
            V1SignedWorkLeastWorkCUDAStateBorrowOperation.BLOCK_SUPPORT_OUTSIDE_PAIR,
        )
        self._closed = False
        self.problem_hash = owner.problem_hash
        self.policy_hash = owner.policy_hash
        self.branch_receipt_hashes = owner.branch_receipt_hashes
        self.state_identity_hash = owner.identity_hash
        self.eligible_mask_identity_hash = owner.eligible_mask_identity_hash
        self.required_mask_identity_hash = owner.required_mask_identity_hash
        self.route_count = owner.route_count
        self.device = owner.device
        self.route_scale_d2h_bytes = 0
        self.route_scale_content_digest_bytes = 0
        self.downstream_authority = False

    @property
    def operation(self) -> V1SignedWorkLeastWorkCUDAStateBorrowOperation:
        self._require_open()
        return self._lease.operation

    def _require_open(self) -> None:
        if self._closed:
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.HANDLE_EVICTED,
                "outside-pair KKT source lease is closed",
            )

    def live_source_tensors(self) -> tuple[Any, Any, Any]:
        """Return ``(resource, eligible, required)`` without copying."""

        self._require_open()
        resource = self._lease.source_tensor("resource")
        eligible = self._lease.source_tensor("eligible")
        required = self._lease.source_tensor("required")
        if (
            tuple(resource.shape) != (self.route_count,)
            or tuple(eligible.shape) != (self.route_count,)
            or tuple(required.shape) != (self.route_count,)
            or str(resource.device) != self.device
            or eligible.device != resource.device
            or required.device != resource.device
        ):
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.CUSTODY_MUTATED,
                "outside-pair KKT source tensors lost their route/device identity",
            )
        return resource, eligible, required

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._lease.__exit__(None, None, None)
        finally:
            self._closed = True

    def __enter__(self) -> "V1SignedWorkLeastWorkCUDAOutsidePairSourceLease":
        self.live_source_tensors()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.close()

    def __reduce__(self) -> NoReturn:
        raise TypeError("outside-pair KKT source leases are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("outside-pair KKT source leases are nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("outside-pair KKT source leases cannot be copied")

    def __deepcopy__(self, memo: dict[int, Any]) -> NoReturn:
        del memo
        raise TypeError("outside-pair KKT source leases cannot be copied")


class V1SignedWorkLeastWorkCUDAStateHandle:
    """Opaque device custody for one or both registered Q branches."""

    def __init__(
        self,
        *,
        problem_hash: str,
        policy_hash: str,
        branch_receipt_hashes: tuple[str, ...],
        primals: Any,
        reduced_advantages: Any,
        source_tensors: Any | None = None,
        dual_tensors: _SignedWorkDualStateTensors | None = None,
        materialize_route_limit: int,
        eligible_mask_identity_hash: str | None = None,
        required_mask_identity_hash: str | None = None,
        external_geometry_lease: (
            V1SampledDeviceResidentDKKTGeometryLease | None
        ) = None,
        branch_order: tuple[V1RelaxedKKTBranch, ...] = (
            V1RelaxedKKTBranch.SPECTRAL,
            V1RelaxedKKTBranch.DIAGONAL_L1,
        ),
        _exclusive_transfer_token: object | None = None,
    ) -> None:
        if tuple(primals.shape) != tuple(reduced_advantages.shape):
            raise ValueError("primal and reduced-advantage custody differ")
        branches = tuple(branch_order)
        canonical = {
            V1RelaxedKKTBranch.SPECTRAL,
            V1RelaxedKKTBranch.DIAGONAL_L1,
        }
        if (
            not branches
            or len(branches) > 2
            or len(set(branches)) != len(branches)
            or any(type(value) is not V1RelaxedKKTBranch for value in branches)
            or not set(branches).issubset(canonical)
            or primals.ndim != 2
            or int(primals.shape[1]) != len(branches)
        ):
            raise ValueError(
                "least-work theorem state must bind its exact one/two-branch axis"
            )
        self._branch_order = branches
        self._problem_hash = _identity(problem_hash, name="problem hash")
        self._policy_hash = _identity(policy_hash, name="policy hash")
        self._receipt_hashes = tuple(
            _identity(item, name="branch receipt hash")
            for item in branch_receipt_hashes
        )
        if len(self._receipt_hashes) != len(self._branch_order):
            raise ValueError("theorem-state receipts differ from its branch axis")
        if (
            str(primals.dtype) != "torch.float64"
            or primals.dtype != reduced_advantages.dtype
            or primals.device != reduced_advantages.device
        ):
            raise TypeError("least-work theorem state requires one float64 device")
        if (
            not primals.is_contiguous()
            or not reduced_advantages.is_contiguous()
            or primals.requires_grad
            or reduced_advantages.requires_grad
        ):
            raise ValueError(
                "exclusive theorem-state transfer requires canonical tensors"
            )
        route_count = int(primals.shape[0])
        self._materialize_route_limit = _nonnegative_int(
            materialize_route_limit, name="materialize route limit"
        )
        source_map: dict[str, Any] = {}
        dual_map: dict[str, Any] = {}
        external_geometry_receipt_hash: str | None = None
        if _exclusive_transfer_token is _STATE_EXCLUSIVE_TRANSFER_TOKEN:
            if type(source_tensors) is not _FusedSourceTensors:
                raise TypeError("least-work state requires exact fused source custody")
            if type(dual_tensors) is not _SignedWorkDualStateTensors:
                raise TypeError(
                    "least-work decision state requires exact final dual custody"
                )
            if tuple(dual_tensors.branch_order) != branches:
                raise ValueError("final dual state differs from the branch axis")
            if (
                eligible_mask_identity_hash is None
                or required_mask_identity_hash is None
            ):
                raise ValueError(
                    "production theorem state requires durable eligible/required "
                    "mask identities"
                )
            source_map = {
                name: getattr(source_tensors, name)
                for name in _STATE_SOURCE_TENSOR_NAMES
            }
            for name, tensor in source_map.items():
                intrinsic_structural_view = (
                    torch.is_tensor(tensor)
                    and name in {"resource", "eligible", "required"}
                    and tensor.ndim == 1
                    and tuple(int(item) for item in tensor.stride()) == (0,)
                    and int(tensor.storage_offset()) == 0
                    and int(tensor.untyped_storage().nbytes())
                    <= int(tensor.element_size())
                )
                if (
                    not torch.is_tensor(tensor)
                    or tensor.device != primals.device
                    or not (tensor.is_contiguous() or intrinsic_structural_view)
                    or tensor.requires_grad
                ):
                    raise ValueError(f"source tensor {name} is not canonical custody")
            for name in (
                "resource",
                "singleton_work",
                "residual_diagonal",
                "sqrt_residual",
                "eligible",
                "required",
            ):
                if tuple(source_map[name].shape) != (route_count,):
                    raise ValueError(f"source tensor {name} differs from route axis")
            if (
                source_map["factor"].ndim != 2
                or int(source_map["factor"].shape[1]) != route_count
            ):
                raise ValueError("source response factor differs from route axis")
            if any(
                source_map[name].dtype != torch.float64
                for name in (
                    "resource",
                    "singleton_work",
                    "factor",
                    "residual_diagonal",
                    "sqrt_residual",
                )
            ) or any(
                source_map[name].dtype != torch.bool
                for name in ("eligible", "required")
            ):
                raise TypeError("least-work source custody has invalid dtypes")
            if external_geometry_lease is not None:
                if type(external_geometry_lease) is not V1SampledDeviceResidentDKKTGeometryLease:
                    raise TypeError("external geometry custody requires the exact lease")
                geometry_receipt = external_geometry_lease.receipt
                aliases = (
                    (
                        source_map["singleton_work"],
                        external_geometry_lease.singleton_work,
                    ),
                    (source_map["factor"], external_geometry_lease.factor),
                    (
                        source_map["residual_diagonal"],
                        external_geometry_lease.residual_diagonal,
                    ),
                )
                if any(
                    left.data_ptr() != right.data_ptr()
                    or _state_tensor_storage_invariant(left)
                    != _state_tensor_storage_invariant(right)
                    for left, right in aliases
                ):
                    _refuse(
                        V1SignedWorkLeastWorkCUDARefusalCode.CUSTODY_MUTATED,
                        "retained theorem state copied or rebound live D geometry",
                    )
                external_geometry_receipt_hash = geometry_receipt.receipt_hash
            dual_map = dict(_dual_state_tensor_items(dual_tensors))
            for name, tensor in dual_map.items():
                if (
                    not torch.is_tensor(tensor)
                    or tensor.dtype != torch.float64
                    or tensor.device != primals.device
                    or not tensor.is_contiguous()
                    or tensor.requires_grad
                ):
                    raise ValueError(f"dual tensor {name} is not canonical custody")
            branch_count = len(branches)
            if (
                dual_map["work_dual"].ndim != 3
                or int(dual_map["work_dual"].shape[2]) != branch_count
                or dual_map["low_rank_dual"].ndim != 2
                or int(dual_map["low_rank_dual"].shape[1]) != branch_count
                or tuple(dual_map["resource_dual"].shape) != (branch_count,)
                or dual_map["block_dual"].ndim != 2
                or int(dual_map["block_dual"].shape[1]) != branch_count
            ):
                raise ValueError("final dual tensors differ from the branch axis")
            for index, branch in enumerate(branches):
                expected_tail_shape = (
                    (route_count,)
                    if branch is V1RelaxedKKTBranch.SPECTRAL
                    else ()
                )
                if tuple(dual_map[f"q_tail_dual_{index}"].shape) != expected_tail_shape:
                    raise ValueError(
                        f"final {branch.value} Q-tail dual has the wrong shape"
                    )
            # The internal solve frame relinquishes these buffers.  Transfer
            # avoids route-scale D2D copies while preserving version custody.
            self._primals = primals.detach()
            self._reduced_advantages = reduced_advantages.detach()
        else:
            if (
                source_tensors is not None
                or dual_tensors is not None
                or external_geometry_lease is not None
                or primals.device.type != "cpu"
                or route_count > self._materialize_route_limit
            ):
                raise PermissionError(
                    "production theorem state can only be issued by its solver"
                )
            # Explicit tiny CPU construction supports deterministic shell tests
            # but cannot lend production source tensors or replay authority.
            self._primals = primals.detach().clone().contiguous()
            self._reduced_advantages = (
                reduced_advantages.detach().clone().contiguous()
            )
        if (eligible_mask_identity_hash is None) != (
            required_mask_identity_hash is None
        ):
            raise ValueError("eligible/required mask identities must be paired")
        self._eligible_mask_identity_hash = (
            None
            if eligible_mask_identity_hash is None
            else _identity(
                eligible_mask_identity_hash,
                name="eligible mask identity hash",
            )
        )
        self._required_mask_identity_hash = (
            None
            if required_mask_identity_hash is None
            else _identity(
                required_mask_identity_hash,
                name="required mask identity hash",
            )
        )
        self._source_tensors = source_map
        self._dual_tensors = dual_map
        self._external_geometry_lease = external_geometry_lease
        self._external_geometry_receipt_hash = external_geometry_receipt_hash
        self._lock = RLock()
        self._leases = 0
        self._evicted = False
        self._integrity_failed = False
        self._storage_invariants = (
            _state_tensor_storage_invariant(self._primals),
            _state_tensor_storage_invariant(self._reduced_advantages),
            *(
                _state_tensor_storage_invariant(self._source_tensors[name])
                for name in _STATE_SOURCE_TENSOR_NAMES
                if name in self._source_tensors
            ),
            *(
                _state_tensor_storage_invariant(tensor)
                for _name, tensor in self._dual_tensors.items()
            ),
        )
        identity_payload: dict[str, object] = {
                "schema": SIGNED_WORK_LEAST_WORK_CUDA_STATE_SCHEMA,
                "problem_hash": self._problem_hash,
                "policy_hash": self._policy_hash,
                "branch_receipt_hashes": self._receipt_hashes,
                "shape": tuple(int(item) for item in primals.shape),
                "dtype": str(primals.dtype),
                "device": str(primals.device),
                "materialize_route_limit": self._materialize_route_limit,
                "source_tensor_custody_available": bool(self._source_tensors),
                "eligible_mask_identity_hash": self._eligible_mask_identity_hash,
                "required_mask_identity_hash": self._required_mask_identity_hash,
                "signed_work_decision_dual_custody_available": bool(
                    self._dual_tensors
                ),
                "storage_shape_dtype_device_commitment": content_hash(
                    tuple(
                        invariant[:4] + invariant[8:]
                        for invariant in self._storage_invariants
                    )
                ),
                "nonserializable_device_custody": True,
            }
        # Preserve the established two-branch identity byte-for-byte.  A
        # single-branch state adds its exact branch axis so it cannot be
        # substituted for legacy two-branch custody.
        if self._branch_order != SIGNED_WORK_LEAST_WORK_BRANCH_ORDER:
            identity_payload["branch_order"] = self._branch_order
        if external_geometry_receipt_hash is not None:
            identity_payload.update(
                {
                    "external_geometry_receipt_hash": (
                        external_geometry_receipt_hash
                    ),
                    "external_geometry_zero_copy_alias": True,
                    "external_geometry_lease_retained_until_state_eviction": True,
                }
            )
        self._identity_hash = content_hash(
            identity_payload,
            prefix=_STATE_PREFIX,
        )

    @property
    def identity_hash(self) -> str:
        return self._identity_hash

    @property
    def problem_hash(self) -> str:
        return self._problem_hash

    @property
    def policy_hash(self) -> str:
        return self._policy_hash

    @property
    def eligible_mask_identity_hash(self) -> str:
        with self._lock:
            self._require_live()
            if self._eligible_mask_identity_hash is None:
                _refuse(
                    V1SignedWorkLeastWorkCUDARefusalCode.AUTHORITY_VIOLATION,
                    "reference-only theorem state has no durable eligible-mask identity",
                )
            return self._eligible_mask_identity_hash

    @property
    def required_mask_identity_hash(self) -> str:
        with self._lock:
            self._require_live()
            if self._required_mask_identity_hash is None:
                _refuse(
                    V1SignedWorkLeastWorkCUDARefusalCode.AUTHORITY_VIOLATION,
                    "reference-only theorem state has no durable required-mask identity",
                )
            return self._required_mask_identity_hash

    @property
    def branch_receipt_hashes(self) -> tuple[str, ...]:
        return self._receipt_hashes

    @property
    def branch_order(self) -> tuple[V1RelaxedKKTBranch, ...]:
        return self._branch_order

    @property
    def route_count(self) -> int:
        with self._lock:
            self._require_live()
            return int(self._primals.shape[0])

    @property
    def device(self) -> str:
        with self._lock:
            self._require_live()
            return str(self._primals.device)

    @property
    def is_evicted(self) -> bool:
        return self._evicted

    @property
    def signed_work_decision_dual_custody_available(self) -> bool:
        with self._lock:
            self._require_live()
            return bool(self._dual_tensors)

    def borrow(self) -> V1SignedWorkLeastWorkCUDAStateLease:
        """Compatibility borrow for bounded reference inspection only."""

        return V1SignedWorkLeastWorkCUDAStateLease(
            self,
            V1SignedWorkLeastWorkCUDAStateBorrowOperation.REFERENCE_INSPECTION,
        )

    def borrow_for_device_shell_rounding(
        self,
        *,
        expected_problem_hash: str,
        expected_policy_hash: str,
        expected_branch_receipt_hashes: tuple[str, ...],
    ) -> V1SignedWorkLeastWorkCUDAStateLease:
        expected = (
            _identity(expected_problem_hash, name="expected problem hash"),
            _identity(expected_policy_hash, name="expected policy hash"),
            tuple(
                _identity(value, name="expected branch receipt hash")
                for value in expected_branch_receipt_hashes
            ),
        )
        if len(expected[2]) != len(self._branch_order) or expected != (
            self._problem_hash,
            self._policy_hash,
            self._receipt_hashes,
        ):
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.STALE_SOURCE,
                "device-shell rounding requested stale theorem state",
            )
        return V1SignedWorkLeastWorkCUDAStateLease(
            self,
            V1SignedWorkLeastWorkCUDAStateBorrowOperation.DEVICE_SHELL_ROUNDING,
        )

    def borrow_for_hard_fq_replay(
        self,
        *,
        expected_problem_hash: str,
        expected_policy_hash: str,
        expected_branch_receipt_hashes: tuple[str, ...],
    ) -> V1SignedWorkLeastWorkCUDAStateLease:
        expected = (
            _identity(expected_problem_hash, name="expected problem hash"),
            _identity(expected_policy_hash, name="expected policy hash"),
            tuple(
                _identity(value, name="expected branch receipt hash")
                for value in expected_branch_receipt_hashes
            ),
        )
        if len(expected[2]) != len(self._branch_order) or expected != (
            self._problem_hash,
            self._policy_hash,
            self._receipt_hashes,
        ):
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.STALE_SOURCE,
                "hard-F/Q replay requested stale theorem-state source tensors",
            )
        return V1SignedWorkLeastWorkCUDAStateLease(
            self,
            V1SignedWorkLeastWorkCUDAStateBorrowOperation.HARD_FQ_REPLAY,
        )

    def borrow_for_signed_work_decision_witness(
        self,
        *,
        expected_problem_hash: str,
        expected_policy_hash: str,
        expected_branch_receipt_hashes: tuple[str, ...],
    ) -> V1SignedWorkLeastWorkCUDAStateLease:
        """Lend exact primal/source/final-dual state to one witness issuer."""

        expected = (
            _identity(expected_problem_hash, name="expected problem hash"),
            _identity(expected_policy_hash, name="expected policy hash"),
            tuple(
                _identity(value, name="expected branch receipt hash")
                for value in expected_branch_receipt_hashes
            ),
        )
        if len(expected[2]) != len(self._branch_order) or expected != (
            self._problem_hash,
            self._policy_hash,
            self._receipt_hashes,
        ):
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.STALE_SOURCE,
                "signed-work decision witness requested stale theorem state",
            )
        if not self._dual_tensors:
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.AUTHORITY_VIOLATION,
                "state lacks final signed-work decision dual custody",
            )
        return V1SignedWorkLeastWorkCUDAStateLease(
            self,
            V1SignedWorkLeastWorkCUDAStateBorrowOperation.SIGNED_WORK_DECISION_WITNESS,
        )

    def borrow_for_binding_constraint_diagnostic(
        self,
        *,
        expected_problem_hash: str,
        expected_policy_hash: str,
        expected_branch_receipt_hashes: tuple[str, ...],
    ) -> V1SignedWorkLeastWorkCUDAStateLease:
        """Lend final duals for bounded scalar summaries only (no tensors)."""

        expected = (
            _identity(expected_problem_hash, name="expected problem hash"),
            _identity(expected_policy_hash, name="expected policy hash"),
            tuple(
                _identity(value, name="expected branch receipt hash")
                for value in expected_branch_receipt_hashes
            ),
        )
        if len(expected[2]) != len(self._branch_order) or expected != (
            self._problem_hash,
            self._policy_hash,
            self._receipt_hashes,
        ):
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.STALE_SOURCE,
                "binding-constraint diagnostic requested stale theorem state",
            )
        if not self._dual_tensors:
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.AUTHORITY_VIOLATION,
                "state lacks final signed-work decision dual custody",
            )
        return V1SignedWorkLeastWorkCUDAStateLease(
            self,
            V1SignedWorkLeastWorkCUDAStateBorrowOperation.BINDING_CONSTRAINT_DIAGNOSTIC,
        )

    def borrow_for_block_support_outside_pair(
        self,
        *,
        expected_problem_hash: str,
        expected_policy_hash: str,
        expected_branch_receipt_hashes: tuple[str, ...],
    ) -> V1SignedWorkLeastWorkCUDAOutsidePairSourceLease:
        """Lend only the exact route resource/legality state to Phase-B."""

        expected = (
            _identity(expected_problem_hash, name="expected problem hash"),
            _identity(expected_policy_hash, name="expected policy hash"),
            tuple(
                _identity(value, name="expected branch receipt hash")
                for value in expected_branch_receipt_hashes
            ),
        )
        if len(expected[2]) != len(self._branch_order) or expected != (
            self._problem_hash,
            self._policy_hash,
            self._receipt_hashes,
        ):
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.STALE_SOURCE,
                "outside-pair support requested stale KKT theorem state",
            )
        if not self._source_tensors:
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.AUTHORITY_VIOLATION,
                "state lacks production source-tensor custody",
            )
        return V1SignedWorkLeastWorkCUDAOutsidePairSourceLease(
            owner=self,
            _issuer_token=_OUTSIDE_PAIR_SOURCE_LEASE_TOKEN,
        )

    def _require_live(self) -> None:
        if self._evicted:
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.HANDLE_EVICTED,
                "least-work state was evicted",
            )
        if self._integrity_failed:
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.CUSTODY_MUTATED,
                "least-work theorem state previously detected mutation",
            )
        if self._external_geometry_lease is not None:
            try:
                receipt = self._external_geometry_lease.receipt
            except BaseException as error:
                self._integrity_failed = True
                raise V1SignedWorkLeastWorkCUDARefusal(
                    V1SignedWorkLeastWorkCUDARefusalCode.CUSTODY_MUTATED,
                    "retained theorem state lost its live D-geometry lease",
                ) from error
            if receipt.receipt_hash != self._external_geometry_receipt_hash:
                self._integrity_failed = True
                _refuse(
                    V1SignedWorkLeastWorkCUDARefusalCode.CUSTODY_MUTATED,
                    "retained theorem-state geometry receipt changed",
                )
        observed = (
            _state_tensor_storage_invariant(self._primals),
            _state_tensor_storage_invariant(self._reduced_advantages),
            *(
                _state_tensor_storage_invariant(self._source_tensors[name])
                for name in _STATE_SOURCE_TENSOR_NAMES
                if name in self._source_tensors
            ),
            *(
                _state_tensor_storage_invariant(tensor)
                for _name, tensor in self._dual_tensors.items()
            ),
        )
        if observed != self._storage_invariants:
            self._integrity_failed = True
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.CUSTODY_MUTATED,
                "least-work theorem-state storage or tensor version changed",
            )

    def _acquire(
        self,
        operation: V1SignedWorkLeastWorkCUDAStateBorrowOperation,
    ) -> tuple[tuple[object, ...], ...]:
        if type(operation) is not V1SignedWorkLeastWorkCUDAStateBorrowOperation:
            raise TypeError("least-work state borrow operation must be exact")
        with self._lock:
            self._require_live()
            self._leases += 1
            return self._storage_invariants

    def _release(
        self,
        operation: V1SignedWorkLeastWorkCUDAStateBorrowOperation,
        invariants: tuple[tuple[object, ...], ...],
    ) -> None:
        with self._lock:
            self._leases -= 1
            if self._leases < 0:  # pragma: no cover - defensive invariant.
                raise RuntimeError("least-work lease accounting underflow")
            if (
                type(operation) is not V1SignedWorkLeastWorkCUDAStateBorrowOperation
                or invariants != self._storage_invariants
            ):
                self._integrity_failed = True
                _refuse(
                    V1SignedWorkLeastWorkCUDARefusalCode.CUSTODY_MUTATED,
                    "least-work theorem-state lease lost its source identity",
                )
            self._require_live()

    def _column(self, branch: V1RelaxedKKTBranch) -> int:
        if type(branch) is not V1RelaxedKKTBranch:
            raise TypeError("unknown least-work Q branch")
        try:
            return self._branch_order.index(branch)
        except ValueError:
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.AUTHORITY_VIOLATION,
                f"{branch.value} is outside this theorem-state branch axis",
            )

    def _primal(
        self,
        branch: V1RelaxedKKTBranch,
        invariants: tuple[tuple[object, ...], ...],
    ):
        self._require_live()
        if invariants != self._storage_invariants:
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.CUSTODY_MUTATED,
                "least-work primal lease is stale",
            )
        return self._primals[:, self._column(branch)]

    def _reduced(
        self,
        branch: V1RelaxedKKTBranch,
        invariants: tuple[tuple[object, ...], ...],
    ):
        self._require_live()
        if invariants != self._storage_invariants:
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.CUSTODY_MUTATED,
                "least-work reduced-advantage lease is stale",
            )
        return self._reduced_advantages[:, self._column(branch)]

    def _source_tensor(
        self,
        name: str,
        invariants: tuple[tuple[object, ...], ...],
    ):
        self._require_live()
        if invariants != self._storage_invariants:
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.CUSTODY_MUTATED,
                "least-work source-tensor lease is stale",
            )
        if type(name) is not str or name not in _STATE_SOURCE_TENSOR_NAMES:
            raise ValueError("unknown least-work source tensor")
        if name not in self._source_tensors:
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.AUTHORITY_VIOLATION,
                "tiny reference state has no production source-tensor custody",
            )
        return self._source_tensors[name]

    def _decision_dual_tensor(
        self,
        name: str,
        branch: V1RelaxedKKTBranch,
        invariants: tuple[tuple[object, ...], ...],
    ):
        self._require_live()
        if invariants != self._storage_invariants:
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.CUSTODY_MUTATED,
                "signed-work decision-dual lease is stale",
            )
        if not self._dual_tensors:
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.AUTHORITY_VIOLATION,
                "least-work state has no final decision duals",
            )
        column = self._column(branch)
        if name == "work_dual":
            return self._dual_tensors[name][:, :, column]
        if name == "low_rank_dual":
            return self._dual_tensors[name][:, column]
        if name == "q_tail_dual":
            return self._dual_tensors[f"q_tail_dual_{column}"]
        if name == "resource_dual":
            return self._dual_tensors[name][column]
        if name == "block_dual":
            return self._dual_tensors[name][:, column]
        raise ValueError("unknown signed-work decision dual tensor")

    def materialize_for_private_test(
        self, branch: V1RelaxedKKTBranch
    ) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            self._require_live()
            if int(self._primals.shape[0]) > self._materialize_route_limit:
                _refuse(
                    V1SignedWorkLeastWorkCUDARefusalCode.AUTHORITY_VIOLATION,
                    "route-scale theorem state cannot be materialized on the host",
                )
            index = self._column(branch)
            return (
                self._primals[:, index].detach().cpu().numpy().copy(),
                self._reduced_advantages[:, index]
                .detach()
                .cpu()
                .numpy()
                .copy(),
            )

    def evict(self) -> None:
        with self._lock:
            if self._evicted:
                return
            if self._leases:
                _refuse(
                    V1SignedWorkLeastWorkCUDARefusalCode.ACTIVE_LEASE,
                    "cannot evict least-work state while a lease is active",
                )
            del self._primals
            del self._reduced_advantages
            del self._source_tensors
            del self._dual_tensors
            self._storage_invariants = ((),)
            self._evicted = True
            geometry_lease = self._external_geometry_lease
            self._external_geometry_lease = None
            if geometry_lease is not None:
                geometry_lease.close()

    def __reduce__(self) -> NoReturn:
        raise TypeError("least-work state handles are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("least-work state handles are nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("least-work state handles cannot be copied")

    def __deepcopy__(self, memo: dict[int, Any]) -> NoReturn:
        del memo
        raise TypeError("least-work state handles cannot be copied")


@dataclass(frozen=True, slots=True)
class V1SignedWorkLeastWorkCUDAExecutionBundle:
    result: V1SignedWorkLeastWorkCUDAResult
    forecast: (
        V1SignedWorkLeastWorkCUDAHBMForecast
        | V1SignedWorkLeastWorkCUDADeviceHBMForecastV2
    )
    state_handle: V1SignedWorkLeastWorkCUDAStateHandle | None = field(
        repr=False,
        compare=False,
    )
    production_cuda_execution: bool
    private_cpu_reference_execution: bool
    schema: str = SIGNED_WORK_LEAST_WORK_CUDA_EXECUTION_SCHEMA

    def __post_init__(self) -> None:
        if self.production_cuda_execution == self.private_cpu_reference_execution:
            raise ValueError("exactly one least-work execution mode must be active")
        if self.result.forecast_hash != self.forecast.forecast_hash:
            raise ValueError("least-work result and forecast differ")
        if self.result.relaxed_state_available != (self.state_handle is not None):
            raise ValueError("least-work state availability is inconsistent")
        if self.state_handle is not None and (
            self.result.state_identity_hash != self.state_handle.identity_hash
        ):
            raise ValueError("least-work state identity differs from its result")
        if self.schema != SIGNED_WORK_LEAST_WORK_CUDA_EXECUTION_SCHEMA:
            raise ValueError("unsupported least-work execution schema")

    @property
    def execution_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "result_hash": self.result.result_hash,
                "forecast_hash": self.forecast.forecast_hash,
                "state_identity_hash": self.result.state_identity_hash,
                "production_cuda_execution": self.production_cuda_execution,
                "private_cpu_reference_execution": self.private_cpu_reference_execution,
            },
            prefix=_EXECUTION_PREFIX,
        )

    # Convenience projections preserve an ergonomic proposal API without
    # conflating serializable result metadata with opaque device custody.
    @property
    def selected_branch(self) -> V1RelaxedKKTBranch | None:
        return self.result.selected_branch

    @property
    def branch_receipts(
        self,
    ) -> tuple[V1SignedWorkLeastWorkCUDABranchReceipt, ...]:
        return self.result.branch_receipts

    @property
    def relaxed_state_available(self) -> bool:
        return self.result.relaxed_state_available

    @property
    def relaxed_state_handle(
        self,
    ) -> V1SignedWorkLeastWorkCUDAStateHandle | None:
        return self.state_handle

    @property
    def state_identity_hash(self) -> str | None:
        return self.result.state_identity_hash

    @property
    def hard_mask_global_optimality_claimed(self) -> bool:
        return self.result.hard_mask_global_optimality_claimed

    @property
    def mask_commit_authority(self) -> bool:
        return self.result.mask_commit_authority

    @property
    def authority_scope(self) -> str:
        return self.result.authority_scope


@dataclass(frozen=True, slots=True)
class V1SignedWorkLeastWorkCUDASingleBranchResult:
    """One Phase-A-supported branch with retained theorem-state custody.

    The numerical program and retained state contain exactly this branch.  A
    different branch that Phase A excluded at this resource coordinate is
    neither evaluated nor represented by this result.
    """

    problem_hash: str
    policy_hash: str
    forecast_hash: str
    branch: V1RelaxedKKTBranch
    branch_receipt: V1SignedWorkLeastWorkCUDABranchReceipt
    solver_state_binding_hashes: tuple[str, ...]
    branch_converged: bool
    relaxed_state_available: bool
    state_identity_hash: str | None
    nonrequested_branch_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_claim_authority: bool = False
    authority_scope: str = SIGNED_WORK_LEAST_WORK_CUDA_AUTHORITY
    schema: str = SIGNED_WORK_LEAST_WORK_CUDA_SINGLE_BRANCH_RESULT_SCHEMA

    def __post_init__(self) -> None:
        for name in ("problem_hash", "policy_hash", "forecast_hash"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("single-branch result requires one exact Q branch")
        if type(self.branch_receipt) is not V1SignedWorkLeastWorkCUDABranchReceipt:
            raise TypeError("single-branch result requires one exact branch receipt")
        if (
            self.branch_receipt.branch is not self.branch
            or self.branch_receipt.problem_hash != self.problem_hash
            or self.branch_receipt.policy_hash != self.policy_hash
        ):
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.STALE_SOURCE,
                "single-branch result and numerical receipt differ",
            )
        hashes = tuple(
            _identity(value, name="solver state binding hash")
            for value in self.solver_state_binding_hashes
        )
        if hashes != (self.branch_receipt.receipt_hash,):
            raise ValueError("single-branch result lost its authorized solver identity")
        object.__setattr__(self, "solver_state_binding_hashes", hashes)
        for name in (
            "branch_converged",
            "relaxed_state_available",
            "nonrequested_branch_authority",
            "c_stream_authority",
            "calibration_authority",
            "bce_oracle_authority",
            "mask_commit_authority",
            "terminal_authority",
            "pareto_claim_authority",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact bool")
        if self.branch_converged is not self.branch_receipt.converged:
            raise ValueError("single-branch convergence differs from its receipt")
        if self.relaxed_state_available is not self.branch_converged:
            raise ValueError("a converged single branch must retain its theorem state")
        if self.relaxed_state_available:
            object.__setattr__(
                self,
                "state_identity_hash",
                _identity(self.state_identity_hash, name="state identity hash"),
            )
        elif self.state_identity_hash is not None:
            raise ValueError("an unresolved single branch cannot carry theorem state")
        if (
            self.nonrequested_branch_authority
            or self.c_stream_authority
            or self.calibration_authority
            or self.bce_oracle_authority
            or self.mask_commit_authority
            or self.terminal_authority
            or self.pareto_claim_authority
            or self.authority_scope != SIGNED_WORK_LEAST_WORK_CUDA_AUTHORITY
            or self.schema != SIGNED_WORK_LEAST_WORK_CUDA_SINGLE_BRANCH_RESULT_SCHEMA
        ):
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.AUTHORITY_VIOLATION,
                "single-branch least-work result crossed proposal-only authority",
            )

    @property
    def result_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "problem_hash": self.problem_hash,
                "policy_hash": self.policy_hash,
                "forecast_hash": self.forecast_hash,
                "branch": self.branch,
                "branch_receipt_hash": self.branch_receipt.receipt_hash,
                "solver_state_binding_hashes": self.solver_state_binding_hashes,
                "branch_converged": self.branch_converged,
                "relaxed_state_available": self.relaxed_state_available,
                "state_identity_hash": self.state_identity_hash,
                "nonrequested_branch_authority": self.nonrequested_branch_authority,
                "c_stream_authority": self.c_stream_authority,
                "calibration_authority": self.calibration_authority,
                "bce_oracle_authority": self.bce_oracle_authority,
                "mask_commit_authority": self.mask_commit_authority,
                "terminal_authority": self.terminal_authority,
                "pareto_claim_authority": self.pareto_claim_authority,
                "authority_scope": self.authority_scope,
            },
            prefix=_RESULT_PREFIX + "single-branch-",
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle:
    result: V1SignedWorkLeastWorkCUDASingleBranchResult
    forecast: V1SignedWorkLeastWorkCUDASingleBranchHBMForecast
    state_handle: V1SignedWorkLeastWorkCUDAStateHandle | None = field(
        repr=False,
        compare=False,
    )
    production_cuda_execution: bool
    private_cpu_reference_execution: bool
    schema: str = SIGNED_WORK_LEAST_WORK_CUDA_SINGLE_BRANCH_EXECUTION_SCHEMA

    def __post_init__(self) -> None:
        if type(self.result) is not V1SignedWorkLeastWorkCUDASingleBranchResult:
            raise TypeError("single-branch execution requires an exact result")
        if (
            type(self.forecast)
            is not V1SignedWorkLeastWorkCUDASingleBranchHBMForecast
        ):
            raise TypeError("single-branch execution requires an exact forecast")
        if self.forecast.branch is not self.result.branch:
            raise ValueError("single-branch result and HBM branch differ")
        if self.production_cuda_execution == self.private_cpu_reference_execution:
            raise ValueError("exactly one single-branch execution mode must be active")
        if self.result.forecast_hash != self.forecast.forecast_hash:
            raise ValueError("single-branch result and forecast differ")
        if self.result.relaxed_state_available is not (self.state_handle is not None):
            raise ValueError("single-branch state availability is inconsistent")
        if self.state_handle is not None and (
            self.result.state_identity_hash != self.state_handle.identity_hash
            or self.result.solver_state_binding_hashes
            != self.state_handle.branch_receipt_hashes
        ):
            raise ValueError("single-branch result and theorem state differ")
        if self.schema != SIGNED_WORK_LEAST_WORK_CUDA_SINGLE_BRANCH_EXECUTION_SCHEMA:
            raise ValueError("unsupported single-branch execution schema")

    @property
    def branch(self) -> V1RelaxedKKTBranch:
        return self.result.branch

    @property
    def branch_receipt(self) -> V1SignedWorkLeastWorkCUDABranchReceipt:
        return self.result.branch_receipt

    @property
    def state_identity_hash(self) -> str | None:
        return self.result.state_identity_hash

    @property
    def execution_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "result_hash": self.result.result_hash,
                "forecast_hash": self.forecast.forecast_hash,
                "state_identity_hash": self.state_identity_hash,
                "production_cuda_execution": self.production_cuda_execution,
                "private_cpu_reference_execution": self.private_cpu_reference_execution,
            },
            prefix=_EXECUTION_PREFIX + "single-branch-",
        )


# ---------------------------------------------------------------------------
# Canonical fused-v2 implementation
# ---------------------------------------------------------------------------
#
# The first implementation above is retained temporarily in this unpromoted
# module as a replay-readable prototype.  The definitions below intentionally
# replace its public estimate/solve names.  They remove every production-scale
# defect of that prototype: the branches are fused into operator lanes, no
# route-scale block matrix or CPU sort is constructed, no route tensor is
# copied to the host, the true signed-work operator norm is registered, and
# the PDHG step is the largest representable strict step admitted by the
# certified operator bound rather than a tunable step-size heuristic.


@dataclass(frozen=True, slots=True)
class _FusedScales:
    work: float
    resource: float
    q_spectral: float
    q_diagonal: float
    block: tuple[float, ...]
    operator_norm_squared_upper: float
    step: float


@dataclass(frozen=True, slots=True)
class _FusedSourceTensors:
    device: Any
    resource: Any
    singleton_work: Any
    factor: Any
    residual_diagonal: Any
    sqrt_residual: Any
    eligible: Any
    required: Any
    residual_work: Any | None
    work_dual_box: Any
    block_bounds: Any
    has_ineligible: bool
    has_required: bool


def _canonical_fused_scales(
    problem: (
        V1SignedWorkLeastWorkCUDAProblem
        | V1SignedWorkLeastWorkCUDADeviceProblemV2
    ),
    *,
    active_branch: V1RelaxedKKTBranch | None = None,
) -> _FusedScales:
    # Exact native dispatch: no old source/FD receipt is fabricated.
    from .modern_v1_factored_spectral_least_work import V1FactoredSpectralLeastWorkProblem
    if type(problem) is V1FactoredSpectralLeastWorkProblem:
        return problem._native_scales(active_branch=active_branch)
    source = problem.source_problem
    operator = problem.signed_work_operator
    work_norm = operator.operator_frobenius_norm_upper
    work_scale = max(1.0, work_norm)
    if type(problem) is V1SignedWorkLeastWorkCUDADeviceProblemV2:
        problem.assert_integrity()
        resource_norm = float(torch.linalg.vector_norm(source.resource).item())
        if not math.isfinite(resource_norm):
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.NONFINITE,
                "device resource norm is not finite",
            )
    else:
        resource_norm = _stable_l2_norm(source.resource)
    resource_scale = max(1.0, resource_norm)
    factor_norm = float(source._factor_frobenius_norm)
    spectral_tail_norm = math.sqrt(source.residual_spectral_bound)
    diagonal_tail_norm = float(source._residual_sqrt_l2_norm)
    q_spectral = max(1.0, math.hypot(factor_norm, spectral_tail_norm))
    q_diagonal = max(1.0, math.hypot(factor_norm, diagonal_tail_norm))
    block_scales = tuple(
        math.sqrt(float(block.stop - block.start))
        for block in source.block_capacities
    )

    work_component = (work_norm / work_scale) ** 2
    resource_component = (resource_norm / resource_scale) ** 2
    q_spectral_component = (
        (factor_norm / q_spectral) ** 2
        + (spectral_tail_norm / q_spectral) ** 2
    )
    q_diagonal_component = (
        (factor_norm / q_diagonal) ** 2
        + (diagonal_tail_norm / q_diagonal) ** 2
    )
    # Compact blocks are disjoint, so their stacked normalized indicator
    # operator has norm at most one, independent of the number of blocks.
    block_component = 1.0 if block_scales else 0.0
    if active_branch is None:
        q_component = max(q_spectral_component, q_diagonal_component)
    elif active_branch is V1RelaxedKKTBranch.SPECTRAL:
        q_component = q_spectral_component
    elif active_branch is V1RelaxedKKTBranch.DIAGONAL_L1:
        q_component = q_diagonal_component
    else:
        raise TypeError("active Q branch must be exact")
    bound = (
        work_component
        + resource_component
        + q_component
        + block_component
    )
    if not math.isfinite(bound) or bound <= 0.0:
        _refuse(
            V1SignedWorkLeastWorkCUDARefusalCode.NONFINITE,
            "fused signed-work operator bound is not positive finite",
        )
    # Chambolle--Pock requires tau*sigma*||K||^2 < 1.  With tau=sigma, choose
    # the largest representable value that is strict under the same rounded
    # scalar product used by the receipt.  One inward IEEE-754 step below the
    # real boundary is sometimes still rounded back to equality when squared and
    # multiplied, so the loop is a numerical legality guard, not a tunable
    # step-size heuristic.
    boundary = 1.0 / math.sqrt(bound)
    step = math.nextafter(boundary, 0.0)
    while 0.0 < step < math.inf and not (step * step * bound < 1.0):
        next_step = math.nextafter(step, 0.0)
        if next_step == step:
            break
        step = next_step
    if not (0.0 < step < math.inf and step * step * bound < 1.0):
        _refuse(
            V1SignedWorkLeastWorkCUDARefusalCode.NONFINITE,
            "strict PDHG step is not representable",
        )
    return _FusedScales(
        work=work_scale,
        resource=resource_scale,
        q_spectral=q_spectral,
        q_diagonal=q_diagonal,
        block=block_scales,
        operator_norm_squared_upper=bound,
        step=step,
    )


def _fused_source_tensors(
    problem: (
        V1SignedWorkLeastWorkCUDAProblem
        | V1SignedWorkLeastWorkCUDADeviceProblemV2
    ),
    *,
    device: Any,
) -> _FusedSourceTensors:
    # Exact native dispatch: no old source/FD receipt is fabricated.
    from .modern_v1_factored_spectral_least_work import V1FactoredSpectralLeastWorkProblem
    if type(problem) is V1FactoredSpectralLeastWorkProblem:
        return problem._native_source_tensors(device=device)
    source = problem.source_problem
    operator = problem.signed_work_operator
    if type(problem) is V1SignedWorkLeastWorkCUDADeviceProblemV2:
        problem.assert_integrity()
        resource = source.resource
        singleton_work = source.corrected_work
        factor = source.low_rank_factor
        residual = source.residual_diagonal
        eligible = source.eligible_mask
        required = source.required_hard_action
        if (
            any(
                value.device != device
                for value in (
                    resource,
                    singleton_work,
                    factor,
                    residual,
                    eligible,
                    required,
                )
            )
            or resource.data_ptr() != problem._resource.data_ptr()
            or singleton_work.data_ptr()
            != problem._geometry_lease.singleton_work.data_ptr()
            or factor.data_ptr() != problem._geometry_lease.factor.data_ptr()
            or residual.data_ptr()
            != problem._geometry_lease.residual_diagonal.data_ptr()
        ):
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.CUSTODY_MUTATED,
                "device solve did not preserve zero-copy source pointers",
            )
        has_ineligible = source.has_ineligible
        has_required = source.has_required
    else:
        resource = torch.tensor(
            np.ascontiguousarray(source.resource),
            dtype=torch.float64,
            device=device,
        ).contiguous()
        singleton_work = torch.tensor(
            np.ascontiguousarray(source.corrected_work),
            dtype=torch.float64,
            device=device,
        ).contiguous()
        factor = torch.tensor(
            np.ascontiguousarray(source.low_rank_factor),
            dtype=torch.float64,
            device=device,
        ).contiguous()
        residual = torch.tensor(
            np.ascontiguousarray(source.residual_diagonal),
            dtype=torch.float64,
            device=device,
        ).contiguous()
        eligible = torch.tensor(
            np.ascontiguousarray(source.eligible_mask),
            dtype=torch.bool,
            device=device,
        ).contiguous()
        if problem.required_hard_action is None:
            required = torch.zeros(
                source.route_count, dtype=torch.bool, device=device
            )
        else:
            required = torch.tensor(
                np.ascontiguousarray(problem.required_hard_action),
                dtype=torch.bool,
                device=device,
            ).contiguous()
        has_ineligible = bool(np.any(~source.eligible_mask))
        has_required = (
            problem.required_hard_action is not None
            and bool(np.any(problem.required_hard_action))
        )
    cut_count = operator.native_fiber_axis_receipt.cut_count
    work_dual_box = (
        operator.context_weights.reshape(-1, 1, 1) / float(cut_count)
    )
    block_bounds = torch.as_tensor(
        tuple(
            block.removal_capacity
            / math.sqrt(float(block.stop - block.start))
            for block in source.block_capacities
        ),
        dtype=torch.float64,
        device=device,
    )
    return _FusedSourceTensors(
        device=device,
        resource=resource,
        singleton_work=singleton_work,
        factor=factor,
        residual_diagonal=residual,
        sqrt_residual=torch.sqrt(residual),
        eligible=eligible,
        required=required,
        residual_work=(
            None
            if operator.residual_is_logical_zero
            else operator.residual_linear / float(cut_count)
        ),
        work_dual_box=work_dual_box,
        block_bounds=block_bounds,
        has_ineligible=has_ineligible,
        has_required=has_required,
    )


def _block_forward_columns(
    source: V1RelaxedKKTProblem,
    scales: _FusedScales,
    columns: Any,
) -> Any:
    if not source.block_capacities:
        return torch.empty(
            (0, int(columns.shape[1])),
            dtype=torch.float64,
            device=columns.device,
        )
    return torch.stack(
        tuple(
            torch.sum(columns[block.start : block.stop, :], dim=0) / scale
            for block, scale in zip(
                source.block_capacities, scales.block, strict=True
            )
        ),
        dim=0,
    )


def _block_adjoint_columns(
    source: V1RelaxedKKTProblem,
    scales: _FusedScales,
    block_duals: Any,
    *,
    route_count: int,
) -> Any:
    result = torch.zeros(
        (route_count, int(block_duals.shape[1])),
        dtype=torch.float64,
        device=block_duals.device,
    )
    for row, (block, scale) in enumerate(
        zip(source.block_capacities, scales.block, strict=True)
    ):
        result[block.start : block.stop, :].add_(
            block_duals[row, :].reshape(1, -1) / scale
        )
    return result


def _uniform_resource_boundary_start(
    problem: V1SignedWorkLeastWorkCUDAProblem,
    tensors: _FusedSourceTensors,
) -> Any:
    """Build a legal, composition-neutral start at the resource boundary."""

    source = problem.source_problem
    required_f64 = tensors.required.to(dtype=torch.float64)
    free = tensors.eligible & ~tensors.required
    direction = torch.zeros_like(required_f64)
    if source.block_capacities:
        for block in source.block_capacities:
            local_free = free[block.start : block.stop]
            free_count = int(torch.count_nonzero(local_free).item())
            required_count = int(
                torch.count_nonzero(
                    tensors.required[block.start : block.stop]
                ).item()
            )
            available = block.removal_capacity - required_count
            if available < 0:
                _refuse(
                    V1SignedWorkLeastWorkCUDARefusalCode.INFEASIBLE_RESOURCE,
                    "required hard action exceeds a compact block capacity",
                )
            if free_count and available:
                direction[block.start : block.stop] = (
                    local_free.to(dtype=torch.float64)
                    * (float(available) / float(free_count))
                )
    else:
        direction = free.to(dtype=torch.float64)
    required_resource = float(torch.dot(tensors.resource, required_f64).item())
    missing = max(0.0, problem.resource_floor - required_resource)
    span_resource = float(torch.dot(tensors.resource, direction).item())
    tolerance = 128.0 * _EPS * max(
        1.0, problem.resource_floor, required_resource, span_resource
    )
    if missing > span_resource + tolerance:
        _refuse(
            V1SignedWorkLeastWorkCUDARefusalCode.INFEASIBLE_RESOURCE,
            "resource floor exceeds the live/selectable compact capacity",
        )
    fraction = 0.0 if missing <= 0.0 else min(1.0, missing / span_resource)
    return required_f64 + fraction * direction


def _native_work_columns(
    problem: V1SignedWorkLeastWorkCUDAProblem,
    tensors: _FusedSourceTensors,
    columns: Any,
) -> tuple[Any, Any]:
    operator = problem.signed_work_operator
    cells = operator._forward_prevalidated(columns)
    signed = torch.sum(
        tensors.work_dual_box * torch.abs(cells), dim=(0, 1)
    )
    if tensors.residual_work is None:
        residual = torch.zeros(
            int(columns.shape[1]), dtype=torch.float64, device=columns.device
        )
    else:
        residual = torch.matmul(tensors.residual_work.reshape(1, -1), columns).reshape(-1)
    return signed + residual, cells


def _q_columns(
    source: V1RelaxedKKTProblem,
    tensors: _FusedSourceTensors,
    columns: Any,
) -> tuple[Any, Any]:
    low = tensors.factor @ columns
    low_q = torch.sum(low * low, dim=0)
    spectral = low_q + source.residual_spectral_bound * torch.sum(
        columns * columns, dim=0
    )
    diagonal_projection = torch.matmul(
        tensors.sqrt_residual.reshape(1, -1), columns
    ).reshape(-1)
    diagonal = low_q + diagonal_projection * diagonal_projection
    return torch.stack((spectral, diagonal), dim=0), low


def _q_single_branch_columns(
    source: V1RelaxedKKTProblem,
    tensors: _FusedSourceTensors,
    columns: Any,
    *,
    branch: V1RelaxedKKTBranch,
) -> tuple[Any, Any]:
    """Evaluate only the Q envelope already authorized by Phase A."""

    low = tensors.factor @ columns
    low_q = torch.sum(low * low, dim=0)
    if branch is V1RelaxedKKTBranch.SPECTRAL:
        q = low_q + source.residual_spectral_bound * torch.sum(
            columns * columns,
            dim=0,
        )
    elif branch is V1RelaxedKKTBranch.DIAGONAL_L1:
        diagonal_projection = torch.matmul(
            tensors.sqrt_residual.reshape(1, -1),
            columns,
        ).reshape(-1)
        q = low_q + diagonal_projection * diagonal_projection
    else:  # pragma: no cover - exact public type checks close this path.
        raise TypeError("unknown signed-work Q branch")
    return q, low


def _candidate_metric_columns(
    problem: V1SignedWorkLeastWorkCUDAProblem,
    tensors: _FusedSourceTensors,
    scales: _FusedScales,
    columns: Any,
    branch_columns: tuple[V1RelaxedKKTBranch, ...],
) -> tuple[np.ndarray, Any, Any]:
    source = problem.source_problem
    work, cells = _native_work_columns(problem, tensors, columns)
    unique_branches = tuple(dict.fromkeys(branch_columns))
    if len(unique_branches) == 1:
        q, low = _q_single_branch_columns(
            source,
            tensors,
            columns,
            branch=unique_branches[0],
        )
    else:
        q_both, low = _q_columns(source, tensors, columns)
        q = torch.stack(
            tuple(
                q_both[
                    0 if branch is V1RelaxedKKTBranch.SPECTRAL else 1,
                    column,
                ]
                for column, branch in enumerate(branch_columns)
            )
        )
    resource = torch.matmul(tensors.resource.reshape(1, -1), columns).reshape(-1)
    block_values = _block_forward_columns(source, scales, columns)
    if int(block_values.numel()):
        block_violation = torch.max(
            torch.clamp(
                block_values - tensors.block_bounds.reshape(-1, 1), min=0.0
            ),
            dim=0,
        ).values
    else:
        block_violation = torch.zeros_like(resource)
    lower_box = torch.clamp(-columns, min=0.0)
    upper_box = torch.clamp(columns - 1.0, min=0.0)
    box_violation = torch.maximum(
        torch.max(lower_box, dim=0).values,
        torch.max(upper_box, dim=0).values,
    )
    if tensors.has_ineligible:
        box_violation = torch.maximum(
            box_violation,
            torch.max(torch.abs(columns[~tensors.eligible, :]), dim=0).values,
        )
    if tensors.has_required:
        box_violation = torch.maximum(
            box_violation,
            torch.max(
                torch.abs(columns[tensors.required, :] - 1.0), dim=0
            ).values,
        )
    resource_violation = torch.clamp(problem.resource_floor - resource, min=0.0)
    q_violation = torch.clamp(q - source.quadratic_budget, min=0.0)
    normalized = torch.maximum(
        box_violation,
        torch.maximum(
            block_violation,
            torch.maximum(
                resource_violation / max(1.0, problem.resource_floor),
                q_violation / max(1.0, source.quadratic_budget),
            ),
        ),
    )
    packed = torch.stack(
        (work, resource, q, box_violation, block_violation, normalized), dim=1
    ).detach().cpu().numpy()
    return packed, cells, low


def _q_dual_adjoint(
    problem: V1SignedWorkLeastWorkCUDAProblem,
    tensors: _FusedSourceTensors,
    scales: _FusedScales,
    low_dual: Any,
    spectral_tail_dual: Any,
    diagonal_tail_dual: Any,
) -> Any:
    scaled_low_dual = low_dual / torch.as_tensor(
        (scales.q_spectral, scales.q_diagonal),
        dtype=torch.float64,
        device=tensors.device,
    ).reshape(1, 2)
    result = tensors.factor.T @ scaled_low_dual
    result[:, 0].add_(
        spectral_tail_dual
        * (math.sqrt(problem.source_problem.residual_spectral_bound) / scales.q_spectral)
    )
    result[:, 1].add_(
        tensors.sqrt_residual * (diagonal_tail_dual / scales.q_diagonal)
    )
    return result


def _empty_branch_receipt(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    branch: V1RelaxedKKTBranch,
    scales: _FusedScales,
    iterations: int,
    reason: str,
    finite: bool,
    production: bool,
    private_cpu: bool,
) -> V1SignedWorkLeastWorkCUDABranchReceipt:
    return V1SignedWorkLeastWorkCUDABranchReceipt(
        branch=branch,
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        operator_receipt_hash=problem.signed_work_operator.receipt_hash,
        converged=False,
        finite_iterates=finite,
        iterations=iterations,
        termination_reason=reason,
        resource_value=0.0,
        resource_floor=problem.resource_floor,
        signed_work_upper=math.inf,
        quadratic_value=math.inf,
        maximum_box_violation=math.inf,
        maximum_block_violation=math.inf,
        normalized_primal_violation=math.inf,
        dual_work_lower=-math.inf,
        absolute_duality_gap=math.inf,
        relative_duality_gap=math.inf,
        stationarity_gap=math.inf,
        normalized_stationarity_gap=math.inf,
        work_fenchel_gap=math.inf,
        resource_complementarity_gap=math.inf,
        quadratic_complementarity_gap=math.inf,
        block_complementarity_gap=math.inf,
        normalized_complementarity_gap=math.inf,
        zero_work_cell_count=0,
        maximum_work_dual_box_violation=math.inf,
        primal_step_size=scales.step,
        dual_step_size=scales.step,
        scaled_operator_norm_squared_upper_bound=(
            scales.operator_norm_squared_upper
        ),
        device=str(_operator_device_object(problem.signed_work_operator)),
        production_cuda_execution=production,
        private_cpu_reference_execution=private_cpu,
    )


def _issue_fused_branch_receipts(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    tensors: _FusedSourceTensors,
    scales: _FusedScales,
    best: Any,
    best_available: tuple[bool, bool],
    work_dual: Any,
    low_dual: Any,
    spectral_tail_dual: Any,
    diagonal_tail_dual: Any,
    resource_dual: Any,
    block_dual: Any,
    iterations: int,
    finite: bool,
    production: bool,
    private_cpu: bool,
) -> tuple[
    tuple[V1SignedWorkLeastWorkCUDABranchReceipt, ...],
    Any,
]:
    source = problem.source_problem
    work_values, cells = _native_work_columns(problem, tensors, best)
    q_both, low = _q_columns(source, tensors, best)
    work_adjoint = (
        problem.signed_work_operator._adjoint_prevalidated(work_dual)
        / scales.work
    )
    q_adjoint = _q_dual_adjoint(
        problem,
        tensors,
        scales,
        low_dual,
        spectral_tail_dual,
        diagonal_tail_dual,
    )
    block_adjoint = _block_adjoint_columns(
        source,
        scales,
        block_dual,
        route_count=source.route_count,
    )
    gradient = (
        work_adjoint
        - tensors.resource.reshape(-1, 1)
        * (resource_dual / scales.resource).reshape(1, 2)
        + q_adjoint
        + block_adjoint
    )
    if tensors.residual_work is not None:
        gradient = gradient + tensors.residual_work.reshape(-1, 1) / scales.work
    reduced_advantage = -gradient * scales.work
    resource_values = torch.matmul(
        tensors.resource.reshape(1, -1), best
    ).reshape(-1)
    block_values = _block_forward_columns(source, scales, best)
    free = tensors.eligible & ~tensors.required
    q_scales = (scales.q_spectral, scales.q_diagonal)
    q_radii = (
        math.sqrt(source.quadratic_budget) / scales.q_spectral,
        math.sqrt(source.quadratic_budget) / scales.q_diagonal,
    )

    receipts: list[V1SignedWorkLeastWorkCUDABranchReceipt] = []
    for column, branch in enumerate(SIGNED_WORK_LEAST_WORK_BRANCH_ORDER):
        if not best_available[column]:
            receipts.append(
                _empty_branch_receipt(
                    problem=problem,
                    policy=policy,
                    branch=branch,
                    scales=scales,
                    iterations=iterations,
                    reason="no_primal_feasible_fixed_resource_candidate",
                    finite=finite,
                    production=production,
                    private_cpu=private_cpu,
                )
            )
            continue
        z = best[:, column]
        native_upper_t = work_values[column]
        resource_t = resource_values[column]
        q_t = q_both[0 if column == 0 else 1, column]
        block_column = block_values[:, column]
        box_violation_t = torch.maximum(
            torch.clamp(-torch.min(z), min=0.0),
            torch.clamp(torch.max(z - 1.0), min=0.0),
        )
        if tensors.has_ineligible:
            box_violation_t = torch.maximum(
                box_violation_t, torch.max(torch.abs(z[~tensors.eligible]))
            )
        if tensors.has_required:
            box_violation_t = torch.maximum(
                box_violation_t,
                torch.max(torch.abs(z[tensors.required] - 1.0)),
            )
        if int(block_column.numel()):
            block_violation_t = torch.max(
                torch.clamp(block_column - tensors.block_bounds, min=0.0)
            )
        else:
            block_violation_t = torch.zeros_like(resource_t)
        primal_violation_t = torch.maximum(
            box_violation_t,
            torch.maximum(
                block_violation_t,
                torch.maximum(
                    torch.clamp(problem.resource_floor - resource_t, min=0.0)
                    / max(1.0, problem.resource_floor),
                    torch.clamp(q_t - source.quadratic_budget, min=0.0)
                    / max(1.0, source.quadratic_budget),
                ),
            ),
        )

        d = gradient[:, column]
        infimum = torch.sum(torch.minimum(torch.zeros_like(d[free]), d[free]))
        if tensors.has_required:
            infimum = infimum + torch.sum(d[tensors.required])
        if branch is V1RelaxedKKTBranch.SPECTRAL:
            tail_dual = spectral_tail_dual
            q_dual_norm = torch.sqrt(
                torch.dot(low_dual[:, column], low_dual[:, column])
                + torch.dot(tail_dual, tail_dual)
            )
            q_pairing = (
                torch.dot(low_dual[:, column], low[:, column])
                + math.sqrt(source.residual_spectral_bound)
                * torch.dot(tail_dual, z)
            ) / q_scales[column]
        else:
            tail_dual = diagonal_tail_dual
            q_dual_norm = torch.sqrt(
                torch.dot(low_dual[:, column], low_dual[:, column])
                + tail_dual * tail_dual
            )
            q_pairing = (
                torch.dot(low_dual[:, column], low[:, column])
                + tail_dual * torch.dot(tensors.sqrt_residual, z)
            ) / q_scales[column]
        block_bound_pair = (
            torch.dot(block_dual[:, column], tensors.block_bounds)
            if int(block_dual.shape[0])
            else torch.zeros((), dtype=torch.float64, device=tensors.device)
        )
        lower_scaled = (
            resource_dual[column]
            * (problem.resource_floor / scales.resource)
            - q_radii[column] * q_dual_norm
            - block_bound_pair
            + infimum
        )
        lower_native_t = lower_scaled * scales.work
        upper_scaled = native_upper_t / scales.work
        gap_scaled = upper_scaled - lower_scaled
        denominator_t = torch.maximum(
            torch.ones((), dtype=torch.float64, device=tensors.device),
            torch.maximum(torch.abs(upper_scaled), torch.abs(lower_scaled)),
        )
        signed_without_residual = (
            torch.sum(
                tensors.work_dual_box[:, :, 0] * torch.abs(cells[:, :, column])
            )
            / scales.work
        )
        work_pairing = torch.sum(
            work_dual[:, :, column] * cells[:, :, column]
        ) / scales.work
        work_fenchel_t = torch.clamp(
            signed_without_residual - work_pairing, min=0.0
        )
        resource_comp_t = torch.clamp(
            resource_dual[column]
            * (resource_t - problem.resource_floor)
            / scales.resource,
            min=0.0,
        )
        q_comp_t = torch.clamp(
            q_radii[column] * q_dual_norm - q_pairing, min=0.0
        )
        block_comp_t = (
            torch.clamp(
                torch.dot(
                    block_dual[:, column],
                    tensors.block_bounds - block_column,
                ),
                min=0.0,
            )
            if int(block_dual.shape[0])
            else torch.zeros((), dtype=torch.float64, device=tensors.device)
        )
        stationarity_t = torch.clamp(torch.dot(d, z) - infimum, min=0.0)
        complementarity_t = (
            work_fenchel_t + resource_comp_t + q_comp_t + block_comp_t
        )
        max_dual_violation_t = torch.max(
            torch.clamp(
                torch.abs(work_dual[:, :, column])
                - tensors.work_dual_box[:, :, 0],
                min=0.0,
            )
        )
        values = torch.stack(
            (
                native_upper_t,
                resource_t,
                q_t,
                box_violation_t,
                block_violation_t,
                primal_violation_t,
                lower_native_t,
                gap_scaled * scales.work,
                denominator_t,
                stationarity_t * scales.work,
                work_fenchel_t * scales.work,
                resource_comp_t * scales.work,
                q_comp_t * scales.work,
                block_comp_t * scales.work,
                complementarity_t * scales.work,
                max_dual_violation_t,
            )
        ).detach().cpu().numpy()
        (
            upper_v,
            resource_v,
            q_v,
            box_v,
            block_v,
            primal_v,
            lower_v,
            gap_v,
            denominator_v,
            stationarity_v,
            work_fenchel_v,
            resource_comp_v,
            q_comp_v,
            block_comp_v,
            complementarity_v,
            max_dual_v,
        ) = (float(item) for item in values)
        native_denominator = max(1.0, abs(upper_v), abs(lower_v))
        relative_gap = max(0.0, gap_v) / native_denominator
        normalized_stationarity = stationarity_v / native_denominator
        normalized_complementarity = complementarity_v / native_denominator
        gap_bound_valid = gap_v >= -(
            policy.absolute_gap_tolerance
            + policy.relative_gap_tolerance * native_denominator
        )
        gap_ok = max(0.0, gap_v) <= (
            policy.absolute_gap_tolerance
            + policy.relative_gap_tolerance * native_denominator
        )
        converged = all(
            (
                finite,
                iterations >= policy.minimum_iterations,
                primal_v <= policy.primal_feasibility_tolerance,
                gap_bound_valid,
                gap_ok,
                normalized_stationarity
                <= policy.normalized_stationarity_tolerance,
                normalized_complementarity
                <= policy.normalized_complementarity_tolerance,
                max_dual_v <= policy.normalized_stationarity_tolerance,
            )
        )
        receipts.append(
            V1SignedWorkLeastWorkCUDABranchReceipt(
                branch=branch,
                problem_hash=problem.problem_hash,
                policy_hash=policy.policy_hash,
                operator_receipt_hash=problem.signed_work_operator.receipt_hash,
                converged=converged,
                finite_iterates=finite,
                iterations=iterations,
                termination_reason=(
                    "certified_relaxed_branch_optimum"
                    if converged
                    else "registered_primal_dual_tolerances_unresolved"
                ),
                resource_value=resource_v,
                resource_floor=problem.resource_floor,
                signed_work_upper=upper_v,
                quadratic_value=q_v,
                maximum_box_violation=box_v,
                maximum_block_violation=block_v,
                normalized_primal_violation=primal_v,
                dual_work_lower=lower_v,
                absolute_duality_gap=max(0.0, gap_v),
                relative_duality_gap=relative_gap,
                stationarity_gap=stationarity_v,
                normalized_stationarity_gap=normalized_stationarity,
                work_fenchel_gap=work_fenchel_v,
                resource_complementarity_gap=resource_comp_v,
                quadratic_complementarity_gap=q_comp_v,
                block_complementarity_gap=block_comp_v,
                normalized_complementarity_gap=normalized_complementarity,
                zero_work_cell_count=int(
                    torch.count_nonzero(cells[:, :, column] == 0.0).item()
                ),
                maximum_work_dual_box_violation=max_dual_v,
                primal_step_size=scales.step,
                dual_step_size=scales.step,
                scaled_operator_norm_squared_upper_bound=(
                    scales.operator_norm_squared_upper
                ),
                device=str(tensors.device),
                production_cuda_execution=production,
                private_cpu_reference_execution=private_cpu,
            )
        )
    return tuple(receipts), reduced_advantage


def _single_q_dual_adjoint(
    problem: V1SignedWorkLeastWorkCUDAProblem,
    tensors: _FusedSourceTensors,
    scales: _FusedScales,
    *,
    branch: V1RelaxedKKTBranch,
    low_dual: Any,
    tail_dual: Any,
) -> Any:
    q_scale = (
        scales.q_spectral
        if branch is V1RelaxedKKTBranch.SPECTRAL
        else scales.q_diagonal
    )
    result = tensors.factor.T @ (low_dual / q_scale)
    if branch is V1RelaxedKKTBranch.SPECTRAL:
        result[:, 0].add_(
            tail_dual
            * (math.sqrt(problem.source_problem.residual_spectral_bound) / q_scale)
        )
    elif branch is V1RelaxedKKTBranch.DIAGONAL_L1:
        result[:, 0].add_(tensors.sqrt_residual * (tail_dual / q_scale))
    else:  # pragma: no cover - exact public type checks close this path.
        raise TypeError("unknown signed-work Q branch")
    return result


def _issue_single_branch_receipt(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    tensors: _FusedSourceTensors,
    scales: _FusedScales,
    branch: V1RelaxedKKTBranch,
    best: Any,
    best_available: bool,
    work_dual: Any,
    low_dual: Any,
    tail_dual: Any,
    resource_dual: Any,
    block_dual: Any,
    iterations: int,
    finite: bool,
    production: bool,
    private_cpu: bool,
) -> tuple[V1SignedWorkLeastWorkCUDABranchReceipt, Any]:
    """Certify one branch without constructing or evaluating its complement."""

    source = problem.source_problem
    work_values, cells = _native_work_columns(problem, tensors, best)
    q_selected, low = _q_single_branch_columns(
        source,
        tensors,
        best,
        branch=branch,
    )
    work_adjoint = (
        problem.signed_work_operator._adjoint_prevalidated(work_dual)
        / scales.work
    )
    q_adjoint = _single_q_dual_adjoint(
        problem,
        tensors,
        scales,
        branch=branch,
        low_dual=low_dual,
        tail_dual=tail_dual,
    )
    block_adjoint = _block_adjoint_columns(
        source,
        scales,
        block_dual,
        route_count=source.route_count,
    )
    gradient = (
        work_adjoint
        - tensors.resource.reshape(-1, 1)
        * (resource_dual / scales.resource).reshape(1, 1)
        + q_adjoint
        + block_adjoint
    )
    if tensors.residual_work is not None:
        gradient = gradient + tensors.residual_work.reshape(-1, 1) / scales.work
    reduced_advantage = -gradient * scales.work
    if not best_available:
        return (
            _empty_branch_receipt(
                problem=problem,
                policy=policy,
                branch=branch,
                scales=scales,
                iterations=iterations,
                reason="no_primal_feasible_fixed_resource_candidate",
                finite=finite,
                production=production,
                private_cpu=private_cpu,
            ),
            reduced_advantage,
        )

    z = best[:, 0]
    native_upper_t = work_values[0]
    resource_t = torch.dot(tensors.resource, z)
    q_t = q_selected[0]
    block_values = _block_forward_columns(source, scales, best)[:, 0]
    box_violation_t = torch.maximum(
        torch.clamp(-torch.min(z), min=0.0),
        torch.clamp(torch.max(z - 1.0), min=0.0),
    )
    if tensors.has_ineligible:
        box_violation_t = torch.maximum(
            box_violation_t,
            torch.max(torch.abs(z[~tensors.eligible])),
        )
    if tensors.has_required:
        box_violation_t = torch.maximum(
            box_violation_t,
            torch.max(torch.abs(z[tensors.required] - 1.0)),
        )
    if int(block_values.numel()):
        block_violation_t = torch.max(
            torch.clamp(block_values - tensors.block_bounds, min=0.0)
        )
    else:
        block_violation_t = torch.zeros_like(resource_t)
    primal_violation_t = torch.maximum(
        box_violation_t,
        torch.maximum(
            block_violation_t,
            torch.maximum(
                torch.clamp(problem.resource_floor - resource_t, min=0.0)
                / max(1.0, problem.resource_floor),
                torch.clamp(q_t - source.quadratic_budget, min=0.0)
                / max(1.0, source.quadratic_budget),
            ),
        ),
    )

    free = tensors.eligible & ~tensors.required
    d = gradient[:, 0]
    infimum = torch.sum(torch.minimum(torch.zeros_like(d[free]), d[free]))
    if tensors.has_required:
        infimum = infimum + torch.sum(d[tensors.required])
    q_scale = (
        scales.q_spectral
        if branch is V1RelaxedKKTBranch.SPECTRAL
        else scales.q_diagonal
    )
    q_radius = math.sqrt(source.quadratic_budget) / q_scale
    if branch is V1RelaxedKKTBranch.SPECTRAL:
        q_dual_norm = torch.sqrt(
            torch.dot(low_dual[:, 0], low_dual[:, 0])
            + torch.dot(tail_dual, tail_dual)
        )
        q_pairing = (
            torch.dot(low_dual[:, 0], low[:, 0])
            + math.sqrt(source.residual_spectral_bound)
            * torch.dot(tail_dual, z)
        ) / q_scale
    else:
        q_dual_norm = torch.sqrt(
            torch.dot(low_dual[:, 0], low_dual[:, 0])
            + tail_dual * tail_dual
        )
        q_pairing = (
            torch.dot(low_dual[:, 0], low[:, 0])
            + tail_dual * torch.dot(tensors.sqrt_residual, z)
        ) / q_scale
    block_bound_pair = (
        torch.dot(block_dual[:, 0], tensors.block_bounds)
        if int(block_dual.shape[0])
        else torch.zeros((), dtype=torch.float64, device=tensors.device)
    )
    lower_scaled = (
        resource_dual[0] * (problem.resource_floor / scales.resource)
        - q_radius * q_dual_norm
        - block_bound_pair
        + infimum
    )
    lower_native_t = lower_scaled * scales.work
    upper_scaled = native_upper_t / scales.work
    gap_scaled = upper_scaled - lower_scaled
    denominator_t = torch.maximum(
        torch.ones((), dtype=torch.float64, device=tensors.device),
        torch.maximum(torch.abs(upper_scaled), torch.abs(lower_scaled)),
    )
    signed_without_residual = (
        torch.sum(tensors.work_dual_box[:, :, 0] * torch.abs(cells[:, :, 0]))
        / scales.work
    )
    work_pairing = (
        torch.sum(work_dual[:, :, 0] * cells[:, :, 0]) / scales.work
    )
    work_fenchel_t = torch.clamp(
        signed_without_residual - work_pairing,
        min=0.0,
    )
    resource_comp_t = torch.clamp(
        resource_dual[0]
        * (resource_t - problem.resource_floor)
        / scales.resource,
        min=0.0,
    )
    q_comp_t = torch.clamp(q_radius * q_dual_norm - q_pairing, min=0.0)
    block_comp_t = (
        torch.clamp(
            torch.dot(
                block_dual[:, 0],
                tensors.block_bounds - block_values,
            ),
            min=0.0,
        )
        if int(block_dual.shape[0])
        else torch.zeros((), dtype=torch.float64, device=tensors.device)
    )
    stationarity_t = torch.clamp(torch.dot(d, z) - infimum, min=0.0)
    complementarity_t = (
        work_fenchel_t + resource_comp_t + q_comp_t + block_comp_t
    )
    max_dual_violation_t = torch.max(
        torch.clamp(
            torch.abs(work_dual[:, :, 0]) - tensors.work_dual_box[:, :, 0],
            min=0.0,
        )
    )
    values = torch.stack(
        (
            native_upper_t,
            resource_t,
            q_t,
            box_violation_t,
            block_violation_t,
            primal_violation_t,
            lower_native_t,
            gap_scaled * scales.work,
            denominator_t,
            stationarity_t * scales.work,
            work_fenchel_t * scales.work,
            resource_comp_t * scales.work,
            q_comp_t * scales.work,
            block_comp_t * scales.work,
            complementarity_t * scales.work,
            max_dual_violation_t,
        )
    ).detach().cpu().numpy()
    (
        upper_v,
        resource_v,
        q_v,
        box_v,
        block_v,
        primal_v,
        lower_v,
        gap_v,
        _denominator_v,
        stationarity_v,
        work_fenchel_v,
        resource_comp_v,
        q_comp_v,
        block_comp_v,
        complementarity_v,
        max_dual_v,
    ) = (float(item) for item in values)
    native_denominator = max(1.0, abs(upper_v), abs(lower_v))
    relative_gap = max(0.0, gap_v) / native_denominator
    normalized_stationarity = stationarity_v / native_denominator
    normalized_complementarity = complementarity_v / native_denominator
    gap_bound_valid = gap_v >= -(
        policy.absolute_gap_tolerance
        + policy.relative_gap_tolerance * native_denominator
    )
    gap_ok = max(0.0, gap_v) <= (
        policy.absolute_gap_tolerance
        + policy.relative_gap_tolerance * native_denominator
    )
    converged = all(
        (
            finite,
            iterations >= policy.minimum_iterations,
            primal_v <= policy.primal_feasibility_tolerance,
            gap_bound_valid,
            gap_ok,
            normalized_stationarity <= policy.normalized_stationarity_tolerance,
            normalized_complementarity
            <= policy.normalized_complementarity_tolerance,
            max_dual_v <= policy.normalized_stationarity_tolerance,
        )
    )
    receipt = V1SignedWorkLeastWorkCUDABranchReceipt(
        branch=branch,
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        operator_receipt_hash=problem.signed_work_operator.receipt_hash,
        converged=converged,
        finite_iterates=finite,
        iterations=iterations,
        termination_reason=(
            "certified_relaxed_branch_optimum"
            if converged
            else "registered_primal_dual_tolerances_unresolved"
        ),
        resource_value=resource_v,
        resource_floor=problem.resource_floor,
        signed_work_upper=upper_v,
        quadratic_value=q_v,
        maximum_box_violation=box_v,
        maximum_block_violation=block_v,
        normalized_primal_violation=primal_v,
        dual_work_lower=lower_v,
        absolute_duality_gap=max(0.0, gap_v),
        relative_duality_gap=relative_gap,
        stationarity_gap=stationarity_v,
        normalized_stationarity_gap=normalized_stationarity,
        work_fenchel_gap=work_fenchel_v,
        resource_complementarity_gap=resource_comp_v,
        quadratic_complementarity_gap=q_comp_v,
        block_complementarity_gap=block_comp_v,
        normalized_complementarity_gap=normalized_complementarity,
        zero_work_cell_count=int(
            torch.count_nonzero(cells[:, :, 0] == 0.0).item()
        ),
        maximum_work_dual_box_violation=max_dual_v,
        primal_step_size=scales.step,
        dual_step_size=scales.step,
        scaled_operator_norm_squared_upper_bound=(
            scales.operator_norm_squared_upper
        ),
        device=str(tensors.device),
        production_cuda_execution=production,
        private_cpu_reference_execution=private_cpu,
    )
    return receipt, reduced_advantage


def _solve_independent_signed_work_branch(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    branch: V1RelaxedKKTBranch,
    device: Any,
    production: bool,
    private_cpu: bool,
    initial_primal: Any | None = None,
) -> tuple[
    V1SignedWorkLeastWorkCUDABranchReceipt,
    Any,
    Any,
    _FusedSourceTensors,
    _SignedWorkDualStateTensors,
]:
    """Run one branch as a one-lane convex program.

    No tensor, dual, candidate metric, or stopping decision for the other Q
    envelope is constructed.  This is the numerical seam required by closed
    Phase-A branch membership.
    """

    source = problem.source_problem
    operator = problem.signed_work_operator
    tensors = _fused_source_tensors(problem, device=device)
    scales = _canonical_fused_scales(problem, active_branch=branch)
    # Only an issued native Phase-A copy may seed a native least-work solve.
    # Old callers retain their original cold-start behavior and cannot inject
    # arbitrary initial points through this private native seam.
    warm = initial_primal is not None
    if warm:
        from .modern_v1_factored_spectral_least_work import V1FactoredSpectralLeastWorkProblem
        if (type(problem) is not V1FactoredSpectralLeastWorkProblem
                or branch is not V1RelaxedKKTBranch.SPECTRAL):
            raise TypeError("initial primal requires the exact native spectral problem")
        start = problem._validated_initial_primal(initial_primal).clone()
    else:
        start = _uniform_resource_boundary_start(problem, tensors)
    z = start.reshape(-1, 1).contiguous()
    z_bar = z.clone()
    average = z.clone() if warm else torch.zeros_like(z)
    best = z.clone()
    best_available = warm
    best_work = (float(_native_work_columns(problem, tensors, best)[0][0].item())
                 if warm else math.inf)
    work_dual = torch.zeros(
        (operator.context_count, operator.fiber_count, 1),
        dtype=torch.float64,
        device=device,
    )
    low_dual = torch.zeros((source.rank, 1), dtype=torch.float64, device=device)
    tail_dual = (
        torch.zeros(source.route_count, dtype=torch.float64, device=device)
        if branch is V1RelaxedKKTBranch.SPECTRAL
        else torch.zeros((), dtype=torch.float64, device=device)
    )
    resource_dual = torch.zeros(1, dtype=torch.float64, device=device)
    block_dual = torch.zeros(
        (len(source.block_capacities), 1),
        dtype=torch.float64,
        device=device,
    )
    q_scale = (
        scales.q_spectral
        if branch is V1RelaxedKKTBranch.SPECTRAL
        else scales.q_diagonal
    )
    q_radius = math.sqrt(source.quadratic_budget) / q_scale
    step = scales.step
    finite = True
    final_receipt = _empty_branch_receipt(
        problem=problem,
        policy=policy,
        branch=branch,
        scales=scales,
        iterations=0,
        reason="solver_not_started",
        finite=True,
        production=production,
        private_cpu=private_cpu,
    )
    reduced = torch.zeros_like(z)
    iterations = 0
    for iteration in range(1, policy.max_iterations + 1):
        iterations = iteration
        cells = operator._forward_prevalidated(z_bar)
        work_dual = torch.clamp(
            work_dual + step * cells / scales.work,
            min=-tensors.work_dual_box,
            max=tensors.work_dual_box,
        )
        resource_projection = torch.dot(tensors.resource, z_bar[:, 0])
        resource_dual = torch.clamp(
            resource_dual
            + step
            * (problem.resource_floor - resource_projection)
            / scales.resource,
            min=0.0,
        )
        block_projection = _block_forward_columns(source, scales, z_bar)
        if int(block_dual.shape[0]):
            block_dual = torch.clamp(
                block_dual
                + step
                * (block_projection - tensors.block_bounds.reshape(-1, 1)),
                min=0.0,
            )
        low_candidate = low_dual + step * (tensors.factor @ z_bar) / q_scale
        if branch is V1RelaxedKKTBranch.SPECTRAL:
            tail_candidate = tail_dual + (
                step
                * math.sqrt(source.residual_spectral_bound)
                * z_bar[:, 0]
                / q_scale
            )
            q_norm = torch.sqrt(
                torch.dot(low_candidate[:, 0], low_candidate[:, 0])
                + torch.dot(tail_candidate, tail_candidate)
            )
        else:
            tail_candidate = tail_dual + (
                step * torch.dot(tensors.sqrt_residual, z_bar[:, 0]) / q_scale
            )
            q_norm = torch.sqrt(
                torch.dot(low_candidate[:, 0], low_candidate[:, 0])
                + tail_candidate * tail_candidate
            )
        shrink = torch.clamp(
            1.0
            - step
            * q_radius
            / torch.clamp(q_norm, min=torch.finfo(torch.float64).tiny),
            min=0.0,
        )
        low_dual = low_candidate * shrink
        tail_dual = tail_candidate * shrink

        work_adjoint = operator._adjoint_prevalidated(work_dual) / scales.work
        q_adjoint = _single_q_dual_adjoint(
            problem,
            tensors,
            scales,
            branch=branch,
            low_dual=low_dual,
            tail_dual=tail_dual,
        )
        block_adjoint = _block_adjoint_columns(
            source,
            scales,
            block_dual,
            route_count=source.route_count,
        )
        gradient = (
            work_adjoint
            - tensors.resource.reshape(-1, 1)
            * (resource_dual / scales.resource).reshape(1, 1)
            + q_adjoint
            + block_adjoint
        )
        if tensors.residual_work is not None:
            gradient = gradient + tensors.residual_work.reshape(-1, 1) / scales.work
        z_new = torch.clamp(z - step * gradient, min=0.0, max=1.0)
        z_new = z_new * tensors.eligible.reshape(-1, 1)
        if tensors.has_required:
            z_new[tensors.required, :] = 1.0
        z_bar = z_new + policy.over_relaxation * (z_new - z)
        z = z_new
        average = average + (z - average) / float(iteration)
        should_check = (
            iteration % policy.check_interval == 0
            or iteration == policy.max_iterations
        )
        if not should_check:
            continue
        finite = bool(
            torch.isfinite(z).all().item()
            and torch.isfinite(z_bar).all().item()
            and torch.isfinite(work_dual).all().item()
            and torch.isfinite(low_dual).all().item()
            and torch.isfinite(tail_dual).all().item()
            and torch.isfinite(resource_dual).all().item()
            and torch.isfinite(block_dual).all().item()
        )
        if not finite:
            break
        candidates = torch.cat((z, average), dim=1)
        metrics, _candidate_cells, _candidate_low = _candidate_metric_columns(
            problem,
            tensors,
            scales,
            candidates,
            (branch, branch),
        )
        del _candidate_cells, _candidate_low
        for proposal_column in range(2):
            work_value = float(metrics[proposal_column, 0])
            normalized_violation = float(metrics[proposal_column, 5])
            if (
                normalized_violation <= policy.primal_feasibility_tolerance
                and work_value < best_work
                and (not warm or problem._admit_native_improving_candidate(
                    candidates[:, proposal_column:proposal_column+1]))
            ):
                best[:, 0].copy_(candidates[:, proposal_column])
                best_work = work_value
                best_available = True
        del candidates
        del work_adjoint, q_adjoint, block_adjoint, gradient, z_new
        final_receipt, reduced = _issue_single_branch_receipt(
            problem=problem,
            policy=policy,
            tensors=tensors,
            scales=scales,
            branch=branch,
            best=best,
            best_available=best_available,
            work_dual=work_dual,
            low_dual=low_dual,
            tail_dual=tail_dual,
            resource_dual=resource_dual,
            block_dual=block_dual,
            iterations=iteration,
            finite=finite,
            production=production,
            private_cpu=private_cpu,
        )
        if final_receipt.converged:
            break
    if not finite:
        final_receipt = _empty_branch_receipt(
            problem=problem,
            policy=policy,
            branch=branch,
            scales=scales,
            iterations=iterations,
            reason="nonfinite_iterate_refusal",
            finite=False,
            production=production,
            private_cpu=private_cpu,
        )
    return (
        final_receipt,
        best,
        reduced,
        tensors,
        _SignedWorkDualStateTensors(
            branch_order=(branch,),
            work_dual=work_dual,
            low_rank_dual=low_dual,
            q_tail_duals=(tail_dual,),
            resource_dual=resource_dual,
            block_dual=block_dual,
        ),
    )


def _solve_fused_signed_work_branches(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    device: Any,
    production: bool,
    private_cpu: bool,
) -> tuple[
    tuple[V1SignedWorkLeastWorkCUDABranchReceipt, ...],
    Any,
    Any,
    _FusedSourceTensors,
    _SignedWorkDualStateTensors,
]:
    source = problem.source_problem
    operator = problem.signed_work_operator
    tensors = _fused_source_tensors(problem, device=device)
    scales = _canonical_fused_scales(problem)
    start = _uniform_resource_boundary_start(problem, tensors)
    z = torch.stack((start, start), dim=1)
    z_bar = z.clone()
    average = torch.zeros_like(z)
    best = z.clone()
    best_available = [False, False]
    best_work = [math.inf, math.inf]
    work_dual = torch.zeros(
        (operator.context_count, operator.fiber_count, 2),
        dtype=torch.float64,
        device=device,
    )
    low_dual = torch.zeros(
        (source.rank, 2), dtype=torch.float64, device=device
    )
    spectral_tail_dual = torch.zeros(
        source.route_count, dtype=torch.float64, device=device
    )
    diagonal_tail_dual = torch.zeros((), dtype=torch.float64, device=device)
    resource_dual = torch.zeros(2, dtype=torch.float64, device=device)
    block_dual = torch.zeros(
        (len(source.block_capacities), 2),
        dtype=torch.float64,
        device=device,
    )
    q_scale = torch.as_tensor(
        (scales.q_spectral, scales.q_diagonal),
        dtype=torch.float64,
        device=device,
    )
    q_radius = torch.as_tensor(
        (
            math.sqrt(source.quadratic_budget) / scales.q_spectral,
            math.sqrt(source.quadratic_budget) / scales.q_diagonal,
        ),
        dtype=torch.float64,
        device=device,
    )
    step = scales.step
    finite = True
    final_receipts: tuple[V1SignedWorkLeastWorkCUDABranchReceipt, ...] = tuple(
        _empty_branch_receipt(
            problem=problem,
            policy=policy,
            branch=branch,
            scales=scales,
            iterations=0,
            reason="solver_not_started",
            finite=True,
            production=production,
            private_cpu=private_cpu,
        )
        for branch in SIGNED_WORK_LEAST_WORK_BRANCH_ORDER
    )
    reduced = torch.zeros_like(z)
    iterations = 0
    for iteration in range(1, policy.max_iterations + 1):
        iterations = iteration
        cells = operator._forward_prevalidated(z_bar)
        work_dual = torch.clamp(
            work_dual + step * cells / scales.work,
            min=-tensors.work_dual_box,
            max=tensors.work_dual_box,
        )
        resource_projection = torch.matmul(
            tensors.resource.reshape(1, -1), z_bar
        ).reshape(-1)
        resource_dual = torch.clamp(
            resource_dual
            + step
            * (problem.resource_floor - resource_projection)
            / scales.resource,
            min=0.0,
        )
        block_projection = _block_forward_columns(source, scales, z_bar)
        if int(block_dual.shape[0]):
            block_dual = torch.clamp(
                block_dual
                + step
                * (block_projection - tensors.block_bounds.reshape(-1, 1)),
                min=0.0,
            )
        low_projection = tensors.factor @ z_bar
        low_candidate = low_dual + step * low_projection / q_scale.reshape(1, 2)
        spectral_candidate = spectral_tail_dual + (
            step
            * math.sqrt(source.residual_spectral_bound)
            * z_bar[:, 0]
            / scales.q_spectral
        )
        diagonal_projection = torch.dot(tensors.sqrt_residual, z_bar[:, 1])
        diagonal_candidate = diagonal_tail_dual + (
            step * diagonal_projection / scales.q_diagonal
        )
        q_norm = torch.stack(
            (
                torch.sqrt(
                    torch.dot(low_candidate[:, 0], low_candidate[:, 0])
                    + torch.dot(spectral_candidate, spectral_candidate)
                ),
                torch.sqrt(
                    torch.dot(low_candidate[:, 1], low_candidate[:, 1])
                    + diagonal_candidate * diagonal_candidate
                ),
            )
        )
        shrink = torch.clamp(
            1.0
            - step
            * q_radius
            / torch.clamp(q_norm, min=torch.finfo(torch.float64).tiny),
            min=0.0,
        )
        low_dual = low_candidate * shrink.reshape(1, 2)
        spectral_tail_dual = spectral_candidate * shrink[0]
        diagonal_tail_dual = diagonal_candidate * shrink[1]

        work_adjoint = operator._adjoint_prevalidated(work_dual) / scales.work
        q_adjoint = _q_dual_adjoint(
            problem,
            tensors,
            scales,
            low_dual,
            spectral_tail_dual,
            diagonal_tail_dual,
        )
        block_adjoint = _block_adjoint_columns(
            source,
            scales,
            block_dual,
            route_count=source.route_count,
        )
        gradient = (
            work_adjoint
            - tensors.resource.reshape(-1, 1)
            * (resource_dual / scales.resource).reshape(1, 2)
            + q_adjoint
            + block_adjoint
        )
        if tensors.residual_work is not None:
            gradient = gradient + tensors.residual_work.reshape(-1, 1) / scales.work
        z_new = torch.clamp(z - step * gradient, min=0.0, max=1.0)
        z_new = z_new * tensors.eligible.reshape(-1, 1)
        if tensors.has_required:
            z_new[tensors.required, :] = 1.0
        z_bar = z_new + policy.over_relaxation * (z_new - z)
        z = z_new
        average = average + (z - average) / float(iteration)
        should_check = (
            iteration % policy.check_interval == 0
            or iteration == policy.max_iterations
        )
        if not should_check:
            continue
        finite = bool(
            torch.isfinite(z).all().item()
            and torch.isfinite(z_bar).all().item()
            and torch.isfinite(work_dual).all().item()
            and torch.isfinite(low_dual).all().item()
            and torch.isfinite(spectral_tail_dual).all().item()
            and torch.isfinite(diagonal_tail_dual).item()
            and torch.isfinite(resource_dual).all().item()
            and torch.isfinite(block_dual).all().item()
        )
        if not finite:
            break
        candidates = torch.cat((z, average), dim=1)
        metrics, _candidate_cells, _candidate_low = _candidate_metric_columns(
            problem,
            tensors,
            scales,
            candidates,
            (
                V1RelaxedKKTBranch.SPECTRAL,
                V1RelaxedKKTBranch.DIAGONAL_L1,
                V1RelaxedKKTBranch.SPECTRAL,
                V1RelaxedKKTBranch.DIAGONAL_L1,
            ),
        )
        del _candidate_cells, _candidate_low
        for branch_column in range(2):
            for proposal_column in (branch_column, branch_column + 2):
                work_value = float(metrics[proposal_column, 0])
                normalized_violation = float(metrics[proposal_column, 5])
                if (
                    normalized_violation <= policy.primal_feasibility_tolerance
                    and work_value < best_work[branch_column]
                ):
                    best[:, branch_column].copy_(candidates[:, proposal_column])
                    best_work[branch_column] = work_value
                    best_available[branch_column] = True
        del candidates
        # The receipt recomputes exact adjoints from the registered dual state.
        # Dropping the iteration-local copies first prevents two complete KKT
        # scratch sets from coexisting at the certification boundary.
        del work_adjoint, q_adjoint, block_adjoint, gradient, z_new
        final_receipts, reduced = _issue_fused_branch_receipts(
            problem=problem,
            policy=policy,
            tensors=tensors,
            scales=scales,
            best=best,
            best_available=(best_available[0], best_available[1]),
            work_dual=work_dual,
            low_dual=low_dual,
            spectral_tail_dual=spectral_tail_dual,
            diagonal_tail_dual=diagonal_tail_dual,
            resource_dual=resource_dual,
            block_dual=block_dual,
            iterations=iteration,
            finite=finite,
            production=production,
            private_cpu=private_cpu,
        )
        if all(receipt.converged for receipt in final_receipts):
            winner = min(
                final_receipts,
                key=lambda receipt: (
                    receipt.signed_work_upper,
                    receipt.branch.value,
                ),
            )
            other = next(
                receipt
                for receipt in final_receipts
                if receipt.branch is not winner.branch
            )
            selection_tolerance = (
                policy.absolute_gap_tolerance
                + policy.relative_gap_tolerance
                * max(1.0, winner.signed_work_upper, abs(other.dual_work_lower))
            )
            if winner.signed_work_upper <= other.dual_work_lower + selection_tolerance:
                break
    if not finite:
        final_receipts = tuple(
            _empty_branch_receipt(
                problem=problem,
                policy=policy,
                branch=branch,
                scales=scales,
                iterations=iterations,
                reason="nonfinite_iterate_refusal",
                finite=False,
                production=production,
                private_cpu=private_cpu,
            )
            for branch in SIGNED_WORK_LEAST_WORK_BRANCH_ORDER
        )
    return (
        final_receipts,
        best,
        reduced,
        tensors,
        _SignedWorkDualStateTensors(
            branch_order=SIGNED_WORK_LEAST_WORK_BRANCH_ORDER,
            work_dual=work_dual,
            low_rank_dual=low_dual,
            q_tail_duals=(spectral_tail_dual, diagonal_tail_dual),
            resource_dual=resource_dual,
            block_dual=block_dual,
        ),
    )


def estimate_v1_signed_work_least_work_cuda_hbm(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    current_reserved_bytes: int | None = None,
    device_total_memory_bytes: int | None = None,
) -> V1SignedWorkLeastWorkCUDAHBMForecast:
    """Canonical alias for the fused-v2 allocation-free HBM forecast."""

    return build_v1_signed_work_least_work_cuda_hbm_forecast(
        problem,
        policy,
        current_reserved_bytes=current_reserved_bytes,
        device_total_memory_bytes=device_total_memory_bytes,
    )


def estimate_v1_signed_work_least_work_single_branch_cuda_hbm(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    branch: V1RelaxedKKTBranch,
    current_reserved_bytes: int | None = None,
    device_total_memory_bytes: int | None = None,
) -> V1SignedWorkLeastWorkCUDASingleBranchHBMForecast:
    """Canonical allocation-free HBM forecast for one supported branch."""

    return build_v1_signed_work_least_work_single_branch_cuda_hbm_forecast(
        problem,
        policy,
        branch=branch,
        current_reserved_bytes=current_reserved_bytes,
        device_total_memory_bytes=device_total_memory_bytes,
    )


def _solve_v1_signed_work_least_work_fused(
    *,
    problem: (
        V1SignedWorkLeastWorkCUDAProblem
        | V1SignedWorkLeastWorkCUDADeviceProblemV2
    ),
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    private_cpu_reference: bool,
) -> V1SignedWorkLeastWorkCUDAExecutionBundle:
    if type(problem) not in {
        V1SignedWorkLeastWorkCUDAProblem,
        V1SignedWorkLeastWorkCUDADeviceProblemV2,
    }:
        raise TypeError("signed-work fused solve requires exact problem")
    if type(policy) is not V1SignedWorkLeastWorkCUDAPolicy:
        raise TypeError("signed-work fused solve requires exact policy")
    problem.assert_integrity()
    if torch is None:
        _refuse(
            V1SignedWorkLeastWorkCUDARefusalCode.CUDA_REQUIRED,
            "PyTorch is required for signed-work fused execution",
        )
    device = _operator_device_object(problem.signed_work_operator)
    production = getattr(device, "type", None) == "cuda"
    private_cpu = getattr(device, "type", None) == "cpu"
    if private_cpu_reference:
        if not (
            private_cpu
            and policy.allow_private_cpu_reference
            and not policy.production_cuda_required
            and problem.signed_work_operator.receipt.private_cpu_reference_execution
        ):
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.CUDA_REQUIRED,
                "private CPU reference execution was not explicitly registered",
            )
    else:
        if not (
            production
            and policy.production_cuda_required
            and problem.source_bundle_authoritative
            and problem.signed_work_operator.receipt.production_cuda_execution
        ):
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.CUDA_REQUIRED,
                "production signed-work solver requires authoritative CUDA custody",
            )
        if policy.require_deterministic_algorithms and not torch.are_deterministic_algorithms_enabled():
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.DETERMINISM_REQUIRED,
                "torch deterministic algorithms must be enabled before production solve",
            )
    device_source = type(problem) is V1SignedWorkLeastWorkCUDADeviceProblemV2
    forecast = (
        build_v1_signed_work_least_work_cuda_device_hbm_forecast_v2(
            problem,
            policy,
        )
        if device_source
        else build_v1_signed_work_least_work_cuda_hbm_forecast(problem, policy)
    )
    if not forecast.admitted:
        _refuse(
            V1SignedWorkLeastWorkCUDARefusalCode.HBM_FORECAST_EXCEEDED,
            "signed-work fused solver exceeds the registered 85-percent HBM ceiling",
        )
    receipts, primals, reduced, source_tensors, dual_tensors = (
        _solve_fused_signed_work_branches(
            problem=problem,
            policy=policy,
            device=device,
            production=production,
            private_cpu=private_cpu,
        )
    )
    all_converged = all(receipt.converged for receipt in receipts)
    selected: V1RelaxedKKTBranch | None = None
    branch_selection = False
    if all_converged:
        winner = min(
            receipts,
            key=lambda receipt: (
                receipt.signed_work_upper,
                receipt.branch.value,
            ),
        )
        other = next(
            receipt for receipt in receipts if receipt.branch is not winner.branch
        )
        tolerance = (
            policy.absolute_gap_tolerance
            + policy.relative_gap_tolerance
            * max(1.0, winner.signed_work_upper, abs(other.dual_work_lower))
        )
        branch_selection = (
            winner.signed_work_upper <= other.dual_work_lower + tolerance
        )
        if branch_selection:
            selected = winner.branch
    state_handle: V1SignedWorkLeastWorkCUDAStateHandle | None = None
    if all_converged and branch_selection:
        external_geometry_lease = (
            problem._take_geometry_lease_for_state()
            if device_source
            else None
        )
        try:
            state_handle = V1SignedWorkLeastWorkCUDAStateHandle(
                problem_hash=problem.problem_hash,
                policy_hash=policy.policy_hash,
                branch_receipt_hashes=(
                    receipts[0].receipt_hash,
                    receipts[1].receipt_hash,
                ),
                primals=primals,
                reduced_advantages=reduced,
                source_tensors=source_tensors,
                dual_tensors=dual_tensors,
                materialize_route_limit=policy.materialize_route_limit,
                eligible_mask_identity_hash=problem.eligible_mask_identity_hash,
                required_mask_identity_hash=problem.required_mask_identity_hash,
                external_geometry_lease=external_geometry_lease,
                _exclusive_transfer_token=_STATE_EXCLUSIVE_TRANSFER_TOKEN,
            )
        except BaseException:
            if external_geometry_lease is not None:
                external_geometry_lease.close()
            raise
        del primals, reduced, source_tensors, dual_tensors
    result = V1SignedWorkLeastWorkCUDAResult(
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        forecast_hash=forecast.forecast_hash,
        branch_receipts=receipts,
        selected_branch=selected,
        branch_selection_certified_within_tolerance=branch_selection,
        relaxed_state_available=state_handle is not None,
        state_identity_hash=(
            None if state_handle is None else state_handle.identity_hash
        ),
        all_branches_converged=all_converged,
    )
    return V1SignedWorkLeastWorkCUDAExecutionBundle(
        result=result,
        forecast=forecast,
        state_handle=state_handle,
        production_cuda_execution=production,
        private_cpu_reference_execution=private_cpu,
    )


def _solve_v1_signed_work_least_work_single_branch_fused(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    branch: V1RelaxedKKTBranch,
    private_cpu_reference: bool,
) -> V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle:
    """Retain theorem state for one Phase-A-supported convex branch.

    This is not a fallback from the two-branch union.  The caller has already
    frozen Phase-A branch membership at one endogenous resource coordinate;
    the numerical program, theorem-state axis, and public authority contain
    only ``branch``.
    """

    if type(problem) is not V1SignedWorkLeastWorkCUDAProblem:
        raise TypeError("signed-work single-branch solve requires exact problem")
    if type(policy) is not V1SignedWorkLeastWorkCUDAPolicy:
        raise TypeError("signed-work single-branch solve requires exact policy")
    if type(branch) is not V1RelaxedKKTBranch:
        raise TypeError("signed-work single-branch solve requires one exact Q branch")
    problem.assert_integrity()
    if torch is None:
        _refuse(
            V1SignedWorkLeastWorkCUDARefusalCode.CUDA_REQUIRED,
            "PyTorch is required for signed-work fused execution",
        )
    device = _operator_device_object(problem.signed_work_operator)
    production = getattr(device, "type", None) == "cuda"
    private_cpu = getattr(device, "type", None) == "cpu"
    if private_cpu_reference:
        if not (
            private_cpu
            and policy.allow_private_cpu_reference
            and not policy.production_cuda_required
            and problem.signed_work_operator.receipt.private_cpu_reference_execution
        ):
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.CUDA_REQUIRED,
                "private CPU single-branch execution was not explicitly registered",
            )
    else:
        if not (
            production
            and policy.production_cuda_required
            and problem.source_bundle_authoritative
            and problem.signed_work_operator.receipt.production_cuda_execution
        ):
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.CUDA_REQUIRED,
                "production single-branch solver requires authoritative CUDA custody",
            )
        if (
            policy.require_deterministic_algorithms
            and not torch.are_deterministic_algorithms_enabled()
        ):
            _refuse(
                V1SignedWorkLeastWorkCUDARefusalCode.DETERMINISM_REQUIRED,
                "torch deterministic algorithms must be enabled before production solve",
            )
    forecast = build_v1_signed_work_least_work_single_branch_cuda_hbm_forecast(
        problem,
        policy,
        branch=branch,
    )
    if not forecast.admitted:
        _refuse(
            V1SignedWorkLeastWorkCUDARefusalCode.HBM_FORECAST_EXCEEDED,
            "signed-work single-branch solver exceeds the registered HBM ceiling",
        )
    receipt, primals, reduced, source_tensors, dual_tensors = (
        _solve_independent_signed_work_branch(
            problem=problem,
            policy=policy,
            branch=branch,
            device=device,
            production=production,
            private_cpu=private_cpu,
        )
    )
    if not receipt.converged:
        # A single-branch continuation exists only for a Phase-A-supported
        # branch.  Returning an unresolved object would require pretending its
        # exact extended-real refusal sentinel were a content-addressed
        # numerical certificate.  Refuse instead; the coordinator cannot
        # freeze or replay this coordinate.
        _refuse(
            V1SignedWorkLeastWorkCUDARefusalCode.NONCONVERGENCE,
            f"supported {branch.value} branch did not obtain a certified optimum",
        )
    receipt_hashes = (receipt.receipt_hash,)
    state_handle: V1SignedWorkLeastWorkCUDAStateHandle | None = None
    state_handle = V1SignedWorkLeastWorkCUDAStateHandle(
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        branch_receipt_hashes=receipt_hashes,
        primals=primals,
        reduced_advantages=reduced,
        source_tensors=source_tensors,
        dual_tensors=dual_tensors,
        materialize_route_limit=policy.materialize_route_limit,
        eligible_mask_identity_hash=problem.eligible_mask_identity_hash,
        required_mask_identity_hash=problem.required_mask_identity_hash,
        branch_order=(branch,),
        _exclusive_transfer_token=_STATE_EXCLUSIVE_TRANSFER_TOKEN,
    )
    del primals, reduced, source_tensors, dual_tensors
    result = V1SignedWorkLeastWorkCUDASingleBranchResult(
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        forecast_hash=forecast.forecast_hash,
        branch=branch,
        branch_receipt=receipt,
        solver_state_binding_hashes=receipt_hashes,
        branch_converged=receipt.converged,
        relaxed_state_available=state_handle is not None,
        state_identity_hash=(
            None if state_handle is None else state_handle.identity_hash
        ),
    )
    return V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle(
        result=result,
        forecast=forecast,
        state_handle=state_handle,
        production_cuda_execution=production,
        private_cpu_reference_execution=private_cpu,
    )


def solve_v1_signed_work_least_work_cuda(
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy = V1SignedWorkLeastWorkCUDAPolicy(),
) -> V1SignedWorkLeastWorkCUDAExecutionBundle:
    """Solve both signed-F/Q convex branches with CUDA-only fused execution."""

    return _solve_v1_signed_work_least_work_fused(
        problem=problem,
        policy=policy,
        private_cpu_reference=False,
    )


def solve_v1_signed_work_least_work_private_cpu_reference(
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
) -> V1SignedWorkLeastWorkCUDAExecutionBundle:
    """Run the identical fused tensor kernel in explicit small-test CPU mode."""

    return _solve_v1_signed_work_least_work_fused(
        problem=problem,
        policy=policy,
        private_cpu_reference=True,
    )


def estimate_v1_signed_work_least_work_device_cuda_hbm_v2(
    *,
    problem: V1SignedWorkLeastWorkCUDADeviceProblemV2,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    current_reserved_bytes: int | None = None,
    device_total_memory_bytes: int | None = None,
) -> V1SignedWorkLeastWorkCUDADeviceHBMForecastV2:
    """Allocation-free forecast for the exact live-device source sibling."""

    return build_v1_signed_work_least_work_cuda_device_hbm_forecast_v2(
        problem,
        policy,
        current_reserved_bytes=current_reserved_bytes,
        device_total_memory_bytes=device_total_memory_bytes,
    )


def solve_v1_signed_work_least_work_device_cuda_v2(
    problem: V1SignedWorkLeastWorkCUDADeviceProblemV2,
    policy: V1SignedWorkLeastWorkCUDAPolicy = V1SignedWorkLeastWorkCUDAPolicy(),
) -> V1SignedWorkLeastWorkCUDAExecutionBundle:
    """Solve the registered two-branch program from zero-copy CUDA sources."""

    if type(problem) is not V1SignedWorkLeastWorkCUDADeviceProblemV2:
        raise TypeError("device CUDA solve requires the exact v2 device problem")
    return _solve_v1_signed_work_least_work_fused(
        problem=problem,
        policy=policy,
        private_cpu_reference=False,
    )


def solve_v1_signed_work_least_work_device_private_cpu_reference_v2(
    problem: V1SignedWorkLeastWorkCUDADeviceProblemV2,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
) -> V1SignedWorkLeastWorkCUDAExecutionBundle:
    """Tiny-domain parity execution of the identical zero-copy tensor path."""

    if type(problem) is not V1SignedWorkLeastWorkCUDADeviceProblemV2:
        raise TypeError("device reference solve requires the exact v2 device problem")
    return _solve_v1_signed_work_least_work_fused(
        problem=problem,
        policy=policy,
        private_cpu_reference=True,
    )


def solve_v1_signed_work_least_work_branch_cuda(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    branch: V1RelaxedKKTBranch,
) -> V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle:
    """Solve and retain exactly one Phase-A-supported branch on CUDA."""

    return _solve_v1_signed_work_least_work_single_branch_fused(
        problem=problem,
        policy=policy,
        branch=branch,
        private_cpu_reference=False,
    )


def solve_v1_signed_work_least_work_branch_private_cpu_reference(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    branch: V1RelaxedKKTBranch,
) -> V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle:
    """Tiny-domain reference for one explicitly registered supporting branch."""

    return _solve_v1_signed_work_least_work_single_branch_fused(
        problem=problem,
        policy=policy,
        branch=branch,
        private_cpu_reference=True,
    )


__all__ = [
    "SIGNED_WORK_LEAST_WORK_CUDA_ALGORITHM_ID",
    "SIGNED_WORK_LEAST_WORK_CUDA_AUTHORITY",
    "SIGNED_WORK_LEAST_WORK_CUDA_BRANCH_RECEIPT_SCHEMA",
    "SIGNED_WORK_LEAST_WORK_CUDA_EXECUTION_SCHEMA",
    "SIGNED_WORK_LEAST_WORK_ELIGIBLE_MASK_IDENTITY_PREFIX",
    "SIGNED_WORK_LEAST_WORK_CUDA_FORECAST_SCHEMA",
    "SIGNED_WORK_LEAST_WORK_CUDA_DEVICE_FORECAST_SCHEMA_V2",
    "SIGNED_WORK_LEAST_WORK_CUDA_POLICY_SCHEMA",
    "SIGNED_WORK_LEAST_WORK_CUDA_PROBLEM_SCHEMA",
    "SIGNED_WORK_LEAST_WORK_CUDA_DEVICE_PROBLEM_SCHEMA_V2",
    "SIGNED_WORK_LEAST_WORK_CUDA_DEVICE_STRUCTURAL_RECEIPT_SEMANTICS_V2",
    "SIGNED_WORK_LEAST_WORK_CUDA_RESULT_SCHEMA",
    "SIGNED_WORK_LEAST_WORK_CUDA_SINGLE_BRANCH_EXECUTION_SCHEMA",
    "SIGNED_WORK_LEAST_WORK_CUDA_SINGLE_BRANCH_FORECAST_SCHEMA",
    "SIGNED_WORK_LEAST_WORK_CUDA_SINGLE_BRANCH_RESULT_SCHEMA",
    "SIGNED_WORK_LEAST_WORK_CUDA_STATE_SCHEMA",
    "SIGNED_WORK_LEAST_WORK_REQUIRED_MASK_IDENTITY_PREFIX",
    "V1SignedWorkLeastWorkCUDABranchReceipt",
    "V1SignedWorkLeastWorkCUDAExecutionBundle",
    "V1SignedWorkLeastWorkCUDAHBMForecast",
    "V1SignedWorkLeastWorkCUDADeviceHBMForecastV2",
    "V1SignedWorkLeastWorkCUDAPolicy",
    "V1SignedWorkLeastWorkCUDAProblem",
    "V1SignedWorkLeastWorkCUDADeviceProblemV2",
    "V1SignedWorkLeastWorkCUDARefusal",
    "V1SignedWorkLeastWorkCUDARefusalCode",
    "V1SignedWorkLeastWorkCUDAResult",
    "V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle",
    "V1SignedWorkLeastWorkCUDASingleBranchHBMForecast",
    "V1SignedWorkLeastWorkCUDASingleBranchResult",
    "V1SignedWorkLeastWorkCUDAOutsidePairSourceLease",
    "V1SignedWorkLeastWorkCUDAStateBorrowOperation",
    "V1SignedWorkLeastWorkCUDAStateHandle",
    "V1SignedWorkLeastWorkCUDAStateLease",
    "build_v1_signed_work_least_work_cuda_hbm_forecast",
    "build_v1_signed_work_least_work_cuda_device_hbm_forecast_v2",
    "build_v1_signed_work_least_work_single_branch_cuda_hbm_forecast",
    "estimate_v1_signed_work_least_work_cuda_hbm",
    "estimate_v1_signed_work_least_work_device_cuda_hbm_v2",
    "estimate_v1_signed_work_least_work_single_branch_cuda_hbm",
    "solve_v1_signed_work_least_work_cuda",
    "solve_v1_signed_work_least_work_device_cuda_v2",
    "solve_v1_signed_work_least_work_device_private_cpu_reference_v2",
    "solve_v1_signed_work_least_work_branch_cuda",
    "solve_v1_signed_work_least_work_branch_private_cpu_reference",
    "solve_v1_signed_work_least_work_private_cpu_reference",
]
