"""Matrix-free convex KKT relaxations for the modern-v1 pruning action.

The binary residual certificate

``min(rho * |S|, (sum_{j in S} sqrt(d_j)) ** 2)``

is the minimum of two convex extensions only on hard masks.  Its relaxed
feasible set is therefore represented by two independently solved convex
branches rather than by taking a pointwise minimum inside one optimization:

``spectral``
    ``q(z) = ||H z||_2**2 + rho ||z||_2**2``;

``diagonal_l1``
    ``q(z) = ||H z||_2**2 + (sqrt(d)^T z)**2``.

Each branch is a linearly constrained second-order-cone program.  The solver
uses deterministic float64 Chambolle--Pock iterations with matrix-free
products.  It emits a primal feasible lower bound, a conic dual upper bound,
and an explicit decomposition of their gap into Lagrangian stationarity and
constraint complementarity.  A converged receipt is authority only for the
registered *relaxed branch* within its numerical tolerances.  It never claims
that deterministic hard rounding is globally optimal.

The NumPy implementation is a bounded small-problem reference.  A route axis
larger than the registered reference limit fails closed.  Target-scale use is
CUDA-only through the torch backend; no production CPU fallback exists.
"""

from __future__ import annotations

import hashlib
import math
import time
from contextlib import ExitStack
from dataclasses import InitVar, dataclass, field, fields, is_dataclass, replace
from enum import Enum
from fractions import Fraction
from itertools import product
from numbers import Integral, Real
from threading import RLock
from typing import TYPE_CHECKING, Any, Final, Iterator, Literal, NoReturn, Sequence

import numpy as np

from mechanistic_systems.contracts.hard_mask_actions import (
    HardMask,
    RemovedRouteIndexSet,
)
from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

from .contracts import content_hash
from .modern_v1_certified_linalg import (
    certified_frobenius_norm_upper_bound,
    certified_gram_perturbation_norm_upper_bound,
    certified_gram_operator_min_eigenvalue_lower_bound,
)
from .modern_v1_confidence_box_external_state import V1ConfidenceBoxExternalView
from .modern_v1_device_shell_ladder import (
    DEVICE_SHELL_DEFAULT_PHYSICAL_CHUNK_SIZE,
    DEVICE_SHELL_MAXIMUM_ENDPOINTS,
    V1DeviceShellCompactBlockCapacity,
    V1DeviceShellFrozenLadderReceipt,
    V1DeviceShellLabelHandle,
    freeze_v1_device_shell_ladder,
)
from .modern_v1_literal_j_operator_census import V1LiteralJOperatorCensusReceipt
from .modern_v1_literal_j_projection_cuda import (
    V1LiteralJPrivatePrimalDirectionCUDAHandle,
    V1LiteralJPrivatePrimalDirectionCUDALease,
    V1LiteralJPrivatePrimalDirectionDescriptor,
    V1LiteralJPrivatePrimalDirectionDescriptorAxis,
)
from .modern_v1_literal_j_projection_inference import (
    V1LiteralJFixedDirectionProjectionReceipt,
)

if TYPE_CHECKING:
    from .modern_v1_literal_j_d_fixed_residual_population_projection import (
        V1LiteralJDFixedResidualPopulationProjectionReceipt,
    )


RELAXED_KKT_POLICY_SCHEMA: Final = "partition_free_v1_relaxed_kkt_policy_v1"
RELAXED_KKT_BLOCK_SCHEMA: Final = (
    "partition_free_v1_relaxed_kkt_compact_block_capacity_v1"
)
RELAXED_KKT_PROBLEM_SCHEMA: Final = "partition_free_v1_relaxed_kkt_problem_v1"
RELAXED_KKT_PRIMAL_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_relaxed_kkt_primal_feasibility_receipt_v1"
)
RELAXED_KKT_STATIONARITY_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_relaxed_kkt_stationarity_receipt_v1"
)
RELAXED_KKT_GAP_RECEIPT_SCHEMA: Final = "partition_free_v1_relaxed_kkt_gap_receipt_v1"
RELAXED_KKT_CONVERGENCE_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_relaxed_kkt_convergence_receipt_v1"
)
RELAXED_KKT_BRANCH_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_relaxed_kkt_branch_receipt_v1"
)
RELAXED_KKT_CANDIDATE_SCHEMA: Final = "partition_free_v1_relaxed_kkt_hard_candidate_v1"
RELAXED_KKT_LADDER_SCHEMA: Final = "partition_free_v1_relaxed_kkt_rounded_ladder_v1"
RELAXED_KKT_BRANCH_RESULT_SCHEMA: Final = (
    "partition_free_v1_relaxed_kkt_branch_result_v1"
)
RELAXED_KKT_RESULT_SCHEMA: Final = "partition_free_v1_relaxed_kkt_result_v1"
RELAXED_KKT_ALGORITHM_ID: Final = (
    "float64_matrix_free_chambolle_pock_two_branch_socp_v1"
)
RELAXED_KKT_FUSED_CUDA_ALGORITHM_ID: Final = (
    "float64_matrix_free_fused_two_branch_chambolle_pock_socp_v2"
)
RELAXED_KKT_CUDA_EXECUTION_DESIGN_SCHEMA: Final = (
    "partition_free_v1_relaxed_kkt_cuda_execution_design_v2"
)
RELAXED_KKT_CUDA_HBM_FORECAST_SCHEMA: Final = (
    "partition_free_v1_relaxed_kkt_cuda_hbm_forecast_v2"
)
RELAXED_KKT_CUDA_TELEMETRY_SCHEMA: Final = (
    "partition_free_v1_relaxed_kkt_cuda_execution_telemetry_v2"
)
RELAXED_KKT_CUDA_EXECUTION_BUNDLE_SCHEMA: Final = (
    "partition_free_v1_relaxed_kkt_cuda_execution_bundle_v2"
)
RELAXED_KKT_DEVICE_SHELL_EXECUTION_DESIGN_SCHEMA: Final = (
    "partition_free_v1_relaxed_kkt_device_shell_execution_design_v1"
)
RELAXED_KKT_DEVICE_SHELL_BRANCH_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_relaxed_kkt_device_shell_branch_identity_receipt_v1"
)
RELAXED_KKT_DEVICE_SHELL_EXECUTION_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_relaxed_kkt_device_shell_execution_receipt_v1"
)
RELAXED_KKT_DEVICE_SHELL_NONCONVERGENCE_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_relaxed_kkt_device_shell_nonconvergence_receipt_v1"
)
RELAXED_KKT_ADAPTIVE_RANK_BRANCH_WITNESS_SCHEMA: Final = (
    "partition_free_v1_relaxed_kkt_adaptive_rank_branch_witness_v1"
)
RELAXED_KKT_FACE_GEOMETRY_UNCERTAINTY_SCHEMA: Final = (
    "partition_free_v1_relaxed_kkt_face_geometry_uncertainty_v1"
)
RELAXED_KKT_ROBUST_SOLUTION_SENSITIVITY_SCHEMA: Final = (
    "partition_free_v1_relaxed_kkt_robust_solution_sensitivity_v1"
)
RELAXED_KKT_DECISION_LEVEL_ROBUST_INVARIANCE_SCHEMA: Final = (
    "partition_free_v1_relaxed_kkt_decision_level_robust_invariance_v1"
)
RELAXED_KKT_OUTSIDE_SHELL_INTERVAL_SCAN_SCHEMA: Final = (
    "partition_free_v1_relaxed_kkt_outside_shell_interval_scan_v1"
)
RELAXED_KKT_ACTIVE_SHELL_RESTRICTED_OPERATOR_SCHEMA: Final = (
    "partition_free_v1_relaxed_kkt_active_shell_restricted_operator_v1"
)
RELAXED_KKT_ROBUST_DUAL_BOUND_SCHEMA: Final = (
    "partition_free_v1_relaxed_kkt_robust_dual_bound_v1"
)
RELAXED_KKT_WORST_CASE_HARD_FQ_REPLAY_SCHEMA: Final = (
    "partition_free_v1_relaxed_kkt_worst_case_hard_fq_replay_v1"
)
RELAXED_KKT_ALL_ADMISSIBLE_SHELL_FACES_SCHEMA: Final = (
    "partition_free_v1_relaxed_kkt_all_admissible_shell_faces_v1"
)
RELAXED_KKT_DECISION_LEVEL_THEOREM_STATE_BORROW_REQUEST_SCHEMA: Final = (
    "partition_free_v1_relaxed_kkt_decision_level_theorem_state_borrow_request_v1"
)
RELAXED_KKT_DECISION_LEVEL_THEOREM_LEASE_HBM_FORECAST_SCHEMA: Final = (
    "partition_free_v1_relaxed_kkt_decision_level_theorem_lease_hbm_forecast_v1"
)
RELAXED_KKT_ROUNDING_KEY_ID: Final = (
    "negative_relaxed_gate_negative_kkt_reduced_advantage_"
    "corrected_work_per_resource_physical_index_v1"
)
RELAXED_KKT_CANONICAL_HARD_Q_ID: Final = (
    "norm_H_indicator_squared_plus_min_rho_cardinality_or_diag_l1_v1"
)
RELAXED_KKT_SINGLETON_TRIANGLE_WORK_ID: Final = (
    "singleton_triangle_corrected_work_upper_envelope_A_dot_z_"
    "not_canonical_signed_action_F_v1"
)
RELAXED_KKT_SIGNED_ACTION_WORK_ID: Final = (
    "canonical_signed_corrected_action_work_E_context_abs_sum_a_xj_zj_"
    "requires_signed_work_operator_or_exact_C_replay_v1"
)
RELAXED_KKT_RELAXED_AUTHORITY_SCOPE: Final = (
    "registered_convex_relaxed_branch_within_numerical_tolerance_only_"
    "no_discrete_global_optimality_v1"
)
RELAXED_KKT_ADAPTIVE_RANK_WITNESS_SCOPE: Final = (
    "routewise_relaxed_face_and_rounded_endpoint_stability_witness_only_"
    "no_fqr_calibration_or_commit_authority_v1"
)
RELAXED_KKT_ROBUST_SOLUTION_SENSITIVITY_SCOPE: Final = (
    "robust_relaxed_kkt_solution_sensitivity_certificate_for_registered_"
    "uncertainty_set_only_no_fqr_calibration_or_commit_authority_v1"
)
RELAXED_KKT_DECISION_LEVEL_ROBUST_INVARIANCE_SCOPE: Final = (
    "all_admissible_registered_shell_faces_round_to_one_hard_ladder_with_"
    "strict_outside_dual_and_FQ_interval_invariance_only_no_C_K_O_commit_"
    "terminal_or_pareto_authority_v1"
)
_ADAPTIVE_RANK_BRANCH_WITNESS_ISSUANCE_TOKEN: Final = object()
_GEOMETRY_UNCERTAINTY_ISSUANCE_TOKEN: Final = object()
_ROBUST_SOLUTION_SENSITIVITY_ISSUANCE_TOKEN: Final = object()
_DECISION_LEVEL_ROBUST_INVARIANCE_ISSUANCE_TOKEN: Final = object()
_DECISION_LEVEL_SOURCE_RECEIPT_ISSUANCE_TOKEN: Final = object()

# The production rounding workspace is bounded independently of the route
# axis.  This is an execution constant rather than a statistical or numerical
# policy: changing it may alter timing, but exact lexicographic membership and
# endpoint identity must not change.
_CUDA_ROUNDING_TILE_SIZE: Final = 1 << 20
_CUDA_DEFAULT_HBM_CEILING_FRACTION: Final = 0.85
_CUDA_DEFAULT_MAXIMUM_HOST_ORDER_BYTES: Final = 64 * 1024 * 1024
# The active fused solver owns four route vectors per branch plus the
# spectral response-tail vector.  A converged branch snapshot is retained
# separately and accounted exactly below; it must not be smuggled into this
# baseline count or the generic transient envelope.
_CUDA_FUSED_ACTIVE_ROUTE_VECTOR_COUNT: Final = 9
# Solver-state plus transient route-vector equivalents.  The deliberately
# conservative 48-vector envelope covers check-time four-column proposals,
# repaired copies, cached reduced advantages, response tails, adjoints, and
# allocator overlap; the five persistent input vectors are counted separately.
V1_RELAXED_KKT_FUSED_ROUTE_VECTOR_FORECAST_COUNT: Final = 48
_CUDA_FUSED_ROUTE_VECTOR_FORECAST_COUNT: Final = (
    V1_RELAXED_KKT_FUSED_ROUTE_VECTOR_FORECAST_COUNT
)
_ROBUST_SENSITIVITY_UNBOUNDED: Final = float(np.finfo(np.float64).max / 4.0)


def _fused_branch_state_snapshot_forecast_bytes(
    *,
    route_count: int,
    rank: int,
    block_count: int,
) -> int:
    """Return the exact maximum bytes retained by two converged snapshots.

    Across the spectral and diagonal branches the private snapshot owns five
    route vectors (two primals, two reduced advantages, and the spectral
    response tail), four rank vectors (two primal projections and two
    response-low duals), two block-dual vectors, and three scalar tensors
    (two work duals and the diagonal response tail).  The scalar numerical
    receipts live on the host and therefore contribute no HBM bytes.
    """

    f64 = np.dtype(np.float64).itemsize
    return f64 * (
        5 * _positive_int(route_count, "route_count")
        + 4 * _nonnegative_int(rank, "rank")
        + 2 * _nonnegative_int(block_count, "block_count")
        + 3
    )


KKTBackend = Literal["numpy_reference", "torch_cuda"]
InputArrayPolicy = Literal["copy_to_bytes", "admit_readonly_c_contiguous"]


class V1RelaxedKKTBranch(str, Enum):
    """The two convex residual-envelope extensions solved independently."""

    SPECTRAL = "spectral"
    DIAGONAL_L1 = "diagonal_l1"


class V1RelaxedKKTClosureMode(str, Enum):
    """The exact theorem that closes relaxed-problem uncertainty for C."""

    EXACT_ZERO = "exact_zero"
    FULL_SOLUTION_SENSITIVITY = "full_solution_sensitivity"
    DECISION_LEVEL_SHELL_INVARIANCE = "decision_level_shell_invariance"


class V1RelaxedKKTRefusalCode(str, Enum):
    """Typed fail-closed outcomes that cannot become selector authority."""

    CPU_REFERENCE_SCALE_EXCEEDED = "cpu_reference_scale_exceeded"
    CUDA_UNAVAILABLE = "cuda_unavailable"
    CUDA_ROUNDING_POOL_EXCEEDED = "cuda_rounding_pool_exceeded"
    CUDA_HBM_FORECAST_EXCEEDED = "cuda_hbm_forecast_exceeded"
    CUDA_OUT_OF_MEMORY = "cuda_out_of_memory"
    CUDA_WORKSPACE_EXCEEDED = "cuda_workspace_exceeded"
    CUDA_HOST_ORDER_LIMIT_EXCEEDED = "cuda_host_order_limit_exceeded"
    CUDA_SHELL_WITNESS_UNAVAILABLE = "cuda_shell_witness_unavailable"
    CUDA_DEVICE_SHELL_IDENTITY_UNAVAILABLE = "cuda_device_shell_identity_unavailable"
    CUDA_DEVICE_SHELL_ACTIVE_LEASES = "cuda_device_shell_active_leases"
    CUDA_THEOREM_STATE_LEASE_UNAVAILABLE = "cuda_theorem_state_lease_unavailable"
    CUDA_THEOREM_STATE_MUTATED = "cuda_theorem_state_mutated"
    ADAPTIVE_RANK_WITNESS_SOURCE_MISMATCH = "adaptive_rank_witness_source_mismatch"
    ADAPTIVE_RANK_WITNESS_BOUNDARY_UNAVAILABLE = (
        "adaptive_rank_witness_boundary_unavailable"
    )
    ROBUST_SOLUTION_SENSITIVITY_UNAVAILABLE = "robust_solution_sensitivity_unavailable"
    DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE = (
        "decision_level_shell_invariance_unavailable"
    )


class V1RelaxedKKTRefusal(RuntimeError):
    """Intentional refusal to substitute an unregistered execution path."""

    def __init__(
        self,
        code: V1RelaxedKKTRefusalCode,
        message: str,
        *,
        device_shell_nonconvergence_receipt: (
            V1RelaxedKKTDeviceShellNonconvergenceReceipt | None
        ) = None,
    ) -> None:
        if (
            device_shell_nonconvergence_receipt is not None
            and type(device_shell_nonconvergence_receipt)
            is not V1RelaxedKKTDeviceShellNonconvergenceReceipt
        ):
            raise TypeError(
                "device_shell_nonconvergence_receipt must be the exact typed "
                "receipt or None"
            )
        if device_shell_nonconvergence_receipt is not None:
            device_shell_nonconvergence_receipt.assert_integrity()
        self.code = code
        self.device_shell_nonconvergence_receipt = device_shell_nonconvergence_receipt
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1RelaxedKKTRefusalCode,
    message: str,
    *,
    device_shell_nonconvergence_receipt: (
        V1RelaxedKKTDeviceShellNonconvergenceReceipt | None
    ) = None,
) -> NoReturn:
    raise V1RelaxedKKTRefusal(
        code,
        message,
        device_shell_nonconvergence_receipt=(device_shell_nonconvergence_receipt),
    )


def _nonempty(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be one positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be one positive integer")
    return result


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be one nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be one nonnegative integer")
    return result


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be one finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative(value: object, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _positive(value: object, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _immutable_bool_vector(
    value: object,
    *,
    size: int | None,
    name: str,
    policy: InputArrayPolicy,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or (size is not None and raw.size != size):
        raise ValueError(f"{name} must be a one-dimensional route-axis vector")
    if policy == "admit_readonly_c_contiguous":
        if (
            raw.dtype != np.dtype(np.bool_)
            or not raw.flags.c_contiguous
            or raw.flags.writeable
        ):
            raise ValueError(
                f"{name} must be readonly contiguous bool for alias admission"
            )
        return raw
    try:
        numeric = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be binary") from error
    if numeric.dtype.kind not in {"b", "i", "u", "f"}:
        raise ValueError(f"{name} must be binary")
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0, 1)).all():
        raise ValueError(f"{name} must be binary")
    contiguous = np.ascontiguousarray(numeric, dtype=np.bool_)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=np.bool_)


def _immutable_f64_vector(
    value: object,
    *,
    size: int,
    name: str,
    nonnegative: bool,
    policy: InputArrayPolicy,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size != size:
        raise ValueError(f"{name} must be a one-dimensional route-axis vector")
    if policy == "admit_readonly_c_contiguous":
        if (
            raw.dtype != np.dtype(np.float64)
            or not raw.flags.c_contiguous
            or raw.flags.writeable
        ):
            raise ValueError(
                f"{name} must be readonly contiguous float64 for alias admission"
            )
        result = raw
    else:
        try:
            contiguous = np.ascontiguousarray(raw, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must be representable as float64") from error
        result = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    if nonnegative and bool(np.any(result < 0.0)):
        raise ValueError(f"{name} must be nonnegative")
    return result


def _immutable_f64_matrix(
    value: object,
    *,
    columns: int,
    name: str,
    policy: InputArrayPolicy,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[1] != columns:
        raise ValueError(f"{name} must have shape (rank, route_count)")
    if policy == "admit_readonly_c_contiguous":
        if (
            raw.dtype != np.dtype(np.float64)
            or not raw.flags.c_contiguous
            or raw.flags.writeable
        ):
            raise ValueError(
                f"{name} must be readonly contiguous float64 for alias admission"
            )
        result = raw
    else:
        try:
            contiguous = np.ascontiguousarray(raw, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must be representable as float64") from error
        result = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64).reshape(
            contiguous.shape
        )
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _readonly_copy(value: np.ndarray, *, dtype: np.dtype[Any]) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=dtype).reshape(
        contiguous.shape
    )


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _stable_l2_norm(value: np.ndarray, *, chunk_size: int = 1 << 20) -> float:
    """Return a bounded-workspace scaled sum-of-squares norm."""

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
        raise ValueError("float64 operator norm is not representable")
    return result


def _stable_sqrt_sum_nonnegative(
    value: np.ndarray, *, chunk_size: int = 1 << 20
) -> float:
    """Return ``sqrt(sum(value))`` without a route-sized square-root copy."""

    flat = np.asarray(value, dtype=np.float64).reshape(-1)
    scale = 0.0
    scaled_sum = 0.0
    for start in range(0, int(flat.size), chunk_size):
        chunk = flat[start : start + chunk_size]
        if not chunk.size:
            continue
        chunk_scale = float(np.max(chunk))
        if chunk_scale == 0.0:
            continue
        chunk_sum = float(np.sum(chunk / chunk_scale, dtype=np.float64))
        if scale < chunk_scale:
            scaled_sum = scaled_sum * (scale / chunk_scale) + chunk_sum
            scale = chunk_scale
        else:
            scaled_sum += chunk_sum * (chunk_scale / scale)
    result = math.sqrt(scale) * math.sqrt(scaled_sum) if scale > 0.0 else 0.0
    if not math.isfinite(result):
        raise ValueError("float64 residual norm is not representable")
    return result


@dataclass(frozen=True, slots=True)
class V1RelaxedKKTBlockCapacity:
    """One contiguous hard/relaxed removal-capacity constraint."""

    block_id: str
    start: int
    stop: int
    removal_capacity: int
    schema: str = RELAXED_KKT_BLOCK_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_id", _nonempty(self.block_id, "block_id"))
        start = _nonnegative_int(self.start, "block start")
        stop = _positive_int(self.stop, "block stop")
        capacity = _nonnegative_int(self.removal_capacity, "removal_capacity")
        if stop <= start:
            raise ValueError("block stop must exceed block start")
        if capacity > stop - start:
            raise ValueError("block removal capacity cannot exceed its span")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)
        object.__setattr__(self, "removal_capacity", capacity)
        if self.schema != RELAXED_KKT_BLOCK_SCHEMA:
            raise ValueError("unsupported relaxed KKT block schema")

    @property
    def block_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-relaxed-kkt-block-")


@dataclass(frozen=True, slots=True)
class V1RelaxedKKTPolicy:
    """Numerical and materialization policy for both convex branches.

    ``maximum_cuda_rounding_pool`` remains a hashed legacy replay coordinate.
    The bounded tiled CUDA rounding path does not consume it and cannot emit
    the retired pool-exhaustion refusal.
    """

    policy_id: str = "modern-v1-relaxed-kkt-default"
    backend: KKTBackend = "numpy_reference"
    torch_device: str = "cuda"
    max_iterations: int = 4096
    minimum_iterations: int = 32
    check_interval: int = 16
    step_safety: float = 0.90
    over_relaxation: float = 1.0
    primal_feasibility_tolerance: float = 1e-9
    normalized_stationarity_tolerance: float = 1e-6
    absolute_gap_tolerance: float = 1e-9
    relative_gap_tolerance: float = 1e-6
    normalized_complementarity_tolerance: float = 1e-6
    cpu_reference_route_limit: int = 4096
    candidate_count_ladder: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    materialize_relaxed_gate_route_limit: int = 4096
    maximum_cuda_rounding_pool: int = 4_000_000
    schema: str = RELAXED_KKT_POLICY_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _nonempty(self.policy_id, "policy_id"))
        if self.backend not in {"numpy_reference", "torch_cuda"}:
            raise ValueError("unsupported relaxed KKT backend")
        object.__setattr__(
            self, "torch_device", _nonempty(self.torch_device, "torch_device")
        )
        for name in (
            "max_iterations",
            "minimum_iterations",
            "check_interval",
            "cpu_reference_route_limit",
            "maximum_cuda_rounding_pool",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        object.__setattr__(
            self,
            "materialize_relaxed_gate_route_limit",
            _nonnegative_int(
                self.materialize_relaxed_gate_route_limit,
                "materialize_relaxed_gate_route_limit",
            ),
        )
        if self.minimum_iterations > self.max_iterations:
            raise ValueError("minimum_iterations cannot exceed max_iterations")
        if self.check_interval > self.max_iterations:
            raise ValueError("check_interval cannot exceed max_iterations")
        safety = _positive(self.step_safety, "step_safety")
        if safety >= 1.0:
            raise ValueError("step_safety must be strictly below one")
        theta = _nonnegative(self.over_relaxation, "over_relaxation")
        if theta > 1.0:
            raise ValueError("over_relaxation must be at most one")
        object.__setattr__(self, "step_safety", safety)
        object.__setattr__(self, "over_relaxation", theta)
        for name in (
            "primal_feasibility_tolerance",
            "normalized_stationarity_tolerance",
            "absolute_gap_tolerance",
            "relative_gap_tolerance",
            "normalized_complementarity_tolerance",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        counts = tuple(
            _positive_int(item, "candidate_count_ladder item")
            for item in self.candidate_count_ladder
        )
        if not counts or tuple(sorted(set(counts))) != counts:
            raise ValueError("candidate_count_ladder must be strictly increasing")
        object.__setattr__(self, "candidate_count_ladder", counts)
        if self.schema != RELAXED_KKT_POLICY_SCHEMA:
            raise ValueError("unsupported relaxed KKT policy schema")

    @property
    def policy_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-relaxed-kkt-policy-")


@dataclass(frozen=True, slots=True)
class V1RelaxedKKTCUDAExecutionDesign:
    """Versioned nonstatistical execution controls for the CUDA solver.

    The design is deliberately separate from :class:`V1RelaxedKKTPolicy` so
    the frozen v1 mathematical-policy schema and its historical wire identity
    remain unchanged.  ``scan_only`` is an authority-inert engineering mode:
    it executes PDHG but suppresses every hard ladder regardless of numerical
    convergence.
    """

    design_id: str = "modern-v1-relaxed-kkt-fused-cuda-v2"
    implementation: Literal["fused_v2", "legacy_v1"] = "fused_v2"
    scan_only: bool = False
    require_adaptive_shell_witness: bool = False
    hbm_ceiling_fraction: float = _CUDA_DEFAULT_HBM_CEILING_FRACTION
    maximum_host_order_bytes: int = _CUDA_DEFAULT_MAXIMUM_HOST_ORDER_BYTES
    rounding_tile_size: int = _CUDA_ROUNDING_TILE_SIZE
    schema: str = RELAXED_KKT_CUDA_EXECUTION_DESIGN_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "design_id", _nonempty(self.design_id, "design_id"))
        if self.implementation not in {"fused_v2", "legacy_v1"}:
            raise ValueError("unsupported CUDA KKT implementation")
        for name in ("scan_only", "require_adaptive_shell_witness"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact bool")
        ceiling = _positive(self.hbm_ceiling_fraction, "hbm_ceiling_fraction")
        if ceiling > 1.0:
            raise ValueError("hbm_ceiling_fraction must be at most one")
        object.__setattr__(self, "hbm_ceiling_fraction", ceiling)
        object.__setattr__(
            self,
            "maximum_host_order_bytes",
            _positive_int(self.maximum_host_order_bytes, "maximum_host_order_bytes"),
        )
        object.__setattr__(
            self,
            "rounding_tile_size",
            _positive_int(self.rounding_tile_size, "rounding_tile_size"),
        )
        if self.schema != RELAXED_KKT_CUDA_EXECUTION_DESIGN_SCHEMA:
            raise ValueError("unsupported CUDA KKT execution design schema")

    @property
    def design_hash(self) -> str:
        return content_hash(
            self, prefix="partition-free-v1-relaxed-kkt-cuda-execution-design-"
        )


@dataclass(frozen=True, slots=True)
class V1RelaxedKKTDeviceShellExecutionDesign:
    """Explicit mode for device-resident shell-label identity.

    The nested CUDA design owns only fused solver and workspace controls.
    This sibling design changes the rounding output representation without
    changing the mathematical policy or the default/legacy CUDA entrypoint.
    When the nested design requires an adaptive shell witness, execution must
    also receive source-derived geometry uncertainty and must emit both typed
    branch witnesses; construction alone grants no downstream authority.
    """

    cuda_execution_design: V1RelaxedKKTCUDAExecutionDesign
    design_id: str = "modern-v1-relaxed-kkt-device-shell-labels-v1"
    output_mode: Literal["device_resident_uint8_first_inclusion_v1"] = (
        "device_resident_uint8_first_inclusion_v1"
    )
    physical_chunk_size: int = DEVICE_SHELL_DEFAULT_PHYSICAL_CHUNK_SIZE
    allow_unconstrained_blocks: bool = False
    schema: str = RELAXED_KKT_DEVICE_SHELL_EXECUTION_DESIGN_SCHEMA

    def __post_init__(self) -> None:
        if type(self.cuda_execution_design) is not V1RelaxedKKTCUDAExecutionDesign:
            raise TypeError("device-shell mode requires an exact CUDA design")
        if self.cuda_execution_design.implementation != "fused_v2":
            raise ValueError("device-shell mode requires explicit fused_v2")
        if self.cuda_execution_design.scan_only:
            raise ValueError("device-shell identity mode is distinct from scan-only")
        object.__setattr__(self, "design_id", _nonempty(self.design_id, "design_id"))
        if self.output_mode != "device_resident_uint8_first_inclusion_v1":
            raise ValueError("unsupported device-shell output mode")
        object.__setattr__(
            self,
            "physical_chunk_size",
            _positive_int(self.physical_chunk_size, "physical_chunk_size"),
        )
        if type(self.allow_unconstrained_blocks) is not bool:
            raise TypeError("allow_unconstrained_blocks must be an exact bool")
        if self.schema != RELAXED_KKT_DEVICE_SHELL_EXECUTION_DESIGN_SCHEMA:
            raise ValueError("unsupported device-shell execution design schema")

    @property
    def design_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-relaxed-kkt-device-shell-execution-design-",
        )


@dataclass(frozen=True, slots=True)
class V1RelaxedKKTProblem:
    """One frozen current-chart relaxed action problem.

    ``corrected_work`` owns the singleton triangle F branch ``A^T z``.  It is
    a lawful conservative rollout envelope for corrected evidence, but it is
    not the canonical signed action work ``E_x |sum_j a_xj z_j|``.  Signed
    action work must enter through the separate signed-work operator/C replay
    path.  ``low_rank_factor``, ``residual_diagonal``, and
    ``residual_spectral_bound`` own Q.  The problem deliberately accepts no
    nonlinear-remainder outcomes and grants no mask commit authority.
    """

    problem_id: str
    resource: np.ndarray = field(repr=False, compare=False)
    corrected_work: np.ndarray = field(repr=False, compare=False)
    low_rank_factor: np.ndarray = field(repr=False, compare=False)
    residual_diagonal: np.ndarray = field(repr=False, compare=False)
    residual_spectral_bound: float
    eligible_mask: np.ndarray = field(repr=False, compare=False)
    work_budget: float
    quadratic_budget: float
    route_axis_hash: str
    source_geometry_hash: str
    block_capacities: tuple[V1RelaxedKKTBlockCapacity, ...] = ()
    input_array_policy: InputArrayPolicy = "copy_to_bytes"
    corrected_work_envelope_id: str = RELAXED_KKT_SINGLETON_TRIANGLE_WORK_ID
    schema: str = RELAXED_KKT_PROBLEM_SCHEMA
    _problem_hash: str = field(init=False, repr=False, compare=False)
    _work_l2_norm: float = field(init=False, repr=False, compare=False)
    _factor_frobenius_norm: float = field(init=False, repr=False, compare=False)
    _residual_sqrt_l2_norm: float = field(init=False, repr=False, compare=False)
    _objective_scale: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "problem_id", _nonempty(self.problem_id, "problem_id"))
        if self.input_array_policy not in {
            "copy_to_bytes",
            "admit_readonly_c_contiguous",
        }:
            raise ValueError("unsupported relaxed KKT input array policy")
        raw_eligible = np.asarray(self.eligible_mask)
        if raw_eligible.ndim != 1:
            raise ValueError("eligible_mask must be one-dimensional")
        route_count = int(raw_eligible.size)
        if route_count <= 0:
            raise ValueError("relaxed KKT route axis must be nonempty")
        eligible = _immutable_bool_vector(
            self.eligible_mask,
            size=route_count,
            name="eligible_mask",
            policy=self.input_array_policy,
        )
        resource = _immutable_f64_vector(
            self.resource,
            size=route_count,
            name="resource",
            nonnegative=True,
            policy=self.input_array_policy,
        )
        work = _immutable_f64_vector(
            self.corrected_work,
            size=route_count,
            name="corrected_work",
            nonnegative=True,
            policy=self.input_array_policy,
        )
        residual = _immutable_f64_vector(
            self.residual_diagonal,
            size=route_count,
            name="residual_diagonal",
            nonnegative=True,
            policy=self.input_array_policy,
        )
        factor = _immutable_f64_matrix(
            self.low_rank_factor,
            columns=route_count,
            name="low_rank_factor",
            policy=self.input_array_policy,
        )
        if bool(np.any(resource[eligible] <= 0.0)):
            raise ValueError("resource must be positive on every eligible route")
        rho = _nonnegative(self.residual_spectral_bound, "residual_spectral_bound")
        max_diagonal = float(np.max(residual)) if residual.size else 0.0
        scale = max(1.0, rho, max_diagonal)
        tolerance = 64.0 * np.finfo(np.float64).eps * scale
        if rho + tolerance < max_diagonal:
            raise ValueError("residual_spectral_bound must cover max residual_diagonal")
        if rho < max_diagonal:
            rho = max_diagonal
        blocks = tuple(self.block_capacities)
        if any(type(item) is not V1RelaxedKKTBlockCapacity for item in blocks):
            raise TypeError("block_capacities require exact compact block objects")
        if blocks:
            previous_stop = 0
            for block in blocks:
                if block.start != previous_stop:
                    raise ValueError(
                        "compact block capacities must be contiguous and ordered"
                    )
                if block.stop > route_count:
                    raise ValueError("compact block exceeds the route axis")
                previous_stop = block.stop
            if previous_stop != route_count:
                raise ValueError("compact block capacities must cover the route axis")
            if len({item.block_id for item in blocks}) != len(blocks):
                raise ValueError("compact block IDs must be unique")
        object.__setattr__(self, "eligible_mask", eligible)
        object.__setattr__(self, "resource", resource)
        object.__setattr__(self, "corrected_work", work)
        object.__setattr__(self, "residual_diagonal", residual)
        object.__setattr__(self, "low_rank_factor", factor)
        object.__setattr__(self, "residual_spectral_bound", rho)
        object.__setattr__(self, "block_capacities", blocks)
        object.__setattr__(
            self, "work_budget", _nonnegative(self.work_budget, "work_budget")
        )
        object.__setattr__(
            self,
            "quadratic_budget",
            _nonnegative(self.quadratic_budget, "quadratic_budget"),
        )
        object.__setattr__(
            self, "route_axis_hash", _nonempty(self.route_axis_hash, "route_axis_hash")
        )
        object.__setattr__(
            self,
            "source_geometry_hash",
            _nonempty(self.source_geometry_hash, "source_geometry_hash"),
        )
        if self.corrected_work_envelope_id != RELAXED_KKT_SINGLETON_TRIANGLE_WORK_ID:
            raise ValueError(
                "V1RelaxedKKTProblem.corrected_work is only the singleton "
                "triangle envelope; signed action work requires its own "
                "registered operator"
            )
        if self.schema != RELAXED_KKT_PROBLEM_SCHEMA:
            raise ValueError("unsupported relaxed KKT problem schema")
        object.__setattr__(self, "_work_l2_norm", _stable_l2_norm(work))
        object.__setattr__(self, "_factor_frobenius_norm", _stable_l2_norm(factor))
        object.__setattr__(
            self,
            "_residual_sqrt_l2_norm",
            _stable_sqrt_sum_nonnegative(residual),
        )
        object.__setattr__(self, "_objective_scale", max(1.0, float(np.max(resource))))
        object.__setattr__(
            self,
            "_problem_hash",
            content_hash(
                {
                    "schema": self.schema,
                    "problem_id": self.problem_id,
                    "resource": resource,
                    "corrected_work": work,
                    "low_rank_factor": factor,
                    "residual_diagonal": residual,
                    "residual_spectral_bound": rho,
                    "eligible_mask": eligible,
                    "work_budget": self.work_budget,
                    "quadratic_budget": self.quadratic_budget,
                    "route_axis_hash": self.route_axis_hash,
                    "source_geometry_hash": self.source_geometry_hash,
                    "block_capacities": blocks,
                    "input_array_policy": self.input_array_policy,
                    "corrected_work_envelope_id": self.corrected_work_envelope_id,
                },
                prefix="partition-free-v1-relaxed-kkt-problem-",
            ),
        )

    @property
    def route_count(self) -> int:
        return int(self.eligible_mask.size)

    @property
    def rank(self) -> int:
        return int(self.low_rank_factor.shape[0])

    @property
    def legal_hard_capacity(self) -> int:
        if not self.block_capacities:
            return int(np.count_nonzero(self.eligible_mask))
        total = 0
        for block in self.block_capacities:
            eligible = int(
                np.count_nonzero(self.eligible_mask[block.start : block.stop])
            )
            total += min(block.removal_capacity, eligible)
        return total

    @property
    def problem_hash(self) -> str:
        return self._problem_hash


@dataclass(frozen=True, slots=True)
class V1RelaxedKKTCUDAHBMForecast:
    """Allocation-free conservative HBM admission receipt."""

    problem_hash: str
    execution_design_hash: str
    device: str
    route_count: int
    rank: int
    maximum_endpoint: int
    host_order_bytes: int
    factor_bytes: int
    persistent_input_bytes: int
    solver_state_bytes: int
    transient_workspace_bytes: int
    rounding_workspace_bytes: int
    forecast_incremental_bytes: int
    current_reserved_bytes: int
    forecast_peak_reserved_bytes: int
    device_total_memory_bytes: int
    hbm_ceiling_bytes: int
    admitted: bool
    scan_only: bool
    private_primal_direction_bytes: int = 0
    schema: str = RELAXED_KKT_CUDA_HBM_FORECAST_SCHEMA
    _admission_memory: InitVar[tuple[int, int] | None] = None

    def __post_init__(self, _admission_memory: tuple[int, int] | None) -> None:
        for name in ("problem_hash", "execution_design_hash", "device"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        for name in (
            "route_count",
            "maximum_endpoint",
            "host_order_bytes",
            "factor_bytes",
            "persistent_input_bytes",
            "solver_state_bytes",
            "transient_workspace_bytes",
            "rounding_workspace_bytes",
            "forecast_incremental_bytes",
            "current_reserved_bytes",
            "forecast_peak_reserved_bytes",
            "device_total_memory_bytes",
            "hbm_ceiling_bytes",
            "private_primal_direction_bytes",
        ):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name))
        object.__setattr__(self, "rank", _nonnegative_int(self.rank, "rank"))
        if type(self.admitted) is not bool or type(self.scan_only) is not bool:
            raise TypeError("forecast admission and scan_only must be exact booleans")
        if self.route_count <= 0 or self.device_total_memory_bytes <= 0:
            raise ValueError("forecast requires positive route and device-memory sizes")
        expected_incremental = (
            self.persistent_input_bytes
            + self.solver_state_bytes
            + self.transient_workspace_bytes
            + self.rounding_workspace_bytes
            + self.private_primal_direction_bytes
        )
        if self.forecast_incremental_bytes != expected_incremental:
            raise ValueError("forecast incremental-byte accounting does not close")
        if self.forecast_peak_reserved_bytes != (
            self.current_reserved_bytes + self.forecast_incremental_bytes
        ):
            raise ValueError("forecast peak-byte accounting does not close")
        available, allocated = (
            (self.hbm_ceiling_bytes, self.current_reserved_bytes)
            if _admission_memory is None
            else _admission_memory
        )
        if self.admitted is not (
            allocated + self.forecast_incremental_bytes <= self.hbm_ceiling_bytes
            and self.forecast_incremental_bytes <= available
        ):
            raise ValueError("forecast admission differs from its byte ceiling")
        if self.schema != RELAXED_KKT_CUDA_HBM_FORECAST_SCHEMA:
            raise ValueError("unsupported CUDA HBM forecast schema")

    @property
    def forecast_hash(self) -> str:
        return content_hash(
            self, prefix="partition-free-v1-relaxed-kkt-cuda-hbm-forecast-"
        )


def estimate_v1_relaxed_kkt_cuda_hbm(
    *,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    execution_design: V1RelaxedKKTCUDAExecutionDesign,
    device_total_memory_bytes: int,
    current_reserved_bytes: int = 0,
    device: str = "cuda",
    device_shell_execution_design: V1RelaxedKKTDeviceShellExecutionDesign | None = None,
    _admission_memory: tuple[int, int] | None = None,
) -> V1RelaxedKKTCUDAHBMForecast:
    """Forecast fused execution without constructing any CUDA tensor.

    Supplying the explicit device-shell design retains two route-axis uint8
    label vectors, accounts for them in HBM, and records zero host *order*
    bytes.  The later identity scan transfers the two label axes and is
    accounted separately in execution telemetry.
    """

    if type(problem) is not V1RelaxedKKTProblem:
        raise TypeError("HBM forecast requires an exact relaxed KKT problem")
    if type(policy) is not V1RelaxedKKTPolicy:
        raise TypeError("HBM forecast requires an exact relaxed KKT policy")
    if type(execution_design) is not V1RelaxedKKTCUDAExecutionDesign:
        raise TypeError("HBM forecast requires an exact CUDA execution design")
    shell_mode = device_shell_execution_design is not None
    if shell_mode:
        if (
            type(device_shell_execution_design)
            is not V1RelaxedKKTDeviceShellExecutionDesign
        ):
            raise TypeError("device-shell HBM forecast requires an exact design")
        if device_shell_execution_design.cuda_execution_design != execution_design:
            raise ValueError("device-shell and CUDA forecast designs differ")
    total = _positive_int(device_total_memory_bytes, "device_total_memory_bytes")
    reserved = _nonnegative_int(current_reserved_bytes, "current_reserved_bytes")
    route_count = problem.route_count
    rank = problem.rank
    f64 = np.dtype(np.float64).itemsize
    factor_bytes = rank * route_count * f64
    # resource, work, residual, sqrt(residual), objective, plus eligibility.
    persistent = factor_bytes + 5 * route_count * f64 + route_count
    # Four active primal-state vectors for each branch, plus the spectral
    # route dual, followed by the exact two-branch converged-state snapshot.
    # The latter remains live while hard identities are frozen and is a
    # future input to source-derived face certification; account it before
    # any outcome or CUDA allocation is observed.
    solver_state = (
        _CUDA_FUSED_ACTIVE_ROUTE_VECTOR_COUNT * route_count * f64
        + _fused_branch_state_snapshot_forecast_bytes(
            route_count=route_count,
            rank=rank,
            block_count=len(problem.block_capacities),
        )
    )
    transient = (
        (
            _CUDA_FUSED_ROUTE_VECTOR_FORECAST_COUNT
            - _CUDA_FUSED_ACTIVE_ROUTE_VECTOR_COUNT
        )
        * route_count
        * f64
    )
    capacity = problem.legal_hard_capacity
    counts = tuple(
        count for count in policy.candidate_count_ladder if count <= capacity
    )
    maximum_endpoint = counts[-1] if counts else 0
    host_order_bytes = (
        0 if shell_mode else maximum_endpoint * np.dtype(np.int64).itemsize
    )
    if execution_design.scan_only or maximum_endpoint <= 0:
        rounding_workspace = 0
    else:
        tile = min(route_count, execution_design.rounding_tile_size)
        # Indices plus the three floating keys and conservative top-k/sort
        # temporaries for one endpoint-sized merge and one physical tile.
        rounding_workspace = 12 * (maximum_endpoint + tile) * f64
        if shell_mode:
            # One first-inclusion byte per route and branch remains live after
            # both endpoint orders have been consumed.  The bounded order/key
            # workspace above is reused branch by branch.
            rounding_workspace += 2 * route_count * np.dtype(np.uint8).itemsize
    incremental = persistent + solver_state + transient + rounding_workspace
    private_primal_direction_bytes = 2 * route_count * f64 if shell_mode else 0
    incremental += private_primal_direction_bytes
    # Preserve the raw reservation projection in the public forecast.  Live
    # admission uses allocated bytes and reusable cache captured by the caller.
    peak = reserved + incremental
    ceiling = int(math.floor(total * execution_design.hbm_ceiling_fraction))
    available, allocated = (
        (ceiling, reserved) if _admission_memory is None else _admission_memory
    )
    return V1RelaxedKKTCUDAHBMForecast(
        problem_hash=problem.problem_hash,
        execution_design_hash=execution_design.design_hash,
        device=_nonempty(device, "device"),
        route_count=route_count,
        rank=rank,
        maximum_endpoint=maximum_endpoint,
        host_order_bytes=host_order_bytes,
        factor_bytes=factor_bytes,
        persistent_input_bytes=persistent,
        solver_state_bytes=solver_state,
        transient_workspace_bytes=transient,
        rounding_workspace_bytes=rounding_workspace,
        forecast_incremental_bytes=incremental,
        current_reserved_bytes=reserved,
        forecast_peak_reserved_bytes=peak,
        device_total_memory_bytes=total,
        hbm_ceiling_bytes=ceiling,
        admitted=allocated + incremental <= ceiling and incremental <= available,
        scan_only=execution_design.scan_only,
        private_primal_direction_bytes=private_primal_direction_bytes,
        _admission_memory=_admission_memory,
    )


@dataclass(frozen=True, slots=True)
class V1RelaxedKKTPrimalFeasibilityReceipt:
    resource_value: float
    work_value: float
    quadratic_value: float
    response_norm: float
    maximum_box_violation: float
    maximum_block_violation: float
    work_violation: float
    quadratic_violation: float
    normalized_maximum_violation: float
    block_sums: tuple[float, ...]
    feasible: bool
    schema: str = RELAXED_KKT_PRIMAL_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "resource_value",
            "work_value",
            "quadratic_value",
            "response_norm",
            "maximum_box_violation",
            "maximum_block_violation",
            "work_violation",
            "quadratic_violation",
            "normalized_maximum_violation",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        blocks = tuple(_nonnegative(item, "block sum") for item in self.block_sums)
        object.__setattr__(self, "block_sums", blocks)
        if type(self.feasible) is not bool:
            raise TypeError("feasible must be an exact bool")
        if self.schema != RELAXED_KKT_PRIMAL_RECEIPT_SCHEMA:
            raise ValueError("unsupported relaxed KKT primal receipt schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-relaxed-kkt-primal-")


@dataclass(frozen=True, slots=True)
class V1RelaxedKKTStationarityReceipt:
    work_dual: float
    block_duals: tuple[float, ...]
    response_dual_norm: float
    lagrangian_support_value: float
    lagrangian_stationarity_gap: float
    normalized_stationarity_gap: float
    complementarity_gap: float
    normalized_complementarity_gap: float
    reduced_advantage_linf: float
    schema: str = RELAXED_KKT_STATIONARITY_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "work_dual",
            "response_dual_norm",
            "lagrangian_support_value",
            "lagrangian_stationarity_gap",
            "normalized_stationarity_gap",
            "complementarity_gap",
            "normalized_complementarity_gap",
            "reduced_advantage_linf",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        object.__setattr__(
            self,
            "block_duals",
            tuple(_nonnegative(item, "block dual") for item in self.block_duals),
        )
        if self.schema != RELAXED_KKT_STATIONARITY_RECEIPT_SCHEMA:
            raise ValueError("unsupported relaxed KKT stationarity receipt schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-relaxed-kkt-stationarity-")


@dataclass(frozen=True, slots=True)
class V1RelaxedKKTGapReceipt:
    primal_resource_lower_bound: float
    conic_dual_resource_upper_bound: float
    absolute_duality_gap: float
    relative_duality_gap: float
    absolute_gap_tolerance: float
    relative_gap_tolerance: float
    within_tolerance: bool
    schema: str = RELAXED_KKT_GAP_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "primal_resource_lower_bound",
            "conic_dual_resource_upper_bound",
            "absolute_duality_gap",
            "relative_duality_gap",
            "absolute_gap_tolerance",
            "relative_gap_tolerance",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        tolerance = (
            128.0
            * np.finfo(np.float64).eps
            * max(
                1.0,
                self.primal_resource_lower_bound,
                self.conic_dual_resource_upper_bound,
            )
        )
        if (
            self.conic_dual_resource_upper_bound + tolerance
            < self.primal_resource_lower_bound
        ):
            raise ValueError("conic dual bound lies below the feasible primal value")
        if type(self.within_tolerance) is not bool:
            raise TypeError("within_tolerance must be an exact bool")
        if self.schema != RELAXED_KKT_GAP_RECEIPT_SCHEMA:
            raise ValueError("unsupported relaxed KKT gap receipt schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-relaxed-kkt-gap-")


@dataclass(frozen=True, slots=True)
class V1RelaxedKKTConvergenceReceipt:
    branch: V1RelaxedKKTBranch
    converged: bool
    iterations: int
    maximum_iterations: int
    termination_reason: str
    backend: KKTBackend
    device: str
    dtype: str
    algorithm_id: str
    scaled_operator_norm_squared_upper_bound: float
    primal_step_size: float
    dual_step_size: float
    finite_iterates: bool
    relaxed_branch_optimality_only: bool = True
    discrete_global_optimality_claim: bool = False
    mask_commit_authority: bool = False
    authority_scope: str = RELAXED_KKT_RELAXED_AUTHORITY_SCOPE
    schema: str = RELAXED_KKT_CONVERGENCE_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in ("converged", "finite_iterates"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact bool")
        iterations = _positive_int(self.iterations, "iterations")
        maximum = _positive_int(self.maximum_iterations, "maximum_iterations")
        if iterations > maximum:
            raise ValueError("iterations cannot exceed maximum_iterations")
        object.__setattr__(self, "iterations", iterations)
        object.__setattr__(self, "maximum_iterations", maximum)
        for name in (
            "termination_reason",
            "device",
            "dtype",
            "algorithm_id",
            "authority_scope",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if self.backend not in {"numpy_reference", "torch_cuda"}:
            raise ValueError("unsupported convergence backend")
        for name in (
            "scaled_operator_norm_squared_upper_bound",
            "primal_step_size",
            "dual_step_size",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        if self.primal_step_size <= 0.0 or self.dual_step_size <= 0.0:
            raise ValueError("primal and dual step sizes must be positive")
        for name, expected in (
            ("relaxed_branch_optimality_only", True),
            ("discrete_global_optimality_claim", False),
            ("mask_commit_authority", False),
        ):
            if (
                type(getattr(self, name)) is not bool
                or getattr(self, name) is not expected
            ):
                raise PermissionError(f"invalid convergence authority flag {name}")
        if self.algorithm_id not in {
            RELAXED_KKT_ALGORITHM_ID,
            RELAXED_KKT_FUSED_CUDA_ALGORITHM_ID,
        }:
            raise ValueError("unsupported relaxed KKT algorithm")
        if self.authority_scope != RELAXED_KKT_RELAXED_AUTHORITY_SCOPE:
            raise ValueError("unsupported relaxed KKT authority scope")
        if self.schema != RELAXED_KKT_CONVERGENCE_RECEIPT_SCHEMA:
            raise ValueError("unsupported relaxed KKT convergence receipt schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-relaxed-kkt-convergence-")


@dataclass(frozen=True, slots=True)
class V1RelaxedKKTBranchReceipt:
    primal: V1RelaxedKKTPrimalFeasibilityReceipt
    stationarity: V1RelaxedKKTStationarityReceipt
    gap: V1RelaxedKKTGapReceipt
    convergence: V1RelaxedKKTConvergenceReceipt
    problem_hash: str
    policy_hash: str
    schema: str = RELAXED_KKT_BRANCH_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if type(self.primal) is not V1RelaxedKKTPrimalFeasibilityReceipt:
            raise TypeError("branch receipt requires the exact primal receipt")
        if type(self.stationarity) is not V1RelaxedKKTStationarityReceipt:
            raise TypeError("branch receipt requires the exact stationarity receipt")
        if type(self.gap) is not V1RelaxedKKTGapReceipt:
            raise TypeError("branch receipt requires the exact gap receipt")
        if type(self.convergence) is not V1RelaxedKKTConvergenceReceipt:
            raise TypeError("branch receipt requires the exact convergence receipt")
        object.__setattr__(
            self, "problem_hash", _nonempty(self.problem_hash, "problem_hash")
        )
        object.__setattr__(
            self, "policy_hash", _nonempty(self.policy_hash, "policy_hash")
        )
        if self.convergence.converged and not (
            self.primal.feasible and self.gap.within_tolerance
        ):
            raise ValueError("converged branch lacks primal/gap certification")
        if self.schema != RELAXED_KKT_BRANCH_RECEIPT_SCHEMA:
            raise ValueError("unsupported relaxed KKT branch receipt schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-relaxed-kkt-branch-")


def _array_free_diagnostic_payload(value: object) -> object:
    """Return a JSON-safe scalar/nested payload and reject array data.

    Device-shell nonconvergence is an exceptional control path, so benchmark
    tooling must be able to persist its diagnostics without accidentally
    turning route tensors into a D2H or serialization surface.  The admitted
    receipt graph contains only exact dataclasses, enums, scalar primitives,
    tuples, and dictionaries.
    """

    if isinstance(value, np.ndarray):
        raise TypeError("nonconvergence diagnostics cannot serialize arrays")
    if isinstance(value, Enum):
        return value.value
    if value is None or type(value) in {bool, int, float, str}:
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _array_free_diagnostic_payload(getattr(value, item.name))
            for item in fields(value)
            if not item.name.startswith("_")
        }
    if isinstance(value, tuple):
        return tuple(_array_free_diagnostic_payload(item) for item in value)
    if type(value) is dict:
        return {
            str(key): _array_free_diagnostic_payload(item)
            for key, item in value.items()
        }
    raise TypeError(
        "nonconvergence diagnostics support only scalar, tuple, and exact "
        "dataclass payloads"
    )


@dataclass(frozen=True, slots=True)
class V1RelaxedKKTDeviceShellNonconvergenceReceipt:
    """Array-free latest numerical state at device-shell refusal.

    The two nested branch receipts are the exact host-scalar receipts returned
    by the fused solver.  They retain primal feasibility, stationarity and
    complementarity, primal/dual gap, iteration count, step sizes, and finite
    iterate status without retaining a route tensor or device handle.  This
    diagnostic has no selector, measurement, oracle, or commit authority.
    """

    problem_id: str
    problem_hash: str
    policy_id: str
    policy_hash: str
    route_axis_hash: str
    source_geometry_hash: str
    route_count: int
    rank: int
    device_shell_execution_design_hash: str
    cuda_execution_design_hash: str
    spectral: V1RelaxedKKTBranchReceipt
    diagonal_l1: V1RelaxedKKTBranchReceipt
    nonconverged_branches: tuple[V1RelaxedKKTBranch, ...]
    latest_scalar_branch_receipts_complete: bool = True
    route_axis_payload_materialized: bool = False
    route_scale_d2h_tensor_payload_bytes: int = 0
    hard_ladder_emitted: bool = False
    relaxed_branch_optimality_authority: bool = False
    face_authority: bool = False
    fqr_authority: bool = False
    c_stream_authority: bool = False
    k_stream_authority: bool = False
    calibration_authority: bool = False
    o_stream_authority: bool = False
    oracle_authority: bool = False
    device_shell_identity_authority: bool = False
    mask_commit_authority: bool = False
    production_authority: bool = False
    pareto_authority: bool = False
    schema: str = RELAXED_KKT_DEVICE_SHELL_NONCONVERGENCE_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "problem_id",
            "problem_hash",
            "policy_id",
            "policy_hash",
            "route_axis_hash",
            "source_geometry_hash",
            "device_shell_execution_design_hash",
            "cuda_execution_design_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, "route_count"),
        )
        object.__setattr__(self, "rank", _nonnegative_int(self.rank, "rank"))
        if type(self.spectral) is not V1RelaxedKKTBranchReceipt:
            raise TypeError("spectral nonconvergence diagnostic receipt must be exact")
        if type(self.diagonal_l1) is not V1RelaxedKKTBranchReceipt:
            raise TypeError(
                "diagonal-l1 nonconvergence diagnostic receipt must be exact"
            )
        if self.spectral.convergence.branch is not V1RelaxedKKTBranch.SPECTRAL:
            raise ValueError("spectral nonconvergence receipt is mislabelled")
        if self.diagonal_l1.convergence.branch is not V1RelaxedKKTBranch.DIAGONAL_L1:
            raise ValueError("diagonal-l1 nonconvergence receipt is mislabelled")
        for branch_receipt in (self.spectral, self.diagonal_l1):
            if branch_receipt.problem_hash != self.problem_hash:
                raise ValueError("nonconvergence receipt problem identities differ")
            if branch_receipt.policy_hash != self.policy_hash:
                raise ValueError("nonconvergence receipt policy identities differ")
        branches = tuple(self.nonconverged_branches)
        if any(type(branch) is not V1RelaxedKKTBranch for branch in branches):
            raise TypeError("nonconverged_branches must contain exact branches")
        expected = tuple(
            branch
            for branch, branch_receipt in (
                (V1RelaxedKKTBranch.SPECTRAL, self.spectral),
                (V1RelaxedKKTBranch.DIAGONAL_L1, self.diagonal_l1),
            )
            if not branch_receipt.convergence.converged
        )
        if not expected:
            raise ValueError(
                "a nonconvergence receipt requires at least one unconverged branch"
            )
        if branches != expected:
            raise ValueError(
                "nonconverged branch set differs from the nested latest receipts"
            )
        object.__setattr__(self, "nonconverged_branches", branches)
        if type(self.latest_scalar_branch_receipts_complete) is not bool:
            raise TypeError("latest receipt completeness must be an exact bool")
        if not self.latest_scalar_branch_receipts_complete:
            raise PermissionError(
                "nonconvergence diagnostics require both latest branch receipts"
            )
        if type(self.route_axis_payload_materialized) is not bool:
            raise TypeError("route materialization flag must be an exact bool")
        if self.route_axis_payload_materialized:
            raise PermissionError(
                "nonconvergence diagnostics cannot materialize the route axis"
            )
        object.__setattr__(
            self,
            "route_scale_d2h_tensor_payload_bytes",
            _nonnegative_int(
                self.route_scale_d2h_tensor_payload_bytes,
                "route_scale_d2h_tensor_payload_bytes",
            ),
        )
        if self.route_scale_d2h_tensor_payload_bytes != 0:
            raise PermissionError(
                "nonconvergence diagnostics cannot transfer route tensor payloads"
            )
        inert_flags = (
            "hard_ladder_emitted",
            "relaxed_branch_optimality_authority",
            "face_authority",
            "fqr_authority",
            "c_stream_authority",
            "k_stream_authority",
            "calibration_authority",
            "o_stream_authority",
            "oracle_authority",
            "device_shell_identity_authority",
            "mask_commit_authority",
            "production_authority",
            "pareto_authority",
        )
        if any(type(getattr(self, name)) is not bool for name in inert_flags):
            raise TypeError("nonconvergence authority flags must be exact booleans")
        if any(getattr(self, name) for name in inert_flags):
            raise PermissionError(
                "nonconvergence diagnostic crossed its authority boundary"
            )
        if self.schema != RELAXED_KKT_DEVICE_SHELL_NONCONVERGENCE_RECEIPT_SCHEMA:
            raise ValueError("unsupported device-shell nonconvergence schema")
        self.assert_integrity()

    def _hash_payload(self) -> dict[str, object]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "_receipt_hash"
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(
            self._hash_payload(),
            prefix=(
                "partition-free-v1-relaxed-kkt-device-shell-" "nonconvergence-receipt-"
            ),
        )
        if self._receipt_hash is not None and self._receipt_hash != observed:
            raise ValueError("device-shell nonconvergence receipt was tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("device-shell nonconvergence receipt hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    def to_serializable_dict(self) -> dict[str, object]:
        """Return the complete JSON-safe diagnostic payload plus its hash."""

        receipt_hash = self.assert_integrity()
        serialized = _array_free_diagnostic_payload(self._hash_payload())
        if type(serialized) is not dict:  # pragma: no cover - fixed above.
            raise RuntimeError("nonconvergence diagnostic payload is not a mapping")
        serialized["receipt_hash"] = receipt_hash
        return serialized


def _issue_v1_relaxed_kkt_device_shell_nonconvergence_receipt(
    *,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    execution_design: V1RelaxedKKTDeviceShellExecutionDesign,
    spectral: V1RelaxedKKTBranchReceipt,
    diagonal_l1: V1RelaxedKKTBranchReceipt,
) -> V1RelaxedKKTDeviceShellNonconvergenceReceipt:
    """Bind the fused solver's latest scalar receipts to the current chart."""

    if type(problem) is not V1RelaxedKKTProblem:
        raise TypeError("problem must be exact for nonconvergence diagnostics")
    if type(policy) is not V1RelaxedKKTPolicy:
        raise TypeError("policy must be exact for nonconvergence diagnostics")
    if type(execution_design) is not V1RelaxedKKTDeviceShellExecutionDesign:
        raise TypeError("execution_design must be exact for nonconvergence diagnostics")
    if (
        type(spectral) is not V1RelaxedKKTBranchReceipt
        or type(diagonal_l1) is not V1RelaxedKKTBranchReceipt
    ):
        raise TypeError("nonconvergence diagnostics require exact branch receipts")
    return V1RelaxedKKTDeviceShellNonconvergenceReceipt(
        problem_id=problem.problem_id,
        problem_hash=problem.problem_hash,
        policy_id=policy.policy_id,
        policy_hash=policy.policy_hash,
        route_axis_hash=problem.route_axis_hash,
        source_geometry_hash=problem.source_geometry_hash,
        route_count=problem.route_count,
        rank=problem.rank,
        device_shell_execution_design_hash=execution_design.design_hash,
        cuda_execution_design_hash=(execution_design.cuda_execution_design.design_hash),
        spectral=spectral,
        diagonal_l1=diagonal_l1,
        nonconverged_branches=tuple(
            branch
            for branch, receipt in (
                (V1RelaxedKKTBranch.SPECTRAL, spectral),
                (V1RelaxedKKTBranch.DIAGONAL_L1, diagonal_l1),
            )
            if not receipt.convergence.converged
        ),
    )


def _refuse_v1_relaxed_kkt_device_shell_nonconvergence(
    *,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    execution_design: V1RelaxedKKTDeviceShellExecutionDesign,
    spectral: V1RelaxedKKTBranchReceipt,
    diagonal_l1: V1RelaxedKKTBranchReceipt,
) -> NoReturn:
    receipt = _issue_v1_relaxed_kkt_device_shell_nonconvergence_receipt(
        problem=problem,
        policy=policy,
        execution_design=execution_design,
        spectral=spectral,
        diagonal_l1=diagonal_l1,
    )
    _refuse(
        V1RelaxedKKTRefusalCode.CUDA_DEVICE_SHELL_IDENTITY_UNAVAILABLE,
        "relaxed branch convergence unavailable",
        device_shell_nonconvergence_receipt=receipt,
    )


@dataclass(frozen=True, slots=True)
class V1RelaxedKKTHardCandidate:
    branch: V1RelaxedKKTBranch
    prefix_count: int
    removed_index_hash: str
    resource_value: float
    corrected_work_value: float
    low_rank_quadratic: float
    spectral_residual_quadratic: float
    diagonal_l1_residual_quadratic: float
    canonical_residual_quadratic: float
    canonical_quadratic: float
    work_feasible: bool
    quadratic_feasible: bool
    feasible: bool
    canonical_q_equation_id: str = RELAXED_KKT_CANONICAL_HARD_Q_ID
    discrete_global_optimality_claim: bool = False
    schema: str = RELAXED_KKT_CANDIDATE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "prefix_count", _positive_int(self.prefix_count, "prefix_count")
        )
        object.__setattr__(
            self,
            "removed_index_hash",
            _nonempty(self.removed_index_hash, "removed_index_hash"),
        )
        for name in (
            "resource_value",
            "corrected_work_value",
            "low_rank_quadratic",
            "spectral_residual_quadratic",
            "diagonal_l1_residual_quadratic",
            "canonical_residual_quadratic",
            "canonical_quadratic",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        for name in (
            "work_feasible",
            "quadratic_feasible",
            "feasible",
            "discrete_global_optimality_claim",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact bool")
        if self.feasible is not (self.work_feasible and self.quadratic_feasible):
            raise ValueError("candidate feasibility differs from its F/Q receipts")
        if self.discrete_global_optimality_claim is not False:
            raise PermissionError("rounded candidates cannot claim global optimality")
        if self.canonical_q_equation_id != RELAXED_KKT_CANONICAL_HARD_Q_ID:
            raise ValueError("unsupported canonical hard-Q equation")
        if self.schema != RELAXED_KKT_CANDIDATE_SCHEMA:
            raise ValueError("unsupported relaxed KKT candidate schema")

    @property
    def candidate_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-relaxed-kkt-candidate-")


@dataclass(frozen=True, slots=True)
class V1RelaxedKKTRoundedLadder:
    branch: V1RelaxedKKTBranch
    ordered_route_indices: np.ndarray = field(repr=False, compare=False)
    candidates: tuple[V1RelaxedKKTHardCandidate, ...]
    rounding_key_id: str = RELAXED_KKT_ROUNDING_KEY_ID
    every_registered_candidate_replayed: bool = True
    nonmonotone_q_early_stop_used: bool = False
    schema: str = RELAXED_KKT_LADDER_SCHEMA

    def __post_init__(self) -> None:
        indices = np.asarray(self.ordered_route_indices)
        if indices.ndim != 1 or indices.dtype.kind not in {"i", "u"}:
            raise ValueError("ordered_route_indices must be one integer vector")
        normalized = _readonly_copy(indices, dtype=np.dtype(np.int64))
        if normalized.size and (
            int(np.min(normalized)) < 0
            or len(set(int(item) for item in normalized)) != int(normalized.size)
        ):
            raise ValueError(
                "rounded route order must contain unique nonnegative indices"
            )
        candidates = tuple(self.candidates)
        if any(type(item) is not V1RelaxedKKTHardCandidate for item in candidates):
            raise TypeError("rounded ladder requires exact hard candidate receipts")
        counts = tuple(item.prefix_count for item in candidates)
        if counts != tuple(sorted(set(counts))):
            raise ValueError("rounded candidate counts must be strictly increasing")
        if counts and counts[-1] != int(normalized.size):
            raise ValueError("rounded order must end at the largest candidate count")
        if any(item.branch is not self.branch for item in candidates):
            raise ValueError("rounded candidates belong to another branch")
        object.__setattr__(self, "ordered_route_indices", normalized)
        object.__setattr__(self, "candidates", candidates)
        if self.rounding_key_id != RELAXED_KKT_ROUNDING_KEY_ID:
            raise ValueError("unsupported relaxed KKT rounding key")
        if self.every_registered_candidate_replayed is not True:
            raise PermissionError(
                "rounded ladder omitted a registered candidate replay"
            )
        if self.nonmonotone_q_early_stop_used is not False:
            raise PermissionError("rounded ladder may not early-stop on nonmonotone Q")
        if self.schema != RELAXED_KKT_LADDER_SCHEMA:
            raise ValueError("unsupported relaxed KKT ladder schema")

    def removed_indices(self, candidate_index: int) -> np.ndarray:
        candidate = self.candidates[candidate_index]
        view = self.ordered_route_indices[: candidate.prefix_count]
        view.setflags(write=False)
        return view

    @property
    def ladder_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-relaxed-kkt-ladder-")


@dataclass(frozen=True, slots=True)
class V1RelaxedKKTBranchResult:
    branch: V1RelaxedKKTBranch
    receipt: V1RelaxedKKTBranchReceipt
    rounded_ladder: V1RelaxedKKTRoundedLadder
    relaxed_gate_values: np.ndarray | None = field(
        default=None, repr=False, compare=False
    )
    relaxed_gate_digest: str = ""
    schema: str = RELAXED_KKT_BRANCH_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.receipt.convergence.branch is not self.branch:
            raise ValueError("branch result receipt belongs to another branch")
        if self.rounded_ladder.branch is not self.branch:
            raise ValueError("branch result ladder belongs to another branch")
        if self.relaxed_gate_values is None:
            if self.relaxed_gate_digest:
                object.__setattr__(
                    self,
                    "relaxed_gate_digest",
                    _nonempty(self.relaxed_gate_digest, "relaxed_gate_digest"),
                )
        else:
            values = np.asarray(self.relaxed_gate_values)
            if values.ndim != 1 or not np.isfinite(values).all():
                raise ValueError("relaxed gate values must be one finite vector")
            normalized = _readonly_copy(values, dtype=np.dtype(np.float64))
            if bool(np.any(normalized < 0.0)) or bool(np.any(normalized > 1.0)):
                raise ValueError("relaxed gate values must lie in [0, 1]")
            digest = _array_digest(normalized)
            if self.relaxed_gate_digest and self.relaxed_gate_digest != digest:
                raise ValueError("relaxed gate digest differs from materialized values")
            object.__setattr__(self, "relaxed_gate_values", normalized)
            object.__setattr__(self, "relaxed_gate_digest", digest)
        if self.schema != RELAXED_KKT_BRANCH_RESULT_SCHEMA:
            raise ValueError("unsupported relaxed KKT branch-result schema")

    @property
    def result_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "branch": self.branch,
                "receipt_hash": self.receipt.receipt_hash,
                "ladder_hash": self.rounded_ladder.ladder_hash,
                "relaxed_gate_digest": self.relaxed_gate_digest,
            },
            prefix="partition-free-v1-relaxed-kkt-branch-result-",
        )


@dataclass(frozen=True, slots=True)
class V1RelaxedKKTResult:
    problem_hash: str
    policy_hash: str
    spectral: V1RelaxedKKTBranchResult
    diagonal_l1: V1RelaxedKKTBranchResult
    selected_branch: V1RelaxedKKTBranch | None
    both_branches_converged: bool
    branch_choice_certified_within_tolerance: bool
    relaxed_union_optimality_only: bool = True
    discrete_global_optimality_claim: bool = False
    nonlinear_remainder_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    schema: str = RELAXED_KKT_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.spectral.branch is not V1RelaxedKKTBranch.SPECTRAL:
            raise ValueError("spectral result is mislabelled")
        if self.diagonal_l1.branch is not V1RelaxedKKTBranch.DIAGONAL_L1:
            raise ValueError("diagonal-l1 result is mislabelled")
        converged = (
            self.spectral.receipt.convergence.converged
            and self.diagonal_l1.receipt.convergence.converged
        )
        if self.both_branches_converged is not converged:
            raise ValueError("aggregate convergence differs from branch receipts")
        if self.selected_branch is not None and not self.both_branches_converged:
            raise PermissionError("an unresolved branch pair cannot select a winner")
        if (
            self.branch_choice_certified_within_tolerance
            and self.selected_branch is None
        ):
            raise ValueError("certified branch choice requires a selected branch")
        if any(
            value is not False
            for value in (
                self.discrete_global_optimality_claim,
                self.nonlinear_remainder_authority,
                self.bce_oracle_authority,
                self.mask_commit_authority,
            )
        ):
            raise PermissionError("relaxed KKT result crossed its authority boundary")
        if self.relaxed_union_optimality_only is not True:
            raise PermissionError(
                "relaxed KKT result cannot claim hard-mask optimality"
            )
        if self.schema != RELAXED_KKT_RESULT_SCHEMA:
            raise ValueError("unsupported relaxed KKT result schema")

    @property
    def result_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "problem_hash": self.problem_hash,
                "policy_hash": self.policy_hash,
                "spectral_result_hash": self.spectral.result_hash,
                "diagonal_l1_result_hash": self.diagonal_l1.result_hash,
                "selected_branch": self.selected_branch,
                "both_branches_converged": self.both_branches_converged,
                "branch_choice_certified_within_tolerance": (
                    self.branch_choice_certified_within_tolerance
                ),
                "relaxed_union_optimality_only": self.relaxed_union_optimality_only,
                "discrete_global_optimality_claim": (
                    self.discrete_global_optimality_claim
                ),
            },
            prefix="partition-free-v1-relaxed-kkt-result-",
        )


@dataclass(frozen=True, slots=True)
class V1RelaxedKKTCUDAExecutionTelemetry:
    """Authority-inert execution evidence for one CUDA selector invocation.

    Bulk transfer bytes count material tensor payloads initiated by this
    implementation.  Scalar ``item()`` synchronizations, CUDA kernel-launch
    parameters, and driver bookkeeping queries are intentionally excluded;
    these fields must not be interpreted as an exact all-transfer byte census.
    """

    result_hash: str
    forecast_hash: str
    execution_design_hash: str
    algorithm_id: str
    device: str
    route_axis_hash: str
    source_geometry_hash: str
    numerical_policy_hash: str
    route_count: int
    rank: int
    scan_only: bool
    forecast_seconds: float
    cold_h2d_seconds: float
    warm_solver_seconds: float
    warm_rounding_seconds: float
    warm_replay_seconds: float
    receipt_seconds: float
    total_wall_seconds: float
    factor_forward_scan_count: int
    factor_adjoint_scan_count: int
    factor_route_visit_count: int
    key_scan_count: int
    key_route_visit_count: int
    replay_factor_element_visit_count: int
    bulk_h2d_tensor_payload_bytes: int
    bulk_d2h_tensor_payload_bytes: int
    peak_allocated_hbm_bytes: int
    peak_reserved_hbm_bytes: int
    maximum_endpoint: int
    rounding_tile_size: int
    spectral_response_scale: float
    diagonal_l1_response_scale: float
    spectral_endpoint_counts: tuple[int, ...]
    diagonal_l1_endpoint_counts: tuple[int, ...]
    spectral_endpoint_index_hashes: tuple[str, ...]
    diagonal_l1_endpoint_index_hashes: tuple[str, ...]
    hard_ladder_emitted: bool
    adaptive_shell_witness_available: bool
    block_separable_repair_factor_forward_scan_count: int = field(
        default=0,
        kw_only=True,
    )
    nonlinear_remainder_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_claim_authority: bool = False
    schema: str = RELAXED_KKT_CUDA_TELEMETRY_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "result_hash",
            "forecast_hash",
            "execution_design_hash",
            "algorithm_id",
            "device",
            "route_axis_hash",
            "source_geometry_hash",
            "numerical_policy_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        for name in (
            "route_count",
            "factor_forward_scan_count",
            "block_separable_repair_factor_forward_scan_count",
            "factor_adjoint_scan_count",
            "factor_route_visit_count",
            "key_scan_count",
            "key_route_visit_count",
            "replay_factor_element_visit_count",
            "bulk_h2d_tensor_payload_bytes",
            "bulk_d2h_tensor_payload_bytes",
            "peak_allocated_hbm_bytes",
            "peak_reserved_hbm_bytes",
            "maximum_endpoint",
            "rounding_tile_size",
        ):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name))
        object.__setattr__(self, "rank", _nonnegative_int(self.rank, "rank"))
        if self.route_count <= 0 or self.rounding_tile_size <= 0:
            raise ValueError("telemetry route count and tile size must be positive")
        if (
            self.block_separable_repair_factor_forward_scan_count
            > self.factor_forward_scan_count
        ):
            raise ValueError(
                "block-separable repair scans cannot exceed total factor-forward "
                "scans"
            )
        if (
            self.block_separable_repair_factor_forward_scan_count
            and self.algorithm_id != RELAXED_KKT_FUSED_CUDA_ALGORITHM_ID
        ):
            raise ValueError(
                "block-separable repair scans require the fused CUDA algorithm"
            )
        for name in (
            "forecast_seconds",
            "cold_h2d_seconds",
            "warm_solver_seconds",
            "warm_rounding_seconds",
            "warm_replay_seconds",
            "receipt_seconds",
            "total_wall_seconds",
            "spectral_response_scale",
            "diagonal_l1_response_scale",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        spectral_counts = tuple(
            _positive_int(item, "spectral_endpoint_counts item")
            for item in self.spectral_endpoint_counts
        )
        diagonal_counts = tuple(
            _positive_int(item, "diagonal_l1_endpoint_counts item")
            for item in self.diagonal_l1_endpoint_counts
        )
        spectral_hashes = tuple(
            _nonempty(item, "spectral endpoint index hash")
            for item in self.spectral_endpoint_index_hashes
        )
        diagonal_hashes = tuple(
            _nonempty(item, "diagonal endpoint index hash")
            for item in self.diagonal_l1_endpoint_index_hashes
        )
        if tuple(sorted(set(spectral_counts))) != spectral_counts:
            raise ValueError("spectral endpoint counts must be strictly increasing")
        if tuple(sorted(set(diagonal_counts))) != diagonal_counts:
            raise ValueError("diagonal endpoint counts must be strictly increasing")
        if len(spectral_counts) != len(spectral_hashes):
            raise ValueError("spectral endpoint counts/hashes differ in length")
        if len(diagonal_counts) != len(diagonal_hashes):
            raise ValueError("diagonal endpoint counts/hashes differ in length")
        object.__setattr__(self, "spectral_endpoint_counts", spectral_counts)
        object.__setattr__(self, "diagonal_l1_endpoint_counts", diagonal_counts)
        object.__setattr__(self, "spectral_endpoint_index_hashes", spectral_hashes)
        object.__setattr__(self, "diagonal_l1_endpoint_index_hashes", diagonal_hashes)
        for name in (
            "scan_only",
            "hard_ladder_emitted",
            "adaptive_shell_witness_available",
            "nonlinear_remainder_authority",
            "bce_oracle_authority",
            "mask_commit_authority",
            "terminal_authority",
            "pareto_claim_authority",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact bool")
        if self.scan_only and self.hard_ladder_emitted:
            raise PermissionError("scan-only execution emitted a hard ladder")
        if self.adaptive_shell_witness_available:
            raise PermissionError(
                "fused-v2 telemetry cannot claim an adaptive shell witness"
            )
        if any(
            getattr(self, name)
            for name in (
                "nonlinear_remainder_authority",
                "bce_oracle_authority",
                "mask_commit_authority",
                "terminal_authority",
                "pareto_claim_authority",
            )
        ):
            raise PermissionError("KKT telemetry crossed its authority boundary")
        if self.schema != RELAXED_KKT_CUDA_TELEMETRY_SCHEMA:
            raise ValueError("unsupported CUDA KKT telemetry schema")

    @property
    def telemetry_hash(self) -> str:
        return content_hash(
            self, prefix="partition-free-v1-relaxed-kkt-cuda-telemetry-"
        )


@dataclass(frozen=True, slots=True)
class V1RelaxedKKTCUDAExecutionBundle:
    """Versioned result-plus-execution evidence with no added authority."""

    result: V1RelaxedKKTResult
    forecast: V1RelaxedKKTCUDAHBMForecast
    telemetry: V1RelaxedKKTCUDAExecutionTelemetry
    execution_design: V1RelaxedKKTCUDAExecutionDesign
    geometry_uncertainty: V1RelaxedKKTGeometryUncertaintySet | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    adaptive_rank_witnesses: tuple[V1RelaxedKKTAdaptiveRankBranchWitness, ...] = ()
    schema: str = RELAXED_KKT_CUDA_EXECUTION_BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        if type(self.result) is not V1RelaxedKKTResult:
            raise TypeError("CUDA execution bundle requires an exact KKT result")
        if type(self.forecast) is not V1RelaxedKKTCUDAHBMForecast:
            raise TypeError("CUDA execution bundle requires an exact HBM forecast")
        if type(self.telemetry) is not V1RelaxedKKTCUDAExecutionTelemetry:
            raise TypeError("CUDA execution bundle requires exact telemetry")
        if type(self.execution_design) is not V1RelaxedKKTCUDAExecutionDesign:
            raise TypeError("CUDA execution bundle requires an exact design")
        if (
            self.result.result_hash != self.telemetry.result_hash
            or self.forecast.forecast_hash != self.telemetry.forecast_hash
            or self.execution_design.design_hash != self.telemetry.execution_design_hash
        ):
            raise ValueError(
                "CUDA result, forecast, design, and telemetry do not close"
            )
        uncertainty = self.geometry_uncertainty
        if uncertainty is not None:
            if type(uncertainty) is not V1RelaxedKKTGeometryUncertaintySet:
                raise TypeError("CUDA execution uncertainty must be exact")
            if (
                uncertainty.problem_hash != self.result.problem_hash
                or uncertainty.route_axis_hash != self.telemetry.route_axis_hash
                or uncertainty.source_geometry_hash
                != self.telemetry.source_geometry_hash
                or not uncertainty.source_derived_from_registered_geometry_receipts
            ):
                raise ValueError(
                    "CUDA execution uncertainty does not bind this source chart"
                )
        witnesses = tuple(self.adaptive_rank_witnesses)
        if any(
            type(value) is not V1RelaxedKKTAdaptiveRankBranchWitness
            for value in witnesses
        ):
            raise TypeError("CUDA execution witnesses must be exact")
        if witnesses:
            if uncertainty is None:
                raise PermissionError(
                    "adaptive-rank witnesses require source-derived uncertainty"
                )
            if tuple(value.branch.value for value in witnesses) != (
                "spectral",
                "diagonal_l1",
            ):
                raise ValueError("CUDA execution witnesses must use canonical branches")
            expected = {
                V1RelaxedKKTBranch.SPECTRAL: self.result.spectral,
                V1RelaxedKKTBranch.DIAGONAL_L1: self.result.diagonal_l1,
            }
            for witness in witnesses:
                branch_result = expected[witness.branch]
                if (
                    witness.problem_hash != self.result.problem_hash
                    or witness.policy_hash != self.result.policy_hash
                    or witness.branch_result_hash != branch_result.result_hash
                    or witness.branch_receipt_hash != branch_result.receipt.receipt_hash
                    or witness.rounded_ladder_hash
                    != branch_result.rounded_ladder.ladder_hash
                    or witness.route_axis_hash != self.telemetry.route_axis_hash
                    or witness.route_count != self.telemetry.route_count
                    or witness.finite_population_geometry_uncertainty_set_hash
                    != uncertainty.uncertainty_hash
                    or witness.literal_j_fixed_primal_direction_axis_hash
                    != uncertainty.literal_j_fixed_primal_direction_axis_hash
                ):
                    raise ValueError(
                        "CUDA execution witness does not bind this fused result"
                    )
        if self.execution_design.require_adaptive_shell_witness and len(witnesses) != 2:
            raise PermissionError(
                "CUDA execution required both source-derived adaptive-rank "
                "branch witnesses"
            )
        object.__setattr__(self, "adaptive_rank_witnesses", witnesses)
        if self.schema != RELAXED_KKT_CUDA_EXECUTION_BUNDLE_SCHEMA:
            raise ValueError("unsupported CUDA execution-bundle schema")

    @property
    def bundle_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "result_hash": self.result.result_hash,
                "forecast_hash": self.forecast.forecast_hash,
                "telemetry_hash": self.telemetry.telemetry_hash,
                "execution_design_hash": self.execution_design.design_hash,
                "geometry_uncertainty_hash": (
                    None
                    if self.geometry_uncertainty is None
                    else self.geometry_uncertainty.uncertainty_hash
                ),
                "adaptive_rank_witness_hashes": tuple(
                    witness.witness_hash for witness in self.adaptive_rank_witnesses
                ),
            },
            prefix="partition-free-v1-relaxed-kkt-cuda-execution-bundle-",
        )


@dataclass(frozen=True, slots=True)
class V1RelaxedKKTDeviceShellBranchIdentityReceipt:
    """Array-free identity of one branch's device-resident hard ladder.

    This receipt deliberately binds only deterministic rounding identity and
    registered hard legality.  In particular, it is not the boundary-face
    witness required for adaptive-rank decision stability and it contains no
    device handle or materialized hard action.
    """

    branch: V1RelaxedKKTBranch
    source_rounding_hash: str
    frozen_ladder: V1DeviceShellFrozenLadderReceipt
    rounding_key_id: str = RELAXED_KKT_ROUNDING_KEY_ID
    ordinary_hard_action_materialized: bool = False
    adaptive_shell_witness_available: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    production_authority: bool = False
    pareto_authority: bool = False
    schema: str = RELAXED_KKT_DEVICE_SHELL_BRANCH_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("device-shell branch receipt requires an exact branch")
        object.__setattr__(
            self,
            "source_rounding_hash",
            _nonempty(self.source_rounding_hash, "source_rounding_hash"),
        )
        if type(self.frozen_ladder) is not V1DeviceShellFrozenLadderReceipt:
            raise TypeError("device-shell branch receipt requires a frozen ladder")
        if self.frozen_ladder.source_ladder_hash != self.source_rounding_hash:
            raise ValueError("frozen ladder differs from its rounding source")
        if self.rounding_key_id != RELAXED_KKT_ROUNDING_KEY_ID:
            raise ValueError("unsupported device-shell rounding key")
        authority_fields = (
            "ordinary_hard_action_materialized",
            "adaptive_shell_witness_available",
            "c_stream_authority",
            "calibration_authority",
            "oracle_authority",
            "mask_commit_authority",
            "production_authority",
            "pareto_authority",
        )
        if any(type(getattr(self, name)) is not bool for name in authority_fields):
            raise TypeError("device-shell branch flags must be exact booleans")
        if any(getattr(self, name) for name in authority_fields):
            raise PermissionError(
                "device-shell branch identity crossed its authority boundary"
            )
        if self.schema != RELAXED_KKT_DEVICE_SHELL_BRANCH_RECEIPT_SCHEMA:
            raise ValueError("unsupported device-shell branch receipt schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-relaxed-kkt-device-shell-branch-receipt-",
        )

    @property
    def endpoint_action_axis_hash(self) -> str:
        frozen = self.frozen_ladder
        return content_hash(
            {
                "schema": "partition_free_v1_relaxed_kkt_device_shell_endpoint_action_axis_v1",
                "branch": self.branch,
                "source_rounding_hash": self.source_rounding_hash,
                "frozen_ladder_receipt_hash": frozen.receipt_hash,
                "route_axis_hash": frozen.route_axis_hash,
                "route_count": frozen.route_count,
                "parent_mask_hash": frozen.parent_mask_hash,
                "cumulative_counts": frozen.cumulative_counts,
                "frozen_actions": tuple(
                    {
                        "action_receipt_hash": action.receipt_hash,
                        "packed_action_hash": action.packed_action_hash,
                        "removed_route_index_hash": action.removed_route_index_hash,
                        "resulting_mask_packed_sha256": (
                            action.resulting_mask_packed_sha256
                        ),
                        "resulting_mask_hash": action.resulting_mask_hash,
                    }
                    for action in frozen.actions
                ),
            },
            prefix=("partition-free-v1-relaxed-kkt-device-shell-endpoint-action-axis-"),
        )


@dataclass(frozen=True, slots=True)
class V1RelaxedKKTDeviceShellExecutionReceipt:
    """Serializable, array-free evidence for the two preserved branch ladders."""

    problem_hash: str
    policy_hash: str
    device_shell_execution_design_hash: str
    cuda_execution_design_hash: str
    base_result_hash: str
    hbm_forecast_hash: str
    cuda_telemetry_hash: str
    spectral: V1RelaxedKKTDeviceShellBranchIdentityReceipt
    diagonal_l1: V1RelaxedKKTDeviceShellBranchIdentityReceipt
    device_label_vector_count: int
    full_axis_label_d2h_scan_count: int
    label_d2h_tensor_payload_bytes: int
    ordinary_hard_action_count: int = 0
    branch_deduplication_used: bool = False
    adaptive_shell_witness_available: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    production_authority: bool = False
    pareto_authority: bool = False
    schema: str = RELAXED_KKT_DEVICE_SHELL_EXECUTION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "problem_hash",
            "policy_hash",
            "device_shell_execution_design_hash",
            "cuda_execution_design_hash",
            "base_result_hash",
            "hbm_forecast_hash",
            "cuda_telemetry_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if type(self.spectral) is not V1RelaxedKKTDeviceShellBranchIdentityReceipt:
            raise TypeError("spectral device-shell identity receipt is required")
        if type(self.diagonal_l1) is not V1RelaxedKKTDeviceShellBranchIdentityReceipt:
            raise TypeError("diagonal-l1 device-shell identity receipt is required")
        if self.spectral.branch is not V1RelaxedKKTBranch.SPECTRAL:
            raise ValueError("spectral device-shell identity is mislabelled")
        if self.diagonal_l1.branch is not V1RelaxedKKTBranch.DIAGONAL_L1:
            raise ValueError("diagonal-l1 device-shell identity is mislabelled")
        if (
            self.spectral.frozen_ladder.route_axis_hash
            != self.diagonal_l1.frozen_ladder.route_axis_hash
            or self.spectral.frozen_ladder.parent_mask_hash
            != self.diagonal_l1.frozen_ladder.parent_mask_hash
            or self.spectral.frozen_ladder.route_count
            != self.diagonal_l1.frozen_ladder.route_count
        ):
            raise ValueError("device-shell branch identities do not share one parent")
        if (
            self.spectral.frozen_ladder.eligibility_commitment
            != self.diagonal_l1.frozen_ladder.eligibility_commitment
            or self.spectral.frozen_ladder.block_capacity_hash
            != self.diagonal_l1.frozen_ladder.block_capacity_hash
            or self.spectral.frozen_ladder.block_legality_scope
            != self.diagonal_l1.frozen_ladder.block_legality_scope
            or self.spectral.frozen_ladder.cumulative_counts
            != self.diagonal_l1.frozen_ladder.cumulative_counts
        ):
            raise ValueError(
                "device-shell branch identities do not share one registered ladder"
            )
        for name in (
            "device_label_vector_count",
            "full_axis_label_d2h_scan_count",
            "label_d2h_tensor_payload_bytes",
            "ordinary_hard_action_count",
        ):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name))
        route_count = self.spectral.frozen_ladder.route_count
        if self.device_label_vector_count != 2:
            raise ValueError("the two branch label vectors must be preserved")
        if self.full_axis_label_d2h_scan_count != 2:
            raise ValueError("each branch requires exactly one bounded label scan")
        if self.label_d2h_tensor_payload_bytes != 2 * route_count:
            raise ValueError("device-shell D2H accounting must equal two uint8 axes")
        if self.ordinary_hard_action_count != 0:
            raise PermissionError("device-shell execution materialized a hard action")
        authority_fields = (
            "branch_deduplication_used",
            "adaptive_shell_witness_available",
            "c_stream_authority",
            "calibration_authority",
            "oracle_authority",
            "mask_commit_authority",
            "production_authority",
            "pareto_authority",
        )
        if any(type(getattr(self, name)) is not bool for name in authority_fields):
            raise TypeError("device-shell execution flags must be exact booleans")
        if any(getattr(self, name) for name in authority_fields):
            raise PermissionError(
                "device-shell execution identity crossed its authority boundary"
            )
        if self.schema != RELAXED_KKT_DEVICE_SHELL_EXECUTION_RECEIPT_SCHEMA:
            raise ValueError("unsupported device-shell execution receipt schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-relaxed-kkt-device-shell-execution-receipt-",
        )


@dataclass(frozen=True, slots=True)
class V1RelaxedKKTGeometryUncertaintySet:
    """Typed uncertainty envelope for one KKT face-stability witness.

    This object is intentionally not a generic scalar slack container.  A
    source-derived instance must be issued by the registered geometry receipt
    builder that binds finite-population corrected work ``A`` and literal-J
    response geometry ``M_J`` on the same route axis.  Deterministic FD
    residuals and numerical tolerances remain explicit uncertainty sets; they
    do not spend alpha and they do not become statistical estimands.
    """

    uncertainty_id: str
    problem_hash: str
    route_axis_hash: str
    source_geometry_hash: str
    corrected_work_A_sampling_linf_radius: float = 0.0
    corrected_work_A_deterministic_linf_radius: float = 0.0
    literal_j_response_operator_MJ_sampling_linf_adjoint_radius: float = 0.0
    literal_j_response_operator_MJ_deterministic_linf_adjoint_radius: float = 0.0
    fd_q_residual_deterministic_linf_adjoint_radius: float = 0.0
    solver_reduced_advantage_numerical_linf_radius: float = 0.0
    relaxed_gate_numerical_linf_radius: float = 0.0
    corrected_work_ratio_sampling_linf_radius: float = 0.0
    corrected_work_ratio_deterministic_linf_radius: float = 0.0
    alpha_spent: float = 0.0
    literal_j_fixed_primal_direction_axis_hash: str = (
        "unbound_literal_j_fixed_primal_direction_axis"
    )
    literal_j_fixed_primal_direction_count: int = 0
    statistical_estimand_id: str = (
        "corrected_work_A_and_literal_J_response_operator_MJ_v1"
    )
    deterministic_fd_role_id: str = (
        "fd_q_residual_envelope_is_deterministic_uncertainty_not_estimand_v1"
    )
    literal_j_response_operator_is_statistical_estimand: bool = True
    gauge_dependent_low_rank_factor_is_statistical_estimand: bool = False
    source_derived_from_registered_geometry_receipts: bool = False
    _geometry_uncertainty_issuance_token: InitVar[object | None] = None
    schema: str = RELAXED_KKT_FACE_GEOMETRY_UNCERTAINTY_SCHEMA

    def __post_init__(
        self,
        _geometry_uncertainty_issuance_token: object | None,
    ) -> None:
        for name in (
            "uncertainty_id",
            "problem_hash",
            "route_axis_hash",
            "source_geometry_hash",
            "literal_j_fixed_primal_direction_axis_hash",
            "statistical_estimand_id",
            "deterministic_fd_role_id",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "literal_j_fixed_primal_direction_count",
            _nonnegative_int(
                self.literal_j_fixed_primal_direction_count,
                "literal_j_fixed_primal_direction_count",
            ),
        )
        for name in (
            "corrected_work_A_sampling_linf_radius",
            "corrected_work_A_deterministic_linf_radius",
            "literal_j_response_operator_MJ_sampling_linf_adjoint_radius",
            "literal_j_response_operator_MJ_deterministic_linf_adjoint_radius",
            "fd_q_residual_deterministic_linf_adjoint_radius",
            "solver_reduced_advantage_numerical_linf_radius",
            "relaxed_gate_numerical_linf_radius",
            "corrected_work_ratio_sampling_linf_radius",
            "corrected_work_ratio_deterministic_linf_radius",
            "alpha_spent",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        for name in (
            "literal_j_response_operator_is_statistical_estimand",
            "gauge_dependent_low_rank_factor_is_statistical_estimand",
            "source_derived_from_registered_geometry_receipts",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact bool")
        if (
            self.statistical_estimand_id
            != "corrected_work_A_and_literal_J_response_operator_MJ_v1"
        ):
            raise ValueError("unsupported KKT geometry statistical estimand")
        if (
            self.deterministic_fd_role_id
            != "fd_q_residual_envelope_is_deterministic_uncertainty_not_estimand_v1"
        ):
            raise ValueError("unsupported deterministic FD uncertainty role")
        if not self.literal_j_response_operator_is_statistical_estimand:
            raise PermissionError("literal-J response geometry must own M_J")
        if self.gauge_dependent_low_rank_factor_is_statistical_estimand:
            raise PermissionError(
                "gauge-dependent H may not be the statistical estimand"
            )
        if (
            self.source_derived_from_registered_geometry_receipts
            and _geometry_uncertainty_issuance_token
            is not _GEOMETRY_UNCERTAINTY_ISSUANCE_TOKEN
        ):
            raise PermissionError(
                "source-derived KKT geometry uncertainty requires the registered "
                "corrected-work/literal-J receipt issuer"
            )
        if self.source_derived_from_registered_geometry_receipts and (
            self.literal_j_fixed_primal_direction_axis_hash
            == "unbound_literal_j_fixed_primal_direction_axis"
            or self.literal_j_fixed_primal_direction_count <= 0
        ):
            raise PermissionError(
                "source-derived KKT geometry uncertainty must bind the fixed "
                "literal-J primal-direction axis"
            )
        if self.schema != RELAXED_KKT_FACE_GEOMETRY_UNCERTAINTY_SCHEMA:
            raise ValueError("unsupported relaxed KKT geometry uncertainty schema")

    def finite_population_face_radius(self, raw_work_dual: float) -> float:
        work_dual = abs(_finite(raw_work_dual, "raw_work_dual"))
        return (
            work_dual * self.corrected_work_A_sampling_linf_radius
            + self.literal_j_response_operator_MJ_sampling_linf_adjoint_radius
        )

    def deterministic_face_radius(self, raw_work_dual: float) -> float:
        work_dual = abs(_finite(raw_work_dual, "raw_work_dual"))
        return (
            work_dual * self.corrected_work_A_deterministic_linf_radius
            + self.literal_j_response_operator_MJ_deterministic_linf_adjoint_radius
            + self.fd_q_residual_deterministic_linf_adjoint_radius
            + self.solver_reduced_advantage_numerical_linf_radius
        )

    def relaxed_problem_perturbation_radius(self, raw_work_dual: float) -> float:
        """Bound perturbations that change the relaxed KKT problem itself.

        Solver/reduced-advantage numerical slack is deliberately excluded:
        small arithmetic uncertainty can be handled by routewise margins for
        the already solved problem.  Sampling, literal-J operator uncertainty,
        and deterministic FD residuals change the objective/response operator
        and therefore require an explicit solution-sensitivity certificate
        before a same-face/same-shell claim may open downstream C.
        """

        work_dual = abs(_finite(raw_work_dual, "raw_work_dual"))
        return (
            work_dual
            * (
                self.corrected_work_A_sampling_linf_radius
                + self.corrected_work_A_deterministic_linf_radius
            )
            + self.literal_j_response_operator_MJ_sampling_linf_adjoint_radius
            + self.literal_j_response_operator_MJ_deterministic_linf_adjoint_radius
            + self.fd_q_residual_deterministic_linf_adjoint_radius
        )

    def finite_population_key_radius(self, raw_work_dual: float) -> float:
        return (
            self.finite_population_face_radius(raw_work_dual)
            + self.corrected_work_ratio_sampling_linf_radius
        )

    def deterministic_key_radius(self, raw_work_dual: float) -> float:
        return (
            self.deterministic_face_radius(raw_work_dual)
            + self.relaxed_gate_numerical_linf_radius
            + self.corrected_work_ratio_deterministic_linf_radius
        )

    @property
    def uncertainty_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "uncertainty_id": self.uncertainty_id,
                "problem_hash": self.problem_hash,
                "route_axis_hash": self.route_axis_hash,
                "source_geometry_hash": self.source_geometry_hash,
                "corrected_work_A_sampling_linf_radius": (
                    self.corrected_work_A_sampling_linf_radius
                ),
                "corrected_work_A_deterministic_linf_radius": (
                    self.corrected_work_A_deterministic_linf_radius
                ),
                "literal_j_response_operator_MJ_sampling_linf_adjoint_radius": (
                    self.literal_j_response_operator_MJ_sampling_linf_adjoint_radius
                ),
                "literal_j_response_operator_MJ_deterministic_linf_adjoint_radius": (
                    self.literal_j_response_operator_MJ_deterministic_linf_adjoint_radius
                ),
                "fd_q_residual_deterministic_linf_adjoint_radius": (
                    self.fd_q_residual_deterministic_linf_adjoint_radius
                ),
                "solver_reduced_advantage_numerical_linf_radius": (
                    self.solver_reduced_advantage_numerical_linf_radius
                ),
                "relaxed_gate_numerical_linf_radius": (
                    self.relaxed_gate_numerical_linf_radius
                ),
                "corrected_work_ratio_sampling_linf_radius": (
                    self.corrected_work_ratio_sampling_linf_radius
                ),
                "corrected_work_ratio_deterministic_linf_radius": (
                    self.corrected_work_ratio_deterministic_linf_radius
                ),
                "alpha_spent": self.alpha_spent,
                "literal_j_fixed_primal_direction_axis_hash": (
                    self.literal_j_fixed_primal_direction_axis_hash
                ),
                "literal_j_fixed_primal_direction_count": (
                    self.literal_j_fixed_primal_direction_count
                ),
                "statistical_estimand_id": self.statistical_estimand_id,
                "deterministic_fd_role_id": self.deterministic_fd_role_id,
                "literal_j_response_operator_is_statistical_estimand": (
                    self.literal_j_response_operator_is_statistical_estimand
                ),
                "gauge_dependent_low_rank_factor_is_statistical_estimand": (
                    self.gauge_dependent_low_rank_factor_is_statistical_estimand
                ),
                "source_derived_from_registered_geometry_receipts": (
                    self.source_derived_from_registered_geometry_receipts
                ),
            },
            prefix="partition-free-v1-relaxed-kkt-geometry-uncertainty-",
        )


@dataclass(frozen=True, slots=True)
class V1RelaxedKKTRobustSolutionSensitivityCertificate:
    """Fail-closed certificate for nonzero relaxed-problem perturbations.

    The adaptive-rank witness owns margins for the nominal solved KKT chart.
    This sibling certificate is the only future mechanism allowed to say those
    margins remain valid when sampled ``A``, sampled/approximated ``M_J``, or
    deterministic FD residual uncertainty changes the relaxed branch problem.
    A positive theorem requires a full operator uncertainty receipt and a
    source-derived KKT conditioning proof.  Fixed-direction ``M_J z`` evidence
    is recorded as insufficient by construction.
    """

    branch: V1RelaxedKKTBranch
    problem_hash: str
    policy_hash: str
    branch_result_hash: str
    branch_receipt_hash: str
    route_axis_hash: str
    route_count: int
    finite_population_geometry_uncertainty_set_hash: str
    full_operator_uncertainty_receipt_hash: str | None = None
    kkt_jacobian_certificate_hash: str | None = None
    active_face_certificate_hash: str | None = None
    rounded_shell_certificate_hash: str | None = None
    perturbation_linf_radius_upper_bound: float = 0.0
    kkt_jacobian_inverse_linf_upper_bound: float = _ROBUST_SENSITIVITY_UNBOUNDED
    kkt_jacobian_lipschitz_linf_upper_bound: float = _ROBUST_SENSITIVITY_UNBOUNDED
    kantorovich_product_upper_bound: float = _ROBUST_SENSITIVITY_UNBOUNDED
    primal_linf_motion_upper_bound: float = _ROBUST_SENSITIVITY_UNBOUNDED
    reduced_advantage_linf_motion_upper_bound: float = _ROBUST_SENSITIVITY_UNBOUNDED
    rounding_key_linf_motion_upper_bound: float = _ROBUST_SENSITIVITY_UNBOUNDED
    minimum_relaxed_gate_boundary_margin_lower_bound: float = 0.0
    minimum_reduced_advantage_face_margin_lower_bound: float = 0.0
    minimum_rounding_key_boundary_margin_lower_bound: float = 0.0
    licq_verified: bool = False
    strict_complementarity_verified: bool = False
    second_order_growth_verified: bool = False
    polyhedral_normal_margin_verified: bool = False
    active_cone_smoothness_verified: bool = False
    critical_cone_growth_or_vertex_verified: bool = False
    unique_solution_on_registered_face: bool = False
    matrix_free_condition_bound_verified: bool = False
    full_operator_uncertainty_receipt_complete: bool = False
    fixed_direction_mjz_only: bool = True
    semidefinite_nonunique_branch_refused: bool = True
    source_derived_from_full_operator_uncertainty_receipts: bool = False
    full_kkt_solution_sensitivity_theorem: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    production_authority: bool = False
    pareto_authority: bool = False
    theorem_id: str = "strong_regular_variational_inequality_linf_perturbation_v1"
    authority_scope: str = RELAXED_KKT_ROBUST_SOLUTION_SENSITIVITY_SCOPE
    _robust_solution_sensitivity_issuance_token: InitVar[object | None] = None
    schema: str = RELAXED_KKT_ROBUST_SOLUTION_SENSITIVITY_SCHEMA

    def __post_init__(
        self,
        _robust_solution_sensitivity_issuance_token: object | None,
    ) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("robust KKT sensitivity requires an exact branch")
        for name in (
            "problem_hash",
            "policy_hash",
            "branch_result_hash",
            "branch_receipt_hash",
            "route_axis_hash",
            "finite_population_geometry_uncertainty_set_hash",
            "theorem_id",
            "authority_scope",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        for name in (
            "full_operator_uncertainty_receipt_hash",
            "kkt_jacobian_certificate_hash",
            "active_face_certificate_hash",
            "rounded_shell_certificate_hash",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _nonempty(value, name))
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, "route_count"),
        )
        for name in (
            "perturbation_linf_radius_upper_bound",
            "minimum_relaxed_gate_boundary_margin_lower_bound",
            "minimum_reduced_advantage_face_margin_lower_bound",
            "minimum_rounding_key_boundary_margin_lower_bound",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        for name in (
            "kkt_jacobian_inverse_linf_upper_bound",
            "kkt_jacobian_lipschitz_linf_upper_bound",
            "kantorovich_product_upper_bound",
            "primal_linf_motion_upper_bound",
            "reduced_advantage_linf_motion_upper_bound",
            "rounding_key_linf_motion_upper_bound",
        ):
            value = getattr(self, name)
            if not isinstance(value, Real):
                raise TypeError(f"{name} must be a real scalar")
            value = float(value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        bool_fields = (
            "licq_verified",
            "strict_complementarity_verified",
            "second_order_growth_verified",
            "polyhedral_normal_margin_verified",
            "active_cone_smoothness_verified",
            "critical_cone_growth_or_vertex_verified",
            "unique_solution_on_registered_face",
            "matrix_free_condition_bound_verified",
            "full_operator_uncertainty_receipt_complete",
            "fixed_direction_mjz_only",
            "semidefinite_nonunique_branch_refused",
            "source_derived_from_full_operator_uncertainty_receipts",
            "full_kkt_solution_sensitivity_theorem",
            "c_stream_authority",
            "calibration_authority",
            "oracle_authority",
            "mask_commit_authority",
            "production_authority",
            "pareto_authority",
        )
        if any(type(getattr(self, name)) is not bool for name in bool_fields):
            raise TypeError("robust KKT sensitivity flags must be exact booleans")
        if self.theorem_id != (
            "strong_regular_variational_inequality_linf_perturbation_v1"
        ):
            raise ValueError("unsupported robust KKT sensitivity theorem")
        if self.authority_scope != RELAXED_KKT_ROBUST_SOLUTION_SENSITIVITY_SCOPE:
            raise ValueError("unsupported robust KKT sensitivity authority scope")
        if self.schema != RELAXED_KKT_ROBUST_SOLUTION_SENSITIVITY_SCHEMA:
            raise ValueError("unsupported robust KKT sensitivity schema")
        if any(
            getattr(self, name)
            for name in (
                "c_stream_authority",
                "calibration_authority",
                "oracle_authority",
                "mask_commit_authority",
                "production_authority",
                "pareto_authority",
            )
        ):
            raise PermissionError(
                "robust KKT sensitivity certificate crossed its authority boundary"
            )
        if self.full_kkt_solution_sensitivity_theorem:
            if (
                _robust_solution_sensitivity_issuance_token
                is not _ROBUST_SOLUTION_SENSITIVITY_ISSUANCE_TOKEN
            ):
                raise PermissionError(
                    "positive robust KKT sensitivity requires the registered "
                    "source issuer"
                )
            if (
                not self.licq_verified
                or not self.strict_complementarity_verified
                or not (
                    self.second_order_growth_verified
                    or self.polyhedral_normal_margin_verified
                    or self.critical_cone_growth_or_vertex_verified
                )
                or not self.unique_solution_on_registered_face
                or not self.matrix_free_condition_bound_verified
                or not self.full_operator_uncertainty_receipt_complete
                or not self.source_derived_from_full_operator_uncertainty_receipts
                or self.fixed_direction_mjz_only
                or self.semidefinite_nonunique_branch_refused
                or self.full_operator_uncertainty_receipt_hash is None
                or self.kkt_jacobian_certificate_hash is None
                or self.active_face_certificate_hash is None
                or self.rounded_shell_certificate_hash is None
            ):
                raise PermissionError(
                    "positive robust KKT sensitivity lacks the required "
                    "regularity/operator receipts"
                )
            if (
                not math.isfinite(self.kkt_jacobian_inverse_linf_upper_bound)
                or not math.isfinite(self.kkt_jacobian_lipschitz_linf_upper_bound)
                or not math.isfinite(self.kantorovich_product_upper_bound)
                or not math.isfinite(self.primal_linf_motion_upper_bound)
                or not math.isfinite(self.reduced_advantage_linf_motion_upper_bound)
                or not math.isfinite(self.rounding_key_linf_motion_upper_bound)
            ):
                raise ValueError(
                    "positive robust KKT sensitivity needs finite motion bounds"
                )
            if self.kantorovich_product_upper_bound >= 0.5:
                raise ValueError(
                    "Kantorovich bound does not stay in the uniqueness basin"
                )
            if self.primal_linf_motion_upper_bound >= (
                self.minimum_relaxed_gate_boundary_margin_lower_bound
            ):
                raise ValueError("relaxed-gate motion can leave the registered face")
            if self.reduced_advantage_linf_motion_upper_bound >= (
                self.minimum_reduced_advantage_face_margin_lower_bound
            ):
                raise ValueError(
                    "reduced-advantage motion can flip the registered face"
                )
            if self.rounding_key_linf_motion_upper_bound >= (
                self.minimum_rounding_key_boundary_margin_lower_bound
            ):
                raise ValueError(
                    "rounding-key motion can flip a registered endpoint shell"
                )
        elif (
            _robust_solution_sensitivity_issuance_token
            is _ROBUST_SOLUTION_SENSITIVITY_ISSUANCE_TOKEN
        ):
            raise PermissionError(
                "registered robust KKT issuer may only emit positive theorem "
                "certificates"
            )

    @property
    def certificate_hash(self) -> str:
        def _hashable_real(value: float) -> float | str:
            if math.isinf(value):
                return "+inf" if value > 0.0 else "-inf"
            return value

        return content_hash(
            {
                "schema": self.schema,
                "branch": self.branch,
                "problem_hash": self.problem_hash,
                "policy_hash": self.policy_hash,
                "branch_result_hash": self.branch_result_hash,
                "branch_receipt_hash": self.branch_receipt_hash,
                "route_axis_hash": self.route_axis_hash,
                "route_count": self.route_count,
                "finite_population_geometry_uncertainty_set_hash": (
                    self.finite_population_geometry_uncertainty_set_hash
                ),
                "full_operator_uncertainty_receipt_hash": (
                    self.full_operator_uncertainty_receipt_hash
                ),
                "kkt_jacobian_certificate_hash": (self.kkt_jacobian_certificate_hash),
                "active_face_certificate_hash": (self.active_face_certificate_hash),
                "rounded_shell_certificate_hash": (self.rounded_shell_certificate_hash),
                "perturbation_linf_radius_upper_bound": (
                    _hashable_real(self.perturbation_linf_radius_upper_bound)
                ),
                "kkt_jacobian_inverse_linf_upper_bound": (
                    _hashable_real(self.kkt_jacobian_inverse_linf_upper_bound)
                ),
                "kkt_jacobian_lipschitz_linf_upper_bound": (
                    _hashable_real(self.kkt_jacobian_lipschitz_linf_upper_bound)
                ),
                "kantorovich_product_upper_bound": (
                    _hashable_real(self.kantorovich_product_upper_bound)
                ),
                "primal_linf_motion_upper_bound": (
                    _hashable_real(self.primal_linf_motion_upper_bound)
                ),
                "reduced_advantage_linf_motion_upper_bound": (
                    _hashable_real(self.reduced_advantage_linf_motion_upper_bound)
                ),
                "rounding_key_linf_motion_upper_bound": (
                    _hashable_real(self.rounding_key_linf_motion_upper_bound)
                ),
                "minimum_relaxed_gate_boundary_margin_lower_bound": (
                    _hashable_real(
                        self.minimum_relaxed_gate_boundary_margin_lower_bound
                    )
                ),
                "minimum_reduced_advantage_face_margin_lower_bound": (
                    _hashable_real(
                        self.minimum_reduced_advantage_face_margin_lower_bound
                    )
                ),
                "minimum_rounding_key_boundary_margin_lower_bound": (
                    _hashable_real(
                        self.minimum_rounding_key_boundary_margin_lower_bound
                    )
                ),
                "licq_verified": self.licq_verified,
                "strict_complementarity_verified": (
                    self.strict_complementarity_verified
                ),
                "second_order_growth_verified": (self.second_order_growth_verified),
                "polyhedral_normal_margin_verified": (
                    self.polyhedral_normal_margin_verified
                ),
                "active_cone_smoothness_verified": (
                    self.active_cone_smoothness_verified
                ),
                "critical_cone_growth_or_vertex_verified": (
                    self.critical_cone_growth_or_vertex_verified
                ),
                "unique_solution_on_registered_face": (
                    self.unique_solution_on_registered_face
                ),
                "matrix_free_condition_bound_verified": (
                    self.matrix_free_condition_bound_verified
                ),
                "full_operator_uncertainty_receipt_complete": (
                    self.full_operator_uncertainty_receipt_complete
                ),
                "fixed_direction_mjz_only": self.fixed_direction_mjz_only,
                "semidefinite_nonunique_branch_refused": (
                    self.semidefinite_nonunique_branch_refused
                ),
                "source_derived_from_full_operator_uncertainty_receipts": (
                    self.source_derived_from_full_operator_uncertainty_receipts
                ),
                "full_kkt_solution_sensitivity_theorem": (
                    self.full_kkt_solution_sensitivity_theorem
                ),
                "theorem_id": self.theorem_id,
                "authority_scope": self.authority_scope,
            },
            prefix="partition-free-v1-relaxed-kkt-robust-sensitivity-",
        )


def explain_v1_relaxed_kkt_robust_solution_sensitivity_gap(
    *,
    problem: V1RelaxedKKTProblem,
    branch_result: V1RelaxedKKTBranchResult,
    geometry_uncertainty: V1RelaxedKKTGeometryUncertaintySet,
) -> V1RelaxedKKTRobustSolutionSensitivityCertificate:
    """Return an inert receipt explaining why fixed-z evidence cannot open C."""

    if type(problem) is not V1RelaxedKKTProblem:
        raise TypeError("robust sensitivity gap requires an exact problem")
    if type(branch_result) is not V1RelaxedKKTBranchResult:
        raise TypeError("robust sensitivity gap requires an exact branch result")
    if type(geometry_uncertainty) is not V1RelaxedKKTGeometryUncertaintySet:
        raise TypeError("robust sensitivity gap requires exact uncertainty")
    if (
        geometry_uncertainty.problem_hash != problem.problem_hash
        or branch_result.receipt.problem_hash != problem.problem_hash
        or geometry_uncertainty.route_axis_hash != problem.route_axis_hash
    ):
        _refuse(
            V1RelaxedKKTRefusalCode.ROBUST_SOLUTION_SENSITIVITY_UNAVAILABLE,
            "geometry uncertainty, branch result, and problem do not close",
        )
    perturbation = geometry_uncertainty.relaxed_problem_perturbation_radius(
        branch_result.receipt.stationarity.work_dual
    )
    return V1RelaxedKKTRobustSolutionSensitivityCertificate(
        branch=branch_result.branch,
        problem_hash=branch_result.receipt.problem_hash,
        policy_hash=branch_result.receipt.policy_hash,
        branch_result_hash=branch_result.result_hash,
        branch_receipt_hash=branch_result.receipt.receipt_hash,
        route_axis_hash=geometry_uncertainty.route_axis_hash,
        route_count=problem.route_count,
        finite_population_geometry_uncertainty_set_hash=(
            geometry_uncertainty.uncertainty_hash
        ),
        perturbation_linf_radius_upper_bound=perturbation,
    )


def _validated_robust_solution_sensitivity_certificate_hash(
    *,
    certificate: V1RelaxedKKTRobustSolutionSensitivityCertificate | None,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    branch_result: V1RelaxedKKTBranchResult,
    geometry_uncertainty: V1RelaxedKKTGeometryUncertaintySet,
) -> str | None:
    if certificate is None:
        return None
    if type(certificate) is not V1RelaxedKKTRobustSolutionSensitivityCertificate:
        raise TypeError(
            "relaxed solution sensitivity certificate must use the exact type"
        )
    if (
        not certificate.full_kkt_solution_sensitivity_theorem
        or not certificate.source_derived_from_full_operator_uncertainty_receipts
        or certificate.fixed_direction_mjz_only
        or certificate.branch is not branch_result.branch
        or certificate.problem_hash != problem.problem_hash
        or certificate.policy_hash != policy.policy_hash
        or certificate.branch_result_hash != branch_result.result_hash
        or certificate.branch_receipt_hash != branch_result.receipt.receipt_hash
        or certificate.route_axis_hash != problem.route_axis_hash
        or certificate.route_count != problem.route_count
        or certificate.finite_population_geometry_uncertainty_set_hash
        != geometry_uncertainty.uncertainty_hash
    ):
        _refuse(
            V1RelaxedKKTRefusalCode.ROBUST_SOLUTION_SENSITIVITY_UNAVAILABLE,
            "robust KKT sensitivity certificate does not bind this solved branch "
            "and full-operator uncertainty set",
        )
    return certificate.certificate_hash


def _decision_level_interval_array(
    value: object,
    *,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    """Copy one finite reference interval axis into immutable float64 bytes."""

    try:
        raw = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be representable as float64") from error
    if raw.shape != shape or not np.isfinite(raw).all():
        raise ValueError(f"{name} must be one finite array with shape {shape!r}")
    return _readonly_copy(raw, dtype=np.dtype(np.float64))


def _decision_level_interval_pair(
    lower: object,
    upper: object,
    *,
    shape: tuple[int, ...],
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    lo = _decision_level_interval_array(lower, shape=shape, name=f"{name} lower")
    hi = _decision_level_interval_array(upper, shape=shape, name=f"{name} upper")
    if bool(np.any(lo > hi)):
        raise ValueError(f"{name} interval is reversed")
    return lo, hi


def _decision_level_fraction_bound(total: Fraction, *, upper: bool) -> float:
    """Return the nearest finite binary64 bound enclosing an exact rational."""

    try:
        rounded = float(total)
    except (OverflowError, ValueError) as error:
        raise ValueError("decision-level interval sum is outside binary64") from error
    if not math.isfinite(rounded):
        raise ValueError("decision-level interval sum is outside binary64")
    rounded_fraction = Fraction.from_float(rounded)
    if upper:
        if rounded_fraction < total:
            rounded = math.nextafter(rounded, math.inf)
            if not math.isfinite(rounded) or Fraction.from_float(rounded) < total:
                raise ValueError("decision-level outward enclosure left binary64")
    elif rounded_fraction > total:
        rounded = math.nextafter(rounded, -math.inf)
        if not math.isfinite(rounded) or Fraction.from_float(rounded) > total:
            raise ValueError("decision-level outward enclosure left binary64")
    return rounded


def _decision_level_outward_sum(values: Sequence[float], *, upper: bool) -> float:
    """Exactly accumulate finite binary64 inputs and return an enclosing bound.

    The decision-level reference theorem is intentionally bounded rather than
    route-scale production machinery.  It can therefore afford exact rational
    accumulation; using ``longdouble`` alone is not a proof-safe enclosure under
    high-dynamic-range cancellation.
    """

    total = Fraction(0, 1)
    for offset, value in enumerate(values):
        total += Fraction.from_float(
            _finite(value, f"decision-level sum term {offset}")
        )
    return _decision_level_fraction_bound(total, upper=upper)


def _decision_level_interval_square(lower: float, upper: float) -> tuple[float, float]:
    if lower > upper or not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("decision-level square interval is invalid")
    if lower <= 0.0 <= upper:
        lo = 0.0
    else:
        lo = math.nextafter(min(abs(lower), abs(upper)) ** 2, -math.inf)
        lo = max(0.0, lo)
    hi = math.nextafter(max(abs(lower), abs(upper)) ** 2, math.inf)
    if not math.isfinite(hi):
        raise ValueError("decision-level square interval overflowed")
    return lo, hi


@dataclass(frozen=True, slots=True)
class V1RelaxedKKTDecisionLevelShellReferenceEvidence:
    """Array-backed evidence consumed only by the bounded reference theorem.

    This carrier has no authority.  It deliberately keeps dense routewise key
    intervals and small-shell factor intervals out of durable receipts.  A
    production CUDA issuer can later produce the same compact receipt family
    from tiled scans without retaining these arrays or a dense ``M_BB``.
    """

    evidence_id: str
    problem_hash: str
    policy_hash: str
    branch_result_hash: str
    branch_receipt_hash: str
    finite_population_geometry_uncertainty_set_hash: str
    full_operator_basis_census_receipt_hash: str
    rounded_ladder_hash: str
    endpoint_action_axis_hash: str
    route_axis_hash: str
    shell_route_indices: np.ndarray = field(repr=False, compare=False)
    canonical_real_key_lower: np.ndarray = field(repr=False, compare=False)
    canonical_real_key_upper: np.ndarray = field(repr=False, compare=False)
    restricted_factor_lower: np.ndarray = field(repr=False, compare=False)
    restricted_factor_upper: np.ndarray = field(repr=False, compare=False)
    factor_offset_lower: np.ndarray = field(repr=False, compare=False)
    factor_offset_upper: np.ndarray = field(repr=False, compare=False)
    corrected_work_offset_lower: float
    corrected_work_offset_upper: float
    corrected_work_shell_lower: np.ndarray = field(repr=False, compare=False)
    corrected_work_shell_upper: np.ndarray = field(repr=False, compare=False)
    residual_direction_lower: np.ndarray = field(repr=False, compare=False)
    residual_direction_upper: np.ndarray = field(repr=False, compare=False)
    residual_offset_lower: float
    residual_offset_upper: float
    spectral_rho_lower: float
    spectral_rho_upper: float
    robust_primal_resource_lower_bound: float
    robust_conic_dual_resource_upper_bound: float
    hard_q_additive_radius_upper_bound: float
    registered_endpoint_removed_indices: tuple[np.ndarray, ...] = field(
        repr=False,
        compare=False,
    )
    schema: str = (
        "partition_free_v1_relaxed_kkt_decision_level_shell_reference_evidence_v1"
    )
    source_derived_from_registered_uncertainty: bool = False
    _issuance_token: InitVar[object | None] = None
    _evidence_hash: str = field(init=False, default="", repr=False, compare=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        object.__setattr__(
            self, "evidence_id", _nonempty(self.evidence_id, "evidence_id")
        )
        for name in (
            "problem_hash",
            "policy_hash",
            "branch_result_hash",
            "branch_receipt_hash",
            "finite_population_geometry_uncertainty_set_hash",
            "full_operator_basis_census_receipt_hash",
            "rounded_ladder_hash",
            "endpoint_action_axis_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "route_axis_hash",
            _nonempty(self.route_axis_hash, "route_axis_hash"),
        )
        raw_keys = np.asarray(self.canonical_real_key_lower)
        if raw_keys.ndim != 2 or raw_keys.shape[1] != 3 or raw_keys.shape[0] <= 0:
            raise ValueError("canonical real-key intervals must have shape (routes, 3)")
        route_count = int(raw_keys.shape[0])
        key_lo, key_hi = _decision_level_interval_pair(
            self.canonical_real_key_lower,
            self.canonical_real_key_upper,
            shape=(route_count, 3),
            name="canonical real-key",
        )
        raw_shell = np.asarray(self.shell_route_indices)
        if raw_shell.ndim != 1 or raw_shell.dtype.kind not in {"i", "u"}:
            raise ValueError("shell route indices must be one integer vector")
        shell = _readonly_copy(raw_shell, dtype=np.dtype(np.int64))
        if (
            shell.size == 0
            or int(np.min(shell)) < 0
            or int(np.max(shell)) >= route_count
            or len(set(int(value) for value in shell)) != int(shell.size)
        ):
            raise ValueError(
                "shell route indices must be nonempty, unique, and in range"
            )
        shell_count = int(shell.size)
        raw_factor = np.asarray(self.restricted_factor_lower)
        if raw_factor.ndim != 2 or raw_factor.shape[1] != shell_count:
            raise ValueError("restricted factor must have shape (rank, shell_count)")
        rank = int(raw_factor.shape[0])
        factor_lo, factor_hi = _decision_level_interval_pair(
            self.restricted_factor_lower,
            self.restricted_factor_upper,
            shape=(rank, shell_count),
            name="restricted factor",
        )
        factor_offset_lo, factor_offset_hi = _decision_level_interval_pair(
            self.factor_offset_lower,
            self.factor_offset_upper,
            shape=(rank,),
            name="factor offset",
        )
        work_lo, work_hi = _decision_level_interval_pair(
            self.corrected_work_shell_lower,
            self.corrected_work_shell_upper,
            shape=(shell_count,),
            name="corrected-work shell",
        )
        residual_lo, residual_hi = _decision_level_interval_pair(
            self.residual_direction_lower,
            self.residual_direction_upper,
            shape=(shell_count,),
            name="residual direction",
        )
        for lower_name, upper_name in (
            ("corrected_work_offset_lower", "corrected_work_offset_upper"),
            ("residual_offset_lower", "residual_offset_upper"),
            ("spectral_rho_lower", "spectral_rho_upper"),
        ):
            lower = _finite(getattr(self, lower_name), lower_name)
            upper = _finite(getattr(self, upper_name), upper_name)
            if lower > upper:
                raise ValueError(f"{lower_name}/{upper_name} interval is reversed")
            object.__setattr__(self, lower_name, lower)
            object.__setattr__(self, upper_name, upper)
        for name in (
            "corrected_work_offset_lower",
            "corrected_work_offset_upper",
            "spectral_rho_lower",
            "spectral_rho_upper",
            "robust_primal_resource_lower_bound",
            "robust_conic_dual_resource_upper_bound",
            "hard_q_additive_radius_upper_bound",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        if (
            self.robust_conic_dual_resource_upper_bound
            < self.robust_primal_resource_lower_bound
        ):
            raise ValueError(
                "robust conic dual bound lies below the robust primal bound"
            )
        endpoints: list[np.ndarray] = []
        for offset, value in enumerate(self.registered_endpoint_removed_indices):
            raw = np.asarray(value)
            if raw.ndim != 1 or raw.dtype.kind not in {"i", "u"}:
                raise ValueError(
                    f"registered endpoint {offset} is not an integer vector"
                )
            normalized = np.unique(raw.astype(np.int64, copy=False))
            if (
                normalized.size != raw.size
                or (normalized.size and int(normalized[0]) < 0)
                or (normalized.size and int(normalized[-1]) >= route_count)
            ):
                raise ValueError(
                    "registered endpoint indices must be unique and in range"
                )
            endpoints.append(_readonly_copy(normalized, dtype=np.dtype(np.int64)))
        if not endpoints:
            raise ValueError("reference evidence requires every registered endpoint")
        object.__setattr__(self, "shell_route_indices", shell)
        object.__setattr__(self, "canonical_real_key_lower", key_lo)
        object.__setattr__(self, "canonical_real_key_upper", key_hi)
        object.__setattr__(self, "restricted_factor_lower", factor_lo)
        object.__setattr__(self, "restricted_factor_upper", factor_hi)
        object.__setattr__(self, "factor_offset_lower", factor_offset_lo)
        object.__setattr__(self, "factor_offset_upper", factor_offset_hi)
        object.__setattr__(self, "corrected_work_shell_lower", work_lo)
        object.__setattr__(self, "corrected_work_shell_upper", work_hi)
        object.__setattr__(self, "residual_direction_lower", residual_lo)
        object.__setattr__(self, "residual_direction_upper", residual_hi)
        object.__setattr__(
            self, "registered_endpoint_removed_indices", tuple(endpoints)
        )
        if type(self.source_derived_from_registered_uncertainty) is not bool:
            raise TypeError("source-derived reference evidence flag must be exact")
        if (
            not self.source_derived_from_registered_uncertainty
            or _issuance_token is not _DECISION_LEVEL_SOURCE_RECEIPT_ISSUANCE_TOKEN
        ):
            raise PermissionError(
                "bindable decision-level evidence requires the registered numerical "
                "uncertainty source issuer"
            )
        if self.schema != (
            "partition_free_v1_relaxed_kkt_decision_level_shell_reference_evidence_v1"
        ):
            raise ValueError("unsupported decision-level reference evidence schema")
        object.__setattr__(self, "_evidence_hash", self._compute_hash())

    @property
    def route_count(self) -> int:
        return int(self.canonical_real_key_lower.shape[0])

    @property
    def shell_count(self) -> int:
        return int(self.shell_route_indices.size)

    @property
    def evidence_hash(self) -> str:
        return self.assert_integrity()

    def _compute_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "evidence_id": self.evidence_id,
                "problem_hash": self.problem_hash,
                "policy_hash": self.policy_hash,
                "branch_result_hash": self.branch_result_hash,
                "branch_receipt_hash": self.branch_receipt_hash,
                "finite_population_geometry_uncertainty_set_hash": (
                    self.finite_population_geometry_uncertainty_set_hash
                ),
                "full_operator_basis_census_receipt_hash": (
                    self.full_operator_basis_census_receipt_hash
                ),
                "rounded_ladder_hash": self.rounded_ladder_hash,
                "endpoint_action_axis_hash": self.endpoint_action_axis_hash,
                "route_axis_hash": self.route_axis_hash,
                "shell_route_indices": self.shell_route_indices,
                "canonical_real_key_lower": self.canonical_real_key_lower,
                "canonical_real_key_upper": self.canonical_real_key_upper,
                "restricted_factor_lower": self.restricted_factor_lower,
                "restricted_factor_upper": self.restricted_factor_upper,
                "factor_offset_lower": self.factor_offset_lower,
                "factor_offset_upper": self.factor_offset_upper,
                "corrected_work_offset_lower": self.corrected_work_offset_lower,
                "corrected_work_offset_upper": self.corrected_work_offset_upper,
                "corrected_work_shell_lower": self.corrected_work_shell_lower,
                "corrected_work_shell_upper": self.corrected_work_shell_upper,
                "residual_direction_lower": self.residual_direction_lower,
                "residual_direction_upper": self.residual_direction_upper,
                "residual_offset_lower": self.residual_offset_lower,
                "residual_offset_upper": self.residual_offset_upper,
                "spectral_rho_lower": self.spectral_rho_lower,
                "spectral_rho_upper": self.spectral_rho_upper,
                "robust_primal_resource_lower_bound": (
                    self.robust_primal_resource_lower_bound
                ),
                "robust_conic_dual_resource_upper_bound": (
                    self.robust_conic_dual_resource_upper_bound
                ),
                "hard_q_additive_radius_upper_bound": (
                    self.hard_q_additive_radius_upper_bound
                ),
                "registered_endpoint_index_hashes": tuple(
                    RemovedRouteIndexSet.from_indices(
                        value,
                        route_count=self.route_count,
                    ).index_hash
                    for value in self.registered_endpoint_removed_indices
                ),
                "source_derived_from_registered_uncertainty": (
                    self.source_derived_from_registered_uncertainty
                ),
            },
            prefix="partition-free-v1-relaxed-kkt-decision-shell-evidence-",
        )

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_hash()
        if self._evidence_hash != observed:
            raise ValueError("decision-level source evidence was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("decision-level source evidence hash does not match")
        return observed


@dataclass(frozen=True, slots=True)
class V1RelaxedKKTOutsideShellIntervalScanReceipt:
    """Compact result of a complete eligible-route outside-shell scan."""

    branch: V1RelaxedKKTBranch
    problem_hash: str
    policy_hash: str
    route_axis_hash: str
    route_count: int
    source_evidence_hash: str
    shell_route_index_hash: str
    shell_count: int
    scanned_eligible_outside_route_count: int
    fixed_prefix_route_index_hash: str
    fixed_prefix_count: int
    fixed_suffix_route_index_hash: str
    fixed_suffix_count: int
    minimum_outside_reduced_advantage_margin_lower_bound: float
    minimum_outside_real_key_margin_lower_bound: float
    maximum_numerical_key_radius_upper_bound: float
    scan_backend: Literal["high_precision_reference", "torch_cuda"]
    source_derived_from_complete_outside_intervals: bool = True
    boundary_touching_route_refused: bool = True
    _issuance_token: InitVar[object | None] = None
    schema: str = RELAXED_KKT_OUTSIDE_SHELL_INTERVAL_SCAN_SCHEMA

    def __post_init__(self, _issuance_token: object | None) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("outside-shell scan requires an exact branch")
        for name in (
            "problem_hash",
            "policy_hash",
            "route_axis_hash",
            "source_evidence_hash",
            "shell_route_index_hash",
            "fixed_prefix_route_index_hash",
            "fixed_suffix_route_index_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        for name in (
            "route_count",
            "shell_count",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        for name in (
            "scanned_eligible_outside_route_count",
            "fixed_prefix_count",
            "fixed_suffix_count",
        ):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name))
        for name in (
            "minimum_outside_reduced_advantage_margin_lower_bound",
            "minimum_outside_real_key_margin_lower_bound",
            "maximum_numerical_key_radius_upper_bound",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        if self.scan_backend not in {"high_precision_reference", "torch_cuda"}:
            raise ValueError("unsupported outside-shell scan backend")
        if (
            type(self.source_derived_from_complete_outside_intervals) is not bool
            or type(self.boundary_touching_route_refused) is not bool
        ):
            raise TypeError("outside-shell source flags must be exact booleans")
        if (
            not self.source_derived_from_complete_outside_intervals
            or not self.boundary_touching_route_refused
            or _issuance_token is not _DECISION_LEVEL_SOURCE_RECEIPT_ISSUANCE_TOKEN
        ):
            raise PermissionError(
                "outside-shell scan must come from the source verifier"
            )
        if self.scanned_eligible_outside_route_count and (
            self.minimum_outside_reduced_advantage_margin_lower_bound <= 0.0
            or self.minimum_outside_real_key_margin_lower_bound <= 0.0
            or self.minimum_outside_real_key_margin_lower_bound
            <= 2.0 * self.maximum_numerical_key_radius_upper_bound
        ):
            raise PermissionError("outside-shell intervals touch a decision boundary")
        if self.schema != RELAXED_KKT_OUTSIDE_SHELL_INTERVAL_SCAN_SCHEMA:
            raise ValueError("unsupported outside-shell interval scan schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {item.name: getattr(self, item.name) for item in fields(self)},
            prefix="partition-free-v1-relaxed-kkt-outside-shell-scan-",
        )


@dataclass(frozen=True, slots=True)
class V1RelaxedKKTActiveShellRestrictedOperatorReceipt:
    """Factored active-shell operator and fixed-prefix offset enclosure.

    The durable representation is ``O(|B| r)``: it binds the restricted
    low-rank factor, branch residual direction, fixed-prefix offsets, and one
    outward operator-norm radius.  A dense ``M_BB`` digest is optional and is
    emitted only by the bounded reference verifier.
    """

    branch: V1RelaxedKKTBranch
    problem_hash: str
    policy_hash: str
    route_axis_hash: str
    source_evidence_hash: str
    shell_route_index_hash: str
    shell_count: int
    factor_rank: int
    restricted_factor_interval_hash: str
    factor_offset_interval_hash: str
    residual_branch_interval_hash: str
    corrected_work_offset_interval_hash: str
    hard_q_offset_interval_hash: str
    shell_operator_norm_sampling_radius_upper_bound: float
    shell_operator_norm_deterministic_radius_upper_bound: float
    strong_convexity_lower_bound: float
    dense_reference_operator_interval_hash: str | None = None
    representation_id: str = "low_rank_outward_enclosed_shell_operator_v1"
    source_derived_from_restricted_factor_and_offsets: bool = True
    _issuance_token: InitVar[object | None] = None
    schema: str = RELAXED_KKT_ACTIVE_SHELL_RESTRICTED_OPERATOR_SCHEMA

    def __post_init__(self, _issuance_token: object | None) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("restricted shell operator requires an exact branch")
        for name in (
            "problem_hash",
            "policy_hash",
            "route_axis_hash",
            "source_evidence_hash",
            "shell_route_index_hash",
            "restricted_factor_interval_hash",
            "factor_offset_interval_hash",
            "residual_branch_interval_hash",
            "corrected_work_offset_interval_hash",
            "hard_q_offset_interval_hash",
            "representation_id",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if self.dense_reference_operator_interval_hash is not None:
            object.__setattr__(
                self,
                "dense_reference_operator_interval_hash",
                _nonempty(
                    self.dense_reference_operator_interval_hash,
                    "dense_reference_operator_interval_hash",
                ),
            )
        object.__setattr__(
            self, "shell_count", _positive_int(self.shell_count, "shell_count")
        )
        object.__setattr__(
            self, "factor_rank", _nonnegative_int(self.factor_rank, "factor_rank")
        )
        for name in (
            "shell_operator_norm_sampling_radius_upper_bound",
            "shell_operator_norm_deterministic_radius_upper_bound",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        object.__setattr__(
            self,
            "strong_convexity_lower_bound",
            _finite(self.strong_convexity_lower_bound, "strong_convexity_lower_bound"),
        )
        if self.representation_id != "low_rank_outward_enclosed_shell_operator_v1":
            raise ValueError("restricted shell operator must remain low-rank/factored")
        if (
            type(self.source_derived_from_restricted_factor_and_offsets) is not bool
            or not self.source_derived_from_restricted_factor_and_offsets
            or _issuance_token is not _DECISION_LEVEL_SOURCE_RECEIPT_ISSUANCE_TOKEN
        ):
            raise PermissionError(
                "restricted shell operator requires the source verifier"
            )
        if self.schema != RELAXED_KKT_ACTIVE_SHELL_RESTRICTED_OPERATOR_SCHEMA:
            raise ValueError("unsupported restricted shell operator schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {item.name: getattr(self, item.name) for item in fields(self)},
            prefix="partition-free-v1-relaxed-kkt-restricted-shell-operator-",
        )


@dataclass(frozen=True, slots=True)
class V1RelaxedKKTRobustDualBoundReceipt:
    """Strict source-derived primal/dual enclosure for the restricted shell."""

    branch: V1RelaxedKKTBranch
    problem_hash: str
    policy_hash: str
    branch_result_hash: str
    branch_receipt_hash: str
    source_evidence_hash: str
    restricted_operator_receipt_hash: str
    robust_primal_resource_lower_bound: float
    robust_conic_dual_resource_upper_bound: float
    robust_duality_gap_upper_bound: float
    registered_strict_gap_tolerance: float
    strict_gap_margin_lower_bound: float
    source_derived_from_outward_primal_dual_bounds: bool = True
    _issuance_token: InitVar[object | None] = None
    schema: str = RELAXED_KKT_ROBUST_DUAL_BOUND_SCHEMA

    def __post_init__(self, _issuance_token: object | None) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("robust dual bound requires an exact branch")
        for name in (
            "problem_hash",
            "policy_hash",
            "branch_result_hash",
            "branch_receipt_hash",
            "source_evidence_hash",
            "restricted_operator_receipt_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        for name in (
            "robust_primal_resource_lower_bound",
            "robust_conic_dual_resource_upper_bound",
            "robust_duality_gap_upper_bound",
            "registered_strict_gap_tolerance",
            "strict_gap_margin_lower_bound",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        if (
            type(self.source_derived_from_outward_primal_dual_bounds) is not bool
            or not self.source_derived_from_outward_primal_dual_bounds
            or _issuance_token is not _DECISION_LEVEL_SOURCE_RECEIPT_ISSUANCE_TOKEN
        ):
            raise PermissionError("robust dual bound requires the source verifier")
        if (
            self.robust_conic_dual_resource_upper_bound
            < self.robust_primal_resource_lower_bound
            or self.robust_duality_gap_upper_bound
            < self.robust_conic_dual_resource_upper_bound
            - self.robust_primal_resource_lower_bound
            or self.robust_duality_gap_upper_bound
            >= self.registered_strict_gap_tolerance
            or self.strict_gap_margin_lower_bound
            > self.registered_strict_gap_tolerance - self.robust_duality_gap_upper_bound
            or self.strict_gap_margin_lower_bound <= 0.0
        ):
            raise PermissionError(
                "robust primal/dual interval has no strict gap margin"
            )
        if self.schema != RELAXED_KKT_ROBUST_DUAL_BOUND_SCHEMA:
            raise ValueError("unsupported robust dual-bound schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {item.name: getattr(self, item.name) for item in fields(self)},
            prefix="partition-free-v1-relaxed-kkt-robust-dual-bound-",
        )


@dataclass(frozen=True, slots=True)
class V1RelaxedKKTWorstCaseHardFQReplayReceipt:
    """Worst-case F/Q status for the one ladder shared by every shell face."""

    branch: V1RelaxedKKTBranch
    problem_hash: str
    policy_hash: str
    source_evidence_hash: str
    rounded_ladder_hash: str
    endpoint_action_axis_hash: str
    endpoint_removed_index_hashes: tuple[str, ...]
    endpoint_f_intervals: tuple[tuple[float, float], ...]
    endpoint_q_intervals: tuple[tuple[float, float], ...]
    endpoint_statuses: tuple[str, ...]
    hard_fq_status_axis_hash: str
    every_endpoint_strictly_classified: bool = True
    nonmonotone_q_early_stop_used: bool = False
    source_derived_from_factor_interval_replay: bool = True
    _issuance_token: InitVar[object | None] = None
    schema: str = RELAXED_KKT_WORST_CASE_HARD_FQ_REPLAY_SCHEMA

    def __post_init__(self, _issuance_token: object | None) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("worst-case hard F/Q replay requires an exact branch")
        for name in (
            "problem_hash",
            "policy_hash",
            "source_evidence_hash",
            "rounded_ladder_hash",
            "endpoint_action_axis_hash",
            "hard_fq_status_axis_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        hashes = tuple(
            _nonempty(value, "endpoint_removed_index_hash")
            for value in self.endpoint_removed_index_hashes
        )
        f_intervals = tuple(
            tuple(_finite(item, "F interval endpoint") for item in value)
            for value in self.endpoint_f_intervals
        )
        q_intervals = tuple(
            tuple(_finite(item, "Q interval endpoint") for item in value)
            for value in self.endpoint_q_intervals
        )
        statuses = tuple(
            _nonempty(value, "endpoint status") for value in self.endpoint_statuses
        )
        if (
            not hashes
            or len(hashes) != len(f_intervals)
            or len(hashes) != len(q_intervals)
            or len(hashes) != len(statuses)
            or any(
                len(value) != 2 or value[0] > value[1]
                for value in (*f_intervals, *q_intervals)
            )
            or any(
                value not in {"safe_feasible", "robust_impossible"}
                for value in statuses
            )
        ):
            raise ValueError("worst-case hard F/Q axis is incomplete or ambiguous")
        flags = (
            self.every_endpoint_strictly_classified,
            self.nonmonotone_q_early_stop_used,
            self.source_derived_from_factor_interval_replay,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("worst-case hard F/Q flags must be exact booleans")
        if (
            flags != (True, False, True)
            or _issuance_token is not _DECISION_LEVEL_SOURCE_RECEIPT_ISSUANCE_TOKEN
        ):
            raise PermissionError("worst-case hard F/Q requires complete source replay")
        object.__setattr__(self, "endpoint_removed_index_hashes", hashes)
        object.__setattr__(self, "endpoint_f_intervals", f_intervals)
        object.__setattr__(self, "endpoint_q_intervals", q_intervals)
        object.__setattr__(self, "endpoint_statuses", statuses)
        if self.schema != RELAXED_KKT_WORST_CASE_HARD_FQ_REPLAY_SCHEMA:
            raise ValueError("unsupported worst-case hard F/Q replay schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {item.name: getattr(self, item.name) for item in fields(self)},
            prefix="partition-free-v1-relaxed-kkt-worst-case-hard-fq-",
        )


@dataclass(frozen=True, slots=True)
class V1RelaxedKKTAllAdmissibleShellFacesReceipt:
    """Exhaustive small-shell face/order enumeration result."""

    branch: V1RelaxedKKTBranch
    problem_hash: str
    policy_hash: str
    source_evidence_hash: str
    shell_route_index_hash: str
    rounded_ladder_hash: str
    endpoint_action_axis_hash: str
    admissible_face_count: int
    admissible_rounding_outcome_count: int
    outcome_ladder_axis_hash: str
    physical_index_only_tie_count: int
    semidefinite_or_nonunique_face_count: int
    all_admissible_faces_round_to_same_ladder: bool = True
    every_face_and_rounding_outcome_enumerated: bool = True
    _issuance_token: InitVar[object | None] = None
    schema: str = RELAXED_KKT_ALL_ADMISSIBLE_SHELL_FACES_SCHEMA

    def __post_init__(self, _issuance_token: object | None) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("shell-face receipt requires an exact branch")
        for name in (
            "problem_hash",
            "policy_hash",
            "source_evidence_hash",
            "shell_route_index_hash",
            "rounded_ladder_hash",
            "endpoint_action_axis_hash",
            "outcome_ladder_axis_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        for name in (
            "admissible_face_count",
            "admissible_rounding_outcome_count",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        for name in (
            "physical_index_only_tie_count",
            "semidefinite_or_nonunique_face_count",
        ):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name))
        flags = (
            self.all_admissible_faces_round_to_same_ladder,
            self.every_face_and_rounding_outcome_enumerated,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("shell-face enumeration flags must be exact booleans")
        if (
            flags != (True, True)
            or self.physical_index_only_tie_count != 0
            or _issuance_token is not _DECISION_LEVEL_SOURCE_RECEIPT_ISSUANCE_TOKEN
        ):
            raise PermissionError(
                "shell-face receipt lacks exhaustive same-ladder proof"
            )
        if self.schema != RELAXED_KKT_ALL_ADMISSIBLE_SHELL_FACES_SCHEMA:
            raise ValueError("unsupported shell-face receipt schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {item.name: getattr(self, item.name) for item in fields(self)},
            prefix="partition-free-v1-relaxed-kkt-all-shell-faces-",
        )


@dataclass(frozen=True, slots=True)
class V1RelaxedKKTDecisionLevelRobustInvarianceCertificate:
    """Decision-level alternative to full optimizer sensitivity.

    The theorem does not claim a unique perturbed relaxed solution.  It says a
    stronger decision fact: after a complete outside scan and exhaustive
    bounded-shell enumeration, every admissible face and real-key ordering
    yields the same registered hard ladder, and every endpoint keeps one
    strict worst-case F/Q status.  Consequently semidefinite or nonunique faces
    are admissible only when their entire rounding image is identical.
    """

    branch: V1RelaxedKKTBranch
    problem_hash: str
    policy_hash: str
    branch_result_hash: str
    branch_receipt_hash: str
    nominal_adaptive_rank_witness_hash: str
    route_axis_hash: str
    route_count: int
    finite_population_geometry_uncertainty_set_hash: str
    rounded_ladder_hash: str
    endpoint_action_axis_hash: str
    outside_shell_interval_scan: V1RelaxedKKTOutsideShellIntervalScanReceipt
    active_shell_restricted_operator: V1RelaxedKKTActiveShellRestrictedOperatorReceipt
    robust_dual_bound: V1RelaxedKKTRobustDualBoundReceipt
    worst_case_hard_fq_replay: V1RelaxedKKTWorstCaseHardFQReplayReceipt
    all_admissible_shell_faces: V1RelaxedKKTAllAdmissibleShellFacesReceipt
    closure_mode: V1RelaxedKKTClosureMode = (
        V1RelaxedKKTClosureMode.DECISION_LEVEL_SHELL_INVARIANCE
    )
    decision_level_shell_invariance_theorem: bool = True
    source_derived_from_numerical_receipts: bool = True
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = RELAXED_KKT_DECISION_LEVEL_ROBUST_INVARIANCE_SCOPE
    _issuance_token: InitVar[object | None] = None
    schema: str = RELAXED_KKT_DECISION_LEVEL_ROBUST_INVARIANCE_SCHEMA

    def __post_init__(self, _issuance_token: object | None) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("decision-level theorem requires an exact branch")
        for name in (
            "problem_hash",
            "policy_hash",
            "branch_result_hash",
            "branch_receipt_hash",
            "nominal_adaptive_rank_witness_hash",
            "route_axis_hash",
            "finite_population_geometry_uncertainty_set_hash",
            "rounded_ladder_hash",
            "endpoint_action_axis_hash",
            "authority_scope",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self, "route_count", _positive_int(self.route_count, "route_count")
        )
        receipt_types = (
            (
                self.outside_shell_interval_scan,
                V1RelaxedKKTOutsideShellIntervalScanReceipt,
            ),
            (
                self.active_shell_restricted_operator,
                V1RelaxedKKTActiveShellRestrictedOperatorReceipt,
            ),
            (self.robust_dual_bound, V1RelaxedKKTRobustDualBoundReceipt),
            (self.worst_case_hard_fq_replay, V1RelaxedKKTWorstCaseHardFQReplayReceipt),
            (
                self.all_admissible_shell_faces,
                V1RelaxedKKTAllAdmissibleShellFacesReceipt,
            ),
        )
        if any(type(value) is not expected for value, expected in receipt_types):
            raise TypeError(
                "decision-level theorem requires exact source receipt types"
            )
        common = (
            self.outside_shell_interval_scan,
            self.active_shell_restricted_operator,
            self.robust_dual_bound,
            self.worst_case_hard_fq_replay,
            self.all_admissible_shell_faces,
        )
        if any(
            value.branch is not self.branch
            or value.problem_hash != self.problem_hash
            or value.policy_hash != self.policy_hash
            for value in common
        ):
            raise ValueError("decision-level source receipts left the solved branch")
        if (
            self.outside_shell_interval_scan.route_axis_hash != self.route_axis_hash
            or self.outside_shell_interval_scan.route_count != self.route_count
            or self.active_shell_restricted_operator.route_axis_hash
            != self.route_axis_hash
            or self.robust_dual_bound.branch_result_hash != self.branch_result_hash
            or self.robust_dual_bound.branch_receipt_hash != self.branch_receipt_hash
            or self.worst_case_hard_fq_replay.rounded_ladder_hash
            != self.rounded_ladder_hash
            or self.worst_case_hard_fq_replay.endpoint_action_axis_hash
            != self.endpoint_action_axis_hash
            or self.all_admissible_shell_faces.rounded_ladder_hash
            != self.rounded_ladder_hash
            or self.all_admissible_shell_faces.endpoint_action_axis_hash
            != self.endpoint_action_axis_hash
            or self.robust_dual_bound.restricted_operator_receipt_hash
            != self.active_shell_restricted_operator.receipt_hash
        ):
            raise ValueError("decision-level theorem coordinate binding is incomplete")
        source_hashes = {value.source_evidence_hash for value in common}
        if len(source_hashes) != 1:
            raise ValueError("decision-level source receipts use different evidence")
        flags = (
            self.decision_level_shell_invariance_theorem,
            self.source_derived_from_numerical_receipts,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("decision-level theorem flags must be exact booleans")
        if (
            self.closure_mode
            is not V1RelaxedKKTClosureMode.DECISION_LEVEL_SHELL_INVARIANCE
            or flags != (True, True, False, False, False, False, False, False)
            or _issuance_token is not _DECISION_LEVEL_ROBUST_INVARIANCE_ISSUANCE_TOKEN
        ):
            raise PermissionError(
                "positive decision-level invariance requires the numerical source issuer"
            )
        if self.authority_scope != RELAXED_KKT_DECISION_LEVEL_ROBUST_INVARIANCE_SCOPE:
            raise ValueError("unsupported decision-level theorem authority scope")
        if self.schema != RELAXED_KKT_DECISION_LEVEL_ROBUST_INVARIANCE_SCHEMA:
            raise ValueError("unsupported decision-level theorem schema")

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "branch": self.branch,
            "problem_hash": self.problem_hash,
            "policy_hash": self.policy_hash,
            "branch_result_hash": self.branch_result_hash,
            "branch_receipt_hash": self.branch_receipt_hash,
            "nominal_adaptive_rank_witness_hash": (
                self.nominal_adaptive_rank_witness_hash
            ),
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "finite_population_geometry_uncertainty_set_hash": (
                self.finite_population_geometry_uncertainty_set_hash
            ),
            "rounded_ladder_hash": self.rounded_ladder_hash,
            "endpoint_action_axis_hash": self.endpoint_action_axis_hash,
            "outside_shell_interval_scan_receipt_hash": (
                self.outside_shell_interval_scan.receipt_hash
            ),
            "active_shell_restricted_operator_receipt_hash": (
                self.active_shell_restricted_operator.receipt_hash
            ),
            "robust_dual_bound_receipt_hash": self.robust_dual_bound.receipt_hash,
            "worst_case_hard_fq_replay_receipt_hash": (
                self.worst_case_hard_fq_replay.receipt_hash
            ),
            "all_admissible_shell_faces_receipt_hash": (
                self.all_admissible_shell_faces.receipt_hash
            ),
            "closure_mode": self.closure_mode,
            "decision_level_shell_invariance_theorem": (
                self.decision_level_shell_invariance_theorem
            ),
            "source_derived_from_numerical_receipts": (
                self.source_derived_from_numerical_receipts
            ),
            "authority_scope": self.authority_scope,
        }

    @property
    def certificate_hash(self) -> str:
        return content_hash(
            self._payload(),
            prefix="partition-free-v1-relaxed-kkt-decision-level-invariance-",
        )

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self.certificate_hash
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("decision-level theorem hash does not match")
        return observed


def issue_v1_relaxed_kkt_decision_level_robust_invariance_from_source_receipts(
    *,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    branch_result: V1RelaxedKKTBranchResult,
    branch_identity: V1RelaxedKKTDeviceShellBranchIdentityReceipt,
    geometry_uncertainty: V1RelaxedKKTGeometryUncertaintySet,
    nominal_witness: V1RelaxedKKTAdaptiveRankBranchWitness,
    outside_shell_interval_scan: V1RelaxedKKTOutsideShellIntervalScanReceipt,
    active_shell_restricted_operator: V1RelaxedKKTActiveShellRestrictedOperatorReceipt,
    robust_dual_bound: V1RelaxedKKTRobustDualBoundReceipt,
    worst_case_hard_fq_replay: V1RelaxedKKTWorstCaseHardFQReplayReceipt,
    all_admissible_shell_faces: V1RelaxedKKTAllAdmissibleShellFacesReceipt,
    issuer_backend: Literal["high_precision_reference", "torch_cuda"],
) -> V1RelaxedKKTDecisionLevelRobustInvarianceCertificate:
    """Promote compact source receipts into the decision-shell theorem.

    This is the production-facing theorem ABI.  The function deliberately does
    not compute interval scans, shell operators, primal/dual enclosures, hard
    F/Q replay, or shell-face enumeration.  Those objects must already have
    been issued by this module's private source-receipt token, either by the
    bounded high-precision reference issuer or by the future tiled CUDA source
    issuer.  Consequently the theorem authority is independent of any dense
    routewise arrays retained by the reference path.
    """

    if type(problem) is not V1RelaxedKKTProblem:
        raise TypeError("decision-shell source receipts require an exact problem")
    if type(policy) is not V1RelaxedKKTPolicy:
        raise TypeError("decision-shell source receipts require an exact policy")
    if type(branch_result) is not V1RelaxedKKTBranchResult:
        raise TypeError("decision-shell source receipts require an exact branch result")
    if type(branch_identity) is not V1RelaxedKKTDeviceShellBranchIdentityReceipt:
        raise TypeError(
            "decision-shell source receipts require an exact shell identity"
        )
    if type(geometry_uncertainty) is not V1RelaxedKKTGeometryUncertaintySet:
        raise TypeError("decision-shell source receipts require exact uncertainty")
    if type(nominal_witness) is not V1RelaxedKKTAdaptiveRankBranchWitness:
        raise TypeError("decision-shell source receipts require an exact witness")
    if issuer_backend not in {"high_precision_reference", "torch_cuda"}:
        raise ValueError("unsupported decision-shell source issuer backend")
    receipt_types = (
        (outside_shell_interval_scan, V1RelaxedKKTOutsideShellIntervalScanReceipt),
        (
            active_shell_restricted_operator,
            V1RelaxedKKTActiveShellRestrictedOperatorReceipt,
        ),
        (robust_dual_bound, V1RelaxedKKTRobustDualBoundReceipt),
        (worst_case_hard_fq_replay, V1RelaxedKKTWorstCaseHardFQReplayReceipt),
        (all_admissible_shell_faces, V1RelaxedKKTAllAdmissibleShellFacesReceipt),
    )
    if any(type(value) is not expected for value, expected in receipt_types):
        raise TypeError("decision-shell source receipt family has an invalid type")
    if outside_shell_interval_scan.scan_backend != issuer_backend:
        _refuse(
            V1RelaxedKKTRefusalCode.DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE,
            "decision-shell source issuer backend differs from the outside-shell scan",
        )
    if (
        branch_result.branch is not branch_identity.branch
        or branch_result.branch is not nominal_witness.branch
        or branch_result.receipt.problem_hash != problem.problem_hash
        or branch_result.receipt.policy_hash != policy.policy_hash
        or branch_result.receipt.receipt_hash != nominal_witness.branch_receipt_hash
        or branch_result.result_hash != nominal_witness.branch_result_hash
        or branch_identity.frozen_ladder.route_axis_hash != problem.route_axis_hash
        or branch_identity.frozen_ladder.route_count != problem.route_count
        or branch_identity.source_rounding_hash != nominal_witness.rounded_ladder_hash
        or branch_identity.endpoint_action_axis_hash
        != nominal_witness.endpoint_action_axis_hash
        or geometry_uncertainty.problem_hash != problem.problem_hash
        or geometry_uncertainty.route_axis_hash != problem.route_axis_hash
        or geometry_uncertainty.source_geometry_hash != problem.source_geometry_hash
        or not geometry_uncertainty.source_derived_from_registered_geometry_receipts
        or nominal_witness.problem_hash != problem.problem_hash
        or nominal_witness.policy_hash != policy.policy_hash
        or nominal_witness.route_axis_hash != problem.route_axis_hash
        or nominal_witness.route_count != problem.route_count
        or nominal_witness.finite_population_geometry_uncertainty_set_hash
        != geometry_uncertainty.uncertainty_hash
        or not nominal_witness.source_derived_from_typed_interval_receipts
    ):
        _refuse(
            V1RelaxedKKTRefusalCode.DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE,
            "nominal witness, shell identity, branch result, and source "
            "uncertainty do not bind one decision-shell chart",
        )
    return V1RelaxedKKTDecisionLevelRobustInvarianceCertificate(
        branch=branch_result.branch,
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        branch_result_hash=branch_result.result_hash,
        branch_receipt_hash=branch_result.receipt.receipt_hash,
        nominal_adaptive_rank_witness_hash=nominal_witness.witness_hash,
        route_axis_hash=problem.route_axis_hash,
        route_count=problem.route_count,
        finite_population_geometry_uncertainty_set_hash=(
            geometry_uncertainty.uncertainty_hash
        ),
        rounded_ladder_hash=branch_identity.source_rounding_hash,
        endpoint_action_axis_hash=branch_identity.endpoint_action_axis_hash,
        outside_shell_interval_scan=outside_shell_interval_scan,
        active_shell_restricted_operator=active_shell_restricted_operator,
        robust_dual_bound=robust_dual_bound,
        worst_case_hard_fq_replay=worst_case_hard_fq_replay,
        all_admissible_shell_faces=all_admissible_shell_faces,
        _issuance_token=_DECISION_LEVEL_ROBUST_INVARIANCE_ISSUANCE_TOKEN,
    )


def issue_v1_relaxed_kkt_geometry_uncertainty_from_registered_receipts(
    *,
    problem: V1RelaxedKKTProblem,
    corrected_work_confidence_box_view: V1ConfidenceBoxExternalView,
    corrected_work_identity_receipt: Any,
    literal_j_operator_census_receipt: (
        V1LiteralJOperatorCensusReceipt
        | V1LiteralJFixedDirectionProjectionReceipt
        | V1LiteralJDFixedResidualPopulationProjectionReceipt
    ),
    fd_q_residual_deterministic_linf_adjoint_radius: float = 0.0,
    solver_reduced_advantage_numerical_linf_radius: float = 0.0,
    relaxed_gate_numerical_linf_radius: float = 0.0,
    corrected_work_ratio_deterministic_linf_radius: float = 0.0,
) -> V1RelaxedKKTGeometryUncertaintySet:
    """Issue the source-owned uncertainty object required by KKT witnesses.

    The issuance token stays private to this module.  The caller cannot make a
    source-derived uncertainty set by supplying hashes, scalars, or a boolean:
    corrected-work sampling uncertainty is read from an exact open
    confidence-box view plus the exact identity receipt that transports the
    ordered confidence axis onto the physical KKT axis.  The literal-J
    response-operator term is read from either an exact finite-population
    ``M_J z`` census receipt, a fixed-direction finite-population projection
    receipt, or the source-custody-bound exact ``D_fixed`` plus stratified
    residual population projection receipt.  Projection receipts certify only
    fixed ``z`` operator uncertainty; they do not prove KKT optimizer
    sensitivity.  Deterministic FD-shell and numerical radii are still
    explicit deterministic uncertainty sets, but they do not by themselves
    create finite-population authority.
    """

    from .modern_v1_kkt_corrected_work_identity import (
        V1KKTCorrectedWorkIdentityReceipt,
    )
    from .modern_v1_literal_j_d_fixed_residual_population_projection import (
        V1LiteralJDFixedResidualPopulationProjectionReceipt,
    )

    if type(problem) is not V1RelaxedKKTProblem:
        raise TypeError("registered geometry uncertainty requires an exact problem")
    if type(corrected_work_confidence_box_view) is not V1ConfidenceBoxExternalView:
        raise TypeError(
            "corrected_work_confidence_box_view must be an exact open "
            "confidence-box view"
        )
    if type(corrected_work_identity_receipt) is not V1KKTCorrectedWorkIdentityReceipt:
        raise TypeError(
            "corrected_work_identity_receipt must be an exact bridge receipt"
        )
    literal_j_receipt_types = (
        V1LiteralJOperatorCensusReceipt,
        V1LiteralJFixedDirectionProjectionReceipt,
        V1LiteralJDFixedResidualPopulationProjectionReceipt,
    )
    if type(literal_j_operator_census_receipt) not in literal_j_receipt_types:
        raise TypeError(
            "literal_j_operator_census_receipt must be an exact literal-J source "
            "receipt"
        )
    confidence = corrected_work_confidence_box_view
    identity = corrected_work_identity_receipt
    literal_j = literal_j_operator_census_receipt
    literal_j.assert_integrity()
    exact_census = type(literal_j) is V1LiteralJOperatorCensusReceipt
    fixed_direction_projection = (
        type(literal_j) is V1LiteralJFixedDirectionProjectionReceipt
    )
    population_projection = (
        type(literal_j) is V1LiteralJDFixedResidualPopulationProjectionReceipt
    )
    projected_fixed_z = fixed_direction_projection or population_projection
    literal_j_direction_axis_hash = (
        literal_j.private_primal_direction_axis_hash
        if population_projection
        else literal_j.direction_axis_hash
    )
    literal_j_direction_count = (
        len(literal_j.direction_ids)
        if population_projection
        else literal_j.direction_count
    )
    literal_j_output_axis_hash = (
        literal_j.combined_numerical_binding_hash
        if population_projection
        else literal_j.output_axis_hash
    )
    literal_j_sampling_radius = (
        literal_j.mjz_sampling_linf_radius_upper_bound
        if population_projection
        else literal_j.sampling_radius
    )
    literal_j_alpha_spent = literal_j.alpha_spent
    if (
        confidence.descriptor.descriptor_hash
        != identity.confidence_box_external_descriptor_hash
        or confidence.descriptor.commit_hash
        != identity.confidence_box_external_commit_hash
        or confidence.descriptor.receipt_hash != identity.confidence_box_receipt_hash
        or confidence.descriptor.route_axis_hash != identity.ordered_route_axis_hash
        or confidence.descriptor.route_count != identity.route_count
        or identity.declared_kkt_problem_hash != problem.problem_hash
        or identity.kkt_problem_id != problem.problem_id
        or identity.kkt_source_geometry_hash != problem.source_geometry_hash
        or identity.physical_route_axis_hash != problem.route_axis_hash
        or identity.route_count != problem.route_count
        or literal_j.route_axis_hash != problem.route_axis_hash
        or literal_j.route_count != problem.route_count
        or literal_j.chart_identity_hash != problem.source_geometry_hash
        or (
            population_projection
            and (
                not literal_j.full_population_literal_j_projection_authority
                or not literal_j.kkt_geometry_uncertainty_input_authority
                or literal_j.fd_rank_authority
                or literal_j.kkt_face_or_solution_authority
            )
        )
        or (
            not population_projection
            and (
                literal_j.literal_j_rows_retained
                or literal_j.face_authority
                or literal_j.shell_authority
                or literal_j.fqr_authority
                or literal_j.production_authority
            )
        )
        or literal_j.c_stream_authority
        or literal_j.calibration_authority
        or literal_j.oracle_authority
        or literal_j.commit_authority
        or literal_j.pareto_authority
    ):
        _refuse(
            V1RelaxedKKTRefusalCode.ADAPTIVE_RANK_WITNESS_SOURCE_MISMATCH,
            "corrected-A confidence box and literal-J census do not bind this "
            "KKT source chart",
        )
    if exact_census:
        if (
            not literal_j.exact_finite_population_census
            or literal_j.observed_counts != literal_j.population_counts
            or not literal_j.directions_frozen_before_census
        ):
            _refuse(
                V1RelaxedKKTRefusalCode.ADAPTIVE_RANK_WITNESS_SOURCE_MISMATCH,
                "literal-J exact census does not bind the complete frozen "
                "direction population",
            )
    if projected_fixed_z:
        if fixed_direction_projection:
            if (
                literal_j.exact_finite_population_census
                or not literal_j.directions_frozen_before_projection
                or not literal_j.independent_or_predictable_witness_stream
                or not literal_j.source_derived_for_fixed_primal_projection
                or not literal_j.fixed_z_only_no_kkt_solution_sensitivity
                or any(
                    observed <= 0 or observed > population
                    for observed, population in zip(
                        literal_j.observed_counts,
                        literal_j.population_counts,
                        strict=True,
                    )
                )
            ):
                _refuse(
                    V1RelaxedKKTRefusalCode.ADAPTIVE_RANK_WITNESS_SOURCE_MISMATCH,
                    "literal-J projection receipt is not a causal fixed-z "
                    "finite-population witness",
                )
        if population_projection:
            if (
                not literal_j.full_population_literal_j_projection_authority
                or not literal_j.kkt_geometry_uncertainty_input_authority
                or any(
                    observed > population
                    for observed, population in zip(
                        literal_j.residual_observed_counts,
                        literal_j.residual_population_counts,
                        strict=True,
                    )
                )
                or not literal_j.array_free_receipt
                or not literal_j.live_numeric_handle_required
            ):
                _refuse(
                    V1RelaxedKKTRefusalCode.ADAPTIVE_RANK_WITNESS_SOURCE_MISMATCH,
                    "literal-J D_fixed plus residual projection receipt is not "
                    "a causal full-population fixed-z witness",
                )
    point = np.asarray(confidence.arrays["class_balanced_absolute_point"])
    lower = np.asarray(confidence.arrays["cumulative_class_balanced_absolute_lower"])
    upper = np.asarray(confidence.arrays["cumulative_class_balanced_absolute_upper"])
    if (
        point.shape != (problem.route_count,)
        or lower.shape != point.shape
        or upper.shape != point.shape
    ):
        raise ValueError("corrected-work confidence arrays do not match route count")
    with np.errstate(invalid="ignore"):
        work_radius = max(
            float(np.max(np.maximum(point - lower, upper - point))),
            0.0,
        )
    if not math.isfinite(work_radius):
        raise ValueError("corrected-work confidence radius is not finite")
    eligible_resources = np.asarray(problem.resource)[np.asarray(problem.eligible_mask)]
    if eligible_resources.size == 0:
        raise ValueError("problem has no eligible resource axis")
    min_resource = float(np.min(eligible_resources))
    if not math.isfinite(min_resource) or min_resource <= 0.0:
        raise ValueError("eligible resource axis must be positive")
    ratio_sampling_radius = work_radius / min_resource
    current_metadata = confidence.metadata.get("current")
    if type(current_metadata) is not dict:
        raise ValueError("confidence-box metadata is missing current payload")
    payload_without_arrays = current_metadata.get("payload_without_arrays")
    if type(payload_without_arrays) is not dict:
        raise ValueError("confidence-box metadata is missing array-free payload")
    literal_j_alpha_name = (
        "literal_j_census_alpha_spent"
        if exact_census
        else "literal_j_projection_alpha_spent"
    )
    alpha_spent = _nonnegative(
        payload_without_arrays.get("cumulative_alpha_spent"),
        "confidence_box_cumulative_alpha_spent",
    ) + _nonnegative(literal_j_alpha_spent, literal_j_alpha_name)
    uncertainty_id = content_hash(
        {
            "problem_hash": problem.problem_hash,
            "route_axis_hash": problem.route_axis_hash,
            "source_geometry_hash": problem.source_geometry_hash,
            "corrected_work_confidence_box_receipt_hash": (
                confidence.descriptor.receipt_hash
            ),
            "corrected_work_confidence_box_commit_hash": (
                confidence.descriptor.commit_hash
            ),
            "corrected_work_identity_receipt_hash": identity.receipt_hash,
            "corrected_work_identity_ordered_route_axis_hash": (
                identity.ordered_route_axis_hash
            ),
            "corrected_work_identity_physical_route_axis_hash": (
                identity.physical_route_axis_hash
            ),
            "corrected_work_A_sampling_linf_radius": work_radius,
            "corrected_work_A_deterministic_linf_radius": 0.0,
            "literal_j_operator_source_receipt_hash": literal_j.receipt_hash,
            "literal_j_operator_source_schema": literal_j.schema,
            "literal_j_operator_source_exact_census": (
                exact_census
                or (population_projection and literal_j.exact_full_population_census)
            ),
            "literal_j_operator_source_fixed_z_projection": projected_fixed_z,
            "literal_j_operator_source_D_fixed_residual_population_projection": (
                population_projection
            ),
            "literal_j_operator_source_output_axis_hash": literal_j_output_axis_hash,
            "literal_j_operator_source_direction_axis_hash": (
                literal_j_direction_axis_hash
            ),
            "literal_j_fixed_primal_direction_axis_hash": (
                literal_j_direction_axis_hash
            ),
            "literal_j_fixed_primal_direction_count": literal_j_direction_count,
            "literal_j_response_operator_MJ_sampling_linf_adjoint_radius": (
                literal_j_sampling_radius
            ),
            "literal_j_response_operator_MJ_deterministic_linf_adjoint_radius": 0.0,
            "fd_q_residual_deterministic_linf_adjoint_radius": (
                fd_q_residual_deterministic_linf_adjoint_radius
            ),
            "solver_reduced_advantage_numerical_linf_radius": (
                solver_reduced_advantage_numerical_linf_radius
            ),
            "relaxed_gate_numerical_linf_radius": (relaxed_gate_numerical_linf_radius),
            "corrected_work_ratio_sampling_linf_radius": ratio_sampling_radius,
            "corrected_work_ratio_deterministic_linf_radius": (
                corrected_work_ratio_deterministic_linf_radius
            ),
            "alpha_spent": alpha_spent,
            "issuer": (
                "registered_corrected_A_confidence_box_and_literal_J_MJz_census_v1"
            ),
        },
        prefix="partition-free-v1-relaxed-kkt-geometry-uncertainty-source-",
    )
    return V1RelaxedKKTGeometryUncertaintySet(
        uncertainty_id=uncertainty_id,
        problem_hash=problem.problem_hash,
        route_axis_hash=problem.route_axis_hash,
        source_geometry_hash=problem.source_geometry_hash,
        corrected_work_A_sampling_linf_radius=work_radius,
        corrected_work_A_deterministic_linf_radius=0.0,
        literal_j_response_operator_MJ_sampling_linf_adjoint_radius=(
            literal_j_sampling_radius
        ),
        literal_j_response_operator_MJ_deterministic_linf_adjoint_radius=0.0,
        fd_q_residual_deterministic_linf_adjoint_radius=(
            fd_q_residual_deterministic_linf_adjoint_radius
        ),
        solver_reduced_advantage_numerical_linf_radius=(
            solver_reduced_advantage_numerical_linf_radius
        ),
        relaxed_gate_numerical_linf_radius=relaxed_gate_numerical_linf_radius,
        corrected_work_ratio_sampling_linf_radius=ratio_sampling_radius,
        corrected_work_ratio_deterministic_linf_radius=(
            corrected_work_ratio_deterministic_linf_radius
        ),
        alpha_spent=alpha_spent,
        literal_j_fixed_primal_direction_axis_hash=literal_j_direction_axis_hash,
        literal_j_fixed_primal_direction_count=literal_j_direction_count,
        source_derived_from_registered_geometry_receipts=True,
        _geometry_uncertainty_issuance_token=(_GEOMETRY_UNCERTAINTY_ISSUANCE_TOKEN),
    )


@dataclass(frozen=True, slots=True)
class V1RelaxedKKTAdaptiveRankBranchWitness:
    """Routewise KKT witness needed before adaptive FD rank may open C.

    This is a selector-side sidecar: it binds routewise reduced-advantage
    enclosures, relaxed-face margins, and deterministic four-key endpoint
    shell margins.  It is not an F/Q/R certificate and it cannot materialize or
    commit a mask by itself.
    """

    branch: V1RelaxedKKTBranch
    problem_hash: str
    policy_hash: str
    branch_result_hash: str
    branch_receipt_hash: str
    rounded_ladder_hash: str
    route_axis_hash: str
    route_count: int
    relaxed_face_witness_hash: str
    rounded_endpoint_set_witness_hash: str
    reduced_advantage_interval_axis_hash: str
    four_key_boundary_interval_axis_hash: str
    block_capacity_boundary_hash: str
    endpoint_action_axis_hash: str
    winning_face_margin_lower_bound: float
    winning_face_residual_perturbation_upper_bound: float
    selected_vs_runner_up_margin_lower_bound: float
    selected_vs_runner_up_residual_perturbation_upper_bound: float
    endpoint_boundary_margin_lower_bound: float
    endpoint_boundary_residual_perturbation_upper_bound: float
    finite_population_geometry_uncertainty_set_hash: str
    finite_population_face_perturbation_upper_bound: float
    finite_population_key_perturbation_upper_bound: float
    relaxed_problem_perturbation_upper_bound: float = 0.0
    relaxed_solution_sensitivity_certificate_hash: str | None = None
    decision_level_robust_invariance_certificate_hash: str | None = None
    literal_j_fixed_primal_direction_axis_hash: str = (
        "unbound_literal_j_fixed_primal_direction_axis"
    )
    complete_optimal_face_enclosed: bool = True
    routewise_reduced_advantage_intervals_complete: bool = True
    finite_population_geometry_intervals_complete: bool = True
    deterministic_four_key_shells_complete: bool = True
    one_live_survivor_block_capacity_verified: bool = True
    source_derived_from_typed_interval_receipts: bool = False
    ordinary_hard_action_materialized: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    production_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = RELAXED_KKT_ADAPTIVE_RANK_WITNESS_SCOPE
    _adaptive_rank_witness_issuance_token: InitVar[object | None] = None
    schema: str = RELAXED_KKT_ADAPTIVE_RANK_BRANCH_WITNESS_SCHEMA

    def __post_init__(
        self,
        _adaptive_rank_witness_issuance_token: object | None,
    ) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("adaptive-rank KKT witness requires an exact branch")
        for name in (
            "problem_hash",
            "policy_hash",
            "branch_result_hash",
            "branch_receipt_hash",
            "rounded_ladder_hash",
            "route_axis_hash",
            "relaxed_face_witness_hash",
            "rounded_endpoint_set_witness_hash",
            "reduced_advantage_interval_axis_hash",
            "four_key_boundary_interval_axis_hash",
            "block_capacity_boundary_hash",
            "endpoint_action_axis_hash",
            "literal_j_fixed_primal_direction_axis_hash",
            "finite_population_geometry_uncertainty_set_hash",
            "authority_scope",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, "route_count"),
        )
        for name in (
            "winning_face_margin_lower_bound",
            "winning_face_residual_perturbation_upper_bound",
            "selected_vs_runner_up_margin_lower_bound",
            "selected_vs_runner_up_residual_perturbation_upper_bound",
            "endpoint_boundary_margin_lower_bound",
            "endpoint_boundary_residual_perturbation_upper_bound",
            "finite_population_face_perturbation_upper_bound",
            "finite_population_key_perturbation_upper_bound",
            "relaxed_problem_perturbation_upper_bound",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        if self.relaxed_solution_sensitivity_certificate_hash is not None:
            object.__setattr__(
                self,
                "relaxed_solution_sensitivity_certificate_hash",
                _nonempty(
                    self.relaxed_solution_sensitivity_certificate_hash,
                    "relaxed_solution_sensitivity_certificate_hash",
                ),
            )
        if self.decision_level_robust_invariance_certificate_hash is not None:
            object.__setattr__(
                self,
                "decision_level_robust_invariance_certificate_hash",
                _nonempty(
                    self.decision_level_robust_invariance_certificate_hash,
                    "decision_level_robust_invariance_certificate_hash",
                ),
            )
        if (
            self.relaxed_solution_sensitivity_certificate_hash is not None
            and self.decision_level_robust_invariance_certificate_hash is not None
        ):
            raise PermissionError(
                "one adaptive-rank witness cannot claim two nonzero closure theorems"
            )
        required_flags = (
            "complete_optimal_face_enclosed",
            "routewise_reduced_advantage_intervals_complete",
            "finite_population_geometry_intervals_complete",
            "deterministic_four_key_shells_complete",
            "one_live_survivor_block_capacity_verified",
            "source_derived_from_typed_interval_receipts",
        )
        authority_flags = (
            "ordinary_hard_action_materialized",
            "c_stream_authority",
            "calibration_authority",
            "oracle_authority",
            "mask_commit_authority",
            "production_authority",
            "pareto_authority",
        )
        if any(
            type(getattr(self, name)) is not bool
            for name in (*required_flags, *authority_flags)
        ):
            raise TypeError("adaptive-rank KKT witness flags must be exact booleans")
        if (
            self.source_derived_from_typed_interval_receipts
            and _adaptive_rank_witness_issuance_token
            is not _ADAPTIVE_RANK_BRANCH_WITNESS_ISSUANCE_TOKEN
        ):
            raise PermissionError(
                "source-derived adaptive-rank KKT witness requires the "
                "registered routewise interval builder"
            )
        if (
            self.source_derived_from_typed_interval_receipts
            and self.literal_j_fixed_primal_direction_axis_hash
            == "unbound_literal_j_fixed_primal_direction_axis"
        ):
            raise PermissionError(
                "source-derived adaptive-rank KKT witness must bind the "
                "literal-J fixed-primal direction axis"
            )
        if any(getattr(self, name) for name in authority_flags):
            raise PermissionError(
                "adaptive-rank KKT witness crossed its authority boundary"
            )
        if self.authority_scope != RELAXED_KKT_ADAPTIVE_RANK_WITNESS_SCOPE:
            raise ValueError("unsupported adaptive-rank KKT witness authority scope")
        if self.schema != RELAXED_KKT_ADAPTIVE_RANK_BRANCH_WITNESS_SCHEMA:
            raise ValueError("unsupported adaptive-rank KKT witness schema")

    @property
    def relaxed_solution_sensitivity_closed(self) -> bool:
        """Whether routewise margins may represent the same relaxed optimizer.

        A positive sampling/FD/operator perturbation can move the KKT duals and
        relaxed primal, so current-dual reduced-advantage margins are only a
        replay witness unless a stronger robust sensitivity certificate is
        attached.  Exact, perturbation-free geometry still admits the nominal
        optimizer certificate.  A decision-level shell-invariance theorem is
        deliberately narrower and is exposed through
        :attr:`decision_level_shell_invariance_closed` instead.
        """

        return (
            self.relaxed_problem_perturbation_upper_bound == 0.0
            or self.relaxed_solution_sensitivity_certificate_hash is not None
        )

    @property
    def closure_mode(self) -> V1RelaxedKKTClosureMode | None:
        if self.relaxed_problem_perturbation_upper_bound == 0.0:
            return V1RelaxedKKTClosureMode.EXACT_ZERO
        if self.relaxed_solution_sensitivity_certificate_hash is not None:
            return V1RelaxedKKTClosureMode.FULL_SOLUTION_SENSITIVITY
        if self.decision_level_robust_invariance_certificate_hash is not None:
            return V1RelaxedKKTClosureMode.DECISION_LEVEL_SHELL_INVARIANCE
        return None

    @property
    def decision_level_shell_invariance_closed(self) -> bool:
        """Whether the narrower shell theorem closes the hard decision image."""

        return (
            self.closure_mode is V1RelaxedKKTClosureMode.DECISION_LEVEL_SHELL_INVARIANCE
            and self.decision_level_robust_invariance_certificate_hash is not None
        )

    @property
    def decision_invariance_closed(self) -> bool:
        """Whether the next hard decision is closed under registered uncertainty."""

        return (
            self.relaxed_solution_sensitivity_closed
            or self.decision_level_shell_invariance_closed
        )

    @property
    def relaxed_face_stable(self) -> bool:
        return (
            self.source_derived_from_typed_interval_receipts
            and self.decision_invariance_closed
            and self.complete_optimal_face_enclosed
            and self.routewise_reduced_advantage_intervals_complete
            and self.finite_population_geometry_intervals_complete
            and self.winning_face_margin_lower_bound
            > (
                self.winning_face_residual_perturbation_upper_bound
                + self.finite_population_face_perturbation_upper_bound
            )
        )

    @property
    def rounded_endpoint_set_stable(self) -> bool:
        return (
            self.source_derived_from_typed_interval_receipts
            and self.decision_invariance_closed
            and self.deterministic_four_key_shells_complete
            and self.finite_population_geometry_intervals_complete
            and self.one_live_survivor_block_capacity_verified
            and self.selected_vs_runner_up_margin_lower_bound
            > (
                self.selected_vs_runner_up_residual_perturbation_upper_bound
                + self.finite_population_key_perturbation_upper_bound
            )
            and self.endpoint_boundary_margin_lower_bound
            > (
                self.endpoint_boundary_residual_perturbation_upper_bound
                + self.finite_population_key_perturbation_upper_bound
            )
        )

    @property
    def rank_local_stable(self) -> bool:
        return self.relaxed_face_stable and self.rounded_endpoint_set_stable

    @property
    def witness_hash(self) -> str:
        payload: dict[str, object] = {
            "schema": self.schema,
            "branch": self.branch,
            "problem_hash": self.problem_hash,
            "policy_hash": self.policy_hash,
            "branch_result_hash": self.branch_result_hash,
            "branch_receipt_hash": self.branch_receipt_hash,
            "rounded_ladder_hash": self.rounded_ladder_hash,
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "relaxed_face_witness_hash": self.relaxed_face_witness_hash,
            "rounded_endpoint_set_witness_hash": (
                self.rounded_endpoint_set_witness_hash
            ),
            "reduced_advantage_interval_axis_hash": (
                self.reduced_advantage_interval_axis_hash
            ),
            "four_key_boundary_interval_axis_hash": (
                self.four_key_boundary_interval_axis_hash
            ),
            "block_capacity_boundary_hash": self.block_capacity_boundary_hash,
            "endpoint_action_axis_hash": self.endpoint_action_axis_hash,
            "literal_j_fixed_primal_direction_axis_hash": (
                self.literal_j_fixed_primal_direction_axis_hash
            ),
            "finite_population_geometry_uncertainty_set_hash": (
                self.finite_population_geometry_uncertainty_set_hash
            ),
            "winning_face_margin_lower_bound": (self.winning_face_margin_lower_bound),
            "winning_face_residual_perturbation_upper_bound": (
                self.winning_face_residual_perturbation_upper_bound
            ),
            "selected_vs_runner_up_margin_lower_bound": (
                self.selected_vs_runner_up_margin_lower_bound
            ),
            "selected_vs_runner_up_residual_perturbation_upper_bound": (
                self.selected_vs_runner_up_residual_perturbation_upper_bound
            ),
            "endpoint_boundary_margin_lower_bound": (
                self.endpoint_boundary_margin_lower_bound
            ),
            "endpoint_boundary_residual_perturbation_upper_bound": (
                self.endpoint_boundary_residual_perturbation_upper_bound
            ),
            "finite_population_face_perturbation_upper_bound": (
                self.finite_population_face_perturbation_upper_bound
            ),
            "finite_population_key_perturbation_upper_bound": (
                self.finite_population_key_perturbation_upper_bound
            ),
            "relaxed_problem_perturbation_upper_bound": (
                self.relaxed_problem_perturbation_upper_bound
            ),
            "relaxed_solution_sensitivity_certificate_hash": (
                self.relaxed_solution_sensitivity_certificate_hash
            ),
            "complete_optimal_face_enclosed": (self.complete_optimal_face_enclosed),
            "routewise_reduced_advantage_intervals_complete": (
                self.routewise_reduced_advantage_intervals_complete
            ),
            "finite_population_geometry_intervals_complete": (
                self.finite_population_geometry_intervals_complete
            ),
            "deterministic_four_key_shells_complete": (
                self.deterministic_four_key_shells_complete
            ),
            "one_live_survivor_block_capacity_verified": (
                self.one_live_survivor_block_capacity_verified
            ),
            "source_derived_from_typed_interval_receipts": (
                self.source_derived_from_typed_interval_receipts
            ),
            "authority_scope": self.authority_scope,
        }
        # Historical exact-zero and full-sensitivity witnesses retain their
        # byte identity.  The new coordinate exists only on the v1 decision-
        # level theorem path.
        if self.decision_level_robust_invariance_certificate_hash is not None:
            payload["decision_level_robust_invariance_certificate_hash"] = (
                self.decision_level_robust_invariance_certificate_hash
            )
        return content_hash(
            payload,
            prefix="partition-free-v1-relaxed-kkt-adaptive-rank-witness-",
        )


def _decision_level_lex_interval_relation(
    lower: np.ndarray,
    upper: np.ndarray,
    left: int,
    right: int,
) -> tuple[int, float]:
    """Return definite lexicographic relation and its first strict margin.

    ``-1`` means left precedes right, ``1`` means right precedes left, ``0``
    means both orders remain admissible, and ``2`` is an exact all-real-key
    tie that would be decided only by physical index.
    """

    for component in range(3):
        left_lo = float(lower[left, component])
        left_hi = float(upper[left, component])
        right_lo = float(lower[right, component])
        right_hi = float(upper[right, component])
        if left_hi < right_lo:
            return -1, math.nextafter(right_lo - left_hi, -math.inf)
        if right_hi < left_lo:
            return 1, math.nextafter(left_lo - right_hi, -math.inf)
        if left_lo == left_hi == right_lo == right_hi:
            continue
        return 0, 0.0
    return 2, 0.0


def _decision_level_topological_orders(
    nodes: tuple[int, ...],
    precedence: dict[int, set[int]],
    *,
    maximum_outcomes: int,
) -> Iterator[tuple[int, ...]]:
    """Enumerate every linear extension, refusing rather than truncating."""

    produced = 0

    def visit(
        prefix: tuple[int, ...], remaining: frozenset[int]
    ) -> Iterator[tuple[int, ...]]:
        nonlocal produced
        if not remaining:
            produced += 1
            if produced > maximum_outcomes:
                _refuse(
                    V1RelaxedKKTRefusalCode.DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE,
                    "active shell has more admissible rounding outcomes than the "
                    "registered exhaustive reference limit",
                )
            yield prefix
            return
        ready = tuple(
            value for value in sorted(remaining) if not (precedence[value] & remaining)
        )
        if not ready:
            raise RuntimeError("decision-level shell precedence graph is cyclic")
        for value in ready:
            yield from visit(prefix + (value,), remaining - {value})

    yield from visit((), frozenset(nodes))


def _decision_level_block_axis(
    problem: V1RelaxedKKTProblem,
) -> tuple[np.ndarray, tuple[int, ...]]:
    route_to_block = np.zeros(problem.route_count, dtype=np.int64)
    if not problem.block_capacities:
        return route_to_block, (problem.route_count,)
    capacities: list[int] = []
    for offset, block in enumerate(problem.block_capacities):
        route_to_block[block.start : block.stop] = offset
        capacities.append(block.removal_capacity)
    return route_to_block, tuple(capacities)


def _decision_level_q_interval(
    *,
    branch: V1RelaxedKKTBranch,
    evidence: V1RelaxedKKTDecisionLevelShellReferenceEvidence,
    selected_shell_offsets: tuple[int, ...],
    fixed_prefix_count: int,
) -> tuple[float, float]:
    low_terms: list[float] = []
    high_terms: list[float] = []
    for rank_offset in range(evidence.restricted_factor_lower.shape[0]):
        lower = _decision_level_outward_sum(
            (
                float(evidence.factor_offset_lower[rank_offset]),
                *(
                    float(evidence.restricted_factor_lower[rank_offset, value])
                    for value in selected_shell_offsets
                ),
            ),
            upper=False,
        )
        upper = _decision_level_outward_sum(
            (
                float(evidence.factor_offset_upper[rank_offset]),
                *(
                    float(evidence.restricted_factor_upper[rank_offset, value])
                    for value in selected_shell_offsets
                ),
            ),
            upper=True,
        )
        lo, hi = _decision_level_interval_square(lower, upper)
        low_terms.append(lo)
        high_terms.append(hi)
    if branch is V1RelaxedKKTBranch.SPECTRAL:
        cardinality = fixed_prefix_count + len(selected_shell_offsets)
        low_terms.append(
            math.nextafter(evidence.spectral_rho_lower * cardinality, -math.inf)
            if cardinality
            else 0.0
        )
        high_terms.append(
            math.nextafter(evidence.spectral_rho_upper * cardinality, math.inf)
            if cardinality
            else 0.0
        )
    else:
        residual_lower = _decision_level_outward_sum(
            (
                evidence.residual_offset_lower,
                *(
                    float(evidence.residual_direction_lower[value])
                    for value in selected_shell_offsets
                ),
            ),
            upper=False,
        )
        residual_upper = _decision_level_outward_sum(
            (
                evidence.residual_offset_upper,
                *(
                    float(evidence.residual_direction_upper[value])
                    for value in selected_shell_offsets
                ),
            ),
            upper=True,
        )
        lo, hi = _decision_level_interval_square(residual_lower, residual_upper)
        low_terms.append(lo)
        high_terms.append(hi)
    nominal_lower = _decision_level_outward_sum(low_terms, upper=False)
    nominal_upper = _decision_level_outward_sum(high_terms, upper=True)
    radius = evidence.hard_q_additive_radius_upper_bound
    return (
        max(0.0, math.nextafter(nominal_lower - radius, -math.inf)),
        max(0.0, math.nextafter(nominal_upper + radius, math.inf)),
    )


def _validated_decision_level_robust_invariance_certificate_hash(
    *,
    certificate: V1RelaxedKKTDecisionLevelRobustInvarianceCertificate | None,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    branch_result: V1RelaxedKKTBranchResult,
    branch_identity: V1RelaxedKKTDeviceShellBranchIdentityReceipt,
    geometry_uncertainty: V1RelaxedKKTGeometryUncertaintySet,
    nominal_witness: V1RelaxedKKTAdaptiveRankBranchWitness,
) -> str | None:
    if certificate is None:
        return None
    if type(certificate) is not V1RelaxedKKTDecisionLevelRobustInvarianceCertificate:
        raise TypeError("decision-level robust invariance certificate must be exact")
    certificate.assert_integrity()
    if (
        certificate.branch is not branch_result.branch
        or certificate.branch is not branch_identity.branch
        or certificate.problem_hash != problem.problem_hash
        or certificate.policy_hash != policy.policy_hash
        or certificate.branch_result_hash != branch_result.result_hash
        or certificate.branch_receipt_hash != branch_result.receipt.receipt_hash
        or certificate.nominal_adaptive_rank_witness_hash
        != nominal_witness.witness_hash
        or certificate.route_axis_hash != problem.route_axis_hash
        or certificate.route_count != problem.route_count
        or certificate.finite_population_geometry_uncertainty_set_hash
        != geometry_uncertainty.uncertainty_hash
        or certificate.rounded_ladder_hash != branch_identity.source_rounding_hash
        or certificate.endpoint_action_axis_hash
        != branch_identity.endpoint_action_axis_hash
    ):
        _refuse(
            V1RelaxedKKTRefusalCode.DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE,
            "decision-level theorem does not bind this branch/problem/policy/"
            "geometry/ladder/action coordinate",
        )
    return certificate.certificate_hash


def bind_v1_relaxed_kkt_decision_level_invariance_to_adaptive_rank_witness(
    *,
    witness: V1RelaxedKKTAdaptiveRankBranchWitness,
    certificate: V1RelaxedKKTDecisionLevelRobustInvarianceCertificate,
) -> V1RelaxedKKTAdaptiveRankBranchWitness:
    """Attach a checked decision theorem to its exact nominal source witness."""

    if type(witness) is not V1RelaxedKKTAdaptiveRankBranchWitness:
        raise TypeError("decision-level binding requires an exact nominal witness")
    if type(certificate) is not V1RelaxedKKTDecisionLevelRobustInvarianceCertificate:
        raise TypeError("decision-level binding requires an exact certificate")
    certificate.assert_integrity()
    if (
        witness.decision_level_robust_invariance_certificate_hash is not None
        or witness.relaxed_solution_sensitivity_certificate_hash is not None
        or certificate.nominal_adaptive_rank_witness_hash != witness.witness_hash
        or certificate.branch is not witness.branch
        or certificate.problem_hash != witness.problem_hash
        or certificate.policy_hash != witness.policy_hash
        or certificate.branch_result_hash != witness.branch_result_hash
        or certificate.branch_receipt_hash != witness.branch_receipt_hash
        or certificate.route_axis_hash != witness.route_axis_hash
        or certificate.route_count != witness.route_count
        or certificate.finite_population_geometry_uncertainty_set_hash
        != witness.finite_population_geometry_uncertainty_set_hash
        or certificate.rounded_ladder_hash != witness.rounded_ladder_hash
        or certificate.endpoint_action_axis_hash != witness.endpoint_action_axis_hash
    ):
        _refuse(
            V1RelaxedKKTRefusalCode.DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE,
            "decision-level theorem cannot be rebound across witness coordinates",
        )
    return replace(
        witness,
        decision_level_robust_invariance_certificate_hash=(
            certificate.certificate_hash
        ),
        _adaptive_rank_witness_issuance_token=(
            _ADAPTIVE_RANK_BRANCH_WITNESS_ISSUANCE_TOKEN
        ),
    )


def v1_relaxed_kkt_decision_level_reference_basis_axis_hash(
    problem: V1RelaxedKKTProblem,
) -> str:
    """Identity of the complete canonical-basis operator probe axis."""

    if type(problem) is not V1RelaxedKKTProblem:
        raise TypeError("reference basis identity requires an exact problem")
    return content_hash(
        {
            "schema": "partition_free_v1_relaxed_kkt_canonical_operator_basis_v1",
            "route_axis_hash": problem.route_axis_hash,
            "route_count": problem.route_count,
            "direction_indices": tuple(range(problem.route_count)),
        },
        prefix="partition-free-v1-relaxed-kkt-canonical-operator-basis-",
    )


def _decision_level_reference_basis_direction_hashes(
    problem: V1RelaxedKKTProblem,
) -> tuple[str, ...]:
    result: list[str] = []
    for index in range(problem.route_count):
        direction = np.zeros(problem.route_count, dtype=np.dtype("<f8"))
        direction[index] = 1.0
        payload = hashlib.sha256(memoryview(direction).cast("B")).hexdigest()
        result.append(
            content_hash(
                {
                    "schema": "literal_j_fp64_array_bytes_v1",
                    "role": "fixed_direction",
                    "dtype": "<f8",
                    "shape": (problem.route_count,),
                    "payload_sha256": payload,
                },
                prefix="partition-free-v1-literal-j-fp64-array-",
            )
        )
    return tuple(result)


def issue_v1_relaxed_kkt_decision_level_shell_reference_evidence(
    *,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    branch_result: V1RelaxedKKTBranchResult,
    branch_identity: V1RelaxedKKTDeviceShellBranchIdentityReceipt,
    geometry_uncertainty: V1RelaxedKKTGeometryUncertaintySet,
    literal_j_full_operator_basis_census: V1LiteralJOperatorCensusReceipt,
    registered_endpoint_removed_indices: tuple[np.ndarray, ...],
    evidence_id: str = "decision-level-complete-basis-reference-v1",
    maximum_reference_routes: int = 64,
) -> V1RelaxedKKTDecisionLevelShellReferenceEvidence:
    """Derive a conservative, bindable small-axis source enclosure.

    The landed reference source intentionally admits no caller-authored gate,
    reduced-cost, or primal/dual bounds.  It requires a complete canonical-
    basis literal-J operator axis, promotes every eligible route into the
    active shell, admits every face and every real-key order, uses the universal
    resource bounds ``[0, sum(resource)]``, and expands hard F/Q by the full
    registered A/operator uncertainty.  It is therefore slow/conservative but
    sound.  Production-tight GPU outside scans and restricted robust dual
    bounds are a separate source-issuer ABI.
    """

    if type(problem) is not V1RelaxedKKTProblem:
        raise TypeError("reference evidence issuer requires an exact problem")
    if type(policy) is not V1RelaxedKKTPolicy:
        raise TypeError("reference evidence issuer requires an exact policy")
    if type(branch_result) is not V1RelaxedKKTBranchResult:
        raise TypeError("reference evidence issuer requires an exact branch result")
    if type(branch_identity) is not V1RelaxedKKTDeviceShellBranchIdentityReceipt:
        raise TypeError("reference evidence issuer requires an exact shell identity")
    if type(geometry_uncertainty) is not V1RelaxedKKTGeometryUncertaintySet:
        raise TypeError("reference evidence issuer requires exact uncertainty")
    if (
        type(literal_j_full_operator_basis_census)
        is not V1LiteralJOperatorCensusReceipt
    ):
        raise TypeError(
            "reference evidence issuer requires an exact full-operator census"
        )
    literal_j_full_operator_basis_census.assert_integrity()
    route_limit = _positive_int(maximum_reference_routes, "maximum_reference_routes")
    if problem.route_count > route_limit:
        _refuse(
            V1RelaxedKKTRefusalCode.DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE,
            "complete-basis reference evidence exceeds its bounded route limit",
        )
    expected_direction_hashes = _decision_level_reference_basis_direction_hashes(
        problem
    )
    if (
        branch_result.branch is not branch_identity.branch
        or branch_result.receipt.problem_hash != problem.problem_hash
        or branch_result.receipt.policy_hash != policy.policy_hash
        or geometry_uncertainty.problem_hash != problem.problem_hash
        or geometry_uncertainty.route_axis_hash != problem.route_axis_hash
        or geometry_uncertainty.source_geometry_hash != problem.source_geometry_hash
        or not geometry_uncertainty.source_derived_from_registered_geometry_receipts
        or literal_j_full_operator_basis_census.route_axis_hash
        != problem.route_axis_hash
        or literal_j_full_operator_basis_census.route_count != problem.route_count
        or literal_j_full_operator_basis_census.chart_identity_hash
        != problem.source_geometry_hash
        or literal_j_full_operator_basis_census.current_mask_hash
        != branch_identity.frozen_ladder.parent_mask_hash
        or literal_j_full_operator_basis_census.direction_count != problem.route_count
        or literal_j_full_operator_basis_census.direction_hashes
        != expected_direction_hashes
        or not literal_j_full_operator_basis_census.exact_finite_population_census
        or literal_j_full_operator_basis_census.sampling_radius != 0.0
    ):
        _refuse(
            V1RelaxedKKTRefusalCode.DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE,
            "reference evidence lacks the complete registered canonical-basis "
            "operator uncertainty source",
        )
    endpoint_indices = tuple(registered_endpoint_removed_indices)
    if len(endpoint_indices) != branch_identity.frozen_ladder.endpoint_count:
        raise ValueError("reference source endpoint axis is incomplete")
    for indices, action in zip(
        endpoint_indices,
        branch_identity.frozen_ladder.actions,
        strict=True,
    ):
        if (
            RemovedRouteIndexSet.from_indices(
                indices,
                route_count=problem.route_count,
            ).index_hash
            != action.removed_route_index_hash
        ):
            _refuse(
                V1RelaxedKKTRefusalCode.DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE,
                "reference source endpoint indices differ from the frozen action",
            )
    shell = np.flatnonzero(problem.eligible_mask).astype(np.int64, copy=False)
    if shell.size == 0:
        raise ValueError("reference source requires a nonempty eligible shell")
    work_radius = (
        geometry_uncertainty.corrected_work_A_sampling_linf_radius
        + geometry_uncertainty.corrected_work_A_deterministic_linf_radius
    )
    operator_radius = (
        geometry_uncertainty.literal_j_response_operator_MJ_sampling_linf_adjoint_radius
        + geometry_uncertainty.literal_j_response_operator_MJ_deterministic_linf_adjoint_radius
        + geometry_uncertainty.fd_q_residual_deterministic_linf_adjoint_radius
    )
    keys_lower = np.zeros((problem.route_count, 3), dtype=np.float64)
    keys_upper = np.zeros((problem.route_count, 3), dtype=np.float64)
    keys_lower[:, 0] = -1.0
    keys_upper[:, 0] = 0.0
    keys_lower[:, 1] = -_ROBUST_SENSITIVITY_UNBOUNDED
    keys_upper[:, 1] = _ROBUST_SENSITIVITY_UNBOUNDED
    ratio = np.divide(
        problem.corrected_work,
        problem.resource,
        out=np.zeros(problem.route_count, dtype=np.float64),
        where=problem.resource > 0.0,
    )
    ratio_radius = np.divide(
        np.full(problem.route_count, work_radius, dtype=np.float64),
        problem.resource,
        out=np.full(problem.route_count, _ROBUST_SENSITIVITY_UNBOUNDED),
        where=problem.resource > 0.0,
    )
    keys_lower[:, 2] = np.maximum(
        0.0,
        np.nextafter(ratio - ratio_radius, -math.inf),
    )
    keys_upper[:, 2] = np.nextafter(ratio + ratio_radius, math.inf)
    shell_work = problem.corrected_work[shell]
    work_lower = np.maximum(
        0.0,
        np.nextafter(shell_work - work_radius, -math.inf),
    )
    work_upper = np.nextafter(shell_work + work_radius, math.inf)
    factor = problem.low_rank_factor[:, shell]
    sqrt_residual = np.sqrt(problem.residual_diagonal[shell])
    if branch_result.branch is V1RelaxedKKTBranch.SPECTRAL:
        residual_direction = np.zeros(int(shell.size), dtype=np.float64)
        rho = problem.residual_spectral_bound
    else:
        residual_direction = sqrt_residual
        rho = 0.0
    # If ||Delta M z||_inf <= r over the complete basis-enclosed operator
    # uncertainty, then |z^T Delta M z| <= ||z||_1 r.  The reference carries
    # the worst legal cardinality so every endpoint is outward enclosed.
    hard_q_radius = math.nextafter(
        float(problem.legal_hard_capacity) * operator_radius,
        math.inf,
    )
    return V1RelaxedKKTDecisionLevelShellReferenceEvidence(
        evidence_id=evidence_id,
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        branch_result_hash=branch_result.result_hash,
        branch_receipt_hash=branch_result.receipt.receipt_hash,
        finite_population_geometry_uncertainty_set_hash=(
            geometry_uncertainty.uncertainty_hash
        ),
        full_operator_basis_census_receipt_hash=(
            literal_j_full_operator_basis_census.receipt_hash
        ),
        rounded_ladder_hash=branch_identity.source_rounding_hash,
        endpoint_action_axis_hash=branch_identity.endpoint_action_axis_hash,
        route_axis_hash=problem.route_axis_hash,
        shell_route_indices=shell,
        canonical_real_key_lower=keys_lower,
        canonical_real_key_upper=keys_upper,
        restricted_factor_lower=factor,
        restricted_factor_upper=factor,
        factor_offset_lower=np.zeros(problem.rank, dtype=np.float64),
        factor_offset_upper=np.zeros(problem.rank, dtype=np.float64),
        corrected_work_offset_lower=0.0,
        corrected_work_offset_upper=0.0,
        corrected_work_shell_lower=work_lower,
        corrected_work_shell_upper=work_upper,
        residual_direction_lower=residual_direction,
        residual_direction_upper=residual_direction,
        residual_offset_lower=0.0,
        residual_offset_upper=0.0,
        spectral_rho_lower=rho,
        spectral_rho_upper=rho,
        robust_primal_resource_lower_bound=0.0,
        robust_conic_dual_resource_upper_bound=_decision_level_outward_sum(
            tuple(float(value) for value in problem.resource[problem.eligible_mask]),
            upper=True,
        ),
        hard_q_additive_radius_upper_bound=hard_q_radius,
        registered_endpoint_removed_indices=endpoint_indices,
        source_derived_from_registered_uncertainty=True,
        _issuance_token=_DECISION_LEVEL_SOURCE_RECEIPT_ISSUANCE_TOKEN,
    )


def issue_v1_relaxed_kkt_decision_level_robust_invariance_reference(
    *,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    branch_result: V1RelaxedKKTBranchResult,
    branch_identity: V1RelaxedKKTDeviceShellBranchIdentityReceipt,
    geometry_uncertainty: V1RelaxedKKTGeometryUncertaintySet,
    nominal_witness: V1RelaxedKKTAdaptiveRankBranchWitness,
    evidence: V1RelaxedKKTDecisionLevelShellReferenceEvidence,
    maximum_shell_routes: int = 8,
    maximum_admissible_faces: int = 6_561,
    maximum_rounding_outcomes: int = 100_000,
) -> V1RelaxedKKTDecisionLevelRobustInvarianceCertificate:
    """Exhaustively certify one small active shell in high precision.

    This is a correctness reference, not the production 57M-route issuer.  It
    consumes complete routewise key intervals, but emits only compact factored
    receipts.  Every face and every linear extension of the interval-induced
    real-key partial order is checked; enumeration is never truncated.
    """

    if type(problem) is not V1RelaxedKKTProblem:
        raise TypeError("decision-level reference issuer requires an exact problem")
    if type(policy) is not V1RelaxedKKTPolicy:
        raise TypeError("decision-level reference issuer requires an exact policy")
    if type(branch_result) is not V1RelaxedKKTBranchResult:
        raise TypeError(
            "decision-level reference issuer requires an exact branch result"
        )
    if type(branch_identity) is not V1RelaxedKKTDeviceShellBranchIdentityReceipt:
        raise TypeError(
            "decision-level reference issuer requires an exact shell identity"
        )
    if type(geometry_uncertainty) is not V1RelaxedKKTGeometryUncertaintySet:
        raise TypeError(
            "decision-level reference issuer requires exact geometry uncertainty"
        )
    if type(nominal_witness) is not V1RelaxedKKTAdaptiveRankBranchWitness:
        raise TypeError(
            "decision-level reference issuer requires an exact nominal witness"
        )
    if type(evidence) is not V1RelaxedKKTDecisionLevelShellReferenceEvidence:
        raise TypeError(
            "decision-level reference issuer requires exact numerical evidence"
        )
    evidence.assert_integrity()
    shell_limit = _positive_int(maximum_shell_routes, "maximum_shell_routes")
    face_limit = _positive_int(maximum_admissible_faces, "maximum_admissible_faces")
    outcome_limit = _positive_int(
        maximum_rounding_outcomes, "maximum_rounding_outcomes"
    )
    if evidence.shell_count > shell_limit:
        _refuse(
            V1RelaxedKKTRefusalCode.DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE,
            "active shell exceeds the registered exhaustive reference limit",
        )
    if (
        branch_result.branch is not branch_identity.branch
        or branch_result.branch is not nominal_witness.branch
        or branch_result.receipt.problem_hash != problem.problem_hash
        or branch_result.receipt.policy_hash != policy.policy_hash
        or branch_result.result_hash != nominal_witness.branch_result_hash
        or branch_result.receipt.receipt_hash != nominal_witness.branch_receipt_hash
        or branch_identity.source_rounding_hash != nominal_witness.rounded_ladder_hash
        or branch_identity.endpoint_action_axis_hash
        != nominal_witness.endpoint_action_axis_hash
        or evidence.route_axis_hash != problem.route_axis_hash
        or evidence.route_count != problem.route_count
        or evidence.problem_hash != problem.problem_hash
        or evidence.policy_hash != policy.policy_hash
        or evidence.branch_result_hash != branch_result.result_hash
        or evidence.branch_receipt_hash != branch_result.receipt.receipt_hash
        or evidence.finite_population_geometry_uncertainty_set_hash
        != geometry_uncertainty.uncertainty_hash
        or evidence.rounded_ladder_hash != branch_identity.source_rounding_hash
        or evidence.endpoint_action_axis_hash
        != branch_identity.endpoint_action_axis_hash
        or not evidence.source_derived_from_registered_uncertainty
        or geometry_uncertainty.problem_hash != problem.problem_hash
        or geometry_uncertainty.route_axis_hash != problem.route_axis_hash
        or geometry_uncertainty.uncertainty_hash
        != nominal_witness.finite_population_geometry_uncertainty_set_hash
        or nominal_witness.route_count != problem.route_count
        or nominal_witness.relaxed_solution_sensitivity_certificate_hash is not None
        or nominal_witness.decision_level_robust_invariance_certificate_hash is not None
        or not nominal_witness.source_derived_from_typed_interval_receipts
        or not branch_result.receipt.convergence.converged
        or nominal_witness.relaxed_problem_perturbation_upper_bound
        != geometry_uncertainty.relaxed_problem_perturbation_radius(
            branch_result.receipt.stationarity.work_dual
        )
    ):
        _refuse(
            V1RelaxedKKTRefusalCode.DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE,
            "nominal witness, branch, shell, geometry, and evidence do not close",
        )
    ladder = branch_identity.frozen_ladder
    registered = evidence.registered_endpoint_removed_indices
    if len(registered) != ladder.endpoint_count:
        raise ValueError("reference endpoint axis differs from the frozen ladder")
    registered_hashes: list[str] = []
    for indices, action, count in zip(
        registered,
        ladder.actions,
        ladder.cumulative_counts,
        strict=True,
    ):
        index_hash = RemovedRouteIndexSet.from_indices(
            indices,
            route_count=problem.route_count,
        ).index_hash
        if int(indices.size) != count or index_hash != action.removed_route_index_hash:
            _refuse(
                V1RelaxedKKTRefusalCode.DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE,
                "registered reference endpoint differs from the frozen action axis",
            )
        registered_hashes.append(index_hash)

    shell = tuple(int(value) for value in evidence.shell_route_indices)
    shell_set = frozenset(shell)
    if any(not bool(problem.eligible_mask[value]) for value in shell):
        _refuse(
            V1RelaxedKKTRefusalCode.DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE,
            "active shell contains an ineligible route",
        )
    key_lower = evidence.canonical_real_key_lower
    key_upper = evidence.canonical_real_key_upper
    nominal_ratio = np.divide(
        problem.corrected_work,
        problem.resource,
        out=np.full(problem.route_count, math.inf, dtype=np.float64),
        where=problem.resource > 0.0,
    )
    eligible = problem.eligible_mask
    if bool(
        np.any(nominal_ratio[eligible] < key_lower[eligible, 2])
        or np.any(nominal_ratio[eligible] > key_upper[eligible, 2])
    ):
        _refuse(
            V1RelaxedKKTRefusalCode.DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE,
            "canonical work/resource intervals do not enclose the registered problem",
        )
    numerical_key_radius = max(
        geometry_uncertainty.solver_reduced_advantage_numerical_linf_radius,
        geometry_uncertainty.relaxed_gate_numerical_linf_radius,
        geometry_uncertainty.corrected_work_ratio_deterministic_linf_radius,
    )
    fixed_prefix: list[int] = []
    fixed_suffix: list[int] = []
    outside_reduced_margins: list[float] = []
    outside_key_margins: list[float] = []
    for route in range(problem.route_count):
        if route in shell_set or not bool(eligible[route]):
            continue
        reduced_lower = -float(key_upper[route, 1])
        reduced_upper = -float(key_lower[route, 1])
        if reduced_lower <= 0.0 <= reduced_upper:
            _refuse(
                V1RelaxedKKTRefusalCode.DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE,
                "outside reduced-advantage interval touches the active-face boundary",
            )
        outside_reduced_margins.append(min(abs(reduced_lower), abs(reduced_upper)))
        relations = tuple(
            _decision_level_lex_interval_relation(
                key_lower,
                key_upper,
                route,
                shell_route,
            )
            for shell_route in shell
        )
        if all(value[0] == -1 for value in relations):
            fixed_prefix.append(route)
        elif all(value[0] == 1 for value in relations):
            fixed_suffix.append(route)
        else:
            _refuse(
                V1RelaxedKKTRefusalCode.DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE,
                "outside real-key interval touches or crosses the active shell",
            )
        route_margin = min(value[1] for value in relations)
        if route_margin <= 2.0 * numerical_key_radius:
            _refuse(
                V1RelaxedKKTRefusalCode.DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE,
                "outside real-key margin is not strict over numerical uncertainty",
            )
        outside_key_margins.append(route_margin)

    prefix_indices = np.asarray(sorted(fixed_prefix), dtype=np.int64)
    suffix_indices = np.asarray(sorted(fixed_suffix), dtype=np.int64)
    prefix_hash = RemovedRouteIndexSet.from_indices(
        prefix_indices,
        route_count=problem.route_count,
    ).index_hash
    suffix_hash = RemovedRouteIndexSet.from_indices(
        suffix_indices,
        route_count=problem.route_count,
    ).index_hash
    shell_hash = RemovedRouteIndexSet.from_indices(
        np.asarray(sorted(shell), dtype=np.int64),
        route_count=problem.route_count,
    ).index_hash
    outside_scan = V1RelaxedKKTOutsideShellIntervalScanReceipt(
        branch=branch_result.branch,
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        route_axis_hash=problem.route_axis_hash,
        route_count=problem.route_count,
        source_evidence_hash=evidence.evidence_hash,
        shell_route_index_hash=shell_hash,
        shell_count=len(shell),
        scanned_eligible_outside_route_count=(len(fixed_prefix) + len(fixed_suffix)),
        fixed_prefix_route_index_hash=prefix_hash,
        fixed_prefix_count=len(fixed_prefix),
        fixed_suffix_route_index_hash=suffix_hash,
        fixed_suffix_count=len(fixed_suffix),
        minimum_outside_reduced_advantage_margin_lower_bound=(
            min(outside_reduced_margins)
            if outside_reduced_margins
            else _ROBUST_SENSITIVITY_UNBOUNDED
        ),
        minimum_outside_real_key_margin_lower_bound=(
            min(outside_key_margins)
            if outside_key_margins
            else _ROBUST_SENSITIVITY_UNBOUNDED
        ),
        maximum_numerical_key_radius_upper_bound=numerical_key_radius,
        scan_backend="high_precision_reference",
        _issuance_token=_DECISION_LEVEL_SOURCE_RECEIPT_ISSUANCE_TOKEN,
    )

    route_to_block, block_capacities = _decision_level_block_axis(problem)
    prefix_block_counts = [0 for _ in block_capacities]
    for route in fixed_prefix:
        prefix_block_counts[int(route_to_block[route])] += 1
    if any(
        count > capacity
        for count, capacity in zip(prefix_block_counts, block_capacities, strict=True)
    ):
        _refuse(
            V1RelaxedKKTRefusalCode.DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE,
            "outside fixed prefix saturates a block before its membership is invariant",
        )
    if any(count < len(fixed_prefix) for count in ladder.cumulative_counts):
        _refuse(
            V1RelaxedKKTRefusalCode.DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE,
            "registered endpoint cuts through the outside fixed prefix",
        )

    shell_array = np.asarray(shell, dtype=np.int64)
    nominal_factor = problem.low_rank_factor[:, shell_array]
    nominal_factor_offset = (
        np.sum(problem.low_rank_factor[:, prefix_indices], axis=1, dtype=np.float64)
        if prefix_indices.size
        else np.zeros(problem.rank, dtype=np.float64)
    )
    if (
        evidence.restricted_factor_lower.shape != nominal_factor.shape
        or evidence.factor_offset_lower.shape != nominal_factor_offset.shape
        or bool(np.any(nominal_factor < evidence.restricted_factor_lower))
        or bool(np.any(nominal_factor > evidence.restricted_factor_upper))
        or bool(np.any(nominal_factor_offset < evidence.factor_offset_lower))
        or bool(np.any(nominal_factor_offset > evidence.factor_offset_upper))
    ):
        _refuse(
            V1RelaxedKKTRefusalCode.DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE,
            "restricted factor/offset intervals do not enclose the registered H chart",
        )
    nominal_work_offset = float(np.sum(problem.corrected_work[prefix_indices]))
    nominal_work_shell = problem.corrected_work[shell_array]
    if (
        not evidence.corrected_work_offset_lower
        <= nominal_work_offset
        <= evidence.corrected_work_offset_upper
        or bool(np.any(nominal_work_shell < evidence.corrected_work_shell_lower))
        or bool(np.any(nominal_work_shell > evidence.corrected_work_shell_upper))
    ):
        _refuse(
            V1RelaxedKKTRefusalCode.DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE,
            "corrected-work shell/offset intervals do not enclose the registered chart",
        )
    sqrt_residual = np.sqrt(problem.residual_diagonal)
    if branch_result.branch is V1RelaxedKKTBranch.SPECTRAL:
        nominal_residual_direction = np.zeros(len(shell), dtype=np.float64)
        nominal_residual_offset = 0.0
        residual_nominal = problem.residual_spectral_bound
    else:
        nominal_residual_direction = sqrt_residual[shell_array]
        nominal_residual_offset = float(np.sum(sqrt_residual[prefix_indices]))
        residual_nominal = 0.0
    if (
        bool(np.any(nominal_residual_direction < evidence.residual_direction_lower))
        or bool(np.any(nominal_residual_direction > evidence.residual_direction_upper))
        or not evidence.residual_offset_lower
        <= nominal_residual_offset
        <= evidence.residual_offset_upper
        or not evidence.spectral_rho_lower
        <= residual_nominal
        <= evidence.spectral_rho_upper
    ):
        _refuse(
            V1RelaxedKKTRefusalCode.DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE,
            "branch residual interval does not enclose the registered Q branch",
        )

    factor_mid = 0.5 * (
        evidence.restricted_factor_lower + evidence.restricted_factor_upper
    )
    factor_radius = np.nextafter(
        np.maximum(
            np.abs(factor_mid - evidence.restricted_factor_lower),
            np.abs(evidence.restricted_factor_upper - factor_mid),
        ),
        math.inf,
    )
    factor_mid_norm = certified_frobenius_norm_upper_bound(factor_mid)
    factor_radius_norm = certified_frobenius_norm_upper_bound(factor_radius)
    operator_mid = factor_mid.T @ factor_mid
    operator_radius = certified_gram_perturbation_norm_upper_bound(
        factor_mid_norm,
        factor_radius_norm,
    )
    if branch_result.branch is V1RelaxedKKTBranch.SPECTRAL:
        rho_mid = 0.5 * (evidence.spectral_rho_lower + evidence.spectral_rho_upper)
        rho_radius = math.nextafter(
            max(
                abs(rho_mid - evidence.spectral_rho_lower),
                abs(evidence.spectral_rho_upper - rho_mid),
            ),
            math.inf,
        )
        operator_mid = operator_mid + rho_mid * np.eye(len(shell), dtype=np.float64)
        operator_radius = _decision_level_outward_sum(
            (operator_radius, rho_radius),
            upper=True,
        )
        residual_interval_hash = content_hash(
            {
                "branch": branch_result.branch,
                "rho_lower": evidence.spectral_rho_lower,
                "rho_upper": evidence.spectral_rho_upper,
            },
            prefix="partition-free-v1-relaxed-kkt-shell-residual-",
        )
        residual_mid_for_certificate = None
        diagonal_shift_for_certificate = rho_mid
    else:
        residual_mid = 0.5 * (
            evidence.residual_direction_lower + evidence.residual_direction_upper
        )
        residual_radius = np.nextafter(
            np.maximum(
                np.abs(residual_mid - evidence.residual_direction_lower),
                np.abs(evidence.residual_direction_upper - residual_mid),
            ),
            math.inf,
        )
        residual_mid_norm = certified_frobenius_norm_upper_bound(residual_mid)
        residual_radius_norm = certified_frobenius_norm_upper_bound(residual_radius)
        operator_mid = operator_mid + np.outer(residual_mid, residual_mid)
        operator_radius = _decision_level_outward_sum(
            (
                operator_radius,
                certified_gram_perturbation_norm_upper_bound(
                    residual_mid_norm,
                    residual_radius_norm,
                ),
            ),
            upper=True,
        )
        residual_interval_hash = content_hash(
            {
                "branch": branch_result.branch,
                "direction_lower": evidence.residual_direction_lower,
                "direction_upper": evidence.residual_direction_upper,
                "offset_lower": evidence.residual_offset_lower,
                "offset_upper": evidence.residual_offset_upper,
            },
            prefix="partition-free-v1-relaxed-kkt-shell-residual-",
        )
        residual_mid_for_certificate = residual_mid
        diagonal_shift_for_certificate = 0.0
    certified_midpoint_lower = certified_gram_operator_min_eigenvalue_lower_bound(
        factor_mid,
        residual_direction=residual_mid_for_certificate,
        diagonal_shift=diagonal_shift_for_certificate,
        maximum_dimension=maximum_shell_routes,
    )
    strong_convexity = _decision_level_outward_sum(
        (certified_midpoint_lower, -operator_radius),
        upper=False,
    )
    dense_lower = np.nextafter(operator_mid - operator_radius, -math.inf)
    dense_upper = np.nextafter(operator_mid + operator_radius, math.inf)
    operator_receipt = V1RelaxedKKTActiveShellRestrictedOperatorReceipt(
        branch=branch_result.branch,
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        route_axis_hash=problem.route_axis_hash,
        source_evidence_hash=evidence.evidence_hash,
        shell_route_index_hash=shell_hash,
        shell_count=len(shell),
        factor_rank=problem.rank,
        restricted_factor_interval_hash=content_hash(
            (
                evidence.restricted_factor_lower,
                evidence.restricted_factor_upper,
            ),
            prefix="partition-free-v1-relaxed-kkt-shell-factor-interval-",
        ),
        factor_offset_interval_hash=content_hash(
            (evidence.factor_offset_lower, evidence.factor_offset_upper),
            prefix="partition-free-v1-relaxed-kkt-shell-factor-offset-",
        ),
        residual_branch_interval_hash=residual_interval_hash,
        corrected_work_offset_interval_hash=content_hash(
            (
                evidence.corrected_work_offset_lower,
                evidence.corrected_work_offset_upper,
                evidence.corrected_work_shell_lower,
                evidence.corrected_work_shell_upper,
            ),
            prefix="partition-free-v1-relaxed-kkt-shell-work-offset-",
        ),
        hard_q_offset_interval_hash=content_hash(
            (
                evidence.factor_offset_lower,
                evidence.factor_offset_upper,
                evidence.residual_offset_lower,
                evidence.residual_offset_upper,
            ),
            prefix="partition-free-v1-relaxed-kkt-shell-q-offset-",
        ),
        shell_operator_norm_sampling_radius_upper_bound=(
            geometry_uncertainty.literal_j_response_operator_MJ_sampling_linf_adjoint_radius
        ),
        shell_operator_norm_deterministic_radius_upper_bound=(
            geometry_uncertainty.literal_j_response_operator_MJ_deterministic_linf_adjoint_radius
            + geometry_uncertainty.fd_q_residual_deterministic_linf_adjoint_radius
        ),
        strong_convexity_lower_bound=strong_convexity,
        dense_reference_operator_interval_hash=content_hash(
            (dense_lower, dense_upper),
            prefix="partition-free-v1-relaxed-kkt-dense-reference-shell-operator-",
        ),
        _issuance_token=_DECISION_LEVEL_SOURCE_RECEIPT_ISSUANCE_TOKEN,
    )

    gap = branch_result.receipt.gap
    if (
        evidence.robust_primal_resource_lower_bound > gap.primal_resource_lower_bound
        or evidence.robust_conic_dual_resource_upper_bound
        < gap.conic_dual_resource_upper_bound
    ):
        _refuse(
            V1RelaxedKKTRefusalCode.DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE,
            "robust primal/dual bounds do not outward-enclose the nominal solve",
        )
    gap_upper = math.nextafter(
        evidence.robust_conic_dual_resource_upper_bound
        - evidence.robust_primal_resource_lower_bound,
        math.inf,
    )
    strict_gap_tolerance = max(
        0.0,
        _decision_level_outward_sum(
            (
                policy.absolute_gap_tolerance,
                policy.relative_gap_tolerance
                * max(1.0, evidence.robust_conic_dual_resource_upper_bound),
            ),
            upper=False,
        ),
    )
    if gap_upper >= strict_gap_tolerance:
        _refuse(
            V1RelaxedKKTRefusalCode.DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE,
            "robust primal/dual interval does not have a strict registered gap",
        )
    strict_gap_margin = math.nextafter(
        strict_gap_tolerance - gap_upper,
        -math.inf,
    )
    dual_receipt = V1RelaxedKKTRobustDualBoundReceipt(
        branch=branch_result.branch,
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        branch_result_hash=branch_result.result_hash,
        branch_receipt_hash=branch_result.receipt.receipt_hash,
        source_evidence_hash=evidence.evidence_hash,
        restricted_operator_receipt_hash=operator_receipt.receipt_hash,
        robust_primal_resource_lower_bound=(
            evidence.robust_primal_resource_lower_bound
        ),
        robust_conic_dual_resource_upper_bound=(
            evidence.robust_conic_dual_resource_upper_bound
        ),
        robust_duality_gap_upper_bound=gap_upper,
        registered_strict_gap_tolerance=strict_gap_tolerance,
        strict_gap_margin_lower_bound=strict_gap_margin,
        _issuance_token=_DECISION_LEVEL_SOURCE_RECEIPT_ISSUANCE_TOKEN,
    )

    reduced_options: list[tuple[int, ...]] = []
    for route in shell:
        reduced_lower = -float(key_upper[route, 1])
        reduced_upper = -float(key_lower[route, 1])
        if reduced_lower > 0.0:
            reduced_options.append((1,))
        elif reduced_upper < 0.0:
            reduced_options.append((0,))
        else:
            reduced_options.append((0, 1, 2))
    face_count = math.prod(len(value) for value in reduced_options)
    if face_count > face_limit:
        _refuse(
            V1RelaxedKKTRefusalCode.DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE,
            "active shell has more admissible faces than the exhaustive limit",
        )
    outcome_count = 0
    outcome_hashes: list[str] = []
    physical_ties = 0
    registered_axis = tuple(registered_hashes)
    for face in product(*reduced_options):
        face_lower = np.array(key_lower, copy=True)
        face_upper = np.array(key_upper, copy=True)
        for route, state in zip(shell, face, strict=True):
            if state == 0:
                face_lower[route, 0] = face_upper[route, 0] = 0.0
            elif state == 1:
                face_lower[route, 0] = face_upper[route, 0] = -1.0
        precedence = {route: set() for route in shell}
        for left_offset, left in enumerate(shell):
            for right in shell[left_offset + 1 :]:
                relation, margin = _decision_level_lex_interval_relation(
                    face_lower,
                    face_upper,
                    left,
                    right,
                )
                if relation == 2:
                    physical_ties += 1
                    _refuse(
                        V1RelaxedKKTRefusalCode.DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE,
                        "active-shell order contains a physical-index-only tie",
                    )
                if relation == -1 and margin > 2.0 * numerical_key_radius:
                    precedence[right].add(left)
                elif relation == 1 and margin > 2.0 * numerical_key_radius:
                    precedence[left].add(right)
        for order in _decision_level_topological_orders(
            shell,
            precedence,
            maximum_outcomes=outcome_limit - outcome_count,
        ):
            block_counts = list(prefix_block_counts)
            selected = list(fixed_prefix)
            endpoints: list[tuple[int, ...]] = []
            target_offset = 0
            while target_offset < len(
                ladder.cumulative_counts
            ) and ladder.cumulative_counts[target_offset] == len(selected):
                endpoints.append(tuple(sorted(selected)))
                target_offset += 1
            for route in order:
                block = int(route_to_block[route])
                if block_counts[block] >= block_capacities[block]:
                    continue
                block_counts[block] += 1
                selected.append(route)
                while target_offset < len(
                    ladder.cumulative_counts
                ) and ladder.cumulative_counts[target_offset] == len(selected):
                    endpoints.append(tuple(sorted(selected)))
                    target_offset += 1
            if target_offset != len(ladder.cumulative_counts):
                _refuse(
                    V1RelaxedKKTRefusalCode.DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE,
                    "one admissible shell ordering cannot reach every frozen endpoint",
                )
            axis = tuple(
                RemovedRouteIndexSet.from_indices(
                    np.asarray(value, dtype=np.int64),
                    route_count=problem.route_count,
                ).index_hash
                for value in endpoints
            )
            outcome_count += 1
            if outcome_count > outcome_limit:
                _refuse(
                    V1RelaxedKKTRefusalCode.DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE,
                    "active shell exceeds the exhaustive rounding-outcome limit",
                )
            if axis != registered_axis:
                _refuse(
                    V1RelaxedKKTRefusalCode.DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE,
                    "admissible nonunique face or key order rounds to a different ladder",
                )
            outcome_hashes.append(
                content_hash(
                    {"face": face, "order": order, "ladder_axis": axis},
                    prefix="partition-free-v1-relaxed-kkt-shell-face-outcome-",
                )
            )
    if outcome_count <= 0:
        raise RuntimeError("decision-level shell enumeration produced no outcomes")
    faces_receipt = V1RelaxedKKTAllAdmissibleShellFacesReceipt(
        branch=branch_result.branch,
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        source_evidence_hash=evidence.evidence_hash,
        shell_route_index_hash=shell_hash,
        rounded_ladder_hash=branch_identity.source_rounding_hash,
        endpoint_action_axis_hash=branch_identity.endpoint_action_axis_hash,
        admissible_face_count=face_count,
        admissible_rounding_outcome_count=outcome_count,
        outcome_ladder_axis_hash=content_hash(
            tuple(outcome_hashes),
            prefix="partition-free-v1-relaxed-kkt-all-shell-outcomes-",
        ),
        physical_index_only_tie_count=physical_ties,
        semidefinite_or_nonunique_face_count=(
            face_count if strong_convexity <= 0.0 else max(0, face_count - 1)
        ),
        _issuance_token=_DECISION_LEVEL_SOURCE_RECEIPT_ISSUANCE_TOKEN,
    )

    f_intervals: list[tuple[float, float]] = []
    q_intervals: list[tuple[float, float]] = []
    statuses: list[str] = []
    shell_offset = {route: offset for offset, route in enumerate(shell)}
    prefix_set = set(fixed_prefix)
    for endpoint in registered:
        endpoint_set = set(int(value) for value in endpoint)
        if not prefix_set.issubset(endpoint_set) or not endpoint_set.issubset(
            prefix_set | shell_set
        ):
            _refuse(
                V1RelaxedKKTRefusalCode.DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE,
                "registered endpoint is not contained in fixed-prefix plus active shell",
            )
        selected_offsets = tuple(
            shell_offset[value] for value in sorted(endpoint_set - prefix_set)
        )
        f_lower = max(
            0.0,
            _decision_level_outward_sum(
                (
                    evidence.corrected_work_offset_lower,
                    *(
                        float(evidence.corrected_work_shell_lower[value])
                        for value in selected_offsets
                    ),
                ),
                upper=False,
            ),
        )
        f_upper = max(
            0.0,
            _decision_level_outward_sum(
                (
                    evidence.corrected_work_offset_upper,
                    *(
                        float(evidence.corrected_work_shell_upper[value])
                        for value in selected_offsets
                    ),
                ),
                upper=True,
            ),
        )
        q_lower, q_upper = _decision_level_q_interval(
            branch=branch_result.branch,
            evidence=evidence,
            selected_shell_offsets=selected_offsets,
            fixed_prefix_count=len(fixed_prefix),
        )
        safe = f_upper < problem.work_budget and q_upper < problem.quadratic_budget
        impossible = f_lower > problem.work_budget or q_lower > problem.quadratic_budget
        if safe == impossible or (not safe and not impossible):
            _refuse(
                V1RelaxedKKTRefusalCode.DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE,
                "worst-case hard F/Q interval touches a feasibility boundary",
            )
        f_intervals.append((f_lower, f_upper))
        q_intervals.append((q_lower, q_upper))
        statuses.append("safe_feasible" if safe else "robust_impossible")
    hard_status_axis_hash = content_hash(
        tuple(zip(registered_hashes, statuses, strict=True)),
        prefix="partition-free-v1-relaxed-kkt-decision-hard-fq-status-axis-",
    )
    hard_fq_receipt = V1RelaxedKKTWorstCaseHardFQReplayReceipt(
        branch=branch_result.branch,
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        source_evidence_hash=evidence.evidence_hash,
        rounded_ladder_hash=branch_identity.source_rounding_hash,
        endpoint_action_axis_hash=branch_identity.endpoint_action_axis_hash,
        endpoint_removed_index_hashes=tuple(registered_hashes),
        endpoint_f_intervals=tuple(f_intervals),
        endpoint_q_intervals=tuple(q_intervals),
        endpoint_statuses=tuple(statuses),
        hard_fq_status_axis_hash=hard_status_axis_hash,
        _issuance_token=_DECISION_LEVEL_SOURCE_RECEIPT_ISSUANCE_TOKEN,
    )
    return V1RelaxedKKTDecisionLevelRobustInvarianceCertificate(
        branch=branch_result.branch,
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        branch_result_hash=branch_result.result_hash,
        branch_receipt_hash=branch_result.receipt.receipt_hash,
        nominal_adaptive_rank_witness_hash=nominal_witness.witness_hash,
        route_axis_hash=problem.route_axis_hash,
        route_count=problem.route_count,
        finite_population_geometry_uncertainty_set_hash=(
            geometry_uncertainty.uncertainty_hash
        ),
        rounded_ladder_hash=branch_identity.source_rounding_hash,
        endpoint_action_axis_hash=branch_identity.endpoint_action_axis_hash,
        outside_shell_interval_scan=outside_scan,
        active_shell_restricted_operator=operator_receipt,
        robust_dual_bound=dual_receipt,
        worst_case_hard_fq_replay=hard_fq_receipt,
        all_admissible_shell_faces=faces_receipt,
        _issuance_token=_DECISION_LEVEL_ROBUST_INVARIANCE_ISSUANCE_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1RelaxedKKTDecisionLevelTheoremStateBorrowRequest:
    """Array-free caller lineage for one post-D_W CUDA theorem borrow.

    The KKT owner verifies every coordinate it owns.  ``source_evidence_hash``
    is deliberately only *bound* here: the separate streaming-digest owner
    remains responsible for attesting evidence, fixed-prefix/suffix hashes,
    and endpoint outcomes.  This request cannot synthesize those identities.
    """

    problem_hash: str
    policy_hash: str
    result_hash: str
    branch: V1RelaxedKKTBranch
    branch_result_hash: str
    branch_receipt_hash: str
    device_shell_execution_receipt_hash: str
    device_shell_execution_design_hash: str
    source_rounding_hash: str
    endpoint_action_axis_hash: str
    geometry_uncertainty_hash: str
    private_primal_descriptor_axis_hash: str
    private_primal_branch_descriptor_hash: str
    literal_j_projection_receipt_hash: str
    literal_j_projection_direction_axis_hash: str
    literal_j_projection_output_axis_hash: str
    literal_j_private_primal_descriptor_axis_hash: str
    source_evidence_hash: str
    maximum_shell_routes: int = 8
    schema: str = RELAXED_KKT_DECISION_LEVEL_THEOREM_STATE_BORROW_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("theorem-state request requires an exact branch")
        for name in (
            "problem_hash",
            "policy_hash",
            "result_hash",
            "branch_result_hash",
            "branch_receipt_hash",
            "device_shell_execution_receipt_hash",
            "device_shell_execution_design_hash",
            "source_rounding_hash",
            "endpoint_action_axis_hash",
            "geometry_uncertainty_hash",
            "private_primal_descriptor_axis_hash",
            "private_primal_branch_descriptor_hash",
            "literal_j_projection_receipt_hash",
            "literal_j_projection_direction_axis_hash",
            "literal_j_projection_output_axis_hash",
            "literal_j_private_primal_descriptor_axis_hash",
            "source_evidence_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "maximum_shell_routes",
            _positive_int(self.maximum_shell_routes, "maximum_shell_routes"),
        )
        if self.schema != (
            RELAXED_KKT_DECISION_LEVEL_THEOREM_STATE_BORROW_REQUEST_SCHEMA
        ):
            raise ValueError("unsupported theorem-state borrow request schema")

    @property
    def request_hash(self) -> str:
        return content_hash(
            self,
            prefix=(
                "partition-free-v1-relaxed-kkt-decision-level-theorem-"
                "state-borrow-request-"
            ),
        )


@dataclass(frozen=True, slots=True)
class V1RelaxedKKTDecisionLevelTheoremLeaseHBMForecast:
    """Pre-allocation admission for one narrow theorem-state CUDA lease."""

    problem_hash: str
    request_hash: str
    device: str
    route_count: int
    rank: int
    endpoint_count: int
    maximum_endpoint: int
    maximum_shell_routes: int
    route_tile_size: int
    current_reserved_bytes: int
    canonical_key_interval_bytes: int
    bounded_shell_workspace_bytes: int
    forecast_incremental_bytes: int
    forecast_peak_reserved_bytes: int
    device_total_memory_bytes: int
    hbm_ceiling_bytes: int
    admitted: bool
    hbm_ceiling_fraction: float = _CUDA_DEFAULT_HBM_CEILING_FRACTION
    schema: str = RELAXED_KKT_DECISION_LEVEL_THEOREM_LEASE_HBM_FORECAST_SCHEMA
    _admission_memory: InitVar[tuple[int, int] | None] = None

    def __post_init__(self, _admission_memory: tuple[int, int] | None) -> None:
        for name in ("problem_hash", "request_hash", "device"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        for name in (
            "route_count",
            "endpoint_count",
            "maximum_endpoint",
            "maximum_shell_routes",
            "route_tile_size",
            "current_reserved_bytes",
            "canonical_key_interval_bytes",
            "bounded_shell_workspace_bytes",
            "forecast_incremental_bytes",
            "forecast_peak_reserved_bytes",
            "device_total_memory_bytes",
            "hbm_ceiling_bytes",
        ):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name))
        object.__setattr__(self, "rank", _nonnegative_int(self.rank, "rank"))
        if (
            self.route_count <= 0
            or self.endpoint_count <= 0
            or self.maximum_endpoint <= 0
            or self.maximum_shell_routes <= 0
            or self.route_tile_size <= 0
            or self.device_total_memory_bytes <= 0
        ):
            raise ValueError("theorem-lease HBM axes and memory must be positive")
        if self.maximum_endpoint > self.route_count:
            raise ValueError("theorem-lease endpoint exceeds the route axis")
        fraction = float(self.hbm_ceiling_fraction)
        if fraction != _CUDA_DEFAULT_HBM_CEILING_FRACTION:
            raise ValueError("theorem-lease HBM ceiling is fixed at 85 percent")
        object.__setattr__(self, "hbm_ceiling_fraction", fraction)
        expected_keys = 2 * self.route_count * 3 * np.dtype(np.float64).itemsize
        if self.canonical_key_interval_bytes != expected_keys:
            raise ValueError("canonical key-interval byte accounting is not exact")
        expected_incremental = (
            self.canonical_key_interval_bytes + self.bounded_shell_workspace_bytes
        )
        if self.forecast_incremental_bytes != expected_incremental:
            raise ValueError("theorem-lease incremental-byte accounting does not close")
        if self.forecast_peak_reserved_bytes != (
            self.current_reserved_bytes + self.forecast_incremental_bytes
        ):
            raise ValueError("theorem-lease peak-byte accounting does not close")
        expected_ceiling = int(
            math.floor(
                self.device_total_memory_bytes * _CUDA_DEFAULT_HBM_CEILING_FRACTION
            )
        )
        if self.hbm_ceiling_bytes != expected_ceiling:
            raise ValueError("theorem-lease 85-percent HBM ceiling is not exact")
        available, allocated = (
            (self.hbm_ceiling_bytes, self.current_reserved_bytes)
            if _admission_memory is None
            else _admission_memory
        )
        if self.admitted is not (
            allocated + self.forecast_incremental_bytes <= self.hbm_ceiling_bytes
            and self.forecast_incremental_bytes <= available
        ):
            raise ValueError("theorem-lease admission differs from its byte ceiling")
        if self.schema != (
            RELAXED_KKT_DECISION_LEVEL_THEOREM_LEASE_HBM_FORECAST_SCHEMA
        ):
            raise ValueError("unsupported theorem-lease HBM forecast schema")

    @property
    def forecast_hash(self) -> str:
        return content_hash(
            self,
            prefix=(
                "partition-free-v1-relaxed-kkt-decision-level-theorem-"
                "lease-hbm-forecast-"
            ),
        )


def estimate_v1_relaxed_kkt_decision_level_theorem_lease_hbm(
    *,
    problem: V1RelaxedKKTProblem,
    request: V1RelaxedKKTDecisionLevelTheoremStateBorrowRequest,
    endpoint_counts: Sequence[int],
    route_tile_size: int,
    device_total_memory_bytes: int,
    current_reserved_bytes: int = 0,
    device: str = "cuda",
    _admission_memory: tuple[int, int] | None = None,
) -> V1RelaxedKKTDecisionLevelTheoremLeaseHBMForecast:
    """Forecast interval construction and bounded-shell work before allocation."""

    if type(problem) is not V1RelaxedKKTProblem:
        raise TypeError("theorem-lease HBM forecast requires an exact problem")
    if type(request) is not V1RelaxedKKTDecisionLevelTheoremStateBorrowRequest:
        raise TypeError("theorem-lease HBM forecast requires an exact request")
    if request.problem_hash != problem.problem_hash:
        raise ValueError("theorem-lease HBM request belongs to another problem")
    counts = tuple(_positive_int(value, "endpoint_count") for value in endpoint_counts)
    if not counts or tuple(sorted(set(counts))) != counts:
        raise ValueError("theorem-lease endpoints must be unique and increasing")
    n = problem.route_count
    tile = min(n, _positive_int(route_tile_size, "route_tile_size"))
    maximum_endpoint = counts[-1]
    if maximum_endpoint > problem.legal_hard_capacity:
        raise ValueError("theorem-lease endpoint exceeds legal hard capacity")
    shell_limit = request.maximum_shell_routes
    f64 = np.dtype(np.float64).itemsize
    i64 = np.dtype(np.int64).itemsize
    u8 = np.dtype(np.uint8).itemsize
    key_bytes = 2 * n * 3 * f64
    order_and_rounding = (maximum_endpoint + 1) * i64 + 12 * (
        maximum_endpoint + tile
    ) * f64
    ambiguity_scan = (
        2 * tile * i64
        + (16 + shell_limit) * tile * u8
        + (4 * shell_limit + 2 * len(counts)) * i64
        + 2 * f64
    )
    shell_workspace = order_and_rounding + ambiguity_scan
    incremental = key_bytes + shell_workspace
    total = _positive_int(device_total_memory_bytes, "device_total_memory_bytes")
    reserved = _nonnegative_int(current_reserved_bytes, "current_reserved_bytes")
    ceiling = int(math.floor(total * _CUDA_DEFAULT_HBM_CEILING_FRACTION))
    peak = reserved + incremental
    available, allocated = (
        (ceiling, reserved) if _admission_memory is None else _admission_memory
    )
    return V1RelaxedKKTDecisionLevelTheoremLeaseHBMForecast(
        problem_hash=problem.problem_hash,
        request_hash=request.request_hash,
        device=_nonempty(device, "device"),
        route_count=n,
        rank=problem.rank,
        endpoint_count=len(counts),
        maximum_endpoint=maximum_endpoint,
        maximum_shell_routes=shell_limit,
        route_tile_size=tile,
        current_reserved_bytes=reserved,
        canonical_key_interval_bytes=key_bytes,
        bounded_shell_workspace_bytes=shell_workspace,
        forecast_incremental_bytes=incremental,
        forecast_peak_reserved_bytes=peak,
        device_total_memory_bytes=total,
        hbm_ceiling_bytes=ceiling,
        admitted=allocated + incremental <= ceiling and incremental <= available,
        _admission_memory=_admission_memory,
    )


def _forecast_v1_relaxed_kkt_theorem_lease_hbm_from_device(
    *,
    problem: V1RelaxedKKTProblem,
    request: V1RelaxedKKTDecisionLevelTheoremStateBorrowRequest,
    endpoint_counts: tuple[int, ...],
    route_tile_size: int,
    device: Any,
) -> V1RelaxedKKTDecisionLevelTheoremLeaseHBMForecast:
    """Read scalar device capacity/reservation before either key allocation."""

    torch = _torch_module()
    if getattr(device, "type", None) != "cuda" or not torch.cuda.is_available():
        _refuse(
            V1RelaxedKKTRefusalCode.CUDA_UNAVAILABLE,
            "theorem-lease HBM admission requires the source CUDA device",
        )
    try:
        properties = torch.cuda.get_device_properties(device)
        current_reserved = int(torch.cuda.memory_reserved(device))
        admission_memory = (
            cuda_reusable_memory_bytes(device, framework=torch),
            int(torch.cuda.memory_allocated(device)),
        )
    except Exception as error:  # pragma: no cover - device dependent.
        _refuse(V1RelaxedKKTRefusalCode.CUDA_UNAVAILABLE, str(error))
    return estimate_v1_relaxed_kkt_decision_level_theorem_lease_hbm(
        problem=problem,
        request=request,
        endpoint_counts=endpoint_counts,
        route_tile_size=route_tile_size,
        device_total_memory_bytes=int(properties.total_memory),
        current_reserved_bytes=current_reserved,
        device=str(device),
        _admission_memory=admission_memory,
    )


def _evict_v1_relaxed_kkt_device_shell_owners_atomically(
    *,
    spectral_label_handle: V1DeviceShellLabelHandle,
    diagonal_l1_label_handle: V1DeviceShellLabelHandle,
    private_primal_direction_handle: V1LiteralJPrivatePrimalDirectionCUDAHandle | None,
    theorem_state_custody: _V1RelaxedKKTTheoremStateCustody | None = None,
) -> None:
    """Release every device-shell owner or leave every owner live.

    The label and private-primal handle implementations each own a lifecycle
    lock, but their public atomic helpers cannot coordinate across handle
    classes.  This aggregate owner acquires the complete lock set in stable
    object-identity order and performs all lease preflights before releasing
    any tensor.  No route payload is inspected, hashed, or transferred.
    """

    labels = (spectral_label_handle, diagonal_l1_label_handle)
    if any(type(handle) is not V1DeviceShellLabelHandle for handle in labels):
        raise TypeError("device-shell atomic eviction requires exact label handles")
    primal = private_primal_direction_handle
    custody = theorem_state_custody
    if custody is not None:
        if type(custody) is not _V1RelaxedKKTTheoremStateCustody:
            raise TypeError("device-shell eviction requires exact theorem custody")
    if (
        primal is not None
        and type(primal) is not V1LiteralJPrivatePrimalDirectionCUDAHandle
    ):
        raise TypeError("device-shell atomic eviction requires an exact primal handle")

    owners: tuple[object, ...] = (
        *labels,
        *((primal,) if primal is not None else ()),
        *((custody,) if custody is not None else ()),
    )
    with ExitStack() as stack:
        for owner in sorted(owners, key=id):
            # Every owner uses a reentrant lifecycle lock, so its public
            # validated eviction operation remains authoritative under this
            # wider aggregate transaction.
            stack.enter_context(owner._lease_lock)  # type: ignore[attr-defined]
        # Cleanup must remain possible after a mutation refusal.  These raw
        # resident checks grant no execution access; every scientific consumer
        # continues through the stricter registered-invariant path.
        spectral_label_handle._require_resident_locked()  # type: ignore[attr-defined]
        diagonal_l1_label_handle._require_resident_locked()  # type: ignore[attr-defined]
        if primal is not None:
            primal._require_live_locked()  # type: ignore[attr-defined]
        if custody is not None:
            custody._require_live_locked()
        if (
            spectral_label_handle._active_lease_tokens  # type: ignore[attr-defined]
            or diagonal_l1_label_handle._active_lease_tokens  # type: ignore[attr-defined]
            or (
                primal is not None
                and primal._active_lease_tokens  # type: ignore[attr-defined]
            )
            or (custody is not None and custody._active_lease_records)
        ):
            _refuse(
                V1RelaxedKKTRefusalCode.CUDA_DEVICE_SHELL_ACTIVE_LEASES,
                "device-shell owner eviction refused because a live lease "
                "appeared during aggregate preflight",
            )
        V1DeviceShellLabelHandle.evict_all_atomically(labels)
        # No lease can appear while the aggregate lock set remains held.
        if primal is not None:
            primal.evict()
        if custody is not None:
            custody.evict()


class V1RelaxedKKTDeviceShellExecutionBundle:
    """Ephemeral handles plus their array-free, authority-inert receipt.

    Handles never enter :class:`V1RelaxedKKTResult` or any content-addressed
    receipt.  Callers must evict this bundle after the downstream identity
    bridge has consumed the selected frozen endpoint.
    """

    __slots__ = (
        "adaptive_rank_witnesses",
        "base_execution",
        "execution_design",
        "geometry_uncertainty",
        "receipt",
        "_spectral_label_handle",
        "_diagonal_l1_label_handle",
        "_private_primal_direction_handle",
        "_theorem_state_custody",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        base_execution: V1RelaxedKKTCUDAExecutionBundle,
        execution_design: V1RelaxedKKTDeviceShellExecutionDesign,
        receipt: V1RelaxedKKTDeviceShellExecutionReceipt,
        spectral_label_handle: V1DeviceShellLabelHandle,
        diagonal_l1_label_handle: V1DeviceShellLabelHandle,
        geometry_uncertainty: V1RelaxedKKTGeometryUncertaintySet | None = None,
        adaptive_rank_witnesses: tuple[V1RelaxedKKTAdaptiveRankBranchWitness, ...] = (),
        private_primal_direction_handle: (
            V1LiteralJPrivatePrimalDirectionCUDAHandle | None
        ) = None,
        _theorem_state_custody: _V1RelaxedKKTTheoremStateCustody | None = None,
    ) -> None:
        if type(base_execution) is not V1RelaxedKKTCUDAExecutionBundle:
            raise TypeError("device-shell bundle requires exact base execution")
        if type(execution_design) is not V1RelaxedKKTDeviceShellExecutionDesign:
            raise TypeError("device-shell bundle requires its exact design")
        if type(receipt) is not V1RelaxedKKTDeviceShellExecutionReceipt:
            raise TypeError("device-shell bundle requires an exact receipt")
        if (
            type(spectral_label_handle) is not V1DeviceShellLabelHandle
            or type(diagonal_l1_label_handle) is not V1DeviceShellLabelHandle
        ):
            raise TypeError("device-shell bundle requires two exact label handles")
        if (
            receipt.problem_hash != base_execution.result.problem_hash
            or receipt.policy_hash != base_execution.result.policy_hash
            or receipt.base_result_hash != base_execution.result.result_hash
            or receipt.hbm_forecast_hash != base_execution.forecast.forecast_hash
            or receipt.cuda_telemetry_hash != base_execution.telemetry.telemetry_hash
            or receipt.device_shell_execution_design_hash
            != execution_design.design_hash
            or receipt.cuda_execution_design_hash
            != execution_design.cuda_execution_design.design_hash
        ):
            raise ValueError("device-shell execution bundle does not close")
        if base_execution.forecast.rank != base_execution.telemetry.rank:
            raise ValueError("device-shell problem/forecast and telemetry ranks differ")
        if (
            spectral_label_handle.route_count
            != receipt.spectral.frozen_ladder.route_count
            or diagonal_l1_label_handle.route_count
            != receipt.diagonal_l1.frozen_ladder.route_count
        ):
            raise ValueError("device-shell handles differ from their frozen receipts")
        uncertainty = geometry_uncertainty
        if uncertainty is not None:
            if type(uncertainty) is not V1RelaxedKKTGeometryUncertaintySet:
                raise TypeError("device-shell uncertainty must be exact")
            if (
                uncertainty.problem_hash != base_execution.result.problem_hash
                or uncertainty.route_axis_hash
                != receipt.spectral.frozen_ladder.route_axis_hash
                or uncertainty.source_geometry_hash
                != base_execution.telemetry.source_geometry_hash
                or not uncertainty.source_derived_from_registered_geometry_receipts
            ):
                raise ValueError(
                    "device-shell uncertainty does not bind this source chart"
                )
        witnesses = tuple(adaptive_rank_witnesses)
        if any(
            type(value) is not V1RelaxedKKTAdaptiveRankBranchWitness
            for value in witnesses
        ):
            raise TypeError("device-shell adaptive-rank witnesses must be exact")
        if witnesses:
            if uncertainty is None:
                raise PermissionError(
                    "device-shell witnesses require source-derived uncertainty"
                )
            if tuple(value.branch.value for value in witnesses) != (
                "spectral",
                "diagonal_l1",
            ):
                raise ValueError(
                    "device-shell witnesses must use the canonical branch axis"
                )
            expected = {
                V1RelaxedKKTBranch.SPECTRAL: (
                    base_execution.result.spectral,
                    receipt.spectral,
                ),
                V1RelaxedKKTBranch.DIAGONAL_L1: (
                    base_execution.result.diagonal_l1,
                    receipt.diagonal_l1,
                ),
            }
            for witness in witnesses:
                branch_result, branch_identity = expected[witness.branch]
                if (
                    witness.problem_hash != base_execution.result.problem_hash
                    or witness.policy_hash != base_execution.result.policy_hash
                    or witness.branch_result_hash != branch_result.result_hash
                    or witness.branch_receipt_hash != branch_result.receipt.receipt_hash
                    or witness.rounded_ladder_hash
                    != branch_identity.source_rounding_hash
                    or witness.endpoint_action_axis_hash
                    != branch_identity.endpoint_action_axis_hash
                    or witness.route_axis_hash
                    != branch_identity.frozen_ladder.route_axis_hash
                    or witness.route_count != branch_identity.frozen_ladder.route_count
                    or witness.finite_population_geometry_uncertainty_set_hash
                    != uncertainty.uncertainty_hash
                    or witness.literal_j_fixed_primal_direction_axis_hash
                    != uncertainty.literal_j_fixed_primal_direction_axis_hash
                ):
                    raise ValueError(
                        "device-shell witness does not bind this shell execution"
                    )
        if (
            execution_design.cuda_execution_design.require_adaptive_shell_witness
            and len(witnesses) != 2
        ):
            raise PermissionError(
                "device-shell execution required both source-derived "
                "adaptive-rank branch witnesses"
            )
        primal_handle = private_primal_direction_handle
        theorem_custody = _theorem_state_custody
        if theorem_custody is not None:
            if type(theorem_custody) is not _V1RelaxedKKTTheoremStateCustody:
                raise TypeError("device-shell theorem-state custody must be exact")
            theorem_custody.assert_execution_lineage(
                result=base_execution.result,
                route_count=base_execution.telemetry.route_count,
            )
            if primal_handle is None:
                raise ValueError(
                    "theorem-state custody requires its private-primal sibling"
                )
        if primal_handle is not None:
            if type(primal_handle) is not V1LiteralJPrivatePrimalDirectionCUDAHandle:
                raise TypeError(
                    "device-shell private-primal direction handle must be exact"
                )
            descriptor_axis = primal_handle.descriptor_axis
            expected_route_axis_hash = receipt.spectral.frozen_ladder.route_axis_hash
            expected_rank_marker = f":rank={base_execution.telemetry.rank}:"
            if (
                descriptor_axis.source_problem_hash
                != base_execution.result.problem_hash
                or descriptor_axis.source_policy_hash
                != base_execution.result.policy_hash
                or descriptor_axis.source_result_hash
                != base_execution.result.result_hash
                or descriptor_axis.source_execution_receipt_hash != receipt.receipt_hash
                or descriptor_axis.source_execution_design_hash
                != execution_design.design_hash
                or descriptor_axis.route_axis_hash != expected_route_axis_hash
                or descriptor_axis.route_count != base_execution.telemetry.route_count
                or primal_handle.route_count != base_execution.telemetry.route_count
                or any(
                    expected_rank_marker not in descriptor.direction_id
                    for descriptor in descriptor_axis.descriptors
                )
            ):
                raise ValueError(
                    "private-primal descriptor axis does not bind this shell execution"
                )
            expected_branches = (
                base_execution.result.spectral,
                base_execution.result.diagonal_l1,
            )
            expected_shell_branches = (receipt.spectral, receipt.diagonal_l1)
            for descriptor, branch_result, branch_identity in zip(
                descriptor_axis.descriptors,
                expected_branches,
                expected_shell_branches,
                strict=True,
            ):
                if (
                    descriptor.source_branch_result_hash != branch_result.result_hash
                    or descriptor.source_branch_receipt_hash
                    != branch_result.receipt.receipt_hash
                    or descriptor.branch != branch_result.branch.value
                    or branch_identity.branch is not branch_result.branch
                ):
                    raise ValueError(
                        "private-primal branch descriptor does not bind its KKT branch"
                    )
            if (
                spectral_label_handle.device_identity
                != diagonal_l1_label_handle.device_identity
                or primal_handle.device_identity
                != spectral_label_handle.device_identity
            ):
                raise ValueError(
                    "device-shell labels and private primals must share one device"
                )
        self.base_execution = base_execution
        self.execution_design = execution_design
        self.geometry_uncertainty = uncertainty
        self.adaptive_rank_witnesses = witnesses
        self.receipt = receipt
        self._spectral_label_handle = spectral_label_handle
        self._diagonal_l1_label_handle = diagonal_l1_label_handle
        self._private_primal_direction_handle = primal_handle
        self._theorem_state_custody = theorem_custody

    @property
    def spectral_label_handle(self) -> V1DeviceShellLabelHandle:
        self._spectral_label_handle._require_live()
        return self._spectral_label_handle

    @property
    def diagonal_l1_label_handle(self) -> V1DeviceShellLabelHandle:
        self._diagonal_l1_label_handle._require_live()
        return self._diagonal_l1_label_handle

    @property
    def problem_rank(self) -> int:
        """Rank of the exact problem/forecast/telemetry execution coordinate."""

        return self.base_execution.telemetry.rank

    @property
    def has_private_primal_direction_custody(self) -> bool:
        return self._private_primal_direction_handle is not None

    @property
    def active_label_lease_count(self) -> int:
        """Live label borrows currently preventing device-shell eviction."""

        return (
            self._spectral_label_handle.active_lease_count
            + self._diagonal_l1_label_handle.active_lease_count
        )

    @property
    def active_private_primal_direction_lease_count(self) -> int:
        """Live private-primal borrows currently preventing eviction."""

        handle = self._private_primal_direction_handle
        if handle is None:
            return 0
        return handle.active_lease_count

    @property
    def active_decision_level_theorem_state_lease_count(self) -> int:
        """Live narrow source-theorem borrows currently preventing eviction."""

        custody = self._theorem_state_custody
        if custody is None:
            return 0
        return custody.active_lease_count

    @property
    def active_device_shell_lease_count(self) -> int:
        """Total live resident-state leases owned by this ephemeral bundle."""

        return (
            self.active_label_lease_count
            + self.active_private_primal_direction_lease_count
            + self.active_decision_level_theorem_state_lease_count
        )

    @property
    def has_private_branch_state_custody(self) -> bool:
        """Whether opaque theorem-state custody remains resident and live."""

        custody = self._theorem_state_custody
        return custody is not None and not custody.is_evicted

    @property
    def private_branch_state_retained_tensor_bytes(self) -> int:
        """Scalar HBM accounting only; no state tensor or accessor is exposed."""

        custody = self._theorem_state_custody
        return 0 if custody is None else custody.retained_tensor_bytes

    def private_branch_state(
        self,
        branch: V1RelaxedKKTBranch,
    ) -> NoReturn:
        """Refuse the superseded direct whole-state access surface."""

        if type(branch) is not V1RelaxedKKTBranch:
            raise TypeError("private branch-state access requires an exact branch")
        _refuse(
            V1RelaxedKKTRefusalCode.CUDA_SHELL_WITNESS_UNAVAILABLE,
            "whole private branch-state custody is unavailable; use the "
            "private-primal literal-J projection lease",
        )

    @property
    def private_primal_direction_descriptor_axis(
        self,
    ) -> V1LiteralJPrivatePrimalDirectionDescriptorAxis:
        handle = self._private_primal_direction_handle
        if handle is None:
            raise RuntimeError(
                "private-primal directions are unavailable on this legacy bundle"
            )
        return handle.descriptor_axis

    @property
    def private_primal_direction_handle(
        self,
    ) -> V1LiteralJPrivatePrimalDirectionCUDAHandle:
        handle = self._private_primal_direction_handle
        if handle is None:
            raise RuntimeError(
                "private-primal directions are unavailable on this legacy bundle"
            )
        # Structural access is also the handle's liveness check.  The route
        # tensor never leaves custody through this property.
        handle.descriptor_axis
        return handle

    def borrow_private_primals_for_literal_j_projection(
        self,
    ) -> V1LiteralJPrivatePrimalDirectionCUDALease:
        """Issue one authority-inert CUDA lease for the exact converged primals."""

        return self.private_primal_direction_handle.borrow_for_literal_j_projection()

    def borrow_decision_level_theorem_state_for_cuda_issuer(
        self,
        *,
        request: V1RelaxedKKTDecisionLevelTheoremStateBorrowRequest,
        geometry_uncertainty: V1RelaxedKKTGeometryUncertaintySet,
        literal_j_projection_receipt: V1LiteralJFixedDirectionProjectionReceipt,
    ) -> V1RelaxedKKTDecisionLevelTheoremStateLease:
        """Issue one narrow post-D_W branch lease after exact lineage closure.

        ``source_evidence_hash`` is echoed from ``request`` but deliberately is
        not attested here.  Prefix, suffix, endpoint-outcome, and streaming
        digest hashes remain outside KKT custody and are joined by the CUDA
        issuer after this borrow.
        """

        if type(request) is not V1RelaxedKKTDecisionLevelTheoremStateBorrowRequest:
            raise TypeError("CUDA theorem-state borrow requires an exact request")
        if type(literal_j_projection_receipt) is not (
            V1LiteralJFixedDirectionProjectionReceipt
        ):
            raise TypeError("CUDA theorem-state borrow requires an exact D_W receipt")
        if type(geometry_uncertainty) is not V1RelaxedKKTGeometryUncertaintySet:
            raise TypeError("CUDA theorem-state borrow requires exact uncertainty")
        literal_j_projection_receipt.assert_integrity()
        custody = self._theorem_state_custody
        uncertainty = geometry_uncertainty
        if custody is None:
            _refuse(
                V1RelaxedKKTRefusalCode.CUDA_THEOREM_STATE_LEASE_UNAVAILABLE,
                "canonical theorem-state issuance requires live custody",
            )
        result = self.base_execution.result
        branch_result = (
            result.spectral
            if request.branch is V1RelaxedKKTBranch.SPECTRAL
            else result.diagonal_l1
        )
        branch_identity = (
            self.receipt.spectral
            if request.branch is V1RelaxedKKTBranch.SPECTRAL
            else self.receipt.diagonal_l1
        )
        descriptor_axis = self.private_primal_direction_descriptor_axis
        branch_descriptor = next(
            (
                descriptor
                for descriptor in descriptor_axis.descriptors
                if descriptor.branch == request.branch.value
            ),
            None,
        )
        if branch_descriptor is None:  # pragma: no cover - constructor closes.
            raise RuntimeError("private-primal axis lost the requested branch")
        expected = {
            "problem_hash": result.problem_hash,
            "policy_hash": result.policy_hash,
            "result_hash": result.result_hash,
            "branch_result_hash": branch_result.result_hash,
            "branch_receipt_hash": branch_result.receipt.receipt_hash,
            "device_shell_execution_receipt_hash": self.receipt.receipt_hash,
            "device_shell_execution_design_hash": self.execution_design.design_hash,
            "source_rounding_hash": branch_identity.source_rounding_hash,
            "endpoint_action_axis_hash": branch_identity.endpoint_action_axis_hash,
            "geometry_uncertainty_hash": uncertainty.uncertainty_hash,
            "private_primal_descriptor_axis_hash": (
                descriptor_axis.descriptor_axis_hash
            ),
            "private_primal_branch_descriptor_hash": (
                branch_descriptor.descriptor_hash
            ),
            "literal_j_projection_receipt_hash": (
                literal_j_projection_receipt.receipt_hash
            ),
            "literal_j_projection_direction_axis_hash": (
                literal_j_projection_receipt.direction_axis_hash
            ),
            "literal_j_projection_output_axis_hash": (
                literal_j_projection_receipt.output_axis_hash
            ),
            "literal_j_private_primal_descriptor_axis_hash": (
                literal_j_projection_receipt.private_primal_direction_descriptor_axis_hash
            ),
        }
        if any(getattr(request, name) != value for name, value in expected.items()):
            _refuse(
                V1RelaxedKKTRefusalCode.CUDA_THEOREM_STATE_LEASE_UNAVAILABLE,
                "theorem-state request left its exact problem/branch/D_W lineage",
            )
        if (
            uncertainty.problem_hash != result.problem_hash
            or uncertainty.route_axis_hash != descriptor_axis.route_axis_hash
            or uncertainty.source_geometry_hash
            != self.base_execution.telemetry.source_geometry_hash
            or not uncertainty.source_derived_from_registered_geometry_receipts
            or literal_j_projection_receipt.private_primal_direction_lineage_available
            is not True
            or literal_j_projection_receipt.route_axis_hash
            != descriptor_axis.route_axis_hash
            or literal_j_projection_receipt.route_count != descriptor_axis.route_count
            or literal_j_projection_receipt.chart_identity_hash
            != self.base_execution.telemetry.source_geometry_hash
            or literal_j_projection_receipt.current_mask_hash
            != branch_identity.frozen_ladder.parent_mask_hash
            or literal_j_projection_receipt.direction_ids
            != descriptor_axis.direction_ids
            or uncertainty.literal_j_fixed_primal_direction_axis_hash
            != literal_j_projection_receipt.direction_axis_hash
            or uncertainty.literal_j_fixed_primal_direction_count
            != literal_j_projection_receipt.direction_count
        ):
            _refuse(
                V1RelaxedKKTRefusalCode.CUDA_THEOREM_STATE_LEASE_UNAVAILABLE,
                "literal-J projection is not the exact post-D_W private-primal "
                "projection for this shell chart",
            )
        return custody.borrow_for_cuda_issuer(
            request=request,
            geometry_uncertainty=uncertainty,
            endpoint_counts=branch_identity.frozen_ladder.cumulative_counts,
            rounding_tile_size=(
                self.execution_design.cuda_execution_design.rounding_tile_size
            ),
        )

    @property
    def is_evicted(self) -> bool:
        labels_evicted = (
            self._spectral_label_handle.is_evicted
            and self._diagonal_l1_label_handle.is_evicted
        )
        primal = self._private_primal_direction_handle
        custody = self._theorem_state_custody
        return (
            labels_evicted
            and (primal is None or primal.is_evicted)
            and (custody is None or custody.is_evicted)
        )

    def evict(self) -> None:
        if self.active_device_shell_lease_count:
            _refuse(
                V1RelaxedKKTRefusalCode.CUDA_DEVICE_SHELL_ACTIVE_LEASES,
                "device-shell execution bundle cannot be evicted while a "
                "registered label, private-primal, or theorem-state lease is live",
            )
        _evict_v1_relaxed_kkt_device_shell_owners_atomically(
            spectral_label_handle=self._spectral_label_handle,
            diagonal_l1_label_handle=self._diagonal_l1_label_handle,
            private_primal_direction_handle=self._private_primal_direction_handle,
            theorem_state_custody=self._theorem_state_custody,
        )

    def __reduce__(self) -> NoReturn:
        raise TypeError("device-shell execution bundles are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("device-shell execution bundles are nonserializable")

    def __getstate__(self) -> NoReturn:
        raise TypeError("device-shell execution bundles are nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("device-shell execution bundles cannot be copied")

    def __deepcopy__(self, memo: dict[int, Any]) -> NoReturn:
        del memo
        raise TypeError("device-shell execution bundles cannot be copied")


@dataclass(slots=True)
class _CUDAExecutionCounters:
    factor_forward_scan_count: int = 0
    block_separable_repair_factor_forward_scan_count: int = 0
    factor_adjoint_scan_count: int = 0
    factor_route_visit_count: int = 0
    key_scan_count: int = 0
    key_route_visit_count: int = 0
    replay_factor_element_visit_count: int = 0
    bulk_h2d_tensor_payload_bytes: int = 0
    bulk_d2h_tensor_payload_bytes: int = 0
    forecast_seconds: float = 0.0
    cold_h2d_seconds: float = 0.0
    warm_solver_seconds: float = 0.0
    warm_rounding_seconds: float = 0.0
    warm_replay_seconds: float = 0.0
    receipt_seconds: float = 0.0


def _tensor_payload_nbytes(value: Any) -> int:
    """Return exact dense tensor payload bytes without a device transfer."""

    return int(value.numel()) * int(value.element_size())


def _single_fused_branch_state_forecast_bytes(
    *,
    branch: V1RelaxedKKTBranch,
    route_count: int,
    rank: int,
    block_count: int,
) -> int:
    """Return exact tensor bytes for one private converged branch state."""

    n = _positive_int(route_count, "route_count")
    r = _nonnegative_int(rank, "rank")
    b = _nonnegative_int(block_count, "block_count")
    f64 = np.dtype(np.float64).itemsize
    if branch is V1RelaxedKKTBranch.SPECTRAL:
        # primal, reduced advantage, and route-sized spectral response tail;
        # primal-low and response-low rank vectors; block and work duals.
        return f64 * (3 * n + 2 * r + b + 1)
    if branch is V1RelaxedKKTBranch.DIAGONAL_L1:
        # primal and reduced advantage; two rank vectors; block/work duals;
        # and the scalar diagonal response tail.
        return f64 * (2 * n + 2 * r + b + 2)
    raise TypeError("branch must be an exact relaxed KKT branch")


@dataclass(frozen=True, slots=True)
class _FusedTorchBranchState:
    """Private, nonserializable state frozen at one branch convergence check.

    The tensors are detached clones: subsequent fused iterations for the
    other branch cannot change the state that generated this branch's scalar
    numerical receipt.  This object is an execution seam only.  It carries no
    face-stability, F/Q/R, calibration, oracle, or commit authority.
    """

    branch: V1RelaxedKKTBranch
    primal: Any
    primal_low: Any
    reduced_advantage: Any
    work_dual: Any
    block_duals: Any
    response_low_dual: Any
    response_tail_dual: Any
    numerical_receipt: V1RelaxedKKTBranchReceipt

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("private branch state requires an exact branch")
        if type(self.numerical_receipt) is not V1RelaxedKKTBranchReceipt:
            raise TypeError("private branch state requires an exact receipt")
        if self.numerical_receipt.convergence.branch is not self.branch:
            raise ValueError("private branch state and receipt branches differ")
        if not self.numerical_receipt.convergence.converged:
            raise PermissionError(
                "an unconverged branch cannot freeze private convergence state"
            )
        tensor_names = (
            "primal",
            "primal_low",
            "reduced_advantage",
            "work_dual",
            "block_duals",
            "response_low_dual",
            "response_tail_dual",
        )
        tensor_values = tuple(getattr(self, name) for name in tensor_names)
        for name, value in zip(tensor_names, tensor_values, strict=True):
            if not all(
                hasattr(value, attribute)
                for attribute in (
                    "device",
                    "dtype",
                    "element_size",
                    "numel",
                    "shape",
                )
            ):
                raise TypeError(f"{name} must be one torch tensor")
            if str(value.dtype) != "torch.float64":
                raise TypeError(f"{name} must remain torch.float64")
        devices = {str(value.device) for value in tensor_values}
        if len(devices) != 1:
            raise ValueError("private branch tensors must share one device")
        if self.primal.ndim != 1 or self.reduced_advantage.shape != self.primal.shape:
            raise ValueError("private primal and reduced advantage axes differ")
        if self.primal_low.ndim != 1 or (
            self.response_low_dual.shape != self.primal_low.shape
        ):
            raise ValueError("private low-rank response axes differ")
        if self.work_dual.ndim != 0 or self.block_duals.ndim != 1:
            raise ValueError("private linear-dual shapes are invalid")
        if self.branch is V1RelaxedKKTBranch.SPECTRAL:
            if self.response_tail_dual.shape != self.primal.shape:
                raise ValueError("spectral response tail must use the route axis")
        elif self.response_tail_dual.ndim != 0:
            raise ValueError("diagonal response tail must be scalar")

    @property
    def retained_tensor_bytes(self) -> int:
        return sum(
            _tensor_payload_nbytes(value)
            for value in (
                self.primal,
                self.primal_low,
                self.reduced_advantage,
                self.work_dual,
                self.block_duals,
                self.response_low_dual,
                self.response_tail_dual,
            )
        )

    def __reduce__(self) -> NoReturn:
        raise TypeError("private fused branch states are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("private fused branch states are nonserializable")

    def __getstate__(self) -> NoReturn:
        raise TypeError("private fused branch states are nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("private fused branch states cannot be copied")

    def __deepcopy__(self, memo: dict[int, Any]) -> NoReturn:
        del memo
        raise TypeError("private fused branch states cannot be copied")


def _theorem_state_tensor_storage_invariant(value: Any) -> tuple[object, ...]:
    """Describe torch storage without reading, hashing, or transferring payload."""

    storage = value.untyped_storage()
    return (
        str(value.device),
        str(value.dtype),
        tuple(int(item) for item in value.shape),
        tuple(int(item) for item in value.stride()),
        int(value.storage_offset()),
        int(value.data_ptr()),
        int(storage.data_ptr()),
        int(getattr(value, "_version")),
    )


def _require_v1_relaxed_kkt_theorem_cuda_tensor(
    value: Any,
    *,
    name: str,
    dtype: str,
    ndim: int,
    device: str,
) -> Any:
    """Validate the production tensor contract without route-scale D2H."""

    torch = _torch_module()
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be one torch tensor")
    if getattr(value.device, "type", None) != "cuda" or not torch.cuda.is_available():
        _refuse(
            V1RelaxedKKTRefusalCode.CUDA_UNAVAILABLE,
            f"{name} must remain CUDA-resident; CPU fallback is forbidden",
        )
    if (
        str(value.device) != device
        or str(value.dtype) != dtype
        or value.ndim != ndim
        or not value.is_contiguous()
    ):
        _refuse(
            V1RelaxedKKTRefusalCode.CUDA_THEOREM_STATE_LEASE_UNAVAILABLE,
            f"{name} left its exact contiguous CUDA tensor contract",
        )
    return value


@dataclass(slots=True)
class _V1RelaxedKKTTheoremStateLeaseRecord:
    branch: V1RelaxedKKTBranch
    request: V1RelaxedKKTDecisionLevelTheoremStateBorrowRequest
    canonical_key_lower: Any
    canonical_key_upper: Any
    shell_route_indices: tuple[int, ...]
    shell_route_index_hash: str
    theorem_state_axis_hash: str
    tensor_names: tuple[str, ...]
    tensor_values: tuple[Any, ...]
    tensor_storage_invariants: tuple[tuple[object, ...], ...]


_RELAXED_KKT_THEOREM_STATE_LEASE_CONSTRUCTION_KEY: Final = object()


class _V1RelaxedKKTTheoremStateCustody:
    """Opaque one-use owner for post-D_W source-theorem CUDA state.

    The owner retains the exact converged reduced/dual state plus only the five
    source-problem tensors required by the CUDA issuer.  It never exposes the
    whole branch snapshot.  A successful branch borrow is one-use, tokenized,
    and mutation checked; the sibling branch remains independently issuable.
    """

    __slots__ = (
        "_active_lease_records",
        "_block_eligible_counts",
        "_diagonal_l1_state",
        "_eligible",
        "_evicted",
        "_factor",
        "_issued_branches",
        "_has_ineligible",
        "_lease_lock",
        "_policy",
        "_problem",
        "_residual",
        "_resource",
        "_spectral_state",
        "_torch",
        "_device",
        "_hbm_forecast",
        "_work",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        problem: V1RelaxedKKTProblem,
        policy: V1RelaxedKKTPolicy,
        tensors: _TorchProblem,
        spectral_state: _FusedTorchBranchState,
        diagonal_l1_state: _FusedTorchBranchState,
    ) -> None:
        if type(problem) is not V1RelaxedKKTProblem:
            raise TypeError("theorem custody requires an exact KKT problem")
        if type(policy) is not V1RelaxedKKTPolicy:
            raise TypeError("theorem custody requires an exact KKT policy")
        if type(tensors) is not _TorchProblem:
            raise TypeError("theorem custody requires exact torch problem tensors")
        if (
            type(spectral_state) is not _FusedTorchBranchState
            or type(diagonal_l1_state) is not _FusedTorchBranchState
        ):
            raise TypeError("theorem custody requires exact private branch states")
        axis_errors: list[str] = []
        if spectral_state.branch is not V1RelaxedKKTBranch.SPECTRAL:
            axis_errors.append("spectral_state.branch")
        if diagonal_l1_state.branch is not V1RelaxedKKTBranch.DIAGONAL_L1:
            axis_errors.append("diagonal_l1_state.branch")
        source_device = getattr(tensors.factor, "device", None)
        for name, value in (
            ("spectral_state.primal", spectral_state.primal),
            ("diagonal_l1_state.primal", diagonal_l1_state.primal),
        ):
            if int(value.numel()) != problem.route_count:
                axis_errors.append(
                    f"{name}.numel={int(value.numel())} "
                    f"route_count={problem.route_count}"
                )
            if value.device != source_device:
                axis_errors.append(
                    f"{name}.device={value.device} source.device={source_device}"
                )
        if int(tensors.factor.shape[0]) != problem.rank:
            axis_errors.append(
                f"factor.rank={int(tensors.factor.shape[0])} "
                f"problem.rank={problem.rank}"
            )
        if int(tensors.factor.shape[1]) != problem.route_count:
            axis_errors.append(
                f"factor.route_count={int(tensors.factor.shape[1])} "
                f"problem.route_count={problem.route_count}"
            )
        for name, value in (
            ("resource", tensors.resource),
            ("work", tensors.work),
            ("residual", tensors.residual),
            ("eligible", tensors.eligible),
        ):
            if tuple(value.shape) != (problem.route_count,):
                axis_errors.append(
                    f"{name}.shape={tuple(value.shape)} "
                    f"expected=({problem.route_count},)"
                )
        for name, value in (
            ("spectral_state.primal_low", spectral_state.primal_low),
            ("diagonal_l1_state.primal_low", diagonal_l1_state.primal_low),
        ):
            if int(value.numel()) != problem.rank:
                axis_errors.append(
                    f"{name}.numel={int(value.numel())} " f"problem.rank={problem.rank}"
                )
        if len(tensors.block_eligible_counts) != len(problem.block_capacities):
            axis_errors.append(
                "block_eligible_counts="
                f"{len(tensors.block_eligible_counts)} "
                f"block_capacities={len(problem.block_capacities)}"
            )
        if axis_errors:
            raise ValueError(
                "theorem custody inputs do not share the problem axis: "
                + "; ".join(axis_errors)
            )
        device = str(source_device)
        for name, value, dtype, ndim in (
            ("low_rank_factor", tensors.factor, "torch.float64", 2),
            ("resource", tensors.resource, "torch.float64", 1),
            ("corrected_work", tensors.work, "torch.float64", 1),
            ("residual_diagonal", tensors.residual, "torch.float64", 1),
            ("eligible_mask", tensors.eligible, "torch.bool", 1),
        ):
            _require_v1_relaxed_kkt_theorem_cuda_tensor(
                value,
                name=name,
                dtype=dtype,
                ndim=ndim,
                device=device,
            )
        self._problem: V1RelaxedKKTProblem | None = problem
        self._policy: V1RelaxedKKTPolicy | None = policy
        self._torch: Any | None = tensors.torch
        self._device: Any | None = tensors.device
        self._factor: Any | None = tensors.factor
        self._resource: Any | None = tensors.resource
        self._work: Any | None = tensors.work
        self._residual: Any | None = tensors.residual
        self._eligible: Any | None = tensors.eligible
        self._block_eligible_counts = tuple(tensors.block_eligible_counts)
        self._has_ineligible = bool(tensors.has_ineligible)
        self._spectral_state: _FusedTorchBranchState | None = spectral_state
        self._diagonal_l1_state: _FusedTorchBranchState | None = diagonal_l1_state
        self._lease_lock = RLock()
        self._active_lease_records: dict[
            object, _V1RelaxedKKTTheoremStateLeaseRecord
        ] = {}
        self._issued_branches: set[V1RelaxedKKTBranch] = set()
        self._evicted = False

    def _require_live_locked(
        self,
    ) -> tuple[
        V1RelaxedKKTProblem,
        V1RelaxedKKTPolicy,
        _FusedTorchBranchState,
        _FusedTorchBranchState,
    ]:
        problem = self._problem
        policy = self._policy
        spectral = self._spectral_state
        diagonal = self._diagonal_l1_state
        if (
            self._evicted
            or problem is None
            or policy is None
            or spectral is None
            or diagonal is None
            or self._factor is None
            or self._resource is None
            or self._work is None
            or self._residual is None
            or self._eligible is None
        ):
            _refuse(
                V1RelaxedKKTRefusalCode.CUDA_THEOREM_STATE_LEASE_UNAVAILABLE,
                "private source-theorem state custody was evicted",
            )
        return problem, policy, spectral, diagonal

    def _require_live(
        self,
    ) -> tuple[
        V1RelaxedKKTProblem,
        V1RelaxedKKTPolicy,
        _FusedTorchBranchState,
        _FusedTorchBranchState,
    ]:
        with self._lease_lock:
            return self._require_live_locked()

    @property
    def is_evicted(self) -> bool:
        with self._lease_lock:
            return bool(self._evicted)

    @property
    def active_lease_count(self) -> int:
        with self._lease_lock:
            return len(self._active_lease_records)

    @property
    def retained_tensor_bytes(self) -> int:
        with self._lease_lock:
            _, _, spectral, diagonal = self._require_live_locked()
            return (
                spectral.retained_tensor_bytes
                + diagonal.retained_tensor_bytes
                + sum(
                    _tensor_payload_nbytes(value)
                    for value in (
                        self._factor,
                        self._resource,
                        self._work,
                        self._residual,
                        self._eligible,
                    )
                )
                + sum(
                    _tensor_payload_nbytes(record.canonical_key_lower)
                    + _tensor_payload_nbytes(record.canonical_key_upper)
                    for record in self._active_lease_records.values()
                )
            )

    def assert_execution_lineage(
        self,
        *,
        result: V1RelaxedKKTResult,
        route_count: int,
    ) -> None:
        """Validate state/receipt identity without exposing either state."""

        if type(result) is not V1RelaxedKKTResult:
            raise TypeError("theorem custody lineage requires an exact result")
        with self._lease_lock:
            problem, policy, spectral, diagonal = self._require_live_locked()
            if (
                problem.problem_hash != result.problem_hash
                or policy.policy_hash != result.policy_hash
                or route_count != problem.route_count
            ):
                raise ValueError("source-theorem problem lineage does not close")
            for state, branch_result in (
                (spectral, result.spectral),
                (diagonal, result.diagonal_l1),
            ):
                if (
                    state.numerical_receipt is not branch_result.receipt
                    or int(state.primal.numel()) != route_count
                    or state.numerical_receipt.problem_hash != result.problem_hash
                    or state.numerical_receipt.policy_hash != result.policy_hash
                ):
                    raise ValueError(
                        "source-theorem custody left its exact solved branch lineage"
                    )

    def _state_for_registered_source_theorem(
        self,
        branch: V1RelaxedKKTBranch,
    ) -> NoReturn:
        """Refuse the superseded whole-state accessor, including internally."""

        if type(branch) is not V1RelaxedKKTBranch:
            raise TypeError("source theorem requires an exact branch")
        _refuse(
            V1RelaxedKKTRefusalCode.CUDA_THEOREM_STATE_LEASE_UNAVAILABLE,
            "whole private theorem state is unavailable; use the narrow "
            "decision-level theorem-state lease",
        )

    def _branch_state_locked(
        self,
        branch: V1RelaxedKKTBranch,
    ) -> _FusedTorchBranchState:
        _, _, spectral, diagonal = self._require_live_locked()
        return spectral if branch is V1RelaxedKKTBranch.SPECTRAL else diagonal

    def _rounding_tensors_locked(self) -> _TorchProblem:
        self._require_live_locked()
        return _TorchProblem(
            torch=self._torch,
            device=self._device,
            resource=self._resource,
            work=self._work,
            factor=self._factor,
            residual=self._residual,
            sqrt_residual=None,
            eligible=self._eligible,
            objective=None,
            block_eligible_counts=self._block_eligible_counts,
            block_bounds=None,
            has_ineligible=self._has_ineligible,
        )

    def borrow_for_cuda_issuer(
        self,
        *,
        request: V1RelaxedKKTDecisionLevelTheoremStateBorrowRequest,
        geometry_uncertainty: V1RelaxedKKTGeometryUncertaintySet,
        endpoint_counts: tuple[int, ...],
        rounding_tile_size: int,
    ) -> V1RelaxedKKTDecisionLevelTheoremStateLease:
        """Issue one branch's KKT-owned CUDA tensors without cloning them."""

        if type(request) is not V1RelaxedKKTDecisionLevelTheoremStateBorrowRequest:
            raise TypeError("theorem-state borrow requires an exact request")
        if type(geometry_uncertainty) is not V1RelaxedKKTGeometryUncertaintySet:
            raise TypeError("theorem-state borrow requires exact uncertainty")
        counts = tuple(
            _positive_int(value, "endpoint_count") for value in endpoint_counts
        )
        if not counts or tuple(sorted(set(counts))) != counts:
            raise ValueError("endpoint counts must be nonempty, unique, and increasing")
        tile_size = _positive_int(rounding_tile_size, "rounding_tile_size")
        with self._lease_lock:
            problem, policy, _, _ = self._require_live_locked()
            branch = request.branch
            if branch in self._issued_branches:
                _refuse(
                    V1RelaxedKKTRefusalCode.CUDA_THEOREM_STATE_LEASE_UNAVAILABLE,
                    "the selected branch theorem-state path was already consumed",
                )
            state = self._branch_state_locked(branch)
            tensors = self._rounding_tensors_locked()
            hbm_forecast = _forecast_v1_relaxed_kkt_theorem_lease_hbm_from_device(
                problem=problem,
                request=request,
                endpoint_counts=counts,
                route_tile_size=tile_size,
                device=self._device,
            )
            if not hbm_forecast.admitted:
                _refuse(
                    V1RelaxedKKTRefusalCode.CUDA_HBM_FORECAST_EXCEEDED,
                    "theorem-lease forecast peak exceeds the fixed 85-percent "
                    "HBM ceiling before canonical key allocation",
                )
            try:
                lower, upper = _build_v1_relaxed_kkt_canonical_key_intervals_cuda(
                    problem=problem,
                    state=state,
                    tensors=tensors,
                    geometry_uncertainty=geometry_uncertainty,
                )
            except Exception as error:
                torch = _torch_module()
                oom_type = getattr(torch, "OutOfMemoryError", ())
                if (oom_type and isinstance(error, oom_type)) or (
                    "out of memory" in str(error).lower()
                ):
                    _refuse(V1RelaxedKKTRefusalCode.CUDA_OUT_OF_MEMORY, str(error))
                raise
            try:
                shell = _derive_v1_relaxed_kkt_bounded_theorem_shell_cuda(
                    problem=problem,
                    policy=policy,
                    state=state,
                    tensors=tensors,
                    canonical_key_lower=lower,
                    canonical_key_upper=upper,
                    endpoint_counts=counts,
                    maximum_shell_routes=request.maximum_shell_routes,
                    tile_size=tile_size,
                )
                shell_hash = RemovedRouteIndexSet.from_indices(
                    np.asarray(shell, dtype=np.int64),
                    route_count=problem.route_count,
                ).index_hash
                axis_hash = content_hash(
                    {
                        "schema": (
                            "partition_free_v1_relaxed_kkt_decision_level_"
                            "theorem_state_axis_v1"
                        ),
                        "request_hash": request.request_hash,
                        "problem_rank": problem.rank,
                        "route_axis_hash": problem.route_axis_hash,
                        "route_count": problem.route_count,
                        "shell_route_index_hash": shell_hash,
                        "theorem_lease_hbm_forecast_hash": (hbm_forecast.forecast_hash),
                        "canonical_key_interval_equation": (
                            "(-z,-reduced_advantage,corrected_work/resource)_v1"
                        ),
                    },
                    prefix=(
                        "partition-free-v1-relaxed-kkt-decision-level-"
                        "theorem-state-axis-"
                    ),
                )
                tensor_names = (
                    "canonical_key_lower",
                    "canonical_key_upper",
                    "low_rank_factor",
                    "resource",
                    "corrected_work",
                    "residual_diagonal",
                    "eligible_mask",
                    "primal",
                    "primal_low",
                    "reduced_advantage",
                    "work_dual",
                    "block_duals",
                    "response_low_dual",
                    "response_tail_dual",
                )
                tensor_values = (
                    lower,
                    upper,
                    self._factor,
                    self._resource,
                    self._work,
                    self._residual,
                    self._eligible,
                    state.primal,
                    state.primal_low,
                    state.reduced_advantage,
                    state.work_dual,
                    state.block_duals,
                    state.response_low_dual,
                    state.response_tail_dual,
                )
                invariants = tuple(
                    _theorem_state_tensor_storage_invariant(value)
                    for value in tensor_values
                )
                token = object()
                record = _V1RelaxedKKTTheoremStateLeaseRecord(
                    branch=branch,
                    request=request,
                    canonical_key_lower=lower,
                    canonical_key_upper=upper,
                    shell_route_indices=shell,
                    shell_route_index_hash=shell_hash,
                    theorem_state_axis_hash=axis_hash,
                    tensor_names=tensor_names,
                    tensor_values=tensor_values,
                    tensor_storage_invariants=invariants,
                )
                lease = V1RelaxedKKTDecisionLevelTheoremStateLease(
                    _construction_key=(
                        _RELAXED_KKT_THEOREM_STATE_LEASE_CONSTRUCTION_KEY
                    ),
                    custody=self,
                    token=token,
                    request=request,
                    route_axis_hash=problem.route_axis_hash,
                    route_count=problem.route_count,
                    rank=problem.rank,
                    device=str(self._device),
                    shell_route_indices=shell,
                    shell_route_index_hash=shell_hash,
                    theorem_state_axis_hash=axis_hash,
                    hbm_forecast=hbm_forecast,
                )
            except BaseException as error:
                del lower, upper
                torch = _torch_module()
                oom_type = getattr(torch, "OutOfMemoryError", ())
                if (oom_type and isinstance(error, oom_type)) or "out of memory" in str(
                    error
                ).lower():
                    _refuse(V1RelaxedKKTRefusalCode.CUDA_OUT_OF_MEMORY, str(error))
                raise
            try:
                self._active_lease_records[token] = record
                self._issued_branches.add(branch)
            except BaseException:
                self._active_lease_records.pop(token, None)
                raise
            return lease

    def _record_for_live_lease_locked(
        self,
        *,
        token: object,
        expected_branch: V1RelaxedKKTBranch,
    ) -> _V1RelaxedKKTTheoremStateLeaseRecord:
        self._require_live_locked()
        record = self._active_lease_records.get(token)
        if record is None or record.branch is not expected_branch:
            _refuse(
                V1RelaxedKKTRefusalCode.CUDA_THEOREM_STATE_LEASE_UNAVAILABLE,
                "decision-level theorem-state lease is closed or misbound",
            )
        for value, expected in zip(
            record.tensor_values,
            record.tensor_storage_invariants,
            strict=True,
        ):
            if _theorem_state_tensor_storage_invariant(value) != expected:
                _refuse(
                    V1RelaxedKKTRefusalCode.CUDA_THEOREM_STATE_MUTATED,
                    "private theorem-state storage or tensor version changed",
                )
        return record

    def _tensor_for_live_lease(
        self,
        *,
        token: object,
        expected_branch: V1RelaxedKKTBranch,
        tensor_name: str,
    ) -> Any:
        with self._lease_lock:
            record = self._record_for_live_lease_locked(
                token=token,
                expected_branch=expected_branch,
            )
            try:
                index = record.tensor_names.index(tensor_name)
            except ValueError as error:  # pragma: no cover - internal contract.
                raise RuntimeError("unknown theorem-state tensor") from error
            return record.tensor_values[index]

    def _assert_live_lease(
        self,
        *,
        token: object,
        expected_branch: V1RelaxedKKTBranch,
    ) -> None:
        with self._lease_lock:
            self._record_for_live_lease_locked(
                token=token,
                expected_branch=expected_branch,
            )

    def _close_live_lease(
        self,
        *,
        token: object,
        expected_branch: V1RelaxedKKTBranch,
    ) -> None:
        with self._lease_lock:
            try:
                self._record_for_live_lease_locked(
                    token=token,
                    expected_branch=expected_branch,
                )
            finally:
                # A mutation refusal must still release the token and derived
                # interval tensors so aggregate eviction cannot deadlock.
                self._active_lease_records.pop(token, None)

    def evict(self) -> None:
        with self._lease_lock:
            self._require_live_locked()
            if self._active_lease_records:
                _refuse(
                    V1RelaxedKKTRefusalCode.CUDA_DEVICE_SHELL_ACTIVE_LEASES,
                    "private theorem state cannot be evicted while a lease is live",
                )
            self._spectral_state = None
            self._diagonal_l1_state = None
            self._factor = None
            self._resource = None
            self._work = None
            self._residual = None
            self._eligible = None
            self._torch = None
            self._device = None
            self._problem = None
            self._policy = None
            self._block_eligible_counts = ()
            self._has_ineligible = False
            self._evicted = True

    def __reduce__(self) -> NoReturn:
        raise TypeError("private theorem-state custody is nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("private theorem-state custody is nonserializable")

    def __getstate__(self) -> NoReturn:
        raise TypeError("private theorem-state custody is nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("private theorem-state custody cannot be copied")

    def __deepcopy__(self, memo: dict[int, Any]) -> NoReturn:
        del memo
        raise TypeError("private theorem-state custody cannot be copied")


class V1RelaxedKKTDecisionLevelTheoremStateLease:
    """Narrow CUDA-only projection of one converged KKT branch.

    Source tensors are borrowed by identity.  The lease provides no whole
    private-state object, no serialization/copy surface, and no CPU or clone
    operation.  Every tensor access revalidates the complete source storage
    invariant, including the hidden dual state used to justify the keys.
    """

    __slots__ = (
        "_branch",
        "_closed",
        "_custody",
        "_device",
        "_hbm_forecast",
        "_rank",
        "_request",
        "_route_axis_hash",
        "_route_count",
        "_shell_route_index_hash",
        "_shell_route_indices",
        "_theorem_state_axis_hash",
        "_token",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        _construction_key: object,
        custody: _V1RelaxedKKTTheoremStateCustody,
        token: object,
        request: V1RelaxedKKTDecisionLevelTheoremStateBorrowRequest,
        route_axis_hash: str,
        route_count: int,
        rank: int,
        device: str,
        shell_route_indices: tuple[int, ...],
        shell_route_index_hash: str,
        theorem_state_axis_hash: str,
        hbm_forecast: V1RelaxedKKTDecisionLevelTheoremLeaseHBMForecast,
    ) -> None:
        if _construction_key is not (_RELAXED_KKT_THEOREM_STATE_LEASE_CONSTRUCTION_KEY):
            raise TypeError("theorem-state leases must be issued by KKT custody")
        if type(custody) is not _V1RelaxedKKTTheoremStateCustody:
            raise TypeError("theorem-state lease requires exact custody")
        if type(request) is not V1RelaxedKKTDecisionLevelTheoremStateBorrowRequest:
            raise TypeError("theorem-state lease requires an exact request")
        if type(hbm_forecast) is not (V1RelaxedKKTDecisionLevelTheoremLeaseHBMForecast):
            raise TypeError("theorem-state lease requires an exact HBM forecast")
        if (
            not hbm_forecast.admitted
            or hbm_forecast.problem_hash != request.problem_hash
            or hbm_forecast.request_hash != request.request_hash
            or hbm_forecast.route_count != route_count
            or hbm_forecast.rank != rank
            or hbm_forecast.device != device
        ):
            raise ValueError("theorem-state lease HBM admission does not close")
        self._custody: _V1RelaxedKKTTheoremStateCustody | None = custody
        self._token: object | None = token
        self._request = request
        self._branch = request.branch
        self._route_axis_hash = _nonempty(route_axis_hash, "route_axis_hash")
        self._route_count = _positive_int(route_count, "route_count")
        self._rank = _nonnegative_int(rank, "rank")
        self._device = _nonempty(device, "device")
        self._hbm_forecast = hbm_forecast
        self._shell_route_indices = tuple(shell_route_indices)
        self._shell_route_index_hash = _nonempty(
            shell_route_index_hash,
            "shell_route_index_hash",
        )
        self._theorem_state_axis_hash = _nonempty(
            theorem_state_axis_hash,
            "theorem_state_axis_hash",
        )
        self._closed = False

    def _require_live(self) -> tuple[_V1RelaxedKKTTheoremStateCustody, object]:
        custody = self._custody
        token = self._token
        if self._closed or custody is None or token is None:
            _refuse(
                V1RelaxedKKTRefusalCode.CUDA_THEOREM_STATE_LEASE_UNAVAILABLE,
                "decision-level theorem-state lease is closed",
            )
        custody._assert_live_lease(token=token, expected_branch=self._branch)
        return custody, token

    def _tensor(self, name: str) -> Any:
        custody, token = self._require_live()
        return custody._tensor_for_live_lease(
            token=token,
            expected_branch=self._branch,
            tensor_name=name,
        )

    @property
    def request(self) -> V1RelaxedKKTDecisionLevelTheoremStateBorrowRequest:
        self._require_live()
        return self._request

    @property
    def branch(self) -> V1RelaxedKKTBranch:
        self._require_live()
        return self._branch

    @property
    def route_axis_hash(self) -> str:
        self._require_live()
        return self._route_axis_hash

    @property
    def route_count(self) -> int:
        self._require_live()
        return self._route_count

    @property
    def rank(self) -> int:
        self._require_live()
        return self._rank

    @property
    def device(self) -> str:
        self._require_live()
        return self._device

    @property
    def hbm_forecast(
        self,
    ) -> V1RelaxedKKTDecisionLevelTheoremLeaseHBMForecast:
        self._require_live()
        return self._hbm_forecast

    @property
    def shell_route_indices(self) -> tuple[int, ...]:
        self._require_live()
        return self._shell_route_indices

    @property
    def shell_route_index_hash(self) -> str:
        self._require_live()
        return self._shell_route_index_hash

    @property
    def shell_count(self) -> int:
        self._require_live()
        return len(self._shell_route_indices)

    @property
    def theorem_state_axis_hash(self) -> str:
        self._require_live()
        return self._theorem_state_axis_hash

    @property
    def descriptor_axis_hash(self) -> str:
        """Compatibility name for the KKT-owned theorem-state axis only."""

        return self.theorem_state_axis_hash

    @property
    def source_evidence_hash(self) -> str:
        self._require_live()
        return self._request.source_evidence_hash

    @property
    def canonical_key_lower(self) -> Any:
        return self._tensor("canonical_key_lower")

    @property
    def canonical_key_upper(self) -> Any:
        return self._tensor("canonical_key_upper")

    @property
    def low_rank_factor(self) -> Any:
        return self._tensor("low_rank_factor")

    @property
    def resource(self) -> Any:
        return self._tensor("resource")

    @property
    def corrected_work(self) -> Any:
        return self._tensor("corrected_work")

    @property
    def residual_diagonal(self) -> Any:
        return self._tensor("residual_diagonal")

    @property
    def eligible_mask(self) -> Any:
        return self._tensor("eligible_mask")

    @property
    def is_closed(self) -> bool:
        return bool(self._closed)

    def close(self) -> None:
        if self._closed:
            return
        custody = self._custody
        token = self._token
        if custody is None or token is None:  # pragma: no cover - defensive.
            _refuse(
                V1RelaxedKKTRefusalCode.CUDA_THEOREM_STATE_LEASE_UNAVAILABLE,
                "decision-level theorem-state lease lost its owner",
            )
        try:
            custody._close_live_lease(
                token=token,
                expected_branch=self._branch,
            )
        finally:
            self._custody = None
            self._token = None
            self._closed = True

    def __enter__(self) -> V1RelaxedKKTDecisionLevelTheoremStateLease:
        self._require_live()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> bool:
        del exc_type, exc_value, traceback
        self.close()
        return False

    def __reduce__(self) -> NoReturn:
        raise TypeError("decision-level theorem-state leases are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("decision-level theorem-state leases are nonserializable")

    def __getstate__(self) -> NoReturn:
        raise TypeError("decision-level theorem-state leases are nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("decision-level theorem-state leases cannot be copied")

    def __deepcopy__(self, memo: dict[int, Any]) -> NoReturn:
        del memo
        raise TypeError("decision-level theorem-state leases cannot be copied")


def _capture_fused_torch_branch_state(
    *,
    branch: V1RelaxedKKTBranch,
    primal: Any,
    primal_low: Any,
    reduced_advantage: Any,
    work_dual: Any,
    block_duals: Any,
    response_low_dual: Any,
    response_tail_dual: Any,
    numerical_receipt: V1RelaxedKKTBranchReceipt,
) -> _FusedTorchBranchState:
    """Atomically detach and clone every tensor behind one scalar receipt."""

    if not numerical_receipt.convergence.converged:
        raise PermissionError("private state capture requires convergence")
    # Construct the result only after every clone succeeds.  A failed
    # allocation leaves no partially authoritative snapshot reachable.
    cloned = tuple(
        value.detach().clone()
        for value in (
            primal,
            primal_low,
            reduced_advantage,
            work_dual,
            block_duals,
            response_low_dual,
            response_tail_dual,
        )
    )
    state = _FusedTorchBranchState(
        branch=branch,
        primal=cloned[0],
        primal_low=cloned[1],
        reduced_advantage=cloned[2],
        work_dual=cloned[3],
        block_duals=cloned[4],
        response_low_dual=cloned[5],
        response_tail_dual=cloned[6],
        numerical_receipt=numerical_receipt,
    )
    expected = _single_fused_branch_state_forecast_bytes(
        branch=branch,
        route_count=int(state.primal.numel()),
        rank=int(state.primal_low.numel()),
        block_count=int(state.block_duals.numel()),
    )
    if state.retained_tensor_bytes != expected:
        raise RuntimeError(
            "private fused branch-state tensor bytes differ from their exact "
            "forecast formula"
        )
    return state


@dataclass(frozen=True, slots=True)
class _FusedTorchSolveOutcome:
    """Backward-unpackable fused output plus private converged snapshots."""

    spectral_primal: Any
    spectral_reduced_advantage: Any
    spectral_receipt: V1RelaxedKKTBranchReceipt
    diagonal_primal: Any
    diagonal_reduced_advantage: Any
    diagonal_receipt: V1RelaxedKKTBranchReceipt
    spectral_state: _FusedTorchBranchState | None
    diagonal_state: _FusedTorchBranchState | None

    def __post_init__(self) -> None:
        pairs = (
            (
                V1RelaxedKKTBranch.SPECTRAL,
                self.spectral_primal,
                self.spectral_reduced_advantage,
                self.spectral_receipt,
                self.spectral_state,
            ),
            (
                V1RelaxedKKTBranch.DIAGONAL_L1,
                self.diagonal_primal,
                self.diagonal_reduced_advantage,
                self.diagonal_receipt,
                self.diagonal_state,
            ),
        )
        for branch, primal, reduced, receipt, state in pairs:
            if type(receipt) is not V1RelaxedKKTBranchReceipt:
                raise TypeError("fused outcome requires exact branch receipts")
            if receipt.convergence.branch is not branch:
                raise ValueError("fused outcome branch receipt is mislabelled")
            if (state is not None) is not receipt.convergence.converged:
                raise ValueError(
                    "private branch-state presence differs from convergence"
                )
            if state is not None:
                if type(state) is not _FusedTorchBranchState:
                    raise TypeError("fused outcome requires exact private states")
                if (
                    state.branch is not branch
                    or state.numerical_receipt is not receipt
                    or state.primal is not primal
                    or state.reduced_advantage is not reduced
                ):
                    raise ValueError(
                        "fused legacy output and private branch state diverged"
                    )

    def __iter__(self) -> Any:
        # Preserve the historical six-value private unpacking contract.
        yield self.spectral_primal
        yield self.spectral_reduced_advantage
        yield self.spectral_receipt
        yield self.diagonal_primal
        yield self.diagonal_reduced_advantage
        yield self.diagonal_receipt

    def __len__(self) -> int:
        return 6

    def __getitem__(self, index: int | slice) -> Any:
        return tuple(iter(self))[index]

    def __reduce__(self) -> NoReturn:
        raise TypeError("private fused solver outcomes are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("private fused solver outcomes are nonserializable")

    def __getstate__(self) -> NoReturn:
        raise TypeError("private fused solver outcomes are nonserializable")


def _private_primal_direction_id(
    *,
    branch: V1RelaxedKKTBranch,
    rank: int,
    problem_hash: str,
    branch_result_hash: str,
    execution_receipt_hash: str,
) -> str:
    """Return a readable, rank-bearing structural direction identity."""

    suffix = content_hash(
        {
            "schema": "partition_free_v1_relaxed_kkt_private_primal_direction_id_v1",
            "branch": branch,
            "rank": rank,
            "problem_hash": problem_hash,
            "branch_result_hash": branch_result_hash,
            "execution_receipt_hash": execution_receipt_hash,
        },
        prefix="partition-free-v1-relaxed-kkt-private-primal-direction-id-",
    )
    return f"relaxed-kkt-private-primal:{branch.value}:rank={rank}:{suffix}"


def _build_v1_relaxed_kkt_private_primal_direction_handle(
    *,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    base_execution: V1RelaxedKKTCUDAExecutionBundle,
    execution_design: V1RelaxedKKTDeviceShellExecutionDesign,
    execution_receipt: V1RelaxedKKTDeviceShellExecutionReceipt,
    spectral_state: _FusedTorchBranchState,
    diagonal_l1_state: _FusedTorchBranchState,
    private_primal_directions: Any,
) -> V1LiteralJPrivatePrimalDirectionCUDAHandle:
    """Transfer exact converged primals into CUDA-only projection custody.

    Descriptor hashes bind every durable coordinate while the route-scale
    tensors are stacked and retained only on their existing CUDA device.  No
    route tensor is copied to the host or hashed.  All other convergence-state
    tensors remain temporary inputs to witness issuance and leave scope after
    this transfer.
    """

    if type(problem) is not V1RelaxedKKTProblem:
        raise TypeError("private-primal custody requires an exact problem")
    if type(policy) is not V1RelaxedKKTPolicy:
        raise TypeError("private-primal custody requires an exact policy")
    if type(base_execution) is not V1RelaxedKKTCUDAExecutionBundle:
        raise TypeError("private-primal custody requires exact base execution")
    if type(execution_design) is not V1RelaxedKKTDeviceShellExecutionDesign:
        raise TypeError("private-primal custody requires exact shell design")
    if type(execution_receipt) is not V1RelaxedKKTDeviceShellExecutionReceipt:
        raise TypeError("private-primal custody requires exact shell receipt")
    states = (spectral_state, diagonal_l1_state)
    if any(type(state) is not _FusedTorchBranchState for state in states):
        raise TypeError("private-primal custody requires two exact branch states")
    result = base_execution.result
    telemetry = base_execution.telemetry
    forecast = base_execution.forecast
    if (
        result.problem_hash != problem.problem_hash
        or result.policy_hash != policy.policy_hash
        or execution_receipt.problem_hash != problem.problem_hash
        or execution_receipt.policy_hash != policy.policy_hash
        or execution_receipt.base_result_hash != result.result_hash
        or execution_receipt.receipt_hash == ""
        or telemetry.route_axis_hash != problem.route_axis_hash
        or telemetry.route_count != problem.route_count
        or telemetry.rank != problem.rank
        or forecast.problem_hash != problem.problem_hash
        or forecast.route_count != problem.route_count
        or forecast.rank != problem.rank
        or execution_receipt.device_shell_execution_design_hash
        != execution_design.design_hash
        or execution_receipt.cuda_execution_design_hash
        != execution_design.cuda_execution_design.design_hash
    ):
        raise ValueError("private-primal source execution coordinates do not close")

    branch_results = (result.spectral, result.diagonal_l1)
    expected_branches = (
        V1RelaxedKKTBranch.SPECTRAL,
        V1RelaxedKKTBranch.DIAGONAL_L1,
    )
    descriptors: list[V1LiteralJPrivatePrimalDirectionDescriptor] = []
    for branch, state, branch_result in zip(
        expected_branches,
        states,
        branch_results,
        strict=True,
    ):
        if (
            state.branch is not branch
            or state.numerical_receipt is not branch_result.receipt
            or int(state.primal.numel()) != problem.route_count
            or state.numerical_receipt.problem_hash != problem.problem_hash
            or state.numerical_receipt.policy_hash != policy.policy_hash
            or not state.numerical_receipt.convergence.converged
        ):
            raise ValueError(
                "private-primal state does not bind its exact converged branch"
            )
        descriptors.append(
            V1LiteralJPrivatePrimalDirectionDescriptor(
                direction_id=_private_primal_direction_id(
                    branch=branch,
                    rank=problem.rank,
                    problem_hash=problem.problem_hash,
                    branch_result_hash=branch_result.result_hash,
                    execution_receipt_hash=execution_receipt.receipt_hash,
                ),
                branch=branch.value,  # type: ignore[arg-type]
                source_problem_hash=problem.problem_hash,
                source_policy_hash=policy.policy_hash,
                source_result_hash=result.result_hash,
                source_branch_result_hash=branch_result.result_hash,
                source_branch_receipt_hash=branch_result.receipt.receipt_hash,
                source_execution_receipt_hash=execution_receipt.receipt_hash,
                source_execution_design_hash=execution_design.design_hash,
                route_axis_hash=problem.route_axis_hash,
                route_count=problem.route_count,
            )
        )
    descriptor_axis = V1LiteralJPrivatePrimalDirectionDescriptorAxis.from_descriptors(
        descriptors[0],
        descriptors[1],
    )
    return V1LiteralJPrivatePrimalDirectionCUDAHandle(
        descriptor_axis=descriptor_axis,
        directions=private_primal_directions,
    )


def _stack_v1_relaxed_kkt_private_primals(
    *,
    spectral_state: _FusedTorchBranchState,
    diagonal_l1_state: _FusedTorchBranchState,
) -> Any:
    """Allocate the sole contiguous two-direction tensor before HBM telemetry."""

    if (
        type(spectral_state) is not _FusedTorchBranchState
        or type(diagonal_l1_state) is not _FusedTorchBranchState
        or spectral_state.branch is not V1RelaxedKKTBranch.SPECTRAL
        or diagonal_l1_state.branch is not V1RelaxedKKTBranch.DIAGONAL_L1
        or spectral_state.primal.shape != diagonal_l1_state.primal.shape
        or str(spectral_state.primal.device) != str(diagonal_l1_state.primal.device)
    ):
        raise ValueError("private-primal stack sources do not share one branch axis")
    torch = _torch_module()
    return torch.stack(
        (spectral_state.primal, diagonal_l1_state.primal),
        dim=0,
    ).detach()


def _validate_fused_branch_state_forecast(
    *,
    problem: V1RelaxedKKTProblem,
    outcome: _FusedTorchSolveOutcome,
    forecast: V1RelaxedKKTCUDAHBMForecast,
    private_primal_custody_required: bool = False,
) -> None:
    """Fail closed unless private retained state matches pre-outcome HBM."""

    if type(outcome) is not _FusedTorchSolveOutcome:
        raise TypeError("state forecast validation requires an exact outcome")
    if type(forecast) is not V1RelaxedKKTCUDAHBMForecast:
        raise TypeError("state forecast validation requires an exact forecast")
    if type(private_primal_custody_required) is not bool:
        raise TypeError("private_primal_custody_required must be an exact bool")
    if (
        forecast.problem_hash != problem.problem_hash
        or forecast.route_count != problem.route_count
        or forecast.rank != problem.rank
    ):
        _refuse(
            V1RelaxedKKTRefusalCode.CUDA_WORKSPACE_EXCEEDED,
            "private branch-state forecast belongs to another problem",
        )
    f64 = np.dtype(np.float64).itemsize
    snapshot_forecast = _fused_branch_state_snapshot_forecast_bytes(
        route_count=problem.route_count,
        rank=problem.rank,
        block_count=len(problem.block_capacities),
    )
    expected_solver_state = (
        _CUDA_FUSED_ACTIVE_ROUTE_VECTOR_COUNT * problem.route_count * f64
        + snapshot_forecast
    )
    expected_private_primal = (
        2 * problem.route_count * f64 if private_primal_custody_required else 0
    )
    if forecast.solver_state_bytes != expected_solver_state:
        _refuse(
            V1RelaxedKKTRefusalCode.CUDA_WORKSPACE_EXCEEDED,
            "pre-outcome HBM forecast does not exactly reserve the private "
            "branch-state snapshot",
        )
    if forecast.private_primal_direction_bytes != expected_private_primal:
        _refuse(
            V1RelaxedKKTRefusalCode.CUDA_WORKSPACE_EXCEEDED,
            "pre-outcome HBM forecast does not exactly reserve the retained "
            "two-direction private-primal stack",
        )
    for branch, receipt, state in (
        (
            V1RelaxedKKTBranch.SPECTRAL,
            outcome.spectral_receipt,
            outcome.spectral_state,
        ),
        (
            V1RelaxedKKTBranch.DIAGONAL_L1,
            outcome.diagonal_receipt,
            outcome.diagonal_state,
        ),
    ):
        if (state is not None) is not receipt.convergence.converged:
            _refuse(
                V1RelaxedKKTRefusalCode.CUDA_WORKSPACE_EXCEEDED,
                f"{branch.value} private-state retention differs from its "
                "convergence receipt",
            )
        if state is None:
            continue
        expected = _single_fused_branch_state_forecast_bytes(
            branch=branch,
            route_count=problem.route_count,
            rank=problem.rank,
            block_count=len(problem.block_capacities),
        )
        if state.retained_tensor_bytes != expected:
            _refuse(
                V1RelaxedKKTRefusalCode.CUDA_WORKSPACE_EXCEEDED,
                f"{branch.value} retained tensor bytes differ from the "
                "pre-outcome forecast",
            )


@dataclass(frozen=True, slots=True)
class _ScaledProblem:
    objective_scale: float
    work_scale: float
    response_scale: float
    block_scales: tuple[float, ...]
    work_bound: float
    response_radius: float
    block_bounds: tuple[float, ...]
    operator_norm_squared_upper_bound: float


@dataclass(frozen=True, slots=True)
class _BranchStateNP:
    primal: np.ndarray
    work_dual: float
    block_duals: np.ndarray
    response_dual: np.ndarray
    iterations: int
    finite: bool
    termination_reason: str


def _scaled_problem(
    problem: V1RelaxedKKTProblem,
    branch: V1RelaxedKKTBranch,
) -> _ScaledProblem:
    objective_scale = problem._objective_scale
    work_norm = problem._work_l2_norm
    work_scale = max(1.0, work_norm)
    work_component = (work_norm / work_scale) ** 2
    block_scales = tuple(
        math.sqrt(float(block.stop - block.start)) for block in problem.block_capacities
    )
    block_component = 1.0 if block_scales else 0.0
    factor_frobenius_norm = problem._factor_frobenius_norm
    if branch is V1RelaxedKKTBranch.SPECTRAL:
        residual_operator_norm = math.sqrt(problem.residual_spectral_bound)
    else:
        residual_operator_norm = problem._residual_sqrt_l2_norm
    response_scale = max(1.0, math.hypot(factor_frobenius_norm, residual_operator_norm))
    if not math.isfinite(response_scale):
        raise ValueError("float64 response-operator scale is not representable")
    response_component = (factor_frobenius_norm / response_scale) ** 2 + (
        residual_operator_norm / response_scale
    ) ** 2
    operator_bound = work_component + block_component + response_component
    return _ScaledProblem(
        objective_scale=objective_scale,
        work_scale=work_scale,
        response_scale=response_scale,
        block_scales=block_scales,
        work_bound=problem.work_budget / work_scale,
        response_radius=math.sqrt(problem.quadratic_budget) / response_scale,
        block_bounds=tuple(
            block.removal_capacity / scale
            for block, scale in zip(problem.block_capacities, block_scales, strict=True)
        ),
        operator_norm_squared_upper_bound=operator_bound,
    )


def _q_forward_np(
    problem: V1RelaxedKKTProblem,
    branch: V1RelaxedKKTBranch,
    z: np.ndarray,
) -> np.ndarray:
    low_rank = problem.low_rank_factor @ z
    if branch is V1RelaxedKKTBranch.SPECTRAL:
        return np.concatenate(
            (low_rank, math.sqrt(problem.residual_spectral_bound) * z)
        )
    return np.concatenate(
        (
            low_rank,
            np.asarray(
                [float(np.dot(np.sqrt(problem.residual_diagonal), z))],
                dtype=np.float64,
            ),
        )
    )


def _q_adjoint_np(
    problem: V1RelaxedKKTProblem,
    branch: V1RelaxedKKTBranch,
    dual: np.ndarray,
) -> np.ndarray:
    rank = problem.rank
    result = problem.low_rank_factor.T @ dual[:rank]
    if branch is V1RelaxedKKTBranch.SPECTRAL:
        result = result + math.sqrt(problem.residual_spectral_bound) * dual[rank:]
    else:
        result = result + np.sqrt(problem.residual_diagonal) * float(dual[rank])
    return result


def _linear_forward_np(
    problem: V1RelaxedKKTProblem,
    scaled: _ScaledProblem,
    z: np.ndarray,
) -> tuple[float, np.ndarray]:
    work = float(np.dot(problem.corrected_work, z)) / scaled.work_scale
    blocks = np.asarray(
        [
            float(np.sum(z[item.start : item.stop], dtype=np.float64)) / scale
            for item, scale in zip(
                problem.block_capacities, scaled.block_scales, strict=True
            )
        ],
        dtype=np.float64,
    )
    return work, blocks


def _linear_adjoint_np(
    problem: V1RelaxedKKTProblem,
    scaled: _ScaledProblem,
    work_dual: float,
    block_duals: np.ndarray,
) -> np.ndarray:
    result = (work_dual / scaled.work_scale) * problem.corrected_work
    if problem.block_capacities:
        result = np.array(result, copy=True)
        for block, scale, dual in zip(
            problem.block_capacities,
            scaled.block_scales,
            block_duals,
            strict=True,
        ):
            result[block.start : block.stop] += float(dual) / scale
    return result


def _branch_q_np(
    problem: V1RelaxedKKTProblem,
    branch: V1RelaxedKKTBranch,
    z: np.ndarray,
) -> tuple[float, float]:
    low = problem.low_rank_factor @ z
    low_q = float(np.dot(low, low))
    if branch is V1RelaxedKKTBranch.SPECTRAL:
        residual_q = problem.residual_spectral_bound * float(np.dot(z, z))
    else:
        diagonal_projection = float(np.dot(np.sqrt(problem.residual_diagonal), z))
        residual_q = diagonal_projection * diagonal_projection
    return low_q, low_q + residual_q


def _outward_safe_repair_scale_np(scale: float) -> float:
    """Move one active homogeneous repair scale safely into feasibility."""

    if scale >= 1.0:
        return 1.0
    return max(
        0.0,
        float(scale) * (1.0 - 128.0 * np.finfo(np.float64).eps),
    )


def _repair_global_radial_np(
    problem: V1RelaxedKKTProblem,
    branch: V1RelaxedKKTBranch,
    clipped: np.ndarray,
) -> np.ndarray:
    """Preserve the historical one-scale feasible-primal repair exactly."""

    scale = 1.0
    work = float(np.dot(problem.corrected_work, clipped))
    if work > problem.work_budget:
        scale = min(scale, 0.0 if work <= 0.0 else problem.work_budget / work)
    _low, q = _branch_q_np(problem, branch, clipped)
    if q > problem.quadratic_budget:
        scale = min(
            scale,
            0.0 if q <= 0.0 else math.sqrt(problem.quadratic_budget / q),
        )
    for block in problem.block_capacities:
        total = float(np.sum(clipped[block.start : block.stop], dtype=np.float64))
        if total > block.removal_capacity:
            scale = min(
                scale,
                0.0 if total <= 0.0 else block.removal_capacity / total,
            )
    if scale < 1.0:
        clipped = clipped * _outward_safe_repair_scale_np(scale)
    return np.asarray(clipped, dtype=np.float64)


def _repair_block_separable_np(
    problem: V1RelaxedKKTProblem,
    branch: V1RelaxedKKTBranch,
    clipped: np.ndarray,
) -> np.ndarray:
    """Repair blocks independently, then apply one fresh F/Q scale.

    Unequal block scales change cancellation in ``H @ z``.  The branch-Q
    evaluation therefore occurs only after every block edit and necessarily
    recomputes the low-rank projection from the resulting vector.
    """

    candidate = np.array(clipped, dtype=np.float64, copy=True)
    for block in problem.block_capacities:
        segment = candidate[block.start : block.stop]
        total = float(np.sum(segment, dtype=np.float64))
        if total > block.removal_capacity:
            raw_scale = 0.0 if total <= 0.0 else float(block.removal_capacity) / total
            segment *= _outward_safe_repair_scale_np(raw_scale)
    scale = 1.0
    work = float(np.dot(problem.corrected_work, candidate))
    if work > problem.work_budget:
        scale = min(
            scale,
            0.0 if work <= 0.0 else problem.work_budget / work,
        )
    # This call must remain after the unequal block scaling above.
    _low, q = _branch_q_np(problem, branch, candidate)
    if q > problem.quadratic_budget:
        scale = min(
            scale,
            0.0 if q <= 0.0 else math.sqrt(problem.quadratic_budget / q),
        )
    if scale < 1.0:
        candidate *= _outward_safe_repair_scale_np(scale)
    return candidate


def _repair_feasible_np_candidates(
    problem: V1RelaxedKKTProblem,
    branch: V1RelaxedKKTBranch,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return historical radial and block-separable feasible candidates."""

    clipped = np.clip(z, 0.0, 1.0)
    clipped = np.where(problem.eligible_mask, clipped, 0.0)
    return (
        _repair_global_radial_np(problem, branch, clipped),
        _repair_block_separable_np(problem, branch, clipped),
    )


def _repair_feasible_np(
    problem: V1RelaxedKKTProblem,
    branch: V1RelaxedKKTBranch,
    z: np.ndarray,
) -> np.ndarray:
    radial, block_separable = _repair_feasible_np_candidates(problem, branch, z)
    radial_resource = float(np.dot(problem.resource, radial))
    block_resource = float(np.dot(problem.resource, block_separable))
    # Preserve the historical candidate on exact ties.  The feasible primal
    # lower bound can improve but cannot regress relative to the old repair.
    return block_separable if block_resource > radial_resource else radial


def _support_box_blocks_np(
    problem: V1RelaxedKKTProblem,
    linear: np.ndarray,
) -> float:
    values = np.where(problem.eligible_mask, linear, -np.inf)
    if not problem.block_capacities:
        return float(
            np.sum(np.maximum(values[np.isfinite(values)], 0.0), dtype=np.float64)
        )
    total = 0.0
    for block in problem.block_capacities:
        capacity = block.removal_capacity
        if capacity <= 0:
            continue
        segment = values[block.start : block.stop]
        positive = segment[np.logical_and(np.isfinite(segment), segment > 0.0)]
        if positive.size <= capacity:
            total += float(np.sum(positive, dtype=np.float64))
        else:
            split = int(positive.size) - capacity
            top = np.partition(positive, split)[split:]
            total += float(np.sum(top, dtype=np.float64))
    return total


def _primal_receipt_np(
    problem: V1RelaxedKKTProblem,
    branch: V1RelaxedKKTBranch,
    z: np.ndarray,
    tolerance: float,
) -> V1RelaxedKKTPrimalFeasibilityReceipt:
    work = float(np.dot(problem.corrected_work, z))
    _low, q = _branch_q_np(problem, branch, z)
    resource = float(np.dot(problem.resource, z))
    block_sums = tuple(
        float(np.sum(z[item.start : item.stop], dtype=np.float64))
        for item in problem.block_capacities
    )
    box_violation = max(
        0.0,
        float(-np.min(z)),
        float(np.max(z - 1.0)),
        (
            float(np.max(np.abs(z[~problem.eligible_mask])))
            if bool(np.any(~problem.eligible_mask))
            else 0.0
        ),
    )
    block_violation = max(
        (
            max(0.0, value - block.removal_capacity)
            for value, block in zip(block_sums, problem.block_capacities, strict=True)
        ),
        default=0.0,
    )
    work_violation = max(0.0, work - problem.work_budget)
    q_violation = max(0.0, q - problem.quadratic_budget)
    normalized = max(
        box_violation,
        work_violation / max(1.0, problem.work_budget),
        q_violation / max(1.0, problem.quadratic_budget),
        max(
            (
                max(0.0, value - block.removal_capacity)
                / max(1.0, float(block.removal_capacity))
                for value, block in zip(
                    block_sums, problem.block_capacities, strict=True
                )
            ),
            default=0.0,
        ),
    )
    return V1RelaxedKKTPrimalFeasibilityReceipt(
        resource_value=resource,
        work_value=work,
        quadratic_value=q,
        response_norm=math.sqrt(max(0.0, q)),
        maximum_box_violation=box_violation,
        maximum_block_violation=block_violation,
        work_violation=work_violation,
        quadratic_violation=q_violation,
        normalized_maximum_violation=normalized,
        block_sums=block_sums,
        feasible=normalized <= tolerance,
    )


def _dual_receipts_np(
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    branch: V1RelaxedKKTBranch,
    scaled: _ScaledProblem,
    primal: np.ndarray,
    work_dual: float,
    block_duals: np.ndarray,
    response_dual: np.ndarray,
) -> tuple[V1RelaxedKKTStationarityReceipt, V1RelaxedKKTGapReceipt, np.ndarray]:
    linear_adjoint = _linear_adjoint_np(problem, scaled, work_dual, block_duals)
    response_adjoint = (
        _q_adjoint_np(problem, branch, response_dual) / scaled.response_scale
    )
    reduced_internal = (
        problem.resource / scaled.objective_scale - linear_adjoint - response_adjoint
    )
    support_internal = _support_box_blocks_np(problem, reduced_internal)
    stationarity_internal = max(
        0.0,
        support_internal - float(np.dot(reduced_internal, primal)),
    )
    work_forward, block_forward = _linear_forward_np(problem, scaled, primal)
    response_forward = _q_forward_np(problem, branch, primal) / scaled.response_scale
    linear_bounds = np.asarray(
        (scaled.work_bound, *scaled.block_bounds), dtype=np.float64
    )
    linear_duals = np.asarray((work_dual, *block_duals), dtype=np.float64)
    linear_values = np.asarray((work_forward, *block_forward), dtype=np.float64)
    response_dual_norm = float(np.linalg.norm(response_dual))
    response_pairing = float(np.dot(response_dual, response_forward))
    complementarity_internal = max(
        0.0,
        float(np.dot(linear_duals, linear_bounds - linear_values))
        + scaled.response_radius * response_dual_norm
        - response_pairing,
    )
    dual_internal = (
        float(np.dot(linear_duals, linear_bounds))
        + scaled.response_radius * response_dual_norm
        + support_internal
    )
    primal_resource = float(np.dot(problem.resource, primal))
    dual_upper = max(primal_resource, dual_internal * scaled.objective_scale)
    absolute_gap = max(0.0, dual_upper - primal_resource)
    denominator = max(1.0, abs(primal_resource), abs(dual_upper))
    relative_gap = absolute_gap / denominator
    stationarity = stationarity_internal * scaled.objective_scale
    complementarity = complementarity_internal * scaled.objective_scale
    stationarity_receipt = V1RelaxedKKTStationarityReceipt(
        work_dual=float(work_dual * scaled.objective_scale / scaled.work_scale),
        block_duals=tuple(
            float(dual * scaled.objective_scale / scale)
            for dual, scale in zip(block_duals, scaled.block_scales, strict=True)
        ),
        response_dual_norm=float(
            response_dual_norm * scaled.objective_scale / scaled.response_scale
        ),
        lagrangian_support_value=support_internal * scaled.objective_scale,
        lagrangian_stationarity_gap=stationarity,
        normalized_stationarity_gap=stationarity / denominator,
        complementarity_gap=complementarity,
        normalized_complementarity_gap=complementarity / denominator,
        reduced_advantage_linf=float(
            np.max(np.abs(reduced_internal)) * scaled.objective_scale
        ),
    )
    gap_receipt = V1RelaxedKKTGapReceipt(
        primal_resource_lower_bound=primal_resource,
        conic_dual_resource_upper_bound=dual_upper,
        absolute_duality_gap=absolute_gap,
        relative_duality_gap=relative_gap,
        absolute_gap_tolerance=policy.absolute_gap_tolerance,
        relative_gap_tolerance=policy.relative_gap_tolerance,
        within_tolerance=(
            absolute_gap
            <= policy.absolute_gap_tolerance
            + policy.relative_gap_tolerance * denominator
        ),
    )
    reduced_raw = _readonly_copy(
        reduced_internal * scaled.objective_scale, dtype=np.dtype(np.float64)
    )
    return stationarity_receipt, gap_receipt, reduced_raw


def _solve_branch_numpy(
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    branch: V1RelaxedKKTBranch,
) -> tuple[_BranchStateNP, _ScaledProblem, V1RelaxedKKTBranchReceipt, np.ndarray]:
    scaled = _scaled_problem(problem, branch)
    operator_bound = scaled.operator_norm_squared_upper_bound
    if operator_bound <= 0.0:
        step = 1.0
    else:
        step = policy.step_safety / math.sqrt(operator_bound)
    route_count = problem.route_count
    response_dimension = problem.rank + (
        route_count if branch is V1RelaxedKKTBranch.SPECTRAL else 1
    )
    z = np.zeros(route_count, dtype=np.float64)
    z_bar = np.zeros_like(z)
    average = np.zeros_like(z)
    best = np.zeros_like(z)
    best_resource = 0.0
    work_dual = 0.0
    block_duals = np.zeros(len(problem.block_capacities), dtype=np.float64)
    response_dual = np.zeros(response_dimension, dtype=np.float64)
    finite = True
    termination = "maximum_iterations_exhausted"
    iterations = 0
    final_primal = _primal_receipt_np(
        problem, branch, best, policy.primal_feasibility_tolerance
    )
    final_stationarity, final_gap, reduced = _dual_receipts_np(
        problem,
        policy,
        branch,
        scaled,
        best,
        work_dual,
        block_duals,
        response_dual,
    )
    converged = False
    objective = problem.resource / scaled.objective_scale
    for iteration in range(1, policy.max_iterations + 1):
        iterations = iteration
        work_forward, block_forward = _linear_forward_np(problem, scaled, z_bar)
        work_dual = max(
            0.0,
            work_dual + step * (work_forward - scaled.work_bound),
        )
        if block_duals.size:
            block_duals = np.maximum(
                0.0,
                block_duals
                + step
                * (block_forward - np.asarray(scaled.block_bounds, dtype=np.float64)),
            )
        q_value = (
            response_dual
            + step * _q_forward_np(problem, branch, z_bar) / scaled.response_scale
        )
        q_norm = float(np.linalg.norm(q_value))
        if q_norm > 0.0:
            shrink = max(0.0, 1.0 - step * scaled.response_radius / q_norm)
            response_dual = q_value * shrink
        else:
            response_dual = q_value
        adjoint = (
            _linear_adjoint_np(problem, scaled, work_dual, block_duals)
            + _q_adjoint_np(problem, branch, response_dual) / scaled.response_scale
        )
        z_new = np.clip(z + step * (objective - adjoint), 0.0, 1.0)
        z_new[~problem.eligible_mask] = 0.0
        z_bar = z_new + policy.over_relaxation * (z_new - z)
        z = z_new
        average += (z - average) / float(iteration)
        if not (
            np.isfinite(z).all()
            and np.isfinite(z_bar).all()
            and np.isfinite(response_dual).all()
            and np.isfinite(block_duals).all()
            and math.isfinite(work_dual)
        ):
            finite = False
            termination = "nonfinite_iterate_refusal"
            break
        should_check = (
            iteration % policy.check_interval == 0 or iteration == policy.max_iterations
        )
        if not should_check:
            continue
        for proposal in (z, average):
            repaired = _repair_feasible_np(problem, branch, proposal)
            value = float(np.dot(problem.resource, repaired))
            if value > best_resource:
                best = repaired
                best_resource = value
        final_primal = _primal_receipt_np(
            problem, branch, best, policy.primal_feasibility_tolerance
        )
        final_stationarity, final_gap, reduced = _dual_receipts_np(
            problem,
            policy,
            branch,
            scaled,
            best,
            work_dual,
            block_duals,
            response_dual,
        )
        converged = all(
            (
                iteration >= policy.minimum_iterations,
                final_primal.feasible,
                final_gap.within_tolerance,
                final_stationarity.normalized_stationarity_gap
                <= policy.normalized_stationarity_tolerance,
                final_stationarity.normalized_complementarity_gap
                <= policy.normalized_complementarity_tolerance,
            )
        )
        if converged:
            termination = "certified_primal_dual_tolerances_satisfied"
            break
    convergence = V1RelaxedKKTConvergenceReceipt(
        branch=branch,
        converged=converged,
        iterations=iterations,
        maximum_iterations=policy.max_iterations,
        termination_reason=termination,
        backend=policy.backend,
        device="cpu",
        dtype="float64",
        algorithm_id=RELAXED_KKT_ALGORITHM_ID,
        scaled_operator_norm_squared_upper_bound=operator_bound,
        primal_step_size=step,
        dual_step_size=step,
        finite_iterates=finite,
    )
    receipt = V1RelaxedKKTBranchReceipt(
        primal=final_primal,
        stationarity=final_stationarity,
        gap=final_gap,
        convergence=convergence,
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
    )
    state = _BranchStateNP(
        primal=_readonly_copy(best, dtype=np.dtype(np.float64)),
        work_dual=work_dual,
        block_duals=_readonly_copy(block_duals, dtype=np.dtype(np.float64)),
        response_dual=_readonly_copy(response_dual, dtype=np.dtype(np.float64)),
        iterations=iterations,
        finite=finite,
        termination_reason=termination,
    )
    return state, scaled, receipt, reduced


def _cpu_rounded_order(
    problem: V1RelaxedKKTProblem,
    relaxed: np.ndarray,
    reduced_advantage: np.ndarray,
    maximum_count: int,
) -> np.ndarray:
    physical = np.flatnonzero(problem.eligible_mask).astype(np.int64, copy=False)
    ratio = problem.corrected_work[physical] / problem.resource[physical]
    order_relative = np.lexsort(
        (
            physical,
            ratio,
            -reduced_advantage[physical],
            -relaxed[physical],
        )
    )
    ordered = physical[order_relative]
    if not problem.block_capacities:
        return _readonly_copy(ordered[:maximum_count], dtype=np.dtype(np.int64))
    block_stops = np.asarray(
        [block.stop for block in problem.block_capacities], dtype=np.int64
    )
    capacities: list[int] = []
    for block in problem.block_capacities:
        capacities.append(block.removal_capacity)
    used = [0 for _ in capacities]
    accepted: list[int] = []
    for index in ordered:
        block_index = int(np.searchsorted(block_stops, int(index), side="right"))
        if used[block_index] >= capacities[block_index]:
            continue
        accepted.append(int(index))
        used[block_index] += 1
        if len(accepted) >= maximum_count:
            break
    return _readonly_copy(
        np.asarray(accepted, dtype=np.int64), dtype=np.dtype(np.int64)
    )


def _hard_candidates_from_order(
    problem: V1RelaxedKKTProblem,
    branch: V1RelaxedKKTBranch,
    order: np.ndarray,
    counts: tuple[int, ...],
) -> tuple[V1RelaxedKKTHardCandidate, ...]:
    candidates: list[V1RelaxedKKTHardCandidate] = []
    previous = 0
    h_sum = np.zeros(problem.rank, dtype=np.float64)
    resource = 0.0
    work = 0.0
    sqrt_d_sum = 0.0
    epsilon = np.finfo(np.float64).eps
    for count in counts:
        shell = order[previous:count]
        for start in range(0, int(shell.size), 1 << 16):
            chunk = shell[start : start + (1 << 16)]
            h_sum += np.sum(problem.low_rank_factor[:, chunk], axis=1, dtype=np.float64)
            resource += float(np.sum(problem.resource[chunk], dtype=np.float64))
            work += float(np.sum(problem.corrected_work[chunk], dtype=np.float64))
            sqrt_d_sum += float(
                np.sum(
                    np.sqrt(problem.residual_diagonal[chunk]),
                    dtype=np.float64,
                )
            )
        low_q = float(np.dot(h_sum, h_sum))
        spectral_q = problem.residual_spectral_bound * float(count)
        diagonal_q = sqrt_d_sum * sqrt_d_sum
        residual_q = min(spectral_q, diagonal_q)
        canonical_q = low_q + residual_q
        work_tolerance = 128.0 * epsilon * max(1.0, work, problem.work_budget)
        q_tolerance = 128.0 * epsilon * max(1.0, canonical_q, problem.quadratic_budget)
        work_feasible = bool(work <= problem.work_budget + work_tolerance)
        q_feasible = bool(canonical_q <= problem.quadratic_budget + q_tolerance)
        prefix = order[:count]
        candidates.append(
            V1RelaxedKKTHardCandidate(
                branch=branch,
                prefix_count=count,
                removed_index_hash=_array_digest(prefix),
                resource_value=resource,
                corrected_work_value=work,
                low_rank_quadratic=low_q,
                spectral_residual_quadratic=spectral_q,
                diagonal_l1_residual_quadratic=diagonal_q,
                canonical_residual_quadratic=residual_q,
                canonical_quadratic=canonical_q,
                work_feasible=work_feasible,
                quadratic_feasible=q_feasible,
                feasible=work_feasible and q_feasible,
            )
        )
        previous = count
    return tuple(candidates)


def _empty_ladder(branch: V1RelaxedKKTBranch) -> V1RelaxedKKTRoundedLadder:
    return V1RelaxedKKTRoundedLadder(
        branch=branch,
        ordered_route_indices=np.asarray((), dtype=np.int64),
        candidates=(),
    )


def _round_numpy_branch(
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    branch: V1RelaxedKKTBranch,
    relaxed: np.ndarray,
    reduced: np.ndarray,
    *,
    authorized: bool,
) -> V1RelaxedKKTRoundedLadder:
    if not authorized:
        return _empty_ladder(branch)
    capacity = problem.legal_hard_capacity
    counts = tuple(
        count for count in policy.candidate_count_ladder if count <= capacity
    )
    if not counts:
        return _empty_ladder(branch)
    order = _cpu_rounded_order(problem, relaxed, reduced, counts[-1])
    if order.size < counts[-1]:
        counts = tuple(count for count in counts if count <= int(order.size))
    if not counts:
        return _empty_ladder(branch)
    order = order[: counts[-1]]
    candidates = _hard_candidates_from_order(problem, branch, order, counts)
    return V1RelaxedKKTRoundedLadder(
        branch=branch,
        ordered_route_indices=order,
        candidates=candidates,
    )


def _torch_module() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - optional dependency.
        _refuse(V1RelaxedKKTRefusalCode.CUDA_UNAVAILABLE, str(error))
    return torch


@dataclass(frozen=True, slots=True)
class _TorchProblem:
    torch: Any
    device: Any
    resource: Any
    work: Any
    factor: Any
    residual: Any
    sqrt_residual: Any
    eligible: Any
    objective: Any
    block_eligible_counts: tuple[int, ...]
    block_bounds: Any | None = None
    has_ineligible: bool | None = None


def _synchronize_torch_device(tensors: _TorchProblem) -> None:
    """Synchronize CUDA timing boundaries; CPU private tests are a no-op."""

    if getattr(tensors.device, "type", None) == "cuda":
        tensors.torch.cuda.synchronize(tensors.device)


def _torch_problem(
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    scaled: _ScaledProblem,
    *,
    counters: _CUDAExecutionCounters | None = None,
) -> _TorchProblem:
    torch = _torch_module()
    device = torch.device(policy.torch_device)
    if device.type != "cuda" or not torch.cuda.is_available():
        _refuse(
            V1RelaxedKKTRefusalCode.CUDA_UNAVAILABLE,
            "backend='torch_cuda' requires an available CUDA device",
        )
    try:
        if counters is not None:
            torch.cuda.synchronize(device)
            started = time.perf_counter()
        resource = torch.tensor(problem.resource, device=device, dtype=torch.float64)
        work = torch.tensor(problem.corrected_work, device=device, dtype=torch.float64)
        factor = torch.tensor(
            problem.low_rank_factor, device=device, dtype=torch.float64
        )
        residual = torch.tensor(
            problem.residual_diagonal, device=device, dtype=torch.float64
        )
        eligible = torch.tensor(problem.eligible_mask, device=device, dtype=torch.bool)
        block_bounds = torch.tensor(
            scaled.block_bounds, device=device, dtype=torch.float64
        )
        sqrt_residual = torch.sqrt(residual)
        objective = resource / scaled.objective_scale
        if counters is not None:
            torch.cuda.synchronize(device)
            counters.cold_h2d_seconds += time.perf_counter() - started
            counters.bulk_h2d_tensor_payload_bytes += (
                int(problem.resource.nbytes)
                + int(problem.corrected_work.nbytes)
                + int(problem.low_rank_factor.nbytes)
                + int(problem.residual_diagonal.nbytes)
                + int(problem.eligible_mask.nbytes)
            )
    except Exception as error:  # pragma: no cover - host/device dependent.
        oom_type = getattr(torch, "OutOfMemoryError", ())
        if (oom_type and isinstance(error, oom_type)) or (
            "out of memory" in str(error).lower()
        ):
            _refuse(V1RelaxedKKTRefusalCode.CUDA_OUT_OF_MEMORY, str(error))
        _refuse(V1RelaxedKKTRefusalCode.CUDA_UNAVAILABLE, str(error))
    return _TorchProblem(
        torch=torch,
        device=device,
        resource=resource,
        work=work,
        factor=factor,
        residual=residual,
        sqrt_residual=sqrt_residual,
        eligible=eligible,
        objective=objective,
        block_bounds=block_bounds,
        has_ineligible=bool(np.any(~problem.eligible_mask)),
        block_eligible_counts=tuple(
            int(np.count_nonzero(problem.eligible_mask[block.start : block.stop]))
            for block in problem.block_capacities
        ),
    )


def _factor_forward_columns_torch(
    tensors: _TorchProblem,
    columns: Any,
    counters: _CUDAExecutionCounters,
) -> Any:
    """Apply the one rank-major factor to one or more RHS columns once."""

    counters.factor_forward_scan_count += 1
    counters.factor_route_visit_count += int(tensors.factor.shape[1])
    return tensors.factor @ columns


def _factor_adjoint_columns_torch(
    tensors: _TorchProblem,
    columns: Any,
    counters: _CUDAExecutionCounters,
) -> Any:
    """Apply the shared factor adjoint to both branch columns once."""

    counters.factor_adjoint_scan_count += 1
    counters.factor_route_visit_count += int(tensors.factor.shape[1])
    return tensors.factor.T @ columns


def _branch_q_from_low_torch(
    tensors: _TorchProblem,
    problem: V1RelaxedKKTProblem,
    branch: V1RelaxedKKTBranch,
    z: Any,
    low: Any,
) -> Any:
    """Evaluate branch Q from a cached ``H @ z`` projection."""

    torch = tensors.torch
    low_q = torch.dot(low, low)
    if branch is V1RelaxedKKTBranch.SPECTRAL:
        return low_q + problem.residual_spectral_bound * torch.dot(z, z)
    projection = torch.dot(tensors.sqrt_residual, z)
    return low_q + projection * projection


def _repair_homogeneous_torch_cached(
    tensors: _TorchProblem,
    problem: V1RelaxedKKTProblem,
    branch: V1RelaxedKKTBranch,
    z: Any,
    low: Any,
    *,
    enforce_blocks: bool,
) -> tuple[Any, Any]:
    """Homogeneously repair a gate and its matching factor projection.

    Fused callers pass only iterates produced by box projection or their
    running average.  They are consequently already in the eligible box;
    preserving that invariant is what lets the low-rank projection scale
    exactly instead of triggering another route-scale factor scan.
    """

    torch = tensors.torch
    ratios: list[Any] = [torch.ones((), device=tensors.device, dtype=torch.float64)]
    work = torch.dot(tensors.work, z)
    if problem.work_budget == 0.0:
        ratios.append(
            torch.where(work > 0.0, torch.zeros_like(work), torch.ones_like(work))
        )
    else:
        ratios.append(
            torch.clamp(
                problem.work_budget / torch.clamp(work, min=np.finfo(np.float64).tiny),
                max=1.0,
            )
        )
    q = _branch_q_from_low_torch(tensors, problem, branch, z, low)
    if problem.quadratic_budget == 0.0:
        ratios.append(torch.where(q > 0.0, torch.zeros_like(q), torch.ones_like(q)))
    else:
        ratios.append(
            torch.clamp(
                torch.sqrt(
                    problem.quadratic_budget
                    / torch.clamp(q, min=np.finfo(np.float64).tiny)
                ),
                max=1.0,
            )
        )
    if enforce_blocks:
        for block in problem.block_capacities:
            total = torch.sum(z[block.start : block.stop])
            if block.removal_capacity == 0:
                ratios.append(
                    torch.where(
                        total > 0.0,
                        torch.zeros_like(total),
                        torch.ones_like(total),
                    )
                )
            else:
                ratios.append(
                    torch.clamp(
                        block.removal_capacity
                        / torch.clamp(total, min=np.finfo(np.float64).tiny),
                        max=1.0,
                    )
                )
    scale = torch.min(torch.stack(ratios))
    safe_scale = torch.where(
        scale < 1.0,
        scale * (1.0 - 128.0 * np.finfo(np.float64).eps),
        scale,
    )
    return z * safe_scale, low * safe_scale


def _repair_feasible_torch_cached(
    tensors: _TorchProblem,
    problem: V1RelaxedKKTProblem,
    branch: V1RelaxedKKTBranch,
    z: Any,
    low: Any,
) -> tuple[Any, Any]:
    """Preserve the historical global-radial cached repair."""

    return _repair_homogeneous_torch_cached(
        tensors,
        problem,
        branch,
        z,
        low,
        enforce_blocks=True,
    )


def _repair_fq_torch_cached(
    tensors: _TorchProblem,
    problem: V1RelaxedKKTProblem,
    branch: V1RelaxedKKTBranch,
    z: Any,
    low: Any,
) -> tuple[Any, Any]:
    """Apply only the global F/Q repair after block feasibility is closed."""

    return _repair_homogeneous_torch_cached(
        tensors,
        problem,
        branch,
        z,
        low,
        enforce_blocks=False,
    )


def _block_separable_columns_torch(
    tensors: _TorchProblem,
    problem: V1RelaxedKKTProblem,
    columns: Any,
) -> tuple[Any, bool]:
    """Independently repair block capacities for one or more route columns."""

    torch = tensors.torch
    if columns.ndim != 2 or int(columns.shape[0]) != problem.route_count:
        raise ValueError("block-separable repair requires route-major columns")
    if not problem.block_capacities:
        return columns, False
    block_scales: list[Any] = []
    safety = 1.0 - 128.0 * np.finfo(np.float64).eps
    for block in problem.block_capacities:
        totals = torch.sum(columns[block.start : block.stop, :], dim=0)
        if block.removal_capacity == 0:
            scales = torch.where(
                totals > 0.0,
                torch.zeros_like(totals),
                torch.ones_like(totals),
            )
        else:
            raw = float(block.removal_capacity) / torch.clamp(
                totals, min=np.finfo(np.float64).tiny
            )
            scales = torch.where(
                totals > float(block.removal_capacity),
                raw * safety,
                torch.ones_like(totals),
            )
        block_scales.append(scales)
    needs_scaling = bool(
        torch.stack(tuple(scale < 1.0 for scale in block_scales), dim=0).any().item()
    )
    if not needs_scaling:
        return columns, False
    repaired = columns.clone()
    for block, scales in zip(problem.block_capacities, block_scales, strict=True):
        repaired[block.start : block.stop, :] *= scales.reshape(1, -1)
    return repaired, True


def _q_forward_torch(
    tensors: _TorchProblem,
    problem: V1RelaxedKKTProblem,
    branch: V1RelaxedKKTBranch,
    z: Any,
) -> Any:
    torch = tensors.torch
    low = tensors.factor @ z
    if branch is V1RelaxedKKTBranch.SPECTRAL:
        return torch.cat((low, math.sqrt(problem.residual_spectral_bound) * z), dim=0)
    return torch.cat((low, torch.dot(tensors.sqrt_residual, z).reshape(1)), dim=0)


def _q_adjoint_torch(
    tensors: _TorchProblem,
    problem: V1RelaxedKKTProblem,
    branch: V1RelaxedKKTBranch,
    dual: Any,
) -> Any:
    rank = problem.rank
    result = tensors.factor.T @ dual[:rank]
    if branch is V1RelaxedKKTBranch.SPECTRAL:
        return result + math.sqrt(problem.residual_spectral_bound) * dual[rank:]
    return result + tensors.sqrt_residual * dual[rank]


def _linear_forward_torch(
    tensors: _TorchProblem,
    problem: V1RelaxedKKTProblem,
    scaled: _ScaledProblem,
    z: Any,
) -> tuple[Any, Any]:
    torch = tensors.torch
    work = torch.dot(tensors.work, z) / scaled.work_scale
    if problem.block_capacities:
        blocks = torch.stack(
            tuple(
                torch.sum(z[block.start : block.stop]) / scale
                for block, scale in zip(
                    problem.block_capacities, scaled.block_scales, strict=True
                )
            )
        )
    else:
        blocks = torch.empty(0, device=tensors.device, dtype=torch.float64)
    return work, blocks


def _linear_adjoint_torch(
    tensors: _TorchProblem,
    problem: V1RelaxedKKTProblem,
    scaled: _ScaledProblem,
    work_dual: Any,
    block_duals: Any,
) -> Any:
    result = (work_dual / scaled.work_scale) * tensors.work
    if problem.block_capacities:
        result = result.clone()
        for block_index, (block, scale) in enumerate(
            zip(problem.block_capacities, scaled.block_scales, strict=True)
        ):
            result[block.start : block.stop].add_(block_duals[block_index] / scale)
    return result


def _branch_q_torch(
    tensors: _TorchProblem,
    problem: V1RelaxedKKTProblem,
    branch: V1RelaxedKKTBranch,
    z: Any,
) -> tuple[Any, Any]:
    torch = tensors.torch
    low = tensors.factor @ z
    low_q = torch.dot(low, low)
    if branch is V1RelaxedKKTBranch.SPECTRAL:
        residual_q = problem.residual_spectral_bound * torch.dot(z, z)
    else:
        projection = torch.dot(tensors.sqrt_residual, z)
        residual_q = projection * projection
    return low_q, low_q + residual_q


def _repair_feasible_torch(
    tensors: _TorchProblem,
    problem: V1RelaxedKKTProblem,
    branch: V1RelaxedKKTBranch,
    z: Any,
) -> Any:
    radial, block_separable, _radial_low, _block_low = (
        _repair_feasible_torch_candidates(tensors, problem, branch, z)
    )
    radial_resource = tensors.torch.dot(tensors.resource, radial)
    block_resource = tensors.torch.dot(tensors.resource, block_separable)
    return tensors.torch.where(
        block_resource > radial_resource,
        block_separable,
        radial,
    )


def _repair_feasible_torch_candidates(
    tensors: _TorchProblem,
    problem: V1RelaxedKKTProblem,
    branch: V1RelaxedKKTBranch,
    z: Any,
) -> tuple[Any, Any, Any, Any]:
    """Return both torch repairs and their exactly matching ``H @ z`` rows."""

    torch = tensors.torch
    clipped = torch.clamp(z, 0.0, 1.0) * tensors.eligible
    low = tensors.factor @ clipped
    radial, radial_low = _repair_feasible_torch_cached(
        tensors, problem, branch, clipped, low
    )
    block_columns, block_changed = _block_separable_columns_torch(
        tensors, problem, clipped.reshape(-1, 1)
    )
    block_candidate = block_columns[:, 0]
    # Unequal block scales can destroy cancellation.  Never reuse ``low``
    # after such a change; the finite candidate owns a fresh projection.
    block_low = tensors.factor @ block_candidate if block_changed else low
    block_repaired, block_repaired_low = _repair_fq_torch_cached(
        tensors,
        problem,
        branch,
        block_candidate,
        block_low,
    )
    return radial, block_repaired, radial_low, block_repaired_low


def _support_box_blocks_torch(
    tensors: _TorchProblem,
    problem: V1RelaxedKKTProblem,
    linear: Any,
) -> Any:
    torch = tensors.torch
    if not problem.block_capacities:
        return torch.sum(
            torch.where(tensors.eligible, torch.clamp(linear, min=0.0), 0.0)
        )
    total = torch.zeros((), device=tensors.device, dtype=torch.float64)
    for block, eligible_count in zip(
        problem.block_capacities,
        tensors.block_eligible_counts,
        strict=True,
    ):
        capacity = min(block.removal_capacity, eligible_count)
        if capacity <= 0:
            continue
        eligible = tensors.eligible[block.start : block.stop]
        positive = torch.where(
            eligible,
            torch.clamp(linear[block.start : block.stop], min=0.0),
            0.0,
        )
        if capacity >= eligible_count:
            total = total + torch.sum(positive)
            continue
        excluded_count = eligible_count - capacity
        if capacity <= excluded_count:
            # Select the smaller side of the capacity split.  This is the
            # ordinary path only when the legal action is itself small.
            total = total + torch.sum(
                torch.topk(positive, capacity, sorted=False).values
            )
        else:
            # The production survivor constraint normally permits all but one
            # live route in a block.  Materializing ``topk(block_size - 1)``
            # would dominate both HBM workspace and receipt time.  The exact
            # support is instead the positive sum minus the few smallest
            # eligible clipped values.  Ineligible zeros are replaced by
            # +inf so they cannot consume the complement selection.
            smallest_input = torch.where(
                eligible,
                positive,
                torch.full_like(positive, torch.inf),
            )
            excluded = torch.topk(
                smallest_input,
                excluded_count,
                largest=False,
                sorted=False,
            ).values
            total = total + torch.sum(positive) - torch.sum(excluded)
    return total


def _torch_receipts(
    tensors: _TorchProblem,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    branch: V1RelaxedKKTBranch,
    scaled: _ScaledProblem,
    primal: Any,
    work_dual: Any,
    block_duals: Any,
    response_dual: Any,
) -> tuple[
    V1RelaxedKKTPrimalFeasibilityReceipt,
    V1RelaxedKKTStationarityReceipt,
    V1RelaxedKKTGapReceipt,
    Any,
]:
    torch = tensors.torch
    work = torch.dot(tensors.work, primal)
    _low, q = _branch_q_torch(tensors, problem, branch, primal)
    resource = torch.dot(tensors.resource, primal)
    block_sums_t = tuple(
        torch.sum(primal[item.start : item.stop]) for item in problem.block_capacities
    )
    box_violation = torch.maximum(
        torch.clamp(-torch.min(primal), min=0.0),
        torch.clamp(torch.max(primal - 1.0), min=0.0),
    )
    if bool(torch.any(~tensors.eligible).item()):
        box_violation = torch.maximum(
            box_violation, torch.max(torch.abs(primal[~tensors.eligible]))
        )
    block_violation_values = tuple(
        torch.clamp(value - block.removal_capacity, min=0.0)
        for value, block in zip(block_sums_t, problem.block_capacities, strict=True)
    )
    block_violation = (
        torch.max(torch.stack(block_violation_values))
        if block_violation_values
        else torch.zeros((), device=tensors.device, dtype=torch.float64)
    )
    work_violation = torch.clamp(work - problem.work_budget, min=0.0)
    q_violation = torch.clamp(q - problem.quadratic_budget, min=0.0)
    normalized_parts = [
        box_violation,
        work_violation / max(1.0, problem.work_budget),
        q_violation / max(1.0, problem.quadratic_budget),
    ]
    normalized_parts.extend(
        value / max(1.0, float(block.removal_capacity))
        for value, block in zip(
            block_violation_values, problem.block_capacities, strict=True
        )
    )
    normalized = torch.max(torch.stack(normalized_parts))
    linear_adjoint = _linear_adjoint_torch(
        tensors, problem, scaled, work_dual, block_duals
    )
    response_adjoint = (
        _q_adjoint_torch(tensors, problem, branch, response_dual)
        / scaled.response_scale
    )
    reduced_internal = tensors.objective - linear_adjoint - response_adjoint
    support_internal = _support_box_blocks_torch(tensors, problem, reduced_internal)
    stationarity_internal = torch.clamp(
        support_internal - torch.dot(reduced_internal, primal), min=0.0
    )
    work_forward, block_forward = _linear_forward_torch(
        tensors, problem, scaled, primal
    )
    linear_values = torch.cat((work_forward.reshape(1), block_forward))
    linear_bounds = torch.as_tensor(
        (scaled.work_bound, *scaled.block_bounds),
        device=tensors.device,
        dtype=torch.float64,
    )
    linear_duals = torch.cat((work_dual.reshape(1), block_duals))
    response_forward = (
        _q_forward_torch(tensors, problem, branch, primal) / scaled.response_scale
    )
    response_dual_norm = torch.linalg.vector_norm(response_dual)
    complementarity_internal = torch.clamp(
        torch.dot(linear_duals, linear_bounds - linear_values)
        + scaled.response_radius * response_dual_norm
        - torch.dot(response_dual, response_forward),
        min=0.0,
    )
    dual_internal = (
        torch.dot(linear_duals, linear_bounds)
        + scaled.response_radius * response_dual_norm
        + support_internal
    )
    dual_upper_t = torch.maximum(resource, dual_internal * scaled.objective_scale)
    gap_t = torch.clamp(dual_upper_t - resource, min=0.0)
    denominator_t = torch.maximum(
        torch.ones((), device=tensors.device, dtype=torch.float64),
        torch.maximum(torch.abs(resource), torch.abs(dual_upper_t)),
    )
    values = (
        torch.stack(
            (
                resource,
                work,
                q,
                torch.sqrt(torch.clamp(q, min=0.0)),
                box_violation,
                block_violation,
                work_violation,
                q_violation,
                normalized,
                response_dual_norm,
                support_internal,
                stationarity_internal,
                complementarity_internal,
                dual_upper_t,
                gap_t,
                denominator_t,
                torch.max(torch.abs(reduced_internal)),
            )
        )
        .detach()
        .cpu()
        .numpy()
    )
    block_sums = tuple(
        float(item)
        for item in (
            torch.stack(block_sums_t).detach().cpu().numpy()
            if block_sums_t
            else np.asarray((), dtype=np.float64)
        )
    )
    block_dual_values = tuple(
        float(item) for item in block_duals.detach().cpu().numpy()
    )
    (
        resource_v,
        work_v,
        q_v,
        norm_v,
        box_v,
        block_v,
        work_violation_v,
        q_violation_v,
        normalized_v,
        response_dual_norm_v,
        support_v,
        stationarity_v,
        complementarity_v,
        dual_upper_v,
        gap_v,
        denominator_v,
        reduced_linf_v,
    ) = (float(item) for item in values)
    primal_receipt = V1RelaxedKKTPrimalFeasibilityReceipt(
        resource_value=resource_v,
        work_value=work_v,
        quadratic_value=q_v,
        response_norm=norm_v,
        maximum_box_violation=box_v,
        maximum_block_violation=block_v,
        work_violation=work_violation_v,
        quadratic_violation=q_violation_v,
        normalized_maximum_violation=normalized_v,
        block_sums=block_sums,
        feasible=normalized_v <= policy.primal_feasibility_tolerance,
    )
    stationarity = stationarity_v * scaled.objective_scale
    complementarity = complementarity_v * scaled.objective_scale
    stationarity_receipt = V1RelaxedKKTStationarityReceipt(
        work_dual=float(work_dual.item() * scaled.objective_scale / scaled.work_scale),
        block_duals=tuple(
            dual * scaled.objective_scale / scale
            for dual, scale in zip(block_dual_values, scaled.block_scales, strict=True)
        ),
        response_dual_norm=response_dual_norm_v
        * scaled.objective_scale
        / scaled.response_scale,
        lagrangian_support_value=support_v * scaled.objective_scale,
        lagrangian_stationarity_gap=stationarity,
        normalized_stationarity_gap=stationarity / denominator_v,
        complementarity_gap=complementarity,
        normalized_complementarity_gap=complementarity / denominator_v,
        reduced_advantage_linf=reduced_linf_v * scaled.objective_scale,
    )
    gap_receipt = V1RelaxedKKTGapReceipt(
        primal_resource_lower_bound=resource_v,
        conic_dual_resource_upper_bound=dual_upper_v,
        absolute_duality_gap=gap_v,
        relative_duality_gap=gap_v / denominator_v,
        absolute_gap_tolerance=policy.absolute_gap_tolerance,
        relative_gap_tolerance=policy.relative_gap_tolerance,
        within_tolerance=(
            gap_v
            <= policy.absolute_gap_tolerance
            + policy.relative_gap_tolerance * denominator_v
        ),
    )
    return (
        primal_receipt,
        stationarity_receipt,
        gap_receipt,
        reduced_internal * scaled.objective_scale,
    )


def _torch_receipts_cached(
    tensors: _TorchProblem,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    branch: V1RelaxedKKTBranch,
    scaled: _ScaledProblem,
    primal: Any,
    primal_low: Any,
    work_dual: Any,
    block_duals: Any,
    response_low_dual: Any,
    response_tail_dual: Any,
    reduced_internal: Any,
    counters: _CUDAExecutionCounters,
) -> tuple[
    V1RelaxedKKTPrimalFeasibilityReceipt,
    V1RelaxedKKTStationarityReceipt,
    V1RelaxedKKTGapReceipt,
    Any,
]:
    """Create a complete branch certificate without another factor scan."""

    torch = tensors.torch
    _synchronize_torch_device(tensors)
    started = time.perf_counter()
    work = torch.dot(tensors.work, primal)
    q = _branch_q_from_low_torch(tensors, problem, branch, primal, primal_low)
    resource = torch.dot(tensors.resource, primal)
    block_sums_t = tuple(
        torch.sum(primal[item.start : item.stop]) for item in problem.block_capacities
    )
    box_violation = torch.maximum(
        torch.clamp(-torch.min(primal), min=0.0),
        torch.clamp(torch.max(primal - 1.0), min=0.0),
    )
    has_ineligible = (
        tensors.has_ineligible
        if tensors.has_ineligible is not None
        else bool(torch.any(~tensors.eligible).item())
    )
    if has_ineligible:
        box_violation = torch.maximum(
            box_violation, torch.max(torch.abs(primal[~tensors.eligible]))
        )
    block_violation_values = tuple(
        torch.clamp(value - block.removal_capacity, min=0.0)
        for value, block in zip(block_sums_t, problem.block_capacities, strict=True)
    )
    block_violation = (
        torch.max(torch.stack(block_violation_values))
        if block_violation_values
        else torch.zeros((), device=tensors.device, dtype=torch.float64)
    )
    work_violation = torch.clamp(work - problem.work_budget, min=0.0)
    q_violation = torch.clamp(q - problem.quadratic_budget, min=0.0)
    normalized_parts = [
        box_violation,
        work_violation / max(1.0, problem.work_budget),
        q_violation / max(1.0, problem.quadratic_budget),
    ]
    normalized_parts.extend(
        value / max(1.0, float(block.removal_capacity))
        for value, block in zip(
            block_violation_values, problem.block_capacities, strict=True
        )
    )
    normalized = torch.max(torch.stack(normalized_parts))
    support_internal = _support_box_blocks_torch(tensors, problem, reduced_internal)
    stationarity_internal = torch.clamp(
        support_internal - torch.dot(reduced_internal, primal), min=0.0
    )
    work_forward, block_forward = _linear_forward_torch(
        tensors, problem, scaled, primal
    )
    linear_values = torch.cat((work_forward.reshape(1), block_forward))
    linear_bounds = torch.cat(
        (
            torch.as_tensor(
                (scaled.work_bound,),
                device=tensors.device,
                dtype=torch.float64,
            ),
            (
                tensors.block_bounds
                if tensors.block_bounds is not None
                else torch.as_tensor(
                    scaled.block_bounds,
                    device=tensors.device,
                    dtype=torch.float64,
                )
            ),
        )
    )
    linear_duals = torch.cat((work_dual.reshape(1), block_duals))
    low_pairing = torch.dot(response_low_dual, primal_low) / scaled.response_scale
    if branch is V1RelaxedKKTBranch.SPECTRAL:
        response_dual_norm = torch.sqrt(
            torch.dot(response_low_dual, response_low_dual)
            + torch.dot(response_tail_dual, response_tail_dual)
        )
        response_pairing = low_pairing + (
            math.sqrt(problem.residual_spectral_bound)
            * torch.dot(response_tail_dual, primal)
            / scaled.response_scale
        )
    else:
        response_dual_norm = torch.sqrt(
            torch.dot(response_low_dual, response_low_dual)
            + response_tail_dual * response_tail_dual
        )
        response_pairing = low_pairing + (
            response_tail_dual
            * torch.dot(tensors.sqrt_residual, primal)
            / scaled.response_scale
        )
    complementarity_internal = torch.clamp(
        torch.dot(linear_duals, linear_bounds - linear_values)
        + scaled.response_radius * response_dual_norm
        - response_pairing,
        min=0.0,
    )
    dual_internal = (
        torch.dot(linear_duals, linear_bounds)
        + scaled.response_radius * response_dual_norm
        + support_internal
    )
    dual_upper_t = torch.maximum(resource, dual_internal * scaled.objective_scale)
    gap_t = torch.clamp(dual_upper_t - resource, min=0.0)
    denominator_t = torch.maximum(
        torch.ones((), device=tensors.device, dtype=torch.float64),
        torch.maximum(torch.abs(resource), torch.abs(dual_upper_t)),
    )
    values_t = torch.stack(
        (
            resource,
            work,
            q,
            torch.sqrt(torch.clamp(q, min=0.0)),
            box_violation,
            block_violation,
            work_violation,
            q_violation,
            normalized,
            response_dual_norm,
            support_internal,
            stationarity_internal,
            complementarity_internal,
            dual_upper_t,
            gap_t,
            denominator_t,
            torch.max(torch.abs(reduced_internal)),
        )
    )
    values = values_t.detach().cpu().numpy()
    block_sums_array = (
        torch.stack(block_sums_t).detach().cpu().numpy()
        if block_sums_t
        else np.asarray((), dtype=np.float64)
    )
    block_dual_array = block_duals.detach().cpu().numpy()
    (
        resource_v,
        work_v,
        q_v,
        norm_v,
        box_v,
        block_v,
        work_violation_v,
        q_violation_v,
        normalized_v,
        response_dual_norm_v,
        support_v,
        stationarity_v,
        complementarity_v,
        dual_upper_v,
        gap_v,
        denominator_v,
        reduced_linf_v,
    ) = (float(item) for item in values)
    block_sums = tuple(float(item) for item in block_sums_array)
    block_dual_values = tuple(float(item) for item in block_dual_array)
    primal_receipt = V1RelaxedKKTPrimalFeasibilityReceipt(
        resource_value=resource_v,
        work_value=work_v,
        quadratic_value=q_v,
        response_norm=norm_v,
        maximum_box_violation=box_v,
        maximum_block_violation=block_v,
        work_violation=work_violation_v,
        quadratic_violation=q_violation_v,
        normalized_maximum_violation=normalized_v,
        block_sums=block_sums,
        feasible=normalized_v <= policy.primal_feasibility_tolerance,
    )
    stationarity = stationarity_v * scaled.objective_scale
    complementarity = complementarity_v * scaled.objective_scale
    stationarity_receipt = V1RelaxedKKTStationarityReceipt(
        work_dual=float(work_dual.item() * scaled.objective_scale / scaled.work_scale),
        block_duals=tuple(
            dual * scaled.objective_scale / scale
            for dual, scale in zip(block_dual_values, scaled.block_scales, strict=True)
        ),
        response_dual_norm=(
            response_dual_norm_v * scaled.objective_scale / scaled.response_scale
        ),
        lagrangian_support_value=support_v * scaled.objective_scale,
        lagrangian_stationarity_gap=stationarity,
        normalized_stationarity_gap=stationarity / denominator_v,
        complementarity_gap=complementarity,
        normalized_complementarity_gap=complementarity / denominator_v,
        reduced_advantage_linf=reduced_linf_v * scaled.objective_scale,
    )
    gap_receipt = V1RelaxedKKTGapReceipt(
        primal_resource_lower_bound=resource_v,
        conic_dual_resource_upper_bound=dual_upper_v,
        absolute_duality_gap=gap_v,
        relative_duality_gap=gap_v / denominator_v,
        absolute_gap_tolerance=policy.absolute_gap_tolerance,
        relative_gap_tolerance=policy.relative_gap_tolerance,
        within_tolerance=(
            gap_v
            <= policy.absolute_gap_tolerance
            + policy.relative_gap_tolerance * denominator_v
        ),
    )
    _synchronize_torch_device(tensors)
    counters.receipt_seconds += time.perf_counter() - started
    return (
        primal_receipt,
        stationarity_receipt,
        gap_receipt,
        reduced_internal * scaled.objective_scale,
    )


def _solve_branch_torch(
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    branch: V1RelaxedKKTBranch,
    tensors: _TorchProblem | None = None,
) -> tuple[Any, Any, _ScaledProblem, V1RelaxedKKTBranchReceipt]:
    scaled = _scaled_problem(problem, branch)
    if tensors is None:
        tensors = _torch_problem(problem, policy, scaled)
    torch = tensors.torch
    operator_bound = scaled.operator_norm_squared_upper_bound
    step = (
        1.0 if operator_bound <= 0.0 else policy.step_safety / math.sqrt(operator_bound)
    )
    route_count = problem.route_count
    response_dimension = problem.rank + (
        route_count if branch is V1RelaxedKKTBranch.SPECTRAL else 1
    )
    z = torch.zeros(route_count, device=tensors.device, dtype=torch.float64)
    z_bar = torch.zeros_like(z)
    average = torch.zeros_like(z)
    best = torch.zeros_like(z)
    best_resource = 0.0
    work_dual = torch.zeros((), device=tensors.device, dtype=torch.float64)
    block_duals = torch.zeros(
        len(problem.block_capacities), device=tensors.device, dtype=torch.float64
    )
    response_dual = torch.zeros(
        response_dimension, device=tensors.device, dtype=torch.float64
    )
    finite = True
    termination = "maximum_iterations_exhausted"
    converged = False
    iterations = 0
    final_primal, final_stationarity, final_gap, reduced = _torch_receipts(
        tensors,
        problem,
        policy,
        branch,
        scaled,
        best,
        work_dual,
        block_duals,
        response_dual,
    )
    for iteration in range(1, policy.max_iterations + 1):
        iterations = iteration
        work_forward, block_forward = _linear_forward_torch(
            tensors, problem, scaled, z_bar
        )
        work_dual = torch.clamp(
            work_dual + step * (work_forward - scaled.work_bound), min=0.0
        )
        if int(block_duals.numel()):
            block_bounds = (
                tensors.block_bounds
                if tensors.block_bounds is not None
                else torch.as_tensor(
                    scaled.block_bounds,
                    device=tensors.device,
                    dtype=torch.float64,
                )
            )
            block_duals = torch.clamp(
                block_duals + step * (block_forward - block_bounds), min=0.0
            )
        q_value = (
            response_dual
            + step
            * _q_forward_torch(tensors, problem, branch, z_bar)
            / scaled.response_scale
        )
        q_norm = torch.linalg.vector_norm(q_value)
        shrink = torch.clamp(
            1.0
            - step
            * scaled.response_radius
            / torch.clamp(q_norm, min=np.finfo(np.float64).tiny),
            min=0.0,
        )
        response_dual = q_value * shrink
        adjoint = (
            _linear_adjoint_torch(tensors, problem, scaled, work_dual, block_duals)
            + _q_adjoint_torch(tensors, problem, branch, response_dual)
            / scaled.response_scale
        )
        z_new = torch.clamp(z + step * (tensors.objective - adjoint), 0.0, 1.0)
        z_new = z_new * tensors.eligible
        z_bar = z_new + policy.over_relaxation * (z_new - z)
        z = z_new
        average = average + (z - average) / float(iteration)
        should_check = (
            iteration % policy.check_interval == 0 or iteration == policy.max_iterations
        )
        if not should_check:
            continue
        finite = bool(
            torch.isfinite(z).all().item()
            and torch.isfinite(z_bar).all().item()
            and torch.isfinite(response_dual).all().item()
            and torch.isfinite(block_duals).all().item()
            and torch.isfinite(work_dual).item()
        )
        if not finite:
            termination = "nonfinite_iterate_refusal"
            break
        for proposal in (z, average):
            repaired = _repair_feasible_torch(tensors, problem, branch, proposal)
            value = float(torch.dot(tensors.resource, repaired).item())
            if value > best_resource:
                best = repaired.clone()
                best_resource = value
        final_primal, final_stationarity, final_gap, reduced = _torch_receipts(
            tensors,
            problem,
            policy,
            branch,
            scaled,
            best,
            work_dual,
            block_duals,
            response_dual,
        )
        converged = all(
            (
                iteration >= policy.minimum_iterations,
                final_primal.feasible,
                final_gap.within_tolerance,
                final_stationarity.normalized_stationarity_gap
                <= policy.normalized_stationarity_tolerance,
                final_stationarity.normalized_complementarity_gap
                <= policy.normalized_complementarity_tolerance,
            )
        )
        if converged:
            termination = "certified_primal_dual_tolerances_satisfied"
            break
    convergence = V1RelaxedKKTConvergenceReceipt(
        branch=branch,
        converged=converged,
        iterations=iterations,
        maximum_iterations=policy.max_iterations,
        termination_reason=termination,
        backend=policy.backend,
        device=str(tensors.device),
        dtype="float64",
        algorithm_id=RELAXED_KKT_ALGORITHM_ID,
        scaled_operator_norm_squared_upper_bound=operator_bound,
        primal_step_size=step,
        dual_step_size=step,
        finite_iterates=finite,
    )
    receipt = V1RelaxedKKTBranchReceipt(
        primal=final_primal,
        stationarity=final_stationarity,
        gap=final_gap,
        convergence=convergence,
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
    )
    return best, reduced, scaled, receipt


def _fused_branch_receipt(
    *,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    branch: V1RelaxedKKTBranch,
    scaled: _ScaledProblem,
    primal: V1RelaxedKKTPrimalFeasibilityReceipt,
    stationarity: V1RelaxedKKTStationarityReceipt,
    gap: V1RelaxedKKTGapReceipt,
    iteration: int,
    step: float,
    finite: bool,
    converged: bool,
    termination_reason: str,
    device: str,
) -> V1RelaxedKKTBranchReceipt:
    convergence = V1RelaxedKKTConvergenceReceipt(
        branch=branch,
        converged=converged,
        iterations=iteration,
        maximum_iterations=policy.max_iterations,
        termination_reason=termination_reason,
        backend=policy.backend,
        device=device,
        dtype="float64",
        algorithm_id=RELAXED_KKT_FUSED_CUDA_ALGORITHM_ID,
        scaled_operator_norm_squared_upper_bound=(
            scaled.operator_norm_squared_upper_bound
        ),
        primal_step_size=step,
        dual_step_size=step,
        finite_iterates=finite,
    )
    return V1RelaxedKKTBranchReceipt(
        primal=primal,
        stationarity=stationarity,
        gap=gap,
        convergence=convergence,
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
    )


def _solve_branches_fused_torch(
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    tensors: _TorchProblem,
    counters: _CUDAExecutionCounters,
    *,
    _retain_private_state: bool = False,
) -> (
    tuple[
        Any,
        Any,
        V1RelaxedKKTBranchReceipt,
        Any,
        Any,
        V1RelaxedKKTBranchReceipt,
    ]
    | _FusedTorchSolveOutcome
):
    """Solve both convex branches with one pair of factor scans per step.

    Only the rank-major multiplication is shared.  Every primal state, linear
    dual, response tail, scale, step size, feasibility repair, convergence
    decision, and gap certificate remains branch-specific.
    """

    if type(_retain_private_state) is not bool:
        raise TypeError("_retain_private_state must be an exact bool")
    torch = tensors.torch
    spectral = V1RelaxedKKTBranch.SPECTRAL
    diagonal = V1RelaxedKKTBranch.DIAGONAL_L1
    scaled_s = _scaled_problem(problem, spectral)
    scaled_d = _scaled_problem(problem, diagonal)
    step_s = (
        1.0
        if scaled_s.operator_norm_squared_upper_bound <= 0.0
        else policy.step_safety / math.sqrt(scaled_s.operator_norm_squared_upper_bound)
    )
    step_d = (
        1.0
        if scaled_d.operator_norm_squared_upper_bound <= 0.0
        else policy.step_safety / math.sqrt(scaled_d.operator_norm_squared_upper_bound)
    )
    route_count = problem.route_count
    block_count = len(problem.block_capacities)

    def zeros() -> Any:
        return torch.zeros(route_count, device=tensors.device, dtype=torch.float64)

    z_s, zbar_s, average_s, best_s = zeros(), zeros(), zeros(), zeros()
    z_d, zbar_d, average_d, best_d = zeros(), zeros(), zeros(), zeros()
    best_low_s = torch.zeros(problem.rank, device=tensors.device, dtype=torch.float64)
    best_low_d = torch.zeros_like(best_low_s)
    best_resource_s = 0.0
    best_resource_d = 0.0
    work_duals = torch.zeros(2, device=tensors.device, dtype=torch.float64)
    block_duals = torch.zeros(
        (block_count, 2), device=tensors.device, dtype=torch.float64
    )
    response_low_duals = torch.zeros(
        (problem.rank, 2), device=tensors.device, dtype=torch.float64
    )
    response_tail_s = zeros()
    response_tail_d = torch.zeros((), device=tensors.device, dtype=torch.float64)
    reduced_s = tensors.objective.clone()
    reduced_d = tensors.objective.clone()
    latest_reduced_s: Any | None = None
    latest_reduced_d: Any | None = None
    latest_s: V1RelaxedKKTBranchReceipt | None = None
    latest_d: V1RelaxedKKTBranchReceipt | None = None
    captured_s: _FusedTorchBranchState | None = None
    captured_d: _FusedTorchBranchState | None = None
    latest_iteration = 1
    finite = True
    sqrt_rho = math.sqrt(problem.residual_spectral_bound)
    scales = torch.as_tensor(
        (scaled_s.response_scale, scaled_d.response_scale),
        device=tensors.device,
        dtype=torch.float64,
    )
    steps = torch.as_tensor(
        (step_s, step_d), device=tensors.device, dtype=torch.float64
    )
    work_bounds = torch.as_tensor(
        (scaled_s.work_bound, scaled_d.work_bound),
        device=tensors.device,
        dtype=torch.float64,
    )
    block_bounds = (
        tensors.block_bounds
        if tensors.block_bounds is not None
        else torch.as_tensor(
            scaled_s.block_bounds,
            device=tensors.device,
            dtype=torch.float64,
        )
    )

    for iteration in range(1, policy.max_iterations + 1):
        latest_iteration = iteration
        zbar_columns = torch.stack((zbar_s, zbar_d), dim=1)
        work_forward = torch.matmul(tensors.work, zbar_columns)
        work_forward = work_forward / torch.as_tensor(
            (scaled_s.work_scale, scaled_d.work_scale),
            device=tensors.device,
            dtype=torch.float64,
        )
        work_duals = torch.clamp(
            work_duals + steps * (work_forward - work_bounds), min=0.0
        )
        if block_count:
            block_rows = torch.stack(
                tuple(
                    torch.sum(zbar_columns[block.start : block.stop, :], dim=0) / scale
                    for block, scale in zip(
                        problem.block_capacities,
                        scaled_s.block_scales,
                        strict=True,
                    )
                ),
                dim=0,
            )
            block_duals = torch.clamp(
                block_duals
                + steps.reshape(1, 2) * (block_rows - block_bounds.reshape(-1, 1)),
                min=0.0,
            )

        low_forward = _factor_forward_columns_torch(tensors, zbar_columns, counters)
        low_values = response_low_duals + low_forward * (steps / scales).reshape(1, 2)
        tail_value_s = response_tail_s + (
            step_s * sqrt_rho * zbar_s / scaled_s.response_scale
        )
        tail_value_d = response_tail_d + (
            step_d * torch.dot(tensors.sqrt_residual, zbar_d) / scaled_d.response_scale
        )
        norm_s = torch.sqrt(
            torch.dot(low_values[:, 0], low_values[:, 0])
            + torch.dot(tail_value_s, tail_value_s)
        )
        norm_d = torch.sqrt(
            torch.dot(low_values[:, 1], low_values[:, 1]) + tail_value_d * tail_value_d
        )
        shrink_s = torch.clamp(
            1.0
            - step_s
            * scaled_s.response_radius
            / torch.clamp(norm_s, min=np.finfo(np.float64).tiny),
            min=0.0,
        )
        shrink_d = torch.clamp(
            1.0
            - step_d
            * scaled_d.response_radius
            / torch.clamp(norm_d, min=np.finfo(np.float64).tiny),
            min=0.0,
        )
        response_low_duals = low_values * torch.stack((shrink_s, shrink_d)).reshape(
            1, 2
        )
        response_tail_s = tail_value_s * shrink_s
        response_tail_d = tail_value_d * shrink_d

        low_adjoint = _factor_adjoint_columns_torch(
            tensors, response_low_duals, counters
        )
        response_adjoint_s = (
            low_adjoint[:, 0] + sqrt_rho * response_tail_s
        ) / scaled_s.response_scale
        response_adjoint_d = (
            low_adjoint[:, 1] + tensors.sqrt_residual * response_tail_d
        ) / scaled_d.response_scale
        linear_adjoint_s = _linear_adjoint_torch(
            tensors,
            problem,
            scaled_s,
            work_duals[0],
            block_duals[:, 0],
        )
        linear_adjoint_d = _linear_adjoint_torch(
            tensors,
            problem,
            scaled_d,
            work_duals[1],
            block_duals[:, 1],
        )
        reduced_s = tensors.objective - linear_adjoint_s - response_adjoint_s
        reduced_d = tensors.objective - linear_adjoint_d - response_adjoint_d
        znew_s = torch.clamp(z_s + step_s * reduced_s, 0.0, 1.0)
        znew_d = torch.clamp(z_d + step_d * reduced_d, 0.0, 1.0)
        znew_s = znew_s * tensors.eligible
        znew_d = znew_d * tensors.eligible
        zbar_s = znew_s + policy.over_relaxation * (znew_s - z_s)
        zbar_d = znew_d + policy.over_relaxation * (znew_d - z_d)
        z_s, z_d = znew_s, znew_d
        average_s = average_s + (z_s - average_s) / float(iteration)
        average_d = average_d + (z_d - average_d) / float(iteration)
        should_check = (
            iteration % policy.check_interval == 0 or iteration == policy.max_iterations
        )
        if not should_check:
            continue

        finite = bool(
            torch.stack(
                (
                    torch.isfinite(z_s).all(),
                    torch.isfinite(z_d).all(),
                    torch.isfinite(zbar_s).all(),
                    torch.isfinite(zbar_d).all(),
                    torch.isfinite(response_low_duals).all(),
                    torch.isfinite(response_tail_s).all(),
                    torch.isfinite(response_tail_d),
                    torch.isfinite(block_duals).all(),
                    torch.isfinite(work_duals).all(),
                )
            )
            .all()
            .item()
        )
        if not finite:
            break

        proposals = torch.stack((z_s, z_d, average_s, average_d), dim=1)
        proposal_lows = _factor_forward_columns_torch(tensors, proposals, counters)
        block_proposals, block_proposals_changed = _block_separable_columns_torch(
            tensors, problem, proposals
        )
        if block_proposals_changed:
            # Unequal per-block scales destroy the validity of the original
            # batched low projection.  One conditional batched scan refreshes
            # all four proposal columns without adding persistent route state.
            counters.block_separable_repair_factor_forward_scan_count += 1
            block_proposal_lows = _factor_forward_columns_torch(
                tensors, block_proposals, counters
            )
        else:
            block_proposal_lows = proposal_lows
        repaired_s, repaired_low_s = _repair_feasible_torch_cached(
            tensors, problem, spectral, proposals[:, 0], proposal_lows[:, 0]
        )
        repaired_d, repaired_low_d = _repair_feasible_torch_cached(
            tensors, problem, diagonal, proposals[:, 1], proposal_lows[:, 1]
        )
        repaired_average_s, repaired_average_low_s = _repair_feasible_torch_cached(
            tensors,
            problem,
            spectral,
            proposals[:, 2],
            proposal_lows[:, 2],
        )
        repaired_average_d, repaired_average_low_d = _repair_feasible_torch_cached(
            tensors,
            problem,
            diagonal,
            proposals[:, 3],
            proposal_lows[:, 3],
        )
        if block_proposals_changed:
            block_repaired_s, block_repaired_low_s = _repair_fq_torch_cached(
                tensors,
                problem,
                spectral,
                block_proposals[:, 0],
                block_proposal_lows[:, 0],
            )
            block_repaired_d, block_repaired_low_d = _repair_fq_torch_cached(
                tensors,
                problem,
                diagonal,
                block_proposals[:, 1],
                block_proposal_lows[:, 1],
            )
            block_repaired_average_s, block_repaired_average_low_s = (
                _repair_fq_torch_cached(
                    tensors,
                    problem,
                    spectral,
                    block_proposals[:, 2],
                    block_proposal_lows[:, 2],
                )
            )
            block_repaired_average_d, block_repaired_average_low_d = (
                _repair_fq_torch_cached(
                    tensors,
                    problem,
                    diagonal,
                    block_proposals[:, 3],
                    block_proposal_lows[:, 3],
                )
            )
        else:
            block_repaired_s, block_repaired_low_s = repaired_s, repaired_low_s
            block_repaired_d, block_repaired_low_d = repaired_d, repaired_low_d
            block_repaired_average_s, block_repaired_average_low_s = (
                repaired_average_s,
                repaired_average_low_s,
            )
            block_repaired_average_d, block_repaired_average_low_d = (
                repaired_average_d,
                repaired_average_low_d,
            )
        repaired_values_t = torch.stack(
            (
                torch.dot(tensors.resource, repaired_s),
                torch.dot(tensors.resource, repaired_d),
                torch.dot(tensors.resource, repaired_average_s),
                torch.dot(tensors.resource, repaired_average_d),
                torch.dot(tensors.resource, block_repaired_s),
                torch.dot(tensors.resource, block_repaired_d),
                torch.dot(tensors.resource, block_repaired_average_s),
                torch.dot(tensors.resource, block_repaired_average_d),
            )
        )
        repaired_values = repaired_values_t.detach().cpu().numpy()
        candidate_s, candidate_low_s, candidate_value_s = (
            repaired_s,
            repaired_low_s,
            float(repaired_values[0]),
        )
        if float(repaired_values[4]) > candidate_value_s:
            candidate_s = block_repaired_s
            candidate_low_s = block_repaired_low_s
            candidate_value_s = float(repaired_values[4])
        candidate_average_s, candidate_average_low_s, candidate_average_value_s = (
            repaired_average_s,
            repaired_average_low_s,
            float(repaired_values[2]),
        )
        if float(repaired_values[6]) > candidate_average_value_s:
            candidate_average_s = block_repaired_average_s
            candidate_average_low_s = block_repaired_average_low_s
            candidate_average_value_s = float(repaired_values[6])
        candidate_d, candidate_low_d, candidate_value_d = (
            repaired_d,
            repaired_low_d,
            float(repaired_values[1]),
        )
        if float(repaired_values[5]) > candidate_value_d:
            candidate_d = block_repaired_d
            candidate_low_d = block_repaired_low_d
            candidate_value_d = float(repaired_values[5])
        candidate_average_d, candidate_average_low_d, candidate_average_value_d = (
            repaired_average_d,
            repaired_average_low_d,
            float(repaired_values[3]),
        )
        if float(repaired_values[7]) > candidate_average_value_d:
            candidate_average_d = block_repaired_average_d
            candidate_average_low_d = block_repaired_average_low_d
            candidate_average_value_d = float(repaired_values[7])
        if candidate_value_s > best_resource_s:
            best_s = candidate_s.clone()
            best_low_s = candidate_low_s.clone()
            best_resource_s = candidate_value_s
        if candidate_average_value_s > best_resource_s:
            best_s = candidate_average_s.clone()
            best_low_s = candidate_average_low_s.clone()
            best_resource_s = candidate_average_value_s
        if candidate_value_d > best_resource_d:
            best_d = candidate_d.clone()
            best_low_d = candidate_low_d.clone()
            best_resource_d = candidate_value_d
        if candidate_average_value_d > best_resource_d:
            best_d = candidate_average_d.clone()
            best_low_d = candidate_average_low_d.clone()
            best_resource_d = candidate_average_value_d
        del (
            block_proposals,
            block_proposal_lows,
            block_repaired_s,
            block_repaired_low_s,
            block_repaired_d,
            block_repaired_low_d,
            block_repaired_average_s,
            block_repaired_average_low_s,
            block_repaired_average_d,
            block_repaired_average_low_d,
        )

        if captured_s is None:
            numeric_s = _torch_receipts_cached(
                tensors,
                problem,
                policy,
                spectral,
                scaled_s,
                best_s,
                best_low_s,
                work_duals[0],
                block_duals[:, 0],
                response_low_duals[:, 0],
                response_tail_s,
                reduced_s,
                counters,
            )
            converged_s = all(
                (
                    iteration >= policy.minimum_iterations,
                    numeric_s[0].feasible,
                    numeric_s[2].within_tolerance,
                    numeric_s[1].normalized_stationarity_gap
                    <= policy.normalized_stationarity_tolerance,
                    numeric_s[1].normalized_complementarity_gap
                    <= policy.normalized_complementarity_tolerance,
                )
            )
            latest_s = _fused_branch_receipt(
                problem=problem,
                policy=policy,
                branch=spectral,
                scaled=scaled_s,
                primal=numeric_s[0],
                stationarity=numeric_s[1],
                gap=numeric_s[2],
                iteration=iteration,
                step=step_s,
                finite=True,
                converged=converged_s,
                termination_reason=(
                    "certified_primal_dual_tolerances_satisfied"
                    if converged_s
                    else "maximum_iterations_exhausted"
                ),
                device=str(tensors.device),
            )
            latest_reduced_s = numeric_s[3]
            if converged_s:
                captured_s = _capture_fused_torch_branch_state(
                    branch=spectral,
                    primal=best_s,
                    primal_low=best_low_s,
                    reduced_advantage=numeric_s[3],
                    work_dual=work_duals[0],
                    block_duals=block_duals[:, 0],
                    response_low_dual=response_low_duals[:, 0],
                    response_tail_dual=response_tail_s,
                    numerical_receipt=latest_s,
                )
        if captured_d is None:
            numeric_d = _torch_receipts_cached(
                tensors,
                problem,
                policy,
                diagonal,
                scaled_d,
                best_d,
                best_low_d,
                work_duals[1],
                block_duals[:, 1],
                response_low_duals[:, 1],
                response_tail_d,
                reduced_d,
                counters,
            )
            converged_d = all(
                (
                    iteration >= policy.minimum_iterations,
                    numeric_d[0].feasible,
                    numeric_d[2].within_tolerance,
                    numeric_d[1].normalized_stationarity_gap
                    <= policy.normalized_stationarity_tolerance,
                    numeric_d[1].normalized_complementarity_gap
                    <= policy.normalized_complementarity_tolerance,
                )
            )
            latest_d = _fused_branch_receipt(
                problem=problem,
                policy=policy,
                branch=diagonal,
                scaled=scaled_d,
                primal=numeric_d[0],
                stationarity=numeric_d[1],
                gap=numeric_d[2],
                iteration=iteration,
                step=step_d,
                finite=True,
                converged=converged_d,
                termination_reason=(
                    "certified_primal_dual_tolerances_satisfied"
                    if converged_d
                    else "maximum_iterations_exhausted"
                ),
                device=str(tensors.device),
            )
            latest_reduced_d = numeric_d[3]
            if converged_d:
                captured_d = _capture_fused_torch_branch_state(
                    branch=diagonal,
                    primal=best_d,
                    primal_low=best_low_d,
                    reduced_advantage=numeric_d[3],
                    work_dual=work_duals[1],
                    block_duals=block_duals[:, 1],
                    response_low_dual=response_low_duals[:, 1],
                    response_tail_dual=response_tail_d,
                    numerical_receipt=latest_d,
                )
        if captured_s is not None and captured_d is not None:
            break

    if not finite:
        if latest_s is None or latest_reduced_s is None:
            numeric_s = _torch_receipts_cached(
                tensors,
                problem,
                policy,
                spectral,
                scaled_s,
                best_s,
                best_low_s,
                torch.zeros_like(work_duals[0]),
                torch.zeros_like(block_duals[:, 0]),
                torch.zeros_like(response_low_duals[:, 0]),
                torch.zeros_like(response_tail_s),
                tensors.objective,
                counters,
            )
            latest_reduced_s = numeric_s[3]
            latest_s = _fused_branch_receipt(
                problem=problem,
                policy=policy,
                branch=spectral,
                scaled=scaled_s,
                primal=numeric_s[0],
                stationarity=numeric_s[1],
                gap=numeric_s[2],
                iteration=latest_iteration,
                step=step_s,
                finite=False,
                converged=False,
                termination_reason="nonfinite_iterate_refusal",
                device=str(tensors.device),
            )
        if latest_d is None or latest_reduced_d is None:
            numeric_d = _torch_receipts_cached(
                tensors,
                problem,
                policy,
                diagonal,
                scaled_d,
                best_d,
                best_low_d,
                torch.zeros_like(work_duals[1]),
                torch.zeros_like(block_duals[:, 1]),
                torch.zeros_like(response_low_duals[:, 1]),
                torch.zeros_like(response_tail_d),
                tensors.objective,
                counters,
            )
            latest_reduced_d = numeric_d[3]
            latest_d = _fused_branch_receipt(
                problem=problem,
                policy=policy,
                branch=diagonal,
                scaled=scaled_d,
                primal=numeric_d[0],
                stationarity=numeric_d[1],
                gap=numeric_d[2],
                iteration=latest_iteration,
                step=step_d,
                finite=False,
                converged=False,
                termination_reason="nonfinite_iterate_refusal",
                device=str(tensors.device),
            )
        latest_s = _fused_branch_receipt(
            problem=problem,
            policy=policy,
            branch=spectral,
            scaled=scaled_s,
            primal=latest_s.primal,
            stationarity=latest_s.stationarity,
            gap=latest_s.gap,
            iteration=latest_iteration,
            step=step_s,
            finite=False,
            converged=False,
            termination_reason="nonfinite_iterate_refusal",
            device=str(tensors.device),
        )
        latest_d = _fused_branch_receipt(
            problem=problem,
            policy=policy,
            branch=diagonal,
            scaled=scaled_d,
            primal=latest_d.primal,
            stationarity=latest_d.stationarity,
            gap=latest_d.gap,
            iteration=latest_iteration,
            step=step_d,
            finite=False,
            converged=False,
            termination_reason="nonfinite_iterate_refusal",
            device=str(tensors.device),
        )
        captured_s = None
        captured_d = None
    if (
        latest_s is None
        or latest_d is None
        or latest_reduced_s is None
        or latest_reduced_d is None
    ):
        raise RuntimeError("fused solver ended without a branch certificate")
    if captured_s is None:
        final_primal_s = best_s
        final_reduced_s = latest_reduced_s
        final_receipt_s = latest_s
    else:
        final_primal_s = captured_s.primal
        final_reduced_s = captured_s.reduced_advantage
        final_receipt_s = captured_s.numerical_receipt
    if captured_d is None:
        final_primal_d = best_d
        final_reduced_d = latest_reduced_d
        final_receipt_d = latest_d
    else:
        final_primal_d = captured_d.primal
        final_reduced_d = captured_d.reduced_advantage
        final_receipt_d = captured_d.numerical_receipt
    outcome = _FusedTorchSolveOutcome(
        spectral_primal=final_primal_s,
        spectral_reduced_advantage=final_reduced_s,
        spectral_receipt=final_receipt_s,
        diagonal_primal=final_primal_d,
        diagonal_reduced_advantage=final_reduced_d,
        diagonal_receipt=final_receipt_d,
        spectral_state=captured_s,
        diagonal_state=captured_d,
    )
    if _retain_private_state:
        return outcome
    # Preserve the exact historical private return type and six-value order
    # for reference/test-double callers that do not request the new seam.
    return tuple(outcome)


def _torch_exact_lexicographic_order(
    tensors: _TorchProblem,
    relaxed: Any,
    reduced: Any,
    indices: Any,
    *,
    counters: _CUDAExecutionCounters | None = None,
) -> Any:
    """Order one bounded index set by the registered four-component key."""

    torch = tensors.torch
    if counters is not None:
        count = int(indices.numel())
        counters.key_scan_count += 4
        counters.key_route_visit_count += 4 * count
    # Stable least-to-most-significant sorts implement exactly
    # (-relaxed, -reduced, work/resource, physical_index).  The physical sort
    # is not optional: it makes every exact tie independent of tile and device
    # scheduling.
    ordered = torch.sort(indices, stable=True).values
    ratio = tensors.work[ordered] / tensors.resource[ordered]
    relative = torch.argsort(ratio, descending=False, stable=True)
    ordered = ordered[relative]
    relative = torch.argsort(reduced[ordered], descending=True, stable=True)
    ordered = ordered[relative]
    relative = torch.argsort(relaxed[ordered], descending=True, stable=True)
    return ordered[relative]


def _torch_lexicographic_topk_from_indices(
    tensors: _TorchProblem,
    relaxed: Any,
    reduced: Any,
    indices: Any,
    target: int,
    *,
    counters: _CUDAExecutionCounters | None = None,
) -> Any:
    """Return the exact top-``target`` subset of an explicit index set.

    Threshold reconstruction removes ``topk``'s unspecified tie membership.
    The returned set is unordered; the physical-index key resolves the final
    equality face exactly.  Callers keep ``indices`` bounded by one tile or by
    the registered endpoint count.
    """

    torch = tensors.torch
    active = indices
    selected: list[Any] = []
    remaining = min(target, int(active.numel()))
    for key_index, largest in ((0, True), (1, True), (2, False)):
        if remaining <= 0 or int(active.numel()) <= remaining:
            break
        if counters is not None:
            counters.key_scan_count += 1
            counters.key_route_visit_count += int(active.numel())
        if key_index == 0:
            active_values = relaxed[active]
        elif key_index == 1:
            active_values = reduced[active]
        else:
            # Compute ratios only for the bounded active equality face.  A
            # route-scale ratio temporary here would defeat tiled workspace.
            active_values = tensors.work[active] / tensors.resource[active]
        threshold = torch.topk(
            active_values, remaining, largest=largest, sorted=False
        ).values
        threshold_value = torch.min(threshold) if largest else torch.max(threshold)
        strict_mask = (
            active_values > threshold_value
            if largest
            else active_values < threshold_value
        )
        strict = active[strict_mask]
        if int(strict.numel()):
            selected.append(strict)
            remaining -= int(strict.numel())
        active = active[active_values == threshold_value]
    if remaining > 0:
        if counters is not None:
            counters.key_scan_count += 1
            counters.key_route_visit_count += int(active.numel())
        if int(active.numel()) <= remaining:
            physical = active
        else:
            # Physical indices are unique, so unlike the floating-point key
            # faces this final top-k has no ambiguous threshold membership.
            physical = torch.topk(active, remaining, largest=False, sorted=False).values
        selected.append(physical)
    if not selected:
        return torch.empty(0, device=tensors.device, dtype=torch.int64)
    return torch.cat(tuple(selected), dim=0)


def _torch_tiled_lexicographic_topk(
    tensors: _TorchProblem,
    relaxed: Any,
    reduced: Any,
    *,
    start: int,
    stop: int,
    target: int,
    tile_size: int,
    counters: _CUDAExecutionCounters | None = None,
) -> Any:
    """Return an exact interval top-k with route-axis-independent workspace.

    At most one physical tile is expanded by ``nonzero``.  Each tile is
    reduced to ``target`` candidates before it is merged with the running
    ``target``-sized winner set.  Top-k is associative for a fixed total
    lexicographic order, so this streaming reduction is exactly equivalent to
    materializing and sorting every eligible route in ``[start, stop)``.
    """

    if tile_size <= 0:
        raise ValueError("CUDA rounding tile_size must be positive")
    torch = tensors.torch
    winners = torch.empty(0, device=tensors.device, dtype=torch.int64)
    if target <= 0 or stop <= start:
        return winners
    for tile_start in range(start, stop, tile_size):
        tile_stop = min(stop, tile_start + tile_size)
        relative = torch.nonzero(
            tensors.eligible[tile_start:tile_stop], as_tuple=False
        ).reshape(-1)
        if int(relative.numel()) == 0:
            continue
        active = relative + tile_start
        tile_winners = _torch_lexicographic_topk_from_indices(
            tensors,
            relaxed,
            reduced,
            active,
            target,
            counters=counters,
        )
        merged = torch.cat((winners, tile_winners), dim=0)
        winners = _torch_lexicographic_topk_from_indices(
            tensors,
            relaxed,
            reduced,
            merged,
            target,
            counters=counters,
        )
    return winners


def _torch_rounded_order_device(
    tensors: _TorchProblem,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    relaxed: Any,
    reduced: Any,
    maximum_count: int,
    *,
    tile_size: int = _CUDA_ROUNDING_TILE_SIZE,
    counters: _CUDAExecutionCounters | None = None,
) -> Any:
    """Return the exact bounded endpoint order on the tensor device.

    Keeping this seam separate from :func:`_torch_rounded_order` lets the
    explicit shell-label mode consume the order without a route-scale int64
    host transfer.  The legacy helper below preserves its historical D2H
    result and byte accounting exactly.
    """

    del policy  # Kept in the private signature for a backward-compatible seam.
    torch = tensors.torch
    if maximum_count <= 0:
        return torch.empty(0, device=tensors.device, dtype=torch.int64)
    if tile_size <= 0:
        raise ValueError("CUDA rounding tile_size must be positive")

    if not problem.block_capacities:
        accepted = _torch_tiled_lexicographic_topk(
            tensors,
            relaxed,
            reduced,
            start=0,
            stop=problem.route_count,
            target=maximum_count,
            tile_size=tile_size,
            counters=counters,
        )
    else:
        # Capacity filtering commutes with the global registered order: a
        # legal prefix can contain only the first ``capacity`` routes of each
        # block.  It is also sufficient to retain ``maximum_count`` from any
        # one block.  Both the per-block and cross-block merges are therefore
        # exact while their candidate vectors stay bounded by the endpoint.
        accepted = torch.empty(0, device=tensors.device, dtype=torch.int64)
        for block, eligible_count in zip(
            problem.block_capacities,
            tensors.block_eligible_counts,
            strict=True,
        ):
            block_target = min(
                maximum_count,
                block.removal_capacity,
                eligible_count,
            )
            if block_target <= 0:
                continue
            block_winners = _torch_tiled_lexicographic_topk(
                tensors,
                relaxed,
                reduced,
                start=block.start,
                stop=block.stop,
                target=block_target,
                tile_size=tile_size,
                counters=counters,
            )
            merged = torch.cat((accepted, block_winners), dim=0)
            accepted = _torch_lexicographic_topk_from_indices(
                tensors,
                relaxed,
                reduced,
                merged,
                maximum_count,
                counters=counters,
            )

    ordered = _torch_exact_lexicographic_order(
        tensors, relaxed, reduced, accepted, counters=counters
    )[:maximum_count]
    return ordered


def _build_v1_relaxed_kkt_canonical_key_intervals_cuda(
    *,
    problem: V1RelaxedKKTProblem,
    state: _FusedTorchBranchState,
    tensors: _TorchProblem,
    geometry_uncertainty: V1RelaxedKKTGeometryUncertaintySet,
) -> tuple[Any, Any]:
    """Build outward three-key intervals from KKT-owned CUDA state.

    The component radii retain their distinct semantics: relaxed-gate
    arithmetic, reduced-advantage face perturbation, and corrected-work ratio
    uncertainty are not collapsed into one heuristic radius.
    """

    if type(problem) is not V1RelaxedKKTProblem:
        raise TypeError("canonical key intervals require an exact problem")
    if type(state) is not _FusedTorchBranchState:
        raise TypeError("canonical key intervals require exact branch state")
    if type(tensors) is not _TorchProblem:
        raise TypeError("canonical key intervals require exact torch tensors")
    if type(geometry_uncertainty) is not V1RelaxedKKTGeometryUncertaintySet:
        raise TypeError("canonical key intervals require exact uncertainty")
    if (
        geometry_uncertainty.problem_hash != problem.problem_hash
        or geometry_uncertainty.route_axis_hash != problem.route_axis_hash
        or geometry_uncertainty.source_geometry_hash != problem.source_geometry_hash
        or not geometry_uncertainty.source_derived_from_registered_geometry_receipts
    ):
        _refuse(
            V1RelaxedKKTRefusalCode.CUDA_THEOREM_STATE_LEASE_UNAVAILABLE,
            "canonical key uncertainty does not bind the solved KKT chart",
        )
    work_dual = state.numerical_receipt.stationarity.work_dual
    gate_radius = geometry_uncertainty.relaxed_gate_numerical_linf_radius
    reduced_radius = geometry_uncertainty.finite_population_face_radius(
        work_dual
    ) + geometry_uncertainty.deterministic_face_radius(work_dual)
    ratio_radius = (
        geometry_uncertainty.corrected_work_ratio_sampling_linf_radius
        + geometry_uncertainty.corrected_work_ratio_deterministic_linf_radius
    )
    if not all(
        math.isfinite(value) and value >= 0.0
        for value in (gate_radius, reduced_radius, ratio_radius)
    ):
        _refuse(
            V1RelaxedKKTRefusalCode.CUDA_THEOREM_STATE_LEASE_UNAVAILABLE,
            "canonical key uncertainty radius is not finite",
        )
    torch = tensors.torch
    lower = torch.empty(
        (problem.route_count, 3),
        device=tensors.device,
        dtype=torch.float64,
    )
    upper = torch.empty_like(lower)
    negative_infinity = torch.tensor(
        -math.inf,
        device=tensors.device,
        dtype=torch.float64,
    )
    positive_infinity = torch.tensor(
        math.inf,
        device=tensors.device,
        dtype=torch.float64,
    )
    # Populate the two final buffers columnwise.  This avoids a third
    # route-by-three nominal tensor and keeps peak key construction at the two
    # retained interval tensors plus scalar temporaries.
    for component, source, radius_value in (
        (0, state.primal, gate_radius),
        (1, state.reduced_advantage, reduced_radius),
    ):
        torch.neg(source, out=lower[:, component])
        upper[:, component].copy_(lower[:, component])
        lower[:, component].sub_(radius_value)
        upper[:, component].add_(radius_value)
        torch.nextafter(
            lower[:, component],
            negative_infinity,
            out=lower[:, component],
        )
        torch.nextafter(
            upper[:, component],
            positive_infinity,
            out=upper[:, component],
        )
    torch.div(tensors.work, tensors.resource, out=lower[:, 2])
    lower[:, 2].masked_fill_(~tensors.eligible, 0.0)
    upper[:, 2].copy_(lower[:, 2])
    lower[:, 2].sub_(ratio_radius)
    upper[:, 2].add_(ratio_radius)
    torch.nextafter(lower[:, 2], negative_infinity, out=lower[:, 2])
    torch.nextafter(upper[:, 2], positive_infinity, out=upper[:, 2])
    # Intersect outward arithmetic intervals with exact support constraints:
    # 0 <= z <= 1 and corrected_work/resource >= 0.  These are theorem-owned
    # facts of the registered problem, not empirical tightening.
    lower[:, 0].clamp_(min=-1.0, max=0.0)
    upper[:, 0].clamp_(min=-1.0, max=0.0)
    lower[:, 2].clamp_(min=0.0)
    _require_v1_relaxed_kkt_theorem_cuda_tensor(
        lower,
        name="canonical_key_lower",
        dtype="torch.float64",
        ndim=2,
        device=str(tensors.device),
    )
    _require_v1_relaxed_kkt_theorem_cuda_tensor(
        upper,
        name="canonical_key_upper",
        dtype="torch.float64",
        ndim=2,
        device=str(tensors.device),
    )
    return lower, upper


def _v1_relaxed_kkt_lex_interval_ambiguity_against_seed_cuda(
    *,
    torch: Any,
    lower: Any,
    upper: Any,
    seed_lower: Any,
    seed_upper: Any,
) -> Any:
    """Return routes not strictly ordered from one three-key seed interval."""

    active = torch.ones(
        (int(lower.shape[0]),),
        device=lower.device,
        dtype=torch.bool,
    )
    ambiguous = torch.zeros_like(active)
    for component in range(3):
        left_lo = lower[:, component]
        left_hi = upper[:, component]
        right_lo = seed_lower[component]
        right_hi = seed_upper[component]
        left_precedes = active & (left_hi < right_lo)
        right_precedes = active & (right_hi < left_lo)
        exact_equal = (
            active
            & (left_lo == left_hi)
            & (left_hi == right_lo)
            & (right_lo == right_hi)
        )
        ambiguous |= active & ~(left_precedes | right_precedes | exact_equal)
        active = exact_equal
    # An exact three-component tie is decided only by physical index.  Keep
    # every such route in the bounded shell; the downstream scan must not
    # silently treat it as a strict real-key separation.
    return ambiguous | active


def _derive_v1_relaxed_kkt_bounded_theorem_shell_cuda(
    *,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    state: _FusedTorchBranchState,
    tensors: _TorchProblem,
    canonical_key_lower: Any,
    canonical_key_upper: Any,
    endpoint_counts: tuple[int, ...],
    maximum_shell_routes: int,
    tile_size: int,
) -> tuple[int, ...]:
    """Derive a conservative bounded shell with only bounded index D2H.

    The nominal ladder band between the smallest and largest registered
    endpoints is necessarily variable.  Full-axis CUDA scans then add every
    route whose interval can cross that band, an active-face boundary, or a
    saturated legality-block boundary.  Exceeding the registered shell bound
    refuses before any route-scale host materialization.
    """

    limit = _positive_int(maximum_shell_routes, "maximum_shell_routes")
    tile = _positive_int(tile_size, "tile_size")
    counts = tuple(endpoint_counts)
    minimum_count = counts[0]
    maximum_count = counts[-1]
    if maximum_count > problem.legal_hard_capacity:
        raise ValueError("endpoint count exceeds legal hard capacity")
    order_target = min(problem.legal_hard_capacity, maximum_count + 1)
    order = _torch_rounded_order_device(
        tensors,
        problem,
        policy,
        state.primal,
        state.reduced_advantage,
        order_target,
        tile_size=tile,
    )
    if int(order.numel()) < maximum_count:
        _refuse(
            V1RelaxedKKTRefusalCode.DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE,
            "registered endpoint order is shorter than its frozen count",
        )
    band_start = max(0, minimum_count - 1)
    band_stop = min(int(order.numel()), maximum_count + 1)
    base_shell = order[band_start:band_stop]
    if int(base_shell.numel()) > limit:
        _refuse(
            V1RelaxedKKTRefusalCode.DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE,
            "registered endpoint band exceeds the bounded theorem shell",
        )

    # Capacity filtering can change a high fixed-prefix route without crossing
    # a global endpoint boundary.  Add the saturated block cutoff pair as
    # comparison seeds exactly when that block contributes its full removal
    # capacity to the largest endpoint.
    comparison_seeds: list[Any] = [base_shell]
    selected_order = order[:maximum_count]
    torch = tensors.torch
    for block, eligible_count in zip(
        problem.block_capacities,
        tensors.block_eligible_counts,
        strict=True,
    ):
        capacity = block.removal_capacity
        if capacity <= 0 or eligible_count <= capacity:
            continue
        selected_from_block = torch.sum(
            (selected_order >= block.start) & (selected_order < block.stop)
        )
        if int(selected_from_block.detach().cpu().item()) != capacity:
            continue
        block_pool = _torch_tiled_lexicographic_topk(
            tensors,
            state.primal,
            state.reduced_advantage,
            start=block.start,
            stop=block.stop,
            target=capacity + 1,
            tile_size=tile,
        )
        block_order = _torch_exact_lexicographic_order(
            tensors,
            state.primal,
            state.reduced_advantage,
            block_pool,
        )
        comparison_seeds.append(block_order[capacity - 1 : capacity + 1])
    fixed_comparison_seeds = torch.unique(
        torch.cat(tuple(comparison_seeds), dim=0),
        sorted=True,
    )

    def scan_shell(seed_axis: Any) -> Any:
        found_chunks: list[Any] = []
        found_count = 0
        for start in range(0, problem.route_count, tile):
            stop = min(problem.route_count, start + tile)
            lower = canonical_key_lower[start:stop]
            upper = canonical_key_upper[start:stop]
            eligible = tensors.eligible[start:stop]
            route_indices = torch.arange(
                start,
                stop,
                device=tensors.device,
                dtype=torch.int64,
            )
            ambiguous = torch.zeros(
                (stop - start,),
                device=tensors.device,
                dtype=torch.bool,
            )
            for seed_offset in range(int(seed_axis.numel())):
                seed = seed_axis[seed_offset]
                ambiguous |= _v1_relaxed_kkt_lex_interval_ambiguity_against_seed_cuda(
                    torch=torch,
                    lower=lower,
                    upper=upper,
                    seed_lower=canonical_key_lower[seed],
                    seed_upper=canonical_key_upper[seed],
                )
            reduced_lower = -upper[:, 1]
            reduced_upper = -lower[:, 1]
            # Bound gates at 0/1 are the ordinary KKT faces, not ambiguity by
            # themselves.  The reduced-advantage sign is the complementary-
            # slack witness; only a reduced interval touching zero enters B.
            ambiguous |= (reduced_lower <= 0.0) & (reduced_upper >= 0.0)
            in_base_shell = (
                route_indices.reshape(-1, 1) == base_shell.reshape(1, -1)
            ).any(dim=1)
            selected = (
                torch.nonzero(
                    eligible & (ambiguous | in_base_shell),
                    as_tuple=False,
                ).reshape(-1)
                + start
            )
            selected_count = int(selected.numel())
            if found_count + selected_count > limit:
                _refuse(
                    V1RelaxedKKTRefusalCode.DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE,
                    "canonical key/face ambiguity exceeds the bounded theorem shell",
                )
            if selected_count:
                found_chunks.append(selected)
                found_count += selected_count
        if not found_chunks:
            _refuse(
                V1RelaxedKKTRefusalCode.DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE,
                "decision-level theorem shell is empty",
            )
        result = torch.unique(torch.cat(tuple(found_chunks), dim=0), sorted=True)
        if int(result.numel()) > limit:
            _refuse(
                V1RelaxedKKTRefusalCode.DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE,
                "deduplicated theorem shell exceeds its registered bound",
            )
        return result

    # Close the overlap relation transitively.  This prevents a route that is
    # strict relative to the nominal boundary but ambiguous relative to a
    # newly admitted shell route from leaking into the downstream outside set.
    shell_tensor: Any | None = None
    seed_axis = fixed_comparison_seeds
    for _ in range(limit + 1):
        expanded = scan_shell(seed_axis)
        if shell_tensor is not None and torch.equal(expanded, shell_tensor):
            shell_tensor = expanded
            break
        shell_tensor = expanded
        seed_axis = torch.unique(
            torch.cat((fixed_comparison_seeds, shell_tensor), dim=0),
            sorted=True,
        )
    else:  # pragma: no cover - each strict iteration adds a bounded route.
        _refuse(
            V1RelaxedKKTRefusalCode.DECISION_LEVEL_SHELL_INVARIANCE_UNAVAILABLE,
            "bounded theorem-shell closure did not stabilize",
        )
    if shell_tensor is None:  # pragma: no cover - loop executes at least once.
        raise RuntimeError("theorem-shell closure produced no tensor")
    # The sole index-vector transfer is bounded by ``maximum_shell_routes``.
    return tuple(int(value) for value in shell_tensor.detach().cpu().tolist())


def _torch_rounded_order(
    tensors: _TorchProblem,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    relaxed: Any,
    reduced: Any,
    maximum_count: int,
    *,
    tile_size: int = _CUDA_ROUNDING_TILE_SIZE,
    counters: _CUDAExecutionCounters | None = None,
) -> np.ndarray:
    ordered = _torch_rounded_order_device(
        tensors,
        problem,
        policy,
        relaxed,
        reduced,
        maximum_count,
        tile_size=tile_size,
        counters=counters,
    )
    # This is the only index-vector D2H transfer in rounding, and its length is
    # bounded by the largest pre-registered candidate endpoint rather than by
    # route count or by a legality retry pool.
    result = ordered.detach().cpu().numpy()
    if counters is not None:
        counters.bulk_d2h_tensor_payload_bytes += int(result.nbytes)
    return _readonly_copy(result, dtype=np.dtype(np.int64))


def _torch_scalar(value: Any) -> float:
    """Materialize one tensor scalar as a finite Python float."""

    result = float(value.detach().cpu().item())
    if not math.isfinite(result):
        raise ValueError("torch scalar is not finite")
    return result


def _source_support_margin_for_values_torch(
    values: Any,
    *,
    capacity: int,
) -> Any:
    """Return a sufficient strict support-face margin for one capacity block."""

    count = int(values.numel())
    if count <= 0:
        return values.new_tensor(math.inf)
    cap = min(_nonnegative_int(capacity, "support capacity"), count)
    if cap <= 0:
        maximum = values.max()
        return values.new_tensor(0.0) if bool(maximum > 0.0) else -maximum
    positive = values[values > 0.0]
    positive_count = int(positive.numel())
    if positive_count <= 0:
        return -values.max()
    selected_count = min(cap, positive_count)
    if selected_count < cap:
        positive_floor = positive.min()
        nonpositive = values[values <= 0.0]
        if int(nonpositive.numel()):
            negative_gap = -nonpositive.max()
            return values.new_tensor(0.0).maximum(positive_floor.minimum(negative_gap))
        return values.new_tensor(0.0).maximum(positive_floor)
    selected = values.topk(selected_count, largest=True, sorted=False).values
    selected_floor = selected.min()
    if selected_count >= count:
        return values.new_tensor(0.0).maximum(selected_floor)
    threshold = selected_floor
    excluded = values[values < threshold]
    if int(excluded.numel()) == 0:
        return values.new_tensor(0.0)
    runner = excluded.max()
    return values.new_tensor(0.0).maximum((threshold - runner).minimum(threshold))


def _torch_source_face_margin(
    *,
    tensors: _TorchProblem,
    problem: V1RelaxedKKTProblem,
    reduced: Any,
) -> tuple[float, str]:
    """Compute a sufficient normal-cone support-face margin on device.

    The margin is source-derived from the private reduced-advantage vector.
    It certifies stability of the routewise support face used by the KKT
    stationarity receipt, not hard-mask optimality and not nonlinear F/Q/R
    feasibility.
    """

    torch = tensors.torch
    margins: list[Any] = []
    block_receipts: list[dict[str, Any]] = []
    if not problem.block_capacities:
        values = reduced[tensors.eligible]
        margins.append(
            _source_support_margin_for_values_torch(
                values,
                capacity=int(values.numel()),
            )
        )
        block_receipts.append(
            {
                "block": "__unconstrained__",
                "eligible_count": int(values.numel()),
                "capacity": int(values.numel()),
            }
        )
    else:
        for block in problem.block_capacities:
            eligible = tensors.eligible[block.start : block.stop]
            values = reduced[block.start : block.stop][eligible]
            margins.append(
                _source_support_margin_for_values_torch(
                    values,
                    capacity=block.removal_capacity,
                )
            )
            block_receipts.append(
                {
                    "block_id": block.block_id,
                    "eligible_count": int(values.numel()),
                    "capacity": block.removal_capacity,
                }
            )
    if not margins:
        margin_t = torch.zeros((), device=tensors.device, dtype=torch.float64)
    else:
        margin_t = torch.min(torch.stack(tuple(margins)))
    margin = _torch_scalar(torch.clamp(margin_t, min=0.0))
    return (
        margin,
        content_hash(
            {
                "problem_hash": problem.problem_hash,
                "route_axis_hash": problem.route_axis_hash,
                "source_geometry_hash": problem.source_geometry_hash,
                "margin": margin,
                "block_receipts": tuple(block_receipts),
                "source": "private_reduced_advantage_support_face",
            },
            prefix="partition-free-v1-relaxed-kkt-source-face-",
        ),
    )


def _lexicographic_pair_margin_torch(
    tensors: _TorchProblem,
    relaxed: Any,
    reduced: Any,
    winner: Any,
    runner: Any,
) -> float:
    """Return the first positive real-key margin for the registered order.

    A physical-index-only tie returns zero: any nonzero real-key uncertainty
    could flip such a pair, so the scalar legacy witness schema must refuse
    rather than treating the deterministic physical tiebreak as a robust
    floating margin.
    """

    w = int(winner.detach().cpu().item())
    r = int(runner.detach().cpu().item())
    winner_relaxed = _torch_scalar(relaxed[w])
    runner_relaxed = _torch_scalar(relaxed[r])
    if winner_relaxed > runner_relaxed:
        return winner_relaxed - runner_relaxed
    if winner_relaxed < runner_relaxed:
        return 0.0
    winner_reduced = _torch_scalar(reduced[w])
    runner_reduced = _torch_scalar(reduced[r])
    if winner_reduced > runner_reduced:
        return winner_reduced - runner_reduced
    if winner_reduced < runner_reduced:
        return 0.0
    winner_ratio = _torch_scalar(tensors.work[w] / tensors.resource[w])
    runner_ratio = _torch_scalar(tensors.work[r] / tensors.resource[r])
    if winner_ratio < runner_ratio:
        return runner_ratio - winner_ratio
    if winner_ratio > runner_ratio:
        return 0.0
    return 0.0


def _torch_endpoint_boundary_margins(
    *,
    tensors: _TorchProblem,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    branch: V1RelaxedKKTBranch,
    relaxed: Any,
    reduced: Any,
    ladder: V1RelaxedKKTRoundedLadder,
    tile_size: int,
) -> tuple[float, float, str, str, str]:
    """Replay bounded rounded-order boundaries without route-scale host keys."""

    if type(ladder) is not V1RelaxedKKTRoundedLadder:
        raise TypeError("endpoint boundary margins require an exact ladder")
    if ladder.branch is not branch:
        _refuse(
            V1RelaxedKKTRefusalCode.ADAPTIVE_RANK_WITNESS_SOURCE_MISMATCH,
            "rounded ladder belongs to another branch",
        )
    counts = tuple(candidate.prefix_count for candidate in ladder.candidates)
    if not counts:
        _refuse(
            V1RelaxedKKTRefusalCode.ADAPTIVE_RANK_WITNESS_BOUNDARY_UNAVAILABLE,
            "adaptive-rank witness requires at least one rounded endpoint",
        )
    maximum = counts[-1]
    if maximum <= 0:
        _refuse(
            V1RelaxedKKTRefusalCode.ADAPTIVE_RANK_WITNESS_BOUNDARY_UNAVAILABLE,
            "adaptive-rank witness cannot certify an empty endpoint",
        )
    replay_count = min(problem.legal_hard_capacity, maximum + 1)
    replay_order = _torch_rounded_order_device(
        tensors,
        problem,
        policy,
        relaxed,
        reduced,
        replay_count,
        tile_size=tile_size,
        counters=None,
    )
    replay_prefix = replay_order[:maximum].detach().cpu().numpy()
    if _array_digest(replay_prefix) != _array_digest(ladder.ordered_route_indices):
        _refuse(
            V1RelaxedKKTRefusalCode.ADAPTIVE_RANK_WITNESS_SOURCE_MISMATCH,
            "private solver state does not replay the rounded ladder source order",
        )
    global_margins: list[float] = []
    symbolic_global_margins: list[float | str] = []
    for count in counts:
        if int(replay_order.numel()) > count:
            margin = _lexicographic_pair_margin_torch(
                tensors,
                relaxed,
                reduced,
                replay_order[count - 1],
                replay_order[count],
            )
            global_margins.append(margin)
            symbolic_global_margins.append(margin)
        else:
            symbolic_global_margins.append("no_legal_runner")
    per_block_margins: list[float] = []
    symbolic_per_block_margins: list[float | str] = []
    block_receipts: list[dict[str, Any]] = []
    if problem.block_capacities:
        labels = tensors.torch.full(
            (problem.route_count,),
            -1,
            device=tensors.device,
            dtype=tensors.torch.int64,
        )
        for position, index in enumerate(replay_order[:maximum]):
            labels[index] = position
        for count in counts:
            selected_prefix = replay_order[:count]
            for block_index, block in enumerate(problem.block_capacities):
                block_selected = selected_prefix[
                    (selected_prefix >= block.start) & (selected_prefix < block.stop)
                ]
                selected_count = int(block_selected.numel())
                eligible_count = int(tensors.block_eligible_counts[block_index])
                if selected_count <= 0:
                    continue
                target = min(
                    selected_count + 1,
                    block.removal_capacity,
                    eligible_count,
                )
                if target <= selected_count:
                    symbolic_per_block_margins.append("block_capacity_saturated")
                    continue
                block_top = _torch_tiled_lexicographic_topk(
                    tensors,
                    relaxed,
                    reduced,
                    start=block.start,
                    stop=block.stop,
                    target=target,
                    tile_size=tile_size,
                    counters=None,
                )
                block_order = _torch_exact_lexicographic_order(
                    tensors,
                    relaxed,
                    reduced,
                    block_top,
                    counters=None,
                )
                selected_replay = block_order[:selected_count]
                if set(
                    int(item) for item in selected_replay.detach().cpu().numpy()
                ) != set(int(item) for item in block_selected.detach().cpu().numpy()):
                    _refuse(
                        V1RelaxedKKTRefusalCode.ADAPTIVE_RANK_WITNESS_SOURCE_MISMATCH,
                        "per-block source replay differs from the rounded endpoint",
                    )
                margin = _lexicographic_pair_margin_torch(
                    tensors,
                    relaxed,
                    reduced,
                    block_order[selected_count - 1],
                    block_order[selected_count],
                )
                per_block_margins.append(margin)
                symbolic_per_block_margins.append(margin)
                block_receipts.append(
                    {
                        "count": count,
                        "block_id": block.block_id,
                        "selected_count": selected_count,
                        "eligible_count": eligible_count,
                    }
                )
        del labels
    all_margins = [*global_margins, *per_block_margins]
    if not all_margins:
        endpoint_margin = float(np.finfo(np.float64).max / 4.0)
    else:
        endpoint_margin = max(0.0, min(all_margins))
    if endpoint_margin == 0.0:
        _refuse(
            V1RelaxedKKTRefusalCode.ADAPTIVE_RANK_WITNESS_BOUNDARY_UNAVAILABLE,
            "registered rounded endpoint boundary has no positive real-key margin",
        )
    selected_margin = endpoint_margin
    return (
        selected_margin,
        endpoint_margin,
        content_hash(
            {
                "problem_hash": problem.problem_hash,
                "branch": branch,
                "counts": counts,
                "ordered_route_indices_hash": _array_digest(
                    ladder.ordered_route_indices
                ),
                "global_margins": tuple(symbolic_global_margins),
                "per_block_margins": tuple(symbolic_per_block_margins),
            },
            prefix="partition-free-v1-relaxed-kkt-endpoint-shell-",
        ),
        content_hash(
            {
                "problem_hash": problem.problem_hash,
                "branch": branch,
                "counts": counts,
                "block_receipts": tuple(block_receipts),
            },
            prefix="partition-free-v1-relaxed-kkt-block-boundary-",
        ),
        content_hash(
            {
                "problem_hash": problem.problem_hash,
                "branch": branch,
                "counts": counts,
                "ordered_route_indices_hash": _array_digest(
                    ladder.ordered_route_indices
                ),
            },
            prefix="partition-free-v1-relaxed-kkt-endpoint-action-axis-",
        ),
    )


def _torch_device_shell_endpoint_boundary_margins(
    *,
    tensors: _TorchProblem,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    branch: V1RelaxedKKTBranch,
    relaxed: Any,
    reduced: Any,
    branch_identity: V1RelaxedKKTDeviceShellBranchIdentityReceipt,
    tile_size: int,
) -> tuple[float, float, str, str, str]:
    """Replay device-shell endpoint boundaries without an ordinary ladder."""

    if type(branch_identity) is not V1RelaxedKKTDeviceShellBranchIdentityReceipt:
        raise TypeError("device-shell endpoint witness requires exact identity")
    if branch_identity.branch is not branch:
        _refuse(
            V1RelaxedKKTRefusalCode.ADAPTIVE_RANK_WITNESS_SOURCE_MISMATCH,
            "device-shell branch identity belongs to another branch",
        )
    frozen = branch_identity.frozen_ladder
    counts = frozen.cumulative_counts
    if not counts:
        _refuse(
            V1RelaxedKKTRefusalCode.ADAPTIVE_RANK_WITNESS_BOUNDARY_UNAVAILABLE,
            "device-shell adaptive witness requires at least one endpoint",
        )
    maximum = counts[-1]
    if maximum <= 0:
        _refuse(
            V1RelaxedKKTRefusalCode.ADAPTIVE_RANK_WITNESS_BOUNDARY_UNAVAILABLE,
            "device-shell adaptive witness cannot certify an empty endpoint",
        )
    replay_count = min(problem.legal_hard_capacity, maximum + 1)
    replay_order = _torch_rounded_order_device(
        tensors,
        problem,
        policy,
        relaxed,
        reduced,
        replay_count,
        tile_size=tile_size,
        counters=None,
    )
    replay_prefix = replay_order[:maximum].detach().cpu().numpy()
    if int(replay_prefix.size) != maximum:
        _refuse(
            V1RelaxedKKTRefusalCode.ADAPTIVE_RANK_WITNESS_SOURCE_MISMATCH,
            "device-shell source replay cannot cover the frozen endpoint axis",
        )
    for count, action in zip(counts, frozen.actions, strict=True):
        observed = RemovedRouteIndexSet.from_indices(
            replay_prefix[:count],
            route_count=problem.route_count,
        ).index_hash
        if observed != action.removed_route_index_hash:
            _refuse(
                V1RelaxedKKTRefusalCode.ADAPTIVE_RANK_WITNESS_SOURCE_MISMATCH,
                "device-shell frozen action axis differs from source rounding",
            )
    global_margins: list[float] = []
    symbolic_global_margins: list[float | str] = []
    for count in counts:
        if int(replay_order.numel()) > count:
            margin = _lexicographic_pair_margin_torch(
                tensors,
                relaxed,
                reduced,
                replay_order[count - 1],
                replay_order[count],
            )
            global_margins.append(margin)
            symbolic_global_margins.append(margin)
        else:
            symbolic_global_margins.append("no_legal_runner")
    per_block_margins: list[float] = []
    symbolic_per_block_margins: list[float | str] = []
    block_receipts: list[dict[str, Any]] = []
    if problem.block_capacities:
        for count in counts:
            selected_prefix = replay_order[:count]
            for block_index, block in enumerate(problem.block_capacities):
                block_selected = selected_prefix[
                    (selected_prefix >= block.start) & (selected_prefix < block.stop)
                ]
                selected_count = int(block_selected.numel())
                eligible_count = int(tensors.block_eligible_counts[block_index])
                if selected_count <= 0:
                    continue
                target = min(
                    selected_count + 1,
                    block.removal_capacity,
                    eligible_count,
                )
                if target <= selected_count:
                    symbolic_per_block_margins.append("block_capacity_saturated")
                    continue
                block_top = _torch_tiled_lexicographic_topk(
                    tensors,
                    relaxed,
                    reduced,
                    start=block.start,
                    stop=block.stop,
                    target=target,
                    tile_size=tile_size,
                    counters=None,
                )
                block_order = _torch_exact_lexicographic_order(
                    tensors,
                    relaxed,
                    reduced,
                    block_top,
                    counters=None,
                )
                selected_replay = block_order[:selected_count]
                if set(
                    int(item) for item in selected_replay.detach().cpu().numpy()
                ) != set(int(item) for item in block_selected.detach().cpu().numpy()):
                    _refuse(
                        V1RelaxedKKTRefusalCode.ADAPTIVE_RANK_WITNESS_SOURCE_MISMATCH,
                        "device-shell per-block source replay differs from endpoint",
                    )
                margin = _lexicographic_pair_margin_torch(
                    tensors,
                    relaxed,
                    reduced,
                    block_order[selected_count - 1],
                    block_order[selected_count],
                )
                per_block_margins.append(margin)
                symbolic_per_block_margins.append(margin)
                block_receipts.append(
                    {
                        "count": count,
                        "block_id": block.block_id,
                        "selected_count": selected_count,
                        "eligible_count": eligible_count,
                    }
                )
    all_margins = [*global_margins, *per_block_margins]
    if not all_margins:
        endpoint_margin = float(np.finfo(np.float64).max / 4.0)
    else:
        endpoint_margin = max(0.0, min(all_margins))
    if endpoint_margin == 0.0:
        _refuse(
            V1RelaxedKKTRefusalCode.ADAPTIVE_RANK_WITNESS_BOUNDARY_UNAVAILABLE,
            "registered device-shell endpoint boundary has no positive real-key margin",
        )
    return (
        endpoint_margin,
        endpoint_margin,
        content_hash(
            {
                "problem_hash": problem.problem_hash,
                "branch": branch,
                "source_rounding_hash": branch_identity.source_rounding_hash,
                "frozen_ladder_receipt_hash": frozen.receipt_hash,
                "counts": counts,
                "global_margins": tuple(symbolic_global_margins),
                "per_block_margins": tuple(symbolic_per_block_margins),
            },
            prefix="partition-free-v1-relaxed-kkt-device-shell-endpoint-shell-",
        ),
        content_hash(
            {
                "problem_hash": problem.problem_hash,
                "branch": branch,
                "source_rounding_hash": branch_identity.source_rounding_hash,
                "frozen_ladder_receipt_hash": frozen.receipt_hash,
                "counts": counts,
                "block_receipts": tuple(block_receipts),
            },
            prefix="partition-free-v1-relaxed-kkt-device-shell-block-boundary-",
        ),
        branch_identity.endpoint_action_axis_hash,
    )


def issue_v1_relaxed_kkt_adaptive_rank_branch_witness_from_state(
    *,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    branch_result: V1RelaxedKKTBranchResult,
    private_state: _FusedTorchBranchState,
    tensors: _TorchProblem,
    geometry_uncertainty: V1RelaxedKKTGeometryUncertaintySet,
    relaxed_solution_sensitivity_certificate: (
        V1RelaxedKKTRobustSolutionSensitivityCertificate | None
    ) = None,
    tile_size: int = _CUDA_ROUNDING_TILE_SIZE,
) -> V1RelaxedKKTAdaptiveRankBranchWitness:
    """Issue an authority-inert, source-derived KKT face-stability witness."""

    if type(problem) is not V1RelaxedKKTProblem:
        raise TypeError("problem must be an exact relaxed KKT problem")
    if type(policy) is not V1RelaxedKKTPolicy:
        raise TypeError("policy must be an exact relaxed KKT policy")
    if type(branch_result) is not V1RelaxedKKTBranchResult:
        raise TypeError("branch_result must be exact")
    if type(private_state) is not _FusedTorchBranchState:
        raise TypeError("private_state must be an exact fused branch state")
    if type(geometry_uncertainty) is not V1RelaxedKKTGeometryUncertaintySet:
        raise TypeError("geometry_uncertainty must be exact")
    if not geometry_uncertainty.source_derived_from_registered_geometry_receipts:
        _refuse(
            V1RelaxedKKTRefusalCode.ADAPTIVE_RANK_WITNESS_SOURCE_MISMATCH,
            "caller-authored scalar uncertainty cannot issue a source-derived "
            "adaptive-rank KKT witness",
        )
    if (
        geometry_uncertainty.problem_hash != problem.problem_hash
        or geometry_uncertainty.route_axis_hash != problem.route_axis_hash
        or geometry_uncertainty.source_geometry_hash != problem.source_geometry_hash
    ):
        _refuse(
            V1RelaxedKKTRefusalCode.ADAPTIVE_RANK_WITNESS_SOURCE_MISMATCH,
            "geometry uncertainty belongs to another KKT coordinate chart",
        )
    if (
        branch_result.branch is not private_state.branch
        or branch_result.receipt is not private_state.numerical_receipt
        or private_state.numerical_receipt.problem_hash != problem.problem_hash
        or private_state.numerical_receipt.policy_hash != policy.policy_hash
    ):
        _refuse(
            V1RelaxedKKTRefusalCode.ADAPTIVE_RANK_WITNESS_SOURCE_MISMATCH,
            "private branch state, branch result, problem, and policy do not close",
        )
    if not branch_result.receipt.convergence.converged:
        _refuse(
            V1RelaxedKKTRefusalCode.ADAPTIVE_RANK_WITNESS_BOUNDARY_UNAVAILABLE,
            "unconverged branch cannot issue an adaptive-rank witness",
        )
    if int(private_state.primal.numel()) != problem.route_count:
        _refuse(
            V1RelaxedKKTRefusalCode.ADAPTIVE_RANK_WITNESS_SOURCE_MISMATCH,
            "private branch state route count differs from the problem",
        )
    face_margin, face_hash = _torch_source_face_margin(
        tensors=tensors,
        problem=problem,
        reduced=private_state.reduced_advantage,
    )
    if face_margin == 0.0:
        _refuse(
            V1RelaxedKKTRefusalCode.ADAPTIVE_RANK_WITNESS_BOUNDARY_UNAVAILABLE,
            "normal-cone support face has no positive source margin",
        )
    (
        selected_margin,
        endpoint_margin,
        endpoint_hash,
        block_boundary_hash,
        endpoint_axis_hash,
    ) = _torch_endpoint_boundary_margins(
        tensors=tensors,
        problem=problem,
        policy=policy,
        branch=private_state.branch,
        relaxed=private_state.primal,
        reduced=private_state.reduced_advantage,
        ladder=branch_result.rounded_ladder,
        tile_size=tile_size,
    )
    raw_work_dual = branch_result.receipt.stationarity.work_dual
    finite_face = geometry_uncertainty.finite_population_face_radius(raw_work_dual)
    deterministic_face = geometry_uncertainty.deterministic_face_radius(raw_work_dual)
    finite_key = geometry_uncertainty.finite_population_key_radius(raw_work_dual)
    deterministic_key = geometry_uncertainty.deterministic_key_radius(raw_work_dual)
    relaxed_problem_perturbation = (
        geometry_uncertainty.relaxed_problem_perturbation_radius(raw_work_dual)
    )
    sensitivity_certificate_hash = (
        _validated_robust_solution_sensitivity_certificate_hash(
            certificate=relaxed_solution_sensitivity_certificate,
            problem=problem,
            policy=policy,
            branch_result=branch_result,
            geometry_uncertainty=geometry_uncertainty,
        )
    )
    return V1RelaxedKKTAdaptiveRankBranchWitness(
        branch=private_state.branch,
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        branch_result_hash=branch_result.result_hash,
        branch_receipt_hash=branch_result.receipt.receipt_hash,
        rounded_ladder_hash=branch_result.rounded_ladder.ladder_hash,
        route_axis_hash=problem.route_axis_hash,
        route_count=problem.route_count,
        relaxed_face_witness_hash=face_hash,
        rounded_endpoint_set_witness_hash=endpoint_hash,
        reduced_advantage_interval_axis_hash=content_hash(
            {
                "problem_hash": problem.problem_hash,
                "branch": private_state.branch,
                "geometry_uncertainty_hash": (geometry_uncertainty.uncertainty_hash),
                "finite_population_face_radius": finite_face,
                "deterministic_face_radius": deterministic_face,
            },
            prefix="partition-free-v1-relaxed-kkt-reduced-interval-axis-",
        ),
        four_key_boundary_interval_axis_hash=content_hash(
            {
                "problem_hash": problem.problem_hash,
                "branch": private_state.branch,
                "geometry_uncertainty_hash": (geometry_uncertainty.uncertainty_hash),
                "finite_population_key_radius": finite_key,
                "deterministic_key_radius": deterministic_key,
            },
            prefix="partition-free-v1-relaxed-kkt-four-key-interval-axis-",
        ),
        block_capacity_boundary_hash=block_boundary_hash,
        endpoint_action_axis_hash=endpoint_axis_hash,
        literal_j_fixed_primal_direction_axis_hash=(
            geometry_uncertainty.literal_j_fixed_primal_direction_axis_hash
        ),
        winning_face_margin_lower_bound=face_margin,
        winning_face_residual_perturbation_upper_bound=deterministic_face,
        selected_vs_runner_up_margin_lower_bound=selected_margin,
        selected_vs_runner_up_residual_perturbation_upper_bound=deterministic_key,
        endpoint_boundary_margin_lower_bound=endpoint_margin,
        endpoint_boundary_residual_perturbation_upper_bound=deterministic_key,
        finite_population_geometry_uncertainty_set_hash=(
            geometry_uncertainty.uncertainty_hash
        ),
        finite_population_face_perturbation_upper_bound=finite_face,
        finite_population_key_perturbation_upper_bound=finite_key,
        relaxed_problem_perturbation_upper_bound=(relaxed_problem_perturbation),
        relaxed_solution_sensitivity_certificate_hash=(sensitivity_certificate_hash),
        source_derived_from_typed_interval_receipts=True,
        _adaptive_rank_witness_issuance_token=(
            _ADAPTIVE_RANK_BRANCH_WITNESS_ISSUANCE_TOKEN
        ),
    )


def issue_v1_relaxed_kkt_adaptive_rank_branch_witness_from_device_shell_state(
    *,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    branch_result: V1RelaxedKKTBranchResult,
    branch_identity: V1RelaxedKKTDeviceShellBranchIdentityReceipt,
    private_state: _FusedTorchBranchState,
    tensors: _TorchProblem,
    geometry_uncertainty: V1RelaxedKKTGeometryUncertaintySet,
    relaxed_solution_sensitivity_certificate: (
        V1RelaxedKKTRobustSolutionSensitivityCertificate | None
    ) = None,
    tile_size: int = _CUDA_ROUNDING_TILE_SIZE,
) -> V1RelaxedKKTAdaptiveRankBranchWitness:
    """Issue a source-derived KKT witness for an action-free shell ladder."""

    if type(problem) is not V1RelaxedKKTProblem:
        raise TypeError("problem must be an exact relaxed KKT problem")
    if type(policy) is not V1RelaxedKKTPolicy:
        raise TypeError("policy must be an exact relaxed KKT policy")
    if type(branch_result) is not V1RelaxedKKTBranchResult:
        raise TypeError("branch_result must be exact")
    if type(branch_identity) is not V1RelaxedKKTDeviceShellBranchIdentityReceipt:
        raise TypeError("branch_identity must be exact")
    if type(private_state) is not _FusedTorchBranchState:
        raise TypeError("private_state must be an exact fused branch state")
    if type(geometry_uncertainty) is not V1RelaxedKKTGeometryUncertaintySet:
        raise TypeError("geometry_uncertainty must be exact")
    if not geometry_uncertainty.source_derived_from_registered_geometry_receipts:
        _refuse(
            V1RelaxedKKTRefusalCode.ADAPTIVE_RANK_WITNESS_SOURCE_MISMATCH,
            "caller-authored scalar uncertainty cannot issue a source-derived "
            "device-shell adaptive-rank KKT witness",
        )
    if (
        geometry_uncertainty.problem_hash != problem.problem_hash
        or geometry_uncertainty.route_axis_hash != problem.route_axis_hash
        or geometry_uncertainty.source_geometry_hash != problem.source_geometry_hash
    ):
        _refuse(
            V1RelaxedKKTRefusalCode.ADAPTIVE_RANK_WITNESS_SOURCE_MISMATCH,
            "geometry uncertainty belongs to another KKT coordinate chart",
        )
    if (
        branch_result.branch is not branch_identity.branch
        or branch_result.branch is not private_state.branch
        or branch_result.receipt is not private_state.numerical_receipt
        or private_state.numerical_receipt.problem_hash != problem.problem_hash
        or private_state.numerical_receipt.policy_hash != policy.policy_hash
        or branch_result.rounded_ladder.candidates
        or branch_result.rounded_ladder.ordered_route_indices.size
        or branch_result.relaxed_gate_values is not None
        or branch_identity.frozen_ladder.route_axis_hash != problem.route_axis_hash
        or branch_identity.frozen_ladder.route_count != problem.route_count
    ):
        _refuse(
            V1RelaxedKKTRefusalCode.ADAPTIVE_RANK_WITNESS_SOURCE_MISMATCH,
            "private branch state, action-free result, and shell identity do not close",
        )
    if not branch_result.receipt.convergence.converged:
        _refuse(
            V1RelaxedKKTRefusalCode.ADAPTIVE_RANK_WITNESS_BOUNDARY_UNAVAILABLE,
            "unconverged device-shell branch cannot issue an adaptive-rank witness",
        )
    if int(private_state.primal.numel()) != problem.route_count:
        _refuse(
            V1RelaxedKKTRefusalCode.ADAPTIVE_RANK_WITNESS_SOURCE_MISMATCH,
            "private branch state route count differs from the problem",
        )
    face_margin, face_hash = _torch_source_face_margin(
        tensors=tensors,
        problem=problem,
        reduced=private_state.reduced_advantage,
    )
    if face_margin == 0.0:
        _refuse(
            V1RelaxedKKTRefusalCode.ADAPTIVE_RANK_WITNESS_BOUNDARY_UNAVAILABLE,
            "normal-cone support face has no positive source margin",
        )
    (
        selected_margin,
        endpoint_margin,
        endpoint_hash,
        block_boundary_hash,
        endpoint_axis_hash,
    ) = _torch_device_shell_endpoint_boundary_margins(
        tensors=tensors,
        problem=problem,
        policy=policy,
        branch=private_state.branch,
        relaxed=private_state.primal,
        reduced=private_state.reduced_advantage,
        branch_identity=branch_identity,
        tile_size=tile_size,
    )
    raw_work_dual = branch_result.receipt.stationarity.work_dual
    finite_face = geometry_uncertainty.finite_population_face_radius(raw_work_dual)
    deterministic_face = geometry_uncertainty.deterministic_face_radius(raw_work_dual)
    finite_key = geometry_uncertainty.finite_population_key_radius(raw_work_dual)
    deterministic_key = geometry_uncertainty.deterministic_key_radius(raw_work_dual)
    relaxed_problem_perturbation = (
        geometry_uncertainty.relaxed_problem_perturbation_radius(raw_work_dual)
    )
    sensitivity_certificate_hash = (
        _validated_robust_solution_sensitivity_certificate_hash(
            certificate=relaxed_solution_sensitivity_certificate,
            problem=problem,
            policy=policy,
            branch_result=branch_result,
            geometry_uncertainty=geometry_uncertainty,
        )
    )
    return V1RelaxedKKTAdaptiveRankBranchWitness(
        branch=private_state.branch,
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        branch_result_hash=branch_result.result_hash,
        branch_receipt_hash=branch_result.receipt.receipt_hash,
        rounded_ladder_hash=branch_identity.source_rounding_hash,
        route_axis_hash=problem.route_axis_hash,
        route_count=problem.route_count,
        relaxed_face_witness_hash=face_hash,
        rounded_endpoint_set_witness_hash=endpoint_hash,
        reduced_advantage_interval_axis_hash=content_hash(
            {
                "problem_hash": problem.problem_hash,
                "branch": private_state.branch,
                "geometry_uncertainty_hash": (geometry_uncertainty.uncertainty_hash),
                "finite_population_face_radius": finite_face,
                "deterministic_face_radius": deterministic_face,
            },
            prefix="partition-free-v1-relaxed-kkt-reduced-interval-axis-",
        ),
        four_key_boundary_interval_axis_hash=content_hash(
            {
                "problem_hash": problem.problem_hash,
                "branch": private_state.branch,
                "geometry_uncertainty_hash": (geometry_uncertainty.uncertainty_hash),
                "finite_population_key_radius": finite_key,
                "deterministic_key_radius": deterministic_key,
            },
            prefix="partition-free-v1-relaxed-kkt-four-key-interval-axis-",
        ),
        block_capacity_boundary_hash=block_boundary_hash,
        endpoint_action_axis_hash=endpoint_axis_hash,
        literal_j_fixed_primal_direction_axis_hash=(
            geometry_uncertainty.literal_j_fixed_primal_direction_axis_hash
        ),
        winning_face_margin_lower_bound=face_margin,
        winning_face_residual_perturbation_upper_bound=deterministic_face,
        selected_vs_runner_up_margin_lower_bound=selected_margin,
        selected_vs_runner_up_residual_perturbation_upper_bound=deterministic_key,
        endpoint_boundary_margin_lower_bound=endpoint_margin,
        endpoint_boundary_residual_perturbation_upper_bound=deterministic_key,
        finite_population_geometry_uncertainty_set_hash=(
            geometry_uncertainty.uncertainty_hash
        ),
        finite_population_face_perturbation_upper_bound=finite_face,
        finite_population_key_perturbation_upper_bound=finite_key,
        relaxed_problem_perturbation_upper_bound=(relaxed_problem_perturbation),
        relaxed_solution_sensitivity_certificate_hash=(sensitivity_certificate_hash),
        source_derived_from_typed_interval_receipts=True,
        _adaptive_rank_witness_issuance_token=(
            _ADAPTIVE_RANK_BRANCH_WITNESS_ISSUANCE_TOKEN
        ),
    )


def _torch_first_inclusion_labels_from_order(
    tensors: _TorchProblem,
    ordered: Any,
    counts: tuple[int, ...],
) -> Any:
    """Encode one device-resident uint8 first-inclusion label axis.

    Label zero means the route is outside every registered endpoint; label
    ``k`` means it first appears at cumulative endpoint ``k``.  No cumulative
    endpoint mask is materialized.
    """

    normalized_counts = tuple(
        _positive_int(value, "device-shell endpoint count") for value in counts
    )
    if not normalized_counts or tuple(sorted(set(normalized_counts))) != (
        normalized_counts
    ):
        raise ValueError("device-shell endpoint counts must be strictly increasing")
    if len(normalized_counts) > DEVICE_SHELL_MAXIMUM_ENDPOINTS:
        _refuse(
            V1RelaxedKKTRefusalCode.CUDA_DEVICE_SHELL_IDENTITY_UNAVAILABLE,
            "uint8 first-inclusion labels support at most 255 endpoints",
        )
    if int(ordered.numel()) < normalized_counts[-1]:
        _refuse(
            V1RelaxedKKTRefusalCode.CUDA_DEVICE_SHELL_IDENTITY_UNAVAILABLE,
            "device order does not cover every registered endpoint",
        )
    torch = tensors.torch
    labels = torch.zeros(
        tensors.resource.numel(), device=tensors.device, dtype=torch.uint8
    )
    previous = 0
    for endpoint_label, count in enumerate(normalized_counts, start=1):
        labels[ordered[previous:count]] = endpoint_label
        previous = count
    if not labels.is_contiguous():  # pragma: no cover - torch zeros is contiguous.
        labels = labels.contiguous()
    return labels


def _device_shell_source_rounding_hash(
    *,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    branch: V1RelaxedKKTBranch,
    branch_receipt: V1RelaxedKKTBranchReceipt,
    parent_mask: HardMask,
    counts: tuple[int, ...],
    execution_design: V1RelaxedKKTDeviceShellExecutionDesign,
) -> str:
    return content_hash(
        {
            "problem_hash": problem.problem_hash,
            "policy_hash": policy.policy_hash,
            "branch": branch,
            "branch_receipt_hash": branch_receipt.receipt_hash,
            "parent_mask_hash": parent_mask.mask_hash,
            "counts": counts,
            "rounding_key_id": RELAXED_KKT_ROUNDING_KEY_ID,
            "device_shell_execution_design_hash": execution_design.design_hash,
        },
        prefix="partition-free-v1-relaxed-kkt-device-shell-rounding-source-",
    )


def _freeze_torch_device_shell_branch(
    *,
    tensors: _TorchProblem,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    branch: V1RelaxedKKTBranch,
    branch_receipt: V1RelaxedKKTBranchReceipt,
    relaxed: Any,
    reduced: Any,
    parent_mask: HardMask,
    counts: tuple[int, ...],
    execution_design: V1RelaxedKKTDeviceShellExecutionDesign,
    counters: _CUDAExecutionCounters,
) -> tuple[
    V1DeviceShellLabelHandle,
    V1RelaxedKKTDeviceShellBranchIdentityReceipt,
]:
    """Freeze one branch identity while retaining its labels on device."""

    _synchronize_torch_device(tensors)
    rounding_started = time.perf_counter()
    order = _torch_rounded_order_device(
        tensors,
        problem,
        policy,
        relaxed,
        reduced,
        counts[-1],
        tile_size=execution_design.cuda_execution_design.rounding_tile_size,
        counters=counters,
    )
    labels = _torch_first_inclusion_labels_from_order(tensors, order, counts)
    _synchronize_torch_device(tensors)
    del order
    counters.warm_rounding_seconds += time.perf_counter() - rounding_started
    handle = V1DeviceShellLabelHandle.from_torch_cuda(labels)
    source_hash = _device_shell_source_rounding_hash(
        problem=problem,
        policy=policy,
        branch=branch,
        branch_receipt=branch_receipt,
        parent_mask=parent_mask,
        counts=counts,
        execution_design=execution_design,
    )
    blocks = tuple(
        V1DeviceShellCompactBlockCapacity.from_registered_block(block)
        for block in problem.block_capacities
    )
    replay_started = time.perf_counter()
    try:
        frozen = freeze_v1_device_shell_ladder(
            handle=handle,
            parent_mask=parent_mask,
            eligible_mask=problem.eligible_mask,
            block_capacities=blocks,
            endpoint_count=len(counts),
            source_ladder_hash=source_hash,
            physical_chunk_size=execution_design.physical_chunk_size,
            allow_unconstrained_blocks=execution_design.allow_unconstrained_blocks,
        )
    except Exception:
        handle.evict()
        raise
    # ``freeze_v1_device_shell_ladder`` makes exactly one bounded physical-axis
    # scan.  Its CUDA handle copies one uint8 slab at a time; scalar driver
    # bookkeeping remains outside the bulk-payload convention.
    counters.bulk_d2h_tensor_payload_bytes += problem.route_count
    counters.warm_replay_seconds += time.perf_counter() - replay_started
    return (
        handle,
        V1RelaxedKKTDeviceShellBranchIdentityReceipt(
            branch=branch,
            source_rounding_hash=source_hash,
            frozen_ladder=frozen,
        ),
    )


def _hard_candidates_from_order_torch(
    tensors: _TorchProblem,
    problem: V1RelaxedKKTProblem,
    branch: V1RelaxedKKTBranch,
    order: np.ndarray,
    counts: tuple[int, ...],
    *,
    counters: _CUDAExecutionCounters | None = None,
) -> tuple[V1RelaxedKKTHardCandidate, ...]:
    torch = tensors.torch
    order_device = torch.tensor(order, device=tensors.device, dtype=torch.int64)
    if counters is not None:
        counters.bulk_h2d_tensor_payload_bytes += int(order.nbytes)
    h_sum = torch.zeros(problem.rank, device=tensors.device, dtype=torch.float64)
    resource = torch.zeros((), device=tensors.device, dtype=torch.float64)
    work = torch.zeros((), device=tensors.device, dtype=torch.float64)
    sqrt_d_sum = torch.zeros((), device=tensors.device, dtype=torch.float64)
    metric_rows: list[Any] = []
    previous = 0
    for count in counts:
        shell = order_device[previous:count]
        if counters is not None:
            counters.replay_factor_element_visit_count += problem.rank * int(
                shell.numel()
            )
        for start in range(0, int(shell.numel()), 1 << 16):
            chunk = shell[start : start + (1 << 16)]
            h_sum = h_sum + torch.sum(tensors.factor[:, chunk], dim=1)
            resource = resource + torch.sum(tensors.resource[chunk])
            work = work + torch.sum(tensors.work[chunk])
            sqrt_d_sum = sqrt_d_sum + torch.sum(tensors.sqrt_residual[chunk])
        low_q = torch.dot(h_sum, h_sum)
        spectral_q = torch.as_tensor(
            problem.residual_spectral_bound * float(count),
            device=tensors.device,
            dtype=torch.float64,
        )
        diagonal_q = sqrt_d_sum * sqrt_d_sum
        residual_q = torch.minimum(spectral_q, diagonal_q)
        metric_rows.append(
            torch.stack(
                (
                    resource,
                    work,
                    low_q,
                    spectral_q,
                    diagonal_q,
                    residual_q,
                    low_q + residual_q,
                )
            )
        )
        previous = count
    metrics = torch.stack(tuple(metric_rows)).detach().cpu().numpy()
    candidates: list[V1RelaxedKKTHardCandidate] = []
    epsilon = np.finfo(np.float64).eps
    for count, row in zip(counts, metrics, strict=True):
        (
            resource_value,
            work_value,
            low_q_value,
            spectral_q_value,
            diagonal_q_value,
            residual_q_value,
            canonical_q_value,
        ) = (float(item) for item in row)
        work_tolerance = 128.0 * epsilon * max(1.0, work_value, problem.work_budget)
        q_tolerance = (
            128.0 * epsilon * max(1.0, canonical_q_value, problem.quadratic_budget)
        )
        work_feasible = bool(work_value <= problem.work_budget + work_tolerance)
        q_feasible = bool(canonical_q_value <= problem.quadratic_budget + q_tolerance)
        candidates.append(
            V1RelaxedKKTHardCandidate(
                branch=branch,
                prefix_count=count,
                removed_index_hash=_array_digest(order[:count]),
                resource_value=resource_value,
                corrected_work_value=work_value,
                low_rank_quadratic=low_q_value,
                spectral_residual_quadratic=spectral_q_value,
                diagonal_l1_residual_quadratic=diagonal_q_value,
                canonical_residual_quadratic=residual_q_value,
                canonical_quadratic=canonical_q_value,
                work_feasible=work_feasible,
                quadratic_feasible=q_feasible,
                feasible=work_feasible and q_feasible,
            )
        )
    return tuple(candidates)


def _round_torch_branch(
    tensors: _TorchProblem,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    branch: V1RelaxedKKTBranch,
    relaxed: Any,
    reduced: Any,
    *,
    authorized: bool,
    tile_size: int = _CUDA_ROUNDING_TILE_SIZE,
    counters: _CUDAExecutionCounters | None = None,
) -> V1RelaxedKKTRoundedLadder:
    if not authorized:
        return _empty_ladder(branch)
    capacity = problem.legal_hard_capacity
    counts = tuple(
        count for count in policy.candidate_count_ladder if count <= capacity
    )
    if not counts:
        return _empty_ladder(branch)
    if counters is not None:
        _synchronize_torch_device(tensors)
        rounding_started = time.perf_counter()
    order = _torch_rounded_order(
        tensors,
        problem,
        policy,
        relaxed,
        reduced,
        counts[-1],
        tile_size=tile_size,
        counters=counters,
    )
    if counters is not None:
        _synchronize_torch_device(tensors)
        counters.warm_rounding_seconds += time.perf_counter() - rounding_started
    if order.size < counts[-1]:
        counts = tuple(count for count in counts if count <= int(order.size))
    if not counts:
        return _empty_ladder(branch)
    order = order[: counts[-1]]
    if counters is not None:
        _synchronize_torch_device(tensors)
        replay_started = time.perf_counter()
    candidates = _hard_candidates_from_order_torch(
        tensors,
        problem,
        branch,
        order,
        counts,
        counters=counters,
    )
    if counters is not None:
        _synchronize_torch_device(tensors)
        counters.warm_replay_seconds += time.perf_counter() - replay_started
    return V1RelaxedKKTRoundedLadder(
        branch=branch,
        ordered_route_indices=order,
        candidates=candidates,
    )


def _numpy_branch_result(
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    branch: V1RelaxedKKTBranch,
) -> V1RelaxedKKTBranchResult:
    state, _scaled, receipt, reduced = _solve_branch_numpy(problem, policy, branch)
    ladder = _round_numpy_branch(
        problem,
        policy,
        branch,
        state.primal,
        reduced,
        authorized=receipt.convergence.converged,
    )
    materialized = (
        state.primal
        if problem.route_count <= policy.materialize_relaxed_gate_route_limit
        else None
    )
    digest = _array_digest(state.primal)
    return V1RelaxedKKTBranchResult(
        branch=branch,
        receipt=receipt,
        rounded_ladder=ladder,
        relaxed_gate_values=materialized,
        relaxed_gate_digest=digest,
    )


def _torch_branch_result(
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    branch: V1RelaxedKKTBranch,
) -> V1RelaxedKKTBranchResult:
    scaled = _scaled_problem(problem, branch)
    tensors = _torch_problem(problem, policy, scaled)
    # Solve and round from one tensor owner so rank-major geometry is never
    # duplicated on device.
    relaxed, reduced, _scaled_again, receipt = _solve_branch_torch(
        problem, policy, branch, tensors=tensors
    )
    ladder = _round_torch_branch(
        tensors,
        problem,
        policy,
        branch,
        relaxed,
        reduced,
        authorized=receipt.convergence.converged,
    )
    if problem.route_count <= policy.materialize_relaxed_gate_route_limit:
        materialized = relaxed.detach().cpu().numpy()
        digest = _array_digest(materialized)
    else:
        materialized = None
        # A receipt hash is not a byte digest of a device-resident vector.
        # Leave the digest absent instead of assigning a false identity; the
        # authoritative observable is the rounded ladder plus KKT receipts.
        digest = ""
    return V1RelaxedKKTBranchResult(
        branch=branch,
        receipt=receipt,
        rounded_ladder=ladder,
        relaxed_gate_values=materialized,
        relaxed_gate_digest=digest,
    )


def _aggregate_relaxed_kkt_result(
    *,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    spectral: V1RelaxedKKTBranchResult,
    diagonal: V1RelaxedKKTBranchResult,
) -> V1RelaxedKKTResult:
    both = (
        spectral.receipt.convergence.converged
        and diagonal.receipt.convergence.converged
    )
    selected: V1RelaxedKKTBranch | None = None
    certified = False
    if both:
        spectral_lower = spectral.receipt.gap.primal_resource_lower_bound
        diagonal_lower = diagonal.receipt.gap.primal_resource_lower_bound
        if spectral_lower >= diagonal_lower:
            selected = V1RelaxedKKTBranch.SPECTRAL
            chosen = spectral.receipt.gap
            other = diagonal.receipt.gap
        else:
            selected = V1RelaxedKKTBranch.DIAGONAL_L1
            chosen = diagonal.receipt.gap
            other = spectral.receipt.gap
        comparison_tolerance = max(
            chosen.absolute_gap_tolerance,
            other.absolute_gap_tolerance,
        ) + max(
            chosen.relative_gap_tolerance,
            other.relative_gap_tolerance,
        ) * max(
            1.0,
            abs(chosen.primal_resource_lower_bound),
            abs(other.conic_dual_resource_upper_bound),
        )
        certified = (
            chosen.primal_resource_lower_bound + comparison_tolerance
            >= other.conic_dual_resource_upper_bound
        )
    return V1RelaxedKKTResult(
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        spectral=spectral,
        diagonal_l1=diagonal,
        selected_branch=selected,
        both_branches_converged=both,
        branch_choice_certified_within_tolerance=certified,
    )


def _cuda_forecast_from_device(
    *,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    execution_design: V1RelaxedKKTCUDAExecutionDesign,
    counters: _CUDAExecutionCounters,
    device_shell_execution_design: V1RelaxedKKTDeviceShellExecutionDesign | None = None,
) -> tuple[Any, Any, V1RelaxedKKTCUDAHBMForecast]:
    """Query device capacity and run the tensor-allocation-free forecast."""

    started = time.perf_counter()
    torch = _torch_module()
    device = torch.device(policy.torch_device)
    if device.type != "cuda" or not torch.cuda.is_available():
        _refuse(
            V1RelaxedKKTRefusalCode.CUDA_UNAVAILABLE,
            "backend='torch_cuda' requires an available CUDA device",
        )
    try:
        properties = torch.cuda.get_device_properties(device)
        current_reserved = int(torch.cuda.memory_reserved(device))
        admission_memory = (
            cuda_reusable_memory_bytes(device, framework=torch),
            int(torch.cuda.memory_allocated(device)),
        )
    except Exception as error:  # pragma: no cover - host/device dependent.
        _refuse(V1RelaxedKKTRefusalCode.CUDA_UNAVAILABLE, str(error))
    forecast = estimate_v1_relaxed_kkt_cuda_hbm(
        problem=problem,
        policy=policy,
        execution_design=execution_design,
        device_total_memory_bytes=int(properties.total_memory),
        current_reserved_bytes=current_reserved,
        device=str(device),
        device_shell_execution_design=device_shell_execution_design,
        _admission_memory=admission_memory,
    )
    counters.forecast_seconds += time.perf_counter() - started
    if (
        device_shell_execution_design is None
        and not execution_design.scan_only
        and forecast.host_order_bytes > execution_design.maximum_host_order_bytes
    ):
        _refuse(
            V1RelaxedKKTRefusalCode.CUDA_HOST_ORDER_LIMIT_EXCEEDED,
            f"registered endpoint requires {forecast.host_order_bytes} host "
            f"bytes, exceeding ceiling "
            f"{execution_design.maximum_host_order_bytes}; use the explicit "
            "device-shell identity interface after its rollout gate closes",
        )
    if not forecast.admitted:
        code = (
            V1RelaxedKKTRefusalCode.CUDA_WORKSPACE_EXCEEDED
            if forecast.rounding_workspace_bytes > forecast.hbm_ceiling_bytes
            else V1RelaxedKKTRefusalCode.CUDA_HBM_FORECAST_EXCEEDED
        )
        _refuse(
            code,
            f"forecast peak {forecast.forecast_peak_reserved_bytes} exceeds "
            f"the registered {execution_design.hbm_ceiling_fraction:.3f} HBM "
            f"ceiling {forecast.hbm_ceiling_bytes}",
        )
    return torch, device, forecast


def _solve_v1_relaxed_kkt_fused_cuda(
    *,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    execution_design: V1RelaxedKKTCUDAExecutionDesign,
    device_shell_execution_design: V1RelaxedKKTDeviceShellExecutionDesign | None = None,
    parent_mask: HardMask | None = None,
    geometry_uncertainty: V1RelaxedKKTGeometryUncertaintySet | None = None,
) -> V1RelaxedKKTCUDAExecutionBundle | V1RelaxedKKTDeviceShellExecutionBundle:
    """Shared fused implementation for ordinary and explicit shell output."""

    if type(problem) is not V1RelaxedKKTProblem:
        raise TypeError("problem must be V1RelaxedKKTProblem")
    if type(policy) is not V1RelaxedKKTPolicy:
        raise TypeError("policy must be V1RelaxedKKTPolicy")
    if policy.backend != "torch_cuda":
        raise ValueError("CUDA telemetry requires backend='torch_cuda'")
    design = execution_design
    if type(design) is not V1RelaxedKKTCUDAExecutionDesign:
        raise TypeError("execution_design must be an exact CUDA design")
    if design.implementation != "fused_v2":
        raise ValueError("execution telemetry is defined only for fused_v2")
    if geometry_uncertainty is not None:
        if type(geometry_uncertainty) is not V1RelaxedKKTGeometryUncertaintySet:
            raise TypeError("geometry_uncertainty must be exact")
        if (
            geometry_uncertainty.problem_hash != problem.problem_hash
            or geometry_uncertainty.route_axis_hash != problem.route_axis_hash
            or geometry_uncertainty.source_geometry_hash != problem.source_geometry_hash
            or not geometry_uncertainty.source_derived_from_registered_geometry_receipts
        ):
            _refuse(
                V1RelaxedKKTRefusalCode.ADAPTIVE_RANK_WITNESS_SOURCE_MISMATCH,
                "adaptive-rank witness uncertainty does not bind this KKT chart",
            )
    elif design.require_adaptive_shell_witness:
        _refuse(
            V1RelaxedKKTRefusalCode.CUDA_SHELL_WITNESS_UNAVAILABLE,
            "adaptive shell witness execution requires source-derived corrected-A, "
            "literal-J, FD, and numerical geometry uncertainty",
        )
    shell_design = device_shell_execution_design
    shell_mode = shell_design is not None
    if shell_mode:
        if type(shell_design) is not V1RelaxedKKTDeviceShellExecutionDesign:
            raise TypeError("device_shell_execution_design must be exact")
        if shell_design.cuda_execution_design != design:
            raise ValueError("device-shell and fused CUDA designs differ")
        if type(parent_mask) is not HardMask:
            raise TypeError("device-shell execution requires an exact parent mask")
        if (
            parent_mask.route_axis_hash != problem.route_axis_hash
            or parent_mask.route_count != problem.route_count
        ):
            raise ValueError("device-shell parent differs from the KKT route axis")
        if not problem.block_capacities and not shell_design.allow_unconstrained_blocks:
            _refuse(
                V1RelaxedKKTRefusalCode.CUDA_DEVICE_SHELL_IDENTITY_UNAVAILABLE,
                "device-shell hard legality requires registered compact block "
                "capacities; unconstrained reference use must be explicit",
            )
        counts = tuple(
            count
            for count in policy.candidate_count_ladder
            if count <= problem.legal_hard_capacity
        )
        if not counts:
            _refuse(
                V1RelaxedKKTRefusalCode.CUDA_DEVICE_SHELL_IDENTITY_UNAVAILABLE,
                "no nonempty registered hard endpoint can be labelled",
            )
        if len(counts) > DEVICE_SHELL_MAXIMUM_ENDPOINTS:
            _refuse(
                V1RelaxedKKTRefusalCode.CUDA_DEVICE_SHELL_IDENTITY_UNAVAILABLE,
                "uint8 first-inclusion labels support at most 255 endpoints",
            )
    elif parent_mask is not None:
        raise ValueError("parent_mask is valid only for explicit device-shell mode")
    else:
        counts = ()
    counters = _CUDAExecutionCounters()
    total_started = time.perf_counter()
    torch, device, forecast = _cuda_forecast_from_device(
        problem=problem,
        policy=policy,
        execution_design=design,
        counters=counters,
        device_shell_execution_design=shell_design,
    )
    spectral_handle: V1DeviceShellLabelHandle | None = None
    diagonal_handle: V1DeviceShellLabelHandle | None = None
    spectral_shell: V1RelaxedKKTDeviceShellBranchIdentityReceipt | None = None
    diagonal_shell: V1RelaxedKKTDeviceShellBranchIdentityReceipt | None = None
    private_primal_directions: Any | None = None
    try:
        scaled = _scaled_problem(problem, V1RelaxedKKTBranch.SPECTRAL)
        tensors = _torch_problem(problem, policy, scaled, counters=counters)
        torch.cuda.synchronize(device)
        solver_started = time.perf_counter()
        solve_outcome = _solve_branches_fused_torch(
            problem,
            policy,
            tensors,
            counters,
            _retain_private_state=True,
        )
        if type(solve_outcome) is not _FusedTorchSolveOutcome:
            raise RuntimeError("fused CUDA solver dropped private branch state")
        (
            relaxed_s,
            reduced_s,
            receipt_s,
            relaxed_d,
            reduced_d,
            receipt_d,
        ) = solve_outcome
        _validate_fused_branch_state_forecast(
            problem=problem,
            outcome=solve_outcome,
            forecast=forecast,
            private_primal_custody_required=shell_design is not None,
        )
        torch.cuda.synchronize(device)
        counters.warm_solver_seconds += time.perf_counter() - solver_started
        if shell_design is not None:
            if not (
                receipt_s.convergence.converged and receipt_d.convergence.converged
            ):
                _refuse_v1_relaxed_kkt_device_shell_nonconvergence(
                    problem=problem,
                    policy=policy,
                    execution_design=shell_design,
                    spectral=receipt_s,
                    diagonal_l1=receipt_d,
                )
            if parent_mask is None:  # pragma: no cover - guarded above.
                raise RuntimeError("device-shell parent mask was not retained")
            spectral_handle, spectral_shell = _freeze_torch_device_shell_branch(
                tensors=tensors,
                problem=problem,
                policy=policy,
                branch=V1RelaxedKKTBranch.SPECTRAL,
                branch_receipt=receipt_s,
                relaxed=relaxed_s,
                reduced=reduced_s,
                parent_mask=parent_mask,
                counts=counts,
                execution_design=shell_design,
                counters=counters,
            )
            try:
                diagonal_handle, diagonal_shell = _freeze_torch_device_shell_branch(
                    tensors=tensors,
                    problem=problem,
                    policy=policy,
                    branch=V1RelaxedKKTBranch.DIAGONAL_L1,
                    branch_receipt=receipt_d,
                    relaxed=relaxed_d,
                    reduced=reduced_d,
                    parent_mask=parent_mask,
                    counts=counts,
                    execution_design=shell_design,
                    counters=counters,
                )
            except Exception:
                spectral_handle.evict()
                spectral_handle = None
                raise
            # The serializable KKT result stays array-free and action-free.
            # Exact hard identities live only in the sibling shell receipts.
            ladder_s = _empty_ladder(V1RelaxedKKTBranch.SPECTRAL)
            ladder_d = _empty_ladder(V1RelaxedKKTBranch.DIAGONAL_L1)
        elif design.scan_only:
            # Do not even enter the rounding surface in engineering canaries;
            # this is stronger than calling it with ``authorized=False``.
            ladder_s = _empty_ladder(V1RelaxedKKTBranch.SPECTRAL)
            ladder_d = _empty_ladder(V1RelaxedKKTBranch.DIAGONAL_L1)
        else:
            ladder_s = _round_torch_branch(
                tensors,
                problem,
                policy,
                V1RelaxedKKTBranch.SPECTRAL,
                relaxed_s,
                reduced_s,
                authorized=receipt_s.convergence.converged,
                tile_size=design.rounding_tile_size,
                counters=counters,
            )
            ladder_d = _round_torch_branch(
                tensors,
                problem,
                policy,
                V1RelaxedKKTBranch.DIAGONAL_L1,
                relaxed_d,
                reduced_d,
                authorized=receipt_d.convergence.converged,
                tile_size=design.rounding_tile_size,
                counters=counters,
            )
        materialized_s: np.ndarray | None
        materialized_d: np.ndarray | None
        if (
            shell_design is None
            and problem.route_count <= policy.materialize_relaxed_gate_route_limit
        ):
            materialized_s = relaxed_s.detach().cpu().numpy()
            materialized_d = relaxed_d.detach().cpu().numpy()
            counters.bulk_d2h_tensor_payload_bytes += int(
                materialized_s.nbytes + materialized_d.nbytes
            )
            digest_s = _array_digest(materialized_s)
            digest_d = _array_digest(materialized_d)
        else:
            materialized_s = None
            materialized_d = None
            digest_s = ""
            digest_d = ""
        branch_s = V1RelaxedKKTBranchResult(
            branch=V1RelaxedKKTBranch.SPECTRAL,
            receipt=receipt_s,
            rounded_ladder=ladder_s,
            relaxed_gate_values=materialized_s,
            relaxed_gate_digest=digest_s,
        )
        branch_d = V1RelaxedKKTBranchResult(
            branch=V1RelaxedKKTBranch.DIAGONAL_L1,
            receipt=receipt_d,
            rounded_ladder=ladder_d,
            relaxed_gate_values=materialized_d,
            relaxed_gate_digest=digest_d,
        )
        if design.scan_only:
            result = V1RelaxedKKTResult(
                problem_hash=problem.problem_hash,
                policy_hash=policy.policy_hash,
                spectral=branch_s,
                diagonal_l1=branch_d,
                selected_branch=None,
                both_branches_converged=(
                    receipt_s.convergence.converged and receipt_d.convergence.converged
                ),
                branch_choice_certified_within_tolerance=False,
            )
        else:
            result = _aggregate_relaxed_kkt_result(
                problem=problem,
                policy=policy,
                spectral=branch_s,
                diagonal=branch_d,
            )
        adaptive_rank_witnesses: tuple[V1RelaxedKKTAdaptiveRankBranchWitness, ...] = ()
        if geometry_uncertainty is not None:
            if (
                solve_outcome.spectral_state is None
                or solve_outcome.diagonal_state is None
            ):
                _refuse(
                    V1RelaxedKKTRefusalCode.ADAPTIVE_RANK_WITNESS_BOUNDARY_UNAVAILABLE,
                    "adaptive-rank witness requires both retained converged branch states",
                )
            if shell_design is None:
                adaptive_rank_witnesses = (
                    issue_v1_relaxed_kkt_adaptive_rank_branch_witness_from_state(
                        problem=problem,
                        policy=policy,
                        branch_result=branch_s,
                        private_state=solve_outcome.spectral_state,
                        tensors=tensors,
                        geometry_uncertainty=geometry_uncertainty,
                        tile_size=design.rounding_tile_size,
                    ),
                    issue_v1_relaxed_kkt_adaptive_rank_branch_witness_from_state(
                        problem=problem,
                        policy=policy,
                        branch_result=branch_d,
                        private_state=solve_outcome.diagonal_state,
                        tensors=tensors,
                        geometry_uncertainty=geometry_uncertainty,
                        tile_size=design.rounding_tile_size,
                    ),
                )
            else:
                if spectral_shell is None or diagonal_shell is None:
                    raise RuntimeError("device-shell identities were not frozen")
                adaptive_rank_witnesses = (
                    issue_v1_relaxed_kkt_adaptive_rank_branch_witness_from_device_shell_state(
                        problem=problem,
                        policy=policy,
                        branch_result=branch_s,
                        branch_identity=spectral_shell,
                        private_state=solve_outcome.spectral_state,
                        tensors=tensors,
                        geometry_uncertainty=geometry_uncertainty,
                        tile_size=design.rounding_tile_size,
                    ),
                    issue_v1_relaxed_kkt_adaptive_rank_branch_witness_from_device_shell_state(
                        problem=problem,
                        policy=policy,
                        branch_result=branch_d,
                        branch_identity=diagonal_shell,
                        private_state=solve_outcome.diagonal_state,
                        tensors=tensors,
                        geometry_uncertainty=geometry_uncertainty,
                        tile_size=design.rounding_tile_size,
                    ),
                )
        if shell_design is not None:
            if (
                solve_outcome.spectral_state is None
                or solve_outcome.diagonal_state is None
            ):
                raise RuntimeError(
                    "device-shell execution lost converged private primals"
                )
            # Allocate the post-solve projection owner before measuring peak
            # HBM.  The later descriptor construction hashes metadata only and
            # transfers this tensor without another route-scale allocation.
            private_primal_directions = _stack_v1_relaxed_kkt_private_primals(
                spectral_state=solve_outcome.spectral_state,
                diagonal_l1_state=solve_outcome.diagonal_state,
            )
        torch.cuda.synchronize(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    except Exception as error:  # pragma: no cover - host/device dependent.
        if spectral_handle is not None and not spectral_handle.is_evicted:
            spectral_handle.evict()
        if diagonal_handle is not None and not diagonal_handle.is_evicted:
            diagonal_handle.evict()
        oom_type = getattr(torch, "OutOfMemoryError", ())
        if (oom_type and isinstance(error, oom_type)) or (
            "out of memory" in str(error).lower()
        ):
            _refuse(V1RelaxedKKTRefusalCode.CUDA_OUT_OF_MEMORY, str(error))
        raise
    total_wall = time.perf_counter() - total_started
    hard_ladder_emitted = bool(
        branch_s.rounded_ladder.candidates or branch_d.rounded_ladder.candidates
    )
    if shell_design is not None:
        if spectral_shell is None or diagonal_shell is None:
            raise RuntimeError("device-shell identities were not frozen")
        spectral_endpoint_counts = spectral_shell.frozen_ladder.cumulative_counts
        diagonal_endpoint_counts = diagonal_shell.frozen_ladder.cumulative_counts
        spectral_endpoint_hashes = tuple(
            action.removed_route_index_hash
            for action in spectral_shell.frozen_ladder.actions
        )
        diagonal_endpoint_hashes = tuple(
            action.removed_route_index_hash
            for action in diagonal_shell.frozen_ladder.actions
        )
    else:
        spectral_endpoint_counts = tuple(
            item.prefix_count for item in ladder_s.candidates
        )
        diagonal_endpoint_counts = tuple(
            item.prefix_count for item in ladder_d.candidates
        )
        spectral_endpoint_hashes = tuple(
            item.removed_index_hash for item in ladder_s.candidates
        )
        diagonal_endpoint_hashes = tuple(
            item.removed_index_hash for item in ladder_d.candidates
        )
    telemetry = V1RelaxedKKTCUDAExecutionTelemetry(
        result_hash=result.result_hash,
        forecast_hash=forecast.forecast_hash,
        execution_design_hash=design.design_hash,
        algorithm_id=RELAXED_KKT_FUSED_CUDA_ALGORITHM_ID,
        device=str(device),
        route_axis_hash=problem.route_axis_hash,
        source_geometry_hash=problem.source_geometry_hash,
        numerical_policy_hash=policy.policy_hash,
        route_count=problem.route_count,
        rank=problem.rank,
        scan_only=design.scan_only,
        forecast_seconds=counters.forecast_seconds,
        cold_h2d_seconds=counters.cold_h2d_seconds,
        warm_solver_seconds=counters.warm_solver_seconds,
        warm_rounding_seconds=counters.warm_rounding_seconds,
        warm_replay_seconds=counters.warm_replay_seconds,
        receipt_seconds=counters.receipt_seconds,
        total_wall_seconds=total_wall,
        factor_forward_scan_count=counters.factor_forward_scan_count,
        block_separable_repair_factor_forward_scan_count=(
            counters.block_separable_repair_factor_forward_scan_count
        ),
        factor_adjoint_scan_count=counters.factor_adjoint_scan_count,
        factor_route_visit_count=counters.factor_route_visit_count,
        key_scan_count=counters.key_scan_count,
        key_route_visit_count=counters.key_route_visit_count,
        replay_factor_element_visit_count=(counters.replay_factor_element_visit_count),
        bulk_h2d_tensor_payload_bytes=(counters.bulk_h2d_tensor_payload_bytes),
        bulk_d2h_tensor_payload_bytes=(counters.bulk_d2h_tensor_payload_bytes),
        peak_allocated_hbm_bytes=peak_allocated,
        peak_reserved_hbm_bytes=peak_reserved,
        maximum_endpoint=forecast.maximum_endpoint,
        rounding_tile_size=design.rounding_tile_size,
        spectral_response_scale=_scaled_problem(
            problem, V1RelaxedKKTBranch.SPECTRAL
        ).response_scale,
        diagonal_l1_response_scale=_scaled_problem(
            problem, V1RelaxedKKTBranch.DIAGONAL_L1
        ).response_scale,
        spectral_endpoint_counts=spectral_endpoint_counts,
        diagonal_l1_endpoint_counts=diagonal_endpoint_counts,
        spectral_endpoint_index_hashes=spectral_endpoint_hashes,
        diagonal_l1_endpoint_index_hashes=diagonal_endpoint_hashes,
        hard_ladder_emitted=hard_ladder_emitted,
        adaptive_shell_witness_available=False,
    )
    base_execution = V1RelaxedKKTCUDAExecutionBundle(
        result=result,
        forecast=forecast,
        telemetry=telemetry,
        execution_design=design,
        geometry_uncertainty=geometry_uncertainty,
        adaptive_rank_witnesses=(
            adaptive_rank_witnesses if shell_design is None else ()
        ),
    )
    if shell_design is None:
        return base_execution
    if (
        spectral_shell is None
        or diagonal_shell is None
        or spectral_handle is None
        or diagonal_handle is None
        or solve_outcome.spectral_state is None
        or solve_outcome.diagonal_state is None
        or private_primal_directions is None
    ):
        raise RuntimeError("device-shell execution did not retain both branches")
    private_primal_handle: V1LiteralJPrivatePrimalDirectionCUDAHandle | None = None
    theorem_state_custody: _V1RelaxedKKTTheoremStateCustody | None = None
    try:
        shell_receipt = V1RelaxedKKTDeviceShellExecutionReceipt(
            problem_hash=problem.problem_hash,
            policy_hash=policy.policy_hash,
            device_shell_execution_design_hash=shell_design.design_hash,
            cuda_execution_design_hash=design.design_hash,
            base_result_hash=result.result_hash,
            hbm_forecast_hash=forecast.forecast_hash,
            cuda_telemetry_hash=telemetry.telemetry_hash,
            spectral=spectral_shell,
            diagonal_l1=diagonal_shell,
            device_label_vector_count=2,
            full_axis_label_d2h_scan_count=2,
            label_d2h_tensor_payload_bytes=2 * problem.route_count,
        )
        theorem_state_custody = _V1RelaxedKKTTheoremStateCustody(
            problem=problem,
            policy=policy,
            tensors=tensors,
            spectral_state=solve_outcome.spectral_state,
            diagonal_l1_state=solve_outcome.diagonal_state,
        )
        private_primal_handle = _build_v1_relaxed_kkt_private_primal_direction_handle(
            problem=problem,
            policy=policy,
            base_execution=base_execution,
            execution_design=shell_design,
            execution_receipt=shell_receipt,
            spectral_state=solve_outcome.spectral_state,
            diagonal_l1_state=solve_outcome.diagonal_state,
            private_primal_directions=private_primal_directions,
        )
        return V1RelaxedKKTDeviceShellExecutionBundle(
            base_execution=base_execution,
            execution_design=shell_design,
            receipt=shell_receipt,
            spectral_label_handle=spectral_handle,
            diagonal_l1_label_handle=diagonal_handle,
            geometry_uncertainty=geometry_uncertainty,
            adaptive_rank_witnesses=adaptive_rank_witnesses,
            private_primal_direction_handle=private_primal_handle,
            _theorem_state_custody=theorem_state_custody,
        )
    except Exception:
        _evict_v1_relaxed_kkt_device_shell_owners_atomically(
            spectral_label_handle=spectral_handle,
            diagonal_l1_label_handle=diagonal_handle,
            private_primal_direction_handle=private_primal_handle,
            theorem_state_custody=theorem_state_custody,
        )
        raise


def solve_v1_relaxed_kkt_with_telemetry(
    *,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    execution_design: V1RelaxedKKTCUDAExecutionDesign,
    geometry_uncertainty: V1RelaxedKKTGeometryUncertaintySet | None = None,
) -> V1RelaxedKKTCUDAExecutionBundle:
    """Run explicitly fused CUDA with the preexisting ordinary ladder output."""

    result = _solve_v1_relaxed_kkt_fused_cuda(
        problem=problem,
        policy=policy,
        execution_design=execution_design,
        geometry_uncertainty=geometry_uncertainty,
    )
    if type(result) is not V1RelaxedKKTCUDAExecutionBundle:
        raise RuntimeError("ordinary fused execution returned device-shell output")
    return result


def solve_v1_relaxed_kkt_with_device_shell_ladders(
    *,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    parent_mask: HardMask,
    execution_design: V1RelaxedKKTDeviceShellExecutionDesign,
    geometry_uncertainty: V1RelaxedKKTGeometryUncertaintySet | None = None,
) -> V1RelaxedKKTDeviceShellExecutionBundle:
    """Run explicit fused CUDA and freeze two device-resident shell ladders.

    This unpromoted interface provides exact hard-mask identities without
    materializing an ordinary action.  Legacy/proposal-only designs still carry
    no adaptive boundary-face witness; source-stable designs require the
    registered geometry uncertainty set and return typed branch witnesses in
    the shell bundle.  The shell itself cannot open calibration, oracle, commit,
    or a terminal/Pareto claim without downstream F/Q/R and paired-BCE receipts.
    """

    if type(execution_design) is not V1RelaxedKKTDeviceShellExecutionDesign:
        raise TypeError("execution_design must be an exact device-shell design")
    result = _solve_v1_relaxed_kkt_fused_cuda(
        problem=problem,
        policy=policy,
        execution_design=execution_design.cuda_execution_design,
        device_shell_execution_design=execution_design,
        parent_mask=parent_mask,
        geometry_uncertainty=geometry_uncertainty,
    )
    if type(result) is not V1RelaxedKKTDeviceShellExecutionBundle:
        raise RuntimeError("device-shell execution returned ordinary output")
    return result


def solve_v1_relaxed_kkt(
    *,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    cuda_execution_design: V1RelaxedKKTCUDAExecutionDesign | None = None,
) -> V1RelaxedKKTResult:
    """Solve both registered convex branches and deterministically round them.

    The returned branch choice exists only when both relaxations meet their
    registered primal, stationarity, complementarity, and gap tolerances.
    Candidate ladders are empty for an unconverged branch.  This function has
    no nonlinear-remainder, BCE-oracle, or irreversible mask-commit authority.
    """

    if type(problem) is not V1RelaxedKKTProblem:
        raise TypeError("problem must be V1RelaxedKKTProblem")
    if type(policy) is not V1RelaxedKKTPolicy:
        raise TypeError("policy must be V1RelaxedKKTPolicy")
    if policy.backend == "numpy_reference":
        if cuda_execution_design is not None:
            raise ValueError("cuda_execution_design is invalid for numpy_reference")
        if problem.route_count > policy.cpu_reference_route_limit:
            _refuse(
                V1RelaxedKKTRefusalCode.CPU_REFERENCE_SCALE_EXCEEDED,
                f"route_count={problem.route_count} exceeds registered reference "
                f"limit={policy.cpu_reference_route_limit}; use torch_cuda",
            )
        spectral = _numpy_branch_result(problem, policy, V1RelaxedKKTBranch.SPECTRAL)
        diagonal = _numpy_branch_result(problem, policy, V1RelaxedKKTBranch.DIAGONAL_L1)
    else:
        if cuda_execution_design is None:
            # Rollout safety: the fused-v2 CUDA path remains explicit until
            # H200 parity and the registered >=20% kernel-improvement gate
            # close.  Omitting a design preserves the historical CUDA path.
            spectral = _torch_branch_result(
                problem, policy, V1RelaxedKKTBranch.SPECTRAL
            )
            diagonal = _torch_branch_result(
                problem, policy, V1RelaxedKKTBranch.DIAGONAL_L1
            )
            return _aggregate_relaxed_kkt_result(
                problem=problem,
                policy=policy,
                spectral=spectral,
                diagonal=diagonal,
            )
        if type(cuda_execution_design) is not V1RelaxedKKTCUDAExecutionDesign:
            raise TypeError("cuda_execution_design must be an exact CUDA design")
        design = cuda_execution_design
        if design.implementation == "fused_v2":
            return solve_v1_relaxed_kkt_with_telemetry(
                problem=problem,
                policy=policy,
                execution_design=design,
            ).result
        if design.scan_only:
            raise ValueError("legacy_v1 does not define scan-only execution")
        spectral = _torch_branch_result(problem, policy, V1RelaxedKKTBranch.SPECTRAL)
        diagonal = _torch_branch_result(problem, policy, V1RelaxedKKTBranch.DIAGONAL_L1)
    return _aggregate_relaxed_kkt_result(
        problem=problem,
        policy=policy,
        spectral=spectral,
        diagonal=diagonal,
    )


__all__ = [
    "RELAXED_KKT_ALGORITHM_ID",
    "RELAXED_KKT_CANONICAL_HARD_Q_ID",
    "RELAXED_KKT_FUSED_CUDA_ALGORITHM_ID",
    "RELAXED_KKT_RELAXED_AUTHORITY_SCOPE",
    "RELAXED_KKT_ROUNDING_KEY_ID",
    "RELAXED_KKT_SIGNED_ACTION_WORK_ID",
    "RELAXED_KKT_SINGLETON_TRIANGLE_WORK_ID",
    "RELAXED_KKT_ROBUST_SOLUTION_SENSITIVITY_SCOPE",
    "RELAXED_KKT_DECISION_LEVEL_ROBUST_INVARIANCE_SCOPE",
    "V1RelaxedKKTActiveShellRestrictedOperatorReceipt",
    "V1RelaxedKKTAllAdmissibleShellFacesReceipt",
    "V1RelaxedKKTBlockCapacity",
    "V1RelaxedKKTBranch",
    "V1RelaxedKKTAdaptiveRankBranchWitness",
    "V1RelaxedKKTBranchReceipt",
    "V1RelaxedKKTBranchResult",
    "V1RelaxedKKTConvergenceReceipt",
    "V1RelaxedKKTClosureMode",
    "V1RelaxedKKTCUDAExecutionBundle",
    "V1RelaxedKKTCUDAExecutionDesign",
    "V1RelaxedKKTCUDAExecutionTelemetry",
    "V1RelaxedKKTCUDAHBMForecast",
    "V1RelaxedKKTDeviceShellBranchIdentityReceipt",
    "V1RelaxedKKTDeviceShellExecutionBundle",
    "V1RelaxedKKTDeviceShellExecutionDesign",
    "V1RelaxedKKTDeviceShellExecutionReceipt",
    "V1RelaxedKKTDeviceShellNonconvergenceReceipt",
    "V1RelaxedKKTDecisionLevelRobustInvarianceCertificate",
    "V1RelaxedKKTDecisionLevelShellReferenceEvidence",
    "V1RelaxedKKTDecisionLevelTheoremStateBorrowRequest",
    "V1RelaxedKKTDecisionLevelTheoremLeaseHBMForecast",
    "V1RelaxedKKTDecisionLevelTheoremStateLease",
    "V1RelaxedKKTGapReceipt",
    "V1RelaxedKKTGeometryUncertaintySet",
    "V1RelaxedKKTHardCandidate",
    "V1RelaxedKKTPolicy",
    "V1RelaxedKKTPrimalFeasibilityReceipt",
    "V1RelaxedKKTProblem",
    "V1RelaxedKKTRefusal",
    "V1RelaxedKKTRefusalCode",
    "V1RelaxedKKTResult",
    "V1RelaxedKKTRobustSolutionSensitivityCertificate",
    "V1RelaxedKKTRobustDualBoundReceipt",
    "V1RelaxedKKTRoundedLadder",
    "V1RelaxedKKTStationarityReceipt",
    "V1RelaxedKKTOutsideShellIntervalScanReceipt",
    "V1RelaxedKKTWorstCaseHardFQReplayReceipt",
    "V1_RELAXED_KKT_FUSED_ROUTE_VECTOR_FORECAST_COUNT",
    "bind_v1_relaxed_kkt_decision_level_invariance_to_adaptive_rank_witness",
    "estimate_v1_relaxed_kkt_cuda_hbm",
    "estimate_v1_relaxed_kkt_decision_level_theorem_lease_hbm",
    "explain_v1_relaxed_kkt_robust_solution_sensitivity_gap",
    "issue_v1_relaxed_kkt_adaptive_rank_branch_witness_from_device_shell_state",
    "issue_v1_relaxed_kkt_adaptive_rank_branch_witness_from_state",
    "issue_v1_relaxed_kkt_geometry_uncertainty_from_registered_receipts",
    "issue_v1_relaxed_kkt_decision_level_robust_invariance_from_source_receipts",
    "issue_v1_relaxed_kkt_decision_level_robust_invariance_reference",
    "issue_v1_relaxed_kkt_decision_level_shell_reference_evidence",
    "solve_v1_relaxed_kkt",
    "solve_v1_relaxed_kkt_with_device_shell_ladders",
    "solve_v1_relaxed_kkt_with_telemetry",
    "v1_relaxed_kkt_decision_level_reference_basis_axis_hash",
]
