"""CUDA-facing endogenous resource frontier for signed-work modern-v1.

The canonical signed-work selector separates step-size discovery from
step-composition.  This module owns Phase A: find the resource frontier implied
by the two independent Q branches, legality, eligibility, and the required
hard lower rung.  It deliberately does not scan the signed-F operator.

Production execution is CUDA/determinism gated and uses the tensor
resource-frontier solver; target-hardware parity and timing evidence remain a
separate qualification gate.  The private CPU path is a tiny-domain theorem
reference used to bind the interface, authority receipts, and the downstream
device-shell endpoint plan without introducing an ad hoc endpoint schedule.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import InitVar, dataclass, field
from enum import Enum
import hashlib
import math
from numbers import Integral, Real
import os
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, Final, Mapping, NoReturn

import numpy as np

from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

from .contracts import content_hash
from .modern_v1_signed_work_device_shell_cuda import (
    V1SignedWorkDeviceShellEndpointPlanReceipt,
)
from .modern_v1_sampled_device_resident_d_session import (
    V1SampledDeviceResidentDKKTGeometryDeviceHandle,
    V1SampledDeviceResidentDKKTGeometryLease,
)
from .modern_v1_signed_work_kkt_cuda import (
    V1SignedWorkLeastWorkCUDADeviceProblemV2,
    V1SignedWorkLeastWorkCUDAProblem,
)
from .modern_v1_signed_work_operator_cuda import V1SignedWorkOperatorCUDA
from .modern_v1_relaxed_kkt import (
    V1RelaxedKKTBlockCapacity,
    V1RelaxedKKTProblem,
)
from .modern_v1_device_shell_ladder import DEVICE_SHELL_MAXIMUM_ENDPOINTS
from .modern_v1_signed_work_least_work_reference import (
    V1SignedWorkLeastWorkBranch,
    V1SignedWorkKKTReferenceProblem,
    V1SignedWorkLiveBlock,
)
from .modern_v1_signed_work_resource_frontier import (
    SIGNED_WORK_RESOURCE_FRONTIER_COORDINATE_RULE_ID,
    V1SignedWorkEndogenousResourceCoordinate,
    V1SignedWorkResourceBranchReceipt,
    V1SignedWorkResourceBranchStatus,
    V1SignedWorkResourceFrontierPolicy,
    V1SignedWorkResourceFrontierProblem,
    V1SignedWorkResourceFrontierResult,
    V1SignedWorkResourceUnionStatus,
    solve_v1_signed_work_resource_frontier_reference,
)

try:  # pragma: no cover - availability is environment dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from .modern_v1_signed_work_decision_witness_cuda import (
        V1SignedWorkCUDAResourceCellBinding,
    )


SIGNED_WORK_RESOURCE_FRONTIER_CUDA_POLICY_SCHEMA: Final = (
    "partition_free_v1_signed_work_resource_frontier_cuda_policy_v1"
)
SIGNED_WORK_RESOURCE_FRONTIER_CUDA_PROBLEM_SCHEMA: Final = (
    "partition_free_v1_signed_work_resource_frontier_cuda_problem_v2"
)
SIGNED_WORK_RESOURCE_FRONTIER_CUDA_DEVICE_PROBLEM_SCHEMA_V1: Final = (
    "partition_free_v1_signed_work_resource_frontier_cuda_device_problem_v1"
)
SIGNED_WORK_RESOURCE_FRONTIER_CUDA_DEVICE_EXECUTION_SCHEMA_V1: Final = (
    "partition_free_v1_signed_work_resource_frontier_cuda_device_execution_v1"
)
SIGNED_WORK_RESOURCE_FRONTIER_CUDA_DEVICE_FLOOR_FAMILY_SCHEMA_V1: Final = (
    "partition_free_v1_signed_work_resource_frontier_cuda_device_floor_family_v1"
)
SIGNED_WORK_RESOURCE_FRONTIER_CUDA_FORECAST_SCHEMA: Final = (
    "partition_free_v1_signed_work_resource_frontier_cuda_hbm_forecast_v1"
)
SIGNED_WORK_RESOURCE_FRONTIER_CUDA_RESULT_SCHEMA: Final = (
    "partition_free_v1_signed_work_resource_frontier_cuda_result_v1"
)
SIGNED_WORK_RESOURCE_FRONTIER_CUDA_EXECUTION_SCHEMA: Final = (
    "partition_free_v1_signed_work_resource_frontier_cuda_execution_v2"
)
SIGNED_WORK_RESOURCE_FRONTIER_CUDA_STATE_SCHEMA: Final = (
    "partition_free_v1_signed_work_resource_frontier_cuda_state_v1"
)
SIGNED_WORK_RESOURCE_FRONTIER_CUDA_DECISION_MARGIN_STATE_SCHEMA: Final = (
    "partition_free_v1_signed_work_resource_frontier_cuda_decision_margin_state_v1"
)
SIGNED_WORK_RESOURCE_FRONTIER_CUDA_SEQUENTIAL_BOUNDARY_SCHEMA_V2: Final = (
    "partition_free_v1_signed_work_resource_frontier_cuda_sequential_boundary_v2"
)
SIGNED_WORK_RESOURCE_FRONTIER_CUDA_DECISION_MARGIN_POLICY: Final = (
    "live_phase_A_result_and_exclusive_theorem_state_plus_exact_registered_"
    "resource_cells_with_outward_fp64_scalar_replay_and_zero_downstream_"
    "authority_v1"
)
SIGNED_WORK_RESOURCE_FRONTIER_CUDA_AUTHORITY: Final = (
    "phase_A_endogenous_resource_frontier_proposal_only_no_signed_F_scan_no_FQR_"
    "C_K_O_oracle_mask_commit_terminal_or_pareto_v1"
)
SIGNED_WORK_RESOURCE_FRONTIER_CUDA_ALGORITHM_ID: Final = (
    "independent_two_Q_branch_registered_resource_frontier_no_signed_F_scan_v1"
)
SIGNED_WORK_RESOURCE_FRONTIER_CUDA_ENDPOINT_RULE_ID: Final = (
    "derive_integer_endpoint_counts_from_certified_unit_resource_frontier_only_v1"
)
SIGNED_WORK_RESOURCE_FRONTIER_CUDA_COMPLETE_PREFIX_RULE_ID: Final = (
    "derive_every_unit_resource_prefix_count_up_to_certified_frontier_maximum_"
    "when_uint8_endpoint_capacity_suffices_v1"
)
SIGNED_WORK_RESOURCE_FRONTIER_CUDA_NO_F_OPERATOR_ID: Final = (
    "phase_A_response_geometry_only_no_signed_F_operator_v1"
)
SIGNED_WORK_RESOURCE_FRONTIER_CUDA_H200_QUALIFICATION_SCHEMA: Final = (
    "partition_free_v1_signed_work_resource_frontier_cuda_h200_qualification_v1"
)
SIGNED_WORK_RESOURCE_FRONTIER_CUDA_H200_ENGINEERING_CAPABILITY_SCHEMA: Final = (
    "partition_free_v1_signed_work_resource_frontier_cuda_h200_engineering_"
    "qualification_capability_v1"
)
SIGNED_WORK_RESOURCE_FRONTIER_CUDA_H200_BENCHMARK_SCHEMA: Final = (
    "modern_v1_phase_A_resource_frontier_h200_qualifier_v1"
)
SIGNED_WORK_RESOURCE_FRONTIER_CUDA_H200_DEVICE_CLASS: Final = "NVIDIA_H200"
# Qualified device classes: each class is qualified by its own measured
# benchmark artifact on that hardware; nothing is transferred between
# classes.  The class is resolved from the CUDA device name.
SIGNED_WORK_RESOURCE_FRONTIER_CUDA_QUALIFIED_DEVICE_CLASSES: Final = {
    "NVIDIA_H200": ("H200",),
    "NVIDIA_H100_80GB": ("H100 80GB", "H100-SXM", "H100 SXM", "H100-PCIE", "H100 PCIE"),
    "NVIDIA_A100_80GB": ("A100-SXM4-80GB", "A100 80GB", "A100-PCIE-80GB"),
}


def qualified_device_class_for_name(device_name: str) -> str | None:
    """Map a CUDA device name to its qualified class, or ``None``."""

    upper = str(device_name).upper()
    for device_class, markers in SIGNED_WORK_RESOURCE_FRONTIER_CUDA_QUALIFIED_DEVICE_CLASSES.items():
        if any(marker.upper() in upper for marker in markers):
            return device_class
    return None


def runtime_qualified_device_class() -> str:
    """The qualified class of the visible CUDA device (H200 on CUDA-less hosts).

    The Phase-A qualification is per device class: the benchmark artifact and
    the runtime that consumes it are always produced on the same node, so
    resolving the class from the visible device keeps them consistent.  A
    CUDA device outside the qualified set refuses.
    """

    if torch is None or not bool(torch.cuda.is_available()):
        return SIGNED_WORK_RESOURCE_FRONTIER_CUDA_H200_DEVICE_CLASS
    name = str(torch.cuda.get_device_name(0))
    device_class = qualified_device_class_for_name(name)
    if device_class is None:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.ROUTE_SCALE_CUDA_PENDING,
            f"CUDA device {name!r} is outside the qualified device classes",
        )
    return device_class
SIGNED_WORK_RESOURCE_FRONTIER_CUDA_FULL_ROUTE_COUNT: Final = 56_994_496
SIGNED_WORK_RESOURCE_FRONTIER_CUDA_QUALIFIED_RANKS: Final = (8, 16, 32)
SIGNED_WORK_RESOURCE_FRONTIER_CUDA_NUMERICAL_DTYPE: Final = "float64"
SIGNED_WORK_RESOURCE_FRONTIER_CUDA_QUALIFIED_BRANCHES: Final = (
    "spectral",
    "diagonal_l1",
)

_POLICY_PREFIX: Final = "partition-free-v1-signed-resource-frontier-cuda-policy-"
_DEVICE_PROBLEM_PREFIX_V1: Final = (
    "partition-free-v1-signed-resource-frontier-cuda-device-problem-v1-"
)
_DEVICE_FLOOR_FAMILY_PREFIX_V1: Final = (
    "partition-free-v1-signed-resource-frontier-cuda-device-floor-family-v1-"
)
_DEVICE_PHASE_A_ISSUANCE_TOKEN: Final = object()
_PROBLEM_PREFIX: Final = "partition-free-v1-signed-resource-frontier-cuda-problem-"
_RESPONSE_GEOMETRY_PREFIX: Final = (
    "partition-free-v1-signed-resource-frontier-response-geometry-"
)
_FORECAST_PREFIX: Final = "partition-free-v1-signed-resource-frontier-cuda-forecast-"
_RESULT_PREFIX: Final = "partition-free-v1-signed-resource-frontier-cuda-result-"
_EXECUTION_PREFIX: Final = "partition-free-v1-signed-resource-frontier-cuda-execution-"
_STATE_PREFIX: Final = "partition-free-v1-signed-resource-frontier-cuda-state-"
_FRONTIER_STATE_EXCLUSIVE_TRANSFER_TOKEN: Final = object()
_FRONTIER_DECISION_MARGIN_STATE_TOKEN: Final = object()
_QUALIFICATION_PREFIX: Final = (
    "partition-free-v1-signed-resource-frontier-cuda-h200-qualification-"
)
_QUALIFICATION_ARTIFACT_PREFIX: Final = (
    "partition-free-v1-signed-resource-frontier-cuda-h200-benchmark-artifact-"
)
_QUALIFICATION_POLICY_PREFIX: Final = (
    "partition-free-v1-signed-resource-frontier-cuda-qualified-policy-semantics-"
)
_ENGINEERING_CAPABILITY_PREFIX: Final = (
    "partition-free-v1-signed-resource-frontier-cuda-h200-engineering-capability-"
)
_ENGINEERING_CAPABILITY_TOKEN: Final = object()
_QUALIFICATION_TOKEN: Final = object()
_QUALIFIED_CUBLAS_WORKSPACE_CONFIGS: Final = (":4096:8", ":16:8")
_H200_ENGINEERING_ALEXNET_BLOCK_SIZES: Final = (
    23_232,
    307_200,
    663_552,
    884_736,
    589_824,
    37_748_736,
    16_777_216,
)
_F64_BYTES: Final = np.dtype(np.float64).itemsize
_BOOL_BYTES: Final = np.dtype(np.bool_).itemsize
_EPS: Final = np.finfo(np.float64).eps

_FRONTIER_STATE_SOURCE_TENSOR_NAMES: Final = (
    "resource",
    "factor",
    "sqrt_residual",
    "lower",
    "upper",
    "block_capacities",
    "block_scales",
)


class V1SignedWorkResourceFrontierCUDARefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_SOURCE = "stale_source"
    CUDA_REQUIRED = "cuda_required"
    ROUTE_SCALE_CUDA_PENDING = "route_scale_cuda_pending"
    PRIVATE_REFERENCE_LIMIT_EXCEEDED = "private_reference_limit_exceeded"
    HBM_FORECAST_EXCEEDED = "hbm_forecast_exceeded"
    NONFINITE = "nonfinite"
    UNRESOLVED_FRONTIER = "unresolved_frontier"
    NON_UNIT_RESOURCE = "non_unit_resource"
    NO_INTEGER_ENDPOINT = "no_integer_endpoint"
    AUTHORITY_VIOLATION = "authority_violation"
    HANDLE_EVICTED = "handle_evicted"
    ACTIVE_LEASE = "active_lease"
    CUSTODY_MUTATED = "custody_mutated"
    INCOMPLETE_DECISION_MARGIN_SOURCE = "incomplete_decision_margin_source"


class V1SignedWorkResourceFrontierCUDARefusal(RuntimeError):
    def __init__(
        self,
        code: V1SignedWorkResourceFrontierCUDARefusalCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SignedWorkResourceFrontierCUDARefusalCode,
    message: str,
) -> NoReturn:
    raise V1SignedWorkResourceFrontierCUDARefusal(code, message)


def _frontier_state_tensor_storage_invariant(value: Any) -> tuple[object, ...]:
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


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one positive integer",
        )
    result = int(value)
    if result <= 0:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.INVALID_TYPE,
            f"{name} must be nonnegative",
        )
    return result


def _positive(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one positive finite real",
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.NONFINITE,
            f"{name} must be positive finite",
        )
    return result


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one finite real",
        )
    result = float(value)
    if not math.isfinite(result):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.NONFINITE,
            f"{name} must be finite",
        )
    return result


def _nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative finite real",
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.NONFINITE,
            f"{name} must be nonnegative finite",
        )
    return result


def _closed_flags(values: tuple[bool, ...], *, owner: str) -> None:
    if any(type(value) is not bool for value in values):
        raise TypeError(f"{owner} authority flags must be exact booleans")
    if any(values):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.AUTHORITY_VIOLATION,
            f"{owner} crossed its pre-C proposal authority",
        )


def _readonly_required_action(
    value: object,
    *,
    size: int,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype != np.dtype(np.bool_) or raw.ndim != 1 or raw.shape != (size,):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.INVALID_TYPE,
            "required hard action must be one exact bool route-axis vector",
        )
    result = np.array(raw, dtype=np.bool_, order="C", copy=True)
    result.setflags(write=False)
    return result


def _readonly_array(
    value: object,
    *,
    dtype: np.dtype,
    ndim: int,
    name: str,
    shape: tuple[int | None, ...],
    nonnegative: bool,
    input_array_policy: str,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != ndim or any(
        expected is not None and observed != expected
        for observed, expected in zip(raw.shape, shape, strict=True)
    ):
        raise ValueError(f"{name} has the wrong response-geometry shape")
    if input_array_policy == "admit_readonly_c_contiguous":
        if (
            raw.dtype != dtype
            or not raw.flags.c_contiguous
            or raw.flags.writeable
        ):
            raise ValueError(
                f"{name} must be exact-dtype, readonly, and C-contiguous under "
                "the admission policy"
            )
        result = raw
    else:
        try:
            result = np.array(value, dtype=dtype, order="C", copy=True)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} is not representable in {dtype}") from error
        result.setflags(write=False)
    if dtype == np.dtype(np.float64):
        flat = result.reshape(-1)
        for start in range(0, int(flat.size), 1 << 20):
            chunk = flat[start : start + (1 << 20)]
            if not bool(np.isfinite(chunk).all()):
                raise ValueError(f"{name} must be finite")
            if nonnegative and bool(np.any(chunk < 0.0)):
                raise ValueError(f"{name} must be nonnegative")
    return result


def _all_eligible_resource_positive(
    resource: np.ndarray,
    eligible: np.ndarray,
    *,
    chunk_size: int = 1 << 20,
) -> bool:
    """Check the route predicate with bounded temporary workspace."""

    for start in range(0, int(resource.size), chunk_size):
        stop = min(int(resource.size), start + chunk_size)
        resource_chunk = resource[start:stop]
        eligible_chunk = eligible[start:stop]
        if bool(np.any(eligible_chunk & (resource_chunk <= 0.0))):
            return False
    return True


def _required_is_eligible(
    required: np.ndarray,
    eligible: np.ndarray,
    *,
    chunk_size: int = 1 << 20,
) -> bool:
    """Check required-action support without route-scale boolean temporaries."""

    for start in range(0, int(required.size), chunk_size):
        stop = min(int(required.size), start + chunk_size)
        if bool(np.any(required[start:stop] & ~eligible[start:stop])):
            return False
    return True


def _stable_l2_norm(value: np.ndarray, *, chunk_size: int = 1 << 20) -> float:
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
        raise ValueError("Phase-A factor norm is not representable in float64")
    return result


def _phase_a_response_geometry_hash(
    source: V1SignedWorkResourceFrontierCUDAProblem,
) -> str:
    """Hash exactly the resource/Q/legality coordinates consumed by Phase A.

    Phase A deliberately binds only registered resource, literal-J Q geometry,
    eligibility, and compact block capacities.  There is no corrected-work
    array or signed-F operator in this input surface.
    """

    return content_hash(
        {
            "resource": source.resource,
            "low_rank_factor": source.low_rank_factor,
            "residual_diagonal": source.residual_diagonal,
            "residual_spectral_bound": source.residual_spectral_bound,
            "eligible_mask": source.eligible_mask,
            "quadratic_budget": source.quadratic_budget,
            "route_axis_hash": source.route_axis_hash,
            "source_geometry_hash": source.source_geometry_hash,
            "block_capacities": source.block_capacities,
        },
        prefix=_RESPONSE_GEOMETRY_PREFIX,
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkResourceFrontierCUDAProblem:
    """Canonical operator-free input for endogenous Phase-A resource discovery.

    A factory extracts these coordinates exactly from ``V1RelaxedKKTProblem``;
    direct construction supports geometry-only qualification without ever
    allocating its unrelated corrected-work vector.  The problem owns only
    the registered resource axis, literal-J response geometry, eligibility,
    and compact legality blocks.
    Corrected signed work is a Phase-B coordinate and is neither retained as a
    ``V1SignedWorkOperatorCUDA`` nor scanned by this object.  Construction is a
    pre-C proposal registration only and grants no downstream authority.
    """

    problem_id: str
    resource: np.ndarray = field(repr=False, compare=False)
    low_rank_factor: np.ndarray = field(repr=False, compare=False)
    residual_diagonal: np.ndarray = field(repr=False, compare=False)
    residual_spectral_bound: float
    eligible_mask: np.ndarray = field(repr=False, compare=False)
    quadratic_budget: float
    route_axis_hash: str
    source_geometry_hash: str
    source_registration_hash: str
    registered_d_coordinate_hash: str
    source_bundle_authoritative: bool
    block_capacities: tuple[V1RelaxedKKTBlockCapacity, ...] = ()
    required_hard_action: np.ndarray | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    input_array_policy: str = "copy_to_bytes"
    no_signed_f_operator_retained: bool = True
    no_signed_f_scan: bool = True
    c_stream_outcomes_seen: bool = False
    fqr_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_RESOURCE_FRONTIER_CUDA_AUTHORITY
    schema: str = SIGNED_WORK_RESOURCE_FRONTIER_CUDA_PROBLEM_SCHEMA
    response_geometry_hash: str = field(init=False)
    required_hard_action_hash: str | None = field(init=False)
    required_lower_bound_hash: str = field(init=False)
    baseline_resource: float = field(init=False)
    _factor_frobenius_norm: float = field(init=False, repr=False, compare=False)
    _problem_hash: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "problem_id", _identity(self.problem_id, name="problem id"))
        if self.input_array_policy not in {
            "copy_to_bytes",
            "admit_readonly_c_contiguous",
        }:
            raise ValueError("unsupported Phase-A input array policy")
        raw_eligible = np.asarray(self.eligible_mask)
        if (
            raw_eligible.dtype != np.dtype(np.bool_)
            or raw_eligible.ndim != 1
            or raw_eligible.size <= 0
        ):
            raise ValueError("eligible_mask must be one exact nonempty bool route axis")
        route_count = int(raw_eligible.size)
        eligible = _readonly_array(
            self.eligible_mask,
            dtype=np.dtype(np.bool_),
            ndim=1,
            name="eligible_mask",
            shape=(route_count,),
            nonnegative=False,
            input_array_policy=self.input_array_policy,
        )
        resource = _readonly_array(
            self.resource,
            dtype=np.dtype(np.float64),
            ndim=1,
            name="resource",
            shape=(route_count,),
            nonnegative=True,
            input_array_policy=self.input_array_policy,
        )
        factor = _readonly_array(
            self.low_rank_factor,
            dtype=np.dtype(np.float64),
            ndim=2,
            name="low_rank_factor",
            shape=(None, route_count),
            nonnegative=False,
            input_array_policy=self.input_array_policy,
        )
        residual = _readonly_array(
            self.residual_diagonal,
            dtype=np.dtype(np.float64),
            ndim=1,
            name="residual_diagonal",
            shape=(route_count,),
            nonnegative=True,
            input_array_policy=self.input_array_policy,
        )
        if not _all_eligible_resource_positive(resource, eligible):
            raise ValueError("resource must be positive on every eligible route")
        rho = _nonnegative(
            self.residual_spectral_bound,
            name="residual_spectral_bound",
        )
        maximum_residual = float(np.max(residual)) if residual.size else 0.0
        guard = 64.0 * _EPS * max(1.0, rho, maximum_residual)
        if rho + guard < maximum_residual:
            raise ValueError(
                "residual_spectral_bound must cover max residual_diagonal"
            )
        rho = max(rho, maximum_residual)
        quadratic_budget = _nonnegative(
            self.quadratic_budget,
            name="quadratic_budget",
        )
        blocks = tuple(self.block_capacities)
        if any(type(item) is not V1RelaxedKKTBlockCapacity for item in blocks):
            raise TypeError("block_capacities require exact compact block objects")
        if blocks:
            previous_stop = 0
            seen: set[str] = set()
            for block in blocks:
                if block.start != previous_stop or block.stop > route_count:
                    raise ValueError(
                        "compact block capacities must be contiguous and on-axis"
                    )
                if block.block_id in seen:
                    raise ValueError("compact block capacity IDs must be unique")
                seen.add(block.block_id)
                previous_stop = block.stop
            if previous_stop != route_count:
                raise ValueError("compact block capacities must cover the route axis")
        for name in (
            "route_axis_hash",
            "source_geometry_hash",
            "source_registration_hash",
            "registered_d_coordinate_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(self, "eligible_mask", eligible)
        object.__setattr__(self, "resource", resource)
        object.__setattr__(self, "low_rank_factor", factor)
        object.__setattr__(self, "residual_diagonal", residual)
        object.__setattr__(self, "residual_spectral_bound", rho)
        object.__setattr__(self, "quadratic_budget", quadratic_budget)
        object.__setattr__(self, "block_capacities", blocks)
        object.__setattr__(self, "_factor_frobenius_norm", _stable_l2_norm(factor))
        if type(self.source_bundle_authoritative) is not bool:
            raise TypeError("source_bundle_authoritative must be an exact bool")
        if not self.source_bundle_authoritative:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.AUTHORITY_VIOLATION,
                "operator-free Phase A requires an authoritative registered D source",
            )
        required: np.ndarray | None = None
        if self.required_hard_action is not None:
            required = _readonly_required_action(
                self.required_hard_action,
                size=route_count,
            )
            if not _required_is_eligible(required, eligible):
                _refuse(
                    V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                    "required hard action contains an ineligible route",
                )
            for block in blocks:
                used = int(np.count_nonzero(required[block.start : block.stop]))
                if used > block.removal_capacity:
                    _refuse(
                        V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                        "required hard action exceeds a registered block capacity",
                    )
        for name in ("no_signed_f_operator_retained", "no_signed_f_scan"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact bool")
            if not getattr(self, name):
                _refuse(
                    V1SignedWorkResourceFrontierCUDARefusalCode.AUTHORITY_VIOLATION,
                    "operator-free Phase A cannot retain or scan signed F",
                )
        _closed_flags(
            (
                self.c_stream_outcomes_seen,
                self.fqr_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="operator-free resource-frontier CUDA problem",
        )
        if (
            self.authority_scope != SIGNED_WORK_RESOURCE_FRONTIER_CUDA_AUTHORITY
            or self.schema != SIGNED_WORK_RESOURCE_FRONTIER_CUDA_PROBLEM_SCHEMA
        ):
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.AUTHORITY_VIOLATION,
                "operator-free resource-frontier problem semantics changed",
            )
        response_hash = _phase_a_response_geometry_hash(self)
        required_hash = None if required is None else content_hash(required)
        required_lower_bound = (
            np.zeros(route_count, dtype=np.bool_)
            if required is None
            else required
        )
        lower_bound_hash = content_hash(required_lower_bound)
        baseline_resource = float(
            np.dot(resource, required_lower_bound.astype(np.float64))
        )
        object.__setattr__(self, "required_hard_action", required)
        object.__setattr__(self, "response_geometry_hash", response_hash)
        object.__setattr__(self, "required_hard_action_hash", required_hash)
        object.__setattr__(self, "required_lower_bound_hash", lower_bound_hash)
        object.__setattr__(self, "baseline_resource", baseline_resource)
        object.__setattr__(
            self,
            "_problem_hash",
            content_hash(
                {
                    "schema": self.schema,
                    "problem_id": self.problem_id,
                    "response_geometry_hash": response_hash,
                    "source_registration_hash": self.source_registration_hash,
                    "registered_d_coordinate_hash": self.registered_d_coordinate_hash,
                    "source_bundle_authoritative": True,
                    "required_hard_action_hash": required_hash,
                    "required_lower_bound_hash": lower_bound_hash,
                    "baseline_resource": baseline_resource,
                    "no_signed_f_operator_retained": True,
                    "no_signed_f_scan": True,
                    "authority_scope": self.authority_scope,
                },
                prefix=_PROBLEM_PREFIX,
            ),
        )

    @property
    def problem_hash(self) -> str:
        return self._problem_hash

    @property
    def route_count(self) -> int:
        return int(self.eligible_mask.size)

    @property
    def response_rank(self) -> int:
        return int(self.low_rank_factor.shape[0])

    @property
    def rank(self) -> int:
        return self.response_rank


def _device_phase_a_resource_summaries(
    *,
    resource: Any,
    eligible: Any,
    required: Any,
    blocks: tuple[V1RelaxedKKTBlockCapacity, ...],
) -> tuple[float, float | None, bool, int, int, float, tuple[int, ...]]:
    """Reduce live structural coordinates to bounded host scalars only.

    The reductions are descriptive registration facts, not reconstructed
    route arrays.  The canonical unit-resource axis is handled analytically
    from its one-scalar broadcast plus scalar mask counts.  A nonuniform axis
    is admitted without workspace only when every route is eligible and the
    required action is empty; other cases refuse until an upstream structural
    receipt supplies fused masked summaries.  No advanced-index gather,
    route-scale temporary, or route value D2H occurs here.
    """

    if torch is None:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.CUDA_REQUIRED,
            "device Phase A requires PyTorch tensor support",
        )
    route_count = int(resource.numel())
    windows = (
        tuple((block.start, block.stop) for block in blocks)
        if blocks
        else ((0, route_count),)
    )
    eligible_counts: list[int] = []
    for start, stop in windows:
        local_eligible = eligible[start:stop]
        count = int(torch.count_nonzero(local_eligible).item())
        eligible_counts.append(count)
    eligible_count = sum(eligible_counts)
    authenticated_empty_support = eligible_count == 0
    required_count = int(torch.count_nonzero(required).item())
    route_minimum = float(torch.min(resource).item())
    route_maximum = float(torch.max(resource).item())
    literal_unit_resource = route_minimum == 1.0 and route_maximum == 1.0
    if literal_unit_resource:
        uniform: float | None = None if authenticated_empty_support else 1.0
        resource_support = float(eligible_count)
        baseline_resource = float(required_count)
    elif eligible_count == route_count and required_count == 0:
        uniform = route_minimum if route_minimum == route_maximum else None
        resource_support = float(torch.sum(resource).item())
        baseline_resource = 0.0
    else:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.UNRESOLVED_FRONTIER,
            "nonuniform masked device resource requires a preregistered fused "
            "structural-summary receipt; Phase A will not allocate an "
            "unforecast route gather",
        )
    if (
        not math.isfinite(resource_support)
        or not math.isfinite(baseline_resource)
        or resource_support < 0.0
        or baseline_resource < 0.0
    ):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.NONFINITE,
            "device resource summaries are not finite nonnegative scalars",
        )
    return (
        resource_support,
        uniform,
        literal_unit_resource,
        eligible_count,
        required_count,
        baseline_resource,
        tuple(eligible_counts) if blocks else (),
    )


class V1SignedWorkResourceFrontierCUDADeviceProblemV1:
    """Floor-free, operator-free, zero-copy device input for Phase A.

    This live view borrows its own literal-J geometry lease and aliases the
    exact structural route tensors registered by the caller.  It never owns or
    observes a signed-work operator and has no resource-floor field.  Its
    serial identities commit sealed source receipts and tensor custody IDs,
    never tensor contents or a seed Phase-B problem hash.
    """

    __slots__ = (
        "problem_id",
        "quadratic_budget",
        "block_capacities",
        "source_registration_hash",
        "registered_d_coordinate_hash",
        "structural_source_receipt_hash",
        "source_bundle_authoritative",
        "route_axis_hash",
        "source_geometry_hash",
        "resource_tensor_identity_hash",
        "eligible_mask_identity_hash",
        "required_mask_identity_hash",
        "response_geometry_hash",
        "required_hard_action_hash",
        "required_lower_bound_hash",
        "baseline_resource",
        "authority_scope",
        "schema",
        "_geometry_receipt_hash",
        "_geometry_lease",
        "_resource",
        "_eligible",
        "_required",
        "_factor",
        "_residual",
        "_route_count",
        "_rank",
        "_factor_frobenius_norm",
        "_residual_sqrt_l2_norm",
        "_residual_diagonal_sum",
        "_residual_spectral_bound",
        "_resource_support",
        "_uniform_eligible_resource",
        "_literal_unit_resource",
        "_eligible_count",
        "_required_count",
        "_block_eligible_counts",
        "_tensor_invariants",
        "_problem_hash",
        "_lease_transferred",
        "_closed",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        _issuance_token: object,
        geometry_handle: V1SampledDeviceResidentDKKTGeometryDeviceHandle,
        resource_tensor: Any,
        eligible_mask_tensor: Any,
        required_hard_action_tensor: Any,
        quadratic_budget: float,
        block_capacities: tuple[V1RelaxedKKTBlockCapacity, ...],
        route_axis_hash: str,
        source_geometry_hash: str,
        source_registration_hash: str,
        registered_d_coordinate_hash: str,
        structural_source_receipt_hash: str,
        resource_tensor_identity_hash: str,
        eligible_mask_identity_hash: str,
        required_mask_identity_hash: str,
        source_bundle_authoritative: bool,
    ) -> None:
        if _issuance_token is not _DEVICE_PHASE_A_ISSUANCE_TOKEN:
            raise TypeError("device Phase-A problems are factory-issued only")
        if torch is None:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.CUDA_REQUIRED,
                "device Phase A requires PyTorch",
            )
        if type(geometry_handle) is not V1SampledDeviceResidentDKKTGeometryDeviceHandle:
            raise TypeError("device Phase A requires the exact D-geometry handle")
        if type(source_bundle_authoritative) is not bool or not source_bundle_authoritative:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.AUTHORITY_VIOLATION,
                "device Phase A requires an authoritative frozen D source",
            )
        route_axis = _identity(route_axis_hash, name="route axis hash")
        source_geometry = _identity(
            source_geometry_hash,
            name="source geometry hash",
        )
        receipt = geometry_handle.receipt
        if receipt.route_axis_hash != route_axis:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                "device Phase-A geometry and structural route axes differ",
            )
        route_count = int(receipt.route_count)
        rank = int(receipt.factor_shape[0])
        device = torch.device(receipt.device_identity)

        def _admit(value: Any, *, shape: tuple[int, ...], dtype: Any, name: str) -> Any:
            scalar_broadcast = (
                torch.is_tensor(value)
                and int(value.numel()) == math.prod(shape)
                and int(value.untyped_storage().nbytes()) == int(value.element_size())
                and all(int(step) == 0 for step in value.stride())
            )
            if (
                not torch.is_tensor(value)
                or type(value) is not torch.Tensor
                or tuple(int(item) for item in value.shape) != shape
                or value.dtype != dtype
                or value.device != device
                or value.requires_grad
                or not (value.is_contiguous() or scalar_broadcast)
            ):
                _refuse(
                    V1SignedWorkResourceFrontierCUDARefusalCode.INVALID_TYPE,
                    f"{name} is not an exact same-device registered tensor",
                )
            return value

        resource = _admit(
            resource_tensor,
            shape=(route_count,),
            dtype=torch.float64,
            name="resource",
        )
        eligible = _admit(
            eligible_mask_tensor,
            shape=(route_count,),
            dtype=torch.bool,
            name="eligible mask",
        )
        required = _admit(
            required_hard_action_tensor,
            shape=(route_count,),
            dtype=torch.bool,
            name="required hard action",
        )
        resource_minimum = float(torch.min(resource).item())
        resource_maximum = float(torch.max(resource).item())
        if (
            not math.isfinite(resource_minimum)
            or not math.isfinite(resource_maximum)
            or resource_minimum <= 0.0
        ):
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.NONFINITE,
                "device Phase-A resource must be finite positive on its sealed axis",
            )
        blocks = tuple(block_capacities)
        if any(type(item) is not V1RelaxedKKTBlockCapacity for item in blocks):
            raise TypeError("device Phase A requires exact compact block capacities")
        if blocks:
            previous = 0
            for block in blocks:
                if block.start != previous or block.stop > route_count:
                    _refuse(
                        V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                        "device Phase-A blocks do not form an ordered route cover",
                    )
                previous = block.stop
                if int(torch.count_nonzero(required[block.start:block.stop]).item()) > (
                    block.removal_capacity
                ):
                    _refuse(
                        V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                        "required action exceeds a compact block capacity",
                    )
            if previous != route_count or len({item.block_id for item in blocks}) != len(blocks):
                _refuse(
                    V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                    "device Phase-A blocks are incomplete or duplicated",
                )
        q_budget = _nonnegative(quadratic_budget, name="quadratic budget")
        lease = geometry_handle.borrow_for_kkt(
            expected_geometry_receipt_hash=receipt.receipt_hash,
            expected_session_design_hash=receipt.session_design_hash,
            expected_row_free_finalization_receipt_hash=(
                receipt.row_free_finalization_receipt_hash
            ),
            expected_materialization_rank_receipt_hash=(
                receipt.materialization_rank_receipt_hash
            ),
            expected_registered_proposal_fd_rank=receipt.registered_proposal_fd_rank,
            expected_route_axis_hash=route_axis,
            expected_device=device,
        )
        try:
            factor = lease.factor
            residual = lease.residual_diagonal
            summaries = _device_phase_a_resource_summaries(
                resource=resource,
                eligible=eligible,
                required=required,
                blocks=blocks,
            )
            (
                resource_support,
                uniform_resource,
                literal_unit,
                eligible_count,
                required_count,
                baseline_resource,
                block_eligible_counts,
            ) = summaries
            resource_identity = _identity(
                resource_tensor_identity_hash,
                name="resource tensor identity hash",
            )
            eligible_identity = _identity(
                eligible_mask_identity_hash,
                name="eligible-mask identity hash",
            )
            required_identity = _identity(
                required_mask_identity_hash,
                name="required-mask identity hash",
            )
            structural_receipt = _identity(
                structural_source_receipt_hash,
                name="structural source receipt hash",
            )
            source_registration = _identity(
                source_registration_hash,
                name="source registration hash",
            )
            d_coordinate = _identity(
                registered_d_coordinate_hash,
                name="registered D coordinate hash",
            )
            source_identity = content_hash(
                {
                    "geometry_receipt_hash": receipt.receipt_hash,
                    "route_axis_hash": route_axis,
                    "source_geometry_hash": source_geometry,
                    "source_registration_hash": source_registration,
                    "registered_d_coordinate_hash": d_coordinate,
                    "structural_source_receipt_hash": structural_receipt,
                    "resource_tensor_identity_hash": resource_identity,
                    "eligible_mask_identity_hash": eligible_identity,
                    "required_mask_identity_hash": required_identity,
                    "quadratic_budget": q_budget,
                    "block_capacities": blocks,
                    "floor_free": True,
                    "operator_free": True,
                },
                prefix=_DEVICE_PROBLEM_PREFIX_V1 + "source-",
            )
            self.problem_id = f"{source_registration}-endogenous-device-resource-frontier"
            self.quadratic_budget = q_budget
            self.block_capacities = blocks
            self.source_registration_hash = source_registration
            self.registered_d_coordinate_hash = d_coordinate
            self.structural_source_receipt_hash = structural_receipt
            self.source_bundle_authoritative = True
            self.route_axis_hash = route_axis
            self.source_geometry_hash = source_geometry
            self.resource_tensor_identity_hash = resource_identity
            self.eligible_mask_identity_hash = eligible_identity
            self.required_mask_identity_hash = required_identity
            self.response_geometry_hash = source_identity
            self.required_hard_action_hash = (
                required_identity if required_count else None
            )
            self.required_lower_bound_hash = required_identity
            self.baseline_resource = baseline_resource
            self.authority_scope = SIGNED_WORK_RESOURCE_FRONTIER_CUDA_AUTHORITY
            self.schema = SIGNED_WORK_RESOURCE_FRONTIER_CUDA_DEVICE_PROBLEM_SCHEMA_V1
            self._geometry_receipt_hash = receipt.receipt_hash
            self._geometry_lease = lease
            self._resource = resource
            self._eligible = eligible
            self._required = required
            self._factor = factor
            self._residual = residual
            self._route_count = route_count
            self._rank = rank
            self._factor_frobenius_norm = float(receipt.factor_frobenius_norm)
            self._residual_sqrt_l2_norm = float(receipt.residual_sqrt_l2_norm)
            self._residual_diagonal_sum = self._residual_sqrt_l2_norm ** 2
            self._residual_spectral_bound = float(receipt.residual_spectral_bound)
            self._resource_support = resource_support
            self._uniform_eligible_resource = uniform_resource
            self._literal_unit_resource = literal_unit
            self._eligible_count = eligible_count
            self._required_count = required_count
            self._block_eligible_counts = block_eligible_counts
            self._tensor_invariants = tuple(
                _frontier_state_tensor_storage_invariant(item)
                for item in (resource, eligible, required, factor, residual)
            )
            self._lease_transferred = False
            self._closed = False
            self._problem_hash = content_hash(
                {
                    "schema": self.schema,
                    "problem_id": self.problem_id,
                    "sealed_source_identity_hash": source_identity,
                    "required_lower_bound_hash": required_identity,
                    "no_resource_floor": True,
                    "no_signed_f_operator": True,
                    "route_scale_d2h_bytes": 0,
                    "route_scale_numpy_arrays_constructed": 0,
                    "authority_scope": self.authority_scope,
                },
                prefix=_DEVICE_PROBLEM_PREFIX_V1,
            )
            self.assert_integrity()
        except BaseException:
            lease.close()
            raise

    @property
    def problem_hash(self) -> str:
        return self._problem_hash

    @property
    def route_count(self) -> int:
        return self._route_count

    @property
    def response_rank(self) -> int:
        return self._rank

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def resource(self) -> Any:
        self.assert_structural_identity()
        return self._resource

    @property
    def low_rank_factor(self) -> Any:
        self.assert_integrity()
        return self._factor

    @property
    def residual_diagonal(self) -> Any:
        self.assert_integrity()
        return self._residual

    @property
    def residual_spectral_bound(self) -> float:
        return self._residual_spectral_bound

    @property
    def eligible_mask(self) -> Any:
        self.assert_structural_identity()
        return self._eligible

    @property
    def required_hard_action(self) -> Any:
        self.assert_structural_identity()
        return self._required

    @property
    def geometry_receipt_hash(self) -> str:
        return self._geometry_receipt_hash

    @property
    def resource_support(self) -> float:
        return self._resource_support

    @property
    def uniform_eligible_resource(self) -> float | None:
        return self._uniform_eligible_resource

    @property
    def literal_unit_resource(self) -> bool:
        return self._literal_unit_resource

    @property
    def eligible_count(self) -> int:
        return self._eligible_count

    @property
    def required_count(self) -> int:
        return self._required_count

    @property
    def block_eligible_counts(self) -> tuple[int, ...]:
        return self._block_eligible_counts

    @property
    def device(self) -> Any:
        return self._resource.device

    @property
    def route_scale_d2h_bytes(self) -> int:
        return 0

    @property
    def route_scale_numpy_arrays_constructed(self) -> int:
        return 0

    @property
    def lease_transferred(self) -> bool:
        return self._lease_transferred

    def assert_structural_identity(self) -> str:
        observed = tuple(
            _frontier_state_tensor_storage_invariant(item)
            for item in (
                self._resource,
                self._eligible,
                self._required,
                self._factor,
                self._residual,
            )
        )
        if observed != self._tensor_invariants:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.CUSTODY_MUTATED,
                "device Phase-A source tensor storage or version changed",
            )
        return self._problem_hash

    def assert_integrity(self) -> str:
        if self._closed or self._lease_transferred or self._geometry_lease is None:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.HANDLE_EVICTED,
                "device Phase-A problem no longer owns its geometry lease",
            )
        receipt = self._geometry_lease.receipt
        if receipt.receipt_hash != self._geometry_receipt_hash:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                "device Phase-A geometry receipt changed",
            )
        return self.assert_structural_identity()

    def _take_geometry_lease_for_state(self) -> V1SampledDeviceResidentDKKTGeometryLease:
        self.assert_integrity()
        lease = self._geometry_lease
        if lease is None:  # pragma: no cover - closed by assert_integrity.
            raise RuntimeError("device Phase-A geometry lease disappeared")
        self._geometry_lease = None
        self._lease_transferred = True
        return lease

    def close(self) -> None:
        if self._closed or self._lease_transferred:
            return
        lease = self._geometry_lease
        self._geometry_lease = None
        self._closed = True
        if lease is not None:
            lease.close()

    def __enter__(self) -> "V1SignedWorkResourceFrontierCUDADeviceProblemV1":
        self.assert_integrity()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __reduce__(self) -> NoReturn:
        raise TypeError("device Phase-A problems are nonserializable")

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("device Phase-A problems are nonserializable")


def issue_v1_signed_work_resource_frontier_cuda_device_problem_v1(
    *,
    geometry_handle: V1SampledDeviceResidentDKKTGeometryDeviceHandle,
    resource_tensor: Any,
    eligible_mask_tensor: Any,
    required_hard_action_tensor: Any,
    quadratic_budget: float,
    block_capacities: tuple[V1RelaxedKKTBlockCapacity, ...],
    route_axis_hash: str,
    source_geometry_hash: str,
    source_registration_hash: str,
    registered_d_coordinate_hash: str,
    structural_source_receipt_hash: str,
    resource_tensor_identity_hash: str,
    eligible_mask_identity_hash: str,
    required_mask_identity_hash: str,
    source_bundle_authoritative: bool,
) -> V1SignedWorkResourceFrontierCUDADeviceProblemV1:
    """Issue the canonical floor-free view from live registered sources."""

    return V1SignedWorkResourceFrontierCUDADeviceProblemV1(
        _issuance_token=_DEVICE_PHASE_A_ISSUANCE_TOKEN,
        geometry_handle=geometry_handle,
        resource_tensor=resource_tensor,
        eligible_mask_tensor=eligible_mask_tensor,
        required_hard_action_tensor=required_hard_action_tensor,
        quadratic_budget=quadratic_budget,
        block_capacities=block_capacities,
        route_axis_hash=route_axis_hash,
        source_geometry_hash=source_geometry_hash,
        source_registration_hash=source_registration_hash,
        registered_d_coordinate_hash=registered_d_coordinate_hash,
        structural_source_receipt_hash=structural_source_receipt_hash,
        resource_tensor_identity_hash=resource_tensor_identity_hash,
        eligible_mask_identity_hash=eligible_mask_identity_hash,
        required_mask_identity_hash=required_mask_identity_hash,
        source_bundle_authoritative=source_bundle_authoritative,
    )


def adapt_v1_signed_work_device_problem_v2_to_resource_frontier_cuda_v1(
    *,
    source_problem: V1SignedWorkLeastWorkCUDADeviceProblemV2,
    geometry_handle: V1SampledDeviceResidentDKKTGeometryDeviceHandle,
) -> V1SignedWorkResourceFrontierCUDADeviceProblemV1:
    """Parity/replay adapter; canonical issuance does not require a seed floor."""

    if type(source_problem) is not V1SignedWorkLeastWorkCUDADeviceProblemV2:
        raise TypeError("device Phase-A adapter requires exact DeviceProblemV2")
    source_problem.assert_integrity()
    return issue_v1_signed_work_resource_frontier_cuda_device_problem_v1(
        geometry_handle=geometry_handle,
        resource_tensor=source_problem.resource,
        eligible_mask_tensor=source_problem.eligible_mask,
        required_hard_action_tensor=source_problem.required_hard_action,
        quadratic_budget=source_problem.quadratic_budget,
        block_capacities=source_problem.block_capacities,
        route_axis_hash=source_problem.route_axis_hash,
        source_geometry_hash=source_problem.source_geometry_hash,
        source_registration_hash=source_problem.source_registration_hash,
        registered_d_coordinate_hash=source_problem.registered_d_coordinate_hash,
        structural_source_receipt_hash=(
            source_problem.structural_source_receipt_hash
        ),
        resource_tensor_identity_hash=source_problem.resource_tensor_identity_hash,
        eligible_mask_identity_hash=source_problem.eligible_mask_identity_hash,
        required_mask_identity_hash=source_problem.required_mask_identity_hash,
        source_bundle_authoritative=source_problem.source_bundle_authoritative,
    )


def build_v1_signed_work_resource_frontier_cuda_problem_from_relaxed_kkt(
    *,
    problem_id: str,
    response_geometry: V1RelaxedKKTProblem,
    source_registration_hash: str,
    registered_d_coordinate_hash: str,
    source_bundle_authoritative: bool,
    required_hard_action: np.ndarray | None = None,
) -> V1SignedWorkResourceFrontierCUDAProblem:
    """Extract exactly Phase-A resource/Q/legality coordinates without F."""

    if type(response_geometry) is not V1RelaxedKKTProblem:
        raise TypeError("Phase-A geometry extraction requires exact relaxed-KKT source")
    return V1SignedWorkResourceFrontierCUDAProblem(
        problem_id=problem_id,
        resource=response_geometry.resource,
        low_rank_factor=response_geometry.low_rank_factor,
        residual_diagonal=response_geometry.residual_diagonal,
        residual_spectral_bound=response_geometry.residual_spectral_bound,
        eligible_mask=response_geometry.eligible_mask,
        quadratic_budget=response_geometry.quadratic_budget,
        route_axis_hash=response_geometry.route_axis_hash,
        source_geometry_hash=response_geometry.source_geometry_hash,
        source_registration_hash=source_registration_hash,
        registered_d_coordinate_hash=registered_d_coordinate_hash,
        source_bundle_authoritative=source_bundle_authoritative,
        block_capacities=response_geometry.block_capacities,
        required_hard_action=required_hard_action,
        input_array_policy="admit_readonly_c_contiguous",
    )


def adapt_v1_signed_work_least_work_problem_to_resource_frontier_cuda(
    problem: V1SignedWorkLeastWorkCUDAProblem,
) -> V1SignedWorkResourceFrontierCUDAProblem:
    """Read a legacy Phase-B carrier into the canonical operator-free seam."""

    if type(problem) is not V1SignedWorkLeastWorkCUDAProblem:
        raise TypeError("legacy resource-frontier adapter requires exact least-work problem")
    problem.assert_integrity()
    return build_v1_signed_work_resource_frontier_cuda_problem_from_relaxed_kkt(
        problem_id=f"{problem.source_registration_hash}-endogenous-resource-frontier",
        response_geometry=problem.source_problem,
        source_registration_hash=problem.source_registration_hash,
        registered_d_coordinate_hash=problem.registered_d_coordinate_hash,
        source_bundle_authoritative=problem.source_bundle_authoritative,
        required_hard_action=problem.required_hard_action,
    )


def _canonical_phase_a_problem(
    problem: V1SignedWorkResourceFrontierCUDAProblem
    | V1SignedWorkResourceFrontierCUDADeviceProblemV1
    | V1SignedWorkLeastWorkCUDAProblem,
) -> V1SignedWorkResourceFrontierCUDAProblem | V1SignedWorkResourceFrontierCUDADeviceProblemV1:
    if type(problem) is V1SignedWorkResourceFrontierCUDAProblem:
        return problem
    if type(problem) is V1SignedWorkResourceFrontierCUDADeviceProblemV1:
        return problem
    if type(problem) is V1SignedWorkLeastWorkCUDAProblem:
        return adapt_v1_signed_work_least_work_problem_to_resource_frontier_cuda(problem)
    raise TypeError(
        "resource-frontier Phase A requires exact operator-free or legacy least-work problem"
    )


def current_v1_signed_work_resource_frontier_cuda_solver_implementation_sha256() -> str:
    """Hash the exact module bytes that own the qualified CUDA solver seam."""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class V1SignedWorkResourceFrontierCUDAH200EngineeringCapability:
    """Issuer-only bootstrap for the exact synthetic engineering benchmark."""

    source_tree_hash: str
    source_archive_sha256: str
    solver_implementation_sha256: str
    qualified_policy_semantics_hash: str
    route_count: int
    ranks: tuple[int, ...]
    synthetic_target_fraction: float
    device_class: str
    synthetic_engineering_only: bool
    production_decision_authority: bool
    schema: str = SIGNED_WORK_RESOURCE_FRONTIER_CUDA_H200_ENGINEERING_CAPABILITY_SCHEMA
    _capability_hash: str = field(init=False, repr=False, compare=False)

    def __init__(self, *, _issuer_token: object, **values: object) -> None:
        if _issuer_token is not _ENGINEERING_CAPABILITY_TOKEN:
            raise TypeError("H200 engineering capabilities are issuer-only")
        expected = {
            name
            for name, declaration in self.__dataclass_fields__.items()
            if declaration.init and name != "schema"
        }
        if set(values) != expected:
            raise TypeError("H200 engineering capability fields differ")
        for name in expected:
            object.__setattr__(self, name, values[name])
        object.__setattr__(
            self,
            "schema",
            SIGNED_WORK_RESOURCE_FRONTIER_CUDA_H200_ENGINEERING_CAPABILITY_SCHEMA,
        )
        object.__setattr__(self, "_capability_hash", "")
        self.__post_init__()
        object.__setattr__(
            self,
            "_capability_hash",
            content_hash(self._payload(), prefix=_ENGINEERING_CAPABILITY_PREFIX),
        )

    def __post_init__(self) -> None:
        _identity(self.source_tree_hash, name="source tree hash")
        for name in ("source_archive_sha256", "solver_implementation_sha256"):
            value = _identity(getattr(self, name), name=name.replace("_", " "))
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{name} is not a lowercase SHA-256")
        _identity(
            self.qualified_policy_semantics_hash,
            name="qualified policy semantics hash",
        )
        if (
            self.route_count != SIGNED_WORK_RESOURCE_FRONTIER_CUDA_FULL_ROUTE_COUNT
            or self.ranks != SIGNED_WORK_RESOURCE_FRONTIER_CUDA_QUALIFIED_RANKS
            or self.synthetic_target_fraction != 1.0 / 32.0
            or self.device_class
            not in SIGNED_WORK_RESOURCE_FRONTIER_CUDA_QUALIFIED_DEVICE_CLASSES
            or self.schema
            != SIGNED_WORK_RESOURCE_FRONTIER_CUDA_H200_ENGINEERING_CAPABILITY_SCHEMA
        ):
            raise ValueError("H200 engineering capability contract changed")
        if self.synthetic_engineering_only is not True:
            raise ValueError("H200 engineering capability is not synthetic-only")
        if self.production_decision_authority is not False:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.AUTHORITY_VIOLATION,
                "engineering capability cannot grant production decision authority",
            )

    def _payload(self) -> Mapping[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "_capability_hash"
        }

    @property
    def capability_hash(self) -> str:
        return self._capability_hash

    def assert_integrity(self) -> None:
        if self._capability_hash != content_hash(
            self._payload(), prefix=_ENGINEERING_CAPABILITY_PREFIX
        ):
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                "H200 engineering capability content changed",
            )


@dataclass(frozen=True, slots=True, init=False)
class V1SignedWorkResourceFrontierCUDAH200QualificationReceipt:
    """Issuer-only authority to execute the qualified full-route H200 seam.

    The receipt is engineering evidence only.  It opens Phase-A CUDA execution
    for the exact numerical policy and implementation it names; it grants no
    scientific, F/Q/R, calibration, oracle, commit, terminal, or Pareto
    authority.
    """

    qualification_id: str
    benchmark_artifact_hash: str
    source_tree_hash: str
    source_archive_sha256: str
    solver_implementation_sha256: str
    qualified_policy_semantics_hash: str
    device_class: str
    device_name: str
    device_capability: str
    torch_version: str
    cuda_version: str
    cublas_workspace_config: str
    route_count: int
    ranks: tuple[int, ...]
    rank_case_hashes: tuple[str, ...]
    maximum_measured_hbm_fraction: float
    small_domain_parity_hash: str
    strict_deterministic_algorithms: bool
    deterministic_replay: bool
    zero_route_vector_d2h: bool
    zero_signed_f_scan: bool
    synthetic_engineering_only: bool
    production_decision_authority: bool
    algorithm_id: str
    policy_schema: str
    problem_schema: str
    execution_schema: str
    no_f_operator_id: str
    numerical_dtype: str
    branches: tuple[str, ...]
    schema: str = SIGNED_WORK_RESOURCE_FRONTIER_CUDA_H200_QUALIFICATION_SCHEMA
    _qualification_hash: str = field(init=False, repr=False, compare=False)

    def __init__(self, *, _issuer_token: object, **values: object) -> None:
        if _issuer_token is not _QUALIFICATION_TOKEN:
            raise TypeError("H200 Phase-A qualification receipts are issuer-only")
        expected = {
            name
            for name, declaration in self.__dataclass_fields__.items()
            if declaration.init and name != "schema"
        }
        if set(values) != expected:
            raise TypeError(
                "H200 qualification receipt fields differ: "
                f"missing={sorted(expected.difference(values))}, "
                f"extra={sorted(set(values).difference(expected))}"
            )
        for name in expected:
            object.__setattr__(self, name, values[name])
        object.__setattr__(
            self,
            "schema",
            SIGNED_WORK_RESOURCE_FRONTIER_CUDA_H200_QUALIFICATION_SCHEMA,
        )
        object.__setattr__(self, "_qualification_hash", "")
        self.__post_init__()
        object.__setattr__(
            self,
            "_qualification_hash",
            content_hash(self._payload(), prefix=_QUALIFICATION_PREFIX),
        )

    def __post_init__(self) -> None:
        for name in (
            "qualification_id",
            "benchmark_artifact_hash",
            "source_tree_hash",
            "source_archive_sha256",
            "solver_implementation_sha256",
            "qualified_policy_semantics_hash",
            "device_name",
            "device_capability",
            "torch_version",
            "cuda_version",
            "cublas_workspace_config",
            "small_domain_parity_hash",
        ):
            _identity(getattr(self, name), name=name.replace("_", " "))
        if self.route_count != SIGNED_WORK_RESOURCE_FRONTIER_CUDA_FULL_ROUTE_COUNT:
            raise ValueError("H200 qualification receipt has another route count")
        if self.ranks != SIGNED_WORK_RESOURCE_FRONTIER_CUDA_QUALIFIED_RANKS:
            raise ValueError("H200 qualification receipt has another rank ladder")
        if len(self.rank_case_hashes) != len(self.ranks) or any(
            type(value) is not str or not value for value in self.rank_case_hashes
        ):
            raise ValueError("H200 qualification receipt rank hashes are incomplete")
        fraction = float(self.maximum_measured_hbm_fraction)
        if not math.isfinite(fraction) or fraction < 0.0 or fraction > 0.85:
            raise ValueError("H200 qualification measured HBM exceeds 85 percent")
        object.__setattr__(self, "maximum_measured_hbm_fraction", fraction)
        for name in (
            "strict_deterministic_algorithms",
            "deterministic_replay",
            "zero_route_vector_d2h",
            "zero_signed_f_scan",
            "synthetic_engineering_only",
            "production_decision_authority",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact bool")
        if not all(
            (
                self.strict_deterministic_algorithms,
                self.deterministic_replay,
                self.zero_route_vector_d2h,
                self.zero_signed_f_scan,
                self.synthetic_engineering_only,
            )
        ):
            raise ValueError("H200 qualification receipt is not closed")
        if self.production_decision_authority:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.AUTHORITY_VIOLATION,
                "H200 engineering qualification cannot grant decision authority",
            )
        if "H200" not in self.device_name.upper() or self.device_capability != "9.0":
            raise ValueError("H200 qualification receipt has another device class")
        if self.cublas_workspace_config not in _QUALIFIED_CUBLAS_WORKSPACE_CONFIGS:
            raise ValueError("H200 qualification has no deterministic cuBLAS config")
        if len(self.source_archive_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.source_archive_sha256
        ):
            raise ValueError("source archive SHA-256 is malformed")
        if len(self.solver_implementation_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.solver_implementation_sha256
        ):
            raise ValueError("solver implementation SHA-256 is malformed")
        if (
            self.device_class
            not in SIGNED_WORK_RESOURCE_FRONTIER_CUDA_QUALIFIED_DEVICE_CLASSES
            or self.algorithm_id
            != SIGNED_WORK_RESOURCE_FRONTIER_CUDA_ALGORITHM_ID
            or self.policy_schema != SIGNED_WORK_RESOURCE_FRONTIER_CUDA_POLICY_SCHEMA
            or self.problem_schema != SIGNED_WORK_RESOURCE_FRONTIER_CUDA_PROBLEM_SCHEMA
            or self.execution_schema
            != SIGNED_WORK_RESOURCE_FRONTIER_CUDA_EXECUTION_SCHEMA
            or self.no_f_operator_id
            != SIGNED_WORK_RESOURCE_FRONTIER_CUDA_NO_F_OPERATOR_ID
            or self.numerical_dtype
            != SIGNED_WORK_RESOURCE_FRONTIER_CUDA_NUMERICAL_DTYPE
            or self.branches
            != SIGNED_WORK_RESOURCE_FRONTIER_CUDA_QUALIFIED_BRANCHES
            or self.schema
            != SIGNED_WORK_RESOURCE_FRONTIER_CUDA_H200_QUALIFICATION_SCHEMA
        ):
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.AUTHORITY_VIOLATION,
                "H200 qualification solver semantics changed",
            )

    def _payload(self) -> Mapping[str, object]:
        return {
            name: getattr(self, name)
            for name, declaration in self.__dataclass_fields__.items()
            if name != "_qualification_hash"
        }

    @property
    def qualification_hash(self) -> str:
        return self._qualification_hash

    def assert_integrity(self) -> None:
        expected = content_hash(self._payload(), prefix=_QUALIFICATION_PREFIX)
        if self._qualification_hash != expected:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                "H200 qualification receipt content hash changed",
            )


@dataclass(frozen=True, slots=True)
class V1SignedWorkResourceFrontierCUDAPolicy:
    policy_id: str = "modern-v1-signed-resource-frontier-cuda-default"
    torch_device: str = "cuda"
    hbm_ceiling_fraction: float = 0.85
    max_iterations: int = 4096
    minimum_iterations: int = 64
    check_interval: int = 16
    over_relaxation: float = 1.0
    primal_feasibility_tolerance: float = 1e-9
    absolute_gap_tolerance: float = 1e-9
    relative_gap_tolerance: float = 1e-6
    private_reference_route_limit: int = 32
    production_cuda_required: bool = True
    allow_private_cpu_reference: bool = False
    use_tensor_private_reference: bool = False
    require_deterministic_algorithms: bool = True
    no_signed_f_scan: bool = True
    route_scale_cuda_solver_qualified: bool = False
    route_scale_cuda_qualification: (
        V1SignedWorkResourceFrontierCUDAH200QualificationReceipt | None
    ) = None
    fqr_authority: bool = False
    mask_commit_authority: bool = False
    authority_scope: str = SIGNED_WORK_RESOURCE_FRONTIER_CUDA_AUTHORITY
    algorithm_id: str = SIGNED_WORK_RESOURCE_FRONTIER_CUDA_ALGORITHM_ID
    schema: str = SIGNED_WORK_RESOURCE_FRONTIER_CUDA_POLICY_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _identity(self.policy_id, name="policy id"))
        object.__setattr__(
            self,
            "torch_device",
            _identity(self.torch_device, name="torch device"),
        )
        ceiling = _positive(self.hbm_ceiling_fraction, name="HBM ceiling fraction")
        if ceiling > 0.85:
            raise ValueError("canonical resource-frontier HBM ceiling cannot exceed 0.85")
        object.__setattr__(self, "hbm_ceiling_fraction", ceiling)
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
        theta = _positive(self.over_relaxation, name="over relaxation")
        if theta > 1.0:
            raise ValueError("over_relaxation must be in (0, 1]")
        object.__setattr__(self, "over_relaxation", theta)
        for name in (
            "primal_feasibility_tolerance",
            "absolute_gap_tolerance",
            "relative_gap_tolerance",
        ):
            object.__setattr__(
                self,
                name,
                _positive(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "private_reference_route_limit",
            _nonnegative_int(
                self.private_reference_route_limit,
                name="private reference route limit",
            ),
        )
        for name in (
            "production_cuda_required",
            "allow_private_cpu_reference",
            "use_tensor_private_reference",
            "require_deterministic_algorithms",
            "no_signed_f_scan",
            "route_scale_cuda_solver_qualified",
            "fqr_authority",
            "mask_commit_authority",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact bool")
        qualification = self.route_scale_cuda_qualification
        if qualification is not None:
            if (
                type(qualification)
                is not V1SignedWorkResourceFrontierCUDAH200QualificationReceipt
            ):
                raise TypeError("route-scale CUDA qualification has another type")
            qualification.assert_integrity()
        if self.production_cuda_required and self.allow_private_cpu_reference:
            raise ValueError("production resource-frontier policy cannot allow CPU reference")
        if not self.no_signed_f_scan:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.AUTHORITY_VIOLATION,
                "Phase-A resource frontier may not scan signed F",
            )
        _closed_flags(
            (self.fqr_authority, self.mask_commit_authority),
            owner="resource-frontier CUDA policy",
        )
        if (
            self.authority_scope != SIGNED_WORK_RESOURCE_FRONTIER_CUDA_AUTHORITY
            or self.algorithm_id != SIGNED_WORK_RESOURCE_FRONTIER_CUDA_ALGORITHM_ID
            or self.schema != SIGNED_WORK_RESOURCE_FRONTIER_CUDA_POLICY_SCHEMA
        ):
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.AUTHORITY_VIOLATION,
                "resource-frontier CUDA policy semantics changed",
            )

    @property
    def policy_hash(self) -> str:
        return content_hash(self, prefix=_POLICY_PREFIX)


def _qualification_policy_semantics(
    policy: V1SignedWorkResourceFrontierCUDAPolicy,
) -> Mapping[str, object]:
    return {
        "algorithm_id": policy.algorithm_id,
        "policy_schema": policy.schema,
        "problem_schema": SIGNED_WORK_RESOURCE_FRONTIER_CUDA_PROBLEM_SCHEMA,
        "execution_schema": SIGNED_WORK_RESOURCE_FRONTIER_CUDA_EXECUTION_SCHEMA,
        "no_f_operator_id": SIGNED_WORK_RESOURCE_FRONTIER_CUDA_NO_F_OPERATOR_ID,
        "numerical_dtype": SIGNED_WORK_RESOURCE_FRONTIER_CUDA_NUMERICAL_DTYPE,
        "branches": SIGNED_WORK_RESOURCE_FRONTIER_CUDA_QUALIFIED_BRANCHES,
        "device_class": runtime_qualified_device_class(),
        "hbm_ceiling_fraction": policy.hbm_ceiling_fraction,
        "max_iterations": policy.max_iterations,
        "minimum_iterations": policy.minimum_iterations,
        "check_interval": policy.check_interval,
        "over_relaxation": policy.over_relaxation,
        "primal_feasibility_tolerance": policy.primal_feasibility_tolerance,
        "absolute_gap_tolerance": policy.absolute_gap_tolerance,
        "relative_gap_tolerance": policy.relative_gap_tolerance,
        "production_cuda_required": policy.production_cuda_required,
        "allow_private_cpu_reference": policy.allow_private_cpu_reference,
        "use_tensor_private_reference": policy.use_tensor_private_reference,
        "require_deterministic_algorithms": policy.require_deterministic_algorithms,
        "no_signed_f_scan": policy.no_signed_f_scan,
    }


def _qualification_policy_semantics_hash(
    policy: V1SignedWorkResourceFrontierCUDAPolicy,
) -> str:
    return content_hash(
        _qualification_policy_semantics(policy),
        prefix=_QUALIFICATION_POLICY_PREFIX,
    )


def v1_signed_work_resource_frontier_cuda_qualification_policy_semantics_hash(
    policy: V1SignedWorkResourceFrontierCUDAPolicy,
) -> str:
    """Expose the exact pre-outcome policy projection a benchmark must bind."""

    if type(policy) is not V1SignedWorkResourceFrontierCUDAPolicy:
        raise TypeError("qualification policy projection requires an exact policy")
    return _qualification_policy_semantics_hash(policy)


def issue_v1_signed_work_resource_frontier_cuda_h200_engineering_capability(
    *,
    policy: V1SignedWorkResourceFrontierCUDAPolicy,
    source_tree_hash: str,
    source_archive_sha256: str,
) -> V1SignedWorkResourceFrontierCUDAH200EngineeringCapability:
    """Open only the synthetic H200 benchmark needed to qualify this source."""

    if type(policy) is not V1SignedWorkResourceFrontierCUDAPolicy:
        raise TypeError("engineering capability requires an exact policy")
    if (
        not policy.production_cuda_required
        or policy.allow_private_cpu_reference
        or policy.use_tensor_private_reference
        or not policy.require_deterministic_algorithms
        or not policy.no_signed_f_scan
        or policy.route_scale_cuda_qualification is not None
        or policy.route_scale_cuda_solver_qualified
    ):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.AUTHORITY_VIOLATION,
            "engineering capability requires the closed unqualified CUDA policy",
        )
    return V1SignedWorkResourceFrontierCUDAH200EngineeringCapability(
        _issuer_token=_ENGINEERING_CAPABILITY_TOKEN,
        source_tree_hash=_identity(source_tree_hash, name="source tree hash"),
        source_archive_sha256=_identity(
            source_archive_sha256, name="source archive SHA-256"
        ),
        solver_implementation_sha256=(
            current_v1_signed_work_resource_frontier_cuda_solver_implementation_sha256()
        ),
        qualified_policy_semantics_hash=_qualification_policy_semantics_hash(policy),
        route_count=SIGNED_WORK_RESOURCE_FRONTIER_CUDA_FULL_ROUTE_COUNT,
        ranks=SIGNED_WORK_RESOURCE_FRONTIER_CUDA_QUALIFIED_RANKS,
        synthetic_target_fraction=1.0 / 32.0,
        device_class=runtime_qualified_device_class(),
        synthetic_engineering_only=True,
        production_decision_authority=False,
    )


def _assert_exact_h200_engineering_problem(
    problem: V1SignedWorkResourceFrontierCUDAProblem,
    *,
    capability: V1SignedWorkResourceFrontierCUDAH200EngineeringCapability,
    chunk_size: int = 1 << 20,
) -> None:
    """Check the complete registered synthetic geometry with bounded workspace."""

    if type(problem) is not V1SignedWorkResourceFrontierCUDAProblem:
        raise TypeError("engineering execution requires the operator-free problem")
    rank = problem.rank
    route_count = problem.route_count
    if route_count != capability.route_count or rank not in capability.ranks:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.AUTHORITY_VIOLATION,
            "engineering execution received another route/rank problem",
        )
    expected_id = f"synthetic-phase-A-routes-{route_count}-rank-{rank}"
    if (
        problem.problem_id != expected_id
        or problem.route_axis_hash != f"synthetic-route-axis-{route_count}"
        or problem.source_geometry_hash
        != f"synthetic-response-geometry-routes-{route_count}-rank-{rank}-v1"
        or problem.source_registration_hash
        != f"synthetic-D-registration-routes-{route_count}-rank-{rank}"
        or problem.registered_d_coordinate_hash
        != f"synthetic-D-coordinate-routes-{route_count}-rank-{rank}"
        or not problem.source_bundle_authoritative
        or not problem.no_signed_f_operator_retained
        or not problem.no_signed_f_scan
    ):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.AUTHORITY_VIOLATION,
            "engineering execution received non-synthetic source identity",
        )
    cuts = tuple(problem.block_capacities)
    start = 0
    if len(cuts) != len(_H200_ENGINEERING_ALEXNET_BLOCK_SIZES):
        raise ValueError("engineering problem has another compact block map")
    for index, (block, size) in enumerate(
        zip(cuts, _H200_ENGINEERING_ALEXNET_BLOCK_SIZES, strict=True)
    ):
        stop = start + size
        if (
            block.block_id != f"synthetic-alexnet-cut-{index}"
            or block.start != start
            or block.stop != stop
            or block.removal_capacity != size - 1
        ):
            raise ValueError("engineering problem block legality changed")
        start = stop
    inverse_n = 1.0 / float(route_count)
    if (
        problem.residual_spectral_bound != inverse_n * inverse_n
        or problem.quadratic_budget
        != 2.0 * capability.synthetic_target_fraction**2
    ):
        raise ValueError("engineering synthetic scalar geometry changed")
    for start in range(0, route_count, chunk_size):
        stop = min(route_count, start + chunk_size)
        if (
            not bool(np.all(problem.resource[start:stop] == 1.0))
            or not bool(np.all(problem.eligible_mask[start:stop]))
            or not bool(
                np.all(
                    problem.residual_diagonal[start:stop]
                    == inverse_n * inverse_n
                )
            )
            or not bool(np.all(problem.low_rank_factor[0, start:stop] == inverse_n))
        ):
            raise ValueError("engineering synthetic base coordinates changed")
    amplitude = inverse_n / (128.0 * math.sqrt(max(1, rank - 1)))
    for row in range(1, rank):
        half_period = row + 1
        period = 2 * half_period
        for start in range(0, route_count, chunk_size):
            stop = min(route_count, start + chunk_size)
            indices = np.arange(start, stop, dtype=np.int64)
            phase = np.remainder(indices + row, period)
            expected = np.where(phase < half_period, amplitude, -amplitude)
            if not np.array_equal(problem.low_rank_factor[row, start:stop], expected):
                raise ValueError("engineering synthetic factor rule changed")


def _evidence_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.INVALID_TYPE,
            f"H200 benchmark {name} must be a mapping",
        )
    return value


def _evidence_bool(value: object, *, name: str, expected: bool) -> None:
    if type(value) is not bool or value is not expected:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.ROUTE_SCALE_CUDA_PENDING,
            f"H200 benchmark {name} did not close",
        )


def _evidence_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.INVALID_TYPE,
            f"H200 benchmark {name} must be an integer",
        )
    return int(value)


def _evidence_fraction(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.INVALID_TYPE,
            f"H200 benchmark {name} must be a real scalar",
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 0.85:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.ROUTE_SCALE_CUDA_PENDING,
            f"H200 benchmark {name} is outside the closed HBM range",
        )
    return result


def issue_v1_signed_work_resource_frontier_cuda_h200_qualification(
    *,
    benchmark_artifact: Mapping[str, object],
    policy: V1SignedWorkResourceFrontierCUDAPolicy,
) -> V1SignedWorkResourceFrontierCUDAH200QualificationReceipt:
    """Validate one closed benchmark artifact and mint its narrow gate.

    The artifact must augment the benchmark JSON with ``source_identity`` and
    ``solver_contract`` projections.  Issuance performs no CUDA work and never
    treats the synthetic engineering benchmark as pruning efficacy evidence.
    """

    if type(policy) is not V1SignedWorkResourceFrontierCUDAPolicy:
        raise TypeError("H200 qualification issuance requires an exact policy")
    if policy.route_scale_cuda_qualification is not None:
        raise ValueError("H200 qualification issuance requires an unqualified policy")
    if (
        not policy.production_cuda_required
        or policy.allow_private_cpu_reference
        or policy.use_tensor_private_reference
        or not policy.require_deterministic_algorithms
        or not policy.no_signed_f_scan
    ):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.AUTHORITY_VIOLATION,
            "H200 qualification policy is not the canonical CUDA-only seam",
        )
    artifact = _evidence_mapping(benchmark_artifact, name="artifact")
    try:
        schema = artifact["schema"]
        status = artifact["status"]
        regression_flags = artifact["regression_flags"]
        authority = _evidence_mapping(artifact["authority"], name="authority")
        contract = _evidence_mapping(artifact["contract"], name="contract")
        environment = _evidence_mapping(
            artifact["environment"], name="environment"
        )
        parity = _evidence_mapping(
            artifact["small_domain_parity"], name="small-domain parity"
        )
        source = _evidence_mapping(
            artifact["source_identity"], name="source identity"
        )
        solver = _evidence_mapping(
            artifact["solver_contract"], name="solver contract"
        )
        engineering_capability_hash = artifact["engineering_capability_hash"]
        rank_cases_raw = artifact["rank_cases"]
    except KeyError as error:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.ROUTE_SCALE_CUDA_PENDING,
            f"H200 benchmark artifact is incomplete at {error.args[0]!r}",
        )
    if schema != SIGNED_WORK_RESOURCE_FRONTIER_CUDA_H200_BENCHMARK_SCHEMA:
        raise ValueError("unsupported H200 benchmark artifact schema")
    if status != "passed" or regression_flags not in ([], ()):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.ROUTE_SCALE_CUDA_PENDING,
            "H200 benchmark did not close without regression flags",
        )
    for name, expected in (
        ("synthetic_engineering_only", True),
        ("model_or_dataset_data", False),
        ("signed_f_operator_or_row_bank", False),
        ("runtime_policy_default_promoted", False),
        ("production_decision_authority", False),
        ("fqr_authority", False),
        ("mask_commit_authority", False),
        ("terminal_authority", False),
        ("pareto_authority", False),
    ):
        try:
            _evidence_bool(authority[name], name=f"authority.{name}", expected=expected)
        except KeyError:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.ROUTE_SCALE_CUDA_PENDING,
                f"H200 benchmark authority is incomplete at {name!r}",
            )
    try:
        route_count = _evidence_int(contract["route_count"], name="route count")
        ranks = tuple(
            _evidence_int(value, name="rank") for value in contract["ranks"]
        )
        _evidence_bool(
            contract["exact_alexnet_all_route_axis"],
            name="exact AlexNet route axis",
            expected=True,
        )
        contract_ceiling = _evidence_fraction(
            contract["hbm_ceiling_fraction"], name="contract HBM ceiling"
        )
    except (KeyError, TypeError):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.ROUTE_SCALE_CUDA_PENDING,
            "H200 benchmark route/rank contract is incomplete",
        )
    if (
        route_count != SIGNED_WORK_RESOURCE_FRONTIER_CUDA_FULL_ROUTE_COUNT
        or ranks != SIGNED_WORK_RESOURCE_FRONTIER_CUDA_QUALIFIED_RANKS
        or contract_ceiling != policy.hbm_ceiling_fraction
    ):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.ROUTE_SCALE_CUDA_PENDING,
            "H200 benchmark route/rank/HBM contract differs from production",
        )
    for name in ("max_iterations", "minimum_iterations", "check_interval"):
        if _evidence_int(contract.get(name), name=name) != getattr(policy, name):
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.ROUTE_SCALE_CUDA_PENDING,
                f"H200 benchmark {name} differs from the policy",
            )
    qualified_policy_hash = _qualification_policy_semantics_hash(policy)
    expected_solver_contract = {
        "policy_semantics_hash": qualified_policy_hash,
        "algorithm_id": SIGNED_WORK_RESOURCE_FRONTIER_CUDA_ALGORITHM_ID,
        "policy_schema": SIGNED_WORK_RESOURCE_FRONTIER_CUDA_POLICY_SCHEMA,
        "problem_schema": SIGNED_WORK_RESOURCE_FRONTIER_CUDA_PROBLEM_SCHEMA,
        "execution_schema": SIGNED_WORK_RESOURCE_FRONTIER_CUDA_EXECUTION_SCHEMA,
        "no_f_operator_id": SIGNED_WORK_RESOURCE_FRONTIER_CUDA_NO_F_OPERATOR_ID,
        "numerical_dtype": SIGNED_WORK_RESOURCE_FRONTIER_CUDA_NUMERICAL_DTYPE,
        "branches": list(SIGNED_WORK_RESOURCE_FRONTIER_CUDA_QUALIFIED_BRANCHES),
        "device_class": runtime_qualified_device_class(),
    }
    if dict(solver) != expected_solver_contract:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.ROUTE_SCALE_CUDA_PENDING,
            "H200 benchmark solver contract differs from the runtime",
        )
    try:
        source_tree_hash = _identity(
            source["source_tree_hash"], name="source tree hash"
        )
        source_archive_sha256 = _identity(
            source["source_archive_sha256"], name="source archive SHA-256"
        )
        solver_implementation_sha256 = _identity(
            source["solver_implementation_sha256"],
            name="solver implementation SHA-256",
        )
    except KeyError:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.ROUTE_SCALE_CUDA_PENDING,
            "H200 benchmark source identity is incomplete",
        )
    if (
        solver_implementation_sha256
        != current_v1_signed_work_resource_frontier_cuda_solver_implementation_sha256()
    ):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
            "H200 benchmark was produced by another solver implementation",
        )
    expected_engineering_capability = (
        issue_v1_signed_work_resource_frontier_cuda_h200_engineering_capability(
            policy=policy,
            source_tree_hash=source_tree_hash,
            source_archive_sha256=source_archive_sha256,
        )
    )
    if (
        engineering_capability_hash
        != expected_engineering_capability.capability_hash
    ):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
            "H200 benchmark capability does not bind its source/policy",
        )
    try:
        device_name = _identity(environment["device_name"], name="device name")
        device_count = _evidence_int(environment["device_count"], name="device count")
        device_capability = _identity(
            environment["device_capability"], name="device capability"
        )
        torch_version = _identity(environment["torch"], name="torch version")
        cuda_version = _identity(environment["torch_cuda"], name="CUDA version")
        cublas_workspace_config = _identity(
            environment["cublas_workspace_config"],
            name="cuBLAS workspace config",
        )
    except KeyError:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.ROUTE_SCALE_CUDA_PENDING,
            "H200 benchmark environment identity is incomplete",
        )
    if (
        "H200" not in device_name.upper()
        or device_count != 1
        or device_capability != "9.0"
        or cublas_workspace_config not in _QUALIFIED_CUBLAS_WORKSPACE_CONFIGS
    ):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.ROUTE_SCALE_CUDA_PENDING,
            "H200 benchmark has another device/determinism class",
        )
    if parity.get("status") != "passed" or parity.get("mismatches") not in ([], ()):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.ROUTE_SCALE_CUDA_PENDING,
            "H200 benchmark small-domain FP64 parity did not close",
        )
    if not isinstance(rank_cases_raw, (tuple, list)):
        raise TypeError("H200 benchmark rank cases must be a sequence")
    rank_cases = tuple(
        _evidence_mapping(value, name="rank case") for value in rank_cases_raw
    )
    if tuple(case.get("rank") for case in rank_cases) != ranks:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.ROUTE_SCALE_CUDA_PENDING,
            "H200 benchmark rank cases are incomplete or reordered",
        )
    rank_case_hashes: list[str] = []
    maximum_measured_hbm_fraction = 0.0
    for rank, case in zip(ranks, rank_cases, strict=True):
        if case.get("status") != "passed":
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.ROUTE_SCALE_CUDA_PENDING,
                f"H200 rank-{rank} benchmark did not pass",
            )
        for name in ("deterministic_replay",):
            _evidence_bool(case.get(name), name=f"rank-{rank} {name}", expected=True)
        for name in (
            "route_vector_d2h_bytes",
            "route_vector_d2h_transfer_count",
            "python_tuple_witnesses",
        ):
            if _evidence_int(case.get(name), name=f"rank-{rank} {name}") != 0:
                _refuse(
                    V1SignedWorkResourceFrontierCUDARefusalCode.ROUTE_SCALE_CUDA_PENDING,
                    f"H200 rank-{rank} materialized a route-scale witness",
                )
        _evidence_bool(
            case.get("signed_f_operator_or_row_bank"),
            name=f"rank-{rank} signed-F scan",
            expected=False,
        )
        runs_raw = case.get("runs")
        if not isinstance(runs_raw, (tuple, list)) or len(runs_raw) < 2:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.ROUTE_SCALE_CUDA_PENDING,
                f"H200 rank-{rank} lacks cold/warm replay",
            )
        frontier_hashes: list[str] = []
        witness_hashes: list[tuple[str, ...]] = []
        for run_raw in runs_raw:
            run = _evidence_mapping(run_raw, name=f"rank-{rank} run")
            measured = _evidence_fraction(
                run.get("measured_peak_fraction_of_total"),
                name=f"rank-{rank} measured HBM",
            )
            maximum_measured_hbm_fraction = max(
                maximum_measured_hbm_fraction, measured
            )
            receipt = _evidence_mapping(
                run.get("receipt"), name=f"rank-{rank} CUDA receipt"
            )
            if (
                receipt.get("union_status") != "certified_coordinates"
                or receipt.get("production_cuda_execution") is not False
                or receipt.get("private_cpu_reference_execution") is not False
                or receipt.get("engineering_h200_qualification_execution")
                is not True
                or receipt.get("engineering_capability_hash")
                != engineering_capability_hash
                or receipt.get("no_signed_f_scan") is not True
                or receipt.get("all_branch_alternatives_closed") is not True
            ):
                _refuse(
                    V1SignedWorkResourceFrontierCUDARefusalCode.ROUTE_SCALE_CUDA_PENDING,
                    f"H200 rank-{rank} CUDA receipt did not close",
                )
            forecast = _evidence_mapping(
                receipt.get("forecast"), name=f"rank-{rank} forecast"
            )
            if (
                _evidence_int(forecast.get("route_count"), name="forecast routes")
                != route_count
                or _evidence_int(forecast.get("response_rank"), name="forecast rank")
                != rank
                or forecast.get("admitted") is not True
                or _evidence_int(
                    forecast.get("signed_f_operator_scan_bytes"),
                    name="forecast signed-F bytes",
                )
                != 0
                or forecast.get("route_scale_cpu_order_materialized") is not False
            ):
                _refuse(
                    V1SignedWorkResourceFrontierCUDARefusalCode.ROUTE_SCALE_CUDA_PENDING,
                    f"H200 rank-{rank} forecast crossed the bounded CUDA seam",
                )
            branches_raw = receipt.get("branches")
            if not isinstance(branches_raw, (tuple, list)):
                raise TypeError("H200 benchmark branches must be a sequence")
            branches = tuple(
                _evidence_mapping(value, name="branch") for value in branches_raw
            )
            if tuple(branch.get("branch") for branch in branches) != (
                SIGNED_WORK_RESOURCE_FRONTIER_CUDA_QUALIFIED_BRANCHES
            ):
                _refuse(
                    V1SignedWorkResourceFrontierCUDARefusalCode.ROUTE_SCALE_CUDA_PENDING,
                    f"H200 rank-{rank} did not execute both registered branches",
                )
            for branch in branches:
                if (
                    branch.get("status") != "optimal"
                    or branch.get("primal_feasible") is not True
                    or branch.get("dual_enclosure_closed") is not True
                    or branch.get("ephemeral_cuda_witness_certified") is not True
                    or branch.get("route_vector_retained") is not False
                    or branch.get("python_tuple_witness_present") is not False
                    or _evidence_int(
                        branch.get("route_vector_d2h_bytes"), name="route D2H"
                    )
                    != 0
                    or _evidence_int(
                        branch.get("route_vector_d2h_transfer_count"),
                        name="route D2H transfers",
                    )
                    != 0
                ):
                    _refuse(
                        V1SignedWorkResourceFrontierCUDARefusalCode.ROUTE_SCALE_CUDA_PENDING,
                        f"H200 rank-{rank} branch certificate did not close",
                    )
            frontier_hashes.append(
                _identity(receipt.get("frontier_result_hash"), name="frontier hash")
            )
            witness_hashes.append(
                tuple(
                    _identity(
                        branch.get("witness_summary_hash"), name="witness hash"
                    )
                    for branch in branches
                )
            )
        if len(set(frontier_hashes)) != 1 or len(set(witness_hashes)) != 1:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.ROUTE_SCALE_CUDA_PENDING,
                f"H200 rank-{rank} cold/warm replay changed",
            )
        rank_case_hashes.append(
            content_hash(case, prefix=f"phase-A-H200-rank-{rank}-measurement-")
        )
    try:
        benchmark_artifact_hash = content_hash(
            artifact, prefix=_QUALIFICATION_ARTIFACT_PREFIX
        )
    except (TypeError, ValueError) as error:
        raise ValueError("H200 benchmark artifact is not content-addressable") from error
    parity_hash = content_hash(parity, prefix="phase-A-H200-small-domain-parity-")
    return V1SignedWorkResourceFrontierCUDAH200QualificationReceipt(
        _issuer_token=_QUALIFICATION_TOKEN,
        qualification_id=f"phase-A-H200-{benchmark_artifact_hash[-16:]}",
        benchmark_artifact_hash=benchmark_artifact_hash,
        source_tree_hash=source_tree_hash,
        source_archive_sha256=source_archive_sha256,
        solver_implementation_sha256=solver_implementation_sha256,
        qualified_policy_semantics_hash=qualified_policy_hash,
        device_class=runtime_qualified_device_class(),
        device_name=device_name,
        device_capability=device_capability,
        torch_version=torch_version,
        cuda_version=cuda_version,
        cublas_workspace_config=cublas_workspace_config,
        route_count=route_count,
        ranks=ranks,
        rank_case_hashes=tuple(rank_case_hashes),
        maximum_measured_hbm_fraction=maximum_measured_hbm_fraction,
        small_domain_parity_hash=parity_hash,
        strict_deterministic_algorithms=True,
        deterministic_replay=True,
        zero_route_vector_d2h=True,
        zero_signed_f_scan=True,
        synthetic_engineering_only=True,
        production_decision_authority=False,
        algorithm_id=SIGNED_WORK_RESOURCE_FRONTIER_CUDA_ALGORITHM_ID,
        policy_schema=SIGNED_WORK_RESOURCE_FRONTIER_CUDA_POLICY_SCHEMA,
        problem_schema=SIGNED_WORK_RESOURCE_FRONTIER_CUDA_PROBLEM_SCHEMA,
        execution_schema=SIGNED_WORK_RESOURCE_FRONTIER_CUDA_EXECUTION_SCHEMA,
        no_f_operator_id=SIGNED_WORK_RESOURCE_FRONTIER_CUDA_NO_F_OPERATOR_ID,
        numerical_dtype=SIGNED_WORK_RESOURCE_FRONTIER_CUDA_NUMERICAL_DTYPE,
        branches=SIGNED_WORK_RESOURCE_FRONTIER_CUDA_QUALIFIED_BRANCHES,
    )


def validate_v1_signed_work_resource_frontier_cuda_h200_qualification(
    *,
    route_count: int,
    response_rank: int,
    policy: V1SignedWorkResourceFrontierCUDAPolicy,
    device_name: str,
    device_capability: str,
    torch_version: str,
    cuda_version: str,
    cublas_workspace_config: str | None,
) -> V1SignedWorkResourceFrontierCUDAH200QualificationReceipt:
    """Fail closed unless one exact issued receipt authorizes this solve."""

    receipt = policy.route_scale_cuda_qualification
    if (
        type(receipt)
        is not V1SignedWorkResourceFrontierCUDAH200QualificationReceipt
    ):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.ROUTE_SCALE_CUDA_PENDING,
            "legacy route-scale qualification booleans have no production authority",
        )
    receipt.assert_integrity()
    if (
        receipt.solver_implementation_sha256
        != current_v1_signed_work_resource_frontier_cuda_solver_implementation_sha256()
        or receipt.qualified_policy_semantics_hash
        != _qualification_policy_semantics_hash(policy)
    ):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
            "H200 qualification does not bind this solver/policy",
        )
    if (
        _positive_int(route_count, name="qualification route count")
        != receipt.route_count
        or _positive_int(response_rank, name="qualification response rank")
        not in receipt.ranks
    ):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.ROUTE_SCALE_CUDA_PENDING,
            "H200 qualification does not cover this route/rank problem",
        )
    if (
        "H200" not in _identity(device_name, name="device name").upper()
        or _identity(device_capability, name="device capability")
        != receipt.device_capability
        or _identity(torch_version, name="torch version") != receipt.torch_version
        or _identity(cuda_version, name="CUDA version") != receipt.cuda_version
        or cublas_workspace_config != receipt.cublas_workspace_config
    ):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.ROUTE_SCALE_CUDA_PENDING,
            "H200 qualification does not cover the live device/determinism class",
        )
    return receipt


@dataclass(frozen=True, slots=True)
class V1SignedWorkResourceFrontierCUDAHBMForecast:
    problem_hash: str
    policy_hash: str
    device: str
    route_count: int
    response_rank: int
    block_count: int
    persistent_source_bytes: int
    solver_state_bytes: int
    retained_theorem_state_bytes: int
    transient_workspace_bytes: int
    forecast_incremental_bytes: int
    current_reserved_bytes: int
    forecast_peak_reserved_bytes: int
    device_total_memory_bytes: int
    hbm_ceiling_bytes: int
    admitted: bool
    signed_f_operator_scan_bytes: int = 0
    route_scale_cpu_order_materialized: bool = False
    schema: str = SIGNED_WORK_RESOURCE_FRONTIER_CUDA_FORECAST_SCHEMA

    def __post_init__(self) -> None:
        for name in ("problem_hash", "policy_hash", "device"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "route_count",
            "device_total_memory_bytes",
            "hbm_ceiling_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name),
            )
        for name in (
            "response_rank",
            "block_count",
            "persistent_source_bytes",
            "solver_state_bytes",
            "retained_theorem_state_bytes",
            "transient_workspace_bytes",
            "forecast_incremental_bytes",
            "current_reserved_bytes",
            "forecast_peak_reserved_bytes",
            "signed_f_operator_scan_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name),
            )
        if type(self.admitted) is not bool:
            raise TypeError("admitted must be an exact bool")
        _closed_flags(
            (self.route_scale_cpu_order_materialized,),
            owner="resource-frontier CUDA forecast",
        )
        if self.signed_f_operator_scan_bytes != 0:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.AUTHORITY_VIOLATION,
                "Phase-A forecast contains a signed-F scan",
            )
        if self.retained_theorem_state_bytes > self.solver_state_bytes:
            raise ValueError(
                "retained Phase-A theorem state exceeds forecast solver state"
            )
        if self.schema != SIGNED_WORK_RESOURCE_FRONTIER_CUDA_FORECAST_SCHEMA:
            raise ValueError("unsupported resource-frontier CUDA forecast schema")

    @property
    def forecast_hash(self) -> str:
        return content_hash(self, prefix=_FORECAST_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkResourceFrontierCUDADeviceHBMForecastV1:
    """Incremental HBM forecast for the live zero-copy Phase-A path."""

    problem_hash: str
    policy_hash: str
    device: str
    route_count: int
    response_rank: int
    block_count: int
    already_resident_aliased_source_bytes: int
    sqrt_residual_bytes: int
    compact_block_coordinate_bytes: int
    derived_bound_bytes: int
    solver_state_bytes: int
    retained_theorem_state_bytes: int
    transient_workspace_bytes: int
    scalar_reduction_workspace_bytes: int
    forecast_incremental_bytes: int
    current_reserved_bytes: int
    forecast_peak_reserved_bytes: int
    device_total_memory_bytes: int
    hbm_ceiling_bytes: int
    admitted: bool
    aliased_source_counted_as_new_allocation: bool = False
    route_scale_d2h_bytes: int = 0
    route_scale_numpy_arrays_constructed: int = 0
    summary_route_gather_bytes: int = 0
    schema: str = (
        "partition_free_v1_signed_work_resource_frontier_cuda_device_hbm_forecast_v1"
    )

    def __post_init__(self) -> None:
        for name in ("problem_hash", "policy_hash", "device"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in ("route_count", "device_total_memory_bytes", "hbm_ceiling_bytes"):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name),
            )
        for name in (
            "response_rank",
            "block_count",
            "already_resident_aliased_source_bytes",
            "sqrt_residual_bytes",
            "compact_block_coordinate_bytes",
            "derived_bound_bytes",
            "solver_state_bytes",
            "retained_theorem_state_bytes",
            "transient_workspace_bytes",
            "scalar_reduction_workspace_bytes",
            "forecast_incremental_bytes",
            "current_reserved_bytes",
            "forecast_peak_reserved_bytes",
            "route_scale_d2h_bytes",
            "route_scale_numpy_arrays_constructed",
            "summary_route_gather_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name),
            )
        if type(self.admitted) is not bool:
            raise TypeError("device Phase-A forecast admitted flag must be exact")
        if self.aliased_source_counted_as_new_allocation:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.AUTHORITY_VIOLATION,
                "zero-copy Phase-A forecast double-counted borrowed source storage",
            )
        if (
            self.route_scale_d2h_bytes
            or self.route_scale_numpy_arrays_constructed
            or self.summary_route_gather_bytes
        ):
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.AUTHORITY_VIOLATION,
                "zero-copy Phase-A forecast crossed the bounded device seam",
            )
        expected = (
            self.sqrt_residual_bytes
            + self.compact_block_coordinate_bytes
            + self.derived_bound_bytes
            + self.solver_state_bytes
            + self.transient_workspace_bytes
            + self.scalar_reduction_workspace_bytes
        )
        if expected != self.forecast_incremental_bytes:
            raise ValueError("device Phase-A incremental HBM accounting is incomplete")

    @property
    def forecast_hash(self) -> str:
        return content_hash(self, prefix=_FORECAST_PREFIX + "device-v1-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkResourceFrontierCUDAResult:
    problem_hash: str
    policy_hash: str
    forecast_hash: str
    reference_result: V1SignedWorkResourceFrontierResult
    endpoint_plan: V1SignedWorkDeviceShellEndpointPlanReceipt | None
    coordinate_hashes_bound_to_endpoint_plan: tuple[str, ...]
    production_cuda_execution: bool
    private_cpu_reference_execution: bool
    engineering_h200_qualification_execution: bool = False
    engineering_capability_hash: str | None = None
    no_signed_f_scan: bool = True
    c_stream_authority: bool = False
    fqr_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_RESOURCE_FRONTIER_CUDA_AUTHORITY
    algorithm_id: str = SIGNED_WORK_RESOURCE_FRONTIER_CUDA_ALGORITHM_ID
    endpoint_rule_id: str = SIGNED_WORK_RESOURCE_FRONTIER_CUDA_ENDPOINT_RULE_ID
    schema: str = SIGNED_WORK_RESOURCE_FRONTIER_CUDA_RESULT_SCHEMA

    def __post_init__(self) -> None:
        for name in ("problem_hash", "policy_hash", "forecast_hash"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if type(self.reference_result) is not V1SignedWorkResourceFrontierResult:
            raise TypeError("resource-frontier CUDA result requires exact reference result")
        if self.endpoint_plan is not None and type(self.endpoint_plan) is not (
            V1SignedWorkDeviceShellEndpointPlanReceipt
        ):
            raise TypeError("endpoint plan must be exact or absent")
        hashes = tuple(
            _identity(item, name="resource coordinate hash")
            for item in self.coordinate_hashes_bound_to_endpoint_plan
        )
        object.__setattr__(self, "coordinate_hashes_bound_to_endpoint_plan", hashes)
        if self.endpoint_plan is None and hashes:
            raise ValueError("coordinate hashes require an endpoint plan")
        if self.endpoint_plan is not None and not hashes:
            raise ValueError("endpoint plan requires bound resource coordinates")
        execution_modes = (
            self.production_cuda_execution,
            self.private_cpu_reference_execution,
            self.engineering_h200_qualification_execution,
        )
        if any(type(value) is not bool for value in execution_modes) or sum(
            execution_modes
        ) != 1:
            raise ValueError("exactly one resource-frontier execution mode must be active")
        if self.engineering_h200_qualification_execution:
            _identity(
                self.engineering_capability_hash,
                name="engineering capability hash",
            )
        elif self.engineering_capability_hash is not None:
            raise ValueError("non-engineering execution cannot bind a capability")
        if not self.no_signed_f_scan:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.AUTHORITY_VIOLATION,
                "resource-frontier result performed a signed-F scan",
            )
        _closed_flags(
            (
                self.c_stream_authority,
                self.fqr_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="resource-frontier CUDA result",
        )
        if (
            self.authority_scope != SIGNED_WORK_RESOURCE_FRONTIER_CUDA_AUTHORITY
            or self.algorithm_id != SIGNED_WORK_RESOURCE_FRONTIER_CUDA_ALGORITHM_ID
            or self.endpoint_rule_id != SIGNED_WORK_RESOURCE_FRONTIER_CUDA_ENDPOINT_RULE_ID
            or self.schema != SIGNED_WORK_RESOURCE_FRONTIER_CUDA_RESULT_SCHEMA
        ):
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.AUTHORITY_VIOLATION,
                "resource-frontier CUDA result semantics changed",
            )

    @property
    def union_status(self) -> V1SignedWorkResourceUnionStatus:
        return self.reference_result.union_status

    @property
    def spectral_status(self) -> V1SignedWorkResourceBranchStatus:
        return self.reference_result.spectral.status

    @property
    def diagonal_l1_status(self) -> V1SignedWorkResourceBranchStatus:
        return self.reference_result.diagonal_l1.status

    @property
    def result_hash(self) -> str:
        return content_hash(self, prefix=_RESULT_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkResourceCoordinateLeastWorkBinding:
    """Bind one endogenous resource coordinate to one fixed-resource solve."""

    least_work_problem_hash: str
    source_registration_hash: str
    registered_d_coordinate_hash: str
    resource_coordinate_hash: str
    least_work_problem: V1SignedWorkLeastWorkCUDAProblem
    endpoint_plan: V1SignedWorkDeviceShellEndpointPlanReceipt | None
    coordinate_total_resource_floor: float
    coordinate_incremental_resource: float
    coordinate_rule_id: str = SIGNED_WORK_RESOURCE_FRONTIER_COORDINATE_RULE_ID
    endpoint_rule_id: str = SIGNED_WORK_RESOURCE_FRONTIER_CUDA_ENDPOINT_RULE_ID
    c_stream_authority: bool = False
    fqr_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_RESOURCE_FRONTIER_CUDA_AUTHORITY
    schema: str = (
        "partition_free_v1_signed_work_coordinate_least_work_binding_v1"
    )

    def __post_init__(self) -> None:
        for name in (
            "least_work_problem_hash",
            "source_registration_hash",
            "registered_d_coordinate_hash",
            "resource_coordinate_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if type(self.least_work_problem) is not V1SignedWorkLeastWorkCUDAProblem:
            raise TypeError("coordinate binding requires exact least-work problem")
        if self.endpoint_plan is not None and type(self.endpoint_plan) is not (
            V1SignedWorkDeviceShellEndpointPlanReceipt
        ):
            raise TypeError("coordinate binding endpoint plan must be exact or absent")
        for name in ("coordinate_total_resource_floor", "coordinate_incremental_resource"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                _refuse(
                    V1SignedWorkResourceFrontierCUDARefusalCode.NONFINITE,
                    f"{name} must be positive finite",
                )
            object.__setattr__(self, name, value)
        if (
            self.least_work_problem.problem_hash != self.least_work_problem_hash
            or self.least_work_problem.source_registration_hash
            != self.source_registration_hash
            or self.least_work_problem.registered_d_coordinate_hash
            != self.registered_d_coordinate_hash
            or self.least_work_problem.resource_floor
            != self.coordinate_total_resource_floor
        ):
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                "coordinate-specific least-work problem left its source identity",
            )
        if self.coordinate_rule_id != SIGNED_WORK_RESOURCE_FRONTIER_COORDINATE_RULE_ID:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                "coordinate binding uses a noncanonical resource rule",
            )
        _closed_flags(
            (
                self.c_stream_authority,
                self.fqr_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="coordinate-specific least-work binding",
        )
        if (
            self.endpoint_rule_id != SIGNED_WORK_RESOURCE_FRONTIER_CUDA_ENDPOINT_RULE_ID
            or self.authority_scope != SIGNED_WORK_RESOURCE_FRONTIER_CUDA_AUTHORITY
        ):
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.AUTHORITY_VIOLATION,
                "coordinate binding semantics changed",
            )

    @property
    def binding_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "least_work_problem_hash": self.least_work_problem_hash,
                "source_registration_hash": self.source_registration_hash,
                "registered_d_coordinate_hash": self.registered_d_coordinate_hash,
                "resource_coordinate_hash": self.resource_coordinate_hash,
                "endpoint_plan_hash": (
                    None
                    if self.endpoint_plan is None
                    else self.endpoint_plan.receipt_hash
                ),
                "coordinate_total_resource_floor": self.coordinate_total_resource_floor,
                "coordinate_incremental_resource": self.coordinate_incremental_resource,
                "coordinate_rule_id": self.coordinate_rule_id,
                "endpoint_rule_id": self.endpoint_rule_id,
                "authority_scope": self.authority_scope,
            },
            prefix=_EXECUTION_PREFIX + "coordinate-binding-",
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkResourceCoordinateLeastWorkBindingFamily:
    """A one-coordinate/one-fixed-resource binding family for Phase B.

    The family intentionally stores independent fixed-resource problems rather
    than a shared endpoint schedule.  This makes the seam categorical: each
    certified resource coordinate must receive its own least-work optimization
    before any hard endpoint can move toward C-stream certification.
    """

    frontier_result_hash: str
    base_least_work_problem_hash: str
    source_registration_hash: str
    registered_d_coordinate_hash: str
    coordinate_bindings: tuple[V1SignedWorkResourceCoordinateLeastWorkBinding, ...]
    all_bindings_coordinate_specific: bool = True
    common_endpoint_plan_authority: bool = False
    c_stream_authority: bool = False
    fqr_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_RESOURCE_FRONTIER_CUDA_AUTHORITY
    schema: str = (
        "partition_free_v1_signed_work_coordinate_least_work_binding_family_v1"
    )

    def __post_init__(self) -> None:
        for name in (
            "frontier_result_hash",
            "base_least_work_problem_hash",
            "source_registration_hash",
            "registered_d_coordinate_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        bindings = tuple(self.coordinate_bindings)
        if not bindings:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.NO_INTEGER_ENDPOINT,
                "coordinate binding family requires at least one certified coordinate",
            )
        for binding in bindings:
            if type(binding) is not V1SignedWorkResourceCoordinateLeastWorkBinding:
                raise TypeError("coordinate binding family requires exact bindings")
            if (
                binding.source_registration_hash != self.source_registration_hash
                or binding.registered_d_coordinate_hash
                != self.registered_d_coordinate_hash
            ):
                _refuse(
                    V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                    "coordinate binding family mixes D-source identities",
                )
        floors = tuple(binding.coordinate_total_resource_floor for binding in bindings)
        for previous, current in zip(floors, floors[1:]):
            guard = 1024.0 * _EPS * max(1.0, abs(previous), abs(current))
            if current <= previous + guard:
                _refuse(
                    V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                    "coordinate binding family is not strictly increasing in resource",
                )
        if type(self.all_bindings_coordinate_specific) is not bool:
            raise TypeError("all_bindings_coordinate_specific must be an exact bool")
        if not self.all_bindings_coordinate_specific:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.AUTHORITY_VIOLATION,
                "coordinate binding family cannot contain shared-resource solves",
            )
        _closed_flags(
            (
                self.common_endpoint_plan_authority,
                self.c_stream_authority,
                self.fqr_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="coordinate-specific least-work binding family",
        )
        if (
            self.authority_scope != SIGNED_WORK_RESOURCE_FRONTIER_CUDA_AUTHORITY
            or self.schema
            != "partition_free_v1_signed_work_coordinate_least_work_binding_family_v1"
        ):
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.AUTHORITY_VIOLATION,
                "coordinate binding family semantics changed",
            )
        object.__setattr__(self, "coordinate_bindings", bindings)

    @property
    def binding_hashes(self) -> tuple[str, ...]:
        return tuple(binding.binding_hash for binding in self.coordinate_bindings)

    @property
    def endpoint_plan_hashes(self) -> tuple[str, ...]:
        return tuple(
            binding.endpoint_plan.receipt_hash
            for binding in self.coordinate_bindings
            if binding.endpoint_plan is not None
        )

    @property
    def family_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "frontier_result_hash": self.frontier_result_hash,
                "base_least_work_problem_hash": self.base_least_work_problem_hash,
                "source_registration_hash": self.source_registration_hash,
                "registered_d_coordinate_hash": self.registered_d_coordinate_hash,
                "binding_hashes": self.binding_hashes,
                "endpoint_plan_hashes": self.endpoint_plan_hashes,
                "all_bindings_coordinate_specific": (
                    self.all_bindings_coordinate_specific
                ),
                "common_endpoint_plan_authority": self.common_endpoint_plan_authority,
                "authority_scope": self.authority_scope,
            },
            prefix=_EXECUTION_PREFIX + "coordinate-family-",
        )


def _frontier_tensor_version_fingerprint(
    tensors: _FrontierTorchTensors,
    branch_states: tuple[_FrontierTensorBranchTheoremState, ...],
) -> tuple[tuple[str, int, int, tuple[int, ...]], ...]:
    """Capture tensor object identity and in-place mutation versions.

    The fingerprint is intentionally metadata-only: it does not read tensor
    contents, move route-scale vectors to host, or enter any serialized receipt.
    PyTorch increments ``_version`` after in-place mutation, which makes this a
    cheap guard against stale witness reuse across the next CUDA seam.
    """

    entries: list[tuple[str, int, int, tuple[int, ...]]] = []

    def _add(name: str, value: object | None) -> None:
        if value is None:
            return
        version = getattr(value, "_version", None)
        shape = getattr(value, "shape", ())
        if version is None:
            raise TypeError(f"{name} is not a versioned tensor object")
        entries.append(
            (
                name,
                id(value),
                int(version),
                tuple(int(item) for item in tuple(shape)),
            )
        )

    for name in (
        "resource",
        "factor",
        "sqrt_residual",
        "lower",
        "upper",
        "eligible_source",
        "required_source",
        "block_capacities",
        "block_scales",
    ):
        _add(f"source.{name}", getattr(tensors, name))
    for state in branch_states:
        prefix = f"branch.{state.branch.value}"
        _add(f"{prefix}.primal", state.primal)
        _add(f"{prefix}.low_q_dual", state.low_q_dual)
        _add(f"{prefix}.tail_q_dual", state.tail_q_dual)
        _add(f"{prefix}.block_dual", state.block_dual)
    return tuple(entries)


@dataclass(frozen=True, slots=True)
class V1SignedWorkResourceFrontierCUDALiveBranchResourceReduction:
    """One bounded scalar replay of a retained Phase-A primal witness.

    The route vector never leaves its source device.  Only the physical-resource
    dot product is reduced to one binary64 scalar, then outward-enclosed with
    the same reduction model used by the Phase-A solver.
    """

    branch: V1SignedWorkLeastWorkBranch
    branch_receipt_hash: str
    physical_resource_center: float
    physical_resource_lower_bound: float
    physical_resource_upper_bound: float
    scalar_host_transfer_bytes: int
    route_scale_d2h_bytes: int = 0
    route_scale_host_tensor_materializations: int = 0
    schema: str = (
        "partition_free_v1_signed_work_phase_A_live_branch_resource_reduction_v1"
    )

    def __post_init__(self) -> None:
        if type(self.branch) is not V1SignedWorkLeastWorkBranch:
            raise TypeError("live Phase-A reduction requires one exact branch")
        object.__setattr__(
            self,
            "branch_receipt_hash",
            _identity(self.branch_receipt_hash, name="branch receipt hash"),
        )
        center = _nonnegative(
            self.physical_resource_center,
            name="live physical-resource center",
        )
        lower = _nonnegative(
            self.physical_resource_lower_bound,
            name="live physical-resource lower bound",
        )
        upper = _nonnegative(
            self.physical_resource_upper_bound,
            name="live physical-resource upper bound",
        )
        if lower > center or center > upper:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                "live Phase-A resource reduction has a reversed enclosure",
            )
        object.__setattr__(self, "physical_resource_center", center)
        object.__setattr__(self, "physical_resource_lower_bound", lower)
        object.__setattr__(self, "physical_resource_upper_bound", upper)
        object.__setattr__(
            self,
            "scalar_host_transfer_bytes",
            _nonnegative_int(
                self.scalar_host_transfer_bytes,
                name="bounded scalar host-transfer bytes",
            ),
        )
        route_d2h = _nonnegative_int(
            self.route_scale_d2h_bytes,
            name="route-scale D2H bytes",
        )
        route_host = _nonnegative_int(
            self.route_scale_host_tensor_materializations,
            name="route-scale host tensor materializations",
        )
        if route_d2h != 0 or route_host != 0:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.AUTHORITY_VIOLATION,
                "live Phase-A scalar replay materialized a route-scale host tensor",
            )

    @property
    def numerical_radius_upper_bound(self) -> float:
        return math.nextafter(
            max(
                self.physical_resource_center
                - self.physical_resource_lower_bound,
                self.physical_resource_upper_bound
                - self.physical_resource_center,
            ),
            math.inf,
        )

    def __getstate__(self) -> NoReturn:
        raise TypeError("live Phase-A decision-margin state is nonserializable")

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("live Phase-A decision-margin state is nonserializable")


@dataclass(frozen=True, slots=True)
class V1SignedWorkResourceFrontierCUDAResourceCellBoundary:
    """Outward-safe nominal slack for one exact registered resource cell."""

    resource_coordinate_hash: str
    resource_cell_binding_hash: str
    resource_cell_hash: str
    coordinate_ordinal: int
    coordinate_resource_value: float
    cell_lower_bound: float
    cell_upper_bound: float
    nominal_boundary_slack_lower_bound: float
    scalar_replay_enclosed_boundary_slack_lower_bound: float
    coordinate_endpoint_linf_sensitivity_upper_bound: float = 1.0
    schema: str = (
        "partition_free_v1_signed_work_phase_A_resource_cell_boundary_v1"
    )

    def __post_init__(self) -> None:
        for name in (
            "resource_coordinate_hash",
            "resource_cell_binding_hash",
            "resource_cell_hash",
        ):
            object.__setattr__(
                self,
                name,
                _identity(getattr(self, name), name=name.replace("_", " ")),
            )
        object.__setattr__(
            self,
            "coordinate_ordinal",
            _positive_int(self.coordinate_ordinal, name="coordinate ordinal"),
        )
        value = _nonnegative(
            self.coordinate_resource_value,
            name="coordinate resource value",
        )
        lower = _finite_real(self.cell_lower_bound, name="resource-cell lower bound")
        upper = _finite_real(self.cell_upper_bound, name="resource-cell upper bound")
        if lower > value or value > upper:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                "resource coordinate left its registered closed cell",
            )
        object.__setattr__(self, "coordinate_resource_value", value)
        object.__setattr__(self, "cell_lower_bound", lower)
        object.__setattr__(self, "cell_upper_bound", upper)
        nominal = _nonnegative(
            self.nominal_boundary_slack_lower_bound,
            name="nominal resource-cell boundary slack",
        )
        enclosed = _nonnegative(
            self.scalar_replay_enclosed_boundary_slack_lower_bound,
            name="scalar-replay-enclosed resource-cell boundary slack",
        )
        if enclosed > nominal:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                "numerically enclosed resource-cell slack exceeds nominal slack",
            )
        object.__setattr__(self, "nominal_boundary_slack_lower_bound", nominal)
        object.__setattr__(
            self,
            "scalar_replay_enclosed_boundary_slack_lower_bound",
            enclosed,
        )
        sensitivity = _nonnegative(
            self.coordinate_endpoint_linf_sensitivity_upper_bound,
            name="coordinate endpoint sensitivity",
        )
        if sensitivity != 1.0:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.AUTHORITY_VIOLATION,
                "a registered resource coordinate must remain one-Lipschitz in its endpoint",
            )

    @property
    def strictly_interior_after_scalar_replay(self) -> bool:
        return self.scalar_replay_enclosed_boundary_slack_lower_bound > 0.0

    def __getstate__(self) -> NoReturn:
        raise TypeError("live Phase-A decision-margin state is nonserializable")

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("live Phase-A decision-margin state is nonserializable")


@dataclass(frozen=True, slots=True)
class V1SignedWorkResourceFrontierCUDABranchSupportBoundary:
    """One nominal branch-support boundary in its native scalar unit.

    These records deliberately do not claim route-face robustness.  They say
    how a *registered branch endpoint* can cross the nominal merge/split or Q
    feasibility boundary.  Translating literal-J/FD uncertainty into that
    endpoint coordinate, and proving that any crossed face aliases to the same
    downstream physical action, require separate source-owned premises.
    """

    boundary_kind: str
    branches: tuple[V1SignedWorkLeastWorkBranch, ...]
    native_scalar_unit: str
    nominal_slack_lower_bound: float
    endpoint_linf_sensitivity_upper_bound: float
    registered_resource_cell_alias_proven: bool
    missing_alias_premise: str
    schema: str = (
        "partition_free_v1_signed_work_phase_A_branch_support_boundary_v1"
    )

    def __post_init__(self) -> None:
        if self.boundary_kind not in {
            "optimal_interval_merge_split",
            "certified_infeasible_Q_boundary",
        }:
            raise ValueError("unsupported Phase-A branch-support boundary")
        branches = tuple(self.branches)
        if (
            not branches
            or any(type(value) is not V1SignedWorkLeastWorkBranch for value in branches)
            or len(branches) != len(set(branches))
        ):
            raise TypeError("branch-support boundary requires unique exact branches")
        object.__setattr__(self, "branches", branches)
        if self.native_scalar_unit not in {
            "physical_resource_endpoint",
            "quadratic_response_lower_bound",
        }:
            raise ValueError("unsupported branch-support sensitivity unit")
        object.__setattr__(
            self,
            "nominal_slack_lower_bound",
            _nonnegative(
                self.nominal_slack_lower_bound,
                name="branch-support boundary slack",
            ),
        )
        sensitivity = _nonnegative(
            self.endpoint_linf_sensitivity_upper_bound,
            name="branch-support endpoint sensitivity",
        )
        expected = 2.0 if self.boundary_kind == "optimal_interval_merge_split" else 1.0
        if sensitivity != expected:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.AUTHORITY_VIOLATION,
                "branch-support endpoint sensitivity changed from its exact affine bound",
            )
        if type(self.registered_resource_cell_alias_proven) is not bool:
            raise TypeError("resource-cell alias proof flag must be exact")
        object.__setattr__(
            self,
            "missing_alias_premise",
            _identity(self.missing_alias_premise, name="missing alias premise"),
        )
        if self.registered_resource_cell_alias_proven:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.AUTHORITY_VIOLATION,
                "Phase-A endpoint state cannot prove a downstream physical-action alias",
            )

    def __getstate__(self) -> NoReturn:
        raise TypeError("live Phase-A decision-margin state is nonserializable")

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("live Phase-A decision-margin state is nonserializable")


@dataclass(frozen=True, slots=True)
class V1SignedWorkResourceFrontierCUDADecisionMarginState:
    """Compact live-derived Phase-A boundary state with all authority closed."""

    handle_hash: str
    result_hash: str
    route_axis_hash: str
    route_count: int
    resource_cell_binding_hashes: tuple[str, ...]
    live_branch_resource_reductions: tuple[
        V1SignedWorkResourceFrontierCUDALiveBranchResourceReduction, ...
    ]
    resource_cell_boundaries: tuple[
        V1SignedWorkResourceFrontierCUDAResourceCellBoundary, ...
    ]
    branch_support_boundaries: tuple[
        V1SignedWorkResourceFrontierCUDABranchSupportBoundary, ...
    ]
    minimum_resource_cell_boundary_slack_lower_bound: float
    minimum_alias_safe_branch_support_face_margin_lower_bound: float
    branch_endpoint_linf_sensitivity_upper_bound: float
    missing_premises: tuple[str, ...]
    scalar_host_transfer_bytes: int
    _issuer_token: InitVar[object]
    alias_safe_branch_support_face_sensitivity_certified: bool = False
    source_derived: bool = True
    outward_fp64_enclosed: bool = True
    route_scale_d2h_bytes: int = 0
    route_scale_host_tensor_materializations: int = 0
    c_stream_authority: bool = False
    fqr_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    derivation_policy: str = SIGNED_WORK_RESOURCE_FRONTIER_CUDA_DECISION_MARGIN_POLICY
    schema: str = SIGNED_WORK_RESOURCE_FRONTIER_CUDA_DECISION_MARGIN_STATE_SCHEMA

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _FRONTIER_DECISION_MARGIN_STATE_TOKEN:
            raise TypeError("Phase-A decision-margin states are issuer-only")
        for name in ("handle_hash", "result_hash", "route_axis_hash"):
            object.__setattr__(
                self,
                name,
                _identity(getattr(self, name), name=name.replace("_", " ")),
            )
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="route count"),
        )
        hashes = tuple(
            _identity(value, name="resource-cell binding hash")
            for value in self.resource_cell_binding_hashes
        )
        if not hashes or len(hashes) != len(set(hashes)):
            raise ValueError("decision-margin state requires unique resource-cell bindings")
        object.__setattr__(self, "resource_cell_binding_hashes", hashes)
        reductions = tuple(self.live_branch_resource_reductions)
        cells = tuple(self.resource_cell_boundaries)
        support = tuple(self.branch_support_boundaries)
        if not reductions or any(
            type(value) is not V1SignedWorkResourceFrontierCUDALiveBranchResourceReduction
            for value in reductions
        ):
            raise TypeError("decision-margin state requires live branch reductions")
        if not cells or any(
            type(value) is not V1SignedWorkResourceFrontierCUDAResourceCellBoundary
            for value in cells
        ):
            raise TypeError("decision-margin state requires resource-cell boundaries")
        if not support or any(
            type(value) is not V1SignedWorkResourceFrontierCUDABranchSupportBoundary
            for value in support
        ):
            raise TypeError("decision-margin state requires branch-support boundaries")
        object.__setattr__(self, "live_branch_resource_reductions", reductions)
        object.__setattr__(self, "resource_cell_boundaries", cells)
        object.__setattr__(self, "branch_support_boundaries", support)
        minimum_cell = _nonnegative(
            self.minimum_resource_cell_boundary_slack_lower_bound,
            name="minimum resource-cell boundary slack",
        )
        observed_cell = min(
            value.scalar_replay_enclosed_boundary_slack_lower_bound for value in cells
        )
        if minimum_cell != observed_cell:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                "minimum resource-cell slack does not replay its cell family",
            )
        object.__setattr__(
            self,
            "minimum_resource_cell_boundary_slack_lower_bound",
            minimum_cell,
        )
        face_margin = _nonnegative(
            self.minimum_alias_safe_branch_support_face_margin_lower_bound,
            name="alias-safe branch-support/face margin",
        )
        if face_margin != 0.0:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.AUTHORITY_VIOLATION,
                "current Phase-A state cannot claim a positive alias-safe route-face margin",
            )
        object.__setattr__(
            self,
            "minimum_alias_safe_branch_support_face_margin_lower_bound",
            face_margin,
        )
        sensitivity = _nonnegative(
            self.branch_endpoint_linf_sensitivity_upper_bound,
            name="branch endpoint sensitivity",
        )
        observed_sensitivity = max(
            value.endpoint_linf_sensitivity_upper_bound for value in support
        )
        if sensitivity != observed_sensitivity:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                "branch endpoint sensitivity does not replay its boundary family",
            )
        object.__setattr__(
            self,
            "branch_endpoint_linf_sensitivity_upper_bound",
            sensitivity,
        )
        premises = tuple(
            _identity(value, name="missing Phase-A margin premise")
            for value in self.missing_premises
        )
        if not premises or len(premises) != len(set(premises)):
            raise ValueError("uncertified alias-safe face state must name missing premises")
        object.__setattr__(self, "missing_premises", premises)
        object.__setattr__(
            self,
            "scalar_host_transfer_bytes",
            _nonnegative_int(
                self.scalar_host_transfer_bytes,
                name="bounded scalar host-transfer bytes",
            ),
        )
        if (
            type(self.alias_safe_branch_support_face_sensitivity_certified) is not bool
            or self.alias_safe_branch_support_face_sensitivity_certified
            or type(self.source_derived) is not bool
            or not self.source_derived
            or type(self.outward_fp64_enclosed) is not bool
            or not self.outward_fp64_enclosed
        ):
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.AUTHORITY_VIOLATION,
                "Phase-A margin state overstated its source or face authority",
            )
        route_d2h = _nonnegative_int(
            self.route_scale_d2h_bytes,
            name="route-scale D2H bytes",
        )
        route_host = _nonnegative_int(
            self.route_scale_host_tensor_materializations,
            name="route-scale host tensor materializations",
        )
        if route_d2h != 0 or route_host != 0:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.AUTHORITY_VIOLATION,
                "Phase-A decision-margin state crossed its bounded device-native seam",
            )
        _closed_flags(
            (
                self.c_stream_authority,
                self.fqr_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="Phase-A decision-margin state",
        )
        if (
            self.derivation_policy
            != SIGNED_WORK_RESOURCE_FRONTIER_CUDA_DECISION_MARGIN_POLICY
            or self.schema
            != SIGNED_WORK_RESOURCE_FRONTIER_CUDA_DECISION_MARGIN_STATE_SCHEMA
        ):
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.AUTHORITY_VIOLATION,
                "Phase-A decision-margin semantics changed",
            )

    @property
    def continuum_robustness_claim(self) -> bool:
        return False

    @property
    def state_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "handle_hash": self.handle_hash,
                "result_hash": self.result_hash,
                "route_axis_hash": self.route_axis_hash,
                "route_count": self.route_count,
                "resource_cell_binding_hashes": self.resource_cell_binding_hashes,
                "live_branch_resource_reductions": tuple(
                    (
                        value.branch.value,
                        value.branch_receipt_hash,
                        value.physical_resource_center,
                        value.physical_resource_lower_bound,
                        value.physical_resource_upper_bound,
                    )
                    for value in self.live_branch_resource_reductions
                ),
                "resource_cell_boundaries": tuple(
                    (
                        value.resource_coordinate_hash,
                        value.resource_cell_binding_hash,
                        value.resource_cell_hash,
                        value.nominal_boundary_slack_lower_bound,
                        value.scalar_replay_enclosed_boundary_slack_lower_bound,
                    )
                    for value in self.resource_cell_boundaries
                ),
                "branch_support_boundaries": tuple(
                    (
                        value.boundary_kind,
                        tuple(branch.value for branch in value.branches),
                        value.native_scalar_unit,
                        value.nominal_slack_lower_bound,
                        value.endpoint_linf_sensitivity_upper_bound,
                        value.registered_resource_cell_alias_proven,
                        value.missing_alias_premise,
                    )
                    for value in self.branch_support_boundaries
                ),
                "minimum_resource_cell_boundary_slack_lower_bound": (
                    self.minimum_resource_cell_boundary_slack_lower_bound
                ),
                "minimum_alias_safe_branch_support_face_margin_lower_bound": 0.0,
                "branch_endpoint_linf_sensitivity_upper_bound": (
                    self.branch_endpoint_linf_sensitivity_upper_bound
                ),
                "missing_premises": self.missing_premises,
                "scalar_host_transfer_bytes": self.scalar_host_transfer_bytes,
                "route_scale_d2h_bytes": 0,
                "all_downstream_authority": False,
                "derivation_policy": self.derivation_policy,
            },
            prefix=_STATE_PREFIX + "decision-margin-",
        )

    def __getstate__(self) -> NoReturn:
        raise TypeError("live Phase-A decision-margin state is nonserializable")

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("live Phase-A decision-margin state is nonserializable")


@dataclass(frozen=True, slots=True)
class V1SignedWorkResourceFrontierCUDASequentialBoundaryReceiptV2:
    """Compact Phase-A side of a future sequential v2 D witness.

    The old registered-uncertainty-cell witness requires Phase-A and Phase-B
    theorem tensors to coexist.  This receipt makes no such compatibility
    claim.  It freezes all Phase-A resource-cell, branch-support, and compact
    dual identities before tensor eviction, together with the preregistered
    uncertainty-coordinate axis.  A future v2 witness must still join the
    later Phase-B hard-action alias and uncertainty bounds; this receipt alone
    has no decision or downstream authority.
    """

    phase_a_problem_hash: str
    phase_a_result_hash: str
    phase_a_handle_hash: str
    route_axis_hash: str
    route_count: int
    decision_margin_state_hash: str
    resource_cell_binding_hashes: tuple[str, ...]
    live_branch_resource_reduction_hashes: tuple[str, ...]
    resource_cell_boundary_hashes: tuple[str, ...]
    branch_support_boundary_hashes: tuple[str, ...]
    branch_receipt_hashes: tuple[str, ...]
    dual_margin_identity_hashes: tuple[str, ...]
    registered_uncertainty_coordinate_hashes: tuple[str, ...]
    scalar_host_transfer_bytes: int
    required_later_v2_inputs: tuple[str, ...]
    phase_a_side_complete_for_sequential_v2: bool = True
    compatible_with_old_simultaneous_live_state_witness: bool = False
    standalone_decision_stability_authority: bool = False
    route_scale_d2h_bytes: int = 0
    route_scale_host_tensor_materializations: int = 0
    schema: str = SIGNED_WORK_RESOURCE_FRONTIER_CUDA_SEQUENTIAL_BOUNDARY_SCHEMA_V2

    def __post_init__(self) -> None:
        for name in (
            "phase_a_problem_hash",
            "phase_a_result_hash",
            "phase_a_handle_hash",
            "route_axis_hash",
            "decision_margin_state_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="route count"),
        )
        for name in (
            "resource_cell_binding_hashes",
            "live_branch_resource_reduction_hashes",
            "resource_cell_boundary_hashes",
            "branch_support_boundary_hashes",
            "branch_receipt_hashes",
            "dual_margin_identity_hashes",
            "registered_uncertainty_coordinate_hashes",
            "required_later_v2_inputs",
        ):
            values = tuple(
                _identity(value, name=name.replace("_", " "))
                for value in getattr(self, name)
            )
            if not values or len(values) != len(set(values)):
                raise ValueError(f"{name} must be one nonempty unique axis")
            object.__setattr__(self, name, values)
        object.__setattr__(
            self,
            "scalar_host_transfer_bytes",
            _nonnegative_int(
                self.scalar_host_transfer_bytes,
                name="scalar host transfer bytes",
            ),
        )
        if (
            type(self.phase_a_side_complete_for_sequential_v2) is not bool
            or not self.phase_a_side_complete_for_sequential_v2
            or type(self.compatible_with_old_simultaneous_live_state_witness) is not bool
            or self.compatible_with_old_simultaneous_live_state_witness
            or type(self.standalone_decision_stability_authority) is not bool
            or self.standalone_decision_stability_authority
            or self.route_scale_d2h_bytes != 0
            or self.route_scale_host_tensor_materializations != 0
        ):
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.AUTHORITY_VIOLATION,
                "sequential Phase-A boundary receipt overstated compatibility or authority",
            )
        if self.schema != SIGNED_WORK_RESOURCE_FRONTIER_CUDA_SEQUENTIAL_BOUNDARY_SCHEMA_V2:
            raise ValueError("unsupported sequential Phase-A boundary schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_STATE_PREFIX + "sequential-boundary-v2-")


def _outward_nonnegative_difference_lower(
    upper: float,
    lower: float,
) -> float:
    """Lower-bound a known nonnegative binary64 difference."""

    if not math.isfinite(upper) or not math.isfinite(lower) or upper < lower:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.NONFINITE,
            "cannot enclose an invalid nonnegative difference",
        )
    if upper == lower:
        return 0.0
    return max(0.0, math.nextafter(upper - lower, -math.inf))


def _phase_a_live_branch_resource_reductions(
    *,
    handle: "V1SignedWorkResourceFrontierCUDATheoremStateHandle",
    result: V1SignedWorkResourceFrontierCUDAResult,
) -> tuple[V1SignedWorkResourceFrontierCUDALiveBranchResourceReduction, ...]:
    if torch is None:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.INVALID_TYPE,
            "live Phase-A margin derivation requires tensor support",
        )
    receipt_by_branch = {
        receipt.branch: receipt
        for receipt in (
            result.reference_result.spectral,
            result.reference_result.diagonal_l1,
        )
    }
    reductions: list[
        V1SignedWorkResourceFrontierCUDALiveBranchResourceReduction
    ] = []
    tensors = handle._tensors
    if tensors is None:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.HANDLE_EVICTED,
            "Phase-A theorem-state tensors were evicted before scalar replay",
        )
    for state in handle._branch_states:
        receipt = receipt_by_branch[state.branch]
        state.assert_receipt_identity(receipt)
        resource = tensors.resource
        primal = state.primal
        if (
            not isinstance(resource, torch.Tensor)
            or not isinstance(primal, torch.Tensor)
            or resource.dtype is not torch.float64
            or primal.dtype is not torch.float64
            or resource.device != primal.device
            or resource.ndim != 1
            or primal.ndim != 1
            or int(resource.numel()) != handle.route_count
            or int(primal.numel()) != handle.route_count
        ):
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                "retained Phase-A resource/primal tensors changed geometry",
            )
        center_tensor = _frontier_physical_resource_dot(tensors, primal)
        if center_tensor.ndim != 0 or center_tensor.dtype is not torch.float64:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                "Phase-A resource replay did not produce one FP64 scalar",
            )
        center = float(center_tensor.item())
        lower, upper = _frontier_resource_interval(
            center,
            route_count=handle.route_count,
            absolute_scale=abs(center),
        )
        achieved = receipt.achieved_resource
        if achieved is None:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                "retained optimal Phase-A state lost its achieved resource",
            )
        guard = 64.0 * _EPS * max(1.0, abs(achieved), abs(lower), abs(upper))
        if achieved < lower - guard or achieved > upper + guard:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.CUSTODY_MUTATED,
                "live Phase-A primal no longer replays its branch resource interval",
            )
        reductions.append(
            V1SignedWorkResourceFrontierCUDALiveBranchResourceReduction(
                branch=state.branch,
                branch_receipt_hash=state.branch_receipt_hash,
                physical_resource_center=center,
                physical_resource_lower_bound=lower,
                physical_resource_upper_bound=upper,
                scalar_host_transfer_bytes=np.dtype(np.float64).itemsize,
            )
        )
    if not reductions:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.UNRESOLVED_FRONTIER,
            "Phase-A decision-margin derivation retained no optimal branch state",
        )
    return tuple(reductions)


def _phase_a_resource_cell_boundaries(
    *,
    result: V1SignedWorkResourceFrontierCUDAResult,
    bindings: tuple["V1SignedWorkCUDAResourceCellBinding", ...],
    live_reductions: tuple[
        V1SignedWorkResourceFrontierCUDALiveBranchResourceReduction, ...
    ],
) -> tuple[V1SignedWorkResourceFrontierCUDAResourceCellBoundary, ...]:
    # Local import avoids reversing the authoritative dependency: the CUDA D
    # witness owns the issuer/registration surface and imports this Phase-A
    # solver module for its retained state.
    from .modern_v1_signed_work_decision_witness_cuda import (
        V1SignedWorkCUDAResourceCellBinding,
    )

    coordinates = tuple(result.reference_result.coordinates)
    if (
        not coordinates
        or len(bindings) != len(coordinates)
        or any(type(value) is not V1SignedWorkCUDAResourceCellBinding for value in bindings)
    ):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.INCOMPLETE_DECISION_MARGIN_SOURCE,
            "exact registered resource-cell bindings do not cover Phase A",
        )
    if len({value.partition_registration_hash for value in bindings}) != 1:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
            "Phase-A decision-margin bindings crossed resource partitions",
        )
    maximum_scalar_radius = max(
        value.numerical_radius_upper_bound for value in live_reductions
    )
    boundaries: list[V1SignedWorkResourceFrontierCUDAResourceCellBoundary] = []
    for coordinate, binding in zip(coordinates, bindings, strict=True):
        provenance_tolerance = 64.0 * math.ulp(
            max(1.0, abs(coordinate.total_resource_floor))
        )
        if (
            binding.resource_coordinate_hash != coordinate.coordinate_hash
            or binding.coordinate_ordinal != coordinate.ordinal
            or abs(
                binding.coordinate_resource_value
                - coordinate.total_resource_floor
            )
            > provenance_tolerance
        ):
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                "registered resource-cell binding is stale for its Phase-A coordinate",
            )
        lower_slack = _outward_nonnegative_difference_lower(
            coordinate.total_resource_floor,
            binding.cell_lower_bound,
        )
        upper_slack = _outward_nonnegative_difference_lower(
            binding.cell_upper_bound,
            coordinate.total_resource_floor,
        )
        nominal = min(lower_slack, upper_slack)
        replay_enclosed = _outward_nonnegative_difference_lower(
            nominal,
            min(nominal, maximum_scalar_radius),
        )
        boundaries.append(
            V1SignedWorkResourceFrontierCUDAResourceCellBoundary(
                resource_coordinate_hash=coordinate.coordinate_hash,
                resource_cell_binding_hash=binding.binding_hash,
                resource_cell_hash=binding.resource_cell_hash,
                coordinate_ordinal=coordinate.ordinal,
                coordinate_resource_value=coordinate.total_resource_floor,
                cell_lower_bound=binding.cell_lower_bound,
                cell_upper_bound=binding.cell_upper_bound,
                nominal_boundary_slack_lower_bound=nominal,
                scalar_replay_enclosed_boundary_slack_lower_bound=replay_enclosed,
            )
        )
    return tuple(boundaries)


def _phase_a_branch_support_boundaries(
    *,
    result: V1SignedWorkResourceFrontierCUDAResult,
) -> tuple[V1SignedWorkResourceFrontierCUDABranchSupportBoundary, ...]:
    receipts = (
        result.reference_result.spectral,
        result.reference_result.diagonal_l1,
    )
    if any(
        value.status is V1SignedWorkResourceBranchStatus.UNRESOLVED
        for value in receipts
    ):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.UNRESOLVED_FRONTIER,
            "Phase-A branch support is unresolved",
        )
    optimal = tuple(
        value for value in receipts if value.status is V1SignedWorkResourceBranchStatus.OPTIMAL
    )
    infeasible = tuple(
        value
        for value in receipts
        if value.status is V1SignedWorkResourceBranchStatus.CERTIFIED_INFEASIBLE
    )
    missing_alias = (
        "complete_registered_phase_A_route_face_to_resource_cell_and_phase_B_"
        "physical_action_alias_map"
    )
    boundaries: list[V1SignedWorkResourceFrontierCUDABranchSupportBoundary] = []
    if len(optimal) == 2:
        low, high = sorted(
            optimal,
            key=lambda value: (float(value.achieved_resource), value.branch.value),
        )
        low_upper = float(low.resource_upper_bound)
        high_lower = float(high.achieved_resource)
        slack = _outward_nonnegative_difference_lower(
            max(low_upper, high_lower),
            min(low_upper, high_lower),
        )
        boundaries.append(
            V1SignedWorkResourceFrontierCUDABranchSupportBoundary(
                boundary_kind="optimal_interval_merge_split",
                branches=(low.branch, high.branch),
                native_scalar_unit="physical_resource_endpoint",
                nominal_slack_lower_bound=slack,
                endpoint_linf_sensitivity_upper_bound=2.0,
                registered_resource_cell_alias_proven=False,
                missing_alias_premise=missing_alias,
            )
        )
    for receipt in infeasible:
        slack = _outward_nonnegative_difference_lower(
            receipt.q_min_lower_bound,
            receipt.quadratic_budget,
        )
        boundaries.append(
            V1SignedWorkResourceFrontierCUDABranchSupportBoundary(
                boundary_kind="certified_infeasible_Q_boundary",
                branches=(receipt.branch,),
                native_scalar_unit="quadratic_response_lower_bound",
                nominal_slack_lower_bound=slack,
                endpoint_linf_sensitivity_upper_bound=1.0,
                registered_resource_cell_alias_proven=False,
                missing_alias_premise=missing_alias,
            )
        )
    if not boundaries:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.UNRESOLVED_FRONTIER,
            "Phase-A result has no closed branch-support boundary",
        )
    return tuple(boundaries)


def _derive_phase_a_decision_margin_state(
    *,
    handle: "V1SignedWorkResourceFrontierCUDATheoremStateHandle",
    result: V1SignedWorkResourceFrontierCUDAResult,
    resource_cell_bindings: tuple["V1SignedWorkCUDAResourceCellBinding", ...],
) -> V1SignedWorkResourceFrontierCUDADecisionMarginState:
    reductions = _phase_a_live_branch_resource_reductions(
        handle=handle,
        result=result,
    )
    cells = _phase_a_resource_cell_boundaries(
        result=result,
        bindings=resource_cell_bindings,
        live_reductions=reductions,
    )
    support = _phase_a_branch_support_boundaries(result=result)
    missing_premises = (
        "complete_registered_phase_A_route_face_to_resource_cell_and_phase_B_"
        "physical_action_alias_map",
        "source_norm_literal_J_FD_and_solver_radius_to_phase_A_branch_endpoint_"
        "perturbation_map",
    )
    return V1SignedWorkResourceFrontierCUDADecisionMarginState(
        handle_hash=handle.handle_hash,
        result_hash=result.result_hash,
        route_axis_hash=handle.route_axis_hash,
        route_count=handle.route_count,
        resource_cell_binding_hashes=tuple(
            value.resource_cell_binding_hash for value in cells
        ),
        live_branch_resource_reductions=reductions,
        resource_cell_boundaries=cells,
        branch_support_boundaries=support,
        minimum_resource_cell_boundary_slack_lower_bound=min(
            value.scalar_replay_enclosed_boundary_slack_lower_bound
            for value in cells
        ),
        minimum_alias_safe_branch_support_face_margin_lower_bound=0.0,
        branch_endpoint_linf_sensitivity_upper_bound=max(
            value.endpoint_linf_sensitivity_upper_bound for value in support
        ),
        missing_premises=missing_premises,
        scalar_host_transfer_bytes=sum(
            value.scalar_host_transfer_bytes for value in reductions
        ),
        _issuer_token=_FRONTIER_DECISION_MARGIN_STATE_TOKEN,
    )


def _freeze_phase_a_sequential_boundary_receipt_v2(
    *,
    handle: "V1SignedWorkResourceFrontierCUDATheoremStateHandle",
    result: V1SignedWorkResourceFrontierCUDAResult,
    decision_margin_state: V1SignedWorkResourceFrontierCUDADecisionMarginState,
    registered_uncertainty_coordinate_hashes: tuple[str, ...],
) -> V1SignedWorkResourceFrontierCUDASequentialBoundaryReceiptV2:
    handle.assert_integrity(result=result)
    if (
        type(decision_margin_state)
        is not V1SignedWorkResourceFrontierCUDADecisionMarginState
        or decision_margin_state.handle_hash != handle.handle_hash
        or decision_margin_state.result_hash != result.result_hash
        or decision_margin_state.route_axis_hash != handle.route_axis_hash
        or decision_margin_state.route_count != handle.route_count
    ):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
            "sequential boundary freeze received stale Phase-A margins",
        )
    uncertainty_hashes = tuple(
        _identity(value, name="registered uncertainty coordinate hash")
        for value in registered_uncertainty_coordinate_hashes
    )
    if not uncertainty_hashes or len(uncertainty_hashes) != len(set(uncertainty_hashes)):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.INCOMPLETE_DECISION_MARGIN_SOURCE,
            "sequential boundary freeze requires the preregistered uncertainty axis",
        )
    receipt_by_branch = {
        value.branch: value
        for value in (
            result.reference_result.spectral,
            result.reference_result.diagonal_l1,
        )
    }
    dual_hashes: list[str] = []
    scalar_bytes = decision_margin_state.scalar_host_transfer_bytes
    for branch_state in handle._branch_states:
        branch_receipt = receipt_by_branch[branch_state.branch]
        branch_state.assert_receipt_identity(branch_receipt)
        dual_norms: list[tuple[str, float]] = []
        for role, value in (
            ("low_q_dual", branch_state.low_q_dual),
            ("tail_q_dual", branch_state.tail_q_dual),
            ("block_dual", branch_state.block_dual),
        ):
            if value is not None:
                center = float(torch.linalg.vector_norm(value).item())  # type: ignore[union-attr]
                if not math.isfinite(center):
                    _refuse(
                        V1SignedWorkResourceFrontierCUDARefusalCode.NONFINITE,
                        "Phase-A compact dual norm is nonfinite",
                    )
                dual_norms.append((role, center))
                scalar_bytes += _F64_BYTES
        dual_hashes.append(
            content_hash(
                {
                    "branch": branch_state.branch.value,
                    "branch_receipt_hash": branch_state.branch_receipt_hash,
                    "witness_mode": branch_state.witness_mode,
                    "iterations": branch_state.iterations,
                    "dual_norms": tuple(dual_norms),
                    "route_scale_d2h_bytes": 0,
                },
                prefix=_STATE_PREFIX + "sequential-dual-margin-",
            )
        )
    return V1SignedWorkResourceFrontierCUDASequentialBoundaryReceiptV2(
        phase_a_problem_hash=handle.problem_hash,
        phase_a_result_hash=result.result_hash,
        phase_a_handle_hash=handle.handle_hash,
        route_axis_hash=handle.route_axis_hash,
        route_count=handle.route_count,
        decision_margin_state_hash=decision_margin_state.state_hash,
        resource_cell_binding_hashes=(
            decision_margin_state.resource_cell_binding_hashes
        ),
        live_branch_resource_reduction_hashes=tuple(
            content_hash(value, prefix=_STATE_PREFIX + "live-resource-reduction-")
            for value in decision_margin_state.live_branch_resource_reductions
        ),
        resource_cell_boundary_hashes=tuple(
            content_hash(value, prefix=_STATE_PREFIX + "resource-cell-boundary-")
            for value in decision_margin_state.resource_cell_boundaries
        ),
        branch_support_boundary_hashes=tuple(
            content_hash(value, prefix=_STATE_PREFIX + "branch-support-boundary-")
            for value in decision_margin_state.branch_support_boundaries
        ),
        branch_receipt_hashes=tuple(
            value.branch_receipt_hash
            for value in (
                result.reference_result.spectral,
                result.reference_result.diagonal_l1,
            )
        ),
        dual_margin_identity_hashes=tuple(dual_hashes),
        registered_uncertainty_coordinate_hashes=uncertainty_hashes,
        scalar_host_transfer_bytes=scalar_bytes,
        required_later_v2_inputs=(
            "phase_B_hard_action_to_resource_cell_alias_receipt",
            "registered_uncertainty_bounds_joined_to_phase_A_dual_margins",
        ),
    )


class V1SignedWorkResourceFrontierCUDADecisionMarginLease(
    AbstractContextManager[
        "V1SignedWorkResourceFrontierCUDADecisionMarginLease"
    ]
):
    """Exclusive nonserializable lease over source-derived Phase-A margins."""

    __slots__ = ("_handle", "_token", "_released", "_state")

    def __init__(
        self,
        *,
        _issuer_token: object,
        handle: "V1SignedWorkResourceFrontierCUDATheoremStateHandle",
        state: V1SignedWorkResourceFrontierCUDADecisionMarginState,
    ) -> None:
        if _issuer_token is not _FRONTIER_STATE_EXCLUSIVE_TRANSFER_TOKEN:
            raise TypeError("Phase-A decision-margin leases are issuer-only")
        if type(state) is not V1SignedWorkResourceFrontierCUDADecisionMarginState:
            raise TypeError("decision-margin lease requires exact live-derived state")
        self._handle = handle
        self._token = _issuer_token
        self._released = False
        self._state = state

    def __enter__(self) -> "V1SignedWorkResourceFrontierCUDADecisionMarginLease":
        self._handle.assert_integrity()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> bool:
        if not self._released:
            self._released = True
            self._handle._release_lease(_FRONTIER_STATE_EXCLUSIVE_TRANSFER_TOKEN)
        return False

    @property
    def state(self) -> V1SignedWorkResourceFrontierCUDADecisionMarginState:
        if self._released:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                "Phase-A decision-margin lease was already released",
            )
        self._handle.assert_integrity()
        return self._state

    @property
    def phase_a_resource_cell_boundary_slack_lower_bound(self) -> float:
        return self.state.minimum_resource_cell_boundary_slack_lower_bound

    @property
    def phase_a_alias_safe_branch_support_face_margin_lower_bound(self) -> float:
        return self.state.minimum_alias_safe_branch_support_face_margin_lower_bound

    @property
    def phase_a_branch_endpoint_linf_sensitivity_upper_bound(self) -> float:
        return self.state.branch_endpoint_linf_sensitivity_upper_bound

    @property
    def alias_safe_branch_support_face_sensitivity_certified(self) -> bool:
        return self.state.alias_safe_branch_support_face_sensitivity_certified

    def __getstate__(self) -> NoReturn:
        raise TypeError("Phase-A decision-margin leases are nonserializable")

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("Phase-A decision-margin leases are nonserializable")


class V1SignedWorkResourceFrontierCUDATheoremStateLease(
    AbstractContextManager["V1SignedWorkResourceFrontierCUDATheoremStateLease"]
):
    """Exclusive temporary access to retained Phase-A theorem-state tensors."""

    __slots__ = ("_handle", "_token", "_released")

    def __init__(
        self,
        *,
        _issuer_token: object,
        handle: "V1SignedWorkResourceFrontierCUDATheoremStateHandle",
    ) -> None:
        if _issuer_token is not _FRONTIER_STATE_EXCLUSIVE_TRANSFER_TOKEN:
            raise TypeError("Phase-A theorem-state leases are issuer-only")
        self._handle = handle
        self._token = _issuer_token
        self._released = False

    def __enter__(self) -> "V1SignedWorkResourceFrontierCUDATheoremStateLease":
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> bool:
        if not self._released:
            self._released = True
            self._handle._release_lease(_FRONTIER_STATE_EXCLUSIVE_TRANSFER_TOKEN)
        return False

    def __getstate__(self) -> NoReturn:
        raise TypeError("Phase-A tensor theorem-state leases are nonserializable")

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("Phase-A tensor theorem-state leases are nonserializable")

    @property
    def source_tensors(self) -> _FrontierTorchTensors:
        self._handle.assert_integrity()
        return self._handle._tensors

    @property
    def branch_states(self) -> tuple[_FrontierTensorBranchTheoremState, ...]:
        self._handle.assert_integrity()
        return self._handle._branch_states


class V1SignedWorkResourceFrontierCUDATheoremStateHandle:
    """Identity-bound, nonserializable retained tensor state for Phase A.

    The handle is deliberately outside receipt hashing.  It keeps the certified
    Phase-A branch primal witnesses and final duals alive only inside the
    current process, binds them to the exact result/forecast identities, and
    grants access through a single active lease.
    """

    __slots__ = (
        "problem_hash",
        "policy_hash",
        "forecast_hash",
        "result_hash",
        "response_geometry_hash",
        "route_axis_hash",
        "route_count",
        "response_rank",
        "schema",
        "_tensors",
        "_branch_states",
        "_tensor_versions",
        "_source_geometry_lease",
        "_source_geometry_receipt_hash",
        "_active_lease",
        "_evicted",
        "_stale",
        "_lock",
        "_handle_hash",
    )

    def __init__(
        self,
        *,
        _issuer_token: object,
        problem: V1SignedWorkResourceFrontierCUDAProblem
        | V1SignedWorkResourceFrontierCUDADeviceProblemV1,
        result: V1SignedWorkResourceFrontierCUDAResult,
        forecast: V1SignedWorkResourceFrontierCUDAHBMForecast
        | V1SignedWorkResourceFrontierCUDADeviceHBMForecastV1,
        tensors: _FrontierTorchTensors,
        branch_states: tuple[_FrontierTensorBranchTheoremState, ...],
    ) -> None:
        if _issuer_token is not _FRONTIER_STATE_EXCLUSIVE_TRANSFER_TOKEN:
            raise TypeError("Phase-A theorem-state handles are issuer-only")
        if type(problem) not in {
            V1SignedWorkResourceFrontierCUDAProblem,
            V1SignedWorkResourceFrontierCUDADeviceProblemV1,
        }:
            raise TypeError("theorem-state handle requires an exact Phase-A problem")
        if type(result) is not V1SignedWorkResourceFrontierCUDAResult:
            raise TypeError("theorem-state handle requires an exact result")
        if type(forecast) not in {
            V1SignedWorkResourceFrontierCUDAHBMForecast,
            V1SignedWorkResourceFrontierCUDADeviceHBMForecastV1,
        }:
            raise TypeError("theorem-state handle requires an exact forecast")
        if (
            result.problem_hash != problem.problem_hash
            or result.policy_hash != forecast.policy_hash
            or result.forecast_hash != forecast.forecast_hash
            or forecast.problem_hash != problem.problem_hash
        ):
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                "Phase-A theorem state does not bind the execution identities",
            )
        states = tuple(branch_states)
        if any(type(state) is not _FrontierTensorBranchTheoremState for state in states):
            raise TypeError("theorem-state handle requires exact branch states")
        receipt_by_branch = {
            receipt.branch: receipt
            for receipt in (
                result.reference_result.spectral,
                result.reference_result.diagonal_l1,
            )
        }
        for state in states:
            state.assert_receipt_identity(receipt_by_branch[state.branch])
            if receipt_by_branch[state.branch].status is not (
                V1SignedWorkResourceBranchStatus.OPTIMAL
            ):
                _refuse(
                    V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                    "Phase-A theorem state cannot bind a non-optimal branch",
                )
        self.problem_hash = problem.problem_hash
        self.policy_hash = result.policy_hash
        self.forecast_hash = forecast.forecast_hash
        self.result_hash = result.result_hash
        self.response_geometry_hash = problem.response_geometry_hash
        self.route_axis_hash = problem.route_axis_hash
        self.route_count = problem.route_count
        self.response_rank = problem.rank
        self.schema = SIGNED_WORK_RESOURCE_FRONTIER_CUDA_STATE_SCHEMA
        self._tensors = tensors
        self._branch_states = states
        self._tensor_versions = _frontier_tensor_version_fingerprint(tensors, states)
        self._source_geometry_lease = None
        self._source_geometry_receipt_hash = (
            problem.geometry_receipt_hash
            if type(problem) is V1SignedWorkResourceFrontierCUDADeviceProblemV1
            else None
        )
        self._active_lease = False
        self._evicted = False
        self._stale = False
        self._lock = RLock()
        self._handle_hash = content_hash(
            {
                "schema": self.schema,
                "problem_hash": self.problem_hash,
                "policy_hash": self.policy_hash,
                "forecast_hash": self.forecast_hash,
                "result_hash": self.result_hash,
                "response_geometry_hash": self.response_geometry_hash,
                "route_axis_hash": self.route_axis_hash,
                "route_count": self.route_count,
                "response_rank": self.response_rank,
                "branch_receipt_hashes": tuple(
                    state.branch_receipt_hash for state in states
                ),
                "witness_modes": tuple(state.witness_mode for state in states),
            },
            prefix=_STATE_PREFIX,
        )
        if type(problem) is V1SignedWorkResourceFrontierCUDADeviceProblemV1:
            self._source_geometry_lease = problem._take_geometry_lease_for_state()

    def __getstate__(self) -> NoReturn:
        raise TypeError("Phase-A tensor theorem-state handles are nonserializable")

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("Phase-A tensor theorem-state handles are nonserializable")

    @property
    def handle_hash(self) -> str:
        return self._handle_hash

    @property
    def branch_receipt_hashes(self) -> tuple[str, ...]:
        return tuple(state.branch_receipt_hash for state in self._branch_states)

    @property
    def evicted(self) -> bool:
        return self._evicted

    def assert_integrity(
        self,
        *,
        result: V1SignedWorkResourceFrontierCUDAResult | None = None,
        forecast: V1SignedWorkResourceFrontierCUDAHBMForecast
        | V1SignedWorkResourceFrontierCUDADeviceHBMForecastV1
        | None = None,
    ) -> None:
        if self._evicted:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.HANDLE_EVICTED,
                "Phase-A theorem-state tensors have been evicted",
            )
        if self._stale:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                "Phase-A theorem state has been marked stale",
            )
        if result is not None:
            if (
                type(result) is not V1SignedWorkResourceFrontierCUDAResult
                or result.result_hash != self.result_hash
                or result.problem_hash != self.problem_hash
                or result.policy_hash != self.policy_hash
                or result.forecast_hash != self.forecast_hash
            ):
                _refuse(
                    V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                    "Phase-A theorem state result identity is stale",
                )
        if forecast is not None:
            if (
                type(forecast)
                not in {
                    V1SignedWorkResourceFrontierCUDAHBMForecast,
                    V1SignedWorkResourceFrontierCUDADeviceHBMForecastV1,
                }
                or forecast.forecast_hash != self.forecast_hash
                or forecast.problem_hash != self.problem_hash
                or forecast.policy_hash != self.policy_hash
            ):
                _refuse(
                    V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                    "Phase-A theorem state forecast identity is stale",
                )
        current = _frontier_tensor_version_fingerprint(
            self._tensors,
            self._branch_states,
        )
        if current != self._tensor_versions:
            self._stale = True
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.CUSTODY_MUTATED,
                "Phase-A theorem-state tensor identity or version changed",
            )
        if self._source_geometry_lease is not None:
            receipt = self._source_geometry_lease.receipt
            if receipt.receipt_hash != self._source_geometry_receipt_hash:
                _refuse(
                    V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                    "retained device Phase-A geometry receipt changed",
                )

    def acquire_witness_lease(
        self,
        *,
        result: V1SignedWorkResourceFrontierCUDAResult | None = None,
        forecast: V1SignedWorkResourceFrontierCUDAHBMForecast | None = None,
    ) -> V1SignedWorkResourceFrontierCUDATheoremStateLease:
        with self._lock:
            self.assert_integrity(result=result, forecast=forecast)
            if self._active_lease:
                _refuse(
                    V1SignedWorkResourceFrontierCUDARefusalCode.ACTIVE_LEASE,
                    "Phase-A theorem state already has an active exclusive lease",
                )
            self._active_lease = True
        return V1SignedWorkResourceFrontierCUDATheoremStateLease(
            _issuer_token=_FRONTIER_STATE_EXCLUSIVE_TRANSFER_TOKEN,
            handle=self,
        )

    def acquire_decision_margin_lease(
        self,
        *,
        result: V1SignedWorkResourceFrontierCUDAResult,
        resource_cell_bindings: tuple["V1SignedWorkCUDAResourceCellBinding", ...],
    ) -> V1SignedWorkResourceFrontierCUDADecisionMarginLease:
        """Acquire source-derived Phase-A boundary state under exclusive custody.

        No caller-authored margin, sensitivity, numerical tolerance, or source
        hash is accepted.  The only non-state input is the exact preregistered
        resource-cell binding family.  This method derives nominal cell slack,
        replays every retained optimal branch through one FP64 scalar reduction,
        and exposes the exact affine endpoint sensitivity of branch-support
        boundaries.

        Retained Phase-A state is not sufficient to prove a positive
        *alias-safe route-face* margin: it lacks both the complete face-to-hard-
        action alias map and a norm-compatible map from literal-J/FD/solver
        uncertainty to branch-endpoint motion.  The returned state therefore
        certifies that margin as zero and names both missing premises.  It never
        opens C/FQR/K/O/commit/terminal/Pareto authority.
        """

        if type(resource_cell_bindings) is not tuple:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.INVALID_TYPE,
                "resource-cell bindings must be one exact ordered tuple",
            )
        with self._lock:
            self.assert_integrity(result=result)
            if self._active_lease:
                _refuse(
                    V1SignedWorkResourceFrontierCUDARefusalCode.ACTIVE_LEASE,
                    "Phase-A theorem state already has an active exclusive lease",
                )
            self._active_lease = True
            try:
                state = _derive_phase_a_decision_margin_state(
                    handle=self,
                    result=result,
                    resource_cell_bindings=resource_cell_bindings,
                )
            except BaseException:
                self._active_lease = False
                raise
        return V1SignedWorkResourceFrontierCUDADecisionMarginLease(
            _issuer_token=_FRONTIER_STATE_EXCLUSIVE_TRANSFER_TOKEN,
            handle=self,
            state=state,
        )

    def freeze_sequential_boundary_receipt_v2(
        self,
        *,
        result: V1SignedWorkResourceFrontierCUDAResult,
        decision_margin_state: V1SignedWorkResourceFrontierCUDADecisionMarginState,
        registered_uncertainty_coordinate_hashes: tuple[str, ...],
    ) -> V1SignedWorkResourceFrontierCUDASequentialBoundaryReceiptV2:
        """Compact Phase-A evidence before low-peak theorem-state eviction."""

        with self._lock:
            self.assert_integrity(result=result)
            if self._active_lease:
                _refuse(
                    V1SignedWorkResourceFrontierCUDARefusalCode.ACTIVE_LEASE,
                    "cannot freeze sequential boundaries under another state lease",
                )
            self._active_lease = True
            try:
                return _freeze_phase_a_sequential_boundary_receipt_v2(
                    handle=self,
                    result=result,
                    decision_margin_state=decision_margin_state,
                    registered_uncertainty_coordinate_hashes=(
                        registered_uncertainty_coordinate_hashes
                    ),
                )
            finally:
                self._active_lease = False

    def _release_lease(self, token: object) -> None:
        if token is not _FRONTIER_STATE_EXCLUSIVE_TRANSFER_TOKEN:
            raise TypeError("invalid Phase-A theorem-state lease token")
        try:
            self.assert_integrity()
        finally:
            with self._lock:
                self._active_lease = False

    def evict(self) -> None:
        """Drop live tensor custody after accept/reject/relinearization.

        Eviction is intentionally irreversible for this handle.  It may run
        only when no downstream seam holds the exclusive lease, then it releases
        references to source tensors and branch witness tensors so HBM/RAM can
        be reclaimed.  The content-addressed receipt survives separately; this
        handle can no longer authorize any tensor transfer.
        """

        with self._lock:
            if self._active_lease:
                _refuse(
                    V1SignedWorkResourceFrontierCUDARefusalCode.ACTIVE_LEASE,
                    "cannot evict Phase-A theorem state while a lease is active",
                )
            self._evicted = True
            self._tensors = None
            self._branch_states = ()
            self._tensor_versions = ()
            source_lease = self._source_geometry_lease
            self._source_geometry_lease = None
            if source_lease is not None:
                source_lease.close()


@dataclass(frozen=True, slots=True)
class V1SignedWorkResourceFrontierCUDAExecutionBundle:
    result: V1SignedWorkResourceFrontierCUDAResult
    forecast: V1SignedWorkResourceFrontierCUDAHBMForecast
    theorem_state: V1SignedWorkResourceFrontierCUDATheoremStateHandle | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    schema: str = SIGNED_WORK_RESOURCE_FRONTIER_CUDA_EXECUTION_SCHEMA

    def __post_init__(self) -> None:
        if type(self.result) is not V1SignedWorkResourceFrontierCUDAResult:
            raise TypeError("resource-frontier execution requires exact result")
        if type(self.forecast) is not V1SignedWorkResourceFrontierCUDAHBMForecast:
            raise TypeError("resource-frontier execution requires exact forecast")
        if self.result.forecast_hash != self.forecast.forecast_hash:
            raise ValueError("resource-frontier result and forecast differ")
        if self.theorem_state is not None:
            if type(self.theorem_state) is not (
                V1SignedWorkResourceFrontierCUDATheoremStateHandle
            ):
                raise TypeError("resource-frontier theorem state must be exact")
            self.theorem_state.assert_integrity(
                result=self.result,
                forecast=self.forecast,
            )
        if self.schema != SIGNED_WORK_RESOURCE_FRONTIER_CUDA_EXECUTION_SCHEMA:
            raise ValueError("unsupported resource-frontier execution schema")

    @property
    def theorem_state_available(self) -> bool:
        return self.theorem_state is not None and not self.theorem_state.evicted

    def acquire_theorem_state_lease(
        self,
    ) -> V1SignedWorkResourceFrontierCUDATheoremStateLease:
        if self.theorem_state is None:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.AUTHORITY_VIOLATION,
                "resource-frontier execution retained no tensor theorem state",
            )
        return self.theorem_state.acquire_witness_lease(
            result=self.result,
            forecast=self.forecast,
        )

    def evict_theorem_state(self) -> None:
        if self.theorem_state is not None:
            self.theorem_state.evict()

    @property
    def execution_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "result_hash": self.result.result_hash,
                "forecast_hash": self.forecast.forecast_hash,
            },
            prefix=_EXECUTION_PREFIX,
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkResourceFrontierCUDADeviceExecutionBundleV1:
    """Live zero-copy Phase-A execution and its exact incremental forecast."""

    result: V1SignedWorkResourceFrontierCUDAResult
    forecast: V1SignedWorkResourceFrontierCUDADeviceHBMForecastV1
    theorem_state: V1SignedWorkResourceFrontierCUDATheoremStateHandle | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    schema: str = SIGNED_WORK_RESOURCE_FRONTIER_CUDA_DEVICE_EXECUTION_SCHEMA_V1

    def __post_init__(self) -> None:
        if type(self.result) is not V1SignedWorkResourceFrontierCUDAResult:
            raise TypeError("device Phase-A execution requires exact result")
        if type(self.forecast) is not V1SignedWorkResourceFrontierCUDADeviceHBMForecastV1:
            raise TypeError("device Phase-A execution requires exact device forecast")
        if self.result.forecast_hash != self.forecast.forecast_hash:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                "device Phase-A result and forecast differ",
            )
        if self.theorem_state is not None:
            if type(self.theorem_state) is not V1SignedWorkResourceFrontierCUDATheoremStateHandle:
                raise TypeError("device Phase-A theorem state must be exact")
            self.theorem_state.assert_integrity(
                result=self.result,
                forecast=self.forecast,
            )
        if self.schema != SIGNED_WORK_RESOURCE_FRONTIER_CUDA_DEVICE_EXECUTION_SCHEMA_V1:
            raise ValueError("unsupported device Phase-A execution schema")

    @property
    def theorem_state_available(self) -> bool:
        return self.theorem_state is not None and not self.theorem_state.evicted

    def acquire_theorem_state_lease(
        self,
    ) -> V1SignedWorkResourceFrontierCUDATheoremStateLease:
        if self.theorem_state is None:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.AUTHORITY_VIOLATION,
                "device Phase-A execution retained no theorem state",
            )
        return self.theorem_state.acquire_witness_lease(
            result=self.result,
            forecast=self.forecast,
        )

    def evict_theorem_state(self) -> None:
        if self.theorem_state is not None:
            self.theorem_state.evict()

    @property
    def execution_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "result_hash": self.result.result_hash,
                "forecast_hash": self.forecast.forecast_hash,
            },
            prefix=_EXECUTION_PREFIX + "device-v1-",
        )


def estimate_v1_signed_work_resource_frontier_device_cuda_hbm_v1(
    problem: V1SignedWorkResourceFrontierCUDADeviceProblemV1,
    policy: V1SignedWorkResourceFrontierCUDAPolicy | None = None,
    *,
    device_total_memory_bytes: int | None = None,
    current_reserved_bytes: int | None = None,
) -> V1SignedWorkResourceFrontierCUDADeviceHBMForecastV1:
    """Forecast only allocations added by the live zero-copy Phase-A solve."""

    if type(problem) is not V1SignedWorkResourceFrontierCUDADeviceProblemV1:
        raise TypeError("device Phase-A forecast requires the exact live problem")
    problem.assert_integrity()
    policy = V1SignedWorkResourceFrontierCUDAPolicy() if policy is None else policy
    if type(policy) is not V1SignedWorkResourceFrontierCUDAPolicy:
        raise TypeError("device Phase-A forecast requires exact policy")
    if torch is None or torch.device(policy.torch_device) != problem.device:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.CUDA_REQUIRED,
            "device Phase-A policy and source device differ",
        )
    route_count = problem.route_count
    rank = problem.rank
    blocks = len(problem.block_capacities)

    structural_sources = (
        problem.resource,
        problem.eligible_mask,
        problem.required_hard_action,
    )
    unique_structural_storage: dict[tuple[str, int], int] = {}
    for value in structural_sources:
        storage = value.untyped_storage()
        unique_structural_storage[(str(value.device), int(storage.data_ptr()))] = int(
            storage.nbytes()
        )
    receipt = problem._geometry_lease.receipt
    geometry_bytes = (
        math.prod(receipt.singleton_work_shape)
        + math.prod(receipt.factor_shape)
        + math.prod(receipt.residual_diagonal_shape)
    ) * _F64_BYTES
    aliased = geometry_bytes + sum(unique_structural_storage.values())
    sqrt_residual = route_count * _F64_BYTES
    compact = 2 * blocks * _F64_BYTES

    def _float_bound_bytes(value: Any) -> int:
        scalar_broadcast = (
            int(value.untyped_storage().nbytes()) == int(value.element_size())
            and all(int(step) == 0 for step in value.stride())
        )
        return _F64_BYTES if scalar_broadcast else route_count * _F64_BYTES

    derived_bounds = _float_bound_bytes(problem.eligible_mask) + _float_bound_bytes(
        problem.required_hard_action
    )
    solver_state = (8 * route_count * 2 + route_count) * _F64_BYTES
    solver_state += (2 * rank + 2 + 2 * blocks) * _F64_BYTES
    retained = (3 * route_count + 2 * rank + 2 * blocks + 1) * _F64_BYTES
    transient = (
        10 * route_count + 4 * rank + 4 * max(1, blocks)
    ) * _F64_BYTES
    scalar_workspace = max(1, blocks + 8) * _F64_BYTES
    incremental = (
        sqrt_residual
        + compact
        + derived_bounds
        + solver_state
        + transient
        + scalar_workspace
    )
    device = str(problem.device)
    admission_memory = None
    if device_total_memory_bytes is None or current_reserved_bytes is None:
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                _refuse(
                    V1SignedWorkResourceFrontierCUDARefusalCode.CUDA_REQUIRED,
                    "device Phase-A CUDA forecast requested without CUDA",
                )
            if device_total_memory_bytes is None:
                device_total_memory_bytes = int(
                    torch.cuda.get_device_properties(problem.device).total_memory
                )
            if current_reserved_bytes is None:
                current_reserved_bytes = int(torch.cuda.memory_reserved(problem.device))
            admission_memory = (
                cuda_reusable_memory_bytes(problem.device, framework=torch),
                int(torch.cuda.memory_allocated(problem.device)),
            )
        else:
            if device_total_memory_bytes is None:
                device_total_memory_bytes = max(1 << 30, 4 * incremental)
            if current_reserved_bytes is None:
                current_reserved_bytes = 0
    total = _positive_int(device_total_memory_bytes, name="device total memory")
    current = _nonnegative_int(current_reserved_bytes, name="current reserved bytes")
    ceiling = int(math.floor(policy.hbm_ceiling_fraction * total))
    peak = current + incremental
    return V1SignedWorkResourceFrontierCUDADeviceHBMForecastV1(
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        device=device,
        route_count=route_count,
        response_rank=rank,
        block_count=blocks,
        already_resident_aliased_source_bytes=aliased,
        sqrt_residual_bytes=sqrt_residual,
        compact_block_coordinate_bytes=compact,
        derived_bound_bytes=derived_bounds,
        solver_state_bytes=solver_state,
        retained_theorem_state_bytes=retained,
        transient_workspace_bytes=transient,
        scalar_reduction_workspace_bytes=scalar_workspace,
        forecast_incremental_bytes=incremental,
        current_reserved_bytes=current,
        forecast_peak_reserved_bytes=peak,
        device_total_memory_bytes=total,
        hbm_ceiling_bytes=ceiling,
        admitted=(
            peak <= ceiling
            if admission_memory is None
            else incremental <= admission_memory[0]
            and admission_memory[1] + incremental <= ceiling
        ),
    )


def estimate_v1_signed_work_resource_frontier_cuda_hbm(
    problem: V1SignedWorkResourceFrontierCUDAProblem
    | V1SignedWorkResourceFrontierCUDADeviceProblemV1
    | V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkResourceFrontierCUDAPolicy | None = None,
    *,
    device_total_memory_bytes: int | None = None,
    current_reserved_bytes: int | None = None,
) -> V1SignedWorkResourceFrontierCUDAHBMForecast | V1SignedWorkResourceFrontierCUDADeviceHBMForecastV1:
    if type(problem) is V1SignedWorkResourceFrontierCUDADeviceProblemV1:
        return estimate_v1_signed_work_resource_frontier_device_cuda_hbm_v1(
            problem,
            policy,
            device_total_memory_bytes=device_total_memory_bytes,
            current_reserved_bytes=current_reserved_bytes,
        )
    phase_a_problem = _canonical_phase_a_problem(problem)
    policy = V1SignedWorkResourceFrontierCUDAPolicy() if policy is None else policy
    if type(policy) is not V1SignedWorkResourceFrontierCUDAPolicy:
        raise TypeError("resource-frontier CUDA forecast requires exact policy")
    source = phase_a_problem
    route_count = source.route_count
    rank = source.rank
    blocks = len(source.block_capacities)
    # Phase A owns no signed-F operator state.  Persistent tensors are resource,
    # H, sqrt-residual, lower gate, upper gate, and compact block capacities.
    # Solver state is conservatively counted for two branch lanes even though
    # the landed tensor source currently solves those lanes sequentially.  The
    # transient route envelope covers fused gradients, extrapolated iterates,
    # feasibility buffers, and receipts.
    persistent_source = (4 * route_count + rank * route_count) * _F64_BYTES
    persistent_source += 2 * route_count * _BOOL_BYTES
    persistent_source += max(1, blocks) * 4 * _F64_BYTES
    solver_state = (8 * route_count * 2 + route_count) * _F64_BYTES
    solver_state += (2 * rank + 2 + 2 * blocks) * _F64_BYTES
    retained_theorem_state = (
        3 * route_count + 2 * rank + 2 * blocks + 1
    ) * _F64_BYTES
    transient = (10 * route_count + 4 * rank + 4 * max(1, blocks)) * _F64_BYTES
    incremental = persistent_source + solver_state + transient
    device = policy.torch_device
    admission_memory = None
    if device_total_memory_bytes is None or current_reserved_bytes is None:
        if device.startswith("cuda"):
            if torch is None or not torch.cuda.is_available():
                _refuse(
                    V1SignedWorkResourceFrontierCUDARefusalCode.CUDA_REQUIRED,
                    "resource-frontier CUDA forecast requested without CUDA",
                )
            torch_device = torch.device(device)
            if device_total_memory_bytes is None:
                device_total_memory_bytes = int(
                    torch.cuda.get_device_properties(torch_device).total_memory
                )
            if current_reserved_bytes is None:
                current_reserved_bytes = int(torch.cuda.memory_reserved(torch_device))
            admission_memory = (
                cuda_reusable_memory_bytes(torch_device, framework=torch),
                int(torch.cuda.memory_allocated(torch_device)),
            )
        else:
            if device_total_memory_bytes is None:
                device_total_memory_bytes = max(1 << 30, 4 * incremental)
            if current_reserved_bytes is None:
                current_reserved_bytes = 0
    total = _positive_int(device_total_memory_bytes, name="device total memory")
    current = _nonnegative_int(current_reserved_bytes, name="current reserved bytes")
    ceiling = int(math.floor(policy.hbm_ceiling_fraction * total))
    peak = current + incremental
    return V1SignedWorkResourceFrontierCUDAHBMForecast(
        problem_hash=phase_a_problem.problem_hash,
        policy_hash=policy.policy_hash,
        device=device,
        route_count=route_count,
        response_rank=rank,
        block_count=blocks,
        persistent_source_bytes=persistent_source,
        solver_state_bytes=solver_state,
        retained_theorem_state_bytes=retained_theorem_state,
        transient_workspace_bytes=transient,
        forecast_incremental_bytes=incremental,
        current_reserved_bytes=current,
        forecast_peak_reserved_bytes=peak,
        device_total_memory_bytes=total,
        hbm_ceiling_bytes=ceiling,
        admitted=(
            peak <= ceiling
            if admission_memory is None
            else incremental <= admission_memory[0]
            and admission_memory[1] + incremental <= ceiling
        ),
    )


def _reference_problem_from_cuda_problem(
    problem: V1SignedWorkResourceFrontierCUDAProblem,
) -> V1SignedWorkResourceFrontierProblem:
    source = problem
    eligible = np.asarray(source.eligible_mask, dtype=np.bool_)
    required = (
        np.zeros(source.route_count, dtype=np.bool_)
        if problem.required_hard_action is None
        else np.asarray(problem.required_hard_action, dtype=np.bool_).reshape(-1)
    )
    blocks: list[V1SignedWorkLiveBlock] = []
    for block in source.block_capacities:
        live = int(np.count_nonzero(eligible[block.start : block.stop]))
        required_live_survivors = max(0, live - int(block.removal_capacity))
        blocks.append(
            V1SignedWorkLiveBlock(
                block.block_id,
                block.start,
                block.stop,
                required_live_survivors=required_live_survivors,
            )
        )
    if not blocks:
        blocks.append(
            V1SignedWorkLiveBlock(
                "phase-A-unconstrained",
                0,
                source.route_count,
                required_live_survivors=0,
            )
        )
    geometry = V1SignedWorkKKTReferenceProblem(
        problem_id=f"{problem.source_registration_hash}-resource-frontier-reference",
        resource=source.resource,
        signed_work_matrix=np.zeros((1, source.route_count), dtype=np.float64),
        work_row_weights=np.ones(1, dtype=np.float64),
        low_rank_factor=source.low_rank_factor,
        residual_diagonal=source.residual_diagonal,
        residual_spectral_bound=source.residual_spectral_bound,
        parent_live_axis=eligible,
        selectable_axis=eligible,
        live_blocks=tuple(blocks),
        resource_floor=0.0,
        quadratic_budget=source.quadratic_budget,
        route_axis_hash=source.route_axis_hash,
        signed_work_operator_hash=SIGNED_WORK_RESOURCE_FRONTIER_CUDA_NO_F_OPERATOR_ID,
        literal_j_operator_hash=source.source_geometry_hash,
    )
    return V1SignedWorkResourceFrontierProblem(
        problem_id=f"{problem.source_registration_hash}-endogenous-resource-frontier",
        geometry=geometry,
        required_lower_bound=required,
    )


@dataclass(frozen=True, slots=True)
class _FrontierTorchTensors:
    # ``resource`` may be stored in physical units or in legacy normalized
    # units.  The solver objective scale is separate from the scale used to
    # convert dual support values back to registered physical resource.
    resource: object
    resource_scale: float
    resource_stored_normalized: bool
    resource_objective_alpha: float
    resource_dual_to_physical_scale: float
    resource_objective_support_upper: float
    resource_objective_scale_law: str
    uniform_eligible_resource: float | None
    factor: object
    sqrt_residual: object
    lower: object
    upper: object
    eligible_source: object
    required_source: object
    block_capacities: object
    block_scales: object
    block_starts_stops: tuple[tuple[int, int], ...]
    device: object


@dataclass(slots=True)
class _FrontierQMinimumState:
    z: object
    q_candidate: float
    q_lower_bound: float
    optimality_gap: float
    linear_feasible: bool
    dual_enclosure_closed: bool
    certified_infeasible: bool
    iterations: int


class _FrontierQCandidateSemantics(str, Enum):
    """State what the legacy ``q_min_candidate`` scalar actually witnesses."""

    MINIMIZATION_CANDIDATE_WITH_DUAL_LOWER_BOUND = (
        "minimization_candidate_with_dual_lower_bound"
    )
    CAP_FEASIBILITY_WITNESS_NOT_MINIMUM_Q_RESULT = (
        "cap_feasibility_witness_not_minimum_Q_result"
    )


@dataclass(frozen=True, slots=True)
class _FrontierTensorBranchOutcome:
    """Internal authority carrier retained until coordinates are derived.

    ``V1SignedWorkResourceBranchReceipt`` predates physical-unit compact-block
    replay and cannot be extended here without a cross-schema migration.  This
    wrapper prevents the tensor path from dropping that certification before
    it creates an endogenous coordinate.  It also separates a cap-point
    feasibility witness from a candidate produced by the minimum-Q solve.
    """

    receipt: V1SignedWorkResourceBranchReceipt
    physical_block_replay_certified: bool
    q_candidate_semantics: _FrontierQCandidateSemantics
    theorem_state: "_FrontierTensorBranchTheoremState | None" = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self.receipt) is not V1SignedWorkResourceBranchReceipt:
            raise TypeError("tensor branch outcome requires an exact receipt")
        if type(self.physical_block_replay_certified) is not bool:
            raise TypeError("physical block replay flag must be an exact bool")
        if type(self.q_candidate_semantics) is not _FrontierQCandidateSemantics:
            raise TypeError("tensor branch outcome requires exact Q-candidate semantics")
        if (
            self.receipt.status is V1SignedWorkResourceBranchStatus.OPTIMAL
            and not self.physical_block_replay_certified
        ):
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.UNRESOLVED_FRONTIER,
                "optimal tensor branch lacks physical-unit compact-block replay",
            )
        if (
            self.receipt.status is V1SignedWorkResourceBranchStatus.OPTIMAL
            and self.theorem_state is None
        ):
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.UNRESOLVED_FRONTIER,
                "optimal tensor branch lacks retained Phase-A theorem state",
            )
        if (
            self.receipt.status is not V1SignedWorkResourceBranchStatus.OPTIMAL
            and self.theorem_state is not None
        ):
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.UNRESOLVED_FRONTIER,
                "non-optimal tensor branch cannot retain Phase-A theorem state",
            )
        if self.theorem_state is not None:
            if type(self.theorem_state) is not _FrontierTensorBranchTheoremState:
                raise TypeError("tensor theorem state must be exact or absent")
            self.theorem_state.assert_receipt_identity(self.receipt)


@dataclass(frozen=True, slots=True)
class _FrontierTensorBranchTheoremState:
    """Nonserializable branch witness tensors retained for the next CUDA seam."""

    branch: V1SignedWorkLeastWorkBranch
    branch_receipt_hash: str
    witness_mode: str
    primal: object
    low_q_dual: object | None
    tail_q_dual: object | None
    block_dual: object | None
    iterations: int

    def __post_init__(self) -> None:
        if type(self.branch) is not V1SignedWorkLeastWorkBranch:
            raise TypeError("branch theorem state requires an exact branch")
        object.__setattr__(
            self,
            "branch_receipt_hash",
            _identity(self.branch_receipt_hash, name="branch receipt hash"),
        )
        if self.witness_mode not in {"exact_cap", "pdhg_resource_dual"}:
            raise ValueError("unsupported Phase-A theorem-state witness mode")
        object.__setattr__(
            self,
            "iterations",
            _nonnegative_int(self.iterations, name="theorem-state iterations"),
        )
        if self.primal is None:
            raise TypeError("branch theorem state requires a retained primal witness")
        duals = (self.low_q_dual, self.tail_q_dual, self.block_dual)
        if self.witness_mode == "exact_cap":
            if any(item is not None for item in duals):
                raise ValueError("exact-cap theorem state must not retain duals")
        elif any(item is None for item in duals):
            raise ValueError("PDHG theorem state requires final Q and block duals")

    def __getstate__(self) -> NoReturn:
        raise TypeError("Phase-A tensor theorem state is nonserializable")

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("Phase-A tensor theorem state is nonserializable")

    def assert_receipt_identity(
        self,
        receipt: V1SignedWorkResourceBranchReceipt,
    ) -> None:
        if (
            type(receipt) is not V1SignedWorkResourceBranchReceipt
            or receipt.branch is not self.branch
            or receipt.branch_receipt_hash != self.branch_receipt_hash
        ):
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                "Phase-A theorem state no longer matches its branch receipt",
            )


@dataclass(frozen=True, slots=True)
class _FrontierTensorSolveBundle:
    result: V1SignedWorkResourceFrontierResult
    tensors: _FrontierTorchTensors
    branch_states: tuple[_FrontierTensorBranchTheoremState, ...]


def _frontier_resource_units(
    problem: V1SignedWorkResourceFrontierCUDAProblem,
    *,
    chunk_size: int = 1 << 20,
) -> tuple[float, float | None, float, bool]:
    """Return the objective support unit and exact uniform-resource witness.

    Scaling a positive linear objective by this deterministic constant leaves
    every primal optimizer and KKT face unchanged.  The chunked reduction
    avoids a route-scale masked gather and ``math.fsum`` stabilizes the small
    list of chunk totals.
    """

    partials: list[float] = []
    uniform: float | None = None
    uniform_possible = True
    eligible_count = 0
    for start in range(0, problem.route_count, chunk_size):
        stop = min(problem.route_count, start + chunk_size)
        eligible = problem.eligible_mask[start:stop]
        selected = problem.resource[start:stop][eligible]
        eligible_count += int(selected.size)
        if selected.size:
            chunk_min = float(np.min(selected))
            chunk_max = float(np.max(selected))
            if chunk_min != chunk_max:
                uniform_possible = False
            elif uniform is None:
                uniform = chunk_min
            elif uniform != chunk_min:
                uniform_possible = False
        partials.append(
            float(
                np.sum(
                    problem.resource[start:stop],
                    where=eligible,
                    dtype=np.float64,
                )
            )
        )
    support = math.fsum(partials)
    if not math.isfinite(support) or support < 0.0:
        raise ValueError("Phase-A resource support is not finite")
    # An empty eligible set has the identically-zero objective.  Unit scaling
    # is then exact and avoids a meaningless division by zero.
    scale = support if support > 0.0 else 1.0
    exact_support = eligible_count == 0 or (uniform_possible and uniform == 1.0)
    return scale, (uniform if uniform_possible else None), support, exact_support


def _frontier_resource_physical_support_upper(
    *,
    center: float,
    route_count: int,
    exact_physical_support: bool,
    device_reduction_may_flush_subnormals: bool,
) -> float:
    """Upper-bound a nonnegative physical resource support reduction.

    Host NumPy paths rely on IEEE gradual-underflow binary64 arithmetic and use
    the ordinary relative/subnormal floor in ``_frontier_relative_error_radius``.
    Device torch reductions are admitted under the broader RN-or-FTZ arithmetic
    contract used by the native owners, so a positive subnormal input or
    intermediate may be flushed to zero.  In that case each reduction operation
    can lose less than one minimum normal value, not merely one smallest
    subnormal value.
    """

    value = float(center)
    if not math.isfinite(value) or value < 0.0:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.NONFINITE,
            "resource objective support center is not finite nonnegative",
        )
    if exact_physical_support:
        return value
    operations = max(1, int(route_count) + 2)
    radius = _frontier_relative_error_radius(value, operation_count=operations)
    if device_reduction_may_flush_subnormals:
        radius = math.nextafter(
            radius + float(operations) * float(np.finfo(np.float64).tiny),
            math.inf,
        )
    return math.nextafter(value + radius, math.inf)


def _frontier_resource_objective_support_upper(
    *,
    physical_support_center: float,
    route_count: int,
    exact_physical_support: bool,
    physical_to_objective_scale: float,
    device_reduction_may_flush_subnormals: bool = False,
) -> float:
    """Upper-bound the resource objective support in solver-objective units."""

    scale = float(physical_to_objective_scale)
    if not math.isfinite(scale) or scale < 0.0:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.NONFINITE,
            "resource objective support scale is not finite nonnegative",
        )
    physical_upper = _frontier_resource_physical_support_upper(
        center=physical_support_center,
        route_count=route_count,
        exact_physical_support=exact_physical_support,
        device_reduction_may_flush_subnormals=device_reduction_may_flush_subnormals,
    )
    upper = physical_upper * scale
    if not (exact_physical_support and scale == 1.0):
        upper = math.nextafter(upper, math.inf)
    if not math.isfinite(upper) or upper < 0.0:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.NONFINITE,
            "resource objective support upper overflowed",
        )
    return upper


def _frontier_resource_objective_support_quotient_upper(
    *,
    physical_support_center: float,
    route_count: int,
    exact_physical_support: bool,
    physical_denominator: float,
    device_reduction_may_flush_subnormals: bool = False,
) -> float:
    """Upper-bound normalized support without forming `1 / denominator`."""

    denominator = float(physical_denominator)
    if not math.isfinite(denominator) or denominator <= 0.0:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.NONFINITE,
            "resource objective quotient scale is not finite positive",
        )
    physical_upper = _frontier_resource_physical_support_upper(
        center=physical_support_center,
        route_count=route_count,
        exact_physical_support=exact_physical_support,
        device_reduction_may_flush_subnormals=device_reduction_may_flush_subnormals,
    )
    upper = physical_upper / denominator
    if not (exact_physical_support and physical_upper == denominator):
        upper = math.nextafter(upper, math.inf)
    if not math.isfinite(upper) or upper < 0.0:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.NONFINITE,
            "resource objective quotient upper overflowed",
        )
    return upper


def _frontier_torch_tensors(
    problem: V1SignedWorkResourceFrontierCUDAProblem
    | V1SignedWorkResourceFrontierCUDADeviceProblemV1,
    *,
    device: object,
) -> _FrontierTorchTensors:
    from .modern_v1_factored_spectral_resource_frontier import V1FactoredSignedWorkResourceFrontierCUDADeviceProblemV1
    native = type(problem) is V1FactoredSignedWorkResourceFrontierCUDADeviceProblemV1
    if native or type(problem) is V1SignedWorkResourceFrontierCUDADeviceProblemV1:
        problem.assert_integrity()
        if torch.device(device) != problem.device:  # type: ignore[union-attr]
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                "device Phase-A solve requested another source device",
            )
        resource = problem.resource
        factor = problem.low_rank_factor
        residual = None if native else problem.residual_diagonal
        eligible = problem.eligible_mask
        required = problem.required_hard_action

        def _float_bound(mask: Any) -> Any:
            if (
                int(mask.untyped_storage().nbytes()) == int(mask.element_size())
                and all(int(step) == 0 for step in mask.stride())
            ):
                return mask[0].to(dtype=torch.float64).expand(problem.route_count)
            return mask.to(dtype=torch.float64)

        block_scales = tuple(
            math.sqrt(
                float(max(1, block.stop - block.start))
                * float(problem.route_count)
            )
            for block in problem.block_capacities
        )
        resource_scale = (
            problem.resource_support if problem.resource_support > 0.0 else 1.0
        )
        physical_support_upper = _frontier_resource_objective_support_upper(
            physical_support_center=problem.resource_support,
            route_count=problem.route_count,
            exact_physical_support=problem.literal_unit_resource,
            physical_to_objective_scale=1.0,
            device_reduction_may_flush_subnormals=True,
        )
        if native:
            resource_objective_alpha = 1.0
            resource_dual_to_physical_scale = 1.0
            resource_objective_support_upper = physical_support_upper
            resource_objective_scale_law = (
                "native_physical_unit_resource_objective_beta_1_v1"
            )
        else:
            resource_objective_alpha = 1.0 / resource_scale
            resource_dual_to_physical_scale = resource_scale
            resource_objective_support_upper = _frontier_resource_objective_support_upper(
                physical_support_center=problem.resource_support,
                route_count=problem.route_count,
                exact_physical_support=problem.literal_unit_resource,
                physical_to_objective_scale=resource_objective_alpha,
                device_reduction_may_flush_subnormals=True,
            )
            resource_objective_scale_law = (
                "legacy_device_l1_normalized_resource_objective_v1"
            )
        return _FrontierTorchTensors(
            resource=resource,
            resource_scale=resource_scale,
            resource_stored_normalized=False,
            resource_objective_alpha=resource_objective_alpha,
            resource_dual_to_physical_scale=resource_dual_to_physical_scale,
            resource_objective_support_upper=resource_objective_support_upper,
            resource_objective_scale_law=resource_objective_scale_law,
            uniform_eligible_resource=problem.uniform_eligible_resource,
            factor=factor,
            # Native rounded-Q admits no diagonal branch; do not allocate a
            # route-sized zero vector or invent a diagonal residual bound.
            sqrt_residual=None if native else torch.sqrt(residual),  # type: ignore[union-attr]
            lower=_float_bound(required),
            upper=_float_bound(eligible),
            eligible_source=eligible,
            required_source=required,
            block_capacities=torch.tensor(  # type: ignore[union-attr]
                [
                    float(block.removal_capacity) / scale
                    for block, scale in zip(
                        problem.block_capacities,
                        block_scales,
                        strict=True,
                    )
                ],
                dtype=torch.float64,  # type: ignore[union-attr]
                device=device,
            ),
            block_scales=torch.tensor(  # type: ignore[union-attr]
                block_scales,
                dtype=torch.float64,  # type: ignore[union-attr]
                device=device,
            ),
            block_starts_stops=tuple(
                (int(block.start), int(block.stop))
                for block in problem.block_capacities
            ),
            device=device,
        )
    source = problem
    required = (
        np.zeros(source.route_count, dtype=np.bool_)
        if problem.required_hard_action is None
        else np.asarray(problem.required_hard_action, dtype=np.bool_).reshape(-1)
    )
    (
        resource_scale,
        uniform_resource,
        physical_support_center,
        exact_physical_support,
    ) = _frontier_resource_units(problem)
    resource_objective_support_upper = (
        _frontier_resource_objective_support_quotient_upper(
            physical_support_center=physical_support_center,
            route_count=problem.route_count,
            exact_physical_support=exact_physical_support,
            physical_denominator=resource_scale,
        )
    )
    block_scales = tuple(
        math.sqrt(
            float(max(1, block.stop - block.start))
            * float(source.route_count)
        )
        for block in source.block_capacities
    )
    # ``torch.tensor`` owns the device copy directly.  Wrapping an eager
    # ``np.array(..., copy=True)`` first would transiently duplicate the full
    # rank-major factor in host RAM (about 14.6 GiB at rank 32 / 57M routes)
    # without strengthening immutability: the problem arrays are already
    # readonly and content-addressed.
    return _FrontierTorchTensors(
        resource=torch.tensor(  # type: ignore[union-attr]
            source.resource,
            dtype=torch.float64,  # type: ignore[union-attr]
            device=device,
        ).div_(resource_scale),
        resource_scale=resource_scale,
        resource_stored_normalized=True,
        resource_objective_alpha=1.0,
        resource_dual_to_physical_scale=resource_scale,
        resource_objective_support_upper=resource_objective_support_upper,
        resource_objective_scale_law="legacy_host_l1_normalized_resource_objective_v1",
        uniform_eligible_resource=uniform_resource,
        factor=torch.tensor(  # type: ignore[union-attr]
            source.low_rank_factor,
            dtype=torch.float64,  # type: ignore[union-attr]
            device=device,
        ),
        sqrt_residual=torch.sqrt(  # type: ignore[union-attr]
            torch.tensor(  # type: ignore[union-attr]
                source.residual_diagonal,
                dtype=torch.float64,  # type: ignore[union-attr]
                device=device,
            )
        ),
        lower=torch.tensor(  # type: ignore[union-attr]
            required,
            dtype=torch.float64,  # type: ignore[union-attr]
            device=device,
        ),
        upper=torch.tensor(  # type: ignore[union-attr]
            source.eligible_mask,
            dtype=torch.float64,  # type: ignore[union-attr]
            device=device,
        ),
        eligible_source=torch.tensor(  # type: ignore[union-attr]
            source.eligible_mask,
            dtype=torch.bool,  # type: ignore[union-attr]
            device=device,
        ),
        required_source=torch.tensor(  # type: ignore[union-attr]
            required,
            dtype=torch.bool,  # type: ignore[union-attr]
            device=device,
        ),
        block_capacities=torch.tensor(  # type: ignore[union-attr]
            [
                float(block.removal_capacity) / scale
                for block, scale in zip(
                    source.block_capacities,
                    block_scales,
                    strict=True,
                )
            ],
            dtype=torch.float64,  # type: ignore[union-attr]
            device=device,
        ),
        block_scales=torch.tensor(  # type: ignore[union-attr]
            block_scales,
            dtype=torch.float64,  # type: ignore[union-attr]
            device=device,
        ),
        block_starts_stops=tuple(
            (int(block.start), int(block.stop)) for block in source.block_capacities
        ),
        device=device,
    )


def _frontier_block_forward(tensors: _FrontierTorchTensors, z: object) -> object:
    if not tensors.block_starts_stops:
        return torch.empty(0, dtype=torch.float64, device=tensors.device)  # type: ignore[union-attr]
    return torch.stack(  # type: ignore[union-attr]
        [
            torch.sum(z[start:stop]) / tensors.block_scales[row]  # type: ignore[index,union-attr]
            for row, (start, stop) in enumerate(tensors.block_starts_stops)
        ]
    )


def _frontier_normalized_resource_alpha(tensors: _FrontierTorchTensors) -> float:
    return tensors.resource_objective_alpha


def _frontier_physical_resource_dot(
    tensors: _FrontierTorchTensors,
    z: object,
) -> object:
    value = torch.dot(tensors.resource, z)  # type: ignore[union-attr]
    return (
        value * tensors.resource_scale
        if tensors.resource_stored_normalized
        else value
    )


def _frontier_block_adjoint(
    tensors: _FrontierTorchTensors,
    block_dual: object,
    *,
    route_count: int,
) -> object:
    result = torch.zeros(route_count, dtype=torch.float64, device=tensors.device)  # type: ignore[union-attr]
    for row, (start, stop) in enumerate(tensors.block_starts_stops):
        result[start:stop] = (  # type: ignore[index]
            result[start:stop]
            + block_dual[row] / tensors.block_scales[row]
        )
    return result


def _frontier_linear_cap_point(tensors: _FrontierTorchTensors) -> object:
    """Construct a deterministic feasible box/block cap point.

    Within each disjoint compact block, the available slack is filled at one
    common fractional gate.  For exactly uniform positive route resource this
    is an exact maximizer of the linear box/block problem; otherwise it is only
    a safe warm-start candidate.
    """

    if not tensors.block_starts_stops:
        return tensors.upper.clone()  # type: ignore[attr-defined]
    z = tensors.lower.clone()  # type: ignore[attr-defined]
    for row, (start, stop) in enumerate(tensors.block_starts_stops):
        slack = tensors.upper[start:stop] - tensors.lower[start:stop]  # type: ignore[index,operator]
        available = torch.sum(slack) / tensors.block_scales[row]  # type: ignore[union-attr]
        used = torch.sum(tensors.lower[start:stop]) / tensors.block_scales[row]  # type: ignore[index,union-attr]
        remaining = torch.clamp(  # type: ignore[union-attr]
            tensors.block_capacities[row] - used,
            min=0.0,
        )
        alpha = torch.clamp(  # type: ignore[union-attr]
            remaining
            / torch.clamp(available, min=torch.finfo(torch.float64).tiny),  # type: ignore[union-attr]
            min=0.0,
            max=1.0,
        )
        z[start:stop] = tensors.lower[start:stop] + alpha * slack  # type: ignore[index,operator]
    return z


def _frontier_uniform_cap_resource(
    problem: V1SignedWorkResourceFrontierCUDAProblem
    | V1SignedWorkResourceFrontierCUDADeviceProblemV1,
    uniform_resource: float,
) -> float:
    """Return the exact relaxed linear-cap resource for a uniform axis."""

    from .modern_v1_factored_spectral_resource_frontier import V1FactoredSignedWorkResourceFrontierCUDADeviceProblemV1
    device_problem = type(problem) in (V1SignedWorkResourceFrontierCUDADeviceProblemV1,
                                      V1FactoredSignedWorkResourceFrontierCUDADeviceProblemV1)
    if not problem.block_capacities:
        count = (
            problem.eligible_count
            if device_problem
            else int(np.count_nonzero(problem.eligible_mask))
        )
        return float(uniform_resource) * float(count)
    counts: list[float] = []
    for block_index, block in enumerate(problem.block_capacities):
        eligible_count = (
            problem.block_eligible_counts[block_index]
            if device_problem
            else int(
                np.count_nonzero(
                    problem.eligible_mask[block.start : block.stop]
                )
            )
        )
        counts.append(float(min(eligible_count, block.removal_capacity)))
    return float(uniform_resource) * math.fsum(counts)


def _frontier_outward_interval(
    center: float,
    *,
    operation_count: int,
    absolute_scale: float | None = None,
) -> tuple[float, float]:
    """Outward-enclose one FP64 reduction without assuming exact rounding.

    For a nonnegative dot/sum, ``absolute_scale == abs(center)`` is enough.
    Dual objectives can contain cancellation, so their caller supplies the
    sum of absolute term magnitudes instead.  ``gamma/(1-gamma)`` converts the
    usual forward-error statement (which is relative to the exact result)
    into an enclosure around the computed center; using ``gamma * center``
    directly is not outward on the upper side.
    """

    if not math.isfinite(center):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.NONFINITE,
            "cannot enclose a nonfinite FP64 reduction",
        )
    count = max(1, int(operation_count))
    raw_gamma = (count * _EPS) / max(_EPS, 1.0 - count * _EPS)
    gamma = min(0.25, raw_gamma)
    scale = max(
        1.0,
        abs(center),
        0.0 if absolute_scale is None else abs(float(absolute_scale)),
    )
    radius = (gamma / max(_EPS, 1.0 - gamma)) * scale
    return (
        math.nextafter(center - radius, -math.inf),
        math.nextafter(center + radius, math.inf),
    )


def _frontier_relative_error_radius(
    absolute_scale: float,
    *,
    operation_count: int,
) -> float:
    """Return an outward FP64 forward-error radius for one term.

    Unlike ``_frontier_outward_interval``, this primitive intentionally does
    not impose a unit absolute scale.  It is used to add independent error
    sources with their own natural magnitudes; forcing every source to scale
    one would turn a rank-length matvec error into an O(N eps) route penalty.
    A subnormal absolute term covers the edge where the relative model alone
    is undefined.
    """

    scale = abs(float(absolute_scale))
    if not math.isfinite(scale):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.NONFINITE,
            "cannot bound a nonfinite FP64 error scale",
        )
    count = max(1, int(operation_count))
    raw_gamma = (count * _EPS) / max(_EPS, 1.0 - count * _EPS)
    gamma = min(0.25, raw_gamma)
    smallest_subnormal = float(np.nextafter(np.float64(0.0), np.float64(1.0)))
    radius = (
        (gamma / max(_EPS, 1.0 - gamma)) * scale
        + float(count) * smallest_subnormal
    )
    return math.nextafter(radius, math.inf)


def _frontier_resource_interval(
    center: float,
    *,
    route_count: int,
    absolute_scale: float | None = None,
) -> tuple[float, float]:
    """Outward-enclose a nonnegative physical resource reduction."""

    lower, upper = _frontier_outward_interval(
        center,
        operation_count=route_count + 2,
        absolute_scale=absolute_scale,
    )
    return max(0.0, lower), max(0.0, upper)


def _frontier_repair_and_certify_physical_blocks(
    problem: V1SignedWorkResourceFrontierCUDAProblem,
    tensors: _FrontierTorchTensors,
    z: object,
    *,
    allow_inward_repair: bool,
) -> bool:
    """Certify compact capacities in their registered physical units.

    PDHG uses algebraically equivalent scaled block rows.  Its normalized
    feasibility tolerance is not authority to exceed an integer physical
    capacity.  This replay therefore performs bounded scalar reductions in
    unscaled units.  A fractional witness that lands on an unresolved FP64
    boundary is moved a few reduction-error radii toward the already-legal
    required lower action, then Q and resource are replayed by the caller.

    The route vector remains device-resident.  Only one scalar per compact
    block crosses to the host, and the function returns whether it modified
    ``z`` in place.
    """

    if not tensors.block_starts_stops:
        return False
    repaired = False
    for (start, stop), block in zip(
        tensors.block_starts_stops,
        problem.block_capacities,
        strict=True,
    ):
        segment = z[start:stop]  # type: ignore[index]
        lower = tensors.lower[start:stop]  # type: ignore[index]
        span = stop - start
        capacity = float(block.removal_capacity)
        if capacity == 0.0:
            if int(torch.count_nonzero(segment).item()) == 0:  # type: ignore[union-attr]
                continue
            if not allow_inward_repair:
                _refuse(
                    V1SignedWorkResourceFrontierCUDARefusalCode.UNRESOLVED_FRONTIER,
                    "zero-capacity physical compact block is nonempty",
                )
            segment.copy_(lower)  # type: ignore[attr-defined]
            repaired = True
            continue
        center = float(torch.sum(segment).item())  # type: ignore[union-attr]
        _lower_sum, upper_sum = _frontier_outward_interval(
            center,
            operation_count=span,
            absolute_scale=abs(center),
        )
        if upper_sum <= capacity:
            continue
        # An exact required-lower segment has a structural certificate: the
        # problem constructor already proved its integer count <= capacity.
        if bool(torch.equal(segment, lower)):  # type: ignore[union-attr]
            continue
        if not allow_inward_repair:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.UNRESOLVED_FRONTIER,
                "physical compact-block replay straddled or exceeded capacity",
            )

        lower_center = float(torch.sum(lower).item())  # type: ignore[union-attr]
        slack_center = center - lower_center
        if slack_center <= 0.0:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.UNRESOLVED_FRONTIER,
                "physical compact-block boundary has no removable slack",
            )
        if lower_center > capacity:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                "required lower action exceeds a physical compact capacity",
            )

        # Four error radii leave room for the scaling multiply, assignment,
        # and certification replay.  Recompute from the current device values;
        # no host approximation is allowed to certify the result by itself.
        _unused, cap_guarded_upper = _frontier_outward_interval(
            capacity,
            operation_count=span + 3,
            absolute_scale=capacity,
        )
        guard = max(
            math.ulp(capacity if capacity > 0.0 else 1.0),
            4.0 * max(0.0, cap_guarded_upper - capacity),
        )
        safe_target = max(
            lower_center,
            math.nextafter(capacity - guard, -math.inf),
        )
        if safe_target <= lower_center:
            segment.copy_(lower)  # type: ignore[attr-defined]
        else:
            alpha = min(
                1.0,
                max(0.0, (safe_target - lower_center) / slack_center),
            )
            # Reuse the route segment as its own scratch so physical replay
            # does not add a block- or route-scale workspace allocation.
            segment.sub_(lower).mul_(alpha).add_(lower)  # type: ignore[attr-defined]
        repaired = True

        if bool(torch.equal(segment, lower)):  # type: ignore[union-attr]
            continue
        replay_center = float(torch.sum(segment).item())  # type: ignore[union-attr]
        _replay_lower, replay_upper = _frontier_outward_interval(
            replay_center,
            operation_count=span,
            absolute_scale=abs(replay_center),
        )
        if replay_upper > capacity:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.UNRESOLVED_FRONTIER,
                "inward-repaired physical compact block did not certify",
            )
    return repaired


def _frontier_q_value(
    problem: V1SignedWorkResourceFrontierCUDAProblem,
    tensors: _FrontierTorchTensors,
    branch: V1SignedWorkLeastWorkBranch,
    z: object,
) -> tuple[object, object, object]:
    low = tensors.factor @ z  # type: ignore[operator]
    common = torch.dot(low, low)  # type: ignore[union-attr]
    if branch is V1SignedWorkLeastWorkBranch.SPECTRAL:
        tail = math.sqrt(problem.residual_spectral_bound) * z
        return common + torch.dot(tail, tail), low, tail  # type: ignore[union-attr]
    if branch is V1SignedWorkLeastWorkBranch.DIAGONAL_L1:
        tail = torch.dot(tensors.sqrt_residual, z)  # type: ignore[union-attr]
        return common + tail * tail, low, tail
    raise TypeError("unknown signed-work Q branch")


def _frontier_q_adjoint(
    problem: V1SignedWorkResourceFrontierCUDAProblem,
    tensors: _FrontierTorchTensors,
    branch: V1SignedWorkLeastWorkBranch,
    low_dual: object,
    tail_dual: object,
) -> object:
    result = tensors.factor.T @ low_dual  # type: ignore[union-attr,operator]
    if branch is V1SignedWorkLeastWorkBranch.SPECTRAL:
        return result + math.sqrt(problem.residual_spectral_bound) * tail_dual
    if branch is V1SignedWorkLeastWorkBranch.DIAGONAL_L1:
        return result + tensors.sqrt_residual * tail_dual  # type: ignore[operator]
    raise TypeError("unknown signed-work Q branch")


def _frontier_q_dual_norm(
    branch: V1SignedWorkLeastWorkBranch,
    low_dual: object,
    tail_dual: object,
) -> object:
    low_norm_sq = torch.dot(low_dual, low_dual)  # type: ignore[union-attr]
    if branch is V1SignedWorkLeastWorkBranch.SPECTRAL:
        return torch.sqrt(low_norm_sq + torch.dot(tail_dual, tail_dual))  # type: ignore[union-attr]
    if branch is V1SignedWorkLeastWorkBranch.DIAGONAL_L1:
        return torch.sqrt(low_norm_sq + tail_dual * tail_dual)  # type: ignore[union-attr]
    raise TypeError("unknown signed-work Q branch")


def _frontier_step_size(
    problem: V1SignedWorkResourceFrontierCUDAProblem
    | V1SignedWorkResourceFrontierCUDADeviceProblemV1,
    branch: V1SignedWorkLeastWorkBranch,
) -> float:
    source = problem
    factor_norm_sq = float(source._factor_frobenius_norm * source._factor_frobenius_norm)
    if branch is V1SignedWorkLeastWorkBranch.SPECTRAL:
        q_norm_sq = factor_norm_sq + float(source.residual_spectral_bound)
    elif branch is V1SignedWorkLeastWorkBranch.DIAGONAL_L1:
        q_norm_sq = factor_norm_sq + (
            float(source._residual_diagonal_sum)
            if type(source) is V1SignedWorkResourceFrontierCUDADeviceProblemV1
            else float(np.sum(source.residual_diagonal, dtype=np.float64))
        )
    else:
        raise TypeError("unknown signed-work Q branch")
    # Each compact row and RHS are divided by ``sqrt(|b| N)``.  This is exactly
    # the same half-space, but every disjoint row now has squared norm ``1/N``
    # in the global route coordinate.  Dividing by ``|b|`` alone is not enough:
    # AlexNet's 23k-route first block would still throttle a 57M-route objective.
    block_norm_sq = (
        0.0 if not source.block_capacities else 1.0 / float(source.route_count)
    )
    operator_norm = math.sqrt(max(0.0, q_norm_sq + block_norm_sq))
    if operator_norm == 0.0:
        # K=0 and no compact rows: the feasible set is just the registered
        # box.  A unit step is finite and the positive normalized objective
        # still moves toward its exact upper-box solution.
        return 1.0
    step = 0.98 / operator_norm
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("Phase-A PDHG step is not positive finite")
    return step


def _frontier_primal_violation(
    problem: V1SignedWorkResourceFrontierCUDAProblem,
    tensors: _FrontierTorchTensors,
    branch: V1SignedWorkLeastWorkBranch,
    z: object,
) -> tuple[float, float, float]:
    source = problem
    q_value, _low, _tail = _frontier_q_value(problem, tensors, branch, z)
    block_values = _frontier_block_forward(tensors, z)
    if int(block_values.numel()):  # type: ignore[attr-defined]
        block_violation = torch.max(  # type: ignore[union-attr]
            torch.clamp(block_values - tensors.block_capacities, min=0.0)  # type: ignore[operator]
        )
    else:
        block_violation = torch.zeros((), dtype=torch.float64, device=tensors.device)  # type: ignore[union-attr]
    box_violation = torch.maximum(  # type: ignore[union-attr]
        torch.max(torch.clamp(tensors.lower - z, min=0.0)),  # type: ignore[operator,union-attr]
        torch.max(torch.clamp(z - tensors.upper, min=0.0)),  # type: ignore[operator,union-attr]
    )
    q_violation = torch.clamp(q_value - source.quadratic_budget, min=0.0)  # type: ignore[union-attr]
    normalized = torch.maximum(  # type: ignore[union-attr]
        box_violation,
        torch.maximum(
            block_violation / max(1.0, float(tensors.block_capacities.max().item()) if int(tensors.block_capacities.numel()) else 1.0),  # type: ignore[attr-defined]
            q_violation / max(1.0, float(source.quadratic_budget)),
        ),
    )
    return (
        float(normalized.item()),  # type: ignore[attr-defined]
        float(q_value.item()),  # type: ignore[attr-defined]
        float(_frontier_physical_resource_dot(tensors, z).item()),  # type: ignore[attr-defined]
    )


def _frontier_linear_violation(
    tensors: _FrontierTorchTensors,
    z: object,
) -> float:
    block_values = _frontier_block_forward(tensors, z)
    if int(block_values.numel()):  # type: ignore[attr-defined]
        block_violation = torch.max(  # type: ignore[union-attr]
            torch.clamp(  # type: ignore[union-attr]
                block_values - tensors.block_capacities,  # type: ignore[operator]
                min=0.0,
            )
        )
        block_scale = max(
            1.0,
            float(tensors.block_capacities.max().item()),  # type: ignore[attr-defined]
        )
    else:
        block_violation = torch.zeros(  # type: ignore[union-attr]
            (),
            dtype=torch.float64,
            device=tensors.device,
        )
        block_scale = 1.0
    box_violation = torch.maximum(  # type: ignore[union-attr]
        torch.max(torch.clamp(tensors.lower - z, min=0.0)),  # type: ignore[operator,union-attr]
        torch.max(torch.clamp(z - tensors.upper, min=0.0)),  # type: ignore[operator,union-attr]
    )
    return float(  # type: ignore[attr-defined]
        torch.maximum(box_violation, block_violation / block_scale).item()
    )


def _frontier_q_min_dual_lower_bound(
    *,
    problem: V1SignedWorkResourceFrontierCUDAProblem,
    tensors: _FrontierTorchTensors,
    branch: V1SignedWorkLeastWorkBranch,
    low_dual: object,
    tail_dual: object,
    block_dual: object,
) -> float:
    """Evaluate a conservative Fenchel/Lagrange lower bound on ``min Q``."""

    source = problem
    stationarity = _frontier_q_adjoint(
        problem,
        tensors,
        branch,
        low_dual,
        tail_dual,
    ) + _frontier_block_adjoint(  # type: ignore[operator]
        tensors,
        block_dual,
        route_count=source.route_count,
    )
    box_infimum = torch.sum(  # type: ignore[union-attr]
        torch.where(  # type: ignore[union-attr]
            stationarity >= 0.0,  # type: ignore[operator]
            tensors.lower * stationarity,  # type: ignore[operator]
            tensors.upper * stationarity,  # type: ignore[operator]
        )
    )
    dual_norm_sq = torch.dot(low_dual, low_dual)  # type: ignore[union-attr]
    if branch is V1SignedWorkLeastWorkBranch.SPECTRAL:
        dual_norm_sq = dual_norm_sq + torch.dot(tail_dual, tail_dual)  # type: ignore[union-attr]
    elif branch is V1SignedWorkLeastWorkBranch.DIAGONAL_L1:
        dual_norm_sq = dual_norm_sq + tail_dual * tail_dual
    else:
        raise TypeError("unknown signed-work Q branch")
    block_term = (
        torch.dot(block_dual, tensors.block_capacities)  # type: ignore[union-attr]
        if int(block_dual.numel())  # type: ignore[attr-defined]
        else torch.zeros((), dtype=torch.float64, device=tensors.device)  # type: ignore[union-attr]
    )
    lower = box_infimum - 0.25 * dual_norm_sq - block_term
    raw = float(lower.item())  # type: ignore[attr-defined]
    dimension = source.route_count + source.rank + len(source.block_capacities) + 1
    gamma = min(0.25, (dimension * _EPS) / max(_EPS, 1.0 - dimension * _EPS))
    scale = max(
        1.0,
        abs(raw),
        float(torch.abs(box_infimum).item()),  # type: ignore[attr-defined,union-attr]
        float(torch.abs(dual_norm_sq).item()),  # type: ignore[attr-defined,union-attr]
        float(torch.abs(block_term).item()),  # type: ignore[attr-defined,union-attr]
    )
    guarded = math.nextafter(raw - gamma * scale, -math.inf)
    return max(0.0, guarded)


def _minimize_frontier_q_tensor(
    *,
    problem: V1SignedWorkResourceFrontierCUDAProblem,
    tensors: _FrontierTorchTensors,
    policy: V1SignedWorkResourceFrontierCUDAPolicy,
    branch: V1SignedWorkLeastWorkBranch,
) -> _FrontierQMinimumState:
    """Find a cancellation-feasible chart point or certify Q infeasibility."""

    source = problem
    z = tensors.lower.clone()  # type: ignore[attr-defined]
    initial_q, _low, _tail = _frontier_q_value(problem, tensors, branch, z)
    best_q = float(initial_q.item())  # type: ignore[attr-defined]
    best = z.clone()  # type: ignore[attr-defined]
    linear_violation = _frontier_linear_violation(tensors, z)
    linear_feasible = linear_violation <= policy.primal_feasibility_tolerance
    if linear_feasible and best_q <= source.quadratic_budget:
        return _FrontierQMinimumState(
            z=best,
            q_candidate=best_q,
            q_lower_bound=0.0,
            optimality_gap=best_q,
            linear_feasible=True,
            dual_enclosure_closed=False,
            certified_infeasible=False,
            iterations=0,
        )

    z_bar = z.clone()  # type: ignore[attr-defined]
    low_dual = torch.zeros(source.rank, dtype=torch.float64, device=tensors.device)  # type: ignore[union-attr]
    tail_dual = (
        torch.zeros(source.route_count, dtype=torch.float64, device=tensors.device)  # type: ignore[union-attr]
        if branch is V1SignedWorkLeastWorkBranch.SPECTRAL
        else torch.zeros((), dtype=torch.float64, device=tensors.device)  # type: ignore[union-attr]
    )
    block_dual = torch.zeros(  # type: ignore[union-attr]
        len(source.block_capacities),
        dtype=torch.float64,
        device=tensors.device,
    )
    step = _frontier_step_size(problem, branch)
    lower_bound = 0.0
    gap = math.inf
    closed = False
    certified_infeasible = False
    iterations = 0
    finite = True
    for iteration in range(1, policy.max_iterations + 1):
        iterations = iteration
        low_bar = tensors.factor @ z_bar  # type: ignore[operator]
        if branch is V1SignedWorkLeastWorkBranch.SPECTRAL:
            tail_bar = math.sqrt(source.residual_spectral_bound) * z_bar
        else:
            tail_bar = torch.dot(tensors.sqrt_residual, z_bar)  # type: ignore[union-attr]
        low_dual = (low_dual + step * low_bar) / (1.0 + 0.5 * step)
        tail_dual = (tail_dual + step * tail_bar) / (1.0 + 0.5 * step)
        block_projection = _frontier_block_forward(tensors, z_bar)
        if int(block_dual.numel()):  # type: ignore[attr-defined]
            block_dual = torch.clamp(  # type: ignore[union-attr]
                block_dual
                + step * (block_projection - tensors.block_capacities),  # type: ignore[operator]
                min=0.0,
            )
        gradient = _frontier_q_adjoint(
            problem,
            tensors,
            branch,
            low_dual,
            tail_dual,
        ) + _frontier_block_adjoint(  # type: ignore[operator]
            tensors,
            block_dual,
            route_count=source.route_count,
        )
        z_next = torch.clamp(  # type: ignore[union-attr]
            z - step * gradient,  # type: ignore[operator]
            min=tensors.lower,
            max=tensors.upper,
        )
        z_bar = z_next + policy.over_relaxation * (z_next - z)
        z = z_next
        should_check = iteration >= policy.minimum_iterations and (
            iteration % policy.check_interval == 0
            or iteration == policy.max_iterations
        )
        if not should_check:
            continue
        finite = bool(
            torch.isfinite(z).all().item()  # type: ignore[union-attr]
            and torch.isfinite(z_bar).all().item()  # type: ignore[union-attr]
            and torch.isfinite(low_dual).all().item()  # type: ignore[union-attr]
            and torch.isfinite(tail_dual).all().item()  # type: ignore[union-attr]
            and torch.isfinite(block_dual).all().item()  # type: ignore[union-attr]
        )
        if not finite:
            break
        candidate_q_t, _candidate_low, _candidate_tail = _frontier_q_value(
            problem,
            tensors,
            branch,
            z,
        )
        candidate_q = float(candidate_q_t.item())  # type: ignore[attr-defined]
        candidate_linear = _frontier_linear_violation(tensors, z)
        if (
            candidate_linear <= policy.primal_feasibility_tolerance
            and candidate_q < best_q
        ):
            best.copy_(z)  # type: ignore[attr-defined]
            best_q = candidate_q
            linear_feasible = True
        lower_bound = min(
            best_q,
            _frontier_q_min_dual_lower_bound(
                problem=problem,
                tensors=tensors,
                branch=branch,
                low_dual=low_dual,
                tail_dual=tail_dual,
                block_dual=block_dual,
            ),
        )
        gap = max(0.0, best_q - lower_bound)
        tolerance = policy.absolute_gap_tolerance + policy.relative_gap_tolerance * max(
            1.0,
            abs(best_q),
            abs(lower_bound),
        )
        closed = gap <= tolerance
        certified_infeasible = lower_bound > source.quadratic_budget
        if certified_infeasible or (
            linear_feasible and best_q <= source.quadratic_budget
        ) or closed:
            break
    if not finite:
        linear_feasible = False
    return _FrontierQMinimumState(
        z=best,
        q_candidate=best_q,
        q_lower_bound=lower_bound,
        optimality_gap=gap,
        linear_feasible=linear_feasible,
        dual_enclosure_closed=closed or certified_infeasible,
        certified_infeasible=certified_infeasible,
        iterations=iterations,
    )


def _frontier_ephemeral_witness_summary(
    *,
    problem: V1SignedWorkResourceFrontierCUDAProblem,
    policy: V1SignedWorkResourceFrontierCUDAPolicy,
    branch: V1SignedWorkLeastWorkBranch,
    z: object,
    normalized_primal_violation: float,
    achieved_q: float,
    achieved_resource: float,
    iterations: int,
) -> tuple[str, int]:
    """Commit bounded scalar reductions without retaining or exporting ``z``.

    The Phase-A optimum is not consumed by Phase B; only its certified resource
    interval is.  Production therefore checks the primal witness on device,
    records deterministic diagnostic reductions, and evicts the route vector.
    This receipt is an execution summary, not an independently replayable copy
    of the 57M-coordinate witness.
    """

    source = problem
    if torch is None or not isinstance(z, torch.Tensor) or z.device.type != "cuda":
        raise TypeError("ephemeral frontier summaries require one CUDA tensor")
    if z.dtype is not torch.float64 or z.ndim != 1 or int(z.numel()) != source.route_count:
        raise ValueError("ephemeral frontier witness has the wrong CUDA geometry")
    index_moment = torch.zeros((), dtype=torch.float64, device=z.device)
    tile = min(1 << 20, source.route_count)
    for start in range(0, source.route_count, tile):
        stop = min(source.route_count, start + tile)
        weights = torch.arange(
            start + 1,
            stop + 1,
            dtype=torch.float64,
            device=z.device,
        )
        index_moment = index_moment + torch.dot(z[start:stop], weights)
        del weights
    summary = torch.stack(
        (
            torch.sum(z),
            torch.dot(z, z),
            torch.min(z),
            torch.max(z),
            torch.count_nonzero(z).to(dtype=torch.float64),
            index_moment,
        )
    )
    if not bool(torch.isfinite(summary).all().item()):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.NONFINITE,
            "ephemeral frontier witness summary is nonfinite",
        )
    compact = np.asarray(summary.detach().to(device="cpu").numpy(), dtype="<f8")
    if compact.shape != (6,):
        raise RuntimeError("ephemeral frontier summary changed shape")
    summary_hash = content_hash(
        {
            "schema": "partition_free_v1_phase_A_ephemeral_cuda_witness_summary_v1",
            "problem_hash": problem.problem_hash,
            "policy_hash": policy.policy_hash,
            "branch": branch.value,
            "route_count": source.route_count,
            "normalized_primal_violation": normalized_primal_violation,
            "achieved_q": achieved_q,
            "achieved_resource": achieved_resource,
            "iterations": iterations,
            "reductions": tuple(float(value) for value in compact),
            "route_vector_d2h_bytes": 0,
            "route_vector_retained": False,
        },
        prefix="partition-free-v1-phase-A-ephemeral-cuda-witness-",
    )
    return summary_hash, int(compact.nbytes)


def _frontier_resource_upper_bound(
    problem: V1SignedWorkResourceFrontierCUDAProblem,
    tensors: _FrontierTorchTensors,
    branch: V1SignedWorkLeastWorkBranch,
    low_dual: object,
    tail_dual: object,
    block_dual: object,
) -> float:
    source = problem
    q_adjoint = _frontier_q_adjoint(problem, tensors, branch, low_dual, tail_dual)
    block_adjoint = _frontier_block_adjoint(
        tensors,
        block_dual,
        route_count=source.route_count,
    )
    support_arg = -q_adjoint - block_adjoint  # type: ignore[operator]
    support_arg.add_(  # type: ignore[attr-defined]
        tensors.resource,
        alpha=_frontier_normalized_resource_alpha(tensors),
    )
    positive_arg = torch.clamp(support_arg, min=0.0)  # type: ignore[union-attr]
    support_positive = torch.dot(tensors.upper, positive_arg)  # type: ignore[union-attr]
    del positive_arg
    negative_arg = torch.clamp(support_arg, max=0.0)  # type: ignore[union-attr]
    support_negative = torch.dot(tensors.lower, negative_arg)  # type: ignore[union-attr]
    del negative_arg
    support = support_positive + support_negative
    support_absolute = support_positive - support_negative
    q_radius = math.sqrt(source.quadratic_budget)
    q_norm = _frontier_q_dual_norm(branch, low_dual, tail_dual)
    block_support = (
        torch.dot(block_dual, tensors.block_capacities)  # type: ignore[union-attr]
        if int(tensors.block_capacities.numel())  # type: ignore[attr-defined]
        else torch.zeros((), dtype=torch.float64, device=tensors.device)  # type: ignore[union-attr]
    )
    q_support = q_radius * q_norm
    bound = support + q_support + block_support
    # Guard each numerical source at its own algebraic dimension.  In
    # particular, H.T@y contains rank-length route-wise dots: multiplying its
    # norm bound by gamma_N would confuse the number of output coordinates
    # with each dot's reduction length and can destroy route-scale closure.
    q_norm_value = float(q_norm.item())  # type: ignore[attr-defined]
    q_dimension = source.rank + (
        source.route_count
        if branch is V1SignedWorkLeastWorkBranch.SPECTRAL
        else 1
    )
    q_norm_upper = math.nextafter(
        q_norm_value
        + _frontier_relative_error_radius(
            q_norm_value,
            operation_count=2 * q_dimension + 4,
        ),
        math.inf,
    )
    operator_scale = (
        math.sqrt(float(source.route_count))
        * float(source._factor_frobenius_norm)
        * q_norm_upper
    )
    if branch is V1SignedWorkLeastWorkBranch.SPECTRAL:
        tail_operator_scale = (
            math.sqrt(float(source.route_count))
            * math.sqrt(source.residual_spectral_bound)
            # ||tail_dual|| <= ||(low_dual, tail_dual)||.  Reusing the
            # already-computed joint norm avoids a second route-scale scan.
            * q_norm_upper
        )
    else:
        tail_operator_scale = (
            math.sqrt(float(source.route_count))
            * math.sqrt(
                float(source._residual_diagonal_sum)
                if type(source) is V1SignedWorkResourceFrontierCUDADeviceProblemV1
                else float(np.sum(source.residual_diagonal, dtype=np.float64))
            )
            * abs(float(tail_dual.item()))  # type: ignore[attr-defined]
        )
    block_operator_scale = (
        float(  # type: ignore[attr-defined]
            torch.dot(  # type: ignore[union-attr]
                torch.abs(block_dual),  # type: ignore[union-attr]
                tensors.block_scales / float(source.route_count),  # type: ignore[operator]
            ).item()
        )
        if int(block_dual.numel())  # type: ignore[attr-defined]
        else 0.0
    )
    block_operator_scale = math.nextafter(
        block_operator_scale
        + _frontier_relative_error_radius(
            block_operator_scale,
            operation_count=2 * len(source.block_capacities) + 2,
        ),
        math.inf,
    )
    support_absolute_value = float(support_absolute.item())  # type: ignore[attr-defined]
    q_support_value = float(q_support.item())  # type: ignore[attr-defined]
    block_support_value = float(block_support.item())  # type: ignore[attr-defined]
    normalized_center = float(bound.item())  # type: ignore[attr-defined]
    support_reduction_error = _frontier_relative_error_radius(
        support_absolute_value,
        operation_count=2 * source.route_count + 3,
    )
    response_adjoint_error = (
        _frontier_relative_error_radius(
            operator_scale,
            operation_count=source.rank + 2,
        )
        + _frontier_relative_error_radius(
            tail_operator_scale,
            operation_count=3,
        )
    )
    block_adjoint_error = _frontier_relative_error_radius(
        block_operator_scale,
        operation_count=3,
    )
    # The positive resource objective can be stored in physical units for
    # native factored Phase A.  Bound the alpha/add contribution in the units
    # actually used by the dual support before converting the final certificate
    # to registered physical resource.
    resource_normalization_error = _frontier_relative_error_radius(
        tensors.resource_objective_support_upper,
        operation_count=2,
    )
    q_support_error = (
        q_radius * max(0.0, q_norm_upper - q_norm_value)
        + _frontier_relative_error_radius(
            abs(q_support_value),
            operation_count=3,
        )
    )
    block_support_error = _frontier_relative_error_radius(
        abs(block_support_value),
        operation_count=2 * len(source.block_capacities) + 2,
    )
    scalar_add_error = _frontier_relative_error_radius(
        abs(float(support.item()))  # type: ignore[attr-defined]
        + abs(q_support_value)
        + abs(block_support_value),
        operation_count=3,
    )
    normalized_radius = math.fsum(
        (
            support_reduction_error,
            response_adjoint_error,
            block_adjoint_error,
            resource_normalization_error,
            q_support_error,
            block_support_error,
            scalar_add_error,
        ),
    )
    normalized_upper = math.nextafter(
        normalized_center + normalized_radius,
        math.inf,
    )
    physical_center = normalized_upper * tensors.resource_dual_to_physical_scale
    physical_radius = _frontier_relative_error_radius(
        abs(physical_center),
        operation_count=2,
    )
    physical_upper = math.nextafter(
        physical_center,
        math.inf,
    )
    physical_upper = math.nextafter(physical_upper + physical_radius, math.inf)
    return max(0.0, physical_upper)


def _solve_frontier_tensor_branch(
    *,
    resource_problem: V1SignedWorkResourceFrontierCUDAProblem,
    problem: V1SignedWorkResourceFrontierCUDAProblem,
    policy: V1SignedWorkResourceFrontierCUDAPolicy,
    branch: V1SignedWorkLeastWorkBranch,
    tensors: _FrontierTorchTensors,
) -> _FrontierTensorBranchOutcome:
    source = problem
    device = tensors.device
    device_type = getattr(device, "type", str(device).split(":", 1)[0])
    production_cuda_witness = device_type == "cuda"
    # Exact loose-Q fast path.  When every eligible route carries the same
    # registered resource (the canonical unit-count case), the common-fraction
    # cap point is a global maximizer of the box/block problem.  If that point
    # also satisfies this Q branch, Q cannot lower the optimum and no iterative
    # drift is needed.  Nonuniform objectives merely skip this proof path.
    uniform_resource = tensors.uniform_eligible_resource
    if uniform_resource is not None:
        cap = _frontier_linear_cap_point(tensors)
        _frontier_repair_and_certify_physical_blocks(
            problem,
            tensors,
            cap,
            allow_inward_repair=True,
        )
        cap_violation, cap_q, cap_resource_replay = _frontier_primal_violation(
            problem,
            tensors,
            branch,
            cap,
        )
        if cap_violation <= policy.primal_feasibility_tolerance:
            cap_resource = _frontier_uniform_cap_resource(
                problem,
                uniform_resource,
            )
            cap_replay_lower, _cap_replay_upper = _frontier_resource_interval(
                cap_resource_replay,
                route_count=source.route_count,
            )
            _cap_exact_lower, cap_upper = _frontier_resource_interval(
                cap_resource,
                route_count=source.route_count,
            )
            if cap_resource_replay > cap_upper:
                _refuse(
                    V1SignedWorkResourceFrontierCUDARefusalCode.UNRESOLVED_FRONTIER,
                    "uniform-resource cap point did not replay its physical interval",
                )
            witness_summary_hash: str | None = None
            witness_summary_d2h_bytes = 0
            if production_cuda_witness:
                witness_summary_hash, witness_summary_d2h_bytes = (
                    _frontier_ephemeral_witness_summary(
                        problem=problem,
                        policy=policy,
                        branch=branch,
                        z=cap,
                        normalized_primal_violation=cap_violation,
                        achieved_q=cap_q,
                        achieved_resource=cap_replay_lower,
                        iterations=0,
                    )
                )
                z_payload: tuple[float, ...] | None = None
            else:
                z_payload = tuple(float(item) for item in cap.detach().cpu().numpy())
            receipt = V1SignedWorkResourceBranchReceipt(
                problem_hash=resource_problem.problem_hash,
                policy_hash=policy.policy_hash,
                branch=branch,
                status=V1SignedWorkResourceBranchStatus.OPTIMAL,
                required_lower_bound_hash=resource_problem.required_lower_bound_hash,
                baseline_resource=resource_problem.baseline_resource,
                quadratic_budget=source.quadratic_budget,
                # This is a Q-feasible cap witness.  It is deliberately not
                # represented internally as a minimum-Q solver outcome.
                q_min_candidate=cap_q,
                q_min_lower_bound=0.0,
                achieved_resource=cap_replay_lower,
                resource_upper_bound=cap_upper,
                resource_optimality_gap=cap_upper - cap_replay_lower,
                achieved_q=cap_q,
                z=z_payload,
                primal_feasible=True,
                dual_enclosure_closed=True,
                status_reason=(
                    "uniform_resource_linear_cap_is_Q_feasible_exact_optimum_"
                    "q_min_candidate_is_cap_feasibility_witness_not_minimum_Q_result"
                ),
                primal_witness_summary_hash=witness_summary_hash,
                primal_witness_route_count=source.route_count,
                primal_witness_scalar_d2h_bytes=witness_summary_d2h_bytes,
                route_vector_d2h_bytes=0,
                route_vector_retained=not production_cuda_witness,
                ephemeral_cuda_witness_certified=production_cuda_witness,
            )
            return _FrontierTensorBranchOutcome(
                receipt=receipt,
                physical_block_replay_certified=True,
                q_candidate_semantics=(
                    _FrontierQCandidateSemantics.CAP_FEASIBILITY_WITNESS_NOT_MINIMUM_Q_RESULT
                ),
                theorem_state=_FrontierTensorBranchTheoremState(
                    branch=branch,
                    branch_receipt_hash=receipt.branch_receipt_hash,
                    witness_mode="exact_cap",
                    primal=cap,
                    low_q_dual=None,
                    tail_q_dual=None,
                    block_dual=None,
                    iterations=0,
                ),
            )
        del cap
    q_minimum = _minimize_frontier_q_tensor(
        problem=problem,
        tensors=tensors,
        policy=policy,
        branch=branch,
    )
    if q_minimum.certified_infeasible:
        return _FrontierTensorBranchOutcome(
            receipt=V1SignedWorkResourceBranchReceipt(
                problem_hash=resource_problem.problem_hash,
                policy_hash=policy.policy_hash,
                branch=branch,
                status=V1SignedWorkResourceBranchStatus.CERTIFIED_INFEASIBLE,
                required_lower_bound_hash=resource_problem.required_lower_bound_hash,
                baseline_resource=resource_problem.baseline_resource,
                quadratic_budget=source.quadratic_budget,
                q_min_candidate=q_minimum.q_candidate,
                q_min_lower_bound=q_minimum.q_lower_bound,
                achieved_resource=None,
                resource_upper_bound=None,
                resource_optimality_gap=None,
                achieved_q=None,
                z=None,
                primal_feasible=False,
                dual_enclosure_closed=True,
                status_reason="tensor_minimum_Q_dual_lower_bound_exceeds_budget",
                primal_witness_route_count=source.route_count,
                route_vector_retained=False,
            ),
            physical_block_replay_certified=False,
            q_candidate_semantics=(
                _FrontierQCandidateSemantics.MINIMIZATION_CANDIDATE_WITH_DUAL_LOWER_BOUND
            ),
        )
    if (
        not q_minimum.linear_feasible
        or q_minimum.q_candidate > source.quadratic_budget
    ):
        return _FrontierTensorBranchOutcome(
            receipt=V1SignedWorkResourceBranchReceipt(
                problem_hash=resource_problem.problem_hash,
                policy_hash=policy.policy_hash,
                branch=branch,
                status=V1SignedWorkResourceBranchStatus.UNRESOLVED,
                required_lower_bound_hash=resource_problem.required_lower_bound_hash,
                baseline_resource=resource_problem.baseline_resource,
                quadratic_budget=source.quadratic_budget,
                q_min_candidate=q_minimum.q_candidate,
                q_min_lower_bound=q_minimum.q_lower_bound,
                achieved_resource=None,
                resource_upper_bound=None,
                resource_optimality_gap=None,
                achieved_q=None,
                z=None,
                primal_feasible=False,
                dual_enclosure_closed=False,
                status_reason=(
                    "tensor_minimum_Q_has_no_feasible_witness_or_closed_infeasibility"
                ),
                primal_witness_route_count=source.route_count,
                route_vector_retained=False,
            ),
            physical_block_replay_certified=False,
            q_candidate_semantics=(
                _FrontierQCandidateSemantics.MINIMIZATION_CANDIDATE_WITH_DUAL_LOWER_BOUND
            ),
        )
    # Transfer rather than clone the feasible Q-minimum witness.  Keeping the
    # original alive would pin one unnecessary FP64 route vector (roughly
    # 456 MB on the AlexNet axis) throughout the resource maximization.
    z = q_minimum.z
    q_minimum.z = None
    _frontier_repair_and_certify_physical_blocks(
        problem,
        tensors,
        z,
        allow_inward_repair=True,
    )
    z_bar = z.clone()  # type: ignore[attr-defined]
    best = z.clone()  # type: ignore[attr-defined]
    best_available = False
    best_resource = -math.inf
    best_resource_center = -math.inf
    best_q = math.inf
    normalized, q_value, resource_value = _frontier_primal_violation(
        problem,
        tensors,
        branch,
        z,
    )
    if normalized <= policy.primal_feasibility_tolerance:
        resource_lower, _resource_upper = _frontier_resource_interval(
            resource_value,
            route_count=source.route_count,
        )
        best_available = True
        best_resource = resource_lower
        best_resource_center = resource_value
        best_q = q_value
    low_dual = torch.zeros(source.rank, dtype=torch.float64, device=device)  # type: ignore[union-attr]
    tail_dual = (
        torch.zeros(source.route_count, dtype=torch.float64, device=device)  # type: ignore[union-attr]
        if branch is V1SignedWorkLeastWorkBranch.SPECTRAL
        else torch.zeros((), dtype=torch.float64, device=device)  # type: ignore[union-attr]
    )
    block_dual = torch.zeros(  # type: ignore[union-attr]
        len(source.block_capacities),
        dtype=torch.float64,
        device=device,
    )
    step = _frontier_step_size(problem, branch)
    q_radius = math.sqrt(source.quadratic_budget)
    finite = True
    upper_bound = math.inf
    gap = math.inf
    iterations = 0
    for iteration in range(1, policy.max_iterations + 1):
        iterations = iteration
        low_bar = tensors.factor @ z_bar  # type: ignore[operator]
        if branch is V1SignedWorkLeastWorkBranch.SPECTRAL:
            tail_projection = math.sqrt(source.residual_spectral_bound) * z_bar
        else:
            tail_projection = torch.dot(tensors.sqrt_residual, z_bar)  # type: ignore[union-attr]
        low_candidate = low_dual + step * low_bar
        tail_candidate = tail_dual + step * tail_projection
        q_norm = _frontier_q_dual_norm(branch, low_candidate, tail_candidate)
        shrink = torch.clamp(  # type: ignore[union-attr]
            1.0
            - step
            * q_radius
            / torch.clamp(q_norm, min=torch.finfo(torch.float64).tiny),  # type: ignore[union-attr]
            min=0.0,
        )
        low_dual = low_candidate * shrink
        tail_dual = tail_candidate * shrink
        block_projection = _frontier_block_forward(tensors, z_bar)
        if int(block_dual.numel()):  # type: ignore[attr-defined]
            block_dual = torch.clamp(  # type: ignore[union-attr]
                block_dual + step * (block_projection - tensors.block_capacities),  # type: ignore[operator]
                min=0.0,
            )
        gradient = _frontier_q_adjoint(
            problem,
            tensors,
            branch,
            low_dual,
            tail_dual,
        ) + _frontier_block_adjoint(
            tensors,
            block_dual,
            route_count=source.route_count,
        )
        gradient.add_(  # type: ignore[attr-defined]
            tensors.resource,
            alpha=-_frontier_normalized_resource_alpha(tensors),
        )
        z_next = torch.clamp(z - step * gradient, min=tensors.lower, max=tensors.upper)  # type: ignore[union-attr,operator]
        z_bar = z_next + policy.over_relaxation * (z_next - z)
        z = z_next
        should_check = (
            iteration >= policy.minimum_iterations
            and (
                iteration % policy.check_interval == 0
                or iteration == policy.max_iterations
            )
        )
        if not should_check:
            continue
        finite = bool(
            torch.isfinite(z).all().item()  # type: ignore[union-attr]
            and torch.isfinite(z_bar).all().item()  # type: ignore[union-attr]
            and torch.isfinite(low_dual).all().item()  # type: ignore[union-attr]
            and torch.isfinite(tail_dual).all().item()  # type: ignore[union-attr]
            and torch.isfinite(block_dual).all().item()  # type: ignore[union-attr]
        )
        if not finite:
            break
        repaired = _frontier_repair_and_certify_physical_blocks(
            problem,
            tensors,
            z,
            allow_inward_repair=True,
        )
        if repaired:
            # The replay repair is a projection onto a strict interior point;
            # reset extrapolation so the next PDHG step does not reintroduce
            # the pre-repair boundary excess.
            z_bar.copy_(z)  # type: ignore[attr-defined]
        normalized, q_value, resource_value = _frontier_primal_violation(
            problem,
            tensors,
            branch,
            z,
        )
        resource_lower, _resource_upper = _frontier_resource_interval(
            resource_value,
            route_count=source.route_count,
        )
        if normalized <= policy.primal_feasibility_tolerance and (
            resource_lower > best_resource
        ):
            best.copy_(z)  # type: ignore[attr-defined]
            best_available = True
            best_resource = resource_lower
            best_resource_center = resource_value
            best_q = q_value
        upper_bound = _frontier_resource_upper_bound(
            problem,
            tensors,
            branch,
            low_dual,
            tail_dual,
            block_dual,
        )
        if best_available:
            upper_bound = max(upper_bound, best_resource)
            gap = max(0.0, upper_bound - best_resource)
            tolerance = policy.absolute_gap_tolerance + policy.relative_gap_tolerance * max(
                1.0,
                abs(upper_bound),
                abs(best_resource),
            )
            if gap <= tolerance:
                break
    if not finite or not best_available:
        return _FrontierTensorBranchOutcome(
            receipt=V1SignedWorkResourceBranchReceipt(
                problem_hash=resource_problem.problem_hash,
                policy_hash=policy.policy_hash,
                branch=branch,
                status=V1SignedWorkResourceBranchStatus.UNRESOLVED,
                required_lower_bound_hash=resource_problem.required_lower_bound_hash,
                baseline_resource=resource_problem.baseline_resource,
                quadratic_budget=source.quadratic_budget,
                q_min_candidate=q_minimum.q_candidate,
                q_min_lower_bound=q_minimum.q_lower_bound,
                achieved_resource=None,
                resource_upper_bound=None,
                resource_optimality_gap=None,
                achieved_q=None,
                z=None,
                primal_feasible=False,
                dual_enclosure_closed=False,
                status_reason=(
                    "tensor_frontier_nonfinite_or_no_primal_feasible_resource_witness"
                ),
                primal_witness_route_count=source.route_count,
                route_vector_retained=False,
            ),
            physical_block_replay_certified=False,
            q_candidate_semantics=(
                _FrontierQCandidateSemantics.MINIMIZATION_CANDIDATE_WITH_DUAL_LOWER_BOUND
            ),
        )
    upper_bound = max(upper_bound, best_resource)
    gap = max(0.0, upper_bound - best_resource)
    tolerance = policy.absolute_gap_tolerance + policy.relative_gap_tolerance * max(
        1.0,
        abs(upper_bound),
        abs(best_resource),
    )
    closed = bool(gap <= tolerance and iterations >= policy.minimum_iterations)
    witness_summary_hash: str | None = None
    witness_summary_d2h_bytes = 0
    _frontier_repair_and_certify_physical_blocks(
        problem,
        tensors,
        best,
        allow_inward_repair=False,
    )
    best_normalized, replay_q, replay_resource_center = _frontier_primal_violation(
        problem,
        tensors,
        branch,
        best,
    )
    replay_resource_lower, _replay_resource_upper = _frontier_resource_interval(
        replay_resource_center,
        route_count=source.route_count,
    )
    if (
        best_normalized > policy.primal_feasibility_tolerance
        or not math.isclose(replay_q, best_q, rel_tol=0.0, abs_tol=0.0)
        or not math.isclose(
            replay_resource_center,
            best_resource_center,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            replay_resource_lower,
            best_resource,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.UNRESOLVED_FRONTIER,
            "tensor witness did not replay its physical resource/Q certificate",
        )
    if production_cuda_witness:
        witness_summary_hash, witness_summary_d2h_bytes = (
            _frontier_ephemeral_witness_summary(
                problem=problem,
                policy=policy,
                branch=branch,
                z=best,
                normalized_primal_violation=best_normalized,
                achieved_q=best_q,
                achieved_resource=best_resource,
                iterations=iterations,
            )
        )
        z_payload: tuple[float, ...] | None = None
    else:
        z_payload = tuple(float(item) for item in best.detach().cpu().numpy())
    status = (
        V1SignedWorkResourceBranchStatus.OPTIMAL
        if closed
        else V1SignedWorkResourceBranchStatus.UNRESOLVED
    )
    receipt = V1SignedWorkResourceBranchReceipt(
        problem_hash=resource_problem.problem_hash,
        policy_hash=policy.policy_hash,
        branch=branch,
        status=status,
        required_lower_bound_hash=resource_problem.required_lower_bound_hash,
        baseline_resource=resource_problem.baseline_resource,
        quadratic_budget=source.quadratic_budget,
        q_min_candidate=min(q_minimum.q_candidate, best_q),
        q_min_lower_bound=q_minimum.q_lower_bound,
        achieved_resource=best_resource,
        resource_upper_bound=upper_bound,
        resource_optimality_gap=gap,
        achieved_q=best_q,
        z=z_payload,
        primal_feasible=True,
        dual_enclosure_closed=closed,
        status_reason=(
            "torch_tensor_relaxed_branch_resource_maximum_dual_enclosed_"
            "physical_blocks_replayed"
            if closed
            else "torch_tensor_relaxed_branch_resource_maximum_unresolved"
        ),
        primal_witness_summary_hash=witness_summary_hash,
        primal_witness_route_count=source.route_count,
        primal_witness_scalar_d2h_bytes=witness_summary_d2h_bytes,
        route_vector_d2h_bytes=0,
        route_vector_retained=not production_cuda_witness,
        ephemeral_cuda_witness_certified=production_cuda_witness,
    )
    return _FrontierTensorBranchOutcome(
        receipt=receipt,
        physical_block_replay_certified=True,
        q_candidate_semantics=(
            _FrontierQCandidateSemantics.MINIMIZATION_CANDIDATE_WITH_DUAL_LOWER_BOUND
        ),
        theorem_state=(
            _FrontierTensorBranchTheoremState(
                branch=branch,
                branch_receipt_hash=receipt.branch_receipt_hash,
                witness_mode="pdhg_resource_dual",
                primal=best,
                low_q_dual=low_dual,
                tail_q_dual=tail_dual,
                block_dual=block_dual,
                iterations=iterations,
            )
            if status is V1SignedWorkResourceBranchStatus.OPTIMAL
            else None
        ),
    )


def _derive_tensor_coordinates(
    problem: V1SignedWorkResourceFrontierCUDAProblem,
    policy: V1SignedWorkResourceFrontierCUDAPolicy,
    outcomes: tuple[_FrontierTensorBranchOutcome, ...],
) -> tuple[V1SignedWorkEndogenousResourceCoordinate, ...]:
    if any(
        outcome.receipt.status is V1SignedWorkResourceBranchStatus.OPTIMAL
        and not outcome.physical_block_replay_certified
        for outcome in outcomes
    ):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.UNRESOLVED_FRONTIER,
            "tensor coordinate emission requires physical compact-block replay",
        )
    receipts = tuple(outcome.receipt for outcome in outcomes)
    optimal = [
        item
        for item in receipts
        if item.status is V1SignedWorkResourceBranchStatus.OPTIMAL
    ]
    nonempty_resolution = policy.absolute_gap_tolerance + (
        policy.relative_gap_tolerance * max(1.0, abs(problem.baseline_resource))
    )
    if not optimal:
        return ()
    optimal.sort(key=lambda item: (float(item.achieved_resource), item.branch.value))
    groups: list[list[V1SignedWorkResourceBranchReceipt]] = []
    current: list[V1SignedWorkResourceBranchReceipt] = []
    current_upper = -math.inf
    for receipt in optimal:
        lower_endpoint = float(receipt.achieved_resource)
        if not current or lower_endpoint <= current_upper:
            current.append(receipt)
            current_upper = max(current_upper, float(receipt.resource_upper_bound))
        else:
            groups.append(current)
            current = [receipt]
            current_upper = float(receipt.resource_upper_bound)
    groups.append(current)
    coordinates: list[V1SignedWorkEndogenousResourceCoordinate] = []
    all_hashes = tuple(item.branch_receipt_hash for item in receipts)
    canonical = (
        V1SignedWorkLeastWorkBranch.SPECTRAL,
        V1SignedWorkLeastWorkBranch.DIAGONAL_L1,
    )
    receipt_by_branch = {item.branch: item for item in receipts}
    for group in groups:
        rho = min(float(item.achieved_resource) for item in group)
        if rho <= problem.baseline_resource + nonempty_resolution:
            continue
        witnessed = tuple(
            branch
            for branch in canonical
            if receipt_by_branch[branch].status
            is V1SignedWorkResourceBranchStatus.OPTIMAL
            and float(receipt_by_branch[branch].achieved_resource) >= rho
        )
        excluded = tuple(branch for branch in canonical if branch not in witnessed)
        for branch in excluded:
            receipt = receipt_by_branch[branch]
            if receipt.status is V1SignedWorkResourceBranchStatus.OPTIMAL and not (
                float(receipt.resource_upper_bound) < rho
            ):
                _refuse(
                    V1SignedWorkResourceFrontierCUDARefusalCode.UNRESOLVED_FRONTIER,
                    "tensor resource endpoint intervals did not close branch membership",
                )
        coordinates.append(
            V1SignedWorkEndogenousResourceCoordinate(
                problem_hash=problem.problem_hash,
                policy_hash=policy.policy_hash,
                ordinal=len(coordinates) + 1,
                total_resource_floor=rho,
                incremental_resource=rho - problem.baseline_resource,
                baseline_resource=problem.baseline_resource,
                required_lower_bound_hash=problem.required_lower_bound_hash,
                witnessed_feasible_branches=witnessed,
                certified_excluded_branches=excluded,
                source_branch_receipt_hashes=all_hashes,
            )
        )
    return tuple(coordinates)


def _solve_resource_frontier_tensor(
    *,
    problem: V1SignedWorkResourceFrontierCUDAProblem,
    policy: V1SignedWorkResourceFrontierCUDAPolicy,
    device: object,
) -> _FrontierTensorSolveBundle:
    resource_problem = problem
    # Both convex residual branches consume the same resource, H, residual,
    # box, and compact-capacity tensors.  Build/upload them once and reuse
    # them sequentially; rebuilding here would duplicate the dominant
    # rank-major H2D transfer without changing either branch problem.
    tensors = _frontier_torch_tensors(problem, device=device)
    spectral_outcome = _solve_frontier_tensor_branch(
        resource_problem=resource_problem,
        problem=problem,
        policy=policy,
        branch=V1SignedWorkLeastWorkBranch.SPECTRAL,
        tensors=tensors,
    )
    diagonal_outcome = _solve_frontier_tensor_branch(
        resource_problem=resource_problem,
        problem=problem,
        policy=policy,
        branch=V1SignedWorkLeastWorkBranch.DIAGONAL_L1,
        tensors=tensors,
    )
    outcomes = (spectral_outcome, diagonal_outcome)
    receipts = tuple(outcome.receipt for outcome in outcomes)
    spectral, diagonal = receipts
    if any(
        item.status is V1SignedWorkResourceBranchStatus.UNRESOLVED
        for item in receipts
    ):
        union_status = V1SignedWorkResourceUnionStatus.UNRESOLVED
        coordinates: tuple[V1SignedWorkEndogenousResourceCoordinate, ...] = ()
        closed = False
    else:
        coordinates = _derive_tensor_coordinates(resource_problem, policy, outcomes)
        union_status = (
            V1SignedWorkResourceUnionStatus.CERTIFIED_COORDINATES
            if coordinates
            else V1SignedWorkResourceUnionStatus.CERTIFIED_NO_NONEMPTY_ACTION
        )
        closed = True
    result = V1SignedWorkResourceFrontierResult(
        problem_hash=resource_problem.problem_hash,
        policy_hash=policy.policy_hash,
        spectral=spectral,
        diagonal_l1=diagonal,
        union_status=union_status,
        coordinates=coordinates,
        all_branch_alternatives_closed=closed,
    )
    return _FrontierTensorSolveBundle(
        result=result,
        tensors=tensors,
        branch_states=tuple(
            state
            for state in (outcome.theorem_state for outcome in outcomes)
            if state is not None
        ),
    )


def _frontier_tensor_solve_result(
    tensor_solve: object,
) -> V1SignedWorkResourceFrontierResult:
    """Return the serializable result from current tensor solves or old stubs."""

    if type(tensor_solve) is V1SignedWorkResourceFrontierResult:
        return tensor_solve
    if type(tensor_solve) is not _FrontierTensorSolveBundle:
        raise TypeError("tensor resource-frontier solve returned an unknown carrier")
    return tensor_solve.result


def _frontier_tensor_solve_theorem_state(
    tensor_solve: object,
    *,
    problem: V1SignedWorkResourceFrontierCUDAProblem
    | V1SignedWorkResourceFrontierCUDADeviceProblemV1,
    result: V1SignedWorkResourceFrontierCUDAResult,
    forecast: V1SignedWorkResourceFrontierCUDAHBMForecast
    | V1SignedWorkResourceFrontierCUDADeviceHBMForecastV1,
) -> V1SignedWorkResourceFrontierCUDATheoremStateHandle | None:
    """Normalize current tensor solves and legacy test stubs.

    Older focused tests monkeypatch ``_solve_resource_frontier_tensor`` to
    return a bare serializable frontier result.  That compatibility seam keeps
    SciPy/reference replay behavior intact, but cannot authorize a live tensor
    theorem-state handle because no source tensors are present.
    """

    if type(tensor_solve) is V1SignedWorkResourceFrontierResult:
        return None
    if type(tensor_solve) is not _FrontierTensorSolveBundle:
        raise TypeError("tensor resource-frontier solve returned an unknown carrier")
    return V1SignedWorkResourceFrontierCUDATheoremStateHandle(
        _issuer_token=_FRONTIER_STATE_EXCLUSIVE_TRANSFER_TOKEN,
        problem=problem,
        result=result,
        forecast=forecast,
        tensors=tensor_solve.tensors,
        branch_states=tensor_solve.branch_states,
    )


def _certified_coordinates(
    result: V1SignedWorkResourceFrontierResult,
) -> tuple[V1SignedWorkEndogenousResourceCoordinate, ...]:
    try:
        return result.require_phase_b_coordinates()
    except PermissionError as error:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.UNRESOLVED_FRONTIER,
            f"resource frontier has no certified endpoint coordinates: {error}",
        )


def _integer_endpoint_counts_from_unit_resource(
    *,
    problem: V1SignedWorkResourceFrontierCUDAProblem
    | V1SignedWorkResourceFrontierCUDADeviceProblemV1,
    coordinates: tuple[V1SignedWorkEndogenousResourceCoordinate, ...],
) -> tuple[int, ...]:
    source = problem
    unit_resource = (
        source.literal_unit_resource
        if type(source) is V1SignedWorkResourceFrontierCUDADeviceProblemV1
        else np.array_equal(
            source.resource,
            np.ones(source.route_count, dtype=np.float64),
        )
    )
    if not unit_resource:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.NON_UNIT_RESOURCE,
            "hard endpoint-count derivation is valid only for literal unit route resource",
        )
    required_total = (
        problem.required_count
        if type(problem) is V1SignedWorkResourceFrontierCUDADeviceProblemV1
        else (
            0
            if problem.required_hard_action is None
            else int(np.count_nonzero(problem.required_hard_action))
        )
    )
    counts: list[int] = []
    for coordinate in coordinates:
        if coordinate.coordinate_rule_id != SIGNED_WORK_RESOURCE_FRONTIER_COORDINATE_RULE_ID:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                "resource coordinate was not issued by the canonical frontier rule",
            )
        # A hard unit-resource action satisfying ``Resource(z) >= rho`` must
        # contain at least ``ceil(rho)`` routes.  Flooring a nonintegral
        # relaxed frontier coordinate silently changes the registered
        # resource constraint and can admit an action the Phase-B problem did
        # not solve.
        count = int(math.ceil(float(coordinate.total_resource_floor)))
        if count <= required_total:
            continue
        if count > source.route_count:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.NO_INTEGER_ENDPOINT,
                "resource coordinate exceeds the route axis",
            )
        if not counts or count != counts[-1]:
            counts.append(count)
    if not counts:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.NO_INTEGER_ENDPOINT,
            "certified relaxed frontier contains no nonempty integer hard endpoint",
        )
    return tuple(counts)


def derive_v1_signed_work_endpoint_plan_from_resource_frontier(
    *,
    problem: V1SignedWorkResourceFrontierCUDAProblem
    | V1SignedWorkResourceFrontierCUDADeviceProblemV1
    | V1SignedWorkLeastWorkCUDAProblem,
    frontier_result: V1SignedWorkResourceFrontierResult,
) -> tuple[V1SignedWorkDeviceShellEndpointPlanReceipt, tuple[str, ...]]:
    """Turn certified unit-resource coordinates into a pre-C shell plan."""

    phase_a_problem = _canonical_phase_a_problem(problem)
    if type(frontier_result) is not V1SignedWorkResourceFrontierResult:
        raise TypeError("endpoint-plan derivation requires exact frontier result")
    coordinates = _certified_coordinates(frontier_result)
    endpoint_counts = _integer_endpoint_counts_from_unit_resource(
        problem=phase_a_problem,
        coordinates=coordinates,
    )
    coordinate_hashes = tuple(coordinate.coordinate_hash for coordinate in coordinates)
    return (
        V1SignedWorkDeviceShellEndpointPlanReceipt(
            endpoint_counts=endpoint_counts,
            route_axis_hash=phase_a_problem.route_axis_hash,
            route_count=phase_a_problem.route_count,
            source_registration_hash=phase_a_problem.source_registration_hash,
            registered_d_coordinate_hash=phase_a_problem.registered_d_coordinate_hash,
        ),
        coordinate_hashes,
    )


def derive_v1_signed_work_complete_prefix_endpoint_plan_from_resource_frontier(
    *,
    problem: V1SignedWorkResourceFrontierCUDAProblem
    | V1SignedWorkResourceFrontierCUDADeviceProblemV1
    | V1SignedWorkLeastWorkCUDAProblem,
    frontier_result: V1SignedWorkResourceFrontierResult,
) -> tuple[V1SignedWorkDeviceShellEndpointPlanReceipt, tuple[str, ...]]:
    """Derive every unit-count prefix up to the certified resource maximum.

    This is the exact small/medium-domain bridge for C-stage fallback tests:
    the whole nested prefix family is frozen before C opens, so a later
    smaller action after endpoint refusal is not a post-outcome composition.
    The current uint8 shell representation supports at most 255 endpoint
    labels, so route-scale production must use the planned compact rank or
    threshold representation instead of silently truncating this family.
    """

    phase_a_problem = _canonical_phase_a_problem(problem)
    if type(frontier_result) is not V1SignedWorkResourceFrontierResult:
        raise TypeError("complete-prefix derivation requires exact frontier result")
    coordinates = _certified_coordinates(frontier_result)
    endpoint_counts = _integer_endpoint_counts_from_unit_resource(
        problem=phase_a_problem,
        coordinates=coordinates,
    )
    maximum = endpoint_counts[-1]
    required_total = (
        phase_a_problem.required_count
        if type(phase_a_problem) is V1SignedWorkResourceFrontierCUDADeviceProblemV1
        else (
            0
            if phase_a_problem.required_hard_action is None
            else int(np.count_nonzero(phase_a_problem.required_hard_action))
        )
    )
    complete_counts = tuple(range(required_total + 1, maximum + 1))
    if len(complete_counts) > DEVICE_SHELL_MAXIMUM_ENDPOINTS:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.NO_INTEGER_ENDPOINT,
            "the complete unit-prefix endpoint family exceeds the current "
            "uint8 shell-label capacity; route-scale production requires the "
            "compact rank/threshold prefix representation",
        )
    coordinate_hashes = tuple(coordinate.coordinate_hash for coordinate in coordinates)
    return (
        V1SignedWorkDeviceShellEndpointPlanReceipt(
            endpoint_counts=complete_counts,
            route_axis_hash=phase_a_problem.route_axis_hash,
            route_count=phase_a_problem.route_count,
            source_registration_hash=phase_a_problem.source_registration_hash,
            registered_d_coordinate_hash=phase_a_problem.registered_d_coordinate_hash,
        ),
        coordinate_hashes,
    )


def derive_v1_signed_work_endpoint_plan_from_resource_coordinate(
    *,
    problem: V1SignedWorkResourceFrontierCUDAProblem
    | V1SignedWorkResourceFrontierCUDADeviceProblemV1
    | V1SignedWorkLeastWorkCUDAProblem,
    coordinate: V1SignedWorkEndogenousResourceCoordinate,
) -> V1SignedWorkDeviceShellEndpointPlanReceipt:
    """Derive the one-endpoint hard plan for one unit-resource coordinate."""

    if type(coordinate) is not V1SignedWorkEndogenousResourceCoordinate:
        raise TypeError("endpoint-plan derivation requires exact resource coordinate")
    phase_a_problem = _canonical_phase_a_problem(problem)
    endpoint_counts = _integer_endpoint_counts_from_unit_resource(
        problem=phase_a_problem,
        coordinates=(coordinate,),
    )
    return V1SignedWorkDeviceShellEndpointPlanReceipt(
        endpoint_counts=endpoint_counts,
        route_axis_hash=phase_a_problem.route_axis_hash,
        route_count=phase_a_problem.route_count,
        source_registration_hash=phase_a_problem.source_registration_hash,
        registered_d_coordinate_hash=phase_a_problem.registered_d_coordinate_hash,
    )


class V1SignedWorkResourceCoordinateCUDADeviceBindingV1:
    """Exclusive live DeviceProblemV2 binding for one certified coordinate."""

    __slots__ = (
        "resource_coordinate",
        "least_work_problem",
        "endpoint_plan",
        "family_hash",
        "schema",
        "_owner",
        "_closed",
    )

    def __init__(
        self,
        *,
        _issuance_token: object,
        owner: "V1SignedWorkResourceCoordinateCUDADeviceFamilyV1",
        resource_coordinate: V1SignedWorkEndogenousResourceCoordinate,
        least_work_problem: V1SignedWorkLeastWorkCUDADeviceProblemV2,
        endpoint_plan: V1SignedWorkDeviceShellEndpointPlanReceipt | None,
    ) -> None:
        if _issuance_token is not _DEVICE_PHASE_A_ISSUANCE_TOKEN:
            raise TypeError("device coordinate bindings are family-issued only")
        if type(resource_coordinate) is not V1SignedWorkEndogenousResourceCoordinate:
            raise TypeError("device coordinate binding requires exact coordinate")
        if type(least_work_problem) is not V1SignedWorkLeastWorkCUDADeviceProblemV2:
            raise TypeError("device coordinate binding requires exact DeviceProblemV2")
        if endpoint_plan is not None and type(endpoint_plan) is not (
            V1SignedWorkDeviceShellEndpointPlanReceipt
        ):
            raise TypeError("device coordinate endpoint plan must be exact")
        least_work_problem.assert_integrity()
        if least_work_problem.resource_floor != resource_coordinate.total_resource_floor:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                "device coordinate problem has another resource floor",
            )
        self.resource_coordinate = resource_coordinate
        self.least_work_problem = least_work_problem
        self.endpoint_plan = endpoint_plan
        self.family_hash = owner.family_hash
        self.schema = (
            "partition_free_v1_signed_work_resource_coordinate_cuda_device_binding_v1"
        )
        self._owner = owner
        self._closed = False

    @property
    def binding_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "family_hash": self.family_hash,
                "resource_coordinate_hash": self.resource_coordinate.coordinate_hash,
                "least_work_problem_hash": self.least_work_problem.problem_hash,
                "endpoint_plan_hash": (
                    None
                    if self.endpoint_plan is None
                    else self.endpoint_plan.receipt_hash
                ),
            },
            prefix=_DEVICE_FLOOR_FAMILY_PREFIX_V1 + "binding-",
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.least_work_problem.close()
        finally:
            self._closed = True
            self._owner._release_binding(self)

    def __enter__(self) -> "V1SignedWorkResourceCoordinateCUDADeviceBindingV1":
        if self._closed:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.HANDLE_EVICTED,
                "device coordinate binding is closed",
            )
        self.least_work_problem.assert_integrity()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __reduce__(self) -> NoReturn:
        raise TypeError("live device coordinate bindings are nonserializable")

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("live device coordinate bindings are nonserializable")


class V1SignedWorkResourceCoordinateCUDADeviceFamilyV1:
    """Sequential issuer for certified DeviceProblemV2 resource floors.

    The family is created only after Phase-A theorem custody has been evicted.
    It then permits exactly one live fixed-floor problem at a time, in the
    certified coordinate order.  This prevents a route-scale family of source
    leases or solver states from accumulating in HBM.
    """

    __slots__ = (
        "phase_a_problem_hash",
        "frontier_result_hash",
        "decision_margin_state_hash",
        "coordinate_hashes",
        "schema",
        "_phase_a_problem",
        "_execution",
        "_geometry_handle",
        "_signed_work_operator",
        "_coordinates",
        "_next_index",
        "_active_binding",
        "_closed",
        "_family_hash",
    )

    def __init__(
        self,
        *,
        _issuance_token: object,
        phase_a_problem: V1SignedWorkResourceFrontierCUDADeviceProblemV1,
        phase_a_execution: V1SignedWorkResourceFrontierCUDADeviceExecutionBundleV1,
        sequential_boundary_receipt: V1SignedWorkResourceFrontierCUDASequentialBoundaryReceiptV2,
        geometry_handle: V1SampledDeviceResidentDKKTGeometryDeviceHandle,
        signed_work_operator: V1SignedWorkOperatorCUDA,
    ) -> None:
        if _issuance_token is not _DEVICE_PHASE_A_ISSUANCE_TOKEN:
            raise TypeError("device coordinate families are issuer-only")
        if type(phase_a_problem) is not V1SignedWorkResourceFrontierCUDADeviceProblemV1:
            raise TypeError("device floor family requires exact Phase-A problem")
        if type(phase_a_execution) is not V1SignedWorkResourceFrontierCUDADeviceExecutionBundleV1:
            raise TypeError("device floor family requires exact Phase-A execution")
        if type(sequential_boundary_receipt) is not (
            V1SignedWorkResourceFrontierCUDASequentialBoundaryReceiptV2
        ):
            raise TypeError("device floor family requires exact sequential Phase-A evidence")
        if type(geometry_handle) is not V1SampledDeviceResidentDKKTGeometryDeviceHandle:
            raise TypeError("device floor family requires exact geometry handle")
        if type(signed_work_operator) is not V1SignedWorkOperatorCUDA:
            raise TypeError("device floor family requires exact signed-work operator")
        if (
            phase_a_execution.result.problem_hash != phase_a_problem.problem_hash
            or phase_a_execution.result.reference_result.problem_hash
            != phase_a_problem.problem_hash
        ):
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                "device floor family mixes Phase-A problem/result identities",
            )
        if not phase_a_problem.lease_transferred:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.ACTIVE_LEASE,
                "device floor family requires transferred Phase-A theorem custody",
            )
        if phase_a_execution.theorem_state is None or not (
            phase_a_execution.theorem_state.evicted
        ):
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.ACTIVE_LEASE,
                "evict Phase-A theorem custody before issuing resource floors",
            )
        theorem_state = phase_a_execution.theorem_state
        if (
            sequential_boundary_receipt.phase_a_handle_hash
            != theorem_state.handle_hash
            or sequential_boundary_receipt.phase_a_result_hash
            != phase_a_execution.result.result_hash
            or sequential_boundary_receipt.phase_a_problem_hash
            != phase_a_problem.problem_hash
            or sequential_boundary_receipt.route_axis_hash
            != phase_a_problem.route_axis_hash
            or sequential_boundary_receipt.route_count != phase_a_problem.route_count
            or not sequential_boundary_receipt.phase_a_side_complete_for_sequential_v2
            or sequential_boundary_receipt.standalone_decision_stability_authority
        ):
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                "compact Phase-A decision margins are missing or stale",
            )
        phase_a_problem.assert_structural_identity()
        receipt = geometry_handle.receipt
        operator_receipt = signed_work_operator.receipt
        if (
            receipt.receipt_hash != phase_a_problem.geometry_receipt_hash
            or receipt.route_axis_hash != phase_a_problem.route_axis_hash
            # The operator's physical block axis joins the route-ID proposal
            # plane through its ordered route-ID identity only.
            or operator_receipt.route_id_axis_hash != phase_a_problem.route_axis_hash
            or operator_receipt.source_geometry_hash
            != phase_a_problem.source_geometry_hash
        ):
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                "Phase-B operator/geometry differ from the frozen Phase-A source",
            )
        coordinates = _certified_coordinates(phase_a_execution.result.reference_result)
        self.phase_a_problem_hash = phase_a_problem.problem_hash
        self.frontier_result_hash = phase_a_execution.result.reference_result.result_hash
        self.decision_margin_state_hash = (
            sequential_boundary_receipt.decision_margin_state_hash
        )
        self.coordinate_hashes = tuple(item.coordinate_hash for item in coordinates)
        self.schema = SIGNED_WORK_RESOURCE_FRONTIER_CUDA_DEVICE_FLOOR_FAMILY_SCHEMA_V1
        self._phase_a_problem = phase_a_problem
        self._execution = phase_a_execution
        self._geometry_handle = geometry_handle
        self._signed_work_operator = signed_work_operator
        self._coordinates = coordinates
        self._next_index = 0
        self._active_binding = None
        self._closed = False
        self._family_hash = content_hash(
            {
                "schema": self.schema,
                "phase_a_problem_hash": self.phase_a_problem_hash,
                "frontier_result_hash": self.frontier_result_hash,
                "decision_margin_state_hash": self.decision_margin_state_hash,
                "sequential_boundary_receipt_hash": (
                    sequential_boundary_receipt.receipt_hash
                ),
                "coordinate_hashes": self.coordinate_hashes,
                "geometry_receipt_hash": receipt.receipt_hash,
                "signed_work_operator_receipt_hash": signed_work_operator.receipt_hash,
                "one_live_floor_at_a_time": True,
            },
            prefix=_DEVICE_FLOOR_FAMILY_PREFIX_V1,
        )

    @property
    def family_hash(self) -> str:
        return self._family_hash

    @property
    def exhausted(self) -> bool:
        return self._next_index >= len(self._coordinates)

    def issue_next(
        self,
        *,
        derive_endpoint_plan: bool = True,
    ) -> V1SignedWorkResourceCoordinateCUDADeviceBindingV1:
        if self._closed:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.HANDLE_EVICTED,
                "device coordinate family is closed",
            )
        if self._active_binding is not None:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.ACTIVE_LEASE,
                "close the current device floor before issuing the next",
            )
        if self.exhausted:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.NO_INTEGER_ENDPOINT,
                "device coordinate family is exhausted",
            )
        source = self._phase_a_problem
        source.assert_structural_identity()
        receipt = self._geometry_handle.receipt
        coordinate = self._coordinates[self._next_index]
        problem = V1SignedWorkLeastWorkCUDADeviceProblemV2(
            problem_id=(
                f"{source.source_registration_hash}-resource-coordinate-"
                f"{coordinate.ordinal}"
            ),
            geometry_handle=self._geometry_handle,
            signed_work_operator=self._signed_work_operator,
            resource_floor=coordinate.total_resource_floor,
            quadratic_budget=source.quadratic_budget,
            block_capacities=source.block_capacities,
            source_registration_hash=source.source_registration_hash,
            registered_d_coordinate_hash=source.registered_d_coordinate_hash,
            structural_source_receipt_hash=source.structural_source_receipt_hash,
            source_bundle_authoritative=True,
            expected_geometry_receipt_hash=receipt.receipt_hash,
            expected_session_design_hash=receipt.session_design_hash,
            expected_row_free_finalization_receipt_hash=(
                receipt.row_free_finalization_receipt_hash
            ),
            expected_materialization_rank_receipt_hash=(
                receipt.materialization_rank_receipt_hash
            ),
            expected_registered_proposal_fd_rank=receipt.registered_proposal_fd_rank,
            expected_route_axis_hash=source.route_axis_hash,
            resource_tensor=source.resource,
            eligible_mask_tensor=source.eligible_mask,
            required_hard_action_tensor=source.required_hard_action,
            resource_tensor_identity_hash=source.resource_tensor_identity_hash,
            eligible_mask_identity_hash=source.eligible_mask_identity_hash,
            required_mask_identity_hash=source.required_mask_identity_hash,
        )
        try:
            endpoint_plan = (
                derive_v1_signed_work_endpoint_plan_from_resource_coordinate(
                    problem=source,
                    coordinate=coordinate,
                )
                if derive_endpoint_plan
                else None
            )
            binding = V1SignedWorkResourceCoordinateCUDADeviceBindingV1(
                _issuance_token=_DEVICE_PHASE_A_ISSUANCE_TOKEN,
                owner=self,
                resource_coordinate=coordinate,
                least_work_problem=problem,
                endpoint_plan=endpoint_plan,
            )
        except BaseException:
            problem.close()
            raise
        self._active_binding = binding
        self._next_index += 1
        return binding

    def _release_binding(
        self,
        binding: V1SignedWorkResourceCoordinateCUDADeviceBindingV1,
    ) -> None:
        if binding is not self._active_binding:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                "device coordinate binding does not belong to the live family",
            )
        self._active_binding = None

    def close(self) -> None:
        if self._closed:
            return
        if self._active_binding is not None:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.ACTIVE_LEASE,
                "cannot close device coordinate family with a live floor",
            )
        self._closed = True

    def __reduce__(self) -> NoReturn:
        raise TypeError("live device coordinate families are nonserializable")

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("live device coordinate families are nonserializable")


def bind_v1_signed_work_device_problem_family_to_resource_frontier_v1(
    *,
    phase_a_problem: V1SignedWorkResourceFrontierCUDADeviceProblemV1,
    phase_a_execution: V1SignedWorkResourceFrontierCUDADeviceExecutionBundleV1,
    sequential_boundary_receipt: V1SignedWorkResourceFrontierCUDASequentialBoundaryReceiptV2,
    geometry_handle: V1SampledDeviceResidentDKKTGeometryDeviceHandle,
    signed_work_operator: V1SignedWorkOperatorCUDA,
) -> V1SignedWorkResourceCoordinateCUDADeviceFamilyV1:
    """Open the one-at-a-time Phase-B floor issuer after Phase-A eviction."""

    return V1SignedWorkResourceCoordinateCUDADeviceFamilyV1(
        _issuance_token=_DEVICE_PHASE_A_ISSUANCE_TOKEN,
        phase_a_problem=phase_a_problem,
        phase_a_execution=phase_a_execution,
        sequential_boundary_receipt=sequential_boundary_receipt,
        geometry_handle=geometry_handle,
        signed_work_operator=signed_work_operator,
    )


def bind_v1_signed_work_least_work_problem_to_resource_coordinate(
    *,
    base_problem: V1SignedWorkLeastWorkCUDAProblem,
    coordinate: V1SignedWorkEndogenousResourceCoordinate,
    derive_endpoint_plan: bool = True,
) -> V1SignedWorkResourceCoordinateLeastWorkBinding:
    """Create the fixed-resource signed-work problem for one coordinate."""

    if type(base_problem) is not V1SignedWorkLeastWorkCUDAProblem:
        raise TypeError("coordinate binding requires exact base least-work problem")
    if type(coordinate) is not V1SignedWorkEndogenousResourceCoordinate:
        raise TypeError("coordinate binding requires exact resource coordinate")
    source = base_problem.source_problem
    required = (
        np.zeros(source.route_count, dtype=np.bool_)
        if base_problem.required_hard_action is None
        else np.asarray(base_problem.required_hard_action, dtype=np.bool_).reshape(-1)
    )
    baseline = float(np.dot(source.resource, required.astype(np.float64)))
    guard = 1024.0 * _EPS * max(
        1.0,
        abs(baseline),
        abs(coordinate.baseline_resource),
        abs(coordinate.total_resource_floor),
    )
    if abs(baseline - coordinate.baseline_resource) > guard:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
            "resource coordinate baseline differs from the required hard rung",
        )
    if coordinate.total_resource_floor <= baseline + guard:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.NO_INTEGER_ENDPOINT,
            "resource coordinate does not advance beyond the required hard rung",
        )
    least_work_problem = V1SignedWorkLeastWorkCUDAProblem(
        source_problem=source,
        signed_work_operator=base_problem.signed_work_operator,
        resource_floor=coordinate.total_resource_floor,
        source_registration_hash=base_problem.source_registration_hash,
        registered_d_coordinate_hash=base_problem.registered_d_coordinate_hash,
        source_bundle_authoritative=base_problem.source_bundle_authoritative,
        required_hard_action=base_problem.required_hard_action,
    )
    endpoint_plan = (
        derive_v1_signed_work_endpoint_plan_from_resource_coordinate(
            problem=least_work_problem,
            coordinate=coordinate,
        )
        if derive_endpoint_plan
        else None
    )
    return V1SignedWorkResourceCoordinateLeastWorkBinding(
        least_work_problem_hash=least_work_problem.problem_hash,
        source_registration_hash=least_work_problem.source_registration_hash,
        registered_d_coordinate_hash=least_work_problem.registered_d_coordinate_hash,
        resource_coordinate_hash=coordinate.coordinate_hash,
        least_work_problem=least_work_problem,
        endpoint_plan=endpoint_plan,
        coordinate_total_resource_floor=coordinate.total_resource_floor,
        coordinate_incremental_resource=coordinate.incremental_resource,
    )


def bind_v1_signed_work_least_work_family_to_resource_frontier(
    *,
    base_problem: V1SignedWorkLeastWorkCUDAProblem,
    frontier_result: V1SignedWorkResourceFrontierResult,
    derive_endpoint_plans: bool = True,
) -> V1SignedWorkResourceCoordinateLeastWorkBindingFamily:
    """Bind every certified frontier coordinate to its own Phase-B problem."""

    if type(base_problem) is not V1SignedWorkLeastWorkCUDAProblem:
        raise TypeError("frontier binding requires exact base least-work problem")
    if type(frontier_result) is not V1SignedWorkResourceFrontierResult:
        raise TypeError("frontier binding requires exact resource-frontier result")
    base_problem.assert_integrity()
    coordinates = _certified_coordinates(frontier_result)
    bindings = tuple(
        bind_v1_signed_work_least_work_problem_to_resource_coordinate(
            base_problem=base_problem,
            coordinate=coordinate,
            derive_endpoint_plan=derive_endpoint_plans,
        )
        for coordinate in coordinates
    )
    return V1SignedWorkResourceCoordinateLeastWorkBindingFamily(
        frontier_result_hash=frontier_result.result_hash,
        base_least_work_problem_hash=base_problem.problem_hash,
        source_registration_hash=base_problem.source_registration_hash,
        registered_d_coordinate_hash=base_problem.registered_d_coordinate_hash,
        coordinate_bindings=bindings,
    )


def solve_v1_signed_work_resource_frontier_cuda_h200_engineering_qualification(
    problem: V1SignedWorkResourceFrontierCUDAProblem,
    policy: V1SignedWorkResourceFrontierCUDAPolicy,
    *,
    capability: V1SignedWorkResourceFrontierCUDAH200EngineeringCapability,
    device_total_memory_bytes: int | None = None,
    current_reserved_bytes: int | None = None,
) -> V1SignedWorkResourceFrontierCUDAExecutionBundle:
    """Execute only the exact synthetic benchmark that qualifies this source."""

    if type(capability) is not V1SignedWorkResourceFrontierCUDAH200EngineeringCapability:
        raise TypeError("engineering solve requires an exact issued capability")
    capability.assert_integrity()
    if type(policy) is not V1SignedWorkResourceFrontierCUDAPolicy:
        raise TypeError("engineering solve requires an exact policy")
    if (
        capability.solver_implementation_sha256
        != current_v1_signed_work_resource_frontier_cuda_solver_implementation_sha256()
        or capability.qualified_policy_semantics_hash
        != _qualification_policy_semantics_hash(policy)
        or policy.route_scale_cuda_qualification is not None
        or policy.route_scale_cuda_solver_qualified
    ):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
            "engineering capability does not bind this unqualified solver/policy",
        )
    _assert_exact_h200_engineering_problem(problem, capability=capability)
    if torch is None or not torch.cuda.is_available():
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.CUDA_REQUIRED,
            "H200 engineering qualification requires CUDA",
        )
    if torch.cuda.device_count() != 1:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.CUDA_REQUIRED,
            "H200 engineering qualification requires one visible device",
        )
    live_device = torch.device(policy.torch_device)
    live_name = str(torch.cuda.get_device_name(live_device))
    live_capability = tuple(
        int(value) for value in torch.cuda.get_device_capability(live_device)
    )
    if "H200" not in live_name.upper() or live_capability != (9, 0):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.CUDA_REQUIRED,
            "engineering qualification requires one H200 compute-capability 9.0",
        )
    if (
        not torch.are_deterministic_algorithms_enabled()
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        not in _QUALIFIED_CUBLAS_WORKSPACE_CONFIGS
    ):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.CUDA_REQUIRED,
            "engineering qualification requires strict deterministic CUDA",
        )
    forecast = estimate_v1_signed_work_resource_frontier_cuda_hbm(
        problem,
        policy,
        device_total_memory_bytes=device_total_memory_bytes,
        current_reserved_bytes=current_reserved_bytes,
    )
    if not forecast.admitted:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.HBM_FORECAST_EXCEEDED,
            "engineering qualification exceeds the registered HBM ceiling",
        )
    tensor_solve = _solve_resource_frontier_tensor(
        problem=problem,
        policy=policy,
        device=live_device,
    )
    frontier_result = _frontier_tensor_solve_result(tensor_solve)
    result = V1SignedWorkResourceFrontierCUDAResult(
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        forecast_hash=forecast.forecast_hash,
        reference_result=frontier_result,
        endpoint_plan=None,
        coordinate_hashes_bound_to_endpoint_plan=(),
        production_cuda_execution=False,
        private_cpu_reference_execution=False,
        engineering_h200_qualification_execution=True,
        engineering_capability_hash=capability.capability_hash,
    )
    return V1SignedWorkResourceFrontierCUDAExecutionBundle(
        result=result,
        forecast=forecast,
        theorem_state=_frontier_tensor_solve_theorem_state(
            tensor_solve,
            problem=problem,
            result=result,
            forecast=forecast,
        ),
    )


def solve_v1_signed_work_resource_frontier_device_cuda_v1(
    problem: V1SignedWorkResourceFrontierCUDADeviceProblemV1,
    policy: V1SignedWorkResourceFrontierCUDAPolicy | None = None,
    *,
    derive_endpoint_plan: bool = True,
    device_total_memory_bytes: int | None = None,
    current_reserved_bytes: int | None = None,
) -> V1SignedWorkResourceFrontierCUDADeviceExecutionBundleV1:
    """Solve floor-free Phase A directly on borrowed device geometry."""

    if type(problem) is not V1SignedWorkResourceFrontierCUDADeviceProblemV1:
        raise TypeError("device Phase-A solve requires the exact live problem")
    problem.assert_integrity()
    policy = V1SignedWorkResourceFrontierCUDAPolicy() if policy is None else policy
    if type(policy) is not V1SignedWorkResourceFrontierCUDAPolicy:
        raise TypeError("device Phase-A solve requires exact policy")
    forecast = estimate_v1_signed_work_resource_frontier_device_cuda_hbm_v1(
        problem,
        policy,
        device_total_memory_bytes=device_total_memory_bytes,
        current_reserved_bytes=current_reserved_bytes,
    )
    if not forecast.admitted:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.HBM_FORECAST_EXCEEDED,
            "device Phase-A forecast exceeds the registered HBM ceiling",
        )
    production = policy.production_cuda_required
    if production:
        if (
            torch is None
            or not torch.cuda.is_available()
            or not str(problem.device).startswith("cuda")
        ):
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.CUDA_REQUIRED,
                "production device Phase-A execution requires CUDA",
            )
        if (
            policy.require_deterministic_algorithms
            and not torch.are_deterministic_algorithms_enabled()
        ):
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.CUDA_REQUIRED,
                "production device Phase A requires deterministic torch algorithms",
            )
        live_capability = ".".join(
            str(int(value))
            for value in torch.cuda.get_device_capability(problem.device)
        )
        # The device sibling changes source custody only.  It deliberately
        # inherits, rather than bypasses, the exact H200 solver qualification.
        validate_v1_signed_work_resource_frontier_cuda_h200_qualification(
            route_count=problem.route_count,
            response_rank=problem.rank,
            policy=policy,
            device_name=str(torch.cuda.get_device_name(problem.device)),
            device_capability=live_capability,
            torch_version=str(torch.__version__),
            cuda_version=str(torch.version.cuda),
            cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        )
    else:
        if not policy.allow_private_cpu_reference:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.CUDA_REQUIRED,
                "private device Phase-A reference was not explicitly enabled",
            )
        if problem.route_count > policy.private_reference_route_limit:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.PRIVATE_REFERENCE_LIMIT_EXCEEDED,
                "private device Phase-A reference exceeds its route limit",
            )
    tensor_solve = _solve_resource_frontier_tensor(
        problem=problem,
        policy=policy,
        device=problem.device,
    )
    frontier_result = _frontier_tensor_solve_result(tensor_solve)
    endpoint_plan = None
    coordinate_hashes: tuple[str, ...] = ()
    if derive_endpoint_plan and (
        frontier_result.union_status
        is V1SignedWorkResourceUnionStatus.CERTIFIED_COORDINATES
    ):
        endpoint_plan, coordinate_hashes = (
            derive_v1_signed_work_endpoint_plan_from_resource_frontier(
                problem=problem,
                frontier_result=frontier_result,
            )
        )
    result = V1SignedWorkResourceFrontierCUDAResult(
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        forecast_hash=forecast.forecast_hash,
        reference_result=frontier_result,
        endpoint_plan=endpoint_plan,
        coordinate_hashes_bound_to_endpoint_plan=coordinate_hashes,
        production_cuda_execution=production,
        private_cpu_reference_execution=not production,
    )
    theorem_state = _frontier_tensor_solve_theorem_state(
        tensor_solve,
        problem=problem,
        result=result,
        forecast=forecast,
    )
    return V1SignedWorkResourceFrontierCUDADeviceExecutionBundleV1(
        result=result,
        forecast=forecast,
        theorem_state=theorem_state,
    )


def solve_v1_signed_work_resource_frontier_cuda(
    problem: V1SignedWorkResourceFrontierCUDAProblem
    | V1SignedWorkResourceFrontierCUDADeviceProblemV1
    | V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkResourceFrontierCUDAPolicy | None = None,
    *,
    derive_endpoint_plan: bool = True,
    device_total_memory_bytes: int | None = None,
    current_reserved_bytes: int | None = None,
) -> V1SignedWorkResourceFrontierCUDAExecutionBundle | V1SignedWorkResourceFrontierCUDADeviceExecutionBundleV1:
    """Issue a CUDA-facing Phase-A frontier receipt.

    Production uses the CUDA tensor resource-frontier solver.  The old SciPy
    small-domain reference remains the default private CPU compatibility path;
    tests may explicitly request the tensor private path without changing
    production authority.
    """

    if type(problem) is V1SignedWorkResourceFrontierCUDADeviceProblemV1:
        return solve_v1_signed_work_resource_frontier_device_cuda_v1(
            problem,
            policy,
            derive_endpoint_plan=derive_endpoint_plan,
            device_total_memory_bytes=device_total_memory_bytes,
            current_reserved_bytes=current_reserved_bytes,
        )
    phase_a_problem = _canonical_phase_a_problem(problem)
    policy = V1SignedWorkResourceFrontierCUDAPolicy() if policy is None else policy
    if type(policy) is not V1SignedWorkResourceFrontierCUDAPolicy:
        raise TypeError("resource-frontier CUDA solve requires exact policy")
    forecast = estimate_v1_signed_work_resource_frontier_cuda_hbm(
        phase_a_problem,
        policy,
        device_total_memory_bytes=device_total_memory_bytes,
        current_reserved_bytes=current_reserved_bytes,
    )
    if not forecast.admitted:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.HBM_FORECAST_EXCEEDED,
            "resource-frontier forecast exceeds the registered HBM ceiling",
        )
    if policy.production_cuda_required:
        if torch is None or not torch.cuda.is_available() or not policy.torch_device.startswith(
            "cuda"
        ):
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.CUDA_REQUIRED,
                "production resource-frontier execution requires CUDA",
            )
        if policy.require_deterministic_algorithms and not torch.are_deterministic_algorithms_enabled():
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.CUDA_REQUIRED,
                "production resource-frontier execution requires deterministic torch algorithms",
            )
        live_device = torch.device(policy.torch_device)
        live_capability = ".".join(
            str(int(value))
            for value in torch.cuda.get_device_capability(live_device)
        )
        validate_v1_signed_work_resource_frontier_cuda_h200_qualification(
            route_count=phase_a_problem.route_count,
            response_rank=phase_a_problem.rank,
            policy=policy,
            device_name=str(torch.cuda.get_device_name(live_device)),
            device_capability=live_capability,
            torch_version=str(torch.__version__),
            cuda_version=str(torch.version.cuda),
            cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        )
        tensor_solve = _solve_resource_frontier_tensor(
            problem=phase_a_problem,
            policy=policy,
            device=live_device,
        )
        frontier_result = _frontier_tensor_solve_result(tensor_solve)
        endpoint_plan = None
        coordinate_hashes: tuple[str, ...] = ()
        if derive_endpoint_plan and (
            frontier_result.union_status
            is V1SignedWorkResourceUnionStatus.CERTIFIED_COORDINATES
        ):
            endpoint_plan, coordinate_hashes = (
                derive_v1_signed_work_endpoint_plan_from_resource_frontier(
                    problem=phase_a_problem,
                    frontier_result=frontier_result,
                )
            )
        result = V1SignedWorkResourceFrontierCUDAResult(
            problem_hash=phase_a_problem.problem_hash,
            policy_hash=policy.policy_hash,
            forecast_hash=forecast.forecast_hash,
            reference_result=frontier_result,
            endpoint_plan=endpoint_plan,
            coordinate_hashes_bound_to_endpoint_plan=coordinate_hashes,
            production_cuda_execution=True,
            private_cpu_reference_execution=False,
        )
        return V1SignedWorkResourceFrontierCUDAExecutionBundle(
            result=result,
            forecast=forecast,
            theorem_state=_frontier_tensor_solve_theorem_state(
                tensor_solve,
                problem=phase_a_problem,
                result=result,
                forecast=forecast,
            ),
        )
    if not policy.allow_private_cpu_reference:
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.CUDA_REQUIRED,
            "private CPU resource-frontier reference was not explicitly enabled",
        )
    if (
        phase_a_problem.route_count
        > policy.private_reference_route_limit
    ):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.PRIVATE_REFERENCE_LIMIT_EXCEEDED,
            "private resource-frontier reference exceeds its route limit",
        )
    if policy.use_tensor_private_reference:
        if torch is None:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.CUDA_REQUIRED,
                "PyTorch is required for tensor private resource-frontier execution",
            )
        tensor_solve = _solve_resource_frontier_tensor(
            problem=phase_a_problem,
            policy=policy,
            device=torch.device("cpu"),
        )
        reference_result = _frontier_tensor_solve_result(tensor_solve)
    else:
        tensor_solve = None
        reference_policy = V1SignedWorkResourceFrontierPolicy(
            small_domain_route_limit=policy.private_reference_route_limit,
        )
        reference_problem = _reference_problem_from_cuda_problem(phase_a_problem)
        reference_result = solve_v1_signed_work_resource_frontier_reference(
            reference_problem,
            reference_policy,
        )
    endpoint_plan = None
    coordinate_hashes: tuple[str, ...] = ()
    if derive_endpoint_plan and (
        reference_result.union_status
        is V1SignedWorkResourceUnionStatus.CERTIFIED_COORDINATES
    ):
        endpoint_plan, coordinate_hashes = (
            derive_v1_signed_work_endpoint_plan_from_resource_frontier(
                problem=phase_a_problem,
                frontier_result=reference_result,
            )
        )
    result = V1SignedWorkResourceFrontierCUDAResult(
        problem_hash=phase_a_problem.problem_hash,
        policy_hash=policy.policy_hash,
        forecast_hash=forecast.forecast_hash,
        reference_result=reference_result,
        endpoint_plan=endpoint_plan,
        coordinate_hashes_bound_to_endpoint_plan=coordinate_hashes,
        production_cuda_execution=False,
        private_cpu_reference_execution=True,
    )
    return V1SignedWorkResourceFrontierCUDAExecutionBundle(
        result=result,
        forecast=forecast,
        theorem_state=(
            None
            if tensor_solve is None
            else _frontier_tensor_solve_theorem_state(
                tensor_solve,
                problem=phase_a_problem,
                result=result,
                forecast=forecast,
            )
        ),
    )


# ---------------------------------------------------------------------------
# Sequential boundary V2: margin-state replay
# ---------------------------------------------------------------------------

SIGNED_WORK_SEQUENTIAL_BOUNDARY_MARGIN_STATE_REPLAY_SCHEMA_V2: Final = (
    "partition_free_v1_signed_work_sequential_boundary_margin_state_replay_v2"
)


@dataclass(frozen=True, slots=True)
class V1SignedWorkSequentialBoundaryMarginStateReplayReceiptV2:
    """Proof that a retained decision-margin state is the boundary's parent.

    Every hash the boundary receipt derived from the margin state is
    recomputed here exactly as the issuer computed it and compared.  The
    dual-margin identities need the live Phase-A theorem handle and are not
    replayable from the state alone; that absence is typed, not filled.
    """

    sequential_boundary_receipt_hash: str
    decision_margin_state_hash: str
    phase_a_result_hash: str
    route_axis_hash: str
    route_count: int
    resource_cell_binding_count: int
    live_branch_resource_reduction_count: int
    resource_cell_boundary_count: int
    branch_support_boundary_count: int
    scalar_host_transfer_bytes: int
    dual_margin_identities_replayed: bool = False
    dual_margin_identity_absence: str = "requires_live_phase_a_theorem_handle"
    route_scale_d2h_bytes: int = 0
    c_stream_authority: bool = False
    fqr_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    standalone_decision_stability_authority: bool = False
    schema: str = SIGNED_WORK_SEQUENTIAL_BOUNDARY_MARGIN_STATE_REPLAY_SCHEMA_V2

    def __post_init__(self) -> None:
        for name in (
            "sequential_boundary_receipt_hash",
            "decision_margin_state_hash",
            "phase_a_result_hash",
            "route_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "route_count",
            "resource_cell_binding_count",
            "live_branch_resource_reduction_count",
            "resource_cell_boundary_count",
            "branch_support_boundary_count",
            "scalar_host_transfer_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative int")
        if self.route_count <= 0 or self.resource_cell_binding_count <= 0:
            raise ValueError("margin-state replay requires routes and resource cells")
        if int(self.route_scale_d2h_bytes) != 0:
            raise ValueError("margin-state replay transfers no route-scale rows")
        if (
            type(self.dual_margin_identities_replayed) is not bool
            or self.dual_margin_identities_replayed
        ):
            raise ValueError(
                "dual-margin identities are never replayed from the state alone"
            )
        flags = (
            self.c_stream_authority,
            self.fqr_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
            self.standalone_decision_stability_authority,
        )
        if any(type(value) is not bool for value in flags) or any(flags):
            raise ValueError("margin-state replay carries no authority")
        if self.schema != SIGNED_WORK_SEQUENTIAL_BOUNDARY_MARGIN_STATE_REPLAY_SCHEMA_V2:
            raise ValueError("margin-state replay semantics changed")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_STATE_PREFIX + "margin-state-replay-")


def replay_v1_signed_work_sequential_boundary_margin_state_v2(
    *,
    receipt: V1SignedWorkResourceFrontierCUDASequentialBoundaryReceiptV2,
    decision_margin_state: V1SignedWorkResourceFrontierCUDADecisionMarginState,
) -> V1SignedWorkSequentialBoundaryMarginStateReplayReceiptV2:
    """Recompute every state-derived hash on the boundary receipt and compare."""

    if type(receipt) is not V1SignedWorkResourceFrontierCUDASequentialBoundaryReceiptV2:
        raise TypeError("margin-state replay requires the exact sequential boundary receipt")
    if type(decision_margin_state) is not V1SignedWorkResourceFrontierCUDADecisionMarginState:
        raise TypeError("margin-state replay requires the exact decision-margin state")
    state = decision_margin_state
    if (
        state.state_hash != receipt.decision_margin_state_hash
        or state.result_hash != receipt.phase_a_result_hash
        or state.handle_hash != receipt.phase_a_handle_hash
        or state.route_axis_hash != receipt.route_axis_hash
        or state.route_count != receipt.route_count
    ):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
            "decision-margin state is not the sequential boundary's parent",
        )
    recomputed = {
        "resource_cell_binding_hashes": tuple(state.resource_cell_binding_hashes),
        "live_branch_resource_reduction_hashes": tuple(
            content_hash(value, prefix=_STATE_PREFIX + "live-resource-reduction-")
            for value in state.live_branch_resource_reductions
        ),
        "resource_cell_boundary_hashes": tuple(
            content_hash(value, prefix=_STATE_PREFIX + "resource-cell-boundary-")
            for value in state.resource_cell_boundaries
        ),
        "branch_support_boundary_hashes": tuple(
            content_hash(value, prefix=_STATE_PREFIX + "branch-support-boundary-")
            for value in state.branch_support_boundaries
        ),
    }
    for name, values in recomputed.items():
        if tuple(getattr(receipt, name)) != values:
            _refuse(
                V1SignedWorkResourceFrontierCUDARefusalCode.STALE_SOURCE,
                f"sequential boundary {name} do not replay from the margin state",
            )
    if int(receipt.scalar_host_transfer_bytes) < int(state.scalar_host_transfer_bytes):
        _refuse(
            V1SignedWorkResourceFrontierCUDARefusalCode.AUTHORITY_VIOLATION,
            "sequential boundary under-declares the margin state's scalar transfer",
        )
    return V1SignedWorkSequentialBoundaryMarginStateReplayReceiptV2(
        sequential_boundary_receipt_hash=receipt.receipt_hash,
        decision_margin_state_hash=state.state_hash,
        phase_a_result_hash=receipt.phase_a_result_hash,
        route_axis_hash=receipt.route_axis_hash,
        route_count=int(receipt.route_count),
        resource_cell_binding_count=len(recomputed["resource_cell_binding_hashes"]),
        live_branch_resource_reduction_count=len(
            recomputed["live_branch_resource_reduction_hashes"]
        ),
        resource_cell_boundary_count=len(recomputed["resource_cell_boundary_hashes"]),
        branch_support_boundary_count=len(recomputed["branch_support_boundary_hashes"]),
        scalar_host_transfer_bytes=int(state.scalar_host_transfer_bytes),
    )


__all__ = [
    "SIGNED_WORK_SEQUENTIAL_BOUNDARY_MARGIN_STATE_REPLAY_SCHEMA_V2",
    "V1SignedWorkSequentialBoundaryMarginStateReplayReceiptV2",
    "replay_v1_signed_work_sequential_boundary_margin_state_v2",
    "SIGNED_WORK_RESOURCE_FRONTIER_CUDA_ALGORITHM_ID",
    "SIGNED_WORK_RESOURCE_FRONTIER_CUDA_AUTHORITY",
    "SIGNED_WORK_RESOURCE_FRONTIER_CUDA_COMPLETE_PREFIX_RULE_ID",
    "SIGNED_WORK_RESOURCE_FRONTIER_CUDA_DECISION_MARGIN_POLICY",
    "SIGNED_WORK_RESOURCE_FRONTIER_CUDA_DECISION_MARGIN_STATE_SCHEMA",
    "SIGNED_WORK_RESOURCE_FRONTIER_CUDA_DEVICE_EXECUTION_SCHEMA_V1",
    "SIGNED_WORK_RESOURCE_FRONTIER_CUDA_DEVICE_FLOOR_FAMILY_SCHEMA_V1",
    "SIGNED_WORK_RESOURCE_FRONTIER_CUDA_DEVICE_PROBLEM_SCHEMA_V1",
    "SIGNED_WORK_RESOURCE_FRONTIER_CUDA_ENDPOINT_RULE_ID",
    "SIGNED_WORK_RESOURCE_FRONTIER_CUDA_FULL_ROUTE_COUNT",
    "SIGNED_WORK_RESOURCE_FRONTIER_CUDA_H200_BENCHMARK_SCHEMA",
    "SIGNED_WORK_RESOURCE_FRONTIER_CUDA_H200_DEVICE_CLASS",
    "SIGNED_WORK_RESOURCE_FRONTIER_CUDA_QUALIFIED_DEVICE_CLASSES",
    "qualified_device_class_for_name",
    "runtime_qualified_device_class",
    "SIGNED_WORK_RESOURCE_FRONTIER_CUDA_H200_ENGINEERING_CAPABILITY_SCHEMA",
    "SIGNED_WORK_RESOURCE_FRONTIER_CUDA_H200_QUALIFICATION_SCHEMA",
    "SIGNED_WORK_RESOURCE_FRONTIER_CUDA_NO_F_OPERATOR_ID",
    "SIGNED_WORK_RESOURCE_FRONTIER_CUDA_NUMERICAL_DTYPE",
    "SIGNED_WORK_RESOURCE_FRONTIER_CUDA_PROBLEM_SCHEMA",
    "SIGNED_WORK_RESOURCE_FRONTIER_CUDA_SEQUENTIAL_BOUNDARY_SCHEMA_V2",
    "SIGNED_WORK_RESOURCE_FRONTIER_CUDA_QUALIFIED_BRANCHES",
    "SIGNED_WORK_RESOURCE_FRONTIER_CUDA_QUALIFIED_RANKS",
    "V1SignedWorkResourceFrontierCUDAExecutionBundle",
    "V1SignedWorkResourceFrontierCUDADeviceExecutionBundleV1",
    "V1SignedWorkResourceFrontierCUDADeviceHBMForecastV1",
    "V1SignedWorkResourceFrontierCUDADeviceProblemV1",
    "V1SignedWorkResourceFrontierCUDABranchSupportBoundary",
    "V1SignedWorkResourceFrontierCUDADecisionMarginLease",
    "V1SignedWorkResourceFrontierCUDADecisionMarginState",
    "V1SignedWorkResourceFrontierCUDAH200EngineeringCapability",
    "V1SignedWorkResourceFrontierCUDAH200QualificationReceipt",
    "V1SignedWorkResourceFrontierCUDAHBMForecast",
    "V1SignedWorkResourceFrontierCUDALiveBranchResourceReduction",
    "V1SignedWorkResourceFrontierCUDAResourceCellBoundary",
    "V1SignedWorkResourceFrontierCUDATheoremStateHandle",
    "V1SignedWorkResourceFrontierCUDATheoremStateLease",
    "V1SignedWorkResourceFrontierCUDAPolicy",
    "V1SignedWorkResourceFrontierCUDAProblem",
    "V1SignedWorkResourceFrontierCUDARefusal",
    "V1SignedWorkResourceFrontierCUDARefusalCode",
    "V1SignedWorkResourceFrontierCUDAResult",
    "V1SignedWorkResourceFrontierCUDASequentialBoundaryReceiptV2",
    "V1SignedWorkResourceCoordinateLeastWorkBinding",
    "V1SignedWorkResourceCoordinateLeastWorkBindingFamily",
    "V1SignedWorkResourceCoordinateCUDADeviceBindingV1",
    "V1SignedWorkResourceCoordinateCUDADeviceFamilyV1",
    "adapt_v1_signed_work_device_problem_v2_to_resource_frontier_cuda_v1",
    "adapt_v1_signed_work_least_work_problem_to_resource_frontier_cuda",
    "bind_v1_signed_work_least_work_family_to_resource_frontier",
    "bind_v1_signed_work_least_work_problem_to_resource_coordinate",
    "bind_v1_signed_work_device_problem_family_to_resource_frontier_v1",
    "build_v1_signed_work_resource_frontier_cuda_problem_from_relaxed_kkt",
    "current_v1_signed_work_resource_frontier_cuda_solver_implementation_sha256",
    "derive_v1_signed_work_endpoint_plan_from_resource_coordinate",
    "derive_v1_signed_work_complete_prefix_endpoint_plan_from_resource_frontier",
    "derive_v1_signed_work_endpoint_plan_from_resource_frontier",
    "estimate_v1_signed_work_resource_frontier_cuda_hbm",
    "estimate_v1_signed_work_resource_frontier_device_cuda_hbm_v1",
    "issue_v1_signed_work_resource_frontier_cuda_device_problem_v1",
    "issue_v1_signed_work_resource_frontier_cuda_h200_qualification",
    "issue_v1_signed_work_resource_frontier_cuda_h200_engineering_capability",
    "solve_v1_signed_work_resource_frontier_cuda",
    "solve_v1_signed_work_resource_frontier_device_cuda_v1",
    "solve_v1_signed_work_resource_frontier_cuda_h200_engineering_qualification",
    "validate_v1_signed_work_resource_frontier_cuda_h200_qualification",
    "v1_signed_work_resource_frontier_cuda_qualification_policy_semantics_hash",
]
