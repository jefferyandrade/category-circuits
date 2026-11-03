"""CUDA execution owner for fixed-direction literal-J projections.

This module is the bounded device sibling of
``modern_v1_literal_j_projection_inference``.  It owns only execution:
directions and support bounds are frozen on CUDA, literal-J rows are consumed in
registered prefix order, and the accumulated sufficient statistics are

``M_J z = E[j * (j^T z)]`` and ``Q(z) = E[(j^T z)^2]``.

Raw rows are never retained.  The durable output is the existing array-free
``V1LiteralJFixedDirectionProjectionReceipt`` plus CUDA forecast/telemetry.  No
face, KKT-sensitivity, C/F/Q/R, calibration, oracle, mask-commit, production, or
Pareto authority is granted here.
"""

from __future__ import annotations

from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
from numbers import Integral, Real
from threading import RLock
from typing import Any, Final, Literal, NoReturn, Sequence

from .contracts import content_hash
from .modern_v1_literal_j_projection_inference import (
    LITERAL_J_FIXED_DIRECTION_PROJECTION_ALGORITHM,
    LITERAL_J_FIXED_DIRECTION_PROJECTION_AUTHORITY_SCOPE,
    LITERAL_J_FIXED_DIRECTION_PROJECTION_EQUATION,
    LITERAL_J_FIXED_DIRECTION_PROJECTION_SCHEMA,
    V1LiteralJFixedDirectionProjectionReceipt,
    _axis_hash as _projection_axis_hash,
    _serfling_radius,
)


LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_DESIGN_SCHEMA: Final = (
    "partition_free_v1_literal_j_fixed_direction_projection_cuda_design_v1"
)
LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_HBM_FORECAST_SCHEMA: Final = (
    "partition_free_v1_literal_j_fixed_direction_projection_cuda_hbm_forecast_v1"
)
LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_TELEMETRY_SCHEMA: Final = (
    "partition_free_v1_literal_j_fixed_direction_projection_cuda_telemetry_v1"
)
LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_BUNDLE_SCHEMA: Final = (
    "partition_free_v1_literal_j_fixed_direction_projection_cuda_bundle_v1"
)
LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_ALGORITHM: Final = (
    "torch_cuda_route_tiled_fp64_fixed_direction_literal_J_projection_v1"
)
LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_HBM_CEILING_FRACTION: Final = 0.85
LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_MINIMUM_GUARD_BYTES: Final = (
    64 * 1024 * 1024
)
LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_GUARD_FRACTION: Final = 0.10
LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_NUMERICAL_ENCLOSURE_ID: Final = (
    "fp64_gamma_bound_added_to_simultaneous_projection_radii_v1"
)
LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_SMALL_DOMAIN_HASH_LIMIT: Final = 65_536
LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_OUTPUT_FINGERPRINT_SCALAR_COUNT: Final = 8
LITERAL_J_PRIVATE_PRIMAL_DIRECTION_DESCRIPTOR_SCHEMA: Final = (
    "partition_free_v1_literal_j_private_primal_direction_descriptor_v1"
)
LITERAL_J_PRIVATE_PRIMAL_DIRECTION_DESCRIPTOR_AXIS_SCHEMA: Final = (
    "partition_free_v1_literal_j_private_primal_direction_descriptor_axis_v1"
)

_DESIGN_PREFIX: Final = "partition-free-v1-literal-j-projection-cuda-design-"
_FORECAST_PREFIX: Final = "partition-free-v1-literal-j-projection-cuda-hbm-"
_TELEMETRY_PREFIX: Final = "partition-free-v1-literal-j-projection-cuda-telemetry-"
_BUNDLE_PREFIX: Final = "partition-free-v1-literal-j-projection-cuda-bundle-"
_ARRAY_PREFIX: Final = "partition-free-v1-literal-j-fixed-direction-array-"
_PRIVATE_PRIMAL_DESCRIPTOR_PREFIX: Final = (
    "partition-free-v1-literal-j-private-primal-direction-descriptor-"
)
_PRIVATE_PRIMAL_DESCRIPTOR_AXIS_PREFIX: Final = (
    "partition-free-v1-literal-j-private-primal-direction-axis-"
)
_STRUCTURAL_VECTOR_PREFIX: Final = (
    "partition-free-v1-literal-j-projection-cuda-structural-vector-"
)
_OUTPUT_FINGERPRINT_PREFIX: Final = (
    "partition-free-v1-literal-j-projection-cuda-output-fingerprint-"
)


class V1LiteralJFixedDirectionProjectionCUDARefusalCode(str, Enum):
    """Typed fail-closed CUDA projection outcomes."""

    INVALID_DESIGN = "invalid_design"
    CUDA_UNAVAILABLE = "cuda_unavailable"
    CPU_FALLBACK_FORBIDDEN = "cpu_fallback_forbidden"
    DETERMINISM_DISABLED = "determinism_disabled"
    HBM_FORECAST_EXCEEDED = "hbm_forecast_exceeded"
    MEASURED_HBM_CEILING_EXCEEDED = "measured_hbm_ceiling_exceeded"
    MEASURED_HBM_FORECAST_INVALIDATED = "measured_hbm_forecast_invalidated"
    STALE_IDENTITY = "stale_identity"
    ORDER_MISMATCH = "order_mismatch"
    DUPLICATE_IDENTITY = "duplicate_identity"
    INVALID_ROW = "invalid_row"
    EMPTY_PREFIX = "empty_prefix"
    FINALIZED = "finalized"
    NONFINITE_CUDA_SUMMARY = "nonfinite_cuda_summary"


class V1LiteralJFixedDirectionProjectionCUDARefusal(RuntimeError):
    """The CUDA fixed-direction projection path refused closed."""

    def __init__(
        self,
        code: V1LiteralJFixedDirectionProjectionCUDARefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1LiteralJFixedDirectionProjectionCUDARefusalCode:
            raise TypeError("CUDA projection refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1LiteralJFixedDirectionProjectionCUDARefusalCode,
    message: str,
) -> NoReturn:
    raise V1LiteralJFixedDirectionProjectionCUDARefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _identity_tuple(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of identities")
    try:
        result = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be finite: {error}") from error
    if not result:
        raise ValueError(f"{name} must be nonempty")
    for index, value in enumerate(result):
        _identity(value, name=f"{name}[{index}]")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate identities")
    return result


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be one exact integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be one exact integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be one real scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _positive_float(value: object, *, name: str) -> float:
    result = _finite_nonnegative(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        _refuse(
            V1LiteralJFixedDirectionProjectionCUDARefusalCode.CUDA_UNAVAILABLE,
            "torch is required for CUDA literal-J projection",
        )
    return torch


def _require_determinism(torch: Any) -> None:
    if not bool(torch.are_deterministic_algorithms_enabled()):
        _refuse(
            V1LiteralJFixedDirectionProjectionCUDARefusalCode.DETERMINISM_DISABLED,
            "torch deterministic algorithms must already be enabled",
        )
    warn_only = getattr(
        torch,
        "is_deterministic_algorithms_warn_only_enabled",
        lambda: False,
    )
    if bool(warn_only()):
        _refuse(
            V1LiteralJFixedDirectionProjectionCUDARefusalCode.DETERMINISM_DISABLED,
            "warn-only deterministic mode is not an execution guarantee",
        )


def _resolve_cuda_device(torch: Any, device: str) -> Any:
    resolved = torch.device(_identity(device, name="device"))
    if resolved.type != "cuda" or not bool(torch.cuda.is_available()):
        _refuse(
            V1LiteralJFixedDirectionProjectionCUDARefusalCode.CUDA_UNAVAILABLE,
            "literal-J projection CUDA execution requires an available CUDA device",
        )
    if resolved.index is None:
        resolved = torch.device("cuda", int(torch.cuda.current_device()))
    return resolved


def _require_cuda_tensor(
    torch: Any,
    value: object,
    *,
    name: str,
    device: Any | None,
    ndim: int,
    nonnegative: bool = False,
) -> Any:
    if not isinstance(value, torch.Tensor):
        _refuse(
            V1LiteralJFixedDirectionProjectionCUDARefusalCode.CPU_FALLBACK_FORBIDDEN,
            f"{name} must be a torch CUDA tensor; CPU/NumPy fallback is forbidden",
        )
    if value.device.type != "cuda":
        _refuse(
            V1LiteralJFixedDirectionProjectionCUDARefusalCode.CPU_FALLBACK_FORBIDDEN,
            f"{name} is not on CUDA",
        )
    if device is not None and value.device != device:
        _refuse(
            V1LiteralJFixedDirectionProjectionCUDARefusalCode.INVALID_DESIGN,
            f"{name} is on {value.device}, expected {device}",
        )
    if value.dtype is not torch.float64:
        _refuse(
            V1LiteralJFixedDirectionProjectionCUDARefusalCode.INVALID_DESIGN,
            f"{name} must have dtype torch.float64",
        )
    if value.ndim != ndim:
        _refuse(
            V1LiteralJFixedDirectionProjectionCUDARefusalCode.INVALID_DESIGN,
            f"{name} must have ndim={ndim}",
        )
    if not bool(torch.isfinite(value).all().item()):
        _refuse(
            V1LiteralJFixedDirectionProjectionCUDARefusalCode.INVALID_DESIGN,
            f"{name} must contain only finite values",
        )
    if nonnegative and bool((value < 0.0).any().item()):
        _refuse(
            V1LiteralJFixedDirectionProjectionCUDARefusalCode.INVALID_DESIGN,
            f"{name} must be nonnegative",
        )
    return value.contiguous()


def _fp64_cuda_vector_hash(
    torch: Any,
    value: Any,
    *,
    role: str,
    chunk_size: int,
) -> str:
    if value.ndim != 1 or value.dtype is not torch.float64 or value.device.type != "cuda":
        raise TypeError("CUDA vector hash requires one fp64 CUDA vector")
    chunk = _positive_int(chunk_size, name="hash chunk size")
    digest = hashlib.sha256()
    for start in range(0, int(value.numel()), chunk):
        piece = value[start : start + chunk].detach().to(
            device="cpu",
            dtype=torch.float64,
            copy=True,
        )
        array = piece.numpy()
        digest.update(memoryview(array).cast("B"))
        del piece, array
    return content_hash(
        {
            "schema": "literal_j_fixed_direction_fp64_array_bytes_v1",
            "role": role,
            "dtype": "<f8",
            "shape": (int(value.numel()),),
            "payload_sha256": digest.hexdigest(),
        },
        prefix=_ARRAY_PREFIX,
    )


def _cuda_direction_hashes(
    torch: Any,
    directions: Any,
    *,
    chunk_size: int,
) -> tuple[str, ...]:
    return tuple(
        _fp64_cuda_vector_hash(
            torch,
            directions[index],
            role="fixed_primal_direction",
            chunk_size=chunk_size,
        )
        for index in range(int(directions.shape[0]))
    )


def _cuda_contribution_bound_hashes(
    torch: Any,
    literal_j_abs_bound: Any,
    projection_abs_bounds: Any,
    *,
    chunk_size: int,
) -> tuple[str, ...]:
    hashes: list[str] = []
    for index in range(int(projection_abs_bounds.numel())):
        vector = literal_j_abs_bound * projection_abs_bounds[index]
        hashes.append(
            _fp64_cuda_vector_hash(
                torch,
                vector,
                role="literal_j_MJz_projection_point",
                chunk_size=chunk_size,
            )
        )
        del vector
    return tuple(hashes)


def _gamma(count: int) -> float:
    if count <= 0:
        return 0.0
    eps = 2.0**-53
    scale = float(count) * eps
    if scale >= 0.5:
        return float("inf")
    return scale / (1.0 - scale)


def _max_contribution_bound(
    torch: Any,
    literal_j_abs_bound: Any,
    projection_abs_bounds: Any,
) -> float:
    return float(
        (
            torch.max(literal_j_abs_bound)
            * torch.max(projection_abs_bounds)
        )
        .detach()
        .cpu()
        .item()
    )


def _structural_direction_hashes(
    *,
    route_axis_hash: str,
    route_count: int,
    direction_ids: tuple[str, ...],
    private_primal_direction_descriptor_axis_hash: str,
) -> tuple[str, ...]:
    return tuple(
        content_hash(
            {
                "schema": "literal_j_cuda_private_primal_direction_identity_v1",
                "route_axis_hash": route_axis_hash,
                "route_count": route_count,
                "direction_id": direction_id,
                "direction_index": index,
                "private_primal_direction_descriptor_axis_hash": (
                    private_primal_direction_descriptor_axis_hash
                ),
                "route_scale_d2h_array_hash": False,
            },
            prefix=_STRUCTURAL_VECTOR_PREFIX,
        )
        for index, direction_id in enumerate(direction_ids)
    )


def _structural_contribution_bound_hashes(
    *,
    route_axis_hash: str,
    route_count: int,
    direction_ids: tuple[str, ...],
    literal_j_abs_bound_source_hash: str,
    projection_abs_bound_source_hash: str,
) -> tuple[str, ...]:
    return tuple(
        content_hash(
            {
                "schema": "literal_j_cuda_contribution_bound_identity_v1",
                "route_axis_hash": route_axis_hash,
                "route_count": route_count,
                "direction_id": direction_id,
                "direction_index": index,
                "literal_j_abs_bound_source_hash": literal_j_abs_bound_source_hash,
                "projection_abs_bound_source_hash": projection_abs_bound_source_hash,
                "formula": "literal_j_abs_bound[j]*projection_abs_bound[k]",
                "route_scale_d2h_array_hash": False,
            },
            prefix=_STRUCTURAL_VECTOR_PREFIX,
        )
        for index, direction_id in enumerate(direction_ids)
    )


def _cuda_output_fingerprint_hashes(
    torch: Any,
    output: Any,
    *,
    route_axis_hash: str,
    direction_ids: tuple[str, ...],
    direction_hashes: tuple[str, ...],
    quadratic_values: tuple[float, ...],
) -> tuple[str, ...]:
    hashes: list[str] = []
    for index, direction_id in enumerate(direction_ids):
        vector = output[index]
        stats = torch.stack(
            (
                torch.sum(vector),
                torch.sum(torch.abs(vector)),
                torch.sum(vector * vector),
                torch.amin(vector),
                torch.amax(vector),
                torch.amax(torch.abs(vector)),
                vector[0],
                vector[-1],
            )
        )
        if int(stats.numel()) != (
            LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_OUTPUT_FINGERPRINT_SCALAR_COUNT
        ):
            raise AssertionError("literal-J CUDA fingerprint scalar count drifted")
        payload = tuple(float(item) for item in stats.detach().cpu().tolist())
        hashes.append(
            content_hash(
                {
                    "schema": "literal_j_cuda_MJz_output_compact_fingerprint_v1",
                    "route_axis_hash": route_axis_hash,
                    "route_count": int(vector.numel()),
                    "direction_id": direction_id,
                    "direction_index": index,
                    "direction_identity_hash": direction_hashes[index],
                    "quadratic_value": quadratic_values[index],
                    "fingerprint_scalars": payload,
                    "fingerprint_scalar_names": (
                        "sum",
                        "l1_sum",
                        "l2_square_sum",
                        "min",
                        "max",
                        "max_abs",
                        "first",
                        "last",
                    ),
                    "deterministic_device_reduction": True,
                    "route_scale_d2h_array_hash": False,
                },
                prefix=_OUTPUT_FINGERPRINT_PREFIX,
            )
        )
        del stats
    return tuple(hashes)


CanonicalPrivatePrimalBranch = Literal["spectral", "diagonal_l1"]


def _canonical_private_primal_branch(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    if value not in ("spectral", "diagonal_l1"):
        raise ValueError(f"{name} must be spectral or diagonal_l1")
    return value


@dataclass(frozen=True, slots=True)
class V1LiteralJPrivatePrimalDirectionDescriptor:
    """Array-free identity for one KKT private-primal direction.

    This descriptor is the structural lineage that lets the CUDA literal-J
    projection identify a direction without hashing the route-scale relaxed
    primal tensor.  It is deliberately only metadata: the corresponding CUDA
    tensor must still arrive through an ephemeral lease or exact CUDA tensor
    path, and this descriptor grants no KKT-sensitivity, C, oracle, commit, or
    Pareto authority.
    """

    direction_id: str
    branch: CanonicalPrivatePrimalBranch
    source_problem_hash: str
    source_policy_hash: str
    source_result_hash: str
    source_branch_result_hash: str
    source_branch_receipt_hash: str
    source_execution_receipt_hash: str
    source_execution_design_hash: str
    route_axis_hash: str
    route_count: int
    direction_role: str = "relaxed_kkt_converged_private_primal_z"
    direction_tensor_payload_omitted: bool = True
    source_converged_branch_receipt: bool = True
    literal_j_projection_identity_only: bool = True
    kkt_solution_sensitivity_authority: bool = False
    c_stream_authority: bool = False
    fqr_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    production_authority: bool = False
    pareto_authority: bool = False
    schema: str = LITERAL_J_PRIVATE_PRIMAL_DIRECTION_DESCRIPTOR_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "direction_id",
            "source_problem_hash",
            "source_policy_hash",
            "source_result_hash",
            "source_branch_result_hash",
            "source_branch_receipt_hash",
            "source_execution_receipt_hash",
            "source_execution_design_hash",
            "route_axis_hash",
            "direction_role",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "branch",
            _canonical_private_primal_branch(self.branch, name="branch"),
        )
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="route_count"),
        )
        if self.direction_role != "relaxed_kkt_converged_private_primal_z":
            raise ValueError("unsupported private-primal direction role")
        bool_fields = (
            "direction_tensor_payload_omitted",
            "source_converged_branch_receipt",
            "literal_j_projection_identity_only",
            "kkt_solution_sensitivity_authority",
            "c_stream_authority",
            "fqr_authority",
            "calibration_authority",
            "oracle_authority",
            "mask_commit_authority",
            "production_authority",
            "pareto_authority",
        )
        if any(type(getattr(self, name)) is not bool for name in bool_fields):
            raise TypeError("private-primal descriptor flags must be exact booleans")
        if (
            self.direction_tensor_payload_omitted is not True
            or self.source_converged_branch_receipt is not True
            or self.literal_j_projection_identity_only is not True
            or self.schema != LITERAL_J_PRIVATE_PRIMAL_DIRECTION_DESCRIPTOR_SCHEMA
        ):
            raise PermissionError("private-primal descriptor crossed a closed gate")
        forbidden = (
            self.kkt_solution_sensitivity_authority,
            self.c_stream_authority,
            self.fqr_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.production_authority,
            self.pareto_authority,
        )
        if any(forbidden):
            raise PermissionError("private-primal descriptor crossed its authority boundary")

    @property
    def descriptor_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "direction_id": self.direction_id,
                "branch": self.branch,
                "source_problem_hash": self.source_problem_hash,
                "source_policy_hash": self.source_policy_hash,
                "source_result_hash": self.source_result_hash,
                "source_branch_result_hash": self.source_branch_result_hash,
                "source_branch_receipt_hash": self.source_branch_receipt_hash,
                "source_execution_receipt_hash": self.source_execution_receipt_hash,
                "source_execution_design_hash": self.source_execution_design_hash,
                "route_axis_hash": self.route_axis_hash,
                "route_count": self.route_count,
                "direction_role": self.direction_role,
                "direction_tensor_payload_omitted": self.direction_tensor_payload_omitted,
                "source_converged_branch_receipt": self.source_converged_branch_receipt,
                "literal_j_projection_identity_only": self.literal_j_projection_identity_only,
            },
            prefix=_PRIVATE_PRIMAL_DESCRIPTOR_PREFIX,
        )


@dataclass(frozen=True, slots=True)
class V1LiteralJPrivatePrimalDirectionDescriptorAxis:
    """Canonical two-branch private-primal direction axis for literal-J projection."""

    descriptors: tuple[V1LiteralJPrivatePrimalDirectionDescriptor, ...]
    direction_ids: tuple[str, str]
    route_axis_hash: str
    route_count: int
    source_problem_hash: str
    source_policy_hash: str
    source_result_hash: str
    source_execution_receipt_hash: str
    source_execution_design_hash: str
    canonical_branch_order: tuple[str, str] = ("spectral", "diagonal_l1")
    route_scale_d2h_array_hashing_forbidden: bool = True
    descriptor_axis_is_structural_not_tensor_payload: bool = True
    schema: str = LITERAL_J_PRIVATE_PRIMAL_DIRECTION_DESCRIPTOR_AXIS_SCHEMA

    def __post_init__(self) -> None:
        descriptors = tuple(self.descriptors)
        if (
            len(descriptors) != 2
            or any(
                type(item) is not V1LiteralJPrivatePrimalDirectionDescriptor
                for item in descriptors
            )
        ):
            raise TypeError("private-primal descriptor axis requires exactly two descriptors")
        object.__setattr__(self, "descriptors", descriptors)
        direction_ids = tuple(self.direction_ids)
        if len(direction_ids) != 2:
            raise ValueError("private-primal descriptor axis must expose two direction ids")
        for index, value in enumerate(direction_ids):
            _identity(value, name=f"direction_ids[{index}]")
        object.__setattr__(self, "direction_ids", direction_ids)
        for name in (
            "route_axis_hash",
            "source_problem_hash",
            "source_policy_hash",
            "source_result_hash",
            "source_execution_receipt_hash",
            "source_execution_design_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="route_count"),
        )
        if self.canonical_branch_order != ("spectral", "diagonal_l1"):
            raise ValueError("private-primal descriptor axis branch order is canonical")
        if (
            self.route_scale_d2h_array_hashing_forbidden is not True
            or self.descriptor_axis_is_structural_not_tensor_payload is not True
            or self.schema != LITERAL_J_PRIVATE_PRIMAL_DIRECTION_DESCRIPTOR_AXIS_SCHEMA
        ):
            raise PermissionError("private-primal descriptor axis weakened a closed gate")
        expected_branches = ("spectral", "diagonal_l1")
        if tuple(item.branch for item in descriptors) != expected_branches:
            raise ValueError("private-primal descriptor axis branch order is not canonical")
        if tuple(item.direction_id for item in descriptors) != direction_ids:
            raise ValueError("private-primal descriptor direction ids do not close")
        for descriptor in descriptors:
            if (
                descriptor.route_axis_hash != self.route_axis_hash
                or descriptor.route_count != self.route_count
                or descriptor.source_problem_hash != self.source_problem_hash
                or descriptor.source_policy_hash != self.source_policy_hash
                or descriptor.source_result_hash != self.source_result_hash
                or descriptor.source_execution_receipt_hash
                != self.source_execution_receipt_hash
                or descriptor.source_execution_design_hash
                != self.source_execution_design_hash
            ):
                raise ValueError("private-primal descriptor left the common source axis")

    @classmethod
    def from_descriptors(
        cls,
        spectral: V1LiteralJPrivatePrimalDirectionDescriptor,
        diagonal_l1: V1LiteralJPrivatePrimalDirectionDescriptor,
    ) -> "V1LiteralJPrivatePrimalDirectionDescriptorAxis":
        if type(spectral) is not V1LiteralJPrivatePrimalDirectionDescriptor:
            raise TypeError("spectral descriptor must be exact")
        if type(diagonal_l1) is not V1LiteralJPrivatePrimalDirectionDescriptor:
            raise TypeError("diagonal_l1 descriptor must be exact")
        return cls(
            descriptors=(spectral, diagonal_l1),
            direction_ids=(spectral.direction_id, diagonal_l1.direction_id),
            route_axis_hash=spectral.route_axis_hash,
            route_count=spectral.route_count,
            source_problem_hash=spectral.source_problem_hash,
            source_policy_hash=spectral.source_policy_hash,
            source_result_hash=spectral.source_result_hash,
            source_execution_receipt_hash=spectral.source_execution_receipt_hash,
            source_execution_design_hash=spectral.source_execution_design_hash,
        )

    @property
    def descriptor_hashes(self) -> tuple[str, str]:
        return tuple(item.descriptor_hash for item in self.descriptors)  # type: ignore[return-value]

    @property
    def descriptor_axis_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "canonical_branch_order": self.canonical_branch_order,
                "direction_ids": self.direction_ids,
                "direction_descriptor_hashes": self.descriptor_hashes,
                "route_axis_hash": self.route_axis_hash,
                "route_count": self.route_count,
                "source_problem_hash": self.source_problem_hash,
                "source_policy_hash": self.source_policy_hash,
                "source_result_hash": self.source_result_hash,
                "source_execution_receipt_hash": self.source_execution_receipt_hash,
                "source_execution_design_hash": self.source_execution_design_hash,
                "route_scale_d2h_array_hashing_forbidden": (
                    self.route_scale_d2h_array_hashing_forbidden
                ),
                "descriptor_axis_is_structural_not_tensor_payload": (
                    self.descriptor_axis_is_structural_not_tensor_payload
                ),
            },
            prefix=_PRIVATE_PRIMAL_DESCRIPTOR_AXIS_PREFIX,
        )


_PRIVATE_PRIMAL_DIRECTION_LEASE_CONSTRUCTION_KEY: Final = object()


class V1LiteralJPrivatePrimalDirectionCUDALease:
    """Ephemeral lease for exactly two device-resident private-primal directions."""

    __slots__ = (
        "_closed",
        "_descriptor_axis",
        "_device_identity",
        "_directions",
        "_handle",
        "_storage_invariant",
        "_token",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        descriptor_axis: V1LiteralJPrivatePrimalDirectionDescriptorAxis,
        directions: Any | None = None,
        _construction_key: object | None = None,
        _handle: Any | None = None,
        _storage_invariant: tuple[object, ...] | None = None,
        _token: object | None = None,
    ) -> None:
        if type(descriptor_axis) is not V1LiteralJPrivatePrimalDirectionDescriptorAxis:
            raise TypeError("private-primal direction lease requires an exact descriptor axis")
        if _handle is not None:
            if _construction_key is not _PRIVATE_PRIMAL_DIRECTION_LEASE_CONSTRUCTION_KEY:
                raise TypeError("private-primal direction handle must issue handle-backed leases")
            if _token is None or not _storage_invariant:
                raise ValueError("handle-backed private-primal lease lacks lifecycle state")
            self._descriptor_axis = descriptor_axis
            self._directions = None
            self._handle = _handle
            self._storage_invariant = tuple(_storage_invariant)
            self._token = _token
            self._device_identity = _handle.device_identity
            self._closed = False
            return
        if _construction_key is not None or _storage_invariant is not None or _token is not None:
            raise TypeError("direct private-primal direction leases cannot carry handle state")
        torch = _require_torch()
        tensor = _require_cuda_tensor(
            torch,
            directions,
            name="private_primal_directions",
            device=None,
            ndim=2,
        )
        if tuple(int(item) for item in tensor.shape) != (2, descriptor_axis.route_count):
            _refuse(
                V1LiteralJFixedDirectionProjectionCUDARefusalCode.INVALID_DESIGN,
                "private-primal directions must have shape (2, route_count)",
            )
        self._descriptor_axis = descriptor_axis
        self._directions = tensor
        self._handle = None
        self._storage_invariant = ()
        self._token = None
        self._device_identity = str(tensor.device)
        self._closed = False

    @property
    def descriptor_axis(self) -> V1LiteralJPrivatePrimalDirectionDescriptorAxis:
        self._require_live()
        return self._descriptor_axis

    @property
    def descriptor_axis_hash(self) -> str:
        return self.descriptor_axis.descriptor_axis_hash

    @property
    def direction_ids(self) -> tuple[str, str]:
        return self.descriptor_axis.direction_ids

    @property
    def route_axis_hash(self) -> str:
        return self.descriptor_axis.route_axis_hash

    @property
    def route_count(self) -> int:
        return self.descriptor_axis.route_count

    @property
    def device_identity(self) -> str:
        self._require_live()
        return self._device_identity

    @property
    def is_closed(self) -> bool:
        return bool(self._closed)

    def _require_live(self) -> Any:
        if self._closed:
            _refuse(
                V1LiteralJFixedDirectionProjectionCUDARefusalCode.FINALIZED,
                "private-primal direction lease is closed",
            )
        if self._handle is not None:
            return self._handle._directions_for_live_lease(
                token=self._token,
                expected_storage_invariant=self._storage_invariant,
            )
        if self._directions is None:
            _refuse(
                V1LiteralJFixedDirectionProjectionCUDARefusalCode.FINALIZED,
                "private-primal direction lease is closed",
            )
        return self._directions

    def directions_for_literal_j_projection(self) -> Any:
        """Return the resident CUDA tensor to the projection owner only."""

        return self._require_live()

    def close(self) -> None:
        if self._closed:
            return
        handle = self._handle
        token = self._token
        invariant = self._storage_invariant
        if handle is not None:
            try:
                handle._close_live_lease(
                    token=token,
                    expected_storage_invariant=invariant,
                )
            finally:
                self._handle = None
                self._token = None
                self._storage_invariant = ()
                self._closed = True
            return
        self._directions = None
        self._closed = True

    def __enter__(self) -> "V1LiteralJPrivatePrimalDirectionCUDALease":
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
        raise TypeError("private-primal direction leases are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("private-primal direction leases are nonserializable")

    def __getstate__(self) -> NoReturn:
        raise TypeError("private-primal direction leases are nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("private-primal direction leases cannot be copied")

    def __deepcopy__(self, memo: dict[int, Any]) -> NoReturn:
        del memo
        raise TypeError("private-primal direction leases cannot be copied")


class V1LiteralJPrivatePrimalDirectionCUDAHandle:
    """Ephemeral owner for KKT-produced two-branch private-primal directions."""

    __slots__ = (
        "_active_lease_tokens",
        "_descriptor_axis",
        "_directions",
        "_device_identity",
        "_evicted",
        "_lease_lock",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        descriptor_axis: V1LiteralJPrivatePrimalDirectionDescriptorAxis,
        directions: Any,
    ) -> None:
        if type(descriptor_axis) is not V1LiteralJPrivatePrimalDirectionDescriptorAxis:
            raise TypeError("private-primal direction handle requires an exact descriptor axis")
        torch = _require_torch()
        tensor = _require_cuda_tensor(
            torch,
            directions,
            name="private_primal_directions",
            device=None,
            ndim=2,
        )
        if tuple(int(item) for item in tensor.shape) != (2, descriptor_axis.route_count):
            _refuse(
                V1LiteralJFixedDirectionProjectionCUDARefusalCode.INVALID_DESIGN,
                "private-primal directions must have shape (2, route_count)",
            )
        self._descriptor_axis = descriptor_axis
        self._directions = tensor
        self._device_identity = str(tensor.device)
        self._active_lease_tokens: set[object] = set()
        self._lease_lock = RLock()
        self._evicted = False

    def _storage_invariant_locked(self, directions: Any) -> tuple[object, ...]:
        return (
            str(directions.device),
            str(directions.dtype),
            tuple(int(item) for item in directions.shape),
            bool(directions.is_contiguous()),
            int(directions.data_ptr()),
            int(getattr(directions, "_version", 0)),
        )

    def _require_live_locked(self) -> Any:
        if self._evicted or self._directions is None:
            _refuse(
                V1LiteralJFixedDirectionProjectionCUDARefusalCode.FINALIZED,
                "private-primal direction handle was evicted",
            )
        return self._directions

    @property
    def descriptor_axis(self) -> V1LiteralJPrivatePrimalDirectionDescriptorAxis:
        with self._lease_lock:
            self._require_live_locked()
            return self._descriptor_axis

    @property
    def descriptor_axis_hash(self) -> str:
        return self.descriptor_axis.descriptor_axis_hash

    @property
    def device_identity(self) -> str:
        with self._lease_lock:
            self._require_live_locked()
            return self._device_identity

    @property
    def route_count(self) -> int:
        return self.descriptor_axis.route_count

    @property
    def active_lease_count(self) -> int:
        with self._lease_lock:
            self._require_live_locked()
            return len(self._active_lease_tokens)

    @property
    def is_evicted(self) -> bool:
        return bool(self._evicted)

    def borrow_for_literal_j_projection(self) -> V1LiteralJPrivatePrimalDirectionCUDALease:
        """Borrow resident directions for one registered CUDA projection."""

        with self._lease_lock:
            directions = self._require_live_locked()
            token = object()
            self._active_lease_tokens.add(token)
            try:
                return V1LiteralJPrivatePrimalDirectionCUDALease(
                    descriptor_axis=self._descriptor_axis,
                    _construction_key=_PRIVATE_PRIMAL_DIRECTION_LEASE_CONSTRUCTION_KEY,
                    _handle=self,
                    _storage_invariant=self._storage_invariant_locked(directions),
                    _token=token,
                )
            except BaseException:
                self._active_lease_tokens.remove(token)
                raise

    def _directions_for_live_lease(
        self,
        *,
        token: object,
        expected_storage_invariant: tuple[object, ...],
    ) -> Any:
        with self._lease_lock:
            directions = self._require_live_locked()
            if token not in self._active_lease_tokens:
                _refuse(
                    V1LiteralJFixedDirectionProjectionCUDARefusalCode.FINALIZED,
                    "private-primal direction lease is no longer registered",
                )
            if self._storage_invariant_locked(directions) != expected_storage_invariant:
                _refuse(
                    V1LiteralJFixedDirectionProjectionCUDARefusalCode.INVALID_DESIGN,
                    "private-primal direction storage mutated during a lease",
                )
            return directions

    def _close_live_lease(
        self,
        *,
        token: object,
        expected_storage_invariant: tuple[object, ...],
    ) -> None:
        with self._lease_lock:
            directions = self._require_live_locked()
            if token not in self._active_lease_tokens:
                _refuse(
                    V1LiteralJFixedDirectionProjectionCUDARefusalCode.FINALIZED,
                    "private-primal direction lease is no longer registered",
                )
            self._active_lease_tokens.remove(token)
            if self._storage_invariant_locked(directions) != expected_storage_invariant:
                _refuse(
                    V1LiteralJFixedDirectionProjectionCUDARefusalCode.INVALID_DESIGN,
                    "private-primal direction storage mutated during a lease",
                )

    def evict(self) -> None:
        with self._lease_lock:
            self._require_live_locked()
            if self._active_lease_tokens:
                _refuse(
                    V1LiteralJFixedDirectionProjectionCUDARefusalCode.FINALIZED,
                    "private-primal directions cannot be evicted while a lease is live",
                )
            self._directions = None
            self._evicted = True

    def __reduce__(self) -> NoReturn:
        raise TypeError("private-primal direction handles are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("private-primal direction handles are nonserializable")

    def __getstate__(self) -> NoReturn:
        raise TypeError("private-primal direction handles are nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("private-primal direction handles cannot be copied")

    def __deepcopy__(self, memo: dict[int, Any]) -> NoReturn:
        del memo
        raise TypeError("private-primal direction handles cannot be copied")


def _quadratic_bound_max(torch: Any, projection_abs_bounds: Any) -> float:
    value = torch.max(projection_abs_bounds * projection_abs_bounds)
    return float(value.detach().cpu().item())


@dataclass(frozen=True, slots=True)
class V1LiteralJFixedDirectionProjectionCUDAExecutionDesign:
    """Frozen deterministic CUDA execution design for fixed ``M_J z``."""

    design_id: str = "modern-v1-literal-j-fixed-direction-projection-cuda-v1"
    backend: Literal["torch_cuda"] = "torch_cuda"
    algorithm_id: str = LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_ALGORITHM
    numerical_enclosure_id: str = (
        LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_NUMERICAL_ENCLOSURE_ID
    )
    device: str = "cuda"
    route_tile_size: int = 1_048_576
    maximum_context_batch_size: int = 4
    hash_chunk_size: int = 1_048_576
    small_domain_reference_hash_route_limit: int = (
        LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_SMALL_DOMAIN_HASH_LIMIT
    )
    hbm_ceiling_fraction: float = (
        LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_HBM_CEILING_FRACTION
    )
    require_torch_deterministic_algorithms: bool = True
    require_fp64_accumulation: bool = True
    allow_cpu_fallback: bool = False
    allow_route_scale_d2h_array_hashing: bool = False
    allow_small_domain_reference_array_hashing: bool = True
    retain_literal_j_rows: bool = False
    claim_kkt_solution_sensitivity: bool = False
    schema: str = LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_DESIGN_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "design_id", _identity(self.design_id, name="design_id"))
        object.__setattr__(self, "device", _identity(self.device, name="device"))
        if self.backend != "torch_cuda":
            raise ValueError("literal-J projection CUDA has no alternate backend")
        if self.algorithm_id != LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_ALGORITHM:
            raise ValueError("unsupported CUDA literal-J projection algorithm")
        if (
            self.numerical_enclosure_id
            != LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_NUMERICAL_ENCLOSURE_ID
        ):
            raise ValueError("unsupported numerical enclosure")
        for name in (
            "route_tile_size",
            "maximum_context_batch_size",
            "hash_chunk_size",
            "small_domain_reference_hash_route_limit",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name=name))
        ceiling = _finite_nonnegative(
            self.hbm_ceiling_fraction,
            name="HBM ceiling fraction",
        )
        if ceiling != LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_HBM_CEILING_FRACTION:
            raise ValueError("literal-J projection CUDA HBM ceiling is frozen at 85 percent")
        object.__setattr__(self, "hbm_ceiling_fraction", ceiling)
        flags = (
            self.require_torch_deterministic_algorithms,
            self.require_fp64_accumulation,
            self.allow_cpu_fallback,
            self.allow_route_scale_d2h_array_hashing,
            self.allow_small_domain_reference_array_hashing,
            self.retain_literal_j_rows,
            self.claim_kkt_solution_sensitivity,
        )
        if any(type(item) is not bool for item in flags):
            raise TypeError("CUDA projection design flags must be exact booleans")
        if flags != (True, True, False, False, True, False, False):
            raise PermissionError("literal-J CUDA projection design weakened a closed gate")
        if self.schema != LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_DESIGN_SCHEMA:
            raise ValueError("unsupported literal-J CUDA projection design schema")

    @property
    def design_hash(self) -> str:
        return content_hash(self, prefix=_DESIGN_PREFIX)


@dataclass(frozen=True, slots=True)
class V1LiteralJFixedDirectionProjectionCUDAHBMForecast:
    """Pre-outcome HBM forecast under the fixed 85 percent CUDA ceiling."""

    design_hash: str
    inference_id: str
    route_axis_hash: str
    route_count: int
    direction_count: int
    device: str
    device_total_memory_bytes: int
    device_free_memory_bytes: int
    process_allocated_before_bytes: int
    process_reserved_before_bytes: int
    hbm_ceiling_fraction: float
    hbm_ceiling_bytes: int
    route_tile_size: int
    maximum_context_batch_size: int
    hash_chunk_size: int
    resident_direction_bytes: int
    resident_bound_bytes: int
    resident_sufficient_stat_bytes: int
    incoming_row_batch_peak_bytes: int
    projection_batch_bytes: int
    route_tile_workspace_bytes: int
    hash_chunk_host_staging_bytes: int
    allocator_guard_bytes: int
    estimated_incremental_peak_bytes: int
    estimated_process_peak_bytes: int
    admitted: bool
    schema: str = LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_HBM_FORECAST_SCHEMA

    def __post_init__(self) -> None:
        for name in ("design_hash", "inference_id", "route_axis_hash", "device"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "route_count",
            "direction_count",
            "device_total_memory_bytes",
            "hbm_ceiling_bytes",
            "route_tile_size",
            "maximum_context_batch_size",
            "hash_chunk_size",
            "resident_direction_bytes",
            "resident_bound_bytes",
            "resident_sufficient_stat_bytes",
            "incoming_row_batch_peak_bytes",
            "projection_batch_bytes",
            "route_tile_workspace_bytes",
            "hash_chunk_host_staging_bytes",
            "allocator_guard_bytes",
            "estimated_incremental_peak_bytes",
            "estimated_process_peak_bytes",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name=name))
        for name in (
            "device_free_memory_bytes",
            "process_allocated_before_bytes",
            "process_reserved_before_bytes",
        ):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name=name))
        ceiling = _finite_nonnegative(
            self.hbm_ceiling_fraction,
            name="HBM ceiling fraction",
        )
        if ceiling != LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_HBM_CEILING_FRACTION:
            raise ValueError("literal-J CUDA forecast must use the fixed 85 percent ceiling")
        object.__setattr__(self, "hbm_ceiling_fraction", ceiling)
        if self.device_free_memory_bytes > self.device_total_memory_bytes:
            raise ValueError("free CUDA memory cannot exceed total memory")
        if self.process_allocated_before_bytes > self.process_reserved_before_bytes:
            raise ValueError("allocated CUDA memory cannot exceed reserved CUDA memory")
        n = self.route_count
        d = self.direction_count
        b = self.maximum_context_batch_size
        tile = min(self.route_tile_size, n)
        hash_chunk = min(self.hash_chunk_size, n)
        expected = {
            "hbm_ceiling_bytes": math.floor(
                self.device_total_memory_bytes * self.hbm_ceiling_fraction
            ),
            "resident_direction_bytes": 8 * d * n,
            "resident_bound_bytes": 8 * (n + d),
            "resident_sufficient_stat_bytes": 8 * (2 * d * n + 2 * d),
            "incoming_row_batch_peak_bytes": 8 * b * n,
            "projection_batch_bytes": 8 * b * d,
            "route_tile_workspace_bytes": 8 * d * tile,
            "hash_chunk_host_staging_bytes": 8 * hash_chunk,
        }
        subtotal = (
            expected["resident_direction_bytes"]
            + expected["resident_bound_bytes"]
            + expected["resident_sufficient_stat_bytes"]
            + expected["incoming_row_batch_peak_bytes"]
            + expected["projection_batch_bytes"]
            + expected["route_tile_workspace_bytes"]
        )
        expected_guard = max(
            LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_MINIMUM_GUARD_BYTES,
            math.ceil(LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_GUARD_FRACTION * subtotal),
        )
        expected_incremental = subtotal + expected_guard
        expected_process = self.process_reserved_before_bytes + expected_incremental
        expected_admitted = (
            self.process_allocated_before_bytes + expected_incremental <= expected["hbm_ceiling_bytes"]
            and expected_incremental <= self.device_free_memory_bytes + max(0, self.process_reserved_before_bytes - self.process_allocated_before_bytes)
        )
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError("literal-J CUDA HBM forecast does not replay")
        if (
            self.allocator_guard_bytes != expected_guard
            or self.estimated_incremental_peak_bytes != expected_incremental
            or self.estimated_process_peak_bytes != expected_process
            or self.admitted is not expected_admitted
        ):
            raise ValueError("literal-J CUDA HBM forecast does not replay")
        if type(self.admitted) is not bool:
            raise TypeError("HBM forecast admission must be an exact bool")
        if self.schema != LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_HBM_FORECAST_SCHEMA:
            raise ValueError("unsupported literal-J CUDA HBM forecast schema")

    @property
    def forecast_hash(self) -> str:
        return content_hash(self, prefix=_FORECAST_PREFIX)


@dataclass(frozen=True, slots=True)
class V1LiteralJFixedDirectionProjectionCUDATelemetry:
    """Array-free CUDA execution counters and numerical enclosure metadata."""

    receipt_hash: str
    forecast_hash: str
    design_hash: str
    inference_id: str
    route_axis_hash: str
    device: str
    route_count: int
    direction_count: int
    observed_counts: tuple[int, int]
    route_tile_size: int
    maximum_context_batch_size: int
    update_batch_count: int
    context_row_count: int
    route_tile_visit_count: int
    literal_j_route_value_visit_count: int
    projection_matmul_count: int
    support_reduction_count: int
    fp64_operator_tile_accumulation_count: int
    output_hash_vector_count: int
    output_hash_d2h_tensor_payload_bytes: int
    scalar_d2h_tensor_payload_bytes: int
    peak_allocated_hbm_bytes: int
    peak_reserved_hbm_bytes: int
    numerical_route_radius: float
    numerical_quadratic_radius: float
    hbm_forecast_admitted: bool
    torch_deterministic_algorithms_enabled: bool = True
    torch_deterministic_warn_only: bool = False
    backend: Literal["torch_cuda"] = "torch_cuda"
    algorithm_id: str = LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_ALGORITHM
    numerical_enclosure_id: str = (
        LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_NUMERICAL_ENCLOSURE_ID
    )
    literal_j_rows_retained: bool = False
    row_scale_cpu_fallback_used: bool = False
    full_output_arrays_retained_in_receipt: bool = False
    route_scale_raw_rows_d2h: bool = False
    route_scale_workspace_released: bool = True
    private_primal_lineage_claimed_by_cuda: bool = False
    kkt_solution_sensitivity_authority: bool = False
    c_stream_authority: bool = False
    fqr_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    production_authority: bool = False
    pareto_authority: bool = False
    schema: str = LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_TELEMETRY_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "receipt_hash",
            "forecast_hash",
            "design_hash",
            "inference_id",
            "route_axis_hash",
            "device",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "route_count",
            "direction_count",
            "route_tile_size",
            "maximum_context_batch_size",
            "update_batch_count",
            "context_row_count",
            "route_tile_visit_count",
            "literal_j_route_value_visit_count",
            "projection_matmul_count",
            "support_reduction_count",
            "fp64_operator_tile_accumulation_count",
            "output_hash_vector_count",
            "output_hash_d2h_tensor_payload_bytes",
            "scalar_d2h_tensor_payload_bytes",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name=name))
        for name in ("peak_allocated_hbm_bytes", "peak_reserved_hbm_bytes"):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name=name))
        if (
            type(self.observed_counts) is not tuple
            or len(self.observed_counts) != 2
            or any(type(item) is not int or item <= 0 for item in self.observed_counts)
            or sum(self.observed_counts) != self.context_row_count
        ):
            raise ValueError("CUDA telemetry observed counts must be a nonempty pair")
        if self.peak_allocated_hbm_bytes > self.peak_reserved_hbm_bytes:
            raise ValueError("allocated HBM cannot exceed reserved HBM")
        expected_tiles = (
            math.ceil(self.route_count / self.route_tile_size) * self.context_row_count
        )
        if (
            self.route_tile_visit_count != expected_tiles
            or self.literal_j_route_value_visit_count
            != self.context_row_count * self.route_count
            or self.projection_matmul_count != self.update_batch_count
            or self.support_reduction_count != 2 * self.update_batch_count
            or self.fp64_operator_tile_accumulation_count != expected_tiles
            or self.output_hash_vector_count != self.direction_count
            or self.output_hash_d2h_tensor_payload_bytes
            != 8
            * self.direction_count
            * LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_OUTPUT_FINGERPRINT_SCALAR_COUNT
            or self.scalar_d2h_tensor_payload_bytes < 1
        ):
            raise ValueError("CUDA projection telemetry counters do not replay")
        for name in ("numerical_route_radius", "numerical_quadratic_radius"):
            object.__setattr__(self, name, _finite_nonnegative(getattr(self, name), name=name))
        bool_fields = (
            "hbm_forecast_admitted",
            "torch_deterministic_algorithms_enabled",
            "torch_deterministic_warn_only",
            "literal_j_rows_retained",
            "row_scale_cpu_fallback_used",
            "full_output_arrays_retained_in_receipt",
            "route_scale_raw_rows_d2h",
            "route_scale_workspace_released",
            "private_primal_lineage_claimed_by_cuda",
            "kkt_solution_sensitivity_authority",
            "c_stream_authority",
            "fqr_authority",
            "calibration_authority",
            "oracle_authority",
            "mask_commit_authority",
            "production_authority",
            "pareto_authority",
        )
        if any(type(getattr(self, name)) is not bool for name in bool_fields):
            raise TypeError("CUDA projection telemetry flags must be exact booleans")
        if (
            self.hbm_forecast_admitted is not True
            or self.torch_deterministic_algorithms_enabled is not True
            or self.torch_deterministic_warn_only is not False
            or self.backend != "torch_cuda"
            or self.algorithm_id != LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_ALGORITHM
            or self.numerical_enclosure_id
            != LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_NUMERICAL_ENCLOSURE_ID
            or self.schema != LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_TELEMETRY_SCHEMA
        ):
            raise PermissionError("CUDA projection telemetry crossed a closed gate")
        forbidden = (
            self.literal_j_rows_retained,
            self.row_scale_cpu_fallback_used,
            self.full_output_arrays_retained_in_receipt,
            self.route_scale_raw_rows_d2h,
            self.private_primal_lineage_claimed_by_cuda,
            self.kkt_solution_sensitivity_authority,
            self.c_stream_authority,
            self.fqr_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.production_authority,
            self.pareto_authority,
        )
        if self.route_scale_workspace_released is not True or any(forbidden):
            raise PermissionError("CUDA projection telemetry crossed its authority boundary")

    @property
    def telemetry_hash(self) -> str:
        return content_hash(self, prefix=_TELEMETRY_PREFIX)


@dataclass(frozen=True, slots=True)
class V1LiteralJFixedDirectionProjectionCUDABundle:
    """Array-free CUDA execution bundle for a fixed-direction projection."""

    receipt: V1LiteralJFixedDirectionProjectionReceipt
    forecast: V1LiteralJFixedDirectionProjectionCUDAHBMForecast
    telemetry: V1LiteralJFixedDirectionProjectionCUDATelemetry
    design: V1LiteralJFixedDirectionProjectionCUDAExecutionDesign
    schema: str = LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        if type(self.receipt) is not V1LiteralJFixedDirectionProjectionReceipt:
            raise TypeError("CUDA projection bundle requires the exact projection receipt")
        if type(self.forecast) is not V1LiteralJFixedDirectionProjectionCUDAHBMForecast:
            raise TypeError("CUDA projection bundle requires the exact HBM forecast")
        if type(self.telemetry) is not V1LiteralJFixedDirectionProjectionCUDATelemetry:
            raise TypeError("CUDA projection bundle requires exact telemetry")
        if type(self.design) is not V1LiteralJFixedDirectionProjectionCUDAExecutionDesign:
            raise TypeError("CUDA projection bundle requires exact design")
        if (
            self.forecast.design_hash != self.design.design_hash
            or self.telemetry.design_hash != self.design.design_hash
            or self.telemetry.forecast_hash != self.forecast.forecast_hash
            or self.telemetry.receipt_hash != self.receipt.receipt_hash
            or self.receipt.route_axis_hash != self.forecast.route_axis_hash
            or self.receipt.route_axis_hash != self.telemetry.route_axis_hash
            or self.receipt.inference_id != self.forecast.inference_id
            or self.receipt.inference_id != self.telemetry.inference_id
            or self.receipt.direction_count != self.forecast.direction_count
            or self.receipt.direction_count != self.telemetry.direction_count
            or self.receipt.observed_counts != self.telemetry.observed_counts
        ):
            raise ValueError("CUDA projection bundle receipt, forecast, and telemetry do not close")
        if self.schema != LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_BUNDLE_SCHEMA:
            raise ValueError("unsupported CUDA projection bundle schema")

    @property
    def bundle_hash(self) -> str:
        return content_hash(self, prefix=_BUNDLE_PREFIX)


def build_v1_literal_j_projection_cuda_hbm_forecast(
    *,
    design: V1LiteralJFixedDirectionProjectionCUDAExecutionDesign,
    inference_id: str,
    route_axis_hash: str,
    route_count: int,
    direction_count: int,
    torch_module: Any | None = None,
    device: Any | None = None,
    device_total_memory_bytes: int | None = None,
    device_free_memory_bytes: int | None = None,
    process_allocated_before_bytes: int | None = None,
    process_reserved_before_bytes: int | None = None,
) -> V1LiteralJFixedDirectionProjectionCUDAHBMForecast:
    """Construct the replayed HBM forecast, optionally with supplied device facts."""

    if type(design) is not V1LiteralJFixedDirectionProjectionCUDAExecutionDesign:
        raise TypeError("design must be exact CUDA projection design")
    n = _positive_int(route_count, name="route_count")
    d = _positive_int(direction_count, name="direction_count")
    if torch_module is not None and device is not None and device_total_memory_bytes is None:
        free_bytes, total_bytes = torch_module.cuda.mem_get_info(device)
        allocated = int(torch_module.cuda.memory_allocated(device))
        reserved = int(torch_module.cuda.memory_reserved(device))
        available = cuda_reusable_memory_bytes(device, framework=torch_module, free_bytes=free_bytes)
    else:
        total_bytes = _positive_int(
            device_total_memory_bytes,
            name="device_total_memory_bytes",
        )
        free_bytes = _nonnegative_int(
            device_free_memory_bytes,
            name="device_free_memory_bytes",
        )
        allocated = _nonnegative_int(
            process_allocated_before_bytes,
            name="process_allocated_before_bytes",
        )
        reserved = _nonnegative_int(
            process_reserved_before_bytes,
            name="process_reserved_before_bytes",
        )
    if torch_module is None or device is None or device_total_memory_bytes is not None:
        available = int(free_bytes) + max(0, reserved - allocated)
    tile = min(design.route_tile_size, n)
    hash_chunk = min(design.hash_chunk_size, n)
    resident_direction = 8 * d * n
    resident_bound = 8 * (n + d)
    resident_stats = 8 * (2 * d * n + 2 * d)
    incoming = 8 * design.maximum_context_batch_size * n
    projection = 8 * design.maximum_context_batch_size * d
    tile_bytes = 8 * d * tile
    subtotal = (
        resident_direction
        + resident_bound
        + resident_stats
        + incoming
        + projection
        + tile_bytes
    )
    guard = max(
        LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_MINIMUM_GUARD_BYTES,
        math.ceil(LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_GUARD_FRACTION * subtotal),
    )
    incremental = subtotal + guard
    ceiling = math.floor(int(total_bytes) * design.hbm_ceiling_fraction)
    return V1LiteralJFixedDirectionProjectionCUDAHBMForecast(
        design_hash=design.design_hash,
        inference_id=_identity(inference_id, name="inference_id"),
        route_axis_hash=_identity(route_axis_hash, name="route_axis_hash"),
        route_count=n,
        direction_count=d,
        device=(str(device) if device is not None else design.device),
        device_total_memory_bytes=int(total_bytes),
        device_free_memory_bytes=int(free_bytes),
        process_allocated_before_bytes=int(allocated),
        process_reserved_before_bytes=int(reserved),
        hbm_ceiling_fraction=design.hbm_ceiling_fraction,
        hbm_ceiling_bytes=ceiling,
        route_tile_size=design.route_tile_size,
        maximum_context_batch_size=design.maximum_context_batch_size,
        hash_chunk_size=design.hash_chunk_size,
        resident_direction_bytes=resident_direction,
        resident_bound_bytes=resident_bound,
        resident_sufficient_stat_bytes=resident_stats,
        incoming_row_batch_peak_bytes=incoming,
        projection_batch_bytes=projection,
        route_tile_workspace_bytes=tile_bytes,
        hash_chunk_host_staging_bytes=8 * hash_chunk,
        allocator_guard_bytes=guard,
        estimated_incremental_peak_bytes=incremental,
        estimated_process_peak_bytes=int(reserved) + incremental,
        admitted=(
            allocated + incremental <= ceiling
            and incremental <= available
        ),
    )


def _verify_measured_hbm(
    *,
    forecast: V1LiteralJFixedDirectionProjectionCUDAHBMForecast,
    peak_allocated_bytes: int,
    peak_reserved_bytes: int,
) -> None:
    allocated = _nonnegative_int(peak_allocated_bytes, name="peak allocated HBM")
    reserved = _nonnegative_int(peak_reserved_bytes, name="peak reserved HBM")
    if allocated > reserved:
        _refuse(
            V1LiteralJFixedDirectionProjectionCUDARefusalCode.MEASURED_HBM_FORECAST_INVALIDATED,
            "measured allocated HBM exceeds reserved HBM",
        )
    if allocated > forecast.hbm_ceiling_bytes:
        _refuse(
            V1LiteralJFixedDirectionProjectionCUDARefusalCode.MEASURED_HBM_CEILING_EXCEEDED,
            "measured literal-J projection crossed the registered 85 percent HBM ceiling",
        )
    if (
        allocated
        > forecast.process_allocated_before_bytes + forecast.estimated_incremental_peak_bytes
        or reserved > forecast.estimated_process_peak_bytes
    ):
        _refuse(
            V1LiteralJFixedDirectionProjectionCUDARefusalCode.MEASURED_HBM_FORECAST_INVALIDATED,
            "measured literal-J projection exceeded its pre-outcome HBM forecast",
        )


class V1LiteralJFixedDirectionProjectionCUDAAccumulator:
    """Streaming CUDA prefix accumulator for fixed ``M_J z`` inference."""

    __slots__ = (
        "_torch",
        "_design",
        "_device",
        "_forecast",
        "_inference_id",
        "_model_identity_hash",
        "_chart_identity_hash",
        "_current_mask_hash",
        "_proposal_generation",
        "_discovery_geometry_source_hash",
        "_witness_stream_source_hash",
        "_literal_j_stream_source_hash",
        "_route_axis_hash",
        "_route_count",
        "_direction_ids",
        "_directions",
        "_direction_hashes",
        "_direction_axis_hash",
        "_literal_j_abs_bound",
        "_projection_abs_bounds",
        "_literal_j_abs_bound_source_hash",
        "_projection_abs_bound_source_hash",
        "_contribution_bound_hashes",
        "_quadratic_bound_values",
        "_stratum_ids",
        "_sealed_context_ids",
        "_sealed_order_hashes",
        "_familywise_alpha",
        "_private_primal_direction_descriptor_axis_hash",
        "_next_indices",
        "_seen_context_ids",
        "_seen_draw_ids",
        "_seen_source_measurement_ids",
        "_stratum_draw_ids",
        "_stratum_source_ids",
        "_sum_operator",
        "_sum_quadratic",
        "_batch_count",
        "_row_count",
        "_tile_visit_count",
        "_scalar_d2h_count",
        "_finalized",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        inference_id: str,
        model_identity_hash: str,
        chart_identity_hash: str,
        current_mask_hash: str,
        proposal_generation: int,
        discovery_geometry_source_hash: str,
        witness_stream_source_hash: str,
        literal_j_stream_source_hash: str,
        route_axis_hash: str,
        route_count: int,
        direction_ids: Sequence[str],
        directions: Any,
        literal_j_abs_bound: Any,
        projection_abs_bounds: Any,
        stratum_ids: Sequence[str],
        sealed_context_ids_by_stratum: Sequence[Sequence[str]],
        familywise_alpha: float,
        design: V1LiteralJFixedDirectionProjectionCUDAExecutionDesign | None = None,
        private_primal_direction_descriptor_axis: (
            V1LiteralJPrivatePrimalDirectionDescriptorAxis | None
        ) = None,
        private_primal_direction_descriptor_axis_hash: str | None = None,
        literal_j_abs_bound_source_hash: str | None = None,
        projection_abs_bound_source_hash: str | None = None,
    ) -> None:
        self._torch = _require_torch()
        self._design = (
            V1LiteralJFixedDirectionProjectionCUDAExecutionDesign()
            if design is None
            else design
        )
        if type(self._design) is not V1LiteralJFixedDirectionProjectionCUDAExecutionDesign:
            raise TypeError("design must be exact CUDA projection design")
        leased_directions = (
            directions
            if type(directions) is V1LiteralJPrivatePrimalDirectionCUDALease
            else None
        )
        descriptor_axis = private_primal_direction_descriptor_axis
        if descriptor_axis is not None and type(descriptor_axis) is not (
            V1LiteralJPrivatePrimalDirectionDescriptorAxis
        ):
            raise TypeError("private-primal descriptor axis must be exact")
        if leased_directions is not None:
            lease_axis = leased_directions.descriptor_axis
            if descriptor_axis is not None and descriptor_axis != lease_axis:
                _refuse(
                    V1LiteralJFixedDirectionProjectionCUDARefusalCode.INVALID_DESIGN,
                    "private-primal lease and descriptor axis differ",
                )
            descriptor_axis = lease_axis
        if private_primal_direction_descriptor_axis_hash is not None:
            supplied_hash = _identity(
                private_primal_direction_descriptor_axis_hash,
                name="private_primal_direction_descriptor_axis_hash",
            )
            if descriptor_axis is None:
                _refuse(
                    V1LiteralJFixedDirectionProjectionCUDARefusalCode.INVALID_DESIGN,
                    "a bare private-primal descriptor-axis hash cannot authorize "
                    "lineage; pass the exact typed descriptor axis or CUDA lease",
                )
            if supplied_hash != descriptor_axis.descriptor_axis_hash:
                _refuse(
                    V1LiteralJFixedDirectionProjectionCUDARefusalCode.INVALID_DESIGN,
                    "private-primal descriptor-axis hash does not match its typed descriptor",
                )
        self._device = _resolve_cuda_device(self._torch, self._design.device)
        _require_determinism(self._torch)
        self._inference_id = _identity(inference_id, name="inference_id")
        self._model_identity_hash = _identity(model_identity_hash, name="model_identity_hash")
        self._chart_identity_hash = _identity(chart_identity_hash, name="chart_identity_hash")
        self._current_mask_hash = _identity(current_mask_hash, name="current_mask_hash")
        self._proposal_generation = _nonnegative_int(
            proposal_generation,
            name="proposal_generation",
        )
        self._discovery_geometry_source_hash = _identity(
            discovery_geometry_source_hash,
            name="discovery_geometry_source_hash",
        )
        self._witness_stream_source_hash = _identity(
            witness_stream_source_hash,
            name="witness_stream_source_hash",
        )
        self._literal_j_stream_source_hash = _identity(
            literal_j_stream_source_hash,
            name="literal_j_stream_source_hash",
        )
        self._route_axis_hash = _identity(route_axis_hash, name="route_axis_hash")
        self._route_count = _positive_int(route_count, name="route_count")
        self._direction_ids = _identity_tuple(direction_ids, name="direction_ids")
        if descriptor_axis is not None:
            if (
                descriptor_axis.route_axis_hash != self._route_axis_hash
                or descriptor_axis.route_count != self._route_count
                or descriptor_axis.direction_ids != self._direction_ids
            ):
                _refuse(
                    V1LiteralJFixedDirectionProjectionCUDARefusalCode.INVALID_DESIGN,
                    "private-primal descriptor axis left the projection route/direction axis",
                )
        direction_count = len(self._direction_ids)
        direction_source = (
            leased_directions.directions_for_literal_j_projection()
            if leased_directions is not None
            else directions
        )
        directions_tensor = _require_cuda_tensor(
            self._torch,
            direction_source,
            name="directions",
            device=self._device,
            ndim=2,
        )
        if tuple(int(item) for item in directions_tensor.shape) != (
            direction_count,
            self._route_count,
        ):
            _refuse(
                V1LiteralJFixedDirectionProjectionCUDARefusalCode.INVALID_DESIGN,
                "directions must have shape (direction_count, route_count)",
            )
        bound_tensor = _require_cuda_tensor(
            self._torch,
            literal_j_abs_bound,
            name="literal_j_abs_bound",
            device=self._device,
            ndim=1,
            nonnegative=True,
        )
        if int(bound_tensor.numel()) != self._route_count:
            _refuse(
                V1LiteralJFixedDirectionProjectionCUDARefusalCode.INVALID_DESIGN,
                "literal_j_abs_bound must cover the route axis",
            )
        projection_bound_tensor = _require_cuda_tensor(
            self._torch,
            projection_abs_bounds,
            name="projection_abs_bounds",
            device=self._device,
            ndim=1,
            nonnegative=True,
        )
        if int(projection_bound_tensor.numel()) != direction_count:
            _refuse(
                V1LiteralJFixedDirectionProjectionCUDARefusalCode.INVALID_DESIGN,
                "projection_abs_bounds must cover the direction axis",
            )
        self._private_primal_direction_descriptor_axis_hash = (
            None if descriptor_axis is None else descriptor_axis.descriptor_axis_hash
        )
        self._literal_j_abs_bound_source_hash = (
            None
            if literal_j_abs_bound_source_hash is None
            else _identity(
                literal_j_abs_bound_source_hash,
                name="literal_j_abs_bound_source_hash",
            )
        )
        self._projection_abs_bound_source_hash = (
            None
            if projection_abs_bound_source_hash is None
            else _identity(
                projection_abs_bound_source_hash,
                name="projection_abs_bound_source_hash",
            )
        )
        self._forecast = build_v1_literal_j_projection_cuda_hbm_forecast(
            design=self._design,
            inference_id=self._inference_id,
            route_axis_hash=self._route_axis_hash,
            route_count=self._route_count,
            direction_count=direction_count,
            torch_module=self._torch,
            device=self._device,
        )
        if not self._forecast.admitted:
            _refuse(
                V1LiteralJFixedDirectionProjectionCUDARefusalCode.HBM_FORECAST_EXCEEDED,
                "literal-J projection CUDA forecast exceeded available HBM",
            )
        self._directions = directions_tensor.detach().clone().contiguous()
        self._literal_j_abs_bound = bound_tensor.detach().clone().contiguous()
        self._projection_abs_bounds = (
            projection_bound_tensor.detach().clone().contiguous()
        )
        small_domain = (
            self._route_count
            <= self._design.small_domain_reference_hash_route_limit
            and self._design.allow_small_domain_reference_array_hashing
        )
        if self._private_primal_direction_descriptor_axis_hash is not None:
            self._direction_hashes = _structural_direction_hashes(
                route_axis_hash=self._route_axis_hash,
                route_count=self._route_count,
                direction_ids=self._direction_ids,
                private_primal_direction_descriptor_axis_hash=(
                    self._private_primal_direction_descriptor_axis_hash
                ),
            )
        elif small_domain:
            self._direction_hashes = _cuda_direction_hashes(
                self._torch,
                self._directions,
                chunk_size=self._design.hash_chunk_size,
            )
        else:
            _refuse(
                V1LiteralJFixedDirectionProjectionCUDARefusalCode.INVALID_DESIGN,
                "production-scale CUDA projection requires a private-primal "
                "direction descriptor instead of route-scale D2H array hashing",
            )
        self._direction_axis_hash = _projection_axis_hash(
            "fixed_primal_directions",
            tuple(zip(self._direction_ids, self._direction_hashes, strict=True)),
        )
        if (
            self._literal_j_abs_bound_source_hash is not None
            and self._projection_abs_bound_source_hash is not None
        ):
            self._contribution_bound_hashes = _structural_contribution_bound_hashes(
                route_axis_hash=self._route_axis_hash,
                route_count=self._route_count,
                direction_ids=self._direction_ids,
                literal_j_abs_bound_source_hash=(
                    self._literal_j_abs_bound_source_hash
                ),
                projection_abs_bound_source_hash=(
                    self._projection_abs_bound_source_hash
                ),
            )
        elif small_domain:
            self._contribution_bound_hashes = _cuda_contribution_bound_hashes(
                self._torch,
                self._literal_j_abs_bound,
                self._projection_abs_bounds,
                chunk_size=self._design.hash_chunk_size,
            )
        else:
            _refuse(
                V1LiteralJFixedDirectionProjectionCUDARefusalCode.INVALID_DESIGN,
                "production-scale CUDA projection requires bound source "
                "descriptors instead of route-scale D2H bound hashing",
            )
        self._quadratic_bound_values = tuple(
            float(item)
            for item in (
                self._projection_abs_bounds * self._projection_abs_bounds
            )
            .detach()
            .cpu()
            .tolist()
        )
        self._scalar_d2h_count = 3
        stratum_tuple = _identity_tuple(stratum_ids, name="stratum_ids")
        if len(stratum_tuple) != 2:
            _refuse(
                V1LiteralJFixedDirectionProjectionCUDARefusalCode.INVALID_DESIGN,
                "projection CUDA inference requires exactly two strata",
            )
        self._stratum_ids = (stratum_tuple[0], stratum_tuple[1])
        if isinstance(sealed_context_ids_by_stratum, (str, bytes)):
            raise ValueError("sealed contexts must contain two ordered strata")
        sealed = tuple(tuple(row) for row in sealed_context_ids_by_stratum)
        if len(sealed) != 2:
            raise ValueError("sealed contexts must contain exactly two strata")
        first = _identity_tuple(sealed[0], name="sealed_contexts[0]")
        second = _identity_tuple(sealed[1], name="sealed_contexts[1]")
        if set(first).intersection(second):
            raise ValueError("context identities must be unique across strata")
        self._sealed_context_ids = (first, second)
        self._sealed_order_hashes = (
            _projection_axis_hash(
                "sealed_witness_population_order",
                {"stratum_id": self._stratum_ids[0], "context_ids": first},
            ),
            _projection_axis_hash(
                "sealed_witness_population_order",
                {"stratum_id": self._stratum_ids[1], "context_ids": second},
            ),
        )
        self._familywise_alpha = _positive_float(
            familywise_alpha,
            name="familywise_alpha",
        )
        if self._familywise_alpha >= 1.0:
            raise ValueError("familywise_alpha must be smaller than one")
        self._next_indices = [0, 0]
        self._seen_context_ids: set[str] = set()
        self._seen_draw_ids: set[str] = set()
        self._seen_source_measurement_ids: set[str] = set()
        self._stratum_draw_ids: list[list[str]] = [[], []]
        self._stratum_source_ids: list[list[str]] = [[], []]
        self._sum_operator = self._torch.zeros(
            (2, direction_count, self._route_count),
            device=self._device,
            dtype=self._torch.float64,
        )
        self._sum_quadratic = self._torch.zeros(
            (2, direction_count),
            device=self._device,
            dtype=self._torch.float64,
        )
        self._batch_count = 0
        self._row_count = 0
        self._tile_visit_count = 0
        self._finalized = False

    @property
    def observed_counts(self) -> tuple[int, int]:
        return (int(self._next_indices[0]), int(self._next_indices[1]))

    @property
    def forecast(self) -> V1LiteralJFixedDirectionProjectionCUDAHBMForecast:
        return self._forecast

    def update(
        self,
        *,
        stratum_id: str,
        context_id: str,
        draw_id: str,
        source_measurement_id: str,
        literal_j_row: Any,
        model_identity_hash: str,
        chart_identity_hash: str,
        current_mask_hash: str,
        proposal_generation: int,
        literal_j_stream_source_hash: str,
        route_axis_hash: str,
    ) -> None:
        self.update_batch(
            stratum_id=stratum_id,
            context_ids=(context_id,),
            draw_ids=(draw_id,),
            source_measurement_ids=(source_measurement_id,),
            literal_j_rows=literal_j_row,
            model_identity_hash=model_identity_hash,
            chart_identity_hash=chart_identity_hash,
            current_mask_hash=current_mask_hash,
            proposal_generation=proposal_generation,
            literal_j_stream_source_hash=literal_j_stream_source_hash,
            route_axis_hash=route_axis_hash,
        )

    def update_batch(
        self,
        *,
        stratum_id: str,
        context_ids: Sequence[str],
        draw_ids: Sequence[str],
        source_measurement_ids: Sequence[str],
        literal_j_rows: Any,
        model_identity_hash: str,
        chart_identity_hash: str,
        current_mask_hash: str,
        proposal_generation: int,
        literal_j_stream_source_hash: str,
        route_axis_hash: str,
    ) -> None:
        if self._finalized:
            _refuse(
                V1LiteralJFixedDirectionProjectionCUDARefusalCode.FINALIZED,
                "projection CUDA accumulator already finalized",
            )
        if (
            _identity(model_identity_hash, name="model_identity_hash")
            != self._model_identity_hash
            or _identity(chart_identity_hash, name="chart_identity_hash")
            != self._chart_identity_hash
            or _identity(current_mask_hash, name="current_mask_hash")
            != self._current_mask_hash
            or _nonnegative_int(proposal_generation, name="proposal_generation")
            != self._proposal_generation
            or _identity(literal_j_stream_source_hash, name="literal_j_stream_source_hash")
            != self._literal_j_stream_source_hash
            or _identity(route_axis_hash, name="route_axis_hash") != self._route_axis_hash
        ):
            _refuse(
                V1LiteralJFixedDirectionProjectionCUDARefusalCode.STALE_IDENTITY,
                "literal-J row left the frozen projection chart",
            )
        if stratum_id not in self._stratum_ids:
            _refuse(
                V1LiteralJFixedDirectionProjectionCUDARefusalCode.ORDER_MISMATCH,
                "unknown projection stratum",
            )
        stratum_index = self._stratum_ids.index(stratum_id)
        contexts = _identity_tuple(context_ids, name="context_ids")
        draws = _identity_tuple(draw_ids, name="draw_ids")
        sources = _identity_tuple(source_measurement_ids, name="source_measurement_ids")
        if len(contexts) != len(draws) or len(contexts) != len(sources):
            raise ValueError("context, draw, and source axes must have equal length")
        rows = _require_cuda_tensor(
            self._torch,
            literal_j_rows,
            name="literal_j_rows",
            device=self._device,
            ndim=(2 if getattr(literal_j_rows, "ndim", None) == 2 else 1),
        )
        if rows.ndim == 1:
            if int(rows.numel()) != self._route_count:
                _refuse(
                    V1LiteralJFixedDirectionProjectionCUDARefusalCode.INVALID_ROW,
                    "literal-J row shape differs from route axis",
                )
            rows = rows.reshape(1, self._route_count)
        if tuple(int(item) for item in rows.shape) != (len(contexts), self._route_count):
            _refuse(
                V1LiteralJFixedDirectionProjectionCUDARefusalCode.INVALID_ROW,
                "literal-J row batch shape differs from registered context and route axes",
            )
        if len(contexts) > self._design.maximum_context_batch_size:
            _refuse(
                V1LiteralJFixedDirectionProjectionCUDARefusalCode.INVALID_ROW,
                "literal-J row batch exceeds the forecast maximum context batch size",
            )
        next_index = self._next_indices[stratum_index]
        expected = self._sealed_context_ids[stratum_index][
            next_index : next_index + len(contexts)
        ]
        if contexts != expected:
            _refuse(
                V1LiteralJFixedDirectionProjectionCUDARefusalCode.ORDER_MISMATCH,
                "literal-J witness contexts must be consumed in sealed prefix order",
            )
        if (
            set(contexts).intersection(self._seen_context_ids)
            or set(draws).intersection(self._seen_draw_ids)
            or set(sources).intersection(self._seen_source_measurement_ids)
        ):
            _refuse(
                V1LiteralJFixedDirectionProjectionCUDARefusalCode.DUPLICATE_IDENTITY,
                "literal-J projection witness identities must be unique",
            )
        if bool((self._torch.abs(rows) > self._literal_j_abs_bound).any().item()):
            _refuse(
                V1LiteralJFixedDirectionProjectionCUDARefusalCode.INVALID_ROW,
                "literal-J row exceeds its pre-outcome absolute support bound",
            )
        projections = rows @ self._directions.T
        if bool((self._torch.abs(projections) > self._projection_abs_bounds).any().item()):
            _refuse(
                V1LiteralJFixedDirectionProjectionCUDARefusalCode.INVALID_ROW,
                "literal-J projection exceeds its pre-outcome absolute support bound",
            )
        self._sum_quadratic[stratum_index] += self._torch.sum(
            projections * projections,
            dim=0,
        )
        tile_size = self._design.route_tile_size
        for start in range(0, self._route_count, tile_size):
            stop = min(start + tile_size, self._route_count)
            self._sum_operator[stratum_index, :, start:stop] += (
                projections.T @ rows[:, start:stop]
            )
            self._tile_visit_count += len(contexts)
        self._next_indices[stratum_index] += len(contexts)
        self._seen_context_ids.update(contexts)
        self._seen_draw_ids.update(draws)
        self._seen_source_measurement_ids.update(sources)
        self._stratum_draw_ids[stratum_index].extend(draws)
        self._stratum_source_ids[stratum_index].extend(sources)
        self._batch_count += 1
        self._row_count += len(contexts)
        self._scalar_d2h_count += 2

    def finalize_prefix(self) -> V1LiteralJFixedDirectionProjectionCUDABundle:
        if self._finalized:
            _refuse(
                V1LiteralJFixedDirectionProjectionCUDARefusalCode.FINALIZED,
                "projection CUDA accumulator already finalized",
            )
        observed = self.observed_counts
        if observed[0] <= 0 or observed[1] <= 0:
            _refuse(
                V1LiteralJFixedDirectionProjectionCUDARefusalCode.EMPTY_PREFIX,
                "both strata require nonempty witness prefixes",
            )
        self._finalized = True
        self._torch.cuda.synchronize(self._device)
        means = self._torch.empty_like(self._sum_operator)
        quad_means = self._torch.empty_like(self._sum_quadratic)
        for stratum_index, count in enumerate(observed):
            means[stratum_index] = self._sum_operator[stratum_index] / float(count)
            quad_means[stratum_index] = self._sum_quadratic[stratum_index] / float(count)
        output = 0.5 * (means[0] + means[1])
        quadratic = 0.5 * (quad_means[0] + quad_means[1])
        if not bool(self._torch.isfinite(output).all().item()) or not bool(
            self._torch.isfinite(quadratic).all().item()
        ):
            _refuse(
                V1LiteralJFixedDirectionProjectionCUDARefusalCode.NONFINITE_CUDA_SUMMARY,
                "literal-J CUDA projection produced a nonfinite summary",
            )
        quadratic_values = tuple(float(item) for item in quadratic.detach().cpu().tolist())
        output_hashes = _cuda_output_fingerprint_hashes(
            self._torch,
            output,
            route_axis_hash=self._route_axis_hash,
            direction_ids=self._direction_ids,
            direction_hashes=self._direction_hashes,
            quadratic_values=quadratic_values,
        )
        output_axis = _projection_axis_hash(
            "literal_j_fixed_direction_outputs",
            tuple(
                zip(
                    self._direction_ids,
                    output_hashes,
                    quadratic_values,
                    strict=True,
                )
            ),
        )
        # Canonical simultaneous family: both strata are bounded separately
        # before their registered 1/2 mixture, so the Bonferroni family
        # carries the stratum axis explicitly (matches the CPU inference and
        # the receipt's own check).
        statement_count = 2 * len(self._direction_ids) * (self._route_count + 1)
        per_statement_alpha = self._familywise_alpha / float(statement_count)
        populations = (len(self._sealed_context_ids[0]), len(self._sealed_context_ids[1]))
        sampling_radius = 0.0
        quadratic_radius = 0.0
        contribution_max = _max_contribution_bound(
            self._torch,
            self._literal_j_abs_bound,
            self._projection_abs_bounds,
        )
        quadratic_bound_max = _quadratic_bound_max(self._torch, self._projection_abs_bounds)
        for count, population in zip(observed, populations, strict=True):
            for direction_index in range(len(self._direction_ids)):
                quadratic_radius = max(
                    quadratic_radius,
                    0.5
                    * _serfling_radius(
                        value_range=float(self._quadratic_bound_values[direction_index]),
                        observed=count,
                        population=population,
                        alpha=per_statement_alpha,
                    ),
                )
                sampling_radius = max(
                    sampling_radius,
                    0.5
                    * _serfling_radius(
                        value_range=2.0 * contribution_max,
                        observed=count,
                        population=population,
                        alpha=per_statement_alpha,
                    ),
                )
        numerical_route_radius = (
            0.5
            * sum(_gamma(count) for count in observed)
            * contribution_max
        )
        numerical_quadratic_radius = (
            0.5
            * sum(_gamma(count) for count in observed)
            * quadratic_bound_max
        )
        sampling_radius += numerical_route_radius
        quadratic_radius += numerical_quadratic_radius
        receipt = V1LiteralJFixedDirectionProjectionReceipt(
            inference_id=self._inference_id,
            model_identity_hash=self._model_identity_hash,
            chart_identity_hash=self._chart_identity_hash,
            current_mask_hash=self._current_mask_hash,
            proposal_generation=self._proposal_generation,
            discovery_geometry_source_hash=self._discovery_geometry_source_hash,
            witness_stream_source_hash=self._witness_stream_source_hash,
            literal_j_stream_source_hash=self._literal_j_stream_source_hash,
            route_axis_hash=self._route_axis_hash,
            route_count=self._route_count,
            direction_ids=self._direction_ids,
            direction_hashes=self._direction_hashes,
            direction_axis_hash=self._direction_axis_hash,
            stratum_ids=self._stratum_ids,
            population_counts=populations,
            observed_counts=observed,
            sealed_population_order_hashes=self._sealed_order_hashes,
            observed_context_axis_hashes=(
                _projection_axis_hash(
                    "observed_witness_context_prefix",
                    {
                        "stratum_id": self._stratum_ids[0],
                        "context_ids": self._sealed_context_ids[0][: observed[0]],
                    },
                ),
                _projection_axis_hash(
                    "observed_witness_context_prefix",
                    {
                        "stratum_id": self._stratum_ids[1],
                        "context_ids": self._sealed_context_ids[1][: observed[1]],
                    },
                ),
            ),
            observed_draw_axis_hashes=(
                _projection_axis_hash(
                    "observed_witness_draw_prefix",
                    {
                        "stratum_id": self._stratum_ids[0],
                        "draw_ids": tuple(self._stratum_draw_ids[0]),
                    },
                ),
                _projection_axis_hash(
                    "observed_witness_draw_prefix",
                    {
                        "stratum_id": self._stratum_ids[1],
                        "draw_ids": tuple(self._stratum_draw_ids[1]),
                    },
                ),
            ),
            observed_source_measurement_axis_hashes=(
                _projection_axis_hash(
                    "observed_witness_source_prefix",
                    {
                        "stratum_id": self._stratum_ids[0],
                        "source_measurement_ids": tuple(self._stratum_source_ids[0]),
                    },
                ),
                _projection_axis_hash(
                    "observed_witness_source_prefix",
                    {
                        "stratum_id": self._stratum_ids[1],
                        "source_measurement_ids": tuple(self._stratum_source_ids[1]),
                    },
                ),
            ),
            output_vector_hashes=output_hashes,
            quadratic_values=quadratic_values,
            output_axis_hash=output_axis,
            contribution_bound_vector_hashes=self._contribution_bound_hashes,
            quadratic_bound_values=self._quadratic_bound_values,
            simultaneous_statement_count=statement_count,
            per_statement_two_sided_alpha=float(per_statement_alpha),
            alpha_spent=float(self._familywise_alpha),
            sampling_radius=float(sampling_radius),
            quadratic_sampling_radius=float(quadratic_radius),
            private_primal_direction_descriptor_axis_hash=(
                self._private_primal_direction_descriptor_axis_hash
            ),
            private_primal_direction_lineage_available=(
                self._private_primal_direction_descriptor_axis_hash is not None
            ),
            algorithm_id=LITERAL_J_FIXED_DIRECTION_PROJECTION_ALGORITHM,
            literal_j_equation=LITERAL_J_FIXED_DIRECTION_PROJECTION_EQUATION,
            schema=LITERAL_J_FIXED_DIRECTION_PROJECTION_SCHEMA,
            authority_scope=LITERAL_J_FIXED_DIRECTION_PROJECTION_AUTHORITY_SCOPE,
        )
        self._torch.cuda.synchronize(self._device)
        peak_allocated = int(self._torch.cuda.max_memory_allocated(self._device))
        peak_reserved = int(self._torch.cuda.max_memory_reserved(self._device))
        _verify_measured_hbm(
            forecast=self._forecast,
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
        )
        del means, quad_means, output, quadratic
        telemetry = V1LiteralJFixedDirectionProjectionCUDATelemetry(
            receipt_hash=receipt.receipt_hash,
            forecast_hash=self._forecast.forecast_hash,
            design_hash=self._design.design_hash,
            inference_id=self._inference_id,
            route_axis_hash=self._route_axis_hash,
            device=str(self._device),
            route_count=self._route_count,
            direction_count=len(self._direction_ids),
            observed_counts=observed,
            route_tile_size=self._design.route_tile_size,
            maximum_context_batch_size=self._design.maximum_context_batch_size,
            update_batch_count=self._batch_count,
            context_row_count=self._row_count,
            route_tile_visit_count=self._tile_visit_count,
            literal_j_route_value_visit_count=self._row_count * self._route_count,
            projection_matmul_count=self._batch_count,
            support_reduction_count=2 * self._batch_count,
            fp64_operator_tile_accumulation_count=self._tile_visit_count,
            output_hash_vector_count=len(self._direction_ids),
            output_hash_d2h_tensor_payload_bytes=(
                8
                * len(self._direction_ids)
                * LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_OUTPUT_FINGERPRINT_SCALAR_COUNT
            ),
            scalar_d2h_tensor_payload_bytes=8 * self._scalar_d2h_count,
            peak_allocated_hbm_bytes=peak_allocated,
            peak_reserved_hbm_bytes=peak_reserved,
            numerical_route_radius=float(numerical_route_radius),
            numerical_quadratic_radius=float(numerical_quadratic_radius),
            hbm_forecast_admitted=self._forecast.admitted,
        )
        self._sum_operator = self._torch.empty(0, device=self._device, dtype=self._torch.float64)
        self._sum_quadratic = self._torch.empty(0, device=self._device, dtype=self._torch.float64)
        self._directions = self._torch.empty(
            (0, 0), device=self._device, dtype=self._torch.float64
        )
        self._literal_j_abs_bound = self._torch.empty(
            0, device=self._device, dtype=self._torch.float64
        )
        self._projection_abs_bounds = self._torch.empty(
            0, device=self._device, dtype=self._torch.float64
        )
        return V1LiteralJFixedDirectionProjectionCUDABundle(
            receipt=receipt,
            forecast=self._forecast,
            telemetry=telemetry,
            design=self._design,
        )


__all__ = [
    "LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_ALGORITHM",
    "LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_BUNDLE_SCHEMA",
    "LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_DESIGN_SCHEMA",
    "LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_HBM_CEILING_FRACTION",
    "LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_HBM_FORECAST_SCHEMA",
    "LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_NUMERICAL_ENCLOSURE_ID",
    "LITERAL_J_FIXED_DIRECTION_PROJECTION_CUDA_TELEMETRY_SCHEMA",
    "LITERAL_J_PRIVATE_PRIMAL_DIRECTION_DESCRIPTOR_AXIS_SCHEMA",
    "LITERAL_J_PRIVATE_PRIMAL_DIRECTION_DESCRIPTOR_SCHEMA",
    "V1LiteralJFixedDirectionProjectionCUDAAccumulator",
    "V1LiteralJFixedDirectionProjectionCUDABundle",
    "V1LiteralJFixedDirectionProjectionCUDAExecutionDesign",
    "V1LiteralJFixedDirectionProjectionCUDAHBMForecast",
    "V1LiteralJFixedDirectionProjectionCUDARefusal",
    "V1LiteralJFixedDirectionProjectionCUDARefusalCode",
    "V1LiteralJFixedDirectionProjectionCUDATelemetry",
    "V1LiteralJPrivatePrimalDirectionCUDAHandle",
    "V1LiteralJPrivatePrimalDirectionCUDALease",
    "V1LiteralJPrivatePrimalDirectionDescriptor",
    "V1LiteralJPrivatePrimalDirectionDescriptorAxis",
    "build_v1_literal_j_projection_cuda_hbm_forecast",
]
