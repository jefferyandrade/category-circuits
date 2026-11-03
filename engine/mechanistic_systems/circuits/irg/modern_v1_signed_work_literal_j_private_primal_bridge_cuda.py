"""Source-bound CUDA bridge from signed-work KKT primals to literal-J custody.

The signed-work least-work runtime solves the two certified ``Q`` branches in
separate single-branch executions.  Literal-J replay, by contrast, consumes a
single canonical direction axis ordered ``(spectral, diagonal_l1)``.  This
module is the narrow process-local bridge between those owners:

* both exact theorem-state handles are borrowed through their registered
  ``SIGNED_WORK_DECISION_WITNESS`` operation;
* their relaxed primals are copied device-to-device into one contiguous FP64
  ``(2, R)`` tensor in canonical branch order;
* the existing literal-J private-primal descriptor and handle types receive
  that tensor; and
* only compact, array-free source, memory, and execution receipts persist.

The bridge is CUDA-only in production.  Its memory receipt is planning and
custody evidence, not a KKT-face, C, K, O, commit, terminal, or Pareto
certificate.  No route-scale tensor is copied to or hashed on the host.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
import math
from numbers import Integral, Real
from threading import RLock
from typing import Any, Final, NoReturn

from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

from .contracts import content_hash
from .modern_v1_literal_j_projection_cuda import (
    V1LiteralJFixedDirectionProjectionCUDARefusal,
    V1LiteralJPrivatePrimalDirectionCUDAHandle,
    V1LiteralJPrivatePrimalDirectionCUDALease,
    V1LiteralJPrivatePrimalDirectionDescriptor,
    V1LiteralJPrivatePrimalDirectionDescriptorAxis,
)
from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch
from .modern_v1_signed_work_kkt_cuda import (
    V1SignedWorkLeastWorkCUDAPolicy,
    V1SignedWorkLeastWorkCUDAProblem,
    V1SignedWorkLeastWorkCUDAStateBorrowOperation,
    V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle,
)

try:  # pragma: no cover - environment dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


SIGNED_WORK_LITERAL_J_PRIVATE_PRIMAL_BRIDGE_CUDA_DESIGN_SCHEMA: Final = (
    "partition_free_v1_signed_work_literal_j_private_primal_bridge_cuda_design_v1"
)
SIGNED_WORK_LITERAL_J_PRIVATE_PRIMAL_BRIDGE_SOURCE_SCHEMA: Final = (
    "partition_free_v1_signed_work_literal_j_private_primal_bridge_source_v1"
)
SIGNED_WORK_LITERAL_J_PRIVATE_PRIMAL_BRIDGE_HBM_SCHEMA: Final = (
    "partition_free_v1_signed_work_literal_j_private_primal_bridge_hbm_v1"
)
SIGNED_WORK_LITERAL_J_PRIVATE_PRIMAL_BRIDGE_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_signed_work_literal_j_private_primal_bridge_receipt_v1"
)
SIGNED_WORK_LITERAL_J_PRIVATE_PRIMAL_BRIDGE_EXECUTION_SCHEMA: Final = (
    "partition_free_v1_signed_work_literal_j_private_primal_bridge_execution_v1"
)
SIGNED_WORK_LITERAL_J_PRIVATE_PRIMAL_BRIDGE_CUDA_ALGORITHM: Final = (
    "exact_two_single_branch_signed_work_primal_d2d_stack_literal_j_custody_v1"
)
SIGNED_WORK_LITERAL_J_PRIVATE_PRIMAL_BRIDGE_BRANCH_ORDER: Final = (
    V1RelaxedKKTBranch.SPECTRAL,
    V1RelaxedKKTBranch.DIAGONAL_L1,
)
SIGNED_WORK_LITERAL_J_PRIVATE_PRIMAL_BRIDGE_HBM_CEILING: Final = 0.85
SIGNED_WORK_LITERAL_J_PRIVATE_PRIMAL_BRIDGE_MINIMUM_GUARD_BYTES: Final = (
    64 * 1024 * 1024
)
SIGNED_WORK_LITERAL_J_PRIVATE_PRIMAL_BRIDGE_GUARD_FRACTION: Final = 0.10

_DESIGN_PREFIX: Final = "partition-free-v1-signed-work-literal-j-bridge-design-"
_SOURCE_PREFIX: Final = "partition-free-v1-signed-work-literal-j-bridge-source-"
_RESULT_PAIR_PREFIX: Final = (
    "partition-free-v1-signed-work-literal-j-bridge-result-pair-"
)
_EXECUTION_PAIR_PREFIX: Final = (
    "partition-free-v1-signed-work-literal-j-bridge-execution-pair-"
)
_DEVICE_PREFIX: Final = "partition-free-v1-signed-work-literal-j-bridge-device-"
_FORECAST_PREFIX: Final = "partition-free-v1-signed-work-literal-j-bridge-hbm-"
_RECEIPT_PREFIX: Final = "partition-free-v1-signed-work-literal-j-bridge-receipt-"
_DIRECTION_ID_PREFIX: Final = (
    "partition-free-v1-signed-work-literal-j-private-primal-direction-"
)
_F64_BYTES: Final = 8


class V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    BRANCH_ORDER_MISMATCH = "branch_order_mismatch"
    STALE_LINEAGE = "stale_lineage"
    CUDA_REQUIRED = "cuda_required"
    CPU_FALLBACK_FORBIDDEN = "cpu_fallback_forbidden"
    DETERMINISM_REQUIRED = "determinism_required"
    HBM_FORECAST_EXCEEDED = "hbm_forecast_exceeded"
    MEASURED_HBM_CEILING_EXCEEDED = "measured_hbm_ceiling_exceeded"
    DEVICE_MISMATCH = "device_mismatch"
    CUSTODY_MUTATED = "custody_mutated"
    ACTIVE_LEASE = "active_lease"
    EVICTED = "evicted"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusal(RuntimeError):
    def __init__(
        self,
        code: V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode:
            raise TypeError("bridge refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode,
    message: str,
) -> NoReturn:
    raise V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _identity_pair(value: object, *, name: str) -> tuple[str, str]:
    if type(value) is not tuple or len(value) != 2:
        _refuse(
            V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one exact pair",
        )
    result = tuple(_identity(item, name=f"{name}[{index}]") for index, item in enumerate(value))
    return result  # type: ignore[return-value]


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one positive integer",
        )
    result = int(value)
    if result <= 0:
        _refuse(
            V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.INVALID_TYPE,
            f"{name} must be nonnegative",
        )
    return result


def _fraction(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one finite real",
        )
    result = float(value)
    if not math.isfinite(result) or not (0.0 < result <= 1.0):
        _refuse(
            V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.INVALID_TYPE,
            f"{name} must lie in (0, 1]",
        )
    return result


def _require_torch() -> Any:
    if torch is None:
        _refuse(
            V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.CUDA_REQUIRED,
            "PyTorch is unavailable",
        )
    return torch


@dataclass(frozen=True, slots=True)
class V1SignedWorkLiteralJPrivatePrimalBridgeCUDAExecutionDesign:
    design_id: str = "modern-v1-signed-work-literal-j-private-primal-bridge-cuda"
    hbm_ceiling_fraction: float = (
        SIGNED_WORK_LITERAL_J_PRIVATE_PRIMAL_BRIDGE_HBM_CEILING
    )
    allocator_guard_fraction: float = (
        SIGNED_WORK_LITERAL_J_PRIVATE_PRIMAL_BRIDGE_GUARD_FRACTION
    )
    minimum_allocator_guard_bytes: int = (
        SIGNED_WORK_LITERAL_J_PRIVATE_PRIMAL_BRIDGE_MINIMUM_GUARD_BYTES
    )
    require_strict_torch_determinism: bool = True
    production_cuda_required: bool = True
    route_scale_d2h_forbidden: bool = True
    algorithm_id: str = SIGNED_WORK_LITERAL_J_PRIVATE_PRIMAL_BRIDGE_CUDA_ALGORITHM
    schema: str = SIGNED_WORK_LITERAL_J_PRIVATE_PRIMAL_BRIDGE_CUDA_DESIGN_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "design_id", _identity(self.design_id, name="design ID"))
        ceiling = _fraction(self.hbm_ceiling_fraction, name="HBM ceiling fraction")
        guard = _fraction(self.allocator_guard_fraction, name="allocator guard fraction")
        if ceiling > SIGNED_WORK_LITERAL_J_PRIVATE_PRIMAL_BRIDGE_HBM_CEILING:
            _refuse(
                V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.AUTHORITY_VIOLATION,
                "private-primal bridge HBM ceiling cannot exceed 85 percent",
            )
        object.__setattr__(self, "hbm_ceiling_fraction", ceiling)
        object.__setattr__(self, "allocator_guard_fraction", guard)
        object.__setattr__(
            self,
            "minimum_allocator_guard_bytes",
            _nonnegative_int(
                self.minimum_allocator_guard_bytes,
                name="minimum allocator guard bytes",
            ),
        )
        for name in (
            "require_strict_torch_determinism",
            "production_cuda_required",
            "route_scale_d2h_forbidden",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact bool")
        if (
            self.production_cuda_required is not True
            or self.route_scale_d2h_forbidden is not True
            or self.algorithm_id
            != SIGNED_WORK_LITERAL_J_PRIVATE_PRIMAL_BRIDGE_CUDA_ALGORITHM
            or self.schema
            != SIGNED_WORK_LITERAL_J_PRIVATE_PRIMAL_BRIDGE_CUDA_DESIGN_SCHEMA
        ):
            _refuse(
                V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.AUTHORITY_VIOLATION,
                "private-primal bridge design weakened a closed production gate",
            )

    @property
    def design_hash(self) -> str:
        return content_hash(self, prefix=_DESIGN_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkLiteralJPrivatePrimalBridgeSourceReceipt:
    """Array-free registration of the exact two single-branch source owners."""

    problem_hash: str
    policy_hash: str
    source_relaxed_problem_hash: str
    source_registration_hash: str
    registered_d_coordinate_hash: str
    signed_work_operator_receipt_hash: str
    signed_work_operator_forecast_hash: str
    source_geometry_hash: str
    route_axis_hash: str
    route_id_axis_hash: str
    eligible_mask_identity_hash: str
    required_mask_identity_hash: str
    route_count: int
    branch_order: tuple[str, str]
    branch_result_hashes: tuple[str, str]
    branch_receipt_hashes: tuple[str, str]
    source_execution_hashes: tuple[str, str]
    source_forecast_hashes: tuple[str, str]
    source_state_identity_hashes: tuple[str, str]
    production_cuda_executions: bool = True
    source_state_borrow_operation: str = "signed_work_decision_witness"
    route_scale_tensor_payload_omitted: bool = True
    kkt_face_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    pareto_authority: bool = False
    schema: str = SIGNED_WORK_LITERAL_J_PRIVATE_PRIMAL_BRIDGE_SOURCE_SCHEMA
    _source_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "problem_hash",
            "policy_hash",
            "source_relaxed_problem_hash",
            "source_registration_hash",
            "registered_d_coordinate_hash",
            "signed_work_operator_receipt_hash",
            "signed_work_operator_forecast_hash",
            "source_geometry_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "eligible_mask_identity_hash",
            "required_mask_identity_hash",
            "source_state_borrow_operation",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(self, "route_count", _positive_int(self.route_count, name="route count"))
        for name in (
            "branch_result_hashes",
            "branch_receipt_hashes",
            "source_execution_hashes",
            "source_forecast_hashes",
            "source_state_identity_hashes",
        ):
            object.__setattr__(self, name, _identity_pair(getattr(self, name), name=name))
        if self.branch_order != ("spectral", "diagonal_l1"):
            _refuse(
                V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.BRANCH_ORDER_MISMATCH,
                "source receipt branch order must be spectral then diagonal_l1",
            )
        bool_fields = (
            "production_cuda_executions",
            "route_scale_tensor_payload_omitted",
            "kkt_face_authority",
            "c_stream_authority",
            "calibration_authority",
            "oracle_authority",
            "mask_commit_authority",
            "pareto_authority",
        )
        if any(type(getattr(self, name)) is not bool for name in bool_fields):
            raise TypeError("source receipt authority fields must be exact booleans")
        if (
            self.production_cuda_executions is not True
            or self.route_scale_tensor_payload_omitted is not True
            or self.source_state_borrow_operation != "signed_work_decision_witness"
            or any(
                getattr(self, name)
                for name in (
                    "kkt_face_authority",
                    "c_stream_authority",
                    "calibration_authority",
                    "oracle_authority",
                    "mask_commit_authority",
                    "pareto_authority",
                )
            )
            or self.schema != SIGNED_WORK_LITERAL_J_PRIVATE_PRIMAL_BRIDGE_SOURCE_SCHEMA
        ):
            _refuse(
                V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.AUTHORITY_VIOLATION,
                "source receipt crossed its planning/custody authority boundary",
            )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
        }

    def assert_integrity(self) -> str:
        observed = content_hash(self._payload(), prefix=_SOURCE_PREFIX)
        if self._source_hash is not None and observed != self._source_hash:
            _refuse(
                V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.CUSTODY_MUTATED,
                "source receipt was mutated",
            )
        object.__setattr__(self, "_source_hash", observed)
        return observed

    @property
    def source_hash(self) -> str:
        return self.assert_integrity()

    @property
    def joint_result_hash(self) -> str:
        return content_hash(
            {
                "problem_hash": self.problem_hash,
                "policy_hash": self.policy_hash,
                "branch_order": self.branch_order,
                "branch_result_hashes": self.branch_result_hashes,
                "branch_receipt_hashes": self.branch_receipt_hashes,
            },
            prefix=_RESULT_PAIR_PREFIX,
        )

    @property
    def joint_execution_hash(self) -> str:
        return content_hash(
            {
                "source_hash": self.source_hash,
                "source_execution_hashes": self.source_execution_hashes,
                "source_forecast_hashes": self.source_forecast_hashes,
                "source_state_identity_hashes": self.source_state_identity_hashes,
            },
            prefix=_EXECUTION_PAIR_PREFIX,
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkLiteralJPrivatePrimalBridgeCUDAHBMForecast:
    design_hash: str
    source_hash: str
    device: str
    device_identity_hash: str
    route_count: int
    borrowed_source_primal_bytes: int
    stacked_direction_bytes: int
    allocator_guard_bytes: int
    incremental_bytes_upper: int
    current_allocated_device_bytes: int
    current_reserved_device_bytes: int
    device_free_memory_bytes: int
    device_total_memory_bytes: int
    current_global_occupied_bytes: int
    projected_global_occupied_bytes_upper: int
    hbm_ceiling_fraction: float
    allocator_guard_fraction: float
    minimum_allocator_guard_bytes: int
    hbm_ceiling_bytes: int
    admitted: bool
    planning_and_custody_authority_only: bool = True
    kkt_face_authority: bool = False
    c_stream_authority: bool = False
    mask_commit_authority: bool = False
    pareto_authority: bool = False
    schema: str = SIGNED_WORK_LITERAL_J_PRIVATE_PRIMAL_BRIDGE_HBM_SCHEMA
    _forecast_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    _admission_available_bytes: InitVar[int | None] = None

    def __post_init__(self, _admission_available_bytes: int | None) -> None:
        for name in ("design_hash", "source_hash", "device", "device_identity_hash"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(self, "route_count", _positive_int(self.route_count, name="route count"))
        names = (
            "borrowed_source_primal_bytes",
            "stacked_direction_bytes",
            "allocator_guard_bytes",
            "incremental_bytes_upper",
            "current_allocated_device_bytes",
            "current_reserved_device_bytes",
            "device_free_memory_bytes",
            "device_total_memory_bytes",
            "current_global_occupied_bytes",
            "projected_global_occupied_bytes_upper",
            "hbm_ceiling_bytes",
        )
        for name in names:
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name=name))
        ceiling = _fraction(self.hbm_ceiling_fraction, name="HBM ceiling fraction")
        guard_fraction = _fraction(
            self.allocator_guard_fraction,
            name="allocator guard fraction",
        )
        object.__setattr__(self, "hbm_ceiling_fraction", ceiling)
        object.__setattr__(self, "allocator_guard_fraction", guard_fraction)
        object.__setattr__(
            self,
            "minimum_allocator_guard_bytes",
            _nonnegative_int(
                self.minimum_allocator_guard_bytes,
                name="minimum allocator guard bytes",
            ),
        )
        expected_stacked = 2 * self.route_count * _F64_BYTES
        expected_guard = max(
            self.minimum_allocator_guard_bytes,
            math.ceil(expected_stacked * guard_fraction),
        )
        expected_incremental = expected_stacked + expected_guard
        expected_occupied = self.device_total_memory_bytes - self.device_free_memory_bytes
        expected_projected = expected_occupied + expected_incremental
        expected_ceiling = math.floor(self.device_total_memory_bytes * ceiling)
        available = (
            self.device_free_memory_bytes + max(
                0, self.current_reserved_device_bytes - self.current_allocated_device_bytes
            )
            if _admission_available_bytes is None else _admission_available_bytes
        )
        expected_admitted = bool(
            expected_incremental <= available
            and self.current_allocated_device_bytes + expected_incremental <= expected_ceiling
        )
        if (
            self.borrowed_source_primal_bytes != expected_stacked
            or self.stacked_direction_bytes != expected_stacked
            or self.allocator_guard_bytes != expected_guard
            or self.incremental_bytes_upper != expected_incremental
            or self.current_global_occupied_bytes != expected_occupied
            or self.projected_global_occupied_bytes_upper != expected_projected
            or self.hbm_ceiling_bytes != expected_ceiling
            or self.admitted is not expected_admitted
            or self.current_allocated_device_bytes > self.current_reserved_device_bytes
            or self.current_reserved_device_bytes > self.device_total_memory_bytes
            or self.device_free_memory_bytes > self.device_total_memory_bytes
        ):
            _refuse(
                V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.STALE_LINEAGE,
                "bridge HBM forecast does not replay its exact allocation formula",
            )
        for name in (
            "planning_and_custody_authority_only",
            "kkt_face_authority",
            "c_stream_authority",
            "mask_commit_authority",
            "pareto_authority",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError("HBM authority fields must be exact booleans")
        if (
            self.planning_and_custody_authority_only is not True
            or self.kkt_face_authority
            or self.c_stream_authority
            or self.mask_commit_authority
            or self.pareto_authority
            or self.schema != SIGNED_WORK_LITERAL_J_PRIVATE_PRIMAL_BRIDGE_HBM_SCHEMA
        ):
            _refuse(
                V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.AUTHORITY_VIOLATION,
                "HBM forecast crossed planning/custody authority",
            )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
        }

    def assert_integrity(self) -> str:
        observed = content_hash(self._payload(), prefix=_FORECAST_PREFIX)
        if self._forecast_hash is not None and observed != self._forecast_hash:
            _refuse(
                V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.CUSTODY_MUTATED,
                "bridge HBM forecast was mutated",
            )
        object.__setattr__(self, "_forecast_hash", observed)
        return observed

    @property
    def forecast_hash(self) -> str:
        return self.assert_integrity()

    @property
    def identity_hash(self) -> str:
        """Forecast identity without live allocator occupancy.

        The occupancy fields are telemetry of the moment the forecast was
        taken; two issuances of the same bridge on the same source differ in
        them, so the execution identity binds the forecast through this
        occupancy-free projection.  ``forecast_hash`` still covers the full
        payload for custody.
        """

        payload = dict(self._payload())
        for name in (
            "current_allocated_device_bytes",
            "current_reserved_device_bytes",
            "device_free_memory_bytes",
            "current_global_occupied_bytes",
            "projected_global_occupied_bytes_upper",
            "admitted",
        ):
            payload.pop(name, None)
        return content_hash(payload, prefix=_FORECAST_PREFIX + "identity-")


def build_v1_signed_work_literal_j_private_primal_bridge_cuda_hbm_forecast(
    *,
    design: V1SignedWorkLiteralJPrivatePrimalBridgeCUDAExecutionDesign,
    source_receipt: V1SignedWorkLiteralJPrivatePrimalBridgeSourceReceipt,
    device: str,
    device_identity_hash: str,
    current_allocated_device_bytes: int,
    current_reserved_device_bytes: int,
    device_free_memory_bytes: int,
    device_total_memory_bytes: int,
    _admission_available_bytes: int | None = None,
) -> V1SignedWorkLiteralJPrivatePrimalBridgeCUDAHBMForecast:
    """Replayable forecast builder; public execution supplies live measurements."""

    if type(design) is not V1SignedWorkLiteralJPrivatePrimalBridgeCUDAExecutionDesign:
        raise TypeError("bridge HBM forecast requires an exact design")
    if type(source_receipt) is not V1SignedWorkLiteralJPrivatePrimalBridgeSourceReceipt:
        raise TypeError("bridge HBM forecast requires an exact source receipt")
    source_receipt.assert_integrity()
    route_count = source_receipt.route_count
    stacked = 2 * route_count * _F64_BYTES
    guard = max(
        design.minimum_allocator_guard_bytes,
        math.ceil(stacked * design.allocator_guard_fraction),
    )
    incremental = stacked + guard
    total = _positive_int(device_total_memory_bytes, name="device total memory bytes")
    free = _nonnegative_int(device_free_memory_bytes, name="device free memory bytes")
    allocated = _nonnegative_int(
        current_allocated_device_bytes,
        name="current allocated device bytes",
    )
    reserved = _nonnegative_int(
        current_reserved_device_bytes,
        name="current reserved device bytes",
    )
    occupied = total - free
    projected = occupied + incremental
    ceiling = math.floor(total * design.hbm_ceiling_fraction)
    available = (free + max(0, reserved - allocated)
                 if _admission_available_bytes is None else _admission_available_bytes)
    return V1SignedWorkLiteralJPrivatePrimalBridgeCUDAHBMForecast(
        design_hash=design.design_hash,
        source_hash=source_receipt.source_hash,
        device=_identity(device, name="device"),
        device_identity_hash=_identity(device_identity_hash, name="device identity hash"),
        route_count=route_count,
        borrowed_source_primal_bytes=stacked,
        stacked_direction_bytes=stacked,
        allocator_guard_bytes=guard,
        incremental_bytes_upper=incremental,
        current_allocated_device_bytes=allocated,
        current_reserved_device_bytes=reserved,
        device_free_memory_bytes=free,
        device_total_memory_bytes=total,
        current_global_occupied_bytes=occupied,
        projected_global_occupied_bytes_upper=projected,
        hbm_ceiling_fraction=design.hbm_ceiling_fraction,
        allocator_guard_fraction=design.allocator_guard_fraction,
        minimum_allocator_guard_bytes=design.minimum_allocator_guard_bytes,
        hbm_ceiling_bytes=ceiling,
        admitted=bool(incremental <= available and allocated + incremental <= ceiling),
        _admission_available_bytes=_admission_available_bytes,
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkLiteralJPrivatePrimalBridgeCUDAReceipt:
    source_hash: str
    joint_result_hash: str
    joint_execution_hash: str
    design_hash: str
    forecast_hash: str
    descriptor_axis_hash: str
    direction_ids: tuple[str, str]
    direction_descriptor_hashes: tuple[str, str]
    route_axis_hash: str
    route_count: int
    device: str
    device_identity_hash: str
    source_borrow_operations: tuple[str, str]
    device_to_device_copied_bytes: int
    route_scale_d2h_array_bytes: int
    compact_scalar_d2h_bytes: int
    post_stack_allocated_device_bytes: int
    post_stack_reserved_device_bytes: int
    post_stack_free_device_bytes: int
    post_stack_global_occupied_bytes: int
    source_leases_released: bool
    contiguous_fp64_canonical_branch_stack: bool
    hbm_forecast_admitted: bool
    measured_hbm_ceiling_respected: bool
    planning_and_custody_authority_only: bool = True
    kkt_face_authority: bool = False
    c_stream_authority: bool = False
    fqr_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = SIGNED_WORK_LITERAL_J_PRIVATE_PRIMAL_BRIDGE_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "source_hash",
            "joint_result_hash",
            "joint_execution_hash",
            "design_hash",
            "forecast_hash",
            "descriptor_axis_hash",
            "route_axis_hash",
            "device",
            "device_identity_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in ("direction_ids", "direction_descriptor_hashes", "source_borrow_operations"):
            object.__setattr__(self, name, _identity_pair(getattr(self, name), name=name))
        object.__setattr__(self, "route_count", _positive_int(self.route_count, name="route count"))
        for name in (
            "device_to_device_copied_bytes",
            "route_scale_d2h_array_bytes",
            "compact_scalar_d2h_bytes",
            "post_stack_allocated_device_bytes",
            "post_stack_reserved_device_bytes",
            "post_stack_free_device_bytes",
            "post_stack_global_occupied_bytes",
        ):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name=name))
        expected_bytes = 2 * self.route_count * _F64_BYTES
        if (
            self.source_borrow_operations
            != ("signed_work_decision_witness", "signed_work_decision_witness")
            or self.device_to_device_copied_bytes != expected_bytes
            or self.route_scale_d2h_array_bytes != 0
            or self.compact_scalar_d2h_bytes != 0
            or self.post_stack_allocated_device_bytes
            > self.post_stack_reserved_device_bytes
        ):
            _refuse(
                V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.STALE_LINEAGE,
                "bridge execution counters do not replay",
            )
        bool_fields = (
            "source_leases_released",
            "contiguous_fp64_canonical_branch_stack",
            "hbm_forecast_admitted",
            "measured_hbm_ceiling_respected",
            "planning_and_custody_authority_only",
            "kkt_face_authority",
            "c_stream_authority",
            "fqr_authority",
            "calibration_authority",
            "oracle_authority",
            "mask_commit_authority",
            "terminal_authority",
            "pareto_authority",
        )
        if any(type(getattr(self, name)) is not bool for name in bool_fields):
            raise TypeError("bridge receipt authority fields must be exact booleans")
        if (
            self.source_leases_released is not True
            or self.contiguous_fp64_canonical_branch_stack is not True
            or self.hbm_forecast_admitted is not True
            or self.measured_hbm_ceiling_respected is not True
            or self.planning_and_custody_authority_only is not True
            or any(
                getattr(self, name)
                for name in (
                    "kkt_face_authority",
                    "c_stream_authority",
                    "fqr_authority",
                    "calibration_authority",
                    "oracle_authority",
                    "mask_commit_authority",
                    "terminal_authority",
                    "pareto_authority",
                )
            )
            or self.schema != SIGNED_WORK_LITERAL_J_PRIVATE_PRIMAL_BRIDGE_RECEIPT_SCHEMA
        ):
            _refuse(
                V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.AUTHORITY_VIOLATION,
                "bridge receipt crossed its planning/custody authority boundary",
            )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
        }

    def assert_integrity(self) -> str:
        observed = content_hash(self._payload(), prefix=_RECEIPT_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            _refuse(
                V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.CUSTODY_MUTATED,
                "bridge execution receipt was mutated",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    @property
    def identity_hash(self) -> str:
        """Receipt identity without allocator occupancy telemetry.

        ``post_stack_*`` occupancy and the full ``forecast_hash`` differ
        between replays of the same source; the execution identity binds the
        receipt through this projection (forecast via its own identity hash).
        """

        payload = dict(self._payload())
        for name in (
            "forecast_hash",
            "post_stack_allocated_device_bytes",
            "post_stack_reserved_device_bytes",
            "post_stack_free_device_bytes",
            "post_stack_global_occupied_bytes",
            "hbm_forecast_admitted",
            "measured_hbm_ceiling_respected",
        ):
            payload.pop(name, None)
        payload["forecast_identity_hash"] = None
        return content_hash(payload, prefix=_RECEIPT_PREFIX + "identity-")


def _direction_id(
    *,
    branch: V1RelaxedKKTBranch,
    source_receipt: V1SignedWorkLiteralJPrivatePrimalBridgeSourceReceipt,
    design_hash: str,
) -> str:
    suffix = content_hash(
        {
            "branch": branch.value,
            "source_hash": source_receipt.source_hash,
            "joint_result_hash": source_receipt.joint_result_hash,
            "joint_execution_hash": source_receipt.joint_execution_hash,
            "design_hash": design_hash,
            "route_axis_hash": source_receipt.route_axis_hash,
            "route_count": source_receipt.route_count,
        },
        prefix=_DIRECTION_ID_PREFIX,
    )
    return f"signed-work-kkt-private-primal:{branch.value}:{suffix}"


def build_v1_signed_work_literal_j_private_primal_descriptor_axis(
    *,
    source_receipt: V1SignedWorkLiteralJPrivatePrimalBridgeSourceReceipt,
    design: V1SignedWorkLiteralJPrivatePrimalBridgeCUDAExecutionDesign,
) -> V1LiteralJPrivatePrimalDirectionDescriptorAxis:
    """Build the canonical structural axis without touching route tensors."""

    if type(source_receipt) is not V1SignedWorkLiteralJPrivatePrimalBridgeSourceReceipt:
        raise TypeError("descriptor bridge requires an exact source receipt")
    if type(design) is not V1SignedWorkLiteralJPrivatePrimalBridgeCUDAExecutionDesign:
        raise TypeError("descriptor bridge requires an exact execution design")
    source_receipt.assert_integrity()
    descriptors: list[V1LiteralJPrivatePrimalDirectionDescriptor] = []
    for index, branch in enumerate(SIGNED_WORK_LITERAL_J_PRIVATE_PRIMAL_BRIDGE_BRANCH_ORDER):
        descriptors.append(
            V1LiteralJPrivatePrimalDirectionDescriptor(
                direction_id=_direction_id(
                    branch=branch,
                    source_receipt=source_receipt,
                    design_hash=design.design_hash,
                ),
                branch=branch.value,  # type: ignore[arg-type]
                source_problem_hash=source_receipt.problem_hash,
                source_policy_hash=source_receipt.policy_hash,
                source_result_hash=source_receipt.joint_result_hash,
                source_branch_result_hash=source_receipt.branch_result_hashes[index],
                source_branch_receipt_hash=source_receipt.branch_receipt_hashes[index],
                source_execution_receipt_hash=source_receipt.source_hash,
                source_execution_design_hash=design.design_hash,
                route_axis_hash=source_receipt.route_axis_hash,
                route_count=source_receipt.route_count,
            )
        )
    return V1LiteralJPrivatePrimalDirectionDescriptorAxis.from_descriptors(
        descriptors[0],
        descriptors[1],
    )


def _register_source_pair(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    spectral_execution: V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle,
    diagonal_l1_execution: V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle,
) -> V1SignedWorkLiteralJPrivatePrimalBridgeSourceReceipt:
    if type(problem) is not V1SignedWorkLeastWorkCUDAProblem:
        raise TypeError("private-primal bridge requires an exact least-work problem")
    if type(policy) is not V1SignedWorkLeastWorkCUDAPolicy:
        raise TypeError("private-primal bridge requires an exact least-work policy")
    executions = (spectral_execution, diagonal_l1_execution)
    if any(
        type(value) is not V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle
        for value in executions
    ):
        raise TypeError("private-primal bridge requires two exact single-branch executions")
    if tuple(value.branch for value in executions) != (
        V1RelaxedKKTBranch.SPECTRAL,
        V1RelaxedKKTBranch.DIAGONAL_L1,
    ):
        _refuse(
            V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.BRANCH_ORDER_MISMATCH,
            "single-branch executions must be supplied spectral then diagonal_l1",
        )
    problem_hash = problem.assert_integrity()
    policy_hash = policy.policy_hash
    route_count = problem.source_problem.route_count
    route_axis_hash = problem.source_problem.route_axis_hash
    operator = problem.signed_work_operator
    operator.receipt.assert_integrity()
    for execution, branch in zip(
        executions,
        SIGNED_WORK_LITERAL_J_PRIVATE_PRIMAL_BRIDGE_BRANCH_ORDER,
        strict=True,
    ):
        state = execution.state_handle
        if (
            execution.result.problem_hash != problem_hash
            or execution.result.policy_hash != policy_hash
            or execution.branch is not branch
            or execution.branch_receipt.branch is not branch
            or execution.branch_receipt.problem_hash != problem_hash
            or execution.branch_receipt.policy_hash != policy_hash
            or execution.forecast.problem_hash != problem_hash
            or execution.forecast.policy_hash != policy_hash
            or execution.forecast.operator_forecast_hash
            != operator.forecast.forecast_hash
            or execution.forecast.route_count != route_count
            or execution.result.branch_converged is not True
            or execution.result.relaxed_state_available is not True
            or state is None
            or execution.state_identity_hash != state.identity_hash
            or state.problem_hash != problem_hash
            or state.policy_hash != policy_hash
            or state.branch_order != (branch,)
            or state.branch_receipt_hashes != (execution.branch_receipt.receipt_hash,)
            or state.route_count != route_count
            or state.eligible_mask_identity_hash != problem.eligible_mask_identity_hash
            or state.required_mask_identity_hash != problem.required_mask_identity_hash
            or state.signed_work_decision_dual_custody_available is not True
        ):
            _refuse(
                V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.STALE_LINEAGE,
                f"{branch.value} execution does not close against its exact source state",
            )
    # Device mode is checked only after the complete typed lineage.  This
    # prevents a CPU/reference source from hiding a crossed problem/axis in
    # bounded adversarial tests and preserves the same fail-closed order in
    # production.
    if (
        problem.source_bundle_authoritative is not True
        or policy.production_cuda_required is not True
        or policy.allow_private_cpu_reference is not False
        or policy.require_deterministic_algorithms is not True
        or operator.receipt.production_cuda_execution is not True
        or operator.receipt.private_cpu_reference_execution is not False
    ):
        _refuse(
            V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.CPU_FALLBACK_FORBIDDEN,
            "bridge source problem, policy, and signed-work operator must be production CUDA",
        )
    for execution in executions:
        if (
            execution.production_cuda_execution is not True
            or execution.private_cpu_reference_execution is not False
            or execution.branch_receipt.production_cuda_execution is not True
            or execution.branch_receipt.private_cpu_reference_execution is not False
        ):
            _refuse(
                V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.CPU_FALLBACK_FORBIDDEN,
                "production private-primal bridge refuses CPU/reference execution custody",
            )
    return V1SignedWorkLiteralJPrivatePrimalBridgeSourceReceipt(
        problem_hash=problem_hash,
        policy_hash=policy_hash,
        source_relaxed_problem_hash=problem.source_problem.problem_hash,
        source_registration_hash=problem.source_registration_hash,
        registered_d_coordinate_hash=problem.registered_d_coordinate_hash,
        signed_work_operator_receipt_hash=operator.receipt_hash,
        signed_work_operator_forecast_hash=operator.forecast.forecast_hash,
        source_geometry_hash=problem.source_problem.source_geometry_hash,
        route_axis_hash=route_axis_hash,
        route_id_axis_hash=operator.receipt.route_id_axis_hash,
        eligible_mask_identity_hash=problem.eligible_mask_identity_hash,
        required_mask_identity_hash=problem.required_mask_identity_hash,
        route_count=route_count,
        branch_order=("spectral", "diagonal_l1"),
        branch_result_hashes=tuple(value.result.result_hash for value in executions),  # type: ignore[arg-type]
        branch_receipt_hashes=tuple(value.branch_receipt.receipt_hash for value in executions),  # type: ignore[arg-type]
        source_execution_hashes=tuple(value.execution_hash for value in executions),  # type: ignore[arg-type]
        source_forecast_hashes=tuple(value.forecast.forecast_hash for value in executions),  # type: ignore[arg-type]
        source_state_identity_hashes=tuple(value.state_identity_hash for value in executions),  # type: ignore[arg-type]
    )


def _normalized_cuda_device(torch_module: Any, value: Any) -> Any:
    device = torch_module.device(value)
    if device.type != "cuda":
        _refuse(
            V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.CUDA_REQUIRED,
            "private-primal bridge requires CUDA theorem-state custody",
        )
    index = torch_module.cuda.current_device() if device.index is None else int(device.index)
    return torch_module.device("cuda", index)


def _device_identity(torch_module: Any, device: Any) -> tuple[str, str]:
    normalized = _normalized_cuda_device(torch_module, device)
    index = int(normalized.index)
    properties = torch_module.cuda.get_device_properties(normalized)
    payload = {
        "backend": "torch_cuda",
        "device": str(normalized),
        "index": index,
        "name": str(properties.name),
        "compute_capability": (int(properties.major), int(properties.minor)),
        "total_memory": int(properties.total_memory),
        "uuid": str(getattr(properties, "uuid", "unavailable")),
    }
    return str(normalized), content_hash(payload, prefix=_DEVICE_PREFIX)


def _tensor_guard(value: Any) -> tuple[object, ...]:
    storage = value.untyped_storage()
    return (
        str(value.device),
        str(value.dtype),
        tuple(int(item) for item in value.shape),
        tuple(int(item) for item in value.stride()),
        int(value.storage_offset()),
        int(value.data_ptr()),
        int(storage.data_ptr()),
        int(getattr(value, "_version", 0)),
        bool(value.requires_grad),
        bool(value.is_contiguous()),
    )


class V1SignedWorkLiteralJPrivatePrimalBridgeCUDAExecution:
    """Process-local owner of the derived exact literal-J direction handle."""

    __slots__ = (
        "_descriptor_axis",
        "_direction_handle",
        "_evicted",
        "_forecast",
        "_lock",
        "_poisoned",
        "_receipt",
        "_source_receipt",
        "_storage_guard",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        source_receipt: V1SignedWorkLiteralJPrivatePrimalBridgeSourceReceipt,
        forecast: V1SignedWorkLiteralJPrivatePrimalBridgeCUDAHBMForecast,
        receipt: V1SignedWorkLiteralJPrivatePrimalBridgeCUDAReceipt,
        descriptor_axis: V1LiteralJPrivatePrimalDirectionDescriptorAxis,
        direction_handle: V1LiteralJPrivatePrimalDirectionCUDAHandle,
        storage_guard: tuple[object, ...],
    ) -> None:
        expected = (
            (source_receipt, V1SignedWorkLiteralJPrivatePrimalBridgeSourceReceipt),
            (forecast, V1SignedWorkLiteralJPrivatePrimalBridgeCUDAHBMForecast),
            (receipt, V1SignedWorkLiteralJPrivatePrimalBridgeCUDAReceipt),
            (descriptor_axis, V1LiteralJPrivatePrimalDirectionDescriptorAxis),
            (direction_handle, V1LiteralJPrivatePrimalDirectionCUDAHandle),
        )
        if any(type(value) is not owner for value, owner in expected):
            raise TypeError("private-primal bridge execution requires exact component types")
        if type(storage_guard) is not tuple or not storage_guard:
            raise TypeError("private-primal bridge execution requires a storage guard")
        source_receipt.assert_integrity()
        forecast.assert_integrity()
        receipt.assert_integrity()
        if (
            forecast.source_hash != source_receipt.source_hash
            or receipt.source_hash != source_receipt.source_hash
            or receipt.joint_result_hash != source_receipt.joint_result_hash
            or receipt.joint_execution_hash != source_receipt.joint_execution_hash
            or receipt.forecast_hash != forecast.forecast_hash
            or receipt.design_hash != forecast.design_hash
            or receipt.descriptor_axis_hash != descriptor_axis.descriptor_axis_hash
            or receipt.direction_ids != descriptor_axis.direction_ids
            or receipt.direction_descriptor_hashes != descriptor_axis.descriptor_hashes
            or receipt.route_axis_hash != source_receipt.route_axis_hash
            or receipt.route_count != source_receipt.route_count
            or forecast.route_count != source_receipt.route_count
            or receipt.device != forecast.device
            or receipt.device_identity_hash != forecast.device_identity_hash
            or receipt.post_stack_free_device_bytes
            > forecast.device_total_memory_bytes
            or receipt.post_stack_global_occupied_bytes
            != forecast.device_total_memory_bytes
            - receipt.post_stack_free_device_bytes
            or receipt.post_stack_allocated_device_bytes > forecast.hbm_ceiling_bytes
            or direction_handle.descriptor_axis_hash != descriptor_axis.descriptor_axis_hash
            or direction_handle.route_count != source_receipt.route_count
            or direction_handle.device_identity != forecast.device
        ):
            _refuse(
                V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.STALE_LINEAGE,
                "private-primal bridge execution components do not close",
            )
        self._source_receipt = source_receipt
        self._forecast = forecast
        self._receipt = receipt
        self._descriptor_axis = descriptor_axis
        self._direction_handle = direction_handle
        self._storage_guard = storage_guard
        self._lock = RLock()
        self._poisoned = False
        self._evicted = False
        self.assert_live_identity()

    @property
    def source_receipt(self) -> V1SignedWorkLiteralJPrivatePrimalBridgeSourceReceipt:
        return self._source_receipt

    @property
    def forecast(self) -> V1SignedWorkLiteralJPrivatePrimalBridgeCUDAHBMForecast:
        return self._forecast

    @property
    def receipt(self) -> V1SignedWorkLiteralJPrivatePrimalBridgeCUDAReceipt:
        return self._receipt

    @property
    def descriptor_axis(self) -> V1LiteralJPrivatePrimalDirectionDescriptorAxis:
        self.assert_live_identity()
        return self._descriptor_axis

    @property
    def direction_handle(self) -> V1LiteralJPrivatePrimalDirectionCUDAHandle:
        self.assert_live_identity()
        return self._direction_handle

    @property
    def is_evicted(self) -> bool:
        return bool(self._evicted)

    @property
    def is_poisoned(self) -> bool:
        return bool(self._poisoned)

    @property
    def schema(self) -> str:
        return SIGNED_WORK_LITERAL_J_PRIVATE_PRIMAL_BRIDGE_EXECUTION_SCHEMA

    @property
    def execution_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "source_hash": self._source_receipt.source_hash,
                # Occupancy-free forecast identity: deterministic across
                # replays of the same source on the same device class.
                "forecast_identity_hash": self._forecast.identity_hash,
                "receipt_identity_hash": self._receipt.identity_hash,
                "descriptor_axis_hash": self._descriptor_axis.descriptor_axis_hash,
                "nonserializable_device_custody": True,
            },
            prefix=_RECEIPT_PREFIX + "execution-",
        )

    def assert_live_identity(self) -> None:
        with self._lock:
            if self._evicted:
                _refuse(
                    V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.EVICTED,
                    "private-primal bridge execution was evicted",
                )
            if self._poisoned:
                _refuse(
                    V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.CUSTODY_MUTATED,
                    "private-primal bridge execution was poisoned by mutation",
                )
            try:
                with self._direction_handle.borrow_for_literal_j_projection() as lease:
                    observed = _tensor_guard(
                        lease.directions_for_literal_j_projection()
                    )
            except V1LiteralJFixedDirectionProjectionCUDARefusal as error:
                self._poisoned = True
                _refuse(
                    V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.CUSTODY_MUTATED,
                    f"literal-J direction handle lost custody: {error}",
                )
            if observed != self._storage_guard:
                self._poisoned = True
                _refuse(
                    V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.CUSTODY_MUTATED,
                    "literal-J private-primal storage changed after bridge issuance",
                )

    def borrow_for_literal_j_projection(
        self,
    ) -> V1LiteralJPrivatePrimalDirectionCUDALease:
        self.assert_live_identity()
        return self._direction_handle.borrow_for_literal_j_projection()

    def evict(self) -> None:
        with self._lock:
            if self._evicted:
                return
            try:
                active = self._direction_handle.active_lease_count
            except V1LiteralJFixedDirectionProjectionCUDARefusal as error:
                _refuse(
                    V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.EVICTED,
                    f"literal-J direction handle is unavailable: {error}",
                )
            if active:
                _refuse(
                    V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.ACTIVE_LEASE,
                    "cannot evict private-primal bridge while a projection lease is live",
                )
            self._direction_handle.evict()
            self._evicted = True

    def __reduce__(self) -> NoReturn:
        raise TypeError("private-primal bridge executions are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("private-primal bridge executions are nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("private-primal bridge executions cannot be copied")

    def __deepcopy__(self, memo: dict[int, Any]) -> NoReturn:
        del memo
        raise TypeError("private-primal bridge executions cannot be copied")


def issue_v1_signed_work_literal_j_private_primal_bridge_cuda(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    spectral_execution: V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle,
    diagonal_l1_execution: V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle,
    design: V1SignedWorkLiteralJPrivatePrimalBridgeCUDAExecutionDesign | None = None,
) -> V1SignedWorkLiteralJPrivatePrimalBridgeCUDAExecution:
    """Issue canonical two-direction literal-J custody from exact KKT states."""

    if design is None:
        design = V1SignedWorkLiteralJPrivatePrimalBridgeCUDAExecutionDesign()
    if type(design) is not V1SignedWorkLiteralJPrivatePrimalBridgeCUDAExecutionDesign:
        raise TypeError("private-primal bridge design must be exact")
    if design.hbm_ceiling_fraction > policy.hbm_ceiling_fraction:
        _refuse(
            V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.AUTHORITY_VIOLATION,
            "bridge HBM ceiling cannot exceed its source KKT policy ceiling",
        )
    source_receipt = _register_source_pair(
        problem=problem,
        policy=policy,
        spectral_execution=spectral_execution,
        diagonal_l1_execution=diagonal_l1_execution,
    )
    torch_module = _require_torch()
    if not torch_module.cuda.is_available():
        _refuse(
            V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.CUDA_REQUIRED,
            "CUDA runtime is unavailable and production has no CPU fallback",
        )
    warn_only = getattr(
        torch_module,
        "is_deterministic_algorithms_warn_only_enabled",
        lambda: False,
    )
    if design.require_strict_torch_determinism and (
        not bool(torch_module.are_deterministic_algorithms_enabled())
        or bool(warn_only())
    ):
        _refuse(
            V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.DETERMINISM_REQUIRED,
            "strict torch deterministic algorithms are required before bridge issuance",
        )
    spectral_state = spectral_execution.state_handle
    diagonal_state = diagonal_l1_execution.state_handle
    if spectral_state is None or diagonal_state is None:  # guarded above; keep type narrow.
        _refuse(
            V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.STALE_LINEAGE,
            "single-branch theorem-state custody is unavailable",
        )
    spectral_device = _normalized_cuda_device(torch_module, spectral_state.device)
    diagonal_device = _normalized_cuda_device(torch_module, diagonal_state.device)
    if spectral_device != diagonal_device:
        _refuse(
            V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.DEVICE_MISMATCH,
            "single-branch primals do not share one CUDA device",
        )
    device, device_identity_hash = _device_identity(torch_module, spectral_device)
    free_before, total_before = torch_module.cuda.mem_get_info(spectral_device)
    forecast = build_v1_signed_work_literal_j_private_primal_bridge_cuda_hbm_forecast(
        design=design,
        source_receipt=source_receipt,
        device=device,
        device_identity_hash=device_identity_hash,
        current_allocated_device_bytes=int(
            torch_module.cuda.memory_allocated(spectral_device)
        ),
        current_reserved_device_bytes=int(
            torch_module.cuda.memory_reserved(spectral_device)
        ),
        device_free_memory_bytes=int(free_before),
        device_total_memory_bytes=int(total_before),
        _admission_available_bytes=cuda_reusable_memory_bytes(
            spectral_device, framework=torch_module, free_bytes=int(free_before)
        ),
    )
    if not forecast.admitted:
        _refuse(
            V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.HBM_FORECAST_EXCEEDED,
            "two-direction stack exceeds the live 85-percent HBM forecast",
        )

    descriptor_axis = build_v1_signed_work_literal_j_private_primal_descriptor_axis(
        source_receipt=source_receipt,
        design=design,
    )
    expected_receipts = source_receipt.branch_receipt_hashes
    directions: Any | None = None
    handle: V1LiteralJPrivatePrimalDirectionCUDAHandle | None = None
    try:
        with spectral_state.borrow_for_signed_work_decision_witness(
            expected_problem_hash=source_receipt.problem_hash,
            expected_policy_hash=source_receipt.policy_hash,
            expected_branch_receipt_hashes=(expected_receipts[0],),
        ) as spectral_lease:
            with diagonal_state.borrow_for_signed_work_decision_witness(
                expected_problem_hash=source_receipt.problem_hash,
                expected_policy_hash=source_receipt.policy_hash,
                expected_branch_receipt_hashes=(expected_receipts[1],),
            ) as diagonal_lease:
                if (
                    spectral_lease.operation
                    is not V1SignedWorkLeastWorkCUDAStateBorrowOperation.SIGNED_WORK_DECISION_WITNESS
                    or diagonal_lease.operation
                    is not V1SignedWorkLeastWorkCUDAStateBorrowOperation.SIGNED_WORK_DECISION_WITNESS
                ):
                    _refuse(
                        V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.AUTHORITY_VIOLATION,
                        "private primals were not borrowed through decision-witness custody",
                    )
                spectral = spectral_lease.primal(V1RelaxedKKTBranch.SPECTRAL)
                diagonal = diagonal_lease.primal(V1RelaxedKKTBranch.DIAGONAL_L1)
                expected_shape = (source_receipt.route_count,)
                if (
                    tuple(int(item) for item in spectral.shape) != expected_shape
                    or tuple(int(item) for item in diagonal.shape) != expected_shape
                    or spectral.dtype != torch_module.float64
                    or diagonal.dtype != torch_module.float64
                    or spectral.device != spectral_device
                    or diagonal.device != spectral_device
                    or spectral.requires_grad
                    or diagonal.requires_grad
                ):
                    _refuse(
                        V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.STALE_LINEAGE,
                        "borrowed private primals left their FP64 route/device axis",
                    )
                # ``stack`` is the sole route-scale allocation and performs a
                # device-to-device copy.  It neither hashes nor materializes a
                # route tensor on the host.
                directions = torch_module.stack((spectral, diagonal), dim=0).detach()
                if (
                    tuple(int(item) for item in directions.shape)
                    != (2, source_receipt.route_count)
                    or directions.dtype != torch_module.float64
                    or directions.device != spectral_device
                    or not directions.is_contiguous()
                    or directions.requires_grad
                ):
                    _refuse(
                        V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.CUSTODY_MUTATED,
                        "device stack did not produce canonical private-primal custody",
                    )
        storage_guard = _tensor_guard(directions)
        handle = V1LiteralJPrivatePrimalDirectionCUDAHandle(
            descriptor_axis=descriptor_axis,
            directions=directions,
        )
        del directions
        free_after, total_after = torch_module.cuda.mem_get_info(spectral_device)
        allocated_after = int(torch_module.cuda.memory_allocated(spectral_device))
        reserved_after = int(torch_module.cuda.memory_reserved(spectral_device))
        occupied_after = int(total_after) - int(free_after)
        if (
            int(total_after) != forecast.device_total_memory_bytes
            or allocated_after > forecast.hbm_ceiling_bytes
            or allocated_after > reserved_after
        ):
            handle.evict()
            handle = None
            _refuse(
                V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode.MEASURED_HBM_CEILING_EXCEEDED,
                "measured post-stack HBM crossed its registered device ceiling",
            )
        receipt = V1SignedWorkLiteralJPrivatePrimalBridgeCUDAReceipt(
            source_hash=source_receipt.source_hash,
            joint_result_hash=source_receipt.joint_result_hash,
            joint_execution_hash=source_receipt.joint_execution_hash,
            design_hash=design.design_hash,
            forecast_hash=forecast.forecast_hash,
            descriptor_axis_hash=descriptor_axis.descriptor_axis_hash,
            direction_ids=descriptor_axis.direction_ids,
            direction_descriptor_hashes=descriptor_axis.descriptor_hashes,
            route_axis_hash=source_receipt.route_axis_hash,
            route_count=source_receipt.route_count,
            device=device,
            device_identity_hash=device_identity_hash,
            source_borrow_operations=(
                "signed_work_decision_witness",
                "signed_work_decision_witness",
            ),
            device_to_device_copied_bytes=forecast.stacked_direction_bytes,
            route_scale_d2h_array_bytes=0,
            compact_scalar_d2h_bytes=0,
            post_stack_allocated_device_bytes=allocated_after,
            post_stack_reserved_device_bytes=reserved_after,
            post_stack_free_device_bytes=int(free_after),
            post_stack_global_occupied_bytes=occupied_after,
            source_leases_released=True,
            contiguous_fp64_canonical_branch_stack=True,
            hbm_forecast_admitted=True,
            measured_hbm_ceiling_respected=True,
        )
        return V1SignedWorkLiteralJPrivatePrimalBridgeCUDAExecution(
            source_receipt=source_receipt,
            forecast=forecast,
            receipt=receipt,
            descriptor_axis=descriptor_axis,
            direction_handle=handle,
            storage_guard=storage_guard,
        )
    except BaseException:
        if handle is not None and not handle.is_evicted:
            try:
                handle.evict()
            except BaseException:
                pass
        raise


__all__ = [
    "SIGNED_WORK_LITERAL_J_PRIVATE_PRIMAL_BRIDGE_BRANCH_ORDER",
    "SIGNED_WORK_LITERAL_J_PRIVATE_PRIMAL_BRIDGE_CUDA_ALGORITHM",
    "SIGNED_WORK_LITERAL_J_PRIVATE_PRIMAL_BRIDGE_CUDA_DESIGN_SCHEMA",
    "SIGNED_WORK_LITERAL_J_PRIVATE_PRIMAL_BRIDGE_EXECUTION_SCHEMA",
    "SIGNED_WORK_LITERAL_J_PRIVATE_PRIMAL_BRIDGE_HBM_SCHEMA",
    "SIGNED_WORK_LITERAL_J_PRIVATE_PRIMAL_BRIDGE_RECEIPT_SCHEMA",
    "SIGNED_WORK_LITERAL_J_PRIVATE_PRIMAL_BRIDGE_SOURCE_SCHEMA",
    "V1SignedWorkLiteralJPrivatePrimalBridgeCUDAExecution",
    "V1SignedWorkLiteralJPrivatePrimalBridgeCUDAExecutionDesign",
    "V1SignedWorkLiteralJPrivatePrimalBridgeCUDAHBMForecast",
    "V1SignedWorkLiteralJPrivatePrimalBridgeCUDAReceipt",
    "V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusal",
    "V1SignedWorkLiteralJPrivatePrimalBridgeCUDARefusalCode",
    "V1SignedWorkLiteralJPrivatePrimalBridgeSourceReceipt",
    "build_v1_signed_work_literal_j_private_primal_bridge_cuda_hbm_forecast",
    "build_v1_signed_work_literal_j_private_primal_descriptor_axis",
    "issue_v1_signed_work_literal_j_private_primal_bridge_cuda",
]
