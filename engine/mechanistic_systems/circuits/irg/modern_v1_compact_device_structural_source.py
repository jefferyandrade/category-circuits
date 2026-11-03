"""Compact live structural coordinates for the modern-v1 device proposal.

This module owns the non-statistical route coordinates shared by Phase A and
Phase B of the compact post-D proposal join.  Its canonical constructor moves
only the bit-packed current :class:`HardMask` to the registered device and
unpacks it there into the final eligibility tensor.  It never expands the
mask on the host and never transfers a route-scale tensor back from the
device.

The source is deliberately process-local and nonserializable.  A durable
receipt binds semantic identities and compact block capacities, while a live
storage/version invariant protects every tensor alias.  Neither the receipt
nor the live source carries C, calibration, oracle, commit, terminal, or
Pareto authority.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
import math
from numbers import Integral, Real
from typing import Any, Final, NoReturn

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask
from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

from .modern_v1_neutral_runtime import DeclaredRouteAxis
from .modern_v1_relaxed_kkt import V1RelaxedKKTBlockCapacity

try:  # pragma: no cover - availability is environment dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


COMPACT_DEVICE_STRUCTURAL_SOURCE_HBM_FORECAST_SCHEMA: Final = (
    "partition_free_v1_compact_device_structural_source_hbm_forecast_v1"
)
COMPACT_DEVICE_BLOCK_CAPACITY_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_compact_device_block_capacity_receipt_v1"
)
COMPACT_DEVICE_STRUCTURAL_SOURCE_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_compact_device_structural_source_receipt_v1"
)
COMPACT_DEVICE_STRUCTURAL_SOURCE_AUTHORITY_SCOPE: Final = (
    "phase_A_phase_B_structural_proposal_input_only_no_C_K_O_calibration_"
    "oracle_commit_terminal_validation_or_pareto_authority_v1"
)
COMPACT_DEVICE_STRUCTURAL_SOURCE_HBM_SCOPE: Final = (
    "current_reserved_plus_incremental_packed_mask_device_upload_final_"
    "structural_tensor_storage_compact_block_coordinates_and_bounded_device_"
    "unpack_workspace_with_preexisting_exact_tensors_counted_once_v1"
)
COMPACT_DEVICE_STRUCTURAL_SOURCE_HBM_FRACTION: Final = 0.85
COMPACT_DEVICE_STRUCTURAL_SOURCE_KKT_RECEIPT_SEMANTICS_V2: Final = (
    "pre_outcome_exact_same_device_resource_legality_receipt_certifies_finite_"
    "nonnegative_resource_positive_on_eligible_required_subset_block_legal_"
    "and_registered_legal_and_required_resource_scalars_v2"
)
COMPACT_DEVICE_STRUCTURAL_SOURCE_DEFAULT_MAXIMUM_UNPACK_WORKSPACE_BYTES: Final = (
    1 << 20
)

_FORECAST_PREFIX: Final = "partition-free-v1-compact-structural-forecast-"
_CAPACITY_PREFIX: Final = "partition-free-v1-compact-structural-capacity-"
_RECEIPT_PREFIX: Final = "partition-free-v1-compact-structural-receipt-"
_LIVE_PREFIX: Final = "partition-free-v1-compact-structural-live-"
_IDENTITY_PREFIX: Final = "partition-free-v1-compact-structural-coordinate-"
_CAPACITY_ISSUANCE_TOKEN: Final = object()
_RECEIPT_ISSUANCE_TOKEN: Final = object()
_SOURCE_ISSUANCE_TOKEN: Final = object()
_INT64_BYTES: Final = 8
_FLOAT64_BYTES: Final = 8
_BOOL_BYTES: Final = 1
_BLOCK_COORDINATE_WIDTH: Final = 3


class V1CompactDeviceStructuralSourceExecutionMode(str, Enum):
    """Whether the live tensors can serve the production CUDA proposal."""

    CUDA_PRODUCTION = "cuda_production"
    CPU_REFERENCE_TEST_ONLY = "cpu_reference_test_only"


class V1CompactDeviceStructuralSourcePath(str, Enum):
    """Versioned construction path for route-scale structural tensors."""

    PACKED_CURRENT_MASK_INTRINSIC = "packed_current_mask_intrinsic_v1"
    EXACT_SAME_DEVICE_TENSORS = "exact_same_device_tensors_v1"


class V1CompactDeviceStructuralSourceRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    INVALID_DESIGN = "invalid_design"
    AXIS_MISMATCH = "axis_mismatch"
    MASK_MISMATCH = "mask_mismatch"
    DEVICE_MISMATCH = "device_mismatch"
    MEMORY_INADMISSIBLE = "memory_inadmissible"
    STALE_IDENTITY = "stale_identity"
    INVALID_STRUCTURAL_TENSOR = "invalid_structural_tensor"
    INFEASIBLE_REQUIRED_ACTION = "infeasible_required_action"
    MUTATED_CUSTODY = "mutated_custody"
    ACTIVE_LEASE = "active_lease"
    EVICTED = "evicted"
    AUTHORITY_VIOLATION = "authority_violation"


class V1CompactDeviceStructuralSourceRefusal(RuntimeError):
    """Typed fail-closed boundary for live structural custody."""

    def __init__(
        self,
        code: V1CompactDeviceStructuralSourceRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1CompactDeviceStructuralSourceRefusalCode:
            raise TypeError("structural-source refusal code must have exact enum type")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1CompactDeviceStructuralSourceRefusalCode,
    message: str,
) -> NoReturn:
    raise V1CompactDeviceStructuralSourceRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1CompactDeviceStructuralSourceRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1CompactDeviceStructuralSourceRefusalCode.INVALID_TYPE,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1CompactDeviceStructuralSourceRefusalCode.INVALID_DESIGN,
            f"{name} must be nonnegative",
        )
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result <= 0:
        _refuse(
            V1CompactDeviceStructuralSourceRefusalCode.INVALID_DESIGN,
            f"{name} must be positive",
        )
    return result


def _nonnegative_finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1CompactDeviceStructuralSourceRefusalCode.INVALID_TYPE,
            f"{name} must be one finite nonnegative real",
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        _refuse(
            V1CompactDeviceStructuralSourceRefusalCode.INVALID_DESIGN,
            f"{name} must be finite and nonnegative",
        )
    return 0.0 if result == 0.0 else result


def _device(value: object) -> Any:
    if torch is None:
        _refuse(
            V1CompactDeviceStructuralSourceRefusalCode.DEVICE_MISMATCH,
            "compact structural custody requires PyTorch",
        )
    try:
        result = torch.device(_identity(value, name="declared device"))
    except (RuntimeError, ValueError) as error:
        raise V1CompactDeviceStructuralSourceRefusal(
            V1CompactDeviceStructuralSourceRefusalCode.DEVICE_MISMATCH,
            "declared device is invalid",
        ) from error
    if result.type == "cuda" and result.index is None:
        _refuse(
            V1CompactDeviceStructuralSourceRefusalCode.DEVICE_MISMATCH,
            "production CUDA custody must name an explicit device index",
        )
    return result


def _validate_execution_device(
    *,
    device: Any,
    execution_mode: V1CompactDeviceStructuralSourceExecutionMode,
) -> None:
    if type(execution_mode) is not V1CompactDeviceStructuralSourceExecutionMode:
        _refuse(
            V1CompactDeviceStructuralSourceRefusalCode.INVALID_TYPE,
            "execution mode must have exact enum type",
        )
    if execution_mode is V1CompactDeviceStructuralSourceExecutionMode.CUDA_PRODUCTION:
        if device.type != "cuda" or torch is None or not torch.cuda.is_available():
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.DEVICE_MISMATCH,
                "production structural custody requires an available CUDA device",
            )
        try:
            torch.cuda.get_device_properties(device)
        except (AssertionError, RuntimeError) as error:  # pragma: no cover - driver.
            raise V1CompactDeviceStructuralSourceRefusal(
                V1CompactDeviceStructuralSourceRefusalCode.DEVICE_MISMATCH,
                "registered CUDA device is unavailable",
            ) from error
    elif device.type != "cpu" or str(device) != "cpu":
        _refuse(
            V1CompactDeviceStructuralSourceRefusalCode.DEVICE_MISMATCH,
            "CPU reference custody requires the exact cpu device",
        )


def _validate_axis_and_mask(
    *,
    current_mask: HardMask,
    route_axis: DeclaredRouteAxis,
) -> tuple[int, tuple[Any, ...]]:
    if type(current_mask) is not HardMask:
        _refuse(
            V1CompactDeviceStructuralSourceRefusalCode.INVALID_TYPE,
            "current mask must be the exact HardMask type",
        )
    if type(route_axis) is not DeclaredRouteAxis:
        _refuse(
            V1CompactDeviceStructuralSourceRefusalCode.INVALID_TYPE,
            "route axis must be the exact DeclaredRouteAxis type",
        )
    routes = route_axis.route_count
    if (
        current_mask.route_count != routes
        or current_mask.route_axis_hash != route_axis.route_axis_hash
    ):
        _refuse(
            V1CompactDeviceStructuralSourceRefusalCode.MASK_MISMATCH,
            "packed current mask differs from the declared physical route axis",
        )
    blocks = tuple(route_axis.route_blocks)
    cursor = 0
    for block in blocks:
        if int(block.offset) != cursor or block.stop > routes:
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.AXIS_MISMATCH,
                "declared physical blocks do not form one contiguous cover",
            )
        cursor = block.stop
    if not blocks or cursor != routes:
        _refuse(
            V1CompactDeviceStructuralSourceRefusalCode.AXIS_MISMATCH,
            "declared physical blocks do not cover the route axis",
        )
    return routes, blocks


def _tensor_storage_bytes(value: Any) -> int:
    return int(value.untyped_storage().nbytes())


def _tensor_storage_invariant(value: Any) -> tuple[object, ...]:
    if torch is None or type(value) is not torch.Tensor:
        _refuse(
            V1CompactDeviceStructuralSourceRefusalCode.MUTATED_CUSTODY,
            "structural coordinate is no longer one exact torch tensor",
        )
    storage = value.untyped_storage()
    return (
        str(value.device),
        str(value.dtype),
        tuple(int(item) for item in value.shape),
        tuple(int(item) for item in value.stride()),
        int(value.storage_offset()),
        int(value.data_ptr()),
        int(storage.data_ptr()),
        int(storage.nbytes()),
        int(value._version),
        bool(value.requires_grad),
        bool(value.is_contiguous()),
    )


def _exact_same_device_tensor(
    value: object,
    *,
    route_count: int,
    dtype: Any,
    device: Any,
    name: str,
) -> Any:
    if torch is None or type(value) is not torch.Tensor:
        _refuse(
            V1CompactDeviceStructuralSourceRefusalCode.INVALID_STRUCTURAL_TENSOR,
            f"{name} must be one exact torch.Tensor",
        )
    expected_bytes = route_count * int(torch.empty((), dtype=dtype).element_size())
    if (
        tuple(int(item) for item in value.shape) != (route_count,)
        or value.dtype != dtype
        or value.device != device
        or not value.is_contiguous()
        or value.requires_grad
        or int(value.storage_offset()) != 0
        or _tensor_storage_bytes(value) != expected_bytes
    ):
        _refuse(
            V1CompactDeviceStructuralSourceRefusalCode.INVALID_STRUCTURAL_TENSOR,
            f"{name} differs from its exact route/dtype/device/storage contract",
        )
    return value


def _unique_storage_bytes(values: tuple[Any, ...]) -> int:
    seen: set[tuple[str, int]] = set()
    total = 0
    for value in values:
        key = (str(value.device), int(value.untyped_storage().data_ptr()))
        if key in seen:
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.INVALID_STRUCTURAL_TENSOR,
                "explicit structural tensors may not alias one storage",
            )
        seen.add(key)
        total += _tensor_storage_bytes(value)
    return total


def _normalize_unpack_workspace(
    *, route_count: int, maximum_unpack_workspace_bytes: int
) -> tuple[int, int]:
    maximum = _positive_int(
        maximum_unpack_workspace_bytes,
        name="maximum unpack workspace bytes",
    )
    if maximum < 16:
        _refuse(
            V1CompactDeviceStructuralSourceRefusalCode.INVALID_DESIGN,
            "unpack workspace must cover one byte and eight shift coordinates",
        )
    # One uint8 entry is produced for each padded route bit.  A chunk must
    # cover at least one full byte so its semantics cannot depend on packet
    # boundaries.
    padded_routes = ((route_count + 7) // 8) * 8
    chunk_routes = min(
        padded_routes,
        max(8, ((maximum - 8) // 8) * 8),
    )
    # The eight shift coordinates are retained while one expanded bit tile is
    # live.  This is a conservative, allocator-independent payload forecast.
    return chunk_routes, chunk_routes + 8


@dataclass(frozen=True, slots=True)
class V1CompactDeviceStructuralSourceHBMForecast:
    """Pre-allocation admission forecast for one structural source."""

    route_count: int
    block_count: int
    construction_path: V1CompactDeviceStructuralSourcePath
    current_reserved_device_bytes: int
    total_device_memory_bytes: int
    memory_budget_bytes: int
    packed_mask_upload_device_bytes: int
    retained_structural_tensor_storage_bytes: int
    newly_allocated_structural_tensor_storage_bytes: int
    borrowed_structural_tensor_storage_bytes: int
    compact_block_coordinate_device_bytes: int
    temporary_current_mask_tensor_device_bytes: int
    bounded_unpack_workspace_device_bytes: int
    incremental_peak_device_bytes: int
    joint_peak_device_bytes: int
    admitted: bool
    hbm_admission_fraction: float = COMPACT_DEVICE_STRUCTURAL_SOURCE_HBM_FRACTION
    route_scale_host_expansion_bytes: int = 0
    route_scale_device_to_host_bytes: int = 0
    retained_tensor_storage_counted_once: bool = True
    memory_admission_only: bool = True
    memory_accounting_scope: str = COMPACT_DEVICE_STRUCTURAL_SOURCE_HBM_SCOPE
    schema: str = COMPACT_DEVICE_STRUCTURAL_SOURCE_HBM_FORECAST_SCHEMA
    _sealed_hash: str = field(init=False, default="", repr=False)
    _admission_memory: InitVar[tuple[int, int] | None] = None

    def __post_init__(self, _admission_memory: tuple[int, int] | None) -> None:
        routes = _positive_int(self.route_count, name="forecast route count")
        blocks = _positive_int(self.block_count, name="forecast block count")
        if type(self.construction_path) is not V1CompactDeviceStructuralSourcePath:
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.INVALID_TYPE,
                "forecast construction path must have exact enum type",
            )
        current = _nonnegative_int(
            self.current_reserved_device_bytes,
            name="current reserved device bytes",
        )
        total = _positive_int(
            self.total_device_memory_bytes,
            name="total device memory bytes",
        )
        expected_budget = (total * 85) // 100
        if _admission_memory is None:
            admitted = self.joint_peak_device_bytes <= expected_budget
        else:
            available, allocated = _admission_memory
            admitted = (
                allocated + self.incremental_peak_device_bytes <= expected_budget
                and self.incremental_peak_device_bytes <= available
            )
        if self.memory_budget_bytes != expected_budget:
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.INVALID_DESIGN,
                "forecast memory budget does not replay the 85% admission ceiling",
            )
        for item in fields(self):
            if item.name.endswith("_bytes") and item.name not in {
                "total_device_memory_bytes",
                "memory_budget_bytes",
            }:
                _nonnegative_int(getattr(self, item.name), name=item.name)
        if (
            self.packed_mask_upload_device_bytes != (routes + 7) // 8
            or self.compact_block_coordinate_device_bytes
            != blocks * _BLOCK_COORDINATE_WIDTH * _INT64_BYTES
            or self.incremental_peak_device_bytes
            != (
                self.packed_mask_upload_device_bytes
                + self.newly_allocated_structural_tensor_storage_bytes
                + self.compact_block_coordinate_device_bytes
                + self.temporary_current_mask_tensor_device_bytes
                + self.bounded_unpack_workspace_device_bytes
            )
            or self.joint_peak_device_bytes
            != current + self.incremental_peak_device_bytes
            or self.admitted is not admitted
            or self.hbm_admission_fraction
            != COMPACT_DEVICE_STRUCTURAL_SOURCE_HBM_FRACTION
            or self.route_scale_host_expansion_bytes != 0
            or self.route_scale_device_to_host_bytes != 0
            or self.retained_tensor_storage_counted_once is not True
            or self.memory_admission_only is not True
            or self.memory_accounting_scope != COMPACT_DEVICE_STRUCTURAL_SOURCE_HBM_SCOPE
            or self.schema != COMPACT_DEVICE_STRUCTURAL_SOURCE_HBM_FORECAST_SCHEMA
        ):
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.INVALID_DESIGN,
                "structural-source HBM accounting does not replay",
            )
        if self.construction_path is (
            V1CompactDeviceStructuralSourcePath.PACKED_CURRENT_MASK_INTRINSIC
        ):
            expected_retained = routes * _BOOL_BYTES + _FLOAT64_BYTES + _BOOL_BYTES
            if (
                self.retained_structural_tensor_storage_bytes != expected_retained
                or self.newly_allocated_structural_tensor_storage_bytes
                != expected_retained
                or self.borrowed_structural_tensor_storage_bytes != 0
                or self.temporary_current_mask_tensor_device_bytes != 0
            ):
                _refuse(
                    V1CompactDeviceStructuralSourceRefusalCode.INVALID_DESIGN,
                    "intrinsic structural storage was double-counted or omitted",
                )
        elif (
            self.newly_allocated_structural_tensor_storage_bytes != 0
            or self.borrowed_structural_tensor_storage_bytes
            != self.retained_structural_tensor_storage_bytes
            or self.temporary_current_mask_tensor_device_bytes != routes
        ):
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.INVALID_DESIGN,
                "exact-device structural storage was double-counted or omitted",
            )
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "block_count", blocks)
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        return content_hash(
            {
                item.name: (
                    getattr(self, item.name).value
                    if isinstance(getattr(self, item.name), Enum)
                    else getattr(self, item.name)
                )
                for item in fields(self)
                if not item.name.startswith("_")
            },
            prefix=_FORECAST_PREFIX,
        )

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.STALE_IDENTITY,
                "structural-source HBM forecast changed after construction",
            )
        return observed

    @property
    def forecast_hash(self) -> str:
        return self.assert_integrity()

    def assert_admissible(self) -> None:
        self.assert_integrity()
        if not self.admitted:
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.MEMORY_INADMISSIBLE,
                "structural-source peak exceeds the 85% device-memory ceiling",
            )


def forecast_v1_compact_device_structural_source_hbm(
    *,
    route_count: int,
    block_count: int,
    construction_path: V1CompactDeviceStructuralSourcePath,
    current_reserved_device_bytes: int,
    total_device_memory_bytes: int,
    maximum_unpack_workspace_bytes: int = (
        COMPACT_DEVICE_STRUCTURAL_SOURCE_DEFAULT_MAXIMUM_UNPACK_WORKSPACE_BYTES
    ),
    borrowed_structural_tensor_storage_bytes: int = 0,
    _admission_memory: tuple[int, int] | None = None,
) -> V1CompactDeviceStructuralSourceHBMForecast:
    """Forecast exact payload storage without allocating a tensor."""

    routes = _positive_int(route_count, name="route count")
    blocks = _positive_int(block_count, name="block count")
    if type(construction_path) is not V1CompactDeviceStructuralSourcePath:
        _refuse(
            V1CompactDeviceStructuralSourceRefusalCode.INVALID_TYPE,
            "construction path must have exact enum type",
        )
    _chunk_routes, workspace_bytes = _normalize_unpack_workspace(
        route_count=routes,
        maximum_unpack_workspace_bytes=maximum_unpack_workspace_bytes,
    )
    borrowed = _nonnegative_int(
        borrowed_structural_tensor_storage_bytes,
        name="borrowed structural tensor storage bytes",
    )
    if construction_path is (
        V1CompactDeviceStructuralSourcePath.PACKED_CURRENT_MASK_INTRINSIC
    ):
        if borrowed != 0:
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.INVALID_DESIGN,
                "intrinsic construction cannot pre-borrow structural tensors",
            )
        retained = routes * _BOOL_BYTES + _FLOAT64_BYTES + _BOOL_BYTES
        newly_allocated = retained
        temporary_mask = 0
    else:
        if borrowed <= 0:
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.INVALID_DESIGN,
                "exact-device construction requires its borrowed storage count",
            )
        retained = borrowed
        newly_allocated = 0
        temporary_mask = routes * _BOOL_BYTES
    packed = (routes + 7) // 8
    coordinates = blocks * _BLOCK_COORDINATE_WIDTH * _INT64_BYTES
    incremental = packed + newly_allocated + coordinates + temporary_mask + workspace_bytes
    current = _nonnegative_int(
        current_reserved_device_bytes,
        name="current reserved device bytes",
    )
    total = _positive_int(total_device_memory_bytes, name="total device memory bytes")
    budget = (total * 85) // 100
    joint = current + incremental
    if _admission_memory is None:
        admitted = joint <= budget
    else:
        available, allocated = _admission_memory
        admitted = allocated + incremental <= budget and incremental <= available
    return V1CompactDeviceStructuralSourceHBMForecast(
        route_count=routes,
        block_count=blocks,
        construction_path=construction_path,
        current_reserved_device_bytes=current,
        total_device_memory_bytes=total,
        memory_budget_bytes=budget,
        packed_mask_upload_device_bytes=packed,
        retained_structural_tensor_storage_bytes=retained,
        newly_allocated_structural_tensor_storage_bytes=newly_allocated,
        borrowed_structural_tensor_storage_bytes=borrowed,
        compact_block_coordinate_device_bytes=coordinates,
        temporary_current_mask_tensor_device_bytes=temporary_mask,
        bounded_unpack_workspace_device_bytes=workspace_bytes,
        incremental_peak_device_bytes=incremental,
        joint_peak_device_bytes=joint,
        admitted=admitted,
        _admission_memory=_admission_memory,
    )


@dataclass(frozen=True, slots=True)
class V1CompactDeviceBlockCapacityReceipt:
    """Issuer proof of one-live-survivor capacities from the packed mask."""

    route_axis_hash: str
    route_axis_content_hash: str
    current_mask_hash: str
    block_axis_hash: str
    route_count: int
    block_live_counts: tuple[int, ...]
    block_capacities: tuple[V1RelaxedKKTBlockCapacity, ...]
    block_live_count_scalar_device_to_host_count: int
    capacity_rule: str = "max_current_live_count_minus_one_zero_floor_v1"
    one_live_survivor_per_nonempty_physical_block: bool = True
    route_scale_device_to_host_bytes: int = 0
    schema: str = COMPACT_DEVICE_BLOCK_CAPACITY_RECEIPT_SCHEMA
    _issuance_token: object = field(default=None, repr=False, compare=False)
    _sealed_hash: str = field(init=False, default="", repr=False)

    def __post_init__(self) -> None:
        if self._issuance_token is not _CAPACITY_ISSUANCE_TOKEN:
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.AUTHORITY_VIOLATION,
                "block-capacity receipts are issued only from live-mask reductions",
            )
        for name in (
            "route_axis_hash",
            "route_axis_content_hash",
            "current_mask_hash",
            "block_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        routes = _positive_int(self.route_count, name="capacity route count")
        counts = tuple(
            _nonnegative_int(value, name="block live count")
            for value in self.block_live_counts
        )
        capacities = tuple(self.block_capacities)
        if (
            not counts
            or len(counts) != len(capacities)
            or any(type(value) is not V1RelaxedKKTBlockCapacity for value in capacities)
            or self.block_live_count_scalar_device_to_host_count != len(counts)
        ):
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.AXIS_MISMATCH,
                "block live counts and compact capacities do not align",
            )
        cursor = 0
        for count, block in zip(counts, capacities, strict=True):
            if (
                block.start != cursor
                or block.stop > routes
                or count > block.stop - block.start
                or block.removal_capacity != max(0, count - 1)
            ):
                _refuse(
                    V1CompactDeviceStructuralSourceRefusalCode.AXIS_MISMATCH,
                    "compact capacity does not replay its live physical block",
                )
            cursor = block.stop
        if (
            cursor != routes
            or self.capacity_rule
            != "max_current_live_count_minus_one_zero_floor_v1"
            or self.one_live_survivor_per_nonempty_physical_block is not True
            or self.route_scale_device_to_host_bytes != 0
            or self.schema != COMPACT_DEVICE_BLOCK_CAPACITY_RECEIPT_SCHEMA
        ):
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.AUTHORITY_VIOLATION,
                "block-capacity receipt changed its derivation or authority",
            )
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "block_live_counts", counts)
        object.__setattr__(self, "block_capacities", capacities)
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        return content_hash(
            {
                item.name: getattr(self, item.name)
                for item in fields(self)
                if not item.name.startswith("_")
            },
            prefix=_CAPACITY_PREFIX,
        )

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.STALE_IDENTITY,
                "block-capacity receipt changed after issuance",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1CompactDeviceStructuralSourceReceipt:
    """Durable semantic receipt for one nonserializable structural source."""

    capacity_receipt_hash: str
    route_axis_hash: str
    route_axis_content_hash: str
    route_id_axis_hash: str
    current_mask_hash: str
    native_fiber_axis_receipt_hash: str
    structural_registration_hash: str
    resource_tensor_identity_hash: str
    eligible_mask_identity_hash: str
    required_hard_action_identity_hash: str
    route_count: int
    block_count: int
    quadratic_budget: float
    registered_legal_resource_maximum: float
    registered_required_resource_value: float
    declared_device: str
    execution_mode: V1CompactDeviceStructuralSourceExecutionMode
    construction_path: V1CompactDeviceStructuralSourcePath
    resource_storage_bytes: int
    eligible_mask_storage_bytes: int
    required_hard_action_storage_bytes: int
    compact_block_coordinate_storage_bytes: int
    resource_is_intrinsic_unit: bool
    eligible_mask_is_exact_current_mask: bool
    required_hard_action_is_intrinsic_empty: bool
    packed_mask_uploaded_before_device_unpack: bool = True
    host_route_mask_expanded: bool = False
    route_scale_device_to_host_bytes: int = 0
    process_local_storage_version_custody: bool = True
    phase_a_structural_input_authority: bool = True
    phase_b_structural_input_authority: bool = True
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    validation_authority: bool = False
    pareto_authority: bool = False
    structural_source_receipt_semantics: str = (
        COMPACT_DEVICE_STRUCTURAL_SOURCE_KKT_RECEIPT_SEMANTICS_V2
    )
    authority_scope: str = COMPACT_DEVICE_STRUCTURAL_SOURCE_AUTHORITY_SCOPE
    schema: str = COMPACT_DEVICE_STRUCTURAL_SOURCE_RECEIPT_SCHEMA
    _issuance_token: object = field(default=None, repr=False, compare=False)
    _sealed_hash: str = field(init=False, default="", repr=False)

    def __post_init__(self) -> None:
        if self._issuance_token is not _RECEIPT_ISSUANCE_TOKEN:
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.AUTHORITY_VIOLATION,
                "structural-source receipts are issuer sealed",
            )
        for name in (
            "capacity_receipt_hash",
            "route_axis_hash",
            "route_axis_content_hash",
            "route_id_axis_hash",
            "current_mask_hash",
            "native_fiber_axis_receipt_hash",
            "structural_registration_hash",
            "resource_tensor_identity_hash",
            "eligible_mask_identity_hash",
            "required_hard_action_identity_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        routes = _positive_int(self.route_count, name="receipt route count")
        blocks = _positive_int(self.block_count, name="receipt block count")
        q_budget = _nonnegative_finite(self.quadratic_budget, name="quadratic budget")
        legal_maximum = _nonnegative_finite(
            self.registered_legal_resource_maximum,
            name="registered legal resource maximum",
        )
        required_value = _nonnegative_finite(
            self.registered_required_resource_value,
            name="registered required resource value",
        )
        for name in (
            "resource_storage_bytes",
            "eligible_mask_storage_bytes",
            "required_hard_action_storage_bytes",
            "compact_block_coordinate_storage_bytes",
        ):
            _positive_int(getattr(self, name), name=name)
        if type(self.execution_mode) is not V1CompactDeviceStructuralSourceExecutionMode:
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.INVALID_TYPE,
                "receipt execution mode must have exact enum type",
            )
        if type(self.construction_path) is not V1CompactDeviceStructuralSourcePath:
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.INVALID_TYPE,
                "receipt construction path must have exact enum type",
            )
        production = self.execution_mode is (
            V1CompactDeviceStructuralSourceExecutionMode.CUDA_PRODUCTION
        )
        true_flags = (
            self.packed_mask_uploaded_before_device_unpack,
            self.process_local_storage_version_custody,
        )
        false_flags = (
            self.host_route_mask_expanded,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.validation_authority,
            self.pareto_authority,
        )
        if (
            any(type(value) is not bool for value in (
                self.resource_is_intrinsic_unit,
                self.eligible_mask_is_exact_current_mask,
                self.required_hard_action_is_intrinsic_empty,
                self.phase_a_structural_input_authority,
                self.phase_b_structural_input_authority,
                *true_flags,
                *false_flags,
            ))
            or any(value is not True for value in true_flags)
            or any(value is not False for value in false_flags)
            or self.phase_a_structural_input_authority is not production
            or self.phase_b_structural_input_authority is not production
            or (production and not self.declared_device.startswith("cuda:"))
            or (not production and self.declared_device != "cpu")
            or self.route_scale_device_to_host_bytes != 0
            or required_value > legal_maximum
            or self.compact_block_coordinate_storage_bytes
            != blocks * _BLOCK_COORDINATE_WIDTH * _INT64_BYTES
            or self.authority_scope != COMPACT_DEVICE_STRUCTURAL_SOURCE_AUTHORITY_SCOPE
            or self.structural_source_receipt_semantics
            != COMPACT_DEVICE_STRUCTURAL_SOURCE_KKT_RECEIPT_SEMANTICS_V2
            or self.schema != COMPACT_DEVICE_STRUCTURAL_SOURCE_RECEIPT_SCHEMA
        ):
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.AUTHORITY_VIOLATION,
                "structural-source receipt crossed its execution or authority boundary",
            )
        intrinsic = self.construction_path is (
            V1CompactDeviceStructuralSourcePath.PACKED_CURRENT_MASK_INTRINSIC
        )
        if intrinsic != (
            self.resource_is_intrinsic_unit
            and self.eligible_mask_is_exact_current_mask
            and self.required_hard_action_is_intrinsic_empty
        ):
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.AUTHORITY_VIOLATION,
                "construction path does not match its structural tensor semantics",
            )
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "block_count", blocks)
        object.__setattr__(self, "quadratic_budget", q_budget)
        object.__setattr__(self, "registered_legal_resource_maximum", legal_maximum)
        object.__setattr__(self, "registered_required_resource_value", required_value)
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        return content_hash(
            {
                item.name: (
                    getattr(self, item.name).value
                    if isinstance(getattr(self, item.name), Enum)
                    else getattr(self, item.name)
                )
                for item in fields(self)
                if not item.name.startswith("_")
            },
            prefix=_RECEIPT_PREFIX,
        )

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.STALE_IDENTITY,
                "structural-source receipt changed after issuance",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


def _unpack_packed_big_endian_mask_on_device(
    *,
    current_mask: HardMask,
    device: Any,
    maximum_unpack_workspace_bytes: int,
) -> Any:
    """Upload compact bytes and unpack them in bounded device tiles."""

    routes = current_mask.route_count
    chunk_routes, _workspace_bytes = _normalize_unpack_workspace(
        route_count=routes,
        maximum_unpack_workspace_bytes=maximum_unpack_workspace_bytes,
    )
    # ``bytearray`` supplies writable packed-only host storage for frombuffer;
    # its size is ceil(R/8), never R.  The first route-scale object allocated
    # here is the final device bool tensor.
    packed_host = torch.frombuffer(bytearray(current_mask.packed_bits), dtype=torch.uint8)
    packed_device = packed_host.to(device=device, dtype=torch.uint8, copy=True)
    result = torch.empty((routes,), dtype=torch.bool, device=device)
    shifts = torch.tensor((7, 6, 5, 4, 3, 2, 1, 0), dtype=torch.uint8, device=device)
    byte_chunk = max(1, chunk_routes // 8)
    route_cursor = 0
    for byte_start in range(0, int(packed_device.numel()), byte_chunk):
        byte_stop = min(int(packed_device.numel()), byte_start + byte_chunk)
        expanded = torch.bitwise_right_shift(
            packed_device[byte_start:byte_stop].unsqueeze(1),
            shifts.unsqueeze(0),
        )
        expanded.bitwise_and_(1)
        available = min(routes - route_cursor, int(expanded.numel()))
        result[route_cursor : route_cursor + available].copy_(
            expanded.reshape(-1)[:available]
        )
        route_cursor += available
    if route_cursor != routes:
        _refuse(
            V1CompactDeviceStructuralSourceRefusalCode.MASK_MISMATCH,
            "device unpack did not cover the packed current-mask axis",
        )
    return result


def _capacity_axis_hash(route_axis: DeclaredRouteAxis) -> str:
    return content_hash(
        tuple(
            {
                "block_id": block.block_id,
                "layer_id": block.layer_id,
                "start": int(block.offset),
                "stop": int(block.stop),
                "route_kind": block.route_kind,
                "shape": tuple(int(value) for value in block.shape),
            }
            for block in route_axis.route_blocks
        ),
        prefix=_CAPACITY_PREFIX + "axis-",
    )


def _derive_block_capacities(
    *,
    current_live_mask: Any,
    current_mask: HardMask,
    route_axis: DeclaredRouteAxis,
) -> V1CompactDeviceBlockCapacityReceipt:
    live_counts: list[int] = []
    capacities: list[V1RelaxedKKTBlockCapacity] = []
    for block in route_axis.route_blocks:
        start = int(block.offset)
        stop = int(block.stop)
        live = int(torch.count_nonzero(current_live_mask[start:stop]).item())
        live_counts.append(live)
        capacities.append(
            V1RelaxedKKTBlockCapacity(
                block_id=block.block_id,
                start=start,
                stop=stop,
                removal_capacity=max(0, live - 1),
            )
        )
    return V1CompactDeviceBlockCapacityReceipt(
        route_axis_hash=route_axis.route_axis_hash,
        route_axis_content_hash=route_axis.content_hash,
        current_mask_hash=current_mask.mask_hash,
        block_axis_hash=_capacity_axis_hash(route_axis),
        route_count=route_axis.route_count,
        block_live_counts=tuple(live_counts),
        block_capacities=tuple(capacities),
        block_live_count_scalar_device_to_host_count=len(live_counts),
        _issuance_token=_CAPACITY_ISSUANCE_TOKEN,
    )


def _intrinsic_coordinate_hash(
    *, kind: str, route_axis: DeclaredRouteAxis, current_mask_hash: str
) -> str:
    return content_hash(
        {
            "kind": kind,
            "route_axis_hash": route_axis.route_axis_hash,
            "route_axis_content_hash": route_axis.content_hash,
            "current_mask_hash": current_mask_hash,
            "route_count": route_axis.route_count,
        },
        prefix=_IDENTITY_PREFIX,
    )


def _chunked_explicit_tensor_validation(
    *,
    resource: Any,
    eligible: Any,
    required: Any,
    current_live: Any,
    maximum_chunk_routes: int,
) -> bool:
    minimum = float(torch.amin(resource).item())
    maximum = float(torch.amax(resource).item())
    if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum <= 0.0:
        _refuse(
            V1CompactDeviceStructuralSourceRefusalCode.INVALID_STRUCTURAL_TENSOR,
            "explicit resource must be finite and positive on the full route axis",
        )
    equal_current = True
    routes = int(resource.numel())
    chunk = max(1, maximum_chunk_routes)
    for start in range(0, routes, chunk):
        stop = min(routes, start + chunk)
        live_chunk = current_live[start:stop]
        eligible_chunk = eligible[start:stop]
        required_chunk = required[start:stop]
        if bool(torch.any(eligible_chunk & ~live_chunk).item()):
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.INVALID_STRUCTURAL_TENSOR,
                "explicit eligibility includes a route absent from the current mask",
            )
        if bool(torch.any(required_chunk & ~eligible_chunk).item()):
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.INFEASIBLE_REQUIRED_ACTION,
                "explicit required action contains an ineligible route",
            )
        equal_current = equal_current and bool(torch.equal(eligible_chunk, live_chunk))
    return equal_current


def _registered_explicit_resource_scalars(
    *,
    resource: Any,
    eligible: Any,
    required: Any,
    capacity_receipt: V1CompactDeviceBlockCapacityReceipt,
) -> tuple[float, float]:
    """Derive exact legal/required values without a route gather or top-k.

    The separately identified structural path presently admits a globally
    uniform resource coordinate.  This is the compact field whose exact
    block-constrained maximum is shared by floor-free Device Phase A and
    DeviceProblemV2 using only compact device scalar reductions.  A genuinely
    nonuniform resource requires its own bounded Phase-A summary receipt and
    therefore refuses here.
    """

    global_minimum = float(torch.amin(resource).item())
    global_maximum = float(torch.amax(resource).item())
    if global_minimum != global_maximum:
        _refuse(
            V1CompactDeviceStructuralSourceRefusalCode.INVALID_STRUCTURAL_TENSOR,
            "explicit nonuniform resource lacks a bounded exact Phase-A summary receipt",
        )
    legal_terms: list[float] = []
    required_terms: list[float] = []
    for block in capacity_receipt.block_capacities:
        eligible_count = int(
            torch.count_nonzero(eligible[block.start : block.stop]).item()
        )
        required_count = int(
            torch.count_nonzero(required[block.start : block.stop]).item()
        )
        if required_count > block.removal_capacity:
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.INFEASIBLE_REQUIRED_ACTION,
                "required hard action exceeds a one-live-survivor block capacity",
            )
        legal_terms.append(
            global_maximum * float(min(block.removal_capacity, eligible_count))
        )
        required_terms.append(global_maximum * float(required_count))
    legal = math.fsum(legal_terms)
    required_value = math.fsum(required_terms)
    if not math.isfinite(legal) or not math.isfinite(required_value):
        _refuse(
            V1CompactDeviceStructuralSourceRefusalCode.INVALID_STRUCTURAL_TENSOR,
            "registered structural resource scalars are not representable",
        )
    return legal, required_value


def _runtime_forecast(
    *,
    route_count: int,
    block_count: int,
    construction_path: V1CompactDeviceStructuralSourcePath,
    device: Any,
    maximum_unpack_workspace_bytes: int,
    borrowed_structural_tensor_storage_bytes: int,
) -> V1CompactDeviceStructuralSourceHBMForecast:
    if device.type != "cuda" or torch is None:  # CPU tests must register explicitly.
        _refuse(
            V1CompactDeviceStructuralSourceRefusalCode.INVALID_DESIGN,
            "CPU reference construction requires an explicit deterministic forecast",
        )
    properties = torch.cuda.get_device_properties(device)
    return forecast_v1_compact_device_structural_source_hbm(
        route_count=route_count,
        block_count=block_count,
        construction_path=construction_path,
        current_reserved_device_bytes=int(torch.cuda.memory_reserved(device)),
        total_device_memory_bytes=int(properties.total_memory),
        maximum_unpack_workspace_bytes=maximum_unpack_workspace_bytes,
        borrowed_structural_tensor_storage_bytes=(
            borrowed_structural_tensor_storage_bytes
        ),
        _admission_memory=(
            cuda_reusable_memory_bytes(device, framework=torch),
            int(torch.cuda.memory_allocated(device)),
        ),
    )


class V1CompactDeviceStructuralSourceLease:
    """Narrow zero-copy borrow spanning Phase-A/Phase-B tensor use."""

    __slots__ = ("_source", "_role", "_invariants", "_closed")

    def __init__(
        self,
        source: "V1CompactDeviceStructuralSource",
        *,
        role: str,
    ) -> None:
        self._source = source
        self._role = role
        self._invariants = source._acquire_lease(role=role)
        self._closed = False

    @property
    def role(self) -> str:
        return self._role

    def _tensor(self, role: str) -> Any:
        if self._closed:
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.EVICTED,
                "structural-source lease is closed",
            )
        return self._source._leased_tensor(role, self._invariants)

    @property
    def resource_tensor(self) -> Any:
        return self._tensor("resource")

    @property
    def eligible_mask_tensor(self) -> Any:
        return self._tensor("eligible_mask")

    @property
    def required_hard_action_tensor(self) -> Any:
        return self._tensor("required_hard_action")

    @property
    def compact_block_coordinate_tensor(self) -> Any:
        return self._tensor("block_coordinates")

    @property
    def block_capacities(self) -> tuple[V1RelaxedKKTBlockCapacity, ...]:
        self._require_open()
        return self._source.block_capacities

    @property
    def quadratic_budget(self) -> float:
        self._require_open()
        return self._source.quadratic_budget

    @property
    def receipt(self) -> V1CompactDeviceStructuralSourceReceipt:
        self._require_open()
        return self._source.receipt

    def device_problem_v2_structural_kwargs(self) -> dict[str, object]:
        """Return the exact structural keyword joint consumed by DeviceProblemV2.

        The returned tensors remain borrowed aliases.  The caller must keep
        this lease open for the complete lifetime of the receiving problem or
        transfer equivalent custody at the integration boundary.
        """

        receipt = self.receipt
        return {
            "quadratic_budget": receipt.quadratic_budget,
            "block_capacities": self.block_capacities,
            "structural_source_receipt_hash": receipt.receipt_hash,
            "resource_tensor": self.resource_tensor,
            "eligible_mask_tensor": self.eligible_mask_tensor,
            "required_hard_action_tensor": self.required_hard_action_tensor,
            "resource_tensor_identity_hash": receipt.resource_tensor_identity_hash,
            "eligible_mask_identity_hash": receipt.eligible_mask_identity_hash,
            "required_mask_identity_hash": (
                receipt.required_hard_action_identity_hash
            ),
            "registered_legal_resource_maximum": (
                receipt.registered_legal_resource_maximum
            ),
            "registered_required_resource_value": (
                receipt.registered_required_resource_value
            ),
            "structural_source_receipt_semantics": (
                receipt.structural_source_receipt_semantics
            ),
        }

    def device_phase_a_structural_kwargs(self) -> dict[str, object]:
        """Return structural keywords for floor-free Device Phase A."""

        receipt = self.receipt
        return {
            "resource_tensor": self.resource_tensor,
            "eligible_mask_tensor": self.eligible_mask_tensor,
            "required_hard_action_tensor": self.required_hard_action_tensor,
            "quadratic_budget": receipt.quadratic_budget,
            "block_capacities": self.block_capacities,
            "route_axis_hash": receipt.route_axis_hash,
            "structural_source_receipt_hash": receipt.receipt_hash,
            "resource_tensor_identity_hash": receipt.resource_tensor_identity_hash,
            "eligible_mask_identity_hash": receipt.eligible_mask_identity_hash,
            "required_mask_identity_hash": (
                receipt.required_hard_action_identity_hash
            ),
        }

    @property
    def storage_invariants(self) -> tuple[tuple[str, tuple[object, ...]], ...]:
        self._require_open()
        return tuple((role, self._invariants[role]) for role in sorted(self._invariants))

    def _require_open(self) -> None:
        if self._closed:
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.EVICTED,
                "structural-source lease is closed",
            )
        self._source.assert_live_identity()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._source._release_lease(self._invariants, role=self._role)
        finally:
            self._closed = True

    def __enter__(self) -> "V1CompactDeviceStructuralSourceLease":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __reduce__(self) -> NoReturn:
        raise TypeError("structural-source leases are nonserializable")

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("structural-source leases are nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("structural-source leases cannot be copied")

    def __deepcopy__(self, _memo: dict[int, object]) -> NoReturn:
        raise TypeError("structural-source leases cannot be copied")


class V1CompactDeviceStructuralSource:
    """Issuer-only owner of compact structural proposal coordinates."""

    __slots__ = (
        "_receipt",
        "_capacity_receipt",
        "_tensors",
        "_invariants",
        "_active_lease_count",
        "_active_lease_roles",
        "_evicted",
        "_integrity_failed",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        receipt: V1CompactDeviceStructuralSourceReceipt,
        capacity_receipt: V1CompactDeviceBlockCapacityReceipt,
        tensors: dict[str, Any],
        _issuance_token: object,
    ) -> None:
        if (
            _issuance_token is not _SOURCE_ISSUANCE_TOKEN
            or type(receipt) is not V1CompactDeviceStructuralSourceReceipt
            or type(capacity_receipt) is not V1CompactDeviceBlockCapacityReceipt
        ):
            raise PermissionError("structural sources are issued only by reviewed factories")
        receipt.assert_integrity()
        capacity_receipt.assert_integrity()
        if receipt.capacity_receipt_hash != capacity_receipt.receipt_hash:
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.STALE_IDENTITY,
                "structural receipt does not own the supplied capacity receipt",
            )
        expected = {"resource", "eligible_mask", "required_hard_action", "block_coordinates"}
        if set(tensors) != expected or torch is None or any(
            type(value) is not torch.Tensor for value in tensors.values()
        ):
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.INVALID_STRUCTURAL_TENSOR,
                "structural source tensor roles are incomplete",
            )
        resource = tensors["resource"]
        eligible = tensors["eligible_mask"]
        required = tensors["required_hard_action"]
        coordinates = tensors["block_coordinates"]
        if (
            tuple(resource.shape) != (receipt.route_count,)
            or resource.dtype != torch.float64
            or tuple(eligible.shape) != (receipt.route_count,)
            or eligible.dtype != torch.bool
            or tuple(required.shape) != (receipt.route_count,)
            or required.dtype != torch.bool
            or tuple(coordinates.shape) != (receipt.block_count, _BLOCK_COORDINATE_WIDTH)
            or coordinates.dtype != torch.int64
            or len({str(value.device) for value in tensors.values()}) != 1
            or str(resource.device) != receipt.declared_device
            or any(value.requires_grad for value in tensors.values())
        ):
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.INVALID_STRUCTURAL_TENSOR,
                "structural tensors differ from their issued receipt",
            )
        self._receipt = receipt
        self._capacity_receipt = capacity_receipt
        self._tensors = dict(tensors)
        self._invariants = {
            role: _tensor_storage_invariant(value)
            for role, value in self._tensors.items()
        }
        self._active_lease_count = 0
        self._active_lease_roles = {
            "phase_a": 0,
            "phase_b": 0,
            "phase_a_phase_b": 0,
        }
        self._evicted = False
        self._integrity_failed = False
        self.assert_live_identity()

    @property
    def receipt(self) -> V1CompactDeviceStructuralSourceReceipt:
        self.assert_live_identity()
        return self._receipt

    @property
    def receipt_hash(self) -> str:
        return self.receipt.receipt_hash

    @property
    def capacity_receipt(self) -> V1CompactDeviceBlockCapacityReceipt:
        self.assert_live_identity()
        return self._capacity_receipt

    @property
    def block_capacities(self) -> tuple[V1RelaxedKKTBlockCapacity, ...]:
        return self.capacity_receipt.block_capacities

    @property
    def quadratic_budget(self) -> float:
        return self.receipt.quadratic_budget

    @property
    def route_count(self) -> int:
        return self.receipt.route_count

    @property
    def active_lease_count(self) -> int:
        return self._active_lease_count

    @property
    def active_phase_a_lease_count(self) -> int:
        return self._active_lease_roles["phase_a"]

    @property
    def active_phase_b_lease_count(self) -> int:
        return self._active_lease_roles["phase_b"]

    @property
    def is_evicted(self) -> bool:
        return self._evicted

    @property
    def retained_tensor_storage_bytes(self) -> int:
        self.assert_live_identity()
        unique: dict[tuple[str, int], int] = {}
        for tensor in self._tensors.values():
            key = (str(tensor.device), int(tensor.untyped_storage().data_ptr()))
            unique[key] = int(tensor.untyped_storage().nbytes())
        return sum(unique.values())

    def assert_live_identity(self) -> str:
        if self._evicted:
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.EVICTED,
                "structural source was evicted",
            )
        if self._integrity_failed:
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.MUTATED_CUSTODY,
                "structural source previously detected tensor mutation",
            )
        self._receipt.assert_integrity()
        self._capacity_receipt.assert_integrity()
        for role, tensor in self._tensors.items():
            if _tensor_storage_invariant(tensor) != self._invariants[role]:
                self._integrity_failed = True
                _refuse(
                    V1CompactDeviceStructuralSourceRefusalCode.MUTATED_CUSTODY,
                    f"structural {role} storage or tensor version changed",
                )
        return content_hash(
            {
                "receipt_hash": self._receipt.receipt_hash,
                "capacity_receipt_hash": self._capacity_receipt.receipt_hash,
                "storage_invariants": tuple(
                    (role, self._invariants[role]) for role in sorted(self._invariants)
                ),
                "process_local_nonserializable": True,
            },
            prefix=_LIVE_PREFIX,
        )

    def borrow_for_phase_a_phase_b(
        self,
        *,
        expected_receipt_hash: str,
        expected_route_axis_hash: str,
        expected_current_mask_hash: str,
        expected_native_fiber_axis_receipt_hash: str,
        expected_structural_registration_hash: str,
        expected_device: object,
    ) -> V1CompactDeviceStructuralSourceLease:
        return self._borrow_registered(
            role="phase_a_phase_b",
            expected_receipt_hash=expected_receipt_hash,
            expected_route_axis_hash=expected_route_axis_hash,
            expected_current_mask_hash=expected_current_mask_hash,
            expected_native_fiber_axis_receipt_hash=(
                expected_native_fiber_axis_receipt_hash
            ),
            expected_structural_registration_hash=(
                expected_structural_registration_hash
            ),
            expected_device=expected_device,
        )

    def borrow_for_phase_a(
        self,
        *,
        expected_receipt_hash: str,
        expected_route_axis_hash: str,
        expected_current_mask_hash: str,
        expected_native_fiber_axis_receipt_hash: str,
        expected_structural_registration_hash: str,
        expected_device: object,
    ) -> V1CompactDeviceStructuralSourceLease:
        """Borrow structural coordinates for Phase A without reserving Phase B."""

        return self._borrow_registered(
            role="phase_a",
            expected_receipt_hash=expected_receipt_hash,
            expected_route_axis_hash=expected_route_axis_hash,
            expected_current_mask_hash=expected_current_mask_hash,
            expected_native_fiber_axis_receipt_hash=(
                expected_native_fiber_axis_receipt_hash
            ),
            expected_structural_registration_hash=(
                expected_structural_registration_hash
            ),
            expected_device=expected_device,
        )

    def borrow_for_phase_b(
        self,
        *,
        expected_receipt_hash: str,
        expected_route_axis_hash: str,
        expected_current_mask_hash: str,
        expected_native_fiber_axis_receipt_hash: str,
        expected_structural_registration_hash: str,
        expected_device: object,
    ) -> V1CompactDeviceStructuralSourceLease:
        """Borrow structural coordinates after Phase A has released its lease."""

        return self._borrow_registered(
            role="phase_b",
            expected_receipt_hash=expected_receipt_hash,
            expected_route_axis_hash=expected_route_axis_hash,
            expected_current_mask_hash=expected_current_mask_hash,
            expected_native_fiber_axis_receipt_hash=(
                expected_native_fiber_axis_receipt_hash
            ),
            expected_structural_registration_hash=(
                expected_structural_registration_hash
            ),
            expected_device=expected_device,
        )

    def _borrow_registered(
        self,
        *,
        role: str,
        expected_receipt_hash: str,
        expected_route_axis_hash: str,
        expected_current_mask_hash: str,
        expected_native_fiber_axis_receipt_hash: str,
        expected_structural_registration_hash: str,
        expected_device: object,
    ) -> V1CompactDeviceStructuralSourceLease:
        if role not in self._active_lease_roles:
            raise AssertionError("unknown internal structural lease role")
        receipt = self.receipt
        requested = (
            (_identity(expected_receipt_hash, name="expected receipt hash"), receipt.receipt_hash),
            (_identity(expected_route_axis_hash, name="expected route axis hash"), receipt.route_axis_hash),
            (_identity(expected_current_mask_hash, name="expected current mask hash"), receipt.current_mask_hash),
            (
                _identity(
                    expected_native_fiber_axis_receipt_hash,
                    name="expected native-fiber receipt hash",
                ),
                receipt.native_fiber_axis_receipt_hash,
            ),
            (
                _identity(
                    expected_structural_registration_hash,
                    name="expected structural registration hash",
                ),
                receipt.structural_registration_hash,
            ),
            (str(_device(expected_device)), receipt.declared_device),
        )
        if any(observed != registered for observed, registered in requested):
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.STALE_IDENTITY,
                f"{role} borrow requested stale structural custody",
            )
        return V1CompactDeviceStructuralSourceLease(self, role=role)

    def _acquire_lease(self, *, role: str) -> dict[str, tuple[object, ...]]:
        self.assert_live_identity()
        if role not in self._active_lease_roles:
            raise AssertionError("unknown internal structural lease role")
        self._active_lease_count += 1
        self._active_lease_roles[role] += 1
        return dict(self._invariants)

    def _leased_tensor(
        self,
        role: str,
        invariants: dict[str, tuple[object, ...]],
    ) -> Any:
        self.assert_live_identity()
        if invariants != self._invariants or role not in self._tensors:
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.MUTATED_CUSTODY,
                "structural lease belongs to stale or unknown storage",
            )
        return self._tensors[role]

    def _release_lease(
        self,
        invariants: dict[str, tuple[object, ...]],
        *,
        role: str,
    ) -> None:
        if (
            role not in self._active_lease_roles
            or self._active_lease_count <= 0
            or self._active_lease_roles[role] <= 0
        ):
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.MUTATED_CUSTODY,
                "structural-source lease ledger underflowed",
            )
        self._active_lease_count -= 1
        self._active_lease_roles[role] -= 1
        if self._evicted or invariants != self._invariants:
            self._integrity_failed = True
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.MUTATED_CUSTODY,
                "structural lease was released against stale storage",
            )
        self.assert_live_identity()

    def evict(self) -> None:
        if self._active_lease_count:
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.ACTIVE_LEASE,
                "cannot evict structural custody while a lease is active",
            )
        self.assert_live_identity()
        self._tensors = {}
        self._invariants = {}
        self._evicted = True

    def __reduce__(self) -> NoReturn:
        raise TypeError("structural-source handles are nonserializable")

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("structural-source handles are nonserializable")

    def __getstate__(self) -> NoReturn:
        raise TypeError("structural-source handles are nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("structural-source handles cannot be copied")

    def __deepcopy__(self, _memo: dict[int, object]) -> NoReturn:
        raise TypeError("structural-source handles cannot be copied")


def _issue_source(
    *,
    current_mask: HardMask,
    route_axis: DeclaredRouteAxis,
    native_fiber_axis_receipt_hash: str,
    structural_registration_hash: str,
    quadratic_budget: float,
    declared_device: object,
    execution_mode: V1CompactDeviceStructuralSourceExecutionMode,
    construction_path: V1CompactDeviceStructuralSourcePath,
    maximum_unpack_workspace_bytes: int,
    memory_forecast: V1CompactDeviceStructuralSourceHBMForecast | None,
    resource_tensor: Any | None,
    eligible_mask_tensor: Any | None,
    required_hard_action_tensor: Any | None,
    resource_tensor_identity_hash: str | None,
    eligible_mask_identity_hash: str | None,
    required_hard_action_identity_hash: str | None,
) -> V1CompactDeviceStructuralSource:
    routes, blocks = _validate_axis_and_mask(
        current_mask=current_mask,
        route_axis=route_axis,
    )
    device = _device(declared_device)
    _validate_execution_device(device=device, execution_mode=execution_mode)
    native_identity = _identity(
        native_fiber_axis_receipt_hash,
        name="native-fiber axis receipt hash",
    )
    structural_identity = _identity(
        structural_registration_hash,
        name="structural registration hash",
    )
    q_budget = _nonnegative_finite(quadratic_budget, name="quadratic budget")
    _chunk_routes, _workspace_bytes = _normalize_unpack_workspace(
        route_count=routes,
        maximum_unpack_workspace_bytes=maximum_unpack_workspace_bytes,
    )

    explicit = construction_path is (
        V1CompactDeviceStructuralSourcePath.EXACT_SAME_DEVICE_TENSORS
    )
    if explicit:
        resource = _exact_same_device_tensor(
            resource_tensor,
            route_count=routes,
            dtype=torch.float64,
            device=device,
            name="resource tensor",
        )
        eligible = _exact_same_device_tensor(
            eligible_mask_tensor,
            route_count=routes,
            dtype=torch.bool,
            device=device,
            name="eligible-mask tensor",
        )
        required = _exact_same_device_tensor(
            required_hard_action_tensor,
            route_count=routes,
            dtype=torch.bool,
            device=device,
            name="required-hard-action tensor",
        )
        borrowed_bytes = _unique_storage_bytes((resource, eligible, required))
        resource_identity = _identity(
            resource_tensor_identity_hash,
            name="resource tensor identity hash",
        )
        eligible_identity = _identity(
            eligible_mask_identity_hash,
            name="eligible-mask identity hash",
        )
        required_identity = _identity(
            required_hard_action_identity_hash,
            name="required-hard-action identity hash",
        )
    else:
        if any(
            value is not None
            for value in (
                resource_tensor,
                eligible_mask_tensor,
                required_hard_action_tensor,
                resource_tensor_identity_hash,
                eligible_mask_identity_hash,
                required_hard_action_identity_hash,
            )
        ):
            _refuse(
                V1CompactDeviceStructuralSourceRefusalCode.INVALID_DESIGN,
                "intrinsic path cannot accept explicit structural tensors or identities",
            )
        borrowed_bytes = 0
        resource_identity = _intrinsic_coordinate_hash(
            kind="unit_resource_scalar_broadcast_v1",
            route_axis=route_axis,
            current_mask_hash=current_mask.mask_hash,
        )
        eligible_identity = _intrinsic_coordinate_hash(
            kind="packed_current_mask_device_eligibility_v1",
            route_axis=route_axis,
            current_mask_hash=current_mask.mask_hash,
        )
        required_identity = _intrinsic_coordinate_hash(
            kind="empty_required_action_scalar_broadcast_v1",
            route_axis=route_axis,
            current_mask_hash=current_mask.mask_hash,
        )

    forecast = memory_forecast
    if forecast is None:
        forecast = _runtime_forecast(
            route_count=routes,
            block_count=len(blocks),
            construction_path=construction_path,
            device=device,
            maximum_unpack_workspace_bytes=maximum_unpack_workspace_bytes,
            borrowed_structural_tensor_storage_bytes=borrowed_bytes,
        )
    if type(forecast) is not V1CompactDeviceStructuralSourceHBMForecast:
        _refuse(
            V1CompactDeviceStructuralSourceRefusalCode.INVALID_TYPE,
            "memory forecast must have exact structural forecast type",
        )
    forecast.assert_admissible()
    if (
        forecast.route_count != routes
        or forecast.block_count != len(blocks)
        or forecast.construction_path is not construction_path
        or forecast.borrowed_structural_tensor_storage_bytes != borrowed_bytes
    ):
        _refuse(
            V1CompactDeviceStructuralSourceRefusalCode.STALE_IDENTITY,
            "memory forecast differs from the registered structural construction",
        )

    current_live = _unpack_packed_big_endian_mask_on_device(
        current_mask=current_mask,
        device=device,
        maximum_unpack_workspace_bytes=maximum_unpack_workspace_bytes,
    )
    capacity_receipt = _derive_block_capacities(
        current_live_mask=current_live,
        current_mask=current_mask,
        route_axis=route_axis,
    )
    if explicit:
        eligible_is_current = _chunked_explicit_tensor_validation(
            resource=resource,
            eligible=eligible,
            required=required,
            current_live=current_live,
            maximum_chunk_routes=_chunk_routes,
        )
        for block in capacity_receipt.block_capacities:
            required_count = int(
                torch.count_nonzero(required[block.start : block.stop]).item()
            )
            if required_count > block.removal_capacity:
                _refuse(
                    V1CompactDeviceStructuralSourceRefusalCode.INFEASIBLE_REQUIRED_ACTION,
                    "required hard action exceeds a one-live-survivor block capacity",
                )
        legal_resource_maximum, required_resource_value = (
            _registered_explicit_resource_scalars(
                resource=resource,
                eligible=eligible,
                required=required,
                capacity_receipt=capacity_receipt,
            )
        )
        resource_intrinsic = False
        required_intrinsic = False
    else:
        eligible = current_live
        resource = torch.ones((), dtype=torch.float64, device=device).expand(routes)
        required = torch.zeros((), dtype=torch.bool, device=device).expand(routes)
        eligible_is_current = True
        resource_intrinsic = True
        required_intrinsic = True
        legal_resource_maximum = float(
            sum(
                block.removal_capacity
                for block in capacity_receipt.block_capacities
            )
        )
        required_resource_value = 0.0

    block_coordinates = torch.tensor(
        tuple(
            (block.start, block.stop, block.removal_capacity)
            for block in capacity_receipt.block_capacities
        ),
        dtype=torch.int64,
        device=device,
    )
    receipt = V1CompactDeviceStructuralSourceReceipt(
        capacity_receipt_hash=capacity_receipt.receipt_hash,
        route_axis_hash=route_axis.route_axis_hash,
        route_axis_content_hash=route_axis.content_hash,
        route_id_axis_hash=route_axis.route_id_axis_hash,
        current_mask_hash=current_mask.mask_hash,
        native_fiber_axis_receipt_hash=native_identity,
        structural_registration_hash=structural_identity,
        resource_tensor_identity_hash=resource_identity,
        eligible_mask_identity_hash=eligible_identity,
        required_hard_action_identity_hash=required_identity,
        route_count=routes,
        block_count=len(blocks),
        quadratic_budget=q_budget,
        registered_legal_resource_maximum=legal_resource_maximum,
        registered_required_resource_value=required_resource_value,
        declared_device=str(device),
        execution_mode=execution_mode,
        construction_path=construction_path,
        resource_storage_bytes=_tensor_storage_bytes(resource),
        eligible_mask_storage_bytes=_tensor_storage_bytes(eligible),
        required_hard_action_storage_bytes=_tensor_storage_bytes(required),
        compact_block_coordinate_storage_bytes=_tensor_storage_bytes(block_coordinates),
        resource_is_intrinsic_unit=resource_intrinsic,
        eligible_mask_is_exact_current_mask=eligible_is_current,
        required_hard_action_is_intrinsic_empty=required_intrinsic,
        phase_a_structural_input_authority=(
            execution_mode
            is V1CompactDeviceStructuralSourceExecutionMode.CUDA_PRODUCTION
        ),
        phase_b_structural_input_authority=(
            execution_mode
            is V1CompactDeviceStructuralSourceExecutionMode.CUDA_PRODUCTION
        ),
        _issuance_token=_RECEIPT_ISSUANCE_TOKEN,
    )
    return V1CompactDeviceStructuralSource(
        receipt=receipt,
        capacity_receipt=capacity_receipt,
        tensors={
            "resource": resource,
            "eligible_mask": eligible,
            "required_hard_action": required,
            "block_coordinates": block_coordinates,
        },
        _issuance_token=_SOURCE_ISSUANCE_TOKEN,
    )


def issue_v1_compact_device_structural_source_from_hard_mask(
    *,
    current_mask: HardMask,
    route_axis: DeclaredRouteAxis,
    native_fiber_axis_receipt_hash: str,
    structural_registration_hash: str,
    quadratic_budget: float,
    declared_device: object,
    execution_mode: V1CompactDeviceStructuralSourceExecutionMode,
    memory_forecast: V1CompactDeviceStructuralSourceHBMForecast | None = None,
    maximum_unpack_workspace_bytes: int = (
        COMPACT_DEVICE_STRUCTURAL_SOURCE_DEFAULT_MAXIMUM_UNPACK_WORKSPACE_BYTES
    ),
) -> V1CompactDeviceStructuralSource:
    """Issue canonical unit-resource/current-live/empty-required custody."""

    return _issue_source(
        current_mask=current_mask,
        route_axis=route_axis,
        native_fiber_axis_receipt_hash=native_fiber_axis_receipt_hash,
        structural_registration_hash=structural_registration_hash,
        quadratic_budget=quadratic_budget,
        declared_device=declared_device,
        execution_mode=execution_mode,
        construction_path=(
            V1CompactDeviceStructuralSourcePath.PACKED_CURRENT_MASK_INTRINSIC
        ),
        maximum_unpack_workspace_bytes=maximum_unpack_workspace_bytes,
        memory_forecast=memory_forecast,
        resource_tensor=None,
        eligible_mask_tensor=None,
        required_hard_action_tensor=None,
        resource_tensor_identity_hash=None,
        eligible_mask_identity_hash=None,
        required_hard_action_identity_hash=None,
    )


def issue_v1_compact_device_structural_source_from_exact_device_tensors(
    *,
    current_mask: HardMask,
    route_axis: DeclaredRouteAxis,
    native_fiber_axis_receipt_hash: str,
    structural_registration_hash: str,
    quadratic_budget: float,
    declared_device: object,
    execution_mode: V1CompactDeviceStructuralSourceExecutionMode,
    resource_tensor: Any,
    eligible_mask_tensor: Any,
    required_hard_action_tensor: Any,
    resource_tensor_identity_hash: str,
    eligible_mask_identity_hash: str,
    required_hard_action_identity_hash: str,
    memory_forecast: V1CompactDeviceStructuralSourceHBMForecast | None = None,
    maximum_unpack_workspace_bytes: int = (
        COMPACT_DEVICE_STRUCTURAL_SOURCE_DEFAULT_MAXIMUM_UNPACK_WORKSPACE_BYTES
    ),
) -> V1CompactDeviceStructuralSource:
    """Issue the separately identified exact same-device tensor path."""

    return _issue_source(
        current_mask=current_mask,
        route_axis=route_axis,
        native_fiber_axis_receipt_hash=native_fiber_axis_receipt_hash,
        structural_registration_hash=structural_registration_hash,
        quadratic_budget=quadratic_budget,
        declared_device=declared_device,
        execution_mode=execution_mode,
        construction_path=V1CompactDeviceStructuralSourcePath.EXACT_SAME_DEVICE_TENSORS,
        maximum_unpack_workspace_bytes=maximum_unpack_workspace_bytes,
        memory_forecast=memory_forecast,
        resource_tensor=resource_tensor,
        eligible_mask_tensor=eligible_mask_tensor,
        required_hard_action_tensor=required_hard_action_tensor,
        resource_tensor_identity_hash=resource_tensor_identity_hash,
        eligible_mask_identity_hash=eligible_mask_identity_hash,
        required_hard_action_identity_hash=required_hard_action_identity_hash,
    )


__all__ = [
    "COMPACT_DEVICE_STRUCTURAL_SOURCE_DEFAULT_MAXIMUM_UNPACK_WORKSPACE_BYTES",
    "COMPACT_DEVICE_STRUCTURAL_SOURCE_HBM_FRACTION",
    "COMPACT_DEVICE_STRUCTURAL_SOURCE_KKT_RECEIPT_SEMANTICS_V2",
    "V1CompactDeviceBlockCapacityReceipt",
    "V1CompactDeviceStructuralSource",
    "V1CompactDeviceStructuralSourceExecutionMode",
    "V1CompactDeviceStructuralSourceHBMForecast",
    "V1CompactDeviceStructuralSourceLease",
    "V1CompactDeviceStructuralSourcePath",
    "V1CompactDeviceStructuralSourceReceipt",
    "V1CompactDeviceStructuralSourceRefusal",
    "V1CompactDeviceStructuralSourceRefusalCode",
    "forecast_v1_compact_device_structural_source_hbm",
    "issue_v1_compact_device_structural_source_from_exact_device_tensors",
    "issue_v1_compact_device_structural_source_from_hard_mask",
]
