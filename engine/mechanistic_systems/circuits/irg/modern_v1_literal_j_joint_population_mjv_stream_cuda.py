"""Bounded route-major full-population literal-J ``M v`` stream.

This sibling consumes the *unopened* action-complete residual stream.  It
zips independently windowed target and complement raw literal-J attachments,
reconstructs their registered per-stratum ``M v`` intervals from the retained
pass-1 projections, and mixes them with the exact action-complete ``D_fixed``
source family using the population registration's rational masses.

The emitted direction axis is always

``(outside(action 0), ..., outside(action A-1), shell basis 0, ..., basis B-1)``.

For ``D_fixed``, the outside row for action ``a`` comes from row zero of that
action's affine-shell replay.  The redundant shell rows are intersected across
*all* action replays before population mixing.  This is a source-consistency
check, not an averaging operation.

Only one ``(A+B) x route_tile`` center/radius pair may be live.  The original
residual pass-2 cursor is never opened or advanced; an exclusive typed claim
prevents it from being consumed in parallel or replayed after this sibling has
read source rows.  No direction-by-route bank is retained, and no route-scale
tensor is transferred to the host.
Production is CUDA-only.  The private CPU entry point is a bounded numerical
reference and grants no production or downstream decision authority.
"""

from __future__ import annotations

from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
from fractions import Fraction
import math
from types import MappingProxyType
from typing import Any, Final, Iterator, Mapping, NoReturn

from mechanistic_systems.contracts.content_address import content_hash

from . import modern_v1_literal_j_joint_affine_shell_residual_population_stream_cuda as _residual
from .modern_v1_literal_j_d_fixed_residual_population_projection import (
    V1LiteralJDFixedResidualPopulationProjectionRegistration,
)
from .modern_v1_literal_j_joint_affine_shell_family_cuda import (
    V1LiteralJJointAffineShellFamilyDeviceHandle,
    V1LiteralJJointAffineShellPreoutcomeRegistration,
)
from .modern_v1_literal_j_joint_affine_shell_loewner_stream_bridge import (
    _validate_d1_atomic_population_sources_nonauthoritative,
    _validate_lineage as _validate_joint_population_lineage,
)
from .modern_v1_literal_j_joint_affine_shell_d1_population_stream_cuda import (
    V1LiteralJJointAffineShellD1PopulationStreamRefusal,
    V1LiteralJJointAffineShellD1PopulationStreamDeviceHandle,
    V1LiteralJJointAffineShellD1PopulationStreamTileDeviceHandle,
)
from .modern_v1_literal_j_joint_affine_shell_residual_population_stream_cuda import (
    V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle,
    _MIN_SUBNORMAL,
    _add_up,
    _div_up,
    _gamma,
    _mul_up,
    _scale_interval_by_uniform_mean,
    _serfling_factor,
    _stage_source_tile,
)
from .modern_v1_literal_j_two_pass_d_fixed_shard_replay import (
    V1LiteralJAffineShellShardReplayResult,
)
from .modern_v1_sampled_dual_chart_row_custody import (
    V1SampledDualChartRowShardAttachment,
)
from .modern_v1_literal_j_joint_population_source import (
    V1LiteralJJointPopulationSource,
)
from .modern_v1_literal_j_joint_population_action_axis import (
    V1LiteralJJointPopulationActionAxisSource,
)

try:  # pragma: no cover - runtime-image dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


LITERAL_J_JOINT_POPULATION_MJV_STREAM_FORECAST_SCHEMA: Final = (
    "partition_free_v1_literal_J_joint_population_Mv_stream_hbm_forecast_v1"
)
LITERAL_J_JOINT_POPULATION_MJV_STREAM_TILE_SCHEMA: Final = (
    "partition_free_v1_literal_J_joint_population_Mv_stream_tile_v1"
)
LITERAL_J_JOINT_POPULATION_MJV_STREAM_EXECUTION_SCHEMA: Final = (
    "partition_free_v1_literal_J_joint_population_Mv_stream_execution_v2"
)
LITERAL_J_JOINT_POPULATION_MJV_STREAM_ALGORITHM: Final = (
    "route_major_target_complement_window_zip_exact_rational_D_fixed_residual_"
    "population_Mv_interval_stream_with_exclusive_source_claim_v2"
)
LITERAL_J_JOINT_POPULATION_MJV_STREAM_AUTHORITY: Final = (
    "full_population_literal_J_Mv_interval_numerics_only_no_D_face_C_K_O_commit_"
    "runtime_terminal_or_pareto_authority_v1"
)
LITERAL_J_JOINT_D1_POPULATION_MJV_STREAM_FORECAST_SCHEMA: Final = (
    "partition_free_v1_literal_J_joint_D1_population_Mv_stream_hbm_forecast_v1"
)
LITERAL_J_JOINT_D1_POPULATION_MJV_STREAM_TILE_SCHEMA: Final = (
    "partition_free_v1_literal_J_joint_D1_population_Mv_stream_tile_v1"
)
LITERAL_J_JOINT_D1_POPULATION_MJV_STREAM_EXECUTION_SCHEMA: Final = (
    "partition_free_v1_literal_J_joint_D1_population_Mv_stream_execution_v2"
)
LITERAL_J_JOINT_D1_POPULATION_MJV_STREAM_ALGORITHM: Final = (
    "exclusive_joint_D1_and_segmented_residual_route_stream_union_boundary_"
    "exact_rational_population_interval_mix_no_direction_route_bank_v1"
)
LITERAL_J_JOINT_D1_POPULATION_MJV_STREAM_AUTHORITY: Final = (
    "action_complete_source_bound_full_population_literal_J_Mv_interval_"
    "numerics_only_private_atomic_tuple_reference_nonauthoritative_no_D_face_"
    "C_K_O_commit_runtime_terminal_or_pareto_authority_v1"
)

_PREFIX: Final = "partition-free-v1-literal-j-joint-population-mjv-stream-"
_FORECAST_TOKEN: Final = object()
_TILE_TOKEN: Final = object()
_EXECUTION_TOKEN: Final = object()
_HANDLE_TOKEN: Final = object()
_F64_BYTES: Final = 8
_CPU_REFERENCE_ROUTE_LIMIT: Final = 4096

__all__ = (
    "LITERAL_J_JOINT_POPULATION_MJV_STREAM_FORECAST_SCHEMA",
    "LITERAL_J_JOINT_POPULATION_MJV_STREAM_TILE_SCHEMA",
    "LITERAL_J_JOINT_POPULATION_MJV_STREAM_EXECUTION_SCHEMA",
    "LITERAL_J_JOINT_POPULATION_MJV_STREAM_ALGORITHM",
    "LITERAL_J_JOINT_POPULATION_MJV_STREAM_AUTHORITY",
    "LITERAL_J_JOINT_D1_POPULATION_MJV_STREAM_FORECAST_SCHEMA",
    "LITERAL_J_JOINT_D1_POPULATION_MJV_STREAM_TILE_SCHEMA",
    "LITERAL_J_JOINT_D1_POPULATION_MJV_STREAM_EXECUTION_SCHEMA",
    "LITERAL_J_JOINT_D1_POPULATION_MJV_STREAM_ALGORITHM",
    "LITERAL_J_JOINT_D1_POPULATION_MJV_STREAM_AUTHORITY",
    "V1LiteralJJointPopulationMvStreamCUDARefusalCode",
    "V1LiteralJJointPopulationMvStreamCUDARefusal",
    "V1LiteralJJointPopulationMvStreamCUDAHBMForecast",
    "V1LiteralJJointPopulationMvStreamTileReceipt",
    "V1LiteralJJointPopulationMvStreamTileLease",
    "V1LiteralJJointPopulationMvStreamTileDeviceHandle",
    "V1LiteralJJointPopulationMvStreamCUDAExecutionReceipt",
    "V1LiteralJJointPopulationMvStreamCUDAHandle",
    "open_v1_literal_j_joint_population_mjv_stream_cuda",
    "V1LiteralJJointD1PopulationMvStreamCUDAHBMForecast",
    "V1LiteralJJointD1PopulationMvStreamTileReceipt",
    "V1LiteralJJointD1PopulationMvStreamCUDAExecutionReceipt",
    "V1LiteralJJointD1PopulationMvStreamCUDAHandle",
    "open_v1_literal_j_joint_d1_population_mjv_stream_cuda",
    "_open_v1_literal_j_joint_d1_population_mjv_stream_cpu_reference",
)


class V1LiteralJJointPopulationMvStreamCUDARefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_IDENTITY = "stale_identity"
    SOURCE_MISMATCH = "source_mismatch"
    ORDER_COVERAGE_MISMATCH = "order_coverage_mismatch"
    DEVICE_REQUIRED = "device_required"
    CPU_PRODUCTION_FALLBACK = "cpu_production_fallback"
    DETERMINISM_DISABLED = "determinism_disabled"
    HBM_FORECAST_EXCEEDED = "hbm_forecast_exceeded"
    SUPPORT_VIOLATION = "support_violation"
    NUMERICAL_REFUSAL = "numerical_refusal"
    EMPTY_INTERSECTION = "empty_intersection"
    ACTIVE_TILE = "active_tile"
    STREAM_NOT_EXHAUSTED = "stream_not_exhausted"
    MUTATED_CUSTODY = "mutated_custody"
    EVICTED = "evicted"
    ACTION_COMPLETE_SOURCE_BINDING_REQUIRED = (
        "action_complete_source_binding_required"
    )
    AUTHORITY_VIOLATION = "authority_violation"


class V1LiteralJJointPopulationMvStreamCUDARefusal(RuntimeError):
    def __init__(
        self,
        code: V1LiteralJJointPopulationMvStreamCUDARefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1LiteralJJointPopulationMvStreamCUDARefusalCode:
            raise TypeError("population-Mv refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1LiteralJJointPopulationMvStreamCUDARefusalCode,
    message: str,
) -> NoReturn:
    raise V1LiteralJJointPopulationMvStreamCUDARefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1LiteralJJointPopulationMvStreamCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty exact string",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _refuse(
            V1LiteralJJointPopulationMvStreamCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one positive exact integer",
        )
    return int(value)


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _refuse(
            V1LiteralJJointPopulationMvStreamCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative exact integer",
        )
    return int(value)


def _identity_axis(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        _refuse(
            V1LiteralJJointPopulationMvStreamCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one immutable tuple",
        )
    result = tuple(_identity(item, name=f"{name} entry") for item in value)
    if not result or len(result) != len(set(result)):
        _refuse(
            V1LiteralJJointPopulationMvStreamCUDARefusalCode.INVALID_TYPE,
            f"{name} must be nonempty and unique",
        )
    return result


def _payload_hash(payload: Mapping[str, object], *, role: str) -> str:
    return content_hash(payload, prefix=_PREFIX + role + "-")


def _tensor_invariant(value: Any) -> tuple[object, ...]:
    return (
        id(value),
        int(value.data_ptr()),
        str(value.device),
        str(value.dtype),
        tuple(int(item) for item in value.shape),
        tuple(int(item) for item in value.stride()),
        int(value.storage_offset()),
        int(value._version),
        bool(value.requires_grad),
    )


def _require_tensor(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    device: object | None = None,
    nonnegative: bool = False,
) -> Any:
    if (
        torch is None
        or type(value) is not torch.Tensor
        or value.dtype != torch.float64
        or tuple(int(item) for item in value.shape) != shape
        or not value.is_contiguous()
        or bool(value.requires_grad)
        or not bool(torch.isfinite(value).all().item())
        or (device is not None and value.device != device)
    ):
        _refuse(
            V1LiteralJJointPopulationMvStreamCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one finite contiguous detached FP64 tensor with shape {shape}",
        )
    if nonnegative and bool((value < 0.0).any().item()):
        _refuse(
            V1LiteralJJointPopulationMvStreamCUDARefusalCode.NUMERICAL_REFUSAL,
            f"{name} contains a negative interval radius",
        )
    return value


def _down(value: Any) -> Any:
    return torch.nextafter(value, torch.full_like(value, -math.inf))


def _up(value: Any) -> Any:
    return torch.nextafter(value, torch.full_like(value, math.inf))


def _bounds(center: Any, radius: Any) -> tuple[Any, Any]:
    radius = torch.clamp(radius, min=0.0)
    return _down(center - radius), _up(center + radius)


def _center_radius(lower: Any, upper: Any) -> tuple[Any, Any]:
    if bool((lower > upper).any().item()):
        _refuse(
            V1LiteralJJointPopulationMvStreamCUDARefusalCode.EMPTY_INTERSECTION,
            "an interval intersection is empty",
        )
    center = lower * 0.5 + upper * 0.5
    radius = torch.maximum(_up(center - lower), _up(upper - center))
    radius = _up(torch.clamp(radius, min=0.0) + _MIN_SUBNORMAL)
    if not bool(torch.isfinite(center).all().item()) or not bool(
        torch.isfinite(radius).all().item()
    ):
        _refuse(
            V1LiteralJJointPopulationMvStreamCUDARefusalCode.NUMERICAL_REFUSAL,
            "finite interval conversion overflowed binary64",
        )
    return center.contiguous(), radius.contiguous()


def _weight_bounds(weight: Fraction) -> tuple[float, float]:
    if weight < 0:
        _refuse(
            V1LiteralJJointPopulationMvStreamCUDARefusalCode.AUTHORITY_VIOLATION,
            "population mixture weights must be nonnegative",
        )
    if weight == 0:
        return 0.0, 0.0
    center = float(weight)
    if not math.isfinite(center):
        _refuse(
            V1LiteralJJointPopulationMvStreamCUDARefusalCode.NUMERICAL_REFUSAL,
            "a rational population mass overflows binary64",
        )
    lower = math.nextafter(center, -math.inf)
    upper = math.nextafter(center, math.inf)
    if not (Fraction.from_float(lower) <= weight <= Fraction.from_float(upper)):
        _refuse(
            V1LiteralJJointPopulationMvStreamCUDARefusalCode.NUMERICAL_REFUSAL,
            "binary64 rational-mass enclosure failed",
        )
    return lower, upper


def _scale_bounds(lower: Any, upper: Any, weight: Fraction) -> tuple[Any, Any]:
    if weight == 0:
        zero = torch.zeros_like(lower)
        return zero, zero.clone()
    weight_lower, weight_upper = _weight_bounds(weight)
    candidates = torch.stack(
        (
            lower * weight_lower,
            lower * weight_upper,
            upper * weight_lower,
            upper * weight_upper,
        ),
        dim=0,
    )
    result_lower = _down(torch.amin(candidates, dim=0))
    result_upper = _up(torch.amax(candidates, dim=0))
    if not bool(torch.isfinite(result_lower).all().item()) or not bool(
        torch.isfinite(result_upper).all().item()
    ):
        _refuse(
            V1LiteralJJointPopulationMvStreamCUDARefusalCode.NUMERICAL_REFUSAL,
            "population mass scaling overflowed binary64",
        )
    return result_lower, result_upper


def _mix_population_bounds(
    components: tuple[tuple[Any, Any, Fraction], ...],
) -> tuple[Any, Any]:
    if not components:
        raise ValueError("population interval mixture requires components")
    lower, upper = _scale_bounds(*components[0])
    for next_lower, next_upper, weight in components[1:]:
        scaled_lower, scaled_upper = _scale_bounds(next_lower, next_upper, weight)
        lower = _down(lower + scaled_lower)
        upper = _up(upper + scaled_upper)
    if not bool(torch.isfinite(lower).all().item()) or not bool(
        torch.isfinite(upper).all().item()
    ):
        _refuse(
            V1LiteralJJointPopulationMvStreamCUDARefusalCode.NUMERICAL_REFUSAL,
            "population interval addition overflowed binary64",
        )
    return lower.contiguous(), upper.contiguous()


@dataclass(frozen=True, slots=True)
class V1LiteralJJointPopulationMvStreamCUDAHBMForecast:
    joint_preoutcome_registration_hash: str
    joint_family_receipt_hash: str
    population_projection_registration_hash: str
    residual_stream_registration_hash: str
    residual_stream_forecast_hash: str
    d_fixed_projection_receipt_hashes: tuple[str, ...]
    measurement_device_identity: str
    route_count: int
    action_count: int
    shell_size: int
    direction_count: int
    observed_counts: tuple[int, int]
    route_tile_size: int
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
    source_resident_storage_excluded_from_incremental_peak: bool = True
    route_scale_d2h_forecasted: bool = False
    direction_route_bank_forecasted: bool = False
    schema: str = LITERAL_J_JOINT_POPULATION_MJV_STREAM_FORECAST_SCHEMA
    _admission_memory: InitVar[tuple[int, int] | None] = None
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _admission_memory: tuple[int, int] | None, _issuance_token: object | None) -> None:
        available, allocated = _admission_memory or (self.device_free_memory_bytes, self.current_reserved_device_bytes)
        if _issuance_token is not _FORECAST_TOKEN:
            raise PermissionError("population-Mv forecasts require their issuer")
        for name in (
            "joint_preoutcome_registration_hash",
            "joint_family_receipt_hash",
            "population_projection_registration_hash",
            "residual_stream_registration_hash",
            "residual_stream_forecast_hash",
            "measurement_device_identity",
        ):
            _identity(getattr(self, name), name=name)
        hashes = _identity_axis(
            self.d_fixed_projection_receipt_hashes,
            name="d_fixed_projection_receipt_hashes",
        )
        for name in (
            "route_count",
            "action_count",
            "shell_size",
            "direction_count",
            "route_tile_size",
            "device_total_memory_bytes",
        ):
            _positive_int(getattr(self, name), name=name)
        for name in (
            "projected_incremental_peak_bytes",
            "current_reserved_device_bytes",
            "projected_reserved_device_bytes",
            "device_free_memory_bytes",
            "hbm_ceiling_bytes",
        ):
            _nonnegative_int(getattr(self, name), name=name)
        if (
            type(self.observed_counts) is not tuple
            or len(self.observed_counts) != 2
            or any(type(item) is not int or item <= 0 for item in self.observed_counts)
            or type(self.component_bytes) is not tuple
            or not self.component_bytes
            or any(
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not str
                or not item[0]
                or type(item[1]) is not int
                or item[1] < 0
                for item in self.component_bytes
            )
            or len(hashes) != self.action_count
            or self.direction_count != self.action_count + self.shell_size
            or self.projected_incremental_peak_bytes
            != sum(value for _name, value in self.component_bytes)
            or self.projected_reserved_device_bytes
            != self.current_reserved_device_bytes
            + self.projected_incremental_peak_bytes
            or self.hbm_ceiling_fraction != 0.85
            or self.hbm_ceiling_bytes
            != math.floor(self.device_total_memory_bytes * 0.85)
            or self.admitted_by_ceiling
            != (allocated + self.projected_incremental_peak_bytes <= self.hbm_ceiling_bytes)
            or self.admitted_by_free_memory
            != (self.projected_incremental_peak_bytes <= available)
            or self.admitted
            != (self.admitted_by_ceiling and self.admitted_by_free_memory)
            or self.production_cuda_execution == self.private_cpu_reference_execution
            or self.source_resident_storage_excluded_from_incremental_peak is not True
            or self.route_scale_d2h_forecasted is not False
            or self.direction_route_bank_forecasted is not False
            or self.schema != LITERAL_J_JOINT_POPULATION_MJV_STREAM_FORECAST_SCHEMA
        ):
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.AUTHORITY_VIOLATION,
                "population-Mv forecast crossed its bounded-workspace contract",
            )
        object.__setattr__(self, "d_fixed_projection_receipt_hashes", hashes)
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

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash or (
            expected_hash is not None and observed != expected_hash
        ):
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.MUTATED_CUSTODY,
                "population-Mv forecast no longer replays",
            )
        return observed

    @property
    def forecast_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1LiteralJJointD1PopulationMvStreamCUDAHBMForecast:
    action_axis_preoutcome_receipt_hash: str
    action_axis_source_receipt_hash: str
    joint_preoutcome_registration_hash: str
    joint_family_receipt_hash: str
    d1_stream_receipt_hash: str
    residual_stream_registration_hash: str
    measurement_device_identity: str
    route_count: int
    action_count: int
    shell_size: int
    direction_count: int
    d1_context_count: int
    residual_observed_counts: tuple[int, int]
    total_source_row_count: int
    d1_parent_route_tile_size: int
    residual_parent_route_tile_size: int
    maximum_output_route_tile_size: int
    source_resident_projection_bytes: int
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
    source_packet_count_is_not_a_forecast_axis: bool = True
    source_resident_storage_excluded_from_incremental_peak: bool = True
    route_scale_d2h_forecasted: bool = False
    direction_route_bank_forecasted: bool = False
    schema: str = LITERAL_J_JOINT_D1_POPULATION_MJV_STREAM_FORECAST_SCHEMA
    _admission_memory: InitVar[tuple[int, int] | None] = None
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _admission_memory: tuple[int, int] | None, _issuance_token: object | None) -> None:
        available, allocated = _admission_memory or (self.device_free_memory_bytes, self.current_reserved_device_bytes)
        if _issuance_token is not _FORECAST_TOKEN:
            raise PermissionError("D1 population-Mv forecasts require their issuer")
        for name in (
            "action_axis_preoutcome_receipt_hash",
            "action_axis_source_receipt_hash",
            "joint_preoutcome_registration_hash",
            "joint_family_receipt_hash",
            "d1_stream_receipt_hash",
            "residual_stream_registration_hash",
            "measurement_device_identity",
        ):
            _identity(getattr(self, name), name=name)
        for name in (
            "route_count",
            "action_count",
            "shell_size",
            "direction_count",
            "d1_context_count",
            "total_source_row_count",
            "d1_parent_route_tile_size",
            "residual_parent_route_tile_size",
            "maximum_output_route_tile_size",
            "device_total_memory_bytes",
        ):
            _positive_int(getattr(self, name), name=name)
        for name in (
            "source_resident_projection_bytes",
            "projected_incremental_peak_bytes",
            "current_reserved_device_bytes",
            "projected_reserved_device_bytes",
            "device_free_memory_bytes",
            "hbm_ceiling_bytes",
        ):
            _nonnegative_int(getattr(self, name), name=name)
        if (
            type(self.residual_observed_counts) is not tuple
            or len(self.residual_observed_counts) != 2
            or any(type(value) is not int or value <= 0 for value in self.residual_observed_counts)
            or self.direction_count != self.action_count + self.shell_size
            or self.total_source_row_count
            != self.d1_context_count + sum(self.residual_observed_counts)
            or self.source_resident_projection_bytes
            != 2 * self.total_source_row_count * self.direction_count * _F64_BYTES
            or type(self.component_bytes) is not tuple
            or not self.component_bytes
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
            or self.hbm_ceiling_bytes
            != math.floor(self.device_total_memory_bytes * self.hbm_ceiling_fraction)
            or self.admitted_by_ceiling
            != (allocated + self.projected_incremental_peak_bytes <= self.hbm_ceiling_bytes)
            or self.admitted_by_free_memory
            != (self.projected_incremental_peak_bytes <= available)
            or self.admitted != (self.admitted_by_ceiling and self.admitted_by_free_memory)
            or self.production_cuda_execution == self.private_cpu_reference_execution
            or self.source_packet_count_is_not_a_forecast_axis is not True
            or self.source_resident_storage_excluded_from_incremental_peak is not True
            or self.route_scale_d2h_forecasted is not False
            or self.direction_route_bank_forecasted is not False
            or self.schema != LITERAL_J_JOINT_D1_POPULATION_MJV_STREAM_FORECAST_SCHEMA
        ):
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.AUTHORITY_VIOLATION,
                "D1 population-Mv forecast crossed its row-count/bounded-workspace contract",
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
        return _payload_hash(self._payload(), role="d1-forecast")

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.MUTATED_CUSTODY,
                "D1 population-Mv forecast no longer replays",
            )
        return observed

    @property
    def forecast_hash(self) -> str:
        return self.assert_integrity()


class _AttachmentWindowCursor:
    """One verified immutable attachment cursor at an arbitrary window layout."""

    __slots__ = (
        "attachment",
        "route_count",
        "_iterator",
        "_current",
        "_position",
        "_next_window_start",
        "window_read_count",
    )

    def __init__(
        self,
        attachment: V1SampledDualChartRowShardAttachment,
        *,
        route_count: int,
    ) -> None:
        if type(attachment) is not V1SampledDualChartRowShardAttachment:
            raise TypeError("population-Mv zipper requires exact shard attachments")
        self.attachment = attachment
        self.route_count = route_count
        self._iterator: Iterator[tuple[int, int, object]] = attachment.iter_route_windows(
            verify=True
        )
        self._current: tuple[int, int, object] | None = None
        self._position = 0
        self._next_window_start = 0
        self.window_read_count = 0

    @property
    def position(self) -> int:
        return self._position

    def _load(self) -> None:
        if self._current is not None or self._position >= self.route_count:
            return
        try:
            start, stop, rows = next(self._iterator)
        except StopIteration:
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.ORDER_COVERAGE_MISMATCH,
                "a raw literal-J attachment ended before the registered route axis",
            )
        if (
            type(start) is not int
            or type(stop) is not int
            or start != self._next_window_start
            or start != self._position
            or not start < stop <= self.route_count
            or getattr(rows, "ndim", None) != 2
            or int(rows.shape[0]) != len(self.attachment.receipt.context_ids)
            or int(rows.shape[1]) != stop - start
        ):
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.ORDER_COVERAGE_MISMATCH,
                "one raw literal-J attachment has a gap, overlap, or malformed window",
            )
        self._current = (start, stop, rows)
        self._next_window_start = stop
        self.window_read_count += 1

    def available_stop(self) -> int:
        self._load()
        if self._current is None:
            return self.route_count
        return self._current[1]

    def take(self, stop: int) -> object:
        self._load()
        if self._current is None:
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.ORDER_COVERAGE_MISMATCH,
                "attempted to read past a raw literal-J attachment",
            )
        start, parent_stop, rows = self._current
        if not self._position == start or not self._position < stop <= parent_stop:
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.ORDER_COVERAGE_MISMATCH,
                "paired-window zipper requested a noncurrent route interval",
            )
        width = stop - self._position
        result = rows[:, :width]
        if stop == parent_stop:
            self._current = None
        else:
            self._current = (stop, parent_stop, rows[:, width:])
        self._position = stop
        return result

    def assert_exhausted(self) -> None:
        if self._position != self.route_count or self._current is not None:
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.ORDER_COVERAGE_MISMATCH,
                "raw attachment route coverage is incomplete",
            )
        try:
            next(self._iterator)
        except StopIteration:
            return
        _refuse(
            V1LiteralJJointPopulationMvStreamCUDARefusalCode.ORDER_COVERAGE_MISMATCH,
            "raw attachment emitted routes after the registered axis",
        )


class _PairedAttachmentWindowZipper:
    __slots__ = ("_cursors", "_route_count", "_tile_size", "tile_count")

    def __init__(
        self,
        attachments: tuple[
            V1SampledDualChartRowShardAttachment,
            V1SampledDualChartRowShardAttachment,
        ],
        *,
        route_count: int,
        tile_size: int,
    ) -> None:
        self._cursors = tuple(
            _AttachmentWindowCursor(item, route_count=route_count)
            for item in attachments
        )
        self._route_count = route_count
        self._tile_size = tile_size
        self.tile_count = 0

    @property
    def position(self) -> int:
        first, second = self._cursors
        if first.position != second.position:
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.ORDER_COVERAGE_MISMATCH,
                "target/complement raw cursors left a common route position",
            )
        return first.position

    @property
    def window_read_counts(self) -> tuple[int, int]:
        return tuple(item.window_read_count for item in self._cursors)  # type: ignore[return-value]

    def next(self, *, stop_limit: int | None = None) -> tuple[int, int, object, object] | None:
        start = self.position
        if start == self._route_count:
            for cursor in self._cursors:
                cursor.assert_exhausted()
            return None
        if stop_limit is not None and (
            type(stop_limit) is not int or not start < stop_limit <= self._route_count
        ):
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.ORDER_COVERAGE_MISMATCH,
                "paired-window zipper received an invalid external route boundary",
            )
        stops = tuple(cursor.available_stop() for cursor in self._cursors)
        stop = min(
            start + self._tile_size,
            stops[0],
            stops[1],
            self._route_count if stop_limit is None else stop_limit,
        )
        if stop <= start:
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.ORDER_COVERAGE_MISMATCH,
                "paired-window zipper made no route progress",
            )
        target = self._cursors[0].take(stop)
        complement = self._cursors[1].take(stop)
        self.tile_count += 1
        return start, stop, target, complement


def _residual_component_bounds(
    *,
    parent_rows: object,
    projection: Any,
    projection_radius: Any,
    literal_bound: Any,
    projection_bound: Any,
    population_count: int,
    alpha: float,
) -> tuple[Any, Any, int]:
    rows = _stage_source_tile(parent_rows, device=projection.device)
    width = int(rows.shape[1])
    route_bound = literal_bound
    if tuple(route_bound.shape) != (width,):
        _refuse(
            V1LiteralJJointPopulationMvStreamCUDARefusalCode.ORDER_COVERAGE_MISMATCH,
            "route support slice differs from the zipped raw tile",
        )
    if bool((torch.abs(rows) > route_bound.reshape(1, -1)).any().item()):
        _refuse(
            V1LiteralJJointPopulationMvStreamCUDARefusalCode.SUPPORT_VIOLATION,
            "zipped residual literal-J tile exceeded its analytic route support",
        )
    center_sum = projection.T @ rows
    abs_rows = torch.abs(rows)
    round_mass = torch.abs(projection).T @ abs_rows
    input_uncertainty = projection_radius.T @ abs_rows
    operations = 2 * int(projection.shape[0]) + 1
    relative = _gamma(operations)
    if not math.isfinite(relative) or relative >= 1.0:
        _refuse(
            V1LiteralJJointPopulationMvStreamCUDARefusalCode.NUMERICAL_REFUSAL,
            "residual Mv FP64 error factor is not finite below one",
        )
    recovery = _div_up(relative, 1.0 - relative)
    uncertainty_upper = _add_up(
        input_uncertainty,
        _mul_up(input_uncertainty, recovery),
    )
    sum_radius = _add_up(
        _add_up(_mul_up(round_mass, recovery), uncertainty_upper),
        _mul_up(float(operations), _MIN_SUBNORMAL),
    )
    center, numeric_radius = _scale_interval_by_uniform_mean(
        center_sum=center_sum,
        radius_sum=sum_radius,
        count=int(projection.shape[0]),
    )
    factor = _serfling_factor(
        observed=int(projection.shape[0]),
        population=population_count,
        alpha=alpha,
    )
    if factor == 0.0:
        sampling_radius = torch.zeros_like(center)
    else:
        sampling_radius = _mul_up(
            _mul_up(
                _mul_up(2.0, factor),
                projection_bound.reshape(-1, 1),
            ),
            route_bound.reshape(1, -1),
        )
    radius = _add_up(numeric_radius, sampling_radius).contiguous()
    if not bool(torch.isfinite(center).all().item()) or not bool(
        torch.isfinite(radius).all().item()
    ) or bool((radius < 0.0).any().item()):
        _refuse(
            V1LiteralJJointPopulationMvStreamCUDARefusalCode.NUMERICAL_REFUSAL,
            "residual Mv interval is not finite and outward",
        )
    return (*_bounds(center.contiguous(), radius), int(rows.numel()) * _F64_BYTES)


def _d_fixed_component_bounds(
    *,
    d_fixed_results: tuple[V1LiteralJAffineShellShardReplayResult, ...],
    action_count: int,
    shell_size: int,
    route_start: int,
    route_stop: int,
    route_count: int,
    device: object,
) -> tuple[Any, Any]:
    width = route_stop - route_start
    direction_count = action_count + shell_size
    lower = torch.empty(
        (direction_count, width), dtype=torch.float64, device=device
    )
    upper = torch.empty_like(lower)
    shared_shell_lower = shared_shell_upper = None
    for action, result in enumerate(d_fixed_results):
        result.projection.device_handle.assert_live()
        with result.projection.device_handle.borrow() as lease:
            center = _require_tensor(
                lease.mjz_center,
                name="D_fixed Mv center",
                shape=(shell_size + 1, route_count),
                device=device,
            )[:, route_start:route_stop].contiguous()
            radius = _require_tensor(
                lease.mjz_radius,
                name="D_fixed Mv radius",
                shape=(shell_size + 1, route_count),
                device=device,
                nonnegative=True,
            )[:, route_start:route_stop].contiguous()
            candidate_lower, candidate_upper = _bounds(center, radius)
        lower[action] = candidate_lower[0]
        upper[action] = candidate_upper[0]
        if shared_shell_lower is None:
            shared_shell_lower = candidate_lower[1:].clone()
            shared_shell_upper = candidate_upper[1:].clone()
        else:
            shared_shell_lower = torch.maximum(
                shared_shell_lower, candidate_lower[1:]
            )
            shared_shell_upper = torch.minimum(
                shared_shell_upper, candidate_upper[1:]
            )
            if bool((shared_shell_lower > shared_shell_upper).any().item()):
                _refuse(
                    V1LiteralJJointPopulationMvStreamCUDARefusalCode.EMPTY_INTERSECTION,
                    "redundant D_fixed shell-basis Mv rows have empty intersection",
                )
    if shared_shell_lower is None or shared_shell_upper is None:
        raise AssertionError("validated action-complete D_fixed family disappeared")
    lower[action_count:] = shared_shell_lower
    upper[action_count:] = shared_shell_upper
    return lower.contiguous(), upper.contiguous()


@dataclass(frozen=True, slots=True)
class V1LiteralJJointPopulationMvStreamTileReceipt:
    forecast_hash: str
    residual_stream_registration_hash: str
    population_projection_registration_hash: str
    joint_direction_axis_hash: str
    direction_ids: tuple[str, ...]
    target_shard_receipt_hash: str
    complement_shard_receipt_hash: str
    target_content_sha256: str
    complement_content_sha256: str
    tile_ordinal: int
    route_start: int
    route_stop: int
    route_count: int
    action_count: int
    shell_size: int
    direction_count: int
    d_fixed_mass_fraction: tuple[int, int]
    residual_mass_fractions: tuple[tuple[int, int], tuple[int, int]]
    exact_rational_population_mixing: bool = True
    redundant_d_fixed_shell_rows_intersected: bool = True
    target_complement_window_intersection: bool = True
    route_scale_d2h: bool = False
    d_face_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    runtime_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = LITERAL_J_JOINT_POPULATION_MJV_STREAM_TILE_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _TILE_TOKEN:
            raise PermissionError("population-Mv tile receipts require their issuer")
        for name in (
            "forecast_hash",
            "residual_stream_registration_hash",
            "population_projection_registration_hash",
            "joint_direction_axis_hash",
            "target_shard_receipt_hash",
            "complement_shard_receipt_hash",
            "target_content_sha256",
            "complement_content_sha256",
        ):
            _identity(getattr(self, name), name=name)
        ids = _identity_axis(self.direction_ids, name="direction_ids")
        if (
            type(self.tile_ordinal) is not int
            or self.tile_ordinal < 0
            or not 0 <= self.route_start < self.route_stop <= self.route_count
            or self.action_count <= 0
            or self.shell_size <= 0
            or self.direction_count != self.action_count + self.shell_size
            or len(ids) != self.direction_count
            or Fraction(*self.d_fixed_mass_fraction)
            + Fraction(*self.residual_mass_fractions[0])
            + Fraction(*self.residual_mass_fractions[1])
            != 1
            or any(
                type(value) is not bool or value is not True
                for value in (
                    self.exact_rational_population_mixing,
                    self.redundant_d_fixed_shell_rows_intersected,
                    self.target_complement_window_intersection,
                )
            )
            or any(
                type(value) is not bool or value is not False
                for value in (
                    self.route_scale_d2h,
                    self.d_face_authority,
                    self.c_stream_authority,
                    self.calibration_authority,
                    self.oracle_authority,
                    self.commit_authority,
                    self.runtime_authority,
                    self.terminal_authority,
                    self.pareto_authority,
                )
            )
            or self.schema != LITERAL_J_JOINT_POPULATION_MJV_STREAM_TILE_SCHEMA
        ):
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.AUTHORITY_VIOLATION,
                "population-Mv tile receipt crossed its interval-only authority",
            )
        object.__setattr__(self, "direction_ids", ids)
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
        return _payload_hash(self._payload(), role="tile")

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash or (
            expected_hash is not None and observed != expected_hash
        ):
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.MUTATED_CUSTODY,
                "population-Mv tile receipt no longer replays",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1LiteralJJointD1PopulationMvStreamTileReceipt:
    forecast_hash: str
    action_axis_source_receipt_hash: str
    d1_stream_receipt_hash: str
    d1_parent_tile_receipt_hash: str
    residual_stream_registration_hash: str
    residual_target_shard_receipt_hash: str
    residual_complement_shard_receipt_hash: str
    residual_target_content_sha256: str
    residual_complement_content_sha256: str
    joint_direction_axis_hash: str
    direction_ids: tuple[str, ...]
    tile_ordinal: int
    route_start: int
    route_stop: int
    route_count: int
    action_count: int
    shell_size: int
    direction_count: int
    d1_mass_fraction: tuple[int, int]
    residual_mass_fractions: tuple[tuple[int, int], tuple[int, int]]
    exact_rational_population_mixing: bool = True
    union_parent_route_boundaries: bool = True
    centers_may_cancel_radii_may_not: bool = True
    exclusive_d1_and_residual_claims: bool = True
    route_scale_d2h: bool = False
    direction_route_bank_retained: bool = False
    d_face_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    runtime_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = LITERAL_J_JOINT_D1_POPULATION_MJV_STREAM_TILE_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _TILE_TOKEN:
            raise PermissionError("D1 population-Mv tile receipts require their issuer")
        for name in (
            "forecast_hash",
            "action_axis_source_receipt_hash",
            "d1_stream_receipt_hash",
            "d1_parent_tile_receipt_hash",
            "residual_stream_registration_hash",
            "residual_target_shard_receipt_hash",
            "residual_complement_shard_receipt_hash",
            "residual_target_content_sha256",
            "residual_complement_content_sha256",
            "joint_direction_axis_hash",
        ):
            _identity(getattr(self, name), name=name)
        ids = _identity_axis(self.direction_ids, name="direction_ids")
        if (
            type(self.tile_ordinal) is not int
            or self.tile_ordinal < 0
            or not 0 <= self.route_start < self.route_stop <= self.route_count
            or self.direction_count != self.action_count + self.shell_size
            or len(ids) != self.direction_count
            or Fraction(*self.d1_mass_fraction)
            + Fraction(*self.residual_mass_fractions[0])
            + Fraction(*self.residual_mass_fractions[1])
            != 1
            or any(
                type(value) is not bool or value is not True
                for value in (
                    self.exact_rational_population_mixing,
                    self.union_parent_route_boundaries,
                    self.centers_may_cancel_radii_may_not,
                    self.exclusive_d1_and_residual_claims,
                )
            )
            or any(
                type(value) is not bool or value is not False
                for value in (
                    self.route_scale_d2h,
                    self.direction_route_bank_retained,
                    self.d_face_authority,
                    self.c_stream_authority,
                    self.calibration_authority,
                    self.oracle_authority,
                    self.commit_authority,
                    self.runtime_authority,
                    self.terminal_authority,
                    self.pareto_authority,
                )
            )
            or self.schema != LITERAL_J_JOINT_D1_POPULATION_MJV_STREAM_TILE_SCHEMA
        ):
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.AUTHORITY_VIOLATION,
                "D1 population-Mv tile crossed its interval-only authority",
            )
        object.__setattr__(self, "direction_ids", ids)
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
        return _payload_hash(self._payload(), role="d1-tile")

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.MUTATED_CUSTODY,
                "D1 population-Mv tile receipt no longer replays",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


class V1LiteralJJointPopulationMvStreamTileLease:
    __slots__ = ("_owner", "_token", "_closed", "_center", "_radius")

    def __init__(
        self,
        owner: "V1LiteralJJointPopulationMvStreamTileDeviceHandle",
        token: object,
        center: Any,
        radius: Any,
    ) -> None:
        self._owner = owner
        self._token = token
        self._closed = False
        self._center = center
        self._radius = radius

    def __enter__(self) -> "V1LiteralJJointPopulationMvStreamTileLease":
        self._owner.assert_live()
        return self

    def intervals(self) -> tuple[Any, Any]:
        self._owner.assert_live()
        return self._center, self._radius

    def close(self) -> None:
        if not self._closed:
            try:
                self._owner._close_lease(self._token)
            finally:
                self._closed = True

    def __exit__(self, *_: object) -> None:
        self.close()

    def __reduce__(self) -> NoReturn:
        raise TypeError("population-Mv tile leases are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("population-Mv tile leases are nonserializable")


class V1LiteralJJointPopulationMvStreamTileDeviceHandle:
    __slots__ = (
        "receipt",
        "_center",
        "_radius",
        "_center_guard",
        "_radius_guard",
        "_center_invariant",
        "_radius_invariant",
        "_tokens",
        "_release",
        "_evicted",
    )

    def __init__(
        self,
        *,
        receipt: V1LiteralJJointPopulationMvStreamTileReceipt,
        center: Any,
        radius: Any,
        release: Any,
        issuance_token: object | None = None,
    ) -> None:
        if issuance_token is not _TILE_TOKEN:
            raise PermissionError("population-Mv tile handles require their issuer")
        self.receipt = receipt
        shape = (receipt.direction_count, receipt.route_stop - receipt.route_start)
        self._center = _require_tensor(center, name="tile center", shape=shape)
        self._radius = _require_tensor(
            radius,
            name="tile radius",
            shape=shape,
            device=self._center.device,
            nonnegative=True,
        )
        self._center_guard = self._center.detach().clone()
        self._radius_guard = self._radius.detach().clone()
        self._center_invariant = _tensor_invariant(self._center)
        self._radius_invariant = _tensor_invariant(self._radius)
        self._tokens: set[object] = set()
        self._release = release
        self._evicted = False
        self.assert_live()

    @property
    def is_evicted(self) -> bool:
        return self._evicted

    def assert_live(self) -> None:
        if self._evicted:
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.EVICTED,
                "population-Mv tile was evicted",
            )
        self.receipt.assert_integrity()
        if (
            _tensor_invariant(self._center) != self._center_invariant
            or _tensor_invariant(self._radius) != self._radius_invariant
            or not bool(torch.equal(self._center, self._center_guard))
            or not bool(torch.equal(self._radius, self._radius_guard))
        ):
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.MUTATED_CUSTODY,
                "population-Mv tile tensors mutated",
            )

    def borrow(self) -> V1LiteralJJointPopulationMvStreamTileLease:
        self.assert_live()
        token = object()
        self._tokens.add(token)
        return V1LiteralJJointPopulationMvStreamTileLease(
            self, token, self._center, self._radius
        )

    def _close_lease(self, token: object) -> None:
        if token not in self._tokens:
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.STALE_IDENTITY,
                "population-Mv tile lease token is stale",
            )
        try:
            self.assert_live()
        finally:
            self._tokens.remove(token)

    def evict(self) -> None:
        self.assert_live()
        if self._tokens:
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.ACTIVE_TILE,
                "cannot evict population-Mv tile under an active lease",
            )
        receipt_hash = self.receipt.receipt_hash
        self._center = None
        self._radius = None
        self._center_guard = None
        self._radius_guard = None
        self._evicted = True
        self._release(receipt_hash)

    def __reduce__(self) -> NoReturn:
        raise TypeError("population-Mv tile handles are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("population-Mv tile handles are nonserializable")


@dataclass(frozen=True, slots=True)
class V1LiteralJJointPopulationMvStreamCUDAExecutionReceipt:
    forecast_hash: str
    joint_preoutcome_registration_hash: str
    joint_family_receipt_hash: str
    population_projection_registration_hash: str
    residual_stream_registration_hash: str
    residual_q_receipt_hash: str
    residual_compact_shell_m_receipt_hash: str
    d_fixed_affine_replay_receipt_hashes: tuple[str, ...]
    d_fixed_projection_receipt_hashes: tuple[str, ...]
    literal_j_shard_receipt_hashes: tuple[str, str]
    literal_j_content_sha256s: tuple[str, str]
    joint_direction_axis_hash: str
    direction_ids: tuple[str, ...]
    route_count: int
    action_count: int
    shell_size: int
    direction_count: int
    d_fixed_mass_fraction: tuple[int, int]
    residual_mass_fractions: tuple[tuple[int, int], tuple[int, int]]
    residual_population_counts: tuple[int, int]
    residual_observed_counts: tuple[int, int]
    target_route_window_read_count: int
    complement_route_window_read_count: int
    zipped_route_tile_count: int
    residual_raw_source_payload_read_bytes: int
    d_fixed_projection_payload_read_bytes: int
    total_projection_scan_bytes: int
    host_to_device_payload_bytes: int
    tile_receipt_hashes: tuple[str, ...]
    route_ranges: tuple[tuple[int, int], ...]
    production_cuda_execution: bool
    private_cpu_reference_execution: bool
    exact_rational_population_mixing: bool = True
    independently_windowed_strata_zipped: bool = True
    redundant_d_fixed_shell_rows_intersected: bool = True
    original_residual_pass2_unopened: bool = True
    exclusive_residual_pass2_claim_held_during_execution: bool = True
    consumed_source_cannot_be_replayed_by_legacy_cursor: bool = True
    one_live_tile_enforced: bool = True
    full_route_cover: bool = True
    source_mutation_replayed_per_tile: bool = True
    array_free: bool = True
    context_route_bank_retained: bool = False
    direction_route_bank_retained: bool = False
    route_scale_d2h: bool = False
    d_face_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    runtime_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    algorithm: str = LITERAL_J_JOINT_POPULATION_MJV_STREAM_ALGORITHM
    authority_scope: str = LITERAL_J_JOINT_POPULATION_MJV_STREAM_AUTHORITY
    schema: str = LITERAL_J_JOINT_POPULATION_MJV_STREAM_EXECUTION_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _EXECUTION_TOKEN:
            raise PermissionError("population-Mv execution receipts require their issuer")
        for name in (
            "forecast_hash",
            "joint_preoutcome_registration_hash",
            "joint_family_receipt_hash",
            "population_projection_registration_hash",
            "residual_stream_registration_hash",
            "residual_q_receipt_hash",
            "residual_compact_shell_m_receipt_hash",
            "joint_direction_axis_hash",
        ):
            _identity(getattr(self, name), name=name)
        ids = _identity_axis(self.direction_ids, name="direction_ids")
        action_replays = _identity_axis(
            self.d_fixed_affine_replay_receipt_hashes,
            name="d_fixed_affine_replay_receipt_hashes",
        )
        action_projections = _identity_axis(
            self.d_fixed_projection_receipt_hashes,
            name="d_fixed_projection_receipt_hashes",
        )
        shard_hashes = tuple(
            _identity(item, name="literal_j_shard_receipt_hash")
            for item in self.literal_j_shard_receipt_hashes
        )
        content_hashes = tuple(
            _identity(item, name="literal_j_content_sha256")
            for item in self.literal_j_content_sha256s
        )
        tile_hashes = tuple(
            _identity(item, name="tile_receipt_hash")
            for item in self.tile_receipt_hashes
        )
        count_axes = (
            self.residual_population_counts,
            self.residual_observed_counts,
        )
        if (
            self.route_count <= 0
            or self.action_count <= 0
            or self.shell_size <= 0
            or self.direction_count != self.action_count + self.shell_size
            or len(ids) != self.direction_count
            or len(action_replays) != self.action_count
            or len(action_projections) != self.action_count
            or len(shard_hashes) != 2
            or len(content_hashes) != 2
            or len(tile_hashes) != self.zipped_route_tile_count
            or len(set(tile_hashes)) != len(tile_hashes)
            or len(self.route_ranges) != self.zipped_route_tile_count
            or not self.route_ranges
            or any(
                type(axis) is not tuple
                or len(axis) != 2
                or any(type(item) is not int or item <= 0 for item in axis)
                for axis in count_axes
            )
            or any(
                self.residual_observed_counts[h]
                > self.residual_population_counts[h]
                for h in (0, 1)
            )
            or any(
                type(value) is not int or value <= 0
                for value in (
                    self.target_route_window_read_count,
                    self.complement_route_window_read_count,
                    self.zipped_route_tile_count,
                )
            )
            or type(self.residual_raw_source_payload_read_bytes) is not int
            or self.residual_raw_source_payload_read_bytes
            != sum(self.residual_observed_counts) * self.route_count * _F64_BYTES
            or type(self.d_fixed_projection_payload_read_bytes) is not int
            or self.d_fixed_projection_payload_read_bytes
            != 2
            * self.action_count
            * (self.shell_size + 1)
            * self.route_count
            * _F64_BYTES
            or type(self.total_projection_scan_bytes) is not int
            or self.total_projection_scan_bytes
            != self.residual_raw_source_payload_read_bytes
            + self.d_fixed_projection_payload_read_bytes
            or type(self.host_to_device_payload_bytes) is not int
            or self.host_to_device_payload_bytes < 0
            or self.route_ranges[0][0] != 0
            or self.route_ranges[-1][1] != self.route_count
            or any(
                type(item) is not tuple
                or len(item) != 2
                or any(type(value) is not int for value in item)
                or not 0 <= item[0] < item[1] <= self.route_count
                for item in self.route_ranges
            )
            or any(
                self.route_ranges[index][1] != self.route_ranges[index + 1][0]
                for index in range(len(self.route_ranges) - 1)
            )
            or Fraction(*self.d_fixed_mass_fraction)
            + Fraction(*self.residual_mass_fractions[0])
            + Fraction(*self.residual_mass_fractions[1])
            != 1
            or self.production_cuda_execution == self.private_cpu_reference_execution
            or self.host_to_device_payload_bytes
            != (
                self.residual_raw_source_payload_read_bytes
                if self.production_cuda_execution
                else 0
            )
            or any(
                type(value) is not bool or value is not True
                for value in (
                    self.exact_rational_population_mixing,
                    self.independently_windowed_strata_zipped,
                    self.redundant_d_fixed_shell_rows_intersected,
                    self.original_residual_pass2_unopened,
                    self.exclusive_residual_pass2_claim_held_during_execution,
                    self.consumed_source_cannot_be_replayed_by_legacy_cursor,
                    self.one_live_tile_enforced,
                    self.full_route_cover,
                    self.source_mutation_replayed_per_tile,
                    self.array_free,
                )
            )
            or any(
                type(value) is not bool or value is not False
                for value in (
                    self.context_route_bank_retained,
                    self.direction_route_bank_retained,
                    self.route_scale_d2h,
                    self.d_face_authority,
                    self.c_stream_authority,
                    self.calibration_authority,
                    self.oracle_authority,
                    self.commit_authority,
                    self.runtime_authority,
                    self.terminal_authority,
                    self.pareto_authority,
                )
            )
            or self.algorithm != LITERAL_J_JOINT_POPULATION_MJV_STREAM_ALGORITHM
            or self.authority_scope != LITERAL_J_JOINT_POPULATION_MJV_STREAM_AUTHORITY
            or self.schema != LITERAL_J_JOINT_POPULATION_MJV_STREAM_EXECUTION_SCHEMA
        ):
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.AUTHORITY_VIOLATION,
                "population-Mv execution receipt crossed its array-free numerical scope",
            )
        object.__setattr__(self, "direction_ids", ids)
        object.__setattr__(
            self,
            "d_fixed_affine_replay_receipt_hashes",
            action_replays,
        )
        object.__setattr__(
            self,
            "d_fixed_projection_receipt_hashes",
            action_projections,
        )
        object.__setattr__(self, "literal_j_shard_receipt_hashes", shard_hashes)
        object.__setattr__(self, "literal_j_content_sha256s", content_hashes)
        object.__setattr__(self, "tile_receipt_hashes", tile_hashes)
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
        return _payload_hash(self._payload(), role="execution")

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash or (
            expected_hash is not None and observed != expected_hash
        ):
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.MUTATED_CUSTODY,
                "population-Mv execution receipt no longer replays",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1LiteralJJointD1PopulationMvStreamCUDAExecutionReceipt:
    forecast_hash: str
    action_axis_preoutcome_receipt_hash: str
    action_axis_source_receipt_hash: str
    joint_preoutcome_registration_hash: str
    joint_family_receipt_hash: str
    d1_stream_receipt_hash: str
    d1_q_receipt_hash: str
    d1_compact_shell_m_receipt_hash: str
    residual_stream_registration_hash: str
    residual_q_receipt_hash: str
    residual_compact_shell_m_receipt_hash: str
    atomic_source_receipt_hashes: tuple[str, ...]
    atomic_preoutcome_receipt_hashes: tuple[str, ...]
    d1_literal_j_shard_receipt_hashes: tuple[str, ...]
    d1_literal_j_content_sha256s: tuple[str, ...]
    residual_source_proof_closure_hash_axes: tuple[tuple[str, ...], tuple[str, ...]]
    residual_source_shard_receipt_hash_axes: tuple[tuple[str, ...], tuple[str, ...]]
    residual_source_content_sha256_axes: tuple[tuple[str, ...], tuple[str, ...]]
    joint_direction_axis_hash: str
    route_axis_hash: str
    route_id_axis_hash: str
    direction_ids: tuple[str, ...]
    route_count: int
    action_count: int
    shell_size: int
    direction_count: int
    d1_context_count: int
    residual_population_counts: tuple[int, int]
    residual_observed_counts: tuple[int, int]
    population_count: int
    d1_mass_fraction: tuple[int, int]
    residual_mass_fractions: tuple[tuple[int, int], tuple[int, int]]
    d1_parent_tile_receipt_hashes: tuple[str, ...]
    d1_parent_route_ranges: tuple[tuple[int, int], ...]
    residual_parent_pair_count: int
    residual_parent_route_ranges: tuple[tuple[int, int], ...]
    output_tile_receipt_hashes: tuple[str, ...]
    output_route_ranges: tuple[tuple[int, int], ...]
    d1_pass2_source_payload_read_bytes: int
    residual_pass2_source_payload_read_bytes: int
    total_source_payload_read_bytes: int
    host_to_device_payload_bytes: int
    production_cuda_execution: bool
    private_cpu_reference_execution: bool
    action_complete_source_binding_present: bool
    atomic_source_tuple_nonauthoritative: bool
    exact_rational_population_mixing: bool = True
    arbitrary_parent_tile_boundaries_supported: bool = True
    exclusive_d1_and_residual_claims_held: bool = True
    source_owned_route_claim_protocol_used: bool = True
    legacy_opaque_pass2_compatibility_used: bool = False
    consumed_sources_cannot_reopen: bool = True
    one_live_output_tile_enforced: bool = True
    full_route_cover: bool = True
    array_free_receipt: bool = True
    context_route_bank_retained: bool = False
    direction_route_bank_retained: bool = False
    route_scale_d2h: bool = False
    d_face_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    runtime_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    algorithm: str = LITERAL_J_JOINT_D1_POPULATION_MJV_STREAM_ALGORITHM
    authority_scope: str = LITERAL_J_JOINT_D1_POPULATION_MJV_STREAM_AUTHORITY
    schema: str = LITERAL_J_JOINT_D1_POPULATION_MJV_STREAM_EXECUTION_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _EXECUTION_TOKEN:
            raise PermissionError("D1 population-Mv execution receipts require their issuer")
        for name in (
            "forecast_hash",
            "action_axis_preoutcome_receipt_hash",
            "action_axis_source_receipt_hash",
            "joint_preoutcome_registration_hash",
            "joint_family_receipt_hash",
            "d1_stream_receipt_hash",
            "d1_q_receipt_hash",
            "d1_compact_shell_m_receipt_hash",
            "residual_stream_registration_hash",
            "residual_q_receipt_hash",
            "residual_compact_shell_m_receipt_hash",
            "joint_direction_axis_hash",
            "route_axis_hash",
            "route_id_axis_hash",
        ):
            _identity(getattr(self, name), name=name)
        ids = _identity_axis(self.direction_ids, name="direction_ids")
        atomic = _identity_axis(
            self.atomic_source_receipt_hashes,
            name="atomic_source_receipt_hashes",
        )
        atomic_pre = _identity_axis(
            self.atomic_preoutcome_receipt_hashes,
            name="atomic_preoutcome_receipt_hashes",
        )
        d1_shards = _identity_axis(
            self.d1_literal_j_shard_receipt_hashes,
            name="d1_literal_j_shard_receipt_hashes",
        )
        d1_contents = _identity_axis(
            self.d1_literal_j_content_sha256s,
            name="d1_literal_j_content_sha256s",
        )
        d1_tiles = _identity_axis(
            self.d1_parent_tile_receipt_hashes,
            name="d1_parent_tile_receipt_hashes",
        )
        output_tiles = _identity_axis(
            self.output_tile_receipt_hashes,
            name="output_tile_receipt_hashes",
        )
        if (
            self.route_count <= 0
            or self.action_count <= 0
            or self.shell_size <= 0
            or self.direction_count != self.action_count + self.shell_size
            or len(ids) != self.direction_count
            or len(atomic) != self.action_count
            or len(atomic_pre) != self.action_count
            or len(d1_shards) != len(d1_contents)
            or self.d1_context_count <= 0
            or self.population_count <= 0
            or type(self.residual_population_counts) is not tuple
            or type(self.residual_observed_counts) is not tuple
            or len(self.residual_population_counts) != 2
            or len(self.residual_observed_counts) != 2
            or any(
                self.residual_observed_counts[h] <= 0
                or self.residual_observed_counts[h] > self.residual_population_counts[h]
                for h in (0, 1)
            )
            or Fraction(*self.d1_mass_fraction)
            + Fraction(*self.residual_mass_fractions[0])
            + Fraction(*self.residual_mass_fractions[1])
            != 1
            or len(d1_tiles) != len(self.d1_parent_route_ranges)
            or len(output_tiles) != len(self.output_route_ranges)
            or self.residual_parent_pair_count <= 0
            or len(self.residual_parent_route_ranges) != self.residual_parent_pair_count
            or not self.d1_parent_route_ranges
            or not self.residual_parent_route_ranges
            or not self.output_route_ranges
            or self.d1_parent_route_ranges[0][0] != 0
            or self.d1_parent_route_ranges[-1][1] != self.route_count
            or self.residual_parent_route_ranges[0][0] != 0
            or self.residual_parent_route_ranges[-1][1] != self.route_count
            or self.output_route_ranges[0][0] != 0
            or self.output_route_ranges[-1][1] != self.route_count
            or any(
                axis[index][1] != axis[index + 1][0]
                for axis in (
                    self.d1_parent_route_ranges,
                    self.residual_parent_route_ranges,
                    self.output_route_ranges,
                )
                for index in range(len(axis) - 1)
            )
            or self.d1_pass2_source_payload_read_bytes
            != self.d1_context_count * self.route_count * _F64_BYTES
            or self.residual_pass2_source_payload_read_bytes
            != sum(self.residual_observed_counts) * self.route_count * _F64_BYTES
            or self.total_source_payload_read_bytes
            != self.d1_pass2_source_payload_read_bytes
            + self.residual_pass2_source_payload_read_bytes
            or self.host_to_device_payload_bytes
            != (self.total_source_payload_read_bytes if self.production_cuda_execution else 0)
            or self.production_cuda_execution == self.private_cpu_reference_execution
            or self.action_complete_source_binding_present != self.production_cuda_execution
            or self.atomic_source_tuple_nonauthoritative != self.private_cpu_reference_execution
        ):
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.AUTHORITY_VIOLATION,
                "D1 population-Mv execution axes, masses, coverage, or accounting do not close",
            )
        nested_axes = (
            self.residual_source_proof_closure_hash_axes,
            self.residual_source_shard_receipt_hash_axes,
            self.residual_source_content_sha256_axes,
        )
        if any(
            type(axes) is not tuple
            or len(axes) != 2
            or any(type(axis) is not tuple or not axis for axis in axes)
            for axes in nested_axes
        ) or any(
            not (
                len(nested_axes[0][h])
                == len(nested_axes[1][h])
                == len(nested_axes[2][h])
            )
            for h in (0, 1)
        ):
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.SOURCE_MISMATCH,
                "residual proof/shard/content segment axes do not nest identically",
            )
        positive = (
            self.exact_rational_population_mixing,
            self.arbitrary_parent_tile_boundaries_supported,
            self.exclusive_d1_and_residual_claims_held,
            self.source_owned_route_claim_protocol_used,
            self.consumed_sources_cannot_reopen,
            self.one_live_output_tile_enforced,
            self.full_route_cover,
            self.array_free_receipt,
        )
        closed = (
            self.context_route_bank_retained,
            self.direction_route_bank_retained,
            self.route_scale_d2h,
            self.legacy_opaque_pass2_compatibility_used,
            self.d_face_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.runtime_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if (
            any(type(value) is not bool or value is not True for value in positive)
            or any(type(value) is not bool or value is not False for value in closed)
            or self.algorithm != LITERAL_J_JOINT_D1_POPULATION_MJV_STREAM_ALGORITHM
            or self.authority_scope != LITERAL_J_JOINT_D1_POPULATION_MJV_STREAM_AUTHORITY
            or self.schema != LITERAL_J_JOINT_D1_POPULATION_MJV_STREAM_EXECUTION_SCHEMA
        ):
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.AUTHORITY_VIOLATION,
                "D1 population-Mv execution crossed its array-free interval authority",
            )
        object.__setattr__(self, "direction_ids", ids)
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
        return _payload_hash(self._payload(), role="d1-execution")

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.MUTATED_CUSTODY,
                "D1 population-Mv execution receipt no longer replays",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


def _forecast_components(
    *,
    direction_count: int,
    observed_counts: tuple[int, int],
    route_tile_size: int,
) -> tuple[tuple[str, int], ...]:
    directions = direction_count
    contexts = sum(observed_counts)
    tile = route_tile_size
    return (
        ("pass1_projection_mutation_guards", 2 * contexts * directions * _F64_BYTES),
        ("paired_raw_literal_j_staging", contexts * tile * _F64_BYTES),
        ("residual_interval_workspace", 18 * directions * tile * _F64_BYTES),
        ("d_fixed_intersection_workspace", 8 * directions * tile * _F64_BYTES),
        ("exact_mass_interval_workspace", 10 * directions * tile * _F64_BYTES),
        ("one_live_tile_center_radius_and_guards", 4 * directions * tile * _F64_BYTES),
    )


def _build_forecast(
    *,
    preregistration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    population: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    residual_stream: V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle,
    d_fixed_results: tuple[V1LiteralJAffineShellShardReplayResult, ...],
    device: object,
    production: bool,
    device_total_memory_bytes: int,
    device_free_memory_bytes: int,
    current_reserved_device_bytes: int,
    _admission_memory: tuple[int, int] | None = None,
) -> V1LiteralJJointPopulationMvStreamCUDAHBMForecast:
    route_tile = min(
        preregistration.route_count,
        residual_stream._cursor._design.route_tile_size,
    )
    components = _forecast_components(
        direction_count=preregistration.direction_count,
        observed_counts=residual_stream.registration.registered_observed_counts,
        route_tile_size=route_tile,
    )
    incremental = sum(value for _name, value in components)
    total = _positive_int(device_total_memory_bytes, name="device_total_memory_bytes")
    free = _nonnegative_int(device_free_memory_bytes, name="device_free_memory_bytes")
    reserved = _nonnegative_int(
        current_reserved_device_bytes,
        name="current_reserved_device_bytes",
    )
    ceiling = math.floor(total * 0.85)
    projected = reserved + incremental
    available, allocated = _admission_memory or (free, reserved)
    return V1LiteralJJointPopulationMvStreamCUDAHBMForecast(
        joint_preoutcome_registration_hash=preregistration.registration_hash,
        joint_family_receipt_hash=family_handle.receipt.receipt_hash,
        population_projection_registration_hash=population.registration_hash,
        residual_stream_registration_hash=residual_stream.registration.registration_hash,
        residual_stream_forecast_hash=residual_stream.forecast.forecast_hash,
        d_fixed_projection_receipt_hashes=tuple(
            item.projection.receipt.receipt_hash for item in d_fixed_results
        ),
        measurement_device_identity=str(device),
        route_count=preregistration.route_count,
        action_count=preregistration.action_count,
        shell_size=preregistration.shell_size,
        direction_count=preregistration.direction_count,
        observed_counts=residual_stream.registration.registered_observed_counts,
        route_tile_size=route_tile,
        component_bytes=components,
        projected_incremental_peak_bytes=incremental,
        current_reserved_device_bytes=reserved,
        projected_reserved_device_bytes=projected,
        device_total_memory_bytes=total,
        device_free_memory_bytes=free,
        hbm_ceiling_fraction=0.85,
        hbm_ceiling_bytes=ceiling,
        admitted_by_ceiling=allocated + incremental <= ceiling,
        admitted_by_free_memory=incremental <= available,
        admitted=allocated + incremental <= ceiling and incremental <= available,
        production_cuda_execution=production,
        private_cpu_reference_execution=not production,
        _admission_memory=_admission_memory,
        _issuance_token=_FORECAST_TOKEN,
    )


def _validate_private_source_seam(
    source: V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle,
) -> tuple[
    tuple[V1SampledDualChartRowShardAttachment, V1SampledDualChartRowShardAttachment],
    tuple[Any, Any],
    tuple[Any, Any],
    Any,
    Any,
]:
    if type(source) is not V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle:
        raise TypeError("population-Mv stream requires the exact residual stream handle")
    try:
        source.assert_live()
        source.assert_pass2_unopened()
    except _residual.V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusal as error:
        code = (
            V1LiteralJJointPopulationMvStreamCUDARefusalCode.ORDER_COVERAGE_MISMATCH
            if error.code
            is _residual.V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.PASS2_ALREADY_OPENED
            else V1LiteralJJointPopulationMvStreamCUDARefusalCode.MUTATED_CUSTODY
        )
        _refuse(
            code,
            "the residual source cannot grant unopened pass-1 custody: "
            f"{error}",
        )
    cursor = source._cursor
    if (
        type(cursor) is not _residual._Pass2Cursor
        or source._active_tile_hash is not None
        or source._tile_receipt_hashes != []
        or source._exhausted
        or cursor._attachments is not source._attachments
        or cursor._projection_centers is not source._projection_centers
        or cursor._projection_radii is not source._projection_radii
        or cursor._literal_bound is not source._literal_bound
        or cursor._projection_bound is not source._projection_bound
        or cursor._design.design_hash != source.registration.design_hash
    ):
        _refuse(
            V1LiteralJJointPopulationMvStreamCUDARefusalCode.STALE_IDENTITY,
            "the retained pass-1 source graph or unopened pass-2 cursor changed",
        )
    attachments = source._attachments
    if (
        type(attachments) is not tuple
        or len(attachments) != 2
        or any(type(item) is not V1SampledDualChartRowShardAttachment for item in attachments)
        or cursor._attachments[0] is not attachments[0]
        or cursor._attachments[1] is not attachments[1]
        or tuple(item.receipt.receipt_hash for item in attachments)
        != source.compact_shell_m_handle.receipt.literal_j_shard_receipt_hashes
        or tuple(item.receipt.content_sha256 for item in attachments)
        != source.compact_shell_m_handle.receipt.literal_j_content_sha256s
    ):
        _refuse(
            V1LiteralJJointPopulationMvStreamCUDARefusalCode.SOURCE_MISMATCH,
            "the raw target/complement attachment graph changed after pass 1",
        )
    counts = source.registration.registered_observed_counts
    directions = source.registration.direction_count
    device = source._projection_centers[0].device
    projection_centers = tuple(
        _require_tensor(
            source._projection_centers[h],
            name=f"retained stratum-{h} projection center",
            shape=(counts[h], directions),
            device=device,
        )
        for h in (0, 1)
    )
    projection_radii = tuple(
        _require_tensor(
            source._projection_radii[h],
            name=f"retained stratum-{h} projection radius",
            shape=(counts[h], directions),
            device=device,
            nonnegative=True,
        )
        for h in (0, 1)
    )
    literal_bound = _require_tensor(
        source._literal_bound,
        name="retained literal-J route support",
        shape=(source.registration.route_count,),
        device=device,
        nonnegative=True,
    )
    projection_bound = _require_tensor(
        source._projection_bound,
        name="retained projection support",
        shape=(directions,),
        device=device,
        nonnegative=True,
    )
    return (
        attachments,
        projection_centers,  # type: ignore[return-value]
        projection_radii,  # type: ignore[return-value]
        literal_bound,
        projection_bound,
    )


class V1LiteralJJointPopulationMvStreamCUDAHandle:
    """One-live-tile full-population Mv stream over a common route axis."""

    __slots__ = (
        "forecast",
        "_preregistration",
        "_family_handle",
        "_population",
        "_residual_stream",
        "_residual_pass2_claim_token",
        "_d_fixed_results",
        "_attachments",
        "_projection_centers",
        "_projection_radii",
        "_projection_guards",
        "_projection_invariants",
        "_parent_identity",
        "_literal_bound",
        "_projection_bound",
        "_zipper",
        "_production",
        "_active_tile_hash",
        "_tile_receipt_hashes",
        "_route_ranges",
        "_payload_read_bytes",
        "_source_rows_consumed",
        "_exhausted",
        "_final_receipt",
        "_evicted",
    )

    def __init__(
        self,
        *,
        forecast: V1LiteralJJointPopulationMvStreamCUDAHBMForecast,
        preregistration: V1LiteralJJointAffineShellPreoutcomeRegistration,
        family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
        population: V1LiteralJDFixedResidualPopulationProjectionRegistration,
        residual_stream: V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle,
        residual_pass2_claim_token: object,
        d_fixed_results: tuple[V1LiteralJAffineShellShardReplayResult, ...],
        attachments: tuple[
            V1SampledDualChartRowShardAttachment,
            V1SampledDualChartRowShardAttachment,
        ],
        projection_centers: tuple[Any, Any],
        projection_radii: tuple[Any, Any],
        literal_bound: Any,
        projection_bound: Any,
        production: bool,
        issuance_token: object | None = None,
    ) -> None:
        if issuance_token is not _HANDLE_TOKEN:
            raise PermissionError("population-Mv stream handles require their issuer")
        self.forecast = forecast
        self._preregistration = preregistration
        self._family_handle = family_handle
        self._population = population
        self._residual_stream = residual_stream
        self._residual_pass2_claim_token = residual_pass2_claim_token
        self._d_fixed_results = d_fixed_results
        self._attachments = attachments
        self._projection_centers = projection_centers
        self._projection_radii = projection_radii
        tensors = projection_centers + projection_radii
        self._projection_guards = tuple(item.detach().clone() for item in tensors)
        self._projection_invariants = tuple(_tensor_invariant(item) for item in tensors)
        self._parent_identity = (
            preregistration.registration_hash,
            family_handle.receipt.receipt_hash,
            population.registration_hash,
            residual_stream.registration.registration_hash,
            residual_stream.forecast.forecast_hash,
            residual_stream.q_handle.receipt.receipt_hash,
            residual_stream.compact_shell_m_handle.receipt.receipt_hash,
            tuple(item.receipt.receipt_hash for item in d_fixed_results),
            tuple(item.projection.receipt.receipt_hash for item in d_fixed_results),
            tuple(item.receipt.receipt_hash for item in attachments),
            tuple(item.receipt.content_sha256 for item in attachments),
        )
        self._literal_bound = literal_bound
        self._projection_bound = projection_bound
        self._zipper = _PairedAttachmentWindowZipper(
            attachments,
            route_count=preregistration.route_count,
            tile_size=forecast.route_tile_size,
        )
        self._production = production
        self._active_tile_hash: str | None = None
        self._tile_receipt_hashes: list[str] = []
        self._route_ranges: list[tuple[int, int]] = []
        self._payload_read_bytes = 0
        self._source_rows_consumed = False
        self._exhausted = False
        self._final_receipt: V1LiteralJJointPopulationMvStreamCUDAExecutionReceipt | None = None
        self._evicted = False
        self.assert_live()

    @property
    def exhausted(self) -> bool:
        self.assert_live()
        return self._exhausted

    @property
    def active_tile(self) -> bool:
        self.assert_live()
        return self._active_tile_hash is not None

    def assert_live(self) -> None:
        if self._evicted:
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.EVICTED,
                "population-Mv stream was evicted",
            )
        self.forecast.assert_integrity()
        try:
            self._preregistration.assert_integrity()
            self._population.assert_integrity()
            self._family_handle.receipt.assert_integrity()
            self._residual_stream.assert_live()
            self._residual_stream._assert_population_mjv_pass2_claim(
                self._residual_pass2_claim_token
            )
            for result in self._d_fixed_results:
                result.receipt.assert_integrity()
                result.source_bound_replay.receipt.assert_integrity()
                result.registration.assert_integrity()
                result.projection.device_handle.assert_live()
            self._residual_stream._row_group_commit.attachment.verify_live_witnesses()
        except V1LiteralJJointPopulationMvStreamCUDARefusal:
            raise
        except Exception as error:
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.MUTATED_CUSTODY,
                "one retained population-Mv source no longer replays: "
                f"{type(error).__name__}: {error}",
            )
        observed_parent_identity = (
            self._preregistration.registration_hash,
            self._family_handle.receipt.receipt_hash,
            self._population.registration_hash,
            self._residual_stream.registration.registration_hash,
            self._residual_stream.forecast.forecast_hash,
            self._residual_stream.q_handle.receipt.receipt_hash,
            self._residual_stream.compact_shell_m_handle.receipt.receipt_hash,
            tuple(item.receipt.receipt_hash for item in self._d_fixed_results),
            tuple(
                item.projection.receipt.receipt_hash
                for item in self._d_fixed_results
            ),
            tuple(item.receipt.receipt_hash for item in self._attachments),
            tuple(item.receipt.content_sha256 for item in self._attachments),
        )
        if (
            observed_parent_identity != self._parent_identity
            or self.forecast.joint_preoutcome_registration_hash
            != observed_parent_identity[0]
            or self.forecast.joint_family_receipt_hash
            != observed_parent_identity[1]
            or self.forecast.population_projection_registration_hash
            != observed_parent_identity[2]
            or self.forecast.residual_stream_registration_hash
            != observed_parent_identity[3]
            or self.forecast.residual_stream_forecast_hash
            != observed_parent_identity[4]
            or self.forecast.d_fixed_projection_receipt_hashes
            != observed_parent_identity[8]
        ):
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.STALE_IDENTITY,
                "the population-Mv parent object graph changed after opening",
            )
        tensors = self._projection_centers + self._projection_radii
        if any(
            _tensor_invariant(value) != invariant
            or not bool(torch.equal(value, guard))
            for value, guard, invariant in zip(
                tensors,
                self._projection_guards,
                self._projection_invariants,
                strict=True,
            )
        ):
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.MUTATED_CUSTODY,
                "retained residual pass-1 projections mutated",
            )

    def next_tile(self) -> V1LiteralJJointPopulationMvStreamTileDeviceHandle | None:
        self.assert_live()
        if self._active_tile_hash is not None:
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.ACTIVE_TILE,
                "the previous population-Mv tile must be evicted before advancement",
            )
        if self._exhausted:
            return None
        zipped = self._zipper.next()
        if zipped is None:
            self._exhausted = True
            return None
        self._source_rows_consumed = True
        start, stop, target_rows, complement_rows = zipped
        prereg = self._preregistration
        device = self._projection_centers[0].device
        route_bound = self._literal_bound[start:stop]
        residual_bounds = []
        payload_bytes = 0
        for h, rows in enumerate((target_rows, complement_rows)):
            lower, upper, payload = _residual_component_bounds(
                parent_rows=rows,
                projection=self._projection_centers[h],
                projection_radius=self._projection_radii[h],
                literal_bound=route_bound,
                projection_bound=self._projection_bound,
                population_count=self._population.residual_stratum_counts[h],
                alpha=self._residual_stream.registration.per_statement_two_sided_alpha,
            )
            residual_bounds.append((lower, upper))
            payload_bytes += payload
        d_lower, d_upper = _d_fixed_component_bounds(
            d_fixed_results=self._d_fixed_results,
            action_count=prereg.action_count,
            shell_size=prereg.shell_size,
            route_start=start,
            route_stop=stop,
            route_count=prereg.route_count,
            device=device,
        )
        d_weight = Fraction(*self._population.d_fixed_mass_fraction)
        residual_weights = tuple(
            Fraction(*item)
            for item in self._population.residual_stratum_mass_fractions
        )
        lower, upper = _mix_population_bounds(
            (
                (d_lower, d_upper, d_weight),
                (residual_bounds[0][0], residual_bounds[0][1], residual_weights[0]),
                (residual_bounds[1][0], residual_bounds[1][1], residual_weights[1]),
            )
        )
        center, radius = _center_radius(lower, upper)
        ordinal = len(self._tile_receipt_hashes)
        receipt = V1LiteralJJointPopulationMvStreamTileReceipt(
            forecast_hash=self.forecast.forecast_hash,
            residual_stream_registration_hash=self._residual_stream.registration.registration_hash,
            population_projection_registration_hash=self._population.registration_hash,
            joint_direction_axis_hash=prereg.joint_direction_axis_hash,
            direction_ids=prereg.joint_direction_ids,
            target_shard_receipt_hash=self._attachments[0].receipt.receipt_hash,
            complement_shard_receipt_hash=self._attachments[1].receipt.receipt_hash,
            target_content_sha256=self._attachments[0].receipt.content_sha256,
            complement_content_sha256=self._attachments[1].receipt.content_sha256,
            tile_ordinal=ordinal,
            route_start=start,
            route_stop=stop,
            route_count=prereg.route_count,
            action_count=prereg.action_count,
            shell_size=prereg.shell_size,
            direction_count=prereg.direction_count,
            d_fixed_mass_fraction=self._population.d_fixed_mass_fraction,
            residual_mass_fractions=self._population.residual_stratum_mass_fractions,
            _issuance_token=_TILE_TOKEN,
        )
        receipt_hash = receipt.receipt_hash
        self._active_tile_hash = receipt_hash
        self._tile_receipt_hashes.append(receipt_hash)
        self._route_ranges.append((start, stop))
        self._payload_read_bytes += payload_bytes
        return V1LiteralJJointPopulationMvStreamTileDeviceHandle(
            receipt=receipt,
            center=center,
            radius=radius,
            release=self._release_tile,
            issuance_token=_TILE_TOKEN,
        )

    def _release_tile(self, receipt_hash: str) -> None:
        if self._active_tile_hash != receipt_hash:
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.STALE_IDENTITY,
                "released population-Mv tile is not the active tile",
            )
        self._active_tile_hash = None

    def finalize(self) -> V1LiteralJJointPopulationMvStreamCUDAExecutionReceipt:
        self.assert_live()
        if self._active_tile_hash is not None:
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.ACTIVE_TILE,
                "cannot finalize with a live population-Mv tile",
            )
        if not self._exhausted:
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.STREAM_NOT_EXHAUSTED,
                "call next_tile until it returns None before finalization",
            )
        if self._final_receipt is None:
            prereg = self._preregistration
            population = self._population
            self._final_receipt = V1LiteralJJointPopulationMvStreamCUDAExecutionReceipt(
                forecast_hash=self.forecast.forecast_hash,
                joint_preoutcome_registration_hash=prereg.registration_hash,
                joint_family_receipt_hash=self._family_handle.receipt.receipt_hash,
                population_projection_registration_hash=population.registration_hash,
                residual_stream_registration_hash=self._residual_stream.registration.registration_hash,
                residual_q_receipt_hash=self._residual_stream.q_handle.receipt.receipt_hash,
                residual_compact_shell_m_receipt_hash=(
                    self._residual_stream.compact_shell_m_handle.receipt.receipt_hash
                ),
                d_fixed_affine_replay_receipt_hashes=tuple(
                    item.receipt.receipt_hash for item in self._d_fixed_results
                ),
                d_fixed_projection_receipt_hashes=tuple(
                    item.projection.receipt.receipt_hash for item in self._d_fixed_results
                ),
                literal_j_shard_receipt_hashes=tuple(
                    item.receipt.receipt_hash for item in self._attachments
                ),  # type: ignore[arg-type]
                literal_j_content_sha256s=tuple(
                    item.receipt.content_sha256 for item in self._attachments
                ),  # type: ignore[arg-type]
                joint_direction_axis_hash=prereg.joint_direction_axis_hash,
                direction_ids=prereg.joint_direction_ids,
                route_count=prereg.route_count,
                action_count=prereg.action_count,
                shell_size=prereg.shell_size,
                direction_count=prereg.direction_count,
                d_fixed_mass_fraction=population.d_fixed_mass_fraction,
                residual_mass_fractions=population.residual_stratum_mass_fractions,
                residual_population_counts=population.residual_stratum_counts,
                residual_observed_counts=self._residual_stream.registration.registered_observed_counts,
                target_route_window_read_count=self._zipper.window_read_counts[0],
                complement_route_window_read_count=self._zipper.window_read_counts[1],
                zipped_route_tile_count=self._zipper.tile_count,
                residual_raw_source_payload_read_bytes=self._payload_read_bytes,
                d_fixed_projection_payload_read_bytes=(
                    2
                    * prereg.action_count
                    * (prereg.shell_size + 1)
                    * prereg.route_count
                    * _F64_BYTES
                ),
                total_projection_scan_bytes=(
                    self._payload_read_bytes
                    + 2
                    * prereg.action_count
                    * (prereg.shell_size + 1)
                    * prereg.route_count
                    * _F64_BYTES
                ),
                host_to_device_payload_bytes=(
                    self._payload_read_bytes if self._production else 0
                ),
                tile_receipt_hashes=tuple(self._tile_receipt_hashes),
                route_ranges=tuple(self._route_ranges),
                production_cuda_execution=self._production,
                private_cpu_reference_execution=not self._production,
                _issuance_token=_EXECUTION_TOKEN,
            )
        return self._final_receipt

    def evict(self) -> None:
        replay_error: BaseException | None = None
        try:
            self.assert_live()
        except V1LiteralJJointPopulationMvStreamCUDARefusal as error:
            if error.code is V1LiteralJJointPopulationMvStreamCUDARefusalCode.EVICTED:
                raise
            replay_error = error
        if self._active_tile_hash is not None:
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.ACTIVE_TILE,
                "cannot evict population-Mv stream with a live tile",
            )
        release_error: BaseException | None = None
        try:
            self._residual_stream._release_population_mjv_pass2_claim(
                self._residual_pass2_claim_token,
                source_rows_consumed=self._source_rows_consumed,
            )
        except BaseException as error:
            release_error = error
        self._projection_centers = ()
        self._projection_radii = ()
        self._projection_guards = ()
        self._projection_invariants = ()
        self._literal_bound = None
        self._projection_bound = None
        self._zipper = None
        self._residual_pass2_claim_token = None
        self._evicted = True
        if replay_error is not None:
            raise replay_error
        if release_error is not None:
            raise V1LiteralJJointPopulationMvStreamCUDARefusal(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.MUTATED_CUSTODY,
                "exclusive residual pass-2 claim failed during eviction: "
                f"{type(release_error).__name__}: {release_error}",
            )

    def __reduce__(self) -> NoReturn:
        raise TypeError("population-Mv stream handles are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("population-Mv stream handles are nonserializable")


def _open_v1_literal_j_joint_population_mjv_stream(
    *,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    d_fixed_source_results: tuple[V1LiteralJAffineShellShardReplayResult, ...],
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    residual_stream: V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle,
    production: bool,
    device_total_memory_bytes: int | None,
    device_free_memory_bytes: int | None,
    current_reserved_device_bytes: int | None,
) -> V1LiteralJJointPopulationMvStreamCUDAHandle:
    """Open the coherent-legacy full-D-fixed population stream.

    This path remains readable for replay and for the small high-precision
    oracle.  Canonical production uses the D1-plus-residual route-claim stream
    below; retaining this function must not be interpreted as promotion of
    its opaque residual-row compatibility protocol.
    """

    if torch is None:
        _refuse(
            V1LiteralJJointPopulationMvStreamCUDARefusalCode.INVALID_TYPE,
            "population-Mv stream requires PyTorch",
        )
    exact = (
        (
            joint_preoutcome_registration,
            V1LiteralJJointAffineShellPreoutcomeRegistration,
        ),
        (joint_family_handle, V1LiteralJJointAffineShellFamilyDeviceHandle),
        (
            population_registration,
            V1LiteralJDFixedResidualPopulationProjectionRegistration,
        ),
        (
            residual_stream,
            V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle,
        ),
    )
    if any(type(value) is not owner for value, owner in exact):
        raise TypeError("population-Mv opening requires exact source owners")
    if type(d_fixed_source_results) is not tuple or any(
        type(item) is not V1LiteralJAffineShellShardReplayResult
        for item in d_fixed_source_results
    ):
        raise TypeError("D_fixed sources must be one exact immutable action tuple")
    if type(production) is not bool:
        raise TypeError("production must be one exact boolean")

    q_mode = residual_stream.q_handle.receipt.production_cuda_execution
    if production and not q_mode:
        _refuse(
            V1LiteralJJointPopulationMvStreamCUDARefusalCode.CPU_PRODUCTION_FALLBACK,
            "production population-Mv execution refuses a CPU residual source",
        )
    if not production and q_mode:
        _refuse(
            V1LiteralJJointPopulationMvStreamCUDARefusalCode.INVALID_TYPE,
            "private CPU reference cannot relabel a CUDA residual source",
        )
    try:
        _family, _q, _registrations, observed_production, joint_custody = (
            _validate_joint_population_lineage(
                preregistration=joint_preoutcome_registration,
                family_handle=joint_family_handle,
                d_fixed_results=d_fixed_source_results,
                population=population_registration,
                residual_stream=residual_stream,
                allow_private_reference_for_tests=not production,
            )
        )
    except Exception as error:
        _refuse(
            V1LiteralJJointPopulationMvStreamCUDARefusalCode.SOURCE_MISMATCH,
            "joint preregistration/family/D_fixed/residual lineage did not close: "
            f"{type(error).__name__}: {error}",
        )
    if observed_production != production or not joint_custody:
        _refuse(
            V1LiteralJJointPopulationMvStreamCUDARefusalCode.AUTHORITY_VIOLATION,
            "population-Mv execution requires the public action-complete population graph",
        )

    attachments, centers, radii, literal_bound, projection_bound = (
        _validate_private_source_seam(residual_stream)
    )
    device = centers[0].device
    if production:
        if device.type != "cuda":
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.DEVICE_REQUIRED,
                "production population-Mv execution requires CUDA",
            )
        if any(
            value is not None
            for value in (
                device_total_memory_bytes,
                device_free_memory_bytes,
                current_reserved_device_bytes,
            )
        ):
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.AUTHORITY_VIOLATION,
                "production HBM admission must use live device facts",
            )
        if not torch.are_deterministic_algorithms_enabled():
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.DETERMINISM_DISABLED,
                "production population-Mv execution requires deterministic algorithms",
            )
        free, total = torch.cuda.mem_get_info(device)
        reserved = torch.cuda.memory_reserved(device)
    else:
        if (
            device.type != "cpu"
            or joint_preoutcome_registration.route_count > _CPU_REFERENCE_ROUTE_LIMIT
        ):
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.CPU_PRODUCTION_FALLBACK,
                "private population-Mv reference is bounded to 4096 CPU routes",
            )
        total = (
            1_000_000_000
            if device_total_memory_bytes is None
            else device_total_memory_bytes
        )
        free = total if device_free_memory_bytes is None else device_free_memory_bytes
        reserved = (
            0
            if current_reserved_device_bytes is None
            else current_reserved_device_bytes
        )
    forecast = _build_forecast(
        preregistration=joint_preoutcome_registration,
        family_handle=joint_family_handle,
        population=population_registration,
        residual_stream=residual_stream,
        d_fixed_results=d_fixed_source_results,
        device=device,
        production=production,
        device_total_memory_bytes=int(total),
        device_free_memory_bytes=int(free),
        current_reserved_device_bytes=int(reserved),
        _admission_memory=((cuda_reusable_memory_bytes(device, framework=torch, free_bytes=free), int(torch.cuda.memory_allocated(device))) if production else None),
    )
    if not forecast.admitted:
        _refuse(
            V1LiteralJJointPopulationMvStreamCUDARefusalCode.HBM_FORECAST_EXCEEDED,
            "population-Mv route zipper exceeds the exact 85% incremental forecast",
        )
    claim_token = residual_stream._claim_unopened_pass2_for_population_mjv()
    try:
        return V1LiteralJJointPopulationMvStreamCUDAHandle(
            forecast=forecast,
            preregistration=joint_preoutcome_registration,
            family_handle=joint_family_handle,
            population=population_registration,
            residual_stream=residual_stream,
            residual_pass2_claim_token=claim_token,
            d_fixed_results=d_fixed_source_results,
            attachments=attachments,
            projection_centers=centers,
            projection_radii=radii,
            literal_bound=literal_bound,
            projection_bound=projection_bound,
            production=production,
            issuance_token=_HANDLE_TOKEN,
        )
    except BaseException:
        residual_stream._release_population_mjv_pass2_claim(
            claim_token,
            source_rows_consumed=False,
        )
        raise


def open_v1_literal_j_joint_population_mjv_stream_cuda(
    *,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    d_fixed_source_results: tuple[V1LiteralJAffineShellShardReplayResult, ...],
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    residual_stream: V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle,
) -> V1LiteralJJointPopulationMvStreamCUDAHandle:
    """Open the coherent-legacy CUDA full-D-fixed population stream."""

    return _open_v1_literal_j_joint_population_mjv_stream(
        joint_preoutcome_registration=joint_preoutcome_registration,
        joint_family_handle=joint_family_handle,
        d_fixed_source_results=d_fixed_source_results,
        population_registration=population_registration,
        residual_stream=residual_stream,
        production=True,
        device_total_memory_bytes=None,
        device_free_memory_bytes=None,
        current_reserved_device_bytes=None,
    )


def _open_v1_literal_j_joint_population_mjv_stream_cpu_reference(
    *,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    d_fixed_source_results: tuple[V1LiteralJAffineShellShardReplayResult, ...],
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    residual_stream: V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle,
    device_total_memory_bytes: int = 1_000_000_000,
    device_free_memory_bytes: int = 1_000_000_000,
    current_reserved_device_bytes: int = 0,
) -> V1LiteralJJointPopulationMvStreamCUDAHandle:
    """Small-domain coherent-legacy numerical oracle; never a fallback."""

    return _open_v1_literal_j_joint_population_mjv_stream(
        joint_preoutcome_registration=joint_preoutcome_registration,
        joint_family_handle=joint_family_handle,
        d_fixed_source_results=d_fixed_source_results,
        population_registration=population_registration,
        residual_stream=residual_stream,
        production=False,
        device_total_memory_bytes=device_total_memory_bytes,
        device_free_memory_bytes=device_free_memory_bytes,
        current_reserved_device_bytes=current_reserved_device_bytes,
    )


def _d1_population_mjv_forecast_components(
    *,
    direction_count: int,
    d1_parent_tile_size: int,
    residual_parent_tile_size: int,
    output_tile_size: int,
) -> tuple[tuple[str, int], ...]:
    directions = direction_count
    return (
        (
            "one_D1_parent_center_radius",
            2 * directions * d1_parent_tile_size * _F64_BYTES,
        ),
        (
            "paired_residual_parent_centers_radii",
            4 * directions * residual_parent_tile_size * _F64_BYTES,
        ),
        (
            "outward_exact_mass_interval_workspace",
            12 * directions * output_tile_size * _F64_BYTES,
        ),
        (
            "one_live_output_center_radius_and_guards",
            4 * directions * output_tile_size * _F64_BYTES,
        ),
    )


def _build_d1_population_mjv_forecast(
    *,
    preregistration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    action_axis_source: V1LiteralJJointPopulationActionAxisSource | None,
    d1_stream: V1LiteralJJointAffineShellD1PopulationStreamDeviceHandle,
    residual_stream: V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle,
    device: object,
    production: bool,
    device_total_memory_bytes: int,
    device_free_memory_bytes: int,
    current_reserved_device_bytes: int,
    _admission_memory: tuple[int, int] | None = None,
) -> V1LiteralJJointD1PopulationMvStreamCUDAHBMForecast:
    d1_tile = min(
        preregistration.route_count,
        d1_stream._cursor._tile,
    )
    residual_tile = min(
        preregistration.route_count,
        residual_stream.forecast.route_tile_size,
    )
    output_tile = min(d1_tile, residual_tile)
    components = _d1_population_mjv_forecast_components(
        direction_count=preregistration.direction_count,
        d1_parent_tile_size=d1_tile,
        residual_parent_tile_size=residual_tile,
        output_tile_size=output_tile,
    )
    incremental = sum(value for _name, value in components)
    total = _positive_int(device_total_memory_bytes, name="device_total_memory_bytes")
    free = _nonnegative_int(device_free_memory_bytes, name="device_free_memory_bytes")
    reserved = _nonnegative_int(
        current_reserved_device_bytes,
        name="current_reserved_device_bytes",
    )
    ceiling = math.floor(total * 0.85)
    projected = reserved + incremental
    d1_rows = d1_stream.receipt.context_count
    residual_rows = residual_stream.registration.registered_observed_counts
    if action_axis_source is None:
        pre_hash = "private-reference:no-action-axis-preoutcome"
        post_hash = "private-reference:no-action-axis-postsource"
    else:
        pre_hash = action_axis_source.preoutcome_source.receipt.receipt_hash
        post_hash = action_axis_source.receipt.receipt_hash
    available, allocated = _admission_memory or (free, reserved)
    return V1LiteralJJointD1PopulationMvStreamCUDAHBMForecast(
        action_axis_preoutcome_receipt_hash=pre_hash,
        action_axis_source_receipt_hash=post_hash,
        joint_preoutcome_registration_hash=preregistration.registration_hash,
        joint_family_receipt_hash=family_handle.receipt.receipt_hash,
        d1_stream_receipt_hash=d1_stream.receipt.receipt_hash,
        residual_stream_registration_hash=residual_stream.registration.registration_hash,
        measurement_device_identity=str(device),
        route_count=preregistration.route_count,
        action_count=preregistration.action_count,
        shell_size=preregistration.shell_size,
        direction_count=preregistration.direction_count,
        d1_context_count=d1_rows,
        residual_observed_counts=residual_rows,
        total_source_row_count=d1_rows + sum(residual_rows),
        d1_parent_route_tile_size=d1_tile,
        residual_parent_route_tile_size=residual_tile,
        maximum_output_route_tile_size=output_tile,
        source_resident_projection_bytes=(
            2 * (d1_rows + sum(residual_rows))
            * preregistration.direction_count
            * _F64_BYTES
        ),
        component_bytes=components,
        projected_incremental_peak_bytes=incremental,
        current_reserved_device_bytes=reserved,
        projected_reserved_device_bytes=projected,
        device_total_memory_bytes=total,
        device_free_memory_bytes=free,
        hbm_ceiling_fraction=0.85,
        hbm_ceiling_bytes=ceiling,
        admitted_by_ceiling=allocated + incremental <= ceiling,
        admitted_by_free_memory=incremental <= available,
        admitted=allocated + incremental <= ceiling and incremental <= available,
        production_cuda_execution=production,
        private_cpu_reference_execution=not production,
        _admission_memory=_admission_memory,
        _issuance_token=_FORECAST_TOKEN,
    )


class V1LiteralJJointD1PopulationMvStreamCUDAHandle:
    """Union-boundary zipper over exclusively claimed D1 and residual tiles."""

    __slots__ = (
        "forecast",
        "_preregistration",
        "_family_handle",
        "_action_axis_source",
        "_atomic_sources",
        "_atomic_hashes",
        "_atomic_pre_hashes",
        "_d1_stream",
        "_residual_stream",
        "_d1_claim_token",
        "_residual_claim_token",
        "_population_count",
        "_d1_mass",
        "_residual_masses",
        "_production",
        "_position",
        "_d1_parent",
        "_residual_parent",
        "_d1_parent_receipt_hashes",
        "_d1_parent_ranges",
        "_residual_parent_pair_count",
        "_residual_parent_ranges",
        "_active_tile_hash",
        "_tile_receipt_hashes",
        "_route_ranges",
        "_d1_source_consumed",
        "_residual_source_consumed",
        "_exhausted",
        "_final_receipt",
        "_evicted",
    )

    def __init__(
        self,
        *,
        forecast: V1LiteralJJointD1PopulationMvStreamCUDAHBMForecast,
        preregistration: V1LiteralJJointAffineShellPreoutcomeRegistration,
        family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
        action_axis_source: V1LiteralJJointPopulationActionAxisSource | None,
        atomic_sources: tuple[V1LiteralJJointPopulationSource, ...],
        atomic_hashes: tuple[str, ...],
        atomic_pre_hashes: tuple[str, ...],
        d1_stream: V1LiteralJJointAffineShellD1PopulationStreamDeviceHandle,
        residual_stream: V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle,
        d1_claim_token: object,
        residual_claim_token: object,
        population_count: int,
        d1_mass: Fraction,
        residual_masses: tuple[Fraction, Fraction],
        production: bool,
        issuance_token: object | None = None,
    ) -> None:
        if issuance_token is not _HANDLE_TOKEN:
            raise PermissionError("D1 population-Mv handles require their issuer")
        self.forecast = forecast
        self._preregistration = preregistration
        self._family_handle = family_handle
        self._action_axis_source = action_axis_source
        self._atomic_sources = atomic_sources
        self._atomic_hashes = atomic_hashes
        self._atomic_pre_hashes = atomic_pre_hashes
        self._d1_stream = d1_stream
        self._residual_stream = residual_stream
        self._d1_claim_token = d1_claim_token
        self._residual_claim_token = residual_claim_token
        self._population_count = population_count
        self._d1_mass = d1_mass
        self._residual_masses = residual_masses
        self._production = production
        self._position = 0
        self._d1_parent: tuple[object, Any, Any] | None = None
        self._residual_parent: tuple[object, object, Any, Any, Any, Any] | None = None
        self._d1_parent_receipt_hashes: list[str] = []
        self._d1_parent_ranges: list[tuple[int, int]] = []
        self._residual_parent_pair_count = 0
        self._residual_parent_ranges: list[tuple[int, int]] = []
        self._active_tile_hash: str | None = None
        self._tile_receipt_hashes: list[str] = []
        self._route_ranges: list[tuple[int, int]] = []
        self._d1_source_consumed = False
        self._residual_source_consumed = False
        self._exhausted = False
        self._final_receipt: V1LiteralJJointD1PopulationMvStreamCUDAExecutionReceipt | None = None
        self._evicted = False
        self.assert_live()

    @property
    def exhausted(self) -> bool:
        self.assert_live()
        return self._exhausted

    @property
    def active_tile(self) -> bool:
        self.assert_live()
        return self._active_tile_hash is not None

    def assert_live(self) -> None:
        if self._evicted:
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.EVICTED,
                "D1 population-Mv stream was evicted",
            )
        self.forecast.assert_integrity()
        try:
            self._preregistration.assert_integrity()
            self._family_handle.receipt.assert_integrity()
            self._d1_stream._assert_population_mjv_route_claim(self._d1_claim_token)
            self._residual_stream._assert_population_mjv_route_claim(
                self._residual_claim_token
            )
            if self._action_axis_source is not None:
                self._action_axis_source.assert_live()
            observed = tuple(source.assert_live() for source in self._atomic_sources)
        except V1LiteralJJointPopulationMvStreamCUDARefusal:
            raise
        except Exception as error:
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.MUTATED_CUSTODY,
                "one retained D1 population-Mv parent no longer replays: "
                f"{type(error).__name__}: {error}",
            )
        if observed != self._atomic_hashes:
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.STALE_IDENTITY,
                "atomic action sources changed after stream opening",
            )

    def _load_d1_parent(self) -> bool:
        if self._d1_parent is not None:
            return True
        tile = self._d1_stream._next_claimed_population_mjv_tile(
            self._d1_claim_token
        )
        if tile is None:
            return False
        if type(tile) is not V1LiteralJJointAffineShellD1PopulationStreamTileDeviceHandle:
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.SOURCE_MISMATCH,
                "D1 claimed-next returned an inexact tile",
            )
        receipt = tile.receipt
        with tile.borrow() as lease:
            center, radius = lease.intervals()
            center = center.clone().contiguous()
            radius = radius.clone().contiguous()
        tile.evict()
        self._d1_source_consumed = True
        self._d1_parent_receipt_hashes.append(receipt.receipt_hash)
        self._d1_parent_ranges.append((receipt.route_start, receipt.route_stop))
        self._d1_parent = (receipt, center, radius)
        return True

    def _load_residual_parent(self) -> bool:
        if self._residual_parent is not None:
            return True
        next_claimed = getattr(
            self._residual_stream,
            "_next_claimed_population_mjv_tile",
            None,
        )
        if not callable(next_claimed):
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.ACTION_COMPLETE_SOURCE_BINDING_REQUIRED,
                "residual stream lacks the source-owned claimed-next population-Mv protocol",
            )
        parent_tiles = []
        for expected_h in (0, 1):
            tile = next_claimed(self._residual_claim_token)
            if tile is None:
                if expected_h == 0:
                    return False
                _refuse(
                    V1LiteralJJointPopulationMvStreamCUDARefusalCode.ORDER_COVERAGE_MISMATCH,
                    "residual complement ended before its target tile",
                )
            if type(tile) is not _residual.V1LiteralJJointAffineShellResidualPopulationStreamTileDeviceHandle:
                _refuse(
                    V1LiteralJJointPopulationMvStreamCUDARefusalCode.SOURCE_MISMATCH,
                    "residual claimed-next returned an inexact tile",
                )
            self._residual_source_consumed = True
            receipt = tile.receipt
            if receipt.stratum_ordinal != expected_h:
                tile.evict()
                _refuse(
                    V1LiteralJJointPopulationMvStreamCUDARefusalCode.ORDER_COVERAGE_MISMATCH,
                    "residual claimed-next did not preserve target then complement order",
                )
            with tile.borrow() as lease:
                center, radius = lease.intervals()
                center = center.clone().contiguous()
                radius = radius.clone().contiguous()
            tile.evict()
            parent_tiles.append((receipt, center, radius))
        target, complement = parent_tiles
        if (
            target[0].route_start != complement[0].route_start
            or target[0].route_stop != complement[0].route_stop
            or target[0].joint_direction_axis_hash
            != complement[0].joint_direction_axis_hash
            or target[0].direction_ids != complement[0].direction_ids
        ):
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.ORDER_COVERAGE_MISMATCH,
                "residual target/complement claimed tiles do not form one route pair",
            )
        self._residual_source_consumed = True
        self._residual_parent_pair_count += 1
        self._residual_parent_ranges.append(
            (target[0].route_start, target[0].route_stop)
        )
        self._residual_parent = (
            target[0],
            complement[0],
            target[1],
            target[2],
            complement[1],
            complement[2],
        )
        return True

    def next_tile(self) -> V1LiteralJJointPopulationMvStreamTileDeviceHandle | None:
        self.assert_live()
        if self._active_tile_hash is not None:
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.ACTIVE_TILE,
                "evict the previous D1 population-Mv tile before advancement",
            )
        if self._exhausted:
            return None
        has_d1 = self._load_d1_parent()
        has_residual = self._load_residual_parent()
        if not has_d1 or not has_residual:
            if has_d1 != has_residual or self._position != self._preregistration.route_count:
                _refuse(
                    V1LiteralJJointPopulationMvStreamCUDARefusalCode.ORDER_COVERAGE_MISMATCH,
                    "D1 and residual claimed streams ended at different route positions",
                )
            self._exhausted = True
            return None
        assert self._d1_parent is not None
        assert self._residual_parent is not None
        d1_receipt, d1_center, d1_radius = self._d1_parent
        (
            target_receipt,
            complement_receipt,
            target_center,
            target_radius,
            complement_center,
            complement_radius,
        ) = self._residual_parent
        start = self._position
        if d1_receipt.route_start > start or target_receipt.route_start > start:
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.ORDER_COVERAGE_MISMATCH,
                "one parent route tile starts after the shared union cursor",
            )
        stop = min(d1_receipt.route_stop, target_receipt.route_stop)
        if stop <= start:
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.ORDER_COVERAGE_MISMATCH,
                "parent route tile intersection is empty",
            )
        d1_slice = slice(start - d1_receipt.route_start, stop - d1_receipt.route_start)
        residual_slice = slice(
            start - target_receipt.route_start,
            stop - target_receipt.route_start,
        )
        d1_lower, d1_upper = _bounds(
            d1_center[:, d1_slice],
            d1_radius[:, d1_slice],
        )
        target_lower, target_upper = _bounds(
            target_center[:, residual_slice],
            target_radius[:, residual_slice],
        )
        complement_lower, complement_upper = _bounds(
            complement_center[:, residual_slice],
            complement_radius[:, residual_slice],
        )
        lower, upper = _mix_population_bounds(
            (
                (d1_lower, d1_upper, self._d1_mass),
                (target_lower, target_upper, self._residual_masses[0]),
                (complement_lower, complement_upper, self._residual_masses[1]),
            )
        )
        center, radius = _center_radius(lower, upper)
        self._position = stop
        if stop == d1_receipt.route_stop:
            self._d1_parent = None
        if stop == target_receipt.route_stop:
            self._residual_parent = None
        ordinal = len(self._tile_receipt_hashes)
        action_source_hash = (
            "private-reference:no-action-axis-postsource"
            if self._action_axis_source is None
            else self._action_axis_source.receipt.receipt_hash
        )
        receipt = V1LiteralJJointD1PopulationMvStreamTileReceipt(
            forecast_hash=self.forecast.forecast_hash,
            action_axis_source_receipt_hash=action_source_hash,
            d1_stream_receipt_hash=self._d1_stream.receipt.receipt_hash,
            d1_parent_tile_receipt_hash=d1_receipt.receipt_hash,
            residual_stream_registration_hash=self._residual_stream.registration.registration_hash,
            residual_target_shard_receipt_hash=target_receipt.source_shard_receipt_hash,
            residual_complement_shard_receipt_hash=complement_receipt.source_shard_receipt_hash,
            residual_target_content_sha256=target_receipt.source_content_sha256,
            residual_complement_content_sha256=complement_receipt.source_content_sha256,
            joint_direction_axis_hash=self._preregistration.joint_direction_axis_hash,
            direction_ids=self._preregistration.joint_direction_ids,
            tile_ordinal=ordinal,
            route_start=start,
            route_stop=stop,
            route_count=self._preregistration.route_count,
            action_count=self._preregistration.action_count,
            shell_size=self._preregistration.shell_size,
            direction_count=self._preregistration.direction_count,
            d1_mass_fraction=(self._d1_mass.numerator, self._d1_mass.denominator),
            residual_mass_fractions=tuple(
                (value.numerator, value.denominator)
                for value in self._residual_masses
            ),
            _issuance_token=_TILE_TOKEN,
        )
        receipt_hash = receipt.receipt_hash
        self._active_tile_hash = receipt_hash
        self._tile_receipt_hashes.append(receipt_hash)
        self._route_ranges.append((start, stop))
        return V1LiteralJJointPopulationMvStreamTileDeviceHandle(
            receipt=receipt,  # type: ignore[arg-type]
            center=center,
            radius=radius,
            release=self._release_tile,
            issuance_token=_TILE_TOKEN,
        )

    def _release_tile(self, receipt_hash: str) -> None:
        if self._active_tile_hash != receipt_hash:
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.STALE_IDENTITY,
                "released D1 population-Mv tile is not active",
            )
        self._active_tile_hash = None

    def finalize(self) -> V1LiteralJJointD1PopulationMvStreamCUDAExecutionReceipt:
        self.assert_live()
        if self._active_tile_hash is not None:
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.ACTIVE_TILE,
                "cannot finalize with one live D1 population-Mv tile",
            )
        if not self._exhausted:
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.STREAM_NOT_EXHAUSTED,
                "call next_tile through terminal None before finalization",
            )
        if self._final_receipt is None:
            prereg = self._preregistration
            d1_receipt = self._d1_stream.receipt
            residual_registration = self._residual_stream.registration
            residual_compact = self._residual_stream.compact_shell_m_handle.receipt
            pre_hash = (
                "private-reference:no-action-axis-preoutcome"
                if self._action_axis_source is None
                else self._action_axis_source.preoutcome_source.receipt.receipt_hash
            )
            post_hash = (
                "private-reference:no-action-axis-postsource"
                if self._action_axis_source is None
                else self._action_axis_source.receipt.receipt_hash
            )
            d1_bytes = d1_receipt.context_count * prereg.route_count * _F64_BYTES
            residual_bytes = (
                sum(residual_registration.registered_observed_counts)
                * prereg.route_count
                * _F64_BYTES
            )
            self._final_receipt = V1LiteralJJointD1PopulationMvStreamCUDAExecutionReceipt(
                forecast_hash=self.forecast.forecast_hash,
                action_axis_preoutcome_receipt_hash=pre_hash,
                action_axis_source_receipt_hash=post_hash,
                joint_preoutcome_registration_hash=prereg.registration_hash,
                joint_family_receipt_hash=self._family_handle.receipt.receipt_hash,
                d1_stream_receipt_hash=d1_receipt.receipt_hash,
                d1_q_receipt_hash=self._d1_stream.q_handle.receipt.receipt_hash,
                d1_compact_shell_m_receipt_hash=self._d1_stream.compact_shell_m_handle.receipt.receipt_hash,
                residual_stream_registration_hash=residual_registration.registration_hash,
                residual_q_receipt_hash=self._residual_stream.q_handle.receipt.receipt_hash,
                residual_compact_shell_m_receipt_hash=residual_compact.receipt_hash,
                atomic_source_receipt_hashes=self._atomic_hashes,
                atomic_preoutcome_receipt_hashes=self._atomic_pre_hashes,
                d1_literal_j_shard_receipt_hashes=d1_receipt.literal_j_shard_receipt_hashes,
                d1_literal_j_content_sha256s=d1_receipt.literal_j_content_sha256s,
                residual_source_proof_closure_hash_axes=residual_compact.source_proof_closure_hash_axes,
                residual_source_shard_receipt_hash_axes=residual_compact.source_shard_receipt_hash_axes,
                residual_source_content_sha256_axes=residual_compact.source_content_sha256_axes,
                joint_direction_axis_hash=prereg.joint_direction_axis_hash,
                route_axis_hash=prereg.route_axis_hash,
                route_id_axis_hash=prereg.route_id_axis_hash,
                direction_ids=prereg.joint_direction_ids,
                route_count=prereg.route_count,
                action_count=prereg.action_count,
                shell_size=prereg.shell_size,
                direction_count=prereg.direction_count,
                d1_context_count=d1_receipt.context_count,
                residual_population_counts=residual_registration.residual_population_counts,
                residual_observed_counts=residual_registration.registered_observed_counts,
                population_count=self._population_count,
                d1_mass_fraction=(self._d1_mass.numerator, self._d1_mass.denominator),
                residual_mass_fractions=tuple(
                    (value.numerator, value.denominator)
                    for value in self._residual_masses
                ),
                d1_parent_tile_receipt_hashes=tuple(self._d1_parent_receipt_hashes),
                d1_parent_route_ranges=tuple(self._d1_parent_ranges),
                residual_parent_pair_count=self._residual_parent_pair_count,
                residual_parent_route_ranges=tuple(self._residual_parent_ranges),
                output_tile_receipt_hashes=tuple(self._tile_receipt_hashes),
                output_route_ranges=tuple(self._route_ranges),
                d1_pass2_source_payload_read_bytes=d1_bytes,
                residual_pass2_source_payload_read_bytes=residual_bytes,
                total_source_payload_read_bytes=d1_bytes + residual_bytes,
                host_to_device_payload_bytes=(d1_bytes + residual_bytes if self._production else 0),
                production_cuda_execution=self._production,
                private_cpu_reference_execution=not self._production,
                action_complete_source_binding_present=self._action_axis_source is not None,
                atomic_source_tuple_nonauthoritative=self._action_axis_source is None,
                _issuance_token=_EXECUTION_TOKEN,
            )
        return self._final_receipt

    def evict(self) -> None:
        replay_error: BaseException | None = None
        try:
            self.assert_live()
        except V1LiteralJJointPopulationMvStreamCUDARefusal as error:
            if error.code is V1LiteralJJointPopulationMvStreamCUDARefusalCode.EVICTED:
                raise
            replay_error = error
        if self._active_tile_hash is not None:
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.ACTIVE_TILE,
                "cannot evict D1 population-Mv with one live output tile",
            )
        partial_consumption = (
            (self._d1_source_consumed or self._residual_source_consumed)
            and not self._exhausted
        )
        release_errors: list[BaseException] = []
        for source, token, consumed, method in (
            (
                self._d1_stream,
                self._d1_claim_token,
                self._d1_source_consumed,
                "_abort_population_mjv_route_claim"
                if partial_consumption
                else "_release_population_mjv_route_claim",
            ),
            (
                self._residual_stream,
                self._residual_claim_token,
                self._residual_source_consumed,
                "_abort_population_mjv_route_claim"
                if partial_consumption
                else "_release_population_mjv_route_claim",
            ),
        ):
            try:
                getattr(source, method)(
                    token,
                    source_tiles_consumed=consumed,
                )
            except BaseException as error:
                release_errors.append(error)
        self._d1_parent = None
        self._residual_parent = None
        self._d1_claim_token = None
        self._residual_claim_token = None
        self._evicted = True
        if replay_error is not None:
            raise replay_error
        if release_errors:
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.MUTATED_CUSTODY,
                "one exclusive source claim failed during eviction: "
                + "; ".join(
                    f"{type(error).__name__}: {error}" for error in release_errors
                ),
            )
        if partial_consumption:
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.STREAM_NOT_EXHAUSTED,
                "partially consumed D1/residual source streams were burned "
                "during cleanup and cannot be retried",
            )

    def __reduce__(self) -> NoReturn:
        raise TypeError("D1 population-Mv handles are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("D1 population-Mv handles are nonserializable")


def _validate_d1_action_axis_source(
    *,
    source: V1LiteralJJointPopulationActionAxisSource,
    preregistration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    d1_stream: V1LiteralJJointAffineShellD1PopulationStreamDeviceHandle,
    residual_stream: V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle,
) -> tuple[V1LiteralJJointPopulationSource, ...]:
    if type(source) is not V1LiteralJJointPopulationActionAxisSource:
        _refuse(
            V1LiteralJJointPopulationMvStreamCUDARefusalCode.ACTION_COMPLETE_SOURCE_BINDING_REQUIRED,
            "production D1 population-Mv requires the exact POST action-axis aggregate",
        )
    try:
        source.assert_live()
    except Exception as error:
        _refuse(
            V1LiteralJJointPopulationMvStreamCUDARefusalCode.MUTATED_CUSTODY,
            "POST action-axis aggregate no longer replays: "
            f"{type(error).__name__}: {error}",
        )
    post = source.receipt
    pre = source.preoutcome_source.receipt
    if (
        post.preoutcome_action_axis_receipt_hash != pre.receipt_hash
        or post.joint_preoutcome_registration_hash != preregistration.registration_hash
        or pre.joint_preoutcome_registration_hash != preregistration.registration_hash
        or post.joint_direction_axis_hash != preregistration.joint_direction_axis_hash
        or pre.joint_direction_axis_hash != preregistration.joint_direction_axis_hash
        or pre.joint_direction_ids != preregistration.joint_direction_ids
        or post.physical_action_axis_hash != preregistration.physical_action_axis_hash
        or pre.physical_action_axis_hash != preregistration.physical_action_axis_hash
        or post.physical_action_identity_hashes
        != preregistration.physical_action_identity_hashes
        or post.physical_action_hashes != preregistration.physical_action_hashes
        or post.packed_action_hashes != preregistration.packed_action_hashes
        or pre.d1_context_count != d1_stream.receipt.context_count
        or pre.d1_stratum_counts != d1_stream.receipt.d1_stratum_counts
        or pre.residual_sample_stratum_counts
        != residual_stream.registration.registered_observed_counts
        or tuple(
            pre.population_stratum_counts[h] - pre.d1_stratum_counts[h]
            for h in (0, 1)
        )
        != residual_stream.registration.residual_population_counts
    ):
        _refuse(
            V1LiteralJJointPopulationMvStreamCUDARefusalCode.SOURCE_MISMATCH,
            "action-axis aggregate, D1, residual, and joint axes do not close",
        )
    return source._sources


def _open_v1_literal_j_joint_d1_population_mjv_stream(
    *,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    action_complete_population_source_binding: V1LiteralJJointPopulationActionAxisSource | None,
    atomic_population_sources: tuple[V1LiteralJJointPopulationSource, ...] | None,
    d1_stream: V1LiteralJJointAffineShellD1PopulationStreamDeviceHandle,
    residual_stream: V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle,
    production: bool,
    device_total_memory_bytes: int | None,
    device_free_memory_bytes: int | None,
    current_reserved_device_bytes: int | None,
) -> V1LiteralJJointD1PopulationMvStreamCUDAHandle:
    if torch is None:
        _refuse(
            V1LiteralJJointPopulationMvStreamCUDARefusalCode.INVALID_TYPE,
            "D1 population-Mv requires PyTorch",
        )
    if production:
        if atomic_population_sources is not None:
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.AUTHORITY_VIOLATION,
                "production cannot accept a caller-supplied duplicate atomic tuple",
            )
        if action_complete_population_source_binding is None:
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.ACTION_COMPLETE_SOURCE_BINDING_REQUIRED,
                "production requires one POST action-axis aggregate",
            )
        atomic_sources = _validate_d1_action_axis_source(
            source=action_complete_population_source_binding,
            preregistration=joint_preoutcome_registration,
            d1_stream=d1_stream,
            residual_stream=residual_stream,
        )
    else:
        if action_complete_population_source_binding is not None:
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.AUTHORITY_VIOLATION,
                "private atomic reference cannot impersonate aggregate authority",
            )
        if atomic_population_sources is None:
            raise TypeError("private reference requires exact atomic sources")
        atomic_sources = atomic_population_sources
    (
        atomic_hashes,
        atomic_pre_hashes,
        population_strata,
        d1_strata,
        residual_mass_pairs,
    ) = _validate_d1_atomic_population_sources_nonauthoritative(
        joint_preoutcome_registration=joint_preoutcome_registration,
        joint_family_handle=joint_family_handle,
        atomic_population_sources=atomic_sources,
        d1_stream=d1_stream,
        residual_stream=residual_stream,
    )
    if not callable(
        getattr(residual_stream, "_next_claimed_population_mjv_tile", None)
    ):
        _refuse(
            V1LiteralJJointPopulationMvStreamCUDARefusalCode.ACTION_COMPLETE_SOURCE_BINDING_REQUIRED,
            "residual stream has not landed its source-owned claimed-next Mv protocol",
        )
    # Device compatibility is established through the public compact-Q
    # custody surfaces.  The consumer never reaches into either producer's
    # route cursor or retained projection storage to discover execution state.
    with d1_stream.q_handle.borrow() as d1_q_lease:
        d1_q_center, _d1_q_radius = d1_q_lease.intervals()
        device = d1_q_center.device
    with residual_stream.q_handle.borrow() as residual_q_lease:
        residual_q_center, _residual_q_radius = residual_q_lease.intervals()
        residual_device = residual_q_center.device
    if residual_device != device:
        _refuse(
            V1LiteralJJointPopulationMvStreamCUDARefusalCode.SOURCE_MISMATCH,
            "D1 and residual compact-Q sources are not colocated",
        )
    if production:
        if device.type != "cuda":
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.DEVICE_REQUIRED,
                "production D1 population-Mv requires colocated CUDA parents",
            )
        if any(
            value is not None
            for value in (
                device_total_memory_bytes,
                device_free_memory_bytes,
                current_reserved_device_bytes,
            )
        ):
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.AUTHORITY_VIOLATION,
                "production HBM admission must use live device facts",
            )
        if not torch.are_deterministic_algorithms_enabled():
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.DETERMINISM_DISABLED,
                "production D1 population-Mv requires deterministic algorithms",
            )
        free, total = torch.cuda.mem_get_info(device)
        reserved = torch.cuda.memory_reserved(device)
    else:
        if device.type != "cpu" or joint_preoutcome_registration.route_count > _CPU_REFERENCE_ROUTE_LIMIT:
            _refuse(
                V1LiteralJJointPopulationMvStreamCUDARefusalCode.CPU_PRODUCTION_FALLBACK,
                "private D1 population-Mv is bounded to 4096 CPU routes",
            )
        total = 1_000_000_000 if device_total_memory_bytes is None else device_total_memory_bytes
        free = total if device_free_memory_bytes is None else device_free_memory_bytes
        reserved = 0 if current_reserved_device_bytes is None else current_reserved_device_bytes
    forecast = _build_d1_population_mjv_forecast(
        preregistration=joint_preoutcome_registration,
        family_handle=joint_family_handle,
        action_axis_source=action_complete_population_source_binding,
        d1_stream=d1_stream,
        residual_stream=residual_stream,
        device=device,
        production=production,
        device_total_memory_bytes=int(total),
        device_free_memory_bytes=int(free),
        current_reserved_device_bytes=int(reserved),
        _admission_memory=((cuda_reusable_memory_bytes(device, framework=torch, free_bytes=free), int(torch.cuda.memory_allocated(device))) if production else None),
    )
    if not forecast.admitted:
        _refuse(
            V1LiteralJJointPopulationMvStreamCUDARefusalCode.HBM_FORECAST_EXCEEDED,
            "D1/residual union-boundary zipper exceeds its exact 85% forecast",
        )
    d1_claim = d1_stream._claim_unopened_route_stream_for_population_mjv()
    residual_claim = None
    try:
        residual_claim = residual_stream._claim_unopened_route_stream_for_population_mjv()
        population_count = sum(population_strata)
        d1_mass = Fraction(sum(d1_strata), population_count)
        residual_masses = tuple(Fraction(*value) for value in residual_mass_pairs)
        return V1LiteralJJointD1PopulationMvStreamCUDAHandle(
            forecast=forecast,
            preregistration=joint_preoutcome_registration,
            family_handle=joint_family_handle,
            action_axis_source=action_complete_population_source_binding,
            atomic_sources=atomic_sources,
            atomic_hashes=atomic_hashes,
            atomic_pre_hashes=atomic_pre_hashes,
            d1_stream=d1_stream,
            residual_stream=residual_stream,
            d1_claim_token=d1_claim,
            residual_claim_token=residual_claim,
            population_count=population_count,
            d1_mass=d1_mass,
            residual_masses=(residual_masses[0], residual_masses[1]),
            production=production,
            issuance_token=_HANDLE_TOKEN,
        )
    except BaseException:
        if residual_claim is not None:
            residual_stream._release_population_mjv_route_claim(
                residual_claim,
                source_tiles_consumed=False,
            )
        d1_stream._release_population_mjv_route_claim(
            d1_claim,
            source_tiles_consumed=False,
        )
        raise


def open_v1_literal_j_joint_d1_population_mjv_stream_cuda(
    *,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    action_complete_population_source_binding: V1LiteralJJointPopulationActionAxisSource,
    d1_stream: V1LiteralJJointAffineShellD1PopulationStreamDeviceHandle,
    residual_stream: V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle,
) -> V1LiteralJJointD1PopulationMvStreamCUDAHandle:
    """Open the authoritative CUDA D1+residual population-Mv stream."""

    return _open_v1_literal_j_joint_d1_population_mjv_stream(
        joint_preoutcome_registration=joint_preoutcome_registration,
        joint_family_handle=joint_family_handle,
        action_complete_population_source_binding=action_complete_population_source_binding,
        atomic_population_sources=None,
        d1_stream=d1_stream,
        residual_stream=residual_stream,
        production=True,
        device_total_memory_bytes=None,
        device_free_memory_bytes=None,
        current_reserved_device_bytes=None,
    )


def _open_v1_literal_j_joint_d1_population_mjv_stream_cpu_reference(
    *,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    atomic_population_sources: tuple[V1LiteralJJointPopulationSource, ...],
    d1_stream: V1LiteralJJointAffineShellD1PopulationStreamDeviceHandle,
    residual_stream: V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle,
    device_total_memory_bytes: int = 1_000_000_000,
    device_free_memory_bytes: int = 1_000_000_000,
    current_reserved_device_bytes: int = 0,
) -> V1LiteralJJointD1PopulationMvStreamCUDAHandle:
    """Nonauthoritative small-domain reference over exact singular sources."""

    return _open_v1_literal_j_joint_d1_population_mjv_stream(
        joint_preoutcome_registration=joint_preoutcome_registration,
        joint_family_handle=joint_family_handle,
        action_complete_population_source_binding=None,
        atomic_population_sources=atomic_population_sources,
        d1_stream=d1_stream,
        residual_stream=residual_stream,
        production=False,
        device_total_memory_bytes=device_total_memory_bytes,
        device_free_memory_bytes=device_free_memory_bytes,
        current_reserved_device_bytes=current_reserved_device_bytes,
    )
