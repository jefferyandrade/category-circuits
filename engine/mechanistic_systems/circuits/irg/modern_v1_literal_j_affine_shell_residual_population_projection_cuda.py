"""Structured affine-shell residual literal-J population projection.

This sibling of the legacy residual CUDA producer keeps the dynamic direction
axis in its native affine-shell form

``(z_outside, e_b1, ..., e_bB)``, ``1 <= B <= 8``.

The implementation reuses the same two-pass literal-J projection arithmetic as
the existing residual owner, but never materializes a dense shell-basis bank and
never describes the direction family as a private-primal pair.  Production
authority stops at residual population geometry; KKT-face stability, C/K/O, mask
commit, terminal validation, runtime, and Pareto claims remain closed.
"""

from __future__ import annotations

from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

from dataclasses import InitVar, dataclass, field
from enum import Enum
from fractions import Fraction
import math
from types import MappingProxyType
from typing import Any, Final, Mapping, NoReturn, Sequence

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_literal_j_affine_shell_analytic_support_cuda import (
    V1LiteralJAffineShellAnalyticSupportDeviceHandle,
)
from .modern_v1_literal_j_affine_shell_direction_cuda import (
    V1LiteralJAffineShellDirectionDeviceHandle,
)
from .modern_v1_literal_j_d_fixed_residual_population_projection import (
    V1LiteralJDFixedResidualPopulationProjectionRegistration,
    V1LiteralJResidualPopulationProjectionResult,
    _AFFINE_SHELL_RESIDUAL_CUDA_PRODUCER_CAPABILITY,
    _issue_v1_literal_j_affine_shell_residual_population_projection_from_cuda_producer,
    _observed_prefix_hash as _canonical_observed_prefix_hash,
)
from .modern_v1_literal_j_residual_population_projection_cuda import (
    V1LiteralJResidualAnalyticSupportSourceCUDAResult,
    _add_up,
    _div_up,
    _gamma,
    _MIN_SUBNORMAL,
    _mul_up,
    _pass2_mjz,
    _quadratic_interval,
    _stage_source_tile,
    fingerprint_v1_literal_j_residual_cuda_tensor,
)
from .modern_v1_sampled_dual_chart_row_custody import (
    V1SampledDualChartRowGroupCommit,
    V1SampledDualChartRowGroupPacketProofClosure,
    V1SampledDualChartRowGroupPlan,
    V1SampledDualChartRowRole,
    V1SampledDualChartRowShardAttachment,
)
from .modern_v1_sampled_v7_affine_shell_preoutcome_fanout import (
    V1SampledV7AffineShellPreoutcomeFanoutReceipt,
)
from .modern_v1_sampled_variable_shell_dw2_suffix import (
    V1SampledVariableShellDW2SuffixPacket,
    V1SampledVariableShellDW2SuffixPrefixFreezeReceipt,
    V1SampledVariableShellDW2SuffixRegistration,
)

try:  # pragma: no cover - runtime image dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


LITERAL_J_AFFINE_SHELL_RESIDUAL_CUDA_DESIGN_SCHEMA: Final = (
    "partition_free_v1_literal_J_affine_shell_residual_cuda_design_v1"
)
LITERAL_J_AFFINE_SHELL_RESIDUAL_CUDA_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_literal_J_affine_shell_residual_cuda_registration_v1"
)
LITERAL_J_AFFINE_SHELL_RESIDUAL_CUDA_FORECAST_SCHEMA: Final = (
    "partition_free_v1_literal_J_affine_shell_residual_cuda_hbm_forecast_v1"
)
LITERAL_J_AFFINE_SHELL_RESIDUAL_CUDA_EXECUTION_SCHEMA: Final = (
    "partition_free_v1_literal_J_affine_shell_residual_cuda_execution_v1"
)
LITERAL_J_AFFINE_SHELL_RESIDUAL_CUDA_ALGORITHM: Final = (
    "structured_outside_plus_shell_basis_two_pass_literal_J_residual_projection_v1"
)
LITERAL_J_AFFINE_SHELL_RESIDUAL_CUDA_AUTHORITY: Final = (
    "affine_shell_residual_population_projection_numerics_only_no_private_primal_"
    "pair_authority_no_adaptive_stopping_rank_kkt_face_C_K_O_commit_terminal_or_"
    "pareto_authority_v1"
)

_PREFIX: Final = "partition-free-v1-literal-j-affine-shell-residual-cuda-"
_DESIGN_TOKEN: Final = object()
_FORECAST_TOKEN: Final = object()
_REGISTRATION_TOKEN: Final = object()
_EXECUTION_TOKEN: Final = object()
_F64_BYTES: Final = 8
_CPU_REFERENCE_ROUTE_LIMIT: Final = 4096
_FINGERPRINT_TILE_CELLS: Final = 1_048_576


class V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    INVALID_DESIGN = "invalid_design"
    STALE_IDENTITY = "stale_identity"
    ORDER_MISMATCH = "order_mismatch"
    PREFIX_MISMATCH = "prefix_mismatch"
    DEVICE_REQUIRED = "device_required"
    CPU_PRODUCTION_FALLBACK = "cpu_production_fallback"
    DETERMINISM_DISABLED = "determinism_disabled"
    SUPPORT_VIOLATION = "support_violation"
    NUMERICAL_REFUSAL = "numerical_refusal"
    HBM_FORECAST_EXCEEDED = "hbm_forecast_exceeded"
    JOINT_ACTION_AXIS_REQUIRED = "joint_action_axis_required"
    AUTHORITY_VIOLATION = "authority_violation"


class V1LiteralJAffineShellResidualPopulationProjectionCUDARefusal(RuntimeError):
    def __init__(
        self,
        code: V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode,
        message: str,
    ) -> None:
        if (
            type(code)
            is not V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode
        ):
            raise TypeError("affine-shell residual CUDA refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode,
    message: str,
) -> NoReturn:
    raise V1LiteralJAffineShellResidualPopulationProjectionCUDARefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty exact string",
        )
    return value


def _optional_identity(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _identity(value, name=name)


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _refuse(
            V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative exact integer",
        )
    return int(value)


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result <= 0:
        _refuse(
            V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _identity_axis(value: object, *, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _refuse(
            V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one ordered identity axis",
        )
    axis = tuple(_identity(item, name=f"{name} entry") for item in value)
    if not axis or len(axis) != len(set(axis)):
        _refuse(
            V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.INVALID_TYPE,
            f"{name} must be nonempty and unique",
        )
    return axis


def _identity_pair(value: Sequence[str], *, name: str) -> tuple[str, str]:
    if isinstance(value, (str, bytes)):
        _refuse(
            V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one two-entry identity axis",
        )
    axis = tuple(_identity(item, name=f"{name} entry") for item in value)
    if len(axis) != 2 or axis[0] == axis[1]:
        _refuse(
            V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.INVALID_TYPE,
            f"{name} must contain two distinct identities",
        )
    return axis[0], axis[1]


def _count_pair(value: object, *, name: str) -> tuple[int, int]:
    if type(value) is not tuple or len(value) != 2:
        _refuse(
            V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one immutable two-entry count axis",
        )
    return (
        _nonnegative_int(value[0], name=f"{name}[0]"),
        _nonnegative_int(value[1], name=f"{name}[1]"),
    )


def _finite_probability(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or not 0.0 < value < 1.0:
        _refuse(
            V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one finite float in (0, 1)",
        )
    return value


def _divide_probability_down(value: float, count: int) -> float:
    divisor = _positive_int(count, name="probability family divisor")
    exact = Fraction.from_float(value) / divisor
    result = float(exact)
    if Fraction.from_float(result) > exact:
        result = math.nextafter(result, 0.0)
    if not math.isfinite(result) or result <= 0.0:
        _refuse(
            V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.NUMERICAL_REFUSAL,
            "per-statement probability underflowed binary64",
        )
    return result


def _payload_hash(payload: Mapping[str, object], *, role: str) -> str:
    return content_hash(payload, prefix=_PREFIX + role + "-")


def _axis_hash(role: str, values: Sequence[str]) -> str:
    return _payload_hash(
        {"schema": "registered_ordered_axis_v1", "role": role, "values": tuple(values)},
        role="axis",
    )


def _require_fp64_tensor(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    device: Any | None = None,
    nonnegative: bool = False,
) -> Any:
    if torch is None or type(value) is not torch.Tensor:
        _refuse(
            V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one exact torch.Tensor",
        )
    if (
        value.dtype != torch.float64
        or tuple(int(item) for item in value.shape) != shape
        or not value.is_contiguous()
        or value.requires_grad
        or (device is not None and value.device != device)
    ):
        _refuse(
            V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.DEVICE_REQUIRED,
            f"{name} left its registered contiguous FP64 device/shape",
        )
    if not bool(torch.isfinite(value).all().item()):
        _refuse(
            V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.NUMERICAL_REFUSAL,
            f"{name} contains nonfinite values",
        )
    if nonnegative and bool((value < 0.0).any().item()):
        _refuse(
            V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.SUPPORT_VIOLATION,
            f"{name} contains a negative support bound",
        )
    return value


def fingerprint_v1_literal_j_affine_shell_residual_cuda_tensor(
    value: Any, *, role: str
) -> str:
    return fingerprint_v1_literal_j_residual_cuda_tensor(value, role=role)


@dataclass(frozen=True, slots=True)
class V1LiteralJAffineShellResidualPopulationProjectionCUDADesign:
    route_tile_size: int = 131_072
    hbm_ceiling_fraction: float = 0.85
    registered_external_workspace_bytes: int = 0
    deterministic_algorithms_required: bool = True
    production_cuda_only: bool = True
    route_tile_size_is_hardware_only: bool = True
    statistical_sample_count_controlled_by_tile_size: bool = False
    adaptive_stopping_authorized: bool = False
    schema: str = LITERAL_J_AFFINE_SHELL_RESIDUAL_CUDA_DESIGN_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _design_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _DESIGN_TOKEN:
            raise PermissionError("affine-shell residual CUDA designs require issuer")
        object.__setattr__(
            self,
            "route_tile_size",
            _positive_int(self.route_tile_size, name="route_tile_size"),
        )
        object.__setattr__(
            self,
            "registered_external_workspace_bytes",
            _nonnegative_int(
                self.registered_external_workspace_bytes,
                name="registered_external_workspace_bytes",
            ),
        )
        if (
            type(self.hbm_ceiling_fraction) is not float
            or not math.isfinite(self.hbm_ceiling_fraction)
            or self.hbm_ceiling_fraction != 0.85
            or any(
                type(value) is not bool
                for value in (
                    self.deterministic_algorithms_required,
                    self.production_cuda_only,
                    self.route_tile_size_is_hardware_only,
                    self.statistical_sample_count_controlled_by_tile_size,
                    self.adaptive_stopping_authorized,
                )
            )
            or self.deterministic_algorithms_required is not True
            or self.production_cuda_only is not True
            or self.route_tile_size_is_hardware_only is not True
            or self.statistical_sample_count_controlled_by_tile_size is not False
            or self.adaptive_stopping_authorized is not False
            or self.schema != LITERAL_J_AFFINE_SHELL_RESIDUAL_CUDA_DESIGN_SCHEMA
        ):
            _refuse(
                V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.AUTHORITY_VIOLATION,
                "affine-shell residual CUDA design crossed its hardware boundary",
            )
        self.assert_integrity()

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            }
        )

    @property
    def design_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = _payload_hash(self._payload(), role="design")
        if self._design_hash is not None and observed != self._design_hash:
            _refuse(
                V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
                "affine-shell residual design mutated",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
                "affine-shell residual design hash differs",
            )
        object.__setattr__(self, "_design_hash", observed)
        return observed


def register_v1_literal_j_affine_shell_residual_population_projection_cuda_design(
    *,
    route_tile_size: int = 131_072,
    registered_external_workspace_bytes: int = 0,
) -> V1LiteralJAffineShellResidualPopulationProjectionCUDADesign:
    return V1LiteralJAffineShellResidualPopulationProjectionCUDADesign(
        route_tile_size=route_tile_size,
        registered_external_workspace_bytes=registered_external_workspace_bytes,
        _issuance_token=_DESIGN_TOKEN,
    )


def _forecast_components(
    *,
    route_count: int,
    shell_size: int,
    direction_count: int,
    observed_counts: tuple[int, int],
    route_tile_size: int,
) -> tuple[tuple[str, int], ...]:
    routes = route_count
    directions = direction_count
    contexts = sum(observed_counts)
    maximum_contexts = max(observed_counts)
    tile = min(route_tile_size, routes)
    output_cells = 4 * directions * routes + 4 * directions
    return (
        ("affine_direction_custody_and_guard", 2 * (routes + shell_size) * _F64_BYTES),
        ("route_bound_source_custody_and_guard", 2 * routes * _F64_BYTES),
        ("compact_affine_support_custody_and_guard", 2 * directions * _F64_BYTES),
        ("projection_center_abs_radius", 3 * contexts * directions * _F64_BYTES),
        ("mjz_center_and_radius", 4 * directions * routes * _F64_BYTES),
        ("q_center_and_radius", 4 * directions * _F64_BYTES),
        ("source_tile_and_absolute", 2 * maximum_contexts * tile * _F64_BYTES),
        (
            "gemm_and_interval_temporaries",
            (
                6 * maximum_contexts * directions
                + 6 * directions * tile
                + maximum_contexts * tile
            )
            * _F64_BYTES,
        ),
        (
            "fingerprint_tile_temporaries",
            2 * min(_FINGERPRINT_TILE_CELLS, max(output_cells, routes)) * _F64_BYTES,
        ),
        ("producer_output_mutation_guards", output_cells * _F64_BYTES),
        ("promotion_result_clone_and_guard_peak", 2 * output_cells * _F64_BYTES),
    )


@dataclass(frozen=True, slots=True)
class V1LiteralJAffineShellResidualPopulationProjectionCUDAHBMForecast:
    design_hash: str
    measurement_device_identity: str
    affine_direction_source_hash: str
    affine_support_receipt_hash: str
    route_support_source_receipt_hash: str
    route_count: int
    shell_size: int
    direction_count: int
    observed_counts: tuple[int, int]
    route_tile_size: int
    component_bytes: tuple[tuple[str, int], ...]
    shared_resident_bytes: int
    producer_owned_peak_bytes: int
    promotion_handoff_peak_bytes: int
    registered_external_workspace_bytes: int
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
    includes_structured_direction_custody: bool = True
    includes_route_bound_source_custody: bool = True
    includes_compact_affine_support_custody: bool = True
    includes_source_tile_and_gemm_temporaries: bool = True
    includes_output_mutation_guards: bool = True
    includes_promotion_clone_and_guard_peak: bool = True
    context_by_route_hbm_bank_retained: bool = False
    dense_basis_bank_materialized: bool = False
    route_scale_device_to_host_transfer: bool = False
    private_primal_pair_axis_claimed: bool = False
    planning_only: bool = True
    production_execution_admission_authority: bool = False
    schema: str = LITERAL_J_AFFINE_SHELL_RESIDUAL_CUDA_FORECAST_SCHEMA
    _admission_memory: InitVar[tuple[int, int] | None] = None
    _issuance_token: InitVar[object | None] = None
    _forecast_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _admission_memory: tuple[int, int] | None, _issuance_token: object | None) -> None:
        available, allocated = _admission_memory or (self.device_free_memory_bytes, self.current_reserved_device_bytes)
        if _issuance_token is not _FORECAST_TOKEN:
            raise PermissionError("affine-shell residual forecasts require issuer")
        for name in (
            "design_hash",
            "measurement_device_identity",
            "affine_direction_source_hash",
            "affine_support_receipt_hash",
            "route_support_source_receipt_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        routes = _positive_int(self.route_count, name="route_count")
        shell = _positive_int(self.shell_size, name="shell_size")
        directions = _positive_int(self.direction_count, name="direction_count")
        counts = _count_pair(self.observed_counts, name="observed_counts")
        tile = _positive_int(self.route_tile_size, name="route_tile_size")
        components = tuple(self.component_bytes)
        expected = _forecast_components(
            route_count=routes,
            shell_size=shell,
            direction_count=directions,
            observed_counts=counts,
            route_tile_size=tile,
        )
        ledger = dict(expected)
        shared = (
            ledger["affine_direction_custody_and_guard"]
            + ledger["route_bound_source_custody_and_guard"]
            + ledger["compact_affine_support_custody_and_guard"]
        )
        promotion = ledger["promotion_result_clone_and_guard_peak"]
        producer = sum(ledger.values()) - shared - promotion
        external = _nonnegative_int(
            self.registered_external_workspace_bytes,
            name="registered_external_workspace_bytes",
        )
        incremental = producer + promotion + external
        reserved = _nonnegative_int(
            self.current_reserved_device_bytes,
            name="current_reserved_device_bytes",
        )
        total = _positive_int(self.device_total_memory_bytes, name="device_total_memory_bytes")
        free = _nonnegative_int(self.device_free_memory_bytes, name="device_free_memory_bytes")
        ceiling = math.floor(total * 0.85)
        projected = reserved + incremental
        flags = (
            self.admitted_by_ceiling,
            self.admitted_by_free_memory,
            self.admitted,
            self.includes_structured_direction_custody,
            self.includes_route_bound_source_custody,
            self.includes_compact_affine_support_custody,
            self.includes_source_tile_and_gemm_temporaries,
            self.includes_output_mutation_guards,
            self.includes_promotion_clone_and_guard_peak,
            self.context_by_route_hbm_bank_retained,
            self.dense_basis_bank_materialized,
            self.route_scale_device_to_host_transfer,
            self.private_primal_pair_axis_claimed,
            self.planning_only,
            self.production_execution_admission_authority,
        )
        if (
            components != expected
            or shell + 1 != directions
            or not 1 <= shell <= 8
            or self.shared_resident_bytes != shared
            or self.producer_owned_peak_bytes != producer
            or self.promotion_handoff_peak_bytes != promotion
            or self.projected_incremental_peak_bytes != incremental
            or self.projected_reserved_device_bytes != projected
            or self.hbm_ceiling_fraction != 0.85
            or self.hbm_ceiling_bytes != ceiling
            or self.admitted_by_ceiling != (allocated + incremental <= ceiling)
            or self.admitted_by_free_memory != (incremental <= available)
            or self.admitted != (allocated + incremental <= ceiling and incremental <= available)
            or any(type(value) is not bool for value in flags)
            or any(value is not True for value in flags[3:9])
            or any(value is not False for value in flags[9:13])
            or self.planning_only is not True
            or self.production_execution_admission_authority is not False
            or self.schema != LITERAL_J_AFFINE_SHELL_RESIDUAL_CUDA_FORECAST_SCHEMA
        ):
            _refuse(
                V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.INVALID_DESIGN,
                "affine-shell residual HBM forecast arithmetic does not replay",
            )
        self.assert_integrity()

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            }
        )

    @property
    def forecast_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = _payload_hash(self._payload(), role="hbm-forecast")
        if self._forecast_hash is not None and observed != self._forecast_hash:
            _refuse(
                V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
                "affine-shell residual HBM forecast mutated",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
                "affine-shell residual HBM forecast hash differs",
            )
        object.__setattr__(self, "_forecast_hash", observed)
        return observed


def _build_hbm_forecast_from_measurement(
    *,
    design: V1LiteralJAffineShellResidualPopulationProjectionCUDADesign,
    measurement_device_identity: str,
    affine_direction_source_hash: str,
    affine_support_receipt_hash: str,
    route_support_source_receipt_hash: str,
    route_count: int,
    shell_size: int,
    observed_counts: tuple[int, int],
    device_total_memory_bytes: int,
    device_free_memory_bytes: int,
    current_reserved_device_bytes: int,
    _admission_memory: tuple[int, int] | None = None,
) -> V1LiteralJAffineShellResidualPopulationProjectionCUDAHBMForecast:
    design.assert_integrity()
    directions = shell_size + 1
    components = _forecast_components(
        route_count=route_count,
        shell_size=shell_size,
        direction_count=directions,
        observed_counts=observed_counts,
        route_tile_size=design.route_tile_size,
    )
    ledger = dict(components)
    shared = (
        ledger["affine_direction_custody_and_guard"]
        + ledger["route_bound_source_custody_and_guard"]
        + ledger["compact_affine_support_custody_and_guard"]
    )
    promotion = ledger["promotion_result_clone_and_guard_peak"]
    producer = sum(ledger.values()) - shared - promotion
    incremental = (
        producer + promotion + design.registered_external_workspace_bytes
    )
    total = _positive_int(device_total_memory_bytes, name="device_total_memory_bytes")
    free = _nonnegative_int(device_free_memory_bytes, name="device_free_memory_bytes")
    reserved = _nonnegative_int(
        current_reserved_device_bytes, name="current_reserved_device_bytes"
    )
    ceiling = math.floor(total * 0.85)
    projected = reserved + incremental
    available, allocated = _admission_memory or (free, reserved)
    return V1LiteralJAffineShellResidualPopulationProjectionCUDAHBMForecast(
        design_hash=design.design_hash,
        measurement_device_identity=_identity(
            measurement_device_identity, name="measurement_device_identity"
        ),
        affine_direction_source_hash=_identity(
            affine_direction_source_hash, name="affine_direction_source_hash"
        ),
        affine_support_receipt_hash=_identity(
            affine_support_receipt_hash, name="affine_support_receipt_hash"
        ),
        route_support_source_receipt_hash=_identity(
            route_support_source_receipt_hash,
            name="route_support_source_receipt_hash",
        ),
        route_count=route_count,
        shell_size=shell_size,
        direction_count=directions,
        observed_counts=observed_counts,
        route_tile_size=design.route_tile_size,
        component_bytes=components,
        shared_resident_bytes=shared,
        producer_owned_peak_bytes=producer,
        promotion_handoff_peak_bytes=promotion,
        registered_external_workspace_bytes=(
            design.registered_external_workspace_bytes
        ),
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
        _admission_memory=_admission_memory,
        _issuance_token=_FORECAST_TOKEN,
    )


def _build_v1_literal_j_affine_shell_residual_population_projection_cuda_hbm_forecast_cpu_reference(
    *,
    design: V1LiteralJAffineShellResidualPopulationProjectionCUDADesign,
    route_count: int,
    shell_size: int,
    observed_counts: tuple[int, int],
    device_total_memory_bytes: int,
    device_free_memory_bytes: int,
    current_reserved_device_bytes: int,
) -> V1LiteralJAffineShellResidualPopulationProjectionCUDAHBMForecast:
    return _build_hbm_forecast_from_measurement(
        design=design,
        measurement_device_identity="private-cpu-affine-shell-residual-reference",
        affine_direction_source_hash="private-cpu-affine-direction-source",
        affine_support_receipt_hash="private-cpu-affine-support",
        route_support_source_receipt_hash="private-cpu-route-support-source",
        route_count=route_count,
        shell_size=shell_size,
        observed_counts=observed_counts,
        device_total_memory_bytes=device_total_memory_bytes,
        device_free_memory_bytes=device_free_memory_bytes,
        current_reserved_device_bytes=current_reserved_device_bytes,
    )


def build_v1_literal_j_affine_shell_residual_population_projection_cuda_hbm_forecast(
    *,
    design: V1LiteralJAffineShellResidualPopulationProjectionCUDADesign,
    affine_directions: V1LiteralJAffineShellDirectionDeviceHandle,
    affine_support: V1LiteralJAffineShellAnalyticSupportDeviceHandle,
    route_support_source: V1LiteralJResidualAnalyticSupportSourceCUDAResult,
    observed_counts: tuple[int, int],
) -> V1LiteralJAffineShellResidualPopulationProjectionCUDAHBMForecast:
    if (
        type(design)
        is not V1LiteralJAffineShellResidualPopulationProjectionCUDADesign
        or type(affine_directions) is not V1LiteralJAffineShellDirectionDeviceHandle
        or type(affine_support)
        is not V1LiteralJAffineShellAnalyticSupportDeviceHandle
        or type(route_support_source)
        is not V1LiteralJResidualAnalyticSupportSourceCUDAResult
    ):
        raise TypeError("affine-shell residual HBM forecast requires exact live custody")
    if torch is None or not torch.cuda.is_available():
        _refuse(
            V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.DEVICE_REQUIRED,
            "production affine-shell residual HBM forecast requires a real CUDA device",
        )
    design.assert_integrity()
    affine_support.receipt.assert_integrity()
    route_support_source.source_handle.assert_live()
    direction = affine_directions.receipt
    support = affine_support.receipt
    route_source = route_support_source.receipt
    if (
        support.affine_direction_receipt_hash != direction.receipt_hash
        or support.generic_support_source_receipt_hash != route_source.receipt_hash
        or support.direction_ids != direction.direction_ids
        or support.execution_mode != direction.execution_mode
    ):
        _refuse(
            V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
            "affine support does not bind the supplied direction and route-bound source",
        )
    if (
        design.deterministic_algorithms_required
        and not torch.are_deterministic_algorithms_enabled()
    ):
        _refuse(
            V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.DETERMINISM_DISABLED,
            "production forecast requires deterministic CUDA algorithms",
        )
    with affine_directions.borrow() as direction_lease:
        outside, _shell = direction_lease.structured_tensors()
        if outside.device.type != "cuda":
            _refuse(
                V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.DEVICE_REQUIRED,
                "production affine-shell residual custody must be CUDA-resident",
            )
        free, total = torch.cuda.mem_get_info(outside.device)
        reserved = torch.cuda.memory_reserved(outside.device)
    return _build_hbm_forecast_from_measurement(
        design=design,
        measurement_device_identity=str(outside.device),
        affine_direction_source_hash=direction.direction_source_hash,
        affine_support_receipt_hash=support.receipt_hash,
        route_support_source_receipt_hash=route_source.receipt_hash,
        route_count=direction.route_count,
        shell_size=len(direction.shell_indices),
        observed_counts=_count_pair(observed_counts, name="observed_counts"),
        device_total_memory_bytes=int(total),
        device_free_memory_bytes=int(free),
        current_reserved_device_bytes=int(reserved),
        _admission_memory=(cuda_reusable_memory_bytes(outside.device, framework=torch, free_bytes=free), int(torch.cuda.memory_allocated(outside.device))),
    )


@dataclass(frozen=True, slots=True)
class V1LiteralJAffineShellResidualPopulationProjectionCUDARegistration:
    population_projection_registration_hash: str
    suffix_registration_hash: str
    packet_receipt_hash: str
    prefix_freeze_receipt_hash: str
    row_group_plan_hash: str
    design_hash: str
    affine_direction_receipt_hash: str
    affine_direction_source_hash: str
    affine_direction_lineage_hash: str
    affine_direction_preoutcome_registration_hash: str
    preoutcome_fanout_receipt_hash: str
    preoutcome_physical_action_axis_hash: str
    preoutcome_binding_axis_hash: str
    preoutcome_destination_binding_hash: str
    affine_shell_shard_replay_receipt_hash: str
    affine_support_receipt_hash: str
    route_support_source_receipt_hash: str
    literal_j_abs_bound_source_hash: str
    compact_projection_support_value_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    joint_physical_action_count: int
    joint_shell_basis_count: int
    complete_preoutcome_action_axis_destination_count: int
    shell_size: int
    shell_indices: tuple[int, ...]
    shell_route_ids: tuple[str, ...]
    direction_ids: tuple[str, ...]
    d_fixed_projection_direction_axis_hash: str
    stratum_ids: tuple[str, str]
    residual_population_counts: tuple[int, int]
    registered_observed_counts: tuple[int, int]
    residual_sealed_population_order_hashes: tuple[str, str]
    packet_context_axis_hashes: tuple[str, str]
    packet_draw_axis_hashes: tuple[str, str]
    scorer_source_capture_identity_hashes: tuple[str, str]
    scorer_packet_plan_hashes: tuple[str, str]
    scorer_provider_plan_hashes: tuple[str, str]
    literal_j_stream_source_hash: str
    frozen_dw2_prefix_axis_hashes: tuple[str, str]
    frozen_dw2_pair_prefix_axis_hash: str
    familywise_alpha: float
    simultaneous_statement_count: int
    per_statement_two_sided_alpha: float
    fixed_prefix_registered_before_outcomes: bool = True
    affine_direction_preregistered_before_outcomes: bool = True
    exact_preoutcome_postsource_direction_custody: bool = True
    action_qualified_joint_statement_axis: bool = True
    multi_action_fanout_requires_joint_axis_owner: bool = True
    single_action_executor_only: bool = True
    no_private_primal_pair_axis: bool = True
    one_paired_packet_only: bool = True
    target_then_complement_scorer_order: bool = True
    adaptive_stopping_authorized: bool = False
    outcome_payload_present: bool = False
    decision_or_commit_authority: bool = False
    algorithm: str = LITERAL_J_AFFINE_SHELL_RESIDUAL_CUDA_ALGORITHM
    authority_scope: str = LITERAL_J_AFFINE_SHELL_RESIDUAL_CUDA_AUTHORITY
    schema: str = LITERAL_J_AFFINE_SHELL_RESIDUAL_CUDA_REGISTRATION_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _registration_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _REGISTRATION_TOKEN:
            raise PermissionError("affine-shell residual registrations require issuer")
        for name in (
            "population_projection_registration_hash",
            "suffix_registration_hash",
            "packet_receipt_hash",
            "prefix_freeze_receipt_hash",
            "row_group_plan_hash",
            "design_hash",
            "affine_direction_receipt_hash",
            "affine_direction_source_hash",
            "affine_direction_lineage_hash",
            "affine_direction_preoutcome_registration_hash",
            "preoutcome_fanout_receipt_hash",
            "preoutcome_physical_action_axis_hash",
            "preoutcome_binding_axis_hash",
            "preoutcome_destination_binding_hash",
            "affine_shell_shard_replay_receipt_hash",
            "affine_support_receipt_hash",
            "route_support_source_receipt_hash",
            "literal_j_abs_bound_source_hash",
            "compact_projection_support_value_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "d_fixed_projection_direction_axis_hash",
            "literal_j_stream_source_hash",
            "frozen_dw2_pair_prefix_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "direction_ids",
            "shell_route_ids",
            "residual_sealed_population_order_hashes",
            "packet_context_axis_hashes",
            "packet_draw_axis_hashes",
            "scorer_source_capture_identity_hashes",
            "scorer_packet_plan_hashes",
            "scorer_provider_plan_hashes",
            "frozen_dw2_prefix_axis_hashes",
        ):
            value = getattr(self, name)
            object.__setattr__(
                self,
                name,
                _identity_pair(value, name=name)
                if name.endswith("_hashes") and name != "frozen_dw2_prefix_axis_hashes"
                else _identity_axis(value, name=name),
            )
        object.__setattr__(
            self, "stratum_ids", _identity_pair(self.stratum_ids, name="stratum_ids")
        )
        routes = _positive_int(self.route_count, name="route_count")
        joint_actions = _positive_int(
            self.joint_physical_action_count,
            name="joint_physical_action_count",
        )
        joint_shell = _positive_int(
            self.joint_shell_basis_count,
            name="joint_shell_basis_count",
        )
        fanout_count = _positive_int(
            self.complete_preoutcome_action_axis_destination_count,
            name="complete_preoutcome_action_axis_destination_count",
        )
        shell = _positive_int(self.shell_size, name="shell_size")
        indices = tuple(
            _nonnegative_int(item, name="shell index") for item in self.shell_indices
        )
        directions = self.direction_ids
        if (
            joint_actions != 1
            or fanout_count != 1
            or joint_shell != shell
            or
            not 1 <= shell <= 8
            or shell != len(indices)
            or shell != len(self.shell_route_ids)
            or len(indices) != len(set(indices))
            or any(item >= routes for item in indices)
            or len(directions) != shell + 1
            or directions
            != ("literal-j:outside",)
            + tuple(f"literal-j:shell-basis:{item}" for item in self.shell_route_ids)
        ):
            _refuse(
                V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.INVALID_DESIGN,
                "affine-shell residual direction axis does not replay",
            )
        object.__setattr__(self, "shell_indices", indices)
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal_generation"),
        )
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "joint_physical_action_count", joint_actions)
        object.__setattr__(self, "joint_shell_basis_count", joint_shell)
        object.__setattr__(
            self,
            "complete_preoutcome_action_axis_destination_count",
            fanout_count,
        )
        object.__setattr__(self, "shell_size", shell)
        for name in ("residual_population_counts", "registered_observed_counts"):
            object.__setattr__(self, name, _count_pair(getattr(self, name), name=name))
        alpha = _finite_probability(self.familywise_alpha, name="familywise_alpha")
        statements = _positive_int(
            self.simultaneous_statement_count,
            name="simultaneous_statement_count",
        )
        per_statement = _finite_probability(
            self.per_statement_two_sided_alpha,
            name="per_statement_two_sided_alpha",
        )
        flags = (
            self.fixed_prefix_registered_before_outcomes,
            self.affine_direction_preregistered_before_outcomes,
            self.exact_preoutcome_postsource_direction_custody,
            self.action_qualified_joint_statement_axis,
            self.multi_action_fanout_requires_joint_axis_owner,
            self.single_action_executor_only,
            self.no_private_primal_pair_axis,
            self.one_paired_packet_only,
            self.target_then_complement_scorer_order,
            self.adaptive_stopping_authorized,
            self.outcome_payload_present,
            self.decision_or_commit_authority,
        )
        if (
            statements != 2 * (joint_actions + joint_shell) * (routes + 1)
            or per_statement != _divide_probability_down(alpha, statements)
            or any(
                observed <= 0 or observed > population
                for observed, population in zip(
                    self.registered_observed_counts,
                    self.residual_population_counts,
                    strict=True,
                )
            )
            or any(type(value) is not bool for value in flags)
            or any(value is not True for value in flags[:9])
            or any(value is not False for value in flags[9:])
            or self.algorithm != LITERAL_J_AFFINE_SHELL_RESIDUAL_CUDA_ALGORITHM
            or self.authority_scope != LITERAL_J_AFFINE_SHELL_RESIDUAL_CUDA_AUTHORITY
            or self.schema != LITERAL_J_AFFINE_SHELL_RESIDUAL_CUDA_REGISTRATION_SCHEMA
        ):
            _refuse(
                V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.AUTHORITY_VIOLATION,
                "affine-shell residual registration crossed its causal boundary",
            )
        object.__setattr__(self, "familywise_alpha", alpha)
        object.__setattr__(self, "simultaneous_statement_count", statements)
        object.__setattr__(self, "per_statement_two_sided_alpha", per_statement)
        self.assert_integrity()

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            }
        )

    @property
    def registration_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = _payload_hash(self._payload(), role="registration")
        if self._registration_hash is not None and observed != self._registration_hash:
            _refuse(
                V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
                "affine-shell residual registration mutated",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
                "affine-shell residual registration hash differs",
            )
        object.__setattr__(self, "_registration_hash", observed)
        return observed


def _register_v1_literal_j_affine_shell_residual_population_projection_cuda(
    *,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    suffix_registration: V1SampledVariableShellDW2SuffixRegistration,
    packet: V1SampledVariableShellDW2SuffixPacket,
    prefix_freeze: V1SampledVariableShellDW2SuffixPrefixFreezeReceipt,
    row_group_plan: V1SampledDualChartRowGroupPlan,
    affine_directions: V1LiteralJAffineShellDirectionDeviceHandle,
    affine_support: V1LiteralJAffineShellAnalyticSupportDeviceHandle,
    route_support_source: V1LiteralJResidualAnalyticSupportSourceCUDAResult,
    preoutcome_fanout_receipt: V1SampledV7AffineShellPreoutcomeFanoutReceipt,
    design: V1LiteralJAffineShellResidualPopulationProjectionCUDADesign,
    production_cuda_required: bool,
) -> V1LiteralJAffineShellResidualPopulationProjectionCUDARegistration:
    exact_types = (
        (
            population_registration,
            V1LiteralJDFixedResidualPopulationProjectionRegistration,
        ),
        (suffix_registration, V1SampledVariableShellDW2SuffixRegistration),
        (packet, V1SampledVariableShellDW2SuffixPacket),
        (prefix_freeze, V1SampledVariableShellDW2SuffixPrefixFreezeReceipt),
        (row_group_plan, V1SampledDualChartRowGroupPlan),
        (affine_directions, V1LiteralJAffineShellDirectionDeviceHandle),
        (affine_support, V1LiteralJAffineShellAnalyticSupportDeviceHandle),
        (route_support_source, V1LiteralJResidualAnalyticSupportSourceCUDAResult),
        (preoutcome_fanout_receipt, V1SampledV7AffineShellPreoutcomeFanoutReceipt),
        (design, V1LiteralJAffineShellResidualPopulationProjectionCUDADesign),
    )
    if any(type(value) is not expected for value, expected in exact_types):
        raise TypeError("affine-shell residual registration received an inexact parent")
    population_registration.assert_integrity()
    suffix_registration.assert_integrity()
    packet.assert_integrity()
    prefix_freeze.assert_integrity()
    row_group_plan.assert_integrity()
    affine_support.receipt.assert_integrity()
    route_support_source.receipt.assert_integrity()
    route_support_source.source_handle.assert_live()
    preoutcome_fanout_receipt.assert_integrity()
    design.assert_integrity()
    direction = affine_directions.receipt
    support = affine_support.receipt
    route_source = route_support_source.receipt
    observed = packet.after_suffix_stratum_counts
    scorer_packets = row_group_plan.packet_plans
    expected_contexts = (packet.target_context_ids, packet.complement_context_ids)
    if (
        population_registration.affine_shell_shard_replay_receipt_hash is None
        or population_registration.suffix_registration_hash
        != suffix_registration.receipt_hash
        or packet.suffix_registration_hash != suffix_registration.receipt_hash
        or prefix_freeze.suffix_registration_hash != suffix_registration.receipt_hash
        or prefix_freeze.completed_packet_hashes != (packet.receipt_hash,)
        or prefix_freeze.consumed_dw2_stratum_counts != observed
        or prefix_freeze.generation_index != population_registration.proposal_generation
        or prefix_freeze.generation_index != suffix_registration.generation_index
        or prefix_freeze.consumed_d1_stratum_counts
        != suffix_registration.consumed_d1_stratum_counts
        or prefix_freeze.population_stratum_counts
        != suffix_registration.population_stratum_counts
        or packet.packet_ordinal != 0
        or packet.before_suffix_stratum_counts != (0, 0)
        or observed
        != (len(packet.target_context_ids), len(packet.complement_context_ids))
        or any(value <= 0 for value in observed)
        or any(
            observed[h] > population_registration.residual_stratum_counts[h]
            for h in (0, 1)
        )
        or len(scorer_packets) != 2
        or row_group_plan.first_prefix_look_ordinal != 0
        or row_group_plan.expected_prior_manifest_hash is not None
        or tuple(item.packet_ordinal for item in scorer_packets) != (0, 1)
        or tuple(item.context_ids for item in scorer_packets) != expected_contexts
        or tuple(item.stratum_labels for item in scorer_packets)
        != (
            (population_registration.stratum_ids[0],) * observed[0],
            (population_registration.stratum_ids[1],) * observed[1],
        )
        or row_group_plan.chart_identity_hash
        != population_registration.chart_identity_hash
        or row_group_plan.current_mask_hash != population_registration.current_mask_hash
        or row_group_plan.proposal_generation
        != population_registration.proposal_generation
        or row_group_plan.route_axis_hash != population_registration.route_axis_hash
        or row_group_plan.route_id_axis_hash
        != population_registration.route_id_axis_hash
        or row_group_plan.route_count != population_registration.route_count
        or row_group_plan.literal_j_stream_identity_hash
        != population_registration.residual_literal_j_stream_source_hash
    ):
        _refuse(
            V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
            "paired suffix, scorer plan, and population chart diverged",
        )
    if (
        direction.exact_preoutcome_postsource_custody_bound is not True
        or direction.outside_direction_exact_action_custody is not True
        or direction.preoutcome_source_registration_hash is None
        or direction.preoutcome_fanout_receipt_hash is None
        or direction.preoutcome_destination_binding_hash is None
        or direction.preoutcome_binding_axis_hash is None
        or direction.preoutcome_physical_action_axis_hash is None
        or direction.postsource_receipt_hash is None
        or direction.projection_replay_registration_receipt_hash is None
        or direction.preoutcome_fanout_receipt_hash
        != preoutcome_fanout_receipt.receipt_hash
        or direction.preoutcome_binding_axis_hash
        != preoutcome_fanout_receipt.binding_axis_hash
        or direction.preoutcome_physical_action_axis_hash
        != preoutcome_fanout_receipt.physical_action_axis_hash
        or direction.route_axis_hash != population_registration.route_axis_hash
        or direction.route_id_axis_hash != population_registration.route_id_axis_hash
        or direction.route_count != population_registration.route_count
        or direction.current_mask_hash != population_registration.current_mask_hash
        or direction.proposal_generation != population_registration.proposal_generation
        or direction.direction_ids != population_registration.direction_ids
        or direction.direction_source_hash != population_registration.direction_source_hash
        or direction.direction_source_hash
        != population_registration.private_primal_direction_axis_hash
        or support.affine_direction_receipt_hash != direction.receipt_hash
        or support.affine_direction_source_hash != direction.direction_source_hash
        or support.generic_support_source_receipt_hash != route_source.receipt_hash
        or support.population_projection_registration_hash
        != population_registration.registration_hash
        or support.population_alpha_lineage_hash
        != population_registration.alpha_lineage_hash
        or support.direction_ids != direction.direction_ids
        or support.literal_j_stream_source_hash
        != population_registration.residual_literal_j_stream_source_hash
        or route_source.model_identity_hash != population_registration.model_identity_hash
        or route_source.chart_identity_hash != population_registration.chart_identity_hash
        or route_source.current_mask_hash != population_registration.current_mask_hash
        or route_source.proposal_generation
        != population_registration.proposal_generation
        or route_source.route_axis_hash != population_registration.route_axis_hash
        or route_source.route_id_axis_hash
        != population_registration.route_id_axis_hash
        or route_source.route_count != population_registration.route_count
        or route_source.literal_j_abs_bound_source_hash
        != support.literal_j_abs_bound_source_hash
        or support.execution_mode != direction.execution_mode
        or (
            production_cuda_required
            and (
                direction.execution_mode != "cuda_production"
                or support.execution_mode != "cuda_production"
            )
        )
        or (
            not production_cuda_required
            and direction.execution_mode != "cpu_reference_test_only"
        )
    ):
        _refuse(
            V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
            "affine direction, support source, and population registration do not close",
        )
    if preoutcome_fanout_receipt.destination_count != 1:
        _refuse(
            V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.JOINT_ACTION_AXIS_REQUIRED,
            "single-action affine residual projection cannot be replayed repeatedly across a multi-action fanout; register one joint (A outside directions + shared shell basis) axis first",
        )
    statements = 2 * len(direction.direction_ids) * (
        population_registration.route_count + 1
    )
    return V1LiteralJAffineShellResidualPopulationProjectionCUDARegistration(
        population_projection_registration_hash=population_registration.registration_hash,
        suffix_registration_hash=suffix_registration.receipt_hash,
        packet_receipt_hash=packet.receipt_hash,
        prefix_freeze_receipt_hash=prefix_freeze.receipt_hash,
        row_group_plan_hash=row_group_plan.group_plan_hash,
        design_hash=design.design_hash,
        affine_direction_receipt_hash=direction.receipt_hash,
        affine_direction_source_hash=direction.direction_source_hash,
        affine_direction_lineage_hash=direction.lineage_hash,
        affine_direction_preoutcome_registration_hash=(
            direction.preoutcome_source_registration_hash
        ),
        preoutcome_fanout_receipt_hash=preoutcome_fanout_receipt.receipt_hash,
        preoutcome_physical_action_axis_hash=(
            preoutcome_fanout_receipt.physical_action_axis_hash
        ),
        preoutcome_binding_axis_hash=preoutcome_fanout_receipt.binding_axis_hash,
        preoutcome_destination_binding_hash=(
            direction.preoutcome_destination_binding_hash
        ),
        affine_shell_shard_replay_receipt_hash=(
            population_registration.affine_shell_shard_replay_receipt_hash
        ),
        affine_support_receipt_hash=support.receipt_hash,
        route_support_source_receipt_hash=route_source.receipt_hash,
        literal_j_abs_bound_source_hash=support.literal_j_abs_bound_source_hash,
        compact_projection_support_value_hash=support.support_value_hash,
        model_identity_hash=population_registration.model_identity_hash,
        chart_identity_hash=population_registration.chart_identity_hash,
        current_mask_hash=population_registration.current_mask_hash,
        proposal_generation=population_registration.proposal_generation,
        route_axis_hash=population_registration.route_axis_hash,
        route_id_axis_hash=population_registration.route_id_axis_hash,
        route_count=population_registration.route_count,
        joint_physical_action_count=1,
        joint_shell_basis_count=len(direction.shell_indices),
        complete_preoutcome_action_axis_destination_count=(
            preoutcome_fanout_receipt.destination_count
        ),
        shell_size=len(direction.shell_indices),
        shell_indices=direction.shell_indices,
        shell_route_ids=direction.shell_route_ids,
        direction_ids=direction.direction_ids,
        d_fixed_projection_direction_axis_hash=(
            population_registration.d_fixed_projection_direction_axis_hash
        ),
        stratum_ids=population_registration.stratum_ids,
        residual_population_counts=population_registration.residual_stratum_counts,
        registered_observed_counts=observed,
        residual_sealed_population_order_hashes=(
            population_registration.residual_sealed_population_order_hashes
        ),
        packet_context_axis_hashes=(
            _axis_hash("target-contexts", scorer_packets[0].context_ids),
            _axis_hash("complement-contexts", scorer_packets[1].context_ids),
        ),
        packet_draw_axis_hashes=(
            _axis_hash("target-draws", scorer_packets[0].draw_ids),
            _axis_hash("complement-draws", scorer_packets[1].draw_ids),
        ),
        scorer_source_capture_identity_hashes=tuple(
            item.source_capture_identity_hash for item in scorer_packets
        ),  # type: ignore[arg-type]
        scorer_packet_plan_hashes=tuple(
            item.packet_plan_hash for item in scorer_packets
        ),  # type: ignore[arg-type]
        scorer_provider_plan_hashes=tuple(
            item.provider_plan_identity_hash for item in scorer_packets
        ),  # type: ignore[arg-type]
        literal_j_stream_source_hash=(
            population_registration.residual_literal_j_stream_source_hash
        ),
        frozen_dw2_prefix_axis_hashes=(
            prefix_freeze.dw2_target_prefix_axis_hash,
            prefix_freeze.dw2_complement_prefix_axis_hash,
        ),
        frozen_dw2_pair_prefix_axis_hash=prefix_freeze.dw2_pair_prefix_axis_hash,
        familywise_alpha=population_registration.familywise_alpha,
        simultaneous_statement_count=statements,
        per_statement_two_sided_alpha=_divide_probability_down(
            population_registration.familywise_alpha, statements
        ),
        _issuance_token=_REGISTRATION_TOKEN,
    )


def register_v1_literal_j_affine_shell_residual_population_projection_cuda(
    *,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    suffix_registration: V1SampledVariableShellDW2SuffixRegistration,
    packet: V1SampledVariableShellDW2SuffixPacket,
    prefix_freeze: V1SampledVariableShellDW2SuffixPrefixFreezeReceipt,
    row_group_plan: V1SampledDualChartRowGroupPlan,
    affine_directions: V1LiteralJAffineShellDirectionDeviceHandle,
    affine_support: V1LiteralJAffineShellAnalyticSupportDeviceHandle,
    route_support_source: V1LiteralJResidualAnalyticSupportSourceCUDAResult,
    preoutcome_fanout_receipt: V1SampledV7AffineShellPreoutcomeFanoutReceipt,
    design: V1LiteralJAffineShellResidualPopulationProjectionCUDADesign,
) -> V1LiteralJAffineShellResidualPopulationProjectionCUDARegistration:
    return _register_v1_literal_j_affine_shell_residual_population_projection_cuda(
        population_registration=population_registration,
        suffix_registration=suffix_registration,
        packet=packet,
        prefix_freeze=prefix_freeze,
        row_group_plan=row_group_plan,
        affine_directions=affine_directions,
        affine_support=affine_support,
        route_support_source=route_support_source,
        preoutcome_fanout_receipt=preoutcome_fanout_receipt,
        design=design,
        production_cuda_required=True,
    )


def _register_v1_literal_j_affine_shell_residual_population_projection_cpu_reference(
    *,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    suffix_registration: V1SampledVariableShellDW2SuffixRegistration,
    packet: V1SampledVariableShellDW2SuffixPacket,
    prefix_freeze: V1SampledVariableShellDW2SuffixPrefixFreezeReceipt,
    row_group_plan: V1SampledDualChartRowGroupPlan,
    affine_directions: V1LiteralJAffineShellDirectionDeviceHandle,
    affine_support: V1LiteralJAffineShellAnalyticSupportDeviceHandle,
    route_support_source: V1LiteralJResidualAnalyticSupportSourceCUDAResult,
    preoutcome_fanout_receipt: V1SampledV7AffineShellPreoutcomeFanoutReceipt,
    design: V1LiteralJAffineShellResidualPopulationProjectionCUDADesign,
) -> V1LiteralJAffineShellResidualPopulationProjectionCUDARegistration:
    return _register_v1_literal_j_affine_shell_residual_population_projection_cuda(
        population_registration=population_registration,
        suffix_registration=suffix_registration,
        packet=packet,
        prefix_freeze=prefix_freeze,
        row_group_plan=row_group_plan,
        affine_directions=affine_directions,
        affine_support=affine_support,
        route_support_source=route_support_source,
        preoutcome_fanout_receipt=preoutcome_fanout_receipt,
        design=design,
        production_cuda_required=False,
    )


def _validate_execution_source_closure(
    *,
    cuda_registration: V1LiteralJAffineShellResidualPopulationProjectionCUDARegistration,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    suffix_registration: V1SampledVariableShellDW2SuffixRegistration,
    packet: V1SampledVariableShellDW2SuffixPacket,
    prefix_freeze: V1SampledVariableShellDW2SuffixPrefixFreezeReceipt,
    row_group_plan: V1SampledDualChartRowGroupPlan,
    row_group_commit: V1SampledDualChartRowGroupCommit,
    packet_proof_closures: Sequence[V1SampledDualChartRowGroupPacketProofClosure],
    design: V1LiteralJAffineShellResidualPopulationProjectionCUDADesign,
) -> tuple[
    tuple[V1SampledDualChartRowShardAttachment, V1SampledDualChartRowShardAttachment],
    tuple[V1SampledDualChartRowGroupPacketProofClosure, V1SampledDualChartRowGroupPacketProofClosure],
    tuple[str, str],
    tuple[str, str],
    tuple[str, str],
]:
    exact_types = (
        (
            cuda_registration,
            V1LiteralJAffineShellResidualPopulationProjectionCUDARegistration,
        ),
        (
            population_registration,
            V1LiteralJDFixedResidualPopulationProjectionRegistration,
        ),
        (suffix_registration, V1SampledVariableShellDW2SuffixRegistration),
        (packet, V1SampledVariableShellDW2SuffixPacket),
        (prefix_freeze, V1SampledVariableShellDW2SuffixPrefixFreezeReceipt),
        (row_group_plan, V1SampledDualChartRowGroupPlan),
        (row_group_commit, V1SampledDualChartRowGroupCommit),
        (design, V1LiteralJAffineShellResidualPopulationProjectionCUDADesign),
    )
    if any(type(value) is not expected for value, expected in exact_types):
        raise TypeError("affine-shell residual execution received an inexact owner")
    closures = tuple(packet_proof_closures)
    if len(closures) != 2 or any(
        type(item) is not V1SampledDualChartRowGroupPacketProofClosure
        for item in closures
    ):
        raise TypeError("affine-shell residual execution requires two proof closures")
    cuda_registration.assert_integrity()
    population_registration.assert_integrity()
    suffix_registration.assert_integrity()
    packet.assert_integrity()
    prefix_freeze.assert_integrity()
    row_group_plan.assert_integrity()
    row_group_commit.receipt.assert_integrity()
    design.assert_integrity()
    for closure in closures:
        closure.assert_integrity()
    try:
        row_group_commit.attachment.verify_live_witnesses()
    except Exception as error:
        raise V1LiteralJAffineShellResidualPopulationProjectionCUDARefusal(
            V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
            "published residual row-group shard custody is stale or mutated",
        ) from error
    commit_receipt = row_group_commit.receipt
    manifest = row_group_commit.attachment.manifest
    packet_plans = row_group_plan.packet_plans
    attachments = tuple(row_group_commit.attachment.literal_j_attachments)
    if len(packet_plans) != 2 or len(manifest.looks) != 2 or len(attachments) != 2:
        _refuse(
            V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.ORDER_MISMATCH,
            "affine-shell residual execution requires target then complement packets",
        )
    if (
        cuda_registration.population_projection_registration_hash
        != population_registration.registration_hash
        or cuda_registration.suffix_registration_hash
        != suffix_registration.receipt_hash
        or cuda_registration.packet_receipt_hash != packet.receipt_hash
        or cuda_registration.prefix_freeze_receipt_hash != prefix_freeze.receipt_hash
        or cuda_registration.row_group_plan_hash != row_group_plan.group_plan_hash
        or cuda_registration.design_hash != design.design_hash
        or commit_receipt.group_plan_hash != row_group_plan.group_plan_hash
        or commit_receipt.packet_plan_hashes
        != tuple(item.packet_plan_hash for item in packet_plans)
        or commit_receipt.packet_count != 2
        or commit_receipt.role_shard_count != 4
        or commit_receipt.proof_closure_hashes
        != tuple(item.closure_hash for item in closures)
        or commit_receipt.provider_proof_hashes
        != tuple(item.provider_proof_hash for item in closures)
        or tuple(item.packet_ordinal for item in closures) != (0, 1)
        or tuple(item.packet_plan_hash for item in closures)
        != cuda_registration.scorer_packet_plan_hashes
        or tuple(item.provider_plan_identity_hash for item in closures)
        != cuda_registration.scorer_provider_plan_hashes
        or manifest.manifest_hash != commit_receipt.manifest_hash
        or manifest.chart_identity_hash != cuda_registration.chart_identity_hash
        or manifest.current_mask_hash != cuda_registration.current_mask_hash
        or manifest.proposal_generation != cuda_registration.proposal_generation
        or manifest.route_axis_hash != cuda_registration.route_axis_hash
        or manifest.route_id_axis_hash != cuda_registration.route_id_axis_hash
        or manifest.route_count != cuda_registration.route_count
        or manifest.literal_j_stream_identity_hash
        != cuda_registration.literal_j_stream_source_hash
        or prefix_freeze.completed_packet_hashes != (packet.receipt_hash,)
        or prefix_freeze.consumed_dw2_stratum_counts
        != cuda_registration.registered_observed_counts
    ):
        _refuse(
            V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
            "affine-shell residual preregistration, scorer proofs, or frozen prefix diverged",
        )
    measurement_hashes: list[str] = []
    pass_hashes: list[str] = []
    same_capture_hashes: list[str] = []
    for index, (plan, closure, look, attachment) in enumerate(
        zip(packet_plans, closures, manifest.looks, attachments, strict=True)
    ):
        shard = attachment.receipt
        expected_stratum = cuda_registration.stratum_ids[index]
        context_role = "target-contexts" if index == 0 else "complement-contexts"
        draw_role = "target-draws" if index == 0 else "complement-draws"
        if (
            type(attachment) is not V1SampledDualChartRowShardAttachment
            or closure.group_plan_hash != row_group_plan.group_plan_hash
            or closure.packet_plan_hash != plan.packet_plan_hash
            or closure.packet_ordinal != index
            or closure.provider_plan_identity_hash
            != plan.provider_plan_identity_hash
            or look.literal_j_shard.receipt_hash != shard.receipt_hash
            or shard.role is not V1SampledDualChartRowRole.LITERAL_RELAXED_GATE_JACOBIAN
            or shard.context_ids != plan.context_ids
            or shard.draw_ids != plan.draw_ids
            or shard.stratum_labels != plan.stratum_labels
            or plan.stratum_labels != (expected_stratum,) * len(plan.context_ids)
            or shard.source_capture_identity_hash
            != plan.source_capture_identity_hash
            or look.corrected_a_shard.source_capture_identity_hash
            != shard.source_capture_identity_hash
            or shard.stream_identity_hash
            != cuda_registration.literal_j_stream_source_hash
            or shard.chart_identity_hash != cuda_registration.chart_identity_hash
            or shard.current_mask_hash != cuda_registration.current_mask_hash
            or shard.proposal_generation != cuda_registration.proposal_generation
            or shard.route_axis_hash != cuda_registration.route_axis_hash
            or shard.route_id_axis_hash != cuda_registration.route_id_axis_hash
            or shard.route_count != cuda_registration.route_count
            or _axis_hash(context_role, shard.context_ids)
            != cuda_registration.packet_context_axis_hashes[index]
            or _axis_hash(draw_role, shard.draw_ids)
            != cuda_registration.packet_draw_axis_hashes[index]
        ):
            _refuse(
                V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
                "literal-J shard or provider closure left its scorer packet",
            )
        measurement_hashes.append(
            _payload_hash(
                {
                    "schema": "affine_shell_residual_literal_j_source_measurement_axis_v1",
                    "cuda_registration_hash": cuda_registration.registration_hash,
                    "stratum_id": expected_stratum,
                    "literal_j_shard_receipt_hash": shard.receipt_hash,
                    "literal_j_content_sha256": shard.content_sha256,
                    "context_ids": shard.context_ids,
                    "draw_ids": shard.draw_ids,
                    "source_capture_identity_hash": shard.source_capture_identity_hash,
                    "provider_proof_closure_hash": closure.closure_hash,
                },
                role="source-measurement-axis",
            )
        )
        pass_hashes.append(
            _payload_hash(
                {
                    "schema": "affine_shell_immutable_literal_j_two_pass_axis_v1",
                    "literal_j_shard_receipt_hash": shard.receipt_hash,
                    "content_sha256": shard.content_sha256,
                    "registered_route_windows": shard.route_windows,
                    "route_tile_size": design.route_tile_size,
                    "passes": ("p_equals_Jz_outside_and_shell_gather", "mean_pJ"),
                    "q_reduced_from_resident_p": True,
                    "dense_basis_bank_materialized": False,
                },
                role="source-pass-axis",
            )
        )
        same_capture_hashes.append(
            _payload_hash(
                {
                    "schema": "affine_shell_residual_same_capture_vjp_source_v1",
                    "source_capture_identity_hash": shard.source_capture_identity_hash,
                    "provider_plan_identity_hash": plan.provider_plan_identity_hash,
                    "provider_proof_hash": closure.provider_proof_hash,
                    "provider_proof_closure_hash": closure.closure_hash,
                    "corrected_a_shard_receipt_hash": (
                        look.corrected_a_shard.receipt_hash
                    ),
                    "literal_j_shard_receipt_hash": shard.receipt_hash,
                    "both_roles_share_source_capture": (
                        look.corrected_a_shard.source_capture_identity_hash
                        == shard.source_capture_identity_hash
                    ),
                },
                role="same-capture-vjp-source",
            )
        )
    return (
        attachments,  # type: ignore[return-value]
        closures,  # type: ignore[return-value]
        tuple(measurement_hashes),  # type: ignore[return-value]
        tuple(pass_hashes),  # type: ignore[return-value]
        tuple(same_capture_hashes),  # type: ignore[return-value]
    )


def _pass1_affine_projections(
    *,
    attachment: V1SampledDualChartRowShardAttachment,
    outside: Any,
    shell_indices: tuple[int, ...],
    literal_j_abs_bound: Any,
    design: V1LiteralJAffineShellResidualPopulationProjectionCUDADesign,
) -> tuple[Any, Any, int, int, int]:
    context_count = len(attachment.receipt.context_ids)
    route_count = int(outside.numel())
    direction_count = 1 + len(shell_indices)
    center = torch.zeros(
        (context_count, direction_count), device=outside.device, dtype=torch.float64
    )
    absolute_mass = torch.zeros_like(center)
    tile_count = 0
    window_count = 0
    payload_bytes = 0
    for parent_start, parent_stop, parent_rows in attachment.iter_route_windows(
        verify=True
    ):
        window_count += 1
        width = parent_stop - parent_start
        for local_start in range(0, width, design.route_tile_size):
            local_stop = min(width, local_start + design.route_tile_size)
            start = parent_start + local_start
            stop = parent_start + local_stop
            rows = _stage_source_tile(
                parent_rows[:, local_start:local_stop], device=outside.device
            )
            route_bound = literal_j_abs_bound[start:stop]
            if bool((torch.abs(rows) > route_bound.reshape(1, -1)).any().item()):
                _refuse(
                    V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.SUPPORT_VIOLATION,
                    "literal-J source tile exceeded its analytic route support",
                )
            z_tile = outside[start:stop]
            center[:, 0].add_(rows @ z_tile)
            absolute_mass[:, 0].add_(torch.abs(rows) @ torch.abs(z_tile))
            rows_abs = torch.abs(rows)
            for ordinal, physical in enumerate(shell_indices, start=1):
                if start <= physical < stop:
                    column = physical - start
                    values = rows[:, column]
                    center[:, ordinal].add_(values)
                    absolute_mass[:, ordinal].add_(rows_abs[:, column])
            payload_bytes += int(rows.numel()) * _F64_BYTES
            tile_count += 1
    operations = 2 * route_count + 2 * tile_count + 1
    relative = _gamma(operations)
    recovery = _div_up(relative, 1.0 - relative)
    radius = _add_up(
        _mul_up(absolute_mass, recovery),
        _mul_up(float(operations), _MIN_SUBNORMAL),
    ).contiguous()
    return center.contiguous(), radius, window_count, tile_count, payload_bytes


@dataclass(frozen=True, slots=True)
class _V1LiteralJAffineShellResidualTwoPassNumerics:
    mjz_center: Any = field(repr=False, compare=False)
    mjz_radius: Any = field(repr=False, compare=False)
    q_center: Any = field(repr=False, compare=False)
    q_radius: Any = field(repr=False, compare=False)
    source_route_window_read_count: int
    source_payload_read_bytes: int
    source_tile_submission_count: int


def _compute_v1_literal_j_affine_shell_residual_two_pass_device_numerics(
    *,
    attachments: Sequence[V1SampledDualChartRowShardAttachment],
    affine_directions: V1LiteralJAffineShellDirectionDeviceHandle,
    literal_j_abs_bound: Any,
    projection_abs_bounds: Any,
    residual_population_counts: tuple[int, int],
    familywise_alpha: float,
    design: V1LiteralJAffineShellResidualPopulationProjectionCUDADesign,
    production_cuda_required: bool,
    failure_injection: str | None = None,
) -> _V1LiteralJAffineShellResidualTwoPassNumerics:
    if torch is None:
        raise RuntimeError("affine-shell residual numerics require PyTorch")
    if type(design) is not V1LiteralJAffineShellResidualPopulationProjectionCUDADesign:
        raise TypeError("affine-shell residual numerics require the exact design")
    shards = tuple(attachments)
    if len(shards) != 2 or any(
        type(item) is not V1SampledDualChartRowShardAttachment for item in shards
    ):
        raise TypeError("affine-shell residual numerics require two shard attachments")
    if failure_injection not in (None, "after_target_pass1", "after_target_pass2"):
        raise ValueError("unknown affine-shell residual failure injection")
    direction = affine_directions.receipt
    route_count = direction.route_count
    direction_count = len(direction.direction_ids)
    if not production_cuda_required and route_count > _CPU_REFERENCE_ROUTE_LIMIT:
        _refuse(
            V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.CPU_PRODUCTION_FALLBACK,
            "private CPU affine-shell residual oracle is limited to 4096 routes",
        )
    population = _count_pair(
        residual_population_counts, name="residual_population_counts"
    )
    statements = 2 * direction_count * (route_count + 1)
    per_statement = _divide_probability_down(
        _finite_probability(familywise_alpha, name="familywise_alpha"), statements
    )
    with affine_directions.borrow() as direction_lease:
        outside, shell_tensor = direction_lease.structured_tensors()
        if production_cuda_required and outside.device.type != "cuda":
            _refuse(
                V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.CPU_PRODUCTION_FALLBACK,
                "production affine-shell residual numerics require CUDA",
            )
        if not production_cuda_required and outside.device.type != "cpu":
            _refuse(
                V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.CPU_PRODUCTION_FALLBACK,
                "private affine-shell residual reference requires CPU tensors",
            )
        shell_indices = tuple(int(item) for item in shell_tensor.detach().cpu().tolist())
        j_bound = _require_fp64_tensor(
            literal_j_abs_bound,
            name="literal_j_abs_bound",
            shape=(route_count,),
            device=outside.device,
            nonnegative=True,
        )
        p_bound = _require_fp64_tensor(
            projection_abs_bounds,
            name="affine_projection_abs_bounds",
            shape=(direction_count,),
            device=outside.device,
            nonnegative=True,
        )
        observed = tuple(len(item.receipt.context_ids) for item in shards)
        if any(observed[h] <= 0 or observed[h] > population[h] for h in (0, 1)):
            _refuse(
                V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.INVALID_DESIGN,
                "literal-J shard counts are not nonempty residual prefixes",
            )
        all_center_m = torch.empty(
            (2, direction_count, route_count),
            device=outside.device,
            dtype=torch.float64,
        )
        all_radius_m = torch.empty_like(all_center_m)
        all_center_q = torch.empty(
            (2, direction_count), device=outside.device, dtype=torch.float64
        )
        all_radius_q = torch.empty_like(all_center_q)
        window_reads = tile_submissions = payload_bytes = 0
        for h, attachment in enumerate(shards):
            projection, projection_radius, windows, tiles, payload = (
                _pass1_affine_projections(
                    attachment=attachment,
                    outside=outside,
                    shell_indices=shell_indices,
                    literal_j_abs_bound=j_bound,
                    design=design,
                )
            )
            window_reads += windows
            tile_submissions += tiles
            payload_bytes += payload
            projection_upper = _add_up(torch.abs(projection), projection_radius)
            if bool((projection_upper > p_bound.reshape(1, -1)).any().item()):
                _refuse(
                    V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.SUPPORT_VIOLATION,
                    "affine-shell projection interval exceeded its pre-outcome support",
                )
            if h == 0 and failure_injection == "after_target_pass1":
                raise RuntimeError("injected failure after target affine pass 1")
            q_c, q_r = _quadratic_interval(
                projection=projection,
                projection_radius=projection_radius,
                projection_bound=p_bound,
                population_count=population[h],
                alpha=per_statement,
            )
            m_c, m_r, windows, tiles, payload = _pass2_mjz(
                attachment=attachment,
                projection=projection,
                projection_radius=projection_radius,
                literal_bound=j_bound,
                projection_bound=p_bound,
                population_count=population[h],
                alpha=per_statement,
                design=design,
            )
            window_reads += windows
            tile_submissions += tiles
            payload_bytes += payload
            if h == 0 and failure_injection == "after_target_pass2":
                raise RuntimeError("injected failure after target affine pass 2")
            all_center_m[h].copy_(m_c)
            all_radius_m[h].copy_(m_r)
            all_center_q[h].copy_(q_c)
            all_radius_q[h].copy_(q_r)
    return _V1LiteralJAffineShellResidualTwoPassNumerics(
        mjz_center=all_center_m.contiguous(),
        mjz_radius=all_radius_m.contiguous(),
        q_center=all_center_q.contiguous(),
        q_radius=all_radius_q.contiguous(),
        source_route_window_read_count=window_reads,
        source_payload_read_bytes=payload_bytes,
        source_tile_submission_count=tile_submissions,
    )


@dataclass(frozen=True, slots=True)
class V1LiteralJAffineShellResidualPopulationProjectionCUDAExecutionReceipt:
    cuda_registration_hash: str
    population_projection_registration_hash: str
    suffix_registration_hash: str
    packet_receipt_hash: str
    prefix_freeze_receipt_hash: str
    row_group_plan_hash: str
    row_group_commit_receipt_hash: str
    scorer_provider_proof_hashes: tuple[str, str]
    scorer_proof_closure_hashes: tuple[str, str]
    literal_j_shard_receipt_hashes: tuple[str, str]
    literal_j_content_sha256s: tuple[str, str]
    literal_j_shard_payload_bytes: tuple[int, int]
    literal_j_shard_route_window_counts: tuple[int, int]
    literal_j_stream_source_hash: str
    same_capture_vjp_source_hashes: tuple[str, str]
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    shell_size: int
    shell_indices: tuple[int, ...]
    shell_route_ids: tuple[str, ...]
    direction_ids: tuple[str, ...]
    d_fixed_projection_direction_axis_hash: str
    affine_direction_receipt_hash: str
    affine_direction_source_hash: str
    affine_direction_preoutcome_registration_hash: str
    preoutcome_fanout_receipt_hash: str
    preoutcome_physical_action_axis_hash: str
    preoutcome_binding_axis_hash: str
    preoutcome_destination_binding_hash: str
    affine_shell_shard_replay_receipt_hash: str
    affine_support_receipt_hash: str
    route_support_source_receipt_hash: str
    joint_physical_action_count: int
    joint_shell_basis_count: int
    complete_preoutcome_action_axis_destination_count: int
    stratum_ids: tuple[str, str]
    residual_population_counts: tuple[int, int]
    observed_counts: tuple[int, int]
    residual_sealed_population_order_hashes: tuple[str, str]
    observed_prefix_axis_hashes: tuple[str, str]
    observed_source_measurement_axis_hashes: tuple[str, str]
    source_pass_axis_hashes: tuple[str, str]
    mjz_center_fingerprint: str
    mjz_radius_fingerprint: str
    q_center_fingerprint: str
    q_radius_fingerprint: str
    simultaneous_statement_count: int
    familywise_alpha: float
    per_statement_two_sided_alpha: float
    alpha_spent: float
    exact_residual_census: bool
    planning_hbm_forecast_hash: str
    immutable_shard_read_pass_count: int
    source_route_window_read_count: int
    source_payload_read_bytes: int
    host_to_device_payload_bytes: int
    source_tile_submission_count: int
    output_receipt_scalar_d2h_count: int
    output_receipt_scalar_d2h_bytes: int
    production_cuda_execution: bool = True
    private_cpu_reference_execution: bool = False
    exact_fixed_time_family: bool = True
    structured_affine_shell_direction_axis: bool = True
    action_qualified_joint_statement_axis: bool = True
    multi_action_fanout_requires_joint_axis_owner: bool = True
    single_action_executor_only: bool = True
    dense_basis_bank_materialized: bool = False
    context_by_route_hbm_bank_retained: bool = False
    route_scale_device_to_host_transfer: bool = False
    private_primal_pair_axis_claimed: bool = False
    adaptive_stopping_authorized: bool = False
    rank_or_kkt_face_authority: bool = False
    c_k_o_commit_terminal_or_pareto_authority: bool = False
    algorithm: str = LITERAL_J_AFFINE_SHELL_RESIDUAL_CUDA_ALGORITHM
    authority_scope: str = LITERAL_J_AFFINE_SHELL_RESIDUAL_CUDA_AUTHORITY
    schema: str = LITERAL_J_AFFINE_SHELL_RESIDUAL_CUDA_EXECUTION_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _EXECUTION_TOKEN:
            raise PermissionError("affine-shell residual executions require issuer")
        for name in (
            "cuda_registration_hash",
            "population_projection_registration_hash",
            "suffix_registration_hash",
            "packet_receipt_hash",
            "prefix_freeze_receipt_hash",
            "row_group_plan_hash",
            "row_group_commit_receipt_hash",
            "literal_j_stream_source_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "d_fixed_projection_direction_axis_hash",
            "affine_direction_receipt_hash",
            "affine_direction_source_hash",
            "affine_direction_preoutcome_registration_hash",
            "preoutcome_fanout_receipt_hash",
            "preoutcome_physical_action_axis_hash",
            "preoutcome_binding_axis_hash",
            "preoutcome_destination_binding_hash",
            "affine_shell_shard_replay_receipt_hash",
            "affine_support_receipt_hash",
            "route_support_source_receipt_hash",
            "mjz_center_fingerprint",
            "mjz_radius_fingerprint",
            "q_center_fingerprint",
            "q_radius_fingerprint",
            "planning_hbm_forecast_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "scorer_provider_proof_hashes",
            "scorer_proof_closure_hashes",
            "literal_j_shard_receipt_hashes",
            "literal_j_content_sha256s",
            "same_capture_vjp_source_hashes",
            "stratum_ids",
            "residual_sealed_population_order_hashes",
            "observed_prefix_axis_hashes",
            "observed_source_measurement_axis_hashes",
            "source_pass_axis_hashes",
        ):
            object.__setattr__(
                self, name, _identity_pair(getattr(self, name), name=name)
            )
        for name in (
            "shell_route_ids",
            "direction_ids",
            "frozen_dw2_prefix_axis_hashes",
        ):
            if hasattr(self, name):
                object.__setattr__(
                    self, name, _identity_axis(getattr(self, name), name=name)
                )
        for name in (
            "literal_j_shard_payload_bytes",
            "literal_j_shard_route_window_counts",
            "residual_population_counts",
            "observed_counts",
        ):
            object.__setattr__(self, name, _count_pair(getattr(self, name), name=name))
        routes = _positive_int(self.route_count, name="route_count")
        joint_actions = _positive_int(
            self.joint_physical_action_count,
            name="joint_physical_action_count",
        )
        joint_shell = _positive_int(
            self.joint_shell_basis_count,
            name="joint_shell_basis_count",
        )
        fanout_count = _positive_int(
            self.complete_preoutcome_action_axis_destination_count,
            name="complete_preoutcome_action_axis_destination_count",
        )
        shell = _positive_int(self.shell_size, name="shell_size")
        indices = tuple(
            _nonnegative_int(item, name="shell index") for item in self.shell_indices
        )
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal_generation"),
        )
        for name in (
            "simultaneous_statement_count",
            "immutable_shard_read_pass_count",
            "source_route_window_read_count",
            "source_payload_read_bytes",
            "host_to_device_payload_bytes",
            "source_tile_submission_count",
            "output_receipt_scalar_d2h_count",
            "output_receipt_scalar_d2h_bytes",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name=name))
        alpha = _finite_probability(self.familywise_alpha, name="familywise_alpha")
        per_statement = _finite_probability(
            self.per_statement_two_sided_alpha,
            name="per_statement_two_sided_alpha",
        )
        if type(self.alpha_spent) is not float or not math.isfinite(self.alpha_spent):
            raise TypeError("alpha_spent must be one finite exact float")
        census = self.observed_counts == self.residual_population_counts
        flags = (
            self.exact_residual_census,
            self.production_cuda_execution,
            self.private_cpu_reference_execution,
            self.exact_fixed_time_family,
            self.structured_affine_shell_direction_axis,
            self.action_qualified_joint_statement_axis,
            self.multi_action_fanout_requires_joint_axis_owner,
            self.single_action_executor_only,
            self.dense_basis_bank_materialized,
            self.context_by_route_hbm_bank_retained,
            self.route_scale_device_to_host_transfer,
            self.private_primal_pair_axis_claimed,
            self.adaptive_stopping_authorized,
            self.rank_or_kkt_face_authority,
            self.c_k_o_commit_terminal_or_pareto_authority,
        )
        if (
            any(type(value) is not bool for value in flags)
            or shell != len(indices)
            or joint_actions != 1
            or fanout_count != 1
            or joint_shell != shell
            or not 1 <= shell <= 8
            or len(self.direction_ids) != shell + 1
            or self.simultaneous_statement_count
            != 2 * (joint_actions + joint_shell) * (routes + 1)
            or per_statement
            != _divide_probability_down(alpha, self.simultaneous_statement_count)
            or any(
                observed <= 0 or observed > population
                for observed, population in zip(
                    self.observed_counts,
                    self.residual_population_counts,
                    strict=True,
                )
            )
            or self.exact_residual_census != census
            or (census and self.alpha_spent != 0.0)
            or (not census and self.alpha_spent != alpha)
            or self.production_cuda_execution == self.private_cpu_reference_execution
            or self.immutable_shard_read_pass_count != 2
            or self.source_payload_read_bytes
            != 2 * sum(self.literal_j_shard_payload_bytes)
            or self.source_route_window_read_count
            != 2 * sum(self.literal_j_shard_route_window_counts)
            or self.source_payload_read_bytes != self.host_to_device_payload_bytes
            or self.output_receipt_scalar_d2h_count != 32
            or self.output_receipt_scalar_d2h_bytes != 32 * _F64_BYTES
            or self.exact_fixed_time_family is not True
            or self.structured_affine_shell_direction_axis is not True
            or self.action_qualified_joint_statement_axis is not True
            or self.multi_action_fanout_requires_joint_axis_owner is not True
            or self.single_action_executor_only is not True
            or any(flags[8:])
            or self.algorithm != LITERAL_J_AFFINE_SHELL_RESIDUAL_CUDA_ALGORITHM
            or self.authority_scope != LITERAL_J_AFFINE_SHELL_RESIDUAL_CUDA_AUTHORITY
            or self.schema != LITERAL_J_AFFINE_SHELL_RESIDUAL_CUDA_EXECUTION_SCHEMA
        ):
            _refuse(
                V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.AUTHORITY_VIOLATION,
                "affine-shell residual execution crossed its statistical boundary",
            )
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "joint_physical_action_count", joint_actions)
        object.__setattr__(self, "joint_shell_basis_count", joint_shell)
        object.__setattr__(
            self,
            "complete_preoutcome_action_axis_destination_count",
            fanout_count,
        )
        object.__setattr__(self, "shell_size", shell)
        object.__setattr__(self, "shell_indices", indices)
        object.__setattr__(self, "familywise_alpha", alpha)
        object.__setattr__(self, "per_statement_two_sided_alpha", per_statement)
        self.assert_integrity()

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            }
        )

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = _payload_hash(self._payload(), role="execution-receipt")
        if self._receipt_hash is not None and observed != self._receipt_hash:
            _refuse(
                V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
                "affine-shell residual execution receipt mutated",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
                "affine-shell residual execution receipt hash differs",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1LiteralJAffineShellResidualPopulationProjectionCUDAExecution:
    receipt: V1LiteralJAffineShellResidualPopulationProjectionCUDAExecutionReceipt
    planning_hbm_forecast: V1LiteralJAffineShellResidualPopulationProjectionCUDAHBMForecast
    population_result: V1LiteralJResidualPopulationProjectionResult = field(
        repr=False, compare=False
    )
    _issuance_token: InitVar[object | None] = None

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _EXECUTION_TOKEN:
            raise PermissionError("affine-shell residual executions require issuer")
        if (
            type(self.receipt)
            is not V1LiteralJAffineShellResidualPopulationProjectionCUDAExecutionReceipt
            or type(self.planning_hbm_forecast)
            is not V1LiteralJAffineShellResidualPopulationProjectionCUDAHBMForecast
            or type(self.population_result) is not V1LiteralJResidualPopulationProjectionResult
        ):
            raise TypeError("affine-shell residual execution components must be exact")
        self.receipt.assert_integrity()
        self.planning_hbm_forecast.assert_integrity(
            self.receipt.planning_hbm_forecast_hash
        )
        self.population_result.device_handle.assert_live()
        result = self.population_result.receipt
        if (
            result.registration_hash
            != self.receipt.population_projection_registration_hash
            or result.prefix_freeze_receipt_hash
            != self.receipt.prefix_freeze_receipt_hash
            or result.observed_counts != self.receipt.observed_counts
            or result.mjz_center_fingerprint != self.receipt.mjz_center_fingerprint
            or result.mjz_radius_fingerprint != self.receipt.mjz_radius_fingerprint
            or result.q_center_fingerprint != self.receipt.q_center_fingerprint
            or result.q_radius_fingerprint != self.receipt.q_radius_fingerprint
            or result.production_cuda_execution != self.receipt.production_cuda_execution
            or result.private_cpu_reference_execution
            != self.receipt.private_cpu_reference_execution
        ):
            _refuse(
                V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
                "affine-shell residual execution and canonical result diverged",
            )


def _execute_v1_literal_j_affine_shell_residual_population_projection_cuda(
    *,
    cuda_registration: V1LiteralJAffineShellResidualPopulationProjectionCUDARegistration,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    suffix_registration: V1SampledVariableShellDW2SuffixRegistration,
    packet: V1SampledVariableShellDW2SuffixPacket,
    prefix_freeze: V1SampledVariableShellDW2SuffixPrefixFreezeReceipt,
    row_group_plan: V1SampledDualChartRowGroupPlan,
    row_group_commit: V1SampledDualChartRowGroupCommit,
    packet_proof_closures: Sequence[V1SampledDualChartRowGroupPacketProofClosure],
    affine_directions: V1LiteralJAffineShellDirectionDeviceHandle,
    affine_support: V1LiteralJAffineShellAnalyticSupportDeviceHandle,
    route_support_source: V1LiteralJResidualAnalyticSupportSourceCUDAResult,
    design: V1LiteralJAffineShellResidualPopulationProjectionCUDADesign,
    planning_hbm_forecast: V1LiteralJAffineShellResidualPopulationProjectionCUDAHBMForecast,
    production_cuda_required: bool,
    failure_injection: str | None = None,
) -> V1LiteralJAffineShellResidualPopulationProjectionCUDAExecution:
    attachments, closures, measurement_hashes, pass_hashes, same_capture_hashes = (
        _validate_execution_source_closure(
            cuda_registration=cuda_registration,
            population_registration=population_registration,
            suffix_registration=suffix_registration,
            packet=packet,
            prefix_freeze=prefix_freeze,
            row_group_plan=row_group_plan,
            row_group_commit=row_group_commit,
            packet_proof_closures=packet_proof_closures,
            design=design,
        )
    )
    if (
        type(affine_directions) is not V1LiteralJAffineShellDirectionDeviceHandle
        or type(affine_support)
        is not V1LiteralJAffineShellAnalyticSupportDeviceHandle
        or type(route_support_source)
        is not V1LiteralJResidualAnalyticSupportSourceCUDAResult
        or type(planning_hbm_forecast)
        is not V1LiteralJAffineShellResidualPopulationProjectionCUDAHBMForecast
    ):
        raise TypeError("affine-shell residual execution requires exact live support")
    affine_support.receipt.assert_integrity()
    route_support_source.receipt.assert_integrity()
    route_support_source.source_handle.assert_live()
    direction = affine_directions.receipt
    support = affine_support.receipt
    route_source = route_support_source.receipt
    if (
        cuda_registration.affine_direction_receipt_hash != direction.receipt_hash
        or cuda_registration.affine_direction_source_hash
        != direction.direction_source_hash
        or cuda_registration.preoutcome_fanout_receipt_hash
        != direction.preoutcome_fanout_receipt_hash
        or cuda_registration.preoutcome_physical_action_axis_hash
        != direction.preoutcome_physical_action_axis_hash
        or cuda_registration.preoutcome_binding_axis_hash
        != direction.preoutcome_binding_axis_hash
        or cuda_registration.preoutcome_destination_binding_hash
        != direction.preoutcome_destination_binding_hash
        or cuda_registration.affine_support_receipt_hash != support.receipt_hash
        or cuda_registration.route_support_source_receipt_hash
        != route_source.receipt_hash
        or planning_hbm_forecast.design_hash != design.design_hash
        or planning_hbm_forecast.affine_direction_source_hash
        != direction.direction_source_hash
        or planning_hbm_forecast.affine_support_receipt_hash != support.receipt_hash
        or planning_hbm_forecast.route_support_source_receipt_hash
        != route_source.receipt_hash
        or planning_hbm_forecast.observed_counts
        != cuda_registration.registered_observed_counts
        or planning_hbm_forecast.admitted is not True
    ):
        _refuse(
            V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
            "affine-shell residual support or HBM forecast left execution custody",
        )
    if production_cuda_required:
        if torch is None or not torch.cuda.is_available():
            _refuse(
                V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.DEVICE_REQUIRED,
                "production affine-shell residual execution requires CUDA",
            )
        if (
            design.deterministic_algorithms_required
            and not torch.are_deterministic_algorithms_enabled()
        ):
            _refuse(
                V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode.DETERMINISM_DISABLED,
                "production execution requires deterministic CUDA algorithms",
            )
    with route_support_source.source_handle.borrow() as route_lease:
        with affine_support.borrow() as support_lease:
            literal_j_abs_bound = route_lease.literal_j_abs_bound
            projection_abs_bounds = support_lease.support_vector()
            numerics = (
                _compute_v1_literal_j_affine_shell_residual_two_pass_device_numerics(
                    attachments=attachments,
                    affine_directions=affine_directions,
                    literal_j_abs_bound=literal_j_abs_bound,
                    projection_abs_bounds=projection_abs_bounds,
                    residual_population_counts=(
                        population_registration.residual_stratum_counts
                    ),
                    familywise_alpha=population_registration.familywise_alpha,
                    design=design,
                    production_cuda_required=production_cuda_required,
                    failure_injection=failure_injection,
                )
            )
    observed = cuda_registration.registered_observed_counts
    residual = population_registration.residual_stratum_counts
    observed_prefix_hashes = (
        _canonical_observed_prefix_hash(
            registration_hash=population_registration.registration_hash,
            stratum_id=population_registration.stratum_ids[0],
            observed=observed[0],
            source_hash=prefix_freeze.dw2_target_prefix_axis_hash,
        ),
        _canonical_observed_prefix_hash(
            registration_hash=population_registration.registration_hash,
            stratum_id=population_registration.stratum_ids[1],
            observed=observed[1],
            source_hash=prefix_freeze.dw2_complement_prefix_axis_hash,
        ),
    )
    output_fingerprints = (
        fingerprint_v1_literal_j_affine_shell_residual_cuda_tensor(
            numerics.mjz_center, role="affine_shell_residual_mjz_center"
        ),
        fingerprint_v1_literal_j_affine_shell_residual_cuda_tensor(
            numerics.mjz_radius, role="affine_shell_residual_mjz_radius"
        ),
        fingerprint_v1_literal_j_affine_shell_residual_cuda_tensor(
            numerics.q_center, role="affine_shell_residual_q_center"
        ),
        fingerprint_v1_literal_j_affine_shell_residual_cuda_tensor(
            numerics.q_radius, role="affine_shell_residual_q_radius"
        ),
    )
    shards = tuple(item.receipt for item in attachments)
    exact_census = observed == residual
    receipt = V1LiteralJAffineShellResidualPopulationProjectionCUDAExecutionReceipt(
        cuda_registration_hash=cuda_registration.registration_hash,
        population_projection_registration_hash=(
            population_registration.registration_hash
        ),
        suffix_registration_hash=suffix_registration.receipt_hash,
        packet_receipt_hash=packet.receipt_hash,
        prefix_freeze_receipt_hash=prefix_freeze.receipt_hash,
        row_group_plan_hash=row_group_plan.group_plan_hash,
        row_group_commit_receipt_hash=row_group_commit.receipt.receipt_hash,
        scorer_provider_proof_hashes=tuple(
            item.provider_proof_hash for item in closures
        ),  # type: ignore[arg-type]
        scorer_proof_closure_hashes=tuple(
            item.closure_hash for item in closures
        ),  # type: ignore[arg-type]
        literal_j_shard_receipt_hashes=tuple(
            item.receipt_hash for item in shards
        ),  # type: ignore[arg-type]
        literal_j_content_sha256s=tuple(
            item.content_sha256 for item in shards
        ),  # type: ignore[arg-type]
        literal_j_shard_payload_bytes=tuple(
            item.exact_payload_file_bytes for item in shards
        ),  # type: ignore[arg-type]
        literal_j_shard_route_window_counts=tuple(
            len(item.route_windows) for item in shards
        ),  # type: ignore[arg-type]
        literal_j_stream_source_hash=(
            population_registration.residual_literal_j_stream_source_hash
        ),
        same_capture_vjp_source_hashes=same_capture_hashes,
        model_identity_hash=population_registration.model_identity_hash,
        chart_identity_hash=population_registration.chart_identity_hash,
        current_mask_hash=population_registration.current_mask_hash,
        proposal_generation=population_registration.proposal_generation,
        route_axis_hash=population_registration.route_axis_hash,
        route_id_axis_hash=population_registration.route_id_axis_hash,
        route_count=population_registration.route_count,
        shell_size=len(direction.shell_indices),
        shell_indices=direction.shell_indices,
        shell_route_ids=direction.shell_route_ids,
        direction_ids=direction.direction_ids,
        d_fixed_projection_direction_axis_hash=(
            population_registration.d_fixed_projection_direction_axis_hash
        ),
        affine_direction_receipt_hash=direction.receipt_hash,
        affine_direction_source_hash=direction.direction_source_hash,
        affine_direction_preoutcome_registration_hash=(
            direction.preoutcome_source_registration_hash
        ),
        preoutcome_fanout_receipt_hash=(
            cuda_registration.preoutcome_fanout_receipt_hash
        ),
        preoutcome_physical_action_axis_hash=(
            cuda_registration.preoutcome_physical_action_axis_hash
        ),
        preoutcome_binding_axis_hash=cuda_registration.preoutcome_binding_axis_hash,
        preoutcome_destination_binding_hash=(
            cuda_registration.preoutcome_destination_binding_hash
        ),
        affine_shell_shard_replay_receipt_hash=(
            population_registration.affine_shell_shard_replay_receipt_hash
        ),
        affine_support_receipt_hash=support.receipt_hash,
        route_support_source_receipt_hash=route_source.receipt_hash,
        joint_physical_action_count=cuda_registration.joint_physical_action_count,
        joint_shell_basis_count=cuda_registration.joint_shell_basis_count,
        complete_preoutcome_action_axis_destination_count=(
            cuda_registration.complete_preoutcome_action_axis_destination_count
        ),
        stratum_ids=population_registration.stratum_ids,
        residual_population_counts=residual,
        observed_counts=observed,
        residual_sealed_population_order_hashes=(
            population_registration.residual_sealed_population_order_hashes
        ),
        observed_prefix_axis_hashes=observed_prefix_hashes,
        observed_source_measurement_axis_hashes=measurement_hashes,
        source_pass_axis_hashes=pass_hashes,
        mjz_center_fingerprint=output_fingerprints[0],
        mjz_radius_fingerprint=output_fingerprints[1],
        q_center_fingerprint=output_fingerprints[2],
        q_radius_fingerprint=output_fingerprints[3],
        simultaneous_statement_count=(
            cuda_registration.simultaneous_statement_count
        ),
        familywise_alpha=population_registration.familywise_alpha,
        per_statement_two_sided_alpha=(
            cuda_registration.per_statement_two_sided_alpha
        ),
        alpha_spent=(
            0.0 if exact_census else population_registration.familywise_alpha
        ),
        exact_residual_census=exact_census,
        planning_hbm_forecast_hash=planning_hbm_forecast.forecast_hash,
        immutable_shard_read_pass_count=2,
        source_route_window_read_count=numerics.source_route_window_read_count,
        source_payload_read_bytes=numerics.source_payload_read_bytes,
        host_to_device_payload_bytes=numerics.source_payload_read_bytes,
        source_tile_submission_count=numerics.source_tile_submission_count,
        output_receipt_scalar_d2h_count=32,
        output_receipt_scalar_d2h_bytes=32 * _F64_BYTES,
        production_cuda_execution=production_cuda_required,
        private_cpu_reference_execution=not production_cuda_required,
        _issuance_token=_EXECUTION_TOKEN,
    )
    population_result = (
        _issue_v1_literal_j_affine_shell_residual_population_projection_from_cuda_producer(
            producer_capability=_AFFINE_SHELL_RESIDUAL_CUDA_PRODUCER_CAPABILITY,
            registration=population_registration,
            prefix_freeze=prefix_freeze,
            cuda_registration=cuda_registration,
            cuda_execution_receipt=receipt,
            mjz_center=numerics.mjz_center,
            mjz_radius=numerics.mjz_radius,
            q_center=numerics.q_center,
            q_radius=numerics.q_radius,
        )
    )
    return V1LiteralJAffineShellResidualPopulationProjectionCUDAExecution(
        receipt=receipt,
        planning_hbm_forecast=planning_hbm_forecast,
        population_result=population_result,
        _issuance_token=_EXECUTION_TOKEN,
    )


def execute_v1_literal_j_affine_shell_residual_population_projection_cuda(
    *,
    cuda_registration: V1LiteralJAffineShellResidualPopulationProjectionCUDARegistration,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    suffix_registration: V1SampledVariableShellDW2SuffixRegistration,
    packet: V1SampledVariableShellDW2SuffixPacket,
    prefix_freeze: V1SampledVariableShellDW2SuffixPrefixFreezeReceipt,
    row_group_plan: V1SampledDualChartRowGroupPlan,
    row_group_commit: V1SampledDualChartRowGroupCommit,
    packet_proof_closures: Sequence[V1SampledDualChartRowGroupPacketProofClosure],
    affine_directions: V1LiteralJAffineShellDirectionDeviceHandle,
    affine_support: V1LiteralJAffineShellAnalyticSupportDeviceHandle,
    route_support_source: V1LiteralJResidualAnalyticSupportSourceCUDAResult,
    design: V1LiteralJAffineShellResidualPopulationProjectionCUDADesign,
    planning_hbm_forecast: V1LiteralJAffineShellResidualPopulationProjectionCUDAHBMForecast,
) -> V1LiteralJAffineShellResidualPopulationProjectionCUDAExecution:
    return _execute_v1_literal_j_affine_shell_residual_population_projection_cuda(
        cuda_registration=cuda_registration,
        population_registration=population_registration,
        suffix_registration=suffix_registration,
        packet=packet,
        prefix_freeze=prefix_freeze,
        row_group_plan=row_group_plan,
        row_group_commit=row_group_commit,
        packet_proof_closures=packet_proof_closures,
        affine_directions=affine_directions,
        affine_support=affine_support,
        route_support_source=route_support_source,
        design=design,
        planning_hbm_forecast=planning_hbm_forecast,
        production_cuda_required=True,
    )


def _execute_v1_literal_j_affine_shell_residual_population_projection_cpu_reference(
    *,
    cuda_registration: V1LiteralJAffineShellResidualPopulationProjectionCUDARegistration,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    suffix_registration: V1SampledVariableShellDW2SuffixRegistration,
    packet: V1SampledVariableShellDW2SuffixPacket,
    prefix_freeze: V1SampledVariableShellDW2SuffixPrefixFreezeReceipt,
    row_group_plan: V1SampledDualChartRowGroupPlan,
    row_group_commit: V1SampledDualChartRowGroupCommit,
    packet_proof_closures: Sequence[V1SampledDualChartRowGroupPacketProofClosure],
    affine_directions: V1LiteralJAffineShellDirectionDeviceHandle,
    affine_support: V1LiteralJAffineShellAnalyticSupportDeviceHandle,
    route_support_source: V1LiteralJResidualAnalyticSupportSourceCUDAResult,
    design: V1LiteralJAffineShellResidualPopulationProjectionCUDADesign,
    planning_hbm_forecast: V1LiteralJAffineShellResidualPopulationProjectionCUDAHBMForecast,
    failure_injection: str | None = None,
) -> V1LiteralJAffineShellResidualPopulationProjectionCUDAExecution:
    return _execute_v1_literal_j_affine_shell_residual_population_projection_cuda(
        cuda_registration=cuda_registration,
        population_registration=population_registration,
        suffix_registration=suffix_registration,
        packet=packet,
        prefix_freeze=prefix_freeze,
        row_group_plan=row_group_plan,
        row_group_commit=row_group_commit,
        packet_proof_closures=packet_proof_closures,
        affine_directions=affine_directions,
        affine_support=affine_support,
        route_support_source=route_support_source,
        design=design,
        planning_hbm_forecast=planning_hbm_forecast,
        production_cuda_required=False,
        failure_injection=failure_injection,
    )


__all__ = [
    "LITERAL_J_AFFINE_SHELL_RESIDUAL_CUDA_ALGORITHM",
    "LITERAL_J_AFFINE_SHELL_RESIDUAL_CUDA_AUTHORITY",
    "LITERAL_J_AFFINE_SHELL_RESIDUAL_CUDA_DESIGN_SCHEMA",
    "LITERAL_J_AFFINE_SHELL_RESIDUAL_CUDA_EXECUTION_SCHEMA",
    "LITERAL_J_AFFINE_SHELL_RESIDUAL_CUDA_FORECAST_SCHEMA",
    "LITERAL_J_AFFINE_SHELL_RESIDUAL_CUDA_REGISTRATION_SCHEMA",
    "V1LiteralJAffineShellResidualPopulationProjectionCUDADesign",
    "V1LiteralJAffineShellResidualPopulationProjectionCUDAExecution",
    "V1LiteralJAffineShellResidualPopulationProjectionCUDAExecutionReceipt",
    "V1LiteralJAffineShellResidualPopulationProjectionCUDAHBMForecast",
    "V1LiteralJAffineShellResidualPopulationProjectionCUDARefusal",
    "V1LiteralJAffineShellResidualPopulationProjectionCUDARefusalCode",
    "V1LiteralJAffineShellResidualPopulationProjectionCUDARegistration",
    "build_v1_literal_j_affine_shell_residual_population_projection_cuda_hbm_forecast",
    "execute_v1_literal_j_affine_shell_residual_population_projection_cuda",
    "fingerprint_v1_literal_j_affine_shell_residual_cuda_tensor",
    "register_v1_literal_j_affine_shell_residual_population_projection_cuda",
    "register_v1_literal_j_affine_shell_residual_population_projection_cuda_design",
]
