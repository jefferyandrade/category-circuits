"""Bounded joint literal-J residual population projection stream.

The statistical family is the one pre-outcome, action-complete family

``(z_outside^1, ..., z_outside^A, e_b1, ..., e_bB)``

registered by :mod:`modern_v1_literal_j_joint_affine_shell_family_cuda`.
Pass 1 reads every immutable raw literal-J tile once and retains only the
``context x (A+B)`` projections needed for Q.  Before pass 2 opens, their
registered shell-basis columns also produce one guarded compact
``2 x (A+B) x B`` slice ``(M_h v_d)|_shell``.  Pass 2 then rereads every tile
once in route-major target/complement pairs and exposes one guarded
``(A+B) x route_tile`` M_J v interval at a time.  The paired ordering lets the
global support-plane consumer combine the target and complement contributions
for a route interval before eviction, without an O(route_count) partial
coefficient buffer.  The consumer must evict the active tile before the stream
can advance.
Therefore neither a context-by-route row bank nor a direction-by-route result
bank can be materialised or retained by this owner.

The canonical production entry points accept the preoutcome current-source
logical-look source and its exact postoutcome bound result.  Arbitrarily many
physical transport shards are grouped into the two registered statistical
strata without concatenating raw rows or rerunning the provider/VJP.  The
historical one-packet/two-shard entry points remain compatibility readers.
CPU execution is a small private numerical reference used by focused contract
tests; it grants no production authority.
This module never grants D-face, C, K, O, mask-commit, terminal-validation,
runtime-promotion, or Pareto authority.
"""

from __future__ import annotations

from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
from fractions import Fraction
import math
from threading import RLock
from types import MappingProxyType
from typing import Any, Final, Iterator, Mapping, NoReturn

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_literal_j_affine_shell_analytic_support_cuda import (
    V1LiteralJAffineShellAnalyticSupportDeviceHandle,
    V1LiteralJAffineShellAnalyticSupportLease,
)
from .modern_v1_literal_j_d_fixed_residual_population_projection import (
    V1LiteralJDFixedResidualPopulationProjectionRegistration,
    V1LiteralJJointAxisResidualPopulationMeasureRegistration,
    V1LiteralJJointAxisResidualPopulationMeasureRegistrationReceipt,
)
from .modern_v1_literal_j_joint_affine_shell_family_cuda import (
    V1LiteralJJointAffineShellFamilyDeviceHandle,
    V1LiteralJJointAffineShellFamilyLease,
    V1LiteralJJointAffineShellPreoutcomeRegistration,
)
from .modern_v1_literal_j_joint_population_source import (
    V1LiteralJJointPopulationPreoutcomeSource,
    V1LiteralJJointPopulationSource,
)
from .modern_v1_literal_j_joint_population_action_axis import (
    V1LiteralJJointPopulationActionAxisPreoutcomeSource,
    V1LiteralJJointPopulationActionAxisSource,
)
from .modern_v1_literal_j_residual_population_projection_cuda import (
    V1LiteralJResidualAnalyticSupportSourceCUDAResult,
    _AnalyticSupportSourceLease,
    _add_up,
    _div_up,
    _gamma,
    _MIN_SUBNORMAL,
    _mul_up,
    _quadratic_interval,
    _scale_interval_by_uniform_mean,
    _serfling_factor,
    _stage_source_tile,
)
from .modern_v1_sampled_dual_chart_row_custody import (
    V1SampledDualChartRowGroupCommit,
    V1SampledDualChartRowGroupPacketProofClosure,
    V1SampledDualChartRowGroupPlan,
    V1SampledDualChartRowRole,
    V1SampledDualChartRowShardAttachment,
)
from .modern_v1_sampled_variable_shell_dw2_suffix import (
    V1SampledVariableShellDW2SuffixPacket,
    V1SampledVariableShellDW2SuffixPrefixFreezeReceipt,
    V1SampledVariableShellDW2SuffixRegistration,
)
from .modern_v1_sampled_v7_shared_dw2_literal_j_logical_look_source import (
    V1SampledV7SharedDW2LiteralJLogicalLookResult,
    V1SampledV7SharedDW2LiteralJLogicalLookSource,
)

try:  # pragma: no cover - runtime-image dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_DESIGN_SCHEMA: Final = (
    "partition_free_v1_literal_J_joint_affine_shell_residual_stream_design_v1"
)
LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_literal_J_joint_affine_shell_residual_stream_registration_v2"
)
LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_PREOUTCOME_PLAN_SCHEMA: Final = (
    "partition_free_v1_literal_J_joint_affine_shell_residual_stream_"
    "action_complete_preoutcome_plan_v2"
)
LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_POSTSOURCE_BINDING_SCHEMA: Final = (
    "partition_free_v1_literal_J_joint_affine_shell_residual_stream_"
    "action_complete_postsource_binding_v2"
)
LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_FORECAST_SCHEMA: Final = (
    "partition_free_v1_literal_J_joint_affine_shell_residual_stream_hbm_forecast_v1"
)
LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_Q_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_literal_J_joint_affine_shell_residual_stream_Q_receipt_v1"
)
LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_COMPACT_SHELL_M_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_literal_J_joint_affine_shell_residual_stream_"
    "pass1_compact_shell_M_receipt_v2"
)
LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_TILE_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_literal_J_joint_affine_shell_residual_stream_tile_receipt_v4"
)
LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_EXECUTION_SCHEMA: Final = (
    "partition_free_v1_literal_J_joint_affine_shell_residual_stream_execution_v4"
)
LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_ALGORITHM: Final = (
    "two_pass_raw_literal_J_single_read_per_tile_joint_action_shell_projection_"
    "with_context_direction_Q_and_route_major_intersection_zipped_target_"
    "complement_M_stream_with_segmented_logical_look_source_v4"
)
LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_AUTHORITY: Final = (
    "joint_fixed_time_residual_MJv_and_Q_interval_numerics_only_no_D_face_C_K_O_"
    "commit_terminal_runtime_promotion_or_pareto_authority_v1"
)

_PREFIX: Final = "partition-free-v1-literal-j-joint-affine-shell-residual-stream-"
_DESIGN_TOKEN: Final = object()
_REGISTRATION_TOKEN: Final = object()
_FORECAST_TOKEN: Final = object()
_Q_TOKEN: Final = object()
_COMPACT_SHELL_M_TOKEN: Final = object()
_TILE_TOKEN: Final = object()
_EXECUTION_TOKEN: Final = object()
_CPU_SUPPORT_TOKEN: Final = object()
_PREOUTCOME_PLAN_RECEIPT_TOKEN: Final = object()
_PREOUTCOME_PLAN_OWNER_TOKEN: Final = object()
_POSTSOURCE_BINDING_RECEIPT_TOKEN: Final = object()
_POSTSOURCE_BINDING_OWNER_TOKEN: Final = object()
_F64_BYTES: Final = 8
_CPU_REFERENCE_ROUTE_LIMIT: Final = 4096

__all__ = (
    "LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_DESIGN_SCHEMA",
    "LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_REGISTRATION_SCHEMA",
    "LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_PREOUTCOME_PLAN_SCHEMA",
    "LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_POSTSOURCE_BINDING_SCHEMA",
    "LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_FORECAST_SCHEMA",
    "LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_Q_RECEIPT_SCHEMA",
    "LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_COMPACT_SHELL_M_RECEIPT_SCHEMA",
    "LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_TILE_RECEIPT_SCHEMA",
    "LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_EXECUTION_SCHEMA",
    "LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_ALGORITHM",
    "LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_AUTHORITY",
    "V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode",
    "V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusal",
    "V1LiteralJJointAffineShellResidualPopulationStreamCUDADesign",
    "V1LiteralJJointAffineShellResidualPopulationStreamCUDARegistration",
    "V1LiteralJJointAffineShellResidualPopulationStreamCUDAPreoutcomePlanReceipt",
    "V1LiteralJJointAffineShellResidualPopulationStreamCUDAPreoutcomePlan",
    "V1LiteralJJointAffineShellResidualPopulationStreamCUDAPostsourceBindingReceipt",
    "V1LiteralJJointAffineShellResidualPopulationStreamCUDAPostsourceBinding",
    "V1LiteralJJointAffineShellResidualPopulationStreamCUDAHBMForecast",
    "V1LiteralJJointAffineShellResidualPopulationStreamQReceipt",
    "V1LiteralJJointAffineShellResidualPopulationStreamQLease",
    "V1LiteralJJointAffineShellResidualPopulationStreamQDeviceHandle",
    "V1LiteralJJointAffineShellResidualPopulationStreamCompactShellMReceipt",
    "V1LiteralJJointAffineShellResidualPopulationStreamCompactShellMLease",
    "V1LiteralJJointAffineShellResidualPopulationStreamCompactShellMDeviceHandle",
    "V1LiteralJJointAffineShellResidualPopulationStreamTileReceipt",
    "V1LiteralJJointAffineShellResidualPopulationStreamTileLease",
    "V1LiteralJJointAffineShellResidualPopulationStreamTileDeviceHandle",
    "V1LiteralJJointAffineShellResidualPopulationStreamCUDAExecutionReceipt",
    "V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle",
    "register_v1_literal_j_joint_affine_shell_residual_population_stream_cuda_design",
    "register_v1_literal_j_joint_affine_shell_residual_population_stream_cuda",
    "register_v1_literal_j_joint_affine_shell_residual_population_stream_cuda_from_logical_look_source",
    "register_v1_literal_j_joint_affine_shell_residual_population_stream_cuda_from_joint_population_source",
    "register_v1_literal_j_joint_affine_shell_residual_population_stream_cuda_preoutcome_plan",
    "bind_v1_literal_j_joint_affine_shell_residual_population_stream_cuda_postsource",
    "build_v1_literal_j_joint_affine_shell_residual_population_stream_cuda_hbm_forecast",
    "execute_v1_literal_j_joint_affine_shell_residual_population_stream_cuda",
    "execute_v1_literal_j_joint_affine_shell_residual_population_stream_cuda_from_logical_look_result",
    "execute_v1_literal_j_joint_affine_shell_residual_population_stream_cuda_from_joint_population_source",
    "execute_v1_literal_j_joint_affine_shell_residual_population_stream_cuda_from_postsource_binding",
)


class V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    INVALID_DESIGN = "invalid_design"
    STALE_IDENTITY = "stale_identity"
    PREFIX_MISMATCH = "prefix_mismatch"
    ORDER_MISMATCH = "order_mismatch"
    POSTOUTCOME_REGISTRATION = "postoutcome_registration"
    PREOUTCOME_PLAN_REQUIRED = "preoutcome_plan_required"
    ACTION_AXIS_REQUIRED = "action_axis_required"
    SOURCE_DESCENT_MISMATCH = "source_descent_mismatch"
    JOINT_POPULATION_AXIS_REQUIRED = "joint_population_axis_required"
    ANALYTIC_SUPPORT_REQUIRED = "analytic_support_required"
    SUPPORT_VIOLATION = "support_violation"
    DEVICE_REQUIRED = "device_required"
    CPU_PRODUCTION_FALLBACK = "cpu_production_fallback"
    DETERMINISM_DISABLED = "determinism_disabled"
    HBM_FORECAST_EXCEEDED = "hbm_forecast_exceeded"
    NUMERICAL_REFUSAL = "numerical_refusal"
    ACTIVE_TILE = "active_tile"
    PASS2_ALREADY_OPENED = "pass2_already_opened"
    STREAM_NOT_EXHAUSTED = "stream_not_exhausted"
    MUTATED_CUSTODY = "mutated_custody"
    EVICTED = "evicted"
    AUTHORITY_VIOLATION = "authority_violation"


class V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusal(RuntimeError):
    def __init__(
        self,
        code: V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode,
        message: str,
    ) -> None:
        if (
            type(code)
            is not V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode
        ):
            raise TypeError("joint residual-stream refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode,
    message: str,
) -> NoReturn:
    raise V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusal(code, message)


_PopulationMeasureRegistration = (
    V1LiteralJDFixedResidualPopulationProjectionRegistration
    | V1LiteralJJointAxisResidualPopulationMeasureRegistration
)


def _population_measure_receipt(
    population: _PopulationMeasureRegistration,
) -> (
    V1LiteralJDFixedResidualPopulationProjectionRegistration
    | V1LiteralJJointAxisResidualPopulationMeasureRegistrationReceipt
):
    """Return the array-free measure view without relabelling its J axis."""

    if type(population) is V1LiteralJDFixedResidualPopulationProjectionRegistration:
        population.assert_integrity()
        return population
    if type(population) is V1LiteralJJointAxisResidualPopulationMeasureRegistration:
        population.assert_integrity()
        return population.receipt
    raise TypeError("population measure registration has an inexact owner type")


def _population_residual_counts(
    population: _PopulationMeasureRegistration,
) -> tuple[int, int]:
    view = _population_measure_receipt(population)
    if type(view) is V1LiteralJJointAxisResidualPopulationMeasureRegistrationReceipt:
        return view.residual_population_stratum_counts
    return view.residual_stratum_counts


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty exact string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative exact integer",
        )
    return int(value)


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result <= 0:
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _count_pair(value: object, *, name: str) -> tuple[int, int]:
    if type(value) is not tuple or len(value) != 2:
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one immutable two-entry tuple",
        )
    return (
        _nonnegative_int(value[0], name=f"{name}[0]"),
        _nonnegative_int(value[1], name=f"{name}[1]"),
    )


def _identity_pair(value: object, *, name: str) -> tuple[str, str]:
    if type(value) is not tuple or len(value) != 2:
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one immutable two-entry tuple",
        )
    return (
        _identity(value[0], name=f"{name}[0]"),
        _identity(value[1], name=f"{name}[1]"),
    )


def _nested_identity_pair(
    value: object,
    *,
    name: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if type(value) is not tuple or len(value) != 2:
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one immutable two-stratum tuple",
        )
    return (
        _identity_sequence(value[0], name=f"{name}[0]"),
        _identity_sequence(value[1], name=f"{name}[1]"),
    )


def _identity_axis(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one immutable tuple",
        )
    result = tuple(_identity(item, name=f"{name} entry") for item in value)
    if not result or len(result) != len(set(result)):
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.INVALID_DESIGN,
            f"{name} must be nonempty and unique",
        )
    return result


def _identity_sequence(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one immutable tuple",
        )
    result = tuple(_identity(item, name=f"{name} entry") for item in value)
    if not result:
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.INVALID_DESIGN,
            f"{name} must be nonempty",
        )
    return result


def _finite_alpha(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or not 0.0 < value < 1.0:
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one finite binary64 probability in (0, 1)",
        )
    return value


def _divide_probability_down(value: float, count: int) -> float:
    alpha = _finite_alpha(value, name="familywise_alpha")
    divisor = _positive_int(count, name="simultaneous_statement_count")
    exact = Fraction.from_float(alpha) / divisor
    result = float(exact)
    if Fraction.from_float(result) > exact:
        result = math.nextafter(result, 0.0)
    if not math.isfinite(result) or result <= 0.0:
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.NUMERICAL_REFUSAL,
            "the preregistered per-statement alpha underflowed binary64",
        )
    return result


def _payload_hash(payload: Mapping[str, object], *, role: str) -> str:
    return content_hash(payload, prefix=_PREFIX + role + "-")


def _axis_hash(role: str, values: tuple[str, ...]) -> str:
    return _payload_hash(
        {"schema": "joint_residual_stream_axis_v1", "role": role, "values": values},
        role="axis",
    )


def _collapse_segment_axis(
    *,
    role: str,
    values: tuple[str, ...],
) -> str:
    """Preserve legacy singleton identities; hash multi-segment axes."""

    axis = _identity_sequence(values, name=f"{role} segment axis")
    if len(axis) == 1:
        return axis[0]
    return _axis_hash(role, axis)


def _device_identity(device: object) -> str:
    return str(device)


def _tensor_invariant(value: Any) -> tuple[object, ...]:
    return (
        id(value),
        int(value.data_ptr()),
        tuple(int(item) for item in value.shape),
        tuple(int(item) for item in value.stride()),
        value.dtype,
        value.device,
        bool(value.requires_grad),
    )


def _require_fp64_tensor(
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
        or tuple(value.shape) != shape
        or not value.is_contiguous()
        or bool(value.requires_grad)
        or not bool(torch.isfinite(value).all().item())
        or (device is not None and value.device != device)
    ):
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one finite contiguous FP64 tensor on the exact device",
        )
    if nonnegative and bool((value < 0.0).any().item()):
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.SUPPORT_VIOLATION,
            f"{name} contains a negative analytic bound",
        )
    return value


@dataclass(frozen=True, slots=True)
class V1LiteralJJointAffineShellResidualPopulationStreamCUDADesign:
    route_tile_size: int = 131_072
    action_tile_size: int = 32
    registered_external_workspace_bytes: int = 0
    hbm_ceiling_fraction: float = 0.85
    deterministic_algorithms_required: bool = True
    production_cuda_only: bool = True
    route_and_action_tiles_are_hardware_only: bool = True
    statistical_sample_count_controlled_by_tiles: bool = False
    adaptive_stopping_authorized: bool = False
    schema: str = LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_DESIGN_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _DESIGN_TOKEN:
            raise PermissionError("joint residual-stream designs require their issuer")
        object.__setattr__(
            self, "route_tile_size", _positive_int(self.route_tile_size, name="route_tile_size")
        )
        object.__setattr__(
            self,
            "action_tile_size",
            _positive_int(self.action_tile_size, name="action_tile_size"),
        )
        object.__setattr__(
            self,
            "registered_external_workspace_bytes",
            _nonnegative_int(
                self.registered_external_workspace_bytes,
                name="registered_external_workspace_bytes",
            ),
        )
        positive = (
            self.deterministic_algorithms_required,
            self.production_cuda_only,
            self.route_and_action_tiles_are_hardware_only,
        )
        closed = (
            self.statistical_sample_count_controlled_by_tiles,
            self.adaptive_stopping_authorized,
        )
        if (
            self.hbm_ceiling_fraction != 0.85
            or any(type(item) is not bool or item is not True for item in positive)
            or any(type(item) is not bool or item is not False for item in closed)
            or self.schema != LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_DESIGN_SCHEMA
        ):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.AUTHORITY_VIOLATION,
                "joint residual-stream design crossed its hardware-only boundary",
            )
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {item.name: getattr(self, item.name) for item in fields(self) if not item.name.startswith("_")}
        )

    def _compute_hash(self) -> str:
        return _payload_hash(self._payload(), role="design")

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash or (expected_hash is not None and observed != expected_hash):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.MUTATED_CUSTODY,
                "joint residual-stream design no longer replays",
            )
        return observed

    @property
    def design_hash(self) -> str:
        return self.assert_integrity()


def register_v1_literal_j_joint_affine_shell_residual_population_stream_cuda_design(
    *,
    route_tile_size: int = 131_072,
    action_tile_size: int = 32,
    registered_external_workspace_bytes: int = 0,
) -> V1LiteralJJointAffineShellResidualPopulationStreamCUDADesign:
    return V1LiteralJJointAffineShellResidualPopulationStreamCUDADesign(
        route_tile_size=route_tile_size,
        action_tile_size=action_tile_size,
        registered_external_workspace_bytes=registered_external_workspace_bytes,
        _issuance_token=_DESIGN_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1LiteralJJointAffineShellResidualPopulationStreamCUDAPreoutcomePlanReceipt:
    """Array-free proof that the complete residual action/source axis froze in time."""

    action_axis_preoutcome_receipt_hash: str
    joint_preoutcome_registration_hash: str
    physical_action_axis_hash: str
    joint_direction_axis_hash: str
    joint_direction_ids: tuple[str, ...]
    logical_look_source_receipt_hash: str
    population_projection_registration_hash: str
    population_measure_design_hash: str
    population_measure_weight_receipt_hash: str
    design_hash: str
    analytic_support_derivation_input_hash: str
    analytic_support_derivation_source_receipt_hash: str
    literal_j_abs_bound_source_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    literal_j_stream_source_hash: str
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    suffix_registration_hash: str
    shell_indices: tuple[int, ...]
    action_count: int
    shell_size: int
    direction_count: int
    residual_population_counts: tuple[int, int]
    registered_observed_counts: tuple[int, int]
    residual_sealed_population_order_hashes: tuple[str, str]
    generalized_logical_prefix_identity_hash: str
    row_group_plan_hash: str
    source_physical_call_offset: int
    source_expected_prior_manifest_hash: str | None
    source_physical_roles: tuple[str, ...]
    source_physical_context_counts: tuple[int, ...]
    source_packet_plan_hashes: tuple[str, ...]
    source_provider_plan_hashes: tuple[str, ...]
    source_capture_identity_hashes: tuple[str, ...]
    source_transport_packet_hashes: tuple[str, ...]
    source_stratum_segment_ordinals: tuple[tuple[int, ...], tuple[int, ...]]
    scorer_context_axis_hashes: tuple[str, str]
    scorer_draw_axis_hashes: tuple[str, str]
    familywise_alpha: float
    simultaneous_statement_count: int
    per_statement_two_sided_alpha: float
    d1_context_weight_fraction: tuple[int, int]
    residual_context_weight_fractions: tuple[tuple[int, int], tuple[int, int]]
    residual_stratum_mass_fractions: tuple[tuple[int, int], tuple[int, int]]
    residual_population_mass_fraction: tuple[int, int]
    common_measure_denominator: int
    support_derivation_mode: str
    provider_outcomes_seen: bool = False
    logical_source_bound_at_registration: bool = False
    joint_family_attached: bool = False
    postsource_supports_attached: bool = False
    complete_ordered_action_axis: bool = True
    exact_population_measure_bound: bool = True
    registered_before_residual_outcomes: bool = True
    outcome_payload_present: bool = False
    new_alpha_allocated: bool = False
    d_face_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_AUTHORITY
    schema: str = LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_PREOUTCOME_PLAN_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _PREOUTCOME_PLAN_RECEIPT_TOKEN:
            raise PermissionError("joint residual PRE plans require their issuer")
        for name in (
            "action_axis_preoutcome_receipt_hash",
            "joint_preoutcome_registration_hash",
            "physical_action_axis_hash",
            "joint_direction_axis_hash",
            "logical_look_source_receipt_hash",
            "population_projection_registration_hash",
            "population_measure_design_hash",
            "population_measure_weight_receipt_hash",
            "design_hash",
            "analytic_support_derivation_input_hash",
            "analytic_support_derivation_source_receipt_hash",
            "literal_j_abs_bound_source_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "literal_j_stream_source_hash",
            "registered_d_prefix_hash",
            "registered_d_w2_prefix_hash",
            "suffix_registration_hash",
            "generalized_logical_prefix_identity_hash",
            "row_group_plan_hash",
        ):
            _identity(getattr(self, name), name=name)
        directions = _identity_axis(self.joint_direction_ids, name="joint_direction_ids")
        actions = _positive_int(self.action_count, name="action_count")
        shell = _positive_int(self.shell_size, name="shell_size")
        routes = _positive_int(self.route_count, name="route_count")
        observed = _count_pair(self.registered_observed_counts, name="registered_observed_counts")
        population = _count_pair(self.residual_population_counts, name="residual_population_counts")
        closed = (
            self.provider_outcomes_seen,
            self.logical_source_bound_at_registration,
            self.joint_family_attached,
            self.postsource_supports_attached,
            self.outcome_payload_present,
            self.new_alpha_allocated,
            self.d_face_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        positive = (
            self.complete_ordered_action_axis,
            self.exact_population_measure_bound,
            self.registered_before_residual_outcomes,
        )
        if (
            self.direction_count != actions + shell
            or len(directions) != self.direction_count
            or len(self.shell_indices) != shell
            or len(set(self.shell_indices)) != shell
            or any(type(index) is not int or index < 0 or index >= routes for index in self.shell_indices)
            or any(observed[h] <= 0 or observed[h] > population[h] for h in (0, 1))
            or self.simultaneous_statement_count != 2 * self.direction_count * (routes + 1)
            or self.per_statement_two_sided_alpha
            != _divide_probability_down(self.familywise_alpha, self.simultaneous_statement_count)
            or self.support_derivation_mode not in {
                "production_route_bound_source_v1",
                "private_cpu_reference_support_v1",
            }
            or any(type(value) is not bool or value is not True for value in positive)
            or any(type(value) is not bool or value is not False for value in closed)
            or self.authority_scope != LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_AUTHORITY
            or self.schema != LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_PREOUTCOME_PLAN_SCHEMA
        ):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.AUTHORITY_VIOLATION,
                "joint residual PRE plan crossed its causal or authority boundary",
            )
        object.__setattr__(self, "joint_direction_ids", directions)
        object.__setattr__(self, "registered_observed_counts", observed)
        object.__setattr__(self, "residual_population_counts", population)
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {item.name: getattr(self, item.name) for item in fields(self) if not item.name.startswith("_")}
        )

    def _compute_hash(self) -> str:
        return _payload_hash(self._payload(), role="action-complete-preoutcome-plan")

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.MUTATED_CUSTODY,
                "joint residual PRE plan receipt no longer replays",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


class V1LiteralJJointAffineShellResidualPopulationStreamCUDAPreoutcomePlan:
    """Process-local causal owner for one action-complete residual execution."""

    __slots__ = (
        "receipt",
        "_action_axis",
        "_preregistration",
        "_population",
        "_logical_source",
        "_design",
        "_support_derivation_source",
        "_state",
        "_lock",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        receipt: V1LiteralJJointAffineShellResidualPopulationStreamCUDAPreoutcomePlanReceipt,
        action_axis: V1LiteralJJointPopulationActionAxisPreoutcomeSource,
        preregistration: V1LiteralJJointAffineShellPreoutcomeRegistration,
        population: _PopulationMeasureRegistration,
        logical_source: V1SampledV7SharedDW2LiteralJLogicalLookSource,
        design: V1LiteralJJointAffineShellResidualPopulationStreamCUDADesign,
        support_derivation_source: object,
        _issuance_token: object | None = None,
    ) -> None:
        if _issuance_token is not _PREOUTCOME_PLAN_OWNER_TOKEN:
            raise PermissionError("joint residual PRE plan owners require their issuer")
        self.receipt = receipt
        self._action_axis = action_axis
        self._preregistration = preregistration
        self._population = population
        self._logical_source = logical_source
        self._design = design
        self._support_derivation_source = support_derivation_source
        self._state = "preoutcome"
        self._lock = RLock()
        self.assert_preoutcome_live()

    @property
    def is_bound(self) -> bool:
        return self._state == "bound"

    @property
    def is_evicted(self) -> bool:
        return self._state == "evicted"

    def _replay_support_derivation(self) -> tuple[str, str]:
        source = self._support_derivation_source
        if self.receipt.support_derivation_mode == "production_route_bound_source_v1":
            if type(source) is not V1LiteralJResidualAnalyticSupportSourceCUDAResult:
                raise TypeError("production PRE support derivation source changed type")
            source.receipt.assert_integrity()
            source.source_handle.assert_live()
            return source.receipt.receipt_hash, source.receipt.literal_j_abs_bound_source_hash
        if type(source) is not _V1LiteralJJointResidualCPUAnalyticSupport:
            raise TypeError("private PRE support derivation source changed type")
        source.assert_live()
        return source.source_hash, source.source_hash

    def _replay_common(self, *, require_preoutcome: bool) -> str:
        self.receipt.assert_integrity()
        self._preregistration.assert_integrity()
        self._population.assert_integrity()
        self._design.assert_integrity()
        if require_preoutcome:
            self._action_axis.assert_preoutcome_live()
        else:
            self._action_axis.receipt.assert_integrity()
        try:
            source_hash = self._logical_source.assert_live()
        except Exception as error:
            raise V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusal(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.MUTATED_CUSTODY,
                "the PRE plan lost its exact logical source",
            ) from error
        support_hash, literal_hash = self._replay_support_derivation()
        if (
            source_hash != self.receipt.logical_look_source_receipt_hash
            or self._action_axis.receipt.receipt_hash
            != self.receipt.action_axis_preoutcome_receipt_hash
            or self._preregistration.registration_hash
            != self.receipt.joint_preoutcome_registration_hash
            or self._population.registration_hash
            != self.receipt.population_projection_registration_hash
            or self._design.design_hash != self.receipt.design_hash
            or support_hash != self.receipt.analytic_support_derivation_source_receipt_hash
            or literal_hash != self.receipt.literal_j_abs_bound_source_hash
            or (require_preoutcome and self._logical_source.is_bound)
        ):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.MUTATED_CUSTODY,
                "the action/source/measure/support axes changed after PRE issuance",
            )
        return self.receipt.receipt_hash

    def assert_preoutcome_live(self) -> str:
        with self._lock:
            if self._state == "evicted":
                _refuse(
                    V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.EVICTED,
                    "joint residual PRE plan was evicted",
                )
            if self._state != "preoutcome":
                _refuse(
                    V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.POSTOUTCOME_REGISTRATION,
                    "joint residual PRE plan was already bound",
                )
            return self._replay_common(require_preoutcome=True)

    def _assert_bound_replay(self) -> str:
        with self._lock:
            if self._state == "evicted":
                _refuse(
                    V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.EVICTED,
                    "joint residual PRE plan was evicted after binding",
                )
            if self._state != "bound" or not self._logical_source.is_bound:
                _refuse(
                    V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.SOURCE_DESCENT_MISMATCH,
                    "joint residual PRE plan has no exact POST descent",
                )
            return self._replay_common(require_preoutcome=False)

    def evict(self) -> None:
        with self._lock:
            self._state = "evicted"

    def __reduce__(self) -> NoReturn:
        raise TypeError("live joint residual PRE plans are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live joint residual PRE plans are nonserializable")


@dataclass(frozen=True, slots=True)
class V1LiteralJJointAffineShellResidualPopulationStreamCUDAPostsourceBindingReceipt:
    preoutcome_plan_receipt_hash: str
    action_axis_preoutcome_receipt_hash: str
    action_axis_postsource_receipt_hash: str
    logical_source_receipt_hash: str
    logical_result_receipt_hash: str
    joint_preoutcome_registration_hash: str
    joint_family_receipt_hash: str
    joint_postsource_custody_axis_hash: str
    affine_support_receipt_hashes: tuple[str, ...]
    route_support_source_receipt_hash: str
    literal_j_abs_bound_source_hash: str
    registration_hash: str
    physical_action_axis_hash: str
    joint_direction_axis_hash: str
    provider_outcomes_seen: bool = True
    exact_preoutcome_plan_descent: bool = True
    postsource_binding_registered_preoutcome: bool = False
    new_alpha_allocated: bool = False
    d_face_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_AUTHORITY
    schema: str = LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_POSTSOURCE_BINDING_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _POSTSOURCE_BINDING_RECEIPT_TOKEN:
            raise PermissionError("joint residual POST bindings require their binder")
        for name in (
            "preoutcome_plan_receipt_hash",
            "action_axis_preoutcome_receipt_hash",
            "action_axis_postsource_receipt_hash",
            "logical_source_receipt_hash",
            "logical_result_receipt_hash",
            "joint_preoutcome_registration_hash",
            "joint_family_receipt_hash",
            "joint_postsource_custody_axis_hash",
            "route_support_source_receipt_hash",
            "literal_j_abs_bound_source_hash",
            "registration_hash",
            "physical_action_axis_hash",
            "joint_direction_axis_hash",
        ):
            _identity(getattr(self, name), name=name)
        _identity_axis(self.affine_support_receipt_hashes, name="affine_support_receipt_hashes")
        closed = (
            self.postsource_binding_registered_preoutcome,
            self.new_alpha_allocated,
            self.d_face_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if (
            self.provider_outcomes_seen is not True
            or self.exact_preoutcome_plan_descent is not True
            or any(type(value) is not bool or value is not False for value in closed)
            or self.authority_scope != LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_AUTHORITY
            or self.schema != LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_POSTSOURCE_BINDING_SCHEMA
        ):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.AUTHORITY_VIOLATION,
                "joint residual POST binding crossed its causal or authority boundary",
            )
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {item.name: getattr(self, item.name) for item in fields(self) if not item.name.startswith("_")}
        )

    def _compute_hash(self) -> str:
        return _payload_hash(self._payload(), role="action-complete-postsource-binding")

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.MUTATED_CUSTODY,
                "joint residual POST binding receipt no longer replays",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


class V1LiteralJJointAffineShellResidualPopulationStreamCUDAPostsourceBinding:
    """Exact POST attachment for one causally frozen residual PRE plan."""

    __slots__ = (
        "receipt",
        "registration",
        "preoutcome_plan",
        "action_axis_source",
        "logical_result",
        "joint_family_handle",
        "affine_supports",
        "route_support_source",
        "private_support",
        "_state",
        "_lock",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        receipt: V1LiteralJJointAffineShellResidualPopulationStreamCUDAPostsourceBindingReceipt,
        registration: "V1LiteralJJointAffineShellResidualPopulationStreamCUDARegistration",
        preoutcome_plan: V1LiteralJJointAffineShellResidualPopulationStreamCUDAPreoutcomePlan,
        action_axis_source: V1LiteralJJointPopulationActionAxisSource,
        logical_result: V1SampledV7SharedDW2LiteralJLogicalLookResult,
        joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
        affine_supports: tuple[V1LiteralJAffineShellAnalyticSupportDeviceHandle, ...],
        route_support_source: V1LiteralJResidualAnalyticSupportSourceCUDAResult | None,
        private_support: object | None,
        _issuance_token: object | None = None,
    ) -> None:
        if _issuance_token is not _POSTSOURCE_BINDING_OWNER_TOKEN:
            raise PermissionError("joint residual POST binding owners require their binder")
        self.receipt = receipt
        self.registration = registration
        self.preoutcome_plan = preoutcome_plan
        self.action_axis_source = action_axis_source
        self.logical_result = logical_result
        self.joint_family_handle = joint_family_handle
        self.affine_supports = affine_supports
        self.route_support_source = route_support_source
        self.private_support = private_support
        self._state = "live"
        self._lock = RLock()
        self.assert_live()

    @property
    def is_evicted(self) -> bool:
        return self._state == "evicted"

    def assert_live(self) -> str:
        with self._lock:
            if self._state == "evicted":
                _refuse(
                    V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.EVICTED,
                    "joint residual POST binding was evicted",
                )
            try:
                self.receipt.assert_integrity()
                self.preoutcome_plan._assert_bound_replay()
                self.action_axis_source.assert_live()
                self.logical_result.assert_live()
                self.joint_family_handle.receipt.assert_integrity()
                self.registration.assert_integrity()
                if self.route_support_source is not None:
                    self.route_support_source.receipt.assert_integrity()
                    self.route_support_source.source_handle.assert_live()
                if self.private_support is not None:
                    self.private_support.assert_live()
                for support in self.affine_supports:
                    support.assert_live()
            except V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusal:
                raise
            except Exception as error:
                raise V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusal(
                    V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.MUTATED_CUSTODY,
                    "one exact POST binding parent is stale or mutated",
                ) from error
            if (
                self.receipt.registration_hash != self.registration.registration_hash
                or self.receipt.preoutcome_plan_receipt_hash
                != self.preoutcome_plan.receipt.receipt_hash
                or self.receipt.action_axis_postsource_receipt_hash
                != self.action_axis_source.receipt.receipt_hash
                or self.receipt.logical_result_receipt_hash
                != self.logical_result.receipt.receipt_hash
            ):
                _refuse(
                    V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.MUTATED_CUSTODY,
                    "joint residual POST binding lost one exact parent",
                )
            return self.receipt.receipt_hash

    def evict(self) -> None:
        with self._lock:
            self._state = "evicted"

    def __reduce__(self) -> NoReturn:
        raise TypeError("live joint residual POST bindings are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live joint residual POST bindings are nonserializable")


def _forecast_components(
    *,
    route_count: int,
    action_count: int,
    shell_size: int,
    observed_counts: tuple[int, int],
    route_tile_size: int,
    action_tile_size: int,
    include_compact_shell_m: bool = True,
) -> tuple[tuple[str, int], ...]:
    directions = action_count + shell_size
    contexts = sum(observed_counts)
    maximum_contexts = max(observed_counts)
    route_tile = min(route_count, route_tile_size)
    action_tile = min(action_count, action_tile_size)
    legacy = (
        ("compact_joint_projection_support_and_guard", 2 * directions * _F64_BYTES),
        ("pass1_projection_center_radius", 2 * contexts * directions * _F64_BYTES),
        ("pass1_absolute_mass_peak", maximum_contexts * directions * _F64_BYTES),
        ("q_center_radius_and_guards", 4 * 2 * directions * _F64_BYTES),
        ("source_tile_and_absolute", 2 * maximum_contexts * route_tile * _F64_BYTES),
        ("action_route_tile_and_absolute_transient", 2 * action_tile * route_tile * _F64_BYTES),
        ("action_projection_transients", 4 * maximum_contexts * action_tile * _F64_BYTES),
        ("mjv_tile_center_radius_and_guards", 4 * directions * route_tile * _F64_BYTES),
        ("interval_arithmetic_transients", 10 * directions * route_tile * _F64_BYTES),
    )
    if type(include_compact_shell_m) is not bool:
        raise TypeError("compact shell-M forecast selector must be an exact boolean")
    if not include_compact_shell_m:
        return legacy
    # Persistent center/radius custody plus its immutable mutation guards is
    # 2 strata x 2 interval tensors x 2 (live + guard) x D x B FP64 values.
    # The interval calculation uses at most twelve additional D x B FP64
    # matrices while one stratum is active.  This is deliberately explicit:
    # the compact Gram slice is small, but it is not hidden in external
    # workspace or charged after outcomes.
    return legacy + (
        (
            "pass1_compact_shell_m_center_radius_and_guards",
            8 * directions * shell_size * _F64_BYTES,
        ),
        (
            "pass1_compact_shell_m_gemm_interval_peak",
            12 * directions * shell_size * _F64_BYTES,
        ),
    )


@dataclass(frozen=True, slots=True)
class V1LiteralJJointAffineShellResidualPopulationStreamCUDAHBMForecast:
    design_hash: str
    joint_preoutcome_registration_hash: str
    joint_family_receipt_hash: str
    route_support_source_receipt_hash: str
    affine_support_receipt_hashes: tuple[str, ...]
    measurement_device_identity: str
    route_count: int
    action_count: int
    shell_size: int
    direction_count: int
    observed_counts: tuple[int, int]
    route_tile_size: int
    action_tile_size: int
    component_bytes: tuple[tuple[str, int], ...]
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
    includes_context_direction_projection: bool = True
    includes_single_direction_route_tile_and_guards: bool = True
    includes_action_tile_and_interval_temporaries: bool = True
    context_route_bank_retained: bool = False
    direction_route_bank_retained: bool = False
    route_scale_d2h: bool = False
    planning_only: bool = True
    production_authority: bool = False
    schema: str = LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_FORECAST_SCHEMA
    _admission_memory: InitVar[tuple[int, int] | None] = None
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _admission_memory: tuple[int, int] | None, _issuance_token: object | None) -> None:
        available, allocated = _admission_memory or (self.device_free_memory_bytes, self.current_reserved_device_bytes)
        if _issuance_token is not _FORECAST_TOKEN:
            raise PermissionError("joint residual-stream forecasts require their issuer")
        for name in (
            "design_hash",
            "joint_preoutcome_registration_hash",
            "joint_family_receipt_hash",
            "route_support_source_receipt_hash",
            "measurement_device_identity",
        ):
            _identity(getattr(self, name), name=name)
        supports = _identity_axis(self.affine_support_receipt_hashes, name="affine_support_receipt_hashes")
        routes = _positive_int(self.route_count, name="route_count")
        actions = _positive_int(self.action_count, name="action_count")
        shell = _positive_int(self.shell_size, name="shell_size")
        directions = _positive_int(self.direction_count, name="direction_count")
        counts = _count_pair(self.observed_counts, name="observed_counts")
        route_tile = _positive_int(self.route_tile_size, name="route_tile_size")
        action_tile = _positive_int(self.action_tile_size, name="action_tile_size")
        expected = _forecast_components(
            route_count=routes,
            action_count=actions,
            shell_size=shell,
            observed_counts=counts,
            route_tile_size=route_tile,
            action_tile_size=action_tile,
        )
        legacy_expected = _forecast_components(
            route_count=routes,
            action_count=actions,
            shell_size=shell,
            observed_counts=counts,
            route_tile_size=route_tile,
            action_tile_size=action_tile,
            include_compact_shell_m=False,
        )
        external = _nonnegative_int(
            self.registered_external_workspace_bytes,
            name="registered_external_workspace_bytes",
        )
        supplied_components = tuple(self.component_bytes)
        if supplied_components == expected:
            selected_components = expected
        elif supplied_components == legacy_expected:
            # Historical v1 forecast receipts remain replay-readable.  New
            # issuers always select ``expected`` and therefore fund compact
            # shell-M custody before any outcome is observed.
            selected_components = legacy_expected
        else:
            selected_components = expected
        incremental = sum(value for _name, value in selected_components) + external
        reserved = _nonnegative_int(self.current_reserved_device_bytes, name="current_reserved_device_bytes")
        total = _positive_int(self.device_total_memory_bytes, name="device_total_memory_bytes")
        free = _nonnegative_int(self.device_free_memory_bytes, name="device_free_memory_bytes")
        ceiling = math.floor(total * 0.85)
        projected = reserved + incremental
        positive = (
            self.includes_context_direction_projection,
            self.includes_single_direction_route_tile_and_guards,
            self.includes_action_tile_and_interval_temporaries,
            self.planning_only,
        )
        closed = (
            self.context_route_bank_retained,
            self.direction_route_bank_retained,
            self.route_scale_d2h,
            self.production_authority,
        )
        if (
            len(supports) != actions
            or directions != actions + shell
            or not 1 <= shell <= 8
            or supplied_components != selected_components
            or self.projected_incremental_peak_bytes != incremental
            or self.projected_reserved_device_bytes != projected
            or self.hbm_ceiling_fraction != 0.85
            or self.hbm_ceiling_bytes != ceiling
            or self.admitted_by_ceiling != (allocated + incremental <= ceiling)
            or self.admitted_by_free_memory != (incremental <= available)
            or self.admitted != (allocated + incremental <= ceiling and incremental <= available)
            or any(type(item) is not bool or item is not True for item in positive)
            or any(type(item) is not bool or item is not False for item in closed)
            or self.schema != LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_FORECAST_SCHEMA
        ):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.INVALID_DESIGN,
                "joint residual-stream HBM forecast does not replay exactly",
            )
        object.__setattr__(self, "affine_support_receipt_hashes", supports)
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {item.name: getattr(self, item.name) for item in fields(self) if not item.name.startswith("_")}
        )

    def _compute_hash(self) -> str:
        return _payload_hash(self._payload(), role="hbm-forecast")

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash or (expected_hash is not None and observed != expected_hash):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.MUTATED_CUSTODY,
                "joint residual-stream HBM forecast no longer replays",
            )
        return observed

    @property
    def forecast_hash(self) -> str:
        return self.assert_integrity()

    @property
    def includes_pass1_compact_shell_m(self) -> bool:
        """Whether this preallocation forecast funds the v1 compact output."""

        self.assert_integrity()
        return any(
            name == "pass1_compact_shell_m_center_radius_and_guards"
            for name, _bytes in self.component_bytes
        )


def _make_forecast(
    *,
    design: V1LiteralJJointAffineShellResidualPopulationStreamCUDADesign,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    affine_support_receipt_hashes: tuple[str, ...],
    route_support_source_receipt_hash: str,
    observed_counts: tuple[int, int],
    device: object,
    total_bytes: int,
    free_bytes: int,
    reserved_bytes: int,
    _admission_memory: tuple[int, int] | None = None,
) -> V1LiteralJJointAffineShellResidualPopulationStreamCUDAHBMForecast:
    design.assert_integrity()
    prereg = joint_preoutcome_registration
    prereg.assert_integrity()
    components = _forecast_components(
        route_count=prereg.route_count,
        action_count=prereg.action_count,
        shell_size=prereg.shell_size,
        observed_counts=observed_counts,
        route_tile_size=design.route_tile_size,
        action_tile_size=design.action_tile_size,
    )
    incremental = sum(value for _name, value in components) + design.registered_external_workspace_bytes
    total = _positive_int(total_bytes, name="device_total_memory_bytes")
    free = _nonnegative_int(free_bytes, name="device_free_memory_bytes")
    reserved = _nonnegative_int(reserved_bytes, name="current_reserved_device_bytes")
    ceiling = math.floor(total * 0.85)
    projected = reserved + incremental
    available, allocated = _admission_memory or (free, reserved)
    return V1LiteralJJointAffineShellResidualPopulationStreamCUDAHBMForecast(
        design_hash=design.design_hash,
        joint_preoutcome_registration_hash=prereg.registration_hash,
        joint_family_receipt_hash=joint_family_handle.receipt.receipt_hash,
        route_support_source_receipt_hash=route_support_source_receipt_hash,
        affine_support_receipt_hashes=affine_support_receipt_hashes,
        measurement_device_identity=_device_identity(device),
        route_count=prereg.route_count,
        action_count=prereg.action_count,
        shell_size=prereg.shell_size,
        direction_count=prereg.direction_count,
        observed_counts=observed_counts,
        route_tile_size=design.route_tile_size,
        action_tile_size=design.action_tile_size,
        component_bytes=components,
        registered_external_workspace_bytes=design.registered_external_workspace_bytes,
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


@dataclass(frozen=True, slots=True)
class V1LiteralJJointAffineShellResidualPopulationStreamCUDARegistration:
    """Outcome-independent joint residual estimator registration.

    The registration reuses the exact alpha and denominator already frozen by
    the joint family.  It allocates no new cell and binds target/complement in
    that order.  A historical single-action population registration is an
    exact typed parent, but cannot pass the joint-axis equality below.
    """

    joint_preoutcome_registration_hash: str
    joint_family_receipt_hash: str
    joint_postsource_custody_axis_hash: str
    population_projection_registration_hash: str
    suffix_registration_hash: str
    suffix_packet_hash: str
    suffix_prefix_freeze_hash: str
    row_group_plan_hash: str
    design_hash: str
    affine_support_receipt_hashes: tuple[str, ...]
    route_support_source_receipt_hash: str
    literal_j_abs_bound_source_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    literal_j_stream_source_hash: str
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    joint_direction_ids: tuple[str, ...]
    joint_direction_axis_hash: str
    action_count: int
    shell_size: int
    direction_count: int
    shell_indices: tuple[int, ...]
    stratum_ids: tuple[str, str]
    residual_population_counts: tuple[int, int]
    registered_observed_counts: tuple[int, int]
    residual_sealed_population_order_hashes: tuple[str, str]
    frozen_dw2_prefix_axis_hashes: tuple[str, str]
    frozen_dw2_pair_prefix_axis_hash: str
    scorer_packet_plan_hashes: tuple[str, str]
    scorer_provider_plan_hashes: tuple[str, str]
    scorer_context_axis_hashes: tuple[str, str]
    scorer_draw_axis_hashes: tuple[str, str]
    familywise_alpha: float
    simultaneous_statement_count: int
    per_statement_two_sided_alpha: float
    source_representation: str = "legacy_single_packet_pair_v1"
    logical_look_source_receipt_hash: str = ""
    generalized_logical_prefix_identity_hash: str = ""
    source_physical_call_offset: int = 0
    source_expected_prior_manifest_hash: str | None = None
    source_physical_roles: tuple[str, ...] = ("target", "complement")
    source_physical_context_counts: tuple[int, ...] = ()
    source_packet_plan_hashes: tuple[str, ...] = ()
    source_provider_plan_hashes: tuple[str, ...] = ()
    source_capture_identity_hashes: tuple[str, ...] = ()
    source_transport_packet_hashes: tuple[str, ...] = ()
    source_stratum_segment_ordinals: tuple[tuple[int, ...], tuple[int, ...]] = (
        (0,),
        (1,),
    )
    source_stratum_segment_counts: tuple[int, int] = (1, 1)
    joint_population_preoutcome_receipt_hash: str = ""
    causal_preoutcome_plan_receipt_hash: str = ""
    joint_population_action_axis_preoutcome_receipt_hash: str = ""
    joint_population_action_axis_postsource_receipt_hash: str = ""
    postsource_binding_registered_preoutcome: bool = False
    registered_before_residual_outcomes: bool = True
    exact_joint_action_shell_axis: bool = True
    exact_one_cell_denominator_reused: bool = True
    target_then_complement_order_frozen: bool = True
    mandatory_preoutcome_analytic_support: bool = True
    outcome_payload_present: bool = False
    new_alpha_allocated: bool = False
    adaptive_stopping_authorized: bool = False
    d_face_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    runtime_promotion_authority: bool = False
    pareto_authority: bool = False
    algorithm: str = LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_ALGORITHM
    authority_scope: str = LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_AUTHORITY
    schema: str = LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_REGISTRATION_SCHEMA
    _logical_look_source_owner: V1SampledV7SharedDW2LiteralJLogicalLookSource | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _joint_population_preoutcome_owner: V1LiteralJJointPopulationPreoutcomeSource | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _causal_preoutcome_plan_owner: V1LiteralJJointAffineShellResidualPopulationStreamCUDAPreoutcomePlan | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _joint_population_action_axis_postsource_owner: V1LiteralJJointPopulationActionAxisSource | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _REGISTRATION_TOKEN:
            raise PermissionError("joint residual-stream registrations require their issuer")
        for name in (
            "joint_preoutcome_registration_hash",
            "joint_family_receipt_hash",
            "joint_postsource_custody_axis_hash",
            "population_projection_registration_hash",
            "suffix_registration_hash",
            "suffix_packet_hash",
            "suffix_prefix_freeze_hash",
            "row_group_plan_hash",
            "design_hash",
            "route_support_source_receipt_hash",
            "literal_j_abs_bound_source_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "literal_j_stream_source_hash",
            "registered_d_prefix_hash",
            "registered_d_w2_prefix_hash",
            "joint_direction_axis_hash",
            "frozen_dw2_pair_prefix_axis_hash",
        ):
            _identity(getattr(self, name), name=name)
        supports = _identity_axis(self.affine_support_receipt_hashes, name="affine_support_receipt_hashes")
        directions = _identity_axis(self.joint_direction_ids, name="joint_direction_ids")
        actions = _positive_int(self.action_count, name="action_count")
        shell = _positive_int(self.shell_size, name="shell_size")
        direction_count = _positive_int(self.direction_count, name="direction_count")
        routes = _positive_int(self.route_count, name="route_count")
        _nonnegative_int(self.proposal_generation, name="proposal_generation")
        if (
            type(self.shell_indices) is not tuple
            or len(self.shell_indices) != shell
            or any(type(item) is not int or item < 0 or item >= routes for item in self.shell_indices)
            or len(set(self.shell_indices)) != shell
        ):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.INVALID_DESIGN,
                "registered shared shell is not one unique bounded route subset",
            )
        if type(self.stratum_ids) is not tuple or len(self.stratum_ids) != 2:
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.INVALID_TYPE,
                "stratum_ids must be one target/complement pair",
            )
        strata = tuple(_identity(item, name="stratum id") for item in self.stratum_ids)
        if strata != ("target", "complement"):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.ORDER_MISMATCH,
                "the registered estimator must preserve target then complement order",
            )
        populations = _count_pair(self.residual_population_counts, name="residual_population_counts")
        observed = _count_pair(self.registered_observed_counts, name="registered_observed_counts")
        alpha = _finite_alpha(self.familywise_alpha, name="familywise_alpha")
        statements = _positive_int(self.simultaneous_statement_count, name="simultaneous_statement_count")
        per_statement = _finite_alpha(self.per_statement_two_sided_alpha, name="per_statement_two_sided_alpha")
        two_axes = (
            self.residual_sealed_population_order_hashes,
            self.frozen_dw2_prefix_axis_hashes,
            self.scorer_packet_plan_hashes,
            self.scorer_provider_plan_hashes,
            self.scorer_context_axis_hashes,
            self.scorer_draw_axis_hashes,
        )
        if any(type(axis) is not tuple or len(axis) != 2 or any(type(item) is not str or not item for item in axis) for axis in two_axes):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.INVALID_TYPE,
                "paired suffix/scorer axes must each contain two exact identities",
            )
        source_representation = _identity(
            self.source_representation,
            name="source_representation",
        )
        if source_representation not in {
            "legacy_single_packet_pair_v1",
            "current_source_segmented_logical_look_v1",
            "current_source_segmented_joint_population_v1",
            "current_source_segmented_joint_population_action_axis_v2",
        }:
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.INVALID_DESIGN,
                "unknown literal-J residual source representation",
            )
        if (
            type(self.source_physical_roles) is not tuple
            or type(self.source_physical_context_counts) is not tuple
            or len(self.source_physical_roles)
            != len(self.source_physical_context_counts)
            or any(role not in ("target", "complement") for role in self.source_physical_roles)
            or any(
                type(count) is not int or count <= 0
                for count in self.source_physical_context_counts
            )
            or type(self.source_stratum_segment_ordinals) is not tuple
            or len(self.source_stratum_segment_ordinals) != 2
            or any(type(axis) is not tuple or not axis for axis in self.source_stratum_segment_ordinals)
            or self.source_stratum_segment_counts
            != tuple(len(axis) for axis in self.source_stratum_segment_ordinals)
            or tuple(
                index
                for axis in self.source_stratum_segment_ordinals
                for index in axis
            )
            != tuple(
                index
                for role in ("target", "complement")
                for index, observed_role in enumerate(self.source_physical_roles)
                if observed_role == role
            )
        ):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.ORDER_MISMATCH,
                "literal-J source segment axes do not replay target/complement grouping",
            )
        expected_observed = tuple(
            sum(
                self.source_physical_context_counts[index]
                for index in self.source_stratum_segment_ordinals[h]
            )
            for h in (0, 1)
        )
        if self.source_physical_context_counts and expected_observed != observed:
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.PREFIX_MISMATCH,
                "literal-J source segment counts do not equal the registered statistical strata",
            )
        if source_representation in {
            "current_source_segmented_logical_look_v1",
            "current_source_segmented_joint_population_v1",
            "current_source_segmented_joint_population_action_axis_v2",
        }:
            physical_count = len(self.source_physical_roles)
            source_axes = (
                self.source_packet_plan_hashes,
                self.source_provider_plan_hashes,
                self.source_capture_identity_hashes,
            )
            if (
                not self.logical_look_source_receipt_hash
                or not self.generalized_logical_prefix_identity_hash
                or any(
                    type(axis) is not tuple
                    or len(axis) != physical_count
                    or any(type(item) is not str or not item for item in axis)
                    for axis in source_axes
                )
                or type(self.source_transport_packet_hashes) is not tuple
                or not self.source_transport_packet_hashes
                or any(
                    type(item) is not str or not item
                    for item in self.source_transport_packet_hashes
                )
                or type(self.source_physical_call_offset) is not int
                or self.source_physical_call_offset < 0
                or type(self._logical_look_source_owner)
                is not V1SampledV7SharedDW2LiteralJLogicalLookSource
                or self._logical_look_source_owner.receipt.receipt_hash
                != self.logical_look_source_receipt_hash
            ):
                _refuse(
                    V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.PREFIX_MISMATCH,
                    "segmented logical-look registration is not source-complete",
                )
            if source_representation == "current_source_segmented_joint_population_v1":
                if (
                    not self.joint_population_preoutcome_receipt_hash
                    or type(self._joint_population_preoutcome_owner)
                    is not V1LiteralJJointPopulationPreoutcomeSource
                    or self._joint_population_preoutcome_owner.receipt.receipt_hash
                    != self.joint_population_preoutcome_receipt_hash
                    or self._joint_population_preoutcome_owner.receipt.logical_source_receipt_hash
                    != self.logical_look_source_receipt_hash
                    or self._joint_population_preoutcome_owner.receipt.joint_preoutcome_registration_hash
                    != self.joint_preoutcome_registration_hash
                    or self._joint_population_preoutcome_owner.receipt.joint_direction_axis_hash
                    != self.joint_direction_axis_hash
                    or self._joint_population_preoutcome_owner.receipt.literal_j_stream_source_hash
                    != self.literal_j_stream_source_hash
                    or self._joint_population_preoutcome_owner.receipt.route_count
                    != self.route_count
                ):
                    _refuse(
                        V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.PREFIX_MISMATCH,
                        "joint-population segmented registration lost its preoutcome source/action/D1 owner",
                    )
            elif source_representation == "current_source_segmented_joint_population_action_axis_v2":
                plan = self._causal_preoutcome_plan_owner
                post_axis = self._joint_population_action_axis_postsource_owner
                if (
                    not self.causal_preoutcome_plan_receipt_hash
                    or not self.joint_population_action_axis_preoutcome_receipt_hash
                    or not self.joint_population_action_axis_postsource_receipt_hash
                    or type(plan)
                    is not V1LiteralJJointAffineShellResidualPopulationStreamCUDAPreoutcomePlan
                    or type(post_axis)
                    is not V1LiteralJJointPopulationActionAxisSource
                    or plan.receipt.receipt_hash
                    != self.causal_preoutcome_plan_receipt_hash
                    or post_axis.preoutcome_source is not plan._action_axis
                    or post_axis.receipt.preoutcome_action_axis_receipt_hash
                    != self.joint_population_action_axis_preoutcome_receipt_hash
                    or post_axis.receipt.receipt_hash
                    != self.joint_population_action_axis_postsource_receipt_hash
                    or plan._logical_source is not self._logical_look_source_owner
                    or plan.receipt.logical_look_source_receipt_hash
                    != self.logical_look_source_receipt_hash
                    or plan.receipt.joint_preoutcome_registration_hash
                    != self.joint_preoutcome_registration_hash
                    or plan.receipt.joint_direction_axis_hash
                    != self.joint_direction_axis_hash
                    or plan.receipt.physical_action_axis_hash
                    != post_axis.receipt.physical_action_axis_hash
                    or self.joint_population_preoutcome_receipt_hash
                    or self._joint_population_preoutcome_owner is not None
                ):
                    _refuse(
                        V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.SOURCE_DESCENT_MISMATCH,
                        "v2 segmented registration lost its causal PRE plan or exact action-axis POST descent",
                    )
            elif (
                self.joint_population_preoutcome_receipt_hash
                or self._joint_population_preoutcome_owner is not None
                or self.causal_preoutcome_plan_receipt_hash
                or self.joint_population_action_axis_preoutcome_receipt_hash
                or self.joint_population_action_axis_postsource_receipt_hash
                or self._causal_preoutcome_plan_owner is not None
                or self._joint_population_action_axis_postsource_owner is not None
            ):
                _refuse(
                    V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.AUTHORITY_VIOLATION,
                    "logical-look segmented registration cannot impersonate joint-population custody",
                )
        elif (
            self.logical_look_source_receipt_hash
            or self.generalized_logical_prefix_identity_hash
            or self.source_expected_prior_manifest_hash is not None
            or self.source_physical_call_offset != 0
            or self.source_physical_roles != ("target", "complement")
            or self.source_stratum_segment_ordinals != ((0,), (1,))
            or self.source_stratum_segment_counts != (1, 1)
            or self._logical_look_source_owner is not None
            or self.joint_population_preoutcome_receipt_hash
            or self._joint_population_preoutcome_owner is not None
            or self.causal_preoutcome_plan_receipt_hash
            or self.joint_population_action_axis_preoutcome_receipt_hash
            or self.joint_population_action_axis_postsource_receipt_hash
            or self._causal_preoutcome_plan_owner is not None
            or self._joint_population_action_axis_postsource_owner is not None
        ):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.AUTHORITY_VIOLATION,
                "legacy pair registration cannot impersonate a segmented logical look",
            )
        positive = (
            self.exact_joint_action_shell_axis,
            self.exact_one_cell_denominator_reused,
            self.target_then_complement_order_frozen,
            self.mandatory_preoutcome_analytic_support,
        )
        closed = (
            self.outcome_payload_present,
            self.new_alpha_allocated,
            self.adaptive_stopping_authorized,
            self.d_face_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.runtime_promotion_authority,
            self.pareto_authority,
        )
        if (
            len(supports) != actions
            or direction_count != actions + shell
            or len(directions) != direction_count
            or not 1 <= shell <= 8
            or any(observed[h] <= 0 or observed[h] > populations[h] for h in (0, 1))
            or statements != 2 * direction_count * (routes + 1)
            or per_statement != _divide_probability_down(alpha, statements)
            or (
                self.registered_before_residual_outcomes
                is not (source_representation != "current_source_segmented_joint_population_action_axis_v2")
            )
            or self.postsource_binding_registered_preoutcome is not False
            or any(type(item) is not bool or item is not True for item in positive)
            or any(type(item) is not bool or item is not False for item in closed)
            or self.algorithm != LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_ALGORITHM
            or self.authority_scope != LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_AUTHORITY
            or self.schema != LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_REGISTRATION_SCHEMA
        ):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.AUTHORITY_VIOLATION,
                "joint residual-stream registration crossed its statistical or authority boundary",
            )
        object.__setattr__(self, "affine_support_receipt_hashes", supports)
        object.__setattr__(self, "joint_direction_ids", directions)
        object.__setattr__(self, "stratum_ids", strata)
        object.__setattr__(self, "residual_population_counts", populations)
        object.__setattr__(self, "registered_observed_counts", observed)
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {item.name: getattr(self, item.name) for item in fields(self) if not item.name.startswith("_")}
        )

    def _compute_hash(self) -> str:
        return _payload_hash(self._payload(), role="registration")

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        if self.source_representation == "current_source_segmented_logical_look_v1":
            if (
                type(self._logical_look_source_owner)
                is not V1SampledV7SharedDW2LiteralJLogicalLookSource
                or self._logical_look_source_owner.assert_live()
                != self.logical_look_source_receipt_hash
            ):
                _refuse(
                    V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.MUTATED_CUSTODY,
                    "segmented residual registration lost its exact preoutcome source owner",
                )
        if self.source_representation == "current_source_segmented_joint_population_v1":
            if (
                type(self._joint_population_preoutcome_owner)
                is not V1LiteralJJointPopulationPreoutcomeSource
                or self._joint_population_preoutcome_owner.is_evicted
                or self._joint_population_preoutcome_owner.receipt.receipt_hash
                != self.joint_population_preoutcome_receipt_hash
                or self._joint_population_preoutcome_owner.receipt.logical_source_receipt_hash
                != self.logical_look_source_receipt_hash
                or type(self._logical_look_source_owner)
                is not V1SampledV7SharedDW2LiteralJLogicalLookSource
                or self._logical_look_source_owner.assert_live()
                != self.logical_look_source_receipt_hash
            ):
                _refuse(
                    V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.MUTATED_CUSTODY,
                    "joint-population residual registration lost its exact preoutcome source owner",
                )
        if self.source_representation == "current_source_segmented_joint_population_action_axis_v2":
            plan = self._causal_preoutcome_plan_owner
            post_axis = self._joint_population_action_axis_postsource_owner
            if (
                type(plan)
                is not V1LiteralJJointAffineShellResidualPopulationStreamCUDAPreoutcomePlan
                or type(post_axis)
                is not V1LiteralJJointPopulationActionAxisSource
                or plan._assert_bound_replay()
                != self.causal_preoutcome_plan_receipt_hash
                or post_axis.assert_live()
                != self.joint_population_action_axis_postsource_receipt_hash
                or type(self._logical_look_source_owner)
                is not V1SampledV7SharedDW2LiteralJLogicalLookSource
                or not self._logical_look_source_owner.is_bound
                or self._logical_look_source_owner.receipt.receipt_hash
                != self.logical_look_source_receipt_hash
            ):
                _refuse(
                    V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.MUTATED_CUSTODY,
                    "v2 segmented registration lost its causal PRE/POST source custody",
                )
        observed = self._compute_hash()
        if observed != self._sealed_hash or (expected_hash is not None and observed != expected_hash):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.MUTATED_CUSTODY,
                "joint residual-stream registration no longer replays",
            )
        return observed

    @property
    def registration_hash(self) -> str:
        return self.assert_integrity()


def _assert_joint_population_axis(
    *,
    preregistration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    population: _PopulationMeasureRegistration,
) -> None:
    view = _population_measure_receipt(population)
    if type(view) is V1LiteralJJointAxisResidualPopulationMeasureRegistrationReceipt:
        axis_matches = (
            view.joint_preoutcome_registration_hash
            == preregistration.registration_hash
            and view.joint_direction_ids == preregistration.joint_direction_ids
            and view.joint_direction_axis_hash
            == preregistration.joint_direction_axis_hash
            and view.exact_complete_action_shell_axis is True
            and view.joint_axis_registration_authority is True
            and view.numerical_d1_projection_claim is False
            and view.resident_two_direction_registration_relabelled is False
        )
    else:
        # Coherent-legacy/private-reference registrations can still be read,
        # but canonical production enters through the separate measure-only
        # owner above.  Never infer equality between its resident two-vector
        # numerical axis and the joint action+shell axis.
        axis_matches = (
            view.direction_ids == preregistration.joint_direction_ids
            and view.d_fixed_projection_direction_axis_hash
            == preregistration.joint_direction_axis_hash
        )
    if not axis_matches:
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.JOINT_POPULATION_AXIS_REQUIRED,
            "the D/W2 population registration is not the preregistered action-complete "
            "literal-J family; a single-action/private-primal registration cannot be "
            "relabelled after outcomes",
        )


def _assert_common_parent_lineage(
    *,
    preregistration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    population: _PopulationMeasureRegistration,
) -> None:
    preregistration.assert_integrity()
    view = _population_measure_receipt(population)
    joint = joint_family_handle.receipt
    if (
        joint.joint_preoutcome_registration_hash != preregistration.registration_hash
        or joint.preoutcome_joint_direction_axis_hash != preregistration.joint_direction_axis_hash
        or joint.joint_direction_ids != preregistration.joint_direction_ids
        or joint.action_count != preregistration.action_count
        or joint.shell_size != preregistration.shell_size
        or joint.route_count != preregistration.route_count
        or view.model_identity_hash != preregistration.model_identity_hash
        or view.chart_identity_hash != preregistration.chart_identity_hash
        or view.current_mask_hash != preregistration.current_mask_hash
        or view.proposal_generation != preregistration.proposal_generation
        or view.route_axis_hash != preregistration.route_axis_hash
        or view.route_id_axis_hash != preregistration.route_id_axis_hash
        or view.route_count != preregistration.route_count
        or view.residual_literal_j_stream_source_hash
        != preregistration.literal_j_stream_source_hash
        or view.alpha_lineage_hash != preregistration.historical_alpha_lineage_hash
        or view.familywise_alpha != preregistration.familywise_alpha
    ):
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.STALE_IDENTITY,
            "joint family and D/W2 population do not close on one chart/J cell",
        )
    _assert_joint_population_axis(preregistration=preregistration, population=population)


def _validate_suffix_and_scorer_plan(
    *,
    preregistration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    population: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    suffix_registration: V1SampledVariableShellDW2SuffixRegistration,
    packet: V1SampledVariableShellDW2SuffixPacket,
    prefix_freeze: V1SampledVariableShellDW2SuffixPrefixFreezeReceipt,
    row_group_plan: V1SampledDualChartRowGroupPlan,
) -> tuple[tuple[int, int], tuple[object, object]]:
    suffix_registration.assert_integrity()
    packet.assert_integrity()
    prefix_freeze.assert_integrity()
    row_group_plan.assert_integrity()
    scorer = tuple(row_group_plan.packet_plans)
    observed = packet.after_suffix_stratum_counts
    expected_contexts = (packet.target_context_ids, packet.complement_context_ids)
    if (
        population.suffix_registration_hash != suffix_registration.receipt_hash
        or packet.suffix_registration_hash != suffix_registration.receipt_hash
        or prefix_freeze.suffix_registration_hash != suffix_registration.receipt_hash
        or prefix_freeze.completed_packet_hashes != (packet.receipt_hash,)
        or packet.packet_ordinal != 0
        or packet.before_suffix_stratum_counts != (0, 0)
        or prefix_freeze.consumed_dw2_stratum_counts != observed
        or observed != (len(packet.target_context_ids), len(packet.complement_context_ids))
        or any(observed[h] <= 0 or observed[h] > population.residual_stratum_counts[h] for h in (0, 1))
        or prefix_freeze.generation_index != preregistration.proposal_generation
        or prefix_freeze.population_stratum_counts != suffix_registration.population_stratum_counts
        or len(scorer) != 2
        or tuple(item.packet_ordinal for item in scorer) != (0, 1)
        or tuple(item.context_ids for item in scorer) != expected_contexts
        or tuple(item.stratum_labels for item in scorer)
        != (("target",) * observed[0], ("complement",) * observed[1])
        or row_group_plan.first_prefix_look_ordinal != 0
        or row_group_plan.expected_prior_manifest_hash is not None
        or row_group_plan.chart_identity_hash != preregistration.chart_identity_hash
        or row_group_plan.current_mask_hash != preregistration.current_mask_hash
        or row_group_plan.proposal_generation != preregistration.proposal_generation
        or row_group_plan.route_axis_hash != preregistration.route_axis_hash
        or row_group_plan.route_id_axis_hash != preregistration.route_id_axis_hash
        or row_group_plan.route_count != preregistration.route_count
        or row_group_plan.literal_j_stream_identity_hash != preregistration.literal_j_stream_source_hash
    ):
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.PREFIX_MISMATCH,
            "target/complement D/W2 suffix and preregistered scorer plan diverged",
        )
    return observed, (scorer[0], scorer[1])


def _validate_production_analytic_supports(
    *,
    preregistration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    affine_supports: tuple[V1LiteralJAffineShellAnalyticSupportDeviceHandle, ...],
    route_support_source: V1LiteralJResidualAnalyticSupportSourceCUDAResult,
) -> object:
    if (
        type(affine_supports) is not tuple
        or len(affine_supports) != preregistration.action_count
        or any(
            type(item) is not V1LiteralJAffineShellAnalyticSupportDeviceHandle
            for item in affine_supports
        )
        or type(route_support_source)
        is not V1LiteralJResidualAnalyticSupportSourceCUDAResult
    ):
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.ANALYTIC_SUPPORT_REQUIRED,
            "production requires exactly one canonical affine analytic-support "
            "handle per physical action plus the exact route-bound source",
        )
    route_support_source.receipt.assert_integrity()
    route_support_source.source_handle.assert_live()
    joint = joint_family_handle.receipt
    receipts = tuple(item.receipt for item in affine_supports)
    for receipt in receipts:
        receipt.assert_integrity()
    common = (
        "model_identity_hash",
        "chart_identity_hash",
        "current_mask_hash",
        "proposal_generation",
        "route_axis_hash",
        "route_id_axis_hash",
        "route_count",
        "shell_freeze_receipt_hash",
        "shell_indices",
        "shell_route_ids",
        "literal_j_stream_source_hash",
        "literal_j_abs_bound_source_hash",
    )
    first = receipts[0]
    if any(any(getattr(item, name) != getattr(first, name) for name in common) for item in receipts):
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.ANALYTIC_SUPPORT_REQUIRED,
            "per-action analytic supports do not share one chart, shell, and route bound",
        )
    if (
        tuple(item.affine_direction_receipt_hash for item in receipts)
        != joint.direction_receipt_hashes
        or tuple(item.affine_direction_preoutcome_source_registration_hash for item in receipts)
        != preregistration.preoutcome_registration_hashes
        or any(item.execution_mode != "cuda_production" for item in receipts)
        or first.model_identity_hash != preregistration.model_identity_hash
        or first.chart_identity_hash != preregistration.chart_identity_hash
        or first.current_mask_hash != preregistration.current_mask_hash
        or first.proposal_generation != preregistration.proposal_generation
        or first.route_axis_hash != preregistration.route_axis_hash
        or first.route_id_axis_hash != preregistration.route_id_axis_hash
        or first.route_count != preregistration.route_count
        or first.shell_freeze_receipt_hash != preregistration.shell_freeze_receipt_hash
        or first.shell_indices != preregistration.shell_indices
        or first.shell_route_ids != preregistration.shell_route_ids
        or first.literal_j_stream_source_hash != preregistration.literal_j_stream_source_hash
        or any(
            item.generic_support_source_receipt_hash
            != route_support_source.receipt.receipt_hash
            for item in receipts
        )
    ):
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.STALE_IDENTITY,
            "analytic support tuple is not action-by-action custody for this joint family",
        )
    route = route_support_source.receipt
    if (
        route.model_identity_hash != preregistration.model_identity_hash
        or route.chart_identity_hash != preregistration.chart_identity_hash
        or route.current_mask_hash != preregistration.current_mask_hash
        or route.proposal_generation != preregistration.proposal_generation
        or route.route_axis_hash != preregistration.route_axis_hash
        or route.route_id_axis_hash != preregistration.route_id_axis_hash
        or route.route_count != preregistration.route_count
        or route.literal_j_abs_bound_source_hash != first.literal_j_abs_bound_source_hash
    ):
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.STALE_IDENTITY,
            "route analytic support source left the registered chart or bound derivation",
        )
    if torch is None or not torch.cuda.is_available():
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.DEVICE_REQUIRED,
            "production joint residual projection requires a real CUDA device",
        )
    devices: list[object] = [item.device for item in affine_supports]
    with joint_family_handle.borrow() as lease:
        outside, shell = lease.projection_custody()
        devices.extend(item.device for item in outside)
        devices.append(shell.device)
    with route_support_source.source_handle.borrow() as lease:
        devices.append(lease.literal_j_abs_bound.device)
    reference = devices[0]
    if reference.type != "cuda" or any(item != reference for item in devices[1:]):
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.DEVICE_REQUIRED,
            "every direction, shared shell, and analytic bound must occupy the same exact CUDA ordinal",
        )
    return reference


def register_v1_literal_j_joint_affine_shell_residual_population_stream_cuda(
    *,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    suffix_registration: V1SampledVariableShellDW2SuffixRegistration,
    packet: V1SampledVariableShellDW2SuffixPacket,
    prefix_freeze: V1SampledVariableShellDW2SuffixPrefixFreezeReceipt,
    row_group_plan: V1SampledDualChartRowGroupPlan,
    affine_supports: tuple[V1LiteralJAffineShellAnalyticSupportDeviceHandle, ...],
    route_support_source: V1LiteralJResidualAnalyticSupportSourceCUDAResult,
    design: V1LiteralJJointAffineShellResidualPopulationStreamCUDADesign,
) -> V1LiteralJJointAffineShellResidualPopulationStreamCUDARegistration:
    """Compatibility registration for the historical two-shard source."""

    exact = (
        (joint_preoutcome_registration, V1LiteralJJointAffineShellPreoutcomeRegistration),
        (joint_family_handle, V1LiteralJJointAffineShellFamilyDeviceHandle),
        (population_registration, V1LiteralJDFixedResidualPopulationProjectionRegistration),
        (suffix_registration, V1SampledVariableShellDW2SuffixRegistration),
        (packet, V1SampledVariableShellDW2SuffixPacket),
        (prefix_freeze, V1SampledVariableShellDW2SuffixPrefixFreezeReceipt),
        (row_group_plan, V1SampledDualChartRowGroupPlan),
        (design, V1LiteralJJointAffineShellResidualPopulationStreamCUDADesign),
    )
    if any(type(value) is not owner for value, owner in exact):
        raise TypeError("joint residual-stream registration received an inexact owner")
    design.assert_integrity()
    _assert_common_parent_lineage(
        preregistration=joint_preoutcome_registration,
        joint_family_handle=joint_family_handle,
        population=population_registration,
    )
    observed, scorer = _validate_suffix_and_scorer_plan(
        preregistration=joint_preoutcome_registration,
        population=population_registration,
        suffix_registration=suffix_registration,
        packet=packet,
        prefix_freeze=prefix_freeze,
        row_group_plan=row_group_plan,
    )
    _validate_production_analytic_supports(
        preregistration=joint_preoutcome_registration,
        joint_family_handle=joint_family_handle,
        affine_supports=affine_supports,
        route_support_source=route_support_source,
    )
    prereg = joint_preoutcome_registration
    joint = joint_family_handle.receipt
    return V1LiteralJJointAffineShellResidualPopulationStreamCUDARegistration(
        joint_preoutcome_registration_hash=prereg.registration_hash,
        joint_family_receipt_hash=joint.receipt_hash,
        joint_postsource_custody_axis_hash=joint.postsource_custody_axis_hash,
        population_projection_registration_hash=population_registration.registration_hash,
        suffix_registration_hash=suffix_registration.receipt_hash,
        suffix_packet_hash=packet.receipt_hash,
        suffix_prefix_freeze_hash=prefix_freeze.receipt_hash,
        row_group_plan_hash=row_group_plan.group_plan_hash,
        design_hash=design.design_hash,
        affine_support_receipt_hashes=tuple(item.receipt.receipt_hash for item in affine_supports),
        route_support_source_receipt_hash=route_support_source.receipt.receipt_hash,
        literal_j_abs_bound_source_hash=route_support_source.receipt.literal_j_abs_bound_source_hash,
        model_identity_hash=prereg.model_identity_hash,
        chart_identity_hash=prereg.chart_identity_hash,
        current_mask_hash=prereg.current_mask_hash,
        proposal_generation=prereg.proposal_generation,
        route_axis_hash=prereg.route_axis_hash,
        route_id_axis_hash=prereg.route_id_axis_hash,
        route_count=prereg.route_count,
        literal_j_stream_source_hash=prereg.literal_j_stream_source_hash,
        registered_d_prefix_hash=prereg.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=prereg.registered_d_w2_prefix_hash,
        joint_direction_ids=prereg.joint_direction_ids,
        joint_direction_axis_hash=prereg.joint_direction_axis_hash,
        action_count=prereg.action_count,
        shell_size=prereg.shell_size,
        direction_count=prereg.direction_count,
        shell_indices=prereg.shell_indices,
        stratum_ids=("target", "complement"),
        residual_population_counts=population_registration.residual_stratum_counts,
        registered_observed_counts=observed,
        residual_sealed_population_order_hashes=population_registration.residual_sealed_population_order_hashes,
        frozen_dw2_prefix_axis_hashes=(
            prefix_freeze.dw2_target_prefix_axis_hash,
            prefix_freeze.dw2_complement_prefix_axis_hash,
        ),
        frozen_dw2_pair_prefix_axis_hash=prefix_freeze.dw2_pair_prefix_axis_hash,
        scorer_packet_plan_hashes=(scorer[0].packet_plan_hash, scorer[1].packet_plan_hash),
        scorer_provider_plan_hashes=(
            scorer[0].provider_plan_identity_hash,
            scorer[1].provider_plan_identity_hash,
        ),
        scorer_context_axis_hashes=(
            _axis_hash("target-contexts", scorer[0].context_ids),
            _axis_hash("complement-contexts", scorer[1].context_ids),
        ),
        scorer_draw_axis_hashes=(
            _axis_hash("target-draws", scorer[0].draw_ids),
            _axis_hash("complement-draws", scorer[1].draw_ids),
        ),
        familywise_alpha=prereg.familywise_alpha,
        simultaneous_statement_count=prereg.simultaneous_statement_count,
        per_statement_two_sided_alpha=prereg.per_statement_two_sided_alpha,
        source_physical_context_counts=observed,
        source_packet_plan_hashes=(
            scorer[0].packet_plan_hash,
            scorer[1].packet_plan_hash,
        ),
        source_provider_plan_hashes=(
            scorer[0].provider_plan_identity_hash,
            scorer[1].provider_plan_identity_hash,
        ),
        source_capture_identity_hashes=(
            scorer[0].source_capture_identity_hash,
            scorer[1].source_capture_identity_hash,
        ),
        source_transport_packet_hashes=(packet.receipt_hash,),
        _issuance_token=_REGISTRATION_TOKEN,
    )


def _segmented_source_preoutcome_axes(
    *,
    source: V1SampledV7SharedDW2LiteralJLogicalLookSource,
    preregistration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    population: _PopulationMeasureRegistration,
) -> tuple[
    tuple[int, int],
    tuple[object, ...],
    tuple[tuple[int, ...], tuple[int, ...]],
    tuple[tuple[str, ...], tuple[str, ...]],
    tuple[tuple[str, ...], tuple[str, ...]],
]:
    """Validate and project an outcome-independent logical look onto two strata."""

    if type(source) is not V1SampledV7SharedDW2LiteralJLogicalLookSource:
        raise TypeError("segmented residual registration requires its exact source")
    view = _population_measure_receipt(population)
    residual_counts = _population_residual_counts(population)
    source_hash = source.assert_live()
    if source.is_bound:
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.POSTOUTCOME_REGISTRATION,
            "segmented residual geometry must be registered before the logical-look source is bound to provider outcomes",
        )
    receipt = source.receipt
    group_plan = source.coordinator.acquisition_plan.group_plan
    group_plan.assert_integrity()
    plans = tuple(group_plan.packet_plans)
    roles = tuple(receipt.physical_roles)
    counts = tuple(receipt.physical_context_counts)
    if (
        source_hash != receipt.receipt_hash
        or receipt.provider_outcomes_seen is not False
        or receipt.chart_identity_hash != preregistration.chart_identity_hash
        or receipt.current_mask_hash != preregistration.current_mask_hash
        or receipt.proposal_generation != preregistration.proposal_generation
        or receipt.route_axis_hash != preregistration.route_axis_hash
        or receipt.route_id_axis_hash != preregistration.route_id_axis_hash
        or receipt.route_count != preregistration.route_count
        or receipt.literal_j_stream_identity_hash
        != preregistration.literal_j_stream_source_hash
        or receipt.raw_row_prefix_identity_hash
        != preregistration.registered_d_prefix_hash
        or receipt.suffix_registration_hash != view.suffix_registration_hash
        or receipt.row_group_plan_hash != group_plan.group_plan_hash
        or receipt.row_group_packet_plan_hashes
        != tuple(plan.packet_plan_hash for plan in plans)
        or len(plans) != len(roles)
        or len(counts) != len(roles)
        or tuple(plan.stratum_labels[0] for plan in plans) != roles
        or tuple(len(plan.context_ids) for plan in plans) != counts
        or tuple(plan.source_capture_identity_hash for plan in plans)
        != receipt.source_capture_identity_hashes
    ):
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.PREFIX_MISMATCH,
            "segmented literal-J logical look diverged from its preregistered chart/population",
        )
    ordinals = tuple(
        tuple(index for index, observed_role in enumerate(roles) if observed_role == role)
        for role in ("target", "complement")
    )
    if any(not axis for axis in ordinals):
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.ORDER_MISMATCH,
            "segmented logical look must contain both statistical strata",
        )
    context_axes = tuple(
        tuple(
            context
            for index in ordinals[h]
            for context in plans[index].context_ids
        )
        for h in (0, 1)
    )
    draw_axes = tuple(
        tuple(
            draw
            for index in ordinals[h]
            for draw in plans[index].draw_ids
        )
        for h in (0, 1)
    )
    observed = (len(context_axes[0]), len(context_axes[1]))
    if any(
        observed[h] <= 0
        or observed[h] > residual_counts[h]
        for h in (0, 1)
    ):
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.PREFIX_MISMATCH,
            "segmented logical-look sample mass exceeds its registered residual population",
        )
    return observed, plans, ordinals, context_axes, draw_axes  # type: ignore[return-value]


def _validate_action_axis_preoutcome_lineage(
    *,
    action_axis: V1LiteralJJointPopulationActionAxisPreoutcomeSource,
    preregistration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    population: _PopulationMeasureRegistration,
) -> V1SampledV7SharedDW2LiteralJLogicalLookSource:
    if (
        type(action_axis)
        is not V1LiteralJJointPopulationActionAxisPreoutcomeSource
        or type(preregistration)
        is not V1LiteralJJointAffineShellPreoutcomeRegistration
        or type(population)
        not in {
            V1LiteralJDFixedResidualPopulationProjectionRegistration,
            V1LiteralJJointAxisResidualPopulationMeasureRegistration,
        }
    ):
        raise TypeError("v2 residual PRE plan requires exact action-axis parents")
    try:
        action_axis.assert_preoutcome_live()
    except Exception as error:
        raise V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusal(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.POSTOUTCOME_REGISTRATION,
            "the action-complete source axis was not live before residual outcomes",
        ) from error
    preregistration.assert_integrity()
    view = _population_measure_receipt(population)
    axis = action_axis.receipt
    sources = action_axis._sources
    if (
        type(sources) is not tuple
        or len(sources) != axis.action_count
        or not sources
        or any(source._logical_source is not sources[0]._logical_source for source in sources)
    ):
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.ACTION_AXIS_REQUIRED,
            "the action-complete PRE aggregate does not retain one exact residual logical source",
        )
    logical_source = sources[0]._logical_source
    if type(logical_source) is not V1SampledV7SharedDW2LiteralJLogicalLookSource:
        raise TypeError("v2 residual PRE action axis retained an inexact logical source")
    if (
        axis.joint_preoutcome_registration_hash != preregistration.registration_hash
        or axis.joint_direction_axis_hash != preregistration.joint_direction_axis_hash
        or axis.joint_direction_ids != preregistration.joint_direction_ids
        or axis.physical_action_axis_hash != preregistration.physical_action_axis_hash
        or axis.action_count != preregistration.action_count
        or axis.shell_size != preregistration.shell_size
        or axis.direction_count != preregistration.direction_count
        or axis.shell_indices != preregistration.shell_indices
        or axis.model_identity_hash != preregistration.model_identity_hash
        or axis.chart_identity_hash != preregistration.chart_identity_hash
        or axis.current_mask_hash != preregistration.current_mask_hash
        or axis.proposal_generation != preregistration.proposal_generation
        or axis.route_axis_hash != preregistration.route_axis_hash
        or axis.route_id_axis_hash != preregistration.route_id_axis_hash
        or axis.route_count != preregistration.route_count
        or axis.literal_j_stream_source_hash
        != preregistration.literal_j_stream_source_hash
        or axis.registered_d_prefix_hash != preregistration.registered_d_prefix_hash
        or axis.registered_d_w2_prefix_hash
        != preregistration.registered_d_w2_prefix_hash
        or axis.alpha_lineage_hash != preregistration.historical_alpha_lineage_hash
        or view.population_measure_design_hash != axis.measure_design_hash
        or view.population_stratum_counts != axis.population_stratum_counts
        or view.residual_literal_j_stream_source_hash
        != axis.literal_j_stream_source_hash
        or view.alpha_lineage_hash != axis.alpha_lineage_hash
        or view.familywise_alpha != preregistration.familywise_alpha
    ):
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.PREFIX_MISMATCH,
            "the action axis, exact population measure, and residual J cell do not close",
        )
    _assert_joint_population_axis(preregistration=preregistration, population=population)
    return logical_source


def _preoutcome_support_derivation_axes(
    *,
    preregistration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    action_axis: V1LiteralJJointPopulationActionAxisPreoutcomeSource,
    route_support_source: V1LiteralJResidualAnalyticSupportSourceCUDAResult | None,
    private_support: object | None,
) -> tuple[str, str, str, object]:
    if route_support_source is not None:
        if (
            type(route_support_source)
            is not V1LiteralJResidualAnalyticSupportSourceCUDAResult
            or private_support is not None
        ):
            raise TypeError("production PRE support derivation requires one exact route-bound source")
        route_support_source.receipt.assert_integrity()
        route_support_source.source_handle.assert_live()
        support = route_support_source.receipt
        if (
            support.model_identity_hash != preregistration.model_identity_hash
            or support.chart_identity_hash != preregistration.chart_identity_hash
            or support.current_mask_hash != preregistration.current_mask_hash
            or support.proposal_generation != preregistration.proposal_generation
            or support.route_axis_hash != preregistration.route_axis_hash
            or support.route_id_axis_hash != preregistration.route_id_axis_hash
            or support.route_count != preregistration.route_count
        ):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.STALE_IDENTITY,
                "the PRE route-bound derivation source belongs to another chart",
            )
        source_hash = support.receipt_hash
        literal_hash = support.literal_j_abs_bound_source_hash
        mode = "production_route_bound_source_v1"
        owner: object = route_support_source
    else:
        if type(private_support) is not _V1LiteralJJointResidualCPUAnalyticSupport:
            raise TypeError("private PRE support derivation requires its exact sealed support")
        private_support.assert_live()
        if private_support.joint_preoutcome_registration_hash != preregistration.registration_hash:
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.STALE_IDENTITY,
                "private PRE support derivation belongs to another joint registration",
            )
        source_hash = private_support.source_hash
        literal_hash = private_support.source_hash
        mode = "private_cpu_reference_support_v1"
        owner = private_support
    derivation_hash = _payload_hash(
        {
            "support_source_receipt_hash": source_hash,
            "literal_j_abs_bound_source_hash": literal_hash,
            "action_axis_preoutcome_receipt_hash": action_axis.receipt.receipt_hash,
            "physical_action_axis_hash": action_axis.receipt.physical_action_axis_hash,
            "packed_action_hashes": action_axis.receipt.packed_action_hashes,
            "joint_direction_axis_hash": action_axis.receipt.joint_direction_axis_hash,
            "shell_indices": preregistration.shell_indices,
            "derivation_mode": mode,
            "residual_outcomes_read": False,
        },
        role="preoutcome-analytic-support-derivation-inputs",
    )
    return derivation_hash, source_hash, literal_hash, owner


def _register_v1_literal_j_joint_affine_shell_residual_population_stream_preoutcome_plan(
    *,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    action_axis_preoutcome_source: V1LiteralJJointPopulationActionAxisPreoutcomeSource,
    population_registration: _PopulationMeasureRegistration,
    route_support_source: V1LiteralJResidualAnalyticSupportSourceCUDAResult | None,
    private_support: object | None,
    design: V1LiteralJJointAffineShellResidualPopulationStreamCUDADesign,
) -> V1LiteralJJointAffineShellResidualPopulationStreamCUDAPreoutcomePlan:
    if type(design) is not V1LiteralJJointAffineShellResidualPopulationStreamCUDADesign:
        raise TypeError("v2 residual PRE plan requires its exact design")
    design.assert_integrity()
    logical_source = _validate_action_axis_preoutcome_lineage(
        action_axis=action_axis_preoutcome_source,
        preregistration=joint_preoutcome_registration,
        population=population_registration,
    )
    observed, plans, ordinals, contexts, draws = _segmented_source_preoutcome_axes(
        source=logical_source,
        preregistration=joint_preoutcome_registration,
        population=population_registration,
    )
    derivation_hash, support_hash, literal_hash, support_owner = (
        _preoutcome_support_derivation_axes(
            preregistration=joint_preoutcome_registration,
            action_axis=action_axis_preoutcome_source,
            route_support_source=route_support_source,
            private_support=private_support,
        )
    )
    axis = action_axis_preoutcome_source.receipt
    source_receipt = logical_source.receipt
    receipt = V1LiteralJJointAffineShellResidualPopulationStreamCUDAPreoutcomePlanReceipt(
        action_axis_preoutcome_receipt_hash=axis.receipt_hash,
        joint_preoutcome_registration_hash=joint_preoutcome_registration.registration_hash,
        physical_action_axis_hash=axis.physical_action_axis_hash,
        joint_direction_axis_hash=axis.joint_direction_axis_hash,
        joint_direction_ids=axis.joint_direction_ids,
        logical_look_source_receipt_hash=source_receipt.receipt_hash,
        population_projection_registration_hash=population_registration.registration_hash,
        population_measure_design_hash=axis.measure_design_hash,
        population_measure_weight_receipt_hash=axis.measure_weight_receipt_hash,
        design_hash=design.design_hash,
        analytic_support_derivation_input_hash=derivation_hash,
        analytic_support_derivation_source_receipt_hash=support_hash,
        literal_j_abs_bound_source_hash=literal_hash,
        model_identity_hash=axis.model_identity_hash,
        chart_identity_hash=axis.chart_identity_hash,
        current_mask_hash=axis.current_mask_hash,
        proposal_generation=axis.proposal_generation,
        route_axis_hash=axis.route_axis_hash,
        route_id_axis_hash=axis.route_id_axis_hash,
        route_count=axis.route_count,
        literal_j_stream_source_hash=axis.literal_j_stream_source_hash,
        registered_d_prefix_hash=axis.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=axis.registered_d_w2_prefix_hash,
        suffix_registration_hash=source_receipt.suffix_registration_hash,
        shell_indices=axis.shell_indices,
        action_count=axis.action_count,
        shell_size=axis.shell_size,
        direction_count=axis.direction_count,
        residual_population_counts=_population_residual_counts(population_registration),
        registered_observed_counts=observed,
        residual_sealed_population_order_hashes=(
            _population_measure_receipt(
                population_registration
            ).residual_sealed_population_order_hashes
        ),
        generalized_logical_prefix_identity_hash=(
            source_receipt.generalized_logical_prefix_identity_hash
        ),
        row_group_plan_hash=source_receipt.row_group_plan_hash,
        source_physical_call_offset=source_receipt.physical_call_offset,
        source_expected_prior_manifest_hash=source_receipt.expected_prior_manifest_hash,
        source_physical_roles=source_receipt.physical_roles,
        source_physical_context_counts=source_receipt.physical_context_counts,
        source_packet_plan_hashes=tuple(plan.packet_plan_hash for plan in plans),
        source_provider_plan_hashes=tuple(plan.provider_plan_identity_hash for plan in plans),
        source_capture_identity_hashes=source_receipt.source_capture_identity_hashes,
        source_transport_packet_hashes=source_receipt.transport_packet_hashes,
        source_stratum_segment_ordinals=(ordinals[0], ordinals[1]),
        scorer_context_axis_hashes=(
            _axis_hash("target-contexts", contexts[0]),
            _axis_hash("complement-contexts", contexts[1]),
        ),
        scorer_draw_axis_hashes=(
            _axis_hash("target-draws", draws[0]),
            _axis_hash("complement-draws", draws[1]),
        ),
        familywise_alpha=joint_preoutcome_registration.familywise_alpha,
        simultaneous_statement_count=(
            joint_preoutcome_registration.simultaneous_statement_count
        ),
        per_statement_two_sided_alpha=(
            joint_preoutcome_registration.per_statement_two_sided_alpha
        ),
        d1_context_weight_fraction=axis.d1_context_weight_fraction,
        residual_context_weight_fractions=axis.residual_context_weight_fractions,
        residual_stratum_mass_fractions=axis.residual_stratum_mass_fractions,
        residual_population_mass_fraction=axis.residual_population_mass_fraction,
        common_measure_denominator=axis.common_measure_denominator,
        support_derivation_mode=(
            "production_route_bound_source_v1"
            if route_support_source is not None
            else "private_cpu_reference_support_v1"
        ),
        _issuance_token=_PREOUTCOME_PLAN_RECEIPT_TOKEN,
    )
    return V1LiteralJJointAffineShellResidualPopulationStreamCUDAPreoutcomePlan(
        receipt=receipt,
        action_axis=action_axis_preoutcome_source,
        preregistration=joint_preoutcome_registration,
        population=population_registration,
        logical_source=logical_source,
        design=design,
        support_derivation_source=support_owner,
        _issuance_token=_PREOUTCOME_PLAN_OWNER_TOKEN,
    )


def register_v1_literal_j_joint_affine_shell_residual_population_stream_cuda_preoutcome_plan(
    *,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    action_axis_preoutcome_source: V1LiteralJJointPopulationActionAxisPreoutcomeSource,
    population_registration: V1LiteralJJointAxisResidualPopulationMeasureRegistration,
    route_support_source: V1LiteralJResidualAnalyticSupportSourceCUDAResult,
    design: V1LiteralJJointAffineShellResidualPopulationStreamCUDADesign,
) -> V1LiteralJJointAffineShellResidualPopulationStreamCUDAPreoutcomePlan:
    """Freeze the production v2 residual plan before any D2 outcome exists."""

    if (
        type(population_registration)
        is not V1LiteralJJointAxisResidualPopulationMeasureRegistration
    ):
        raise TypeError(
            "production v2 residual PRE requires the exact measure-only "
            "action+shell population registration"
        )

    return _register_v1_literal_j_joint_affine_shell_residual_population_stream_preoutcome_plan(
        joint_preoutcome_registration=joint_preoutcome_registration,
        action_axis_preoutcome_source=action_axis_preoutcome_source,
        population_registration=population_registration,
        route_support_source=route_support_source,
        private_support=None,
        design=design,
    )


def _register_v1_literal_j_joint_affine_shell_residual_population_stream_cpu_reference_preoutcome_plan(
    *,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    action_axis_preoutcome_source: V1LiteralJJointPopulationActionAxisPreoutcomeSource,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    private_support: object,
    design: V1LiteralJJointAffineShellResidualPopulationStreamCUDADesign,
) -> V1LiteralJJointAffineShellResidualPopulationStreamCUDAPreoutcomePlan:
    return _register_v1_literal_j_joint_affine_shell_residual_population_stream_preoutcome_plan(
        joint_preoutcome_registration=joint_preoutcome_registration,
        action_axis_preoutcome_source=action_axis_preoutcome_source,
        population_registration=population_registration,
        route_support_source=None,
        private_support=private_support,
        design=design,
    )


def _register_v1_literal_j_joint_affine_shell_residual_population_stream_from_logical_look_source(
    *,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    logical_look_source: V1SampledV7SharedDW2LiteralJLogicalLookSource,
    joint_population_preoutcome_source: V1LiteralJJointPopulationPreoutcomeSource | None = None,
    affine_supports: tuple[V1LiteralJAffineShellAnalyticSupportDeviceHandle, ...],
    route_support_source: V1LiteralJResidualAnalyticSupportSourceCUDAResult | None,
    design: V1LiteralJJointAffineShellResidualPopulationStreamCUDADesign,
    private_support: _V1LiteralJJointResidualCPUAnalyticSupport | None,
    production: bool,
) -> V1LiteralJJointAffineShellResidualPopulationStreamCUDARegistration:
    """Canonical registration over the exact outcome-independent current-source look."""

    exact = (
        (joint_preoutcome_registration, V1LiteralJJointAffineShellPreoutcomeRegistration),
        (joint_family_handle, V1LiteralJJointAffineShellFamilyDeviceHandle),
        (population_registration, V1LiteralJDFixedResidualPopulationProjectionRegistration),
        (logical_look_source, V1SampledV7SharedDW2LiteralJLogicalLookSource),
        (design, V1LiteralJJointAffineShellResidualPopulationStreamCUDADesign),
    )
    if any(type(value) is not owner for value, owner in exact):
        raise TypeError("segmented residual registration received an inexact owner")
    joint_population_receipt_hash = ""
    if joint_population_preoutcome_source is not None:
        if type(joint_population_preoutcome_source) is not V1LiteralJJointPopulationPreoutcomeSource:
            raise TypeError("joint-population residual registration received an inexact owner")
        joint_population_preoutcome_source.assert_preoutcome_live()
        joint_population_receipt = joint_population_preoutcome_source.receipt
        if (
            joint_population_receipt.logical_source_receipt_hash
            != logical_look_source.receipt.receipt_hash
            or joint_population_receipt.joint_preoutcome_registration_hash
            != joint_preoutcome_registration.registration_hash
            or joint_population_receipt.joint_direction_axis_hash
            != joint_preoutcome_registration.joint_direction_axis_hash
            or joint_population_receipt.route_count
            != joint_preoutcome_registration.route_count
            or joint_population_receipt.literal_j_stream_source_hash
            != joint_preoutcome_registration.literal_j_stream_source_hash
        ):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.PREFIX_MISMATCH,
                "joint-population preoutcome source does not bind the same residual action family",
            )
        joint_population_receipt_hash = joint_population_receipt.receipt_hash
    design.assert_integrity()
    _assert_common_parent_lineage(
        preregistration=joint_preoutcome_registration,
        joint_family_handle=joint_family_handle,
        population=population_registration,
    )
    observed, plans, ordinals, contexts, draws = _segmented_source_preoutcome_axes(
        source=logical_look_source,
        preregistration=joint_preoutcome_registration,
        population=population_registration,
    )
    prereg = joint_preoutcome_registration
    joint = joint_family_handle.receipt
    if production:
        if private_support is not None or type(route_support_source) is not V1LiteralJResidualAnalyticSupportSourceCUDAResult:
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.ANALYTIC_SUPPORT_REQUIRED,
                "production segmented registration requires canonical CUDA analytic support",
            )
        _validate_production_analytic_supports(
            preregistration=joint_preoutcome_registration,
            joint_family_handle=joint_family_handle,
            affine_supports=affine_supports,
            route_support_source=route_support_source,
        )
        support_hashes = tuple(
            item.receipt.receipt_hash for item in affine_supports
        )
        route_support_hash = route_support_source.receipt.receipt_hash
        literal_bound_hash = (
            route_support_source.receipt.literal_j_abs_bound_source_hash
        )
    else:
        if (
            type(private_support) is not _V1LiteralJJointResidualCPUAnalyticSupport
            or route_support_source is not None
            or affine_supports != ()
        ):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.ANALYTIC_SUPPORT_REQUIRED,
                "private segmented reference registration requires its sealed CPU support only",
            )
        private_support.assert_live()
        if (
            private_support.joint_preoutcome_registration_hash
            != prereg.registration_hash
            or private_support.joint_family_receipt_hash != joint.receipt_hash
        ):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.STALE_IDENTITY,
                "private segmented support belongs to another joint family",
            )
        support_hashes = tuple(
            _payload_hash(
                {
                    "private_support_hash": private_support.source_hash,
                    "action_ordinal": index,
                },
                role="private-cpu-segmented-action-support",
            )
            for index in range(prereg.action_count)
        )
        route_support_hash = private_support.source_hash
        literal_bound_hash = private_support.source_hash
    receipt = logical_look_source.receipt
    packet_axis_hashes = tuple(
        _axis_hash(
            f"{role}-segmented-packet-plans",
            tuple(plans[index].packet_plan_hash for index in ordinals[h]),
        )
        for h, role in enumerate(("target", "complement"))
    )
    provider_axis_hashes = tuple(
        _axis_hash(
            f"{role}-segmented-provider-plans",
            tuple(
                plans[index].provider_plan_identity_hash
                for index in ordinals[h]
            ),
        )
        for h, role in enumerate(("target", "complement"))
    )
    return V1LiteralJJointAffineShellResidualPopulationStreamCUDARegistration(
        joint_preoutcome_registration_hash=prereg.registration_hash,
        joint_family_receipt_hash=joint.receipt_hash,
        joint_postsource_custody_axis_hash=joint.postsource_custody_axis_hash,
        population_projection_registration_hash=population_registration.registration_hash,
        suffix_registration_hash=receipt.suffix_registration_hash,
        suffix_packet_hash=_payload_hash(
            {"transport_packet_hashes": receipt.transport_packet_hashes},
            role="segmented-transport-packet-axis",
        ),
        suffix_prefix_freeze_hash=receipt.generalized_logical_prefix_identity_hash,
        row_group_plan_hash=receipt.row_group_plan_hash,
        design_hash=design.design_hash,
        affine_support_receipt_hashes=support_hashes,
        route_support_source_receipt_hash=route_support_hash,
        literal_j_abs_bound_source_hash=literal_bound_hash,
        model_identity_hash=prereg.model_identity_hash,
        chart_identity_hash=prereg.chart_identity_hash,
        current_mask_hash=prereg.current_mask_hash,
        proposal_generation=prereg.proposal_generation,
        route_axis_hash=prereg.route_axis_hash,
        route_id_axis_hash=prereg.route_id_axis_hash,
        route_count=prereg.route_count,
        literal_j_stream_source_hash=prereg.literal_j_stream_source_hash,
        registered_d_prefix_hash=prereg.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=prereg.registered_d_w2_prefix_hash,
        joint_direction_ids=prereg.joint_direction_ids,
        joint_direction_axis_hash=prereg.joint_direction_axis_hash,
        action_count=prereg.action_count,
        shell_size=prereg.shell_size,
        direction_count=prereg.direction_count,
        shell_indices=prereg.shell_indices,
        stratum_ids=("target", "complement"),
        residual_population_counts=population_registration.residual_stratum_counts,
        registered_observed_counts=observed,
        residual_sealed_population_order_hashes=(
            population_registration.residual_sealed_population_order_hashes
        ),
        frozen_dw2_prefix_axis_hashes=(
            _axis_hash("target-segmented-contexts", contexts[0]),
            _axis_hash("complement-segmented-contexts", contexts[1]),
        ),
        frozen_dw2_pair_prefix_axis_hash=receipt.generalized_logical_prefix_identity_hash,
        scorer_packet_plan_hashes=(packet_axis_hashes[0], packet_axis_hashes[1]),
        scorer_provider_plan_hashes=(
            provider_axis_hashes[0],
            provider_axis_hashes[1],
        ),
        scorer_context_axis_hashes=(
            _axis_hash("target-contexts", contexts[0]),
            _axis_hash("complement-contexts", contexts[1]),
        ),
        scorer_draw_axis_hashes=(
            _axis_hash("target-draws", draws[0]),
            _axis_hash("complement-draws", draws[1]),
        ),
        familywise_alpha=prereg.familywise_alpha,
        simultaneous_statement_count=prereg.simultaneous_statement_count,
        per_statement_two_sided_alpha=prereg.per_statement_two_sided_alpha,
        source_representation=(
            "current_source_segmented_joint_population_v1"
            if joint_population_preoutcome_source is not None
            else "current_source_segmented_logical_look_v1"
        ),
        logical_look_source_receipt_hash=receipt.receipt_hash,
        generalized_logical_prefix_identity_hash=(
            receipt.generalized_logical_prefix_identity_hash
        ),
        source_physical_call_offset=receipt.physical_call_offset,
        source_expected_prior_manifest_hash=receipt.expected_prior_manifest_hash,
        source_physical_roles=receipt.physical_roles,
        source_physical_context_counts=receipt.physical_context_counts,
        source_packet_plan_hashes=tuple(plan.packet_plan_hash for plan in plans),
        source_provider_plan_hashes=tuple(
            plan.provider_plan_identity_hash for plan in plans
        ),
        source_capture_identity_hashes=receipt.source_capture_identity_hashes,
        source_transport_packet_hashes=receipt.transport_packet_hashes,
        source_stratum_segment_ordinals=(ordinals[0], ordinals[1]),
        source_stratum_segment_counts=(len(ordinals[0]), len(ordinals[1])),
        joint_population_preoutcome_receipt_hash=joint_population_receipt_hash,
        _logical_look_source_owner=logical_look_source,
        _joint_population_preoutcome_owner=joint_population_preoutcome_source,
        _issuance_token=_REGISTRATION_TOKEN,
    )


def register_v1_literal_j_joint_affine_shell_residual_population_stream_cuda_from_logical_look_source(
    *,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    logical_look_source: V1SampledV7SharedDW2LiteralJLogicalLookSource,
    affine_supports: tuple[V1LiteralJAffineShellAnalyticSupportDeviceHandle, ...],
    route_support_source: V1LiteralJResidualAnalyticSupportSourceCUDAResult,
    design: V1LiteralJJointAffineShellResidualPopulationStreamCUDADesign,
) -> V1LiteralJJointAffineShellResidualPopulationStreamCUDARegistration:
    del (
        joint_preoutcome_registration,
        joint_family_handle,
        population_registration,
        logical_look_source,
        affine_supports,
        route_support_source,
        design,
    )
    _refuse(
        V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.PREOUTCOME_PLAN_REQUIRED,
        "legacy logical-look registration receives POST joint/support owners and cannot prove causal PRE timing; use the action-complete v2 PRE plan",
    )


def register_v1_literal_j_joint_affine_shell_residual_population_stream_cuda_from_joint_population_source(
    *,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    joint_population_preoutcome_source: V1LiteralJJointPopulationPreoutcomeSource,
    affine_supports: tuple[V1LiteralJAffineShellAnalyticSupportDeviceHandle, ...],
    route_support_source: V1LiteralJResidualAnalyticSupportSourceCUDAResult,
    design: V1LiteralJJointAffineShellResidualPopulationStreamCUDADesign,
) -> V1LiteralJJointAffineShellResidualPopulationStreamCUDARegistration:
    del (
        joint_preoutcome_registration,
        joint_family_handle,
        population_registration,
        joint_population_preoutcome_source,
        affine_supports,
        route_support_source,
        design,
    )
    _refuse(
        V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.PREOUTCOME_PLAN_REQUIRED,
        "a singular action owner cannot authorize an action-complete residual stream; use the v2 PRE/POST action-axis seam",
    )


def _register_v1_literal_j_joint_affine_shell_residual_population_stream_cpu_reference_from_logical_look_source(
    *,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    logical_look_source: V1SampledV7SharedDW2LiteralJLogicalLookSource,
    private_support: _V1LiteralJJointResidualCPUAnalyticSupport,
    design: V1LiteralJJointAffineShellResidualPopulationStreamCUDADesign,
) -> V1LiteralJJointAffineShellResidualPopulationStreamCUDARegistration:
    del (
        joint_preoutcome_registration,
        joint_family_handle,
        population_registration,
        logical_look_source,
        private_support,
        design,
    )
    _refuse(
        V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.PREOUTCOME_PLAN_REQUIRED,
        "legacy CPU logical-look registration cannot prove the v2 causal PRE plan",
    )


def _register_v1_literal_j_joint_affine_shell_residual_population_stream_cpu_reference_from_joint_population_source(
    *,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    joint_population_preoutcome_source: V1LiteralJJointPopulationPreoutcomeSource,
    private_support: _V1LiteralJJointResidualCPUAnalyticSupport,
    design: V1LiteralJJointAffineShellResidualPopulationStreamCUDADesign,
) -> V1LiteralJJointAffineShellResidualPopulationStreamCUDARegistration:
    del (
        joint_preoutcome_registration,
        joint_family_handle,
        population_registration,
        joint_population_preoutcome_source,
        private_support,
        design,
    )
    _refuse(
        V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.PREOUTCOME_PLAN_REQUIRED,
        "legacy singular CPU source cannot prove the action-complete v2 causal PRE plan",
    )


def build_v1_literal_j_joint_affine_shell_residual_population_stream_cuda_hbm_forecast(
    *,
    registration: V1LiteralJJointAffineShellResidualPopulationStreamCUDARegistration,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    affine_supports: tuple[V1LiteralJAffineShellAnalyticSupportDeviceHandle, ...],
    route_support_source: V1LiteralJResidualAnalyticSupportSourceCUDAResult,
    design: V1LiteralJJointAffineShellResidualPopulationStreamCUDADesign,
) -> V1LiteralJJointAffineShellResidualPopulationStreamCUDAHBMForecast:
    if type(registration) is not V1LiteralJJointAffineShellResidualPopulationStreamCUDARegistration:
        raise TypeError("HBM forecast requires the exact joint stream registration")
    registration.assert_integrity()
    if (
        registration.design_hash != design.design_hash
        or registration.joint_preoutcome_registration_hash
        != joint_preoutcome_registration.registration_hash
        or registration.joint_family_receipt_hash != joint_family_handle.receipt.receipt_hash
    ):
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.STALE_IDENTITY,
            "HBM forecast parents differ from the registered stream",
        )
    device = _validate_production_analytic_supports(
        preregistration=joint_preoutcome_registration,
        joint_family_handle=joint_family_handle,
        affine_supports=affine_supports,
        route_support_source=route_support_source,
    )
    if design.deterministic_algorithms_required and not torch.are_deterministic_algorithms_enabled():
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.DETERMINISM_DISABLED,
            "deterministic PyTorch algorithms must be enabled before forecast",
        )
    free, total = torch.cuda.mem_get_info(device)
    reserved = torch.cuda.memory_reserved(device)
    return _make_forecast(
        design=design,
        joint_preoutcome_registration=joint_preoutcome_registration,
        joint_family_handle=joint_family_handle,
        affine_support_receipt_hashes=registration.affine_support_receipt_hashes,
        route_support_source_receipt_hash=registration.route_support_source_receipt_hash,
        observed_counts=registration.registered_observed_counts,
        device=device,
        total_bytes=int(total),
        free_bytes=int(free),
        reserved_bytes=int(reserved),
        _admission_memory=(cuda_reusable_memory_bytes(device, framework=torch, free_bytes=free), int(torch.cuda.memory_allocated(device))),
    )


class _V1LiteralJJointResidualCPUAnalyticSupport:
    """Issuer-sealed small CPU oracle support; never accepted by production."""

    __slots__ = (
        "joint_preoutcome_registration_hash",
        "joint_family_receipt_hash",
        "source_hash",
        "_literal_bound",
        "_projection_bound",
        "_literal_guard",
        "_projection_guard",
        "_literal_invariant",
        "_projection_invariant",
    )

    def __init__(
        self,
        *,
        joint_preoutcome_registration_hash: str,
        joint_family_receipt_hash: str,
        literal_bound: Any,
        projection_bound: Any,
        issuance_token: object | None = None,
    ) -> None:
        if issuance_token is not _CPU_SUPPORT_TOKEN:
            raise PermissionError("private CPU support requires its test-only issuer")
        self.joint_preoutcome_registration_hash = _identity(
            joint_preoutcome_registration_hash,
            name="joint_preoutcome_registration_hash",
        )
        self.joint_family_receipt_hash = _identity(
            joint_family_receipt_hash,
            name="joint_family_receipt_hash",
        )
        if literal_bound.device.type != "cpu" or projection_bound.device.type != "cpu":
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.INVALID_TYPE,
                "private reference analytic support must remain on CPU",
            )
        self._literal_bound = literal_bound.detach().clone().contiguous()
        self._projection_bound = projection_bound.detach().clone().contiguous()
        self._literal_guard = self._literal_bound.detach().clone()
        self._projection_guard = self._projection_bound.detach().clone()
        self._literal_invariant = _tensor_invariant(self._literal_bound)
        self._projection_invariant = _tensor_invariant(self._projection_bound)
        self.source_hash = _payload_hash(
            {
                "joint_preoutcome_registration_hash": self.joint_preoutcome_registration_hash,
                "joint_family_receipt_hash": self.joint_family_receipt_hash,
                "literal_bound": tuple(float(item) for item in self._literal_bound.tolist()),
                "projection_bound": tuple(float(item) for item in self._projection_bound.tolist()),
                "private_cpu_reference_only": True,
            },
            role="private-cpu-support",
        )
        self.assert_live()

    def assert_live(self) -> None:
        if (
            self._literal_bound is None
            or self._projection_bound is None
            or _tensor_invariant(self._literal_bound) != self._literal_invariant
            or _tensor_invariant(self._projection_bound) != self._projection_invariant
            or not bool(torch.equal(self._literal_bound, self._literal_guard))
            or not bool(torch.equal(self._projection_bound, self._projection_guard))
        ):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.MUTATED_CUSTODY,
                "private CPU analytic support mutated or was evicted",
            )

    def tensors(self) -> tuple[Any, Any]:
        self.assert_live()
        return self._literal_bound, self._projection_bound

    def evict(self) -> None:
        self.assert_live()
        self._literal_bound = None
        self._projection_bound = None
        self._literal_guard = None
        self._projection_guard = None

    def __reduce__(self) -> NoReturn:
        raise TypeError("private CPU analytic support is nonserializable")


def _issue_v1_literal_j_joint_affine_shell_residual_cpu_reference_support(
    *,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    literal_j_abs_bound: Any,
    projection_abs_bound: Any,
) -> _V1LiteralJJointResidualCPUAnalyticSupport:
    """Issue bounded caller-authored support solely for the small CPU oracle."""

    if (
        type(joint_preoutcome_registration)
        is not V1LiteralJJointAffineShellPreoutcomeRegistration
        or type(joint_family_handle) is not V1LiteralJJointAffineShellFamilyDeviceHandle
    ):
        raise TypeError("private support requires exact joint owners")
    prereg = joint_preoutcome_registration
    prereg.assert_integrity()
    if prereg.route_count > _CPU_REFERENCE_ROUTE_LIMIT:
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.CPU_PRODUCTION_FALLBACK,
            "private CPU oracle is limited to 4096 routes",
        )
    literal = _require_fp64_tensor(
        literal_j_abs_bound,
        name="private literal_j_abs_bound",
        shape=(prereg.route_count,),
        nonnegative=True,
    )
    projection = _require_fp64_tensor(
        projection_abs_bound,
        name="private projection_abs_bound",
        shape=(prereg.direction_count,),
        device=literal.device,
        nonnegative=True,
    )
    if literal.device.type != "cpu":
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.CPU_PRODUCTION_FALLBACK,
            "private reference support accepts CPU tensors only",
        )
    with joint_family_handle.borrow() as lease:
        outside, shell = lease.projection_custody()
        if any(item.device != literal.device for item in outside) or shell.device != literal.device:
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.DEVICE_REQUIRED,
                "private support and joint directions must share one CPU device",
            )
    return _V1LiteralJJointResidualCPUAnalyticSupport(
        joint_preoutcome_registration_hash=prereg.registration_hash,
        joint_family_receipt_hash=joint_family_handle.receipt.receipt_hash,
        literal_bound=literal,
        projection_bound=projection,
        issuance_token=_CPU_SUPPORT_TOKEN,
    )


def _bind_v1_literal_j_joint_affine_shell_residual_population_stream_postsource(
    *,
    preoutcome_plan: V1LiteralJJointAffineShellResidualPopulationStreamCUDAPreoutcomePlan,
    action_axis_source: V1LiteralJJointPopulationActionAxisSource,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    affine_supports: tuple[V1LiteralJAffineShellAnalyticSupportDeviceHandle, ...],
    route_support_source: V1LiteralJResidualAnalyticSupportSourceCUDAResult | None,
    private_support: object | None,
) -> V1LiteralJJointAffineShellResidualPopulationStreamCUDAPostsourceBinding:
    if (
        type(preoutcome_plan)
        is not V1LiteralJJointAffineShellResidualPopulationStreamCUDAPreoutcomePlan
        or type(action_axis_source) is not V1LiteralJJointPopulationActionAxisSource
        or type(joint_family_handle) is not V1LiteralJJointAffineShellFamilyDeviceHandle
        or type(affine_supports) is not tuple
    ):
        raise TypeError("v2 residual POST binding requires exact typed parents")
    with preoutcome_plan._lock:
        if preoutcome_plan._state != "preoutcome":
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.SOURCE_DESCENT_MISMATCH,
                "the residual PRE plan may acquire exactly one POST descent",
            )
        plan = preoutcome_plan.receipt
        plan.assert_integrity()
        action_axis_source.assert_live()
        post_axis = action_axis_source.receipt
        if (
            action_axis_source.preoutcome_source is not preoutcome_plan._action_axis
            or post_axis.preoutcome_action_axis_receipt_hash
            != plan.action_axis_preoutcome_receipt_hash
            or post_axis.joint_preoutcome_registration_hash
            != plan.joint_preoutcome_registration_hash
            or post_axis.joint_direction_axis_hash != plan.joint_direction_axis_hash
            or post_axis.physical_action_axis_hash != plan.physical_action_axis_hash
            or post_axis.action_count != plan.action_count
        ):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.SOURCE_DESCENT_MISMATCH,
                "the POST action axis does not descend from this complete PRE plan",
            )
        post_sources = action_axis_source._sources
        if (
            not post_sources
            or any(source.logical_result is not post_sources[0].logical_result for source in post_sources)
        ):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.SOURCE_DESCENT_MISMATCH,
                "POST action members do not retain one exact residual result",
            )
        logical_result = post_sources[0].logical_result
        if (
            type(logical_result)
            is not V1SampledV7SharedDW2LiteralJLogicalLookResult
            or logical_result.source is not preoutcome_plan._logical_source
            or not preoutcome_plan._logical_source.is_bound
        ):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.SOURCE_DESCENT_MISMATCH,
                "the action-axis POST result is not the exact bound PRE logical source",
            )
        logical_result.assert_live()
        preregistration = preoutcome_plan._preregistration
        population = preoutcome_plan._population
        design = preoutcome_plan._design
        _assert_common_parent_lineage(
            preregistration=preregistration,
            joint_family_handle=joint_family_handle,
            population=population,
        )
        joint = joint_family_handle.receipt
        if (
            joint.physical_action_axis_hash != plan.physical_action_axis_hash
            or joint.preoutcome_joint_direction_axis_hash
            != plan.joint_direction_axis_hash
            or joint.joint_direction_ids != plan.joint_direction_ids
            or joint.action_count != plan.action_count
            or joint.shell_size != plan.shell_size
            or joint.direction_count != plan.direction_count
        ):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.ORDER_MISMATCH,
                "the POST joint family reorders or replaces the frozen PRE action axis",
            )
        if plan.support_derivation_mode == "production_route_bound_source_v1":
            if (
                route_support_source is not preoutcome_plan._support_derivation_source
                or private_support is not None
            ):
                _refuse(
                    V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.ANALYTIC_SUPPORT_REQUIRED,
                    "POST production support must descend from the exact PRE route-bound source",
                )
            _validate_production_analytic_supports(
                preregistration=preregistration,
                joint_family_handle=joint_family_handle,
                affine_supports=affine_supports,
                route_support_source=route_support_source,
            )
            support_hashes = tuple(item.receipt.receipt_hash for item in affine_supports)
            route_support_hash = route_support_source.receipt.receipt_hash
            literal_bound_hash = route_support_source.receipt.literal_j_abs_bound_source_hash
        else:
            if (
                private_support is not preoutcome_plan._support_derivation_source
                or type(private_support) is not _V1LiteralJJointResidualCPUAnalyticSupport
                or route_support_source is not None
                or affine_supports != ()
            ):
                _refuse(
                    V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.ANALYTIC_SUPPORT_REQUIRED,
                    "POST private reference support must be the exact PRE derivation owner",
                )
            private_support.assert_live()
            if (
                private_support.joint_preoutcome_registration_hash
                != preregistration.registration_hash
                or private_support.joint_family_receipt_hash != joint.receipt_hash
            ):
                _refuse(
                    V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.STALE_IDENTITY,
                    "private reference support belongs to another POST joint family",
                )
            support_hashes = tuple(
                _payload_hash(
                    {
                        "private_support_hash": private_support.source_hash,
                        "action_ordinal": index,
                    },
                    role="private-cpu-v2-action-support",
                )
                for index in range(plan.action_count)
            )
            route_support_hash = private_support.source_hash
            literal_bound_hash = private_support.source_hash
        source_receipt = preoutcome_plan._logical_source.receipt
        packet_axis_hashes = tuple(
            _axis_hash(
                f"{role}-segmented-packet-plans",
                tuple(plan.source_packet_plan_hashes[index] for index in plan.source_stratum_segment_ordinals[h]),
            )
            for h, role in enumerate(("target", "complement"))
        )
        provider_axis_hashes = tuple(
            _axis_hash(
                f"{role}-segmented-provider-plans",
                tuple(plan.source_provider_plan_hashes[index] for index in plan.source_stratum_segment_ordinals[h]),
            )
            for h, role in enumerate(("target", "complement"))
        )
        registration = V1LiteralJJointAffineShellResidualPopulationStreamCUDARegistration(
            joint_preoutcome_registration_hash=preregistration.registration_hash,
            joint_family_receipt_hash=joint.receipt_hash,
            joint_postsource_custody_axis_hash=joint.postsource_custody_axis_hash,
            population_projection_registration_hash=population.registration_hash,
            suffix_registration_hash=plan.suffix_registration_hash,
            suffix_packet_hash=_payload_hash(
                {"transport_packet_hashes": plan.source_transport_packet_hashes},
                role="segmented-transport-packet-axis",
            ),
            suffix_prefix_freeze_hash=plan.generalized_logical_prefix_identity_hash,
            row_group_plan_hash=plan.row_group_plan_hash,
            design_hash=design.design_hash,
            affine_support_receipt_hashes=support_hashes,
            route_support_source_receipt_hash=route_support_hash,
            literal_j_abs_bound_source_hash=literal_bound_hash,
            model_identity_hash=plan.model_identity_hash,
            chart_identity_hash=plan.chart_identity_hash,
            current_mask_hash=plan.current_mask_hash,
            proposal_generation=plan.proposal_generation,
            route_axis_hash=plan.route_axis_hash,
            route_id_axis_hash=plan.route_id_axis_hash,
            route_count=plan.route_count,
            literal_j_stream_source_hash=plan.literal_j_stream_source_hash,
            registered_d_prefix_hash=plan.registered_d_prefix_hash,
            registered_d_w2_prefix_hash=plan.registered_d_w2_prefix_hash,
            joint_direction_ids=plan.joint_direction_ids,
            joint_direction_axis_hash=plan.joint_direction_axis_hash,
            action_count=plan.action_count,
            shell_size=plan.shell_size,
            direction_count=plan.direction_count,
            shell_indices=plan.shell_indices,
            stratum_ids=("target", "complement"),
            residual_population_counts=plan.residual_population_counts,
            registered_observed_counts=plan.registered_observed_counts,
            residual_sealed_population_order_hashes=plan.residual_sealed_population_order_hashes,
            frozen_dw2_prefix_axis_hashes=plan.scorer_context_axis_hashes,
            frozen_dw2_pair_prefix_axis_hash=plan.generalized_logical_prefix_identity_hash,
            scorer_packet_plan_hashes=(packet_axis_hashes[0], packet_axis_hashes[1]),
            scorer_provider_plan_hashes=(provider_axis_hashes[0], provider_axis_hashes[1]),
            scorer_context_axis_hashes=plan.scorer_context_axis_hashes,
            scorer_draw_axis_hashes=plan.scorer_draw_axis_hashes,
            familywise_alpha=plan.familywise_alpha,
            simultaneous_statement_count=plan.simultaneous_statement_count,
            per_statement_two_sided_alpha=plan.per_statement_two_sided_alpha,
            source_representation="current_source_segmented_joint_population_action_axis_v2",
            logical_look_source_receipt_hash=plan.logical_look_source_receipt_hash,
            generalized_logical_prefix_identity_hash=plan.generalized_logical_prefix_identity_hash,
            source_physical_call_offset=plan.source_physical_call_offset,
            source_expected_prior_manifest_hash=plan.source_expected_prior_manifest_hash,
            source_physical_roles=plan.source_physical_roles,
            source_physical_context_counts=plan.source_physical_context_counts,
            source_packet_plan_hashes=plan.source_packet_plan_hashes,
            source_provider_plan_hashes=plan.source_provider_plan_hashes,
            source_capture_identity_hashes=plan.source_capture_identity_hashes,
            source_transport_packet_hashes=plan.source_transport_packet_hashes,
            source_stratum_segment_ordinals=plan.source_stratum_segment_ordinals,
            source_stratum_segment_counts=tuple(
                len(axis) for axis in plan.source_stratum_segment_ordinals
            ),
            causal_preoutcome_plan_receipt_hash=plan.receipt_hash,
            joint_population_action_axis_preoutcome_receipt_hash=(
                plan.action_axis_preoutcome_receipt_hash
            ),
            joint_population_action_axis_postsource_receipt_hash=post_axis.receipt_hash,
            registered_before_residual_outcomes=False,
            postsource_binding_registered_preoutcome=False,
            _logical_look_source_owner=preoutcome_plan._logical_source,
            _causal_preoutcome_plan_owner=preoutcome_plan,
            _joint_population_action_axis_postsource_owner=action_axis_source,
            _issuance_token=_REGISTRATION_TOKEN,
        )
        preoutcome_plan._state = "bound"
        try:
            registration_hash = registration.registration_hash
            receipt = V1LiteralJJointAffineShellResidualPopulationStreamCUDAPostsourceBindingReceipt(
                preoutcome_plan_receipt_hash=plan.receipt_hash,
                action_axis_preoutcome_receipt_hash=plan.action_axis_preoutcome_receipt_hash,
                action_axis_postsource_receipt_hash=post_axis.receipt_hash,
                logical_source_receipt_hash=plan.logical_look_source_receipt_hash,
                logical_result_receipt_hash=logical_result.receipt.receipt_hash,
                joint_preoutcome_registration_hash=preregistration.registration_hash,
                joint_family_receipt_hash=joint.receipt_hash,
                joint_postsource_custody_axis_hash=joint.postsource_custody_axis_hash,
                affine_support_receipt_hashes=support_hashes,
                route_support_source_receipt_hash=route_support_hash,
                literal_j_abs_bound_source_hash=literal_bound_hash,
                registration_hash=registration_hash,
                physical_action_axis_hash=plan.physical_action_axis_hash,
                joint_direction_axis_hash=plan.joint_direction_axis_hash,
                _issuance_token=_POSTSOURCE_BINDING_RECEIPT_TOKEN,
            )
            return V1LiteralJJointAffineShellResidualPopulationStreamCUDAPostsourceBinding(
                receipt=receipt,
                registration=registration,
                preoutcome_plan=preoutcome_plan,
                action_axis_source=action_axis_source,
                logical_result=logical_result,
                joint_family_handle=joint_family_handle,
                affine_supports=affine_supports,
                route_support_source=route_support_source,
                private_support=private_support,
                _issuance_token=_POSTSOURCE_BINDING_OWNER_TOKEN,
            )
        except Exception:
            preoutcome_plan._state = "bind_failed"
            raise


def bind_v1_literal_j_joint_affine_shell_residual_population_stream_cuda_postsource(
    *,
    preoutcome_plan: V1LiteralJJointAffineShellResidualPopulationStreamCUDAPreoutcomePlan,
    action_axis_source: V1LiteralJJointPopulationActionAxisSource,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    affine_supports: tuple[V1LiteralJAffineShellAnalyticSupportDeviceHandle, ...],
    route_support_source: V1LiteralJResidualAnalyticSupportSourceCUDAResult,
) -> V1LiteralJJointAffineShellResidualPopulationStreamCUDAPostsourceBinding:
    """Attach exact POST outcomes/supports to an already-frozen production plan."""

    return _bind_v1_literal_j_joint_affine_shell_residual_population_stream_postsource(
        preoutcome_plan=preoutcome_plan,
        action_axis_source=action_axis_source,
        joint_family_handle=joint_family_handle,
        affine_supports=affine_supports,
        route_support_source=route_support_source,
        private_support=None,
    )


def _bind_v1_literal_j_joint_affine_shell_residual_population_stream_cpu_reference_postsource(
    *,
    preoutcome_plan: V1LiteralJJointAffineShellResidualPopulationStreamCUDAPreoutcomePlan,
    action_axis_source: V1LiteralJJointPopulationActionAxisSource,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    private_support: object,
) -> V1LiteralJJointAffineShellResidualPopulationStreamCUDAPostsourceBinding:
    return _bind_v1_literal_j_joint_affine_shell_residual_population_stream_postsource(
        preoutcome_plan=preoutcome_plan,
        action_axis_source=action_axis_source,
        joint_family_handle=joint_family_handle,
        affine_supports=(),
        route_support_source=None,
        private_support=private_support,
    )


def _register_v1_literal_j_joint_affine_shell_residual_population_stream_cpu_reference(
    *,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    suffix_registration: V1SampledVariableShellDW2SuffixRegistration,
    packet: V1SampledVariableShellDW2SuffixPacket,
    prefix_freeze: V1SampledVariableShellDW2SuffixPrefixFreezeReceipt,
    row_group_plan: V1SampledDualChartRowGroupPlan,
    private_support: _V1LiteralJJointResidualCPUAnalyticSupport,
    design: V1LiteralJJointAffineShellResidualPopulationStreamCUDADesign,
) -> V1LiteralJJointAffineShellResidualPopulationStreamCUDARegistration:
    if type(private_support) is not _V1LiteralJJointResidualCPUAnalyticSupport:
        raise TypeError("CPU reference registration requires exact private support")
    private_support.assert_live()
    design.assert_integrity()
    _assert_common_parent_lineage(
        preregistration=joint_preoutcome_registration,
        joint_family_handle=joint_family_handle,
        population=population_registration,
    )
    observed, scorer = _validate_suffix_and_scorer_plan(
        preregistration=joint_preoutcome_registration,
        population=population_registration,
        suffix_registration=suffix_registration,
        packet=packet,
        prefix_freeze=prefix_freeze,
        row_group_plan=row_group_plan,
    )
    prereg = joint_preoutcome_registration
    joint = joint_family_handle.receipt
    if (
        private_support.joint_preoutcome_registration_hash != prereg.registration_hash
        or private_support.joint_family_receipt_hash != joint.receipt_hash
    ):
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.STALE_IDENTITY,
            "private CPU support belongs to another joint family",
        )
    return V1LiteralJJointAffineShellResidualPopulationStreamCUDARegistration(
        joint_preoutcome_registration_hash=prereg.registration_hash,
        joint_family_receipt_hash=joint.receipt_hash,
        joint_postsource_custody_axis_hash=joint.postsource_custody_axis_hash,
        population_projection_registration_hash=population_registration.registration_hash,
        suffix_registration_hash=suffix_registration.receipt_hash,
        suffix_packet_hash=packet.receipt_hash,
        suffix_prefix_freeze_hash=prefix_freeze.receipt_hash,
        row_group_plan_hash=row_group_plan.group_plan_hash,
        design_hash=design.design_hash,
        affine_support_receipt_hashes=tuple(
            _payload_hash(
                {"private_support_hash": private_support.source_hash, "action_ordinal": index},
                role="private-cpu-action-support",
            )
            for index in range(prereg.action_count)
        ),
        route_support_source_receipt_hash=private_support.source_hash,
        literal_j_abs_bound_source_hash=private_support.source_hash,
        model_identity_hash=prereg.model_identity_hash,
        chart_identity_hash=prereg.chart_identity_hash,
        current_mask_hash=prereg.current_mask_hash,
        proposal_generation=prereg.proposal_generation,
        route_axis_hash=prereg.route_axis_hash,
        route_id_axis_hash=prereg.route_id_axis_hash,
        route_count=prereg.route_count,
        literal_j_stream_source_hash=prereg.literal_j_stream_source_hash,
        registered_d_prefix_hash=prereg.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=prereg.registered_d_w2_prefix_hash,
        joint_direction_ids=prereg.joint_direction_ids,
        joint_direction_axis_hash=prereg.joint_direction_axis_hash,
        action_count=prereg.action_count,
        shell_size=prereg.shell_size,
        direction_count=prereg.direction_count,
        shell_indices=prereg.shell_indices,
        stratum_ids=("target", "complement"),
        residual_population_counts=population_registration.residual_stratum_counts,
        registered_observed_counts=observed,
        residual_sealed_population_order_hashes=population_registration.residual_sealed_population_order_hashes,
        frozen_dw2_prefix_axis_hashes=(
            prefix_freeze.dw2_target_prefix_axis_hash,
            prefix_freeze.dw2_complement_prefix_axis_hash,
        ),
        frozen_dw2_pair_prefix_axis_hash=prefix_freeze.dw2_pair_prefix_axis_hash,
        scorer_packet_plan_hashes=(scorer[0].packet_plan_hash, scorer[1].packet_plan_hash),
        scorer_provider_plan_hashes=(scorer[0].provider_plan_identity_hash, scorer[1].provider_plan_identity_hash),
        scorer_context_axis_hashes=(
            _axis_hash("target-contexts", scorer[0].context_ids),
            _axis_hash("complement-contexts", scorer[1].context_ids),
        ),
        scorer_draw_axis_hashes=(
            _axis_hash("target-draws", scorer[0].draw_ids),
            _axis_hash("complement-draws", scorer[1].draw_ids),
        ),
        familywise_alpha=prereg.familywise_alpha,
        simultaneous_statement_count=prereg.simultaneous_statement_count,
        per_statement_two_sided_alpha=prereg.per_statement_two_sided_alpha,
        source_physical_context_counts=observed,
        source_packet_plan_hashes=(
            scorer[0].packet_plan_hash,
            scorer[1].packet_plan_hash,
        ),
        source_provider_plan_hashes=(
            scorer[0].provider_plan_identity_hash,
            scorer[1].provider_plan_identity_hash,
        ),
        source_capture_identity_hashes=(
            scorer[0].source_capture_identity_hash,
            scorer[1].source_capture_identity_hash,
        ),
        source_transport_packet_hashes=(packet.receipt_hash,),
        _issuance_token=_REGISTRATION_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class _Pass1Result:
    projection_centers: tuple[Any, Any] = field(repr=False, compare=False)
    projection_radii: tuple[Any, Any] = field(repr=False, compare=False)
    q_center: Any = field(repr=False, compare=False)
    q_radius: Any = field(repr=False, compare=False)
    compact_shell_m_center: Any = field(repr=False, compare=False)
    compact_shell_m_radius: Any = field(repr=False, compare=False)
    route_window_read_count: int
    route_tile_read_count: int
    action_tile_submission_count: int
    shared_shell_gather_tile_count: int
    payload_read_bytes: int
    scalar_support_check_count: int


def _normalize_literal_j_attachment_groups(
    attachments: object,
) -> tuple[
    tuple[V1SampledDualChartRowShardAttachment, ...],
    tuple[V1SampledDualChartRowShardAttachment, ...],
]:
    """Return target/complement segment groups without copying row payloads.

    Historical callers supply one attachment per stratum.  The current-source
    path supplies an immutable pair of nonempty attachment tuples.  Keeping
    normalization at the stream boundary lets both paths share the exact same
    interval numerics while making the production segmented representation
    explicit.
    """

    if type(attachments) is not tuple or len(attachments) != 2:
        raise TypeError(
            "literal-J residual source must be one target/complement pair"
        )
    if all(
        type(item) is V1SampledDualChartRowShardAttachment
        for item in attachments
    ):
        return ((attachments[0],), (attachments[1],))  # type: ignore[return-value]
    if any(type(group) is not tuple or not group for group in attachments):
        raise TypeError(
            "segmented literal-J residual source requires two nonempty tuples"
        )
    groups = tuple(tuple(group) for group in attachments)
    if any(
        type(item) is not V1SampledDualChartRowShardAttachment
        for group in groups
        for item in group
    ):
        raise TypeError(
            "segmented literal-J residual source contains an inexact attachment"
        )
    return groups  # type: ignore[return-value]


def _compact_shell_m_interval(
    *,
    projection: Any,
    projection_radius: Any,
    projection_bound: Any,
    literal_bound: Any,
    shell_indices: tuple[int, ...],
    action_count: int,
    population_count: int,
    alpha: float,
) -> tuple[Any, Any]:
    """Enclose ``E[(J V_d) J_b]`` for every registered direction/shell pair.

    The shell basis directions are already the final columns of ``projection``.
    Consequently this compact Gram slice is available from pass 1 alone.  The
    source propagates the independent projection enclosures without exposing
    either retained context-by-direction tensor to a consumer.
    """

    context_count, direction_count = tuple(int(item) for item in projection.shape)
    shell_size = len(shell_indices)
    if (
        tuple(projection_radius.shape) != (context_count, direction_count)
        or tuple(projection_bound.shape) != (direction_count,)
        or tuple(literal_bound.shape) != (int(literal_bound.numel()),)
        or action_count + shell_size != direction_count
        or context_count <= 0
    ):
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.INVALID_DESIGN,
            "pass-1 compact shell-M inputs do not replay the joint direction axis",
        )
    shell_projection = projection[:, action_count:]
    shell_projection_radius = projection_radius[:, action_count:]
    center_sum = projection.T @ shell_projection

    absolute_projection = torch.abs(projection)
    absolute_shell_projection = absolute_projection[:, action_count:]
    round_mass = absolute_projection.T @ absolute_shell_projection
    uncertainty_terms = (
        absolute_projection.T @ shell_projection_radius,
        projection_radius.T @ absolute_shell_projection,
        projection_radius.T @ shell_projection_radius,
    )
    operations = 2 * context_count + 1
    relative = _gamma(operations)
    if not math.isfinite(relative) or relative >= 1.0:
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.NUMERICAL_REFUSAL,
            "pass-1 compact shell-M FP64 error factor is not finite below one",
        )
    recovery = _div_up(relative, 1.0 - relative)
    dot_floor = _mul_up(float(operations), _MIN_SUBNORMAL)
    uncertainty_upper = torch.zeros_like(center_sum)
    for term in uncertainty_terms:
        uncertainty_upper = _add_up(
            uncertainty_upper,
            _add_up(
                _add_up(term, _mul_up(term, recovery)),
                dot_floor,
            ),
        )
    sum_radius = _add_up(
        _add_up(_mul_up(round_mass, recovery), uncertainty_upper),
        dot_floor,
    )
    center, numeric_radius = _scale_interval_by_uniform_mean(
        center_sum=center_sum,
        radius_sum=sum_radius,
        count=context_count,
    )

    shell_index = torch.tensor(
        shell_indices,
        dtype=torch.long,
        device=literal_bound.device,
    )
    shell_bound = literal_bound.index_select(0, shell_index)
    factor = _serfling_factor(
        observed=context_count,
        population=population_count,
        alpha=alpha,
    )
    # A signed product in [-P_d J_b, +P_d J_b] has range 2 P_d J_b.
    if factor == 0.0:
        # A residual census has no sampling uncertainty.  Avoid applying the
        # generic upward multiplication helper to exact zero: nextafter(0,+∞)
        # would otherwise manufacture a support-dependent subnormal radius.
        sampling_radius = torch.zeros_like(center)
    else:
        sampling_radius = _mul_up(
            _mul_up(
                _mul_up(2.0, factor),
                projection_bound.reshape(-1, 1),
            ),
            shell_bound.reshape(1, -1),
        )
    radius = _add_up(numeric_radius, sampling_radius).contiguous()
    if (
        tuple(center.shape) != (direction_count, shell_size)
        or tuple(radius.shape) != (direction_count, shell_size)
        or not bool(torch.isfinite(center).all().item())
        or not bool(torch.isfinite(radius).all().item())
        or bool((radius < 0.0).any().item())
    ):
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.NUMERICAL_REFUSAL,
            "pass-1 compact shell-M interval is not one finite outward enclosure",
        )
    return center.contiguous(), radius


def _pass1_joint_projections(
    *,
    attachments: object,
    outside_vectors: tuple[Any, ...],
    shell_indices: tuple[int, ...],
    literal_bound: Any,
    projection_bound: Any,
    population_counts: tuple[int, int],
    per_statement_alpha: float,
    design: V1LiteralJJointAffineShellResidualPopulationStreamCUDADesign,
    failure_after_first_real_tile: bool,
) -> _Pass1Result:
    attachment_groups = _normalize_literal_j_attachment_groups(attachments)
    action_count = len(outside_vectors)
    route_count = int(literal_bound.numel())
    direction_count = action_count + len(shell_indices)
    device = outside_vectors[0].device
    centers: list[Any] = []
    radii: list[Any] = []
    q_centers: list[Any] = []
    q_radii: list[Any] = []
    compact_shell_m_centers: list[Any] = []
    compact_shell_m_radii: list[Any] = []
    window_reads = tile_reads = action_submissions = shell_gathers = payload = checks = 0
    for h, stratum_attachments in enumerate(attachment_groups):
        context_count = sum(
            len(attachment.receipt.context_ids)
            for attachment in stratum_attachments
        )
        center = torch.zeros((context_count, direction_count), dtype=torch.float64, device=device)
        radius = torch.zeros_like(center)
        local_tiles = 0
        row_offset = 0
        for attachment in stratum_attachments:
            segment_count = len(attachment.receipt.context_ids)
            segment_center = center[row_offset : row_offset + segment_count]
            segment_absolute = torch.zeros_like(segment_center)
            segment_tiles = 0
            for parent_start, parent_stop, parent_rows in attachment.iter_route_windows(verify=True):
                window_reads += 1
                width = parent_stop - parent_start
                for local_start in range(0, width, design.route_tile_size):
                    local_stop = min(width, local_start + design.route_tile_size)
                    start = parent_start + local_start
                    stop = parent_start + local_stop
                    rows = _stage_source_tile(parent_rows[:, local_start:local_stop], device=device)
                    route_bound = literal_bound[start:stop]
                    checks += 1
                    if bool((torch.abs(rows) > route_bound.reshape(1, -1)).any().item()):
                        _refuse(
                            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.SUPPORT_VIOLATION,
                            "pass-1 literal-J tile exceeded its mandatory analytic route support",
                        )
                    rows_abs = torch.abs(rows)
                    for action_start in range(0, action_count, design.action_tile_size):
                        action_stop = min(action_count, action_start + design.action_tile_size)
                        z_tile = torch.stack(
                            tuple(
                                outside_vectors[index][start:stop]
                                for index in range(action_start, action_stop)
                            ),
                            dim=0,
                        ).contiguous()
                        segment_center[:, action_start:action_stop].add_(rows @ z_tile.T)
                        segment_absolute[:, action_start:action_stop].add_(
                            rows_abs @ torch.abs(z_tile).T
                        )
                        action_submissions += 1
                    for shell_ordinal, physical in enumerate(
                        shell_indices,
                        start=action_count,
                    ):
                        if start <= physical < stop:
                            column = physical - start
                            segment_center[:, shell_ordinal].add_(rows[:, column])
                            segment_absolute[:, shell_ordinal].add_(rows_abs[:, column])
                            shell_gathers += 1
                    tile_reads += 1
                    local_tiles += 1
                    segment_tiles += 1
                    payload += int(rows.numel()) * _F64_BYTES
                    if failure_after_first_real_tile and tile_reads == 1:
                        raise RuntimeError(
                            "injected failure after one real pass-1 literal-J tile read"
                        )
            # Each segment is one separately accumulated route dot product.
            # Its numerical enclosure therefore uses only that segment's
            # actual tiling history; concatenating raw context-by-route rows
            # is neither needed nor permitted.
            operations = 2 * route_count + 2 * segment_tiles + 1
            relative = _gamma(operations)
            if not math.isfinite(relative) or relative >= 1.0:
                _refuse(
                    V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.NUMERICAL_REFUSAL,
                    "pass-1 FP64 error factor is not finite below one",
                )
            recovery = _div_up(relative, 1.0 - relative)
            radius[row_offset : row_offset + segment_count].copy_(
                _add_up(
                    _mul_up(segment_absolute, recovery),
                    _mul_up(float(operations), _MIN_SUBNORMAL),
                )
            )
            del segment_absolute
            row_offset += segment_count
        if row_offset != context_count:
            raise AssertionError("segmented pass-1 row cursor did not close")
        radius = radius.contiguous()
        upper = _add_up(torch.abs(center), radius)
        checks += 1
        if bool((upper > projection_bound.reshape(1, -1)).any().item()):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.SUPPORT_VIOLATION,
                "joint projection interval exceeded its pre-outcome analytic support",
            )
        # The original accumulation mass is dead here.  Release it before the
        # compact Gram calculation so its forecasted n x D allocation can be
        # reused by the absolute projection workspace below.
        q_center, q_radius = _quadratic_interval(
            projection=center,
            projection_radius=radius,
            projection_bound=projection_bound,
            population_count=population_counts[h],
            alpha=per_statement_alpha,
        )
        compact_shell_m_center, compact_shell_m_radius = _compact_shell_m_interval(
            projection=center,
            projection_radius=radius,
            projection_bound=projection_bound,
            literal_bound=literal_bound,
            shell_indices=shell_indices,
            action_count=action_count,
            population_count=population_counts[h],
            alpha=per_statement_alpha,
        )
        centers.append(center.contiguous())
        radii.append(radius)
        q_centers.append(q_center)
        q_radii.append(q_radius)
        compact_shell_m_centers.append(compact_shell_m_center)
        compact_shell_m_radii.append(compact_shell_m_radius)
    return _Pass1Result(
        projection_centers=(centers[0], centers[1]),
        projection_radii=(radii[0], radii[1]),
        q_center=torch.stack(q_centers, dim=0).contiguous(),
        q_radius=torch.stack(q_radii, dim=0).contiguous(),
        compact_shell_m_center=torch.stack(
            compact_shell_m_centers, dim=0
        ).contiguous(),
        compact_shell_m_radius=torch.stack(
            compact_shell_m_radii, dim=0
        ).contiguous(),
        route_window_read_count=window_reads,
        route_tile_read_count=tile_reads,
        action_tile_submission_count=action_submissions,
        shared_shell_gather_tile_count=shell_gathers,
        payload_read_bytes=payload,
        scalar_support_check_count=checks,
    )


@dataclass(frozen=True, slots=True)
class V1LiteralJJointAffineShellResidualPopulationStreamQReceipt:
    registration_hash: str
    forecast_hash: str
    row_group_commit_receipt_hash: str
    joint_direction_axis_hash: str
    joint_direction_ids: tuple[str, ...]
    route_count: int
    stratum_ids: tuple[str, str]
    residual_population_counts: tuple[int, int]
    observed_counts: tuple[int, int]
    q_value_hash: str
    familywise_alpha: float
    simultaneous_statement_count: int
    per_statement_two_sided_alpha: float
    compact_q_scalar_receipt_count: int
    compact_q_scalar_d2h_count: int
    compact_q_scalar_d2h_bytes: int
    production_cuda_execution: bool
    private_cpu_reference_execution: bool
    exact_fixed_time_wor_serfling_intervals: bool = True
    outward_numerical_mass_included: bool = True
    sampling_radius_included: bool = True
    uncancelled_projection_mass_propagated: bool = True
    route_scale_d2h: bool = False
    population_authority: bool = False
    d_face_authority: bool = False
    c_k_o_commit_terminal_or_pareto_authority: bool = False
    schema: str = LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_Q_RECEIPT_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _Q_TOKEN:
            raise PermissionError("joint residual Q receipts require their issuer")
        for name in (
            "registration_hash",
            "forecast_hash",
            "row_group_commit_receipt_hash",
            "joint_direction_axis_hash",
            "q_value_hash",
        ):
            _identity(getattr(self, name), name=name)
        directions = _identity_axis(self.joint_direction_ids, name="joint_direction_ids")
        routes = _positive_int(self.route_count, name="route_count")
        populations = _count_pair(self.residual_population_counts, name="residual_population_counts")
        observed = _count_pair(self.observed_counts, name="observed_counts")
        scalar_count = _positive_int(self.compact_q_scalar_receipt_count, name="compact_q_scalar_receipt_count")
        d2h_count = _nonnegative_int(self.compact_q_scalar_d2h_count, name="compact_q_scalar_d2h_count")
        statements = _positive_int(self.simultaneous_statement_count, name="simultaneous_statement_count")
        alpha = _finite_alpha(self.familywise_alpha, name="familywise_alpha")
        if (
            self.stratum_ids != ("target", "complement")
            or scalar_count != 4 * len(directions)
            or self.production_cuda_execution == self.private_cpu_reference_execution
            or d2h_count != (scalar_count if self.production_cuda_execution else 0)
            or self.compact_q_scalar_d2h_bytes != d2h_count * _F64_BYTES
            or statements != 2 * len(directions) * (routes + 1)
            or self.per_statement_two_sided_alpha != _divide_probability_down(alpha, statements)
            or any(observed[h] <= 0 or observed[h] > populations[h] for h in (0, 1))
            or any(
                type(item) is not bool or item is not True
                for item in (
                    self.exact_fixed_time_wor_serfling_intervals,
                    self.outward_numerical_mass_included,
                    self.sampling_radius_included,
                    self.uncancelled_projection_mass_propagated,
                )
            )
            or any(
                type(item) is not bool or item is not False
                for item in (
                    self.route_scale_d2h,
                    self.population_authority,
                    self.d_face_authority,
                    self.c_k_o_commit_terminal_or_pareto_authority,
                )
            )
            or self.schema != LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_Q_RECEIPT_SCHEMA
        ):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.AUTHORITY_VIOLATION,
                "joint Q receipt crossed its fixed-time numerical boundary",
            )
        object.__setattr__(self, "joint_direction_ids", directions)
        object.__setattr__(self, "residual_population_counts", populations)
        object.__setattr__(self, "observed_counts", observed)
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {item.name: getattr(self, item.name) for item in fields(self) if not item.name.startswith("_")}
        )

    def _compute_hash(self) -> str:
        return _payload_hash(self._payload(), role="q-receipt")

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash or (expected_hash is not None and observed != expected_hash):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.MUTATED_CUSTODY,
                "joint Q receipt no longer replays",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


class V1LiteralJJointAffineShellResidualPopulationStreamQLease:
    __slots__ = ("_owner", "_token", "_closed")

    def __init__(self, owner: "V1LiteralJJointAffineShellResidualPopulationStreamQDeviceHandle", token: object) -> None:
        self._owner = owner
        self._token = token
        self._closed = False

    def intervals(self) -> tuple[Any, Any]:
        if self._closed:
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.EVICTED,
                "joint Q lease is closed",
            )
        return self._owner._require_live()

    def close(self) -> None:
        if not self._closed:
            self._owner._close_lease(self._token)
            self._closed = True

    def __enter__(self) -> "V1LiteralJJointAffineShellResidualPopulationStreamQLease":
        self.intervals()
        return self

    def __exit__(self, *_: object) -> bool:
        self.close()
        return False

    def __reduce__(self) -> NoReturn:
        raise TypeError("joint Q leases are nonserializable")


class V1LiteralJJointAffineShellResidualPopulationStreamQDeviceHandle:
    __slots__ = ("receipt", "_center", "_radius", "_guards", "_invariants", "_tokens")

    def __init__(self, *, receipt: V1LiteralJJointAffineShellResidualPopulationStreamQReceipt, center: Any, radius: Any, issuance_token: object | None = None) -> None:
        if issuance_token is not _Q_TOKEN:
            raise PermissionError("joint Q handles require their issuer")
        if type(receipt) is not V1LiteralJJointAffineShellResidualPopulationStreamQReceipt:
            raise TypeError("joint Q handle requires its exact receipt")
        receipt.assert_integrity()
        shape = (2, len(receipt.joint_direction_ids))
        center = _require_fp64_tensor(center, name="q_center", shape=shape)
        radius = _require_fp64_tensor(radius, name="q_radius", shape=shape, device=center.device, nonnegative=True)
        self.receipt = receipt
        self._center = center
        self._radius = radius
        self._guards = (center.detach().clone(), radius.detach().clone())
        self._invariants = (_tensor_invariant(center), _tensor_invariant(radius))
        self._tokens: set[object] = set()
        self._require_live()

    def _require_live(self) -> tuple[Any, Any]:
        if self._center is None or self._radius is None:
            _refuse(V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.EVICTED, "joint Q custody was evicted")
        values = (self._center, self._radius)
        if any(_tensor_invariant(value) != invariant or not bool(torch.equal(value, guard)) for value, invariant, guard in zip(values, self._invariants, self._guards, strict=True)):
            _refuse(V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.MUTATED_CUSTODY, "joint Q interval tensor mutated")
        self.receipt.assert_integrity()
        return values

    def borrow(self) -> V1LiteralJJointAffineShellResidualPopulationStreamQLease:
        self._require_live()
        token = object()
        self._tokens.add(token)
        return V1LiteralJJointAffineShellResidualPopulationStreamQLease(self, token)

    def _close_lease(self, token: object) -> None:
        if token not in self._tokens:
            _refuse(V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.STALE_IDENTITY, "joint Q lease token is stale")
        self._require_live()
        self._tokens.remove(token)

    def evict(self) -> None:
        self._require_live()
        if self._tokens:
            _refuse(V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.ACTIVE_TILE, "cannot evict joint Q under an active lease")
        self._center = None
        self._radius = None
        self._guards = ()
        self._invariants = ()

    def __reduce__(self) -> NoReturn:
        raise TypeError("joint Q handles are nonserializable")


@dataclass(frozen=True, slots=True)
class V1LiteralJJointAffineShellResidualPopulationStreamCompactShellMReceipt:
    """Array-free custody receipt for the pass-1 ``(M_h v_d)|_shell`` slice."""

    registration_hash: str
    design_hash: str
    forecast_hash: str
    q_receipt_hash: str
    joint_preoutcome_registration_hash: str
    joint_family_receipt_hash: str
    joint_postsource_custody_axis_hash: str
    population_projection_registration_hash: str
    suffix_prefix_freeze_hash: str
    row_group_plan_hash: str
    row_group_commit_receipt_hash: str
    packet_proof_closure_hashes: tuple[str, str]
    literal_j_shard_receipt_hashes: tuple[str, str]
    literal_j_content_sha256s: tuple[str, str]
    source_measurement_axis_hashes: tuple[str, str]
    source_proof_closure_hash_axes: tuple[tuple[str, ...], tuple[str, ...]]
    source_shard_receipt_hash_axes: tuple[tuple[str, ...], tuple[str, ...]]
    source_content_sha256_axes: tuple[tuple[str, ...], tuple[str, ...]]
    source_measurement_hash_axes: tuple[tuple[str, ...], tuple[str, ...]]
    source_preoutcome_axis_hash: str
    source_result_axis_hash: str
    source_representation: str
    affine_support_receipt_hashes: tuple[str, ...]
    route_support_source_receipt_hash: str
    literal_j_abs_bound_source_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    joint_direction_axis_hash: str
    joint_direction_ids: tuple[str, ...]
    action_count: int
    shell_size: int
    direction_count: int
    shell_indices: tuple[int, ...]
    shell_route_ids: tuple[str, ...]
    mv_coordinate_subset_hash: str
    stratum_ids: tuple[str, str]
    residual_population_counts: tuple[int, int]
    observed_counts: tuple[int, int]
    familywise_alpha: float
    simultaneous_statement_count: int
    per_statement_two_sided_alpha: float
    compact_shell_m_value_hash: str
    compact_scalar_receipt_count: int
    compact_scalar_d2h_count: int
    compact_scalar_d2h_bytes: int
    production_cuda_execution: bool
    private_cpu_reference_execution: bool
    pass2_unopened_at_issuance: bool = True
    source_owned_pass1_output: bool = True
    exact_registered_shell_coordinate_subset: bool = True
    exact_fixed_time_wor_serfling_intervals: bool = True
    outward_numerical_mass_included: bool = True
    empirical_product_uncertainty_propagated: bool = True
    signed_product_range_used_for_sampling: bool = True
    no_projection_rows_exposed: bool = True
    no_new_alpha_allocated: bool = True
    adaptive_stopping_authorized: bool = False
    route_scale_d2h: bool = False
    population_d_face_c_k_o_commit_terminal_or_pareto_authority: bool = False
    schema: str = (
        LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_COMPACT_SHELL_M_RECEIPT_SCHEMA
    )
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _COMPACT_SHELL_M_TOKEN:
            raise PermissionError("compact shell-M receipts require their issuer")
        for name in (
            "registration_hash",
            "design_hash",
            "forecast_hash",
            "q_receipt_hash",
            "joint_preoutcome_registration_hash",
            "joint_family_receipt_hash",
            "joint_postsource_custody_axis_hash",
            "population_projection_registration_hash",
            "suffix_prefix_freeze_hash",
            "row_group_plan_hash",
            "row_group_commit_receipt_hash",
            "route_support_source_receipt_hash",
            "literal_j_abs_bound_source_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "joint_direction_axis_hash",
            "mv_coordinate_subset_hash",
            "compact_shell_m_value_hash",
            "source_preoutcome_axis_hash",
            "source_result_axis_hash",
        ):
            _identity(getattr(self, name), name=name)
        directions = _identity_axis(self.joint_direction_ids, name="joint_direction_ids")
        supports = _identity_axis(
            self.affine_support_receipt_hashes,
            name="affine_support_receipt_hashes",
        )
        proof_hashes = _identity_pair(
            self.packet_proof_closure_hashes,
            name="packet_proof_closure_hashes",
        )
        shard_hashes = _identity_pair(
            self.literal_j_shard_receipt_hashes,
            name="literal_j_shard_receipt_hashes",
        )
        content_hashes = _identity_pair(
            self.literal_j_content_sha256s,
            name="literal_j_content_sha256s",
        )
        source_axes = _identity_pair(
            self.source_measurement_axis_hashes,
            name="source_measurement_axis_hashes",
        )
        proof_hash_axes = _nested_identity_pair(
            self.source_proof_closure_hash_axes,
            name="source_proof_closure_hash_axes",
        )
        shard_hash_axes = _nested_identity_pair(
            self.source_shard_receipt_hash_axes,
            name="source_shard_receipt_hash_axes",
        )
        content_hash_axes = _nested_identity_pair(
            self.source_content_sha256_axes,
            name="source_content_sha256_axes",
        )
        measurement_hash_axes = _nested_identity_pair(
            self.source_measurement_hash_axes,
            name="source_measurement_hash_axes",
        )
        shell_route_ids = _identity_axis(self.shell_route_ids, name="shell_route_ids")
        routes = _positive_int(self.route_count, name="route_count")
        actions = _positive_int(self.action_count, name="action_count")
        shell = _positive_int(self.shell_size, name="shell_size")
        direction_count = _positive_int(self.direction_count, name="direction_count")
        populations = _count_pair(
            self.residual_population_counts,
            name="residual_population_counts",
        )
        observed = _count_pair(self.observed_counts, name="observed_counts")
        _nonnegative_int(self.proposal_generation, name="proposal_generation")
        statements = _positive_int(
            self.simultaneous_statement_count,
            name="simultaneous_statement_count",
        )
        alpha = _finite_alpha(self.familywise_alpha, name="familywise_alpha")
        scalar_count = _positive_int(
            self.compact_scalar_receipt_count,
            name="compact_scalar_receipt_count",
        )
        d2h_count = _nonnegative_int(
            self.compact_scalar_d2h_count,
            name="compact_scalar_d2h_count",
        )
        expected_subset_hash = _payload_hash(
            {
                "joint_direction_axis_hash": self.joint_direction_axis_hash,
                "shell_indices": self.shell_indices,
                "shell_route_ids": shell_route_ids,
                "route_axis_hash": self.route_axis_hash,
                "route_id_axis_hash": self.route_id_axis_hash,
            },
            role="pass1-compact-shell-m-coordinate-subset",
        )
        positive = (
            self.pass2_unopened_at_issuance,
            self.source_owned_pass1_output,
            self.exact_registered_shell_coordinate_subset,
            self.exact_fixed_time_wor_serfling_intervals,
            self.outward_numerical_mass_included,
            self.empirical_product_uncertainty_propagated,
            self.signed_product_range_used_for_sampling,
            self.no_projection_rows_exposed,
            self.no_new_alpha_allocated,
        )
        closed = (
            self.adaptive_stopping_authorized,
            self.route_scale_d2h,
            self.population_d_face_c_k_o_commit_terminal_or_pareto_authority,
        )
        if (
            len(proof_hashes) != 2
            or len(shard_hashes) != 2
            or len(content_hashes) != 2
            or len(source_axes) != 2
            or any(
                len(proof_hash_axes[h]) != len(shard_hash_axes[h])
                or len(shard_hash_axes[h]) != len(content_hash_axes[h])
                or len(content_hash_axes[h]) != len(measurement_hash_axes[h])
                or proof_hashes[h]
                != _collapse_segment_axis(
                    role=f"stratum-{h}-proofs",
                    values=proof_hash_axes[h],
                )
                or shard_hashes[h]
                != _collapse_segment_axis(
                    role=f"stratum-{h}-shards",
                    values=shard_hash_axes[h],
                )
                or content_hashes[h]
                != _collapse_segment_axis(
                    role=f"stratum-{h}-contents",
                    values=content_hash_axes[h],
                )
                or source_axes[h]
                != _collapse_segment_axis(
                    role=f"stratum-{h}-measurements",
                    values=measurement_hash_axes[h],
                )
                for h in (0, 1)
            )
            or self.source_representation
            not in {
                "legacy_single_packet_pair_v1",
                "current_source_segmented_logical_look_v1",
                "current_source_segmented_joint_population_action_axis_v2",
            }
            or len(supports) != actions
            or actions + shell != direction_count
            or len(directions) != direction_count
            or len(shell_route_ids) != shell
            or type(self.shell_indices) is not tuple
            or len(self.shell_indices) != shell
            or len(set(self.shell_indices)) != shell
            or any(type(item) is not int or item < 0 or item >= routes for item in self.shell_indices)
            or self.stratum_ids != ("target", "complement")
            or any(observed[h] <= 0 or observed[h] > populations[h] for h in (0, 1))
            or statements != 2 * direction_count * (routes + 1)
            or self.per_statement_two_sided_alpha
            != _divide_probability_down(alpha, statements)
            or scalar_count != 4 * direction_count * shell
            or self.production_cuda_execution == self.private_cpu_reference_execution
            or d2h_count != (scalar_count if self.production_cuda_execution else 0)
            or self.compact_scalar_d2h_bytes != d2h_count * _F64_BYTES
            or self.mv_coordinate_subset_hash != expected_subset_hash
            or any(type(item) is not bool or item is not True for item in positive)
            or any(type(item) is not bool or item is not False for item in closed)
            or self.schema
            != LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_COMPACT_SHELL_M_RECEIPT_SCHEMA
        ):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.AUTHORITY_VIOLATION,
                "compact shell-M receipt crossed its source-owned pass-1 boundary",
            )
        object.__setattr__(self, "joint_direction_ids", directions)
        object.__setattr__(self, "affine_support_receipt_hashes", supports)
        object.__setattr__(self, "packet_proof_closure_hashes", (proof_hashes[0], proof_hashes[1]))
        object.__setattr__(self, "literal_j_shard_receipt_hashes", (shard_hashes[0], shard_hashes[1]))
        object.__setattr__(self, "literal_j_content_sha256s", (content_hashes[0], content_hashes[1]))
        object.__setattr__(self, "source_measurement_axis_hashes", (source_axes[0], source_axes[1]))
        object.__setattr__(self, "source_proof_closure_hash_axes", proof_hash_axes)
        object.__setattr__(self, "source_shard_receipt_hash_axes", shard_hash_axes)
        object.__setattr__(self, "source_content_sha256_axes", content_hash_axes)
        object.__setattr__(self, "source_measurement_hash_axes", measurement_hash_axes)
        object.__setattr__(self, "shell_route_ids", shell_route_ids)
        object.__setattr__(self, "residual_population_counts", populations)
        object.__setattr__(self, "observed_counts", observed)
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
        return _payload_hash(self._payload(), role="pass1-compact-shell-m-receipt")

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash or (
            expected_hash is not None and observed != expected_hash
        ):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.MUTATED_CUSTODY,
                "compact shell-M receipt no longer replays",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


class V1LiteralJJointAffineShellResidualPopulationStreamCompactShellMLease:
    __slots__ = ("_owner", "_token", "_closed")

    def __init__(
        self,
        owner: "V1LiteralJJointAffineShellResidualPopulationStreamCompactShellMDeviceHandle",
        token: object,
    ) -> None:
        self._owner = owner
        self._token = token
        self._closed = False

    def intervals(self) -> tuple[Any, Any]:
        if self._closed:
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.EVICTED,
                "compact shell-M lease is closed",
            )
        return self._owner._require_live()

    def close(self) -> None:
        if not self._closed:
            self._owner._close_lease(self._token)
            self._closed = True

    def __enter__(
        self,
    ) -> "V1LiteralJJointAffineShellResidualPopulationStreamCompactShellMLease":
        self.intervals()
        return self

    def __exit__(self, *_: object) -> bool:
        self.close()
        return False

    def __reduce__(self) -> NoReturn:
        raise TypeError("compact shell-M leases are nonserializable")


class V1LiteralJJointAffineShellResidualPopulationStreamCompactShellMDeviceHandle:
    __slots__ = ("receipt", "_center", "_radius", "_guards", "_invariants", "_tokens")

    def __init__(
        self,
        *,
        receipt: V1LiteralJJointAffineShellResidualPopulationStreamCompactShellMReceipt,
        center: Any,
        radius: Any,
        issuance_token: object | None = None,
    ) -> None:
        if issuance_token is not _COMPACT_SHELL_M_TOKEN:
            raise PermissionError("compact shell-M handles require their issuer")
        if (
            type(receipt)
            is not V1LiteralJJointAffineShellResidualPopulationStreamCompactShellMReceipt
        ):
            raise TypeError("compact shell-M handle requires its exact receipt")
        receipt.assert_integrity()
        shape = (2, receipt.direction_count, receipt.shell_size)
        center = _require_fp64_tensor(center, name="compact_shell_m_center", shape=shape)
        radius = _require_fp64_tensor(
            radius,
            name="compact_shell_m_radius",
            shape=shape,
            device=center.device,
            nonnegative=True,
        )
        self.receipt = receipt
        self._center = center
        self._radius = radius
        self._guards = (center.detach().clone(), radius.detach().clone())
        self._invariants = (_tensor_invariant(center), _tensor_invariant(radius))
        self._tokens: set[object] = set()
        self._require_live()

    def _require_live(self) -> tuple[Any, Any]:
        if self._center is None or self._radius is None:
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.EVICTED,
                "compact shell-M custody was evicted",
            )
        values = (self._center, self._radius)
        if any(
            _tensor_invariant(value) != invariant
            or not bool(torch.equal(value, guard))
            for value, invariant, guard in zip(
                values, self._invariants, self._guards, strict=True
            )
        ):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.MUTATED_CUSTODY,
                "compact shell-M interval tensor mutated",
            )
        self.receipt.assert_integrity()
        return values

    def borrow(
        self,
    ) -> V1LiteralJJointAffineShellResidualPopulationStreamCompactShellMLease:
        self._require_live()
        token = object()
        self._tokens.add(token)
        return V1LiteralJJointAffineShellResidualPopulationStreamCompactShellMLease(
            self, token
        )

    def _close_lease(self, token: object) -> None:
        if token not in self._tokens:
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.STALE_IDENTITY,
                "compact shell-M lease token is stale",
            )
        self._require_live()
        self._tokens.remove(token)

    def evict(self) -> None:
        self._require_live()
        if self._tokens:
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.ACTIVE_TILE,
                "cannot evict compact shell-M under an active lease",
            )
        self._center = None
        self._radius = None
        self._guards = ()
        self._invariants = ()

    def __reduce__(self) -> NoReturn:
        raise TypeError("compact shell-M handles are nonserializable")


@dataclass(frozen=True, slots=True)
class V1LiteralJJointAffineShellResidualPopulationStreamTileReceipt:
    registration_hash: str
    q_receipt_hash: str
    row_group_commit_receipt_hash: str
    tile_ordinal: int
    stratum_ordinal: int
    stratum_id: str
    source_shard_receipt_hash: str
    source_content_sha256: str
    source_proof_closure_hashes: tuple[str, ...]
    source_shard_receipt_hashes: tuple[str, ...]
    source_content_sha256s: tuple[str, ...]
    source_measurement_hashes: tuple[str, ...]
    source_representation: str
    route_start: int
    route_stop: int
    route_count: int
    direction_count: int
    joint_direction_axis_hash: str
    direction_ids: tuple[str, ...]
    per_statement_two_sided_alpha: float
    statement_coordinate_equation: str = "h*D*(R+1)+d*(R+1)+r_for_MJv;_r=R_for_Q"
    fixed_time_wor_serfling_sampling_radius_included: bool = True
    outward_dot_sum_division_and_underflow_radius_included: bool = True
    uncancelled_projection_and_J_mass_propagated: bool = True
    deterministic_loewner_slicing_input_only: bool = True
    no_entry_input_only: bool = True
    single_stratum_payload_under_route_major_pairing: bool = True
    retained_full_mjv_bank: bool = False
    route_scale_d2h: bool = False
    paired_route_major_no_entry_coefficient_authority: bool = False
    d_face_c_k_o_commit_terminal_or_pareto_authority: bool = False
    schema: str = LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_TILE_RECEIPT_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _TILE_TOKEN:
            raise PermissionError("joint residual tile receipts require their issuer")
        for name in ("registration_hash", "q_receipt_hash", "row_group_commit_receipt_hash", "stratum_id", "source_shard_receipt_hash", "source_content_sha256", "joint_direction_axis_hash"):
            _identity(getattr(self, name), name=name)
        ordinal = _nonnegative_int(self.tile_ordinal, name="tile_ordinal")
        h = _nonnegative_int(self.stratum_ordinal, name="stratum_ordinal")
        start = _nonnegative_int(self.route_start, name="route_start")
        stop = _positive_int(self.route_stop, name="route_stop")
        routes = _positive_int(self.route_count, name="route_count")
        directions = _identity_axis(self.direction_ids, name="direction_ids")
        proof_axis = _identity_axis(
            self.source_proof_closure_hashes,
            name="source_proof_closure_hashes",
        )
        shard_axis = _identity_axis(
            self.source_shard_receipt_hashes,
            name="source_shard_receipt_hashes",
        )
        content_axis = _identity_axis(
            self.source_content_sha256s,
            name="source_content_sha256s",
        )
        measurement_axis = _identity_axis(
            self.source_measurement_hashes,
            name="source_measurement_hashes",
        )
        _finite_alpha(
            self.per_statement_two_sided_alpha,
            name="per_statement_two_sided_alpha",
        )
        if (
            h not in (0, 1)
            or self.stratum_id != ("target", "complement")[h]
            or not start < stop <= routes
            or self.direction_count != len(directions)
            or len(proof_axis) != len(shard_axis)
            or len(shard_axis) != len(content_axis)
            or len(content_axis) != len(measurement_axis)
            or self.source_shard_receipt_hash
            != _collapse_segment_axis(
                role=f"stratum-{h}-shards",
                values=shard_axis,
            )
            or self.source_content_sha256
            != _collapse_segment_axis(
                role=f"stratum-{h}-contents",
                values=content_axis,
            )
            or self.source_representation
            not in {
                "legacy_single_packet_pair_v1",
                "current_source_segmented_logical_look_v1",
                "current_source_segmented_joint_population_action_axis_v2",
            }
            or self.statement_coordinate_equation != "h*D*(R+1)+d*(R+1)+r_for_MJv;_r=R_for_Q"
            or any(
                type(item) is not bool or item is not True
                for item in (
                    self.fixed_time_wor_serfling_sampling_radius_included,
                    self.outward_dot_sum_division_and_underflow_radius_included,
                    self.uncancelled_projection_and_J_mass_propagated,
                    self.deterministic_loewner_slicing_input_only,
                    self.no_entry_input_only,
                    self.single_stratum_payload_under_route_major_pairing,
                )
            )
            or any(
                type(item) is not bool or item is not False
                for item in (
                    self.retained_full_mjv_bank,
                    self.route_scale_d2h,
                    self.paired_route_major_no_entry_coefficient_authority,
                    self.d_face_c_k_o_commit_terminal_or_pareto_authority,
                )
            )
            or self.schema != LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_TILE_RECEIPT_SCHEMA
        ):
            _refuse(V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.AUTHORITY_VIOLATION, "joint residual tile receipt crossed its stream-only boundary")
        object.__setattr__(self, "tile_ordinal", ordinal)
        object.__setattr__(self, "stratum_ordinal", h)
        object.__setattr__(self, "route_start", start)
        object.__setattr__(self, "route_stop", stop)
        object.__setattr__(self, "direction_ids", directions)
        object.__setattr__(self, "source_proof_closure_hashes", proof_axis)
        object.__setattr__(self, "source_shard_receipt_hashes", shard_axis)
        object.__setattr__(self, "source_content_sha256s", content_axis)
        object.__setattr__(self, "source_measurement_hashes", measurement_axis)
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType({item.name: getattr(self, item.name) for item in fields(self) if not item.name.startswith("_")})

    def _compute_hash(self) -> str:
        return _payload_hash(self._payload(), role="tile-receipt")

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash or (expected_hash is not None and observed != expected_hash):
            _refuse(V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.MUTATED_CUSTODY, "joint residual tile receipt no longer replays")
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


class V1LiteralJJointAffineShellResidualPopulationStreamTileLease:
    __slots__ = ("_owner", "_token", "_closed")

    def __init__(self, owner: "V1LiteralJJointAffineShellResidualPopulationStreamTileDeviceHandle", token: object) -> None:
        self._owner = owner
        self._token = token
        self._closed = False

    def intervals(self) -> tuple[Any, Any]:
        if self._closed:
            _refuse(V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.EVICTED, "joint residual tile lease is closed")
        return self._owner._require_live()

    def close(self) -> None:
        if not self._closed:
            self._owner._close_lease(self._token)
            self._closed = True

    def __enter__(self) -> "V1LiteralJJointAffineShellResidualPopulationStreamTileLease":
        self.intervals()
        return self

    def __exit__(self, *_: object) -> bool:
        self.close()
        return False

    def __reduce__(self) -> NoReturn:
        raise TypeError("joint residual tile leases are nonserializable")


class V1LiteralJJointAffineShellResidualPopulationStreamTileDeviceHandle:
    """One mutation-guarded M_J v tile; eviction unlocks stream advance."""

    __slots__ = ("receipt", "_center", "_radius", "_guards", "_invariants", "_tokens", "_release")

    def __init__(
        self,
        *,
        receipt: V1LiteralJJointAffineShellResidualPopulationStreamTileReceipt,
        center: Any,
        radius: Any,
        release: object,
        issuance_token: object | None = None,
    ) -> None:
        if issuance_token is not _TILE_TOKEN:
            raise PermissionError("joint residual tile handles require their issuer")
        if type(receipt) is not V1LiteralJJointAffineShellResidualPopulationStreamTileReceipt or not callable(release):
            raise TypeError("joint residual tile handle requires exact receipt and release callback")
        shape = (receipt.direction_count, receipt.route_stop - receipt.route_start)
        center = _require_fp64_tensor(center, name="mjv_tile_center", shape=shape)
        radius = _require_fp64_tensor(radius, name="mjv_tile_radius", shape=shape, device=center.device, nonnegative=True)
        self.receipt = receipt
        self._center = center
        self._radius = radius
        self._guards = (center.detach().clone(), radius.detach().clone())
        self._invariants = (_tensor_invariant(center), _tensor_invariant(radius))
        self._tokens: set[object] = set()
        self._release = release
        self._require_live()

    def _require_live(self) -> tuple[Any, Any]:
        if self._center is None or self._radius is None:
            _refuse(V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.EVICTED, "joint residual tile was evicted")
        values = (self._center, self._radius)
        if any(_tensor_invariant(value) != invariant or not bool(torch.equal(value, guard)) for value, invariant, guard in zip(values, self._invariants, self._guards, strict=True)):
            _refuse(V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.MUTATED_CUSTODY, "joint residual tile interval tensor mutated")
        self.receipt.assert_integrity()
        return values

    def borrow(self) -> V1LiteralJJointAffineShellResidualPopulationStreamTileLease:
        self._require_live()
        token = object()
        self._tokens.add(token)
        return V1LiteralJJointAffineShellResidualPopulationStreamTileLease(self, token)

    def _close_lease(self, token: object) -> None:
        if token not in self._tokens:
            _refuse(V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.STALE_IDENTITY, "joint residual tile lease token is stale")
        self._require_live()
        self._tokens.remove(token)

    def evict(self) -> None:
        replay_error: Exception | None = None
        try:
            self._require_live()
        except V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusal as error:
            if error.code is V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.EVICTED:
                raise
            replay_error = error
        if self._tokens:
            _refuse(V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.ACTIVE_TILE, "cannot evict joint residual tile under an active lease")
        receipt_hash = self.receipt.receipt_hash
        release = self._release
        self._center = None
        self._radius = None
        self._guards = ()
        self._invariants = ()
        self._release = None
        release(receipt_hash)
        if replay_error is not None:
            raise replay_error

    def __reduce__(self) -> NoReturn:
        raise TypeError("joint residual tile handles are nonserializable")


@dataclass(frozen=True, slots=True)
class _Pass2TileNumerics:
    stratum_ordinal: int
    route_start: int
    route_stop: int
    center: Any = field(repr=False, compare=False)
    radius: Any = field(repr=False, compare=False)
    source_shard_receipt_hash: str
    source_content_sha256: str


class _Pass2Cursor:
    """Route-major target/complement pass-2 cursor with no tile retention."""

    __slots__ = (
        "_attachments",
        "_attachment_groups",
        "_row_slices",
        "_route_count",
        "_projection_centers",
        "_projection_radii",
        "_literal_bound",
        "_projection_bound",
        "_population_counts",
        "_alpha",
        "_design",
        "_window_iterators",
        "_parent_windows",
        "_position",
        "_tile_stop",
        "_next_h",
        "_exhausted",
        "route_window_read_count",
        "route_tile_read_count",
        "payload_read_bytes",
        "scalar_support_check_count",
        "expected_tile_count",
    )

    def __init__(
        self,
        *,
        attachments: object,
        projection_centers: tuple[Any, Any],
        projection_radii: tuple[Any, Any],
        literal_bound: Any,
        projection_bound: Any,
        population_counts: tuple[int, int],
        alpha: float,
        design: V1LiteralJJointAffineShellResidualPopulationStreamCUDADesign,
    ) -> None:
        self._attachments = attachments
        self._attachment_groups = _normalize_literal_j_attachment_groups(attachments)
        route_counts = {
            attachment.receipt.route_count
            for group in self._attachment_groups
            for attachment in group
        }
        if len(route_counts) != 1:
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.ORDER_MISMATCH,
                "segmented pass-2 attachments do not share one route axis",
            )
        self._route_count = next(iter(route_counts))
        row_slices: list[tuple[tuple[int, int], ...]] = []
        for h, group in enumerate(self._attachment_groups):
            offset = 0
            local: list[tuple[int, int]] = []
            for attachment in group:
                stop = offset + len(attachment.receipt.context_ids)
                local.append((offset, stop))
                offset = stop
            if offset != int(projection_centers[h].shape[0]):
                _refuse(
                    V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.ORDER_MISMATCH,
                    "segmented pass-2 context axis does not match pass-1 projections",
                )
            row_slices.append(tuple(local))
        self._row_slices = (row_slices[0], row_slices[1])
        self._projection_centers = projection_centers
        self._projection_radii = projection_radii
        self._literal_bound = literal_bound
        self._projection_bound = projection_bound
        self._population_counts = population_counts
        self._alpha = alpha
        self._design = design
        self._window_iterators: tuple[tuple[Iterator[tuple[int, int, object]], ...], tuple[Iterator[tuple[int, int, object]], ...]] | None = None
        self._parent_windows: list[list[tuple[int, int, object] | None]] = [
            [None] * len(self._attachment_groups[0]),
            [None] * len(self._attachment_groups[1]),
        ]
        self._position = 0
        self._tile_stop: int | None = None
        self._next_h = 0
        self._exhausted = False
        self.route_window_read_count = 0
        self.route_tile_read_count = 0
        self.payload_read_bytes = 0
        self.scalar_support_check_count = 0
        boundaries = tuple(
            sorted(
                {
                    endpoint
                    for group in self._attachment_groups
                    for attachment in group
                    for window in attachment.receipt.route_windows
                    for endpoint in window
                }
            )
        )
        self.expected_tile_count = 2 * sum(
            math.ceil((stop - start) / design.route_tile_size)
            for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
        )

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    @property
    def pass2_unopened(self) -> bool:
        return (
            self._window_iterators is None
            and all(
                all(window is None for window in group)
                for group in self._parent_windows
            )
            and self._position == 0
            and self._tile_stop is None
            and self._next_h == 0
            and not self._exhausted
            and self.route_window_read_count == 0
            and self.route_tile_read_count == 0
            and self.payload_read_bytes == 0
            and self.scalar_support_check_count == 0
        )

    def assert_pass2_unopened(self) -> None:
        if not self.pass2_unopened:
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.PASS2_ALREADY_OPENED,
                "the route-scale M_J v pass has already opened",
            )

    def _ensure_parent_windows(self) -> bool:
        if self._exhausted:
            return False
        if self._window_iterators is None:
            self._window_iterators = tuple(
                tuple(
                    attachment.iter_route_windows(verify=True)
                    for attachment in group
                )
                for group in self._attachment_groups
            )
        for h, group in enumerate(self._attachment_groups):
            for segment_ordinal, _attachment in enumerate(group):
                if self._parent_windows[h][segment_ordinal] is not None:
                    continue
                try:
                    window = next(self._window_iterators[h][segment_ordinal])
                except StopIteration:
                    if self._position == self._route_count:
                        continue
                    _refuse(
                        V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.ORDER_MISMATCH,
                        "one segmented pass-2 source ended before complete route coverage",
                    )
                start, stop, _rows = window
                if start != self._position or stop <= start:
                    _refuse(
                        V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.ORDER_MISMATCH,
                        "one segmented pass-2 route window is not the next contiguous source interval",
                    )
                self._parent_windows[h][segment_ordinal] = window
                self.route_window_read_count += 1
        if self._position == self._route_count:
            if any(
                window is not None
                for group in self._parent_windows
                for window in group
            ):
                _refuse(
                    V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.ORDER_MISMATCH,
                    "a pass-2 source window outlived the registered route axis",
                )
            for group in self._window_iterators:
                for iterator in group:
                    try:
                        next(iterator)
                    except StopIteration:
                        continue
                    _refuse(
                        V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.ORDER_MISMATCH,
                        "a pass-2 segment retained trailing route windows",
                    )
            self._exhausted = True
            return False
        if any(
            window is None
            for group in self._parent_windows
            for window in group
        ):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.ORDER_MISMATCH,
                "segmented target/complement pass-2 route coverage ended asymmetrically",
            )
        if any(
            window is None
            or not (window[0] <= self._position < window[1])
            for group in self._parent_windows
            for window in group
        ):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.ORDER_MISMATCH,
                "one retained pass-2 route window does not contain the shared cursor",
            )
        return True

    def _next_source_tile(
        self,
    ) -> tuple[int, int, int, tuple[object, ...]] | None:
        if not self._ensure_parent_windows():
            return None
        h = self._next_h
        parents = self._parent_windows[h]
        start = self._position
        if h == 0:
            stop = min(
                (
                    start + self._design.route_tile_size,
                    *(
                        window[1]
                        for group in self._parent_windows
                        for window in group
                        if window is not None
                    ),
                )
            )
            self._tile_stop = stop
            self._next_h = 1
        else:
            if self._tile_stop is None:
                raise AssertionError("route-major complement lost its target interval")
            stop = self._tile_stop
        rows: list[object] = []
        for parent in parents:
            assert parent is not None
            parent_start, _parent_stop, parent_rows = parent
            local_start = start - parent_start
            local_stop = stop - parent_start
            rows.append(parent_rows[:, local_start:local_stop])
        if h == 1:
            self._next_h = 0
            self._tile_stop = None
            self._position = stop
            for stratum in (0, 1):
                for segment_ordinal, live in enumerate(
                    self._parent_windows[stratum]
                ):
                    assert live is not None
                    if live[1] == stop:
                        self._parent_windows[stratum][segment_ordinal] = None
        return h, start, stop, tuple(rows)

    def next(self) -> _Pass2TileNumerics | None:
        source = self._next_source_tile()
        if source is None:
            return None
        h, start, stop, parent_rows = source
        projection = self._projection_centers[h]
        projection_radius = self._projection_radii[h]
        route_bound = self._literal_bound[start:stop]
        center_sum = torch.zeros(
            (int(projection.shape[1]), stop - start),
            dtype=torch.float64,
            device=projection.device,
        )
        round_mass = torch.zeros_like(center_sum)
        input_uncertainty = torch.zeros_like(center_sum)
        for rows_source, (row_start, row_stop) in zip(
            parent_rows,
            self._row_slices[h],
            strict=True,
        ):
            rows = _stage_source_tile(rows_source, device=projection.device)
            self.scalar_support_check_count += 1
            if bool((torch.abs(rows) > route_bound.reshape(1, -1)).any().item()):
                _refuse(
                    V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.SUPPORT_VIOLATION,
                    "pass-2 literal-J tile exceeded its mandatory analytic route support",
                )
            segment_projection = projection[row_start:row_stop]
            segment_radius = projection_radius[row_start:row_stop]
            abs_rows = torch.abs(rows)
            center_sum.add_(segment_projection.T @ rows)
            # These masses are deliberately uncancelled.  Neither a small
            # final Mv center nor a small projection center can shrink the FP
            # enclosure, including across transport segments.
            round_mass.add_(torch.abs(segment_projection).T @ abs_rows)
            input_uncertainty.add_(segment_radius.T @ abs_rows)
            self.payload_read_bytes += int(rows.numel()) * _F64_BYTES
        # Account for all dot products plus the additions that merge segment
        # sums.  The division by n_h happens once below, after every segment
        # in the statistical stratum has contributed.
        operations = (
            2 * int(projection.shape[0])
            + 2 * len(self._attachment_groups[h])
            + 1
        )
        relative = _gamma(operations)
        if not math.isfinite(relative) or relative >= 1.0:
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.NUMERICAL_REFUSAL,
                "pass-2 FP64 error factor is not finite below one",
            )
        recovery = _div_up(relative, 1.0 - relative)
        input_uncertainty_upper = _add_up(input_uncertainty, _mul_up(input_uncertainty, recovery))
        sum_radius = _add_up(
            _add_up(_mul_up(round_mass, recovery), input_uncertainty_upper),
            _mul_up(float(operations), _MIN_SUBNORMAL),
        )
        center, numeric_radius = _scale_interval_by_uniform_mean(
            center_sum=center_sum,
            radius_sum=sum_radius,
            count=int(projection.shape[0]),
        )
        factor = _serfling_factor(
            observed=int(projection.shape[0]),
            population=self._population_counts[h],
            alpha=self._alpha,
        )
        if factor == 0.0:
            # An exact residual census has no sampling uncertainty.  The
            # outward multiplication helper deliberately advances every
            # finite product by one ulp, including an exact zero; bypass it
            # here so analytic support magnitudes cannot manufacture a
            # support-dependent statistical radius for a census.
            sampling_radius = torch.zeros_like(center)
        else:
            sampling_radius = _mul_up(
                _mul_up(
                    _mul_up(2.0, factor),
                    self._projection_bound.reshape(-1, 1),
                ),
                route_bound.reshape(1, -1),
            )
        radius = _add_up(numeric_radius, sampling_radius).contiguous()
        self.route_tile_read_count += 1
        shard_hashes = tuple(
            attachment.receipt.receipt_hash
            for attachment in self._attachment_groups[h]
        )
        content_hashes = tuple(
            attachment.receipt.content_sha256
            for attachment in self._attachment_groups[h]
        )
        return _Pass2TileNumerics(
            stratum_ordinal=h,
            route_start=start,
            route_stop=stop,
            center=center.contiguous(),
            radius=radius,
            source_shard_receipt_hash=_collapse_segment_axis(
                role=f"stratum-{h}-shards",
                values=shard_hashes,
            ),
            source_content_sha256=_collapse_segment_axis(
                role=f"stratum-{h}-contents",
                values=content_hashes,
            ),
        )


def _validate_execution_source_closure(
    *,
    registration: V1LiteralJJointAffineShellResidualPopulationStreamCUDARegistration,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    suffix_registration: V1SampledVariableShellDW2SuffixRegistration,
    packet: V1SampledVariableShellDW2SuffixPacket,
    prefix_freeze: V1SampledVariableShellDW2SuffixPrefixFreezeReceipt,
    row_group_plan: V1SampledDualChartRowGroupPlan,
    row_group_commit: V1SampledDualChartRowGroupCommit,
    packet_proof_closures: tuple[
        V1SampledDualChartRowGroupPacketProofClosure,
        V1SampledDualChartRowGroupPacketProofClosure,
    ],
    design: V1LiteralJJointAffineShellResidualPopulationStreamCUDADesign,
) -> tuple[
    tuple[V1SampledDualChartRowShardAttachment, V1SampledDualChartRowShardAttachment],
    tuple[str, str],
]:
    exact = (
        (registration, V1LiteralJJointAffineShellResidualPopulationStreamCUDARegistration),
        (population_registration, V1LiteralJDFixedResidualPopulationProjectionRegistration),
        (suffix_registration, V1SampledVariableShellDW2SuffixRegistration),
        (packet, V1SampledVariableShellDW2SuffixPacket),
        (prefix_freeze, V1SampledVariableShellDW2SuffixPrefixFreezeReceipt),
        (row_group_plan, V1SampledDualChartRowGroupPlan),
        (row_group_commit, V1SampledDualChartRowGroupCommit),
        (design, V1LiteralJJointAffineShellResidualPopulationStreamCUDADesign),
    )
    if any(type(value) is not owner for value, owner in exact):
        raise TypeError("joint residual-stream execution received an inexact owner")
    closures = tuple(packet_proof_closures)
    if len(closures) != 2 or any(type(item) is not V1SampledDualChartRowGroupPacketProofClosure for item in closures):
        raise TypeError("joint residual stream requires exact target/complement proof closures")
    registration.assert_integrity()
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
        raise V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusal(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.MUTATED_CUSTODY,
            "published dual-chart shard custody failed its live witness",
        ) from error
    commit = row_group_commit.receipt
    manifest = row_group_commit.attachment.manifest
    plans = tuple(row_group_plan.packet_plans)
    attachments = tuple(row_group_commit.attachment.literal_j_attachments)
    if (
        len(plans) != 2
        or len(attachments) != 2
        or any(type(item) is not V1SampledDualChartRowShardAttachment for item in attachments)
        or registration.population_projection_registration_hash != population_registration.registration_hash
        or registration.suffix_registration_hash != suffix_registration.receipt_hash
        or registration.suffix_packet_hash != packet.receipt_hash
        or registration.suffix_prefix_freeze_hash != prefix_freeze.receipt_hash
        or registration.row_group_plan_hash != row_group_plan.group_plan_hash
        or registration.design_hash != design.design_hash
        or commit.group_plan_hash != row_group_plan.group_plan_hash
        or commit.packet_plan_hashes != tuple(item.packet_plan_hash for item in plans)
        or commit.packet_count != 2
        or commit.role_shard_count != 4
        or commit.proof_closure_hashes != tuple(item.closure_hash for item in closures)
        or commit.provider_proof_hashes != tuple(item.provider_proof_hash for item in closures)
        or tuple(item.packet_ordinal for item in closures) != (0, 1)
        or tuple(item.packet_plan_hash for item in closures) != registration.scorer_packet_plan_hashes
        or tuple(item.provider_plan_identity_hash for item in closures) != registration.scorer_provider_plan_hashes
        or manifest.manifest_hash != commit.manifest_hash
        or manifest.chart_identity_hash != registration.chart_identity_hash
        or manifest.current_mask_hash != registration.current_mask_hash
        or manifest.proposal_generation != registration.proposal_generation
        or manifest.route_axis_hash != registration.route_axis_hash
        or manifest.route_id_axis_hash != registration.route_id_axis_hash
        or manifest.route_count != registration.route_count
        or manifest.literal_j_stream_identity_hash != registration.literal_j_stream_source_hash
    ):
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.STALE_IDENTITY,
            "registered scorer plan, proof closure, commit, or suffix parent diverged",
        )
    measurement_hashes: list[str] = []
    for h, (plan, closure, look, attachment) in enumerate(
        zip(plans, closures, manifest.looks, attachments, strict=True)
    ):
        shard = attachment.receipt
        expected_stratum = ("target", "complement")[h]
        if (
            closure.group_plan_hash != row_group_plan.group_plan_hash
            or closure.packet_plan_hash != plan.packet_plan_hash
            or closure.provider_plan_identity_hash != plan.provider_plan_identity_hash
            or look.literal_j_shard.receipt_hash != shard.receipt_hash
            or shard.role is not V1SampledDualChartRowRole.LITERAL_RELAXED_GATE_JACOBIAN
            or shard.context_ids != plan.context_ids
            or shard.draw_ids != plan.draw_ids
            or shard.stratum_labels != (expected_stratum,) * len(plan.context_ids)
            or shard.source_capture_identity_hash != plan.source_capture_identity_hash
            or look.corrected_a_shard.source_capture_identity_hash != shard.source_capture_identity_hash
            or shard.stream_identity_hash != registration.literal_j_stream_source_hash
            or shard.chart_identity_hash != registration.chart_identity_hash
            or shard.current_mask_hash != registration.current_mask_hash
            or shard.proposal_generation != registration.proposal_generation
            or shard.route_axis_hash != registration.route_axis_hash
            or shard.route_id_axis_hash != registration.route_id_axis_hash
            or shard.route_count != registration.route_count
            or _axis_hash(f"{expected_stratum}-contexts", shard.context_ids)
            != registration.scorer_context_axis_hashes[h]
            or _axis_hash(f"{expected_stratum}-draws", shard.draw_ids)
            != registration.scorer_draw_axis_hashes[h]
        ):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.STALE_IDENTITY,
                "one raw literal-J shard left its typed target/complement scorer packet",
            )
        measurement_hashes.append(
            _payload_hash(
                {
                    "registration_hash": registration.registration_hash,
                    "stratum_id": expected_stratum,
                    "shard_receipt_hash": shard.receipt_hash,
                    "content_sha256": shard.content_sha256,
                    "context_ids": shard.context_ids,
                    "draw_ids": shard.draw_ids,
                    "source_capture_identity_hash": shard.source_capture_identity_hash,
                    "provider_proof_closure_hash": closure.closure_hash,
                },
                role="source-measurement-axis",
            )
        )
    return (attachments[0], attachments[1]), (measurement_hashes[0], measurement_hashes[1])


@dataclass(frozen=True, slots=True)
class _SegmentedExecutionSourceClosure:
    attachments: tuple[
        tuple[V1SampledDualChartRowShardAttachment, ...],
        tuple[V1SampledDualChartRowShardAttachment, ...],
    ] = field(repr=False, compare=False)
    proof_closures: tuple[V1SampledDualChartRowGroupPacketProofClosure, ...] = field(
        repr=False,
        compare=False,
    )
    row_group_commit: V1SampledDualChartRowGroupCommit = field(
        repr=False,
        compare=False,
    )
    proof_hash_axes: tuple[tuple[str, ...], tuple[str, ...]]
    shard_hash_axes: tuple[tuple[str, ...], tuple[str, ...]]
    content_hash_axes: tuple[tuple[str, ...], tuple[str, ...]]
    measurement_hash_axes: tuple[tuple[str, ...], tuple[str, ...]]
    stratum_measurement_axis_hashes: tuple[str, str]
    result: V1SampledV7SharedDW2LiteralJLogicalLookResult = field(
        repr=False,
        compare=False,
    )


def _validate_segmented_execution_source_closure(
    *,
    registration: V1LiteralJJointAffineShellResidualPopulationStreamCUDARegistration,
    population_registration: _PopulationMeasureRegistration,
    logical_look_result: V1SampledV7SharedDW2LiteralJLogicalLookResult,
    design: V1LiteralJJointAffineShellResidualPopulationStreamCUDADesign,
) -> _SegmentedExecutionSourceClosure:
    if (
        type(registration)
        is not V1LiteralJJointAffineShellResidualPopulationStreamCUDARegistration
        or type(population_registration)
        not in {
            V1LiteralJDFixedResidualPopulationProjectionRegistration,
            V1LiteralJJointAxisResidualPopulationMeasureRegistration,
        }
        or type(logical_look_result)
        is not V1SampledV7SharedDW2LiteralJLogicalLookResult
        or type(design)
        is not V1LiteralJJointAffineShellResidualPopulationStreamCUDADesign
    ):
        raise TypeError("segmented residual execution received an inexact owner")
    registration.assert_integrity()
    population_registration.assert_integrity()
    design.assert_integrity()
    try:
        result_hash = logical_look_result.assert_live()
    except Exception as error:
        raise V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusal(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.MUTATED_CUSTODY,
            "segmented logical-look result failed its live source witness",
        ) from error
    source = logical_look_result.source
    source_hash = source.assert_live()
    source_receipt = source.receipt
    result_receipt = logical_look_result.receipt
    result_receipt.assert_integrity()
    acquisition = logical_look_result.acquisition
    commit = logical_look_result.group_commit
    group_plan = source.coordinator.acquisition_plan.group_plan
    plans = tuple(group_plan.packet_plans)
    closures = tuple(logical_look_result.packet_proof_closures)
    views = logical_look_result.stratum_views
    if (
        source._bound_result is not logical_look_result
        or logical_look_result.execution_result
        is not source.coordinator._result
        or logical_look_result.acquisition is not acquisition
        or logical_look_result.group_commit is not commit
        or logical_look_result.packet_proof_closures != closures
        or result_hash != logical_look_result.result_hash
        or registration.source_representation
        not in {
            "current_source_segmented_logical_look_v1",
            "current_source_segmented_joint_population_action_axis_v2",
        }
        or registration._logical_look_source_owner is not source
        or registration.logical_look_source_receipt_hash != source_hash
        or result_receipt.source_receipt_hash != source_hash
        or registration.population_projection_registration_hash
        != population_registration.registration_hash
        or registration.design_hash != design.design_hash
        or registration.row_group_plan_hash != group_plan.group_plan_hash
        or registration.suffix_registration_hash
        != source_receipt.suffix_registration_hash
        or registration.suffix_prefix_freeze_hash
        != source_receipt.generalized_logical_prefix_identity_hash
        or registration.generalized_logical_prefix_identity_hash
        != source_receipt.generalized_logical_prefix_identity_hash
        or registration.source_physical_call_offset
        != source_receipt.physical_call_offset
        or registration.source_expected_prior_manifest_hash
        != source_receipt.expected_prior_manifest_hash
        or registration.source_physical_roles != source_receipt.physical_roles
        or registration.source_physical_context_counts
        != source_receipt.physical_context_counts
        or registration.source_packet_plan_hashes
        != tuple(plan.packet_plan_hash for plan in plans)
        or registration.source_provider_plan_hashes
        != tuple(plan.provider_plan_identity_hash for plan in plans)
        or registration.source_capture_identity_hashes
        != source_receipt.source_capture_identity_hashes
        or registration.source_transport_packet_hashes
        != source_receipt.transport_packet_hashes
        or result_receipt.group_commit_receipt_hash != commit.receipt.receipt_hash
        or result_receipt.physical_call_offset != registration.source_physical_call_offset
        or result_receipt.physical_roles != registration.source_physical_roles
        or result_receipt.physical_context_counts
        != registration.source_physical_context_counts
        or len(closures) != len(plans)
        or views != (logical_look_result.target, logical_look_result.complement)
    ):
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.STALE_IDENTITY,
            "segmented result is not the exact bound outcome of this preoutcome source",
        )
    attachment_groups = (
        tuple(logical_look_result.target.shards),
        tuple(logical_look_result.complement.shards),
    )
    ordinals = (
        tuple(logical_look_result.target.packet_ordinals),
        tuple(logical_look_result.complement.packet_ordinals),
    )
    if ordinals != registration.source_stratum_segment_ordinals:
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.ORDER_MISMATCH,
            "segmented result reordered the preregistered statistical strata",
        )
    proof_axes: list[tuple[str, ...]] = []
    shard_axes: list[tuple[str, ...]] = []
    content_axes: list[tuple[str, ...]] = []
    measurement_axes: list[tuple[str, ...]] = []
    for h, role in enumerate(("target", "complement")):
        view = views[h]
        if (
            view.role != role
            or len(view.shards) != len(ordinals[h])
            or len(view.shards) != registration.source_stratum_segment_counts[h]
            or len(view.context_ids) != registration.registered_observed_counts[h]
            or _axis_hash(f"{role}-contexts", tuple(view.context_ids))
            != registration.scorer_context_axis_hashes[h]
            or _axis_hash(f"{role}-draws", tuple(view.draw_ids))
            != registration.scorer_draw_axis_hashes[h]
        ):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.ORDER_MISMATCH,
                "one segmented stratum view changed its preregistered sample axis",
            )
        local_proofs: list[str] = []
        local_shards: list[str] = []
        local_contents: list[str] = []
        local_measurements: list[str] = []
        for shard, packet_ordinal, attachment_index in zip(
            view.shards,
            view.packet_ordinals,
            view.attachment_indices,
            strict=True,
        ):
            plan = plans[packet_ordinal]
            closure = closures[packet_ordinal]
            receipt = shard.receipt
            if (
                type(shard) is not V1SampledDualChartRowShardAttachment
                or type(closure)
                is not V1SampledDualChartRowGroupPacketProofClosure
                or closure.packet_ordinal != packet_ordinal
                or closure.packet_plan_hash != plan.packet_plan_hash
                or closure.provider_plan_identity_hash
                != plan.provider_plan_identity_hash
                or receipt.look_ordinal != attachment_index
                or receipt.role
                is not V1SampledDualChartRowRole.LITERAL_RELAXED_GATE_JACOBIAN
                or receipt.context_ids != plan.context_ids
                or receipt.draw_ids != plan.draw_ids
                or receipt.stratum_labels != (role,) * len(plan.context_ids)
                or receipt.source_capture_identity_hash
                != plan.source_capture_identity_hash
                or receipt.stream_identity_hash
                != registration.literal_j_stream_source_hash
                or receipt.route_axis_hash != registration.route_axis_hash
                or receipt.route_id_axis_hash != registration.route_id_axis_hash
                or receipt.route_count != registration.route_count
            ):
                _refuse(
                    V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.STALE_IDENTITY,
                    "one segmented literal-J shard/proof left its preregistered packet",
                )
            local_proofs.append(closure.closure_hash)
            local_shards.append(receipt.receipt_hash)
            local_contents.append(receipt.content_sha256)
            local_measurements.append(
                _payload_hash(
                    {
                        "registration_hash": registration.registration_hash,
                        "stratum_id": role,
                        "packet_ordinal": packet_ordinal,
                        "attachment_index": attachment_index,
                        "shard_receipt_hash": receipt.receipt_hash,
                        "content_sha256": receipt.content_sha256,
                        "context_ids": receipt.context_ids,
                        "draw_ids": receipt.draw_ids,
                        "source_capture_identity_hash": (
                            receipt.source_capture_identity_hash
                        ),
                        "provider_proof_closure_hash": closure.closure_hash,
                    },
                    role="segmented-source-measurement",
                )
            )
        proof_axes.append(tuple(local_proofs))
        shard_axes.append(tuple(local_shards))
        content_axes.append(tuple(local_contents))
        measurement_axes.append(tuple(local_measurements))
    stratum_measurement_hashes = tuple(
        _collapse_segment_axis(
            role=f"stratum-{h}-measurements",
            values=measurement_axes[h],
        )
        for h in (0, 1)
    )
    return _SegmentedExecutionSourceClosure(
        attachments=attachment_groups,
        proof_closures=closures,
        row_group_commit=commit,
        proof_hash_axes=(proof_axes[0], proof_axes[1]),
        shard_hash_axes=(shard_axes[0], shard_axes[1]),
        content_hash_axes=(content_axes[0], content_axes[1]),
        measurement_hash_axes=(measurement_axes[0], measurement_axes[1]),
        stratum_measurement_axis_hashes=(
            stratum_measurement_hashes[0],
            stratum_measurement_hashes[1],
        ),
        result=logical_look_result,
    )


@dataclass(frozen=True, slots=True)
class V1LiteralJJointAffineShellResidualPopulationStreamCUDAExecutionReceipt:
    registration_hash: str
    forecast_hash: str
    q_receipt_hash: str
    row_group_plan_hash: str
    row_group_commit_receipt_hash: str
    packet_proof_closure_hashes: tuple[str, str]
    literal_j_shard_receipt_hashes: tuple[str, str]
    literal_j_content_sha256s: tuple[str, str]
    source_measurement_axis_hashes: tuple[str, str]
    source_proof_closure_hash_axes: tuple[tuple[str, ...], tuple[str, ...]]
    source_shard_receipt_hash_axes: tuple[tuple[str, ...], tuple[str, ...]]
    source_content_sha256_axes: tuple[tuple[str, ...], tuple[str, ...]]
    source_measurement_hash_axes: tuple[tuple[str, ...], tuple[str, ...]]
    source_segment_counts: tuple[int, int]
    source_preoutcome_axis_hash: str
    source_result_axis_hash: str
    source_representation: str
    joint_preoutcome_registration_hash: str
    joint_direction_axis_hash: str
    joint_direction_ids: tuple[str, ...]
    stratum_ids: tuple[str, str]
    route_count: int
    action_count: int
    shell_size: int
    direction_count: int
    residual_population_counts: tuple[int, int]
    observed_counts: tuple[int, int]
    familywise_alpha: float
    simultaneous_statement_count: int
    per_statement_two_sided_alpha: float
    alpha_spent: float
    exact_residual_census: bool
    source_pass_count: int
    pass1_route_window_read_count: int
    pass1_route_tile_read_count: int
    pass1_action_tile_submission_count: int
    pass1_shared_shell_direct_gather_count: int
    pass1_payload_read_bytes: int
    pass2_route_window_read_count: int
    pass2_route_tile_read_count: int
    pass2_payload_read_bytes: int
    source_payload_read_bytes: int
    host_to_device_payload_bytes: int
    scalar_support_check_count: int
    tile_receipt_hashes: tuple[str, ...]
    production_cuda_execution: bool
    private_cpu_reference_execution: bool
    exact_fixed_time_wor_serfling_family: bool = True
    target_complement_contributions_preserved: bool = True
    every_raw_route_tile_read_once_per_pass_across_all_directions: bool = True
    outward_numerical_and_sampling_radius_per_tile: bool = True
    route_major_target_complement_pass2_order: bool = True
    independent_route_window_partitions_supported: bool = True
    context_route_bank_retained: bool = False
    direction_route_bank_retained: bool = False
    route_scale_d2h: bool = False
    paired_route_major_no_entry_coefficient_authority: bool = False
    cpu_production_fallback: bool = False
    adaptive_stopping_authorized: bool = False
    d_face_c_k_o_commit_terminal_runtime_or_pareto_authority: bool = False
    algorithm: str = LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_ALGORITHM
    authority_scope: str = LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_AUTHORITY
    schema: str = LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_EXECUTION_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _EXECUTION_TOKEN:
            raise PermissionError("joint residual-stream execution receipts require their issuer")
        for name in ("registration_hash", "forecast_hash", "q_receipt_hash", "row_group_plan_hash", "row_group_commit_receipt_hash", "joint_preoutcome_registration_hash", "joint_direction_axis_hash", "source_preoutcome_axis_hash", "source_result_axis_hash"):
            _identity(getattr(self, name), name=name)
        directions = _identity_axis(self.joint_direction_ids, name="joint_direction_ids")
        tile_hashes = _identity_axis(self.tile_receipt_hashes, name="tile_receipt_hashes")
        routes = _positive_int(self.route_count, name="route_count")
        actions = _positive_int(self.action_count, name="action_count")
        shell = _positive_int(self.shell_size, name="shell_size")
        direction_count = _positive_int(self.direction_count, name="direction_count")
        populations = _count_pair(self.residual_population_counts, name="residual_population_counts")
        observed = _count_pair(self.observed_counts, name="observed_counts")
        segment_counts = _count_pair(
            self.source_segment_counts,
            name="source_segment_counts",
        )
        proof_hashes = _identity_pair(
            self.packet_proof_closure_hashes,
            name="packet_proof_closure_hashes",
        )
        shard_hashes = _identity_pair(
            self.literal_j_shard_receipt_hashes,
            name="literal_j_shard_receipt_hashes",
        )
        content_hashes = _identity_pair(
            self.literal_j_content_sha256s,
            name="literal_j_content_sha256s",
        )
        source_axes = _identity_pair(
            self.source_measurement_axis_hashes,
            name="source_measurement_axis_hashes",
        )
        proof_hash_axes = _nested_identity_pair(
            self.source_proof_closure_hash_axes,
            name="source_proof_closure_hash_axes",
        )
        shard_hash_axes = _nested_identity_pair(
            self.source_shard_receipt_hash_axes,
            name="source_shard_receipt_hash_axes",
        )
        content_hash_axes = _nested_identity_pair(
            self.source_content_sha256_axes,
            name="source_content_sha256_axes",
        )
        measurement_hash_axes = _nested_identity_pair(
            self.source_measurement_hash_axes,
            name="source_measurement_hash_axes",
        )
        census = observed == populations
        alpha = _finite_alpha(self.familywise_alpha, name="familywise_alpha")
        statements = _positive_int(self.simultaneous_statement_count, name="simultaneous_statement_count")
        positive = (
            self.exact_fixed_time_wor_serfling_family,
            self.target_complement_contributions_preserved,
            self.every_raw_route_tile_read_once_per_pass_across_all_directions,
            self.outward_numerical_and_sampling_radius_per_tile,
            self.route_major_target_complement_pass2_order,
            self.independent_route_window_partitions_supported,
        )
        closed = (
            self.context_route_bank_retained,
            self.direction_route_bank_retained,
            self.route_scale_d2h,
            self.paired_route_major_no_entry_coefficient_authority,
            self.cpu_production_fallback,
            self.adaptive_stopping_authorized,
            self.d_face_c_k_o_commit_terminal_runtime_or_pareto_authority,
        )
        if (
            self.stratum_ids != ("target", "complement")
            or actions + shell != direction_count
            or direction_count != len(directions)
            or statements != 2 * direction_count * (routes + 1)
            or self.per_statement_two_sided_alpha != _divide_probability_down(alpha, statements)
            or self.exact_residual_census != census
            or self.alpha_spent != (0.0 if census else alpha)
            or self.source_pass_count != 2
            or segment_counts
            != (len(shard_hash_axes[0]), len(shard_hash_axes[1]))
            or any(count <= 0 for count in segment_counts)
            or any(
                len(proof_hash_axes[h]) != segment_counts[h]
                or len(content_hash_axes[h]) != segment_counts[h]
                or len(measurement_hash_axes[h]) != segment_counts[h]
                or proof_hashes[h]
                != _collapse_segment_axis(
                    role=f"stratum-{h}-proofs",
                    values=proof_hash_axes[h],
                )
                or shard_hashes[h]
                != _collapse_segment_axis(
                    role=f"stratum-{h}-shards",
                    values=shard_hash_axes[h],
                )
                or content_hashes[h]
                != _collapse_segment_axis(
                    role=f"stratum-{h}-contents",
                    values=content_hash_axes[h],
                )
                or source_axes[h]
                != _collapse_segment_axis(
                    role=f"stratum-{h}-measurements",
                    values=measurement_hash_axes[h],
                )
                for h in (0, 1)
            )
            or self.source_representation
            not in {
                "legacy_single_packet_pair_v1",
                "current_source_segmented_logical_look_v1",
                "current_source_segmented_joint_population_action_axis_v2",
            }
            or self.pass1_route_window_read_count != self.pass2_route_window_read_count
            or self.pass2_route_tile_read_count != len(tile_hashes)
            or self.pass1_shared_shell_direct_gather_count
            != sum(segment_counts) * shell
            or self.pass1_payload_read_bytes != self.pass2_payload_read_bytes
            or self.source_payload_read_bytes != self.pass1_payload_read_bytes + self.pass2_payload_read_bytes
            or self.host_to_device_payload_bytes != (self.source_payload_read_bytes if self.production_cuda_execution else 0)
            or self.production_cuda_execution == self.private_cpu_reference_execution
            or any(type(item) is not bool or item is not True for item in positive)
            or any(type(item) is not bool or item is not False for item in closed)
            or self.algorithm != LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_ALGORITHM
            or self.authority_scope != LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_AUTHORITY
            or self.schema != LITERAL_J_JOINT_AFFINE_SHELL_RESIDUAL_STREAM_EXECUTION_SCHEMA
        ):
            _refuse(V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.AUTHORITY_VIOLATION, "joint residual-stream execution receipt crossed its bounded numerical authority")
        object.__setattr__(self, "joint_direction_ids", directions)
        object.__setattr__(self, "tile_receipt_hashes", tile_hashes)
        object.__setattr__(self, "source_segment_counts", segment_counts)
        object.__setattr__(self, "source_proof_closure_hash_axes", proof_hash_axes)
        object.__setattr__(self, "source_shard_receipt_hash_axes", shard_hash_axes)
        object.__setattr__(self, "source_content_sha256_axes", content_hash_axes)
        object.__setattr__(self, "source_measurement_hash_axes", measurement_hash_axes)
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType({item.name: getattr(self, item.name) for item in fields(self) if not item.name.startswith("_")})

    def _compute_hash(self) -> str:
        return _payload_hash(self._payload(), role="execution-receipt")

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash or (expected_hash is not None and observed != expected_hash):
            _refuse(V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.MUTATED_CUSTODY, "joint residual-stream execution receipt no longer replays")
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


class V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle:
    """Live two-pass stream; at most one M_J v tile can exist at a time."""

    __slots__ = (
        "registration",
        "forecast",
        "q_handle",
        "compact_shell_m_handle",
        "_cursor",
        "_joint_lease",
        "_support_leases",
        "_route_support_lease",
        "_route_support_result",
        "_private_support",
        "_projection_centers",
        "_projection_radii",
        "_literal_bound",
        "_projection_bound",
        "_row_group_commit",
        "_proof_closures",
        "_attachments",
        "_source_measurement_axis_hashes",
        "_source_proof_hash_axes",
        "_source_shard_hash_axes",
        "_source_content_hash_axes",
        "_source_measurement_hash_axes",
        "_segmented_source_result",
        "_joint_population_source",
        "_joint_population_execution_binding",
        "_pass1",
        "_production",
        "_active_tile_hash",
        "_tile_receipt_hashes",
        "_exhausted",
        "_final_receipt",
        "_external_pass2_claim_token",
        "_pass2_externally_consumed",
        "_evicted",
    )

    def __init__(
        self,
        *,
        registration: V1LiteralJJointAffineShellResidualPopulationStreamCUDARegistration,
        forecast: V1LiteralJJointAffineShellResidualPopulationStreamCUDAHBMForecast,
        q_handle: V1LiteralJJointAffineShellResidualPopulationStreamQDeviceHandle,
        compact_shell_m_handle: V1LiteralJJointAffineShellResidualPopulationStreamCompactShellMDeviceHandle,
        cursor: _Pass2Cursor,
        joint_lease: V1LiteralJJointAffineShellFamilyLease,
        support_leases: tuple[V1LiteralJAffineShellAnalyticSupportLease, ...],
        route_support_lease: _AnalyticSupportSourceLease | None,
        route_support_result: V1LiteralJResidualAnalyticSupportSourceCUDAResult | None,
        private_support: _V1LiteralJJointResidualCPUAnalyticSupport | None,
        projection_centers: tuple[Any, Any],
        projection_radii: tuple[Any, Any],
        literal_bound: Any,
        projection_bound: Any,
        row_group_commit: V1SampledDualChartRowGroupCommit,
        proof_closures: tuple[V1SampledDualChartRowGroupPacketProofClosure, ...],
        attachments: object,
        source_measurement_axis_hashes: tuple[str, str],
        source_proof_hash_axes: tuple[tuple[str, ...], tuple[str, ...]],
        source_shard_hash_axes: tuple[tuple[str, ...], tuple[str, ...]],
        source_content_hash_axes: tuple[tuple[str, ...], tuple[str, ...]],
        source_measurement_hash_axes: tuple[tuple[str, ...], tuple[str, ...]],
        segmented_source_result: V1SampledV7SharedDW2LiteralJLogicalLookResult | None,
        joint_population_source: V1LiteralJJointPopulationSource | None,
        joint_population_execution_binding: V1LiteralJJointAffineShellResidualPopulationStreamCUDAPostsourceBinding | None,
        pass1: _Pass1Result,
        production: bool,
        issuance_token: object | None = None,
    ) -> None:
        if issuance_token is not _EXECUTION_TOKEN:
            raise PermissionError("joint residual stream handles require their issuer")
        if (
            type(registration) is not V1LiteralJJointAffineShellResidualPopulationStreamCUDARegistration
            or type(forecast) is not V1LiteralJJointAffineShellResidualPopulationStreamCUDAHBMForecast
            or type(q_handle) is not V1LiteralJJointAffineShellResidualPopulationStreamQDeviceHandle
            or type(compact_shell_m_handle)
            is not V1LiteralJJointAffineShellResidualPopulationStreamCompactShellMDeviceHandle
            or type(cursor) is not _Pass2Cursor
            or type(joint_lease) is not V1LiteralJJointAffineShellFamilyLease
            or type(row_group_commit) is not V1SampledDualChartRowGroupCommit
            or any(type(item) is not V1SampledDualChartRowGroupPacketProofClosure for item in proof_closures)
            or type(pass1) is not _Pass1Result
            or type(production) is not bool
        ):
            raise TypeError("joint residual stream handle received an inexact parent")
        attachment_groups = _normalize_literal_j_attachment_groups(attachments)
        if (
            tuple(
                tuple(item.receipt.receipt_hash for item in group)
                for group in attachment_groups
            )
            != source_shard_hash_axes
            or any(
                type(axis) is not tuple or not axis
                for axes in (
                    source_proof_hash_axes,
                    source_shard_hash_axes,
                    source_content_hash_axes,
                    source_measurement_hash_axes,
                )
                for axis in axes
            )
            or (
                registration.source_representation
                in {
                    "current_source_segmented_logical_look_v1",
                    "current_source_segmented_joint_population_v1",
                    "current_source_segmented_joint_population_action_axis_v2",
                }
            )
            != (type(segmented_source_result) is V1SampledV7SharedDW2LiteralJLogicalLookResult)
            or (
                registration.source_representation
                == "current_source_segmented_joint_population_v1"
            )
            != (type(joint_population_source) is V1LiteralJJointPopulationSource)
            or (
                registration.source_representation
                == "current_source_segmented_joint_population_action_axis_v2"
            )
            != (
                type(joint_population_execution_binding)
                is V1LiteralJJointAffineShellResidualPopulationStreamCUDAPostsourceBinding
            )
        ):
            raise TypeError("joint residual stream source axes are incomplete")
        self.registration = registration
        self.forecast = forecast
        self.q_handle = q_handle
        self.compact_shell_m_handle = compact_shell_m_handle
        self._cursor = cursor
        self._joint_lease = joint_lease
        self._support_leases = support_leases
        self._route_support_lease = route_support_lease
        self._route_support_result = route_support_result
        self._private_support = private_support
        self._projection_centers = projection_centers
        self._projection_radii = projection_radii
        self._literal_bound = literal_bound
        self._projection_bound = projection_bound
        self._row_group_commit = row_group_commit
        self._proof_closures = proof_closures
        self._attachments = attachments
        self._source_measurement_axis_hashes = source_measurement_axis_hashes
        self._source_proof_hash_axes = source_proof_hash_axes
        self._source_shard_hash_axes = source_shard_hash_axes
        self._source_content_hash_axes = source_content_hash_axes
        self._source_measurement_hash_axes = source_measurement_hash_axes
        self._segmented_source_result = segmented_source_result
        self._joint_population_source = joint_population_source
        self._joint_population_execution_binding = joint_population_execution_binding
        self._pass1 = pass1
        self._production = production
        self._active_tile_hash: str | None = None
        self._tile_receipt_hashes: list[str] = []
        self._exhausted = False
        self._final_receipt: V1LiteralJJointAffineShellResidualPopulationStreamCUDAExecutionReceipt | None = None
        self._external_pass2_claim_token: object | None = None
        self._pass2_externally_consumed = False
        self._evicted = False
        self.assert_live()

    def assert_live(self) -> None:
        if self._evicted:
            _refuse(V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.EVICTED, "joint residual stream was evicted")
        self.registration.assert_integrity()
        self.forecast.assert_integrity()
        self.q_handle._require_live()
        self.compact_shell_m_handle._require_live()
        try:
            self._joint_lease.projection_custody()
            for lease in self._support_leases:
                lease.support_vector()
            if self._route_support_result is not None:
                self._route_support_result.source_handle.assert_live()
            if self._private_support is not None:
                self._private_support.assert_live()
            self._row_group_commit.attachment.verify_live_witnesses()
            if self._segmented_source_result is not None:
                self._segmented_source_result.assert_live()
            if self._joint_population_source is not None:
                self._joint_population_source.assert_live()
            if self._joint_population_execution_binding is not None:
                self._joint_population_execution_binding.assert_live()
        except V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusal:
            raise
        except Exception as error:
            raise V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusal(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.MUTATED_CUSTODY,
                "one retained joint residual parent is stale or mutated",
            ) from error

    @property
    def exhausted(self) -> bool:
        self.assert_live()
        return self._exhausted

    @property
    def active_tile(self) -> bool:
        self.assert_live()
        return self._active_tile_hash is not None

    @property
    def pass2_unopened(self) -> bool:
        self.assert_live()
        return (
            self._cursor.pass2_unopened
            and self._external_pass2_claim_token is None
            and not self._pass2_externally_consumed
        )

    def assert_pass2_unopened(self) -> None:
        """Require source-owned proof that no route-scale pass-2 read occurred."""

        self.assert_live()
        if self._external_pass2_claim_token is not None or self._pass2_externally_consumed:
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.PASS2_ALREADY_OPENED,
                "the route-scale source is claimed or consumed by its population-Mv successor",
            )
        self._cursor.assert_pass2_unopened()

    def _claim_unopened_pass2_for_population_mjv(self) -> object:
        """Transfer exclusive pass-2 read custody to the population-Mv owner."""

        return self._claim_unopened_route_stream_for_population_mjv()

    def _assert_population_mjv_pass2_claim(self, token: object) -> None:
        self._assert_population_mjv_route_claim(token)

    def _release_population_mjv_pass2_claim(
        self,
        token: object,
        *,
        source_rows_consumed: bool,
    ) -> None:
        """Compatibility release for the historical opaque-row claimant.

        The legacy claimant did not advance through this owner's guarded tile
        issuer, so a positive consumption declaration remains terminal and
        cannot authorize this handle's final receipt.  New population-Mv
        consumers must use the route-stream methods below.
        """

        self._assert_population_mjv_route_claim(token)
        if type(source_rows_consumed) is not bool:
            raise TypeError("external pass-2 consumption flag must be exact")
        if self._active_tile_hash is not None:
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.ACTIVE_TILE,
                "cannot release pass 2 while a guarded route tile remains live",
            )
        if not source_rows_consumed:
            self._cursor.assert_pass2_unopened()
        self._external_pass2_claim_token = None
        self._pass2_externally_consumed = source_rows_consumed

    def _claim_unopened_route_stream_for_population_mjv(self) -> object:
        """Claim the exact guarded pass-2 tile stream before its first read.

        This is a process-local capability, not a durable receipt.  Public
        ``next_tile`` remains closed while the capability is live, preventing
        interleaved route reads by two consumers.
        """

        self.assert_pass2_unopened()
        token = object()
        self._external_pass2_claim_token = token
        return token

    def _assert_population_mjv_route_claim(self, token: object) -> None:
        """Replay one live exclusive population-Mv route-stream claim."""

        self.assert_live()
        if (
            token is not self._external_pass2_claim_token
            or self._pass2_externally_consumed
        ):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.STALE_IDENTITY,
                "population-Mv route-stream claim is stale or already consumed",
            )

    def _next_claimed_population_mjv_tile(
        self,
        token: object,
    ) -> V1LiteralJJointAffineShellResidualPopulationStreamTileDeviceHandle | None:
        """Advance pass 2 under the exact exclusive population-Mv claim."""

        self._assert_population_mjv_route_claim(token)
        return self._next_tile(allow_population_mjv_claim=True)

    def _release_population_mjv_route_claim(
        self,
        token: object,
        *,
        source_tiles_consumed: bool,
    ) -> None:
        """Return or complete claimed route-stream custody.

        ``False`` is a clean rollback and is legal only before the first tile
        read.  ``True`` is legal only after the exact claimed iterator reached
        ``None`` and every issued tile was evicted.  Because claimed iteration
        uses this owner's ordinary guarded tile ledger, successful completion
        preserves normal finalization semantics.
        """

        self._assert_population_mjv_route_claim(token)
        if type(source_tiles_consumed) is not bool:
            raise TypeError("claimed source-tile consumption flag must be exact")
        if self._active_tile_hash is not None:
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.ACTIVE_TILE,
                "cannot release a population-Mv claim with one live route tile",
            )
        if source_tiles_consumed:
            if not self._exhausted:
                _refuse(
                    V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.STREAM_NOT_EXHAUSTED,
                    "claimed route-stream completion requires iteration through None",
                )
        else:
            self._cursor.assert_pass2_unopened()
        self._external_pass2_claim_token = None

    def _abort_population_mjv_route_claim(
        self,
        token: object,
        *,
        source_tiles_consumed: bool,
    ) -> None:
        """Burn a claimed pass-2 route stream after partial consumption.

        A clean unread claim still rolls back.  Once any route tile has been
        read, cleanup may not leave the claim token stranded and may not make
        the source retryable; the source is marked externally consumed without
        granting legacy finalization authority.
        """

        self._assert_population_mjv_route_claim(token)
        if type(source_tiles_consumed) is not bool:
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.INVALID_TYPE,
                "claimed source-tile consumption flag must be exact",
            )
        if self._active_tile_hash is not None:
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.ACTIVE_TILE,
                "cannot abort a population-Mv claim with one live route tile",
            )
        observed_consumption = self._cursor.route_tile_read_count > 0
        if observed_consumption != source_tiles_consumed:
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.STALE_IDENTITY,
                "population-Mv abort declaration differs from source custody",
            )
        if not source_tiles_consumed:
            self._cursor.assert_pass2_unopened()
        self._external_pass2_claim_token = None
        if source_tiles_consumed:
            self._pass2_externally_consumed = True

    def next_tile(self) -> V1LiteralJJointAffineShellResidualPopulationStreamTileDeviceHandle | None:
        self.assert_live()
        if self._external_pass2_claim_token is not None or self._pass2_externally_consumed:
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.PASS2_ALREADY_OPENED,
                "pass 2 belongs to or was consumed by the population-Mv successor",
            )
        return self._next_tile(allow_population_mjv_claim=False)

    def _next_tile(
        self,
        *,
        allow_population_mjv_claim: bool,
    ) -> V1LiteralJJointAffineShellResidualPopulationStreamTileDeviceHandle | None:
        if type(allow_population_mjv_claim) is not bool:
            raise TypeError("population-Mv claim mode must be exact")
        if allow_population_mjv_claim:
            if self._external_pass2_claim_token is None:
                _refuse(
                    V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.STALE_IDENTITY,
                    "claimed tile iteration requires one live route-stream claim",
                )
        elif self._external_pass2_claim_token is not None:
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.PASS2_ALREADY_OPENED,
                "public pass 2 cannot interleave with a population-Mv claimant",
            )
        if self._active_tile_hash is not None:
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.ACTIVE_TILE,
                "the prior M_J v tile must be evicted before stream advancement",
            )
        if self._exhausted:
            return None
        numerics = self._cursor.next()
        if numerics is None:
            self._exhausted = True
            return None
        ordinal = len(self._tile_receipt_hashes)
        h = numerics.stratum_ordinal
        receipt = V1LiteralJJointAffineShellResidualPopulationStreamTileReceipt(
            registration_hash=self.registration.registration_hash,
            q_receipt_hash=self.q_handle.receipt.receipt_hash,
            row_group_commit_receipt_hash=self._row_group_commit.receipt.receipt_hash,
            tile_ordinal=ordinal,
            stratum_ordinal=h,
            stratum_id=("target", "complement")[h],
            source_shard_receipt_hash=numerics.source_shard_receipt_hash,
            source_content_sha256=numerics.source_content_sha256,
            source_proof_closure_hashes=self._source_proof_hash_axes[h],
            source_shard_receipt_hashes=self._source_shard_hash_axes[h],
            source_content_sha256s=self._source_content_hash_axes[h],
            source_measurement_hashes=self._source_measurement_hash_axes[h],
            source_representation=self.registration.source_representation,
            route_start=numerics.route_start,
            route_stop=numerics.route_stop,
            route_count=self.registration.route_count,
            direction_count=self.registration.direction_count,
            joint_direction_axis_hash=self.registration.joint_direction_axis_hash,
            direction_ids=self.registration.joint_direction_ids,
            per_statement_two_sided_alpha=self.registration.per_statement_two_sided_alpha,
            _issuance_token=_TILE_TOKEN,
        )
        receipt_hash = receipt.receipt_hash
        self._active_tile_hash = receipt_hash
        self._tile_receipt_hashes.append(receipt_hash)
        return V1LiteralJJointAffineShellResidualPopulationStreamTileDeviceHandle(
            receipt=receipt,
            center=numerics.center,
            radius=numerics.radius,
            release=self._release_tile,
            issuance_token=_TILE_TOKEN,
        )

    def _release_tile(self, receipt_hash: str) -> None:
        if self._active_tile_hash != receipt_hash:
            _refuse(V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.STALE_IDENTITY, "released M_J v tile is not the active stream tile")
        self._active_tile_hash = None

    def finalize(self) -> V1LiteralJJointAffineShellResidualPopulationStreamCUDAExecutionReceipt:
        self.assert_live()
        if self._external_pass2_claim_token is not None or self._pass2_externally_consumed:
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.PASS2_ALREADY_OPENED,
                "the legacy residual execution cannot finalize an externally consumed pass 2",
            )
        if self._active_tile_hash is not None:
            _refuse(V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.ACTIVE_TILE, "cannot finalize while an M_J v tile remains live")
        if not self._exhausted:
            _refuse(V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.STREAM_NOT_EXHAUSTED, "call next_tile until it returns None before finalization")
        if len(self._tile_receipt_hashes) != self._cursor.expected_tile_count:
            _refuse(V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.STREAM_NOT_EXHAUSTED, "stream tile ledger is incomplete")
        if self._final_receipt is None:
            registration = self.registration
            census = registration.registered_observed_counts == registration.residual_population_counts
            self._final_receipt = V1LiteralJJointAffineShellResidualPopulationStreamCUDAExecutionReceipt(
                registration_hash=registration.registration_hash,
                forecast_hash=self.forecast.forecast_hash,
                q_receipt_hash=self.q_handle.receipt.receipt_hash,
                row_group_plan_hash=registration.row_group_plan_hash,
                row_group_commit_receipt_hash=self._row_group_commit.receipt.receipt_hash,
                packet_proof_closure_hashes=tuple(
                    _collapse_segment_axis(
                        role=f"stratum-{h}-proofs",
                        values=self._source_proof_hash_axes[h],
                    )
                    for h in (0, 1)
                ),
                literal_j_shard_receipt_hashes=tuple(
                    _collapse_segment_axis(
                        role=f"stratum-{h}-shards",
                        values=self._source_shard_hash_axes[h],
                    )
                    for h in (0, 1)
                ),
                literal_j_content_sha256s=tuple(
                    _collapse_segment_axis(
                        role=f"stratum-{h}-contents",
                        values=self._source_content_hash_axes[h],
                    )
                    for h in (0, 1)
                ),
                source_measurement_axis_hashes=self._source_measurement_axis_hashes,
                source_proof_closure_hash_axes=self._source_proof_hash_axes,
                source_shard_receipt_hash_axes=self._source_shard_hash_axes,
                source_content_sha256_axes=self._source_content_hash_axes,
                source_measurement_hash_axes=self._source_measurement_hash_axes,
                source_segment_counts=registration.source_stratum_segment_counts,
                source_preoutcome_axis_hash=(
                    registration.logical_look_source_receipt_hash
                    or registration.suffix_prefix_freeze_hash
                ),
                source_result_axis_hash=(
                    self._segmented_source_result.receipt.receipt_hash
                    if self._segmented_source_result is not None
                    else self._row_group_commit.receipt.receipt_hash
                ),
                source_representation=registration.source_representation,
                joint_preoutcome_registration_hash=registration.joint_preoutcome_registration_hash,
                joint_direction_axis_hash=registration.joint_direction_axis_hash,
                joint_direction_ids=registration.joint_direction_ids,
                stratum_ids=("target", "complement"),
                route_count=registration.route_count,
                action_count=registration.action_count,
                shell_size=registration.shell_size,
                direction_count=registration.direction_count,
                residual_population_counts=registration.residual_population_counts,
                observed_counts=registration.registered_observed_counts,
                familywise_alpha=registration.familywise_alpha,
                simultaneous_statement_count=registration.simultaneous_statement_count,
                per_statement_two_sided_alpha=registration.per_statement_two_sided_alpha,
                alpha_spent=0.0 if census else registration.familywise_alpha,
                exact_residual_census=census,
                source_pass_count=2,
                pass1_route_window_read_count=self._pass1.route_window_read_count,
                pass1_route_tile_read_count=self._pass1.route_tile_read_count,
                pass1_action_tile_submission_count=self._pass1.action_tile_submission_count,
                pass1_shared_shell_direct_gather_count=self._pass1.shared_shell_gather_tile_count,
                pass1_payload_read_bytes=self._pass1.payload_read_bytes,
                pass2_route_window_read_count=self._cursor.route_window_read_count,
                pass2_route_tile_read_count=self._cursor.route_tile_read_count,
                pass2_payload_read_bytes=self._cursor.payload_read_bytes,
                source_payload_read_bytes=self._pass1.payload_read_bytes + self._cursor.payload_read_bytes,
                host_to_device_payload_bytes=(self._pass1.payload_read_bytes + self._cursor.payload_read_bytes) if self._production else 0,
                scalar_support_check_count=self._pass1.scalar_support_check_count + self._cursor.scalar_support_check_count,
                tile_receipt_hashes=tuple(self._tile_receipt_hashes),
                production_cuda_execution=self._production,
                private_cpu_reference_execution=not self._production,
                _issuance_token=_EXECUTION_TOKEN,
            )
        return self._final_receipt

    def evict(self) -> None:
        replay_error: Exception | None = None
        try:
            self.assert_live()
        except V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusal as error:
            if error.code is V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.EVICTED:
                raise
            replay_error = error
        if self._external_pass2_claim_token is not None:
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.ACTIVE_TILE,
                "cannot evict the residual source while population-Mv holds pass-2 custody",
            )
        if self._active_tile_hash is not None:
            _refuse(V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.ACTIVE_TILE, "cannot evict stream while an M_J v tile remains live")
        if self.q_handle._tokens or self.compact_shell_m_handle._tokens:
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.ACTIVE_TILE,
                "cannot evict stream while a compact interval lease remains live",
            )
        compact_error: Exception | None = None
        try:
            self.compact_shell_m_handle.evict()
        except V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusal as error:
            compact_error = error
        q_error: Exception | None = None
        try:
            self.q_handle.evict()
        except V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusal as error:
            if error.code is V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.ACTIVE_TILE:
                raise
            q_error = error
        close_error: Exception | None = None
        for lease in self._support_leases:
            try:
                lease.close()
            except Exception as error:
                if close_error is None:
                    close_error = error
        if self._route_support_lease is not None:
            try:
                self._route_support_lease.close()
            except Exception as error:
                if close_error is None:
                    close_error = error
        try:
            self._joint_lease.close()
        except Exception as error:
            if close_error is None:
                close_error = error
        self._projection_centers = ()
        self._projection_radii = ()
        self._literal_bound = None
        self._projection_bound = None
        self._attachments = ()
        self._proof_closures = ()
        self._segmented_source_result = None
        self._joint_population_source = None
        self._joint_population_execution_binding = None
        self._row_group_commit = None
        self._cursor = None
        self._evicted = True
        if replay_error is not None:
            raise replay_error
        if compact_error is not None:
            raise compact_error
        if q_error is not None:
            raise q_error
        if close_error is not None:
            raise V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusal(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.MUTATED_CUSTODY,
                "a retained parent failed release during stream eviction",
            ) from close_error

    def __reduce__(self) -> NoReturn:
        raise TypeError("joint residual streams are nonserializable")


def _close_execution_parents(
    *,
    support_leases: list[V1LiteralJAffineShellAnalyticSupportLease],
    route_support_lease: _AnalyticSupportSourceLease | None,
    joint_lease: V1LiteralJJointAffineShellFamilyLease | None,
) -> None:
    for lease in support_leases:
        try:
            lease.close()
        except Exception:
            pass
    if route_support_lease is not None:
        try:
            route_support_lease.close()
        except Exception:
            pass
    if joint_lease is not None:
        try:
            joint_lease.close()
        except Exception:
            pass


def _q_value_hash(center: Any, radius: Any) -> str:
    # Q is compact on the action+shell axis; this is never a route-scale D2H.
    center_values = tuple(float(item) for item in center.detach().cpu().reshape(-1).tolist())
    radius_values = tuple(float(item) for item in radius.detach().cpu().reshape(-1).tolist())
    return _payload_hash(
        {
            "center": center_values,
            "radius": radius_values,
            "shape": tuple(int(item) for item in center.shape),
            "outward_interval": True,
        },
        role="compact-q-values",
    )


def _compact_shell_m_value_hash(center: Any, radius: Any) -> str:
    # The only host transfer is the explicitly forecast compact 2 x D x B
    # receipt payload.  Retained context projections never leave source
    # custody, and no route-scale result is materialised.
    center_values = tuple(float(item) for item in center.detach().cpu().reshape(-1).tolist())
    radius_values = tuple(float(item) for item in radius.detach().cpu().reshape(-1).tolist())
    return _payload_hash(
        {
            "center": center_values,
            "radius": radius_values,
            "shape": tuple(int(item) for item in center.shape),
            "outward_interval": True,
            "coordinate_role": "registered-shell-route-subset",
        },
        role="pass1-compact-shell-m-values",
    )


def _validate_execution_crosswalk(
    *,
    registration: V1LiteralJJointAffineShellResidualPopulationStreamCUDARegistration,
    preregistration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    population_registration: _PopulationMeasureRegistration,
    design: V1LiteralJJointAffineShellResidualPopulationStreamCUDADesign,
) -> None:
    registration.assert_integrity()
    preregistration.assert_integrity()
    design.assert_integrity()
    _assert_common_parent_lineage(
        preregistration=preregistration,
        joint_family_handle=joint_family_handle,
        population=population_registration,
    )
    joint = joint_family_handle.receipt
    if (
        registration.joint_preoutcome_registration_hash != preregistration.registration_hash
        or registration.joint_family_receipt_hash != joint.receipt_hash
        or registration.joint_postsource_custody_axis_hash != joint.postsource_custody_axis_hash
        or registration.population_projection_registration_hash != population_registration.registration_hash
        or registration.design_hash != design.design_hash
        or registration.joint_direction_ids != preregistration.joint_direction_ids
        or registration.joint_direction_axis_hash != preregistration.joint_direction_axis_hash
        or registration.familywise_alpha != preregistration.familywise_alpha
        or registration.simultaneous_statement_count != preregistration.simultaneous_statement_count
        or registration.per_statement_two_sided_alpha != preregistration.per_statement_two_sided_alpha
    ):
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.STALE_IDENTITY,
            "execution crosswalk attempted to alter the preregistered joint axis, denominator, or alpha",
        )


def _execute_v1_literal_j_joint_affine_shell_residual_population_stream(
    *,
    registration: V1LiteralJJointAffineShellResidualPopulationStreamCUDARegistration,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    population_registration: _PopulationMeasureRegistration,
    suffix_registration: V1SampledVariableShellDW2SuffixRegistration | None,
    packet: V1SampledVariableShellDW2SuffixPacket | None,
    prefix_freeze: V1SampledVariableShellDW2SuffixPrefixFreezeReceipt | None,
    row_group_plan: V1SampledDualChartRowGroupPlan | None,
    row_group_commit: V1SampledDualChartRowGroupCommit | None,
    packet_proof_closures: tuple[
        V1SampledDualChartRowGroupPacketProofClosure,
        V1SampledDualChartRowGroupPacketProofClosure,
    ] | None,
    logical_look_result: V1SampledV7SharedDW2LiteralJLogicalLookResult | None,
    joint_population_source: V1LiteralJJointPopulationSource | None,
    joint_population_execution_binding: V1LiteralJJointAffineShellResidualPopulationStreamCUDAPostsourceBinding | None,
    design: V1LiteralJJointAffineShellResidualPopulationStreamCUDADesign,
    hbm_forecast: V1LiteralJJointAffineShellResidualPopulationStreamCUDAHBMForecast,
    affine_supports: tuple[V1LiteralJAffineShellAnalyticSupportDeviceHandle, ...],
    route_support_source: V1LiteralJResidualAnalyticSupportSourceCUDAResult | None,
    private_support: _V1LiteralJJointResidualCPUAnalyticSupport | None,
    production: bool,
    failure_after_first_real_tile: bool,
) -> V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle:
    if type(production) is not bool or type(failure_after_first_real_tile) is not bool:
        raise TypeError("joint residual execution mode flags must be exact booleans")
    _validate_execution_crosswalk(
        registration=registration,
        preregistration=joint_preoutcome_registration,
        joint_family_handle=joint_family_handle,
        population_registration=population_registration,
        design=design,
    )
    if joint_population_execution_binding is not None:
        if (
            type(joint_population_execution_binding)
            is not V1LiteralJJointAffineShellResidualPopulationStreamCUDAPostsourceBinding
            or joint_population_source is not None
            or registration.source_representation
            != "current_source_segmented_joint_population_action_axis_v2"
            or joint_population_execution_binding.registration is not registration
            or joint_population_execution_binding.joint_family_handle
            is not joint_family_handle
            or joint_population_execution_binding.preoutcome_plan._preregistration
            is not joint_preoutcome_registration
            or joint_population_execution_binding.preoutcome_plan._population
            is not population_registration
            or joint_population_execution_binding.preoutcome_plan._design is not design
            or joint_population_execution_binding.affine_supports is not affine_supports
            or joint_population_execution_binding.route_support_source
            is not route_support_source
            or joint_population_execution_binding.private_support is not private_support
        ):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.SOURCE_DESCENT_MISMATCH,
                "v2 execution parents do not equal the exact POST binding graph",
            )
        joint_population_execution_binding.assert_live()
        if (
            logical_look_result is not None
            and logical_look_result is not joint_population_execution_binding.logical_result
        ):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.SOURCE_DESCENT_MISMATCH,
                "v2 execution cannot substitute a different logical result",
            )
        logical_look_result = joint_population_execution_binding.logical_result
    if joint_population_source is not None:
        if (
            type(joint_population_source) is not V1LiteralJJointPopulationSource
            or registration.source_representation
            != "current_source_segmented_joint_population_v1"
            or joint_population_source.preoutcome_receipt.receipt_hash
            != registration.joint_population_preoutcome_receipt_hash
            or joint_population_source.receipt.logical_source_receipt_hash
            != registration.logical_look_source_receipt_hash
        ):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.PREFIX_MISMATCH,
                "joint-population execution source is not the registered residual source",
            )
        joint_population_source.assert_live()
        if logical_look_result is not None and logical_look_result is not joint_population_source.logical_result:
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.PREFIX_MISMATCH,
                "joint-population execution cannot substitute a different logical result",
            )
        logical_look_result = joint_population_source.logical_result
    if logical_look_result is None:
        if (
            type(suffix_registration)
            is not V1SampledVariableShellDW2SuffixRegistration
            or type(packet) is not V1SampledVariableShellDW2SuffixPacket
            or type(prefix_freeze)
            is not V1SampledVariableShellDW2SuffixPrefixFreezeReceipt
            or type(row_group_plan) is not V1SampledDualChartRowGroupPlan
            or type(row_group_commit) is not V1SampledDualChartRowGroupCommit
            or type(packet_proof_closures) is not tuple
            or registration.source_representation
            != "legacy_single_packet_pair_v1"
        ):
            raise TypeError("legacy residual execution requires its exact pair graph")
        legacy_attachments, source_axes = _validate_execution_source_closure(
            registration=registration,
            population_registration=population_registration,
            suffix_registration=suffix_registration,
            packet=packet,
            prefix_freeze=prefix_freeze,
            row_group_plan=row_group_plan,
            row_group_commit=row_group_commit,
            packet_proof_closures=packet_proof_closures,
            design=design,
        )
        attachments: object = legacy_attachments
        proof_closure_axis = tuple(packet_proof_closures)
        proof_hash_axes = tuple(
            (proof_closure_axis[h].closure_hash,) for h in (0, 1)
        )
        shard_hash_axes = tuple(
            (legacy_attachments[h].receipt.receipt_hash,) for h in (0, 1)
        )
        content_hash_axes = tuple(
            (legacy_attachments[h].receipt.content_sha256,) for h in (0, 1)
        )
        measurement_hash_axes = tuple((source_axes[h],) for h in (0, 1))
        segmented_result = None
        source_row_group_commit = row_group_commit
    else:
        if any(
            value is not None
            for value in (
                suffix_registration,
                packet,
                prefix_freeze,
                row_group_plan,
                row_group_commit,
                packet_proof_closures,
            )
        ):
            raise TypeError(
                "segmented residual execution cannot mix legacy source parents"
            )
        segmented = _validate_segmented_execution_source_closure(
            registration=registration,
            population_registration=population_registration,
            logical_look_result=logical_look_result,
            design=design,
        )
        attachments = segmented.attachments
        proof_closure_axis = segmented.proof_closures
        proof_hash_axes = segmented.proof_hash_axes
        shard_hash_axes = segmented.shard_hash_axes
        content_hash_axes = segmented.content_hash_axes
        measurement_hash_axes = segmented.measurement_hash_axes
        source_axes = segmented.stratum_measurement_axis_hashes
        segmented_result = segmented.result
        source_row_group_commit = segmented.row_group_commit
    if type(hbm_forecast) is not V1LiteralJJointAffineShellResidualPopulationStreamCUDAHBMForecast:
        raise TypeError("joint residual execution requires its exact preallocation HBM forecast")
    hbm_forecast.assert_integrity()
    if (
        hbm_forecast.design_hash != registration.design_hash
        or hbm_forecast.joint_preoutcome_registration_hash
        != registration.joint_preoutcome_registration_hash
        or hbm_forecast.joint_family_receipt_hash != registration.joint_family_receipt_hash
        or hbm_forecast.observed_counts != registration.registered_observed_counts
        or hbm_forecast.route_count != registration.route_count
        or hbm_forecast.direction_count != registration.direction_count
        or not hbm_forecast.admitted
        or not hbm_forecast.includes_pass1_compact_shell_m
    ):
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.HBM_FORECAST_EXCEEDED,
            "execution is not covered by one admitted exact 85% HBM forecast",
        )
    support_leases: list[V1LiteralJAffineShellAnalyticSupportLease] = []
    route_lease: _AnalyticSupportSourceLease | None = None
    joint_lease: V1LiteralJJointAffineShellFamilyLease | None = None
    q_handle: V1LiteralJJointAffineShellResidualPopulationStreamQDeviceHandle | None = None
    compact_shell_m_handle: V1LiteralJJointAffineShellResidualPopulationStreamCompactShellMDeviceHandle | None = None
    try:
        joint_lease = joint_family_handle.borrow()
        outside_vectors, shell_tensor = joint_lease.projection_custody()
        device = outside_vectors[0].device
        if production:
            if private_support is not None or type(route_support_source) is not V1LiteralJResidualAnalyticSupportSourceCUDAResult:
                _refuse(
                    V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.ANALYTIC_SUPPORT_REQUIRED,
                    "production cannot use private or absent analytic support",
                )
            support_device = _validate_production_analytic_supports(
                preregistration=joint_preoutcome_registration,
                joint_family_handle=joint_family_handle,
                affine_supports=affine_supports,
                route_support_source=route_support_source,
            )
            if support_device != device or device.type != "cuda":
                _refuse(
                    V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.DEVICE_REQUIRED,
                    "production execution parents left the forecast CUDA ordinal",
                )
            if not torch.are_deterministic_algorithms_enabled():
                _refuse(
                    V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.DETERMINISM_DISABLED,
                    "production execution requires deterministic PyTorch algorithms",
                )
            current_free, current_total = torch.cuda.mem_get_info(device)
            current_allocated = torch.cuda.memory_allocated(device)
            current_available = cuda_reusable_memory_bytes(device, framework=torch, free_bytes=current_free)
            if (
                int(current_total) != hbm_forecast.device_total_memory_bytes
                or int(current_allocated) + hbm_forecast.projected_incremental_peak_bytes
                > hbm_forecast.hbm_ceiling_bytes
                or hbm_forecast.projected_incremental_peak_bytes > current_available
            ):
                _refuse(
                    V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.HBM_FORECAST_EXCEEDED,
                    "live preallocation memory state is outside the registered 85% forecast",
                )
            for support in affine_supports:
                support_leases.append(support.borrow())
            support_values = tuple(lease.support_vector() for lease in support_leases)
            shell_support = support_values[0][1:]
            if any(not bool(torch.equal(value[1:], shell_support)) for value in support_values[1:]):
                _refuse(
                    V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.ANALYTIC_SUPPORT_REQUIRED,
                    "per-action analytic supports do not expose one exact shared shell support",
                )
            projection_bound = torch.cat(
                (torch.stack(tuple(value[0] for value in support_values)), shell_support),
                dim=0,
            ).contiguous()
            route_lease = route_support_source.source_handle.borrow()
            literal_bound = route_lease.literal_j_abs_bound
        else:
            if (
                type(private_support) is not _V1LiteralJJointResidualCPUAnalyticSupport
                or route_support_source is not None
                or affine_supports != ()
                or device.type != "cpu"
                or registration.route_count > _CPU_REFERENCE_ROUTE_LIMIT
            ):
                _refuse(
                    V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.CPU_PRODUCTION_FALLBACK,
                    "private reference execution requires only its sealed small CPU support",
                )
            literal_bound, projection_bound = private_support.tensors()
        if (
            shell_tensor.device != device
            or any(item.device != device for item in outside_vectors)
            or literal_bound.device != device
            or projection_bound.device != device
            or _device_identity(device) != hbm_forecast.measurement_device_identity
        ):
            _refuse(
                V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.DEVICE_REQUIRED,
                "directions, shell, analytic support, and forecast must share one exact device ordinal",
            )
        literal_bound = _require_fp64_tensor(
            literal_bound,
            name="literal_j_abs_bound",
            shape=(registration.route_count,),
            device=device,
            nonnegative=True,
        )
        projection_bound = _require_fp64_tensor(
            projection_bound,
            name="joint_projection_abs_bound",
            shape=(registration.direction_count,),
            device=device,
            nonnegative=True,
        )
        pass1 = _pass1_joint_projections(
            attachments=attachments,
            outside_vectors=outside_vectors,
            shell_indices=registration.shell_indices,
            literal_bound=literal_bound,
            projection_bound=projection_bound,
            population_counts=registration.residual_population_counts,
            per_statement_alpha=registration.per_statement_two_sided_alpha,
            design=design,
            failure_after_first_real_tile=failure_after_first_real_tile,
        )
        q_hash = _q_value_hash(pass1.q_center, pass1.q_radius)
        compact_count = 4 * registration.direction_count
        q_receipt = V1LiteralJJointAffineShellResidualPopulationStreamQReceipt(
            registration_hash=registration.registration_hash,
            forecast_hash=hbm_forecast.forecast_hash,
            row_group_commit_receipt_hash=(
                source_row_group_commit.receipt.receipt_hash
            ),
            joint_direction_axis_hash=registration.joint_direction_axis_hash,
            joint_direction_ids=registration.joint_direction_ids,
            route_count=registration.route_count,
            stratum_ids=("target", "complement"),
            residual_population_counts=registration.residual_population_counts,
            observed_counts=registration.registered_observed_counts,
            q_value_hash=q_hash,
            familywise_alpha=registration.familywise_alpha,
            simultaneous_statement_count=registration.simultaneous_statement_count,
            per_statement_two_sided_alpha=registration.per_statement_two_sided_alpha,
            compact_q_scalar_receipt_count=compact_count,
            compact_q_scalar_d2h_count=compact_count if production else 0,
            compact_q_scalar_d2h_bytes=(compact_count * _F64_BYTES) if production else 0,
            production_cuda_execution=production,
            private_cpu_reference_execution=not production,
            _issuance_token=_Q_TOKEN,
        )
        q_handle = V1LiteralJJointAffineShellResidualPopulationStreamQDeviceHandle(
            receipt=q_receipt,
            center=pass1.q_center,
            radius=pass1.q_radius,
            issuance_token=_Q_TOKEN,
        )
        cursor = _Pass2Cursor(
            attachments=attachments,
            projection_centers=pass1.projection_centers,
            projection_radii=pass1.projection_radii,
            literal_bound=literal_bound,
            projection_bound=projection_bound,
            population_counts=registration.residual_population_counts,
            alpha=registration.per_statement_two_sided_alpha,
            design=design,
        )
        cursor.assert_pass2_unopened()
        compact_shell_m_value_hash = _compact_shell_m_value_hash(
            pass1.compact_shell_m_center,
            pass1.compact_shell_m_radius,
        )
        compact_scalar_count = (
            4 * registration.direction_count * registration.shell_size
        )
        mv_coordinate_subset_hash = _payload_hash(
            {
                "joint_direction_axis_hash": registration.joint_direction_axis_hash,
                "shell_indices": registration.shell_indices,
                "shell_route_ids": joint_preoutcome_registration.shell_route_ids,
                "route_axis_hash": registration.route_axis_hash,
                "route_id_axis_hash": registration.route_id_axis_hash,
            },
            role="pass1-compact-shell-m-coordinate-subset",
        )
        compact_shell_m_receipt = (
            V1LiteralJJointAffineShellResidualPopulationStreamCompactShellMReceipt(
                registration_hash=registration.registration_hash,
                design_hash=design.design_hash,
                forecast_hash=hbm_forecast.forecast_hash,
                q_receipt_hash=q_receipt.receipt_hash,
                joint_preoutcome_registration_hash=registration.joint_preoutcome_registration_hash,
                joint_family_receipt_hash=registration.joint_family_receipt_hash,
                joint_postsource_custody_axis_hash=registration.joint_postsource_custody_axis_hash,
                population_projection_registration_hash=registration.population_projection_registration_hash,
                suffix_prefix_freeze_hash=registration.suffix_prefix_freeze_hash,
                row_group_plan_hash=registration.row_group_plan_hash,
                row_group_commit_receipt_hash=(
                    source_row_group_commit.receipt.receipt_hash
                ),
                packet_proof_closure_hashes=tuple(
                    _collapse_segment_axis(
                        role=f"stratum-{h}-proofs",
                        values=proof_hash_axes[h],
                    )
                    for h in (0, 1)
                ),
                literal_j_shard_receipt_hashes=tuple(
                    _collapse_segment_axis(
                        role=f"stratum-{h}-shards",
                        values=shard_hash_axes[h],
                    )
                    for h in (0, 1)
                ),
                literal_j_content_sha256s=tuple(
                    _collapse_segment_axis(
                        role=f"stratum-{h}-contents",
                        values=content_hash_axes[h],
                    )
                    for h in (0, 1)
                ),
                source_measurement_axis_hashes=source_axes,
                source_proof_closure_hash_axes=proof_hash_axes,
                source_shard_receipt_hash_axes=shard_hash_axes,
                source_content_sha256_axes=content_hash_axes,
                source_measurement_hash_axes=measurement_hash_axes,
                source_preoutcome_axis_hash=(
                    registration.logical_look_source_receipt_hash
                    or registration.suffix_prefix_freeze_hash
                ),
                source_result_axis_hash=(
                    segmented_result.receipt.receipt_hash
                    if segmented_result is not None
                    else source_row_group_commit.receipt.receipt_hash
                ),
                source_representation=registration.source_representation,
                affine_support_receipt_hashes=registration.affine_support_receipt_hashes,
                route_support_source_receipt_hash=registration.route_support_source_receipt_hash,
                literal_j_abs_bound_source_hash=registration.literal_j_abs_bound_source_hash,
                model_identity_hash=registration.model_identity_hash,
                chart_identity_hash=registration.chart_identity_hash,
                current_mask_hash=registration.current_mask_hash,
                proposal_generation=registration.proposal_generation,
                route_axis_hash=registration.route_axis_hash,
                route_id_axis_hash=registration.route_id_axis_hash,
                route_count=registration.route_count,
                joint_direction_axis_hash=registration.joint_direction_axis_hash,
                joint_direction_ids=registration.joint_direction_ids,
                action_count=registration.action_count,
                shell_size=registration.shell_size,
                direction_count=registration.direction_count,
                shell_indices=registration.shell_indices,
                shell_route_ids=joint_preoutcome_registration.shell_route_ids,
                mv_coordinate_subset_hash=mv_coordinate_subset_hash,
                stratum_ids=("target", "complement"),
                residual_population_counts=registration.residual_population_counts,
                observed_counts=registration.registered_observed_counts,
                familywise_alpha=registration.familywise_alpha,
                simultaneous_statement_count=registration.simultaneous_statement_count,
                per_statement_two_sided_alpha=registration.per_statement_two_sided_alpha,
                compact_shell_m_value_hash=compact_shell_m_value_hash,
                compact_scalar_receipt_count=compact_scalar_count,
                compact_scalar_d2h_count=compact_scalar_count if production else 0,
                compact_scalar_d2h_bytes=(compact_scalar_count * _F64_BYTES)
                if production
                else 0,
                production_cuda_execution=production,
                private_cpu_reference_execution=not production,
                _issuance_token=_COMPACT_SHELL_M_TOKEN,
            )
        )
        compact_shell_m_handle = (
            V1LiteralJJointAffineShellResidualPopulationStreamCompactShellMDeviceHandle(
                receipt=compact_shell_m_receipt,
                center=pass1.compact_shell_m_center,
                radius=pass1.compact_shell_m_radius,
                issuance_token=_COMPACT_SHELL_M_TOKEN,
            )
        )
        handle = V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle(
            registration=registration,
            forecast=hbm_forecast,
            q_handle=q_handle,
            compact_shell_m_handle=compact_shell_m_handle,
            cursor=cursor,
            joint_lease=joint_lease,
            support_leases=tuple(support_leases),
            route_support_lease=route_lease,
            route_support_result=route_support_source,
            private_support=private_support,
            projection_centers=pass1.projection_centers,
            projection_radii=pass1.projection_radii,
            literal_bound=literal_bound,
            projection_bound=projection_bound,
            row_group_commit=source_row_group_commit,
            proof_closures=proof_closure_axis,
            attachments=attachments,
            source_measurement_axis_hashes=source_axes,
            source_proof_hash_axes=proof_hash_axes,
            source_shard_hash_axes=shard_hash_axes,
            source_content_hash_axes=content_hash_axes,
            source_measurement_hash_axes=measurement_hash_axes,
            segmented_source_result=segmented_result,
            joint_population_source=joint_population_source,
            joint_population_execution_binding=joint_population_execution_binding,
            pass1=pass1,
            production=production,
            issuance_token=_EXECUTION_TOKEN,
        )
        # Ownership of all open leases transferred to the stream.
        support_leases = []
        route_lease = None
        joint_lease = None
        q_handle = None
        compact_shell_m_handle = None
        return handle
    except Exception:
        if compact_shell_m_handle is not None:
            try:
                compact_shell_m_handle.evict()
            except Exception:
                pass
        if q_handle is not None:
            try:
                q_handle.evict()
            except Exception:
                pass
        _close_execution_parents(
            support_leases=support_leases,
            route_support_lease=route_lease,
            joint_lease=joint_lease,
        )
        raise


def execute_v1_literal_j_joint_affine_shell_residual_population_stream_cuda(
    *,
    registration: V1LiteralJJointAffineShellResidualPopulationStreamCUDARegistration,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    suffix_registration: V1SampledVariableShellDW2SuffixRegistration,
    packet: V1SampledVariableShellDW2SuffixPacket,
    prefix_freeze: V1SampledVariableShellDW2SuffixPrefixFreezeReceipt,
    row_group_plan: V1SampledDualChartRowGroupPlan,
    row_group_commit: V1SampledDualChartRowGroupCommit,
    packet_proof_closures: tuple[
        V1SampledDualChartRowGroupPacketProofClosure,
        V1SampledDualChartRowGroupPacketProofClosure,
    ],
    affine_supports: tuple[V1LiteralJAffineShellAnalyticSupportDeviceHandle, ...],
    route_support_source: V1LiteralJResidualAnalyticSupportSourceCUDAResult,
    design: V1LiteralJJointAffineShellResidualPopulationStreamCUDADesign,
    hbm_forecast: V1LiteralJJointAffineShellResidualPopulationStreamCUDAHBMForecast,
) -> V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle:
    """Execute the historical two-shard compatibility path."""

    return _execute_v1_literal_j_joint_affine_shell_residual_population_stream(
        registration=registration,
        joint_preoutcome_registration=joint_preoutcome_registration,
        joint_family_handle=joint_family_handle,
        population_registration=population_registration,
        suffix_registration=suffix_registration,
        packet=packet,
        prefix_freeze=prefix_freeze,
        row_group_plan=row_group_plan,
        row_group_commit=row_group_commit,
        packet_proof_closures=packet_proof_closures,
        logical_look_result=None,
        joint_population_source=None,
        joint_population_execution_binding=None,
        design=design,
        hbm_forecast=hbm_forecast,
        affine_supports=affine_supports,
        route_support_source=route_support_source,
        private_support=None,
        production=True,
        failure_after_first_real_tile=False,
    )


def execute_v1_literal_j_joint_affine_shell_residual_population_stream_cuda_from_logical_look_result(
    *,
    registration: V1LiteralJJointAffineShellResidualPopulationStreamCUDARegistration,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    logical_look_result: V1SampledV7SharedDW2LiteralJLogicalLookResult,
    affine_supports: tuple[V1LiteralJAffineShellAnalyticSupportDeviceHandle, ...],
    route_support_source: V1LiteralJResidualAnalyticSupportSourceCUDAResult,
    design: V1LiteralJJointAffineShellResidualPopulationStreamCUDADesign,
    hbm_forecast: V1LiteralJJointAffineShellResidualPopulationStreamCUDAHBMForecast,
) -> V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle:
    """Historical reader; it cannot prove an action-complete PRE registration."""

    del (
        registration,
        joint_preoutcome_registration,
        joint_family_handle,
        population_registration,
        logical_look_result,
        affine_supports,
        route_support_source,
        design,
        hbm_forecast,
    )
    _refuse(
        V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.PREOUTCOME_PLAN_REQUIRED,
        "authoritative segmented execution requires the exact v2 PRE/POST binding",
    )


def execute_v1_literal_j_joint_affine_shell_residual_population_stream_cuda_from_joint_population_source(
    *,
    registration: V1LiteralJJointAffineShellResidualPopulationStreamCUDARegistration,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    joint_population_source: V1LiteralJJointPopulationSource,
    affine_supports: tuple[V1LiteralJAffineShellAnalyticSupportDeviceHandle, ...],
    route_support_source: V1LiteralJResidualAnalyticSupportSourceCUDAResult,
    design: V1LiteralJJointAffineShellResidualPopulationStreamCUDADesign,
    hbm_forecast: V1LiteralJJointAffineShellResidualPopulationStreamCUDAHBMForecast,
) -> V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle:
    """Historical singular reader; action-complete v2 custody is mandatory."""

    del (
        registration,
        joint_preoutcome_registration,
        joint_family_handle,
        population_registration,
        joint_population_source,
        affine_supports,
        route_support_source,
        design,
        hbm_forecast,
    )
    _refuse(
        V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.PREOUTCOME_PLAN_REQUIRED,
        "a singular source cannot authorize the action-complete v2 residual execution",
    )


def execute_v1_literal_j_joint_affine_shell_residual_population_stream_cuda_from_postsource_binding(
    *,
    binding: V1LiteralJJointAffineShellResidualPopulationStreamCUDAPostsourceBinding,
    hbm_forecast: V1LiteralJJointAffineShellResidualPopulationStreamCUDAHBMForecast,
) -> V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle:
    """Execute only the exact causally frozen v2 production binding."""

    if (
        type(binding)
        is not V1LiteralJJointAffineShellResidualPopulationStreamCUDAPostsourceBinding
    ):
        raise TypeError("v2 residual execution requires its exact POST binding")
    binding.assert_live()
    plan = binding.preoutcome_plan
    return _execute_v1_literal_j_joint_affine_shell_residual_population_stream(
        registration=binding.registration,
        joint_preoutcome_registration=plan._preregistration,
        joint_family_handle=binding.joint_family_handle,
        population_registration=plan._population,
        suffix_registration=None,
        packet=None,
        prefix_freeze=None,
        row_group_plan=None,
        row_group_commit=None,
        packet_proof_closures=None,
        logical_look_result=None,
        joint_population_source=None,
        joint_population_execution_binding=binding,
        design=plan._design,
        hbm_forecast=hbm_forecast,
        affine_supports=binding.affine_supports,
        route_support_source=binding.route_support_source,
        private_support=None,
        production=True,
        failure_after_first_real_tile=False,
    )


def _execute_v1_literal_j_joint_affine_shell_residual_population_stream_cpu_reference(
    *,
    registration: V1LiteralJJointAffineShellResidualPopulationStreamCUDARegistration,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    suffix_registration: V1SampledVariableShellDW2SuffixRegistration,
    packet: V1SampledVariableShellDW2SuffixPacket,
    prefix_freeze: V1SampledVariableShellDW2SuffixPrefixFreezeReceipt,
    row_group_plan: V1SampledDualChartRowGroupPlan,
    row_group_commit: V1SampledDualChartRowGroupCommit,
    packet_proof_closures: tuple[
        V1SampledDualChartRowGroupPacketProofClosure,
        V1SampledDualChartRowGroupPacketProofClosure,
    ],
    private_support: _V1LiteralJJointResidualCPUAnalyticSupport,
    design: V1LiteralJJointAffineShellResidualPopulationStreamCUDADesign,
    device_total_memory_bytes: int = 1_000_000_000,
    device_free_memory_bytes: int = 1_000_000_000,
    failure_injection: str | None = None,
) -> V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle:
    if failure_injection not in (None, "after_first_real_pass1_tile"):
        raise ValueError("unknown private joint residual-stream failure injection")
    private_support.assert_live()
    forecast = _make_forecast(
        design=design,
        joint_preoutcome_registration=joint_preoutcome_registration,
        joint_family_handle=joint_family_handle,
        affine_support_receipt_hashes=registration.affine_support_receipt_hashes,
        route_support_source_receipt_hash=registration.route_support_source_receipt_hash,
        observed_counts=registration.registered_observed_counts,
        device=torch.device("cpu"),
        total_bytes=device_total_memory_bytes,
        free_bytes=device_free_memory_bytes,
        reserved_bytes=0,
    )
    return _execute_v1_literal_j_joint_affine_shell_residual_population_stream(
        registration=registration,
        joint_preoutcome_registration=joint_preoutcome_registration,
        joint_family_handle=joint_family_handle,
        population_registration=population_registration,
        suffix_registration=suffix_registration,
        packet=packet,
        prefix_freeze=prefix_freeze,
        row_group_plan=row_group_plan,
        row_group_commit=row_group_commit,
        packet_proof_closures=packet_proof_closures,
        logical_look_result=None,
        joint_population_source=None,
        joint_population_execution_binding=None,
        design=design,
        hbm_forecast=forecast,
        affine_supports=(),
        route_support_source=None,
        private_support=private_support,
        production=False,
        failure_after_first_real_tile=(failure_injection == "after_first_real_pass1_tile"),
    )


def _execute_v1_literal_j_joint_affine_shell_residual_population_stream_cpu_reference_from_logical_look_result(
    *,
    registration: V1LiteralJJointAffineShellResidualPopulationStreamCUDARegistration,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    logical_look_result: V1SampledV7SharedDW2LiteralJLogicalLookResult,
    private_support: _V1LiteralJJointResidualCPUAnalyticSupport,
    design: V1LiteralJJointAffineShellResidualPopulationStreamCUDADesign,
    device_total_memory_bytes: int = 1_000_000_000,
    device_free_memory_bytes: int = 1_000_000_000,
    failure_injection: str | None = None,
) -> V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle:
    _refuse(
        V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.PREOUTCOME_PLAN_REQUIRED,
        "legacy CPU logical-result execution cannot prove the action-complete v2 PRE plan",
    )
    if failure_injection not in (None, "after_first_real_pass1_tile"):
        raise ValueError("unknown private segmented residual-stream failure injection")
    private_support.assert_live()
    forecast = _make_forecast(
        design=design,
        joint_preoutcome_registration=joint_preoutcome_registration,
        joint_family_handle=joint_family_handle,
        affine_support_receipt_hashes=registration.affine_support_receipt_hashes,
        route_support_source_receipt_hash=(
            registration.route_support_source_receipt_hash
        ),
        observed_counts=registration.registered_observed_counts,
        device=torch.device("cpu"),
        total_bytes=device_total_memory_bytes,
        free_bytes=device_free_memory_bytes,
        reserved_bytes=0,
    )
    return _execute_v1_literal_j_joint_affine_shell_residual_population_stream(
        registration=registration,
        joint_preoutcome_registration=joint_preoutcome_registration,
        joint_family_handle=joint_family_handle,
        population_registration=population_registration,
        suffix_registration=None,
        packet=None,
        prefix_freeze=None,
        row_group_plan=None,
        row_group_commit=None,
        packet_proof_closures=None,
        logical_look_result=logical_look_result,
        joint_population_source=None,
        joint_population_execution_binding=None,
        design=design,
        hbm_forecast=forecast,
        affine_supports=(),
        route_support_source=None,
        private_support=private_support,
        production=False,
        failure_after_first_real_tile=(
            failure_injection == "after_first_real_pass1_tile"
        ),
    )


def _execute_v1_literal_j_joint_affine_shell_residual_population_stream_cpu_reference_from_joint_population_source(
    *,
    registration: V1LiteralJJointAffineShellResidualPopulationStreamCUDARegistration,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    joint_population_source: V1LiteralJJointPopulationSource,
    private_support: _V1LiteralJJointResidualCPUAnalyticSupport,
    design: V1LiteralJJointAffineShellResidualPopulationStreamCUDADesign,
    device_total_memory_bytes: int = 1_000_000_000,
    device_free_memory_bytes: int = 1_000_000_000,
    failure_injection: str | None = None,
) -> V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle:
    _refuse(
        V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.PREOUTCOME_PLAN_REQUIRED,
        "legacy singular CPU execution cannot prove the action-complete v2 PRE plan",
    )
    if failure_injection not in (None, "after_first_real_pass1_tile"):
        raise ValueError("unknown private joint-population residual-stream failure injection")
    if type(joint_population_source) is not V1LiteralJJointPopulationSource:
        raise TypeError("joint-population residual execution requires its exact bound source")
    private_support.assert_live()
    forecast = _make_forecast(
        design=design,
        joint_preoutcome_registration=joint_preoutcome_registration,
        joint_family_handle=joint_family_handle,
        affine_support_receipt_hashes=registration.affine_support_receipt_hashes,
        route_support_source_receipt_hash=(
            registration.route_support_source_receipt_hash
        ),
        observed_counts=registration.registered_observed_counts,
        device=torch.device("cpu"),
        total_bytes=device_total_memory_bytes,
        free_bytes=device_free_memory_bytes,
        reserved_bytes=0,
    )
    return _execute_v1_literal_j_joint_affine_shell_residual_population_stream(
        registration=registration,
        joint_preoutcome_registration=joint_preoutcome_registration,
        joint_family_handle=joint_family_handle,
        population_registration=population_registration,
        suffix_registration=None,
        packet=None,
        prefix_freeze=None,
        row_group_plan=None,
        row_group_commit=None,
        packet_proof_closures=None,
        logical_look_result=None,
        joint_population_source=joint_population_source,
        joint_population_execution_binding=None,
        design=design,
        hbm_forecast=forecast,
        affine_supports=(),
        route_support_source=None,
        private_support=private_support,
        production=False,
        failure_after_first_real_tile=(
            failure_injection == "after_first_real_pass1_tile"
        ),
    )


def _execute_v1_literal_j_joint_affine_shell_residual_population_stream_cpu_reference_from_postsource_binding(
    *,
    binding: V1LiteralJJointAffineShellResidualPopulationStreamCUDAPostsourceBinding,
    device_total_memory_bytes: int = 1_000_000_000,
    device_free_memory_bytes: int = 1_000_000_000,
    failure_injection: str | None = None,
) -> V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle:
    if (
        type(binding)
        is not V1LiteralJJointAffineShellResidualPopulationStreamCUDAPostsourceBinding
    ):
        raise TypeError("private v2 execution requires its exact POST binding")
    if failure_injection not in (None, "after_first_real_pass1_tile"):
        raise ValueError("unknown private v2 residual-stream failure injection")
    binding.assert_live()
    plan = binding.preoutcome_plan
    if (
        plan.receipt.support_derivation_mode
        != "private_cpu_reference_support_v1"
        or type(binding.private_support)
        is not _V1LiteralJJointResidualCPUAnalyticSupport
        or binding.route_support_source is not None
        or binding.affine_supports != ()
    ):
        _refuse(
            V1LiteralJJointAffineShellResidualPopulationStreamCUDARefusalCode.CPU_PRODUCTION_FALLBACK,
            "private v2 executor accepts only its sealed CPU reference support",
        )
    forecast = _make_forecast(
        design=plan._design,
        joint_preoutcome_registration=plan._preregistration,
        joint_family_handle=binding.joint_family_handle,
        affine_support_receipt_hashes=binding.registration.affine_support_receipt_hashes,
        route_support_source_receipt_hash=(
            binding.registration.route_support_source_receipt_hash
        ),
        observed_counts=binding.registration.registered_observed_counts,
        device=torch.device("cpu"),
        total_bytes=device_total_memory_bytes,
        free_bytes=device_free_memory_bytes,
        reserved_bytes=0,
    )
    return _execute_v1_literal_j_joint_affine_shell_residual_population_stream(
        registration=binding.registration,
        joint_preoutcome_registration=plan._preregistration,
        joint_family_handle=binding.joint_family_handle,
        population_registration=plan._population,
        suffix_registration=None,
        packet=None,
        prefix_freeze=None,
        row_group_plan=None,
        row_group_commit=None,
        packet_proof_closures=None,
        logical_look_result=None,
        joint_population_source=None,
        joint_population_execution_binding=binding,
        design=plan._design,
        hbm_forecast=forecast,
        affine_supports=binding.affine_supports,
        route_support_source=None,
        private_support=binding.private_support,
        production=False,
        failure_after_first_real_tile=(
            failure_injection == "after_first_real_pass1_tile"
        ),
    )
