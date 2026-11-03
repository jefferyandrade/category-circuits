r"""Source-bound Q-dropped outside-pair lower bounds for block support.

For one registered block-support alias this module minimizes the weighted
``D`` plus residual support plane over the two exact outside-pair rectangles
from :mod:`modern_v1_signed_work_variable_shell_outside_exclusion_cuda`.
Resource and compact-capacity constraints are dualized; alternate ``Q`` is
deliberately dropped.  The alias cardinality identifies the rounded top-k
subset but is not a relaxed gate-mass equality.  With

``s = c_b - lambda r + B.T mu``

the common box dual is

``L0 = lambda rho - mu.T kappa``
``     + sum_required s_j + sum_free min(0, s_j)``.

The order cell ``z_p <= z_q`` adds the exact correction
``min(-s_p, s_q)`` only when ``s_p < 0 < s_q``.  A rectangle minimum is
therefore obtained from two tiled endpoint reductions; a route-pair matrix is
never formed.  Numerically ambiguous endpoint signs contribute the safe
correction lower bound zero.

Production requires CUDA-resident, mutation-checked sources and strict
deterministic algorithms.  The CPU path is an exact-rational, small-domain
reference only.  Receipts are array-free and intentionally carry no canonical
block, full-U, FQR, C/K/O, commit, terminal, or Pareto authority.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
from fractions import Fraction
import math
from numbers import Integral, Real
from typing import Any, Final, NoReturn, Sequence

from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

from .contracts import content_hash
from .modern_v1_signed_work_block_support_source_cuda import (
    V1SignedWorkBlockSupportPreferredActionBinding,
    V1SignedWorkBlockSupportSourceDeviceHandle,
    V1SignedWorkBlockSupportSourceRole,
)
from .modern_v1_signed_work_block_support_witness import (
    V1BlockSupportAliasBinding,
    V1BlockSupportWitnessSource,
)
from .modern_v1_signed_work_kkt_cuda import (
    V1RelaxedKKTBranch,
    V1SignedWorkLeastWorkCUDAStateHandle,
)
from .modern_v1_signed_work_device_shell_cuda import (
    V1SignedWorkDeviceShellExecutionBundle,
)
from .modern_v1_signed_work_physical_alternate_domain_cuda import (
    V1SignedWorkPhysicalAlternateBlockSupportBinding,
)
from .modern_v1_signed_work_variable_shell_action_separation_cuda import (
    V1SignedWorkVariableShellActionCellRegistry,
    V1SignedWorkVariableShellQBranch,
    deterministic_v1_signed_work_variable_shell_rounding_subset,
)
from .modern_v1_signed_work_variable_shell_outside_exclusion_cuda import (
    V1SignedWorkVariableShellOutsidePairCoverEnumeration,
    V1SignedWorkVariableShellOutsidePairCoverReceipt,
    V1SignedWorkVariableShellOutsidePairRectangle,
    evaluate_v1_signed_work_variable_shell_outside_pair_dual_bounds_reference,
)
from mechanistic_systems.contracts.hard_mask_actions import RemovedRouteIndexSet

try:  # pragma: no cover - runtime image dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


OUTSIDE_PAIR_DESIGN_SCHEMA: Final = (
    "partition_free_v1_block_support_outside_pair_cuda_design_v1"
)
OUTSIDE_PAIR_HBM_SCHEMA: Final = (
    "partition_free_v1_block_support_outside_pair_cuda_hbm_v1"
)
OUTSIDE_PAIR_DUAL_SOURCE_SCHEMA: Final = (
    "partition_free_v1_block_support_outside_pair_resource_capacity_dual_source_v1"
)
OUTSIDE_PAIR_DUAL_LEASE_SCHEMA: Final = (
    "partition_free_v1_block_support_outside_pair_resource_capacity_dual_lease_v1"
)
OUTSIDE_PAIR_MEMBERSHIP_SOURCE_SCHEMA: Final = (
    "partition_free_v1_block_support_outside_pair_resident_membership_source_v1"
)
OUTSIDE_PAIR_MEMBERSHIP_LEASE_SCHEMA: Final = (
    "partition_free_v1_block_support_outside_pair_resident_membership_lease_v1"
)
OUTSIDE_PAIR_FAMILY_SCHEMA: Final = (
    "partition_free_v1_block_support_q_dropped_outside_pair_family_v1"
)
OUTSIDE_PAIR_COVER_BINDING_SCHEMA: Final = (
    "partition_free_v1_block_support_outside_pair_typed_cover_binding_v1"
)
OUTSIDE_PAIR_COVER_BINDING_RULE: Final = (
    "source_bound_rounding_anchor_shell_eligible_required_parent_selectable_"
    "pair_axis_equivalence_v1"
)
OUTSIDE_PAIR_EQUATION: Final = (
    "s_equals_weighted_D_plus_residual_c_minus_lambda_r_plus_B_transpose_mu_"
    "and_L0_equals_lambda_rho_minus_mu_transpose_kappa_plus_required_s_plus_"
    "free_min_0_s_and_pair_correction_equals_"
    "min_minus_sp_sq_iff_sp_negative_sq_positive_else_zero_v1"
)
OUTSIDE_PAIR_AUTHORITY_SCOPE: Final = (
    "one_block_one_alias_Q_dropped_convex_outside_pair_lower_bound_only_no_"
    "canonical_block_observation_full_U_FQR_action_invariance_C_K_O_commit_"
    "terminal_or_pareto_authority_v1"
)

_PREFIX: Final = "partition-free-v1-block-support-outside-pair-"
_CANONICAL_CONTEXT_PREFIX: Final = (
    "partition-free-v1-block-support-domain-binding-canonical-context-tuple-"
)
_DESIGN_TOKEN: Final = object()
_FORECAST_TOKEN: Final = object()
_DUAL_RECEIPT_TOKEN: Final = object()
_DUAL_HANDLE_TOKEN: Final = object()
_MEMBERSHIP_RECEIPT_TOKEN: Final = object()
_MEMBERSHIP_HANDLE_TOKEN: Final = object()
_FAMILY_TOKEN: Final = object()
_COVER_BINDING_TOKEN: Final = object()
_F64_BYTES: Final = 8
_BOOL_BYTES: Final = 1
_F64_UNIT_ROUNDOFF: Final = 2.0**-53
_F64_MIN_NORMAL: Final = 2.0**-1022
_INDEX_SENTINEL: Final = 2**52


class V1BlockSupportOutsidePairExecutionMode(str, Enum):
    CUDA_PRODUCTION = "cuda_production"
    CPU_EXACT_REFERENCE_TEST_ONLY = "cpu_exact_reference_test_only"


class V1BlockSupportOutsidePairRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    INVALID_DESIGN = "invalid_design"
    SOURCE_MISMATCH = "source_mismatch"
    ALIAS_MISMATCH = "alias_mismatch"
    PAIR_COVER_MISMATCH = "pair_cover_mismatch"
    DUAL_INFEASIBLE = "dual_infeasible"
    NUMERICAL_REFUSAL = "numerical_refusal"
    NUMERICAL_SIGN_AMBIGUITY = "numerical_sign_ambiguity"
    CUDA_REQUIRED = "cuda_required"
    CPU_PRODUCTION_FALLBACK = "cpu_production_fallback"
    CPU_REFERENCE_SCALE_EXCEEDED = "cpu_reference_scale_exceeded"
    HBM_FORECAST_EXCEEDED = "hbm_forecast_exceeded"
    MUTATED_CUSTODY = "mutated_custody"
    ACTIVE_LEASE = "active_lease"
    EVICTED = "evicted"
    STALE_IDENTITY = "stale_identity"
    AUTHORITY_VIOLATION = "authority_violation"


class V1BlockSupportOutsidePairRefusal(RuntimeError):
    def __init__(
        self, code: V1BlockSupportOutsidePairRefusalCode, message: str
    ) -> None:
        if type(code) is not V1BlockSupportOutsidePairRefusalCode:
            raise TypeError("outside-pair refusal code has another type")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(code: V1BlockSupportOutsidePairRefusalCode, message: str) -> NoReturn:
    raise V1BlockSupportOutsidePairRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value.strip():
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative integer",
        )
    return int(value)


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.INVALID_TYPE,
            f"{name} must be one finite real",
        )
    result = float(value)
    if not math.isfinite(result):
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.NUMERICAL_REFUSAL,
            f"{name} must be finite",
        )
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if result < 0.0:
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.INVALID_TYPE,
            f"{name} must be nonnegative",
        )
    return result


def _gamma_n(operation_count: int) -> float:
    count = _nonnegative_int(operation_count, name="rounding operation count")
    product = float(count) * _F64_UNIT_ROUNDOFF
    if product >= 1.0:
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.NUMERICAL_REFUSAL,
            "binary64 gamma_n is undefined for this operation count",
        )
    return product / (1.0 - product)


def _nonnegative_arithmetic_upper(nominal: float, *, operation_count: int) -> float:
    """Outward upper enclosure for a nonnegative FP64 arithmetic result."""

    value = _finite_nonnegative(nominal, name="nonnegative arithmetic result")
    operations = _positive_int(operation_count, name="arithmetic operation count")
    gamma = _gamma_n(operations)
    if gamma >= 1.0:
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.NUMERICAL_REFUSAL,
            "arithmetic enclosure condition number reached one",
        )
    inflated = (value + float(operations) * _F64_MIN_NORMAL) / (1.0 - gamma)
    return _up(inflated)


def _down(value: float) -> float:
    result = math.nextafter(float(value), -math.inf)
    if math.isnan(result):
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.NUMERICAL_REFUSAL,
            "downward enclosure became NaN",
        )
    return result


def _up(value: float) -> float:
    result = math.nextafter(float(value), math.inf)
    if not math.isfinite(result):
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.NUMERICAL_REFUSAL,
            "outward numerical radius became nonfinite",
        )
    return result


def _tensor_invariant(value: object) -> tuple[object, ...]:
    if torch is None or type(value) is not torch.Tensor:
        raise TypeError("outside-pair custody requires exact torch tensors")
    return (
        tuple(int(item) for item in value.shape),
        tuple(int(item) for item in value.stride()),
        str(value.dtype),
        str(value.device),
        int(value.data_ptr()),
        int(value.storage_offset()),
        int(getattr(value, "_version", -1)),
        int(id(value)),
    )


def _closed_authority(values: Sequence[object], *, owner: str) -> None:
    if any(type(value) is not bool or value for value in values):
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.AUTHORITY_VIOLATION,
            f"{owner} crossed its authority boundary",
        )


def _canonical_context_axis_hash(contexts: Sequence[str]) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_canonical_sealed_context_tuple_v1",
            "context_ids": tuple(contexts),
        },
        prefix=_CANONICAL_CONTEXT_PREFIX,
    )


def _payload(instance: object) -> dict[str, object]:
    def normalize(value: object) -> object:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, float) and not math.isfinite(value):
            if math.isnan(value):
                return "nan_forbidden_v1"
            return "positive_infinity_v1" if value > 0.0 else "negative_infinity_v1"
        if isinstance(value, tuple):
            return tuple(normalize(item) for item in value)
        return value

    return {
        item.name: normalize(getattr(instance, item.name))
        for item in fields(instance)  # type: ignore[arg-type]
        if not item.name.startswith("_")
    }


@dataclass(frozen=True, slots=True)
class V1BlockSupportOutsidePairCUDAExecutionDesign:
    execution_mode: V1BlockSupportOutsidePairExecutionMode
    route_tile_size: int = 131_072
    hbm_ceiling_fraction: float = 0.85
    cpu_reference_maximum_routes: int = 64
    deterministic_fp64_required: bool = True
    forbid_pair_matrix: bool = True
    forbid_route_scale_d2h: bool = True
    forbid_route_scale_tensor_digest: bool = True
    schema: str = OUTSIDE_PAIR_DESIGN_SCHEMA
    _issuer_token: InitVar[object] = None

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _DESIGN_TOKEN:
            raise TypeError("outside-pair designs are issuer-only")
        if type(self.execution_mode) is not V1BlockSupportOutsidePairExecutionMode:
            raise TypeError("outside-pair execution mode has another type")
        object.__setattr__(
            self,
            "route_tile_size",
            _positive_int(self.route_tile_size, name="route tile size"),
        )
        object.__setattr__(
            self,
            "cpu_reference_maximum_routes",
            _positive_int(
                self.cpu_reference_maximum_routes,
                name="CPU reference maximum routes",
            ),
        )
        ceiling = _finite_nonnegative(
            self.hbm_ceiling_fraction, name="HBM ceiling fraction"
        )
        if not 0.0 < ceiling <= 0.85:
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.INVALID_DESIGN,
                "outside-pair HBM ceiling must lie in (0, 0.85]",
            )
        object.__setattr__(self, "hbm_ceiling_fraction", ceiling)
        if (
            any(
                type(value) is not bool or value is not True
                for value in (
                    self.deterministic_fp64_required,
                    self.forbid_pair_matrix,
                    self.forbid_route_scale_d2h,
                    self.forbid_route_scale_tensor_digest,
                )
            )
            or self.schema != OUTSIDE_PAIR_DESIGN_SCHEMA
        ):
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.INVALID_DESIGN,
                "outside-pair deterministic/memory invariants changed",
            )

    @property
    def design_hash(self) -> str:
        return content_hash(_payload(self), prefix=_PREFIX + "design-")


def build_v1_block_support_outside_pair_cuda_design(
    *,
    execution_mode: V1BlockSupportOutsidePairExecutionMode,
    route_tile_size: int = 131_072,
    hbm_ceiling_fraction: float = 0.85,
    cpu_reference_maximum_routes: int = 64,
) -> V1BlockSupportOutsidePairCUDAExecutionDesign:
    return V1BlockSupportOutsidePairCUDAExecutionDesign(
        execution_mode=execution_mode,
        route_tile_size=route_tile_size,
        hbm_ceiling_fraction=hbm_ceiling_fraction,
        cpu_reference_maximum_routes=cpu_reference_maximum_routes,
        _issuer_token=_DESIGN_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1BlockSupportOutsidePairHBMForecast:
    design_hash: str
    execution_mode: V1BlockSupportOutsidePairExecutionMode
    route_count: int
    capacity_count: int
    owned_resource_bytes: int
    resource_guard_bytes: int
    owned_capacity_adjoint_bytes: int
    capacity_adjoint_guard_bytes: int
    compact_dual_bytes: int
    borrowed_live_membership_bytes: int
    borrowed_live_numeric_source_bytes: int
    newly_owned_membership_bytes: int
    route_tile_workspace_bytes: int
    pair_matrix_bytes: int
    incremental_peak_bytes: int
    retained_device_bytes: int
    current_reserved_device_bytes: int
    projected_peak_reserved_bytes: int
    device_total_memory_bytes: int
    device_free_memory_bytes: int
    hbm_ceiling_fraction: float
    hbm_ceiling_bytes: int
    admitted: bool
    route_scale_d2h_bytes: int = 0
    route_scale_tensor_digest_bytes: int = 0
    schema: str = OUTSIDE_PAIR_HBM_SCHEMA
    _issuer_token: InitVar[object] = None

    _admission_memory: InitVar[tuple[int, int] | None] = None

    def __post_init__(self, _issuer_token: object, _admission_memory: tuple[int, int] | None) -> None:
        if _issuer_token is not _FORECAST_TOKEN:
            raise TypeError("outside-pair HBM forecasts are issuer-only")
        object.__setattr__(
            self, "design_hash", _identity(self.design_hash, name="design hash")
        )
        if type(self.execution_mode) is not V1BlockSupportOutsidePairExecutionMode:
            raise TypeError("outside-pair HBM mode has another type")
        for name in (
            "route_count",
            "capacity_count",
            "owned_resource_bytes",
            "resource_guard_bytes",
            "owned_capacity_adjoint_bytes",
            "capacity_adjoint_guard_bytes",
            "compact_dual_bytes",
            "borrowed_live_membership_bytes",
            "borrowed_live_numeric_source_bytes",
            "newly_owned_membership_bytes",
            "route_tile_workspace_bytes",
            "pair_matrix_bytes",
            "incremental_peak_bytes",
            "retained_device_bytes",
            "current_reserved_device_bytes",
            "projected_peak_reserved_bytes",
            "device_total_memory_bytes",
            "device_free_memory_bytes",
            "hbm_ceiling_bytes",
            "route_scale_d2h_bytes",
            "route_scale_tensor_digest_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name.replace("_", " ")),
            )
        if self.route_count <= 0:
            raise ValueError("outside-pair forecast route axis must be nonempty")
        route_bytes = self.route_count * _F64_BYTES
        expected_compact = (2 * self.capacity_count + 2) * _F64_BYTES
        production = (
            self.execution_mode
            is V1BlockSupportOutsidePairExecutionMode.CUDA_PRODUCTION
        )
        expected_borrowed_membership = (
            3 * self.route_count * _BOOL_BYTES + (self.route_count + 7) // 8
            if production
            else 0
        )
        expected_new_membership = (
            0 if production else 5 * self.route_count * _BOOL_BYTES
        )
        expected_borrowed_numeric = (3 if production else 2) * route_bytes
        expected_owned_resource = 0 if production else route_bytes
        expected_resource_guard = 0 if production else route_bytes
        expected_retained = (
            expected_owned_resource
            + expected_resource_guard
            + 2 * route_bytes
            + expected_compact
            + expected_new_membership
        )
        expected_incremental = expected_retained + self.route_tile_workspace_bytes
        ceiling = _finite_nonnegative(
            self.hbm_ceiling_fraction, name="HBM ceiling fraction"
        )
        expected_ceiling = math.floor(self.device_total_memory_bytes * ceiling)
        available, allocated = _admission_memory or (
            self.device_free_memory_bytes, self.current_reserved_device_bytes
        )
        expected_admitted = (
            allocated + expected_incremental <= expected_ceiling
            and expected_incremental <= available
        )
        if (
            self.owned_resource_bytes != expected_owned_resource
            or self.resource_guard_bytes != expected_resource_guard
            or self.owned_capacity_adjoint_bytes != route_bytes
            or self.capacity_adjoint_guard_bytes != route_bytes
            or self.compact_dual_bytes != expected_compact
            or self.borrowed_live_membership_bytes != expected_borrowed_membership
            or self.borrowed_live_numeric_source_bytes != expected_borrowed_numeric
            or self.newly_owned_membership_bytes != expected_new_membership
            or self.retained_device_bytes != expected_retained
            or self.incremental_peak_bytes != expected_incremental
            or self.projected_peak_reserved_bytes
            != self.current_reserved_device_bytes + expected_incremental
            or self.hbm_ceiling_bytes != expected_ceiling
            or type(self.admitted) is not bool
            or self.admitted is not expected_admitted
            or self.pair_matrix_bytes != 0
            or self.route_scale_d2h_bytes != 0
            or self.route_scale_tensor_digest_bytes != 0
            or self.schema != OUTSIDE_PAIR_HBM_SCHEMA
        ):
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.INVALID_DESIGN,
                "outside-pair HBM accounting or no-pair-matrix invariant changed",
            )

    @property
    def forecast_hash(self) -> str:
        return content_hash(_payload(self), prefix=_PREFIX + "hbm-")


def build_v1_block_support_outside_pair_hbm_forecast(
    *,
    design: V1BlockSupportOutsidePairCUDAExecutionDesign,
    route_count: int,
    capacity_count: int,
    current_reserved_device_bytes: int,
    device_total_memory_bytes: int,
    device_free_memory_bytes: int,
) -> V1BlockSupportOutsidePairHBMForecast:
    admission_memory = None
    if type(design) is not V1BlockSupportOutsidePairCUDAExecutionDesign:
        raise TypeError("outside-pair HBM forecast requires the exact design")
    count = _positive_int(route_count, name="route count")
    capacities = _nonnegative_int(capacity_count, name="capacity count")
    reserved = _nonnegative_int(
        current_reserved_device_bytes, name="current reserved device bytes"
    )
    total = _positive_int(device_total_memory_bytes, name="device total memory bytes")
    free = _nonnegative_int(device_free_memory_bytes, name="device free memory bytes")
    if design.execution_mode is V1BlockSupportOutsidePairExecutionMode.CUDA_PRODUCTION:
        if torch is None or not torch.cuda.is_available():
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.CUDA_REQUIRED,
                "production outside-pair forecast requires live CUDA memory facts",
            )
        device = torch.device("cuda", torch.cuda.current_device())
        live_free, live_total = torch.cuda.mem_get_info(device)
        admission_memory = (
            cuda_reusable_memory_bytes(device, framework=torch, free_bytes=int(live_free)),
            int(torch.cuda.memory_allocated(device)),
        )
        total = int(live_total)
        free = int(live_free)
        reserved = int(torch.cuda.memory_reserved(device))
    elif free > total or reserved > total:
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.INVALID_DESIGN,
            "reference memory facts exceed total memory",
        )
    route_bytes = count * _F64_BYTES
    compact = (2 * capacities + 2) * _F64_BYTES
    # Conservative simultaneous tile envelope: weighted/stationarity/radius
    # FP64 intermediates, Boolean structural/legality masks, and packed-axis
    # integer indexing.  No N x N pair storage is admitted.
    workspace = design.route_tile_size * (
        24 * _F64_BYTES + 18 * _BOOL_BYTES + 4 * _F64_BYTES
    )
    production = (
        design.execution_mode is V1BlockSupportOutsidePairExecutionMode.CUDA_PRODUCTION
    )
    new_membership = 0 if production else 5 * count * _BOOL_BYTES
    borrowed_membership = (
        3 * count * _BOOL_BYTES + (count + 7) // 8 if production else 0
    )
    borrowed_numeric = (3 if production else 2) * route_bytes
    owned_resource = 0 if production else route_bytes
    resource_guard = 0 if production else route_bytes
    retained = (
        owned_resource + resource_guard + 2 * route_bytes + compact + new_membership
    )
    incremental = retained + workspace
    available, allocated = admission_memory or (free, reserved)
    peak = reserved + incremental
    ceiling = math.floor(total * design.hbm_ceiling_fraction)
    return V1BlockSupportOutsidePairHBMForecast(
        design_hash=design.design_hash,
        execution_mode=design.execution_mode,
        route_count=count,
        capacity_count=capacities,
        owned_resource_bytes=owned_resource,
        resource_guard_bytes=resource_guard,
        owned_capacity_adjoint_bytes=route_bytes,
        capacity_adjoint_guard_bytes=route_bytes,
        compact_dual_bytes=compact,
        borrowed_live_membership_bytes=borrowed_membership,
        borrowed_live_numeric_source_bytes=borrowed_numeric,
        newly_owned_membership_bytes=new_membership,
        route_tile_workspace_bytes=workspace,
        pair_matrix_bytes=0,
        incremental_peak_bytes=incremental,
        retained_device_bytes=retained,
        current_reserved_device_bytes=reserved,
        projected_peak_reserved_bytes=peak,
        device_total_memory_bytes=total,
        device_free_memory_bytes=free,
        hbm_ceiling_fraction=design.hbm_ceiling_fraction,
        hbm_ceiling_bytes=ceiling,
        admitted=(allocated + incremental <= ceiling and incremental <= available),
        _issuer_token=_FORECAST_TOKEN,
        _admission_memory=admission_memory,
    )


def _ulp_radius(value: float) -> float:
    """One conservative binary64 source-conversion radius."""

    scalar = _finite(value, name="converted scalar")
    upper = math.nextafter(scalar, math.inf)
    lower = math.nextafter(scalar, -math.inf)
    radius = max(abs(upper - scalar), abs(scalar - lower))
    return _up(radius)


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
        or tuple(int(item) for item in value.shape) != shape
        or not value.is_contiguous()
        or (device is not None and value.device != device)
        or not bool(torch.isfinite(value).all().item())
        or (nonnegative and bool((value < 0.0).any().item()))
    ):
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.INVALID_TYPE,
            f"{name} must be one finite contiguous FP64 tensor on the bound device",
        )
    return value


def _require_fp64_tensor_layout(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    device: object,
) -> Any:
    """Validate layout without a device synchronization.

    Live source owners already validate finiteness at binding.  The scan folds
    numerical finiteness into its single compact device-to-host status vector.
    """

    if (
        torch is None
        or type(value) is not torch.Tensor
        or value.dtype != torch.float64
        or tuple(int(item) for item in value.shape) != shape
        or value.device != device
        or not value.is_contiguous()
    ):
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.INVALID_TYPE,
            f"{name} must be one contiguous FP64 tensor on the bound device",
        )
    return value


def _require_bool_tensor(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    device: object | None = None,
) -> Any:
    if (
        torch is None
        or type(value) is not torch.Tensor
        or value.dtype != torch.bool
        or tuple(int(item) for item in value.shape) != shape
        or not value.is_contiguous()
        or (device is not None and value.device != device)
    ):
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.INVALID_TYPE,
            f"{name} must be one contiguous Boolean tensor on the bound device",
        )
    return value


@dataclass(frozen=True, slots=True)
class V1BlockSupportOutsidePairCoverBindingReceipt:
    """Exact source join for one compact outside-pair cover.

    The legacy cover receipt binds host-computed membership hashes, but those
    hashes are not by themselves evidence that the membership came from the
    registered live anchor/KKT/parent sources.  This sibling receipt names
    every owner separately.  The small reference binder below can prove the
    join exactly.  CUDA production deliberately has no binder until the KKT
    owner exports its durable eligible/required mask identities.
    """

    alias_hash: str
    branch_id: str
    resource_alias_id: str
    source_receipt_hash: str
    source_replay_hash: str
    legacy_pair_cover_receipt_hash: str
    legacy_pair_axis_commitment_hash: str
    rounding_anchor_action_coordinate_hash: str
    rounding_anchor_removed_route_index_hash: str
    rounding_anchor_subset_identity_hash: str
    rounding_anchor_preferred_action_index_hash: str
    rounding_anchor_shell_gate_hash: str
    shell_freeze_receipt_hash: str
    shell_index_hash: str
    eligible_mask_hash: str
    required_mask_hash: str
    parent_mask_hash: str
    already_removed_mask_hash: str
    selectable_mask_hash: str
    kkt_problem_hash: str | None
    kkt_state_identity_hash: str | None
    kkt_eligible_mask_identity_hash: str | None
    kkt_required_mask_identity_hash: str | None
    source_rounding_hash: str | None
    frozen_ladder_receipt_hash: str | None
    route_count: int
    alias_fixed_outside_removed_count: int
    alias_shell_prefix_count: int
    alias_total_removed_count: int
    rounding_anchor_route_count: int
    rounding_anchor_selected_shell_offsets: tuple[int, ...]
    exact_pair_axis_equivalence: bool = True
    deterministic_physical_index_topk_replayed: bool = True
    production_kkt_mask_identity_export_joined: bool = False
    cpu_exact_reference_binding: bool = True
    production_binding: bool = False
    route_scale_d2h_bytes: int = 0
    route_scale_content_digest_bytes: int = 0
    downstream_authority: bool = False
    full_u_authority: bool = False
    commit_authority: bool = False
    pareto_authority: bool = False
    theorem_rule: str = OUTSIDE_PAIR_COVER_BINDING_RULE
    schema: str = OUTSIDE_PAIR_COVER_BINDING_SCHEMA
    _issuer_token: InitVar[object] = None
    _integrity_seal: str = field(default="", repr=False, compare=False)

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _COVER_BINDING_TOKEN:
            raise TypeError("outside-pair cover bindings are issuer-only")
        for item in fields(self):
            if (
                not item.name.startswith("_")
                and item.name.endswith("_hash")
                and getattr(self, item.name) is not None
            ):
                object.__setattr__(
                    self,
                    item.name,
                    _identity(
                        getattr(self, item.name), name=item.name.replace("_", " ")
                    ),
                )
        for name in ("branch_id", "resource_alias_id"):
            object.__setattr__(
                self, name, _identity(getattr(self, name), name=name.replace("_", " "))
            )
        for name in (
            "route_count",
            "alias_fixed_outside_removed_count",
            "alias_shell_prefix_count",
            "alias_total_removed_count",
            "rounding_anchor_route_count",
            "route_scale_d2h_bytes",
            "route_scale_content_digest_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name.replace("_", " ")),
            )
        offsets = tuple(
            _nonnegative_int(value, name="rounding-anchor shell offset")
            for value in self.rounding_anchor_selected_shell_offsets
        )
        if (
            self.route_count <= 0
            or offsets != tuple(sorted(set(offsets)))
            or len(offsets) != self.alias_shell_prefix_count
            or self.alias_total_removed_count
            != self.alias_fixed_outside_removed_count + self.alias_shell_prefix_count
            or self.rounding_anchor_route_count != self.alias_total_removed_count
            or self.rounding_anchor_route_count > self.route_count
        ):
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.PAIR_COVER_MISMATCH,
                "typed cover binding anchor/cardinality axes do not close",
            )
        object.__setattr__(self, "rounding_anchor_selected_shell_offsets", offsets)
        required_true = (
            self.exact_pair_axis_equivalence,
            self.deterministic_physical_index_topk_replayed,
        )
        if any(type(value) is not bool or value is not True for value in required_true):
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.AUTHORITY_VIOLATION,
                "typed cover binding lost exact source equivalence",
            )
        if (
            type(self.cpu_exact_reference_binding) is not bool
            or type(self.production_binding) is not bool
            or self.cpu_exact_reference_binding == self.production_binding
        ):
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.AUTHORITY_VIOLATION,
                "typed cover binding must select exactly one execution custody mode",
            )
        production_fields = (
            self.kkt_problem_hash,
            self.kkt_state_identity_hash,
            self.kkt_eligible_mask_identity_hash,
            self.kkt_required_mask_identity_hash,
            self.source_rounding_hash,
            self.frozen_ladder_receipt_hash,
        )
        if self.production_binding:
            if self.production_kkt_mask_identity_export_joined is not True or any(
                value is None for value in production_fields
            ):
                _refuse(
                    V1BlockSupportOutsidePairRefusalCode.AUTHORITY_VIOLATION,
                    "production cover binding lacks frozen rounding/KKT mask custody",
                )
        elif self.production_kkt_mask_identity_export_joined is not False or any(
            value is not None for value in production_fields
        ):
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.AUTHORITY_VIOLATION,
                "reference cover binding crossed into production custody",
            )
        _closed_authority(
            (
                self.downstream_authority,
                self.full_u_authority,
                self.commit_authority,
                self.pareto_authority,
            ),
            owner="outside-pair typed cover binding",
        )
        if (
            self.route_scale_d2h_bytes != 0
            or self.route_scale_content_digest_bytes != 0
            or self.theorem_rule != OUTSIDE_PAIR_COVER_BINDING_RULE
            or self.schema != OUTSIDE_PAIR_COVER_BINDING_SCHEMA
        ):
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.AUTHORITY_VIOLATION,
                "typed cover binding changed theorem or route-scale custody",
            )
        observed = content_hash(_payload(self), prefix=_PREFIX + "cover-binding-")
        if self._integrity_seal and self._integrity_seal != observed:
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.MUTATED_CUSTODY,
                "typed cover binding receipt was mutated",
            )
        object.__setattr__(self, "_integrity_seal", observed)

    def assert_integrity(self) -> str:
        observed = content_hash(_payload(self), prefix=_PREFIX + "cover-binding-")
        if observed != self._integrity_seal:
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.MUTATED_CUSTODY,
                "typed cover binding integrity changed",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


def bind_v1_block_support_outside_pair_cover_reference(
    *,
    registry: V1SignedWorkVariableShellActionCellRegistry,
    alias_binding: V1BlockSupportAliasBinding,
    pair_cover: V1SignedWorkVariableShellOutsidePairCoverEnumeration,
    frozen_shell_gates: Sequence[Real],
) -> V1BlockSupportOutsidePairCoverBindingReceipt:
    """Prove one small-domain cover from a live branch and exact gate replay."""

    if (
        type(registry) is not V1SignedWorkVariableShellActionCellRegistry
        or type(alias_binding) is not V1BlockSupportAliasBinding
        or type(pair_cover) is not V1SignedWorkVariableShellOutsidePairCoverEnumeration
    ):
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.INVALID_TYPE,
            "reference cover binding requires exact registry/alias/cover sources",
        )
    registry.assert_live_sources()
    try:
        branch = V1SignedWorkVariableShellQBranch(alias_binding.branch_id)
    except ValueError:
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.ALIAS_MISMATCH,
            "cover-binding alias branch is not registered",
        )
    source = registry.branch_source(branch)
    resource_alias = source.resource_alias(alias_binding.resource_alias_id)
    source_receipt = source.assert_live_source().receipt
    cover = pair_cover.receipt
    if (
        alias_binding.source_receipt_hash != source.registration.source_receipt_hash
        or alias_binding.source_replay_hash != source.registration.source_replay_hash
        or alias_binding.cardinality != resource_alias.total_removed_count
        or alias_binding.outside_pair_cover_hash != cover.receipt_hash
        or cover.route_count != source_receipt.route_count
        or tuple(pair_cover.shell_indices) != source.registration.shell_indices
        or cover.fixed_cardinality != resource_alias.total_removed_count
    ):
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.ALIAS_MISMATCH,
            "reference cover does not bind the live branch/resource/shell alias",
        )
    gate_evaluation = source.assert_live_source().evaluate(frozen_shell_gates)
    offsets = deterministic_v1_signed_work_variable_shell_rounding_subset(
        shell_gates=gate_evaluation.shell_gates,
        shell_route_indices=source.registration.shell_indices,
        shell_prefix_count=resource_alias.shell_prefix_count,
    )
    expected_shell_routes = tuple(
        source.registration.shell_indices[offset] for offset in offsets
    )
    preferred = pair_cover.preferred_action_indices
    shell_set = set(source.registration.shell_indices)
    observed_shell_routes = tuple(index for index in preferred if index in shell_set)
    if observed_shell_routes != expected_shell_routes:
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.PAIR_COVER_MISMATCH,
            "cover anchor differs from deterministic live shell-gate top-k replay",
        )
    anchor_hash = RemovedRouteIndexSet.from_indices(
        preferred, route_count=cover.route_count
    ).index_hash
    subset_matches = tuple(
        identity
        for identity in source.subset_identities
        if identity.shell_offsets == offsets
        and identity.removed_route_index_hash == anchor_hash
    )
    if len(subset_matches) != 1:
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.PAIR_COVER_MISMATCH,
            "cover anchor does not replay one live source subset identity",
        )
    subset = subset_matches[0]
    return V1BlockSupportOutsidePairCoverBindingReceipt(
        alias_hash=alias_binding.alias_hash,
        branch_id=alias_binding.branch_id,
        resource_alias_id=alias_binding.resource_alias_id,
        source_receipt_hash=alias_binding.source_receipt_hash,
        source_replay_hash=alias_binding.source_replay_hash,
        legacy_pair_cover_receipt_hash=cover.receipt_hash,
        legacy_pair_axis_commitment_hash=cover.pair_axis_commitment_hash,
        rounding_anchor_action_coordinate_hash=subset.physical_action_coordinate_hash,
        rounding_anchor_removed_route_index_hash=subset.removed_route_index_hash,
        rounding_anchor_subset_identity_hash=subset.subset_hash,
        rounding_anchor_preferred_action_index_hash=(cover.preferred_action_index_hash),
        rounding_anchor_shell_gate_hash=gate_evaluation.shell_gate_hash,
        shell_freeze_receipt_hash=source.registration.shell_freeze_receipt_hash,
        shell_index_hash=cover.shell_index_hash,
        eligible_mask_hash=cover.eligible_mask_hash,
        required_mask_hash=cover.required_mask_hash,
        parent_mask_hash=source.registration.parent_mask_hash,
        already_removed_mask_hash=cover.already_removed_mask_hash,
        selectable_mask_hash=cover.selectable_mask_hash,
        kkt_problem_hash=None,
        kkt_state_identity_hash=None,
        kkt_eligible_mask_identity_hash=None,
        kkt_required_mask_identity_hash=None,
        source_rounding_hash=None,
        frozen_ladder_receipt_hash=None,
        route_count=cover.route_count,
        alias_fixed_outside_removed_count=resource_alias.fixed_outside_removed_count,
        alias_shell_prefix_count=resource_alias.shell_prefix_count,
        alias_total_removed_count=resource_alias.total_removed_count,
        rounding_anchor_route_count=cover.fixed_cardinality,
        rounding_anchor_selected_shell_offsets=offsets,
        _issuer_token=_COVER_BINDING_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1BlockSupportOutsidePairMembershipSourceReceipt:
    pair_cover_receipt_hash: str
    pair_axis_commitment_hash: str
    preferred_action_index_hash: str
    comparison_preferred_removed_route_index_hash: str
    shell_index_hash: str
    selectable_mask_hash: str
    required_mask_hash: str
    owner_source_receipt_hash: str
    route_count: int
    fixed_cardinality: int
    rounding_anchor_count: int
    comparison_preferred_count: int
    required_count: int
    selectable_count: int
    shell_count: int
    add_outside_rectangle_pair_count: int
    drop_outside_add_inside_rectangle_pair_count: int
    complete_pair_count: int
    device_type: str
    exact_cover_count_replay: bool = True
    resident_route_membership_custody: bool = True
    no_route_scale_content_digest: bool = True
    route_scale_d2h_bytes: int = 0
    route_scale_tensor_digest_bytes: int = 0
    array_free_receipt: bool = True
    cpu_reference_test_only: bool = True
    production_membership_authority: bool = False
    downstream_authority: bool = False
    full_u_authority: bool = False
    commit_authority: bool = False
    pareto_authority: bool = False
    schema: str = OUTSIDE_PAIR_MEMBERSHIP_SOURCE_SCHEMA
    _issuer_token: InitVar[object] = None
    _integrity_seal: str = field(default="", repr=False, compare=False)

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _MEMBERSHIP_RECEIPT_TOKEN:
            raise TypeError("outside-pair membership receipts are issuer-only")
        for item in fields(self):
            if not item.name.startswith("_") and item.name.endswith("_hash"):
                object.__setattr__(
                    self,
                    item.name,
                    _identity(
                        getattr(self, item.name), name=item.name.replace("_", " ")
                    ),
                )
        object.__setattr__(
            self, "device_type", _identity(self.device_type, name="device type")
        )
        for name in (
            "route_count",
            "fixed_cardinality",
            "rounding_anchor_count",
            "comparison_preferred_count",
            "required_count",
            "selectable_count",
            "shell_count",
            "add_outside_rectangle_pair_count",
            "drop_outside_add_inside_rectangle_pair_count",
            "complete_pair_count",
            "route_scale_d2h_bytes",
            "route_scale_tensor_digest_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name.replace("_", " ")),
            )
        if (
            self.route_count <= 0
            or self.rounding_anchor_count != self.fixed_cardinality
        ):
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.PAIR_COVER_MISMATCH,
                "membership dimensions do not replay the fixed-cardinality cover",
            )
        required_true = (
            self.exact_cover_count_replay,
            self.resident_route_membership_custody,
            self.no_route_scale_content_digest,
            self.array_free_receipt,
            self.cpu_reference_test_only,
        )
        if any(type(value) is not bool or value is not True for value in required_true):
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.AUTHORITY_VIOLATION,
                "membership source lost a custody invariant",
            )
        _closed_authority(
            (
                self.downstream_authority,
                self.production_membership_authority,
                self.full_u_authority,
                self.commit_authority,
                self.pareto_authority,
            ),
            owner="outside-pair membership receipt",
        )
        if (
            self.route_scale_d2h_bytes != 0
            or self.route_scale_tensor_digest_bytes != 0
            or self.schema != OUTSIDE_PAIR_MEMBERSHIP_SOURCE_SCHEMA
        ):
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.AUTHORITY_VIOLATION,
                "membership source crossed the route-scale host boundary",
            )
        observed = content_hash(_payload(self), prefix=_PREFIX + "membership-receipt-")
        if self._integrity_seal and self._integrity_seal != observed:
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.MUTATED_CUSTODY,
                "membership receipt was mutated",
            )
        object.__setattr__(self, "_integrity_seal", observed)

    def assert_integrity(self) -> str:
        observed = content_hash(_payload(self), prefix=_PREFIX + "membership-receipt-")
        if observed != self._integrity_seal:
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.MUTATED_CUSTODY,
                "membership receipt integrity changed",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


class V1BlockSupportOutsidePairMembershipLease:
    __slots__ = (
        "_owner",
        "_token",
        "_closed",
        "membership_source_receipt_hash",
        "mutation_identity_hash",
        "downstream_authority",
        "schema",
    )
    __hash__ = None

    def __init__(
        self, owner: "V1BlockSupportOutsidePairMembershipSourceHandle", token: object
    ) -> None:
        if type(owner) is not V1BlockSupportOutsidePairMembershipSourceHandle:
            raise TypeError("outside-pair membership lease requires the exact owner")
        self._owner = owner
        self._token = token
        self._closed = False
        self.membership_source_receipt_hash = owner.receipt.receipt_hash
        self.mutation_identity_hash = owner.mutation_identity_hash
        self.downstream_authority = False
        self.schema = OUTSIDE_PAIR_MEMBERSHIP_LEASE_SCHEMA

    def live_membership(self) -> tuple[Any, Any, Any, Any, Any]:
        if self._closed:
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.EVICTED,
                "outside-pair membership lease is closed",
            )
        if self._owner.mutation_identity_hash != self.mutation_identity_hash:
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.MUTATED_CUSTODY,
                "outside-pair membership changed during its lease",
            )
        return self._owner._require_live()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._owner._close_lease(self._token)
        finally:
            self._closed = True

    def __enter__(self) -> "V1BlockSupportOutsidePairMembershipLease":
        self.live_membership()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        del exc_type, exc, traceback
        self.close()
        return False

    def __reduce__(self) -> NoReturn:
        raise TypeError("outside-pair membership leases are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("outside-pair membership leases are nonserializable")


class V1BlockSupportOutsidePairMembershipSourceHandle:
    """Small CPU-reference five-mask owner; never a production source."""

    __slots__ = (
        "_tensors",
        "_guards",
        "_invariants",
        "_guard_invariants",
        "_active_leases",
        "_evicted",
        "receipt",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        rounding_anchor: Any,
        comparison_preferred: Any,
        required: Any,
        selectable: Any,
        shell: Any,
        receipt: V1BlockSupportOutsidePairMembershipSourceReceipt,
        _issuer_token: object,
    ) -> None:
        if _issuer_token is not _MEMBERSHIP_HANDLE_TOKEN:
            raise TypeError("outside-pair membership handles are issuer-only")
        if type(receipt) is not V1BlockSupportOutsidePairMembershipSourceReceipt:
            raise TypeError("outside-pair membership handle requires its exact receipt")
        anchor_tensor = _require_bool_tensor(
            rounding_anchor,
            name="rounding-anchor membership",
            shape=(receipt.route_count,),
        )
        device = anchor_tensor.device
        tensors = (
            anchor_tensor,
            _require_bool_tensor(
                comparison_preferred,
                name="comparison-preferred membership",
                shape=(receipt.route_count,),
                device=device,
            ),
            _require_bool_tensor(
                required,
                name="required membership",
                shape=(receipt.route_count,),
                device=device,
            ),
            _require_bool_tensor(
                selectable,
                name="selectable membership",
                shape=(receipt.route_count,),
                device=device,
            ),
            _require_bool_tensor(
                shell,
                name="shell membership",
                shape=(receipt.route_count,),
                device=device,
            ),
        )
        self._tensors = tensors
        self._guards = tuple(value.detach().clone() for value in tensors)
        self._invariants = tuple(_tensor_invariant(value) for value in tensors)
        self._guard_invariants = tuple(
            _tensor_invariant(value) for value in self._guards
        )
        self._active_leases: set[object] = set()
        self._evicted = False
        self.receipt = receipt
        self.assert_live()

    def _require_live(self) -> tuple[Any, Any, Any, Any, Any]:
        if self._evicted:
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.EVICTED,
                "outside-pair membership source was evicted",
            )
        if (
            tuple(_tensor_invariant(value) for value in self._tensors)
            != self._invariants
            or tuple(_tensor_invariant(value) for value in self._guards)
            != self._guard_invariants
            or not all(
                bool(torch.equal(live, guard))
                for live, guard in zip(self._tensors, self._guards, strict=True)
            )
        ):
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.MUTATED_CUSTODY,
                "outside-pair membership tensors changed after sealing",
            )
        self.receipt.assert_integrity()
        return self._tensors  # type: ignore[return-value]

    def assert_live(self) -> None:
        self._require_live()

    @property
    def mutation_identity_hash(self) -> str:
        self._require_live()
        return content_hash(
            self._invariants,
            prefix=_PREFIX + "membership-live-mutation-identity-",
        )

    def open_membership_lease(self) -> V1BlockSupportOutsidePairMembershipLease:
        self.assert_live()
        token = object()
        self._active_leases.add(token)
        return V1BlockSupportOutsidePairMembershipLease(self, token)

    def _close_lease(self, token: object) -> None:
        if token not in self._active_leases:
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.STALE_IDENTITY,
                "outside-pair membership lease token is not active",
            )
        try:
            self.assert_live()
        finally:
            self._active_leases.remove(token)

    def evict(self) -> None:
        self.assert_live()
        if self._active_leases:
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.ACTIVE_LEASE,
                "cannot evict outside-pair membership with an active lease",
            )
        self._tensors = ()
        self._guards = ()
        self._invariants = ()
        self._guard_invariants = ()
        self._evicted = True

    def __reduce__(self) -> NoReturn:
        raise TypeError("outside-pair membership handles are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("outside-pair membership handles are nonserializable")


def bind_v1_block_support_outside_pair_membership_source(
    *,
    pair_cover_receipt: V1SignedWorkVariableShellOutsidePairCoverReceipt,
    owner_source_receipt_hash: str,
    rounding_anchor_membership: object,
    comparison_preferred_membership: object,
    comparison_preferred_removed_route_index_hash: str,
    required_membership: object,
    selectable_membership: object,
    shell_membership: object,
) -> V1BlockSupportOutsidePairMembershipSourceHandle:
    """Seal small CPU-reference membership; never production authority."""

    if type(pair_cover_receipt) is not V1SignedWorkVariableShellOutsidePairCoverReceipt:
        raise TypeError("membership binding requires the exact compact cover receipt")
    count = pair_cover_receipt.route_count
    anchor = _require_bool_tensor(
        rounding_anchor_membership, name="rounding-anchor membership", shape=(count,)
    )
    device = anchor.device
    comparison_preferred = _require_bool_tensor(
        comparison_preferred_membership,
        name="comparison-preferred membership",
        shape=(count,),
        device=device,
    )
    required = _require_bool_tensor(
        required_membership,
        name="required membership",
        shape=(count,),
        device=device,
    )
    selectable = _require_bool_tensor(
        selectable_membership,
        name="selectable membership",
        shape=(count,),
        device=device,
    )
    shell = _require_bool_tensor(
        shell_membership,
        name="shell membership",
        shape=(count,),
        device=device,
    )
    if device.type != "cpu":
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.CPU_PRODUCTION_FALLBACK,
            "caller-owned five-mask membership is CPU-reference-only",
        )
    if bool((required & ~anchor).any().item()) or bool(
        (anchor & ~selectable).any().item()
    ):
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.PAIR_COVER_MISMATCH,
            "required must be preferred and preferred must be selectable",
        )
    droppable = anchor & ~required
    addable = selectable & ~anchor
    addable_outside = addable & ~shell
    droppable_outside = droppable & ~shell
    addable_inside = addable & shell
    scalar_counts = torch.stack(
        (
            anchor.sum(dtype=torch.int64),
            comparison_preferred.sum(dtype=torch.int64),
            required.sum(dtype=torch.int64),
            selectable.sum(dtype=torch.int64),
            shell.sum(dtype=torch.int64),
            droppable.sum(dtype=torch.int64),
            addable_outside.sum(dtype=torch.int64),
            droppable_outside.sum(dtype=torch.int64),
            addable_inside.sum(dtype=torch.int64),
        )
    ).to(device="cpu")
    (
        anchor_count,
        comparison_preferred_count,
        required_count,
        selectable_count,
        shell_count,
        droppable_count,
        addable_outside_count,
        droppable_outside_count,
        addable_inside_count,
    ) = (int(value) for value in scalar_counts.tolist())
    first = droppable_count * addable_outside_count
    second = droppable_outside_count * addable_inside_count
    if (
        anchor_count != pair_cover_receipt.fixed_cardinality
        or droppable_count != pair_cover_receipt.droppable_preferred_count
        or addable_outside_count != pair_cover_receipt.addable_outside_shell_count
        or droppable_outside_count
        != pair_cover_receipt.droppable_preferred_outside_shell_count
        or addable_inside_count != pair_cover_receipt.addable_inside_shell_count
        or first != pair_cover_receipt.add_outside_rectangle_pair_count
        or second != pair_cover_receipt.drop_outside_add_inside_rectangle_pair_count
        or first + second != pair_cover_receipt.complete_pair_count
    ):
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.PAIR_COVER_MISMATCH,
            "resident membership counts do not replay the compact pair cover",
        )
    receipt = V1BlockSupportOutsidePairMembershipSourceReceipt(
        pair_cover_receipt_hash=pair_cover_receipt.receipt_hash,
        pair_axis_commitment_hash=pair_cover_receipt.pair_axis_commitment_hash,
        preferred_action_index_hash=pair_cover_receipt.preferred_action_index_hash,
        comparison_preferred_removed_route_index_hash=_identity(
            comparison_preferred_removed_route_index_hash,
            name="comparison preferred removed-route index hash",
        ),
        shell_index_hash=pair_cover_receipt.shell_index_hash,
        selectable_mask_hash=pair_cover_receipt.selectable_mask_hash,
        required_mask_hash=pair_cover_receipt.required_mask_hash,
        owner_source_receipt_hash=_identity(
            owner_source_receipt_hash, name="membership owner source receipt hash"
        ),
        route_count=count,
        fixed_cardinality=pair_cover_receipt.fixed_cardinality,
        rounding_anchor_count=anchor_count,
        comparison_preferred_count=comparison_preferred_count,
        required_count=required_count,
        selectable_count=selectable_count,
        shell_count=shell_count,
        add_outside_rectangle_pair_count=first,
        drop_outside_add_inside_rectangle_pair_count=second,
        complete_pair_count=first + second,
        device_type=device.type,
        _issuer_token=_MEMBERSHIP_RECEIPT_TOKEN,
    )
    owned = tuple(
        value.detach().clone(memory_format=torch.contiguous_format)
        for value in (anchor, comparison_preferred, required, selectable, shell)
    )
    return V1BlockSupportOutsidePairMembershipSourceHandle(
        rounding_anchor=owned[0],
        comparison_preferred=owned[1],
        required=owned[2],
        selectable=owned[3],
        shell=owned[4],
        receipt=receipt,
        _issuer_token=_MEMBERSHIP_HANDLE_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1BlockSupportOutsidePairDualSourceReceipt:
    alias_hash: str
    branch_id: str
    fd_rank: int
    resource_alias_id: str
    source_receipt_hash: str
    source_replay_hash: str
    cardinality_alias_id: str
    cardinality: int
    constraint_cover_hash: str
    outside_pair_cover_hash: str
    pair_axis_commitment_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    preferred_action_hash: str
    preferred_action_binding_hash: str
    resource_axis_hash: str
    resource_coefficient_source_hash: str
    capacity_operator_source_hash: str
    resource_multiplier_source_hash: str
    capacity_multiplier_source_hash: str
    route_count: int
    capacity_count: int
    resource_multiplier_nominal: float
    resource_floor_nominal: float
    capacity_multiplier_dot_limit_nominal: float
    resource_multiplier_conversion_radius: float
    capacity_adjoint_roundoff_l1_radius: float
    capacity_adjoint_roundoff_linf_radius: float
    dual_constant_roundoff_radius: float
    capacity_multiplier_axis_hash: str
    capacity_limit_axis_hash: str
    design_hash: str
    hbm_forecast_hash: str
    device_type: str
    resource_custody_kind: str
    alternate_Q_dropped: bool = True
    rounded_subset_cardinality_identity_only: bool = True
    relaxed_cardinality_equality_enforced: bool = False
    resource_and_capacity_duals_source_bound: bool = True
    production_resource_copy_owned: bool = False
    route_scale_d2h_bytes: int = 0
    route_scale_tensor_digest_bytes: int = 0
    array_free_receipt: bool = True
    solver_authority: bool = False
    canonical_block_observation_authority: bool = False
    full_u_authority: bool = False
    fqr_authority: bool = False
    c_authority: bool = False
    k_authority: bool = False
    o_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    equation: str = OUTSIDE_PAIR_EQUATION
    authority_scope: str = OUTSIDE_PAIR_AUTHORITY_SCOPE
    schema: str = OUTSIDE_PAIR_DUAL_SOURCE_SCHEMA
    _issuer_token: InitVar[object] = None
    _integrity_seal: str = field(default="", repr=False, compare=False)

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _DUAL_RECEIPT_TOKEN:
            raise TypeError("outside-pair dual-source receipts are issuer-only")
        for item in fields(self):
            if not item.name.startswith("_") and item.name.endswith("_hash"):
                object.__setattr__(
                    self,
                    item.name,
                    _identity(
                        getattr(self, item.name), name=item.name.replace("_", " ")
                    ),
                )
        for name in (
            "branch_id",
            "resource_alias_id",
            "cardinality_alias_id",
            "device_type",
            "resource_custody_kind",
        ):
            object.__setattr__(
                self, name, _identity(getattr(self, name), name=name.replace("_", " "))
            )
        for name in (
            "fd_rank",
            "route_count",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name.replace("_", " ")),
            )
        for name in (
            "cardinality",
            "capacity_count",
            "proposal_generation",
            "route_scale_d2h_bytes",
            "route_scale_tensor_digest_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name.replace("_", " ")),
            )
        for name in (
            "resource_multiplier_nominal",
            "resource_floor_nominal",
            "capacity_multiplier_dot_limit_nominal",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name=name))
        if self.resource_multiplier_nominal < 0.0:
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.DUAL_INFEASIBLE,
                "resource multiplier must be nonnegative",
            )
        for name in (
            "resource_multiplier_conversion_radius",
            "capacity_adjoint_roundoff_l1_radius",
            "capacity_adjoint_roundoff_linf_radius",
            "dual_constant_roundoff_radius",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name=name.replace("_", " ")),
            )
        required_true = (
            self.alternate_Q_dropped,
            self.rounded_subset_cardinality_identity_only,
            self.resource_and_capacity_duals_source_bound,
            self.array_free_receipt,
        )
        if any(type(value) is not bool or value is not True for value in required_true):
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.AUTHORITY_VIOLATION,
                "outside-pair dual source lost a mandatory source-binding flag",
            )
        if type(self.relaxed_cardinality_equality_enforced) is not bool or (
            self.relaxed_cardinality_equality_enforced
        ):
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.AUTHORITY_VIOLATION,
                "top-k rounding preimage may not impose relaxed gate-mass equality",
            )
        _closed_authority(
            (
                self.solver_authority,
                self.production_resource_copy_owned,
                self.canonical_block_observation_authority,
                self.full_u_authority,
                self.fqr_authority,
                self.c_authority,
                self.k_authority,
                self.o_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="outside-pair dual-source receipt",
        )
        if (
            self.route_scale_d2h_bytes != 0
            or self.route_scale_tensor_digest_bytes != 0
            or self.equation != OUTSIDE_PAIR_EQUATION
            or self.authority_scope != OUTSIDE_PAIR_AUTHORITY_SCOPE
            or self.schema != OUTSIDE_PAIR_DUAL_SOURCE_SCHEMA
            or self.resource_custody_kind
            not in {"cpu_reference_owned_copy", "production_kkt_live_borrow"}
            or (
                self.device_type == "cuda"
                and self.resource_custody_kind != "production_kkt_live_borrow"
            )
            or (
                self.device_type == "cpu"
                and self.resource_custody_kind != "cpu_reference_owned_copy"
            )
        ):
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.INVALID_DESIGN,
                "outside-pair dual-source receipt semantics changed",
            )
        observed = content_hash(self._payload(), prefix=_PREFIX + "dual-receipt-")
        if self._integrity_seal and self._integrity_seal != observed:
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.MUTATED_CUSTODY,
                "outside-pair dual-source receipt was mutated",
            )
        object.__setattr__(self, "_integrity_seal", observed)

    def _payload(self) -> dict[str, object]:
        return _payload(self)

    def assert_integrity(self) -> str:
        observed = content_hash(self._payload(), prefix=_PREFIX + "dual-receipt-")
        if observed != self._integrity_seal:
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.MUTATED_CUSTODY,
                "outside-pair dual-source receipt integrity changed",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


class V1BlockSupportOutsidePairDualLease:
    __slots__ = (
        "_owner",
        "_token",
        "_closed",
        "dual_source_receipt_hash",
        "mutation_identity_hash",
        "alternate_Q_dropped",
        "solver_authority",
        "schema",
    )
    __hash__ = None

    def __init__(
        self, owner: "V1BlockSupportOutsidePairDualSourceHandle", token: object
    ) -> None:
        if type(owner) is not V1BlockSupportOutsidePairDualSourceHandle:
            raise TypeError("outside-pair dual lease requires the exact owner")
        self._owner = owner
        self._token = token
        self._closed = False
        self.dual_source_receipt_hash = owner.receipt.receipt_hash
        self.mutation_identity_hash = owner.mutation_identity_hash
        self.alternate_Q_dropped = True
        self.solver_authority = False
        self.schema = OUTSIDE_PAIR_DUAL_LEASE_SCHEMA

    def live_route_tensors(self) -> tuple[Any, Any]:
        if self._closed:
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.EVICTED,
                "outside-pair dual lease is closed",
            )
        if self._owner.mutation_identity_hash != self.mutation_identity_hash:
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.MUTATED_CUSTODY,
                "outside-pair dual source changed during its lease",
            )
        resource, capacity_adjoint = self._owner._require_live()
        return resource, capacity_adjoint

    def live_capacity_adjoint(self) -> Any:
        """Return the owned adjoint; CUDA resource stays in KKT custody."""

        return self.live_route_tensors()[1]

    @property
    def compact_duals(
        self,
    ) -> tuple[float, float, tuple[float, ...], tuple[float, ...]]:
        self.live_route_tensors()
        return self._owner._compact_duals

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._owner._close_lease(self._token)
        finally:
            self._closed = True

    def __enter__(self) -> "V1BlockSupportOutsidePairDualLease":
        self.live_route_tensors()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        del exc_type, exc, traceback
        self.close()
        return False

    def __reduce__(self) -> NoReturn:
        raise TypeError("outside-pair dual leases are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("outside-pair dual leases are nonserializable")


class V1BlockSupportOutsidePairDualSourceHandle:
    """Live capacity-dual owner plus CPU-reference resource custody."""

    __slots__ = (
        "_resource",
        "_capacity_adjoint",
        "_resource_guard",
        "_capacity_adjoint_guard",
        "_invariants",
        "_guard_invariants",
        "_compact_duals",
        "_active_leases",
        "_evicted",
        "receipt",
        "design",
        "forecast",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        resource: Any,
        capacity_adjoint: Any,
        compact_duals: tuple[float, float, tuple[float, ...], tuple[float, ...]],
        receipt: V1BlockSupportOutsidePairDualSourceReceipt,
        design: V1BlockSupportOutsidePairCUDAExecutionDesign,
        forecast: V1BlockSupportOutsidePairHBMForecast,
        _issuer_token: object,
    ) -> None:
        if _issuer_token is not _DUAL_HANDLE_TOKEN:
            raise TypeError("outside-pair dual-source handles are issuer-only")
        if (
            type(receipt) is not V1BlockSupportOutsidePairDualSourceReceipt
            or type(design) is not V1BlockSupportOutsidePairCUDAExecutionDesign
            or type(forecast) is not V1BlockSupportOutsidePairHBMForecast
        ):
            raise TypeError("outside-pair dual source requires exact typed receipts")
        production = (
            design.execution_mode
            is V1BlockSupportOutsidePairExecutionMode.CUDA_PRODUCTION
        )
        if production:
            if resource is not None:
                _refuse(
                    V1BlockSupportOutsidePairRefusalCode.AUTHORITY_VIOLATION,
                    "production dual owner may not clone the KKT resource vector",
                )
            resource_tensor = None
        else:
            resource_tensor = _require_fp64_tensor(
                resource,
                name="CPU-reference resource coefficients",
                shape=(receipt.route_count,),
            )
        capacity_tensor = _require_fp64_tensor(
            capacity_adjoint,
            name="capacity adjoint coefficients",
            shape=(receipt.route_count,),
            device=(None if resource_tensor is None else resource_tensor.device),
        )
        if (
            production and receipt.resource_custody_kind != "production_kkt_live_borrow"
        ) or (
            not production
            and receipt.resource_custody_kind != "cpu_reference_owned_copy"
        ):
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.AUTHORITY_VIOLATION,
                "dual resource custody kind differs from its execution mode",
            )
        if (
            receipt.design_hash != design.design_hash
            or receipt.hbm_forecast_hash != forecast.forecast_hash
            or forecast.design_hash != design.design_hash
            or forecast.route_count != receipt.route_count
            or forecast.capacity_count != receipt.capacity_count
            or not forecast.admitted
        ):
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.HBM_FORECAST_EXCEEDED,
                "outside-pair dual source differs from its admitted HBM forecast",
            )
        self._resource = resource_tensor
        self._capacity_adjoint = capacity_tensor
        self._resource_guard = (
            None if resource_tensor is None else resource_tensor.detach().clone()
        )
        self._capacity_adjoint_guard = capacity_tensor.detach().clone()
        live_tensors = (
            (capacity_tensor,)
            if resource_tensor is None
            else (resource_tensor, capacity_tensor)
        )
        guard_tensors = (
            (self._capacity_adjoint_guard,)
            if self._resource_guard is None
            else (self._resource_guard, self._capacity_adjoint_guard)
        )
        self._invariants = tuple(_tensor_invariant(value) for value in live_tensors)
        self._guard_invariants = tuple(
            _tensor_invariant(value) for value in guard_tensors
        )
        self._compact_duals = compact_duals
        self._active_leases: set[object] = set()
        self._evicted = False
        self.receipt = receipt
        self.design = design
        self.forecast = forecast
        self.assert_live()

    def _require_live(self) -> tuple[Any | None, Any]:
        if self._evicted:
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.EVICTED,
                "outside-pair dual source was evicted",
            )
        tensors = (
            (self._capacity_adjoint,)
            if self._resource is None
            else (self._resource, self._capacity_adjoint)
        )
        guards = (
            (self._capacity_adjoint_guard,)
            if self._resource_guard is None
            else (self._resource_guard, self._capacity_adjoint_guard)
        )
        if (
            tuple(_tensor_invariant(value) for value in tensors) != self._invariants
            or tuple(_tensor_invariant(value) for value in guards)
            != self._guard_invariants
            or not all(
                bool(torch.equal(live, guard))
                for live, guard in zip(tensors, guards, strict=True)
            )
        ):
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.MUTATED_CUSTODY,
                "outside-pair route dual tensors changed after sealing",
            )
        self.receipt.assert_integrity()
        return self._resource, self._capacity_adjoint

    def assert_live(self) -> None:
        self._require_live()

    @property
    def mutation_identity_hash(self) -> str:
        self._require_live()
        return content_hash(
            self._invariants,
            prefix=_PREFIX + "dual-live-mutation-identity-",
        )

    @property
    def active_lease_count(self) -> int:
        return len(self._active_leases)

    def open_dual_lease(self) -> V1BlockSupportOutsidePairDualLease:
        self.assert_live()
        token = object()
        self._active_leases.add(token)
        return V1BlockSupportOutsidePairDualLease(self, token)

    def _close_lease(self, token: object) -> None:
        if token not in self._active_leases:
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.STALE_IDENTITY,
                "outside-pair dual lease token is not active",
            )
        try:
            self.assert_live()
        finally:
            self._active_leases.remove(token)

    def evict(self) -> None:
        self.assert_live()
        if self._active_leases:
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.ACTIVE_LEASE,
                "cannot evict an outside-pair dual source with an active lease",
            )
        self._resource = None
        self._capacity_adjoint = None
        self._resource_guard = None
        self._capacity_adjoint_guard = None
        self._invariants = ()
        self._guard_invariants = ()
        self._compact_duals = (0.0, 0.0, (), ())
        self._evicted = True

    def __reduce__(self) -> NoReturn:
        raise TypeError("outside-pair dual-source handles are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("outside-pair dual-source handles are nonserializable")


def _validate_block_alias_sources(
    *,
    block_witness_source: V1BlockSupportWitnessSource,
    alias_binding: V1BlockSupportAliasBinding,
    d_fixed_support: V1SignedWorkBlockSupportSourceDeviceHandle,
    residual_support: V1SignedWorkBlockSupportSourceDeviceHandle,
    preferred_action_binding: V1SignedWorkBlockSupportPreferredActionBinding,
    pair_cover_receipt: V1SignedWorkVariableShellOutsidePairCoverReceipt,
    cover_binding_receipt: V1BlockSupportOutsidePairCoverBindingReceipt | None,
    block_index: int,
) -> tuple[Any, Any, Any]:
    if (
        type(block_witness_source) is not V1BlockSupportWitnessSource
        or type(alias_binding) is not V1BlockSupportAliasBinding
        or type(d_fixed_support) is not V1SignedWorkBlockSupportSourceDeviceHandle
        or type(residual_support) is not V1SignedWorkBlockSupportSourceDeviceHandle
        or type(preferred_action_binding)
        is not V1SignedWorkBlockSupportPreferredActionBinding
        or type(pair_cover_receipt)
        is not V1SignedWorkVariableShellOutsidePairCoverReceipt
        or (
            cover_binding_receipt is not None
            and type(cover_binding_receipt)
            is not V1BlockSupportOutsidePairCoverBindingReceipt
        )
    ):
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.INVALID_TYPE,
            "outside-pair source join requires exact live typed owners",
        )
    index = _nonnegative_int(block_index, name="block index")
    registration = block_witness_source.registration
    registration.assert_integrity()
    block_witness_source.block_context_ids(index)
    preferred_action_binding.assert_live()
    d_fixed_support.assert_live()
    residual_support.assert_live()
    d_receipt = d_fixed_support.receipt
    r_receipt = residual_support.receipt
    if cover_binding_receipt is not None:
        cover_binding_receipt.assert_integrity()
    if (
        d_receipt.source_role is not V1SignedWorkBlockSupportSourceRole.D_FIXED
        or r_receipt.source_role
        is not V1SignedWorkBlockSupportSourceRole.RESIDUAL_BLOCK
        or r_receipt.block_ordinal != index
    ):
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.SOURCE_MISMATCH,
            "outside-pair supports do not have the registered D/residual roles",
        )
    common = (
        "source_registration_hash",
        "model_identity_hash",
        "chart_identity_hash",
        "current_mask_hash",
        "proposal_generation",
        "route_axis_hash",
        "route_id_axis_hash",
        "native_fiber_axis_receipt_hash",
        "native_fiber_measure_hash",
        "registered_d_prefix_hash",
        "registered_d_w2_prefix_hash",
        "preferred_action_binding_hash",
        "preferred_action_hash",
        "preferred_resulting_mask_hash",
        "preferred_removed_route_index_hash",
        "preferred_selected_count",
        "route_count",
    )
    if any(getattr(d_receipt, name) != getattr(r_receipt, name) for name in common):
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.SOURCE_MISMATCH,
            "D and residual support handles do not share one source/action/chart",
        )
    if (
        alias_binding.alias_hash
        not in {value.alias_hash for value in registration.alias_bindings}
        or alias_binding.outside_pair_cover_hash != pair_cover_receipt.receipt_hash
        or registration.preferred_action_hash != d_receipt.preferred_action_hash
        or registration.preferred_action_hash
        != preferred_action_binding.preferred_action_hash
        or registration.preferred_action_cardinality
        != preferred_action_binding.preferred_selected_count
        or registration.chart_hash != d_receipt.chart_identity_hash
        or registration.current_mask_hash != d_receipt.current_mask_hash
        or registration.generation_index != d_receipt.proposal_generation
        or registration.route_axis_hash != d_receipt.route_axis_hash
        or registration.registered_d_prefix_hash != d_receipt.registered_d_prefix_hash
        or registration.registered_d_w2_source_hash
        != d_receipt.registered_d_w2_prefix_hash
        or preferred_action_binding.binding_hash
        != d_receipt.preferred_action_binding_hash
        or pair_cover_receipt.route_count != d_receipt.route_count
    ):
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.ALIAS_MISMATCH,
            "block, alias, support, preferred action, and pair-cover identities differ",
        )
    same_cardinality = (
        alias_binding.cardinality == registration.preferred_action_cardinality
        and pair_cover_receipt.fixed_cardinality
        == registration.preferred_action_cardinality
    )
    if cover_binding_receipt is None:
        if not same_cardinality:
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.ALIAS_MISMATCH,
                "cross-cardinality outside-pair execution requires the typed source cover binding",
            )
    else:
        binding = cover_binding_receipt
        if (
            binding.alias_hash != alias_binding.alias_hash
            or binding.branch_id != alias_binding.branch_id
            or binding.resource_alias_id != alias_binding.resource_alias_id
            or binding.source_receipt_hash != alias_binding.source_receipt_hash
            or binding.source_replay_hash != alias_binding.source_replay_hash
            or binding.legacy_pair_cover_receipt_hash != pair_cover_receipt.receipt_hash
            or binding.legacy_pair_axis_commitment_hash
            != pair_cover_receipt.pair_axis_commitment_hash
            or binding.rounding_anchor_preferred_action_index_hash
            != pair_cover_receipt.preferred_action_index_hash
            or binding.shell_index_hash != pair_cover_receipt.shell_index_hash
            or binding.eligible_mask_hash != pair_cover_receipt.eligible_mask_hash
            or binding.required_mask_hash != pair_cover_receipt.required_mask_hash
            or binding.already_removed_mask_hash
            != pair_cover_receipt.already_removed_mask_hash
            or binding.selectable_mask_hash != pair_cover_receipt.selectable_mask_hash
            or binding.route_count != pair_cover_receipt.route_count
            or binding.rounding_anchor_route_count
            != pair_cover_receipt.fixed_cardinality
            or binding.rounding_anchor_route_count != binding.alias_total_removed_count
            or binding.alias_total_removed_count != alias_binding.cardinality
        ):
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.PAIR_COVER_MISMATCH,
                "typed cover binding differs from alias or compact cover identity",
            )
    return registration, d_receipt, r_receipt


def bind_v1_block_support_outside_pair_dual_source(
    *,
    block_witness_source: V1BlockSupportWitnessSource,
    alias_binding: V1BlockSupportAliasBinding,
    d_fixed_support: V1SignedWorkBlockSupportSourceDeviceHandle,
    residual_support: V1SignedWorkBlockSupportSourceDeviceHandle,
    preferred_action_binding: V1SignedWorkBlockSupportPreferredActionBinding,
    pair_cover_receipt: V1SignedWorkVariableShellOutsidePairCoverReceipt,
    cover_binding_receipt: V1BlockSupportOutsidePairCoverBindingReceipt | None = None,
    block_index: int,
    resource_axis_hash: str,
    resource_coefficient_source_hash: str,
    capacity_operator_source_hash: str,
    resource_multiplier_source_hash: str,
    capacity_multiplier_source_hash: str,
    resource_coefficients: object | None,
    kkt_state: V1SignedWorkLeastWorkCUDAStateHandle | None,
    capacity_matrix: object,
    resource_floor: float,
    capacity_limits: object,
    resource_multiplier: float,
    capacity_multipliers: object,
    design: V1BlockSupportOutsidePairCUDAExecutionDesign,
    hbm_forecast: V1BlockSupportOutsidePairHBMForecast,
) -> V1BlockSupportOutsidePairDualSourceHandle:
    """Bind one feasible resource/capacity dual for the rounding overcover.

    ``capacity_matrix`` is a compact row family ``B``.  Only ``B.T @ mu`` is
    retained by the owner; the input matrix is never copied to the host or
    content-digested and no pair-scale tensor is formed.
    """

    if (
        type(design) is not V1BlockSupportOutsidePairCUDAExecutionDesign
        or type(hbm_forecast) is not V1BlockSupportOutsidePairHBMForecast
    ):
        raise TypeError("outside-pair dual binding requires exact design and forecast")
    registration, d_receipt, _r_receipt = _validate_block_alias_sources(
        block_witness_source=block_witness_source,
        alias_binding=alias_binding,
        d_fixed_support=d_fixed_support,
        residual_support=residual_support,
        preferred_action_binding=preferred_action_binding,
        pair_cover_receipt=pair_cover_receipt,
        cover_binding_receipt=cover_binding_receipt,
        block_index=block_index,
    )
    count = d_receipt.route_count
    if torch is None or type(capacity_matrix) is not torch.Tensor:
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.INVALID_TYPE,
            "outside-pair dual binding requires torch capacity storage",
        )
    if capacity_matrix.ndim != 2 or int(capacity_matrix.shape[1]) != count:
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.INVALID_TYPE,
            "capacity matrix does not align with the route axis",
        )
    capacity_count = int(capacity_matrix.shape[0])
    production = (
        design.execution_mode is V1BlockSupportOutsidePairExecutionMode.CUDA_PRODUCTION
    )
    if production:
        if (
            type(kkt_state) is not V1SignedWorkLeastWorkCUDAStateHandle
            or resource_coefficients is not None
            or resource_coefficient_source_hash != kkt_state.identity_hash
            or kkt_state.route_count != count
        ):
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.SOURCE_MISMATCH,
                "production resource must be borrowed from the exact KKT state",
            )
        with kkt_state.borrow_for_block_support_outside_pair(
            expected_problem_hash=kkt_state.problem_hash,
            expected_policy_hash=kkt_state.policy_hash,
            expected_branch_receipt_hashes=kkt_state.branch_receipt_hashes,
        ) as kkt_lease:
            resource, _eligible, _required = kkt_lease.live_source_tensors()
            _require_fp64_tensor(
                resource,
                name="KKT resource coefficients",
                shape=(count,),
            )
            device = resource.device
    else:
        if kkt_state is not None:
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.CPU_PRODUCTION_FALLBACK,
                "CPU reference dual binding may not borrow production KKT custody",
            )
        resource = _require_fp64_tensor(
            resource_coefficients,
            name="CPU-reference resource coefficients",
            shape=(count,),
        )
        device = resource.device
    capacity = _require_fp64_tensor(
        capacity_matrix,
        name="compact capacity matrix",
        shape=(capacity_count, count),
        device=device,
        nonnegative=True,
    )
    limits = _require_fp64_tensor(
        capacity_limits,
        name="capacity limits",
        shape=(capacity_count,),
        device=device,
    )
    multipliers = _require_fp64_tensor(
        capacity_multipliers,
        name="capacity multipliers",
        shape=(capacity_count,),
        device=device,
        nonnegative=True,
    )
    lam = _finite_nonnegative(resource_multiplier, name="resource multiplier")
    rho = _finite(resource_floor, name="resource floor")
    if production:
        if device.type != "cuda" or not torch.cuda.is_available():
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.CUDA_REQUIRED,
                "production outside-pair dual source requires CUDA-resident operands",
            )
        if not torch.are_deterministic_algorithms_enabled():
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.INVALID_DESIGN,
                "production outside-pair execution requires deterministic algorithms",
            )
    elif device.type != "cpu":
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.CPU_PRODUCTION_FALLBACK,
            "exact-reference outside-pair dual sources must reside on CPU",
        )
    if (
        hbm_forecast.design_hash != design.design_hash
        or hbm_forecast.route_count != count
        or hbm_forecast.capacity_count != capacity_count
        or not hbm_forecast.admitted
    ):
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.HBM_FORECAST_EXCEEDED,
            "outside-pair dual source was not admitted by the exact forecast",
        )
    capacity_adjoint = torch.mv(capacity.transpose(0, 1), multipliers).contiguous()
    if not bool(torch.isfinite(capacity_adjoint).all().item()):
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.NUMERICAL_REFUSAL,
            "capacity dual adjoint became nonfinite",
        )

    # Small O(B) scalar capture is allowed.  No route-scale value or digest
    # crosses the device boundary.
    multiplier_axis = tuple(float(value) for value in multipliers.tolist())
    limit_axis = tuple(float(value) for value in limits.tolist())
    multiplier_axis_hash = content_hash(
        multiplier_axis, prefix=_PREFIX + "capacity-multipliers-"
    )
    limit_axis_hash = content_hash(limit_axis, prefix=_PREFIX + "capacity-limits-")
    mu_kappa = float(torch.dot(multipliers, limits).item())
    if not math.isfinite(mu_kappa):
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.NUMERICAL_REFUSAL,
            "capacity dual constant became nonfinite",
        )

    lam_radius = _ulp_radius(lam)
    rho_radius = _ulp_radius(rho)
    if capacity_count:
        next_up_mu = torch.nextafter(
            multipliers, torch.full_like(multipliers, math.inf)
        )
        next_down_mu = torch.nextafter(
            multipliers, torch.full_like(multipliers, -math.inf)
        )
        mu_radius = torch.maximum(
            (next_up_mu - multipliers).abs(),
            (multipliers - next_down_mu).abs(),
        )
        next_up_kappa = torch.nextafter(limits, torch.full_like(limits, math.inf))
        next_down_kappa = torch.nextafter(limits, torch.full_like(limits, -math.inf))
        kappa_radius = torch.maximum(
            (next_up_kappa - limits).abs(),
            (limits - next_down_kappa).abs(),
        )
        absolute_adjoint_mass = torch.mv(
            capacity.abs().transpose(0, 1), multipliers.abs()
        )
        mu_conversion_adjoint = torch.mv(capacity.abs().transpose(0, 1), mu_radius)
        matmul_gamma = _gamma_n(capacity_count)
        capacity_radius_axis = (
            matmul_gamma * absolute_adjoint_mass
            + mu_conversion_adjoint
            + float(capacity_count) * _F64_MIN_NORMAL
        )
        capacity_local_operations = 8 * capacity_count + 8
        capacity_l1_radius = _nonnegative_arithmetic_upper(
            float(capacity_radius_axis.sum().item()),
            operation_count=capacity_local_operations * count + count,
        )
        capacity_linf_radius = _nonnegative_arithmetic_upper(
            float(capacity_radius_axis.max().item()),
            operation_count=capacity_local_operations,
        )
        dot_abs_mass = _nonnegative_arithmetic_upper(
            float((multipliers.abs() * limits.abs()).sum().item()),
            operation_count=3 * capacity_count + 1,
        )
        dot_conversion = _nonnegative_arithmetic_upper(
            float(
                (
                    limits.abs() * mu_radius
                    + multipliers.abs() * kappa_radius
                    + mu_radius * kappa_radius
                )
                .sum()
                .item()
            ),
            operation_count=10 * capacity_count + 1,
        )
        dot_radius = _nonnegative_arithmetic_upper(
            _gamma_n(capacity_count) * dot_abs_mass
            + dot_conversion
            + float(capacity_count) * _F64_MIN_NORMAL,
            operation_count=6,
        )
    else:
        capacity_l1_radius = 0.0
        capacity_linf_radius = 0.0
        dot_radius = 0.0
    lambda_rho_mass = abs(lam * rho)
    lambda_rho_radius = _nonnegative_arithmetic_upper(
        _gamma_n(1) * lambda_rho_mass
        + abs(rho) * lam_radius
        + abs(lam) * rho_radius
        + lam_radius * rho_radius
        + _F64_MIN_NORMAL,
        operation_count=12,
    )
    constant_mass = lambda_rho_mass + abs(mu_kappa)
    constant_radius = _nonnegative_arithmetic_upper(
        lambda_rho_radius + dot_radius + _gamma_n(1) * constant_mass + _F64_MIN_NORMAL,
        operation_count=8,
    )
    receipt = V1BlockSupportOutsidePairDualSourceReceipt(
        alias_hash=alias_binding.alias_hash,
        branch_id=alias_binding.branch_id,
        fd_rank=alias_binding.fd_rank,
        resource_alias_id=alias_binding.resource_alias_id,
        source_receipt_hash=alias_binding.source_receipt_hash,
        source_replay_hash=alias_binding.source_replay_hash,
        cardinality_alias_id=alias_binding.cardinality_alias_id,
        cardinality=alias_binding.cardinality,
        constraint_cover_hash=alias_binding.constraint_cover_hash,
        outside_pair_cover_hash=pair_cover_receipt.receipt_hash,
        pair_axis_commitment_hash=pair_cover_receipt.pair_axis_commitment_hash,
        model_identity_hash=d_receipt.model_identity_hash,
        chart_identity_hash=d_receipt.chart_identity_hash,
        current_mask_hash=d_receipt.current_mask_hash,
        proposal_generation=d_receipt.proposal_generation,
        route_axis_hash=d_receipt.route_axis_hash,
        route_id_axis_hash=d_receipt.route_id_axis_hash,
        registered_d_prefix_hash=d_receipt.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=d_receipt.registered_d_w2_prefix_hash,
        preferred_action_hash=preferred_action_binding.preferred_action_hash,
        preferred_action_binding_hash=preferred_action_binding.binding_hash,
        resource_axis_hash=_identity(resource_axis_hash, name="resource axis hash"),
        resource_coefficient_source_hash=_identity(
            resource_coefficient_source_hash,
            name="resource coefficient source hash",
        ),
        capacity_operator_source_hash=_identity(
            capacity_operator_source_hash,
            name="capacity operator source hash",
        ),
        resource_multiplier_source_hash=_identity(
            resource_multiplier_source_hash,
            name="resource multiplier source hash",
        ),
        capacity_multiplier_source_hash=_identity(
            capacity_multiplier_source_hash,
            name="capacity multiplier source hash",
        ),
        route_count=count,
        capacity_count=capacity_count,
        resource_multiplier_nominal=lam,
        resource_floor_nominal=rho,
        capacity_multiplier_dot_limit_nominal=mu_kappa,
        resource_multiplier_conversion_radius=lam_radius,
        capacity_adjoint_roundoff_l1_radius=capacity_l1_radius,
        capacity_adjoint_roundoff_linf_radius=capacity_linf_radius,
        dual_constant_roundoff_radius=constant_radius,
        capacity_multiplier_axis_hash=multiplier_axis_hash,
        capacity_limit_axis_hash=limit_axis_hash,
        design_hash=design.design_hash,
        hbm_forecast_hash=hbm_forecast.forecast_hash,
        device_type=device.type,
        resource_custody_kind=(
            "production_kkt_live_borrow" if production else "cpu_reference_owned_copy"
        ),
        _issuer_token=_DUAL_RECEIPT_TOKEN,
    )
    compact_duals = (lam, rho, multiplier_axis, limit_axis)
    return V1BlockSupportOutsidePairDualSourceHandle(
        resource=(
            None
            if production
            else resource.detach().clone(memory_format=torch.contiguous_format)
        ),
        # ``capacity_adjoint`` is freshly derived by this binder and is
        # transferred directly into the owner; an extra route-scale clone
        # would add an unforecast transient N-vector.
        capacity_adjoint=capacity_adjoint.detach(),
        compact_duals=compact_duals,
        receipt=receipt,
        design=design,
        forecast=hbm_forecast,
        _issuer_token=_DUAL_HANDLE_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1BlockSupportOutsidePairFamilyReceipt:
    block_support_registration_hash: str
    block_index: int
    block_id: str
    block_context_axis_hash: str
    d_fixed_support_receipt_hash: str
    residual_support_receipt_hash: str
    support_source_registration_hash: str
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    preferred_action_hash: str
    preferred_action_binding_hash: str
    preferred_resulting_mask_hash: str
    preferred_removed_route_index_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    alias_hash: str
    branch_id: str
    fd_rank: int
    resource_alias_id: str
    source_receipt_hash: str
    source_replay_hash: str
    cardinality_alias_id: str
    rounded_subset_cardinality: int
    alias_fixed_outside_removed_count: int
    alias_shell_prefix_count: int
    rounding_anchor_selected_count: int
    comparison_preferred_selected_count: int
    rounding_anchor_action_coordinate_hash: str
    rounding_anchor_removed_route_index_hash: str
    rounding_anchor_subset_identity_hash: str
    rounding_anchor_shell_gate_hash: str
    constraint_cover_hash: str
    pair_cover_receipt_hash: str
    pair_axis_commitment_hash: str
    typed_cover_binding_receipt_hash: str | None
    membership_source_receipt_hash: str
    membership_source_kind: str
    structural_domain_receipt_hash: str | None
    physical_block_support_binding_receipt_hash: str | None
    fixed_membership_binding_hash: str | None
    shell_freeze_receipt_hash: str | None
    kkt_state_identity_hash: str | None
    kkt_problem_hash: str | None
    kkt_policy_hash: str | None
    kkt_branch_receipt_hash: str | None
    dual_source_receipt_hash: str
    resource_axis_hash: str
    resource_coefficient_source_hash: str
    capacity_operator_source_hash: str
    resource_multiplier_source_hash: str
    capacity_multiplier_source_hash: str
    route_count: int
    capacity_count: int
    route_tile_size: int
    route_tile_scan_count: int
    complete_route_coefficient_scan_count: int
    add_outside_rectangle_pair_count: int
    drop_outside_add_inside_rectangle_pair_count: int
    complete_pair_cover_count: int
    add_outside_rectangle_empty: bool
    drop_outside_add_inside_rectangle_empty: bool
    vacuous_no_outside_pair: bool
    base_dual_nominal: float
    base_dual_downward_radius: float
    base_dual_lower_outward: float
    comparison_preferred_dot_nominal: float
    comparison_preferred_dot_absolute_radius: float
    comparison_preferred_dot_upper_outward: float
    support_coefficient_source_l1_radius: float
    weight_conversion_l1_radius: float
    weighted_combination_roundoff_l1_radius: float
    resource_dual_roundoff_l1_radius: float
    capacity_dual_roundoff_l1_radius: float
    stationarity_addition_roundoff_l1_radius: float
    box_reduction_roundoff_radius: float
    dual_constant_roundoff_radius: float
    stationarity_sign_linf_radius: float
    ambiguous_p_endpoint_count: int
    ambiguous_q_endpoint_count: int
    add_outside_correction_lower: float
    drop_outside_add_inside_correction_lower: float
    add_outside_pair_lower_outward: float
    drop_outside_add_inside_pair_lower_outward: float
    worst_pair_rectangle: V1SignedWorkVariableShellOutsidePairRectangle | None
    worst_pair_preferred_selected_index: int | None
    worst_pair_preferred_unselected_index: int | None
    worst_pair_ordinal: int | None
    worst_pair_lower_center: float
    worst_pair_total_downward_radius: float
    worst_pair_lower_outward: float
    exact_reference_worst_pair_numerator: int | None
    exact_reference_worst_pair_denominator: int | None
    design_hash: str
    hbm_forecast_hash: str
    execution_mode: V1BlockSupportOutsidePairExecutionMode
    production_typed_membership_sources_joined: bool
    cpu_reference_four_mask_only: bool
    scalar_device_to_host_transfer_bytes: int
    scan_device_to_host_synchronization_count: int
    route_scale_device_to_host_transfer_bytes: int = 0
    route_scale_tensor_digest_bytes: int = 0
    pair_matrix_bytes: int = 0
    alternate_Q_dropped: bool = True
    relaxed_cardinality_equality_enforced: bool = False
    rounded_subset_cardinality_identity_only: bool = True
    comparison_preferred_is_rounding_anchor: bool = True
    same_cardinality_alias_only: bool = True
    cross_cardinality_alias_authority: bool = False
    typed_cover_equivalence_joined: bool = False
    complete_two_rectangle_pair_cover_scanned: bool = True
    deterministic_physical_index_ties: bool = True
    deterministic_for_frozen_design_only: bool = True
    cross_tile_size_nominal_identity_authority: bool = False
    ambiguous_sign_correction_lower_is_zero: bool = True
    array_free_receipt: bool = True
    canonical_block_observation_authority: bool = False
    complete_physical_alternate_union_authority: bool = False
    full_u_authority: bool = False
    fqr_authority: bool = False
    action_invariance_authority: bool = False
    c_authority: bool = False
    k_authority: bool = False
    o_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    equation: str = OUTSIDE_PAIR_EQUATION
    authority_scope: str = OUTSIDE_PAIR_AUTHORITY_SCOPE
    schema: str = OUTSIDE_PAIR_FAMILY_SCHEMA
    _issuer_token: InitVar[object] = None
    _integrity_seal: str = field(default="", repr=False, compare=False)

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _FAMILY_TOKEN:
            raise TypeError("outside-pair family receipts are issuer-only")
        for item in fields(self):
            if (
                not item.name.startswith("_")
                and item.name.endswith("_hash")
                and getattr(self, item.name) is not None
            ):
                object.__setattr__(
                    self,
                    item.name,
                    _identity(
                        getattr(self, item.name), name=item.name.replace("_", " ")
                    ),
                )
        for name in (
            "block_id",
            "branch_id",
            "resource_alias_id",
            "source_receipt_hash",
            "source_replay_hash",
            "cardinality_alias_id",
            "membership_source_kind",
        ):
            object.__setattr__(
                self, name, _identity(getattr(self, name), name=name.replace("_", " "))
            )
        for name in (
            "fd_rank",
            "route_count",
            "route_tile_size",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name.replace("_", " ")),
            )
        for name in (
            "block_index",
            "proposal_generation",
            "rounded_subset_cardinality",
            "alias_fixed_outside_removed_count",
            "alias_shell_prefix_count",
            "rounding_anchor_selected_count",
            "comparison_preferred_selected_count",
            "capacity_count",
            "route_tile_scan_count",
            "complete_route_coefficient_scan_count",
            "add_outside_rectangle_pair_count",
            "drop_outside_add_inside_rectangle_pair_count",
            "complete_pair_cover_count",
            "ambiguous_p_endpoint_count",
            "ambiguous_q_endpoint_count",
            "scalar_device_to_host_transfer_bytes",
            "scan_device_to_host_synchronization_count",
            "route_scale_device_to_host_transfer_bytes",
            "route_scale_tensor_digest_bytes",
            "pair_matrix_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name.replace("_", " ")),
            )
        for name in (
            "base_dual_nominal",
            "base_dual_lower_outward",
            "comparison_preferred_dot_nominal",
            "comparison_preferred_dot_upper_outward",
            "add_outside_pair_lower_outward",
            "drop_outside_add_inside_pair_lower_outward",
            "worst_pair_lower_center",
            "worst_pair_lower_outward",
        ):
            value = float(getattr(self, name))
            if math.isnan(value):
                _refuse(
                    V1BlockSupportOutsidePairRefusalCode.NUMERICAL_REFUSAL,
                    f"{name} may not be NaN",
                )
            object.__setattr__(self, name, value)
        for name in (
            "base_dual_downward_radius",
            "comparison_preferred_dot_absolute_radius",
            "support_coefficient_source_l1_radius",
            "weight_conversion_l1_radius",
            "weighted_combination_roundoff_l1_radius",
            "resource_dual_roundoff_l1_radius",
            "capacity_dual_roundoff_l1_radius",
            "stationarity_addition_roundoff_l1_radius",
            "box_reduction_roundoff_radius",
            "dual_constant_roundoff_radius",
            "stationarity_sign_linf_radius",
            "add_outside_correction_lower",
            "drop_outside_add_inside_correction_lower",
            "worst_pair_total_downward_radius",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name=name.replace("_", " ")),
            )
        if type(self.execution_mode) is not V1BlockSupportOutsidePairExecutionMode:
            raise TypeError("outside-pair family execution mode has another type")
        production_join_fields = (
            self.structural_domain_receipt_hash,
            self.physical_block_support_binding_receipt_hash,
            self.fixed_membership_binding_hash,
            self.shell_freeze_receipt_hash,
            self.kkt_state_identity_hash,
            self.kkt_problem_hash,
            self.kkt_policy_hash,
            self.kkt_branch_receipt_hash,
        )
        if (
            self.execution_mode
            is V1BlockSupportOutsidePairExecutionMode.CUDA_PRODUCTION
        ):
            if (
                self.production_typed_membership_sources_joined is not True
                or self.cpu_reference_four_mask_only is not False
                or any(value is None for value in production_join_fields)
                or self.membership_source_kind
                != "structural_physical_plus_kkt_live_cuda"
                or self.typed_cover_binding_receipt_hash is None
            ):
                _refuse(
                    V1BlockSupportOutsidePairRefusalCode.AUTHORITY_VIOLATION,
                    "production receipt lacks the typed structural/KKT source join",
                )
        elif (
            self.production_typed_membership_sources_joined is not False
            or self.cpu_reference_four_mask_only is not True
            or any(value is not None for value in production_join_fields)
            or self.membership_source_kind != "cpu_reference_four_mask"
        ):
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.AUTHORITY_VIOLATION,
                "CPU reference receipt crossed into production source custody",
            )
        if self.complete_route_coefficient_scan_count != self.route_count:
            raise ValueError("outside-pair family did not scan the complete route axis")
        if self.scan_device_to_host_synchronization_count != 1:
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.AUTHORITY_VIOLATION,
                "outside-pair execution must use exactly one compact D2H synchronization",
            )
        if (
            self.add_outside_rectangle_pair_count
            + self.drop_outside_add_inside_rectangle_pair_count
            != self.complete_pair_cover_count
        ):
            raise ValueError("outside-pair family rectangle counts do not close")
        if (
            self.add_outside_rectangle_empty
            is not (self.add_outside_rectangle_pair_count == 0)
            or self.drop_outside_add_inside_rectangle_empty
            is not (self.drop_outside_add_inside_rectangle_pair_count == 0)
            or self.vacuous_no_outside_pair is not (self.complete_pair_cover_count == 0)
        ):
            raise ValueError("outside-pair empty-rectangle flags are inconsistent")
        pair_fields = (
            self.worst_pair_rectangle,
            self.worst_pair_preferred_selected_index,
            self.worst_pair_preferred_unselected_index,
            self.worst_pair_ordinal,
        )
        if self.vacuous_no_outside_pair:
            if any(value is not None for value in pair_fields) or not math.isinf(
                self.worst_pair_lower_outward
            ):
                raise ValueError("vacuous outside-pair receipt has a worst pair")
        else:
            if (
                type(self.worst_pair_rectangle)
                is not V1SignedWorkVariableShellOutsidePairRectangle
                or any(value is None for value in pair_fields[1:])
                or not math.isfinite(self.worst_pair_lower_outward)
            ):
                raise ValueError("nonvacuous outside-pair receipt lacks a worst pair")
            for name in (
                "worst_pair_preferred_selected_index",
                "worst_pair_preferred_unselected_index",
                "worst_pair_ordinal",
            ):
                object.__setattr__(
                    self,
                    name,
                    _nonnegative_int(getattr(self, name), name=name.replace("_", " ")),
                )
        if (self.exact_reference_worst_pair_numerator is None) != (
            self.exact_reference_worst_pair_denominator is None
        ) or (
            self.exact_reference_worst_pair_denominator is not None
            and self.exact_reference_worst_pair_denominator <= 0
        ):
            raise ValueError("exact-reference rational enclosure is malformed")
        required_true = (
            self.alternate_Q_dropped,
            self.rounded_subset_cardinality_identity_only,
            self.complete_two_rectangle_pair_cover_scanned,
            self.deterministic_physical_index_ties,
            self.deterministic_for_frozen_design_only,
            self.ambiguous_sign_correction_lower_is_zero,
            self.array_free_receipt,
        )
        if any(type(value) is not bool or value is not True for value in required_true):
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.AUTHORITY_VIOLATION,
                "outside-pair family lost a mandatory theorem invariant",
            )
        for name in (
            "comparison_preferred_is_rounding_anchor",
            "same_cardinality_alias_only",
            "cross_cardinality_alias_authority",
            "typed_cover_equivalence_joined",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact boolean")
        cross_cardinality = (
            self.rounded_subset_cardinality != self.comparison_preferred_selected_count
        )
        if (
            self.same_cardinality_alias_only is not (not cross_cardinality)
            or self.cross_cardinality_alias_authority
            is not (cross_cardinality and self.typed_cover_equivalence_joined)
            or self.typed_cover_equivalence_joined
            is not (self.typed_cover_binding_receipt_hash is not None)
            or (
                self.comparison_preferred_is_rounding_anchor
                is not (
                    self.rounding_anchor_removed_route_index_hash
                    == self.preferred_removed_route_index_hash
                )
            )
            or self.rounded_subset_cardinality
            != self.alias_fixed_outside_removed_count + self.alias_shell_prefix_count
            or self.rounding_anchor_selected_count != self.rounded_subset_cardinality
            or self.rounding_anchor_selected_count > self.route_count
        ):
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.AUTHORITY_VIOLATION,
                "outside-pair same/cross-cardinality or anchor identities are inconsistent",
            )
        if type(self.relaxed_cardinality_equality_enforced) is not bool or (
            self.relaxed_cardinality_equality_enforced
        ):
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.AUTHORITY_VIOLATION,
                "outside-pair rounding overcover may not impose cardinality equality",
            )
        _closed_authority(
            (
                self.canonical_block_observation_authority,
                self.complete_physical_alternate_union_authority,
                self.full_u_authority,
                self.fqr_authority,
                self.action_invariance_authority,
                self.cross_tile_size_nominal_identity_authority,
                self.c_authority,
                self.k_authority,
                self.o_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="outside-pair family receipt",
        )
        if (
            self.route_scale_device_to_host_transfer_bytes != 0
            or self.route_scale_tensor_digest_bytes != 0
            or self.pair_matrix_bytes != 0
            or self.equation != OUTSIDE_PAIR_EQUATION
            or self.authority_scope != OUTSIDE_PAIR_AUTHORITY_SCOPE
            or self.schema != OUTSIDE_PAIR_FAMILY_SCHEMA
        ):
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.AUTHORITY_VIOLATION,
                "outside-pair family crossed its memory or authority boundary",
            )
        observed = content_hash(_payload(self), prefix=_PREFIX + "family-receipt-")
        if self._integrity_seal and self._integrity_seal != observed:
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.MUTATED_CUSTODY,
                "outside-pair family receipt was mutated",
            )
        object.__setattr__(self, "_integrity_seal", observed)

    def assert_integrity(self) -> str:
        observed = content_hash(_payload(self), prefix=_PREFIX + "family-receipt-")
        if observed != self._integrity_seal:
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.MUTATED_CUSTODY,
                "outside-pair family receipt integrity changed",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def to_dict(self) -> dict[str, object]:
        return {**_payload(self), "receipt_hash": self.receipt_hash}


@dataclass(frozen=True, slots=True)
class _EndpointScan:
    count: int
    ambiguous_count: int
    first_index: int | None
    first_rank: int | None
    first_ambiguous_index: int | None
    first_ambiguous_rank: int | None
    certified_minimum_lower: float
    certified_minimum_index: int | None
    certified_minimum_rank: int | None


@dataclass(frozen=True, slots=True)
class _RectangleScan:
    rectangle: V1SignedWorkVariableShellOutsidePairRectangle
    pair_count: int
    p_count: int
    q_count: int
    correction_lower: float
    p_index: int | None
    q_index: int | None
    pair_ordinal: int | None
    ambiguous_p_count: int
    ambiguous_q_count: int


@dataclass(frozen=True, slots=True)
class _NumericScan:
    base_nominal: float
    base_radius: float
    base_lower: float
    support_source_l1_radius: float
    weight_conversion_l1_radius: float
    weighted_combination_l1_radius: float
    resource_dual_l1_radius: float
    capacity_dual_l1_radius: float
    stationarity_addition_l1_radius: float
    box_reduction_radius: float
    constant_radius: float
    stationarity_sign_linf_radius: float
    preferred_dot_nominal: float
    preferred_dot_reduction_radius: float
    rounding_anchor_count: int
    comparison_preferred_count: int
    required_count: int
    selectable_count: int
    shell_count: int
    first_rectangle: _RectangleScan
    second_rectangle: _RectangleScan
    worst_rectangle: _RectangleScan | None
    worst_center: float
    worst_radius: float
    worst_lower: float
    route_tile_scan_count: int
    scalar_d2h_bytes: int
    d2h_synchronization_count: int


def _weight_and_conversion_radius(
    numerator: int, denominator: int
) -> tuple[float, float]:
    exact = Fraction(numerator, denominator)
    nominal = float(exact)
    radius_exact = abs(Fraction.from_float(nominal) - exact)
    radius = float(radius_exact)
    if Fraction.from_float(radius) < radius_exact:
        radius = math.nextafter(radius, math.inf)
    return nominal, radius


def _first_and_minimum_on_mask(
    *,
    mask: Any,
    certified: Any,
    correction_values: Any,
    physical_indices: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    """Return device scalars; callers perform one final scalar transfer."""

    sentinel = _INDEX_SENTINEL
    first_index = torch.where(
        mask,
        physical_indices,
        torch.full_like(physical_indices, sentinel),
    ).min()
    ambiguous = mask & ~certified
    first_ambiguous = torch.where(
        ambiguous,
        physical_indices,
        torch.full_like(physical_indices, sentinel),
    ).min()
    masked_values = torch.where(
        mask & certified,
        correction_values,
        torch.full_like(correction_values, math.inf),
    )
    minimum_value = masked_values.min()
    minimum_index = torch.where(
        (mask & certified) & (masked_values == minimum_value),
        physical_indices,
        torch.full_like(physical_indices, sentinel),
    ).min()
    local_ranks = torch.cumsum(mask.to(dtype=torch.int64), dim=0) - 1
    first_rank = torch.where(
        mask & (physical_indices == first_index),
        local_ranks,
        torch.full_like(local_ranks, sentinel),
    ).min()
    first_ambiguous_rank = torch.where(
        ambiguous & (physical_indices == first_ambiguous),
        local_ranks,
        torch.full_like(local_ranks, sentinel),
    ).min()
    minimum_rank = torch.where(
        (mask & certified) & (physical_indices == minimum_index),
        local_ranks,
        torch.full_like(local_ranks, sentinel),
    ).min()
    return (
        mask.sum(dtype=torch.int64),
        ambiguous.sum(dtype=torch.int64),
        first_index,
        first_rank,
        first_ambiguous,
        first_ambiguous_rank,
        minimum_value,
        minimum_index,
        minimum_rank,
    )


def _merge_endpoint_tile(
    state: tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any] | None,
    tile: tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any],
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    if state is None:
        return tile
    count = state[0] + tile[0]
    ambiguous = state[1] + tile[1]
    first = torch.minimum(state[2], tile[2])
    first_rank = torch.where(tile[2] < state[2], tile[3] + state[0], state[3])
    first_ambiguous = torch.minimum(state[4], tile[4])
    first_ambiguous_rank = torch.where(tile[4] < state[4], tile[5] + state[0], state[5])
    state_value, state_index = state[6], state[7]
    tile_value, tile_index = tile[6], tile[7]
    choose_tile = (tile_value < state_value) | (
        (tile_value == state_value) & (tile_index < state_index)
    )
    minimum_value = torch.where(choose_tile, tile_value, state_value)
    minimum_index = torch.where(choose_tile, tile_index, state_index)
    minimum_rank = torch.where(choose_tile, tile[8] + state[0], state[8])
    return (
        count,
        ambiguous,
        first,
        first_rank,
        first_ambiguous,
        first_ambiguous_rank,
        minimum_value,
        minimum_index,
        minimum_rank,
    )


def _decode_endpoint_state(values: Sequence[float]) -> _EndpointScan:
    sentinel = _INDEX_SENTINEL
    count = int(values[0])
    ambiguous = int(values[1])
    first_raw = int(values[2])
    first_rank_raw = int(values[3])
    first_ambiguous_raw = int(values[4])
    first_ambiguous_rank_raw = int(values[5])
    minimum = float(values[6])
    minimum_index_raw = int(values[7])
    minimum_rank_raw = int(values[8])
    return _EndpointScan(
        count=count,
        ambiguous_count=ambiguous,
        first_index=None if first_raw == sentinel else first_raw,
        first_rank=None if first_rank_raw == sentinel else first_rank_raw,
        first_ambiguous_index=(
            None if first_ambiguous_raw == sentinel else first_ambiguous_raw
        ),
        first_ambiguous_rank=(
            None if first_ambiguous_rank_raw == sentinel else first_ambiguous_rank_raw
        ),
        certified_minimum_lower=minimum,
        certified_minimum_index=(
            None if minimum_index_raw == sentinel else minimum_index_raw
        ),
        certified_minimum_rank=(
            None if minimum_rank_raw == sentinel else minimum_rank_raw
        ),
    )


def _rectangle_from_endpoints(
    *,
    rectangle: V1SignedWorkVariableShellOutsidePairRectangle,
    p: _EndpointScan,
    q: _EndpointScan,
    ordinal_offset: int,
) -> _RectangleScan:
    pair_count = p.count * q.count
    if pair_count == 0:
        return _RectangleScan(
            rectangle=rectangle,
            pair_count=0,
            p_count=p.count,
            q_count=q.count,
            correction_lower=math.inf,
            p_index=None,
            q_index=None,
            pair_ordinal=None,
            ambiguous_p_count=p.ambiguous_count,
            ambiguous_q_count=q.ambiguous_count,
        )
    if p.ambiguous_count or q.ambiguous_count:
        correction = 0.0
        p_index = (
            p.first_ambiguous_index
            if p.first_ambiguous_index is not None
            else p.first_index
        )
        p_rank = (
            p.first_ambiguous_rank
            if p.first_ambiguous_rank is not None
            else p.first_rank
        )
        q_index = (
            q.first_ambiguous_index
            if q.first_ambiguous_index is not None
            else q.first_index
        )
        q_rank = (
            q.first_ambiguous_rank
            if q.first_ambiguous_rank is not None
            else q.first_rank
        )
    elif p.certified_minimum_lower <= q.certified_minimum_lower:
        correction = max(0.0, p.certified_minimum_lower)
        p_index = p.certified_minimum_index
        p_rank = p.certified_minimum_rank
        q_index = q.first_index
        q_rank = q.first_rank
    else:
        correction = max(0.0, q.certified_minimum_lower)
        p_index = p.first_index
        p_rank = p.first_rank
        q_index = q.certified_minimum_index
        q_rank = q.certified_minimum_rank
    if p_index is None or q_index is None or p_rank is None or q_rank is None:
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.NUMERICAL_REFUSAL,
            "nonempty rectangle lost a deterministic endpoint witness",
        )
    ordinal = ordinal_offset + p_rank * q.count + q_rank
    return _RectangleScan(
        rectangle=rectangle,
        pair_count=pair_count,
        p_count=p.count,
        q_count=q.count,
        correction_lower=(0.0 if correction <= 0.0 else _down(correction)),
        p_index=p_index,
        q_index=q_index,
        pair_ordinal=ordinal,
        ambiguous_p_count=p.ambiguous_count,
        ambiguous_q_count=q.ambiguous_count,
    )


def _scan_live_outside_pairs(
    *,
    design: V1BlockSupportOutsidePairCUDAExecutionDesign,
    d_coefficients: Any,
    residual_coefficients: Any,
    resource_coefficients: Any,
    capacity_adjoint: Any,
    required: Any,
    rounding_anchor: Any | None = None,
    comparison_preferred: Any | None = None,
    selectable: Any | None = None,
    shell: Any | None = None,
    d_weight_numerator: int,
    residual_weight_numerator: int,
    population_weight_denominator: int,
    d_source_l1_radius: float,
    residual_source_l1_radius: float,
    resource_multiplier: float,
    resource_floor: float,
    capacity_multiplier_dot_limit: float,
    resource_multiplier_conversion_radius: float,
    capacity_adjoint_l1_radius: float,
    capacity_adjoint_linf_radius: float,
    dual_constant_radius: float,
    eligible: Any | None = None,
    fixed_membership: Any | None = None,
    resident_parent_packed: Any | None = None,
    shell_indices: tuple[int, ...] = (),
    rounding_anchor_shell_indices: tuple[int, ...] = (),
    comparison_preferred_shell_indices: tuple[int, ...] = (),
) -> _NumericScan:
    count = int(d_coefficients.numel())
    if count >= _INDEX_SENTINEL:
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.INVALID_DESIGN,
            "route axis exceeds the exact compact FP64 index-transfer range",
        )
    device = d_coefficients.device
    for value, name in (
        (residual_coefficients, "residual support coefficients"),
        (resource_coefficients, "resource coefficients"),
        (capacity_adjoint, "capacity adjoint"),
    ):
        _require_fp64_tensor_layout(
            value,
            name=name,
            shape=(count,),
            device=device,
        )
    _require_bool_tensor(
        required,
        name="required membership",
        shape=(count,),
        device=device,
    )
    tiled_structural_membership = rounding_anchor is None
    if tiled_structural_membership:
        if comparison_preferred is not None:
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.INVALID_TYPE,
                "structural scan derives both actions from typed structural custody",
            )
        for value, name in (
            (eligible, "eligible membership"),
            (fixed_membership, "fixed membership"),
        ):
            _require_bool_tensor(
                value,
                name=name,
                shape=(count,),
                device=device,
            )
        if (
            torch is None
            or type(resident_parent_packed) is not torch.Tensor
            or resident_parent_packed.dtype is not torch.uint8
            or tuple(int(item) for item in resident_parent_packed.shape)
            != ((count + 7) // 8,)
            or resident_parent_packed.device != device
            or not resident_parent_packed.is_contiguous()
            or shell_indices != tuple(sorted(set(shell_indices)))
            or rounding_anchor_shell_indices
            != tuple(sorted(set(rounding_anchor_shell_indices)))
            or comparison_preferred_shell_indices
            != tuple(sorted(set(comparison_preferred_shell_indices)))
            or not set(rounding_anchor_shell_indices).issubset(shell_indices)
            or not set(comparison_preferred_shell_indices).issubset(shell_indices)
            or any(index < 0 or index >= count for index in shell_indices)
        ):
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.SOURCE_MISMATCH,
                "structural membership inputs do not close on one packed route axis",
            )
    else:
        if (
            eligible is not None
            or fixed_membership is not None
            or resident_parent_packed is not None
        ):
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.INVALID_TYPE,
                "reference and structural membership paths may not be mixed",
            )
        for value, name in (
            (rounding_anchor, "rounding-anchor membership"),
            (comparison_preferred, "comparison-preferred membership"),
            (selectable, "selectable membership"),
            (shell, "shell membership"),
        ):
            _require_bool_tensor(
                value,
                name=name,
                shape=(count,),
                device=device,
            )
    d_weight, d_weight_radius = _weight_and_conversion_radius(
        d_weight_numerator, population_weight_denominator
    )
    residual_weight, residual_weight_radius = _weight_and_conversion_radius(
        residual_weight_numerator, population_weight_denominator
    )
    support_source_l1 = _nonnegative_arithmetic_upper(
        abs(d_weight) * d_source_l1_radius
        + abs(residual_weight) * residual_source_l1_radius
        + d_weight_radius * d_source_l1_radius
        + residual_weight_radius * residual_source_l1_radius,
        operation_count=8,
    )
    zero = torch.zeros((), dtype=torch.float64, device=device)
    base_terms = zero.clone()
    base_abs_terms = zero.clone()
    preferred_dot_terms = zero.clone()
    preferred_dot_abs_terms = zero.clone()
    weight_conversion_l1 = zero.clone()
    weighted_combination_l1 = zero.clone()
    resource_dual_l1 = zero.clone()
    stationarity_addition_l1 = zero.clone()
    sign_linf = zero.clone()
    finite_ok = torch.ones((), dtype=torch.bool, device=device)
    legality_ok = torch.ones((), dtype=torch.bool, device=device)
    membership_counts = torch.zeros((5,), dtype=torch.int64, device=device)
    endpoint_states: list[tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any] | None] = [
        None,
        None,
        None,
        None,
    ]
    tile_count = 0
    for start in range(0, count, design.route_tile_size):
        stop = min(count, start + design.route_tile_size)
        tile_count += 1
        d_tile = d_coefficients[start:stop]
        residual_tile = residual_coefficients[start:stop]
        resource_tile = resource_coefficients[start:stop]
        capacity_tile = capacity_adjoint[start:stop]
        required_tile = required[start:stop]
        if tiled_structural_membership:
            assert eligible is not None
            assert fixed_membership is not None
            assert resident_parent_packed is not None
            physical = torch.arange(start, stop, dtype=torch.int64, device=device)
            byte_indices = torch.bitwise_right_shift(physical, 3)
            shifts = 7 - torch.bitwise_and(physical, 7)
            parent_live = torch.bitwise_and(
                torch.bitwise_right_shift(
                    resident_parent_packed.index_select(0, byte_indices), shifts
                ),
                1,
            ).to(dtype=torch.bool)
            selectable_tile = eligible[start:stop] & parent_live
            shell_tile = torch.zeros((stop - start,), dtype=torch.bool, device=device)
            rounding_anchor_tile = fixed_membership[start:stop].clone()
            comparison_preferred_tile = fixed_membership[start:stop].clone()
            shell_left = bisect_left(shell_indices, start)
            shell_right = bisect_left(shell_indices, stop, lo=shell_left)
            local_shell = shell_indices[shell_left:shell_right]
            if local_shell:
                shell_tile.index_fill_(
                    0,
                    torch.tensor(
                        tuple(route - start for route in local_shell),
                        dtype=torch.int64,
                        device=device,
                    ),
                    True,
                )
            anchor_left = bisect_left(rounding_anchor_shell_indices, start)
            anchor_right = bisect_left(
                rounding_anchor_shell_indices, stop, lo=anchor_left
            )
            local_anchor = rounding_anchor_shell_indices[anchor_left:anchor_right]
            if local_anchor:
                rounding_anchor_tile.index_fill_(
                    0,
                    torch.tensor(
                        tuple(route - start for route in local_anchor),
                        dtype=torch.int64,
                        device=device,
                    ),
                    True,
                )
            comparison_left = bisect_left(comparison_preferred_shell_indices, start)
            comparison_right = bisect_left(
                comparison_preferred_shell_indices, stop, lo=comparison_left
            )
            local_comparison = comparison_preferred_shell_indices[
                comparison_left:comparison_right
            ]
            if local_comparison:
                comparison_preferred_tile.index_fill_(
                    0,
                    torch.tensor(
                        tuple(route - start for route in local_comparison),
                        dtype=torch.int64,
                        device=device,
                    ),
                    True,
                )
        else:
            assert rounding_anchor is not None
            assert comparison_preferred is not None
            assert selectable is not None
            assert shell is not None
            rounding_anchor_tile = rounding_anchor[start:stop]
            comparison_preferred_tile = comparison_preferred[start:stop]
            selectable_tile = selectable[start:stop]
            shell_tile = shell[start:stop]
            physical = torch.arange(start, stop, dtype=torch.int64, device=device)
        d_term = d_weight * d_tile
        residual_term = residual_weight * residual_tile
        weighted_mass = d_term.abs() + residual_term.abs()
        c_tile = d_term + residual_term
        lambda_resource = resource_multiplier * resource_tile
        stationarity_mass = c_tile.abs() + lambda_resource.abs() + capacity_tile.abs()
        s_tile = c_tile - lambda_resource + capacity_tile
        weight_radius_tile = (
            d_weight_radius * d_tile.abs()
            + residual_weight_radius * residual_tile.abs()
        )
        combination_radius_tile = _gamma_n(1) * weighted_mass + 2.0 * _F64_MIN_NORMAL
        resource_radius_tile = (
            _gamma_n(1) * lambda_resource.abs()
            + resource_multiplier_conversion_radius * resource_tile.abs()
            + _F64_MIN_NORMAL
        )
        addition_radius_tile = _gamma_n(2) * stationarity_mass + 2.0 * _F64_MIN_NORMAL
        sign_radius_tile = (
            support_source_l1
            + weight_radius_tile
            + combination_radius_tile
            + resource_radius_tile
            + capacity_adjoint_linf_radius
            + addition_radius_tile
        )
        sign_radius_tile = torch.nextafter(
            (sign_radius_tile + 32.0 * _F64_MIN_NORMAL) / (1.0 - _gamma_n(32)),
            torch.full_like(sign_radius_tile, math.inf),
        )
        finite_ok = finite_ok & torch.isfinite(s_tile).all()
        finite_ok = finite_ok & torch.isfinite(sign_radius_tile).all()
        free_tile = selectable_tile & ~required_tile
        box_terms = torch.where(
            required_tile,
            s_tile,
            torch.where(free_tile, torch.minimum(s_tile, zero), zero),
        )
        preferred_c = torch.where(comparison_preferred_tile, c_tile, zero)
        base_terms = base_terms + box_terms.sum() - preferred_c.sum()
        base_abs_terms = (
            base_abs_terms + box_terms.abs().sum() + preferred_c.abs().sum()
        )
        preferred_dot_terms = preferred_dot_terms + preferred_c.sum()
        preferred_dot_abs_terms = preferred_dot_abs_terms + preferred_c.abs().sum()
        weight_conversion_l1 = weight_conversion_l1 + weight_radius_tile.sum()
        weighted_combination_l1 = (
            weighted_combination_l1 + combination_radius_tile.sum()
        )
        resource_dual_l1 = resource_dual_l1 + resource_radius_tile.sum()
        stationarity_addition_l1 = stationarity_addition_l1 + addition_radius_tile.sum()
        sign_linf = torch.maximum(sign_linf, sign_radius_tile.max())
        membership_counts = membership_counts + torch.stack(
            (
                rounding_anchor_tile.sum(dtype=torch.int64),
                comparison_preferred_tile.sum(dtype=torch.int64),
                required_tile.sum(dtype=torch.int64),
                selectable_tile.sum(dtype=torch.int64),
                shell_tile.sum(dtype=torch.int64),
            )
        )
        legality_ok = legality_ok & ~(required_tile & ~rounding_anchor_tile).any()
        legality_ok = legality_ok & ~(rounding_anchor_tile & ~selectable_tile).any()

        p_first = rounding_anchor_tile & ~required_tile
        q_first = selectable_tile & ~rounding_anchor_tile & ~shell_tile
        p_second = p_first & ~shell_tile
        q_second = selectable_tile & ~rounding_anchor_tile & shell_tile
        s_upper = torch.nextafter(
            s_tile + sign_radius_tile,
            torch.full_like(s_tile, math.inf),
        )
        s_lower = torch.nextafter(
            s_tile - sign_radius_tile,
            torch.full_like(s_tile, -math.inf),
        )
        p_certified = s_upper < 0.0
        q_certified = s_lower > 0.0
        p_correction = torch.nextafter(
            -s_tile - sign_radius_tile,
            torch.full_like(s_tile, -math.inf),
        )
        q_correction = s_lower
        for ordinal, (mask, certified, correction) in enumerate(
            (
                (p_first, p_certified, p_correction),
                (q_first, q_certified, q_correction),
                (p_second, p_certified, p_correction),
                (q_second, q_certified, q_correction),
            )
        ):
            endpoint_states[ordinal] = _merge_endpoint_tile(
                endpoint_states[ordinal],
                _first_and_minimum_on_mask(
                    mask=mask,
                    certified=certified,
                    correction_values=correction,
                    physical_indices=physical,
                ),
            )

    if any(value is None for value in endpoint_states):
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.NUMERICAL_REFUSAL,
            "outside-pair endpoint reductions did not execute",
        )
    compact_device_scalars: list[Any] = []
    for state in endpoint_states:
        assert state is not None
        compact_device_scalars.extend(
            value.to(dtype=torch.float64).reshape(()) for value in state
        )
    compact_device_scalars.extend(
        value.reshape(())
        for value in (
            base_terms,
            base_abs_terms,
            weight_conversion_l1,
            weighted_combination_l1,
            resource_dual_l1,
            stationarity_addition_l1,
            sign_linf,
            preferred_dot_terms,
            preferred_dot_abs_terms,
        )
    )
    compact_device_scalars.extend(
        value.to(dtype=torch.float64).reshape(()) for value in membership_counts
    )
    compact_device_scalars.extend(
        value.to(dtype=torch.float64).reshape(()) for value in (finite_ok, legality_ok)
    )
    compact = tuple(
        float(value)
        for value in torch.stack(compact_device_scalars)
        .to(device="cpu", dtype=torch.float64)
        .tolist()
    )
    endpoints = tuple(
        _decode_endpoint_state(compact[start : start + 9]) for start in range(0, 36, 9)
    )
    scalar_values = compact[36:45]
    (
        base_box_nominal,
        base_box_abs,
        weight_conversion_value,
        combination_value,
        resource_value,
        addition_value,
        sign_linf_value,
        preferred_dot_value,
        preferred_dot_abs,
    ) = scalar_values
    (
        rounding_anchor_count,
        comparison_preferred_count,
        required_count,
        selectable_count,
        shell_count,
    ) = (int(value) for value in compact[45:50])
    if not bool(compact[50]):
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.NUMERICAL_REFUSAL,
            "outside-pair stationarity or radius became nonfinite",
        )
    if not bool(compact[51]):
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.PAIR_COVER_MISMATCH,
            "required must lie in the preferred anchor and the anchor must be selectable",
        )
    constant_nominal = (
        resource_multiplier * resource_floor - capacity_multiplier_dot_limit
    )
    base_nominal = constant_nominal + base_box_nominal
    reduction_count = 2 * count + 2 * tile_count + 2
    reduction_radius = _nonnegative_arithmetic_upper(
        _gamma_n(reduction_count) * (abs(constant_nominal) + base_box_abs)
        + float(reduction_count) * _F64_MIN_NORMAL,
        operation_count=6,
    )
    radius_operation_count = 16 * count + 8 * tile_count + 1
    weight_conversion_upper = _nonnegative_arithmetic_upper(
        weight_conversion_value,
        operation_count=radius_operation_count,
    )
    combination_upper = _nonnegative_arithmetic_upper(
        combination_value,
        operation_count=radius_operation_count,
    )
    resource_upper = _nonnegative_arithmetic_upper(
        resource_value,
        operation_count=radius_operation_count,
    )
    addition_upper = _nonnegative_arithmetic_upper(
        addition_value,
        operation_count=radius_operation_count,
    )
    sign_linf_upper = _nonnegative_arithmetic_upper(
        sign_linf_value,
        operation_count=32,
    )
    base_radius = _nonnegative_arithmetic_upper(
        support_source_l1
        + weight_conversion_upper
        + combination_upper
        + resource_upper
        + capacity_adjoint_l1_radius
        + addition_upper
        + reduction_radius
        + dual_constant_radius,
        operation_count=12,
    )
    if not math.isfinite(base_nominal) or not math.isfinite(base_radius):
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.NUMERICAL_REFUSAL,
            "outside-pair base dual enclosure became nonfinite",
        )
    base_lower = _down(base_nominal - base_radius)
    first = _rectangle_from_endpoints(
        rectangle=V1SignedWorkVariableShellOutsidePairRectangle.ADD_OUTSIDE,
        p=endpoints[0],
        q=endpoints[1],
        ordinal_offset=0,
    )
    second = _rectangle_from_endpoints(
        rectangle=(
            V1SignedWorkVariableShellOutsidePairRectangle.DROP_OUTSIDE_ADD_INSIDE
        ),
        p=endpoints[2],
        q=endpoints[3],
        # The compact cover is diagnostic identity only.  The live first
        # rectangle determines the exact offset, so equal legacy counts can
        # never substitute a different pair axis.
        ordinal_offset=first.pair_count,
    )
    nonempty = tuple(value for value in (first, second) if value.pair_count)
    if not nonempty:
        worst_rectangle = None
        worst_center = math.inf
        worst_radius = 0.0
        worst_lower = math.inf
    else:
        # Lower-bound ties are resolved by the exact pair-cover ordinal, which
        # is physical-index ordered within the fixed rectangle order.
        worst_rectangle = min(
            nonempty,
            key=lambda value: (
                base_lower + value.correction_lower,
                value.pair_ordinal,
            ),
        )
        worst_center = base_nominal + worst_rectangle.correction_lower
        worst_lower = _down(base_lower + worst_rectangle.correction_lower)
        worst_radius = _up(max(0.0, worst_center - worst_lower))
    preferred_reduction_radius = _nonnegative_arithmetic_upper(
        _gamma_n(count + tile_count) * preferred_dot_abs
        + float(count + tile_count) * _F64_MIN_NORMAL,
        operation_count=6,
    )
    # Endpoint state (36), common reductions (9), five membership counts, and
    # two finite/legality predicates.  This is O(1) transfer independent of N.
    scalar_d2h = len(compact) * _F64_BYTES
    return _NumericScan(
        base_nominal=base_nominal,
        base_radius=base_radius,
        base_lower=base_lower,
        support_source_l1_radius=support_source_l1,
        weight_conversion_l1_radius=weight_conversion_upper,
        weighted_combination_l1_radius=combination_upper,
        resource_dual_l1_radius=resource_upper,
        capacity_dual_l1_radius=capacity_adjoint_l1_radius,
        stationarity_addition_l1_radius=addition_upper,
        box_reduction_radius=reduction_radius,
        constant_radius=dual_constant_radius,
        stationarity_sign_linf_radius=sign_linf_upper,
        preferred_dot_nominal=preferred_dot_value,
        preferred_dot_reduction_radius=preferred_reduction_radius,
        rounding_anchor_count=rounding_anchor_count,
        comparison_preferred_count=comparison_preferred_count,
        required_count=required_count,
        selectable_count=selectable_count,
        shell_count=shell_count,
        first_rectangle=first,
        second_rectangle=second,
        worst_rectangle=worst_rectangle,
        worst_center=worst_center,
        worst_radius=worst_radius,
        worst_lower=worst_lower,
        route_tile_scan_count=tile_count,
        scalar_d2h_bytes=scalar_d2h,
        d2h_synchronization_count=1,
    )


def _validate_execution_join(
    *,
    registration: Any,
    alias_binding: V1BlockSupportAliasBinding,
    d_receipt: Any,
    pair_cover_receipt: V1SignedWorkVariableShellOutsidePairCoverReceipt,
    membership_source: V1BlockSupportOutsidePairMembershipSourceHandle,
    dual_source: V1BlockSupportOutsidePairDualSourceHandle,
    design: V1BlockSupportOutsidePairCUDAExecutionDesign,
    hbm_forecast: V1BlockSupportOutsidePairHBMForecast,
) -> None:
    if (
        type(membership_source) is not V1BlockSupportOutsidePairMembershipSourceHandle
        or type(dual_source) is not V1BlockSupportOutsidePairDualSourceHandle
        or type(design) is not V1BlockSupportOutsidePairCUDAExecutionDesign
        or type(hbm_forecast) is not V1BlockSupportOutsidePairHBMForecast
    ):
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.INVALID_TYPE,
            "outside-pair execution requires exact live providers and receipts",
        )
    membership_source.assert_live()
    dual_source.assert_live()
    membership = membership_source.receipt
    dual = dual_source.receipt
    if (
        membership.pair_cover_receipt_hash != pair_cover_receipt.receipt_hash
        or membership.pair_axis_commitment_hash
        != pair_cover_receipt.pair_axis_commitment_hash
        or membership.route_count != pair_cover_receipt.route_count
        or membership.complete_pair_count != pair_cover_receipt.complete_pair_count
        or membership.comparison_preferred_removed_route_index_hash
        != d_receipt.preferred_removed_route_index_hash
        or membership.comparison_preferred_count
        != registration.preferred_action_cardinality
        or dual.outside_pair_cover_hash != pair_cover_receipt.receipt_hash
        or dual.pair_axis_commitment_hash
        != pair_cover_receipt.pair_axis_commitment_hash
        or dual.alias_hash != alias_binding.alias_hash
        or dual.route_count != d_receipt.route_count
        or dual.model_identity_hash != d_receipt.model_identity_hash
        or dual.chart_identity_hash != d_receipt.chart_identity_hash
        or dual.current_mask_hash != d_receipt.current_mask_hash
        or dual.proposal_generation != d_receipt.proposal_generation
        or dual.route_axis_hash != d_receipt.route_axis_hash
        or dual.route_id_axis_hash != d_receipt.route_id_axis_hash
        or dual.registered_d_prefix_hash != d_receipt.registered_d_prefix_hash
        or dual.registered_d_w2_prefix_hash != d_receipt.registered_d_w2_prefix_hash
        or dual.preferred_action_hash != registration.preferred_action_hash
        or dual.design_hash != design.design_hash
        or dual.hbm_forecast_hash != hbm_forecast.forecast_hash
        or hbm_forecast.design_hash != design.design_hash
        or not hbm_forecast.admitted
    ):
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.STALE_IDENTITY,
            "outside-pair membership, dual, alias, source, design, or HBM identities differ",
        )


def _exact_reference_worst_pair(
    *,
    pair_cover: V1SignedWorkVariableShellOutsidePairCoverEnumeration,
    d_coefficients: Any,
    residual_coefficients: Any,
    resource_coefficients: Any,
    capacity_adjoint: Any,
    d_weight_numerator: int,
    residual_weight_numerator: int,
    population_weight_denominator: int,
    resource_multiplier: float,
    resource_floor: float,
    capacity_multiplier_dot_limit: float,
    preferred_membership: Any,
) -> tuple[Fraction | None, tuple[int, int, int] | None]:
    d_weight = Fraction(d_weight_numerator, population_weight_denominator)
    residual_weight = Fraction(residual_weight_numerator, population_weight_denominator)
    stationarity: list[float] = []
    preferred_dot = Fraction(0, 1)
    for index, (d_value, residual_value, resource_value, capacity_value) in enumerate(
        zip(
            d_coefficients.tolist(),
            residual_coefficients.tolist(),
            resource_coefficients.tolist(),
            capacity_adjoint.tolist(),
            strict=True,
        )
    ):
        comparison = d_weight * Fraction.from_float(float(d_value))
        comparison += residual_weight * Fraction.from_float(float(residual_value))
        if bool(preferred_membership[index]):
            preferred_dot += comparison
        exact = comparison
        exact -= Fraction.from_float(resource_multiplier) * Fraction.from_float(
            float(resource_value)
        )
        exact += Fraction.from_float(float(capacity_value))
        stationarity.append(float(exact))
    constant_exact = (
        Fraction.from_float(resource_multiplier) * Fraction.from_float(resource_floor)
        - Fraction.from_float(capacity_multiplier_dot_limit)
        - preferred_dot
    )
    constant = float(constant_exact)
    bounds = evaluate_v1_signed_work_variable_shell_outside_pair_dual_bounds_reference(
        pair_cover=pair_cover,
        stationarity_coefficients=tuple(stationarity),
        linear_constant=constant,
    )
    if not bounds:
        return None, None
    best = min(
        bounds,
        key=lambda value: (
            Fraction(
                value.exact_dual_lower_numerator,
                value.exact_dual_lower_denominator,
            ),
            value.pair.pair_ordinal,
        ),
    )
    exact = Fraction(
        best.exact_dual_lower_numerator,
        best.exact_dual_lower_denominator,
    )
    return exact, (
        best.pair.preferred_selected_index,
        best.pair.preferred_unselected_index,
        best.pair.pair_ordinal,
    )


def _issue_v1_block_support_outside_pair_family_from_membership(
    *,
    block_witness_source: V1BlockSupportWitnessSource,
    alias_binding: V1BlockSupportAliasBinding,
    d_fixed_support: V1SignedWorkBlockSupportSourceDeviceHandle,
    residual_support: V1SignedWorkBlockSupportSourceDeviceHandle,
    preferred_action_binding: V1SignedWorkBlockSupportPreferredActionBinding,
    pair_cover_receipt: V1SignedWorkVariableShellOutsidePairCoverReceipt,
    cover_binding_receipt: V1BlockSupportOutsidePairCoverBindingReceipt | None = None,
    membership_source: V1BlockSupportOutsidePairMembershipSourceHandle,
    dual_source: V1BlockSupportOutsidePairDualSourceHandle,
    block_index: int,
    design: V1BlockSupportOutsidePairCUDAExecutionDesign,
    hbm_forecast: V1BlockSupportOutsidePairHBMForecast,
    exact_reference_pair_cover: (
        V1SignedWorkVariableShellOutsidePairCoverEnumeration | None
    ) = None,
) -> V1BlockSupportOutsidePairFamilyReceipt:
    registration, d_receipt, r_receipt = _validate_block_alias_sources(
        block_witness_source=block_witness_source,
        alias_binding=alias_binding,
        d_fixed_support=d_fixed_support,
        residual_support=residual_support,
        preferred_action_binding=preferred_action_binding,
        pair_cover_receipt=pair_cover_receipt,
        cover_binding_receipt=cover_binding_receipt,
        block_index=block_index,
    )
    _validate_execution_join(
        registration=registration,
        alias_binding=alias_binding,
        d_receipt=d_receipt,
        pair_cover_receipt=pair_cover_receipt,
        membership_source=membership_source,
        dual_source=dual_source,
        design=design,
        hbm_forecast=hbm_forecast,
    )
    reference_mode = (
        design.execution_mode
        is V1BlockSupportOutsidePairExecutionMode.CPU_EXACT_REFERENCE_TEST_ONLY
    )
    if reference_mode:
        if exact_reference_pair_cover is None:
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.INVALID_TYPE,
                "CPU reference requires the exact live cover enumeration",
            )
        if d_receipt.route_count > design.cpu_reference_maximum_routes:
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.CPU_REFERENCE_SCALE_EXCEEDED,
                "CPU exact-reference route scale exceeds its registered maximum",
            )
    elif exact_reference_pair_cover is not None:
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.CPU_PRODUCTION_FALLBACK,
            "production outside-pair execution may not consume host cover enumeration",
        )
    with (
        d_fixed_support.open_coefficient_lease() as d_lease,
        residual_support.open_coefficient_lease() as residual_lease,
        dual_source.open_dual_lease() as dual_lease,
        membership_source.open_membership_lease() as membership_lease,
    ):
        d_coefficients = d_lease.tensor
        residual_coefficients = residual_lease.tensor
        resource_coefficients, capacity_adjoint = dual_lease.live_route_tensors()
        (
            rounding_anchor,
            comparison_preferred,
            required,
            selectable,
            shell,
        ) = membership_lease.live_membership()
        lam, rho, _mu_axis, _limit_axis = dual_lease.compact_duals
        device = d_coefficients.device
        tensors = (
            residual_coefficients,
            resource_coefficients,
            capacity_adjoint,
            rounding_anchor,
            comparison_preferred,
            required,
            selectable,
            shell,
        )
        if any(value.device != device for value in tensors):
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.SOURCE_MISMATCH,
                "outside-pair live sources are not colocated on one device",
            )
        if reference_mode and device.type != "cpu":
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.CPU_PRODUCTION_FALLBACK,
                "CPU reference sources must reside on CPU",
            )
        if not reference_mode and device.type != "cuda":
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.CUDA_REQUIRED,
                "production outside-pair execution requires CUDA",
            )
        scan = _scan_live_outside_pairs(
            design=design,
            d_coefficients=d_coefficients,
            residual_coefficients=residual_coefficients,
            resource_coefficients=resource_coefficients,
            capacity_adjoint=capacity_adjoint,
            rounding_anchor=rounding_anchor,
            comparison_preferred=comparison_preferred,
            required=required,
            selectable=selectable,
            shell=shell,
            d_weight_numerator=registration.d_weight_numerator,
            residual_weight_numerator=registration.residual_weight_numerator,
            population_weight_denominator=registration.population_weight_denominator,
            d_source_l1_radius=d_receipt.coefficient_construction_l1_radius_upper,
            residual_source_l1_radius=(
                r_receipt.coefficient_construction_l1_radius_upper
            ),
            resource_multiplier=lam,
            resource_floor=rho,
            capacity_multiplier_dot_limit=(
                dual_source.receipt.capacity_multiplier_dot_limit_nominal
            ),
            resource_multiplier_conversion_radius=(
                dual_source.receipt.resource_multiplier_conversion_radius
            ),
            capacity_adjoint_l1_radius=(
                dual_source.receipt.capacity_adjoint_roundoff_l1_radius
            ),
            capacity_adjoint_linf_radius=(
                dual_source.receipt.capacity_adjoint_roundoff_linf_radius
            ),
            dual_constant_radius=dual_source.receipt.dual_constant_roundoff_radius,
        )
        exact_worst: Fraction | None = None
        if exact_reference_pair_cover is not None:
            exact_worst, _exact_pair = _exact_reference_worst_pair(
                pair_cover=exact_reference_pair_cover,
                d_coefficients=d_coefficients,
                residual_coefficients=residual_coefficients,
                resource_coefficients=resource_coefficients,
                capacity_adjoint=capacity_adjoint,
                d_weight_numerator=registration.d_weight_numerator,
                residual_weight_numerator=registration.residual_weight_numerator,
                population_weight_denominator=(
                    registration.population_weight_denominator
                ),
                resource_multiplier=lam,
                resource_floor=rho,
                capacity_multiplier_dot_limit=(
                    dual_source.receipt.capacity_multiplier_dot_limit_nominal
                ),
                preferred_membership=comparison_preferred,
            )
            if (
                exact_worst is not None
                and Fraction.from_float(scan.worst_lower) > exact_worst
            ):
                _refuse(
                    V1BlockSupportOutsidePairRefusalCode.NUMERICAL_REFUSAL,
                    "tiled outside-pair enclosure exceeds the exact reference dual",
                )
    first = scan.first_rectangle
    second = scan.second_rectangle
    first_pair_lower = (
        math.inf
        if not first.pair_count
        else _down(scan.base_lower + first.correction_lower)
    )
    second_pair_lower = (
        math.inf
        if not second.pair_count
        else _down(scan.base_lower + second.correction_lower)
    )
    worst = scan.worst_rectangle
    block_contexts = block_witness_source.block_context_ids(block_index)
    dual_receipt = dual_source.receipt
    preferred_dot_radius = _up(
        scan.support_source_l1_radius
        + scan.weight_conversion_l1_radius
        + scan.weighted_combination_l1_radius
        + scan.preferred_dot_reduction_radius
    )
    if cover_binding_receipt is None:
        anchor_action_coordinate_hash = preferred_action_binding.preferred_action_hash
        anchor_removed_route_index_hash = (
            preferred_action_binding.preferred_removed_route_index_hash
        )
        anchor_subset_identity_hash = preferred_action_binding.binding_hash
        anchor_shell_gate_hash = pair_cover_receipt.preferred_action_index_hash
        anchor_shell_prefix_count = (
            pair_cover_receipt.fixed_cardinality
            - pair_cover_receipt.preferred_outside_shell_count
        )
        anchor_fixed_count = pair_cover_receipt.preferred_outside_shell_count
        cover_binding_hash = None
    else:
        anchor_action_coordinate_hash = (
            cover_binding_receipt.rounding_anchor_action_coordinate_hash
        )
        anchor_removed_route_index_hash = (
            cover_binding_receipt.rounding_anchor_removed_route_index_hash
        )
        anchor_subset_identity_hash = (
            cover_binding_receipt.rounding_anchor_subset_identity_hash
        )
        anchor_shell_gate_hash = cover_binding_receipt.rounding_anchor_shell_gate_hash
        anchor_shell_prefix_count = cover_binding_receipt.alias_shell_prefix_count
        anchor_fixed_count = cover_binding_receipt.alias_fixed_outside_removed_count
        cover_binding_hash = cover_binding_receipt.receipt_hash
    same_cardinality = (
        alias_binding.cardinality == registration.preferred_action_cardinality
    )
    return V1BlockSupportOutsidePairFamilyReceipt(
        block_support_registration_hash=registration.registration_hash,
        block_index=block_index,
        block_id=block_witness_source.block_ids[block_index],
        block_context_axis_hash=_canonical_context_axis_hash(block_contexts),
        d_fixed_support_receipt_hash=d_receipt.receipt_hash,
        residual_support_receipt_hash=r_receipt.receipt_hash,
        support_source_registration_hash=d_receipt.source_registration_hash,
        native_fiber_axis_receipt_hash=d_receipt.native_fiber_axis_receipt_hash,
        native_fiber_measure_hash=d_receipt.native_fiber_measure_hash,
        preferred_action_hash=preferred_action_binding.preferred_action_hash,
        preferred_action_binding_hash=preferred_action_binding.binding_hash,
        preferred_resulting_mask_hash=(
            preferred_action_binding.preferred_resulting_mask_hash
        ),
        preferred_removed_route_index_hash=(
            d_receipt.preferred_removed_route_index_hash
        ),
        model_identity_hash=d_receipt.model_identity_hash,
        chart_identity_hash=d_receipt.chart_identity_hash,
        current_mask_hash=d_receipt.current_mask_hash,
        proposal_generation=d_receipt.proposal_generation,
        route_axis_hash=d_receipt.route_axis_hash,
        route_id_axis_hash=d_receipt.route_id_axis_hash,
        registered_d_prefix_hash=d_receipt.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=d_receipt.registered_d_w2_prefix_hash,
        alias_hash=alias_binding.alias_hash,
        branch_id=alias_binding.branch_id,
        fd_rank=alias_binding.fd_rank,
        resource_alias_id=alias_binding.resource_alias_id,
        source_receipt_hash=alias_binding.source_receipt_hash,
        source_replay_hash=alias_binding.source_replay_hash,
        cardinality_alias_id=alias_binding.cardinality_alias_id,
        rounded_subset_cardinality=alias_binding.cardinality,
        alias_fixed_outside_removed_count=anchor_fixed_count,
        alias_shell_prefix_count=anchor_shell_prefix_count,
        rounding_anchor_selected_count=scan.rounding_anchor_count,
        comparison_preferred_selected_count=scan.comparison_preferred_count,
        rounding_anchor_action_coordinate_hash=anchor_action_coordinate_hash,
        rounding_anchor_removed_route_index_hash=anchor_removed_route_index_hash,
        rounding_anchor_subset_identity_hash=anchor_subset_identity_hash,
        rounding_anchor_shell_gate_hash=anchor_shell_gate_hash,
        constraint_cover_hash=alias_binding.constraint_cover_hash,
        pair_cover_receipt_hash=pair_cover_receipt.receipt_hash,
        pair_axis_commitment_hash=pair_cover_receipt.pair_axis_commitment_hash,
        typed_cover_binding_receipt_hash=cover_binding_hash,
        membership_source_receipt_hash=membership_source.receipt.receipt_hash,
        membership_source_kind="cpu_reference_four_mask",
        structural_domain_receipt_hash=None,
        physical_block_support_binding_receipt_hash=None,
        fixed_membership_binding_hash=None,
        shell_freeze_receipt_hash=None,
        kkt_state_identity_hash=None,
        kkt_problem_hash=None,
        kkt_policy_hash=None,
        kkt_branch_receipt_hash=None,
        dual_source_receipt_hash=dual_receipt.receipt_hash,
        resource_axis_hash=dual_receipt.resource_axis_hash,
        resource_coefficient_source_hash=(
            dual_receipt.resource_coefficient_source_hash
        ),
        capacity_operator_source_hash=dual_receipt.capacity_operator_source_hash,
        resource_multiplier_source_hash=(dual_receipt.resource_multiplier_source_hash),
        capacity_multiplier_source_hash=(dual_receipt.capacity_multiplier_source_hash),
        route_count=d_receipt.route_count,
        capacity_count=dual_receipt.capacity_count,
        route_tile_size=design.route_tile_size,
        route_tile_scan_count=scan.route_tile_scan_count,
        complete_route_coefficient_scan_count=d_receipt.route_count,
        add_outside_rectangle_pair_count=first.pair_count,
        drop_outside_add_inside_rectangle_pair_count=second.pair_count,
        complete_pair_cover_count=first.pair_count + second.pair_count,
        add_outside_rectangle_empty=not bool(first.pair_count),
        drop_outside_add_inside_rectangle_empty=not bool(second.pair_count),
        vacuous_no_outside_pair=worst is None,
        base_dual_nominal=scan.base_nominal,
        base_dual_downward_radius=scan.base_radius,
        base_dual_lower_outward=scan.base_lower,
        comparison_preferred_dot_nominal=scan.preferred_dot_nominal,
        comparison_preferred_dot_absolute_radius=preferred_dot_radius,
        comparison_preferred_dot_upper_outward=_up(
            scan.preferred_dot_nominal + preferred_dot_radius
        ),
        support_coefficient_source_l1_radius=scan.support_source_l1_radius,
        weight_conversion_l1_radius=scan.weight_conversion_l1_radius,
        weighted_combination_roundoff_l1_radius=(scan.weighted_combination_l1_radius),
        resource_dual_roundoff_l1_radius=scan.resource_dual_l1_radius,
        capacity_dual_roundoff_l1_radius=scan.capacity_dual_l1_radius,
        stationarity_addition_roundoff_l1_radius=(scan.stationarity_addition_l1_radius),
        box_reduction_roundoff_radius=scan.box_reduction_radius,
        dual_constant_roundoff_radius=scan.constant_radius,
        stationarity_sign_linf_radius=scan.stationarity_sign_linf_radius,
        ambiguous_p_endpoint_count=(first.ambiguous_p_count + second.ambiguous_p_count),
        ambiguous_q_endpoint_count=(first.ambiguous_q_count + second.ambiguous_q_count),
        add_outside_correction_lower=(
            0.0 if not first.pair_count else first.correction_lower
        ),
        drop_outside_add_inside_correction_lower=(
            0.0 if not second.pair_count else second.correction_lower
        ),
        add_outside_pair_lower_outward=first_pair_lower,
        drop_outside_add_inside_pair_lower_outward=second_pair_lower,
        worst_pair_rectangle=None if worst is None else worst.rectangle,
        worst_pair_preferred_selected_index=(None if worst is None else worst.p_index),
        worst_pair_preferred_unselected_index=(
            None if worst is None else worst.q_index
        ),
        worst_pair_ordinal=None if worst is None else worst.pair_ordinal,
        worst_pair_lower_center=scan.worst_center,
        worst_pair_total_downward_radius=scan.worst_radius,
        worst_pair_lower_outward=scan.worst_lower,
        exact_reference_worst_pair_numerator=(
            None if exact_worst is None else exact_worst.numerator
        ),
        exact_reference_worst_pair_denominator=(
            None if exact_worst is None else exact_worst.denominator
        ),
        design_hash=design.design_hash,
        hbm_forecast_hash=hbm_forecast.forecast_hash,
        execution_mode=design.execution_mode,
        production_typed_membership_sources_joined=False,
        cpu_reference_four_mask_only=True,
        scalar_device_to_host_transfer_bytes=scan.scalar_d2h_bytes,
        scan_device_to_host_synchronization_count=(scan.d2h_synchronization_count),
        comparison_preferred_is_rounding_anchor=(
            anchor_removed_route_index_hash
            == preferred_action_binding.preferred_removed_route_index_hash
        ),
        same_cardinality_alias_only=same_cardinality,
        cross_cardinality_alias_authority=(
            not same_cardinality and cover_binding_receipt is not None
        ),
        typed_cover_equivalence_joined=cover_binding_receipt is not None,
        _issuer_token=_FAMILY_TOKEN,
    )


def evaluate_v1_block_support_outside_pair_lower_bound_reference(
    *,
    block_witness_source: V1BlockSupportWitnessSource,
    alias_binding: V1BlockSupportAliasBinding,
    d_fixed_support: V1SignedWorkBlockSupportSourceDeviceHandle,
    residual_support: V1SignedWorkBlockSupportSourceDeviceHandle,
    preferred_action_binding: V1SignedWorkBlockSupportPreferredActionBinding,
    pair_cover: V1SignedWorkVariableShellOutsidePairCoverEnumeration,
    cover_binding_receipt: V1BlockSupportOutsidePairCoverBindingReceipt | None = None,
    registry: V1SignedWorkVariableShellActionCellRegistry | None = None,
    frozen_shell_gates: Sequence[Real] = (),
    dual_source: V1BlockSupportOutsidePairDualSourceHandle,
    block_index: int,
    design: V1BlockSupportOutsidePairCUDAExecutionDesign,
    hbm_forecast: V1BlockSupportOutsidePairHBMForecast,
) -> V1BlockSupportOutsidePairFamilyReceipt:
    """Run the bounded exact-rational CPU reference; never production."""

    if type(pair_cover) is not V1SignedWorkVariableShellOutsidePairCoverEnumeration:
        raise TypeError("outside-pair reference requires the exact cover enumeration")
    if (
        type(design) is not V1BlockSupportOutsidePairCUDAExecutionDesign
        or design.execution_mode
        is not V1BlockSupportOutsidePairExecutionMode.CPU_EXACT_REFERENCE_TEST_ONLY
    ):
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.CPU_PRODUCTION_FALLBACK,
            "outside-pair reference requires exact CPU-reference mode",
        )
    count = pair_cover.receipt.route_count
    anchor_set = set(pair_cover.preferred_action_indices)
    required_set = set(pair_cover.required_action_indices)
    shell_set = set(pair_cover.shell_indices)
    rounding_anchor = torch.tensor(
        tuple(index in anchor_set for index in range(count)), dtype=torch.bool
    )
    comparison_preferred, _packed_host_bytes = preferred_action_binding.materialize(
        device=torch.device("cpu"),
        tile_bytes=max(1, (count + 7) // 8),
    )
    comparison_preferred = _require_bool_tensor(
        comparison_preferred,
        name="source-bound comparison preferred membership",
        shape=(count,),
    )
    if int(comparison_preferred.sum(dtype=torch.int64).item()) != (
        preferred_action_binding.preferred_selected_count
    ):
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.MUTATED_CUSTODY,
            "materialized comparison preferred action changed cardinality",
        )
    if cover_binding_receipt is None:
        if not bool(torch.equal(rounding_anchor, comparison_preferred)):
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.PAIR_COVER_MISMATCH,
                "legacy same-cardinality cover must exactly replay global P membership",
            )
    else:
        if registry is None or not frozen_shell_gates:
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.INVALID_TYPE,
                "typed reference cover requires its live registry and frozen shell gates",
            )
        replay = bind_v1_block_support_outside_pair_cover_reference(
            registry=registry,
            alias_binding=alias_binding,
            pair_cover=pair_cover,
            frozen_shell_gates=frozen_shell_gates,
        )
        if replay.receipt_hash != cover_binding_receipt.receipt_hash:
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.STALE_IDENTITY,
                "typed reference cover no longer replays its live branch source",
            )
    required = torch.tensor(
        tuple(index in required_set for index in range(count)), dtype=torch.bool
    )
    selectable = torch.tensor(pair_cover.selectable_mask, dtype=torch.bool)
    shell = torch.tensor(
        tuple(index in shell_set for index in range(count)), dtype=torch.bool
    )
    membership = bind_v1_block_support_outside_pair_membership_source(
        pair_cover_receipt=pair_cover.receipt,
        owner_source_receipt_hash=(
            pair_cover.receipt.receipt_hash
            if cover_binding_receipt is None
            else cover_binding_receipt.receipt_hash
        ),
        rounding_anchor_membership=rounding_anchor,
        comparison_preferred_membership=comparison_preferred,
        comparison_preferred_removed_route_index_hash=(
            preferred_action_binding.preferred_removed_route_index_hash
        ),
        required_membership=required,
        selectable_membership=selectable,
        shell_membership=shell,
    )
    try:
        return _issue_v1_block_support_outside_pair_family_from_membership(
            block_witness_source=block_witness_source,
            alias_binding=alias_binding,
            d_fixed_support=d_fixed_support,
            residual_support=residual_support,
            preferred_action_binding=preferred_action_binding,
            pair_cover_receipt=pair_cover.receipt,
            cover_binding_receipt=cover_binding_receipt,
            membership_source=membership,
            dual_source=dual_source,
            block_index=block_index,
            design=design,
            hbm_forecast=hbm_forecast,
            exact_reference_pair_cover=pair_cover,
        )
    finally:
        membership.evict()


def _preferred_shell_routes_from_structural_provider(
    binding: V1SignedWorkPhysicalAlternateBlockSupportBinding,
) -> tuple[int, ...]:
    provider = binding.structural_provider
    receipt = provider.receipt
    excluded = set(receipt.excluded_preferred_action_cell_hashes)
    matched = tuple(
        cell
        for cell in provider.registry.iter_action_cells()
        if cell.action_cell_hash in excluded
    )
    offset_axes = {cell.selected_shell_offsets for cell in matched}
    if len(matched) != len(excluded) or len(offset_axes) != 1 or not offset_axes:
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.SOURCE_MISMATCH,
            "structural preferred cells do not determine one shell subset",
        )
    offsets = next(iter(offset_axes))
    routes = tuple(receipt.shell_indices[offset] for offset in offsets)
    if (
        receipt.fixed_outside_removed_count + len(routes)
        != binding.receipt.preferred_selected_count
    ):
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.SOURCE_MISMATCH,
            "fixed plus shell preferred membership has the wrong cardinality",
        )
    return routes


def _rounding_anchor_from_structural_provider(
    *,
    binding: V1SignedWorkPhysicalAlternateBlockSupportBinding,
    alias_binding: V1BlockSupportAliasBinding,
) -> tuple[tuple[int, ...], Any, Any, V1SignedWorkDeviceShellExecutionBundle]:
    """Replay ``P_alpha`` from one frozen branch ladder and subset custody."""

    provider = binding.structural_provider
    receipt = provider.receipt
    try:
        q_branch = V1SignedWorkVariableShellQBranch(alias_binding.branch_id)
        kkt_branch = V1RelaxedKKTBranch(alias_binding.branch_id)
    except ValueError:
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.ALIAS_MISMATCH,
            "rounding-anchor alias branch is not registered",
        )
    source = provider.registry.branch_source(q_branch)
    resource_alias = source.resource_alias(alias_binding.resource_alias_id)
    bundles = tuple(
        value
        for value in provider.branch_bundles
        if value.execution_receipt.selected_branch is kkt_branch
    )
    if (
        len(bundles) != 1
        or alias_binding.cardinality != resource_alias.total_removed_count
        or resource_alias.fixed_outside_removed_count
        != receipt.fixed_outside_removed_count
    ):
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.ALIAS_MISMATCH,
            "rounding anchor does not match the registered branch resource alias",
        )
    bundle = bundles[0]
    frozen = bundle.frozen_ladder
    endpoints = tuple(
        value
        for value in frozen.actions
        if value.cumulative_removed_count == resource_alias.total_removed_count
    )
    if len(endpoints) != 1:
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.SOURCE_MISMATCH,
            "frozen branch ladder has no unique alias-cardinality endpoint",
        )
    endpoint = endpoints[0]
    subsets = tuple(
        value
        for value in source.subset_identities
        if value.removed_count == endpoint.cumulative_removed_count
        and value.removed_route_index_hash == endpoint.removed_route_index_hash
        and len(value.shell_offsets) == resource_alias.shell_prefix_count
    )
    if len(subsets) != 1:
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.SOURCE_MISMATCH,
            "frozen branch endpoint does not replay one exact live subset identity",
        )
    subset = subsets[0]
    routes = tuple(receipt.shell_indices[offset] for offset in subset.shell_offsets)
    if (
        tuple(sorted(routes)) != routes
        or receipt.fixed_outside_removed_count + len(routes)
        != endpoint.cumulative_removed_count
    ):
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.SOURCE_MISMATCH,
            "frozen rounding anchor lost deterministic physical-index order",
        )
    return routes, resource_alias, subset, bundle


def _validate_live_scan_against_compact_cover(
    *,
    scan: _NumericScan,
    pair_cover_receipt: V1SignedWorkVariableShellOutsidePairCoverReceipt,
) -> None:
    first = scan.first_rectangle
    second = scan.second_rectangle
    if (
        scan.rounding_anchor_count != pair_cover_receipt.fixed_cardinality
        or scan.shell_count <= 0
        or first.p_count != pair_cover_receipt.droppable_preferred_count
        or first.q_count != pair_cover_receipt.addable_outside_shell_count
        or second.p_count != pair_cover_receipt.droppable_preferred_outside_shell_count
        or second.q_count != pair_cover_receipt.addable_inside_shell_count
        or first.pair_count != pair_cover_receipt.add_outside_rectangle_pair_count
        or second.pair_count
        != pair_cover_receipt.drop_outside_add_inside_rectangle_pair_count
        or first.pair_count + second.pair_count
        != pair_cover_receipt.complete_pair_count
    ):
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.PAIR_COVER_MISMATCH,
            "live source membership does not replay the compact two-rectangle cover",
        )


def issue_v1_block_support_outside_pair_lower_bound_cuda(
    *,
    block_witness_source: V1BlockSupportWitnessSource,
    alias_binding: V1BlockSupportAliasBinding,
    physical_block_support_binding: V1SignedWorkPhysicalAlternateBlockSupportBinding,
    kkt_state: V1SignedWorkLeastWorkCUDAStateHandle,
    pair_cover_receipt: V1SignedWorkVariableShellOutsidePairCoverReceipt,
    cover_binding_receipt: V1BlockSupportOutsidePairCoverBindingReceipt | None = None,
    dual_source: V1BlockSupportOutsidePairDualSourceHandle,
    block_index: int,
    design: V1BlockSupportOutsidePairCUDAExecutionDesign,
    hbm_forecast: V1BlockSupportOutsidePairHBMForecast,
) -> V1BlockSupportOutsidePairFamilyReceipt:
    """Issue one production CUDA outside-pair family from typed live sources.

    The structural provider lends fixed/global-P/shell/parent custody; the
    purpose-bound KKT lease lends resource/eligibility/required custody.  The
    frozen branch ladder supplies the alias-specific rounding anchor.  A
    compact legacy cover is diagnostic only: production requires a typed
    source-equivalence binding for its complete pair axis.
    """

    if (
        type(physical_block_support_binding)
        is not V1SignedWorkPhysicalAlternateBlockSupportBinding
        or type(kkt_state) is not V1SignedWorkLeastWorkCUDAStateHandle
        or type(pair_cover_receipt)
        is not V1SignedWorkVariableShellOutsidePairCoverReceipt
        or (
            cover_binding_receipt is not None
            and type(cover_binding_receipt)
            is not V1BlockSupportOutsidePairCoverBindingReceipt
        )
        or type(dual_source) is not V1BlockSupportOutsidePairDualSourceHandle
        or type(design) is not V1BlockSupportOutsidePairCUDAExecutionDesign
        or type(hbm_forecast) is not V1BlockSupportOutsidePairHBMForecast
    ):
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.INVALID_TYPE,
            "production outside-pair execution requires exact typed providers",
        )
    if (
        design.execution_mode
        is not V1BlockSupportOutsidePairExecutionMode.CUDA_PRODUCTION
    ):
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.CPU_PRODUCTION_FALLBACK,
            "production outside-pair issuer requires CUDA-production mode",
        )
    if cover_binding_receipt is None:
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.SOURCE_MISMATCH,
            "production typed cover equivalence unavailable: no upstream issuer "
            "proves the KKT address-identity masks equivalent to the legacy "
            "cover membership encoding",
        )
    cover_binding_receipt.assert_integrity()
    if (
        cover_binding_receipt.production_binding is not True
        or cover_binding_receipt.cpu_exact_reference_binding is not False
        or cover_binding_receipt.production_kkt_mask_identity_export_joined is not True
    ):
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.AUTHORITY_VIOLATION,
            "production outside-pair execution requires the exact KKT mask-identity "
            "cover binding",
        )
    if torch is None or not torch.cuda.is_available():
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.CUDA_REQUIRED,
            "production outside-pair issuer fails closed without CUDA",
        )
    if not torch.are_deterministic_algorithms_enabled():
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.INVALID_DESIGN,
            "production outside-pair issuer requires deterministic algorithms",
        )
    physical_block_support_binding.assert_live()
    dual_source.assert_live()
    structural_provider = physical_block_support_binding.structural_provider
    structural = structural_provider.receipt
    binding_receipt = physical_block_support_binding.receipt
    d_fixed_support = physical_block_support_binding.d_fixed_support
    residual_support = physical_block_support_binding.residual_support
    preferred_binding = structural_provider.preferred_action_binding
    registration, d_receipt, r_receipt = _validate_block_alias_sources(
        block_witness_source=block_witness_source,
        alias_binding=alias_binding,
        d_fixed_support=d_fixed_support,
        residual_support=residual_support,
        preferred_action_binding=preferred_binding,
        pair_cover_receipt=pair_cover_receipt,
        cover_binding_receipt=cover_binding_receipt,
        block_index=block_index,
    )
    dual = dual_source.receipt
    try:
        q_branch = V1SignedWorkVariableShellQBranch(alias_binding.branch_id)
        kkt_branch = V1RelaxedKKTBranch(alias_binding.branch_id)
    except ValueError:
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.ALIAS_MISMATCH,
            "alias branch is not one registered Q/KKT branch",
        )
    (
        rounding_anchor_shell_routes,
        resource_alias,
        rounding_anchor_subset,
        rounding_anchor_bundle,
    ) = _rounding_anchor_from_structural_provider(
        binding=physical_block_support_binding,
        alias_binding=alias_binding,
    )
    branch_source = structural_provider.registry.branch_source(q_branch)
    branch_registration = branch_source.registration
    bundle_receipt = rounding_anchor_bundle.execution_receipt
    try:
        state_branch_index = kkt_state.branch_order.index(kkt_branch)
    except ValueError:
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.SOURCE_MISMATCH,
            "KKT state does not retain the alias branch",
        )
    selected_branch_receipt_hash = kkt_state.branch_receipt_hashes[state_branch_index]
    if (
        branch_registration.source_receipt_hash != alias_binding.source_receipt_hash
        or branch_registration.source_replay_hash != alias_binding.source_replay_hash
        or bundle_receipt.state_identity_hash != kkt_state.identity_hash
        or bundle_receipt.problem_hash != kkt_state.problem_hash
        or bundle_receipt.policy_hash != kkt_state.policy_hash
        or bundle_receipt.branch_identity.signed_work_branch_receipt_hash
        != selected_branch_receipt_hash
        or structural.route_count != kkt_state.route_count
        or structural.resident_device_identity != kkt_state.device
        or structural.route_axis_hash != d_receipt.route_axis_hash
        or binding_receipt.d_fixed_support_receipt_hash != d_receipt.receipt_hash
        or binding_receipt.residual_support_receipt_hash != r_receipt.receipt_hash
        or dual.resource_coefficient_source_hash != kkt_state.identity_hash
        or dual.outside_pair_cover_hash != pair_cover_receipt.receipt_hash
        or dual.pair_axis_commitment_hash
        != pair_cover_receipt.pair_axis_commitment_hash
        or dual.alias_hash != alias_binding.alias_hash
        or dual.design_hash != design.design_hash
        or dual.hbm_forecast_hash != hbm_forecast.forecast_hash
        or hbm_forecast.execution_mode
        is not V1BlockSupportOutsidePairExecutionMode.CUDA_PRODUCTION
        or not hbm_forecast.admitted
        or cover_binding_receipt.kkt_problem_hash != kkt_state.problem_hash
        or cover_binding_receipt.kkt_state_identity_hash != kkt_state.identity_hash
        or cover_binding_receipt.kkt_eligible_mask_identity_hash
        != kkt_state.eligible_mask_identity_hash
        or cover_binding_receipt.kkt_required_mask_identity_hash
        != kkt_state.required_mask_identity_hash
        or cover_binding_receipt.source_rounding_hash
        != rounding_anchor_bundle.branch_identity.source_rounding_hash
        or cover_binding_receipt.frozen_ladder_receipt_hash
        != rounding_anchor_bundle.frozen_ladder.receipt_hash
        or cover_binding_receipt.shell_freeze_receipt_hash
        != structural.shell_freeze_receipt_hash
        or cover_binding_receipt.parent_mask_hash != structural.current_mask_hash
        or cover_binding_receipt.rounding_anchor_action_coordinate_hash
        != rounding_anchor_subset.physical_action_coordinate_hash
        or cover_binding_receipt.rounding_anchor_removed_route_index_hash
        != rounding_anchor_subset.removed_route_index_hash
        or cover_binding_receipt.rounding_anchor_subset_identity_hash
        != rounding_anchor_subset.subset_hash
        or cover_binding_receipt.rounding_anchor_selected_shell_offsets
        != rounding_anchor_subset.shell_offsets
        or cover_binding_receipt.alias_fixed_outside_removed_count
        != resource_alias.fixed_outside_removed_count
        or cover_binding_receipt.alias_shell_prefix_count
        != resource_alias.shell_prefix_count
        or cover_binding_receipt.alias_total_removed_count
        != resource_alias.total_removed_count
    ):
        _refuse(
            V1BlockSupportOutsidePairRefusalCode.STALE_IDENTITY,
            "physical/KKT/support/dual/cover source identities do not close",
        )
    comparison_preferred_shell_routes = (
        _preferred_shell_routes_from_structural_provider(physical_block_support_binding)
    )
    borrow_purpose = content_hash(
        {
            "operation": "block_support_outside_pair",
            "block_binding_receipt_hash": binding_receipt.receipt_hash,
            "alias_hash": alias_binding.alias_hash,
            "pair_cover_receipt_hash": pair_cover_receipt.receipt_hash,
            "dual_source_receipt_hash": dual.receipt_hash,
        },
        prefix=_PREFIX + "structural-borrow-purpose-",
    )
    with (
        d_fixed_support.open_coefficient_lease() as d_lease,
        residual_support.open_coefficient_lease() as residual_lease,
        dual_source.open_dual_lease() as dual_lease,
        physical_block_support_binding.open_fixed_membership_lease(
            borrow_purpose_hash=borrow_purpose
        ) as structural_lease,
        kkt_state.borrow_for_block_support_outside_pair(
            expected_problem_hash=kkt_state.problem_hash,
            expected_policy_hash=kkt_state.policy_hash,
            expected_branch_receipt_hashes=kkt_state.branch_receipt_hashes,
        ) as kkt_lease,
    ):
        d_coefficients = d_lease.tensor
        residual_coefficients = residual_lease.tensor
        dual_resource_copy, capacity_adjoint = dual_lease.live_route_tensors()
        if dual_resource_copy is not None:
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.AUTHORITY_VIOLATION,
                "production scan refuses a duplicated dual-owner resource vector",
            )
        resource_coefficients, eligible, required = kkt_lease.live_source_tensors()
        fixed = structural_lease.fixed_membership()
        parent_packed = structural_lease.resident_parent_packed()
        lam, rho, _mu_axis, _limit_axis = dual_lease.compact_duals
        device = d_coefficients.device
        if (
            device.type != "cuda"
            or any(
                value.device != device
                for value in (
                    residual_coefficients,
                    capacity_adjoint,
                    resource_coefficients,
                    eligible,
                    required,
                    fixed,
                    parent_packed,
                )
            )
            or str(device) != structural_lease.device_identity
            or str(device) != kkt_lease.device
        ):
            _refuse(
                V1BlockSupportOutsidePairRefusalCode.SOURCE_MISMATCH,
                "production outside-pair sources are not colocated on one CUDA device",
            )
        scan = _scan_live_outside_pairs(
            design=design,
            d_coefficients=d_coefficients,
            residual_coefficients=residual_coefficients,
            resource_coefficients=resource_coefficients,
            capacity_adjoint=capacity_adjoint,
            required=required,
            selectable=None,
            shell=None,
            eligible=eligible,
            fixed_membership=fixed,
            resident_parent_packed=parent_packed,
            shell_indices=structural_lease.shell_indices,
            rounding_anchor_shell_indices=rounding_anchor_shell_routes,
            comparison_preferred_shell_indices=(comparison_preferred_shell_routes),
            d_weight_numerator=registration.d_weight_numerator,
            residual_weight_numerator=registration.residual_weight_numerator,
            population_weight_denominator=registration.population_weight_denominator,
            d_source_l1_radius=d_receipt.coefficient_construction_l1_radius_upper,
            residual_source_l1_radius=(
                r_receipt.coefficient_construction_l1_radius_upper
            ),
            resource_multiplier=lam,
            resource_floor=rho,
            capacity_multiplier_dot_limit=(dual.capacity_multiplier_dot_limit_nominal),
            resource_multiplier_conversion_radius=(
                dual.resource_multiplier_conversion_radius
            ),
            capacity_adjoint_l1_radius=(dual.capacity_adjoint_roundoff_l1_radius),
            capacity_adjoint_linf_radius=(dual.capacity_adjoint_roundoff_linf_radius),
            dual_constant_radius=dual.dual_constant_roundoff_radius,
        )
        _validate_live_scan_against_compact_cover(
            scan=scan,
            pair_cover_receipt=pair_cover_receipt,
        )
    first = scan.first_rectangle
    second = scan.second_rectangle
    first_pair_lower = (
        math.inf
        if not first.pair_count
        else _down(scan.base_lower + first.correction_lower)
    )
    second_pair_lower = (
        math.inf
        if not second.pair_count
        else _down(scan.base_lower + second.correction_lower)
    )
    worst = scan.worst_rectangle
    preferred_dot_radius = _up(
        scan.support_source_l1_radius
        + scan.weight_conversion_l1_radius
        + scan.weighted_combination_l1_radius
        + scan.preferred_dot_reduction_radius
    )
    block_contexts = block_witness_source.block_context_ids(block_index)
    return V1BlockSupportOutsidePairFamilyReceipt(
        block_support_registration_hash=registration.registration_hash,
        block_index=block_index,
        block_id=block_witness_source.block_ids[block_index],
        block_context_axis_hash=_canonical_context_axis_hash(block_contexts),
        d_fixed_support_receipt_hash=d_receipt.receipt_hash,
        residual_support_receipt_hash=r_receipt.receipt_hash,
        support_source_registration_hash=d_receipt.source_registration_hash,
        native_fiber_axis_receipt_hash=d_receipt.native_fiber_axis_receipt_hash,
        native_fiber_measure_hash=d_receipt.native_fiber_measure_hash,
        preferred_action_hash=preferred_binding.preferred_action_hash,
        preferred_action_binding_hash=preferred_binding.binding_hash,
        preferred_resulting_mask_hash=(preferred_binding.preferred_resulting_mask_hash),
        preferred_removed_route_index_hash=(
            d_receipt.preferred_removed_route_index_hash
        ),
        model_identity_hash=d_receipt.model_identity_hash,
        chart_identity_hash=d_receipt.chart_identity_hash,
        current_mask_hash=d_receipt.current_mask_hash,
        proposal_generation=d_receipt.proposal_generation,
        route_axis_hash=d_receipt.route_axis_hash,
        route_id_axis_hash=d_receipt.route_id_axis_hash,
        registered_d_prefix_hash=d_receipt.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=d_receipt.registered_d_w2_prefix_hash,
        alias_hash=alias_binding.alias_hash,
        branch_id=alias_binding.branch_id,
        fd_rank=alias_binding.fd_rank,
        resource_alias_id=alias_binding.resource_alias_id,
        source_receipt_hash=alias_binding.source_receipt_hash,
        source_replay_hash=alias_binding.source_replay_hash,
        cardinality_alias_id=alias_binding.cardinality_alias_id,
        rounded_subset_cardinality=alias_binding.cardinality,
        alias_fixed_outside_removed_count=(resource_alias.fixed_outside_removed_count),
        alias_shell_prefix_count=resource_alias.shell_prefix_count,
        rounding_anchor_selected_count=scan.rounding_anchor_count,
        comparison_preferred_selected_count=scan.comparison_preferred_count,
        rounding_anchor_action_coordinate_hash=(
            rounding_anchor_subset.physical_action_coordinate_hash
        ),
        rounding_anchor_removed_route_index_hash=(
            rounding_anchor_subset.removed_route_index_hash
        ),
        rounding_anchor_subset_identity_hash=rounding_anchor_subset.subset_hash,
        rounding_anchor_shell_gate_hash=(
            cover_binding_receipt.rounding_anchor_shell_gate_hash
        ),
        constraint_cover_hash=alias_binding.constraint_cover_hash,
        pair_cover_receipt_hash=pair_cover_receipt.receipt_hash,
        pair_axis_commitment_hash=pair_cover_receipt.pair_axis_commitment_hash,
        typed_cover_binding_receipt_hash=cover_binding_receipt.receipt_hash,
        membership_source_receipt_hash=structural.receipt_hash,
        membership_source_kind="structural_physical_plus_kkt_live_cuda",
        structural_domain_receipt_hash=structural.receipt_hash,
        physical_block_support_binding_receipt_hash=binding_receipt.receipt_hash,
        fixed_membership_binding_hash=structural.fixed_membership_binding_hash,
        shell_freeze_receipt_hash=structural.shell_freeze_receipt_hash,
        kkt_state_identity_hash=kkt_state.identity_hash,
        kkt_problem_hash=kkt_state.problem_hash,
        kkt_policy_hash=kkt_state.policy_hash,
        kkt_branch_receipt_hash=selected_branch_receipt_hash,
        dual_source_receipt_hash=dual.receipt_hash,
        resource_axis_hash=dual.resource_axis_hash,
        resource_coefficient_source_hash=dual.resource_coefficient_source_hash,
        capacity_operator_source_hash=dual.capacity_operator_source_hash,
        resource_multiplier_source_hash=dual.resource_multiplier_source_hash,
        capacity_multiplier_source_hash=dual.capacity_multiplier_source_hash,
        route_count=d_receipt.route_count,
        capacity_count=dual.capacity_count,
        route_tile_size=design.route_tile_size,
        route_tile_scan_count=scan.route_tile_scan_count,
        complete_route_coefficient_scan_count=d_receipt.route_count,
        add_outside_rectangle_pair_count=first.pair_count,
        drop_outside_add_inside_rectangle_pair_count=second.pair_count,
        complete_pair_cover_count=first.pair_count + second.pair_count,
        add_outside_rectangle_empty=not bool(first.pair_count),
        drop_outside_add_inside_rectangle_empty=not bool(second.pair_count),
        vacuous_no_outside_pair=worst is None,
        base_dual_nominal=scan.base_nominal,
        base_dual_downward_radius=scan.base_radius,
        base_dual_lower_outward=scan.base_lower,
        comparison_preferred_dot_nominal=scan.preferred_dot_nominal,
        comparison_preferred_dot_absolute_radius=preferred_dot_radius,
        comparison_preferred_dot_upper_outward=_up(
            scan.preferred_dot_nominal + preferred_dot_radius
        ),
        support_coefficient_source_l1_radius=scan.support_source_l1_radius,
        weight_conversion_l1_radius=scan.weight_conversion_l1_radius,
        weighted_combination_roundoff_l1_radius=(scan.weighted_combination_l1_radius),
        resource_dual_roundoff_l1_radius=scan.resource_dual_l1_radius,
        capacity_dual_roundoff_l1_radius=scan.capacity_dual_l1_radius,
        stationarity_addition_roundoff_l1_radius=(scan.stationarity_addition_l1_radius),
        box_reduction_roundoff_radius=scan.box_reduction_radius,
        dual_constant_roundoff_radius=scan.constant_radius,
        stationarity_sign_linf_radius=scan.stationarity_sign_linf_radius,
        ambiguous_p_endpoint_count=(first.ambiguous_p_count + second.ambiguous_p_count),
        ambiguous_q_endpoint_count=(first.ambiguous_q_count + second.ambiguous_q_count),
        add_outside_correction_lower=(
            0.0 if not first.pair_count else first.correction_lower
        ),
        drop_outside_add_inside_correction_lower=(
            0.0 if not second.pair_count else second.correction_lower
        ),
        add_outside_pair_lower_outward=first_pair_lower,
        drop_outside_add_inside_pair_lower_outward=second_pair_lower,
        worst_pair_rectangle=None if worst is None else worst.rectangle,
        worst_pair_preferred_selected_index=(None if worst is None else worst.p_index),
        worst_pair_preferred_unselected_index=(
            None if worst is None else worst.q_index
        ),
        worst_pair_ordinal=None if worst is None else worst.pair_ordinal,
        worst_pair_lower_center=scan.worst_center,
        worst_pair_total_downward_radius=scan.worst_radius,
        worst_pair_lower_outward=scan.worst_lower,
        exact_reference_worst_pair_numerator=None,
        exact_reference_worst_pair_denominator=None,
        design_hash=design.design_hash,
        hbm_forecast_hash=hbm_forecast.forecast_hash,
        execution_mode=design.execution_mode,
        production_typed_membership_sources_joined=True,
        cpu_reference_four_mask_only=False,
        scalar_device_to_host_transfer_bytes=scan.scalar_d2h_bytes,
        scan_device_to_host_synchronization_count=(scan.d2h_synchronization_count),
        comparison_preferred_is_rounding_anchor=(
            rounding_anchor_subset.removed_route_index_hash
            == preferred_binding.preferred_removed_route_index_hash
        ),
        same_cardinality_alias_only=(
            alias_binding.cardinality == registration.preferred_action_cardinality
        ),
        cross_cardinality_alias_authority=(
            alias_binding.cardinality != registration.preferred_action_cardinality
        ),
        typed_cover_equivalence_joined=True,
        _issuer_token=_FAMILY_TOKEN,
    )


__all__ = (
    "V1BlockSupportOutsidePairCUDAExecutionDesign",
    "V1BlockSupportOutsidePairCoverBindingReceipt",
    "V1BlockSupportOutsidePairDualLease",
    "V1BlockSupportOutsidePairDualSourceHandle",
    "V1BlockSupportOutsidePairDualSourceReceipt",
    "V1BlockSupportOutsidePairExecutionMode",
    "V1BlockSupportOutsidePairFamilyReceipt",
    "V1BlockSupportOutsidePairHBMForecast",
    "V1BlockSupportOutsidePairMembershipLease",
    "V1BlockSupportOutsidePairMembershipSourceHandle",
    "V1BlockSupportOutsidePairMembershipSourceReceipt",
    "V1BlockSupportOutsidePairRefusal",
    "V1BlockSupportOutsidePairRefusalCode",
    "bind_v1_block_support_outside_pair_cover_reference",
    "bind_v1_block_support_outside_pair_dual_source",
    "bind_v1_block_support_outside_pair_membership_source",
    "build_v1_block_support_outside_pair_cuda_design",
    "build_v1_block_support_outside_pair_hbm_forecast",
    "evaluate_v1_block_support_outside_pair_lower_bound_reference",
    "issue_v1_block_support_outside_pair_lower_bound_cuda",
)
