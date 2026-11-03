"""Projected-dual CUDA lower bounds for the registered within-shell family.

This module is the production-device successor of the small SciPy reference
solver in :mod:`modern_v1_signed_work_block_support_alternate_union_cuda`, but
its authority name is intentionally narrower than that predecessor's name.
The live inputs currently describe only the branch x resource-alias x closed
*within-shell* cell family.  No live provider binds alias-specific fixed-
outside membership or the complete outside-pair cover.  Consequently this
module cannot issue the canonical full-U block observation and keeps full-U,
action-invariance, C, K, O, commit, terminal, and Pareto authority closed.

For one registered cell the relaxed problem is

    min c^T z
    s.t. r^T z >= R, Bz <= k, 0 <= z <= 1,
         z_i >= t (i selected), z_i <= t (i unselected), 0 <= t <= 1.

The scalar ``t`` is an exact compact representation of every pairwise closed
cell-order inequality.  A deterministic FP64 Chambolle--Pock iteration
produces nonnegative multipliers.  The numerical authority is *not* its
primal iterate or objective.  It is the projected-dual value

    lambda R - mu^T k + sum_i min(0, s_i) + min(0, w),

with ``s = c-lambda*r+B^T*mu+O^T*q`` and the analogous scalar ``w``.  Every
finite nonnegative multiplier vector is dual feasible.  An explicit rounded-
operation upper count and absolute-mass/gamma enclosure turn the device
reduction into a downward binary64 bound.  A feasible primal objective is
reported only as an upper-bound diagnostic and can never authorize a lower
bound.

All route-scale support coefficients remain under issuer-owned device leases.
Only the registered shell is gathered.  Production refuses CPU fallback,
requires strict deterministic PyTorch algorithms, forecasts HBM before
reading coefficient values, tiles the cell family with bounded workspace,
performs no route-scale D2H or tensor content digest, and emits array-free,
issuer-only receipts.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
from fractions import Fraction
import math
from numbers import Integral, Real
from typing import Any, Final, NoReturn, Sequence

from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

from .contracts import content_hash
from .modern_v1_signed_work_block_support_source_cuda import (
    V1SignedWorkBlockSupportExecutionMode,
    V1SignedWorkBlockSupportPreferredActionBinding,
    V1SignedWorkBlockSupportRefusal,
    V1SignedWorkBlockSupportSourceDeviceHandle,
    V1SignedWorkBlockSupportSourceRole,
)
from .modern_v1_signed_work_block_support_witness import (
    V1BlockSupportWitnessSource,
)
from .modern_v1_signed_work_variable_shell_action_separation_cuda import (
    V1SignedWorkVariableShellActionCellBlockCapacity,
    V1SignedWorkVariableShellActionCellRegistry,
    V1SignedWorkVariableShellActionSeparationMode,
    V1SignedWorkVariableShellActionSeparationRefusal,
    V1SignedWorkVariableShellQBranch,
    build_v1_signed_work_variable_shell_restricted_cell_request,
)
from .modern_v1_signed_work_variable_shell_source_cuda import (
    V1SignedWorkVariableShellSourceRegistration,
)

try:  # pragma: no cover - runtime-image dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


WITHIN_SHELL_SOLVER_SCHEMA: Final = (
    "partition_free_v1_block_support_within_shell_device_solver_v1"
)
WITHIN_SHELL_FORECAST_SCHEMA: Final = (
    "partition_free_v1_block_support_within_shell_hbm_forecast_v1"
)
WITHIN_SHELL_CELL_SCHEMA: Final = (
    "partition_free_v1_block_support_within_shell_cell_dual_lower_v1"
)
WITHIN_SHELL_EXECUTION_SCHEMA: Final = (
    "partition_free_v1_block_support_within_shell_device_execution_v1"
)
WITHIN_SHELL_FAMILY_SCHEMA: Final = (
    "partition_free_v1_block_support_within_shell_lower_bound_family_v1"
)
WITHIN_SHELL_OBSERVATION_SCHEMA: Final = (
    "partition_free_v1_block_support_within_shell_margin_observation_v1"
)
WITHIN_SHELL_RULE: Final = (
    "complete_registered_branch_alias_closed_within_shell_cells_projected_"
    "dual_downward_enclosure_fixed_outside_and_outside_pair_absent_v1"
)
WITHIN_SHELL_DUAL_RULE: Final = (
    "dual_equals_lambda_resource_floor_minus_mu_capacity_plus_box_minima_"
    "for_route_and_compact_order_threshold_with_mass_gamma_debit_v1"
)
WITHIN_SHELL_AUTHORITY: Final = (
    "registered_within_shell_lower_bound_only_no_full_U_outside_pair_action_"
    "invariance_C_K_O_commit_terminal_or_pareto_authority_v1"
)
WITHIN_SHELL_BACKEND: Final = (
    "deterministic_torch_fp64_tiled_compact_order_projected_dual_pdhg_v1"
)

_PREFIX: Final = "partition-free-v1-block-support-within-shell-device-"
_DESIGN_TOKEN: Final = object()
_FORECAST_TOKEN: Final = object()
_CELL_TOKEN: Final = object()
_EXECUTION_TOKEN: Final = object()
_FAMILY_TOKEN: Final = object()
_OBSERVATION_TOKEN: Final = object()
_F64_EPS: Final = 2.0**-52
_F64_MIN: Final = math.nextafter(0.0, math.inf)


class V1BlockSupportWithinShellSolverRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    INVALID_DESIGN = "invalid_design"
    STALE_OR_MUTATED_SOURCE = "stale_or_mutated_source"
    INCOMPLETE_WITHIN_SHELL_FAMILY = "incomplete_within_shell_family"
    OUTSIDE_PROVIDER_REQUIRED = "outside_provider_required"
    CUDA_REQUIRED = "cuda_required"
    DETERMINISM_REQUIRED = "determinism_required"
    HBM_FORECAST_EXCEEDED = "hbm_forecast_exceeded"
    NONFINITE_SOLVER = "nonfinite_solver"
    NUMERICAL_ENCLOSURE_REFUSAL = "numerical_enclosure_refusal"
    AUTHORITY_VIOLATION = "authority_violation"
    RECEIPT_TAMPERING = "receipt_tampering"


class V1BlockSupportWithinShellSolverRefusal(RuntimeError):
    def __init__(
        self,
        code: V1BlockSupportWithinShellSolverRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1BlockSupportWithinShellSolverRefusalCode:
            raise TypeError("within-shell refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1BlockSupportWithinShellSolverRefusalCode,
    message: str,
) -> NoReturn:
    raise V1BlockSupportWithinShellSolverRefusal(code, message)


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _integer(value: object, *, name: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.INVALID_TYPE,
            f"{name} must be integral",
        )
    answer = int(value)
    if answer < (1 if positive else 0):
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.INVALID_DESIGN,
            f"{name} is out of range",
        )
    return answer


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.INVALID_TYPE,
            f"{name} must be one real scalar",
        )
    answer = float(value)
    if not math.isfinite(answer):
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.NONFINITE_SOLVER,
            f"{name} must be finite",
        )
    return 0.0 if answer == 0.0 else answer


def _nonnegative(value: object, *, name: str) -> float:
    answer = _finite(value, name=name)
    if answer < 0.0:
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.INVALID_DESIGN,
            f"{name} must be nonnegative",
        )
    return answer


def _gamma(operation_count: int) -> float:
    count = _integer(operation_count, name="operation count")
    product = count * _F64_EPS
    if product >= 0.125:
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.NUMERICAL_ENCLOSURE_REFUSAL,
            "binary64 gamma model exceeded its registered safety domain",
        )
    return product / (1.0 - product)


def _up(value: float) -> float:
    return math.nextafter(float(value), math.inf)


def _down(value: float) -> float:
    return math.nextafter(float(value), -math.inf)


def _fraction_upper(value: Fraction) -> float:
    center = float(value)
    if not math.isfinite(center):
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.NONFINITE_SOLVER,
            "exact registered scalar exceeds binary64",
        )
    if Fraction.from_float(center) < value:
        center = math.nextafter(center, math.inf)
    return center


def _closed(values: Sequence[object], *, owner: str) -> None:
    if any(type(value) is not bool or value is not False for value in values):
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.AUTHORITY_VIOLATION,
            f"{owner} crossed its within-shell-only authority boundary",
        )


def _payload(instance: object) -> dict[str, object]:
    fields = getattr(instance, "__dataclass_fields__")
    return {
        name: (
            getattr(instance, name).value
            if isinstance(getattr(instance, name), Enum)
            else getattr(instance, name)
        )
        for name in fields
        if not name.startswith("_") and name not in {"receipt_hash"}
    }


def _seal(instance: object, *, prefix: str) -> tuple[str, str]:
    receipt_hash = content_hash(_payload(instance), prefix=prefix)
    integrity = content_hash(
        {"receipt_hash": receipt_hash, "payload": _payload(instance)},
        prefix=prefix + "integrity-",
    )
    return receipt_hash, integrity


def _assert_seal(instance: object, *, prefix: str) -> str:
    expected_hash, expected_integrity = _seal(instance, prefix=prefix)
    if (
        getattr(instance, "receipt_hash") != expected_hash
        or getattr(instance, "_integrity_seal") != expected_integrity
    ):
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.RECEIPT_TAMPERING,
            "issuer receipt no longer replays",
        )
    return expected_hash


@dataclass(frozen=True, slots=True)
class V1BlockSupportWithinShellCUDAExecutionDesign:
    """Numerical/workspace design; iteration count is compute, not science."""

    execution_mode: V1SignedWorkBlockSupportExecutionMode
    pdhg_iteration_count: int = 2_048
    maximum_cell_tile_size: int = 128
    hbm_ceiling_fraction: float = 0.85
    deterministic_fp64_required: bool = True
    compact_threshold_order_required: bool = True
    no_route_scale_d2h_or_digest: bool = True
    full_u_authority_requested: bool = False
    schema: str = WITHIN_SHELL_SOLVER_SCHEMA
    _issuer_token: InitVar[object] = _DESIGN_TOKEN

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _DESIGN_TOKEN:
            raise TypeError("within-shell designs are factory-bound")
        if type(self.execution_mode) is not V1SignedWorkBlockSupportExecutionMode:
            raise TypeError("within-shell execution mode must be exact")
        object.__setattr__(
            self,
            "pdhg_iteration_count",
            _integer(
                self.pdhg_iteration_count,
                name="PDHG iteration count",
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "maximum_cell_tile_size",
            _integer(
                self.maximum_cell_tile_size,
                name="maximum cell tile size",
                positive=True,
            ),
        )
        fraction = _finite(self.hbm_ceiling_fraction, name="HBM ceiling fraction")
        if not (0.0 < fraction <= 0.85):
            _refuse(
                V1BlockSupportWithinShellSolverRefusalCode.INVALID_DESIGN,
                "HBM ceiling fraction must be in (0,.85]",
            )
        object.__setattr__(self, "hbm_ceiling_fraction", fraction)
        if (
            self.deterministic_fp64_required is not True
            or self.compact_threshold_order_required is not True
            or self.no_route_scale_d2h_or_digest is not True
            or self.full_u_authority_requested is not False
            or self.schema != WITHIN_SHELL_SOLVER_SCHEMA
        ):
            _refuse(
                V1BlockSupportWithinShellSolverRefusalCode.AUTHORITY_VIOLATION,
                "within-shell design changed a mandatory numerical/authority invariant",
            )

    @property
    def design_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "design-")


@dataclass(frozen=True, slots=True)
class V1BlockSupportWithinShellHBMForecast:
    design_hash: str
    execution_mode: V1SignedWorkBlockSupportExecutionMode
    registry_receipt_hash: str
    d_fixed_support_receipt_hash: str
    residual_support_receipt_hash: str
    preferred_action_binding_hash: str
    route_count: int
    shell_size: int
    registered_cell_count: int
    cell_tile_size: int
    block_count: int
    live_source_bytes_in_current_reservation: int
    live_source_unaccounted_bytes: int
    packed_action_device_bytes: int
    packed_action_tile_staging_bytes: int
    route_action_materialization_bytes: int
    shell_index_bytes: int
    shell_fp64_vector_bytes: int
    shell_constraint_bytes: int
    shell_static_bytes: int
    tiled_state_bytes: int
    backend_workspace_bytes: int
    tiled_solver_workspace_bytes: int
    incremental_bytes: int
    current_reserved_bytes: int
    projected_bytes: int
    device_total_bytes: int
    device_free_bytes: int
    ceiling_bytes: int
    fits: bool
    receipt_hash: str = field(init=False)
    _issuer_token: InitVar[object] = None
    computed_before_support_coefficient_contraction: bool = True
    production_live_device_query_bound: bool = False
    reference_registered_counter_mode: bool = True
    source_colive_accounted_in_current_reservation: bool = True
    source_colive_accounted_in_forecast: bool = True
    route_scale_d2h_bytes: int = 0
    schema: str = WITHIN_SHELL_FORECAST_SCHEMA
    _integrity_seal: str = field(default="", repr=False, compare=False)

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _FORECAST_TOKEN:
            raise TypeError("within-shell HBM forecasts are issuer-only")
        for name in (
            "design_hash",
            "registry_receipt_hash",
            "d_fixed_support_receipt_hash",
            "residual_support_receipt_hash",
            "preferred_action_binding_hash",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        if type(self.execution_mode) is not V1SignedWorkBlockSupportExecutionMode:
            raise TypeError("forecast execution mode must be exact")
        for name in (
            "route_count",
            "shell_size",
            "registered_cell_count",
            "cell_tile_size",
            "block_count",
            "live_source_bytes_in_current_reservation",
            "live_source_unaccounted_bytes",
            "packed_action_device_bytes",
            "packed_action_tile_staging_bytes",
            "route_action_materialization_bytes",
            "shell_index_bytes",
            "shell_fp64_vector_bytes",
            "shell_constraint_bytes",
            "shell_static_bytes",
            "tiled_state_bytes",
            "backend_workspace_bytes",
            "tiled_solver_workspace_bytes",
            "incremental_bytes",
            "current_reserved_bytes",
            "projected_bytes",
            "device_total_bytes",
            "device_free_bytes",
            "ceiling_bytes",
            "route_scale_d2h_bytes",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name=name))
        expected_incremental = (
            self.live_source_unaccounted_bytes
            + self.route_action_materialization_bytes
            + self.shell_static_bytes
            + self.tiled_solver_workspace_bytes
        )
        expected_unaccounted = max(
            0,
            self.live_source_bytes_in_current_reservation - self.current_reserved_bytes,
        )
        if (
            min(
                self.route_count,
                self.shell_size,
                self.registered_cell_count,
                self.cell_tile_size,
                self.block_count,
            )
            <= 0
            or self.route_action_materialization_bytes
            != self.packed_action_device_bytes + self.packed_action_tile_staging_bytes
            or self.live_source_unaccounted_bytes != expected_unaccounted
            or self.shell_static_bytes
            != self.shell_index_bytes
            + self.shell_fp64_vector_bytes
            + self.shell_constraint_bytes
            or self.tiled_solver_workspace_bytes
            != self.tiled_state_bytes + self.backend_workspace_bytes
            or self.incremental_bytes != expected_incremental
            or self.projected_bytes
            != self.current_reserved_bytes + self.incremental_bytes
            or self.route_scale_d2h_bytes != 0
            or type(self.fits) is not bool
            or self.computed_before_support_coefficient_contraction is not True
            or (
                self.production_live_device_query_bound,
                self.reference_registered_counter_mode,
            )
            != (
                self.execution_mode
                is V1SignedWorkBlockSupportExecutionMode.CUDA_PRODUCTION,
                self.execution_mode
                is V1SignedWorkBlockSupportExecutionMode.CPU_REFERENCE_TEST_ONLY,
            )
            or self.source_colive_accounted_in_current_reservation
            is not (self.live_source_unaccounted_bytes == 0)
            or self.source_colive_accounted_in_forecast is not True
            or self.schema != WITHIN_SHELL_FORECAST_SCHEMA
        ):
            _refuse(
                V1BlockSupportWithinShellSolverRefusalCode.INVALID_DESIGN,
                "within-shell HBM forecast arithmetic changed",
            )
        receipt_hash, integrity = _seal(self, prefix=_PREFIX + "forecast-")
        object.__setattr__(self, "receipt_hash", receipt_hash)
        object.__setattr__(self, "_integrity_seal", integrity)

    def assert_integrity(self) -> str:
        return _assert_seal(self, prefix=_PREFIX + "forecast-")


@dataclass(frozen=True, slots=True)
class V1BlockSupportWithinShellCellDualLowerBoundReceipt:
    registry_receipt_hash: str
    request_hash: str
    action_cell_hash: str
    branch: V1SignedWorkVariableShellQBranch
    resource_alias_id: str
    action_cell_ordinal: int
    selected_shell_offset_hash: str
    problem_hash: str
    pdhg_iteration_count: int
    primal_dual_operator_frobenius_upper: float
    primal_step_size: float
    dual_step_size: float
    projected_dual_center: float
    projected_dual_roundoff_radius: float
    projected_dual_lower: float
    feasible_primal_objective_upper: float | None
    diagnostic_duality_gap_upper: float | None
    projected_dual_operation_count: int
    pdhg_rounded_operation_upper_bound: int
    receipt_hash: str = field(init=False)
    _issuer_token: InitVar[object] = None
    compact_threshold_exact_closed_order_representation: bool = True
    projected_nonnegative_dual_witness: bool = True
    downward_mass_gamma_enclosure: bool = True
    feasible_primal_is_upper_diagnostic_only: bool = True
    primal_objective_used_as_lower_bound: bool = False
    global_relaxed_optimality_claim: bool = False
    full_u_authority: bool = False
    outside_pair_authority: bool = False
    action_invariance_authority: bool = False
    C_stream_open_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    dual_rule: str = WITHIN_SHELL_DUAL_RULE
    authority_scope: str = WITHIN_SHELL_AUTHORITY
    schema: str = WITHIN_SHELL_CELL_SCHEMA
    _integrity_seal: str = field(default="", repr=False, compare=False)

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _CELL_TOKEN:
            raise TypeError("within-shell cell lower bounds are issuer-only")
        for name in (
            "registry_receipt_hash",
            "request_hash",
            "action_cell_hash",
            "resource_alias_id",
            "selected_shell_offset_hash",
            "problem_hash",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        if type(self.branch) is not V1SignedWorkVariableShellQBranch:
            raise TypeError("within-shell cell branch must be exact")
        for name in (
            "action_cell_ordinal",
            "pdhg_iteration_count",
            "projected_dual_operation_count",
            "pdhg_rounded_operation_upper_bound",
        ):
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    name=name,
                    positive=name != "action_cell_ordinal",
                ),
            )
        for name in (
            "primal_dual_operator_frobenius_upper",
            "primal_step_size",
            "dual_step_size",
        ):
            value = _nonnegative(getattr(self, name), name=name)
            if value <= 0.0:
                _refuse(
                    V1BlockSupportWithinShellSolverRefusalCode.INVALID_DESIGN,
                    f"{name} must be positive",
                )
            object.__setattr__(self, name, value)
        center = _finite(self.projected_dual_center, name="projected dual center")
        radius = _nonnegative(
            self.projected_dual_roundoff_radius,
            name="projected dual roundoff radius",
        )
        lower = _finite(self.projected_dual_lower, name="projected dual lower")
        if lower > _down(center - radius):
            _refuse(
                V1BlockSupportWithinShellSolverRefusalCode.NUMERICAL_ENCLOSURE_REFUSAL,
                "projected dual lower is not downward of its center/radius",
            )
        object.__setattr__(self, "projected_dual_center", center)
        object.__setattr__(self, "projected_dual_roundoff_radius", radius)
        object.__setattr__(self, "projected_dual_lower", lower)
        upper = self.feasible_primal_objective_upper
        gap = self.diagnostic_duality_gap_upper
        if upper is None:
            if gap is not None:
                _refuse(
                    V1BlockSupportWithinShellSolverRefusalCode.INVALID_DESIGN,
                    "absent primal diagnostic cannot carry a gap",
                )
        else:
            upper_value = _finite(upper, name="feasible primal objective upper")
            gap_value = _nonnegative(gap, name="diagnostic duality gap upper")
            if gap_value < max(0.0, upper_value - lower):
                _refuse(
                    V1BlockSupportWithinShellSolverRefusalCode.NUMERICAL_ENCLOSURE_REFUSAL,
                    "reported primal/dual gap does not enclose its endpoints",
                )
            object.__setattr__(self, "feasible_primal_objective_upper", upper_value)
            object.__setattr__(self, "diagnostic_duality_gap_upper", gap_value)
        required_true = (
            self.compact_threshold_exact_closed_order_representation,
            self.projected_nonnegative_dual_witness,
            self.downward_mass_gamma_enclosure,
            self.feasible_primal_is_upper_diagnostic_only,
        )
        if any(value is not True for value in required_true):
            _refuse(
                V1BlockSupportWithinShellSolverRefusalCode.AUTHORITY_VIOLATION,
                "cell lower-bound proof is incomplete",
            )
        _closed(
            (
                self.primal_objective_used_as_lower_bound,
                self.global_relaxed_optimality_claim,
                self.full_u_authority,
                self.outside_pair_authority,
                self.action_invariance_authority,
                self.C_stream_open_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="within-shell cell lower bound",
        )
        if (
            self.dual_rule != WITHIN_SHELL_DUAL_RULE
            or self.authority_scope != WITHIN_SHELL_AUTHORITY
            or self.schema != WITHIN_SHELL_CELL_SCHEMA
        ):
            _refuse(
                V1BlockSupportWithinShellSolverRefusalCode.AUTHORITY_VIOLATION,
                "cell lower-bound semantics changed",
            )
        receipt_hash, integrity = _seal(self, prefix=_PREFIX + "cell-")
        object.__setattr__(self, "receipt_hash", receipt_hash)
        object.__setattr__(self, "_integrity_seal", integrity)

    def assert_integrity(self) -> str:
        return _assert_seal(self, prefix=_PREFIX + "cell-")


@dataclass(frozen=True, slots=True)
class V1BlockSupportWithinShellCUDAExecutionReceipt:
    design_hash: str
    forecast_receipt_hash: str
    block_support_registration_hash: str
    variable_shell_source_registration_hash: str
    registry_receipt_hash: str
    d_fixed_support_receipt_hash: str
    residual_support_receipt_hash: str
    preferred_action_binding_hash: str
    preferred_action_hash: str
    block_index: int
    route_count: int
    device_type: str
    device_index: int
    registered_cell_count: int
    solved_cell_count: int
    cell_tile_size: int
    tile_count: int
    branch_axis: tuple[str, ...]
    resource_alias_axis: tuple[tuple[str, str], ...]
    action_cell_hash_axis: tuple[str, ...]
    cell_receipt_hash_axis: tuple[str, ...]
    pdhg_iteration_count_per_cell: int
    pdhg_rounded_operation_upper_bound: int
    projected_dual_operation_count: int
    objective_combination_operation_count: int
    cell_scale_d2h_bytes: int
    packed_host_bytes_read: int
    route_scale_h2d_payload_bytes: int
    maximum_h2d_staging_bytes: int
    receipt_hash: str = field(init=False)
    _issuer_token: InitVar[object] = None
    deterministic_fp64_tiled_device_execution: bool = True
    source_custody_leases_held_through_execution: bool = True
    complete_registered_within_shell_family_solved: bool = True
    complete_physical_alternate_union_solved: bool = False
    outside_pair_provider_present: bool = False
    fixed_outside_membership_provider_present: bool = False
    route_scale_d2h_bytes: int = 0
    route_scale_tensor_digest_bytes: int = 0
    CPU_reference_test_mode: bool = False
    production_CUDA_execution: bool = False
    full_u_authority: bool = False
    action_invariance_authority: bool = False
    C_stream_open_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    backend: str = WITHIN_SHELL_BACKEND
    rule: str = WITHIN_SHELL_RULE
    authority_scope: str = WITHIN_SHELL_AUTHORITY
    schema: str = WITHIN_SHELL_EXECUTION_SCHEMA
    _integrity_seal: str = field(default="", repr=False, compare=False)

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _EXECUTION_TOKEN:
            raise TypeError("within-shell execution receipts are issuer-only")
        for name in (
            "design_hash",
            "forecast_receipt_hash",
            "block_support_registration_hash",
            "variable_shell_source_registration_hash",
            "registry_receipt_hash",
            "d_fixed_support_receipt_hash",
            "residual_support_receipt_hash",
            "preferred_action_binding_hash",
            "preferred_action_hash",
            "device_type",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        for name in (
            "block_index",
            "route_count",
            "device_index",
            "registered_cell_count",
            "solved_cell_count",
            "cell_tile_size",
            "tile_count",
            "pdhg_iteration_count_per_cell",
            "pdhg_rounded_operation_upper_bound",
            "projected_dual_operation_count",
            "objective_combination_operation_count",
            "cell_scale_d2h_bytes",
            "packed_host_bytes_read",
            "route_scale_h2d_payload_bytes",
            "maximum_h2d_staging_bytes",
            "route_scale_d2h_bytes",
            "route_scale_tensor_digest_bytes",
        ):
            value = _integer(getattr(self, name), name=name)
            if (
                name
                not in {
                    "block_index",
                    "device_index",
                    "route_scale_d2h_bytes",
                    "route_scale_tensor_digest_bytes",
                }
                and value <= 0
            ):
                _refuse(
                    V1BlockSupportWithinShellSolverRefusalCode.INVALID_DESIGN,
                    f"{name} must be positive",
                )
            object.__setattr__(self, name, value)
        branches = tuple(_text(value, name="branch") for value in self.branch_axis)
        aliases = tuple(
            (_text(branch, name="alias branch"), _text(alias, name="alias id"))
            for branch, alias in self.resource_alias_axis
        )
        cells = tuple(
            _text(value, name="action-cell hash")
            for value in self.action_cell_hash_axis
        )
        receipts = tuple(
            _text(value, name="cell receipt hash")
            for value in self.cell_receipt_hash_axis
        )
        if (
            not branches
            or len(branches) != len(set(branches))
            or not aliases
            or len(aliases) != len(set(aliases))
            or not cells
            or len(cells) != len(set(cells))
            or len(receipts) != len(cells)
            or len(receipts) != len(set(receipts))
            or self.solved_cell_count != self.registered_cell_count
            or self.solved_cell_count != len(cells)
        ):
            _refuse(
                V1BlockSupportWithinShellSolverRefusalCode.INCOMPLETE_WITHIN_SHELL_FAMILY,
                "execution receipt does not cover the exact registered cell family",
            )
        object.__setattr__(self, "branch_axis", branches)
        object.__setattr__(self, "resource_alias_axis", aliases)
        object.__setattr__(self, "action_cell_hash_axis", cells)
        object.__setattr__(self, "cell_receipt_hash_axis", receipts)
        expected_modes = (
            self.device_type == "cpu",
            self.device_type == "cuda",
        )
        if (
            self.CPU_reference_test_mode,
            self.production_CUDA_execution,
        ) != expected_modes:
            _refuse(
                V1BlockSupportWithinShellSolverRefusalCode.AUTHORITY_VIOLATION,
                "execution mode evidence does not match the device",
            )
        if (
            self.deterministic_fp64_tiled_device_execution is not True
            or self.source_custody_leases_held_through_execution is not True
            or self.complete_registered_within_shell_family_solved is not True
            or self.complete_physical_alternate_union_solved is not False
            or self.outside_pair_provider_present is not False
            or self.fixed_outside_membership_provider_present is not False
            or self.route_scale_h2d_payload_bytes != self.route_count
            or self.maximum_h2d_staging_bytes != min(self.route_count, 8 * (1 << 20))
            or self.route_scale_d2h_bytes != 0
            or self.route_scale_tensor_digest_bytes != 0
        ):
            _refuse(
                V1BlockSupportWithinShellSolverRefusalCode.AUTHORITY_VIOLATION,
                "execution receipt overstates source coverage or transfers",
            )
        _closed(
            (
                self.full_u_authority,
                self.action_invariance_authority,
                self.C_stream_open_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="within-shell execution",
        )
        if (
            self.backend != WITHIN_SHELL_BACKEND
            or self.rule != WITHIN_SHELL_RULE
            or self.authority_scope != WITHIN_SHELL_AUTHORITY
            or self.schema != WITHIN_SHELL_EXECUTION_SCHEMA
        ):
            _refuse(
                V1BlockSupportWithinShellSolverRefusalCode.AUTHORITY_VIOLATION,
                "execution semantics changed",
            )
        receipt_hash, integrity = _seal(self, prefix=_PREFIX + "execution-")
        object.__setattr__(self, "receipt_hash", receipt_hash)
        object.__setattr__(self, "_integrity_seal", integrity)

    def assert_integrity(self) -> str:
        return _assert_seal(self, prefix=_PREFIX + "execution-")


@dataclass(frozen=True, slots=True)
class V1BlockSupportWithinShellLowerBoundFamilyReceipt:
    execution_receipt_hash: str
    block_support_registration_hash: str
    registry_receipt_hash: str
    block_index: int
    block_id: str
    block_context_axis_hash: str
    block_context_count: int
    preferred_action_hash: str
    preferred_action_binding_hash: str
    branch_axis: tuple[str, ...]
    resource_alias_axis: tuple[tuple[str, str], ...]
    action_cell_hash_axis: tuple[str, ...]
    cell_receipt_hash_axis: tuple[str, ...]
    minimum_cell_hash: str
    weighted_within_shell_slope_center: float
    solver_downward_numerical_radius: float
    weighted_within_shell_slope_lower: float
    objective_combination_l1_radius_upper: float
    preferred_dot_roundoff_radius_upper: float
    receipt_hash: str = field(init=False)
    _issuer_token: InitVar[object] = None
    complete_registered_branch_axis: bool = True
    complete_registered_resource_alias_axis: bool = True
    complete_registered_within_shell_cell_axis: bool = True
    physical_action_crosswalk_present: bool = False
    P_excluded_physical_alternate_union_deduplicated: bool = False
    alias_specific_fixed_outside_membership_present: bool = False
    outside_pair_cover_present: bool = False
    canonical_block_observation_authority: bool = False
    full_u_authority: bool = False
    action_invariance_authority: bool = False
    C_stream_open_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    rule: str = WITHIN_SHELL_RULE
    authority_scope: str = WITHIN_SHELL_AUTHORITY
    schema: str = WITHIN_SHELL_FAMILY_SCHEMA
    _integrity_seal: str = field(default="", repr=False, compare=False)

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _FAMILY_TOKEN:
            raise TypeError("within-shell family receipts are issuer-only")
        for name in (
            "execution_receipt_hash",
            "block_support_registration_hash",
            "registry_receipt_hash",
            "block_id",
            "block_context_axis_hash",
            "preferred_action_hash",
            "preferred_action_binding_hash",
            "minimum_cell_hash",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        for name in ("block_index", "block_context_count"):
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    name=name,
                    positive=name == "block_context_count",
                ),
            )
        branches = tuple(_text(value, name="branch") for value in self.branch_axis)
        aliases = tuple(
            (_text(branch, name="alias branch"), _text(alias, name="alias id"))
            for branch, alias in self.resource_alias_axis
        )
        cells = tuple(
            _text(value, name="action cell") for value in self.action_cell_hash_axis
        )
        receipts = tuple(
            _text(value, name="cell receipt") for value in self.cell_receipt_hash_axis
        )
        if (
            not branches
            or len(branches) != len(set(branches))
            or not aliases
            or len(aliases) != len(set(aliases))
            or not cells
            or len(cells) != len(set(cells))
            or len(receipts) != len(cells)
            or len(receipts) != len(set(receipts))
            or self.minimum_cell_hash not in cells
        ):
            _refuse(
                V1BlockSupportWithinShellSolverRefusalCode.INCOMPLETE_WITHIN_SHELL_FAMILY,
                "family receipt does not bind one exact registered cell axis",
            )
        object.__setattr__(self, "branch_axis", branches)
        object.__setattr__(self, "resource_alias_axis", aliases)
        object.__setattr__(self, "action_cell_hash_axis", cells)
        object.__setattr__(self, "cell_receipt_hash_axis", receipts)
        center = _finite(
            self.weighted_within_shell_slope_center,
            name="within-shell slope center",
        )
        radius = _nonnegative(
            self.solver_downward_numerical_radius,
            name="solver downward numerical radius",
        )
        lower = _finite(
            self.weighted_within_shell_slope_lower,
            name="within-shell slope lower",
        )
        if lower > _down(center - radius):
            _refuse(
                V1BlockSupportWithinShellSolverRefusalCode.NUMERICAL_ENCLOSURE_REFUSAL,
                "family slope is not enclosed downward",
            )
        object.__setattr__(self, "weighted_within_shell_slope_center", center)
        object.__setattr__(self, "solver_downward_numerical_radius", radius)
        object.__setattr__(self, "weighted_within_shell_slope_lower", lower)
        for name in (
            "objective_combination_l1_radius_upper",
            "preferred_dot_roundoff_radius_upper",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative(getattr(self, name), name=name),
            )
        if (
            self.complete_registered_branch_axis is not True
            or self.complete_registered_resource_alias_axis is not True
            or self.complete_registered_within_shell_cell_axis is not True
        ):
            _refuse(
                V1BlockSupportWithinShellSolverRefusalCode.INCOMPLETE_WITHIN_SHELL_FAMILY,
                "registered within-shell coverage is incomplete",
            )
        _closed(
            (
                self.physical_action_crosswalk_present,
                self.P_excluded_physical_alternate_union_deduplicated,
                self.alias_specific_fixed_outside_membership_present,
                self.outside_pair_cover_present,
                self.canonical_block_observation_authority,
                self.full_u_authority,
                self.action_invariance_authority,
                self.C_stream_open_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="within-shell lower-bound family",
        )
        if (
            self.rule != WITHIN_SHELL_RULE
            or self.authority_scope != WITHIN_SHELL_AUTHORITY
            or self.schema != WITHIN_SHELL_FAMILY_SCHEMA
        ):
            _refuse(
                V1BlockSupportWithinShellSolverRefusalCode.AUTHORITY_VIOLATION,
                "family semantics changed",
            )
        receipt_hash, integrity = _seal(self, prefix=_PREFIX + "family-")
        object.__setattr__(self, "receipt_hash", receipt_hash)
        object.__setattr__(self, "_integrity_seal", integrity)

    def assert_integrity(self) -> str:
        return _assert_seal(self, prefix=_PREFIX + "family-")


@dataclass(frozen=True, slots=True)
class V1BlockSupportWithinShellMarginLowerBoundObservation:
    """Narrow observation; canonical block certifiers must reject its type."""

    family_receipt_hash: str
    registration_hash: str
    block_index: int
    block_id: str
    block_context_axis_hash: str
    block_context_count: int
    within_shell_margin_lower_bound: float
    receipt_hash: str = field(init=False)
    _issuer_token: InitVar[object] = None
    projected_dual_downward_certified: bool = True
    complete_registered_within_shell_family: bool = True
    canonical_full_u_observation: bool = False
    accepted_by_canonical_block_certifier: bool = False
    full_u_authority: bool = False
    action_invariance_authority: bool = False
    C_stream_open_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = WITHIN_SHELL_AUTHORITY
    schema: str = WITHIN_SHELL_OBSERVATION_SCHEMA
    _integrity_seal: str = field(default="", repr=False, compare=False)

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _OBSERVATION_TOKEN:
            raise TypeError("within-shell observations are issuer-only")
        for name in (
            "family_receipt_hash",
            "registration_hash",
            "block_id",
            "block_context_axis_hash",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        object.__setattr__(
            self, "block_index", _integer(self.block_index, name="block index")
        )
        object.__setattr__(
            self,
            "block_context_count",
            _integer(
                self.block_context_count, name="block context count", positive=True
            ),
        )
        object.__setattr__(
            self,
            "within_shell_margin_lower_bound",
            _finite(
                self.within_shell_margin_lower_bound,
                name="within-shell margin lower bound",
            ),
        )
        if (
            self.projected_dual_downward_certified is not True
            or self.complete_registered_within_shell_family is not True
        ):
            _refuse(
                V1BlockSupportWithinShellSolverRefusalCode.INCOMPLETE_WITHIN_SHELL_FAMILY,
                "within-shell observation lacks its lower-bound proof",
            )
        _closed(
            (
                self.canonical_full_u_observation,
                self.accepted_by_canonical_block_certifier,
                self.full_u_authority,
                self.action_invariance_authority,
                self.C_stream_open_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="within-shell observation",
        )
        if (
            self.authority_scope != WITHIN_SHELL_AUTHORITY
            or self.schema != WITHIN_SHELL_OBSERVATION_SCHEMA
        ):
            _refuse(
                V1BlockSupportWithinShellSolverRefusalCode.AUTHORITY_VIOLATION,
                "within-shell observation semantics changed",
            )
        receipt_hash, integrity = _seal(self, prefix=_PREFIX + "observation-")
        object.__setattr__(self, "receipt_hash", receipt_hash)
        object.__setattr__(self, "_integrity_seal", integrity)

    def assert_integrity(self) -> str:
        return _assert_seal(self, prefix=_PREFIX + "observation-")


def _support_hash(handle: V1SignedWorkBlockSupportSourceDeviceHandle) -> str:
    handle.assert_live()
    return handle.receipt.receipt_hash


def _validate_sources(
    *,
    block_witness_source: V1BlockSupportWitnessSource,
    variable_source_registration: V1SignedWorkVariableShellSourceRegistration,
    registry: V1SignedWorkVariableShellActionCellRegistry,
    d_fixed_support: V1SignedWorkBlockSupportSourceDeviceHandle,
    residual_support: V1SignedWorkBlockSupportSourceDeviceHandle,
    preferred_action_binding: V1SignedWorkBlockSupportPreferredActionBinding,
    block_index: int,
    design: V1BlockSupportWithinShellCUDAExecutionDesign,
) -> tuple[Any, int, tuple[int, ...]]:
    if (
        type(block_witness_source) is not V1BlockSupportWitnessSource
        or type(variable_source_registration)
        is not V1SignedWorkVariableShellSourceRegistration
        or type(registry) is not V1SignedWorkVariableShellActionCellRegistry
        or type(d_fixed_support) is not V1SignedWorkBlockSupportSourceDeviceHandle
        or type(residual_support) is not V1SignedWorkBlockSupportSourceDeviceHandle
        or type(preferred_action_binding)
        is not V1SignedWorkBlockSupportPreferredActionBinding
        or type(design) is not V1BlockSupportWithinShellCUDAExecutionDesign
    ):
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.INVALID_TYPE,
            "within-shell solver inputs must have exact canonical types",
        )
    index = _integer(block_index, name="block index")
    try:
        registration = block_witness_source.registration
        registration.assert_integrity()
        block_witness_source.block_context_ids(index)
        preferred_action_binding.assert_live()
        d_fixed_support.assert_live()
        residual_support.assert_live()
        registry_handles = registry.assert_live_sources()
    except (
        V1SignedWorkBlockSupportRefusal,
        V1SignedWorkVariableShellActionSeparationRefusal,
    ) as error:
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.STALE_OR_MUTATED_SOURCE,
            f"live source custody refused: {error}",
        )
    d_receipt = d_fixed_support.receipt
    r_receipt = residual_support.receipt
    if (
        d_receipt.source_role is not V1SignedWorkBlockSupportSourceRole.D_FIXED
        or r_receipt.source_role
        is not V1SignedWorkBlockSupportSourceRole.RESIDUAL_BLOCK
    ):
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.STALE_OR_MUTATED_SOURCE,
            "support handles have the wrong D-fixed/residual roles",
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
            V1BlockSupportWithinShellSolverRefusalCode.STALE_OR_MUTATED_SOURCE,
            "D-fixed and residual supports do not share one source/action/chart",
        )
    registry_aliases = {
        (branch_source.branch.value, alias.resource_alias_id): (
            branch_source.registration.source_receipt_hash,
            branch_source.registration.source_replay_hash,
            alias.total_removed_count,
        )
        for branch_source in registry.branch_sources
        for alias in branch_source.resource_aliases
    }
    witness_aliases = {
        (alias.branch_id, alias.resource_alias_id): (
            alias.source_receipt_hash,
            alias.source_replay_hash,
            alias.cardinality,
        )
        for alias in registration.alias_bindings
    }
    required_branches = {
        V1SignedWorkVariableShellQBranch.SPECTRAL,
        V1SignedWorkVariableShellQBranch.DIAGONAL_L1,
    }
    observed_branches = {source.branch for source in registry.branch_sources}
    shell_indices = tuple(
        int(value) for value in variable_source_registration.shell_indices
    )
    if (
        not shell_indices
        or observed_branches != required_branches
        or not registry.receipt.complete_branch_axis
        or not registry.receipt.complete_action_cell_coverage
        or registry.receipt.total_action_cell_count <= 0
        or witness_aliases != registry_aliases
        or d_receipt.source_registration_hash
        != variable_source_registration.registration_hash
        or registry.receipt.model_identity_hash != d_receipt.model_identity_hash
        or registry.receipt.chart_identity_hash != d_receipt.chart_identity_hash
        or registry.receipt.current_mask_hash != d_receipt.current_mask_hash
        or registry.receipt.proposal_generation != d_receipt.proposal_generation
        or registry.receipt.route_axis_hash != d_receipt.route_axis_hash
        or registry.receipt.route_id_axis_hash != d_receipt.route_id_axis_hash
        or registry.receipt.registered_d_prefix_hash
        != d_receipt.registered_d_prefix_hash
        or registry.receipt.registered_d_w2_prefix_hash
        != d_receipt.registered_d_w2_prefix_hash
        or registry.receipt.native_fiber_axis_receipt_hash
        != d_receipt.native_fiber_axis_receipt_hash
        or registry.receipt.native_fiber_measure_hash
        != d_receipt.native_fiber_measure_hash
        or registration.chart_hash != d_receipt.chart_identity_hash
        or registration.current_mask_hash != d_receipt.current_mask_hash
        or registration.generation_index != d_receipt.proposal_generation
        or registration.route_axis_hash != d_receipt.route_axis_hash
        or registration.registered_d_prefix_hash != d_receipt.registered_d_prefix_hash
        or registration.registered_d_w2_source_hash
        != d_receipt.registered_d_w2_prefix_hash
        or registration.preferred_action_hash != d_receipt.preferred_action_hash
        or registration.preferred_action_cardinality
        != d_receipt.preferred_selected_count
        or r_receipt.block_ordinal != index
        or preferred_action_binding.binding_hash
        != d_receipt.preferred_action_binding_hash
        or preferred_action_binding.preferred_action_hash
        != d_receipt.preferred_action_hash
        or tuple(variable_source_registration.shell_indices)
        != tuple(registry.branch_sources[0].registration.shell_indices)
        or any(
            tuple(source.registration.shell_indices) != tuple(shell_indices)
            for source in registry.branch_sources
        )
    ):
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.STALE_OR_MUTATED_SOURCE,
            "support, witness, registry, shell, and preferred action identities diverge",
        )
    expected_registry_mode = (
        V1SignedWorkVariableShellActionSeparationMode.CUDA_PRODUCTION
        if design.execution_mode
        is V1SignedWorkBlockSupportExecutionMode.CUDA_PRODUCTION
        else V1SignedWorkVariableShellActionSeparationMode.EXACT_REFERENCE_TEST_ONLY
    )
    if (
        registry.design.execution_mode is not expected_registry_mode
        or d_receipt.execution_mode is not design.execution_mode
        or r_receipt.execution_mode is not design.execution_mode
    ):
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.INVALID_DESIGN,
            "solver, registry, and support-source execution modes differ",
        )
    devices: list[Any] = []
    try:
        with d_fixed_support.open_coefficient_lease() as lease:
            devices.append(lease.tensor.device)
        with residual_support.open_coefficient_lease() as lease:
            devices.append(lease.tensor.device)
        for handle in registry_handles:
            purpose = content_hash(
                {
                    "registry_receipt_hash": registry.receipt.receipt_hash,
                    "source_receipt_hash": handle.receipt.receipt_hash,
                    "operation": "within_shell_device_identity_probe",
                },
                prefix=_PREFIX + "device-probe-",
            )
            with handle.borrow_compact_tensors(borrow_purpose_hash=purpose) as lease:
                devices.append(lease.compact_tensors()[0].device)
    except (
        V1SignedWorkBlockSupportRefusal,
        V1SignedWorkVariableShellActionSeparationRefusal,
    ) as error:
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.STALE_OR_MUTATED_SOURCE,
            f"device source probe refused: {error}",
        )
    if not devices or any(device != devices[0] for device in devices):
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.STALE_OR_MUTATED_SOURCE,
            "all source families must share one physical device",
        )
    device = devices[0]
    if design.execution_mode is V1SignedWorkBlockSupportExecutionMode.CUDA_PRODUCTION:
        if device.type != "cuda":
            _refuse(
                V1BlockSupportWithinShellSolverRefusalCode.CUDA_REQUIRED,
                "production within-shell solver has no CPU fallback",
            )
        if torch is None:
            _refuse(
                V1BlockSupportWithinShellSolverRefusalCode.CUDA_REQUIRED,
                "PyTorch is unavailable",
            )
        warn_only = getattr(
            torch,
            "is_deterministic_algorithms_warn_only_enabled",
            lambda: False,
        )
        if not torch.are_deterministic_algorithms_enabled() or bool(warn_only()):
            _refuse(
                V1BlockSupportWithinShellSolverRefusalCode.DETERMINISM_REQUIRED,
                "production requires strict deterministic PyTorch algorithms",
            )
    elif device.type != "cpu":
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.INVALID_DESIGN,
            "reference-test mode is CPU-only",
        )
    return device, index, shell_indices


def build_v1_block_support_within_shell_hbm_forecast(
    *,
    block_witness_source: V1BlockSupportWitnessSource,
    variable_source_registration: V1SignedWorkVariableShellSourceRegistration,
    registry: V1SignedWorkVariableShellActionCellRegistry,
    d_fixed_support: V1SignedWorkBlockSupportSourceDeviceHandle,
    residual_support: V1SignedWorkBlockSupportSourceDeviceHandle,
    preferred_action_binding: V1SignedWorkBlockSupportPreferredActionBinding,
    block_index: int,
    block_capacities: Sequence[V1SignedWorkVariableShellActionCellBlockCapacity],
    design: V1BlockSupportWithinShellCUDAExecutionDesign,
    device_total_memory_bytes: int,
    device_free_memory_bytes: int,
    current_reserved_device_bytes: int,
    _admission_memory: tuple[int, int] | None = None,
) -> V1BlockSupportWithinShellHBMForecast:
    _device, _index, shell_indices = _validate_sources(
        block_witness_source=block_witness_source,
        variable_source_registration=variable_source_registration,
        registry=registry,
        d_fixed_support=d_fixed_support,
        residual_support=residual_support,
        preferred_action_binding=preferred_action_binding,
        block_index=block_index,
        design=design,
    )
    capacities = tuple(block_capacities)
    if not capacities or any(
        type(value) is not V1SignedWorkVariableShellActionCellBlockCapacity
        for value in capacities
    ):
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.INVALID_TYPE,
            "forecast block capacities must have exact types",
        )
    shell = len(shell_indices)
    flattened = tuple(
        offset for capacity in capacities for offset in capacity.shell_offsets
    )
    if (
        any(capacity.shell_size != shell for capacity in capacities)
        or tuple(sorted(flattened)) != tuple(range(shell))
        or len(set(flattened)) != shell
    ):
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.INVALID_DESIGN,
            "forecast block capacities must partition the registered shell",
        )
    total = _integer(
        device_total_memory_bytes, name="device total memory", positive=True
    )
    free = _integer(device_free_memory_bytes, name="device free memory")
    reserved = _integer(current_reserved_device_bytes, name="reserved device memory")
    if free > total or reserved > total:
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.INVALID_DESIGN,
            "device memory counters are incoherent",
        )
    route_count = d_fixed_support.receipt.route_count
    cell_count = registry.receipt.total_action_cell_count
    tile = min(cell_count, design.maximum_cell_tile_size)
    blocks = len(capacities)
    # Both source handles own one FP64 coefficient and one FP64 guard vector.
    # Production allocator counters normally already contain these live
    # buffers.  Reference counters are caller-registered, however, so any
    # shortfall is included explicitly instead of asserting co-liveness that
    # the supplied reservation cannot cover.
    live_source = 4 * route_count * 8
    live_source_unaccounted = max(0, live_source - reserved)
    packed_action_device = route_count
    # `_materialize_packed_preferred_action` unpacks at most one 1 MiB packed
    # host tile into an 8 Mi-route Boolean slab before its device copy.
    packed_action_staging = min(route_count, 8 * (1 << 20))
    route_action = packed_action_device + packed_action_staging
    shell_index_bytes = 8 * shell
    # Simultaneously live: preferred, d/r, objective + guard, combination
    # radius, resource, conversion operands, absolute-value operands, and
    # reduction scratch.  Thirty-two FP64 vectors is intentionally above the
    # observed expression graph so fusion choices cannot invalidate forecast.
    shell_fp64_vectors = 8 * (32 * shell + 64)
    shell_constraints = 8 * (blocks * shell + blocks)
    shell_static = shell_index_bytes + shell_fp64_vectors + shell_constraints
    # Conservative co-live tile states: z/zbar/znext, t variants, lambda,
    # block/order duals, residuals, reductions, and two temporary guards.
    tiled_state = 8 * tile * (32 * shell + 12 * blocks + 64) + tile * shell
    # Conservative allowance for deterministic CUDA BLAS/reduction workspaces
    # and allocator alignment.  CPU-reference forecasts retain the same bound.
    backend_workspace = 8 << 20
    workspace = tiled_state + backend_workspace
    incremental = live_source_unaccounted + route_action + shell_static + workspace
    projected = reserved + incremental
    ceiling = math.floor(total * design.hbm_ceiling_fraction)
    available, allocated = _admission_memory or (free, reserved)
    fits = allocated + incremental <= ceiling and incremental <= available
    return V1BlockSupportWithinShellHBMForecast(
        design_hash=design.design_hash,
        execution_mode=design.execution_mode,
        registry_receipt_hash=registry.receipt.receipt_hash,
        d_fixed_support_receipt_hash=_support_hash(d_fixed_support),
        residual_support_receipt_hash=_support_hash(residual_support),
        preferred_action_binding_hash=preferred_action_binding.binding_hash,
        route_count=route_count,
        shell_size=shell,
        registered_cell_count=cell_count,
        cell_tile_size=tile,
        block_count=blocks,
        live_source_bytes_in_current_reservation=live_source,
        live_source_unaccounted_bytes=live_source_unaccounted,
        packed_action_device_bytes=packed_action_device,
        packed_action_tile_staging_bytes=packed_action_staging,
        route_action_materialization_bytes=route_action,
        shell_index_bytes=shell_index_bytes,
        shell_fp64_vector_bytes=shell_fp64_vectors,
        shell_constraint_bytes=shell_constraints,
        shell_static_bytes=shell_static,
        tiled_state_bytes=tiled_state,
        backend_workspace_bytes=backend_workspace,
        tiled_solver_workspace_bytes=workspace,
        incremental_bytes=incremental,
        current_reserved_bytes=reserved,
        projected_bytes=projected,
        device_total_bytes=total,
        device_free_bytes=free,
        ceiling_bytes=ceiling,
        fits=fits,
        source_colive_accounted_in_current_reservation=(live_source_unaccounted == 0),
        production_live_device_query_bound=(
            design.execution_mode
            is V1SignedWorkBlockSupportExecutionMode.CUDA_PRODUCTION
        ),
        reference_registered_counter_mode=(
            design.execution_mode
            is V1SignedWorkBlockSupportExecutionMode.CPU_REFERENCE_TEST_ONLY
        ),
        _issuer_token=_FORECAST_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class _WithinShellCellProblem:
    branch: V1SignedWorkVariableShellQBranch
    resource_alias_id: str
    action_cell_ordinal: int
    action_cell_hash: str
    request_hash: str
    selected_offsets: tuple[int, ...]
    resource_floor: float


@dataclass(frozen=True, slots=True)
class _WithinShellProjectedDualCoordinatesReferenceTestOnly:
    """Exact-replay coordinates copied only by the CPU reference test seam.

    These values are not receipts and carry no runtime authority.  Production
    may never materialize them on the host.  Their sole purpose is to let the
    focused numerical tests replay the *same* rounded projected multipliers
    that generated a cell receipt, using :class:`Fraction` arithmetic over the
    exact binary64 operands.
    """

    cell_receipt_hash: str
    action_cell_hash: str
    objective: tuple[float, ...]
    resources: tuple[float, ...]
    resource_floor: float
    capacities: tuple[V1SignedWorkVariableShellActionCellBlockCapacity, ...]
    selected_offsets: tuple[int, ...]
    resource_multiplier: float
    block_multipliers: tuple[float, ...]
    order_multipliers: tuple[float, ...]

    def exact_projected_dual(self) -> Fraction:
        return _compact_threshold_dual_fraction_reference_test_only(
            objective=self.objective,
            resources=self.resources,
            resource_floor=self.resource_floor,
            capacities=self.capacities,
            selected_offsets=self.selected_offsets,
            resource_multiplier=self.resource_multiplier,
            block_multipliers=self.block_multipliers,
            order_multipliers=self.order_multipliers,
        )


def _enumerate_cell_problems(
    *,
    registry: V1SignedWorkVariableShellActionCellRegistry,
    shell_resource_coefficients: Sequence[float],
    literal_j_response_radius: float,
    block_capacities: Sequence[V1SignedWorkVariableShellActionCellBlockCapacity],
) -> tuple[
    tuple[_WithinShellCellProblem, ...],
    tuple[str, ...],
    tuple[tuple[str, str], ...],
]:
    problems: list[_WithinShellCellProblem] = []
    branches: list[str] = []
    aliases: list[tuple[str, str]] = []
    for branch_source in registry.branch_sources:
        branch = branch_source.branch
        branches.append(branch.value)
        for alias in branch_source.resource_aliases:
            aliases.append((branch.value, alias.resource_alias_id))
            cells = branch_source.closed_cells(alias.resource_alias_id)
            for cell in cells:
                request = build_v1_signed_work_variable_shell_restricted_cell_request(
                    registry=registry,
                    branch=branch,
                    resource_alias_id=alias.resource_alias_id,
                    action_cell_ordinal=cell.cell_ordinal,
                    shell_resource_coefficients=shell_resource_coefficients,
                    literal_j_response_radius=literal_j_response_radius,
                    block_capacities=block_capacities,
                )
                problems.append(
                    _WithinShellCellProblem(
                        branch=branch,
                        resource_alias_id=alias.resource_alias_id,
                        action_cell_ordinal=cell.cell_ordinal,
                        action_cell_hash=request.action_cell.action_cell_hash,
                        request_hash=request.request_hash,
                        selected_offsets=tuple(
                            int(value)
                            for value in request.action_cell.selected_shell_offsets
                        ),
                        resource_floor=float(request.registered_phase_a_resource_floor),
                    )
                )
    cell_hashes = tuple(problem.action_cell_hash for problem in problems)
    expected_iter_hashes = tuple(
        cell.action_cell_hash for cell in registry.iter_action_cells()
    )
    branch_axis = tuple(dict.fromkeys(branches))
    alias_axis = tuple(dict.fromkeys(aliases))
    if (
        not problems
        or len(problems) != registry.receipt.total_action_cell_count
        or len(cell_hashes) != len(set(cell_hashes))
        or cell_hashes != expected_iter_hashes
        or branch_axis != tuple(branch.value for branch in registry.receipt.branch_axis)
        or len(alias_axis)
        != sum(len(source.resource_aliases) for source in registry.branch_sources)
    ):
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.INCOMPLETE_WITHIN_SHELL_FAMILY,
            "live request enumeration differs from the registered cell/alias axis",
        )
    return tuple(problems), branch_axis, alias_axis


def _fraction_down(value: Fraction) -> float:
    center = float(value)
    if not math.isfinite(center):
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.NONFINITE_SOLVER,
            "exact enclosure endpoint exceeds binary64",
        )
    if Fraction.from_float(center) > value:
        center = math.nextafter(center, -math.inf)
    return center


def _mass_upper(center: float, operation_count: int) -> float:
    value = _nonnegative(center, name="absolute-mass reduction center")
    gamma = _gamma(operation_count)
    upper = value / (1.0 - gamma)
    if not math.isfinite(upper):
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.NONFINITE_SOLVER,
            "absolute-mass upper enclosure is nonfinite",
        )
    return _up(upper)


def _weighted_support_objective_and_error_terms(
    *,
    d_shell: Any,
    residual_shell: Any,
    d_weight_exact: Fraction,
    residual_weight_exact: Fraction,
) -> tuple[Any, Any]:
    """Construct the weighted objective and its positive rounding-error mass.

    The population weights are exact rationals.  Their binary64 conversion
    error is charged explicitly, while ``gamma(3)`` encloses the two products
    and one addition used for each objective coordinate.  The caller still
    applies :func:`_mass_upper` to the positive error-term construction and
    reduction before using the result as an outward radius.
    """

    if (
        torch is None
        or not isinstance(d_shell, torch.Tensor)
        or not isinstance(residual_shell, torch.Tensor)
        or d_shell.shape != residual_shell.shape
        or d_shell.dtype != torch.float64
        or residual_shell.dtype != torch.float64
        or d_shell.device != residual_shell.device
        or type(d_weight_exact) is not Fraction
        or type(residual_weight_exact) is not Fraction
    ):
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.INVALID_TYPE,
            "weighted support objective inputs must be aligned FP64 tensors and exact weights",
        )
    d_weight = float(d_weight_exact)
    residual_weight = float(residual_weight_exact)
    if not math.isfinite(d_weight) or not math.isfinite(residual_weight):
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.NONFINITE_SOLVER,
            "exact population weights are not representable in binary64",
        )
    d_weight_error = _fraction_upper(
        abs(Fraction.from_float(d_weight) - d_weight_exact)
    )
    residual_weight_error = _fraction_upper(
        abs(Fraction.from_float(residual_weight) - residual_weight_exact)
    )
    combine_gamma = _gamma(3)
    objective = d_weight * d_shell + residual_weight * residual_shell
    error_terms = torch.abs(d_shell) * (
        d_weight_error + combine_gamma * abs(d_weight)
    ) + torch.abs(residual_shell) * (
        residual_weight_error + combine_gamma * abs(residual_weight)
    )
    return objective, error_terms


def _dual_operation_count(*, shell: int, blocks: int) -> int:
    # Upper count includes dense B products (including structural zeros),
    # stationarity assembly, both box reductions, and final scalar assembly.
    return max(1, 10 * shell + 2 * shell * blocks + 4 * blocks + 32)


def _mass_operation_count(*, shell: int, blocks: int) -> int:
    return max(1, 12 * shell + 2 * shell * blocks + 6 * blocks + 32)


def _pdhg_operation_count(*, shell: int, blocks: int, iterations: int) -> int:
    per_iteration = 24 * shell + 4 * shell * blocks + 10 * blocks + 64
    return max(1, per_iteration * iterations)


def _operator_and_step(
    *,
    resources: Sequence[float],
    shell: int,
) -> tuple[float, float]:
    # B is a partition: exactly one unit coefficient per shell route.  The
    # compact threshold order contributes one row of squared norm two per
    # route.  Thus ||K||_F^2 = ||r||^2 + 3B exactly over registered floats.
    square = sum(
        (Fraction.from_float(float(value)) ** 2 for value in resources),
        Fraction(3 * shell),
    )
    square_upper = _fraction_upper(square)
    operator_upper = _up(math.sqrt(square_upper))
    step = _down(1.0 / operator_upper)
    if (
        not math.isfinite(operator_upper)
        or operator_upper <= 0.0
        or not math.isfinite(step)
        or step <= 0.0
    ):
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.NONFINITE_SOLVER,
            "derived strict Chambolle--Pock step is not representable",
        )
    return operator_upper, step


def _hard_cell_is_feasible(
    *,
    problem: _WithinShellCellProblem,
    resources: Sequence[float],
    capacities: Sequence[V1SignedWorkVariableShellActionCellBlockCapacity],
) -> bool:
    selected = set(problem.selected_offsets)
    resource = sum(
        (
            Fraction.from_float(float(value))
            for index, value in enumerate(resources)
            if index in selected
        ),
        Fraction(0),
    )
    if resource < Fraction.from_float(problem.resource_floor):
        return False
    return not any(
        sum(1 for index in capacity.shell_offsets if index in selected)
        > capacity.remaining_capacity
        for capacity in capacities
    )


def _compact_threshold_dual_fraction_reference_test_only(
    *,
    objective: Sequence[float],
    resources: Sequence[float],
    resource_floor: float,
    capacities: Sequence[V1SignedWorkVariableShellActionCellBlockCapacity],
    selected_offsets: Sequence[int],
    resource_multiplier: float,
    block_multipliers: Sequence[float],
    order_multipliers: Sequence[float],
) -> Fraction:
    """Exact binary64-operand replay of the compact-threshold dual signs."""

    shell = len(objective)
    if (
        len(resources) != shell
        or len(order_multipliers) != shell
        or len(block_multipliers) != len(capacities)
    ):
        raise ValueError("compact-threshold reference axes do not align")
    selected = set(int(value) for value in selected_offsets)
    lam = max(Fraction(0), Fraction.from_float(float(resource_multiplier)))
    mu = tuple(
        max(Fraction(0), Fraction.from_float(float(value)))
        for value in block_multipliers
    )
    q = tuple(
        max(Fraction(0), Fraction.from_float(float(value)))
        for value in order_multipliers
    )
    block_for_offset: dict[int, int] = {}
    for block_index, capacity in enumerate(capacities):
        for offset in capacity.shell_offsets:
            if offset in block_for_offset:
                raise ValueError("compact-threshold capacities overlap")
            block_for_offset[int(offset)] = block_index
    if tuple(sorted(block_for_offset)) != tuple(range(shell)):
        raise ValueError("compact-threshold capacities do not partition the shell")
    stationarity: list[Fraction] = []
    for index, (coefficient, resource) in enumerate(
        zip(objective, resources, strict=True)
    ):
        value = Fraction.from_float(float(coefficient))
        value -= lam * Fraction.from_float(float(resource))
        value += mu[block_for_offset[index]]
        value += -q[index] if index in selected else q[index]
        stationarity.append(value)
    threshold_stationarity = sum(
        (q[index] if index in selected else -q[index] for index in range(shell)),
        Fraction(0),
    )
    return (
        lam * Fraction.from_float(float(resource_floor))
        - sum(
            (
                multiplier * capacity.remaining_capacity
                for multiplier, capacity in zip(mu, capacities, strict=True)
            ),
            Fraction(0),
        )
        + sum((min(Fraction(0), value) for value in stationarity), Fraction(0))
        + min(Fraction(0), threshold_stationarity)
    )


def _solve_cell_tiles(
    *,
    objective: Any,
    resources: tuple[float, ...],
    capacities: tuple[V1SignedWorkVariableShellActionCellBlockCapacity, ...],
    problems: tuple[_WithinShellCellProblem, ...],
    design: V1BlockSupportWithinShellCUDAExecutionDesign,
    registry_receipt_hash: str,
    source_problem_hash: str,
    collect_projected_dual_coordinates_reference_test_only: bool = False,
) -> tuple[
    tuple[V1BlockSupportWithinShellCellDualLowerBoundReceipt, ...],
    int,
    tuple[_WithinShellProjectedDualCoordinatesReferenceTestOnly, ...],
]:
    if type(collect_projected_dual_coordinates_reference_test_only) is not bool:
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.INVALID_TYPE,
            "reference projected-dual collection flag must be exact bool",
        )
    if (
        collect_projected_dual_coordinates_reference_test_only
        and design.execution_mode
        is not V1SignedWorkBlockSupportExecutionMode.CPU_REFERENCE_TEST_ONLY
    ):
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.AUTHORITY_VIOLATION,
            "production may not materialize projected-dual coordinates on the host",
        )
    device = objective.device
    shell = len(resources)
    blocks = len(capacities)
    operator_upper, step_value = _operator_and_step(
        resources=resources,
        shell=shell,
    )
    resource = torch.tensor(resources, dtype=torch.float64, device=device)
    block_matrix = torch.zeros((blocks, shell), dtype=torch.float64, device=device)
    for row, capacity in enumerate(capacities):
        block_matrix[row, tuple(capacity.shell_offsets)] = 1.0
    block_caps = torch.tensor(
        tuple(float(capacity.remaining_capacity) for capacity in capacities),
        dtype=torch.float64,
        device=device,
    )
    step = torch.tensor(step_value, dtype=torch.float64, device=device)
    dual_ops = _dual_operation_count(shell=shell, blocks=blocks)
    mass_ops = _mass_operation_count(shell=shell, blocks=blocks)
    pdhg_ops = _pdhg_operation_count(
        shell=shell,
        blocks=blocks,
        iterations=design.pdhg_iteration_count,
    )
    receipts: list[V1BlockSupportWithinShellCellDualLowerBoundReceipt] = []
    exact_replay_witnesses: list[
        _WithinShellProjectedDualCoordinatesReferenceTestOnly
    ] = []
    objective_values: tuple[float, ...] | None = None
    cell_scale_d2h_bytes = 0
    if collect_projected_dual_coordinates_reference_test_only:
        objective_values = tuple(
            float(value) for value in objective.detach().cpu().tolist()
        )
        cell_scale_d2h_bytes += shell * 8
    tile_size = min(len(problems), design.maximum_cell_tile_size)
    zero = torch.zeros((), dtype=torch.float64, device=device)
    for start in range(0, len(problems), tile_size):
        tile = problems[start : start + tile_size]
        count = len(tile)
        selected = torch.zeros((count, shell), dtype=torch.bool, device=device)
        for row, problem in enumerate(tile):
            if problem.selected_offsets:
                selected[row, tuple(problem.selected_offsets)] = True
        floors = torch.tensor(
            tuple(problem.resource_floor for problem in tile),
            dtype=torch.float64,
            device=device,
        )
        z = torch.zeros((count, shell), dtype=torch.float64, device=device)
        z_bar = z.clone()
        threshold = torch.zeros((count,), dtype=torch.float64, device=device)
        threshold_bar = threshold.clone()
        lam = torch.zeros((count,), dtype=torch.float64, device=device)
        mu = torch.zeros((count, blocks), dtype=torch.float64, device=device)
        q = torch.zeros((count, shell), dtype=torch.float64, device=device)
        for _iteration in range(design.pdhg_iteration_count):
            resource_residual = floors - torch.mv(z_bar, resource)
            block_residual = torch.mm(z_bar, block_matrix.t()) - block_caps
            order_residual = torch.where(
                selected,
                threshold_bar.unsqueeze(1) - z_bar,
                z_bar - threshold_bar.unsqueeze(1),
            )
            lam = torch.clamp(lam + step * resource_residual, min=0.0)
            mu = torch.clamp(mu + step * block_residual, min=0.0)
            q = torch.clamp(q + step * order_residual, min=0.0)
            stationarity = (
                objective.unsqueeze(0)
                - lam.unsqueeze(1) * resource.unsqueeze(0)
                + torch.mm(mu, block_matrix)
                + torch.where(selected, -q, q)
            )
            threshold_gradient = torch.sum(torch.where(selected, q, -q), dim=1)
            z_next = torch.clamp(z - step * stationarity, min=0.0, max=1.0)
            threshold_next = torch.clamp(
                threshold - step * threshold_gradient,
                min=0.0,
                max=1.0,
            )
            z_bar = 2.0 * z_next - z
            threshold_bar = 2.0 * threshold_next - threshold
            z = z_next
            threshold = threshold_next

        # Reproject immediately before certification.  The resulting f64
        # values are treated as exact witness coordinates; no convergence
        # premise enters weak duality.
        lam = torch.clamp(lam, min=0.0)
        mu = torch.clamp(mu, min=0.0)
        q = torch.clamp(q, min=0.0)
        mu_stationarity = torch.mm(mu, block_matrix)
        order_stationarity = torch.where(selected, -q, q)
        stationarity = (
            objective.unsqueeze(0)
            - lam.unsqueeze(1) * resource.unsqueeze(0)
            + mu_stationarity
            + order_stationarity
        )
        threshold_stationarity = torch.sum(torch.where(selected, q, -q), dim=1)
        dual_center = (
            lam * floors
            - torch.sum(mu * block_caps.unsqueeze(0), dim=1)
            + torch.sum(torch.minimum(stationarity, zero), dim=1)
            + torch.minimum(threshold_stationarity, zero)
        )
        absolute_mass = (
            torch.abs(lam * floors)
            + torch.sum(torch.abs(mu * block_caps.unsqueeze(0)), dim=1)
            + torch.sum(torch.abs(objective), dim=0).expand(count)
            + torch.sum(
                torch.abs(lam).unsqueeze(1) * torch.abs(resource).unsqueeze(0),
                dim=1,
            )
            + torch.sum(torch.abs(mu_stationarity), dim=1)
            + 2.0 * torch.sum(torch.abs(q), dim=1)
            + torch.sum(torch.abs(stationarity), dim=1)
            + torch.abs(threshold_stationarity)
        )
        selected_fp64 = selected.to(dtype=torch.float64)
        primal_center = torch.sum(selected_fp64 * objective.unsqueeze(0), dim=1)
        primal_mass = torch.sum(
            selected_fp64 * torch.abs(objective).unsqueeze(0), dim=1
        )
        scalars = torch.stack(
            (dual_center, absolute_mass, primal_center, primal_mass), dim=1
        )
        scalar_rows = tuple(
            tuple(float(value) for value in row)
            for row in scalars.detach().cpu().tolist()
        )
        cell_scale_d2h_bytes += count * 4 * 8
        if collect_projected_dual_coordinates_reference_test_only:
            multiplier_rows: tuple[
                tuple[float, tuple[float, ...], tuple[float, ...]], ...
            ] = tuple(
                (
                    float(lam_value),
                    tuple(float(value) for value in mu_values),
                    tuple(float(value) for value in q_values),
                )
                for lam_value, mu_values, q_values in zip(
                    lam.detach().cpu().tolist(),
                    mu.detach().cpu().tolist(),
                    q.detach().cpu().tolist(),
                    strict=True,
                )
            )
            cell_scale_d2h_bytes += count * (1 + blocks + shell) * 8
        else:
            multiplier_rows = tuple((0.0, tuple(), tuple()) for _unused in range(count))
        for problem, scalar_row, multiplier_row in zip(
            tile,
            scalar_rows,
            multiplier_rows,
            strict=True,
        ):
            center, mass_center, primal_center_value, primal_mass_center = scalar_row
            if not all(math.isfinite(value) for value in scalar_row):
                _refuse(
                    V1BlockSupportWithinShellSolverRefusalCode.NONFINITE_SOLVER,
                    "cell projected-dual reduction became nonfinite",
                )
            mass_upper = _mass_upper(mass_center, mass_ops)
            radius = _up(_gamma(dual_ops) * mass_upper + _F64_MIN)
            lower = _down(center - radius)
            primal_upper: float | None = None
            gap_upper: float | None = None
            if _hard_cell_is_feasible(
                problem=problem,
                resources=resources,
                capacities=capacities,
            ):
                primal_radius = _up(
                    _gamma(max(1, 2 * max(1, len(problem.selected_offsets)) - 1))
                    * _mass_upper(primal_mass_center, max(1, shell))
                    + _F64_MIN
                )
                primal_upper = _up(primal_center_value + primal_radius)
                gap_upper = _up(max(0.0, primal_upper - lower))
                if lower > primal_upper:
                    _refuse(
                        V1BlockSupportWithinShellSolverRefusalCode.NUMERICAL_ENCLOSURE_REFUSAL,
                        "projected dual exceeded a verified feasible primal upper",
                    )
            selected_hash = content_hash(
                problem.selected_offsets,
                prefix=_PREFIX + "selected-offsets-",
            )
            problem_hash = content_hash(
                {
                    "source_problem_hash": source_problem_hash,
                    "request_hash": problem.request_hash,
                    "selected_shell_offset_hash": selected_hash,
                    "compact_threshold_order": True,
                    "Q_dropped": True,
                },
                prefix=_PREFIX + "cell-problem-",
            )
            receipt = V1BlockSupportWithinShellCellDualLowerBoundReceipt(
                registry_receipt_hash=registry_receipt_hash,
                request_hash=problem.request_hash,
                action_cell_hash=problem.action_cell_hash,
                branch=problem.branch,
                resource_alias_id=problem.resource_alias_id,
                action_cell_ordinal=problem.action_cell_ordinal,
                selected_shell_offset_hash=selected_hash,
                problem_hash=problem_hash,
                pdhg_iteration_count=design.pdhg_iteration_count,
                primal_dual_operator_frobenius_upper=operator_upper,
                primal_step_size=step_value,
                dual_step_size=step_value,
                projected_dual_center=center,
                projected_dual_roundoff_radius=radius,
                projected_dual_lower=lower,
                feasible_primal_objective_upper=primal_upper,
                diagnostic_duality_gap_upper=gap_upper,
                projected_dual_operation_count=dual_ops,
                pdhg_rounded_operation_upper_bound=pdhg_ops,
                _issuer_token=_CELL_TOKEN,
            )
            receipts.append(receipt)
            if collect_projected_dual_coordinates_reference_test_only:
                if objective_values is None:  # pragma: no cover - local invariant.
                    raise AssertionError("reference objective values were not copied")
                lam_value, mu_values, q_values = multiplier_row
                exact_replay_witnesses.append(
                    _WithinShellProjectedDualCoordinatesReferenceTestOnly(
                        cell_receipt_hash=receipt.receipt_hash,
                        action_cell_hash=problem.action_cell_hash,
                        objective=objective_values,
                        resources=resources,
                        resource_floor=problem.resource_floor,
                        capacities=capacities,
                        selected_offsets=problem.selected_offsets,
                        resource_multiplier=lam_value,
                        block_multipliers=mu_values,
                        order_multipliers=q_values,
                    )
                )
    if len(receipts) != len(problems):
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.INCOMPLETE_WITHIN_SHELL_FAMILY,
            "tiled solver omitted registered cells",
        )
    if len(exact_replay_witnesses) not in (0, len(problems)):
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.INCOMPLETE_WITHIN_SHELL_FAMILY,
            "reference projected-dual collection omitted registered cells",
        )
    return (
        tuple(receipts),
        cell_scale_d2h_bytes,
        tuple(exact_replay_witnesses),
    )


def issue_v1_block_support_within_shell_lower_bound_device(
    *,
    block_witness_source: V1BlockSupportWitnessSource,
    variable_source_registration: V1SignedWorkVariableShellSourceRegistration,
    registry: V1SignedWorkVariableShellActionCellRegistry,
    d_fixed_support: V1SignedWorkBlockSupportSourceDeviceHandle,
    residual_support: V1SignedWorkBlockSupportSourceDeviceHandle,
    preferred_action_binding: V1SignedWorkBlockSupportPreferredActionBinding,
    block_index: int,
    shell_resource_coefficients: Sequence[float],
    literal_j_response_radius: float,
    block_capacities: Sequence[V1SignedWorkVariableShellActionCellBlockCapacity],
    design: V1BlockSupportWithinShellCUDAExecutionDesign,
    reference_device_total_memory_bytes: int = 1 << 40,
    reference_device_free_memory_bytes: int = 1 << 40,
    reference_current_reserved_device_bytes: int = 0,
    _projected_dual_coordinates_reference_test_only: (
        list[_WithinShellProjectedDualCoordinatesReferenceTestOnly] | None
    ) = None,
) -> tuple[
    V1BlockSupportWithinShellMarginLowerBoundObservation,
    V1BlockSupportWithinShellLowerBoundFamilyReceipt,
    V1BlockSupportWithinShellCUDAExecutionReceipt,
    tuple[V1BlockSupportWithinShellCellDualLowerBoundReceipt, ...],
]:
    """Issue a source-bound lower family for all registered within-shell cells.

    This function intentionally cannot issue
    ``V1BlockSupportMarginLowerBoundObservation``.  The complete outside-pair
    and physical-action crosswalk providers do not yet exist.
    """

    if torch is None:
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.CUDA_REQUIRED,
            "PyTorch is unavailable",
        )
    if _projected_dual_coordinates_reference_test_only is not None and (
        type(_projected_dual_coordinates_reference_test_only) is not list
        or _projected_dual_coordinates_reference_test_only
    ):
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.INVALID_TYPE,
            "reference projected-dual witness sink must be one empty exact list",
        )
    device, index, shell_indices = _validate_sources(
        block_witness_source=block_witness_source,
        variable_source_registration=variable_source_registration,
        registry=registry,
        d_fixed_support=d_fixed_support,
        residual_support=residual_support,
        preferred_action_binding=preferred_action_binding,
        block_index=block_index,
        design=design,
    )
    resources = tuple(
        _nonnegative(value, name="shell resource coefficient")
        for value in shell_resource_coefficients
    )
    literal_radius = _nonnegative(
        literal_j_response_radius,
        name="literal-J response radius",
    )
    capacities = tuple(block_capacities)
    if (
        len(resources) != len(shell_indices)
        or not capacities
        or any(
            type(value) is not V1SignedWorkVariableShellActionCellBlockCapacity
            for value in capacities
        )
    ):
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.INVALID_DESIGN,
            "resource/capacity inputs do not align with the registered shell",
        )
    flattened = tuple(
        offset for capacity in capacities for offset in capacity.shell_offsets
    )
    if (
        any(capacity.shell_size != len(resources) for capacity in capacities)
        or tuple(sorted(flattened)) != tuple(range(len(resources)))
        or len(set(flattened)) != len(resources)
    ):
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.INVALID_DESIGN,
            "block capacities must partition the registered shell exactly once",
        )
    if design.execution_mode is V1SignedWorkBlockSupportExecutionMode.CUDA_PRODUCTION:
        free_memory, total_memory = torch.cuda.mem_get_info(device)
        admission_memory = (
            cuda_reusable_memory_bytes(device, framework=torch, free_bytes=int(free_memory)),
            int(torch.cuda.memory_allocated(device)),
        )
        reserved_memory = torch.cuda.memory_reserved(device)
    else:
        admission_memory = None
        total_memory = _integer(
            reference_device_total_memory_bytes,
            name="reference total memory",
            positive=True,
        )
        free_memory = _integer(
            reference_device_free_memory_bytes,
            name="reference free memory",
        )
        reserved_memory = _integer(
            reference_current_reserved_device_bytes,
            name="reference reserved memory",
        )
    forecast = build_v1_block_support_within_shell_hbm_forecast(
        block_witness_source=block_witness_source,
        variable_source_registration=variable_source_registration,
        registry=registry,
        d_fixed_support=d_fixed_support,
        residual_support=residual_support,
        preferred_action_binding=preferred_action_binding,
        block_index=index,
        block_capacities=capacities,
        design=design,
        device_total_memory_bytes=int(total_memory),
        device_free_memory_bytes=int(free_memory),
        current_reserved_device_bytes=int(reserved_memory),
        _admission_memory=admission_memory,
    )
    if not forecast.fits:
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.HBM_FORECAST_EXCEEDED,
            "pre-contraction HBM forecast exceeds the registered ceiling/free memory",
        )
    problems, branch_axis, alias_axis = _enumerate_cell_problems(
        registry=registry,
        shell_resource_coefficients=resources,
        literal_j_response_radius=literal_radius,
        block_capacities=capacities,
    )
    registration = block_witness_source.registration
    d_exact = Fraction(
        registration.d_weight_numerator,
        registration.population_weight_denominator,
    )
    r_exact = Fraction(
        registration.residual_weight_numerator,
        registration.population_weight_denominator,
    )
    shell_index_tensor = torch.tensor(
        shell_indices,
        dtype=torch.int64,
        device=device,
    )
    preferred_route, packed_host_bytes_read = preferred_action_binding.materialize(
        device=device,
        tile_bytes=1 << 20,
    )
    preferred_shell = torch.index_select(
        preferred_route,
        0,
        shell_index_tensor,
    ).to(dtype=torch.float64)
    del preferred_route
    source_problem_hash = content_hash(
        {
            "d_fixed_support_receipt_hash": _support_hash(d_fixed_support),
            "residual_support_receipt_hash": _support_hash(residual_support),
            "preferred_action_binding_hash": preferred_action_binding.binding_hash,
            "registry_receipt_hash": registry.receipt.receipt_hash,
            "resource_coefficients": resources,
            "literal_J_radius_registered_but_Q_dropped": literal_radius,
            "block_capacity_hash_axis": tuple(
                capacity.capacity_hash for capacity in capacities
            ),
            "dual_rule": WITHIN_SHELL_DUAL_RULE,
        },
        prefix=_PREFIX + "source-problem-",
    )
    objective_scalar_d2h_bytes = 0
    try:
        with d_fixed_support.open_coefficient_lease() as d_lease, residual_support.open_coefficient_lease() as r_lease:
            d_coefficients = d_lease.tensor
            r_coefficients = r_lease.tensor
            d_shell = torch.index_select(d_coefficients, 0, shell_index_tensor)
            r_shell = torch.index_select(r_coefficients, 0, shell_index_tensor)
            objective, combination_error_terms = (
                _weighted_support_objective_and_error_terms(
                    d_shell=d_shell,
                    residual_shell=r_shell,
                    d_weight_exact=d_exact,
                    residual_weight_exact=r_exact,
                )
            )
            objective_guard = objective.detach().clone()
            combination_error_center = torch.sum(combination_error_terms)
            preferred_center_tensor = torch.dot(objective, preferred_shell)
            preferred_mass_tensor = torch.dot(
                torch.abs(objective), torch.abs(preferred_shell)
            )
            objective_scalars = torch.stack(
                (
                    combination_error_center,
                    preferred_center_tensor,
                    preferred_mass_tensor,
                )
            )
            combination_center, preferred_center, preferred_mass = tuple(
                float(value) for value in objective_scalars.detach().cpu().tolist()
            )
            objective_scalar_d2h_bytes += 3 * 8
            if not all(
                math.isfinite(value)
                for value in (combination_center, preferred_center, preferred_mass)
            ):
                _refuse(
                    V1BlockSupportWithinShellSolverRefusalCode.NONFINITE_SOLVER,
                    "objective construction became nonfinite",
                )
            combination_radius = _mass_upper(
                combination_center,
                max(1, 8 * len(resources)),
            )
            preferred_mass_upper = _mass_upper(
                preferred_mass,
                max(1, 2 * len(resources)),
            )
            preferred_radius = _up(
                _gamma(max(1, 2 * len(resources) - 1)) * preferred_mass_upper + _F64_MIN
            )
            cell_receipts, cell_d2h_bytes, exact_replay_witnesses = _solve_cell_tiles(
                objective=objective,
                resources=resources,
                capacities=capacities,
                problems=problems,
                design=design,
                registry_receipt_hash=registry.receipt.receipt_hash,
                source_problem_hash=source_problem_hash,
                collect_projected_dual_coordinates_reference_test_only=(
                    _projected_dual_coordinates_reference_test_only is not None
                ),
            )
            if not bool(torch.equal(objective, objective_guard)):
                _refuse(
                    V1BlockSupportWithinShellSolverRefusalCode.STALE_OR_MUTATED_SOURCE,
                    "shell objective mutated while the source leases were active",
                )
    except (
        V1SignedWorkBlockSupportRefusal,
        V1SignedWorkVariableShellActionSeparationRefusal,
    ) as error:
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.STALE_OR_MUTATED_SOURCE,
            f"source lease refused during solve: {error}",
        )
    registry.assert_live_sources()
    minimum = min(cell_receipts, key=lambda value: value.projected_dual_lower)
    center_fraction = Fraction.from_float(
        minimum.projected_dual_center
    ) - Fraction.from_float(preferred_center)
    target_lower_fraction = (
        Fraction.from_float(minimum.projected_dual_lower)
        - Fraction.from_float(_up(preferred_center + preferred_radius))
        - Fraction.from_float(combination_radius)
    )
    slope_center = float(center_fraction)
    if not math.isfinite(slope_center):
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.NONFINITE_SOLVER,
            "within-shell slope center became nonfinite",
        )
    target_lower = _fraction_down(target_lower_fraction)
    slope_radius = _fraction_upper(
        max(
            Fraction(0),
            Fraction.from_float(slope_center) - Fraction.from_float(target_lower),
        )
    )
    slope_lower = _down(slope_center - slope_radius)
    cell_hash_axis = tuple(problem.action_cell_hash for problem in problems)
    cell_receipt_hash_axis = tuple(
        receipt.assert_integrity() for receipt in cell_receipts
    )
    total_pdhg_operations = sum(
        receipt.pdhg_rounded_operation_upper_bound for receipt in cell_receipts
    )
    total_dual_operations = sum(
        receipt.projected_dual_operation_count for receipt in cell_receipts
    )
    execution = V1BlockSupportWithinShellCUDAExecutionReceipt(
        design_hash=design.design_hash,
        forecast_receipt_hash=forecast.assert_integrity(),
        block_support_registration_hash=registration.registration_hash,
        variable_shell_source_registration_hash=(
            variable_source_registration.registration_hash
        ),
        registry_receipt_hash=registry.receipt.receipt_hash,
        d_fixed_support_receipt_hash=_support_hash(d_fixed_support),
        residual_support_receipt_hash=_support_hash(residual_support),
        preferred_action_binding_hash=preferred_action_binding.binding_hash,
        preferred_action_hash=preferred_action_binding.preferred_action_hash,
        block_index=index,
        route_count=d_fixed_support.receipt.route_count,
        device_type=device.type,
        device_index=0 if device.index is None else int(device.index),
        registered_cell_count=registry.receipt.total_action_cell_count,
        solved_cell_count=len(cell_receipts),
        cell_tile_size=min(len(cell_receipts), design.maximum_cell_tile_size),
        tile_count=(len(cell_receipts) + design.maximum_cell_tile_size - 1)
        // design.maximum_cell_tile_size,
        branch_axis=branch_axis,
        resource_alias_axis=alias_axis,
        action_cell_hash_axis=cell_hash_axis,
        cell_receipt_hash_axis=cell_receipt_hash_axis,
        pdhg_iteration_count_per_cell=design.pdhg_iteration_count,
        pdhg_rounded_operation_upper_bound=total_pdhg_operations,
        projected_dual_operation_count=total_dual_operations,
        objective_combination_operation_count=max(1, 8 * len(resources)),
        cell_scale_d2h_bytes=cell_d2h_bytes + objective_scalar_d2h_bytes,
        packed_host_bytes_read=int(packed_host_bytes_read),
        route_scale_h2d_payload_bytes=d_fixed_support.receipt.route_count,
        maximum_h2d_staging_bytes=min(
            d_fixed_support.receipt.route_count,
            8 * (1 << 20),
        ),
        CPU_reference_test_mode=(
            design.execution_mode
            is V1SignedWorkBlockSupportExecutionMode.CPU_REFERENCE_TEST_ONLY
        ),
        production_CUDA_execution=(
            design.execution_mode
            is V1SignedWorkBlockSupportExecutionMode.CUDA_PRODUCTION
        ),
        _issuer_token=_EXECUTION_TOKEN,
    )
    block_context_ids = block_witness_source.block_context_ids(index)
    block_id = block_witness_source.block_ids[index]
    block_context_axis_hash = content_hash(
        block_context_ids,
        prefix=_PREFIX + "block-context-axis-",
    )
    family = V1BlockSupportWithinShellLowerBoundFamilyReceipt(
        execution_receipt_hash=execution.assert_integrity(),
        block_support_registration_hash=registration.registration_hash,
        registry_receipt_hash=registry.receipt.receipt_hash,
        block_index=index,
        block_id=block_id,
        block_context_axis_hash=block_context_axis_hash,
        block_context_count=len(block_context_ids),
        preferred_action_hash=preferred_action_binding.preferred_action_hash,
        preferred_action_binding_hash=preferred_action_binding.binding_hash,
        branch_axis=branch_axis,
        resource_alias_axis=alias_axis,
        action_cell_hash_axis=cell_hash_axis,
        cell_receipt_hash_axis=cell_receipt_hash_axis,
        minimum_cell_hash=minimum.action_cell_hash,
        weighted_within_shell_slope_center=slope_center,
        solver_downward_numerical_radius=slope_radius,
        weighted_within_shell_slope_lower=slope_lower,
        objective_combination_l1_radius_upper=combination_radius,
        preferred_dot_roundoff_radius_upper=preferred_radius,
        _issuer_token=_FAMILY_TOKEN,
    )
    weighted_deficit = d_exact * Fraction.from_float(
        registration.d_preferred_plane_deficit_upper_outward
    ) + r_exact * Fraction.from_float(
        residual_support.receipt.preferred_plane_deficit_upper
    )
    within_shell_margin = _fraction_down(
        Fraction.from_float(family.weighted_within_shell_slope_lower) - weighted_deficit
    )
    observation = V1BlockSupportWithinShellMarginLowerBoundObservation(
        family_receipt_hash=family.assert_integrity(),
        registration_hash=registration.registration_hash,
        block_index=index,
        block_id=block_id,
        block_context_axis_hash=block_context_axis_hash,
        block_context_count=len(block_context_ids),
        within_shell_margin_lower_bound=within_shell_margin,
        _issuer_token=_OBSERVATION_TOKEN,
    )
    if _projected_dual_coordinates_reference_test_only is not None:
        _projected_dual_coordinates_reference_test_only.extend(exact_replay_witnesses)
    return observation, family, execution, cell_receipts


def issue_v1_block_support_canonical_observation_from_within_shell(
    *,
    family: V1BlockSupportWithinShellLowerBoundFamilyReceipt,
    outside_provider_receipt: object | None = None,
) -> NoReturn:
    """Fail closed until a typed full-axis provider and join are implemented."""

    if type(family) is not V1BlockSupportWithinShellLowerBoundFamilyReceipt:
        _refuse(
            V1BlockSupportWithinShellSolverRefusalCode.INVALID_TYPE,
            "canonical promotion requires the exact within-shell family",
        )
    family.assert_integrity()
    del outside_provider_receipt
    _refuse(
        V1BlockSupportWithinShellSolverRefusalCode.OUTSIDE_PROVIDER_REQUIRED,
        "no live alias-specific fixed-outside/physical-action/outside-pair provider is landed",
    )


__all__ = [
    "WITHIN_SHELL_AUTHORITY",
    "WITHIN_SHELL_BACKEND",
    "WITHIN_SHELL_DUAL_RULE",
    "WITHIN_SHELL_RULE",
    "V1BlockSupportWithinShellCUDAExecutionDesign",
    "V1BlockSupportWithinShellCUDAExecutionReceipt",
    "V1BlockSupportWithinShellCellDualLowerBoundReceipt",
    "V1BlockSupportWithinShellHBMForecast",
    "V1BlockSupportWithinShellLowerBoundFamilyReceipt",
    "V1BlockSupportWithinShellMarginLowerBoundObservation",
    "V1BlockSupportWithinShellSolverRefusal",
    "V1BlockSupportWithinShellSolverRefusalCode",
    "build_v1_block_support_within_shell_hbm_forecast",
    "issue_v1_block_support_canonical_observation_from_within_shell",
    "issue_v1_block_support_within_shell_lower_bound_device",
]
