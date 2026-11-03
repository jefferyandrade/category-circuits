"""Authority-inert robust Phase-A endpoint/action-image premise.

For each source-issued physical action this module solves the compact affine
shell problem twice.  The population Loewner upper matrix gives a guaranteed
feasible resource lower endpoint; the (possibly indefinite) Loewner lower
matrix gives a permissive resource upper endpoint and a genuine infeasibility
test.  The historical spectral and diagonal-L1 FD branches remain distinct
aliases of their packed physical action, but their residual terms are never
added to the literal-J population geometry.

The resource objective and one-live-survivor capacities are derived from the
exact relaxed-KKT problem and packed parent-to-child deltas.  Registered
resource cells and packed-action quotients come through the issuer-only exact
action-pair/alias-map source-graph join; a raw alias map is insufficient even
when all of its public hashes agree.  Caller-authored cell records are
accepted only by an explicit small-domain private-reference path and can
never acquire production authority.

This owner intentionally stops short of the outside-no-entry/fixed-point
theorem.  A closed result is therefore a local endpoint/action-image premise,
not D stability and never C, K, O, commit, terminal, performance, or Pareto
authority.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
import math
from types import MappingProxyType
from typing import Final, Mapping, NoReturn, Sequence

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask

from .modern_v1_literal_j_joint_affine_shell_loewner_stream_bridge import (
    V1LiteralJJointAffineShellLoewnerStreamResult,
)
from .modern_v1_phase_a_bounded_shell_global_quadratic import (
    V1PhaseABoundedShellGlobalQuadraticReceipt,
    V1PhaseABoundedShellGlobalQuadraticSearchDesign,
    V1PhaseAShellBlockCapacityConstraint,
    issue_v1_phase_a_bounded_shell_global_quadratic_problem,
    solve_v1_phase_a_bounded_shell_global_quadratic,
)
from .modern_v1_phase_a_global_q_support_plane_no_entry import (
    V1PhaseAOutsideShellFixedPointReferenceReceipt,
    V1PhaseAOutsideShellFixedPointStatus,
)
from .modern_v1_relaxed_kkt import (
    V1RelaxedKKTBranch,
    V1RelaxedKKTProblem,
)
from .modern_v1_sampled_v7_branch_action_pair import (
    V1SampledV7BranchActionPair,
)
from .modern_v1_sampled_v7_action_pair_alias_source_join import (
    V1SampledV7ActionPairAliasSourceJoinReceipt,
)
from .modern_v1_sampled_v7_d_source_image_binding import (
    V1SampledV7DSourceImageBindingReceipt,
)
from .modern_v1_signed_work_phase_a_alias_safe_margin_cuda import (
    V1PhaseACompleteRegisteredAliasMapReceipt,
)
from .modern_v1_signed_work_variable_shell_action_separation_cuda import (
    V1SignedWorkVariableShellActionCellBlockCapacity,
)

try:  # pragma: no cover - runtime image dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


SAMPLED_V7_PHASE_A_ROBUST_SHELL_DECISION_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_phase_A_robust_shell_decision_premise_v3"
)
SAMPLED_V7_PHASE_A_ROBUST_SHELL_DECISION_ALGORITHM: Final = (
    "population_literal_J_Loewner_upper_feasible_lower_and_indefinite_lower_"
    "permissive_upper_qmin_then_exact_source_joined_registered_cell_packed_"
    "action_quotient_v2"
)
SAMPLED_V7_PHASE_A_ROBUST_SHELL_DECISION_AUTHORITY: Final = (
    "local_robust_endpoint_and_registered_action_image_premise_only_outside_"
    "no_entry_and_fixed_point_unproved_no_D_C_K_O_commit_terminal_performance_"
    "or_pareto_authority_v1"
)
_PRIVATE_CELL_SCHEMA: Final = (
    "partition_free_v1_phase_A_private_reference_resource_cell_v1"
)
_PRIVATE_QUOTIENT_SCHEMA: Final = (
    "partition_free_v1_phase_A_private_reference_action_quotient_v1"
)
_PREFIX: Final = "partition-free-v1-sampled-v7-phase-a-robust-shell-decision-"
_RECEIPT_TOKEN: Final = object()


class V1SampledV7PhaseARobustShellDecisionRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_SOURCE = "stale_source"
    ACTION_AXIS_MISMATCH = "action_axis_mismatch"
    PACKED_ACTION_MISMATCH = "packed_action_mismatch"
    LOEWNER_AXIS_MISMATCH = "loewner_axis_mismatch"
    EXACT_SOURCE_JOIN_REQUIRED = "exact_source_join_required"
    SOURCE_RESOURCE_ADAPTER_REQUIRED = "source_resource_adapter_required"
    BLOCK_CAPACITY_ILLEGAL = "block_capacity_illegal"
    NUMERICAL_REFUSAL = "numerical_refusal"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SampledV7PhaseARobustShellDecisionRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SampledV7PhaseARobustShellDecisionRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledV7PhaseARobustShellDecisionRefusalCode:
            raise TypeError("robust-shell refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledV7PhaseARobustShellDecisionRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledV7PhaseARobustShellDecisionRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledV7PhaseARobustShellDecisionRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty exact string",
        )
    return value


def _finite(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        _refuse(
            V1SampledV7PhaseARobustShellDecisionRefusalCode.NUMERICAL_REFUSAL,
            f"{name} must be one finite binary64 scalar",
        )
    return value


def _nonnegative(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if result < 0.0:
        _refuse(
            V1SampledV7PhaseARobustShellDecisionRefusalCode.NUMERICAL_REFUSAL,
            f"{name} must be nonnegative",
        )
    return result


def _payload_hash(payload: Mapping[str, object], *, role: str) -> str:
    return content_hash(payload, prefix=_PREFIX + role + "-")


class V1SampledV7PhaseARobustPairStatus(str, Enum):
    CLOSED_ENDPOINT = "closed_endpoint"
    CERTIFIED_INFEASIBLE = "certified_infeasible"
    UNRESOLVED_FEASIBILITY = "unresolved_feasibility"
    UNRESOLVED_ENGINEERING = "unresolved_engineering"
    UNRESOLVED_ACTION_IMAGE = "unresolved_action_image"


class V1SampledV7PhaseARobustDecisionStatus(str, Enum):
    STABLE_PHYSICAL_ACTION_PREMISE = "stable_physical_action_premise"
    STABLE_NO_ACTION_PREMISE = "stable_no_action_premise"
    UNRESOLVED_ENDPOINT_INTERVAL = "unresolved_endpoint_interval"
    UNRESOLVED_NONDOMINATED_ACTION_IMAGE = "unresolved_nondominated_action_image"


@dataclass(frozen=True, slots=True)
class V1SampledV7PhaseAPrivateResourceCell:
    """Explicitly non-production resource cell for small-domain tests only."""

    resource_cell_hash: str
    cell_lower_bound: float
    cell_upper_bound: float
    private_reference_only: bool = True
    production_authority: bool = False
    schema: str = _PRIVATE_CELL_SCHEMA

    def __post_init__(self) -> None:
        _identity(self.resource_cell_hash, name="resource_cell_hash")
        lower = _finite(self.cell_lower_bound, name="cell_lower_bound")
        upper = _finite(self.cell_upper_bound, name="cell_upper_bound")
        if lower > upper:
            _refuse(
                V1SampledV7PhaseARobustShellDecisionRefusalCode.NUMERICAL_REFUSAL,
                "private resource-cell bounds are reversed",
            )
        if (
            self.private_reference_only is not True
            or self.production_authority is not False
            or self.schema != _PRIVATE_CELL_SCHEMA
        ):
            _refuse(
                V1SampledV7PhaseARobustShellDecisionRefusalCode.AUTHORITY_VIOLATION,
                "private resource cell acquired production authority",
            )

    @property
    def cell_hash(self) -> str:
        return _payload_hash(
            {item.name: getattr(self, item.name) for item in fields(self)},
            role="private-cell",
        )


@dataclass(frozen=True, slots=True)
class V1SampledV7PhaseAPrivateActionQuotient:
    """Explicitly non-production cell/branch packed-action image."""

    resource_cell_hash: str
    branch: V1RelaxedKKTBranch
    physical_action_identity_hashes: tuple[str, ...]
    private_reference_only: bool = True
    production_authority: bool = False
    schema: str = _PRIVATE_QUOTIENT_SCHEMA

    def __post_init__(self) -> None:
        _identity(self.resource_cell_hash, name="resource_cell_hash")
        if type(self.branch) is not V1RelaxedKKTBranch:
            _refuse(
                V1SampledV7PhaseARobustShellDecisionRefusalCode.INVALID_TYPE,
                "private quotient branch must be exact",
            )
        axis = tuple(
            _identity(value, name="physical action identity")
            for value in self.physical_action_identity_hashes
        )
        if not axis or len(axis) != len(set(axis)) or axis != tuple(sorted(axis)):
            _refuse(
                V1SampledV7PhaseARobustShellDecisionRefusalCode.INVALID_TYPE,
                "private quotient action axis must be nonempty, sorted, and unique",
            )
        object.__setattr__(self, "physical_action_identity_hashes", axis)
        if (
            self.private_reference_only is not True
            or self.production_authority is not False
            or self.schema != _PRIVATE_QUOTIENT_SCHEMA
        ):
            _refuse(
                V1SampledV7PhaseARobustShellDecisionRefusalCode.AUTHORITY_VIOLATION,
                "private action quotient acquired production authority",
            )

    @property
    def quotient_hash(self) -> str:
        return _payload_hash(
            {item.name: getattr(self, item.name) for item in fields(self)},
            role="private-quotient",
        )


@dataclass(frozen=True, slots=True)
class _ResourceCell:
    resource_cell_hash: str
    lower: float
    upper: float
    source_hash: str


@dataclass(frozen=True, slots=True)
class _ActionQuotient:
    resource_cell_hash: str
    branch: V1RelaxedKKTBranch
    physical_action_identity_hashes: tuple[str, ...]
    source_hash: str


@dataclass(frozen=True, slots=True)
class _ActionResourceProblem:
    physical_action_ordinal: int
    physical_action_identity_hash: str
    packed_action_hash: str
    outside_resource_lower_bound: float
    outside_resource_upper_bound: float
    shell_resource_coefficients: tuple[float, ...]
    block_capacities: tuple[V1SignedWorkVariableShellActionCellBlockCapacity, ...]
    source_problem_hash: str
    resource_problem_hash: str


@dataclass(frozen=True, slots=True)
class V1SampledV7PhaseARobustBranchActionResult:
    physical_action_ordinal: int
    physical_action_identity_hash: str
    packed_action_hash: str
    branch: V1RelaxedKKTBranch
    branch_alias_hash: str
    loewner_action_ordinal: int
    loewner_small_axis_ids: tuple[str, ...]
    source_resource_problem_hash: str
    block_capacity_hashes: tuple[str, ...]
    upper_loewner_problem_hash: str
    upper_loewner_bounded_receipt_hash: str
    lower_loewner_problem_hash: str
    lower_loewner_bounded_receipt_hash: str
    guaranteed_feasible_resource_lower_bound: float | None
    permissive_resource_upper_bound: float | None
    lower_loewner_q_min_lower_bound: float
    lower_loewner_q_min_upper_bound: float
    lower_loewner_certifies_true_q_infeasible: bool
    all_required_binary64_bounds_closed: bool
    engineering_budget_exhausted: bool
    crossed_resource_cell_hashes: tuple[str, ...]
    crossed_action_quotient_hashes: tuple[str, ...]
    crossed_physical_action_identity_hashes: tuple[str, ...]
    registered_endpoint_cell_cover_closed: bool
    registered_packed_action_image_closed: bool
    status: V1SampledV7PhaseARobustPairStatus
    nondominated: bool = False
    result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch or type(
            self.status
        ) is not V1SampledV7PhaseARobustPairStatus:
            raise TypeError("robust branch/action result lost its exact enum")
        if self.physical_action_ordinal <= 0 or self.loewner_action_ordinal <= 0:
            raise ValueError("robust branch/action ordinals must be positive")
        for name in (
            "physical_action_identity_hash",
            "packed_action_hash",
            "branch_alias_hash",
            "source_resource_problem_hash",
            "upper_loewner_problem_hash",
            "upper_loewner_bounded_receipt_hash",
            "lower_loewner_problem_hash",
            "lower_loewner_bounded_receipt_hash",
        ):
            _identity(getattr(self, name), name=name)
        _finite(
            self.lower_loewner_q_min_lower_bound,
            name="lower Loewner q-min lower bound",
        )
        _finite(
            self.lower_loewner_q_min_upper_bound,
            name="lower Loewner q-min upper bound",
        )
        if self.lower_loewner_q_min_lower_bound > self.lower_loewner_q_min_upper_bound:
            raise ValueError("lower-Loewner q-min interval is reversed")
        for name in (
            "guaranteed_feasible_resource_lower_bound",
            "permissive_resource_upper_bound",
        ):
            value = getattr(self, name)
            if value is not None:
                _finite(value, name=name)
        if (
            self.guaranteed_feasible_resource_lower_bound is not None
            and self.permissive_resource_upper_bound is not None
            and self.guaranteed_feasible_resource_lower_bound
            > self.permissive_resource_upper_bound
        ):
            raise ValueError("robust resource endpoint interval is reversed")
        for name in (
            "lower_loewner_certifies_true_q_infeasible",
            "all_required_binary64_bounds_closed",
            "engineering_budget_exhausted",
            "registered_endpoint_cell_cover_closed",
            "registered_packed_action_image_closed",
            "nondominated",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be exact bool")
        object.__setattr__(
            self,
            "result_hash",
            self._compute_hash(),
        )

    def _compute_hash(self) -> str:
        return _payload_hash(
            {
                item.name: getattr(self, item.name)
                for item in fields(self)
                if item.name != "result_hash"
            },
            role="branch-action-result",
        )

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self.result_hash:
            _refuse(
                V1SampledV7PhaseARobustShellDecisionRefusalCode.STALE_SOURCE,
                "one robust branch/action result no longer replays",
            )
        return observed


@dataclass(frozen=True, slots=True)
class V1SampledV7PhaseARobustShellDecisionReceipt:
    action_pair_hash: str
    action_pair_alias_source_join_receipt_hash: str | None
    loewner_stream_receipt_hash: str
    relaxed_kkt_problem_hash: str
    parent_mask_hash: str
    alias_map_receipt_hash: str | None
    search_design_hash: str
    route_axis_hash: str
    route_count: int
    current_mask_hash: str
    shell_indices: tuple[int, ...]
    shell_route_ids: tuple[str, ...]
    physical_action_identity_hashes: tuple[str, ...]
    branch_action_results: tuple[V1SampledV7PhaseARobustBranchActionResult, ...]
    branch_action_result_axis_hash: str
    q_budget: float
    maximum_guaranteed_resource_lower_endpoint: float | None
    nondominated_result_hashes: tuple[str, ...]
    nondominated_physical_action_identity_hashes: tuple[str, ...]
    selected_physical_action_identity_hash: str | None
    status: V1SampledV7PhaseARobustDecisionStatus
    source_owned_resource_and_capacity_derivation: bool
    source_owned_registered_cell_action_quotient: bool
    exact_action_pair_alias_source_graph_joined: bool
    source_bound_population_loewner_geometry: bool
    private_reference_only: bool
    local_endpoint_action_image_premise_closed: bool
    outside_shell_fixed_point_reference_hash: str | None
    outside_shell_fixed_point_reference_status: V1PhaseAOutsideShellFixedPointStatus | None
    outside_shell_fixed_point_reference_certified: bool
    inherited_alpha_ledger_hash: str
    inherited_alpha_coordinate_hash: str
    inherited_alpha_lineage_hash: str
    inherited_familywise_alpha: float
    inherited_alpha_spent: float
    additional_alpha_spent: float = 0.0
    loewner_population_geometry_used_directly: bool = True
    spectral_rho_residual_readded: bool = False
    diagonal_d_residual_readded: bool = False
    loewner_lower_psd_projection_performed: bool = False
    corrected_a_phase_a_sensitivity_upper_bound: float = 0.0
    signed_f_phase_a_sensitivity_upper_bound: float = 0.0
    branch_aliases_preserved_separately_from_physical_actions: bool = True
    exact_packed_physical_action_quotient_used: bool = True
    packed_delta_resource_derivation_bounded_workspace: bool = True
    route_scale_resource_copy_materialized: bool = False
    outside_no_entry_authority: bool = False
    adaptive_shell_fixed_point_authority: bool = False
    d_stability_authority: bool = False
    c_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    algorithm: str = SAMPLED_V7_PHASE_A_ROBUST_SHELL_DECISION_ALGORITHM
    authority_scope: str = SAMPLED_V7_PHASE_A_ROBUST_SHELL_DECISION_AUTHORITY
    schema: str = SAMPLED_V7_PHASE_A_ROBUST_SHELL_DECISION_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _RECEIPT_TOKEN:
            raise PermissionError("robust-shell decision receipts require their issuer")
        self._validate()
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _validate(self) -> None:
        for name in (
            "action_pair_hash",
            "loewner_stream_receipt_hash",
            "relaxed_kkt_problem_hash",
            "parent_mask_hash",
            "search_design_hash",
            "route_axis_hash",
            "current_mask_hash",
            "branch_action_result_axis_hash",
            "inherited_alpha_ledger_hash",
            "inherited_alpha_coordinate_hash",
            "inherited_alpha_lineage_hash",
        ):
            _identity(getattr(self, name), name=name)
        if self.action_pair_alias_source_join_receipt_hash is not None:
            _identity(
                self.action_pair_alias_source_join_receipt_hash,
                name="action_pair_alias_source_join_receipt_hash",
            )
        if self.alias_map_receipt_hash is not None:
            _identity(self.alias_map_receipt_hash, name="alias_map_receipt_hash")
        if self.outside_shell_fixed_point_reference_hash is not None:
            _identity(
                self.outside_shell_fixed_point_reference_hash,
                name="outside_shell_fixed_point_reference_hash",
            )
        if (
            self.outside_shell_fixed_point_reference_hash is None
        ) is not (self.outside_shell_fixed_point_reference_status is None):
            raise ValueError("fixed-point reference hash/status presence does not align")
        if (
            self.outside_shell_fixed_point_reference_status is not None
            and type(self.outside_shell_fixed_point_reference_status)
            is not V1PhaseAOutsideShellFixedPointStatus
        ):
            raise TypeError("fixed-point reference status must be exact")
        if type(self.outside_shell_fixed_point_reference_certified) is not bool:
            raise TypeError("fixed-point reference certification flag must be exact bool")
        expected_reference_certified = (
            self.outside_shell_fixed_point_reference_status
            is V1PhaseAOutsideShellFixedPointStatus.CERTIFIED_OUTSIDE_STRICT_SHELL_IMAGE_CLOSED_REFERENCE
        )
        if (
            self.outside_shell_fixed_point_reference_certified
            != expected_reference_certified
        ):
            raise ValueError("fixed-point reference certification flag does not match status")
        if type(self.status) is not V1SampledV7PhaseARobustDecisionStatus:
            raise TypeError("robust decision status must be exact")
        if self.route_count <= 0 or not self.shell_indices:
            raise ValueError("robust decision route/shell axes must be nonempty")
        if len(self.shell_indices) != len(self.shell_route_ids):
            raise ValueError("robust decision shell axes do not align")
        if not self.physical_action_identity_hashes:
            raise ValueError("robust decision requires a physical-action axis")
        rows = tuple(self.branch_action_results)
        if (
            not rows
            or any(type(value) is not V1SampledV7PhaseARobustBranchActionResult for value in rows)
            or len({value.result_hash for value in rows}) != len(rows)
        ):
            raise TypeError("robust decision result axis is empty or duplicated")
        tuple(value.assert_integrity() for value in rows)
        expected_axis_hash = _payload_hash(
            {"result_hashes": tuple(value.result_hash for value in rows)},
            role="result-axis",
        )
        if self.branch_action_result_axis_hash != expected_axis_hash:
            raise ValueError("robust decision result-axis hash does not replay")
        nondominated = tuple(value.result_hash for value in rows if value.nondominated)
        if self.nondominated_result_hashes != nondominated:
            raise ValueError("nondominated result hashes do not replay")
        _nonnegative(self.q_budget, name="q_budget")
        if self.maximum_guaranteed_resource_lower_endpoint is not None:
            _finite(
                self.maximum_guaranteed_resource_lower_endpoint,
                name="maximum guaranteed resource lower endpoint",
            )
        for name in (
            "inherited_familywise_alpha",
            "inherited_alpha_spent",
            "additional_alpha_spent",
            "corrected_a_phase_a_sensitivity_upper_bound",
            "signed_f_phase_a_sensitivity_upper_bound",
        ):
            _nonnegative(getattr(self, name), name=name)
        true_flags = (
            self.source_owned_resource_and_capacity_derivation,
            self.loewner_population_geometry_used_directly,
            self.branch_aliases_preserved_separately_from_physical_actions,
            self.exact_packed_physical_action_quotient_used,
            self.packed_delta_resource_derivation_bounded_workspace,
        )
        false_flags = (
            self.spectral_rho_residual_readded,
            self.diagonal_d_residual_readded,
            self.loewner_lower_psd_projection_performed,
            self.route_scale_resource_copy_materialized,
            self.outside_no_entry_authority,
            self.adaptive_shell_fixed_point_authority,
            self.d_stability_authority,
            self.c_measurement_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.performance_authority,
            self.pareto_authority,
        )
        if (
            any(type(value) is not bool for value in true_flags + false_flags)
            or any(value is not True for value in true_flags)
            or any(value is not False for value in false_flags)
            or type(self.source_owned_registered_cell_action_quotient) is not bool
            or type(self.exact_action_pair_alias_source_graph_joined) is not bool
            or type(self.source_bound_population_loewner_geometry) is not bool
            or type(self.private_reference_only) is not bool
            or self.source_owned_registered_cell_action_quotient
            is self.private_reference_only
            or self.exact_action_pair_alias_source_graph_joined
            is not self.source_owned_registered_cell_action_quotient
            or (
                self.action_pair_alias_source_join_receipt_hash is None
            )
            is not self.private_reference_only
            or (self.alias_map_receipt_hash is None) is not self.private_reference_only
            or self.local_endpoint_action_image_premise_closed
            is not (
                self.status
                in {
                    V1SampledV7PhaseARobustDecisionStatus.STABLE_PHYSICAL_ACTION_PREMISE,
                    V1SampledV7PhaseARobustDecisionStatus.STABLE_NO_ACTION_PREMISE,
                }
            )
            or self.additional_alpha_spent != 0.0
            or self.corrected_a_phase_a_sensitivity_upper_bound != 0.0
            or self.signed_f_phase_a_sensitivity_upper_bound != 0.0
            or self.algorithm != SAMPLED_V7_PHASE_A_ROBUST_SHELL_DECISION_ALGORITHM
            or self.authority_scope != SAMPLED_V7_PHASE_A_ROBUST_SHELL_DECISION_AUTHORITY
            or self.schema != SAMPLED_V7_PHASE_A_ROBUST_SHELL_DECISION_SCHEMA
        ):
            _refuse(
                V1SampledV7PhaseARobustShellDecisionRefusalCode.AUTHORITY_VIOLATION,
                "robust-shell receipt crossed its local-premise authority",
            )
        if self.status is V1SampledV7PhaseARobustDecisionStatus.STABLE_PHYSICAL_ACTION_PREMISE:
            if (
                self.selected_physical_action_identity_hash is None
                or self.nondominated_physical_action_identity_hashes
                != (self.selected_physical_action_identity_hash,)
            ):
                raise ValueError("stable physical decision is not a singleton quotient")
        elif self.selected_physical_action_identity_hash is not None:
            raise ValueError("non-action decision retained a selected physical action")

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                item.name: (
                    tuple(value.result_hash for value in self.branch_action_results)
                    if item.name == "branch_action_results"
                    else getattr(self, item.name)
                )
                for item in fields(self)
                if not item.name.startswith("_")
            }
        )

    def _compute_hash(self) -> str:
        return _payload_hash(self._payload(), role="receipt")

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self._validate()
        observed = self._compute_hash()
        if observed != self._sealed_hash or (
            expected_hash is not None and observed != expected_hash
        ):
            _refuse(
                V1SampledV7PhaseARobustShellDecisionRefusalCode.STALE_SOURCE,
                "robust-shell receipt no longer replays",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


def _source_resource_problem(
    *,
    action_pair: V1SampledV7BranchActionPair,
    loewner_result: V1LiteralJJointAffineShellLoewnerStreamResult,
    relaxed_kkt_problem: V1RelaxedKKTProblem,
    parent_mask: HardMask,
) -> tuple[_ActionResourceProblem, ...]:
    receipt = loewner_result.receipt
    if (
        relaxed_kkt_problem.problem_hash != action_pair.response_geometry_problem_hash
        or relaxed_kkt_problem.route_axis_hash != action_pair.route_axis_hash
        or relaxed_kkt_problem.route_count != action_pair.route_count
        or parent_mask.mask_hash != action_pair.parent_mask_hash
        or parent_mask.route_axis_hash != action_pair.route_axis_hash
        or parent_mask.route_count != action_pair.route_count
        or receipt.current_mask_hash != parent_mask.mask_hash
        or receipt.route_axis_hash != action_pair.route_axis_hash
        or receipt.route_count != action_pair.route_count
    ):
        _refuse(
            V1SampledV7PhaseARobustShellDecisionRefusalCode.STALE_SOURCE,
            "KKT problem, parent mask, action pair, and Loewner chart are not one source",
        )
    if not relaxed_kkt_problem.block_capacities:
        _refuse(
            V1SampledV7PhaseARobustShellDecisionRefusalCode.SOURCE_RESOURCE_ADAPTER_REQUIRED,
            "source problem has no exact compact one-live-survivor block capacities",
        )
    shell = receipt.shell_indices
    if any(not bool(relaxed_kkt_problem.eligible_mask[index]) for index in shell):
        _refuse(
            V1SampledV7PhaseARobustShellDecisionRefusalCode.BLOCK_CAPACITY_ILLEGAL,
            "the registered shell contains a source-ineligible route",
        )
    coefficients = tuple(float(relaxed_kkt_problem.resource[index]) for index in shell)
    block_stops = np.asarray(
        [value.stop for value in relaxed_kkt_problem.block_capacities],
        dtype=np.int64,
    )
    shell_offsets_by_block: dict[int, tuple[int, ...]] = {}
    for offset, physical in enumerate(shell):
        block = int(np.searchsorted(block_stops, physical, side="right"))
        shell_offsets_by_block.setdefault(block, ())
        shell_offsets_by_block[block] += (offset,)
    problems: list[_ActionResourceProblem] = []
    for custody in action_pair.physical_actions:
        action = custody.packed_action
        child = action.resulting_mask
        try:
            count, index_hash = child.removed_index_count_and_hash_from(parent_mask)
        except Exception as error:
            _refuse(
                V1SampledV7PhaseARobustShellDecisionRefusalCode.PACKED_ACTION_MISMATCH,
                f"packed action is not an irreversible child: {type(error).__name__}: {error}",
            )
        if count != action.removed_count or index_hash != action.removed_route_index_hash:
            _refuse(
                V1SampledV7PhaseARobustShellDecisionRefusalCode.PACKED_ACTION_MISMATCH,
                "packed parent-to-child delta does not replay its source identity",
            )
        outside_chunk_sums: list[float] = []
        outside_chunk_rounding_radii: list[float] = []
        outside_block_counts = [0] * len(relaxed_kkt_problem.block_capacities)
        removed_shell_offsets: set[int] = set()
        shell_offset_by_physical = {physical: offset for offset, physical in enumerate(shell)}
        for chunk in child.iter_removed_index_chunks_from(parent_mask):
            indices = np.asarray(chunk, dtype=np.int64)
            if bool(np.any(~relaxed_kkt_problem.eligible_mask[indices])):
                _refuse(
                    V1SampledV7PhaseARobustShellDecisionRefusalCode.BLOCK_CAPACITY_ILLEGAL,
                    "packed action removes a source-ineligible route",
                )
            shell_membership = np.isin(
                indices,
                np.asarray(shell, dtype=np.int64),
                assume_unique=True,
            )
            for physical in (int(value) for value in indices[shell_membership]):
                removed_shell_offsets.add(shell_offset_by_physical[physical])
            outside_indices = indices[~shell_membership]
            if outside_indices.size:
                values = relaxed_kkt_problem.resource[outside_indices]
                chunk_sum = math.fsum(float(value) for value in values)
                outside_chunk_sums.append(chunk_sum)
                # ``fsum`` is normally correctly rounded and may very rarely
                # double-round on non-Windows builds.  One full ulp per bounded
                # chunk plus one full ulp for the final reduction is a strict,
                # deliberately conservative enclosure for this nonnegative sum.
                outside_chunk_rounding_radii.append(math.ulp(chunk_sum))
                block_indices = np.searchsorted(
                    block_stops,
                    outside_indices,
                    side="right",
                )
                counts = np.bincount(
                    block_indices,
                    minlength=len(outside_block_counts),
                )
                for block, observed in enumerate(counts):
                    outside_block_counts[block] += int(observed)
        for block_index, block in enumerate(relaxed_kkt_problem.block_capacities):
            if outside_block_counts[block_index] > block.removal_capacity:
                _refuse(
                    V1SampledV7PhaseARobustShellDecisionRefusalCode.BLOCK_CAPACITY_ILLEGAL,
                    "packed outside anchor exceeds a source block capacity even "
                    "though that block has no variable shell coordinate",
                )
        outside_center = math.fsum(outside_chunk_sums)
        if not math.isfinite(outside_center):
            _refuse(
                V1SampledV7PhaseARobustShellDecisionRefusalCode.NUMERICAL_REFUSAL,
                "outside resource accumulation overflowed binary64",
            )
        outside_radius = math.fsum(outside_chunk_rounding_radii) + math.ulp(
            outside_center
        )
        outside_lower = max(
            0.0,
            math.nextafter(
                outside_center - outside_radius,
                -math.inf,
            ),
        )
        outside_upper = math.nextafter(
            outside_center + outside_radius,
            math.inf,
        )
        capacities: list[V1SignedWorkVariableShellActionCellBlockCapacity] = []
        for block_index, offsets in sorted(shell_offsets_by_block.items()):
            block = relaxed_kkt_problem.block_capacities[block_index]
            remaining = block.removal_capacity - outside_block_counts[block_index]
            if remaining < 0 or len(removed_shell_offsets.intersection(offsets)) > remaining:
                _refuse(
                    V1SampledV7PhaseARobustShellDecisionRefusalCode.BLOCK_CAPACITY_ILLEGAL,
                    "packed outside anchor or its physical shell action exceeds a source block capacity",
                )
            capacities.append(
                V1SignedWorkVariableShellActionCellBlockCapacity(
                    shell_size=len(shell),
                    shell_offsets=offsets,
                    remaining_capacity=min(remaining, len(offsets)),
                    block_id=block.block_id,
                )
            )
        flattened = tuple(offset for value in capacities for offset in value.shell_offsets)
        if tuple(sorted(flattened)) != tuple(range(len(shell))):
            _refuse(
                V1SampledV7PhaseARobustShellDecisionRefusalCode.BLOCK_CAPACITY_ILLEGAL,
                "source-derived compact capacities do not partition the shell",
            )
        resource_hash = _payload_hash(
            {
                "source_problem_hash": relaxed_kkt_problem.problem_hash,
                "physical_action_identity_hash": custody.physical_action_identity_hash,
                "packed_action_hash": custody.packed_action_hash,
                "shell_indices": shell,
                "outside_resource_lower_bound": outside_lower,
                "outside_resource_upper_bound": outside_upper,
                "shell_resource_coefficients": coefficients,
                "block_capacity_hashes": tuple(value.capacity_hash for value in capacities),
                "outside_delta_count": count - len(removed_shell_offsets),
            },
            role="source-resource-problem",
        )
        problems.append(
            _ActionResourceProblem(
                physical_action_ordinal=custody.physical_action_ordinal,
                physical_action_identity_hash=custody.physical_action_identity_hash,
                packed_action_hash=custody.packed_action_hash,
                outside_resource_lower_bound=outside_lower,
                outside_resource_upper_bound=outside_upper,
                shell_resource_coefficients=coefficients,
                block_capacities=tuple(capacities),
                source_problem_hash=relaxed_kkt_problem.problem_hash,
                resource_problem_hash=resource_hash,
            )
        )
    return tuple(problems)


def _resource_cell_image(
    *,
    action_pair_alias_source_join: V1SampledV7ActionPairAliasSourceJoinReceipt | None,
    action_pair: V1SampledV7BranchActionPair,
    private_cells: Sequence[V1SampledV7PhaseAPrivateResourceCell],
    private_quotients: Sequence[V1SampledV7PhaseAPrivateActionQuotient],
    allow_private_reference_for_tests: bool,
) -> tuple[
    tuple[_ResourceCell, ...],
    tuple[_ActionQuotient, ...],
    str | None,
    str | None,
    bool,
]:
    if action_pair_alias_source_join is not None:
        if (
            type(action_pair_alias_source_join)
            is not V1SampledV7ActionPairAliasSourceJoinReceipt
            or private_cells
            or private_quotients
        ):
            _refuse(
                V1SampledV7PhaseARobustShellDecisionRefusalCode.INVALID_TYPE,
                "production exact-source custody cannot be mixed with private records",
            )
        try:
            source_join_hash = action_pair_alias_source_join.assert_integrity()
            if action_pair_alias_source_join._registration._action_pair is not action_pair:
                _refuse(
                    V1SampledV7PhaseARobustShellDecisionRefusalCode.EXACT_SOURCE_JOIN_REQUIRED,
                    "source join does not retain this exact action-pair object",
                )
            alias_map = action_pair_alias_source_join._alias_map
            if type(alias_map) is not V1PhaseACompleteRegisteredAliasMapReceipt:
                raise TypeError("source join lost its exact alias-map parent")
            alias_hash = alias_map.assert_live_identity()
            source_image = alias_map.source_image
            if type(source_image) is not V1SampledV7DSourceImageBindingReceipt:
                raise TypeError("registered alias map lost its exact source image")
            source_image_hash = source_image.assert_integrity()
        except Exception as error:
            if isinstance(error, V1SampledV7PhaseARobustShellDecisionRefusal):
                raise
            _refuse(
                V1SampledV7PhaseARobustShellDecisionRefusalCode.STALE_SOURCE,
                "exact action-pair/alias source join is stale: "
                f"{type(error).__name__}: {error}",
            )
        pair_physical = tuple(
            value.physical_action_identity_hash for value in action_pair.physical_actions
        )
        if (
            alias_map.source_image_receipt_hash != source_image_hash
            or source_image.parent_mask_hash != action_pair.parent_mask_hash
            or source_image.route_axis_hash != action_pair.route_axis_hash
            or source_image.route_count != action_pair.route_count
            or alias_map.route_axis_hash != action_pair.route_axis_hash
            or alias_map.route_count != action_pair.route_count
            or not set(pair_physical).issubset(
                {physical for _cell, physical in source_image.cuda_physical_action_family}
            )
            or any(
                alias.parent_mask_hash != action_pair.parent_mask_hash
                or alias.route_axis_hash != action_pair.route_axis_hash
                or alias.route_count != action_pair.route_count
                or alias.resource_coordinate_hash
                != action_pair.resource_coordinate_hash
                for alias in action_pair.branch_aliases
            )
        ):
            _refuse(
                V1SampledV7PhaseARobustShellDecisionRefusalCode.ACTION_AXIS_MISMATCH,
                "registered alias map and action pair do not share one exact source image",
            )
        if (
            action_pair_alias_source_join.pair_hash != action_pair.pair_hash
            or action_pair_alias_source_join.alias_map_receipt_hash != alias_hash
            or action_pair_alias_source_join.source_image_receipt_hash
            != source_image_hash
            or alias_map.complete_crosswalk_receipt_hash
            != source_image.complete_action_crosswalk_receipt_hash
            or alias_map.action_cell_registry_receipt_hash
            != source_image.phase_a_registry_receipt_hash
            or alias_map.qualification_hash != source_image.cuda_qualification_hash
            or action_pair_alias_source_join.uncertainty_set_hash
            != source_image.uncertainty_set_hash
        ):
            _refuse(
                V1SampledV7PhaseARobustShellDecisionRefusalCode.ACTION_AXIS_MISMATCH,
                "registered alias map and action pair do not replay one exact D source image",
            )
        for alias in action_pair.branch_aliases:
            expected = (
                ()
                if alias.physical_action_identity_hash is None
                else (alias.physical_action_identity_hash,)
            )
            observed = tuple(
                sorted(
                    {
                        physical
                        for quotient in alias_map.action_cell_quotients
                        if quotient.branch is alias.branch
                        and quotient.resource_coordinate_hash
                        == action_pair.resource_coordinate_hash
                        for physical in quotient.physical_action_identity_hashes
                    }
                )
            )
            if observed != expected:
                _refuse(
                    V1SampledV7PhaseARobustShellDecisionRefusalCode.ACTION_AXIS_MISMATCH,
                    "registered alias map current-coordinate action image differs "
                    "from the exact branch action pair",
                )
        state = alias_map.decision_margin_lease.state
        cells = tuple(
            _ResourceCell(
                resource_cell_hash=value.resource_cell_hash,
                lower=float(value.cell_lower_bound),
                upper=float(value.cell_upper_bound),
                source_hash=value.resource_cell_binding_hash,
            )
            for value in state.resource_cell_boundaries
        )
        quotients = tuple(
            _ActionQuotient(
                resource_cell_hash=value.resource_cell_hash,
                branch=value.branch,
                physical_action_identity_hashes=tuple(
                    sorted(value.physical_action_identity_hashes)
                ),
                source_hash=value.quotient_hash,
            )
            for value in alias_map.action_cell_quotients
        )
        return cells, quotients, alias_hash, source_join_hash, True
    if not allow_private_reference_for_tests:
        _refuse(
            V1SampledV7PhaseARobustShellDecisionRefusalCode.EXACT_SOURCE_JOIN_REQUIRED,
            "production decisions require the exact action-pair/alias source-graph join",
        )
    cells_in = tuple(private_cells)
    quotients_in = tuple(private_quotients)
    if (
        not cells_in
        or not quotients_in
        or any(type(value) is not V1SampledV7PhaseAPrivateResourceCell for value in cells_in)
        or any(type(value) is not V1SampledV7PhaseAPrivateActionQuotient for value in quotients_in)
    ):
        _refuse(
            V1SampledV7PhaseARobustShellDecisionRefusalCode.INVALID_TYPE,
            "private reference requires exact nonempty cell and quotient records",
        )
    cells = tuple(
        _ResourceCell(
            resource_cell_hash=value.resource_cell_hash,
            lower=value.cell_lower_bound,
            upper=value.cell_upper_bound,
            source_hash=value.cell_hash,
        )
        for value in cells_in
    )
    quotients = tuple(
        _ActionQuotient(
            resource_cell_hash=value.resource_cell_hash,
            branch=value.branch,
            physical_action_identity_hashes=value.physical_action_identity_hashes,
            source_hash=value.quotient_hash,
        )
        for value in quotients_in
    )
    return cells, quotients, None, None, False


def _source_image_action_pair_join_graph(
    *,
    source_image: V1SampledV7DSourceImageBindingReceipt,
    action_pair: V1SampledV7BranchActionPair,
) -> Mapping[str, object]:
    """Replay the legacy public D-source image as a diagnostic only.

    Public hash equality cannot prove the pre-C crosswalk ancestry.  Canonical
    production issuance therefore never calls this helper; it consumes
    ``V1SampledV7ActionPairAliasSourceJoinReceipt`` instead.  The helper stays
    readable only for legacy replay/refusal tests and grants no source,
    D-stability, or downstream authority.
    """

    if type(source_image) is not V1SampledV7DSourceImageBindingReceipt:
        _refuse(
            V1SampledV7PhaseARobustShellDecisionRefusalCode.INVALID_TYPE,
            "source-image join requires the exact D source-image receipt",
        )
    try:
        branches = tuple(source_image._branch_evidence)
        projections = tuple(source_image._live_projections)
        common_source = source_image._common_source
        crosswalk = source_image._complete_crosswalk
        registry = source_image._phase_a_registry
        qualification = source_image._cuda_qualification
        common_receipt = common_source.receipt
        branch_hashes = tuple(value.assert_integrity() for value in branches)
        projection_hashes = tuple(value.assert_integrity() for value in projections)
        crosswalk_hash = crosswalk.assert_integrity()
        registry.assert_live_sources()
        qualification.assert_live_identity()
    except Exception as error:
        _refuse(
            V1SampledV7PhaseARobustShellDecisionRefusalCode.STALE_SOURCE,
            "D source-image live graph failed exact replay: "
            f"{type(error).__name__}: {error}",
        )
    if (
        len(branches) != 2
        or len(projections) != 2
        or tuple(value.branch for value in branches) != action_pair.branch_axis
        or tuple(getattr(value.phase_a_branch, "value", None) for value in projections)
        != tuple(value.value for value in action_pair.branch_axis)
        or branch_hashes != source_image.branch_evidence_receipt_hashes
        or projection_hashes != source_image.source_root_projection_receipt_hashes
        or crosswalk_hash != source_image.complete_action_crosswalk_receipt_hash
        or crosswalk.branch_evidence_receipt_hashes != branch_hashes
        or crosswalk.source_root_projection_receipt_hashes != projection_hashes
        or crosswalk.all_action_projection_family_receipt_hash
        != source_image.all_action_projection_family_receipt_hash
        or crosswalk.phase_a_registry_receipt_hash
        != source_image.phase_a_registry_receipt_hash
        or crosswalk.cuda_qualification_hash != source_image.cuda_qualification_hash
        or common_receipt.receipt_hash != source_image.common_source_receipt_hash
        or common_receipt.source_replay_hash != source_image.common_source_replay_hash
        or qualification.qualification_hash != source_image.cuda_qualification_hash
        or tuple(qualification.physical_action_family)
        != source_image.cuda_physical_action_family
        or tuple(crosswalk.cuda_physical_action_family)
        != source_image.cuda_physical_action_family
        or tuple(crosswalk.branch_action_projection_domain)
        != source_image.branch_action_projection_domain
    ):
        _refuse(
            V1SampledV7PhaseARobustShellDecisionRefusalCode.ACTION_AXIS_MISMATCH,
            "D source image does not replay the exact branch/projection/crosswalk graph",
        )
    pair_by_branch = {alias.branch: alias for alias in action_pair.branch_aliases}
    pair_physical = {
        custody.physical_action_identity_hash: custody
        for custody in action_pair.physical_actions
    }
    for evidence in branches:
        alias = pair_by_branch[evidence.branch]
        if (
            evidence.source_receipt_hash != source_image.common_source_receipt_hash
            or evidence.source_replay_hash != source_image.common_source_replay_hash
            or evidence.parent_mask_hash != action_pair.parent_mask_hash
            or evidence.route_axis_hash != action_pair.route_axis_hash
            or evidence.route_count != action_pair.route_count
            or evidence.uncertainty_set_hash != source_image.uncertainty_set_hash
        ):
            _refuse(
                V1SampledV7PhaseARobustShellDecisionRefusalCode.ACTION_AXIS_MISMATCH,
                "branch evidence is not on the action pair source axis",
            )
        if alias.has_action:
            custody = pair_physical.get(evidence.preferred_physical_action_identity_hash)
            if (
                custody is None
                or alias.physical_action_identity_hash
                != evidence.preferred_physical_action_identity_hash
                or alias.packed_action_hash != evidence.preferred_packed_action_hash
                or alias.resulting_mask_hash != evidence.preferred_resulting_mask_hash
                or alias.removed_route_index_hash
                != evidence.preferred_removed_route_index_hash
                or alias.removed_count != evidence.preferred_removed_count
                or custody.packed_action_hash != evidence.preferred_packed_action_hash
                or custody.resulting_mask_hash != evidence.preferred_resulting_mask_hash
                or custody.removed_route_index_hash
                != evidence.preferred_removed_route_index_hash
                or custody.removed_count != evidence.preferred_removed_count
            ):
                _refuse(
                    V1SampledV7PhaseARobustShellDecisionRefusalCode.ACTION_AXIS_MISMATCH,
                    "branch evidence preferred action differs from action-pair custody",
                )
    for projection in projections:
        if (
            projection.common_source_receipt_hash
            != source_image.common_source_receipt_hash
            or projection.common_source_replay_hash
            != source_image.common_source_replay_hash
            or projection.current_mask_hash != action_pair.parent_mask_hash
            or projection.route_axis_hash != action_pair.route_axis_hash
            or projection.route_count != action_pair.route_count
            or projection.origin_q_source_receipt_hash
            not in {
                evidence.origin_branch_source_receipt_hash
                for evidence in branches
                if getattr(evidence.branch, "value", None)
                == getattr(projection.phase_a_branch, "value", None)
            }
        ):
            _refuse(
                V1SampledV7PhaseARobustShellDecisionRefusalCode.ACTION_AXIS_MISMATCH,
                "live common-D/origin-Q projection is not on the action pair source axis",
            )
    return MappingProxyType(
        {
            "parent_mask_hash": source_image.parent_mask_hash,
            "route_axis_hash": source_image.route_axis_hash,
            "route_count": source_image.route_count,
            "uncertainty_set_hash": source_image.uncertainty_set_hash,
            "source_image_receipt_hash": source_image.receipt_hash,
            "complete_crosswalk_receipt_hash": crosswalk_hash,
        }
    )


def _capacity_constraints(
    values: tuple[V1SignedWorkVariableShellActionCellBlockCapacity, ...],
) -> tuple[V1PhaseAShellBlockCapacityConstraint, ...]:
    if not values or any(
        type(value) is not V1SignedWorkVariableShellActionCellBlockCapacity
        for value in values
    ):
        _refuse(
            V1SampledV7PhaseARobustShellDecisionRefusalCode.BLOCK_CAPACITY_ILLEGAL,
            "endpoint solve requires exact source-derived block capacities",
        )
    return tuple(
        V1PhaseAShellBlockCapacityConstraint(
            block_id=value.block_id,
            shell_indices=value.shell_offsets,
            removal_capacity=float(value.remaining_capacity),
        )
        for value in values
    )


def _covered_cells(
    *,
    lower: float,
    upper: float,
    cells: tuple[_ResourceCell, ...],
) -> tuple[tuple[_ResourceCell, ...], bool]:
    crossed = tuple(
        sorted(
            (value for value in cells if value.upper >= lower and value.lower <= upper),
            key=lambda value: (value.lower, value.upper, value.resource_cell_hash),
        )
    )
    if not crossed:
        return (), False
    cursor = lower
    for cell in crossed:
        # This is coverage of a real-valued endpoint interval, not a search
        # over representable FP64 points.  Even a one-ulp open gap can contain
        # the true resource endpoint, so positive authority may not bridge it
        # with a numerical tolerance.
        if cell.lower > cursor:
            return crossed, False
        cursor = max(cursor, cell.upper)
        if cursor >= upper:
            return crossed, True
    return crossed, cursor >= upper


def _solve_action(
    *,
    action_problem: _ActionResourceProblem,
    action_index: int,
    lower_matrix: tuple[tuple[float, ...], ...],
    upper_matrix: tuple[tuple[float, ...], ...],
    q_budget: float,
    search_design: V1PhaseABoundedShellGlobalQuadraticSearchDesign,
) -> tuple[
    V1PhaseABoundedShellGlobalQuadraticReceipt,
    V1PhaseABoundedShellGlobalQuadraticReceipt,
    str,
    str,
]:
    constraints = _capacity_constraints(action_problem.block_capacities)
    upper_problem = issue_v1_phase_a_bounded_shell_global_quadratic_problem(
        problem_id=f"{action_problem.resource_problem_hash}:upper:{action_index}",
        loewner_matrix=upper_matrix,
        resource_offset=action_problem.outside_resource_lower_bound,
        resource_coefficients=action_problem.shell_resource_coefficients,
        q_budget=q_budget,
        shell_block_capacity_constraints=constraints,
    )
    lower_problem = issue_v1_phase_a_bounded_shell_global_quadratic_problem(
        problem_id=f"{action_problem.resource_problem_hash}:lower:{action_index}",
        loewner_matrix=lower_matrix,
        resource_offset=action_problem.outside_resource_upper_bound,
        resource_coefficients=action_problem.shell_resource_coefficients,
        q_budget=q_budget,
        shell_block_capacity_constraints=constraints,
    )
    upper_receipt = solve_v1_phase_a_bounded_shell_global_quadratic(
        problem=upper_problem,
        design=search_design,
    )
    lower_receipt = solve_v1_phase_a_bounded_shell_global_quadratic(
        problem=lower_problem,
        design=search_design,
    )
    return (
        upper_receipt,
        lower_receipt,
        upper_problem.problem_hash,
        lower_problem.problem_hash,
    )


def _row_without_nondominated(
    *,
    action_problem: _ActionResourceProblem,
    action_index: int,
    branch: V1RelaxedKKTBranch,
    branch_alias_hash: str,
    small_axis_ids: tuple[str, ...],
    upper_receipt: V1PhaseABoundedShellGlobalQuadraticReceipt,
    lower_receipt: V1PhaseABoundedShellGlobalQuadraticReceipt,
    upper_problem_hash: str,
    lower_problem_hash: str,
    q_budget: float,
    cells: tuple[_ResourceCell, ...],
    quotients: tuple[_ActionQuotient, ...],
) -> V1SampledV7PhaseARobustBranchActionResult:
    exhausted = upper_receipt.engineering_budget_exhausted or lower_receipt.engineering_budget_exhausted
    all_closed = (
        upper_receipt.q_min_binary64_closed
        and lower_receipt.q_min_binary64_closed
        and (
            upper_receipt.binary64_endpoint_closed
            or upper_receipt.certified_infeasible
        )
        and (
            lower_receipt.binary64_endpoint_closed
            or lower_receipt.certified_infeasible
        )
        and not exhausted
    )
    true_infeasible = (
        lower_receipt.certified_infeasible
        or lower_receipt.q_min_lower_bound > q_budget
    )
    guaranteed = upper_receipt.resource_lower_bound
    permissive = lower_receipt.resource_upper_bound
    crossed: tuple[_ResourceCell, ...] = ()
    quotient_hashes: tuple[str, ...] = ()
    physical_image: tuple[str, ...] = ()
    cell_cover = False
    image_closed = False
    if true_infeasible:
        # The lower-Loewner Q-minimum lower bound is already a rigorous
        # population-wide lower bound even when its upper mate or the
        # unrelated resource-endpoint search remains open.  Once that lower
        # bound exceeds the budget (or the lower problem independently proves
        # infeasibility), extra branch-and-bound work cannot change the
        # no-action conclusion.
        status = V1SampledV7PhaseARobustPairStatus.CERTIFIED_INFEASIBLE
        guaranteed = None
        permissive = None
    elif exhausted or not all_closed:
        status = V1SampledV7PhaseARobustPairStatus.UNRESOLVED_ENGINEERING
    elif guaranteed is None or permissive is None:
        status = V1SampledV7PhaseARobustPairStatus.UNRESOLVED_FEASIBILITY
    else:
        crossed, cell_cover = _covered_cells(
            lower=guaranteed,
            upper=permissive,
            cells=cells,
        )
        matched = tuple(
            value
            for cell in crossed
            for value in quotients
            if value.resource_cell_hash == cell.resource_cell_hash
            and value.branch is branch
        )
        quotient_hashes = tuple(sorted({value.source_hash for value in matched}))
        physical_image = tuple(
            sorted(
                {
                    physical
                    for value in matched
                    for physical in value.physical_action_identity_hashes
                }
            )
        )
        image_closed = (
            cell_cover
            and bool(matched)
            and all(
                any(
                    value.resource_cell_hash == cell.resource_cell_hash
                    and value.branch is branch
                    for value in matched
                )
                for cell in crossed
            )
            and action_problem.physical_action_identity_hash in physical_image
        )
        status = (
            V1SampledV7PhaseARobustPairStatus.CLOSED_ENDPOINT
            if image_closed
            else V1SampledV7PhaseARobustPairStatus.UNRESOLVED_ACTION_IMAGE
        )
    return V1SampledV7PhaseARobustBranchActionResult(
        physical_action_ordinal=action_problem.physical_action_ordinal,
        physical_action_identity_hash=action_problem.physical_action_identity_hash,
        packed_action_hash=action_problem.packed_action_hash,
        branch=branch,
        branch_alias_hash=branch_alias_hash,
        loewner_action_ordinal=action_index + 1,
        loewner_small_axis_ids=small_axis_ids,
        source_resource_problem_hash=action_problem.resource_problem_hash,
        block_capacity_hashes=tuple(
            value.capacity_hash for value in action_problem.block_capacities
        ),
        upper_loewner_problem_hash=upper_problem_hash,
        upper_loewner_bounded_receipt_hash=upper_receipt.receipt_hash,
        lower_loewner_problem_hash=lower_problem_hash,
        lower_loewner_bounded_receipt_hash=lower_receipt.receipt_hash,
        guaranteed_feasible_resource_lower_bound=guaranteed,
        permissive_resource_upper_bound=permissive,
        lower_loewner_q_min_lower_bound=lower_receipt.q_min_lower_bound,
        lower_loewner_q_min_upper_bound=lower_receipt.q_min_upper_bound,
        lower_loewner_certifies_true_q_infeasible=true_infeasible,
        all_required_binary64_bounds_closed=all_closed,
        engineering_budget_exhausted=exhausted,
        crossed_resource_cell_hashes=tuple(
            value.resource_cell_hash for value in crossed
        ),
        crossed_action_quotient_hashes=quotient_hashes,
        crossed_physical_action_identity_hashes=physical_image,
        registered_endpoint_cell_cover_closed=cell_cover,
        registered_packed_action_image_closed=image_closed,
        status=status,
    )


def issue_v1_sampled_v7_phase_a_robust_shell_decision(
    *,
    action_pair: V1SampledV7BranchActionPair,
    loewner_result: V1LiteralJJointAffineShellLoewnerStreamResult,
    relaxed_kkt_problem: V1RelaxedKKTProblem,
    parent_mask: HardMask,
    search_design: V1PhaseABoundedShellGlobalQuadraticSearchDesign,
    action_pair_alias_source_join: (
        V1SampledV7ActionPairAliasSourceJoinReceipt | None
    ) = None,
    outside_shell_fixed_point_reference: (
        V1PhaseAOutsideShellFixedPointReferenceReceipt | None
    ) = None,
    private_resource_cells: Sequence[V1SampledV7PhaseAPrivateResourceCell] = (),
    private_action_quotients: Sequence[V1SampledV7PhaseAPrivateActionQuotient] = (),
    allow_private_reference_for_tests: bool = False,
) -> V1SampledV7PhaseARobustShellDecisionReceipt:
    """Issue one complete local robust endpoint/action-image premise.

    Production issuance requires ``action_pair_alias_source_join``.  Omitting
    it is legal only for the explicit small-domain private-reference path.
    """

    if (
        torch is None
        or type(action_pair) is not V1SampledV7BranchActionPair
        or type(loewner_result) is not V1LiteralJJointAffineShellLoewnerStreamResult
        or type(relaxed_kkt_problem) is not V1RelaxedKKTProblem
        or type(parent_mask) is not HardMask
        or type(search_design) is not V1PhaseABoundedShellGlobalQuadraticSearchDesign
        or type(allow_private_reference_for_tests) is not bool
    ):
        _refuse(
            V1SampledV7PhaseARobustShellDecisionRefusalCode.INVALID_TYPE,
            "robust-shell issuer requires exact typed source parents",
        )
    try:
        pair_hash = action_pair.assert_integrity()
        loewner_result.device_handle.assert_live()
        loewner_receipt = loewner_result.receipt
        loewner_hash = loewner_receipt.assert_integrity()
        design_hash = search_design.design_hash
    except Exception as error:
        _refuse(
            V1SampledV7PhaseARobustShellDecisionRefusalCode.STALE_SOURCE,
            f"one robust-shell source failed replay: {type(error).__name__}: {error}",
        )
    pair_identity_axis = tuple(
        value.physical_action_identity_hash for value in action_pair.physical_actions
    )
    pair_action_axis = tuple(
        value.packed_action_hash for value in action_pair.physical_actions
    )
    if (
        not pair_identity_axis
        or pair_identity_axis != loewner_receipt.physical_action_identity_hashes
        or pair_action_axis != loewner_receipt.packed_action_hashes
        or action_pair.route_axis_hash != loewner_receipt.route_axis_hash
        or action_pair.route_count != loewner_receipt.route_count
        or action_pair.parent_mask_hash != loewner_receipt.current_mask_hash
    ):
        _refuse(
            V1SampledV7PhaseARobustShellDecisionRefusalCode.LOEWNER_AXIS_MISMATCH,
            "Loewner action/packed/route axes differ from exact action custody",
        )
    resource_problems = _source_resource_problem(
        action_pair=action_pair,
        loewner_result=loewner_result,
        relaxed_kkt_problem=relaxed_kkt_problem,
        parent_mask=parent_mask,
    )
    cells, quotients, alias_hash, source_join_hash, source_owned_alias = (
        _resource_cell_image(
            action_pair_alias_source_join=action_pair_alias_source_join,
            action_pair=action_pair,
            private_cells=private_resource_cells,
            private_quotients=private_action_quotients,
            allow_private_reference_for_tests=allow_private_reference_for_tests,
        )
    )
    source_bound_loewner = bool(
        getattr(loewner_receipt, "population_loewner_authority", False)
        and getattr(
            loewner_receipt,
            "kkt_geometry_uncertainty_input_authority",
            False,
        )
    )
    if source_owned_alias and not source_bound_loewner:
        _refuse(
            V1SampledV7PhaseARobustShellDecisionRefusalCode.AUTHORITY_VIOLATION,
            "a production registered alias map requires production-authoritative "
            "population Loewner geometry",
        )
    if source_owned_alias and any(
        not any(
            quotient.branch is branch
            and custody.physical_action_identity_hash
            in quotient.physical_action_identity_hashes
            for quotient in quotients
        )
        for custody in action_pair.physical_actions
        for branch in custody.branch_alias_axis
    ):
        _refuse(
            V1SampledV7PhaseARobustShellDecisionRefusalCode.ACTION_AXIS_MISMATCH,
            "the registered action-cell quotient omits one exact physical-action/branch alias",
        )
    fixed_point_hash: str | None = None
    fixed_point_status: V1PhaseAOutsideShellFixedPointStatus | None = None
    fixed_point_certified = False
    if outside_shell_fixed_point_reference is not None:
        if (
            type(outside_shell_fixed_point_reference)
            is not V1PhaseAOutsideShellFixedPointReferenceReceipt
        ):
            _refuse(
                V1SampledV7PhaseARobustShellDecisionRefusalCode.INVALID_TYPE,
                "fixed-point diagnostic requires the exact reference receipt",
            )
        try:
            fixed_point_hash = outside_shell_fixed_point_reference.receipt_hash
        except Exception as error:
            _refuse(
                V1SampledV7PhaseARobustShellDecisionRefusalCode.STALE_SOURCE,
                "fixed-point diagnostic no longer replays: "
                f"{type(error).__name__}: {error}",
            )
        if (
            outside_shell_fixed_point_reference.route_axis_hash
            != action_pair.route_axis_hash
            or outside_shell_fixed_point_reference.current_mask_hash
            != action_pair.parent_mask_hash
            or tuple(outside_shell_fixed_point_reference.shell_route_identity_hashes)
            != loewner_receipt.shell_route_ids
            or outside_shell_fixed_point_reference.production_source_authority
            or outside_shell_fixed_point_reference.d_stability_authority
            or outside_shell_fixed_point_reference.c_k_o_commit_terminal_runtime_pareto_authority
        ):
            _refuse(
                V1SampledV7PhaseARobustShellDecisionRefusalCode.ACTION_AXIS_MISMATCH,
                "fixed-point diagnostic is not on the robust shell/action source axis",
            )
        fixed_point_status = outside_shell_fixed_point_reference.status
        fixed_point_certified = (
            fixed_point_status
            is V1PhaseAOutsideShellFixedPointStatus.CERTIFIED_OUTSIDE_STRICT_SHELL_IMAGE_CLOSED_REFERENCE
        )
    dimension = loewner_receipt.shell_size + 1
    with loewner_result.device_handle.borrow() as lease:
        if tuple(int(value) for value in lease.loewner_lower.shape) != (
            len(resource_problems),
            dimension,
            dimension,
        ) or tuple(int(value) for value in lease.loewner_upper.shape) != (
            len(resource_problems),
            dimension,
            dimension,
        ):
            _refuse(
                V1SampledV7PhaseARobustShellDecisionRefusalCode.LOEWNER_AXIS_MISMATCH,
                "live Loewner matrices do not match the action/shell axes",
            )
        lower_matrices = tuple(
            tuple(
                tuple(float(lease.loewner_lower[a, i, j].item()) for j in range(dimension))
                for i in range(dimension)
            )
            for a in range(len(resource_problems))
        )
        upper_matrices = tuple(
            tuple(
                tuple(float(lease.loewner_upper[a, i, j].item()) for j in range(dimension))
                for i in range(dimension)
            )
            for a in range(len(resource_problems))
        )
    rows: list[V1SampledV7PhaseARobustBranchActionResult] = []
    for action_index, action_problem in enumerate(resource_problems):
        custody = action_pair.physical_actions[action_index]
        solved = _solve_action(
            action_problem=action_problem,
            action_index=action_index,
            lower_matrix=lower_matrices[action_index],
            upper_matrix=upper_matrices[action_index],
            q_budget=relaxed_kkt_problem.quadratic_budget,
            search_design=search_design,
        )
        for branch in custody.branch_alias_axis:
            alias = action_pair.alias(branch)
            if (
                alias.physical_action_ordinal != custody.physical_action_ordinal
                or alias.physical_action_identity_hash
                != custody.physical_action_identity_hash
            ):
                _refuse(
                    V1SampledV7PhaseARobustShellDecisionRefusalCode.ACTION_AXIS_MISMATCH,
                    "branch alias no longer resolves to its exact physical action",
                )
            rows.append(
                _row_without_nondominated(
                    action_problem=action_problem,
                    action_index=action_index,
                    branch=branch,
                    branch_alias_hash=alias.alias_hash,
                    small_axis_ids=loewner_receipt.small_axis_ids[action_index],
                    upper_receipt=solved[0],
                    lower_receipt=solved[1],
                    upper_problem_hash=solved[2],
                    lower_problem_hash=solved[3],
                    q_budget=relaxed_kkt_problem.quadratic_budget,
                    cells=cells,
                    quotients=quotients,
                )
            )
    lower_witness_rows = tuple(
        value
        for value in rows
        if value.status is not V1SampledV7PhaseARobustPairStatus.CERTIFIED_INFEASIBLE
        and value.guaranteed_feasible_resource_lower_bound is not None
    )
    potentially_feasible_rows = tuple(
        value
        for value in rows
        if value.status is not V1SampledV7PhaseARobustPairStatus.CERTIFIED_INFEASIBLE
    )
    if lower_witness_rows:
        maximum_lower = max(
            value.guaranteed_feasible_resource_lower_bound
            for value in lower_witness_rows
            if value.guaranteed_feasible_resource_lower_bound is not None
        )
        nondominated_hashes = {
            value.result_hash
            for value in potentially_feasible_rows
            if value.permissive_resource_upper_bound is None
            or value.permissive_resource_upper_bound >= maximum_lower
        }
    else:
        maximum_lower = None
        # Without one certified feasible witness, no potentially feasible row
        # can be eliminated by resource dominance.
        nondominated_hashes = {
            value.result_hash for value in potentially_feasible_rows
        }
    marked_rows = tuple(
        V1SampledV7PhaseARobustBranchActionResult(
            **{
                item.name: (
                    value.result_hash in nondominated_hashes
                    if item.name == "nondominated"
                    else getattr(value, item.name)
                )
                for item in fields(value)
                if item.name != "result_hash"
            }
        )
        for value in rows
    )
    nondominated_rows = tuple(value for value in marked_rows if value.nondominated)
    unresolved_nondominated = tuple(
        value
        for value in nondominated_rows
        if value.status
        not in {
            V1SampledV7PhaseARobustPairStatus.CLOSED_ENDPOINT,
            V1SampledV7PhaseARobustPairStatus.CERTIFIED_INFEASIBLE,
        }
    )
    action_image = tuple(
        sorted(
            {
                physical
                for value in nondominated_rows
                for physical in value.crossed_physical_action_identity_hashes
            }
        )
    )
    all_infeasible = all(
        value.status is V1SampledV7PhaseARobustPairStatus.CERTIFIED_INFEASIBLE
        for value in marked_rows
    )
    if unresolved_nondominated:
        status = (
            V1SampledV7PhaseARobustDecisionStatus.UNRESOLVED_NONDOMINATED_ACTION_IMAGE
            if any(
                value.status is V1SampledV7PhaseARobustPairStatus.UNRESOLVED_ACTION_IMAGE
                for value in unresolved_nondominated
            )
            else V1SampledV7PhaseARobustDecisionStatus.UNRESOLVED_ENDPOINT_INTERVAL
        )
        selected = None
    elif all_infeasible:
        status = V1SampledV7PhaseARobustDecisionStatus.STABLE_NO_ACTION_PREMISE
        selected = None
    elif len(action_image) == 1 and nondominated_rows:
        status = V1SampledV7PhaseARobustDecisionStatus.STABLE_PHYSICAL_ACTION_PREMISE
        selected = action_image[0]
    else:
        status = V1SampledV7PhaseARobustDecisionStatus.UNRESOLVED_NONDOMINATED_ACTION_IMAGE
        selected = None
    result_axis_hash = _payload_hash(
        {"result_hashes": tuple(value.result_hash for value in marked_rows)},
        role="result-axis",
    )
    return V1SampledV7PhaseARobustShellDecisionReceipt(
        action_pair_hash=pair_hash,
        action_pair_alias_source_join_receipt_hash=source_join_hash,
        loewner_stream_receipt_hash=loewner_hash,
        relaxed_kkt_problem_hash=relaxed_kkt_problem.problem_hash,
        parent_mask_hash=parent_mask.mask_hash,
        alias_map_receipt_hash=alias_hash,
        search_design_hash=design_hash,
        route_axis_hash=action_pair.route_axis_hash,
        route_count=action_pair.route_count,
        current_mask_hash=loewner_receipt.current_mask_hash,
        shell_indices=loewner_receipt.shell_indices,
        shell_route_ids=loewner_receipt.shell_route_ids,
        physical_action_identity_hashes=pair_identity_axis,
        branch_action_results=marked_rows,
        branch_action_result_axis_hash=result_axis_hash,
        q_budget=relaxed_kkt_problem.quadratic_budget,
        maximum_guaranteed_resource_lower_endpoint=maximum_lower,
        nondominated_result_hashes=tuple(
            value.result_hash for value in marked_rows if value.nondominated
        ),
        nondominated_physical_action_identity_hashes=action_image,
        selected_physical_action_identity_hash=selected,
        status=status,
        source_owned_resource_and_capacity_derivation=True,
        source_owned_registered_cell_action_quotient=source_owned_alias,
        exact_action_pair_alias_source_graph_joined=source_owned_alias,
        source_bound_population_loewner_geometry=source_bound_loewner,
        private_reference_only=not source_owned_alias,
        local_endpoint_action_image_premise_closed=status
        in {
            V1SampledV7PhaseARobustDecisionStatus.STABLE_PHYSICAL_ACTION_PREMISE,
            V1SampledV7PhaseARobustDecisionStatus.STABLE_NO_ACTION_PREMISE,
        },
        outside_shell_fixed_point_reference_hash=fixed_point_hash,
        outside_shell_fixed_point_reference_status=fixed_point_status,
        outside_shell_fixed_point_reference_certified=fixed_point_certified,
        inherited_alpha_ledger_hash=loewner_receipt.alpha_ledger_hash,
        inherited_alpha_coordinate_hash=loewner_receipt.alpha_coordinate_hash,
        inherited_alpha_lineage_hash=loewner_receipt.alpha_lineage_hash,
        inherited_familywise_alpha=loewner_receipt.inherited_familywise_alpha,
        inherited_alpha_spent=loewner_receipt.inherited_alpha_spent,
        _issuance_token=_RECEIPT_TOKEN,
    )


__all__ = (
    "SAMPLED_V7_PHASE_A_ROBUST_SHELL_DECISION_ALGORITHM",
    "SAMPLED_V7_PHASE_A_ROBUST_SHELL_DECISION_AUTHORITY",
    "SAMPLED_V7_PHASE_A_ROBUST_SHELL_DECISION_SCHEMA",
    "V1SampledV7PhaseAPrivateActionQuotient",
    "V1SampledV7PhaseAPrivateResourceCell",
    "V1SampledV7PhaseARobustBranchActionResult",
    "V1SampledV7PhaseARobustDecisionStatus",
    "V1SampledV7PhaseARobustPairStatus",
    "V1SampledV7PhaseARobustShellDecisionReceipt",
    "V1SampledV7PhaseARobustShellDecisionRefusal",
    "V1SampledV7PhaseARobustShellDecisionRefusalCode",
    "issue_v1_sampled_v7_phase_a_robust_shell_decision",
)
