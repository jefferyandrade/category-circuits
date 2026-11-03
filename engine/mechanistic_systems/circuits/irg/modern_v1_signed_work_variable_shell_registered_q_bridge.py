"""Conditional registered-Q custody for one preferred Phase-B hard action.

The variable-shell Phase-B solver obtains conservative alternate-cell lower
bounds by dropping Q.  Its preferred hard action still has to be replayed in
the *originating* registered ``q_rho`` or ``q_d`` branch before that asymmetric
argument is even conditionally well formed.  This module records that narrow
join.

The join is deliberately authority-inert.  Existing hard-F/Q endpoint and
branch receipts are public durable values rather than issuer capabilities and
do not themselves carry the variable-shell source/chart lineage.  We can bind
them exactly to the issuer-only action-coordinate custody and to the actual
packed parent/child delta, but this module cannot manufacture upstream source
custody, action invariance, or permission to enter C/K/O.  A later live
integration joint must consume this receipt together with the independently
closed action-separation theorem.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass
from enum import Enum
import hashlib
import math
from numbers import Integral, Real
from typing import Final, NoReturn

from mechanistic_systems.contracts.hard_mask_actions import HardMask

from .contracts import content_hash
from .modern_v1_candidate_composition import V1PackedRemovalAction
from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch
from .modern_v1_signed_work_hard_fq_replay_cuda import (
    V1SignedWorkHardFQBranchReceipt,
    V1SignedWorkHardFQEndpointReceipt,
    V1SignedWorkHardFQEndpointStatus,
)
from .modern_v1_signed_work_variable_shell_action_separation_cuda import (
    V1SignedWorkVariableShellActionCoordinateCustodyIssuance,
    V1SignedWorkVariableShellActionSubsetIdentity,
)
from .modern_v1_signed_work_variable_shell_source_cuda import (
    V1SignedWorkVariableShellSourceReceipt,
)


REGISTERED_Q_PREFERRED_HARD_ACTION_BRIDGE_SCHEMA: Final = (
    "partition_free_v1_variable_shell_registered_q_preferred_hard_action_"
    "conditional_bridge_v2"
)
REGISTERED_Q_PREFERRED_HARD_ACTION_BRIDGE_AUTHORITY: Final = (
    "conditional_origin_branch_registered_Q_preferred_hard_action_custody_"
    "only_no_action_invariance_production_C_K_O_commit_terminal_or_pareto_"
    "authority_v1"
)
REGISTERED_Q_PREFERRED_HARD_ACTION_RULE: Final = (
    "alternate_cell_lower_bounds_drop_Q_while_the_preferred_actual_hard_"
    "action_requires_strict_origin_branch_q_upper_below_registered_budget_"
    "and_D_W2_empirical_Q_never_substitutes_for_registered_q_rho_or_q_d_v1"
)

_PREFIX: Final = "partition-free-v1-variable-shell-registered-q-preferred-"
_ISSUER_TOKEN: Final = object()


class V1SignedWorkVariableShellRegisteredQBridgeRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    INVALID_ISSUER = "invalid_issuer"
    STALE_SOURCE = "stale_source"
    BRANCH_LINEAGE_MISMATCH = "branch_lineage_mismatch"
    ENDPOINT_NOT_REGISTERED = "endpoint_not_registered"
    ACTION_COORDINATE_MISMATCH = "action_coordinate_mismatch"
    PACKED_ACTION_MISMATCH = "packed_action_mismatch"
    REGISTERED_Q_UNSAFE_OR_BOUNDARY = "registered_q_unsafe_or_boundary"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SignedWorkVariableShellRegisteredQBridgeRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SignedWorkVariableShellRegisteredQBridgeRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SignedWorkVariableShellRegisteredQBridgeRefusalCode:
            raise TypeError("registered-Q bridge refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SignedWorkVariableShellRegisteredQBridgeRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SignedWorkVariableShellRegisteredQBridgeRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SignedWorkVariableShellRegisteredQBridgeRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SignedWorkVariableShellRegisteredQBridgeRefusalCode.INVALID_TYPE,
            f"{name} must be one positive integer",
        )
    result = int(value)
    if result <= 0:
        _refuse(
            V1SignedWorkVariableShellRegisteredQBridgeRefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SignedWorkVariableShellRegisteredQBridgeRefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SignedWorkVariableShellRegisteredQBridgeRefusalCode.INVALID_TYPE,
            f"{name} must be nonnegative",
        )
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1SignedWorkVariableShellRegisteredQBridgeRefusalCode.INVALID_TYPE,
            f"{name} must be one finite nonnegative scalar",
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        _refuse(
            V1SignedWorkVariableShellRegisteredQBridgeRefusalCode.INVALID_TYPE,
            f"{name} must be finite and nonnegative",
        )
    return result


def _q_component_hash(
    *,
    component: str,
    endpoint: V1SignedWorkHardFQEndpointReceipt,
    lower: float,
    nominal: float,
    upper: float,
) -> str:
    """Hash an action-specific response component, not a global matrix ID."""

    return content_hash(
        {
            "component": component,
            "endpoint_receipt_hash": endpoint.receipt_hash,
            "packed_action_hash": endpoint.packed_action_hash,
            "lower": lower,
            "nominal": nominal,
            "upper": upper,
        },
        prefix=_PREFIX + "action-response-component-",
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellRegisteredQPreferredHardActionBridgeReceipt:
    """Exact but conditional origin-branch Q-safe hard-action binding."""

    physical_action_coordinate_hash: str
    action_coordinate_custody_issuance_hash: str
    action_coordinate_family_receipt_hash: str
    action_subset_identity_hash: str
    source_receipt_hash: str
    source_replay_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    parent_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    shell_freeze_receipt_hash: str
    source_kkt_branch_id: str
    source_kkt_branch_receipt_hash: str
    branch: V1RelaxedKKTBranch
    hard_fq_problem_hash: str
    hard_fq_operator_receipt_hash: str
    hard_fq_branch_identity_receipt_hash: str
    hard_fq_branch_receipt_hash: str
    hard_fq_endpoint_receipt_hash: str
    endpoint_label: int
    removed_count: int
    packed_action_hash: str
    resulting_mask_hash: str
    resulting_mask_packed_sha256: str
    removed_route_index_hash: str
    registered_h_action_response_hash: str
    registered_d_action_response_hash: str
    registered_rho_action_response_hash: str
    registered_endpoint_signed_f_lower_diagnostic: float
    registered_endpoint_signed_f_nominal_diagnostic: float
    registered_endpoint_signed_f_upper_diagnostic: float
    origin_branch_q_upper: float
    quadratic_budget: float
    origin_branch_status: V1SignedWorkHardFQEndpointStatus
    canonical_min_status_diagnostic_only: V1SignedWorkHardFQEndpointStatus
    origin_branch_specific_q_used: bool = True
    canonical_min_q_used_for_admission: bool = False
    strict_q_budget_inequality_used: bool = True
    endpoint_registered_on_exact_branch_receipt: bool = True
    actual_parent_child_delta_verified: bool = True
    alternate_cell_lower_bound_Q_dropped: bool = True
    d_w2_empirical_q_silently_substituted_for_registered_branch: bool = False
    same_source_signed_f_authority: bool = False
    conditional_bridge_only: bool = True
    production_registered_q_bridge_authority: bool = False
    action_invariance_authority: bool = False
    c_stream_authority: bool = False
    k_stream_authority: bool = False
    o_stream_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    rule: str = REGISTERED_Q_PREFERRED_HARD_ACTION_RULE
    authority_scope: str = REGISTERED_Q_PREFERRED_HARD_ACTION_BRIDGE_AUTHORITY
    schema: str = REGISTERED_Q_PREFERRED_HARD_ACTION_BRIDGE_SCHEMA
    _issuer_token: InitVar[object] = None

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _ISSUER_TOKEN:
            _refuse(
                V1SignedWorkVariableShellRegisteredQBridgeRefusalCode.INVALID_ISSUER,
                "registered-Q preferred-action receipts are issuer-only",
            )
        for name in (
            "physical_action_coordinate_hash",
            "action_coordinate_custody_issuance_hash",
            "action_coordinate_family_receipt_hash",
            "action_subset_identity_hash",
            "source_receipt_hash",
            "source_replay_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "parent_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "registered_d_prefix_hash",
            "registered_d_w2_prefix_hash",
            "shell_freeze_receipt_hash",
            "source_kkt_branch_id",
            "source_kkt_branch_receipt_hash",
            "hard_fq_problem_hash",
            "hard_fq_operator_receipt_hash",
            "hard_fq_branch_identity_receipt_hash",
            "hard_fq_branch_receipt_hash",
            "hard_fq_endpoint_receipt_hash",
            "packed_action_hash",
            "resulting_mask_hash",
            "resulting_mask_packed_sha256",
            "removed_route_index_hash",
            "registered_h_action_response_hash",
            "registered_d_action_response_hash",
            "registered_rho_action_response_hash",
            "rule",
            "authority_scope",
        ):
            _identity(getattr(self, name), name=name)
        if len(self.resulting_mask_packed_sha256) != 64:
            _refuse(
                V1SignedWorkVariableShellRegisteredQBridgeRefusalCode.INVALID_TYPE,
                "packed child digest must be one SHA-256 hex digest",
            )
        try:
            int(self.resulting_mask_packed_sha256, 16)
        except ValueError:
            _refuse(
                V1SignedWorkVariableShellRegisteredQBridgeRefusalCode.INVALID_TYPE,
                "packed child digest must be hexadecimal",
            )
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        for name in ("route_count", "endpoint_label", "removed_count"):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "origin_branch_q_upper",
            _finite_nonnegative(
                self.origin_branch_q_upper,
                name="origin branch Q upper",
            ),
        )
        object.__setattr__(
            self,
            "quadratic_budget",
            _finite_nonnegative(self.quadratic_budget, name="quadratic budget"),
        )
        for name in (
            "registered_endpoint_signed_f_lower_diagnostic",
            "registered_endpoint_signed_f_nominal_diagnostic",
            "registered_endpoint_signed_f_upper_diagnostic",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name=name),
            )
        if not (
            self.registered_endpoint_signed_f_lower_diagnostic
            <= self.registered_endpoint_signed_f_nominal_diagnostic
            <= self.registered_endpoint_signed_f_upper_diagnostic
        ):
            _refuse(
                V1SignedWorkVariableShellRegisteredQBridgeRefusalCode.INVALID_TYPE,
                "preferred hard-action signed-F interval is not ordered",
            )
        if type(self.branch) is not V1RelaxedKKTBranch:
            _refuse(
                V1SignedWorkVariableShellRegisteredQBridgeRefusalCode.INVALID_TYPE,
                "registered-Q bridge branch must be exact",
            )
        if (
            type(self.origin_branch_status) is not V1SignedWorkHardFQEndpointStatus
            or type(self.canonical_min_status_diagnostic_only)
            is not V1SignedWorkHardFQEndpointStatus
        ):
            _refuse(
                V1SignedWorkVariableShellRegisteredQBridgeRefusalCode.INVALID_TYPE,
                "registered-Q bridge statuses must be exact",
            )
        if (
            self.origin_branch_status
            is not V1SignedWorkHardFQEndpointStatus.Q_SAFE_FEASIBLE
            or not self.origin_branch_q_upper < self.quadratic_budget
        ):
            _refuse(
                V1SignedWorkVariableShellRegisteredQBridgeRefusalCode.REGISTERED_Q_UNSAFE_OR_BOUNDARY,
                "preferred hard action is not strictly safe in its origin branch",
            )
        exact_flags = (
            self.origin_branch_specific_q_used,
            self.canonical_min_q_used_for_admission,
            self.strict_q_budget_inequality_used,
            self.endpoint_registered_on_exact_branch_receipt,
            self.actual_parent_child_delta_verified,
            self.alternate_cell_lower_bound_Q_dropped,
            self.d_w2_empirical_q_silently_substituted_for_registered_branch,
            self.same_source_signed_f_authority,
            self.conditional_bridge_only,
            self.production_registered_q_bridge_authority,
            self.action_invariance_authority,
            self.c_stream_authority,
            self.k_stream_authority,
            self.o_stream_authority,
            self.commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in exact_flags) or exact_flags != (
            True,
            False,
            True,
            True,
            True,
            True,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ):
            _refuse(
                V1SignedWorkVariableShellRegisteredQBridgeRefusalCode.AUTHORITY_VIOLATION,
                "registered-Q bridge changed its conditional authority boundary",
            )
        if (
            self.source_kkt_branch_id != self.branch.value
            or self.rule != REGISTERED_Q_PREFERRED_HARD_ACTION_RULE
            or self.authority_scope
            != REGISTERED_Q_PREFERRED_HARD_ACTION_BRIDGE_AUTHORITY
            or self.schema != REGISTERED_Q_PREFERRED_HARD_ACTION_BRIDGE_SCHEMA
        ):
            _refuse(
                V1SignedWorkVariableShellRegisteredQBridgeRefusalCode.BRANCH_LINEAGE_MISMATCH,
                "registered-Q bridge branch or semantics changed",
            )

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
        }

    @property
    def receipt_hash(self) -> str:
        return content_hash(self._payload(), prefix=_PREFIX + "bridge-receipt-")


def issue_v1_signed_work_variable_shell_registered_q_preferred_hard_action_bridge(
    *,
    source_receipt: V1SignedWorkVariableShellSourceReceipt,
    action_coordinate_custody: V1SignedWorkVariableShellActionCoordinateCustodyIssuance,
    action_subset_identity: V1SignedWorkVariableShellActionSubsetIdentity,
    hard_fq_branch_receipt: V1SignedWorkHardFQBranchReceipt,
    hard_fq_endpoint: V1SignedWorkHardFQEndpointReceipt,
    parent_mask: HardMask,
    packed_action: V1PackedRemovalAction,
) -> V1SignedWorkVariableShellRegisteredQPreferredHardActionBridgeReceipt:
    """Issue the maximal sound conditional registered-Q/action binding.

    The packed parent/child scan is bounded in workspace by the canonical mask
    contract.  It validates the actual route delta without retaining a
    route-scale index vector.
    """

    if type(source_receipt) is not V1SignedWorkVariableShellSourceReceipt:
        _refuse(
            V1SignedWorkVariableShellRegisteredQBridgeRefusalCode.INVALID_TYPE,
            "source receipt must be exact",
        )
    if type(action_coordinate_custody) is not (
        V1SignedWorkVariableShellActionCoordinateCustodyIssuance
    ):
        _refuse(
            V1SignedWorkVariableShellRegisteredQBridgeRefusalCode.INVALID_TYPE,
            "action-coordinate custody must be exact",
        )
    if type(action_subset_identity) is not V1SignedWorkVariableShellActionSubsetIdentity:
        _refuse(
            V1SignedWorkVariableShellRegisteredQBridgeRefusalCode.INVALID_TYPE,
            "action subset identity must be exact",
        )
    if type(hard_fq_branch_receipt) is not V1SignedWorkHardFQBranchReceipt:
        _refuse(
            V1SignedWorkVariableShellRegisteredQBridgeRefusalCode.INVALID_TYPE,
            "hard-F/Q branch receipt must be exact",
        )
    if type(hard_fq_endpoint) is not V1SignedWorkHardFQEndpointReceipt:
        _refuse(
            V1SignedWorkVariableShellRegisteredQBridgeRefusalCode.INVALID_TYPE,
            "hard-F/Q endpoint must be exact",
        )
    if type(parent_mask) is not HardMask or type(packed_action) is not V1PackedRemovalAction:
        _refuse(
            V1SignedWorkVariableShellRegisteredQBridgeRefusalCode.INVALID_TYPE,
            "preferred hard action requires exact parent and packed child objects",
        )

    # Re-run the durable upstream replay invariants at the consumption seam.
    hard_fq_endpoint.__post_init__()
    hard_fq_branch_receipt.__post_init__()

    family = action_coordinate_custody.family_receipt
    if (
        family.source_receipt_hash != source_receipt.receipt_hash
        or family.source_replay_hash != source_receipt.source_replay_hash
        or family.parent_mask_hash != source_receipt.current_mask_hash
        or family.route_axis_hash != source_receipt.route_axis_hash
        or family.route_id_axis_hash != source_receipt.route_id_axis_hash
        or family.route_count != source_receipt.route_count
        or family.shell_freeze_receipt_hash != source_receipt.shell_freeze_receipt_hash
    ):
        _refuse(
            V1SignedWorkVariableShellRegisteredQBridgeRefusalCode.STALE_SOURCE,
            "action-coordinate custody and variable-shell source are not one chart",
        )
    subset_matches = tuple(
        value
        for value in action_coordinate_custody.subset_identities
        if value.subset_hash == action_subset_identity.subset_hash
    )
    if len(subset_matches) != 1 or subset_matches[0] != action_subset_identity:
        _refuse(
            V1SignedWorkVariableShellRegisteredQBridgeRefusalCode.ACTION_COORDINATE_MISMATCH,
            "preferred action is not one exact issuer-owned shell coordinate",
        )

    if (
        parent_mask.mask_hash != source_receipt.current_mask_hash
        or parent_mask.route_axis_hash != source_receipt.route_axis_hash
        or parent_mask.route_count != source_receipt.route_count
        or packed_action.resulting_mask.route_axis_hash != source_receipt.route_axis_hash
        or packed_action.resulting_mask.route_count != source_receipt.route_count
    ):
        _refuse(
            V1SignedWorkVariableShellRegisteredQBridgeRefusalCode.STALE_SOURCE,
            "actual packed parent/child masks do not share the source route chart",
        )
    try:
        removed_count, removed_hash = (
            packed_action.resulting_mask.removed_index_count_and_hash_from(parent_mask)
        )
    except ValueError as error:
        _refuse(
            V1SignedWorkVariableShellRegisteredQBridgeRefusalCode.PACKED_ACTION_MISMATCH,
            f"actual packed action is not an irreversible parent-to-child delta: {error}",
        )
    if (
        removed_count != packed_action.removed_count
        or removed_hash != packed_action.removed_route_index_hash
        or action_subset_identity.parent_mask_hash != parent_mask.mask_hash
        or action_subset_identity.removed_count != removed_count
        or action_subset_identity.removed_route_index_hash != removed_hash
    ):
        _refuse(
            V1SignedWorkVariableShellRegisteredQBridgeRefusalCode.ACTION_COORDINATE_MISMATCH,
            "actual packed action differs from its streamed physical coordinate",
        )

    endpoint_matches = tuple(
        value
        for value in hard_fq_branch_receipt.endpoints
        if value.endpoint_label == hard_fq_endpoint.endpoint_label
        and value.receipt_hash == hard_fq_endpoint.receipt_hash
    )
    if len(endpoint_matches) != 1 or endpoint_matches[0] != hard_fq_endpoint:
        _refuse(
            V1SignedWorkVariableShellRegisteredQBridgeRefusalCode.ENDPOINT_NOT_REGISTERED,
            "hard-F/Q endpoint is not the exact registered origin-branch endpoint",
        )
    if (
        hard_fq_branch_receipt.branch is not hard_fq_endpoint.branch
        or source_receipt.kkt_branch_id != hard_fq_endpoint.branch.value
        or source_receipt.kkt_branch_source_receipt_hash
        != hard_fq_branch_receipt.branch_identity_receipt_hash
    ):
        _refuse(
            V1SignedWorkVariableShellRegisteredQBridgeRefusalCode.BRANCH_LINEAGE_MISMATCH,
            "source, registered branch identity, and hard endpoint do not close",
        )
    if (
        hard_fq_endpoint.cumulative_removed_count != removed_count
        or hard_fq_endpoint.packed_action_hash != packed_action.action_hash
        or hard_fq_endpoint.resulting_mask_hash
        != packed_action.resulting_mask.mask_hash
        or hard_fq_endpoint.removed_route_index_hash != removed_hash
    ):
        _refuse(
            V1SignedWorkVariableShellRegisteredQBridgeRefusalCode.PACKED_ACTION_MISMATCH,
            "registered hard-Q endpoint differs from the actual packed action",
        )
    if (
        hard_fq_endpoint.origin_branch_status
        is not V1SignedWorkHardFQEndpointStatus.Q_SAFE_FEASIBLE
        or not hard_fq_endpoint.branch_q_upper < hard_fq_endpoint.quadratic_budget
    ):
        _refuse(
            V1SignedWorkVariableShellRegisteredQBridgeRefusalCode.REGISTERED_Q_UNSAFE_OR_BOUNDARY,
            "origin-branch Q upper must be strictly below its registered budget",
        )

    return V1SignedWorkVariableShellRegisteredQPreferredHardActionBridgeReceipt(
        physical_action_coordinate_hash=(
            action_subset_identity.physical_action_coordinate_hash
        ),
        action_coordinate_custody_issuance_hash=(
            action_coordinate_custody.issuance_hash
        ),
        action_coordinate_family_receipt_hash=family.receipt_hash,
        action_subset_identity_hash=action_subset_identity.subset_hash,
        source_receipt_hash=source_receipt.receipt_hash,
        source_replay_hash=source_receipt.source_replay_hash,
        model_identity_hash=source_receipt.model_identity_hash,
        chart_identity_hash=source_receipt.chart_identity_hash,
        parent_mask_hash=source_receipt.current_mask_hash,
        proposal_generation=source_receipt.proposal_generation,
        route_axis_hash=source_receipt.route_axis_hash,
        route_id_axis_hash=source_receipt.route_id_axis_hash,
        route_count=source_receipt.route_count,
        registered_d_prefix_hash=source_receipt.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=source_receipt.registered_d_w2_prefix_hash,
        shell_freeze_receipt_hash=source_receipt.shell_freeze_receipt_hash,
        source_kkt_branch_id=source_receipt.kkt_branch_id,
        source_kkt_branch_receipt_hash=(
            source_receipt.kkt_branch_source_receipt_hash
        ),
        branch=hard_fq_endpoint.branch,
        hard_fq_problem_hash=hard_fq_branch_receipt.problem_hash,
        hard_fq_operator_receipt_hash=hard_fq_branch_receipt.operator_receipt_hash,
        hard_fq_branch_identity_receipt_hash=(
            hard_fq_branch_receipt.branch_identity_receipt_hash
        ),
        hard_fq_branch_receipt_hash=hard_fq_branch_receipt.receipt_hash,
        hard_fq_endpoint_receipt_hash=hard_fq_endpoint.receipt_hash,
        endpoint_label=hard_fq_endpoint.endpoint_label,
        removed_count=removed_count,
        packed_action_hash=packed_action.action_hash,
        resulting_mask_hash=packed_action.resulting_mask.mask_hash,
        resulting_mask_packed_sha256=hashlib.sha256(
            packed_action.resulting_mask.packed_bits
        ).hexdigest(),
        removed_route_index_hash=removed_hash,
        registered_h_action_response_hash=_q_component_hash(
            component="low_rank_H_action_response",
            endpoint=hard_fq_endpoint,
            lower=hard_fq_endpoint.low_rank_q_lower,
            nominal=hard_fq_endpoint.low_rank_q_nominal,
            upper=hard_fq_endpoint.low_rank_q_upper,
        ),
        registered_d_action_response_hash=_q_component_hash(
            component="diagonal_d_action_response",
            endpoint=hard_fq_endpoint,
            lower=hard_fq_endpoint.diagonal_residual_q_lower,
            nominal=hard_fq_endpoint.diagonal_residual_q_nominal,
            upper=hard_fq_endpoint.diagonal_residual_q_upper,
        ),
        registered_rho_action_response_hash=_q_component_hash(
            component="spectral_rho_action_response",
            endpoint=hard_fq_endpoint,
            lower=hard_fq_endpoint.spectral_residual_q_lower,
            nominal=hard_fq_endpoint.spectral_residual_q_nominal,
            upper=hard_fq_endpoint.spectral_residual_q_upper,
        ),
        registered_endpoint_signed_f_lower_diagnostic=(
            hard_fq_endpoint.signed_f_lower
        ),
        registered_endpoint_signed_f_nominal_diagnostic=(
            hard_fq_endpoint.signed_f_nominal
        ),
        registered_endpoint_signed_f_upper_diagnostic=(
            hard_fq_endpoint.signed_f_upper
        ),
        origin_branch_q_upper=hard_fq_endpoint.branch_q_upper,
        quadratic_budget=hard_fq_endpoint.quadratic_budget,
        origin_branch_status=hard_fq_endpoint.origin_branch_status,
        canonical_min_status_diagnostic_only=hard_fq_endpoint.status,
        _issuer_token=_ISSUER_TOKEN,
    )


__all__ = [
    "REGISTERED_Q_PREFERRED_HARD_ACTION_BRIDGE_AUTHORITY",
    "REGISTERED_Q_PREFERRED_HARD_ACTION_BRIDGE_SCHEMA",
    "REGISTERED_Q_PREFERRED_HARD_ACTION_RULE",
    "V1SignedWorkVariableShellRegisteredQBridgeRefusal",
    "V1SignedWorkVariableShellRegisteredQBridgeRefusalCode",
    "V1SignedWorkVariableShellRegisteredQPreferredHardActionBridgeReceipt",
    "issue_v1_signed_work_variable_shell_registered_q_preferred_hard_action_bridge",
]
