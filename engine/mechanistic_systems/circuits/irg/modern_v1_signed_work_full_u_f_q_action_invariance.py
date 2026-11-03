"""Canonical D-side conjunction for one physical v1 pruning action.

This module is intentionally a *join*, not another estimator.  Production C
may open only after the same packed irreversible action has all three of the
following source-bound witnesses:

* strict full-population corrected-work separation over the complete hard
  alternate domain;
* strict registered hard-Q safety in the action's originating convex branch;
* literal-J FD replay whose complete two-branch decision is invariant under
  the registered deterministic residual/rank uncertainty.

The older rank-local ``C_stream_open_authority`` bit is deliberately ignored.
It does not know about the fresh full-U corrected-work witness or the exact
origin-branch hard-Q endpoint.  The issuer below forms their D-side
conjunction only.  Its sibling readiness receipt still keeps C closed: a
source-bound H200 completion and an independently frozen physical-C
registration must be joined later.  K, O, mask commit, terminal, and Pareto
authority remain closed.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
import hashlib
import math
from numbers import Integral, Real
from typing import Final, NoReturn, Sequence

from mechanistic_systems.contracts.hard_mask_actions import HardMask

from .contracts import content_hash
from .modern_v1_candidate_composition import V1PackedRemovalAction
from .modern_v1_fqr_adaptive_rank import (
    FQR_ADAPTIVE_RANK_DECISION_CERTIFICATE_V2_SCHEMA,
    FQR_ADAPTIVE_RANK_STABILITY_V2_SCHEMA,
    V1FQRAdaptiveRankReplayLedger,
    V1FQRAdaptiveRankStabilityCertificate,
)
from .modern_v1_sampled_literal_j_fd_rank_replay import (
    V1SampledLiteralJFDRankReplayReceipt,
    V1SampledLiteralJOriginalCaptureBinding,
)
from .modern_v1_signed_work_block_support_witness import (
    V1BlockSupportWitnessDecision,
    V1BlockSupportWitnessRegistration,
    V1BlockSupportWitnessStatus,
)
from .modern_v1_signed_work_hard_fq_replay_cuda import (
    V1SignedWorkHardFQEndpointStatus,
)
from .modern_v1_signed_work_variable_shell_registered_q_bridge import (
    V1SignedWorkVariableShellRegisteredQPreferredHardActionBridgeReceipt,
)


FULL_U_F_Q_ACTION_DOMAIN_SCHEMA: Final = (
    "partition_free_v1_full_U_corrected_F_literal_J_Q_complete_hard_action_domain_v2"
)
FULL_U_F_Q_ACTION_INVARIANCE_SCHEMA: Final = (
    "partition_free_v1_full_U_corrected_F_origin_Q_literal_J_action_invariance_v2"
)
FULL_U_F_Q_PRE_C_READINESS_SCHEMA: Final = (
    "partition_free_v1_full_U_F_Q_action_invariance_pre_C_readiness_v1"
)
FULL_U_F_Q_ACTION_INVARIANCE_RULE: Final = (
    "strict_full_U_corrected_F_separation_and_strict_origin_branch_registered_"
    "hard_Q_safety_and_literal_J_FD_complete_two_branch_residual_decision_"
    "stability_for_one_exact_packed_irreversible_action_v2"
)
FULL_U_F_Q_ACTION_INVARIANCE_SCOPE: Final = (
    "physical_D_action_invariance_only_no_C_K_O_commit_terminal_or_pareto_authority_v1"
)
FULL_U_F_Q_PRE_C_READINESS_SCOPE: Final = (
    "D_side_action_invariance_ready_for_future_source_bound_H200_and_"
    "independent_C_registration_join_no_C_K_O_commit_terminal_or_pareto_"
    "authority_v1"
)

_PREFIX: Final = "partition-free-v1-full-U-F-Q-action-invariance-"
_INVARIANCE_TOKEN: Final = object()
_ADMISSION_TOKEN: Final = object()
_BRANCH_AXIS: Final = ("spectral", "diagonal_l1")


class V1SignedWorkFullUFQActionInvarianceRefusalCode(str, Enum):
    """Stable refusal reasons for the canonical D-side conjunction."""

    INVALID_TYPE = "invalid_type"
    RECEIPT_INTEGRITY_FAILURE = "receipt_integrity_failure"
    STRICT_FULL_U_F_SEPARATION_REQUIRED = "strict_full_U_F_separation_required"
    STRICT_ORIGIN_BRANCH_Q_SAFETY_REQUIRED = "strict_origin_branch_Q_safety_required"
    LITERAL_J_REPLAY_REQUIRED = "literal_J_replay_required"
    FD_RANK_STABILITY_REQUIRED = "FD_rank_stability_required"
    INCOMPLETE_ALTERNATE_DOMAIN = "incomplete_alternate_domain"
    PHYSICAL_ACTION_MISMATCH = "physical_action_mismatch"
    CHART_OR_GENERATION_MISMATCH = "chart_or_generation_mismatch"
    D_PREFIX_MISMATCH = "D_prefix_mismatch"
    BRANCH_OR_ALIAS_MISMATCH = "branch_or_alias_mismatch"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SignedWorkFullUFQActionInvarianceRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SignedWorkFullUFQActionInvarianceRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SignedWorkFullUFQActionInvarianceRefusalCode:
            raise TypeError("full-U F/Q refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SignedWorkFullUFQActionInvarianceRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SignedWorkFullUFQActionInvarianceRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SignedWorkFullUFQActionInvarianceRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        _refuse(
            V1SignedWorkFullUFQActionInvarianceRefusalCode.INVALID_TYPE,
            f"{name} must be one positive integer",
        )
    return int(value)


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        _refuse(
            V1SignedWorkFullUFQActionInvarianceRefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative integer",
        )
    return int(value)


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1SignedWorkFullUFQActionInvarianceRefusalCode.INVALID_TYPE,
            f"{name} must be one finite nonnegative scalar",
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        _refuse(
            V1SignedWorkFullUFQActionInvarianceRefusalCode.INVALID_TYPE,
            f"{name} must be finite and nonnegative",
        )
    return result


def _axis(values: Sequence[str], *, name: str, unique: bool = True) -> tuple[str, ...]:
    result = tuple(_identity(value, name=f"{name} item") for value in values)
    if not result or (unique and len(set(result)) != len(result)):
        _refuse(
            V1SignedWorkFullUFQActionInvarianceRefusalCode.INVALID_TYPE,
            f"{name} must be nonempty" + (" and unique" if unique else ""),
        )
    return result


def _assert_integrity(value: object, *, owner: str) -> str:
    method = getattr(value, "assert_integrity", None)
    if not callable(method):
        _refuse(
            V1SignedWorkFullUFQActionInvarianceRefusalCode.INVALID_TYPE,
            f"{owner} lacks its canonical integrity method",
        )
    try:
        observed = method()
    except (
        ArithmeticError,
        PermissionError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        _refuse(
            V1SignedWorkFullUFQActionInvarianceRefusalCode.RECEIPT_INTEGRITY_FAILURE,
            f"{owner} failed replay: {error}",
        )
    return _identity(observed, name=f"{owner} integrity hash")


def _fresh_mask_hash(mask: HardMask, *, name: str) -> str:
    """Replay live packed bytes rather than trusting a cached mask property."""

    if type(mask) is not HardMask:
        _refuse(
            V1SignedWorkFullUFQActionInvarianceRefusalCode.INVALID_TYPE,
            f"{name} must use the exact HardMask type",
        )
    try:
        snapshot = HardMask(
            route_axis_hash=mask.route_axis_hash,
            route_count=mask.route_count,
            packed_bits=bytes(mask.packed_bits),
        )
    except (TypeError, ValueError) as error:
        _refuse(
            V1SignedWorkFullUFQActionInvarianceRefusalCode.PHYSICAL_ACTION_MISMATCH,
            f"{name} packed content is malformed: {error}",
        )
    return snapshot.mask_hash


def _required_false(values: Sequence[object], *, owner: str) -> None:
    if any(type(value) is not bool or value for value in values):
        _refuse(
            V1SignedWorkFullUFQActionInvarianceRefusalCode.AUTHORITY_VIOLATION,
            f"{owner} crossed its downstream authority boundary",
        )


def v1_full_u_f_q_complete_alternate_domain_hash(
    *,
    current_mask_hash: str,
    route_axis_hash: str,
    route_count: int,
    proposal_generation: int,
    frozen_rank: int,
    branch_action_axes: Sequence[tuple[str, Sequence[str]]],
    preferred_packed_action_hash: str,
    origin_branch: str,
) -> str:
    """Canonical crosswalk from complete rank-local ladders to F alternates.

    Physical masks are deduplicated across the two convex residual branches.
    Branch/action incidence remains in the commitment so a caller cannot turn
    a one-branch scan into an asserted two-branch union.
    """

    preferred = _identity(
        preferred_packed_action_hash,
        name="preferred packed action hash",
    )
    origin = _identity(origin_branch, name="origin branch")
    if origin not in _BRANCH_AXIS:
        _refuse(
            V1SignedWorkFullUFQActionInvarianceRefusalCode.INCOMPLETE_ALTERNATE_DOMAIN,
            "origin branch is outside the canonical two-branch axis",
        )
    branches = tuple(branch_action_axes)
    if len(branches) != 2 or tuple(value[0] for value in branches) != _BRANCH_AXIS:
        _refuse(
            V1SignedWorkFullUFQActionInvarianceRefusalCode.INCOMPLETE_ALTERNATE_DOMAIN,
            "complete action coverage requires spectral then diagonal-l1 axes",
        )
    normalized: list[tuple[str, tuple[str, ...]]] = []
    for branch, actions in branches:
        axis = _axis(actions, name=f"{branch} hard-action axis")
        if branch == origin and preferred not in axis:
            _refuse(
                V1SignedWorkFullUFQActionInvarianceRefusalCode.INCOMPLETE_ALTERNATE_DOMAIN,
                "preferred physical action is absent from its registered origin ladder",
            )
        normalized.append((branch, axis))
    alternates = tuple(
        sorted(
            {
                action
                for _branch, actions in normalized
                for action in actions
                if action != preferred
            }
        )
    )
    if not alternates:
        _refuse(
            V1SignedWorkFullUFQActionInvarianceRefusalCode.INCOMPLETE_ALTERNATE_DOMAIN,
            "strict separation requires at least one physical alternate",
        )
    return content_hash(
        {
            "schema": FULL_U_F_Q_ACTION_DOMAIN_SCHEMA,
            "current_mask_hash": _identity(current_mask_hash, name="current mask hash"),
            "route_axis_hash": _identity(route_axis_hash, name="route axis hash"),
            "route_count": _positive_int(route_count, name="route count"),
            "proposal_generation": _nonnegative_int(
                proposal_generation,
                name="proposal generation",
            ),
            "frozen_rank": _positive_int(frozen_rank, name="frozen FD rank"),
            "branch_action_axes": tuple(normalized),
            "preferred_packed_action_hash": preferred,
            "origin_branch": origin,
            "deduplicated_alternate_action_hashes": alternates,
            "complete_two_branch_hard_ladders": True,
            "physical_mask_deduplication": True,
        },
        prefix=_PREFIX + "alternate-domain-",
    )


def v1_full_u_f_q_complete_domain_crosswalk_receipt_hash(
    *,
    preferred_packed_action_hash: str,
    complete_alternate_domain_hash: str,
    alternate_action_count: int,
    alias_binding_hashes: Sequence[str],
    complete_constraint_union_hash: str,
    shared_action_minimum_domain_hash: str,
    frozen_rank: int,
) -> str:
    """Commit the physical action domain and every F solver alias jointly."""

    return content_hash(
        {
            "schema": (
                "partition_free_v1_full_U_F_Q_physical_domain_alias_"
                "crosswalk_receipt_v2"
            ),
            "preferred_packed_action_hash": _identity(
                preferred_packed_action_hash,
                name="preferred packed action hash",
            ),
            "complete_alternate_domain_hash": _identity(
                complete_alternate_domain_hash,
                name="complete alternate domain hash",
            ),
            "alternate_action_count": _positive_int(
                alternate_action_count,
                name="alternate action count",
            ),
            "alias_binding_hashes": _axis(
                alias_binding_hashes,
                name="domain crosswalk alias hashes",
            ),
            "complete_constraint_union_hash": _identity(
                complete_constraint_union_hash,
                name="complete constraint union hash",
            ),
            "shared_action_minimum_domain_hash": _identity(
                shared_action_minimum_domain_hash,
                name="shared-action minimum domain hash",
            ),
            "frozen_rank": _positive_int(frozen_rank, name="frozen FD rank"),
            "one_shared_action_domain_for_every_alias": True,
            "physical_action_domain_deduplicated_across_branches": True,
        },
        prefix=_PREFIX + "domain-crosswalk-receipt-",
    )


def _alternate_axis(
    *,
    branches: Sequence[object],
    preferred_action_hash: str,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                action
                for branch in branches
                for action in branch.ordered_packed_action_hashes
                if action != preferred_action_hash
            }
        )
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkFullUFQActionInvarianceReceipt:
    """Issuer-only proof that D freezes one complete-domain physical action."""

    packed_action_hash: str
    resulting_mask_hash: str
    resulting_mask_packed_sha256: str
    removed_route_index_hash: str
    removed_count: int
    physical_action_coordinate_hash: str
    action_coordinate_custody_issuance_hash: str
    action_coordinate_family_receipt_hash: str
    action_subset_identity_hash: str
    parent_mask_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    corrected_a_stream_identity_hash: str
    literal_j_stream_identity_hash: str
    original_capture_binding_hash: str
    literal_j_rank_replay_receipt_hash: str
    literal_j_fd_design_hash: str
    literal_j_fd_snapshot_hash: str
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    block_support_registration_hash: str
    block_support_decision_hash: str
    block_support_state_hash: str
    block_support_receipt_chain_hash: str
    full_axis_support_plane_source_hash: str
    native_fiber_axis_hash: str
    full_population_axis_hash: str
    full_population_count: int
    complete_alternate_domain_hash: str
    complete_alternate_domain_receipt_hash: str
    deduplicated_alternate_action_hashes: tuple[str, ...]
    alternate_action_count: int
    complete_constraint_union_hash: str
    shared_action_minimum_domain_hash: str
    alias_axis_hash: str
    alias_binding_hashes: tuple[str, ...]
    alias_branch_rank_resource_source_cardinality_axis: tuple[
        tuple[str, int, str, str, str, str, int], ...
    ]
    origin_branch: str
    origin_branch_source_receipt_hash: str
    origin_branch_source_replay_hash: str
    origin_branch_source_identity_receipt_hash: str
    shell_freeze_receipt_hash: str
    registered_q_bridge_receipt_hash: str
    origin_branch_hard_fq_problem_hash: str
    origin_branch_hard_fq_operator_receipt_hash: str
    origin_branch_hard_fq_receipt_hash: str
    origin_branch_hard_fq_endpoint_receipt_hash: str
    origin_branch_q_upper: float
    quadratic_budget: float
    adaptive_rank_policy_hash: str
    adaptive_rank_uncertainty_set_hash: str
    adaptive_rank_ledger_hash: str
    adaptive_rank_stability_certificate_hash: str
    adaptive_rank_decision_certificate_hash: str
    frozen_replay_record_hash: str
    sampled_geometry_source_hash: str
    kkt_problem_hash: str
    kkt_result_hash: str
    frozen_rank: int
    executed_rank_prefix: tuple[int, ...]
    branch_action_axes: tuple[tuple[str, tuple[str, ...]], ...]
    branch_selected_action_hashes: tuple[tuple[str, str], ...]
    source_pre_c_branch_family_hash: str
    robust_branch_family_hash: str
    receipt_hash: str = field(init=False)
    _issuer_token: InitVar[object] = None
    strict_full_population_corrected_F_separation: bool = True
    strict_origin_branch_registered_Q_safety: bool = True
    literal_J_same_capture_response_geometry: bool = True
    complete_two_branch_hard_domain_replayed: bool = True
    deterministic_FD_residual_decision_stable: bool = True
    complete_physical_alternate_domain_crosswalk: bool = True
    # This receipt proves the mathematical/source-identity conjunction carried
    # by its typed inputs.  The block controller deliberately does not retain
    # production CUDA execution objects, so production source custody must be
    # established by a later sibling join rather than inferred here.
    physical_action_invariance_authority: bool = True
    source_bound_production_solver_evidence_present: bool = False
    current_source_h200_completion_present: bool = False
    production_D_action_invariance_authority: bool = False
    legacy_rank_only_C_bit_used_as_authority: bool = False
    C_stream_open_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    array_free_receipt: bool = True
    rule: str = FULL_U_F_Q_ACTION_INVARIANCE_RULE
    authority_scope: str = FULL_U_F_Q_ACTION_INVARIANCE_SCOPE
    schema: str = FULL_U_F_Q_ACTION_INVARIANCE_SCHEMA

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _INVARIANCE_TOKEN:
            raise TypeError("full-U F/Q action-invariance receipts are issuer-only")
        for item in fields(type(self)):
            if item.name.endswith("_hash") and item.name != "receipt_hash":
                object.__setattr__(
                    self,
                    item.name,
                    _identity(getattr(self, item.name), name=item.name),
                )
        object.__setattr__(
            self,
            "removed_count",
            _positive_int(self.removed_count, name="removed route count"),
        )
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="route count"),
        )
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        object.__setattr__(
            self,
            "full_population_count",
            _positive_int(self.full_population_count, name="full population count"),
        )
        object.__setattr__(
            self,
            "alternate_action_count",
            _positive_int(self.alternate_action_count, name="alternate action count"),
        )
        object.__setattr__(
            self,
            "frozen_rank",
            _positive_int(self.frozen_rank, name="frozen FD rank"),
        )
        ranks = tuple(
            _positive_int(value, name="executed FD rank")
            for value in self.executed_rank_prefix
        )
        if not ranks or ranks[-1] != self.frozen_rank:
            _refuse(
                V1SignedWorkFullUFQActionInvarianceRefusalCode.FD_RANK_STABILITY_REQUIRED,
                "frozen rank is not the terminal executed rank",
            )
        object.__setattr__(self, "executed_rank_prefix", ranks)
        alternates = _axis(
            self.deduplicated_alternate_action_hashes,
            name="deduplicated alternate action hashes",
        )
        if len(alternates) != self.alternate_action_count:
            _refuse(
                V1SignedWorkFullUFQActionInvarianceRefusalCode.INCOMPLETE_ALTERNATE_DOMAIN,
                "alternate count differs from the complete physical action axis",
            )
        object.__setattr__(self, "deduplicated_alternate_action_hashes", alternates)
        branch_axes = tuple(self.branch_action_axes)
        selected = tuple(self.branch_selected_action_hashes)
        if (
            tuple(value[0] for value in branch_axes) != _BRANCH_AXIS
            or tuple(value[0] for value in selected) != _BRANCH_AXIS
            or any(
                selected_action not in dict(branch_axes)[branch]
                for branch, selected_action in selected
            )
            or self.origin_branch not in dict(branch_axes)
            or self.packed_action_hash not in dict(branch_axes)[self.origin_branch]
        ):
            _refuse(
                V1SignedWorkFullUFQActionInvarianceRefusalCode.FD_RANK_STABILITY_REQUIRED,
                "the stable branch decisions or origin action are absent from their ladders",
            )
        object.__setattr__(self, "branch_action_axes", branch_axes)
        object.__setattr__(self, "branch_selected_action_hashes", selected)
        origin = _identity(self.origin_branch, name="origin branch")
        if origin not in _BRANCH_AXIS:
            _refuse(
                V1SignedWorkFullUFQActionInvarianceRefusalCode.BRANCH_OR_ALIAS_MISMATCH,
                "receipt origin branch is outside the canonical branch axis",
            )
        object.__setattr__(self, "origin_branch", origin)
        aliases = tuple(self.alias_branch_rank_resource_source_cardinality_axis)
        alias_hashes = _axis(self.alias_binding_hashes, name="alias binding hashes")
        if not aliases or len(aliases) != len(alias_hashes):
            _refuse(
                V1SignedWorkFullUFQActionInvarianceRefusalCode.BRANCH_OR_ALIAS_MISMATCH,
                "alias identity and hash axes differ",
            )
        object.__setattr__(
            self,
            "alias_branch_rank_resource_source_cardinality_axis",
            aliases,
        )
        object.__setattr__(self, "alias_binding_hashes", alias_hashes)
        if len(self.resulting_mask_packed_sha256) != 64:
            _refuse(
                V1SignedWorkFullUFQActionInvarianceRefusalCode.INVALID_TYPE,
                "packed child digest must be SHA-256",
            )
        try:
            int(self.resulting_mask_packed_sha256, 16)
        except ValueError:
            _refuse(
                V1SignedWorkFullUFQActionInvarianceRefusalCode.INVALID_TYPE,
                "packed child digest must be hexadecimal",
            )
        q_upper = _finite_nonnegative(
            self.origin_branch_q_upper,
            name="origin-branch Q upper",
        )
        budget = _finite_nonnegative(
            self.quadratic_budget,
            name="quadratic budget",
        )
        object.__setattr__(self, "origin_branch_q_upper", q_upper)
        object.__setattr__(self, "quadratic_budget", budget)
        if not q_upper < budget:
            _refuse(
                V1SignedWorkFullUFQActionInvarianceRefusalCode.STRICT_ORIGIN_BRANCH_Q_SAFETY_REQUIRED,
                "registered origin-branch Q upper is not strictly below budget",
            )
        required_true = (
            self.strict_full_population_corrected_F_separation,
            self.strict_origin_branch_registered_Q_safety,
            self.literal_J_same_capture_response_geometry,
            self.complete_two_branch_hard_domain_replayed,
            self.deterministic_FD_residual_decision_stable,
            self.complete_physical_alternate_domain_crosswalk,
            self.physical_action_invariance_authority,
            self.array_free_receipt,
        )
        if any(type(value) is not bool or not value for value in required_true):
            _refuse(
                V1SignedWorkFullUFQActionInvarianceRefusalCode.AUTHORITY_VIOLATION,
                "action-invariance receipt weakened a required conjunction premise",
            )
        _required_false(
            (
                self.source_bound_production_solver_evidence_present,
                self.current_source_h200_completion_present,
                self.production_D_action_invariance_authority,
                self.legacy_rank_only_C_bit_used_as_authority,
                self.C_stream_open_authority,
                self.calibration_K_open_authority,
                self.paired_oracle_O_open_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="full-U F/Q action invariance",
        )
        if (
            self.rule != FULL_U_F_Q_ACTION_INVARIANCE_RULE
            or self.authority_scope != FULL_U_F_Q_ACTION_INVARIANCE_SCOPE
            or self.schema != FULL_U_F_Q_ACTION_INVARIANCE_SCHEMA
        ):
            _refuse(
                V1SignedWorkFullUFQActionInvarianceRefusalCode.AUTHORITY_VIOLATION,
                "action-invariance schema or theorem changed",
            )
        object.__setattr__(self, "receipt_hash", self._compute_hash())

    def _payload(self) -> dict[str, object]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(type(self))
            if item.name != "receipt_hash"
        }

    def _compute_hash(self) -> str:
        return content_hash(self._payload(), prefix=_PREFIX + "receipt-")

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self.receipt_hash:
            _refuse(
                V1SignedWorkFullUFQActionInvarianceRefusalCode.RECEIPT_INTEGRITY_FAILURE,
                "full-U F/Q action-invariance receipt was tampered",
            )
        return observed


@dataclass(frozen=True, slots=True)
class V1SignedWorkFullUFQPreCReadinessReceipt:
    """Closed D-side readiness for a future H200 + physical-C join."""

    action_invariance_receipt_hash: str
    packed_action_hash: str
    resulting_mask_hash: str
    removed_route_index_hash: str
    removed_count: int
    parent_mask_hash: str
    chart_identity_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    corrected_a_stream_identity_hash: str
    literal_j_stream_identity_hash: str
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    complete_alternate_domain_hash: str
    alias_axis_hash: str
    origin_branch: str
    frozen_rank: int
    source_pre_c_branch_family_hash: str
    receipt_hash: str = field(init=False)
    _issuer_token: InitVar[object] = None
    full_u_f_q_physical_action_invariance_verified: bool = True
    source_bound_production_solver_evidence_present: bool = False
    legacy_rank_only_C_bit_used_as_authority: bool = False
    source_bound_h200_completion_present: bool = False
    independent_physical_C_registration_present: bool = False
    production_C_stream_open_authority: bool = False
    independent_preoutcome_C_registration_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    array_free_receipt: bool = True
    authority_scope: str = FULL_U_F_Q_PRE_C_READINESS_SCOPE
    schema: str = FULL_U_F_Q_PRE_C_READINESS_SCHEMA

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _ADMISSION_TOKEN:
            raise TypeError("full-U F/Q pre-C readiness receipts are issuer-only")
        for item in fields(type(self)):
            if item.name.endswith("_hash") and item.name != "receipt_hash":
                object.__setattr__(
                    self,
                    item.name,
                    _identity(getattr(self, item.name), name=item.name),
                )
        object.__setattr__(
            self,
            "removed_count",
            _positive_int(self.removed_count, name="removed count"),
        )
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="route count"),
        )
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        object.__setattr__(
            self,
            "frozen_rank",
            _positive_int(self.frozen_rank, name="frozen rank"),
        )
        if (
            self.full_u_f_q_physical_action_invariance_verified is not True
            or self.array_free_receipt is not True
        ):
            _refuse(
                V1SignedWorkFullUFQActionInvarianceRefusalCode.AUTHORITY_VIOLATION,
                "pre-C readiness weakened its D-side premise",
            )
        _required_false(
            (
                self.source_bound_production_solver_evidence_present,
                self.legacy_rank_only_C_bit_used_as_authority,
                self.source_bound_h200_completion_present,
                self.independent_physical_C_registration_present,
                self.production_C_stream_open_authority,
                self.independent_preoutcome_C_registration_authority,
                self.calibration_K_open_authority,
                self.paired_oracle_O_open_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="full-U F/Q pre-C readiness",
        )
        if (
            self.authority_scope != FULL_U_F_Q_PRE_C_READINESS_SCOPE
            or self.schema != FULL_U_F_Q_PRE_C_READINESS_SCHEMA
        ):
            _refuse(
                V1SignedWorkFullUFQActionInvarianceRefusalCode.AUTHORITY_VIOLATION,
                "pre-C readiness schema or scope changed",
            )
        object.__setattr__(self, "receipt_hash", self._compute_hash())

    def _payload(self) -> dict[str, object]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(type(self))
            if item.name != "receipt_hash"
        }

    def _compute_hash(self) -> str:
        return content_hash(self._payload(), prefix=_PREFIX + "pre-C-readiness-")

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self.receipt_hash:
            _refuse(
                V1SignedWorkFullUFQActionInvarianceRefusalCode.RECEIPT_INTEGRITY_FAILURE,
                "full-U F/Q pre-C readiness was tampered",
            )
        return observed


def issue_v1_signed_work_full_u_f_q_action_invariance(
    *,
    block_support_registration: V1BlockSupportWitnessRegistration,
    block_support_decision: V1BlockSupportWitnessDecision,
    registered_q_bridge: V1SignedWorkVariableShellRegisteredQPreferredHardActionBridgeReceipt,
    literal_j_capture_binding: V1SampledLiteralJOriginalCaptureBinding,
    literal_j_rank_replay_receipt: V1SampledLiteralJFDRankReplayReceipt,
    adaptive_rank_ledger: V1FQRAdaptiveRankReplayLedger,
    adaptive_rank_stability: V1FQRAdaptiveRankStabilityCertificate,
    parent_mask: HardMask,
    packed_action: V1PackedRemovalAction,
) -> V1SignedWorkFullUFQActionInvarianceReceipt:
    """Issue the first canonical complete-domain D physical-action witness."""

    expected_types = (
        (block_support_registration, V1BlockSupportWitnessRegistration),
        (block_support_decision, V1BlockSupportWitnessDecision),
        (
            registered_q_bridge,
            V1SignedWorkVariableShellRegisteredQPreferredHardActionBridgeReceipt,
        ),
        (literal_j_capture_binding, V1SampledLiteralJOriginalCaptureBinding),
        (literal_j_rank_replay_receipt, V1SampledLiteralJFDRankReplayReceipt),
        (adaptive_rank_ledger, V1FQRAdaptiveRankReplayLedger),
        (adaptive_rank_stability, V1FQRAdaptiveRankStabilityCertificate),
        (parent_mask, HardMask),
        (packed_action, V1PackedRemovalAction),
    )
    if any(type(value) is not expected for value, expected in expected_types):
        _refuse(
            V1SignedWorkFullUFQActionInvarianceRefusalCode.INVALID_TYPE,
            "canonical conjunction requires every exact upstream type",
        )

    registration_hash = _assert_integrity(
        block_support_registration,
        owner="block-support registration",
    )
    decision_hash = _assert_integrity(
        block_support_decision,
        owner="block-support decision",
    )
    capture_hash = _assert_integrity(
        literal_j_capture_binding,
        owner="literal-J original capture binding",
    )
    literal_replay_hash = _assert_integrity(
        literal_j_rank_replay_receipt,
        owner="literal-J rank replay receipt",
    )
    ledger_hash = _assert_integrity(
        adaptive_rank_ledger,
        owner="adaptive-rank replay ledger",
    )
    stability_hash = _assert_integrity(
        adaptive_rank_stability,
        owner="adaptive-rank stability certificate",
    )

    if (
        block_support_decision.registration_hash != registration_hash
        or block_support_decision.status
        is not V1BlockSupportWitnessStatus.STABLE_STRICT_SEPARATION
        or block_support_decision.sampled_full_population_F_strict_separation_authority
        is not True
    ):
        _refuse(
            V1SignedWorkFullUFQActionInvarianceRefusalCode.STRICT_FULL_U_F_SEPARATION_REQUIRED,
            "the exact block-support decision is not strict full-U F separation",
        )

    if (
        registered_q_bridge.origin_branch_status
        is not V1SignedWorkHardFQEndpointStatus.Q_SAFE_FEASIBLE
        or not registered_q_bridge.origin_branch_q_upper
        < registered_q_bridge.quadratic_budget
        or registered_q_bridge.origin_branch_specific_q_used is not True
        or registered_q_bridge.canonical_min_q_used_for_admission is not False
        or registered_q_bridge.actual_parent_child_delta_verified is not True
        or registered_q_bridge.source_kkt_branch_id != registered_q_bridge.branch.value
        or registered_q_bridge.source_kkt_branch_receipt_hash
        != registered_q_bridge.hard_fq_branch_identity_receipt_hash
        or registered_q_bridge.endpoint_registered_on_exact_branch_receipt is not True
        or registered_q_bridge.alternate_cell_lower_bound_Q_dropped is not True
        or registered_q_bridge.d_w2_empirical_q_silently_substituted_for_registered_branch
        is not False
        or registered_q_bridge.same_source_signed_f_authority is not False
        or registered_q_bridge.conditional_bridge_only is not True
    ):
        _refuse(
            V1SignedWorkFullUFQActionInvarianceRefusalCode.STRICT_ORIGIN_BRANCH_Q_SAFETY_REQUIRED,
            "the preferred packed action lacks strict exact origin-branch hard-Q safety",
        )

    if (
        literal_j_rank_replay_receipt.original_capture_binding_hash != capture_hash
        or literal_j_rank_replay_receipt.literal_j_stream_identity_hash
        != literal_j_capture_binding.literal_j_stream_identity_hash
        or literal_j_rank_replay_receipt.corrected_a_stream_identity_hash
        != literal_j_capture_binding.corrected_a_stream_identity_hash
        or literal_j_capture_binding.corrected_rows_excluded_from_response_geometry
        is not True
        or literal_j_capture_binding.canonical_q_input_authority is not True
        or literal_j_capture_binding.same_capture_and_vjp is not True
    ):
        _refuse(
            V1SignedWorkFullUFQActionInvarianceRefusalCode.LITERAL_J_REPLAY_REQUIRED,
            "rank replay is not the same-capture literal-J response geometry",
        )

    if (
        adaptive_rank_stability.schema != FQR_ADAPTIVE_RANK_STABILITY_V2_SCHEMA
        or adaptive_rank_stability.rank_local_fd_residual_bound_proof is not True
        or adaptive_rank_stability.complete_two_branch_hard_ladders_replayed is not True
        or adaptive_rank_stability.robust_branch_family_hash is None
        or adaptive_rank_stability.source_pre_c_branch_family_hash is None
        or adaptive_rank_ledger.policy.policy_hash
        != adaptive_rank_stability.policy_hash
        or adaptive_rank_ledger.policy.uncertainty_set_hash
        != adaptive_rank_stability.uncertainty_set_hash
        or ledger_hash != adaptive_rank_stability.replay_ledger_hash
        or tuple(value.record_hash for value in adaptive_rank_ledger.records)
        != adaptive_rank_stability.replay_record_hashes
    ):
        _refuse(
            V1SignedWorkFullUFQActionInvarianceRefusalCode.FD_RANK_STABILITY_REQUIRED,
            "literal-J FD rank stability lacks the complete source-derived v2 family",
        )
    frozen_record = adaptive_rank_ledger.records[-1]
    frozen_decision = frozen_record.decision_certificate
    if (
        frozen_decision.schema != FQR_ADAPTIVE_RANK_DECISION_CERTIFICATE_V2_SCHEMA
        or frozen_decision.certificate_hash
        != adaptive_rank_stability.rank_local_decision_certificate_hash
        or frozen_record.record_hash
        != adaptive_rank_stability.frozen_replay_record_hash
        or frozen_record.rank != adaptive_rank_stability.frozen_rank
        or literal_j_rank_replay_receipt.requested_physical_rank
        != adaptive_rank_stability.frozen_rank
        or literal_j_rank_replay_receipt.fd_design_hash != frozen_record.fd_design_hash
        or literal_j_rank_replay_receipt.fd_snapshot_hash
        != frozen_record.fd_snapshot_hash
        or literal_j_rank_replay_receipt.context_axis_hash
        != frozen_record.D_context_axis_hash
        or literal_j_rank_replay_receipt.draw_axis_hash
        != frozen_record.D_draw_axis_hash
        or literal_j_rank_replay_receipt.source_measurement_axis_hash
        != frozen_record.D_source_measurement_axis_hash
    ):
        _refuse(
            V1SignedWorkFullUFQActionInvarianceRefusalCode.FD_RANK_STABILITY_REQUIRED,
            "frozen adaptive-rank record is not the exact literal-J replay",
        )

    try:
        removed_count, removed_hash = (
            packed_action.resulting_mask.removed_index_count_and_hash_from(parent_mask)
        )
    except ValueError as error:
        _refuse(
            V1SignedWorkFullUFQActionInvarianceRefusalCode.PHYSICAL_ACTION_MISMATCH,
            f"packed action is not an irreversible parent-child delta: {error}",
        )
    packed_digest = hashlib.sha256(packed_action.resulting_mask.packed_bits).hexdigest()
    parent_mask_hash = _fresh_mask_hash(parent_mask, name="parent mask")
    resulting_mask_hash = _fresh_mask_hash(
        packed_action.resulting_mask,
        name="resulting mask",
    )
    if (
        removed_count != packed_action.removed_count
        or removed_hash != packed_action.removed_route_index_hash
        or registered_q_bridge.packed_action_hash != packed_action.action_hash
        or registered_q_bridge.resulting_mask_hash != resulting_mask_hash
        or registered_q_bridge.resulting_mask_packed_sha256 != packed_digest
        or registered_q_bridge.removed_route_index_hash != removed_hash
        or registered_q_bridge.removed_count != removed_count
        or block_support_registration.preferred_action_hash != packed_action.action_hash
    ):
        _refuse(
            V1SignedWorkFullUFQActionInvarianceRefusalCode.PHYSICAL_ACTION_MISMATCH,
            "F, Q, and the actual packed parent-child delta name different actions",
        )

    common_chart = (
        registered_q_bridge.parent_mask_hash,
        registered_q_bridge.chart_identity_hash,
        registered_q_bridge.proposal_generation,
        registered_q_bridge.route_axis_hash,
        registered_q_bridge.route_id_axis_hash,
        registered_q_bridge.route_count,
    )
    if common_chart != (
        parent_mask_hash,
        literal_j_capture_binding.chart_identity_hash,
        literal_j_capture_binding.proposal_generation,
        parent_mask.route_axis_hash,
        literal_j_capture_binding.route_id_axis_hash,
        parent_mask.route_count,
    ) or common_chart != (
        literal_j_rank_replay_receipt.current_mask_hash,
        literal_j_rank_replay_receipt.chart_identity_hash,
        literal_j_rank_replay_receipt.proposal_generation,
        literal_j_rank_replay_receipt.route_axis_hash,
        literal_j_rank_replay_receipt.route_id_axis_hash,
        literal_j_rank_replay_receipt.route_count,
    ):
        _refuse(
            V1SignedWorkFullUFQActionInvarianceRefusalCode.CHART_OR_GENERATION_MISMATCH,
            "Q bridge, packed mask, literal-J capture, and replay are not one chart",
        )
    if (
        block_support_registration.chart_hash != common_chart[1]
        or block_support_registration.current_mask_hash != common_chart[0]
        or block_support_registration.generation_index != common_chart[2]
        or block_support_registration.route_axis_hash != common_chart[3]
        or adaptive_rank_stability.current_mask_hash != common_chart[0]
        or adaptive_rank_stability.proposal_generation != common_chart[2]
        or adaptive_rank_stability.route_axis_hash != common_chart[3]
        or adaptive_rank_stability.route_count != common_chart[5]
    ):
        _refuse(
            V1SignedWorkFullUFQActionInvarianceRefusalCode.CHART_OR_GENERATION_MISMATCH,
            "F witness or adaptive rank names another chart/generation/route axis",
        )
    if (
        block_support_registration.corrected_evidence_source_hash
        != literal_j_capture_binding.corrected_a_stream_identity_hash
    ):
        _refuse(
            V1SignedWorkFullUFQActionInvarianceRefusalCode.LITERAL_J_REPLAY_REQUIRED,
            "full-U F is not the corrected-A stream paired with literal J",
        )
    if (
        block_support_registration.registered_d_prefix_hash
        != registered_q_bridge.registered_d_prefix_hash
        or block_support_registration.registered_d_prefix_hash
        != literal_j_capture_binding.registered_d_prefix_hash
        or block_support_registration.registered_d_w2_source_hash
        != registered_q_bridge.registered_d_w2_prefix_hash
    ):
        _refuse(
            V1SignedWorkFullUFQActionInvarianceRefusalCode.D_PREFIX_MISMATCH,
            "F, Q, and literal-J replay do not descend from one registered D prefix",
        )

    branches = tuple(frozen_decision.branch_decisions)
    if (
        tuple(value.branch for value in branches) != _BRANCH_AXIS
        or any(
            value.rank_local_residual_decision_stable is not True for value in branches
        )
        or registered_q_bridge.branch.value not in _BRANCH_AXIS
        or packed_action.action_hash
        not in next(
            value.ordered_packed_action_hashes
            for value in branches
            if value.branch == registered_q_bridge.branch.value
        )
    ):
        _refuse(
            V1SignedWorkFullUFQActionInvarianceRefusalCode.FD_RANK_STABILITY_REQUIRED,
            "the frozen literal-J family is unstable or omits the origin action",
        )

    branch_axes = tuple(
        (value.branch, value.ordered_packed_action_hashes) for value in branches
    )
    alternate_domain_hash = v1_full_u_f_q_complete_alternate_domain_hash(
        current_mask_hash=common_chart[0],
        route_axis_hash=common_chart[3],
        route_count=common_chart[5],
        proposal_generation=common_chart[2],
        frozen_rank=adaptive_rank_stability.frozen_rank,
        branch_action_axes=branch_axes,
        preferred_packed_action_hash=packed_action.action_hash,
        origin_branch=registered_q_bridge.branch.value,
    )
    alternates = _alternate_axis(
        branches=branches,
        preferred_action_hash=packed_action.action_hash,
    )
    expected_domain_receipt_hash = v1_full_u_f_q_complete_domain_crosswalk_receipt_hash(
        preferred_packed_action_hash=packed_action.action_hash,
        complete_alternate_domain_hash=alternate_domain_hash,
        alternate_action_count=len(alternates),
        alias_binding_hashes=tuple(
            alias.alias_hash for alias in block_support_registration.alias_bindings
        ),
        complete_constraint_union_hash=(
            block_support_registration.complete_constraint_union_hash
        ),
        shared_action_minimum_domain_hash=(
            block_support_registration.shared_action_minimum_domain_hash
        ),
        frozen_rank=adaptive_rank_stability.frozen_rank,
    )
    if (
        block_support_registration.complete_alternate_domain_hash
        != alternate_domain_hash
        or block_support_registration.complete_alternate_domain_receipt_hash
        != expected_domain_receipt_hash
        or block_support_registration.alternate_action_count != len(alternates)
        or block_support_registration.complete_alternate_domain_registered is not True
        or block_support_registration.preferred_excluded_from_alternate_domain
        is not True
        or block_support_registration.complete_constraint_within_shell_outside_pair_union
        is not True
        or block_support_registration.complete_branch_rank_resource_source_cardinality_axis
        is not True
        or block_support_registration.shared_action_minimum_required is not True
    ):
        _refuse(
            V1SignedWorkFullUFQActionInvarianceRefusalCode.INCOMPLETE_ALTERNATE_DOMAIN,
            "full-U F witness is not crosswalked to every frozen branch hard action",
        )

    aliases = tuple(block_support_registration.alias_bindings)
    if {alias.branch_id for alias in aliases} != set(_BRANCH_AXIS) or any(
        alias.fd_rank != adaptive_rank_stability.frozen_rank for alias in aliases
    ):
        _refuse(
            V1SignedWorkFullUFQActionInvarianceRefusalCode.BRANCH_OR_ALIAS_MISMATCH,
            "complete F aliases must cover both branches at exactly the frozen rank",
        )
    matching_origin_aliases = tuple(
        alias
        for alias in aliases
        if alias.branch_id == registered_q_bridge.branch.value
        and alias.fd_rank == adaptive_rank_stability.frozen_rank
        and alias.source_receipt_hash == registered_q_bridge.source_receipt_hash
        and alias.source_replay_hash == registered_q_bridge.source_replay_hash
        and alias.cardinality == removed_count
    )
    if len(matching_origin_aliases) != 1:
        _refuse(
            V1SignedWorkFullUFQActionInvarianceRefusalCode.BRANCH_OR_ALIAS_MISMATCH,
            "origin branch/rank/source/cardinality has no unique complete-domain F alias",
        )

    alias_axis = tuple(
        (
            alias.branch_id,
            alias.fd_rank,
            alias.resource_alias_id,
            alias.source_receipt_hash,
            alias.source_replay_hash,
            alias.cardinality_alias_id,
            alias.cardinality,
        )
        for alias in aliases
    )
    return V1SignedWorkFullUFQActionInvarianceReceipt(
        packed_action_hash=packed_action.action_hash,
        resulting_mask_hash=resulting_mask_hash,
        resulting_mask_packed_sha256=packed_digest,
        removed_route_index_hash=removed_hash,
        removed_count=removed_count,
        physical_action_coordinate_hash=(
            registered_q_bridge.physical_action_coordinate_hash
        ),
        action_coordinate_custody_issuance_hash=(
            registered_q_bridge.action_coordinate_custody_issuance_hash
        ),
        action_coordinate_family_receipt_hash=(
            registered_q_bridge.action_coordinate_family_receipt_hash
        ),
        action_subset_identity_hash=registered_q_bridge.action_subset_identity_hash,
        parent_mask_hash=parent_mask_hash,
        model_identity_hash=registered_q_bridge.model_identity_hash,
        chart_identity_hash=common_chart[1],
        proposal_generation=common_chart[2],
        route_axis_hash=common_chart[3],
        route_id_axis_hash=common_chart[4],
        route_count=common_chart[5],
        corrected_a_stream_identity_hash=(
            literal_j_capture_binding.corrected_a_stream_identity_hash
        ),
        literal_j_stream_identity_hash=(
            literal_j_capture_binding.literal_j_stream_identity_hash
        ),
        original_capture_binding_hash=capture_hash,
        literal_j_rank_replay_receipt_hash=literal_replay_hash,
        literal_j_fd_design_hash=literal_j_rank_replay_receipt.fd_design_hash,
        literal_j_fd_snapshot_hash=literal_j_rank_replay_receipt.fd_snapshot_hash,
        registered_d_prefix_hash=block_support_registration.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=(
            block_support_registration.registered_d_w2_source_hash
        ),
        block_support_registration_hash=registration_hash,
        block_support_decision_hash=decision_hash,
        block_support_state_hash=block_support_decision.state_hash,
        block_support_receipt_chain_hash=block_support_decision.receipt_chain_hash,
        full_axis_support_plane_source_hash=(
            block_support_registration.full_axis_support_plane_source_hash
        ),
        native_fiber_axis_hash=block_support_registration.native_fiber_axis_hash,
        full_population_axis_hash=block_support_registration.full_population_axis_hash,
        full_population_count=block_support_registration.full_population_count,
        complete_alternate_domain_hash=alternate_domain_hash,
        complete_alternate_domain_receipt_hash=(expected_domain_receipt_hash),
        deduplicated_alternate_action_hashes=alternates,
        alternate_action_count=len(alternates),
        complete_constraint_union_hash=(
            block_support_registration.complete_constraint_union_hash
        ),
        shared_action_minimum_domain_hash=(
            block_support_registration.shared_action_minimum_domain_hash
        ),
        alias_axis_hash=block_support_registration.alias_axis_hash,
        alias_binding_hashes=tuple(alias.alias_hash for alias in aliases),
        alias_branch_rank_resource_source_cardinality_axis=alias_axis,
        origin_branch=registered_q_bridge.branch.value,
        origin_branch_source_receipt_hash=registered_q_bridge.source_receipt_hash,
        origin_branch_source_replay_hash=registered_q_bridge.source_replay_hash,
        origin_branch_source_identity_receipt_hash=(
            registered_q_bridge.source_kkt_branch_receipt_hash
        ),
        shell_freeze_receipt_hash=registered_q_bridge.shell_freeze_receipt_hash,
        registered_q_bridge_receipt_hash=registered_q_bridge.receipt_hash,
        origin_branch_hard_fq_problem_hash=registered_q_bridge.hard_fq_problem_hash,
        origin_branch_hard_fq_operator_receipt_hash=(
            registered_q_bridge.hard_fq_operator_receipt_hash
        ),
        origin_branch_hard_fq_receipt_hash=(
            registered_q_bridge.hard_fq_branch_receipt_hash
        ),
        origin_branch_hard_fq_endpoint_receipt_hash=(
            registered_q_bridge.hard_fq_endpoint_receipt_hash
        ),
        origin_branch_q_upper=registered_q_bridge.origin_branch_q_upper,
        quadratic_budget=registered_q_bridge.quadratic_budget,
        adaptive_rank_policy_hash=adaptive_rank_stability.policy_hash,
        adaptive_rank_uncertainty_set_hash=(
            adaptive_rank_stability.uncertainty_set_hash
        ),
        adaptive_rank_ledger_hash=ledger_hash,
        adaptive_rank_stability_certificate_hash=stability_hash,
        adaptive_rank_decision_certificate_hash=frozen_decision.certificate_hash,
        frozen_replay_record_hash=frozen_record.record_hash,
        sampled_geometry_source_hash=frozen_record.sampled_geometry_source_hash,
        kkt_problem_hash=frozen_record.kkt_problem_hash,
        kkt_result_hash=frozen_record.kkt_result_hash,
        frozen_rank=adaptive_rank_stability.frozen_rank,
        executed_rank_prefix=adaptive_rank_stability.executed_rank_prefix,
        branch_action_axes=branch_axes,
        branch_selected_action_hashes=tuple(
            (value.branch, value.selected_packed_action_hash) for value in branches
        ),
        source_pre_c_branch_family_hash=(
            adaptive_rank_stability.source_pre_c_branch_family_hash
        ),
        robust_branch_family_hash=(adaptive_rank_stability.robust_branch_family_hash),
        _issuer_token=_INVARIANCE_TOKEN,
    )


def issue_v1_signed_work_full_u_f_q_pre_c_readiness(
    *,
    action_invariance: V1SignedWorkFullUFQActionInvarianceReceipt,
) -> V1SignedWorkFullUFQPreCReadinessReceipt:
    """Record D readiness while leaving C closed pending physical evidence."""

    if type(action_invariance) is not V1SignedWorkFullUFQActionInvarianceReceipt:
        _refuse(
            V1SignedWorkFullUFQActionInvarianceRefusalCode.INVALID_TYPE,
            "pre-C readiness requires the exact full-U F/Q witness",
        )
    action_invariance.assert_integrity()
    if (
        action_invariance.physical_action_invariance_authority is not True
        or action_invariance.C_stream_open_authority is not False
        or action_invariance.legacy_rank_only_C_bit_used_as_authority is not False
    ):
        _refuse(
            V1SignedWorkFullUFQActionInvarianceRefusalCode.AUTHORITY_VIOLATION,
            "pre-C readiness did not originate at the closed full-U conjunction seam",
        )
    return V1SignedWorkFullUFQPreCReadinessReceipt(
        action_invariance_receipt_hash=action_invariance.receipt_hash,
        packed_action_hash=action_invariance.packed_action_hash,
        resulting_mask_hash=action_invariance.resulting_mask_hash,
        removed_route_index_hash=action_invariance.removed_route_index_hash,
        removed_count=action_invariance.removed_count,
        parent_mask_hash=action_invariance.parent_mask_hash,
        chart_identity_hash=action_invariance.chart_identity_hash,
        proposal_generation=action_invariance.proposal_generation,
        route_axis_hash=action_invariance.route_axis_hash,
        route_id_axis_hash=action_invariance.route_id_axis_hash,
        route_count=action_invariance.route_count,
        corrected_a_stream_identity_hash=(
            action_invariance.corrected_a_stream_identity_hash
        ),
        literal_j_stream_identity_hash=action_invariance.literal_j_stream_identity_hash,
        registered_d_prefix_hash=action_invariance.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=action_invariance.registered_d_w2_prefix_hash,
        complete_alternate_domain_hash=(
            action_invariance.complete_alternate_domain_hash
        ),
        alias_axis_hash=action_invariance.alias_axis_hash,
        origin_branch=action_invariance.origin_branch,
        frozen_rank=action_invariance.frozen_rank,
        source_pre_c_branch_family_hash=(
            action_invariance.source_pre_c_branch_family_hash
        ),
        _issuer_token=_ADMISSION_TOKEN,
    )


__all__ = (
    "FULL_U_F_Q_ACTION_DOMAIN_SCHEMA",
    "FULL_U_F_Q_ACTION_INVARIANCE_RULE",
    "FULL_U_F_Q_ACTION_INVARIANCE_SCOPE",
    "FULL_U_F_Q_ACTION_INVARIANCE_SCHEMA",
    "FULL_U_F_Q_PRE_C_READINESS_SCOPE",
    "FULL_U_F_Q_PRE_C_READINESS_SCHEMA",
    "V1SignedWorkFullUFQActionInvarianceReceipt",
    "V1SignedWorkFullUFQActionInvarianceRefusal",
    "V1SignedWorkFullUFQActionInvarianceRefusalCode",
    "V1SignedWorkFullUFQPreCReadinessReceipt",
    "issue_v1_signed_work_full_u_f_q_action_invariance",
    "issue_v1_signed_work_full_u_f_q_pre_c_readiness",
    "v1_full_u_f_q_complete_alternate_domain_hash",
    "v1_full_u_f_q_complete_domain_crosswalk_receipt_hash",
)
