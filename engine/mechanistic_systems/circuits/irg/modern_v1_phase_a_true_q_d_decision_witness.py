"""Source-bound Phase-A true-Q component of the D decision witness.

This is the narrow join between two independently scoped facts:

* the robust local shell/action-image premise selected one physical action;
* the compact global true-Q hard-face theorem certified that every
  resource-competitive hard action shares that action's outside membership;
* the selected action is the best certified hard vertex on that frozen face.

This is deliberately only the Q-side component.  Signed-F separation and the
registered A/F/J alpha-family closure are separate premises of the final D
decision-stability issuer.  This receipt never opens C/K/O, commits a mask, or
claims terminal/runtime/Pareto authority.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
import math
from types import MappingProxyType
from typing import Final, Mapping, NoReturn

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask

from .modern_v1_phase_a_global_q_hard_face_margin_certificate import (
    V1PhaseAGlobalQHardFaceMarginCertificateResult,
    V1PhaseAGlobalQHardFaceMarginCertificateStatus,
)
from .modern_v1_phase_a_global_q_outside_face_cuda import (
    V1PhaseAGlobalQOutsideFaceResult,
)
from .modern_v1_sampled_v7_action_pair_alias_source_join import (
    V1SampledV7ActionPairAliasSourceJoinReceipt,
)
from .modern_v1_sampled_v7_phase_a_robust_shell_decision import (
    V1SampledV7PhaseARobustDecisionStatus,
    V1SampledV7PhaseARobustShellDecisionReceipt,
)


PHASE_A_TRUE_Q_D_DECISION_WITNESS_SCHEMA: Final = (
    "partition_free_v1_phase_a_true_q_d_decision_witness_v2"
)
PHASE_A_TRUE_Q_D_DECISION_WITNESS_ALGORITHM: Final = (
    "robust_local_physical_action_premise_joined_to_compact_global_true_q_"
    "hard_outside_face_margin_and_exact_best_shell_vertex_v2"
)
PHASE_A_TRUE_Q_D_DECISION_WITNESS_AUTHORITY: Final = (
    "Q_side_D_decision_component_only_signed_F_and_A_F_J_closure_still_"
    "required_independent_C_still_required_no_C_stream_open_K_O_commit_"
    "runtime_terminal_or_pareto_authority_v2"
)

_PREFIX: Final = "partition-free-v1-phase-a-true-q-D-decision-witness-"
_RECEIPT_TOKEN: Final = object()


class V1PhaseATrueQDDecisionWitnessRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    SOURCE_MISMATCH = "source_mismatch"
    LOCAL_ACTION_UNSTABLE = "local_action_unstable"
    OUTSIDE_FACE_UNCERTIFIED = "outside_face_uncertified"
    SELECTED_SHELL_VERTEX_MISMATCH = "selected_shell_vertex_mismatch"
    PRODUCTION_MODE_MISMATCH = "production_mode_mismatch"
    AUTHORITY_VIOLATION = "authority_violation"


class V1PhaseATrueQDDecisionWitnessRefusal(RuntimeError):
    def __init__(
        self,
        code: V1PhaseATrueQDDecisionWitnessRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1PhaseATrueQDDecisionWitnessRefusalCode:
            raise TypeError("true-Q D decision witness refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1PhaseATrueQDDecisionWitnessRefusalCode,
    message: str,
) -> NoReturn:
    raise V1PhaseATrueQDDecisionWitnessRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1PhaseATrueQDDecisionWitnessRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty exact string",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value <= 0:
        _refuse(
            V1PhaseATrueQDDecisionWitnessRefusalCode.INVALID_TYPE,
            f"{name} must be one positive exact integer",
        )
    return value


def _payload_hash(payload: Mapping[str, object], *, role: str) -> str:
    return content_hash(dict(payload), prefix=_PREFIX + role + "-")


def _hard_mask_bit(mask: HardMask, index: int) -> bool:
    byte = mask.packed_bits[index // 8]
    return bool(byte & (1 << (7 - (index % 8))))


@dataclass(frozen=True, slots=True)
class V1PhaseATrueQDDecisionWitnessReceipt:
    robust_shell_decision_receipt_hash: str
    outside_face_scan_receipt_hash: str
    hard_face_margin_receipt_hash: str
    dual_proposal_receipt_hash: str
    relaxed_kkt_problem_hash: str
    action_pair_alias_source_join_receipt_hash: str
    action_pair_hash: str
    parent_mask_hash: str
    selected_resulting_mask_hash: str
    selected_physical_action_identity_hash: str
    selected_packed_action_hash: str
    route_axis_hash: str
    route_id_axis_hash: str
    shell_axis_hash: str
    block_axis_hash: str
    route_count: int
    direction_count: int
    selected_total_count: int
    selected_outside_count: int
    selected_shell_count: int
    best_hard_face_resource_lower: float
    structural_dual_resource_upper: float
    dual_gap_upper: float
    strict_outside_bit_margin_lower: float
    local_robust_action_premise: bool = True
    compact_global_true_q_hard_face_certified: bool = True
    selected_action_matches_best_certified_hard_face_vertex: bool = True
    q_side_d_decision_component: bool = True
    signed_F_decision_component_still_required: bool = True
    A_F_J_alpha_family_closure_still_required: bool = True
    independent_C_still_required: bool = True
    production_cuda_execution: bool = False
    private_cpu_reference_execution: bool = True
    d_stability_authority: bool = False
    C_stream_open_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    runtime_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    algorithm: str = PHASE_A_TRUE_Q_D_DECISION_WITNESS_ALGORITHM
    authority_scope: str = PHASE_A_TRUE_Q_D_DECISION_WITNESS_AUTHORITY
    schema: str = PHASE_A_TRUE_Q_D_DECISION_WITNESS_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _RECEIPT_TOKEN:
            raise PermissionError("true-Q D decision witness receipts require their issuer")
        self._validate()
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _validate(self) -> None:
        for name in (
            "robust_shell_decision_receipt_hash",
            "outside_face_scan_receipt_hash",
            "hard_face_margin_receipt_hash",
            "dual_proposal_receipt_hash",
            "relaxed_kkt_problem_hash",
            "action_pair_alias_source_join_receipt_hash",
            "action_pair_hash",
            "parent_mask_hash",
            "selected_resulting_mask_hash",
            "selected_physical_action_identity_hash",
            "selected_packed_action_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "shell_axis_hash",
            "block_axis_hash",
        ):
            _identity(getattr(self, name), name=name)
        for name in (
            "route_count",
            "direction_count",
            "selected_total_count",
        ):
            _positive_int(getattr(self, name), name=name)
        for name in ("selected_outside_count", "selected_shell_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                _refuse(
                    V1PhaseATrueQDDecisionWitnessRefusalCode.INVALID_TYPE,
                    f"{name} must be one nonnegative exact integer",
                )
        if self.selected_total_count != self.selected_outside_count + self.selected_shell_count:
            _refuse(
                V1PhaseATrueQDDecisionWitnessRefusalCode.SOURCE_MISMATCH,
                "selected D action counts do not close",
            )
        if self.selected_total_count > self.route_count:
            _refuse(
                V1PhaseATrueQDDecisionWitnessRefusalCode.SOURCE_MISMATCH,
                "selected D action exceeds the registered route axis",
            )
        for name in (
            "best_hard_face_resource_lower",
            "structural_dual_resource_upper",
            "dual_gap_upper",
            "strict_outside_bit_margin_lower",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                _refuse(
                    V1PhaseATrueQDDecisionWitnessRefusalCode.INVALID_TYPE,
                    f"{name} must be one finite exact binary64 scalar",
                )
        if (
            self.best_hard_face_resource_lower < 0.0
            or self.structural_dual_resource_upper
            < self.best_hard_face_resource_lower
            or self.dual_gap_upper < 0.0
            or self.strict_outside_bit_margin_lower <= 0.0
        ):
            _refuse(
                V1PhaseATrueQDDecisionWitnessRefusalCode.OUTSIDE_FACE_UNCERTIFIED,
                "hard-face resource, gap, or strict-margin bounds are invalid",
            )
        if not self.dual_gap_upper < self.strict_outside_bit_margin_lower:
            _refuse(
                V1PhaseATrueQDDecisionWitnessRefusalCode.OUTSIDE_FACE_UNCERTIFIED,
                "hard-face dual gap is not below the strict outside margin",
            )
        positive = (
            self.local_robust_action_premise,
            self.compact_global_true_q_hard_face_certified,
            self.selected_action_matches_best_certified_hard_face_vertex,
            self.q_side_d_decision_component,
            self.signed_F_decision_component_still_required,
            self.A_F_J_alpha_family_closure_still_required,
            self.independent_C_still_required,
        )
        closed = (
            self.d_stability_authority,
            self.C_stream_open_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
            self.runtime_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if (
            any(type(value) is not bool or value is not True for value in positive)
            or any(type(value) is not bool or value is not False for value in closed)
            or type(self.production_cuda_execution) is not bool
            or type(self.private_cpu_reference_execution) is not bool
            or self.production_cuda_execution == self.private_cpu_reference_execution
            or self.algorithm != PHASE_A_TRUE_Q_D_DECISION_WITNESS_ALGORITHM
            or self.authority_scope != PHASE_A_TRUE_Q_D_DECISION_WITNESS_AUTHORITY
            or self.schema != PHASE_A_TRUE_Q_D_DECISION_WITNESS_SCHEMA
        ):
            _refuse(
                V1PhaseATrueQDDecisionWitnessRefusalCode.AUTHORITY_VIOLATION,
                "true-Q D decision witness crossed its authority boundary",
            )

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                item.name: getattr(self, item.name)
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
                V1PhaseATrueQDDecisionWitnessRefusalCode.SOURCE_MISMATCH,
                "true-Q D decision witness receipt no longer replays",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


def issue_v1_phase_a_true_q_d_decision_witness(
    *,
    robust_shell_decision: V1SampledV7PhaseARobustShellDecisionReceipt,
    outside_face_scan: V1PhaseAGlobalQOutsideFaceResult,
    hard_face_margin: V1PhaseAGlobalQHardFaceMarginCertificateResult,
    action_pair_alias_source_join: V1SampledV7ActionPairAliasSourceJoinReceipt,
    parent_mask: HardMask,
) -> V1PhaseATrueQDDecisionWitnessReceipt:
    if (
        type(robust_shell_decision) is not V1SampledV7PhaseARobustShellDecisionReceipt
        or type(outside_face_scan) is not V1PhaseAGlobalQOutsideFaceResult
        or type(hard_face_margin)
        is not V1PhaseAGlobalQHardFaceMarginCertificateResult
        or type(action_pair_alias_source_join)
        is not V1SampledV7ActionPairAliasSourceJoinReceipt
        or type(parent_mask) is not HardMask
    ):
        _refuse(
            V1PhaseATrueQDDecisionWitnessRefusalCode.INVALID_TYPE,
            "true-Q D decision witness requires exact source receipts",
        )
    robust_hash = robust_shell_decision.assert_integrity()
    outside = outside_face_scan.receipt
    outside_hash = outside.assert_integrity()
    outside_face_scan.compact_handle.assert_live()
    hard = hard_face_margin.receipt
    hard_hash = hard.assert_integrity()
    hard_face_margin.compact_handle.assert_live()
    source_join_hash = action_pair_alias_source_join.assert_integrity()
    pair = action_pair_alias_source_join.action_pair
    pair_hash = pair.assert_integrity()
    if (
        robust_shell_decision.status
        is not V1SampledV7PhaseARobustDecisionStatus.STABLE_PHYSICAL_ACTION_PREMISE
    ):
        _refuse(
            V1PhaseATrueQDDecisionWitnessRefusalCode.LOCAL_ACTION_UNSTABLE,
            "robust local shell premise did not select one stable physical action",
        )
    if (
        hard.status
        is not V1PhaseAGlobalQHardFaceMarginCertificateStatus.CERTIFIED_RESOURCE_COMPETITIVE_HARD_OUTSIDE_FACE
        or not hard.all_resource_competitive_hard_actions_share_outside_membership
        or hard.best_hard_face_resource_lower is None
        or hard.dual_gap_upper is None
        or hard.strict_outside_bit_margin_lower is None
        or hard.best_hard_shell_gate_bits is None
    ):
        _refuse(
            V1PhaseATrueQDDecisionWitnessRefusalCode.OUTSIDE_FACE_UNCERTIFIED,
            "compact global true-Q hard outside-face certificate is not closed",
        )
    matches = tuple(
        action
        for action in pair.physical_actions
        if action.physical_action_identity_hash
        == robust_shell_decision.selected_physical_action_identity_hash
    )
    if len(matches) != 1:
        _refuse(
            V1PhaseATrueQDDecisionWitnessRefusalCode.SOURCE_MISMATCH,
            "source join does not contain exactly one robustly selected physical action",
        )
    selected = matches[0]
    selected.assert_integrity()
    child = selected.packed_action.resulting_mask
    if (
        outside.robust_shell_decision_receipt_hash != robust_hash
        or hard.outside_face_receipt_hash != outside_hash
        or hard.robust_shell_decision_receipt_hash != robust_hash
        or hard.action_pair_alias_source_join_receipt_hash != source_join_hash
        or outside.action_pair_alias_source_join_receipt_hash != source_join_hash
        or hard.action_pair_hash != pair_hash
        or outside.action_pair_hash != pair_hash
        or hard.dual_proposal_receipt_hash != outside.dual_proposal_receipt_hash
        or hard.relaxed_kkt_problem_hash != outside.relaxed_kkt_problem_hash
        or robust_shell_decision.selected_physical_action_identity_hash
        != outside.selected_physical_action_identity_hash
        or hard.selected_physical_action_identity_hash
        != outside.selected_physical_action_identity_hash
        or selected.physical_action_identity_hash
        != outside.selected_physical_action_identity_hash
        or selected.packed_action_hash != outside.selected_packed_action_hash
        or child.mask_hash != outside.selected_resulting_mask_hash
        or hard.selected_packed_action_hash != selected.packed_action_hash
        or hard.selected_resulting_mask_hash != child.mask_hash
        or hard.parent_mask_hash != pair.parent_mask_hash
        or outside.parent_mask_hash != pair.parent_mask_hash
        or parent_mask.mask_hash != pair.parent_mask_hash
        or hard.route_axis_hash != pair.route_axis_hash
        or outside.route_axis_hash != pair.route_axis_hash
        or parent_mask.route_axis_hash != pair.route_axis_hash
        or hard.route_count != pair.route_count
        or outside.route_count != pair.route_count
        or parent_mask.route_count != pair.route_count
    ):
        _refuse(
            V1PhaseATrueQDDecisionWitnessRefusalCode.SOURCE_MISMATCH,
            "robust local premise, outside scan, and hard-face certificate do not name one D action",
        )
    # A packed action removes a route iff the parent bit is live and its child
    # bit is zero.  Read only B <= 8 bits from the packed masks; never unpack
    # the route-scale axis to establish this compact identity.
    selected_shell_bits = tuple(
        _hard_mask_bit(parent_mask, route_index)
        and not _hard_mask_bit(child, route_index)
        for route_index in hard.shell_indices
    )
    if selected_shell_bits != hard.best_hard_shell_gate_bits:
        _refuse(
            V1PhaseATrueQDDecisionWitnessRefusalCode.SELECTED_SHELL_VERTEX_MISMATCH,
            "the robustly selected action is not the best certified hard "
            "vertex on the frozen outside face",
        )
    if (
        hard.production_cuda_execution != outside.production_cuda_execution
        or hard.private_cpu_reference_execution
        != outside.private_cpu_reference_execution
    ):
        _refuse(
            V1PhaseATrueQDDecisionWitnessRefusalCode.PRODUCTION_MODE_MISMATCH,
            "outside scan and hard-face theorem used different execution modes",
        )
    return V1PhaseATrueQDDecisionWitnessReceipt(
        robust_shell_decision_receipt_hash=robust_hash,
        outside_face_scan_receipt_hash=outside_hash,
        hard_face_margin_receipt_hash=hard_hash,
        dual_proposal_receipt_hash=outside.dual_proposal_receipt_hash,
        relaxed_kkt_problem_hash=outside.relaxed_kkt_problem_hash,
        action_pair_alias_source_join_receipt_hash=source_join_hash,
        action_pair_hash=pair_hash,
        parent_mask_hash=outside.parent_mask_hash,
        selected_resulting_mask_hash=outside.selected_resulting_mask_hash,
        selected_physical_action_identity_hash=outside.selected_physical_action_identity_hash,
        selected_packed_action_hash=outside.selected_packed_action_hash,
        route_axis_hash=outside.route_axis_hash,
        route_id_axis_hash=outside.route_id_axis_hash,
        shell_axis_hash=outside.shell_axis_hash,
        block_axis_hash=outside.block_axis_hash,
        route_count=outside.route_count,
        direction_count=outside.direction_count,
        selected_total_count=outside.selected_total_count,
        selected_outside_count=outside.selected_outside_count,
        selected_shell_count=outside.selected_shell_count,
        best_hard_face_resource_lower=hard.best_hard_face_resource_lower,
        structural_dual_resource_upper=hard.structural_dual_resource_upper,
        dual_gap_upper=hard.dual_gap_upper,
        strict_outside_bit_margin_lower=hard.strict_outside_bit_margin_lower,
        production_cuda_execution=hard.production_cuda_execution,
        private_cpu_reference_execution=hard.private_cpu_reference_execution,
        _issuance_token=_RECEIPT_TOKEN,
    )


__all__ = (
    "PHASE_A_TRUE_Q_D_DECISION_WITNESS_SCHEMA",
    "PHASE_A_TRUE_Q_D_DECISION_WITNESS_ALGORITHM",
    "PHASE_A_TRUE_Q_D_DECISION_WITNESS_AUTHORITY",
    "V1PhaseATrueQDDecisionWitnessRefusalCode",
    "V1PhaseATrueQDDecisionWitnessRefusal",
    "V1PhaseATrueQDDecisionWitnessReceipt",
    "issue_v1_phase_a_true_q_d_decision_witness",
)
