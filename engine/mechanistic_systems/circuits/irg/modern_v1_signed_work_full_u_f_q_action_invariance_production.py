"""Production promotion of the complete-domain v1 D action witness.

The existing full-U F/Q receipt proves the physical-action, registered-Q, and
literal-J/FD conjunction, but its issuance also required the authority-inert
legacy block-support decision.  This module therefore consumes only an
explicit Q/J/action-domain projection of that receipt: its full hash, legacy F
fields, and legacy registration are excluded from the authority payload.  The
source-bound block decision V2 supplies corrected F plus the source-native
action/domain/chart identity.  Only those independent coordinates are joined
before production D action invariance is promoted.

The promotion remains pre-C.  A current-source H200 execution completion and
an independently frozen physical-C registration are still required before C
may open; K/O, commit, terminal, performance, and Pareto authority remain
closed.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
import hashlib
from numbers import Integral, Real
import math
from typing import Final, Mapping, NoReturn, Sequence

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
from .modern_v1_signed_work_block_support_domain_binding import (
    V1BlockSupportPreOutcomeDomainBindingReceipt,
)
from .modern_v1_signed_work_block_support_production_witness import (
    V1BlockSupportProductionWitnessDecisionV2,
    require_v1_block_support_production_witness_decision_v2,
)
from .modern_v1_signed_work_full_u_f_q_action_invariance import (
    V1SignedWorkFullUFQActionInvarianceReceipt,
    v1_full_u_f_q_complete_alternate_domain_hash,
    v1_full_u_f_q_complete_domain_crosswalk_receipt_hash,
)
from .modern_v1_signed_work_hard_fq_replay_cuda import (
    V1SignedWorkHardFQEndpointStatus,
)
from .modern_v1_signed_work_variable_shell_registered_q_bridge import (
    V1SignedWorkVariableShellRegisteredQPreferredHardActionBridgeReceipt,
)


FULL_U_F_Q_PRODUCTION_ACTION_INVARIANCE_V3_SCHEMA: Final = (
    "partition_free_v1_full_U_F_Q_production_action_invariance_v3"
)
FULL_U_F_Q_PRODUCTION_ACTION_INVARIANCE_V3_RULE: Final = (
    "exact_source_native_action_domain_chart_join_of_full_U_corrected_F_"
    "separation_with_legacy_F_excluded_Q_J_physical_action_projection_v3"
)
FULL_U_F_Q_PRODUCTION_ACTION_INVARIANCE_V3_SCOPE: Final = (
    "production_D_action_invariance_only_pending_current_source_H200_and_"
    "independent_physical_C_registration_no_C_K_O_commit_terminal_"
    "performance_or_pareto_authority_v3"
)
FULL_U_F_Q_PRODUCTION_Q_J_DOMAIN_PROJECTION_SCHEMA: Final = (
    "partition_free_v1_full_U_Q_J_domain_projection_for_production_v1"
)
FULL_U_F_Q_PRODUCTION_Q_J_DOMAIN_PROJECTION_RULE: Final = (
    "F_free_literal_J_registered_Q_complete_physical_domain_projection_v1"
)

_PREFIX: Final = "partition-free-v1-full-U-F-Q-production-invariance-"
_ISSUER_TOKEN: Final = object()
_PROJECTION_TOKEN: Final = object()
_BRANCH_AXIS: Final = ("spectral", "diagonal_l1")
_Q_J_DOMAIN_PROJECTION_FIELDS: Final = (
    "packed_action_hash",
    "resulting_mask_hash",
    "resulting_mask_packed_sha256",
    "removed_route_index_hash",
    "removed_count",
    "physical_action_coordinate_hash",
    "action_coordinate_custody_issuance_hash",
    "action_coordinate_family_receipt_hash",
    "action_subset_identity_hash",
    "parent_mask_hash",
    "model_identity_hash",
    "chart_identity_hash",
    "proposal_generation",
    "route_axis_hash",
    "route_id_axis_hash",
    "route_count",
    "corrected_a_stream_identity_hash",
    "literal_j_stream_identity_hash",
    "original_capture_binding_hash",
    "literal_j_rank_replay_receipt_hash",
    "literal_j_fd_design_hash",
    "literal_j_fd_snapshot_hash",
    "registered_d_prefix_hash",
    "registered_d_w2_prefix_hash",
    "complete_alternate_domain_hash",
    "complete_alternate_domain_receipt_hash",
    "deduplicated_alternate_action_hashes",
    "alternate_action_count",
    "alias_axis_hash",
    "origin_branch",
    "origin_branch_source_receipt_hash",
    "origin_branch_source_replay_hash",
    "origin_branch_source_identity_receipt_hash",
    "shell_freeze_receipt_hash",
    "registered_q_bridge_receipt_hash",
    "origin_branch_hard_fq_problem_hash",
    "origin_branch_hard_fq_operator_receipt_hash",
    "origin_branch_hard_fq_receipt_hash",
    "origin_branch_hard_fq_endpoint_receipt_hash",
    "origin_branch_q_upper",
    "quadratic_budget",
    "adaptive_rank_policy_hash",
    "adaptive_rank_uncertainty_set_hash",
    "adaptive_rank_ledger_hash",
    "adaptive_rank_stability_certificate_hash",
    "adaptive_rank_decision_certificate_hash",
    "frozen_replay_record_hash",
    "sampled_geometry_source_hash",
    "kkt_problem_hash",
    "kkt_result_hash",
    "frozen_rank",
    "executed_rank_prefix",
    "branch_action_axes",
    "branch_selected_action_hashes",
    "source_pre_c_branch_family_hash",
    "robust_branch_family_hash",
    "strict_origin_branch_registered_Q_safety",
    "literal_J_same_capture_response_geometry",
    "complete_two_branch_hard_domain_replayed",
    "deterministic_FD_residual_decision_stable",
    "complete_physical_alternate_domain_crosswalk",
    "physical_action_invariance_authority",
    "array_free_receipt",
)


class V1SignedWorkFullUFQProductionRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    RECEIPT_TAMPERING = "receipt_tampering"
    F_WITNESS_MISMATCH = "F_witness_mismatch"
    SOURCE_NATIVE_PARENT_MISMATCH = "source_native_parent_mismatch"
    INCOMPLETE_BASE_INVARIANCE = "incomplete_base_invariance"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SignedWorkFullUFQProductionRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SignedWorkFullUFQProductionRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SignedWorkFullUFQProductionRefusalCode:
            raise TypeError("production full-U refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SignedWorkFullUFQProductionRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SignedWorkFullUFQProductionRefusal(code, message)


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SignedWorkFullUFQProductionRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SignedWorkFullUFQProductionRefusalCode.INVALID_TYPE,
            f"{name} must be one exact integer",
        )
    result = int(value)
    if result < minimum:
        _refuse(
            V1SignedWorkFullUFQProductionRefusalCode.INVALID_TYPE,
            f"{name} must be at least {minimum}",
        )
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1SignedWorkFullUFQProductionRefusalCode.INVALID_TYPE,
            f"{name} must be one real scalar",
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        _refuse(
            V1SignedWorkFullUFQProductionRefusalCode.INVALID_TYPE,
            f"{name} must be finite and nonnegative",
        )
    return result


def _closed(values: Sequence[object], *, owner: str) -> None:
    if any(type(value) is not bool or value for value in values):
        _refuse(
            V1SignedWorkFullUFQProductionRefusalCode.AUTHORITY_VIOLATION,
            f"{owner} attempted to open downstream authority",
        )


def _payload(value: object) -> dict[str, object]:
    return {
        item.name: (
            getattr(value, item.name).value
            if isinstance(getattr(value, item.name), Enum)
            else getattr(value, item.name)
        )
        for item in fields(value)
        if not item.name.startswith("_") and item.name != "receipt_hash"
    }


def _base_q_j_domain_projection_payload(
    base: V1SignedWorkFullUFQActionInvarianceReceipt,
) -> dict[str, object]:
    """Project only independently proven Q/J/physical-domain coordinates."""

    if type(base) is not V1SignedWorkFullUFQActionInvarianceReceipt:
        raise TypeError("Q/J/domain projection requires the exact base receipt")
    if any(
        "block_support" in name
        or name == "strict_full_population_corrected_F_separation"
        for name in _Q_J_DOMAIN_PROJECTION_FIELDS
    ):
        raise AssertionError("legacy F coordinates entered Q/J/domain projection")
    return {name: getattr(base, name) for name in _Q_J_DOMAIN_PROJECTION_FIELDS}


def _base_q_j_domain_projection_hash(
    base: V1SignedWorkFullUFQActionInvarianceReceipt,
) -> str:
    return content_hash(
        _base_q_j_domain_projection_payload(base),
        prefix=_PREFIX + "Q-J-domain-projection-",
    )


def _projection_hash(payload: Mapping[str, object]) -> str:
    return content_hash(
        {name: payload[name] for name in _Q_J_DOMAIN_PROJECTION_FIELDS},
        prefix=_PREFIX + "Q-J-domain-projection-",
    )


def _assert_parent_integrity(value: object, *, owner: str) -> str:
    method = getattr(value, "assert_integrity", None)
    if not callable(method):
        _refuse(
            V1SignedWorkFullUFQProductionRefusalCode.INVALID_TYPE,
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
            V1SignedWorkFullUFQProductionRefusalCode.RECEIPT_TAMPERING,
            f"{owner} failed replay: {error}",
        )
    return _text(observed, name=f"{owner} integrity hash")


def _fresh_mask_hash(mask: HardMask, *, name: str) -> str:
    if type(mask) is not HardMask:
        raise TypeError(f"{name} must use the exact HardMask type")
    try:
        snapshot = HardMask(
            route_axis_hash=mask.route_axis_hash,
            route_count=mask.route_count,
            packed_bits=bytes(mask.packed_bits),
        )
    except (TypeError, ValueError) as error:
        _refuse(
            V1SignedWorkFullUFQProductionRefusalCode.SOURCE_NATIVE_PARENT_MISMATCH,
            f"{name} packed content is malformed: {error}",
        )
    return snapshot.mask_hash


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
class V1SignedWorkFullUFQProductionQJDomainProjection:
    """Issuer-only F-free carrier for full-U literal-J/Q/domain facts."""

    values: tuple[tuple[str, object], ...]
    receipt_hash: str = field(init=False)
    _issuer_token: InitVar[object] = None
    source_role: str = "F_free_literal_J_registered_Q_physical_domain_projection"
    legacy_F_parent_fields_excluded: bool = True
    schema: str = FULL_U_F_Q_PRODUCTION_Q_J_DOMAIN_PROJECTION_SCHEMA
    rule: str = FULL_U_F_Q_PRODUCTION_Q_J_DOMAIN_PROJECTION_RULE
    _integrity_seal: str = field(default="", repr=False, compare=False)

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _PROJECTION_TOKEN:
            raise TypeError("production Q/J/domain projections are issuer-only")
        for name in ("source_role", "schema", "rule"):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        if (
            self.schema != FULL_U_F_Q_PRODUCTION_Q_J_DOMAIN_PROJECTION_SCHEMA
            or self.rule != FULL_U_F_Q_PRODUCTION_Q_J_DOMAIN_PROJECTION_RULE
        ):
            _refuse(
                V1SignedWorkFullUFQProductionRefusalCode.AUTHORITY_VIOLATION,
                "unsupported production Q/J/domain projection semantics",
            )
        if self.legacy_F_parent_fields_excluded is not True:
            _refuse(
                V1SignedWorkFullUFQProductionRefusalCode.AUTHORITY_VIOLATION,
                "projection did not exclude legacy F parent fields",
            )
        values = tuple(self.values)
        names = tuple(name for name, _value in values)
        if names != _Q_J_DOMAIN_PROJECTION_FIELDS:
            _refuse(
                V1SignedWorkFullUFQProductionRefusalCode.INVALID_TYPE,
                "projection fields are not the exact registered Q/J/domain axis",
            )
        if any(
            "block_support" in name
            or name == "strict_full_population_corrected_F_separation"
            for name in names
        ):
            _refuse(
                V1SignedWorkFullUFQProductionRefusalCode.AUTHORITY_VIOLATION,
                "legacy F coordinate leaked into production Q/J/domain projection",
            )
        object.__setattr__(self, "values", values)
        observed = content_hash(
            self._payload(include_seal=False),
            prefix=_PREFIX + "Q-J-domain-projection-integrity-",
        )
        if self._integrity_seal and self._integrity_seal != observed:
            _refuse(
                V1SignedWorkFullUFQProductionRefusalCode.RECEIPT_TAMPERING,
                "production Q/J/domain projection was mutated",
            )
        object.__setattr__(self, "_integrity_seal", observed)
        object.__setattr__(
            self,
            "receipt_hash",
            content_hash(
                self._payload(include_seal=False),
                prefix=_PREFIX + "Q-J-domain-projection-receipt-",
            ),
        )

    def _payload(self, *, include_seal: bool) -> dict[str, object]:
        payload = {
            "values": self.values,
            "source_role": self.source_role,
            "legacy_F_parent_fields_excluded": self.legacy_F_parent_fields_excluded,
            "schema": self.schema,
            "rule": self.rule,
        }
        if include_seal:
            payload["_integrity_seal"] = self._integrity_seal
        return payload

    def as_dict(self) -> dict[str, object]:
        self.assert_integrity()
        return dict(self.values)

    def projection_hash(self) -> str:
        return _projection_hash(self.as_dict())

    def assert_integrity(self) -> str:
        observed = content_hash(
            self._payload(include_seal=False),
            prefix=_PREFIX + "Q-J-domain-projection-integrity-",
        )
        expected = content_hash(
            self._payload(include_seal=False),
            prefix=_PREFIX + "Q-J-domain-projection-receipt-",
        )
        if observed != self._integrity_seal or expected != self.receipt_hash:
            _refuse(
                V1SignedWorkFullUFQProductionRefusalCode.RECEIPT_TAMPERING,
                "production Q/J/domain projection no longer replays",
            )
        return expected


def issue_v1_signed_work_full_u_f_q_production_q_j_domain_projection_from_base_v1(
    base_action_invariance: V1SignedWorkFullUFQActionInvarianceReceipt,
) -> V1SignedWorkFullUFQProductionQJDomainProjection:
    """Mint the transitional F-free Q/J/domain projection from a base receipt."""

    if type(base_action_invariance) is not V1SignedWorkFullUFQActionInvarianceReceipt:
        raise TypeError("base projection requires the exact full-U base receipt")
    try:
        base_action_invariance.assert_integrity()
    except (RuntimeError, PermissionError, TypeError, ValueError) as error:
        _refuse(
            V1SignedWorkFullUFQProductionRefusalCode.RECEIPT_TAMPERING,
            f"base Q/J/domain projection parent refused: {error}",
        )
    required_base = (
        base_action_invariance.strict_origin_branch_registered_Q_safety,
        base_action_invariance.literal_J_same_capture_response_geometry,
        base_action_invariance.complete_two_branch_hard_domain_replayed,
        base_action_invariance.deterministic_FD_residual_decision_stable,
        base_action_invariance.complete_physical_alternate_domain_crosswalk,
        base_action_invariance.physical_action_invariance_authority,
    )
    if any(type(value) is not bool or value is not True for value in required_base):
        _refuse(
            V1SignedWorkFullUFQProductionRefusalCode.INCOMPLETE_BASE_INVARIANCE,
            "base receipt lacks its Q/J/domain physical-invariance premises",
        )
    _closed(
        (
            base_action_invariance.source_bound_production_solver_evidence_present,
            base_action_invariance.current_source_h200_completion_present,
            base_action_invariance.production_D_action_invariance_authority,
            base_action_invariance.C_stream_open_authority,
            base_action_invariance.calibration_K_open_authority,
            base_action_invariance.paired_oracle_O_open_authority,
            base_action_invariance.mask_commit_authority,
            base_action_invariance.terminal_authority,
            base_action_invariance.pareto_authority,
        ),
        owner="base Q/J/domain projection",
    )
    payload = _base_q_j_domain_projection_payload(base_action_invariance)
    return V1SignedWorkFullUFQProductionQJDomainProjection(
        values=tuple((name, payload[name]) for name in _Q_J_DOMAIN_PROJECTION_FIELDS),
        _issuer_token=_PROJECTION_TOKEN,
    )


def issue_v1_signed_work_full_u_f_q_production_q_j_domain_projection_from_source_native_parents_v2(
    *,
    registered_q_bridge: V1SignedWorkVariableShellRegisteredQPreferredHardActionBridgeReceipt,
    literal_j_capture_binding: V1SampledLiteralJOriginalCaptureBinding,
    literal_j_rank_replay_receipt: V1SampledLiteralJFDRankReplayReceipt,
    adaptive_rank_ledger: V1FQRAdaptiveRankReplayLedger,
    adaptive_rank_stability: V1FQRAdaptiveRankStabilityCertificate,
    preoutcome_domain: V1BlockSupportPreOutcomeDomainBindingReceipt,
    parent_mask: HardMask,
    packed_action: V1PackedRemovalAction,
) -> V1SignedWorkFullUFQProductionQJDomainProjection:
    """Issue the canonical projection directly from Q/J/domain parents.

    This path deliberately does not construct, inspect, or hash the legacy
    full-U F/Q action-invariance receipt.  The pre-outcome parent contributes
    only the already-frozen physical action-domain crosswalk; no F outcome,
    block-support decision, confidence interval, or corrected-work value can
    enter the projection.
    """

    expected_types = (
        (
            registered_q_bridge,
            V1SignedWorkVariableShellRegisteredQPreferredHardActionBridgeReceipt,
        ),
        (literal_j_capture_binding, V1SampledLiteralJOriginalCaptureBinding),
        (literal_j_rank_replay_receipt, V1SampledLiteralJFDRankReplayReceipt),
        (adaptive_rank_ledger, V1FQRAdaptiveRankReplayLedger),
        (adaptive_rank_stability, V1FQRAdaptiveRankStabilityCertificate),
        (preoutcome_domain, V1BlockSupportPreOutcomeDomainBindingReceipt),
        (parent_mask, HardMask),
        (packed_action, V1PackedRemovalAction),
    )
    if any(type(value) is not expected for value, expected in expected_types):
        raise TypeError("direct Q/J/domain projection requires exact parent types")

    capture_hash = _assert_parent_integrity(
        literal_j_capture_binding,
        owner="literal-J original capture binding",
    )
    literal_replay_hash = _assert_parent_integrity(
        literal_j_rank_replay_receipt,
        owner="literal-J rank replay receipt",
    )
    ledger_hash = _assert_parent_integrity(
        adaptive_rank_ledger,
        owner="adaptive-rank replay ledger",
    )
    stability_hash = _assert_parent_integrity(
        adaptive_rank_stability,
        owner="adaptive-rank stability certificate",
    )
    _assert_parent_integrity(
        preoutcome_domain,
        owner="pre-outcome physical domain binding",
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
            V1SignedWorkFullUFQProductionRefusalCode.INCOMPLETE_BASE_INVARIANCE,
            "registered-Q parent lacks strict exact origin-branch Q safety",
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
            V1SignedWorkFullUFQProductionRefusalCode.SOURCE_NATIVE_PARENT_MISMATCH,
            "literal-J replay is not the registered same-capture response geometry",
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
            V1SignedWorkFullUFQProductionRefusalCode.INCOMPLETE_BASE_INVARIANCE,
            "adaptive literal-J rank replay lacks its complete source-derived family",
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
            V1SignedWorkFullUFQProductionRefusalCode.SOURCE_NATIVE_PARENT_MISMATCH,
            "frozen adaptive-rank state is not the exact literal-J replay",
        )

    try:
        removed_count, removed_hash = (
            packed_action.resulting_mask.removed_index_count_and_hash_from(parent_mask)
        )
    except ValueError as error:
        _refuse(
            V1SignedWorkFullUFQProductionRefusalCode.SOURCE_NATIVE_PARENT_MISMATCH,
            f"packed action is not an irreversible parent-child delta: {error}",
        )
    parent_mask_hash = _fresh_mask_hash(parent_mask, name="parent mask")
    resulting_mask_hash = _fresh_mask_hash(
        packed_action.resulting_mask,
        name="resulting mask",
    )
    packed_digest = hashlib.sha256(packed_action.resulting_mask.packed_bits).hexdigest()
    if (
        removed_count != packed_action.removed_count
        or removed_hash != packed_action.removed_route_index_hash
        or registered_q_bridge.packed_action_hash != packed_action.action_hash
        or registered_q_bridge.resulting_mask_hash != resulting_mask_hash
        or registered_q_bridge.resulting_mask_packed_sha256 != packed_digest
        or registered_q_bridge.removed_route_index_hash != removed_hash
        or registered_q_bridge.removed_count != removed_count
    ):
        _refuse(
            V1SignedWorkFullUFQProductionRefusalCode.SOURCE_NATIVE_PARENT_MISMATCH,
            "registered Q and the actual packed parent-child delta name different actions",
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
            V1SignedWorkFullUFQProductionRefusalCode.SOURCE_NATIVE_PARENT_MISMATCH,
            "Q bridge, masks, literal-J capture, and replay are not one chart",
        )
    if common_chart != (
        preoutcome_domain.current_mask_hash,
        preoutcome_domain.chart_identity_hash,
        preoutcome_domain.proposal_generation,
        preoutcome_domain.route_axis_hash,
        preoutcome_domain.route_id_axis_hash,
        preoutcome_domain.route_count,
    ) or (
        preoutcome_domain.model_identity_hash != registered_q_bridge.model_identity_hash
        or adaptive_rank_stability.current_mask_hash != common_chart[0]
        or adaptive_rank_stability.proposal_generation != common_chart[2]
        or adaptive_rank_stability.route_axis_hash != common_chart[3]
        or adaptive_rank_stability.route_count != common_chart[5]
    ):
        _refuse(
            V1SignedWorkFullUFQProductionRefusalCode.SOURCE_NATIVE_PARENT_MISMATCH,
            "pre-outcome domain or adaptive rank names another chart/generation",
        )
    if (
        registered_q_bridge.registered_d_prefix_hash
        != literal_j_capture_binding.registered_d_prefix_hash
        or preoutcome_domain.registered_d_prefix_hash
        != registered_q_bridge.registered_d_prefix_hash
        or preoutcome_domain.registered_d_w2_prefix_hash
        != registered_q_bridge.registered_d_w2_prefix_hash
    ):
        _refuse(
            V1SignedWorkFullUFQProductionRefusalCode.SOURCE_NATIVE_PARENT_MISMATCH,
            "Q, literal J, and the physical domain do not share one D prefix",
        )
    source_pairs = tuple(
        zip(
            preoutcome_domain.source_receipt_hashes,
            preoutcome_domain.source_replay_hashes,
            strict=True,
        )
    )
    if (
        registered_q_bridge.source_receipt_hash,
        registered_q_bridge.source_replay_hash,
    ) not in source_pairs:
        _refuse(
            V1SignedWorkFullUFQProductionRefusalCode.SOURCE_NATIVE_PARENT_MISMATCH,
            "origin Q source is absent from the registered physical domain",
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
            V1SignedWorkFullUFQProductionRefusalCode.INCOMPLETE_BASE_INVARIANCE,
            "frozen literal-J family is unstable or omits the origin action",
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
        alias_binding_hashes=preoutcome_domain.alias_binding_hashes,
        complete_constraint_union_hash=(
            preoutcome_domain.constraint_union_registration_hash
        ),
        shared_action_minimum_domain_hash=(
            preoutcome_domain.shared_action_minimum_domain_hash
        ),
        frozen_rank=adaptive_rank_stability.frozen_rank,
    )
    if (
        preoutcome_domain.preferred_action_hash != packed_action.action_hash
        or preoutcome_domain.preferred_resulting_mask_hash != resulting_mask_hash
        or preoutcome_domain.preferred_removed_route_index_hash != removed_hash
        or preoutcome_domain.preferred_selected_count != removed_count
        or preoutcome_domain.complete_alternate_domain_hash != alternate_domain_hash
        or preoutcome_domain.complete_alternate_domain_receipt_hash
        != expected_domain_receipt_hash
        or preoutcome_domain.hard_alternate_action_count != len(alternates)
        or preoutcome_domain.complete_alias_axis_bound is not True
        or preoutcome_domain.complete_structural_cell_partition_bound is not True
        or preoutcome_domain.structural_within_shell_domain_only is not True
        or preoutcome_domain.complete_constraint_cover_source_bound is not False
    ):
        _refuse(
            V1SignedWorkFullUFQProductionRefusalCode.SOURCE_NATIVE_PARENT_MISMATCH,
            "pre-outcome parent is not the exact complete frozen physical domain",
        )

    payload: dict[str, object] = {
        "packed_action_hash": packed_action.action_hash,
        "resulting_mask_hash": resulting_mask_hash,
        "resulting_mask_packed_sha256": packed_digest,
        "removed_route_index_hash": removed_hash,
        "removed_count": removed_count,
        "physical_action_coordinate_hash": (
            registered_q_bridge.physical_action_coordinate_hash
        ),
        "action_coordinate_custody_issuance_hash": (
            registered_q_bridge.action_coordinate_custody_issuance_hash
        ),
        "action_coordinate_family_receipt_hash": (
            registered_q_bridge.action_coordinate_family_receipt_hash
        ),
        "action_subset_identity_hash": (
            registered_q_bridge.action_subset_identity_hash
        ),
        "parent_mask_hash": parent_mask_hash,
        "model_identity_hash": registered_q_bridge.model_identity_hash,
        "chart_identity_hash": common_chart[1],
        "proposal_generation": common_chart[2],
        "route_axis_hash": common_chart[3],
        "route_id_axis_hash": common_chart[4],
        "route_count": common_chart[5],
        "corrected_a_stream_identity_hash": (
            literal_j_capture_binding.corrected_a_stream_identity_hash
        ),
        "literal_j_stream_identity_hash": (
            literal_j_capture_binding.literal_j_stream_identity_hash
        ),
        "original_capture_binding_hash": capture_hash,
        "literal_j_rank_replay_receipt_hash": literal_replay_hash,
        "literal_j_fd_design_hash": literal_j_rank_replay_receipt.fd_design_hash,
        "literal_j_fd_snapshot_hash": literal_j_rank_replay_receipt.fd_snapshot_hash,
        "registered_d_prefix_hash": registered_q_bridge.registered_d_prefix_hash,
        "registered_d_w2_prefix_hash": (
            registered_q_bridge.registered_d_w2_prefix_hash
        ),
        "complete_alternate_domain_hash": alternate_domain_hash,
        "complete_alternate_domain_receipt_hash": expected_domain_receipt_hash,
        "deduplicated_alternate_action_hashes": alternates,
        "alternate_action_count": len(alternates),
        "alias_axis_hash": preoutcome_domain.alias_axis_hash,
        "origin_branch": registered_q_bridge.branch.value,
        "origin_branch_source_receipt_hash": registered_q_bridge.source_receipt_hash,
        "origin_branch_source_replay_hash": registered_q_bridge.source_replay_hash,
        "origin_branch_source_identity_receipt_hash": (
            registered_q_bridge.source_kkt_branch_receipt_hash
        ),
        "shell_freeze_receipt_hash": registered_q_bridge.shell_freeze_receipt_hash,
        "registered_q_bridge_receipt_hash": registered_q_bridge.receipt_hash,
        "origin_branch_hard_fq_problem_hash": (
            registered_q_bridge.hard_fq_problem_hash
        ),
        "origin_branch_hard_fq_operator_receipt_hash": (
            registered_q_bridge.hard_fq_operator_receipt_hash
        ),
        "origin_branch_hard_fq_receipt_hash": (
            registered_q_bridge.hard_fq_branch_receipt_hash
        ),
        "origin_branch_hard_fq_endpoint_receipt_hash": (
            registered_q_bridge.hard_fq_endpoint_receipt_hash
        ),
        "origin_branch_q_upper": registered_q_bridge.origin_branch_q_upper,
        "quadratic_budget": registered_q_bridge.quadratic_budget,
        "adaptive_rank_policy_hash": adaptive_rank_stability.policy_hash,
        "adaptive_rank_uncertainty_set_hash": (
            adaptive_rank_stability.uncertainty_set_hash
        ),
        "adaptive_rank_ledger_hash": ledger_hash,
        "adaptive_rank_stability_certificate_hash": stability_hash,
        "adaptive_rank_decision_certificate_hash": frozen_decision.certificate_hash,
        "frozen_replay_record_hash": frozen_record.record_hash,
        "sampled_geometry_source_hash": frozen_record.sampled_geometry_source_hash,
        "kkt_problem_hash": frozen_record.kkt_problem_hash,
        "kkt_result_hash": frozen_record.kkt_result_hash,
        "frozen_rank": adaptive_rank_stability.frozen_rank,
        "executed_rank_prefix": adaptive_rank_stability.executed_rank_prefix,
        "branch_action_axes": branch_axes,
        "branch_selected_action_hashes": tuple(
            (value.branch, value.selected_packed_action_hash) for value in branches
        ),
        "source_pre_c_branch_family_hash": (
            adaptive_rank_stability.source_pre_c_branch_family_hash
        ),
        "robust_branch_family_hash": (
            adaptive_rank_stability.robust_branch_family_hash
        ),
        "strict_origin_branch_registered_Q_safety": True,
        "literal_J_same_capture_response_geometry": True,
        "complete_two_branch_hard_domain_replayed": True,
        "deterministic_FD_residual_decision_stable": True,
        "complete_physical_alternate_domain_crosswalk": True,
        "physical_action_invariance_authority": True,
        "array_free_receipt": True,
    }
    if tuple(payload) != _Q_J_DOMAIN_PROJECTION_FIELDS:
        raise AssertionError("direct Q/J/domain projection field axis drifted")
    return V1SignedWorkFullUFQProductionQJDomainProjection(
        values=tuple((name, payload[name]) for name in _Q_J_DOMAIN_PROJECTION_FIELDS),
        _issuer_token=_PROJECTION_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkFullUFQProductionActionInvarianceReceiptV3:
    base_q_j_domain_projection_hash: str
    q_j_domain_projection_receipt_hash: str
    production_block_support_decision_hash: str
    production_block_support_state_hash: str
    production_block_support_receipt_chain_hash: str
    bound_block_support_registration_receipt_hash: str
    preoutcome_domain_receipt_hash: str
    packed_action_hash: str
    resulting_mask_hash: str
    resulting_mask_packed_sha256: str
    removed_route_index_hash: str
    removed_count: int
    alternate_action_count: int
    physical_action_coordinate_hash: str
    parent_mask_hash: str
    model_identity_hash: str
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
    complete_alternate_domain_receipt_hash: str
    alias_axis_hash: str
    origin_branch: str
    origin_branch_q_upper: float
    quadratic_budget: float
    frozen_rank: int
    source_pre_c_branch_family_hash: str
    robust_branch_family_hash: str
    receipt_hash: str = field(init=False)
    _issuer_token: InitVar[object] = None
    source_bound_full_population_corrected_F_separation: bool = True
    strict_origin_branch_registered_Q_safety: bool = True
    literal_J_same_capture_response_geometry: bool = True
    complete_two_branch_hard_domain_replayed: bool = True
    deterministic_FD_residual_decision_stable: bool = True
    production_D_action_invariance_authority: bool = True
    current_source_h200_completion_present: bool = False
    independent_physical_C_registration_present: bool = False
    production_C_stream_open_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    array_free_receipt: bool = True
    legacy_F_parent_fields_excluded: bool = True
    rule: str = FULL_U_F_Q_PRODUCTION_ACTION_INVARIANCE_V3_RULE
    authority_scope: str = FULL_U_F_Q_PRODUCTION_ACTION_INVARIANCE_V3_SCOPE
    schema: str = FULL_U_F_Q_PRODUCTION_ACTION_INVARIANCE_V3_SCHEMA
    _integrity_seal: str = field(default="", repr=False, compare=False)

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _ISSUER_TOKEN:
            raise TypeError("production full-U invariance receipts are issuer-only")
        for item in fields(self):
            if (
                not item.name.startswith("_")
                and item.name.endswith("_hash")
                and item.name != "receipt_hash"
            ):
                object.__setattr__(
                    self,
                    item.name,
                    _text(getattr(self, item.name), name=item.name),
                )
        for name in ("origin_branch", "rule", "authority_scope", "schema"):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), name=name),
            )
        for name in ("proposal_generation",):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name=name),
            )
        for name in (
            "removed_count",
            "alternate_action_count",
            "route_count",
            "frozen_rank",
        ):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name=name, minimum=1),
            )
        if len(self.resulting_mask_packed_sha256) != 64:
            _refuse(
                V1SignedWorkFullUFQProductionRefusalCode.INVALID_TYPE,
                "resulting mask digest must be SHA-256",
            )
        try:
            int(self.resulting_mask_packed_sha256, 16)
        except ValueError:
            _refuse(
                V1SignedWorkFullUFQProductionRefusalCode.INVALID_TYPE,
                "resulting mask digest must be hexadecimal",
            )
        q_upper = _finite_nonnegative(
            self.origin_branch_q_upper,
            name="origin branch Q upper",
        )
        budget = _finite_nonnegative(
            self.quadratic_budget,
            name="quadratic budget",
        )
        object.__setattr__(self, "origin_branch_q_upper", q_upper)
        object.__setattr__(self, "quadratic_budget", budget)
        if not q_upper < budget:
            _refuse(
                V1SignedWorkFullUFQProductionRefusalCode.INCOMPLETE_BASE_INVARIANCE,
                "origin-branch Q is not strictly safe",
            )
        required = (
            self.source_bound_full_population_corrected_F_separation,
            self.strict_origin_branch_registered_Q_safety,
            self.literal_J_same_capture_response_geometry,
            self.complete_two_branch_hard_domain_replayed,
            self.deterministic_FD_residual_decision_stable,
            self.production_D_action_invariance_authority,
            self.array_free_receipt,
            self.legacy_F_parent_fields_excluded,
        )
        if any(type(value) is not bool or value is not True for value in required):
            _refuse(
                V1SignedWorkFullUFQProductionRefusalCode.AUTHORITY_VIOLATION,
                "production D invariance lost a mandatory premise",
            )
        _closed(
            (
                self.current_source_h200_completion_present,
                self.independent_physical_C_registration_present,
                self.production_C_stream_open_authority,
                self.calibration_K_open_authority,
                self.paired_oracle_O_open_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.performance_authority,
                self.pareto_authority,
            ),
            owner="production full-U action invariance",
        )
        if (
            self.rule != FULL_U_F_Q_PRODUCTION_ACTION_INVARIANCE_V3_RULE
            or self.authority_scope != FULL_U_F_Q_PRODUCTION_ACTION_INVARIANCE_V3_SCOPE
            or self.schema != FULL_U_F_Q_PRODUCTION_ACTION_INVARIANCE_V3_SCHEMA
        ):
            _refuse(
                V1SignedWorkFullUFQProductionRefusalCode.AUTHORITY_VIOLATION,
                "unsupported production full-U semantics",
            )
        observed = content_hash(
            _payload(self),
            prefix=_PREFIX + "integrity-",
        )
        if self._integrity_seal and self._integrity_seal != observed:
            _refuse(
                V1SignedWorkFullUFQProductionRefusalCode.RECEIPT_TAMPERING,
                "production full-U receipt was mutated",
            )
        object.__setattr__(self, "_integrity_seal", observed)
        object.__setattr__(
            self,
            "receipt_hash",
            content_hash(_payload(self), prefix=_PREFIX + "receipt-"),
        )

    def assert_integrity(self) -> str:
        payload = _payload(self)
        observed = content_hash(payload, prefix=_PREFIX + "integrity-")
        expected = content_hash(payload, prefix=_PREFIX + "receipt-")
        if observed != self._integrity_seal or expected != self.receipt_hash:
            _refuse(
                V1SignedWorkFullUFQProductionRefusalCode.RECEIPT_TAMPERING,
                "production full-U receipt no longer replays",
            )
        return expected


def issue_v1_signed_work_full_u_f_q_production_action_invariance_v3(
    *,
    base_action_invariance: V1SignedWorkFullUFQActionInvarianceReceipt,
    production_block_support_decision: V1BlockSupportProductionWitnessDecisionV2,
) -> V1SignedWorkFullUFQProductionActionInvarianceReceiptV3:
    """Transitionally project the base receipt, then use the canonical join."""

    if (
        type(base_action_invariance) is not V1SignedWorkFullUFQActionInvarianceReceipt
        or type(production_block_support_decision)
        is not V1BlockSupportProductionWitnessDecisionV2
    ):
        raise TypeError("production full-U join requires exact parent types")
    projection = (
        issue_v1_signed_work_full_u_f_q_production_q_j_domain_projection_from_base_v1(
            base_action_invariance
        )
    )
    return (
        issue_v1_signed_work_full_u_f_q_production_action_invariance_from_projection_v3(
            q_j_domain_projection=projection,
            production_block_support_decision=production_block_support_decision,
        )
    )


def issue_v1_signed_work_full_u_f_q_production_action_invariance_from_projection_v3(
    *,
    q_j_domain_projection: V1SignedWorkFullUFQProductionQJDomainProjection,
    production_block_support_decision: V1BlockSupportProductionWitnessDecisionV2,
) -> V1SignedWorkFullUFQProductionActionInvarianceReceiptV3:
    """Join source-bound corrected F to an explicitly F-free Q/J projection."""

    if (
        type(q_j_domain_projection)
        is not V1SignedWorkFullUFQProductionQJDomainProjection
        or type(production_block_support_decision)
        is not V1BlockSupportProductionWitnessDecisionV2
    ):
        raise TypeError("production full-U join requires exact projection parents")
    try:
        q_j_domain_projection.assert_integrity()
        projected = q_j_domain_projection.as_dict()
        decision = require_v1_block_support_production_witness_decision_v2(
            production_block_support_decision
        )
    except (RuntimeError, PermissionError, TypeError, ValueError) as error:
        _refuse(
            V1SignedWorkFullUFQProductionRefusalCode.RECEIPT_TAMPERING,
            f"one production full-U parent refused: {error}",
        )
    if (
        projected["packed_action_hash"] != decision.preferred_action_hash
        or projected["complete_alternate_domain_hash"]
        != decision.complete_alternate_domain_hash
        or projected["complete_alternate_domain_receipt_hash"]
        != decision.complete_alternate_domain_receipt_hash
        or projected["alternate_action_count"] != decision.alternate_action_count
        or projected["alias_axis_hash"] != decision.alias_axis_hash
        or projected["model_identity_hash"] != decision.model_identity_hash
        or projected["chart_identity_hash"] != decision.chart_identity_hash
        or projected["parent_mask_hash"] != decision.current_mask_hash
        or projected["proposal_generation"] != decision.proposal_generation
        or projected["route_axis_hash"] != decision.route_axis_hash
        or projected["route_id_axis_hash"] != decision.route_id_axis_hash
        or projected["route_count"] != decision.route_count
        or projected["corrected_a_stream_identity_hash"]
        != decision.corrected_evidence_source_hash
        or projected["registered_d_prefix_hash"] != decision.registered_d_prefix_hash
        or projected["registered_d_w2_prefix_hash"]
        != decision.registered_d_w2_prefix_hash
    ):
        _refuse(
            V1SignedWorkFullUFQProductionRefusalCode.F_WITNESS_MISMATCH,
            "source-bound F and projected Q/J name another action/domain/chart",
        )
    required_projection = (
        projected["strict_origin_branch_registered_Q_safety"],
        projected["literal_J_same_capture_response_geometry"],
        projected["complete_two_branch_hard_domain_replayed"],
        projected["deterministic_FD_residual_decision_stable"],
        projected["complete_physical_alternate_domain_crosswalk"],
        projected["physical_action_invariance_authority"],
    )
    if any(
        type(value) is not bool or value is not True for value in required_projection
    ):
        _refuse(
            V1SignedWorkFullUFQProductionRefusalCode.INCOMPLETE_BASE_INVARIANCE,
            "Q/J/domain projection lacks its physical-invariance premises",
        )
    _closed(
        (
            decision.q_authority,
            decision.action_invariance_authority,
            decision.c_authority,
            decision.k_authority,
            decision.o_authority,
            decision.commit_authority,
            decision.terminal_authority,
            decision.performance_authority,
            decision.pareto_authority,
        ),
        owner="production full-U parent join",
    )
    return V1SignedWorkFullUFQProductionActionInvarianceReceiptV3(
        base_q_j_domain_projection_hash=q_j_domain_projection.projection_hash(),
        q_j_domain_projection_receipt_hash=q_j_domain_projection.receipt_hash,
        production_block_support_decision_hash=decision.decision_hash,
        production_block_support_state_hash=decision.state_hash,
        production_block_support_receipt_chain_hash=decision.receipt_chain_hash,
        bound_block_support_registration_receipt_hash=(
            decision.bound_registration_receipt_hash
        ),
        preoutcome_domain_receipt_hash=decision.preoutcome_domain_receipt_hash,
        packed_action_hash=projected["packed_action_hash"],
        resulting_mask_hash=projected["resulting_mask_hash"],
        resulting_mask_packed_sha256=projected["resulting_mask_packed_sha256"],
        removed_route_index_hash=projected["removed_route_index_hash"],
        removed_count=projected["removed_count"],
        alternate_action_count=projected["alternate_action_count"],
        physical_action_coordinate_hash=projected["physical_action_coordinate_hash"],
        parent_mask_hash=projected["parent_mask_hash"],
        model_identity_hash=projected["model_identity_hash"],
        chart_identity_hash=projected["chart_identity_hash"],
        proposal_generation=projected["proposal_generation"],
        route_axis_hash=projected["route_axis_hash"],
        route_id_axis_hash=projected["route_id_axis_hash"],
        route_count=projected["route_count"],
        corrected_a_stream_identity_hash=projected["corrected_a_stream_identity_hash"],
        literal_j_stream_identity_hash=projected["literal_j_stream_identity_hash"],
        registered_d_prefix_hash=projected["registered_d_prefix_hash"],
        registered_d_w2_prefix_hash=projected["registered_d_w2_prefix_hash"],
        complete_alternate_domain_hash=projected["complete_alternate_domain_hash"],
        complete_alternate_domain_receipt_hash=projected[
            "complete_alternate_domain_receipt_hash"
        ],
        alias_axis_hash=projected["alias_axis_hash"],
        origin_branch=projected["origin_branch"],
        origin_branch_q_upper=projected["origin_branch_q_upper"],
        quadratic_budget=projected["quadratic_budget"],
        frozen_rank=projected["frozen_rank"],
        source_pre_c_branch_family_hash=projected["source_pre_c_branch_family_hash"],
        robust_branch_family_hash=projected["robust_branch_family_hash"],
        _issuer_token=_ISSUER_TOKEN,
    )


__all__ = (
    "FULL_U_F_Q_PRODUCTION_ACTION_INVARIANCE_V3_RULE",
    "FULL_U_F_Q_PRODUCTION_ACTION_INVARIANCE_V3_SCHEMA",
    "FULL_U_F_Q_PRODUCTION_ACTION_INVARIANCE_V3_SCOPE",
    "FULL_U_F_Q_PRODUCTION_Q_J_DOMAIN_PROJECTION_RULE",
    "FULL_U_F_Q_PRODUCTION_Q_J_DOMAIN_PROJECTION_SCHEMA",
    "V1SignedWorkFullUFQProductionActionInvarianceReceiptV3",
    "V1SignedWorkFullUFQProductionQJDomainProjection",
    "V1SignedWorkFullUFQProductionRefusal",
    "V1SignedWorkFullUFQProductionRefusalCode",
    "issue_v1_signed_work_full_u_f_q_production_action_invariance_from_projection_v3",
    "issue_v1_signed_work_full_u_f_q_production_action_invariance_v3",
    "issue_v1_signed_work_full_u_f_q_production_q_j_domain_projection_from_base_v1",
    "issue_v1_signed_work_full_u_f_q_production_q_j_domain_projection_from_source_native_parents_v2",
)
