"""Fail-closed bridge from joint F/Q/R evidence to a proposal queue.

The bridge is intentionally narrower than calibration or oracle execution.  It
binds one packed, pre-outcome frozen ladder to the current chart, mask, and
proposal generation; verifies the complete typed F/Q/R evidence graph; and
materializes a descending view of *independently* certified endpoints.  It
never fills gaps in that set and therefore makes no prefix-monotonicity claim.

The returned authorization permits only proposal-queue materialization.  It
does not authorize endpoint freeze, calibration, a BCE-oracle look, a mask
commit, terminal validation access, or an irreversible decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral
from typing import Final, NoReturn

from mechanistic_systems.contracts.hard_mask_actions import HardMask

from .contracts import content_hash
from .modern_v1_candidate_composition import V1PackedRemovalAction
from .modern_v1_fqr_certification import (
    FQR_PROPOSAL_AUTHORITY,
    FQR_STABILITY_SOURCES,
    V1DecisionStabilityCertificate,
    V1FQRCertificationDecision,
    V1FQRCertificationDesign,
    V1FQRCertificationReceipt,
    V1FQRCertificationRefusal,
    V1FQRCertificationRefusalCode,
    decide_v1_fqr,
    verify_v1_fqr_certification_receipt,
)
from .modern_v1_ladder_statistics import (
    V1FrozenCandidateLadder,
    V1FrozenLadderCertificationReceipt,
    V1FrozenLadderDecision,
)


FQR_CANDIDATE_QUEUE_ENTRY_SCHEMA: Final = (
    "partition_free_v1_fqr_candidate_queue_entry_v1"
)
FQR_CANDIDATE_QUEUE_AUTHORIZATION_SCHEMA: Final = (
    "partition_free_v1_fqr_candidate_queue_authorization_v1"
)
FQR_CANDIDATE_QUEUE_AUTHORITY_SCOPE: Final = (
    "joint_fqr_certified_frozen_endpoint_to_proposal_queue_only_"
    "no_freeze_calibration_oracle_commit_or_terminal_authority_v1"
)
FQR_QUEUE_ORDER_POLICY: Final = (
    "immutable_frozen_ladder_certified_subset_descending_endpoint_and_resource_v1"
)
FQR_SINGLE_LADDER_SCOPE: Final = (
    "one_registered_prefrozen_ladder_no_cross_branch_completeness_claim_v1"
)

_ENTRY_PREFIX: Final = "partition-free-v1-fqr-candidate-queue-entry-"
_AUTHORIZATION_PREFIX: Final = "partition-free-v1-fqr-candidate-queue-authorization-"


class V1FQRCandidateQueueRefusalCode(str, Enum):
    """Stable refusal reasons at the F/Q/R-to-queue authority boundary."""

    INVALID_TYPE = "invalid_type"
    FQ_ONLY_EVIDENCE = "fq_only_evidence"
    COORDINATE_MISMATCH = "coordinate_mismatch"
    STALE_CHART = "stale_chart"
    POST_SELECTION = "post_selection"
    UNSTABLE_DECISION = "unstable_decision"
    NO_CERTIFIED_ENDPOINT = "no_certified_endpoint"
    AUTHORITY_VIOLATION = "authority_violation"


class V1FQRCandidateQueueRefusal(ValueError):
    """Typed, fail-closed refusal before proposal-queue materialization."""

    def __init__(self, code: V1FQRCandidateQueueRefusalCode, message: str) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(code: V1FQRCandidateQueueRefusalCode, message: str) -> NoReturn:
    raise V1FQRCandidateQueueRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1FQRCandidateQueueRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be one nonempty string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1FQRCandidateQueueRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1FQRCandidateQueueRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be nonnegative",
        )
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        _refuse(
            V1FQRCandidateQueueRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be positive",
        )
    return result


def _direct_mask_hash(mask: HardMask) -> str:
    """Recompute a mask hash without trusting its cached property."""

    if type(mask) is not HardMask:
        _refuse(
            V1FQRCandidateQueueRefusalCode.INVALID_TYPE,
            "the live current mask must be an exact HardMask",
        )
    return content_hash(
        {
            "route_axis_hash": mask.route_axis_hash,
            "route_count": mask.route_count,
            "packed_bits": mask.packed_bits,
        },
        prefix="irg-mask-",
    )


def _same_mask(left: HardMask, right: HardMask) -> bool:
    return (
        type(left) is HardMask
        and type(right) is HardMask
        and left.route_axis_hash == right.route_axis_hash
        and left.route_count == right.route_count
        and left.packed_bits == right.packed_bits
        and _direct_mask_hash(left) == _direct_mask_hash(right)
    )


def _translate_fqr_refusal(error: V1FQRCertificationRefusal) -> NoReturn:
    mapping = {
        V1FQRCertificationRefusalCode.INVALID_TYPE: (
            V1FQRCandidateQueueRefusalCode.INVALID_TYPE
        ),
        V1FQRCertificationRefusalCode.COORDINATE_MISMATCH: (
            V1FQRCandidateQueueRefusalCode.COORDINATE_MISMATCH
        ),
        V1FQRCertificationRefusalCode.HASH_MISMATCH: (
            V1FQRCandidateQueueRefusalCode.COORDINATE_MISMATCH
        ),
        V1FQRCertificationRefusalCode.STALE_CHART: (
            V1FQRCandidateQueueRefusalCode.STALE_CHART
        ),
        V1FQRCertificationRefusalCode.POST_SELECTION: (
            V1FQRCandidateQueueRefusalCode.POST_SELECTION
        ),
        V1FQRCertificationRefusalCode.AUTHORITY_VIOLATION: (
            V1FQRCandidateQueueRefusalCode.AUTHORITY_VIOLATION
        ),
        V1FQRCertificationRefusalCode.UNSTABLE_DECISION: (
            V1FQRCandidateQueueRefusalCode.UNSTABLE_DECISION
        ),
    }
    _refuse(mapping[error.code], f"invalid F/Q/R evidence graph: {error}")


def _require_exact_types(
    *,
    design: object,
    receipt: object,
    decision: object,
    stability_certificate: object,
    ladder: object,
    current_mask: object,
) -> None:
    # The historical F/Q receipt/decision are explicitly recognized rather
    # than falling through to a generic duck-typed path.  Canonical F/Q/R
    # queue authority can never be synthesized from their fields.
    if (
        type(receipt) is V1FrozenLadderCertificationReceipt
        or type(decision) is V1FrozenLadderDecision
    ):
        _refuse(
            V1FQRCandidateQueueRefusalCode.FQ_ONLY_EVIDENCE,
            "an F/Q-only receipt or decision cannot authorize an F/Q/R queue",
        )
    expected = (
        (design, V1FQRCertificationDesign, "design"),
        (receipt, V1FQRCertificationReceipt, "receipt"),
        (decision, V1FQRCertificationDecision, "decision"),
        (
            stability_certificate,
            V1DecisionStabilityCertificate,
            "decision-stability certificate",
        ),
        (ladder, V1FrozenCandidateLadder, "frozen ladder"),
        (current_mask, HardMask, "current mask"),
    )
    for value, expected_type, name in expected:
        if type(value) is not expected_type:
            _refuse(
                V1FQRCandidateQueueRefusalCode.INVALID_TYPE,
                f"queue authorization requires the exact typed {name}",
            )


def _preflight_causal_and_authority_flags(
    *,
    design: V1FQRCertificationDesign,
    receipt: V1FQRCertificationReceipt,
    decision: V1FQRCertificationDecision,
    stability_certificate: V1DecisionStabilityCertificate,
) -> None:
    """Reject causal/authority drift before consulting cached hashes."""

    if (
        design.issued_before_fqr_outcomes is not True
        or design.ladder_frozen_before_fqr_outcomes is not True
        or design.validation_outcomes_seen is not False
        or receipt.endpoint_order_unchanged_after_outcomes is not True
        or receipt.postselection_occurred is not False
    ):
        _refuse(
            V1FQRCandidateQueueRefusalCode.POST_SELECTION,
            "the ladder or endpoint axis was selected/reordered after F/Q/R outcomes",
        )
    if (
        design.proposal_authority != FQR_PROPOSAL_AUTHORITY
        or receipt.proposal_authority != FQR_PROPOSAL_AUTHORITY
        or decision.proposal_authority != FQR_PROPOSAL_AUTHORITY
        or design.bce_oracle_authority is not False
        or design.mask_commit_authority is not False
        or design.terminal_authority is not False
        or receipt.bce_oracle_authority is not False
        or receipt.mask_commit_authority is not False
        or receipt.terminal_authority is not False
        or decision.bce_oracle_authority is not False
        or decision.mask_commit_authority is not False
        or decision.oracle_required_before_mask_commit is not True
        or stability_certificate.candidate_endpoint_stability_only is not True
        or stability_certificate.irreversible_decision_authority is not False
        or stability_certificate.paired_bce_oracle_still_required is not True
    ):
        _refuse(
            V1FQRCandidateQueueRefusalCode.AUTHORITY_VIOLATION,
            "F/Q/R evidence crossed its proposal-screening authority boundary",
        )
    if stability_certificate.all_admissible_combinations_same is not True:
        _refuse(
            V1FQRCandidateQueueRefusalCode.UNSTABLE_DECISION,
            "registered uncertainty sets do not determine one candidate endpoint",
        )


def _preflight_endpoint_axes(
    *,
    ladder: V1FrozenCandidateLadder,
    receipt: V1FQRCertificationReceipt,
    decision: V1FQRCertificationDecision,
    stability_certificate: V1DecisionStabilityCertificate,
) -> None:
    expected = tuple(enumerate(ladder.action_hashes, start=1))
    receipt_axis = tuple(enumerate(receipt.endpoint_action_hashes, start=1))
    try:
        decision_axis = tuple((row[0], row[1]) for row in decision.endpoint_outcomes)
    except (IndexError, TypeError):
        _refuse(
            V1FQRCandidateQueueRefusalCode.POST_SELECTION,
            "decision endpoint records no longer preserve the frozen ladder axis",
        )
    if receipt_axis != expected or decision_axis != expected:
        _refuse(
            V1FQRCandidateQueueRefusalCode.POST_SELECTION,
            "receipt or decision endpoints were omitted, inserted, or reordered",
        )
    try:
        sources = tuple(
            source
            for source, _values in (
                stability_certificate.admissible_endpoint_sets_by_source
            )
        )
        axes = tuple(
            values
            for _source, values in (
                stability_certificate.admissible_endpoint_sets_by_source
            )
        )
        uncertainty_sources = tuple(
            source
            for source, _receipt_hash in (
                stability_certificate.uncertainty_receipt_hashes_by_source
            )
        )
    except (TypeError, ValueError):
        _refuse(
            V1FQRCandidateQueueRefusalCode.UNSTABLE_DECISION,
            "decision-stability source axes are malformed",
        )
    target = decision.selected_endpoint_index
    if (
        sources != FQR_STABILITY_SOURCES
        or uncertainty_sources != FQR_STABILITY_SOURCES
        or axes != tuple((target,) for _ in FQR_STABILITY_SOURCES)
        or stability_certificate.selected_endpoint_index != target
        or stability_certificate.selected_action_hash != decision.selected_action_hash
    ):
        _refuse(
            V1FQRCandidateQueueRefusalCode.UNSTABLE_DECISION,
            "not every registered uncertainty source selects the same endpoint",
        )


def _verify_evidence_graph(
    *,
    design: V1FQRCertificationDesign,
    receipt: V1FQRCertificationReceipt,
    decision: V1FQRCertificationDecision,
    stability_certificate: V1DecisionStabilityCertificate,
    ladder: V1FrozenCandidateLadder,
    current_mask: HardMask,
    live_chart_identity_hash: str,
    live_proposal_generation: int,
    live_ladder_hash: str,
) -> tuple[str, str, str, str, str, int]:
    _preflight_causal_and_authority_flags(
        design=design,
        receipt=receipt,
        decision=decision,
        stability_certificate=stability_certificate,
    )
    _preflight_endpoint_axes(
        ladder=ladder,
        receipt=receipt,
        decision=decision,
        stability_certificate=stability_certificate,
    )
    try:
        ladder_hash = ladder.assert_integrity()
        design_hash = design.assert_integrity()
        receipt_hash = verify_v1_fqr_certification_receipt(
            design=design,
            receipt=receipt,
        )
        decision_hash = decision.assert_integrity()
        stability_hash = stability_certificate.assert_integrity()
    except V1FQRCertificationRefusal as error:
        _translate_fqr_refusal(error)
    except (TypeError, ValueError) as error:
        _refuse(
            V1FQRCandidateQueueRefusalCode.COORDINATE_MISMATCH,
            f"frozen ladder integrity failed: {error}",
        )

    mask_hash = _direct_mask_hash(current_mask)
    parent = ladder.parent_mask
    if not ladder.is_packed or type(parent) is not HardMask:
        _refuse(
            V1FQRCandidateQueueRefusalCode.INVALID_TYPE,
            "canonical F/Q/R queue materialization requires a packed ladder",
        )
    if not _same_mask(parent, current_mask):
        _refuse(
            V1FQRCandidateQueueRefusalCode.COORDINATE_MISMATCH,
            "the frozen ladder parent is not the exact live current mask",
        )

    embedded_ladder = design.secant_design.ladder
    if (
        embedded_ladder.ladder_hash != ladder_hash
        or embedded_ladder.action_hashes != ladder.action_hashes
        or design.ladder_hash != ladder_hash
        or receipt.ladder_hash != ladder_hash
        or decision.ladder_hash != ladder_hash
        or stability_certificate.ladder_hash != ladder_hash
        or receipt.design_hash != design_hash
        or decision.design_hash != design_hash
        or stability_certificate.design_hash != design_hash
        or decision.receipt_hash != receipt_hash
        or stability_certificate.receipt_hash != receipt_hash
        or stability_certificate.decision_hash != decision_hash
    ):
        _refuse(
            V1FQRCandidateQueueRefusalCode.COORDINATE_MISMATCH,
            "the design, ladder, receipt, decision, and stability graph differ",
        )

    live_chart = _identity(live_chart_identity_hash, name="live chart identity hash")
    live_generation = _nonnegative_int(
        live_proposal_generation,
        name="live proposal generation",
    )
    live_ladder = _identity(live_ladder_hash, name="live ladder hash")
    chart_axis = (
        design.chart_identity_hash,
        receipt.chart_identity_hash,
        decision.chart_identity_hash,
        stability_certificate.chart_identity_hash,
    )
    mask_axis = (
        ladder.current_mask_hash,
        receipt.current_mask_hash,
        decision.current_mask_hash,
        stability_certificate.current_mask_hash,
    )
    generation_axis = (
        ladder.proposal_generation,
        receipt.proposal_generation,
        decision.proposal_generation,
        stability_certificate.proposal_generation,
    )
    if (
        any(value != live_chart for value in chart_axis)
        or any(value != mask_hash for value in mask_axis)
        or any(value != live_generation for value in generation_axis)
        or live_ladder != ladder_hash
    ):
        _refuse(
            V1FQRCandidateQueueRefusalCode.STALE_CHART,
            "F/Q/R evidence does not belong to the live chart/mask/generation/ladder",
        )
    # Replay the registered inequalities rather than trusting the decision's
    # cached booleans.  This independently re-establishes each admitted F/Q/R
    # endpoint from the receipt and the derived loss-unit radii.
    try:
        replayed_decision = decide_v1_fqr(
            design=design,
            receipt=receipt,
            expected_chart_identity_hash=live_chart,
            expected_current_mask_hash=mask_hash,
            expected_proposal_generation=live_generation,
            expected_ladder_hash=ladder_hash,
        )
    except V1FQRCertificationRefusal as error:
        _translate_fqr_refusal(error)
    if replayed_decision.decision_hash != decision_hash:
        _refuse(
            V1FQRCandidateQueueRefusalCode.COORDINATE_MISMATCH,
            "supplied F/Q/R decision differs from deterministic receipt replay",
        )
    return (
        design_hash,
        receipt_hash,
        decision_hash,
        stability_hash,
        ladder_hash,
        live_generation,
    )


@dataclass(frozen=True, slots=True)
class V1FQRCandidateQueueEntry:
    """One independently certified packed endpoint in the proposal queue."""

    endpoint_index: int
    frozen_action_hash: str
    packed_removal_action: V1PackedRemovalAction = field(repr=False, compare=False)
    cumulative_removed_count: int
    resulting_mask_hash: str
    schema: str = FQR_CANDIDATE_QUEUE_ENTRY_SCHEMA
    _entry_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        endpoint = _positive_int(self.endpoint_index, name="endpoint index")
        action_hash = _identity(self.frozen_action_hash, name="frozen action hash")
        if type(self.packed_removal_action) is not V1PackedRemovalAction:
            _refuse(
                V1FQRCandidateQueueRefusalCode.INVALID_TYPE,
                "queue entries require an exact packed removal action",
            )
        count = _positive_int(
            self.cumulative_removed_count,
            name="cumulative removed resource",
        )
        mask_hash = _identity(self.resulting_mask_hash, name="resulting mask hash")
        if (
            self.packed_removal_action.removed_count != count
            or _direct_mask_hash(self.packed_removal_action.resulting_mask) != mask_hash
            or self.schema != FQR_CANDIDATE_QUEUE_ENTRY_SCHEMA
        ):
            _refuse(
                V1FQRCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                "queue entry differs from its packed action or schema",
            )
        object.__setattr__(self, "endpoint_index", endpoint)
        object.__setattr__(self, "frozen_action_hash", action_hash)
        object.__setattr__(self, "cumulative_removed_count", count)
        object.__setattr__(self, "resulting_mask_hash", mask_hash)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "endpoint_index": self.endpoint_index,
            "frozen_action_hash": self.frozen_action_hash,
            "packed_action_hash": self.packed_removal_action.action_hash,
            "cumulative_removed_count": self.cumulative_removed_count,
            "resulting_mask_hash": self.resulting_mask_hash,
        }

    @property
    def packed_action_hash(self) -> str:
        return self.packed_removal_action.action_hash

    @property
    def entry_hash(self) -> str:
        cached = self._entry_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        if (
            self.packed_removal_action.removed_count != self.cumulative_removed_count
            or _direct_mask_hash(self.packed_removal_action.resulting_mask)
            != self.resulting_mask_hash
        ):
            _refuse(
                V1FQRCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                "queue entry packed action was recontextualized",
            )
        observed = content_hash(self._payload(), prefix=_ENTRY_PREFIX)
        cached = self._entry_hash
        if cached is not None and cached != observed:
            _refuse(
                V1FQRCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                "queue entry was tampered after hashing",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1FQRCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                "queue entry hash does not match",
            )
        object.__setattr__(self, "_entry_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1FQRCandidateQueueAuthorization:
    """Proposal-only authority for a descending view of F/Q/R endpoints."""

    design: V1FQRCertificationDesign = field(repr=False, compare=False)
    receipt: V1FQRCertificationReceipt = field(repr=False, compare=False)
    decision: V1FQRCertificationDecision = field(repr=False, compare=False)
    stability_certificate: V1DecisionStabilityCertificate = field(
        repr=False,
        compare=False,
    )
    ladder: V1FrozenCandidateLadder = field(repr=False, compare=False)
    current_mask: HardMask = field(repr=False, compare=False)
    chart_identity_hash: str
    proposal_generation: int
    certified_endpoint_indices: tuple[int, ...]
    certified_action_hashes: tuple[str, ...]
    queue_entries: tuple[V1FQRCandidateQueueEntry, ...]
    fqr_proposal_screening_authority: bool = True
    proposal_queue_materialization_authority: bool = True
    frozen_ladder_order_preserved: bool = True
    fqr_monotonicity_assumed: bool = False
    candidate_freeze_authority: bool = False
    calibration_authority: bool = False
    bce_oracle_authority: bool = False
    irreversible_decision_authority: bool = False
    mask_commit_authority: bool = False
    terminal_validation_access_authority: bool = False
    terminal_authority: bool = False
    paired_bce_oracle_still_required: bool = True
    cross_branch_completeness_authority: bool = False
    prefrozen_multiladder_family_required_for_cross_branch_selection: bool = True
    ladder_family_scope: str = FQR_SINGLE_LADDER_SCOPE
    queue_order_policy: str = FQR_QUEUE_ORDER_POLICY
    authority_scope: str = FQR_CANDIDATE_QUEUE_AUTHORITY_SCOPE
    schema: str = FQR_CANDIDATE_QUEUE_AUTHORIZATION_SCHEMA
    _authorization_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _require_exact_types(
            design=self.design,
            receipt=self.receipt,
            decision=self.decision,
            stability_certificate=self.stability_certificate,
            ladder=self.ladder,
            current_mask=self.current_mask,
        )
        chart = _identity(self.chart_identity_hash, name="chart identity hash")
        generation = _nonnegative_int(
            self.proposal_generation,
            name="proposal generation",
        )
        certified_indices = tuple(
            _positive_int(value, name="certified endpoint index")
            for value in self.certified_endpoint_indices
        )
        certified_hashes = tuple(
            _identity(value, name="certified action hash")
            for value in self.certified_action_hashes
        )
        entries = tuple(self.queue_entries)
        if any(type(entry) is not V1FQRCandidateQueueEntry for entry in entries):
            _refuse(
                V1FQRCandidateQueueRefusalCode.INVALID_TYPE,
                "authorization queue requires exact typed entries",
            )
        if (
            not certified_indices
            or certified_indices != tuple(sorted(set(certified_indices)))
            or len(certified_indices) != len(certified_hashes)
        ):
            _refuse(
                V1FQRCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                "certified endpoints must be a unique ascending frozen subset",
            )
        expected_descending = tuple(reversed(certified_indices))
        if tuple(entry.endpoint_index for entry in entries) != expected_descending:
            _refuse(
                V1FQRCandidateQueueRefusalCode.POST_SELECTION,
                "queue entries are not the exact descending certified subset",
            )
        if tuple(entry.frozen_action_hash for entry in reversed(entries)) != (
            certified_hashes
        ):
            _refuse(
                V1FQRCandidateQueueRefusalCode.POST_SELECTION,
                "queue action identities differ from the frozen certified subset",
            )
        counts = tuple(entry.cumulative_removed_count for entry in entries)
        if any(left <= right for left, right in zip(counts, counts[1:])):
            _refuse(
                V1FQRCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                "queue resource must be strictly descending with endpoint index",
            )
        flags = (
            self.fqr_proposal_screening_authority,
            self.proposal_queue_materialization_authority,
            self.frozen_ladder_order_preserved,
            not self.fqr_monotonicity_assumed,
            not self.candidate_freeze_authority,
            not self.calibration_authority,
            not self.bce_oracle_authority,
            not self.irreversible_decision_authority,
            not self.mask_commit_authority,
            not self.terminal_validation_access_authority,
            not self.terminal_authority,
            self.paired_bce_oracle_still_required,
            not self.cross_branch_completeness_authority,
            self.prefrozen_multiladder_family_required_for_cross_branch_selection,
        )
        if any(type(flag) is not bool or not flag for flag in flags):
            _refuse(
                V1FQRCandidateQueueRefusalCode.AUTHORITY_VIOLATION,
                "F/Q/R queue authorization crossed its proposal-only boundary",
            )
        if (
            self.ladder_family_scope != FQR_SINGLE_LADDER_SCOPE
            or self.queue_order_policy != FQR_QUEUE_ORDER_POLICY
            or self.authority_scope != FQR_CANDIDATE_QUEUE_AUTHORITY_SCOPE
            or self.schema != FQR_CANDIDATE_QUEUE_AUTHORIZATION_SCHEMA
        ):
            _refuse(
                V1FQRCandidateQueueRefusalCode.AUTHORITY_VIOLATION,
                "F/Q/R queue order, scope, or schema drifted",
            )
        object.__setattr__(self, "chart_identity_hash", chart)
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "certified_endpoint_indices", certified_indices)
        object.__setattr__(self, "certified_action_hashes", certified_hashes)
        object.__setattr__(self, "queue_entries", entries)
        self.assert_integrity()

    @property
    def queue_endpoint_indices_descending(self) -> tuple[int, ...]:
        return tuple(entry.endpoint_index for entry in self.queue_entries)

    @property
    def queue_action_hashes_descending(self) -> tuple[str, ...]:
        return tuple(entry.frozen_action_hash for entry in self.queue_entries)

    @property
    def queue_packed_actions_descending(self) -> tuple[V1PackedRemovalAction, ...]:
        return tuple(entry.packed_removal_action for entry in self.queue_entries)

    @property
    def queue_head_endpoint_index(self) -> int:
        return self.queue_entries[0].endpoint_index

    @property
    def queue_head_action_hash(self) -> str:
        return self.queue_entries[0].frozen_action_hash

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "design_hash": self.design.design_hash,
            "receipt_hash": self.receipt.receipt_hash,
            "decision_hash": self.decision.decision_hash,
            "stability_certificate_hash": (self.stability_certificate.certificate_hash),
            "ladder_hash": self.ladder.ladder_hash,
            "current_mask_hash": _direct_mask_hash(self.current_mask),
            "chart_identity_hash": self.chart_identity_hash,
            "proposal_generation": self.proposal_generation,
            "certified_endpoint_indices": self.certified_endpoint_indices,
            "certified_action_hashes": self.certified_action_hashes,
            "queue_entry_hashes": tuple(
                entry.entry_hash for entry in self.queue_entries
            ),
            "fqr_proposal_screening_authority": (self.fqr_proposal_screening_authority),
            "proposal_queue_materialization_authority": (
                self.proposal_queue_materialization_authority
            ),
            "frozen_ladder_order_preserved": self.frozen_ladder_order_preserved,
            "fqr_monotonicity_assumed": self.fqr_monotonicity_assumed,
            "candidate_freeze_authority": self.candidate_freeze_authority,
            "calibration_authority": self.calibration_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "irreversible_decision_authority": self.irreversible_decision_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_validation_access_authority": (
                self.terminal_validation_access_authority
            ),
            "terminal_authority": self.terminal_authority,
            "paired_bce_oracle_still_required": (self.paired_bce_oracle_still_required),
            "cross_branch_completeness_authority": (
                self.cross_branch_completeness_authority
            ),
            "prefrozen_multiladder_family_required_for_cross_branch_selection": (
                self.prefrozen_multiladder_family_required_for_cross_branch_selection
            ),
            "ladder_family_scope": self.ladder_family_scope,
            "queue_order_policy": self.queue_order_policy,
            "authority_scope": self.authority_scope,
        }

    @property
    def authorization_hash(self) -> str:
        cached = self._authorization_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        _verify_evidence_graph(
            design=self.design,
            receipt=self.receipt,
            decision=self.decision,
            stability_certificate=self.stability_certificate,
            ladder=self.ladder,
            current_mask=self.current_mask,
            live_chart_identity_hash=self.chart_identity_hash,
            live_proposal_generation=self.proposal_generation,
            live_ladder_hash=self.ladder.ladder_hash,
        )
        expected_indices, expected_hashes, expected_entries = _materialize_entries(
            ladder=self.ladder,
            current_mask=self.current_mask,
            decision=self.decision,
        )
        if (
            self.certified_endpoint_indices != expected_indices
            or self.certified_action_hashes != expected_hashes
            or tuple(entry.entry_hash for entry in self.queue_entries)
            != tuple(entry.entry_hash for entry in expected_entries)
        ):
            _refuse(
                V1FQRCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                "authorization queue no longer equals independently certified endpoints",
            )
        observed = content_hash(self._payload(), prefix=_AUTHORIZATION_PREFIX)
        cached = self._authorization_hash
        if cached is not None and cached != observed:
            _refuse(
                V1FQRCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                "F/Q/R queue authorization was tampered after hashing",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1FQRCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                "F/Q/R queue authorization hash does not match",
            )
        object.__setattr__(self, "_authorization_hash", observed)
        return observed


def _materialize_entries(
    *,
    ladder: V1FrozenCandidateLadder,
    current_mask: HardMask,
    decision: V1FQRCertificationDecision,
) -> tuple[
    tuple[int, ...],
    tuple[str, ...],
    tuple[V1FQRCandidateQueueEntry, ...],
]:
    certified_rows = tuple(row for row in decision.endpoint_outcomes if row[11])
    if (
        decision.status != "certified_largest_endpoint"
        or decision.selected_endpoint_index is None
        or decision.selected_action_hash is None
        or not certified_rows
    ):
        _refuse(
            V1FQRCandidateQueueRefusalCode.NO_CERTIFIED_ENDPOINT,
            f"F/Q/R status {decision.status!r} cannot materialize a queue",
        )
    certified_indices = tuple(int(row[0]) for row in certified_rows)
    certified_hashes = tuple(str(row[1]) for row in certified_rows)
    certified_hash_by_index = dict(
        zip(certified_indices, certified_hashes, strict=True)
    )
    if (
        decision.selected_endpoint_index != certified_indices[-1]
        or decision.selected_action_hash != certified_hashes[-1]
    ):
        _refuse(
            V1FQRCandidateQueueRefusalCode.COORDINATE_MISMATCH,
            "selected endpoint is not the largest independently certified rung",
        )

    entries: list[V1FQRCandidateQueueEntry] = []
    for endpoint in reversed(certified_indices):
        frozen = ladder.actions[endpoint - 1]
        packed = frozen.packed_removal_action
        if type(packed) is not V1PackedRemovalAction:
            _refuse(
                V1FQRCandidateQueueRefusalCode.INVALID_TYPE,
                "a certified endpoint is not an exact packed removal action",
            )
        resolved_count, resolved_hash = (
            packed.resulting_mask.removed_index_count_and_hash_from(current_mask)
        )
        if (
            resolved_count != packed.removed_count
            or resolved_hash != packed.removed_route_index_hash
            or frozen.action_hash != certified_hash_by_index[endpoint]
        ):
            _refuse(
                V1FQRCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                "certified packed action differs from its frozen parent-mask delta",
            )
        entries.append(
            V1FQRCandidateQueueEntry(
                endpoint_index=endpoint,
                frozen_action_hash=frozen.action_hash,
                packed_removal_action=packed,
                cumulative_removed_count=packed.removed_count,
                resulting_mask_hash=_direct_mask_hash(packed.resulting_mask),
            )
        )
    return certified_indices, certified_hashes, tuple(entries)


def authorize_v1_fqr_candidate_queue(
    *,
    design: V1FQRCertificationDesign,
    receipt: V1FQRCertificationReceipt,
    decision: V1FQRCertificationDecision,
    stability_certificate: V1DecisionStabilityCertificate,
    frozen_ladder: V1FrozenCandidateLadder,
    current_mask: HardMask,
    live_chart_identity_hash: str,
    live_proposal_generation: int,
    live_ladder_hash: str,
) -> V1FQRCandidateQueueAuthorization:
    """Authorize a proposal-only queue from one stable joint F/Q/R graph."""

    _require_exact_types(
        design=design,
        receipt=receipt,
        decision=decision,
        stability_certificate=stability_certificate,
        ladder=frozen_ladder,
        current_mask=current_mask,
    )
    _verify_evidence_graph(
        design=design,
        receipt=receipt,
        decision=decision,
        stability_certificate=stability_certificate,
        ladder=frozen_ladder,
        current_mask=current_mask,
        live_chart_identity_hash=live_chart_identity_hash,
        live_proposal_generation=live_proposal_generation,
        live_ladder_hash=live_ladder_hash,
    )
    indices, hashes, entries = _materialize_entries(
        ladder=frozen_ladder,
        current_mask=current_mask,
        decision=decision,
    )
    return V1FQRCandidateQueueAuthorization(
        design=design,
        receipt=receipt,
        decision=decision,
        stability_certificate=stability_certificate,
        ladder=frozen_ladder,
        current_mask=current_mask,
        chart_identity_hash=live_chart_identity_hash,
        proposal_generation=live_proposal_generation,
        certified_endpoint_indices=indices,
        certified_action_hashes=hashes,
        queue_entries=entries,
    )


__all__ = [
    "FQR_CANDIDATE_QUEUE_AUTHORIZATION_SCHEMA",
    "FQR_CANDIDATE_QUEUE_AUTHORITY_SCOPE",
    "FQR_CANDIDATE_QUEUE_ENTRY_SCHEMA",
    "FQR_QUEUE_ORDER_POLICY",
    "FQR_SINGLE_LADDER_SCOPE",
    "V1FQRCandidateQueueAuthorization",
    "V1FQRCandidateQueueEntry",
    "V1FQRCandidateQueueRefusal",
    "V1FQRCandidateQueueRefusalCode",
    "authorize_v1_fqr_candidate_queue",
]
