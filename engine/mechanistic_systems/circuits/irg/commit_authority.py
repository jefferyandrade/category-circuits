"""Typed authority boundary for complete hard-mask commits.

The common IRG boundary search is useful guidance, but its historical
mean/standard-error rule is not a claim-bearing sequential test.  This module
keeps that exploratory path intact while providing a content-addressed gate
through which a shadow glocal-v3 campaign can authorize the exact physical
parent-to-candidate transition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Mapping, Protocol, runtime_checkable

from .contracts import canonical_json, content_hash

if TYPE_CHECKING:
    from .candidates import ActionCandidate
    from .state import PruningSessionState


CommitAuthorityOutcome = Literal["accepted", "rejected", "unresolved"]
CommitAuthorityMode = Literal["exploratory", "glocal_v3_shadow"]
_LIVE_COMMIT_AUTHORITY_TOKEN = object()


def _require_sha256(value: str, label: str) -> str:
    identity = str(value)
    if len(identity) != 64 or any(
        character not in "0123456789abcdef" for character in identity
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return identity


@dataclass(frozen=True, slots=True)
class ExactHardMaskCommitDecision:
    """One immutable authority verdict for an exact IRG mask transition."""

    authority_id: str
    outcome: CommitAuthorityOutcome
    reason: str
    proposal_id: str
    candidate_id: str
    irg_parent_state_hash: str
    parent_mask_hash: str
    candidate_mask_hash: str
    candidate_definition_hash: str
    campaign_receipt_hash: str
    registration_hash: str | None = None
    decision_evidence_receipt_hash: str | None = None
    statistical_parent_state_hash: str | None = None
    statistical_candidate_state_hash: str | None = None
    parent_calibration_hash: str | None = None
    candidate_calibration_hash: str | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    local_upper_boundary: float | None = None
    distortion_debit: float | None = None
    distortion_spent_after: float | None = None
    distortion_budget: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_name: str = "irg_exact_hard_mask_commit_decision_v1"

    def __post_init__(self) -> None:
        if (
            not self.authority_id
            or not self.reason
            or not self.proposal_id
            or not self.candidate_id
            or not self.irg_parent_state_hash
            or not self.parent_mask_hash
            or not self.candidate_mask_hash
            or not self.schema_name
        ):
            raise ValueError("commit-authority decision has an empty identity field")
        if self.outcome not in {"accepted", "rejected", "unresolved"}:
            raise ValueError("commit-authority decision has an invalid outcome")
        _require_sha256(
            self.candidate_definition_hash,
            "candidate definition hash",
        )
        _require_sha256(self.campaign_receipt_hash, "campaign receipt hash")
        for value, label in (
            (self.registration_hash, "proposal registration hash"),
            (
                self.decision_evidence_receipt_hash,
                "decision evidence receipt hash",
            ),
            (self.statistical_parent_state_hash, "statistical parent state hash"),
            (
                self.statistical_candidate_state_hash,
                "statistical candidate state hash",
            ),
            (self.parent_calibration_hash, "parent calibration hash"),
            (self.candidate_calibration_hash, "candidate calibration hash"),
        ):
            if value is not None:
                _require_sha256(value, label)
        numerical = (
            self.lower_bound,
            self.upper_bound,
            self.local_upper_boundary,
            self.distortion_debit,
            self.distortion_spent_after,
            self.distortion_budget,
        )
        if any(
            value is not None and not math.isfinite(float(value)) for value in numerical
        ):
            raise ValueError("commit-authority decision contains a nonfinite value")
        if any(
            value is not None and float(value) < 0.0
            for value in (
                self.distortion_debit,
                self.distortion_spent_after,
                self.distortion_budget,
            )
        ):
            raise ValueError("commit-authority distortion values must be nonnegative")
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and float(self.lower_bound) > float(self.upper_bound) + 1e-12
        ):
            raise ValueError("commit-authority interval is inverted")
        if self.outcome in {"accepted", "rejected"}:
            required = (
                self.registration_hash,
                self.decision_evidence_receipt_hash,
                self.statistical_parent_state_hash,
                self.statistical_candidate_state_hash,
                self.parent_calibration_hash,
                self.candidate_calibration_hash,
                self.lower_bound,
                self.upper_bound,
                self.local_upper_boundary,
                self.distortion_debit,
                self.distortion_spent_after,
                self.distortion_budget,
            )
            if any(value is None for value in required):
                raise ValueError(
                    "terminal commit-authority verdict lacks its evidence lineage"
                )
        if self.outcome == "accepted":
            assert self.upper_bound is not None
            assert self.local_upper_boundary is not None
            assert self.distortion_debit is not None
            assert self.distortion_spent_after is not None
            assert self.distortion_budget is not None
            if self.upper_bound > self.local_upper_boundary + 1e-12:
                raise ValueError(
                    "accepted authority verdict exceeds its local boundary"
                )
            if not math.isclose(
                self.distortion_debit,
                max(self.upper_bound, 0.0),
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "accepted authority debit must equal max(upper bound, zero)"
                )
            if self.distortion_spent_after > self.distortion_budget + 1e-12:
                raise ValueError(
                    "accepted authority verdict exceeds the campaign distortion budget"
                )
        elif self.distortion_debit not in {None, 0, 0.0}:
            raise ValueError("a nonaccepted authority verdict cannot debit distortion")
        object.__setattr__(self, "metadata", dict(self.metadata))
        canonical_json(self.metadata)

    @property
    def accepted(self) -> bool:
        return self.outcome == "accepted"

    @property
    def content_hash(self) -> str:
        return content_hash(self, prefix="irg-commit-authority-")

    def validate_transition(
        self,
        *,
        candidate_id: str,
        irg_parent_state_hash: str,
        parent_mask_hash: str,
        candidate_mask_hash: str,
    ) -> None:
        """Fail closed if an authority verdict is replayed onto another action."""

        observed = (
            str(candidate_id),
            str(irg_parent_state_hash),
            str(parent_mask_hash),
            str(candidate_mask_hash),
        )
        expected = (
            self.candidate_id,
            self.irg_parent_state_hash,
            self.parent_mask_hash,
            self.candidate_mask_hash,
        )
        if observed != expected:
            raise ValueError(
                "commit-authority verdict does not bind the selected exact transition"
            )


class ExactHardMaskCommitAuthorization:
    """Single-use live capability for one evidence-bound exact-mask commit.

    ``ExactHardMaskCommitDecision`` is a detached, content-addressed receipt and
    therefore cannot itself carry runtime authority.  This handle retains the
    exact authority, campaign, session, and candidate owners and revalidates
    their live graph immediately before the state transition.  Its constructor
    is token-gated; only the physical glocal issuer below can mint it.
    """

    __slots__ = (
        "_authority_owner",
        "_campaign_owner",
        "_candidate_owner",
        "_consumed",
        "_decision",
        "_runtime_owner",
        "_session_owner",
    )

    def __init__(
        self,
        *,
        authority_owner: object,
        runtime_owner: object,
        campaign_owner: object,
        session_owner: "PruningSessionState",
        candidate_owner: "ActionCandidate",
        decision: ExactHardMaskCommitDecision,
        _token: object,
    ) -> None:
        if _token is not _LIVE_COMMIT_AUTHORITY_TOKEN:
            raise TypeError(
                "live exact-mask commit authorization must be issued by the "
                "physical glocal campaign authority"
            )
        _require_physical_glocal_commit_authority(authority_owner)
        runtime = getattr(authority_owner, "runtime", None)
        if runtime is not runtime_owner or getattr(runtime, "campaign", None) is not (
            campaign_owner
        ):
            raise TypeError(
                "live exact-mask commit authorization owners are not one exact "
                "physical glocal graph"
            )
        self._authority_owner = authority_owner
        self._runtime_owner = runtime_owner
        self._campaign_owner = campaign_owner
        self._session_owner = session_owner
        self._candidate_owner = candidate_owner
        self._decision = decision
        self._consumed = False
        self._validate_live_graph(session=session_owner, candidate=candidate_owner)

    def __setattr__(self, name: str, value: Any) -> None:
        if hasattr(self, name):
            raise AttributeError("live exact-mask commit authorization is immutable")
        object.__setattr__(self, name, value)

    def __reduce__(self) -> object:
        raise TypeError("live exact-mask commit authorization is nonserializable")

    @property
    def decision(self) -> ExactHardMaskCommitDecision:
        return self._decision

    @property
    def consumed(self) -> bool:
        return bool(self._consumed)

    def _validate_live_graph(
        self,
        *,
        session: "PruningSessionState",
        candidate: "ActionCandidate",
    ) -> None:
        """Invoke the exact nominal authority validator, never caller code."""

        from mechanistic_systems.circuits.evidence_glocal import (
            GlocalIRGCommitAuthority,
        )

        _require_physical_glocal_commit_authority(self._authority_owner)
        validator = GlocalIRGCommitAuthority._revalidate_commit_authorization
        validator(
            self._authority_owner,
            runtime_owner=self._runtime_owner,
            campaign_owner=self._campaign_owner,
            session=session,
            candidate=candidate,
            decision=self._decision,
        )

    def require_current(
        self,
        *,
        session: "PruningSessionState",
        candidate: "ActionCandidate",
    ) -> ExactHardMaskCommitDecision:
        """Revalidate the exact live campaign graph without consuming the handle."""

        if self._consumed:
            raise RuntimeError("exact-mask commit authorization was already consumed")
        if session is not self._session_owner or candidate is not self._candidate_owner:
            raise ValueError(
                "exact-mask commit authorization belongs to another live "
                "session or candidate"
            )
        self._validate_live_graph(session=session, candidate=candidate)
        return self._decision

    def consume(
        self,
        *,
        session: "PruningSessionState",
        candidate: "ActionCandidate",
    ) -> ExactHardMaskCommitDecision:
        """Revalidate and irreversibly spend this one-transition capability."""

        decision = self.require_current(session=session, candidate=candidate)
        object.__setattr__(self, "_consumed", True)
        return decision


def _require_physical_glocal_commit_authority(authority: object) -> None:
    """Refuse structural lookalikes at the shadow commit boundary."""

    # Import lazily to avoid commit_authority -> evidence_glocal -> runner
    # import cycles.  Exact nominal types are intentional: a caller-supplied
    # object with the right string identity is not a statistical authority.
    from mechanistic_systems.circuits.evidence_glocal import (
        GlocalBrierCampaignRuntime,
        GlocalIRGCommitAuthority,
    )
    from mechanistic_systems.circuits.sampled_torch import (
        TorchSampledPathEvaluationBackend,
    )
    from mechanistic_systems.evidence import CampaignEvidenceState

    if type(authority) is not GlocalIRGCommitAuthority:
        raise TypeError(
            "shadow glocal-v3 execution requires the exact physical "
            "GlocalIRGCommitAuthority, not an authority-id lookalike"
        )
    runtime = getattr(authority, "runtime", None)
    if type(runtime) is not GlocalBrierCampaignRuntime:
        raise TypeError(
            "shadow glocal-v3 commit authority is not tied to the exact "
            "GlocalBrierCampaignRuntime"
        )
    if type(getattr(runtime, "campaign", None)) is not CampaignEvidenceState:
        raise TypeError(
            "shadow glocal-v3 authority lacks the exact live campaign owner"
        )
    if type(getattr(runtime, "backend", None)) is not TorchSampledPathEvaluationBackend:
        raise TypeError(
            "shadow glocal-v3 authority lacks the exact Torch sampled-path backend"
        )


def _issue_physical_exact_mask_commit_authorization(
    *,
    authority_owner: object,
    runtime_owner: object,
    campaign_owner: object,
    session_owner: "PruningSessionState",
    candidate_owner: "ActionCandidate",
    decision: ExactHardMaskCommitDecision,
) -> ExactHardMaskCommitAuthorization:
    """Private issuer for one live glocal exact-mask transition."""

    from mechanistic_systems.circuits.evidence_glocal import (
        GlocalBrierCampaignRuntime,
        GlocalIRGCommitAuthority,
    )
    from mechanistic_systems.evidence import CampaignEvidenceState

    _require_physical_glocal_commit_authority(authority_owner)
    if type(runtime_owner) is not GlocalBrierCampaignRuntime:
        raise TypeError(
            "exact-mask commit authorization requires the exact glocal runtime"
        )
    if type(campaign_owner) is not CampaignEvidenceState:
        raise TypeError(
            "exact-mask commit authorization requires the exact campaign state"
        )
    if (
        authority_owner.runtime is not runtime_owner
        or runtime_owner.campaign is not campaign_owner
    ):
        raise RuntimeError(
            "exact-mask commit authority, runtime, and campaign are not one "
            "live ownership graph"
        )
    GlocalIRGCommitAuthority._revalidate_commit_authorization(
        authority_owner,
        runtime_owner=runtime_owner,
        campaign_owner=campaign_owner,
        session=session_owner,
        candidate=candidate_owner,
        decision=decision,
    )
    return ExactHardMaskCommitAuthorization(
        authority_owner=authority_owner,
        runtime_owner=runtime_owner,
        campaign_owner=campaign_owner,
        session_owner=session_owner,
        candidate_owner=candidate_owner,
        decision=decision,
        _token=_LIVE_COMMIT_AUTHORITY_TOKEN,
    )


@runtime_checkable
class ExactHardMaskCommitAuthority(Protocol):
    """Provider of one evidence-bound verdict for a selected complete mask."""

    @property
    def authority_id(self) -> str: ...

    def authorize(
        self,
        *,
        session: "PruningSessionState",
        candidate: "ActionCandidate",
    ) -> ExactHardMaskCommitDecision: ...


__all__ = [
    "CommitAuthorityMode",
    "CommitAuthorityOutcome",
    "ExactHardMaskCommitAuthority",
    "ExactHardMaskCommitDecision",
]
