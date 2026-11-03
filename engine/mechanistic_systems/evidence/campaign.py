"""Campaign-global ledgers for glocal v3 sequential pruning safety.

Alpha is reserved when a predictable proposal is born.  It is never refunded
after rejection, abandonment, or supersession.  Accepted moves debit only the
nonnegative upper confidence bound, so statistical credits cannot hide later
distortion.  One campaign object owns every query/group proposal and resource
counter, preventing a reset-at-group composition bug.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Literal

from .finite_population import (
    STRATIFIED_WOR_CS_METHOD,
    StratifiedCSSnapshot,
    StratifiedPairedEProcess,
)
from .glocal_contracts import (
    CalibrationFitState,
    EvidenceRole,
    JOINT_RISK_QUERY_ID,
    ObservationSetRef,
    PredictableAllocationLookReceipt,
    PredictableProposalRegistration,
    RiskEstimandSpec,
    SamplingDesignSpec,
    TerminalAuditEvidenceReceipt,
    TerminalAuditPrecommitment,
    TerminalParityCertificate,
    ValidationLock,
    content_hash,
)


ProposalOutcome = Literal[
    "registered",
    "accepted",
    "rejected",
    "unresolved",
    "abandoned",
    "superseded",
]

_ADDITIVE_RESOURCE_FIELDS = (
    "unique_contexts",
    "context_state_evaluations",
    "forward_passes",
    "backward_passes",
    "calibration_fits",
    "host_to_device_bytes",
    "device_to_host_bytes",
    "wall_seconds",
    "device_seconds",
    "prefetch_contexts",
    "cache_hits",
)
_RESOURCE_FIELDS = _ADDITIVE_RESOURCE_FIELDS + ("peak_memory_bytes",)


def _require_sha256(value: str, label: str) -> str:
    identity = str(value)
    if len(identity) != 64 or any(
        character not in "0123456789abcdef" for character in identity
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return identity


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    unique_contexts: int = 0
    context_state_evaluations: int = 0
    forward_passes: int = 0
    backward_passes: int = 0
    calibration_fits: int = 0
    host_to_device_bytes: int = 0
    device_to_host_bytes: int = 0
    wall_seconds: float = 0.0
    device_seconds: float = 0.0
    peak_memory_bytes: int = 0
    prefetch_contexts: int = 0
    cache_hits: int = 0

    def __post_init__(self) -> None:
        for field_name in _RESOURCE_FIELDS:
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"resource usage {field_name} must be finite and nonnegative"
                )
        integer_fields = set(_RESOURCE_FIELDS) - {"wall_seconds", "device_seconds"}
        for field_name in integer_fields:
            value = getattr(self, field_name)
            if int(value) != value:
                raise ValueError(f"resource usage {field_name} must be integral")

    def to_dict(self) -> dict[str, int | float]:
        return {name: getattr(self, name) for name in _RESOURCE_FIELDS}


class ResourceLedger:
    """Hard-cap resource ledger shared by every proposal in a campaign."""

    def __init__(self, *, caps: Mapping[str, int | float] | None = None) -> None:
        self._caps = {str(key): float(value) for key, value in dict(caps or {}).items()}
        unknown = set(self._caps) - set(_RESOURCE_FIELDS)
        if unknown:
            raise ValueError("unknown resource cap(s): " + ",".join(sorted(unknown)))
        if any(
            not math.isfinite(value) or value < 0.0 for value in self._caps.values()
        ):
            raise ValueError("resource caps must be finite and nonnegative")
        self._usage = ResourceUsage()
        self._seen_context_ids: set[str] = set()
        self._seen_context_state_pairs: set[tuple[str, str]] = set()

    @property
    def usage(self) -> ResourceUsage:
        return self._usage

    @property
    def caps(self) -> dict[str, float]:
        return dict(self._caps)

    def _combined(self, delta: ResourceUsage) -> ResourceUsage:
        current = self._usage.to_dict()
        incoming = delta.to_dict()
        combined = {
            name: (
                max(current[name], incoming[name])
                if name == "peak_memory_bytes"
                else current[name] + incoming[name]
            )
            for name in _RESOURCE_FIELDS
        }
        return ResourceUsage(**combined)

    def can_consume(self, delta: ResourceUsage) -> bool:
        prospective = self._combined(delta)
        return all(
            float(getattr(prospective, name)) <= cap + 1e-12
            for name, cap in self._caps.items()
        )

    def consume(self, delta: ResourceUsage) -> ResourceUsage:
        prospective = self._combined(delta)
        exceeded = [
            name
            for name, cap in self._caps.items()
            if float(getattr(prospective, name)) > cap + 1e-12
        ]
        if exceeded:
            raise RuntimeError(
                "resource cap would be exceeded before the next packet: "
                + ",".join(sorted(exceeded))
            )
        self._usage = prospective
        return self._usage

    def record_context_state_evaluations(
        self,
        *,
        context_ids: tuple[str, ...],
        state_hashes: tuple[str, ...],
        forward_passes: int = 0,
        backward_passes: int = 0,
        calibration_fits: int = 0,
        host_to_device_bytes: int = 0,
        device_to_host_bytes: int = 0,
        wall_seconds: float = 0.0,
        device_seconds: float = 0.0,
        peak_memory_bytes: int = 0,
        prefetched: bool = False,
        cache_hits: int = 0,
    ) -> ResourceUsage:
        delta, contexts, pairs = self.preview_context_state_evaluations(
            context_ids=context_ids,
            state_hashes=state_hashes,
            forward_passes=int(forward_passes),
            backward_passes=int(backward_passes),
            calibration_fits=int(calibration_fits),
            host_to_device_bytes=int(host_to_device_bytes),
            device_to_host_bytes=int(device_to_host_bytes),
            wall_seconds=float(wall_seconds),
            device_seconds=float(device_seconds),
            peak_memory_bytes=int(peak_memory_bytes),
            prefetched=bool(prefetched),
            cache_hits=int(cache_hits),
        )
        usage = self.consume(delta)
        self._seen_context_ids.update(contexts)
        self._seen_context_state_pairs.update(pairs)
        return usage

    def preview_context_state_evaluations(
        self,
        *,
        context_ids: tuple[str, ...],
        state_hashes: tuple[str, ...],
        forward_passes: int = 0,
        backward_passes: int = 0,
        calibration_fits: int = 0,
        host_to_device_bytes: int = 0,
        device_to_host_bytes: int = 0,
        wall_seconds: float = 0.0,
        device_seconds: float = 0.0,
        peak_memory_bytes: int = 0,
        prefetched: bool = False,
        cache_hits: int = 0,
    ) -> tuple[
        ResourceUsage,
        tuple[str, ...],
        set[tuple[str, str]],
    ]:
        """Return the exact deduplicated delta without mutating the ledger."""

        contexts = tuple(str(value) for value in context_ids)
        states = tuple(str(value) for value in state_hashes)
        if len(contexts) != len(set(contexts)):
            raise ValueError("resource preview context IDs must be unique")
        if not states or any(not value for value in states):
            raise ValueError("resource preview requires execution-state identities")
        new_contexts = set(contexts) - self._seen_context_ids
        pairs = {(context, state) for context in contexts for state in states}
        new_pairs = pairs - self._seen_context_state_pairs
        delta = ResourceUsage(
            unique_contexts=len(new_contexts),
            context_state_evaluations=len(new_pairs),
            forward_passes=int(forward_passes),
            backward_passes=int(backward_passes),
            calibration_fits=int(calibration_fits),
            host_to_device_bytes=int(host_to_device_bytes),
            device_to_host_bytes=int(device_to_host_bytes),
            wall_seconds=float(wall_seconds),
            device_seconds=float(device_seconds),
            peak_memory_bytes=int(peak_memory_bytes),
            prefetch_contexts=len(new_contexts) if prefetched else 0,
            cache_hits=int(cache_hits),
        )
        return delta, contexts, pairs

    def receipt(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_name": "glocal_resource_ledger_v1",
            "caps": dict(self._caps),
            "usage": self._usage.to_dict(),
            "seen_context_count": len(self._seen_context_ids),
            "seen_context_state_pair_count": len(self._seen_context_state_pairs),
        }
        payload["ledger_hash"] = content_hash(payload)
        return payload


@dataclass(frozen=True, slots=True)
class DecisionEvidenceReceipt:
    proposal_registration_hash: str
    proposal_id: str
    look_index: int
    evidence_snapshot_hash: str
    allocation_look_receipt_hash: str
    sampling_design_hash: str
    adjudication_observation_set_hash: str
    context_prefix_hash: str
    stratum_alpha: tuple[tuple[str, float], ...]
    evidence_method: str
    lower_bound: float
    upper_bound: float
    local_upper_boundary: float
    risk_query_weight: float
    alpha_reserved: float
    outcome: ProposalOutcome
    distortion_debit: float
    distortion_spent_before: float
    distortion_spent_after: float
    resource_ledger_hash: str
    panel_prefix_hashes: tuple[str, ...] = ()
    schema_name: str = "glocal_decision_evidence_receipt_v2"

    def __post_init__(self) -> None:
        for value, label in (
            (self.proposal_registration_hash, "proposal registration hash"),
            (self.evidence_snapshot_hash, "evidence snapshot hash"),
            (
                self.allocation_look_receipt_hash,
                "allocation look receipt hash",
            ),
            (self.sampling_design_hash, "sampling design hash"),
            (
                self.adjudication_observation_set_hash,
                "adjudication observation set hash",
            ),
            (self.context_prefix_hash, "context prefix hash"),
            (self.resource_ledger_hash, "resource ledger hash"),
        ):
            _require_sha256(value, label)
        if not self.proposal_id or int(self.look_index) <= 0:
            raise ValueError(
                "decision receipt requires proposal identity and positive look"
            )
        values = (
            self.lower_bound,
            self.upper_bound,
            self.local_upper_boundary,
            self.risk_query_weight,
            self.alpha_reserved,
            self.distortion_debit,
            self.distortion_spent_before,
            self.distortion_spent_after,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("decision receipt values must be finite")
        if not 0.0 < self.alpha_reserved < 1.0:
            raise ValueError("decision receipt alpha must lie in (0,1)")
        if not 0.0 < self.risk_query_weight <= 1.0:
            raise ValueError("decision receipt query weight must lie in (0,1]")
        if self.outcome not in {
            "accepted",
            "rejected",
            "unresolved",
            "abandoned",
            "superseded",
        }:
            raise ValueError("decision receipt has an invalid outcome")
        if self.evidence_method != STRATIFIED_WOR_CS_METHOD:
            raise ValueError("decision receipt has an unsupported evidence method")
        stratum_alpha = tuple(
            (str(stratum_id), float(value)) for stratum_id, value in self.stratum_alpha
        )
        if (
            not stratum_alpha
            or len({stratum_id for stratum_id, _ in stratum_alpha})
            != len(stratum_alpha)
            or any(not 0.0 < value < 1.0 for _, value in stratum_alpha)
        ):
            raise ValueError("decision receipt stratum alpha axis is invalid")
        if not math.isclose(
            math.fsum(value for _, value in stratum_alpha),
            self.alpha_reserved,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "decision receipt stratum alpha does not exhaust its reservation"
            )
        if self.distortion_debit < 0.0:
            raise ValueError("decision receipt cannot contain a statistical credit")
        if not math.isclose(
            self.distortion_spent_after,
            self.distortion_spent_before + self.distortion_debit,
            abs_tol=1e-12,
        ):
            raise ValueError("decision receipt distortion ledger does not reconcile")
        if self.outcome == "accepted":
            if self.upper_bound > self.local_upper_boundary + 1e-12:
                raise ValueError("accepted proposal exceeds its frozen local boundary")
            if not math.isclose(
                self.distortion_debit, max(self.upper_bound, 0.0), abs_tol=1e-12
            ):
                raise ValueError("accepted proposal debit must equal max(upper,0)")
        elif self.distortion_debit != 0.0:
            raise ValueError("nonaccepted proposals cannot debit distortion")
        object.__setattr__(self, "look_index", int(self.look_index))
        object.__setattr__(
            self,
            "panel_prefix_hashes",
            tuple(str(value) for value in self.panel_prefix_hashes),
        )
        for panel_hash in self.panel_prefix_hashes:
            _require_sha256(panel_hash, "panel prefix hash")
        if len(self.panel_prefix_hashes) != len(set(self.panel_prefix_hashes)):
            raise ValueError("decision receipt panel prefix hashes must be unique")
        object.__setattr__(self, "stratum_alpha", tuple(sorted(stratum_alpha)))

    @property
    def content_hash(self) -> str:
        return content_hash(self)


class CampaignEvidenceState:
    """One global alpha, distortion, proposal, and resource owner."""

    def __init__(
        self,
        *,
        campaign_id: str,
        risk_estimand: RiskEstimandSpec,
        initial_state_hash: str,
        initial_calibration: CalibrationFitState,
        distortion_budget: float,
        total_epsilon: float = 0.05,
        sequential_alpha: float = 0.04,
        terminal_alpha: float = 0.01,
        maximum_proposals: int = 512,
        resource_caps: Mapping[str, int | float] | None = None,
        observation_sets: Sequence[ObservationSetRef] = (),
        calibration_sampling_design: SamplingDesignSpec | None = None,
        terminal_precommitment: TerminalAuditPrecommitment | None = None,
    ) -> None:
        if not str(campaign_id):
            raise ValueError("campaign requires campaign_id")
        _require_sha256(initial_state_hash, "campaign initial state hash")
        initial_calibration.require_claim_bearing()
        query_weights = tuple(risk_estimand.query_weights)
        budget = float(distortion_budget)
        total = float(total_epsilon)
        sequential = float(sequential_alpha)
        terminal = float(terminal_alpha)
        maximum = int(maximum_proposals)
        if not math.isfinite(budget) or budget < 0.0:
            raise ValueError(
                "campaign distortion budget must be finite and nonnegative"
            )
        if not 0.0 < total < 1.0:
            raise ValueError("campaign total epsilon must lie in (0,1)")
        if not 0.0 < sequential < 1.0 or not 0.0 < terminal < 1.0:
            raise ValueError("campaign sequential/terminal alpha must lie in (0,1)")
        if sequential + terminal > total + 1e-15:
            raise ValueError("campaign alpha split exceeds total epsilon")
        if maximum <= 0:
            raise ValueError("campaign maximum_proposals must be positive")
        self.campaign_id = str(campaign_id)
        self.risk_estimand = risk_estimand
        self.initial_state_hash = str(initial_state_hash)
        self.initial_calibration_hash = initial_calibration.content_hash
        self.distortion_budget = budget
        self.total_epsilon = total
        self.sequential_alpha = sequential
        self.terminal_alpha = terminal
        self.maximum_proposals = maximum
        self.query_ids = tuple(str(query_id) for query_id, _ in query_weights)
        self.query_id = (
            self.query_ids[0] if len(self.query_ids) == 1 else JOINT_RISK_QUERY_ID
        )
        self.resource_ledger = ResourceLedger(caps=resource_caps)
        if terminal_precommitment is not None:
            if not isinstance(
                terminal_precommitment,
                TerminalAuditPrecommitment,
            ):
                raise TypeError(
                    "campaign terminal precommitment must use the typed contract"
                )
            estimand = terminal_precommitment.estimand
            if (
                estimand.reference_state_hash != str(initial_state_hash)
                or estimand.reference_calibration_hash
                != initial_calibration.content_hash
                or estimand.adjudication_population_hash
                != risk_estimand.context_population_hash
                or estimand.prediction_rule_hash
                != risk_estimand.prediction_rule.content_hash
                or not math.isclose(
                    terminal_precommitment.terminal_alpha,
                    terminal,
                    abs_tol=1e-15,
                )
            ):
                raise ValueError(
                    "terminal precommitment differs from the campaign-birth "
                    "state, risk, or alpha allocation"
                )
        self._terminal_precommitment = terminal_precommitment
        if calibration_sampling_design is not None and not isinstance(
            calibration_sampling_design,
            SamplingDesignSpec,
        ):
            raise TypeError(
                "campaign calibration sampling design must use the typed contract"
            )
        self._calibration_sampling_design = calibration_sampling_design
        self._terminal_support_context_ids = (
            frozenset()
            if terminal_precommitment is None
            else frozenset(terminal_precommitment.support_context_ids)
        )

        self._observation_sets_by_id: dict[str, ObservationSetRef] = {}
        self._observation_sets_by_hash: dict[str, ObservationSetRef] = {}
        self._nonclaim_guidance_reuse: list[dict[str, object]] = []
        self._sampling_designs: dict[str, SamplingDesignSpec] = {}
        self._adjudication_sets: dict[str, ObservationSetRef] = {}
        self._last_draw_history: dict[str, tuple[tuple[str, str], ...]] = {}
        self._allocation_receipts: dict[str, list[PredictableAllocationLookReceipt]] = (
            {}
        )
        self._pending_allocation_receipts: dict[
            str, PredictableAllocationLookReceipt
        ] = {}
        self._terminal_lock: ValidationLock | None = None
        self._terminal_lock_registered_hash: str | None = None
        self._terminal_certificate: TerminalParityCertificate | None = None
        self._terminal_audit_evidence_receipt: TerminalAuditEvidenceReceipt | None = (
            None
        )
        self._terminal_audit_failure: dict[str, str] | None = None
        for observation_set in observation_sets:
            self.register_observation_set(observation_set)
        initial_fit_set = self._observation_sets_by_hash.get(
            initial_calibration.fit_observation_set_hash
        )
        if (
            initial_fit_set is None
            or initial_fit_set.role is not EvidenceRole.CALIBRATION_FIT
        ):
            raise ValueError(
                "campaign must own the initial calibration-fit observation set"
            )
        if calibration_sampling_design is not None:
            if (
                initial_fit_set.sampling_design_hash
                != calibration_sampling_design.content_hash
            ):
                raise ValueError(
                    "campaign calibration support design differs from the initial "
                    "fit observation set"
                )
            initial_fit_set.validate_against(calibration_sampling_design)

        self._current_state_hash = self.initial_state_hash
        self._current_calibration_hash = self.initial_calibration_hash
        self._distortion_spent = 0.0
        self._registrations: dict[str, PredictableProposalRegistration] = {}
        self._status: dict[str, ProposalOutcome] = {}
        self._look_counts: dict[str, int] = {}
        self._receipts: list[DecisionEvidenceReceipt] = []

    @property
    def observation_sets(self) -> tuple[ObservationSetRef, ...]:
        return tuple(self._observation_sets_by_id.values())

    @property
    def calibration_sampling_design(self) -> SamplingDesignSpec | None:
        """Campaign-birth replay object for adaptive calibration support."""

        return self._calibration_sampling_design

    def _validate_observation_set_registration(
        self,
        observation_set: ObservationSetRef,
        *,
        allow_nonclaim_guidance_reuse: bool,
    ) -> tuple[str, ...]:
        if observation_set.observation_set_id in self._observation_sets_by_id:
            raise ValueError("campaign observation-set ID is already registered")
        if observation_set.content_hash in self._observation_sets_by_hash:
            raise ValueError("campaign observation stream is already registered")
        terminal_support_overlap = self._terminal_support_context_ids.intersection(
            observation_set.context_ids
        )
        if (
            terminal_support_overlap
            and observation_set.role is not EvidenceRole.TERMINAL_AUDIT
        ):
            raise ValueError(
                "campaign observation rows overlap the birth-frozen terminal "
                "support: " + ",".join(sorted(terminal_support_overlap))
            )
        existing = tuple(self._observation_sets_by_id.values())
        if not allow_nonclaim_guidance_reuse:
            for prior in existing:
                shared = observation_set.overlap(prior)
                if not shared:
                    continue
                roles = {observation_set.role, prior.role}
                if EvidenceRole.TERMINAL_AUDIT in roles:
                    raise ValueError(
                        "terminal audit contexts overlap an earlier statistical role: "
                        + ",".join(sorted(shared))
                    )
                if (
                    EvidenceRole.PROPOSAL_ADJUDICATION in roles
                    and observation_set.role is not prior.role
                ):
                    raise ValueError(
                        "proposal-adjudication contexts overlap a different "
                        "statistical role: " + ",".join(sorted(shared))
                    )
                if observation_set.role is not prior.role:
                    raise ValueError(
                        "contexts overlap different evidence roles without an "
                        "explicit nonclaim-guidance reuse registration: "
                        + ",".join(sorted(shared))
                    )
            return ()
        if observation_set.role is not EvidenceRole.GUIDANCE_GEOMETRY:
            raise ValueError(
                "only an explicitly nonclaim guidance set may reuse prior contexts"
            )
        overlaps: list[str] = []
        for prior in existing:
            shared = observation_set.overlap(prior)
            if not shared:
                continue
            if prior.role is EvidenceRole.TERMINAL_AUDIT:
                raise ValueError(
                    "nonclaim guidance cannot reopen terminal audit contexts"
                )
            if prior.role is EvidenceRole.PROPOSAL_ADJUDICATION:
                raise ValueError(
                    "nonclaim guidance cannot reuse proposal-adjudication contexts"
                )
            overlaps.extend(shared)
        return tuple(sorted(set(overlaps)))

    def register_observation_set(
        self,
        observation_set: ObservationSetRef,
        *,
        allow_nonclaim_guidance_reuse: bool = False,
    ) -> ObservationSetRef:
        overlaps = self._validate_observation_set_registration(
            observation_set,
            allow_nonclaim_guidance_reuse=allow_nonclaim_guidance_reuse,
        )
        self._observation_sets_by_id[observation_set.observation_set_id] = (
            observation_set
        )
        self._observation_sets_by_hash[observation_set.content_hash] = observation_set
        if overlaps:
            self._nonclaim_guidance_reuse.append(
                {
                    "observation_set_hash": observation_set.content_hash,
                    "overlap_context_ids": overlaps,
                    "claim_eligible": False,
                }
            )
        return observation_set

    def _require_owned_calibration_fit(
        self, calibration: CalibrationFitState
    ) -> ObservationSetRef:
        calibration.require_claim_bearing()
        observation_set = self._observation_sets_by_hash.get(
            calibration.fit_observation_set_hash
        )
        if (
            observation_set is None
            or observation_set.role is not EvidenceRole.CALIBRATION_FIT
        ):
            raise ValueError(
                "claim-bearing calibration fit references an unowned calibration set"
            )
        return observation_set

    def register_terminal_lock(self, lock: ValidationLock) -> ValidationLock:
        precommitment = self._terminal_precommitment
        if precommitment is None:
            raise RuntimeError(
                "sequential-only campaign has no birth-frozen terminal audit"
            )
        if self._terminal_lock is not None:
            raise RuntimeError("campaign terminal validation is already locked")
        if any(status == "registered" for status in self._status.values()):
            raise RuntimeError(
                "campaign cannot lock terminal validation while proposals remain live"
            )
        if lock.audit_opened:
            raise ValueError(
                "campaign must register terminal lock before audit opening"
            )
        if lock.final_state_hash != self.current_state_hash:
            raise ValueError(
                "terminal lock state differs from the committed campaign state"
            )
        if lock.final_calibration_hash != self.current_calibration_hash:
            raise ValueError(
                "terminal lock calibration differs from the committed campaign fit"
            )
        if not math.isclose(lock.terminal_alpha, self.terminal_alpha, abs_tol=1e-15):
            raise ValueError("terminal lock alpha differs from the campaign allocation")
        if (
            lock.terminal_precommitment_hash != precommitment.content_hash
            or lock.terminal_population_hash
            != precommitment.estimand.terminal_population_hash
            or lock.prediction_rule_hash != precommitment.estimand.prediction_rule_hash
            or not math.isclose(
                lock.parity_tolerance,
                precommitment.estimand.parity_tolerance,
                abs_tol=1e-15,
            )
        ):
            raise ValueError(
                "terminal lock differs from the campaign-birth precommitment"
            )
        counts: dict[str, int] = {}
        for stratum_id in lock.audit_observation_set.stratum_ids:
            counts[stratum_id] = counts.get(stratum_id, 0) + 1
        if counts != dict(precommitment.target_sample_sizes_by_stratum):
            raise ValueError(
                "terminal lock sample counts differ from the birth-frozen targets"
            )
        if precommitment.design_kind != "iid_fixed_n" and not set(
            lock.audit_observation_set.context_ids
        ).issubset(self._terminal_support_context_ids):
            raise ValueError(
                "finite terminal lock contains a context outside its frozen support"
            )
        self.register_observation_set(lock.audit_observation_set)
        self._terminal_lock = lock
        self._terminal_lock_registered_hash = lock.content_hash
        return lock

    @property
    def terminal_lock(self) -> ValidationLock | None:
        return self._terminal_lock

    @property
    def terminal_precommitment(self) -> TerminalAuditPrecommitment | None:
        return self._terminal_precommitment

    @property
    def terminal_certificate(self) -> TerminalParityCertificate | None:
        return self._terminal_certificate

    @property
    def terminal_audit_evidence_receipt(
        self,
    ) -> TerminalAuditEvidenceReceipt | None:
        return self._terminal_audit_evidence_receipt

    def execute_terminal_audit(
        self,
        evaluator: Callable[
            [ValidationLock, str],
            TerminalAuditEvidenceReceipt,
        ],
    ) -> TerminalAuditEvidenceReceipt:
        """Open the untouched audit exactly once, before any evaluator read.

        The opened state is retained even when execution or validation fails;
        a failed exposure cannot be retried as though it had remained hidden.
        """

        lock = self._terminal_lock
        registered_hash = self._terminal_lock_registered_hash
        if lock is None or registered_hash is None:
            raise RuntimeError("campaign terminal validation is not locked")
        if (
            lock.audit_opened
            or self._terminal_certificate is not None
            or self._terminal_audit_failure is not None
        ):
            raise RuntimeError("campaign terminal audit may be executed only once")
        if lock.final_state_hash != self.current_state_hash:
            raise RuntimeError("campaign state changed after terminal locking")
        if lock.final_calibration_hash != self.current_calibration_hash:
            raise RuntimeError("campaign calibration changed after terminal locking")
        if lock.prediction_rule_hash != self.risk_estimand.prediction_rule.content_hash:
            raise RuntimeError(
                "terminal lock prediction rule differs from the campaign risk"
            )
        opened = lock.open_audit()
        # Mutation precedes the callback: the audit is consumed even if the
        # evaluator raises after reading one row.
        self._terminal_lock = opened
        try:
            evidence_receipt = evaluator(opened, registered_hash)
            if not isinstance(
                evidence_receipt,
                TerminalAuditEvidenceReceipt,
            ):
                raise TypeError(
                    "terminal evaluator did not return typed physical audit evidence"
                )
            precommitment = self._terminal_precommitment
            assert precommitment is not None
            if (
                evidence_receipt.terminal_precommitment_hash
                != precommitment.content_hash
                or evidence_receipt.policy_hash != precommitment.policy_hash
            ):
                raise ValueError(
                    "terminal audit evidence differs from its birth precommitment"
                )
            certificate = evidence_receipt.certificate
            if certificate.validation_lock_hash != registered_hash:
                raise ValueError(
                    "terminal certificate differs from the registered lock"
                )
            if certificate.final_state_hash != self.current_state_hash:
                raise ValueError(
                    "terminal certificate differs from the final campaign state"
                )
            if (
                certificate.audit_observation_set_hash
                != lock.audit_observation_set.content_hash
            ):
                raise ValueError(
                    "terminal certificate differs from the locked audit rows"
                )
            if certificate.sample_size != len(lock.audit_observation_set.context_ids):
                raise ValueError(
                    "terminal certificate does not cover the locked audit sample"
                )
            if not math.isclose(
                certificate.tolerance,
                lock.parity_tolerance,
                abs_tol=1e-15,
            ):
                raise ValueError("terminal certificate tolerance differs from the lock")
            if (
                evidence_receipt.sampling_design_hash
                != lock.audit_observation_set.sampling_design_hash
            ):
                raise ValueError(
                    "terminal audit evidence differs from the locked sampling design"
                )
            expected_by_stratum: dict[str, list[str]] = {}
            for context_id, stratum_id in zip(
                lock.audit_observation_set.context_ids,
                lock.audit_observation_set.stratum_ids,
                strict=True,
            ):
                expected_by_stratum.setdefault(stratum_id, []).append(context_id)
            expected_axis = tuple(
                (stratum_id, tuple(context_ids))
                for stratum_id, context_ids in expected_by_stratum.items()
            )
            if evidence_receipt.observed_context_ids_by_stratum != expected_axis:
                raise ValueError(
                    "terminal audit evidence does not enumerate the exact locked "
                    "context/stratum axis"
                )
            if (
                evidence_receipt.resource_ledger_hash
                != self.resource_ledger.receipt()["ledger_hash"]
            ):
                raise ValueError(
                    "terminal audit evidence is detached from the campaign resource "
                    "ledger"
                )
        except BaseException as error:
            self._terminal_audit_failure = {
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
            raise
        self._terminal_certificate = certificate
        self._terminal_audit_evidence_receipt = evidence_receipt
        return evidence_receipt

    @property
    def proposal_alpha(self) -> float:
        return self.sequential_alpha / self.maximum_proposals

    @property
    def reserved_alpha(self) -> float:
        return len(self._registrations) * self.proposal_alpha

    @property
    def remaining_alpha(self) -> float:
        return max(0.0, self.sequential_alpha - self.reserved_alpha)

    @property
    def distortion_spent(self) -> float:
        return self._distortion_spent

    @property
    def distortion_remaining(self) -> float:
        return max(0.0, self.distortion_budget - self._distortion_spent)

    @property
    def current_state_hash(self) -> str:
        return self._current_state_hash

    @property
    def current_calibration_hash(self) -> str:
        return self._current_calibration_hash

    @property
    def receipts(self) -> tuple[DecisionEvidenceReceipt, ...]:
        return tuple(self._receipts)

    def proposal_status(self, proposal_id: str) -> ProposalOutcome:
        return self._status[str(proposal_id)]

    def register_proposal(
        self,
        *,
        proposal_id: str,
        query_id: str,
        group_id: str,
        parent_state_hash: str,
        candidate_state_hash: str,
        parent_calibration: CalibrationFitState,
        candidate_calibration: CalibrationFitState,
        sampling_design: SamplingDesignSpec,
        adjudication_observation_set: ObservationSetRef,
        local_upper_boundary: float,
        allocation_rule_id: str,
        allocation_rule_hash: str,
        candidate_definition_hash: str,
    ) -> PredictableProposalRegistration:
        identity = str(proposal_id)
        if self._terminal_lock is not None:
            raise RuntimeError(
                "campaign is terminally locked; no further proposals are legal"
            )
        if identity in self._registrations:
            raise ValueError(f"proposal {identity!r} is already registered")
        if len(self._registrations) >= self.maximum_proposals:
            raise RuntimeError(
                "claim proposal horizon exhausted; freeze a new campaign design "
                "before registering another proposal"
            )
        if str(parent_state_hash) != self._current_state_hash:
            raise ValueError("proposal parent is not the campaign's committed state")
        if str(query_id) != self.query_id:
            raise ValueError(
                "proposal query scope differs from the campaign's frozen risk"
            )
        self._require_owned_calibration_fit(parent_calibration)
        self._require_owned_calibration_fit(candidate_calibration)
        if parent_calibration.content_hash != self._current_calibration_hash:
            raise ValueError("proposal parent calibration is not the committed fit")
        if adjudication_observation_set.role is not EvidenceRole.PROPOSAL_ADJUDICATION:
            raise ValueError(
                "proposal registration requires adjudication-role observations"
            )
        if not sampling_design.claim_bearing:
            raise ValueError(
                "campaign proposals require a claim-bearing sampling design"
            )
        if (
            sampling_design.design_kind
            != "sealed_uniform_without_replacement_stratified"
        ):
            raise ValueError(
                "sequential campaign proposals require sealed finite-population "
                "sampling without replacement; fixed-n IID is terminal-only"
            )
        if (
            sampling_design.context_population_hash
            != self.risk_estimand.context_population_hash
        ):
            raise ValueError(
                "proposal sampling population differs from the risk estimand"
            )
        if any(
            prior.randomization_commitment_hash
            == sampling_design.randomization_commitment_hash
            for prior in self._sampling_designs.values()
        ):
            raise ValueError(
                "each adaptive proposal requires a fresh sealed randomization "
                "commitment drawn after its candidate is frozen"
            )
        permutation_birth_bindings = {
            item.birth_binding for item in sampling_design.permutation_receipts
        }
        if len(permutation_birth_bindings) != 1:
            raise ValueError(
                "proposal sampling receipts lack one frozen candidate-birth binding"
            )
        birth_binding = dict(next(iter(permutation_birth_bindings)))
        if (
            birth_binding.get("role") != EvidenceRole.PROPOSAL_ADJUDICATION.value
            or birth_binding.get("proposal_id") != identity
            or birth_binding.get("candidate_definition_hash")
            != str(candidate_definition_hash)
        ):
            raise ValueError(
                "proposal sampling entropy was not bound to this frozen candidate "
                "before adjudication"
            )
        query_weights = dict(self.risk_estimand.query_weights)
        strata_by_query = {
            frozen_query_id: dict(rows)
            for frozen_query_id, rows in self.risk_estimand.strata_by_query
        }
        classes_by_query = {
            frozen_query_id: dict(rows)
            for frozen_query_id, rows in (
                self.risk_estimand.binary_class_by_stratum_by_query
            )
        }
        component_rows = tuple(sampling_design.risk_component_by_stratum)
        if len(query_weights) == 1 and not component_rows:
            sole_query_id = next(iter(query_weights))
            expected_strata = dict(strata_by_query[sole_query_id])
            expected_binary_mapping = dict(classes_by_query[sole_query_id])
        else:
            if not component_rows:
                raise ValueError(
                    "joint-query risk requires an explicit query-by-stratum "
                    "sampling-component mapping"
                )
            component_by_sampling = {
                sampling_stratum_id: (component_query_id, risk_stratum_id)
                for sampling_stratum_id, component_query_id, risk_stratum_id in (
                    component_rows
                )
            }
            expected_components = {
                (frozen_query_id, risk_stratum_id)
                for frozen_query_id, rows in strata_by_query.items()
                for risk_stratum_id in rows
            }
            if set(component_by_sampling.values()) != expected_components:
                raise ValueError(
                    "proposal sampling components do not cover the complete "
                    "frozen query-by-stratum risk"
                )
            expected_strata = {
                sampling_stratum_id: (
                    float(query_weights[component_query_id])
                    * float(strata_by_query[component_query_id][risk_stratum_id])
                )
                for sampling_stratum_id, (
                    component_query_id,
                    risk_stratum_id,
                ) in component_by_sampling.items()
            }
            expected_binary_mapping = (
                {
                    sampling_stratum_id: int(
                        classes_by_query[component_query_id][risk_stratum_id]
                    )
                    for sampling_stratum_id, (
                        component_query_id,
                        risk_stratum_id,
                    ) in component_by_sampling.items()
                }
                if classes_by_query
                else {}
            )
        observed_strata = {
            item.stratum_id: item.risk_weight for item in sampling_design.strata
        }
        if set(expected_strata) != set(observed_strata) or any(
            not math.isclose(
                float(expected_strata[stratum_id]),
                float(observed_strata[stratum_id]),
                abs_tol=1e-12,
            )
            for stratum_id in expected_strata
        ):
            raise ValueError(
                "proposal sampling stratum weights differ from the frozen risk estimand"
            )
        observed_binary_mapping = dict(sampling_design.binary_class_by_stratum)
        if expected_binary_mapping != observed_binary_mapping:
            raise ValueError(
                "proposal sampling stratum-to-class mapping differs from the "
                "frozen risk estimand"
            )
        adjudication_observation_set.validate_against(sampling_design)
        expected_membership = {
            context_id: item.stratum_id
            for item in sampling_design.strata
            for context_id in item.context_ids_in_sealed_order
        }
        observed_membership = dict(
            zip(
                adjudication_observation_set.context_ids,
                adjudication_observation_set.stratum_ids,
                strict=True,
            )
        )
        if observed_membership != expected_membership:
            raise ValueError(
                "proposal adjudication set must enumerate the complete sealed population"
            )
        self._validate_observation_set_registration(
            adjudication_observation_set,
            allow_nonclaim_guidance_reuse=False,
        )
        proposal_alpha = self.proposal_alpha
        alpha_rows: list[tuple[str, float]] = []
        for index, item in enumerate(sampling_design.strata):
            allocated = (
                proposal_alpha - math.fsum(value for _, value in alpha_rows)
                if index == len(sampling_design.strata) - 1
                else proposal_alpha * item.risk_weight
            )
            alpha_rows.append((item.stratum_id, allocated))
        stratum_alpha = tuple(alpha_rows)
        registration = PredictableProposalRegistration(
            proposal_id=identity,
            query_id=str(query_id),
            group_id=str(group_id),
            registration_index=len(self._registrations) + 1,
            parent_state_hash=str(parent_state_hash),
            candidate_state_hash=str(candidate_state_hash),
            parent_calibration_hash=parent_calibration.content_hash,
            candidate_calibration_hash=candidate_calibration.content_hash,
            risk_estimand_hash=self.risk_estimand.content_hash,
            sampling_design_hash=sampling_design.content_hash,
            adjudication_observation_set_hash=adjudication_observation_set.content_hash,
            local_upper_boundary=float(local_upper_boundary),
            alpha_reserved=proposal_alpha,
            stratum_alpha=stratum_alpha,
            allocation_rule_id=str(allocation_rule_id),
            allocation_rule_hash=str(allocation_rule_hash),
            candidate_definition_hash=str(candidate_definition_hash),
        )
        self.register_observation_set(adjudication_observation_set)
        self._registrations[identity] = registration
        self._status[identity] = "registered"
        self._look_counts[identity] = 0
        self._sampling_designs[identity] = sampling_design
        self._adjudication_sets[identity] = adjudication_observation_set
        self._last_draw_history[identity] = ()
        self._allocation_receipts[identity] = []
        return registration

    @staticmethod
    def _prefix_hash(
        design: SamplingDesignSpec,
        draw_history: tuple[tuple[str, str], ...],
    ) -> str:
        return content_hash(
            {
                "sampling_design_hash": design.content_hash,
                "observed_context_ids_by_stratum": tuple(
                    (
                        stratum.stratum_id,
                        tuple(
                            context_id
                            for observed_stratum, context_id in draw_history
                            if observed_stratum == stratum.stratum_id
                        ),
                    )
                    for stratum in sorted(
                        design.strata,
                        key=lambda item: item.stratum_id,
                    )
                ),
                "draw_history": draw_history,
            }
        )

    def register_allocation_look(
        self,
        proposal_id: str,
        receipt: PredictableAllocationLookReceipt,
    ) -> PredictableAllocationLookReceipt:
        """Commit exact next targets before the corresponding outcomes exist."""

        identity = str(proposal_id)
        registration = self._registrations[identity]
        if self._status[identity] != "registered":
            raise RuntimeError(
                "only a live proposal may register another allocation look"
            )
        if identity in self._pending_allocation_receipts:
            raise RuntimeError("proposal already has an unconsumed allocation look")
        design = self._sampling_designs[identity]
        history = self._last_draw_history[identity]
        prior_receipts = self._allocation_receipts[identity]
        if (
            receipt.proposal_id != identity
            or receipt.proposal_registration_hash != registration.content_hash
            or receipt.look_index != self._look_counts[identity] + 1
            or receipt.sampling_design_hash != design.content_hash
            or receipt.allocation_rule_hash != registration.allocation_rule_hash
            or receipt.previous_allocation_receipt_hash
            != (prior_receipts[-1].content_hash if prior_receipts else None)
            or receipt.previous_context_prefix_hash
            != self._prefix_hash(design, history)
        ):
            raise ValueError(
                "allocation look does not extend the frozen proposal lineage"
            )
        targets = dict(receipt.cumulative_targets)
        expected_strata = {item.stratum_id for item in design.strata}
        if set(targets) != expected_strata:
            raise ValueError(
                "allocation look target axis differs from the sampling design"
            )
        previous_counts = {
            stratum_id: sum(
                observed_stratum == stratum_id for observed_stratum, _ in history
            )
            for stratum_id in expected_strata
        }
        growth = {
            stratum_id
            for stratum_id in expected_strata
            if targets[stratum_id] > previous_counts[stratum_id]
        }
        if not growth or any(
            targets[item.stratum_id] < previous_counts[item.stratum_id]
            or targets[item.stratum_id] > item.population_size
            for item in design.strata
        ):
            raise ValueError(
                "allocation look must be a legal growing finite-population prefix"
            )
        if not history:
            if (
                receipt.selected_stratum != "__initial_all_strata__"
                or growth != expected_strata
            ):
                raise ValueError(
                    "the initial packet must allocate every nonempty stratum"
                )
        elif receipt.selected_stratum == "__initial_all_strata__":
            raise ValueError(
                "all-strata allocation is legal only for the initial packet"
            )
        elif receipt.selected_stratum not in expected_strata or growth != {
            receipt.selected_stratum
        }:
            raise ValueError(
                "allocation selected stratum differs from the growing target"
            )
        prior_contexts = {context_id for _, context_id in history}
        expected_new_contexts = tuple(
            context_id
            for item in design.strata
            for context_id in item.context_ids_in_sealed_order[
                : targets[item.stratum_id]
            ]
            if context_id not in prior_contexts
        )
        if receipt.planned_new_context_ids != expected_new_contexts:
            raise ValueError(
                "allocation look contexts differ from the exact sealed prefixes"
            )
        self._pending_allocation_receipts[identity] = receipt
        return receipt

    def _validate_evidence_binding(
        self,
        proposal_id: str,
        evidence: StratifiedCSSnapshot,
        *,
        panel_prefix_hashes: tuple[str, ...],
        allocation_receipt: PredictableAllocationLookReceipt,
    ) -> None:
        registration = self._registrations[proposal_id]
        design = self._sampling_designs[proposal_id]
        observation_set = self._adjudication_sets[proposal_id]
        for panel_hash in panel_prefix_hashes:
            _require_sha256(panel_hash, "panel prefix hash")
        expected_panel_prefixes = tuple(
            item.planned_panel_hash
            for item in (
                *self._allocation_receipts[proposal_id],
                allocation_receipt,
            )
        )
        if panel_prefix_hashes != expected_panel_prefixes:
            raise ValueError(
                "delivered panel-prefix chain differs from its pre-outcome "
                "allocation receipts"
            )
        if not evidence.claim_bearing:
            raise ValueError("campaign adjudication refuses nonclaim evidence")
        if evidence.sample_size <= 0:
            raise ValueError(
                "campaign adjudication requires a nonempty evidence prefix"
            )
        if evidence.method != STRATIFIED_WOR_CS_METHOD:
            raise ValueError("evidence method differs from the registered v3 method")
        if evidence.sampling_design_hash != registration.sampling_design_hash:
            raise ValueError(
                "evidence sampling design differs from proposal registration"
            )
        if evidence.sampling_design_hash != design.content_hash:
            raise ValueError(
                "evidence sampling design differs from campaign-owned design"
            )
        if not math.isclose(evidence.alpha, registration.alpha_reserved, abs_tol=1e-15):
            raise ValueError("evidence alpha differs from the proposal reservation")
        registered_alpha = tuple(sorted(registration.stratum_alpha))
        if tuple(stratum_id for stratum_id, _ in evidence.stratum_alpha) != tuple(
            stratum_id for stratum_id, _ in registered_alpha
        ) or any(
            not math.isclose(observed, expected, abs_tol=1e-15)
            for (_, observed), (_, expected) in zip(
                evidence.stratum_alpha, registered_alpha, strict=True
            )
        ):
            raise ValueError(
                "evidence stratum alpha differs from the proposal birth allocation"
            )
        design_strata = {item.stratum_id: item for item in design.strata}
        observation_membership = dict(
            zip(
                observation_set.context_ids,
                observation_set.stratum_ids,
                strict=True,
            )
        )
        if tuple(item.stratum_id for item in evidence.strata) != tuple(
            sorted(design_strata)
        ):
            raise ValueError("evidence stratum axis differs from the registered design")
        for snapshot in evidence.strata:
            stratum = design_strata[snapshot.stratum_id]
            if snapshot.population_size != stratum.population_size:
                raise ValueError(
                    "evidence population size differs from the registered design"
                )
            if snapshot.sealed_order_hash != stratum.sealed_order_hash:
                raise ValueError(
                    "evidence sealed order differs from the registered design"
                )
            expected_prefix = stratum.context_ids_in_sealed_order[
                : snapshot.sample_size
            ]
            if snapshot.observed_context_ids != expected_prefix:
                raise ValueError(
                    "evidence contexts are not the registered sealed prefix"
                )
            if any(
                observation_membership.get(context_id) != snapshot.stratum_id
                for context_id in snapshot.observed_context_ids
            ):
                raise ValueError(
                    "evidence context prefix differs from the adjudication set"
                )
        previous_history = self._last_draw_history[proposal_id]
        if evidence.draw_history[: len(previous_history)] != previous_history:
            raise ValueError("evidence look does not extend the previous sealed prefix")
        if previous_history and len(evidence.draw_history) <= len(previous_history):
            raise ValueError("evidence look replayed a previously consumed prefix")
        if dict(allocation_receipt.cumulative_targets) != {
            item.stratum_id: item.sample_size for item in evidence.strata
        }:
            raise ValueError(
                "evidence stratum counts differ from the pre-outcome targets"
            )
        new_history = evidence.draw_history[len(previous_history) :]
        if tuple(context_id for _, context_id in new_history) != (
            allocation_receipt.planned_new_context_ids
        ):
            raise ValueError(
                "evidence draw extension differs from the pre-outcome contexts"
            )
        raw_values = {
            stratum_id: values
            for stratum_id, values in evidence.observed_values_by_stratum
        }
        consumed = {stratum_id: 0 for stratum_id in raw_values}
        recomputed_process = StratifiedPairedEProcess(
            sampling_design=design,
            alpha=registration.alpha_reserved,
            stratum_alpha=dict(registration.stratum_alpha),
            claim_bearing=True,
        )
        for stratum_id, context_id in evidence.draw_history:
            try:
                index = consumed[stratum_id]
                value = raw_values[stratum_id][index]
            except (KeyError, IndexError) as error:
                raise ValueError(
                    "evidence draw history has no corresponding raw observation"
                ) from error
            recomputed_process.record(
                stratum_id,
                value,
                context_id=context_id,
            )
            consumed[stratum_id] = index + 1
        if any(
            consumed[stratum_id] != len(values)
            for stratum_id, values in raw_values.items()
        ):
            raise ValueError(
                "evidence raw observations are not exhausted by its draw history"
            )
        recomputed = recomputed_process.interval()
        if recomputed.content_hash != evidence.content_hash:
            raise ValueError(
                "evidence confidence bounds were not deterministically recomputed "
                "from the retained bounded observations"
            )

    def adjudicate(
        self,
        proposal_id: str,
        evidence: StratifiedCSSnapshot,
        *,
        panel_prefix_hashes: tuple[str, ...] = (),
    ) -> DecisionEvidenceReceipt:
        identity = str(proposal_id)
        registration = self._registrations[identity]
        if self._status[identity] != "registered":
            raise RuntimeError(
                "only a live registered proposal may receive another look"
            )
        allocation_receipt = self._pending_allocation_receipts.get(identity)
        if allocation_receipt is None:
            raise RuntimeError(
                "campaign adjudication requires a pre-outcome allocation look"
            )
        normalized_panel_hashes = tuple(str(value) for value in panel_prefix_hashes)
        self._validate_evidence_binding(
            identity,
            evidence,
            panel_prefix_hashes=normalized_panel_hashes,
            allocation_receipt=allocation_receipt,
        )
        before = self._distortion_spent
        # The design weights already project every frozen query-by-stratum
        # component onto the complete global-risk estimand.
        query_weight = 1.0
        lower = float(evidence.lower)
        upper = float(evidence.upper)
        debit = max(upper, 0.0)
        if (
            not evidence.empty
            and upper <= registration.local_upper_boundary
            and before + debit <= self.distortion_budget + 1e-12
        ):
            outcome: ProposalOutcome = "accepted"
            after = before + debit
        elif not evidence.empty and lower > registration.local_upper_boundary:
            outcome = "rejected"
            debit = 0.0
            after = before
        else:
            outcome = "unresolved"
            debit = 0.0
            after = before
        next_look_index = self._look_counts[identity] + 1
        receipt = DecisionEvidenceReceipt(
            proposal_registration_hash=registration.content_hash,
            proposal_id=identity,
            look_index=next_look_index,
            evidence_snapshot_hash=evidence.content_hash,
            allocation_look_receipt_hash=allocation_receipt.content_hash,
            sampling_design_hash=registration.sampling_design_hash,
            adjudication_observation_set_hash=(
                registration.adjudication_observation_set_hash
            ),
            context_prefix_hash=evidence.context_prefix_hash,
            stratum_alpha=registration.stratum_alpha,
            evidence_method=evidence.method,
            lower_bound=lower,
            upper_bound=upper,
            local_upper_boundary=registration.local_upper_boundary,
            risk_query_weight=query_weight,
            alpha_reserved=registration.alpha_reserved,
            outcome=outcome,
            distortion_debit=debit,
            distortion_spent_before=before,
            distortion_spent_after=after,
            resource_ledger_hash=str(self.resource_ledger.receipt()["ledger_hash"]),
            panel_prefix_hashes=normalized_panel_hashes,
        )
        # Commit only after every design, lineage, and receipt validation has
        # succeeded.  Foreign or malformed evidence cannot partially mutate a
        # campaign.
        self._look_counts[identity] = next_look_index
        self._distortion_spent = after
        self._last_draw_history[identity] = evidence.draw_history
        self._allocation_receipts[identity].append(allocation_receipt)
        self._pending_allocation_receipts.pop(identity)
        if outcome == "accepted":
            self._current_state_hash = registration.candidate_state_hash
            self._current_calibration_hash = registration.candidate_calibration_hash
            self._status[identity] = outcome
            # Other live branches were born from the now-retired parent.  Their
            # alpha remains reserved; they cannot be silently retargeted.
            for other_id, other_status in tuple(self._status.items()):
                if other_id != identity and other_status == "registered":
                    self._status[other_id] = "superseded"
        elif outcome == "rejected":
            self._status[identity] = outcome
        self._receipts.append(receipt)
        return receipt

    def retire_proposal(
        self,
        proposal_id: str,
        *,
        outcome: Literal["abandoned", "superseded"],
    ) -> None:
        identity = str(proposal_id)
        if self._status[identity] != "registered":
            raise RuntimeError("only a live proposal may be retired")
        if identity in self._pending_allocation_receipts:
            raise RuntimeError(
                "proposal with delivered-or-pending allocation evidence cannot "
                "be retired before adjudication"
            )
        self._status[identity] = outcome

    def receipt(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_name": "campaign_evidence_state_v3",
            "foundation": "evidence_foundation_glocal_v3",
            "campaign_id": self.campaign_id,
            "risk_estimand_hash": self.risk_estimand.content_hash,
            "query_scope_id": self.query_id,
            "query_ids": self.query_ids,
            "initial_state_hash": self.initial_state_hash,
            "initial_calibration_hash": self.initial_calibration_hash,
            "current_state_hash": self.current_state_hash,
            "current_calibration_hash": self.current_calibration_hash,
            "total_epsilon": self.total_epsilon,
            "sequential_alpha": self.sequential_alpha,
            "terminal_alpha": self.terminal_alpha,
            "terminal_precommitment": (
                None
                if self._terminal_precommitment is None
                else asdict(self._terminal_precommitment)
            ),
            "terminal_precommitment_hash": (
                None
                if self._terminal_precommitment is None
                else self._terminal_precommitment.content_hash
            ),
            "calibration_sampling_design": (
                None
                if self._calibration_sampling_design is None
                else asdict(self._calibration_sampling_design)
            ),
            "calibration_sampling_design_hash": (
                None
                if self._calibration_sampling_design is None
                else self._calibration_sampling_design.content_hash
            ),
            "maximum_proposals": self.maximum_proposals,
            "proposal_alpha": self.proposal_alpha,
            "reserved_alpha": self.reserved_alpha,
            "distortion_budget": self.distortion_budget,
            "distortion_spent": self.distortion_spent,
            "proposal_registrations": [
                {
                    "registration": asdict(registration),
                    "registration_hash": registration.content_hash,
                    "status": self._status[proposal_id],
                }
                for proposal_id, registration in self._registrations.items()
            ],
            "sampling_designs": [
                {
                    "proposal_id": proposal_id,
                    "sampling_design": asdict(self._sampling_designs[proposal_id]),
                    "sampling_design_hash": self._sampling_designs[
                        proposal_id
                    ].content_hash,
                }
                for proposal_id in self._registrations
            ],
            "decision_receipt_hashes": [
                receipt.content_hash for receipt in self._receipts
            ],
            "allocation_look_receipts": [
                {
                    "proposal_id": proposal_id,
                    "look_index": receipt.look_index,
                    "allocation_receipt_hash": receipt.content_hash,
                }
                for proposal_id in self._registrations
                for receipt in self._allocation_receipts[proposal_id]
            ],
            "pending_allocation_receipt_hashes": {
                proposal_id: receipt.content_hash
                for proposal_id, receipt in (self._pending_allocation_receipts.items())
            },
            "observation_sets": [
                {
                    "observation_set": asdict(observation_set),
                    "observation_set_hash": observation_set.content_hash,
                }
                for observation_set in self.observation_sets
            ],
            "nonclaim_guidance_reuse": list(self._nonclaim_guidance_reuse),
            "terminal_lock_hash": (self._terminal_lock_registered_hash),
            "terminal_opened_lock_hash": (
                self._terminal_lock.content_hash
                if self._terminal_lock is not None and self._terminal_lock.audit_opened
                else None
            ),
            "terminal_certificate_hash": (
                self._terminal_certificate.content_hash
                if self._terminal_certificate is not None
                else None
            ),
            "terminal_audit_evidence_receipt_hash": (
                self._terminal_audit_evidence_receipt.content_hash
                if self._terminal_audit_evidence_receipt is not None
                else None
            ),
            "terminal_audit_failure": self._terminal_audit_failure,
            "resource_ledger": self.resource_ledger.receipt(),
        }
        payload["campaign_hash"] = content_hash(payload)
        return payload


__all__ = [
    "CampaignEvidenceState",
    "DecisionEvidenceReceipt",
    "ProposalOutcome",
    "ResourceLedger",
    "ResourceUsage",
]
