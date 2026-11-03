"""Run and decision state for the common IRG pruning engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .candidates import HardMask
from .commit_authority import (
    CommitAuthorityMode,
    ExactHardMaskCommitAuthorization,
    ExactHardMaskCommitDecision,
)
from .contracts import HistoricalResponseProfileName, ScientificIdentity, content_hash
from .costs import ComputeLedger
from .frontier import FrontierLedger


@dataclass(frozen=True)
class StoppingCriteria:
    """Run limits.

    In the default v1+ ``paired_bce_total_increment`` boundary mode,
    ``maximum_quality_loss`` is the compatibility storage name for the
    maximum complete-mask BCE increment above the locked dense baseline. A
    relative policy (for example 20%) is compiled to this absolute increment
    from the measured baseline before candidate outcomes are inspected.
    """

    maximum_quality_loss: float
    minimum_released_cost: float = 0.0
    practical_indifference: float = 0.0
    maximum_oracle_evaluations: int = 32
    maximum_iterations: int | None = 128
    cost_bracket_tolerance: float = 0.0

    def __post_init__(self) -> None:
        import math

        for name in (
            "maximum_quality_loss",
            "minimum_released_cost",
            "practical_indifference",
            "cost_bracket_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or (
                name != "maximum_quality_loss" and value < 0.0
            ):
                raise ValueError(f"{name} has an invalid value")
        if int(self.maximum_oracle_evaluations) <= 0:
            raise ValueError("maximum_oracle_evaluations must be positive")
        if self.maximum_iterations is not None and int(self.maximum_iterations) <= 0:
            raise ValueError("maximum_iterations must be positive when set")

    @property
    def maximum_bce_increment(self) -> float:
        return float(self.maximum_quality_loss)


@dataclass
class DecisionState:
    active_candidate_ids: set[str] = field(default_factory=set)
    practical_top_set: set[str] = field(default_factory=set)
    active_contrasts: dict[tuple[str, str], Mapping[str, float]] = field(
        default_factory=dict
    )
    acquisition_queue: list[str] = field(default_factory=list)
    context_coverage: dict[str, int] = field(default_factory=dict)
    calibration_uncertainty: float = 0.0
    calibration_sample_count: int = 0
    model_residual_scale: float = 0.0
    pending_model_validity_action: str = "reuse"
    recommended_action_scale: float = 1.0
    requested_capability_transition: str | None = None
    geometry_anchor_mask_hash: str | None = None
    geometry_current_mask_hash: str | None = None
    cumulative_geometry_removed_indices: set[int] = field(default_factory=set)
    geometry_transition_receipt_hashes: list[str] = field(default_factory=list)
    predictive_validity_receipt_hashes: list[str] = field(default_factory=list)
    acquisition_receipt_hashes: list[str] = field(default_factory=list)
    scientific_fallback_receipt_hashes: list[str] = field(default_factory=list)
    transitions: list[dict[str, Any]] = field(default_factory=list)

    def record_transition(self, action: str, reason: str, **details: Any) -> str:
        if not action or not reason:
            raise ValueError("decision transition requires action and reason")
        row = {
            "sequence": len(self.transitions),
            "action": str(action),
            "reason": str(reason),
            "details": dict(details),
        }
        row["transition_hash"] = content_hash(row, prefix="irg-transition-")
        self.transitions.append(row)
        return str(row["transition_hash"])

    @property
    def geometry_transport_hash(self) -> str:
        return content_hash(
            {
                "geometry_anchor_mask_hash": self.geometry_anchor_mask_hash,
                "geometry_current_mask_hash": self.geometry_current_mask_hash,
                "cumulative_geometry_removed_indices": tuple(
                    sorted(
                        int(value) for value in self.cumulative_geometry_removed_indices
                    )
                ),
                "geometry_transition_receipt_hashes": tuple(
                    self.geometry_transition_receipt_hashes
                ),
            },
            prefix="irg-geometry-transport-",
        )

    def reset_geometry_transport(self, *, measured_mask_hash: str) -> None:
        if not measured_mask_hash:
            raise ValueError("geometry transport reset requires a measured mask hash")
        self.geometry_anchor_mask_hash = str(measured_mask_hash)
        self.geometry_current_mask_hash = str(measured_mask_hash)
        self.cumulative_geometry_removed_indices.clear()
        self.geometry_transition_receipt_hashes.clear()


@dataclass
class PruningSessionState:
    identity: ScientificIdentity
    current_mask: HardMask
    active_version: HistoricalResponseProfileName
    version_state: Any
    stopping: StoppingCriteria
    run_id: str
    random_seed: int = 0
    commit_authority_mode: CommitAuthorityMode = "exploratory"
    compute_ledger: ComputeLedger = field(default_factory=ComputeLedger)
    frontier: FrontierLedger | None = None
    decision: DecisionState = field(default_factory=DecisionState)
    cache: dict[str, Any] = field(default_factory=dict)
    committed_candidate_ids: list[str] = field(default_factory=list)
    iteration: int = 0

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("pruning session requires a run_id")
        if self.commit_authority_mode not in {
            "exploratory",
            "glocal_v3_shadow",
        }:
            raise ValueError("pruning session has an unknown commit-authority mode")
        if self.current_mask.route_axis_hash != self.identity.route_axis_hash:
            raise ValueError("current mask does not match the scientific route axis")
        if self.frontier is None:
            self.frontier = FrontierLedger(
                primary_cost_coordinate=self.identity.cost.primary_coordinate
            )
        measured_mask_hash = getattr(self.version_state, "measured_mask_hash", None)
        if (
            self.decision.geometry_anchor_mask_hash is None
            and measured_mask_hash == self.current_mask.mask_hash
        ):
            self.decision.reset_geometry_transport(
                measured_mask_hash=self.current_mask.mask_hash
            )

    @property
    def state_hash(self) -> str:
        version_state_hash = getattr(self.version_state, "state_hash", None)
        if callable(version_state_hash):
            version_state_hash = version_state_hash()
        if version_state_hash is None:
            version_state_hash = content_hash(
                repr(self.version_state), prefix="opaque-"
            )
        return content_hash(
            {
                "identity_hash": self.identity.identity_hash,
                "current_mask_hash": self.current_mask.mask_hash,
                "active_version": self.active_version,
                "commit_authority_mode": self.commit_authority_mode,
                "version_state_hash": version_state_hash,
                "geometry_transport_hash": self.decision.geometry_transport_hash,
                "committed_candidate_ids": tuple(self.committed_candidate_ids),
                "iteration": self.iteration,
            },
            prefix="irg-state-",
        )

    def assert_request_state(self, state_hash: str) -> None:
        if str(state_hash) != self.state_hash:
            raise ValueError(
                "measurement request/outcome belongs to a different pruning state"
            )

    def advance_iteration(self) -> None:
        self.iteration += 1
        if self.stopping.maximum_iterations is not None and self.iteration > int(
            self.stopping.maximum_iterations
        ):
            raise RuntimeError("pruning session exceeded maximum_iterations")

    def commit_candidate(
        self,
        candidate_id: str,
        *,
        authorization: ExactHardMaskCommitAuthorization | None = None,
    ) -> None:
        entry = self.frontier.entry(candidate_id)
        self.current_mask.assert_same_axis(entry.candidate.resulting_mask)
        if entry.candidate.parent_state_hash != self.state_hash:
            raise ValueError("candidate parent state is stale at the commit boundary")
        if entry.candidate.parent_mask_hash != self.current_mask.mask_hash:
            raise ValueError("candidate parent mask differs from the current hard mask")
        entry.candidate.resulting_mask.removed_index_set_from(self.current_mask)
        self.frontier.preflight_commit(candidate_id)
        if self.commit_authority_mode != "glocal_v3_shadow":
            raise ValueError(
                "exploratory sessions are nonmutating; an exact hard-mask "
                "commit requires a registered live commit controller"
            )
        if type(authorization) is not ExactHardMaskCommitAuthorization:
            raise ValueError(
                "shadow glocal-v3 commit requires a live exact-mask "
                "authorization capability; a detached decision receipt "
                "cannot authorize state mutation"
            )
        decision = authorization.consume(
            session=self,
            candidate=entry.candidate,
        )
        if not decision.accepted:
            raise ValueError(
                "shadow glocal-v3 commit requires an accepted exact-mask "
                "authority verdict"
            )
        if decision.authority_id != "evidence_foundation_glocal_v3":
            raise ValueError(
                "shadow commit authorization has the wrong foundation identity"
            )
        decision.validate_transition(
            candidate_id=entry.candidate.candidate_id,
            irg_parent_state_hash=entry.candidate.parent_state_hash,
            parent_mask_hash=self.current_mask.mask_hash,
            candidate_mask_hash=entry.candidate.resulting_mask.mask_hash,
        )
        authority_details = {
            "commit_authority_mode": self.commit_authority_mode,
            "authority_id": decision.authority_id,
            "authority_decision_hash": decision.content_hash,
        }
        self.frontier.commit(
            candidate_id,
            reason="authorized by bounded paired Brier evidence",
            **authority_details,
        )
        self.current_mask = entry.candidate.resulting_mask
        self.committed_candidate_ids.append(candidate_id)
        self.decision.record_transition(
            "commit",
            "bounded paired Brier evidence authorized the exact hard-mask transition",
            candidate_id=candidate_id,
            resulting_mask_hash=self.current_mask.mask_hash,
            **authority_details,
        )


__all__ = [
    "DecisionState",
    "PruningSessionState",
    "StoppingCriteria",
]
