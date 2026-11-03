"""Canonical statistical sampled-path frontier.

Each proposal is a paired experiment: committed and candidate masks are
calibrated and scored on the same nested panel, each cross-fit score is
evaluated on contexts excluded from its calibration fold, and route cost uses
the requested branch-density change times the complete layer route count.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from mechanistic_systems.contracts.sampled_evidence import (
    CrossfitPanel,
    SampledReferenceEvidence,
    sampled_path_digest as _digest,
    sampled_path_mask_hash,
)


SAMPLED_PATH_FRONTIER_V1 = "sampled_path_frontier_v1"
SAMPLED_PATH_EVIDENCE_SCHEMA = "mechanistic_sampled_path_paired_evidence_v1"


@dataclass(frozen=True)
class PairedCandidateEvidence:
    panel: CrossfitPanel
    committed_loss: np.ndarray
    candidate_loss: np.ndarray
    committed_calibration_by_fold: tuple[Mapping[str, Any], ...]
    candidate_calibration_by_fold: tuple[Mapping[str, Any], ...]
    committed_mask_hash: str
    candidate_mask_hash: str
    common_random_numbers: bool = True
    schema_name: str = SAMPLED_PATH_EVIDENCE_SCHEMA

    def validate(self, *, fold_count: int) -> None:
        self.panel.validate(fold_count=fold_count)
        committed = np.asarray(self.committed_loss, dtype=np.float64).reshape(-1)
        candidate = np.asarray(self.candidate_loss, dtype=np.float64).reshape(-1)
        if (
            committed.shape != candidate.shape
            or committed.size != len(self.panel.context_ids)
            or not np.isfinite(committed).all()
            or not np.isfinite(candidate).all()
        ):
            raise ValueError("paired losses must be finite on the complete panel")
        if not self.common_random_numbers:
            raise ValueError("sampled-path evidence requires common random numbers")
        if (
            len(self.committed_calibration_by_fold) != fold_count
            or len(self.candidate_calibration_by_fold) != fold_count
        ):
            raise ValueError("each held-out fold requires separately fitted calibration")
        for fold in range(fold_count):
            calibration = set(self.panel.calibration_indices(fold).tolist())
            extraction = set(self.panel.extraction_indices(fold).tolist())
            if not calibration or not extraction or calibration & extraction:
                raise ValueError("calibration and extraction sets must be disjoint per fold")
        if len(self.committed_mask_hash) != 64 or len(self.candidate_mask_hash) != 64:
            raise ValueError("paired evidence requires immutable candidate identities")


@dataclass(frozen=True)
class SampledComparisonRequest:
    query_id: str
    layer_id: str
    partition_id: str
    previous_density: float
    proposed_density: float
    stage_index: int
    stage_per_class: int
    proposal_index: int
    random_seed: int
    fold_count: int
    prior_panel: CrossfitPanel | None = None
    committed_warm_start: Mapping[str, Any] | None = None
    candidate_warm_start: Mapping[str, Any] | None = None
    allocation_evidence: Mapping[str, Any] = field(default_factory=dict)


class SampledPathEvaluationBackend(Protocol):
    backend_id: str

    def evaluate_dense_reference(
        self,
        masks: Mapping[str, np.ndarray],
        request: SampledComparisonRequest,
    ) -> SampledReferenceEvidence:
        ...

    def evaluate_pair(
        self,
        committed_masks: Mapping[str, np.ndarray],
        candidate_masks: Mapping[str, np.ndarray],
        request: SampledComparisonRequest,
    ) -> PairedCandidateEvidence:
        ...


@dataclass(frozen=True)
class SampledPathPolicy:
    sample_stages_per_class: tuple[int, ...]
    fold_count: int = 2
    posterior_accept_probability: float = 0.95
    posterior_reject_probability: float = 0.95
    variance_prior_strength: float = 4.0
    variance_prior_scale: float = 1e-6
    finite_population_correction: bool = True
    design_weighted: bool = True
    unresolved_final_action: str = "reject"
    random_seed: int = 0
    precision_grid_units: float | None = None

    def validate(self) -> None:
        stages = self.sample_stages_per_class
        if not stages or tuple(sorted(set(stages))) != stages or min(stages) <= 0:
            raise ValueError("sample stages must be positive, unique, and increasing")
        if self.fold_count < 2:
            raise ValueError("sampled paths require at least two cross-fit folds")
        for value in (self.posterior_accept_probability, self.posterior_reject_probability):
            if not 0.5 < value < 1.0:
                raise ValueError("posterior probabilities must lie in (0.5,1)")
        if self.variance_prior_strength <= 0.0:
            raise ValueError("variance prior strength must be positive")
        if not math.isfinite(self.variance_prior_scale) or self.variance_prior_scale <= 0.0:
            raise ValueError("variance prior scale must be finite and positive")
        if self.unresolved_final_action not in {"reject", "accept"}:
            raise ValueError("unresolved final action must be reject or accept")
        if self.precision_grid_units is not None and (
            not math.isfinite(float(self.precision_grid_units))
            or not 0.0 < float(self.precision_grid_units) <= 1.0
        ):
            raise ValueError("precision grid units must lie in (0,1]")


@dataclass(frozen=True)
class PairedPosteriorDecision:
    action: str
    posterior_mean_sensitivity: float
    posterior_sd_sensitivity: float
    accept_probability: float
    reject_probability: float
    normalized_loss_delta: float
    removed_route_cost: float
    context_count: int
    stage_index: int
    panel_hash: str
    reason: str
    posterior_accept_regret: float = 0.0
    posterior_reject_regret: float = 0.0
    posterior_regret_tolerance: float | None = None
    decision_rule: str = "posterior_probability_and_boundary_width"


def _weighted_mean_variance(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    prior_strength: float,
    prior_variance: float,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.shape != weights.shape or values.size < 2:
        raise ValueError("posterior evidence requires at least two observations")
    normalized = weights / weights.sum()
    mean = float(np.sum(normalized * values))
    raw_variance = float(np.sum(normalized * np.square(values - mean)))
    effective_n = float(np.square(weights.sum()) / np.square(weights).sum())
    degrees = max(effective_n - 1.0, 1.0)
    stabilized = (
        raw_variance * degrees + float(prior_variance) * float(prior_strength)
    ) / (degrees + float(prior_strength))
    standard_error = math.sqrt(max(stabilized, 1e-18) / max(effective_n, 1.0))
    return mean, standard_error, effective_n


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))


def _normal_pdf(value: float) -> float:
    return math.exp(-0.5 * float(value) * float(value)) / math.sqrt(2.0 * math.pi)


def _expected_upper_hinge(mean: float, sd: float, threshold: float) -> float:
    sd = max(0.0, float(sd))
    if sd <= 1e-12:
        return max(float(mean) - float(threshold), 0.0)
    z = (float(mean) - float(threshold)) / sd
    return float(sd * _normal_pdf(z) + (float(mean) - float(threshold)) * _normal_cdf(z))


def _expected_lower_hinge(mean: float, sd: float, threshold: float) -> float:
    sd = max(0.0, float(sd))
    if sd <= 1e-12:
        return max(float(threshold) - float(mean), 0.0)
    z = (float(mean) - float(threshold)) / sd
    return float(sd * _normal_pdf(z) + (float(threshold) - float(mean)) * _normal_cdf(-z))


def paired_sensitivity_decision(
    evidence: PairedCandidateEvidence,
    *,
    policy: SampledPathPolicy,
    dense_anchor_loss: float,
    fully_pruned_anchor_loss: float,
    requested_density_delta: float,
    total_layer_route_count: int,
    sensitivity_threshold: float,
    final_stage: bool,
) -> PairedPosteriorDecision:
    policy.validate()
    evidence.validate(fold_count=policy.fold_count)
    span = float(fully_pruned_anchor_loss) - float(dense_anchor_loss)
    denominator = float(requested_density_delta) * int(total_layer_route_count)
    removed = abs(denominator)
    if abs(span) <= 1e-12 or removed <= 0.0:
        raise ValueError("sensitivity requires nondegenerate anchors and route cost")
    raw_delta = np.asarray(evidence.candidate_loss) - np.asarray(evidence.committed_loss)
    allocation_rule = str(evidence.panel.allocation_metadata.get("allocation_rule", ""))
    if allocation_rule == "cost_optimal_two_stage_neyman":
        # Imported lazily to keep the panel/evidence module acyclic.
        from .neyman import summarize_hierarchical_paired_risk

        risk = summarize_hierarchical_paired_risk(evidence)
        normalized_delta = float(risk.posterior_mean_delta / span)
        mean = -float(risk.posterior_mean_delta) / float(span) / float(denominator)
        standard_error = float(risk.posterior_sd_delta) / max(
            abs(float(span) * float(denominator)), 1e-12
        )
    else:
        # Schedules are expressed as retained densities, so an accepted pruning
        # move has ``proposed - previous < 0``.  Historical frontier sensitivity
        # is damage per removed route and therefore negates that signed density
        # displacement.
        sensitivity = -raw_delta / (span * denominator)
        weights = (
            np.asarray(evidence.panel.sampling_weights, dtype=np.float64)
            if policy.design_weighted and evidence.panel.sampling_weights
            else np.ones(sensitivity.size, dtype=np.float64)
        )
        mean, standard_error, effective_n = _weighted_mean_variance(
            sensitivity,
            weights,
            prior_strength=policy.variance_prior_strength,
            prior_variance=policy.variance_prior_scale,
        )
        if policy.finite_population_correction and evidence.panel.sampling_weights:
            population_size = max(
                float(sensitivity.size), float(sum(evidence.panel.sampling_weights))
            )
            standard_error *= math.sqrt(max(0.0, 1.0 - effective_n / population_size))
        normalized_delta = float(np.average(raw_delta / span, weights=weights))
    threshold = float(sensitivity_threshold)
    if standard_error <= 1e-15:
        accept_probability = 1.0 if mean <= threshold else 0.0
    else:
        z = (threshold - mean) / standard_error
        accept_probability = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    reject_probability = 1.0 - accept_probability
    accept_regret = _expected_upper_hinge(mean, standard_error, threshold)
    reject_regret = _expected_lower_hinge(mean, standard_error, threshold)
    regret_tolerance: float | None = None
    decision_rule = "posterior_probability_and_boundary_width"
    if policy.precision_grid_units is not None:
        decision_rule = "bayes_expected_frontier_regret_cost_optimal_panel"
        regret_tolerance = float(policy.precision_grid_units) / (
            2.0 * 1.96 * math.sqrt(2.0 * math.pi)
        )
        if accept_regret <= regret_tolerance and accept_regret <= reject_regret:
            action, reason = "accept", "posterior_accept_regret_below_precision_budget"
        elif reject_regret <= regret_tolerance:
            action, reason = "reject", "posterior_reject_regret_below_precision_budget"
        elif final_stage:
            action, reason = (
                policy.unresolved_final_action,
                "posterior_regret_unresolved_at_final_stage",
            )
        else:
            action, reason = "sample_more", "posterior_expected_regret_above_precision_budget"
    elif accept_probability >= policy.posterior_accept_probability:
        action, reason = "accept", "posterior_below_threshold"
    elif reject_probability >= policy.posterior_reject_probability:
        action, reason = "reject", "posterior_above_threshold"
    elif final_stage:
        action, reason = policy.unresolved_final_action, "posterior_unresolved_at_final_stage"
    else:
        action, reason = "sample_more", "posterior_boundary_uncertain"
    return PairedPosteriorDecision(
        action=action,
        posterior_mean_sensitivity=mean,
        posterior_sd_sensitivity=standard_error,
        accept_probability=accept_probability,
        reject_probability=reject_probability,
        normalized_loss_delta=normalized_delta,
        removed_route_cost=removed,
        context_count=raw_delta.size,
        stage_index=evidence.panel.stage_index,
        panel_hash=evidence.panel.panel_hash,
        reason=reason,
        posterior_accept_regret=float(accept_regret),
        posterior_reject_regret=float(reject_regret),
        posterior_regret_tolerance=regret_tolerance,
        decision_rule=decision_rule,
    )


@dataclass(frozen=True)
class SampledPathStep:
    proposal_index: int
    partition_id: str
    previous_density: float
    proposed_density: float
    accepted: bool
    decision: PairedPosteriorDecision
    evidence_stage_hashes: tuple[str, ...]


@dataclass(frozen=True)
class SampledPathResult:
    positive_density: float
    negative_density: float
    committed_masks: Mapping[str, np.ndarray]
    steps: tuple[SampledPathStep, ...]
    stopping_reason: str
    algorithm_version: str = SAMPLED_PATH_FRONTIER_V1


def fold_warm_start(
    calibrations: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Carry calibration state without collapsing the held-out-fold axis."""

    if calibrations and all("fold" in item for item in calibrations):
        return {
            "fits_by_fold": {
                str(int(item["fold"])): dict(item)
                for item in calibrations
            }
        }
    return dict(calibrations[-1]) if calibrations else {}


def trace_statistical_symmetric_sampled_path(
    *,
    backend: SampledPathEvaluationBackend,
    committed_masks: Mapping[str, np.ndarray],
    build_candidate: Any,
    query_id: str,
    layer_id: str,
    positive_schedule: Sequence[float],
    negative_schedule: Sequence[float],
    total_layer_route_count: int,
    dense_anchor_loss: float,
    fully_pruned_anchor_loss: float,
    sensitivity_threshold: float,
    policy: SampledPathPolicy,
    initial_committed_warm_start: Mapping[str, Any] | None = None,
    initial_turn: str = "negative",
    maximum_proposals: int = 10000,
) -> SampledPathResult:
    """Run symmetric full turns with staged paired sampled-path decisions."""

    policy.validate()
    schedules = {
        "positive": tuple(float(value) for value in positive_schedule),
        "negative": tuple(float(value) for value in negative_schedule),
    }
    for partition_id, schedule in schedules.items():
        if not schedule or schedule[0] != 1.0 or any(
            left <= right for left, right in zip(schedule, schedule[1:])
        ):
            raise ValueError(f"{partition_id} schedule must begin at one and decrease")
    if initial_turn not in schedules:
        raise ValueError("initial turn must be positive or negative")
    current_masks = {key: np.asarray(value, dtype=bool).copy() for key, value in committed_masks.items()}
    cursors = {"positive": 0, "negative": 0}
    densities = {"positive": 1.0, "negative": 1.0}
    # Calibration state belongs to an immutable mask and held-out fold, not to
    # the transient committed/candidate role. An accepted candidate is the next
    # committed baseline and must retain the same foldwise warm start.
    initial_mask_hash = sampled_path_mask_hash(current_masks)
    warm_starts: dict[str, Mapping[str, Any]] = {}
    if initial_committed_warm_start:
        warm_starts[initial_mask_hash] = dict(initial_committed_warm_start)
    turn = initial_turn
    stalled_turns = 0
    proposal_index = 0
    steps: list[SampledPathStep] = []
    while proposal_index < maximum_proposals:
        schedule = schedules[turn]
        advanced = False
        if cursors[turn] + 1 < len(schedule):
            while cursors[turn] + 1 < len(schedule):
                proposed = schedule[cursors[turn] + 1]
                candidate_masks = build_candidate(
                    current_masks,
                    partition_id=turn,
                    positive_density=proposed if turn == "positive" else densities["positive"],
                    negative_density=proposed if turn == "negative" else densities["negative"],
                )
                prior_panel: CrossfitPanel | None = None
                allocation_evidence: Mapping[str, Any] = {}
                evidence_hashes: list[str] = []
                final_decision: PairedPosteriorDecision | None = None
                for stage_index, stage_per_class in enumerate(policy.sample_stages_per_class):
                    committed_hash = sampled_path_mask_hash(current_masks)
                    candidate_hash = sampled_path_mask_hash(candidate_masks)
                    request = SampledComparisonRequest(
                        query_id=query_id,
                        layer_id=layer_id,
                        partition_id=turn,
                        previous_density=densities[turn],
                        proposed_density=proposed,
                        stage_index=stage_index,
                        stage_per_class=stage_per_class,
                        proposal_index=proposal_index,
                        random_seed=policy.random_seed,
                        fold_count=policy.fold_count,
                        prior_panel=prior_panel,
                        committed_warm_start=warm_starts.get(committed_hash),
                        candidate_warm_start=warm_starts.get(candidate_hash),
                        allocation_evidence=allocation_evidence,
                    )
                    evidence = backend.evaluate_pair(current_masks, candidate_masks, request)
                    evidence.validate(fold_count=policy.fold_count)
                    if (
                        prior_panel is not None
                        and str(evidence.panel.allocation_metadata.get("allocation_rule", ""))
                        != "cost_optimal_two_stage_neyman"
                    ):
                        evidence.panel.assert_nested_extension(prior_panel)
                    if evidence.committed_mask_hash != committed_hash:
                        raise ValueError("backend committed-mask identity differs from request")
                    if evidence.candidate_mask_hash != candidate_hash:
                        raise ValueError("backend candidate-mask identity differs from request")
                    evidence_hashes.append(
                        _digest(
                            {
                                "panel": evidence.panel.panel_hash,
                                "committed": evidence.committed_mask_hash,
                                "candidate": evidence.candidate_mask_hash,
                                "committed_loss": np.asarray(evidence.committed_loss).tolist(),
                                "candidate_loss": np.asarray(evidence.candidate_loss).tolist(),
                            }
                        )
                    )
                    final_decision = paired_sensitivity_decision(
                        evidence,
                        policy=policy,
                        dense_anchor_loss=dense_anchor_loss,
                        fully_pruned_anchor_loss=fully_pruned_anchor_loss,
                        requested_density_delta=proposed - densities[turn],
                        total_layer_route_count=total_layer_route_count,
                        sensitivity_threshold=sensitivity_threshold,
                        final_stage=stage_index == len(policy.sample_stages_per_class) - 1,
                    )
                    warm_starts[committed_hash] = fold_warm_start(
                        evidence.committed_calibration_by_fold
                    )
                    warm_starts[candidate_hash] = fold_warm_start(
                        evidence.candidate_calibration_by_fold
                    )
                    if (
                        str(evidence.panel.allocation_metadata.get("allocation_rule", ""))
                        == "cost_optimal_two_stage_neyman"
                    ):
                        from .neyman import summarize_hierarchical_paired_risk

                        allocation_evidence = summarize_hierarchical_paired_risk(
                            evidence
                        ).to_allocation_state()
                    prior_panel = evidence.panel
                    if final_decision.action != "sample_more":
                        break
                if final_decision is None:
                    raise RuntimeError("sampled-path proposal produced no evidence")
                accepted = final_decision.action == "accept"
                steps.append(
                    SampledPathStep(
                        proposal_index=proposal_index,
                        partition_id=turn,
                        previous_density=densities[turn],
                        proposed_density=proposed,
                        accepted=accepted,
                        decision=final_decision,
                        evidence_stage_hashes=tuple(evidence_hashes),
                    )
                )
                proposal_index += 1
                if not accepted:
                    break
                current_masks = {key: np.asarray(value, dtype=bool).copy() for key, value in candidate_masks.items()}
                densities[turn] = proposed
                cursors[turn] += 1
                advanced = True
                if proposal_index >= maximum_proposals:
                    break
        stalled_turns = 0 if advanced else stalled_turns + 1
        if stalled_turns >= 2:
            return SampledPathResult(
                positive_density=densities["positive"],
                negative_density=densities["negative"],
                committed_masks=current_masks,
                steps=tuple(steps),
                stopping_reason="both_partitions_stalled",
            )
        turn = "negative" if turn == "positive" else "positive"
    raise RuntimeError("statistical symmetric sampled path exceeded its proposal guard")


__all__ = [
    "CrossfitPanel",
    "PairedCandidateEvidence",
    "PairedPosteriorDecision",
    "SAMPLED_PATH_EVIDENCE_SCHEMA",
    "SAMPLED_PATH_FRONTIER_V1",
    "SampledComparisonRequest",
    "SampledPathEvaluationBackend",
    "SampledPathPolicy",
    "SampledReferenceEvidence",
    "SampledPathResult",
    "SampledPathStep",
    "fold_warm_start",
    "paired_sensitivity_decision",
    "sampled_path_mask_hash",
    "trace_statistical_symmetric_sampled_path",
]
