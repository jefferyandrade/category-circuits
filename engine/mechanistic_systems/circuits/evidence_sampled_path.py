"""Evidence-foundation sampled-path: staged-valid decisions under one ledger.

Parallel to ``sampled_path.trace_statistical_symmetric_sampled_path``: the
proposal loop, panels, cross-fitting, warm starts, and Neyman hooks are
unchanged.  What changes is the inferential engine:

- interim looks across the sealed stage ladder use an exact union-bound alpha
  split (``staged_alpha``), so stopping early on a clear stage is valid rather
  than a silent peeking violation;
- variances are moderated by an empirical-Bayes prior fitted across the run's
  own proposal population instead of fixed constants;
- every accept spends from one pre-registered run-level alpha ledger in
  proportion to its removed route cost, and the spent ledger ships in the
  result;
- images consumed are counted per stage, because the design's efficiency claim
  is an output, not an assumption.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mechanistic_systems.evidence import (
    AlphaWealthLedger,
    staged_alpha,
    student_t_interval,
)
from mechanistic_systems.evidence.moderation import (
    VarianceModerationPrior,
    fit_variance_prior,
    moderated_variances,
)

from .sampled_path import (
    CrossfitPanel,
    PairedCandidateEvidence,
    PairedPosteriorDecision,
    SampledComparisonRequest,
    SampledPathEvaluationBackend,
    SampledPathStep,
    fold_warm_start,
    sampled_path_mask_hash,
)

EVIDENCE_SAMPLED_PATH_V1 = "evidence_foundation_sampled_path_v1"


def _normal_quantile(alpha: float) -> float:
    """Upper-tail normal quantile used only for futility projection."""

    from statistics import NormalDist

    return float(NormalDist().inv_cdf(1.0 - float(alpha)))


@dataclass(frozen=True)
class EvidenceSampledPathPolicy:
    """Frozen decision policy for the evidence-foundation sampled path."""

    sample_stages_per_class: tuple[int, ...] = (4, 8, 16, 32)
    fold_count: int = 2
    run_alpha: float = 0.05
    reject_alpha: float = 0.05
    minimum_decision_alpha: float = 1e-5
    moderation_minimum_cells: int = 8
    moderation_refresh_interval: int = 4
    futility_stopping: bool = True
    futility_margin: float = 1.0
    voi_images_per_route: float | None = None
    finite_population_correction: bool = True
    design_weighted: bool = True
    unresolved_final_action: str = "reject"
    random_seed: int = 0

    def validate(self) -> None:
        stages = self.sample_stages_per_class
        if not stages or tuple(sorted(set(stages))) != stages or min(stages) <= 0:
            raise ValueError("sample stages must be positive, unique, and increasing")
        if self.fold_count < 2:
            raise ValueError("evidence sampled paths require at least two cross-fit folds")
        if not 0.0 < self.run_alpha < 1.0:
            raise ValueError("run alpha must lie in (0,1)")
        if not 0.0 < self.reject_alpha < 1.0:
            raise ValueError("reject alpha must lie in (0,1)")
        if not 0.0 < self.minimum_decision_alpha < self.run_alpha:
            raise ValueError("minimum decision alpha must lie below the run alpha")
        if self.moderation_minimum_cells < 2 or self.moderation_refresh_interval < 1:
            raise ValueError("moderation accumulator policy is invalid")
        if self.unresolved_final_action not in {"reject", "accept"}:
            raise ValueError("unresolved final action must be reject or accept")
        if not math.isfinite(self.futility_margin) or self.futility_margin <= 0.0:
            raise ValueError("futility margin must be finite and positive")
        if self.voi_images_per_route is not None and (
            not math.isfinite(self.voi_images_per_route)
            or self.voi_images_per_route <= 0.0
        ):
            raise ValueError("VOI images-per-route must be finite and positive")


class ModerationAccumulator:
    """Population of per-proposal variances feeding the moderation prior."""

    def __init__(self, *, minimum_cells: int, refresh_interval: int) -> None:
        self.minimum_cells = int(minimum_cells)
        self.refresh_interval = int(refresh_interval)
        self._variances: list[float] = []
        self._dof: list[float] = []
        self._prior: VarianceModerationPrior | None = None
        self._stale = 0

    def observe(self, variance: float, dof: float) -> None:
        if math.isfinite(variance) and variance > 0.0 and dof >= 1.0:
            self._variances.append(float(variance))
            self._dof.append(float(dof))
            self._stale += 1

    def prior(self) -> VarianceModerationPrior | None:
        if len(self._variances) < self.minimum_cells:
            return None
        if self._prior is None or self._stale >= self.refresh_interval:
            self._prior = fit_variance_prior(
                np.asarray(self._variances), np.asarray(self._dof)
            )
            self._stale = 0
        return self._prior


def _weighted_sensitivity_moments(
    values: np.ndarray, weights: np.ndarray
) -> tuple[float, float, float]:
    normalized = weights / weights.sum()
    mean = float(np.sum(normalized * values))
    raw_variance = float(np.sum(normalized * np.square(values - mean)))
    effective_n = float(np.square(weights.sum()) / np.square(weights).sum())
    if effective_n > 1.0:
        raw_variance *= effective_n / (effective_n - 1.0)
    return mean, raw_variance, effective_n


def evidence_paired_sensitivity_decision(
    evidence: PairedCandidateEvidence,
    *,
    policy: EvidenceSampledPathPolicy,
    dense_anchor_loss: float,
    fully_pruned_anchor_loss: float,
    requested_density_delta: float,
    total_layer_route_count: int,
    sensitivity_threshold: float,
    stage_index: int,
    decision_alpha: float,
    moderation_prior: VarianceModerationPrior | None,
) -> tuple[PairedPosteriorDecision, float, float]:
    """One staged-valid look; returns (decision, variance, dof) for moderation."""

    policy.validate()
    evidence.validate(fold_count=policy.fold_count)
    span = float(fully_pruned_anchor_loss) - float(dense_anchor_loss)
    denominator = float(requested_density_delta) * int(total_layer_route_count)
    removed = abs(denominator)
    if abs(span) <= 1e-12 or removed <= 0.0:
        raise ValueError("sensitivity requires nondegenerate anchors and route cost")
    raw_delta = np.asarray(evidence.candidate_loss, dtype=np.float64) - np.asarray(
        evidence.committed_loss, dtype=np.float64
    )
    allocation_rule = str(evidence.panel.allocation_metadata.get("allocation_rule", ""))
    planned_looks = len(policy.sample_stages_per_class)
    # The run-level guarantee is one-sided on accepts, so only the accept
    # bound spends ledger-grade alpha; rejection is the conservative free
    # action and uses the policy's lenient staged alpha.
    accept_alpha = staged_alpha(float(decision_alpha), planned_looks=planned_looks)
    reject_alpha = staged_alpha(float(policy.reject_alpha), planned_looks=planned_looks)
    if allocation_rule == "cost_optimal_two_stage_neyman":
        from .neyman import summarize_hierarchical_paired_risk

        risk = summarize_hierarchical_paired_risk(evidence)
        mean = -float(risk.posterior_mean_delta) / float(span) / float(denominator)
        scale = abs(float(span) * float(denominator))
        raw_variance = float(risk.posterior_sd_delta) ** 2 / max(scale * scale, 1e-24)
        effective_n = 1.0
        count_for_interval = 1.0
        sample_variance = raw_variance
        dof = max(
            float(
                min(
                    (item["sample_count"] for item in risk.metadata["fold_target_support"]),
                    default=2,
                )
            )
            - 1.0,
            1.0,
        )
        normalized_delta = float(risk.posterior_mean_delta / span)
    else:
        sensitivity = -raw_delta / (span * denominator)
        weights = (
            np.asarray(evidence.panel.sampling_weights, dtype=np.float64)
            if policy.design_weighted and evidence.panel.sampling_weights
            else np.ones(sensitivity.size, dtype=np.float64)
        )
        mean, sample_variance, effective_n = _weighted_sensitivity_moments(
            sensitivity, weights
        )
        dof = max(effective_n - 1.0, 1.0)
        count_for_interval = max(effective_n, 1.0)
        if policy.finite_population_correction and evidence.panel.sampling_weights:
            population_size = max(
                float(sensitivity.size), float(sum(evidence.panel.sampling_weights))
            )
            sample_variance *= max(0.0, 1.0 - effective_n / population_size)
        normalized_delta = float(np.average(raw_delta / span, weights=weights))
    variance_for_bound = sample_variance
    dof_for_bound = dof
    if moderation_prior is not None:
        moderated, total_dof = moderated_variances(
            np.asarray([sample_variance]), np.asarray([dof]), moderation_prior
        )
        variance_for_bound = float(moderated[0])
        dof_for_bound = float(total_dof[0])
    if allocation_rule == "cost_optimal_two_stage_neyman":
        # The stratified estimator already returns the variance OF THE MEAN.
        interval_variance = variance_for_bound
        interval_count = 1.0
    else:
        interval_variance = variance_for_bound
        interval_count = count_for_interval
    accept_interval = student_t_interval(
        mean,
        interval_variance,
        interval_count,
        alpha=accept_alpha,
        dof=dof_for_bound,
        one_sided=True,
    )
    reject_interval = student_t_interval(
        mean,
        interval_variance,
        interval_count,
        alpha=reject_alpha,
        dof=dof_for_bound,
        one_sided=True,
    )
    upper = float(accept_interval.one_sided_upper()[()])
    lower = float(reject_interval.one_sided_lower()[()])
    standard_error = float(accept_interval.standard_error[()])
    threshold = float(sensitivity_threshold)
    final_stage = stage_index == planned_looks - 1
    if standard_error <= 1e-15 or not math.isfinite(standard_error):
        accept_probability = 1.0 if mean <= threshold else 0.0
    else:
        z = (threshold - mean) / standard_error
        accept_probability = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    if upper <= threshold:
        action, reason = "accept", "staged_upper_bound_below_threshold"
    elif lower > threshold:
        action, reason = "reject", "staged_lower_bound_above_threshold"
    elif final_stage:
        action, reason = (
            policy.unresolved_final_action,
            "staged_bounds_unresolved_at_final_stage",
        )
    else:
        action, reason = "sample_more", "staged_bounds_straddle_threshold"
    if action == "sample_more" and (
        policy.futility_stopping or policy.voi_images_per_route is not None
    ):
        # Escalation gates: both affect only defaults, never certificates,
        # so validity is untouched. Futility declines when even the final
        # stage's projected information cannot resolve either bound; the VOI
        # gate declines when resolving is possible but not worth the images
        # (a confirmed reject equals a default reject, so only the accept
        # branch carries value).
        stage_growth = float(policy.sample_stages_per_class[-1]) / float(
            max(policy.sample_stages_per_class[stage_index], 1)
        )
        deviation = float(standard_error) * math.sqrt(max(interval_count, 1.0))
        z_accept = abs(_normal_quantile(accept_alpha))
        z_reject = abs(_normal_quantile(reject_alpha))
        required = math.inf
        if mean < threshold:
            required = (z_accept * deviation / max(threshold - mean, 1e-18)) ** 2
        elif mean > threshold:
            required = (z_reject * deviation / max(mean - threshold, 1e-18)) ** 2
        information_ratio = required / max(interval_count, 1.0)
        if policy.futility_stopping and information_ratio > (
            stage_growth * policy.futility_margin
        ):
            action, reason = (
                policy.unresolved_final_action,
                "futility_projected_final_information_insufficient",
            )
        elif policy.voi_images_per_route is not None:
            panel_images = float(raw_delta.size)
            additional_images = (
                min(information_ratio, stage_growth) - 1.0
            ) * panel_images
            expected_value = (
                accept_probability * float(policy.voi_images_per_route) * removed
            )
            if additional_images > expected_value:
                action, reason = (
                    policy.unresolved_final_action,
                    "value_of_information_below_escalation_cost",
                )
    decision = PairedPosteriorDecision(
        action=action,
        posterior_mean_sensitivity=float(mean),
        posterior_sd_sensitivity=standard_error,
        accept_probability=float(accept_probability),
        reject_probability=float(1.0 - accept_probability),
        normalized_loss_delta=normalized_delta,
        removed_route_cost=removed,
        context_count=int(raw_delta.size),
        stage_index=int(evidence.panel.stage_index),
        panel_hash=evidence.panel.panel_hash,
        reason=reason,
        decision_rule="evidence_staged_asymmetric_t_ledgered_accept_free_reject",
    )
    return decision, float(sample_variance), float(dof)


@dataclass(frozen=True)
class EvidenceSampledPathResult:
    positive_density: float
    negative_density: float
    committed_masks: Mapping[str, np.ndarray]
    steps: tuple[SampledPathStep, ...]
    stopping_reason: str
    images_consumed: int
    images_consumed_by_stage: Mapping[int, int]
    ledger_receipt: Mapping[str, Any]
    moderation_prior: Mapping[str, Any] | None
    algorithm_version: str = EVIDENCE_SAMPLED_PATH_V1


def _schedule_total_weight(
    schedules: Mapping[str, tuple[float, ...]], total_layer_route_count: int
) -> float:
    total = 0.0
    for schedule in schedules.values():
        for left, right in zip(schedule, schedule[1:]):
            total += abs(left - right) * float(total_layer_route_count)
    return max(total, 1e-12)


def trace_evidence_symmetric_sampled_path(
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
    policy: EvidenceSampledPathPolicy,
    initial_committed_warm_start: Mapping[str, Any] | None = None,
    initial_turn: str = "negative",
    maximum_proposals: int = 10000,
) -> EvidenceSampledPathResult:
    """Symmetric full turns with staged-valid decisions and one alpha ledger."""

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
    ledger = AlphaWealthLedger(
        total_alpha=policy.run_alpha,
        total_weight=_schedule_total_weight(schedules, total_layer_route_count),
        minimum_alpha=policy.minimum_decision_alpha,
    )
    moderation = ModerationAccumulator(
        minimum_cells=policy.moderation_minimum_cells,
        refresh_interval=policy.moderation_refresh_interval,
    )
    current_masks = {
        key: np.asarray(value, dtype=bool).copy() for key, value in committed_masks.items()
    }
    cursors = {"positive": 0, "negative": 0}
    densities = {"positive": 1.0, "negative": 1.0}
    initial_mask_hash = sampled_path_mask_hash(current_masks)
    warm_starts: dict[str, Mapping[str, Any]] = {}
    if initial_committed_warm_start:
        warm_starts[initial_mask_hash] = dict(initial_committed_warm_start)
    turn = initial_turn
    stalled_turns = 0
    proposal_index = 0
    steps: list[SampledPathStep] = []
    images_consumed = 0
    images_by_stage: dict[int, int] = {}
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
                proposal_weight = abs(proposed - densities[turn]) * float(
                    total_layer_route_count
                )
                decision_id = f"{query_id}:{layer_id}:{turn}:{proposal_index}"
                decision_alpha = ledger.preview(proposal_weight)
                prior_panel: CrossfitPanel | None = None
                allocation_evidence: Mapping[str, Any] = {}
                evidence_hashes: list[str] = []
                final_decision: PairedPosteriorDecision | None = None
                for stage_index, stage_per_class in enumerate(
                    policy.sample_stages_per_class
                ):
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
                    evidence = backend.evaluate_pair(
                        current_masks, candidate_masks, request
                    )
                    evidence.validate(fold_count=policy.fold_count)
                    if (
                        prior_panel is not None
                        and str(
                            evidence.panel.allocation_metadata.get("allocation_rule", "")
                        )
                        != "cost_optimal_two_stage_neyman"
                    ):
                        evidence.panel.assert_nested_extension(prior_panel)
                    if evidence.committed_mask_hash != committed_hash:
                        raise ValueError(
                            "backend committed-mask identity differs from request"
                        )
                    if evidence.candidate_mask_hash != candidate_hash:
                        raise ValueError(
                            "backend candidate-mask identity differs from request"
                        )
                    previous_panel_size = (
                        len(prior_panel.context_ids) if prior_panel is not None else 0
                    )
                    stage_images = max(
                        0, len(evidence.panel.context_ids) - previous_panel_size
                    )
                    images_consumed += stage_images
                    images_by_stage[stage_index] = (
                        images_by_stage.get(stage_index, 0) + stage_images
                    )
                    evidence_hashes.append(evidence.panel.panel_hash)
                    final_decision, variance, dof = evidence_paired_sensitivity_decision(
                        evidence,
                        policy=policy,
                        dense_anchor_loss=dense_anchor_loss,
                        fully_pruned_anchor_loss=fully_pruned_anchor_loss,
                        requested_density_delta=proposed - densities[turn],
                        total_layer_route_count=total_layer_route_count,
                        sensitivity_threshold=sensitivity_threshold,
                        stage_index=stage_index,
                        decision_alpha=decision_alpha,
                        moderation_prior=moderation.prior(),
                    )
                    moderation.observe(variance, dof)
                    warm_starts[committed_hash] = fold_warm_start(
                        evidence.committed_calibration_by_fold
                    )
                    warm_starts[candidate_hash] = fold_warm_start(
                        evidence.candidate_calibration_by_fold
                    )
                    if (
                        str(
                            evidence.panel.allocation_metadata.get("allocation_rule", "")
                        )
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
                    raise RuntimeError("evidence sampled-path proposal produced no evidence")
                accepted = final_decision.action == "accept"
                ledger.record(
                    decision_id=decision_id,
                    kind="density_step",
                    weight=proposal_weight,
                    alpha_allocated=decision_alpha,
                    outcome="accepted" if accepted else "rejected",
                    charged=accepted,
                    metadata={
                        "reason": final_decision.reason,
                        "stage_index": final_decision.stage_index,
                        "panel_hash": final_decision.panel_hash,
                    },
                )
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
                current_masks = {
                    key: np.asarray(value, dtype=bool).copy()
                    for key, value in candidate_masks.items()
                }
                densities[turn] = proposed
                cursors[turn] += 1
                advanced = True
                if proposal_index >= maximum_proposals:
                    break
        stalled_turns = 0 if advanced else stalled_turns + 1
        if stalled_turns >= 2:
            fitted = moderation.prior()
            return EvidenceSampledPathResult(
                positive_density=densities["positive"],
                negative_density=densities["negative"],
                committed_masks=current_masks,
                steps=tuple(steps),
                stopping_reason="both_partitions_stalled",
                images_consumed=images_consumed,
                images_consumed_by_stage=dict(images_by_stage),
                ledger_receipt=ledger.receipt(),
                moderation_prior=fitted.to_dict() if fitted is not None else None,
            )
        turn = "negative" if turn == "positive" else "positive"
    raise RuntimeError("evidence symmetric sampled path exceeded its proposal guard")


__all__ = [
    "EVIDENCE_SAMPLED_PATH_V1",
    "EvidenceSampledPathPolicy",
    "EvidenceSampledPathResult",
    "ModerationAccumulator",
    "evidence_paired_sensitivity_decision",
    "trace_evidence_symmetric_sampled_path",
]
