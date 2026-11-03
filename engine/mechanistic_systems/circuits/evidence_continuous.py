"""Historical exploratory continuous-v2 evidence controller.

Implements the continuous-v2 compatibility design:

- **Interface**: the caller declares a distortion budget Δ and a run
  confidence ε. Per-step thresholds are the marginal price of the remaining
  budget; every accept debits Δ by its one-sided interval upper bound.
- **Continuous looks**: alpha spending over information fraction with
  Bonferroni increments and O'Brien–Fleming-/Pocock-like pacing. Ordinary
  weighted Student-t intervals and empirical-Bayes variance moderation do
  not by themselves establish conditional anytime validity under adaptive
  allocation, so this identity is exploratory rather than claim-bearing.
- **Adaptive bisection**: the density schedule is replaced by per-partition
  root finding — an accept doubles the next step and a reject halves it —
  with masks nested along the sealed retention order.
- **Gates**: futility projects against the far horizon (physics); the VOI
  price governs escalation (economics).

Claim-bearing v0--v5 campaigns use ``evidence_foundation_glocal_v3`` in
shadow until its separate promotion gates pass.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

import numpy as np

from mechanistic_systems.evidence import (
    AlphaWealthLedger,
    student_t_interval,
)
from mechanistic_systems.evidence.moderation import (
    moderated_variances,
)

from .evidence_sampled_path import ModerationAccumulator, _weighted_sensitivity_moments
from .glocal_sampled_path import (
    GlocalSampledComparisonRequest,
    GlocalSampledPathEvaluationBackend,
)
from .sampled_path import (
    CrossfitPanel,
    PairedCandidateEvidence,
    fold_warm_start,
    sampled_path_mask_hash,
)

EVIDENCE_CONTINUOUS_V2 = "evidence_foundation_continuous_v2"


def _phi_inverse(probability: float) -> float:
    return float(NormalDist().inv_cdf(probability))


def obrien_fleming_spent(alpha: float, fraction: float) -> float:
    """Cumulative alpha spent at information fraction t (one-sided OBF-like)."""

    t = min(max(float(fraction), 1e-12), 1.0)
    boundary = _phi_inverse(1.0 - float(alpha) / 2.0) / math.sqrt(t)
    return float(min(alpha, 2.0 * (1.0 - NormalDist().cdf(boundary))))


def pocock_spent(alpha: float, fraction: float) -> float:
    """Cumulative alpha spent at information fraction t (Pocock-like)."""

    t = min(max(float(fraction), 0.0), 1.0)
    return float(alpha * math.log(1.0 + (math.e - 1.0) * t))


@dataclass(frozen=True)
class NaturalizationContext:
    """Runtime facts from which the derived parameters are computed.

    Nothing here is a tuning choice: class count and bank depth are facts of
    the sealed data, the GPU batch is a fact of the hardware, partition route
    counts are facts of the architecture, and the frontier prior is learned
    from previously completed queries in the same campaign.
    """

    class_count: int
    bank_depth_per_class: int
    gpu_batch_size: int
    partition_route_counts: Mapping[str, int]
    frontier_prior: Mapping[str, float] | None = None

    def validate(self) -> None:
        if self.class_count < 2 or self.bank_depth_per_class < 2:
            raise ValueError("naturalization needs classes and bank depth")
        if self.gpu_batch_size < 1 or not self.partition_route_counts:
            raise ValueError("naturalization needs a batch size and partitions")


@dataclass(frozen=True)
class ContinuousEvidencePolicy:
    """Two declared outcomes; every None below is derived, not defaulted."""

    distortion_budget: float
    run_alpha: float = 0.05
    image_budget: float | None = None
    voi_images_per_route: float | None = None
    reject_alpha: float = 0.05
    futility_horizon_images: int | None = None
    futility_margin: float = 1.0
    look_batch_images: int | None = None
    fold_count: int = 2
    minimum_decision_alpha: float = 1e-5
    moderation_minimum_cells: int = 8
    moderation_refresh_interval: int = 4
    initial_density_step: float | None = 0.25
    minimum_density_step: float | None = 0.02
    minimum_density: float = 0.05
    unresolved_action: str = "reject"
    adaptive_voi_pacing: bool = False
    random_seed: int = 0

    def validate(self) -> None:
        if not math.isfinite(self.distortion_budget) or self.distortion_budget <= 0.0:
            raise ValueError("the distortion budget must be finite and positive")
        for name, value in (("run", self.run_alpha), ("reject", self.reject_alpha)):
            if not 0.0 < value < 1.0:
                raise ValueError(f"the {name} alpha must lie in (0,1)")
        if self.image_budget is not None and self.image_budget <= 0.0:
            raise ValueError("the image budget must be positive when declared")
        if self.voi_images_per_route is not None and self.voi_images_per_route <= 0.0:
            raise ValueError("the VOI price must be positive when declared")
        if self.fold_count < 2:
            raise ValueError("cross-fitting requires at least two folds")
        if self.look_batch_images is not None and self.look_batch_images <= 0:
            raise ValueError("the look batch must be positive when declared")
        if (
            self.futility_horizon_images is not None
            and self.look_batch_images is not None
            and self.futility_horizon_images < self.look_batch_images
        ):
            raise ValueError("the futility horizon must be at least one look batch")
        if (
            self.minimum_density_step is not None
            and self.initial_density_step is not None
        ):
            if not 0.0 < self.minimum_density_step <= self.initial_density_step <= 0.95:
                raise ValueError("bisection steps must satisfy 0 < min <= initial")
        if not 0.0 < self.minimum_density < 1.0:
            raise ValueError("the minimum density must lie in (0,1)")
        if self.unresolved_action not in {"reject", "accept"}:
            raise ValueError("the unresolved action must be reject or accept")

    def resolve(self, context: NaturalizationContext) -> "ContinuousEvidencePolicy":
        """Derive every None parameter from runtime facts.

        Returns a fully concrete policy plus provenance via
        ``naturalization_provenance``. Derivations: one look per GPU batch;
        the futility horizon is what the sealed bank holds, tightened by the
        image budget; the resolution floor is route granularity or the
        ledger's minimum certifiable stake, whichever is coarser; the initial
        step aims at the cross-query frontier prior when one exists; the VOI
        price paces the declared image budget adaptively.
        """

        context.validate()
        # The look unit is one saturating GPU batch (a measured hardware
        # fact); composition belongs to the allocator; information is counted
        # in images, never in per-class units.
        look = self.look_batch_images or int(context.gpu_batch_size)
        horizon = self.futility_horizon_images
        if horizon is None:
            horizon = int(context.bank_depth_per_class) * int(context.class_count)
            if self.image_budget is not None:
                horizon = min(horizon, max(look, int(self.image_budget)))
        total_routes = float(sum(context.partition_route_counts.values()))
        minimum_step = self.minimum_density_step
        if minimum_step is None:
            granularity = 1.0 / float(min(context.partition_route_counts.values()))
            ledger_floor = (
                (self.minimum_decision_alpha / self.run_alpha)
                * 2.0
                * (1.0 - self.minimum_density)
            )
            minimum_step = max(granularity, ledger_floor)
        initial_step = self.initial_density_step
        if initial_step is None:
            if context.frontier_prior:
                # Aim at the learned frontier, but never let a "nothing was
                # prunable" prior strangle the next query's exploration: the
                # first step keeps a 0.1 floor so bisection can still probe.
                prior = float(np.mean(list(context.frontier_prior.values())))
                initial_step = min(
                    max(1.0 - prior, minimum_step, 0.1),
                    1.0 - self.minimum_density,
                )
            else:
                initial_step = max(0.25, minimum_step)
        voi = self.voi_images_per_route
        pacing = self.adaptive_voi_pacing
        if voi is None and self.image_budget is not None:
            voi = float(self.image_budget) / (
                2.0 * (1.0 - self.minimum_density) * total_routes
            )
            pacing = True
        resolved = ContinuousEvidencePolicy(
            distortion_budget=self.distortion_budget,
            run_alpha=self.run_alpha,
            image_budget=self.image_budget,
            voi_images_per_route=voi,
            reject_alpha=self.reject_alpha,
            futility_horizon_images=int(horizon),
            futility_margin=self.futility_margin,
            look_batch_images=int(look),
            fold_count=self.fold_count,
            minimum_decision_alpha=self.minimum_decision_alpha,
            moderation_minimum_cells=self.moderation_minimum_cells,
            moderation_refresh_interval=self.moderation_refresh_interval,
            initial_density_step=float(initial_step),
            minimum_density_step=float(minimum_step),
            minimum_density=self.minimum_density,
            unresolved_action=self.unresolved_action,
            adaptive_voi_pacing=pacing,
            random_seed=self.random_seed,
        )
        resolved.validate()
        return resolved


@dataclass(frozen=True)
class ContinuousDecision:
    partition_id: str
    previous_density: float
    proposed_density: float
    action: str
    reason: str
    looks: int
    panel_images: int
    mean_sensitivity: float
    certified_upper_bound: float
    threshold: float
    delta_debit: float
    alpha_allocated: float
    panel_hashes: tuple[str, ...]
    voi_price: float | None = None
    removed_route_count: int = 0
    certified_lower_bound: float = -math.inf
    confidence_level: float | None = None
    interval_method: str | None = None


@dataclass(frozen=True)
class ContinuousEvidenceResult:
    positive_density: float
    negative_density: float
    committed_masks: Mapping[str, np.ndarray]
    decisions: tuple[ContinuousDecision, ...]
    stopping_reason: str
    delta_budget: float
    delta_spent: float
    images_consumed: int
    ledger_receipt: Mapping[str, Any]
    moderation_prior: Mapping[str, Any] | None
    resolved_policy: Mapping[str, Any] | None = None
    resource_usage: Mapping[str, Any] | None = None
    algorithm_version: str = EVIDENCE_CONTINUOUS_V2


def _mask_hamming_distance(
    committed: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
) -> int:
    # A candidate view over exactly this committed state knows
    # its delta; it only clears retained bits, so the distance is the count.
    if getattr(candidate, "parent", None) is committed and hasattr(
        candidate, "removed_count"
    ):
        return int(candidate.removed_count)
    if set(committed) != set(candidate):
        raise ValueError("candidate mask layers differ from the committed state")
    distance = 0
    for layer_id in committed:
        left = np.asarray(committed[layer_id], dtype=bool)
        right = np.asarray(candidate[layer_id], dtype=bool)
        if left.shape != right.shape:
            raise ValueError(f"candidate mask shape changed for {layer_id}")
        distance += int(np.count_nonzero(left != right))
    return distance


def _decide_one_step(
    *,
    backend: GlocalSampledPathEvaluationBackend,
    committed_masks: Mapping[str, np.ndarray],
    candidate_masks: Mapping[str, np.ndarray],
    query_id: str,
    layer_id: str,
    partition_id: str,
    previous_density: float,
    proposed_density: float,
    proposal_index: int,
    removed_route_count: int,
    dense_anchor_loss: float,
    fully_pruned_anchor_loss: float,
    threshold: float,
    decision_alpha: float,
    policy: ContinuousEvidencePolicy,
    moderation: ModerationAccumulator,
    warm_starts: dict[str, Mapping[str, Any]],
    voi_price: float | None,
    class_count: int,
    image_budget: float | None,
) -> tuple[str, str, int, int, float, float, tuple[str, ...]]:
    """Run continuous looks for one bisection proposal.

    Information is accounted in images actually delivered by the backend —
    the fraction driving the spending functions is measured from the panel,
    never assumed from the request — with a plateau safeguard: if the panel
    stops growing before the horizon, the decision defaults rather than
    spending against information that is not arriving.

    Returns (action, reason, looks, panel_images, mean, certified_upper,
    panel_hashes)."""

    span = float(fully_pruned_anchor_loss) - float(dense_anchor_loss)
    direction = float(proposed_density) - float(previous_density)
    if abs(direction) <= 1e-15:
        raise ValueError("continuous decisions require a nonzero density direction")
    denominator = math.copysign(float(removed_route_count), direction)
    removed = float(removed_route_count)
    if abs(span) <= 1e-12 or removed <= 0.0:
        raise ValueError("continuous decisions need nondegenerate anchors and stakes")
    horizon_images = int(policy.futility_horizon_images)
    look_images = int(policy.look_batch_images)
    prior_panel: CrossfitPanel | None = None
    allocation_evidence: Mapping[str, Any] = {}
    panel_hashes: list[str] = []
    spent_accept = 0.0
    spent_reject = 0.0
    action, reason = policy.unresolved_action, "horizon_exhausted_unresolved"
    mean = 0.0
    upper_bound = math.inf
    looks = 0
    target_images = 0
    actual_images = 0
    stagnant_looks = 0
    # The masks are fixed for the whole proposal, so their identity hashes are
    # loop-invariant: compute once, not once per look.
    committed_hash = sampled_path_mask_hash(committed_masks)
    candidate_hash = sampled_path_mask_hash(candidate_masks)
    while target_images < horizon_images:
        maximum_new_contexts: int | None = None
        maximum_target = horizon_images
        if image_budget is not None:
            maximum_new_contexts = max(
                0,
                int(
                    math.floor(
                        float(image_budget)
                        - float(getattr(backend, "decision_images_consumed", 0))
                    )
                ),
            )
            maximum_target = min(
                maximum_target,
                actual_images + maximum_new_contexts,
            )
        next_target = min(target_images + look_images, maximum_target)
        if next_target <= target_images:
            action, reason = (
                policy.unresolved_action,
                "image_budget_exhausted_unresolved",
            )
            break
        if next_target < 2 * int(policy.fold_count):
            action, reason = (
                policy.unresolved_action,
                "image_budget_minimum_packet_unavailable",
            )
            break
        target_images = next_target
        per_class = max(1, math.ceil(target_images / max(class_count, 1)))
        looks += 1
        request = GlocalSampledComparisonRequest(
            query_id=query_id,
            layer_id=layer_id,
            partition_id=partition_id,
            previous_density=float(previous_density),
            proposed_density=float(proposed_density),
            stage_index=looks - 1,
            stage_per_class=per_class,
            proposal_index=int(proposal_index),
            random_seed=policy.random_seed,
            fold_count=policy.fold_count,
            prior_panel=prior_panel,
            committed_warm_start=warm_starts.get(committed_hash),
            candidate_warm_start=warm_starts.get(candidate_hash),
            allocation_evidence=allocation_evidence,
            cumulative_target_images=target_images,
            maximum_new_contexts=maximum_new_contexts,
        )
        if target_images < horizon_images and hasattr(backend, "prefetch_request"):
            # Panels are mask-independent, so the next look's panel is
            # knowable now — hint it so the fill runs under this look's
            # forwards (full hiding of first-touch loads).
            hinted_target = min(
                target_images + look_images,
                maximum_target,
            )
            if hinted_target > target_images:
                backend.prefetch_request(
                    GlocalSampledComparisonRequest(
                        query_id=query_id,
                        layer_id=layer_id,
                        partition_id=partition_id,
                        previous_density=float(previous_density),
                        proposed_density=float(proposed_density),
                        stage_index=looks,
                        stage_per_class=max(
                            1, math.ceil(hinted_target / max(class_count, 1))
                        ),
                        proposal_index=int(proposal_index),
                        random_seed=policy.random_seed,
                        fold_count=policy.fold_count,
                        prior_panel=None,
                        allocation_evidence=allocation_evidence,
                        cumulative_target_images=hinted_target,
                        maximum_new_contexts=maximum_new_contexts,
                    )
                )
        evidence = backend.evaluate_pair(committed_masks, candidate_masks, request)
        evidence.validate(fold_count=policy.fold_count)
        allocation_rule = str(
            evidence.panel.allocation_metadata.get("allocation_rule", "")
        )
        if prior_panel is not None and (
            allocation_rule != "cost_optimal_two_stage_neyman"
            or request.cumulative_target_images is not None
        ):
            evidence.panel.assert_nested_extension(prior_panel)
        panel_hashes.append(evidence.panel.panel_hash)
        previous_actual = actual_images
        actual_images = len(evidence.panel.context_ids)
        stagnant_looks = stagnant_looks + 1 if actual_images <= previous_actual else 0
        warm_starts[committed_hash] = fold_warm_start(
            evidence.committed_calibration_by_fold
        )
        warm_starts[candidate_hash] = fold_warm_start(
            evidence.candidate_calibration_by_fold
        )
        raw_delta = np.asarray(evidence.candidate_loss, dtype=np.float64) - np.asarray(
            evidence.committed_loss, dtype=np.float64
        )
        sensitivity = -raw_delta / (span * denominator)
        weights = (
            np.asarray(evidence.panel.sampling_weights, dtype=np.float64)
            if evidence.panel.sampling_weights
            else np.ones(sensitivity.size, dtype=np.float64)
        )
        mean, variance, effective_n = _weighted_sensitivity_moments(
            sensitivity, weights
        )
        dof = max(effective_n - 1.0, 1.0)
        moderation.observe(variance, dof)
        prior = moderation.prior()
        variance_b, dof_b = variance, dof
        if prior is not None:
            moderated, total_dof = moderated_variances(
                np.asarray([variance]), np.asarray([dof]), prior
            )
            variance_b, dof_b = float(moderated[0]), float(total_dof[0])
        fraction = min(1.0, actual_images / float(horizon_images))
        accept_level = max(
            obrien_fleming_spent(decision_alpha, fraction) - spent_accept, 1e-12
        )
        reject_level = max(
            pocock_spent(policy.reject_alpha, fraction) - spent_reject, 1e-12
        )
        spent_accept += accept_level
        spent_reject += reject_level
        upper_bound = float(
            student_t_interval(
                mean,
                variance_b,
                max(effective_n, 1.0),
                alpha=accept_level,
                dof=dof_b,
                one_sided=True,
            ).one_sided_upper()[()]
        )
        lower_bound = float(
            student_t_interval(
                mean,
                variance_b,
                max(effective_n, 1.0),
                alpha=reject_level,
                dof=dof_b,
                one_sided=True,
            ).one_sided_lower()[()]
        )
        if allocation_rule == "cost_optimal_two_stage_neyman":
            from .neyman import summarize_hierarchical_paired_risk

            allocation_evidence = summarize_hierarchical_paired_risk(
                evidence
            ).to_allocation_state()
        prior_panel = evidence.panel
        if upper_bound <= threshold:
            action, reason = "accept", "spending_upper_bound_below_threshold"
            break
        if lower_bound > threshold:
            action, reason = "reject", "spending_lower_bound_above_threshold"
            break
        if stagnant_looks >= 2:
            action, reason = (
                policy.unresolved_action,
                "panel_growth_exhausted_unresolved",
            )
            break
        standard_error = math.sqrt(max(variance_b, 1e-24) / max(effective_n, 1.0))
        gap = abs(mean - threshold)
        z_reference = abs(
            _phi_inverse(
                1.0 - (decision_alpha if mean < threshold else policy.reject_alpha)
            )
        )
        deviation = standard_error * math.sqrt(max(effective_n, 1.0))
        required_ratio = (z_reference * deviation / max(gap, 1e-18)) ** 2 / max(
            effective_n, 1.0
        )
        horizon_ratio = horizon_images / float(max(actual_images, 1))
        if required_ratio > horizon_ratio * policy.futility_margin:
            action, reason = (
                policy.unresolved_action,
                "futility_projected_horizon_information_insufficient",
            )
            break
        if voi_price is not None:
            accept_probability = (
                1.0
                if standard_error <= 1e-15
                else 0.5
                * (1.0 + math.erf((threshold - mean) / standard_error / math.sqrt(2.0)))
            )
            panel_images = float(raw_delta.size)
            additional_images = (
                min(required_ratio, horizon_ratio) - 1.0
            ) * panel_images
            expected_value = accept_probability * float(voi_price) * removed
            if additional_images > expected_value:
                action, reason = (
                    policy.unresolved_action,
                    "value_of_information_below_escalation_cost",
                )
                break
    return (
        action,
        reason,
        looks,
        actual_images,
        float(mean),
        upper_bound,
        tuple(panel_hashes),
    )


def trace_continuous_evidence_frontier(
    *,
    backend: GlocalSampledPathEvaluationBackend,
    committed_masks: Mapping[str, np.ndarray],
    build_candidate: Any,
    query_id: str,
    layer_id: str,
    total_layer_route_count: int,
    dense_anchor_loss: float,
    fully_pruned_anchor_loss: float,
    policy: ContinuousEvidencePolicy,
    class_count: int,
    context: NaturalizationContext | None = None,
    initial_turn: str = "negative",
    maximum_decisions: int = 512,
    active_partitions: tuple[str, ...] = ("positive", "negative"),
) -> ContinuousEvidenceResult:
    """Certified bisection over the active partitions under (Δ, ε).

    A structurally empty partition (no routes carry its label) is passed as
    inactive: it is born resolved at density 1.0, receives no proposals, no
    hints, and no ledger stake.
    """

    if context is not None:
        policy = policy.resolve(context)
    policy.validate()
    active = tuple(dict.fromkeys(str(value) for value in active_partitions))
    if not active or not set(active) <= {"positive", "negative"}:
        raise ValueError(
            "active partitions must be a non-empty subset of positive/negative"
        )
    for name in (
        "futility_horizon_images",
        "look_batch_images",
        "initial_density_step",
        "minimum_density_step",
    ):
        if getattr(policy, name) is None:
            raise ValueError(
                f"{name} is underived; pass a NaturalizationContext to resolve it"
            )
    total_cost = float(total_layer_route_count)
    # Both partitions can each remove up to (1 - minimum_density) of the route
    # axis, and every accept's weight is its density displacement times the
    # full route count, so the pre-registered ledger weight is the two-sided
    # maximum prunable mass — not the route count alone.
    # One-sided prunable mass per ACTIVE partition; an inactive partition
    # contributes no displacement, so staking against it would be pure
    # conservatism, not validity.
    ledger = AlphaWealthLedger(
        total_alpha=policy.run_alpha,
        total_weight=(float(len(active)) * (1.0 - policy.minimum_density) * total_cost),
        minimum_alpha=policy.minimum_decision_alpha,
    )
    moderation = ModerationAccumulator(
        minimum_cells=policy.moderation_minimum_cells,
        refresh_interval=policy.moderation_refresh_interval,
    )
    masks = {
        key: np.asarray(value, dtype=bool).copy()
        for key, value in committed_masks.items()
    }
    densities = {"positive": 1.0, "negative": 1.0}
    steps = {
        "positive": float(policy.initial_density_step),
        "negative": float(policy.initial_density_step),
    }
    warm_starts: dict[str, Mapping[str, Any]] = {}
    decisions: list[ContinuousDecision] = []
    delta_remaining = float(policy.distortion_budget)
    turn = initial_turn if initial_turn in active else active[0]
    proposal_index = 0
    stopping_reason = "frontier_resolved_in_both_partitions"

    def _partition_done(partition: str) -> bool:
        if partition not in active:
            return True
        return steps[partition] < policy.minimum_density_step or densities[
            partition
        ] <= (policy.minimum_density + 1e-12)

    while proposal_index < maximum_decisions:
        if all(_partition_done(partition) for partition in densities):
            break
        if (
            policy.image_budget is not None
            and getattr(backend, "decision_images_consumed", 0) >= policy.image_budget
        ):
            stopping_reason = "image_budget_exhausted"
            break
        if _partition_done(turn):
            turn = "negative" if turn == "positive" else "positive"
            continue
        # One constant global price for the whole sweep: τ = Δ over the
        # initial prunable mass. Per-decision re-pricing would make the price
        # drift within the run (and vary across layers in the multi-layer
        # generalization); the budget is enforced through certified debits,
        # not through the price.
        threshold = float(policy.distortion_budget) / max(
            (2.0 - 2.0 * policy.minimum_density) * total_cost / 2.0, 1e-9
        )
        proposed = max(densities[turn] - steps[turn], policy.minimum_density)
        candidate = build_candidate(
            masks,
            partition_id=turn,
            positive_density=proposed if turn == "positive" else densities["positive"],
            negative_density=proposed if turn == "negative" else densities["negative"],
        )
        removed = float(_mask_hamming_distance(masks, candidate))
        if removed <= 0.0:
            # Density arithmetic can fall below the mask's discrete route
            # resolution. Such a no-op is not a statistical proposal and must
            # neither spend alpha nor claim released cost.
            steps[turn] = 0.0
            turn = "negative" if turn == "positive" else "positive"
            continue
        if hasattr(backend, "prefetch_request"):
            # Cross-decision hint: alternation makes the next decision's
            # partition state independent of this decision's outcome, so its
            # first look is knowable now — its data loads under this
            # decision's forwards. A mispredicted hint (peer partition
            # finishing first) costs only cached bytes.
            peer = "negative" if turn == "positive" else "positive"
            if not _partition_done(peer):
                peer_proposed = max(
                    densities[peer] - steps[peer], policy.minimum_density
                )
                remaining_hint_budget = None
                if policy.image_budget is not None:
                    remaining_hint_budget = max(
                        0,
                        int(
                            math.floor(
                                float(policy.image_budget)
                                - float(
                                    getattr(
                                        backend,
                                        "decision_images_consumed",
                                        0,
                                    )
                                )
                            )
                        ),
                    )
                first_target = min(
                    int(policy.look_batch_images),
                    int(policy.futility_horizon_images),
                )
                if remaining_hint_budget is None or remaining_hint_budget >= 2 * int(
                    policy.fold_count
                ):
                    if remaining_hint_budget is not None:
                        first_target = min(first_target, remaining_hint_budget)
                    backend.prefetch_request(
                        GlocalSampledComparisonRequest(
                            query_id=query_id,
                            layer_id=layer_id,
                            partition_id=peer,
                            previous_density=densities[peer],
                            proposed_density=peer_proposed,
                            stage_index=0,
                            stage_per_class=max(
                                1,
                                math.ceil(first_target / max(class_count, 1)),
                            ),
                            proposal_index=proposal_index + 1,
                            random_seed=policy.random_seed,
                            fold_count=policy.fold_count,
                            prior_panel=None,
                            allocation_evidence={},
                            cumulative_target_images=first_target,
                            maximum_new_contexts=remaining_hint_budget,
                        )
                    )
        decision_alpha = ledger.preview(removed)
        voi_price = policy.voi_images_per_route
        if policy.adaptive_voi_pacing and policy.image_budget is not None:
            remaining_images = max(
                float(policy.image_budget)
                - float(getattr(backend, "decision_images_consumed", 0)),
                0.0,
            )
            remaining_mass = max(
                sum(
                    (densities[partition] - policy.minimum_density) * total_cost
                    for partition in densities
                ),
                1e-9,
            )
            voi_price = max(remaining_images / remaining_mass, 1e-6)
        action, reason, looks, panel_images, mean, upper, hashes = _decide_one_step(
            backend=backend,
            committed_masks=masks,
            candidate_masks=candidate,
            query_id=query_id,
            layer_id=layer_id,
            partition_id=turn,
            previous_density=densities[turn],
            proposed_density=proposed,
            proposal_index=proposal_index,
            removed_route_count=int(removed),
            dense_anchor_loss=dense_anchor_loss,
            fully_pruned_anchor_loss=fully_pruned_anchor_loss,
            threshold=threshold,
            decision_alpha=decision_alpha,
            policy=policy,
            moderation=moderation,
            warm_starts=warm_starts,
            voi_price=voi_price,
            class_count=class_count,
            image_budget=policy.image_budget,
        )
        debit = 0.0
        accepted = action == "accept"
        if accepted:
            debit = max(upper, 0.0) * removed
            if debit > delta_remaining + 1e-12:
                accepted = False
                action, reason = "reject", "certified_debit_exceeds_remaining_budget"
        decision_id = f"{query_id}:{layer_id}:{turn}:{proposal_index}"
        if accepted:
            ledger.allocate(removed, decision_id=decision_id, kind="frontier_accept")
        ledger.record(
            decision_id=decision_id,
            kind="frontier_accept" if accepted else "frontier_decline",
            weight=removed,
            alpha_allocated=decision_alpha,
            outcome="accepted" if accepted else "rejected",
            charged=accepted,
            metadata={"reason": reason, "looks": looks, "panel_images": panel_images},
        )
        decisions.append(
            ContinuousDecision(
                partition_id=turn,
                previous_density=densities[turn],
                proposed_density=proposed,
                action="accept" if accepted else "reject",
                reason=reason,
                looks=looks,
                panel_images=panel_images,
                mean_sensitivity=mean,
                certified_upper_bound=upper,
                threshold=threshold,
                delta_debit=debit,
                alpha_allocated=decision_alpha,
                panel_hashes=hashes,
                voi_price=voi_price,
                removed_route_count=int(removed),
            )
        )
        proposal_index += 1
        if accepted:
            delta_remaining -= debit
            masks = {
                key: np.asarray(value, dtype=bool).copy()
                for key, value in candidate.items()
            }
            densities[turn] = proposed
            steps[turn] = min(
                2.0 * steps[turn], max(densities[turn] - policy.minimum_density, 0.0)
            )
        else:
            steps[turn] = steps[turn] / 2.0
        turn = "negative" if turn == "positive" else "positive"
    else:
        stopping_reason = "maximum_decisions_reached"
    fitted = moderation.prior()
    return ContinuousEvidenceResult(
        positive_density=densities["positive"],
        negative_density=densities["negative"],
        committed_masks=masks,
        decisions=tuple(decisions),
        stopping_reason=stopping_reason,
        delta_budget=float(policy.distortion_budget),
        delta_spent=float(policy.distortion_budget) - delta_remaining,
        images_consumed=int(getattr(backend, "decision_images_consumed", 0)),
        ledger_receipt=ledger.receipt(),
        moderation_prior=fitted.to_dict() if fitted is not None else None,
        resolved_policy={
            "futility_horizon_images": policy.futility_horizon_images,
            "look_batch_images": policy.look_batch_images,
            "initial_density_step": policy.initial_density_step,
            "minimum_density_step": policy.minimum_density_step,
            "voi_images_per_route": policy.voi_images_per_route,
            "adaptive_voi_pacing": policy.adaptive_voi_pacing,
            "image_budget": policy.image_budget,
        },
        resource_usage=(
            dict(backend.resource_usage)
            if isinstance(getattr(backend, "resource_usage", None), Mapping)
            else None
        ),
    )


__all__ = [
    "EVIDENCE_CONTINUOUS_V2",
    "NaturalizationContext",
    "ContinuousDecision",
    "ContinuousEvidencePolicy",
    "ContinuousEvidenceResult",
    "obrien_fleming_spent",
    "pocock_spent",
    "trace_continuous_evidence_frontier",
]
