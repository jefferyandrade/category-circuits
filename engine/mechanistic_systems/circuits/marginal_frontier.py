"""Reverse-hierarchical equalized-marginal pruning semantics."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


def geometric_retention_schedule(
    *,
    decay: float,
    final_density: float,
    group_size: int,
) -> tuple[float, ...]:
    """Reproduce the historical geometric schedule with explicit endpoints."""

    decay_value = float(decay)
    final_value = float(final_density)
    size = int(group_size)
    if not 0.0 < decay_value < 1.0 or not 0.0 <= final_value < 1.0 or size <= 0:
        raise ValueError("geometric schedule requires decay in (0,1), final in [0,1), and size > 0")
    if final_value > 0.0:
        interior_count = int(math.log(final_value) / math.log(decay_value))
    else:
        interior_count = -int(math.log(size) / math.log(decay_value))
    values = [1.0]
    values.extend(decay_value**index for index in range(1, max(0, interior_count) + 1))
    values.append(0.0)
    return tuple(dict.fromkeys(float(max(0.0, min(1.0, value))) for value in values))


@dataclass(frozen=True)
class MarginalFrontierStep:
    proposal_index: int
    partition_id: str
    previous_density: float
    proposed_density: float
    previous_loss: float
    proposed_loss: float
    normalized_loss_delta: float
    layer_scaled_removed_cost: float
    marginal_sensitivity: float
    threshold: float
    accepted: bool
    reason: str


@dataclass(frozen=True)
class MarginalFrontierPath:
    positive_density: float
    negative_density: float
    final_loss: float
    steps: tuple[MarginalFrontierStep, ...]
    stopping_reason: str
    denominator_policy: str = "requested_sign_density_delta_times_total_layer_route_count"


def marginal_step_sensitivity(
    *,
    previous_loss: float,
    proposed_loss: float,
    dense_anchor_loss: float,
    fully_pruned_anchor_loss: float,
    previous_density: float,
    proposed_density: float,
    total_layer_route_count: int,
) -> tuple[float, float, float]:
    """Return normalized loss delta, layer-scaled cost, and sensitivity."""

    span = abs(float(fully_pruned_anchor_loss) - float(dense_anchor_loss))
    if not np.isfinite(span) or span <= 1e-12:
        raise ValueError("fully pruned and dense layer anchors must differ")
    removed = (float(previous_density) - float(proposed_density)) * int(total_layer_route_count)
    if not np.isfinite(removed) or removed <= 0.0:
        raise ValueError("marginal proposal must remove positive layer-scaled cost")
    normalized_delta = (float(proposed_loss) - float(previous_loss)) / span
    sensitivity = normalized_delta / removed
    if not np.isfinite(sensitivity):
        raise ValueError("marginal sensitivity must be finite")
    return float(normalized_delta), float(removed), float(sensitivity)


def trace_single_marginal_path(
    evaluate_loss: Callable[[float], float],
    *,
    schedule: Sequence[float],
    total_layer_route_count: int,
    dense_anchor_loss: float,
    fully_pruned_anchor_loss: float,
    sensitivity_threshold: float,
) -> MarginalFrontierPath:
    """Trace the same marginal controller for an unpartitioned layer."""

    values = tuple(float(value) for value in schedule)
    if not values or abs(values[0] - 1.0) > 1e-12:
        raise ValueError("single-branch schedule must begin at one")
    if any(left <= right for left, right in zip(values, values[1:], strict=False)):
        raise ValueError("single-branch schedule must be strictly decreasing")
    threshold = float(sensitivity_threshold)
    current_density = 1.0
    current_loss = float(evaluate_loss(current_density))
    steps: list[MarginalFrontierStep] = []
    for proposal_index, proposed_density in enumerate(values[1:]):
        proposed_loss = float(evaluate_loss(proposed_density))
        normalized_delta, removed_cost, sensitivity = marginal_step_sensitivity(
            previous_loss=current_loss,
            proposed_loss=proposed_loss,
            dense_anchor_loss=dense_anchor_loss,
            fully_pruned_anchor_loss=fully_pruned_anchor_loss,
            previous_density=current_density,
            proposed_density=proposed_density,
            total_layer_route_count=total_layer_route_count,
        )
        accepted = bool(sensitivity <= threshold)
        steps.append(
            MarginalFrontierStep(
                proposal_index=proposal_index,
                partition_id="all",
                previous_density=current_density,
                proposed_density=proposed_density,
                previous_loss=current_loss,
                proposed_loss=proposed_loss,
                normalized_loss_delta=normalized_delta,
                layer_scaled_removed_cost=removed_cost,
                marginal_sensitivity=sensitivity,
                threshold=threshold,
                accepted=accepted,
                reason="below_or_equal_threshold" if accepted else "threshold_exceeded",
            )
        )
        if not accepted:
            return MarginalFrontierPath(
                positive_density=current_density,
                negative_density=current_density,
                final_loss=current_loss,
                steps=tuple(steps),
                stopping_reason="marginal_threshold_exceeded",
                denominator_policy="requested_density_delta_times_total_layer_route_count",
            )
        current_density = proposed_density
        current_loss = proposed_loss
    return MarginalFrontierPath(
        positive_density=current_density,
        negative_density=current_density,
        final_loss=current_loss,
        steps=tuple(steps),
        stopping_reason="schedule_exhausted",
        denominator_policy="requested_density_delta_times_total_layer_route_count",
    )


def trace_symmetric_marginal_path(
    evaluate_loss: Callable[[float, float], float],
    *,
    positive_schedule: Sequence[float],
    negative_schedule: Sequence[float],
    total_layer_route_count: int,
    dense_anchor_loss: float,
    fully_pruned_anchor_loss: float,
    sensitivity_threshold: float,
    initial_turn: str = "negative",
    maximum_proposals: int = 10000,
) -> MarginalFrontierPath:
    """Trace symmetric full-branch turns until neither partition can advance.

    One turn advances a partition through its remaining schedule until its
    first rejected proposal. Rejected proposals do not advance their cursor
    and may be retried after the opposite partition changes. This reproduces
    the symmetric historical frontier rather than alternating every proposal.
    """

    schedules = {
        "positive": tuple(float(value) for value in positive_schedule),
        "negative": tuple(float(value) for value in negative_schedule),
    }
    for partition_id, schedule in schedules.items():
        if not schedule or abs(schedule[0] - 1.0) > 1e-12:
            raise ValueError(f"{partition_id} schedule must begin at one")
        if any(left <= right for left, right in zip(schedule, schedule[1:], strict=False)):
            raise ValueError(f"{partition_id} schedule must be strictly decreasing")
    if initial_turn not in schedules:
        raise ValueError("initial marginal turn must be positive or negative")
    threshold = float(sensitivity_threshold)
    if not np.isfinite(threshold):
        raise ValueError("marginal sensitivity threshold must be finite")

    cursors = {"positive": 0, "negative": 0}
    densities = {"positive": 1.0, "negative": 1.0}
    current_loss = float(evaluate_loss(1.0, 1.0))
    if not np.isfinite(current_loss):
        raise ValueError("dense marginal-path loss must be finite")
    turn = initial_turn
    stalled_turns = 0
    steps: list[MarginalFrontierStep] = []
    proposal_index = 0
    while proposal_index < int(maximum_proposals):
        schedule = schedules[turn]
        turn_advanced = False
        if cursors[turn] + 1 >= len(schedule):
            step = MarginalFrontierStep(
                proposal_index=proposal_index,
                partition_id=turn,
                previous_density=densities[turn],
                proposed_density=densities[turn],
                previous_loss=current_loss,
                proposed_loss=current_loss,
                normalized_loss_delta=0.0,
                layer_scaled_removed_cost=0.0,
                marginal_sensitivity=math.inf,
                threshold=threshold,
                accepted=False,
                reason="schedule_exhausted",
            )
            steps.append(step)
            proposal_index += 1
        else:
            while cursors[turn] + 1 < len(schedule):
                proposed_density = schedule[cursors[turn] + 1]
                candidate = dict(densities)
                candidate[turn] = proposed_density
                proposed_loss = float(
                    evaluate_loss(candidate["positive"], candidate["negative"])
                )
                normalized_delta, removed_cost, sensitivity = marginal_step_sensitivity(
                    previous_loss=current_loss,
                    proposed_loss=proposed_loss,
                    dense_anchor_loss=dense_anchor_loss,
                    fully_pruned_anchor_loss=fully_pruned_anchor_loss,
                    previous_density=densities[turn],
                    proposed_density=proposed_density,
                    total_layer_route_count=total_layer_route_count,
                )
                accepted = bool(sensitivity <= threshold)
                steps.append(
                    MarginalFrontierStep(
                        proposal_index=proposal_index,
                        partition_id=turn,
                        previous_density=densities[turn],
                        proposed_density=proposed_density,
                        previous_loss=current_loss,
                        proposed_loss=proposed_loss,
                        normalized_loss_delta=normalized_delta,
                        layer_scaled_removed_cost=removed_cost,
                        marginal_sensitivity=sensitivity,
                        threshold=threshold,
                        accepted=accepted,
                        reason=(
                            "below_or_equal_threshold" if accepted else "threshold_exceeded"
                        ),
                    )
                )
                proposal_index += 1
                if not accepted:
                    break
                cursors[turn] += 1
                densities[turn] = proposed_density
                current_loss = proposed_loss
                turn_advanced = True
                if proposal_index >= int(maximum_proposals):
                    break
        stalled_turns = 0 if turn_advanced else stalled_turns + 1
        if stalled_turns >= 2:
            return MarginalFrontierPath(
                positive_density=densities["positive"],
                negative_density=densities["negative"],
                final_loss=current_loss,
                steps=tuple(steps),
                stopping_reason="both_partitions_stalled",
            )
        turn = "negative" if turn == "positive" else "positive"
    raise RuntimeError("symmetric marginal frontier exceeded its proposal guard")


__all__ = [
    "MarginalFrontierPath",
    "MarginalFrontierStep",
    "geometric_retention_schedule",
    "marginal_step_sensitivity",
    "trace_single_marginal_path",
    "trace_symmetric_marginal_path",
]
