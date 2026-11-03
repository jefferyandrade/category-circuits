"""Canonical rate-distortion semantics for classic circuit identification."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from circuitflow.contracts import RuntimePlan
from circuitflow.engines import ClassicFrontierEngine, EngineCompilation
from mechanistic_systems.contracts import CircuitPlan


@dataclass(frozen=True)
class CircuitCandidate:
    realization_id: str
    response_distortion: float
    route_cost: float
    target_preservation: float
    retained_route_count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        values = (self.response_distortion, self.route_cost, self.target_preservation)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("circuit candidate metrics must be finite")
        if self.response_distortion < 0.0 or self.route_cost < 0.0 or self.retained_route_count < 0:
            raise ValueError("circuit distortion, cost, and route count must be nonnegative")


def nested_rate_distortion_frontier(candidates: Sequence[CircuitCandidate]) -> tuple[CircuitCandidate, ...]:
    """Return the exact nondominated frontier in increasing route cost.

    This function is search-algorithm independent.  Existing greedy pruning
    workers may generate the candidates, but only this declared objective
    decides which realizations belong to the scientific frontier.
    """

    if not candidates:
        raise ValueError("rate-distortion frontier requires candidates")
    for candidate in candidates:
        candidate.validate()
    ordered = sorted(
        candidates,
        key=lambda item: (
            float(item.route_cost),
            float(item.response_distortion),
            -float(item.target_preservation),
            item.realization_id,
        ),
    )
    frontier: list[CircuitCandidate] = []
    best_distortion = float("inf")
    best_preservation = float("-inf")
    for candidate in ordered:
        improves_distortion = candidate.response_distortion < best_distortion - 1e-12
        ties_and_improves_preservation = (
            abs(candidate.response_distortion - best_distortion) <= 1e-12
            and candidate.target_preservation > best_preservation + 1e-12
        )
        if improves_distortion or ties_and_improves_preservation:
            frontier.append(candidate)
            best_distortion = min(best_distortion, candidate.response_distortion)
            best_preservation = max(best_preservation, candidate.target_preservation)
    return tuple(frontier)


def select_circuit(
    candidates: Sequence[CircuitCandidate],
    *,
    maximum_cost: float,
    maximum_distortion: float | None = None,
    minimum_preservation: float | None = None,
) -> CircuitCandidate:
    frontier = nested_rate_distortion_frontier(candidates)
    feasible = [candidate for candidate in frontier if candidate.route_cost <= maximum_cost]
    if maximum_distortion is not None:
        feasible = [candidate for candidate in feasible if candidate.response_distortion <= maximum_distortion]
    if minimum_preservation is not None:
        feasible = [candidate for candidate in feasible if candidate.target_preservation >= minimum_preservation]
    if not feasible:
        raise ValueError("no circuit realization satisfies the frozen rate-distortion constraints")
    if maximum_distortion is not None or minimum_preservation is not None:
        return min(
            feasible,
            key=lambda item: (
                item.route_cost,
                item.response_distortion,
                -item.target_preservation,
                item.realization_id,
            ),
        )
    return min(
        feasible,
        key=lambda item: (
            item.response_distortion,
            item.route_cost,
            -item.target_preservation,
            item.realization_id,
        ),
    )


@dataclass(frozen=True)
class PartitionAssignment:
    """A route partition plus the nuisance strata it must preserve."""

    policy_id: str
    route_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    layer_ids: tuple[str, ...]
    magnitude_bins: tuple[str, ...]
    degree_bins: tuple[str, ...]

    def __post_init__(self) -> None:
        lengths = {
            len(self.route_ids),
            len(self.group_ids),
            len(self.layer_ids),
            len(self.magnitude_bins),
            len(self.degree_bins),
        }
        if len(lengths) != 1:
            raise ValueError("partition coordinates must have identical lengths")
        if len(set(self.route_ids)) != len(self.route_ids):
            raise ValueError("partition route IDs must be unique")

    def nuisance_counts(self) -> Counter[tuple[str, str, str, str]]:
        return Counter(
            zip(
                self.group_ids,
                self.layer_ids,
                self.magnitude_bins,
                self.degree_bins,
                strict=True,
            )
        )


def validate_matched_partition_family(
    reference: PartitionAssignment,
    alternatives: Sequence[PartitionAssignment],
) -> None:
    """Require every null to preserve route and nuisance support exactly."""

    route_axis = reference.route_ids
    expected_counts = reference.nuisance_counts()
    for alternative in alternatives:
        if alternative.route_ids != route_axis:
            raise ValueError(f"partition {alternative.policy_id!r} changes the route universe or order")
        if alternative.nuisance_counts() != expected_counts:
            raise ValueError(f"partition {alternative.policy_id!r} is not nuisance matched")


class ClassicCircuitEngine:
    """Scientific adapter over the tested classic frontier worker boundary."""

    engine_id = "classic_circuit_identification"

    def compile(
        self,
        plan: CircuitPlan,
        runtime: RuntimePlan,
        *,
        work_units: Sequence[str],
    ) -> EngineCompilation:
        plan.validate(claim_bearing=plan.status == "frozen").require_valid()
        return ClassicFrontierEngine().compile(
            plan.to_operational(),
            runtime,
            work_units=tuple(str(unit) for unit in work_units),
        )


__all__ = [
    "CircuitCandidate",
    "ClassicCircuitEngine",
    "PartitionAssignment",
    "nested_rate_distortion_frontier",
    "select_circuit",
    "validate_matched_partition_family",
]
