"""The oracle's per-stratum bookkeeping as arrays over the fixed stratum axis.

The class-stratified oracle has one stratum per class (1,000 on ImageNet).
Rather than a dictionary per quantity per look (existing counts, remaining
rows, capacities, minima, SDs, the census disclosure) and one allocator spec
object per stratum, the stratum axis is frozen once per provider as
arrays in declaration order; capacities, minima, targets, selection codes and
the census disclosure are array operations over all strata at once, and the
cost-aware Neyman law runs as the tensor allocator ``device_bce.neyman_counts``
on the device that holds its inputs (the host moment arrays; device
moments unchanged).

A first look has zero counts, no moments (every allocation SD is the pooled
default 1.0), the fixed populations (or, with replacement, the request) as
capacities, and the fixed weights, costs and minima.  Its plan is therefore one
constant per run and packet size: it is computed once and reused by every
decision.

Same algorithm as ``neyman_count_target_allocation``: the same marginal
priority ``w_h^2 max(sd_h^2, floor) / (c_h n (n + 1))``, the same starting
support ``max(existing, minimum)``, the same declaration-order tie rule.  Only
float association differs (the priority's division order, and the pooled SD's
parallel rather than sequential moment merge): a last-ulp effect at a near tie
between strata with different inputs.  No near-tie re-resolution (a
bitwise-parity device) is carried.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch

from mechanistic_systems.inference.device_allocation import allocation_sds as _allocation_sds
from mechanistic_systems.inference.device_bce import neyman_counts
from mechanistic_systems.measurement.neyman_allocation import (
    UNIFIED_NEYMAN_ALLOCATION_SCHEMA,
)

# ``NeymanAllocationRequest.allocation_variance_floor``'s default, which the
# oracle request never overrides.
ALLOCATION_VARIANCE_FLOOR = 1e-12
_CONTINUOUS_OPTIMUM = "n_h proportional to w_h*sigma_h/sqrt(c_h)"


@dataclass(frozen=True, eq=False)
class OracleStratumAxis:
    """The oracle's stratum axis, frozen as arrays in declaration order."""

    names: tuple[str, ...]
    populations: np.ndarray
    weights: np.ndarray
    costs: np.ndarray
    minimum: int
    population_by_stratum: Mapping[str, int]
    source_population_counts: Mapping[str, int]
    sources: tuple[Any, ...]
    # (packet total, law, policy) -> LookPlan of a first look
    first_looks: dict = field(default_factory=dict)

    @classmethod
    def from_provider(cls, provider: Any) -> "OracleStratumAxis":
        names = tuple(provider._oracle_statistical_strata)
        populations = provider._oracle_statistical_populations
        counts = np.fromiter(
            (len(populations[name]) for name in names), dtype=np.int64, count=len(names)
        )
        counts.setflags(write=False)
        weights = np.fromiter(
            (float(provider._oracle_statistical_weights[name]) for name in names),
            dtype=np.float64, count=len(names),
        )
        weights.setflags(write=False)
        costs = np.fromiter(
            (float(provider._oracle_stratum_costs[name]) for name in names),
            dtype=np.float64, count=len(names),
        )
        costs.setflags(write=False)
        return cls(
            names=names,
            populations=counts,
            weights=weights,
            costs=costs,
            minimum=int(provider.oracle_minimum_per_stratum),
            population_by_stratum=dict(zip(names, counts.tolist())),
            source_population_counts={
                key: len(values) for key, values in populations.items()
            },
            sources=cls._sources(provider),
        )

    @staticmethod
    def _sources(provider: Any) -> tuple[Any, ...]:
        return (
            provider._oracle_statistical_strata,
            provider._oracle_statistical_populations,
            provider._oracle_statistical_weights,
            provider._oracle_stratum_costs,
            int(provider.oracle_minimum_per_stratum),
        )

    def matches(self, provider: Any) -> bool:
        """True while the provider still holds the objects this axis froze."""

        current = self._sources(provider)
        return all(a is b for a, b in zip(self.sources[:4], current[:4])) and (
            self.sources[4] == current[4]
        )

    @property
    def size(self) -> int:
        return len(self.names)

    def existing_counts(self, history: Mapping[str, Any] | None) -> np.ndarray:
        """Drawn rows per stratum (every reader of the history uses ``len``)."""

        if not history:
            return np.zeros(self.size, dtype=np.int64)
        return np.fromiter(
            (len(history.get(name, ())) for name in self.names),
            dtype=np.int64, count=self.size,
        )


@dataclass(frozen=True, eq=False)
class LookPlan:
    """One look's allocation: targets, draw ranges, selection codes, receipts.

    A first look's plan is shared by every decision of the run, so its arrays
    are read-only and its receipts are read-only by contract (consumers copy
    the top level; nothing writes into a receipt).
    """

    targets: np.ndarray
    ranges: tuple[tuple[str, int, int], ...]
    codes: np.ndarray
    allocation_receipt: Mapping[str, Any]
    census_contract: Mapping[str, Any] | None


def allocation_sds(counts: Any, means: Any, m2: Any) -> torch.Tensor:
    """``_oracle_allocation_sds`` over moment arrays, all strata at once.

    ``counts`` is ``[S]``, ``means`` and ``m2`` are ``[S, C]`` (numpy or
    tensors; the result stays on their device).  A stratum with more than one
    row uses its own SD (maximized over columns), every other stratum the
    pooled SD, and the pooled SD defaults to 1.0 when the pooled count is at
    most one.  The pooled moment is the parallel (sum) form of a
    sequential Chan merge: the same moment, reassociated.
    """

    count = torch.as_tensor(counts, dtype=torch.float64)
    device = count.device
    S = int(count.shape[0])
    mean = torch.as_tensor(means, dtype=torch.float64, device=device).reshape(S, -1).T
    second = torch.as_tensor(m2, dtype=torch.float64, device=device).reshape(S, -1).T
    return _allocation_sds(count, mean, second.clamp_min(0.0))


def look_targets(
    *,
    total: int,
    existing: np.ndarray,
    capacities: np.ndarray,
    minima: np.ndarray,
    weights: np.ndarray,
    costs: np.ndarray,
    sds: Any,
) -> np.ndarray:
    """Cumulative per-stratum Neyman targets of one look, as host int64.

    Runs ``neyman_counts`` on the device of ``sds`` (a tensor) or on the CPU
    (an array); the integer targets are the one value the host needs back,
    because the draws are host streams.
    """

    sd = torch.as_tensor(sds, dtype=torch.float64)
    device = sd.device

    def tensor(values: Any) -> torch.Tensor:
        return torch.from_numpy(np.array(values, dtype=np.float64)).to(device)

    targets = neyman_counts(
        torch.tensor(float(total), dtype=torch.float64, device=device),
        tensor(existing), tensor(minima), tensor(capacities), tensor(weights), sd, tensor(costs),
    )
    return targets.cpu().numpy().astype(np.int64, copy=False)


def allocation_receipt(
    axis: OracleStratumAxis,
    *,
    total: int,
    existing: np.ndarray,
    targets: np.ndarray,
    capacities: np.ndarray,
    minima: np.ndarray,
    sds: np.ndarray,
    finite_population_correction: bool,
    role: str,
    policy_id: str,
) -> dict[str, Any]:
    """``NeymanAllocationResult.to_receipt()`` of the same allocation, from arrays.

    The fields, their order and their scalar formulas are those of the
    vectorized greedy (``evidence_plane_v6.neyman_count_target_allocation``);
    per-element float arithmetic is the same IEEE operation in the same order,
    and every sum is the same sequential sum in declaration order.
    """

    names = axis.names
    weights, costs = axis.weights, axis.costs
    existing = np.asarray(existing, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    sds = np.asarray(sds, dtype=np.float64)
    required = np.maximum(existing, np.asarray(minima, dtype=np.int64))
    minimum_added = required - existing
    optimized = targets - required
    coefficient = weights**2 * sds**2
    trace: list[dict[str, Any]] = []
    for index in np.flatnonzero(minimum_added).tolist():
        added = int(minimum_added[index])
        trace.append(
            {
                "action": "minimum_stratum_support",
                "stratum": names[index],
                "add_count": added,
                "target_count": int(required[index]),
                "realized_cost": float(added * float(costs[index])),
            }
        )
    added_index = np.flatnonzero(optimized > 0)
    if added_index.size:
        floored = np.maximum(coefficient, weights**2 * ALLOCATION_VARIANCE_FLOOR)[added_index]
        last = (targets[added_index] - 1).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            priority = floored / (costs[added_index] * last * (last + 1.0))
        priority = np.where(last <= 0.0, math.inf, priority)
        priority = np.where(floored <= 0.0, 0.0, priority)
        continuous = weights[added_index] * sds[added_index] / np.sqrt(costs[added_index])
        for index, value, weight in zip(
            added_index.tolist(), priority.tolist(), continuous.tolist()
        ):
            trace.append(
                {
                    "action": "cost_aware_neyman_add",
                    "stratum": names[index],
                    "add_count": int(optimized[index]),
                    "target_count": int(targets[index]),
                    "last_predicted_variance_reduction_per_cost": (
                        value if math.isfinite(value) else "unobserved_stratum_infinite_priority"
                    ),
                    "continuous_neyman_weight": float(weight),
                }
            )
    positive = coefficient > 0.0
    if bool(np.any(positive & (targets <= 0))):
        variance = math.inf
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            component = coefficient / targets.astype(np.float64)
            if finite_population_correction:
                component = np.maximum(0.0, component - coefficient / capacities.astype(np.float64))
        variance = 0.0
        for value in component[positive].tolist():  # sequential running total
            variance += value
    additions = targets - existing
    target_list, addition_list = targets.tolist(), additions.tolist()
    return {
        "schema": UNIFIED_NEYMAN_ALLOCATION_SCHEMA,
        "policy_id": str(policy_id),
        "role": str(role),
        "target_counts": dict(zip(names, target_list)),
        "additions": dict(zip(names, addition_list)),
        "expected_variance": variance if math.isfinite(variance) else None,
        "expected_variance_unbounded": not math.isfinite(variance),
        "incremental_cost": float(sum((additions * costs).tolist())),
        "unused_cost_budget": 0.0,
        "requested_cumulative_count": int(total),
        "delivered_cumulative_count": int(sum(target_list)),
        "population_count": int(sum(capacities.tolist())),
        "full_census": bool(np.array_equal(targets, capacities)),
        "finite_population_correction": bool(finite_population_correction),
        "allocation_trace": tuple(trace),
        "allocation_only_variance": True,
        "continuous_optimum": _CONTINUOUS_OPTIMUM,
    }


def selection_codes(existing: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Stratum ordinal of every addition, in the order the draws are made."""

    additions = np.asarray(targets, dtype=np.int64) - np.asarray(existing, dtype=np.int64)
    return np.repeat(np.arange(additions.size, dtype=np.int64), additions)


def draw_triples(
    axis: OracleStratumAxis, existing: np.ndarray, targets: np.ndarray
) -> tuple[tuple[str, int, int], ...]:
    """``(stratum, start, stop)`` for every stratum, declaration order."""

    return tuple(zip(axis.names, np.asarray(existing).tolist(), np.asarray(targets).tolist()))


__all__ = [
    "ALLOCATION_VARIANCE_FLOOR",
    "LookPlan",
    "OracleStratumAxis",
    "allocation_receipt",
    "allocation_sds",
    "draw_triples",
    "look_targets",
    "selection_codes",
]
