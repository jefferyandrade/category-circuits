"""Unified cost-aware Neyman allocation for measurement acquisition.

This module owns only the predictable allocation decision: given frozen
strata, current prefix lengths, allocation-only standard-deviation estimates,
and a count or realized-cost budget, it returns nested cumulative targets.
Modality adapters continue to own what an observation means and when their
scientific stopping rule is resolved.

The discrete priority

    w_h^2 sigma_h^2 / (c_h n_h (n_h + 1))

is the predicted reduction in stratified-mean variance per unit cost from one
additional observation.  Its continuous relaxation is
``n_h proportional to w_h sigma_h / sqrt(c_h)``.  Allocation variances never
narrow downstream confidence bounds.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any, Mapping


UNIFIED_NEYMAN_ALLOCATION_SCHEMA = "unified_neyman_allocation_receipt_v1"
UNIFIED_NEYMAN_POLICY_ID = "unified_cost_aware_neyman_allocator_v1"
BINARY_NEYMAN_POLICY_ID = "binary_cost_aware_neyman_v2"
BINARY_NEYMAN_STRATA = ("on_target", "off_target")


class InsufficientNeymanBudget(ValueError):
    """A hard resource cap cannot supply the frozen minimum support."""


@dataclass(frozen=True, slots=True)
class NeymanStratumSpec:
    """One finite stratum in a frozen acquisition design."""

    stratum_id: str
    capacity: int
    estimand_weight: float
    allocation_standard_deviation: float = 1.0
    unit_cost: float = 1.0
    minimum_count: int = 0

    def __post_init__(self) -> None:
        stratum_id = str(self.stratum_id)
        capacity = int(self.capacity)
        minimum = int(self.minimum_count)
        weight = float(self.estimand_weight)
        standard_deviation = float(self.allocation_standard_deviation)
        unit_cost = float(self.unit_cost)
        if not stratum_id:
            raise ValueError("Neyman stratum requires a stable nonempty ID")
        if capacity < 0 or not 0 <= minimum <= capacity:
            raise ValueError("Neyman stratum support lies outside its capacity")
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("Neyman estimand weight must be finite and nonnegative")
        if weight > 0.0 and capacity == 0:
            raise ValueError("a positive-weight Neyman stratum cannot be empty")
        if (
            not math.isfinite(standard_deviation)
            or standard_deviation < 0.0
        ):
            raise ValueError(
                "Neyman allocation standard deviation must be finite and nonnegative"
            )
        if not math.isfinite(unit_cost) or unit_cost <= 0.0:
            raise ValueError("Neyman unit cost must be finite and positive")
        object.__setattr__(self, "stratum_id", stratum_id)
        object.__setattr__(self, "capacity", capacity)
        object.__setattr__(self, "minimum_count", minimum)
        object.__setattr__(self, "estimand_weight", weight)
        object.__setattr__(
            self,
            "allocation_standard_deviation",
            standard_deviation,
        )
        object.__setattr__(self, "unit_cost", unit_cost)

    @property
    def variance_coefficient(self) -> float:
        """The allocation-only coefficient ``w_h^2 sigma_h^2``."""

        return (
            float(self.estimand_weight) ** 2
            * float(self.allocation_standard_deviation) ** 2
        )

    @property
    def continuous_neyman_weight(self) -> float:
        return (
            float(self.estimand_weight)
            * float(self.allocation_standard_deviation)
            / math.sqrt(float(self.unit_cost))
        )


@dataclass(frozen=True, slots=True)
class NeymanAllocationRequest:
    """A predictable cumulative-target or incremental-cost allocation."""

    strata: tuple[NeymanStratumSpec, ...]
    existing_counts: Mapping[str, int] = field(default_factory=dict)
    cumulative_count_target: int | None = None
    additional_cost_budget: float | None = None
    exact_target_counts: Mapping[str, int] | None = None
    finite_population_correction: bool = True
    require_normalized_estimand_weights: bool = False
    allocation_variance_floor: float = 1e-12
    role: str = "measurement"
    policy_id: str = UNIFIED_NEYMAN_POLICY_ID

    def __post_init__(self) -> None:
        strata = tuple(self.strata)
        if not strata or len({item.stratum_id for item in strata}) != len(strata):
            raise ValueError("Neyman request requires unique nonempty strata")
        modes = sum(
            value is not None
            for value in (
                self.cumulative_count_target,
                self.additional_cost_budget,
                self.exact_target_counts,
            )
        )
        if modes != 1:
            raise ValueError(
                "Neyman request requires exactly one count, cost, or exact-target mode"
            )
        existing = {
            str(key): int(value) for key, value in self.existing_counts.items()
        }
        known = {item.stratum_id for item in strata}
        if set(existing).difference(known):
            raise ValueError("Neyman existing counts contain unknown strata")
        for item in strata:
            count = int(existing.get(item.stratum_id, 0))
            if not 0 <= count <= item.capacity:
                raise ValueError("Neyman existing count lies outside stratum capacity")
        if self.cumulative_count_target is not None and int(
            self.cumulative_count_target
        ) < 0:
            raise ValueError("Neyman cumulative count target must be nonnegative")
        if self.additional_cost_budget is not None and (
            not math.isfinite(float(self.additional_cost_budget))
            or float(self.additional_cost_budget) < 0.0
        ):
            raise ValueError("Neyman additional cost budget must be nonnegative")
        if self.exact_target_counts is not None:
            exact = {
                str(key): int(value)
                for key, value in self.exact_target_counts.items()
            }
            if set(exact) != known:
                raise ValueError("exact Neyman targets must cover every stratum")
            object.__setattr__(self, "exact_target_counts", exact)
        floor = float(self.allocation_variance_floor)
        if not math.isfinite(floor) or floor < 0.0:
            raise ValueError("Neyman allocation variance floor must be nonnegative")
        if not str(self.role) or not str(self.policy_id):
            raise ValueError("Neyman request requires role and policy identities")
        weight_sum = sum(item.estimand_weight for item in strata)
        if weight_sum <= 0.0:
            raise ValueError("Neyman request requires positive aggregate weight")
        if self.require_normalized_estimand_weights and not math.isclose(
            weight_sum,
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("claim-estimand Neyman weights must sum to one")
        object.__setattr__(self, "strata", strata)
        object.__setattr__(self, "existing_counts", existing)
        object.__setattr__(self, "allocation_variance_floor", floor)
        object.__setattr__(self, "role", str(self.role))
        object.__setattr__(self, "policy_id", str(self.policy_id))


@dataclass(frozen=True, slots=True)
class NeymanAllocationResult:
    """Nested cumulative targets plus a serializable allocation receipt."""

    target_counts: Mapping[str, int]
    additions: Mapping[str, int]
    expected_variance: float
    incremental_cost: float
    unused_cost_budget: float
    requested_cumulative_count: int | None
    delivered_cumulative_count: int
    population_count: int
    full_census: bool
    finite_population_correction: bool
    trace: tuple[Mapping[str, Any], ...]
    role: str
    policy_id: str
    schema: str = UNIFIED_NEYMAN_ALLOCATION_SCHEMA

    def to_receipt(self) -> dict[str, Any]:
        variance = float(self.expected_variance)
        return {
            "schema": self.schema,
            "policy_id": self.policy_id,
            "role": self.role,
            "target_counts": {
                str(key): int(value) for key, value in self.target_counts.items()
            },
            "additions": {
                str(key): int(value) for key, value in self.additions.items()
            },
            "expected_variance": variance if math.isfinite(variance) else None,
            "expected_variance_unbounded": not math.isfinite(variance),
            "incremental_cost": float(self.incremental_cost),
            "unused_cost_budget": float(self.unused_cost_budget),
            "requested_cumulative_count": self.requested_cumulative_count,
            "delivered_cumulative_count": int(self.delivered_cumulative_count),
            "population_count": int(self.population_count),
            "full_census": bool(self.full_census),
            "finite_population_correction": bool(
                self.finite_population_correction
            ),
            "allocation_trace": tuple(dict(item) for item in self.trace),
            "allocation_only_variance": True,
            "continuous_optimum": "n_h proportional to w_h*sigma_h/sqrt(c_h)",
        }


class UnifiedNeymanAllocator:
    """Deterministic integer allocator shared by all measurement modalities."""

    @staticmethod
    def _priority(
        spec: NeymanStratumSpec,
        count: int,
        variance_floor: float,
    ) -> float:
        coefficient = max(
            float(spec.variance_coefficient),
            float(spec.estimand_weight) ** 2 * float(variance_floor),
        )
        if coefficient <= 0.0:
            return 0.0
        if int(count) <= 0:
            return math.inf
        return coefficient / (
            float(spec.unit_cost) * float(count) * float(count + 1)
        )

    @staticmethod
    def _expected_variance(
        strata: tuple[NeymanStratumSpec, ...],
        counts: Mapping[str, int],
        *,
        finite_population_correction: bool,
    ) -> float:
        total = 0.0
        for spec in strata:
            coefficient = float(spec.variance_coefficient)
            if coefficient <= 0.0:
                continue
            count = int(counts[spec.stratum_id])
            if count <= 0:
                return math.inf
            component = coefficient / float(count)
            if finite_population_correction:
                component = max(
                    0.0,
                    component - coefficient / float(spec.capacity),
                )
            total += component
        return float(total)

    @staticmethod
    def _heap(
        strata: tuple[NeymanStratumSpec, ...],
        counts: Mapping[str, int],
        *,
        variance_floor: float,
    ) -> list[tuple[float, int, str]]:
        heap: list[tuple[float, int, str]] = []
        for order, spec in enumerate(strata):
            if int(counts[spec.stratum_id]) >= spec.capacity:
                continue
            priority = UnifiedNeymanAllocator._priority(
                spec,
                int(counts[spec.stratum_id]),
                variance_floor,
            )
            heapq.heappush(heap, (-priority, order, spec.stratum_id))
        return heap

    def marginal_priorities(
        self,
        strata: tuple[NeymanStratumSpec, ...],
        counts: Mapping[str, int],
        *,
        allocation_variance_floor: float = 1e-12,
    ) -> dict[str, float]:
        """Return the next-observation variance reduction per unit cost."""

        specs = tuple(strata)
        if not specs or len({item.stratum_id for item in specs}) != len(specs):
            raise ValueError("Neyman priorities require unique nonempty strata")
        floor = float(allocation_variance_floor)
        if not math.isfinite(floor) or floor < 0.0:
            raise ValueError("Neyman allocation variance floor must be nonnegative")
        unknown = set(map(str, counts)).difference(
            item.stratum_id for item in specs
        )
        if unknown:
            raise ValueError("Neyman priority counts contain unknown strata")
        output: dict[str, float] = {}
        for spec in specs:
            count = int(counts.get(spec.stratum_id, 0))
            if not 0 <= count <= spec.capacity:
                raise ValueError("Neyman priority count lies outside capacity")
            if count < spec.capacity:
                output[spec.stratum_id] = self._priority(spec, count, floor)
        return output

    def allocate(self, request: NeymanAllocationRequest) -> NeymanAllocationResult:
        strata = request.strata
        by_id = {item.stratum_id: item for item in strata}
        existing = {
            item.stratum_id: int(request.existing_counts.get(item.stratum_id, 0))
            for item in strata
        }
        required = {
            item.stratum_id: max(existing[item.stratum_id], item.minimum_count)
            for item in strata
        }
        counts = dict(existing)
        trace: list[Mapping[str, Any]] = []
        minimum_additions = {
            item.stratum_id: required[item.stratum_id] - existing[item.stratum_id]
            for item in strata
        }
        mandatory_cost = sum(
            minimum_additions[item.stratum_id] * item.unit_cost for item in strata
        )

        requested_count: int | None = None
        unused_cost = 0.0
        if request.exact_target_counts is not None:
            targets = dict(request.exact_target_counts)
            for item in strata:
                target = int(targets[item.stratum_id])
                if (
                    target < required[item.stratum_id]
                    or target > item.capacity
                ):
                    raise ValueError(
                        "exact Neyman target violates support, nesting, or capacity"
                    )
            counts = targets
            trace.append(
                {
                    "action": "honor_exact_targets",
                    "target_counts": dict(counts),
                }
            )
        else:
            if request.cumulative_count_target is not None:
                requested_count = int(request.cumulative_count_target)
                if requested_count < sum(existing.values()):
                    raise ValueError("cumulative Neyman target cannot shrink")
                reachable = min(
                    requested_count,
                    sum(item.capacity for item in strata),
                )
                if reachable < sum(required.values()):
                    raise InsufficientNeymanBudget(
                        "cumulative Neyman target cannot supply frozen support minima"
                    )
            else:
                budget = float(request.additional_cost_budget or 0.0)
                tolerance = 1e-12 * max(1.0, budget, mandatory_cost)
                if mandatory_cost > budget + tolerance:
                    raise InsufficientNeymanBudget(
                        "Neyman cost cap cannot supply frozen support minima"
                    )
                reachable = None
                unused_cost = max(0.0, budget - mandatory_cost)

            counts = dict(required)
            for item in strata:
                added = minimum_additions[item.stratum_id]
                if added:
                    trace.append(
                        {
                            "action": "minimum_stratum_support",
                            "stratum": item.stratum_id,
                            "add_count": int(added),
                            "target_count": int(counts[item.stratum_id]),
                            "realized_cost": float(added * item.unit_cost),
                        }
                    )

            optimized = {item.stratum_id: 0 for item in strata}
            last_priority = {item.stratum_id: 0.0 for item in strata}
            heap = self._heap(
                strata,
                counts,
                variance_floor=request.allocation_variance_floor,
            )
            if reachable is not None:
                # A running total: re-summing all strata per added row made large terminal closures
                # O(rows x strata) on the host.
                total = sum(counts.values())
                while total < reachable and heap:
                    neg_priority, order, stratum_id = heapq.heappop(heap)
                    spec = by_id[stratum_id]
                    counts[stratum_id] += 1
                    total += 1
                    optimized[stratum_id] += 1
                    last_priority[stratum_id] = float(-neg_priority)
                    if counts[stratum_id] < spec.capacity:
                        heapq.heappush(
                            heap,
                            (
                                -self._priority(
                                    spec,
                                    counts[stratum_id],
                                    request.allocation_variance_floor,
                                ),
                                order,
                                stratum_id,
                            ),
                        )
            else:
                tolerance = 1e-12 * max(
                    1.0,
                    float(request.additional_cost_budget or 0.0),
                )
                deferred: list[tuple[float, int, str]] = []
                while heap:
                    neg_priority, order, stratum_id = heapq.heappop(heap)
                    spec = by_id[stratum_id]
                    if spec.unit_cost > unused_cost + tolerance:
                        deferred.append((neg_priority, order, stratum_id))
                        continue
                    counts[stratum_id] += 1
                    optimized[stratum_id] += 1
                    last_priority[stratum_id] = float(-neg_priority)
                    unused_cost = max(0.0, unused_cost - spec.unit_cost)
                    for item in deferred:
                        heapq.heappush(heap, item)
                    deferred.clear()
                    if counts[stratum_id] < spec.capacity:
                        heapq.heappush(
                            heap,
                            (
                                -self._priority(
                                    spec,
                                    counts[stratum_id],
                                    request.allocation_variance_floor,
                                ),
                                order,
                                stratum_id,
                            ),
                        )
                # ``deferred`` strata cannot fit now and cannot fit after the
                # remaining budget decreases, so the residual is intentional.

            for item in strata:
                added = optimized[item.stratum_id]
                if added:
                    trace.append(
                        {
                            "action": "cost_aware_neyman_add",
                            "stratum": item.stratum_id,
                            "add_count": int(added),
                            "target_count": int(counts[item.stratum_id]),
                            "last_predicted_variance_reduction_per_cost": (
                                float(last_priority[item.stratum_id])
                                if math.isfinite(
                                    last_priority[item.stratum_id]
                                )
                                else "unobserved_stratum_infinite_priority"
                            ),
                            "continuous_neyman_weight": float(
                                item.continuous_neyman_weight
                            ),
                        }
                    )

        additions = {
            item.stratum_id: counts[item.stratum_id] - existing[item.stratum_id]
            for item in strata
        }
        incremental_cost = sum(
            additions[item.stratum_id] * item.unit_cost for item in strata
        )
        delivered = sum(counts.values())
        population = sum(item.capacity for item in strata)
        return NeymanAllocationResult(
            target_counts=dict(counts),
            additions=additions,
            expected_variance=self._expected_variance(
                strata,
                counts,
                finite_population_correction=request.finite_population_correction,
            ),
            incremental_cost=float(incremental_cost),
            unused_cost_budget=float(unused_cost),
            requested_cumulative_count=requested_count,
            delivered_cumulative_count=int(delivered),
            population_count=int(population),
            full_census=all(
                counts[item.stratum_id] == item.capacity for item in strata
            ),
            finite_population_correction=bool(
                request.finite_population_correction
            ),
            trace=tuple(trace),
            role=request.role,
            policy_id=request.policy_id,
        )


def binary_neyman_request(
    *,
    total_count: int | None = None,
    additional_cost_budget: float | None = None,
    available_by_stratum: Mapping[str, int],
    minimum_by_stratum: Mapping[str, int],
    existing_by_stratum: Mapping[str, int] | None = None,
    sd_by_stratum: Mapping[str, float] | None = None,
    weight_by_stratum: Mapping[str, float] | None = None,
    cost_by_stratum: Mapping[str, float] | None = None,
    finite_population_correction: bool = True,
    role: str = "binary_measurement",
) -> NeymanAllocationRequest:
    """Build the canonical on-target/pooled-off-target request.

    Exactly one of ``total_count`` (cumulative target) or
    ``additional_cost_budget`` (incremental spend) must be given.

    **The incremental mode was unreachable until this signature carried it.**
    ``NeymanAllocationRequest`` has always documented itself as *"a predictable
    cumulative-target OR incremental-cost allocation"*, and ``allocate()``
    implements both branches -- but this builder is the only one the census uses
    and it hardcoded ``cumulative_count_target``. MEASURED: zero call sites for
    ``additional_cost_budget`` anywhere outside ``neyman_allocation.py``. The
    streaming mode was working code that nothing could ask for, which is why the
    census fell back on a doubling schedule and walked its panel into a 337 GiB
    OOM.
    """

    if (total_count is None) == (additional_cost_budget is None):
        raise ValueError(
            "binary Neyman request needs exactly one of total_count "
            "(cumulative target) or additional_cost_budget (incremental spend)"
        )

    existing = dict(existing_by_stratum or {})
    sds = dict(sd_by_stratum or {})
    weights = dict(weight_by_stratum or {})
    costs = dict(cost_by_stratum or {})
    return NeymanAllocationRequest(
        strata=tuple(
            NeymanStratumSpec(
                stratum_id=key,
                capacity=int(available_by_stratum.get(key, -1)),
                minimum_count=int(minimum_by_stratum.get(key, -1)),
                estimand_weight=float(weights.get(key, 0.5)),
                allocation_standard_deviation=float(sds.get(key, 1.0)),
                unit_cost=float(costs.get(key, 1.0)),
            )
            for key in BINARY_NEYMAN_STRATA
        ),
        existing_counts={
            key: int(existing.get(key, 0)) for key in BINARY_NEYMAN_STRATA
        },
        cumulative_count_target=(
            None if total_count is None else int(total_count)
        ),
        additional_cost_budget=(
            None
            if additional_cost_budget is None
            else float(additional_cost_budget)
        ),
        finite_population_correction=bool(finite_population_correction),
        require_normalized_estimand_weights=True,
        role=role,
        policy_id=BINARY_NEYMAN_POLICY_ID,
    )


__all__ = [
    "BINARY_NEYMAN_POLICY_ID",
    "BINARY_NEYMAN_STRATA",
    "InsufficientNeymanBudget",
    "NeymanAllocationRequest",
    "NeymanAllocationResult",
    "NeymanStratumSpec",
    "UNIFIED_NEYMAN_ALLOCATION_SCHEMA",
    "UNIFIED_NEYMAN_POLICY_ID",
    "UnifiedNeymanAllocator",
    "binary_neyman_request",
]
