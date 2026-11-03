"""IRG circuit costs and compatibility exports for measurement costs.

The scientific objective and the hardware accounting objective are deliberately
separate.  ``CostVector`` describes the deployable circuit.  ``CostEstimate``
and ``RealizedCost`` are the exact capability-neutral classes re-exported from
``mechanistic_systems.contracts.measurement_costs``.  This facade grants them
no IRG workflow authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mechanistic_systems.contracts.circuit_cost import CostVector
from mechanistic_systems.contracts.measurement_costs import (
    CostEstimate,
    RealizedCost,
    _finite_nonnegative,
)


@dataclass
class ComputeLedger:
    """Mutable run-local accumulator with an optional hard wall-time budget."""

    budget_wall_seconds: float | None = None
    outcomes: list[RealizedCost] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.budget_wall_seconds is not None:
            _finite_nonnegative(self.budget_wall_seconds, name="budget_wall_seconds")

    @property
    def wall_seconds(self) -> float:
        return float(sum(item.wall_seconds for item in self.outcomes))

    @property
    def device_seconds(self) -> float:
        return float(sum(item.device_seconds for item in self.outcomes))

    @property
    def forward_passes(self) -> int:
        return int(sum(item.forward_passes for item in self.outcomes))

    @property
    def backward_passes(self) -> int:
        return int(sum(item.backward_passes for item in self.outcomes))

    @property
    def remaining_wall_seconds(self) -> float | None:
        if self.budget_wall_seconds is None:
            return None
        return max(0.0, float(self.budget_wall_seconds) - self.wall_seconds)

    def can_afford(self, estimate: CostEstimate) -> bool:
        remaining = self.remaining_wall_seconds
        return remaining is None or estimate.wall_seconds <= remaining + 1e-12

    def record(self, cost: RealizedCost) -> None:
        if not self.can_afford(cost.to_estimate()):
            raise RuntimeError("realized measurement cost exceeded the remaining hard budget")
        self.outcomes.append(cost)

    def summary(self) -> dict[str, object]:
        return {
            "budget_wall_seconds": self.budget_wall_seconds,
            "wall_seconds": self.wall_seconds,
            "device_seconds": self.device_seconds,
            "forward_passes": self.forward_passes,
            "backward_passes": self.backward_passes,
            "measurement_count": len(self.outcomes),
        }


__all__ = [
    "ComputeLedger",
    "CostEstimate",
    "CostVector",
    "RealizedCost",
]
