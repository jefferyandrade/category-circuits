"""Capability-neutral deployable-circuit cost contract.

The vector describes physical resource coordinates of a candidate circuit. It
does not select a pruning action, authorize an intervention, or confer workflow
or scientific-claim authority.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

from .measurement_costs import _finite_nonnegative


@dataclass(frozen=True)
class CostVector:
    """Multi-objective deployable-circuit cost."""

    route_count: float = 0.0
    parameter_count: float = 0.0
    flop_count: float = 0.0
    memory_bytes: float = 0.0
    latency_ms: float = 0.0
    custom: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "route_count",
            "parameter_count",
            "flop_count",
            "memory_bytes",
            "latency_ms",
        ):
            _finite_nonnegative(getattr(self, name), name=name)
        normalized: dict[str, float] = {}
        for key, value in sorted(dict(self.custom).items()):
            label = str(key)
            if not label:
                raise ValueError("custom cost coordinates require nonempty names")
            normalized[label] = _finite_nonnegative(value, name=f"custom[{label!r}]")
        object.__setattr__(self, "custom", normalized)

    @classmethod
    def zero(cls) -> "CostVector":
        return cls()

    def coordinate(self, name: str) -> float:
        if name in {
            "route_count",
            "parameter_count",
            "flop_count",
            "memory_bytes",
            "latency_ms",
        }:
            return float(getattr(self, name))
        if name in self.custom:
            return float(self.custom[name])
        raise KeyError(f"unknown cost coordinate {name!r}")

    def scalarize(self, weights: Mapping[str, float]) -> float:
        if not weights:
            raise ValueError("cost scalarization requires at least one weight")
        total = 0.0
        for coordinate, weight in weights.items():
            coefficient = float(weight)
            if not math.isfinite(coefficient) or coefficient < 0.0:
                raise ValueError(
                    "cost scalarization weights must be finite and nonnegative"
                )
            total += coefficient * self.coordinate(str(coordinate))
        return float(total)

    def __add__(self, other: "CostVector") -> "CostVector":
        keys = sorted(set(self.custom) | set(other.custom))
        return CostVector(
            route_count=self.route_count + other.route_count,
            parameter_count=self.parameter_count + other.parameter_count,
            flop_count=self.flop_count + other.flop_count,
            memory_bytes=self.memory_bytes + other.memory_bytes,
            latency_ms=self.latency_ms + other.latency_ms,
            custom={
                key: self.custom.get(key, 0.0) + other.custom.get(key, 0.0)
                for key in keys
            },
        )

    def __sub__(self, other: "CostVector") -> "CostVector":
        values = {
            "route_count": self.route_count - other.route_count,
            "parameter_count": self.parameter_count - other.parameter_count,
            "flop_count": self.flop_count - other.flop_count,
            "memory_bytes": self.memory_bytes - other.memory_bytes,
            "latency_ms": self.latency_ms - other.latency_ms,
        }
        if any(value < -1e-12 for value in values.values()):
            raise ValueError(
                "circuit cost subtraction would create a negative coordinate"
            )
        keys = sorted(set(self.custom) | set(other.custom))
        custom = {
            key: self.custom.get(key, 0.0) - other.custom.get(key, 0.0) for key in keys
        }
        if any(value < -1e-12 for value in custom.values()):
            raise ValueError(
                "custom circuit cost subtraction would create a negative coordinate"
            )
        return CostVector(
            **{key: max(0.0, value) for key, value in values.items()},
            custom={key: max(0.0, value) for key, value in custom.items()},
        )

    def dominates(self, other: "CostVector", *, tolerance: float = 1e-12) -> bool:
        """Return whether this cost is no larger in every shared coordinate."""

        coordinates = (
            "route_count",
            "parameter_count",
            "flop_count",
            "memory_bytes",
            "latency_ms",
        )
        if any(
            self.coordinate(name) > other.coordinate(name) + tolerance
            for name in coordinates
        ):
            return False
        keys = set(self.custom) | set(other.custom)
        return all(
            self.custom.get(key, 0.0) <= other.custom.get(key, 0.0) + tolerance
            for key in keys
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "route_count": float(self.route_count),
            "parameter_count": float(self.parameter_count),
            "flop_count": float(self.flop_count),
            "memory_bytes": float(self.memory_bytes),
            "latency_ms": float(self.latency_ms),
            "custom": dict(self.custom),
        }


__all__ = ["CostVector"]
