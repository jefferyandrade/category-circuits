"""Capability-neutral compute-cost contracts for measurement transport.

These objects account for the computation used to acquire a measurement.  They
do not describe a circuit's deployable cost and grant no pruning-workflow
authority.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _finite_nonnegative(value: float, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


@dataclass(frozen=True)
class CostEstimate:
    """Prospective compute cost used by a measurement acquisition policy."""

    wall_seconds: float = 0.0
    device_seconds: float = 0.0
    forward_passes: float = 0.0
    backward_passes: float = 0.0
    calibration_fits: float = 0.0
    transferred_bytes: float = 0.0
    peak_memory_bytes: float = 0.0
    uncertainty_fraction: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "wall_seconds",
            "device_seconds",
            "forward_passes",
            "backward_passes",
            "calibration_fits",
            "transferred_bytes",
            "peak_memory_bytes",
            "uncertainty_fraction",
        ):
            _finite_nonnegative(getattr(self, name), name=name)

    def scalar(
        self,
        *,
        wall_weight: float = 1.0,
        device_weight: float = 0.0,
    ) -> float:
        wall = _finite_nonnegative(wall_weight, name="wall_weight")
        device = _finite_nonnegative(device_weight, name="device_weight")
        value = wall * self.wall_seconds + device * self.device_seconds
        if value <= 0.0:
            # Keep zero-cost metadata requests selectable without creating an
            # infinite score. One nanosecond is a deterministic numerical floor.
            return 1e-9
        return float(value)

    def to_dict(self) -> dict[str, float]:
        return {
            "wall_seconds": float(self.wall_seconds),
            "device_seconds": float(self.device_seconds),
            "forward_passes": float(self.forward_passes),
            "backward_passes": float(self.backward_passes),
            "calibration_fits": float(self.calibration_fits),
            "transferred_bytes": float(self.transferred_bytes),
            "peak_memory_bytes": float(self.peak_memory_bytes),
            "uncertainty_fraction": float(self.uncertainty_fraction),
        }


@dataclass(frozen=True)
class RealizedCost:
    """Measured compute cost returned by a measurement provider."""

    wall_seconds: float = 0.0
    device_seconds: float = 0.0
    forward_passes: int = 0
    backward_passes: int = 0
    calibration_fits: int = 0
    transferred_bytes: int = 0
    peak_memory_bytes: int = 0
    packing_degree: int = 1
    cache_hit: bool = False

    def __post_init__(self) -> None:
        _finite_nonnegative(self.wall_seconds, name="wall_seconds")
        _finite_nonnegative(self.device_seconds, name="device_seconds")
        for name in (
            "forward_passes",
            "backward_passes",
            "calibration_fits",
            "transferred_bytes",
            "peak_memory_bytes",
        ):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if int(self.packing_degree) <= 0:
            raise ValueError("packing_degree must be positive")

    @classmethod
    def zero(cls) -> "RealizedCost":
        return cls()

    def to_estimate(self) -> CostEstimate:
        return CostEstimate(
            wall_seconds=self.wall_seconds,
            device_seconds=self.device_seconds,
            forward_passes=float(self.forward_passes),
            backward_passes=float(self.backward_passes),
            calibration_fits=float(self.calibration_fits),
            transferred_bytes=float(self.transferred_bytes),
            peak_memory_bytes=float(self.peak_memory_bytes),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "wall_seconds": float(self.wall_seconds),
            "device_seconds": float(self.device_seconds),
            "forward_passes": int(self.forward_passes),
            "backward_passes": int(self.backward_passes),
            "calibration_fits": int(self.calibration_fits),
            "transferred_bytes": int(self.transferred_bytes),
            "peak_memory_bytes": int(self.peak_memory_bytes),
            "packing_degree": int(self.packing_degree),
            "cache_hit": bool(self.cache_hit),
        }


__all__ = ["CostEstimate", "RealizedCost"]
