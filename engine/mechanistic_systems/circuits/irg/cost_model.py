"""Online realized-cost model for measurement acquisition and packing decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .contracts import MeasurementRequest
from .costs import CostEstimate, RealizedCost


@dataclass(frozen=True)
class CostObservation:
    kind: str
    contexts: int
    candidates: int
    packing_degree: int
    wall_seconds: float
    device_seconds: float
    forward_passes: int
    backward_passes: int
    calibration_fits: int

    def __post_init__(self) -> None:
        if not self.kind or min(self.contexts, self.candidates, self.packing_degree) <= 0:
            raise ValueError("cost observation identity/counts are invalid")
        for name in ("wall_seconds", "device_seconds"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")


class OnlineRequestCostModel:
    """Per-kind ridge regression over measured request geometry.

    The feature map separates fixed launch overhead, contexts, candidates, and
    packed context-candidate work.  Until enough observations exist, the model
    returns the request's declared cost rather than inventing precision.
    """

    def __init__(self, *, ridge: float = 1e-6, minimum_observations: int = 4) -> None:
        self.ridge = float(ridge)
        self.minimum_observations = int(minimum_observations)
        if self.ridge < 0.0 or self.minimum_observations <= 0:
            raise ValueError("online cost model controls are invalid")
        self._observations: dict[str, list[CostObservation]] = {}

    @staticmethod
    def _features(contexts: int, candidates: int, packing_degree: int) -> np.ndarray:
        c = float(contexts)
        k = float(candidates)
        p = float(max(1, packing_degree))
        return np.asarray([1.0, c, k, c * k / p], dtype=np.float64)

    def observe(
        self,
        request: MeasurementRequest,
        cost: RealizedCost,
        *,
        context_count: int,
    ) -> None:
        candidates = max(1, len(request.candidate_ids))
        observation = CostObservation(
            kind=request.kind,
            contexts=max(1, int(context_count)),
            candidates=candidates,
            packing_degree=max(1, int(cost.packing_degree)),
            wall_seconds=float(cost.wall_seconds),
            device_seconds=float(cost.device_seconds),
            forward_passes=int(cost.forward_passes),
            backward_passes=int(cost.backward_passes),
            calibration_fits=int(cost.calibration_fits),
        )
        self._observations.setdefault(request.kind, []).append(observation)

    def observation_count(self, kind: str | None = None) -> int:
        if kind is None:
            return sum(len(items) for items in self._observations.values())
        return len(self._observations.get(str(kind), ()))

    def _fit_coordinate(self, observations: Sequence[CostObservation], coordinate: str) -> tuple[np.ndarray, float]:
        x = np.vstack(
            [self._features(item.contexts, item.candidates, item.packing_degree) for item in observations]
        )
        y = np.asarray([float(getattr(item, coordinate)) for item in observations], dtype=np.float64)
        coefficients = np.linalg.solve(x.T @ x + self.ridge * np.eye(x.shape[1]), x.T @ y)
        residual = y - x @ coefficients
        uncertainty = float(np.sqrt(np.mean(residual * residual))) if residual.size else 0.0
        return coefficients, uncertainty

    def predict(
        self,
        request: MeasurementRequest,
        *,
        context_count: int | None = None,
        packing_degree: int | None = None,
    ) -> CostEstimate:
        observations = self._observations.get(request.kind, ())
        if len(observations) < self.minimum_observations:
            return request.expected_cost
        contexts = max(1, int(context_count or request.context_selector.requested_count or 1))
        candidates = max(1, len(request.candidate_ids))
        packing = max(1, int(packing_degree or candidates))
        features = self._features(contexts, candidates, packing)
        wall_coefficients, wall_uncertainty = self._fit_coordinate(observations, "wall_seconds")
        device_coefficients, _device_uncertainty = self._fit_coordinate(observations, "device_seconds")
        wall = max(0.0, float(features @ wall_coefficients))
        device = max(0.0, float(features @ device_coefficients))
        uncertainty_fraction = min(10.0, wall_uncertainty / max(wall, 1e-12))
        return CostEstimate(
            wall_seconds=wall,
            device_seconds=device,
            forward_passes=request.expected_cost.forward_passes,
            backward_passes=request.expected_cost.backward_passes,
            calibration_fits=request.expected_cost.calibration_fits,
            transferred_bytes=request.expected_cost.transferred_bytes,
            peak_memory_bytes=request.expected_cost.peak_memory_bytes,
            uncertainty_fraction=uncertainty_fraction,
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "schema": "irg_online_request_cost_model_v1",
            "minimum_observations": self.minimum_observations,
            "ridge": self.ridge,
            "counts_by_kind": {
                kind: len(items) for kind, items in sorted(self._observations.items())
            },
        }


__all__ = ["CostObservation", "OnlineRequestCostModel"]
