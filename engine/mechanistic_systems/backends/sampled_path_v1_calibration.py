"""Immutable finite-horizon calibration used by sampled_path_frontier_v1.

This module is deliberately separate from finite-MLE calibration.  The sealed
v1 circuit algorithm defines calibration by a fixed number of positive-affine
LBFGS steps.  A separating panel therefore has a finite algorithmic output even
though the corresponding unregularized logistic MLE does not exist.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping

from contracts.hashing import stable_digest
from mechanistic_systems.contracts import (
    SAMPLED_PATH_V1_CALIBRATION_ALGORITHM,
)

from .calibration_problem import CalibrationFactorization
from .contracts import BackendProfile

try:  # Optional for import smoke; required by executable calibration.
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - environment dependent.
    torch = F = None  # type: ignore[assignment]


@dataclass(frozen=True)
class SampledPathV1CalibrationReceipt:
    """Completion receipt for the sealed algorithmic v1 optimizer.

    ``algorithm_completed`` is intentionally not called convergence: on a
    separating panel the finite-horizon optimizer is well defined while the
    unregularized finite MLE is not.  New estimands use the finite-MLE receipt
    and continue to fail closed on separation.
    """

    dependency_hash: str
    factorization_hash: str
    component_hashes: tuple[str, ...]
    algorithm_id: str
    warm_start_objective: tuple[float, ...]
    initial_objective: tuple[float, ...]
    final_objective: tuple[float, ...]
    configured_iterations: int
    closure_evaluations: tuple[int, ...]
    finite: tuple[bool, ...]
    objective_not_worsened: tuple[bool, ...]
    separation_detected: tuple[bool, ...]
    warm_start_policy: str
    backend_profile: Mapping[str, Any]

    def validate(self) -> None:
        count = len(self.component_hashes)
        aligned = (
            self.warm_start_objective,
            self.initial_objective,
            self.final_objective,
            self.closure_evaluations,
            self.finite,
            self.objective_not_worsened,
            self.separation_detected,
        )
        if count == 0 or any(len(values) != count for values in aligned):
            raise ValueError("v1 calibration receipt axes are empty or misaligned")
        if len(self.dependency_hash) != 64 or len(self.factorization_hash) != 64:
            raise ValueError("v1 calibration receipt hashes are invalid")
        if self.algorithm_id != SAMPLED_PATH_V1_CALIBRATION_ALGORITHM:
            raise ValueError("v1 calibration receipt has the wrong algorithm ID")
        if self.configured_iterations <= 0:
            raise ValueError("v1 calibration iteration horizon must be positive")
        if any(
            not math.isfinite(float(value))
            for value in (
                *self.warm_start_objective,
                *self.initial_objective,
                *self.final_objective,
            )
        ):
            raise ValueError("v1 calibration receipt contains non-finite objectives")

    @property
    def passed(self) -> bool:
        self.validate()
        return all(self.finite) and all(self.objective_not_worsened)

    @property
    def content_hash(self) -> str:
        return stable_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_name": "mechanistic_sampled_path_v1_calibration_receipt",
            "schema_version": "1",
            "dependency_hash": self.dependency_hash,
            "factorization_hash": self.factorization_hash,
            "component_hashes": list(self.component_hashes),
            "algorithm_id": self.algorithm_id,
            "warm_start_objective": list(self.warm_start_objective),
            "initial_objective": list(self.initial_objective),
            "final_objective": list(self.final_objective),
            "configured_iterations": int(self.configured_iterations),
            "closure_evaluations": list(self.closure_evaluations),
            "finite": list(self.finite),
            "objective_not_worsened": list(self.objective_not_worsened),
            "separation_detected": list(self.separation_detected),
            "warm_start_policy": self.warm_start_policy,
            "backend_profile": dict(self.backend_profile),
            "passed": self.passed,
        }


@dataclass(frozen=True)
class SampledPathV1CalibrationFit:
    scale: Any
    offset: Any
    input_center: Any
    factorization: CalibrationFactorization
    receipt: SampledPathV1CalibrationReceipt
    family: str = "sampled_path_frontier_v1_coordinate_affine"

    @property
    def backend(self) -> str:
        return "torch_lbfgs_finite_horizon_v1"

    @property
    def calibration_backend(self) -> str:
        return self.backend

    @property
    def readout_scale(self) -> Any:
        return self.scale

    @property
    def readout_offset(self) -> Any:
        return self.offset

    def apply(self, values: Any) -> Any:
        tensor = torch.as_tensor(values)
        if tensor.ndim != 2 or int(tensor.shape[1]) != int(self.scale.shape[1]):
            raise ValueError("v1 calibration values must have shape [sample,query]")
        center = self.input_center.to(tensor).reshape(1, -1)
        return (
            (tensor - center) * self.scale.to(tensor)
            + self.offset.to(tensor)
        )


def fit_sampled_path_v1_coordinate_affine(
    values: Any,
    binary_targets: Any,
    *,
    factorization: CalibrationFactorization,
    input_center: Any,
    weighting: str = "pos_weighted",
    sample_weights: Any | None = None,
    minimum_scale: float = 1e-8,
    configured_iterations: int = 64,
    learning_rate: float = 0.25,
    damping_init: float = 1e-4,
    objective_tolerance: float = 1e-6,
) -> SampledPathV1CalibrationFit:
    """Execute the exact finite-horizon scalar calibration contract.

    The uniform-weight scalar path first fits an intercept with a fixed unit
    scale, then warm-starts a positive-scale LBFGS fit from that intercept.
    Nonuniform design-weighted fits start directly at identity. Both
    branches reinitialize for every fold rather than carrying parameters
    between sampled-path candidates.
    """

    if torch is None or F is None:
        raise RuntimeError("sampled-path v1 calibration requires PyTorch")
    if not factorization.is_fully_factorized:
        raise ValueError("sampled-path v1 coordinate calibration requires singleton factors")
    x = torch.as_tensor(values, dtype=torch.float32)
    if x.ndim != 2:
        raise ValueError("sampled-path v1 calibration values must be [sample,query]")
    y = torch.as_tensor(binary_targets, device=x.device, dtype=torch.float32)
    if tuple(y.shape) != tuple(x.shape):
        raise ValueError("sampled-path v1 targets differ from the value axis")
    if not bool(torch.isfinite(x).all().item()) or not bool(torch.isfinite(y).all().item()):
        raise ValueError("sampled-path v1 calibration inputs must be finite")
    if not bool(((y == 0.0) | (y == 1.0)).all().item()):
        raise ValueError("sampled-path v1 calibration targets must be binary")
    positives = y.sum(dim=0)
    negatives = float(y.shape[0]) - positives
    if bool(((positives <= 0.0) | (negatives <= 0.0)).any().item()):
        raise ValueError("every sampled-path v1 query requires both classes")
    if weighting not in {"pos_weighted", "unweighted"}:
        raise ValueError("sampled-path v1 supports positive-weighted or unweighted BCE")

    center = torch.as_tensor(input_center, device=x.device, dtype=torch.float32).reshape(-1)
    if int(center.numel()) == 1 and int(x.shape[1]) != 1:
        center = center.expand(int(x.shape[1]))
    if int(center.numel()) != int(x.shape[1]):
        raise ValueError("sampled-path v1 input center differs from the query axis")
    centered = x - center.reshape(1, -1)

    design = None
    if sample_weights is not None:
        design = torch.as_tensor(
            sample_weights, device=x.device, dtype=torch.float32
        ).reshape(-1)
        if int(design.numel()) != int(x.shape[0]):
            raise ValueError("sampled-path v1 design weights differ from samples")
        if not bool(torch.isfinite(design).all().item()) or bool((design < 0).any().item()):
            raise ValueError("sampled-path v1 design weights must be finite and nonnegative")
        if float(design.sum().item()) <= 0.0:
            raise ValueError("sampled-path v1 design weights must have positive mass")
        design = design / design.sum()
        if torch.allclose(design, design[:1].expand_as(design), rtol=1e-5, atol=1e-8):
            design = None

    positive_weight = (
        negatives / positives.clamp_min(1e-12)
        if weighting == "pos_weighted"
        else torch.ones_like(positives)
    )

    def objective(logits: Any, query_index: int) -> Any:
        target = y[:, query_index]
        row_weight = torch.where(
            target > 0.5,
            positive_weight[query_index],
            torch.ones_like(target),
        )
        row = torch.where(
            target > 0.5,
            F.softplus(-logits),
            F.softplus(logits),
        ) * row_weight
        return row.mean() if design is None else (row * design).sum()

    warm_start_objective: list[float] = []
    initial_objective: list[float] = []
    final_objective: list[float] = []
    scales: list[Any] = []
    offsets: list[Any] = []
    closure_evaluations: list[int] = []
    finite: list[bool] = []
    objective_not_worsened: list[bool] = []
    separation_detected: list[bool] = []
    started = time.perf_counter()
    damping = torch.tensor(
        float(damping_init), device=x.device, dtype=torch.float32
    ).clamp_min(1e-12)

    for query_index in range(int(x.shape[1])):
        target = y[:, query_index]
        row_weight = torch.where(
            target > 0.5,
            positive_weight[query_index],
            torch.ones_like(target),
        )
        bias_offset = torch.zeros((), device=x.device, dtype=torch.float32)
        if design is None:
            denom = torch.tensor(
                float(x.shape[0]), device=x.device, dtype=torch.float32
            ).clamp_min(1.0)
            with torch.no_grad():
                for _ in range(max(1, int(configured_iterations))):
                    eta = centered[:, query_index] + bias_offset
                    probability = torch.sigmoid(eta)
                    difference = (probability - target) * row_weight / denom
                    curvature = (
                        probability * (1.0 - probability) * row_weight / denom
                    )
                    gradient = difference.sum()
                    hessian = curvature.sum().clamp_min(damping)
                    bias_offset = bias_offset - gradient / hessian
        raw_scale = torch.tensor(
            0.54132485,
            device=x.device,
            dtype=torch.float32,
            requires_grad=True,
        )
        offset = torch.tensor(
            float(bias_offset.item()),
            device=x.device,
            dtype=torch.float32,
            requires_grad=True,
        )
        optimizer = torch.optim.LBFGS(
            (raw_scale, offset),
            lr=float(learning_rate),
            max_iter=max(1, int(configured_iterations)),
            line_search_fn="strong_wolfe",
        )
        calls = 0

        def closure() -> Any:
            nonlocal calls
            calls += 1
            optimizer.zero_grad(set_to_none=True)
            scale = F.softplus(raw_scale) + float(minimum_scale)
            loss = objective(scale * centered[:, query_index] + offset, query_index)
            loss.backward()
            return loss

        with torch.no_grad():
            initial_scale = F.softplus(raw_scale) + float(minimum_scale)
            warm_start = objective(
                centered[:, query_index] + bias_offset,
                query_index,
            )
            initial = objective(
                initial_scale * centered[:, query_index] + offset,
                query_index,
            )
        optimizer.step(closure)
        with torch.no_grad():
            scale = F.softplus(raw_scale) + float(minimum_scale)
            final = objective(scale * centered[:, query_index] + offset, query_index)
            values_finite = bool(
                torch.isfinite(scale).item()
                and torch.isfinite(offset).item()
                and torch.isfinite(final).item()
            )
            not_worsened = bool(
                values_finite
                and float(final.item())
                <= float(initial.item()) + float(objective_tolerance)
            )
            active = torch.ones_like(y[:, query_index], dtype=torch.bool)
            if design is not None:
                active = design > 0.0
            positive_min = centered[
                active & (y[:, query_index] > 0.5), query_index
            ].amin()
            negative_max = centered[
                active & (y[:, query_index] <= 0.5), query_index
            ].amax()
            active_values = centered[active, query_index]
            separated = bool(
                (active_values.max() > active_values.min()).item()
                and (negative_max <= positive_min).item()
            )
        warm_start_objective.append(float(warm_start.item()))
        initial_objective.append(float(initial.item()))
        final_objective.append(float(final.item()))
        scales.append(scale.detach())
        offsets.append(offset.detach())
        closure_evaluations.append(int(calls))
        finite.append(values_finite)
        objective_not_worsened.append(not_worsened)
        separation_detected.append(separated)

    scale_tensor = torch.stack(scales).reshape(1, -1)
    offset_tensor = torch.stack(offsets).reshape(1, -1)
    profile = BackendProfile(
        backend_id="mechanistic_systems.sampled_path_v1_calibration",
        capability="finite_horizon_singleton_positive_affine_calibration",
        implementation="torch_float32_lbfgs_strong_wolfe",
        device_type=str(x.device.type),
        numerical_mode="sealed_v1_bias_newton_then_softplus_lbfgs",
        calls=sum(closure_evaluations),
        wall_time_s=max(0.0, time.perf_counter() - started),
        peak_hbm_bytes=(
            int(torch.cuda.max_memory_allocated(x.device))
            if x.device.type == "cuda"
            else None
        ),
        processed_contexts=int(x.shape[0]),
        processed_queries=int(x.shape[1]),
        metadata={
            "configured_iterations": int(configured_iterations),
            "learning_rate": float(learning_rate),
            "bias_newton_damping": float(damping_init),
            "minimum_scale": float(minimum_scale),
            "separation_is_diagnostic_not_failure": True,
        },
    )
    receipt = SampledPathV1CalibrationReceipt(
        dependency_hash=factorization.dependency_hash,
        factorization_hash=factorization.factorization_hash,
        component_hashes=tuple(
            component.component_id for component in factorization.components
        ),
        algorithm_id=SAMPLED_PATH_V1_CALIBRATION_ALGORITHM,
        warm_start_objective=tuple(warm_start_objective),
        initial_objective=tuple(initial_objective),
        final_objective=tuple(final_objective),
        configured_iterations=int(configured_iterations),
        closure_evaluations=tuple(closure_evaluations),
        finite=tuple(finite),
        objective_not_worsened=tuple(objective_not_worsened),
        separation_detected=tuple(separation_detected),
        warm_start_policy=(
            "sealed_v1_uniform_bias_only_newton_then_identity_scale_lbfgs_"
            "else_design_weighted_identity"
        ),
        backend_profile=profile.to_runtime_dict(),
    )
    if not receipt.passed:
        raise RuntimeError(
            "sampled_path_frontier_v1 finite-horizon calibration failed its "
            f"algorithmic receipt: {receipt.to_dict()}"
        )
    return SampledPathV1CalibrationFit(
        scale=scale_tensor,
        offset=offset_tensor,
        input_center=center.reshape(1, -1),
        factorization=factorization,
        receipt=receipt,
    )


__all__ = [
    "SAMPLED_PATH_V1_CALIBRATION_ALGORITHM",
    "SampledPathV1CalibrationFit",
    "SampledPathV1CalibrationReceipt",
    "fit_sampled_path_v1_coordinate_affine",
]
