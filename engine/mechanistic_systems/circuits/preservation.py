"""Target-preservation estimands for rate-distortion circuit search."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _binary_labels(labels: np.ndarray, target_class_id: int) -> np.ndarray:
    values = np.asarray(labels).reshape(-1)
    if values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("target-preservation labels must be finite and nonempty")
    binary = values.astype(np.int64, copy=False) == int(target_class_id)
    positive_count = int(np.count_nonzero(binary))
    if positive_count == 0 or positive_count == binary.size:
        raise ValueError("balanced target preservation requires positive and negative contexts")
    return binary.astype(np.float64)


def balanced_binary_bce(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    target_class_id: int,
) -> float:
    """Class-balanced binary cross entropy over a class-query logit."""

    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    binary = _binary_labels(labels, target_class_id)
    if values.shape != binary.shape or not np.all(np.isfinite(values)):
        raise ValueError("target logits and labels are incompatible or non-finite")
    positive_count = float(np.sum(binary))
    negative_count = float(binary.size - positive_count)
    positive_weight = negative_count / positive_count
    losses = (1.0 - binary) * np.logaddexp(0.0, values)
    losses += binary * positive_weight * np.logaddexp(0.0, -values)
    # Match ``torch.nn.BCEWithLogitsLoss(pos_weight=..., reduction="mean")``:
    # weights alter the summed loss, while the mean still divides by the
    # number of contexts rather than the sum of effective weights.
    return float(np.sum(losses) / binary.size)


@dataclass(frozen=True)
class BalancedBCEDamage:
    full_loss: float
    masked_loss: float
    fully_pruned_loss: float
    raw_damage: float
    response_distortion: float
    target_preservation: float

    def to_metrics(self) -> dict[str, float]:
        return {
            "balanced_bce_full": self.full_loss,
            "balanced_bce_masked": self.masked_loss,
            "balanced_bce_fully_pruned": self.fully_pruned_loss,
            "balanced_bce_damage_raw": self.raw_damage,
            "balanced_bce_damage": self.response_distortion,
            "balanced_bce_preservation": self.target_preservation,
        }


@dataclass(frozen=True)
class AffineCalibrationFit:
    """Deterministic two-parameter calibration for one target logit."""

    scale: float
    offset: float
    loss: float
    iterations: int
    converged: bool

    def calibrated_logits(self, logits: np.ndarray) -> np.ndarray:
        values = np.asarray(logits, dtype=np.float64)
        return self.scale * values + self.offset


def fit_balanced_affine_calibration(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    target_class_id: int,
    initial_scale: float = 1.0,
    initial_offset: float = 0.0,
    max_iterations: int = 80,
    tolerance: float = 1e-10,
    hessian_ridge: float = 1e-8,
) -> AffineCalibrationFit:
    """Fit class-balanced logistic scale and offset by damped Newton steps.

    This is the compact canonical calibration primitive. It deliberately has
    no model or pruning dependencies, making exact legacy-backend parity a
    separately testable boundary.
    """

    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    binary = _binary_labels(labels, target_class_id)
    if values.shape != binary.shape or not np.all(np.isfinite(values)):
        raise ValueError("target logits and labels are incompatible or non-finite")
    positive_count = float(np.sum(binary))
    negative_count = float(binary.size - positive_count)
    weights = np.where(binary > 0.5, negative_count / positive_count, 1.0)
    design = np.column_stack((values, np.ones_like(values)))
    theta = np.asarray([initial_scale, initial_offset], dtype=np.float64)
    if not np.all(np.isfinite(theta)):
        raise ValueError("affine initialization must be finite")

    def objective(parameters: np.ndarray) -> float:
        eta = design @ parameters
        losses = (1.0 - binary) * np.logaddexp(0.0, eta)
        losses += binary * weights * np.logaddexp(0.0, -eta)
        return float(np.sum(losses) / binary.size)

    loss = objective(theta)
    converged = False
    completed = 0
    for iteration in range(1, int(max_iterations) + 1):
        eta = np.clip(design @ theta, -50.0, 50.0)
        probabilities = 1.0 / (1.0 + np.exp(-eta))
        gradient = design.T @ (weights * (probabilities - binary)) / binary.size
        curvature = weights * probabilities * (1.0 - probabilities)
        hessian = (design.T * curvature) @ design / binary.size
        hessian.flat[::3] += float(hessian_ridge)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        if not np.all(np.isfinite(step)):
            raise ValueError("affine calibration produced a non-finite Newton step")
        completed = iteration
        if float(np.linalg.norm(step)) <= float(tolerance) * (1.0 + float(np.linalg.norm(theta))):
            converged = True
            break
        accepted = False
        step_scale = 1.0
        for _ in range(30):
            proposal = theta - step_scale * step
            proposal_loss = objective(proposal)
            if np.isfinite(proposal_loss) and proposal_loss <= loss + 1e-14:
                theta = proposal
                improvement = loss - proposal_loss
                loss = proposal_loss
                accepted = True
                if improvement <= float(tolerance) * (1.0 + abs(loss)):
                    converged = True
                break
            step_scale *= 0.5
        if not accepted or converged:
            break
    return AffineCalibrationFit(
        scale=float(theta[0]),
        offset=float(theta[1]),
        loss=float(loss),
        iterations=completed,
        converged=converged,
    )


def balanced_bce_damage(
    full_logits: np.ndarray,
    masked_logits: np.ndarray,
    fully_pruned_logits: np.ndarray,
    labels: np.ndarray,
    *,
    target_class_id: int,
) -> BalancedBCEDamage:
    """Normalize masked damage between the full and fully pruned executions.

    The distortion endpoint is zero for the full model and one for the fully
    pruned model. A mask that improves the balanced objective retains a raw
    negative damage, while its optimization distortion is clipped at zero.
    """

    full_loss = balanced_binary_bce(
        full_logits,
        labels,
        target_class_id=target_class_id,
    )
    masked_loss = balanced_binary_bce(
        masked_logits,
        labels,
        target_class_id=target_class_id,
    )
    fully_pruned_loss = balanced_binary_bce(
        fully_pruned_logits,
        labels,
        target_class_id=target_class_id,
    )
    denominator = fully_pruned_loss - full_loss
    if not np.isfinite(denominator) or denominator <= 1e-12:
        raise ValueError(
            "fully pruned target loss must exceed the full-model target loss"
        )
    raw_damage = float((masked_loss - full_loss) / denominator)
    distortion = max(0.0, raw_damage)
    return BalancedBCEDamage(
        full_loss=full_loss,
        masked_loss=masked_loss,
        fully_pruned_loss=fully_pruned_loss,
        raw_damage=raw_damage,
        response_distortion=distortion,
        target_preservation=1.0 - raw_damage,
    )


__all__ = [
    "AffineCalibrationFit",
    "BalancedBCEDamage",
    "balanced_bce_damage",
    "balanced_binary_bce",
    "fit_balanced_affine_calibration",
]
