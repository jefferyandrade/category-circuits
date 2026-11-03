"""Empirical-Bayes variance moderation fitted across the parallel cell population.

The v1 design measures thousands of cells under one frozen protocol, which is
a massive internal replication that fixed prior constants would ignore.
Following the moderated-t construction (Smyth 2004), per-cell sample variances
are shrunk toward a scaled-inverse-chi-square prior whose scale and degrees of
freedom are fitted by moment matching on log-variances across the population.
The moderated variance behaves like ``dof + prior_dof`` observations, which is
what makes honest small-stage inference affordable, and cells with one
observation receive the finite prior variance instead of a silent zero
standard error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import digamma, polygamma


_PRIOR_DOF_CAP = 1e6
_MINIMUM_VARIANCE = 1e-24


@dataclass(frozen=True)
class VarianceModerationPrior:
    prior_dof: float
    prior_variance: float
    fitted_cell_count: int
    fit_method: str = "log_variance_moment_matching_smyth_2004"

    def validate(self) -> None:
        if not math.isfinite(self.prior_variance) or self.prior_variance <= 0.0:
            raise ValueError("moderation prior variance must be finite and positive")
        if not 0.0 < self.prior_dof <= _PRIOR_DOF_CAP:
            raise ValueError("moderation prior dof must be positive and capped")
        if int(self.fitted_cell_count) < 2:
            raise ValueError("moderation prior requires at least two fitted cells")

    def to_dict(self) -> dict[str, object]:
        return {
            "prior_dof": float(self.prior_dof),
            "prior_variance": float(self.prior_variance),
            "fitted_cell_count": int(self.fitted_cell_count),
            "fit_method": self.fit_method,
        }


def _trigamma(value: np.ndarray | float) -> np.ndarray:
    return polygamma(1, value)


def _trigamma_inverse(target: float) -> float:
    """Invert the trigamma function by Newton iteration (limma's recipe)."""

    if target <= 0.0:
        return _PRIOR_DOF_CAP * 2.0
    if target > 1e7:
        return 1.0 / math.sqrt(target)
    guess = 0.5 + 1.0 / target
    for _ in range(64):
        current = float(_trigamma(guess))
        gradient = float(polygamma(2, guess))
        if gradient == 0.0:
            break
        step = current * (1.0 - current / target) / gradient
        guess += step
        if -step < 1e-8 * guess:
            break
    return float(guess)


def fit_variance_prior(
    sample_variances: object,
    dof: object,
    *,
    minimum_dof: float = 1.0,
) -> VarianceModerationPrior:
    """Fit (prior_dof, prior_variance) across the population of cell variances."""

    variances = np.asarray(sample_variances, dtype=np.float64).reshape(-1)
    dof_a = np.broadcast_to(
        np.asarray(dof, dtype=np.float64), variances.shape
    ).astype(np.float64)
    keep = (
        np.isfinite(variances)
        & (variances > _MINIMUM_VARIANCE)
        & np.isfinite(dof_a)
        & (dof_a >= float(minimum_dof))
    )
    variances = variances[keep]
    dof_kept = dof_a[keep]
    if variances.size < 2:
        raise ValueError("variance-prior fitting requires at least two informative cells")
    log_values = np.log(variances) - (digamma(dof_kept / 2.0) - np.log(dof_kept / 2.0))
    mean_log = float(np.mean(log_values))
    var_log = float(np.var(log_values, ddof=1))
    excess = var_log - float(np.mean(_trigamma(dof_kept / 2.0)))
    if excess <= 0.0:
        prior_dof = _PRIOR_DOF_CAP
        prior_variance = math.exp(mean_log)
    else:
        half_dof = _trigamma_inverse(excess)
        prior_dof = min(2.0 * half_dof, _PRIOR_DOF_CAP)
        prior_variance = math.exp(
            mean_log + float(digamma(half_dof)) - math.log(half_dof)
        )
    prior = VarianceModerationPrior(
        prior_dof=float(prior_dof),
        prior_variance=float(max(prior_variance, _MINIMUM_VARIANCE)),
        fitted_cell_count=int(variances.size),
    )
    prior.validate()
    return prior


def moderated_variances(
    sample_variances: object,
    dof: object,
    prior: VarianceModerationPrior,
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior-mean variances and total degrees of freedom per cell.

    Cells with zero residual degrees of freedom (count <= 1) return the prior
    variance with the prior degrees of freedom: finite, honest, and strictly
    wider than pretending certainty.
    """

    prior.validate()
    variances = np.asarray(sample_variances, dtype=np.float64)
    dof_a = np.broadcast_to(np.asarray(dof, dtype=np.float64), variances.shape).astype(
        np.float64
    )
    safe_dof = np.maximum(dof_a, 0.0)
    safe_variances = np.where(
        np.isfinite(variances) & (variances >= 0.0) & (safe_dof > 0.0),
        variances,
        0.0,
    )
    moderated = (
        float(prior.prior_dof) * float(prior.prior_variance)
        + safe_dof * safe_variances
    ) / (float(prior.prior_dof) + safe_dof)
    total_dof = float(prior.prior_dof) + safe_dof
    return moderated, total_dof


__all__ = [
    "VarianceModerationPrior",
    "fit_variance_prior",
    "moderated_variances",
]
