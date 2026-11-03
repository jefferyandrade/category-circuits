"""Jordan roles as exact simultaneous one-sided tests.

The role taxonomy (supportive / opposing / context_bipolar / negligible
/ unresolved) is fixed; each label is the
conclusion of one-sided moderated-t bounds holding simultaneously at the
declared per-cell alpha (Bonferroni across the five one-sided bounds a label
consumes), so a printed label is a guarantee rather than a description.
Cells whose evidence cannot separate from the floor stay honestly
``unresolved`` instead of being silently forced by a zero-width interval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bounds import student_t_interval
from .moderation import VarianceModerationPrior, moderated_variances

ROLE_IDS = (
    "supportive",
    "opposing",
    "context_bipolar",
    "negligible",
    "unresolved",
)

_BOUNDS_PER_CELL = 5


@dataclass(frozen=True)
class ExactRoleEvidence:
    role_labels: np.ndarray
    positive_lower: np.ndarray
    positive_upper: np.ndarray
    negative_lower: np.ndarray
    negative_upper: np.ndarray
    activity_upper: np.ndarray
    per_bound_alpha: float
    method: str = "exact_simultaneous_one_sided_moderated_t"

    def validate(self) -> None:
        arrays = (
            self.positive_lower,
            self.positive_upper,
            self.negative_lower,
            self.negative_upper,
            self.activity_upper,
        )
        if len({np.asarray(value).shape for value in arrays}) != 1:
            raise ValueError("exact role evidence must share one cell axis")
        if not set(np.asarray(self.role_labels).astype(str).reshape(-1)).issubset(
            set(ROLE_IDS)
        ):
            raise ValueError("exact role evidence contains an unknown role")
        if not 0.0 < float(self.per_bound_alpha) < 1.0:
            raise ValueError("exact role evidence requires per-bound alpha in (0,1)")


def _variance_from_moments(
    total: np.ndarray, squares: np.ndarray, count: np.ndarray
) -> np.ndarray:
    mean = np.divide(
        total, count, out=np.zeros_like(total, dtype=np.float64), where=count > 0
    )
    return np.divide(
        squares - count * mean * mean,
        count - 1.0,
        out=np.zeros_like(total, dtype=np.float64),
        where=count > 1,
    )


def exact_simultaneous_roles(
    *,
    positive_sum: np.ndarray,
    positive_sum_squares: np.ndarray,
    negative_sum: np.ndarray,
    negative_sum_squares: np.ndarray,
    absolute_sum: np.ndarray,
    absolute_sum_squares: np.ndarray,
    count: np.ndarray,
    cell_alpha: float,
    negligible_effect_floor: float,
    moderation_prior: VarianceModerationPrior | None = None,
) -> ExactRoleEvidence:
    """Label every cell with simultaneous one-sided guarantees at ``cell_alpha``."""

    count_a = np.asarray(count, dtype=np.float64)
    floor = float(negligible_effect_floor)
    if not np.isfinite(floor) or floor < 0.0:
        raise ValueError("negligible-effect floor must be finite and nonnegative")
    per_bound_alpha = float(cell_alpha) / _BOUNDS_PER_CELL
    residual_dof = np.maximum(count_a - 1.0, 0.0)

    def _bounds(total: np.ndarray, squares: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        total_a = np.asarray(total, dtype=np.float64)
        squares_a = np.asarray(squares, dtype=np.float64)
        mean = np.divide(
            total_a,
            count_a,
            out=np.zeros_like(total_a, dtype=np.float64),
            where=count_a > 0,
        )
        variance = np.maximum(_variance_from_moments(total_a, squares_a, count_a), 0.0)
        if moderation_prior is not None:
            variance, dof = moderated_variances(variance, residual_dof, moderation_prior)
        else:
            dof = residual_dof
        lower = student_t_interval(
            mean, variance, count_a, alpha=per_bound_alpha, dof=dof, one_sided=True
        ).one_sided_lower()
        upper = student_t_interval(
            mean, variance, count_a, alpha=per_bound_alpha, dof=dof, one_sided=True
        ).one_sided_upper()
        return np.maximum(lower, 0.0), upper

    positive_lower, positive_upper = _bounds(positive_sum, positive_sum_squares)
    negative_lower, negative_upper = _bounds(negative_sum, negative_sum_squares)
    _activity_lower, activity_upper = _bounds(absolute_sum, absolute_sum_squares)

    labels = np.full(positive_lower.shape, "unresolved", dtype="U17")
    labels[activity_upper <= floor] = "negligible"
    labels[(positive_lower > floor) & (negative_upper <= floor)] = "supportive"
    labels[(negative_lower > floor) & (positive_upper <= floor)] = "opposing"
    labels[(positive_lower > floor) & (negative_lower > floor)] = "context_bipolar"

    evidence = ExactRoleEvidence(
        role_labels=labels,
        positive_lower=positive_lower,
        positive_upper=positive_upper,
        negative_lower=negative_lower,
        negative_upper=negative_upper,
        activity_upper=activity_upper,
        per_bound_alpha=per_bound_alpha,
    )
    evidence.validate()
    return evidence


__all__ = ["ExactRoleEvidence", "ROLE_IDS", "exact_simultaneous_roles"]
