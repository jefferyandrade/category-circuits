"""Finite-sample-exact and anytime-valid interval family.

Three constructions cover every consumer:

- ``student_t_interval``: fixed-design exact intervals from (mean, variance,
  count) sufficient statistics, vectorized over cells, with optional
  moderation-extended degrees of freedom.  This replaces every plug-in
  ``NormalDist().inv_cdf`` z-interval.
- ``empirical_bernstein_interval``: fixed-n distribution-free intervals for
  statistics with a declared bounded range (Maurer-Pontil), used where
  normality is not defensible.
- ``eb_confidence_sequence``: a time-uniform predictable-mixture empirical
  Bernstein confidence sequence for bounded observations, valid under
  continuous peeking (Waudby-Smith & Ramdas).  Used when a stopping rule
  consumes evidence between stages of the sealed prefix ladder.

Planned interim looks at pre-registered stages are handled explicitly by
``staged_alpha``: an exact union-bound spend over the declared number of
looks.  With the four-stage ladder this widens each look mildly and keeps the
guarantee exactly valid at every stage, with no asymptotic argument.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import stdtrit


EVIDENCE_FOUNDATION_V1 = "evidence_foundation_v1"

_MINIMUM_ALPHA = 1e-12
_MINIMUM_VARIANCE = 0.0


@dataclass(frozen=True)
class EvidenceInterval:
    """One two-sided interval (vectorized) with its construction provenance."""

    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    standard_error: np.ndarray
    count: np.ndarray
    dof: np.ndarray
    alpha: float
    method: str

    def validate(self) -> None:
        arrays = (self.mean, self.lower, self.upper, self.standard_error)
        if len({np.asarray(value).shape for value in arrays}) != 1:
            raise ValueError("evidence interval fields must share one shape")
        if not 0.0 < float(self.alpha) < 1.0:
            raise ValueError("evidence interval alpha must lie in (0,1)")
        if np.any(np.asarray(self.lower) > np.asarray(self.upper)):
            raise ValueError("evidence interval bounds are inverted")
        if not self.method:
            raise ValueError("evidence interval requires a method label")

    def one_sided_upper(self) -> np.ndarray:
        """Upper bound at level ``alpha`` one-sided (the accept-gate form)."""

        return np.asarray(self.upper)

    def one_sided_lower(self) -> np.ndarray:
        return np.asarray(self.lower)


def staged_alpha(alpha: float, *, planned_looks: int) -> float:
    """Exact union-bound alpha for a pre-registered interim-look ladder."""

    looks = int(planned_looks)
    if looks <= 0:
        raise ValueError("staged alpha requires at least one planned look")
    value = float(alpha) / float(looks)
    if not _MINIMUM_ALPHA < value < 1.0:
        raise ValueError("staged alpha left the representable range")
    return value


def _as_float_arrays(*values: object) -> tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in values)
    shape = np.broadcast_shapes(*(array.shape for array in arrays))
    return tuple(np.broadcast_to(array, shape).astype(np.float64) for array in arrays)


def student_t_interval(
    mean: object,
    variance: object,
    count: object,
    *,
    alpha: float,
    dof: object | None = None,
    one_sided: bool = False,
) -> EvidenceInterval:
    """Exact t-interval from sufficient statistics at the exchangeable unit.

    ``variance`` is the (possibly moderated) variance of one observation;
    ``dof`` defaults to ``count - 1`` and should include prior degrees of
    freedom when the variance was moderated.  Cells with ``count < 1`` or
    nonpositive degrees of freedom produce unbounded intervals rather than a
    silent zero-width collapse.
    """

    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("student t interval requires alpha in (0,1)")
    mean_a, variance_a, count_a = _as_float_arrays(mean, variance, count)
    if np.any(variance_a < _MINIMUM_VARIANCE) or not np.isfinite(mean_a).all():
        raise ValueError("student t interval requires finite means and nonnegative variances")
    dof_a = (
        np.maximum(count_a - 1.0, 0.0)
        if dof is None
        else _as_float_arrays(dof, mean_a)[0]
    )
    tail = float(alpha) if one_sided else float(alpha) / 2.0
    standard_error = np.sqrt(
        np.divide(
            variance_a,
            count_a,
            out=np.full_like(variance_a, np.inf),
            where=count_a >= 1.0,
        )
    )
    usable = (dof_a > 0.0) & np.isfinite(standard_error)
    radius = np.full_like(mean_a, np.inf)
    if np.any(usable):
        radius[usable] = stdtrit(dof_a[usable], 1.0 - tail) * standard_error[usable]
    interval = EvidenceInterval(
        mean=mean_a,
        lower=mean_a - radius,
        upper=mean_a + radius,
        standard_error=standard_error,
        count=count_a,
        dof=dof_a,
        alpha=float(alpha),
        method="student_t_one_sided" if one_sided else "student_t",
    )
    interval.validate()
    return interval


def empirical_bernstein_interval(
    mean: object,
    variance: object,
    count: object,
    *,
    alpha: float,
    value_range: float,
) -> EvidenceInterval:
    """Fixed-n Maurer-Pontil empirical-Bernstein interval for bounded values."""

    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("empirical Bernstein interval requires alpha in (0,1)")
    span = float(value_range)
    if not math.isfinite(span) or span <= 0.0:
        raise ValueError("empirical Bernstein interval requires a positive value range")
    mean_a, variance_a, count_a = _as_float_arrays(mean, variance, count)
    if np.any(variance_a < _MINIMUM_VARIANCE):
        raise ValueError("empirical Bernstein interval requires nonnegative variances")
    log_term = math.log(2.0 / float(alpha))
    usable = count_a >= 2.0
    variance_part = np.sqrt(
        np.divide(
            2.0 * variance_a * log_term,
            count_a,
            out=np.full_like(variance_a, np.inf),
            where=usable,
        )
    )
    range_part = np.divide(
        7.0 * span * log_term,
        3.0 * (count_a - 1.0),
        out=np.full_like(count_a, np.inf),
        where=usable,
    )
    radius = variance_part + range_part
    standard_error = np.sqrt(
        np.divide(
            variance_a,
            count_a,
            out=np.full_like(variance_a, np.inf),
            where=count_a >= 1.0,
        )
    )
    interval = EvidenceInterval(
        mean=mean_a,
        lower=mean_a - radius,
        upper=mean_a + radius,
        standard_error=standard_error,
        count=count_a,
        dof=np.maximum(count_a - 1.0, 0.0),
        alpha=float(alpha),
        method="empirical_bernstein",
    )
    interval.validate()
    return interval


def eb_confidence_sequence(
    values: object,
    *,
    alpha: float,
    lower_bound: float,
    upper_bound: float,
) -> tuple[float, float]:
    """Running-intersection empirical-Bernstein confidence sequence.

    Returns the final (lower, upper) interval for the mean of a bounded
    stream, valid simultaneously over every prefix, so any stopping rule may
    consume intermediate intervals without invalidating the guarantee.  The
    stream is rescaled to [0,1] internally; endpoints are mapped back.
    """

    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("confidence sequence requires alpha in (0,1)")
    low, high = float(lower_bound), float(upper_bound)
    if not (math.isfinite(low) and math.isfinite(high)) or high <= low:
        raise ValueError("confidence sequence requires a finite nondegenerate range")
    stream = np.asarray(values, dtype=np.float64).reshape(-1)
    if stream.size == 0 or not np.isfinite(stream).all():
        raise ValueError("confidence sequence requires finite observations")
    if np.any(stream < low - 1e-12) or np.any(stream > high + 1e-12):
        raise ValueError("confidence sequence observations left their declared range")
    scaled = np.clip((stream - low) / (high - low), 0.0, 1.0)
    log_term = math.log(2.0 / float(alpha))
    weighted_sum = 0.0
    weight_total = 0.0
    psi_total = 0.0
    running_mean = 0.5
    running_second = 0.25
    count = 0
    lower_final = 0.0
    upper_final = 1.0
    for observation in scaled:
        count += 1
        predictable_sd = math.sqrt(max(running_second, 1e-12))
        lam = min(
            math.sqrt(2.0 * log_term / (predictable_sd * predictable_sd * count * math.log(count + 1.0)))
            if count > 0
            else 0.5,
            0.5,
        )
        lam = max(lam, 1e-6)
        deviation = observation - running_mean
        psi_total += (deviation * deviation) * (-math.log1p(-lam) - lam)
        weighted_sum += lam * observation
        weight_total += lam
        margin = (log_term + psi_total) / max(weight_total, 1e-12)
        center = weighted_sum / max(weight_total, 1e-12)
        lower_final = max(lower_final, center - margin)
        upper_final = min(upper_final, center + margin)
        running_mean = (0.5 + float(np.sum(scaled[:count]))) / (count + 1.0)
        running_second = (0.25 + float(np.sum((scaled[:count] - running_mean) ** 2))) / (
            count + 1.0
        )
    if lower_final > upper_final:
        midpoint = 0.5 * (lower_final + upper_final)
        lower_final = upper_final = midpoint
    return (
        low + lower_final * (high - low),
        low + upper_final * (high - low),
    )


__all__ = [
    "EVIDENCE_FOUNDATION_V1",
    "EvidenceInterval",
    "eb_confidence_sequence",
    "empirical_bernstein_interval",
    "staged_alpha",
    "student_t_interval",
]
