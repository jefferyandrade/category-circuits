"""Sample-size forecasting primitives for sequential acquisition criteria.

A sequential acquisition loop answers two separate questions each look: how to
SPLIT the next purchase across strata, and how MANY units to buy at all. The
first has a principled answer in this tree already -- ``UnifiedNeymanAllocator``
allocates by marginal variance reduction per unit cost. The second, in the
census loops, did not: they hand the allocator a total chosen by a fixed
schedule (double while unstable, add a packet while stable), so the allocator
splits a size it was never allowed to choose.

This module supplies the missing half in a criterion-agnostic form. A criterion
is anything that is monotone in the cumulative count and is compared against a
tolerance; given a predicate expressing it, ``smallest_count_meeting_tolerance``
inverts it exactly.

TWO DESIGN RULES, both learned from measured failures.

**Bisect the predicate; never re-derive it analytically.** The obvious
implementation of a sign-bar forecast computes a per-route certification count
``n_r`` in closed form and sorts. That requires transcribing the standard-error
model into the forecaster, where it can silently drift from the model the bar
actually evaluates -- and the finite-population correction is exactly the term
an analytic re-derivation drops. Evaluating the loop's own predicate at a
candidate count makes forecaster/bar divergence structurally impossible. The
cost is trivial: ~17 bisection steps over a 1.39M-route axis is sub-second
against a look measured in minutes.

**Carry the finite-population correction.** MEASURED: ``decision_snapshot``
multiplies every standard error by ``max(0, (N - n) / max(N - 1, 1))``, so at
``n == N`` every standard error is exactly zero and the sign bar's unresolved
set empties by construction. The sign bar therefore ALWAYS resolves over a
sealed census, and a forecast that drops the correction reports ``n = infinity``
for routes whose estimated mean sits near zero -- concluding "unreachable"
precisely where the truth is "buy the remaining census". Any futility claim
built on the uncorrected form is wrong in the dangerous direction.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from statistics import NormalDist

import numpy as np

__all__ = [
    "smallest_count_meeting_tolerance",
    "finite_population_correction",
    "stratified_mean_standard_error_at",
    "sigma_from_snapshot",
    "unresolved_activity_fraction",
    "expected_material_misresolution_fraction",
    "expected_wrongsign_consequence_fraction",
    "wrongsign_consequence_upper_bound_fraction",
    "deciding_oracle_resolution_floor",
    "normal_mixture_boundary",
    "hoeffding_variance_process",
    "feasible_mean_interval",
    "anytime_valid_mean_interval",
]


def normal_mixture_boundary(
    *,
    variance_process: np.ndarray | float,
    alpha: float,
    rho: float,
) -> np.ndarray:
    """Robbins normal-mixture boundary on the SUM scale, valid at every ``n``.

    Derivation, so the closed form is checkable rather than cited. The mixture
    martingale over a normal prior with variance ``1/rho`` is

        M_t = sqrt( rho / (V_t + rho) ) * exp( S_t^2 / (2 (V_t + rho)) )

    which is a nonnegative martingale with ``M_0 = 1``. Ville's inequality gives
    ``P(exists t : M_t >= 1/alpha) <= alpha``, and inverting ``M_t >= 1/alpha``:

        |S_t| >= sqrt( (V_t + rho) * log( (V_t + rho) / (rho * alpha^2) ) )

    so the right-hand side is a boundary crossed with probability at most
    ``alpha`` **over the whole trajectory**, not at a fixed ``t``. That is what
    licenses looking after every purchase and choosing what to buy next.

    **This form is the reason a mixture is used rather than a betting
    sequence.** It depends on the data only through the variance process
    ``V_t``, which is a sufficient statistic; a betting sequence accumulates
    ``prod_t (1 + lambda_t (X_t - m))`` with a history-dependent ``lambda_t``
    and does not reduce. One Python object per route across ~55M routes x 2
    heads x 2 strata is impossible, so vectorizing over sufficient statistics
    is not a preference, it is the only viable construction.

    ``rho`` tunes where the boundary is tightest; it is a free parameter of the
    prior, not of the data, so any fixed choice keeps validity and only trades
    tightness across the sample-size range.

    **VALIDITY IS CONDITIONAL, and the condition is not decorative.** The
    mixture is sub-Gaussian, so ``variance_process`` must be a *predictable*
    sub-Gaussian variance proxy. Feeding it the *realized* empirical variance
    silently assumes sub-Gaussian increments, and MEASURED under adversarial
    stopping at alpha = 0.05 that assumption fails hard on heavy tails:

        gaussian 0.023 | uniform 0.033 | t(3) 0.025 | exp(1)-1 0.048
        LOGNORMAL 0.245  <-- five times the nominal rate

    Use :func:`hoeffding_variance_process` for bounded observations, which is
    the regime this programme measures in and which yields a deterministic --
    hence trivially predictable -- proxy. The empirical-variance path is kept
    for tighter intervals where sub-Gaussianity is independently established,
    and it is the caller's obligation to establish it.
    """

    variance = np.asarray(variance_process, dtype=np.float64)
    rho_value = float(rho)
    alpha_value = float(alpha)
    if not math.isfinite(rho_value) or rho_value <= 0.0:
        raise ValueError("mixture boundary requires a positive rho")
    if not 0.0 < alpha_value < 1.0:
        raise ValueError("mixture boundary requires alpha in (0,1)")
    if np.any(variance < 0.0) or not np.all(np.isfinite(variance)):
        raise ValueError("variance process must be finite and nonnegative")
    shifted = variance + rho_value
    # shifted/(rho*alpha^2) >= 1/alpha^2 > 1, so the log is strictly positive.
    return np.sqrt(shifted * np.log(shifted / (rho_value * alpha_value**2)))


def hoeffding_variance_process(
    *,
    count: int,
    value_lower: float,
    value_upper: float,
) -> float:
    """Deterministic sub-Gaussian variance proxy for bounded observations.

    For ``X`` confined to ``[a, b]``, Hoeffding's lemma gives a sub-Gaussian
    parameter ``(b - a) / 2`` per observation, so ``V_n = n ((b-a)/2)^2``.

    Being deterministic it is **predictable by construction**, which is what the
    mixture martingale requires and what the realized empirical variance does
    not supply. It is conservative -- it ignores that the actual spread may be
    far smaller than the range permits -- and conservative is the correct
    direction for a quantity that licenses stopping whenever one likes.
    """

    span = float(value_upper) - float(value_lower)
    if not math.isfinite(span) or span <= 0.0:
        raise ValueError("Hoeffding proxy requires a positive value range")
    if int(count) < 0:
        raise ValueError("Hoeffding proxy requires a nonnegative count")
    return float(int(count)) * (span / 2.0) ** 2


def feasible_mean_interval(
    *,
    observed_total: np.ndarray | float,
    count: int,
    population: int,
    value_lower: float,
    value_upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Population means consistent with what has been seen and what has not.

    With ``count`` of ``population`` values observed summing to
    ``observed_total``, the unseen remainder lies in
    ``[value_lower, value_upper]`` per item, so the population mean is confined
    to a hard interval **by arithmetic, with no probability involved**.

    At ``count == population`` the interval is a single point. This is why
    exactness over a sealed census is structural here rather than a correction
    term that happens to vanish: nothing is being approximated.
    """

    total = np.asarray(observed_total, dtype=np.float64)
    seen = int(count)
    size = int(population)
    if seen < 0 or size <= 0 or seen > size:
        raise ValueError("feasible mean interval has invalid support")
    low, high = float(value_lower), float(value_upper)
    if not (math.isfinite(low) and math.isfinite(high)) or high < low:
        raise ValueError("feasible mean interval requires a valid value range")
    remaining = size - seen
    return (
        (total + remaining * low) / float(size),
        (total + remaining * high) / float(size),
    )


def anytime_valid_mean_interval(
    *,
    observed_total: np.ndarray,
    variance_process: np.ndarray,
    count: int,
    population: int,
    alpha: float,
    rho: float,
    value_lower: float,
    value_upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The region: a mixture boundary intersected with what is arithmetically possible.

    Two guarantees of different kinds, and intersecting them keeps both: the
    boundary holds with probability ``1 - alpha`` uniformly over ``n``, and the
    feasible interval holds always. The intersection therefore inherits
    anytime-validity from the first and exactness-at-census from the second.
    """

    total = np.asarray(observed_total, dtype=np.float64)
    seen = max(int(count), 1)
    radius = normal_mixture_boundary(
        variance_process=variance_process, alpha=alpha, rho=rho
    ) / float(seen)
    centre = total / float(seen)
    lower, upper = centre - radius, centre + radius
    feasible_low, feasible_high = feasible_mean_interval(
        observed_total=total,
        count=count,
        population=population,
        value_lower=value_lower,
        value_upper=value_upper,
    )
    return np.maximum(lower, feasible_low), np.minimum(upper, feasible_high)


def anytime_valid_critical_value(
    *,
    count: int,
    alpha: float,
    rho: float,
    value_lower: float,
    value_upper: float,
) -> float:
    """The mixture boundary expressed as a ``z``, so it can replace a ladder.

    The census multiplies a standard error by a constant ``z`` computed once,
    before the loop. A constant ``z`` must pre-pay for every look it will ever
    be used at, which is precisely why the incumbent splits alpha across
    ``maximum_looks`` planned looks -- a **flat Bonferroni ladder in time**.
    That ladder is why the census pays ``z = 7.1092`` where a single look would
    pay ``6.2334``.

    The mixture boundary needs no such split: it is valid at **every** ``n``
    simultaneously. Written against the Hoeffding standard error
    ``sigma_H = (b - a) / (2 sqrt(n))`` it reduces to a pure function of the
    variance process, because ``V_n = n sigma_H^2 n / n`` gives

        z_eff(n) = B(V_n) / sqrt(V_n) = sqrt( (1 + rho/V_n) log( (V_n + rho) / (rho alpha^2) ) )

    which grows like ``sqrt(log n)`` rather than paying a fixed ladder toll.

    **VALIDITY CONDITION -- the returned number is a ``z`` FOR THE HOEFFDING
    STANDARD ERROR, not for an arbitrary one.** ``z_eff * sigma_H`` equals the
    boundary radius ``B(V_n)/n`` exactly; multiplying it by a *measured*
    standard error instead gives something SMALLER, because for observations
    confined to ``[a, b]`` the sample deviation is at most ``(b - a) / 2``. The
    error would therefore be in the anticonservative direction and would not
    announce itself. Callers wanting the tight, realized-variance version need
    an empirical-Bernstein confidence sequence, whose boundedness assumption
    replaces sub-Gaussianity -- not this function with a smaller ``sigma``.

    Prefer :func:`anytime_valid_mean_interval`, which returns the radius and
    cannot be misapplied this way. This exists for the one caller that is
    structurally z-shaped.
    """

    seen = int(count)
    if seen <= 0:
        raise ValueError("anytime-valid critical value requires count >= 1")
    variance = hoeffding_variance_process(
        count=seen, value_lower=value_lower, value_upper=value_upper
    )
    boundary = float(
        normal_mixture_boundary(
            variance_process=variance, alpha=alpha, rho=rho
        )
    )
    return boundary / math.sqrt(variance)


def deciding_oracle_resolution_floor(
    *,
    context_standard_error: float,
    context_count: int,
    panel_rail: int,
    population_size: int,
    confidence_level: float,
) -> float:
    """Smallest damage increment the deciding oracle can resolve.

    This is a **derived** quantity, not a declared tolerance, and that is the
    whole point: ambiguity whose induced damage sits below what the deciding
    oracle could measure cannot change any decision the field feeds, so
    certifying an ordering more finely than this buys nothing.

    The formula is the half-width the oracle would achieve at its maximum
    panel: recover the per-context spread from a certificate's standard error,
    project it to the panel rail, apply the finite-population correction, and
    scale by the two-sided normal critical value.

    Transcribed from the v1 trust-walk chart builder so both paths share one
    derivation. v0's census abandoned this principle in favour of an absolute
    activity-mass tolerance, which is the origin of the granularity
    incoherence: 0.001 of global activity means "up to 13 routes" at kernel
    granularity and "less than one route" at coarse.
    """

    spread = float(context_standard_error) * math.sqrt(
        max(int(context_count), 1)
    )
    if not math.isfinite(spread) or spread <= 0.0:
        raise ValueError(
            "resolution floor requires a positive measured context spread"
        )
    rail = max(int(panel_rail), 1)
    standard_error = spread / math.sqrt(rail)
    population = int(population_size)
    if population > 0:
        # The oracle cannot be more uncertain than the population permits; at
        # a rail covering the whole population the floor collapses to zero.
        population_fraction = min(rail / float(population), 1.0)
        standard_error *= math.sqrt(max(1.0 - population_fraction, 0.0))
    tail = max(1.0 - float(confidence_level), 1e-12)
    return float(NormalDist().inv_cdf(1.0 - 0.5 * tail)) * standard_error


def smallest_count_meeting_tolerance(
    *,
    predicate: Callable[[int], bool],
    current_count: int,
    maximum_count: int,
    minimum_increment: int = 1,
) -> int | None:
    """Smallest count in ``(current, maximum]`` at which ``predicate`` holds.

    ``predicate`` must be monotone: once true at some count it stays true for
    every larger count. Returns ``None`` when it does not hold even at
    ``maximum_count``, which is the caller's signal that the criterion is not
    reachable within the available population -- a budget statement, not a
    statistical one.

    The returned count is raised to ``current_count + minimum_increment`` so a
    caller can never be handed a target that buys nothing. Bisection is exact
    for a step function, so no smoothing or envelope is required.
    """

    count = int(current_count)
    maximum = int(maximum_count)
    increment = max(1, int(minimum_increment))
    if count < 0 or maximum < count:
        raise ValueError("invalid forecast support")
    floor = min(count + increment, maximum)
    if maximum == count:
        return None
    if predicate(floor):
        return floor
    if not predicate(maximum):
        return None
    # Invariant: predicate(low) is False, predicate(high) is True.
    low, high = floor, maximum
    while high - low > 1:
        middle = (low + high) // 2
        if predicate(middle):
            high = middle
        else:
            low = middle
    return high


def finite_population_correction(
    *,
    count: int,
    population: int,
) -> float:
    """``(N - n) / max(N - 1, 1)``, clamped at zero.

    Transcribed from ``DeviceTripartiteMomentAccumulator.decision_snapshot`` so
    the forecast and the bar share one definition. It reaches exactly zero at
    ``n == N``: over a complete sealed census the estimand is exact.
    """

    sampled = int(count)
    total = int(population)
    if sampled <= 0 or total <= 0 or sampled > total:
        raise ValueError("invalid finite-population inputs")
    return max(0.0, (total - sampled) / max(total - 1, 1))


def sigma_from_snapshot(
    *,
    standard_error: np.ndarray,
    count: int,
    population: int,
) -> np.ndarray:
    """Recover per-route contextual sd from the standard error a bar reads.

    Anchoring to the realized ``mean_standard_error`` array rather than to
    ``contextual_variance`` is deliberate: the forecast is then tied to the
    exact quantity the criterion consumes, including whatever weighting the
    accumulator applied, so the two cannot disagree.

    Under the equal-weight panel this loop delivers, the snapshot's standard
    error reduces to ``sigma^2 / (n - 1) * fpc``, so ``sigma`` is recovered by
    inverting that. Returns zeros where the correction has already collapsed
    (``n == N``), which carries no information and must not be extrapolated.
    """

    sampled = int(count)
    total = int(population)
    errors = np.asarray(standard_error, dtype=np.float64)
    if sampled <= 1:
        return np.zeros_like(errors)
    fpc = finite_population_correction(count=sampled, population=total)
    if fpc <= 0.0:
        return np.zeros_like(errors)
    return np.sqrt(
        np.maximum(errors * errors, 0.0) * float(sampled - 1) / fpc
    )


def stratified_mean_standard_error_at(
    *,
    sigma: np.ndarray,
    count: int,
    population: int,
) -> np.ndarray:
    """Forward standard-error model at a hypothetical cumulative ``count``.

    The exact inverse of :func:`sigma_from_snapshot`, so
    ``stratified_mean_standard_error_at(sigma_from_snapshot(se, n, N), n, N)``
    reproduces ``se``. Returns exact zeros at ``count == population``.
    """

    sampled = int(count)
    total = int(population)
    values = np.asarray(sigma, dtype=np.float64)
    if sampled <= 1:
        return np.full_like(values, np.inf)
    fpc = finite_population_correction(count=sampled, population=total)
    if fpc <= 0.0:
        return np.zeros_like(values)
    return np.sqrt(
        np.maximum(values * values, 0.0) * fpc / float(sampled - 1)
    )


def unresolved_activity_fraction(
    *,
    mean: np.ndarray,
    standard_error: np.ndarray,
    activity: np.ndarray,
    critical: float,
    materiality_floor: float | None = None,
) -> float:
    """Activity share of routes whose signed mean has not resolved.

    Lifted VERBATIM from the census loop's own computation, including the two
    clauses that are easy to lose in a re-derivation:

    * ``standard_error > 0`` excludes exact zero-variance ties. Those are known
      zeros using the declared nonnegative tie class, not statistically
      unresolved routes -- and this clause is what empties the set when the
      finite-population correction collapses at full census.
    * the ``tiny`` mass guard returns 0.0 rather than dividing by zero when the
      activity field carries no mass at all.

    ANTI-DRIFT CONTRACT: evaluated on a look's own arrays this must reproduce
    that look's ``risks[view]`` bit for bit. A test pins it.
    """

    means = np.asarray(mean, dtype=np.float64)
    errors = np.asarray(standard_error, dtype=np.float64)
    weights = np.asarray(activity, dtype=np.float64)
    ambiguous = np.abs(means) <= float(critical) * errors
    ambiguous &= errors > 0.0
    if materiality_floor is not None:
        floor = float(materiality_floor)
        if not math.isfinite(floor) or floor <= 0.0:
            raise ValueError("materiality floor must be finite and positive")
        ambiguous &= (np.abs(means) + float(critical) * errors) >= floor
    mass = float(np.sum(weights))
    if mass <= np.finfo(np.float64).tiny:
        return 0.0
    return float(np.sum(weights[ambiguous]) / mass)


def expected_material_misresolution_fraction(
    *,
    mean: np.ndarray,
    standard_error: np.ndarray,
    activity: np.ndarray,
    critical: float,
    materiality_floor: float,
) -> float:
    """Expected materially wrong-signed activity share.

    This Gaussian plug-in quantity is acquisition telemetry only.  It may
    prioritize the next purchase, but it is never a stopping certificate.
    Closure uses :func:`wrongsign_consequence_upper_bound_fraction` instead.
    """

    from scipy.special import ndtr

    means = np.asarray(mean, dtype=np.float64)
    errors = np.asarray(standard_error, dtype=np.float64)
    weights = np.asarray(activity, dtype=np.float64)
    floor = float(materiality_floor)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("decision-referenced census requires a positive floor")
    mass = float(np.sum(weights))
    if mass <= np.finfo(np.float64).tiny:
        return 0.0
    contributing = (errors > 0.0) & (
        (np.abs(means) + float(critical) * errors) >= floor
    )
    if not np.any(contributing):
        return 0.0
    z = np.abs(means[contributing]) / errors[contributing]
    return float(np.sum(weights[contributing] * ndtr(-z)) / mass)


def expected_wrongsign_consequence_fraction(
    *,
    mean: np.ndarray,
    standard_error: np.ndarray,
    activity: np.ndarray,
) -> float:
    """Gaussian plug-in expected attainable wrong-sign consequence.

    The numerator is twice the activity-weighted expected wrong-side
    magnitude of each route.  This is useful for acquisition ordering and
    telemetry, but remains explicitly model-based and cannot authorize stop.
    """

    from scipy.special import ndtr

    means = np.asarray(mean, dtype=np.float64)
    errors = np.asarray(standard_error, dtype=np.float64)
    weights = np.asarray(activity, dtype=np.float64)
    magnitude = np.abs(means)
    consequence_mass = float(np.sum(weights * magnitude))
    if consequence_mass <= np.finfo(np.float64).tiny:
        return 0.0
    live = errors > 0.0
    if not np.any(live):
        return 0.0
    z = magnitude[live] / errors[live]
    phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    wrong_side = errors[live] * phi - magnitude[live] * ndtr(-z)
    wrong_side = np.maximum(wrong_side, 0.0)
    return float(np.sum(2.0 * weights[live] * wrong_side)) / consequence_mass


def wrongsign_consequence_upper_bound_fraction(
    *,
    mean: np.ndarray,
    standard_error: np.ndarray,
    activity: np.ndarray,
    critical: float,
) -> float:
    """Simultaneous-CI upper bound on attainable wrong-sign consequence.

    On the simultaneous confidence event, a route whose interval excludes
    zero cannot be wrong-signed.  A straddling route can contribute at most
    ``2 * activity * max(0, critical * se - abs(mean))``.  The finite-
    population correction makes this bound exactly zero at the sealed census.
    """

    means = np.asarray(mean, dtype=np.float64)
    errors = np.asarray(standard_error, dtype=np.float64)
    weights = np.asarray(activity, dtype=np.float64)
    magnitude = np.abs(means)
    consequence_mass = float(np.sum(weights * magnitude))
    if consequence_mass <= np.finfo(np.float64).tiny:
        return 0.0
    wrong_side = np.maximum(0.0, float(critical) * errors - magnitude)
    return float(np.sum(2.0 * weights * wrong_side)) / consequence_mass


def capacity_reachability_ratio(
    *,
    count: int,
    population: int,
) -> float:
    """How much standard-error contraction the remaining census can still buy.

    ``sqrt( (fpc(n)/(n-1)) / (fpc(N-1)/(N-2)) )`` -- the factor by which every
    route's standard error would shrink if the entire remaining population were
    purchased. This is the honest replacement for a statistical futility test
    on the sign bar: since the correction drives every standard error to zero
    at full census, nothing is ever statistically unreachable, and the only
    real question is how much precision is left to buy. Returns ``inf`` when
    the census can still collapse the error to zero.
    """

    sampled = int(count)
    total = int(population)
    if sampled <= 1 or total <= sampled:
        return 1.0
    current = finite_population_correction(
        count=sampled, population=total
    ) / float(sampled - 1)
    if current <= 0.0:
        return 1.0
    return float(math.sqrt(current / max(_terminal_variance_scale(total), 1e-300)))


def _terminal_variance_scale(population: int) -> float:
    """Variance scale one short of the full census (the last informative n)."""

    total = int(population)
    if total <= 2:
        return 0.0
    return finite_population_correction(
        count=total - 1, population=total
    ) / float(total - 2)
