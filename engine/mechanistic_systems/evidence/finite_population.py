"""Anytime-valid betting confidence sequences for finite populations.

For a finite population of ``N`` bounded values with mean ``mu``, uniform
sampling without replacement has the predictable null conditional mean

``m_t(mu) = (N * mu - sum_{i<t} X_i) / (N - t + 1)``.

For every fixed, safely bounded bet ``lambda``,

``prod_t (1 + lambda * (X_t - m_t(mu)))``

is a nonnegative test martingale under the null.  This module mixes a frozen
grid of positive and negative bets. In exact arithmetic, inversion gives an
interval valid at every stopping time under that sampling law. ``interval``
retains the ordinary float64 reference inversion. ``certified_interval`` uses
it only to propose endpoints and requires directed arithmetic to certify each
narrowing for the exact supplied binary64 values and realised bet grid.
Sampling all ``N`` units reveals the finite-population mean exactly; certified
census endpoints enclose that rational mean with outward binary64 rounding.

Adaptive Neyman allocation may choose the next *stratum* from prior
information.  Each stratum retains its sealed within-stratum order and its own
e-process; a Bonferroni alpha split plus weighted projection gives the global
contrast interval.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Context, Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from fractions import Fraction

import numpy as np
from scipy.special import logsumexp

from .glocal_contracts import SamplingDesignSpec, content_hash


WOR_BETTING_CS_METHOD = "bounded_wor_predictable_betting_mixture_v1"
STRATIFIED_WOR_CS_METHOD = "stratified_bounded_wor_weighted_projection_v1"

_DEFAULT_NORMALIZED_BETS = (0.03125, 0.0625, 0.125, 0.25, 0.5, 0.8)
_ROOT_ITERATIONS = 72
_FEASIBILITY_TOLERANCE = 1e-13
_CERTIFICATION_PRECISIONS = (96, 160, 256, 384)
_CERTIFICATION_EMAX = 999_999_999
_CERTIFICATION_EMIN = -999_999_999


def _fraction_from_float(value: float, *, name: str) -> Fraction:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return Fraction.from_float(number)


def _fraction_to_float_outward(
    value: Fraction, *, direction: str
) -> float:
    rounded = float(value)
    if not math.isfinite(rounded):
        raise ValueError("finite-population endpoint overflowed binary64")
    exact = Fraction.from_float(rounded)
    if direction == "lower" and exact > value:
        return math.nextafter(rounded, -math.inf)
    if direction == "upper" and exact < value:
        return math.nextafter(rounded, math.inf)
    return rounded


def _decimal_bound(value: Fraction, *, precision: int, direction: str) -> Decimal:
    with localcontext() as context:
        context.prec = int(precision)
        context.Emax = _CERTIFICATION_EMAX
        context.Emin = _CERTIFICATION_EMIN
        context.rounding = ROUND_FLOOR if direction == "lower" else ROUND_CEILING
        return Decimal(value.numerator) / Decimal(value.denominator)


def _is_sha256(value: str) -> bool:
    identity = str(value)
    return len(identity) == 64 and all(
        character in "0123456789abcdef" for character in identity
    )


def finite_population_feasible_mean_bounds(
    *, population_size: int, sample_size: int,
    observed_lower_sum: Fraction, observed_upper_sum: Fraction,
    lower_bound: Fraction, upper_bound: Fraction,
) -> tuple[Fraction, Fraction]:
    """Exact completion bounds for distinct observed population units.

    The caller owns identities and valid point/support enclosures. This pure
    arithmetic needs no random-sampling assumption or alpha: unseen units can
    take any value in the registered support. It therefore remains valid under
    arbitrary stopping and yields the numerical census envelope at n=N.
    """
    if (type(population_size) is not int or population_size <= 0
            or type(sample_size) is not int or not 0 <= sample_size <= population_size):
        raise ValueError("finite-population completion requires valid exact counts")
    values=(observed_lower_sum,observed_upper_sum,lower_bound,upper_bound)
    if any(type(v) is not Fraction for v in values):
        raise TypeError("finite-population completion requires exact rational endpoints")
    if (lower_bound > upper_bound or observed_lower_sum > observed_upper_sum
            or observed_lower_sum < sample_size*lower_bound
            or observed_upper_sum > sample_size*upper_bound):
        raise ValueError("observed interval sums exceed registered population support")
    remaining=population_size-sample_size
    return ((observed_lower_sum+remaining*lower_bound)/population_size,
            (observed_upper_sum+remaining*upper_bound)/population_size)


@dataclass(frozen=True, slots=True)
class FinitePopulationCSSnapshot:
    stratum_id: str
    population_size: int
    sample_size: int
    sample_mean: float | None
    lower: float
    upper: float
    alpha: float
    value_lower_bound: float
    value_upper_bound: float
    full_census: bool
    empty: bool
    sealed_order_hash: str
    observed_context_ids: tuple[str, ...]
    claim_bearing: bool = True
    method: str = WOR_BETTING_CS_METHOD

    def __post_init__(self) -> None:
        if not self.stratum_id:
            raise ValueError("finite-population CS requires stratum identity")
        if not _is_sha256(self.sealed_order_hash):
            raise ValueError("finite-population CS requires a sealed-order SHA-256")
        contexts = tuple(str(value) for value in self.observed_context_ids)
        if len(contexts) != len(set(contexts)):
            raise ValueError("finite-population CS repeated a context identity")
        if self.claim_bearing and len(contexts) != self.sample_size:
            raise ValueError(
                "claim-bearing finite-population CS must identify every observed context"
            )
        if not self.claim_bearing and contexts and len(contexts) != self.sample_size:
            raise ValueError("nonclaim CS context prefix is partially identified")
        if (
            self.population_size <= 0
            or not 0 <= self.sample_size <= self.population_size
        ):
            raise ValueError("finite-population CS sample counts are invalid")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("finite-population CS alpha must lie in (0,1)")
        if not self.value_lower_bound < self.value_upper_bound:
            raise ValueError("finite-population CS requires a nondegenerate range")
        if not self.empty and self.lower > self.upper:
            raise ValueError("nonempty finite-population CS has inverted bounds")
        if self.empty and self.lower <= self.upper:
            raise ValueError(
                "empty finite-population CS must retain its separating bounds"
            )
        if self.sample_mean is not None and not (
            self.value_lower_bound - 1e-12
            <= self.sample_mean
            <= self.value_upper_bound + 1e-12
        ):
            raise ValueError("finite-population sample mean left its declared range")
        if self.full_census and (
            self.sample_mean is None
            or self.empty
            or not math.isclose(self.lower, self.sample_mean, abs_tol=1e-12)
            or not math.isclose(self.upper, self.sample_mean, abs_tol=1e-12)
        ):
            raise ValueError("full census must collapse exactly to its observed mean")
        if self.method != WOR_BETTING_CS_METHOD:
            raise ValueError("unsupported finite-population confidence-sequence method")
        object.__setattr__(self, "observed_context_ids", contexts)

    @property
    def content_hash(self) -> str:
        return content_hash(self)


class BoundedWoRBettingCS:
    """One bounded sampling-without-replacement betting confidence sequence."""

    def __init__(
        self,
        *,
        stratum_id: str,
        population_size: int,
        alpha: float,
        lower_bound: float = -1.0,
        upper_bound: float = 1.0,
        sealed_context_order: Sequence[str] = (),
        sealed_order_hash: str | None = None,
        claim_bearing: bool = True,
        normalized_bets: Sequence[float] = _DEFAULT_NORMALIZED_BETS,
    ) -> None:
        if not str(stratum_id):
            raise ValueError("bounded WoR CS requires stratum_id")
        size = int(population_size)
        if size <= 0:
            raise ValueError("bounded WoR CS population_size must be positive")
        alpha_f = float(alpha)
        if not 0.0 < alpha_f < 1.0:
            raise ValueError("bounded WoR CS alpha must lie in (0,1)")
        low, high = float(lower_bound), float(upper_bound)
        if not (math.isfinite(low) and math.isfinite(high)) or high <= low:
            raise ValueError("bounded WoR CS requires finite nondegenerate bounds")
        span = high - low
        if not math.isfinite(span) or span <= 0.0:
            raise ValueError("bounded WoR CS requires a finite support span")
        bets = np.asarray(
            tuple(float(value) for value in normalized_bets), dtype=np.float64
        )
        if bets.ndim != 1 or bets.size == 0:
            raise ValueError("bounded WoR CS requires a nonempty bet grid")
        if not np.isfinite(bets).all() or np.any(bets <= 0.0) or np.any(bets >= 1.0):
            raise ValueError("normalized bets must lie strictly inside (0,1)")
        contexts = tuple(str(value) for value in sealed_context_order)
        if claim_bearing and not contexts:
            raise ValueError(
                "claim-bearing WoR confidence sequences require exact sealed context order"
            )
        if contexts and (len(contexts) != size or len(set(contexts)) != size):
            raise ValueError("sealed context order must enumerate the population once")
        expected_order_hash = content_hash(
            {"stratum_id": str(stratum_id), "context_ids": contexts}
        )
        order_hash = str(sealed_order_hash or expected_order_hash)
        if not _is_sha256(order_hash):
            raise ValueError("bounded WoR CS requires a sealed-order SHA-256")
        if contexts and order_hash != expected_order_hash:
            raise ValueError("bounded WoR sealed context order differs from its hash")

        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            positive_bets = bets / span
        if (
            not np.isfinite(positive_bets).all()
            or np.any(positive_bets <= 0.0)
        ):
            raise ValueError("bounded WoR CS produced nonfinite bet rates")
        lower_fraction = _fraction_from_float(low, name="lower bound")
        upper_fraction = _fraction_from_float(high, name="upper bound")
        span_fraction = upper_fraction - lower_fraction
        if span_fraction <= 0:
            raise ValueError("bounded WoR CS requires an exact positive support span")
        bet_fractions = tuple(
            Fraction.from_float(float(value)) for value in positive_bets
        )
        if any(value * span_fraction >= 1 for value in bet_fractions):
            raise ValueError("bounded WoR CS bet rates are unsafe for the support span")

        self.stratum_id = str(stratum_id)
        self.population_size = size
        self.alpha = alpha_f
        self.lower_bound = low
        self.upper_bound = high
        self._lower_fraction = lower_fraction
        self._upper_fraction = upper_fraction
        self._span_fraction = span_fraction
        self._positive_bets = positive_bets
        self._positive_bet_fractions = bet_fractions
        self._sealed_context_order = contexts
        self.sealed_order_hash = order_hash
        self.claim_bearing = bool(claim_bearing)
        self._values: list[float] = []
        self._context_ids: list[str] = []

    @property
    def sample_size(self) -> int:
        return len(self._values)

    @property
    def values(self) -> tuple[float, ...]:
        return tuple(self._values)

    @property
    def context_ids(self) -> tuple[str, ...]:
        return tuple(self._context_ids)

    def record(self, value: float, *, context_id: str | None = None) -> None:
        """Append one sealed observation without repeatedly inverting the CS."""

        if self.sample_size >= self.population_size:
            raise RuntimeError(
                "bounded WoR CS cannot sample beyond its finite population"
            )
        observation = float(value)
        if (
            not math.isfinite(observation)
            or observation < self.lower_bound
            or observation > self.upper_bound
        ):
            raise ValueError("bounded WoR observation left its declared range")
        if self.claim_bearing and context_id is None:
            raise ValueError(
                "claim-bearing WoR updates require the exact next context identity"
            )
        expected_context = (
            self._sealed_context_order[self.sample_size]
            if self._sealed_context_order
            else None
        )
        if expected_context is not None:
            if context_id is None:
                raise ValueError(
                    "sealed WoR sampling requires the next context identity"
                )
            if str(context_id) != expected_context:
                raise ValueError(
                    f"unpredictable within-stratum draw: expected {expected_context!r}, "
                    f"received {str(context_id)!r}"
                )
        elif context_id is not None and str(context_id) in self._context_ids:
            raise ValueError("sampling without replacement repeated a context identity")
        self._values.append(observation)
        if context_id is not None:
            self._context_ids.append(str(context_id))

    def update(
        self, value: float, *, context_id: str | None = None
    ) -> FinitePopulationCSSnapshot:
        self.record(value, context_id=context_id)
        return self.interval()

    def _feasible_mean_interval(self) -> tuple[float, float]:
        count = self.sample_size
        if count == 0:
            return self.lower_bound, self.upper_bound
        total = math.fsum(self._values)
        remaining = self.population_size - count
        feasible_lower = (total + remaining * self.lower_bound) / self.population_size
        feasible_upper = (total + remaining * self.upper_bound) / self.population_size
        return (
            max(self.lower_bound, feasible_lower),
            min(self.upper_bound, feasible_upper),
        )

    def _feasible_mean_interval_fraction(self) -> tuple[Fraction, Fraction]:
        count = self.sample_size
        if count == 0:
            return self._lower_fraction, self._upper_fraction
        total = sum(
            (Fraction.from_float(value) for value in self._values),
            Fraction(0, 1),
        )
        return finite_population_feasible_mean_bounds(
            population_size=self.population_size, sample_size=count,
            observed_lower_sum=total, observed_upper_sum=total,
            lower_bound=self._lower_fraction, upper_bound=self._upper_fraction,
        )

    def _certified_mixture_relation(
        self,
        candidate_mean: float,
        *,
        direction: str,
    ) -> str | None:
        """Return whether the exact realised bet law includes/excludes a mean.

        The result is relative to the exact binary64 values supplied to this
        object.  ``"included"`` means the mixture e-value is certified below
        ``2 / alpha``.  ``"excluded"`` means it is certified at or above that
        threshold.  ``None`` means Decimal interval arithmetic could not
        separate the comparison; callers must widen rather than decide.
        """

        if direction not in {"positive", "negative"}:
            raise ValueError("betting direction must be 'positive' or 'negative'")
        candidate = _fraction_from_float(candidate_mean, name="candidate mean")
        feasible_lower, feasible_upper = self._feasible_mean_interval_fraction()
        if candidate < feasible_lower or candidate > feasible_upper:
            return "excluded"
        if not self._values:
            return "included"
        values = tuple(Fraction.from_float(value) for value in self._values)
        alpha = Fraction.from_float(self.alpha)
        threshold = Fraction(2, 1) / alpha
        bet_fractions = (
            self._positive_bet_fractions
            if direction == "positive"
            else tuple(-value for value in self._positive_bet_fractions)
        )
        prefix = Fraction(0, 1)
        deviations: list[Fraction] = []
        for index, value in enumerate(values):
            conditional_mean = (self.population_size * candidate - prefix) / (
                self.population_size - index
            )
            deviations.append(value - conditional_mean)
            prefix += value

        for precision in _CERTIFICATION_PRECISIONS:
            lower_context = Context(prec=precision, rounding=ROUND_FLOOR,
                                    Emax=_CERTIFICATION_EMAX, Emin=_CERTIFICATION_EMIN)
            upper_context = Context(prec=precision, rounding=ROUND_CEILING,
                                    Emax=_CERTIFICATION_EMAX, Emin=_CERTIFICATION_EMIN)
            mixture_lower = Decimal(0)
            mixture_upper = Decimal(0)
            for bet in bet_fractions:
                product_lower = product_upper = Decimal(1)
                for deviation in deviations:
                    factor = 1 + bet * deviation
                    if factor <= 0:
                        # A supported feasible null and the strictly safe bet
                        # grid cannot do this. Never turn a failure into exclusion.
                        return None
                    numerator, denominator = Decimal(factor.numerator), Decimal(factor.denominator)
                    factor_lower = lower_context.divide(numerator, denominator)
                    factor_upper = upper_context.divide(numerator, denominator)
                    product_lower = lower_context.multiply(product_lower, factor_lower)
                    product_upper = upper_context.multiply(product_upper, factor_upper)
                mixture_lower = lower_context.add(mixture_lower, product_lower)
                mixture_upper = upper_context.add(mixture_upper, product_upper)
            divisor = Decimal(len(bet_fractions))
            mixture_lower = lower_context.divide(mixture_lower, divisor)
            mixture_upper = upper_context.divide(mixture_upper, divisor)
            threshold_lower = _decimal_bound(threshold, precision=precision, direction="lower")
            threshold_upper = _decimal_bound(threshold, precision=precision, direction="upper")
            if mixture_upper < threshold_lower:
                return "included"
            if mixture_lower >= threshold_upper:
                return "excluded"
        return None

    def _certified_one_sided_endpoint(
        self,
        *,
        direction: str,
        feasible_lower: float,
        feasible_upper: float,
        conservative_lower: float,
        conservative_upper: float,
    ) -> float:
        """Certify an outward endpoint; float inversion is only a proposal.

        For positive bets wealth decreases in the candidate mean on the exact
        feasible interval. A certified excluded candidate is therefore a safe
        lower endpoint. Negative bets give the symmetric upper endpoint. If
        certification cannot establish exclusion, retain the deterministic
        finite-population feasible bound. No epsilon changes the tested event.
        """
        if direction not in {"positive", "negative"}:
            raise ValueError("betting direction must be 'positive' or 'negative'")
        fallback = conservative_lower if direction == "positive" else conservative_upper
        exact_lower, exact_upper = self._feasible_mean_interval_fraction()
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                candidate = self._invert_one_side(
                    direction=direction, feasible_lower=feasible_lower,
                    feasible_upper=feasible_upper,
                    log_threshold=math.log(2.0) - math.log(self.alpha),
                )
        except (ArithmeticError, ValueError, RuntimeWarning):
            return fallback
        if not math.isfinite(candidate):
            return fallback
        outward = -math.inf if direction == "positive" else math.inf
        # A bounded number of representable outward proposals controls replay
        # cost. Exhaustion widens to the exact feasible interval; it cannot
        # turn uncertain arithmetic into a narrower confidence claim.
        for attempt in range(8):
            candidate_fraction = Fraction.from_float(candidate)
            if not exact_lower <= candidate_fraction <= exact_upper:
                return fallback
            if self._certified_mixture_relation(candidate, direction=direction) == "excluded":
                return candidate
            for _ in range(1 << attempt):
                candidate = math.nextafter(candidate, outward)
        return fallback

    def log_e_value(self, candidate_mean: float, *, direction: str) -> float:
        """Log mixture e-value for a candidate finite-population mean.

        ``direction='positive'`` bets on observations above the candidate and
        rejects candidate means that are too small.  ``'negative'`` rejects
        candidate means that are too large.
        """

        if direction not in {"positive", "negative"}:
            raise ValueError("betting direction must be 'positive' or 'negative'")
        candidate = float(candidate_mean)
        if not math.isfinite(candidate):
            raise ValueError("candidate mean must be finite")
        feasible_lower, feasible_upper = self._feasible_mean_interval()
        if (
            candidate < feasible_lower - _FEASIBILITY_TOLERANCE
            or candidate > feasible_upper + _FEASIBILITY_TOLERANCE
        ):
            return math.inf
        if not self._values:
            return 0.0
        values = np.asarray(self._values, dtype=np.float64)
        prefix_before = np.concatenate(
            (np.zeros(1, dtype=np.float64), np.cumsum(values[:-1], dtype=np.float64))
        )
        remaining_before = self.population_size - np.arange(
            values.size, dtype=np.float64
        )
        conditional_mean = (
            self.population_size * candidate - prefix_before
        ) / remaining_before
        deviations = values - conditional_mean
        bets = self._positive_bets if direction == "positive" else -self._positive_bets
        factors = 1.0 + bets[:, None] * deviations[None, :]
        if np.any(factors <= 0.0):
            # A feasible bounded null with bets strictly below 1/range cannot
            # produce this.  Treat numerical or design corruption as refusal,
            # never as finite supporting evidence.
            return math.inf
        component_log_wealth = np.log1p(bets[:, None] * deviations[None, :]).sum(axis=1)
        return float(logsumexp(component_log_wealth) - math.log(bets.size))

    def _invert_one_side(
        self,
        *,
        direction: str,
        feasible_lower: float,
        feasible_upper: float,
        log_threshold: float,
    ) -> float:
        lower_log_e = self.log_e_value(feasible_lower, direction=direction)
        upper_log_e = self.log_e_value(feasible_upper, direction=direction)
        if direction == "positive":
            # Positive-bet wealth decreases monotonically with candidate mean.
            if lower_log_e < log_threshold:
                return feasible_lower
            if upper_log_e >= log_threshold:
                return math.nextafter(feasible_upper, math.inf)
            left, right = feasible_lower, feasible_upper
            for _ in range(_ROOT_ITERATIONS):
                midpoint = 0.5 * (left + right)
                if self.log_e_value(midpoint, direction=direction) >= log_threshold:
                    left = midpoint
                else:
                    right = midpoint
            return right
        # Negative-bet wealth increases monotonically with candidate mean.
        if upper_log_e < log_threshold:
            return feasible_upper
        if lower_log_e >= log_threshold:
            return math.nextafter(feasible_lower, -math.inf)
        left, right = feasible_lower, feasible_upper
        for _ in range(_ROOT_ITERATIONS):
            midpoint = 0.5 * (left + right)
            if self.log_e_value(midpoint, direction=direction) >= log_threshold:
                right = midpoint
            else:
                left = midpoint
        return left

    def interval(self) -> FinitePopulationCSSnapshot:
        """Retained float64 inversion, without an outward rounding guarantee."""
        return self._interval(certified=False)

    def certified_interval(self) -> FinitePopulationCSSnapshot:
        """Outward inversion of the realised law; uncertain endpoints widen.

        This establishes arithmetic enclosure, not the source authenticity,
        boundedness or conditional sampling assumptions supplied by the caller.
        """
        return self._interval(certified=True)

    def _interval(self, *, certified: bool) -> FinitePopulationCSSnapshot:
        count = self.sample_size
        if count == 0:
            return FinitePopulationCSSnapshot(
                stratum_id=self.stratum_id,
                population_size=self.population_size,
                sample_size=0,
                sample_mean=None,
                lower=self.lower_bound,
                upper=self.upper_bound,
                alpha=self.alpha,
                value_lower_bound=self.lower_bound,
                value_upper_bound=self.upper_bound,
                full_census=False,
                empty=False,
                sealed_order_hash=self.sealed_order_hash,
                observed_context_ids=self.context_ids,
                claim_bearing=self.claim_bearing,
            )
        exact_mean = (sum((Fraction.from_float(v) for v in self._values), Fraction(0)) / count
                      if certified else None)
        mean = float(exact_mean) if certified else math.fsum(self._values) / count
        if count == self.population_size:
            return FinitePopulationCSSnapshot(
                stratum_id=self.stratum_id,
                population_size=self.population_size,
                sample_size=count,
                sample_mean=mean,
                lower=(_fraction_to_float_outward(exact_mean, direction="lower") if certified else mean),
                upper=(_fraction_to_float_outward(exact_mean, direction="upper") if certified else mean),
                alpha=self.alpha,
                value_lower_bound=self.lower_bound,
                value_upper_bound=self.upper_bound,
                full_census=True,
                empty=False,
                sealed_order_hash=self.sealed_order_hash,
                observed_context_ids=self.context_ids,
                claim_bearing=self.claim_bearing,
            )
        if certified:
            feasible_lower_fraction, feasible_upper_fraction = (
                self._feasible_mean_interval_fraction()
            )
            feasible_lower = _fraction_to_float_outward(
                feasible_lower_fraction, direction="lower"
            )
            feasible_upper = _fraction_to_float_outward(
                feasible_upper_fraction, direction="upper"
            )
            proof_lower = _fraction_to_float_outward(
                feasible_lower_fraction, direction="upper"
            )
            proof_upper = _fraction_to_float_outward(
                feasible_upper_fraction, direction="lower"
            )
        else:
            feasible_lower, feasible_upper = self._feasible_mean_interval()
            proof_lower, proof_upper = feasible_lower, feasible_upper
        # The two tails receive alpha/2 each.
        if certified:
            if proof_lower > proof_upper:
                lower, upper = feasible_lower, feasible_upper
            else:
                lower = self._certified_one_sided_endpoint(
                    direction="positive",
                    feasible_lower=proof_lower,
                    feasible_upper=proof_upper,
                    conservative_lower=feasible_lower,
                    conservative_upper=feasible_upper,
                )
                upper = self._certified_one_sided_endpoint(
                    direction="negative",
                    feasible_lower=proof_lower,
                    feasible_upper=proof_upper,
                    conservative_lower=feasible_lower,
                    conservative_upper=feasible_upper,
                )
        else:
            log_threshold = math.log(2.0 / self.alpha)
            lower = self._invert_one_side(
                direction="positive",
                feasible_lower=feasible_lower,
                feasible_upper=feasible_upper,
                log_threshold=log_threshold,
            )
            upper = self._invert_one_side(
                direction="negative",
                feasible_lower=feasible_lower,
                feasible_upper=feasible_upper,
                log_threshold=log_threshold,
            )
        return FinitePopulationCSSnapshot(
            stratum_id=self.stratum_id,
            population_size=self.population_size,
            sample_size=count,
            sample_mean=mean,
            lower=lower,
            upper=upper,
            alpha=self.alpha,
            value_lower_bound=self.lower_bound,
            value_upper_bound=self.upper_bound,
            full_census=False,
            empty=bool(lower > upper),
            sealed_order_hash=self.sealed_order_hash,
            observed_context_ids=self.context_ids,
            claim_bearing=self.claim_bearing,
        )


def bounded_wor_betting_confidence_sequence(
    values: Sequence[float],
    *,
    population_size: int,
    alpha: float,
    lower_bound: float = -1.0,
    upper_bound: float = 1.0,
    stratum_id: str = "__single_stratum__",
    context_ids_in_sealed_order: Sequence[str] = (),
    claim_bearing: bool = True,
) -> FinitePopulationCSSnapshot:
    """Functional wrapper returning the CS after the supplied sealed prefix."""

    if len(values) > int(population_size):
        raise ValueError(
            "finite-population observation prefix exceeds the population size"
        )
    if context_ids_in_sealed_order and len(values) > len(context_ids_in_sealed_order):
        raise ValueError(
            "finite-population observation prefix exceeds its sealed context axis"
        )
    sequence = BoundedWoRBettingCS(
        stratum_id=stratum_id,
        population_size=population_size,
        alpha=alpha,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        sealed_context_order=context_ids_in_sealed_order,
        claim_bearing=claim_bearing,
    )
    contexts = tuple(str(value) for value in context_ids_in_sealed_order)
    for index, value in enumerate(values):
        sequence.update(
            float(value),
            context_id=contexts[index] if contexts else None,
        )
    return sequence.interval()


@dataclass(frozen=True, slots=True)
class StratifiedCSSnapshot:
    sampling_design_hash: str
    stratum_alpha: tuple[tuple[str, float], ...]
    observed_context_ids_by_stratum: tuple[tuple[str, tuple[str, ...]], ...]
    observed_values_by_stratum: tuple[tuple[str, tuple[float, ...]], ...]
    draw_history: tuple[tuple[str, str], ...]
    alpha: float
    lower: float
    upper: float
    sample_size: int
    population_size: int
    strata: tuple[FinitePopulationCSSnapshot, ...]
    empty: bool
    full_census: bool
    claim_bearing: bool = True
    method: str = STRATIFIED_WOR_CS_METHOD

    def __post_init__(self) -> None:
        if not _is_sha256(self.sampling_design_hash):
            raise ValueError("stratified CS requires a sampling-design SHA-256")
        if self.method != STRATIFIED_WOR_CS_METHOD:
            raise ValueError("unsupported stratified confidence-sequence method")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("stratified CS alpha must lie in (0,1)")
        if not self.strata:
            raise ValueError("stratified CS requires stratum snapshots")
        stratum_ids = tuple(item.stratum_id for item in self.strata)
        alpha_axis = tuple(
            (str(stratum_id), float(value)) for stratum_id, value in self.stratum_alpha
        )
        observed_axis = tuple(
            (str(stratum_id), tuple(str(value) for value in context_ids))
            for stratum_id, context_ids in self.observed_context_ids_by_stratum
        )
        value_axis = tuple(
            (
                str(stratum_id),
                tuple(float(value) for value in observed_values),
            )
            for stratum_id, observed_values in self.observed_values_by_stratum
        )
        if tuple(stratum_id for stratum_id, _ in alpha_axis) != stratum_ids:
            raise ValueError("stratified CS alpha axis differs from its strata")
        if tuple(stratum_id for stratum_id, _ in observed_axis) != stratum_ids:
            raise ValueError(
                "stratified CS context-prefix axis differs from its strata"
            )
        if tuple(stratum_id for stratum_id, _ in value_axis) != stratum_ids:
            raise ValueError(
                "stratified CS observed-value axis differs from its strata"
            )
        if any(not 0.0 < value < 1.0 for _, value in alpha_axis):
            raise ValueError("stratified CS stratum alpha must lie in (0,1)")
        if sum(value for _, value in alpha_axis) > self.alpha + 1e-15:
            raise ValueError("stratified CS stratum alpha exceeds total alpha")
        for snapshot, (_, alpha_value), (_, context_ids), (_, observed_values) in zip(
            self.strata,
            alpha_axis,
            observed_axis,
            value_axis,
            strict=True,
        ):
            if not math.isclose(snapshot.alpha, alpha_value, abs_tol=1e-15):
                raise ValueError(
                    "stratified CS stratum alpha does not bind its snapshot"
                )
            if snapshot.observed_context_ids != context_ids:
                raise ValueError(
                    "stratified CS context prefix does not bind its snapshot"
                )
            if (
                len(observed_values) != snapshot.sample_size
                or any(
                    not math.isfinite(value)
                    or value < snapshot.value_lower_bound - 1e-12
                    or value > snapshot.value_upper_bound + 1e-12
                    for value in observed_values
                )
                or (
                    snapshot.sample_size > 0
                    and (
                        snapshot.sample_mean is None
                        or not math.isclose(
                            math.fsum(observed_values) / snapshot.sample_size,
                            snapshot.sample_mean,
                            abs_tol=1e-12,
                        )
                    )
                )
            ):
                raise ValueError(
                    "stratified CS raw observations do not bind their snapshot"
                )
            if self.claim_bearing and not snapshot.claim_bearing:
                raise ValueError(
                    "claim-bearing stratified CS contains nonclaim evidence"
                )
        history = tuple(
            (str(stratum_id), str(context_id))
            for stratum_id, context_id in self.draw_history
        )
        if len(history) != self.sample_size:
            raise ValueError("stratified CS draw history differs from its sample count")
        history_by_stratum = {
            stratum_id: tuple(
                context_id
                for observed_stratum, context_id in history
                if observed_stratum == stratum_id
            )
            for stratum_id in stratum_ids
        }
        if self.claim_bearing and any(not context_id for _, context_id in history):
            raise ValueError("claim-bearing draw history contains an opaque context")
        if self.claim_bearing and any(
            history_by_stratum[stratum_id] != context_ids
            for stratum_id, context_ids in observed_axis
        ):
            raise ValueError("stratified CS draw history differs from sealed prefixes")
        if self.sample_size != sum(item.sample_size for item in self.strata):
            raise ValueError("stratified CS sample count differs from its strata")
        if self.population_size != sum(item.population_size for item in self.strata):
            raise ValueError("stratified CS population count differs from its strata")
        if self.empty != (
            self.lower > self.upper or any(item.empty for item in self.strata)
        ):
            raise ValueError("stratified CS empty flag disagrees with projected bounds")
        if self.full_census != all(item.full_census for item in self.strata):
            raise ValueError("stratified CS census flag disagrees with its strata")
        object.__setattr__(self, "stratum_alpha", alpha_axis)
        object.__setattr__(self, "observed_context_ids_by_stratum", observed_axis)
        object.__setattr__(self, "observed_values_by_stratum", value_axis)
        object.__setattr__(self, "draw_history", history)

    @property
    def content_hash(self) -> str:
        return content_hash(self)

    @property
    def context_prefix_hash(self) -> str:
        return content_hash(
            {
                "sampling_design_hash": self.sampling_design_hash,
                "observed_context_ids_by_stratum": self.observed_context_ids_by_stratum,
                "draw_history": self.draw_history,
            }
        )


class StratifiedPairedEProcess:
    """Stratified paired increment CS under predictable adaptive allocation."""

    def __init__(
        self,
        *,
        sampling_design: SamplingDesignSpec,
        alpha: float,
        stratum_alpha: Mapping[str, float] | None = None,
        lower_bound: float = -1.0,
        upper_bound: float = 1.0,
        claim_bearing: bool = True,
    ) -> None:
        total_alpha = float(alpha)
        if not 0.0 < total_alpha < 1.0:
            raise ValueError("stratified e-process alpha must lie in (0,1)")
        if stratum_alpha is None:
            allocation = {
                item.stratum_id: total_alpha * item.risk_weight
                for item in sampling_design.strata
            }
        else:
            allocation = {
                str(key): float(value) for key, value in stratum_alpha.items()
            }
        expected = {item.stratum_id for item in sampling_design.strata}
        if set(allocation) != expected:
            raise ValueError("stratum alpha axis differs from the sampling design")
        if any(not 0.0 < value < 1.0 for value in allocation.values()):
            raise ValueError("stratum alpha allocations must lie in (0,1)")
        if sum(allocation.values()) > total_alpha + 1e-15:
            raise ValueError("stratum alpha allocations exceed total proposal alpha")
        if claim_bearing and not sampling_design.claim_bearing:
            raise ValueError(
                "claim-bearing e-process requires a claim-bearing sampling design"
            )
        if (
            sampling_design.design_kind
            != "sealed_uniform_without_replacement_stratified"
        ):
            raise ValueError(
                "sampling-without-replacement e-process refuses fixed-n IID " "designs"
            )
        self.sampling_design = sampling_design
        self.alpha = total_alpha
        self.claim_bearing = bool(claim_bearing)
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self._weights = {
            item.stratum_id: item.risk_weight for item in sampling_design.strata
        }
        self._stratum_alpha = tuple(
            (stratum_id, allocation[stratum_id]) for stratum_id in sorted(allocation)
        )
        self._draw_history: list[tuple[str, str]] = []
        self._sequences = {
            item.stratum_id: BoundedWoRBettingCS(
                stratum_id=item.stratum_id,
                population_size=item.population_size,
                alpha=allocation[item.stratum_id],
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                sealed_context_order=item.context_ids_in_sealed_order,
                sealed_order_hash=item.sealed_order_hash,
                claim_bearing=claim_bearing,
            )
            for item in sampling_design.strata
        }

    @property
    def stratum_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._sequences))

    def record(
        self,
        stratum_id: str,
        value: float,
        *,
        context_id: str | None = None,
    ) -> None:
        identity = str(stratum_id)
        try:
            sequence = self._sequences[identity]
        except KeyError as error:
            raise KeyError(f"unknown sampling stratum {identity!r}") from error
        sequence.record(value, context_id=context_id)
        self._draw_history.append((identity, str(context_id or "")))

    def update(
        self,
        stratum_id: str,
        value: float,
        *,
        context_id: str | None = None,
    ) -> StratifiedCSSnapshot:
        self.record(stratum_id, value, context_id=context_id)
        return self.interval()

    def stratum_snapshot(self, stratum_id: str) -> FinitePopulationCSSnapshot:
        return self._sequences[str(stratum_id)].interval()

    def interval(self) -> StratifiedCSSnapshot:
        snapshots = tuple(
            self._sequences[stratum_id].interval() for stratum_id in self.stratum_ids
        )
        lower = math.fsum(
            self._weights[item.stratum_id] * item.lower for item in snapshots
        )
        upper = math.fsum(
            self._weights[item.stratum_id] * item.upper for item in snapshots
        )
        return StratifiedCSSnapshot(
            sampling_design_hash=self.sampling_design.content_hash,
            stratum_alpha=self._stratum_alpha,
            observed_context_ids_by_stratum=tuple(
                (item.stratum_id, item.observed_context_ids) for item in snapshots
            ),
            observed_values_by_stratum=tuple(
                (
                    stratum_id,
                    self._sequences[stratum_id].values,
                )
                for stratum_id in self.stratum_ids
            ),
            draw_history=tuple(self._draw_history),
            alpha=self.alpha,
            lower=lower,
            upper=upper,
            sample_size=sum(item.sample_size for item in snapshots),
            population_size=sum(item.population_size for item in snapshots),
            strata=snapshots,
            empty=bool(lower > upper or any(item.empty for item in snapshots)),
            full_census=all(item.full_census for item in snapshots),
            claim_bearing=self.claim_bearing,
        )


__all__ = [
    "STRATIFIED_WOR_CS_METHOD",
    "WOR_BETTING_CS_METHOD",
    "BoundedWoRBettingCS",
    "FinitePopulationCSSnapshot",
    "StratifiedCSSnapshot",
    "StratifiedPairedEProcess",
    "bounded_wor_betting_confidence_sequence",
]
