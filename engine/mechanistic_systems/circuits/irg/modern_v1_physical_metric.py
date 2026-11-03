"""Physical observation laws and outward moment-to-length conversion.

These formulas carry no sampling or decision authority. The legacy C law uses
population-size weights and absolute R/D observations. The function-preserving
law uses equal target/complement mass and squared R/D observations, taking the
root only after population aggregation. F remains mean work; Q remains the
second moment of the signed literal response in both laws.
"""
from dataclasses import fields
from decimal import Decimal, localcontext
from fractions import Fraction
import math

LEGACY_UNIFORM_ABSOLUTE = "uniform_population_absolute_displacement_v1"
BALANCED_RMS = "balanced_target_complement_rms_displacement_v1"
BALANCED_MEASURE = "equal_target_complement_mass_uniform_within_each_stratum"


def validate_physical_measurement_profile(profile: str) -> None:
    if type(profile) is not str or profile not in (LEGACY_UNIFORM_ABSOLUTE, BALANCED_RMS):
        raise ValueError("unsupported physical measurement profile")


def profile_coordinates(profile: str) -> dict[str, object]:
    """Only the new law adds fields; historical receipt payloads remain exact."""
    validate_physical_measurement_profile(profile)
    if profile == LEGACY_UNIFORM_ABSOLUTE:
        return {}
    return dict(physical_measurement_profile=profile,
                population_measure=BALANCED_MEASURE,
                remainder_and_displacement_observation_power=2,
                remainder_and_displacement_root_after_aggregation=True)


def restore_profile_state(owner: object, state: list[object]) -> None:
    """Restore the appended profile slot; old slot order and meaning are exact.

    This only reconstructs serialization state. The receiving source/receipt
    owner retains integrity, live custody and admission checks.
    """
    names = tuple(value.name for value in fields(owner))
    if not names or names[-1] != "physical_measurement_profile":
        raise ValueError("physical profile must be appended after historical slots")
    if len(state) == len(names) - 1:
        values = tuple(state) + (LEGACY_UNIFORM_ABSOLUTE,)
    elif len(state) == len(names):
        values = tuple(state)
    else:
        raise ValueError("unsupported physical receipt serialization state")
    validate_physical_measurement_profile(values[-1])
    for name, value in zip(names, values, strict=True):
        object.__setattr__(owner, name, value)


def _outward_fraction(value: Fraction, *, direction: str) -> float:
    if direction not in ("lower", "upper") or value < 0:
        raise ValueError("nonnegative value and lower/upper direction required")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError("physical moment exceeds finite binary64") from exc
    if not math.isfinite(result):
        raise ValueError("physical moment exceeds finite binary64")
    observed = Fraction.from_float(result)
    if direction == "upper" and observed < value:
        result = math.nextafter(result, math.inf)
    elif direction == "lower" and observed > value:
        result = math.nextafter(result, -math.inf)
    if not math.isfinite(result):
        raise ValueError("outward physical moment exceeds finite binary64")
    return result


def squared_support_upper(value: float) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError("finite nonnegative physical support required")
    return _outward_fraction(Fraction.from_float(float(value)) ** 2, direction="upper")


def squared_displacement_interval(*, parent_lower: float, parent_upper: float,
                                  child_lower: float, child_upper: float) -> tuple[float, float]:
    """Enclose squared displacement from supplied finite logit intervals.

    This is exact endpoint arithmetic plus outward binary64 conversion. The
    caller remains responsible for the validity and source of the input bounds.
    """
    values = (parent_lower, parent_upper, child_lower, child_upper)
    if (any(not math.isfinite(value) for value in values)
            or parent_lower > parent_upper or child_lower > child_upper):
        raise ValueError("finite ordered parent/child intervals required")
    pl, pu, cl, cu = (Fraction.from_float(float(value)) for value in values)
    lower, upper = cl - pu, cu - pl
    absolute_lower = Fraction() if lower <= 0 <= upper else min(abs(lower), abs(upper))
    absolute_upper = max(abs(lower), abs(upper))
    return (_outward_fraction(absolute_lower ** 2, direction="lower"),
            _outward_fraction(absolute_upper ** 2, direction="upper"))


def _sqrt_fraction_bound(value: Fraction, *, direction: str) -> float:
    if direction not in ("lower", "upper") or value < 0:
        raise ValueError("nonnegative moment and lower/upper direction required")
    # Decimal supplies a candidate even when float(value) would underflow.
    # The exact rational square comparison, not Decimal rounding, authorizes it.
    with localcontext() as ctx:
        ctx.prec = 100
        result = float((Decimal(value.numerator) / Decimal(value.denominator)).sqrt())
    if not math.isfinite(result):
        raise ValueError("physical response length exceeds finite binary64")
    for _ in range(4):
        square = Fraction.from_float(result) ** 2
        if (direction == "lower" and square <= value) or (direction == "upper" and square >= value):
            return result
        result = math.nextafter(result, -math.inf if direction == "lower" else math.inf)
    raise ValueError("physical response length could not be enclosed")


def balanced_role_bound(*, stratum_ids: tuple[str, ...], values: tuple[float, ...],
                        take_root: bool, direction: str) -> float:
    if (len(stratum_ids) != 2 or set(stratum_ids) != {"target", "complement"}
            or len(values) != 2 or type(take_root) is not bool
            or any(not math.isfinite(v) or v < 0 for v in values)):
        raise ValueError("balanced physical bounds require exact target/complement strata and finite values")
    mean = sum((Fraction.from_float(float(v)) / 2 for v in values), Fraction())
    return (_sqrt_fraction_bound(mean, direction=direction) if take_root
            else _outward_fraction(mean, direction=direction))


def six_role_observation_bounds(*, f_lower, f_upper, j_center, j_radius,
                                child_center, child_radius, parent_center,
                                parent_radius, doses=(0.5, 1.0),
                                physical_measurement_profile=LEGACY_UNIFORM_ABSOLUTE):
    """Outward arithmetic for supplied per-action F/J/finite-logit intervals.

    No sampling, input-rounding, model or population authority is supplied by
    this function. Zero input radii enclose the recorded floats only. Shapes
    are F/J [actions], child [actions, two doses], and scalar parent.
    """
    import numpy as np
    from .modern_v1_fibered_directional_record import (
        _absolute_interval, _decimal, _outward_float,
    )
    validate_physical_measurement_profile(physical_measurement_profile)
    f_lower, f_upper, j_center, j_radius, child_center, child_radius = (
        np.asarray(value, dtype=np.float64) for value in
        (f_lower, f_upper, j_center, j_radius, child_center, child_radius))
    if j_center.ndim != 1 or not j_center.size or tuple(doses) != (0.5, 1.0):
        raise ValueError("nonempty action vector and registered half/full doses required")
    action_count = j_center.size
    if (any(value.shape != (action_count,) for value in (f_lower, f_upper, j_radius))
            or any(value.shape != (action_count, 2) for value in (child_center, child_radius))):
        raise ValueError("physical observation interval shapes differ")
    if (any(not np.isfinite(value).all() for value in
            (f_lower, f_upper, j_center, j_radius, child_center, child_radius))
            or not math.isfinite(parent_center) or not math.isfinite(parent_radius)
            or parent_radius < 0 or (f_lower < 0).any() or (f_upper < f_lower).any()
            or (j_radius < 0).any() or (child_radius < 0).any()):
        raise ValueError("finite ordered physical input intervals required")
    observations_lower = np.empty(
        (action_count, 6),
        dtype=np.float64,
    )
    observations_upper = np.empty_like(observations_lower)
    observations_lower[:, 0] = f_lower
    observations_upper[:, 0] = f_upper
    with localcontext() as decimal_context:
        decimal_context.prec = 2500
        p_center = _decimal(parent_center)
        p_radius = _decimal(parent_radius)
        p_lower = p_center - p_radius
        p_upper = p_center + p_radius
        for action in range(action_count):
            jc = _decimal(float(j_center[action]))
            jr = _decimal(float(j_radius[action]))
            jl = jc - jr
            ju = jc + jr
            abs_j_lower, abs_j_upper = _absolute_interval(jl, ju)
            observations_lower[action, 1] = _outward_float(
                abs_j_lower * abs_j_lower,
                upward=False,
                name="Q_J lower bound",
            )
            observations_upper[action, 1] = _outward_float(
                abs_j_upper * abs_j_upper,
                upward=True,
                name="Q_J upper bound",
            )
            for dose_index, dose in enumerate(doses):
                cc = _decimal(float(child_center[action, dose_index]))
                cr = _decimal(float(child_radius[action, dose_index]))
                raw_lower = (cc - cr) - p_upper
                raw_upper = (cc + cr) - p_lower
                eta = _decimal(dose)
                remainder_lower = raw_lower - eta * ju
                remainder_upper = raw_upper - eta * jl
                abs_r_lower, abs_r_upper = _absolute_interval(
                    remainder_lower,
                    remainder_upper,
                )
                abs_d_lower, abs_d_upper = _absolute_interval(
                    raw_lower,
                    raw_upper,
                )
                if physical_measurement_profile == BALANCED_RMS:
                    abs_r_lower, abs_r_upper = abs_r_lower ** 2, abs_r_upper ** 2
                    abs_d_lower, abs_d_upper = abs_d_lower ** 2, abs_d_upper ** 2
                r_role = 2 + dose_index
                d_role = 4 + dose_index
                observations_lower[action, r_role] = _outward_float(
                    abs_r_lower,
                    upward=False,
                    name="absolute nonlinear remainder lower bound",
                )
                observations_upper[action, r_role] = _outward_float(
                    abs_r_upper,
                    upward=True,
                    name="absolute nonlinear remainder upper bound",
                )
                observations_lower[action, d_role] = _outward_float(
                    abs_d_lower,
                    upward=False,
                    name="absolute raw-logit displacement lower bound",
                )
                observations_upper[action, d_role] = _outward_float(
                    abs_d_upper,
                    upward=True,
                    name="absolute raw-logit displacement upper bound",
                )
    return observations_lower, observations_upper
