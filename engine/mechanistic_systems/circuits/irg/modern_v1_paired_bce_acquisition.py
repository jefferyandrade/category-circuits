"""Compact finite-population acquisition for the modern-v1 paired-loss oracle.

This module owns a deliberately narrow statistical seam.  A candidate mask,
dense reference, candidate-specific calibration, loss ceiling, finite
population, within-stratum randomizations, pointwise loss bounds, and one
global candidate-attempt alpha coordinate are frozen before any oracle
paired-loss row is requested.  In the production causal design, calibration
uses an adaptively stopped prefix ``K`` and then opens a fresh generation-local
without-replacement order on the exact residual population ``O = U \\ K``.
Physical chunks advance the two sealed residual prefixes (``target`` and
``complement``) while durable state retains only counts, exact binary64 sums,
and compact receipt-chain coordinates.  The exact paired-loss contribution of
the frozen calibration prefix is retained in the dynamic oracle receipt, so
the estimand remains the complete finite training population ``U`` rather
than only its out-of-fit residual.
The receipts themselves remain in the external append-only store; checkpoint
state keeps only their count, last hash, and hash-chain head.

The canonical partial decision rehydrates the repository's bounded
sampling-without-replacement betting confidence sequence from externally
stored, hash-bound chunk payloads.  Raw prefixes and full sealed axes are
transient replay inputs, never checkpoint state.  The resulting weighted
interval is anytime-valid under the declared sampling law.  A preregistered
group-sequential Hoeffding--Serfling/Bonferroni construction remains available
only behind an explicit lower-authority fallback flag.  If either stratum
lacks a finite, nonvacuous, pre-outcome bound, partial looks have no
accept/reject authority and acquisition proceeds to census.

At a partial look, each residual confidence interval is lifted affinely by its
exact ``K`` mass and its ``|O| / |U|`` weight before the two strata are
combined.  At the joint residual census no concentration theorem or alpha is
needed.  Every binary64 paired-loss row from ``K`` and ``O`` is accumulated as
an integer multiple of ``2**-1074``; the equally weighted complete-population
mean is compared to the binary64 ceiling as an exact rational.  Consequently
``mean <= ceiling`` (including exact equality) accepts and ``mean > ceiling``
rejects.  The result remains authority-inert: it is evidence for a controller
and never commits a mask by itself.  The older static disjoint-fold receipt is
retained only as an explicit compatibility surface for nonproduction fixtures.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from decimal import Decimal, ROUND_CEILING, localcontext
from fractions import Fraction
import hashlib
import math
import struct
import sys
from types import MappingProxyType
import warnings
from typing import Any, Final, Literal

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.evidence.finite_population import (
    STRATIFIED_WOR_CS_METHOD,
    BoundedWoRBettingCS,
)
from mechanistic_systems.evidence.glocal_contracts import (
    content_hash as sampling_content_hash,
)


TARGET_STRATUM: Final = "target"
COMPLEMENT_STRATUM: Final = "complement"
PAIRED_BCE_STRATA: Final = (TARGET_STRATUM, COMPLEMENT_STRATUM)
PAIRED_BCE_STRATUM_WEIGHTS: Final = (0.5, 0.5)

CLIPPED_LOGIT_PAIRED_BCE_KIND: Final = (
    "paired_clipped_logit_binary_cross_entropy_difference_v1"
)
BOUNDED_BRIER_PAIRED_LOSS_KIND: Final = "paired_bounded_brier_difference_v1"
ANALYTIC_FIXED_AFFINE_PAIRED_BCE_KIND: Final = (
    "paired_analytic_alexnet_fixed_affine_binary_cross_entropy_difference_v1"
)
CENSUS_ONLY_PAIRED_BCE_KIND: Final = "unbounded_paired_bce_census_only_v1"
PARTIAL_BOUND_CAPABLE_LOSS_KINDS: Final = (
    CLIPPED_LOGIT_PAIRED_BCE_KIND,
    BOUNDED_BRIER_PAIRED_LOSS_KIND,
    ANALYTIC_FIXED_AFFINE_PAIRED_BCE_KIND,
)

SEALED_HMAC_WOR_SAMPLING_LAW: Final = (
    "closed_hmac_sha256_fisher_yates_independent_stratum_permutations_v1"
)
PREDICTABLE_STRATUM_POLICY: Final = (
    "predictable_max_projected_serfling_width_reduction_v1"
)
GROUP_SEQUENTIAL_METHOD: Final = (
    "registered_group_sequential_hoeffding_serfling_bonferroni_v1"
)
GROUP_SEQUENTIAL_EVIDENCE_CEILING: Final = (
    "lower_authority_computational_design_based_fwer_at_registered_joint_looks_"
    "conditional_on_"
    "independent_hmac_sha256_pseudorandom_uniform_stratum_permutations_v1"
)
CANONICAL_ANYTIME_BETTING_METHOD: Final = STRATIFIED_WOR_CS_METHOD
CANONICAL_ANYTIME_BETTING_EVIDENCE_CEILING: Final = (
    "anytime_valid_stratified_bounded_wor_betting_cs_conditional_on_frozen_"
    "finite_bounds_and_independent_hmac_sha256_pseudorandom_uniform_stratum_"
    "permutations_v1"
)
ALPHA_NONRECYCLING_POLICY: Final = (
    "global_candidate_attempt_alpha_axis_fixed_preoutcome_nontransferable_"
    "nonrecycling_v1"
)
PARTIAL_LOOK_ALPHA_POLICY: Final = (
    "uniform_bonferroni_over_registered_noncensus_joint_looks_strata_and_tails_v1"
)
EXACT_CENSUS_METHOD: Final = (
    "exact_binary64_superaccumulator_equal_stratum_finite_set_mean_v1"
)
AUTHORITY_SCOPE: Final = "statistical_candidate_attempt_evidence_only_no_mask_commit_v1"

STRATUM_REGISTRATION_SCHEMA: Final = "modern_v1_paired_bce_stratum_registration_v1"
ORACLE_ROLE_SEPARATION_SCHEMA: Final = (
    "modern_v1_paired_bce_oracle_role_separation_receipt_v1"
)
ACQUISITION_REGISTRATION_SCHEMA: Final = (
    "modern_v1_paired_bce_acquisition_registration_v2"
)
DYNAMIC_ORACLE_RECEIPT_SCHEMA: Final = (
    "modern_v1_paired_bce_dynamic_full_train_oracle_receipt_v1"
)
CHUNK_REQUEST_SCHEMA: Final = "modern_v1_paired_bce_pending_chunk_request_v2"
CHUNK_RECEIPT_SCHEMA: Final = "modern_v1_paired_bce_chunk_receipt_v2"
ACQUISITION_STATE_SCHEMA: Final = "modern_v1_paired_bce_acquisition_state_v2"
DECISION_SCHEMA: Final = "modern_v1_paired_bce_decision_v2"
CALIBRATION_HORIZON_REFUSAL_SCHEMA: Final = (
    "partition_free_v1_alexnet_causal_calibration_horizon_refusal_v1"
)
CALIBRATION_HORIZON_REFUSAL_STATUS: Final = (
    "failed_closed_calibration_convergence_by_n_minus_1"
)
CALIBRATION_HORIZON_SKIP_DECISION_SCHEMA: Final = (
    "partition_free_v1_paired_bce_calibration_horizon_skip_decision_v1"
)
CALIBRATION_HORIZON_SKIP_STATUS: Final = "skip_candidate_calibration_refusal"
CALIBRATION_HORIZON_SKIP_REASON: Final = (
    "causal_candidate_calibration_failed_by_registered_n_minus_1_horizon"
)
CALIBRATION_HORIZON_REFUSAL_PREFIX: Final = (
    "partition-free-v1-alexnet-causal-calibration-horizon-refusal-"
)
CALIBRATION_FAILURE_PREFIX: Final = (
    "partition-free-v1-alexnet-mask-calibration-failure-"
)

_BINARY64_SCALE_EXPONENT: Final = 1074
_MIN_SAFE_TWO_STRATUM_ALPHA: Final = 4.0 / sys.float_info.max
_EMPTY_RECEIPT_CHAIN_HASH: Final = hashlib.sha256(
    b"modern-v1-paired-bce-empty-receipt-chain-v1"
).hexdigest()

DecisionStatus = Literal["acquire_more", "accept_candidate", "reject_candidate"]


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _strict_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a strict boolean")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be one nonnegative integer")
    if value < 0:
        raise ValueError(f"{name} must be one nonnegative integer")
    return int(value)


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be one positive integer")
    return result


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _unit_alpha(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if not _MIN_SAFE_TWO_STRATUM_ALPHA <= result < 1.0:
        raise ValueError(
            f"{name} must lie in the numerically supported subset of (0,1)"
        )
    return result


def _optional_finite(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    return _finite(value, name=name)


def _pair_of_nonnegative_ints(
    values: object, *, name: str
) -> tuple[int, int]:
    rows = tuple(values) if isinstance(values, (tuple, list)) else ()
    if len(rows) != 2:
        raise ValueError(f"{name} must contain exactly two entries")
    return (
        _nonnegative_int(rows[0], name=f"{name}[0]"),
        _nonnegative_int(rows[1], name=f"{name}[1]"),
    )


def _pair_of_optional_finite(
    values: object, *, name: str
) -> tuple[float | None, float | None]:
    rows = tuple(values) if isinstance(values, (tuple, list)) else ()
    if len(rows) != 2:
        raise ValueError(f"{name} must contain exactly two entries")
    return (
        _optional_finite(rows[0], name=f"{name}[0]"),
        _optional_finite(rows[1], name=f"{name}[1]"),
    )


def _pair_of_identities(values: object, *, name: str) -> tuple[str, str]:
    rows = tuple(values) if isinstance(values, (tuple, list)) else ()
    if len(rows) != 2:
        raise ValueError(f"{name} must contain exactly two entries")
    return (
        _identity(rows[0], name=f"{name}[0]"),
        _identity(rows[1], name=f"{name}[1]"),
    )


def _pair_of_ints(values: object, *, name: str) -> tuple[int, int]:
    rows = tuple(values) if isinstance(values, (tuple, list)) else ()
    if len(rows) != 2 or any(
        isinstance(value, bool) or not isinstance(value, int) for value in rows
    ):
        raise ValueError(f"{name} must contain exactly two integers")
    return int(rows[0]), int(rows[1])


def _constructor_payload(value: object) -> dict[str, Any]:
    return {
        item.name: getattr(value, item.name)
        for item in fields(type(value))
        if item.init
    }


def _freeze_payload(value: Any, *, name: str) -> Any:
    """Recursively freeze one content-addressed receipt payload."""

    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str or not key:
                raise TypeError(f"{name} keys must be nonempty strings")
            frozen[key] = _freeze_payload(item, name=f"{name}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (tuple, list)):
        return tuple(
            _freeze_payload(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, float):
        return _finite(value, name=name)
    if isinstance(value, (str, bytes, int, bool)) or value is None:
        return value
    raise TypeError(f"{name} contains unsupported value {type(value)!r}")


def validate_paired_bce_calibration_horizon_refusal_receipt(
    value: object,
) -> Mapping[str, Any]:
    """Validate and freeze the exact causal ``K=N-1`` refusal receipt.

    This is intentionally stricter than checking the exception type.  Only a
    content-addressed causal refusal that exhausted the registered nonempty-O
    horizon can authorize a candidate skip; every other solver failure remains
    fatal and cannot consume a controller attempt.
    """

    if not isinstance(value, Mapping):
        raise TypeError("causal calibration horizon refusal must be a mapping")
    expected_fields = {
        "schema",
        "status",
        "candidate_mask_hash",
        "candidate_attempt_index",
        "candidate_freeze_hash",
        "calibration_stream_hash",
        "full_population_stratum_counts",
        "calibration_sampling_capacity_stratum_counts",
        "registered_look_cumulative_stratum_counts",
        "last_attempt_cumulative_stratum_counts",
        "last_fit_failure_receipt",
        "last_fit_failure_receipt_hash",
        "oracle_stream_opened",
        "oracle_population_reserved_nonempty",
        "terminal_outcomes_opened",
        "refusal_receipt_hash",
    }
    if set(value) != expected_fields:
        raise ValueError("causal calibration horizon refusal fields are not exact")
    payload = dict(value)
    if (
        payload["schema"] != CALIBRATION_HORIZON_REFUSAL_SCHEMA
        or payload["status"] != CALIBRATION_HORIZON_REFUSAL_STATUS
    ):
        raise ValueError("causal calibration horizon refusal schema/status is invalid")
    for name in (
        "candidate_mask_hash",
        "candidate_freeze_hash",
        "calibration_stream_hash",
        "last_fit_failure_receipt_hash",
        "refusal_receipt_hash",
    ):
        payload[name] = _identity(payload[name], name=name)
    payload["candidate_attempt_index"] = _nonnegative_int(
        payload["candidate_attempt_index"], name="candidate attempt index"
    )
    full = _pair_of_nonnegative_ints(
        payload["full_population_stratum_counts"],
        name="full population stratum counts",
    )
    capacity = _pair_of_nonnegative_ints(
        payload["calibration_sampling_capacity_stratum_counts"],
        name="calibration sampling capacity stratum counts",
    )
    if any(population <= 1 for population in full) or capacity != (
        full[0] - 1,
        full[1] - 1,
    ):
        raise ValueError("causal calibration refusal did not reserve nonempty O")
    raw_schedule = payload["registered_look_cumulative_stratum_counts"]
    if isinstance(raw_schedule, (str, bytes)) or not isinstance(
        raw_schedule, (tuple, list)
    ):
        raise TypeError("causal calibration refusal schedule must be a sequence")
    schedule = tuple(
        _pair_of_nonnegative_ints(row, name=f"registered K look {index}")
        for index, row in enumerate(raw_schedule)
    )
    if not schedule:
        raise ValueError("causal calibration refusal lacks a registered K schedule")
    previous = (0, 0)
    for row in schedule:
        if (
            any(row[index] < previous[index] for index in (0, 1))
            or not any(row[index] > previous[index] for index in (0, 1))
            or any(row[index] <= 0 for index in (0, 1))
            or any(row[index] > capacity[index] for index in (0, 1))
        ):
            raise ValueError("causal calibration refusal schedule is not append-only")
        previous = row
    terminal = _pair_of_nonnegative_ints(
        payload["last_attempt_cumulative_stratum_counts"],
        name="last K attempt counts",
    )
    if terminal != schedule[-1] or terminal != capacity:
        raise ValueError("causal calibration refusal did not exhaust the N-1 horizon")
    failure = payload["last_fit_failure_receipt"]
    if not isinstance(failure, Mapping):
        raise TypeError("causal calibration refusal lacks its fit-failure receipt")
    failure_payload = dict(failure)
    failure_hash = failure_payload.pop("failure_receipt_hash", None)
    if (
        failure_hash != payload["last_fit_failure_receipt_hash"]
        or failure_hash
        != content_hash(failure_payload, prefix=CALIBRATION_FAILURE_PREFIX)
    ):
        raise ValueError("causal calibration failure receipt hash does not replay")
    for name, expected in (
        ("oracle_stream_opened", False),
        ("oracle_population_reserved_nonempty", True),
        ("terminal_outcomes_opened", False),
    ):
        if type(payload[name]) is not bool or payload[name] is not expected:
            raise ValueError(f"causal calibration refusal requires {name}={expected}")
    declared_hash = payload.pop("refusal_receipt_hash")
    if declared_hash != content_hash(
        payload,
        prefix=CALIBRATION_HORIZON_REFUSAL_PREFIX,
    ):
        raise ValueError("causal calibration horizon refusal hash does not replay")
    payload["refusal_receipt_hash"] = declared_hash
    payload["full_population_stratum_counts"] = full
    payload["calibration_sampling_capacity_stratum_counts"] = capacity
    payload["registered_look_cumulative_stratum_counts"] = schedule
    payload["last_attempt_cumulative_stratum_counts"] = terminal
    payload["last_fit_failure_receipt"] = dict(failure)
    return _freeze_payload(payload, name="causal calibration horizon refusal")


def _rebuild_hashed(
    value_type: type[Any], payload: dict[str, Any], expected_hash: str
) -> Any:
    result = value_type(**payload)
    result.assert_integrity(expected_hash)
    return result


def _assert_hash_integrity(
    owner: object,
    *,
    payload: object,
    cache_name: str,
    label: str,
    expected_hash: str | None = None,
) -> str:
    observed = content_hash(payload)
    cached = getattr(owner, cache_name)
    if cached is not None and observed != cached:
        raise ValueError(f"{label} was tampered after hashing")
    if expected_hash is not None and observed != expected_hash:
        raise ValueError(f"{label} hash does not match")
    object.__setattr__(owner, cache_name, observed)
    return observed


def _binary64_scaled_integer(value: float) -> int:
    """Return the exact integer ``value * 2**1074`` for finite binary64."""

    observation = _finite(value, name="paired-loss row")
    numerator, denominator = observation.as_integer_ratio()
    denominator_exponent = denominator.bit_length() - 1
    if denominator != 1 << denominator_exponent:  # pragma: no cover - float law.
        raise RuntimeError("binary64 denominator was not a power of two")
    return numerator << (_BINARY64_SCALE_EXPONENT - denominator_exponent)


def _scaled_mean(scaled_sum: int, count: int) -> Fraction:
    if count <= 0:
        raise ValueError("an exact mean requires at least one observation")
    return Fraction(int(scaled_sum), int(count) << _BINARY64_SCALE_EXPONENT)


def _weighted_exact_mean(
    scaled_sums: tuple[int, int], counts: tuple[int, int]
) -> Fraction:
    return (_scaled_mean(scaled_sums[0], counts[0]) + _scaled_mean(
        scaled_sums[1], counts[1]
    )) / 2


def _stable_equal_weight_mean(left: float, right: float) -> float:
    """Average two finite binary64 values without overflowing their sum."""

    return math.fsum((0.5 * left, 0.5 * right))


def _fraction_to_float_outward(
    value: Fraction, *, direction: Literal["lower", "upper"]
) -> float:
    rounded = float(value)
    if not math.isfinite(rounded):
        raise ValueError("confidence endpoint overflowed binary64")
    rounded_exact = Fraction.from_float(rounded)
    if direction == "lower" and rounded_exact > value:
        return math.nextafter(rounded, -math.inf)
    if direction == "upper" and rounded_exact < value:
        return math.nextafter(rounded, math.inf)
    return rounded


def _weighted_confidence_endpoint(
    left: float,
    right: float,
    *,
    direction: Literal["lower", "upper"],
) -> float:
    exact = (Fraction.from_float(left) + Fraction.from_float(right)) / 2
    return _fraction_to_float_outward(exact, direction=direction)


def _serfling_radius_upper(
    *,
    lower_bound: float,
    upper_bound: float,
    count: int,
    population: int,
    look_alpha: float,
) -> float:
    """Compute a high-precision, binary64-upward Serfling radius.

    Every finite bound and alpha enters as its exact binary64 value.  Decimal
    work is performed at far more precision than binary64, the final Decimal
    result is advanced once at that precision, conversion to binary64 is
    directed upward, and one additional binary64 ULP is reserved as a guard
    against transcendental rounding.  This prevents the ordinary float
    expression from silently shrinking a claim-bearing interval.
    """

    span_fraction = Fraction.from_float(upper_bound) - Fraction.from_float(
        lower_bound
    )
    correction_fraction = Fraction(population - count + 1, population)
    with localcontext() as context:
        context.prec = 100
        context.rounding = ROUND_CEILING
        span = Decimal(span_fraction.numerator) / Decimal(span_fraction.denominator)
        correction = Decimal(correction_fraction.numerator) / Decimal(
            correction_fraction.denominator
        )
        alpha = Decimal.from_float(look_alpha)
        log_factor = (Decimal(4) / alpha).ln().next_plus()
        radius = (
            span
            * (correction * log_factor / Decimal(2 * count)).sqrt()
        ).next_plus()
    rounded = float(radius)
    if not math.isfinite(rounded):
        return math.inf
    if Decimal.from_float(rounded) < radius:
        rounded = math.nextafter(rounded, math.inf)
    # Decimal's 100-digit ln/sqrt error is vastly below one binary64 ULP, but
    # a claim-bearing fallback should not depend on a nearest-rounded
    # transcendental boundary.  The explicit guard makes the returned radius
    # strictly outward relative to the already upward Decimal construction.
    return math.nextafter(rounded, math.inf)


def _full_population_stratum_point_mean(
    *,
    registration: V1PairedBCEAcquisitionRegistration,
    state: V1PairedBCEAcquisitionState,
    index: int,
) -> Fraction | None:
    residual_count = state.stratum_counts[index]
    if residual_count == 0:
        return None
    dynamic = registration.dynamic_oracle_receipt
    if dynamic is None:
        return _scaled_mean(state.exact_scaled_loss_sums[index], residual_count)
    full_count = dynamic.full_population_stratum_counts[index]
    residual_population = dynamic.oracle_population_stratum_counts[index]
    exact_k_total = Fraction(
        dynamic.exact_calibration_scaled_loss_sums[index],
        1 << _BINARY64_SCALE_EXPONENT,
    )
    residual_sample_mean = _scaled_mean(
        state.exact_scaled_loss_sums[index], residual_count
    )
    return (
        exact_k_total + residual_population * residual_sample_mean
    ) / full_count


def _full_population_exact_mean(
    *,
    registration: V1PairedBCEAcquisitionRegistration,
    state: V1PairedBCEAcquisitionState,
) -> Fraction:
    dynamic = registration.dynamic_oracle_receipt
    if dynamic is None:
        return _weighted_exact_mean(
            state.exact_scaled_loss_sums, state.stratum_counts
        )
    means = tuple(
        Fraction(
            dynamic.exact_calibration_scaled_loss_sums[index]
            + state.exact_scaled_loss_sums[index],
            dynamic.full_population_stratum_counts[index]
            << _BINARY64_SCALE_EXPONENT,
        )
        for index in (0, 1)
    )
    return (means[0] + means[1]) / 2


def _lift_residual_interval_to_full_population(
    *,
    registration: V1PairedBCEAcquisitionRegistration,
    index: int,
    interval: tuple[float, float],
) -> tuple[float, float]:
    dynamic = registration.dynamic_oracle_receipt
    if dynamic is None:
        return interval
    full_count = dynamic.full_population_stratum_counts[index]
    residual_population = dynamic.oracle_population_stratum_counts[index]
    exact_k_total = Fraction(
        dynamic.exact_calibration_scaled_loss_sums[index],
        1 << _BINARY64_SCALE_EXPONENT,
    )
    exact_endpoints = tuple(
        (
            exact_k_total
            + residual_population * Fraction.from_float(float(value))
        )
        / full_count
        for value in interval
    )
    endpoints = (
        _fraction_to_float_outward(exact_endpoints[0], direction="lower"),
        _fraction_to_float_outward(exact_endpoints[1], direction="upper"),
    )
    return endpoints[0], endpoints[1]


def paired_bce_loss_ceiling_identity_hash(
    *,
    dense_reference_hash: str,
    candidate_calibration_freeze_hash: str,
    paired_loss_kind: str,
    loss_ceiling: float,
) -> str:
    """Construct the required identity of the frozen loss boundary."""

    return content_hash(
        {
            "schema": "modern_v1_paired_bce_loss_ceiling_identity_v1",
            "dense_reference_hash": _identity(
                dense_reference_hash, name="dense reference hash"
            ),
            "candidate_calibration_freeze_hash": _identity(
                candidate_calibration_freeze_hash,
                name="candidate calibration freeze hash",
            ),
            "paired_loss_kind": _identity(
                paired_loss_kind, name="paired loss kind"
            ),
            "loss_ceiling": _finite(loss_ceiling, name="loss ceiling"),
        }
    )


def paired_bce_attempt_alpha_coordinate_hash(
    *,
    candidate_attempt_alpha_axis_hash: str,
    candidate_attempt_index: int,
    maximum_candidate_attempts: int,
    attempt_alpha_allocation: float,
) -> str:
    """Bind one attempt to its pre-outcome global alpha-axis coordinate."""

    index = _nonnegative_int(candidate_attempt_index, name="candidate attempt index")
    maximum = _positive_int(
        maximum_candidate_attempts, name="maximum candidate attempts"
    )
    if index >= maximum:
        raise ValueError("candidate attempt index leaves its registered alpha axis")
    return content_hash(
        {
            "schema": "modern_v1_paired_bce_attempt_alpha_coordinate_v1",
            "candidate_attempt_alpha_axis_hash": _identity(
                candidate_attempt_alpha_axis_hash,
                name="candidate attempt alpha axis hash",
            ),
            "candidate_attempt_index": index,
            "maximum_candidate_attempts": maximum,
            "attempt_alpha_allocation": _unit_alpha(
                attempt_alpha_allocation, name="attempt alpha allocation"
            ),
            "alpha_nonrecycling_policy": ALPHA_NONRECYCLING_POLICY,
        }
    )


def paired_bce_sealed_order_hash(
    *, stratum_id: str, context_ids_in_sealed_order: Sequence[str]
) -> str:
    """Return the exact order identity consumed by ``BoundedWoRBettingCS``."""

    identity = _identity(stratum_id, name="stratum id")
    if identity not in PAIRED_BCE_STRATA:
        raise ValueError("paired-BCE sealed order names an unknown stratum")
    contexts = tuple(
        _identity(value, name="sealed-order context id")
        for value in context_ids_in_sealed_order
    )
    if not contexts or len(contexts) != len(set(contexts)):
        raise ValueError("sealed context order must enumerate a nonempty population once")
    return sampling_content_hash(
        {"stratum_id": identity, "context_ids": contexts}
    )


def geometric_paired_bce_look_schedule(
    target_population_size: int,
    complement_population_size: int,
    *,
    initial_count: int = 1,
    growth_factor: int = 2,
) -> tuple[tuple[int, int], ...]:
    """Return nested geometric joint looks ending at the exact two-stratum census."""

    populations = (
        _positive_int(target_population_size, name="target population size"),
        _positive_int(
            complement_population_size, name="complement population size"
        ),
    )
    initial = _positive_int(initial_count, name="initial count")
    growth = _positive_int(growth_factor, name="growth factor")
    if growth <= 1:
        raise ValueError("geometric look growth factor must exceed one")
    current = (min(initial, populations[0]), min(initial, populations[1]))
    looks = [current]
    while current != populations:
        following: list[int] = []
        for count, population in zip(current, populations, strict=True):
            if count == population:
                following.append(population)
            else:
                following.append(min(population, max(count + 1, count * growth)))
        current = (following[0], following[1])
        looks.append(current)
    return tuple(looks)


def uniform_partial_look_alpha_schedule(
    *,
    attempt_alpha_allocation: float,
    registered_look_stratum_counts: Sequence[tuple[int, int]],
    population_stratum_counts: tuple[int, int],
) -> tuple[float, ...]:
    """Spend attempt alpha uniformly over noncensus registered looks.

    The exact census consumes no alpha.  Allocations for unvisited future looks
    remain nontransferable under :data:`ALPHA_NONRECYCLING_POLICY`.
    """

    allocation = _unit_alpha(
        attempt_alpha_allocation, name="attempt alpha allocation"
    )
    looks = tuple(tuple(row) for row in registered_look_stratum_counts)
    partial_count = sum(tuple(row) != tuple(population_stratum_counts) for row in looks)
    if partial_count == 0:
        return tuple(0.0 for _ in looks)
    per_look = allocation / partial_count
    return tuple(
        0.0 if tuple(row) == tuple(population_stratum_counts) else per_look
        for row in looks
    )


def _context_set_hash(*, role: str, context_ids: Sequence[str]) -> str:
    identities = tuple(
        _identity(value, name=f"{role} context id") for value in context_ids
    )
    if len(identities) != len(set(identities)):
        raise ValueError(f"{role} context axis repeats an identity")
    return content_hash(
        {
            "schema": "modern_v1_context_set_identity_v1",
            "role": role,
            "context_ids_in_canonical_set_order": tuple(sorted(identities)),
        }
    )


@dataclass(frozen=True, slots=True)
class V1PairedBCEOracleRoleSeparationReceipt:
    """Compact proof receipt that the direct oracle fold was untouched.

    The constructor is intentionally private-by-convention; use
    :func:`register_paired_bce_oracle_role_separation`, which inspects the
    exact role axes before any candidate outcome exists.  Discovery and
    calibration may overlap each other, but neither (nor F/Q certification)
    may overlap either direct-oracle stratum.
    """

    discovery_fold_identity_hash: str
    fq_certification_fold_identity_hash: str
    calibration_fold_identity_hash: str
    oracle_fold_identity_hash: str
    discovery_context_set_hash: str
    fq_certification_context_set_hash: str
    calibration_context_set_hash: str
    oracle_target_population_identity_hash: str
    oracle_complement_population_identity_hash: str
    discovery_context_count: int
    fq_certification_context_count: int
    calibration_context_count: int
    oracle_target_context_count: int
    oracle_complement_context_count: int
    disjointness_verifier_identity_hash: str
    oracle_disjoint_from_discovery: bool = True
    oracle_disjoint_from_fq_certification: bool = True
    oracle_disjoint_from_calibration: bool = True
    oracle_strata_disjoint: bool = True
    oracle_untouched_before_candidate_freeze: bool = True
    candidate_calibration_excludes_oracle: bool = True
    schema: str = ORACLE_ROLE_SEPARATION_SCHEMA
    _receipt_hash_cache: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "discovery_fold_identity_hash",
            "fq_certification_fold_identity_hash",
            "calibration_fold_identity_hash",
            "oracle_fold_identity_hash",
            "discovery_context_set_hash",
            "fq_certification_context_set_hash",
            "calibration_context_set_hash",
            "oracle_target_population_identity_hash",
            "oracle_complement_population_identity_hash",
            "disjointness_verifier_identity_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "discovery_context_count",
            "fq_certification_context_count",
            "calibration_context_count",
            "oracle_target_context_count",
            "oracle_complement_context_count",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name),
            )
        for name in (
            "oracle_disjoint_from_discovery",
            "oracle_disjoint_from_fq_certification",
            "oracle_disjoint_from_calibration",
            "oracle_strata_disjoint",
            "oracle_untouched_before_candidate_freeze",
            "candidate_calibration_excludes_oracle",
        ):
            if _strict_bool(getattr(self, name), name=name) is not True:
                raise ValueError(f"paired-BCE oracle separation requires {name}=True")
        if self.schema != ORACLE_ROLE_SEPARATION_SCHEMA:
            raise ValueError("paired-BCE oracle role-separation schema is invalid")

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def _hash_payload(self) -> dict[str, Any]:
        return _constructor_payload(self)

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        return _assert_hash_integrity(
            self,
            payload=self._hash_payload(),
            cache_name="_receipt_hash_cache",
            label="paired-BCE oracle role-separation receipt",
            expected_hash=expected_hash,
        )

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return _rebuild_hashed, (
            type(self),
            _constructor_payload(self),
            self.assert_integrity(),
        )


def register_paired_bce_oracle_role_separation(
    *,
    discovery_fold_identity_hash: str,
    fq_certification_fold_identity_hash: str,
    calibration_fold_identity_hash: str,
    oracle_fold_identity_hash: str,
    discovery_context_ids: Sequence[str],
    fq_certification_context_ids: Sequence[str],
    calibration_context_ids: Sequence[str],
    oracle_target_context_ids: Sequence[str],
    oracle_complement_context_ids: Sequence[str],
    disjointness_verifier_identity_hash: str,
) -> V1PairedBCEOracleRoleSeparationReceipt:
    """Inspect exact axes and emit only their compact role-separation receipt."""

    axes = {
        "discovery": tuple(
            _identity(value, name="discovery context id")
            for value in discovery_context_ids
        ),
        "fq_certification": tuple(
            _identity(value, name="F/Q certification context id")
            for value in fq_certification_context_ids
        ),
        "calibration": tuple(
            _identity(value, name="calibration context id")
            for value in calibration_context_ids
        ),
        "oracle_target": tuple(
            _identity(value, name="oracle target context id")
            for value in oracle_target_context_ids
        ),
        "oracle_complement": tuple(
            _identity(value, name="oracle complement context id")
            for value in oracle_complement_context_ids
        ),
    }
    if any(not values or len(values) != len(set(values)) for values in axes.values()):
        raise ValueError("every paired-BCE role axis must be nonempty and unique")
    oracle_target = set(axes["oracle_target"])
    oracle_complement = set(axes["oracle_complement"])
    if oracle_target & oracle_complement:
        raise ValueError("paired-BCE target and complement oracle strata overlap")
    oracle = oracle_target | oracle_complement
    for role in ("discovery", "fq_certification", "calibration"):
        if oracle & set(axes[role]):
            raise ValueError(f"paired-BCE oracle overlaps the {role} role")
    return V1PairedBCEOracleRoleSeparationReceipt(
        discovery_fold_identity_hash=discovery_fold_identity_hash,
        fq_certification_fold_identity_hash=fq_certification_fold_identity_hash,
        calibration_fold_identity_hash=calibration_fold_identity_hash,
        oracle_fold_identity_hash=oracle_fold_identity_hash,
        discovery_context_set_hash=_context_set_hash(
            role="discovery", context_ids=axes["discovery"]
        ),
        fq_certification_context_set_hash=_context_set_hash(
            role="fq_certification", context_ids=axes["fq_certification"]
        ),
        calibration_context_set_hash=_context_set_hash(
            role="calibration", context_ids=axes["calibration"]
        ),
        oracle_target_population_identity_hash=_context_set_hash(
            role="oracle_target", context_ids=axes["oracle_target"]
        ),
        oracle_complement_population_identity_hash=_context_set_hash(
            role="oracle_complement", context_ids=axes["oracle_complement"]
        ),
        discovery_context_count=len(axes["discovery"]),
        fq_certification_context_count=len(axes["fq_certification"]),
        calibration_context_count=len(axes["calibration"]),
        oracle_target_context_count=len(axes["oracle_target"]),
        oracle_complement_context_count=len(axes["oracle_complement"]),
        disjointness_verifier_identity_hash=disjointness_verifier_identity_hash,
    )


def _calibration_paired_row_payload_hash(
    *,
    oracle_sampling_frame_receipt_hash: str,
    calibration_prefix_freeze_hash: str,
    dense_affine_calibration_hash: str,
    candidate_affine_calibration_hash: str,
    stratum_id: str,
    context_ids: Sequence[str],
    paired_loss_rows: Sequence[float],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"modern-v1-paired-bce-exact-k-row-payload-v1\0")
    for identity in (
        oracle_sampling_frame_receipt_hash,
        calibration_prefix_freeze_hash,
        dense_affine_calibration_hash,
        candidate_affine_calibration_hash,
        stratum_id,
    ):
        encoded = _identity(identity, name="exact-K row coordinate").encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    for context_id, value in zip(context_ids, paired_loss_rows, strict=True):
        encoded = _identity(context_id, name="exact-K context id").encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(struct.pack(">d", _finite(value, name="exact-K paired-loss row")))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class V1PairedBCEDynamicOracleReceipt:
    """Compact authority for one causally opened ``O = U \\ K`` attempt.

    The full context axes and paired-loss rows are inspected transiently by
    :func:`register_paired_bce_dynamic_oracle`.  This durable receipt retains
    only their content identities, exact binary64 K sums, finite-population
    counts, and the candidate/affine/attempt coordinates that were frozen
    before O outcomes.  It is therefore sufficient to reject stale replay
    without inflating checkpoints with the complete ImageNet train axis.
    """

    generation_frame_hash: str
    oracle_sampling_frame_receipt_hash: str
    generation_index: int
    candidate_attempt_index: int
    dense_reference_hash: str
    candidate_mask_hash: str
    candidate_calibration_freeze_hash: str
    dense_affine_calibration_hash: str
    candidate_affine_calibration_hash: str
    calibration_prefix_freeze_hash: str
    loss_ceiling_identity_hash: str
    candidate_attempt_alpha_coordinate_hash: str
    full_population_identity_hashes: tuple[str, str]
    full_population_stratum_counts: tuple[int, int]
    calibration_prefix_membership_hashes: tuple[str, str]
    calibration_prefix_stratum_counts: tuple[int, int]
    exact_calibration_scaled_loss_sums: tuple[int, int]
    calibration_paired_row_payload_hashes: tuple[str, str]
    calibration_paired_loss_evaluator_receipt_hash: str
    oracle_population_identity_hashes: tuple[str, str]
    oracle_population_stratum_counts: tuple[int, int]
    oracle_sealed_order_hashes: tuple[str, str]
    oracle_permutation_receipt_hashes: tuple[str, str]
    calibration_prefix_frozen_before_oracle_open: bool = True
    oracle_is_exact_full_population_minus_calibration_prefix: bool = True
    candidate_and_calibrations_immutable_after_oracle_open: bool = True
    replacement_requires_new_attempt_alpha_coordinate: bool = True
    terminal_residual_census_plus_exact_k_is_full_population_census: bool = True
    schema: str = DYNAMIC_ORACLE_RECEIPT_SCHEMA
    _receipt_hash_cache: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "generation_frame_hash",
            "oracle_sampling_frame_receipt_hash",
            "dense_reference_hash",
            "candidate_mask_hash",
            "candidate_calibration_freeze_hash",
            "dense_affine_calibration_hash",
            "candidate_affine_calibration_hash",
            "calibration_prefix_freeze_hash",
            "loss_ceiling_identity_hash",
            "candidate_attempt_alpha_coordinate_hash",
            "calibration_paired_loss_evaluator_receipt_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        generation = _nonnegative_int(self.generation_index, name="generation index")
        attempt = _nonnegative_int(
            self.candidate_attempt_index, name="candidate attempt index"
        )
        full_hashes = _pair_of_identities(
            self.full_population_identity_hashes,
            name="full population identity hashes",
        )
        raw_full_counts = (
            tuple(self.full_population_stratum_counts)
            if isinstance(self.full_population_stratum_counts, (tuple, list))
            else ()
        )
        if len(raw_full_counts) != 2:
            raise ValueError("full population stratum counts must contain two entries")
        full_counts = (
            _positive_int(raw_full_counts[0], name="target full population"),
            _positive_int(raw_full_counts[1], name="complement full population"),
        )
        k_hashes = _pair_of_identities(
            self.calibration_prefix_membership_hashes,
            name="calibration prefix membership hashes",
        )
        k_counts = _pair_of_nonnegative_ints(
            self.calibration_prefix_stratum_counts,
            name="calibration prefix stratum counts",
        )
        if any(value == 0 for value in k_counts):
            raise ValueError("dynamic paired-BCE calibration prefix must cover both strata")
        k_sums = _pair_of_ints(
            self.exact_calibration_scaled_loss_sums,
            name="exact calibration scaled loss sums",
        )
        k_payloads = _pair_of_identities(
            self.calibration_paired_row_payload_hashes,
            name="calibration paired-row payload hashes",
        )
        oracle_hashes = _pair_of_identities(
            self.oracle_population_identity_hashes,
            name="oracle population identity hashes",
        )
        raw_oracle_counts = (
            tuple(self.oracle_population_stratum_counts)
            if isinstance(self.oracle_population_stratum_counts, (tuple, list))
            else ()
        )
        if len(raw_oracle_counts) != 2:
            raise ValueError("oracle population stratum counts must contain two entries")
        oracle_counts = (
            _positive_int(raw_oracle_counts[0], name="target oracle population"),
            _positive_int(
                raw_oracle_counts[1], name="complement oracle population"
            ),
        )
        order_hashes = _pair_of_identities(
            self.oracle_sealed_order_hashes,
            name="oracle sealed order hashes",
        )
        permutation_hashes = _pair_of_identities(
            self.oracle_permutation_receipt_hashes,
            name="oracle permutation receipt hashes",
        )
        if any(
            k_counts[index] + oracle_counts[index] != full_counts[index]
            for index in (0, 1)
        ):
            raise ValueError("dynamic oracle populations do not exactly exhaust U minus K")
        if self.dense_reference_hash == self.candidate_mask_hash:
            raise ValueError("dynamic paired-BCE candidate equals its dense referent")
        for name in (
            "calibration_prefix_frozen_before_oracle_open",
            "oracle_is_exact_full_population_minus_calibration_prefix",
            "candidate_and_calibrations_immutable_after_oracle_open",
            "replacement_requires_new_attempt_alpha_coordinate",
            "terminal_residual_census_plus_exact_k_is_full_population_census",
        ):
            if _strict_bool(getattr(self, name), name=name) is not True:
                raise PermissionError(f"dynamic paired-BCE authority requires {name}=True")
        if self.schema != DYNAMIC_ORACLE_RECEIPT_SCHEMA:
            raise ValueError("dynamic paired-BCE oracle receipt schema is invalid")
        object.__setattr__(self, "generation_index", generation)
        object.__setattr__(self, "candidate_attempt_index", attempt)
        object.__setattr__(self, "full_population_identity_hashes", full_hashes)
        object.__setattr__(self, "full_population_stratum_counts", full_counts)
        object.__setattr__(self, "calibration_prefix_membership_hashes", k_hashes)
        object.__setattr__(self, "calibration_prefix_stratum_counts", k_counts)
        object.__setattr__(self, "exact_calibration_scaled_loss_sums", k_sums)
        object.__setattr__(self, "calibration_paired_row_payload_hashes", k_payloads)
        object.__setattr__(self, "oracle_population_identity_hashes", oracle_hashes)
        object.__setattr__(self, "oracle_population_stratum_counts", oracle_counts)
        object.__setattr__(self, "oracle_sealed_order_hashes", order_hashes)
        object.__setattr__(self, "oracle_permutation_receipt_hashes", permutation_hashes)

    @property
    def source_population_identity_hash(self) -> str:
        return content_hash(
            {
                "schema": "modern_v1_paired_bce_full_train_population_pair_v1",
                "strata": tuple(
                    (
                        name,
                        self.full_population_identity_hashes[index],
                        self.full_population_stratum_counts[index],
                    )
                    for index, name in enumerate(PAIRED_BCE_STRATA)
                ),
            }
        )

    @property
    def oracle_order_entropy_commitment_hash(self) -> str:
        return content_hash(
            {
                "schema": "modern_v1_paired_bce_dynamic_oracle_order_commitment_v1",
                "oracle_sampling_frame_receipt_hash": (
                    self.oracle_sampling_frame_receipt_hash
                ),
                "oracle_sealed_order_hashes": self.oracle_sealed_order_hashes,
                "oracle_permutation_receipt_hashes": (
                    self.oracle_permutation_receipt_hashes
                ),
                "candidate_attempt_index": self.candidate_attempt_index,
            }
        )

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def _hash_payload(self) -> dict[str, Any]:
        return _constructor_payload(self)

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        return _assert_hash_integrity(
            self,
            payload=self._hash_payload(),
            cache_name="_receipt_hash_cache",
            label="dynamic paired-BCE oracle receipt",
            expected_hash=expected_hash,
        )

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return _rebuild_hashed, (
            type(self),
            _constructor_payload(self),
            self.assert_integrity(),
        )


def register_paired_bce_dynamic_oracle(
    *,
    oracle_sampling_frame: object,
    candidate_calibration_freeze_hash: str,
    loss_ceiling_identity_hash: str,
    candidate_attempt_alpha_coordinate_hash: str,
    calibration_paired_loss_rows_by_stratum: Sequence[
        tuple[str, Sequence[float]]
    ],
    calibration_paired_loss_evaluator_receipt_hash: str,
) -> V1PairedBCEDynamicOracleReceipt:
    """Inspect an AlexNet causal O frame and reduce frozen K rows compactly."""

    # Local import keeps this generic statistical core free of an import cycle.
    from .modern_v1_alexnet_fq_registration import (  # noqa: PLC0415
        V1AlexNetFQOracleSamplingFrameReceipt,
    )

    if type(oracle_sampling_frame) is not V1AlexNetFQOracleSamplingFrameReceipt:
        raise TypeError("dynamic paired-BCE registration requires the exact O frame")
    frame = oracle_sampling_frame
    frame.assert_integrity()
    k_axes = dict(frame.calibration_prefix_context_ids_by_stratum)
    o_axes = dict(frame.oracle_context_ids_by_stratum)
    raw_rows = dict(calibration_paired_loss_rows_by_stratum)
    if set(k_axes) != set(PAIRED_BCE_STRATA) or set(o_axes) != set(PAIRED_BCE_STRATA):
        raise ValueError("dynamic O frame must name target and complement")
    if set(raw_rows) != set(PAIRED_BCE_STRATA):
        raise ValueError("exact-K paired-loss rows must name target and complement")
    scaled_sums: list[int] = []
    payload_hashes: list[str] = []
    k_membership_hashes: list[str] = []
    full_identity_hashes: list[str] = []
    o_identity_hashes: list[str] = []
    o_order_hashes: list[str] = []
    permutation_hashes: list[str] = []
    receipts = (
        frame.target_assignment_receipt,
        frame.complement_assignment_receipt,
    )
    declared_full_identity_hashes = (
        frame.target_full_population_membership_hash,
        frame.complement_full_population_membership_hash,
    )
    for index, name in enumerate(PAIRED_BCE_STRATA):
        k_contexts = tuple(k_axes[name])
        o_contexts = tuple(o_axes[name])
        rows = tuple(
            _finite(value, name=f"{name} exact-K paired-loss row")
            for value in raw_rows[name]
        )
        if len(rows) != len(k_contexts):
            raise ValueError("exact-K paired-loss rows differ from the frozen K prefix")
        if (
            len(k_contexts) != len(set(k_contexts))
            or len(o_contexts) != len(set(o_contexts))
            or set(k_contexts).intersection(o_contexts)
        ):
            raise ValueError("dynamic O frame does not provide disjoint unique K/O axes")
        full_contexts = k_contexts + o_contexts
        if len(full_contexts) != frame.full_population_stratum_counts[index]:
            raise ValueError("dynamic K/O axes do not exhaust the declared full stratum")
        scaled_sums.append(sum(_binary64_scaled_integer(value) for value in rows))
        payload_hashes.append(
            _calibration_paired_row_payload_hash(
                oracle_sampling_frame_receipt_hash=frame.receipt_hash,
                calibration_prefix_freeze_hash=frame.calibration_prefix_freeze_hash,
                dense_affine_calibration_hash=frame.dense_calibration_hash,
                candidate_affine_calibration_hash=frame.candidate_calibration_hash,
                stratum_id=name,
                context_ids=k_contexts,
                paired_loss_rows=rows,
            )
        )
        k_membership_hashes.append(
            _context_set_hash(role=f"calibration_{name}", context_ids=k_contexts)
        )
        # The frame has already verified this hash against the exact K/O union.
        # Retaining that same identity makes the compact receipt directly
        # comparable to the generation owner, rather than inventing a second
        # content-addressing domain for the same full-train stratum.
        full_identity_hashes.append(declared_full_identity_hashes[index])
        o_identity_hashes.append(
            _context_set_hash(role=f"oracle_{name}", context_ids=o_contexts)
        )
        o_order_hashes.append(
            paired_bce_sealed_order_hash(
                stratum_id=name,
                context_ids_in_sealed_order=o_contexts,
            )
        )
        receipts[index].verify()
        permutation_hashes.append(receipts[index].content_hash)
    return V1PairedBCEDynamicOracleReceipt(
        generation_frame_hash=frame.generation_frame_hash,
        oracle_sampling_frame_receipt_hash=frame.receipt_hash,
        generation_index=frame.generation_index,
        candidate_attempt_index=frame.candidate_attempt_index,
        dense_reference_hash=frame.dense_referent_mask_hash,
        candidate_mask_hash=frame.candidate_mask_hash,
        candidate_calibration_freeze_hash=candidate_calibration_freeze_hash,
        dense_affine_calibration_hash=frame.dense_calibration_hash,
        candidate_affine_calibration_hash=frame.candidate_calibration_hash,
        calibration_prefix_freeze_hash=frame.calibration_prefix_freeze_hash,
        loss_ceiling_identity_hash=loss_ceiling_identity_hash,
        candidate_attempt_alpha_coordinate_hash=(
            candidate_attempt_alpha_coordinate_hash
        ),
        full_population_identity_hashes=(
            full_identity_hashes[0],
            full_identity_hashes[1],
        ),
        full_population_stratum_counts=frame.full_population_stratum_counts,
        calibration_prefix_membership_hashes=(
            k_membership_hashes[0],
            k_membership_hashes[1],
        ),
        calibration_prefix_stratum_counts=(
            len(k_axes[TARGET_STRATUM]),
            len(k_axes[COMPLEMENT_STRATUM]),
        ),
        exact_calibration_scaled_loss_sums=(scaled_sums[0], scaled_sums[1]),
        calibration_paired_row_payload_hashes=(
            payload_hashes[0],
            payload_hashes[1],
        ),
        calibration_paired_loss_evaluator_receipt_hash=(
            calibration_paired_loss_evaluator_receipt_hash
        ),
        oracle_population_identity_hashes=(
            o_identity_hashes[0],
            o_identity_hashes[1],
        ),
        oracle_population_stratum_counts=frame.oracle_population_stratum_counts,
        oracle_sealed_order_hashes=(o_order_hashes[0], o_order_hashes[1]),
        oracle_permutation_receipt_hashes=(
            permutation_hashes[0],
            permutation_hashes[1],
        ),
    )


@dataclass(frozen=True, slots=True)
class V1PairedBCEStratumRegistration:
    """Pre-outcome population, order, and pointwise-bound contract."""

    stratum_id: str
    stratum_weight: float
    population_size: int
    population_identity_hash: str
    sealed_order_hash: str
    sealed_permutation_receipt_hash: str
    bound_certificate_hash: str
    paired_loss_lower_bound: float | None
    paired_loss_upper_bound: float | None
    bound_frozen_before_rows: bool = True
    without_replacement: bool = True
    schema: str = STRATUM_REGISTRATION_SCHEMA
    _content_hash_cache: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        identity = _identity(self.stratum_id, name="stratum id")
        if identity not in PAIRED_BCE_STRATA:
            raise ValueError("paired-BCE stratum must be target or complement")
        weight = _finite(self.stratum_weight, name="stratum weight")
        if weight != 0.5:
            raise ValueError("paired-BCE strata require exact weight 0.5")
        object.__setattr__(
            self,
            "population_size",
            _positive_int(self.population_size, name="stratum population size"),
        )
        for name in (
            "population_identity_hash",
            "sealed_order_hash",
            "sealed_permutation_receipt_hash",
            "bound_certificate_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        lower = _optional_finite(
            self.paired_loss_lower_bound, name="paired-loss lower bound"
        )
        upper = _optional_finite(
            self.paired_loss_upper_bound, name="paired-loss upper bound"
        )
        if (lower is None) != (upper is None):
            raise ValueError("paired-loss bounds must be supplied together or omitted")
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError("paired-loss bounds must be nondegenerate")
        if _strict_bool(
            self.bound_frozen_before_rows, name="bound frozen before rows"
        ) is not True:
            raise ValueError("paired-loss bound must be frozen before rows")
        if _strict_bool(
            self.without_replacement, name="without replacement"
        ) is not True:
            raise ValueError("paired-BCE acquisition requires without-replacement rows")
        if self.schema != STRATUM_REGISTRATION_SCHEMA:
            raise ValueError("paired-BCE stratum registration schema is invalid")
        object.__setattr__(self, "stratum_id", identity)
        object.__setattr__(self, "stratum_weight", weight)
        object.__setattr__(self, "paired_loss_lower_bound", lower)
        object.__setattr__(self, "paired_loss_upper_bound", upper)

    @property
    def finite_partial_bound_available(self) -> bool:
        lower, upper = self.paired_loss_lower_bound, self.paired_loss_upper_bound
        return bool(
            lower is not None
            and upper is not None
            and math.isfinite(upper - lower)
            and upper - lower >= sys.float_info.min
            and upper > lower
        )

    @property
    def bound_span(self) -> float | None:
        if not self.finite_partial_bound_available:
            return None
        assert self.paired_loss_lower_bound is not None
        assert self.paired_loss_upper_bound is not None
        return self.paired_loss_upper_bound - self.paired_loss_lower_bound

    def _hash_payload(self) -> dict[str, Any]:
        return _constructor_payload(self)

    @property
    def content_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        return _assert_hash_integrity(
            self,
            payload=self._hash_payload(),
            cache_name="_content_hash_cache",
            label="paired-BCE stratum registration",
            expected_hash=expected_hash,
        )

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return _rebuild_hashed, (
            type(self),
            _constructor_payload(self),
            self.assert_integrity(),
        )


@dataclass(frozen=True, slots=True)
class V1PairedBCEAcquisitionRegistration:
    """Complete pre-outcome registration for one global candidate attempt."""

    registration_id: str
    campaign_contract_hash: str
    candidate_mask_hash: str
    dense_reference_hash: str
    candidate_calibration_freeze_hash: str
    paired_loss_kind: str
    loss_ceiling: float
    loss_ceiling_identity_hash: str
    source_population_identity_hash: str
    oracle_fold_identity_hash: str
    oracle_role_separation_receipt: V1PairedBCEOracleRoleSeparationReceipt | None
    strata: tuple[V1PairedBCEStratumRegistration, V1PairedBCEStratumRegistration]
    oracle_order_entropy_commitment_hash: str
    sampling_law_identity_hash: str
    sealed_order_slice_verifier_identity_hash: str
    candidate_attempt_alpha_axis_hash: str
    candidate_attempt_alpha_coordinate_hash: str
    candidate_attempt_index: int
    maximum_candidate_attempts: int
    oracle_familywise_alpha: float
    alpha_committed_before_attempt: float
    attempt_alpha_allocation: float
    registered_look_stratum_counts: tuple[tuple[int, int], ...]
    alpha_by_look: tuple[float, ...]
    dynamic_oracle_receipt: V1PairedBCEDynamicOracleReceipt | None = None
    stratum_populations_disjoint: bool = True
    identities_frozen_before_rows: bool = True
    alpha_allocation_nonrecycling: bool = True
    alpha_nonrecycling_policy: str = ALPHA_NONRECYCLING_POLICY
    partial_look_alpha_policy: str = PARTIAL_LOOK_ALPHA_POLICY
    sampling_law_kind: str = SEALED_HMAC_WOR_SAMPLING_LAW
    predictable_stratum_selection_policy: str = PREDICTABLE_STRATUM_POLICY
    group_sequential_fallback_enabled: bool = False
    partial_evidence_method: str = CANONICAL_ANYTIME_BETTING_METHOD
    partial_evidence_ceiling: str = CANONICAL_ANYTIME_BETTING_EVIDENCE_CEILING
    authority_scope: str = AUTHORITY_SCOPE
    schema: str = ACQUISITION_REGISTRATION_SCHEMA
    _registration_hash_cache: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "registration_id", _identity(self.registration_id, name="registration id")
        )
        for name in (
            "campaign_contract_hash",
            "candidate_mask_hash",
            "dense_reference_hash",
            "candidate_calibration_freeze_hash",
            "loss_ceiling_identity_hash",
            "source_population_identity_hash",
            "oracle_fold_identity_hash",
            "oracle_order_entropy_commitment_hash",
            "sampling_law_identity_hash",
            "sealed_order_slice_verifier_identity_hash",
            "candidate_attempt_alpha_axis_hash",
            "candidate_attempt_alpha_coordinate_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        loss_kind = _identity(self.paired_loss_kind, name="paired loss kind")
        if loss_kind not in (
            *PARTIAL_BOUND_CAPABLE_LOSS_KINDS,
            CENSUS_ONLY_PAIRED_BCE_KIND,
        ):
            raise ValueError("paired-loss kind is not supported by this acquisition core")
        ceiling = _finite(self.loss_ceiling, name="loss ceiling")
        expected_ceiling_hash = paired_bce_loss_ceiling_identity_hash(
            dense_reference_hash=self.dense_reference_hash,
            candidate_calibration_freeze_hash=(
                self.candidate_calibration_freeze_hash
            ),
            paired_loss_kind=loss_kind,
            loss_ceiling=ceiling,
        )
        if self.loss_ceiling_identity_hash != expected_ceiling_hash:
            raise ValueError("loss ceiling differs from its frozen identity")

        strata = tuple(self.strata)
        if len(strata) != 2 or any(
            type(item) is not V1PairedBCEStratumRegistration for item in strata
        ):
            raise TypeError("paired-BCE registration requires two canonical strata")
        if tuple(item.stratum_id for item in strata) != PAIRED_BCE_STRATA:
            raise ValueError("paired-BCE strata must use target, complement order")
        if tuple(item.stratum_weight for item in strata) != PAIRED_BCE_STRATUM_WEIGHTS:
            raise ValueError("paired-BCE registration requires equal stratum weights")
        for item in strata:
            item.assert_integrity()
        separation = self.oracle_role_separation_receipt
        dynamic = self.dynamic_oracle_receipt
        if dynamic is None:
            if type(separation) is not V1PairedBCEOracleRoleSeparationReceipt:
                raise TypeError(
                    "legacy paired-BCE registration requires its role-separation receipt"
                )
            separation.assert_integrity()
            if separation.oracle_fold_identity_hash != self.oracle_fold_identity_hash:
                raise ValueError("paired-BCE oracle fold differs from its separation receipt")
            expected_population_hashes = (
                separation.oracle_target_population_identity_hash,
                separation.oracle_complement_population_identity_hash,
            )
            expected_population_counts = (
                separation.oracle_target_context_count,
                separation.oracle_complement_context_count,
            )
        else:
            if separation is not None:
                raise PermissionError(
                    "dynamic paired-BCE authority cannot come from a static role manifest"
                )
            if type(dynamic) is not V1PairedBCEDynamicOracleReceipt:
                raise TypeError("paired-BCE dynamic oracle receipt has the wrong type")
            dynamic.assert_integrity()
            expected_population_hashes = dynamic.oracle_population_identity_hashes
            expected_population_counts = dynamic.oracle_population_stratum_counts
            if (
                self.oracle_fold_identity_hash
                != dynamic.oracle_sampling_frame_receipt_hash
                or self.source_population_identity_hash
                != dynamic.source_population_identity_hash
                or self.oracle_order_entropy_commitment_hash
                != dynamic.oracle_order_entropy_commitment_hash
                or self.candidate_mask_hash != dynamic.candidate_mask_hash
                or self.dense_reference_hash != dynamic.dense_reference_hash
                or self.candidate_calibration_freeze_hash
                != dynamic.candidate_calibration_freeze_hash
                or self.loss_ceiling_identity_hash
                != dynamic.loss_ceiling_identity_hash
                or self.candidate_attempt_index != dynamic.candidate_attempt_index
                or self.candidate_attempt_alpha_coordinate_hash
                != dynamic.candidate_attempt_alpha_coordinate_hash
            ):
                raise ValueError(
                    "dynamic paired-BCE receipt has stale frame/candidate/calibration/"
                    "ceiling/attempt coordinates"
                )
        if tuple(item.population_identity_hash for item in strata) != (
            expected_population_hashes
        ):
            raise ValueError(
                "paired-BCE oracle stratum populations differ from their authority receipt"
            )
        if tuple(item.population_size for item in strata) != expected_population_counts:
            raise ValueError(
                "paired-BCE oracle stratum counts differ from their authority receipt"
            )
        if dynamic is not None and (
            tuple(item.sealed_order_hash for item in strata)
            != dynamic.oracle_sealed_order_hashes
            or tuple(item.sealed_permutation_receipt_hash for item in strata)
            != dynamic.oracle_permutation_receipt_hashes
        ):
            raise ValueError("paired-BCE strata name stale dynamic O permutations")
        if loss_kind == CENSUS_ONLY_PAIRED_BCE_KIND and any(
            item.finite_partial_bound_available for item in strata
        ):
            raise ValueError(
                "census-only paired BCE cannot advertise partial bound authority"
            )
        if loss_kind in PARTIAL_BOUND_CAPABLE_LOSS_KINDS and any(
            item.paired_loss_lower_bound is None for item in strata
        ):
            # A bounded loss kind may temporarily lack a certificate, but then
            # the entire attempt is correctly partial-inert until census.
            pass

        index = _nonnegative_int(
            self.candidate_attempt_index, name="candidate attempt index"
        )
        maximum = _positive_int(
            self.maximum_candidate_attempts, name="maximum candidate attempts"
        )
        if index >= maximum:
            raise ValueError("candidate attempt index leaves its global alpha axis")
        familywise = _unit_alpha(
            self.oracle_familywise_alpha, name="oracle familywise alpha"
        )
        committed = _finite(
            self.alpha_committed_before_attempt,
            name="alpha committed before attempt",
        )
        if committed < 0.0:
            raise ValueError("alpha committed before attempt cannot be negative")
        allocation = _unit_alpha(
            self.attempt_alpha_allocation, name="attempt alpha allocation"
        )
        tolerance = 32.0 * math.ulp(max(1.0, familywise, committed, allocation))
        if committed + allocation > familywise + tolerance:
            raise ValueError("candidate-attempt alpha exceeds the oracle family reserve")
        expected_coordinate_hash = paired_bce_attempt_alpha_coordinate_hash(
            candidate_attempt_alpha_axis_hash=self.candidate_attempt_alpha_axis_hash,
            candidate_attempt_index=index,
            maximum_candidate_attempts=maximum,
            attempt_alpha_allocation=allocation,
        )
        if self.candidate_attempt_alpha_coordinate_hash != expected_coordinate_hash:
            raise ValueError("candidate-attempt alpha coordinate does not bind the attempt")

        populations = tuple(item.population_size for item in strata)
        looks = tuple(
            _pair_of_nonnegative_ints(row, name="registered look counts")
            for row in self.registered_look_stratum_counts
        )
        if not looks:
            raise ValueError("paired-BCE acquisition requires registered looks")
        previous = (0, 0)
        for row in looks:
            if any(value <= 0 for value in row):
                raise ValueError("every registered look must observe both strata")
            if any(value > total for value, total in zip(row, populations, strict=True)):
                raise ValueError("registered look exceeds a finite population")
            if any(value < old for value, old in zip(row, previous, strict=True)):
                raise ValueError("registered paired-BCE looks must be nested")
            if row == previous:
                raise ValueError("each registered look must advance at least one stratum")
            previous = row
        if looks[-1] != populations:
            raise ValueError("registered paired-BCE schedule must end at joint census")
        alphas = tuple(float(value) for value in self.alpha_by_look)
        if len(alphas) != len(looks) or any(
            not math.isfinite(value) or value < 0.0 or value >= 1.0
            for value in alphas
        ):
            raise ValueError("paired-BCE look alpha schedule is invalid")
        for row, value in zip(looks, alphas, strict=True):
            if row == populations and value != 0.0:
                raise ValueError("exact census must not spend concentration alpha")
            if row != populations and value <= 0.0:
                raise ValueError("every partial registered look requires positive alpha")
        if math.fsum(alphas) > allocation + tolerance:
            raise ValueError("registered look alpha exceeds candidate-attempt allocation")

        for name in (
            "stratum_populations_disjoint",
            "identities_frozen_before_rows",
            "alpha_allocation_nonrecycling",
        ):
            if _strict_bool(getattr(self, name), name=name) is not True:
                raise ValueError(f"{name} must remain true")
        _strict_bool(
            self.group_sequential_fallback_enabled,
            name="group sequential fallback enabled",
        )
        expected_constants = {
            "alpha_nonrecycling_policy": ALPHA_NONRECYCLING_POLICY,
            "partial_look_alpha_policy": PARTIAL_LOOK_ALPHA_POLICY,
            "sampling_law_kind": SEALED_HMAC_WOR_SAMPLING_LAW,
            "predictable_stratum_selection_policy": PREDICTABLE_STRATUM_POLICY,
            "partial_evidence_method": CANONICAL_ANYTIME_BETTING_METHOD,
            "partial_evidence_ceiling": (
                CANONICAL_ANYTIME_BETTING_EVIDENCE_CEILING
            ),
            "authority_scope": AUTHORITY_SCOPE,
            "schema": ACQUISITION_REGISTRATION_SCHEMA,
        }
        for name, expected in expected_constants.items():
            if getattr(self, name) != expected:
                raise ValueError(f"paired-BCE registration has invalid {name}")

        object.__setattr__(self, "paired_loss_kind", loss_kind)
        object.__setattr__(self, "loss_ceiling", ceiling)
        object.__setattr__(self, "strata", strata)
        object.__setattr__(self, "oracle_role_separation_receipt", separation)
        object.__setattr__(self, "dynamic_oracle_receipt", dynamic)
        object.__setattr__(self, "candidate_attempt_index", index)
        object.__setattr__(self, "maximum_candidate_attempts", maximum)
        object.__setattr__(self, "oracle_familywise_alpha", familywise)
        object.__setattr__(self, "alpha_committed_before_attempt", committed)
        object.__setattr__(self, "attempt_alpha_allocation", allocation)
        object.__setattr__(self, "registered_look_stratum_counts", looks)
        object.__setattr__(self, "alpha_by_look", alphas)

    @property
    def population_stratum_counts(self) -> tuple[int, int]:
        return tuple(item.population_size for item in self.strata)  # type: ignore[return-value]

    @property
    def full_population_stratum_counts(self) -> tuple[int, int]:
        dynamic = self.dynamic_oracle_receipt
        if dynamic is None:
            return self.population_stratum_counts
        return dynamic.full_population_stratum_counts

    @property
    def calibration_prefix_stratum_counts(self) -> tuple[int, int]:
        dynamic = self.dynamic_oracle_receipt
        if dynamic is None:
            return (0, 0)
        return dynamic.calibration_prefix_stratum_counts

    @property
    def dynamic_oracle_receipt_hash(self) -> str | None:
        dynamic = self.dynamic_oracle_receipt
        return None if dynamic is None else dynamic.receipt_hash

    @property
    def has_dynamic_full_population_authority(self) -> bool:
        return self.dynamic_oracle_receipt is not None

    @property
    def finite_partial_bounds_available(self) -> bool:
        return bool(
            self.paired_loss_kind in PARTIAL_BOUND_CAPABLE_LOSS_KINDS
            and all(item.finite_partial_bound_available for item in self.strata)
        )

    @property
    def registration_hash(self) -> str:
        return self.assert_integrity()

    @property
    def total_registered_alpha(self) -> float:
        return math.fsum(self.alpha_by_look)

    def stratum(self, stratum_id: str) -> V1PairedBCEStratumRegistration:
        identity = _identity(stratum_id, name="stratum id")
        try:
            return self.strata[PAIRED_BCE_STRATA.index(identity)]
        except ValueError as error:
            raise KeyError(f"unknown paired-BCE stratum {identity!r}") from error

    def _hash_payload(self) -> dict[str, Any]:
        payload = _constructor_payload(self)
        payload["stratum_registration_hashes"] = tuple(
            item.assert_integrity() for item in self.strata
        )
        payload["oracle_role_separation_receipt_hash"] = (
            None
            if self.oracle_role_separation_receipt is None
            else self.oracle_role_separation_receipt.assert_integrity()
        )
        payload["dynamic_oracle_receipt_hash"] = (
            None
            if self.dynamic_oracle_receipt is None
            else self.dynamic_oracle_receipt.assert_integrity()
        )
        payload.pop("strata")
        payload.pop("oracle_role_separation_receipt")
        payload.pop("dynamic_oracle_receipt")
        return payload

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        return _assert_hash_integrity(
            self,
            payload=self._hash_payload(),
            cache_name="_registration_hash_cache",
            label="paired-BCE acquisition registration",
            expected_hash=expected_hash,
        )

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return _rebuild_hashed, (
            type(self),
            _constructor_payload(self),
            self.assert_integrity(),
        )


def build_paired_bce_acquisition_registration(
    *,
    initial_look_count: int = 1,
    look_growth_factor: int = 2,
    **payload: Any,
) -> V1PairedBCEAcquisitionRegistration:
    """Build a canonical geometric schedule and uniform partial-look spend."""

    strata = tuple(payload["strata"])
    if len(strata) != 2:
        raise ValueError("paired-BCE registration builder requires two strata")
    populations = tuple(int(item.population_size) for item in strata)
    looks = geometric_paired_bce_look_schedule(
        populations[0],
        populations[1],
        initial_count=initial_look_count,
        growth_factor=look_growth_factor,
    )
    allocation = float(payload["attempt_alpha_allocation"])
    alphas = uniform_partial_look_alpha_schedule(
        attempt_alpha_allocation=allocation,
        registered_look_stratum_counts=looks,
        population_stratum_counts=(populations[0], populations[1]),
    )
    if "registered_look_stratum_counts" in payload or "alpha_by_look" in payload:
        raise ValueError("registration builder owns the look and alpha schedules")
    return V1PairedBCEAcquisitionRegistration(
        **payload,
        registered_look_stratum_counts=looks,
        alpha_by_look=alphas,
    )


@dataclass(frozen=True, slots=True)
class V1PairedBCEAcquisitionState:
    """Compact replay state; no context axis or paired-loss vector is retained."""

    registration_hash: str
    candidate_attempt_index: int
    stratum_counts: tuple[int, int]
    exact_scaled_loss_sums: tuple[int, int]
    observed_minima: tuple[float | None, float | None]
    observed_maxima: tuple[float | None, float | None]
    completed_look_count: int
    completed_chunk_count: int
    last_chunk_receipt_hash: str | None
    receipt_chain_hash: str
    dynamic_oracle_receipt_hash: str | None = None
    schema: str = ACQUISITION_STATE_SCHEMA
    _state_hash_cache: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "registration_hash",
            _identity(self.registration_hash, name="registration hash"),
        )
        index = _nonnegative_int(
            self.candidate_attempt_index, name="candidate attempt index"
        )
        counts = _pair_of_nonnegative_ints(self.stratum_counts, name="stratum counts")
        sums = tuple(self.exact_scaled_loss_sums)
        if len(sums) != 2 or any(isinstance(value, bool) or not isinstance(value, int) for value in sums):
            raise ValueError("exact scaled loss sums must contain two integers")
        minima = _pair_of_optional_finite(self.observed_minima, name="observed minima")
        maxima = _pair_of_optional_finite(self.observed_maxima, name="observed maxima")
        for count, minimum, maximum in zip(counts, minima, maxima, strict=True):
            if count == 0 and (minimum is not None or maximum is not None):
                raise ValueError("empty stratum cannot retain observed extrema")
            if count > 0 and (minimum is None or maximum is None):
                raise ValueError("observed stratum requires finite extrema")
            if minimum is not None and maximum is not None and minimum > maximum:
                raise ValueError("observed stratum extrema are inverted")
        completed = _nonnegative_int(
            self.completed_look_count, name="completed look count"
        )
        chunk_count = _nonnegative_int(
            self.completed_chunk_count, name="completed chunk count"
        )
        last_receipt = self.last_chunk_receipt_hash
        if last_receipt is not None:
            last_receipt = _identity(last_receipt, name="last chunk receipt hash")
        chain = _identity(self.receipt_chain_hash, name="receipt chain hash")
        dynamic_hash = self.dynamic_oracle_receipt_hash
        if dynamic_hash is not None:
            dynamic_hash = _identity(
                dynamic_hash, name="dynamic oracle receipt hash"
            )
        if chunk_count == 0 and (
            last_receipt is not None or chain != _EMPTY_RECEIPT_CHAIN_HASH
        ):
            raise ValueError("empty acquisition state has a non-genesis receipt chain")
        if chunk_count > 0 and (
            last_receipt is None or chain == _EMPTY_RECEIPT_CHAIN_HASH
        ):
            raise ValueError("nonempty acquisition state lacks its receipt-chain head")
        if self.schema != ACQUISITION_STATE_SCHEMA:
            raise ValueError("paired-BCE acquisition state schema is invalid")
        object.__setattr__(self, "candidate_attempt_index", index)
        object.__setattr__(self, "stratum_counts", counts)
        object.__setattr__(self, "exact_scaled_loss_sums", (int(sums[0]), int(sums[1])))
        object.__setattr__(self, "observed_minima", minima)
        object.__setattr__(self, "observed_maxima", maxima)
        object.__setattr__(self, "completed_look_count", completed)
        object.__setattr__(self, "completed_chunk_count", chunk_count)
        object.__setattr__(self, "last_chunk_receipt_hash", last_receipt)
        object.__setattr__(self, "receipt_chain_hash", chain)
        object.__setattr__(self, "dynamic_oracle_receipt_hash", dynamic_hash)

    @property
    def chunk_count(self) -> int:
        return self.completed_chunk_count

    @property
    def state_hash(self) -> str:
        return self.assert_integrity()

    def _hash_payload(self) -> dict[str, Any]:
        return _constructor_payload(self)

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        return _assert_hash_integrity(
            self,
            payload=self._hash_payload(),
            cache_name="_state_hash_cache",
            label="paired-BCE acquisition state",
            expected_hash=expected_hash,
        )

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return _rebuild_hashed, (
            type(self),
            _constructor_payload(self),
            self.assert_integrity(),
        )


def initial_paired_bce_acquisition_state(
    registration: V1PairedBCEAcquisitionRegistration,
) -> V1PairedBCEAcquisitionState:
    if type(registration) is not V1PairedBCEAcquisitionRegistration:
        raise TypeError("initial paired-BCE state requires the canonical registration")
    return V1PairedBCEAcquisitionState(
        registration_hash=registration.assert_integrity(),
        candidate_attempt_index=registration.candidate_attempt_index,
        stratum_counts=(0, 0),
        exact_scaled_loss_sums=(0, 0),
        observed_minima=(None, None),
        observed_maxima=(None, None),
        completed_look_count=0,
        completed_chunk_count=0,
        last_chunk_receipt_hash=None,
        receipt_chain_hash=_EMPTY_RECEIPT_CHAIN_HASH,
        dynamic_oracle_receipt_hash=registration.dynamic_oracle_receipt_hash,
    )


def _validate_state_against_registration(
    registration: V1PairedBCEAcquisitionRegistration,
    state: V1PairedBCEAcquisitionState,
) -> None:
    if type(registration) is not V1PairedBCEAcquisitionRegistration:
        raise TypeError("paired-BCE transition requires the canonical registration")
    if type(state) is not V1PairedBCEAcquisitionState:
        raise TypeError("paired-BCE transition requires the canonical state")
    registration_hash = registration.assert_integrity()
    state.assert_integrity()
    if state.registration_hash != registration_hash:
        raise ValueError("paired-BCE state names another registration")
    if state.candidate_attempt_index != registration.candidate_attempt_index:
        raise ValueError("paired-BCE state names another candidate attempt")
    if state.dynamic_oracle_receipt_hash != registration.dynamic_oracle_receipt_hash:
        raise ValueError("paired-BCE state names a stale dynamic O frame")
    if any(
        count > population
        for count, population in zip(
            state.stratum_counts,
            registration.population_stratum_counts,
            strict=True,
        )
    ):
        raise ValueError("paired-BCE state exceeds a registered finite population")
    if state.completed_look_count > len(registration.registered_look_stratum_counts):
        raise ValueError("paired-BCE state completed an unregistered look")
    expected_completed = 0
    for row in registration.registered_look_stratum_counts:
        if all(
            count >= target
            for count, target in zip(state.stratum_counts, row, strict=True)
        ):
            expected_completed += 1
        else:
            break
    if state.completed_look_count != expected_completed:
        raise ValueError("paired-BCE state look cursor disagrees with its counts")
    if state.completed_look_count < len(registration.registered_look_stratum_counts):
        target = registration.registered_look_stratum_counts[
            state.completed_look_count
        ]
        if any(
            count > target_count
            for count, target_count in zip(state.stratum_counts, target, strict=True)
        ):
            raise ValueError("paired-BCE state crossed an uncommitted registered look")


def _context_slice_hash(
    *,
    registration_hash: str,
    stratum_id: str,
    prefix_start: int,
    context_ids: Sequence[str],
) -> str:
    return content_hash(
        {
            "schema": "modern_v1_paired_bce_context_slice_v1",
            "registration_hash": registration_hash,
            "stratum_id": stratum_id,
            "prefix_start": prefix_start,
            "context_ids": tuple(context_ids),
        }
    )


def _sealed_slice_registration_hash(
    *,
    sealed_order_hash: str,
    verifier_identity_hash: str,
    stratum_id: str,
    prefix_start: int,
    prefix_stop: int,
    context_slice_hash: str,
) -> str:
    return content_hash(
        {
            "schema": "modern_v1_paired_bce_sealed_order_slice_registration_v1",
            "sealed_order_hash": sealed_order_hash,
            "verifier_identity_hash": verifier_identity_hash,
            "stratum_id": stratum_id,
            "prefix_start": prefix_start,
            "prefix_stop": prefix_stop,
            "context_slice_hash": context_slice_hash,
        }
    )


def _serfling_width_proxy(
    stratum: V1PairedBCEStratumRegistration,
    count: int,
) -> float | None:
    span = stratum.bound_span
    if span is None:
        return None
    if count >= stratum.population_size:
        return 0.0
    if count == 0:
        return span
    correction = 1.0 - (count - 1.0) / stratum.population_size
    return span * math.sqrt(correction / count)


def _predictable_next_stratum_index(
    registration: V1PairedBCEAcquisitionRegistration,
    state: V1PairedBCEAcquisitionState,
    *,
    maximum_rows: int,
) -> int:
    if state.completed_look_count >= len(registration.registered_look_stratum_counts):
        raise RuntimeError("paired-BCE acquisition already reached joint census")
    target_counts = registration.registered_look_stratum_counts[
        state.completed_look_count
    ]
    candidates: list[tuple[tuple[float, float, int], int]] = []
    for index, (stratum, count, target_count) in enumerate(
        zip(registration.strata, state.stratum_counts, target_counts, strict=True)
    ):
        deficit = target_count - count
        if deficit <= 0:
            continue
        added = min(maximum_rows, deficit)
        before = _serfling_width_proxy(stratum, count)
        after = _serfling_width_proxy(stratum, count + added)
        if before is None or after is None:
            width_gain = 0.0
        else:
            width_gain = 0.5 * max(0.0, before - after)
        normalized_deficit = deficit / target_count
        # Final component makes target the deterministic exact tie-breaker.
        priority = (width_gain, normalized_deficit, 1 - index)
        candidates.append((priority, index))
    if not candidates:  # pragma: no cover - state validation excludes this.
        raise RuntimeError("paired-BCE look has no remaining acquisition deficit")
    return max(candidates)[1]


def next_paired_bce_chunk_coordinates(
    *,
    registration: V1PairedBCEAcquisitionRegistration,
    state: V1PairedBCEAcquisitionState,
    maximum_rows: int,
    betting_replay: V1PairedBCEBettingReplay | None = None,
) -> tuple[str, int, int]:
    """Return predictable coordinates so the caller can read the sealed slice.

    This descriptor contains no context identities and no outcomes.  The
    caller reads exactly ``[start:stop]`` from the named immutable stratum
    order, then passes that transient slice to
    :func:`make_next_paired_bce_chunk_request` for content binding.
    """

    _validate_state_against_registration(registration, state)
    maximum = _positive_int(maximum_rows, name="maximum chunk rows")
    decision = decide_paired_bce_acquisition(
        registration=registration,
        state=state,
        betting_replay=betting_replay,
    )
    if decision.status != "acquire_more":
        raise RuntimeError("terminal paired-BCE decision has no next chunk")
    index = _predictable_next_stratum_index(
        registration, state, maximum_rows=maximum
    )
    target_counts = registration.registered_look_stratum_counts[
        state.completed_look_count
    ]
    start = state.stratum_counts[index]
    stop = min(target_counts[index], start + maximum)
    return PAIRED_BCE_STRATA[index], start, stop




@dataclass(frozen=True, slots=True)
class V1PairedBCEVerifiedSealedOrder:
    """Cached verification of one sealed oracle order for a registration stratum.

    This handle does not create sampling or population authority. It only binds
    an immutable order tuple to an already-issued paired-BCE registration so
    repeated chunk requests do not have to replay full-order hashing.
    """

    registration_hash: str
    stratum_id: str
    stratum_index: int
    population_size: int
    sealed_context_order: tuple[str, ...]
    sealed_order_hash: str
    population_identity_hash: str
    sealed_order_slice_verifier_identity_hash: str
    schema: str = "modern_v1_paired_bce_verified_sealed_order_v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "registration_hash", _identity(self.registration_hash, name="registration hash"))
        stratum = _identity(self.stratum_id, name="stratum id")
        if stratum not in PAIRED_BCE_STRATA:
            raise ValueError("verified sealed order stratum must be target or complement")
        index = _nonnegative_int(self.stratum_index, name="stratum index")
        if index >= len(PAIRED_BCE_STRATA) or PAIRED_BCE_STRATA[index] != stratum:
            raise ValueError("verified sealed order stratum index is inconsistent")
        population = _positive_int(self.population_size, name="sealed order population size")
        order = tuple(_identity(value, name="sealed context id") for value in self.sealed_context_order)
        if len(order) != population or len(order) != len(set(order)):
            raise ValueError("verified sealed order must enumerate the stratum exactly once")
        sealed_hash = _identity(self.sealed_order_hash, name="sealed order hash")
        if paired_bce_sealed_order_hash(stratum_id=stratum, context_ids_in_sealed_order=order) != sealed_hash:
            raise ValueError("verified sealed order hash does not replay")
        population_hash = _identity(self.population_identity_hash, name="population identity hash")
        if _context_set_hash(role=f"oracle_{stratum}", context_ids=order) != population_hash:
            raise ValueError("verified sealed order population identity does not replay")
        object.__setattr__(self, "stratum_id", stratum)
        object.__setattr__(self, "stratum_index", index)
        object.__setattr__(self, "population_size", population)
        object.__setattr__(self, "sealed_context_order", order)
        object.__setattr__(self, "sealed_order_hash", sealed_hash)
        object.__setattr__(self, "population_identity_hash", population_hash)
        object.__setattr__(self, "sealed_order_slice_verifier_identity_hash", _identity(self.sealed_order_slice_verifier_identity_hash, name="sealed order slice verifier identity hash"))
        if self.schema != "modern_v1_paired_bce_verified_sealed_order_v1":
            raise ValueError("verified sealed order schema is invalid")


def verify_paired_bce_sealed_order(
    *,
    registration: V1PairedBCEAcquisitionRegistration,
    stratum_id: str,
    sealed_context_order: Sequence[str],
) -> V1PairedBCEVerifiedSealedOrder:
    """Verify one full sealed order once and return an immutable handle."""

    if type(registration) is not V1PairedBCEAcquisitionRegistration:
        raise TypeError("sealed order verification requires the paired-BCE acquisition registration")
    registration.assert_integrity()
    stratum = _identity(stratum_id, name="stratum id")
    if stratum not in PAIRED_BCE_STRATA:
        raise ValueError("paired-BCE stratum must be target or complement")
    index = PAIRED_BCE_STRATA.index(stratum)
    registered = registration.strata[index]
    return V1PairedBCEVerifiedSealedOrder(
        registration_hash=registration.registration_hash,
        stratum_id=stratum,
        stratum_index=index,
        population_size=registered.population_size,
        sealed_context_order=tuple(sealed_context_order),
        sealed_order_hash=registered.sealed_order_hash,
        population_identity_hash=registered.population_identity_hash,
        sealed_order_slice_verifier_identity_hash=registration.sealed_order_slice_verifier_identity_hash,
    )


def _order_from_legacy_or_handle(
    *,
    registration: V1PairedBCEAcquisitionRegistration,
    stratum_index: int,
    sealed_context_order: Sequence[str] | None,
    verified_sealed_order: V1PairedBCEVerifiedSealedOrder | None,
) -> tuple[str, ...]:
    stratum = registration.strata[stratum_index]
    if verified_sealed_order is not None and sealed_context_order is not None:
        raise ValueError("provide either sealed_context_order or verified_sealed_order, not both")
    if verified_sealed_order is not None:
        if type(verified_sealed_order) is not V1PairedBCEVerifiedSealedOrder:
            raise TypeError("verified_sealed_order must be a V1PairedBCEVerifiedSealedOrder")
        if verified_sealed_order.registration_hash != registration.registration_hash:
            raise ValueError("verified sealed order belongs to another registration")
        if verified_sealed_order.stratum_id != stratum.stratum_id or verified_sealed_order.stratum_index != stratum_index:
            raise ValueError("verified sealed order belongs to another stratum")
        if verified_sealed_order.population_size != stratum.population_size:
            raise ValueError("verified sealed order population size differs from registration")
        if verified_sealed_order.population_identity_hash != stratum.population_identity_hash:
            raise ValueError("verified sealed order population identity differs from registration")
        if verified_sealed_order.sealed_order_hash != stratum.sealed_order_hash:
            raise ValueError("verified sealed order hash differs from registration")
        if verified_sealed_order.sealed_order_slice_verifier_identity_hash != registration.sealed_order_slice_verifier_identity_hash:
            raise ValueError("verified sealed order slice verifier differs from registration")
        return verified_sealed_order.sealed_context_order
    if sealed_context_order is None:
        raise ValueError("sealed context order or verified sealed order is required")
    full_order = tuple(
        _identity(value, name="sealed-order context id")
        for value in sealed_context_order
    )
    if len(full_order) != stratum.population_size or len(full_order) != len(set(full_order)):
        raise ValueError("sealed context order must enumerate the stratum exactly once")
    if paired_bce_sealed_order_hash(
        stratum_id=stratum.stratum_id,
        context_ids_in_sealed_order=full_order,
    ) != stratum.sealed_order_hash:
        raise ValueError("sealed context order differs from its frozen identity")
    if _context_set_hash(
        role=f"oracle_{stratum.stratum_id}", context_ids=full_order
    ) != stratum.population_identity_hash:
        raise ValueError("sealed context order is not a permutation of its oracle fold")
    return full_order


def validate_paired_bce_chunk_request_preoutcome(
    *,
    registration: V1PairedBCEAcquisitionRegistration,
    state: V1PairedBCEAcquisitionState,
    request: V1PairedBCEChunkRequest,
    context_ids_in_registered_slice: Sequence[str] | None = None,
    sealed_context_order: Sequence[str] | None = None,
    verified_sealed_order: V1PairedBCEVerifiedSealedOrder | None = None,
) -> tuple[str, ...]:
    """Validate request/state/slice before any paired-loss rows may be opened."""

    _validate_state_against_registration(registration, state)
    if type(request) is not V1PairedBCEChunkRequest:
        raise TypeError("paired-BCE preoutcome validation requires the canonical request")
    request.assert_integrity()
    if request.registration_hash != registration.registration_hash:
        raise ValueError("paired-BCE request names another registration")
    if request.prior_state_hash != state.state_hash:
        raise ValueError("paired-BCE request names a stale state")
    if request.candidate_attempt_alpha_coordinate_hash != registration.candidate_attempt_alpha_coordinate_hash:
        raise ValueError("paired-BCE request names another candidate-attempt coordinate")
    if request.candidate_attempt_index != registration.candidate_attempt_index:
        raise ValueError("paired-BCE request names another candidate attempt")
    if request.dynamic_oracle_receipt_hash != registration.dynamic_oracle_receipt_hash:
        raise ValueError("paired-BCE request names a stale dynamic O frame")
    if request.request_index != state.chunk_count:
        raise ValueError("paired-BCE request append index is stale")
    if request.stratum_id not in PAIRED_BCE_STRATA:
        raise ValueError("paired-BCE request stratum is invalid")
    index = PAIRED_BCE_STRATA.index(request.stratum_id)
    if request.prefix_start != state.stratum_counts[index]:
        raise ValueError("paired-BCE request overlaps or skips the sealed prefix")
    if state.completed_look_count >= len(registration.registered_look_stratum_counts):
        raise ValueError("terminal paired-BCE acquisition state has no pre-outcome request")
    target_counts = registration.registered_look_stratum_counts[state.completed_look_count]
    if request.target_registered_look_index != state.completed_look_count or request.target_registered_look_stratum_counts != target_counts:
        raise ValueError("paired-BCE request names another registered look")
    if request.prefix_stop > target_counts[index] or request.prefix_stop <= request.prefix_start:
        raise ValueError("paired-BCE request crosses a registered look boundary")
    order = None
    if sealed_context_order is not None or verified_sealed_order is not None:
        order = _order_from_legacy_or_handle(
            registration=registration,
            stratum_index=index,
            sealed_context_order=sealed_context_order,
            verified_sealed_order=verified_sealed_order,
        )
    if request.sealed_order_hash != registration.strata[index].sealed_order_hash:
        raise ValueError("paired-BCE request names another sealed stratum order")
    if context_ids_in_registered_slice is None:
        if order is None:
            raise ValueError("context slice or sealed order handle is required")
        contexts = order[request.prefix_start:request.prefix_stop]
    else:
        try:
            supplied_len = len(context_ids_in_registered_slice)
        except TypeError as exc:
            raise TypeError("context_ids_in_registered_slice must be a sized sequence") from exc
        if supplied_len != request.row_count:
            raise ValueError("paired-BCE context payload differs from request row count")
        contexts = tuple(_identity(value, name="observed context id") for value in context_ids_in_registered_slice)
    if len(contexts) != request.row_count:
        raise ValueError("paired-BCE context payload differs from request row count")
    if len(contexts) != len(set(contexts)):
        raise ValueError("paired-BCE physical chunk repeats a context identity")
    if order is not None and contexts != order[request.prefix_start:request.prefix_stop]:
        raise ValueError("paired-BCE context payload is not the next sealed-order prefix")
    observed_slice_hash = _context_slice_hash(
        registration_hash=registration.registration_hash,
        stratum_id=request.stratum_id,
        prefix_start=request.prefix_start,
        context_ids=contexts,
    )
    if observed_slice_hash != request.context_slice_hash:
        raise ValueError("paired-BCE context payload differs from pre-outcome request")
    expected_sealed_slice_hash = _sealed_slice_registration_hash(
        sealed_order_hash=registration.strata[index].sealed_order_hash,
        verifier_identity_hash=registration.sealed_order_slice_verifier_identity_hash,
        stratum_id=request.stratum_id,
        prefix_start=request.prefix_start,
        prefix_stop=request.prefix_stop,
        context_slice_hash=observed_slice_hash,
    )
    if expected_sealed_slice_hash != request.sealed_slice_registration_hash:
        raise ValueError("paired-BCE request lost its sealed-order slice registration")
    return contexts


@dataclass(frozen=True, slots=True)
class V1PairedBCEChunkRequest:
    """Outcome-independent descriptor frozen before one physical row chunk."""

    registration_hash: str
    prior_state_hash: str
    candidate_attempt_alpha_coordinate_hash: str
    candidate_attempt_index: int
    request_index: int
    stratum_id: str
    sealed_order_hash: str
    prefix_start: int
    prefix_stop: int
    context_slice_hash: str
    sealed_slice_registration_hash: str
    target_registered_look_index: int
    target_registered_look_stratum_counts: tuple[int, int]
    row_provider_identity_hash: str
    paired_loss_evaluator_identity_hash: str
    dynamic_oracle_receipt_hash: str | None = None
    pending_before_outcomes: bool = True
    outcome_fields_absent: bool = True
    schema: str = CHUNK_REQUEST_SCHEMA
    _request_hash_cache: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "registration_hash",
            "prior_state_hash",
            "candidate_attempt_alpha_coordinate_hash",
            "sealed_order_hash",
            "context_slice_hash",
            "sealed_slice_registration_hash",
            "row_provider_identity_hash",
            "paired_loss_evaluator_identity_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        dynamic_hash = self.dynamic_oracle_receipt_hash
        if dynamic_hash is not None:
            dynamic_hash = _identity(dynamic_hash, name="dynamic oracle receipt hash")
        object.__setattr__(
            self,
            "candidate_attempt_index",
            _nonnegative_int(
                self.candidate_attempt_index, name="candidate attempt index"
            ),
        )
        object.__setattr__(
            self,
            "request_index",
            _nonnegative_int(self.request_index, name="request index"),
        )
        stratum_id = _identity(self.stratum_id, name="stratum id")
        if stratum_id not in PAIRED_BCE_STRATA:
            raise ValueError("paired-BCE chunk names an unknown stratum")
        start = _nonnegative_int(self.prefix_start, name="prefix start")
        stop = _positive_int(self.prefix_stop, name="prefix stop")
        if stop <= start:
            raise ValueError("paired-BCE chunk request must acquire at least one row")
        look_index = _nonnegative_int(
            self.target_registered_look_index,
            name="target registered look index",
        )
        target_counts = _pair_of_nonnegative_ints(
            self.target_registered_look_stratum_counts,
            name="target registered look counts",
        )
        if _strict_bool(
            self.pending_before_outcomes, name="pending before outcomes"
        ) is not True or _strict_bool(
            self.outcome_fields_absent, name="outcome fields absent"
        ) is not True:
            raise ValueError("paired-BCE request must be frozen without outcomes")
        if self.schema != CHUNK_REQUEST_SCHEMA:
            raise ValueError("paired-BCE chunk request schema is invalid")
        object.__setattr__(self, "stratum_id", stratum_id)
        object.__setattr__(self, "prefix_start", start)
        object.__setattr__(self, "prefix_stop", stop)
        object.__setattr__(self, "target_registered_look_index", look_index)
        object.__setattr__(self, "target_registered_look_stratum_counts", target_counts)
        object.__setattr__(self, "dynamic_oracle_receipt_hash", dynamic_hash)

    @property
    def row_count(self) -> int:
        return self.prefix_stop - self.prefix_start

    @property
    def request_hash(self) -> str:
        return self.assert_integrity()

    def _hash_payload(self) -> dict[str, Any]:
        return _constructor_payload(self)

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        return _assert_hash_integrity(
            self,
            payload=self._hash_payload(),
            cache_name="_request_hash_cache",
            label="paired-BCE chunk request",
            expected_hash=expected_hash,
        )

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return _rebuild_hashed, (
            type(self),
            _constructor_payload(self),
            self.assert_integrity(),
        )


def make_next_paired_bce_chunk_request(
    *,
    registration: V1PairedBCEAcquisitionRegistration,
    state: V1PairedBCEAcquisitionState,
    maximum_rows: int,
    sealed_context_order: Sequence[str] | None = None,
    context_ids_in_registered_slice: Sequence[str] | None = None,
    row_provider_identity_hash: str,
    paired_loss_evaluator_identity_hash: str,
    betting_replay: V1PairedBCEBettingReplay | None = None,
    verified_sealed_order: V1PairedBCEVerifiedSealedOrder | None = None,
) -> V1PairedBCEChunkRequest:
    """Freeze the next deterministic, outcome-independent physical chunk request."""

    _validate_state_against_registration(registration, state)
    maximum = _positive_int(maximum_rows, name="maximum chunk rows")
    stratum_id, start, stop = next_paired_bce_chunk_coordinates(
        registration=registration,
        state=state,
        maximum_rows=maximum,
        betting_replay=betting_replay,
    )
    stratum_index = PAIRED_BCE_STRATA.index(stratum_id)
    stratum = registration.strata[stratum_index]
    if state.completed_look_count >= len(registration.registered_look_stratum_counts):
        raise ValueError("terminal paired-BCE acquisition state has no next request")
    target_counts = registration.registered_look_stratum_counts[
        state.completed_look_count
    ]
    full_order = _order_from_legacy_or_handle(
        registration=registration,
        stratum_index=stratum_index,
        sealed_context_order=sealed_context_order,
        verified_sealed_order=verified_sealed_order,
    )
    if context_ids_in_registered_slice is None:
        contexts = full_order[start:stop]
    else:
        try:
            supplied_len = len(context_ids_in_registered_slice)
        except TypeError as exc:
            raise TypeError("context_ids_in_registered_slice must be a sized sequence") from exc
        if supplied_len != stop - start:
            raise ValueError("registered context slice length differs from requested prefix")
        contexts = tuple(
            _identity(value, name="registered slice context id")
            for value in context_ids_in_registered_slice
        )
    if len(contexts) != stop - start:
        raise ValueError("registered context slice length differs from requested prefix")
    if len(contexts) != len(set(contexts)):
        raise ValueError("registered context slice repeats a context identity")
    if contexts != full_order[start:stop]:
        raise ValueError("registered context slice is not the next sealed-order prefix")
    registration_hash = registration.assert_integrity()
    slice_hash = _context_slice_hash(
        registration_hash=registration_hash,
        stratum_id=stratum.stratum_id,
        prefix_start=start,
        context_ids=contexts,
    )
    sealed_slice_hash = _sealed_slice_registration_hash(
        sealed_order_hash=stratum.sealed_order_hash,
        verifier_identity_hash=registration.sealed_order_slice_verifier_identity_hash,
        stratum_id=stratum.stratum_id,
        prefix_start=start,
        prefix_stop=stop,
        context_slice_hash=slice_hash,
    )
    return V1PairedBCEChunkRequest(
        registration_hash=registration_hash,
        prior_state_hash=state.assert_integrity(),
        candidate_attempt_alpha_coordinate_hash=(
            registration.candidate_attempt_alpha_coordinate_hash
        ),
        candidate_attempt_index=registration.candidate_attempt_index,
        request_index=state.chunk_count,
        stratum_id=stratum.stratum_id,
        sealed_order_hash=stratum.sealed_order_hash,
        prefix_start=start,
        prefix_stop=stop,
        context_slice_hash=slice_hash,
        sealed_slice_registration_hash=sealed_slice_hash,
        target_registered_look_index=state.completed_look_count,
        target_registered_look_stratum_counts=target_counts,
        row_provider_identity_hash=row_provider_identity_hash,
        paired_loss_evaluator_identity_hash=paired_loss_evaluator_identity_hash,
        dynamic_oracle_receipt_hash=registration.dynamic_oracle_receipt_hash,
    )


def _paired_row_payload_hash(
    *,
    request_hash: str,
    context_ids: Sequence[str],
    paired_loss_rows: Sequence[float],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"modern-v1-paired-bce-row-payload-v1\0")
    digest.update(request_hash.encode("utf-8"))
    for context_id, value in zip(context_ids, paired_loss_rows, strict=True):
        encoded = context_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(struct.pack(">d", value))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class V1PairedBCEChunkReceipt:
    """Compact append-only receipt for one fully evaluated physical chunk."""

    request_hash: str
    registration_hash: str
    prior_state_hash: str
    candidate_attempt_alpha_coordinate_hash: str
    candidate_attempt_index: int
    request_index: int
    stratum_id: str
    prefix_start: int
    prefix_stop: int
    context_slice_hash: str
    sealed_slice_registration_hash: str
    paired_row_payload_hash: str
    exact_scaled_loss_sum: int
    observed_minimum: float
    observed_maximum: float
    paired_loss_evaluator_receipt_hash: str
    dynamic_oracle_receipt_hash: str | None = None
    all_rows_finite: bool = True
    finite_bound_checked_when_available: bool = True
    full_chunk_completed: bool = True
    schema: str = CHUNK_RECEIPT_SCHEMA
    _receipt_hash_cache: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "request_hash",
            "registration_hash",
            "prior_state_hash",
            "candidate_attempt_alpha_coordinate_hash",
            "context_slice_hash",
            "sealed_slice_registration_hash",
            "paired_row_payload_hash",
            "paired_loss_evaluator_receipt_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        dynamic_hash = self.dynamic_oracle_receipt_hash
        if dynamic_hash is not None:
            dynamic_hash = _identity(dynamic_hash, name="dynamic oracle receipt hash")
        object.__setattr__(
            self,
            "candidate_attempt_index",
            _nonnegative_int(
                self.candidate_attempt_index, name="candidate attempt index"
            ),
        )
        object.__setattr__(
            self,
            "request_index",
            _nonnegative_int(self.request_index, name="request index"),
        )
        stratum_id = _identity(self.stratum_id, name="stratum id")
        if stratum_id not in PAIRED_BCE_STRATA:
            raise ValueError("paired-BCE receipt names an unknown stratum")
        start = _nonnegative_int(self.prefix_start, name="prefix start")
        stop = _positive_int(self.prefix_stop, name="prefix stop")
        if stop <= start:
            raise ValueError("paired-BCE receipt must contain at least one row")
        if isinstance(self.exact_scaled_loss_sum, bool) or not isinstance(
            self.exact_scaled_loss_sum, int
        ):
            raise ValueError("paired-BCE exact scaled sum must be an integer")
        minimum = _finite(self.observed_minimum, name="observed minimum")
        maximum = _finite(self.observed_maximum, name="observed maximum")
        if minimum > maximum:
            raise ValueError("paired-BCE receipt extrema are inverted")
        for name in (
            "all_rows_finite",
            "finite_bound_checked_when_available",
            "full_chunk_completed",
        ):
            if _strict_bool(getattr(self, name), name=name) is not True:
                raise ValueError(f"paired-BCE receipt requires {name}=True")
        if self.schema != CHUNK_RECEIPT_SCHEMA:
            raise ValueError("paired-BCE chunk receipt schema is invalid")
        object.__setattr__(self, "stratum_id", stratum_id)
        object.__setattr__(self, "prefix_start", start)
        object.__setattr__(self, "prefix_stop", stop)
        object.__setattr__(self, "observed_minimum", minimum)
        object.__setattr__(self, "observed_maximum", maximum)
        object.__setattr__(self, "dynamic_oracle_receipt_hash", dynamic_hash)

    @property
    def row_count(self) -> int:
        return self.prefix_stop - self.prefix_start

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def _hash_payload(self) -> dict[str, Any]:
        return _constructor_payload(self)

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        return _assert_hash_integrity(
            self,
            payload=self._hash_payload(),
            cache_name="_receipt_hash_cache",
            label="paired-BCE chunk receipt",
            expected_hash=expected_hash,
        )

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return _rebuild_hashed, (
            type(self),
            _constructor_payload(self),
            self.assert_integrity(),
        )


def reduce_paired_bce_chunk(
    *,
    registration: V1PairedBCEAcquisitionRegistration,
    state: V1PairedBCEAcquisitionState,
    request: V1PairedBCEChunkRequest,
    context_ids_in_registered_slice: Sequence[str],
    paired_loss_rows: Sequence[float],
    paired_loss_evaluator_receipt_hash: str,
) -> V1PairedBCEChunkReceipt:
    """Validate and reduce transient row payloads into one compact receipt."""

    contexts = validate_paired_bce_chunk_request_preoutcome(
        registration=registration,
        state=state,
        request=request,
        context_ids_in_registered_slice=context_ids_in_registered_slice,
    )
    request_hash = request.assert_integrity()
    registration_hash = registration.assert_integrity()
    state_hash = state.assert_integrity()
    stratum_index = PAIRED_BCE_STRATA.index(request.stratum_id)
    stratum = registration.strata[stratum_index]
    observed_slice_hash = request.context_slice_hash
    expected_sealed_slice_hash = request.sealed_slice_registration_hash

    rows = tuple(_finite(value, name="paired-loss row") for value in paired_loss_rows)
    if len(rows) != request.row_count:
        raise ValueError("paired-loss row payload differs from request row count")
    lower, upper = stratum.paired_loss_lower_bound, stratum.paired_loss_upper_bound
    if lower is not None and upper is not None and any(
        value < lower or value > upper for value in rows
    ):
        raise ValueError("paired-loss row left its frozen pointwise bound")
    scaled_sum = sum(_binary64_scaled_integer(value) for value in rows)
    return V1PairedBCEChunkReceipt(
        request_hash=request_hash,
        registration_hash=registration_hash,
        prior_state_hash=state_hash,
        candidate_attempt_alpha_coordinate_hash=(
            registration.candidate_attempt_alpha_coordinate_hash
        ),
        candidate_attempt_index=registration.candidate_attempt_index,
        request_index=request.request_index,
        stratum_id=request.stratum_id,
        prefix_start=request.prefix_start,
        prefix_stop=request.prefix_stop,
        context_slice_hash=observed_slice_hash,
        sealed_slice_registration_hash=expected_sealed_slice_hash,
        paired_row_payload_hash=_paired_row_payload_hash(
            request_hash=request_hash,
            context_ids=contexts,
            paired_loss_rows=rows,
        ),
        exact_scaled_loss_sum=scaled_sum,
        observed_minimum=min(rows),
        observed_maximum=max(rows),
        paired_loss_evaluator_receipt_hash=paired_loss_evaluator_receipt_hash,
        dynamic_oracle_receipt_hash=registration.dynamic_oracle_receipt_hash,
    )


@dataclass(frozen=True, slots=True)
class V1PairedBCEBettingReplay:
    """Transient exact replay input for the canonical betting confidence sequence.

    The full sealed orders and observed row prefixes live in the external row
    store.  The production wrapper rehydrates and authenticates them after
    each committed physical chunk; the confidence sequence uses them at
    registered partial looks. This object is intentionally absent from
    :class:`V1PairedBCEAcquisitionState`;
    deterministic resume reconstructs it from the append-only receipts and
    their content-addressed row payloads.
    """

    sealed_context_orders: tuple[tuple[str, ...], tuple[str, ...]]
    observed_context_ids_by_stratum: tuple[tuple[str, ...], tuple[str, ...]]
    observed_paired_loss_rows_by_stratum: tuple[
        tuple[float, ...], tuple[float, ...]
    ]
    chunk_receipts: tuple[V1PairedBCEChunkReceipt, ...]
    dynamic_oracle_receipt_hash: str | None = None
    schema: str = "modern_v1_paired_bce_transient_betting_replay_v1"

    def __post_init__(self) -> None:
        orders = tuple(
            tuple(_identity(value, name="sealed-order context id") for value in row)
            for row in self.sealed_context_orders
        )
        contexts = tuple(
            tuple(_identity(value, name="observed context id") for value in row)
            for row in self.observed_context_ids_by_stratum
        )
        rows = tuple(
            tuple(_finite(value, name="replayed paired-loss row") for value in row)
            for row in self.observed_paired_loss_rows_by_stratum
        )
        if len(orders) != 2 or len(contexts) != 2 or len(rows) != 2:
            raise ValueError("paired-BCE betting replay requires exactly two strata")
        if any(not row or len(row) != len(set(row)) for row in orders):
            raise ValueError("replayed sealed orders must enumerate unique populations")
        if any(
            len(context_row) != len(value_row)
            for context_row, value_row in zip(contexts, rows, strict=True)
        ):
            raise ValueError("replayed paired-loss rows differ from context prefixes")
        receipts = tuple(self.chunk_receipts)
        if any(type(item) is not V1PairedBCEChunkReceipt for item in receipts):
            raise TypeError("betting replay requires canonical compact chunk receipts")
        if self.schema != "modern_v1_paired_bce_transient_betting_replay_v1":
            raise ValueError("paired-BCE betting replay schema is invalid")
        dynamic_hash = self.dynamic_oracle_receipt_hash
        if dynamic_hash is not None:
            dynamic_hash = _identity(dynamic_hash, name="dynamic oracle receipt hash")
        object.__setattr__(self, "sealed_context_orders", orders)
        object.__setattr__(self, "observed_context_ids_by_stratum", contexts)
        object.__setattr__(self, "observed_paired_loss_rows_by_stratum", rows)
        object.__setattr__(self, "chunk_receipts", receipts)
        object.__setattr__(self, "dynamic_oracle_receipt_hash", dynamic_hash)

    def assert_matches(
        self,
        *,
        registration: V1PairedBCEAcquisitionRegistration,
        state: V1PairedBCEAcquisitionState,
    ) -> None:
        """Replay every compact receipt and the exact cumulative statistics."""

        _validate_state_against_registration(registration, state)
        if self.dynamic_oracle_receipt_hash != registration.dynamic_oracle_receipt_hash:
            raise ValueError("betting replay names a stale dynamic O frame")
        if tuple(len(row) for row in self.sealed_context_orders) != (
            registration.population_stratum_counts
        ):
            raise ValueError("betting replay sealed axes differ from populations")
        for index, (stratum, order, contexts, rows) in enumerate(
            zip(
                registration.strata,
                self.sealed_context_orders,
                self.observed_context_ids_by_stratum,
                self.observed_paired_loss_rows_by_stratum,
                strict=True,
            )
        ):
            if paired_bce_sealed_order_hash(
                stratum_id=stratum.stratum_id,
                context_ids_in_sealed_order=order,
            ) != stratum.sealed_order_hash:
                raise ValueError("betting replay names another sealed context order")
            if _context_set_hash(
                role=f"oracle_{stratum.stratum_id}", context_ids=order
            ) != stratum.population_identity_hash:
                raise ValueError(
                    "betting replay order is not a permutation of its oracle fold"
                )
            if contexts != order[: state.stratum_counts[index]]:
                raise ValueError("betting replay is not the live sealed-order prefix")
            lower, upper = (
                stratum.paired_loss_lower_bound,
                stratum.paired_loss_upper_bound,
            )
            if lower is not None and upper is not None and any(
                value < lower or value > upper for value in rows
            ):
                raise ValueError("betting replay row left its frozen pointwise bound")
            if sum(_binary64_scaled_integer(value) for value in rows) != (
                state.exact_scaled_loss_sums[index]
            ):
                raise ValueError("betting replay exact sum differs from compact state")
            if rows:
                if min(rows) != state.observed_minima[index] or max(rows) != (
                    state.observed_maxima[index]
                ):
                    raise ValueError("betting replay extrema differ from compact state")

        receipt_hashes = tuple(item.assert_integrity() for item in self.chunk_receipts)
        if len(receipt_hashes) != state.chunk_count or (
            (receipt_hashes[-1] if receipt_hashes else None)
            != state.last_chunk_receipt_hash
        ):
            raise ValueError("betting replay receipt ledger differs from compact state")
        replay_chain_hash = _EMPTY_RECEIPT_CHAIN_HASH
        for request_index, receipt_hash in enumerate(receipt_hashes):
            replay_chain_hash = content_hash(
                {
                    "schema": "modern_v1_paired_bce_receipt_chain_link_v1",
                    "prior_receipt_chain_hash": replay_chain_hash,
                    "request_index": request_index,
                    "receipt_hash": receipt_hash,
                }
            )
        if replay_chain_hash != state.receipt_chain_hash:
            raise ValueError("betting replay receipt chain differs from compact state")
        cursors = [0, 0]
        for request_index, receipt in enumerate(self.chunk_receipts):
            if receipt.request_index != request_index:
                raise ValueError("betting replay receipt order is not append-only")
            if receipt.registration_hash != registration.registration_hash or (
                receipt.candidate_attempt_alpha_coordinate_hash
                != registration.candidate_attempt_alpha_coordinate_hash
            ) or (
                receipt.dynamic_oracle_receipt_hash
                != registration.dynamic_oracle_receipt_hash
            ):
                raise ValueError("betting replay receipt names another attempt")
            index = PAIRED_BCE_STRATA.index(receipt.stratum_id)
            if receipt.prefix_start != cursors[index]:
                raise ValueError("betting replay receipt prefixes overlap or skip")
            contexts = self.observed_context_ids_by_stratum[index][
                receipt.prefix_start : receipt.prefix_stop
            ]
            rows = self.observed_paired_loss_rows_by_stratum[index][
                receipt.prefix_start : receipt.prefix_stop
            ]
            if len(contexts) != receipt.row_count or (
                _context_slice_hash(
                    registration_hash=registration.registration_hash,
                    stratum_id=receipt.stratum_id,
                    prefix_start=receipt.prefix_start,
                    context_ids=contexts,
                )
                != receipt.context_slice_hash
            ):
                raise ValueError("betting replay context slice differs from receipt")
            if _paired_row_payload_hash(
                request_hash=receipt.request_hash,
                context_ids=contexts,
                paired_loss_rows=rows,
            ) != receipt.paired_row_payload_hash:
                raise ValueError("betting replay row payload differs from receipt")
            cursors[index] = receipt.prefix_stop
        if tuple(cursors) != state.stratum_counts:
            raise ValueError("betting replay receipts do not cover the live prefixes")


def _canonical_betting_intervals(
    *,
    registration: V1PairedBCEAcquisitionRegistration,
    state: V1PairedBCEAcquisitionState,
    replay: V1PairedBCEBettingReplay,
) -> tuple[
    tuple[tuple[float, float], tuple[float, float]] | None,
    Literal["empty", "numerical"] | None,
]:
    replay.assert_matches(registration=registration, state=state)
    stratum_alpha = registration.attempt_alpha_allocation / 2.0
    intervals: list[tuple[float, float]] = []
    for stratum, order, contexts, rows in zip(
        registration.strata,
        replay.sealed_context_orders,
        replay.observed_context_ids_by_stratum,
        replay.observed_paired_loss_rows_by_stratum,
        strict=True,
    ):
        assert stratum.paired_loss_lower_bound is not None
        assert stratum.paired_loss_upper_bound is not None
        try:
            sequence = BoundedWoRBettingCS(
                stratum_id=stratum.stratum_id,
                population_size=stratum.population_size,
                alpha=stratum_alpha,
                lower_bound=stratum.paired_loss_lower_bound,
                upper_bound=stratum.paired_loss_upper_bound,
                sealed_context_order=order,
                sealed_order_hash=stratum.sealed_order_hash,
                claim_bearing=True,
            )
            for context_id, value in zip(contexts, rows, strict=True):
                sequence.record(value, context_id=context_id)
            # Float inversion proposes endpoints; directed replay must prove
            # exclusion before narrowing exact finite-population feasibility.
            # Proposal overflow widens safely. Any remaining arithmetic failure
            # has no partial authority and must continue toward exact census.
            # Replay/type/integrity errors are deliberately not hidden here.
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                snapshot = sequence.certified_interval()
        except (ArithmeticError, FloatingPointError, RuntimeWarning):
            return None, "numerical"
        except ValueError as error:
            # The generic CS reports a nonfinite candidate created by its own
            # raw-scale feasible-mean arithmetic as this exact ValueError.
            # All other ValueErrors retain their programming/integrity force.
            if str(error) != "candidate mean must be finite":
                raise
            return None, "numerical"
        if snapshot.empty or not snapshot.claim_bearing:
            return None, "empty"
        if (
            not math.isfinite(snapshot.lower)
            or not math.isfinite(snapshot.upper)
            or snapshot.lower > snapshot.upper
        ):
            return None, "numerical"
        intervals.append((snapshot.lower, snapshot.upper))
    return (intervals[0], intervals[1]), None


def commit_paired_bce_chunk(
    *,
    registration: V1PairedBCEAcquisitionRegistration,
    state: V1PairedBCEAcquisitionState,
    request: V1PairedBCEChunkRequest,
    receipt: V1PairedBCEChunkReceipt,
) -> V1PairedBCEAcquisitionState:
    """Atomically append one verified compact receipt to cumulative state."""

    _validate_state_against_registration(registration, state)
    if type(request) is not V1PairedBCEChunkRequest:
        raise TypeError("paired-BCE commit requires the canonical request")
    if type(receipt) is not V1PairedBCEChunkReceipt:
        raise TypeError("paired-BCE commit requires the canonical receipt")
    request_hash = request.assert_integrity()
    receipt_hash = receipt.assert_integrity()
    if receipt.request_hash != request_hash:
        raise ValueError("paired-BCE receipt names another request")
    if receipt.registration_hash != registration.registration_hash:
        raise ValueError("paired-BCE receipt names another registration")
    if (
        request.dynamic_oracle_receipt_hash
        != registration.dynamic_oracle_receipt_hash
        or receipt.dynamic_oracle_receipt_hash
        != registration.dynamic_oracle_receipt_hash
    ):
        raise ValueError("paired-BCE request/receipt names a stale dynamic O frame")
    if receipt.prior_state_hash != state.state_hash:
        raise ValueError("paired-BCE receipt names a stale state")
    equality_fields = (
        "candidate_attempt_alpha_coordinate_hash",
        "candidate_attempt_index",
        "request_index",
        "stratum_id",
        "prefix_start",
        "prefix_stop",
        "context_slice_hash",
        "sealed_slice_registration_hash",
    )
    for name in equality_fields:
        if getattr(receipt, name) != getattr(request, name):
            raise ValueError(f"paired-BCE request and receipt disagree on {name}")
    if request.request_index != state.chunk_count:
        raise ValueError("paired-BCE receipt append index is stale")
    index = PAIRED_BCE_STRATA.index(request.stratum_id)
    if request.prefix_start != state.stratum_counts[index]:
        raise ValueError("paired-BCE receipt overlaps or skips the sealed prefix")
    target_counts = registration.registered_look_stratum_counts[
        state.completed_look_count
    ]
    if request.target_registered_look_index != state.completed_look_count or (
        request.target_registered_look_stratum_counts != target_counts
    ):
        raise ValueError("paired-BCE receipt names another registered look")
    if request.prefix_stop > target_counts[index]:
        raise ValueError("paired-BCE receipt crosses a registered look boundary")

    counts = list(state.stratum_counts)
    sums = list(state.exact_scaled_loss_sums)
    minima = list(state.observed_minima)
    maxima = list(state.observed_maxima)
    counts[index] = request.prefix_stop
    sums[index] += receipt.exact_scaled_loss_sum
    minima[index] = (
        receipt.observed_minimum
        if minima[index] is None
        else min(minima[index], receipt.observed_minimum)
    )
    maxima[index] = (
        receipt.observed_maximum
        if maxima[index] is None
        else max(maxima[index], receipt.observed_maximum)
    )
    completed = state.completed_look_count
    if tuple(counts) == target_counts:
        completed += 1
    chain_hash = content_hash(
        {
            "schema": "modern_v1_paired_bce_receipt_chain_link_v1",
            "prior_receipt_chain_hash": state.receipt_chain_hash,
            "request_index": request.request_index,
            "receipt_hash": receipt_hash,
        }
    )
    result = V1PairedBCEAcquisitionState(
        registration_hash=registration.registration_hash,
        candidate_attempt_index=registration.candidate_attempt_index,
        stratum_counts=(counts[0], counts[1]),
        exact_scaled_loss_sums=(sums[0], sums[1]),
        observed_minima=(minima[0], minima[1]),
        observed_maxima=(maxima[0], maxima[1]),
        completed_look_count=completed,
        completed_chunk_count=state.chunk_count + 1,
        last_chunk_receipt_hash=receipt_hash,
        receipt_chain_hash=chain_hash,
        dynamic_oracle_receipt_hash=registration.dynamic_oracle_receipt_hash,
    )
    _validate_state_against_registration(registration, result)
    return result


@dataclass(frozen=True, slots=True)
class V1PairedBCECalibrationHorizonSkipDecision:
    """Typed, authority-inert retirement of one failed calibration attempt.

    No sampling law, O-frame registration, paired-loss row, or mask mutation is
    represented here.  The precommitted attempt-alpha coordinate is consumed
    and permanently retired because the irreversible queue decision itself was
    selected using candidate-specific K outcomes.
    """

    campaign_contract_hash: str
    controller_run_id: str
    acquisition_receipt_hash: str
    acquisition_generation_index: int
    accepted_step_index: int
    queue_index: int
    candidate_id: str
    parent_mask_hash: str
    candidate_mask_hash: str
    candidate_threshold_hash: str
    maximum_mean_damage: float
    candidate_attempt_alpha_axis_hash: str
    candidate_attempt_alpha_coordinate_hash: str
    candidate_attempt_index: int
    maximum_candidate_attempts: int
    oracle_familywise_alpha: float
    alpha_committed_before_attempt: float
    attempt_alpha_allocation: float
    master_decision_alpha_registration_hash: str | None
    causal_horizon_refusal_receipt: Mapping[str, Any]
    status: str = CALIBRATION_HORIZON_SKIP_STATUS
    reason: str = CALIBRATION_HORIZON_SKIP_REASON
    attempt_alpha_retired: bool = True
    unused_alpha_recycled: bool = False
    sampling_law_opened: bool = False
    oracle_registration_opened: bool = False
    oracle_stream_opened: bool = False
    oracle_rows_observed: bool = False
    mask_mutated: bool = False
    mask_commit_authority: bool = False
    alpha_nonrecycling_policy: str = ALPHA_NONRECYCLING_POLICY
    authority_scope: str = AUTHORITY_SCOPE
    schema: str = CALIBRATION_HORIZON_SKIP_DECISION_SCHEMA
    _decision_hash_cache: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash",
            "controller_run_id",
            "acquisition_receipt_hash",
            "candidate_id",
            "parent_mask_hash",
            "candidate_mask_hash",
            "candidate_threshold_hash",
            "candidate_attempt_alpha_axis_hash",
            "candidate_attempt_alpha_coordinate_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "acquisition_generation_index",
            "accepted_step_index",
            "queue_index",
            "candidate_attempt_index",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name),
            )
        maximum = _positive_int(
            self.maximum_candidate_attempts,
            name="maximum candidate attempts",
        )
        if self.candidate_attempt_index >= maximum:
            raise ValueError("calibration skip leaves its candidate-attempt horizon")
        loss_ceiling = _finite(
            self.maximum_mean_damage,
            name="calibration-skip maximum mean damage",
        )
        if loss_ceiling < 0.0:
            raise ValueError(
                "calibration-skip maximum mean damage cannot be negative"
            )
        familywise = _unit_alpha(
            self.oracle_familywise_alpha,
            name="calibration-skip oracle familywise alpha",
        )
        committed = _finite(
            self.alpha_committed_before_attempt,
            name="calibration-skip alpha committed before attempt",
        )
        allocation = _unit_alpha(
            self.attempt_alpha_allocation,
            name="calibration-skip attempt alpha allocation",
        )
        if committed < 0.0:
            raise ValueError("calibration-skip committed alpha cannot be negative")
        tolerance = 64.0 * math.ulp(
            max(1.0, familywise, committed, allocation)
        )
        if committed + allocation > familywise + tolerance:
            raise ValueError("calibration skip exceeds its oracle alpha reserve")
        expected_coordinate_hash = paired_bce_attempt_alpha_coordinate_hash(
            candidate_attempt_alpha_axis_hash=(
                self.candidate_attempt_alpha_axis_hash
            ),
            candidate_attempt_index=self.candidate_attempt_index,
            maximum_candidate_attempts=maximum,
            attempt_alpha_allocation=allocation,
        )
        if self.candidate_attempt_alpha_coordinate_hash != expected_coordinate_hash:
            raise ValueError("calibration skip alpha coordinate does not replay")
        master_hash = self.master_decision_alpha_registration_hash
        if master_hash is not None:
            master_hash = _identity(
                master_hash,
                name="master decision alpha registration hash",
            )
        refusal = validate_paired_bce_calibration_horizon_refusal_receipt(
            self.causal_horizon_refusal_receipt
        )
        if (
            refusal["candidate_mask_hash"] != self.candidate_mask_hash
            or refusal["candidate_attempt_index"] != self.candidate_attempt_index
        ):
            raise ValueError("calibration skip names another causal refusal")
        constants = {
            "status": CALIBRATION_HORIZON_SKIP_STATUS,
            "reason": CALIBRATION_HORIZON_SKIP_REASON,
            "attempt_alpha_retired": True,
            "unused_alpha_recycled": False,
            "sampling_law_opened": False,
            "oracle_registration_opened": False,
            "oracle_stream_opened": False,
            "oracle_rows_observed": False,
            "mask_mutated": False,
            "mask_commit_authority": False,
            "alpha_nonrecycling_policy": ALPHA_NONRECYCLING_POLICY,
            "authority_scope": AUTHORITY_SCOPE,
            "schema": CALIBRATION_HORIZON_SKIP_DECISION_SCHEMA,
        }
        for name, expected in constants.items():
            value = getattr(self, name)
            if value != expected or type(value) is not type(expected):
                raise ValueError(f"calibration horizon skip has invalid {name}")
        object.__setattr__(self, "maximum_candidate_attempts", maximum)
        object.__setattr__(self, "maximum_mean_damage", loss_ceiling)
        object.__setattr__(self, "oracle_familywise_alpha", familywise)
        object.__setattr__(self, "alpha_committed_before_attempt", committed)
        object.__setattr__(self, "attempt_alpha_allocation", allocation)
        object.__setattr__(
            self, "master_decision_alpha_registration_hash", master_hash
        )
        object.__setattr__(self, "causal_horizon_refusal_receipt", refusal)

    @property
    def refusal_receipt_hash(self) -> str:
        return str(self.causal_horizon_refusal_receipt["refusal_receipt_hash"])

    @property
    def refusal_ledger_hash(self) -> str:
        return content_hash(
            {
                "schema": (
                    "partition_free_v1_paired_bce_calibration_horizon_"
                    "refusal_ledger_v1"
                ),
                "causal_horizon_refusal_receipt_hash": self.refusal_receipt_hash,
                "candidate_attempt_alpha_axis_hash": (
                    self.candidate_attempt_alpha_axis_hash
                ),
                "candidate_attempt_alpha_coordinate_hash": (
                    self.candidate_attempt_alpha_coordinate_hash
                ),
                "candidate_attempt_index": self.candidate_attempt_index,
                "attempt_alpha_retired": True,
                "unused_alpha_recycled": False,
            },
            prefix="partition-free-v1-calibration-horizon-refusal-ledger-",
        )

    @property
    def decision_hash(self) -> str:
        return self.assert_integrity()

    def to_payload(self) -> Mapping[str, Any]:
        return MappingProxyType(_constructor_payload(self))

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any]
    ) -> V1PairedBCECalibrationHorizonSkipDecision:
        if not isinstance(payload, Mapping):
            raise TypeError("calibration horizon skip payload must be a mapping")
        return cls(**dict(payload))

    def _hash_payload(self) -> dict[str, Any]:
        return _constructor_payload(self)

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        return _assert_hash_integrity(
            self,
            payload=self._hash_payload(),
            cache_name="_decision_hash_cache",
            label="paired-BCE calibration horizon skip decision",
            expected_hash=expected_hash,
        )

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return _rebuild_hashed, (
            type(self),
            _constructor_payload(self),
            self.assert_integrity(),
        )


@dataclass(frozen=True, slots=True)
class V1PairedBCEDecision:
    """Authority-inert statistical decision at an acquisition boundary."""

    registration_hash: str
    state_hash: str
    candidate_attempt_alpha_axis_hash: str
    candidate_attempt_alpha_coordinate_hash: str
    candidate_attempt_index: int
    status: DecisionStatus
    reason: str
    stratum_counts: tuple[int, int]
    population_stratum_counts: tuple[int, int]
    stratum_point_means: tuple[float | None, float | None]
    stratum_confidence_intervals: tuple[
        tuple[float, float] | None, tuple[float, float] | None
    ]
    weighted_point_mean: float | None
    weighted_confidence_lower: float | None
    weighted_confidence_upper: float | None
    loss_ceiling: float
    evaluated_registered_look_index: int | None
    look_alpha: float
    cumulative_registered_alpha_through_look: float
    full_census: bool
    exact_finite_set_decision: bool
    partial_confidence_claim_bearing: bool
    calibration_prefix_stratum_counts: tuple[int, int]
    residual_oracle_stratum_counts: tuple[int, int]
    residual_oracle_population_stratum_counts: tuple[int, int]
    dynamic_oracle_receipt_hash: str | None
    mask_commit_authority: bool = False
    resource_exhausted: bool = False
    exact_census_method: str = EXACT_CENSUS_METHOD
    partial_evidence_method: str = CANONICAL_ANYTIME_BETTING_METHOD
    partial_evidence_ceiling: str = CANONICAL_ANYTIME_BETTING_EVIDENCE_CEILING
    alpha_nonrecycling_policy: str = ALPHA_NONRECYCLING_POLICY
    authority_scope: str = AUTHORITY_SCOPE
    schema: str = DECISION_SCHEMA
    _decision_hash_cache: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "registration_hash",
            "state_hash",
            "candidate_attempt_alpha_axis_hash",
            "candidate_attempt_alpha_coordinate_hash",
            "reason",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "candidate_attempt_index",
            _nonnegative_int(
                self.candidate_attempt_index, name="candidate attempt index"
            ),
        )
        if self.status not in ("acquire_more", "accept_candidate", "reject_candidate"):
            raise ValueError("paired-BCE decision status is invalid")
        counts = _pair_of_nonnegative_ints(self.stratum_counts, name="stratum counts")
        populations = _pair_of_nonnegative_ints(
            self.population_stratum_counts, name="population stratum counts"
        )
        calibration_counts = _pair_of_nonnegative_ints(
            self.calibration_prefix_stratum_counts,
            name="calibration prefix stratum counts",
        )
        residual_counts = _pair_of_nonnegative_ints(
            self.residual_oracle_stratum_counts,
            name="residual oracle stratum counts",
        )
        residual_populations = _pair_of_nonnegative_ints(
            self.residual_oracle_population_stratum_counts,
            name="residual oracle population stratum counts",
        )
        if any(
            calibration_counts[index] + residual_counts[index] != counts[index]
            or calibration_counts[index] + residual_populations[index]
            != populations[index]
            or residual_counts[index] > residual_populations[index]
            for index in (0, 1)
        ):
            raise ValueError("paired-BCE decision has incoherent K/O/full counts")
        dynamic_hash = self.dynamic_oracle_receipt_hash
        if dynamic_hash is not None:
            dynamic_hash = _identity(dynamic_hash, name="dynamic oracle receipt hash")
        if (dynamic_hash is None) != (calibration_counts == (0, 0)):
            raise ValueError("paired-BCE decision dynamic frame/K counts disagree")
        means = _pair_of_optional_finite(
            self.stratum_point_means, name="stratum point means"
        )
        intervals = tuple(self.stratum_confidence_intervals)
        if len(intervals) != 2:
            raise ValueError("paired-BCE decision requires two stratum intervals")
        normalized_intervals: list[tuple[float, float] | None] = []
        for interval in intervals:
            if interval is None:
                normalized_intervals.append(None)
                continue
            values = tuple(interval)
            if len(values) != 2:
                raise ValueError("paired-BCE stratum interval must have two endpoints")
            lower = _finite(values[0], name="stratum interval lower")
            upper = _finite(values[1], name="stratum interval upper")
            if lower > upper:
                raise ValueError("paired-BCE stratum interval is inverted")
            normalized_intervals.append((lower, upper))
        point = _optional_finite(self.weighted_point_mean, name="weighted point mean")
        lower = _optional_finite(
            self.weighted_confidence_lower, name="weighted confidence lower"
        )
        upper = _optional_finite(
            self.weighted_confidence_upper, name="weighted confidence upper"
        )
        if (lower is None) != (upper is None):
            raise ValueError("weighted confidence endpoints must be supplied together")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("weighted confidence interval is inverted")
        ceiling = _finite(self.loss_ceiling, name="loss ceiling")
        look_index = self.evaluated_registered_look_index
        if look_index is not None:
            look_index = _nonnegative_int(look_index, name="evaluated look index")
        look_alpha = _finite(self.look_alpha, name="look alpha")
        cumulative_alpha = _finite(
            self.cumulative_registered_alpha_through_look,
            name="cumulative registered alpha",
        )
        if look_alpha < 0.0 or cumulative_alpha < 0.0:
            raise ValueError("paired-BCE decision alpha values cannot be negative")
        for name in (
            "full_census",
            "exact_finite_set_decision",
            "partial_confidence_claim_bearing",
            "mask_commit_authority",
            "resource_exhausted",
        ):
            _strict_bool(getattr(self, name), name=name)
        if self.mask_commit_authority or self.resource_exhausted:
            raise ValueError("paired-BCE core cannot commit masks or exhaust at census")
        if self.full_census != self.exact_finite_set_decision:
            raise ValueError("joint census must use the exact finite-set decision")
        if self.full_census and self.status == "acquire_more":
            raise ValueError("joint census must return accept or reject")
        if not self.full_census and self.exact_finite_set_decision:
            raise ValueError("partial acquisition cannot claim an exact finite set")
        if self.status != "acquire_more" and point is None:
            raise ValueError("terminal paired-BCE decision requires a point mean")
        expected_constants = {
            "exact_census_method": EXACT_CENSUS_METHOD,
            "alpha_nonrecycling_policy": ALPHA_NONRECYCLING_POLICY,
            "authority_scope": AUTHORITY_SCOPE,
            "schema": DECISION_SCHEMA,
        }
        for name, expected in expected_constants.items():
            if getattr(self, name) != expected:
                raise ValueError(f"paired-BCE decision has invalid {name}")
        method_ceiling_pairs = {
            CANONICAL_ANYTIME_BETTING_METHOD: (
                CANONICAL_ANYTIME_BETTING_EVIDENCE_CEILING
            ),
            GROUP_SEQUENTIAL_METHOD: GROUP_SEQUENTIAL_EVIDENCE_CEILING,
        }
        if self.partial_evidence_method not in method_ceiling_pairs or (
            self.partial_evidence_ceiling
            != method_ceiling_pairs[self.partial_evidence_method]
        ):
            raise ValueError("paired-BCE decision partial evidence claim is invalid")
        object.__setattr__(self, "stratum_counts", counts)
        object.__setattr__(self, "population_stratum_counts", populations)
        object.__setattr__(
            self, "calibration_prefix_stratum_counts", calibration_counts
        )
        object.__setattr__(self, "residual_oracle_stratum_counts", residual_counts)
        object.__setattr__(
            self,
            "residual_oracle_population_stratum_counts",
            residual_populations,
        )
        object.__setattr__(self, "dynamic_oracle_receipt_hash", dynamic_hash)
        object.__setattr__(self, "stratum_point_means", means)
        object.__setattr__(
            self,
            "stratum_confidence_intervals",
            (normalized_intervals[0], normalized_intervals[1]),
        )
        object.__setattr__(self, "weighted_point_mean", point)
        object.__setattr__(self, "weighted_confidence_lower", lower)
        object.__setattr__(self, "weighted_confidence_upper", upper)
        object.__setattr__(self, "loss_ceiling", ceiling)
        object.__setattr__(self, "evaluated_registered_look_index", look_index)
        object.__setattr__(self, "look_alpha", look_alpha)
        object.__setattr__(
            self, "cumulative_registered_alpha_through_look", cumulative_alpha
        )

    @property
    def decision_hash(self) -> str:
        return self.assert_integrity()

    @property
    def candidate_acceptable(self) -> bool | None:
        if self.status == "accept_candidate":
            return True
        if self.status == "reject_candidate":
            return False
        return None

    def _hash_payload(self) -> dict[str, Any]:
        return _constructor_payload(self)

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        return _assert_hash_integrity(
            self,
            payload=self._hash_payload(),
            cache_name="_decision_hash_cache",
            label="paired-BCE acquisition decision",
            expected_hash=expected_hash,
        )

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return _rebuild_hashed, (
            type(self),
            _constructor_payload(self),
            self.assert_integrity(),
        )


def decide_paired_bce_acquisition(
    *,
    registration: V1PairedBCEAcquisitionRegistration,
    state: V1PairedBCEAcquisitionState,
    betting_replay: V1PairedBCEBettingReplay | None = None,
) -> V1PairedBCEDecision:
    """Evaluate only a registered look, with an exact mandatory census fallback."""

    _validate_state_against_registration(registration, state)
    populations = registration.population_stratum_counts
    full_populations = registration.full_population_stratum_counts
    calibration_counts = registration.calibration_prefix_stratum_counts
    full_census = state.stratum_counts == populations
    mean_fractions = tuple(
        _full_population_stratum_point_mean(
            registration=registration,
            state=state,
            index=index,
        )
        for index in (0, 1)
    )
    means: tuple[float | None, float | None] = tuple(
        None if value is None else float(value) for value in mean_fractions
    )  # type: ignore[assignment]
    residual_means: tuple[float | None, float | None] = tuple(
        None if count == 0 else float(_scaled_mean(total, count))
        for total, count in zip(
            state.exact_scaled_loss_sums, state.stratum_counts, strict=True
        )
    )  # type: ignore[assignment]
    accounted_counts = tuple(
        calibration_counts[index] + state.stratum_counts[index]
        for index in (0, 1)
    )
    look_index = (
        state.completed_look_count - 1 if state.completed_look_count > 0 else None
    )
    common = {
        "registration_hash": registration.registration_hash,
        "state_hash": state.state_hash,
        "candidate_attempt_alpha_axis_hash": (
            registration.candidate_attempt_alpha_axis_hash
        ),
        "candidate_attempt_alpha_coordinate_hash": (
            registration.candidate_attempt_alpha_coordinate_hash
        ),
        "candidate_attempt_index": registration.candidate_attempt_index,
        "stratum_counts": accounted_counts,
        "population_stratum_counts": full_populations,
        "stratum_point_means": means,
        "loss_ceiling": registration.loss_ceiling,
        "calibration_prefix_stratum_counts": calibration_counts,
        "residual_oracle_stratum_counts": state.stratum_counts,
        "residual_oracle_population_stratum_counts": populations,
        "dynamic_oracle_receipt_hash": registration.dynamic_oracle_receipt_hash,
    }
    if full_census:
        exact_mean = _full_population_exact_mean(
            registration=registration,
            state=state,
        )
        ceiling = Fraction.from_float(registration.loss_ceiling)
        accepted = exact_mean <= ceiling
        point = float(exact_mean)
        return V1PairedBCEDecision(
            **common,
            status="accept_candidate" if accepted else "reject_candidate",
            reason=(
                "exact_joint_census_mean_at_or_below_ceiling"
                if accepted
                else "exact_joint_census_mean_above_ceiling"
            ),
            stratum_confidence_intervals=(
                (means[0], means[0]),  # type: ignore[arg-type]
                (means[1], means[1]),  # type: ignore[arg-type]
            ),
            weighted_point_mean=point,
            weighted_confidence_lower=point,
            weighted_confidence_upper=point,
            evaluated_registered_look_index=look_index,
            look_alpha=0.0,
            cumulative_registered_alpha_through_look=math.fsum(
                registration.alpha_by_look
            ),
            full_census=True,
            exact_finite_set_decision=True,
            partial_confidence_claim_bearing=False,
        )

    at_joint_registered_look = bool(
        look_index is not None
        and state.stratum_counts
        == registration.registered_look_stratum_counts[look_index]
    )
    if not at_joint_registered_look:
        return V1PairedBCEDecision(
            **common,
            status="acquire_more",
            reason="between_registered_joint_looks_no_statistical_inspection",
            stratum_confidence_intervals=(None, None),
            weighted_point_mean=(
                None
                if any(value is None for value in means)
                else _stable_equal_weight_mean(means[0], means[1])  # type: ignore[arg-type]
            ),
            weighted_confidence_lower=None,
            weighted_confidence_upper=None,
            evaluated_registered_look_index=None,
            look_alpha=0.0,
            cumulative_registered_alpha_through_look=math.fsum(
                registration.alpha_by_look[: state.completed_look_count]
            ),
            full_census=False,
            exact_finite_set_decision=False,
            partial_confidence_claim_bearing=False,
        )
    assert look_index is not None
    look_alpha = registration.alpha_by_look[look_index]
    if not registration.finite_partial_bounds_available:
        return V1PairedBCEDecision(
            **common,
            status="acquire_more",
            reason="finite_preoutcome_stratum_bounds_unavailable_or_vacuous",
            stratum_confidence_intervals=(None, None),
            weighted_point_mean=_stable_equal_weight_mean(
                means[0], means[1]  # type: ignore[arg-type]
            ),
            weighted_confidence_lower=None,
            weighted_confidence_upper=None,
            evaluated_registered_look_index=look_index,
            look_alpha=look_alpha,
            cumulative_registered_alpha_through_look=math.fsum(
                registration.alpha_by_look[: look_index + 1]
            ),
            full_census=False,
            exact_finite_set_decision=False,
            partial_confidence_claim_bearing=False,
        )

    if betting_replay is not None:
        if type(betting_replay) is not V1PairedBCEBettingReplay:
            raise TypeError("canonical paired-BCE partial decision requires betting replay")
        residual_betting_intervals, betting_refusal = _canonical_betting_intervals(
            registration=registration,
            state=state,
            replay=betting_replay,
        )
        if residual_betting_intervals is None:
            reason = (
                "canonical_bounded_wor_betting_cs_numerical_refusal"
                if betting_refusal == "numerical"
                else "canonical_bounded_wor_betting_cs_empty_refusal"
            )
            return V1PairedBCEDecision(
                **common,
                status="acquire_more",
                reason=reason,
                stratum_confidence_intervals=(None, None),
                weighted_point_mean=_stable_equal_weight_mean(
                    means[0], means[1]  # type: ignore[arg-type]
                ),
                weighted_confidence_lower=None,
                weighted_confidence_upper=None,
                evaluated_registered_look_index=look_index,
                look_alpha=registration.attempt_alpha_allocation,
                cumulative_registered_alpha_through_look=(
                    registration.attempt_alpha_allocation
                ),
                full_census=False,
                exact_finite_set_decision=False,
                partial_confidence_claim_bearing=False,
            )
        betting_intervals = tuple(
            _lift_residual_interval_to_full_population(
                registration=registration,
                index=index,
                interval=residual_betting_intervals[index],
            )
            for index in (0, 1)
        )
        weighted_point = _stable_equal_weight_mean(
            means[0], means[1]  # type: ignore[arg-type]
        )
        weighted_lower = _weighted_confidence_endpoint(
            betting_intervals[0][0],
            betting_intervals[1][0],
            direction="lower",
        )
        weighted_upper = _weighted_confidence_endpoint(
            betting_intervals[0][1],
            betting_intervals[1][1],
            direction="upper",
        )
        if weighted_upper <= registration.loss_ceiling:
            betting_status: DecisionStatus = "accept_candidate"
            betting_reason = "anytime_betting_upper_at_or_below_ceiling"
        elif weighted_lower > registration.loss_ceiling:
            betting_status = "reject_candidate"
            betting_reason = "anytime_betting_lower_above_ceiling"
        else:
            betting_status = "acquire_more"
            betting_reason = "anytime_betting_interval_straddles_ceiling"
        return V1PairedBCEDecision(
            **common,
            status=betting_status,
            reason=betting_reason,
            stratum_confidence_intervals=(
                betting_intervals[0],
                betting_intervals[1],
            ),
            weighted_point_mean=weighted_point,
            weighted_confidence_lower=weighted_lower,
            weighted_confidence_upper=weighted_upper,
            evaluated_registered_look_index=look_index,
            look_alpha=registration.attempt_alpha_allocation,
            cumulative_registered_alpha_through_look=(
                registration.attempt_alpha_allocation
            ),
            full_census=False,
            exact_finite_set_decision=False,
            partial_confidence_claim_bearing=True,
        )
    if not registration.group_sequential_fallback_enabled:
        return V1PairedBCEDecision(
            **common,
            status="acquire_more",
            reason="canonical_anytime_betting_replay_required_for_partial_decision",
            stratum_confidence_intervals=(None, None),
            weighted_point_mean=_stable_equal_weight_mean(
                means[0], means[1]  # type: ignore[arg-type]
            ),
            weighted_confidence_lower=None,
            weighted_confidence_upper=None,
            evaluated_registered_look_index=look_index,
            look_alpha=registration.attempt_alpha_allocation,
            cumulative_registered_alpha_through_look=(
                registration.attempt_alpha_allocation
            ),
            full_census=False,
            exact_finite_set_decision=False,
            partial_confidence_claim_bearing=False,
        )

    # Four simultaneous one-sided statements at this registered joint look:
    # two tails in each of two strata.  Serfling's finite-population factor is
    # 1 - (n - 1) / N.  A censused individual stratum is exact.
    intervals: list[tuple[float, float]] = []
    for stratum, count, population, mean in zip(
        registration.strata,
        state.stratum_counts,
        populations,
        residual_means,
        strict=True,
    ):
        assert mean is not None
        lower_bound = stratum.paired_loss_lower_bound
        upper_bound = stratum.paired_loss_upper_bound
        span = stratum.bound_span
        assert lower_bound is not None and upper_bound is not None and span is not None
        if count == population:
            lower, upper = mean, mean
        else:
            radius = _serfling_radius_upper(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                count=count,
                population=population,
                look_alpha=look_alpha,
            )
            if not math.isfinite(radius):
                return V1PairedBCEDecision(
                    **common,
                    status="acquire_more",
                    reason="finite_bound_produced_vacuous_nonfinite_partial_radius",
                    stratum_confidence_intervals=(None, None),
                    weighted_point_mean=_stable_equal_weight_mean(
                        means[0], means[1]  # type: ignore[arg-type]
                    ),
                    weighted_confidence_lower=None,
                    weighted_confidence_upper=None,
                    evaluated_registered_look_index=look_index,
                    look_alpha=look_alpha,
                    cumulative_registered_alpha_through_look=math.fsum(
                        registration.alpha_by_look[: look_index + 1]
                    ),
                    full_census=False,
                    exact_finite_set_decision=False,
                    partial_confidence_claim_bearing=False,
                    partial_evidence_method=GROUP_SEQUENTIAL_METHOD,
                    partial_evidence_ceiling=GROUP_SEQUENTIAL_EVIDENCE_CEILING,
                )
            exact_mean = Fraction.from_float(mean)
            exact_radius = Fraction.from_float(radius)
            lower_exact = max(
                Fraction.from_float(lower_bound),
                exact_mean - exact_radius,
            )
            upper_exact = min(
                Fraction.from_float(upper_bound),
                exact_mean + exact_radius,
            )
            lower = _fraction_to_float_outward(
                lower_exact, direction="lower"
            )
            upper = _fraction_to_float_outward(
                upper_exact, direction="upper"
            )
        intervals.append((lower, upper))
    intervals = [
        _lift_residual_interval_to_full_population(
            registration=registration,
            index=index,
            interval=interval,
        )
        for index, interval in enumerate(intervals)
    ]
    weighted_point = _stable_equal_weight_mean(
        means[0], means[1]  # type: ignore[arg-type]
    )
    weighted_lower = _weighted_confidence_endpoint(
        intervals[0][0], intervals[1][0], direction="lower"
    )
    weighted_upper = _weighted_confidence_endpoint(
        intervals[0][1], intervals[1][1], direction="upper"
    )
    if weighted_upper <= registration.loss_ceiling:
        status: DecisionStatus = "accept_candidate"
        reason = "registered_serfling_upper_at_or_below_ceiling"
    elif weighted_lower > registration.loss_ceiling:
        status = "reject_candidate"
        reason = "registered_serfling_lower_above_ceiling"
    else:
        status = "acquire_more"
        reason = "registered_serfling_interval_straddles_ceiling"
    return V1PairedBCEDecision(
        **common,
        status=status,
        reason=reason,
        stratum_confidence_intervals=(intervals[0], intervals[1]),
        weighted_point_mean=weighted_point,
        weighted_confidence_lower=weighted_lower,
        weighted_confidence_upper=weighted_upper,
        evaluated_registered_look_index=look_index,
        look_alpha=look_alpha,
        cumulative_registered_alpha_through_look=math.fsum(
            registration.alpha_by_look[: look_index + 1]
        ),
        full_census=False,
        exact_finite_set_decision=False,
        partial_confidence_claim_bearing=True,
        partial_evidence_method=GROUP_SEQUENTIAL_METHOD,
        partial_evidence_ceiling=GROUP_SEQUENTIAL_EVIDENCE_CEILING,
    )


__all__ = [
    "ACQUISITION_REGISTRATION_SCHEMA",
    "ALPHA_NONRECYCLING_POLICY",
    "ANALYTIC_FIXED_AFFINE_PAIRED_BCE_KIND",
    "AUTHORITY_SCOPE",
    "BOUNDED_BRIER_PAIRED_LOSS_KIND",
    "CALIBRATION_HORIZON_REFUSAL_SCHEMA",
    "CALIBRATION_HORIZON_REFUSAL_STATUS",
    "CALIBRATION_HORIZON_SKIP_DECISION_SCHEMA",
    "CALIBRATION_HORIZON_SKIP_REASON",
    "CALIBRATION_HORIZON_SKIP_STATUS",
    "CANONICAL_ANYTIME_BETTING_EVIDENCE_CEILING",
    "CANONICAL_ANYTIME_BETTING_METHOD",
    "CENSUS_ONLY_PAIRED_BCE_KIND",
    "CLIPPED_LOGIT_PAIRED_BCE_KIND",
    "COMPLEMENT_STRATUM",
    "DYNAMIC_ORACLE_RECEIPT_SCHEMA",
    "EXACT_CENSUS_METHOD",
    "GROUP_SEQUENTIAL_EVIDENCE_CEILING",
    "GROUP_SEQUENTIAL_METHOD",
    "PAIRED_BCE_STRATA",
    "PAIRED_BCE_STRATUM_WEIGHTS",
    "PREDICTABLE_STRATUM_POLICY",
    "SEALED_HMAC_WOR_SAMPLING_LAW",
    "TARGET_STRATUM",
    "V1PairedBCEAcquisitionRegistration",
    "V1PairedBCEAcquisitionState",
    "V1PairedBCEBettingReplay",
    "V1PairedBCEChunkReceipt",
    "V1PairedBCEChunkRequest",
    "V1PairedBCECalibrationHorizonSkipDecision",
    "V1PairedBCEDecision",
    "V1PairedBCEDynamicOracleReceipt",
    "V1PairedBCEOracleRoleSeparationReceipt",
    "V1PairedBCEStratumRegistration",
    "V1PairedBCEVerifiedSealedOrder",
    "build_paired_bce_acquisition_registration",
    "commit_paired_bce_chunk",
    "decide_paired_bce_acquisition",
    "geometric_paired_bce_look_schedule",
    "initial_paired_bce_acquisition_state",
    "make_next_paired_bce_chunk_request",
    "next_paired_bce_chunk_coordinates",
    "paired_bce_attempt_alpha_coordinate_hash",
    "paired_bce_loss_ceiling_identity_hash",
    "paired_bce_sealed_order_hash",
    "reduce_paired_bce_chunk",
    "register_paired_bce_dynamic_oracle",
    "validate_paired_bce_chunk_request_preoutcome",
    "verify_paired_bce_sealed_order",
    "register_paired_bce_oracle_role_separation",
    "uniform_partial_look_alpha_schedule",
    "validate_paired_bce_calibration_horizon_refusal_receipt",
]
