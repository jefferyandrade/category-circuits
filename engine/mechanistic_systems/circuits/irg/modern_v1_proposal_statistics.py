"""Finite-population proposal statistics for modern-v1 pruning.

This module owns one narrow, validation-independent proposal-acquisition joint.  It
stores mergeable central moments of the signed corrected logit attribution
``-G phi c`` and its per-draw absolute value, and it turns an independently
verified analytic envelope into a group-sequential confidence box.

The confidence statement is design based.  Within each of two named strata,
the sealed order must be a uniform random permutation of the finite
population, fixed before outcomes are observed.  At a registered look with
``n < N``, for a variable in ``[a, b]`` we use the conservative two-sided
Hoeffding--Serfling implication

``P(|mean_n - mean_N| >= r) <= 2 exp(-2 n r^2 /
((1 - (n - 1) / N) (b - a)^2))``.

At census the finite-population mean is known exactly and the radius is set to
zero.  Ordering authority covers corrected *absolute* salience only, so its
Bonferroni allocation covers both strata, every live route, and every
registered look without spending alpha on the descriptive signed covector.
This is not a Wald, normal, empirical-Bernstein, Bayesian, or optional-stopping
claim.  Its evidence ceiling is the registered random-permutation law plus the
soundness of the pre-outcome analytic bound certificate.

The returned all-route box is the conservative distribution-free tier.  It
does not require a practical proposal controller to close a total order over
the full axis: a separately registered candidate-ladder rule may certify only
the comparisons needed to authorize the next irreversible action.

This module also owns the narrow, theorem-exact robust-regret certificate for
an additive corrected-attribution proposal.  Its action family must be fixed
before outcomes and must be either one global fixed-cardinality family or an
exact-quota partition-matroid base family.  For a registered proposal ``S``
and feasible competitor ``T``, the Cartesian box has the exact support value

``sup_a [a(S) - a(T)] = sum_{S\\T} upper - sum_{T\\S} lower``.

Consequently the worst competitor is obtained by minimizing the additive
cost ``q_i=upper_i`` on ``S`` and ``q_i=lower_i`` off ``S`` over the registered
family.  This certificate is proposal-discovery evidence only.  In
particular it is neither the nonlinear KKT/geometric selector nor authority
to attach or commit a mask.

At the canonical strict zero-regret budget, uniform-matroid and direct-sum
partition-matroid exchange optimality gives a sharper decision-equivalent
certificate.  In every nontrivial partition it is necessary and sufficient
that

``max_{i in S_p} upper_i <= min_{j outside S_p} lower_j``.

The zero-budget path therefore records only these partition boundaries and,
when the inequality fails, one exact positive exchange witness.  Zero- and
full-quota partitions are vacuous.  It neither constructs the full
adversarial cost vector nor materializes or sums the full competing action.
Nonzero regret budgets retain the exact additive optimizer above.

Each design certifies one proposal generation.  It must bind an external,
pre-outcome trajectory alpha ledger, the full alpha allocations already
committed to earlier generations, and this generation's allocation.  This
module does not advance
that trajectory ledger; production must publish a new ledger state before a
later relinearization can spend alpha, preventing an implicit ``0.05`` reset.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Final, Literal

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash


STRATIFIED_CORRECTED_ATTRIBUTION_MOMENTS_SCHEMA: Final = (
    "partition_free_v1_stratified_corrected_attribution_moments_v1"
)
CORRECTED_ATTRIBUTION_ENVELOPE_CERTIFICATE_SCHEMA: Final = (
    "partition_free_v1_corrected_attribution_envelope_certificate_v1"
)
GROUP_SEQUENTIAL_SERFLING_DESIGN_SCHEMA: Final = (
    "partition_free_v1_group_sequential_serfling_design_v1"
)
CORRECTED_ATTRIBUTION_CONFIDENCE_BOX_SCHEMA: Final = (
    "partition_free_v1_corrected_attribution_confidence_box_v1"
)
GROUP_SEQUENTIAL_EMPIRICAL_BERNSTEIN_SERFLING_DESIGN_SCHEMA: Final = (
    "partition_free_v1_group_sequential_empirical_bernstein_serfling_design_v2"
)
CORRECTED_ATTRIBUTION_CONFIDENCE_BOX_V2_SCHEMA: Final = (
    "partition_free_v1_corrected_attribution_confidence_box_v2"
)
CORRECTED_ATTRIBUTION_ACTION_FAMILY_SCHEMA: Final = (
    "partition_free_v1_corrected_attribution_action_family_v1"
)
CORRECTED_ATTRIBUTION_ACTION_REGRET_REQUEST_SCHEMA: Final = (
    "partition_free_v1_corrected_attribution_action_regret_request_v1"
)
CORRECTED_ATTRIBUTION_ACTION_REGRET_REQUEST_V2_SCHEMA: Final = (
    "partition_free_v1_corrected_attribution_action_regret_request_v2"
)
CORRECTED_ATTRIBUTION_ACTION_REGRET_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_corrected_attribution_action_regret_receipt_v2"
)
CORRECTED_ATTRIBUTION_ACTION_REGRET_RECEIPT_V3_SCHEMA: Final = (
    "partition_free_v1_corrected_attribution_action_regret_receipt_v3"
)
CORRECTED_ATTRIBUTION_ZERO_REGRET_EXCHANGE_CERTIFICATE_SCHEMA: Final = (
    "partition_free_v1_corrected_attribution_zero_regret_exchange_certificate_v1"
)

CORRECTED_ATTRIBUTION_MOMENT_REPRESENTATION: Final = (
    "ordered_vector_welford_mean_m2_canonical_little_endian_float64_v1"
)
ANALYTIC_ENVELOPE_DERIVATION_KIND: Final = "analytic_verified_bound_propagation_v1"
SEALED_WITHOUT_REPLACEMENT_SAMPLING_LAW: Final = (
    "sealed_independent_uniform_random_permutations_without_replacement_v1"
)
FIXED_TRAJECTORY_FWER_POLICY: Final = "fixed_nonrecycling_trajectory_bonferroni_fwer_v1"
ABSOLUTE_SALIENCE_ESTIMAND: Final = "absolute_salience"
EQUAL_STRATUM_MACRO_WEIGHTING: Final = "equal_stratum_macro_average_v1"
HOEFFDING_SERFLING_INEQUALITY_ID: Final = (
    "two_sided_hoeffding_serfling_1_minus_n_minus_1_over_N_v1"
)
BARDENET_MAILLARD_EMPIRICAL_BERNSTEIN_SERFLING_INEQUALITY_ID: Final = (
    "bardenet_maillard_2015_theorem_4_3_two_sided_via_delta_equals_"
    "alpha_over_10_v1"
)
HYBRID_SERFLING_CONFIDENCE_BOX_INEQUALITY_ID: Final = (
    "independent_equal_alpha_hs_eb_serfling_deterministic_support_"
    "intersection_v2"
)
EMPIRICAL_BERNSTEIN_BIASED_VARIANCE_CONVENTION: Final = (
    "biased_empirical_variance_m2_over_n_v1"
)
EMPIRICAL_BERNSTEIN_RHO_BRANCH_LE_HALF: Final = (
    "one_minus_n_minus_1_over_N_when_n_le_N_over_2_v1"
)
EMPIRICAL_BERNSTEIN_RHO_BRANCH_GT_HALF: Final = (
    "one_minus_n_over_N_times_one_plus_1_over_n_when_n_gt_N_over_2_v1"
)
DETERMINISTIC_FINITE_POPULATION_INTERVAL_ID: Final = (
    "bounded_unseen_finite_population_completion_interval_v1"
)
FLOAT64_OUTWARD_ROUNDING_POLICY: Final = (
    "float64_nextafter_each_positive_radius_operation_and_endpoint_v1"
)
FINITE_POPULATION_CENSUS_OVERRIDE_ID: Final = (
    "exact_sample_mean_collapse_when_n_equals_N_by_stratum_v1"
)
EMPIRICAL_BERNSTEIN_KAPPA: Final = 7.0 / 3.0 + 3.0 / math.sqrt(2.0)
SERFLING_EVIDENCE_CEILING: Final = (
    "finite_population_design_based_simultaneous_confidence_under_registered_"
    "sealed_uniform_random_permutations_sound_preoutcome_analytic_bounds_and_"
    "externally_advanced_fixed_trajectory_fwer_ledger"
)
HYBRID_SERFLING_EVIDENCE_CEILING: Final = (
    "finite_population_design_based_simultaneous_confidence_under_registered_"
    "sealed_uniform_random_permutations_sound_preoutcome_analytic_bounds_"
    "independent_equal_hs_eb_alpha_split_running_intersection_and_externally_"
    "advanced_fixed_trajectory_fwer_ledger"
)
GLOBAL_FIXED_CARDINALITY_ACTION_FAMILY: Final = (
    "global_fixed_cardinality_uniform_matroid_base_v1"
)
EXACT_QUOTA_PARTITION_MATROID_ACTION_FAMILY: Final = (
    "exact_quota_partition_matroid_base_v1"
)
CORRECTED_ATTRIBUTION_ACTION_REGRET_CURRENCY: Final = (
    "total_class_balanced_absolute_corrected_raw_logit_attribution_work_v1"
)
CORRECTED_ATTRIBUTION_ACTION_REGRET_THEOREM: Final = (
    "cartesian_box_support_via_adversarial_additive_matroid_base_minimization_v1"
)
CORRECTED_ATTRIBUTION_ZERO_REGRET_EXCHANGE_THEOREM: Final = (
    "uniform_and_direct_sum_exact_quota_matroid_base_one_exchange_"
    "optimality_equivalent_to_zero_cartesian_box_regret_v1"
)
CORRECTED_ATTRIBUTION_ACTION_REGRET_TIE_RULE: Final = (
    "minimum_cost_then_ascending_physical_route_index_v1"
)
CORRECTED_ATTRIBUTION_PROPOSAL_ONLY_AUTHORITY: Final = (
    "sampled_corrected_attribution_additive_action_discovery_only_v1"
)
ACTION_REGRET_V1_BOX_SEMANTICS: Final = (
    "v1_current_hoeffding_serfling_cartesian_box_v1"
)
ACTION_REGRET_V2_BOX_SEMANTICS: Final = (
    "v2_cumulative_hs_eb_serfling_running_intersection_cartesian_box_v1"
)
ACTION_PROPOSAL_CURRENT_POINT_COMPOSITION: Final = (
    "deterministic_additive_minimization_of_current_look_point_v1"
)

_MOMENTS_PREFIX: Final = "partition-free-v1-corrected-attribution-moments-"
_ENVELOPE_PREFIX: Final = "partition-free-v1-corrected-attribution-envelope-"
_DESIGN_PREFIX: Final = "partition-free-v1-group-sequential-serfling-design-"
_BOX_PREFIX: Final = "partition-free-v1-corrected-attribution-confidence-box-"
_HYBRID_DESIGN_PREFIX: Final = (
    "partition-free-v1-group-sequential-empirical-bernstein-serfling-design-"
)
_BOX_V2_PREFIX: Final = (
    "partition-free-v1-corrected-attribution-confidence-box-v2-"
)
_ACTION_FAMILY_PREFIX: Final = "partition-free-v1-corrected-attribution-action-family-"
_ACTION_PROPOSAL_PREFIX: Final = (
    "partition-free-v1-corrected-attribution-action-proposal-"
)
_ACTION_REGRET_REQUEST_PREFIX: Final = (
    "partition-free-v1-corrected-attribution-action-regret-request-"
)
_ACTION_REGRET_RECEIPT_PREFIX: Final = (
    "partition-free-v1-corrected-attribution-action-regret-receipt-"
)
_ZERO_REGRET_EXCHANGE_CERTIFICATE_PREFIX: Final = (
    "partition-free-v1-corrected-attribution-zero-regret-exchange-certificate-"
)
_LE_F64: Final = np.dtype("<f8")
_LE_U64: Final = np.dtype("<u8")
_EXCHANGE_BOUNDARY_SCAN_CHUNK_SIZE: Final = 1 << 20

ActionFamilyKind = Literal[
    "global_fixed_cardinality_uniform_matroid_base_v1",
    "exact_quota_partition_matroid_base_v1",
]


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _optional_identity(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _identity(value, name=name)


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be one positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be one positive integer")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be one nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be one nonnegative integer")
    return result


def _finite_nonnegative_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be one finite nonnegative scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be one finite nonnegative scalar")
    return result


def _pair_of_names(value: Sequence[str], *, name: str) -> tuple[str, str]:
    result = tuple(_identity(item, name=f"{name} item") for item in value)
    if len(result) != 2 or result[0] == result[1]:
        raise ValueError(f"{name} must contain exactly two distinct names")
    return result  # type: ignore[return-value]


def _pair_of_identities(value: Sequence[str], *, name: str) -> tuple[str, str]:
    result = tuple(_identity(item, name=f"{name} item") for item in value)
    if len(result) != 2:
        raise ValueError(f"{name} must contain exactly two identities")
    return result  # type: ignore[return-value]


def _pair_of_positive_ints(value: Sequence[int], *, name: str) -> tuple[int, int]:
    result = tuple(_positive_int(item, name=f"{name} item") for item in tuple(value))
    if len(result) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    return result  # type: ignore[return-value]


def _pair_of_nonnegative_ints(value: Sequence[int], *, name: str) -> tuple[int, int]:
    result = tuple(_nonnegative_int(item, name=f"{name} item") for item in tuple(value))
    if len(result) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    return result  # type: ignore[return-value]


def _nested_identity_pair(
    value: Sequence[Sequence[str]], *, name: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    outer = tuple(tuple(row) for row in value)
    if len(outer) != 2:
        raise ValueError(f"{name} must contain exactly two ledgers")
    result = tuple(
        tuple(_identity(item, name=f"{name} item") for item in row) for row in outer
    )
    return result  # type: ignore[return-value]


def _readonly_le_f64(
    value: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...],
    nonnegative: bool = False,
) -> np.ndarray:
    """Return a canonical array whose backing store cannot become writable.

    Merely calling ``setflags(write=False)`` is reversible for an owning NumPy
    array.  Here the public array views immutable ``bytes`` instead.  The
    constructor also normalizes signed zero and byte order so equivalent
    numerical state has one byte representation.
    """

    try:
        canonical = np.array(value, dtype=_LE_F64, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a canonical float64 array") from error
    if canonical.shape != shape or not np.isfinite(canonical).all():
        raise ValueError(f"{name} must be one finite array with shape {shape}")
    if nonnegative and np.any(canonical < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    # Canonicalize -0.0 without creating Python objects per route.
    canonical[canonical == 0.0] = 0.0
    immutable_bytes = canonical.tobytes(order="C")
    result = np.frombuffer(immutable_bytes, dtype=_LE_F64).reshape(shape)
    result.setflags(write=False)
    return result


def _readonly_vector(value: np.ndarray, *, name: str, size: int) -> np.ndarray:
    return _readonly_le_f64(value, name=name, shape=(size,))


def _readonly_le_u64_vector(
    value: Sequence[int] | np.ndarray,
    *,
    name: str,
    size: int | None = None,
) -> np.ndarray:
    """Return a canonical immutable unsigned route-index vector.

    Floating-point and object arrays are deliberately rejected rather than
    silently truncated into physical route coordinates.
    """

    try:
        source = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be one exact integer vector") from error
    if source.ndim != 1 or (source.size and source.dtype.kind not in ("i", "u")):
        raise ValueError(f"{name} must be one exact integer vector")
    if source.dtype.kind == "i" and bool(np.any(source < 0)):
        raise ValueError(f"{name} cannot contain negative coordinates")
    try:
        canonical = np.ascontiguousarray(source, dtype=_LE_U64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must fit the unsigned 64-bit route axis") from error
    if size is not None and int(canonical.size) != size:
        raise ValueError(f"{name} must contain exactly {size} coordinates")
    immutable_bytes = canonical.tobytes(order="C")
    result = np.frombuffer(immutable_bytes, dtype=_LE_U64)
    result.setflags(write=False)
    return result


def _nextafter_up(value: float) -> float:
    """Round one nonnegative scalar toward positive infinity."""

    result = float(np.nextafter(np.float64(value), np.float64(np.inf)))
    if result < 0.0 or math.isnan(result):
        raise ValueError("outward-rounded scalar must remain nonnegative")
    return result


def _positive_product_up(left: float, right: float) -> float:
    return _nextafter_up(float(left) * float(right))


def _positive_quotient_up(numerator: float, denominator: float) -> float:
    return _nextafter_up(float(numerator) / float(denominator))


def _positive_sqrt_up(value: float) -> float:
    return _nextafter_up(math.sqrt(float(value)))


def _hs_correction_and_multiplier_up(
    *, count: int, population: int, log_factor: float
) -> tuple[float, float]:
    correction = _nextafter_up(1.0 - (count - 1.0) / population)
    radicand = _positive_quotient_up(
        _positive_product_up(correction, log_factor),
        2.0 * count,
    )
    return correction, _positive_sqrt_up(radicand)


def _eb_rho_and_multipliers_up(
    *, count: int, population: int, log_factor: float
) -> tuple[float, str, float, float]:
    if count <= population / 2.0:
        rho = _nextafter_up(1.0 - (count - 1.0) / population)
        branch = EMPIRICAL_BERNSTEIN_RHO_BRANCH_LE_HALF
    else:
        rho = _positive_product_up(
            1.0 - count / population,
            1.0 + 1.0 / count,
        )
        branch = EMPIRICAL_BERNSTEIN_RHO_BRANCH_GT_HALF
    sigma_radicand = _positive_quotient_up(
        _positive_product_up(_positive_product_up(2.0, rho), log_factor),
        count,
    )
    sigma_multiplier = _positive_sqrt_up(sigma_radicand)
    range_multiplier = _positive_quotient_up(
        _positive_product_up(EMPIRICAL_BERNSTEIN_KAPPA, log_factor),
        count,
    )
    return rho, branch, sigma_multiplier, range_multiplier


def _canonical_route_set(
    value: Sequence[int] | np.ndarray,
    *,
    route_count: int,
    expected_count: int,
    name: str,
) -> np.ndarray:
    result = np.array(
        _readonly_le_u64_vector(value, name=name, size=expected_count),
        dtype=_LE_U64,
        copy=True,
    )
    if result.size:
        result.sort(kind="stable")
        if int(result[-1]) >= route_count or bool(np.any(result[1:] == result[:-1])):
            raise ValueError(
                f"{name} must be unique and lie on the registered route axis"
            )
    return _readonly_le_u64_vector(result, name=name, size=expected_count)


def _validate_row_inputs(
    *,
    route_count: int | None,
    stratum_names: tuple[str, str],
    population_stratum_counts: tuple[int, int],
    start_counts: tuple[int, int],
    context_ids: Sequence[str],
    draw_ids: Sequence[str],
    expected_context_ids: Sequence[str],
    expected_draw_ids: Sequence[str],
    stratum_labels: Sequence[str],
    sealed_order_indices: Sequence[int],
    signed_rows: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:
    contexts = tuple(_identity(item, name="context id") for item in context_ids)
    draws = tuple(_identity(item, name="draw id") for item in draw_ids)
    expected_contexts = tuple(
        _identity(item, name="expected context id") for item in expected_context_ids
    )
    expected_draws = tuple(
        _identity(item, name="expected draw id") for item in expected_draw_ids
    )
    labels = tuple(_identity(item, name="stratum label") for item in stratum_labels)
    raw_indices = tuple(sealed_order_indices)
    if not contexts or not (
        len(contexts) == len(draws) == len(labels) == len(raw_indices)
    ):
        raise ValueError("a signed-row prefix must contain aligned nonempty ledgers")
    if len(set(contexts)) != len(contexts) or len(set(draws)) != len(draws):
        raise ValueError("a signed-row prefix cannot repeat a context or draw")
    if contexts != expected_contexts or draws != expected_draws:
        raise ValueError("observed rows differ from the sealed request order")

    try:
        rows = np.asarray(signed_rows, dtype=_LE_F64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("signed corrected-attribution rows must be float64") from error
    if (
        rows.ndim != 2
        or rows.shape[0] != len(contexts)
        or rows.shape[1] <= 0
        or not np.isfinite(rows).all()
    ):
        raise ValueError(
            "signed corrected-attribution rows must be one aligned finite matrix"
        )
    observed_route_count = int(rows.shape[1])
    if route_count is not None and observed_route_count != route_count:
        raise ValueError("signed-row route axis differs from the prior state")

    label_to_index = {name: index for index, name in enumerate(stratum_names)}
    stratum_indices = np.empty(len(labels), dtype=np.int8)
    positions = np.empty(len(labels), dtype=np.int64)
    next_positions = [int(start_counts[0]), int(start_counts[1])]
    for row_index, (label, raw_position) in enumerate(zip(labels, raw_indices)):
        if label not in label_to_index:
            raise ValueError("signed-row prefix names an unregistered stratum")
        stratum = label_to_index[label]
        position = _nonnegative_int(raw_position, name="sealed order index")
        if position != next_positions[stratum]:
            raise ValueError(
                "each stratum must extend its sealed population order "
                "by an exact prefix"
            )
        if position >= population_stratum_counts[stratum]:
            raise ValueError("signed-row prefix exhausts its finite population")
        stratum_indices[row_index] = stratum
        positions[row_index] = position
        next_positions[stratum] += 1
    return (
        contexts,
        draws,
        stratum_indices,
        positions,
        np.ascontiguousarray(rows, dtype=_LE_F64),
        observed_route_count,
    )


def _ordered_welford_append(
    *,
    signed_mean_by_stratum: np.ndarray,
    signed_m2_by_stratum: np.ndarray,
    absolute_mean_by_stratum: np.ndarray,
    absolute_m2_by_stratum: np.ndarray,
    start_counts: tuple[int, int],
    stratum_indices: np.ndarray,
    signed_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Append rows with a vectorized route axis and deterministic row order."""

    signed_mean = np.array(signed_mean_by_stratum, dtype=_LE_F64, copy=True)
    signed_m2 = np.array(signed_m2_by_stratum, dtype=_LE_F64, copy=True)
    absolute_mean = np.array(absolute_mean_by_stratum, dtype=_LE_F64, copy=True)
    absolute_m2 = np.array(absolute_m2_by_stratum, dtype=_LE_F64, copy=True)
    counts = [int(start_counts[0]), int(start_counts[1])]

    # Python iteration is over acquired contexts, never routes.  Every route
    # update remains a contiguous vector operation, including at ImageNet
    # route scale.
    for row_index in range(int(signed_rows.shape[0])):
        stratum = int(stratum_indices[row_index])
        counts[stratum] += 1
        count = counts[stratum]
        signed = signed_rows[row_index]
        signed_delta = signed - signed_mean[stratum]
        signed_mean[stratum] += signed_delta / count
        signed_m2[stratum] += signed_delta * (signed - signed_mean[stratum])

        absolute = np.abs(signed)
        absolute_delta = absolute - absolute_mean[stratum]
        absolute_mean[stratum] += absolute_delta / count
        absolute_m2[stratum] += absolute_delta * (absolute - absolute_mean[stratum])

    return (
        signed_mean,
        signed_m2,
        absolute_mean,
        absolute_m2,
        (counts[0], counts[1]),
    )


def _validate_reduced_packet(
    *,
    expected_route_count: int | None,
    population_stratum_counts: tuple[int, int],
    expected_start_counts: tuple[int, int],
    sealed_order_start_indices: Sequence[int],
    packet_stratum_counts: Sequence[int],
    context_ids_by_stratum: Sequence[Sequence[str]],
    draw_ids_by_stratum: Sequence[Sequence[str]],
    expected_context_ids_by_stratum: Sequence[Sequence[str]],
    expected_draw_ids_by_stratum: Sequence[Sequence[str]],
    signed_mean_by_stratum: np.ndarray | Sequence[Sequence[float]],
    signed_m2_by_stratum: np.ndarray | Sequence[Sequence[float]],
    absolute_mean_by_stratum: np.ndarray | Sequence[Sequence[float]],
    absolute_m2_by_stratum: np.ndarray | Sequence[Sequence[float]],
) -> tuple[
    tuple[int, int],
    tuple[tuple[str, ...], tuple[str, ...]],
    tuple[tuple[str, ...], tuple[str, ...]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:
    starts = _pair_of_nonnegative_ints(
        sealed_order_start_indices, name="sealed order start indices"
    )
    if starts != expected_start_counts:
        raise ValueError("a reduced packet must start at the exact persisted prefix")
    packet_counts = _pair_of_nonnegative_ints(
        packet_stratum_counts, name="packet stratum counts"
    )
    if packet_counts == (0, 0):
        raise ValueError("a reduced request packet must contain at least one draw")
    if any(
        start + count > population
        for start, count, population in zip(
            starts, packet_counts, population_stratum_counts
        )
    ):
        raise ValueError("a reduced request packet exhausts its finite population")
    contexts = _nested_identity_pair(
        context_ids_by_stratum, name="packet context ledgers"
    )
    draws = _nested_identity_pair(draw_ids_by_stratum, name="packet draw ledgers")
    expected_contexts = _nested_identity_pair(
        expected_context_ids_by_stratum,
        name="expected packet context ledgers",
    )
    expected_draws = _nested_identity_pair(
        expected_draw_ids_by_stratum,
        name="expected packet draw ledgers",
    )
    if contexts != expected_contexts or draws != expected_draws:
        raise ValueError("reduced packet differs from the sealed request order")
    if any(
        len(contexts[stratum]) != packet_counts[stratum]
        or len(draws[stratum]) != packet_counts[stratum]
        for stratum in (0, 1)
    ):
        raise ValueError("reduced-packet ledgers must exactly enumerate packet draws")
    if len(set(contexts[0] + contexts[1])) != sum(packet_counts):
        raise ValueError("a reduced request packet cannot repeat contexts")
    if len(set(draws[0] + draws[1])) != sum(packet_counts):
        raise ValueError("a reduced request packet cannot repeat draws")

    raw_arrays: list[np.ndarray] = []
    for name, value, nonnegative in (
        ("packet signed means", signed_mean_by_stratum, False),
        ("packet signed M2", signed_m2_by_stratum, True),
        ("packet absolute means", absolute_mean_by_stratum, True),
        ("packet absolute M2", absolute_m2_by_stratum, True),
    ):
        try:
            array = np.array(value, dtype=_LE_F64, order="C", copy=True)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must be float64") from error
        if (
            array.ndim != 2
            or array.shape[0] != 2
            or array.shape[1] <= 0
            or not np.isfinite(array).all()
            or (nonnegative and np.any(array < 0.0))
        ):
            raise ValueError(f"{name} must be one finite 2-by-route array")
        raw_arrays.append(array)
    route_count = int(raw_arrays[0].shape[1])
    if any(array.shape != (2, route_count) for array in raw_arrays):
        raise ValueError("all reduced-packet moments must share one route axis")
    if expected_route_count is not None and route_count != expected_route_count:
        raise ValueError("reduced-packet route axis differs from persisted state")

    signed_mean, signed_m2, absolute_mean, absolute_m2 = raw_arrays
    for stratum, count in enumerate(packet_counts):
        if count == 0 and any(np.any(array[stratum] != 0.0) for array in raw_arrays):
            raise ValueError("an empty reduced-packet stratum must have zero moments")
        if count < 2 and (
            np.any(signed_m2[stratum] != 0.0) or np.any(absolute_m2[stratum] != 0.0)
        ):
            raise ValueError("packet M2 must vanish before two stratum draws")
    tolerance = 128.0 * np.finfo(np.float64).eps * np.maximum(1.0, absolute_mean)
    if np.any(np.abs(signed_mean) > absolute_mean + tolerance):
        raise ValueError("reduced signed and absolute means violate Jensen's bound")
    return (
        packet_counts,
        contexts,
        draws,
        signed_mean,
        signed_m2,
        absolute_mean,
        absolute_m2,
        route_count,
    )


def _chan_merge_stratified_mean_m2(
    *,
    left_mean: np.ndarray,
    left_m2: np.ndarray,
    left_counts: tuple[int, int],
    right_mean: np.ndarray,
    right_m2: np.ndarray,
    right_counts: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    means = np.empty_like(left_mean, dtype=_LE_F64)
    m2s = np.empty_like(left_m2, dtype=_LE_F64)
    for stratum in (0, 1):
        left_count = left_counts[stratum]
        right_count = right_counts[stratum]
        if left_count == 0:
            means[stratum] = right_mean[stratum]
            m2s[stratum] = right_m2[stratum]
            continue
        if right_count == 0:
            means[stratum] = left_mean[stratum]
            m2s[stratum] = left_m2[stratum]
            continue
        total = left_count + right_count
        delta = right_mean[stratum] - left_mean[stratum]
        means[stratum] = left_mean[stratum] + delta * (right_count / total)
        m2s[stratum] = (
            left_m2[stratum]
            + right_m2[stratum]
            + np.square(delta, dtype=_LE_F64) * (left_count * right_count / total)
        )
    return means, m2s


@dataclass(frozen=True, slots=True)
class V1StratifiedCorrectedAttributionMoments:
    """Replayable central moments on two sealed finite-population prefixes."""

    campaign_contract_hash: str
    model_identity_hash: str
    query_contract_hash: str
    preprocessing_hash: str
    current_mask_hash: str
    raw_logit_stream_source_hash: str
    sampling_law_hash: str
    proposal_generation: int
    route_count: int
    route_axis_hash: str
    stratum_names: tuple[str, str]
    population_stratum_counts: tuple[int, int]
    sealed_population_order_hashes: tuple[str, str]
    stratum_counts: tuple[int, int]
    observed_context_ids_by_stratum: tuple[tuple[str, ...], tuple[str, ...]]
    observed_draw_ids_by_stratum: tuple[tuple[str, ...], tuple[str, ...]]
    signed_mean_by_stratum: np.ndarray
    signed_m2_by_stratum: np.ndarray
    absolute_mean_by_stratum: np.ndarray
    absolute_m2_by_stratum: np.ndarray
    look_index: int
    prior_state_hash: str | None = None
    schema: str = STRATIFIED_CORRECTED_ATTRIBUTION_MOMENTS_SCHEMA
    _state_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash",
            "model_identity_hash",
            "query_contract_hash",
            "preprocessing_hash",
            "current_mask_hash",
            "raw_logit_stream_source_hash",
            "sampling_law_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        proposal_generation = _nonnegative_int(
            self.proposal_generation, name="proposal generation"
        )
        route_count = _positive_int(self.route_count, name="route count")
        route_axis_hash = _identity(self.route_axis_hash, name="route axis hash")
        names = _pair_of_names(self.stratum_names, name="stratum names")
        populations = _pair_of_positive_ints(
            self.population_stratum_counts, name="population stratum counts"
        )
        order_hashes = _pair_of_identities(
            self.sealed_population_order_hashes,
            name="sealed population order hashes",
        )
        counts = _pair_of_nonnegative_ints(self.stratum_counts, name="stratum counts")
        contexts = _nested_identity_pair(
            self.observed_context_ids_by_stratum,
            name="observed context ledgers",
        )
        draws = _nested_identity_pair(
            self.observed_draw_ids_by_stratum,
            name="observed draw ledgers",
        )
        if any(count > population for count, population in zip(counts, populations)):
            raise ValueError("stratum counts cannot exceed finite populations")
        if any(
            len(contexts[index]) != counts[index] or len(draws[index]) != counts[index]
            for index in (0, 1)
        ):
            raise ValueError("observed ledgers must exactly identify each prefix")
        flat_contexts = contexts[0] + contexts[1]
        flat_draws = draws[0] + draws[1]
        if len(set(flat_contexts)) != len(flat_contexts):
            raise ValueError("observed finite-population contexts cannot repeat")
        if len(set(flat_draws)) != len(flat_draws):
            raise ValueError("observed finite-population draws cannot repeat")

        look_index = _positive_int(self.look_index, name="look index")
        if look_index == 1:
            if self.prior_state_hash is not None:
                raise ValueError("the first moments look cannot name a prior state")
        else:
            _identity(self.prior_state_hash, name="prior state hash")
        if self.schema != STRATIFIED_CORRECTED_ATTRIBUTION_MOMENTS_SCHEMA:
            raise ValueError("corrected-attribution moments schema is invalid")

        shape = (2, route_count)
        signed_mean = _readonly_le_f64(
            self.signed_mean_by_stratum,
            name="signed means",
            shape=shape,
        )
        signed_m2 = _readonly_le_f64(
            self.signed_m2_by_stratum,
            name="signed M2 central moments",
            shape=shape,
            nonnegative=True,
        )
        absolute_mean = _readonly_le_f64(
            self.absolute_mean_by_stratum,
            name="absolute means",
            shape=shape,
            nonnegative=True,
        )
        absolute_m2 = _readonly_le_f64(
            self.absolute_m2_by_stratum,
            name="absolute M2 central moments",
            shape=shape,
            nonnegative=True,
        )
        for stratum, count in enumerate(counts):
            if count == 0 and any(
                np.any(array[stratum] != 0.0)
                for array in (signed_mean, signed_m2, absolute_mean, absolute_m2)
            ):
                raise ValueError("empty strata must have zero central moments")
            if count < 2 and (
                np.any(signed_m2[stratum] != 0.0) or np.any(absolute_m2[stratum] != 0.0)
            ):
                raise ValueError("M2 must vanish before a stratum has two rows")

        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "proposal_generation", proposal_generation)
        object.__setattr__(self, "route_axis_hash", route_axis_hash)
        object.__setattr__(self, "stratum_names", names)
        object.__setattr__(self, "population_stratum_counts", populations)
        object.__setattr__(self, "sealed_population_order_hashes", order_hashes)
        object.__setattr__(self, "stratum_counts", counts)
        object.__setattr__(self, "observed_context_ids_by_stratum", contexts)
        object.__setattr__(self, "observed_draw_ids_by_stratum", draws)
        object.__setattr__(self, "signed_mean_by_stratum", signed_mean)
        object.__setattr__(self, "signed_m2_by_stratum", signed_m2)
        object.__setattr__(self, "absolute_mean_by_stratum", absolute_mean)
        object.__setattr__(self, "absolute_m2_by_stratum", absolute_m2)
        object.__setattr__(self, "look_index", look_index)

    @classmethod
    def from_signed_rows(
        cls,
        *,
        campaign_contract_hash: str,
        model_identity_hash: str,
        query_contract_hash: str,
        preprocessing_hash: str,
        current_mask_hash: str,
        raw_logit_stream_source_hash: str,
        sampling_law_hash: str,
        proposal_generation: int,
        route_axis_hash: str,
        stratum_names: Sequence[str],
        population_stratum_counts: Sequence[int],
        sealed_population_order_hashes: Sequence[str],
        context_ids: Sequence[str],
        draw_ids: Sequence[str],
        expected_context_ids: Sequence[str],
        expected_draw_ids: Sequence[str],
        stratum_labels: Sequence[str],
        sealed_order_indices: Sequence[int],
        signed_rows: Sequence[Sequence[float]] | np.ndarray,
    ) -> "V1StratifiedCorrectedAttributionMoments":
        """Create look one from signed ``-G phi c`` rows.

        ``sealed_order_indices`` are zero-based within each stratum.  Requiring
        an exact ``0, 1, ...`` prefix prevents duplicates, gaps, and outcome-
        dependent reordering without retaining the unobserved population.
        ``expected_*`` identities must come from the frozen pre-execution
        request; they are compared exactly with the executor's observed
        ledgers and may not be synthesized after measurement.
        """

        names = _pair_of_names(stratum_names, name="stratum names")
        populations = _pair_of_positive_ints(
            population_stratum_counts, name="population stratum counts"
        )
        order_hashes = _pair_of_identities(
            sealed_population_order_hashes,
            name="sealed population order hashes",
        )
        (
            contexts,
            draws,
            stratum_indices,
            _positions,
            rows,
            route_count,
        ) = _validate_row_inputs(
            route_count=None,
            stratum_names=names,
            population_stratum_counts=populations,
            start_counts=(0, 0),
            context_ids=context_ids,
            draw_ids=draw_ids,
            expected_context_ids=expected_context_ids,
            expected_draw_ids=expected_draw_ids,
            stratum_labels=stratum_labels,
            sealed_order_indices=sealed_order_indices,
            signed_rows=signed_rows,
        )
        zeros = np.zeros((2, route_count), dtype=_LE_F64)
        signed_mean, signed_m2, absolute_mean, absolute_m2, counts = (
            _ordered_welford_append(
                signed_mean_by_stratum=zeros,
                signed_m2_by_stratum=zeros,
                absolute_mean_by_stratum=zeros,
                absolute_m2_by_stratum=zeros,
                start_counts=(0, 0),
                stratum_indices=stratum_indices,
                signed_rows=rows,
            )
        )
        context_ledgers = tuple(
            tuple(
                contexts[row]
                for row in range(len(contexts))
                if int(stratum_indices[row]) == stratum
            )
            for stratum in (0, 1)
        )
        draw_ledgers = tuple(
            tuple(
                draws[row]
                for row in range(len(draws))
                if int(stratum_indices[row]) == stratum
            )
            for stratum in (0, 1)
        )
        return cls(
            campaign_contract_hash=campaign_contract_hash,
            model_identity_hash=model_identity_hash,
            query_contract_hash=query_contract_hash,
            preprocessing_hash=preprocessing_hash,
            current_mask_hash=current_mask_hash,
            raw_logit_stream_source_hash=raw_logit_stream_source_hash,
            sampling_law_hash=sampling_law_hash,
            proposal_generation=proposal_generation,
            route_count=route_count,
            route_axis_hash=route_axis_hash,
            stratum_names=names,
            population_stratum_counts=populations,
            sealed_population_order_hashes=order_hashes,
            stratum_counts=counts,
            observed_context_ids_by_stratum=context_ledgers,  # type: ignore[arg-type]
            observed_draw_ids_by_stratum=draw_ledgers,  # type: ignore[arg-type]
            signed_mean_by_stratum=signed_mean,
            signed_m2_by_stratum=signed_m2,
            absolute_mean_by_stratum=absolute_mean,
            absolute_m2_by_stratum=absolute_m2,
            look_index=1,
        )

    @classmethod
    def from_reduced_stratum_moments(
        cls,
        *,
        campaign_contract_hash: str,
        model_identity_hash: str,
        query_contract_hash: str,
        preprocessing_hash: str,
        current_mask_hash: str,
        raw_logit_stream_source_hash: str,
        sampling_law_hash: str,
        proposal_generation: int,
        route_axis_hash: str,
        stratum_names: Sequence[str],
        population_stratum_counts: Sequence[int],
        sealed_population_order_hashes: Sequence[str],
        sealed_order_start_indices: Sequence[int],
        packet_stratum_counts: Sequence[int],
        context_ids_by_stratum: Sequence[Sequence[str]],
        draw_ids_by_stratum: Sequence[Sequence[str]],
        expected_context_ids_by_stratum: Sequence[Sequence[str]],
        expected_draw_ids_by_stratum: Sequence[Sequence[str]],
        signed_mean_by_stratum: np.ndarray | Sequence[Sequence[float]],
        signed_m2_by_stratum: np.ndarray | Sequence[Sequence[float]],
        absolute_mean_by_stratum: np.ndarray | Sequence[Sequence[float]],
        absolute_m2_by_stratum: np.ndarray | Sequence[Sequence[float]],
    ) -> "V1StratifiedCorrectedAttributionMoments":
        """Create look one from one GPU-reduced sealed request packet.

        Production code should reduce each stratum on device to float64
        ``(count, mean, M2)`` arrays and transfer only those route vectors.
        No ``[context, route]`` row tensor is retained or required here.
        The expected ledgers are the frozen request order; the unprefixed
        ledgers are executor evidence and must match exactly.
        """

        names = _pair_of_names(stratum_names, name="stratum names")
        populations = _pair_of_positive_ints(
            population_stratum_counts, name="population stratum counts"
        )
        order_hashes = _pair_of_identities(
            sealed_population_order_hashes,
            name="sealed population order hashes",
        )
        (
            packet_counts,
            contexts,
            draws,
            signed_mean,
            signed_m2,
            absolute_mean,
            absolute_m2,
            route_count,
        ) = _validate_reduced_packet(
            expected_route_count=None,
            population_stratum_counts=populations,
            expected_start_counts=(0, 0),
            sealed_order_start_indices=sealed_order_start_indices,
            packet_stratum_counts=packet_stratum_counts,
            context_ids_by_stratum=context_ids_by_stratum,
            draw_ids_by_stratum=draw_ids_by_stratum,
            expected_context_ids_by_stratum=expected_context_ids_by_stratum,
            expected_draw_ids_by_stratum=expected_draw_ids_by_stratum,
            signed_mean_by_stratum=signed_mean_by_stratum,
            signed_m2_by_stratum=signed_m2_by_stratum,
            absolute_mean_by_stratum=absolute_mean_by_stratum,
            absolute_m2_by_stratum=absolute_m2_by_stratum,
        )
        return cls(
            campaign_contract_hash=campaign_contract_hash,
            model_identity_hash=model_identity_hash,
            query_contract_hash=query_contract_hash,
            preprocessing_hash=preprocessing_hash,
            current_mask_hash=current_mask_hash,
            raw_logit_stream_source_hash=raw_logit_stream_source_hash,
            sampling_law_hash=sampling_law_hash,
            proposal_generation=proposal_generation,
            route_count=route_count,
            route_axis_hash=route_axis_hash,
            stratum_names=names,
            population_stratum_counts=populations,
            sealed_population_order_hashes=order_hashes,
            stratum_counts=packet_counts,
            observed_context_ids_by_stratum=contexts,
            observed_draw_ids_by_stratum=draws,
            signed_mean_by_stratum=signed_mean,
            signed_m2_by_stratum=signed_m2,
            absolute_mean_by_stratum=absolute_mean,
            absolute_m2_by_stratum=absolute_m2,
            look_index=1,
        )

    def append_signed_rows(
        self,
        *,
        context_ids: Sequence[str],
        draw_ids: Sequence[str],
        expected_context_ids: Sequence[str],
        expected_draw_ids: Sequence[str],
        stratum_labels: Sequence[str],
        sealed_order_indices: Sequence[int],
        signed_rows: Sequence[Sequence[float]] | np.ndarray,
    ) -> "V1StratifiedCorrectedAttributionMoments":
        """Append the next disjoint per-stratum prefixes using Welford updates."""

        self.assert_integrity()
        (
            contexts,
            draws,
            stratum_indices,
            _positions,
            rows,
            _route_count,
        ) = _validate_row_inputs(
            route_count=self.route_count,
            stratum_names=self.stratum_names,
            population_stratum_counts=self.population_stratum_counts,
            start_counts=self.stratum_counts,
            context_ids=context_ids,
            draw_ids=draw_ids,
            expected_context_ids=expected_context_ids,
            expected_draw_ids=expected_draw_ids,
            stratum_labels=stratum_labels,
            sealed_order_indices=sealed_order_indices,
            signed_rows=signed_rows,
        )
        prior_contexts = set(
            self.observed_context_ids_by_stratum[0]
            + self.observed_context_ids_by_stratum[1]
        )
        prior_draws = set(
            self.observed_draw_ids_by_stratum[0] + self.observed_draw_ids_by_stratum[1]
        )
        if prior_contexts.intersection(contexts):
            raise ValueError("an appended prefix repeated an observed context")
        if prior_draws.intersection(draws):
            raise ValueError("an appended prefix repeated an observed draw")

        signed_mean, signed_m2, absolute_mean, absolute_m2, counts = (
            _ordered_welford_append(
                signed_mean_by_stratum=self.signed_mean_by_stratum,
                signed_m2_by_stratum=self.signed_m2_by_stratum,
                absolute_mean_by_stratum=self.absolute_mean_by_stratum,
                absolute_m2_by_stratum=self.absolute_m2_by_stratum,
                start_counts=self.stratum_counts,
                stratum_indices=stratum_indices,
                signed_rows=rows,
            )
        )
        added_contexts = tuple(
            tuple(
                contexts[row]
                for row in range(len(contexts))
                if int(stratum_indices[row]) == stratum
            )
            for stratum in (0, 1)
        )
        added_draws = tuple(
            tuple(
                draws[row]
                for row in range(len(draws))
                if int(stratum_indices[row]) == stratum
            )
            for stratum in (0, 1)
        )
        return type(self)(
            campaign_contract_hash=self.campaign_contract_hash,
            model_identity_hash=self.model_identity_hash,
            query_contract_hash=self.query_contract_hash,
            preprocessing_hash=self.preprocessing_hash,
            current_mask_hash=self.current_mask_hash,
            raw_logit_stream_source_hash=self.raw_logit_stream_source_hash,
            sampling_law_hash=self.sampling_law_hash,
            proposal_generation=self.proposal_generation,
            route_count=self.route_count,
            route_axis_hash=self.route_axis_hash,
            stratum_names=self.stratum_names,
            population_stratum_counts=self.population_stratum_counts,
            sealed_population_order_hashes=self.sealed_population_order_hashes,
            stratum_counts=counts,
            observed_context_ids_by_stratum=(
                self.observed_context_ids_by_stratum[0] + added_contexts[0],
                self.observed_context_ids_by_stratum[1] + added_contexts[1],
            ),
            observed_draw_ids_by_stratum=(
                self.observed_draw_ids_by_stratum[0] + added_draws[0],
                self.observed_draw_ids_by_stratum[1] + added_draws[1],
            ),
            signed_mean_by_stratum=signed_mean,
            signed_m2_by_stratum=signed_m2,
            absolute_mean_by_stratum=absolute_mean,
            absolute_m2_by_stratum=absolute_m2,
            look_index=self.look_index + 1,
            prior_state_hash=self.state_hash,
        )

    def append_reduced_stratum_moments(
        self,
        *,
        sealed_order_start_indices: Sequence[int],
        packet_stratum_counts: Sequence[int],
        context_ids_by_stratum: Sequence[Sequence[str]],
        draw_ids_by_stratum: Sequence[Sequence[str]],
        expected_context_ids_by_stratum: Sequence[Sequence[str]],
        expected_draw_ids_by_stratum: Sequence[Sequence[str]],
        signed_mean_by_stratum: np.ndarray | Sequence[Sequence[float]],
        signed_m2_by_stratum: np.ndarray | Sequence[Sequence[float]],
        absolute_mean_by_stratum: np.ndarray | Sequence[Sequence[float]],
        absolute_m2_by_stratum: np.ndarray | Sequence[Sequence[float]],
    ) -> "V1StratifiedCorrectedAttributionMoments":
        """Chan-merge the next GPU-reduced sealed request packet.

        Packet boundaries are part of the registered acquisition protocol.
        Uninterrupted and resumed runs therefore perform the same sequence of
        Chan merges and have bitwise replay without demanding arithmetic that
        is invariant to unregistered regrouping.
        """

        self.assert_integrity()
        (
            packet_counts,
            contexts,
            draws,
            packet_signed_mean,
            packet_signed_m2,
            packet_absolute_mean,
            packet_absolute_m2,
            _route_count,
        ) = _validate_reduced_packet(
            expected_route_count=self.route_count,
            population_stratum_counts=self.population_stratum_counts,
            expected_start_counts=self.stratum_counts,
            sealed_order_start_indices=sealed_order_start_indices,
            packet_stratum_counts=packet_stratum_counts,
            context_ids_by_stratum=context_ids_by_stratum,
            draw_ids_by_stratum=draw_ids_by_stratum,
            expected_context_ids_by_stratum=expected_context_ids_by_stratum,
            expected_draw_ids_by_stratum=expected_draw_ids_by_stratum,
            signed_mean_by_stratum=signed_mean_by_stratum,
            signed_m2_by_stratum=signed_m2_by_stratum,
            absolute_mean_by_stratum=absolute_mean_by_stratum,
            absolute_m2_by_stratum=absolute_m2_by_stratum,
        )
        prior_contexts = set(
            self.observed_context_ids_by_stratum[0]
            + self.observed_context_ids_by_stratum[1]
        )
        prior_draws = set(
            self.observed_draw_ids_by_stratum[0] + self.observed_draw_ids_by_stratum[1]
        )
        if prior_contexts.intersection(contexts[0] + contexts[1]):
            raise ValueError("a reduced packet repeated an observed context")
        if prior_draws.intersection(draws[0] + draws[1]):
            raise ValueError("a reduced packet repeated an observed draw")

        signed_mean, signed_m2 = _chan_merge_stratified_mean_m2(
            left_mean=self.signed_mean_by_stratum,
            left_m2=self.signed_m2_by_stratum,
            left_counts=self.stratum_counts,
            right_mean=packet_signed_mean,
            right_m2=packet_signed_m2,
            right_counts=packet_counts,
        )
        absolute_mean, absolute_m2 = _chan_merge_stratified_mean_m2(
            left_mean=self.absolute_mean_by_stratum,
            left_m2=self.absolute_m2_by_stratum,
            left_counts=self.stratum_counts,
            right_mean=packet_absolute_mean,
            right_m2=packet_absolute_m2,
            right_counts=packet_counts,
        )
        counts = tuple(
            left + right for left, right in zip(self.stratum_counts, packet_counts)
        )
        return type(self)(
            campaign_contract_hash=self.campaign_contract_hash,
            model_identity_hash=self.model_identity_hash,
            query_contract_hash=self.query_contract_hash,
            preprocessing_hash=self.preprocessing_hash,
            current_mask_hash=self.current_mask_hash,
            raw_logit_stream_source_hash=self.raw_logit_stream_source_hash,
            sampling_law_hash=self.sampling_law_hash,
            proposal_generation=self.proposal_generation,
            route_count=self.route_count,
            route_axis_hash=self.route_axis_hash,
            stratum_names=self.stratum_names,
            population_stratum_counts=self.population_stratum_counts,
            sealed_population_order_hashes=self.sealed_population_order_hashes,
            stratum_counts=counts,  # type: ignore[arg-type]
            observed_context_ids_by_stratum=(
                self.observed_context_ids_by_stratum[0] + contexts[0],
                self.observed_context_ids_by_stratum[1] + contexts[1],
            ),
            observed_draw_ids_by_stratum=(
                self.observed_draw_ids_by_stratum[0] + draws[0],
                self.observed_draw_ids_by_stratum[1] + draws[1],
            ),
            signed_mean_by_stratum=signed_mean,
            signed_m2_by_stratum=signed_m2,
            absolute_mean_by_stratum=absolute_mean,
            absolute_m2_by_stratum=absolute_m2,
            look_index=self.look_index + 1,
            prior_state_hash=self.state_hash,
        )

    @property
    def is_finite_population_census(self) -> bool:
        return self.stratum_counts == self.population_stratum_counts

    @property
    def census_by_stratum(self) -> tuple[bool, bool]:
        return tuple(
            count == population
            for count, population in zip(
                self.stratum_counts, self.population_stratum_counts
            )
        )  # type: ignore[return-value]

    def _require_class_balanced_point(self) -> None:
        if any(count <= 0 for count in self.stratum_counts):
            raise RuntimeError(
                "both strata require observations before a class-balanced mean exists"
            )

    @property
    def class_balanced_signed_mean(self) -> np.ndarray:
        self._require_class_balanced_point()
        return _readonly_vector(
            0.5 * (self.signed_mean_by_stratum[0] + self.signed_mean_by_stratum[1]),
            name="class-balanced signed mean",
            size=self.route_count,
        )

    @property
    def class_balanced_absolute_mean(self) -> np.ndarray:
        self._require_class_balanced_point()
        return _readonly_vector(
            0.5 * (self.absolute_mean_by_stratum[0] + self.absolute_mean_by_stratum[1]),
            name="class-balanced absolute mean",
            size=self.route_count,
        )

    def class_balanced_descriptive_standard_error(
        self, *, estimand: Literal["signed", "absolute"]
    ) -> np.ndarray:
        """Return the class-balanced plug-in FPC standard error.

        This quantity is descriptive and fixed-state only.  It is deliberately
        not used as an anytime-valid stopping boundary.
        """

        self._require_class_balanced_point()
        if estimand == "signed":
            m2 = self.signed_m2_by_stratum
        elif estimand == "absolute":
            m2 = self.absolute_m2_by_stratum
        else:
            raise ValueError("estimand must be 'signed' or 'absolute'")
        variance = np.zeros(self.route_count, dtype=_LE_F64)
        for stratum, (count, population) in enumerate(
            zip(self.stratum_counts, self.population_stratum_counts)
        ):
            if count == population:
                continue
            if count < 2:
                raise RuntimeError(
                    "a noncensus descriptive variance requires two rows per stratum"
                )
            sample_variance = m2[stratum] / (count - 1)
            variance += 0.25 * (1.0 - count / population) * sample_variance / count
        return _readonly_vector(
            np.sqrt(np.maximum(variance, 0.0), dtype=_LE_F64),
            name=f"class-balanced {estimand} descriptive standard error",
            size=self.route_count,
        )

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "campaign_contract_hash": self.campaign_contract_hash,
            "model_identity_hash": self.model_identity_hash,
            "query_contract_hash": self.query_contract_hash,
            "preprocessing_hash": self.preprocessing_hash,
            "current_mask_hash": self.current_mask_hash,
            "raw_logit_stream_source_hash": self.raw_logit_stream_source_hash,
            "sampling_law_hash": self.sampling_law_hash,
            "proposal_generation": self.proposal_generation,
            "route_count": self.route_count,
            "route_axis_hash": self.route_axis_hash,
            "stratum_names": self.stratum_names,
            "population_stratum_counts": self.population_stratum_counts,
            "sealed_population_order_hashes": self.sealed_population_order_hashes,
            "stratum_counts": self.stratum_counts,
            "observed_context_ids_by_stratum": self.observed_context_ids_by_stratum,
            "observed_draw_ids_by_stratum": self.observed_draw_ids_by_stratum,
            "moment_representation": CORRECTED_ATTRIBUTION_MOMENT_REPRESENTATION,
            "signed_mean_by_stratum": self.signed_mean_by_stratum,
            "signed_m2_by_stratum": self.signed_m2_by_stratum,
            "absolute_mean_by_stratum": self.absolute_mean_by_stratum,
            "absolute_m2_by_stratum": self.absolute_m2_by_stratum,
            "look_index": self.look_index,
            "prior_state_hash": self.prior_state_hash,
        }

    def _compute_state_hash(self) -> str:
        return content_hash(self._hash_payload(), prefix=_MOMENTS_PREFIX)

    @property
    def state_hash(self) -> str:
        cached = self._state_hash
        if cached is None:
            cached = self._compute_state_hash()
            object.__setattr__(self, "_state_hash", cached)
        return cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_state_hash()
        if self._state_hash is not None and observed != self._state_hash:
            raise ValueError(
                "corrected-attribution moments were tampered after hashing"
            )
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("corrected-attribution moments hash does not match")
        object.__setattr__(self, "_state_hash", observed)
        return observed

    def _constructor_payload(self) -> dict[str, Any]:
        return {
            "campaign_contract_hash": self.campaign_contract_hash,
            "model_identity_hash": self.model_identity_hash,
            "query_contract_hash": self.query_contract_hash,
            "preprocessing_hash": self.preprocessing_hash,
            "current_mask_hash": self.current_mask_hash,
            "raw_logit_stream_source_hash": self.raw_logit_stream_source_hash,
            "sampling_law_hash": self.sampling_law_hash,
            "proposal_generation": self.proposal_generation,
            "route_count": self.route_count,
            "route_axis_hash": self.route_axis_hash,
            "stratum_names": self.stratum_names,
            "population_stratum_counts": self.population_stratum_counts,
            "sealed_population_order_hashes": self.sealed_population_order_hashes,
            "stratum_counts": self.stratum_counts,
            "observed_context_ids_by_stratum": self.observed_context_ids_by_stratum,
            "observed_draw_ids_by_stratum": self.observed_draw_ids_by_stratum,
            "signed_mean_by_stratum": self.signed_mean_by_stratum,
            "signed_m2_by_stratum": self.signed_m2_by_stratum,
            "absolute_mean_by_stratum": self.absolute_mean_by_stratum,
            "absolute_m2_by_stratum": self.absolute_m2_by_stratum,
            "look_index": self.look_index,
            "prior_state_hash": self.prior_state_hash,
            "schema": self.schema,
        }

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any], str]]:
        return _rebuild_moments, (self._constructor_payload(), self.assert_integrity())


@dataclass(frozen=True, slots=True)
class V1CorrectedAttributionEnvelopeCertificate:
    """Pre-outcome proof binding ``|-G phi c|`` to routewise envelopes.

    The only accepted derivation is verified analytic bound propagation.  A
    sample maximum, estimated range, winsorization, or clipping threshold is
    not a bound for the original estimand and is rejected by construction.
    """

    campaign_contract_hash: str
    model_identity_hash: str
    query_contract_hash: str
    preprocessing_hash: str
    current_mask_hash: str
    raw_logit_stream_source_hash: str
    sampling_law_hash: str
    proposal_generation: int
    route_count: int
    route_axis_hash: str
    stratum_names: tuple[str, str]
    population_stratum_counts: tuple[int, int]
    sealed_population_order_hashes: tuple[str, str]
    absolute_upper_envelope_by_stratum: np.ndarray
    verifier_source_hash: str
    proof_artifact_hash: str
    derivation_kind: str = ANALYTIC_ENVELOPE_DERIVATION_KIND
    issued_before_outcomes: bool = True
    schema: str = CORRECTED_ATTRIBUTION_ENVELOPE_CERTIFICATE_SCHEMA
    _certificate_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        identities = (
            "campaign_contract_hash",
            "model_identity_hash",
            "query_contract_hash",
            "preprocessing_hash",
            "current_mask_hash",
            "raw_logit_stream_source_hash",
            "sampling_law_hash",
            "route_axis_hash",
            "verifier_source_hash",
            "proof_artifact_hash",
        )
        for name in identities:
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        route_count = _positive_int(self.route_count, name="route count")
        proposal_generation = _nonnegative_int(
            self.proposal_generation, name="proposal generation"
        )
        names = _pair_of_names(self.stratum_names, name="stratum names")
        populations = _pair_of_positive_ints(
            self.population_stratum_counts, name="population stratum counts"
        )
        order_hashes = _pair_of_identities(
            self.sealed_population_order_hashes,
            name="sealed population order hashes",
        )
        if self.derivation_kind != ANALYTIC_ENVELOPE_DERIVATION_KIND:
            raise ValueError(
                "only pre-outcome analytic verified bound propagation can certify "
                "the unmodified corrected-attribution estimand; sample-estimated "
                "ranges and clipping are not accepted"
            )
        if self.issued_before_outcomes is not True:
            raise ValueError("the analytic envelope must be issued before outcomes")
        if self.schema != CORRECTED_ATTRIBUTION_ENVELOPE_CERTIFICATE_SCHEMA:
            raise ValueError("corrected-attribution envelope schema is invalid")
        envelope = _readonly_le_f64(
            self.absolute_upper_envelope_by_stratum,
            name="absolute corrected-attribution upper envelope",
            shape=(2, route_count),
            nonnegative=True,
        )
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "proposal_generation", proposal_generation)
        object.__setattr__(self, "stratum_names", names)
        object.__setattr__(self, "population_stratum_counts", populations)
        object.__setattr__(self, "sealed_population_order_hashes", order_hashes)
        object.__setattr__(self, "absolute_upper_envelope_by_stratum", envelope)

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "campaign_contract_hash": self.campaign_contract_hash,
            "model_identity_hash": self.model_identity_hash,
            "query_contract_hash": self.query_contract_hash,
            "preprocessing_hash": self.preprocessing_hash,
            "current_mask_hash": self.current_mask_hash,
            "raw_logit_stream_source_hash": self.raw_logit_stream_source_hash,
            "sampling_law_hash": self.sampling_law_hash,
            "proposal_generation": self.proposal_generation,
            "route_count": self.route_count,
            "route_axis_hash": self.route_axis_hash,
            "stratum_names": self.stratum_names,
            "population_stratum_counts": self.population_stratum_counts,
            "sealed_population_order_hashes": self.sealed_population_order_hashes,
            "absolute_upper_envelope_by_stratum": (
                self.absolute_upper_envelope_by_stratum
            ),
            "verifier_source_hash": self.verifier_source_hash,
            "proof_artifact_hash": self.proof_artifact_hash,
            "derivation_kind": self.derivation_kind,
            "issued_before_outcomes": self.issued_before_outcomes,
        }

    def _compute_certificate_hash(self) -> str:
        return content_hash(self._hash_payload(), prefix=_ENVELOPE_PREFIX)

    @property
    def certificate_hash(self) -> str:
        cached = self._certificate_hash
        if cached is None:
            cached = self._compute_certificate_hash()
            object.__setattr__(self, "_certificate_hash", cached)
        return cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_certificate_hash()
        if self._certificate_hash is not None and observed != self._certificate_hash:
            raise ValueError(
                "corrected-attribution envelope was tampered after hashing"
            )
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("corrected-attribution envelope hash does not match")
        object.__setattr__(self, "_certificate_hash", observed)
        return observed

    def _constructor_payload(self) -> dict[str, Any]:
        return {
            **self._hash_payload(),
        }

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any], str]]:
        return _rebuild_envelope, (
            self._constructor_payload(),
            self.assert_integrity(),
        )


@dataclass(frozen=True, slots=True)
class V1GroupSequentialSerflingDesign:
    """Predeclared finite-population looks and explicit alpha spending."""

    design_id: str
    campaign_contract_hash: str
    model_identity_hash: str
    query_contract_hash: str
    preprocessing_hash: str
    current_mask_hash: str
    raw_logit_stream_source_hash: str
    proposal_generation: int
    route_count: int
    route_axis_hash: str
    stratum_names: tuple[str, str]
    population_stratum_counts: tuple[int, int]
    sealed_population_order_hashes: tuple[str, str]
    registered_look_stratum_counts: tuple[tuple[int, int], ...]
    alpha_by_look: tuple[float, ...]
    trajectory_alpha_ledger_hash: str
    trajectory_alpha_ledger_generation: int
    trajectory_familywise_alpha: float
    trajectory_alpha_committed_before_generation: float
    generation_alpha_allocation: float
    sampling_law_hash: str
    covered_estimands: tuple[str, ...] = (ABSOLUTE_SALIENCE_ESTIMAND,)
    estimand_weighting: str = EQUAL_STRATUM_MACRO_WEIGHTING
    sampling_law_kind: str = SEALED_WITHOUT_REPLACEMENT_SAMPLING_LAW
    trajectory_alpha_policy_kind: str = FIXED_TRAJECTORY_FWER_POLICY
    schema: str = GROUP_SEQUENTIAL_SERFLING_DESIGN_SCHEMA
    _design_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "design_id", _identity(self.design_id, name="design id")
        )
        for name in (
            "campaign_contract_hash",
            "model_identity_hash",
            "query_contract_hash",
            "preprocessing_hash",
            "current_mask_hash",
            "raw_logit_stream_source_hash",
            "trajectory_alpha_ledger_hash",
            "sampling_law_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        proposal_generation = _nonnegative_int(
            self.proposal_generation, name="proposal generation"
        )
        ledger_generation = _nonnegative_int(
            self.trajectory_alpha_ledger_generation,
            name="trajectory alpha ledger generation",
        )
        if ledger_generation != proposal_generation:
            raise ValueError(
                "trajectory alpha ledger generation must equal proposal generation"
            )
        route_count = _positive_int(self.route_count, name="route count")
        object.__setattr__(
            self,
            "route_axis_hash",
            _identity(self.route_axis_hash, name="route axis hash"),
        )
        names = _pair_of_names(self.stratum_names, name="stratum names")
        populations = _pair_of_positive_ints(
            self.population_stratum_counts, name="population stratum counts"
        )
        order_hashes = _pair_of_identities(
            self.sealed_population_order_hashes,
            name="sealed population order hashes",
        )
        looks = tuple(
            _pair_of_nonnegative_ints(row, name="registered look stratum counts")
            for row in self.registered_look_stratum_counts
        )
        if not looks:
            raise ValueError("a group-sequential design requires registered looks")
        previous = (0, 0)
        for counts in looks:
            if any(count <= 0 for count in counts):
                raise ValueError("every registered look must observe both strata")
            if any(
                count > population for count, population in zip(counts, populations)
            ):
                raise ValueError("registered looks cannot exceed their populations")
            if any(count < prior for count, prior in zip(counts, previous)):
                raise ValueError("registered stratum counts must be nested")
            if counts == previous:
                raise ValueError("each registered look must acquire at least one draw")
            previous = counts

        alpha = tuple(float(value) for value in self.alpha_by_look)
        trajectory_familywise = float(self.trajectory_familywise_alpha)
        committed_before = float(self.trajectory_alpha_committed_before_generation)
        generation_allocation = float(self.generation_alpha_allocation)
        alpha_axis_invalid = any(
            not math.isfinite(value) or value < 0.0 or value >= 1.0
            for value in alpha
        ) or any(
            value == 0.0 and counts != populations
            for value, counts in zip(alpha, looks, strict=True)
        )
        if (
            len(alpha) != len(looks)
            or not math.isfinite(trajectory_familywise)
            or not 0.0 < trajectory_familywise < 1.0
            or not math.isfinite(committed_before)
            or committed_before < 0.0
            or not math.isfinite(generation_allocation)
            or generation_allocation <= 0.0
            or alpha_axis_invalid
        ):
            raise ValueError("group-sequential alpha spending is invalid")
        allocated = math.fsum(alpha)
        tolerance = 32.0 * math.ulp(
            max(
                1.0,
                trajectory_familywise,
                committed_before,
                generation_allocation,
                allocated,
            )
        )
        if allocated > generation_allocation + tolerance:
            raise ValueError("registered look alpha exceeds its generation allocation")
        if committed_before + generation_allocation > trajectory_familywise + tolerance:
            raise ValueError(
                "generation alpha allocation exceeds the trajectory-wide FWER ledger"
            )
        if self.sampling_law_kind != SEALED_WITHOUT_REPLACEMENT_SAMPLING_LAW:
            raise ValueError(
                "Serfling evidence requires sealed uniform random permutations "
                "without replacement within each stratum"
            )
        covered_estimands = tuple(self.covered_estimands)
        if covered_estimands != (ABSOLUTE_SALIENCE_ESTIMAND,):
            raise ValueError(
                "this ordering design grants authority only to corrected absolute "
                "salience; signed covectors require a separately registered design"
            )
        if self.estimand_weighting != EQUAL_STRATUM_MACRO_WEIGHTING:
            raise ValueError(
                "this design requires the registered equal-stratum macro estimand"
            )
        if self.trajectory_alpha_policy_kind != FIXED_TRAJECTORY_FWER_POLICY:
            raise ValueError(
                "this design requires a fixed non-recycling trajectory FWER ledger"
            )
        if self.schema != GROUP_SEQUENTIAL_SERFLING_DESIGN_SCHEMA:
            raise ValueError("group-sequential Serfling design schema is invalid")

        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "proposal_generation", proposal_generation)
        object.__setattr__(
            self, "trajectory_alpha_ledger_generation", ledger_generation
        )
        object.__setattr__(self, "stratum_names", names)
        object.__setattr__(self, "population_stratum_counts", populations)
        object.__setattr__(self, "sealed_population_order_hashes", order_hashes)
        object.__setattr__(self, "registered_look_stratum_counts", looks)
        object.__setattr__(self, "alpha_by_look", alpha)
        object.__setattr__(self, "trajectory_familywise_alpha", trajectory_familywise)
        object.__setattr__(
            self, "trajectory_alpha_committed_before_generation", committed_before
        )
        object.__setattr__(self, "generation_alpha_allocation", generation_allocation)
        object.__setattr__(self, "covered_estimands", covered_estimands)

    @property
    def registered_look_count(self) -> int:
        return len(self.registered_look_stratum_counts)

    @property
    def total_allocated_alpha(self) -> float:
        return math.fsum(self.alpha_by_look)

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "design_id": self.design_id,
            "campaign_contract_hash": self.campaign_contract_hash,
            "model_identity_hash": self.model_identity_hash,
            "query_contract_hash": self.query_contract_hash,
            "preprocessing_hash": self.preprocessing_hash,
            "current_mask_hash": self.current_mask_hash,
            "raw_logit_stream_source_hash": self.raw_logit_stream_source_hash,
            "proposal_generation": self.proposal_generation,
            "route_count": self.route_count,
            "route_axis_hash": self.route_axis_hash,
            "stratum_names": self.stratum_names,
            "population_stratum_counts": self.population_stratum_counts,
            "sealed_population_order_hashes": self.sealed_population_order_hashes,
            "registered_look_stratum_counts": self.registered_look_stratum_counts,
            "alpha_by_look": self.alpha_by_look,
            "trajectory_alpha_ledger_hash": self.trajectory_alpha_ledger_hash,
            "trajectory_alpha_ledger_generation": (
                self.trajectory_alpha_ledger_generation
            ),
            "trajectory_familywise_alpha": self.trajectory_familywise_alpha,
            "trajectory_alpha_committed_before_generation": (
                self.trajectory_alpha_committed_before_generation
            ),
            "generation_alpha_allocation": self.generation_alpha_allocation,
            "sampling_law_hash": self.sampling_law_hash,
            "covered_estimands": self.covered_estimands,
            "estimand_weighting": self.estimand_weighting,
            "sampling_law_kind": self.sampling_law_kind,
            "trajectory_alpha_policy_kind": self.trajectory_alpha_policy_kind,
            "inequality_id": HOEFFDING_SERFLING_INEQUALITY_ID,
        }

    def _compute_design_hash(self) -> str:
        return content_hash(self._hash_payload(), prefix=_DESIGN_PREFIX)

    @property
    def design_hash(self) -> str:
        cached = self._design_hash
        if cached is None:
            cached = self._compute_design_hash()
            object.__setattr__(self, "_design_hash", cached)
        return cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_design_hash()
        if self._design_hash is not None and observed != self._design_hash:
            raise ValueError(
                "group-sequential Serfling design was tampered after hashing"
            )
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("group-sequential Serfling design hash does not match")
        object.__setattr__(self, "_design_hash", observed)
        return observed

    def _constructor_payload(self) -> dict[str, Any]:
        payload = self._hash_payload()
        payload.pop("inequality_id")
        return payload

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any], str]]:
        return _rebuild_design, (
            self._constructor_payload(),
            self.assert_integrity(),
        )


@dataclass(frozen=True, slots=True)
class V1GroupSequentialEmpiricalBernsteinSerflingDesign:
    """V2 confidence design with an independent equal HS/EB alpha split.

    The v1 Serfling design remains the sole owner of the pre-outcome look and
    trajectory-alpha ledger.  This wrapper binds a second, independently
    charged empirical-Bernstein--Serfling method to each already registered
    cell.  Equal splitting is canonical here: neither method may reuse the
    other's failure budget, and no outcome-dependent allocation knob exists.
    """

    serfling_design: V1GroupSequentialSerflingDesign
    bound_method_names: tuple[str, str] = (
        HOEFFDING_SERFLING_INEQUALITY_ID,
        BARDENET_MAILLARD_EMPIRICAL_BERNSTEIN_SERFLING_INEQUALITY_ID,
    )
    bound_method_two_sided_alpha_fractions: tuple[float, float] = (0.5, 0.5)
    deterministic_interval_id: str = DETERMINISTIC_FINITE_POPULATION_INTERVAL_ID
    outward_rounding_policy: str = FLOAT64_OUTWARD_ROUNDING_POLICY
    census_override_id: str = FINITE_POPULATION_CENSUS_OVERRIDE_ID
    inequality_id: str = HYBRID_SERFLING_CONFIDENCE_BOX_INEQUALITY_ID
    schema: str = GROUP_SEQUENTIAL_EMPIRICAL_BERNSTEIN_SERFLING_DESIGN_SCHEMA
    _design_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if type(self.serfling_design) is not V1GroupSequentialSerflingDesign:
            raise TypeError("v2 confidence design requires the exact v1 design type")
        self.serfling_design.assert_integrity()
        methods = tuple(self.bound_method_names)
        if methods != (
            HOEFFDING_SERFLING_INEQUALITY_ID,
            BARDENET_MAILLARD_EMPIRICAL_BERNSTEIN_SERFLING_INEQUALITY_ID,
        ):
            raise ValueError("v2 confidence design bound-method identities are invalid")
        fractions = tuple(
            float(value) for value in self.bound_method_two_sided_alpha_fractions
        )
        if fractions != (0.5, 0.5) or math.fsum(fractions) != 1.0:
            raise ValueError(
                "v2 confidence design requires an independent equal HS/EB alpha split"
            )
        if self.deterministic_interval_id != DETERMINISTIC_FINITE_POPULATION_INTERVAL_ID:
            raise ValueError("v2 deterministic finite-population interval is invalid")
        if self.outward_rounding_policy != FLOAT64_OUTWARD_ROUNDING_POLICY:
            raise ValueError("v2 confidence design outward-rounding policy is invalid")
        if self.census_override_id != FINITE_POPULATION_CENSUS_OVERRIDE_ID:
            raise ValueError("v2 confidence design census override is invalid")
        if self.inequality_id != HYBRID_SERFLING_CONFIDENCE_BOX_INEQUALITY_ID:
            raise ValueError("v2 confidence design inequality identity is invalid")
        if self.schema != GROUP_SEQUENTIAL_EMPIRICAL_BERNSTEIN_SERFLING_DESIGN_SCHEMA:
            raise ValueError("v2 confidence design schema is invalid")
        object.__setattr__(self, "bound_method_names", methods)
        object.__setattr__(
            self,
            "bound_method_two_sided_alpha_fractions",
            fractions,
        )

    @property
    def base_design_hash(self) -> str:
        return self.serfling_design.design_hash

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "base_design_hash": self.serfling_design.assert_integrity(),
            "bound_method_names": self.bound_method_names,
            "bound_method_two_sided_alpha_fractions": (
                self.bound_method_two_sided_alpha_fractions
            ),
            "deterministic_interval_id": self.deterministic_interval_id,
            "outward_rounding_policy": self.outward_rounding_policy,
            "census_override_id": self.census_override_id,
            "inequality_id": self.inequality_id,
            "empirical_variance_convention": (
                EMPIRICAL_BERNSTEIN_BIASED_VARIANCE_CONVENTION
            ),
            "empirical_bernstein_kappa": EMPIRICAL_BERNSTEIN_KAPPA,
        }

    def _compute_design_hash(self) -> str:
        return content_hash(self._hash_payload(), prefix=_HYBRID_DESIGN_PREFIX)

    @property
    def design_hash(self) -> str:
        cached = self._design_hash
        if cached is None:
            cached = self._compute_design_hash()
            object.__setattr__(self, "_design_hash", cached)
        return cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.serfling_design.assert_integrity()
        observed = self._compute_design_hash()
        if self._design_hash is not None and observed != self._design_hash:
            raise ValueError("v2 confidence design was tampered after hashing")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("v2 confidence design hash does not match")
        object.__setattr__(self, "_design_hash", observed)
        return observed

    def __reduce__(self) -> tuple[Any, tuple[Any, str, str]]:
        return _rebuild_hybrid_design, (
            self.serfling_design,
            self.schema,
            self.assert_integrity(),
        )


@dataclass(frozen=True, slots=True)
class V1CorrectedAttributionConfidenceBoxReceipt:
    """Typed simultaneous confidence box for one registered live look."""

    state_hash: str
    envelope_certificate_hash: str
    design_hash: str
    campaign_contract_hash: str
    model_identity_hash: str
    query_contract_hash: str
    preprocessing_hash: str
    current_mask_hash: str
    raw_logit_stream_source_hash: str
    sampling_law_hash: str
    proposal_generation: int
    route_count: int
    route_axis_hash: str
    stratum_names: tuple[str, str]
    stratum_counts: tuple[int, int]
    population_stratum_counts: tuple[int, int]
    sealed_population_order_hashes: tuple[str, str]
    observed_context_ids_by_stratum: tuple[tuple[str, ...], tuple[str, ...]]
    observed_draw_ids_by_stratum: tuple[tuple[str, ...], tuple[str, ...]]
    class_balanced_stratum_weights: tuple[float, float]
    look_index: int
    absolute_radius_multipliers_by_stratum: tuple[float, float]
    class_balanced_absolute_point: np.ndarray
    class_balanced_absolute_lower: np.ndarray
    class_balanced_absolute_upper: np.ndarray
    trajectory_alpha_ledger_hash: str
    trajectory_alpha_ledger_generation: int
    trajectory_familywise_alpha: float
    trajectory_alpha_committed_before_generation: float
    generation_alpha_allocation: float
    registered_look_stratum_counts: tuple[tuple[int, int], ...]
    alpha_by_look: tuple[float, ...]
    total_registered_alpha: float
    look_alpha: float
    cumulative_alpha_spent: float
    remaining_generation_alpha_after_look: float
    unallocated_generation_alpha_reserve: float
    remaining_trajectory_alpha_after_generation_allocation: float
    per_cell_two_sided_alpha: float
    per_cell_one_sided_tail_alpha: float
    serfling_log_factor: float
    finite_population_corrections: tuple[float, float]
    covered_estimand_count: int
    covered_estimands: tuple[str, ...]
    estimand_weighting: str
    covered_stratum_count: int
    covered_route_count: int
    covered_two_sided_statements_at_look: int
    registered_look_count: int
    covered_two_sided_statement_looks: int
    sampling_law_kind: str
    trajectory_alpha_policy_kind: str
    inequality_id: str = HOEFFDING_SERFLING_INEQUALITY_ID
    evidence_ceiling: str = SERFLING_EVIDENCE_CEILING
    schema: str = CORRECTED_ATTRIBUTION_CONFIDENCE_BOX_SCHEMA
    _receipt_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "state_hash",
            "envelope_certificate_hash",
            "design_hash",
            "campaign_contract_hash",
            "model_identity_hash",
            "query_contract_hash",
            "preprocessing_hash",
            "current_mask_hash",
            "raw_logit_stream_source_hash",
            "sampling_law_hash",
            "trajectory_alpha_ledger_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        proposal_generation = _nonnegative_int(
            self.proposal_generation, name="proposal generation"
        )
        ledger_generation = _nonnegative_int(
            self.trajectory_alpha_ledger_generation,
            name="trajectory alpha ledger generation",
        )
        if ledger_generation != proposal_generation:
            raise ValueError(
                "receipt trajectory ledger generation differs from proposal generation"
            )
        route_count = _positive_int(self.route_count, name="route count")
        object.__setattr__(
            self,
            "route_axis_hash",
            _identity(self.route_axis_hash, name="route axis hash"),
        )
        names = _pair_of_names(self.stratum_names, name="stratum names")
        counts = _pair_of_nonnegative_ints(self.stratum_counts, name="stratum counts")
        populations = _pair_of_positive_ints(
            self.population_stratum_counts, name="population stratum counts"
        )
        if any(count <= 0 for count in counts):
            raise ValueError("a class-balanced confidence box must observe both strata")
        order_hashes = _pair_of_identities(
            self.sealed_population_order_hashes,
            name="sealed population order hashes",
        )
        contexts = _nested_identity_pair(
            self.observed_context_ids_by_stratum,
            name="observed context ledgers",
        )
        draws = _nested_identity_pair(
            self.observed_draw_ids_by_stratum,
            name="observed draw ledgers",
        )
        if any(
            len(contexts[stratum]) != counts[stratum]
            or len(draws[stratum]) != counts[stratum]
            for stratum in (0, 1)
        ):
            raise ValueError("confidence receipt sampled ledgers are incomplete")
        if len(set(contexts[0] + contexts[1])) != sum(counts) or len(
            set(draws[0] + draws[1])
        ) != sum(counts):
            raise ValueError("confidence receipt sampled ledgers contain duplicates")
        weights = tuple(float(value) for value in self.class_balanced_stratum_weights)
        if weights != (0.5, 0.5):
            raise ValueError("corrected attribution uses fixed class-balanced weights")
        if self.estimand_weighting != EQUAL_STRATUM_MACRO_WEIGHTING:
            raise ValueError("confidence receipt estimand weighting is invalid")
        look_index = _positive_int(self.look_index, name="look index")
        vectors: dict[str, np.ndarray] = {}
        for name in (
            "class_balanced_absolute_point",
            "class_balanced_absolute_lower",
            "class_balanced_absolute_upper",
        ):
            nonnegative = "absolute" in name
            vectors[name] = _readonly_le_f64(
                getattr(self, name),
                name=name,
                shape=(route_count,),
                nonnegative=nonnegative,
            )
        if np.any(
            vectors["class_balanced_absolute_lower"]
            > vectors["class_balanced_absolute_point"]
        ) or np.any(
            vectors["class_balanced_absolute_point"]
            > vectors["class_balanced_absolute_upper"]
        ):
            raise ValueError("class-balanced bounds must contain their point means")

        alpha_names = (
            "trajectory_familywise_alpha",
            "generation_alpha_allocation",
            "total_registered_alpha",
            "look_alpha",
            "cumulative_alpha_spent",
            "per_cell_two_sided_alpha",
            "per_cell_one_sided_tail_alpha",
        )
        alphas = {name: float(getattr(self, name)) for name in alpha_names}
        committed_before = float(self.trajectory_alpha_committed_before_generation)
        registered_counts = tuple(
            _pair_of_nonnegative_ints(row, name="registered look stratum counts")
            for row in self.registered_look_stratum_counts
        )
        alpha_by_look = tuple(float(value) for value in self.alpha_by_look)
        prior_registered = (0, 0)
        registered_counts_valid = True
        for registered in registered_counts:
            if (
                any(value <= 0 for value in registered)
                or any(
                    value > population
                    for value, population in zip(registered, populations)
                )
                or any(
                    value < prior for value, prior in zip(registered, prior_registered)
                )
                or registered == prior_registered
            ):
                registered_counts_valid = False
                break
            prior_registered = registered
        alpha_axis_invalid = any(
            not math.isfinite(value) or value < 0.0 or value >= 1.0
            for value in alpha_by_look
        ) or any(
            value == 0.0 and registered != populations
            for value, registered in zip(
                alpha_by_look, registered_counts, strict=True
            )
        )
        if (
            not registered_counts
            or not registered_counts_valid
            or len(alpha_by_look) != len(registered_counts)
            or look_index > len(registered_counts)
            or counts != registered_counts[look_index - 1]
            or alpha_axis_invalid
        ):
            raise ValueError("confidence receipt registered look ledger is invalid")
        remaining_generation = float(self.remaining_generation_alpha_after_look)
        unallocated_generation = float(self.unallocated_generation_alpha_reserve)
        remaining_trajectory = float(
            self.remaining_trajectory_alpha_after_generation_allocation
        )
        log_factor = float(self.serfling_log_factor)
        joint_census_zero_alpha = counts == populations and alphas["look_alpha"] == 0.0
        always_positive_alpha_names = (
            "trajectory_familywise_alpha",
            "generation_alpha_allocation",
        )
        nonnegative_alpha_names = (
            "total_registered_alpha",
            "look_alpha",
            "cumulative_alpha_spent",
            "per_cell_two_sided_alpha",
            "per_cell_one_sided_tail_alpha",
        )
        if any(
            not math.isfinite(alphas[name]) or alphas[name] <= 0.0
            for name in always_positive_alpha_names
        ) or any(
            not math.isfinite(alphas[name]) or alphas[name] < 0.0
            for name in nonnegative_alpha_names
        ):
            raise ValueError("confidence receipt alpha terms are invalid")
        if joint_census_zero_alpha:
            if any(
                alphas[name] != 0.0
                for name in (
                    "look_alpha",
                    "per_cell_two_sided_alpha",
                    "per_cell_one_sided_tail_alpha",
                )
            ):
                raise ValueError("a zero-alpha census look must be deterministic")
        elif any(
            alphas[name] <= 0.0
            for name in (
                "look_alpha",
                "per_cell_two_sided_alpha",
                "per_cell_one_sided_tail_alpha",
            )
        ):
            raise ValueError("a non-census confidence look requires positive alpha")
        if (
            not 0.0 < alphas["trajectory_familywise_alpha"] < 1.0
            or not math.isfinite(committed_before)
            or committed_before < 0.0
            or not math.isfinite(remaining_generation)
            or remaining_generation < 0.0
            or not math.isfinite(unallocated_generation)
            or unallocated_generation < 0.0
            or not math.isfinite(remaining_trajectory)
            or remaining_trajectory < 0.0
            or not math.isfinite(log_factor)
            or (log_factor != 0.0 if joint_census_zero_alpha else log_factor <= 0.0)
        ):
            raise ValueError("confidence receipt spending remainder is invalid")
        if self.sampling_law_kind != SEALED_WITHOUT_REPLACEMENT_SAMPLING_LAW:
            raise ValueError("confidence receipt sampling-law kind is invalid")
        if self.trajectory_alpha_policy_kind != FIXED_TRAJECTORY_FWER_POLICY:
            raise ValueError("confidence receipt trajectory-alpha policy is invalid")
        expected_cells = 2 * route_count
        covered_estimands = tuple(self.covered_estimands)
        if (
            _positive_int(self.covered_estimand_count, name="covered estimand count")
            != 1
            or covered_estimands != (ABSOLUTE_SALIENCE_ESTIMAND,)
            or _positive_int(self.covered_stratum_count, name="covered stratum count")
            != 2
            or _positive_int(self.covered_route_count, name="covered route count")
            != route_count
            or _positive_int(
                self.covered_two_sided_statements_at_look,
                name="covered statements at look",
            )
            != expected_cells
        ):
            raise ValueError("confidence receipt coverage ledger is invalid")
        registered_looks = _positive_int(
            self.registered_look_count, name="registered look count"
        )
        if (
            registered_looks != len(registered_counts)
            or not math.isclose(
                alphas["total_registered_alpha"],
                math.fsum(alpha_by_look),
                rel_tol=4e-15,
                abs_tol=0.0,
            )
            or not math.isclose(
                alphas["look_alpha"],
                alpha_by_look[look_index - 1],
                rel_tol=4e-15,
                abs_tol=0.0,
            )
            or not math.isclose(
                alphas["cumulative_alpha_spent"],
                math.fsum(alpha_by_look[:look_index]),
                rel_tol=4e-15,
                abs_tol=0.0,
            )
            or alphas["total_registered_alpha"] > alphas["generation_alpha_allocation"]
            or committed_before + alphas["generation_alpha_allocation"]
            > alphas["trajectory_familywise_alpha"]
            or not math.isclose(
                remaining_generation,
                alphas["generation_alpha_allocation"]
                - alphas["cumulative_alpha_spent"],
                rel_tol=4e-15,
                abs_tol=4e-18,
            )
            or not math.isclose(
                unallocated_generation,
                alphas["generation_alpha_allocation"]
                - alphas["total_registered_alpha"],
                rel_tol=4e-15,
                abs_tol=4e-18,
            )
            or not math.isclose(
                remaining_trajectory,
                alphas["trajectory_familywise_alpha"]
                - committed_before
                - alphas["generation_alpha_allocation"],
                rel_tol=4e-15,
                abs_tol=4e-18,
            )
            or _positive_int(
                self.covered_two_sided_statement_looks,
                name="covered statement looks",
            )
            != expected_cells * registered_looks
            or not math.isclose(
                alphas["per_cell_two_sided_alpha"],
                alphas["look_alpha"] / expected_cells,
                rel_tol=4e-15,
                abs_tol=0.0,
            )
            or not math.isclose(
                alphas["per_cell_one_sided_tail_alpha"],
                0.5 * alphas["per_cell_two_sided_alpha"],
                rel_tol=4e-15,
                abs_tol=0.0,
            )
            or not math.isclose(
                log_factor,
                (
                    0.0
                    if joint_census_zero_alpha
                    else math.log(2.0 / alphas["per_cell_two_sided_alpha"])
                ),
                rel_tol=4e-15,
                abs_tol=0.0,
            )
        ):
            raise ValueError("confidence receipt union-bound accounting is invalid")
        corrections = tuple(
            float(value) for value in self.finite_population_corrections
        )
        if len(corrections) != 2 or any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in corrections
        ):
            raise ValueError("Serfling finite-population corrections are invalid")
        radius_multipliers = tuple(
            float(value) for value in self.absolute_radius_multipliers_by_stratum
        )
        if len(radius_multipliers) != 2 or any(
            not math.isfinite(value) or value < 0.0 for value in radius_multipliers
        ):
            raise ValueError("Serfling absolute radius multipliers are invalid")
        for stratum, (count, population) in enumerate(zip(counts, populations)):
            if count == population:
                expected_correction = 0.0
                expected_multiplier = 0.0
            else:
                expected_correction = 1.0 - (count - 1.0) / population
                expected_multiplier = math.sqrt(
                    expected_correction * log_factor / (2.0 * count)
                )
            if not math.isclose(
                corrections[stratum],
                expected_correction,
                rel_tol=4e-15,
                abs_tol=0.0,
            ) or not math.isclose(
                radius_multipliers[stratum],
                expected_multiplier,
                rel_tol=4e-15,
                abs_tol=0.0,
            ):
                raise ValueError(
                    "Serfling correction and radius must replay exactly from "
                    "the population, look count, and alpha ledger"
                )
        if self.inequality_id != HOEFFDING_SERFLING_INEQUALITY_ID:
            raise ValueError("confidence receipt inequality identity is invalid")
        if self.evidence_ceiling != SERFLING_EVIDENCE_CEILING:
            raise ValueError("confidence receipt evidence ceiling is invalid")
        if self.schema != CORRECTED_ATTRIBUTION_CONFIDENCE_BOX_SCHEMA:
            raise ValueError("corrected-attribution confidence-box schema is invalid")

        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "proposal_generation", proposal_generation)
        object.__setattr__(
            self, "trajectory_alpha_ledger_generation", ledger_generation
        )
        object.__setattr__(self, "stratum_names", names)
        object.__setattr__(self, "stratum_counts", counts)
        object.__setattr__(self, "population_stratum_counts", populations)
        object.__setattr__(self, "sealed_population_order_hashes", order_hashes)
        object.__setattr__(self, "observed_context_ids_by_stratum", contexts)
        object.__setattr__(self, "observed_draw_ids_by_stratum", draws)
        object.__setattr__(self, "class_balanced_stratum_weights", weights)
        object.__setattr__(self, "look_index", look_index)
        object.__setattr__(self, "registered_look_stratum_counts", registered_counts)
        object.__setattr__(self, "alpha_by_look", alpha_by_look)
        object.__setattr__(self, "finite_population_corrections", corrections)
        object.__setattr__(
            self,
            "absolute_radius_multipliers_by_stratum",
            radius_multipliers,
        )
        object.__setattr__(self, "covered_estimands", covered_estimands)
        object.__setattr__(
            self,
            "remaining_generation_alpha_after_look",
            remaining_generation,
        )
        object.__setattr__(
            self,
            "unallocated_generation_alpha_reserve",
            unallocated_generation,
        )
        object.__setattr__(
            self,
            "remaining_trajectory_alpha_after_generation_allocation",
            remaining_trajectory,
        )
        object.__setattr__(
            self, "trajectory_alpha_committed_before_generation", committed_before
        )
        object.__setattr__(self, "serfling_log_factor", log_factor)
        for name, value in vectors.items():
            object.__setattr__(self, name, value)
        for name, value in alphas.items():
            object.__setattr__(self, name, value)

    @property
    def boxes(self) -> Mapping[str, Mapping[str, np.ndarray]]:
        return MappingProxyType(
            {
                "absolute": MappingProxyType(
                    {
                        "point": self.class_balanced_absolute_point,
                        "lower": self.class_balanced_absolute_lower,
                        "upper": self.class_balanced_absolute_upper,
                    }
                ),
            }
        )

    def _hash_payload(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "state_hash",
                "envelope_certificate_hash",
                "design_hash",
                "campaign_contract_hash",
                "model_identity_hash",
                "query_contract_hash",
                "preprocessing_hash",
                "current_mask_hash",
                "raw_logit_stream_source_hash",
                "sampling_law_hash",
                "proposal_generation",
                "route_count",
                "route_axis_hash",
                "stratum_names",
                "stratum_counts",
                "population_stratum_counts",
                "sealed_population_order_hashes",
                "observed_context_ids_by_stratum",
                "observed_draw_ids_by_stratum",
                "class_balanced_stratum_weights",
                "look_index",
                "absolute_radius_multipliers_by_stratum",
                "class_balanced_absolute_point",
                "class_balanced_absolute_lower",
                "class_balanced_absolute_upper",
                "trajectory_alpha_ledger_hash",
                "trajectory_alpha_ledger_generation",
                "trajectory_familywise_alpha",
                "trajectory_alpha_committed_before_generation",
                "generation_alpha_allocation",
                "registered_look_stratum_counts",
                "alpha_by_look",
                "total_registered_alpha",
                "look_alpha",
                "cumulative_alpha_spent",
                "remaining_generation_alpha_after_look",
                "unallocated_generation_alpha_reserve",
                "remaining_trajectory_alpha_after_generation_allocation",
                "per_cell_two_sided_alpha",
                "per_cell_one_sided_tail_alpha",
                "serfling_log_factor",
                "finite_population_corrections",
                "covered_estimand_count",
                "covered_estimands",
                "estimand_weighting",
                "covered_stratum_count",
                "covered_route_count",
                "covered_two_sided_statements_at_look",
                "registered_look_count",
                "covered_two_sided_statement_looks",
                "sampling_law_kind",
                "trajectory_alpha_policy_kind",
                "inequality_id",
                "evidence_ceiling",
            )
        }

    def _compute_receipt_hash(self) -> str:
        return content_hash(self._hash_payload(), prefix=_BOX_PREFIX)

    @property
    def receipt_hash(self) -> str:
        cached = self._receipt_hash
        if cached is None:
            cached = self._compute_receipt_hash()
            object.__setattr__(self, "_receipt_hash", cached)
        return cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_receipt_hash()
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("corrected-attribution confidence box was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("corrected-attribution confidence-box hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    def _constructor_payload(self) -> dict[str, Any]:
        return self._hash_payload()

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any], str]]:
        return _rebuild_box, (
            self._constructor_payload(),
            self.assert_integrity(),
        )


@dataclass(frozen=True, slots=True)
class V1CorrectedAttributionConfidenceBoxV2Receipt:
    """Hybrid HS/EB finite-population box with a hash-bound running intersection.

    ``class_balanced_absolute_point`` is the current-look estimator.  It must
    lie in the current box, but it need not lie in the cumulative box inherited
    from earlier looks.  The cumulative box alone is action-facing; an empty
    intersection is a refusal and cannot be represented by this receipt.
    """

    state_hash: str
    envelope_certificate_hash: str
    design_hash: str
    base_serfling_design_hash: str
    campaign_contract_hash: str
    model_identity_hash: str
    query_contract_hash: str
    preprocessing_hash: str
    current_mask_hash: str
    raw_logit_stream_source_hash: str
    sampling_law_hash: str
    proposal_generation: int
    route_count: int
    route_axis_hash: str
    stratum_names: tuple[str, str]
    stratum_counts: tuple[int, int]
    population_stratum_counts: tuple[int, int]
    sealed_population_order_hashes: tuple[str, str]
    observed_context_ids_by_stratum: tuple[tuple[str, ...], tuple[str, ...]]
    observed_draw_ids_by_stratum: tuple[tuple[str, ...], tuple[str, ...]]
    class_balanced_stratum_weights: tuple[float, float]
    look_index: int
    prior_confidence_box_receipt_hash: str | None
    prior_statistics_state_hash: str | None
    class_balanced_absolute_point: np.ndarray
    current_class_balanced_absolute_lower: np.ndarray
    current_class_balanced_absolute_upper: np.ndarray
    cumulative_class_balanced_absolute_lower: np.ndarray
    cumulative_class_balanced_absolute_upper: np.ndarray
    trajectory_alpha_ledger_hash: str
    trajectory_alpha_ledger_generation: int
    trajectory_familywise_alpha: float
    trajectory_alpha_committed_before_generation: float
    generation_alpha_allocation: float
    registered_look_stratum_counts: tuple[tuple[int, int], ...]
    alpha_by_look: tuple[float, ...]
    total_registered_alpha: float
    look_alpha: float
    cumulative_alpha_spent: float
    remaining_generation_alpha_after_look: float
    unallocated_generation_alpha_reserve: float
    remaining_trajectory_alpha_after_generation_allocation: float
    per_cell_combined_two_sided_alpha: float
    per_cell_hoeffding_serfling_two_sided_alpha: float
    per_cell_empirical_bernstein_two_sided_alpha: float
    empirical_bernstein_theorem_delta: float
    hoeffding_serfling_log_factor: float
    empirical_bernstein_log_factor: float
    hoeffding_serfling_finite_population_corrections: tuple[float, float]
    hoeffding_serfling_radius_multipliers_by_stratum: tuple[float, float]
    empirical_bernstein_rho_by_stratum: tuple[float, float]
    empirical_bernstein_rho_branches_by_stratum: tuple[str, str]
    empirical_bernstein_sigma_radius_multipliers_by_stratum: tuple[float, float]
    empirical_bernstein_range_radius_multipliers_by_stratum: tuple[float, float]
    census_by_stratum: tuple[bool, bool]
    covered_estimand_count: int
    covered_estimands: tuple[str, ...]
    estimand_weighting: str
    covered_stratum_count: int
    covered_route_count: int
    covered_two_sided_statements_at_look: int
    registered_look_count: int
    covered_two_sided_statement_looks: int
    sampling_law_kind: str
    trajectory_alpha_policy_kind: str
    bound_method_names: tuple[str, str] = (
        HOEFFDING_SERFLING_INEQUALITY_ID,
        BARDENET_MAILLARD_EMPIRICAL_BERNSTEIN_SERFLING_INEQUALITY_ID,
    )
    bound_method_two_sided_alpha_fractions: tuple[float, float] = (0.5, 0.5)
    empirical_variance_convention: str = (
        EMPIRICAL_BERNSTEIN_BIASED_VARIANCE_CONVENTION
    )
    empirical_bernstein_kappa: float = EMPIRICAL_BERNSTEIN_KAPPA
    deterministic_interval_id: str = DETERMINISTIC_FINITE_POPULATION_INTERVAL_ID
    outward_rounding_policy: str = FLOAT64_OUTWARD_ROUNDING_POLICY
    census_override_id: str = FINITE_POPULATION_CENSUS_OVERRIDE_ID
    cumulative_intersection_nonempty: bool = True
    inequality_id: str = HYBRID_SERFLING_CONFIDENCE_BOX_INEQUALITY_ID
    evidence_ceiling: str = HYBRID_SERFLING_EVIDENCE_CEILING
    schema: str = CORRECTED_ATTRIBUTION_CONFIDENCE_BOX_V2_SCHEMA
    _receipt_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "state_hash",
            "envelope_certificate_hash",
            "design_hash",
            "base_serfling_design_hash",
            "campaign_contract_hash",
            "model_identity_hash",
            "query_contract_hash",
            "preprocessing_hash",
            "current_mask_hash",
            "raw_logit_stream_source_hash",
            "sampling_law_hash",
            "trajectory_alpha_ledger_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        prior_hash = _optional_identity(
            self.prior_confidence_box_receipt_hash,
            name="prior confidence-box receipt hash",
        )
        prior_state_hash = _optional_identity(
            self.prior_statistics_state_hash,
            name="prior statistics state hash",
        )
        generation = _nonnegative_int(
            self.proposal_generation, name="proposal generation"
        )
        ledger_generation = _nonnegative_int(
            self.trajectory_alpha_ledger_generation,
            name="trajectory alpha ledger generation",
        )
        if ledger_generation != generation:
            raise ValueError(
                "v2 receipt trajectory ledger generation differs from proposal generation"
            )
        route_count = _positive_int(self.route_count, name="route count")
        object.__setattr__(
            self,
            "route_axis_hash",
            _identity(self.route_axis_hash, name="route axis hash"),
        )
        names = _pair_of_names(self.stratum_names, name="stratum names")
        counts = _pair_of_nonnegative_ints(self.stratum_counts, name="stratum counts")
        populations = _pair_of_positive_ints(
            self.population_stratum_counts, name="population stratum counts"
        )
        if any(count <= 0 or count > population for count, population in zip(counts, populations)):
            raise ValueError("a v2 class-balanced box requires live in-population strata")
        order_hashes = _pair_of_identities(
            self.sealed_population_order_hashes,
            name="sealed population order hashes",
        )
        contexts = _nested_identity_pair(
            self.observed_context_ids_by_stratum,
            name="observed context ledgers",
        )
        draws = _nested_identity_pair(
            self.observed_draw_ids_by_stratum,
            name="observed draw ledgers",
        )
        if any(
            len(contexts[stratum]) != counts[stratum]
            or len(draws[stratum]) != counts[stratum]
            for stratum in (0, 1)
        ):
            raise ValueError("v2 confidence receipt sampled ledgers are incomplete")
        if len(set(contexts[0] + contexts[1])) != sum(counts) or len(
            set(draws[0] + draws[1])
        ) != sum(counts):
            raise ValueError("v2 confidence receipt sampled ledgers contain duplicates")
        weights = tuple(float(value) for value in self.class_balanced_stratum_weights)
        if weights != (0.5, 0.5):
            raise ValueError("corrected attribution uses fixed class-balanced weights")
        if self.estimand_weighting != EQUAL_STRATUM_MACRO_WEIGHTING:
            raise ValueError("v2 confidence receipt estimand weighting is invalid")

        look_index = _positive_int(self.look_index, name="look index")
        if look_index == 1:
            if prior_hash is not None or prior_state_hash is not None:
                raise ValueError("the first v2 look cannot bind prior inference state")
        elif prior_hash is None or prior_state_hash is None:
            raise ValueError(
                "every later v2 look must bind the immediate prior receipt and moments"
            )

        vectors: dict[str, np.ndarray] = {}
        for name in (
            "class_balanced_absolute_point",
            "current_class_balanced_absolute_lower",
            "current_class_balanced_absolute_upper",
            "cumulative_class_balanced_absolute_lower",
            "cumulative_class_balanced_absolute_upper",
        ):
            vectors[name] = _readonly_le_f64(
                getattr(self, name),
                name=name,
                shape=(route_count,),
                nonnegative=True,
            )
        point = vectors["class_balanced_absolute_point"]
        current_lower = vectors["current_class_balanced_absolute_lower"]
        current_upper = vectors["current_class_balanced_absolute_upper"]
        cumulative_lower = vectors["cumulative_class_balanced_absolute_lower"]
        cumulative_upper = vectors["cumulative_class_balanced_absolute_upper"]
        if np.any(current_lower > point) or np.any(point > current_upper):
            raise ValueError("the current v2 box must contain its current-look point")
        if np.any(current_lower > current_upper):
            raise ValueError("the current v2 confidence box is empty")
        if self.cumulative_intersection_nonempty is not True or np.any(
            cumulative_lower > cumulative_upper
        ):
            raise ValueError("an empty cumulative confidence intersection must refuse")
        if np.any(cumulative_lower < current_lower) or np.any(
            cumulative_upper > current_upper
        ):
            raise ValueError("the cumulative v2 box must be a subset of the current box")
        if look_index == 1 and (
            not np.array_equal(cumulative_lower, current_lower)
            or not np.array_equal(cumulative_upper, current_upper)
        ):
            raise ValueError("the first v2 cumulative box must equal its current box")

        registered_counts = tuple(
            _pair_of_nonnegative_ints(row, name="registered look stratum counts")
            for row in self.registered_look_stratum_counts
        )
        alpha_by_look = tuple(float(value) for value in self.alpha_by_look)
        previous = (0, 0)
        for registered in registered_counts:
            if (
                any(value <= 0 for value in registered)
                or any(
                    value > population
                    for value, population in zip(registered, populations)
                )
                or any(value < prior for value, prior in zip(registered, previous))
                or registered == previous
            ):
                raise ValueError("v2 registered look ledger is invalid")
            previous = registered
        alpha_axis_invalid = any(
            not math.isfinite(value) or value < 0.0 or value >= 1.0
            for value in alpha_by_look
        ) or any(
            value == 0.0 and registered != populations
            for value, registered in zip(
                alpha_by_look, registered_counts, strict=True
            )
        )
        if (
            not registered_counts
            or len(alpha_by_look) != len(registered_counts)
            or look_index > len(registered_counts)
            or counts != registered_counts[look_index - 1]
            or alpha_axis_invalid
        ):
            raise ValueError("v2 registered look and alpha ledgers are inconsistent")

        scalar_alpha_names = (
            "trajectory_familywise_alpha",
            "generation_alpha_allocation",
            "total_registered_alpha",
            "look_alpha",
            "cumulative_alpha_spent",
            "per_cell_combined_two_sided_alpha",
            "per_cell_hoeffding_serfling_two_sided_alpha",
            "per_cell_empirical_bernstein_two_sided_alpha",
            "empirical_bernstein_theorem_delta",
        )
        alphas = {name: float(getattr(self, name)) for name in scalar_alpha_names}
        joint_census_zero_alpha = counts == populations and alphas["look_alpha"] == 0.0
        always_positive_alpha_names = (
            "trajectory_familywise_alpha",
            "generation_alpha_allocation",
        )
        nonnegative_alpha_names = tuple(
            name for name in scalar_alpha_names if name not in always_positive_alpha_names
        )
        if any(
            not math.isfinite(alphas[name]) or alphas[name] <= 0.0
            for name in always_positive_alpha_names
        ) or any(
            not math.isfinite(alphas[name]) or alphas[name] < 0.0
            for name in nonnegative_alpha_names
        ):
            raise ValueError("v2 alpha terms are invalid")
        current_inferential_names = (
            "look_alpha",
            "per_cell_combined_two_sided_alpha",
            "per_cell_hoeffding_serfling_two_sided_alpha",
            "per_cell_empirical_bernstein_two_sided_alpha",
            "empirical_bernstein_theorem_delta",
        )
        if joint_census_zero_alpha:
            if any(alphas[name] != 0.0 for name in current_inferential_names):
                raise ValueError("a zero-alpha v2 census look must be deterministic")
        elif any(alphas[name] <= 0.0 for name in current_inferential_names):
            raise ValueError("a non-census v2 confidence look requires positive alpha")
        committed_before = float(self.trajectory_alpha_committed_before_generation)
        remaining_generation = float(self.remaining_generation_alpha_after_look)
        unallocated_generation = float(self.unallocated_generation_alpha_reserve)
        remaining_trajectory = float(
            self.remaining_trajectory_alpha_after_generation_allocation
        )
        if (
            not 0.0 < alphas["trajectory_familywise_alpha"] < 1.0
            or not math.isfinite(committed_before)
            or committed_before < 0.0
            or any(
                not math.isfinite(value) or value < 0.0
                for value in (
                    remaining_generation,
                    unallocated_generation,
                    remaining_trajectory,
                )
            )
        ):
            raise ValueError("v2 trajectory-alpha remainder is invalid")
        expected_cells = 2 * route_count
        registered_look_count = _positive_int(
            self.registered_look_count, name="registered look count"
        )
        coverage_valid = (
            _positive_int(self.covered_estimand_count, name="covered estimand count")
            == 1
            and tuple(self.covered_estimands) == (ABSOLUTE_SALIENCE_ESTIMAND,)
            and _positive_int(self.covered_stratum_count, name="covered stratum count")
            == 2
            and _positive_int(self.covered_route_count, name="covered route count")
            == route_count
            and _positive_int(
                self.covered_two_sided_statements_at_look,
                name="covered statements at look",
            )
            == expected_cells
            and registered_look_count == len(registered_counts)
            and _positive_int(
                self.covered_two_sided_statement_looks,
                name="covered statement looks",
            )
            == expected_cells * registered_look_count
        )
        if not coverage_valid:
            raise ValueError("v2 confidence receipt coverage ledger is invalid")

        tolerance = 4e-15
        expected_look_alpha = alpha_by_look[look_index - 1]
        expected_total_alpha = math.fsum(alpha_by_look)
        expected_cumulative_alpha = math.fsum(alpha_by_look[:look_index])
        expected_cell_alpha = expected_look_alpha / expected_cells
        expected_method_alpha = 0.5 * expected_cell_alpha
        expected_delta = expected_method_alpha / 10.0
        expected_remaining_generation = max(
            0.0, alphas["generation_alpha_allocation"] - expected_cumulative_alpha
        )
        expected_unallocated_generation = max(
            0.0, alphas["generation_alpha_allocation"] - expected_total_alpha
        )
        expected_remaining_trajectory = max(
            0.0,
            alphas["trajectory_familywise_alpha"]
            - committed_before
            - alphas["generation_alpha_allocation"],
        )
        scalar_equalities = (
            (alphas["total_registered_alpha"], expected_total_alpha),
            (alphas["look_alpha"], expected_look_alpha),
            (alphas["cumulative_alpha_spent"], expected_cumulative_alpha),
            (alphas["per_cell_combined_two_sided_alpha"], expected_cell_alpha),
            (
                alphas["per_cell_hoeffding_serfling_two_sided_alpha"],
                expected_method_alpha,
            ),
            (
                alphas["per_cell_empirical_bernstein_two_sided_alpha"],
                expected_method_alpha,
            ),
            (alphas["empirical_bernstein_theorem_delta"], expected_delta),
            (remaining_generation, expected_remaining_generation),
            (unallocated_generation, expected_unallocated_generation),
            (remaining_trajectory, expected_remaining_trajectory),
        )
        if (
            any(
                not math.isclose(left, right, rel_tol=tolerance, abs_tol=4e-18)
                for left, right in scalar_equalities
            )
            or expected_total_alpha > alphas["generation_alpha_allocation"]
            or committed_before + alphas["generation_alpha_allocation"]
            > alphas["trajectory_familywise_alpha"]
        ):
            raise ValueError("v2 confidence receipt alpha accounting is invalid")

        methods = tuple(self.bound_method_names)
        fractions = tuple(
            float(value) for value in self.bound_method_two_sided_alpha_fractions
        )
        if methods != (
            HOEFFDING_SERFLING_INEQUALITY_ID,
            BARDENET_MAILLARD_EMPIRICAL_BERNSTEIN_SERFLING_INEQUALITY_ID,
        ) or fractions != (0.5, 0.5):
            raise ValueError("v2 confidence receipt method split is invalid")
        hs_log = float(self.hoeffding_serfling_log_factor)
        eb_log = float(self.empirical_bernstein_log_factor)
        expected_hs_log = (
            0.0
            if joint_census_zero_alpha
            else _nextafter_up(
                math.log(
                    2.0
                    / alphas["per_cell_hoeffding_serfling_two_sided_alpha"]
                )
            )
        )
        expected_eb_log = (
            0.0
            if joint_census_zero_alpha
            else _nextafter_up(
                math.log(1.0 / alphas["empirical_bernstein_theorem_delta"])
            )
        )
        if (
            not math.isfinite(hs_log)
            or not math.isfinite(eb_log)
            or not math.isclose(
                hs_log,
                expected_hs_log,
                rel_tol=tolerance,
                abs_tol=0.0,
            )
            or not math.isclose(
                eb_log,
                expected_eb_log,
                rel_tol=tolerance,
                abs_tol=0.0,
            )
        ):
            raise ValueError("v2 confidence receipt logarithmic factors are invalid")

        hs_corrections = tuple(
            float(value)
            for value in self.hoeffding_serfling_finite_population_corrections
        )
        hs_multipliers = tuple(
            float(value)
            for value in self.hoeffding_serfling_radius_multipliers_by_stratum
        )
        eb_rhos = tuple(float(value) for value in self.empirical_bernstein_rho_by_stratum)
        eb_branches = tuple(self.empirical_bernstein_rho_branches_by_stratum)
        eb_sigma_multipliers = tuple(
            float(value)
            for value in self.empirical_bernstein_sigma_radius_multipliers_by_stratum
        )
        eb_range_multipliers = tuple(
            float(value)
            for value in self.empirical_bernstein_range_radius_multipliers_by_stratum
        )
        census = tuple(self.census_by_stratum)
        if not all(
            len(value) == 2
            for value in (
                hs_corrections,
                hs_multipliers,
                eb_rhos,
                eb_branches,
                eb_sigma_multipliers,
                eb_range_multipliers,
                census,
            )
        ) or any(type(value) is not bool for value in census):
            raise ValueError("v2 stratum theorem ledger is malformed")
        for stratum, (count, population) in enumerate(zip(counts, populations)):
            is_census = count == population
            if census[stratum] is not is_census:
                raise ValueError("v2 census ledger disagrees with stratum counts")
            if is_census:
                expected_hs_correction = 0.0
                expected_hs_multiplier = 0.0
                expected_rho = 0.0
                expected_sigma_multiplier = 0.0
                expected_range_multiplier = 0.0
                expected_branch = (
                    EMPIRICAL_BERNSTEIN_RHO_BRANCH_LE_HALF
                    if count <= population / 2.0
                    else EMPIRICAL_BERNSTEIN_RHO_BRANCH_GT_HALF
                )
            else:
                (
                    expected_hs_correction,
                    expected_hs_multiplier,
                ) = _hs_correction_and_multiplier_up(
                    count=count,
                    population=population,
                    log_factor=hs_log,
                )
                (
                    expected_rho,
                    expected_branch,
                    expected_sigma_multiplier,
                    expected_range_multiplier,
                ) = _eb_rho_and_multipliers_up(
                    count=count,
                    population=population,
                    log_factor=eb_log,
                )
            theorem_scalars = (
                (hs_corrections[stratum], expected_hs_correction),
                (hs_multipliers[stratum], expected_hs_multiplier),
                (eb_rhos[stratum], expected_rho),
                (eb_sigma_multipliers[stratum], expected_sigma_multiplier),
                (eb_range_multipliers[stratum], expected_range_multiplier),
            )
            if eb_branches[stratum] != expected_branch or any(
                not math.isfinite(left)
                or left < 0.0
                or not math.isclose(left, right, rel_tol=tolerance, abs_tol=0.0)
                for left, right in theorem_scalars
            ):
                raise ValueError("v2 empirical-Bernstein theorem ledger is invalid")

        if self.empirical_variance_convention != EMPIRICAL_BERNSTEIN_BIASED_VARIANCE_CONVENTION:
            raise ValueError("v2 empirical-variance convention is invalid")
        kappa = float(self.empirical_bernstein_kappa)
        if not math.isclose(kappa, EMPIRICAL_BERNSTEIN_KAPPA, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("v2 empirical-Bernstein kappa is invalid")
        if self.deterministic_interval_id != DETERMINISTIC_FINITE_POPULATION_INTERVAL_ID:
            raise ValueError("v2 deterministic interval identity is invalid")
        if self.outward_rounding_policy != FLOAT64_OUTWARD_ROUNDING_POLICY:
            raise ValueError("v2 outward-rounding policy is invalid")
        if self.census_override_id != FINITE_POPULATION_CENSUS_OVERRIDE_ID:
            raise ValueError("v2 census override identity is invalid")
        if self.sampling_law_kind != SEALED_WITHOUT_REPLACEMENT_SAMPLING_LAW:
            raise ValueError("v2 sampling-law kind is invalid")
        if self.trajectory_alpha_policy_kind != FIXED_TRAJECTORY_FWER_POLICY:
            raise ValueError("v2 trajectory-alpha policy is invalid")
        if self.inequality_id != HYBRID_SERFLING_CONFIDENCE_BOX_INEQUALITY_ID:
            raise ValueError("v2 inequality identity is invalid")
        if self.evidence_ceiling != HYBRID_SERFLING_EVIDENCE_CEILING:
            raise ValueError("v2 evidence ceiling is invalid")
        if self.schema != CORRECTED_ATTRIBUTION_CONFIDENCE_BOX_V2_SCHEMA:
            raise ValueError("v2 confidence-box schema is invalid")

        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(
            self, "trajectory_alpha_ledger_generation", ledger_generation
        )
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "stratum_names", names)
        object.__setattr__(self, "stratum_counts", counts)
        object.__setattr__(self, "population_stratum_counts", populations)
        object.__setattr__(self, "sealed_population_order_hashes", order_hashes)
        object.__setattr__(self, "observed_context_ids_by_stratum", contexts)
        object.__setattr__(self, "observed_draw_ids_by_stratum", draws)
        object.__setattr__(self, "class_balanced_stratum_weights", weights)
        object.__setattr__(self, "look_index", look_index)
        object.__setattr__(self, "prior_confidence_box_receipt_hash", prior_hash)
        object.__setattr__(self, "prior_statistics_state_hash", prior_state_hash)
        object.__setattr__(self, "registered_look_stratum_counts", registered_counts)
        object.__setattr__(self, "alpha_by_look", alpha_by_look)
        object.__setattr__(self, "covered_estimands", tuple(self.covered_estimands))
        object.__setattr__(self, "bound_method_names", methods)
        object.__setattr__(
            self, "bound_method_two_sided_alpha_fractions", fractions
        )
        object.__setattr__(self, "census_by_stratum", census)
        object.__setattr__(
            self, "empirical_bernstein_rho_branches_by_stratum", eb_branches
        )
        object.__setattr__(
            self,
            "hoeffding_serfling_finite_population_corrections",
            hs_corrections,
        )
        object.__setattr__(
            self,
            "hoeffding_serfling_radius_multipliers_by_stratum",
            hs_multipliers,
        )
        object.__setattr__(self, "empirical_bernstein_rho_by_stratum", eb_rhos)
        object.__setattr__(
            self,
            "empirical_bernstein_sigma_radius_multipliers_by_stratum",
            eb_sigma_multipliers,
        )
        object.__setattr__(
            self,
            "empirical_bernstein_range_radius_multipliers_by_stratum",
            eb_range_multipliers,
        )
        object.__setattr__(self, "empirical_bernstein_kappa", kappa)
        object.__setattr__(self, "trajectory_alpha_committed_before_generation", committed_before)
        object.__setattr__(self, "remaining_generation_alpha_after_look", remaining_generation)
        object.__setattr__(self, "unallocated_generation_alpha_reserve", unallocated_generation)
        object.__setattr__(
            self,
            "remaining_trajectory_alpha_after_generation_allocation",
            remaining_trajectory,
        )
        object.__setattr__(self, "hoeffding_serfling_log_factor", hs_log)
        object.__setattr__(self, "empirical_bernstein_log_factor", eb_log)
        for name, value in vectors.items():
            object.__setattr__(self, name, value)
        for name, value in alphas.items():
            object.__setattr__(self, name, value)

    @property
    def class_balanced_absolute_lower(self) -> np.ndarray:
        """Action-facing lower bound: the cumulative running intersection."""

        return self.cumulative_class_balanced_absolute_lower

    @property
    def class_balanced_absolute_upper(self) -> np.ndarray:
        """Action-facing upper bound: the cumulative running intersection."""

        return self.cumulative_class_balanced_absolute_upper

    @property
    def boxes(self) -> Mapping[str, Mapping[str, np.ndarray]]:
        return MappingProxyType(
            {
                "absolute": MappingProxyType(
                    {
                        "point": self.class_balanced_absolute_point,
                        "lower": self.class_balanced_absolute_lower,
                        "upper": self.class_balanced_absolute_upper,
                        "current_lower": self.current_class_balanced_absolute_lower,
                        "current_upper": self.current_class_balanced_absolute_upper,
                    }
                )
            }
        )

    def assert_prior_transition(
        self,
        prior: V1CorrectedAttributionConfidenceBoxV2Receipt | None,
        *,
        expected_prior_hash: str | None = None,
    ) -> None:
        """Replay the exact running-intersection transition against its prior."""

        if self.look_index == 1:
            if prior is not None or expected_prior_hash is not None:
                raise ValueError("the first v2 box has no prior transition")
            return
        if type(prior) is not V1CorrectedAttributionConfidenceBoxV2Receipt:
            raise TypeError("v2 transition replay requires the exact prior receipt")
        prior_hash = prior.assert_integrity(expected_prior_hash)
        if prior_hash != self.prior_confidence_box_receipt_hash:
            raise ValueError("v2 transition prior receipt hash does not match")
        if prior.state_hash != self.prior_statistics_state_hash:
            raise ValueError("v2 transition prior moments hash does not match")
        if prior.look_index != self.look_index - 1:
            raise ValueError("v2 transition does not bind the immediately prior look")
        for name in (
            "envelope_certificate_hash",
            "design_hash",
            "base_serfling_design_hash",
            "campaign_contract_hash",
            "model_identity_hash",
            "query_contract_hash",
            "preprocessing_hash",
            "current_mask_hash",
            "raw_logit_stream_source_hash",
            "sampling_law_hash",
            "proposal_generation",
            "route_count",
            "route_axis_hash",
            "stratum_names",
            "population_stratum_counts",
            "sealed_population_order_hashes",
            "trajectory_alpha_ledger_hash",
            "trajectory_alpha_ledger_generation",
        ):
            if getattr(prior, name) != getattr(self, name):
                raise ValueError(f"v2 transition disagrees with its prior on {name}")
        if any(
            current < previous
            for current, previous in zip(self.stratum_counts, prior.stratum_counts)
        ) or self.stratum_counts == prior.stratum_counts:
            raise ValueError("v2 transition sample counts are not strictly nested")
        for stratum in (0, 1):
            prior_count = prior.stratum_counts[stratum]
            if (
                self.observed_context_ids_by_stratum[stratum][:prior_count]
                != prior.observed_context_ids_by_stratum[stratum]
                or self.observed_draw_ids_by_stratum[stratum][:prior_count]
                != prior.observed_draw_ids_by_stratum[stratum]
            ):
                raise ValueError("v2 transition sample ledgers are not nested prefixes")
        expected_lower = np.maximum(
            self.current_class_balanced_absolute_lower,
            prior.cumulative_class_balanced_absolute_lower,
        )
        expected_upper = np.minimum(
            self.current_class_balanced_absolute_upper,
            prior.cumulative_class_balanced_absolute_upper,
        )
        if np.any(expected_lower > expected_upper):
            raise ValueError("v2 transition running intersection is empty")
        if not np.array_equal(
            expected_lower, self.cumulative_class_balanced_absolute_lower
        ) or not np.array_equal(
            expected_upper, self.cumulative_class_balanced_absolute_upper
        ):
            raise ValueError("v2 transition did not preserve the exact intersection")

    def _hash_payload(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "state_hash",
                "envelope_certificate_hash",
                "design_hash",
                "base_serfling_design_hash",
                "campaign_contract_hash",
                "model_identity_hash",
                "query_contract_hash",
                "preprocessing_hash",
                "current_mask_hash",
                "raw_logit_stream_source_hash",
                "sampling_law_hash",
                "proposal_generation",
                "route_count",
                "route_axis_hash",
                "stratum_names",
                "stratum_counts",
                "population_stratum_counts",
                "sealed_population_order_hashes",
                "observed_context_ids_by_stratum",
                "observed_draw_ids_by_stratum",
                "class_balanced_stratum_weights",
                "look_index",
                "prior_confidence_box_receipt_hash",
                "prior_statistics_state_hash",
                "class_balanced_absolute_point",
                "current_class_balanced_absolute_lower",
                "current_class_balanced_absolute_upper",
                "cumulative_class_balanced_absolute_lower",
                "cumulative_class_balanced_absolute_upper",
                "trajectory_alpha_ledger_hash",
                "trajectory_alpha_ledger_generation",
                "trajectory_familywise_alpha",
                "trajectory_alpha_committed_before_generation",
                "generation_alpha_allocation",
                "registered_look_stratum_counts",
                "alpha_by_look",
                "total_registered_alpha",
                "look_alpha",
                "cumulative_alpha_spent",
                "remaining_generation_alpha_after_look",
                "unallocated_generation_alpha_reserve",
                "remaining_trajectory_alpha_after_generation_allocation",
                "per_cell_combined_two_sided_alpha",
                "per_cell_hoeffding_serfling_two_sided_alpha",
                "per_cell_empirical_bernstein_two_sided_alpha",
                "empirical_bernstein_theorem_delta",
                "hoeffding_serfling_log_factor",
                "empirical_bernstein_log_factor",
                "hoeffding_serfling_finite_population_corrections",
                "hoeffding_serfling_radius_multipliers_by_stratum",
                "empirical_bernstein_rho_by_stratum",
                "empirical_bernstein_rho_branches_by_stratum",
                "empirical_bernstein_sigma_radius_multipliers_by_stratum",
                "empirical_bernstein_range_radius_multipliers_by_stratum",
                "census_by_stratum",
                "covered_estimand_count",
                "covered_estimands",
                "estimand_weighting",
                "covered_stratum_count",
                "covered_route_count",
                "covered_two_sided_statements_at_look",
                "registered_look_count",
                "covered_two_sided_statement_looks",
                "sampling_law_kind",
                "trajectory_alpha_policy_kind",
                "bound_method_names",
                "bound_method_two_sided_alpha_fractions",
                "empirical_variance_convention",
                "empirical_bernstein_kappa",
                "deterministic_interval_id",
                "outward_rounding_policy",
                "census_override_id",
                "cumulative_intersection_nonempty",
                "inequality_id",
                "evidence_ceiling",
            )
        }

    def _compute_receipt_hash(self) -> str:
        return content_hash(self._hash_payload(), prefix=_BOX_V2_PREFIX)

    @property
    def receipt_hash(self) -> str:
        cached = self._receipt_hash
        if cached is None:
            cached = self._compute_receipt_hash()
            object.__setattr__(self, "_receipt_hash", cached)
        return cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_receipt_hash()
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("v2 corrected-attribution confidence box was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("v2 confidence-box hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any], str]]:
        return _rebuild_box_v2, (self._hash_payload(), self.assert_integrity())


def _deterministic_bottom_k_on_physical_axis(
    costs: np.ndarray,
    *,
    count: int,
    physical_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Return the exact additive minimizer with a physical-index tie rule.

    A threshold selection avoids a full sort of the route axis.  All routes
    strictly below the ``k``-th cost are retained; only the required number of
    equal-threshold routes are chosen, in ascending physical-index order.
    The returned set itself is sorted because action identity is set valued.
    """

    vector = np.asarray(costs, dtype=_LE_F64).reshape(-1)
    if not np.isfinite(vector).all():
        raise ValueError("action-family additive costs must be finite")
    if physical_indices is None:
        candidate_count = int(vector.size)
        candidate_costs = vector
    else:
        physical = np.asarray(physical_indices, dtype=_LE_U64).reshape(-1)
        if physical.size and int(np.max(physical)) >= int(vector.size):
            raise ValueError("action-family member lies off the cost route axis")
        candidate_count = int(physical.size)
        candidate_costs = vector[physical]
    k = _nonnegative_int(count, name="action-family selection count")
    if k > candidate_count:
        raise ValueError("action-family selection count exceeds its candidate axis")
    if k == 0:
        return _readonly_le_u64_vector((), name="minimum action", size=0)
    if k == candidate_count:
        if physical_indices is None:
            selected = np.arange(candidate_count, dtype=_LE_U64)
        else:
            selected = np.array(physical, dtype=_LE_U64, copy=True)
            selected.sort(kind="stable")
        return _readonly_le_u64_vector(
            selected, name="minimum action", size=candidate_count
        )

    # ``np.partition`` is value-exact: unlike an approximate top-k kernel it
    # does not weaken the combinatorial certificate.  We never rely on its
    # unspecified ordering among equal values.
    threshold = float(np.partition(candidate_costs, k - 1)[k - 1])
    strict_positions = np.flatnonzero(candidate_costs < threshold)
    remaining = k - int(strict_positions.size)
    tie_positions = np.flatnonzero(candidate_costs == threshold)
    if remaining < 0 or remaining > int(tie_positions.size):
        raise RuntimeError("bottom-k threshold failed to bracket the exact action")
    if physical_indices is None:
        strict_physical = strict_positions.astype(_LE_U64, copy=False)
        tie_physical = tie_positions.astype(_LE_U64, copy=False)
    else:
        strict_physical = physical[strict_positions]
        tie_physical = physical[tie_positions]
    if remaining:
        if physical_indices is None:
            chosen_ties = tie_physical[:remaining]
        elif int(tie_physical.size) == remaining:
            chosen_ties = tie_physical
        else:
            chosen_ties = np.partition(tie_physical, remaining - 1)[:remaining]
            chosen_ties.sort(kind="stable")
        selected = np.concatenate((strict_physical, chosen_ties))
    else:
        selected = np.array(strict_physical, dtype=_LE_U64, copy=True)
    selected.sort(kind="stable")
    return _readonly_le_u64_vector(selected, name="minimum action", size=k)


def deterministic_bottom_k_action(
    costs: np.ndarray,
    *,
    count: int,
    physical_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Return the canonical additive bottom-k action on a physical route axis.

    This is the public composition boundary used by preregistered proposal
    inference.  It deliberately exposes the same exact threshold selection and
    ascending-physical-index tie rule used by the action-regret theorem, so a
    caller never has to depend on the theorem owner's private helper.
    """

    return _deterministic_bottom_k_on_physical_axis(
        costs,
        count=count,
        physical_indices=physical_indices,
    )


def _same_float64_multiset(left: np.ndarray, right: np.ndarray) -> bool:
    """Whether two fp64 value ledgers are exactly the same finite multiset."""

    left_values = np.asarray(left, dtype=_LE_F64).reshape(-1)
    right_values = np.asarray(right, dtype=_LE_F64).reshape(-1)
    if left_values.shape != right_values.shape:
        return False
    if left_values.size == 0 or np.array_equal(left_values, right_values):
        return True
    left_sorted = np.sort(left_values, kind="stable")
    right_sorted = np.sort(right_values, kind="stable")
    return bool(np.array_equal(left_sorted, right_sorted))


@dataclass(frozen=True, slots=True)
class V1CorrectedAttributionActionFamily:
    """Pre-outcome registration of one additive proposal action family.

    ``partition_route_indices`` is canonical CSR membership: each partition's
    physical indices occupy one segment delimited by ``partition_offsets``.
    The segments jointly form an exact permutation of the route axis.  Empty
    metadata is mandatory for the global fixed-cardinality family.
    """

    family_id: str
    registration_artifact_hash: str
    registered_design_hash: str
    campaign_contract_hash: str
    model_identity_hash: str
    query_contract_hash: str
    preprocessing_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_count: int
    route_axis_hash: str
    family_kind: ActionFamilyKind
    action_size: int
    maximum_action_regret: float
    partition_names: tuple[str, ...] = ()
    partition_offsets: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=_LE_U64),
        repr=False,
        compare=False,
    )
    partition_route_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=_LE_U64),
        repr=False,
        compare=False,
    )
    partition_exact_quotas: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=_LE_U64),
        repr=False,
        compare=False,
    )
    registered_before_outcomes: bool = True
    proposal_discovery_only: bool = True
    nonlinear_geometric_selector_authority: bool = False
    mask_commit_authority: bool = False
    schema: str = CORRECTED_ATTRIBUTION_ACTION_FAMILY_SCHEMA
    _family_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "family_id",
            "registration_artifact_hash",
            "registered_design_hash",
            "campaign_contract_hash",
            "model_identity_hash",
            "query_contract_hash",
            "preprocessing_hash",
            "current_mask_hash",
            "route_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        generation = _nonnegative_int(
            self.proposal_generation, name="proposal generation"
        )
        route_count = _positive_int(self.route_count, name="route count")
        if route_count > int(np.iinfo(np.int64).max):
            raise ValueError("route count exceeds the supported physical index axis")
        action_size = _positive_int(self.action_size, name="action size")
        if action_size > route_count:
            raise ValueError("action size cannot exceed the registered route axis")
        tolerance = _finite_nonnegative_float(
            self.maximum_action_regret, name="maximum action regret"
        )
        if self.family_kind not in (
            GLOBAL_FIXED_CARDINALITY_ACTION_FAMILY,
            EXACT_QUOTA_PARTITION_MATROID_ACTION_FAMILY,
        ):
            raise ValueError("corrected-attribution action family kind is invalid")
        if (
            self.registered_before_outcomes is not True
            or self.proposal_discovery_only is not True
            or self.nonlinear_geometric_selector_authority is not False
            or self.mask_commit_authority is not False
        ):
            raise PermissionError(
                "corrected-attribution action families are pre-outcome, "
                "proposal-only registrations without geometric or mask authority"
            )
        if self.schema != CORRECTED_ATTRIBUTION_ACTION_FAMILY_SCHEMA:
            raise ValueError("corrected-attribution action-family schema is invalid")

        names = tuple(
            _identity(item, name="partition name") for item in self.partition_names
        )
        if len(set(names)) != len(names):
            raise ValueError("partition names must be unique")
        offsets = _readonly_le_u64_vector(
            self.partition_offsets, name="partition offsets"
        )
        members = _readonly_le_u64_vector(
            self.partition_route_indices, name="partition route indices"
        )
        quotas = _readonly_le_u64_vector(
            self.partition_exact_quotas, name="partition exact quotas"
        )
        if self.family_kind == GLOBAL_FIXED_CARDINALITY_ACTION_FAMILY:
            if names or offsets.size or members.size or quotas.size:
                raise ValueError(
                    "a global fixed-cardinality family cannot carry partition metadata"
                )
        else:
            partition_count = len(names)
            if not partition_count:
                raise ValueError("an exact-quota family requires named partitions")
            if (
                int(offsets.size) != partition_count + 1
                or int(quotas.size) != partition_count
                or int(members.size) != route_count
                or int(offsets[0]) != 0
                or int(offsets[-1]) != route_count
                or bool(np.any(offsets[1:] <= offsets[:-1]))
            ):
                raise ValueError(
                    "partition CSR must be nonempty and exactly cover the route axis"
                )
            canonical_members = np.array(members, dtype=_LE_U64, copy=True)
            for partition in range(partition_count):
                start = int(offsets[partition])
                stop = int(offsets[partition + 1])
                canonical_members[start:stop].sort(kind="stable")
                if int(quotas[partition]) > stop - start:
                    raise ValueError("a partition quota exceeds its partition size")
            if canonical_members.size and int(np.max(canonical_members)) >= route_count:
                raise ValueError("partition membership lies off the route axis")
            seen = np.zeros(route_count, dtype=np.bool_)
            seen[canonical_members] = True
            if int(np.count_nonzero(seen)) != route_count:
                raise ValueError(
                    "partition membership must be a duplicate-free route-axis permutation"
                )
            if int(np.sum(quotas, dtype=np.uint64)) != action_size:
                raise ValueError("partition exact quotas must sum to the action size")
            members = _readonly_le_u64_vector(
                canonical_members,
                name="partition route indices",
                size=route_count,
            )

        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "action_size", action_size)
        object.__setattr__(self, "maximum_action_regret", tolerance)
        object.__setattr__(self, "partition_names", names)
        object.__setattr__(self, "partition_offsets", offsets)
        object.__setattr__(self, "partition_route_indices", members)
        object.__setattr__(self, "partition_exact_quotas", quotas)
        self.assert_integrity()

    @classmethod
    def global_fixed_cardinality(
        cls,
        *,
        family_id: str,
        registration_artifact_hash: str,
        registered_design_hash: str,
        campaign_contract_hash: str,
        model_identity_hash: str,
        query_contract_hash: str,
        preprocessing_hash: str,
        current_mask_hash: str,
        proposal_generation: int,
        route_count: int,
        route_axis_hash: str,
        action_size: int,
        maximum_action_regret: float,
    ) -> "V1CorrectedAttributionActionFamily":
        return cls(
            family_id=family_id,
            registration_artifact_hash=registration_artifact_hash,
            registered_design_hash=registered_design_hash,
            campaign_contract_hash=campaign_contract_hash,
            model_identity_hash=model_identity_hash,
            query_contract_hash=query_contract_hash,
            preprocessing_hash=preprocessing_hash,
            current_mask_hash=current_mask_hash,
            proposal_generation=proposal_generation,
            route_count=route_count,
            route_axis_hash=route_axis_hash,
            family_kind=GLOBAL_FIXED_CARDINALITY_ACTION_FAMILY,
            action_size=action_size,
            maximum_action_regret=maximum_action_regret,
        )

    @classmethod
    def exact_quota_partition_matroid(
        cls,
        *,
        family_id: str,
        registration_artifact_hash: str,
        registered_design_hash: str,
        campaign_contract_hash: str,
        model_identity_hash: str,
        query_contract_hash: str,
        preprocessing_hash: str,
        current_mask_hash: str,
        proposal_generation: int,
        route_count: int,
        route_axis_hash: str,
        partition_names: Sequence[str],
        partition_route_indices: Sequence[Sequence[int] | np.ndarray],
        partition_exact_quotas: Sequence[int] | np.ndarray,
        maximum_action_regret: float,
    ) -> "V1CorrectedAttributionActionFamily":
        groups = tuple(
            _readonly_le_u64_vector(group, name="partition route indices")
            for group in partition_route_indices
        )
        names = tuple(partition_names)
        if len(groups) != len(names):
            raise ValueError("partition names and route groups must be aligned")
        offsets = np.zeros(len(groups) + 1, dtype=_LE_U64)
        for index, group in enumerate(groups):
            offsets[index + 1] = offsets[index] + int(group.size)
        members = (
            np.concatenate(groups).astype(_LE_U64, copy=False)
            if groups
            else np.empty(0, dtype=_LE_U64)
        )
        quotas = _readonly_le_u64_vector(
            partition_exact_quotas,
            name="partition exact quotas",
            size=len(groups),
        )
        return cls(
            family_id=family_id,
            registration_artifact_hash=registration_artifact_hash,
            registered_design_hash=registered_design_hash,
            campaign_contract_hash=campaign_contract_hash,
            model_identity_hash=model_identity_hash,
            query_contract_hash=query_contract_hash,
            preprocessing_hash=preprocessing_hash,
            current_mask_hash=current_mask_hash,
            proposal_generation=proposal_generation,
            route_count=route_count,
            route_axis_hash=route_axis_hash,
            family_kind=EXACT_QUOTA_PARTITION_MATROID_ACTION_FAMILY,
            action_size=int(np.sum(quotas, dtype=np.uint64)),
            maximum_action_regret=maximum_action_regret,
            partition_names=names,
            partition_offsets=offsets,
            partition_route_indices=members,
            partition_exact_quotas=quotas,
        )

    def _hash_payload(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "family_id",
                "registration_artifact_hash",
                "registered_design_hash",
                "campaign_contract_hash",
                "model_identity_hash",
                "query_contract_hash",
                "preprocessing_hash",
                "current_mask_hash",
                "proposal_generation",
                "route_count",
                "route_axis_hash",
                "family_kind",
                "action_size",
                "maximum_action_regret",
                "partition_names",
                "partition_offsets",
                "partition_route_indices",
                "partition_exact_quotas",
                "registered_before_outcomes",
                "proposal_discovery_only",
                "nonlinear_geometric_selector_authority",
                "mask_commit_authority",
            )
        }

    def _compute_family_hash(self) -> str:
        return content_hash(self._hash_payload(), prefix=_ACTION_FAMILY_PREFIX)

    @property
    def family_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_family_hash()
        if self._family_hash is not None and observed != self._family_hash:
            raise ValueError("corrected-attribution action family was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("corrected-attribution action-family hash does not match")
        object.__setattr__(self, "_family_hash", observed)
        return observed

    def canonical_action(self, route_indices: Sequence[int] | np.ndarray) -> np.ndarray:
        """Validate and return one canonical feasible action in this family."""

        self.assert_integrity()
        action = _canonical_route_set(
            route_indices,
            route_count=self.route_count,
            expected_count=self.action_size,
            name="proposed route indices",
        )
        if self.family_kind == EXACT_QUOTA_PARTITION_MATROID_ACTION_FAMILY:
            flags = np.zeros(self.route_count, dtype=np.bool_)
            flags[action] = True
            selected_members = flags[self.partition_route_indices]
            counts = np.add.reduceat(
                selected_members.astype(np.uint64, copy=False),
                self.partition_offsets[:-1].astype(np.intp, copy=False),
                dtype=np.uint64,
            )
            if not np.array_equal(counts, self.partition_exact_quotas):
                raise ValueError(
                    "proposed action violates the registered exact partition quotas"
                )
        return action

    def _minimum_additive_action(self, costs: np.ndarray) -> np.ndarray:
        vector = np.asarray(costs, dtype=_LE_F64).reshape(-1)
        if vector.shape != (self.route_count,) or not np.isfinite(vector).all():
            raise ValueError("adversarial costs must be finite on the route axis")
        if self.family_kind == GLOBAL_FIXED_CARDINALITY_ACTION_FAMILY:
            return _deterministic_bottom_k_on_physical_axis(
                vector, count=self.action_size
            )
        selections: list[np.ndarray] = []
        for partition in range(len(self.partition_names)):
            start = int(self.partition_offsets[partition])
            stop = int(self.partition_offsets[partition + 1])
            selections.append(
                _deterministic_bottom_k_on_physical_axis(
                    vector,
                    count=int(self.partition_exact_quotas[partition]),
                    physical_indices=self.partition_route_indices[start:stop],
                )
            )
        result = np.concatenate(selections).astype(_LE_U64, copy=False)
        result.sort(kind="stable")
        return _readonly_le_u64_vector(
            result, name="minimum partition action", size=self.action_size
        )

    def require_nonlinear_geometric_selector_authority(self) -> None:
        raise PermissionError(
            "additive confidence-box regret does not authorize the nonlinear "
            "KKT/geometric selector"
        )

    def require_mask_commit_authority(self) -> None:
        raise PermissionError(
            "sampled corrected-attribution discovery does not authorize a mask commit"
        )

    def _constructor_payload(self) -> dict[str, Any]:
        return self._hash_payload()

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any], str]]:
        return _rebuild_action_family, (
            self._constructor_payload(),
            self.assert_integrity(),
        )


@dataclass(frozen=True, slots=True)
class V1CorrectedAttributionActionRegretRequest:
    """Content-addressed binding of a live box to one registered proposal."""

    confidence_box: (
        V1CorrectedAttributionConfidenceBoxReceipt
        | V1CorrectedAttributionConfidenceBoxV2Receipt
    ) = field(repr=False, compare=False)
    action_family: V1CorrectedAttributionActionFamily = field(repr=False, compare=False)
    proposed_route_indices: np.ndarray = field(repr=False, compare=False)
    proposal_source_hash: str
    proposal_discovery_only: bool = True
    nonlinear_geometric_selector_authority: bool = False
    mask_commit_authority: bool = False
    schema: str | None = None
    _proposal_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )
    _request_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        box_type = type(self.confidence_box)
        if box_type not in (
            V1CorrectedAttributionConfidenceBoxReceipt,
            V1CorrectedAttributionConfidenceBoxV2Receipt,
        ):
            raise TypeError(
                "action-regret request requires an exact registered v1 or v2 "
                "confidence-box type"
            )
        if type(self.action_family) is not V1CorrectedAttributionActionFamily:
            raise TypeError(
                "action-regret request requires the exact action-family type"
            )
        self.confidence_box.assert_integrity()
        self.action_family.assert_integrity()
        self._assert_coordinate_agreement()
        action = self.action_family.canonical_action(self.proposed_route_indices)
        if box_type is V1CorrectedAttributionConfidenceBoxV2Receipt:
            expected_action = self.action_family._minimum_additive_action(
                self.confidence_box.class_balanced_absolute_point
            )
            if not np.array_equal(action, expected_action):
                raise ValueError(
                    "a v2 action proposal must be the deterministic additive "
                    "minimum of its current-look point"
                )
        object.__setattr__(self, "proposed_route_indices", action)
        object.__setattr__(
            self,
            "proposal_source_hash",
            _identity(self.proposal_source_hash, name="proposal source hash"),
        )
        if (
            self.proposal_discovery_only is not True
            or self.nonlinear_geometric_selector_authority is not False
            or self.mask_commit_authority is not False
        ):
            raise PermissionError(
                "an action-regret request is proposal-only and has no geometric "
                "or mask authority"
            )
        expected_schema = (
            CORRECTED_ATTRIBUTION_ACTION_REGRET_REQUEST_SCHEMA
            if box_type is V1CorrectedAttributionConfidenceBoxReceipt
            else CORRECTED_ATTRIBUTION_ACTION_REGRET_REQUEST_V2_SCHEMA
        )
        schema = expected_schema if self.schema is None else self.schema
        if schema != expected_schema:
            raise ValueError(
                "action-regret request schema does not match its exact "
                "confidence-box type"
            )
        object.__setattr__(self, "schema", schema)
        self.assert_integrity()

    @classmethod
    def from_deterministic_point_composition(
        cls,
        *,
        confidence_box: (
            V1CorrectedAttributionConfidenceBoxReceipt
            | V1CorrectedAttributionConfidenceBoxV2Receipt
        ),
        action_family: V1CorrectedAttributionActionFamily,
        proposal_source_hash: str,
    ) -> "V1CorrectedAttributionActionRegretRequest":
        """Compose the additive proposal from the current-look point only.

        For a v2 box this intentionally ignores both current and cumulative
        confidence endpoints during composition.  Those cumulative endpoints
        enter only the subsequent robust-regret certificate.
        """

        if type(confidence_box) not in (
            V1CorrectedAttributionConfidenceBoxReceipt,
            V1CorrectedAttributionConfidenceBoxV2Receipt,
        ):
            raise TypeError("deterministic composition requires an exact box type")
        if type(action_family) is not V1CorrectedAttributionActionFamily:
            raise TypeError("deterministic composition requires the exact family type")
        proposed = action_family._minimum_additive_action(
            confidence_box.class_balanced_absolute_point
        )
        return cls(
            confidence_box=confidence_box,
            action_family=action_family,
            proposed_route_indices=proposed,
            proposal_source_hash=proposal_source_hash,
        )

    @property
    def statistics_hash(self) -> str:
        return self.confidence_box.state_hash

    @property
    def design_hash(self) -> str:
        return self.confidence_box.design_hash

    @property
    def confidence_box_hash(self) -> str:
        return self.confidence_box.receipt_hash

    @property
    def confidence_box_schema(self) -> str:
        return self.confidence_box.schema

    @property
    def confidence_box_semantics(self) -> str:
        if type(self.confidence_box) is V1CorrectedAttributionConfidenceBoxReceipt:
            return ACTION_REGRET_V1_BOX_SEMANTICS
        if type(self.confidence_box) is V1CorrectedAttributionConfidenceBoxV2Receipt:
            return ACTION_REGRET_V2_BOX_SEMANTICS
        raise RuntimeError("action-regret request confidence-box type changed")

    @property
    def proposal_composition_point(self) -> np.ndarray:
        return self.confidence_box.class_balanced_absolute_point

    @property
    def action_regret_lower(self) -> np.ndarray:
        return self.confidence_box.class_balanced_absolute_lower

    @property
    def action_regret_upper(self) -> np.ndarray:
        return self.confidence_box.class_balanced_absolute_upper

    @property
    def expected_receipt_schema(self) -> str:
        return (
            CORRECTED_ATTRIBUTION_ACTION_REGRET_RECEIPT_SCHEMA
            if type(self.confidence_box)
            is V1CorrectedAttributionConfidenceBoxReceipt
            else CORRECTED_ATTRIBUTION_ACTION_REGRET_RECEIPT_V3_SCHEMA
        )

    @property
    def route_count(self) -> int:
        return self.confidence_box.route_count

    @property
    def route_axis_hash(self) -> str:
        return self.confidence_box.route_axis_hash

    @property
    def action_family_hash(self) -> str:
        return self.action_family.family_hash

    def _assert_coordinate_agreement(self) -> None:
        box = self.confidence_box
        family = self.action_family
        pairs = {
            "design hash": (family.registered_design_hash, box.design_hash),
            "campaign contract hash": (
                family.campaign_contract_hash,
                box.campaign_contract_hash,
            ),
            "model identity hash": (
                family.model_identity_hash,
                box.model_identity_hash,
            ),
            "query contract hash": (
                family.query_contract_hash,
                box.query_contract_hash,
            ),
            "preprocessing hash": (family.preprocessing_hash, box.preprocessing_hash),
            "current mask hash": (family.current_mask_hash, box.current_mask_hash),
            "proposal generation": (
                family.proposal_generation,
                box.proposal_generation,
            ),
            "route count": (family.route_count, box.route_count),
            "route axis hash": (family.route_axis_hash, box.route_axis_hash),
        }
        for name, values in pairs.items():
            if values[0] != values[1]:
                raise ValueError(f"action family and confidence box disagree on {name}")

    def _compute_proposal_hash(self) -> str:
        return content_hash(
            {
                "route_count": self.route_count,
                "route_axis_hash": self.route_axis_hash,
                "action_family_hash": self.action_family_hash,
                "proposed_route_indices": self.proposed_route_indices,
                "proposal_source_hash": self.proposal_source_hash,
            },
            prefix=_ACTION_PROPOSAL_PREFIX,
        )

    @property
    def proposal_hash(self) -> str:
        self.assert_integrity()
        assert self._proposal_hash is not None
        return self._proposal_hash

    def _hash_payload(self, *, proposal_hash: str) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "statistics_hash": self.statistics_hash,
            "design_hash": self.design_hash,
            "confidence_box_hash": self.confidence_box_hash,
            "route_count": self.route_count,
            "route_axis_hash": self.route_axis_hash,
            "action_family_hash": self.action_family_hash,
            "proposal_hash": proposal_hash,
            "proposed_route_indices": self.proposed_route_indices,
            "proposal_source_hash": self.proposal_source_hash,
            "maximum_action_regret": self.action_family.maximum_action_regret,
            "action_regret_currency": CORRECTED_ATTRIBUTION_ACTION_REGRET_CURRENCY,
            "proposal_discovery_only": self.proposal_discovery_only,
            "nonlinear_geometric_selector_authority": (
                self.nonlinear_geometric_selector_authority
            ),
            "mask_commit_authority": self.mask_commit_authority,
        }
        if type(self.confidence_box) is V1CorrectedAttributionConfidenceBoxV2Receipt:
            payload.update(
                {
                    "confidence_box_schema": self.confidence_box_schema,
                    "confidence_box_semantics": self.confidence_box_semantics,
                    "proposal_composition_source": (
                        ACTION_PROPOSAL_CURRENT_POINT_COMPOSITION
                    ),
                    "action_regret_interval_source": (
                        "cumulative_class_balanced_absolute_lower_upper"
                    ),
                }
            )
        return payload

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.confidence_box.assert_integrity()
        self.action_family.assert_integrity()
        self._assert_coordinate_agreement()
        proposal_hash = self._compute_proposal_hash()
        if self._proposal_hash is not None and proposal_hash != self._proposal_hash:
            raise ValueError("corrected-attribution action proposal was tampered")
        observed = content_hash(
            self._hash_payload(proposal_hash=proposal_hash),
            prefix=_ACTION_REGRET_REQUEST_PREFIX,
        )
        if self._request_hash is not None and observed != self._request_hash:
            raise ValueError("corrected-attribution action-regret request was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError(
                "corrected-attribution action-regret request hash does not match"
            )
        object.__setattr__(self, "_proposal_hash", proposal_hash)
        object.__setattr__(self, "_request_hash", observed)
        return observed

    @property
    def request_hash(self) -> str:
        return self.assert_integrity()

    def require_nonlinear_geometric_selector_authority(self) -> None:
        self.action_family.require_nonlinear_geometric_selector_authority()

    def require_mask_commit_authority(self) -> None:
        self.action_family.require_mask_commit_authority()

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any], str]]:
        payload = {
            "confidence_box": self.confidence_box,
            "action_family": self.action_family,
            "proposed_route_indices": self.proposed_route_indices,
            "proposal_source_hash": self.proposal_source_hash,
            "proposal_discovery_only": self.proposal_discovery_only,
            "nonlinear_geometric_selector_authority": (
                self.nonlinear_geometric_selector_authority
            ),
            "mask_commit_authority": self.mask_commit_authority,
            "schema": self.schema,
        }
        return _rebuild_action_regret_request, (payload, self.assert_integrity())


def _deterministic_exchange_boundary(
    values: np.ndarray,
    *,
    selected_flags: np.ndarray,
    require_selected: bool,
    maximize: bool,
    physical_indices: np.ndarray | None = None,
) -> tuple[int, float]:
    """Return one exact partition boundary with a physical-index tie rule.

    The scan is deliberately chunked.  On the 57M-route production axis this
    avoids materializing either a partition-sized gathered value vector or a
    route-sized masked float vector merely to identify one boundary route.
    """

    vector = np.asarray(values, dtype=_LE_F64).reshape(-1)
    flags = np.asarray(selected_flags, dtype=np.bool_).reshape(-1)
    # The only caller accepts an integrity-checked confidence-box receipt, so
    # finiteness has already been established once at that authority boundary.
    # Re-scanning the entire score vector for every direct-sum component would
    # erase the intended route-scale saving.
    if vector.shape != flags.shape:
        raise ValueError("exchange-boundary inputs must share one route axis")
    if physical_indices is None:
        member_count = int(vector.size)
        members: np.ndarray | None = None
    else:
        members = np.asarray(physical_indices, dtype=_LE_U64).reshape(-1)
        member_count = int(members.size)

    best_route: int | None = None
    best_value = -math.inf if maximize else math.inf
    for start in range(0, member_count, _EXCHANGE_BOUNDARY_SCAN_CHUNK_SIZE):
        stop = min(start + _EXCHANGE_BOUNDARY_SCAN_CHUNK_SIZE, member_count)
        if members is None:
            chunk_values = vector[start:stop]
            chunk_flags = flags[start:stop]
            chunk_routes: np.ndarray | None = None
        else:
            chunk_routes = members[start:stop]
            chunk_values = vector[chunk_routes]
            chunk_flags = flags[chunk_routes]
        eligible = chunk_flags if require_selected else np.logical_not(chunk_flags)
        if not bool(np.any(eligible)):
            continue
        if maximize:
            local_value = float(
                np.max(chunk_values, where=eligible, initial=-math.inf)
            )
        else:
            local_value = float(
                np.min(chunk_values, where=eligible, initial=math.inf)
            )
        local_match = eligible & (chunk_values == local_value)
        local_position = int(np.argmax(local_match))
        local_route = (
            start + local_position
            if chunk_routes is None
            else int(chunk_routes[local_position])
        )
        better = local_value > best_value if maximize else local_value < best_value
        if better or (
            local_value == best_value
            and (best_route is None or local_route < best_route)
        ):
            best_value = local_value
            best_route = local_route
    if best_route is None:
        requested = "selected" if require_selected else "outside"
        raise RuntimeError(f"nontrivial partition has no {requested} boundary")
    return best_route, best_value


@dataclass(frozen=True, slots=True)
class V1CorrectedAttributionZeroRegretExchangeCertificate:
    """Compact exact zero-regret certificate for the active matroid bases.

    For every nontrivial direct-sum component, the proposed base has zero
    Cartesian-box regret exactly when its largest selected upper endpoint is
    no larger than its smallest outside lower endpoint.  A strict violation
    is itself a feasible positive one-exchange regret witness.  Equality is
    objective degeneracy, not a refusal.
    """

    request: V1CorrectedAttributionActionRegretRequest = field(
        repr=False, compare=False
    )
    schema: str = CORRECTED_ATTRIBUTION_ZERO_REGRET_EXCHANGE_CERTIFICATE_SCHEMA
    nontrivial_partition_indices: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    vacuous_zero_quota_partition_indices: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    vacuous_full_quota_partition_indices: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    selected_boundary_route_indices: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    outside_boundary_route_indices: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    selected_upper_boundaries: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    outside_lower_boundaries: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    raw_exchange_advantages: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    maximum_feasible_one_exchange_raw_advantage: float | None = field(init=False)
    maximum_feasible_one_exchange_regret: float = field(init=False)
    positive_exchange_witness_partition_index: int | None = field(init=False)
    positive_exchange_witness_partition_name: str | None = field(init=False)
    positive_exchange_witness_selected_route_index: int | None = field(init=False)
    positive_exchange_witness_outside_route_index: int | None = field(init=False)
    positive_exchange_witness_selected_upper: float | None = field(init=False)
    positive_exchange_witness_outside_lower: float | None = field(init=False)
    positive_exchange_witness_regret: float | None = field(init=False)
    zero_total_action_regret_certified: bool = field(init=False)
    strict_action_set_optimality_certified: bool = field(init=False)
    _bound_request_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )
    _certificate_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if type(self.request) is not V1CorrectedAttributionActionRegretRequest:
            raise TypeError(
                "zero-regret exchange certificate requires the exact request"
            )
        bound_request_hash = self.request.assert_integrity()
        if self.request.action_family.maximum_action_regret != 0.0:
            raise ValueError(
                "zero-regret exchange certification requires a registered zero budget"
            )
        if (
            self.schema
            != CORRECTED_ATTRIBUTION_ZERO_REGRET_EXCHANGE_CERTIFICATE_SCHEMA
        ):
            raise ValueError("zero-regret exchange-certificate schema is invalid")

        family = self.request.action_family
        selected_flags = np.zeros(self.request.route_count, dtype=np.bool_)
        selected_flags[self.request.proposed_route_indices] = True

        if family.family_kind == GLOBAL_FIXED_CARDINALITY_ACTION_FAMILY:
            partitions: tuple[tuple[int, str, np.ndarray | None, int, int], ...] = (
                (0, "global", None, family.action_size, family.route_count),
            )
        else:
            partitions = tuple(
                (
                    partition,
                    family.partition_names[partition],
                    family.partition_route_indices[
                        int(family.partition_offsets[partition]) : int(
                            family.partition_offsets[partition + 1]
                        )
                    ],
                    int(family.partition_exact_quotas[partition]),
                    int(
                        family.partition_offsets[partition + 1]
                        - family.partition_offsets[partition]
                    ),
                )
                for partition in range(len(family.partition_names))
            )

        nontrivial: list[int] = []
        zero_quota: list[int] = []
        full_quota: list[int] = []
        selected_routes: list[int] = []
        outside_routes: list[int] = []
        selected_boundaries: list[float] = []
        outside_boundaries: list[float] = []
        advantages: list[float] = []
        candidates: list[tuple[float, int, str, int, int, float, float]] = []

        for partition, name, members, quota, size in partitions:
            if quota == 0:
                zero_quota.append(partition)
                continue
            if quota == size:
                full_quota.append(partition)
                continue
            selected_route, selected_upper = _deterministic_exchange_boundary(
                self.request.action_regret_upper,
                selected_flags=selected_flags,
                require_selected=True,
                maximize=True,
                physical_indices=members,
            )
            outside_route, outside_lower = _deterministic_exchange_boundary(
                self.request.action_regret_lower,
                selected_flags=selected_flags,
                require_selected=False,
                maximize=False,
                physical_indices=members,
            )
            advantage = float(selected_upper - outside_lower)
            nontrivial.append(partition)
            selected_routes.append(selected_route)
            outside_routes.append(outside_route)
            selected_boundaries.append(selected_upper)
            outside_boundaries.append(outside_lower)
            advantages.append(advantage)
            candidates.append(
                (
                    advantage,
                    partition,
                    name,
                    selected_route,
                    outside_route,
                    selected_upper,
                    outside_lower,
                )
            )

        if candidates:
            strongest = min(
                candidates,
                key=lambda item: (-item[0], item[1], item[3], item[4]),
            )
            maximum_advantage: float | None = strongest[0]
        else:
            strongest = None
            maximum_advantage = None
        positive_witness = (
            strongest if strongest is not None and strongest[0] > 0.0 else None
        )
        zero_regret = positive_witness is None
        strict_optimality = strongest is None or strongest[0] < 0.0

        object.__setattr__(
            self,
            "nontrivial_partition_indices",
            _readonly_le_u64_vector(nontrivial, name="nontrivial partition indices"),
        )
        object.__setattr__(
            self,
            "vacuous_zero_quota_partition_indices",
            _readonly_le_u64_vector(zero_quota, name="zero-quota partition indices"),
        )
        object.__setattr__(
            self,
            "vacuous_full_quota_partition_indices",
            _readonly_le_u64_vector(full_quota, name="full-quota partition indices"),
        )
        object.__setattr__(
            self,
            "selected_boundary_route_indices",
            _readonly_le_u64_vector(selected_routes, name="selected boundary routes"),
        )
        object.__setattr__(
            self,
            "outside_boundary_route_indices",
            _readonly_le_u64_vector(outside_routes, name="outside boundary routes"),
        )
        object.__setattr__(
            self,
            "selected_upper_boundaries",
            _readonly_vector(
                np.asarray(selected_boundaries, dtype=_LE_F64),
                name="selected upper boundaries",
                size=len(selected_boundaries),
            ),
        )
        object.__setattr__(
            self,
            "outside_lower_boundaries",
            _readonly_vector(
                np.asarray(outside_boundaries, dtype=_LE_F64),
                name="outside lower boundaries",
                size=len(outside_boundaries),
            ),
        )
        object.__setattr__(
            self,
            "raw_exchange_advantages",
            _readonly_vector(
                np.asarray(advantages, dtype=_LE_F64),
                name="raw exchange advantages",
                size=len(advantages),
            ),
        )
        object.__setattr__(
            self, "maximum_feasible_one_exchange_raw_advantage", maximum_advantage
        )
        object.__setattr__(
            self,
            "maximum_feasible_one_exchange_regret",
            0.0 if positive_witness is None else positive_witness[0],
        )
        object.__setattr__(
            self,
            "positive_exchange_witness_partition_index",
            None if positive_witness is None else positive_witness[1],
        )
        object.__setattr__(
            self,
            "positive_exchange_witness_partition_name",
            None if positive_witness is None else positive_witness[2],
        )
        object.__setattr__(
            self,
            "positive_exchange_witness_selected_route_index",
            None if positive_witness is None else positive_witness[3],
        )
        object.__setattr__(
            self,
            "positive_exchange_witness_outside_route_index",
            None if positive_witness is None else positive_witness[4],
        )
        object.__setattr__(
            self,
            "positive_exchange_witness_selected_upper",
            None if positive_witness is None else positive_witness[5],
        )
        object.__setattr__(
            self,
            "positive_exchange_witness_outside_lower",
            None if positive_witness is None else positive_witness[6],
        )
        object.__setattr__(
            self,
            "positive_exchange_witness_regret",
            None if positive_witness is None else positive_witness[0],
        )
        object.__setattr__(self, "zero_total_action_regret_certified", zero_regret)
        object.__setattr__(
            self, "strict_action_set_optimality_certified", strict_optimality
        )
        object.__setattr__(self, "_bound_request_hash", bound_request_hash)
        object.__setattr__(self, "_certificate_hash", self._compute_certificate_hash())

    @property
    def request_hash(self) -> str:
        return self.request.request_hash

    @property
    def action_family_hash(self) -> str:
        return self.request.action_family_hash

    @property
    def confidence_box_hash(self) -> str:
        return self.request.confidence_box_hash

    @property
    def proposal_hash(self) -> str:
        return self.request.proposal_hash

    @property
    def theorem_id(self) -> str:
        return CORRECTED_ATTRIBUTION_ZERO_REGRET_EXCHANGE_THEOREM

    @property
    def action_regret_resolved(self) -> bool:
        return self.zero_total_action_regret_certified

    @property
    def certified_maximum_action_regret(self) -> float | None:
        return 0.0 if self.zero_total_action_regret_certified else None

    @property
    def full_adversarial_competitor_materialized(self) -> bool:
        return False

    @property
    def total_action_regret_exact(self) -> bool:
        return self.zero_total_action_regret_certified

    def _hash_payload(self) -> dict[str, Any]:
        if self._bound_request_hash is None:
            raise RuntimeError("zero-regret exchange certificate is not bound")
        return {
            "schema": self.schema,
            # The request hash already binds the confidence box, action
            # family, proposal, and every coordinate identity.  Hashing those
            # route-scale children again would add no evidence.
            "request_hash": self._bound_request_hash,
            "nontrivial_partition_indices": self.nontrivial_partition_indices,
            "vacuous_zero_quota_partition_indices": (
                self.vacuous_zero_quota_partition_indices
            ),
            "vacuous_full_quota_partition_indices": (
                self.vacuous_full_quota_partition_indices
            ),
            "selected_boundary_route_indices": self.selected_boundary_route_indices,
            "outside_boundary_route_indices": self.outside_boundary_route_indices,
            "selected_upper_boundaries": self.selected_upper_boundaries,
            "outside_lower_boundaries": self.outside_lower_boundaries,
            "raw_exchange_advantages": self.raw_exchange_advantages,
            "maximum_feasible_one_exchange_raw_advantage": (
                self.maximum_feasible_one_exchange_raw_advantage
            ),
            "maximum_feasible_one_exchange_regret": (
                self.maximum_feasible_one_exchange_regret
            ),
            "positive_exchange_witness_partition_index": (
                self.positive_exchange_witness_partition_index
            ),
            "positive_exchange_witness_partition_name": (
                self.positive_exchange_witness_partition_name
            ),
            "positive_exchange_witness_selected_route_index": (
                self.positive_exchange_witness_selected_route_index
            ),
            "positive_exchange_witness_outside_route_index": (
                self.positive_exchange_witness_outside_route_index
            ),
            "positive_exchange_witness_selected_upper": (
                self.positive_exchange_witness_selected_upper
            ),
            "positive_exchange_witness_outside_lower": (
                self.positive_exchange_witness_outside_lower
            ),
            "positive_exchange_witness_regret": self.positive_exchange_witness_regret,
            "zero_total_action_regret_certified": (
                self.zero_total_action_regret_certified
            ),
            "strict_action_set_optimality_certified": (
                self.strict_action_set_optimality_certified
            ),
            "theorem_id": self.theorem_id,
            "cartesian_confidence_box_support_exact": True,
            "full_adversarial_competitor_materialized": False,
            "calibration_or_oracle_outcomes_observed": False,
            "terminal_validation_observed": False,
        }

    def _compute_certificate_hash(self) -> str:
        return content_hash(
            self._hash_payload(), prefix=_ZERO_REGRET_EXCHANGE_CERTIFICATE_PREFIX
        )

    def _assert_integrity_with_request_hash(
        self,
        request_hash: str,
        expected_hash: str | None = None,
    ) -> str:
        if request_hash != self._bound_request_hash:
            raise ValueError("zero-regret exchange certificate request was replaced")
        observed = self._compute_certificate_hash()
        if self._certificate_hash is not None and observed != self._certificate_hash:
            raise ValueError("zero-regret exchange certificate was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("zero-regret exchange-certificate hash does not match")
        object.__setattr__(self, "_certificate_hash", observed)
        return observed

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        request_hash = self.request.assert_integrity()
        return self._assert_integrity_with_request_hash(request_hash, expected_hash)

    @property
    def certificate_hash(self) -> str:
        return self.assert_integrity()

    def __reduce__(self) -> tuple[Any, tuple[Any, str, str]]:
        return _rebuild_zero_regret_exchange_certificate, (
            self.request,
            self.schema,
            self.assert_integrity(),
        )


@dataclass(frozen=True, slots=True)
class V1CorrectedAttributionActionRegretReceipt:
    """Decision-exact robust regret over one fixed additive action family.

    A zero registered budget uses the compact one-exchange theorem.  A
    nonzero budget retains the materialized exact adversarial optimizer.
    """

    request: V1CorrectedAttributionActionRegretRequest = field(
        repr=False, compare=False
    )
    schema: str | None = None
    worst_competitor_route_indices: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    proposed_only_route_indices: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    competitor_only_route_indices: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    zero_regret_exchange_certificate: (
        V1CorrectedAttributionZeroRegretExchangeCertificate | None
    ) = field(init=False, repr=False, compare=False)
    worst_competitor_action_hash: str | None = field(init=False)
    proposed_only_route_index_hash: str | None = field(init=False)
    competitor_only_route_index_hash: str | None = field(init=False)
    proposed_only_upper_total: float | None = field(init=False)
    competitor_only_lower_total: float | None = field(init=False)
    raw_maximum_action_regret: float | None = field(init=False)
    certified_maximum_action_regret: float | None = field(init=False)
    negative_roundoff_tolerance: float | None = field(init=False)
    fp64_outward_roundoff_tolerance: float | None = field(init=False)
    registered_maximum_action_regret: float = field(init=False)
    action_regret_resolved: bool = field(init=False)
    _receipt_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if type(self.request) is not V1CorrectedAttributionActionRegretRequest:
            raise TypeError("action-regret receipt requires the exact request type")
        self.request.assert_integrity()
        schema = (
            self.request.expected_receipt_schema
            if self.schema is None
            else self.schema
        )
        if schema != self.request.expected_receipt_schema:
            raise ValueError(
                "action-regret receipt schema does not match its request ABI"
            )
        object.__setattr__(self, "schema", schema)
        registered_tolerance = self.request.action_family.maximum_action_regret
        if registered_tolerance == 0.0:
            exchange_certificate = (
                V1CorrectedAttributionZeroRegretExchangeCertificate(
                    request=self.request
                )
            )
            empty = _readonly_le_u64_vector(
                (), name="unmaterialized adversarial action axis", size=0
            )
            object.__setattr__(self, "worst_competitor_route_indices", empty)
            object.__setattr__(self, "proposed_only_route_indices", empty)
            object.__setattr__(self, "competitor_only_route_indices", empty)
            object.__setattr__(
                self, "zero_regret_exchange_certificate", exchange_certificate
            )
            object.__setattr__(self, "worst_competitor_action_hash", None)
            object.__setattr__(self, "proposed_only_route_index_hash", None)
            object.__setattr__(self, "competitor_only_route_index_hash", None)
            object.__setattr__(self, "proposed_only_upper_total", None)
            object.__setattr__(self, "competitor_only_lower_total", None)
            object.__setattr__(
                self,
                "raw_maximum_action_regret",
                0.0 if exchange_certificate.action_regret_resolved else None,
            )
            object.__setattr__(
                self,
                "certified_maximum_action_regret",
                exchange_certificate.certified_maximum_action_regret,
            )
            object.__setattr__(self, "negative_roundoff_tolerance", None)
            object.__setattr__(self, "fp64_outward_roundoff_tolerance", None)
            object.__setattr__(
                self, "registered_maximum_action_regret", registered_tolerance
            )
            object.__setattr__(
                self,
                "action_regret_resolved",
                exchange_certificate.action_regret_resolved,
            )
            self.assert_integrity()
            return

        selected = self.request.proposed_route_indices
        selected_flags = np.zeros(self.request.route_count, dtype=np.bool_)
        selected_flags[selected] = True
        adversarial_costs = np.array(
            self.request.action_regret_lower, dtype=_LE_F64, copy=True
        )
        adversarial_costs[selected] = self.request.action_regret_upper[selected]
        competitor = self.request.action_family._minimum_additive_action(
            adversarial_costs
        )
        # Revalidate the result through the public family predicate.  This is
        # intentionally redundant at the theorem boundary.
        competitor = self.request.action_family.canonical_action(competitor)
        competitor_flags = np.zeros(self.request.route_count, dtype=np.bool_)
        competitor_flags[competitor] = True
        proposed_only = _readonly_le_u64_vector(
            selected[~competitor_flags[selected]], name="proposal-only routes"
        )
        competitor_only = _readonly_le_u64_vector(
            competitor[~selected_flags[competitor]], name="competitor-only routes"
        )
        if proposed_only.size != competitor_only.size:
            raise RuntimeError(
                "fixed-size feasible actions must have balanced symmetric differences"
            )

        proposed_only_upper_values = self.request.action_regret_upper[proposed_only]
        competitor_only_lower_values = self.request.action_regret_lower[competitor_only]
        selected_total = math.fsum(float(value) for value in proposed_only_upper_values)
        competitor_total = math.fsum(
            float(value) for value in competitor_only_lower_values
        )
        raw_regret = math.fsum((selected_total, -competitor_total))
        roundoff_tolerance = (
            8.0
            * np.finfo(np.float64).eps
            * max(1.0, abs(selected_total), abs(competitor_total))
        )
        if raw_regret < -roundoff_tolerance:
            raise RuntimeError(
                "the exact adversarial action minimizer produced negative robust regret"
            )
        exact_zero_by_multiset_cancellation = (
            raw_regret <= roundoff_tolerance
            and _same_float64_multiset(
                proposed_only_upper_values,
                competitor_only_lower_values,
            )
        )
        if exact_zero_by_multiset_cancellation:
            maximum_regret = 0.0
        else:
            maximum_regret = math.nextafter(
                max(0.0, raw_regret) + roundoff_tolerance,
                math.inf,
            )
        object.__setattr__(self, "worst_competitor_route_indices", competitor)
        object.__setattr__(self, "proposed_only_route_indices", proposed_only)
        object.__setattr__(self, "competitor_only_route_indices", competitor_only)
        object.__setattr__(self, "zero_regret_exchange_certificate", None)
        object.__setattr__(
            self,
            "worst_competitor_action_hash",
            content_hash(
                {
                    "action_family_hash": self.request.action_family_hash,
                    "route_axis_hash": self.request.route_axis_hash,
                    "route_indices": competitor,
                },
                prefix=_ACTION_PROPOSAL_PREFIX + "competitor-",
            ),
        )
        object.__setattr__(
            self,
            "proposed_only_route_index_hash",
            content_hash(
                proposed_only, prefix=_ACTION_REGRET_RECEIPT_PREFIX + "proposal-only-"
            ),
        )
        object.__setattr__(
            self,
            "competitor_only_route_index_hash",
            content_hash(
                competitor_only,
                prefix=_ACTION_REGRET_RECEIPT_PREFIX + "competitor-only-",
            ),
        )
        object.__setattr__(self, "proposed_only_upper_total", selected_total)
        object.__setattr__(self, "competitor_only_lower_total", competitor_total)
        object.__setattr__(self, "raw_maximum_action_regret", raw_regret)
        object.__setattr__(self, "certified_maximum_action_regret", maximum_regret)
        object.__setattr__(self, "negative_roundoff_tolerance", roundoff_tolerance)
        object.__setattr__(
            self,
            "fp64_outward_roundoff_tolerance",
            roundoff_tolerance,
        )
        object.__setattr__(
            self, "registered_maximum_action_regret", registered_tolerance
        )
        object.__setattr__(
            self,
            "action_regret_resolved",
            bool(maximum_regret <= registered_tolerance),
        )
        self.assert_integrity()

    @property
    def request_hash(self) -> str:
        return self.request.request_hash

    @property
    def statistics_hash(self) -> str:
        return self.request.statistics_hash

    @property
    def design_hash(self) -> str:
        return self.request.design_hash

    @property
    def confidence_box_hash(self) -> str:
        return self.request.confidence_box_hash

    @property
    def route_axis_hash(self) -> str:
        return self.request.route_axis_hash

    @property
    def action_family_hash(self) -> str:
        return self.request.action_family_hash

    @property
    def proposal_hash(self) -> str:
        return self.request.proposal_hash

    @property
    def proposal_discovery_only(self) -> bool:
        return True

    @property
    def nonlinear_geometric_selector_authority(self) -> bool:
        return False

    @property
    def mask_commit_authority(self) -> bool:
        return False

    @property
    def adversarial_cost_rule(self) -> str:
        return "upper_on_proposal_lower_off_proposal"

    @property
    def action_regret_currency(self) -> str:
        return CORRECTED_ATTRIBUTION_ACTION_REGRET_CURRENCY

    @property
    def theorem_id(self) -> str:
        return CORRECTED_ATTRIBUTION_ACTION_REGRET_THEOREM

    @property
    def deterministic_tie_rule(self) -> str:
        return CORRECTED_ATTRIBUTION_ACTION_REGRET_TIE_RULE

    @property
    def authority_scope(self) -> str:
        return CORRECTED_ATTRIBUTION_PROPOSAL_ONLY_AUTHORITY

    @property
    def zero_regret_exchange_certificate_hash(self) -> str | None:
        certificate = self.zero_regret_exchange_certificate
        return None if certificate is None else certificate.certificate_hash

    @property
    def zero_regret_exchange_certificate_used(self) -> bool:
        return self.zero_regret_exchange_certificate is not None

    @property
    def full_adversarial_competitor_materialized(self) -> bool:
        return self.zero_regret_exchange_certificate is None

    @property
    def total_action_regret_exact(self) -> bool:
        certificate = self.zero_regret_exchange_certificate
        return certificate is None or certificate.total_action_regret_exact

    def _hash_payload(
        self, *, validated_exchange_certificate_hash: str | None
    ) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "request_hash": self.request_hash,
            "statistics_hash": self.statistics_hash,
            "design_hash": self.design_hash,
            "confidence_box_hash": self.confidence_box_hash,
            "route_axis_hash": self.route_axis_hash,
            "action_family_hash": self.action_family_hash,
            "proposal_hash": self.proposal_hash,
            "zero_regret_exchange_certificate_hash": (
                validated_exchange_certificate_hash
            ),
            "zero_regret_exchange_certificate_used": (
                self.zero_regret_exchange_certificate_used
            ),
            "worst_competitor_route_indices": self.worst_competitor_route_indices,
            "worst_competitor_action_hash": self.worst_competitor_action_hash,
            "proposed_only_route_indices": self.proposed_only_route_indices,
            "proposed_only_route_index_hash": self.proposed_only_route_index_hash,
            "competitor_only_route_indices": self.competitor_only_route_indices,
            "competitor_only_route_index_hash": self.competitor_only_route_index_hash,
            "proposed_only_upper_total": self.proposed_only_upper_total,
            "competitor_only_lower_total": self.competitor_only_lower_total,
            "raw_maximum_action_regret": self.raw_maximum_action_regret,
            "certified_maximum_action_regret": self.certified_maximum_action_regret,
            "negative_roundoff_tolerance": self.negative_roundoff_tolerance,
            "fp64_outward_roundoff_tolerance": (self.fp64_outward_roundoff_tolerance),
            "registered_maximum_action_regret": self.registered_maximum_action_regret,
            "action_regret_resolved": self.action_regret_resolved,
            "adversarial_cost_rule": self.adversarial_cost_rule,
            "action_regret_currency": self.action_regret_currency,
            "theorem_id": self.theorem_id,
            "deterministic_tie_rule": self.deterministic_tie_rule,
            "discrete_competitor_optimization_exact": (
                self.full_adversarial_competitor_materialized
            ),
            "full_adversarial_competitor_materialized": (
                self.full_adversarial_competitor_materialized
            ),
            "total_action_regret_exact": self.total_action_regret_exact,
            "cartesian_confidence_box_support_exact": True,
            "authority_scope": self.authority_scope,
            "proposal_discovery_only": self.proposal_discovery_only,
            "nonlinear_geometric_selector_authority": (
                self.nonlinear_geometric_selector_authority
            ),
            "mask_commit_authority": self.mask_commit_authority,
            "calibration_or_oracle_outcomes_observed": False,
            "terminal_validation_observed": False,
        }
        if type(self.request.confidence_box) is (
            V1CorrectedAttributionConfidenceBoxV2Receipt
        ):
            payload.update(
                {
                    "confidence_box_schema": self.request.confidence_box_schema,
                    "confidence_box_semantics": self.request.confidence_box_semantics,
                    "proposal_composition_source": (
                        ACTION_PROPOSAL_CURRENT_POINT_COMPOSITION
                    ),
                    "action_regret_interval_source": (
                        "cumulative_class_balanced_absolute_lower_upper"
                    ),
                }
            )
        return payload

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        request_hash = self.request.assert_integrity()
        certificate = self.zero_regret_exchange_certificate
        validated_certificate_hash = (
            None
            if certificate is None
            else certificate._assert_integrity_with_request_hash(request_hash)
        )
        observed = content_hash(
            self._hash_payload(
                validated_exchange_certificate_hash=validated_certificate_hash
            ),
            prefix=_ACTION_REGRET_RECEIPT_PREFIX,
        )
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("corrected-attribution action-regret receipt was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError(
                "corrected-attribution action-regret receipt hash does not match"
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def require_nonlinear_geometric_selector_authority(self) -> None:
        self.request.require_nonlinear_geometric_selector_authority()

    def require_mask_commit_authority(self) -> None:
        self.request.require_mask_commit_authority()

    def __reduce__(self) -> tuple[Any, tuple[Any, str, str]]:
        return _rebuild_action_regret_receipt, (
            self.request,
            self.schema,
            self.assert_integrity(),
        )


def certify_corrected_attribution_action_regret(
    request: V1CorrectedAttributionActionRegretRequest,
    *,
    expected_request_hash: str | None = None,
) -> V1CorrectedAttributionActionRegretReceipt:
    """Evaluate exact box-robust regret for one registered additive action.

    This public entry point deliberately accepts no geometric response metric,
    KKT state, calibration result, oracle outcome, or mask handle.
    """

    if type(request) is not V1CorrectedAttributionActionRegretRequest:
        raise TypeError("action-regret certification requires the exact request type")
    request.assert_integrity(expected_request_hash)
    return V1CorrectedAttributionActionRegretReceipt(request=request)


def corrected_attribution_confidence_box_v2(
    *,
    statistics: V1StratifiedCorrectedAttributionMoments,
    envelope_certificate: V1CorrectedAttributionEnvelopeCertificate,
    design: V1GroupSequentialEmpiricalBernsteinSerflingDesign,
    prior_confidence_box: V1CorrectedAttributionConfidenceBoxV2Receipt | None = None,
    expected_statistics_hash: str | None = None,
    expected_envelope_certificate_hash: str | None = None,
    expected_design_hash: str | None = None,
    expected_prior_confidence_box_hash: str | None = None,
) -> V1CorrectedAttributionConfidenceBoxV2Receipt:
    """Build the theorem-audited HS/EB confidence box and running intersection.

    This is a new authority surface.  It deliberately does not reinterpret a
    v1 confidence-box receipt or the ``MV1PROP1`` moments sidecar.  Production
    integration must explicitly opt into this builder and persist its receipt
    chain before consuming the cumulative action-facing bounds.
    """

    if type(statistics) is not V1StratifiedCorrectedAttributionMoments:
        raise TypeError("statistics must be canonical corrected-attribution moments")
    if type(envelope_certificate) is not V1CorrectedAttributionEnvelopeCertificate:
        raise TypeError("envelope_certificate must be the canonical certificate")
    if type(design) is not V1GroupSequentialEmpiricalBernsteinSerflingDesign:
        raise TypeError("design must be the exact v2 confidence design")
    if prior_confidence_box is not None and type(prior_confidence_box) is not (
        V1CorrectedAttributionConfidenceBoxV2Receipt
    ):
        raise TypeError("a v2 running intersection requires the exact v2 prior receipt")

    state_hash = statistics.assert_integrity(expected_statistics_hash)
    certificate_hash = envelope_certificate.assert_integrity(
        expected_envelope_certificate_hash
    )
    design_hash = design.assert_integrity(expected_design_hash)
    base_design = design.serfling_design
    base_design_hash = base_design.assert_integrity()

    coordinate_names = (
        "campaign_contract_hash",
        "model_identity_hash",
        "query_contract_hash",
        "preprocessing_hash",
        "current_mask_hash",
        "raw_logit_stream_source_hash",
        "sampling_law_hash",
        "proposal_generation",
        "route_count",
        "route_axis_hash",
        "stratum_names",
        "population_stratum_counts",
        "sealed_population_order_hashes",
    )
    for name in coordinate_names:
        values = (
            getattr(statistics, name),
            getattr(envelope_certificate, name),
            getattr(base_design, name),
        )
        if not (values[0] == values[1] == values[2]):
            raise ValueError(f"statistics, envelope, and v2 design disagree on {name}")
    if statistics.look_index > base_design.registered_look_count:
        raise ValueError("statistics name an unregistered v2 sequential look")
    scheduled_counts = base_design.registered_look_stratum_counts[
        statistics.look_index - 1
    ]
    if statistics.stratum_counts != scheduled_counts:
        raise ValueError("statistics are off the predeclared v2 look schedule")

    prior_hash: str | None = None
    if statistics.look_index == 1:
        if prior_confidence_box is not None or expected_prior_confidence_box_hash is not None:
            raise ValueError("the first v2 look cannot consume a prior confidence box")
    else:
        if prior_confidence_box is None:
            raise ValueError("every later v2 look requires its immediate prior receipt")
        prior_hash = prior_confidence_box.assert_integrity(
            expected_prior_confidence_box_hash
        )
        prior_coordinates = (
            "envelope_certificate_hash",
            "design_hash",
            "base_serfling_design_hash",
            "campaign_contract_hash",
            "model_identity_hash",
            "query_contract_hash",
            "preprocessing_hash",
            "current_mask_hash",
            "raw_logit_stream_source_hash",
            "sampling_law_hash",
            "proposal_generation",
            "route_count",
            "route_axis_hash",
            "stratum_names",
            "population_stratum_counts",
            "sealed_population_order_hashes",
            "trajectory_alpha_ledger_hash",
            "trajectory_alpha_ledger_generation",
        )
        expected_prior_values: dict[str, Any] = {
            "envelope_certificate_hash": certificate_hash,
            "design_hash": design_hash,
            "base_serfling_design_hash": base_design_hash,
            "trajectory_alpha_ledger_hash": base_design.trajectory_alpha_ledger_hash,
            "trajectory_alpha_ledger_generation": (
                base_design.trajectory_alpha_ledger_generation
            ),
        }
        for name in prior_coordinates:
            expected_value = expected_prior_values.get(name, getattr(statistics, name, None))
            if getattr(prior_confidence_box, name) != expected_value:
                raise ValueError(f"prior v2 confidence box disagrees on {name}")
        if prior_confidence_box.look_index != statistics.look_index - 1:
            raise ValueError("v2 confidence boxes must form an immediate-look chain")
        if statistics.prior_state_hash != prior_confidence_box.state_hash:
            raise ValueError("v2 prior receipt does not bind the moments predecessor")
        if any(
            current < prior
            for current, prior in zip(
                statistics.stratum_counts, prior_confidence_box.stratum_counts
            )
        ) or statistics.stratum_counts == prior_confidence_box.stratum_counts:
            raise ValueError("v2 confidence-box sample prefixes must be strictly nested")
        for stratum in (0, 1):
            prior_count = prior_confidence_box.stratum_counts[stratum]
            if (
                statistics.observed_context_ids_by_stratum[stratum][:prior_count]
                != prior_confidence_box.observed_context_ids_by_stratum[stratum]
                or statistics.observed_draw_ids_by_stratum[stratum][:prior_count]
                != prior_confidence_box.observed_draw_ids_by_stratum[stratum]
            ):
                raise ValueError("v2 confidence-box ledgers are not exact nested prefixes")

    route_count = statistics.route_count
    envelope = envelope_certificate.absolute_upper_envelope_by_stratum
    signed_point = statistics.signed_mean_by_stratum
    absolute_point = statistics.absolute_mean_by_stratum
    tolerance = 128.0 * np.finfo(np.float64).eps * np.maximum(1.0, envelope)
    if np.any(np.abs(signed_point) > envelope + tolerance) or np.any(
        absolute_point > envelope + tolerance
    ):
        raise ValueError("the analytic envelope does not enclose observed v2 moments")

    look_alpha = base_design.alpha_by_look[statistics.look_index - 1]
    cells_at_look = 2 * route_count
    deterministic_census_look = (
        statistics.is_finite_population_census and look_alpha == 0.0
    )
    if deterministic_census_look:
        per_cell_alpha = 0.0
        hs_alpha = 0.0
        eb_alpha = 0.0
        eb_delta = 0.0
        hs_log = 0.0
        eb_log = 0.0
    else:
        per_cell_alpha = look_alpha / cells_at_look
        hs_alpha = 0.5 * per_cell_alpha
        eb_alpha = 0.5 * per_cell_alpha
        eb_delta = eb_alpha / 10.0
        if min(per_cell_alpha, hs_alpha, eb_alpha, eb_delta) <= 0.0:
            raise ValueError("v2 per-cell alpha underflowed and cannot grant authority")
        hs_log = _nextafter_up(math.log(2.0 / hs_alpha))
        eb_log = _nextafter_up(math.log(1.0 / eb_delta))

    class_current_lower = np.zeros(route_count, dtype=_LE_F64)
    class_current_upper = np.zeros(route_count, dtype=_LE_F64)
    deterministic_lower = np.empty(route_count, dtype=_LE_F64)
    deterministic_upper = np.empty(route_count, dtype=_LE_F64)
    statistical_radius = np.empty(route_count, dtype=_LE_F64)
    scratch = np.empty(route_count, dtype=_LE_F64)
    hs_corrections: list[float] = []
    hs_multipliers: list[float] = []
    eb_rhos: list[float] = []
    eb_branches: list[str] = []
    eb_sigma_multipliers: list[float] = []
    eb_range_multipliers: list[float] = []
    census_by_stratum: list[bool] = []

    for stratum, (count, population) in enumerate(
        zip(statistics.stratum_counts, statistics.population_stratum_counts)
    ):
        point = absolute_point[stratum]
        upper_envelope = envelope[stratum]
        is_census = count == population
        census_by_stratum.append(is_census)
        branch = (
            EMPIRICAL_BERNSTEIN_RHO_BRANCH_LE_HALF
            if count <= population / 2.0
            else EMPIRICAL_BERNSTEIN_RHO_BRANCH_GT_HALF
        )
        eb_branches.append(branch)
        if is_census:
            hs_corrections.append(0.0)
            hs_multipliers.append(0.0)
            eb_rhos.append(0.0)
            eb_sigma_multipliers.append(0.0)
            eb_range_multipliers.append(0.0)
            np.copyto(deterministic_lower, point)
            np.copyto(deterministic_upper, point)
        else:
            hs_correction, hs_multiplier = _hs_correction_and_multiplier_up(
                count=count,
                population=population,
                log_factor=hs_log,
            )
            (
                eb_rho,
                branch,
                eb_sigma_multiplier,
                eb_range_multiplier,
            ) = _eb_rho_and_multipliers_up(
                count=count,
                population=population,
                log_factor=eb_log,
            )
            eb_branches[-1] = branch
            hs_corrections.append(hs_correction)
            hs_multipliers.append(hs_multiplier)
            eb_rhos.append(eb_rho)
            eb_sigma_multipliers.append(eb_sigma_multiplier)
            eb_range_multipliers.append(eb_range_multiplier)

            # Deterministic completion interval:
            # [n*m/N, (n*m + (N-n)U)/N].  Each positive arithmetic
            # operation is directed outward before the support intersection.
            np.multiply(point, count, out=deterministic_lower)
            np.nextafter(deterministic_lower, -np.inf, out=deterministic_lower)
            np.divide(deterministic_lower, population, out=deterministic_lower)
            np.nextafter(deterministic_lower, -np.inf, out=deterministic_lower)
            np.maximum(deterministic_lower, 0.0, out=deterministic_lower)
            np.multiply(point, count, out=deterministic_upper)
            np.nextafter(deterministic_upper, np.inf, out=deterministic_upper)
            np.multiply(upper_envelope, population - count, out=scratch)
            np.nextafter(scratch, np.inf, out=scratch)
            np.add(deterministic_upper, scratch, out=deterministic_upper)
            np.nextafter(deterministic_upper, np.inf, out=deterministic_upper)
            np.divide(deterministic_upper, population, out=deterministic_upper)
            np.nextafter(deterministic_upper, np.inf, out=deterministic_upper)
            np.minimum(deterministic_upper, upper_envelope, out=deterministic_upper)

            # Hoeffding--Serfling half-width, rounded upward.
            np.multiply(upper_envelope, hs_multiplier, out=statistical_radius)
            np.nextafter(statistical_radius, np.inf, out=statistical_radius)

            # Bardenet--Maillard Theorem 4.3 uses the biased empirical
            # variance M2/n.  The theorem's variance and range terms are each
            # rounded upward before their sum is rounded upward once more.
            np.divide(
                statistics.absolute_m2_by_stratum[stratum],
                count,
                out=deterministic_lower,
            )
            np.nextafter(deterministic_lower, np.inf, out=deterministic_lower)
            np.sqrt(deterministic_lower, out=deterministic_lower)
            np.nextafter(deterministic_lower, np.inf, out=deterministic_lower)
            np.multiply(
                deterministic_lower,
                eb_sigma_multiplier,
                out=deterministic_lower,
            )
            np.nextafter(deterministic_lower, np.inf, out=deterministic_lower)
            np.multiply(upper_envelope, eb_range_multiplier, out=scratch)
            np.nextafter(scratch, np.inf, out=scratch)
            with np.errstate(over="ignore"):
                np.add(deterministic_lower, scratch, out=scratch)
            np.nextafter(scratch, np.inf, out=scratch)
            np.minimum(statistical_radius, scratch, out=statistical_radius)

            # Reconstruct the deterministic lower endpoint after using its
            # scratch storage for the empirical standard deviation.
            np.multiply(point, count, out=deterministic_lower)
            np.nextafter(deterministic_lower, -np.inf, out=deterministic_lower)
            np.divide(deterministic_lower, population, out=deterministic_lower)
            np.nextafter(deterministic_lower, -np.inf, out=deterministic_lower)
            np.maximum(deterministic_lower, 0.0, out=deterministic_lower)

            np.subtract(point, statistical_radius, out=scratch)
            np.nextafter(scratch, -np.inf, out=scratch)
            np.maximum(scratch, 0.0, out=scratch)
            np.maximum(deterministic_lower, scratch, out=deterministic_lower)
            with np.errstate(over="ignore"):
                np.add(point, statistical_radius, out=scratch)
            np.nextafter(scratch, np.inf, out=scratch)
            np.minimum(scratch, upper_envelope, out=scratch)
            np.minimum(deterministic_upper, scratch, out=deterministic_upper)

        # Equal-stratum macro composition is itself directed outward.
        np.multiply(deterministic_lower, 0.5, out=scratch)
        np.nextafter(scratch, -np.inf, out=scratch)
        np.add(class_current_lower, scratch, out=class_current_lower)
        np.nextafter(class_current_lower, -np.inf, out=class_current_lower)
        np.maximum(class_current_lower, 0.0, out=class_current_lower)

        np.multiply(deterministic_upper, 0.5, out=scratch)
        np.nextafter(scratch, np.inf, out=scratch)
        with np.errstate(over="ignore"):
            np.add(class_current_upper, scratch, out=class_current_upper)
        np.nextafter(class_current_upper, np.inf, out=class_current_upper)

    class_point = 0.5 * (absolute_point[0] + absolute_point[1])
    structural_zero = np.logical_and(envelope[0] == 0.0, envelope[1] == 0.0)
    if statistics.is_finite_population_census:
        # A complete finite population has no inferential or rounding width.
        class_current_lower = np.array(class_point, dtype=_LE_F64, copy=True)
        class_current_upper = np.array(class_point, dtype=_LE_F64, copy=True)
    else:
        # The current box must contain the stored current estimator even under
        # directed-rounding edge cases.
        np.minimum(class_current_lower, class_point, out=class_current_lower)
        np.maximum(class_current_upper, class_point, out=class_current_upper)
    if np.any(structural_zero):
        class_point = np.array(class_point, dtype=_LE_F64, copy=True)
        class_point[structural_zero] = 0.0
        class_current_lower[structural_zero] = 0.0
        class_current_upper[structural_zero] = 0.0

    if prior_confidence_box is None:
        cumulative_lower = np.array(class_current_lower, dtype=_LE_F64, copy=True)
        cumulative_upper = np.array(class_current_upper, dtype=_LE_F64, copy=True)
    else:
        cumulative_lower = np.maximum(
            class_current_lower,
            prior_confidence_box.cumulative_class_balanced_absolute_lower,
        )
        cumulative_upper = np.minimum(
            class_current_upper,
            prior_confidence_box.cumulative_class_balanced_absolute_upper,
        )
        if np.any(cumulative_lower > cumulative_upper):
            raise ValueError(
                "v2 confidence running intersection is empty; inference must refuse"
            )

    cumulative_alpha = math.fsum(base_design.alpha_by_look[: statistics.look_index])
    remaining_generation = max(
        0.0, base_design.generation_alpha_allocation - cumulative_alpha
    )
    unallocated_generation = max(
        0.0,
        base_design.generation_alpha_allocation - base_design.total_allocated_alpha,
    )
    remaining_trajectory = max(
        0.0,
        base_design.trajectory_familywise_alpha
        - base_design.trajectory_alpha_committed_before_generation
        - base_design.generation_alpha_allocation,
    )
    result = V1CorrectedAttributionConfidenceBoxV2Receipt(
        state_hash=state_hash,
        envelope_certificate_hash=certificate_hash,
        design_hash=design_hash,
        base_serfling_design_hash=base_design_hash,
        campaign_contract_hash=statistics.campaign_contract_hash,
        model_identity_hash=statistics.model_identity_hash,
        query_contract_hash=statistics.query_contract_hash,
        preprocessing_hash=statistics.preprocessing_hash,
        current_mask_hash=statistics.current_mask_hash,
        raw_logit_stream_source_hash=statistics.raw_logit_stream_source_hash,
        sampling_law_hash=statistics.sampling_law_hash,
        proposal_generation=statistics.proposal_generation,
        route_count=route_count,
        route_axis_hash=statistics.route_axis_hash,
        stratum_names=statistics.stratum_names,
        stratum_counts=statistics.stratum_counts,
        population_stratum_counts=statistics.population_stratum_counts,
        sealed_population_order_hashes=statistics.sealed_population_order_hashes,
        observed_context_ids_by_stratum=statistics.observed_context_ids_by_stratum,
        observed_draw_ids_by_stratum=statistics.observed_draw_ids_by_stratum,
        class_balanced_stratum_weights=(0.5, 0.5),
        look_index=statistics.look_index,
        prior_confidence_box_receipt_hash=prior_hash,
        prior_statistics_state_hash=statistics.prior_state_hash,
        class_balanced_absolute_point=class_point,
        current_class_balanced_absolute_lower=class_current_lower,
        current_class_balanced_absolute_upper=class_current_upper,
        cumulative_class_balanced_absolute_lower=cumulative_lower,
        cumulative_class_balanced_absolute_upper=cumulative_upper,
        trajectory_alpha_ledger_hash=base_design.trajectory_alpha_ledger_hash,
        trajectory_alpha_ledger_generation=(
            base_design.trajectory_alpha_ledger_generation
        ),
        trajectory_familywise_alpha=base_design.trajectory_familywise_alpha,
        trajectory_alpha_committed_before_generation=(
            base_design.trajectory_alpha_committed_before_generation
        ),
        generation_alpha_allocation=base_design.generation_alpha_allocation,
        registered_look_stratum_counts=base_design.registered_look_stratum_counts,
        alpha_by_look=base_design.alpha_by_look,
        total_registered_alpha=base_design.total_allocated_alpha,
        look_alpha=look_alpha,
        cumulative_alpha_spent=cumulative_alpha,
        remaining_generation_alpha_after_look=remaining_generation,
        unallocated_generation_alpha_reserve=unallocated_generation,
        remaining_trajectory_alpha_after_generation_allocation=remaining_trajectory,
        per_cell_combined_two_sided_alpha=per_cell_alpha,
        per_cell_hoeffding_serfling_two_sided_alpha=hs_alpha,
        per_cell_empirical_bernstein_two_sided_alpha=eb_alpha,
        empirical_bernstein_theorem_delta=eb_delta,
        hoeffding_serfling_log_factor=hs_log,
        empirical_bernstein_log_factor=eb_log,
        hoeffding_serfling_finite_population_corrections=(
            hs_corrections[0],
            hs_corrections[1],
        ),
        hoeffding_serfling_radius_multipliers_by_stratum=(
            hs_multipliers[0],
            hs_multipliers[1],
        ),
        empirical_bernstein_rho_by_stratum=(eb_rhos[0], eb_rhos[1]),
        empirical_bernstein_rho_branches_by_stratum=(
            eb_branches[0],
            eb_branches[1],
        ),
        empirical_bernstein_sigma_radius_multipliers_by_stratum=(
            eb_sigma_multipliers[0],
            eb_sigma_multipliers[1],
        ),
        empirical_bernstein_range_radius_multipliers_by_stratum=(
            eb_range_multipliers[0],
            eb_range_multipliers[1],
        ),
        census_by_stratum=(census_by_stratum[0], census_by_stratum[1]),
        covered_estimand_count=1,
        covered_estimands=base_design.covered_estimands,
        estimand_weighting=base_design.estimand_weighting,
        covered_stratum_count=2,
        covered_route_count=route_count,
        covered_two_sided_statements_at_look=cells_at_look,
        registered_look_count=base_design.registered_look_count,
        covered_two_sided_statement_looks=(
            cells_at_look * base_design.registered_look_count
        ),
        sampling_law_kind=base_design.sampling_law_kind,
        trajectory_alpha_policy_kind=base_design.trajectory_alpha_policy_kind,
    )
    result.assert_prior_transition(
        prior_confidence_box,
        expected_prior_hash=expected_prior_confidence_box_hash,
    )
    return result


def corrected_attribution_confidence_box(
    *,
    statistics: V1StratifiedCorrectedAttributionMoments,
    envelope_certificate: V1CorrectedAttributionEnvelopeCertificate,
    design: V1GroupSequentialSerflingDesign,
    expected_statistics_hash: str | None = None,
    expected_envelope_certificate_hash: str | None = None,
    expected_design_hash: str | None = None,
) -> V1CorrectedAttributionConfidenceBoxReceipt:
    """Build the registered simultaneous Hoeffding--Serfling confidence box."""

    if type(statistics) is not V1StratifiedCorrectedAttributionMoments:
        raise TypeError("statistics must be canonical corrected-attribution moments")
    if type(envelope_certificate) is not V1CorrectedAttributionEnvelopeCertificate:
        raise TypeError("envelope_certificate must be the canonical certificate")
    if type(design) is not V1GroupSequentialSerflingDesign:
        raise TypeError("design must be the canonical group-sequential design")
    state_hash = statistics.assert_integrity(expected_statistics_hash)
    certificate_hash = envelope_certificate.assert_integrity(
        expected_envelope_certificate_hash
    )
    design_hash = design.assert_integrity(expected_design_hash)

    coordinate_names = (
        "campaign_contract_hash",
        "model_identity_hash",
        "query_contract_hash",
        "preprocessing_hash",
        "current_mask_hash",
        "raw_logit_stream_source_hash",
        "sampling_law_hash",
        "proposal_generation",
        "route_count",
        "route_axis_hash",
        "stratum_names",
        "population_stratum_counts",
        "sealed_population_order_hashes",
    )
    for name in coordinate_names:
        values = (
            getattr(statistics, name),
            getattr(envelope_certificate, name),
            getattr(design, name),
        )
        if not (values[0] == values[1] == values[2]):
            raise ValueError(f"statistics, envelope, and design disagree on {name}")
    if statistics.look_index > design.registered_look_count:
        raise ValueError("statistics name an unregistered sequential look")
    scheduled_counts = design.registered_look_stratum_counts[statistics.look_index - 1]
    if statistics.stratum_counts != scheduled_counts:
        raise ValueError("statistics are off the predeclared look schedule")

    route_count = statistics.route_count
    envelope = envelope_certificate.absolute_upper_envelope_by_stratum
    signed_point = statistics.signed_mean_by_stratum
    absolute_point = statistics.absolute_mean_by_stratum
    tolerance = 128.0 * np.finfo(np.float64).eps * np.maximum(1.0, envelope)
    if np.any(np.abs(signed_point) > envelope + tolerance) or np.any(
        absolute_point > envelope + tolerance
    ):
        raise ValueError(
            "the analytic certificate does not enclose the observed point means"
        )

    look_alpha = design.alpha_by_look[statistics.look_index - 1]
    estimand_count = 1
    stratum_count = 2
    cells_at_look = estimand_count * stratum_count * route_count
    deterministic_census_look = (
        statistics.is_finite_population_census and look_alpha == 0.0
    )
    if deterministic_census_look:
        per_cell_two_sided_alpha = 0.0
        per_cell_one_sided_tail_alpha = 0.0
        log_factor = 0.0
    else:
        per_cell_two_sided_alpha = look_alpha / cells_at_look
        per_cell_one_sided_tail_alpha = 0.5 * per_cell_two_sided_alpha
        log_factor = math.log(2.0 / per_cell_two_sided_alpha)

    finite_corrections: list[float] = []
    radius_multipliers: list[float] = []
    class_absolute_lower = np.zeros(route_count, dtype=_LE_F64)
    class_absolute_upper = np.zeros(route_count, dtype=_LE_F64)
    scratch = np.empty(route_count, dtype=_LE_F64)
    for stratum, (count, population) in enumerate(
        zip(statistics.stratum_counts, statistics.population_stratum_counts)
    ):
        if count == population:
            finite_corrections.append(0.0)
            scale = 0.0
        else:
            correction = 1.0 - (count - 1.0) / population
            finite_corrections.append(correction)
            scale = math.sqrt(correction * log_factor / (2.0 * count))
        radius_multipliers.append(scale)

        # Materialize only one scratch route vector.  Durable state retains
        # the two scalar multipliers and the three class-balanced vectors;
        # per-stratum boxes are exactly reconstructible from state+certificate.
        np.multiply(envelope[stratum], scale, out=scratch)
        np.subtract(absolute_point[stratum], scratch, out=scratch)
        np.maximum(scratch, 0.0, out=scratch)
        scratch *= 0.5
        class_absolute_lower += scratch

        np.multiply(envelope[stratum], scale, out=scratch)
        np.add(absolute_point[stratum], scratch, out=scratch)
        np.minimum(scratch, envelope[stratum], out=scratch)
        scratch *= 0.5
        class_absolute_upper += scratch

    class_absolute_point = 0.5 * (absolute_point[0] + absolute_point[1])
    if statistics.is_finite_population_census:
        # Preserve bitwise zero macro width at joint census.
        class_absolute_lower = np.array(class_absolute_point, copy=True)
        class_absolute_upper = np.array(class_absolute_point, copy=True)

    cumulative_alpha = math.fsum(design.alpha_by_look[: statistics.look_index])
    remaining_generation = max(
        0.0, design.generation_alpha_allocation - cumulative_alpha
    )
    unallocated_generation = max(
        0.0, design.generation_alpha_allocation - design.total_allocated_alpha
    )
    remaining_trajectory = max(
        0.0,
        design.trajectory_familywise_alpha
        - design.trajectory_alpha_committed_before_generation
        - design.generation_alpha_allocation,
    )
    return V1CorrectedAttributionConfidenceBoxReceipt(
        state_hash=state_hash,
        envelope_certificate_hash=certificate_hash,
        design_hash=design_hash,
        campaign_contract_hash=statistics.campaign_contract_hash,
        model_identity_hash=statistics.model_identity_hash,
        query_contract_hash=statistics.query_contract_hash,
        preprocessing_hash=statistics.preprocessing_hash,
        current_mask_hash=statistics.current_mask_hash,
        raw_logit_stream_source_hash=statistics.raw_logit_stream_source_hash,
        sampling_law_hash=statistics.sampling_law_hash,
        proposal_generation=statistics.proposal_generation,
        route_count=route_count,
        route_axis_hash=statistics.route_axis_hash,
        stratum_names=statistics.stratum_names,
        stratum_counts=statistics.stratum_counts,
        population_stratum_counts=statistics.population_stratum_counts,
        sealed_population_order_hashes=statistics.sealed_population_order_hashes,
        observed_context_ids_by_stratum=(statistics.observed_context_ids_by_stratum),
        observed_draw_ids_by_stratum=statistics.observed_draw_ids_by_stratum,
        class_balanced_stratum_weights=(0.5, 0.5),
        look_index=statistics.look_index,
        absolute_radius_multipliers_by_stratum=(
            radius_multipliers[0],
            radius_multipliers[1],
        ),
        class_balanced_absolute_point=class_absolute_point,
        class_balanced_absolute_lower=class_absolute_lower,
        class_balanced_absolute_upper=class_absolute_upper,
        trajectory_alpha_ledger_hash=design.trajectory_alpha_ledger_hash,
        trajectory_alpha_ledger_generation=(design.trajectory_alpha_ledger_generation),
        trajectory_familywise_alpha=design.trajectory_familywise_alpha,
        trajectory_alpha_committed_before_generation=(
            design.trajectory_alpha_committed_before_generation
        ),
        generation_alpha_allocation=design.generation_alpha_allocation,
        registered_look_stratum_counts=design.registered_look_stratum_counts,
        alpha_by_look=design.alpha_by_look,
        total_registered_alpha=design.total_allocated_alpha,
        look_alpha=look_alpha,
        cumulative_alpha_spent=cumulative_alpha,
        remaining_generation_alpha_after_look=remaining_generation,
        unallocated_generation_alpha_reserve=unallocated_generation,
        remaining_trajectory_alpha_after_generation_allocation=(remaining_trajectory),
        per_cell_two_sided_alpha=per_cell_two_sided_alpha,
        per_cell_one_sided_tail_alpha=per_cell_one_sided_tail_alpha,
        serfling_log_factor=log_factor,
        finite_population_corrections=(
            finite_corrections[0],
            finite_corrections[1],
        ),
        covered_estimand_count=estimand_count,
        covered_estimands=design.covered_estimands,
        estimand_weighting=design.estimand_weighting,
        covered_stratum_count=stratum_count,
        covered_route_count=route_count,
        covered_two_sided_statements_at_look=cells_at_look,
        registered_look_count=design.registered_look_count,
        covered_two_sided_statement_looks=(
            cells_at_look * design.registered_look_count
        ),
        sampling_law_kind=design.sampling_law_kind,
        trajectory_alpha_policy_kind=design.trajectory_alpha_policy_kind,
    )


def _rebuild_moments(
    payload: dict[str, Any], expected_hash: str
) -> V1StratifiedCorrectedAttributionMoments:
    result = V1StratifiedCorrectedAttributionMoments(**payload)
    result.assert_integrity(expected_hash)
    return result


def _rebuild_envelope(
    payload: dict[str, Any], expected_hash: str
) -> V1CorrectedAttributionEnvelopeCertificate:
    result = V1CorrectedAttributionEnvelopeCertificate(**payload)
    result.assert_integrity(expected_hash)
    return result


def _rebuild_design(
    payload: dict[str, Any], expected_hash: str
) -> V1GroupSequentialSerflingDesign:
    result = V1GroupSequentialSerflingDesign(**payload)
    result.assert_integrity(expected_hash)
    return result


def _rebuild_hybrid_design(
    serfling_design: V1GroupSequentialSerflingDesign,
    schema: str,
    expected_hash: str,
) -> V1GroupSequentialEmpiricalBernsteinSerflingDesign:
    result = V1GroupSequentialEmpiricalBernsteinSerflingDesign(
        serfling_design=serfling_design,
        schema=schema,
    )
    result.assert_integrity(expected_hash)
    return result


def _rebuild_box(
    payload: dict[str, Any], expected_hash: str
) -> V1CorrectedAttributionConfidenceBoxReceipt:
    result = V1CorrectedAttributionConfidenceBoxReceipt(**payload)
    result.assert_integrity(expected_hash)
    return result


def _rebuild_box_v2(
    payload: dict[str, Any], expected_hash: str
) -> V1CorrectedAttributionConfidenceBoxV2Receipt:
    result = V1CorrectedAttributionConfidenceBoxV2Receipt(**payload)
    result.assert_integrity(expected_hash)
    return result


def _rebuild_action_family(
    payload: dict[str, Any], expected_hash: str
) -> V1CorrectedAttributionActionFamily:
    result = V1CorrectedAttributionActionFamily(**payload)
    result.assert_integrity(expected_hash)
    return result


def _rebuild_action_regret_request(
    payload: dict[str, Any], expected_hash: str
) -> V1CorrectedAttributionActionRegretRequest:
    result = V1CorrectedAttributionActionRegretRequest(**payload)
    result.assert_integrity(expected_hash)
    return result


def _rebuild_zero_regret_exchange_certificate(
    request: V1CorrectedAttributionActionRegretRequest,
    schema: str,
    expected_hash: str,
) -> V1CorrectedAttributionZeroRegretExchangeCertificate:
    result = V1CorrectedAttributionZeroRegretExchangeCertificate(
        request=request,
        schema=schema,
    )
    result.assert_integrity(expected_hash)
    return result


def _rebuild_action_regret_receipt(
    request: V1CorrectedAttributionActionRegretRequest,
    schema: str,
    expected_hash: str,
) -> V1CorrectedAttributionActionRegretReceipt:
    result = V1CorrectedAttributionActionRegretReceipt(
        request=request,
        schema=schema,
    )
    result.assert_integrity(expected_hash)
    return result


__all__ = [
    "ABSOLUTE_SALIENCE_ESTIMAND",
    "ACTION_PROPOSAL_CURRENT_POINT_COMPOSITION",
    "ACTION_REGRET_V1_BOX_SEMANTICS",
    "ACTION_REGRET_V2_BOX_SEMANTICS",
    "ANALYTIC_ENVELOPE_DERIVATION_KIND",
    "BARDENET_MAILLARD_EMPIRICAL_BERNSTEIN_SERFLING_INEQUALITY_ID",
    "CORRECTED_ATTRIBUTION_ACTION_FAMILY_SCHEMA",
    "CORRECTED_ATTRIBUTION_ACTION_REGRET_CURRENCY",
    "CORRECTED_ATTRIBUTION_ACTION_REGRET_RECEIPT_SCHEMA",
    "CORRECTED_ATTRIBUTION_ACTION_REGRET_RECEIPT_V3_SCHEMA",
    "CORRECTED_ATTRIBUTION_ACTION_REGRET_REQUEST_SCHEMA",
    "CORRECTED_ATTRIBUTION_ACTION_REGRET_REQUEST_V2_SCHEMA",
    "CORRECTED_ATTRIBUTION_ACTION_REGRET_THEOREM",
    "CORRECTED_ATTRIBUTION_ACTION_REGRET_TIE_RULE",
    "CORRECTED_ATTRIBUTION_ZERO_REGRET_EXCHANGE_CERTIFICATE_SCHEMA",
    "CORRECTED_ATTRIBUTION_ZERO_REGRET_EXCHANGE_THEOREM",
    "CORRECTED_ATTRIBUTION_CONFIDENCE_BOX_SCHEMA",
    "CORRECTED_ATTRIBUTION_CONFIDENCE_BOX_V2_SCHEMA",
    "CORRECTED_ATTRIBUTION_ENVELOPE_CERTIFICATE_SCHEMA",
    "CORRECTED_ATTRIBUTION_MOMENT_REPRESENTATION",
    "CORRECTED_ATTRIBUTION_PROPOSAL_ONLY_AUTHORITY",
    "EQUAL_STRATUM_MACRO_WEIGHTING",
    "EMPIRICAL_BERNSTEIN_BIASED_VARIANCE_CONVENTION",
    "EMPIRICAL_BERNSTEIN_KAPPA",
    "EMPIRICAL_BERNSTEIN_RHO_BRANCH_GT_HALF",
    "EMPIRICAL_BERNSTEIN_RHO_BRANCH_LE_HALF",
    "EXACT_QUOTA_PARTITION_MATROID_ACTION_FAMILY",
    "FIXED_TRAJECTORY_FWER_POLICY",
    "GLOBAL_FIXED_CARDINALITY_ACTION_FAMILY",
    "GROUP_SEQUENTIAL_SERFLING_DESIGN_SCHEMA",
    "GROUP_SEQUENTIAL_EMPIRICAL_BERNSTEIN_SERFLING_DESIGN_SCHEMA",
    "HYBRID_SERFLING_CONFIDENCE_BOX_INEQUALITY_ID",
    "HOEFFDING_SERFLING_INEQUALITY_ID",
    "SEALED_WITHOUT_REPLACEMENT_SAMPLING_LAW",
    "SERFLING_EVIDENCE_CEILING",
    "STRATIFIED_CORRECTED_ATTRIBUTION_MOMENTS_SCHEMA",
    "V1CorrectedAttributionActionFamily",
    "V1CorrectedAttributionActionRegretReceipt",
    "V1CorrectedAttributionActionRegretRequest",
    "V1CorrectedAttributionZeroRegretExchangeCertificate",
    "V1CorrectedAttributionConfidenceBoxReceipt",
    "V1CorrectedAttributionConfidenceBoxV2Receipt",
    "V1CorrectedAttributionEnvelopeCertificate",
    "V1GroupSequentialSerflingDesign",
    "V1GroupSequentialEmpiricalBernsteinSerflingDesign",
    "V1StratifiedCorrectedAttributionMoments",
    "certify_corrected_attribution_action_regret",
    "corrected_attribution_confidence_box",
    "corrected_attribution_confidence_box_v2",
    "deterministic_bottom_k_action",
]
