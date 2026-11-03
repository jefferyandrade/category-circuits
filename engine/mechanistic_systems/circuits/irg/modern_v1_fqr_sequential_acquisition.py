"""Bounded-memory sequential C-stream acquisition for modern-v1 F/Q/R.

The directional-secant contracts deliberately describe immutable observations
and terminal/census receipts, not a long-running acquisition state.  This
module supplies that missing state machine.  It consumes one sealed-prefix
``V1DirectionalSecantRecord`` at a time and retains only exact endpoint-level
sums, stratum counts, and a rolling evidence-chain hash.  It never retains a
context-by-endpoint record bank.

For each endpoint, stratum, and role ``F``, ``Q``, ``R_1/2``, ``R_1``, a
pre-outcome analytic bound supplies support ``[0, U]``.  A trajectory-reserved
family alpha is split once across cells.  Within a cell, the all-prefix
schedule

``w_n = N / (n (n + 1) (N - 1)),  n=1,...,N-1``

sums to one.  A one-sided Hoeffding--Serfling bound paid at ``alpha_cell*w_n``
is therefore simultaneous over every possible sealed prefix.  Arbitrary
inspection and optional stopping spend no new alpha.  At a full stratum census
the exact rational finite-population mean replaces the confidence bound and
the current bound spends zero alpha; the reserved family allocation remains
nonrecycling.

Sampling readiness is kept separate from deterministic FD-rank, KKT-solver,
and numerical-enclosure stability.  Neither this acquisition layer nor its
receipts can open calibration K, the paired BCE oracle O, or a mask commit.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from decimal import Decimal, ROUND_CEILING, localcontext
from fractions import Fraction
import math
from numbers import Integral, Real
from typing import Final, Literal, Sequence

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_directional_remainder import V1DirectionalSecantRecord
from .modern_v1_fqr_certification import V1FQRCertificationDesign
from .modern_v1_ladder_statistics import V1FrozenCandidateLadder
from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch


FQR_SEQUENTIAL_DESIGN_SCHEMA_V1: Final = (
    "partition_free_v1_fqr_sequential_acquisition_design_v1"
)
FQR_SEQUENTIAL_CELL_SUMMARY_SCHEMA_V1: Final = (
    "partition_free_v1_fqr_sequential_cell_summary_v1"
)
FQR_SEQUENTIAL_ENDPOINT_SUMMARY_SCHEMA_V1: Final = (
    "partition_free_v1_fqr_sequential_endpoint_summary_v1"
)
FQR_SEQUENTIAL_RECEIPT_SCHEMA_V1: Final = (
    "partition_free_v1_fqr_sequential_acquisition_receipt_v1"
)
FQR_SEQUENTIAL_DESIGN_SCHEMA: Final = (
    "partition_free_v1_fqr_sequential_acquisition_design_v2"
)
FQR_SEQUENTIAL_CELL_SUMMARY_SCHEMA: Final = (
    "partition_free_v1_fqr_sequential_cell_summary_v2"
)
FQR_SEQUENTIAL_ENDPOINT_SUMMARY_SCHEMA: Final = (
    "partition_free_v1_fqr_sequential_endpoint_summary_v2"
)
FQR_SEQUENTIAL_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_fqr_sequential_acquisition_receipt_v2"
)
FQR_SEQUENTIAL_ACCUMULATOR_SNAPSHOT_SCHEMA: Final = (
    "partition_free_v1_fqr_sequential_accumulator_snapshot_v1"
)
FQR_DETERMINISTIC_UNCERTAINTY_SCHEMA: Final = (
    "partition_free_v1_fqr_deterministic_uncertainty_receipt_v1"
)
FQR_DETERMINISTIC_UNCERTAINTY_V2_SCHEMA: Final = (
    "partition_free_v1_fqr_deterministic_uncertainty_receipt_v2"
)
FQR_DETERMINISTIC_UNCERTAINTY_PROPOSAL_V3_SCHEMA: Final = (
    "partition_free_v1_fqr_deterministic_uncertainty_receipt_proposal_v3"
)
_PROPOSAL_DETERMINISTIC_ISSUANCE_TOKEN: Final = object()
FQR_DECISION_READINESS_SCHEMA: Final = (
    "partition_free_v1_fqr_decision_readiness_receipt_v1"
)
FQR_SEQUENTIAL_AUTHORITY_SCOPE: Final = (
    "C_stream_FQR_summary_only_no_candidate_reordering_K_O_or_mask_commit_v1"
)
FQR_ALL_PREFIX_ALPHA_POLICY: Final = (
    "uniform_cell_then_inverse_triangular_all_prefix_nonrecycling_v1"
)
FQR_ONE_SIDED_METHOD: Final = (
    "all_prefix_bonferroni_one_sided_hoeffding_serfling_v1"
)
FQR_TWO_SIDED_METHOD: Final = (
    "all_prefix_bonferroni_two_sided_hoeffding_serfling_v2"
)
FQR_CENSUS_METHOD: Final = "exact_stratified_finite_population_census_v1"
FQR_SAMPLING_LAW: Final = (
    "sealed_independent_uniform_permutations_without_replacement_by_stratum_v1"
)

F_ROLE: Final = "corrected_F"
Q_ROLE: Final = "literal_Jacobian_Q"
R_HALF_ROLE: Final = "directional_R_eta_half"
R_ONE_ROLE: Final = "directional_R_eta_one"
FQR_SEQUENTIAL_ROLES: Final = (F_ROLE, Q_ROLE, R_HALF_ROLE, R_ONE_ROLE)

_DESIGN_PREFIX: Final = "modern-v1-fqr-sequential-design-"
_CELL_PREFIX: Final = "modern-v1-fqr-sequential-cell-"
_ENDPOINT_PREFIX: Final = "modern-v1-fqr-sequential-endpoint-"
_RECEIPT_PREFIX: Final = "modern-v1-fqr-sequential-receipt-"
_DETERMINISTIC_PREFIX: Final = "modern-v1-fqr-deterministic-uncertainty-"
_READINESS_PREFIX: Final = "modern-v1-fqr-decision-readiness-"
_CHAIN_PREFIX: Final = "modern-v1-fqr-observation-chain-"
_SNAPSHOT_PREFIX: Final = "modern-v1-fqr-sequential-accumulator-snapshot-"
_BINARY64_SCALE_EXPONENT: Final = 1074

SamplingDecision = Literal[
    "acquire_more_C",
    "sampling_stable_endpoint",
    "sampling_stable_no_endpoint",
    "sampling_numerical_refusal",
]
ReadinessStatus = Literal[
    "acquire_more_C",
    "resolve_fd_rank",
    "resolve_kkt_numerics",
    "resolve_numerical_enclosure",
    "ready_for_fqr_proposal_screening",
]


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _exact_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be one exact boolean")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be one exact nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be one real scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _finite_probability(value: object, *, name: str) -> float:
    result = _finite_nonnegative(value, name=name)
    if not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie strictly inside (0,1)")
    return result


def _immutable_f64(
    value: object,
    *,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    try:
        array = np.array(value, dtype=np.dtype("<f8"), order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a float64 array") from error
    if (
        array.shape != shape
        or not np.isfinite(array).all()
        or bool(np.any(array < 0.0))
    ):
        raise ValueError(f"{name} must be finite and nonnegative with shape {shape}")
    array[array == 0.0] = 0.0
    result = np.frombuffer(array.tobytes(order="C"), dtype=np.dtype("<f8")).reshape(
        shape
    )
    result.setflags(write=False)
    return result


def _axis_pairs(
    value: object,
    *,
    name: str,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if type(value) not in {tuple, list}:
        raise TypeError(f"{name} must be an ordered sequence")
    result: list[tuple[str, tuple[str, ...]]] = []
    for raw in value:
        if type(raw) not in {tuple, list} or len(raw) != 2:
            raise ValueError(f"{name} rows must be name/axis pairs")
        axis_name = _identity(raw[0], name=f"{name} axis name")
        if type(raw[1]) not in {tuple, list}:
            raise TypeError(f"{name} axis values must be an ordered sequence")
        axis = tuple(_identity(item, name=f"{name} item") for item in raw[1])
        if not axis or len(axis) != len(set(axis)):
            raise ValueError(f"{name} axes must be nonempty and unique")
        result.append((axis_name, axis))
    axes = tuple(result)
    flat = tuple(item for _, axis in axes for item in axis)
    if (
        not axes
        or len({axis_name for axis_name, _ in axes}) != len(axes)
        or len(flat) != len(set(flat))
    ):
        raise ValueError(f"{name} must have globally unique identities")
    return axes


def _binary64_scaled_integer(value: float) -> int:
    """Return the exact integer ``value * 2**1074`` for finite binary64."""

    observation = float(value)
    if not math.isfinite(observation):
        raise ValueError("F/Q/R observation must be finite")
    numerator, denominator = observation.as_integer_ratio()
    exponent = denominator.bit_length() - 1
    if denominator != 1 << exponent:  # pragma: no cover - binary64 invariant.
        raise RuntimeError("binary64 denominator is not a power of two")
    return numerator << (_BINARY64_SCALE_EXPONENT - exponent)


def _scaled_mean(scaled_sum: int, count: int) -> Fraction:
    if count <= 0:
        raise ValueError("an exact F/Q/R mean requires a positive count")
    return Fraction(int(scaled_sum), int(count) << _BINARY64_SCALE_EXPONENT)


def _fraction_to_float_outward(
    value: Fraction,
    *,
    direction: Literal["lower", "upper"],
) -> float:
    rounded = float(value)
    if not math.isfinite(rounded):
        raise ValueError("F/Q/R confidence endpoint overflowed binary64")
    exact_rounded = Fraction.from_float(rounded)
    if direction == "lower" and exact_rounded > value:
        rounded = math.nextafter(rounded, -math.inf)
    elif direction == "upper" and exact_rounded < value:
        rounded = math.nextafter(rounded, math.inf)
    if not math.isfinite(rounded):
        raise ValueError("outward F/Q/R confidence endpoint is nonfinite")
    return rounded


def _downward_fraction_to_probability(value: Fraction) -> float:
    if value <= 0:
        raise ValueError("registered alpha cell is not representably positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("registered alpha cell underflowed binary64")
    if Fraction.from_float(result) > value:
        result = math.nextafter(result, 0.0)
    if result <= 0.0:
        raise ValueError("registered alpha cell underflowed after outward rounding")
    return result


def _outward_weighted_sum(
    weights: Sequence[float],
    values: Sequence[float],
) -> float:
    exact = sum(
        (
            Fraction.from_float(float(weight))
            * Fraction.from_float(float(value))
            for weight, value in zip(weights, values, strict=True)
        ),
        Fraction(0, 1),
    )
    return _fraction_to_float_outward(exact, direction="upper")


def _downward_weighted_sum(
    weights: Sequence[float],
    values: Sequence[float],
) -> float:
    exact = sum(
        (
            Fraction.from_float(float(weight))
            * Fraction.from_float(float(value))
            for weight, value in zip(weights, values, strict=True)
        ),
        Fraction(0, 1),
    )
    return max(0.0, _fraction_to_float_outward(exact, direction="lower"))


def _outward_decimal(value: Decimal) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("derived F/Q/R burden overflowed binary64")
    if Decimal.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    if not math.isfinite(result):
        raise ValueError("outward F/Q/R burden is nonfinite")
    return result


def _downward_decimal(value: Decimal) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("derived F/Q/R burden overflowed binary64")
    if Decimal.from_float(result) > value:
        result = math.nextafter(result, -math.inf)
    if not math.isfinite(result):
        raise ValueError("downward F/Q/R burden is nonfinite")
    return max(0.0, result)


def _serfling_radius_upper(
    *,
    upper_bound: float,
    count: int,
    population: int,
    look_alpha: float,
) -> float:
    """High-precision upward one-sided Hoeffding--Serfling radius."""

    if not 0 < count < population:
        raise ValueError("partial Serfling radius requires 0 < n < N")
    bound = _finite_nonnegative(upper_bound, name="Serfling upper bound")
    alpha = _finite_probability(look_alpha, name="Serfling look alpha")
    if bound == 0.0:
        return 0.0
    correction = Fraction(population - count + 1, population)
    with localcontext() as context:
        context.prec = 100
        context.rounding = ROUND_CEILING
        bound_d = Decimal.from_float(bound)
        correction_d = Decimal(correction.numerator) / Decimal(correction.denominator)
        log_factor = (Decimal(1) / Decimal.from_float(alpha)).ln().next_plus()
        radius = (
            bound_d
            * (correction_d * log_factor / Decimal(2 * count)).sqrt()
        ).next_plus()
    result = float(radius)
    if not math.isfinite(result):
        return math.inf
    if Decimal.from_float(result) < radius:
        result = math.nextafter(result, math.inf)
    return math.nextafter(result, math.inf)


def _two_sided_serfling_radius_upper(
    *,
    upper_bound: float,
    count: int,
    population: int,
    look_alpha: float,
) -> float:
    """Upward radius for a simultaneous two-sided Hoeffding--Serfling look.

    ``look_alpha`` is the complete cell/look allocation.  The factor two pays
    both tails inside that already registered allocation, so adding lower
    bounds does not spend a second alpha family.
    """

    if not 0 < count < population:
        raise ValueError("partial Serfling radius requires 0 < n < N")
    bound = _finite_nonnegative(upper_bound, name="Serfling upper bound")
    alpha = _finite_probability(look_alpha, name="Serfling look alpha")
    if bound == 0.0:
        return 0.0
    correction = Fraction(population - count + 1, population)
    with localcontext() as context:
        context.prec = 100
        context.rounding = ROUND_CEILING
        bound_d = Decimal.from_float(bound)
        correction_d = Decimal(correction.numerator) / Decimal(correction.denominator)
        log_factor = (Decimal(2) / Decimal.from_float(alpha)).ln().next_plus()
        radius = (
            bound_d
            * (correction_d * log_factor / Decimal(2 * count)).sqrt()
        ).next_plus()
    result = float(radius)
    if not math.isfinite(result):
        return math.inf
    if Decimal.from_float(result) < radius:
        result = math.nextafter(result, math.inf)
    return math.nextafter(result, math.inf)


def _response_burden(*, eta: float, q_upper: float, r_upper: float) -> float:
    with localcontext() as context:
        context.prec = 100
        burden = (
            Decimal.from_float(eta)
            * Decimal.from_float(q_upper).sqrt()
            + Decimal.from_float(r_upper)
        )
    return _outward_decimal(burden)


def _response_burden_lower(*, eta: float, q_lower: float, r_lower: float) -> float:
    with localcontext() as context:
        context.prec = 100
        burden = (
            Decimal.from_float(eta)
            * Decimal.from_float(q_lower).sqrt()
            + Decimal.from_float(r_lower)
        )
    return _downward_decimal(burden)


def _registered_stratum_weights(
    value: object,
    *,
    names: tuple[str, ...],
) -> tuple[tuple[str, float], ...]:
    if type(value) not in {tuple, list}:
        raise TypeError("receipt stratum weights must be an ordered sequence")
    rows: list[tuple[str, float]] = []
    for raw in value:
        if type(raw) not in {tuple, list} or len(raw) != 2:
            raise ValueError("receipt stratum weight rows must be name/value pairs")
        rows.append((
            _identity(raw[0], name="receipt stratum weight id"),
            _finite_nonnegative(raw[1], name="receipt stratum weight"),
        ))
    result = tuple(rows)
    if tuple(name for name, _weight in result) != names:
        raise ValueError("receipt stratum weights differ from observed strata")
    if any(weight <= 0.0 for _name, weight in result):
        raise ValueError("receipt stratum weights must be positive")
    total = math.fsum(weight for _name, weight in result)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("receipt stratum weights must sum to one")
    return tuple((name, weight / total) for name, weight in result)


def _registered_look_alpha(
    *,
    cell_family_alpha: float,
    population: int,
    sample_size: int,
) -> float:
    count = _nonnegative_int(sample_size, name="receipt cell sample size")
    total = _positive_int(population, name="receipt cell population")
    if count > total:
        raise ValueError("receipt cell sample exceeds its finite population")
    if count == 0 or count == total:
        return 0.0
    weight = Fraction(total, count * (count + 1) * (total - 1))
    return _downward_fraction_to_probability(
        Fraction.from_float(cell_family_alpha) * weight
    )


@dataclass(frozen=True, slots=True)
class V1FQRSequentialAcquisitionDesign:
    """Pre-outcome C-stream bounds and nonrecycling trajectory alpha binding."""

    design_id: str
    fqr_design: V1FQRCertificationDesign = field(repr=False, compare=False)
    sealed_draw_ids_by_stratum: tuple[tuple[str, tuple[str, ...]], ...]
    analytic_f_upper_by_stratum_endpoint: np.ndarray = field(
        repr=False,
        compare=False,
    )
    analytic_q_upper_by_stratum_endpoint: np.ndarray = field(
        repr=False,
        compare=False,
    )
    analytic_r_upper_by_stratum_endpoint_and_dose: np.ndarray = field(
        repr=False,
        compare=False,
    )
    corrected_work_bound_source_hash: str
    literal_jacobian_bound_source_hash: str
    directional_remainder_bound_source_hash: str
    trajectory_alpha_ledger_hash: str
    trajectory_alpha_ledger_generation: int
    trajectory_familywise_alpha: float
    trajectory_alpha_committed_before_fqr: float
    fqr_family_alpha_allocation: float
    fd_rank_uncertainty_set_hash: str
    kkt_numerical_uncertainty_set_hash: str
    numerical_enclosure_policy_hash: str
    joint_fqr_family_registration_hash: str
    sampling_law: str = FQR_SAMPLING_LAW
    alpha_policy: str = FQR_ALL_PREFIX_ALPHA_POLICY
    alpha_allocation_nonrecycling: bool = True
    issued_before_C_outcomes: bool = True
    endpoint_order_mutable_after_C_outcomes: bool = False
    validation_outcomes_seen: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = FQR_SEQUENTIAL_AUTHORITY_SCOPE
    schema: str = FQR_SEQUENTIAL_DESIGN_SCHEMA
    cell_count: int = field(init=False)
    cell_family_alpha: float = field(init=False)
    registered_cell_alpha_total: float = field(init=False)
    _design_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "design_id",
            _identity(self.design_id, name="design id"),
        )
        if type(self.fqr_design) is not V1FQRCertificationDesign:
            raise TypeError("sequential acquisition requires the exact F/Q/R design")
        self.fqr_design.assert_integrity()
        secant = self.fqr_design.secant_design
        contexts = secant.certification_context_ids_by_stratum
        draws = _axis_pairs(self.sealed_draw_ids_by_stratum, name="sealed draw ledger")
        if (
            tuple(name for name, _ in draws)
            != tuple(name for name, _ in contexts)
            or tuple(len(axis) for _, axis in draws)
            != tuple(len(axis) for _, axis in contexts)
        ):
            raise ValueError("sealed draw axes differ from the C-stream populations")
        endpoints = self.fqr_design.endpoint_action_hashes
        shape = (len(contexts), len(endpoints))
        f_bound = _immutable_f64(
            self.analytic_f_upper_by_stratum_endpoint,
            shape=shape,
            name="analytic F bounds",
        )
        q_bound = _immutable_f64(
            self.analytic_q_upper_by_stratum_endpoint,
            shape=shape,
            name="analytic Q bounds",
        )
        r_bound = _immutable_f64(
            self.analytic_r_upper_by_stratum_endpoint_and_dose,
            shape=(*shape, 2),
            name="analytic R bounds",
        )
        for name in (
            "corrected_work_bound_source_hash",
            "literal_jacobian_bound_source_hash",
            "directional_remainder_bound_source_hash",
            "trajectory_alpha_ledger_hash",
            "fd_rank_uncertainty_set_hash",
            "kkt_numerical_uncertainty_set_hash",
            "numerical_enclosure_policy_hash",
            "joint_fqr_family_registration_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if (
            self.joint_fqr_family_registration_hash
            != self.fqr_design.joint_fqr_family_registration_hash
        ):
            raise ValueError("sequential acquisition names another F/Q/R family")
        generation = _nonnegative_int(
            self.trajectory_alpha_ledger_generation,
            name="trajectory alpha-ledger generation",
        )
        if generation != secant.ladder.proposal_generation:
            raise ValueError("trajectory alpha ledger belongs to another generation")
        familywise = _finite_probability(
            self.trajectory_familywise_alpha,
            name="trajectory familywise alpha",
        )
        committed = _finite_nonnegative(
            self.trajectory_alpha_committed_before_fqr,
            name="trajectory alpha committed before F/Q/R",
        )
        allocation = _finite_probability(
            self.fqr_family_alpha_allocation,
            name="F/Q/R family alpha allocation",
        )
        if (
            Fraction.from_float(committed) + Fraction.from_float(allocation)
            > Fraction.from_float(familywise)
        ):
            raise ValueError("F/Q/R allocation overspends the trajectory alpha ledger")
        cells = len(contexts) * len(endpoints) * len(FQR_SEQUENTIAL_ROLES)
        exact_cell_alpha = Fraction.from_float(allocation) / cells
        cell_alpha = _downward_fraction_to_probability(exact_cell_alpha)
        registered_total = math.fsum(cell_alpha for _ in range(cells))
        if registered_total > allocation:
            raise ValueError("rounded F/Q/R cell allocations overspend their family")
        if (
            self.sampling_law != FQR_SAMPLING_LAW
            or self.alpha_policy != FQR_ALL_PREFIX_ALPHA_POLICY
            or self.alpha_allocation_nonrecycling is not True
        ):
            raise ValueError("sequential F/Q/R sampling or alpha policy is invalid")
        if (
            self.issued_before_C_outcomes is not True
            or self.endpoint_order_mutable_after_C_outcomes is not False
            or self.validation_outcomes_seen is not False
        ):
            raise PermissionError("sequential F/Q/R design was post-selected")
        if any(
            getattr(self, name) is not False
            for name in (
                "calibration_K_open_authority",
                "paired_oracle_O_open_authority",
                "mask_commit_authority",
                "terminal_authority",
            )
        ):
            raise PermissionError("sequential F/Q/R design cannot open K/O or commit")
        if (
            self.authority_scope != FQR_SEQUENTIAL_AUTHORITY_SCOPE
            or self.schema
            not in {FQR_SEQUENTIAL_DESIGN_SCHEMA_V1, FQR_SEQUENTIAL_DESIGN_SCHEMA}
        ):
            raise ValueError("sequential F/Q/R design schema or authority is invalid")
        object.__setattr__(self, "sealed_draw_ids_by_stratum", draws)
        object.__setattr__(self, "analytic_f_upper_by_stratum_endpoint", f_bound)
        object.__setattr__(self, "analytic_q_upper_by_stratum_endpoint", q_bound)
        object.__setattr__(
            self,
            "analytic_r_upper_by_stratum_endpoint_and_dose",
            r_bound,
        )
        object.__setattr__(self, "trajectory_alpha_ledger_generation", generation)
        object.__setattr__(self, "trajectory_familywise_alpha", familywise)
        object.__setattr__(self, "trajectory_alpha_committed_before_fqr", committed)
        object.__setattr__(self, "fqr_family_alpha_allocation", allocation)
        object.__setattr__(self, "cell_count", cells)
        object.__setattr__(self, "cell_family_alpha", cell_alpha)
        object.__setattr__(self, "registered_cell_alpha_total", registered_total)
        self.assert_integrity()

    @property
    def endpoint_action_hashes(self) -> tuple[str, ...]:
        return self.fqr_design.endpoint_action_hashes

    @property
    def two_sided_decision_intervals(self) -> bool:
        return self.schema == FQR_SEQUENTIAL_DESIGN_SCHEMA

    @property
    def stratum_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, _ in (
                self.fqr_design.secant_design.certification_context_ids_by_stratum
            )
        )

    @property
    def population_sizes(self) -> tuple[int, ...]:
        return tuple(
            len(axis)
            for _, axis in (
                self.fqr_design.secant_design.certification_context_ids_by_stratum
            )
        )

    def _payload(self) -> dict[str, object]:
        payload = {
            "schema": self.schema,
            "design_id": self.design_id,
            "fqr_design_hash": self.fqr_design.design_hash,
            "secant_design_hash": self.fqr_design.secant_design.design_hash,
            "ladder_hash": self.fqr_design.ladder_hash,
            "chart_identity_hash": self.fqr_design.chart_identity_hash,
            "endpoint_action_hashes": self.endpoint_action_hashes,
            "certification_context_ids_by_stratum": (
                self.fqr_design.secant_design.certification_context_ids_by_stratum
            ),
            "sealed_draw_ids_by_stratum": self.sealed_draw_ids_by_stratum,
            "stratum_weights": self.fqr_design.secant_design.stratum_weights,
            "analytic_f_upper_by_stratum_endpoint": (
                self.analytic_f_upper_by_stratum_endpoint
            ),
            "analytic_q_upper_by_stratum_endpoint": (
                self.analytic_q_upper_by_stratum_endpoint
            ),
            "analytic_r_upper_by_stratum_endpoint_and_dose": (
                self.analytic_r_upper_by_stratum_endpoint_and_dose
            ),
            "corrected_work_bound_source_hash": self.corrected_work_bound_source_hash,
            "literal_jacobian_bound_source_hash": (
                self.literal_jacobian_bound_source_hash
            ),
            "directional_remainder_bound_source_hash": (
                self.directional_remainder_bound_source_hash
            ),
            "trajectory_alpha_ledger_hash": self.trajectory_alpha_ledger_hash,
            "trajectory_alpha_ledger_generation": (
                self.trajectory_alpha_ledger_generation
            ),
            "trajectory_familywise_alpha": self.trajectory_familywise_alpha,
            "trajectory_alpha_committed_before_fqr": (
                self.trajectory_alpha_committed_before_fqr
            ),
            "fqr_family_alpha_allocation": self.fqr_family_alpha_allocation,
            "cell_count": self.cell_count,
            "cell_family_alpha": self.cell_family_alpha,
            "registered_cell_alpha_total": self.registered_cell_alpha_total,
            "fd_rank_uncertainty_set_hash": self.fd_rank_uncertainty_set_hash,
            "kkt_numerical_uncertainty_set_hash": (
                self.kkt_numerical_uncertainty_set_hash
            ),
            "numerical_enclosure_policy_hash": self.numerical_enclosure_policy_hash,
            "joint_fqr_family_registration_hash": (
                self.joint_fqr_family_registration_hash
            ),
            "sampling_law": self.sampling_law,
            "alpha_policy": self.alpha_policy,
            "alpha_allocation_nonrecycling": self.alpha_allocation_nonrecycling,
            "issued_before_C_outcomes": self.issued_before_C_outcomes,
            "endpoint_order_mutable_after_C_outcomes": (
                self.endpoint_order_mutable_after_C_outcomes
            ),
            "validation_outcomes_seen": self.validation_outcomes_seen,
            "calibration_K_open_authority": self.calibration_K_open_authority,
            "paired_oracle_O_open_authority": self.paired_oracle_O_open_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "authority_scope": self.authority_scope,
        }
        if self.schema == FQR_SEQUENTIAL_DESIGN_SCHEMA:
            payload["two_sided_decision_intervals"] = True
            payload["partial_refutation_authority"] = True
        return payload

    @property
    def design_hash(self) -> str:
        cached = self._design_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.fqr_design.assert_integrity()
        observed = content_hash(self._payload(), prefix=_DESIGN_PREFIX)
        if self._design_hash is not None and observed != self._design_hash:
            raise ValueError("sequential F/Q/R design was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("sequential F/Q/R design hash does not match")
        object.__setattr__(self, "_design_hash", observed)
        return observed

    def look_alpha(self, *, stratum_index: int, sample_size: int) -> float:
        index = _nonnegative_int(stratum_index, name="stratum index")
        if index >= len(self.population_sizes):
            raise ValueError("stratum index left the registered axis")
        count = _positive_int(sample_size, name="sample size")
        population = self.population_sizes[index]
        if count > population:
            raise ValueError("sample size exceeds its finite population")
        if count == population:
            return 0.0
        weight = Fraction(population, count * (count + 1) * (population - 1))
        return _downward_fraction_to_probability(
            Fraction.from_float(self.cell_family_alpha) * weight
        )


@dataclass(frozen=True, slots=True)
class V1FQRSequentialCellSummary:
    endpoint_index: int
    action_hash: str
    role: str
    dose: float | None
    stratum_id: str
    population_size: int
    sample_size: int
    sample_mean_center: float | None
    lower: float = field(default=0.0, kw_only=True)
    upper: float
    value_upper_bound: float
    current_look_alpha: float
    cell_family_alpha: float
    exact_census_mean_numerator: int | None
    exact_census_mean_denominator: int | None
    full_census: bool
    empty: bool
    method: str
    observation_chain_hash: str
    schema: str = FQR_SEQUENTIAL_CELL_SUMMARY_SCHEMA
    _summary_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        endpoint = _positive_int(self.endpoint_index, name="endpoint index")
        action = _identity(self.action_hash, name="action hash")
        if self.role not in FQR_SEQUENTIAL_ROLES:
            raise ValueError("sequential F/Q/R cell has an unknown role")
        if self.role == R_HALF_ROLE and self.dose != 0.5:
            raise ValueError("R-half cell must bind eta=0.5")
        if self.role == R_ONE_ROLE and self.dose != 1.0:
            raise ValueError("R-one cell must bind eta=1")
        if self.role in {F_ROLE, Q_ROLE} and self.dose is not None:
            raise ValueError("F/Q cells do not carry a dose")
        stratum = _identity(self.stratum_id, name="stratum id")
        population = _positive_int(self.population_size, name="population size")
        count = _nonnegative_int(self.sample_size, name="sample size")
        if count > population:
            raise ValueError("cell sample exceeds its finite population")
        mean = self.sample_mean_center
        if mean is not None:
            mean = _finite_nonnegative(mean, name="sample mean center")
        lower = _finite_nonnegative(self.lower, name="cell lower")
        upper = _finite_nonnegative(self.upper, name="cell upper")
        value_bound = _finite_nonnegative(
            self.value_upper_bound,
            name="cell value upper bound",
        )
        alpha = _finite_nonnegative(
            self.current_look_alpha,
            name="current look alpha",
        )
        cell_alpha = _finite_probability(
            self.cell_family_alpha,
            name="cell family alpha",
        )
        full = _exact_bool(self.full_census, name="full census")
        empty = _exact_bool(self.empty, name="empty")
        if (
            lower > upper
            or upper > value_bound
            or (mean is not None and not lower <= mean <= upper)
        ):
            raise ValueError("cell summary exceeds its analytic support")
        if count == 0:
            if (
                mean is not None
                or lower != 0.0
                or alpha != 0.0
                or full
                or self.exact_census_mean_numerator is not None
                or self.exact_census_mean_denominator is not None
            ):
                raise ValueError("unobserved cell summary is inconsistent")
        elif count == population:
            if (
                not full
                or alpha != 0.0
                or self.method != FQR_CENSUS_METHOD
                or self.exact_census_mean_numerator is None
                or self.exact_census_mean_denominator is None
            ):
                raise ValueError("census cell must close with an exact rational mean")
            numerator = int(self.exact_census_mean_numerator)
            denominator = int(self.exact_census_mean_denominator)
            if denominator <= 0 or numerator < 0:
                raise ValueError("exact census rational mean is invalid")
            exact_mean = Fraction(numerator, denominator)
            if (
                mean != float(exact_mean)
                or lower
                != _fraction_to_float_outward(exact_mean, direction="lower")
                or upper
                != _fraction_to_float_outward(exact_mean, direction="upper")
                or exact_mean > Fraction.from_float(value_bound)
            ):
                raise ValueError("census cell does not bind its exact rational mean")
        elif (
            full
            or not 0.0 < alpha <= cell_alpha
            or self.method
            != (
                FQR_TWO_SIDED_METHOD
                if self.schema == FQR_SEQUENTIAL_CELL_SUMMARY_SCHEMA
                else FQR_ONE_SIDED_METHOD
            )
            or self.exact_census_mean_numerator is not None
            or self.exact_census_mean_denominator is not None
        ):
            raise ValueError("partial cell confidence accounting is invalid")
        if (
            self.schema == FQR_SEQUENTIAL_CELL_SUMMARY_SCHEMA_V1
            and lower != 0.0
        ):
            raise ValueError("legacy one-sided cells cannot claim a lower bound")
        if empty is not False:
            raise ValueError("one-sided bounded F/Q/R cells cannot be empty")
        if self.schema not in {
            FQR_SEQUENTIAL_CELL_SUMMARY_SCHEMA_V1,
            FQR_SEQUENTIAL_CELL_SUMMARY_SCHEMA,
        }:
            raise ValueError("sequential F/Q/R cell schema is invalid")
        object.__setattr__(self, "endpoint_index", endpoint)
        object.__setattr__(self, "action_hash", action)
        object.__setattr__(self, "stratum_id", stratum)
        object.__setattr__(self, "population_size", population)
        object.__setattr__(self, "sample_size", count)
        object.__setattr__(self, "sample_mean_center", mean)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "value_upper_bound", value_bound)
        object.__setattr__(self, "current_look_alpha", alpha)
        object.__setattr__(self, "cell_family_alpha", cell_alpha)
        _identity(self.observation_chain_hash, name="observation chain hash")
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        payload = {
            name: getattr(self, name)
            for name in (
                "schema",
                "endpoint_index",
                "action_hash",
                "role",
                "dose",
                "stratum_id",
                "population_size",
                "sample_size",
                "sample_mean_center",
                "value_upper_bound",
                "current_look_alpha",
                "cell_family_alpha",
                "exact_census_mean_numerator",
                "exact_census_mean_denominator",
                "full_census",
                "empty",
                "method",
                "observation_chain_hash",
            )
        }
        if self.schema == FQR_SEQUENTIAL_CELL_SUMMARY_SCHEMA:
            payload["lower"] = self.lower
            payload["upper"] = self.upper
            payload["two_sided_decision_interval"] = True
        else:
            payload["upper"] = self.upper
        return payload

    @property
    def summary_hash(self) -> str:
        cached = self._summary_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_CELL_PREFIX)
        if self._summary_hash is not None and observed != self._summary_hash:
            raise ValueError("sequential F/Q/R cell summary was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("sequential F/Q/R cell-summary hash does not match")
        object.__setattr__(self, "_summary_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1FQRSequentialEndpointSummary:
    endpoint_index: int
    action_hash: str
    f_lower: float = field(default=0.0, kw_only=True)
    f_upper: float
    q_lower: float = field(default=0.0, kw_only=True)
    q_upper: float
    r_half_lower: float = field(default=0.0, kw_only=True)
    r_half_upper: float
    r_one_lower: float = field(default=0.0, kw_only=True)
    r_one_upper: float
    half_response_burden_lower: float = field(default=0.0, kw_only=True)
    half_response_burden_upper: float
    one_response_burden_lower: float = field(default=0.0, kw_only=True)
    one_response_burden_upper: float
    f_certified: bool
    half_certified: bool
    one_certified: bool
    sampling_certified: bool
    sampling_refuted: bool
    all_strata_censused: bool
    schema: str = FQR_SEQUENTIAL_ENDPOINT_SUMMARY_SCHEMA
    _summary_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        endpoint = _positive_int(self.endpoint_index, name="endpoint index")
        action = _identity(self.action_hash, name="action hash")
        for name in (
            "f_lower",
            "f_upper",
            "q_lower",
            "q_upper",
            "r_half_lower",
            "r_half_upper",
            "r_one_lower",
            "r_one_upper",
            "half_response_burden_lower",
            "half_response_burden_upper",
            "one_response_burden_lower",
            "one_response_burden_upper",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name=name),
            )
        for lower_name, upper_name in (
            ("f_lower", "f_upper"),
            ("q_lower", "q_upper"),
            ("r_half_lower", "r_half_upper"),
            ("r_one_lower", "r_one_upper"),
            ("half_response_burden_lower", "half_response_burden_upper"),
            ("one_response_burden_lower", "one_response_burden_upper"),
        ):
            if getattr(self, lower_name) > getattr(self, upper_name):
                raise ValueError("endpoint lower bound exceeds its upper bound")
        flags = (
            self.f_certified,
            self.half_certified,
            self.one_certified,
            self.sampling_certified,
            self.sampling_refuted,
            self.all_strata_censused,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("endpoint summary flags must be exact booleans")
        if self.sampling_certified != (
            self.f_certified and self.half_certified and self.one_certified
        ):
            raise ValueError("endpoint sampling certification omits an F/Q/R condition")
        if self.sampling_refuted != (
            not self.sampling_certified
            and (
                self.f_lower > 0.0
                or self.half_response_burden_lower > 0.0
                or self.one_response_burden_lower > 0.0
            )
        ):
            # Threshold comparisons are design-dependent and are replayed by
            # the acquisition receipt.  A refutation must at least be backed
            # by one nonzero simultaneous lower bound, while exact zero-budget
            # designs may still refute at a strict positive lower bound.
            if self.sampling_refuted:
                raise ValueError("endpoint refutation lacks a lower-bound witness")
        if self.sampling_certified and self.sampling_refuted:
            raise ValueError("an endpoint cannot certify and refute simultaneously")
        if self.schema not in {
            FQR_SEQUENTIAL_ENDPOINT_SUMMARY_SCHEMA_V1,
            FQR_SEQUENTIAL_ENDPOINT_SUMMARY_SCHEMA,
        }:
            raise ValueError("sequential F/Q/R endpoint schema is invalid")
        if self.schema == FQR_SEQUENTIAL_ENDPOINT_SUMMARY_SCHEMA_V1:
            lower_fields = (
                self.f_lower,
                self.q_lower,
                self.r_half_lower,
                self.r_one_lower,
                self.half_response_burden_lower,
                self.one_response_burden_lower,
            )
            if any(value != 0.0 for value in lower_fields):
                raise ValueError("legacy endpoint summaries cannot claim lower bounds")
            if self.sampling_refuted != (
                self.all_strata_censused and not self.sampling_certified
            ):
                raise ValueError("legacy endpoint refutation requires exact census")
        object.__setattr__(self, "endpoint_index", endpoint)
        object.__setattr__(self, "action_hash", action)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        payload = {
            name: getattr(self, name)
            for name in (
                "schema",
                "endpoint_index",
                "action_hash",
                "f_certified",
                "half_certified",
                "one_certified",
                "sampling_certified",
                "sampling_refuted",
                "all_strata_censused",
            )
        }
        for name in (
            "f_upper",
            "q_upper",
            "r_half_upper",
            "r_one_upper",
            "half_response_burden_upper",
            "one_response_burden_upper",
        ):
            payload[name] = getattr(self, name)
        if self.schema == FQR_SEQUENTIAL_ENDPOINT_SUMMARY_SCHEMA:
            for name in (
                "f_lower",
                "q_lower",
                "r_half_lower",
                "r_one_lower",
                "half_response_burden_lower",
                "one_response_burden_lower",
            ):
                payload[name] = getattr(self, name)
            payload["two_sided_decision_interval"] = True
        return payload

    @property
    def summary_hash(self) -> str:
        cached = self._summary_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_ENDPOINT_PREFIX)
        if self._summary_hash is not None and observed != self._summary_hash:
            raise ValueError("sequential F/Q/R endpoint summary was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("sequential F/Q/R endpoint-summary hash does not match")
        object.__setattr__(self, "_summary_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1FQRSequentialAcquisitionReceipt:
    design_hash: str
    fqr_design_hash: str
    secant_design_hash: str
    ladder_hash: str
    chart_identity_hash: str
    endpoint_action_hashes: tuple[str, ...]
    observed_counts_by_stratum: tuple[tuple[str, int], ...]
    next_expected_context_by_stratum: tuple[tuple[str, str | None], ...]
    next_expected_draw_by_stratum: tuple[tuple[str, str | None], ...]
    observed_prefix_hash: str
    observation_chain_hash: str
    cell_summaries: tuple[V1FQRSequentialCellSummary, ...]
    endpoint_summaries: tuple[V1FQRSequentialEndpointSummary, ...]
    sampling_decision: SamplingDecision
    selected_endpoint_index: int | None
    selected_action_hash: str | None
    sampling_ambiguity_present: bool
    trajectory_alpha_ledger_hash: str
    fqr_family_alpha_allocation: float
    cell_family_alpha: float
    registered_cell_alpha_total: float
    stratum_weights: tuple[tuple[str, float], ...]
    tau_f: float
    tau_l: float
    alpha_spent_by_receipt_inspection: float
    alpha_allocation_nonrecycling: bool
    retained_scalar_slots: int
    retained_context_ids: int
    retained_record_arrays: int
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = FQR_SEQUENTIAL_AUTHORITY_SCOPE
    schema: str = FQR_SEQUENTIAL_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "design_hash",
            "fqr_design_hash",
            "secant_design_hash",
            "ladder_hash",
            "chart_identity_hash",
            "observed_prefix_hash",
            "observation_chain_hash",
            "trajectory_alpha_ledger_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        endpoints = tuple(
            _identity(value, name="endpoint action hash")
            for value in self.endpoint_action_hashes
        )
        counts = tuple(
            (
                _identity(name, name="count stratum"),
                _nonnegative_int(value, name="count"),
            )
            for name, value in self.observed_counts_by_stratum
        )
        if not counts or len({name for name, _ in counts}) != len(counts):
            raise ValueError("sequential receipt count strata are invalid")
        next_contexts = tuple(
            (
                _identity(name, name="next-context stratum"),
                None
                if value is None
                else _identity(value, name="next expected context"),
            )
            for name, value in self.next_expected_context_by_stratum
        )
        next_draws = tuple(
            (
                _identity(name, name="next-draw stratum"),
                None if value is None else _identity(value, name="next expected draw"),
            )
            for name, value in self.next_expected_draw_by_stratum
        )
        if (
            tuple(name for name, _ in next_contexts)
            != tuple(name for name, _ in counts)
            or tuple(name for name, _ in next_draws)
            != tuple(name for name, _ in counts)
        ):
            raise ValueError("sequential receipt next-row axes differ from counts")
        weights = _registered_stratum_weights(
            self.stratum_weights,
            names=tuple(name for name, _ in counts),
        )
        tau_f = _finite_nonnegative(self.tau_f, name="receipt F budget")
        tau_l = _finite_nonnegative(self.tau_l, name="receipt response budget")
        cells = tuple(self.cell_summaries)
        endpoint_summaries = tuple(self.endpoint_summaries)
        if (
            not endpoints
            or len(endpoint_summaries) != len(endpoints)
            or any(type(item) is not V1FQRSequentialCellSummary for item in cells)
            or any(
                type(item) is not V1FQRSequentialEndpointSummary
                for item in endpoint_summaries
            )
        ):
            raise TypeError("sequential receipt summary axes are invalid")
        for item in (*cells, *endpoint_summaries):
            item.assert_integrity()
        legacy = self.schema == FQR_SEQUENTIAL_RECEIPT_SCHEMA_V1
        expected_cell_schema = (
            FQR_SEQUENTIAL_CELL_SUMMARY_SCHEMA_V1
            if legacy
            else FQR_SEQUENTIAL_CELL_SUMMARY_SCHEMA
        )
        expected_endpoint_schema = (
            FQR_SEQUENTIAL_ENDPOINT_SUMMARY_SCHEMA_V1
            if legacy
            else FQR_SEQUENTIAL_ENDPOINT_SUMMARY_SCHEMA
        )
        if any(item.schema != expected_cell_schema for item in cells) or any(
            item.schema != expected_endpoint_schema for item in endpoint_summaries
        ):
            raise ValueError("sequential receipt mixed legacy and two-sided summaries")
        expected_endpoint_axis = tuple(enumerate(endpoints, start=1))
        if tuple(
            (item.endpoint_index, item.action_hash) for item in endpoint_summaries
        ) != expected_endpoint_axis:
            raise ValueError("endpoint summaries reordered the frozen ladder")
        expected_cell_axis = tuple(
            (endpoint, action, role, stratum)
            for endpoint, action in expected_endpoint_axis
            for role in FQR_SEQUENTIAL_ROLES
            for stratum, _ in counts
        )
        if tuple(
            (item.endpoint_index, item.action_hash, item.role, item.stratum_id)
            for item in cells
        ) != expected_cell_axis:
            raise ValueError("cell summaries reordered or invented F/Q/R cells")
        family_alpha = _finite_probability(
            self.fqr_family_alpha_allocation,
            name="F/Q/R family alpha allocation",
        )
        cell_alpha = _finite_probability(
            self.cell_family_alpha,
            name="F/Q/R cell alpha",
        )
        registered_total = _finite_nonnegative(
            self.registered_cell_alpha_total,
            name="registered F/Q/R cell alpha",
        )
        if (
            registered_total > family_alpha
            or math.fsum(cell_alpha for _ in range(len(expected_cell_axis)))
            != registered_total
        ):
            raise ValueError("sequential receipt alpha-family accounting is invalid")
        count_by_stratum = dict(counts)
        by_cell = {
            (item.endpoint_index, item.role, item.stratum_id): item
            for item in cells
        }
        for item in cells:
            if (
                item.sample_size != count_by_stratum[item.stratum_id]
                or item.cell_family_alpha != cell_alpha
                or item.current_look_alpha
                != _registered_look_alpha(
                    cell_family_alpha=cell_alpha,
                    population=item.population_size,
                    sample_size=item.sample_size,
                )
                or item.observation_chain_hash != self.observation_chain_hash
            ):
                raise ValueError("cell summary is not a replay of the receipt prefix")
        weight_values = tuple(weight for _name, weight in weights)
        replayed_endpoint_summaries: list[V1FQRSequentialEndpointSummary] = []
        for endpoint, action in expected_endpoint_axis:
            role_lowers = {
                role: _downward_weighted_sum(
                    weight_values,
                    tuple(
                        by_cell[(endpoint, role, stratum)].lower
                        for stratum, _weight in weights
                    ),
                )
                for role in FQR_SEQUENTIAL_ROLES
            }
            role_uppers = {
                role: _outward_weighted_sum(
                    weight_values,
                    tuple(
                        by_cell[(endpoint, role, stratum)].upper
                        for stratum, _weight in weights
                    ),
                )
                for role in FQR_SEQUENTIAL_ROLES
            }
            half_burden_lower = _response_burden_lower(
                eta=0.5,
                q_lower=role_lowers[Q_ROLE],
                r_lower=role_lowers[R_HALF_ROLE],
            )
            half_burden_upper = _response_burden(
                eta=0.5,
                q_upper=role_uppers[Q_ROLE],
                r_upper=role_uppers[R_HALF_ROLE],
            )
            one_burden_lower = _response_burden_lower(
                eta=1.0,
                q_lower=role_lowers[Q_ROLE],
                r_lower=role_lowers[R_ONE_ROLE],
            )
            one_burden_upper = _response_burden(
                eta=1.0,
                q_upper=role_uppers[Q_ROLE],
                r_upper=role_uppers[R_ONE_ROLE],
            )
            f_certified = role_uppers[F_ROLE] <= tau_f
            half_certified = half_burden_upper <= 0.5 * tau_l
            one_certified = one_burden_upper <= tau_l
            certified = f_certified and half_certified and one_certified
            census = all(
                by_cell[(endpoint, F_ROLE, stratum)].full_census
                for stratum, _weight in weights
            )
            if legacy:
                refuted = bool(census and not certified)
            else:
                refuted = bool(
                    role_lowers[F_ROLE] > tau_f
                    or half_burden_lower > 0.5 * tau_l
                    or one_burden_lower > tau_l
                )
            replayed_endpoint_summaries.append(
                V1FQRSequentialEndpointSummary(
                    endpoint_index=endpoint,
                    action_hash=action,
                    f_lower=role_lowers[F_ROLE],
                    f_upper=role_uppers[F_ROLE],
                    q_lower=role_lowers[Q_ROLE],
                    q_upper=role_uppers[Q_ROLE],
                    r_half_lower=role_lowers[R_HALF_ROLE],
                    r_half_upper=role_uppers[R_HALF_ROLE],
                    r_one_lower=role_lowers[R_ONE_ROLE],
                    r_one_upper=role_uppers[R_ONE_ROLE],
                    half_response_burden_lower=half_burden_lower,
                    half_response_burden_upper=half_burden_upper,
                    one_response_burden_lower=one_burden_lower,
                    one_response_burden_upper=one_burden_upper,
                    f_certified=f_certified,
                    half_certified=half_certified,
                    one_certified=one_certified,
                    sampling_certified=certified,
                    sampling_refuted=refuted,
                    all_strata_censused=census,
                    schema=expected_endpoint_schema,
                )
            )
        if tuple(value.summary_hash for value in endpoint_summaries) != tuple(
            value.summary_hash for value in replayed_endpoint_summaries
        ):
            raise ValueError(
                "endpoint summaries are not the stratified replay of F/Q/R cells"
            )
        if self.sampling_decision not in {
            "acquire_more_C",
            "sampling_stable_endpoint",
            "sampling_stable_no_endpoint",
            "sampling_numerical_refusal",
        }:
            raise ValueError("sequential receipt sampling decision is invalid")
        expected_selection: tuple[int, str] | None = None
        for item in reversed(endpoint_summaries):
            if item.sampling_certified and all(
                larger.sampling_refuted
                for larger in endpoint_summaries[item.endpoint_index :]
            ):
                expected_selection = (item.endpoint_index, item.action_hash)
                break
        expected_decision: SamplingDecision
        if expected_selection is not None:
            expected_decision = "sampling_stable_endpoint"
        elif all(item.sampling_refuted for item in endpoint_summaries):
            expected_decision = "sampling_stable_no_endpoint"
        else:
            expected_decision = "acquire_more_C"
        if self.sampling_decision != expected_decision:
            raise ValueError("sampling decision differs from endpoint summaries")
        if self.sampling_decision == "sampling_stable_endpoint":
            if (
                self.selected_endpoint_index,
                self.selected_action_hash,
            ) != expected_selection:
                raise ValueError("stable sampling endpoint decision lost its selection")
        elif (
            self.selected_endpoint_index is not None
            or self.selected_action_hash is not None
        ):
            raise ValueError("nonselection sampling receipt cannot name an endpoint")
        ambiguity = self.sampling_decision == "acquire_more_C"
        if self.sampling_ambiguity_present is not ambiguity:
            raise ValueError("sampling ambiguity flag differs from its decision")
        if (
            _finite_nonnegative(
                self.alpha_spent_by_receipt_inspection,
                name="inspection alpha spend",
            )
            != 0.0
            or self.alpha_allocation_nonrecycling is not True
        ):
            raise ValueError("receipt inspection cannot spend or recycle alpha")
        retained = (
            _positive_int(self.retained_scalar_slots, name="retained scalar slots"),
            _nonnegative_int(self.retained_context_ids, name="retained context IDs"),
            _nonnegative_int(
                self.retained_record_arrays,
                name="retained record arrays",
            ),
        )
        expected_slots = len(expected_cell_axis) + len(counts) + 2
        if retained != (expected_slots, 0, 0):
            raise ValueError(
                "bounded sequential state retained context or record arrays"
            )
        if any(
            getattr(self, name) is not False
            for name in (
                "calibration_K_open_authority",
                "paired_oracle_O_open_authority",
                "mask_commit_authority",
                "terminal_authority",
            )
        ):
            raise PermissionError("sequential receipt cannot open K/O or commit")
        if (
            self.authority_scope != FQR_SEQUENTIAL_AUTHORITY_SCOPE
            or self.schema
            not in {FQR_SEQUENTIAL_RECEIPT_SCHEMA_V1, FQR_SEQUENTIAL_RECEIPT_SCHEMA}
        ):
            raise ValueError("sequential receipt schema or authority is invalid")
        object.__setattr__(self, "endpoint_action_hashes", endpoints)
        object.__setattr__(self, "observed_counts_by_stratum", counts)
        object.__setattr__(self, "next_expected_context_by_stratum", next_contexts)
        object.__setattr__(self, "next_expected_draw_by_stratum", next_draws)
        object.__setattr__(self, "stratum_weights", weights)
        object.__setattr__(self, "tau_f", tau_f)
        object.__setattr__(self, "tau_l", tau_l)
        object.__setattr__(self, "cell_summaries", cells)
        object.__setattr__(self, "endpoint_summaries", endpoint_summaries)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        payload = {
            "schema": self.schema,
            "design_hash": self.design_hash,
            "fqr_design_hash": self.fqr_design_hash,
            "secant_design_hash": self.secant_design_hash,
            "ladder_hash": self.ladder_hash,
            "chart_identity_hash": self.chart_identity_hash,
            "endpoint_action_hashes": self.endpoint_action_hashes,
            "observed_counts_by_stratum": self.observed_counts_by_stratum,
            "next_expected_context_by_stratum": self.next_expected_context_by_stratum,
            "next_expected_draw_by_stratum": self.next_expected_draw_by_stratum,
            "observed_prefix_hash": self.observed_prefix_hash,
            "observation_chain_hash": self.observation_chain_hash,
            "cell_summary_hashes": tuple(
                item.summary_hash for item in self.cell_summaries
            ),
            "endpoint_summary_hashes": tuple(
                item.summary_hash for item in self.endpoint_summaries
            ),
            "sampling_decision": self.sampling_decision,
            "selected_endpoint_index": self.selected_endpoint_index,
            "selected_action_hash": self.selected_action_hash,
            "sampling_ambiguity_present": self.sampling_ambiguity_present,
            "trajectory_alpha_ledger_hash": self.trajectory_alpha_ledger_hash,
            "fqr_family_alpha_allocation": self.fqr_family_alpha_allocation,
            "cell_family_alpha": self.cell_family_alpha,
            "registered_cell_alpha_total": self.registered_cell_alpha_total,
            "stratum_weights": self.stratum_weights,
            "tau_f": self.tau_f,
            "tau_l": self.tau_l,
            "alpha_spent_by_receipt_inspection": (
                self.alpha_spent_by_receipt_inspection
            ),
            "alpha_allocation_nonrecycling": self.alpha_allocation_nonrecycling,
            "retained_scalar_slots": self.retained_scalar_slots,
            "retained_context_ids": self.retained_context_ids,
            "retained_record_arrays": self.retained_record_arrays,
            "calibration_K_open_authority": self.calibration_K_open_authority,
            "paired_oracle_O_open_authority": self.paired_oracle_O_open_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "authority_scope": self.authority_scope,
        }
        if self.schema == FQR_SEQUENTIAL_RECEIPT_SCHEMA:
            payload["two_sided_decision_intervals"] = True
            payload["partial_refutation_authority"] = True
        return payload

    @property
    def receipt_hash(self) -> str:
        cached = self._receipt_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        for item in (*self.cell_summaries, *self.endpoint_summaries):
            item.assert_integrity()
        observed = content_hash(self._payload(), prefix=_RECEIPT_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("sequential F/Q/R receipt was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("sequential F/Q/R receipt hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1FQRDeterministicUncertaintyReceipt:
    acquisition_design_hash: str
    fqr_design_hash: str
    ladder_hash: str
    chart_identity_hash: str
    fd_rank_uncertainty_set_hash: str
    kkt_numerical_uncertainty_set_hash: str
    numerical_enclosure_policy_hash: str
    fd_rank_decision_stable: bool
    kkt_numerical_decision_stable: bool
    numerical_enclosure_valid: bool
    sampling_outcomes_used: bool = False
    alpha_spent: float = 0.0
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    fd_rank_decision_stability_certificate_hash: str | None = None
    proposal_freeze_certificate_hash: str | None = None
    proposal_freeze_graph_hash: str | None = None
    proposal_candidate_family_hash: str | None = None
    device_shell_packed_branch_binding_hash: str | None = None
    device_shell_source_branch_binding_hash: str | None = None
    proposal_branch: V1RelaxedKKTBranch | None = None
    candidate_family_frozen_before_independent_C: bool = False
    D_FD_KKT_composition_uncertainty_efficiency_only: bool = False
    independent_C_FQR_required: bool = False
    _proposal_freeze_issuance_token: InitVar[object | None] = None
    schema: str = FQR_DETERMINISTIC_UNCERTAINTY_SCHEMA
    _receipt_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self, _proposal_freeze_issuance_token: object | None) -> None:
        for name in (
            "acquisition_design_hash",
            "fqr_design_hash",
            "ladder_hash",
            "chart_identity_hash",
            "fd_rank_uncertainty_set_hash",
            "kkt_numerical_uncertainty_set_hash",
            "numerical_enclosure_policy_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if any(
            type(value) is not bool
            for value in (
                self.fd_rank_decision_stable,
                self.kkt_numerical_decision_stable,
                self.numerical_enclosure_valid,
                self.sampling_outcomes_used,
            )
        ):
            raise TypeError("deterministic uncertainty flags must be exact booleans")
        if self.sampling_outcomes_used:
            raise PermissionError(
                "deterministic uncertainty receipt used sampling outcomes"
            )
        if (
            _finite_nonnegative(
                self.alpha_spent,
                name="deterministic alpha spend",
            )
            != 0.0
        ):
            raise ValueError("deterministic uncertainty sets spend no alpha")
        if any(
            getattr(self, name) is not False
            for name in (
                "calibration_K_open_authority",
                "paired_oracle_O_open_authority",
                "mask_commit_authority",
            )
        ):
            raise PermissionError("deterministic uncertainty cannot open K/O or commit")
        certificate_hash = self.fd_rank_decision_stability_certificate_hash
        if certificate_hash is not None:
            certificate_hash = _identity(
                certificate_hash,
                name="FD-rank decision-stability certificate hash",
            )
            object.__setattr__(
                self,
                "fd_rank_decision_stability_certificate_hash",
                certificate_hash,
            )
        if self.schema not in {
            FQR_DETERMINISTIC_UNCERTAINTY_SCHEMA,
            FQR_DETERMINISTIC_UNCERTAINTY_V2_SCHEMA,
            FQR_DETERMINISTIC_UNCERTAINTY_PROPOSAL_V3_SCHEMA,
        }:
            raise ValueError("deterministic uncertainty schema is invalid")
        proposal_hash_fields = (
            "proposal_freeze_certificate_hash",
            "proposal_freeze_graph_hash",
            "proposal_candidate_family_hash",
            "device_shell_packed_branch_binding_hash",
            "device_shell_source_branch_binding_hash",
        )
        proposal_hashes = tuple(
            getattr(self, name) for name in proposal_hash_fields
        )
        proposal_flags = (
            self.candidate_family_frozen_before_independent_C,
            self.D_FD_KKT_composition_uncertainty_efficiency_only,
            self.independent_C_FQR_required,
        )
        if any(type(value) is not bool for value in proposal_flags):
            raise TypeError("proposal-only deterministic flags must be exact booleans")
        if self.schema == FQR_DETERMINISTIC_UNCERTAINTY_SCHEMA:
            if (
                certificate_hash is not None
                or any(value is not None for value in proposal_hashes)
                or self.proposal_branch is not None
                or any(proposal_flags)
            ):
                raise ValueError(
                    "legacy deterministic uncertainty cannot bind newer "
                    "stability or proposal-only fields"
                )
        elif self.schema == FQR_DETERMINISTIC_UNCERTAINTY_V2_SCHEMA:
            if certificate_hash is None or not self.fd_rank_decision_stable:
                raise PermissionError(
                    "canonical deterministic uncertainty requires a real stable "
                    "FD-rank certificate"
                )
            if (
                any(value is not None for value in proposal_hashes)
                or self.proposal_branch is not None
                or any(proposal_flags)
            ):
                raise ValueError(
                    "v2 stable-rank uncertainty cannot bind proposal-only fields"
                )
        else:
            if (
                _proposal_freeze_issuance_token
                is not _PROPOSAL_DETERMINISTIC_ISSUANCE_TOKEN
            ):
                raise PermissionError(
                    "proposal-only deterministic uncertainty requires its "
                    "registered exact-freeze issuer"
                )
            if certificate_hash is not None:
                raise ValueError(
                    "proposal-only deterministic uncertainty cannot claim an "
                    "FD-rank stability certificate"
                )
            if any(value is None for value in proposal_hashes):
                raise ValueError(
                    "proposal-only deterministic uncertainty requires every "
                    "proposal-freeze and branch-binding hash"
                )
            for name, value in zip(
                proposal_hash_fields, proposal_hashes, strict=True
            ):
                object.__setattr__(
                    self,
                    name,
                    _identity(value, name=name.replace("_", " ")),
                )
            if type(self.proposal_branch) is not V1RelaxedKKTBranch:
                raise TypeError(
                    "proposal-only deterministic uncertainty requires an exact branch"
                )
            if proposal_flags != (True, True, True):
                raise PermissionError(
                    "proposal-only deterministic uncertainty lost its frozen-family "
                    "independent-C semantics"
                )
            if self.fd_rank_decision_stable or self.kkt_numerical_decision_stable:
                raise PermissionError(
                    "proposal-only deterministic uncertainty must not impersonate "
                    "legacy FD-rank or KKT-composition stability"
                )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        payload = {
            name: getattr(self, name)
            for name in (
                "schema",
                "acquisition_design_hash",
                "fqr_design_hash",
                "ladder_hash",
                "chart_identity_hash",
                "fd_rank_uncertainty_set_hash",
                "kkt_numerical_uncertainty_set_hash",
                "numerical_enclosure_policy_hash",
                "fd_rank_decision_stable",
                "kkt_numerical_decision_stable",
                "numerical_enclosure_valid",
                "sampling_outcomes_used",
                "alpha_spent",
                "calibration_K_open_authority",
                "paired_oracle_O_open_authority",
                "mask_commit_authority",
            )
        }
        # The v1 payload is frozen.  The v2 carrier adds the certificate hash
        # without changing historical v1 hashes or wire bytes.
        if self.schema == FQR_DETERMINISTIC_UNCERTAINTY_V2_SCHEMA:
            payload["fd_rank_decision_stability_certificate_hash"] = (
                self.fd_rank_decision_stability_certificate_hash
            )
        elif self.schema == FQR_DETERMINISTIC_UNCERTAINTY_PROPOSAL_V3_SCHEMA:
            payload.update(
                {
                    "proposal_freeze_certificate_hash": (
                        self.proposal_freeze_certificate_hash
                    ),
                    "proposal_freeze_graph_hash": self.proposal_freeze_graph_hash,
                    "proposal_candidate_family_hash": (
                        self.proposal_candidate_family_hash
                    ),
                    "device_shell_packed_branch_binding_hash": (
                        self.device_shell_packed_branch_binding_hash
                    ),
                    "device_shell_source_branch_binding_hash": (
                        self.device_shell_source_branch_binding_hash
                    ),
                    "proposal_branch": self.proposal_branch,
                    "candidate_family_frozen_before_independent_C": (
                        self.candidate_family_frozen_before_independent_C
                    ),
                    "D_FD_KKT_composition_uncertainty_efficiency_only": (
                        self.D_FD_KKT_composition_uncertainty_efficiency_only
                    ),
                    "independent_C_FQR_required": self.independent_C_FQR_required,
                }
            )
        return payload

    @property
    def canonical_fd_rank_decision_stable(self) -> bool:
        return (
            self.schema == FQR_DETERMINISTIC_UNCERTAINTY_V2_SCHEMA
            and self.fd_rank_decision_stable
            and self.fd_rank_decision_stability_certificate_hash is not None
        )

    @property
    def resolved_for_independent_C(self) -> bool:
        """Whether deterministic composition uncertainty blocks this C decision.

        Legacy v1/v2 receipts resolve composition only through their declared
        FD/KKT stability path.  Proposal-only v3 instead resolves it by freezing
        the exact candidate family before independent C: D/FD/KKT uncertainty
        may affect proposal efficiency, but it cannot change the already-frozen
        actions whose F/Q/R response is now measured independently.
        """

        if self.schema == FQR_DETERMINISTIC_UNCERTAINTY_PROPOSAL_V3_SCHEMA:
            return bool(
                not self.fd_rank_decision_stable
                and not self.kkt_numerical_decision_stable
                and self.fd_rank_decision_stability_certificate_hash is None
                and self.proposal_freeze_certificate_hash is not None
                and self.proposal_freeze_graph_hash is not None
                and self.proposal_candidate_family_hash is not None
                and self.device_shell_packed_branch_binding_hash is not None
                and self.device_shell_source_branch_binding_hash is not None
                and type(self.proposal_branch) is V1RelaxedKKTBranch
                and self.candidate_family_frozen_before_independent_C
                and self.D_FD_KKT_composition_uncertainty_efficiency_only
                and self.independent_C_FQR_required
            )
        if self.schema == FQR_DETERMINISTIC_UNCERTAINTY_V2_SCHEMA:
            return bool(
                self.canonical_fd_rank_decision_stable
                and self.kkt_numerical_decision_stable
            )
        return bool(
            self.fd_rank_decision_stable
            and self.kkt_numerical_decision_stable
        )

    @property
    def receipt_hash(self) -> str:
        cached = self._receipt_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_DETERMINISTIC_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("deterministic F/Q/R uncertainty receipt was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("deterministic uncertainty receipt hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


def issue_v1_fqr_deterministic_uncertainty_from_adaptive_rank(
    *,
    acquisition_design: V1FQRSequentialAcquisitionDesign,
    ladder: V1FrozenCandidateLadder,
    stability_certificate: object,
    kkt_numerical_decision_stable: bool,
    numerical_enclosure_valid: bool,
) -> V1FQRDeterministicUncertaintyReceipt:
    """Bind deterministic readiness to the live source-issued rank certificate.

    Importing the adaptive-rank type locally keeps the sequential statistics
    owner independent of the D executor.  Caller-authored booleans cannot
    impersonate FD stability: the v2 receipt always carries and revalidates the
    exact live certificate hash and chart coordinates.
    """

    from .modern_v1_fqr_adaptive_rank import (  # local: acyclic authority edge
        V1FQRAdaptiveRankStabilityCertificate,
    )

    if type(acquisition_design) is not V1FQRSequentialAcquisitionDesign:
        raise TypeError("adaptive deterministic receipt requires the exact C design")
    if type(ladder) is not V1FrozenCandidateLadder:
        raise TypeError("adaptive deterministic receipt requires the exact frozen ladder")
    if type(stability_certificate) is not V1FQRAdaptiveRankStabilityCertificate:
        raise TypeError("adaptive deterministic receipt requires the live rank certificate")
    if (
        type(kkt_numerical_decision_stable) is not bool
        or type(numerical_enclosure_valid) is not bool
    ):
        raise TypeError("deterministic numerical flags must be exact booleans")
    certificate = stability_certificate
    certificate.assert_integrity()
    acquisition_design.assert_integrity()
    ladder.assert_integrity()
    fqr = acquisition_design.fqr_design
    if (
        certificate.C_stream_open_authority is not True
        or certificate.uncertainty_set_hash
        != acquisition_design.fd_rank_uncertainty_set_hash
        or certificate.current_mask_hash != ladder.current_mask_hash
        or certificate.route_axis_hash != ladder.route_axis_hash
        or certificate.route_count != ladder.route_count
        or certificate.proposal_generation != ladder.proposal_generation
        or fqr.ladder_hash != ladder.ladder_hash
    ):
        raise ValueError("adaptive-rank certificate is stale for the C design/ladder")
    if not kkt_numerical_decision_stable or not numerical_enclosure_valid:
        raise PermissionError(
            "canonical C cannot open while KKT numerics or enclosures are unresolved"
        )
    return V1FQRDeterministicUncertaintyReceipt(
        acquisition_design_hash=acquisition_design.design_hash,
        fqr_design_hash=fqr.design_hash,
        ladder_hash=ladder.ladder_hash,
        chart_identity_hash=fqr.chart_identity_hash,
        fd_rank_uncertainty_set_hash=(
            acquisition_design.fd_rank_uncertainty_set_hash
        ),
        kkt_numerical_uncertainty_set_hash=(
            acquisition_design.kkt_numerical_uncertainty_set_hash
        ),
        numerical_enclosure_policy_hash=(
            acquisition_design.numerical_enclosure_policy_hash
        ),
        fd_rank_decision_stable=True,
        kkt_numerical_decision_stable=True,
        numerical_enclosure_valid=True,
        fd_rank_decision_stability_certificate_hash=certificate.certificate_hash,
        schema=FQR_DETERMINISTIC_UNCERTAINTY_V2_SCHEMA,
    )


def issue_v1_fqr_deterministic_uncertainty_from_proposal_freeze(
    *,
    acquisition_design: V1FQRSequentialAcquisitionDesign,
    proposal_freeze: object,
    packed_branch_binding: object,
    numerical_enclosure_valid: bool,
) -> V1FQRDeterministicUncertaintyReceipt:
    """Bind a frozen proposal-only device-shell branch to independent C.

    This issuer makes no claim that FD rank or KKT composition was decision
    stable.  The exact candidate family and branch ladder were frozen before C,
    so those uncertainties affect only which proposals were offered and hence
    efficiency.  Independent C F/Q/R still evaluates the exact packed actions,
    and K/O remain required downstream.
    """

    from .modern_v1_fqr_device_shell_materialized_family import (
        V1FQRDeviceShellPackedBranchBinding,
    )
    from .modern_v1_sampled_kkt_device_shell_bridge import (
        V1SampledKKTDeviceShellProposalFreezeForC,
    )

    if type(acquisition_design) is not V1FQRSequentialAcquisitionDesign:
        raise TypeError("proposal deterministic receipt requires the exact C design")
    if type(proposal_freeze) is not V1SampledKKTDeviceShellProposalFreezeForC:
        raise TypeError("proposal deterministic receipt requires the exact freeze")
    if type(packed_branch_binding) is not V1FQRDeviceShellPackedBranchBinding:
        raise TypeError(
            "proposal deterministic receipt requires the exact packed branch binding"
        )
    if type(numerical_enclosure_valid) is not bool:
        raise TypeError("numerical enclosure validity must be one exact boolean")
    acquisition_design.assert_integrity()
    proposal_freeze.assert_integrity()
    packed_branch_binding.assert_integrity()
    graph = proposal_freeze.graph
    source_family = graph.branch_family
    branch = packed_branch_binding.branch
    expected_source = (
        source_family.spectral
        if branch is V1RelaxedKKTBranch.SPECTRAL
        else source_family.diagonal_l1
    )
    ladder = packed_branch_binding.frozen_ladder
    ladder.assert_integrity()
    fqr = acquisition_design.fqr_design
    secant_ladder = fqr.secant_design.ladder
    adaptive_source_freeze = (
        proposal_freeze.adaptive_execution_rank_freeze is not None
    )
    if (
        proposal_freeze.candidate_family_frozen_before_C is not True
        or proposal_freeze.D_sampling_fd_kkt_uncertainty_affects_efficiency_only
        is not True
        or proposal_freeze.independent_C_FQR_required is not True
        or proposal_freeze.candidate_specific_K_and_paired_O_required is not True
        or proposal_freeze.robust_kkt_sensitivity_required_for_C
        is not adaptive_source_freeze
        or proposal_freeze.C_direction_borrow_authority is not True
        or proposal_freeze.C_FQR_decision_authority is not False
        or proposal_freeze.calibration_K_open_authority is not False
        or proposal_freeze.paired_oracle_O_open_authority is not False
        or proposal_freeze.mask_commit_authority is not False
        or proposal_freeze.terminal_authority is not False
        or proposal_freeze.pareto_authority is not False
        or proposal_freeze.frozen_graph_hash != graph.graph_hash
        or proposal_freeze.frozen_candidate_family_hash
        != source_family.family_hash
        or graph.c_stream_outcomes_seen
        or packed_branch_binding.source_graph_hash != graph.graph_hash
        or packed_branch_binding.proposal_freeze_certificate_hash
        != proposal_freeze.certificate_hash
        or packed_branch_binding.source_binding.binding_hash
        != expected_source.binding_hash
        or packed_branch_binding.C_outcomes_seen
        or packed_branch_binding.route_order_reconstructed
        or packed_branch_binding.hard_global_optimality_claim
        or fqr.ladder_hash != ladder.ladder_hash
        or secant_ladder.ladder_hash != ladder.ladder_hash
        or secant_ladder.action_hashes != ladder.action_hashes
        or fqr.chart_identity_hash != packed_branch_binding.chart_identity_hash
        or ladder.current_mask_hash != packed_branch_binding.current_mask_hash
        or ladder.route_axis_hash != packed_branch_binding.route_axis_hash
        or ladder.route_count != packed_branch_binding.route_count
        or ladder.proposal_generation != packed_branch_binding.proposal_generation
    ):
        raise ValueError(
            "proposal freeze, packed branch, and independent-C design are stale"
        )
    return V1FQRDeterministicUncertaintyReceipt(
        acquisition_design_hash=acquisition_design.design_hash,
        fqr_design_hash=fqr.design_hash,
        ladder_hash=ladder.ladder_hash,
        chart_identity_hash=fqr.chart_identity_hash,
        fd_rank_uncertainty_set_hash=(
            acquisition_design.fd_rank_uncertainty_set_hash
        ),
        kkt_numerical_uncertainty_set_hash=(
            acquisition_design.kkt_numerical_uncertainty_set_hash
        ),
        numerical_enclosure_policy_hash=(
            acquisition_design.numerical_enclosure_policy_hash
        ),
        fd_rank_decision_stable=False,
        kkt_numerical_decision_stable=False,
        numerical_enclosure_valid=numerical_enclosure_valid,
        proposal_freeze_certificate_hash=proposal_freeze.certificate_hash,
        proposal_freeze_graph_hash=graph.graph_hash,
        proposal_candidate_family_hash=source_family.family_hash,
        device_shell_packed_branch_binding_hash=(
            packed_branch_binding.binding_hash
        ),
        device_shell_source_branch_binding_hash=(
            packed_branch_binding.source_binding.binding_hash
        ),
        proposal_branch=branch,
        candidate_family_frozen_before_independent_C=True,
        D_FD_KKT_composition_uncertainty_efficiency_only=True,
        independent_C_FQR_required=True,
        _proposal_freeze_issuance_token=(
            _PROPOSAL_DETERMINISTIC_ISSUANCE_TOKEN
        ),
        schema=FQR_DETERMINISTIC_UNCERTAINTY_PROPOSAL_V3_SCHEMA,
    )


@dataclass(frozen=True, slots=True)
class V1FQRDecisionReadinessReceipt:
    acquisition_receipt_hash: str
    deterministic_uncertainty_receipt_hash: str
    design_hash: str
    fqr_design_hash: str
    ladder_hash: str
    chart_identity_hash: str
    status: ReadinessStatus
    dominant_ambiguity: str | None
    selected_endpoint_index: int | None
    selected_action_hash: str | None
    sampling_resolved: bool
    fd_rank_resolved: bool
    kkt_numerics_resolved: bool
    numerical_enclosure_resolved: bool
    ready_for_fqr_proposal_screening: bool
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = FQR_SEQUENTIAL_AUTHORITY_SCOPE
    schema: str = FQR_DECISION_READINESS_SCHEMA
    _receipt_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "acquisition_receipt_hash",
            "deterministic_uncertainty_receipt_hash",
            "design_hash",
            "fqr_design_hash",
            "ladder_hash",
            "chart_identity_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if self.status not in {
            "acquire_more_C",
            "resolve_fd_rank",
            "resolve_kkt_numerics",
            "resolve_numerical_enclosure",
            "ready_for_fqr_proposal_screening",
        }:
            raise ValueError("F/Q/R readiness status is invalid")
        flags = (
            self.sampling_resolved,
            self.fd_rank_resolved,
            self.kkt_numerics_resolved,
            self.numerical_enclosure_resolved,
            self.ready_for_fqr_proposal_screening,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("F/Q/R readiness flags must be exact booleans")
        ready = all(flags[:-1])
        if self.ready_for_fqr_proposal_screening is not ready:
            raise ValueError("F/Q/R readiness omitted an uncertainty source")
        if ready != (self.status == "ready_for_fqr_proposal_screening"):
            raise ValueError("F/Q/R readiness status differs from its flags")
        expected_dominant = (
            None
            if ready
            else "sampling"
            if not self.sampling_resolved
            else "fd_rank"
            if not self.fd_rank_resolved
            else "kkt_numerics"
            if not self.kkt_numerics_resolved
            else "numerical_enclosure"
        )
        if self.dominant_ambiguity != expected_dominant:
            raise ValueError("F/Q/R dominant ambiguity is inconsistent")
        if any(
            getattr(self, name) is not False
            for name in (
                "calibration_K_open_authority",
                "paired_oracle_O_open_authority",
                "mask_commit_authority",
                "terminal_authority",
            )
        ):
            raise PermissionError("F/Q/R readiness cannot open K/O or commit")
        if (
            self.authority_scope != FQR_SEQUENTIAL_AUTHORITY_SCOPE
            or self.schema != FQR_DECISION_READINESS_SCHEMA
        ):
            raise ValueError("F/Q/R readiness schema or authority is invalid")
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "acquisition_receipt_hash",
                "deterministic_uncertainty_receipt_hash",
                "design_hash",
                "fqr_design_hash",
                "ladder_hash",
                "chart_identity_hash",
                "status",
                "dominant_ambiguity",
                "selected_endpoint_index",
                "selected_action_hash",
                "sampling_resolved",
                "fd_rank_resolved",
                "kkt_numerics_resolved",
                "numerical_enclosure_resolved",
                "ready_for_fqr_proposal_screening",
                "calibration_K_open_authority",
                "paired_oracle_O_open_authority",
                "mask_commit_authority",
                "terminal_authority",
                "authority_scope",
            )
        }

    @property
    def receipt_hash(self) -> str:
        cached = self._receipt_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_READINESS_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("F/Q/R readiness receipt was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("F/Q/R readiness receipt hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1FQRSequentialAccumulatorSnapshot:
    """Exact bounded-state checkpoint for one registered C-stream prefix.

    The snapshot stores only constant-cell integer sufficient statistics and
    causal coordinates.  It deliberately excludes context identities beyond
    those already frozen in the acquisition design, record objects, and raw
    observation arrays.  ``validate_against_design`` independently checks the
    finite-population prefixes and every exact sum against the design's
    pre-outcome analytic support before restoration.
    """

    acquisition_design_hash: str
    fqr_design_hash: str
    secant_design_hash: str
    ladder_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_count: int
    route_axis_hash: str
    endpoint_action_hashes: tuple[str, ...]
    stratum_names: tuple[str, ...]
    population_sizes: tuple[int, ...]
    role_axis: tuple[str, ...]
    observed_counts_by_stratum: tuple[tuple[str, int], ...]
    scaled_sums_by_stratum_endpoint_role: tuple[
        tuple[tuple[int, ...], ...], ...
    ]
    total_records_consumed: int
    observation_chain_hash: str
    observation_chain_design_hash: str
    observation_chain_record_count: int
    joint_fqr_family_registration_hash: str
    prior_transaction_commit_hash: str | None = None
    binary64_scale_exponent: int = _BINARY64_SCALE_EXPONENT
    retained_context_ids: int = 0
    retained_record_arrays: int = 0
    raw_secant_records_retained: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = FQR_SEQUENTIAL_AUTHORITY_SCOPE
    schema: str = FQR_SEQUENTIAL_ACCUMULATOR_SNAPSHOT_SCHEMA
    _snapshot_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "acquisition_design_hash",
            "fqr_design_hash",
            "secant_design_hash",
            "ladder_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "observation_chain_hash",
            "observation_chain_design_hash",
            "joint_fqr_family_registration_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        endpoints = tuple(
            _identity(value, name="snapshot endpoint action hash")
            for value in self.endpoint_action_hashes
        )
        strata = tuple(
            _identity(value, name="snapshot stratum") for value in self.stratum_names
        )
        populations = tuple(
            _positive_int(value, name="snapshot stratum population")
            for value in self.population_sizes
        )
        roles = tuple(_identity(value, name="snapshot role") for value in self.role_axis)
        counts = tuple(
            (
                _identity(name, name="snapshot count stratum"),
                _nonnegative_int(value, name="snapshot stratum count"),
            )
            for name, value in self.observed_counts_by_stratum
        )
        if (
            not endpoints
            or len(endpoints) != len(set(endpoints))
            or not strata
            or len(strata) != len(set(strata))
            or len(populations) != len(strata)
            or roles != FQR_SEQUENTIAL_ROLES
            or tuple(name for name, _ in counts) != strata
        ):
            raise ValueError("snapshot endpoint, stratum, or role axis is invalid")
        sums: list[tuple[tuple[int, ...], ...]] = []
        for raw_stratum in self.scaled_sums_by_stratum_endpoint_role:
            endpoint_rows: list[tuple[int, ...]] = []
            for raw_endpoint in raw_stratum:
                role_values = tuple(
                    _nonnegative_int(value, name="exact scaled observation sum")
                    for value in raw_endpoint
                )
                if len(role_values) != len(roles):
                    raise ValueError("snapshot role axis is ragged")
                endpoint_rows.append(role_values)
            if len(endpoint_rows) != len(endpoints):
                raise ValueError("snapshot endpoint axis is ragged")
            sums.append(tuple(endpoint_rows))
        normalized_sums = tuple(sums)
        if len(normalized_sums) != len(strata):
            raise ValueError("snapshot stratum sum axis is ragged")
        total = _nonnegative_int(
            self.total_records_consumed,
            name="snapshot total records",
        )
        chain_count = _nonnegative_int(
            self.observation_chain_record_count,
            name="snapshot chain record count",
        )
        if (
            any(count > population for (_, count), population in zip(
                counts, populations, strict=True
            ))
            or total != sum(count for _, count in counts)
            or chain_count != total
            or self.observation_chain_design_hash != self.acquisition_design_hash
        ):
            raise ValueError("snapshot count or observation-chain coordinates drifted")
        for stratum_index, (_name, count) in enumerate(counts):
            for endpoint_values in normalized_sums[stratum_index]:
                if count == 0 and any(value != 0 for value in endpoint_values):
                    raise ValueError("empty snapshot prefix has nonzero sufficient sums")
        if total == 0:
            initial_chain = content_hash(
                {"design_hash": self.acquisition_design_hash, "record_count": 0},
                prefix=_CHAIN_PREFIX,
            )
            if self.observation_chain_hash != initial_chain:
                raise ValueError("empty snapshot has a noninitial observation chain")
        prior = self.prior_transaction_commit_hash
        if prior is not None:
            prior = _identity(prior, name="prior transaction commit hash")
        if (
            _nonnegative_int(
                self.binary64_scale_exponent,
                name="binary64 scale exponent",
            )
            != _BINARY64_SCALE_EXPONENT
        ):
            raise ValueError("snapshot uses another exact binary64 integer scale")
        if (
            self.retained_context_ids != 0
            or self.retained_record_arrays != 0
            or self.raw_secant_records_retained is not False
        ):
            raise ValueError("snapshot retained raw C-stream state")
        if any(
            getattr(self, name) is not False
            for name in (
                "calibration_K_open_authority",
                "paired_oracle_O_open_authority",
                "mask_commit_authority",
                "terminal_authority",
            )
        ):
            raise PermissionError("snapshot cannot open K/O or mask authority")
        if (
            self.authority_scope != FQR_SEQUENTIAL_AUTHORITY_SCOPE
            or self.schema != FQR_SEQUENTIAL_ACCUMULATOR_SNAPSHOT_SCHEMA
        ):
            raise ValueError("snapshot schema or authority scope is invalid")
        object.__setattr__(self, "proposal_generation", _nonnegative_int(
            self.proposal_generation, name="snapshot proposal generation"
        ))
        object.__setattr__(self, "route_count", _positive_int(
            self.route_count, name="snapshot route count"
        ))
        object.__setattr__(self, "endpoint_action_hashes", endpoints)
        object.__setattr__(self, "stratum_names", strata)
        object.__setattr__(self, "population_sizes", populations)
        object.__setattr__(self, "role_axis", roles)
        object.__setattr__(self, "observed_counts_by_stratum", counts)
        object.__setattr__(
            self,
            "scaled_sums_by_stratum_endpoint_role",
            normalized_sums,
        )
        object.__setattr__(self, "total_records_consumed", total)
        object.__setattr__(self, "observation_chain_record_count", chain_count)
        object.__setattr__(self, "prior_transaction_commit_hash", prior)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "acquisition_design_hash",
                "fqr_design_hash",
                "secant_design_hash",
                "ladder_hash",
                "chart_identity_hash",
                "current_mask_hash",
                "proposal_generation",
                "route_count",
                "route_axis_hash",
                "endpoint_action_hashes",
                "stratum_names",
                "population_sizes",
                "role_axis",
                "observed_counts_by_stratum",
                "scaled_sums_by_stratum_endpoint_role",
                "total_records_consumed",
                "observation_chain_hash",
                "observation_chain_design_hash",
                "observation_chain_record_count",
                "joint_fqr_family_registration_hash",
                "prior_transaction_commit_hash",
                "binary64_scale_exponent",
                "retained_context_ids",
                "retained_record_arrays",
                "raw_secant_records_retained",
                "calibration_K_open_authority",
                "paired_oracle_O_open_authority",
                "mask_commit_authority",
                "terminal_authority",
                "authority_scope",
            )
        }

    @property
    def snapshot_hash(self) -> str:
        cached = self._snapshot_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_SNAPSHOT_PREFIX)
        if self._snapshot_hash is not None and self._snapshot_hash != observed:
            raise ValueError("F/Q/R accumulator snapshot was tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("F/Q/R accumulator snapshot hash does not match")
        object.__setattr__(self, "_snapshot_hash", observed)
        return observed

    def validate_against_design(
        self,
        design: V1FQRSequentialAcquisitionDesign,
    ) -> None:
        """Recompute all causal axes and exact support constraints."""

        if type(design) is not V1FQRSequentialAcquisitionDesign:
            raise TypeError("snapshot validation requires the exact design type")
        self.assert_integrity()
        design.assert_integrity()
        secant = design.fqr_design.secant_design
        ladder = secant.ladder
        expected = (
            design.design_hash,
            design.fqr_design.design_hash,
            secant.design_hash,
            ladder.ladder_hash,
            secant.chart_identity_hash,
            ladder.current_mask_hash,
            ladder.proposal_generation,
            ladder.route_count,
            ladder.route_axis_hash,
            design.endpoint_action_hashes,
            design.stratum_names,
            design.population_sizes,
            FQR_SEQUENTIAL_ROLES,
            design.joint_fqr_family_registration_hash,
        )
        observed = (
            self.acquisition_design_hash,
            self.fqr_design_hash,
            self.secant_design_hash,
            self.ladder_hash,
            self.chart_identity_hash,
            self.current_mask_hash,
            self.proposal_generation,
            self.route_count,
            self.route_axis_hash,
            self.endpoint_action_hashes,
            self.stratum_names,
            self.population_sizes,
            self.role_axis,
            self.joint_fqr_family_registration_hash,
        )
        if observed != expected:
            raise ValueError("F/Q/R accumulator snapshot is stale")
        for stratum_index, (_name, count) in enumerate(
            self.observed_counts_by_stratum
        ):
            bounds = np.stack(
                (
                    design.analytic_f_upper_by_stratum_endpoint[stratum_index],
                    design.analytic_q_upper_by_stratum_endpoint[stratum_index],
                    design.analytic_r_upper_by_stratum_endpoint_and_dose[
                        stratum_index, :, 0
                    ],
                    design.analytic_r_upper_by_stratum_endpoint_and_dose[
                        stratum_index, :, 1
                    ],
                ),
                axis=1,
            )
            for endpoint_index, role_values in enumerate(
                self.scaled_sums_by_stratum_endpoint_role[stratum_index]
            ):
                for role_index, scaled_sum in enumerate(role_values):
                    maximum = count * _binary64_scaled_integer(
                        float(bounds[endpoint_index, role_index])
                    )
                    if scaled_sum > maximum:
                        raise ValueError(
                            "snapshot sufficient sum exceeds its analytic support"
                        )


class V1FQRSequentialAccumulator:
    """Constant-cell-state acquisition over registered C-stream prefixes."""

    __slots__ = (
        "design",
        "_counts",
        "_scaled_sums",
        "_observation_chain_hash",
        "_total_records",
        "_prior_transaction_commit_hash",
    )

    def __init__(self, design: V1FQRSequentialAcquisitionDesign) -> None:
        if type(design) is not V1FQRSequentialAcquisitionDesign:
            raise TypeError("F/Q/R accumulator requires its exact design type")
        design_hash = design.assert_integrity()
        strata = len(design.stratum_names)
        endpoints = len(design.endpoint_action_hashes)
        self.design = design
        self._counts = [0 for _ in range(strata)]
        self._scaled_sums = [
            [[0 for _ in FQR_SEQUENTIAL_ROLES] for _ in range(endpoints)]
            for _ in range(strata)
        ]
        self._observation_chain_hash = content_hash(
            {"design_hash": design_hash, "record_count": 0},
            prefix=_CHAIN_PREFIX,
        )
        self._total_records = 0
        self._prior_transaction_commit_hash: str | None = None

    @classmethod
    def from_snapshot(
        cls,
        design: V1FQRSequentialAcquisitionDesign,
        snapshot: V1FQRSequentialAccumulatorSnapshot,
        *,
        expected_snapshot_hash: str | None = None,
        expected_prior_transaction_commit_hash: str | None = None,
    ) -> "V1FQRSequentialAccumulator":
        """Restore exact sufficient statistics after independent validation."""

        if type(design) is not V1FQRSequentialAcquisitionDesign:
            raise TypeError("snapshot restore requires the exact acquisition design")
        if type(snapshot) is not V1FQRSequentialAccumulatorSnapshot:
            raise TypeError("snapshot restore requires the exact snapshot type")
        if (
            snapshot.total_records_consumed > 0
            and expected_snapshot_hash is None
        ):
            raise ValueError(
                "nonempty snapshot restore requires an expected snapshot hash"
            )
        snapshot.assert_integrity(expected_snapshot_hash)
        snapshot.validate_against_design(design)
        if expected_prior_transaction_commit_hash is not None:
            expected_commit = _identity(
                expected_prior_transaction_commit_hash,
                name="expected prior transaction commit hash",
            )
            if snapshot.prior_transaction_commit_hash != expected_commit:
                raise ValueError("snapshot prior transaction commit hash does not match")
        restored = cls(design)
        restored._counts = [
            count for _name, count in snapshot.observed_counts_by_stratum
        ]
        restored._scaled_sums = [
            [list(role_values) for role_values in endpoint_rows]
            for endpoint_rows in snapshot.scaled_sums_by_stratum_endpoint_role
        ]
        restored._observation_chain_hash = snapshot.observation_chain_hash
        restored._total_records = snapshot.total_records_consumed
        restored._prior_transaction_commit_hash = (
            snapshot.prior_transaction_commit_hash
        )
        # Replay all derived intervals and stopping logic from the exact sums.
        # Constructing this receipt detects any internally inconsistent state
        # before the resumed accumulator is returned to a caller.
        restored.receipt().assert_integrity()
        return restored

    def export_snapshot(
        self,
        prior_transaction_commit_hash: str | None = None,
    ) -> V1FQRSequentialAccumulatorSnapshot:
        """Export exact constant-cell state without raw observations."""

        self.design.assert_integrity()
        supplied = (
            None
            if prior_transaction_commit_hash is None
            else _identity(
                prior_transaction_commit_hash,
                name="prior transaction commit hash",
            )
        )
        if (
            supplied is not None
            and self._prior_transaction_commit_hash is not None
            and supplied != self._prior_transaction_commit_hash
        ):
            raise ValueError("accumulator snapshot transaction ancestry changed")
        prior = supplied or self._prior_transaction_commit_hash
        secant = self.design.fqr_design.secant_design
        ladder = secant.ladder
        snapshot = V1FQRSequentialAccumulatorSnapshot(
            acquisition_design_hash=self.design.design_hash,
            fqr_design_hash=self.design.fqr_design.design_hash,
            secant_design_hash=secant.design_hash,
            ladder_hash=ladder.ladder_hash,
            chart_identity_hash=secant.chart_identity_hash,
            current_mask_hash=ladder.current_mask_hash,
            proposal_generation=ladder.proposal_generation,
            route_count=ladder.route_count,
            route_axis_hash=ladder.route_axis_hash,
            endpoint_action_hashes=self.design.endpoint_action_hashes,
            stratum_names=self.design.stratum_names,
            population_sizes=self.design.population_sizes,
            role_axis=FQR_SEQUENTIAL_ROLES,
            observed_counts_by_stratum=tuple(
                zip(self.design.stratum_names, self._counts, strict=True)
            ),
            scaled_sums_by_stratum_endpoint_role=tuple(
                tuple(tuple(role_values) for role_values in endpoint_rows)
                for endpoint_rows in self._scaled_sums
            ),
            total_records_consumed=self._total_records,
            observation_chain_hash=self._observation_chain_hash,
            observation_chain_design_hash=self.design.design_hash,
            observation_chain_record_count=self._total_records,
            joint_fqr_family_registration_hash=(
                self.design.joint_fqr_family_registration_hash
            ),
            prior_transaction_commit_hash=prior,
        )
        snapshot.validate_against_design(self.design)
        return snapshot

    @property
    def retained_scalar_slots(self) -> int:
        return (
            len(self.design.stratum_names)
            * len(self.design.endpoint_action_hashes)
            * len(FQR_SEQUENTIAL_ROLES)
            + len(self.design.stratum_names)
            + 2
        )

    @property
    def total_records_consumed(self) -> int:
        return self._total_records

    def _record_values(self, record: V1DirectionalSecantRecord) -> np.ndarray:
        return np.stack(
            (
                record.f_upper_observation_by_endpoint,
                record.q_upper_observation_by_endpoint,
                record.r_upper_observation_by_endpoint[:, 0],
                record.r_upper_observation_by_endpoint[:, 1],
            ),
            axis=1,
        )

    def record(self, observation: V1DirectionalSecantRecord) -> None:
        """Atomically consume one exact next record from one sealed stratum."""

        if type(observation) is not V1DirectionalSecantRecord:
            raise TypeError("F/Q/R acquisition requires typed secant records")
        observation.assert_integrity()
        secant = self.design.fqr_design.secant_design
        if (
            observation.design_hash != secant.design_hash
            or observation.ladder_hash != secant.ladder_hash
            or observation.chart_identity_hash != secant.chart_identity_hash
            or observation.endpoint_action_hashes
            != self.design.endpoint_action_hashes
        ):
            raise ValueError("secant record is stale or belongs to another F/Q/R chart")
        try:
            stratum_index = self.design.stratum_names.index(observation.stratum_id)
        except ValueError as error:
            raise ValueError("secant record names an unknown C stratum") from error
        count = self._counts[stratum_index]
        contexts = secant.certification_context_ids_by_stratum[stratum_index][1]
        draws = self.design.sealed_draw_ids_by_stratum[stratum_index][1]
        if count >= len(contexts):
            raise ValueError("secant record exceeds its finite C population")
        if (
            observation.sealed_order_index != count
            or observation.context_id != contexts[count]
            or observation.draw_id != draws[count]
        ):
            raise ValueError("secant record is not the exact next sealed C-prefix row")
        values = self._record_values(observation)
        bounds = np.stack(
            (
                self.design.analytic_f_upper_by_stratum_endpoint[stratum_index],
                self.design.analytic_q_upper_by_stratum_endpoint[stratum_index],
                self.design.analytic_r_upper_by_stratum_endpoint_and_dose[
                    stratum_index, :, 0
                ],
                self.design.analytic_r_upper_by_stratum_endpoint_and_dose[
                    stratum_index, :, 1
                ],
            ),
            axis=1,
        )
        if (
            values.shape != bounds.shape
            or not np.isfinite(values).all()
            or bool(np.any(values < 0.0))
            or bool(np.any(values > bounds))
        ):
            raise ValueError("secant F/Q/R observation exceeded its analytic bound")
        scaled = [
            [_binary64_scaled_integer(float(value)) for value in endpoint]
            for endpoint in values
        ]
        # All refusals precede mutation, so a row updates every endpoint/role
        # and its provenance chain atomically.
        for endpoint_index, endpoint_values in enumerate(scaled):
            for role_index, value in enumerate(endpoint_values):
                self._scaled_sums[stratum_index][endpoint_index][role_index] += value
        self._counts[stratum_index] += 1
        self._total_records += 1
        self._observation_chain_hash = content_hash(
            {
                "previous_chain_hash": self._observation_chain_hash,
                "record_hash": observation.record_hash,
                "record_count": self._total_records,
            },
            prefix=_CHAIN_PREFIX,
        )

    def _cell_summary(
        self,
        *,
        stratum_index: int,
        endpoint_index: int,
        role_index: int,
    ) -> V1FQRSequentialCellSummary:
        count = self._counts[stratum_index]
        population = self.design.population_sizes[stratum_index]
        scaled_sum = self._scaled_sums[stratum_index][endpoint_index][role_index]
        role = FQR_SEQUENTIAL_ROLES[role_index]
        dose = 0.5 if role == R_HALF_ROLE else 1.0 if role == R_ONE_ROLE else None
        if role == F_ROLE:
            value_bound = self.design.analytic_f_upper_by_stratum_endpoint[
                stratum_index, endpoint_index
            ]
        elif role == Q_ROLE:
            value_bound = self.design.analytic_q_upper_by_stratum_endpoint[
                stratum_index, endpoint_index
            ]
        else:
            value_bound = self.design.analytic_r_upper_by_stratum_endpoint_and_dose[
                stratum_index,
                endpoint_index,
                0 if role == R_HALF_ROLE else 1,
            ]
        if count == 0:
            mean = None
            lower = 0.0
            upper = float(value_bound)
            look_alpha = 0.0
            numerator = denominator = None
            full = False
            method = FQR_ONE_SIDED_METHOD
        else:
            exact_mean = _scaled_mean(scaled_sum, count)
            mean = float(exact_mean)
            if count == population:
                lower = _fraction_to_float_outward(
                    exact_mean,
                    direction="lower",
                )
                upper = _fraction_to_float_outward(exact_mean, direction="upper")
                look_alpha = 0.0
                numerator = exact_mean.numerator
                denominator = exact_mean.denominator
                full = True
                method = FQR_CENSUS_METHOD
            else:
                look_alpha = self.design.look_alpha(
                    stratum_index=stratum_index,
                    sample_size=count,
                )
                radius = (
                    _two_sided_serfling_radius_upper
                    if self.design.two_sided_decision_intervals
                    else _serfling_radius_upper
                )(
                    upper_bound=float(value_bound),
                    count=count,
                    population=population,
                    look_alpha=look_alpha,
                )
                if not math.isfinite(radius):
                    lower = 0.0
                    upper = float(value_bound)
                else:
                    exact_radius = Fraction.from_float(radius)
                    lower_candidate = exact_mean - exact_radius
                    lower = (
                        0.0
                        if lower_candidate <= 0
                        else _fraction_to_float_outward(
                            lower_candidate,
                            direction="lower",
                        )
                    )
                    exact_candidate = exact_mean + Fraction.from_float(radius)
                    upper = min(
                        float(value_bound),
                        _fraction_to_float_outward(
                            exact_candidate,
                            direction="upper",
                        ),
                    )
                numerator = denominator = None
                full = False
                method = (
                    FQR_TWO_SIDED_METHOD
                    if self.design.two_sided_decision_intervals
                    else FQR_ONE_SIDED_METHOD
                )
        cell_schema = (
            FQR_SEQUENTIAL_CELL_SUMMARY_SCHEMA
            if self.design.two_sided_decision_intervals
            else FQR_SEQUENTIAL_CELL_SUMMARY_SCHEMA_V1
        )
        return V1FQRSequentialCellSummary(
            endpoint_index=endpoint_index + 1,
            action_hash=self.design.endpoint_action_hashes[endpoint_index],
            role=role,
            dose=dose,
            stratum_id=self.design.stratum_names[stratum_index],
            population_size=population,
            sample_size=count,
            sample_mean_center=mean,
            lower=lower,
            upper=upper,
            value_upper_bound=float(value_bound),
            current_look_alpha=look_alpha,
            cell_family_alpha=self.design.cell_family_alpha,
            exact_census_mean_numerator=numerator,
            exact_census_mean_denominator=denominator,
            full_census=full,
            empty=False,
            method=method,
            observation_chain_hash=self._observation_chain_hash,
            schema=cell_schema,
        )

    def receipt(self) -> V1FQRSequentialAcquisitionReceipt:
        """Return a simultaneous snapshot at any optional stopping time."""

        self.design.assert_integrity()
        cell_summaries = tuple(
            self._cell_summary(
                stratum_index=stratum,
                endpoint_index=endpoint,
                role_index=role,
            )
            for endpoint in range(len(self.design.endpoint_action_hashes))
            for role in range(len(FQR_SEQUENTIAL_ROLES))
            for stratum in range(len(self.design.stratum_names))
        )
        by_cell = {
            (item.endpoint_index, item.role, item.stratum_id): item
            for item in cell_summaries
        }
        weights = tuple(
            weight for _, weight in self.design.fqr_design.secant_design.stratum_weights
        )
        endpoint_summaries: list[V1FQRSequentialEndpointSummary] = []
        for endpoint_index, action_hash in enumerate(
            self.design.endpoint_action_hashes,
            start=1,
        ):
            role_lowers = {
                role: _downward_weighted_sum(
                    weights,
                    tuple(
                        by_cell[(endpoint_index, role, stratum)].lower
                        for stratum in self.design.stratum_names
                    ),
                )
                for role in FQR_SEQUENTIAL_ROLES
            }
            role_uppers = {
                role: _outward_weighted_sum(
                    weights,
                    tuple(
                        by_cell[(endpoint_index, role, stratum)].upper
                        for stratum in self.design.stratum_names
                    ),
                )
                for role in FQR_SEQUENTIAL_ROLES
            }
            half_burden = _response_burden(
                eta=0.5,
                q_upper=role_uppers[Q_ROLE],
                r_upper=role_uppers[R_HALF_ROLE],
            )
            one_burden = _response_burden(
                eta=1.0,
                q_upper=role_uppers[Q_ROLE],
                r_upper=role_uppers[R_ONE_ROLE],
            )
            f_certified = role_uppers[F_ROLE] <= self.design.fqr_design.tau_f
            half_certified = half_burden <= 0.5 * self.design.fqr_design.tau_l
            one_certified = one_burden <= self.design.fqr_design.tau_l
            certified = f_certified and half_certified and one_certified
            half_burden_lower = _response_burden_lower(
                eta=0.5,
                q_lower=role_lowers[Q_ROLE],
                r_lower=role_lowers[R_HALF_ROLE],
            )
            one_burden_lower = _response_burden_lower(
                eta=1.0,
                q_lower=role_lowers[Q_ROLE],
                r_lower=role_lowers[R_ONE_ROLE],
            )
            census = all(count == population for count, population in zip(
                self._counts,
                self.design.population_sizes,
                strict=True,
            ))
            if self.design.two_sided_decision_intervals:
                refuted = bool(
                    role_lowers[F_ROLE] > self.design.fqr_design.tau_f
                    or half_burden_lower
                    > 0.5 * self.design.fqr_design.tau_l
                    or one_burden_lower > self.design.fqr_design.tau_l
                )
            else:
                refuted = bool(census and not certified)
            endpoint_summaries.append(
                V1FQRSequentialEndpointSummary(
                    endpoint_index=endpoint_index,
                    action_hash=action_hash,
                    f_lower=role_lowers[F_ROLE],
                    f_upper=role_uppers[F_ROLE],
                    q_lower=role_lowers[Q_ROLE],
                    q_upper=role_uppers[Q_ROLE],
                    r_half_lower=role_lowers[R_HALF_ROLE],
                    r_half_upper=role_uppers[R_HALF_ROLE],
                    r_one_lower=role_lowers[R_ONE_ROLE],
                    r_one_upper=role_uppers[R_ONE_ROLE],
                    half_response_burden_lower=half_burden_lower,
                    half_response_burden_upper=half_burden,
                    one_response_burden_lower=one_burden_lower,
                    one_response_burden_upper=one_burden,
                    f_certified=f_certified,
                    half_certified=half_certified,
                    one_certified=one_certified,
                    sampling_certified=certified,
                    sampling_refuted=refuted,
                    all_strata_censused=census,
                    schema=(
                        FQR_SEQUENTIAL_ENDPOINT_SUMMARY_SCHEMA
                        if self.design.two_sided_decision_intervals
                        else FQR_SEQUENTIAL_ENDPOINT_SUMMARY_SCHEMA_V1
                    ),
                )
            )
        selected_index: int | None = None
        selected_hash: str | None = None
        for item in reversed(endpoint_summaries):
            if item.sampling_certified and all(
                larger.sampling_refuted
                for larger in endpoint_summaries[item.endpoint_index :]
            ):
                selected_index = item.endpoint_index
                selected_hash = item.action_hash
                break
        if selected_index is not None:
            sampling_decision: SamplingDecision = "sampling_stable_endpoint"
        elif all(item.sampling_refuted for item in endpoint_summaries):
            sampling_decision = "sampling_stable_no_endpoint"
        else:
            sampling_decision = "acquire_more_C"
        secant = self.design.fqr_design.secant_design
        next_contexts: list[tuple[str, str | None]] = []
        next_draws: list[tuple[str, str | None]] = []
        prefix_rows: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
        for index, stratum in enumerate(self.design.stratum_names):
            contexts = secant.certification_context_ids_by_stratum[index][1]
            draws = self.design.sealed_draw_ids_by_stratum[index][1]
            count = self._counts[index]
            next_contexts.append(
                (stratum, None if count == len(contexts) else contexts[count])
            )
            next_draws.append((stratum, None if count == len(draws) else draws[count]))
            prefix_rows.append((stratum, contexts[:count], draws[:count]))
        prefix_hash = content_hash(
            {
                "design_hash": self.design.design_hash,
                "observed_prefixes": tuple(prefix_rows),
                "observation_chain_hash": self._observation_chain_hash,
            },
            prefix="modern-v1-fqr-observed-prefix-",
        )
        return V1FQRSequentialAcquisitionReceipt(
            design_hash=self.design.design_hash,
            fqr_design_hash=self.design.fqr_design.design_hash,
            secant_design_hash=secant.design_hash,
            ladder_hash=self.design.fqr_design.ladder_hash,
            chart_identity_hash=self.design.fqr_design.chart_identity_hash,
            endpoint_action_hashes=self.design.endpoint_action_hashes,
            observed_counts_by_stratum=tuple(
                zip(self.design.stratum_names, self._counts, strict=True)
            ),
            next_expected_context_by_stratum=tuple(next_contexts),
            next_expected_draw_by_stratum=tuple(next_draws),
            observed_prefix_hash=prefix_hash,
            observation_chain_hash=self._observation_chain_hash,
            cell_summaries=cell_summaries,
            endpoint_summaries=tuple(endpoint_summaries),
            sampling_decision=sampling_decision,
            selected_endpoint_index=selected_index,
            selected_action_hash=selected_hash,
            sampling_ambiguity_present=sampling_decision == "acquire_more_C",
            trajectory_alpha_ledger_hash=self.design.trajectory_alpha_ledger_hash,
            fqr_family_alpha_allocation=self.design.fqr_family_alpha_allocation,
            cell_family_alpha=self.design.cell_family_alpha,
            registered_cell_alpha_total=self.design.registered_cell_alpha_total,
            stratum_weights=self.design.fqr_design.secant_design.stratum_weights,
            tau_f=self.design.fqr_design.tau_f,
            tau_l=self.design.fqr_design.tau_l,
            alpha_spent_by_receipt_inspection=0.0,
            alpha_allocation_nonrecycling=True,
            retained_scalar_slots=self.retained_scalar_slots,
            retained_context_ids=0,
            retained_record_arrays=0,
            schema=(
                FQR_SEQUENTIAL_RECEIPT_SCHEMA
                if self.design.two_sided_decision_intervals
                else FQR_SEQUENTIAL_RECEIPT_SCHEMA_V1
            ),
        )

    def decision_readiness(
        self,
        deterministic: V1FQRDeterministicUncertaintyReceipt,
    ) -> V1FQRDecisionReadinessReceipt:
        if type(deterministic) is not V1FQRDeterministicUncertaintyReceipt:
            raise TypeError("F/Q/R readiness requires typed deterministic uncertainty")
        deterministic_hash = deterministic.assert_integrity()
        acquisition = self.receipt()
        if (
            deterministic.acquisition_design_hash != self.design.design_hash
            or deterministic.fqr_design_hash != self.design.fqr_design.design_hash
            or deterministic.ladder_hash != self.design.fqr_design.ladder_hash
            or deterministic.chart_identity_hash
            != self.design.fqr_design.chart_identity_hash
            or deterministic.fd_rank_uncertainty_set_hash
            != self.design.fd_rank_uncertainty_set_hash
            or deterministic.kkt_numerical_uncertainty_set_hash
            != self.design.kkt_numerical_uncertainty_set_hash
            or deterministic.numerical_enclosure_policy_hash
            != self.design.numerical_enclosure_policy_hash
        ):
            raise ValueError("deterministic uncertainty receipt is stale")
        sampling_resolved = not acquisition.sampling_ambiguity_present
        composition_resolved = deterministic.resolved_for_independent_C
        # Legacy v1/v2 receipts retain their diagnostic distinction between
        # unresolved FD rank and unresolved KKT numerics.  Proposal-v3 carries
        # neither claim; its exact pre-C family freeze resolves both solely for
        # the purpose of measuring that fixed family on independent C.
        fd_resolved = (
            deterministic.fd_rank_decision_stable or composition_resolved
        )
        kkt_resolved = (
            deterministic.kkt_numerical_decision_stable or composition_resolved
        )
        numerical_resolved = deterministic.numerical_enclosure_valid
        ready = (
            sampling_resolved
            and fd_resolved
            and kkt_resolved
            and numerical_resolved
        )
        if not sampling_resolved:
            status: ReadinessStatus = "acquire_more_C"
            dominant = "sampling"
        elif not fd_resolved:
            status = "resolve_fd_rank"
            dominant = "fd_rank"
        elif not kkt_resolved:
            status = "resolve_kkt_numerics"
            dominant = "kkt_numerics"
        elif not numerical_resolved:
            status = "resolve_numerical_enclosure"
            dominant = "numerical_enclosure"
        else:
            status = "ready_for_fqr_proposal_screening"
            dominant = None
        return V1FQRDecisionReadinessReceipt(
            acquisition_receipt_hash=acquisition.receipt_hash,
            deterministic_uncertainty_receipt_hash=deterministic_hash,
            design_hash=self.design.design_hash,
            fqr_design_hash=self.design.fqr_design.design_hash,
            ladder_hash=self.design.fqr_design.ladder_hash,
            chart_identity_hash=self.design.fqr_design.chart_identity_hash,
            status=status,
            dominant_ambiguity=dominant,
            selected_endpoint_index=acquisition.selected_endpoint_index,
            selected_action_hash=acquisition.selected_action_hash,
            sampling_resolved=sampling_resolved,
            fd_rank_resolved=fd_resolved,
            kkt_numerics_resolved=kkt_resolved,
            numerical_enclosure_resolved=numerical_resolved,
            ready_for_fqr_proposal_screening=ready,
        )


__all__ = [
    "FQR_ALL_PREFIX_ALPHA_POLICY",
    "FQR_CENSUS_METHOD",
    "FQR_ONE_SIDED_METHOD",
    "FQR_TWO_SIDED_METHOD",
    "FQR_DETERMINISTIC_UNCERTAINTY_SCHEMA",
    "FQR_DETERMINISTIC_UNCERTAINTY_PROPOSAL_V3_SCHEMA",
    "FQR_DETERMINISTIC_UNCERTAINTY_V2_SCHEMA",
    "FQR_SEQUENTIAL_ROLES",
    "F_ROLE",
    "Q_ROLE",
    "R_HALF_ROLE",
    "R_ONE_ROLE",
    "V1FQRDecisionReadinessReceipt",
    "V1FQRDeterministicUncertaintyReceipt",
    "V1FQRSequentialAccumulator",
    "V1FQRSequentialAccumulatorSnapshot",
    "V1FQRSequentialAcquisitionDesign",
    "V1FQRSequentialAcquisitionReceipt",
    "V1FQRSequentialCellSummary",
    "V1FQRSequentialEndpointSummary",
    "issue_v1_fqr_deterministic_uncertainty_from_adaptive_rank",
    "issue_v1_fqr_deterministic_uncertainty_from_proposal_freeze",
]
