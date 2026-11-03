"""Exact finite-population measure for frozen ``D1`` plus fresh ``D_W2``.

The post-shell witness conditions on the already observed discovery prefix
``D1`` and samples without replacement from its registered residual suffix.
The full-population estimand is therefore not a fresh-suffix mean.  For
stratum ``h`` with population ``N_h``, frozen count ``d_h``, residual count
``R_h=N_h-d_h``, and acquired suffix count ``n_h``, its exact decomposition is

```
mu = (1/N) * sum_{i in D1} y_i
   + sum_h (R_h / (N n_h)) * sum_{i in D_W2,h} y_i.
```

This module binds that measure before packet outcomes.  It does not own the
``y_i`` values and therefore cannot certify a decision until an exact compact
``D1`` contribution and the matching residual source are joined.  In
particular, neither ``N_h/(N n_h)`` nor empirical packet proportions may
impersonate the residual weights above.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
import math
from numbers import Integral
from typing import Final, NoReturn

from .contracts import content_hash
from .modern_v1_sampled_variable_shell_dw2_suffix import (
    V1SampledVariableShellDW2SuffixPacket,
    V1SampledVariableShellDW2SuffixRegistration,
)


SAMPLED_RESIDUAL_POPULATION_MEASURE_DESIGN_SCHEMA: Final = (
    "partition_free_v1_sampled_residual_population_measure_design_v1"
)
SAMPLED_RESIDUAL_POPULATION_WEIGHT_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_sampled_residual_population_weight_receipt_v1"
)
SAMPLED_RESIDUAL_POPULATION_MEASURE_AUTHORITY: Final = (
    "pre_outcome_exact_D1_plus_residual_measure_only_no_observation_"
    "confidence_decision_C_K_O_commit_terminal_or_pareto_authority_v1"
)

_PREFIX: Final = "partition-free-v1-sampled-residual-population-measure-"
_ISSUER_TOKEN: Final = object()


class V1SampledResidualPopulationMeasureRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_IDENTITY = "stale_identity"
    COUNT_MISMATCH = "count_mismatch"
    INCOMPLETE_RESIDUAL_SAMPLE = "incomplete_residual_sample"
    WEIGHT_MISMATCH = "weight_mismatch"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SampledResidualPopulationMeasureRefusal(RuntimeError):
    """Typed refusal for an invalid D1/residual finite-population measure."""

    def __init__(
        self,
        code: V1SampledResidualPopulationMeasureRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledResidualPopulationMeasureRefusalCode:
            raise TypeError("residual-measure refusal code has the wrong type")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledResidualPopulationMeasureRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledResidualPopulationMeasureRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledResidualPopulationMeasureRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SampledResidualPopulationMeasureRefusalCode.INVALID_TYPE,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SampledResidualPopulationMeasureRefusalCode.COUNT_MISMATCH,
            f"{name} must be nonnegative",
        )
    return result


def _count_pair(value: object, *, name: str) -> tuple[int, int]:
    if type(value) is not tuple or len(value) != 2:
        _refuse(
            V1SampledResidualPopulationMeasureRefusalCode.INVALID_TYPE,
            f"{name} must be one immutable two-stratum count axis",
        )
    return tuple(
        _nonnegative_int(item, name=f"{name} item") for item in value
    )  # type: ignore[return-value]


def _fraction_pair(value: Fraction) -> tuple[int, int]:
    return value.numerator, value.denominator


def _binary64_fraction_enclosure(value: Fraction) -> tuple[float, float, float]:
    """Return a nearest center and strict outward enclosure of one rational."""

    center = float(value)
    if not math.isfinite(center):
        _refuse(
            V1SampledResidualPopulationMeasureRefusalCode.WEIGHT_MISMATCH,
            "exact context weight overflowed binary64",
        )
    center_fraction = Fraction.from_float(center)
    lower = center
    upper = center
    if center_fraction > value:
        lower = math.nextafter(lower, -math.inf)
    elif center_fraction < value:
        upper = math.nextafter(upper, math.inf)
    lower = math.nextafter(lower, -math.inf)
    upper = math.nextafter(upper, math.inf)
    if not (
        Fraction.from_float(lower) <= value <= Fraction.from_float(upper)
        and lower <= center <= upper
    ):
        _refuse(
            V1SampledResidualPopulationMeasureRefusalCode.WEIGHT_MISMATCH,
            "binary64 context-weight enclosure is not outward",
        )
    return lower, center, upper


def _validate_binary64_fraction_enclosure(
    *,
    exact: tuple[int, int],
    enclosure: tuple[float, float, float],
) -> None:
    if (
        type(enclosure) is not tuple
        or len(enclosure) != 3
        or any(type(item) is not float or not math.isfinite(item) for item in enclosure)
    ):
        _refuse(
            V1SampledResidualPopulationMeasureRefusalCode.WEIGHT_MISMATCH,
            "binary64 weight enclosure must be three finite immutable scalars",
        )
    lower, center, upper = enclosure
    value = Fraction(*exact)
    if not (
        lower <= center <= upper
        and Fraction.from_float(lower) <= value <= Fraction.from_float(upper)
    ):
        _refuse(
            V1SampledResidualPopulationMeasureRefusalCode.WEIGHT_MISMATCH,
            "binary64 weight enclosure does not contain its exact rational",
        )


@dataclass(frozen=True, slots=True)
class V1SampledResidualPopulationMeasureDesign:
    """Source-issued decomposition of the complete train-population measure."""

    suffix_registration_hash: str
    generation_frame_hash: str
    discovery_stream_hash: str
    current_mask_hash: str
    generation_index: int
    population_stratum_counts: tuple[int, int]
    d1_stratum_counts: tuple[int, int]
    residual_stratum_counts: tuple[int, int]
    population_count: int
    d1_pair_prefix_axis_hash: str
    residual_pair_suffix_axis_hash: str
    target_population_membership_hash: str
    complement_population_membership_hash: str
    full_population_uniform_measure: bool = True
    d1_contribution_is_exact_conditional_term: bool = True
    dw2_contribution_samples_residual_suffix: bool = True
    exact_d1_compact_source_join_required: bool = True
    suffix_only_decision_authority: bool = False
    empirical_packet_proportion_weights_allowed: bool = False
    outcome_payload_present: bool = False
    authority_scope: str = SAMPLED_RESIDUAL_POPULATION_MEASURE_AUTHORITY
    schema: str = SAMPLED_RESIDUAL_POPULATION_MEASURE_DESIGN_SCHEMA
    _issuance_token: object = field(default=None, repr=False, compare=False)
    _bound_design_hash: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuance_token is not _ISSUER_TOKEN:
            _refuse(
                V1SampledResidualPopulationMeasureRefusalCode.INVALID_TYPE,
                "residual population designs must be source-issued",
            )
        for name in (
            "suffix_registration_hash",
            "generation_frame_hash",
            "discovery_stream_hash",
            "current_mask_hash",
            "d1_pair_prefix_axis_hash",
            "residual_pair_suffix_axis_hash",
            "target_population_membership_hash",
            "complement_population_membership_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        generation = _nonnegative_int(self.generation_index, name="generation index")
        population = _count_pair(
            self.population_stratum_counts,
            name="population stratum counts",
        )
        d1 = _count_pair(self.d1_stratum_counts, name="D1 stratum counts")
        residual = _count_pair(
            self.residual_stratum_counts,
            name="residual stratum counts",
        )
        count = _nonnegative_int(self.population_count, name="population count")
        if (
            count <= 0
            or count != sum(population)
            or any(d1[index] + residual[index] != population[index] for index in (0, 1))
        ):
            _refuse(
                V1SampledResidualPopulationMeasureRefusalCode.COUNT_MISMATCH,
                "D1 and residual counts do not partition the complete population",
            )
        required_true = (
            self.full_population_uniform_measure,
            self.d1_contribution_is_exact_conditional_term,
            self.dw2_contribution_samples_residual_suffix,
            self.exact_d1_compact_source_join_required,
        )
        required_false = (
            self.suffix_only_decision_authority,
            self.empirical_packet_proportion_weights_allowed,
            self.outcome_payload_present,
        )
        if (
            any(type(item) is not bool for item in required_true + required_false)
            or any(item is not True for item in required_true)
            or any(item is not False for item in required_false)
            or self.authority_scope != SAMPLED_RESIDUAL_POPULATION_MEASURE_AUTHORITY
            or self.schema != SAMPLED_RESIDUAL_POPULATION_MEASURE_DESIGN_SCHEMA
        ):
            _refuse(
                V1SampledResidualPopulationMeasureRefusalCode.AUTHORITY_VIOLATION,
                "residual population design changed its measure or authority",
            )
        object.__setattr__(self, "generation_index", generation)
        object.__setattr__(self, "population_stratum_counts", population)
        object.__setattr__(self, "d1_stratum_counts", d1)
        object.__setattr__(self, "residual_stratum_counts", residual)
        object.__setattr__(self, "population_count", count)
        observed = self._compute_design_hash()
        if self._bound_design_hash is not None and self._bound_design_hash != observed:
            _refuse(
                V1SampledResidualPopulationMeasureRefusalCode.STALE_IDENTITY,
                "residual population design changed after source issuance",
            )
        object.__setattr__(self, "_bound_design_hash", observed)

    def _compute_design_hash(self) -> str:
        return content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            },
            prefix=_PREFIX + "design-",
        )

    def assert_integrity(self) -> str:
        observed = self._compute_design_hash()
        if observed != self._bound_design_hash:
            _refuse(
                V1SampledResidualPopulationMeasureRefusalCode.STALE_IDENTITY,
                "residual population design failed its source-bound hash",
            )
        return observed

    @property
    def design_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1SampledResidualPopulationWeightReceipt:
    """Exact rational weights after one predictable cumulative suffix packet."""

    measure_design_hash: str
    suffix_registration_hash: str
    packet_hash: str
    packet_ordinal: int
    cumulative_dw2_stratum_counts: tuple[int, int]
    population_stratum_counts: tuple[int, int]
    d1_stratum_counts: tuple[int, int]
    residual_stratum_counts: tuple[int, int]
    d1_context_weight_fraction: tuple[int, int]
    dw2_context_weight_fractions: tuple[tuple[int, int], tuple[int, int]]
    residual_conditional_context_weight_fractions: tuple[
        tuple[int, int], tuple[int, int]
    ]
    residual_stratum_mass_fractions: tuple[tuple[int, int], tuple[int, int]]
    residual_population_scale_fraction: tuple[int, int]
    d1_context_weight_binary64_enclosure: tuple[float, float, float]
    dw2_context_weight_binary64_enclosures: tuple[
        tuple[float, float, float], tuple[float, float, float]
    ]
    residual_conditional_weight_binary64_enclosures: tuple[
        tuple[float, float, float], tuple[float, float, float]
    ]
    common_denominator: int
    d1_context_quantum: int
    dw2_context_quanta: tuple[int, int]
    total_weight_quanta: int
    cumulative_prefix_is_preoutcome: bool = True
    arbitrary_packet_sizes_supported: bool = True
    full_population_weights_sum_exactly_one: bool = True
    exact_d1_observation_term_required: bool = True
    residual_observation_term_required: bool = True
    suffix_only_decision_authority: bool = False
    confidence_authority: bool = False
    c_stream_authority: bool = False
    commit_authority: bool = False
    terminal_or_pareto_authority: bool = False
    authority_scope: str = SAMPLED_RESIDUAL_POPULATION_MEASURE_AUTHORITY
    schema: str = SAMPLED_RESIDUAL_POPULATION_WEIGHT_RECEIPT_SCHEMA
    _issuance_token: object = field(default=None, repr=False, compare=False)
    _bound_receipt_hash: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuance_token is not _ISSUER_TOKEN:
            _refuse(
                V1SampledResidualPopulationMeasureRefusalCode.INVALID_TYPE,
                "residual population weight receipts must be source-issued",
            )
        for name in ("measure_design_hash", "suffix_registration_hash", "packet_hash"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "packet_ordinal",
            _nonnegative_int(self.packet_ordinal, name="packet ordinal"),
        )
        for name in (
            "cumulative_dw2_stratum_counts",
            "population_stratum_counts",
            "d1_stratum_counts",
            "residual_stratum_counts",
            "dw2_context_quanta",
        ):
            object.__setattr__(self, name, _count_pair(getattr(self, name), name=name))
        for name in (
            "common_denominator",
            "d1_context_quantum",
            "total_weight_quanta",
        ):
            value = _nonnegative_int(getattr(self, name), name=name)
            if value <= 0:
                _refuse(
                    V1SampledResidualPopulationMeasureRefusalCode.WEIGHT_MISMATCH,
                    f"{name} must be positive",
                )
            object.__setattr__(self, name, value)
        fraction_axes = (
            (self.d1_context_weight_fraction,),
            self.dw2_context_weight_fractions,
            self.residual_conditional_context_weight_fractions,
            self.residual_stratum_mass_fractions,
            (self.residual_population_scale_fraction,),
        )
        for axis in fraction_axes:
            for item in axis:
                if (
                    type(item) is not tuple
                    or len(item) != 2
                    or any(
                        isinstance(value, bool) or not isinstance(value, Integral)
                        for value in item
                    )
                    or int(item[1]) <= 0
                    or int(item[0]) < 0
                ):
                    _refuse(
                        V1SampledResidualPopulationMeasureRefusalCode.WEIGHT_MISMATCH,
                        "weight fractions must be exact nonnegative rational pairs",
                    )
        _validate_binary64_fraction_enclosure(
            exact=self.d1_context_weight_fraction,
            enclosure=self.d1_context_weight_binary64_enclosure,
        )
        for exact, enclosure in zip(
            self.dw2_context_weight_fractions,
            self.dw2_context_weight_binary64_enclosures,
            strict=True,
        ):
            _validate_binary64_fraction_enclosure(
                exact=exact,
                enclosure=enclosure,
            )
        for exact, enclosure in zip(
            self.residual_conditional_context_weight_fractions,
            self.residual_conditional_weight_binary64_enclosures,
            strict=True,
        ):
            _validate_binary64_fraction_enclosure(
                exact=exact,
                enclosure=enclosure,
            )
        required_true = (
            self.cumulative_prefix_is_preoutcome,
            self.arbitrary_packet_sizes_supported,
            self.full_population_weights_sum_exactly_one,
            self.exact_d1_observation_term_required,
            self.residual_observation_term_required,
        )
        required_false = (
            self.suffix_only_decision_authority,
            self.confidence_authority,
            self.c_stream_authority,
            self.commit_authority,
            self.terminal_or_pareto_authority,
        )
        if (
            any(type(item) is not bool for item in required_true + required_false)
            or any(item is not True for item in required_true)
            or any(item is not False for item in required_false)
            or self.total_weight_quanta != self.common_denominator
            or self.authority_scope != SAMPLED_RESIDUAL_POPULATION_MEASURE_AUTHORITY
            or self.schema != SAMPLED_RESIDUAL_POPULATION_WEIGHT_RECEIPT_SCHEMA
        ):
            _refuse(
                V1SampledResidualPopulationMeasureRefusalCode.AUTHORITY_VIOLATION,
                "residual weight receipt changed its exact measure or authority",
            )
        observed = self._compute_receipt_hash()
        if (
            self._bound_receipt_hash is not None
            and self._bound_receipt_hash != observed
        ):
            _refuse(
                V1SampledResidualPopulationMeasureRefusalCode.STALE_IDENTITY,
                "residual population weights changed after source issuance",
            )
        object.__setattr__(self, "_bound_receipt_hash", observed)

    def _compute_receipt_hash(self) -> str:
        return content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            },
            prefix=_PREFIX + "weight-receipt-",
        )

    def assert_integrity(self) -> str:
        observed = self._compute_receipt_hash()
        if observed != self._bound_receipt_hash:
            _refuse(
                V1SampledResidualPopulationMeasureRefusalCode.STALE_IDENTITY,
                "residual population weight receipt failed its bound hash",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


def register_v1_sampled_residual_population_measure(
    *,
    suffix_registration: V1SampledVariableShellDW2SuffixRegistration,
) -> V1SampledResidualPopulationMeasureDesign:
    """Register the exact conditional population measure before outcomes."""

    if type(suffix_registration) is not V1SampledVariableShellDW2SuffixRegistration:
        raise TypeError("residual measure requires the exact D_W2 suffix registration")
    suffix_registration.assert_integrity()
    population = suffix_registration.population_stratum_counts
    d1 = suffix_registration.consumed_d1_stratum_counts
    residual = suffix_registration.suffix_capacity_stratum_counts
    if any(d1[index] + residual[index] != population[index] for index in (0, 1)):
        _refuse(
            V1SampledResidualPopulationMeasureRefusalCode.COUNT_MISMATCH,
            "suffix registration does not partition D1 from the residual population",
        )
    return V1SampledResidualPopulationMeasureDesign(
        suffix_registration_hash=suffix_registration.receipt_hash,
        generation_frame_hash=suffix_registration.generation_frame_hash,
        discovery_stream_hash=suffix_registration.discovery_stream_hash,
        current_mask_hash=suffix_registration.current_mask_hash,
        generation_index=suffix_registration.generation_index,
        population_stratum_counts=population,
        d1_stratum_counts=d1,
        residual_stratum_counts=residual,
        population_count=sum(population),
        d1_pair_prefix_axis_hash=suffix_registration.d1_pair_prefix_axis_hash,
        residual_pair_suffix_axis_hash=suffix_registration.suffix_pair_axis_hash,
        target_population_membership_hash=(
            suffix_registration.target_population_membership_hash
        ),
        complement_population_membership_hash=(
            suffix_registration.complement_population_membership_hash
        ),
        _issuance_token=_ISSUER_TOKEN,
    )


def issue_v1_sampled_residual_population_weights_for_packet(
    *,
    design: V1SampledResidualPopulationMeasureDesign,
    suffix_registration: V1SampledVariableShellDW2SuffixRegistration,
    packet: V1SampledVariableShellDW2SuffixPacket,
) -> V1SampledResidualPopulationWeightReceipt:
    """Issue exact cumulative D1/residual weights before packet outcomes."""

    if type(design) is not V1SampledResidualPopulationMeasureDesign:
        raise TypeError("residual weights require the exact measure design")
    if type(suffix_registration) is not V1SampledVariableShellDW2SuffixRegistration:
        raise TypeError("residual weights require the exact suffix registration")
    if type(packet) is not V1SampledVariableShellDW2SuffixPacket:
        raise TypeError("residual weights require the exact predictable packet")
    suffix_registration.assert_integrity()
    packet.assert_integrity()
    if (
        design.suffix_registration_hash != suffix_registration.receipt_hash
        or packet.suffix_registration_hash != suffix_registration.receipt_hash
        or packet.generation_frame_hash != design.generation_frame_hash
        or packet.discovery_stream_hash != design.discovery_stream_hash
        or design.population_stratum_counts
        != suffix_registration.population_stratum_counts
        or design.d1_stratum_counts != suffix_registration.consumed_d1_stratum_counts
        or design.residual_stratum_counts
        != suffix_registration.suffix_capacity_stratum_counts
    ):
        _refuse(
            V1SampledResidualPopulationMeasureRefusalCode.STALE_IDENTITY,
            "packet, suffix, and residual measure do not share one parent graph",
        )
    acquired = packet.after_suffix_stratum_counts
    residual = design.residual_stratum_counts
    if any(acquired[index] > residual[index] for index in (0, 1)):
        _refuse(
            V1SampledResidualPopulationMeasureRefusalCode.COUNT_MISMATCH,
            "cumulative D_W2 counts exceed the residual population",
        )
    for index in (0, 1):
        if residual[index] > 0 and acquired[index] == 0:
            _refuse(
                V1SampledResidualPopulationMeasureRefusalCode.INCOMPLETE_RESIDUAL_SAMPLE,
                "every positive residual stratum needs a predictable sampled prefix",
            )

    total = design.population_count
    d1_weight = Fraction(1, total)
    residual_masses = tuple(Fraction(count, total) for count in residual)
    residual_total = sum(residual)
    if residual_total <= 0:
        _refuse(
            V1SampledResidualPopulationMeasureRefusalCode.COUNT_MISMATCH,
            "D_W2 weights require a nonempty residual population",
        )
    residual_scale = Fraction(residual_total, total)
    dw2_weights = tuple(
        (
            Fraction(residual[index], total * acquired[index])
            if residual[index] > 0
            else Fraction(0)
        )
        for index in (0, 1)
    )
    residual_conditional_weights = tuple(
        value / residual_scale for value in dw2_weights
    )
    denominators = [d1_weight.denominator]
    denominators.extend(value.denominator for value in dw2_weights if value > 0)
    common_denominator = math.lcm(*denominators)
    d1_quantum = int(d1_weight * common_denominator)
    dw2_quanta = tuple(int(value * common_denominator) for value in dw2_weights)
    total_quanta = sum(design.d1_stratum_counts) * d1_quantum + sum(
        acquired[index] * dw2_quanta[index] for index in (0, 1)
    )
    if total_quanta != common_denominator:
        _refuse(
            V1SampledResidualPopulationMeasureRefusalCode.WEIGHT_MISMATCH,
            "D1 plus residual rational weights do not close the full population",
        )
    return V1SampledResidualPopulationWeightReceipt(
        measure_design_hash=design.design_hash,
        suffix_registration_hash=suffix_registration.receipt_hash,
        packet_hash=packet.receipt_hash,
        packet_ordinal=packet.packet_ordinal,
        cumulative_dw2_stratum_counts=acquired,
        population_stratum_counts=design.population_stratum_counts,
        d1_stratum_counts=design.d1_stratum_counts,
        residual_stratum_counts=residual,
        d1_context_weight_fraction=_fraction_pair(d1_weight),
        dw2_context_weight_fractions=tuple(
            _fraction_pair(value) for value in dw2_weights
        ),  # type: ignore[arg-type]
        residual_conditional_context_weight_fractions=tuple(
            _fraction_pair(value) for value in residual_conditional_weights
        ),  # type: ignore[arg-type]
        residual_stratum_mass_fractions=tuple(
            _fraction_pair(value) for value in residual_masses
        ),  # type: ignore[arg-type]
        residual_population_scale_fraction=_fraction_pair(residual_scale),
        d1_context_weight_binary64_enclosure=_binary64_fraction_enclosure(d1_weight),
        dw2_context_weight_binary64_enclosures=tuple(
            _binary64_fraction_enclosure(value) for value in dw2_weights
        ),  # type: ignore[arg-type]
        residual_conditional_weight_binary64_enclosures=tuple(
            _binary64_fraction_enclosure(value)
            for value in residual_conditional_weights
        ),  # type: ignore[arg-type]
        common_denominator=common_denominator,
        d1_context_quantum=d1_quantum,
        dw2_context_quanta=dw2_quanta,  # type: ignore[arg-type]
        total_weight_quanta=total_quanta,
        _issuance_token=_ISSUER_TOKEN,
    )


__all__ = [
    "SAMPLED_RESIDUAL_POPULATION_MEASURE_AUTHORITY",
    "SAMPLED_RESIDUAL_POPULATION_MEASURE_DESIGN_SCHEMA",
    "SAMPLED_RESIDUAL_POPULATION_WEIGHT_RECEIPT_SCHEMA",
    "V1SampledResidualPopulationMeasureDesign",
    "V1SampledResidualPopulationMeasureRefusal",
    "V1SampledResidualPopulationMeasureRefusalCode",
    "V1SampledResidualPopulationWeightReceipt",
    "issue_v1_sampled_residual_population_weights_for_packet",
    "register_v1_sampled_residual_population_measure",
]
