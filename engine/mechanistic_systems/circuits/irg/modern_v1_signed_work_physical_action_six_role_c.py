"""Branchless six-role C evidence for unique signed-work physical actions.

This module is the statistical owner immediately downstream of the physical
action axis and its independent two-dose direction provider.  Its family is
exactly

``unique physical action x complete-train C stratum x six roles x two tails``.

Coordinate, KKT-branch, and pre-C alias identities are deliberately absent
from that family.  Aliases are measured once and may receive results only via
the separate physical-axis fanout receipt.  Independent actions need not be
nested and no confidence or census logic assumes monotonicity between them.

The six nonnegative per-image observables are, in canonical order,
``F_native, Q_J, R_half, R_one, D_half, D_one``.  ``F_native`` first performs
signed cancellation inside each registered native fiber and only then takes
absolute value; ``Q_J`` squares the literal relaxed-gate Jacobian projection.
The historical profile uses absolute ``R`` and ``D`` observations and weights
strata by their population sizes. The explicit balanced-RMS profile squares
both observations, uses equal target/complement mass, and takes their roots
only after combining stratum moment bounds. F remains mean work and Q remains
a second moment. These laws have distinct source-bound receipt identities.
Under the registered without-replacement sampling and support assumptions,
exact binary64 fixed-point sums and outward Hoeffding--Serfling intervals with
the registered look-alpha schedule support optional stopping. A complete census
collapses to its outward numerical envelope without spending look alpha.

All types are measurement evidence only.  They cannot select an action, open
K or O, calibrate, query an oracle, commit a mask, or claim terminal/Pareto
authority.
"""

from __future__ import annotations

from .modern_v1_physical_metric import (
    six_role_observation_bounds,
    LEGACY_UNIFORM_ABSOLUTE, BALANCED_RMS, balanced_role_bound,
    profile_coordinates, validate_physical_measurement_profile, restore_profile_state,
)

from dataclasses import dataclass, field
from decimal import localcontext
from enum import Enum
from fractions import Fraction
import math
from numbers import Integral, Real
from typing import Final, Literal, NoReturn, Sequence

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_fibered_directional_record import (
    _absolute_interval,
    _decimal,
    _outward_float,
)
from .modern_v1_fqr_sequential_acquisition import (
    _binary64_scaled_integer,
    _downward_fraction_to_probability,
    _fraction_to_float_outward,
    _response_burden,
    _response_burden_lower,
    _scaled_mean,
    _serfling_radius_upper,
)
from .modern_v1_imagenet_population import IMAGENET1K_TRAIN_CONTEXT_COUNT
from .modern_v1_signed_work_physical_action_c_axis import (
    V1SignedWorkPhysicalActionCAxis,
    V1SignedWorkPhysicalActionCAxisEntry,
)
from .modern_v1_signed_work_physical_action_direction_provider import (
    V1PhysicalActionContextExecutionReceipt,
    V1PhysicalActionDirectionReceipt,
)
from .modern_v1_signed_work_six_role_c_acquisition import (
    D_HALF_ROLE,
    D_ONE_ROLE,
    F_NATIVE_ROLE,
    Q_J_ROLE,
    R_HALF_ROLE,
    R_ONE_ROLE,
    SIGNED_WORK_SIX_ROLE_DOSES,
    SIGNED_WORK_SIX_ROLES,
)


PHYSICAL_ACTION_SIX_ROLE_C_STRATUM_SCHEMA: Final = (
    "partition_free_v1_physical_action_six_role_C_stratum_v1"
)
PHYSICAL_ACTION_SIX_ROLE_C_SOURCE_SCHEMA: Final = (
    "partition_free_v1_physical_action_six_role_C_source_execution_v2"
)
PHYSICAL_ACTION_SIX_ROLE_C_DESIGN_SCHEMA: Final = (
    "partition_free_v1_physical_action_six_role_C_design_v1"
)
PHYSICAL_ACTION_SIX_ROLE_C_COORDINATE_SCHEMA: Final = (
    "partition_free_v1_physical_action_six_role_C_coordinate_v1"
)
PHYSICAL_ACTION_SIX_ROLE_C_ROW_SCHEMA: Final = (
    "partition_free_v1_physical_action_six_role_C_row_v1"
)
PHYSICAL_ACTION_SIX_ROLE_C_CELL_SCHEMA: Final = (
    "partition_free_v1_physical_action_six_role_C_cell_v1"
)
PHYSICAL_ACTION_SIX_ROLE_C_ACTION_SCHEMA: Final = (
    "partition_free_v1_physical_action_six_role_C_action_summary_v1"
)
PHYSICAL_ACTION_SIX_ROLE_C_PROGRESS_SCHEMA: Final = (
    "partition_free_v1_physical_action_six_role_C_progress_v1"
)
PHYSICAL_ACTION_SIX_ROLE_C_SNAPSHOT_SCHEMA: Final = (
    "partition_free_v1_physical_action_six_role_C_snapshot_v1"
)
PHYSICAL_ACTION_SIX_ROLE_C_AUTHORITY: Final = (
    "fresh_independent_complete_train_C_unique_physical_action_native_F_"
    "literal_J_Q_directional_R_and_direct_D_measurement_evidence_only_no_"
    "selection_K_O_calibration_oracle_commit_terminal_or_pareto_authority_v1"
)
PHYSICAL_ACTION_SIX_ROLE_C_ALPHA_POLICY: Final = (
    "preoutcome_unique_physical_action_stratum_six_role_two_tail_"
    "nonrecycling_all_prefix_v1"
)
PHYSICAL_ACTION_SIX_ROLE_C_SAMPLING_LAW: Final = (
    "sealed_without_replacement_prefix_of_complete_registered_train_C_v1"
)
PHYSICAL_ACTION_SIX_ROLE_C_INTERVAL_METHOD: Final = (
    "separate_one_sided_lower_and_upper_hoeffding_serfling_v1"
)
PHYSICAL_ACTION_SIX_ROLE_C_CENSUS_METHOD: Final = (
    "exact_finite_population_census_of_outward_row_envelopes_v1"
)

_STRATUM_PREFIX: Final = "modern-v1-physical-action-six-role-C-stratum-"
_SOURCE_PREFIX: Final = "modern-v1-physical-action-six-role-C-source-"
_DESIGN_PREFIX: Final = "modern-v1-physical-action-six-role-C-design-"
_COORDINATE_PREFIX: Final = "modern-v1-physical-action-six-role-C-coordinate-"
_ROW_PREFIX: Final = "modern-v1-physical-action-six-role-C-row-"
_CELL_PREFIX: Final = "modern-v1-physical-action-six-role-C-cell-"
_ACTION_PREFIX: Final = "modern-v1-physical-action-six-role-C-action-"
_PROGRESS_PREFIX: Final = "modern-v1-physical-action-six-role-C-progress-"
_SNAPSHOT_PREFIX: Final = "modern-v1-physical-action-six-role-C-snapshot-"
_CHAIN_PREFIX: Final = "modern-v1-physical-action-six-role-C-chain-"
_LE_F64: Final = np.dtype("<f8")
_ROW_ISSUER_TOKEN: Final = object()

PhysicalActionSixRoleSamplingStatus = Literal[
    "awaiting_C",
    "partial_C_evidence",
    "full_C_census",
]


class V1PhysicalActionSixRoleCRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_SOURCE = "stale_source"
    STALE_COORDINATE = "stale_coordinate"
    ACTION_AXIS_MISMATCH = "action_axis_mismatch"
    DIRECTION_RECEIPT_MISMATCH = "direction_receipt_mismatch"
    STREAM_IDENTITY_MISMATCH = "stream_identity_mismatch"
    INTERVAL_INVALID = "interval_invalid"
    SUPPORT_OVERFLOW = "support_overflow"
    DUPLICATE_ROW = "duplicate_row"
    POSTSELECTION = "postselection"
    SNAPSHOT_MISMATCH = "snapshot_mismatch"
    AUTHORITY_VIOLATION = "authority_violation"


class V1PhysicalActionSixRoleCRefusal(ValueError):
    def __init__(
        self,
        code: V1PhysicalActionSixRoleCRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1PhysicalActionSixRoleCRefusalCode:
            raise TypeError("physical-action six-role refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1PhysicalActionSixRoleCRefusalCode,
    message: str,
) -> NoReturn:
    raise V1PhysicalActionSixRoleCRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1PhysicalActionSixRoleCRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1PhysicalActionSixRoleCRefusalCode.INVALID_TYPE,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1PhysicalActionSixRoleCRefusalCode.STALE_COORDINATE,
            f"{name} must be nonnegative",
        )
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        _refuse(
            V1PhysicalActionSixRoleCRefusalCode.STALE_COORDINATE,
            f"{name} must be positive",
        )
    return result


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1PhysicalActionSixRoleCRefusalCode.INVALID_TYPE,
            f"{name} must be one real scalar",
        )
    result = float(value)
    if not math.isfinite(result):
        _refuse(
            V1PhysicalActionSixRoleCRefusalCode.INTERVAL_INVALID,
            f"{name} must be finite",
        )
    return 0.0 if result == 0.0 else result


def _finite_nonnegative(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if result < 0.0:
        _refuse(
            V1PhysicalActionSixRoleCRefusalCode.INTERVAL_INVALID,
            f"{name} must be nonnegative",
        )
    return result


def _probability(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if not 0.0 < result < 1.0:
        _refuse(
            V1PhysicalActionSixRoleCRefusalCode.INTERVAL_INVALID,
            f"{name} must lie strictly inside (0, 1)",
        )
    return result


def _immutable_f64(
    value: object,
    *,
    shape: tuple[int, ...],
    name: str,
    nonnegative: bool = False,
) -> np.ndarray:
    try:
        array = np.array(value, dtype=_LE_F64, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        _refuse(
            V1PhysicalActionSixRoleCRefusalCode.INVALID_TYPE,
            f"{name} must be convertible to binary64: {error}",
        )
    if array.shape != shape or not bool(np.isfinite(array).all()):
        _refuse(
            V1PhysicalActionSixRoleCRefusalCode.INTERVAL_INVALID,
            f"{name} must be finite with shape {shape}",
        )
    if nonnegative and bool(np.any(array < 0.0)):
        _refuse(
            V1PhysicalActionSixRoleCRefusalCode.INTERVAL_INVALID,
            f"{name} must be nonnegative",
        )
    array[array == 0.0] = 0.0
    result = np.frombuffer(array.tobytes(order="C"), dtype=_LE_F64).reshape(shape)
    result.setflags(write=False)
    return result


def _closed_authority(values: Sequence[object], *, owner: str) -> None:
    if any(type(value) is not bool for value in values) or any(values):
        _refuse(
            V1PhysicalActionSixRoleCRefusalCode.AUTHORITY_VIOLATION,
            f"{owner} crossed its measurement-only authority boundary",
        )


def _population_weighted_sum(
    *,
    population_sizes: Sequence[int],
    values: Sequence[float],
    direction: Literal["lower", "upper"],
) -> float:
    """Apply exact finite-population weights before one outward conversion."""

    sizes = tuple(int(value) for value in population_sizes)
    observations = tuple(float(value) for value in values)
    if (
        not sizes
        or len(sizes) != len(observations)
        or any(value <= 0 for value in sizes)
        or any(not math.isfinite(value) or value < 0.0 for value in observations)
    ):
        _refuse(
            V1PhysicalActionSixRoleCRefusalCode.INTERVAL_INVALID,
            "population-weighted interval inputs are invalid",
        )
    population = sum(sizes)
    exact = sum(
        (
            Fraction(size, population) * Fraction.from_float(value)
            for size, value in zip(sizes, observations, strict=True)
        ),
        Fraction(0, 1),
    )
    result = _fraction_to_float_outward(exact, direction=direction)
    return max(0.0, result) if direction == "lower" else result


def _source_authority_flags(owner: object) -> tuple[object, ...]:
    names = (
        "C_outcomes_seen",
        "validation_outcomes_seen",
        "candidate_selection_authority",
        "calibration_K_open_authority",
        "paired_oracle_O_open_authority",
        "calibration_authority",
        "oracle_authority",
        "mask_commit_authority",
        "terminal_authority",
        "pareto_authority",
    )
    return tuple(getattr(owner, name) for name in names if hasattr(owner, name))


@dataclass(frozen=True, slots=True)
class V1PhysicalActionSixRoleCStratum:
    """One complete, pre-outcome stratum of the sealed train permutation."""

    stratum_id: str
    population_size: int
    ordered_context_axis_hash: str
    sealed_draw_axis_hash: str
    coordinate_proof_commitment_hash: str
    issued_before_C_outcomes: bool = True
    C_outcomes_seen: bool = False
    schema: str = PHYSICAL_ACTION_SIX_ROLE_C_STRATUM_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "stratum_id",
            "ordered_context_axis_hash",
            "sealed_draw_axis_hash",
            "coordinate_proof_commitment_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "population_size",
            _positive_int(self.population_size, name="C stratum population"),
        )
        if (
            type(self.issued_before_C_outcomes) is not bool
            or not self.issued_before_C_outcomes
            or type(self.C_outcomes_seen) is not bool
            or self.C_outcomes_seen
        ):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.POSTSELECTION,
                "C stratum was not frozen before outcomes",
            )
        if self.schema != PHYSICAL_ACTION_SIX_ROLE_C_STRATUM_SCHEMA:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.INVALID_TYPE,
                "unsupported physical-action C stratum schema",
            )

    @property
    def stratum_hash(self) -> str:
        return content_hash(self, prefix=_STRATUM_PREFIX)


@dataclass(frozen=True, slots=True)
class V1PhysicalActionSixRoleCSourceRegistration:
    """Pre-outcome coordinate and execution identities for physical-action C.

    ``raw_logit_coordinate_identity_hash`` is inherited from the frozen D
    chart.  The v2 ABI's historical ``raw_logit_stream_identity_hash`` field
    names the separately derived finite-direction C execution stream; the
    explicit property below is the canonical semantic spelling.
    """

    physical_action_axis_hash: str
    direction_provider_receipt_hash: str
    c_stream_hash: str
    complete_train_population_hash: str
    chart_identity_hash: str
    current_parent_mask_hash: str
    route_axis_hash: str
    route_count: int
    proposal_generation: int
    corrected_evidence_stream_identity_hash: str
    literal_jacobian_stream_identity_hash: str
    raw_logit_coordinate_identity_hash: str
    raw_logit_stream_identity_hash: str
    same_capture_scorer_registration_hash: str
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    C_coordinate_proof_source_registration_hash: str
    C_execution_registration_hash: str
    issued_before_C_outcomes: bool = True
    C_outcomes_seen: bool = False
    observation_values_registered: bool = False
    candidate_selection_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = PHYSICAL_ACTION_SIX_ROLE_C_AUTHORITY
    schema: str = PHYSICAL_ACTION_SIX_ROLE_C_SOURCE_SCHEMA
    _registration_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "physical_action_axis_hash",
            "direction_provider_receipt_hash",
            "c_stream_hash",
            "complete_train_population_hash",
            "chart_identity_hash",
            "current_parent_mask_hash",
            "route_axis_hash",
            "corrected_evidence_stream_identity_hash",
            "literal_jacobian_stream_identity_hash",
            "raw_logit_coordinate_identity_hash",
            "raw_logit_stream_identity_hash",
            "same_capture_scorer_registration_hash",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
            "C_coordinate_proof_source_registration_hash",
            "C_execution_registration_hash",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="registered route count"),
        )
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(
                self.proposal_generation,
                name="registered proposal generation",
            ),
        )
        streams = (
            self.corrected_evidence_stream_identity_hash,
            self.literal_jacobian_stream_identity_hash,
            self.raw_logit_stream_identity_hash,
        )
        if (
            len(set(streams)) != 3
            or self.raw_logit_stream_identity_hash
            == self.raw_logit_coordinate_identity_hash
        ):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.STREAM_IDENTITY_MISMATCH,
                "corrected evidence, literal J, and finite-direction raw logits "
                "must be distinct streams; the finite-direction stream must also "
                "be distinct from its frozen raw-logit coordinate",
            )
        flags = (
            self.C_outcomes_seen,
            self.observation_values_registered,
            self.candidate_selection_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if type(self.issued_before_C_outcomes) is not bool or not (
            self.issued_before_C_outcomes
        ):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.POSTSELECTION,
                "C source identities were not frozen before outcomes",
            )
        _closed_authority(flags, owner="physical-action C source registration")
        if (
            self.authority_scope != PHYSICAL_ACTION_SIX_ROLE_C_AUTHORITY
            or self.schema != PHYSICAL_ACTION_SIX_ROLE_C_SOURCE_SCHEMA
        ):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.AUTHORITY_VIOLATION,
                "unsupported C source authority or schema",
            )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "physical_action_axis_hash",
                "direction_provider_receipt_hash",
                "c_stream_hash",
                "complete_train_population_hash",
                "chart_identity_hash",
                "current_parent_mask_hash",
                "route_axis_hash",
                "route_count",
                "proposal_generation",
                "corrected_evidence_stream_identity_hash",
                "literal_jacobian_stream_identity_hash",
                "raw_logit_coordinate_identity_hash",
                "raw_logit_stream_identity_hash",
                "same_capture_scorer_registration_hash",
                "native_fiber_axis_receipt_hash",
                "native_fiber_measure_hash",
                "C_coordinate_proof_source_registration_hash",
                "C_execution_registration_hash",
                "issued_before_C_outcomes",
                "C_outcomes_seen",
                "observation_values_registered",
                "candidate_selection_authority",
                "calibration_K_open_authority",
                "paired_oracle_O_open_authority",
                "mask_commit_authority",
                "terminal_authority",
                "pareto_authority",
                "authority_scope",
            )
        }

    @property
    def registration_hash(self) -> str:
        return self.assert_integrity()

    @property
    def finite_direction_raw_logit_execution_stream_identity_hash(self) -> str:
        return self.raw_logit_stream_identity_hash

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_SOURCE_PREFIX)
        cached = self._registration_hash
        if cached is not None and cached != observed:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.STALE_SOURCE,
                "physical-action C source registration was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.STALE_SOURCE,
                "physical-action C source registration hash differs",
            )
        object.__setattr__(self, "_registration_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1PhysicalActionSixRoleCDesign:
    """Complete branchless C family frozen before any C observation."""

    design_id: str
    physical_action_axis: V1SignedWorkPhysicalActionCAxis = field(
        repr=False,
        compare=False,
    )
    direction_provider_receipt: V1PhysicalActionDirectionReceipt = field(
        repr=False,
        compare=False,
    )
    source_registration: V1PhysicalActionSixRoleCSourceRegistration = field(
        repr=False,
        compare=False,
    )
    strata: tuple[V1PhysicalActionSixRoleCStratum, ...]
    analytic_support_upper_by_stratum_action_role: np.ndarray = field(
        repr=False,
        compare=False,
    )
    support_bound_source_hashes_by_role: tuple[tuple[str, str], ...]
    six_role_family_alpha_allocation: float
    physical_c_alpha_allocation_receipt: object | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    private_small_population_reference: bool = False
    alpha_policy: str = PHYSICAL_ACTION_SIX_ROLE_C_ALPHA_POLICY
    sampling_law: str = PHYSICAL_ACTION_SIX_ROLE_C_SAMPLING_LAW
    issued_before_C_outcomes: bool = True
    C_outcomes_seen: bool = False
    validation_outcomes_seen: bool = False
    aliases_affect_alpha_family: bool = False
    coordinate_or_branch_axis_registered: bool = False
    nesting_or_monotonicity_assumed: bool = False
    candidate_selection_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = PHYSICAL_ACTION_SIX_ROLE_C_AUTHORITY
    schema: str = PHYSICAL_ACTION_SIX_ROLE_C_DESIGN_SCHEMA
    physical_action_entry_hashes: tuple[str, ...] = field(init=False)
    canonical_action_hashes: tuple[str, ...] = field(init=False)
    parent_mask_hash: str = field(init=False)
    route_axis_hash: str = field(init=False)
    route_count: int = field(init=False)
    proposal_generation: int = field(init=False)
    complete_train_population_hash: str = field(init=False)
    c_stream_hash: str = field(init=False)
    trajectory_alpha_ledger_hash: str = field(init=False)
    trajectory_alpha_ledger_generation: int = field(init=False)
    trajectory_familywise_alpha: float = field(init=False)
    trajectory_alpha_committed_before_C: float = field(init=False)
    population_size: int = field(init=False)
    cell_count: int = field(init=False)
    tail_count: int = field(init=False)
    tail_family_alpha: float = field(init=False)
    registered_tail_alpha_total: float = field(init=False)
    _design_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )
    physical_measurement_profile: str = field(default=LEGACY_UNIFORM_ABSOLUTE, kw_only=True)

    def __setstate__(self, state: list[object]) -> None:
        restore_profile_state(self, state)

    def __post_init__(self) -> None:
        validate_physical_measurement_profile(self.physical_measurement_profile)
        if self.physical_measurement_profile == BALANCED_RMS and (
            len(self.strata) != 2 or {v.stratum_id for v in self.strata} != {"target", "complement"}
        ):
            raise ValueError("balanced RMS C requires target and complement strata")
        object.__setattr__(
            self,
            "design_id",
            _identity(self.design_id, name="physical-action six-role design id"),
        )
        axis = self.physical_action_axis
        receipt = self.direction_provider_receipt
        source = self.source_registration
        if type(axis) is not V1SignedWorkPhysicalActionCAxis:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.INVALID_TYPE,
                "six-role design requires the exact physical-action C axis",
            )
        if type(receipt) is not V1PhysicalActionDirectionReceipt:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.INVALID_TYPE,
                "six-role design requires the exact direction-provider receipt",
            )
        if type(source) is not V1PhysicalActionSixRoleCSourceRegistration:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.INVALID_TYPE,
                "six-role design requires the exact source registration",
            )
        axis_hash = axis.assert_integrity()
        receipt_hash = receipt.assert_integrity()
        source_hash = source.assert_integrity()
        strata = tuple(self.strata)
        if not strata or any(
            type(value) is not V1PhysicalActionSixRoleCStratum for value in strata
        ):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.INVALID_TYPE,
                "six-role design requires exact nonempty C strata",
            )
        stratum_ids = tuple(value.stratum_id for value in strata)
        if len(stratum_ids) != len(set(stratum_ids)):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.STALE_COORDINATE,
                "C stratum identities repeat",
            )
        population_size = sum(value.population_size for value in strata)
        if type(self.private_small_population_reference) is not bool:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.INVALID_TYPE,
                "small-population reference marker must be exact",
            )
        if self.private_small_population_reference:
            if receipt.production_execution:
                _refuse(
                    V1PhysicalActionSixRoleCRefusalCode.AUTHORITY_VIOLATION,
                    "a private small-population reference cannot claim production",
                )
        elif (
            population_size != IMAGENET1K_TRAIN_CONTEXT_COUNT
            or population_size != axis.complete_population.train_context_count
        ):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.STALE_COORDINATE,
                "C strata must partition the complete ImageNet train population",
            )
        actions = tuple(axis.physical_actions)
        if not actions or any(
            type(value) is not V1SignedWorkPhysicalActionCAxisEntry for value in actions
        ):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.ACTION_AXIS_MISMATCH,
                "physical-action C axis is empty or changed type",
            )
        entry_hashes = tuple(value.entry_hash for value in actions)
        canonical_hashes = tuple(value.canonical_action_hash for value in actions)
        direction_design = receipt.design
        if (
            receipt.source_family_hash != axis.packed_action_family_hash
            or direction_design.source_family_hash != axis.packed_action_family_hash
            or direction_design.parent_mask_hash != axis.parent_mask_hash
            or direction_design.route_axis_hash != axis.route_axis_hash
            or direction_design.route_count != axis.route_count
            or direction_design.canonical_action_hashes != canonical_hashes
            or receipt.unique_physical_action_count != len(actions)
            or receipt.alias_count != axis.upstream_q_safe_alias_count
        ):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.DIRECTION_RECEIPT_MISMATCH,
                "direction provider does not execute the exact physical-action axis",
            )
        ledger = axis.alpha_ledger_head
        physical_c_alpha = self.physical_c_alpha_allocation_receipt
        try:
            ledger_hash = ledger.assert_integrity()
        except (TypeError, ValueError, RuntimeError) as error:
            raise V1PhysicalActionSixRoleCRefusal(
                V1PhysicalActionSixRoleCRefusalCode.STALE_SOURCE,
                "trajectory alpha ledger is invalid or exhausted",
            ) from error
        if physical_c_alpha is None:
            # Historical v1 readers retain their byte-compatible proposal-ledger
            # interpretation.  Canonical issuance is owned by the separate
            # physical-action C registrar and always supplies the receipt below.
            try:
                allocation = ledger.generation_alpha_allocation
            except (TypeError, ValueError, RuntimeError) as error:
                raise V1PhysicalActionSixRoleCRefusal(
                    V1PhysicalActionSixRoleCRefusalCode.STALE_SOURCE,
                    "legacy trajectory alpha ledger is exhausted",
                ) from error
            alpha_source_hash = ledger_hash
            alpha_generation = ledger.proposal_generation
            alpha_familywise = ledger.schedule.trajectory_familywise_alpha
            alpha_committed = ledger.cumulative_committed_alpha
        else:
            # Local import avoids making the registration module and this
            # measurement design import each other while module definitions are
            # still loading.
            from .modern_v1_signed_work_physical_action_c_registration import (
                V1PhysicalActionCAlphaAllocationReceipt,
            )

            if type(physical_c_alpha) is not V1PhysicalActionCAlphaAllocationReceipt:
                _refuse(
                    V1PhysicalActionSixRoleCRefusalCode.INVALID_TYPE,
                    "canonical physical C requires its exact alpha receipt",
                )
            alpha_source_hash = physical_c_alpha.assert_integrity()
            allocation = physical_c_alpha.physical_c_family_alpha_allocation
            alpha_generation = physical_c_alpha.proposal_generation
            alpha_familywise = physical_c_alpha.fq_familywise_alpha
            alpha_committed = math.fsum(
                (
                    physical_c_alpha.fq_alpha_committed_before_generation,
                    physical_c_alpha.fq_alpha_committed_before_score_look,
                )
            )
            if (
                physical_c_alpha.physical_action_axis_hash != axis_hash
                or physical_c_alpha.d_freeze_hash != axis.d_freeze_hash
                or physical_c_alpha.proposal_order_ledger_lineage_hash != ledger_hash
                or physical_c_alpha.proposal_order_alpha_schedule_hash
                != ledger.schedule_hash
                or physical_c_alpha.certification_stream_hash
                != axis.c_stream.certification_stream_commitment_hash
                or alpha_generation != axis.alpha_ledger_generation
                or physical_c_alpha.proposal_order_D_alpha_reused_for_C is not False
                or physical_c_alpha.next_proposal_generation_alpha_consumed is not False
            ):
                _refuse(
                    V1PhysicalActionSixRoleCRefusalCode.AUTHORITY_VIOLATION,
                    "canonical physical-C alpha differs from its D/C/action lineage",
                )
        supplied_allocation = _probability(
            self.six_role_family_alpha_allocation,
            name="six-role family alpha allocation",
        )
        if supplied_allocation != allocation:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.AUTHORITY_VIOLATION,
                "six-role family must consume its registered physical-C allocation exactly",
            )
        if Fraction.from_float(alpha_committed) + Fraction.from_float(
            allocation
        ) > Fraction.from_float(alpha_familywise):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.AUTHORITY_VIOLATION,
                "six-role family overspends the trajectory alpha schedule",
            )
        action_count = len(actions)
        support = _immutable_f64(
            self.analytic_support_upper_by_stratum_action_role,
            shape=(len(strata), action_count, len(SIGNED_WORK_SIX_ROLES)),
            name="physical-action six-role analytic support",
            nonnegative=True,
        )
        support_sources = tuple(
            (
                _identity(role, name="support role"),
                _identity(source_hash, name="support source hash"),
            )
            for role, source_hash in self.support_bound_source_hashes_by_role
        )
        if tuple(role for role, _hash in support_sources) != SIGNED_WORK_SIX_ROLES:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.STALE_SOURCE,
                "support sources must cover the canonical six-role axis in order",
            )
        expected_source = {
            "physical_action_axis_hash": axis_hash,
            "direction_provider_receipt_hash": receipt_hash,
            "c_stream_hash": axis.c_stream_hash,
            "complete_train_population_hash": axis.complete_population_hash,
            "current_parent_mask_hash": axis.parent_mask_hash,
            "route_axis_hash": axis.route_axis_hash,
            "route_count": axis.route_count,
            "proposal_generation": axis.alpha_ledger_generation,
        }
        stale = tuple(
            name
            for name, expected in expected_source.items()
            if getattr(source, name) != expected
        )
        if stale:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.STALE_SOURCE,
                "C source registration differs at: " + ", ".join(stale),
            )
        if (
            source.c_stream_hash != axis.c_stream.identity_hash
            or source.complete_train_population_hash
            != axis.complete_population.content_hash
            or source.registration_hash != source_hash
        ):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.STALE_SOURCE,
                "C source registration no longer replays its stream/population",
            )
        cells = len(strata) * action_count * len(SIGNED_WORK_SIX_ROLES)
        tails = 2 * cells
        tail_alpha = _downward_fraction_to_probability(
            Fraction.from_float(allocation) / tails
        )
        registered = math.fsum(tail_alpha for _ in range(tails))
        if registered > allocation:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.AUTHORITY_VIOLATION,
                "rounded tail cells overspend the registered family allocation",
            )
        authority = (
            self.C_outcomes_seen,
            self.validation_outcomes_seen,
            self.aliases_affect_alpha_family,
            self.coordinate_or_branch_axis_registered,
            self.nesting_or_monotonicity_assumed,
            self.candidate_selection_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if type(self.issued_before_C_outcomes) is not bool or not (
            self.issued_before_C_outcomes
        ):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.POSTSELECTION,
                "six-role C design was not frozen before outcomes",
            )
        _closed_authority(authority, owner="physical-action six-role C design")
        if (
            self.alpha_policy != PHYSICAL_ACTION_SIX_ROLE_C_ALPHA_POLICY
            or self.sampling_law != PHYSICAL_ACTION_SIX_ROLE_C_SAMPLING_LAW
            or self.authority_scope != PHYSICAL_ACTION_SIX_ROLE_C_AUTHORITY
            or self.schema != PHYSICAL_ACTION_SIX_ROLE_C_DESIGN_SCHEMA
        ):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.AUTHORITY_VIOLATION,
                "unsupported sampling, alpha, authority, or schema semantics",
            )
        derived = {
            "strata": strata,
            "analytic_support_upper_by_stratum_action_role": support,
            "support_bound_source_hashes_by_role": support_sources,
            "six_role_family_alpha_allocation": allocation,
            "physical_action_entry_hashes": entry_hashes,
            "canonical_action_hashes": canonical_hashes,
            "parent_mask_hash": axis.parent_mask_hash,
            "route_axis_hash": axis.route_axis_hash,
            "route_count": axis.route_count,
            "proposal_generation": axis.alpha_ledger_generation,
            "complete_train_population_hash": axis.complete_population_hash,
            "c_stream_hash": axis.c_stream_hash,
            "trajectory_alpha_ledger_hash": alpha_source_hash,
            "trajectory_alpha_ledger_generation": alpha_generation,
            "trajectory_familywise_alpha": alpha_familywise,
            "trajectory_alpha_committed_before_C": alpha_committed,
            "population_size": population_size,
            "cell_count": cells,
            "tail_count": tails,
            "tail_family_alpha": tail_alpha,
            "registered_tail_alpha_total": registered,
        }
        for name, value in derived.items():
            object.__setattr__(self, name, value)
        self.assert_integrity()

    def aggregate_role_bound(self, *, role: str, values: tuple[float, ...], direction: Literal["lower", "upper"]) -> float:
        if role not in SIGNED_WORK_SIX_ROLES:
            raise ValueError("unknown physical C role")
        validate_physical_measurement_profile(self.physical_measurement_profile)
        if self.physical_measurement_profile == BALANCED_RMS:
            return balanced_role_bound(
                stratum_ids=self.stratum_ids, values=values,
                take_root=role in (R_HALF_ROLE, R_ONE_ROLE, D_HALF_ROLE, D_ONE_ROLE),
                direction=direction,
            )
        return _population_weighted_sum(population_sizes=self.population_sizes, values=values, direction=direction)

    @property
    def stratum_ids(self) -> tuple[str, ...]:
        return tuple(value.stratum_id for value in self.strata)

    @property
    def population_sizes(self) -> tuple[int, ...]:
        return tuple(value.population_size for value in self.strata)

    @property
    def alpha_coordinates(self) -> tuple[tuple[str, str, str, str], ...]:
        return tuple(
            (action, stratum, role, tail)
            for action in self.physical_action_entry_hashes
            for stratum in self.stratum_ids
            for role in SIGNED_WORK_SIX_ROLES
            for tail in ("lower", "upper")
        )

    def look_alpha(self, *, stratum_index: int, sample_size: int) -> float:
        index = _nonnegative_int(stratum_index, name="C stratum index")
        if index >= len(self.strata):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.STALE_COORDINATE,
                "C stratum index left the registered axis",
            )
        count = _positive_int(sample_size, name="C prefix sample size")
        population = self.strata[index].population_size
        if count > population:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.STALE_COORDINATE,
                "C prefix exceeds its finite stratum population",
            )
        if count == population:
            return 0.0
        # Sum_{n=1}^{N-1} N/[n(n+1)(N-1)] = 1.  Thus each cell/tail owns
        # a complete pre-outcome all-prefix alpha schedule.
        schedule = Fraction(population, count * (count + 1) * (population - 1))
        return _downward_fraction_to_probability(
            Fraction.from_float(self.tail_family_alpha) * schedule
        )

    def _payload(self) -> dict[str, object]:
        payload = {
            "schema": self.schema,
            "design_id": self.design_id,
            "physical_action_axis_hash": self.physical_action_axis.axis_hash,
            "direction_provider_receipt_hash": (
                self.direction_provider_receipt.receipt_hash
            ),
            "source_registration_hash": self.source_registration.registration_hash,
            "strata": self.strata,
            "analytic_support_upper_by_stratum_action_role": (
                self.analytic_support_upper_by_stratum_action_role
            ),
            "support_bound_source_hashes_by_role": (
                self.support_bound_source_hashes_by_role
            ),
            "physical_action_entry_hashes": self.physical_action_entry_hashes,
            "canonical_action_hashes": self.canonical_action_hashes,
            "parent_mask_hash": self.parent_mask_hash,
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "proposal_generation": self.proposal_generation,
            "complete_train_population_hash": self.complete_train_population_hash,
            "c_stream_hash": self.c_stream_hash,
            "trajectory_alpha_ledger_hash": self.trajectory_alpha_ledger_hash,
            "trajectory_alpha_ledger_generation": (
                self.trajectory_alpha_ledger_generation
            ),
            "trajectory_familywise_alpha": self.trajectory_familywise_alpha,
            "trajectory_alpha_committed_before_C": (
                self.trajectory_alpha_committed_before_C
            ),
            "six_role_family_alpha_allocation": (self.six_role_family_alpha_allocation),
            "private_small_population_reference": (
                self.private_small_population_reference
            ),
            "roles": SIGNED_WORK_SIX_ROLES,
            "alpha_coordinates": self.alpha_coordinates,
            "population_size": self.population_size,
            "cell_count": self.cell_count,
            "tail_count": self.tail_count,
            "tail_family_alpha": self.tail_family_alpha,
            "registered_tail_alpha_total": self.registered_tail_alpha_total,
            "alpha_policy": self.alpha_policy,
            "sampling_law": self.sampling_law,
            "issued_before_C_outcomes": self.issued_before_C_outcomes,
            "C_outcomes_seen": self.C_outcomes_seen,
            "validation_outcomes_seen": self.validation_outcomes_seen,
            "aliases_affect_alpha_family": self.aliases_affect_alpha_family,
            "coordinate_or_branch_axis_registered": (
                self.coordinate_or_branch_axis_registered
            ),
            "nesting_or_monotonicity_assumed": (self.nesting_or_monotonicity_assumed),
            "candidate_selection_authority": self.candidate_selection_authority,
            "calibration_K_open_authority": self.calibration_K_open_authority,
            "paired_oracle_O_open_authority": (self.paired_oracle_O_open_authority),
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
            "authority_scope": self.authority_scope,
        }
        if self.physical_c_alpha_allocation_receipt is not None:
            payload.update(
                {
                    "physical_c_alpha_allocation_receipt_hash": (
                        self.physical_c_alpha_allocation_receipt.receipt_hash
                    ),
                    "proposal_order_ledger_is_D_lineage_only": True,
                    "next_proposal_generation_alpha_consumed": False,
                }
            )
        payload.update(profile_coordinates(self.physical_measurement_profile))
        return payload

    @property
    def design_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.physical_action_axis.assert_integrity()
        self.direction_provider_receipt.assert_integrity()
        self.source_registration.assert_integrity()
        if self.physical_c_alpha_allocation_receipt is not None:
            self.physical_c_alpha_allocation_receipt.assert_integrity()
        observed = content_hash(self._payload(), prefix=_DESIGN_PREFIX)
        cached = self._design_hash
        if cached is not None and cached != observed:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.STALE_SOURCE,
                "physical-action six-role design was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.STALE_SOURCE,
                "physical-action six-role design hash differs",
            )
        object.__setattr__(self, "_design_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1PhysicalActionSixRoleCCoordinateRegistration:
    """One sealed C prefix coordinate, registered before its outcomes."""

    source_registration_hash: str
    stratum_id: str
    sealed_order_index: int
    context_id: str
    draw_id: str
    C_coordinate_proof_hash: str
    expected_context_packet_hash: str
    issued_before_C_outcomes: bool = True
    C_outcomes_seen: bool = False
    postselection_occurred: bool = False
    schema: str = PHYSICAL_ACTION_SIX_ROLE_C_COORDINATE_SCHEMA
    _registration_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "source_registration_hash",
            "stratum_id",
            "context_id",
            "draw_id",
            "C_coordinate_proof_hash",
            "expected_context_packet_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "sealed_order_index",
            _nonnegative_int(
                self.sealed_order_index,
                name="sealed C order index",
            ),
        )
        flags = (self.C_outcomes_seen, self.postselection_occurred)
        if (
            type(self.issued_before_C_outcomes) is not bool
            or not self.issued_before_C_outcomes
            or any(type(value) is not bool for value in flags)
            or any(flags)
        ):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.POSTSELECTION,
                "C coordinate was not frozen before its outcomes",
            )
        if self.schema != PHYSICAL_ACTION_SIX_ROLE_C_COORDINATE_SCHEMA:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.INVALID_TYPE,
                "unsupported C coordinate schema",
            )
        self.assert_integrity()

    @property
    def registration_hash(self) -> str:
        return self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "source_registration_hash",
                "stratum_id",
                "sealed_order_index",
                "context_id",
                "draw_id",
                "C_coordinate_proof_hash",
                "expected_context_packet_hash",
                "issued_before_C_outcomes",
                "C_outcomes_seen",
                "postselection_occurred",
            )
        }

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        if (
            not self.issued_before_C_outcomes
            or self.C_outcomes_seen
            or self.postselection_occurred
        ):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.POSTSELECTION,
                "C coordinate acquired or registered post-outcome state",
            )
        observed = content_hash(self._payload(), prefix=_COORDINATE_PREFIX)
        cached = self._registration_hash
        if cached is not None and cached != observed:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.STALE_COORDINATE,
                "C coordinate registration was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.STALE_COORDINATE,
                "C coordinate registration hash differs",
            )
        object.__setattr__(self, "_registration_hash", observed)
        return observed


def reduce_signed_native_fiber_work_intervals(
    *,
    signed_center_by_action_fiber: object,
    signed_radius_by_action_fiber: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce signed native-fiber projections to per-image ``F_native``.

    Cancellation has already occurred *inside* each center because each entry
    is the signed sum of removed-route evidence in one native fiber.  This
    routine then takes an outward absolute interval per fiber and sums those
    nonnegative fiber burdens.  It never takes an absolute value route by
    route and never allows cancellation across images.
    """

    try:
        centers = np.asarray(signed_center_by_action_fiber, dtype=np.float64)
        radii = np.asarray(signed_radius_by_action_fiber, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise V1PhysicalActionSixRoleCRefusal(
            V1PhysicalActionSixRoleCRefusalCode.INVALID_TYPE,
            f"native-fiber projections must be binary64 matrices: {error}",
        ) from error
    if (
        centers.ndim != 2
        or centers.shape != radii.shape
        or centers.shape[0] <= 0
        or centers.shape[1] <= 0
        or not bool(np.isfinite(centers).all())
        or not bool(np.isfinite(radii).all())
        or bool(np.any(radii < 0.0))
    ):
        _refuse(
            V1PhysicalActionSixRoleCRefusalCode.INTERVAL_INVALID,
            "native-fiber centers/radii must be finite aligned nonempty matrices",
        )
    lower = np.empty(centers.shape[0], dtype=np.float64)
    upper = np.empty(centers.shape[0], dtype=np.float64)
    with localcontext() as decimal_context:
        decimal_context.prec = 2500
        for action in range(centers.shape[0]):
            lower_sum = _decimal(0.0)
            upper_sum = _decimal(0.0)
            for fiber in range(centers.shape[1]):
                center = _decimal(float(centers[action, fiber]))
                radius = _decimal(float(radii[action, fiber]))
                absolute_lower, absolute_upper = _absolute_interval(
                    center - radius,
                    center + radius,
                )
                lower_sum += absolute_lower
                upper_sum += absolute_upper
            lower[action] = _outward_float(
                lower_sum,
                upward=False,
                name="native-fiber F lower bound",
            )
            upper[action] = _outward_float(
                upper_sum,
                upward=True,
                name="native-fiber F upper bound",
            )
    return (
        _immutable_f64(
            lower,
            shape=lower.shape,
            name="native-fiber F lower bounds",
            nonnegative=True,
        ),
        _immutable_f64(
            upper,
            shape=upper.shape,
            name="native-fiber F upper bounds",
            nonnegative=True,
        ),
    )


@dataclass(frozen=True, slots=True)
class V1PhysicalActionSixRoleCRow:
    """One same-capture row advancing every unique action and all six roles."""

    design_hash: str
    coordinate_registration: V1PhysicalActionSixRoleCCoordinateRegistration
    direction_context_execution_receipt: V1PhysicalActionContextExecutionReceipt = (
        field(repr=False, compare=False)
    )
    physical_action_entry_hashes: tuple[str, ...]
    chart_identity_hash: str
    current_parent_mask_hash: str
    route_axis_hash: str
    route_count: int
    proposal_generation: int
    corrected_evidence_stream_identity_hash: str
    literal_jacobian_stream_identity_hash: str
    raw_logit_stream_identity_hash: str
    same_capture_and_vjp_execution_proof_hash: str
    parent_execution_identity_hash: str
    native_fiber_work_projection_receipt_hash: str
    literal_jacobian_projection_receipt_hash: str
    raw_logit_interval_proof_hash: str
    parent_raw_target_logit_center: float
    parent_raw_target_logit_radius: float
    native_fiber_work_lower_by_action: np.ndarray = field(repr=False, compare=False)
    native_fiber_work_upper_by_action: np.ndarray = field(repr=False, compare=False)
    global_literal_jacobian_center_by_action: np.ndarray = field(
        repr=False,
        compare=False,
    )
    global_literal_jacobian_radius_by_action: np.ndarray = field(
        repr=False,
        compare=False,
    )
    dosed_raw_target_logit_center_by_action: np.ndarray = field(
        repr=False,
        compare=False,
    )
    dosed_raw_target_logit_radius_by_action: np.ndarray = field(
        repr=False,
        compare=False,
    )
    _issuer_token: object | None = field(
        default=None,
        repr=False,
        compare=False,
        kw_only=True,
    )
    doses: tuple[float, float] = (0.5, 1.0)
    native_fiber_signed_cancellation_then_absolute: bool = True
    corrected_evidence_and_literal_J_are_distinct_streams: bool = True
    raw_logit_is_a_distinct_stream: bool = True
    literal_J_is_global_signed_complete_action: bool = True
    secants_execute_complete_independent_actions: bool = True
    action_order_unchanged_after_C_outcomes: bool = True
    postselection_occurred: bool = False
    candidate_selection_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = PHYSICAL_ACTION_SIX_ROLE_C_AUTHORITY
    schema: str = PHYSICAL_ACTION_SIX_ROLE_C_ROW_SCHEMA
    observation_lower_by_action_role: np.ndarray = field(
        init=False,
        repr=False,
        compare=False,
    )
    observation_upper_by_action_role: np.ndarray = field(
        init=False,
        repr=False,
        compare=False,
    )
    _row_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )
    physical_measurement_profile: str = field(default=LEGACY_UNIFORM_ABSOLUTE, kw_only=True)

    def __setstate__(self, state: list[object]) -> None:
        restore_profile_state(self, state)

    def __post_init__(self) -> None:
        validate_physical_measurement_profile(self.physical_measurement_profile)
        if self._issuer_token is not _ROW_ISSUER_TOKEN:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.AUTHORITY_VIOLATION,
                "physical-action C rows are issued only by the typed row factory",
            )
        for name in (
            "design_hash",
            "chart_identity_hash",
            "current_parent_mask_hash",
            "route_axis_hash",
            "corrected_evidence_stream_identity_hash",
            "literal_jacobian_stream_identity_hash",
            "raw_logit_stream_identity_hash",
            "same_capture_and_vjp_execution_proof_hash",
            "parent_execution_identity_hash",
            "native_fiber_work_projection_receipt_hash",
            "literal_jacobian_projection_receipt_hash",
            "raw_logit_interval_proof_hash",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if type(self.coordinate_registration) is not (
            V1PhysicalActionSixRoleCCoordinateRegistration
        ):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.INVALID_TYPE,
                "C row requires the exact coordinate registration",
            )
        if type(self.direction_context_execution_receipt) is not (
            V1PhysicalActionContextExecutionReceipt
        ):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.INVALID_TYPE,
                "C row requires the exact physical direction execution receipt",
            )
        actions = tuple(
            _identity(value, name="physical-action entry hash")
            for value in self.physical_action_entry_hashes
        )
        if not actions or len(actions) != len(set(actions)):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.ACTION_AXIS_MISMATCH,
                "C row action axis is empty or duplicated",
            )
        action_count = len(actions)
        f_lower = _immutable_f64(
            self.native_fiber_work_lower_by_action,
            shape=(action_count,),
            name="native-fiber F lower bounds",
            nonnegative=True,
        )
        f_upper = _immutable_f64(
            self.native_fiber_work_upper_by_action,
            shape=(action_count,),
            name="native-fiber F upper bounds",
            nonnegative=True,
        )
        if bool(np.any(f_lower > f_upper)):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.INTERVAL_INVALID,
                "native-fiber F interval is reversed",
            )
        j_center = _immutable_f64(
            self.global_literal_jacobian_center_by_action,
            shape=(action_count,),
            name="literal-J centers",
        )
        j_radius = _immutable_f64(
            self.global_literal_jacobian_radius_by_action,
            shape=(action_count,),
            name="literal-J radii",
            nonnegative=True,
        )
        child_center = _immutable_f64(
            self.dosed_raw_target_logit_center_by_action,
            shape=(action_count, 2),
            name="dosed raw-logit centers",
        )
        child_radius = _immutable_f64(
            self.dosed_raw_target_logit_radius_by_action,
            shape=(action_count, 2),
            name="dosed raw-logit radii",
            nonnegative=True,
        )
        parent_center = _finite(
            self.parent_raw_target_logit_center,
            name="parent raw target logit center",
        )
        parent_radius = _finite_nonnegative(
            self.parent_raw_target_logit_radius,
            name="parent raw target logit radius",
        )
        doses = tuple(
            _finite_nonnegative(value, name="secant dose") for value in self.doses
        )
        if doses != (0.5, 1.0):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.STALE_COORDINATE,
                "C row requires canonical doses (0.5, 1.0)",
            )
        streams = (
            self.corrected_evidence_stream_identity_hash,
            self.literal_jacobian_stream_identity_hash,
            self.raw_logit_stream_identity_hash,
        )
        required_true = (
            self.native_fiber_signed_cancellation_then_absolute,
            self.corrected_evidence_and_literal_J_are_distinct_streams,
            self.raw_logit_is_a_distinct_stream,
            self.literal_J_is_global_signed_complete_action,
            self.secants_execute_complete_independent_actions,
            self.action_order_unchanged_after_C_outcomes,
        )
        authority = (
            self.postselection_occurred,
            self.candidate_selection_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if (
            len(set(streams)) != 3
            or any(type(value) is not bool for value in required_true + authority)
            or not all(required_true)
        ):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.STREAM_IDENTITY_MISMATCH,
                "C row weakened the corrected/J/raw stream separation",
            )
        if any(authority):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.POSTSELECTION,
                "C row was postselected or crossed measurement authority",
            )
        route_count = _positive_int(self.route_count, name="C row route count")
        generation = _nonnegative_int(
            self.proposal_generation,
            name="C row proposal generation",
        )
        if (
            self.authority_scope != PHYSICAL_ACTION_SIX_ROLE_C_AUTHORITY
            or self.schema != PHYSICAL_ACTION_SIX_ROLE_C_ROW_SCHEMA
        ):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.AUTHORITY_VIOLATION,
                "unsupported C row authority or schema",
            )
        observations_lower, observations_upper = six_role_observation_bounds(
            f_lower=f_lower, f_upper=f_upper, j_center=j_center, j_radius=j_radius,
            child_center=child_center, child_radius=child_radius,
            parent_center=parent_center, parent_radius=parent_radius, doses=doses,
            physical_measurement_profile=self.physical_measurement_profile,
        )
        object.__setattr__(self, "physical_action_entry_hashes", actions)
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "parent_raw_target_logit_center", parent_center)
        object.__setattr__(self, "parent_raw_target_logit_radius", parent_radius)
        object.__setattr__(self, "native_fiber_work_lower_by_action", f_lower)
        object.__setattr__(self, "native_fiber_work_upper_by_action", f_upper)
        object.__setattr__(self, "global_literal_jacobian_center_by_action", j_center)
        object.__setattr__(self, "global_literal_jacobian_radius_by_action", j_radius)
        object.__setattr__(
            self, "dosed_raw_target_logit_center_by_action", child_center
        )
        object.__setattr__(
            self, "dosed_raw_target_logit_radius_by_action", child_radius
        )
        object.__setattr__(self, "doses", doses)
        object.__setattr__(
            self,
            "observation_lower_by_action_role",
            _immutable_f64(
                observations_lower,
                shape=observations_lower.shape,
                name="six-role lower observations",
                nonnegative=True,
            ),
        )
        object.__setattr__(
            self,
            "observation_upper_by_action_role",
            _immutable_f64(
                observations_upper,
                shape=observations_upper.shape,
                name="six-role upper observations",
                nonnegative=True,
            ),
        )
        if bool(
            np.any(
                self.observation_lower_by_action_role
                > self.observation_upper_by_action_role
            )
        ):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.INTERVAL_INVALID,
                "derived six-role observation interval is reversed",
            )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "design_hash",
                "physical_action_entry_hashes",
                "chart_identity_hash",
                "current_parent_mask_hash",
                "route_axis_hash",
                "route_count",
                "proposal_generation",
                "corrected_evidence_stream_identity_hash",
                "literal_jacobian_stream_identity_hash",
                "raw_logit_stream_identity_hash",
                "same_capture_and_vjp_execution_proof_hash",
                "parent_execution_identity_hash",
                "native_fiber_work_projection_receipt_hash",
                "literal_jacobian_projection_receipt_hash",
                "raw_logit_interval_proof_hash",
                "parent_raw_target_logit_center",
                "parent_raw_target_logit_radius",
                "native_fiber_work_lower_by_action",
                "native_fiber_work_upper_by_action",
                "global_literal_jacobian_center_by_action",
                "global_literal_jacobian_radius_by_action",
                "dosed_raw_target_logit_center_by_action",
                "dosed_raw_target_logit_radius_by_action",
                "observation_lower_by_action_role",
                "observation_upper_by_action_role",
                "doses",
                "native_fiber_signed_cancellation_then_absolute",
                "corrected_evidence_and_literal_J_are_distinct_streams",
                "raw_logit_is_a_distinct_stream",
                "literal_J_is_global_signed_complete_action",
                "secants_execute_complete_independent_actions",
                "action_order_unchanged_after_C_outcomes",
                "postselection_occurred",
                "candidate_selection_authority",
                "calibration_K_open_authority",
                "paired_oracle_O_open_authority",
                "mask_commit_authority",
                "terminal_authority",
                "pareto_authority",
                "authority_scope",
            )
        } | {
            "coordinate_registration_hash": (
                self.coordinate_registration.registration_hash
            ),
            "direction_context_execution_receipt_hash": (
                self.direction_context_execution_receipt.receipt_hash
            ),
        } | profile_coordinates(self.physical_measurement_profile)

    @property
    def row_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_ROW_PREFIX)
        cached = self._row_hash
        if cached is not None and cached != observed:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.STALE_SOURCE,
                "physical-action C row was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.STALE_SOURCE,
                "physical-action C row hash differs",
            )
        object.__setattr__(self, "_row_hash", observed)
        return observed


def build_v1_signed_work_physical_action_six_role_c_row(
    *,
    design: V1PhysicalActionSixRoleCDesign,
    coordinate_registration: V1PhysicalActionSixRoleCCoordinateRegistration,
    direction_context_execution_receipt: V1PhysicalActionContextExecutionReceipt,
    same_capture_and_vjp_execution_proof_hash: str,
    parent_execution_identity_hash: str,
    native_fiber_work_projection_receipt_hash: str,
    literal_jacobian_projection_receipt_hash: str,
    raw_logit_interval_proof_hash: str,
    parent_raw_target_logit_center: float,
    parent_raw_target_logit_radius: float,
    native_fiber_work_lower_by_action: object,
    native_fiber_work_upper_by_action: object,
    global_literal_jacobian_center_by_action: object,
    global_literal_jacobian_radius_by_action: object,
    dosed_raw_target_logit_center_by_action: object,
    dosed_raw_target_logit_radius_by_action: object,
) -> V1PhysicalActionSixRoleCRow:
    """Issue one typed row from the exact frozen axis and execution receipt."""

    if type(design) is not V1PhysicalActionSixRoleCDesign:
        _refuse(
            V1PhysicalActionSixRoleCRefusalCode.INVALID_TYPE,
            "row factory requires the exact physical-action six-role design",
        )
    design.assert_integrity()
    if type(coordinate_registration) is not (
        V1PhysicalActionSixRoleCCoordinateRegistration
    ):
        _refuse(
            V1PhysicalActionSixRoleCRefusalCode.INVALID_TYPE,
            "row factory requires the exact C coordinate registration",
        )
    coordinate_registration.assert_integrity()
    if type(direction_context_execution_receipt) is not (
        V1PhysicalActionContextExecutionReceipt
    ):
        _refuse(
            V1PhysicalActionSixRoleCRefusalCode.INVALID_TYPE,
            "row factory requires the exact direction execution receipt",
        )
    source = design.source_registration
    if coordinate_registration.source_registration_hash != source.registration_hash:
        _refuse(
            V1PhysicalActionSixRoleCRefusalCode.STALE_SOURCE,
            "C coordinate belongs to another source registration",
        )
    if coordinate_registration.stratum_id not in design.stratum_ids:
        _refuse(
            V1PhysicalActionSixRoleCRefusalCode.STALE_COORDINATE,
            "C coordinate names an unregistered stratum",
        )
    stratum = design.strata[
        design.stratum_ids.index(coordinate_registration.stratum_id)
    ]
    if coordinate_registration.sealed_order_index >= stratum.population_size:
        _refuse(
            V1PhysicalActionSixRoleCRefusalCode.STALE_COORDINATE,
            "C coordinate exceeds its complete finite stratum",
        )
    execution = direction_context_execution_receipt
    receipt = design.direction_provider_receipt
    if (
        execution.provider_receipt_hash != receipt.receipt_hash
        or execution.context_packet_hash
        != coordinate_registration.expected_context_packet_hash
        or execution.direction_lane_hashes
        != tuple(value.lane_hash for value in receipt.lane_identities)
        or execution.unique_physical_action_count
        != len(design.physical_action_entry_hashes)
        or execution.alias_count
        != design.physical_action_axis.upstream_q_safe_alias_count
        or execution.aliases_affect_lane_count
    ):
        _refuse(
            V1PhysicalActionSixRoleCRefusalCode.DIRECTION_RECEIPT_MISMATCH,
            "row execution does not replay the independent physical-action lanes",
        )
    return V1PhysicalActionSixRoleCRow(
        _issuer_token=_ROW_ISSUER_TOKEN,
        physical_measurement_profile=design.physical_measurement_profile,
        design_hash=design.design_hash,
        coordinate_registration=coordinate_registration,
        direction_context_execution_receipt=execution,
        physical_action_entry_hashes=design.physical_action_entry_hashes,
        chart_identity_hash=source.chart_identity_hash,
        current_parent_mask_hash=design.parent_mask_hash,
        route_axis_hash=design.route_axis_hash,
        route_count=design.route_count,
        proposal_generation=design.proposal_generation,
        corrected_evidence_stream_identity_hash=(
            source.corrected_evidence_stream_identity_hash
        ),
        literal_jacobian_stream_identity_hash=(
            source.literal_jacobian_stream_identity_hash
        ),
        raw_logit_stream_identity_hash=source.raw_logit_stream_identity_hash,
        same_capture_and_vjp_execution_proof_hash=_identity(
            same_capture_and_vjp_execution_proof_hash,
            name="same-capture/VJP execution proof hash",
        ),
        parent_execution_identity_hash=_identity(
            parent_execution_identity_hash,
            name="parent execution identity hash",
        ),
        native_fiber_work_projection_receipt_hash=_identity(
            native_fiber_work_projection_receipt_hash,
            name="native-fiber work projection receipt hash",
        ),
        literal_jacobian_projection_receipt_hash=_identity(
            literal_jacobian_projection_receipt_hash,
            name="literal-J projection receipt hash",
        ),
        raw_logit_interval_proof_hash=_identity(
            raw_logit_interval_proof_hash,
            name="raw-logit interval proof hash",
        ),
        parent_raw_target_logit_center=parent_raw_target_logit_center,
        parent_raw_target_logit_radius=parent_raw_target_logit_radius,
        native_fiber_work_lower_by_action=native_fiber_work_lower_by_action,
        native_fiber_work_upper_by_action=native_fiber_work_upper_by_action,
        global_literal_jacobian_center_by_action=(
            global_literal_jacobian_center_by_action
        ),
        global_literal_jacobian_radius_by_action=(
            global_literal_jacobian_radius_by_action
        ),
        dosed_raw_target_logit_center_by_action=(
            dosed_raw_target_logit_center_by_action
        ),
        dosed_raw_target_logit_radius_by_action=(
            dosed_raw_target_logit_radius_by_action
        ),
    )


@dataclass(frozen=True, slots=True)
class V1PhysicalActionSixRoleCellSummary:
    stratum_id: str
    physical_action_entry_hash: str
    role: str
    dose: float | None
    population_size: int
    sample_size: int
    lower: float
    upper: float
    value_upper_bound: float
    current_lower_look_alpha: float
    current_upper_look_alpha: float
    full_census: bool
    method: str
    observation_chain_hash: str
    schema: str = PHYSICAL_ACTION_SIX_ROLE_C_CELL_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "stratum_id",
            "physical_action_entry_hash",
            "observation_chain_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if (
            self.role not in SIGNED_WORK_SIX_ROLES
            or self.dose != SIGNED_WORK_SIX_ROLE_DOSES[self.role]
        ):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.STALE_COORDINATE,
                "cell role/dose is not canonical",
            )
        population = _positive_int(self.population_size, name="cell population")
        count = _nonnegative_int(self.sample_size, name="cell sample count")
        lower = _finite_nonnegative(self.lower, name="cell lower bound")
        upper = _finite_nonnegative(self.upper, name="cell upper bound")
        support = _finite_nonnegative(
            self.value_upper_bound,
            name="cell analytic support",
        )
        if count > population or not lower <= upper <= support:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.SUPPORT_OVERFLOW,
                "cell interval exceeds its population or analytic support",
            )
        lower_alpha = _finite_nonnegative(
            self.current_lower_look_alpha,
            name="cell lower look alpha",
        )
        upper_alpha = _finite_nonnegative(
            self.current_upper_look_alpha,
            name="cell upper look alpha",
        )
        if type(self.full_census) is not bool or self.full_census != (
            count == population
        ):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.STALE_COORDINATE,
                "cell census flag differs from its count",
            )
        expected_method = PHYSICAL_ACTION_SIX_ROLE_C_INTERVAL_METHOD
        if count == 0:
            expected_method = "unobserved_registered_support_v1"
        elif count == population:
            expected_method = PHYSICAL_ACTION_SIX_ROLE_C_CENSUS_METHOD
        if (
            self.method != expected_method
            or self.schema != PHYSICAL_ACTION_SIX_ROLE_C_CELL_SCHEMA
        ):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.STALE_SOURCE,
                "cell method or schema differs from its sampling state",
            )
        object.__setattr__(self, "population_size", population)
        object.__setattr__(self, "sample_size", count)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "value_upper_bound", support)
        object.__setattr__(self, "current_lower_look_alpha", lower_alpha)
        object.__setattr__(self, "current_upper_look_alpha", upper_alpha)

    @property
    def summary_hash(self) -> str:
        return content_hash(self, prefix=_CELL_PREFIX)


@dataclass(frozen=True, slots=True)
class V1PhysicalActionSixRoleActionSummary:
    physical_action_entry_hash: str
    f_native_lower: float
    f_native_upper: float
    q_j_lower: float
    q_j_upper: float
    r_half_lower: float
    r_half_upper: float
    r_one_lower: float
    r_one_upper: float
    d_half_lower: float
    d_half_upper: float
    d_one_lower: float
    d_one_upper: float
    chart_half_burden_lower: float
    chart_half_burden_upper: float
    chart_one_burden_lower: float
    chart_one_burden_upper: float
    measurement_evidence_only: bool = True
    aliases_receive_only_external_fanout: bool = True
    candidate_selection_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = PHYSICAL_ACTION_SIX_ROLE_C_ACTION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "physical_action_entry_hash",
            _identity(
                self.physical_action_entry_hash,
                name="action summary physical entry hash",
            ),
        )
        for lower_name, upper_name in (
            ("f_native_lower", "f_native_upper"),
            ("q_j_lower", "q_j_upper"),
            ("r_half_lower", "r_half_upper"),
            ("r_one_lower", "r_one_upper"),
            ("d_half_lower", "d_half_upper"),
            ("d_one_lower", "d_one_upper"),
            ("chart_half_burden_lower", "chart_half_burden_upper"),
            ("chart_one_burden_lower", "chart_one_burden_upper"),
        ):
            lower = _finite_nonnegative(getattr(self, lower_name), name=lower_name)
            upper = _finite_nonnegative(getattr(self, upper_name), name=upper_name)
            if lower > upper:
                _refuse(
                    V1PhysicalActionSixRoleCRefusalCode.INTERVAL_INVALID,
                    f"action interval {lower_name}/{upper_name} is reversed",
                )
            object.__setattr__(self, lower_name, lower)
            object.__setattr__(self, upper_name, upper)
        if (
            type(self.measurement_evidence_only) is not bool
            or not self.measurement_evidence_only
            or type(self.aliases_receive_only_external_fanout) is not bool
            or not self.aliases_receive_only_external_fanout
        ):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.AUTHORITY_VIOLATION,
                "action summary weakened its measurement-only semantics",
            )
        _closed_authority(
            (
                self.candidate_selection_authority,
                self.calibration_K_open_authority,
                self.paired_oracle_O_open_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="physical-action C action summary",
        )
        if self.schema != PHYSICAL_ACTION_SIX_ROLE_C_ACTION_SCHEMA:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.INVALID_TYPE,
                "unsupported physical-action summary schema",
            )

    @property
    def summary_hash(self) -> str:
        return content_hash(self, prefix=_ACTION_PREFIX)


@dataclass(frozen=True, slots=True)
class V1PhysicalActionSixRoleProgressReceipt:
    design_hash: str
    physical_action_axis_hash: str
    direction_provider_receipt_hash: str
    source_registration_hash: str
    chart_identity_hash: str
    current_parent_mask_hash: str
    route_axis_hash: str
    route_count: int
    proposal_generation: int
    complete_train_population_hash: str
    c_stream_hash: str
    trajectory_alpha_ledger_hash: str
    observed_counts_by_stratum: tuple[int, ...]
    next_expected_order_index_by_stratum: tuple[int | None, ...]
    cell_summaries: tuple[V1PhysicalActionSixRoleCellSummary, ...]
    action_summaries: tuple[V1PhysicalActionSixRoleActionSummary, ...]
    observation_chain_hash: str
    record_count: int
    sampling_status: PhysicalActionSixRoleSamplingStatus
    tail_count: int
    registered_tail_alpha_total: float
    aliases_measured_as_independent_cells: bool = False
    branch_or_nesting_semantics_used: bool = False
    candidate_selection_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = PHYSICAL_ACTION_SIX_ROLE_C_AUTHORITY
    schema: str = PHYSICAL_ACTION_SIX_ROLE_C_PROGRESS_SCHEMA
    _receipt_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "design_hash",
            "physical_action_axis_hash",
            "direction_provider_receipt_hash",
            "source_registration_hash",
            "chart_identity_hash",
            "current_parent_mask_hash",
            "route_axis_hash",
            "complete_train_population_hash",
            "c_stream_hash",
            "trajectory_alpha_ledger_hash",
            "observation_chain_hash",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="progress route count"),
        )
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(
                self.proposal_generation,
                name="progress proposal generation",
            ),
        )
        counts = tuple(
            _nonnegative_int(value, name="observed stratum count")
            for value in self.observed_counts_by_stratum
        )
        next_indices = tuple(self.next_expected_order_index_by_stratum)
        for value in next_indices:
            if value is not None:
                _nonnegative_int(value, name="next expected order index")
        if (
            not counts
            or len(next_indices) != len(counts)
            or any(
                type(value) is not V1PhysicalActionSixRoleCellSummary
                for value in self.cell_summaries
            )
            or any(
                type(value) is not V1PhysicalActionSixRoleActionSummary
                for value in self.action_summaries
            )
        ):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.INVALID_TYPE,
                "progress axes or summary types are invalid",
            )
        record_count = _nonnegative_int(self.record_count, name="C record count")
        if record_count != sum(counts):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.STALE_COORDINATE,
                "progress record count differs from stratum counts",
            )
        if self.sampling_status not in {
            "awaiting_C",
            "partial_C_evidence",
            "full_C_census",
        }:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.STALE_COORDINATE,
                "unknown physical-action C sampling status",
            )
        _closed_authority(
            (
                self.aliases_measured_as_independent_cells,
                self.branch_or_nesting_semantics_used,
                self.candidate_selection_authority,
                self.calibration_K_open_authority,
                self.paired_oracle_O_open_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="physical-action six-role progress",
        )
        if (
            self.authority_scope != PHYSICAL_ACTION_SIX_ROLE_C_AUTHORITY
            or self.schema != PHYSICAL_ACTION_SIX_ROLE_C_PROGRESS_SCHEMA
        ):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.AUTHORITY_VIOLATION,
                "unsupported progress authority or schema",
            )
        object.__setattr__(self, "observed_counts_by_stratum", counts)
        object.__setattr__(self, "next_expected_order_index_by_stratum", next_indices)
        object.__setattr__(self, "record_count", record_count)
        self.assert_integrity()

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "design_hash",
                "physical_action_axis_hash",
                "direction_provider_receipt_hash",
                "source_registration_hash",
                "chart_identity_hash",
                "current_parent_mask_hash",
                "route_axis_hash",
                "route_count",
                "proposal_generation",
                "complete_train_population_hash",
                "c_stream_hash",
                "trajectory_alpha_ledger_hash",
                "observed_counts_by_stratum",
                "next_expected_order_index_by_stratum",
                "cell_summaries",
                "action_summaries",
                "observation_chain_hash",
                "record_count",
                "sampling_status",
                "tail_count",
                "registered_tail_alpha_total",
                "aliases_measured_as_independent_cells",
                "branch_or_nesting_semantics_used",
                "candidate_selection_authority",
                "calibration_K_open_authority",
                "paired_oracle_O_open_authority",
                "mask_commit_authority",
                "terminal_authority",
                "pareto_authority",
                "authority_scope",
            )
        }

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_PROGRESS_PREFIX)
        cached = self._receipt_hash
        if cached is not None and cached != observed:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.STALE_SOURCE,
                "physical-action C progress receipt was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.STALE_SOURCE,
                "physical-action C progress receipt hash differs",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    def assert_replays(self, design: V1PhysicalActionSixRoleCDesign) -> str:
        if type(design) is not V1PhysicalActionSixRoleCDesign:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.INVALID_TYPE,
                "progress replay requires the exact six-role design",
            )
        self.assert_integrity()
        expected = {
            "design_hash": design.design_hash,
            "physical_action_axis_hash": design.physical_action_axis.axis_hash,
            "direction_provider_receipt_hash": (
                design.direction_provider_receipt.receipt_hash
            ),
            "source_registration_hash": design.source_registration.registration_hash,
            "chart_identity_hash": design.source_registration.chart_identity_hash,
            "current_parent_mask_hash": design.parent_mask_hash,
            "route_axis_hash": design.route_axis_hash,
            "route_count": design.route_count,
            "proposal_generation": design.proposal_generation,
            "complete_train_population_hash": design.complete_train_population_hash,
            "c_stream_hash": design.c_stream_hash,
            "trajectory_alpha_ledger_hash": design.trajectory_alpha_ledger_hash,
            "tail_count": design.tail_count,
            "registered_tail_alpha_total": design.registered_tail_alpha_total,
        }
        stale = tuple(
            name for name, value in expected.items() if getattr(self, name) != value
        )
        if stale:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.STALE_SOURCE,
                "progress differs from design at: " + ", ".join(stale),
            )
        expected_cells = (
            len(design.strata)
            * len(design.physical_action_entry_hashes)
            * len(SIGNED_WORK_SIX_ROLES)
        )
        if len(self.cell_summaries) != expected_cells or len(
            self.action_summaries
        ) != len(design.physical_action_entry_hashes):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.ACTION_AXIS_MISMATCH,
                "progress did not cover every unique physical-action cell",
            )
        expected_cell_axis = tuple(
            (
                stratum.stratum_id,
                action,
                role,
                SIGNED_WORK_SIX_ROLE_DOSES[role],
                stratum.population_size,
                self.observed_counts_by_stratum[stratum_index],
                float(
                    design.analytic_support_upper_by_stratum_action_role[
                        stratum_index,
                        action_index,
                        role_index,
                    ]
                ),
                self.observation_chain_hash,
            )
            for stratum_index, stratum in enumerate(design.strata)
            for action_index, action in enumerate(design.physical_action_entry_hashes)
            for role_index, role in enumerate(SIGNED_WORK_SIX_ROLES)
        )
        observed_cell_axis = tuple(
            (
                value.stratum_id,
                value.physical_action_entry_hash,
                value.role,
                value.dose,
                value.population_size,
                value.sample_size,
                value.value_upper_bound,
                value.observation_chain_hash,
            )
            for value in self.cell_summaries
        )
        if (
            observed_cell_axis != expected_cell_axis
            or tuple(
                value.physical_action_entry_hash for value in self.action_summaries
            )
            != design.physical_action_entry_hashes
        ):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.ACTION_AXIS_MISMATCH,
                "progress cell/action identity axes were reordered or replaced",
            )
        expected_next = tuple(
            None if count == design.strata[index].population_size else count
            for index, count in enumerate(self.observed_counts_by_stratum)
        )
        total = sum(self.observed_counts_by_stratum)
        expected_status: PhysicalActionSixRoleSamplingStatus
        if total == 0:
            expected_status = "awaiting_C"
        elif total == design.population_size:
            expected_status = "full_C_census"
        else:
            expected_status = "partial_C_evidence"
        if (
            self.next_expected_order_index_by_stratum != expected_next
            or self.sampling_status != expected_status
        ):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.STALE_COORDINATE,
                "progress next-coordinate or sampling status is stale",
            )
        return self.receipt_hash


@dataclass(frozen=True, slots=True)
class V1PhysicalActionSixRoleAccumulatorSnapshot:
    design_hash: str
    counts_by_stratum: tuple[int, ...]
    lower_scaled_sums: tuple[int, ...]
    upper_scaled_sums: tuple[int, ...]
    observation_chain_hash: str
    record_count: int
    schema: str = PHYSICAL_ACTION_SIX_ROLE_C_SNAPSHOT_SCHEMA
    _snapshot_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "design_hash",
            _identity(self.design_hash, name="snapshot design hash"),
        )
        counts = tuple(
            _nonnegative_int(value, name="snapshot stratum count")
            for value in self.counts_by_stratum
        )
        lower = tuple(
            _nonnegative_int(value, name="snapshot lower scaled sum")
            for value in self.lower_scaled_sums
        )
        upper = tuple(
            _nonnegative_int(value, name="snapshot upper scaled sum")
            for value in self.upper_scaled_sums
        )
        if any(left > right for left, right in zip(lower, upper, strict=True)):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.SNAPSHOT_MISMATCH,
                "snapshot lower sums exceed upper sums",
            )
        chain = _identity(
            self.observation_chain_hash,
            name="snapshot observation chain",
        )
        record_count = _nonnegative_int(
            self.record_count,
            name="snapshot record count",
        )
        if record_count != sum(counts) or self.schema != (
            PHYSICAL_ACTION_SIX_ROLE_C_SNAPSHOT_SCHEMA
        ):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.SNAPSHOT_MISMATCH,
                "snapshot counts or schema are inconsistent",
            )
        object.__setattr__(self, "counts_by_stratum", counts)
        object.__setattr__(self, "lower_scaled_sums", lower)
        object.__setattr__(self, "upper_scaled_sums", upper)
        object.__setattr__(self, "observation_chain_hash", chain)
        object.__setattr__(self, "record_count", record_count)
        self.assert_integrity()

    @property
    def snapshot_hash(self) -> str:
        return self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "design_hash": self.design_hash,
            "counts_by_stratum": self.counts_by_stratum,
            "lower_scaled_sums": self.lower_scaled_sums,
            "upper_scaled_sums": self.upper_scaled_sums,
            "observation_chain_hash": self.observation_chain_hash,
            "record_count": self.record_count,
        }

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_SNAPSHOT_PREFIX)
        cached = self._snapshot_hash
        if cached is not None and cached != observed:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.SNAPSHOT_MISMATCH,
                "physical-action C accumulator snapshot was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.SNAPSHOT_MISMATCH,
                "physical-action C accumulator snapshot hash differs",
            )
        object.__setattr__(self, "_snapshot_hash", observed)
        return observed


class V1PhysicalActionSixRoleCAccumulator:
    """Atomic fixed-point accumulator for one branchless physical-action axis."""

    def __init__(self, design: V1PhysicalActionSixRoleCDesign) -> None:
        if type(design) is not V1PhysicalActionSixRoleCDesign:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.INVALID_TYPE,
                "accumulator requires the exact physical-action six-role design",
            )
        design.assert_integrity()
        shape = (
            len(design.strata),
            len(design.physical_action_entry_hashes),
            len(SIGNED_WORK_SIX_ROLES),
        )
        self.design = design
        self._counts = np.zeros(len(design.strata), dtype=np.int64)
        self._lower_sums = np.zeros(shape, dtype=object)
        self._upper_sums = np.zeros(shape, dtype=object)
        self._observation_chain = content_hash(
            {
                "design_hash": design.design_hash,
                "physical_action_axis_hash": design.physical_action_axis.axis_hash,
                "initial": True,
            },
            prefix=_CHAIN_PREFIX,
        )
        self._record_count = 0

    @classmethod
    def from_snapshot(
        cls,
        *,
        design: V1PhysicalActionSixRoleCDesign,
        snapshot: V1PhysicalActionSixRoleAccumulatorSnapshot,
    ) -> "V1PhysicalActionSixRoleCAccumulator":
        if type(snapshot) is not V1PhysicalActionSixRoleAccumulatorSnapshot:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.INVALID_TYPE,
                "snapshot restore requires the exact snapshot type",
            )
        snapshot.assert_integrity()
        instance = cls(design)
        if snapshot.design_hash != design.design_hash:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.SNAPSHOT_MISMATCH,
                "snapshot belongs to another six-role C design",
            )
        shape = instance._lower_sums.shape
        expected_cells = int(np.prod(shape))
        if (
            len(snapshot.counts_by_stratum) != len(design.strata)
            or len(snapshot.lower_scaled_sums) != expected_cells
            or len(snapshot.upper_scaled_sums) != expected_cells
        ):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.SNAPSHOT_MISMATCH,
                "snapshot sufficient-statistic axes differ from the design",
            )
        counts = np.array(snapshot.counts_by_stratum, dtype=np.int64)
        if any(
            int(count) > design.strata[index].population_size
            for index, count in enumerate(counts)
        ):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.SNAPSHOT_MISMATCH,
                "snapshot exceeds a finite C stratum",
            )
        lower = np.array(snapshot.lower_scaled_sums, dtype=object).reshape(shape)
        upper = np.array(snapshot.upper_scaled_sums, dtype=object).reshape(shape)
        for stratum, count in enumerate(counts):
            for action in range(shape[1]):
                for role in range(shape[2]):
                    support_scaled = _binary64_scaled_integer(
                        float(
                            design.analytic_support_upper_by_stratum_action_role[
                                stratum,
                                action,
                                role,
                            ]
                        )
                    )
                    if (
                        int(lower[stratum, action, role])
                        > int(upper[stratum, action, role])
                        or int(upper[stratum, action, role])
                        > int(count) * support_scaled
                    ):
                        _refuse(
                            V1PhysicalActionSixRoleCRefusalCode.SNAPSHOT_MISMATCH,
                            "snapshot sums exceed their exact support envelope",
                        )
        instance._counts = counts
        instance._lower_sums = lower
        instance._upper_sums = upper
        instance._observation_chain = snapshot.observation_chain_hash
        instance._record_count = snapshot.record_count
        # Receipt construction is part of restore validation.
        instance.progress_receipt()
        return instance

    @property
    def record_count(self) -> int:
        return self._record_count

    def _validate_row(
        self,
        row: V1PhysicalActionSixRoleCRow,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        if type(row) is not V1PhysicalActionSixRoleCRow:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.INVALID_TYPE,
                "atomic C transaction requires the exact row type",
            )
        row.assert_integrity()
        design = self.design
        source = design.source_registration
        coordinate = row.coordinate_registration
        if row.design_hash != design.design_hash:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.STALE_SOURCE,
                "C row belongs to another design",
            )
        expected = {
            "physical_measurement_profile": design.physical_measurement_profile,
            "chart_identity_hash": source.chart_identity_hash,
            "current_parent_mask_hash": design.parent_mask_hash,
            "route_axis_hash": design.route_axis_hash,
            "route_count": design.route_count,
            "proposal_generation": design.proposal_generation,
            "corrected_evidence_stream_identity_hash": (
                source.corrected_evidence_stream_identity_hash
            ),
            "literal_jacobian_stream_identity_hash": (
                source.literal_jacobian_stream_identity_hash
            ),
            "raw_logit_stream_identity_hash": source.raw_logit_stream_identity_hash,
        }
        if any(getattr(row, name) != value for name, value in expected.items()):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.STALE_SOURCE,
                "C row changed its chart, parent, route, generation, or streams",
            )
        if row.physical_action_entry_hashes != design.physical_action_entry_hashes:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.ACTION_AXIS_MISMATCH,
                "C row changed or reordered the physical-action axis",
            )
        if coordinate.source_registration_hash != source.registration_hash:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.STALE_SOURCE,
                "C coordinate belongs to another source registration",
            )
        try:
            stratum = design.stratum_ids.index(coordinate.stratum_id)
        except ValueError as error:
            raise V1PhysicalActionSixRoleCRefusal(
                V1PhysicalActionSixRoleCRefusalCode.STALE_COORDINATE,
                "C row names an unregistered stratum",
            ) from error
        count = int(self._counts[stratum])
        population = design.strata[stratum].population_size
        if count == population:
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.DUPLICATE_ROW,
                "C stratum is already a complete census",
            )
        if coordinate.sealed_order_index != count:
            code = (
                V1PhysicalActionSixRoleCRefusalCode.DUPLICATE_ROW
                if coordinate.sealed_order_index < count
                else V1PhysicalActionSixRoleCRefusalCode.STALE_COORDINATE
            )
            _refuse(
                code,
                "C row is not the next sealed without-replacement coordinate",
            )
        execution = row.direction_context_execution_receipt
        if (
            execution.provider_receipt_hash
            != design.direction_provider_receipt.receipt_hash
            or execution.context_packet_hash != coordinate.expected_context_packet_hash
        ):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.DIRECTION_RECEIPT_MISMATCH,
                "C row execution receipt changed after row issuance",
            )
        lower = row.observation_lower_by_action_role
        upper = row.observation_upper_by_action_role
        support = design.analytic_support_upper_by_stratum_action_role[stratum]
        if bool(np.any(lower > upper)) or bool(np.any(upper > support)):
            _refuse(
                V1PhysicalActionSixRoleCRefusalCode.SUPPORT_OVERFLOW,
                "C row exceeds a pre-outcome analytic support",
            )
        return stratum, lower, upper

    def record_atomic(
        self,
        row: V1PhysicalActionSixRoleCRow,
    ) -> V1PhysicalActionSixRoleProgressReceipt:
        """Advance all unique actions and six roles, or mutate no live state."""

        self.design.assert_integrity()
        stratum, lower, upper = self._validate_row(row)
        counts = self._counts.copy()
        lower_sums = self._lower_sums.copy()
        upper_sums = self._upper_sums.copy()
        for action in range(lower.shape[0]):
            for role in range(lower.shape[1]):
                lower_sums[stratum, action, role] += _binary64_scaled_integer(
                    float(lower[action, role])
                )
                upper_sums[stratum, action, role] += _binary64_scaled_integer(
                    float(upper[action, role])
                )
        counts[stratum] += 1
        new_count = self._record_count + 1
        chain = content_hash(
            {
                "previous_chain_hash": self._observation_chain,
                "record_count": new_count,
                "row_hash": row.row_hash,
                "coordinate_registration_hash": (
                    row.coordinate_registration.registration_hash
                ),
                "direction_context_execution_receipt_hash": (
                    row.direction_context_execution_receipt.receipt_hash
                ),
                "physical_action_entry_hashes": (
                    self.design.physical_action_entry_hashes
                ),
                "six_roles_advanced_atomically": SIGNED_WORK_SIX_ROLES,
            },
            prefix=_CHAIN_PREFIX,
        )
        # Build and replay the immutable receipt before committing private state.
        receipt = self._build_progress(
            counts=counts,
            lower_sums=lower_sums,
            upper_sums=upper_sums,
            record_count=new_count,
            observation_chain=chain,
        )
        self._counts = counts
        self._lower_sums = lower_sums
        self._upper_sums = upper_sums
        self._record_count = new_count
        self._observation_chain = chain
        return receipt

    def _cell(
        self,
        *,
        stratum: int,
        action: int,
        role: int,
        counts: np.ndarray,
        lower_sums: np.ndarray,
        upper_sums: np.ndarray,
        observation_chain: str,
    ) -> V1PhysicalActionSixRoleCellSummary:
        count = int(counts[stratum])
        population = self.design.strata[stratum].population_size
        bound = float(
            self.design.analytic_support_upper_by_stratum_action_role[
                stratum,
                action,
                role,
            ]
        )
        lower_sum = int(lower_sums[stratum, action, role])
        upper_sum = int(upper_sums[stratum, action, role])
        if count == 0:
            lower, upper, look_alpha = 0.0, bound, 0.0
            method = "unobserved_registered_support_v1"
        elif count == population:
            lower = _fraction_to_float_outward(
                _scaled_mean(lower_sum, count),
                direction="lower",
            )
            upper = _fraction_to_float_outward(
                _scaled_mean(upper_sum, count),
                direction="upper",
            )
            look_alpha = 0.0
            method = PHYSICAL_ACTION_SIX_ROLE_C_CENSUS_METHOD
        else:
            lower_mean = _fraction_to_float_outward(
                _scaled_mean(lower_sum, count),
                direction="lower",
            )
            upper_mean = _fraction_to_float_outward(
                _scaled_mean(upper_sum, count),
                direction="upper",
            )
            look_alpha = self.design.look_alpha(
                stratum_index=stratum,
                sample_size=count,
            )
            radius = _serfling_radius_upper(
                upper_bound=bound,
                count=count,
                population=population,
                look_alpha=look_alpha,
            )
            lower = max(0.0, math.nextafter(lower_mean - radius, -math.inf))
            upper = min(bound, math.nextafter(upper_mean + radius, math.inf))
            method = PHYSICAL_ACTION_SIX_ROLE_C_INTERVAL_METHOD
        role_name = SIGNED_WORK_SIX_ROLES[role]
        return V1PhysicalActionSixRoleCellSummary(
            stratum_id=self.design.strata[stratum].stratum_id,
            physical_action_entry_hash=(
                self.design.physical_action_entry_hashes[action]
            ),
            role=role_name,
            dose=SIGNED_WORK_SIX_ROLE_DOSES[role_name],
            population_size=population,
            sample_size=count,
            lower=lower,
            upper=upper,
            value_upper_bound=bound,
            current_lower_look_alpha=look_alpha,
            current_upper_look_alpha=look_alpha,
            full_census=count == population,
            method=method,
            observation_chain_hash=observation_chain,
        )

    def _build_progress(
        self,
        *,
        counts: np.ndarray,
        lower_sums: np.ndarray,
        upper_sums: np.ndarray,
        record_count: int,
        observation_chain: str,
    ) -> V1PhysicalActionSixRoleProgressReceipt:
        design = self.design
        action_count = len(design.physical_action_entry_hashes)
        cells = tuple(
            self._cell(
                stratum=stratum,
                action=action,
                role=role,
                counts=counts,
                lower_sums=lower_sums,
                upper_sums=upper_sums,
                observation_chain=observation_chain,
            )
            for stratum in range(len(design.strata))
            for action in range(action_count)
            for role in range(len(SIGNED_WORK_SIX_ROLES))
        )
        action_summaries: list[V1PhysicalActionSixRoleActionSummary] = []
        for action, action_hash in enumerate(design.physical_action_entry_hashes):
            lowers: dict[str, float] = {}
            uppers: dict[str, float] = {}
            for role_index, role in enumerate(SIGNED_WORK_SIX_ROLES):
                role_cells = tuple(
                    cells[
                        (stratum * action_count + action) * len(SIGNED_WORK_SIX_ROLES)
                        + role_index
                    ]
                    for stratum in range(len(design.strata))
                )
                lowers[role] = design.aggregate_role_bound(
                    role=role,
                    values=tuple(value.lower for value in role_cells),
                    direction="lower",
                )
                uppers[role] = design.aggregate_role_bound(
                    role=role,
                    values=tuple(value.upper for value in role_cells),
                    direction="upper",
                )
            action_summaries.append(
                V1PhysicalActionSixRoleActionSummary(
                    physical_action_entry_hash=action_hash,
                    f_native_lower=lowers[F_NATIVE_ROLE],
                    f_native_upper=uppers[F_NATIVE_ROLE],
                    q_j_lower=lowers[Q_J_ROLE],
                    q_j_upper=uppers[Q_J_ROLE],
                    r_half_lower=lowers[R_HALF_ROLE],
                    r_half_upper=uppers[R_HALF_ROLE],
                    r_one_lower=lowers[R_ONE_ROLE],
                    r_one_upper=uppers[R_ONE_ROLE],
                    d_half_lower=lowers[D_HALF_ROLE],
                    d_half_upper=uppers[D_HALF_ROLE],
                    d_one_lower=lowers[D_ONE_ROLE],
                    d_one_upper=uppers[D_ONE_ROLE],
                    chart_half_burden_lower=_response_burden_lower(
                        eta=0.5,
                        q_lower=lowers[Q_J_ROLE],
                        r_lower=lowers[R_HALF_ROLE],
                    ),
                    chart_half_burden_upper=_response_burden(
                        eta=0.5,
                        q_upper=uppers[Q_J_ROLE],
                        r_upper=uppers[R_HALF_ROLE],
                    ),
                    chart_one_burden_lower=_response_burden_lower(
                        eta=1.0,
                        q_lower=lowers[Q_J_ROLE],
                        r_lower=lowers[R_ONE_ROLE],
                    ),
                    chart_one_burden_upper=_response_burden(
                        eta=1.0,
                        q_upper=uppers[Q_J_ROLE],
                        r_upper=uppers[R_ONE_ROLE],
                    ),
                )
            )
        total = sum(int(value) for value in counts)
        if total == 0:
            status: PhysicalActionSixRoleSamplingStatus = "awaiting_C"
        elif total == design.population_size:
            status = "full_C_census"
        else:
            status = "partial_C_evidence"
        receipt = V1PhysicalActionSixRoleProgressReceipt(
            design_hash=design.design_hash,
            physical_action_axis_hash=design.physical_action_axis.axis_hash,
            direction_provider_receipt_hash=(
                design.direction_provider_receipt.receipt_hash
            ),
            source_registration_hash=design.source_registration.registration_hash,
            chart_identity_hash=design.source_registration.chart_identity_hash,
            current_parent_mask_hash=design.parent_mask_hash,
            route_axis_hash=design.route_axis_hash,
            route_count=design.route_count,
            proposal_generation=design.proposal_generation,
            complete_train_population_hash=design.complete_train_population_hash,
            c_stream_hash=design.c_stream_hash,
            trajectory_alpha_ledger_hash=design.trajectory_alpha_ledger_hash,
            observed_counts_by_stratum=tuple(int(value) for value in counts),
            next_expected_order_index_by_stratum=tuple(
                (
                    None
                    if int(value) == design.strata[index].population_size
                    else int(value)
                )
                for index, value in enumerate(counts)
            ),
            cell_summaries=cells,
            action_summaries=tuple(action_summaries),
            observation_chain_hash=observation_chain,
            record_count=record_count,
            sampling_status=status,
            tail_count=design.tail_count,
            registered_tail_alpha_total=design.registered_tail_alpha_total,
        )
        receipt.assert_replays(design)
        return receipt

    def progress_receipt(self) -> V1PhysicalActionSixRoleProgressReceipt:
        self.design.assert_integrity()
        return self._build_progress(
            counts=self._counts,
            lower_sums=self._lower_sums,
            upper_sums=self._upper_sums,
            record_count=self._record_count,
            observation_chain=self._observation_chain,
        )

    def export_snapshot(self) -> V1PhysicalActionSixRoleAccumulatorSnapshot:
        return V1PhysicalActionSixRoleAccumulatorSnapshot(
            design_hash=self.design.design_hash,
            counts_by_stratum=tuple(int(value) for value in self._counts),
            lower_scaled_sums=tuple(
                int(value) for value in self._lower_sums.reshape(-1)
            ),
            upper_scaled_sums=tuple(
                int(value) for value in self._upper_sums.reshape(-1)
            ),
            observation_chain_hash=self._observation_chain,
            record_count=self._record_count,
        )


__all__ = [
    "D_HALF_ROLE",
    "D_ONE_ROLE",
    "F_NATIVE_ROLE",
    "PHYSICAL_ACTION_SIX_ROLE_C_ALPHA_POLICY",
    "PHYSICAL_ACTION_SIX_ROLE_C_AUTHORITY",
    "Q_J_ROLE",
    "R_HALF_ROLE",
    "R_ONE_ROLE",
    "SIGNED_WORK_SIX_ROLES",
    "V1PhysicalActionSixRoleAccumulatorSnapshot",
    "V1PhysicalActionSixRoleActionSummary",
    "V1PhysicalActionSixRoleCAccumulator",
    "V1PhysicalActionSixRoleCCoordinateRegistration",
    "V1PhysicalActionSixRoleCDesign",
    "V1PhysicalActionSixRoleCRefusal",
    "V1PhysicalActionSixRoleCRefusalCode",
    "V1PhysicalActionSixRoleCRow",
    "V1PhysicalActionSixRoleCSourceRegistration",
    "V1PhysicalActionSixRoleCStratum",
    "V1PhysicalActionSixRoleCellSummary",
    "V1PhysicalActionSixRoleProgressReceipt",
    "build_v1_signed_work_physical_action_six_role_c_row",
    "reduce_signed_native_fiber_work_intervals",
]
