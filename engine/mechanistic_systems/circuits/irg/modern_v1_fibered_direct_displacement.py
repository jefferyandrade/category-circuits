"""Exact observation-only finite raw-logit displacement for fibered v1 C rows.

The canonical fibered directional record already contains outward intervals for
the signed complete-action raw-logit secants at doses one half and one.  This
module applies only the interval image of ``abs`` to those signed intervals.
It does not replay a model, reconstruct corrected evidence, allocate alpha, or
open any proposal, calibration, oracle, commit, or terminal authority.

The accompanying fibered observation-bounds object is required because it is
the source-bound, pre-outcome identity for the raw-logit support.  If every
parent and dosed logit lies in ``[-B, B]``, each direct displacement lies in
``[0, 2 B]``.  The factor-of-two support is formed exactly from the registered
binary64 value and rounded outward once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, localcontext
from enum import Enum
import math
from numbers import Integral, Real
from typing import Final, NoReturn

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.measurement.modern_v1_directional_secant_provider import (
    V1FiberedDirectionalSecantObservationBounds,
)

from .modern_v1_directional_remainder import DIRECTIONAL_SECANT_DOSES
from .modern_v1_fibered_directional_record import V1FiberedDirectionalRecord


FIBERED_DIRECT_DISPLACEMENT_OBSERVATION_SCHEMA: Final = (
    "partition_free_v1_fibered_direct_raw_logit_displacement_observation_v1"
)
FIBERED_DIRECT_DISPLACEMENT_EQUATION: Final = (
    "D_eta_is_abs_of_complete_action_signed_raw_logit_delta_interval_"
    "for_eta_one_half_and_one_v1"
)
FIBERED_DIRECT_DISPLACEMENT_BOUND_EQUATION: Final = (
    "abs_child_minus_parent_at_most_two_times_source_proved_"
    "raw_logit_absolute_upper_v1"
)
FIBERED_DIRECT_DISPLACEMENT_AUTHORITY_SCOPE: Final = (
    "fibered_direct_raw_logit_displacement_observation_only_no_alpha_"
    "selection_K_O_commit_or_terminal_authority_v1"
)

_OBSERVATION_PREFIX: Final = "partition-free-v1-fibered-direct-displacement-"
_LE_F64: Final = np.dtype("<f8")
_ISSUANCE_TOKEN: Final = object()


class V1FiberedDirectDisplacementRefusalCode(str, Enum):
    """Fail-closed reasons for the direct-displacement evidence joint."""

    INVALID_TYPE = "invalid_type"
    LEGACY_RECORD = "legacy_record"
    LEGACY_BOUND = "legacy_bound"
    HASH_MISMATCH = "hash_mismatch"
    STALE_COORDINATE = "stale_coordinate"
    POST_SELECTION = "post_selection"
    INTERVAL_INVALID = "interval_invalid"
    ANALYTIC_BOUND_VIOLATION = "analytic_bound_violation"
    AUTHORITY_VIOLATION = "authority_violation"


class V1FiberedDirectDisplacementRefusal(ValueError):
    """Typed refusal; an invalid row never becomes displacement evidence."""

    def __init__(
        self,
        code: V1FiberedDirectDisplacementRefusalCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1FiberedDirectDisplacementRefusalCode,
    message: str,
) -> NoReturn:
    raise V1FiberedDirectDisplacementRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1FiberedDirectDisplacementRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _exact_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1FiberedDirectDisplacementRefusalCode.INVALID_TYPE,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1FiberedDirectDisplacementRefusalCode.STALE_COORDINATE,
            f"{name} must be nonnegative",
        )
    return result


def _exact_positive_int(value: object, *, name: str) -> int:
    result = _exact_nonnegative_int(value, name=name)
    if result == 0:
        _refuse(
            V1FiberedDirectDisplacementRefusalCode.STALE_COORDINATE,
            f"{name} must be positive",
        )
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1FiberedDirectDisplacementRefusalCode.INVALID_TYPE,
            f"{name} must be one real scalar",
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        _refuse(
            V1FiberedDirectDisplacementRefusalCode.INTERVAL_INVALID,
            f"{name} must be finite and nonnegative",
        )
    return 0.0 if result == 0.0 else result


def _identity_axis(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) not in {tuple, list}:
        _refuse(
            V1FiberedDirectDisplacementRefusalCode.INVALID_TYPE,
            f"{name} must be one ordered sequence",
        )
    result = tuple(_identity(item, name=f"{name} item") for item in value)
    if not result or len(result) != len(set(result)):
        _refuse(
            V1FiberedDirectDisplacementRefusalCode.STALE_COORDINATE,
            f"{name} must be nonempty and unique",
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
            V1FiberedDirectDisplacementRefusalCode.INVALID_TYPE,
            f"{name} must be convertible to binary64: {error}",
        )
    if array.shape != shape or not bool(np.isfinite(array).all()):
        _refuse(
            V1FiberedDirectDisplacementRefusalCode.INTERVAL_INVALID,
            f"{name} must be finite with shape {shape}",
        )
    if nonnegative and bool(np.any(array < 0.0)):
        _refuse(
            V1FiberedDirectDisplacementRefusalCode.INTERVAL_INVALID,
            f"{name} must be nonnegative",
        )
    array[array == 0.0] = 0.0
    result = np.frombuffer(array.tobytes(order="C"), dtype=_LE_F64).reshape(shape)
    result.setflags(write=False)
    return result


def _decimal(value: float) -> Decimal:
    return Decimal.from_float(value)


def _outward_float(value: Decimal, *, upward: bool, name: str) -> float:
    try:
        result = float(value)
    except (OverflowError, ValueError, InvalidOperation) as error:
        _refuse(
            V1FiberedDirectDisplacementRefusalCode.INTERVAL_INVALID,
            f"{name} cannot be represented in binary64: {error}",
        )
    if not math.isfinite(result):
        _refuse(
            V1FiberedDirectDisplacementRefusalCode.INTERVAL_INVALID,
            f"{name} exceeds finite binary64",
        )
    represented = Decimal.from_float(result)
    if upward and represented < value:
        result = math.nextafter(result, math.inf)
    elif not upward and represented > value:
        result = math.nextafter(result, -math.inf)
    if not math.isfinite(result):
        _refuse(
            V1FiberedDirectDisplacementRefusalCode.INTERVAL_INVALID,
            f"outward {name} exceeds finite binary64",
        )
    return 0.0 if result == 0.0 else result


def _absolute_interval(
    lower: Decimal,
    upper: Decimal,
) -> tuple[Decimal, Decimal]:
    if lower > upper:
        _refuse(
            V1FiberedDirectDisplacementRefusalCode.INTERVAL_INVALID,
            "signed raw-logit displacement interval is reversed",
        )
    if lower <= 0 <= upper:
        minimum = Decimal(0)
    else:
        minimum = min(abs(lower), abs(upper))
    return minimum, max(abs(lower), abs(upper))


def _derive_absolute_displacement(
    signed_lower: np.ndarray,
    signed_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.empty(signed_lower.shape, dtype=np.float64)
    upper = np.empty(signed_upper.shape, dtype=np.float64)
    with localcontext() as decimal_context:
        # Every finite binary64 value, including subnormals, is represented
        # exactly before the one directed conversion at the observation edge.
        decimal_context.prec = 2500
        for index in np.ndindex(signed_lower.shape):
            absolute_lower, absolute_upper = _absolute_interval(
                _decimal(float(signed_lower[index])),
                _decimal(float(signed_upper[index])),
            )
            lower[index] = _outward_float(
                absolute_lower,
                upward=False,
                name="direct-displacement lower bound",
            )
            upper[index] = _outward_float(
                absolute_upper,
                upward=True,
                name="direct-displacement upper bound",
            )
    return lower, upper


def _outward_twice_raw_bound(raw_upper: float) -> float:
    with localcontext() as decimal_context:
        decimal_context.prec = 2500
        exact = Decimal(2) * _decimal(raw_upper)
        return _outward_float(
            exact,
            upward=True,
            name="direct-displacement analytic absolute upper bound",
        )


def _validate_raw_logit_support(
    *,
    record: V1FiberedDirectionalRecord,
    raw_upper: float,
) -> None:
    """Recheck that the source bound encloses every interval used by D."""

    with localcontext() as decimal_context:
        decimal_context.prec = 2500
        bound = _decimal(raw_upper)
        negative_bound = -bound
        parent_center = _decimal(record.parent_raw_target_logit_center)
        parent_radius = _decimal(record.parent_raw_target_logit_radius)
        if (
            parent_center - parent_radius < negative_bound
            or parent_center + parent_radius > bound
        ):
            _refuse(
                V1FiberedDirectDisplacementRefusalCode.ANALYTIC_BOUND_VIOLATION,
                "parent raw-logit interval exceeds its pre-outcome source bound",
            )
        centers = record.dosed_raw_target_logit_center_by_endpoint
        radii = record.dosed_raw_target_logit_radius_by_endpoint
        for index in np.ndindex(centers.shape):
            center = _decimal(float(centers[index]))
            radius = _decimal(float(radii[index]))
            if center - radius < negative_bound or center + radius > bound:
                _refuse(
                    V1FiberedDirectDisplacementRefusalCode.ANALYTIC_BOUND_VIOLATION,
                    "dosed raw-logit interval exceeds its pre-outcome source bound",
                )


@dataclass(frozen=True, slots=True)
class V1FiberedDirectDisplacementObservation:
    """One exact C-row interval observation for finite raw-logit displacement."""

    directional_record_hash: str
    source_observation_bounds_hash: str
    source_bound_source_hash: str
    projection_closure_hash: str
    design_hash: str
    ladder_hash: str
    chart_identity_hash: str
    parent_mask_hash: str
    parent_execution_identity_hash: str
    proposal_generation: int
    route_count: int
    endpoint_action_hashes: tuple[str, ...]
    stratum_id: str
    sealed_order_index: int
    context_id: str
    draw_id: str
    certification_stream_commitment_hash: str
    joint_fqr_family_registration_hash: str
    corrected_evidence_stream_identity_hash: str
    literal_jacobian_stream_identity_hash: str
    raw_logit_stream_identity_hash: str
    same_capture_and_vjp_execution_proof_hash: str
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    packed_direction_execution_proof_hash: str
    raw_logit_interval_proof_hash: str
    raw_logit_absolute_upper: float
    signed_raw_logit_delta_lower_by_endpoint_and_dose: np.ndarray = field(
        repr=False,
        compare=False,
    )
    signed_raw_logit_delta_upper_by_endpoint_and_dose: np.ndarray = field(
        repr=False,
        compare=False,
    )
    doses: tuple[float, float] = DIRECTIONAL_SECANT_DOSES
    direct_displacement_equation: str = FIBERED_DIRECT_DISPLACEMENT_EQUATION
    analytic_bound_equation: str = FIBERED_DIRECT_DISPLACEMENT_BOUND_EQUATION
    source_bounds_registered_before_outcomes: bool = True
    raw_logit_delta_intervals_are_outward: bool = True
    endpoint_order_unchanged_after_outcomes: bool = True
    postselection_occurred: bool = False
    alpha_allocation_performed: bool = False
    alpha_allocation_authority: bool = False
    candidate_selection_authority: bool = False
    calibration_authority: bool = False
    calibration_K_open_authority: bool = False
    bce_oracle_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = FIBERED_DIRECT_DISPLACEMENT_AUTHORITY_SCOPE
    schema: str = FIBERED_DIRECT_DISPLACEMENT_OBSERVATION_SCHEMA
    analytic_direct_displacement_absolute_upper: float = field(
        init=False,
    )
    direct_displacement_lower_by_endpoint_and_dose: np.ndarray = field(
        init=False,
        repr=False,
        compare=False,
    )
    direct_displacement_upper_by_endpoint_and_dose: np.ndarray = field(
        init=False,
        repr=False,
        compare=False,
    )
    _issuance_token: object = field(default=None, repr=False, compare=False)
    _observation_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self._issuance_token is not _ISSUANCE_TOKEN:
            _refuse(
                V1FiberedDirectDisplacementRefusalCode.INVALID_TYPE,
                "direct-displacement observations must be issued by their closer",
            )
        for name in (
            "directional_record_hash",
            "source_observation_bounds_hash",
            "source_bound_source_hash",
            "projection_closure_hash",
            "design_hash",
            "ladder_hash",
            "chart_identity_hash",
            "parent_mask_hash",
            "parent_execution_identity_hash",
            "stratum_id",
            "context_id",
            "draw_id",
            "certification_stream_commitment_hash",
            "joint_fqr_family_registration_hash",
            "corrected_evidence_stream_identity_hash",
            "literal_jacobian_stream_identity_hash",
            "raw_logit_stream_identity_hash",
            "same_capture_and_vjp_execution_proof_hash",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
            "packed_direction_execution_proof_hash",
            "raw_logit_interval_proof_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        generation = _exact_nonnegative_int(
            self.proposal_generation,
            name="proposal generation",
        )
        route_count = _exact_positive_int(self.route_count, name="route count")
        sealed_index = _exact_nonnegative_int(
            self.sealed_order_index,
            name="sealed order index",
        )
        endpoints = _identity_axis(
            self.endpoint_action_hashes,
            name="endpoint action axis",
        )
        shape = (len(endpoints), 2)
        signed_lower = _immutable_f64(
            self.signed_raw_logit_delta_lower_by_endpoint_and_dose,
            shape=shape,
            name="signed raw-logit delta lower bounds",
        )
        signed_upper = _immutable_f64(
            self.signed_raw_logit_delta_upper_by_endpoint_and_dose,
            shape=shape,
            name="signed raw-logit delta upper bounds",
        )
        if bool(np.any(signed_lower > signed_upper)):
            _refuse(
                V1FiberedDirectDisplacementRefusalCode.INTERVAL_INVALID,
                "signed raw-logit displacement lower bound exceeds its upper bound",
            )
        raw_upper = _finite_nonnegative(
            self.raw_logit_absolute_upper,
            name="raw-logit absolute upper bound",
        )
        doses = tuple(
            _finite_nonnegative(value, name="direct-displacement dose")
            for value in self.doses
        )
        if doses != DIRECTIONAL_SECANT_DOSES:
            _refuse(
                V1FiberedDirectDisplacementRefusalCode.STALE_COORDINATE,
                "direct displacement requires doses (0.5, 1.0)",
            )
        required_true = (
            "source_bounds_registered_before_outcomes",
            "raw_logit_delta_intervals_are_outward",
            "endpoint_order_unchanged_after_outcomes",
        )
        if any(getattr(self, name) is not True for name in required_true):
            _refuse(
                V1FiberedDirectDisplacementRefusalCode.POST_SELECTION,
                "source bounds or endpoint order were not frozen before outcomes",
            )
        if self.postselection_occurred is not False:
            _refuse(
                V1FiberedDirectDisplacementRefusalCode.POST_SELECTION,
                "direct displacement was post-selected",
            )
        if any(
            getattr(self, name) is not False
            for name in (
                "alpha_allocation_performed",
                "alpha_allocation_authority",
                "candidate_selection_authority",
                "calibration_authority",
                "calibration_K_open_authority",
                "bce_oracle_authority",
                "paired_oracle_O_open_authority",
                "mask_commit_authority",
                "terminal_authority",
            )
        ):
            _refuse(
                V1FiberedDirectDisplacementRefusalCode.AUTHORITY_VIOLATION,
                "direct-displacement evidence cannot grant downstream authority",
            )
        if (
            self.direct_displacement_equation
            != FIBERED_DIRECT_DISPLACEMENT_EQUATION
            or self.analytic_bound_equation
            != FIBERED_DIRECT_DISPLACEMENT_BOUND_EQUATION
            or self.authority_scope != FIBERED_DIRECT_DISPLACEMENT_AUTHORITY_SCOPE
        ):
            _refuse(
                V1FiberedDirectDisplacementRefusalCode.AUTHORITY_VIOLATION,
                "direct-displacement equations or authority scope are invalid",
            )
        if self.schema != FIBERED_DIRECT_DISPLACEMENT_OBSERVATION_SCHEMA:
            _refuse(
                V1FiberedDirectDisplacementRefusalCode.INVALID_TYPE,
                "direct-displacement observation schema is invalid",
            )

        displacement_lower, displacement_upper = _derive_absolute_displacement(
            signed_lower,
            signed_upper,
        )
        analytic_upper = _outward_twice_raw_bound(raw_upper)
        if bool(np.any(displacement_upper > analytic_upper)):
            _refuse(
                V1FiberedDirectDisplacementRefusalCode.ANALYTIC_BOUND_VIOLATION,
                "direct-displacement interval exceeds twice the source raw-logit bound",
            )

        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "sealed_order_index", sealed_index)
        object.__setattr__(self, "endpoint_action_hashes", endpoints)
        object.__setattr__(self, "raw_logit_absolute_upper", raw_upper)
        object.__setattr__(self, "doses", doses)
        object.__setattr__(
            self,
            "signed_raw_logit_delta_lower_by_endpoint_and_dose",
            signed_lower,
        )
        object.__setattr__(
            self,
            "signed_raw_logit_delta_upper_by_endpoint_and_dose",
            signed_upper,
        )
        object.__setattr__(
            self,
            "analytic_direct_displacement_absolute_upper",
            analytic_upper,
        )
        object.__setattr__(
            self,
            "direct_displacement_lower_by_endpoint_and_dose",
            _immutable_f64(
                displacement_lower,
                shape=shape,
                name="direct-displacement lower bounds",
                nonnegative=True,
            ),
        )
        object.__setattr__(
            self,
            "direct_displacement_upper_by_endpoint_and_dose",
            _immutable_f64(
                displacement_upper,
                shape=shape,
                name="direct-displacement upper bounds",
                nonnegative=True,
            ),
        )
        self.assert_integrity()

    @property
    def d_half_lower_by_endpoint(self) -> np.ndarray:
        return self.direct_displacement_lower_by_endpoint_and_dose[:, 0]

    @property
    def d_half_upper_by_endpoint(self) -> np.ndarray:
        return self.direct_displacement_upper_by_endpoint_and_dose[:, 0]

    @property
    def d_one_lower_by_endpoint(self) -> np.ndarray:
        return self.direct_displacement_lower_by_endpoint_and_dose[:, 1]

    @property
    def d_one_upper_by_endpoint(self) -> np.ndarray:
        return self.direct_displacement_upper_by_endpoint_and_dose[:, 1]

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "directional_record_hash",
                "source_observation_bounds_hash",
                "source_bound_source_hash",
                "projection_closure_hash",
                "design_hash",
                "ladder_hash",
                "chart_identity_hash",
                "parent_mask_hash",
                "parent_execution_identity_hash",
                "proposal_generation",
                "route_count",
                "endpoint_action_hashes",
                "stratum_id",
                "sealed_order_index",
                "context_id",
                "draw_id",
                "certification_stream_commitment_hash",
                "joint_fqr_family_registration_hash",
                "corrected_evidence_stream_identity_hash",
                "literal_jacobian_stream_identity_hash",
                "raw_logit_stream_identity_hash",
                "same_capture_and_vjp_execution_proof_hash",
                "native_fiber_axis_receipt_hash",
                "native_fiber_measure_hash",
                "packed_direction_execution_proof_hash",
                "raw_logit_interval_proof_hash",
                "raw_logit_absolute_upper",
                "analytic_direct_displacement_absolute_upper",
                "signed_raw_logit_delta_lower_by_endpoint_and_dose",
                "signed_raw_logit_delta_upper_by_endpoint_and_dose",
                "direct_displacement_lower_by_endpoint_and_dose",
                "direct_displacement_upper_by_endpoint_and_dose",
                "doses",
                "direct_displacement_equation",
                "analytic_bound_equation",
                "source_bounds_registered_before_outcomes",
                "raw_logit_delta_intervals_are_outward",
                "endpoint_order_unchanged_after_outcomes",
                "postselection_occurred",
                "alpha_allocation_performed",
                "alpha_allocation_authority",
                "candidate_selection_authority",
                "calibration_authority",
                "calibration_K_open_authority",
                "bce_oracle_authority",
                "paired_oracle_O_open_authority",
                "mask_commit_authority",
                "terminal_authority",
                "authority_scope",
            )
        }

    @property
    def observation_hash(self) -> str:
        cached = self._observation_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_OBSERVATION_PREFIX)
        if self._observation_hash is not None and observed != self._observation_hash:
            _refuse(
                V1FiberedDirectDisplacementRefusalCode.HASH_MISMATCH,
                "direct-displacement observation was tampered",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1FiberedDirectDisplacementRefusalCode.HASH_MISMATCH,
                "direct-displacement observation hash does not match",
            )
        object.__setattr__(self, "_observation_hash", observed)
        return observed


def derive_v1_fibered_direct_displacement_observation(
    *,
    record: V1FiberedDirectionalRecord,
    observation_bounds: V1FiberedDirectionalSecantObservationBounds,
) -> V1FiberedDirectDisplacementObservation:
    """Derive exact absolute secant intervals from one source-bound C row."""

    if type(record) is not V1FiberedDirectionalRecord:
        _refuse(
            V1FiberedDirectDisplacementRefusalCode.LEGACY_RECORD,
            "direct displacement requires the exact fibered directional record",
        )
    if type(observation_bounds) is not V1FiberedDirectionalSecantObservationBounds:
        _refuse(
            V1FiberedDirectDisplacementRefusalCode.LEGACY_BOUND,
            "direct displacement requires exact pre-outcome fibered bounds",
        )
    if (
        record.postselection_occurred is not False
        or record.endpoint_order_unchanged_after_outcomes is not True
        or observation_bounds.registered_before_secant_outcomes is not True
    ):
        _refuse(
            V1FiberedDirectDisplacementRefusalCode.POST_SELECTION,
            "record, endpoint order, or source bound was post-selected",
        )
    try:
        record_hash = record.assert_integrity()
    except Exception as error:
        _refuse(
            V1FiberedDirectDisplacementRefusalCode.HASH_MISMATCH,
            f"fibered directional record integrity failed: {error}",
        )
    try:
        bounds_hash = observation_bounds.assert_integrity()
    except Exception as error:
        _refuse(
            V1FiberedDirectDisplacementRefusalCode.HASH_MISMATCH,
            f"fibered source-bound integrity failed: {error}",
        )

    mismatches: list[str] = []
    if observation_bounds.design_hash != record.design_hash:
        mismatches.append("design_hash")
    if observation_bounds.endpoint_action_hashes != record.endpoint_action_hashes:
        mismatches.append("endpoint_action_hashes")
    if (
        observation_bounds.native_fiber_axis_receipt_hash
        != record.native_fiber_axis_receipt_hash
    ):
        mismatches.append("native_fiber_axis_receipt_hash")
    if (
        observation_bounds.native_fiber_measure_hash
        != record.native_fiber_measure_hash
    ):
        mismatches.append("native_fiber_measure_hash")
    if observation_bounds.global_corrected_scalar_projection_used is not False:
        mismatches.append("global_corrected_scalar_projection_used")
    if mismatches:
        _refuse(
            V1FiberedDirectDisplacementRefusalCode.STALE_COORDINATE,
            "source bound differs from the fibered C row at: "
            + ", ".join(mismatches),
        )

    _validate_raw_logit_support(
        record=record,
        raw_upper=observation_bounds.raw_logit_absolute_upper,
    )
    return V1FiberedDirectDisplacementObservation(
        directional_record_hash=record_hash,
        source_observation_bounds_hash=bounds_hash,
        source_bound_source_hash=observation_bounds.bound_source_hash,
        projection_closure_hash=record.projection_closure_hash,
        design_hash=record.design_hash,
        ladder_hash=record.ladder_hash,
        chart_identity_hash=record.chart_identity_hash,
        parent_mask_hash=record.parent_mask_hash,
        parent_execution_identity_hash=record.parent_execution_identity_hash,
        proposal_generation=record.proposal_generation,
        route_count=record.route_count,
        endpoint_action_hashes=record.endpoint_action_hashes,
        stratum_id=record.stratum_id,
        sealed_order_index=record.sealed_order_index,
        context_id=record.context_id,
        draw_id=record.draw_id,
        certification_stream_commitment_hash=(
            record.certification_stream_commitment_hash
        ),
        joint_fqr_family_registration_hash=(
            record.joint_fqr_family_registration_hash
        ),
        corrected_evidence_stream_identity_hash=(
            record.corrected_evidence_stream_identity_hash
        ),
        literal_jacobian_stream_identity_hash=(
            record.literal_jacobian_stream_identity_hash
        ),
        raw_logit_stream_identity_hash=record.raw_logit_stream_identity_hash,
        same_capture_and_vjp_execution_proof_hash=(
            record.same_capture_and_vjp_execution_proof_hash
        ),
        native_fiber_axis_receipt_hash=record.native_fiber_axis_receipt_hash,
        native_fiber_measure_hash=record.native_fiber_measure_hash,
        packed_direction_execution_proof_hash=(
            record.packed_direction_execution_proof_hash
        ),
        raw_logit_interval_proof_hash=record.raw_logit_interval_proof_hash,
        raw_logit_absolute_upper=observation_bounds.raw_logit_absolute_upper,
        signed_raw_logit_delta_lower_by_endpoint_and_dose=(
            record.signed_raw_logit_delta_lower_by_endpoint
        ),
        signed_raw_logit_delta_upper_by_endpoint_and_dose=(
            record.signed_raw_logit_delta_upper_by_endpoint
        ),
        _issuance_token=_ISSUANCE_TOKEN,
    )


__all__ = [
    "FIBERED_DIRECT_DISPLACEMENT_AUTHORITY_SCOPE",
    "FIBERED_DIRECT_DISPLACEMENT_BOUND_EQUATION",
    "FIBERED_DIRECT_DISPLACEMENT_EQUATION",
    "FIBERED_DIRECT_DISPLACEMENT_OBSERVATION_SCHEMA",
    "V1FiberedDirectDisplacementObservation",
    "V1FiberedDirectDisplacementRefusal",
    "V1FiberedDirectDisplacementRefusalCode",
    "derive_v1_fibered_direct_displacement_observation",
]
