"""Close native-fiber corrected work with global-J and whole-action secants.

This is the authority-narrow sibling of :mod:`modern_v1_directional_remainder`.
The historical record takes one globally signed corrected-evidence projection
and forms its magnitude.  That is not the canonical corrected-work estimand:
signed cancellation is lawful only inside a registered native downstream-gate
fiber, before counting fibers and probability-averaging cuts.

The upstream fibered projector therefore supplies an *already nonnegative*
outward interval for corrected work.  This module copies that interval
directly; it never receives, reconstructs, or magnitudes a global corrected
sum.  Literal-J remains one globally signed complete-action projection for
``Q=(Jz)^2``.  Raw-logit evaluations at doses one half and one remain
complete-action secants for the nonlinear remainder ``R``.

The closure and record are evidence only.  They grant no calibration, BCE
oracle, mask-commit, or terminal authority.
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

from .modern_v1_directional_remainder import (
    DIRECTIONAL_SECANT_DOSES,
    SIGNED_REMAINDER_EQUATION,
    V1DirectionalSecantDesign,
)
from .modern_v1_dual_chart_geometry import POSITIVE_REMOVAL_SIGN
from .modern_v1_fibered_work_projection_gpu import (
    V1FiberedWorkProjectionRequest,
    V1FiberedWorkProjectionResult,
)
from .modern_v1_ladder_projection_gpu import (
    V1DirectionalActionProjectionRequest,
    V1LadderProjectionAcquisition,
)
from .modern_v1_native_fiber_axis import (
    UNIFORM_CUT_COUNTING_FIBER_MEASURE_EQUATION,
    V1NativeDestinationFiberAxisReceipt,
)


FIBERED_DIRECTIONAL_PROJECTION_CLOSURE_SCHEMA: Final = (
    "partition_free_v1_fibered_directional_projection_closure_v1"
)
FIBERED_DIRECTIONAL_RECORD_SCHEMA: Final = (
    "partition_free_v1_fibered_directional_record_v1"
)
NATIVE_FIBERED_WORK_PROJECTION_SCHEMA: Final = (
    "partition_free_v1_native_fibered_corrected_work_projection_v1"
)
GLOBAL_LITERAL_J_PROJECTION_EQUATION: Final = (
    "literal_positive_removal_J_projection_is_global_signed_complete_action_v1"
)
FIBERED_FQR_OBSERVATION_EQUATION: Final = (
    "F_from_native_fibered_work_interval_Q_from_global_signed_literal_J_squared_"
    "R_from_complete_action_raw_logit_secants_v1"
)
FIBERED_DIRECTIONAL_AUTHORITY_SCOPE: Final = (
    "native_fibered_fqr_observation_evidence_only_no_calibration_oracle_commit_"
    "or_terminal_authority_v1"
)

_CLOSURE_PREFIX: Final = "partition-free-v1-fibered-directional-closure-"
_RECORD_PREFIX: Final = "partition-free-v1-fibered-directional-record-"
_LE_F64: Final = np.dtype("<f8")
_ISSUANCE_TOKEN: Final = object()


class V1FiberedDirectionalRecordRefusalCode(str, Enum):
    """Fail-closed reasons at the fibered directional seam."""

    INVALID_TYPE = "invalid_type"
    HASH_MISMATCH = "hash_mismatch"
    STALE_COORDINATE = "stale_coordinate"
    FIBER_RECEIPT_MISMATCH = "fiber_receipt_mismatch"
    POST_SELECTION = "post_selection"
    LEGACY_SCALAR_F = "legacy_scalar_f"
    INTERVAL_INVALID = "interval_invalid"
    AUTHORITY_VIOLATION = "authority_violation"


class V1FiberedDirectionalRecordRefusal(ValueError):
    """Typed refusal; unresolved evidence never becomes an F/Q/R record."""

    def __init__(
        self,
        code: V1FiberedDirectionalRecordRefusalCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1FiberedDirectionalRecordRefusalCode,
    message: str,
) -> NoReturn:
    raise V1FiberedDirectionalRecordRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1FiberedDirectionalRecordRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _exact_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1FiberedDirectionalRecordRefusalCode.INVALID_TYPE,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1FiberedDirectionalRecordRefusalCode.STALE_COORDINATE,
            f"{name} must be nonnegative",
        )
    return result


def _exact_positive_int(value: object, *, name: str) -> int:
    result = _exact_nonnegative_int(value, name=name)
    if result == 0:
        _refuse(
            V1FiberedDirectionalRecordRefusalCode.STALE_COORDINATE,
            f"{name} must be positive",
        )
    return result


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1FiberedDirectionalRecordRefusalCode.INVALID_TYPE,
            f"{name} must be one real scalar",
        )
    result = float(value)
    if not math.isfinite(result):
        _refuse(
            V1FiberedDirectionalRecordRefusalCode.INTERVAL_INVALID,
            f"{name} must be finite",
        )
    return 0.0 if result == 0.0 else result


def _finite_nonnegative(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if result < 0.0:
        _refuse(
            V1FiberedDirectionalRecordRefusalCode.INTERVAL_INVALID,
            f"{name} must be nonnegative",
        )
    return result


def _identity_axis(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) not in {tuple, list}:
        _refuse(
            V1FiberedDirectionalRecordRefusalCode.INVALID_TYPE,
            f"{name} must be one ordered sequence",
        )
    result = tuple(_identity(item, name=f"{name} item") for item in value)
    if not result or len(result) != len(set(result)):
        _refuse(
            V1FiberedDirectionalRecordRefusalCode.STALE_COORDINATE,
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
            V1FiberedDirectionalRecordRefusalCode.INVALID_TYPE,
            f"{name} must be convertible to float64: {error}",
        )
    if array.shape != shape or not bool(np.isfinite(array).all()):
        _refuse(
            V1FiberedDirectionalRecordRefusalCode.INTERVAL_INVALID,
            f"{name} must be finite with shape {shape}",
        )
    if nonnegative and bool(np.any(array < 0.0)):
        _refuse(
            V1FiberedDirectionalRecordRefusalCode.INTERVAL_INVALID,
            f"{name} must be nonnegative",
        )
    array[array == 0.0] = 0.0
    result = np.frombuffer(array.tobytes(order="C"), dtype=_LE_F64).reshape(shape)
    result.setflags(write=False)
    return result


def _decimal(value: float) -> Decimal:
    return Decimal.from_float(value)


def _outward_float(value: Decimal, *, upward: bool, name: str) -> float:
    """Convert an exact Decimal expression to one outward binary64 bound."""

    try:
        result = float(value)
    except (OverflowError, ValueError, InvalidOperation) as error:
        _refuse(
            V1FiberedDirectionalRecordRefusalCode.INTERVAL_INVALID,
            f"{name} cannot be represented in binary64: {error}",
        )
    if not math.isfinite(result):
        _refuse(
            V1FiberedDirectionalRecordRefusalCode.INTERVAL_INVALID,
            f"{name} exceeds finite binary64",
        )
    represented = Decimal.from_float(result)
    if upward and represented < value:
        result = math.nextafter(result, math.inf)
    elif not upward and represented > value:
        result = math.nextafter(result, -math.inf)
    if not math.isfinite(result):
        _refuse(
            V1FiberedDirectionalRecordRefusalCode.INTERVAL_INVALID,
            f"outward {name} exceeds finite binary64",
        )
    return 0.0 if result == 0.0 else result


def _absolute_interval(
    lower: Decimal,
    upper: Decimal,
) -> tuple[Decimal, Decimal]:
    if lower > upper:
        _refuse(
            V1FiberedDirectionalRecordRefusalCode.INTERVAL_INVALID,
            "derived signed interval is reversed",
        )
    if lower <= 0 <= upper:
        minimum = Decimal(0)
    else:
        minimum = min(abs(lower), abs(upper))
    return minimum, max(abs(lower), abs(upper))


@dataclass(frozen=True, slots=True)
class V1FiberedDirectionalProjectionClosure:
    """Provider-closed native-fiber F interval and global signed-J interval.

    ``fibered_work_*`` is already the nonnegative quantity
    ``sum_cut pi_cut sum_fiber |sum_route_in_fiber a_j z_j|`` for one context
    and every endpoint.  No globally signed corrected projection is admitted.
    """

    design_hash: str
    ladder_hash: str
    chart_identity_hash: str
    parent_mask_hash: str
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
    fibered_work_projection_receipt_hash: str
    literal_jacobian_projection_receipt_hash: str
    fibered_work_lower_by_endpoint: np.ndarray = field(repr=False, compare=False)
    fibered_work_upper_by_endpoint: np.ndarray = field(repr=False, compare=False)
    global_literal_jacobian_center_by_endpoint: np.ndarray = field(
        repr=False, compare=False
    )
    global_literal_jacobian_radius_by_endpoint: np.ndarray = field(
        repr=False, compare=False
    )
    fibered_work_projection_schema: str = NATIVE_FIBERED_WORK_PROJECTION_SCHEMA
    fibered_work_equation: str = UNIFORM_CUT_COUNTING_FIBER_MEASURE_EQUATION
    global_literal_jacobian_projection_equation: str = (
        GLOBAL_LITERAL_J_PROJECTION_EQUATION
    )
    fibered_work_values_are_nonnegative: bool = True
    fibered_work_bounds_are_outward: bool = True
    within_native_fiber_signed_cancellation_only: bool = True
    magnitude_precedes_cut_measure: bool = True
    global_corrected_scalar_projection_used: bool = False
    literal_jacobian_projection_is_global_signed: bool = True
    same_capture_and_vjp_certified: bool = True
    projection_plan_issued_before_outcomes: bool = True
    endpoint_order_unchanged_after_outcomes: bool = True
    postselection_occurred: bool = False
    calibration_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = FIBERED_DIRECTIONAL_AUTHORITY_SCOPE
    schema: str = FIBERED_DIRECTIONAL_PROJECTION_CLOSURE_SCHEMA
    _closure_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "design_hash",
            "ladder_hash",
            "chart_identity_hash",
            "parent_mask_hash",
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
            "fibered_work_projection_receipt_hash",
            "literal_jacobian_projection_receipt_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        generation = _exact_nonnegative_int(
            self.proposal_generation, name="proposal generation"
        )
        route_count = _exact_positive_int(self.route_count, name="route count")
        index = _exact_nonnegative_int(
            self.sealed_order_index, name="sealed order index"
        )
        endpoints = _identity_axis(
            self.endpoint_action_hashes, name="endpoint action axis"
        )
        shape = (len(endpoints),)
        f_lower = _immutable_f64(
            self.fibered_work_lower_by_endpoint,
            shape=shape,
            name="fibered-work lower bounds",
            nonnegative=True,
        )
        f_upper = _immutable_f64(
            self.fibered_work_upper_by_endpoint,
            shape=shape,
            name="fibered-work upper bounds",
            nonnegative=True,
        )
        if bool(np.any(f_lower > f_upper)):
            _refuse(
                V1FiberedDirectionalRecordRefusalCode.INTERVAL_INVALID,
                "fibered-work lower bound exceeds its upper bound",
            )
        j_center = _immutable_f64(
            self.global_literal_jacobian_center_by_endpoint,
            shape=shape,
            name="global literal-J centers",
        )
        j_radius = _immutable_f64(
            self.global_literal_jacobian_radius_by_endpoint,
            shape=shape,
            name="global literal-J radii",
            nonnegative=True,
        )
        if self.fibered_work_projection_schema != NATIVE_FIBERED_WORK_PROJECTION_SCHEMA:
            _refuse(
                V1FiberedDirectionalRecordRefusalCode.LEGACY_SCALAR_F,
                "F must come from the native-fibered work projector; scalar-F "
                "legacy projections are authority-ineligible",
            )
        if (
            self.fibered_work_equation
            != UNIFORM_CUT_COUNTING_FIBER_MEASURE_EQUATION
            or self.global_literal_jacobian_projection_equation
            != GLOBAL_LITERAL_J_PROJECTION_EQUATION
        ):
            _refuse(
                V1FiberedDirectionalRecordRefusalCode.LEGACY_SCALAR_F,
                "directional projection equations differ from fibered-F/global-J v1",
            )
        required_true = (
            "fibered_work_values_are_nonnegative",
            "fibered_work_bounds_are_outward",
            "within_native_fiber_signed_cancellation_only",
            "magnitude_precedes_cut_measure",
            "literal_jacobian_projection_is_global_signed",
            "same_capture_and_vjp_certified",
            "projection_plan_issued_before_outcomes",
            "endpoint_order_unchanged_after_outcomes",
        )
        if any(getattr(self, name) is not True for name in required_true):
            _refuse(
                V1FiberedDirectionalRecordRefusalCode.LEGACY_SCALAR_F,
                "fibered F or global signed-J projection proof is incomplete",
            )
        if self.global_corrected_scalar_projection_used is not False:
            _refuse(
                V1FiberedDirectionalRecordRefusalCode.LEGACY_SCALAR_F,
                "abs(global corrected sum) is forbidden as canonical F",
            )
        if self.postselection_occurred is not False:
            _refuse(
                V1FiberedDirectionalRecordRefusalCode.POST_SELECTION,
                "fibered projection closure was post-selected",
            )
        if any(
            getattr(self, name) is not False
            for name in (
                "calibration_authority",
                "bce_oracle_authority",
                "mask_commit_authority",
                "terminal_authority",
            )
        ):
            _refuse(
                V1FiberedDirectionalRecordRefusalCode.AUTHORITY_VIOLATION,
                "fibered projection closure cannot grant downstream authority",
            )
        if self.authority_scope != FIBERED_DIRECTIONAL_AUTHORITY_SCOPE:
            _refuse(
                V1FiberedDirectionalRecordRefusalCode.AUTHORITY_VIOLATION,
                "fibered projection closure authority scope is invalid",
            )
        if self.schema != FIBERED_DIRECTIONAL_PROJECTION_CLOSURE_SCHEMA:
            _refuse(
                V1FiberedDirectionalRecordRefusalCode.INVALID_TYPE,
                "fibered directional projection closure schema is invalid",
            )
        if (
            self.corrected_evidence_stream_identity_hash
            == self.literal_jacobian_stream_identity_hash
        ):
            _refuse(
                V1FiberedDirectionalRecordRefusalCode.STALE_COORDINATE,
                "corrected-evidence and literal-J streams collapsed",
            )
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "sealed_order_index", index)
        object.__setattr__(self, "endpoint_action_hashes", endpoints)
        object.__setattr__(self, "fibered_work_lower_by_endpoint", f_lower)
        object.__setattr__(self, "fibered_work_upper_by_endpoint", f_upper)
        object.__setattr__(
            self, "global_literal_jacobian_center_by_endpoint", j_center
        )
        object.__setattr__(
            self, "global_literal_jacobian_radius_by_endpoint", j_radius
        )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "design_hash",
                "ladder_hash",
                "chart_identity_hash",
                "parent_mask_hash",
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
                "fibered_work_projection_receipt_hash",
                "literal_jacobian_projection_receipt_hash",
                "fibered_work_lower_by_endpoint",
                "fibered_work_upper_by_endpoint",
                "global_literal_jacobian_center_by_endpoint",
                "global_literal_jacobian_radius_by_endpoint",
                "fibered_work_projection_schema",
                "fibered_work_equation",
                "global_literal_jacobian_projection_equation",
                "fibered_work_values_are_nonnegative",
                "fibered_work_bounds_are_outward",
                "within_native_fiber_signed_cancellation_only",
                "magnitude_precedes_cut_measure",
                "global_corrected_scalar_projection_used",
                "literal_jacobian_projection_is_global_signed",
                "same_capture_and_vjp_certified",
                "projection_plan_issued_before_outcomes",
                "endpoint_order_unchanged_after_outcomes",
                "postselection_occurred",
                "calibration_authority",
                "bce_oracle_authority",
                "mask_commit_authority",
                "terminal_authority",
                "authority_scope",
            )
        }

    @property
    def closure_hash(self) -> str:
        cached = self._closure_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_CLOSURE_PREFIX)
        if self._closure_hash is not None and observed != self._closure_hash:
            _refuse(
                V1FiberedDirectionalRecordRefusalCode.HASH_MISMATCH,
                "fibered projection closure was tampered",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1FiberedDirectionalRecordRefusalCode.HASH_MISMATCH,
                "fibered projection closure hash does not match",
            )
        object.__setattr__(self, "_closure_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1FiberedDirectionalRecord:
    """One native-fibered F / global-J Q / complete-secant R observation."""

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
    fibered_work_projection_receipt_hash: str
    literal_jacobian_projection_receipt_hash: str
    packed_direction_execution_proof_hash: str
    raw_logit_interval_proof_hash: str
    parent_raw_target_logit_center: float
    parent_raw_target_logit_radius: float
    fibered_work_lower_by_endpoint: np.ndarray = field(repr=False, compare=False)
    fibered_work_upper_by_endpoint: np.ndarray = field(repr=False, compare=False)
    global_literal_jacobian_center_by_endpoint: np.ndarray = field(
        repr=False, compare=False
    )
    global_literal_jacobian_radius_by_endpoint: np.ndarray = field(
        repr=False, compare=False
    )
    dosed_raw_target_logit_center_by_endpoint: np.ndarray = field(
        repr=False, compare=False
    )
    dosed_raw_target_logit_radius_by_endpoint: np.ndarray = field(
        repr=False, compare=False
    )
    doses: tuple[float, float] = DIRECTIONAL_SECANT_DOSES
    positive_removal_sign: str = POSITIVE_REMOVAL_SIGN
    signed_remainder_equation: str = SIGNED_REMAINDER_EQUATION
    observation_equation: str = FIBERED_FQR_OBSERVATION_EQUATION
    fibered_work_projection_schema: str = NATIVE_FIBERED_WORK_PROJECTION_SCHEMA
    global_corrected_scalar_projection_used: bool = False
    literal_jacobian_projection_is_global_signed: bool = True
    secants_execute_complete_actions: bool = True
    endpoint_order_unchanged_after_outcomes: bool = True
    postselection_occurred: bool = False
    calibration_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = FIBERED_DIRECTIONAL_AUTHORITY_SCOPE
    schema: str = FIBERED_DIRECTIONAL_RECORD_SCHEMA
    f_lower_observation_by_endpoint: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    f_upper_observation_by_endpoint: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    global_literal_jacobian_lower_by_endpoint: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    global_literal_jacobian_upper_by_endpoint: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    q_lower_observation_by_endpoint: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    q_upper_observation_by_endpoint: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    signed_raw_logit_delta_lower_by_endpoint: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    signed_raw_logit_delta_upper_by_endpoint: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    signed_remainder_lower_by_endpoint: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    signed_remainder_upper_by_endpoint: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    r_lower_observation_by_endpoint: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    r_upper_observation_by_endpoint: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    midpoint_higher_order_lower_by_endpoint: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    midpoint_higher_order_upper_by_endpoint: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    _issuance_token: object = field(default=None, repr=False, compare=False)
    _record_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self._issuance_token is not _ISSUANCE_TOKEN:
            _refuse(
                V1FiberedDirectionalRecordRefusalCode.INVALID_TYPE,
                "fibered directional records must be issued by their closure",
            )
        for name in (
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
            "fibered_work_projection_receipt_hash",
            "literal_jacobian_projection_receipt_hash",
            "packed_direction_execution_proof_hash",
            "raw_logit_interval_proof_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        generation = _exact_nonnegative_int(
            self.proposal_generation, name="proposal generation"
        )
        route_count = _exact_positive_int(self.route_count, name="route count")
        index = _exact_nonnegative_int(
            self.sealed_order_index, name="sealed order index"
        )
        endpoints = _identity_axis(
            self.endpoint_action_hashes, name="endpoint action axis"
        )
        endpoint_count = len(endpoints)
        vector_shape = (endpoint_count,)
        matrix_shape = (endpoint_count, 2)
        f_lower = _immutable_f64(
            self.fibered_work_lower_by_endpoint,
            shape=vector_shape,
            name="fibered-work lower bounds",
            nonnegative=True,
        )
        f_upper = _immutable_f64(
            self.fibered_work_upper_by_endpoint,
            shape=vector_shape,
            name="fibered-work upper bounds",
            nonnegative=True,
        )
        j_center = _immutable_f64(
            self.global_literal_jacobian_center_by_endpoint,
            shape=vector_shape,
            name="global literal-J centers",
        )
        j_radius = _immutable_f64(
            self.global_literal_jacobian_radius_by_endpoint,
            shape=vector_shape,
            name="global literal-J radii",
            nonnegative=True,
        )
        child_center = _immutable_f64(
            self.dosed_raw_target_logit_center_by_endpoint,
            shape=matrix_shape,
            name="dosed raw-logit centers",
        )
        child_radius = _immutable_f64(
            self.dosed_raw_target_logit_radius_by_endpoint,
            shape=matrix_shape,
            name="dosed raw-logit radii",
            nonnegative=True,
        )
        parent_center = _finite(
            self.parent_raw_target_logit_center, name="parent raw target logit"
        )
        parent_radius = _finite_nonnegative(
            self.parent_raw_target_logit_radius,
            name="parent raw target-logit radius",
        )
        doses = tuple(_finite_nonnegative(value, name="secant dose") for value in self.doses)
        if doses != DIRECTIONAL_SECANT_DOSES:
            _refuse(
                V1FiberedDirectionalRecordRefusalCode.STALE_COORDINATE,
                "fibered directional records require doses (0.5, 1.0)",
            )
        if (
            self.positive_removal_sign != POSITIVE_REMOVAL_SIGN
            or self.signed_remainder_equation != SIGNED_REMAINDER_EQUATION
            or self.observation_equation != FIBERED_FQR_OBSERVATION_EQUATION
            or self.fibered_work_projection_schema
            != NATIVE_FIBERED_WORK_PROJECTION_SCHEMA
        ):
            _refuse(
                V1FiberedDirectionalRecordRefusalCode.LEGACY_SCALAR_F,
                "fibered F/Q/R record semantics differ from canonical v1",
            )
        if (
            self.global_corrected_scalar_projection_used is not False
            or self.literal_jacobian_projection_is_global_signed is not True
            or self.secants_execute_complete_actions is not True
        ):
            _refuse(
                V1FiberedDirectionalRecordRefusalCode.LEGACY_SCALAR_F,
                "record substituted scalar F, local J, or partial-action secants",
            )
        if (
            self.endpoint_order_unchanged_after_outcomes is not True
            or self.postselection_occurred is not False
        ):
            _refuse(
                V1FiberedDirectionalRecordRefusalCode.POST_SELECTION,
                "fibered directional record was post-selected",
            )
        if any(
            getattr(self, name) is not False
            for name in (
                "calibration_authority",
                "bce_oracle_authority",
                "mask_commit_authority",
                "terminal_authority",
            )
        ):
            _refuse(
                V1FiberedDirectionalRecordRefusalCode.AUTHORITY_VIOLATION,
                "fibered directional record cannot grant action authority",
            )
        if self.authority_scope != FIBERED_DIRECTIONAL_AUTHORITY_SCOPE:
            _refuse(
                V1FiberedDirectionalRecordRefusalCode.AUTHORITY_VIOLATION,
                "fibered directional record authority scope is invalid",
            )
        if self.schema != FIBERED_DIRECTIONAL_RECORD_SCHEMA:
            _refuse(
                V1FiberedDirectionalRecordRefusalCode.INVALID_TYPE,
                "fibered directional record schema is invalid",
            )

        j_lower = np.empty(vector_shape, dtype=np.float64)
        j_upper = np.empty(vector_shape, dtype=np.float64)
        q_lower = np.empty(vector_shape, dtype=np.float64)
        q_upper = np.empty(vector_shape, dtype=np.float64)
        delta_lower = np.empty(matrix_shape, dtype=np.float64)
        delta_upper = np.empty(matrix_shape, dtype=np.float64)
        remainder_lower = np.empty(matrix_shape, dtype=np.float64)
        remainder_upper = np.empty(matrix_shape, dtype=np.float64)
        r_lower = np.empty(matrix_shape, dtype=np.float64)
        r_upper = np.empty(matrix_shape, dtype=np.float64)
        higher_lower = np.empty(vector_shape, dtype=np.float64)
        higher_upper = np.empty(vector_shape, dtype=np.float64)

        # All arithmetic below is exact over the supplied binary64 centers and
        # radii.  Only the final conversions round, and they round outward.
        with localcontext() as decimal_context:
            # An exact binary64 subnormal can require 1,074 decimal places;
            # squaring an interval endpoint can require roughly twice that.
            # This fixed precision therefore keeps every finite binary64
            # center/radius expression here exact before the one final,
            # directed conversion back to binary64.
            decimal_context.prec = 2500
            p_center = _decimal(parent_center)
            p_radius = _decimal(parent_radius)
            p_lower = p_center - p_radius
            p_upper = p_center + p_radius
            for endpoint in range(endpoint_count):
                jc = _decimal(float(j_center[endpoint]))
                jr = _decimal(float(j_radius[endpoint]))
                jl = jc - jr
                ju = jc + jr
                j_lower[endpoint] = _outward_float(
                    jl, upward=False, name="literal-J lower bound"
                )
                j_upper[endpoint] = _outward_float(
                    ju, upward=True, name="literal-J upper bound"
                )
                abs_j_lower, abs_j_upper = _absolute_interval(jl, ju)
                q_lower[endpoint] = _outward_float(
                    abs_j_lower * abs_j_lower,
                    upward=False,
                    name="Q lower bound",
                )
                q_upper[endpoint] = _outward_float(
                    abs_j_upper * abs_j_upper,
                    upward=True,
                    name="Q upper bound",
                )
                signed_remainder_bounds: list[tuple[Decimal, Decimal]] = []
                for dose_index, dose in enumerate(doses):
                    cc = _decimal(float(child_center[endpoint, dose_index]))
                    cr = _decimal(float(child_radius[endpoint, dose_index]))
                    raw_lower = (cc - cr) - p_upper
                    raw_upper = (cc + cr) - p_lower
                    eta = _decimal(dose)
                    tangent_lower = eta * jl
                    tangent_upper = eta * ju
                    rem_lower = raw_lower - tangent_upper
                    rem_upper = raw_upper - tangent_lower
                    signed_remainder_bounds.append((rem_lower, rem_upper))
                    delta_lower[endpoint, dose_index] = _outward_float(
                        raw_lower, upward=False, name="raw-logit delta lower bound"
                    )
                    delta_upper[endpoint, dose_index] = _outward_float(
                        raw_upper, upward=True, name="raw-logit delta upper bound"
                    )
                    remainder_lower[endpoint, dose_index] = _outward_float(
                        rem_lower, upward=False, name="signed remainder lower bound"
                    )
                    remainder_upper[endpoint, dose_index] = _outward_float(
                        rem_upper, upward=True, name="signed remainder upper bound"
                    )
                    abs_r_lower, abs_r_upper = _absolute_interval(
                        rem_lower, rem_upper
                    )
                    r_lower[endpoint, dose_index] = _outward_float(
                        abs_r_lower,
                        upward=False,
                        name="absolute remainder lower bound",
                    )
                    r_upper[endpoint, dose_index] = _outward_float(
                        abs_r_upper,
                        upward=True,
                        name="absolute remainder upper bound",
                    )
                half_bounds, one_bounds = signed_remainder_bounds
                # R_1 - 4 R_1/2 cancels an exact quadratic response and exposes
                # higher-order chart failure.  It is a diagnostic only; the
                # canonical feasibility constraints continue to use the two
                # absolute R dose intervals separately.
                high_lower = one_bounds[0] - Decimal(4) * half_bounds[1]
                high_upper = one_bounds[1] - Decimal(4) * half_bounds[0]
                higher_lower[endpoint] = _outward_float(
                    high_lower,
                    upward=False,
                    name="midpoint higher-order lower bound",
                )
                higher_upper[endpoint] = _outward_float(
                    high_upper,
                    upward=True,
                    name="midpoint higher-order upper bound",
                )

        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "sealed_order_index", index)
        object.__setattr__(self, "endpoint_action_hashes", endpoints)
        object.__setattr__(self, "parent_raw_target_logit_center", parent_center)
        object.__setattr__(self, "parent_raw_target_logit_radius", parent_radius)
        object.__setattr__(self, "fibered_work_lower_by_endpoint", f_lower)
        object.__setattr__(self, "fibered_work_upper_by_endpoint", f_upper)
        object.__setattr__(
            self, "global_literal_jacobian_center_by_endpoint", j_center
        )
        object.__setattr__(
            self, "global_literal_jacobian_radius_by_endpoint", j_radius
        )
        object.__setattr__(
            self, "dosed_raw_target_logit_center_by_endpoint", child_center
        )
        object.__setattr__(
            self, "dosed_raw_target_logit_radius_by_endpoint", child_radius
        )
        object.__setattr__(self, "doses", doses)
        # F is copied directly.  In particular there is deliberately no abs()
        # of a globally signed corrected projection anywhere in this module.
        derived = {
            "f_lower_observation_by_endpoint": f_lower,
            "f_upper_observation_by_endpoint": f_upper,
            "global_literal_jacobian_lower_by_endpoint": j_lower,
            "global_literal_jacobian_upper_by_endpoint": j_upper,
            "q_lower_observation_by_endpoint": q_lower,
            "q_upper_observation_by_endpoint": q_upper,
            "signed_raw_logit_delta_lower_by_endpoint": delta_lower,
            "signed_raw_logit_delta_upper_by_endpoint": delta_upper,
            "signed_remainder_lower_by_endpoint": remainder_lower,
            "signed_remainder_upper_by_endpoint": remainder_upper,
            "r_lower_observation_by_endpoint": r_lower,
            "r_upper_observation_by_endpoint": r_upper,
            "midpoint_higher_order_lower_by_endpoint": higher_lower,
            "midpoint_higher_order_upper_by_endpoint": higher_upper,
        }
        for name, value in derived.items():
            object.__setattr__(
                self,
                name,
                _immutable_f64(
                    value,
                    shape=value.shape,
                    name=name,
                    nonnegative=name.startswith(("f_", "q_", "r_")),
                ),
            )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
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
                "fibered_work_projection_receipt_hash",
                "literal_jacobian_projection_receipt_hash",
                "packed_direction_execution_proof_hash",
                "raw_logit_interval_proof_hash",
                "parent_raw_target_logit_center",
                "parent_raw_target_logit_radius",
                "fibered_work_lower_by_endpoint",
                "fibered_work_upper_by_endpoint",
                "global_literal_jacobian_center_by_endpoint",
                "global_literal_jacobian_radius_by_endpoint",
                "dosed_raw_target_logit_center_by_endpoint",
                "dosed_raw_target_logit_radius_by_endpoint",
                "f_lower_observation_by_endpoint",
                "f_upper_observation_by_endpoint",
                "global_literal_jacobian_lower_by_endpoint",
                "global_literal_jacobian_upper_by_endpoint",
                "q_lower_observation_by_endpoint",
                "q_upper_observation_by_endpoint",
                "signed_raw_logit_delta_lower_by_endpoint",
                "signed_raw_logit_delta_upper_by_endpoint",
                "signed_remainder_lower_by_endpoint",
                "signed_remainder_upper_by_endpoint",
                "r_lower_observation_by_endpoint",
                "r_upper_observation_by_endpoint",
                "midpoint_higher_order_lower_by_endpoint",
                "midpoint_higher_order_upper_by_endpoint",
                "doses",
                "positive_removal_sign",
                "signed_remainder_equation",
                "observation_equation",
                "fibered_work_projection_schema",
                "global_corrected_scalar_projection_used",
                "literal_jacobian_projection_is_global_signed",
                "secants_execute_complete_actions",
                "endpoint_order_unchanged_after_outcomes",
                "postselection_occurred",
                "calibration_authority",
                "bce_oracle_authority",
                "mask_commit_authority",
                "terminal_authority",
                "authority_scope",
            )
        }

    @property
    def record_hash(self) -> str:
        cached = self._record_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_RECORD_PREFIX)
        if self._record_hash is not None and observed != self._record_hash:
            _refuse(
                V1FiberedDirectionalRecordRefusalCode.HASH_MISMATCH,
                "fibered directional record was tampered",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1FiberedDirectionalRecordRefusalCode.HASH_MISMATCH,
                "fibered directional record hash does not match",
            )
        object.__setattr__(self, "_record_hash", observed)
        return observed


def close_v1_fibered_directional_record(
    *,
    design: V1DirectionalSecantDesign,
    native_fiber_axis_receipt: V1NativeDestinationFiberAxisReceipt,
    projection_closure: V1FiberedDirectionalProjectionClosure,
    parent_execution_identity_hash: str,
    parent_raw_target_logit_center: float,
    parent_raw_target_logit_radius: float,
    dosed_raw_target_logit_center_by_endpoint: np.ndarray,
    dosed_raw_target_logit_radius_by_endpoint: np.ndarray,
    packed_direction_execution_proof_hash: str,
    raw_logit_interval_proof_hash: str,
    secants_execute_complete_actions: bool = True,
    endpoint_order_unchanged_after_outcomes: bool = True,
    postselection_occurred: bool = False,
) -> V1FiberedDirectionalRecord:
    """Close one C row after exact design, fiber-axis, and stream checks."""

    if type(design) is not V1DirectionalSecantDesign:
        _refuse(
            V1FiberedDirectionalRecordRefusalCode.INVALID_TYPE,
            "fibered directional closure requires the exact secant design",
        )
    if type(native_fiber_axis_receipt) is not V1NativeDestinationFiberAxisReceipt:
        _refuse(
            V1FiberedDirectionalRecordRefusalCode.INVALID_TYPE,
            "fibered directional closure requires the exact native-fiber receipt",
        )
    if type(projection_closure) is not V1FiberedDirectionalProjectionClosure:
        _refuse(
            V1FiberedDirectionalRecordRefusalCode.INVALID_TYPE,
            "fibered directional closure requires its typed projection closure",
        )
    design_hash = design.assert_integrity()
    native_fiber_axis_receipt.assert_integrity()
    closure_hash = projection_closure.assert_integrity()

    expected = {
        "design_hash": design_hash,
        "ladder_hash": design.ladder_hash,
        "chart_identity_hash": design.chart_identity_hash,
        "parent_mask_hash": design.ladder.current_mask_hash,
        "proposal_generation": design.ladder.proposal_generation,
        "route_count": design.ladder.route_count,
        "endpoint_action_hashes": design.endpoint_action_hashes,
        "certification_stream_commitment_hash": (
            design.certification_stream_commitment_hash
        ),
        "joint_fqr_family_registration_hash": (
            design.joint_fqr_family_registration_hash
        ),
        "raw_logit_stream_identity_hash": (
            design.ladder.raw_logit_stream_source_hash
        ),
    }
    stale = tuple(
        name
        for name, value in expected.items()
        if getattr(projection_closure, name) != value
    )
    if stale:
        _refuse(
            V1FiberedDirectionalRecordRefusalCode.STALE_COORDINATE,
            "fibered projection differs from the frozen design at: "
            + ", ".join(stale),
        )

    contexts = dict(design.certification_context_ids_by_stratum)
    axis = contexts.get(projection_closure.stratum_id)
    position = projection_closure.sealed_order_index
    if (
        axis is None
        or position >= len(axis)
        or axis[position] != projection_closure.context_id
    ):
        _refuse(
            V1FiberedDirectionalRecordRefusalCode.STALE_COORDINATE,
            "context/stratum/sealed-order coordinate is not a registered C prefix",
        )

    fiber_mismatches: list[str] = []
    if (
        projection_closure.native_fiber_axis_receipt_hash
        != native_fiber_axis_receipt.receipt_hash
    ):
        fiber_mismatches.append("native_fiber_axis_receipt_hash")
    if (
        projection_closure.native_fiber_measure_hash
        != native_fiber_axis_receipt.measure_hash
    ):
        fiber_mismatches.append("native_fiber_measure_hash")
    if native_fiber_axis_receipt.route_count != design.ladder.route_count:
        fiber_mismatches.append("route_count")
    if native_fiber_axis_receipt.route_axis_hash != design.ladder.route_axis_hash:
        fiber_mismatches.append("route_axis_hash")
    if fiber_mismatches:
        _refuse(
            V1FiberedDirectionalRecordRefusalCode.FIBER_RECEIPT_MISMATCH,
            "native-fiber authority differs at: " + ", ".join(fiber_mismatches),
        )
    if (
        secants_execute_complete_actions is not True
        or endpoint_order_unchanged_after_outcomes is not True
        or postselection_occurred is not False
    ):
        _refuse(
            V1FiberedDirectionalRecordRefusalCode.POST_SELECTION,
            "secants were partial, reordered, or post-selected",
        )

    return V1FiberedDirectionalRecord(
        projection_closure_hash=closure_hash,
        design_hash=design_hash,
        ladder_hash=design.ladder_hash,
        chart_identity_hash=design.chart_identity_hash,
        parent_mask_hash=design.ladder.current_mask_hash,
        parent_execution_identity_hash=_identity(
            parent_execution_identity_hash,
            name="parent execution identity hash",
        ),
        proposal_generation=design.ladder.proposal_generation,
        route_count=design.ladder.route_count,
        endpoint_action_hashes=design.endpoint_action_hashes,
        stratum_id=projection_closure.stratum_id,
        sealed_order_index=projection_closure.sealed_order_index,
        context_id=projection_closure.context_id,
        draw_id=projection_closure.draw_id,
        certification_stream_commitment_hash=(
            design.certification_stream_commitment_hash
        ),
        joint_fqr_family_registration_hash=(
            design.joint_fqr_family_registration_hash
        ),
        corrected_evidence_stream_identity_hash=(
            projection_closure.corrected_evidence_stream_identity_hash
        ),
        literal_jacobian_stream_identity_hash=(
            projection_closure.literal_jacobian_stream_identity_hash
        ),
        raw_logit_stream_identity_hash=(
            projection_closure.raw_logit_stream_identity_hash
        ),
        same_capture_and_vjp_execution_proof_hash=(
            projection_closure.same_capture_and_vjp_execution_proof_hash
        ),
        native_fiber_axis_receipt_hash=native_fiber_axis_receipt.receipt_hash,
        native_fiber_measure_hash=native_fiber_axis_receipt.measure_hash,
        fibered_work_projection_receipt_hash=(
            projection_closure.fibered_work_projection_receipt_hash
        ),
        literal_jacobian_projection_receipt_hash=(
            projection_closure.literal_jacobian_projection_receipt_hash
        ),
        packed_direction_execution_proof_hash=_identity(
            packed_direction_execution_proof_hash,
            name="packed direction execution proof hash",
        ),
        raw_logit_interval_proof_hash=_identity(
            raw_logit_interval_proof_hash,
            name="raw-logit interval proof hash",
        ),
        parent_raw_target_logit_center=parent_raw_target_logit_center,
        parent_raw_target_logit_radius=parent_raw_target_logit_radius,
        fibered_work_lower_by_endpoint=(
            projection_closure.fibered_work_lower_by_endpoint
        ),
        fibered_work_upper_by_endpoint=(
            projection_closure.fibered_work_upper_by_endpoint
        ),
        global_literal_jacobian_center_by_endpoint=(
            projection_closure.global_literal_jacobian_center_by_endpoint
        ),
        global_literal_jacobian_radius_by_endpoint=(
            projection_closure.global_literal_jacobian_radius_by_endpoint
        ),
        dosed_raw_target_logit_center_by_endpoint=(
            dosed_raw_target_logit_center_by_endpoint
        ),
        dosed_raw_target_logit_radius_by_endpoint=(
            dosed_raw_target_logit_radius_by_endpoint
        ),
        secants_execute_complete_actions=secants_execute_complete_actions,
        endpoint_order_unchanged_after_outcomes=(
            endpoint_order_unchanged_after_outcomes
        ),
        postselection_occurred=postselection_occurred,
        _issuance_token=_ISSUANCE_TOKEN,
    )


def build_v1_fibered_directional_projection_closures(
    *,
    design: V1DirectionalSecantDesign,
    native_fiber_axis_receipt: V1NativeDestinationFiberAxisReceipt,
    fibered_work_projection_request: V1FiberedWorkProjectionRequest,
    fibered_work_projection: V1FiberedWorkProjectionResult,
    literal_jacobian_projection_request: V1DirectionalActionProjectionRequest,
    literal_jacobian_acquisition: V1LadderProjectionAcquisition,
    corrected_evidence_stream_identity_hash: str,
    literal_jacobian_stream_identity_hash: str,
    same_capture_and_vjp_execution_proof_hash: str,
    postselection_occurred: bool = False,
) -> tuple[V1FiberedDirectionalProjectionClosure, ...]:
    """Build per-C-row closures from fibered-F and global literal-J projections."""

    if type(design) is not V1DirectionalSecantDesign:
        _refuse(
            V1FiberedDirectionalRecordRefusalCode.INVALID_TYPE,
            "fibered closure bridge requires the exact secant design",
        )
    if type(native_fiber_axis_receipt) is not V1NativeDestinationFiberAxisReceipt:
        _refuse(
            V1FiberedDirectionalRecordRefusalCode.INVALID_TYPE,
            "fibered closure bridge requires the exact native-fiber receipt",
        )
    if type(fibered_work_projection_request) is not V1FiberedWorkProjectionRequest:
        _refuse(
            V1FiberedDirectionalRecordRefusalCode.INVALID_TYPE,
            "fibered closure bridge requires the exact fibered-work request",
        )
    if type(fibered_work_projection) is not V1FiberedWorkProjectionResult:
        _refuse(
            V1FiberedDirectionalRecordRefusalCode.INVALID_TYPE,
            "fibered closure bridge requires the exact fibered-work result",
        )
    if (
        type(literal_jacobian_projection_request)
        is not V1DirectionalActionProjectionRequest
    ):
        _refuse(
            V1FiberedDirectionalRecordRefusalCode.INVALID_TYPE,
            "fibered closure bridge requires the exact literal-J request",
        )
    if type(literal_jacobian_acquisition) is not V1LadderProjectionAcquisition:
        _refuse(
            V1FiberedDirectionalRecordRefusalCode.INVALID_TYPE,
            "fibered closure bridge requires the exact literal-J acquisition",
        )
    design_hash = design.assert_integrity()
    native_fiber_axis_receipt.assert_integrity()
    fibered_request_hash = fibered_work_projection_request.assert_integrity()
    fibered_work_projection.receipt.assert_integrity()
    literal_jacobian_projection_request.assert_integrity()
    literal_jacobian_acquisition.closure.assert_integrity()
    literal_jacobian_acquisition.result.receipt.assert_integrity()
    corrected_stream = _identity(
        corrected_evidence_stream_identity_hash,
        name="corrected-evidence stream identity hash",
    )
    literal_stream = _identity(
        literal_jacobian_stream_identity_hash,
        name="literal-J stream identity hash",
    )
    same_vjp_proof = _identity(
        same_capture_and_vjp_execution_proof_hash,
        name="same-capture/VJP execution proof hash",
    )
    if corrected_stream == literal_stream:
        _refuse(
            V1FiberedDirectionalRecordRefusalCode.STALE_COORDINATE,
            "corrected-evidence and literal-J streams collapsed",
        )
    if postselection_occurred is not False:
        _refuse(
            V1FiberedDirectionalRecordRefusalCode.POST_SELECTION,
            "fibered closure bridge was post-selected",
        )

    fiber_receipt = fibered_work_projection.receipt
    jacobian_receipt = literal_jacobian_acquisition.result.receipt
    evidence_request = fibered_work_projection_request.projection_request
    if (
        type(evidence_request) is not V1DirectionalActionProjectionRequest
        or evidence_request.projection_role != "corrected_evidence"
        or literal_jacobian_projection_request.projection_role
        != "literal_jacobian"
        or evidence_request.raw_logit_stream_identity_hash
        != corrected_stream
        or literal_jacobian_projection_request.raw_logit_stream_identity_hash
        != literal_stream
        or evidence_request.scorer_request_identity_hash
        != jacobian_receipt.scorer_request_identity_hash
        or literal_jacobian_projection_request.scorer_request_identity_hash
        != jacobian_receipt.scorer_request_identity_hash
        or literal_jacobian_projection_request.request_hash
        != literal_jacobian_acquisition.closure.request_hash
        or evidence_request.same_capture_and_vjp_registration_hash
        != literal_jacobian_projection_request.same_capture_and_vjp_registration_hash
        or evidence_request.d_c_causal_registration_hash
        != literal_jacobian_projection_request.d_c_causal_registration_hash
        or jacobian_receipt.raw_logit_stream_identity_hash != literal_stream
        or fiber_receipt.request_hash != fibered_request_hash
        or fiber_receipt.directional_commit_eligible is not True
        or fiber_receipt.chart_identity_hash != design.chart_identity_hash
        or fiber_receipt.joint_fqr_family_registration_hash
        != design.joint_fqr_family_registration_hash
        or fiber_receipt.same_capture_and_vjp_registration_hash
        != evidence_request.same_capture_and_vjp_registration_hash
        or fiber_receipt.d_c_causal_registration_hash
        != evidence_request.d_c_causal_registration_hash
        or fiber_receipt.design_hash != design_hash
        or fiber_receipt.ladder_hash != design.ladder_hash
        or fiber_receipt.native_fiber_axis_receipt_hash
        != native_fiber_axis_receipt.receipt_hash
        or fiber_receipt.native_fiber_measure_hash
        != native_fiber_axis_receipt.measure_hash
        or fiber_receipt.physical_mask_axis_hash != design.ladder.route_axis_hash
        or fiber_receipt.ordered_score_route_axis_hash
        != jacobian_receipt.ordered_score_route_axis_hash
        or fibered_work_projection_request.route_count != design.ladder.route_count
        or fiber_receipt.endpoint_action_hashes != design.endpoint_action_hashes
    ):
        _refuse(
            V1FiberedDirectionalRecordRefusalCode.STALE_COORDINATE,
            "fibered-work projection does not match the frozen directional design",
        )
    if (
        jacobian_receipt.design_hash != design_hash
        or jacobian_receipt.ladder_hash != design.ladder_hash
        or jacobian_receipt.current_mask_hash != design.ladder.current_mask_hash
        or jacobian_receipt.route_count != design.ladder.route_count
        or jacobian_receipt.physical_mask_axis_hash != design.ladder.route_axis_hash
        or jacobian_receipt.endpoint_action_hashes != design.endpoint_action_hashes
        or jacobian_receipt.context_ids != fiber_receipt.context_ids
        or jacobian_receipt.draw_ids != fiber_receipt.draw_ids
        or jacobian_receipt.stratum_ids != fiber_receipt.stratum_ids
    ):
        _refuse(
            V1FiberedDirectionalRecordRefusalCode.STALE_COORDINATE,
            "literal-J projection does not match the frozen fibered-F coordinates",
        )

    # Resolve sealed positions once.  Repeated ``tuple.index`` calls make a
    # full-population C census quadratic in the number of contexts even though
    # the registration already guarantees global context uniqueness.
    context_positions = {
        stratum: {context_id: index for index, context_id in enumerate(axis)}
        for stratum, axis in design.certification_context_ids_by_stratum
    }
    closures: list[V1FiberedDirectionalProjectionClosure] = []
    for row, (context_id, draw_id, stratum_id) in enumerate(
        zip(
            fiber_receipt.context_ids,
            fiber_receipt.draw_ids,
            fiber_receipt.stratum_ids,
            strict=True,
        )
    ):
        positions = context_positions.get(stratum_id)
        if positions is None or context_id not in positions:
            _refuse(
                V1FiberedDirectionalRecordRefusalCode.STALE_COORDINATE,
                "fibered closure row is outside the registered C prefix",
            )
        closures.append(
            V1FiberedDirectionalProjectionClosure(
                design_hash=design_hash,
                ladder_hash=design.ladder_hash,
                chart_identity_hash=design.chart_identity_hash,
                parent_mask_hash=design.ladder.current_mask_hash,
                proposal_generation=design.ladder.proposal_generation,
                route_count=design.ladder.route_count,
                endpoint_action_hashes=design.endpoint_action_hashes,
                stratum_id=stratum_id,
                sealed_order_index=positions[context_id],
                context_id=context_id,
                draw_id=draw_id,
                certification_stream_commitment_hash=(
                    design.certification_stream_commitment_hash
                ),
                joint_fqr_family_registration_hash=(
                    design.joint_fqr_family_registration_hash
                ),
                corrected_evidence_stream_identity_hash=corrected_stream,
                literal_jacobian_stream_identity_hash=literal_stream,
                raw_logit_stream_identity_hash=design.ladder.raw_logit_stream_source_hash,
                same_capture_and_vjp_execution_proof_hash=same_vjp_proof,
                native_fiber_axis_receipt_hash=(
                    native_fiber_axis_receipt.receipt_hash
                ),
                native_fiber_measure_hash=native_fiber_axis_receipt.measure_hash,
                fibered_work_projection_receipt_hash=fiber_receipt.receipt_hash,
                literal_jacobian_projection_receipt_hash=(
                    jacobian_receipt.receipt_hash
                ),
                fibered_work_lower_by_endpoint=(
                    fibered_work_projection.numerical_lower_matrix[row]
                ),
                fibered_work_upper_by_endpoint=(
                    fibered_work_projection.numerical_upper_matrix[row]
                ),
                global_literal_jacobian_center_by_endpoint=(
                    literal_jacobian_acquisition.result.signed_projection_matrix[row]
                ),
                global_literal_jacobian_radius_by_endpoint=(
                    literal_jacobian_acquisition.result.numerical_radius_matrix[row]
                ),
            )
        )
    return tuple(closures)


__all__ = [
    "FIBERED_DIRECTIONAL_AUTHORITY_SCOPE",
    "FIBERED_DIRECTIONAL_PROJECTION_CLOSURE_SCHEMA",
    "FIBERED_DIRECTIONAL_RECORD_SCHEMA",
    "FIBERED_FQR_OBSERVATION_EQUATION",
    "GLOBAL_LITERAL_J_PROJECTION_EQUATION",
    "NATIVE_FIBERED_WORK_PROJECTION_SCHEMA",
    "V1FiberedDirectionalProjectionClosure",
    "V1FiberedDirectionalRecord",
    "V1FiberedDirectionalRecordRefusal",
    "V1FiberedDirectionalRecordRefusalCode",
    "build_v1_fibered_directional_projection_closures",
    "close_v1_fibered_directional_record",
]
