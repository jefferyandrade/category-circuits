"""Frozen-direction secant contracts for genuine modern-v1 chart error.

For a positive removal direction ``z`` and dose ``eta`` the signed raw-logit
change is

``delta_eta = r_raw(m - eta z) - r_raw(m)``.

The canonical row is already oriented toward positive removal,
``j = -partial_u r_raw``.  It therefore predicts ``+eta * jz``.  The signed
local chart remainder is ``delta_eta - eta * jz``.  This is distinct from
the deterministic residual of a Frequent-Directions approximation to the
Jacobian Gram matrix.  The authoritative v1 doses are exactly one half and
one, measured only after a nested action ladder and C-stream design freeze.

This module owns immutable designs, observation records, and remainder-bound
receipts.  It does not own a model executor or a sequential confidence-
sequence engine and it grants no BCE or mask-commit authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral, Real
from typing import Final, Literal

import numpy as np

from .contracts import content_hash
from .modern_v1_dual_chart_geometry import (
    LITERAL_JACOBIAN_EQUATION,
    POSITIVE_REMOVAL_SIGN,
    RELU_KINK_POLICY,
)
from .modern_v1_ladder_statistics import V1FrozenCandidateLadder


DIRECTIONAL_SECANT_DOSES: Final = (0.5, 1.0)
DIRECTIONAL_SECANT_DESIGN_SCHEMA: Final = (
    "partition_free_v1_directional_secant_design_v1"
)
DIRECTIONAL_SECANT_RECORD_SCHEMA: Final = (
    "partition_free_v1_directional_secant_record_v1"
)
DIRECTIONAL_REMAINDER_INTERVAL_SCHEMA: Final = (
    "partition_free_v1_directional_remainder_interval_v1"
)
DIRECTIONAL_REMAINDER_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_directional_remainder_receipt_v1"
)
SIGNED_RAW_LOGIT_DELTA_EQUATION: Final = (
    "candidate_raw_target_logit_minus_parent_raw_target_logit_v1"
)
SIGNED_TANGENT_DELTA_EQUATION: Final = (
    "positive_removal_signed_tangent_delta_equals_plus_eta_times_jz_v1"
)
SIGNED_REMAINDER_EQUATION: Final = (
    "signed_raw_logit_delta_minus_eta_times_positive_removal_jz_v1"
)
MIDPOINT_HIGHER_ORDER_DIAGNOSTIC: Final = (
    "signed_R_eta_1_minus_four_times_signed_R_eta_half_v1"
)
DIRECTIONAL_REMAINDER_AUTHORITY_SCOPE: Final = (
    "fqr_proposal_remainder_evidence_only_no_oracle_mask_commit_or_terminal_"
    "authority_v1"
)
FULL_POPULATION_ENCLOSED_METHOD: Final = (
    "sealed_full_population_stratified_mean_of_numerically_enclosed_"
    "absolute_directional_remainder_v1"
)

_DESIGN_PREFIX: Final = "partition-free-v1-directional-secant-design-"
_RECORD_PREFIX: Final = "partition-free-v1-directional-secant-record-"
_INTERVAL_PREFIX: Final = "partition-free-v1-directional-remainder-interval-"
_RECEIPT_PREFIX: Final = "partition-free-v1-directional-remainder-receipt-"
_RECORD_AXIS_PREFIX: Final = "partition-free-v1-directional-record-axis-"
_LE_F64 = np.dtype("<f8")

RemainderIntervalMethod = Literal[
    "bounded_wor_simultaneous_upper_confidence_sequence_v1",
    "sealed_full_population_stratified_mean_of_numerically_enclosed_absolute_directional_remainder_v1",
]


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
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


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be one real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
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
        raise ValueError(f"{name} must be a float64 array") from error
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    if nonnegative and bool(np.any(array < 0.0)):
        raise ValueError(f"{name} must be nonnegative")
    array[array == 0.0] = 0.0
    result = np.frombuffer(array.tobytes(order="C"), dtype=_LE_F64).reshape(shape)
    result.setflags(write=False)
    return result


def _context_axes(
    value: object,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if type(value) not in {tuple, list}:
        raise TypeError("certification context ledger must be an ordered sequence")
    rows: list[tuple[str, tuple[str, ...]]] = []
    for raw in value:
        if type(raw) not in {tuple, list} or len(raw) != 2:
            raise ValueError("certification context rows must be name/axis pairs")
        name = _identity(raw[0], name="stratum id")
        if type(raw[1]) not in {tuple, list}:
            raise TypeError("certification context axes must be sequences")
        contexts = tuple(
            _identity(item, name="certification context id") for item in raw[1]
        )
        if not contexts or len(contexts) != len(set(contexts)):
            raise ValueError("every certification stratum must be nonempty and unique")
        rows.append((name, contexts))
    result = tuple(rows)
    if not result or len({name for name, _ in result}) != len(result):
        raise ValueError("certification strata must be nonempty and unique")
    flat = tuple(item for _, axis in result for item in axis)
    if len(flat) != len(set(flat)):
        raise ValueError("certification context IDs must be globally unique")
    return result


def _weight_axis(
    value: object,
    *,
    names: tuple[str, ...],
) -> tuple[tuple[str, float], ...]:
    if type(value) not in {tuple, list}:
        raise TypeError("stratum weights must be an ordered sequence")
    rows: list[tuple[str, float]] = []
    for raw in value:
        if type(raw) not in {tuple, list} or len(raw) != 2:
            raise ValueError("stratum weight rows must be name/value pairs")
        rows.append(
            (
                _identity(raw[0], name="stratum weight id"),
                _finite_nonnegative(raw[1], name="stratum weight"),
            )
        )
    result = tuple(rows)
    if tuple(name for name, _ in result) != names:
        raise ValueError("stratum weights differ from the certification strata")
    if any(weight <= 0.0 for _, weight in result):
        raise ValueError("stratum weights must be positive")
    total = math.fsum(weight for _, weight in result)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("stratum weights must sum to one")
    return tuple((name, weight / total) for name, weight in result)


def _ladder_coordinates(ladder: V1FrozenCandidateLadder) -> dict[str, object]:
    return {
        name: getattr(ladder, name)
        for name in (
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
        )
    }


@dataclass(frozen=True, slots=True)
class V1DirectionalSecantDesign:
    """Pre-outcome C-stream design for a frozen nested action ladder."""

    design_id: str
    ladder: V1FrozenCandidateLadder = field(repr=False, compare=False)
    chart_identity_hash: str
    certification_context_ids_by_stratum: tuple[tuple[str, tuple[str, ...]], ...]
    stratum_weights: tuple[tuple[str, float], ...]
    certification_context_population_hash: str
    certification_stream_commitment_hash: str
    packed_direction_provider_source_hash: str
    relaxed_gate_executor_source_hash: str
    raw_logit_interval_verifier_source_hash: str
    joint_fqr_family_registration_hash: str
    doses: tuple[float, float] = DIRECTIONAL_SECANT_DOSES
    positive_removal_sign: str = POSITIVE_REMOVAL_SIGN
    literal_jacobian_equation: str = LITERAL_JACOBIAN_EQUATION
    relu_kink_policy: str = RELU_KINK_POLICY
    signed_raw_logit_delta_equation: str = SIGNED_RAW_LOGIT_DELTA_EQUATION
    signed_tangent_delta_equation: str = SIGNED_TANGENT_DELTA_EQUATION
    signed_remainder_equation: str = SIGNED_REMAINDER_EQUATION
    midpoint_higher_order_diagnostic: str = MIDPOINT_HIGHER_ORDER_DIAGNOSTIC
    issued_before_secant_outcomes: bool = True
    ladder_frozen_before_secant_outcomes: bool = True
    endpoint_order_mutable_after_secant_outcomes: bool = False
    validation_outcomes_seen: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = DIRECTIONAL_REMAINDER_AUTHORITY_SCOPE
    schema: str = DIRECTIONAL_SECANT_DESIGN_SCHEMA
    _design_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "design_id", _identity(self.design_id, name="design id")
        )
        if type(self.ladder) is not V1FrozenCandidateLadder:
            raise TypeError("directional secants require the exact frozen ladder type")
        self.ladder.assert_integrity()
        for name in (
            "chart_identity_hash",
            "certification_context_population_hash",
            "certification_stream_commitment_hash",
            "packed_direction_provider_source_hash",
            "relaxed_gate_executor_source_hash",
            "raw_logit_interval_verifier_source_hash",
            "joint_fqr_family_registration_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        contexts = _context_axes(self.certification_context_ids_by_stratum)
        weights = _weight_axis(
            self.stratum_weights,
            names=tuple(name for name, _ in contexts),
        )
        doses = tuple(
            _finite_nonnegative(value, name="secant dose") for value in self.doses
        )
        if doses != DIRECTIONAL_SECANT_DOSES:
            raise ValueError("canonical directional secants require doses (0.5, 1.0)")
        expected_semantics = (
            self.positive_removal_sign == POSITIVE_REMOVAL_SIGN
            and self.literal_jacobian_equation == LITERAL_JACOBIAN_EQUATION
            and self.relu_kink_policy == RELU_KINK_POLICY
            and self.signed_raw_logit_delta_equation == SIGNED_RAW_LOGIT_DELTA_EQUATION
            and self.signed_tangent_delta_equation == SIGNED_TANGENT_DELTA_EQUATION
            and self.signed_remainder_equation == SIGNED_REMAINDER_EQUATION
            and self.midpoint_higher_order_diagnostic
            == MIDPOINT_HIGHER_ORDER_DIAGNOSTIC
        )
        if not expected_semantics:
            raise ValueError("directional secant semantics differ from canonical v1")
        if (
            self.issued_before_secant_outcomes is not True
            or self.ladder_frozen_before_secant_outcomes is not True
            or self.endpoint_order_mutable_after_secant_outcomes is not False
            or self.validation_outcomes_seen is not False
        ):
            raise PermissionError(
                "directional design was post-selected or saw validation"
            )
        if any(
            getattr(self, name) is not False
            for name in (
                "bce_oracle_authority",
                "mask_commit_authority",
                "terminal_authority",
            )
        ):
            raise PermissionError("directional design cannot grant action authority")
        if self.authority_scope != DIRECTIONAL_REMAINDER_AUTHORITY_SCOPE:
            raise ValueError("directional design authority scope is invalid")
        if self.schema != DIRECTIONAL_SECANT_DESIGN_SCHEMA:
            raise ValueError("directional secant design schema is invalid")
        object.__setattr__(self, "certification_context_ids_by_stratum", contexts)
        object.__setattr__(self, "stratum_weights", weights)
        object.__setattr__(self, "doses", doses)
        self.assert_integrity()

    @property
    def endpoint_action_hashes(self) -> tuple[str, ...]:
        return self.ladder.action_hashes

    @property
    def ladder_hash(self) -> str:
        return self.ladder.ladder_hash

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "design_id": self.design_id,
            **_ladder_coordinates(self.ladder),
            "ladder_hash": self.ladder_hash,
            "endpoint_action_hashes": self.endpoint_action_hashes,
            "chart_identity_hash": self.chart_identity_hash,
            "certification_context_ids_by_stratum": self.certification_context_ids_by_stratum,
            "stratum_weights": self.stratum_weights,
            "certification_context_population_hash": self.certification_context_population_hash,
            "certification_stream_commitment_hash": self.certification_stream_commitment_hash,
            "packed_direction_provider_source_hash": self.packed_direction_provider_source_hash,
            "relaxed_gate_executor_source_hash": self.relaxed_gate_executor_source_hash,
            "raw_logit_interval_verifier_source_hash": self.raw_logit_interval_verifier_source_hash,
            "joint_fqr_family_registration_hash": self.joint_fqr_family_registration_hash,
            "doses": self.doses,
            "positive_removal_sign": self.positive_removal_sign,
            "literal_jacobian_equation": self.literal_jacobian_equation,
            "relu_kink_policy": self.relu_kink_policy,
            "signed_raw_logit_delta_equation": self.signed_raw_logit_delta_equation,
            "signed_tangent_delta_equation": self.signed_tangent_delta_equation,
            "signed_remainder_equation": self.signed_remainder_equation,
            "midpoint_higher_order_diagnostic": self.midpoint_higher_order_diagnostic,
            "issued_before_secant_outcomes": self.issued_before_secant_outcomes,
            "ladder_frozen_before_secant_outcomes": self.ladder_frozen_before_secant_outcomes,
            "endpoint_order_mutable_after_secant_outcomes": self.endpoint_order_mutable_after_secant_outcomes,
            "validation_outcomes_seen": self.validation_outcomes_seen,
            "bce_oracle_authority": self.bce_oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "authority_scope": self.authority_scope,
        }

    @property
    def design_hash(self) -> str:
        cached = self._design_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.ladder.assert_integrity()
        observed = content_hash(self._payload(), prefix=_DESIGN_PREFIX)
        cached = self._design_hash
        if cached is not None and cached != observed:
            raise ValueError("directional secant design was tampered after hashing")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("directional secant design hash does not match")
        object.__setattr__(self, "_design_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1DirectionalSecantRecord:
    """One C-stream observation across every frozen endpoint and dose."""

    design_hash: str
    ladder_hash: str
    chart_identity_hash: str
    endpoint_action_hashes: tuple[str, ...]
    stratum_id: str
    sealed_order_index: int
    context_id: str
    draw_id: str
    parent_raw_target_logit_center: float
    parent_raw_target_logit_radius: float
    corrected_projection_center_by_endpoint: np.ndarray = field(
        repr=False, compare=False
    )
    corrected_projection_radius_by_endpoint: np.ndarray = field(
        repr=False, compare=False
    )
    jacobian_projection_center_by_endpoint: np.ndarray = field(
        repr=False, compare=False
    )
    jacobian_projection_radius_by_endpoint: np.ndarray = field(
        repr=False, compare=False
    )
    dosed_raw_target_logit_center_by_endpoint: np.ndarray = field(
        repr=False, compare=False
    )
    dosed_raw_target_logit_radius_by_endpoint: np.ndarray = field(
        repr=False, compare=False
    )
    dual_chart_projection_proof_hash: str
    packed_direction_execution_proof_hash: str
    raw_logit_interval_proof_hash: str
    doses: tuple[float, float] = DIRECTIONAL_SECANT_DOSES
    positive_removal_sign: str = POSITIVE_REMOVAL_SIGN
    signed_remainder_equation: str = SIGNED_REMAINDER_EQUATION
    terminal_validation_seen: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    schema: str = DIRECTIONAL_SECANT_RECORD_SCHEMA
    signed_raw_logit_delta_by_endpoint: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    signed_remainder_center_by_endpoint: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    signed_remainder_radius_by_endpoint: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    f_upper_observation_by_endpoint: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    q_upper_observation_by_endpoint: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    r_upper_observation_by_endpoint: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    midpoint_higher_order_center_by_endpoint: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    midpoint_higher_order_radius_by_endpoint: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    _record_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "design_hash",
            "ladder_hash",
            "chart_identity_hash",
            "stratum_id",
            "context_id",
            "draw_id",
            "dual_chart_projection_proof_hash",
            "packed_direction_execution_proof_hash",
            "raw_logit_interval_proof_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        endpoint_hashes = tuple(
            _identity(value, name="endpoint action hash")
            for value in self.endpoint_action_hashes
        )
        if not endpoint_hashes or len(endpoint_hashes) != len(set(endpoint_hashes)):
            raise ValueError(
                "secant record endpoint hashes must be nonempty and unique"
            )
        endpoint_count = len(endpoint_hashes)
        index = _nonnegative_int(self.sealed_order_index, name="sealed order index")
        parent = _finite(self.parent_raw_target_logit_center, name="parent raw logit")
        parent_radius = _finite_nonnegative(
            self.parent_raw_target_logit_radius,
            name="parent raw-logit radius",
        )
        corrected = _immutable_f64(
            self.corrected_projection_center_by_endpoint,
            shape=(endpoint_count,),
            name="corrected projection centers",
        )
        corrected_radius = _immutable_f64(
            self.corrected_projection_radius_by_endpoint,
            shape=(endpoint_count,),
            name="corrected projection radii",
            nonnegative=True,
        )
        jacobian = _immutable_f64(
            self.jacobian_projection_center_by_endpoint,
            shape=(endpoint_count,),
            name="Jacobian projection centers",
        )
        jacobian_radius = _immutable_f64(
            self.jacobian_projection_radius_by_endpoint,
            shape=(endpoint_count,),
            name="Jacobian projection radii",
            nonnegative=True,
        )
        dosed = _immutable_f64(
            self.dosed_raw_target_logit_center_by_endpoint,
            shape=(endpoint_count, 2),
            name="dosed raw-logit centers",
        )
        dosed_radius = _immutable_f64(
            self.dosed_raw_target_logit_radius_by_endpoint,
            shape=(endpoint_count, 2),
            name="dosed raw-logit radii",
            nonnegative=True,
        )
        doses = tuple(
            _finite_nonnegative(value, name="secant dose") for value in self.doses
        )
        if doses != DIRECTIONAL_SECANT_DOSES:
            raise ValueError("secant records require doses (0.5, 1.0)")
        if self.positive_removal_sign != POSITIVE_REMOVAL_SIGN:
            raise ValueError("secant record removal orientation is invalid")
        if self.signed_remainder_equation != SIGNED_REMAINDER_EQUATION:
            raise ValueError("secant record remainder equation is invalid")
        if self.terminal_validation_seen is not False:
            raise PermissionError("secant records cannot observe terminal validation")
        if (
            self.bce_oracle_authority is not False
            or self.mask_commit_authority is not False
        ):
            raise PermissionError("secant records cannot grant BCE or commit authority")
        if self.schema != DIRECTIONAL_SECANT_RECORD_SCHEMA:
            raise ValueError("directional secant record schema is invalid")

        delta = dosed - parent
        remainder = delta - jacobian[:, None] * np.asarray(doses)[None, :]
        remainder_radius = (
            dosed_radius
            + parent_radius
            + jacobian_radius[:, None] * np.asarray(doses)[None, :]
        )
        f_upper = np.abs(corrected) + corrected_radius
        q_upper = np.square(np.abs(jacobian) + jacobian_radius)
        r_upper = np.abs(remainder) + remainder_radius
        higher_center = remainder[:, 1] - 4.0 * remainder[:, 0]
        higher_radius = remainder_radius[:, 1] + 4.0 * remainder_radius[:, 0]
        derived = (
            delta,
            remainder,
            remainder_radius,
            f_upper,
            q_upper,
            r_upper,
            higher_center,
            higher_radius,
        )
        if any(not np.isfinite(value).all() for value in derived):
            raise ValueError("derived directional secant observations are nonfinite")

        object.__setattr__(self, "endpoint_action_hashes", endpoint_hashes)
        object.__setattr__(self, "sealed_order_index", index)
        object.__setattr__(self, "parent_raw_target_logit_center", parent)
        object.__setattr__(self, "parent_raw_target_logit_radius", parent_radius)
        object.__setattr__(self, "corrected_projection_center_by_endpoint", corrected)
        object.__setattr__(
            self, "corrected_projection_radius_by_endpoint", corrected_radius
        )
        object.__setattr__(self, "jacobian_projection_center_by_endpoint", jacobian)
        object.__setattr__(
            self, "jacobian_projection_radius_by_endpoint", jacobian_radius
        )
        object.__setattr__(self, "dosed_raw_target_logit_center_by_endpoint", dosed)
        object.__setattr__(
            self, "dosed_raw_target_logit_radius_by_endpoint", dosed_radius
        )
        object.__setattr__(self, "doses", doses)
        for name, value in zip(
            (
                "signed_raw_logit_delta_by_endpoint",
                "signed_remainder_center_by_endpoint",
                "signed_remainder_radius_by_endpoint",
                "f_upper_observation_by_endpoint",
                "q_upper_observation_by_endpoint",
                "r_upper_observation_by_endpoint",
                "midpoint_higher_order_center_by_endpoint",
                "midpoint_higher_order_radius_by_endpoint",
            ),
            derived,
            strict=True,
        ):
            object.__setattr__(
                self,
                name,
                _immutable_f64(value, shape=value.shape, name=name),
            )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "design_hash": self.design_hash,
            "ladder_hash": self.ladder_hash,
            "chart_identity_hash": self.chart_identity_hash,
            "endpoint_action_hashes": self.endpoint_action_hashes,
            "stratum_id": self.stratum_id,
            "sealed_order_index": self.sealed_order_index,
            "context_id": self.context_id,
            "draw_id": self.draw_id,
            "parent_raw_target_logit_center": self.parent_raw_target_logit_center,
            "parent_raw_target_logit_radius": self.parent_raw_target_logit_radius,
            "corrected_projection_center_by_endpoint": self.corrected_projection_center_by_endpoint,
            "corrected_projection_radius_by_endpoint": self.corrected_projection_radius_by_endpoint,
            "jacobian_projection_center_by_endpoint": self.jacobian_projection_center_by_endpoint,
            "jacobian_projection_radius_by_endpoint": self.jacobian_projection_radius_by_endpoint,
            "dosed_raw_target_logit_center_by_endpoint": self.dosed_raw_target_logit_center_by_endpoint,
            "dosed_raw_target_logit_radius_by_endpoint": self.dosed_raw_target_logit_radius_by_endpoint,
            "signed_raw_logit_delta_by_endpoint": self.signed_raw_logit_delta_by_endpoint,
            "signed_remainder_center_by_endpoint": self.signed_remainder_center_by_endpoint,
            "signed_remainder_radius_by_endpoint": self.signed_remainder_radius_by_endpoint,
            "f_upper_observation_by_endpoint": self.f_upper_observation_by_endpoint,
            "q_upper_observation_by_endpoint": self.q_upper_observation_by_endpoint,
            "r_upper_observation_by_endpoint": self.r_upper_observation_by_endpoint,
            "midpoint_higher_order_center_by_endpoint": self.midpoint_higher_order_center_by_endpoint,
            "midpoint_higher_order_radius_by_endpoint": self.midpoint_higher_order_radius_by_endpoint,
            "dual_chart_projection_proof_hash": self.dual_chart_projection_proof_hash,
            "packed_direction_execution_proof_hash": self.packed_direction_execution_proof_hash,
            "raw_logit_interval_proof_hash": self.raw_logit_interval_proof_hash,
            "doses": self.doses,
            "positive_removal_sign": self.positive_removal_sign,
            "signed_remainder_equation": self.signed_remainder_equation,
            "terminal_validation_seen": self.terminal_validation_seen,
            "bce_oracle_authority": self.bce_oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
        }

    @property
    def record_hash(self) -> str:
        cached = self._record_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_RECORD_PREFIX)
        cached = self._record_hash
        if cached is not None and cached != observed:
            raise ValueError("directional secant record was tampered after hashing")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("directional secant record hash does not match")
        object.__setattr__(self, "_record_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1DirectionalRemainderInterval:
    """One simultaneous upper confidence bound for ``E|R_eta(z)|``."""

    endpoint_index: int
    action_hash: str
    dose: float
    sample_size: int
    sample_mean: float
    lower: float
    upper: float
    value_upper_bound: float
    alpha: float
    method: RemainderIntervalMethod
    record_axis_hash: str
    empty: bool = False
    schema: str = DIRECTIONAL_REMAINDER_INTERVAL_SCHEMA
    _interval_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        endpoint = _positive_int(self.endpoint_index, name="endpoint index")
        action = _identity(self.action_hash, name="action hash")
        dose = _finite_nonnegative(self.dose, name="dose")
        if dose not in DIRECTIONAL_SECANT_DOSES:
            raise ValueError("remainder intervals require dose 0.5 or 1.0")
        sample_size = _positive_int(self.sample_size, name="sample size")
        sample_mean = _finite_nonnegative(self.sample_mean, name="sample mean")
        lower = _finite_nonnegative(self.lower, name="remainder lower")
        upper = _finite_nonnegative(self.upper, name="remainder upper")
        value_bound = _finite_nonnegative(
            self.value_upper_bound,
            name="remainder value upper bound",
        )
        alpha = _finite_nonnegative(self.alpha, name="remainder alpha")
        if alpha >= 1.0:
            raise ValueError("remainder interval alpha must be below one")
        if upper > value_bound or sample_mean > value_bound:
            raise ValueError("remainder interval exceeds its analytic value bound")
        empty = bool(lower > upper)
        if self.empty is not empty:
            raise ValueError("remainder interval empty flag is inconsistent")
        if self.method not in {
            "bounded_wor_simultaneous_upper_confidence_sequence_v1",
            FULL_POPULATION_ENCLOSED_METHOD,
        }:
            raise ValueError("unknown directional remainder interval method")
        if self.method == FULL_POPULATION_ENCLOSED_METHOD and alpha != 0.0:
            raise ValueError("a full-population remainder bound spends no alpha")
        _identity(self.record_axis_hash, name="record axis hash")
        if self.schema != DIRECTIONAL_REMAINDER_INTERVAL_SCHEMA:
            raise ValueError("directional remainder interval schema is invalid")
        object.__setattr__(self, "endpoint_index", endpoint)
        object.__setattr__(self, "action_hash", action)
        object.__setattr__(self, "dose", dose)
        object.__setattr__(self, "sample_size", sample_size)
        object.__setattr__(self, "sample_mean", sample_mean)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "value_upper_bound", value_bound)
        object.__setattr__(self, "alpha", alpha)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "endpoint_index",
                "action_hash",
                "dose",
                "sample_size",
                "sample_mean",
                "lower",
                "upper",
                "value_upper_bound",
                "alpha",
                "method",
                "record_axis_hash",
                "empty",
            )
        }

    @property
    def interval_hash(self) -> str:
        cached = self._interval_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_INTERVAL_PREFIX)
        cached = self._interval_hash
        if cached is not None and cached != observed:
            raise ValueError("directional remainder interval was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("directional remainder interval hash does not match")
        object.__setattr__(self, "_interval_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1DirectionalRemainderReceipt:
    """Source-bound simultaneous remainder bounds for every frozen endpoint."""

    design_hash: str
    ladder_hash: str
    chart_identity_hash: str
    endpoint_action_hashes: tuple[str, ...]
    certification_stream_commitment_hash: str
    joint_fqr_family_registration_hash: str
    records: tuple[V1DirectionalSecantRecord, ...] = field(repr=False, compare=False)
    endpoint_intervals: tuple[V1DirectionalRemainderInterval, ...]
    record_axis_hash: str
    statistical_proof_hash: str
    endpoint_order_unchanged_after_outcomes: bool = True
    postselection_occurred: bool = False
    proposal_authority: bool = True
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = DIRECTIONAL_REMAINDER_AUTHORITY_SCOPE
    schema: str = DIRECTIONAL_REMAINDER_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "design_hash",
            "ladder_hash",
            "chart_identity_hash",
            "certification_stream_commitment_hash",
            "joint_fqr_family_registration_hash",
            "record_axis_hash",
            "statistical_proof_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        endpoint_hashes = tuple(
            _identity(value, name="endpoint action hash")
            for value in self.endpoint_action_hashes
        )
        if not endpoint_hashes or len(endpoint_hashes) != len(set(endpoint_hashes)):
            raise ValueError(
                "remainder receipt endpoint axis must be nonempty and unique"
            )
        records = tuple(self.records)
        if not records or any(
            type(record) is not V1DirectionalSecantRecord for record in records
        ):
            raise TypeError("remainder receipt requires typed secant records")
        record_hashes: list[str] = []
        context_draw_pairs: set[tuple[str, str]] = set()
        for record in records:
            record_hashes.append(record.assert_integrity())
            if (
                record.design_hash != self.design_hash
                or record.ladder_hash != self.ladder_hash
                or record.chart_identity_hash != self.chart_identity_hash
                or record.endpoint_action_hashes != endpoint_hashes
            ):
                raise ValueError("secant record was recontextualized in a receipt")
            pair = (record.context_id, record.draw_id)
            if pair in context_draw_pairs:
                raise ValueError("remainder receipt repeats a context/draw row")
            context_draw_pairs.add(pair)
        expected_axis_hash = content_hash(
            tuple(record_hashes), prefix=_RECORD_AXIS_PREFIX
        )
        if self.record_axis_hash != expected_axis_hash:
            raise ValueError(
                "remainder receipt record-axis hash differs from its records"
            )

        intervals = tuple(self.endpoint_intervals)
        expected_cells = tuple(
            (endpoint, action, dose)
            for endpoint, action in enumerate(endpoint_hashes, start=1)
            for dose in DIRECTIONAL_SECANT_DOSES
        )
        observed_cells: list[tuple[int, str, float]] = []
        for interval in intervals:
            if type(interval) is not V1DirectionalRemainderInterval:
                raise TypeError(
                    "remainder receipt intervals must use the typed interval"
                )
            interval.assert_integrity()
            observed_cells.append(
                (interval.endpoint_index, interval.action_hash, interval.dose)
            )
            if interval.record_axis_hash != self.record_axis_hash:
                raise ValueError("remainder interval names another record axis")
            if interval.sample_size != len(records):
                raise ValueError(
                    "remainder interval sample size differs from its record axis"
                )
        if tuple(observed_cells) != expected_cells:
            raise ValueError("remainder receipt must exactly cover endpoint/dose cells")
        if (
            self.endpoint_order_unchanged_after_outcomes is not True
            or self.postselection_occurred is not False
        ):
            raise PermissionError("directional remainder endpoints were post-selected")
        if self.proposal_authority is not True:
            raise ValueError("remainder receipt must retain proposal authority")
        if any(
            getattr(self, name) is not False
            for name in (
                "bce_oracle_authority",
                "mask_commit_authority",
                "terminal_authority",
            )
        ):
            raise PermissionError("remainder receipt cannot grant action authority")
        if self.authority_scope != DIRECTIONAL_REMAINDER_AUTHORITY_SCOPE:
            raise ValueError("remainder receipt authority scope is invalid")
        if self.schema != DIRECTIONAL_REMAINDER_RECEIPT_SCHEMA:
            raise ValueError("directional remainder receipt schema is invalid")
        object.__setattr__(self, "endpoint_action_hashes", endpoint_hashes)
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "endpoint_intervals", intervals)
        self.assert_integrity()

    @property
    def interval_upper_by_endpoint_and_dose(self) -> np.ndarray:
        values = np.asarray(
            [interval.upper for interval in self.endpoint_intervals],
            dtype=np.float64,
        ).reshape(len(self.endpoint_action_hashes), 2)
        return _immutable_f64(
            values,
            shape=values.shape,
            name="remainder interval upper matrix",
        )

    @property
    def empty(self) -> bool:
        return any(interval.empty for interval in self.endpoint_intervals)

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "design_hash": self.design_hash,
            "ladder_hash": self.ladder_hash,
            "chart_identity_hash": self.chart_identity_hash,
            "endpoint_action_hashes": self.endpoint_action_hashes,
            "certification_stream_commitment_hash": self.certification_stream_commitment_hash,
            "joint_fqr_family_registration_hash": self.joint_fqr_family_registration_hash,
            "record_hashes": tuple(record.record_hash for record in self.records),
            "endpoint_interval_hashes": tuple(
                interval.interval_hash for interval in self.endpoint_intervals
            ),
            "record_axis_hash": self.record_axis_hash,
            "statistical_proof_hash": self.statistical_proof_hash,
            "endpoint_order_unchanged_after_outcomes": self.endpoint_order_unchanged_after_outcomes,
            "postselection_occurred": self.postselection_occurred,
            "proposal_authority": self.proposal_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "authority_scope": self.authority_scope,
        }

    @property
    def receipt_hash(self) -> str:
        cached = self._receipt_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        for record in self.records:
            record.assert_integrity()
        for interval in self.endpoint_intervals:
            interval.assert_integrity()
        observed = content_hash(self._payload(), prefix=_RECEIPT_PREFIX)
        cached = self._receipt_hash
        if cached is not None and cached != observed:
            raise ValueError("directional remainder receipt was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("directional remainder receipt hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


def build_full_population_directional_remainder_receipt(
    *,
    design: V1DirectionalSecantDesign,
    records: tuple[V1DirectionalSecantRecord, ...],
    value_upper_bound_by_endpoint_and_dose: object,
    statistical_proof_hash: str,
) -> V1DirectionalRemainderReceipt:
    """Build a zero-alpha stratified census receipt from sealed C records.

    Numerical radii remain conservative: the receipt averages each record's
    upper observation rather than relabeling the computation as point-exact.
    """

    if type(design) is not V1DirectionalSecantDesign:
        raise TypeError("remainder census requires a typed secant design")
    design_hash = design.assert_integrity()
    rows = tuple(records)
    expected_contexts = dict(design.certification_context_ids_by_stratum)
    weights = dict(design.stratum_weights)
    by_stratum: dict[str, list[V1DirectionalSecantRecord]] = {
        name: [] for name in expected_contexts
    }
    seen_draws: set[str] = set()
    for record in rows:
        if type(record) is not V1DirectionalSecantRecord:
            raise TypeError("remainder census rows must use the typed secant record")
        record.assert_integrity()
        if (
            record.design_hash != design_hash
            or record.ladder_hash != design.ladder_hash
            or record.chart_identity_hash != design.chart_identity_hash
            or record.endpoint_action_hashes != design.endpoint_action_hashes
            or record.stratum_id not in by_stratum
        ):
            raise ValueError("remainder census record differs from its frozen design")
        axis = expected_contexts[record.stratum_id]
        expected_position = len(by_stratum[record.stratum_id])
        if (
            record.sealed_order_index != expected_position
            or expected_position >= len(axis)
            or record.context_id != axis[expected_position]
        ):
            raise ValueError("remainder census records are not a sealed stratum prefix")
        if record.draw_id in seen_draws:
            raise ValueError("remainder census repeats a draw identity")
        seen_draws.add(record.draw_id)
        by_stratum[record.stratum_id].append(record)
    if any(
        tuple(record.context_id for record in by_stratum[name]) != axis
        for name, axis in expected_contexts.items()
    ):
        raise ValueError(
            "full-population remainder receipt requires the complete C census"
        )
    record_hashes = tuple(record.record_hash for record in rows)
    record_axis_hash = content_hash(record_hashes, prefix=_RECORD_AXIS_PREFIX)
    endpoint_count = len(design.endpoint_action_hashes)
    bound = _immutable_f64(
        value_upper_bound_by_endpoint_and_dose,
        shape=(endpoint_count, 2),
        name="remainder value upper bounds",
        nonnegative=True,
    )
    # Reference census construction is deliberately bounded-memory.  ``fsum``
    # avoids overflow in an intermediate ``n * value`` reduction, and the
    # final nextafter makes the stored binary64 mean an outward upper
    # enclosure.  GPU production acquisition may use a certified fixed-tree
    # reducer, but it must issue the same interval contract.
    mean = np.zeros((endpoint_count, 2), dtype=np.float64)
    for endpoint in range(endpoint_count):
        for dose_index in range(2):
            weighted_stratum_means = []
            for stratum, stratum_records in by_stratum.items():
                subtotal = math.fsum(
                    float(record.r_upper_observation_by_endpoint[endpoint, dose_index])
                    for record in stratum_records
                )
                weighted_stratum_means.append(
                    weights[stratum] * (subtotal / len(stratum_records))
                )
            value = math.fsum(weighted_stratum_means)
            mean[endpoint, dose_index] = (
                math.nextafter(value, math.inf) if value > 0.0 else 0.0
            )
    if np.any(mean > bound):
        raise ValueError("observed remainder upper value exceeds its analytic bound")
    intervals = tuple(
        V1DirectionalRemainderInterval(
            endpoint_index=endpoint,
            action_hash=action,
            dose=dose,
            sample_size=len(rows),
            sample_mean=float(mean[endpoint - 1, dose_index]),
            lower=0.0,
            upper=float(mean[endpoint - 1, dose_index]),
            value_upper_bound=float(bound[endpoint - 1, dose_index]),
            alpha=0.0,
            method=FULL_POPULATION_ENCLOSED_METHOD,
            record_axis_hash=record_axis_hash,
        )
        for endpoint, action in enumerate(design.endpoint_action_hashes, start=1)
        for dose_index, dose in enumerate(DIRECTIONAL_SECANT_DOSES)
    )
    return V1DirectionalRemainderReceipt(
        design_hash=design_hash,
        ladder_hash=design.ladder_hash,
        chart_identity_hash=design.chart_identity_hash,
        endpoint_action_hashes=design.endpoint_action_hashes,
        certification_stream_commitment_hash=design.certification_stream_commitment_hash,
        joint_fqr_family_registration_hash=design.joint_fqr_family_registration_hash,
        records=rows,
        endpoint_intervals=intervals,
        record_axis_hash=record_axis_hash,
        statistical_proof_hash=_identity(
            statistical_proof_hash,
            name="statistical proof hash",
        ),
    )


def verify_directional_remainder_receipt(
    *,
    design: V1DirectionalSecantDesign,
    receipt: V1DirectionalRemainderReceipt,
    expected_receipt_hash: str | None = None,
) -> str:
    if (
        type(design) is not V1DirectionalSecantDesign
        or type(receipt) is not V1DirectionalRemainderReceipt
    ):
        raise TypeError("remainder verification requires typed design and receipt")
    design_hash = design.assert_integrity()
    receipt_hash = receipt.assert_integrity(expected_receipt_hash)
    if (
        receipt.design_hash != design_hash
        or receipt.ladder_hash != design.ladder_hash
        or receipt.chart_identity_hash != design.chart_identity_hash
        or receipt.endpoint_action_hashes != design.endpoint_action_hashes
        or receipt.certification_stream_commitment_hash
        != design.certification_stream_commitment_hash
        or receipt.joint_fqr_family_registration_hash
        != design.joint_fqr_family_registration_hash
    ):
        raise ValueError("directional remainder receipt differs from its design")
    expected_contexts = dict(design.certification_context_ids_by_stratum)
    observed_count = {name: 0 for name in expected_contexts}
    seen_draws: set[str] = set()
    for record in receipt.records:
        if record.stratum_id not in observed_count:
            raise ValueError("remainder receipt names an unregistered stratum")
        position = observed_count[record.stratum_id]
        axis = expected_contexts[record.stratum_id]
        if (
            position >= len(axis)
            or record.sealed_order_index != position
            or record.context_id != axis[position]
        ):
            raise ValueError("remainder records are not sealed stratum prefixes")
        if record.draw_id in seen_draws:
            raise ValueError("remainder receipt repeats a draw identity")
        seen_draws.add(record.draw_id)
        observed_count[record.stratum_id] += 1
    return receipt_hash


__all__ = [
    "DIRECTIONAL_REMAINDER_AUTHORITY_SCOPE",
    "DIRECTIONAL_REMAINDER_RECEIPT_SCHEMA",
    "DIRECTIONAL_SECANT_DESIGN_SCHEMA",
    "DIRECTIONAL_SECANT_DOSES",
    "DIRECTIONAL_SECANT_RECORD_SCHEMA",
    "FULL_POPULATION_ENCLOSED_METHOD",
    "MIDPOINT_HIGHER_ORDER_DIAGNOSTIC",
    "SIGNED_RAW_LOGIT_DELTA_EQUATION",
    "SIGNED_REMAINDER_EQUATION",
    "SIGNED_TANGENT_DELTA_EQUATION",
    "V1DirectionalRemainderInterval",
    "V1DirectionalRemainderReceipt",
    "V1DirectionalSecantDesign",
    "V1DirectionalSecantRecord",
    "build_full_population_directional_remainder_receipt",
    "verify_directional_remainder_receipt",
]
