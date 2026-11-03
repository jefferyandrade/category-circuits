"""Pre-C analytic F/Q/R bounds for frozen AlexNet action ladders.

This module gives the three canonical local objects separate numerical owners:

* corrected work ``F`` reuses the conserving Aumann--Shapley action envelope;
* response energy ``Q`` is bounded from the literal relaxed-gate Jacobian
  ``-G 1[a>0] c`` at the parent chart; and
* directional remainder ``R`` is bounded from parent/relaxed raw-logit
  intervals and that literal-J projection.

The public builder accepts no C-stream observation or downstream outcome.  It
binds one exact packed, pre-C ladder at doses one half and one and returns a
ready :class:`V1FQRBranchAnalyticInput`.  All arithmetic bounds are finite,
binary64, and outward enclosed.  The legacy hard-mask AlexNet envelope is only
consumed; its binary semantics are not reinterpreted or modified here.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Final

import numpy as np

from mechanistic_systems.circuits.irg import modern_v1_alexnet_envelope as _env
from mechanistic_systems.circuits.irg.modern_v1_alexnet_envelope import (
    V1AlexNetCorrectedAttributionEnvelopeResult,
    V1AlexNetRawTargetLogitIntervalCertificate,
    V1RegisteredNormalizedInputBox,
    admit_v1_alexnet_envelope_layout,
    build_v1_alexnet_action_drive_endpoint_envelope,
)
from mechanistic_systems.circuits.irg.modern_v1_directional_remainder import (
    DIRECTIONAL_SECANT_DOSES,
    SIGNED_RAW_LOGIT_DELTA_EQUATION,
    SIGNED_REMAINDER_EQUATION,
    SIGNED_TANGENT_DELTA_EQUATION,
    V1DirectionalSecantDesign,
)
from mechanistic_systems.circuits.irg.modern_v1_dual_chart_geometry import (
    CORRECTED_EVIDENCE_EQUATION,
    LITERAL_JACOBIAN_EQUATION,
    POSITIVE_REMOVAL_SIGN,
    RELU_KINK_POLICY,
)
from mechanistic_systems.circuits.irg.modern_v1_fqr_branch_design import (
    V1FQRBranchAnalyticInput,
)
from mechanistic_systems.circuits.irg.modern_v1_ladder_statistics import (
    V1FrozenCandidateLadder,
    V1FrozenLadderEndpointEnvelopeCertificate,
)
from mechanistic_systems.circuits.irg.modern_v1_neutral_runtime import (
    DeclaredRouteAxis,
)
from mechanistic_systems.circuits.irg.modern_v1_relaxed_kkt import (
    V1RelaxedKKTBranch,
)
from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask
from mechanistic_systems.contracts.science import QuerySpec
from mechanistic_systems.measurement.models import state_dict_digest
from mechanistic_systems.measurement.modern_v1_directional_secant_provider import (
    RawLogitIntervalVerifier,
    V1DirectionalSecantObservationBounds,
)

ALEXNET_FQR_RELAXED_LOGIT_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_alexnet_fqr_relaxed_logit_interval_receipt_v1"
)
ALEXNET_FQR_BOUND_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_alexnet_fqr_bound_derivation_receipt_v1"
)
ALEXNET_FQR_BOUND_RECEIPT_SCHEMA_V2: Final = (
    "partition_free_v1_alexnet_fqr_bound_derivation_receipt_v2"
)
ALEXNET_FQR_BOUND_RESULT_SCHEMA: Final = "partition_free_v1_alexnet_fqr_branch_analytic_bounds_v1"
ALEXNET_FQR_BOUND_ALGORITHM_ID: Final = (
    "alexnet_corrected_work_literal_jacobian_relaxed_secant_bound_v1"
)
ALEXNET_FQR_LITERAL_J_ALGORITHM_ID: Final = (
    "alexnet_parent_chart_strict_relu_literal_jacobian_action_drive_bound_v1"
)
ALEXNET_FQR_REMAINDER_ALGORITHM_ID: Final = (
    "alexnet_parent_relaxed_raw_logit_interval_minus_literal_j_projection_v1"
)
ALEXNET_FQR_RELAXED_GATE_PATH_ID: Final = (
    "parent_gate_minus_registered_dose_times_packed_removal_direction_v1"
)
ALEXNET_FQR_NUMERICAL_ENCLOSURE_ID: Final = (
    "outward_binary64_nextafter_for_F_J_square_and_secant_interval_v1"
)
ALEXNET_FQR_AUTHORITY_SCOPE: Final = (
    "pre_C_analytic_FQR_observation_bounds_only_no_C_K_O_commit_terminal_or_pareto_authority_v1"
)

EXACT_LINEAR_PROOF_NONE: Final = "no_exact_linear_remainder_proof_v1"
EXACT_LINEAR_PROOF_ZERO_SELECTED_WEIGHT: Final = "all_selected_live_route_weights_are_exact_zero_v1"
_EXACT_LINEAR_PROOFS: Final = frozenset(
    {
        EXACT_LINEAR_PROOF_NONE,
        EXACT_LINEAR_PROOF_ZERO_SELECTED_WEIGHT,
    }
)

_RELAXED_RECEIPT_PREFIX: Final = "partition-free-v1-alexnet-fqr-relaxed-logit-"
_BOUND_RECEIPT_PREFIX: Final = "partition-free-v1-alexnet-fqr-bound-receipt-"
_F_SOURCE_PREFIX: Final = "partition-free-v1-alexnet-fqr-F-bound-source-"
_Q_SOURCE_PREFIX: Final = "partition-free-v1-alexnet-fqr-Q-bound-source-"
_R_SOURCE_PREFIX: Final = "partition-free-v1-alexnet-fqr-R-bound-source-"
_RELAXED_GATE_PREFIX: Final = "partition-free-v1-alexnet-relaxed-gate-path-"
_OBSERVATION_BOUND_ADAPTER_PREFIX: Final = (
    "partition-free-v1-alexnet-fqr-directional-observation-bound-adapter-"
)
_RAW_LOGIT_VERIFIER_PROOF_PREFIX: Final = (
    "partition-free-v1-alexnet-fqr-raw-logit-center-verifier-"
)


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


def _finite(value: object, *, name: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be one real scalar")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        suffix = " and nonnegative" if nonnegative else ""
        raise ValueError(f"{name} must be finite{suffix}")
    return result


def _finite_axis(
    values: Sequence[float],
    *,
    name: str,
    width: int | None = None,
    nonnegative: bool = False,
) -> tuple[float, ...]:
    result = tuple(_finite(value, name=f"{name} item", nonnegative=nonnegative) for value in values)
    if width is not None and len(result) != width:
        raise ValueError(f"{name} has the wrong endpoint width")
    return result


def _finite_matrix(
    values: Sequence[Sequence[float]],
    *,
    name: str,
    rows: int | None = None,
    columns: int | None = None,
    nonnegative: bool = False,
) -> tuple[tuple[float, ...], ...]:
    result = tuple(
        _finite_axis(
            row,
            name=f"{name} row",
            width=columns,
            nonnegative=nonnegative,
        )
        for row in values
    )
    if rows is not None and len(result) != rows:
        raise ValueError(f"{name} has the wrong stratum/dose height")
    return result


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _outward_square_upper(value: float) -> float:
    checked = _finite(value, name="literal-J projection bound", nonnegative=True)
    if checked == 0.0:
        return 0.0
    nominal = checked * checked
    upper = math.nextafter(nominal, math.inf)
    if not math.isfinite(upper):
        raise FloatingPointError("literal-J quadratic bound overflowed binary64")
    return upper


def _outward_interval_difference(
    left_lower: float,
    left_upper: float,
    right_lower: float,
    right_upper: float,
) -> tuple[float, float]:
    nominal_lower = left_lower - right_upper
    nominal_upper = left_upper - right_lower
    if not math.isfinite(nominal_lower) or not math.isfinite(nominal_upper):
        raise FloatingPointError("interval difference overflowed binary64")
    lower = math.nextafter(nominal_lower, -math.inf)
    upper = math.nextafter(nominal_upper, math.inf)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise FloatingPointError("outward interval difference is unusable")
    return lower, upper


def _outward_scaled_nonnegative(value: float, scale: float) -> float:
    checked = _finite(value, name="scaled bound", nonnegative=True)
    dose = _finite(scale, name="dose", nonnegative=True)
    if checked == 0.0 or dose == 0.0:
        return 0.0
    upper = math.nextafter(checked * dose, math.inf)
    if not math.isfinite(upper):
        raise FloatingPointError("scaled bound overflowed binary64")
    return upper


def _interval_remainder_upper(
    *,
    parent_interval: tuple[float, float],
    relaxed_interval: tuple[float, float],
    literal_jacobian_projection_upper: float,
    dose: float,
) -> float:
    delta_lower, delta_upper = _outward_interval_difference(
        relaxed_interval[0],
        relaxed_interval[1],
        parent_interval[0],
        parent_interval[1],
    )
    tangent = _outward_scaled_nonnegative(literal_jacobian_projection_upper, dose)
    nominal_lower = delta_lower - tangent
    nominal_upper = delta_upper + tangent
    if not math.isfinite(nominal_lower) or not math.isfinite(nominal_upper):
        raise FloatingPointError("directional-remainder arithmetic overflowed binary64")
    lower = math.nextafter(nominal_lower, -math.inf)
    upper = math.nextafter(nominal_upper, math.inf)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise FloatingPointError("directional-remainder interval is unusable")
    nominal = max(abs(lower), abs(upper))
    if nominal == 0.0:
        return 0.0
    result = math.nextafter(nominal, math.inf)
    if not math.isfinite(result):
        raise FloatingPointError("directional-remainder bound overflowed binary64")
    return result


def _readout_interval_from_forward(
    *,
    model: Any,
    final_hidden_interval: tuple[Any, Any],
    target_class_id: int,
    profile: Any,
) -> tuple[float, float, float]:
    hidden_lower, hidden_upper = final_hidden_interval
    readout_lower, readout_upper, factor = _env._linear_forward_interval_bound(
        hidden_lower,
        hidden_upper,
        model.classifier[6],
        None,
        profile,
        output_row_chunk=1,
        output_row_start=target_class_id,
        output_row_stop=target_class_id + 1,
    )
    if readout_lower.numel() != 1 or readout_upper.numel() != 1:
        raise RuntimeError("target raw-logit interval changed scalar shape")
    lower = _finite(float(readout_lower[0].item()), name="raw-logit lower")
    upper = _finite(float(readout_upper[0].item()), name="raw-logit upper")
    if lower > upper:
        raise FloatingPointError("raw-logit interval is not ordered")
    return lower, upper, _finite(factor, name="readout factor")


def _mask_hash(mask: HardMask, *, name: str) -> str:
    return _env._direct_hard_mask_hash(mask, name=name)


def _relaxed_gate_path_hash(
    *,
    parent_mask_hash: str,
    endpoint_mask_hash: str,
    action_hash: str,
    ladder_hash: str,
    chart_identity_hash: str,
    proposal_generation: int,
    dose: float,
) -> str:
    return content_hash(
        {
            "algorithm_id": ALEXNET_FQR_RELAXED_GATE_PATH_ID,
            "parent_mask_hash": parent_mask_hash,
            "endpoint_mask_hash": endpoint_mask_hash,
            "action_hash": action_hash,
            "ladder_hash": ladder_hash,
            "chart_identity_hash": chart_identity_hash,
            "proposal_generation": proposal_generation,
            "dose": dose,
            "gate_equation": "m_minus_eta_z",
            "positive_removal_sign": POSITIVE_REMOVAL_SIGN,
        },
        prefix=_RELAXED_GATE_PREFIX,
    )


@dataclass(frozen=True, slots=True)
class V1AlexNetFQRRelaxedLogitIntervalReceipt:
    """One pre-outcome raw-logit interval on a registered relaxed action."""

    endpoint_index: int
    action_hash: str
    dose_index: int
    dose: float
    parent_mask_hash: str
    endpoint_mask_hash: str
    ladder_hash: str
    chart_identity_hash: str
    proposal_generation: int
    route_count: int
    route_axis_hash: str
    model_identity_hash: str
    model_state_hash: str
    query_contract_hash: str
    preprocessing_hash: str
    input_box_hash: str
    layout_hash: str
    graph_semantics_hash: str
    arithmetic_profile_hash: str
    relaxed_gate_path_hash: str
    raw_target_logit_lower: float
    raw_target_logit_upper: float
    readout_rounding_factor: float
    interval_engine_source_hash: str
    verifier_source_hash: str
    algorithm_id: str = _env.ALEXNET_RAW_TARGET_LOGIT_INTERVAL_ALGORITHM_ID
    outward_rounding_id: str = _env.ALEXNET_RAW_TARGET_LOGIT_OUTWARD_ROUNDING_ID
    relaxed_gate_path_id: str = ALEXNET_FQR_RELAXED_GATE_PATH_ID
    full_dose_equals_endpoint_hard_mask: bool = False
    issued_before_C_outcomes: bool = True
    C_outcomes_seen: bool = False
    validation_outcomes_seen: bool = False
    downstream_authority: bool = False
    schema: str = ALEXNET_FQR_RELAXED_LOGIT_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        endpoint = _positive_int(self.endpoint_index, name="endpoint index")
        dose_index = _nonnegative_int(self.dose_index, name="dose index")
        generation = _nonnegative_int(self.proposal_generation, name="proposal generation")
        route_count = _positive_int(self.route_count, name="route count")
        if dose_index >= len(DIRECTIONAL_SECANT_DOSES):
            raise ValueError("relaxed-logit dose index left the registered axis")
        dose = _finite(self.dose, name="dose", nonnegative=True)
        if dose != DIRECTIONAL_SECANT_DOSES[dose_index]:
            raise ValueError("relaxed-logit dose differs from the canonical dose axis")
        for name in (
            "action_hash",
            "parent_mask_hash",
            "endpoint_mask_hash",
            "ladder_hash",
            "chart_identity_hash",
            "route_axis_hash",
            "model_identity_hash",
            "model_state_hash",
            "query_contract_hash",
            "preprocessing_hash",
            "input_box_hash",
            "layout_hash",
            "graph_semantics_hash",
            "arithmetic_profile_hash",
            "relaxed_gate_path_hash",
            "interval_engine_source_hash",
            "verifier_source_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        lower = _finite(self.raw_target_logit_lower, name="raw-logit lower")
        upper = _finite(self.raw_target_logit_upper, name="raw-logit upper")
        if lower > upper:
            raise ValueError("relaxed raw-logit interval is not ordered")
        factor = _finite(
            self.readout_rounding_factor,
            name="readout rounding factor",
            nonnegative=True,
        )
        if factor < 1.0:
            raise ValueError("readout rounding factor must be at least one")
        expected_full = dose == 1.0
        if type(self.full_dose_equals_endpoint_hard_mask) is not bool:
            raise TypeError("full-dose hard-mask marker must be an exact bool")
        if self.full_dose_equals_endpoint_hard_mask is not expected_full:
            raise ValueError("full-dose hard-mask marker differs from the dose")
        expected_path_hash = _relaxed_gate_path_hash(
            parent_mask_hash=self.parent_mask_hash,
            endpoint_mask_hash=self.endpoint_mask_hash,
            action_hash=self.action_hash,
            ladder_hash=self.ladder_hash,
            chart_identity_hash=self.chart_identity_hash,
            proposal_generation=generation,
            dose=dose,
        )
        if self.relaxed_gate_path_hash != expected_path_hash:
            raise ValueError("relaxed gate-path hash is invalid")
        if (
            self.algorithm_id != _env.ALEXNET_RAW_TARGET_LOGIT_INTERVAL_ALGORITHM_ID
            or self.outward_rounding_id != _env.ALEXNET_RAW_TARGET_LOGIT_OUTWARD_ROUNDING_ID
            or self.relaxed_gate_path_id != ALEXNET_FQR_RELAXED_GATE_PATH_ID
            or self.schema != ALEXNET_FQR_RELAXED_LOGIT_RECEIPT_SCHEMA
        ):
            raise ValueError("relaxed-logit interval semantics are invalid")
        if (
            self.issued_before_C_outcomes is not True
            or self.C_outcomes_seen is not False
            or self.validation_outcomes_seen is not False
            or self.downstream_authority is not False
        ):
            raise PermissionError("relaxed-logit receipt crossed its authority boundary")
        object.__setattr__(self, "endpoint_index", endpoint)
        object.__setattr__(self, "dose_index", dose_index)
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "dose", dose)
        object.__setattr__(self, "raw_target_logit_lower", lower)
        object.__setattr__(self, "raw_target_logit_upper", upper)
        object.__setattr__(self, "readout_rounding_factor", factor)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "endpoint_index",
                "action_hash",
                "dose_index",
                "dose",
                "parent_mask_hash",
                "endpoint_mask_hash",
                "ladder_hash",
                "chart_identity_hash",
                "proposal_generation",
                "route_count",
                "route_axis_hash",
                "model_identity_hash",
                "model_state_hash",
                "query_contract_hash",
                "preprocessing_hash",
                "input_box_hash",
                "layout_hash",
                "graph_semantics_hash",
                "arithmetic_profile_hash",
                "relaxed_gate_path_hash",
                "raw_target_logit_lower",
                "raw_target_logit_upper",
                "readout_rounding_factor",
                "interval_engine_source_hash",
                "verifier_source_hash",
                "algorithm_id",
                "outward_rounding_id",
                "relaxed_gate_path_id",
                "full_dose_equals_endpoint_hard_mask",
                "issued_before_C_outcomes",
                "C_outcomes_seen",
                "validation_outcomes_seen",
                "downstream_authority",
            )
        }

    @property
    def receipt_hash(self) -> str:
        cached = self._receipt_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        if (
            self.issued_before_C_outcomes is not True
            or self.C_outcomes_seen is not False
            or self.validation_outcomes_seen is not False
            or self.downstream_authority is not False
        ):
            raise PermissionError("relaxed-logit receipt gained downstream authority")
        observed = content_hash(self._payload(), prefix=_RELAXED_RECEIPT_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("relaxed-logit receipt was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("relaxed-logit receipt hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1AlexNetFQRBoundDerivationReceipt:
    """Closed provenance and arithmetic proof for one branch's F/Q/R arrays."""

    input_id: str
    branch: V1RelaxedKKTBranch
    campaign_contract_hash: str
    model_identity_hash: str
    model_state_hash: str
    query_contract_hash: str
    preprocessing_hash: str
    current_mask_hash: str
    raw_logit_stream_source_hash: str
    sampling_law_hash: str
    proposal_generation: int
    route_count: int
    route_axis_hash: str
    route_axis_content_hash: str
    chart_identity_hash: str
    ladder_hash: str
    endpoint_action_hashes: tuple[str, ...]
    endpoint_mask_hashes: tuple[str, ...]
    doses: tuple[float, float]
    stratum_names: tuple[str, ...]
    population_stratum_counts: tuple[int, ...]
    sealed_population_order_hashes: tuple[str, ...]
    input_box_hash: str
    layout_hash: str
    graph_semantics_hash: str
    arithmetic_profile_hash: str
    parent_raw_target_logit_interval: tuple[float, float]
    corrected_f_upper_by_stratum_endpoint: tuple[tuple[float, ...], ...]
    literal_jacobian_projection_upper_by_endpoint: tuple[float, ...]
    q_upper_by_endpoint: tuple[float, ...]
    raw_interval_remainder_upper_by_endpoint_and_dose: tuple[tuple[float, ...], ...]
    r_upper_by_endpoint_and_dose: tuple[tuple[float, ...], ...]
    exact_linear_proof_by_endpoint: tuple[str, ...]
    relaxed_logit_receipts: tuple[tuple[V1AlexNetFQRRelaxedLogitIntervalReceipt, ...], ...]
    corrected_work_endpoint_certificate_hash: str
    corrected_work_route_envelope_certificate_hash: str
    corrected_work_route_envelope_proof_hash: str
    corrected_work_statistical_design_hash: str
    literal_jacobian_statistical_design_hash: str
    fd_rank_uncertainty_set_hash: str
    kkt_numerical_uncertainty_set_hash: str
    numerical_enclosure_policy_hash: str
    interval_engine_source_hash: str
    verifier_source_hash: str
    # Schema v2 binds the physical packed action separately from the frozen
    # ladder wrapper.  Those are distinct content-addressed objects and their
    # hashes must never impersonate one another.  Empty retains exact v1
    # replay compatibility but is insufficient for physical-action C.
    endpoint_packed_action_hashes: tuple[str, ...] = ()
    positive_removal_sign: str = POSITIVE_REMOVAL_SIGN
    corrected_evidence_equation: str = CORRECTED_EVIDENCE_EQUATION
    literal_jacobian_equation: str = LITERAL_JACOBIAN_EQUATION
    relu_kink_policy: str = RELU_KINK_POLICY
    signed_raw_logit_delta_equation: str = SIGNED_RAW_LOGIT_DELTA_EQUATION
    signed_tangent_delta_equation: str = SIGNED_TANGENT_DELTA_EQUATION
    signed_remainder_equation: str = SIGNED_REMAINDER_EQUATION
    algorithm_id: str = ALEXNET_FQR_BOUND_ALGORITHM_ID
    literal_jacobian_algorithm_id: str = ALEXNET_FQR_LITERAL_J_ALGORITHM_ID
    remainder_algorithm_id: str = ALEXNET_FQR_REMAINDER_ALGORITHM_ID
    numerical_enclosure_id: str = ALEXNET_FQR_NUMERICAL_ENCLOSURE_ID
    issued_before_C_outcomes: bool = True
    C_outcomes_seen: bool = False
    bounds_mutable_after_C: bool = False
    validation_outcomes_seen: bool = False
    calibration_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = ALEXNET_FQR_AUTHORITY_SCOPE
    schema: str = ALEXNET_FQR_BOUND_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_id", _identity(self.input_id, name="input id"))
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("F/Q/R bound receipt requires the exact KKT branch enum")
        for name in (
            "campaign_contract_hash",
            "model_identity_hash",
            "model_state_hash",
            "query_contract_hash",
            "preprocessing_hash",
            "current_mask_hash",
            "raw_logit_stream_source_hash",
            "sampling_law_hash",
            "route_axis_hash",
            "route_axis_content_hash",
            "chart_identity_hash",
            "ladder_hash",
            "input_box_hash",
            "layout_hash",
            "graph_semantics_hash",
            "arithmetic_profile_hash",
            "corrected_work_endpoint_certificate_hash",
            "corrected_work_route_envelope_certificate_hash",
            "corrected_work_route_envelope_proof_hash",
            "corrected_work_statistical_design_hash",
            "literal_jacobian_statistical_design_hash",
            "fd_rank_uncertainty_set_hash",
            "kkt_numerical_uncertainty_set_hash",
            "numerical_enclosure_policy_hash",
            "interval_engine_source_hash",
            "verifier_source_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        generation = _nonnegative_int(self.proposal_generation, name="proposal generation")
        route_count = _positive_int(self.route_count, name="route count")
        actions = tuple(
            _identity(value, name="endpoint action hash") for value in self.endpoint_action_hashes
        )
        packed_actions = tuple(
            _identity(value, name="endpoint packed-action hash")
            for value in self.endpoint_packed_action_hashes
        )
        masks = tuple(
            _identity(value, name="endpoint mask hash") for value in self.endpoint_mask_hashes
        )
        width = len(actions)
        if (
            width == 0
            or len(masks) != width
            or (packed_actions and len(packed_actions) != width)
        ):
            raise ValueError("F/Q/R endpoint identity axes are empty or misaligned")
        if (
            (packed_actions and self.schema != ALEXNET_FQR_BOUND_RECEIPT_SCHEMA_V2)
            or (
                not packed_actions
                and self.schema != ALEXNET_FQR_BOUND_RECEIPT_SCHEMA
            )
        ):
            raise ValueError(
                "F/Q/R packed-action identity requires derivation-receipt schema v2"
            )
        doses = tuple(_finite(value, name="dose") for value in self.doses)
        if doses != DIRECTIONAL_SECANT_DOSES:
            raise ValueError("F/Q/R receipt requires canonical half/full doses")
        strata = tuple(_identity(value, name="stratum name") for value in self.stratum_names)
        counts = tuple(
            _positive_int(value, name="population stratum count")
            for value in self.population_stratum_counts
        )
        orders = tuple(
            _identity(value, name="sealed population order hash")
            for value in self.sealed_population_order_hashes
        )
        if (
            not strata
            or len(set(strata)) != len(strata)
            or not (len(strata) == len(counts) == len(orders))
        ):
            raise ValueError("F/Q/R receipt stratum axes are invalid")
        parent_values = _finite_axis(
            self.parent_raw_target_logit_interval,
            name="parent raw-logit interval",
            width=2,
        )
        parent = (parent_values[0], parent_values[1])
        if parent[0] > parent[1]:
            raise ValueError("parent raw-logit interval is not ordered")
        f_rows = _finite_matrix(
            self.corrected_f_upper_by_stratum_endpoint,
            name="corrected F upper",
            rows=len(strata),
            columns=width,
            nonnegative=True,
        )
        j_axis = _finite_axis(
            self.literal_jacobian_projection_upper_by_endpoint,
            name="literal-J projection upper",
            width=width,
            nonnegative=True,
        )
        q_axis = _finite_axis(
            self.q_upper_by_endpoint,
            name="Q upper",
            width=width,
            nonnegative=True,
        )
        expected_q = tuple(_outward_square_upper(value) for value in j_axis)
        if q_axis != expected_q:
            raise ValueError("Q upper is understated or differs from outward J square")
        raw_r = _finite_matrix(
            self.raw_interval_remainder_upper_by_endpoint_and_dose,
            name="raw interval R upper",
            rows=width,
            columns=len(doses),
            nonnegative=True,
        )
        r_axis = _finite_matrix(
            self.r_upper_by_endpoint_and_dose,
            name="R upper",
            rows=width,
            columns=len(doses),
            nonnegative=True,
        )
        proofs = tuple(
            _identity(value, name="exact-linear proof")
            for value in self.exact_linear_proof_by_endpoint
        )
        if len(proofs) != width or any(value not in _EXACT_LINEAR_PROOFS for value in proofs):
            raise ValueError("exact-linear proof axis is invalid")
        relaxed = tuple(tuple(row) for row in self.relaxed_logit_receipts)
        if len(relaxed) != width or any(len(row) != len(doses) for row in relaxed):
            raise ValueError("relaxed-logit receipt axes are misaligned")
        for endpoint_offset, row in enumerate(relaxed):
            for dose_index, item in enumerate(row):
                if type(item) is not V1AlexNetFQRRelaxedLogitIntervalReceipt:
                    raise TypeError("F/Q/R derivation requires typed relaxed receipts")
                item.assert_integrity()
                if (
                    item.endpoint_index != endpoint_offset + 1
                    or item.action_hash != actions[endpoint_offset]
                    or item.dose_index != dose_index
                    or item.dose != doses[dose_index]
                    or item.parent_mask_hash != self.current_mask_hash
                    or item.endpoint_mask_hash != masks[endpoint_offset]
                    or item.ladder_hash != self.ladder_hash
                    or item.chart_identity_hash != self.chart_identity_hash
                    or item.proposal_generation != generation
                    or item.route_count != route_count
                    or item.route_axis_hash != self.route_axis_hash
                    or item.model_identity_hash != self.model_identity_hash
                    or item.model_state_hash != self.model_state_hash
                    or item.query_contract_hash != self.query_contract_hash
                    or item.preprocessing_hash != self.preprocessing_hash
                    or item.input_box_hash != self.input_box_hash
                    or item.layout_hash != self.layout_hash
                    or item.graph_semantics_hash != self.graph_semantics_hash
                    or item.arithmetic_profile_hash != self.arithmetic_profile_hash
                    or item.interval_engine_source_hash != self.interval_engine_source_hash
                    or item.verifier_source_hash != self.verifier_source_hash
                ):
                    raise ValueError("a relaxed-logit receipt left its F/Q/R coordinates")
                expected_raw = _interval_remainder_upper(
                    parent_interval=parent,
                    relaxed_interval=(
                        item.raw_target_logit_lower,
                        item.raw_target_logit_upper,
                    ),
                    literal_jacobian_projection_upper=j_axis[endpoint_offset],
                    dose=doses[dose_index],
                )
                if raw_r[endpoint_offset][dose_index] != expected_raw:
                    raise ValueError("raw interval remainder upper is understated")
        for endpoint_offset, proof in enumerate(proofs):
            if proof == EXACT_LINEAR_PROOF_ZERO_SELECTED_WEIGHT and (
                j_axis[endpoint_offset] != 0.0
                or q_axis[endpoint_offset] != 0.0
                or any(row[endpoint_offset] != 0.0 for row in f_rows)
                or any(
                    (
                        item.raw_target_logit_lower,
                        item.raw_target_logit_upper,
                    )
                    != parent
                    for item in relaxed[endpoint_offset]
                )
            ):
                raise ValueError("zero-weight exact-linear proof is not numerically closed")
            expected = (0.0, 0.0) if proof != EXACT_LINEAR_PROOF_NONE else raw_r[endpoint_offset]
            if r_axis[endpoint_offset] != expected:
                raise ValueError("certified R upper differs from its registered proof")
        if (
            self.positive_removal_sign != POSITIVE_REMOVAL_SIGN
            or self.corrected_evidence_equation != CORRECTED_EVIDENCE_EQUATION
            or self.literal_jacobian_equation != LITERAL_JACOBIAN_EQUATION
            or self.relu_kink_policy != RELU_KINK_POLICY
            or self.signed_raw_logit_delta_equation != SIGNED_RAW_LOGIT_DELTA_EQUATION
            or self.signed_tangent_delta_equation != SIGNED_TANGENT_DELTA_EQUATION
            or self.signed_remainder_equation != SIGNED_REMAINDER_EQUATION
            or self.algorithm_id != ALEXNET_FQR_BOUND_ALGORITHM_ID
            or self.literal_jacobian_algorithm_id != ALEXNET_FQR_LITERAL_J_ALGORITHM_ID
            or self.remainder_algorithm_id != ALEXNET_FQR_REMAINDER_ALGORITHM_ID
            or self.numerical_enclosure_id != ALEXNET_FQR_NUMERICAL_ENCLOSURE_ID
            or self.authority_scope != ALEXNET_FQR_AUTHORITY_SCOPE
            or self.schema
            not in {
                ALEXNET_FQR_BOUND_RECEIPT_SCHEMA,
                ALEXNET_FQR_BOUND_RECEIPT_SCHEMA_V2,
            }
        ):
            raise ValueError("F/Q/R derivation semantics or schema are invalid")
        if (
            self.issued_before_C_outcomes is not True
            or self.C_outcomes_seen is not False
            or self.bounds_mutable_after_C is not False
            or self.validation_outcomes_seen is not False
            or self.calibration_authority is not False
            or self.bce_oracle_authority is not False
            or self.mask_commit_authority is not False
            or self.terminal_authority is not False
        ):
            raise PermissionError("F/Q/R derivation crossed its pre-C authority boundary")
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "endpoint_action_hashes", actions)
        object.__setattr__(
            self,
            "endpoint_packed_action_hashes",
            packed_actions,
        )
        object.__setattr__(self, "endpoint_mask_hashes", masks)
        object.__setattr__(self, "doses", doses)
        object.__setattr__(self, "stratum_names", strata)
        object.__setattr__(self, "population_stratum_counts", counts)
        object.__setattr__(self, "sealed_population_order_hashes", orders)
        object.__setattr__(self, "parent_raw_target_logit_interval", parent)
        object.__setattr__(self, "corrected_f_upper_by_stratum_endpoint", f_rows)
        object.__setattr__(self, "literal_jacobian_projection_upper_by_endpoint", j_axis)
        object.__setattr__(self, "q_upper_by_endpoint", q_axis)
        object.__setattr__(self, "raw_interval_remainder_upper_by_endpoint_and_dose", raw_r)
        object.__setattr__(self, "r_upper_by_endpoint_and_dose", r_axis)
        object.__setattr__(self, "exact_linear_proof_by_endpoint", proofs)
        object.__setattr__(self, "relaxed_logit_receipts", relaxed)
        self.assert_integrity()

    def _f_source_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "algorithm_id": ALEXNET_FQR_BOUND_ALGORITHM_ID,
            "owner": "conserving_aumann_shapley_corrected_evidence",
            "corrected_evidence_equation": self.corrected_evidence_equation,
            "endpoint_certificate_hash": self.corrected_work_endpoint_certificate_hash,
            "route_envelope_certificate_hash": (
                self.corrected_work_route_envelope_certificate_hash
            ),
            "route_envelope_proof_hash": self.corrected_work_route_envelope_proof_hash,
            "ladder_hash": self.ladder_hash,
            "chart_identity_hash": self.chart_identity_hash,
            "corrected_f_upper": self.corrected_f_upper_by_stratum_endpoint,
            "verifier_source_hash": self.verifier_source_hash,
            "interval_engine_source_hash": self.interval_engine_source_hash,
        }
        if self.endpoint_packed_action_hashes:
            payload["endpoint_packed_action_hashes"] = (
                self.endpoint_packed_action_hashes
            )
        return payload

    def _q_source_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "algorithm_id": self.literal_jacobian_algorithm_id,
            "owner": "literal_relaxed_gate_response_jacobian",
            "literal_jacobian_equation": self.literal_jacobian_equation,
            "relu_kink_policy": self.relu_kink_policy,
            "corrected_rows_used_as_literal_jacobian": False,
            "model_state_hash": self.model_state_hash,
            "input_box_hash": self.input_box_hash,
            "current_mask_hash": self.current_mask_hash,
            "ladder_hash": self.ladder_hash,
            "chart_identity_hash": self.chart_identity_hash,
            "literal_jacobian_projection_upper": (
                self.literal_jacobian_projection_upper_by_endpoint
            ),
            "q_upper": self.q_upper_by_endpoint,
            "verifier_source_hash": self.verifier_source_hash,
            "interval_engine_source_hash": self.interval_engine_source_hash,
        }
        if self.endpoint_packed_action_hashes:
            payload["endpoint_packed_action_hashes"] = (
                self.endpoint_packed_action_hashes
            )
        return payload

    def _r_source_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "algorithm_id": self.remainder_algorithm_id,
            "signed_raw_logit_delta_equation": self.signed_raw_logit_delta_equation,
            "signed_tangent_delta_equation": self.signed_tangent_delta_equation,
            "signed_remainder_equation": self.signed_remainder_equation,
            "parent_interval": self.parent_raw_target_logit_interval,
            "relaxed_receipt_hashes": tuple(
                tuple(item.receipt_hash for item in row) for row in self.relaxed_logit_receipts
            ),
            "doses": self.doses,
            "raw_interval_r_upper": (self.raw_interval_remainder_upper_by_endpoint_and_dose),
            "exact_linear_proofs": self.exact_linear_proof_by_endpoint,
            "r_upper": self.r_upper_by_endpoint_and_dose,
            "verifier_source_hash": self.verifier_source_hash,
            "interval_engine_source_hash": self.interval_engine_source_hash,
        }
        if self.endpoint_packed_action_hashes:
            payload["endpoint_packed_action_hashes"] = (
                self.endpoint_packed_action_hashes
            )
        return payload

    @property
    def corrected_work_bound_source_hash(self) -> str:
        return content_hash(self._f_source_payload(), prefix=_F_SOURCE_PREFIX)

    @property
    def literal_jacobian_bound_source_hash(self) -> str:
        return content_hash(self._q_source_payload(), prefix=_Q_SOURCE_PREFIX)

    @property
    def directional_remainder_bound_source_hash(self) -> str:
        return content_hash(self._r_source_payload(), prefix=_R_SOURCE_PREFIX)

    def _payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            name: getattr(self, name)
            for name in (
                "schema",
                "input_id",
                "branch",
                "campaign_contract_hash",
                "model_identity_hash",
                "model_state_hash",
                "query_contract_hash",
                "preprocessing_hash",
                "current_mask_hash",
                "raw_logit_stream_source_hash",
                "sampling_law_hash",
                "proposal_generation",
                "route_count",
                "route_axis_hash",
                "route_axis_content_hash",
                "chart_identity_hash",
                "ladder_hash",
                "endpoint_action_hashes",
                "endpoint_mask_hashes",
                "doses",
                "stratum_names",
                "population_stratum_counts",
                "sealed_population_order_hashes",
                "input_box_hash",
                "layout_hash",
                "graph_semantics_hash",
                "arithmetic_profile_hash",
                "parent_raw_target_logit_interval",
                "corrected_f_upper_by_stratum_endpoint",
                "literal_jacobian_projection_upper_by_endpoint",
                "q_upper_by_endpoint",
                "raw_interval_remainder_upper_by_endpoint_and_dose",
                "r_upper_by_endpoint_and_dose",
                "exact_linear_proof_by_endpoint",
                "corrected_work_endpoint_certificate_hash",
                "corrected_work_route_envelope_certificate_hash",
                "corrected_work_route_envelope_proof_hash",
                "corrected_work_statistical_design_hash",
                "literal_jacobian_statistical_design_hash",
                "fd_rank_uncertainty_set_hash",
                "kkt_numerical_uncertainty_set_hash",
                "numerical_enclosure_policy_hash",
                "interval_engine_source_hash",
                "verifier_source_hash",
                "positive_removal_sign",
                "corrected_evidence_equation",
                "literal_jacobian_equation",
                "relu_kink_policy",
                "signed_raw_logit_delta_equation",
                "signed_tangent_delta_equation",
                "signed_remainder_equation",
                "algorithm_id",
                "literal_jacobian_algorithm_id",
                "remainder_algorithm_id",
                "numerical_enclosure_id",
                "issued_before_C_outcomes",
                "C_outcomes_seen",
                "bounds_mutable_after_C",
                "validation_outcomes_seen",
                "calibration_authority",
                "bce_oracle_authority",
                "mask_commit_authority",
                "terminal_authority",
                "authority_scope",
            )
        }
        payload.update(
            {
                "relaxed_logit_receipt_hashes": tuple(
                    tuple(item.receipt_hash for item in row) for row in self.relaxed_logit_receipts
                ),
                "corrected_work_bound_source_hash": (self.corrected_work_bound_source_hash),
                "literal_jacobian_bound_source_hash": (self.literal_jacobian_bound_source_hash),
                "directional_remainder_bound_source_hash": (
                    self.directional_remainder_bound_source_hash
                ),
            }
        )
        if self.endpoint_packed_action_hashes:
            payload["endpoint_packed_action_hashes"] = (
                self.endpoint_packed_action_hashes
            )
        return payload

    @property
    def receipt_hash(self) -> str:
        cached = self._receipt_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        if (
            self.issued_before_C_outcomes is not True
            or self.C_outcomes_seen is not False
            or self.bounds_mutable_after_C is not False
            or self.validation_outcomes_seen is not False
            or self.calibration_authority is not False
            or self.bce_oracle_authority is not False
            or self.mask_commit_authority is not False
            or self.terminal_authority is not False
        ):
            raise PermissionError("F/Q/R receipt gained downstream authority")
        for row in self.relaxed_logit_receipts:
            for item in row:
                item.assert_integrity()
        observed = content_hash(self._payload(), prefix=_BOUND_RECEIPT_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("F/Q/R bound receipt was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("F/Q/R bound receipt hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1AlexNetFQRBranchAnalyticBounds:
    """Ready branch input plus the complete pre-outcome analytic proof graph."""

    analytic_input: V1FQRBranchAnalyticInput = field(repr=False, compare=False)
    derivation_receipt: V1AlexNetFQRBoundDerivationReceipt
    corrected_work_endpoint_certificate: V1FrozenLadderEndpointEnvelopeCertificate = field(
        repr=False, compare=False
    )
    schema: str = ALEXNET_FQR_BOUND_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ALEXNET_FQR_BOUND_RESULT_SCHEMA:
            raise ValueError("unsupported AlexNet F/Q/R bound result schema")
        self.assert_integrity()

    def assert_integrity(self) -> tuple[str, str, str]:
        if type(self.analytic_input) is not V1FQRBranchAnalyticInput:
            raise TypeError("AlexNet F/Q/R result requires the exact branch input")
        if type(self.derivation_receipt) is not V1AlexNetFQRBoundDerivationReceipt:
            raise TypeError("AlexNet F/Q/R result requires the exact derivation receipt")
        if (
            type(self.corrected_work_endpoint_certificate)
            is not V1FrozenLadderEndpointEnvelopeCertificate
        ):
            raise TypeError("AlexNet F/Q/R result requires the exact F certificate")
        input_hash = self.analytic_input.assert_integrity()
        receipt_hash = self.derivation_receipt.assert_integrity()
        f_hash = self.corrected_work_endpoint_certificate.assert_integrity()
        receipt = self.derivation_receipt
        analytic = self.analytic_input
        certificate = self.corrected_work_endpoint_certificate
        f_array, q_array, r_array = (
            _sequential_observation_support_arrays(receipt)
        )
        if (
            analytic.input_id != receipt.input_id
            or analytic.branch is not receipt.branch
            or analytic.corrected_work_statistical_design_hash
            != receipt.corrected_work_statistical_design_hash
            or analytic.literal_jacobian_statistical_design_hash
            != receipt.literal_jacobian_statistical_design_hash
            or analytic.corrected_work_bound_source_hash != receipt.corrected_work_bound_source_hash
            or analytic.literal_jacobian_bound_source_hash
            != receipt.literal_jacobian_bound_source_hash
            or analytic.directional_remainder_bound_source_hash
            != receipt.directional_remainder_bound_source_hash
            or analytic.fd_rank_uncertainty_set_hash != receipt.fd_rank_uncertainty_set_hash
            or analytic.kkt_numerical_uncertainty_set_hash
            != receipt.kkt_numerical_uncertainty_set_hash
            or analytic.numerical_enclosure_policy_hash != receipt.numerical_enclosure_policy_hash
            or not np.array_equal(analytic.analytic_f_upper_by_stratum_endpoint, f_array)
            or not np.array_equal(analytic.analytic_q_upper_by_stratum_endpoint, q_array)
            or not np.array_equal(
                analytic.analytic_r_upper_by_stratum_endpoint_and_dose,
                r_array,
            )
            or f_hash != receipt.corrected_work_endpoint_certificate_hash
            or certificate.ladder_hash != receipt.ladder_hash
            or certificate.endpoint_action_hashes != receipt.endpoint_action_hashes
            or tuple(certificate.stratum_names) != receipt.stratum_names
            or tuple(
                tuple(float(value) for value in row)
                for row in certificate.endpoint_absolute_upper_bounds_by_stratum
            )
            != receipt.corrected_f_upper_by_stratum_endpoint
        ):
            raise ValueError("AlexNet F/Q/R result components do not close")
        return input_hash, receipt_hash, f_hash


def _sequential_observation_support_arrays(
    receipt: V1AlexNetFQRBoundDerivationReceipt,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pre-C support for the *reported* conservative F/Q/R rows.

    The receipt keeps the mathematical remainder bound separately in
    ``r_upper_by_endpoint_and_dose``.  Sequential acquisition consumes
    interval-enclosed binary64 observations, however, so its support must also
    cover deterministic measurement enclosure.  In particular, an analytic
    exact-linear proof may establish mathematical ``R=0`` while the reported
    secant still has a nonzero binary64 radius.  Using the raw interval
    remainder support (with one outward spacing), rather than pretending that
    radius is zero, preserves both claims without an independently tuned
    remainder budget.
    """

    if type(receipt) is not V1AlexNetFQRBoundDerivationReceipt:
        raise TypeError("sequential observation support requires the exact receipt")
    receipt.assert_integrity()
    f_mathematical = np.asarray(
        receipt.corrected_f_upper_by_stratum_endpoint,
        dtype=np.float64,
    )
    f_support = np.asarray(
        tuple(
            tuple(
                _strict_outward_nonnegative(
                    float(value),
                    name="corrected-work observation support",
                )
                for value in row
            )
            for row in f_mathematical
        ),
        dtype=np.float64,
    )
    j_support = np.asarray(
        tuple(
            _strict_outward_nonnegative(
                float(value),
                name="literal-J observation support",
            )
            for value in receipt.literal_jacobian_projection_upper_by_endpoint
        ),
        dtype=np.float64,
    )
    q_endpoint_support = np.asarray(
        tuple(_outward_square_upper(float(value)) for value in j_support),
        dtype=np.float64,
    )
    q_support = np.broadcast_to(
        q_endpoint_support,
        f_support.shape,
    ).copy()
    raw_r = np.asarray(
        receipt.raw_interval_remainder_upper_by_endpoint_and_dose,
        dtype=np.float64,
    )
    r_endpoint_support = np.asarray(
        tuple(
            tuple(
                _strict_outward_nonnegative(
                    float(value),
                    name="directional-remainder observation support",
                )
                for value in row
            )
            for row in raw_r
        ),
        dtype=np.float64,
    )
    r_support = np.broadcast_to(
        r_endpoint_support.reshape(1, *r_endpoint_support.shape),
        (*f_support.shape, len(receipt.doses)),
    ).copy()
    return f_support, q_support, r_support


def _assert_directional_adapter_coordinates(
    *,
    result: V1AlexNetFQRBranchAnalyticBounds,
    design: V1DirectionalSecantDesign,
) -> tuple[V1AlexNetFQRBoundDerivationReceipt, str, str]:
    """Close one analytic branch result against its exact finite-dose design."""

    if type(result) is not V1AlexNetFQRBranchAnalyticBounds:
        raise TypeError("AlexNet directional bounds require the exact branch result")
    if type(design) is not V1DirectionalSecantDesign:
        raise TypeError("AlexNet directional bounds require the exact secant design")
    _input_hash, receipt_hash, _f_hash = result.assert_integrity()
    design_hash = design.assert_integrity()
    receipt = result.derivation_receipt
    ladder = design.ladder
    if (
        design.ladder_hash != receipt.ladder_hash
        or design.endpoint_action_hashes != receipt.endpoint_action_hashes
        or design.chart_identity_hash != receipt.chart_identity_hash
        or ladder.current_mask_hash != receipt.current_mask_hash
        or ladder.proposal_generation != receipt.proposal_generation
        or ladder.route_count != receipt.route_count
        or ladder.route_axis_hash != receipt.route_axis_hash
        or ladder.raw_logit_stream_source_hash
        != receipt.raw_logit_stream_source_hash
        or ladder.sampling_law_hash != receipt.sampling_law_hash
        or tuple(
            name for name, _axis in design.certification_context_ids_by_stratum
        )
        != receipt.stratum_names
        or design.doses != receipt.doses
        or design.raw_logit_interval_verifier_source_hash
        != receipt.verifier_source_hash
    ):
        raise ValueError(
            "AlexNet analytic bounds and directional secant design differ"
        )
    if receipt.verifier_source_hash != _source_sha256():
        raise RuntimeError(
            "AlexNet raw-logit verifier source differs from the analytic receipt"
        )
    return receipt, receipt_hash, design_hash


def _strict_outward_nonnegative(value: float, *, name: str) -> float:
    checked = _finite(value, name=name, nonnegative=True)
    result = math.nextafter(checked, math.inf)
    if not math.isfinite(result):
        raise FloatingPointError(f"{name} cannot be outward-enclosed")
    return result


def directional_observation_bounds_from_v1_alexnet_fqr_receipt(
    *,
    result: V1AlexNetFQRBranchAnalyticBounds,
    design: V1DirectionalSecantDesign,
) -> V1DirectionalSecantObservationBounds:
    """Adapt a pre-C AlexNet proof to the native secant provider's ABI.

    The receipt's exact-linear ``R=0`` statement is a mathematical property of
    the registered direction.  The executor-facing observation support instead
    uses the independently derived raw interval remainder and one outward ulp;
    it therefore never asks mathematical zero to impersonate floating-point
    measurement error.  Statistical F/Q/R certification retains its separate
    deterministic numerical-uncertainty owner.
    """

    receipt, receipt_hash, design_hash = _assert_directional_adapter_coordinates(
        result=result,
        design=design,
    )
    raw_candidates = [
        abs(receipt.parent_raw_target_logit_interval[0]),
        abs(receipt.parent_raw_target_logit_interval[1]),
    ]
    for row in receipt.relaxed_logit_receipts:
        for item in row:
            raw_candidates.extend(
                (
                    abs(item.raw_target_logit_lower),
                    abs(item.raw_target_logit_upper),
                )
            )
    raw_upper = _strict_outward_nonnegative(
        max(raw_candidates),
        name="raw-logit observation bound",
    )
    f_rows = np.asarray(
        receipt.corrected_f_upper_by_stratum_endpoint,
        dtype=np.float64,
    )
    corrected = np.asarray(
        tuple(
            _strict_outward_nonnegative(
                float(value),
                name="corrected projection observation bound",
            )
            for value in np.max(f_rows, axis=0)
        ),
        dtype=np.float64,
    )
    jacobian = np.asarray(
        tuple(
            _strict_outward_nonnegative(
                value,
                name="literal-J projection observation bound",
            )
            for value in receipt.literal_jacobian_projection_upper_by_endpoint
        ),
        dtype=np.float64,
    )
    remainder = np.asarray(
        tuple(
            tuple(
                _strict_outward_nonnegative(
                    value,
                    name="directional remainder observation bound",
                )
                for value in row
            )
            for row in receipt.raw_interval_remainder_upper_by_endpoint_and_dose
        ),
        dtype=np.float64,
    )
    source_hash = content_hash(
        {
            "algorithm": (
                "alexnet_pre_C_analytic_receipt_to_directional_observation_"
                "support_v1"
            ),
            "analytic_receipt_hash": receipt_hash,
            "secant_design_hash": design_hash,
            "endpoint_action_hashes": receipt.endpoint_action_hashes,
            "raw_logit_absolute_upper": raw_upper,
            "corrected_projection_absolute_upper_by_endpoint": corrected,
            "jacobian_projection_absolute_upper_by_endpoint": jacobian,
            "remainder_absolute_upper_by_endpoint_and_dose": remainder,
            "mathematical_exact_linear_R_kept_separate": True,
            "binary64_observation_support_outward_ulps": True,
            "registered_before_C_outcomes": True,
        },
        prefix=_OBSERVATION_BOUND_ADAPTER_PREFIX,
    )
    return V1DirectionalSecantObservationBounds(
        design_hash=design_hash,
        endpoint_action_hashes=receipt.endpoint_action_hashes,
        raw_logit_absolute_upper=raw_upper,
        corrected_projection_absolute_upper_by_endpoint=corrected,
        jacobian_projection_absolute_upper_by_endpoint=jacobian,
        remainder_absolute_upper_by_endpoint_and_dose=remainder,
        bound_source_hash=source_hash,
    )


def _reported_binary64_center_radii(values: np.ndarray) -> np.ndarray:
    directions = np.where(values >= 0.0, np.inf, -np.inf)
    neighbor = np.nextafter(values, directions)
    radii = np.abs(neighbor - values)
    radii[values == 0.0] = np.nextafter(0.0, 1.0)
    if not np.isfinite(radii).all() or bool(np.any(radii < 0.0)):
        raise FloatingPointError("binary64 center enclosure is unusable")
    return radii


def build_v1_alexnet_fqr_raw_logit_interval_verifier(
    *,
    result: V1AlexNetFQRBranchAnalyticBounds,
    design: V1DirectionalSecantDesign,
) -> RawLogitIntervalVerifier:
    """Build the exact pre-C interval checker for one branch/dose axis.

    This verifier checks reported centers against the independently registered
    network intervals and returns one-ulp radii around those binary64 centers.
    The radii certify the reported numerical centers only; they deliberately do
    not claim an exact-real forward-error theorem.
    """

    receipt, receipt_hash, design_hash = _assert_directional_adapter_coordinates(
        result=result,
        design=design,
    )
    endpoint_count = len(receipt.endpoint_action_hashes)
    parent_interval = receipt.parent_raw_target_logit_interval
    relaxed_intervals = np.asarray(
        tuple(
            tuple(
                (
                    item.raw_target_logit_lower,
                    item.raw_target_logit_upper,
                )
                for item in row
            )
            for row in receipt.relaxed_logit_receipts
        ),
        dtype=np.float64,
    )
    if relaxed_intervals.shape != (endpoint_count, 2, 2):
        raise ValueError("AlexNet relaxed-logit interval axis is malformed")
    relaxed_receipt_hashes = tuple(
        tuple(item.receipt_hash for item in row)
        for row in receipt.relaxed_logit_receipts
    )
    source_hash = receipt.verifier_source_hash

    def verify(
        parent_centers: np.ndarray,
        dosed_centers: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        parent = np.asarray(parent_centers, dtype=np.float64)
        dosed = np.asarray(dosed_centers, dtype=np.float64)
        if (
            parent.ndim != 1
            or dosed.shape != (parent.shape[0], endpoint_count, 2)
            or not np.isfinite(parent).all()
            or not np.isfinite(dosed).all()
        ):
            raise ValueError("AlexNet raw-logit centers have invalid axes")
        if (
            bool(np.any(parent < parent_interval[0]))
            or bool(np.any(parent > parent_interval[1]))
            or bool(np.any(dosed < relaxed_intervals[None, :, :, 0]))
            or bool(np.any(dosed > relaxed_intervals[None, :, :, 1]))
        ):
            raise ValueError(
                "reported raw-logit center exceeded its pre-C analytic interval"
            )
        if source_hash != _source_sha256():
            raise RuntimeError(
                "AlexNet raw-logit verifier source changed after registration"
            )
        parent_radius = _reported_binary64_center_radii(parent)
        dosed_radius = _reported_binary64_center_radii(dosed)
        proof_hash = content_hash(
            {
                "algorithm": (
                    "alexnet_pre_C_interval_checked_reported_binary64_center_"
                    "enclosure_v1"
                ),
                "source_hash": source_hash,
                "analytic_receipt_hash": receipt_hash,
                "secant_design_hash": design_hash,
                "parent_interval": parent_interval,
                "relaxed_logit_receipt_hashes": relaxed_receipt_hashes,
                "parent_centers": parent,
                "parent_radii": parent_radius,
                "dosed_centers": dosed,
                "dosed_radii": dosed_radius,
                "claim": (
                    "pre_C_analytic_interval_membership_plus_one_ulp_"
                    "reported_binary64_center_enclosure_only"
                ),
                "exact_real_forward_error_claimed": False,
            },
            prefix=_RAW_LOGIT_VERIFIER_PROOF_PREFIX,
        )
        return parent_radius, dosed_radius, proof_hash

    return verify


def _selected_action_gate_by_layer(
    *,
    selected_routes: np.ndarray,
    parent_gates: dict[str, np.ndarray],
    route_axis: DeclaredRouteAxis,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for block in route_axis.route_blocks:
        layer = str(block.layer_id)
        selected = selected_routes[int(block.offset) : int(block.stop)].reshape(block.shape)
        if bool(np.any(np.logical_and(selected, np.logical_not(parent_gates[layer])))):
            raise ValueError("frozen action attempts to remove a non-live parent route")
        result[layer] = np.logical_and(selected, parent_gates[layer])
    return result


def _relaxed_gates(
    *,
    parent_gates: dict[str, np.ndarray],
    selected_gate_by_layer: dict[str, np.ndarray],
    dose: float,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for layer, parent in parent_gates.items():
        selected = selected_gate_by_layer[layer]
        # The registered doses and binary parent gates are all exactly
        # representable in float32.  Keeping the transient gate field in
        # float32 halves the 57M-route host workspace; the interval engine
        # promotes it to its outward float64 arithmetic before contraction.
        relaxed = parent.astype(np.float32, copy=True)
        relaxed[selected] = 1.0 - dose
        if (
            not np.all(np.isfinite(relaxed))
            or bool(np.any(relaxed < 0.0))
            or bool(np.any(relaxed > 1.0))
        ):
            raise FloatingPointError("relaxed action left the closed gate interval")
        result[layer] = relaxed
    return result


def _literal_jacobian_projection_upper(
    *,
    model: Any,
    route_axis: DeclaredRouteAxis,
    selected_gate_by_layer: dict[str, np.ndarray],
    source_interval_by_layer: dict[str, tuple[Any, Any]],
    preactivation_interval_by_layer: dict[str, tuple[Any, Any]],
    cotangent_by_layer: dict[str, Any],
    profile: Any,
    output_row_chunk: int,
) -> tuple[float, tuple[str, ...]]:
    layer_bounds: list[float] = []
    nonzero_weight_layers: list[str] = []
    for block in route_axis.route_blocks:
        layer = str(block.layer_id)
        selected_gate = selected_gate_by_layer[layer]
        if not bool(np.any(selected_gate)):
            layer_bounds.append(0.0)
            continue
        path = _env._AFFINE_PATH_BY_LAYER[layer]
        module = model.get_submodule(path)
        selected_tensor = _env.torch.as_tensor(
            selected_gate,
            dtype=_env.torch.bool,
            device=module.weight.device,
        )
        if bool((module.weight.detach()[selected_tensor] != 0.0).any().item()):
            nonzero_weight_layers.append(layer)
        source_lower, source_upper = source_interval_by_layer[layer]
        if block.route_kind == "conv_kernel_weight":
            drive_lower, drive_upper = _env._conv_action_drive_interval_bound(
                source_lower=source_lower,
                source_upper=source_upper,
                module=module,
                selected_gate=selected_gate,
                profile=profile,
            )
        elif block.route_kind == "linear_weight":
            drive_lower, drive_upper = _env._linear_action_drive_interval_bound(
                source_lower=source_lower,
                source_upper=source_upper,
                module=module,
                selected_gate=selected_gate,
                profile=profile,
                output_row_chunk=output_row_chunk,
            )
        else:  # pragma: no cover - layout admission closes the route kinds.
            raise RuntimeError("unsupported admitted AlexNet route kind")
        drive_absolute = _env._interval_absolute_bound(drive_lower, drive_upper)
        _preactivation_lower, preactivation_upper = preactivation_interval_by_layer[layer]
        # Strict ReLU subgradient: if the registered upper endpoint is not
        # positive, no admitted context can have derivative one.  Otherwise
        # one is the exact finite analytic supremum.  This is intentionally
        # distinct from the Aumann--Shapley path fraction used by F.
        indicator_upper = (preactivation_upper > 0.0).to(dtype=_env.torch.float64)
        layer_bounds.append(
            _env._outward_nonnegative_product_sum(
                (
                    cotangent_by_layer[layer],
                    indicator_upper,
                    drive_absolute,
                ),
                profile=profile,
                name=f"{layer} literal-J action projection",
            )
        )
    return (
        _env._outward_nonnegative_scalar_sum(
            tuple(layer_bounds),
            profile=profile,
            name="cross-layer literal-J action projection",
        ),
        tuple(nonzero_weight_layers),
    )


def _exact_linear_proof(
    *,
    nonzero_weight_layers: tuple[str, ...],
) -> str:
    # A zero selected drive is the one exact-linear case that also closes in
    # the registered floating execution: every relaxed effective weight is
    # bit-identical to its parent value and the literal tangent is exactly
    # zero.  Merely proving a fixed ReLU regime would prove real-arithmetic
    # linearity but not a zero *numerically enclosed* remainder, because the
    # parent, relaxed, and VJP contractions round independently.
    if not nonzero_weight_layers:
        return EXACT_LINEAR_PROOF_ZERO_SELECTED_WEIGHT
    return EXACT_LINEAR_PROOF_NONE


def _assert_semantics(
    *,
    positive_removal_sign: str,
    corrected_evidence_equation: str,
    literal_jacobian_equation: str,
    relu_kink_policy: str,
    corrected_rows_used_as_literal_jacobian: bool,
) -> None:
    if type(corrected_rows_used_as_literal_jacobian) is not bool:
        raise TypeError("corrected-row/J alias marker must be an exact bool")
    if corrected_rows_used_as_literal_jacobian:
        raise PermissionError("corrected attribution rows may not impersonate literal J")
    if (
        positive_removal_sign != POSITIVE_REMOVAL_SIGN
        or corrected_evidence_equation != CORRECTED_EVIDENCE_EQUATION
        or literal_jacobian_equation != LITERAL_JACOBIAN_EQUATION
        or relu_kink_policy != RELU_KINK_POLICY
    ):
        raise ValueError("F/Q/R evidence, Jacobian, sign, or ReLU semantics are invalid")


def build_v1_alexnet_fqr_branch_analytic_bounds(
    *,
    model: Any,
    route_axis: DeclaredRouteAxis,
    current_mask: HardMask,
    query: QuerySpec,
    input_box: V1RegisteredNormalizedInputBox,
    corrected_envelope_result: V1AlexNetCorrectedAttributionEnvelopeResult,
    ladder: V1FrozenCandidateLadder,
    branch: V1RelaxedKKTBranch,
    input_id: str,
    chart_identity_hash: str,
    live_chart_identity_hash: str,
    live_proposal_generation: int,
    corrected_work_statistical_design_hash: str,
    literal_jacobian_statistical_design_hash: str,
    fd_rank_uncertainty_set_hash: str,
    kkt_numerical_uncertainty_set_hash: str,
    numerical_enclosure_policy_hash: str,
    doses: tuple[float, float] = DIRECTIONAL_SECANT_DOSES,
    positive_removal_sign: str = POSITIVE_REMOVAL_SIGN,
    corrected_evidence_equation: str = CORRECTED_EVIDENCE_EQUATION,
    literal_jacobian_equation: str = LITERAL_JACOBIAN_EQUATION,
    relu_kink_policy: str = RELU_KINK_POLICY,
    corrected_rows_used_as_literal_jacobian: bool = False,
) -> V1AlexNetFQRBranchAnalyticBounds:
    """Build finite analytic observation bounds for one frozen branch ladder.

    The caller supplies the current live chart and generation separately from
    the frozen coordinates.  A stale ladder therefore refuses before any
    relaxed propagation.  No observed C row or downstream outcome is accepted.
    """

    _env._require_torch()
    _assert_semantics(
        positive_removal_sign=positive_removal_sign,
        corrected_evidence_equation=corrected_evidence_equation,
        literal_jacobian_equation=literal_jacobian_equation,
        relu_kink_policy=relu_kink_policy,
        corrected_rows_used_as_literal_jacobian=(corrected_rows_used_as_literal_jacobian),
    )
    if type(branch) is not V1RelaxedKKTBranch:
        raise TypeError("branch must use the exact KKT branch enum")
    if type(corrected_envelope_result) is not V1AlexNetCorrectedAttributionEnvelopeResult:
        raise TypeError("F/Q/R bounds require the exact AlexNet corrected envelope")
    if type(ladder) is not V1FrozenCandidateLadder:
        raise TypeError("F/Q/R bounds require a typed frozen ladder")
    if not ladder.is_packed:
        raise ValueError("canonical AlexNet F/Q/R bounds require packed actions")
    dose_values = tuple(_finite(value, name="dose") for value in doses)
    if dose_values != DIRECTIONAL_SECANT_DOSES:
        raise ValueError("canonical AlexNet F/Q/R doses are exactly one half and one")
    requested_doses = (dose_values[0], dose_values[1])
    input_identity = _identity(input_id, name="input id")
    chart = _identity(chart_identity_hash, name="frozen chart identity hash")
    live_chart = _identity(live_chart_identity_hash, name="live chart identity hash")
    generation = _nonnegative_int(live_proposal_generation, name="live proposal generation")
    if chart != live_chart:
        raise ValueError("frozen ladder chart is stale relative to the live chart")
    for name, value in (
        ("corrected work statistical design hash", corrected_work_statistical_design_hash),
        ("literal Jacobian statistical design hash", literal_jacobian_statistical_design_hash),
        ("FD-rank uncertainty-set hash", fd_rank_uncertainty_set_hash),
        ("KKT numerical uncertainty-set hash", kkt_numerical_uncertainty_set_hash),
        ("numerical enclosure policy hash", numerical_enclosure_policy_hash),
    ):
        _identity(value, name=name)

    corrected_envelope_result.assert_integrity()
    ladder_hash = ladder.assert_integrity()
    if ladder.proposal_generation != generation:
        raise ValueError("frozen ladder proposal generation is stale")
    parent_mask_hash = _mask_hash(current_mask, name="F/Q/R parent mask")
    if (
        ladder.current_mask_hash != parent_mask_hash
        or type(ladder.parent_mask) is not HardMask
        or _mask_hash(ladder.parent_mask, name="F/Q/R ladder parent") != parent_mask_hash
    ):
        raise ValueError("frozen packed ladder names another parent mask")

    envelope_certificate = corrected_envelope_result.certificate
    envelope_receipt = corrected_envelope_result.derivation_receipt
    shared_coordinates = (
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
    if any(
        getattr(ladder, name) != getattr(envelope_certificate, name) for name in shared_coordinates
    ):
        raise ValueError("frozen ladder differs from the corrected-envelope chart")

    layout = admit_v1_alexnet_envelope_layout(
        model=model,
        route_axis=route_axis,
        current_mask=current_mask,
        query=query,
        model_identity_hash=envelope_receipt.model_identity_hash,
        expected_model_state_hash=envelope_receipt.model_state_hash,
        preprocessing_hash=envelope_receipt.preprocessing_hash,
        input_box=input_box,
    )
    if (
        layout.route_axis_content_hash != envelope_receipt.route_axis_content_hash
        or layout.input_box_hash != envelope_receipt.input_box_hash
        or layout.graph_semantics_hash != envelope_receipt.graph_semantics_hash
        or layout.arithmetic_profile_hash != envelope_receipt.arithmetic_profile_hash
        or layout.model_state_hash != envelope_receipt.model_state_hash
    ):
        raise ValueError("live AlexNet layout differs from the corrected-envelope proof")

    # F has an independent, already-audited Aumann--Shapley owner.  Calling
    # that builder (rather than deriving F from the J path below) makes an
    # accidental corrected-row/J alias structurally impossible.
    f_certificate = build_v1_alexnet_action_drive_endpoint_envelope(
        model=model,
        route_axis=route_axis,
        current_mask=current_mask,
        query=query,
        input_box=input_box,
        envelope_result=corrected_envelope_result,
        ladder=ladder,
    )
    f_certificate_hash = f_certificate.assert_integrity()
    f_rows = tuple(
        tuple(float(value) for value in row)
        for row in f_certificate.endpoint_absolute_upper_bounds_by_stratum
    )

    profile = _env._arithmetic_profile(model)
    parent_gates = _env._split_gates(current_mask, route_axis)
    output_row_chunk = _env._AFFINE_BOUND_OUTPUT_ROW_CHUNK
    with _env.torch.no_grad():
        (
            parent_source_by_layer,
            parent_source_interval_by_layer,
            parent_preactivation_by_layer,
            _parent_destination_shapes,
            _parent_forward_factors,
            parent_pool_input_shapes,
            parent_avgpool_input_shape,
            parent_final_hidden_interval,
        ) = _env._forward_bounds(
            model=model,
            gates=parent_gates,
            input_box=input_box,
            profile=profile,
            output_row_chunk=output_row_chunk,
        )
        cotangent_by_layer, _backward_factors = _env._backward_bounds(
            model=model,
            gates=parent_gates,
            target_class_id=layout.target_class_id,
            source_by_layer=parent_source_by_layer,
            pool_input_shapes=parent_pool_input_shapes,
            avgpool_input_shape=parent_avgpool_input_shape,
            profile=profile,
            output_row_chunk=output_row_chunk,
        )
        parent_lower, parent_upper, _parent_readout_factor = _readout_interval_from_forward(
            model=model,
            final_hidden_interval=parent_final_hidden_interval,
            target_class_id=layout.target_class_id,
            profile=profile,
        )

        selected_routes = np.zeros(route_axis.route_count, dtype=np.bool_)
        j_axis: list[float] = []
        q_axis: list[float] = []
        raw_r_axis: list[tuple[float, float]] = []
        r_axis: list[tuple[float, float]] = []
        exact_proofs: list[str] = []
        relaxed_receipt_rows: list[tuple[V1AlexNetFQRRelaxedLogitIntervalReceipt, ...]] = []
        endpoint_mask_hashes: list[str] = []
        verifier_source_hash = _source_sha256()
        interval_engine_source_hash = _env._source_sha256()

        for action in ladder.actions:
            for shell in ladder.iter_shell_index_chunks(endpoint_index=action.endpoint_index):
                if shell.ndim != 1 or bool(np.any(selected_routes[shell])):
                    raise ValueError("packed ladder shells are not disjoint and nested")
                selected_routes[shell] = True
            selected_gate_by_layer = _selected_action_gate_by_layer(
                selected_routes=selected_routes,
                parent_gates=parent_gates,
                route_axis=route_axis,
            )
            j_upper, nonzero_weight_layers = _literal_jacobian_projection_upper(
                model=model,
                route_axis=route_axis,
                selected_gate_by_layer=selected_gate_by_layer,
                source_interval_by_layer=parent_source_interval_by_layer,
                preactivation_interval_by_layer=parent_preactivation_by_layer,
                cotangent_by_layer=cotangent_by_layer,
                profile=profile,
                output_row_chunk=output_row_chunk,
            )
            j_axis.append(j_upper)
            q_axis.append(_outward_square_upper(j_upper))
            endpoint_mask = action.resulting_mask
            endpoint_mask_hash = _mask_hash(endpoint_mask, name="F/Q/R endpoint mask")
            endpoint_mask_hashes.append(endpoint_mask_hash)
            relaxed_receipts: list[V1AlexNetFQRRelaxedLogitIntervalReceipt] = []
            raw_r_values: list[float] = []
            for dose_index, dose in enumerate(requested_doses):
                relaxed_gates = _relaxed_gates(
                    parent_gates=parent_gates,
                    selected_gate_by_layer=selected_gate_by_layer,
                    dose=dose,
                )
                (
                    _relaxed_source,
                    _relaxed_source_interval,
                    relaxed_preactivation,
                    _relaxed_destination_shapes,
                    _relaxed_forward_factors,
                    _relaxed_pool_shapes,
                    _relaxed_avgpool_shape,
                    relaxed_final_hidden_interval,
                ) = _env._forward_bounds(
                    model=model,
                    gates=relaxed_gates,
                    input_box=input_box,
                    profile=profile,
                    output_row_chunk=output_row_chunk,
                )
                relaxed_lower, relaxed_upper, relaxed_readout_factor = (
                    _readout_interval_from_forward(
                        model=model,
                        final_hidden_interval=relaxed_final_hidden_interval,
                        target_class_id=layout.target_class_id,
                        profile=profile,
                    )
                )
                # The preactivation intervals remain part of the outward
                # propagation but do not alone certify an exact-zero numeric
                # remainder; see ``_exact_linear_proof``.
                del relaxed_preactivation
                gate_path_hash = _relaxed_gate_path_hash(
                    parent_mask_hash=parent_mask_hash,
                    endpoint_mask_hash=endpoint_mask_hash,
                    action_hash=action.action_hash,
                    ladder_hash=ladder_hash,
                    chart_identity_hash=chart,
                    proposal_generation=generation,
                    dose=dose,
                )
                relaxed_receipt = V1AlexNetFQRRelaxedLogitIntervalReceipt(
                    endpoint_index=action.endpoint_index,
                    action_hash=action.action_hash,
                    dose_index=dose_index,
                    dose=dose,
                    parent_mask_hash=parent_mask_hash,
                    endpoint_mask_hash=endpoint_mask_hash,
                    ladder_hash=ladder_hash,
                    chart_identity_hash=chart,
                    proposal_generation=generation,
                    route_count=layout.route_count,
                    route_axis_hash=layout.route_axis_hash,
                    model_identity_hash=layout.model_identity_hash,
                    model_state_hash=layout.model_state_hash,
                    query_contract_hash=layout.query_contract_hash,
                    preprocessing_hash=layout.preprocessing_hash,
                    input_box_hash=layout.input_box_hash,
                    layout_hash=layout.layout_hash,
                    graph_semantics_hash=layout.graph_semantics_hash,
                    arithmetic_profile_hash=layout.arithmetic_profile_hash,
                    relaxed_gate_path_hash=gate_path_hash,
                    raw_target_logit_lower=relaxed_lower,
                    raw_target_logit_upper=relaxed_upper,
                    readout_rounding_factor=relaxed_readout_factor,
                    interval_engine_source_hash=interval_engine_source_hash,
                    verifier_source_hash=verifier_source_hash,
                    full_dose_equals_endpoint_hard_mask=(dose == 1.0),
                )
                relaxed_receipts.append(relaxed_receipt)
                raw_r_values.append(
                    _interval_remainder_upper(
                        parent_interval=(parent_lower, parent_upper),
                        relaxed_interval=(relaxed_lower, relaxed_upper),
                        literal_jacobian_projection_upper=j_upper,
                        dose=dose,
                    )
                )
                # Bound peak memory by releasing the complete relaxed chart
                # before materializing the next endpoint/dose gate field.
                del (
                    relaxed_gates,
                    _relaxed_source,
                    _relaxed_source_interval,
                    _relaxed_destination_shapes,
                    _relaxed_forward_factors,
                    _relaxed_pool_shapes,
                    _relaxed_avgpool_shape,
                    relaxed_final_hidden_interval,
                )
            exact_proof = _exact_linear_proof(
                nonzero_weight_layers=nonzero_weight_layers,
            )
            exact_proofs.append(exact_proof)
            raw_tuple = tuple(raw_r_values)
            if len(raw_tuple) != 2:  # pragma: no cover - dose invariant.
                raise RuntimeError("directional-remainder dose axis changed")
            raw_r_axis.append((raw_tuple[0], raw_tuple[1]))
            r_axis.append(
                (0.0, 0.0)
                if exact_proof != EXACT_LINEAR_PROOF_NONE
                else (raw_tuple[0], raw_tuple[1])
            )
            relaxed_receipt_rows.append(tuple(relaxed_receipts))

    # Full dose must be byte-equivalent to each exact hard endpoint.  The
    # public binary certificate is intentionally not recomputed here (which
    # would double every full forward); exact equality is covered by the
    # focused compatibility test against that independent public builder.
    if _source_sha256() != verifier_source_hash:
        raise RuntimeError("AlexNet F/Q/R verifier source changed during derivation")
    if _env._source_sha256() != interval_engine_source_hash:
        raise RuntimeError("AlexNet interval-engine source changed during derivation")
    if state_dict_digest(model) != layout.model_state_hash:
        raise RuntimeError("AlexNet model state changed during F/Q/R derivation")
    if _mask_hash(current_mask, name="closing F/Q/R parent mask") != parent_mask_hash:
        raise RuntimeError("parent mask changed during F/Q/R derivation")
    if ladder.assert_integrity() != ladder_hash:
        raise RuntimeError("frozen ladder changed during F/Q/R derivation")
    closing_layout = admit_v1_alexnet_envelope_layout(
        model=model,
        route_axis=route_axis,
        current_mask=current_mask,
        query=query,
        model_identity_hash=layout.model_identity_hash,
        expected_model_state_hash=layout.model_state_hash,
        preprocessing_hash=layout.preprocessing_hash,
        input_box=input_box,
    )
    if closing_layout != layout:
        raise RuntimeError("AlexNet layout changed during F/Q/R derivation")

    receipt = V1AlexNetFQRBoundDerivationReceipt(
        input_id=input_identity,
        branch=branch,
        campaign_contract_hash=ladder.campaign_contract_hash,
        model_identity_hash=layout.model_identity_hash,
        model_state_hash=layout.model_state_hash,
        query_contract_hash=layout.query_contract_hash,
        preprocessing_hash=layout.preprocessing_hash,
        current_mask_hash=parent_mask_hash,
        raw_logit_stream_source_hash=ladder.raw_logit_stream_source_hash,
        sampling_law_hash=ladder.sampling_law_hash,
        proposal_generation=generation,
        route_count=layout.route_count,
        route_axis_hash=layout.route_axis_hash,
        route_axis_content_hash=layout.route_axis_content_hash,
        chart_identity_hash=chart,
        ladder_hash=ladder_hash,
        endpoint_action_hashes=ladder.action_hashes,
        endpoint_packed_action_hashes=tuple(
            action.packed_removal_action.action_hash
            for action in ladder.actions
            if action.packed_removal_action is not None
        ),
        endpoint_mask_hashes=tuple(endpoint_mask_hashes),
        doses=requested_doses,
        stratum_names=tuple(envelope_certificate.stratum_names),
        population_stratum_counts=tuple(envelope_certificate.population_stratum_counts),
        sealed_population_order_hashes=tuple(envelope_certificate.sealed_population_order_hashes),
        input_box_hash=layout.input_box_hash,
        layout_hash=layout.layout_hash,
        graph_semantics_hash=layout.graph_semantics_hash,
        arithmetic_profile_hash=layout.arithmetic_profile_hash,
        parent_raw_target_logit_interval=(parent_lower, parent_upper),
        corrected_f_upper_by_stratum_endpoint=f_rows,
        literal_jacobian_projection_upper_by_endpoint=tuple(j_axis),
        q_upper_by_endpoint=tuple(q_axis),
        raw_interval_remainder_upper_by_endpoint_and_dose=tuple(raw_r_axis),
        r_upper_by_endpoint_and_dose=tuple(r_axis),
        exact_linear_proof_by_endpoint=tuple(exact_proofs),
        relaxed_logit_receipts=tuple(relaxed_receipt_rows),
        corrected_work_endpoint_certificate_hash=f_certificate_hash,
        corrected_work_route_envelope_certificate_hash=(envelope_certificate.certificate_hash),
        corrected_work_route_envelope_proof_hash=(envelope_receipt.receipt_hash),
        corrected_work_statistical_design_hash=(corrected_work_statistical_design_hash),
        literal_jacobian_statistical_design_hash=(literal_jacobian_statistical_design_hash),
        fd_rank_uncertainty_set_hash=fd_rank_uncertainty_set_hash,
        kkt_numerical_uncertainty_set_hash=kkt_numerical_uncertainty_set_hash,
        numerical_enclosure_policy_hash=numerical_enclosure_policy_hash,
        interval_engine_source_hash=interval_engine_source_hash,
        verifier_source_hash=verifier_source_hash,
        schema=ALEXNET_FQR_BOUND_RECEIPT_SCHEMA_V2,
    )
    # The derivation receipt keeps exact-real mathematical bounds (including
    # an exact-linear R=0 proof) distinct from the support of the conservative
    # binary64 observations consumed by sequential C inference.  The latter
    # includes the registered numerical enclosure and is therefore the only
    # valid support array for the accumulator.
    f_array, q_array, r_array = _sequential_observation_support_arrays(receipt)
    analytic_input = V1FQRBranchAnalyticInput(
        input_id=input_identity,
        branch=branch,
        analytic_f_upper_by_stratum_endpoint=f_array,
        analytic_q_upper_by_stratum_endpoint=q_array,
        analytic_r_upper_by_stratum_endpoint_and_dose=r_array,
        corrected_work_statistical_design_hash=(corrected_work_statistical_design_hash),
        literal_jacobian_statistical_design_hash=(literal_jacobian_statistical_design_hash),
        corrected_work_bound_source_hash=receipt.corrected_work_bound_source_hash,
        literal_jacobian_bound_source_hash=(receipt.literal_jacobian_bound_source_hash),
        directional_remainder_bound_source_hash=(receipt.directional_remainder_bound_source_hash),
        fd_rank_uncertainty_set_hash=fd_rank_uncertainty_set_hash,
        kkt_numerical_uncertainty_set_hash=kkt_numerical_uncertainty_set_hash,
        numerical_enclosure_policy_hash=numerical_enclosure_policy_hash,
    )
    return V1AlexNetFQRBranchAnalyticBounds(
        analytic_input=analytic_input,
        derivation_receipt=receipt,
        corrected_work_endpoint_certificate=f_certificate,
    )


def assert_v1_alexnet_fqr_full_dose_binary_compatibility(
    *,
    result: V1AlexNetFQRBranchAnalyticBounds,
    endpoint_index: int,
    binary_certificate: V1AlexNetRawTargetLogitIntervalCertificate,
) -> None:
    """Assert exact eta=1 compatibility with the legacy hard-mask certificate."""

    if type(result) is not V1AlexNetFQRBranchAnalyticBounds:
        raise TypeError("binary compatibility requires the exact F/Q/R result")
    if type(binary_certificate) is not V1AlexNetRawTargetLogitIntervalCertificate:
        raise TypeError("binary compatibility requires the exact hard certificate")
    result.assert_integrity()
    binary_certificate.assert_integrity()
    endpoint = _positive_int(endpoint_index, name="endpoint index")
    receipt = result.derivation_receipt
    if endpoint > len(receipt.endpoint_action_hashes):
        raise ValueError("binary compatibility endpoint left the ladder")
    full = receipt.relaxed_logit_receipts[endpoint - 1][1]
    if (
        full.dose != 1.0
        or not full.full_dose_equals_endpoint_hard_mask
        or binary_certificate.current_mask_hash != full.endpoint_mask_hash
        or binary_certificate.route_axis_hash != full.route_axis_hash
        or binary_certificate.route_count != full.route_count
        or binary_certificate.model_identity_hash != full.model_identity_hash
        or binary_certificate.model_state_hash != full.model_state_hash
        or binary_certificate.query_contract_hash != full.query_contract_hash
        or binary_certificate.preprocessing_hash != full.preprocessing_hash
        or binary_certificate.input_box_hash != full.input_box_hash
        or binary_certificate.graph_semantics_hash != full.graph_semantics_hash
        or binary_certificate.arithmetic_profile_hash != full.arithmetic_profile_hash
        or binary_certificate.raw_target_logit_lower != full.raw_target_logit_lower
        or binary_certificate.raw_target_logit_upper != full.raw_target_logit_upper
        or binary_certificate.readout_rounding_factor != full.readout_rounding_factor
    ):
        raise ValueError("eta=1 relaxed interval differs from the binary certificate")


__all__ = [
    "ALEXNET_FQR_AUTHORITY_SCOPE",
    "ALEXNET_FQR_BOUND_ALGORITHM_ID",
    "ALEXNET_FQR_BOUND_RECEIPT_SCHEMA",
    "ALEXNET_FQR_BOUND_RECEIPT_SCHEMA_V2",
    "ALEXNET_FQR_BOUND_RESULT_SCHEMA",
    "ALEXNET_FQR_LITERAL_J_ALGORITHM_ID",
    "ALEXNET_FQR_NUMERICAL_ENCLOSURE_ID",
    "ALEXNET_FQR_RELAXED_GATE_PATH_ID",
    "ALEXNET_FQR_RELAXED_LOGIT_RECEIPT_SCHEMA",
    "ALEXNET_FQR_REMAINDER_ALGORITHM_ID",
    "EXACT_LINEAR_PROOF_NONE",
    "EXACT_LINEAR_PROOF_ZERO_SELECTED_WEIGHT",
    "V1AlexNetFQRBoundDerivationReceipt",
    "V1AlexNetFQRBranchAnalyticBounds",
    "V1AlexNetFQRRelaxedLogitIntervalReceipt",
    "assert_v1_alexnet_fqr_full_dose_binary_compatibility",
    "build_v1_alexnet_fqr_raw_logit_interval_verifier",
    "build_v1_alexnet_fqr_branch_analytic_bounds",
    "directional_observation_bounds_from_v1_alexnet_fqr_receipt",
]
