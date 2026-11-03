"""Shared sequential calibrated-BCE contracts for current v0 and replay.

The default interval is deliberately modest in its authority: a stratified
Student-t interval under a predictable per-look error schedule. Modern IID
execution spends over a countably unbounded look sequence so precision, rather
than a runtime horizon, determines sample size. Compatibility identities may
retain a finite planned-look ladder. Both remain exploratory under the
declared exchangeability and tail assumptions; neither is a distribution-free
campaign guarantee. The v5 v0 decision uses the scalar delta-method test and
reports its Fieller inversion; v4 remains available for historical replay.

Two estimands share this measurement foundation:

``v0_normalized_loss_span_per_connection_v1``
    A paired parent--candidate BCE increment divided by a positive dense-to-
    empty loss span and the exact number of physical circuit connections
    removed by that transition.

``baseline_excess_calibrated_bce_v1``
    The absolute calibrated BCE risk of one candidate compared with a locked
    dense-state risk certificate. This is retained for typed historical replay
    and requires no parent-mask oracle lane.

Two proposal-adjudication oracle sampling laws are admitted
(``ORACLE_SAMPLING_LAWS``):

``independent_empirical_iid_with_replacement_v1``
    Logical IID draws with replacement from the frozen empirical law; the
    per-look stratified Student-t interval carries no finite-population
    correction and the look sequence is countably unbounded.

``independent_sealed_without_replacement_class_stratified_v1``
    Each class stratum is a sealed uniform permutation of its finite training
    population revealed as a nested prefix (a uniform SRSWOR of adaptively
    chosen size).  The per-look interval applies the exact per-stratum finite
    population correction ``(1 - n_h / N_h)``; a stratum at census contributes
    zero sampling variance, so the decision resolves exactly once every
    stratum is exhausted (``sequential_calibrated_bce_resolution_v4``).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from typing import Any, ClassVar, Literal, Mapping, Sequence

import numpy as np
from scipy.special import gammaincinv
from mechanistic_systems.measurement.lazy_attribute import LazyAttribute

chi2 = LazyAttribute("scipy.stats", "chi2")
norm = LazyAttribute("scipy.stats", "norm")
student_t = LazyAttribute("scipy.stats", "t")

from mechanistic_systems.evidence import staged_alpha, student_t_interval


SEQUENTIAL_CALIBRATED_BCE_V2 = "sequential_calibrated_bce_resolution_v2"
SEQUENTIAL_CALIBRATED_BCE_V3 = "sequential_calibrated_bce_resolution_v3"
# v4: sealed without-replacement class-stratified oracle law with exact
# finite-population correction, census terminal, growth clamp to the remaining
# population, operational acquisition envelope, and decision receipt v4.
SEQUENTIAL_CALIBRATED_BCE_V4 = "sequential_calibrated_bce_resolution_v4"
SEQUENTIAL_CALIBRATED_BCE_V5 = "sequential_calibrated_bce_resolution_v5"
V0_NORMALIZED_LOSS_SPAN_PER_CONNECTION_V1 = "v0_normalized_loss_span_per_connection_v1"
BASELINE_EXCESS_CALIBRATED_BCE_V1 = "baseline_excess_calibrated_bce_v1"
SequentialBCEEstimandPolicy = Literal[
    "v0_normalized_loss_span_per_connection_v1",
    "baseline_excess_calibrated_bce_v1",
]

IID_WITH_REPLACEMENT_SAMPLING_ID = "independent_empirical_iid_with_replacement_v1"
SEALED_WOR_CLASS_STRATIFIED_SAMPLING_ID = (
    "independent_sealed_without_replacement_class_stratified_v1"
)
ORACLE_SAMPLING_LAWS = (
    IID_WITH_REPLACEMENT_SAMPLING_ID,
    SEALED_WOR_CLASS_STRATIFIED_SAMPLING_ID,
)
SEALED_RESPONSE_SAMPLING_ID = "sealed_uniform_without_replacement_v1"
PLANNED_LOOK_SCOPE = "per_decision_planned_looks_exploratory"
PRECISION_ONLY_IID_SCOPE = "per_decision_precision_only_iid_alpha_spending_exploratory"
STRATIFIED_T_METHOD = "stratified_satterthwaite_student_t_planned_looks_v1"
STRATIFIED_T_ALPHA_SPENDING_METHOD = (
    "stratified_satterthwaite_student_t_infinite_alpha_spending_v1"
)
STRATIFIED_T_FPC_METHOD = (
    "stratified_satterthwaite_student_t_fpc_planned_looks_v1"
)
STRATIFIED_T_FPC_ALPHA_SPENDING_METHOD = (
    "stratified_satterthwaite_student_t_fpc_infinite_alpha_spending_v1"
)
SEQUENTIAL_BCE_DECISION_RECEIPT_V4 = "sequential_bce_decision_receipt_v4"
SEQUENTIAL_BCE_DECISION_RECEIPT_V5 = "sequential_bce_decision_receipt_v5"
JOINT_DECISION_WALD_METHOD = (
    "joint_decision_wald_quadratic_alpha_spending_exploratory_v1"
)
SCALAR_DECISION_FIELLER_METHOD = (
    "scalar_delta_fieller_alpha_spending_exploratory_v1"
)
ANALYTIC_BALANCED_BINARY_CHANCE_RISK_V1 = (
    "analytic_balanced_binary_natural_log_bce_chance_risk_v1"
)


def _canonical(value: Any) -> Any:
    if is_dataclass(value):
        return _canonical(asdict(value))
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        return {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "bytes_sha256": hashlib.sha256(
                np.ascontiguousarray(array).tobytes()
            ).hexdigest(),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _content_hash(value: Any, *, prefix: str) -> str:
    payload = json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(prefix.encode("utf-8") + payload).hexdigest()


def sequential_bce_decision_receipt_content_hash(
    receipt_payload: Mapping[str, Any],
) -> str:
    """Hash a decision-receipt payload with its producer-owned contract."""

    payload = dict(receipt_payload)
    payload.pop("content_hash", None)
    return _content_hash(
        payload,
        prefix="sequential-bce-decision-receipt:",
    )


def _removed_connection_axis_hash(rows: Sequence[Sequence[str | int]]) -> str:
    payload = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(b"removed-circuit-connection-axis:" + payload).hexdigest()


def _normalized_weights(
    weights: Mapping[str, float] | Sequence[tuple[str, float]],
) -> tuple[tuple[str, float], ...]:
    rows = tuple(
        (str(key), float(value))
        for key, value in (weights.items() if isinstance(weights, Mapping) else weights)
    )
    if (
        not rows
        or len({key for key, _value in rows}) != len(rows)
        or any(not key for key, _value in rows)
        or any(not math.isfinite(value) or value <= 0.0 for _key, value in rows)
        or not math.isclose(
            sum(value for _key, value in rows),
            1.0,
            rel_tol=1e-10,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(
            "stratum weights must be unique, positive, finite, and sum to one"
        )
    return tuple(sorted(rows))


@dataclass(frozen=True, slots=True)
class SequentialCalibratedBCEResolutionSpec:
    """Hash-bound shared statistical and sampling contract."""

    confidence_level: float
    estimand_policy: SequentialBCEEstimandPolicy
    stratum_weights: tuple[tuple[str, float], ...] = (
        ("off_target", 0.5),
        ("on_target", 0.5),
    )
    maximum_planned_looks: int | None = 32
    precision_only_iid: bool = False
    calibration_sampling_id: str = IID_WITH_REPLACEMENT_SAMPLING_ID
    oracle_sampling_id: str = IID_WITH_REPLACEMENT_SAMPLING_ID
    response_sampling_id: str = SEALED_RESPONSE_SAMPLING_ID
    unresolved_action: str = "no_commit"
    inference_method: str = STRATIFIED_T_METHOD
    decision_interval_method: str = JOINT_DECISION_WALD_METHOD
    confidence_scope: str = PLANNED_LOOK_SCOPE

    def __post_init__(self) -> None:
        if not 0.5 < float(self.confidence_level) < 1.0:
            raise ValueError("confidence_level must lie in (0.5,1)")
        if self.estimand_policy not in {
            V0_NORMALIZED_LOSS_SPAN_PER_CONNECTION_V1,
            BASELINE_EXCESS_CALIBRATED_BCE_V1,
        }:
            raise ValueError("unknown sequential calibrated-BCE estimand")
        if self.precision_only_iid:
            if self.maximum_planned_looks is not None:
                raise ValueError(
                    "precision-only IID inference cannot declare a look horizon"
                )
            object.__setattr__(
                self,
                "confidence_scope",
                PRECISION_ONLY_IID_SCOPE,
            )
            object.__setattr__(
                self,
                "inference_method",
                STRATIFIED_T_ALPHA_SPENDING_METHOD,
            )
        elif self.maximum_planned_looks is None or int(self.maximum_planned_looks) <= 0:
            raise ValueError("maximum_planned_looks must be positive")
        if self.unresolved_action != "no_commit":
            raise ValueError("unresolved sequential BCE evidence must not commit")
        if self.inference_method not in {
            STRATIFIED_T_METHOD,
            STRATIFIED_T_ALPHA_SPENDING_METHOD,
        }:
            raise ValueError("the default shared BCE identity requires stratified t")
        if self.decision_interval_method not in {
            JOINT_DECISION_WALD_METHOD,
            SCALAR_DECISION_FIELLER_METHOD,
        }:
            raise ValueError("modern BCE decisions require the joint decision interval")
        if self.confidence_scope not in {
            PLANNED_LOOK_SCOPE,
            PRECISION_ONLY_IID_SCOPE,
        }:
            raise ValueError("unexpected confidence scope")
        for field_name in (
            "calibration_sampling_id",
            "oracle_sampling_id",
            "response_sampling_id",
        ):
            if not str(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be nonempty")
        object.__setattr__(
            self,
            "stratum_weights",
            _normalized_weights(self.stratum_weights),
        )

    @property
    def error_probability(self) -> float:
        return 1.0 - float(self.confidence_level)

    @property
    def planned_look_alpha(self) -> float:
        if self.precision_only_iid:
            return self.look_alpha(0)
        assert self.maximum_planned_looks is not None
        return staged_alpha(
            self.error_probability,
            planned_looks=int(self.maximum_planned_looks),
        )

    def look_alpha(
        self,
        look_index: int,
        *,
        error_probability: float | None = None,
    ) -> float:
        """Return the predictable error stake for one completed look.

        Precision-only IID execution uses the telescoping schedule
        ``alpha / ((k + 1) * (k + 2))``.  It therefore has no maximum look
        count while the total stake over every possible finite look remains
        at most ``alpha``.  The Student-t primitive remains explicitly
        assumption-dependent; the spending schedule removes a runtime knob,
        not that inferential qualification.
        """

        index = int(look_index)
        if index < 0:
            raise ValueError("look_index must be nonnegative")
        alpha = (
            self.error_probability
            if error_probability is None
            else float(error_probability)
        )
        if not 0.0 < float(alpha) < 1.0:
            raise ValueError("error probability must lie in (0,1)")
        if self.precision_only_iid:
            return float(alpha) / float((index + 1) * (index + 2))
        assert self.maximum_planned_looks is not None
        return staged_alpha(
            float(alpha),
            planned_looks=int(self.maximum_planned_looks),
        )

    @property
    def proposal_error_probability(self) -> float:
        """Compatibility alias for the sole decision-level error probability.

        Modern v0/v1 do not allocate independent error stakes to proposal,
        reference, or calibration components.  All uncertain components enter
        one joint accept/reject statistic at each predictable look.
        """

        return self.error_probability

    @property
    def proposal_confidence_level(self) -> float:
        return 1.0 - self.proposal_error_probability

    @property
    def reference_confidence_level(self) -> float:
        """Compatibility alias for the joint decision confidence level."""

        return self.confidence_level

    @property
    def v0_anchor_confidence_level(self) -> float:
        """Compatibility alias for the joint decision confidence level."""

        return self.confidence_level

    @property
    def calibration_component_error_probability(self) -> float:
        """Compatibility alias; calibration has no separate error stake."""

        return self.error_probability

    @property
    def calibration_component_confidence_level(self) -> float:
        return 1.0 - self.calibration_component_error_probability

    @property
    def component_error_allocation(self) -> Mapping[str, float]:
        """Receipt-facing declaration of the single confidence-bearing object."""

        return {
            "joint_accept_reject_statistic": self.error_probability,
        }

    @property
    def content_hash(self) -> str:
        return _content_hash(self, prefix="sequential-bce-spec:")


@dataclass(frozen=True, slots=True)
class OptimizedCalibrationDecisionEnvelope:
    """Decision interval after accounting for attainable calibration improvement.

    Calibration and oracle rows are independent logical streams.  The oracle
    interval describes the risk of the currently frozen affine fits.  Because
    each affine fit minimizes the same BCE estimand, its remaining statistical
    error is represented by a nonnegative bound on how much further population
    BCE could improve after additional calibration data.  No fixed fraction of
    the pruning boundary is introduced.
    """

    oracle_estimate: float
    oracle_lower: float
    oracle_upper: float
    lower_improvement_bound: float
    upper_improvement_bound: float
    optimized_lower: float
    optimized_upper: float
    schema: str = "optimized_calibration_decision_envelope_v1"

    def __post_init__(self) -> None:
        values = (
            self.oracle_estimate,
            self.oracle_lower,
            self.oracle_upper,
            self.lower_improvement_bound,
            self.upper_improvement_bound,
            self.optimized_lower,
            self.optimized_upper,
        )
        if (
            not all(math.isfinite(float(value)) for value in values)
            or float(self.oracle_lower) > float(self.oracle_upper)
            or float(self.lower_improvement_bound) < 0.0
            or float(self.upper_improvement_bound) < 0.0
            or float(self.optimized_lower) > float(self.optimized_upper)
        ):
            raise ValueError("invalid optimized-calibration decision envelope")

    @classmethod
    def paired_increment(
        cls,
        *,
        oracle_estimate: float,
        oracle_lower: float,
        oracle_upper: float,
        parent_improvement_bound: float,
        candidate_improvement_bound: float,
    ) -> "OptimizedCalibrationDecisionEnvelope":
        """Envelope ``R_candidate* - R_parent*`` under two affine fits."""

        parent = max(0.0, float(parent_improvement_bound))
        candidate = max(0.0, float(candidate_improvement_bound))
        return cls(
            oracle_estimate=float(oracle_estimate),
            oracle_lower=float(oracle_lower),
            oracle_upper=float(oracle_upper),
            lower_improvement_bound=candidate,
            upper_improvement_bound=parent,
            optimized_lower=float(oracle_lower) - candidate,
            optimized_upper=float(oracle_upper) + parent,
        )

    @classmethod
    def absolute_risk(
        cls,
        *,
        oracle_estimate: float,
        oracle_lower: float,
        oracle_upper: float,
        calibration_improvement_bound: float,
    ) -> "OptimizedCalibrationDecisionEnvelope":
        """Envelope for one optimally affine-calibrated absolute BCE risk."""

        improvement = max(0.0, float(calibration_improvement_bound))
        return cls(
            oracle_estimate=float(oracle_estimate),
            oracle_lower=float(oracle_lower),
            oracle_upper=float(oracle_upper),
            lower_improvement_bound=improvement,
            upper_improvement_bound=0.0,
            optimized_lower=float(oracle_lower) - improvement,
            optimized_upper=float(oracle_upper),
        )

    @property
    def content_hash(self) -> str:
        return _content_hash(self, prefix="optimized-calibration-envelope:")


@dataclass(frozen=True, slots=True)
class CalibrationImprovementProfile:
    """Confidence-independent local calibration-risk uncertainty profile.

    For squared joint radius ``q``, the attainable BCE improvement is bounded
    by ``a * sqrt(q) + b * q``.  The profile is computed once from the affine
    score covariance and loss Hessian.  It receives no independent confidence
    level: the complete decision envelope supplies one joint radius.
    """

    first_order_coefficient: float = 0.0
    second_order_coefficient: float = 0.0
    parameter_dimension: int = 2
    schema: str = "calibration_improvement_profile_v1"

    def __post_init__(self) -> None:
        if (
            not math.isfinite(float(self.first_order_coefficient))
            or float(self.first_order_coefficient) < 0.0
            or not math.isfinite(float(self.second_order_coefficient))
            or float(self.second_order_coefficient) < 0.0
            or int(self.parameter_dimension) < 0
        ):
            raise ValueError("invalid calibration improvement profile")

    def bound(self, radius_squared: float) -> float:
        q = float(radius_squared)
        if not math.isfinite(q) or q < 0.0:
            raise ValueError("joint radius squared must be finite and nonnegative")
        return (
            float(self.first_order_coefficient) * math.sqrt(q)
            + float(self.second_order_coefficient) * q
        )

    @property
    def content_hash(self) -> str:
        return _content_hash(self, prefix="calibration-improvement-profile:")


@dataclass(frozen=True, slots=True)
class JointDecisionConfidenceEnvelope:
    """One confidence envelope for one scalar accept/reject statistic."""

    estimate: float
    lower: float
    upper: float
    alpha: float
    joint_dimension: int
    radius_squared: float
    look_index: int
    component_bounds: Mapping[str, float]
    method: str = JOINT_DECISION_WALD_METHOD
    schema: str = "joint_decision_confidence_envelope_v1"

    def __post_init__(self) -> None:
        if (
            not all(
                math.isfinite(float(value))
                for value in (
                    self.estimate,
                    self.lower,
                    self.upper,
                    self.alpha,
                    self.radius_squared,
                )
            )
            or float(self.lower) > float(self.estimate)
            or float(self.estimate) > float(self.upper)
            or not 0.0 < float(self.alpha) < 1.0
            or int(self.joint_dimension) <= 0
            or float(self.radius_squared) <= 0.0
            or int(self.look_index) < 0
            or any(
                not math.isfinite(float(value)) or float(value) < 0.0
                for value in self.component_bounds.values()
            )
        ):
            raise ValueError("invalid joint decision confidence envelope")

    @property
    def content_hash(self) -> str:
        return _content_hash(self, prefix="joint-decision-envelope:")


def joint_wald_radius_squared(*, alpha: float, dimension: int) -> float:
    """Return the one-minus-alpha chi-square radius for a joint Wald region."""

    tail = float(alpha)
    degrees = int(dimension)
    if not 0.0 < tail < 1.0 or degrees <= 0:
        raise ValueError("joint Wald radius requires alpha in (0,1) and d > 0")
    value = 2.0 * float(gammaincinv(0.5 * degrees, 1.0 - tail))
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError("joint Wald radius is not finite and positive")
    return value


def _nonnegative_standard_error(value: float, *, name: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return resolved


def joint_v1_decision_interval(
    *,
    candidate_estimate: float,
    candidate_standard_error: float,
    dense_estimate: float,
    dense_standard_error: float,
    dense_multiplier: float,
    candidate_calibration: CalibrationImprovementProfile,
    dense_calibration: CalibrationImprovementProfile,
    alpha: float,
    look_index: int,
) -> JointDecisionConfidenceEnvelope:
    """Joint interval for ``R_candidate* - m R_dense*``.

    Candidate and dense oracle streams are independent.  Their scalar contrast
    is therefore one uncertainty block.  The two affine calibration fits
    contribute two parameter dimensions each.  Projecting the one joint Wald
    ellipsoid gives simultaneous candidate/dense calibration corrections
    without component-wise alpha splitting.
    """

    candidate_se = _nonnegative_standard_error(
        candidate_standard_error,
        name="candidate standard error",
    )
    dense_se = _nonnegative_standard_error(
        dense_standard_error,
        name="dense standard error",
    )
    multiplier = float(dense_multiplier)
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("dense multiplier must be finite and positive")
    dimension = (
        1
        + int(candidate_calibration.parameter_dimension)
        + int(dense_calibration.parameter_dimension)
    )
    radius_squared = joint_wald_radius_squared(
        alpha=float(alpha),
        dimension=dimension,
    )
    oracle_radius = math.sqrt(radius_squared) * math.sqrt(
        candidate_se * candidate_se + multiplier * multiplier * dense_se * dense_se
    )
    candidate_bound = candidate_calibration.bound(radius_squared)
    dense_bound = dense_calibration.bound(radius_squared)
    estimate = float(candidate_estimate) - multiplier * float(dense_estimate)
    return JointDecisionConfidenceEnvelope(
        estimate=estimate,
        lower=estimate - oracle_radius - candidate_bound,
        upper=estimate + oracle_radius + multiplier * dense_bound,
        alpha=float(alpha),
        joint_dimension=dimension,
        radius_squared=radius_squared,
        look_index=int(look_index),
        component_bounds={
            "oracle_contrast_radius": oracle_radius,
            "candidate_calibration_improvement": candidate_bound,
            "dense_calibration_improvement_scaled": multiplier * dense_bound,
        },
    )


def joint_v0_decision_interval(
    *,
    numerator_estimate: float,
    numerator_standard_error: float,
    dense_estimate: float,
    dense_standard_error: float,
    fully_pruned_risk: float,
    connection_delta: int,
    parent_calibration: CalibrationImprovementProfile,
    candidate_calibration: CalibrationImprovementProfile,
    dense_calibration: CalibrationImprovementProfile,
    alpha: float,
    look_index: int,
) -> JointDecisionConfidenceEnvelope:
    """Joint interval for v0 fractional loss-span per removed connection."""

    numerator_se = _nonnegative_standard_error(
        numerator_standard_error,
        name="paired numerator standard error",
    )
    dense_se = _nonnegative_standard_error(
        dense_standard_error,
        name="dense standard error",
    )
    delta = int(connection_delta)
    if delta <= 0 or not math.isfinite(float(fully_pruned_risk)):
        raise ValueError("v0 joint interval requires a positive connection delta")
    dimension = (
        2
        + int(parent_calibration.parameter_dimension)
        + int(candidate_calibration.parameter_dimension)
        + int(dense_calibration.parameter_dimension)
    )
    radius_squared = joint_wald_radius_squared(
        alpha=float(alpha),
        dimension=dimension,
    )
    root_radius = math.sqrt(radius_squared)
    numerator_oracle_radius = root_radius * numerator_se
    dense_oracle_radius = root_radius * dense_se
    parent_bound = parent_calibration.bound(radius_squared)
    candidate_bound = candidate_calibration.bound(radius_squared)
    dense_bound = dense_calibration.bound(radius_squared)
    numerator_lower = (
        float(numerator_estimate) - numerator_oracle_radius - candidate_bound
    )
    numerator_upper = float(numerator_estimate) + numerator_oracle_radius + parent_bound
    span_estimate = float(fully_pruned_risk) - float(dense_estimate)
    span_lower = float(fully_pruned_risk) - (
        float(dense_estimate) + dense_oracle_radius
    )
    span_upper = float(fully_pruned_risk) - (
        float(dense_estimate) - dense_oracle_radius - dense_bound
    )
    estimate, lower, upper = conservative_ratio_interval(
        numerator_estimate=float(numerator_estimate),
        numerator_lower=numerator_lower,
        numerator_upper=numerator_upper,
        positive_denominator_estimate=span_estimate,
        positive_denominator_lower=span_lower,
        positive_denominator_upper=span_upper,
        connection_delta=delta,
    )
    return JointDecisionConfidenceEnvelope(
        estimate=estimate,
        lower=lower,
        upper=upper,
        alpha=float(alpha),
        joint_dimension=dimension,
        radius_squared=radius_squared,
        look_index=int(look_index),
        component_bounds={
            "paired_oracle_radius": numerator_oracle_radius,
            "dense_oracle_radius": dense_oracle_radius,
            "parent_calibration_improvement": parent_bound,
            "candidate_calibration_improvement": candidate_bound,
            "dense_calibration_improvement": dense_bound,
            "loss_span_lower": span_lower,
            "loss_span_upper": span_upper,
        },
    )


def fieller_ratio_interval(
    *,
    numerator: float,
    denominator: float,
    numerator_variance: float,
    denominator_variance: float,
    covariance: float = 0.0,
    alpha: float,
    numerator_bias_bound: float = 0.0,
    denominator_bias_bound: float = 0.0,
    quantile: float | None = None,
) -> Mapping[str, Any]:
    """Invert the scalar two-sided test at the supplied side alpha.

    A segment endpoint of ``None`` means infinity in that direction.  The
    second-order bounds widen the test by ``b_N + |r| b_B``.  Thus the reported
    set is the exact inversion of the statistic used for a decision at r.
    """

    n, d = float(numerator), float(denominator)
    vn, vd, cov = map(float, (numerator_variance, denominator_variance, covariance))
    bn, bd = float(numerator_bias_bound), float(denominator_bias_bound)
    if (
        not all(math.isfinite(x) for x in (n, d, vn, vd, cov, bn, bd))
        or d <= 0.0 or min(vn, vd, bn, bd) < 0.0
        or cov * cov > vn * vd * (1.0 + 1e-10) + 1e-30
        or not 0.0 < float(alpha) < 0.5
    ):
        raise ValueError("invalid Fieller inputs")
    z = float(norm.isf(alpha) if quantile is None else quantile)
    if not math.isfinite(z) or z <= 0.0:
        raise ValueError("Fieller quantile must be finite and positive")

    def admitted(r: float) -> bool:
        variance = max(0.0, vn - 2.0 * r * cov + r * r * vd)
        radius = z * math.sqrt(variance) + bn + abs(r) * bd
        return abs(n - r * d) <= radius

    # On each half-axis, the absolute-value bias is affine.  Each endpoint
    # solves one of two quadratics after squaring; the sign check below removes
    # the extraneous roots introduced by squaring.
    roots = [0.0]
    for sign in (-1.0, 1.0):
        for side in (-1.0, 1.0):
            a = n + side * bn
            c = -d + side * sign * bd
            coefficients = (
                c * c - z * z * vd,
                2.0 * a * c + 2.0 * z * z * cov * side * side,
                a * a - z * z * vn,
            )
            # The covariance term has the same sign for both squared equations.
            qa, qb, qc = coefficients
            scale = max(abs(qa), abs(qb), abs(qc), 1.0)
            if abs(qa) <= 1e-14 * scale:
                candidates = [] if abs(qb) <= 1e-14 * scale else [-qc / qb]
            else:
                discriminant = qb * qb - 4.0 * qa * qc
                candidates = [] if discriminant < 0.0 else [
                    (-qb - math.sqrt(max(0.0, discriminant))) / (2.0 * qa),
                    (-qb + math.sqrt(max(0.0, discriminant))) / (2.0 * qa),
                ]
            for r in candidates:
                if not math.isfinite(r) or r * sign < -1e-12:
                    continue
                linear = a + c * r
                if side * linear > 1e-9 * max(1.0, abs(linear)):
                    continue
                if abs(abs(n - r * d) - (
                    z * math.sqrt(max(0.0, vn - 2.0 * r * cov + r * r * vd))
                    + bn + abs(r) * bd
                )) <= 1e-8 * max(1.0, abs(n), abs(r * d)):
                    roots.append(r)
    cuts = [-math.inf, *sorted(set(roots)), math.inf]
    pieces: list[tuple[float | None, float | None]] = []
    for left, right in zip(cuts, cuts[1:]):
        if math.isinf(left):
            probe = right - max(1.0, abs(right) * 2.0)
        elif math.isinf(right):
            probe = left + max(1.0, abs(left) * 2.0)
        else:
            probe = 0.5 * (left + right)
        if admitted(probe):
            pieces.append((None if math.isinf(left) else left,
                           None if math.isinf(right) else right))
    # Isolated admitted points can occur at tangency.
    for r in cuts[1:-1]:
        if admitted(r) and not any(
            (lo is None or lo <= r) and (hi is None or r <= hi)
            for lo, hi in pieces
        ):
            pieces.append((r, r))
    pieces.sort(key=lambda pair: -math.inf if pair[0] is None else pair[0])
    merged: list[list[float | None]] = []
    for lo, hi in pieces:
        if merged and (merged[-1][1] is None or lo is None or
                       lo <= float(merged[-1][1]) + 1e-10):
            if merged[-1][1] is not None:
                merged[-1][1] = hi if hi is None else max(float(merged[-1][1]), hi)
        else:
            merged.append([lo, hi])
    return {
        "schema": "fieller_ratio_interval_v1",
        "segments": tuple({"lower": lo, "upper": hi} for lo, hi in merged),
        "unbounded": any(lo is None or hi is None for lo, hi in merged),
        "alpha": float(alpha),
        "one_sided_quantile": z,
        "coverage": 1.0 - 2.0 * float(alpha),
        "coverage_label": "1 - 2*alpha_side",
        "coverage_scope": "nominal_assumption_dependent",
    }


@dataclass(frozen=True, slots=True)
class ScalarV0DecisionEnvelope:
    """Decision coordinates at tau, plus the Fieller inversion over tau."""

    estimate: float
    lower: float
    upper: float
    alpha: float
    joint_dimension: int
    radius_squared: float
    look_index: int
    component_bounds: Mapping[str, float]
    statistic_estimate: float
    statistic_lower: float
    statistic_upper: float
    statistic_variance: float
    fieller_interval: Mapping[str, Any]
    method: str = SCALAR_DECISION_FIELLER_METHOD
    schema: str = "scalar_v0_decision_envelope_v1"

    @property
    def content_hash(self) -> str:
        return _content_hash(self, prefix="scalar-v0-decision-envelope:")


def scalar_v0_decision_interval(
    *,
    numerator_estimate: float,
    numerator_standard_error: float,
    dense_estimate: float,
    dense_standard_error: float,
    fully_pruned_risk: float,
    connection_delta: int,
    parent_calibration: CalibrationImprovementProfile,
    candidate_calibration: CalibrationImprovementProfile,
    dense_calibration: CalibrationImprovementProfile,
    threshold: float,
    alpha: float,
    look_index: int,
    oracle_covariance: float = 0.0,
    calibration_covariance: np.ndarray | None = None,
    degrees_of_freedom: float | None = None,
    v5_sidedness: str = "one_sided_alpha_over_two",
    fully_pruned_standard_error: float = 0.0,
    fully_pruned_bias_bound: float = 0.0,
) -> ScalarV0DecisionEnvelope:
    """Delta-method test for N - tau |S| D, with Fieller inversion.

    Oracle and calibration blocks use independent draw streams.  Within each
    block, cross-covariance is retained when supplied.  The default zero
    cross-covariance for the three mask fits assumes independent mask-specific
    calibration streams; shared calibration rows require their fitted joint
    score covariance here.
    """

    delta, tau = int(connection_delta), float(threshold)
    n, d = float(numerator_estimate), float(fully_pruned_risk) - float(dense_estimate)
    if delta <= 0 or d <= 0.0 or not math.isfinite(tau) or tau < 0.0:
        raise ValueError("scalar v0 decision requires a positive span and delta")
    numerator_se = _nonnegative_standard_error(
        numerator_standard_error, name="paired numerator standard error"
    )
    dense_se = _nonnegative_standard_error(
        dense_standard_error, name="dense standard error"
    )
    empty_se = _nonnegative_standard_error(
        fully_pruned_standard_error, name="fully pruned standard error"
    )
    empty_bias = float(fully_pruned_bias_bound)
    if not math.isfinite(empty_bias) or empty_bias < 0.0:
        raise ValueError("fully pruned bias bound must be finite and nonnegative")
    oracle = np.array([
        [numerator_se ** 2, float(oracle_covariance)],
        [float(oracle_covariance), dense_se ** 2],
    ], dtype=np.float64)
    profiles = (candidate_calibration, parent_calibration, dense_calibration)
    calib = (np.diag([p.first_order_coefficient ** 2 for p in profiles])
             if calibration_covariance is None else np.asarray(calibration_covariance, dtype=np.float64))
    if (calib.shape != (3, 3) or not np.isfinite(calib).all()
            or not np.allclose(calib, calib.T, rtol=1e-10, atol=1e-12)
            or float(np.linalg.eigvalsh(calib).min()) < (
                -1e-10 * max(float(np.diag(calib).max()), 1e-30)
            )
            or not np.allclose(
                np.diag(calib),
                [p.first_order_coefficient ** 2 for p in profiles],
                rtol=1e-8, atol=1e-12,
            )
            or not np.isfinite(oracle).all()
            or oracle[0, 1] ** 2 > (
                oracle[0, 0] * oracle[1, 1] * (1.0 + 1e-10) + 1e-30
            )):
        raise ValueError("scalar v0 covariance must be finite and positive semidefinite")
    n_gradient = np.array([1.0, -1.0, 0.0])
    b_gradient = np.array([0.0, 0.0, -float(delta)])
    var_n = float(oracle[0, 0] + n_gradient @ calib @ n_gradient)
    var_b = float(delta * delta * (oracle[1, 1] + empty_se ** 2)
                  + b_gradient @ calib @ b_gradient)
    cov_nb = float(-delta * oracle[0, 1] + n_gradient @ calib @ b_gradient)
    var_t = max(0.0, var_n - 2.0 * tau * cov_nb + tau * tau * var_b)
    dof = None if degrees_of_freedom is None else float(degrees_of_freedom)
    if dof is not None and (math.isnan(dof) or dof <= 0.0):
        raise ValueError("scalar v0 degrees of freedom must be positive")
    if v5_sidedness != "one_sided_alpha_over_two":
        raise ValueError("unsupported v5 sidedness")
    alpha_side = float(alpha) / 2.0
    z = float(norm.isf(alpha_side) if dof is None else student_t.isf(alpha_side, dof))
    if not math.isfinite(z) or z <= 0.0:
        raise ValueError("scalar v0 alpha must give a positive quantile")
    if any(
        p.parameter_dimension == 0 and p.second_order_coefficient > 0.0
        for p in profiles
    ):
        raise ValueError("a zero-dimensional fit cannot have a quadratic allowance")
    quadratic_radii = tuple(
        0.0
        if p.parameter_dimension == 0
        else float(chi2.isf(alpha_side, p.parameter_dimension))
        for p in profiles
    )
    bias_n = sum(
        q * p.second_order_coefficient
        for q, p in zip(quadratic_radii[:2], profiles[:2])
    )
    bias_b = delta * (quadratic_radii[2] * profiles[2].second_order_coefficient
                      + empty_bias)
    bias_t = bias_n + tau * bias_b
    statistic = n - tau * delta * d
    radius = z * math.sqrt(var_t)
    scale = delta * d
    ratio = n / scale
    fieller = fieller_ratio_interval(
        numerator=n, denominator=scale,
        numerator_variance=var_n, denominator_variance=var_b,
        covariance=cov_nb, alpha=alpha_side,
        numerator_bias_bound=bias_n, denominator_bias_bound=bias_b,
        quantile=z,
    )
    return ScalarV0DecisionEnvelope(
        estimate=ratio,
        lower=tau + (statistic - radius - bias_t) / scale,
        upper=tau + (statistic + radius + bias_t) / scale,
        alpha=float(alpha), joint_dimension=1, radius_squared=z * z,
        look_index=int(look_index),
        component_bounds={
            "paired_oracle_radius": z * math.sqrt(float(oracle[0, 0])),
            "dense_oracle_radius": z * math.sqrt(float(oracle[1, 1])),
            "parent_calibration_improvement": (
                z * profiles[1].first_order_coefficient
                + quadratic_radii[1] * profiles[1].second_order_coefficient
            ),
            "candidate_calibration_improvement": (
                z * profiles[0].first_order_coefficient
                + quadratic_radii[0] * profiles[0].second_order_coefficient
            ),
            "dense_calibration_improvement": (
                z * profiles[2].first_order_coefficient
                + quadratic_radii[2] * profiles[2].second_order_coefficient
            ),
            "scalar_standard_error": math.sqrt(var_t),
            "second_order_bias_bound": bias_t,
            "paired_oracle_variance": float(oracle[0, 0]),
            "dense_oracle_variance_scaled": tau * tau * delta * delta * float(oracle[1, 1]),
            "fully_pruned_variance_scaled": tau * tau * delta * delta * empty_se ** 2,
            "fully_pruned_bias_scaled": tau * delta * empty_bias,
            "candidate_calibration_variance": profiles[0].first_order_coefficient ** 2,
            "parent_calibration_variance": profiles[1].first_order_coefficient ** 2,
            "dense_calibration_variance_scaled": (
                tau * delta * profiles[2].first_order_coefficient
            ) ** 2,
            "candidate_second_order_bias": quadratic_radii[0] * profiles[0].second_order_coefficient,
            "parent_second_order_bias": quadratic_radii[1] * profiles[1].second_order_coefficient,
            "dense_second_order_bias_scaled": quadratic_radii[2] * tau * delta * profiles[2].second_order_coefficient,
            "scalar_quantile": z,
            "alpha_side": alpha_side,
        },
        statistic_estimate=statistic,
        statistic_lower=statistic - radius - bias_t,
        statistic_upper=statistic + radius + bias_t,
        statistic_variance=var_t,
        fieller_interval=fieller,
    )


@dataclass(frozen=True, slots=True)
class JointPrecisionActionForecast:
    """One predictable calibration-or-oracle acquisition option."""

    role: Literal[
        "candidate_calibration",
        "parent_calibration",
        "dense_calibration",
        "dense_oracle",
        "oracle",
    ]
    current_count: int
    target_count: int
    predicted_decision_width_reduction: float
    predicted_device_seconds: float
    utility_per_device_second: float
    target_loss_improvement_bound: float | None = None
    schema: str = "joint_precision_action_forecast_v1"

    def __post_init__(self) -> None:
        if (
            int(self.current_count) < 0
            or int(self.target_count) <= int(self.current_count)
            or not all(
                math.isfinite(float(value)) and float(value) >= 0.0
                for value in (
                    self.predicted_decision_width_reduction,
                    self.predicted_device_seconds,
                    self.utility_per_device_second,
                )
            )
            or float(self.predicted_device_seconds) <= 0.0
            or (
                self.target_loss_improvement_bound is not None
                and (
                    not math.isfinite(float(self.target_loss_improvement_bound))
                    or float(self.target_loss_improvement_bound) < 0.0
                )
            )
        ):
            raise ValueError("invalid joint precision action forecast")

    @property
    def content_hash(self) -> str:
        return _content_hash(self, prefix="joint-precision-action-forecast:")


def select_joint_precision_action(
    forecasts: Sequence[JointPrecisionActionForecast],
) -> JointPrecisionActionForecast:
    """Choose the predictable acquisition with maximum decision value per cost.

    Ties are resolved by a frozen role order and then the smaller target.  The
    rule sees only uncertainty and realized/forecast execution cost; it never
    sees future outcomes.
    """

    options = tuple(forecasts)
    if not options:
        raise ValueError("joint precision allocation requires an action")
    role_order = {
        "candidate_calibration": 0,
        "parent_calibration": 1,
        "dense_calibration": 2,
        "dense_oracle": 3,
        "oracle": 4,
    }
    return max(
        options,
        key=lambda item: (
            float(item.utility_per_device_second),
            float(item.predicted_decision_width_reduction),
            -role_order[item.role],
            -int(item.target_count),
            item.content_hash,
        ),
    )


def scalar_precision_forecasts(
    forecasts: Sequence[JointPrecisionActionForecast],
    components: Mapping[str, float],
) -> tuple[JointPrecisionActionForecast, ...]:
    """Value every next acquisition by the v5 scalar test's width reduction."""

    variance_names = (
        "paired_oracle_variance", "dense_oracle_variance_scaled",
        "candidate_calibration_variance", "parent_calibration_variance",
        "dense_calibration_variance_scaled",
    )
    bias_names = (
        "candidate_second_order_bias", "parent_second_order_bias",
        "dense_second_order_bias_scaled",
    )
    values = {name: float(components[name]) for name in (*variance_names, *bias_names)}
    z = float(components["scalar_quantile"])
    if not math.isfinite(z) or z <= 0.0 or any(
        not math.isfinite(value) or value < 0.0 for value in values.values()
    ):
        raise ValueError("invalid scalar precision components")
    current = 2.0 * (z * math.sqrt(sum(values[name] for name in variance_names))
                     + sum(values[name] for name in bias_names))
    role_variance = {
        "oracle": "paired_oracle_variance",
        "dense_oracle": "dense_oracle_variance_scaled",
        "candidate_calibration": "candidate_calibration_variance",
        "parent_calibration": "parent_calibration_variance",
        "dense_calibration": "dense_calibration_variance_scaled",
    }
    role_bias = {
        "candidate_calibration": "candidate_second_order_bias",
        "parent_calibration": "parent_second_order_bias",
        "dense_calibration": "dense_second_order_bias_scaled",
    }
    result = []
    for forecast in forecasts:
        factor = float(forecast.current_count) / float(forecast.target_count)
        variance = sum(values[name] for name in variance_names)
        bias = sum(values[name] for name in bias_names)
        variance -= values[role_variance[forecast.role]] * (1.0 - factor)
        if forecast.role in role_bias:
            bias -= values[role_bias[forecast.role]] * (1.0 - factor)
        projected = 2.0 * (z * math.sqrt(max(0.0, variance)) + max(0.0, bias))
        reduction = max(0.0, current - projected)
        result.append(replace(
            forecast,
            predicted_decision_width_reduction=reduction,
            utility_per_device_second=reduction / forecast.predicted_device_seconds,
        ))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class CalibrationForecastReceipt:
    """Predictable analytic calibration-prefix forecast."""

    current_count: int
    requested_target_bce: float
    first_order_coefficient: float
    second_order_coefficient: float
    raw_required_count: int
    guarded_required_count: int
    occupancy_quantum: int
    rounded_required_count: int
    forecast_validated: bool
    predicted_bound_at_rounded_count: float
    schema: str = "calibration_forecast_receipt_v1"

    def __post_init__(self) -> None:
        if (
            int(self.current_count) <= 0
            or int(self.raw_required_count) < int(self.current_count)
            or int(self.guarded_required_count) < int(self.current_count)
            or int(self.occupancy_quantum) <= 0
            or int(self.rounded_required_count) < int(self.guarded_required_count)
            or not all(
                math.isfinite(float(value)) and float(value) >= 0.0
                for value in (
                    self.requested_target_bce,
                    self.first_order_coefficient,
                    self.second_order_coefficient,
                    self.predicted_bound_at_rounded_count,
                )
            )
            or float(self.requested_target_bce) <= 0.0
        ):
            raise ValueError("invalid calibration forecast receipt")

    @property
    def content_hash(self) -> str:
        return _content_hash(self, prefix="calibration-forecast-receipt:")


CircuitConnectionSemantics = Literal[
    "fc_scalar_weight",
    "conv_channel_pair",
    "conv_kernel_tap",
    "protected_nonconnection_route",
]


@dataclass(frozen=True, slots=True)
class CircuitConnectionDeltaReceipt:
    """Exact connection-axis change for one parent--candidate transition."""

    parent_mask_hash: str
    candidate_mask_hash: str
    removed_route_index_hash: str
    total_removed_connections: int
    per_layer_removed_connections: tuple[tuple[str, int], ...]
    per_layer_route_semantics: tuple[tuple[str, CircuitConnectionSemantics], ...]
    added_connection_count: int = 0
    schema: str = "circuit_connection_delta_receipt_v1"

    def __post_init__(self) -> None:
        if (
            not self.parent_mask_hash
            or not self.candidate_mask_hash
            or not self.removed_route_index_hash
            or int(self.total_removed_connections) <= 0
            or int(self.added_connection_count) != 0
        ):
            raise ValueError("invalid circuit-connection delta receipt")
        counts = tuple(
            (str(key), int(value)) for key, value in self.per_layer_removed_connections
        )
        semantics = tuple(
            (str(key), str(value)) for key, value in self.per_layer_route_semantics
        )
        if (
            any(value < 0 for _key, value in counts)
            or sum(value for _key, value in counts)
            != int(self.total_removed_connections)
            or {key for key, _value in counts} != {key for key, _value in semantics}
            or any(
                value
                not in {
                    "fc_scalar_weight",
                    "conv_channel_pair",
                    "conv_kernel_tap",
                    "protected_nonconnection_route",
                }
                for _key, value in semantics
            )
            or any(
                dict(counts)[key] != 0
                for key, value in semantics
                if value == "protected_nonconnection_route"
            )
        ):
            raise ValueError("connection counts and route semantics are inconsistent")
        object.__setattr__(self, "per_layer_removed_connections", tuple(sorted(counts)))
        object.__setattr__(self, "per_layer_route_semantics", tuple(sorted(semantics)))

    @classmethod
    def from_masks(
        cls,
        *,
        parent_masks: Mapping[str, np.ndarray],
        candidate_masks: Mapping[str, np.ndarray],
        route_semantics_by_layer: Mapping[str, CircuitConnectionSemantics],
        parent_mask_hash: str,
        candidate_mask_hash: str,
    ) -> "CircuitConnectionDeltaReceipt":
        if set(parent_masks) != set(candidate_masks):
            raise ValueError("parent and candidate mask layer axes differ")
        if set(parent_masks) != set(route_semantics_by_layer):
            raise ValueError("route semantics must cover the exact mask layer axis")
        # The removed-route identity is hashed from each layer's
        # packed removed-bit vector (header [layer, shape, count] then the
        # fixed-length packed bytes), not from a Python list of one
        # [layer, index] row per removed route (up to ~37M rows on fc6). The
        # removed set is encoded exactly -- nothing is lost -- and the framing
        # is unambiguous. The schema prefix is new (v2), so v2 hashes are never
        # compared with historical v1 values; _removed_connection_axis_hash
        # keeps the v1 construction for reading historical receipts.
        digest = hashlib.sha256(b"removed-circuit-connection-axis:v2-packed-bits:")
        counts: list[tuple[str, int]] = []
        additions = 0
        for layer_id in sorted(parent_masks):
            parent = np.asarray(parent_masks[layer_id], dtype=bool)
            candidate = np.asarray(candidate_masks[layer_id], dtype=bool)
            if parent.shape != candidate.shape:
                raise ValueError("parent and candidate mask shapes differ")
            parent_flat = parent.reshape(-1)
            candidate_flat = candidate.reshape(-1)
            removed_bits = parent_flat & ~candidate_flat
            removed_count = int(np.count_nonzero(removed_bits))
            additions += int(np.count_nonzero(~parent_flat & candidate_flat))
            counts.append((str(layer_id), removed_count))
            digest.update(
                json.dumps(
                    [str(layer_id), [int(dim) for dim in parent.shape], removed_count],
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            digest.update(np.packbits(removed_bits, bitorder="little").tobytes())
        return cls(
            parent_mask_hash=str(parent_mask_hash),
            candidate_mask_hash=str(candidate_mask_hash),
            removed_route_index_hash=digest.hexdigest(),
            total_removed_connections=sum(value for _key, value in counts),
            per_layer_removed_connections=tuple(counts),
            per_layer_route_semantics=tuple(
                (str(key), str(value))
                for key, value in route_semantics_by_layer.items()
            ),
            added_connection_count=additions,
        )

    @property
    def content_hash(self) -> str:
        return _content_hash(self, prefix="connection-delta-receipt:")

    @property
    def receipt(self) -> Mapping[str, Any]:
        return {
            **asdict(self),
            "content_hash": self.content_hash,
        }


_CONNECTION_SEMANTICS = (
    "fc_scalar_weight",
    "conv_channel_pair",
    "conv_kernel_tap",
    "protected_nonconnection_route",
)


@dataclass(frozen=True, slots=True)
class ConnectionDelta:
    """The exact connection counts of one transition, no digests.

    The counts are a statistic input (the v0 estimand is normalized per
    removed connection), so they are kept exact: one connection per removed
    route, exactly as ``CircuitConnectionDeltaReceipt.from_masks`` counts.
    The receipt's mask and removed-bit digests are pure audit (kind 4) and
    are not computed.  Validation mirrors the receipt's ``__post_init__``.
    """

    total_removed_connections: int
    per_layer_removed_connections: tuple[tuple[str, int], ...]
    per_layer_route_semantics: tuple[tuple[str, str], ...]
    added_connection_count: int = 0
    schema: str = "circuit_connection_delta_v6"

    def __post_init__(self) -> None:
        counts = tuple(
            sorted((str(key), int(value)) for key, value in self.per_layer_removed_connections)
        )
        semantics = tuple(
            sorted((str(key), str(value)) for key, value in self.per_layer_route_semantics)
        )
        if (
            int(self.total_removed_connections) <= 0
            or int(self.added_connection_count) != 0
            or any(value < 0 for _key, value in counts)
            or sum(value for _key, value in counts) != int(self.total_removed_connections)
            or {key for key, _value in counts} != {key for key, _value in semantics}
            or any(value not in _CONNECTION_SEMANTICS for _key, value in semantics)
            or any(
                dict(counts)[key] != 0
                for key, value in semantics
                if value == "protected_nonconnection_route"
            )
        ):
            raise ValueError("connection counts and route semantics are inconsistent")
        object.__setattr__(self, "per_layer_removed_connections", counts)
        object.__setattr__(self, "per_layer_route_semantics", semantics)

    @property
    def receipt(self) -> dict[str, Any]:
        """The digest-free record of the counts (the ledger's kind-3 row).

        The same count fields as ``CircuitConnectionDeltaReceipt.receipt``;
        the mask, removed-bit and content digests (kind 4) are absent.
        """

        return {
            "schema": self.schema,
            "total_removed_connections": int(self.total_removed_connections),
            "per_layer_removed_connections": tuple(
                (key, int(value)) for key, value in self.per_layer_removed_connections
            ),
            "per_layer_route_semantics": tuple(
                (key, value) for key, value in self.per_layer_route_semantics
            ),
            "added_connection_count": int(self.added_connection_count),
        }

    @classmethod
    def from_layer_counts(
        cls,
        *,
        layer_ids: Sequence[str],
        removed_by_layer: Mapping[str, int],
        route_semantics_by_layer: Mapping[str, CircuitConnectionSemantics],
    ) -> "ConnectionDelta":
        layers = {str(layer) for layer in layer_ids}
        if layers != {str(key) for key in route_semantics_by_layer}:
            raise ValueError("route semantics must cover the exact mask layer axis")
        if not {str(key) for key in removed_by_layer} <= layers:
            raise ValueError("removed-route counts name a layer outside the mask axis")
        counts = tuple(
            (layer, int(removed_by_layer.get(layer, 0))) for layer in sorted(layers)
        )
        return cls(
            total_removed_connections=sum(value for _key, value in counts),
            per_layer_removed_connections=counts,
            per_layer_route_semantics=tuple(
                (str(key), str(value)) for key, value in route_semantics_by_layer.items()
            ),
        )


@dataclass(frozen=True, slots=True)
class RiskIntervalCertificate:
    estimate: float
    lower: float
    upper: float
    context_count: int
    calibration_state_id: str
    sampling_design_id: str
    data_role: str
    code_identity: str
    output_identity: str
    confidence_level: float = 0.95
    confidence_scope: str = PLANNED_LOOK_SCOPE
    standard_error: float = 0.0
    degrees_of_freedom: float | None = None
    look_index: int = 0

    def __post_init__(self) -> None:
        if (
            not all(
                math.isfinite(float(value))
                for value in (self.estimate, self.lower, self.upper)
            )
            or float(self.lower) > float(self.estimate)
            or float(self.estimate) > float(self.upper)
            or int(self.context_count) <= 0
            or not 0.5 < float(self.confidence_level) < 1.0
            or not math.isfinite(float(self.standard_error))
            or float(self.standard_error) < 0.0
            or (
                self.degrees_of_freedom is not None
                and (
                    not math.isfinite(float(self.degrees_of_freedom))
                    or float(self.degrees_of_freedom) <= 0.0
                )
            )
            or int(self.look_index) < 0
            or self.confidence_scope
            not in {
                PLANNED_LOOK_SCOPE,
                PRECISION_ONLY_IID_SCOPE,
            }
            or not all(
                str(value)
                for value in (
                    self.calibration_state_id,
                    self.sampling_design_id,
                    self.data_role,
                    self.code_identity,
                    self.output_identity,
                )
            )
        ):
            raise ValueError("invalid risk interval certificate")


@dataclass(frozen=True, slots=True)
class BaselineRiskCertificate(RiskIntervalCertificate):
    """Locked dense-state calibrated-BCE estimate and uncertainty."""

    mask_hash: str = ""
    calibration_improvement_bound: float = 0.0
    calibration_first_order_coefficient: float = 0.0
    calibration_second_order_coefficient: float = 0.0
    calibration_parameter_dimension: int = 2
    schema: str = "baseline_risk_certificate_v1"

    def __post_init__(self) -> None:
        RiskIntervalCertificate.__post_init__(self)
        if (
            not self.mask_hash
            or not math.isfinite(float(self.calibration_improvement_bound))
            or float(self.calibration_improvement_bound) < 0.0
            or not math.isfinite(float(self.calibration_first_order_coefficient))
            or float(self.calibration_first_order_coefficient) < 0.0
            or not math.isfinite(float(self.calibration_second_order_coefficient))
            or float(self.calibration_second_order_coefficient) < 0.0
            or int(self.calibration_parameter_dimension) < 0
        ):
            raise ValueError("baseline risk certificate requires a mask hash")

    @property
    def calibration_profile(self) -> CalibrationImprovementProfile:
        return CalibrationImprovementProfile(
            first_order_coefficient=(self.calibration_first_order_coefficient),
            second_order_coefficient=(self.calibration_second_order_coefficient),
            parameter_dimension=self.calibration_parameter_dimension,
        )

    @property
    def optimized_lower(self) -> float:
        return float(self.lower) - float(self.calibration_improvement_bound)

    @property
    def optimized_upper(self) -> float:
        return float(self.upper)

    @property
    def content_hash(self) -> str:
        return _content_hash(self, prefix="baseline-risk-certificate:")


@dataclass(frozen=True, slots=True)
class AnalyticChanceRiskCertificate:
    """Exact intercept-only BCE risk of a certified constant response.

    This is a normalization reference, not a calibration fit produced by the
    adaptive trajectory.  If the empty physical route mask is structurally
    certified to emit one context-constant scalar, an unrestricted affine
    intercept can realize probability 1/2.  Under equal on/off-target risk
    weights and natural-log BCE, the resulting population risk is exactly
    ``log(2)``. The per-image schema instead uses the training population
    shares and the binary entropy of the target prior. Both schemas require
    natural-log BCE, a constant response, and an unrestricted intercept.
    """

    mask_hash: str
    constant_response_proof_id: str
    code_identity: str
    confidence_level: float = 0.95
    stratum_weights: tuple[tuple[str, float], ...] = (
        ("off_target", 0.5),
        ("on_target", 0.5),
    )
    loss_family: str = "binary_natural_log_bce"
    calibration_estimand_id: str = (
        "unrestricted_constant_intercept_population_optimum_v1"
    )
    schema: str = ANALYTIC_BALANCED_BINARY_CHANCE_RISK_V1
    estimate: float = field(init=False, default=math.log(2.0))
    lower: float = field(init=False, default=math.log(2.0))
    upper: float = field(init=False, default=math.log(2.0))
    context_count: int = field(init=False, default=0)
    data_role: str = field(init=False, default="fully_pruned_anchor")
    sampling_design_id: str = field(
        init=False,
        default="analytic_no_sampling_v1",
    )
    confidence_scope: str = field(init=False, default="analytic_exact")

    def __post_init__(self) -> None:
        if (
            not self.mask_hash
            or not self.constant_response_proof_id
            or not self.code_identity
            or self.loss_family != "binary_natural_log_bce"
            or self.calibration_estimand_id
            != "unrestricted_constant_intercept_population_optimum_v1"
            or self.schema not in {ANALYTIC_BALANCED_BINARY_CHANCE_RISK_V1,
                                   "analytic_per_image_binary_natural_log_bce_chance_risk_v1"}
            or (self.schema == ANALYTIC_BALANCED_BINARY_CHANCE_RISK_V1
                and _normalized_weights(self.stratum_weights)
                != (("off_target", 0.5), ("on_target", 0.5)))
            or not 0.5 < float(self.confidence_level) < 1.0
        ):
            raise ValueError(
                "analytic chance risk requires a certified constant response, "
                "binary natural-log BCE with valid population weights, and a free intercept"
            )
        if self.schema == "analytic_per_image_binary_natural_log_bce_chance_risk_v1":
            weights = dict(_normalized_weights(self.stratum_weights))
            if set(weights) != {"on_target", "off_target"}:
                raise ValueError("per-image chance risk requires target and off-target population shares")
            prior = weights["on_target"]
            if not 0.0 < prior < 1.0:
                raise ValueError("per-image chance risk requires a target prior in (0, 1)")
            risk = -prior * math.log(prior) - (1.0 - prior) * math.log1p(-prior)
            for name in ("estimate", "lower", "upper"):
                object.__setattr__(self, name, risk)

    @property
    def calibration_state_id(self) -> str:
        return _content_hash(
            {
                "estimand": self.calibration_estimand_id,
                "constant_response_proof_id": self.constant_response_proof_id,
                **({"stratum_weights": self.stratum_weights, "loss_weighting": "per_image"}
                   if self.schema == "analytic_per_image_binary_natural_log_bce_chance_risk_v1" else {}),
            },
            prefix="analytic-chance-calibration-estimand:",
        )

    @property
    def output_identity(self) -> str:
        return _content_hash(
            {
                "mask_hash": self.mask_hash,
                "constant_response_proof_id": self.constant_response_proof_id,
                "risk": self.estimate,
                "loss_family": self.loss_family,
                "stratum_weights": self.stratum_weights,
            },
            prefix="analytic-chance-risk-output:",
        )

    @property
    def content_hash(self) -> str:
        return _content_hash(
            {
                **asdict(self),
                "calibration_state_id": self.calibration_state_id,
                "output_identity": self.output_identity,
            },
            prefix="analytic-chance-risk-certificate:",
        )

    @property
    def receipt(self) -> Mapping[str, Any]:
        return {
            **asdict(self),
            "calibration_state_id": self.calibration_state_id,
            "output_identity": self.output_identity,
            "content_hash": self.content_hash,
            **({"loss_weighting": "per_image"}
               if self.schema == "analytic_per_image_binary_natural_log_bce_chance_risk_v1" else {}),
        }


@dataclass(frozen=True, slots=True)
class LossSpanCertificate:
    """Dense-to-fully-pruned BCE span with conservative positive bounds."""

    dense: BaselineRiskCertificate
    fully_pruned: RiskIntervalCertificate | AnalyticChanceRiskCertificate
    schema: str = "loss_span_certificate_v2"

    @property
    def estimate(self) -> float:
        return float(self.fully_pruned.estimate) - float(self.dense.estimate)

    @property
    def lower(self) -> float:
        return float(self.fully_pruned.lower) - float(self.dense.upper)

    @property
    def upper(self) -> float:
        # The dense fit may still improve by ``B_0`` under additional
        # calibration data. Since the span is empty-risk minus dense-risk,
        # the smallest attainable dense risk produces the largest admissible
        # span. Ignoring this term would make the ratio interval too narrow.
        return float(self.fully_pruned.upper) - float(self.dense.optimized_lower)

    def validate_positive(self) -> None:
        if (
            not math.isfinite(self.estimate)
            or not math.isfinite(self.lower)
            or not math.isfinite(self.upper)
            or self.lower <= 0.0
            or self.estimate <= 0.0
            or self.upper < self.estimate
            or not math.isclose(
                float(self.dense.confidence_level),
                float(self.fully_pruned.confidence_level),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise ValueError(
                "v0 requires a verified strictly positive dense-to-empty BCE span"
            )

    @property
    def content_hash(self) -> str:
        self.validate_positive()
        return _content_hash(self, prefix="loss-span-certificate:")


@dataclass(frozen=True, slots=True)
class StratifiedStudentTSummary:
    mean: float
    lower: float
    upper: float
    standard_error: float
    degrees_of_freedom: float | None
    degenerate_zero_variance: bool
    counts_by_stratum: tuple[tuple[str, int], ...]
    means_by_stratum: tuple[tuple[str, float], ...]
    variances_by_stratum: tuple[tuple[str, float], ...]
    alpha: float
    method: str = STRATIFIED_T_METHOD
    # Sealed without-replacement disclosure.  Defaults reproduce the
    # historical with-replacement summary exactly.
    finite_population_correction_applied: bool = False
    population_by_stratum: tuple[tuple[str, int], ...] = ()
    census_complete: bool = False


@dataclass(frozen=True, slots=True)
class RunningMoment:
    """Mergeable scalar sufficient statistics.

    ``m2`` is the centered sum of squares, not the sample variance.  The
    representation is closed under the parallel/Welford merge and therefore
    permits the native Torch provider to reduce BCE rows on device and return
    only three scalars per stratum and candidate.
    """

    count: int
    mean: float
    m2: float

    def __post_init__(self) -> None:
        if int(self.count) < 0:
            raise ValueError("running-moment count must be nonnegative")
        if not math.isfinite(float(self.mean)) or not math.isfinite(float(self.m2)):
            raise ValueError("running moments must be finite")
        tolerance = (
            1e-12 * max(1.0, abs(float(self.mean))) ** 2 * max(1, int(self.count))
        )
        if float(self.m2) < -tolerance:
            raise ValueError("running-moment m2 must be nonnegative")
        object.__setattr__(self, "count", int(self.count))
        object.__setattr__(self, "mean", float(self.mean))
        object.__setattr__(self, "m2", max(0.0, float(self.m2)))

    @classmethod
    def empty(cls) -> "RunningMoment":
        return cls(count=0, mean=0.0, m2=0.0)

    @classmethod
    def from_values(cls, values: Sequence[float] | np.ndarray) -> "RunningMoment":
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.size == 0 or not np.isfinite(array).all():
            raise ValueError("running moments require finite nonempty values")
        mean = float(np.mean(array))
        return cls(
            count=int(array.size),
            mean=mean,
            m2=float(np.sum((array - mean) ** 2, dtype=np.float64)),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RunningMoment":
        return cls(
            count=int(value["count"]),
            mean=float(value["mean"]),
            m2=float(value["m2"]),
        )

    @property
    def sample_variance(self) -> float:
        return self.m2 / float(self.count - 1) if self.count > 1 else 0.0

    @property
    def standard_error(self) -> float:
        if self.count <= 0:
            return math.inf
        return math.sqrt(max(0.0, self.sample_variance / float(self.count)))

    def merge(self, other: "RunningMoment") -> "RunningMoment":
        if self.count == 0:
            return other
        if other.count == 0:
            return self
        total = self.count + other.count
        delta = other.mean - self.mean
        return RunningMoment(
            count=total,
            mean=self.mean + delta * float(other.count) / float(total),
            m2=(
                self.m2
                + other.m2
                + delta * delta * float(self.count) * float(other.count) / float(total)
            ),
        )

    def scaled(self, scale: float) -> "RunningMoment":
        factor = float(scale)
        if not math.isfinite(factor):
            raise ValueError("running-moment scale must be finite")
        return RunningMoment(
            count=self.count,
            mean=self.mean * factor,
            m2=self.m2 * factor * factor,
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean": self.mean,
            "m2": self.m2,
            "sample_variance": self.sample_variance,
        }


def _stratum_population_counts(
    population_counts: Mapping[str, int],
    *,
    strata: Sequence[str],
) -> dict[str, int]:
    resolved: dict[str, int] = {}
    for key in strata:
        if key not in population_counts:
            raise ValueError(
                "stratified Student-t finite populations must cover every stratum"
            )
        population = int(population_counts[key])
        if population <= 0:
            raise ValueError("stratified Student-t finite populations must be positive")
        resolved[key] = population
    return resolved


def stratified_student_t_summary_from_moments(
    moments: Mapping[str, RunningMoment | Mapping[str, Any]],
    *,
    stratum_weights: Mapping[str, float] | Sequence[tuple[str, float]],
    alpha: float,
    population_counts: Mapping[str, int] | None = None,
) -> StratifiedStudentTSummary:
    """Project mergeable stratum moments with Satterthwaite degrees of freedom.

    ``population_counts`` (stratum -> finite population size ``N_h``) selects
    the sealed without-replacement law: every stratum requires
    ``2 <= n_h <= N_h`` and contributes ``w_h^2 s_h^2 (1 - n_h / N_h) / n_h``
    to the variance of the weighted mean, with Satterthwaite degrees of
    freedom taken over the corrected terms.  A stratum at census
    (``n_h == N_h``) contributes exactly zero to both sums; when every stratum
    is at census the summary is the existing ``degenerate_zero_variance``
    branch (zero standard error, interval equal to the mean).

    Invariant: with ``population_counts=None`` every floating-point
    expression is evaluated in exactly the historical order, so the
    with-replacement summary is bitwise identical to the pre-v4 function.
    """

    normalized_moments = {
        str(key): (
            value
            if isinstance(value, RunningMoment)
            else RunningMoment.from_mapping(value)
        )
        for key, value in moments.items()
    }
    weights = dict(_normalized_weights(stratum_weights))
    if set(normalized_moments) != set(weights) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("stratified Student-t moment design is invalid")
    populations = (
        None
        if population_counts is None
        else _stratum_population_counts(population_counts, strata=tuple(weights))
    )
    counts: list[tuple[str, int]] = []
    means: list[tuple[str, float]] = []
    variances: list[tuple[str, float]] = []
    variance_terms: list[tuple[float, int]] = []
    population_rows: list[tuple[str, int]] = []
    census_strata = 0
    aggregate_mean = 0.0
    for key, weight in sorted(weights.items()):
        moment = normalized_moments[key]
        if moment.count < 2:
            raise ValueError(
                "stratified Student-t requires at least two rows per stratum"
            )
        sample_variance = moment.sample_variance
        if populations is None:
            term = float(weight) ** 2 * sample_variance / float(moment.count)
        else:
            population = populations[key]
            if moment.count > population:
                raise ValueError(
                    "sealed without-replacement stratum holds more rows than "
                    "its finite population"
                )
            # (1 - n/N) is exactly 0.0 at census: n/N divides equal integers.
            term = (
                float(weight) ** 2
                * sample_variance
                * (1.0 - float(moment.count) / float(population))
                / float(moment.count)
            )
            population_rows.append((key, population))
            if moment.count == population:
                census_strata += 1
        aggregate_mean += float(weight) * moment.mean
        counts.append((key, moment.count))
        means.append((key, moment.mean))
        variances.append((key, sample_variance))
        variance_terms.append((term, moment.count))
    variance_of_mean = float(sum(term for term, _count in variance_terms))
    denominator = float(
        sum(term * term / float(count - 1) for term, count in variance_terms)
    )
    degenerate_zero_variance = variance_of_mean <= 0.0
    if degenerate_zero_variance:
        degrees_of_freedom: float | None = None
        interval_lower = aggregate_mean
        interval_upper = aggregate_mean
    else:
        if denominator <= 0.0:
            raise RuntimeError(
                "positive stratified variance has no Satterthwaite denominator"
            )
        degrees_of_freedom = variance_of_mean * variance_of_mean / denominator
        interval = student_t_interval(
            aggregate_mean,
            variance_of_mean,
            1,
            alpha=float(alpha),
            dof=degrees_of_freedom,
        )
        interval_lower = float(interval.lower[()])
        interval_upper = float(interval.upper[()])
    if populations is None:
        return StratifiedStudentTSummary(
            mean=aggregate_mean,
            lower=interval_lower,
            upper=interval_upper,
            standard_error=math.sqrt(max(0.0, variance_of_mean)),
            degrees_of_freedom=degrees_of_freedom,
            degenerate_zero_variance=degenerate_zero_variance,
            counts_by_stratum=tuple(counts),
            means_by_stratum=tuple(means),
            variances_by_stratum=tuple(variances),
            alpha=float(alpha),
        )
    return StratifiedStudentTSummary(
        mean=aggregate_mean,
        lower=interval_lower,
        upper=interval_upper,
        standard_error=math.sqrt(max(0.0, variance_of_mean)),
        degrees_of_freedom=degrees_of_freedom,
        degenerate_zero_variance=degenerate_zero_variance,
        counts_by_stratum=tuple(counts),
        means_by_stratum=tuple(means),
        variances_by_stratum=tuple(variances),
        alpha=float(alpha),
        finite_population_correction_applied=True,
        population_by_stratum=tuple(population_rows),
        census_complete=census_strata == len(counts),
    )


def stratified_student_t_summary_from_moment_arrays(
    names: Sequence[str],
    counts: np.ndarray,
    means: np.ndarray,
    m2: np.ndarray,
    *,
    stratum_weights: Mapping[str, float] | Sequence[tuple[str, float]],
    alpha: float,
    population_counts: Mapping[str, int] | None = None,
) -> StratifiedStudentTSummary:
    """:func:`stratified_student_t_summary_from_moments` on arrays.

    ``names[h]`` owns ``counts[h]``, ``means[h]`` and ``m2[h]`` (the
    ``RunningMoment`` fields).  The result is BITWISE the dict function's on
    the same moments: every per-stratum term is the same IEEE expression with
    the same association, the weighted mean is the same sequential ``+=``
    loop, and the two variance sums are the same builtin ``sum`` over the
    same float sequence in the same (sorted-key) order.  Only the per-stratum
    Python objects are gone.
    """

    names = tuple(str(value) for value in names)
    counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    means = np.asarray(means, dtype=np.float64).reshape(-1)
    m2 = np.maximum(np.asarray(m2, dtype=np.float64).reshape(-1), 0.0)
    if not (counts.size == means.size == m2.size == len(names)):
        raise ValueError("stratified Student-t moment arrays are misaligned")
    if (counts < 0).any():
        raise ValueError("running-moment count must be nonnegative")
    if not (np.isfinite(means).all() and np.isfinite(m2).all()):
        raise ValueError("running moments must be finite")
    weights = dict(_normalized_weights(stratum_weights))
    if set(names) != set(weights) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("stratified Student-t moment design is invalid")
    populations = (
        None
        if population_counts is None
        else _stratum_population_counts(population_counts, strata=tuple(weights))
    )
    position = {name: index for index, name in enumerate(names)}
    ordered_keys = [key for key, _weight in sorted(weights.items())]
    order = np.asarray([position[key] for key in ordered_keys], dtype=np.int64)
    n = counts[order]
    short = np.flatnonzero(n < 2)
    if populations is not None:
        population_array = np.asarray(
            [populations[key] for key in ordered_keys], dtype=np.int64
        )
        over = np.flatnonzero(n > population_array)
    else:
        population_array = None
        over = np.zeros(0, dtype=np.int64)
    if short.size or over.size:
        first_short = int(short[0]) if short.size else len(ordered_keys)
        first_over = int(over[0]) if over.size else len(ordered_keys)
        if first_short <= first_over:
            raise ValueError(
                "stratified Student-t requires at least two rows per stratum"
            )
        raise ValueError(
            "sealed without-replacement stratum holds more rows than "
            "its finite population"
        )
    n_f = n.astype(np.float64)
    ordered_means = means[order]
    sample_variance = m2[order] / (n_f - 1.0)
    weight_values = [float(weights[key]) for key in ordered_keys]
    weight_squared = np.asarray([value**2 for value in weight_values], dtype=np.float64)
    if population_array is None:
        term = weight_squared * sample_variance / n_f
    else:
        term = (
            weight_squared
            * sample_variance
            * (1.0 - n_f / population_array.astype(np.float64))
            / n_f
        )
    aggregate_mean = 0.0
    for weight, mean in zip(weight_values, ordered_means.tolist()):
        aggregate_mean += weight * mean
    term_values = term.tolist()
    count_values = n.tolist()
    variance_of_mean = float(sum(term_values))
    denominator = float(
        sum((term * term / (n_f - 1.0)).tolist())
    )
    degenerate_zero_variance = variance_of_mean <= 0.0
    if degenerate_zero_variance:
        degrees_of_freedom: float | None = None
        interval_lower = aggregate_mean
        interval_upper = aggregate_mean
    else:
        if denominator <= 0.0:
            raise RuntimeError(
                "positive stratified variance has no Satterthwaite denominator"
            )
        degrees_of_freedom = variance_of_mean * variance_of_mean / denominator
        interval = student_t_interval(
            aggregate_mean,
            variance_of_mean,
            1,
            alpha=float(alpha),
            dof=degrees_of_freedom,
        )
        interval_lower = float(interval.lower[()])
        interval_upper = float(interval.upper[()])
    counts_rows = tuple(zip(ordered_keys, count_values))
    means_rows = tuple(zip(ordered_keys, ordered_means.tolist()))
    variance_rows = tuple(zip(ordered_keys, sample_variance.tolist()))
    del term_values
    if populations is None:
        return StratifiedStudentTSummary(
            mean=aggregate_mean,
            lower=interval_lower,
            upper=interval_upper,
            standard_error=math.sqrt(max(0.0, variance_of_mean)),
            degrees_of_freedom=degrees_of_freedom,
            degenerate_zero_variance=degenerate_zero_variance,
            counts_by_stratum=counts_rows,
            means_by_stratum=means_rows,
            variances_by_stratum=variance_rows,
            alpha=float(alpha),
        )
    return StratifiedStudentTSummary(
        mean=aggregate_mean,
        lower=interval_lower,
        upper=interval_upper,
        standard_error=math.sqrt(max(0.0, variance_of_mean)),
        degrees_of_freedom=degrees_of_freedom,
        degenerate_zero_variance=degenerate_zero_variance,
        counts_by_stratum=counts_rows,
        means_by_stratum=means_rows,
        variances_by_stratum=variance_rows,
        alpha=float(alpha),
        finite_population_correction_applied=True,
        population_by_stratum=tuple(
            zip(ordered_keys, population_array.tolist())
        ),
        census_complete=bool((n == population_array).all()),
    )


def stratified_student_t_summary(
    values: Sequence[float] | np.ndarray,
    strata: Sequence[str],
    *,
    stratum_weights: Mapping[str, float] | Sequence[tuple[str, float]],
    alpha: float,
    population_counts: Mapping[str, int] | None = None,
) -> StratifiedStudentTSummary:
    """Project unequal stratified panels with Satterthwaite degrees of freedom.

    ``population_counts`` is forwarded to
    :func:`stratified_student_t_summary_from_moments`; ``None`` keeps the
    historical with-replacement arithmetic bitwise.
    """

    observations = np.asarray(values, dtype=np.float64).reshape(-1)
    labels = tuple(str(value) for value in strata)
    weights = dict(_normalized_weights(stratum_weights))
    if (
        observations.size != len(labels)
        or not np.isfinite(observations).all()
        or set(labels) != set(weights)
        or not 0.0 < float(alpha) < 1.0
    ):
        raise ValueError("stratified Student-t evidence design is invalid")
    moments: dict[str, RunningMoment] = {}
    for key in sorted(weights):
        sample = observations[
            np.fromiter((label == key for label in labels), dtype=bool)
        ]
        moments[key] = RunningMoment.from_values(sample)
    return stratified_student_t_summary_from_moments(
        moments,
        stratum_weights=weights,
        alpha=alpha,
        population_counts=population_counts,
    )


def conservative_ratio_interval(
    *,
    numerator_estimate: float,
    numerator_lower: float,
    numerator_upper: float,
    positive_denominator_estimate: float,
    positive_denominator_lower: float,
    positive_denominator_upper: float,
    connection_delta: int,
) -> tuple[float, float, float]:
    """Conservative interval for numerator / (positive span * connections)."""

    denominator_count = int(connection_delta)
    if (
        denominator_count <= 0
        or positive_denominator_lower <= 0.0
        or not all(
            math.isfinite(float(value))
            for value in (
                numerator_estimate,
                numerator_lower,
                numerator_upper,
                positive_denominator_estimate,
                positive_denominator_lower,
                positive_denominator_upper,
            )
        )
    ):
        raise ValueError(
            "ratio interval requires finite values and positive denominator"
        )
    candidates = (
        float(numerator_lower) / float(positive_denominator_lower),
        float(numerator_lower) / float(positive_denominator_upper),
        float(numerator_upper) / float(positive_denominator_lower),
        float(numerator_upper) / float(positive_denominator_upper),
    )
    # Census point envelope. The
    # point estimate uses exactly the corners' operation sequence,
    # (n / s) / delta. With ordered inputs (n_lower <= n <= n_upper,
    # s_lower <= s <= s_upper) the real n / s lies between the real corner
    # extremes; correctly rounded IEEE-754 division is monotone, so fl(n / s)
    # lies between the rounded corner extremes, and dividing all three by the
    # same positive delta keeps the order. Hence lower <= estimate <= upper
    # holds EXACTLY in floating point, not up to a tolerance.
    #
    # The former n / (s * delta) rounds the product first. It can differ from
    # (n / s) / delta by up to two ulps in general (at most one on the logged
    # AlexNet span/delta values), in about 35% of evaluations. On a census
    # look every radius is exactly zero, the interval is a point, and that
    # rounding step put the estimate outside [lower, upper], so
    # JointDecisionConfidenceEnvelope rejected a valid envelope.
    #
    # What changes: the returned point estimate (and anything derived from or
    # hashed over it, such as envelope/receipt content hashes and forecast
    # inputs) can move by that rounding step on ANY look, not only at census.
    # lower and upper are computed exactly as before, so they are bitwise
    # unchanged, and so are the accept (upper <= tau) and reject
    # (lower > tau) rules. A rule that reads the estimate itself (the
    # mean >= tau no-commit branch) can differ only when the estimate lies
    # within that rounding step of tau.
    return (
        float(numerator_estimate)
        / float(positive_denominator_estimate)
        / denominator_count,
        min(candidates) / denominator_count,
        max(candidates) / denominator_count,
    )


def forecast_cumulative_target(
    *,
    current_count: int,
    current_standard_error: float,
    estimate: float,
    accept_boundary: float,
    reject_boundary: float | None = None,
    current_radius: float,
    minimum_increment: int,
    maximum_count: int | None,
) -> int:
    """Forecast and jump to the next decision-resolving cumulative target."""

    count = int(current_count)
    cap = None if maximum_count is None else int(maximum_count)
    increment = int(minimum_increment)
    reject = float(accept_boundary if reject_boundary is None else reject_boundary)
    if (
        count <= 0
        or (cap is not None and cap < count)
        or increment <= 0
        or not all(
            math.isfinite(float(value))
            for value in (
                current_standard_error,
                estimate,
                accept_boundary,
                reject,
                current_radius,
            )
        )
        or reject < float(accept_boundary)
    ):
        raise ValueError("invalid cumulative-target forecast inputs")
    if cap is not None and count >= cap:
        return count
    if float(estimate) <= float(accept_boundary):
        gap = float(accept_boundary) - float(estimate)
    elif float(estimate) > reject:
        gap = float(estimate) - reject
    else:
        # The estimate itself lies in the indifference region. More precision
        # cannot certify either action without a material estimate change.
        return count + max(increment, count) if cap is None else cap
    if gap <= 0.0 or current_standard_error <= 0.0:
        target = count + increment
        return target if cap is None else min(cap, target)
    required = math.ceil(
        count * (max(float(current_radius), 0.0) / max(gap, 1e-15)) ** 2
    )
    required = max(count + increment, required)
    return required if cap is None else min(cap, required)


def validate_forecast_ratio(
    *,
    realized: float,
    predicted: float,
    lower: float = 0.5,
    upper: float = 2.0,
) -> tuple[float | None, bool]:
    """Check a realized quantity against what a forecast predicted for it.

    Returns ``(ratio, validated)``. ``ratio`` is ``None`` -- and ``validated``
    is ``False`` -- whenever no ratio can be formed (a non-positive or
    non-finite prediction, a non-finite realization, or no prior forecast to
    compare against). Returning ``False`` rather than carrying a previous
    verdict forward is the point: a look that cannot be checked has not been
    validated, and a stale ``True`` would release a growth guard on evidence
    that was never produced.

    BINDING RULE -- a forecast may claim validation only when the realized
    quantity compared here is the SAME QUANTITY the forecast predicted.

    That rule is why two callers in this tree behave differently, and the
    difference is deliberate rather than an oversight:

    * ``_analytic_calibration_forecast`` predicts a bound and is checked
      against the realized bound, against a FIXED pre-declared epsilon. The
      predicted and realized quantities coincide, so the flag is earned and
      is fed back into the growth guard.
    * ``forecast_cumulative_target`` extrapolates the distance to a decision
      boundary, while the available instrument measures the interval RADIUS.
      Confirming the radius model says nothing about the boundary distance,
      which is random, sits squared in the denominator, and therefore has
      infinite expectation. Its caller keeps the guard permanently armed and
      emits this ratio as telemetry only.
    """

    predicted_value = float(predicted)
    realized_value = float(realized)
    if (
        not math.isfinite(predicted_value)
        or predicted_value <= 0.0
        or not math.isfinite(realized_value)
    ):
        return None, False
    ratio = realized_value / predicted_value
    return ratio, bool(float(lower) <= ratio <= float(upper))


def guard_unvalidated_cumulative_forecast(
    *,
    current_count: int,
    analytic_target: int,
    minimum_increment: int,
    forecast_validated: bool,
    maximum_growth_factor: float = 4.0,
) -> tuple[int, bool]:
    """Bound an unvalidated asymptotic forecast without imposing a horizon.

    ``forecast_cumulative_target`` extrapolates the usual inverse-square-root
    contraction law.  That law may be accurate, but a single noisy look is
    insufficient evidence for jumping directly by several orders of
    magnitude.  Until one forecast has been checked against a subsequent
    realized interval radius, cap the *next cumulative target* at a fixed
    geometric multiple of the current count.  This is a forecast safeguard,
    not a statistical sample cap: repeated looks may grow without limit, and
    a validated forecast may jump directly to its analytic target.
    """

    count = int(current_count)
    target = int(analytic_target)
    increment = int(minimum_increment)
    factor = float(maximum_growth_factor)
    if (
        count <= 0
        or target < count
        or increment <= 0
        or not math.isfinite(factor)
        or factor <= 1.0
    ):
        raise ValueError("invalid guarded cumulative-forecast inputs")
    if forecast_validated:
        return max(count + increment, target), False
    guarded_target = max(
        count + increment,
        int(math.ceil(factor * float(count))),
    )
    selected = min(target, guarded_target)
    return selected, selected < target


@dataclass(frozen=True, slots=True)
class SequentialBCEDecisionReceipt:
    """A v5 Fieller set may have two rays even when the signed test accepts.

    Acceptance tests the registered threshold with a positive span; the
    reported set inverts two-sided tests over all ratio values.
    """

    spec_hash: str
    proposal_id: str
    estimand_policy: SequentialBCEEstimandPolicy
    aggregate_estimate: float
    lower: float
    upper: float
    accept_boundary: float
    reject_boundary: float
    action: Literal["accept", "reject", "unresolved"]
    raw_stratum_statistics: Mapping[str, Mapping[str, float | int]]
    allocation_receipts: tuple[Mapping[str, Any], ...]
    calibration_state_ids: tuple[str, ...]
    resource_usage: Mapping[str, Any]
    connection_delta_receipt_hash: str | None = None
    baseline_risk_certificate_hash: str | None = None
    loss_span_certificate_hash: str | None = None
    optimized_calibration_envelope: Mapping[str, Any] = field(default_factory=dict)
    joint_precision_allocation_receipts: tuple[Mapping[str, Any], ...] = ()
    raw_numerator_estimate: float | None = None
    raw_numerator_lower: float | None = None
    raw_numerator_upper: float | None = None
    positive_loss_span_estimate: float | None = None
    positive_loss_span_lower: float | None = None
    positive_loss_span_upper: float | None = None
    connection_delta: int | None = None
    confidence_scope: str = PLANNED_LOOK_SCOPE
    claim_status: str = "exploratory_assumption_dependent"
    # Select/confirm replication: when the accept gate is armed, a
    # provisional accept must replicate on a disjoint confirm stream; the
    # confirm audit rides the decision receipt whether it passed or demoted
    # the accept.
    accept_replication: Mapping[str, Any] | None = None
    step_observables: Mapping[str, Any] = field(default_factory=dict)
    step_observables_enabled: bool | None = None
    # v4 decision-law disclosure.  These fields are serialized only under
    # ``sequential_bce_decision_receipt_v4``; v2/v3 payloads and hashes are
    # byte-identical to the pre-v4 receipt.
    oracle_sampling_law: str = IID_WITH_REPLACEMENT_SAMPLING_ID
    oracle_census_strata_count: int = 0
    oracle_census_exact_weight: float = 0.0
    oracle_census_complete: bool = False
    refinement_rounds: int = 0
    oracle_acquisition_envelope: Mapping[str, Any] | None = None
    fieller_interval: Mapping[str, Any] | None = None
    schema: str = "sequential_bce_decision_receipt_v3"

    _V4_FIELDS: ClassVar[tuple[str, ...]] = (
        "oracle_sampling_law",
        "oracle_census_strata_count",
        "oracle_census_exact_weight",
        "oracle_census_complete",
        "refinement_rounds",
        "oracle_acquisition_envelope",
    )

    def __post_init__(self) -> None:
        normalization_values = (
            self.raw_numerator_estimate,
            self.raw_numerator_lower,
            self.raw_numerator_upper,
            self.positive_loss_span_estimate,
            self.positive_loss_span_lower,
            self.positive_loss_span_upper,
            self.connection_delta,
        )
        normalization_present = any(value is not None for value in normalization_values)
        if (
            not self.spec_hash
            or not self.proposal_id
            or not self.calibration_state_ids
            or self.action not in {"accept", "reject", "unresolved"}
            or self.confidence_scope
            not in {
                PLANNED_LOOK_SCOPE,
                PRECISION_ONLY_IID_SCOPE,
            }
            or self.claim_status != "exploratory_assumption_dependent"
            or not all(
                math.isfinite(float(value))
                for value in (
                    self.aggregate_estimate,
                    self.lower,
                    self.upper,
                    self.accept_boundary,
                    self.reject_boundary,
                )
            )
            or self.lower > self.upper
            or self.accept_boundary > self.reject_boundary
            or self.schema
            not in {
                "sequential_bce_decision_receipt_v2",
                "sequential_bce_decision_receipt_v3",
                SEQUENTIAL_BCE_DECISION_RECEIPT_V4,
                SEQUENTIAL_BCE_DECISION_RECEIPT_V5,
            }
        ):
            raise ValueError("invalid sequential BCE decision receipt")
        exact_weight = float(self.oracle_census_exact_weight)
        if (
            self.oracle_sampling_law not in ORACLE_SAMPLING_LAWS
            or int(self.oracle_census_strata_count) < 0
            or int(self.refinement_rounds) < 0
            or not math.isfinite(exact_weight)
            or not 0.0 <= exact_weight <= 1.0 + 1e-9
            or (
                bool(self.oracle_census_complete)
                and not math.isclose(exact_weight, 1.0, rel_tol=0.0, abs_tol=1e-9)
            )
            or (
                self.oracle_acquisition_envelope is not None
                and not isinstance(self.oracle_acquisition_envelope, Mapping)
            )
        ):
            raise ValueError("invalid sequential BCE decision-law disclosure")
        if normalization_present:
            if (
                any(value is None for value in normalization_values)
                or self.estimand_policy != V0_NORMALIZED_LOSS_SPAN_PER_CONNECTION_V1
            ):
                raise ValueError(
                    "normalization audit coordinates must be complete and v0-only"
                )
            assert self.raw_numerator_estimate is not None
            assert self.raw_numerator_lower is not None
            assert self.raw_numerator_upper is not None
            assert self.positive_loss_span_estimate is not None
            assert self.positive_loss_span_lower is not None
            assert self.positive_loss_span_upper is not None
            assert self.connection_delta is not None
            if (
                not all(math.isfinite(float(value)) for value in normalization_values)
                or self.raw_numerator_lower > self.raw_numerator_upper
                or self.positive_loss_span_lower <= 0.0
                or self.positive_loss_span_lower > self.positive_loss_span_estimate
                or self.positive_loss_span_estimate > self.positive_loss_span_upper
                or int(self.connection_delta) <= 0
            ):
                raise ValueError("invalid v0 normalization audit coordinates")
            recomposed = float(self.raw_numerator_estimate) / (
                float(self.positive_loss_span_estimate) * int(self.connection_delta)
            )
            if not math.isclose(
                recomposed,
                float(self.aggregate_estimate),
                rel_tol=1e-10,
                abs_tol=1e-15,
            ):
                raise ValueError(
                    "v0 aggregate estimate does not equal raw BCE divided by "
                    "loss span and exact connection delta"
                )

    @property
    def content_hash(self) -> str:
        return sequential_bce_decision_receipt_content_hash(self._receipt_payload())

    def _receipt_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        if not self.step_observables:
            payload.pop("step_observables", None)
        if self.step_observables_enabled is None:
            payload.pop("step_observables_enabled", None)
        if self.schema not in {
            SEQUENTIAL_BCE_DECISION_RECEIPT_V4,
            SEQUENTIAL_BCE_DECISION_RECEIPT_V5,
        }:
            # Pre-v4 receipts never carried the decision-law block; keeping
            # it out of their payload keeps their bytes and hashes unchanged.
            for name in self._V4_FIELDS:
                payload.pop(name, None)
        if self.schema != SEQUENTIAL_BCE_DECISION_RECEIPT_V5:
            payload.pop("fieller_interval", None)
        return payload

    @property
    def receipt(self) -> Mapping[str, Any]:
        return {
            **self._receipt_payload(),
            "content_hash": self.content_hash,
        }


__all__ = [
    "ANALYTIC_BALANCED_BINARY_CHANCE_RISK_V1",
    "ConnectionDelta",
    "BASELINE_EXCESS_CALIBRATED_BCE_V1",
    "AnalyticChanceRiskCertificate",
    "BaselineRiskCertificate",
    "CalibrationImprovementProfile",
    "CalibrationForecastReceipt",
    "JointPrecisionActionForecast",
    "JointDecisionConfidenceEnvelope",
    "JOINT_DECISION_WALD_METHOD",
    "SCALAR_DECISION_FIELLER_METHOD",
    "OptimizedCalibrationDecisionEnvelope",
    "select_joint_precision_action",
    "scalar_precision_forecasts",
    "CircuitConnectionDeltaReceipt",
    "IID_WITH_REPLACEMENT_SAMPLING_ID",
    "ORACLE_SAMPLING_LAWS",
    "SEALED_WOR_CLASS_STRATIFIED_SAMPLING_ID",
    "SEQUENTIAL_BCE_DECISION_RECEIPT_V4",
    "SEQUENTIAL_BCE_DECISION_RECEIPT_V5",
    "SEQUENTIAL_CALIBRATED_BCE_V4",
    "SEQUENTIAL_CALIBRATED_BCE_V5",
    "ScalarV0DecisionEnvelope",
    "scalar_v0_decision_interval",
    "fieller_ratio_interval",
    "STRATIFIED_T_FPC_ALPHA_SPENDING_METHOD",
    "STRATIFIED_T_FPC_METHOD",
    "LossSpanCertificate",
    "PLANNED_LOOK_SCOPE",
    "PRECISION_ONLY_IID_SCOPE",
    "RiskIntervalCertificate",
    "SEALED_RESPONSE_SAMPLING_ID",
    "SEQUENTIAL_CALIBRATED_BCE_V2",
    "SEQUENTIAL_CALIBRATED_BCE_V3",
    "STRATIFIED_T_METHOD",
    "STRATIFIED_T_ALPHA_SPENDING_METHOD",
    "SequentialBCEDecisionReceipt",
    "SequentialCalibratedBCEResolutionSpec",
    "StratifiedStudentTSummary",
    "V0_NORMALIZED_LOSS_SPAN_PER_CONNECTION_V1",
    "conservative_ratio_interval",
    "forecast_cumulative_target",
    "guard_unvalidated_cumulative_forecast",
    "sequential_bce_decision_receipt_content_hash",
    "joint_v0_decision_interval",
    "joint_v1_decision_interval",
    "joint_wald_radius_squared",
    "stratified_student_t_summary",
]
