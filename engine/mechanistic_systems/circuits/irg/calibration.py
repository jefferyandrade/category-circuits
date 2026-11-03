"""Decision-relevant calibration uncertainty for the unified measurement economy.

Numerical optimizer convergence and statistical calibration information are
separate.  This module prices additional cached-logit acquisition or an exact
refit only through its contribution to unresolved candidate contrasts.  It
never reads an optional terminal audit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property
from itertools import combinations
from typing import Callable, Mapping, Sequence

import numpy as np

from .acquisition import (
    AcquisitionForecast,
    DecisionUncertaintySnapshot,
    UnifiedAcquisitionAdapter,
)
from .contracts import (
    ContextSelector,
    MeasurementProvenance,
    MeasurementRequest,
    HistoricalResponseProfileName,
    content_hash,
)
from .costs import CostEstimate
from .precision import simultaneous_normal_critical_value


CALIBRATION_UNCERTAINTY_CACHE_KEY = "irg_calibration_uncertainty_state_v1"
CALIBRATION_CONTRASTS_CACHE_KEY = "irg_calibration_active_contrasts_v1"
CALIBRATION_FROZEN_BOUNDARY_REFERENCE_ID = "__irg_frozen_accept_reject_boundary_v1__"


def _semantic_version_state_hash(session: object) -> str:
    state = getattr(session, "version_state", None)
    identity = getattr(state, "state_hash", None)
    if callable(identity):
        identity = identity()
    if isinstance(identity, str) and identity:
        return identity
    try:
        return content_hash(state, prefix="irg-calibration-version-state-")
    except TypeError:
        return content_hash(
            repr(state),
            prefix="irg-calibration-version-state-repr-",
        )


def _active_response_calibration_id(session: object) -> str:
    """Return the unambiguous calibration identity of the active chart."""

    active_state = getattr(session, "version_state", None)
    intrinsic = getattr(active_state, "intrinsic_state", None)
    candidates = tuple(
        str(value)
        for value in (
            getattr(active_state, "calibration_state_id", ""),
            getattr(intrinsic, "calibration_state_id", ""),
        )
        if str(value)
    )
    identities = set(candidates)
    if len(identities) > 1:
        raise RuntimeError(
            "active response chart has ambiguous calibration-state lineage"
        )
    return next(iter(identities), "")


@dataclass(frozen=True)
class CalibrationUncertaintyState:
    calibration_state_id: str
    parameter_mean: np.ndarray
    parameter_covariance: np.ndarray
    fitted_context_count: int
    numerical_converged: bool
    calibration_fit_state_hash: str = ""
    cached_logit_keys: tuple[str, ...] = ()
    available_context_count: int = 0
    calibration_support_design_hash: str = ""
    calibration_support_capacity: int = 0
    sampling_with_replacement: bool = False
    diagnostics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mean = np.asarray(self.parameter_mean, dtype=np.float64).reshape(-1).copy()
        covariance = np.asarray(self.parameter_covariance, dtype=np.float64).copy()
        if not self.calibration_state_id or mean.size == 0:
            raise ValueError("calibration uncertainty state has an empty identity")
        if (
            covariance.shape != (mean.size, mean.size)
            or not np.isfinite(mean).all()
            or not np.isfinite(covariance).all()
        ):
            raise ValueError("calibration mean/covariance are invalid")
        covariance = 0.5 * (covariance + covariance.T)
        if float(np.min(np.linalg.eigvalsh(covariance))) < -1e-10:
            raise ValueError("calibration covariance must be PSD")
        if int(self.fitted_context_count) < 0:
            raise ValueError("fitted_context_count must be nonnegative")
        if int(self.available_context_count) < 0:
            raise ValueError("available_context_count must be nonnegative")
        if int(self.calibration_support_capacity) < 0:
            raise ValueError("calibration_support_capacity must be nonnegative")
        support_hash = str(self.calibration_support_design_hash)
        if support_hash and (
            len(support_hash) != 64
            or any(character not in "0123456789abcdef" for character in support_hash)
        ):
            raise ValueError(
                "calibration_support_design_hash must be a lowercase SHA-256 digest"
            )
        if int(
            self.calibration_support_capacity
        ) > 0 and not bool(
            self.sampling_with_replacement
        ) and self.planning_context_count > int(self.calibration_support_capacity):
            raise ValueError(
                "calibration planning rows exceed the frozen support capacity"
            )
        fit_hash = str(self.calibration_fit_state_hash)
        if fit_hash and (
            len(fit_hash) != 64
            or any(character not in "0123456789abcdef" for character in fit_hash)
        ):
            raise ValueError(
                "calibration_fit_state_hash must be a lowercase SHA-256 digest"
            )
        diagnostics = {
            str(key): float(value) for key, value in self.diagnostics.items()
        }
        if any(not math.isfinite(value) for value in diagnostics.values()):
            raise ValueError("calibration diagnostics must be finite")
        cached_keys = tuple(str(value) for value in self.cached_logit_keys)
        if len(cached_keys) != len(set(cached_keys)) or any(
            not value for value in cached_keys
        ):
            raise ValueError("cached calibration-logit keys must be unique identities")
        mean.setflags(write=False)
        covariance.setflags(write=False)
        object.__setattr__(self, "parameter_mean", mean)
        object.__setattr__(self, "parameter_covariance", covariance)
        object.__setattr__(self, "fitted_context_count", int(self.fitted_context_count))
        object.__setattr__(self, "numerical_converged", bool(self.numerical_converged))
        object.__setattr__(self, "calibration_fit_state_hash", fit_hash)
        object.__setattr__(self, "cached_logit_keys", cached_keys)
        object.__setattr__(
            self,
            "available_context_count",
            int(self.available_context_count),
        )
        object.__setattr__(
            self,
            "calibration_support_design_hash",
            support_hash,
        )
        object.__setattr__(
            self,
            "calibration_support_capacity",
            int(self.calibration_support_capacity),
        )
        object.__setattr__(
            self,
            "sampling_with_replacement",
            bool(self.sampling_with_replacement),
        )
        object.__setattr__(self, "diagnostics", diagnostics)

    @property
    def parameter_dimension(self) -> int:
        return int(self.parameter_mean.size)

    @cached_property
    def state_hash(self) -> str:
        return content_hash(
            {
                "calibration_state_id": self.calibration_state_id,
                "parameter_mean": self.parameter_mean,
                "parameter_covariance": self.parameter_covariance,
                "fitted_context_count": self.fitted_context_count,
                "numerical_converged": self.numerical_converged,
                "calibration_fit_state_hash": self.calibration_fit_state_hash,
                "cached_logit_keys": self.cached_logit_keys,
                "available_context_count": int(self.available_context_count),
                "calibration_support_design_hash": (
                    self.calibration_support_design_hash
                ),
                "calibration_support_capacity": int(self.calibration_support_capacity),
                "diagnostics": self.diagnostics,
            },
            prefix="irg-calibration-state-",
        )

    @cached_property
    def inferential_hash(self) -> str:
        """Identity of the fitted distribution, excluding reusable logit cache keys."""

        return content_hash(
            {
                "calibration_state_id": self.calibration_state_id,
                "parameter_mean": self.parameter_mean,
                "parameter_covariance": self.parameter_covariance,
                "fitted_context_count": self.fitted_context_count,
                "numerical_converged": self.numerical_converged,
                "calibration_fit_state_hash": self.calibration_fit_state_hash,
                "diagnostics": self.diagnostics,
            },
            prefix="irg-calibration-inference-",
        )

    @property
    def planning_context_count(self) -> int:
        """Role-safe calibration rows available to the next exact refit."""

        return max(
            int(self.fitted_context_count),
            int(self.available_context_count),
        )


@dataclass(frozen=True)
class CalibrationSensitivity:
    candidate_id: str
    gradient: np.ndarray

    def __post_init__(self) -> None:
        gradient = np.asarray(self.gradient, dtype=np.float64).reshape(-1).copy()
        if (
            not self.candidate_id
            or gradient.size == 0
            or not np.isfinite(gradient).all()
        ):
            raise ValueError("calibration sensitivity is invalid")
        gradient.setflags(write=False)
        object.__setattr__(self, "gradient", gradient)


@dataclass(frozen=True)
class CalibrationContrastUncertainty:
    candidate_a: str
    candidate_b: str
    variance: float
    standard_error: float
    decision_relevant: bool
    indifference: float
    contrast_kind: str = "candidate_pair"

    def __post_init__(self) -> None:
        if (
            not self.candidate_a
            or not self.candidate_b
            or self.candidate_a == self.candidate_b
        ):
            raise ValueError(
                "calibration contrast requires two distinct candidate identities"
            )
        if self.contrast_kind not in {
            "candidate_pair",
            "candidate_to_frozen_boundary",
        }:
            raise ValueError("unsupported calibration contrast kind")
        if (
            self.contrast_kind == "candidate_to_frozen_boundary"
            and self.candidate_b != CALIBRATION_FROZEN_BOUNDARY_REFERENCE_ID
        ):
            raise ValueError(
                "candidate-boundary calibration contrast has the wrong reference"
            )
        if (
            self.contrast_kind == "candidate_pair"
            and CALIBRATION_FROZEN_BOUNDARY_REFERENCE_ID
            in {self.candidate_a, self.candidate_b}
        ):
            raise ValueError(
                "candidate-pair calibration contrast cannot use the boundary "
                "reference"
            )
        variance = float(self.variance)
        standard_error = float(self.standard_error)
        indifference = float(self.indifference)
        if (
            not math.isfinite(variance)
            or variance < 0.0
            or not math.isfinite(standard_error)
            or standard_error < 0.0
            or not math.isfinite(indifference)
            or indifference < 0.0
        ):
            raise ValueError("calibration contrast uncertainty is invalid")
        if not math.isclose(
            standard_error * standard_error,
            variance,
            rel_tol=1e-8,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "calibration contrast standard error differs from its variance"
            )
        object.__setattr__(self, "variance", variance)
        object.__setattr__(self, "standard_error", standard_error)
        object.__setattr__(self, "indifference", indifference)
        object.__setattr__(self, "decision_relevant", bool(self.decision_relevant))


def calibration_search_identity(
    search: object,
) -> tuple[tuple[str, ...], str]:
    """Bind calibration guidance to one frozen complete-mask boundary chart."""

    points = tuple(getattr(search, "points", ()))
    if not points:
        raise ValueError("calibration contrasts require a nonempty boundary search")
    rows: list[dict[str, object]] = []
    candidate_ids: list[str] = []
    for point in points:
        candidate = getattr(point, "candidate", None)
        candidate_id = str(getattr(candidate, "candidate_id", ""))
        if not candidate_id:
            raise TypeError(
                "calibration boundary point lacks a typed candidate identity"
            )
        candidate_ids.append(candidate_id)
        rows.append(
            {
                "candidate_id": candidate_id,
                "released_cost": float(getattr(point, "released_cost")),
                "predicted_quality_loss": float(
                    getattr(point, "predicted_quality_loss")
                ),
                "predicted_standard_error": float(
                    getattr(point, "predicted_standard_error")
                ),
                "expected_oracle_cost": float(getattr(point, "expected_oracle_cost")),
            }
        )
    identifiers = tuple(candidate_ids)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("calibration boundary search has duplicate candidates")
    controls = {
        name: float(getattr(search, name))
        for name in (
            "maximum_quality_loss",
            "practical_indifference",
            "cost_tolerance",
            "monotonicity_tolerance",
            "z_value",
            "resample_standard_error",
        )
    }
    return identifiers, content_hash(
        {
            "points": tuple(rows),
            "controls": controls,
        },
        prefix="irg-calibration-search-",
    )


@dataclass(frozen=True)
class CalibrationContrastSnapshot:
    """Full active-pair calibration uncertainty frozen before acquisition."""

    parent_state_hash: str
    response_state_hash: str
    calibration_inferential_hash: str
    search_identity_hash: str
    candidate_ids: tuple[str, ...]
    sensitivities: tuple[CalibrationSensitivity, ...]
    contrasts: tuple[CalibrationContrastUncertainty, ...]
    indifference: float
    z_value: float
    producer_id: str
    producer_version: str
    code_hash: str
    coverage: str = "all_active_candidate_pairs"
    schema: str = "irg_calibration_contrast_snapshot_v1"

    def __post_init__(self) -> None:
        identity_values = (
            self.parent_state_hash,
            self.response_state_hash,
            self.calibration_inferential_hash,
            self.search_identity_hash,
            self.producer_id,
            self.producer_version,
            self.code_hash,
            self.schema,
        )
        if any(not str(value) for value in identity_values):
            raise ValueError(
                "calibration contrast snapshot requires complete provenance"
            )
        candidate_ids = tuple(str(value) for value in self.candidate_ids)
        sensitivities = tuple(self.sensitivities)
        contrasts = tuple(self.contrasts)
        if (
            not candidate_ids
            or len(set(candidate_ids)) != len(candidate_ids)
            or any(not value for value in candidate_ids)
        ):
            raise ValueError(
                "calibration contrast snapshot requires a unique candidate axis"
            )
        if self.coverage != "all_active_candidate_pairs":
            raise ValueError("unsupported calibration contrast coverage contract")
        if (
            len(sensitivities) != len(candidate_ids)
            or tuple(item.candidate_id for item in sensitivities) != candidate_ids
            or len({item.gradient.size for item in sensitivities}) != 1
        ):
            raise ValueError(
                "calibration sensitivity axis must match every active candidate"
            )
        indifference = float(self.indifference)
        z_value = float(self.z_value)
        if (
            not math.isfinite(indifference)
            or indifference < 0.0
            or not math.isfinite(z_value)
            or z_value <= 0.0
        ):
            raise ValueError("calibration contrast decision controls are invalid")
        expected_pairs = (
            *(
                (
                    candidate_id,
                    CALIBRATION_FROZEN_BOUNDARY_REFERENCE_ID,
                    "candidate_to_frozen_boundary",
                )
                for candidate_id in candidate_ids
            ),
            *(
                (left, right, "candidate_pair")
                for left, right in combinations(candidate_ids, 2)
            ),
        )
        observed_pairs = tuple(
            (item.candidate_a, item.candidate_b, item.contrast_kind)
            for item in contrasts
        )
        if observed_pairs != expected_pairs:
            raise ValueError(
                "calibration snapshot must cover every active candidate pair "
                "exactly once in boundary order"
            )
        if any(
            not math.isclose(
                item.indifference,
                indifference,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            for item in contrasts
        ):
            raise ValueError(
                "calibration contrasts differ from the frozen indifference"
            )
        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(self, "sensitivities", sensitivities)
        object.__setattr__(self, "contrasts", contrasts)
        object.__setattr__(self, "indifference", indifference)
        object.__setattr__(self, "z_value", z_value)

    @cached_property
    def content_hash(self) -> str:
        return content_hash(
            {
                "parent_state_hash": self.parent_state_hash,
                "response_state_hash": self.response_state_hash,
                "calibration_inferential_hash": self.calibration_inferential_hash,
                "search_identity_hash": self.search_identity_hash,
                "candidate_ids": self.candidate_ids,
                "sensitivities": self.sensitivities,
                "contrasts": self.contrasts,
                "indifference": self.indifference,
                "z_value": self.z_value,
                "producer_id": self.producer_id,
                "producer_version": self.producer_version,
                "code_hash": self.code_hash,
                "coverage": self.coverage,
                "schema": self.schema,
            },
            prefix="irg-calibration-contrasts-",
        )

    def assert_current(
        self,
        *,
        session: object,
        calibration_state: CalibrationUncertaintyState,
        search: object | None,
    ) -> None:
        if self.parent_state_hash != str(getattr(session, "state_hash")):
            raise RuntimeError(
                "calibration contrasts were computed for a different parent state"
            )
        if self.response_state_hash != _semantic_version_state_hash(session):
            raise RuntimeError(
                "calibration contrasts were computed for a stale response chart"
            )
        if self.calibration_inferential_hash != calibration_state.inferential_hash:
            raise RuntimeError(
                "calibration contrasts were computed from stale calibration "
                "covariance"
            )
        if any(
            item.gradient.size != calibration_state.parameter_dimension
            for item in self.sensitivities
        ):
            raise RuntimeError(
                "calibration sensitivity dimension differs from the active fit"
            )
        sensitivity_by_id = {item.candidate_id: item for item in self.sensitivities}
        boundary = CalibrationSensitivity(
            CALIBRATION_FROZEN_BOUNDARY_REFERENCE_ID,
            np.zeros(calibration_state.parameter_dimension, dtype=np.float64),
        )
        expected_contrasts = (
            *(
                calibration_contrast_uncertainty(
                    calibration_state,
                    sensitivity_by_id[candidate_id],
                    boundary,
                    indifference=self.indifference,
                    z_value=self.z_value,
                    contrast_kind="candidate_to_frozen_boundary",
                )
                for candidate_id in self.candidate_ids
            ),
            *(
                calibration_contrast_uncertainty(
                    calibration_state,
                    sensitivity_by_id[left],
                    sensitivity_by_id[right],
                    indifference=self.indifference,
                    z_value=self.z_value,
                )
                for left, right in combinations(self.candidate_ids, 2)
            ),
        )
        if self.contrasts != expected_contrasts:
            raise RuntimeError(
                "calibration contrast values were not derived from the retained "
                "sensitivities and covariance"
            )
        if search is not None:
            candidate_ids, search_hash = calibration_search_identity(search)
            if (
                self.candidate_ids != candidate_ids
                or self.search_identity_hash != search_hash
            ):
                raise RuntimeError(
                    "calibration contrasts do not bind the active boundary search"
                )


def build_calibration_contrast_snapshot(
    *,
    session: object,
    search: object,
    calibration_state: CalibrationUncertaintyState,
    sensitivities: Mapping[str, CalibrationSensitivity],
    indifference: float,
    producer_id: str,
    producer_version: str,
    code_hash: str,
    z_value: float = 2.0,
) -> CalibrationContrastSnapshot:
    """Propagate covariance through every pair on the active boundary chart."""

    candidate_ids, search_hash = calibration_search_identity(search)
    sensitivity_by_id = {str(key): value for key, value in sensitivities.items()}
    if set(sensitivity_by_id) != set(candidate_ids):
        raise ValueError(
            "calibration sensitivities must cover the exact active candidate axis"
        )
    for candidate_id, sensitivity in sensitivity_by_id.items():
        if (
            not isinstance(sensitivity, CalibrationSensitivity)
            or sensitivity.candidate_id != candidate_id
        ):
            raise TypeError(
                "calibration sensitivity mapping has inconsistent typed values"
            )
    boundary = CalibrationSensitivity(
        CALIBRATION_FROZEN_BOUNDARY_REFERENCE_ID,
        np.zeros(calibration_state.parameter_dimension, dtype=np.float64),
    )
    contrasts = (
        *(
            calibration_contrast_uncertainty(
                calibration_state,
                sensitivity_by_id[candidate_id],
                boundary,
                indifference=indifference,
                z_value=z_value,
                contrast_kind="candidate_to_frozen_boundary",
            )
            for candidate_id in candidate_ids
        ),
        *(
            calibration_contrast_uncertainty(
                calibration_state,
                sensitivity_by_id[left],
                sensitivity_by_id[right],
                indifference=indifference,
                z_value=z_value,
            )
            for left, right in combinations(candidate_ids, 2)
        ),
    )
    return CalibrationContrastSnapshot(
        parent_state_hash=str(getattr(session, "state_hash")),
        response_state_hash=_semantic_version_state_hash(session),
        calibration_inferential_hash=calibration_state.inferential_hash,
        search_identity_hash=search_hash,
        candidate_ids=candidate_ids,
        sensitivities=tuple(
            sensitivity_by_id[candidate_id] for candidate_id in candidate_ids
        ),
        contrasts=contrasts,
        indifference=float(indifference),
        z_value=float(z_value),
        producer_id=str(producer_id),
        producer_version=str(producer_version),
        code_hash=str(code_hash),
    )


def calibration_contrast_uncertainty(
    state: CalibrationUncertaintyState,
    left: CalibrationSensitivity,
    right: CalibrationSensitivity,
    *,
    indifference: float,
    z_value: float = 2.0,
    contrast_kind: str = "candidate_pair",
) -> CalibrationContrastUncertainty:
    if (
        left.gradient.size != state.parameter_dimension
        or right.gradient.size != state.parameter_dimension
    ):
        raise ValueError(
            "candidate calibration gradients do not match the calibration state"
        )
    delta = left.gradient - right.gradient
    variance = max(0.0, float(delta @ state.parameter_covariance @ delta))
    standard_error = math.sqrt(variance)
    threshold = float(indifference)
    if threshold < 0.0 or not math.isfinite(threshold):
        raise ValueError("calibration indifference must be finite and nonnegative")
    return CalibrationContrastUncertainty(
        candidate_a=left.candidate_id,
        candidate_b=right.candidate_id,
        variance=variance,
        standard_error=standard_error,
        decision_relevant=bool(float(z_value) * standard_error > threshold),
        indifference=threshold,
        contrast_kind=str(contrast_kind),
    )


class CalibrationMeasurementAdvisor:
    """Build cache-first calibration requests from active contrast uncertainty.

    Sample growth targets the simultaneous active candidate/boundary contrast
    family, not a common parameter-wise covariance tolerance.  The normal
    contraction is an allocation model under regular BCE-fit asymptotics; it
    never narrows the exact-mask decision interval.
    """

    def __init__(
        self,
        *,
        expected_logit_cost: CostEstimate = CostEstimate(forward_passes=1.0),
        expected_refit_cost: CostEstimate = CostEstimate(calibration_fits=1.0),
        minimum_context_increment: int = 32,
        covariance_contraction_exponent: float = 1.0,
        confidence_level: float = 0.95,
        maximum_contexts_without_support_cap: int = 8192,
    ) -> None:
        self.expected_logit_cost = expected_logit_cost
        self.expected_refit_cost = expected_refit_cost
        self.minimum_context_increment = int(minimum_context_increment)
        self.covariance_contraction_exponent = float(covariance_contraction_exponent)
        self.confidence_level = float(confidence_level)
        self.maximum_contexts_without_support_cap = int(
            maximum_contexts_without_support_cap
        )
        if (
            self.minimum_context_increment <= 0
            or not 0.0 < self.covariance_contraction_exponent <= 1.0
            or not 0.5 < self.confidence_level < 1.0
            or self.maximum_contexts_without_support_cap <= 0
        ):
            raise ValueError("calibration advisor controls are invalid")

    def required_context_increment(
        self,
        state: CalibrationUncertaintyState,
        contrasts: Sequence[CalibrationContrastUncertainty],
    ) -> int:
        family = tuple(contrasts)
        if not family:
            return 0
        critical = simultaneous_normal_critical_value(
            self.confidence_level,
            family_size=len(family),
        )
        relevant = [
            item
            for item in family
            if critical * item.standard_error > item.indifference
        ]
        if not relevant:
            return 0
        current = max(1, int(state.fitted_context_count))
        support_cap = (
            int(state.calibration_support_capacity)
            if int(state.calibration_support_capacity) > 0
            else int(self.maximum_contexts_without_support_cap)
        )
        if any(item.indifference <= 0.0 for item in relevant):
            target = support_cap
        else:
            worst_ratio = max(
                critical * item.standard_error / item.indifference
                for item in relevant
            )
            # Under regular asymptotics covariance contracts approximately as
            # 1/n.  The exponent is explicit so nonideal empirical contraction
            # may be made more conservative without changing the estimand.
            target = int(
                math.ceil(
                    current
                    * worst_ratio
                    ** (2.0 / self.covariance_contraction_exponent)
                )
            )
        target = max(
            target,
            current + self.minimum_context_increment,
            int(state.planning_context_count),
        )
        target = min(target, support_cap)
        return max(0, target - current)

    def build_requests(
        self,
        *,
        state_hash: str,
        version: HistoricalResponseProfileName,
        calibration_state: CalibrationUncertaintyState,
        contrasts: Sequence[CalibrationContrastUncertainty],
        random_seed: int,
        code_hash: str,
        contrast_snapshot_hash: str = "",
    ) -> tuple[MeasurementRequest, ...]:
        increment = self.required_context_increment(calibration_state, contrasts)
        if increment <= 0:
            return ()
        fitted_count = int(calibration_state.fitted_context_count)
        target_context_count = fitted_count + increment
        logit_increment = max(
            0,
            target_context_count - int(calibration_state.planning_context_count),
        )
        if (
            calibration_state.cached_logit_keys
            and calibration_state.available_context_count == 0
            and not calibration_state.calibration_support_design_hash
        ):
            # Legacy exploratory states did not retain an exact cached-row
            # count. Preserve their historical refit-only behavior; shadow
            # glocal execution requires the explicit support contract.
            logit_increment = 0
        maximum_se = max(
            item.standard_error for item in contrasts if item.decision_relevant
        )
        provenance = MeasurementProvenance(
            producer_id="irg_calibration_measurement_advisor",
            producer_version="1",
            code_hash=str(code_hash),
        )
        requests = []
        if logit_increment > 0:
            logit_selector = ContextSelector(
                selector_id="irg_calibration_active_contrast_selector_v1",
                requested_count=logit_increment,
                random_seed=int(random_seed),
                parameters={
                    "fit_fold_only": True,
                    "terminal_audit_visible": False,
                    "calibration_target_context_count": target_context_count,
                },
            )
            requests.append(
                MeasurementRequest.create(
                    kind="calibration_logits",
                    state_hash=state_hash,
                    version=version,
                    context_selector=logit_selector,
                    expected_cost=self.expected_logit_cost,
                    expected_decision_value=maximum_se,
                    reuse_keys=(),
                    provenance=provenance,
                    payload={
                        "cache_first": True,
                        "sample_size_policy_id": (
                            "simultaneous_active_calibration_contrast_precision_v1"
                        ),
                        "sample_size_confidence_level": self.confidence_level,
                        "sample_size_active_contrast_count": len(tuple(contrasts)),
                        "calibration_state_id": calibration_state.calibration_state_id,
                        "calibration_contrast_snapshot_hash": (contrast_snapshot_hash),
                        "calibration_current_fit_context_count": fitted_count,
                        "calibration_available_context_count": (
                            calibration_state.planning_context_count
                        ),
                        "calibration_target_context_count": target_context_count,
                        "calibration_support_design_hash": (
                            calibration_state.calibration_support_design_hash
                        ),
                        "calibration_support_capacity": (
                            calibration_state.calibration_support_capacity
                        ),
                    },
                )
            )
        refit_selector = ContextSelector(
            selector_id="irg_calibration_active_contrast_selector_v1",
            requested_count=increment,
            random_seed=int(random_seed),
            parameters={
                "fit_fold_only": True,
                "terminal_audit_visible": False,
                "calibration_target_context_count": target_context_count,
            },
        )
        requests.append(
            MeasurementRequest.create(
                kind="calibration_refit",
                state_hash=state_hash,
                version=version,
                context_selector=refit_selector,
                expected_cost=self.expected_refit_cost,
                expected_decision_value=maximum_se,
                reuse_keys=calibration_state.cached_logit_keys,
                provenance=provenance,
                payload={
                    "sample_size_policy_id": (
                        "simultaneous_active_calibration_contrast_precision_v1"
                    ),
                    "sample_size_confidence_level": self.confidence_level,
                    "sample_size_active_contrast_count": len(tuple(contrasts)),
                    "warm_start_parameter_mean": calibration_state.parameter_mean.tolist(),
                    "fit_fold_only": True,
                    "numerical_converged_before_request": calibration_state.numerical_converged,
                    "calibration_contrast_snapshot_hash": contrast_snapshot_hash,
                    "calibration_current_fit_context_count": fitted_count,
                    "calibration_available_context_count": (
                        calibration_state.planning_context_count
                    ),
                    "calibration_target_context_count": target_context_count,
                    "calibration_support_design_hash": (
                        calibration_state.calibration_support_design_hash
                    ),
                    "calibration_support_capacity": (
                        calibration_state.calibration_support_capacity
                    ),
                },
            )
        )
        return tuple(requests)


def calibration_unified_acquisition_adapter(
    *,
    outcome_consumer: Callable[[object, MeasurementRequest, object], None],
    advisor: CalibrationMeasurementAdvisor | None = None,
    code_hash: str,
    state_reader: Callable[[object], CalibrationUncertaintyState | None] | None = None,
    contrasts_reader: (
        Callable[
            [object, object | None],
            CalibrationContrastSnapshot | None,
        ]
        | None
    ) = None,
) -> UnifiedAcquisitionAdapter:
    """Bind typed calibration uncertainty into the common IRG economy.

    The default readers use explicit session-cache contracts.  A production
    backend may instead provide readers over its retained calibration state.
    The adapter never narrows a safety bound; its covariance forecast is
    allocation guidance only.
    """

    if not code_hash:
        raise ValueError("calibration acquisition adapter requires code identity")
    resolved_advisor = advisor or CalibrationMeasurementAdvisor()

    def read_state(session: object) -> CalibrationUncertaintyState | None:
        if state_reader is not None:
            return state_reader(session)
        cache = getattr(session, "cache", {})
        value = cache.get(CALIBRATION_UNCERTAINTY_CACHE_KEY)
        if value is None:
            return None
        if not isinstance(value, CalibrationUncertaintyState):
            raise TypeError("calibration uncertainty cache has the wrong type")
        return value

    def read_snapshot(
        session: object,
        search: object | None,
        state: CalibrationUncertaintyState,
        *,
        required: bool,
        expected_hash: str = "",
    ) -> CalibrationContrastSnapshot | None:
        cache = getattr(session, "cache", {})
        immutable_key = (
            f"{CALIBRATION_CONTRASTS_CACHE_KEY}:{expected_hash}"
            if expected_hash
            else ""
        )
        if immutable_key and immutable_key in cache:
            value = cache[immutable_key]
        elif search is None and CALIBRATION_CONTRASTS_CACHE_KEY in cache:
            value = cache[CALIBRATION_CONTRASTS_CACHE_KEY]
        elif contrasts_reader is not None:
            value = contrasts_reader(session, search)
        else:
            value = cache.get(CALIBRATION_CONTRASTS_CACHE_KEY)
        if value is None:
            if required:
                raise RuntimeError(
                    "active calibration covariance has no decision-bound "
                    "contrast snapshot"
                )
            return None
        if not isinstance(value, CalibrationContrastSnapshot):
            raise TypeError("calibration contrast cache must contain a typed snapshot")
        value.assert_current(
            session=session,
            calibration_state=state,
            search=search,
        )
        if expected_hash and value.content_hash != expected_hash:
            raise RuntimeError(
                "calibration contrast snapshot differs from its request binding"
            )
        cache[CALIBRATION_CONTRASTS_CACHE_KEY] = value
        cache[f"{CALIBRATION_CONTRASTS_CACHE_KEY}:{value.content_hash}"] = value
        return value

    def synchronize_decision_state(
        session: object,
        state: CalibrationUncertaintyState,
        contrasts: Sequence[CalibrationContrastUncertainty],
    ) -> None:
        decision = getattr(session, "decision")
        relevant = tuple(value for value in contrasts if value.decision_relevant)
        decision.calibration_uncertainty = max(
            (value.standard_error for value in relevant),
            default=0.0,
        )
        decision.calibration_sample_count = int(state.fitted_context_count)

    def request_factory(
        session: object,
        phase: str,
        search: object | None,
    ) -> tuple[MeasurementRequest, ...]:
        # The locked parent calibration must already exist before candidate
        # birth.  Refinement competes only once active candidate contrasts
        # exist; if selected, the runner retires those candidates and rebuilds
        # them under the replacement calibration state.
        if phase != "boundary":
            return ()
        state = read_state(session)
        if state is None:
            return ()
        snapshot = read_snapshot(
            session,
            search,
            state,
            required=True,
        )
        assert snapshot is not None
        contrasts = snapshot.contrasts
        synchronize_decision_state(session, state, contrasts)
        return resolved_advisor.build_requests(
            state_hash=str(getattr(session, "state_hash")),
            version=getattr(session, "active_version"),
            calibration_state=state,
            contrasts=contrasts,
            random_seed=int(getattr(session, "random_seed"))
            + int(getattr(session, "iteration")),
            code_hash=code_hash,
            contrast_snapshot_hash=snapshot.content_hash,
        )

    def forecast_factory(
        session: object,
        request: MeasurementRequest,
        decision_snapshot: DecisionUncertaintySnapshot,
        phase: str,
    ) -> AcquisitionForecast:
        del phase
        state = read_state(session)
        if state is None:
            raise RuntimeError("calibration request lost its uncertainty state")
        contrast_snapshot = read_snapshot(
            session,
            None,
            state,
            required=True,
            expected_hash=str(
                request.payload.get("calibration_contrast_snapshot_hash", "")
            ),
        )
        assert contrast_snapshot is not None
        if (
            request.payload.get("calibration_contrast_snapshot_hash")
            != contrast_snapshot.content_hash
        ):
            raise RuntimeError(
                "calibration request is not bound to its active contrast snapshot"
            )
        contrasts = contrast_snapshot.contrasts
        relevant = tuple(value for value in contrasts if value.decision_relevant)
        variance = max(
            (float(value.variance) for value in relevant),
            default=float(decision_snapshot.calibration_variance),
        )
        current = max(1, int(state.fitted_context_count))
        target_count = int(
            request.payload.get(
                "calibration_target_context_count",
                current + int(request.context_selector.requested_count),
            )
        )
        added = max(1, target_count - current)
        posterior = variance * current / float(current + added)
        return AcquisitionForecast(
            request_id=request.request_id,
            state_hash=request.state_hash,
            forecast_model_id="calibration_active_contrast_covariance_1_over_n_v1",
            uncertainty_source="calibration_contrast",
            current_variance=variance,
            fantasy_posterior_variance=posterior,
            current_sample_count=int(state.fitted_context_count),
            fantasy_sample_count=int(state.fitted_context_count) + added,
            rationale=(
                "propagate calibration covariance only through active "
                "candidate contrasts"
            ),
            metadata={
                "allocation_only": True,
                "claim_bound_narrowing": False,
                "calibration_state_hash": state.state_hash,
                "contrast_count": len(relevant),
                "calibration_target_context_count": target_count,
                "cached_prefix_available": (
                    int(state.planning_context_count) >= target_count
                ),
                "calibration_support_design_hash": (
                    state.calibration_support_design_hash
                ),
                "calibration_support_capacity": state.calibration_support_capacity,
            },
        )

    def consume(
        session: object,
        request: MeasurementRequest,
        outcome: object,
    ) -> None:
        if (
            request.kind == "calibration_refit"
            and str(getattr(session, "commit_authority_mode", "exploratory"))
            == "glocal_v3_shadow"
        ):
            raise RuntimeError(
                "shadow calibration refits must be executed and consumed by the "
                "campaign-certified calibration-transition authority"
            )
        previous = read_state(session)
        if previous is None:
            raise RuntimeError(
                "calibration outcome lost the calibration state that requested it"
            )
        prior_version_state_hash = _semantic_version_state_hash(session)
        prior_snapshot = read_snapshot(
            session,
            None,
            previous,
            required=True,
            expected_hash=str(
                request.payload.get("calibration_contrast_snapshot_hash", "")
            ),
        )
        assert prior_snapshot is not None
        if (
            request.payload.get("calibration_contrast_snapshot_hash")
            != prior_snapshot.content_hash
        ):
            raise RuntimeError(
                "calibration outcome differs from its frozen contrast snapshot"
            )
        outcome_consumer(session, request, outcome)
        updated = read_state(session)
        if updated is None:
            raise RuntimeError(
                "calibration outcome consumer did not retain typed uncertainty state"
            )
        if updated.state_hash == previous.state_hash:
            raise RuntimeError(
                "calibration acquisition outcome left the retained calibration "
                "state unchanged"
            )
        if updated.fitted_context_count < previous.fitted_context_count:
            raise RuntimeError(
                "calibration acquisition regressed the fitted context count"
            )
        if request.kind == "calibration_logits" and (
            set(updated.cached_logit_keys).issubset(previous.cached_logit_keys)
            and updated.planning_context_count <= previous.planning_context_count
        ):
            raise RuntimeError(
                "calibration-logit acquisition retained no new contexts or cache keys"
            )
        if (
            request.kind == "calibration_logits"
            and updated.inferential_hash != previous.inferential_hash
        ):
            raise RuntimeError(
                "calibration-logit acquisition may extend reusable evidence but "
                "cannot silently replace the fitted calibration distribution"
            )
        if request.kind == "calibration_refit" and (
            updated.calibration_state_id
            != str(getattr(outcome, "calibration_state_id", ""))
            or updated.calibration_state_id == previous.calibration_state_id
            or not updated.numerical_converged
        ):
            raise RuntimeError(
                "calibration refit must retain a new, converged fit matching the "
                "physical outcome"
            )
        if (
            request.kind == "calibration_refit"
            and str(getattr(session, "commit_authority_mode", "exploratory"))
            == "glocal_v3_shadow"
            and (
                not updated.calibration_fit_state_hash
                or updated.calibration_fit_state_hash
                == previous.calibration_fit_state_hash
            )
        ):
            raise RuntimeError(
                "shadow calibration refit must retain a new exact "
                "CalibrationFitState lineage"
            )
        if request.kind == "calibration_refit":
            active_calibration = _active_response_calibration_id(session)
            if active_calibration != updated.calibration_state_id:
                raise RuntimeError(
                    "calibration refit updated only its uncertainty/cache state; "
                    "the active response chart does not retain the new calibration"
                )
            if _semantic_version_state_hash(session) == prior_version_state_hash:
                raise RuntimeError(
                    "calibration refit changed calibration identity without a "
                    "semantic version-state transition"
                )
        if request.kind == "calibration_refit":
            getattr(session, "cache").pop(CALIBRATION_CONTRASTS_CACHE_KEY, None)
            synchronize_decision_state(session, updated, ())
        else:
            prior_snapshot.assert_current(
                session=session,
                calibration_state=updated,
                search=None,
            )
            synchronize_decision_state(
                session,
                updated,
                prior_snapshot.contrasts,
            )

    def assert_ready(session: object) -> None:
        state = read_state(session)
        if state is None:
            raise RuntimeError(
                "calibration acquisition adapter has no retained typed "
                "calibration state"
            )
        if not state.numerical_converged or state.fitted_context_count <= 0:
            raise RuntimeError(
                "calibration acquisition adapter requires a converged retained fit"
            )
        if (
            str(getattr(session, "commit_authority_mode", "exploratory"))
            == "glocal_v3_shadow"
            and not state.calibration_fit_state_hash
        ):
            raise RuntimeError(
                "shadow calibration uncertainty lacks exact CalibrationFitState "
                "lineage"
            )
        active_calibration = _active_response_calibration_id(session)
        if active_calibration != state.calibration_state_id:
            raise RuntimeError(
                "calibration acquisition state differs from the active response chart"
            )

    return UnifiedAcquisitionAdapter(
        adapter_id="irg_calibration_active_contrast_adapter_v1",
        supported_kinds=frozenset({"calibration_logits", "calibration_refit"}),
        request_factory=request_factory,
        outcome_consumer=consume,
        forecast_factory=forecast_factory,
        readiness_checker=assert_ready,
    )


__all__ = [
    "CALIBRATION_FROZEN_BOUNDARY_REFERENCE_ID",
    "CalibrationContrastSnapshot",
    "CalibrationContrastUncertainty",
    "CALIBRATION_CONTRASTS_CACHE_KEY",
    "CALIBRATION_UNCERTAINTY_CACHE_KEY",
    "CalibrationMeasurementAdvisor",
    "CalibrationSensitivity",
    "CalibrationUncertaintyState",
    "build_calibration_contrast_snapshot",
    "calibration_search_identity",
    "calibration_contrast_uncertainty",
    "calibration_unified_acquisition_adapter",
]
