"""Frontier-directed acquisition, practical racing, and context allocation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Literal, Mapping, Sequence

import numpy as np

from mechanistic_systems.measurement.neyman_allocation import (
    NeymanAllocationRequest,
    NeymanStratumSpec,
    UNIFIED_NEYMAN_POLICY_ID,
    UnifiedNeymanAllocator,
)

from .contracts import (
    MeasurementKind,
    MeasurementRequest,
    ResponseConstraintSpec,
    content_hash,
)
from .response_geometry import (
    ResponseActionMoments,
    ResponseConstraintEvaluator,
    ResponseRiskEvaluation,
)


@dataclass(frozen=True)
class ActiveResponseDirection:
    """One frozen response-space direction currently limiting pruning.

    The direction is induced by the declared response-safe region and the
    current action moments.  It is not a free attribution score.  Projecting
    context rows onto this direction produces the scalar observation that is
    locally most relevant for resolving the active pruning decision.
    """

    constraint_id: str
    vector: np.ndarray = field(repr=False, compare=False)
    channel_ids: tuple[str, ...]
    tolerance: float
    source: str
    response_constraint_hash: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        vector = np.asarray(self.vector, dtype=np.float64).reshape(-1)
        channels = tuple(map(str, self.channel_ids))
        if (
            not self.constraint_id
            or not self.source
            or not self.response_constraint_hash
        ):
            raise ValueError("active response direction requires scientific identity")
        if (
            not channels
            or len(channels) != vector.size
            or len(set(channels)) != len(channels)
        ):
            raise ValueError("active response direction has an invalid response axis")
        if not np.isfinite(vector).all() or float(np.linalg.norm(vector)) <= 0.0:
            raise ValueError("active response direction must be finite and nonzero")
        tolerance = float(self.tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("active response tolerance must be finite and positive")
        vector = np.array(vector / np.linalg.norm(vector), copy=True, order="C")
        vector.setflags(write=False)
        object.__setattr__(self, "vector", vector)
        object.__setattr__(self, "channel_ids", channels)
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def content_hash(self) -> str:
        return content_hash(
            {
                "constraint_id": self.constraint_id,
                "vector": self.vector,
                "channel_ids": self.channel_ids,
                "tolerance": self.tolerance,
                "source": self.source,
                "response_constraint_hash": self.response_constraint_hash,
                "metadata": dict(self.metadata),
            },
            prefix="active-response-direction-",
        )

    def project(self, rows: np.ndarray) -> np.ndarray:
        values = np.asarray(rows, dtype=np.float64)
        if (
            values.ndim < 1
            or values.shape[-1] != self.vector.size
            or not np.isfinite(values).all()
        ):
            raise ValueError("response rows do not align with the active direction")
        return np.einsum("...d,d->...", values, self.vector, optimize=True)


def _matrix_square_root_psd(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if (
        value.ndim != 2
        or value.shape[0] != value.shape[1]
        or not np.isfinite(value).all()
    ):
        raise ValueError("response metric must be a finite square matrix")
    value = 0.5 * (value + value.T)
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 1.0)
    if eigenvalues.size and float(np.min(eigenvalues)) < -1e-8 * scale:
        raise ValueError("response metric is materially indefinite")
    return (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))) @ eigenvectors.T


def active_response_direction(
    constraint: ResponseConstraintSpec,
    evaluation: ResponseRiskEvaluation,
    moments: ResponseActionMoments,
    *,
    channel_ids: Sequence[str],
) -> ActiveResponseDirection:
    """Recover the response direction inducing the current local decision.

    Directional constraints are exact.  When the energy boundary is active,
    the function returns the principal response direction of the measured
    second moment under the frozen energy metric.  This gives acquisition and
    sampling a deterministic dual direction without pretending that the
    scalar energy itself identifies a unique physical route ordering.
    """

    channels = tuple(map(str, channel_ids))
    if (
        moments.response_dimension != constraint.output_dimension
        or len(channels) != constraint.output_dimension
    ):
        raise ValueError("active response direction requires aligned response axes")
    active = str(evaluation.active_constraint_id)
    for direction in constraint.directions:
        if direction.direction_id == active:
            raw = np.asarray(direction.vector, dtype=np.float64)
            norm = float(np.linalg.norm(raw))
            return ActiveResponseDirection(
                constraint_id=active,
                vector=raw,
                channel_ids=channels,
                tolerance=float(direction.lower_tolerance) / norm,
                source="active_oriented_response_boundary",
                response_constraint_hash=constraint.content_hash,
                metadata={"risk": float(evaluation.risk)},
            )
    if active == "response_energy":
        if not constraint.energy_metric:
            raise ValueError("energy was active without a declared energy metric")
        metric = np.asarray(constraint.energy_metric, dtype=np.float64)
        root = _matrix_square_root_psd(metric)
        transformed = root @ moments.second_moment @ root
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (transformed + transformed.T))
        principal = eigenvectors[:, int(np.argmax(eigenvalues))]
        raw = root @ principal
        if float(np.linalg.norm(raw)) <= 1e-15:
            # A zero measured response has no data-identified energy direction.
            # Use the largest metric axis deterministically and disclose it.
            metric_values, metric_vectors = np.linalg.eigh(0.5 * (metric + metric.T))
            raw = metric_vectors[:, int(np.argmax(metric_values))]
            fallback = True
        else:
            fallback = False
        return ActiveResponseDirection(
            constraint_id=active,
            vector=raw,
            channel_ids=channels,
            tolerance=float(constraint.energy_tolerance),
            source="principal_task_energy_response_direction",
            response_constraint_hash=constraint.content_hash,
            metadata={
                "risk": float(evaluation.risk),
                "principal_task_energy": float(max(0.0, np.max(eigenvalues))),
                "zero_response_fallback": fallback,
            },
        )
    # A fully safe zero-risk action can have no active facet.  Choose the
    # direction with the smallest normalized slack, not an arbitrary score.
    if constraint.directions:
        candidates = []
        for direction in constraint.directions:
            raw = np.asarray(direction.vector, dtype=np.float64)
            norm = float(np.linalg.norm(raw))
            unit = raw / norm
            tolerance = float(direction.lower_tolerance) / norm
            slack = (tolerance + float(unit @ moments.mean)) / tolerance
            candidates.append((slack, direction.direction_id, raw, tolerance))
        slack, direction_id, raw, tolerance = min(
            candidates, key=lambda item: (item[0], item[1])
        )
        return ActiveResponseDirection(
            constraint_id=direction_id,
            vector=raw,
            channel_ids=channels,
            tolerance=tolerance,
            source="minimum_normalized_response_slack",
            response_constraint_hash=constraint.content_hash,
            metadata={"normalized_slack": float(slack), "risk": float(evaluation.risk)},
        )
    raise ValueError(
        "response constraint contains no direction from which acquisition can be induced"
    )


def response_direction_pairwise_contrast(
    candidate_a: str,
    candidate_b: str,
    response_rows_a: np.ndarray,
    response_rows_b: np.ndarray,
    *,
    active_direction: ActiveResponseDirection,
    indifference: float,
    relevance: float = 1.0,
) -> PairwiseContrast:
    """Paired candidate contrast in the currently limiting response direction."""

    left = np.asarray(response_rows_a, dtype=np.float64)
    right = np.asarray(response_rows_b, dtype=np.float64)
    if (
        left.shape != right.shape
        or left.ndim != 2
        or left.shape[1] != active_direction.vector.size
    ):
        raise ValueError(
            "response-direction contrast requires aligned [context,response] rows"
        )
    return paired_contrast(
        candidate_a,
        candidate_b,
        active_direction.project(left),
        active_direction.project(right),
        indifference=float(indifference),
        relevance=float(relevance),
    )


def response_discrimination_context_scores(
    candidate_response_rows: np.ndarray,
    *,
    active_direction: ActiveResponseDirection,
    candidate_standard_errors: np.ndarray | None = None,
    model_residual_rows: np.ndarray | None = None,
) -> np.ndarray:
    """Score contexts by their value for resolving the active candidate set.

    ``candidate_response_rows`` has axes ``[context,candidate,response]``.
    The deterministic score combines the range of candidate predictions in the
    active response direction, optional projected estimator uncertainty, and
    optional local-model residual magnitude.  It is an acquisition diagnostic,
    not a certificate or a new response objective.
    """

    rows = np.asarray(candidate_response_rows, dtype=np.float64)
    if (
        rows.ndim != 3
        or rows.shape[2] != active_direction.vector.size
        or rows.shape[1] < 2
    ):
        raise ValueError(
            "context discrimination requires [context,candidate,response] rows"
        )
    projected = active_direction.project(rows)
    disagreement = np.ptp(projected, axis=1) ** 2
    score = disagreement
    if candidate_standard_errors is not None:
        errors = np.asarray(candidate_standard_errors, dtype=np.float64)
        if (
            errors.shape != rows.shape
            or np.any(errors < 0.0)
            or not np.isfinite(errors).all()
        ):
            raise ValueError(
                "candidate response standard errors must align with response rows"
            )
        projected_variance = np.einsum(
            "nkd,d->nk",
            errors * errors,
            active_direction.vector * active_direction.vector,
            optimize=True,
        )
        score = score + np.max(projected_variance, axis=1)
    if model_residual_rows is not None:
        residual = np.asarray(model_residual_rows, dtype=np.float64)
        if residual.ndim == 2 and residual.shape == (rows.shape[0], rows.shape[2]):
            score = score + active_direction.project(residual) ** 2
        elif residual.shape == rows.shape:
            score = score + np.max(active_direction.project(residual) ** 2, axis=1)
        else:
            raise ValueError("model residual rows have incompatible axes")
    result = np.asarray(score, dtype=np.float64)
    if not np.isfinite(result).all() or np.any(result < -1e-12):
        raise ValueError("context discrimination scores are invalid")
    return np.maximum(result, 0.0)


def select_response_informative_contexts(
    context_ids: Sequence[str],
    scores: Sequence[float] | np.ndarray,
    *,
    budget: int,
    strata: Sequence[str] | None = None,
    minimum_per_stratum: int = 0,
) -> tuple[str, ...]:
    """Select a deterministic high-value context batch with stratum coverage."""

    ids = tuple(map(str, context_ids))
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    count = int(budget)
    minimum = int(minimum_per_stratum)
    if (
        not ids
        or len(ids) != values.size
        or len(set(ids)) != len(ids)
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            "context selection requires unique aligned finite observations"
        )
    if count <= 0 or count > len(ids) or minimum < 0:
        raise ValueError("context selection budget is invalid")
    labels = tuple("all" for _ in ids) if strata is None else tuple(map(str, strata))
    if len(labels) != len(ids):
        raise ValueError("context strata must align with context IDs")
    unique = tuple(sorted(set(labels)))
    if minimum * len(unique) > count:
        raise ValueError("minimum context stratum coverage exceeds budget")
    order = sorted(range(len(ids)), key=lambda index: (-values[index], ids[index]))
    selected: list[int] = []
    selected_set: set[int] = set()
    for label in unique:
        label_order = [index for index in order if labels[index] == label]
        if len(label_order) < minimum:
            raise ValueError(
                f"stratum {label!r} cannot satisfy the minimum context coverage"
            )
        for index in label_order[:minimum]:
            selected.append(index)
            selected_set.add(index)
    for index in order:
        if len(selected) >= count:
            break
        if index not in selected_set:
            selected.append(index)
            selected_set.add(index)
    return tuple(ids[index] for index in selected)


@dataclass(frozen=True)
class CandidateBelief:
    """Gaussian working belief for a decision value where larger is better."""

    candidate_id: str
    mean_value: float
    standard_error: float
    frontier_weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate belief requires candidate_id")
        for name in ("mean_value", "standard_error", "frontier_weight"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or (name != "mean_value" and value < 0.0):
                raise ValueError(f"{name} has an invalid value")


@dataclass(frozen=True)
class PairwiseContrast:
    candidate_a: str
    candidate_b: str
    mean_difference: float
    standard_error: float
    indifference: float
    stratum_variances: Mapping[str, float] = field(default_factory=dict)
    stratum_costs: Mapping[str, float] = field(default_factory=dict)
    relevance: float = 1.0

    def __post_init__(self) -> None:
        if (
            not self.candidate_a
            or not self.candidate_b
            or self.candidate_a == self.candidate_b
        ):
            raise ValueError("pairwise contrast requires two distinct candidate IDs")
        for name in ("mean_difference", "standard_error", "indifference", "relevance"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or (name != "mean_difference" and value < 0.0):
                raise ValueError(f"{name} has an invalid value")
        variances = {
            str(key): float(value) for key, value in self.stratum_variances.items()
        }
        costs = {str(key): float(value) for key, value in self.stratum_costs.items()}
        if set(variances) != set(costs):
            raise ValueError("stratum variances and costs must cover the same strata")
        if any(not math.isfinite(value) or value < 0.0 for value in variances.values()):
            raise ValueError("stratum variances must be finite and nonnegative")
        if any(not math.isfinite(value) or value <= 0.0 for value in costs.values()):
            raise ValueError("stratum costs must be finite and positive")
        object.__setattr__(self, "stratum_variances", variances)
        object.__setattr__(self, "stratum_costs", costs)

    @property
    def resolved(self) -> bool:
        return abs(self.mean_difference) > self.indifference + 2.0 * self.standard_error


@dataclass(frozen=True)
class AcquisitionFactors:
    decision_change_probability: float
    expected_decision_loss_reduction: float
    frontier_relevance: float = 1.0
    model_failure_reduction: float = 0.0
    coverage_gain: float = 0.0
    reuse_bonus: float = 0.0

    def __post_init__(self) -> None:
        probability = float(self.decision_change_probability)
        if not 0.0 <= probability <= 1.0 or not math.isfinite(probability):
            raise ValueError("decision_change_probability must lie in [0,1]")
        for name in (
            "expected_decision_loss_reduction",
            "frontier_relevance",
            "model_failure_reduction",
            "coverage_gain",
            "reuse_bonus",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")

    @property
    def value(self) -> float:
        core = (
            self.decision_change_probability
            * self.expected_decision_loss_reduction
            * self.frontier_relevance
        )
        return float(
            core + self.model_failure_reduction + self.coverage_gain + self.reuse_bonus
        )


@dataclass(frozen=True)
class ScoredRequest:
    request: MeasurementRequest
    factors: AcquisitionFactors
    score: float
    opportunity_hash: str = ""
    forecast_hash: str = ""
    predicted_cost: float = 0.0
    requirement_class: "MeasurementRequirementClass" = "optional"
    obligation_id: str | None = None

    @property
    def stage_gate_required(self) -> bool:
        return self.requirement_class in {
            "candidate_birth_prerequisite",
            "chart_validity_prerequisite",
        }


@dataclass(frozen=True)
class AcquisitionForecast:
    """Predictable, adapter-owned covariance contraction for one request."""

    request_id: str
    state_hash: str
    forecast_model_id: str
    uncertainty_source: str
    current_variance: float
    fantasy_posterior_variance: float
    current_sample_count: int
    fantasy_sample_count: int
    rationale: str
    metadata: Mapping[str, object] = field(default_factory=dict)
    schema: str = "irg_acquisition_forecast_v1"

    def __post_init__(self) -> None:
        if any(
            not value
            for value in (
                self.request_id,
                self.state_hash,
                self.forecast_model_id,
                self.uncertainty_source,
                self.rationale,
                self.schema,
            )
        ):
            raise ValueError("acquisition forecast requires complete identity")
        current = float(self.current_variance)
        posterior = float(self.fantasy_posterior_variance)
        if (
            not math.isfinite(current)
            or current < 0.0
            or not math.isfinite(posterior)
            or posterior < 0.0
            or posterior > current + 1e-12
        ):
            raise ValueError("acquisition forecast variances are invalid")
        if int(self.current_sample_count) < 0 or int(self.fantasy_sample_count) < int(
            self.current_sample_count
        ):
            raise ValueError("acquisition forecast sample counts are invalid")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def content_hash(self) -> str:
        return content_hash(self, prefix="irg-acquisition-forecast-")


@dataclass(frozen=True)
class AcquisitionSelectionReceipt:
    """Content-addressed record of one common-planner choice."""

    state_hash: str
    phase: str
    uncertainty_snapshot_hash: str
    proposed_request_ids: tuple[str, ...]
    request_ids: tuple[str, ...]
    opportunity_hashes: tuple[str, ...]
    forecast_hashes: tuple[str, ...]
    scores: tuple[float, ...]
    predicted_costs: tuple[float, ...]
    selected_request_ids: tuple[str, ...]
    requirement_classes: Mapping[str, str] = field(default_factory=dict)
    obligation_ids: Mapping[str, str] = field(default_factory=dict)
    outside_option_score: float = 0.0
    ineligible_request_reasons: Mapping[str, str] = field(default_factory=dict)
    schema: str = "irg_acquisition_selection_receipt_v4"

    def __post_init__(self) -> None:
        if (
            not self.state_hash
            or not self.phase
            or not self.uncertainty_snapshot_hash
            or not self.proposed_request_ids
            or not self.schema
        ):
            raise ValueError("acquisition selection receipt requires identity")
        size = len(self.request_ids)
        ineligible = {
            str(key): str(value)
            for key, value in self.ineligible_request_reasons.items()
        }
        requirements = {
            str(key): str(value) for key, value in self.requirement_classes.items()
        }
        obligations = {
            str(key): str(value) for key, value in self.obligation_ids.items()
        }
        if (
            len(set(self.proposed_request_ids)) != len(self.proposed_request_ids)
            or len(set(self.request_ids)) != size
            or not set(self.request_ids).issubset(self.proposed_request_ids)
            or len(self.opportunity_hashes) != size
            or len(self.forecast_hashes) != size
            or len(self.scores) != size
            or len(self.predicted_costs) != size
            or any(value not in self.request_ids for value in self.selected_request_ids)
            or any(not math.isfinite(float(value)) for value in self.scores)
            or any(
                not math.isfinite(float(value)) or float(value) <= 0.0
                for value in self.predicted_costs
            )
            or set(ineligible)
            != set(self.proposed_request_ids).difference(self.request_ids)
            or any(not value for value in ineligible.values())
            or set(requirements) != set(self.proposed_request_ids)
            or any(
                value not in MEASUREMENT_REQUIREMENT_CLASSES
                for value in requirements.values()
            )
            or not set(obligations).issubset(self.proposed_request_ids)
            or any(not value for value in obligations.values())
            or any(
                requirements[request_id]
                in {
                    "candidate_birth_prerequisite",
                    "chart_validity_prerequisite",
                    "commit_prerequisite",
                }
                and request_id not in obligations
                for request_id in self.proposed_request_ids
            )
            or not math.isfinite(float(self.outside_option_score))
            or float(self.outside_option_score) < 0.0
        ):
            raise ValueError("acquisition selection receipt axes are invalid")
        object.__setattr__(self, "ineligible_request_reasons", ineligible)
        object.__setattr__(self, "requirement_classes", requirements)
        object.__setattr__(self, "obligation_ids", obligations)

    @property
    def content_hash(self) -> str:
        return content_hash(self, prefix="irg-acquisition-selection-")


@dataclass(frozen=True)
class UnifiedAcquisitionAdapter:
    """One typed producer/consumer channel in the common measurement economy.

    The adapter proposes predictable requests and owns their outcome update.
    It does not assign decision value; the common planner derives that value
    from the frozen uncertainty snapshot and learned realized cost.
    """

    adapter_id: str
    supported_kinds: frozenset[MeasurementKind]
    request_factory: Callable[
        [object, str, object | None], Sequence[MeasurementRequest]
    ]
    outcome_consumer: Callable[[object, MeasurementRequest, object], None]
    forecast_factory: (
        Callable[
            [object, MeasurementRequest, "DecisionUncertaintySnapshot", str],
            AcquisitionForecast | None,
        ]
        | None
    ) = None
    readiness_checker: Callable[[object], None] | None = None
    blocking_reason_checker: Callable[[object, str], str | None] | None = None

    def __post_init__(self) -> None:
        if not self.adapter_id or not self.supported_kinds:
            raise ValueError("unified acquisition adapter requires identity and kinds")
        if not callable(self.request_factory) or not callable(self.outcome_consumer):
            raise ValueError("unified acquisition adapter callbacks must be callable")
        if self.forecast_factory is not None and not callable(self.forecast_factory):
            raise ValueError("acquisition forecast_factory must be callable")
        if self.readiness_checker is not None and not callable(self.readiness_checker):
            raise ValueError("acquisition readiness_checker must be callable")
        if self.blocking_reason_checker is not None and not callable(
            self.blocking_reason_checker
        ):
            raise ValueError("acquisition blocking_reason_checker must be callable")

    def assert_ready(self, session: object) -> None:
        if self.readiness_checker is None:
            raise RuntimeError(
                f"adapter {self.adapter_id!r} has no scientific readiness check"
            )
        self.readiness_checker(session)

    def blocking_reason(self, session: object, phase: str) -> str | None:
        """Return a durable stage-gate refusal that no optional cap may erase."""

        if self.blocking_reason_checker is None:
            return None
        reason = self.blocking_reason_checker(session, str(phase))
        if reason is None:
            return None
        reason = str(reason)
        if not reason:
            raise ValueError(
                f"adapter {self.adapter_id!r} returned an empty blocking reason"
            )
        return reason

    def requests(
        self,
        session: object,
        phase: str,
        search: object | None,
    ) -> tuple[MeasurementRequest, ...]:
        requests = tuple(self.request_factory(session, str(phase), search))
        for request in requests:
            if request.kind not in self.supported_kinds:
                raise ValueError(
                    f"adapter {self.adapter_id!r} emitted undeclared kind "
                    f"{request.kind!r}"
                )
        return requests

    def consume(
        self,
        session: object,
        request: MeasurementRequest,
        outcome: object,
    ) -> None:
        if request.kind not in self.supported_kinds:
            raise ValueError(
                f"adapter {self.adapter_id!r} cannot consume {request.kind!r}"
            )
        self.outcome_consumer(session, request, outcome)

    def forecast(
        self,
        session: object,
        request: MeasurementRequest,
        snapshot: "DecisionUncertaintySnapshot",
        phase: str,
    ) -> AcquisitionForecast | None:
        if self.forecast_factory is None:
            return None
        forecast = self.forecast_factory(
            session,
            request,
            snapshot,
            str(phase),
        )
        if forecast is None:
            return None
        if (
            forecast.request_id != request.request_id
            or forecast.state_hash != request.state_hash
        ):
            raise ValueError(
                f"adapter {self.adapter_id!r} forecast is bound to another request/state"
            )
        return forecast


@dataclass(frozen=True)
class DecisionUncertaintySnapshot:
    """Predictable working uncertainty used only to price measurements.

    The snapshot is guidance: it may use Gaussian/posterior approximations to
    decide what to measure next, but it is never substituted for the bounded
    paired evidence process that authorizes a hard-mask commit.
    """

    state_hash: str
    frontier_boundary_gap: float
    frontier_regret_scale: float
    safety_interval_width: float
    calibration_variance: float
    response_variance: float
    model_failure_probability: float
    screen_uncertainty: float
    uncovered_strata: int = 0
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.state_hash:
            raise ValueError("decision uncertainty snapshot requires state identity")
        for name in (
            "frontier_boundary_gap",
            "frontier_regret_scale",
            "safety_interval_width",
            "calibration_variance",
            "response_variance",
            "model_failure_probability",
            "screen_uncertainty",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.model_failure_probability > 1.0 or self.screen_uncertainty > 1.0:
            raise ValueError("uncertainty probabilities must lie in [0,1]")
        if int(self.uncovered_strata) < 0:
            raise ValueError("uncovered_strata must be nonnegative")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def content_hash(self) -> str:
        return content_hash(self, prefix="irg-decision-uncertainty-")


MeasurementRequirementClass = Literal[
    "optional",
    "candidate_birth_prerequisite",
    "chart_validity_prerequisite",
    "commit_prerequisite",
]
MEASUREMENT_REQUIREMENT_CLASSES: frozenset[str] = frozenset(
    {
        "optional",
        "candidate_birth_prerequisite",
        "chart_validity_prerequisite",
        "commit_prerequisite",
    }
)


@dataclass(frozen=True)
class MeasurementOpportunity:
    """One internally valued measurement and its predictable fantasy update."""

    request: MeasurementRequest
    uncertainty_source: str
    current_variance: float
    fantasy_posterior_variance: float
    resolution_probability: float
    expected_regret_reduction: float
    safety_resolution_value: float = 0.0
    coverage_gain: float = 0.0
    model_failure_reduction: float = 0.0
    reuse_bonus: float = 0.0
    requirement_class: MeasurementRequirementClass = "optional"
    obligation_id: str | None = None
    rationale: str = ""
    forecast_hash: str = ""

    def __post_init__(self) -> None:
        if not self.uncertainty_source or not self.rationale:
            raise ValueError("measurement opportunity requires source and rationale")
        for name in (
            "current_variance",
            "fantasy_posterior_variance",
            "resolution_probability",
            "expected_regret_reduction",
            "safety_resolution_value",
            "coverage_gain",
            "model_failure_reduction",
            "reuse_bonus",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.fantasy_posterior_variance > self.current_variance + 1e-12:
            raise ValueError("measurement fantasy may not increase its target variance")
        if self.resolution_probability > 1.0:
            raise ValueError("resolution_probability must lie in [0,1]")
        if self.requirement_class not in MEASUREMENT_REQUIREMENT_CLASSES:
            raise ValueError("unknown measurement requirement class")
        if self.requirement_class == "optional":
            if self.obligation_id is not None:
                raise ValueError(
                    "optional measurement opportunities may not claim an obligation"
                )
        elif not self.obligation_id:
            raise ValueError(
                "prerequisite and commit opportunities require an obligation ID"
            )

    @property
    def hard_required(self) -> bool:
        """Compatibility view for true pre-measurement stage gates only."""

        return self.requirement_class in {
            "candidate_birth_prerequisite",
            "chart_validity_prerequisite",
        }

    @property
    def content_hash(self) -> str:
        return content_hash(self, prefix="irg-measurement-opportunity-")

    @property
    def expected_value(self) -> float:
        return float(
            self.resolution_probability * self.expected_regret_reduction
            + self.safety_resolution_value
            + self.coverage_gain
            + self.model_failure_reduction
            + self.reuse_bonus
        )


class UnifiedAcquisitionPlanner:
    """Price calibration, response, exact-mask, and refresh measurements alike.

    All values are computed from a frozen ``DecisionUncertaintySnapshot`` and
    a predictable covariance-contraction fantasy. ``expected_decision_value``
    on the legacy request is deliberately ignored.
    """

    _CALIBRATION_KINDS = frozenset({"calibration_logits", "calibration_refit"})
    _RESPONSE_KINDS = frozenset(
        {
            "jacobian_rows",
            "marginal_statistics",
            "response_moments",
            "response_factor",
            "metric_factor",
            "covariance_rank_expansion",
            "directional_curvature",
            "finite_dose_response",
        }
    )
    _ORACLE_KINDS = frozenset({"hard_mask_oracle"})
    _VALIDITY_KINDS = frozenset({"model_validity", "response_transition"})

    @staticmethod
    def _requirement(
        request: MeasurementRequest,
        *,
        phase: str,
        kind: str,
        snapshot: DecisionUncertaintySnapshot,
    ) -> tuple[MeasurementRequirementClass, str | None]:
        declared = request.payload.get("measurement_requirement_class")
        obligation = request.payload.get("measurement_obligation_id")
        if declared is not None:
            requirement = str(declared)
            if requirement not in MEASUREMENT_REQUIREMENT_CLASSES:
                raise ValueError(
                    "measurement request declares an unknown requirement class"
                )
            typed_requirement: MeasurementRequirementClass = requirement  # type: ignore[assignment]
            typed_obligation = None if obligation is None else str(obligation)
            if typed_requirement == "optional":
                if typed_obligation is not None:
                    raise ValueError(
                        "optional measurement request may not declare an obligation ID"
                    )
            elif not typed_obligation:
                raise ValueError(
                    "required measurement request lacks its frozen obligation ID"
                )
            return typed_requirement, typed_obligation
        if (
            phase == "response_state"
            and kind in UnifiedAcquisitionPlanner._RESPONSE_KINDS
            and snapshot.uncovered_strata > 0
        ):
            return (
                "candidate_birth_prerequisite",
                f"candidate-birth-response-state:{snapshot.state_hash}",
            )
        if kind == "hard_mask_oracle":
            candidate_axis = content_hash(
                tuple(request.candidate_ids),
                prefix="irg-oracle-candidate-axis-",
            )
            return (
                "commit_prerequisite",
                f"exact-hard-mask-boundary:{snapshot.state_hash}:{candidate_axis}",
            )
        return "optional", None

    def opportunity(
        self,
        request: MeasurementRequest,
        snapshot: DecisionUncertaintySnapshot,
        *,
        phase: str,
        forecast: AcquisitionForecast | None = None,
    ) -> MeasurementOpportunity:
        if request.state_hash != snapshot.state_hash:
            raise ValueError(
                "measurement opportunity and uncertainty snapshot differ in state"
            )
        kind = str(request.kind)
        if kind in self._CALIBRATION_KINDS:
            source = "calibration_contrast"
            variance = snapshot.calibration_variance
            rationale = "reduce calibration contribution to active candidate contrasts"
        elif kind in self._RESPONSE_KINDS:
            source = "frontier_response_geometry"
            variance = snapshot.response_variance
            rationale = "reduce active response-direction and screen uncertainty"
        elif kind in self._ORACLE_KINDS:
            source = "bounded_exact_mask_contrast"
            variance = (0.5 * snapshot.safety_interval_width) ** 2
            rationale = "resolve the exact hard-mask safety boundary"
        elif kind in self._VALIDITY_KINDS:
            source = "local_model_validity"
            variance = snapshot.model_failure_probability * (
                1.0 - snapshot.model_failure_probability
            )
            rationale = "resolve whether local response geometry may be reused"
        else:
            source = "declared_auxiliary_uncertainty"
            variance = max(
                snapshot.response_variance,
                snapshot.calibration_variance,
                1e-12,
            )
            rationale = "reduce a declared auxiliary uncertainty term"

        count = max(1, int(request.context_selector.requested_count))
        current_count = max(
            1,
            int(snapshot.metadata.get(f"{kind}_sample_count", 0)),
        )
        default_fantasy = variance / (1.0 + count / float(current_count))
        posterior_variance = float(default_fantasy)
        forecast_hash = ""
        if forecast is not None:
            if (
                forecast.request_id != request.request_id
                or forecast.state_hash != snapshot.state_hash
            ):
                raise ValueError(
                    "acquisition forecast differs from its request/snapshot"
                )
            source = forecast.uncertainty_source
            variance = float(forecast.current_variance)
            posterior_variance = float(forecast.fantasy_posterior_variance)
            rationale = forecast.rationale
            forecast_hash = forecast.content_hash
        if variance <= 1e-18:
            posterior_variance = 0.0
        current_sd = math.sqrt(max(variance, 0.0))
        posterior_sd = math.sqrt(max(posterior_variance, 0.0))
        resolution_probability = normal_decision_change_probability(
            mean_gap=snapshot.frontier_boundary_gap,
            standard_error=max(current_sd, 1e-12),
            indifference=0.0,
        )
        regret_reduction = max(current_sd - posterior_sd, 0.0) * max(
            snapshot.frontier_regret_scale, 1.0
        )
        safety_value = (
            max(snapshot.safety_interval_width - 2.0 * posterior_sd, 0.0)
            if kind in self._ORACLE_KINDS
            else 0.0
        )
        coverage_gain = (
            snapshot.screen_uncertainty * (1.0 + float(snapshot.uncovered_strata))
            if kind in self._RESPONSE_KINDS
            else 0.0
        )
        model_failure_reduction = (
            max(current_sd - posterior_sd, 0.0) * snapshot.model_failure_probability
            if kind in self._VALIDITY_KINDS
            else 0.0
        )
        requirement_class, obligation_id = self._requirement(
            request,
            phase=phase,
            kind=kind,
            snapshot=snapshot,
        )
        return MeasurementOpportunity(
            request=request,
            uncertainty_source=source,
            current_variance=variance,
            fantasy_posterior_variance=posterior_variance,
            resolution_probability=resolution_probability,
            expected_regret_reduction=regret_reduction,
            safety_resolution_value=safety_value,
            coverage_gain=coverage_gain,
            model_failure_reduction=model_failure_reduction,
            reuse_bonus=(
                min(regret_reduction * 0.1, regret_reduction)
                if request.reuse_keys
                else 0.0
            ),
            requirement_class=requirement_class,
            obligation_id=obligation_id,
            rationale=rationale,
            forecast_hash=forecast_hash,
        )

    @staticmethod
    def score(
        opportunity: MeasurementOpportunity,
        *,
        predicted_cost: float,
    ) -> ScoredRequest:
        cost = float(predicted_cost)
        if not math.isfinite(cost) or cost <= 0.0:
            raise ValueError("unified acquisition requires positive predicted cost")
        factors = AcquisitionFactors(
            decision_change_probability=opportunity.resolution_probability,
            expected_decision_loss_reduction=opportunity.expected_regret_reduction,
            frontier_relevance=1.0,
            model_failure_reduction=opportunity.model_failure_reduction,
            coverage_gain=opportunity.coverage_gain,
            reuse_bonus=opportunity.reuse_bonus + opportunity.safety_resolution_value,
        )
        score = opportunity.expected_value / cost
        return ScoredRequest(
            request=opportunity.request,
            factors=factors,
            score=float(score),
            opportunity_hash=opportunity.content_hash,
            forecast_hash=opportunity.forecast_hash,
            predicted_cost=cost,
            requirement_class=opportunity.requirement_class,
            obligation_id=opportunity.obligation_id,
        )


def paired_contrast(
    candidate_a: str,
    candidate_b: str,
    values_a: Sequence[float] | np.ndarray,
    values_b: Sequence[float] | np.ndarray,
    *,
    indifference: float,
    relevance: float = 1.0,
) -> PairwiseContrast:
    left = np.asarray(values_a, dtype=np.float64).reshape(-1)
    right = np.asarray(values_b, dtype=np.float64).reshape(-1)
    if (
        left.shape != right.shape
        or left.size == 0
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise ValueError(
            "paired contrast requires aligned finite nonempty observations"
        )
    differences = left - right
    standard_error = (
        float(np.std(differences, ddof=1) / math.sqrt(differences.size))
        if differences.size >= 2
        else 0.0
    )
    return PairwiseContrast(
        candidate_a=str(candidate_a),
        candidate_b=str(candidate_b),
        mean_difference=float(np.mean(differences)),
        standard_error=standard_error,
        indifference=float(indifference),
        relevance=float(relevance),
    )


def practical_top_set(
    beliefs: Sequence[CandidateBelief],
    *,
    indifference: float,
    z_value: float = 2.0,
) -> tuple[str, ...]:
    """Retain candidates that are not practically dominated with uncertainty.

    Values are oriented so larger is better.  A candidate survives when its
    upper bound reaches the best lower bound minus the practical indifference.
    """

    items = tuple(beliefs)
    if not items:
        return ()
    if len({item.candidate_id for item in items}) != len(items):
        raise ValueError("candidate beliefs require unique IDs")
    delta = float(indifference)
    z = float(z_value)
    if delta < 0.0 or not math.isfinite(delta) or z < 0.0 or not math.isfinite(z):
        raise ValueError("indifference and z_value must be finite and nonnegative")
    lower = {
        item.candidate_id: item.mean_value - z * item.standard_error for item in items
    }
    upper = {
        item.candidate_id: item.mean_value + z * item.standard_error for item in items
    }
    best_lower = max(lower.values())
    survivors = [
        item.candidate_id
        for item in items
        if upper[item.candidate_id] >= best_lower - delta
    ]
    return tuple(sorted(survivors))


def normal_decision_change_probability(
    *,
    mean_gap: float,
    standard_error: float,
    indifference: float,
) -> float:
    """Working probability that further information can resolve a contrast."""

    gap = abs(float(mean_gap))
    se = float(standard_error)
    delta = float(indifference)
    if (
        not all(math.isfinite(value) for value in (gap, se, delta))
        or se < 0.0
        or delta < 0.0
    ):
        raise ValueError("contrast parameters are invalid")
    if se <= 1e-15:
        return 0.0 if gap > delta else 1.0
    distance = abs(gap - delta) / se
    # Smooth, monotone proxy: one at the decision boundary, vanishing away.
    return float(math.exp(-0.5 * distance * distance))


def score_request(
    request: MeasurementRequest,
    factors: AcquisitionFactors,
    *,
    wall_weight: float = 1.0,
    device_weight: float = 0.0,
) -> ScoredRequest:
    cost = request.expected_cost.scalar(
        wall_weight=wall_weight, device_weight=device_weight
    )
    return ScoredRequest(
        request=request,
        factors=factors,
        score=float(factors.value / cost),
    )


def select_request_batch(
    scored_requests: Sequence[ScoredRequest],
    *,
    maximum_requests: int,
    maximum_peak_memory_bytes: float | None = None,
    require_kind_diversity: bool = False,
) -> tuple[MeasurementRequest, ...]:
    """Select a deterministic compatible batch by value per measured cost."""

    limit = int(maximum_requests)
    if limit <= 0:
        raise ValueError("maximum_requests must be positive")
    memory_limit = (
        float("inf")
        if maximum_peak_memory_bytes is None
        else float(maximum_peak_memory_bytes)
    )
    if not math.isfinite(memory_limit) and maximum_peak_memory_bytes is not None:
        raise ValueError("maximum_peak_memory_bytes must be finite when provided")
    ordered = sorted(
        scored_requests,
        key=lambda item: (-item.score, item.request.kind, item.request.request_id),
    )
    selected: list[MeasurementRequest] = []
    peak = 0.0
    seen_kinds: set[str] = set()
    for item in ordered:
        request = item.request
        candidate_peak = max(peak, request.expected_cost.peak_memory_bytes)
        if candidate_peak > memory_limit + 1e-12:
            continue
        # Requests sharing a state are always safe to queue together.  Cross-
        # state batching is refused because it can silently mix mask/calibration identity.
        if selected and request.state_hash != selected[0].state_hash:
            continue
        if request.request_id in {chosen.request_id for chosen in selected}:
            continue
        selected.append(request)
        peak = candidate_peak
        seen_kinds.add(request.kind)
        if len(selected) >= limit:
            break
    if (
        require_kind_diversity
        and len({item.request.kind for item in ordered}) > 1
        and len(seen_kinds) < 2
    ):
        alternatives = [
            item.request for item in ordered if item.request.kind not in seen_kinds
        ]
        if alternatives and selected:
            replacement = alternatives[0]
            if (
                replacement.state_hash == selected[0].state_hash
                and replacement.expected_cost.peak_memory_bytes <= memory_limit
            ):
                selected[-1] = replacement
    return tuple(selected)


def allocate_contexts_for_active_contrasts(
    contrasts: Sequence[PairwiseContrast],
    *,
    total_budget: int,
    minimum_per_stratum: int = 0,
) -> dict[str, int]:
    """Allocate whole-route response rows for active decision contrasts.

    Unresolved contrasts contribute relevance- and ambiguity-weighted variance
    to each context stratum.  The common allocator then minimizes their summed
    active-direction estimator variance per physical measurement cost.  A
    physical stratum must have one cost regardless of which contrast consumes
    its row; inconsistent caller costs are rejected rather than averaged.
    """

    budget = int(total_budget)
    minimum = int(minimum_per_stratum)
    if budget <= 0 or minimum < 0:
        raise ValueError("allocation budget must be positive and minimum nonnegative")
    unresolved = [contrast for contrast in contrasts if not contrast.resolved]
    if not unresolved:
        unresolved = list(contrasts)
    strata = sorted(
        {stratum for contrast in unresolved for stratum in contrast.stratum_variances}
    )
    if not strata:
        raise ValueError("active contrasts contain no stratum information")
    floor_total = minimum * len(strata)
    if floor_total > budget:
        raise ValueError("minimum stratum coverage exceeds total budget")
    variance_coefficients = {stratum: 0.0 for stratum in strata}
    physical_costs: dict[str, float] = {}
    for contrast in unresolved:
        ambiguity = normal_decision_change_probability(
            mean_gap=contrast.mean_difference,
            standard_error=max(contrast.standard_error, 1e-12),
            indifference=contrast.indifference,
        )
        weight = max(contrast.relevance * ambiguity, 1e-12)
        for stratum in strata:
            variance = contrast.stratum_variances.get(stratum, 0.0)
            cost = contrast.stratum_costs.get(stratum, 1.0)
            prior_cost = physical_costs.setdefault(stratum, float(cost))
            if not math.isclose(
                prior_cost,
                float(cost),
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "one physical response stratum cannot declare contrast-dependent costs"
                )
            variance_coefficients[stratum] += weight * max(variance, 0.0)
    result = UnifiedNeymanAllocator().allocate(
        NeymanAllocationRequest(
            strata=tuple(
                NeymanStratumSpec(
                    stratum_id=stratum,
                    capacity=budget,
                    minimum_count=minimum,
                    estimand_weight=1.0,
                    allocation_standard_deviation=math.sqrt(
                        variance_coefficients[stratum]
                    ),
                    unit_cost=physical_costs.get(stratum, 1.0),
                )
                for stratum in strata
            ),
            cumulative_count_target=budget,
            role="active_whole_route_response_directions",
            policy_id=UNIFIED_NEYMAN_POLICY_ID,
        )
    )
    return {
        stratum: int(result.target_counts[stratum])
        for stratum in strata
    }


__all__ = [
    "AcquisitionFactors",
    "AcquisitionForecast",
    "AcquisitionSelectionReceipt",
    "ActiveResponseDirection",
    "CandidateBelief",
    "DecisionUncertaintySnapshot",
    "MeasurementOpportunity",
    "MeasurementRequirementClass",
    "MEASUREMENT_REQUIREMENT_CLASSES",
    "PairwiseContrast",
    "ScoredRequest",
    "UnifiedAcquisitionAdapter",
    "UnifiedAcquisitionPlanner",
    "active_response_direction",
    "allocate_contexts_for_active_contrasts",
    "normal_decision_change_probability",
    "paired_contrast",
    "practical_top_set",
    "response_direction_pairwise_contrast",
    "response_discrimination_context_scores",
    "score_request",
    "select_request_batch",
    "select_response_informative_contexts",
]
