"""Inert finite-intervention repricing for the partition-free v1 proposal plane.

The adapter consumes the existing paired response-transition measurement
surface.  It fits only the correction model frozen before those outcomes:
exact-minus-local-predicted action risk is projected onto a declared action
feature map.  The result can improve proposal prices, but it can never certify
or commit a pruning action.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np

from mechanistic_systems.core.axis_identity import (
    canonical_unique_string_axis,
    ordered_string_axis_hash,
)

from .contracts import MeasurementOutcome, content_hash
from .modern_v1_core import (
    V1AnalyticExactGateCorrectionLineage,
    V1InterventionalCorrectionLineage,
)
from .response_geometry import response_transition_receipt_from_outcome

_OBSERVATION_MODEL = "finite_action_residual_linear_feature_v1"
_RISK_FUNCTIONAL_SCHEMA = "partition_free_v1_quadratic_response_risk_v1"
_ACTION_FEATURE_SCHEMA = "partition_free_v1_action_feature_map_v1"
_TANGENT_PRICE_SCHEMA = "partition_free_v1_tangent_price_state_v1"
_ATOM_COST_SCHEMA = "partition_free_v1_atom_cost_state_v1"
_FINITE_ACTION_SCHEMA = "partition_free_v1_finite_action_definition_v1"
_INTERVENTION_DESIGN_SCHEMA = "partition_free_v1_finite_intervention_design_v1"
_CORRECTION_SPEC_SCHEMA = "partition_free_v1_interventional_correction_spec_v1"
_CORRECTION_RESULT_SCHEMA = "partition_free_v1_interventional_correction_result_v1"
_ANALYTIC_EXACT_GATE_RESULT_SCHEMA = (
    "partition_free_v1_analytic_exact_gate_correction_result_v1"
)
REMOVAL_INDICATOR_POSITIVE_DAMAGE_V1 = "removal_indicator_positive_damage_v1"


def _nonempty(value: str, label: str) -> str:
    result = str(value)
    if not result:
        raise ValueError(f"{label} must be nonempty")
    return result


def _readonly_vector(
    value: Sequence[float] | np.ndarray,
    *,
    name: str,
    nonnegative: bool = False,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite nonempty vector")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    owner: object = array
    seen: set[int] = set()
    while isinstance(owner, np.ndarray) and owner.base is not None:
        if id(owner) in seen:  # pragma: no cover - defensive cycle guard
            break
        seen.add(id(owner))
        owner = owner.base
    if (
        not array.flags.writeable
        and array.flags.c_contiguous
        and (
            isinstance(owner, bytes)
            or (isinstance(owner, memoryview) and owner.readonly)
        )
    ):
        # Canonical proposal vectors are frequently handed through several
        # typed lineage adapters.  Reuse an already immutable bytes-backed
        # snapshot instead of copying another census-scale route vector.
        return array
    # ``setflags(write=False)`` on an owning ndarray is reversible.  Hash-
    # bearing proposal state instead uses immutable ``bytes`` as its ultimate
    # buffer, so even ``setflags(write=True)`` cannot reopen the snapshot.
    payload = np.ascontiguousarray(array, dtype=np.float64).tobytes(order="C")
    return np.frombuffer(payload, dtype=np.float64, count=array.size)


def _readonly_matrix(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite nonempty matrix")
    shape = array.shape
    payload = np.ascontiguousarray(array, dtype=np.float64).tobytes(order="C")
    return np.frombuffer(payload, dtype=np.float64, count=array.size).reshape(shape)


def _deep_freeze(value: Any, label: str) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"{label} keys must be nonempty strings")
            frozen[key] = _deep_freeze(item, f"{label}.{key}")
        from types import MappingProxyType

        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _deep_freeze(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} must not contain nonfinite floats")
        return float(value)
    if isinstance(value, (str, bytes, int, bool)) or value is None:
        return value
    raise TypeError(f"{label} contains an unsupported value")


def _frozen_metadata(value: Mapping[str, Any]) -> Mapping[str, Any]:
    # Metadata is descriptive only.  Restricting it to the canonical hashing
    # language and copying it prevents post-hash mutation.
    payload = _deep_freeze(value, "metadata")
    if not isinstance(payload, Mapping):  # pragma: no cover - defensive
        raise TypeError("metadata must be a mapping")
    content_hash(payload, prefix="partition-free-v1-metadata-validation-")
    return payload


@dataclass(frozen=True, slots=True)
class V1QuadraticResponseRiskFunctional:
    """Fixed, independently recomputable risk on measured response rows.

    The generic response-transition provider may evaluate an arbitrary Python
    callback.  Modern-v1 repricing does not trust that callback or its asserted
    string identity.  It accepts this single typed functional and recomputes
    both risks from the concatenated measured rows before fitting a correction.
    Additional risk families require separate typed implementations.
    """

    response_dimension: int
    normalization_metric: np.ndarray
    construction_receipt_hash: str
    aggregation_id: str = "uniform_context_mean_quadratic_energy_v1"
    schema: str = _RISK_FUNCTIONAL_SCHEMA

    def __post_init__(self) -> None:
        if (
            isinstance(self.response_dimension, bool)
            or not isinstance(self.response_dimension, Integral)
            or int(self.response_dimension) <= 0
        ):
            raise ValueError("quadratic risk response_dimension must be positive")
        dimension = int(self.response_dimension)
        metric = np.asarray(self.normalization_metric, dtype=np.float64)
        if metric.shape != (dimension, dimension) or not np.isfinite(metric).all():
            raise ValueError("quadratic risk metric must be finite and dimension-aligned")
        metric = 0.5 * (metric + metric.T)
        eigenvalues = np.linalg.eigvalsh(metric)
        scale = max(1.0, float(np.max(np.abs(eigenvalues))))
        if float(np.min(eigenvalues)) < -1e-10 * scale:
            raise ValueError("quadratic risk metric must be positive semidefinite")
        if self.aggregation_id != "uniform_context_mean_quadratic_energy_v1":
            raise ValueError("unsupported modern-v1 risk aggregation")
        if self.schema != _RISK_FUNCTIONAL_SCHEMA:
            raise ValueError("unsupported modern-v1 risk-functional schema")
        object.__setattr__(self, "response_dimension", dimension)
        object.__setattr__(
            self,
            "normalization_metric",
            _readonly_matrix(metric, name="quadratic response-risk metric"),
        )
        object.__setattr__(
            self,
            "construction_receipt_hash",
            _nonempty(self.construction_receipt_hash, "construction_receipt_hash"),
        )

    @property
    def risk_estimand_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-response-risk-")

    def evaluate(
        self,
        rows: np.ndarray,
        context_ids: Sequence[str],
    ) -> float:
        values = np.asarray(rows, dtype=np.float64)
        contexts = tuple(context_ids)
        if (
            values.ndim != 2
            or values.shape != (len(contexts), self.response_dimension)
            or not contexts
            or len(set(contexts)) != len(contexts)
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                "quadratic risk requires finite response rows on one unique context axis"
            )
        row_risks = np.einsum(
            "ni,ij,nj->n",
            values,
            self.normalization_metric,
            values,
            optimize=True,
        )
        tolerance = 64.0 * np.finfo(np.float64).eps * max(
            1.0,
            float(np.max(np.abs(row_risks))),
        )
        if bool(np.any(row_risks < -tolerance)):
            raise RuntimeError("positive-semidefinite quadratic risk became negative")
        result = float(np.mean(np.maximum(row_risks, 0.0), dtype=np.float64))
        if not math.isfinite(result) or result < 0.0:
            raise RuntimeError("quadratic response risk is invalid")
        return result


@dataclass(frozen=True, slots=True)
class V1ActionFeatureMap:
    """Declared atom features whose action sums identify a correction."""

    route_ids: tuple[str, ...]
    feature_ids: tuple[str, ...]
    atom_feature_rows: np.ndarray
    construction_receipt_hash: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = _ACTION_FEATURE_SCHEMA

    def __post_init__(self) -> None:
        routes = canonical_unique_string_axis(
            self.route_ids,
            name="v1 correction route axis",
        )
        features = canonical_unique_string_axis(
            self.feature_ids,
            name="v1 correction feature axis",
        )
        rows = _readonly_matrix(
            self.atom_feature_rows,
            name="v1 atom feature rows",
        )
        if rows.shape != (len(routes), len(features)):
            raise ValueError("atom feature rows do not align with route/feature axes")
        object.__setattr__(self, "route_ids", routes)
        object.__setattr__(self, "feature_ids", features)
        object.__setattr__(self, "atom_feature_rows", rows)
        object.__setattr__(
            self,
            "construction_receipt_hash",
            _nonempty(self.construction_receipt_hash, "construction_receipt_hash"),
        )
        if self.schema != _ACTION_FEATURE_SCHEMA:
            raise ValueError("unsupported v1 action-feature schema")
        object.__setattr__(self, "metadata", _frozen_metadata(self.metadata))

    @property
    def route_axis_hash(self) -> str:
        return ordered_string_axis_hash(
            self.route_ids,
            name="v1 correction route axis",
        )

    @property
    def feature_map_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-action-feature-map-")


@dataclass(frozen=True, slots=True)
class V1TangentPriceState:
    """Full-axis tangent proposal prices at one exact chart base."""

    route_ids: tuple[str, ...]
    prices: np.ndarray
    base_mask_hash: str
    referent_id: str
    context_mixture_hash: str
    calibration_identity: str
    source_receipt_hash: str
    price_orientation_id: str
    schema: str = _TANGENT_PRICE_SCHEMA

    def __post_init__(self) -> None:
        routes = canonical_unique_string_axis(
            self.route_ids,
            name="v1 tangent-price route axis",
        )
        prices = _readonly_vector(self.prices, name="v1 tangent prices")
        if prices.shape != (len(routes),):
            raise ValueError("tangent prices do not align with the route axis")
        object.__setattr__(self, "route_ids", routes)
        object.__setattr__(self, "prices", prices)
        for name in (
            "base_mask_hash",
            "referent_id",
            "context_mixture_hash",
            "calibration_identity",
            "source_receipt_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if self.price_orientation_id != REMOVAL_INDICATOR_POSITIVE_DAMAGE_V1:
            raise ValueError(
                "tangent prices require the exact positive-damage removal-indicator orientation"
            )
        if self.schema != _TANGENT_PRICE_SCHEMA:
            raise ValueError("unsupported v1 tangent-price schema")

    @property
    def route_axis_hash(self) -> str:
        return ordered_string_axis_hash(
            self.route_ids,
            name="v1 tangent-price route axis",
        )

    @property
    def state_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-tangent-price-")


@dataclass(frozen=True, slots=True)
class V1AtomCostState:
    """Frozen physical/discovery cost functional on the same route axis."""

    route_ids: tuple[str, ...]
    costs: np.ndarray
    cost_functional_id: str
    source_receipt_hash: str
    schema: str = _ATOM_COST_SCHEMA

    def __post_init__(self) -> None:
        routes = canonical_unique_string_axis(
            self.route_ids,
            name="v1 atom-cost route axis",
        )
        costs = _readonly_vector(
            self.costs,
            name="v1 atom costs",
            nonnegative=True,
        )
        if costs.shape != (len(routes),) or not bool(np.any(costs > 0.0)):
            raise ValueError("atom costs must align and contain a positive cost")
        object.__setattr__(self, "route_ids", routes)
        object.__setattr__(self, "costs", costs)
        for name in ("cost_functional_id", "source_receipt_hash"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if self.schema != _ATOM_COST_SCHEMA:
            raise ValueError("unsupported v1 atom-cost schema")

    @property
    def route_axis_hash(self) -> str:
        return ordered_string_axis_hash(
            self.route_ids,
            name="v1 atom-cost route axis",
        )

    @property
    def cost_hash(self) -> str:
        return content_hash(self.costs, prefix="partition-free-v1-atom-cost-values-")

    @property
    def state_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-atom-cost-state-")


@dataclass(frozen=True, slots=True)
class V1FiniteActionDefinition:
    """One action frozen before its paired response outcome is observed."""

    action_id: str
    measurement_request_id: str
    candidate_mask_hash: str
    action_indices: tuple[int, ...]
    coefficients: tuple[float, ...]
    weight: float = 1.0
    schema: str = _FINITE_ACTION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "action_id",
            "measurement_request_id",
            "candidate_mask_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if self.schema != _FINITE_ACTION_SCHEMA:
            raise ValueError("unsupported v1 finite-action schema")
        if any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) < 0
            for value in self.action_indices
        ):
            raise TypeError("finite-action indices must be nonnegative integers")
        indices = tuple(int(value) for value in self.action_indices)
        coefficients = tuple(float(value) for value in self.coefficients)
        if (
            not indices
            or len(indices) != len(coefficients)
            or len(set(indices)) != len(indices)
            or any(index < 0 for index in indices)
            or any(not math.isfinite(value) or value == 0.0 for value in coefficients)
        ):
            raise ValueError("finite action requires unique indices and finite nonzero coefficients")
        ordered = tuple(
            sorted(
                zip(indices, coefficients, strict=True),
                key=lambda item: item[0],
            )
        )
        object.__setattr__(self, "action_indices", tuple(item[0] for item in ordered))
        object.__setattr__(self, "coefficients", tuple(item[1] for item in ordered))
        weight = float(self.weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("finite-action weight must be finite and positive")
        object.__setattr__(self, "weight", weight)

    @property
    def action_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-finite-action-")


@dataclass(frozen=True, slots=True)
class V1FiniteInterventionDesign:
    """Pre-outcome action set and statistical identity for repricing."""

    base_mask_hash: str
    referent_id: str
    context_mixture_hash: str
    calibration_identity: str
    risk_functional: V1QuadraticResponseRiskFunctional
    estimand_contrast_hash: str
    sampling_law_hash: str
    pre_outcome_lock_hash: str
    actions: tuple[V1FiniteActionDefinition, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = _INTERVENTION_DESIGN_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "base_mask_hash",
            "referent_id",
            "context_mixture_hash",
            "calibration_identity",
            "estimand_contrast_hash",
            "sampling_law_hash",
            "pre_outcome_lock_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if type(self.risk_functional) is not V1QuadraticResponseRiskFunctional:
            raise TypeError("finite intervention design requires a typed risk functional")
        if self.schema != _INTERVENTION_DESIGN_SCHEMA:
            raise ValueError("unsupported v1 intervention-design schema")
        actions = tuple(self.actions)
        if not actions or any(
            not isinstance(action, V1FiniteActionDefinition) for action in actions
        ):
            raise TypeError("finite intervention design requires typed actions")
        for label, values in (
            ("action IDs", tuple(action.action_id for action in actions)),
            (
                "measurement request IDs",
                tuple(action.measurement_request_id for action in actions),
            ),
            (
                "candidate mask hashes",
                tuple(action.candidate_mask_hash for action in actions),
            ),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"finite intervention {label} must be unique")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "metadata", _frozen_metadata(self.metadata))

    @property
    def risk_estimand_hash(self) -> str:
        return self.risk_functional.risk_estimand_hash

    @property
    def design_lock_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-intervention-design-")


@dataclass(frozen=True, slots=True)
class V1InterventionalCorrectionSpec:
    """Frozen model/conditioning gate for a full-rank correction fit."""

    feature_map_hash: str
    tangent_price_state_hash: str
    atom_cost_state_hash: str
    intervention_design_hash: str
    required_design_rank: int
    maximum_condition_number: float
    observation_model_id: str = _OBSERVATION_MODEL
    schema: str = _CORRECTION_SPEC_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "feature_map_hash",
            "tangent_price_state_hash",
            "atom_cost_state_hash",
            "intervention_design_hash",
            "observation_model_id",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if self.observation_model_id != _OBSERVATION_MODEL:
            raise ValueError("unsupported finite-intervention observation model")
        if self.schema != _CORRECTION_SPEC_SCHEMA:
            raise ValueError("unsupported v1 correction-spec schema")
        rank = int(self.required_design_rank)
        condition = float(self.maximum_condition_number)
        if rank <= 0 or rank != self.required_design_rank:
            raise ValueError("required_design_rank must be a positive integer")
        if not math.isfinite(condition) or condition < 1.0:
            raise ValueError("maximum_condition_number must be finite and at least one")
        object.__setattr__(self, "required_design_rank", rank)
        object.__setattr__(self, "maximum_condition_number", condition)

    @property
    def spec_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-correction-spec-")


@dataclass(frozen=True, slots=True)
class V1InterventionalCorrectionResult:
    """Proposal-only correction result plus exact producer lineage."""

    lineage: V1InterventionalCorrectionLineage
    corrected_prices: np.ndarray
    correction_coefficients: np.ndarray
    design_matrix: np.ndarray
    observed_action_residuals: np.ndarray
    fitted_action_residuals: np.ndarray
    result_receipt_hash: str
    schema: str = _CORRECTION_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.lineage, V1InterventionalCorrectionLineage):
            raise TypeError("result requires typed v1 correction lineage")
        corrected = _readonly_vector(self.corrected_prices, name="corrected prices")
        coefficients = _readonly_vector(
            self.correction_coefficients,
            name="correction coefficients",
        )
        design = _readonly_matrix(self.design_matrix, name="correction design")
        observed = _readonly_vector(
            self.observed_action_residuals,
            name="observed action residuals",
        )
        fitted = _readonly_vector(
            self.fitted_action_residuals,
            name="fitted action residuals",
        )
        if design.shape != (observed.size, coefficients.size) or fitted.shape != observed.shape:
            raise ValueError("correction result axes are inconsistent")
        object.__setattr__(self, "corrected_prices", corrected)
        object.__setattr__(self, "correction_coefficients", coefficients)
        object.__setattr__(self, "design_matrix", design)
        object.__setattr__(self, "observed_action_residuals", observed)
        object.__setattr__(self, "fitted_action_residuals", fitted)
        object.__setattr__(
            self,
            "result_receipt_hash",
            _nonempty(self.result_receipt_hash, "result_receipt_hash"),
        )
        if self.schema != _CORRECTION_RESULT_SCHEMA:
            raise ValueError("unsupported v1 correction-result schema")

    @property
    def can_certify_decision(self) -> bool:
        return False

    @property
    def claim_execution_certified(self) -> bool:
        return False

    @property
    def price_orientation_id(self) -> str:
        """Return the orientation bound transitively through lineage identity.

        Older hand-built proposal fixtures may lack this metadata, so the
        property fails closed rather than manufacturing a default.  Results
        issued by ``fit_identified_interventional_correction`` always carry it.
        """

        value = self.lineage.metadata.get("price_orientation_id")
        if value != REMOVAL_INDICATOR_POSITIVE_DAMAGE_V1:
            raise ValueError("correction result lacks the exact positive-damage price orientation")
        return value


@dataclass(frozen=True, slots=True)
class V1AnalyticExactGateCorrectionResult:
    """Exact raw-target-logit gate correction with no fitted intervention."""

    lineage: V1AnalyticExactGateCorrectionLineage
    corrected_prices: np.ndarray
    absolute_salience: np.ndarray
    result_receipt_hash: str
    schema: str = _ANALYTIC_EXACT_GATE_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if type(self.lineage) is not V1AnalyticExactGateCorrectionLineage:
            raise TypeError("analytic exact-gate result requires its exact lineage")
        corrected = _readonly_vector(
            self.corrected_prices,
            name="analytic exact-gate corrected prices",
        )
        absolute = _readonly_vector(
            self.absolute_salience,
            name="analytic exact-gate absolute salience",
            nonnegative=True,
        )
        if corrected.shape != absolute.shape:
            raise ValueError("analytic exact-gate fields must share the route axis")
        if content_hash(
            corrected,
            prefix="partition-free-v1-corrected-price-values-",
        ) != self.lineage.corrected_price_hash:
            raise ValueError("analytic exact-gate corrected prices differ from lineage")
        if content_hash(
            absolute,
            prefix="partition-free-v1-exact-gate-absolute-salience-values-",
        ) != self.lineage.absolute_salience_hash:
            raise ValueError("analytic exact-gate salience differs from lineage")
        object.__setattr__(self, "corrected_prices", corrected)
        object.__setattr__(self, "absolute_salience", absolute)
        object.__setattr__(
            self,
            "result_receipt_hash",
            _nonempty(self.result_receipt_hash, "result_receipt_hash"),
        )
        if self.schema != _ANALYTIC_EXACT_GATE_RESULT_SCHEMA:
            raise ValueError("unsupported analytic exact-gate result schema")

    @property
    def can_certify_decision(self) -> bool:
        return False

    @property
    def claim_execution_certified(self) -> bool:
        return False

    @property
    def price_orientation_id(self) -> str:
        if self.lineage.price_orientation_id != REMOVAL_INDICATOR_POSITIVE_DAMAGE_V1:
            raise ValueError("analytic exact-gate result has another price orientation")
        return self.lineage.price_orientation_id


def build_analytic_exact_gate_correction_result(
    *,
    route_ids: Sequence[str],
    corrected_prices: Sequence[float] | np.ndarray,
    absolute_salience: Sequence[float] | np.ndarray,
    base_mask_hash: str,
    referent_id: str,
    context_mixture_hash: str,
    calibration_identity: str,
    source_attribution_hash: str,
    streaming_receipt_hash: str,
    tangent_price_state_hash: str,
    atom_costs: V1AtomCostState,
    metadata: Mapping[str, Any] | None = None,
) -> V1AnalyticExactGateCorrectionResult:
    """Issue the analytic identity adapter without evaluating another mask."""

    routes = canonical_unique_string_axis(
        route_ids,
        name="analytic exact-gate route axis",
    )
    signed = _readonly_vector(
        corrected_prices,
        name="analytic exact-gate corrected prices",
    )
    absolute = _readonly_vector(
        absolute_salience,
        name="analytic exact-gate absolute salience",
        nonnegative=True,
    )
    route_axis_hash = ordered_string_axis_hash(
        routes,
        name="analytic exact-gate route axis",
    )
    if (
        signed.shape != (len(routes),)
        or absolute.shape != signed.shape
        or atom_costs.route_ids != routes
    ):
        raise ValueError("analytic exact-gate sources differ from the route axis")
    corrected_price_hash = content_hash(
        signed,
        prefix="partition-free-v1-corrected-price-values-",
    )
    absolute_hash = content_hash(
        absolute,
        prefix="partition-free-v1-exact-gate-absolute-salience-values-",
    )
    corrected_price_receipt_hash = content_hash(
        {
            "schema": "partition_free_v1_exact_gate_corrected_price_receipt_v1",
            "source_attribution_hash": _nonempty(
                source_attribution_hash,
                "source_attribution_hash",
            ),
            "streaming_receipt_hash": _nonempty(
                streaming_receipt_hash,
                "streaming_receipt_hash",
            ),
            "tangent_price_state_hash": _nonempty(
                tangent_price_state_hash,
                "tangent_price_state_hash",
            ),
            "corrected_price_hash": corrected_price_hash,
            "absolute_salience_hash": absolute_hash,
            "response_pullback": "raw_target_logit_unit_cotangent",
            "empirical_child_parent_mask_evaluation_count": 0,
        },
        prefix="partition-free-v1-exact-gate-corrected-price-",
    )
    lineage = V1AnalyticExactGateCorrectionLineage(
        base_mask_hash=_nonempty(base_mask_hash, "base_mask_hash"),
        route_axis_hash=route_axis_hash,
        referent_id=_nonempty(referent_id, "referent_id"),
        context_mixture_hash=_nonempty(
            context_mixture_hash,
            "context_mixture_hash",
        ),
        calibration_identity=_nonempty(
            calibration_identity,
            "calibration_identity",
        ),
        source_attribution_hash=str(source_attribution_hash),
        streaming_receipt_hash=str(streaming_receipt_hash),
        tangent_price_state_hash=str(tangent_price_state_hash),
        corrected_price_hash=corrected_price_hash,
        corrected_price_receipt_hash=corrected_price_receipt_hash,
        absolute_salience_hash=absolute_hash,
        atom_cost_hash=atom_costs.cost_hash,
        atom_cost_receipt_hash=atom_costs.state_hash,
        price_orientation_id=REMOVAL_INDICATOR_POSITIVE_DAMAGE_V1,
        metadata={
            "proposal_only": True,
            "canonical_exact_gate_correction": True,
            "raw_target_logit_unit_cotangent": True,
            "signed_rows_used_for_covariance": True,
            "absolute_rows_used_for_covariance": False,
            "empirical_interventional_fit": False,
            "empirical_child_parent_mask_evaluation_count": 0,
            **dict(metadata or {}),
        },
    )
    result_receipt_hash = content_hash(
        {
            "schema": _ANALYTIC_EXACT_GATE_RESULT_SCHEMA,
            "lineage_hash": lineage.lineage_hash,
            "corrected_price_hash": corrected_price_hash,
            "absolute_salience_hash": absolute_hash,
            "analytic_identity_no_fitted_residual": True,
        },
        prefix="partition-free-v1-analytic-exact-gate-result-",
    )
    return V1AnalyticExactGateCorrectionResult(
        lineage=lineage,
        corrected_prices=signed,
        absolute_salience=absolute,
        result_receipt_hash=result_receipt_hash,
    )


def _require_metadata_identity(
    outcome: MeasurementOutcome,
    *,
    name: str,
    expected: str,
) -> None:
    observed = str(outcome.metadata.get(name, ""))
    if observed != expected:
        raise ValueError(f"finite-action outcome identity mismatch for {name}")


def fit_identified_interventional_correction(
    *,
    feature_map: V1ActionFeatureMap,
    tangent_prices: V1TangentPriceState,
    atom_costs: V1AtomCostState,
    design: V1FiniteInterventionDesign,
    spec: V1InterventionalCorrectionSpec,
    outcomes: Sequence[MeasurementOutcome],
) -> V1InterventionalCorrectionResult:
    """Fit the preregistered correction and emit proposal-only lineage."""

    if not isinstance(feature_map, V1ActionFeatureMap):
        raise TypeError("feature_map must be V1ActionFeatureMap")
    if not isinstance(tangent_prices, V1TangentPriceState):
        raise TypeError("tangent_prices must be V1TangentPriceState")
    if not isinstance(atom_costs, V1AtomCostState):
        raise TypeError("atom_costs must be V1AtomCostState")
    if not isinstance(design, V1FiniteInterventionDesign):
        raise TypeError("design must be V1FiniteInterventionDesign")
    if not isinstance(spec, V1InterventionalCorrectionSpec):
        raise TypeError("spec must be V1InterventionalCorrectionSpec")
    if (
        spec.feature_map_hash != feature_map.feature_map_hash
        or spec.tangent_price_state_hash != tangent_prices.state_hash
        or spec.atom_cost_state_hash != atom_costs.state_hash
        or spec.intervention_design_hash != design.design_lock_hash
    ):
        raise ValueError("correction spec differs from its frozen inputs")
    if spec.required_design_rank != len(feature_map.feature_ids):
        raise ValueError("required rank must equal the declared feature dimension")
    route_hashes = {
        feature_map.route_axis_hash,
        tangent_prices.route_axis_hash,
        atom_costs.route_axis_hash,
    }
    if len(route_hashes) != 1 or feature_map.route_ids != tangent_prices.route_ids or feature_map.route_ids != atom_costs.route_ids:
        raise ValueError("correction inputs do not share one exact ordered route axis")
    identity = (
        tangent_prices.base_mask_hash,
        tangent_prices.referent_id,
        tangent_prices.context_mixture_hash,
        tangent_prices.calibration_identity,
    )
    if identity != (
        design.base_mask_hash,
        design.referent_id,
        design.context_mixture_hash,
        design.calibration_identity,
    ):
        raise ValueError("tangent-price and intervention identities differ")
    if tangent_prices.price_orientation_id != REMOVAL_INDICATOR_POSITIVE_DAMAGE_V1:
        raise ValueError("unsupported tangent-price orientation")
    observed_outcomes = tuple(outcomes)
    if len(observed_outcomes) != len(design.actions):
        raise ValueError("outcomes must cover every frozen action exactly")

    action_rows: list[np.ndarray] = []
    residuals: list[float] = []
    weights: list[float] = []
    measured_action_hashes: list[str] = []
    intervention_draw_ids: list[str] = []
    for action, outcome in zip(design.actions, observed_outcomes, strict=True):
        if outcome.integrity_hash != outcome.compute_integrity_hash():
            raise ValueError(
                "finite-action outcome changed after its integrity hash was frozen"
            )
        if outcome.request_id != action.measurement_request_id:
            raise ValueError("finite-action outcome request order/identity drifted")
        if outcome.calibration_state_id != design.calibration_identity:
            raise ValueError("finite-action outcome calibration identity drifted")
        for name, expected in (
            ("kind", "response_transition"),
            ("parent_mask_hash", design.base_mask_hash),
            ("candidate_mask_hash", action.candidate_mask_hash),
            ("referent_id", design.referent_id),
            ("context_mixture_hash", design.context_mixture_hash),
            ("risk_estimand_hash", design.risk_estimand_hash),
            ("estimand_contrast_hash", design.estimand_contrast_hash),
            ("sampling_law_hash", design.sampling_law_hash),
            ("pre_outcome_lock_hash", design.pre_outcome_lock_hash),
            ("risk_functional_hash", design.risk_estimand_hash),
        ):
            _require_metadata_identity(outcome, name=name, expected=expected)
        if outcome.metadata.get("risk_values_derived_from_measured_rows") is not True:
            raise ValueError(
                "finite-action risks must be derived from the measured paired rows"
            )
        values = np.asarray(outcome.values, dtype=np.float64)
        dimension = design.risk_functional.response_dimension
        if values.shape != (len(outcome.context_ids), 2 * dimension):
            raise ValueError(
                "finite-action paired response rows differ from the frozen risk dimension"
            )
        observed_metric = np.asarray(
            outcome.metadata.get("normalization_metric"),
            dtype=np.float64,
        )
        if not np.array_equal(
            observed_metric,
            design.risk_functional.normalization_metric,
        ):
            raise ValueError(
                "finite-action normalization metric differs from the frozen risk functional"
            )
        predicted_rows = values[:, :dimension]
        exact_rows = values[:, dimension:]
        predicted_risk = design.risk_functional.evaluate(
            predicted_rows,
            outcome.context_ids,
        )
        exact_risk = design.risk_functional.evaluate(
            exact_rows,
            outcome.context_ids,
        )
        try:
            declared_predicted_risk = float(outcome.metadata["predicted_risk"])
            declared_exact_risk = float(outcome.metadata["exact_risk"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "finite-action outcome lacks provider risk diagnostics"
            ) from error
        risk_tolerance = 64.0 * np.finfo(np.float64).eps * max(
            1.0,
            abs(predicted_risk),
            abs(exact_risk),
        )
        if not math.isclose(
            declared_predicted_risk,
            predicted_risk,
            rel_tol=0.0,
            abs_tol=risk_tolerance,
        ) or not math.isclose(
            declared_exact_risk,
            exact_risk,
            rel_tol=0.0,
            abs_tol=risk_tolerance,
        ):
            raise ValueError(
                "provider risk values differ from independently recomputed measured-row risk"
            )
        receipt = response_transition_receipt_from_outcome(outcome)
        if (
            receipt.action_indices != action.action_indices
            or receipt.coefficients != action.coefficients
            or receipt.candidate_mask_hash != action.candidate_mask_hash
        ):
            raise ValueError("finite-action outcome differs from its frozen action")
        if any(index >= len(feature_map.route_ids) for index in action.action_indices):
            raise ValueError("finite action is outside the declared route axis")
        coefficients = np.asarray(action.coefficients, dtype=np.float64)
        selected_features = feature_map.atom_feature_rows[
            np.asarray(action.action_indices, dtype=np.int64)
        ]
        action_rows.append(coefficients @ selected_features)
        residuals.append(float(exact_risk - predicted_risk))
        weights.append(action.weight)
        measured_action_hashes.append(
            content_hash(
                {
                    "frozen_action_hash": action.action_hash,
                    "outcome_integrity_hash": outcome.integrity_hash,
                    "response_transition_receipt_hash": receipt.content_hash,
                },
                prefix="partition-free-v1-measured-action-",
            )
        )
        intervention_draw_ids.extend(
            f"{action.measurement_request_id}:{context_id}"
            for context_id in outcome.context_ids
        )
    if len(set(intervention_draw_ids)) != len(intervention_draw_ids):
        raise ValueError("finite intervention draw IDs must be globally unique")

    matrix = np.ascontiguousarray(np.vstack(action_rows), dtype=np.float64)
    observed = np.ascontiguousarray(residuals, dtype=np.float64)
    weight_array = np.ascontiguousarray(weights, dtype=np.float64)
    weighted_matrix = matrix * np.sqrt(weight_array)[:, None]
    weighted_observed = observed * np.sqrt(weight_array)
    singular_values = np.linalg.svd(weighted_matrix, compute_uv=False)
    if singular_values.size == 0:
        raise ValueError("finite-intervention design has no singular values")
    tolerance = np.finfo(np.float64).eps * max(weighted_matrix.shape) * singular_values[0]
    rank = int(np.sum(singular_values > tolerance))
    if rank < spec.required_design_rank:
        raise ValueError("finite-intervention correction is underidentified")
    condition = float(singular_values[0] / singular_values[rank - 1])
    if not math.isfinite(condition) or condition > spec.maximum_condition_number:
        raise ValueError("finite-intervention design exceeds its conditioning gate")
    coefficients, _residual_sum, fitted_rank, _singular = np.linalg.lstsq(
        weighted_matrix,
        weighted_observed,
        rcond=None,
    )
    if int(fitted_rank) != rank or not np.isfinite(coefficients).all():
        raise RuntimeError("finite-intervention least-squares fit was numerically invalid")
    fitted = matrix @ coefficients
    corrected = tangent_prices.prices + feature_map.atom_feature_rows @ coefficients
    if not np.isfinite(corrected).all():
        raise RuntimeError("finite-intervention repricing produced nonfinite prices")

    design_matrix_hash = content_hash(
        {
            "design_lock_hash": design.design_lock_hash,
            "matrix": matrix,
            "weights": weight_array,
        },
        prefix="partition-free-v1-correction-design-matrix-",
    )
    support_hash = content_hash(
        {
            "action_hashes": tuple(action.action_hash for action in design.actions),
            "measured_action_hashes": tuple(measured_action_hashes),
            "intervention_draw_ids": tuple(intervention_draw_ids),
        },
        prefix="partition-free-v1-interventional-support-",
    )
    identification_receipt_hash = content_hash(
        {
            "spec_hash": spec.spec_hash,
            "design_matrix_hash": design_matrix_hash,
            "rank": rank,
            "singular_values": singular_values,
            "condition_number": condition,
            "observed_residuals": observed,
            "fitted_residuals": fitted,
            "weighted_residual_sum_squares": float(
                np.dot(weight_array, (observed - fitted) ** 2)
            ),
        },
        prefix="partition-free-v1-identification-receipt-",
    )
    corrected_price_hash = content_hash(
        corrected,
        prefix="partition-free-v1-corrected-price-values-",
    )
    corrected_price_receipt_hash = content_hash(
        {
            "tangent_price_state_hash": tangent_prices.state_hash,
            "feature_map_hash": feature_map.feature_map_hash,
            "identification_receipt_hash": identification_receipt_hash,
            "correction_coefficients": coefficients,
            "corrected_price_hash": corrected_price_hash,
            "price_orientation_id": tangent_prices.price_orientation_id,
        },
        prefix="partition-free-v1-corrected-price-receipt-",
    )
    observation_model_hash = content_hash(
        {
            "observation_model_id": spec.observation_model_id,
            "risk_estimand_hash": design.risk_estimand_hash,
            "equation": "exact_risk_minus_local_predicted_risk_equals_phi_action_beta",
            "weighted_least_squares": True,
            "regularization": "none_full_rank_required",
        },
        prefix="partition-free-v1-observation-model-",
    )
    lineage = V1InterventionalCorrectionLineage(
        correction_kind="identified_finite_intervention_v1",
        base_mask_hash=design.base_mask_hash,
        route_axis_hash=feature_map.route_axis_hash,
        referent_id=design.referent_id,
        context_mixture_hash=design.context_mixture_hash,
        calibration_identity=design.calibration_identity,
        feature_map_hash=feature_map.feature_map_hash,
        estimand_contrast_hash=design.estimand_contrast_hash,
        observation_model_hash=observation_model_hash,
        interventional_support_hash=support_hash,
        design_matrix_hash=design_matrix_hash,
        identification_receipt_hash=identification_receipt_hash,
        corrected_price_hash=corrected_price_hash,
        corrected_price_receipt_hash=corrected_price_receipt_hash,
        atom_cost_hash=atom_costs.cost_hash,
        atom_cost_receipt_hash=atom_costs.state_hash,
        measured_action_hashes=tuple(measured_action_hashes),
        intervention_draw_ids=tuple(intervention_draw_ids),
        identified_design_rank=rank,
        required_design_rank=spec.required_design_rank,
        metadata={
            "proposal_only": True,
            "can_certify_decision": False,
            "feature_dimension": len(feature_map.feature_ids),
            "action_count": len(design.actions),
            "condition_number": condition,
            "tangent_price_source_receipt_hash": tangent_prices.source_receipt_hash,
            "tangent_price_state_hash": tangent_prices.state_hash,
            "price_orientation_id": tangent_prices.price_orientation_id,
            "atom_cost_source_receipt_hash": atom_costs.source_receipt_hash,
            "intervention_design_lock_hash": design.design_lock_hash,
        },
    )
    result_receipt_hash = content_hash(
        {
            "lineage_hash": lineage.lineage_hash,
            "corrected_price_hash": corrected_price_hash,
            "correction_coefficients": coefficients,
            "observed_action_residuals": observed,
            "fitted_action_residuals": fitted,
        },
        prefix="partition-free-v1-correction-result-",
    )
    return V1InterventionalCorrectionResult(
        lineage=lineage,
        corrected_prices=corrected,
        correction_coefficients=coefficients,
        design_matrix=matrix,
        observed_action_residuals=observed,
        fitted_action_residuals=fitted,
        result_receipt_hash=result_receipt_hash,
    )


__all__ = [
    "REMOVAL_INDICATOR_POSITIVE_DAMAGE_V1",
    "V1ActionFeatureMap",
    "V1AnalyticExactGateCorrectionResult",
    "V1AtomCostState",
    "V1FiniteActionDefinition",
    "V1FiniteInterventionDesign",
    "V1InterventionalCorrectionResult",
    "V1InterventionalCorrectionSpec",
    "V1QuadraticResponseRiskFunctional",
    "V1TangentPriceState",
    "build_analytic_exact_gate_correction_result",
    "fit_identified_interventional_correction",
]
