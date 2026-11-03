"""Scientific contracts for mechanism identification.

CircuitFlow owns immutable transport envelopes.  This module owns the
scientific decisions that produce those envelopes.  In particular, a query is
separate from a model study, and measurement is separate from circuit
identification, so one score artifact can support multiple matched circuit
families without changing study identity.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from circuitflow.contracts import (
    ActionPlan,
    AnalysisPlan,
    DataView,
    GeometryPlan,
    KernelContractionSpec,
    MechanismPlan,
    RouteCoordinateSchema,
    RuntimePlan,
    ValidationResult,
)
from circuitflow.contracts import CircuitPlan as OperationalCircuitPlan
from circuitflow.contracts import ScorePlan as OperationalMeasurementPlan
from contracts.hashing import stable_digest

SCHEMA_VERSION = "mechanistic_systems_v3"
QUERY_KINDS = frozenset(
    {
        "class_logit",
        "margin",
        "learned_head",
        "semantic_direction",
        "behavioral_readout",
        "roi_readout",
        "custom",
    }
)
PLAN_STATUSES = frozenset({"frozen", "exploratory", "diagnostic", "legacy_matched"})
PRIMARY_MEASUREMENT_STATISTICS = (
    "count",
    "score_sum",
    "score_sum_squares",
    "absolute_score_sum",
    "absolute_score_sum_squares",
    "positive_count",
    "negative_count",
    "block_support",
)
CANONICAL_CIRCUIT_ENGINE = "classic_circuit_identification"
PRODUCTION_CIRCUIT_SEARCH_ALGORITHM = "statistical_sequential_symmetric_frontier"
SAMPLED_PATH_GROUP_REFERENCE_POLICY = "committed_group_crossfit_reference"
CANONICAL_CIRCUIT_SEARCH_ALGORITHMS = frozenset(
    {
        PRODUCTION_CIRCUIT_SEARCH_ALGORITHM,
        "global_nested_rate_distortion",
        "risk_aware_rate_distortion_v2",
        "reverse_hierarchical_equalized_marginal",
    }
)
ROUTE_GATE_V1 = "route_gate_v1"
RISK_AWARE_ROUTE_GATE_V2 = "risk_aware_route_gate_v2"
SAMPLED_PATH_FRONTIER_V1 = "sampled_path_frontier_v1"
SAMPLED_PATH_V1_CALIBRATION_ALGORITHM = (
    "sampled_path_frontier_v1.finite_horizon_positive_affine_lbfgs"
)
PROPER_SCORE_SHRINKAGE_AFFINE_V2 = "proper_score_identity_shrinkage_affine_v2"
CANONICAL_PRUNING_CALIBRATION_ALGORITHM = PROPER_SCORE_SHRINKAGE_AFFINE_V2
CANONICAL_CALIBRATION_IDENTITY_ANCHOR_STRENGTHS = (0.25, 1.0, 4.0, 16.0, 64.0)
CANONICAL_CALIBRATION_SELECTION_BLOCKS = 4
CANONICAL_CALIBRATION_ANCHOR_PROBABILITY_EPSILON = 1e-12
CANONICAL_CALIBRATION_MAX_ITERATIONS = 64
CANONICAL_CALIBRATION_KKT_TOLERANCE = 1e-7
CANONICAL_CALIBRATION_HESSIAN_RIDGE = 1e-10
CANONICAL_CALIBRATION_OBJECTIVE_TOLERANCE = 1e-9
CANONICAL_CALIBRATION_REFERENCE_TOLERANCE = 1e-8
CANONICAL_TARGET_CALIBRATION_MODE = "overlap_factored_post_query_affine"
RATE_DISTORTION_V2 = "rate_distortion_v2"
CANONICAL_PARTITION_POLICIES = frozenset(
    {
        "unpartitioned",
        "attributional_sign",
        "route_coordinate_sign",
        "random_count_matched",
        "magnitude_stratified_random",
        "wrong_target_attributional_sign",
        "nuisance_stratified_sign_permutation",
    }
)


def canonical_calibration_policy() -> dict[str, Any]:
    """Return the complete scientific policy implied by the canonical ID."""

    return {
        "target_calibration_mode": CANONICAL_TARGET_CALIBRATION_MODE,
        "calibration_optimization_contract": (
            CANONICAL_PRUNING_CALIBRATION_ALGORITHM
        ),
        "calibration_identity_anchor_strengths": list(
            CANONICAL_CALIBRATION_IDENTITY_ANCHOR_STRENGTHS
        ),
        "calibration_selection_blocks": CANONICAL_CALIBRATION_SELECTION_BLOCKS,
        "calibration_anchor_probability_epsilon": (
            CANONICAL_CALIBRATION_ANCHOR_PROBABILITY_EPSILON
        ),
        "calibration_max_iterations": CANONICAL_CALIBRATION_MAX_ITERATIONS,
        "calibration_kkt_tolerance": CANONICAL_CALIBRATION_KKT_TOLERANCE,
        "calibration_hessian_ridge": CANONICAL_CALIBRATION_HESSIAN_RIDGE,
        "calibration_objective_tolerance": (
            CANONICAL_CALIBRATION_OBJECTIVE_TOLERANCE
        ),
        "calibration_reference_tolerance": (
            CANONICAL_CALIBRATION_REFERENCE_TOLERANCE
        ),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def ordered_axis_hash(axis_ids: Sequence[str]) -> str:
    """Hash an ordered coordinate axis independently of its enclosing role."""

    values = tuple(str(value) for value in axis_ids)
    if not values or len(set(values)) != len(values):
        raise ValueError("ordered axis IDs must be nonempty and unique")
    return stable_digest({"ordered_axis_ids": list(values)})


class _ScientificContract:
    schema_name: ClassVar[str]

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @property
    def content_hash(self) -> str:
        return stable_digest(self.to_dict())

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps({**self.to_dict(), "content_hash": self.content_hash}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        return target


@dataclass(frozen=True)
class StudySpec(_ScientificContract):
    """Query-independent identity of one executable model/data study cell."""

    schema_name: ClassVar[str] = "mechanistic_study_spec"

    study_id: str
    model_id: str
    checkpoint_hash: str
    model_definition_hash: str
    label_map_hash: str
    preprocessing: Mapping[str, Any]
    data_source: Mapping[str, Any]
    route_schema: RouteCoordinateSchema
    seeds: Mapping[str, int]
    notes: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": SCHEMA_VERSION,
            "study_id": self.study_id,
            "model_id": self.model_id,
            "checkpoint_hash": self.checkpoint_hash,
            "model_definition_hash": self.model_definition_hash,
            "label_map_hash": self.label_map_hash,
            "preprocessing": _jsonable(self.preprocessing),
            "data_source": _jsonable(self.data_source),
            "route_schema": self.route_schema.to_dict(),
            "seeds": {str(key): int(value) for key, value in self.seeds.items()},
            "notes": _jsonable(self.notes),
        }

    def validate(self) -> ValidationResult:
        issues: list[str] = []
        for name, value in (
            ("study_id", self.study_id),
            ("model_id", self.model_id),
            ("checkpoint_hash", self.checkpoint_hash),
            ("model_definition_hash", self.model_definition_hash),
            ("label_map_hash", self.label_map_hash),
        ):
            if not value:
                issues.append(f"study_{name}_empty")
        for name, value in (
            ("checkpoint", self.checkpoint_hash),
            ("model_definition", self.model_definition_hash),
            ("label_map", self.label_map_hash),
        ):
            if len(value) != 64:
                issues.append(f"study_{name}_hash_invalid")
        if not self.preprocessing:
            issues.append("study_preprocessing_empty")
        if not self.data_source:
            issues.append("study_data_source_empty")
        if not self.seeds or any(not isinstance(value, int) for value in self.seeds.values()):
            issues.append("study_seeds_empty_or_invalid")
        issues.extend(f"route_schema:{issue}" for issue in self.route_schema.validate().issues)
        return ValidationResult(not issues, tuple(issues))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> StudySpec:
        result = cls(
            study_id=str(payload.get("study_id", "")),
            model_id=str(payload.get("model_id", "")),
            checkpoint_hash=str(payload.get("checkpoint_hash", "")),
            model_definition_hash=str(payload.get("model_definition_hash", "")),
            label_map_hash=str(payload.get("label_map_hash", "")),
            preprocessing=dict(payload.get("preprocessing", {}) or {}),
            data_source=dict(payload.get("data_source", {}) or {}),
            route_schema=RouteCoordinateSchema.from_mapping(
                dict(payload.get("route_schema", {}) or {})
            ),
            seeds={
                str(key): int(value)
                for key, value in dict(payload.get("seeds", {}) or {}).items()
            },
            notes=dict(payload.get("notes", {}) or {}),
        )
        result.validate().require_valid()
        return result

    @classmethod
    def load(cls, path: str | Path) -> StudySpec:
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class QuerySpec(_ScientificContract):
    """A target computation independent of architecture and runtime details."""

    schema_name: ClassVar[str] = "mechanistic_query_spec"

    query_id: str
    query_kind: str
    ontology_id: str
    axis_ids: tuple[str, ...]
    evaluator: Mapping[str, Any]
    response_semantics: str
    source_hash: str
    output_metric: Mapping[str, Any] = field(default_factory=dict)
    directional_functional: Mapping[str, Any] = field(default_factory=dict)
    notes: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_name": self.schema_name,
            "schema_version": SCHEMA_VERSION,
            "query_id": self.query_id,
            "query_kind": self.query_kind,
            "ontology_id": self.ontology_id,
            "axis_ids": list(self.axis_ids),
            "evaluator": _jsonable(self.evaluator),
            "response_semantics": self.response_semantics,
            "source_hash": self.source_hash,
            "notes": _jsonable(self.notes),
        }
        # Preserve hashes of sealed scalar-query contracts.  Vector-output
        # metrics become part of identity only when explicitly declared.
        if self.output_metric:
            payload["output_metric"] = _jsonable(self.output_metric)
        if self.directional_functional:
            payload["directional_functional"] = _jsonable(self.directional_functional)
        return payload

    def validate(self) -> ValidationResult:
        issues: list[str] = []
        if not self.query_id:
            issues.append("query_id_empty")
        if self.query_kind not in QUERY_KINDS:
            issues.append("query_kind_unsupported")
        if not self.ontology_id:
            issues.append("ontology_id_empty")
        if not self.axis_ids or len(set(self.axis_ids)) != len(self.axis_ids):
            issues.append("query_axis_ids_empty_or_nonunique")
        if not self.evaluator:
            issues.append("query_evaluator_empty")
        if not self.response_semantics:
            issues.append("query_response_semantics_empty")
        if len(self.source_hash) != 64:
            issues.append("query_source_hash_invalid")
        if self.output_metric:
            metric_axis = tuple(str(value) for value in self.output_metric.get("axis_ids", ()) or ())
            if metric_axis != self.axis_ids:
                issues.append("query_output_metric_axis_mismatch")
            metric_kind = str(self.output_metric.get("metric_kind", "identity"))
            if metric_kind not in {"identity", "diagonal", "full_psd"}:
                issues.append("query_output_metric_unsupported")
            elif metric_kind == "diagonal":
                try:
                    diagonal = tuple(
                        float(value)
                        for value in self.output_metric.get("diagonal", ()) or ()
                    )
                except (TypeError, ValueError):
                    diagonal = tuple()
                if (
                    len(diagonal) != len(self.axis_ids)
                    or any(not math.isfinite(value) or value < 0.0 for value in diagonal)
                ):
                    issues.append("query_output_metric_diagonal_invalid")
            elif metric_kind == "full_psd":
                try:
                    matrix = np.asarray(
                        self.output_metric.get("matrix", ()), dtype=np.float64
                    )
                    valid_matrix = (
                        matrix.shape == (len(self.axis_ids), len(self.axis_ids))
                        and bool(np.isfinite(matrix).all())
                        and bool(np.allclose(matrix, matrix.T, atol=1e-10, rtol=1e-10))
                        and float(np.min(np.linalg.eigvalsh(matrix))) >= -1e-9
                    )
                except (TypeError, ValueError, np.linalg.LinAlgError):
                    valid_matrix = False
                if not valid_matrix:
                    issues.append("query_output_metric_full_psd_invalid")
        if self.directional_functional:
            direction_axis = tuple(
                str(value) for value in self.directional_functional.get("axis_ids", ()) or ()
            )
            weights = tuple(self.directional_functional.get("weights", ()) or ())
            if direction_axis != self.axis_ids or len(weights) != len(self.axis_ids):
                issues.append("query_direction_axis_mismatch")
            else:
                try:
                    numeric_weights = tuple(float(value) for value in weights)
                    if any(not math.isfinite(value) for value in numeric_weights):
                        issues.append("query_direction_nonfinite")
                    elif not any(abs(value) > 0.0 for value in numeric_weights):
                        issues.append("query_direction_zero")
                except (TypeError, ValueError):
                    issues.append("query_direction_nonfinite")
        return ValidationResult(not issues, tuple(issues))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> QuerySpec:
        result = cls(
            query_id=str(payload.get("query_id", "")),
            query_kind=str(payload.get("query_kind", "")),
            ontology_id=str(payload.get("ontology_id", "")),
            axis_ids=tuple(str(item) for item in payload.get("axis_ids", ()) or ()),
            evaluator=dict(payload.get("evaluator", {}) or {}),
            response_semantics=str(payload.get("response_semantics", "")),
            source_hash=str(payload.get("source_hash", "")),
            output_metric=dict(payload.get("output_metric", {}) or {}),
            directional_functional=dict(payload.get("directional_functional", {}) or {}),
            notes=dict(payload.get("notes", {}) or {}),
        )
        result.validate().require_valid()
        return result


@dataclass(frozen=True)
class MeasurementPlan(_ScientificContract):
    """Frozen observation model for local functional-sensitivity measurements."""

    schema_name: ClassVar[str] = "mechanistic_measurement_plan"

    plan_id: str
    study_hash: str
    queries: tuple[QuerySpec, ...]
    measurement_view: DataView
    estimator_id: str
    estimator_version: str
    measurement_semantics: str
    route_schema_hash: str
    context_block_ids: tuple[str, ...]
    compact_statistics: tuple[str, ...] = PRIMARY_MEASUREMENT_STATISTICS
    calibration_estimators: tuple[str, ...] = tuple()
    persistence_policy: str = "sufficient_statistics_only"
    random_seed: int = 0
    status: str = "exploratory"
    estimator_policy: Mapping[str, Any] = field(default_factory=dict)
    notes: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_name": self.schema_name,
            "schema_version": SCHEMA_VERSION,
            "plan_id": self.plan_id,
            "study_hash": self.study_hash,
            "queries": [query.to_dict() for query in self.queries],
            "measurement_view": self.measurement_view.to_dict(),
            "estimator_id": self.estimator_id,
            "estimator_version": self.estimator_version,
            "measurement_semantics": self.measurement_semantics,
            "route_schema_hash": self.route_schema_hash,
            "context_block_ids": list(self.context_block_ids),
            "compact_statistics": list(self.compact_statistics),
            "calibration_estimators": list(self.calibration_estimators),
            "persistence_policy": self.persistence_policy,
            "random_seed": int(self.random_seed),
            "status": self.status,
            "notes": _jsonable(self.notes),
        }
        if self.estimator_policy:
            payload["estimator_policy"] = _jsonable(self.estimator_policy)
        return payload

    def validate(self, *, claim_bearing: bool | None = None) -> ValidationResult:
        issues: list[str] = []
        if not self.plan_id:
            issues.append("measurement_plan_id_empty")
        if len(self.study_hash) != 64:
            issues.append("measurement_study_hash_invalid")
        if not self.queries:
            issues.append("measurement_queries_empty")
        if len({query.query_id for query in self.queries}) != len(self.queries):
            issues.append("measurement_query_ids_nonunique")
        for query in self.queries:
            issues.extend(f"query:{query.query_id}:{issue}" for issue in query.validate().issues)
        issues.extend(f"measurement_view:{issue}" for issue in self.measurement_view.validate().issues)
        if not self.estimator_id or not self.estimator_version:
            issues.append("measurement_estimator_identity_incomplete")
        if self.estimator_version == ROUTE_GATE_V1 and self.estimator_policy:
            mutable_keys = set(self.estimator_policy) - {"algorithm_version", "oracle_receipt_hash"}
            if mutable_keys:
                issues.append("route_gate_v1_policy_mutation_forbidden")
        if self.estimator_version == RISK_AWARE_ROUTE_GATE_V2:
            try:
                threshold = float(self.estimator_policy.get("sign_probability_threshold", 0.95))
                one_sided_z = float(self.estimator_policy.get("one_sided_z", 1.6448536269514722))
            except (TypeError, ValueError):
                threshold, one_sided_z = 0.0, -1.0
            if not 0.5 < threshold < 1.0 or one_sided_z < 0.0:
                issues.append("risk_aware_route_gate_policy_invalid")
            if len(self.context_block_ids) < 2:
                issues.append("risk_aware_route_gate_requires_context_blocks")
        if not self.measurement_semantics:
            issues.append("measurement_semantics_empty")
        if len(self.route_schema_hash) != 64:
            issues.append("measurement_route_schema_hash_invalid")
        if not self.context_block_ids or len(set(self.context_block_ids)) != len(self.context_block_ids):
            issues.append("measurement_context_blocks_empty_or_nonunique")
        if tuple(self.compact_statistics) != PRIMARY_MEASUREMENT_STATISTICS:
            issues.append("measurement_statistics_not_canonical")
        if self.persistence_policy != "sufficient_statistics_only":
            issues.append("measurement_raw_context_route_persistence_forbidden")
        if self.status not in PLAN_STATUSES:
            issues.append("measurement_status_unsupported")
        require_claim = self.status == "frozen" if claim_bearing is None else bool(claim_bearing)
        if require_claim and self.status != "frozen":
            issues.append("claim_bearing_measurement_plan_not_frozen")
        return ValidationResult(not issues, tuple(issues))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> MeasurementPlan:
        result = cls(
            plan_id=str(payload.get("plan_id", "")),
            study_hash=str(payload.get("study_hash", "")),
            queries=tuple(QuerySpec.from_mapping(item) for item in payload.get("queries", ()) or ()),
            measurement_view=DataView.from_mapping(dict(payload.get("measurement_view", {}) or {})),
            estimator_id=str(payload.get("estimator_id", "")),
            estimator_version=str(payload.get("estimator_version", "")),
            measurement_semantics=str(payload.get("measurement_semantics", "")),
            route_schema_hash=str(payload.get("route_schema_hash", "")),
            context_block_ids=tuple(str(item) for item in payload.get("context_block_ids", ()) or ()),
            compact_statistics=tuple(str(item) for item in payload.get("compact_statistics", PRIMARY_MEASUREMENT_STATISTICS) or ()),
            calibration_estimators=tuple(str(item) for item in payload.get("calibration_estimators", ()) or ()),
            persistence_policy=str(payload.get("persistence_policy", "sufficient_statistics_only")),
            random_seed=int(payload.get("random_seed", 0)),
            status=str(payload.get("status", "exploratory")),
            estimator_policy=dict(payload.get("estimator_policy", {}) or {}),
            notes=dict(payload.get("notes", {}) or {}),
        )
        result.validate().require_valid()
        return result

    @classmethod
    def load(cls, path: str | Path) -> MeasurementPlan:
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_operational(self) -> OperationalMeasurementPlan:
        """Lower into CircuitFlow's submission-neutral packet contract."""

        self.validate(claim_bearing=self.status == "frozen").require_valid()
        return OperationalMeasurementPlan(
            plan_id=self.plan_id,
            study_hash=self.study_hash,
            estimator_id=self.estimator_id,
            score_view=self.measurement_view,
            target_evaluator={
                "queries": [query.to_dict() for query in self.queries],
                "measurement_semantics": self.measurement_semantics,
            },
            score_semantics=self.measurement_semantics,
            # These are the pruning-authoritative heads emitted by the exact
            # production ScoreStore. Context moments are additional compact
            # evidence, not substitutes for either aggregate route head.
            score_heads=("route_scores", "route_abs_scores"),
            random_seed=self.random_seed,
            status=self.status,
            notes={
                **dict(self.notes),
                "canonical_measurement_plan_hash": self.content_hash,
                "context_block_ids": list(self.context_block_ids),
                "persistence_policy": self.persistence_policy,
            },
        )


@dataclass(frozen=True)
class CircuitPlan(_ScientificContract):
    """Rate-distortion identity for one circuit family and its matched controls."""

    schema_name: ClassVar[str] = "mechanistic_circuit_plan"

    plan_id: str
    study_hash: str
    measurement_artifact_hash: str
    route_universe_hash: str
    optimization_view: DataView
    evaluation_view: DataView
    response_distortion: Mapping[str, Any]
    route_cost: Mapping[str, Any]
    mask_budget: Mapping[str, Any]
    partition_policy: str
    matched_controls: tuple[str, ...]
    random_seed: int
    fixed_measurement_layer_ids: tuple[str, ...] = ()
    engine_id: str = CANONICAL_CIRCUIT_ENGINE
    search_policy: Mapping[str, Any] = field(
        default_factory=lambda: {"algorithm": "global_nested_rate_distortion"}
    )
    ensemble_policy: Mapping[str, Any] = field(default_factory=dict)
    status: str = "exploratory"
    notes: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": SCHEMA_VERSION,
            "plan_id": self.plan_id,
            "study_hash": self.study_hash,
            "measurement_artifact_hash": self.measurement_artifact_hash,
            "route_universe_hash": self.route_universe_hash,
            "optimization_view": self.optimization_view.to_dict(),
            "evaluation_view": self.evaluation_view.to_dict(),
            "objective": "constrained_rate_distortion",
            "response_distortion": _jsonable(self.response_distortion),
            "route_cost": _jsonable(self.route_cost),
            "mask_budget": _jsonable(self.mask_budget),
            "partition_policy": self.partition_policy,
            "matched_controls": list(self.matched_controls),
            "random_seed": int(self.random_seed),
            "fixed_measurement_layer_ids": list(self.fixed_measurement_layer_ids),
            "engine_id": self.engine_id,
            "search_policy": _jsonable(self.search_policy),
            "ensemble_policy": _jsonable(self.ensemble_policy),
            "status": self.status,
            "notes": _jsonable(self.notes),
        }

    def validate(self, *, claim_bearing: bool | None = None) -> ValidationResult:
        issues: list[str] = []
        require_claim = (
            self.status == "frozen"
            if claim_bearing is None
            else bool(claim_bearing)
        )
        for name, value in (
            ("plan_id", self.plan_id),
            ("study_hash", self.study_hash),
            ("measurement_artifact_hash", self.measurement_artifact_hash),
            ("route_universe_hash", self.route_universe_hash),
        ):
            if not value:
                issues.append(f"circuit_{name}_empty")
        for name, value in (
            ("study", self.study_hash),
            ("measurement", self.measurement_artifact_hash),
            ("route_universe", self.route_universe_hash),
        ):
            if len(value) != 64:
                issues.append(f"circuit_{name}_hash_invalid")
        issues.extend(f"optimization_view:{issue}" for issue in self.optimization_view.validate().issues)
        issues.extend(f"evaluation_view:{issue}" for issue in self.evaluation_view.validate().issues)
        if not self.response_distortion or not self.route_cost or not self.mask_budget:
            issues.append("circuit_rate_distortion_objective_incomplete")
        if not self.partition_policy:
            issues.append("circuit_partition_policy_empty")
        elif self.partition_policy not in CANONICAL_PARTITION_POLICIES:
            issues.append("circuit_partition_policy_not_canonical")
        if len(set(self.matched_controls)) != len(self.matched_controls):
            issues.append("circuit_matched_controls_nonunique")
        if len(set(self.fixed_measurement_layer_ids)) != len(
            self.fixed_measurement_layer_ids
        ):
            issues.append("circuit_fixed_measurement_layers_nonunique")
        if any(not value for value in self.fixed_measurement_layer_ids):
            issues.append("circuit_fixed_measurement_layer_empty")
        if any(control not in CANONICAL_PARTITION_POLICIES for control in self.matched_controls):
            issues.append("circuit_matched_control_not_canonical")
        if self.partition_policy in self.matched_controls:
            issues.append("circuit_partition_policy_repeated_as_control")
        if self.engine_id != CANONICAL_CIRCUIT_ENGINE:
            issues.append("circuit_engine_not_canonical")
        search_algorithm = str(self.search_policy.get("algorithm", ""))
        if search_algorithm not in CANONICAL_CIRCUIT_SEARCH_ALGORITHMS:
            issues.append("circuit_search_algorithm_not_canonical")
        elif search_algorithm == PRODUCTION_CIRCUIT_SEARCH_ALGORITHM:
            required_semantics = {
                "frontier_policy": "level0_symmetric_frontier",
                "candidate_evaluation_mode": "statistical_sequential",
                "statistical_estimand": "crossfit_risk",
                "statistical_sampling_design": "cost_optimal_two_stage_neyman",
                "statistical_inference_engine": "bayesian_hierarchical_risk",
                "statistical_panel_execution": "single_forward_crossfit_reuse",
                "sampling_path": "stochastic_class_indexed_nested_panels",
                "paired_comparison": "common_random_numbers",
                "crossfit_calibration": "fit_complement_score_heldout_swap",
                "slope_numerator": "paired_crossfit_risk_delta",
                "denominator_policy": (
                    "requested_sign_density_delta_times_total_layer_route_count"
                ),
                "partition_schedule": "symmetric_full_turns",
                "initial_partition": "negative",
                "readout_policy": "scored_then_fixed_during_pruning",
                "reference_anchor_policy": SAMPLED_PATH_GROUP_REFERENCE_POLICY,
                "final_validation": "untouched_full_validation_split",
            }
            for key, expected in required_semantics.items():
                if (
                    key == "reference_anchor_policy"
                    and not self.search_policy.get(key)
                    and not require_claim
                ):
                    continue
                if str(self.search_policy.get(key, "")) != expected:
                    issues.append(f"statistical_sequential_{key}_invalid")
            calibration_contract = str(
                self.search_policy.get("calibration_optimization_contract", "")
            )
            if calibration_contract not in {
                CANONICAL_PRUNING_CALIBRATION_ALGORITHM,
                SAMPLED_PATH_V1_CALIBRATION_ALGORITHM,
            }:
                issues.append(
                    "statistical_sequential_calibration_optimization_contract_invalid"
                )
            elif calibration_contract == CANONICAL_PRUNING_CALIBRATION_ALGORITHM:
                if str(
                    self.search_policy.get("target_calibration_mode", "")
                ) != CANONICAL_TARGET_CALIBRATION_MODE:
                    issues.append(
                        "statistical_sequential_target_calibration_mode_invalid"
                    )
                strengths = tuple(
                    float(value)
                    for value in self.search_policy.get(
                        "calibration_identity_anchor_strengths", ()
                    )
                )
                if strengths != CANONICAL_CALIBRATION_IDENTITY_ANCHOR_STRENGTHS:
                    issues.append(
                        "statistical_sequential_calibration_identity_anchor_strengths_invalid"
                    )
                if int(
                    self.search_policy.get("calibration_selection_blocks", 0) or 0
                ) != CANONICAL_CALIBRATION_SELECTION_BLOCKS:
                    issues.append(
                        "statistical_sequential_calibration_selection_blocks_invalid"
                    )
                observed_epsilon = float(
                    self.search_policy.get(
                        "calibration_anchor_probability_epsilon", float("nan")
                    )
                )
                if not math.isfinite(observed_epsilon) or abs(
                    observed_epsilon
                    - CANONICAL_CALIBRATION_ANCHOR_PROBABILITY_EPSILON
                ) > 1e-18:
                    issues.append(
                        "statistical_sequential_calibration_anchor_probability_epsilon_invalid"
                    )
                integer_policy = {
                    "calibration_max_iterations": (
                        CANONICAL_CALIBRATION_MAX_ITERATIONS
                    ),
                }
                for key, expected in integer_policy.items():
                    if int(self.search_policy.get(key, 0) or 0) != expected:
                        issues.append(f"statistical_sequential_{key}_invalid")
                numeric_policy = {
                    "calibration_kkt_tolerance": (
                        CANONICAL_CALIBRATION_KKT_TOLERANCE
                    ),
                    "calibration_hessian_ridge": (
                        CANONICAL_CALIBRATION_HESSIAN_RIDGE
                    ),
                    "calibration_objective_tolerance": (
                        CANONICAL_CALIBRATION_OBJECTIVE_TOLERANCE
                    ),
                    "calibration_reference_tolerance": (
                        CANONICAL_CALIBRATION_REFERENCE_TOLERANCE
                    ),
                }
                for key, expected in numeric_policy.items():
                    observed = float(self.search_policy.get(key, float("nan")))
                    tolerance = max(abs(expected) * 1e-12, 1e-18)
                    if not math.isfinite(observed) or abs(observed - expected) > tolerance:
                        issues.append(f"statistical_sequential_{key}_invalid")
            fold_count = self.search_policy.get("crossfit_fold_count")
            try:
                fold_count_value = int(fold_count)
            except (TypeError, ValueError):
                fold_count_value = 0
            if fold_count_value < 2:
                issues.append("statistical_sequential_crossfit_fold_count_invalid")
            stages = self.search_policy.get("sample_stages_per_class")
            if (
                not isinstance(stages, Sequence)
                or isinstance(stages, (str, bytes))
                or not stages
            ):
                issues.append("statistical_sequential_sample_stages_missing")
            else:
                try:
                    stage_values = tuple(int(value) for value in stages)
                except (TypeError, ValueError):
                    stage_values = tuple()
                if (
                    not stage_values
                    or any(value <= 0 for value in stage_values)
                    or tuple(sorted(set(stage_values))) != stage_values
                ):
                    issues.append("statistical_sequential_sample_stages_invalid")
            threshold = self.search_policy.get("sensitivity_threshold")
            try:
                threshold_value = float(threshold)
            except (TypeError, ValueError):
                threshold_value = float("nan")
            if not 0.0 < threshold_value < float("inf"):
                issues.append("statistical_sequential_sensitivity_threshold_invalid")
        elif search_algorithm == "reverse_hierarchical_equalized_marginal":
            if str(self.search_policy.get("layer_order", "")) != "deep_to_shallow":
                issues.append("historical_circuit_layer_order_invalid")
            partition_schedule = str(self.search_policy.get("partition_schedule", ""))
            if partition_schedule not in {"symmetric_full_turns", "single_branch"}:
                issues.append("historical_circuit_partition_schedule_invalid")
            if (
                partition_schedule == "symmetric_full_turns"
                and str(self.search_policy.get("initial_partition", "")) != "negative"
            ):
                issues.append("historical_circuit_initial_partition_invalid")
            if (
                str(self.search_policy.get("frontier_controller", ""))
                != "normalized_stage_span"
            ):
                issues.append("historical_circuit_frontier_controller_invalid")
            if str(self.search_policy.get("frontier_selection", "")) != "freeze_reached_pair":
                issues.append("historical_circuit_frontier_selection_invalid")
            if (
                str(self.search_policy.get("mask_cardinality", ""))
                != "ceil_retained_fraction_times_partition_count"
            ):
                issues.append("historical_circuit_mask_cardinality_invalid")
            threshold = self.search_policy.get("sensitivity_threshold")
            try:
                threshold_value = float(threshold)
            except (TypeError, ValueError):
                threshold_value = float("nan")
            if not 0.0 <= threshold_value < float("inf"):
                issues.append("historical_circuit_sensitivity_threshold_invalid")
            if (
                str(self.search_policy.get("denominator_policy", ""))
                != "requested_sign_density_delta_times_total_layer_route_count"
            ):
                issues.append("historical_circuit_denominator_policy_invalid")
            if (
                str(self.search_policy.get("affine_calibration", ""))
                != "per_candidate_class_balanced_scale_offset"
            ):
                issues.append("historical_circuit_affine_calibration_invalid")
            schedules = self.search_policy.get("density_schedules")
            if not isinstance(schedules, Mapping) or not schedules:
                issues.append("historical_circuit_density_schedules_missing")
            if str(self.search_policy.get("residual_route_partition_policy", "")) not in {
                "absent",
                "unit_gate_nonnegative",
            }:
                issues.append("historical_circuit_residual_partition_policy_invalid")
        elif search_algorithm == "risk_aware_rate_distortion_v2":
            if str(self.search_policy.get("algorithm_version", "")) != RATE_DISTORTION_V2:
                issues.append("risk_aware_rate_distortion_version_invalid")
            if str(self.search_policy.get("measurement_version", "")) != RISK_AWARE_ROUTE_GATE_V2:
                issues.append("risk_aware_rate_distortion_measurement_invalid")
            if str(self.search_policy.get("uncertain_sign_policy", "")) != "unresolved":
                issues.append("risk_aware_rate_distortion_uncertain_sign_policy_invalid")
            families = tuple(self.search_policy.get("calibration_families", ()) or ())
            if families != ("none", "shared_affine", "classwise_diagonal_affine"):
                issues.append("risk_aware_rate_distortion_calibration_family_invalid")
            if str(self.search_policy.get("calibration_selection", "")) != "simplest_within_one_se":
                issues.append("risk_aware_rate_distortion_calibration_selection_invalid")
        if self.status not in PLAN_STATUSES:
            issues.append("circuit_status_unsupported")
        if require_claim and self.status != "frozen":
            issues.append("claim_bearing_circuit_plan_not_frozen")
        if require_claim and not self.matched_controls:
            issues.append("claim_bearing_circuit_plan_missing_matched_controls")
        return ValidationResult(not issues, tuple(issues))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> CircuitPlan:
        result = cls(
            plan_id=str(payload.get("plan_id", "")),
            study_hash=str(payload.get("study_hash", "")),
            measurement_artifact_hash=str(payload.get("measurement_artifact_hash", "")),
            route_universe_hash=str(payload.get("route_universe_hash", "")),
            optimization_view=DataView.from_mapping(dict(payload.get("optimization_view", {}) or {})),
            evaluation_view=DataView.from_mapping(dict(payload.get("evaluation_view", {}) or {})),
            response_distortion=dict(payload.get("response_distortion", {}) or {}),
            route_cost=dict(payload.get("route_cost", {}) or {}),
            mask_budget=dict(payload.get("mask_budget", {}) or {}),
            partition_policy=str(payload.get("partition_policy", "")),
            matched_controls=tuple(str(item) for item in payload.get("matched_controls", ()) or ()),
            random_seed=int(payload.get("random_seed", 0)),
            fixed_measurement_layer_ids=tuple(
                str(item)
                for item in payload.get("fixed_measurement_layer_ids", ()) or ()
            ),
            engine_id=str(payload.get("engine_id", CANONICAL_CIRCUIT_ENGINE)),
            search_policy=dict(
                payload.get("search_policy", {})
                or {"algorithm": "global_nested_rate_distortion"}
            ),
            ensemble_policy=dict(payload.get("ensemble_policy", {}) or {}),
            status=str(payload.get("status", "exploratory")),
            notes=dict(payload.get("notes", {}) or {}),
        )
        result.validate().require_valid()
        return result

    @classmethod
    def load(cls, path: str | Path) -> CircuitPlan:
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_operational(self) -> OperationalCircuitPlan:
        """Lower the scientific plan into CircuitFlow's shared engine boundary."""

        self.validate().require_valid()
        return OperationalCircuitPlan(
            plan_id=self.plan_id,
            study_hash=self.study_hash,
            engine_id="classic_frontier",
            measurement_artifact_hash=self.measurement_artifact_hash,
            route_universe_hash=self.route_universe_hash,
            optimization_view=self.optimization_view,
            evaluation_view=self.evaluation_view,
            mask_budget=dict(self.mask_budget),
            partition_policy=self.partition_policy,
            matched_controls=self.matched_controls,
            random_seed=self.random_seed,
            ensemble_policy=dict(self.ensemble_policy),
            status=self.status,
            engine_config={
                "canonical_engine_id": self.engine_id,
                "search_policy": dict(self.search_policy),
                "objective": "constrained_rate_distortion",
                "response_distortion": dict(self.response_distortion),
                "route_cost": dict(self.route_cost),
                "fixed_measurement_layer_ids": list(
                    self.fixed_measurement_layer_ids
                ),
            },
        )


@dataclass(frozen=True)
class InterventionPlan(ActionPlan):
    """Scientific prospective-intervention contract.

    The operational fields intentionally match CircuitFlow's ``ActionPlan``,
    but the distinct schema prevents action scheduling from being mistaken for
    a frozen scientific intervention design.
    """

    schema_name: ClassVar[str] = "mechanistic_intervention_plan"


__all__ = [
    "AnalysisPlan",
    "CANONICAL_CIRCUIT_ENGINE",
    "CANONICAL_CIRCUIT_SEARCH_ALGORITHMS",
    "CANONICAL_CALIBRATION_ANCHOR_PROBABILITY_EPSILON",
    "CANONICAL_CALIBRATION_HESSIAN_RIDGE",
    "CANONICAL_CALIBRATION_IDENTITY_ANCHOR_STRENGTHS",
    "CANONICAL_CALIBRATION_KKT_TOLERANCE",
    "CANONICAL_CALIBRATION_MAX_ITERATIONS",
    "CANONICAL_CALIBRATION_OBJECTIVE_TOLERANCE",
    "CANONICAL_CALIBRATION_REFERENCE_TOLERANCE",
    "CANONICAL_CALIBRATION_SELECTION_BLOCKS",
    "CANONICAL_PARTITION_POLICIES",
    "CANONICAL_PRUNING_CALIBRATION_ALGORITHM",
    "CANONICAL_TARGET_CALIBRATION_MODE",
    "CircuitPlan",
    "DataView",
    "GeometryPlan",
    "InterventionPlan",
    "KernelContractionSpec",
    "MeasurementPlan",
    "MechanismPlan",
    "PRIMARY_MEASUREMENT_STATISTICS",
    "PRODUCTION_CIRCUIT_SEARCH_ALGORITHM",
    "RATE_DISTORTION_V2",
    "PROPER_SCORE_SHRINKAGE_AFFINE_V2",
    "RISK_AWARE_ROUTE_GATE_V2",
    "ROUTE_GATE_V1",
    "SAMPLED_PATH_FRONTIER_V1",
    "SAMPLED_PATH_GROUP_REFERENCE_POLICY",
    "SAMPLED_PATH_V1_CALIBRATION_ALGORITHM",
    "QuerySpec",
    "RouteCoordinateSchema",
    "RuntimePlan",
    "StudySpec",
    "canonical_calibration_policy",
    "ordered_axis_hash",
]
