"""Scientific and runtime contracts for CircuitFlow v2.

The old ``ExperimentContract`` combines stable study identity with score and
mask decisions.  That is convenient for a single pipeline but prevents clean
score reuse across circuit policies.  CircuitFlow keeps those layers separate:

``StudySpec -> ScorePlan -> CircuitPlan -> MechanismPlan -> GeometryPlan -> AnalysisPlan``.

Each object is immutable, serialisable, and addressed by a canonical SHA-256
digest.  ``RuntimePlan`` is intentionally excluded from scientific lineage so
the same computation can be rerun on a different GPU partition without
changing the scientific object it produces.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from contracts.hashing import stable_digest

SCHEMA_VERSION = "circuitflow_v3"
PLAN_STATUSES = frozenset({"frozen", "exploratory", "legacy_matched", "diagnostic"})
ENGINE_IDS = frozenset({"classic_frontier", "thermobayes_program"})
VALID_PARTITIONS = frozenset({"kempner_h200", "kempner_h100", "kempner", "kempner_rtx", "cpu"})
RESPONSE_BASIS_KINDS = frozenset({"state", "mechanism"})
CANONICAL_STATE_BASELINE_FAMILIES = (
    "all_layer_direct_sum",
    "spatial_channel_covariance",
    "quadratic_state",
    "readout_response",
    "dense_attribution_or_jacobian",
    "low_level_image",
)
STATE_BASELINE_SCOPES = frozenset({"model", "global"})
MECHANISM_EXECUTION_FAMILIES = frozenset(
    {
        "full_state",
        "coarse_structure_pushforward",
        "structural_pushforward",
        "route_aware_transport",
        "route_transport",
        "aggregate_masked_state",
        "attributional_signed_use",
        "route_coordinate_signed_use",
        "local_signed_use",
        "local_additive_use",
        "endogenous_factorized_use",
        "endogenous_route_coordinate_use",
        "role_counterfactual",
        "interaction",
        "intervention_response",
        "interventional_program",
    }
)
KERNEL_COMPARISON_AXES = frozenset({"stimulus", "query", "role", "joint", "interaction"})
KERNEL_CONTRACTION_TYPES = frozenset({"trace", "partial_trace", "group_diagonal", "low_rank_psd"})
RESPONSE_TRANSFORMS = frozenset({"identity", "two_way_interaction"})
KERNEL_SOURCE_OBJECTS = frozenset(
    {"response_basis", "circuit_anatomy", "state_response", "intervention_response"}
)
RESPONSE_KERNEL_METRICS = frozenset({"raw_dot", "cosine", "correlation"})
STRUCTURAL_KERNEL_METRICS = frozenset(
    {"jaccard", "binary_cosine", "containment_mean", "weighted_cosine"}
)
KERNEL_METRICS = RESPONSE_KERNEL_METRICS | STRUCTURAL_KERNEL_METRICS
KERNEL_CENTERING_POLICIES = frozenset({"none", "kernel", "row"})
KERNEL_NORMALIZATION_POLICIES = frozenset({"none", "row_l2", "group_l2"})
LAYER_AGGREGATION_POLICIES = frozenset({"equal", "direct_sum", "custom"})
KERNEL_PRIMITIVE_RETENTION_POLICIES = frozenset(
    {"aggregate_only", "layer", "layer_and_group"}
)
_INTEGRATED_AXES_BY_COMPARISON = {
    "stimulus": frozenset({"query", "role", "feature", "layer"}),
    "query": frozenset({"stimulus", "role", "feature", "layer"}),
    "role": frozenset({"stimulus", "query", "feature", "layer"}),
    "joint": frozenset({"role", "feature", "layer"}),
    "interaction": frozenset({"role", "feature", "layer"}),
}
_INTEGRATED_AXES_BY_SOURCE_COMPARISON = {
    ("circuit_anatomy", "query"): frozenset({"route", "layer"}),
    ("circuit_anatomy", "role"): frozenset({"query", "route", "layer"}),
    ("state_response", "stimulus"): frozenset({"feature", "layer"}),
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _mapping(value: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return {str(key): _jsonable(item) for key, item in value.items()}


def _nonempty(value: Any, name: str, issues: list[str]) -> None:
    if not str(value).strip():
        issues.append(f"{name} must be nonempty")


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    issues: tuple[str, ...] = tuple()

    def require_valid(self) -> None:
        if not self.valid:
            raise ValueError("invalid CircuitFlow contract: " + "; ".join(self.issues))


class _ContentAddressed:
    """Small mixin used by specs and plans with a canonical JSON payload."""

    schema_name: ClassVar[str]

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover - implemented by subclasses.
        raise NotImplementedError

    @property
    def content_hash(self) -> str:
        return stable_digest(self.to_dict())

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {**self.to_dict(), "content_hash": self.content_hash},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return target


@dataclass(frozen=True)
class DataView:
    """An executable, content-addressed image or stimulus view."""

    view_id: str
    dataset_id: str
    ids_hash: str
    selector: Mapping[str, Any]
    preprocessing_id: str
    label_map_hash: str
    axis_role: str = "stimulus"
    notes: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "dataset_id": self.dataset_id,
            "ids_hash": self.ids_hash,
            "selector": _jsonable(self.selector),
            "preprocessing_id": self.preprocessing_id,
            "label_map_hash": self.label_map_hash,
            "axis_role": self.axis_role,
            "notes": _jsonable(self.notes),
        }

    @property
    def view_hash(self) -> str:
        return stable_digest(self.to_dict())

    def validate(self) -> ValidationResult:
        issues: list[str] = []
        for name, value in (
            ("view_id", self.view_id),
            ("dataset_id", self.dataset_id),
            ("ids_hash", self.ids_hash),
            ("preprocessing_id", self.preprocessing_id),
            ("label_map_hash", self.label_map_hash),
        ):
            _nonempty(value, name, issues)
        if not isinstance(self.selector, Mapping) or not self.selector:
            issues.append("selector must be a nonempty mapping")
        if self.axis_role not in {"stimulus", "category", "query", "query_stimulus_pair"}:
            issues.append("unsupported DataView axis_role")
        return ValidationResult(not issues, tuple(issues))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> DataView:
        result = cls(
            view_id=str(payload.get("view_id", payload.get("id", ""))),
            dataset_id=str(payload.get("dataset_id", payload.get("data_source_id", ""))),
            ids_hash=str(payload.get("ids_hash", payload.get("image_ids_hash", ""))),
            selector=_mapping(payload.get("selector", {}), "selector"),
            preprocessing_id=str(payload.get("preprocessing_id", "")),
            label_map_hash=str(payload.get("label_map_hash", payload.get("class_map_hash", ""))),
            axis_role=str(payload.get("axis_role", "stimulus")),
            notes=_mapping(payload.get("notes", {}), "notes"),
        )
        result.validate().require_valid()
        return result


@dataclass(frozen=True)
class RouteCoordinateSchema:
    """Stable coordinate identity for a model's prunable route universe."""

    schema_id: str
    coordinates_hash: str
    coordinate_table_uri: str
    architecture_family: str
    route_kinds: tuple[str, ...]
    alias_map_hash: str = ""
    lowering_version: str = "v1"
    notes: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "coordinates_hash": self.coordinates_hash,
            "coordinate_table_uri": self.coordinate_table_uri,
            "architecture_family": self.architecture_family,
            "route_kinds": list(self.route_kinds),
            "alias_map_hash": self.alias_map_hash,
            "lowering_version": self.lowering_version,
            "notes": _jsonable(self.notes),
        }

    @property
    def schema_hash(self) -> str:
        return stable_digest(self.to_dict())

    def validate(self) -> ValidationResult:
        issues: list[str] = []
        for name, value in (
            ("schema_id", self.schema_id),
            ("coordinates_hash", self.coordinates_hash),
            ("coordinate_table_uri", self.coordinate_table_uri),
            ("architecture_family", self.architecture_family),
            ("lowering_version", self.lowering_version),
        ):
            _nonempty(value, name, issues)
        if not self.route_kinds:
            issues.append("route_kinds must be nonempty")
        return ValidationResult(not issues, tuple(issues))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> RouteCoordinateSchema:
        result = cls(
            schema_id=str(payload.get("schema_id", "")),
            coordinates_hash=str(payload.get("coordinates_hash", "")),
            coordinate_table_uri=str(payload.get("coordinate_table_uri", "")),
            architecture_family=str(payload.get("architecture_family", "")),
            route_kinds=tuple(str(item) for item in payload.get("route_kinds", ()) or ()),
            alias_map_hash=str(payload.get("alias_map_hash", "")),
            lowering_version=str(payload.get("lowering_version", "v1")),
            notes=_mapping(payload.get("notes", {}), "notes"),
        )
        result.validate().require_valid()
        return result


@dataclass(frozen=True)
class StudySpec(_ContentAddressed):
    """Stage-independent scientific identity for one model/target study cell."""

    schema_name: ClassVar[str] = "circuitflow_study_spec"

    study_id: str
    model_id: str
    checkpoint_hash: str
    model_definition_hash: str
    target_ontology: str
    target_ids_hash: str
    class_map_hash: str
    preprocessing: Mapping[str, Any]
    data_source: Mapping[str, Any]
    route_schema: RouteCoordinateSchema
    seeds: Mapping[str, int]
    notes: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "model_id": self.model_id,
            "checkpoint_hash": self.checkpoint_hash,
            "model_definition_hash": self.model_definition_hash,
            "target_ontology": self.target_ontology,
            "target_ids_hash": self.target_ids_hash,
            "class_map_hash": self.class_map_hash,
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
            ("target_ontology", self.target_ontology),
            ("target_ids_hash", self.target_ids_hash),
            ("class_map_hash", self.class_map_hash),
        ):
            _nonempty(value, name, issues)
        if not isinstance(self.preprocessing, Mapping) or not self.preprocessing:
            issues.append("preprocessing must be a nonempty mapping")
        if not isinstance(self.data_source, Mapping) or not self.data_source:
            issues.append("data_source must be a nonempty mapping")
        if not self.seeds or any(not isinstance(value, int) for value in self.seeds.values()):
            issues.append("seeds must be a nonempty mapping of integers")
        issues.extend(self.route_schema.validate().issues)
        return ValidationResult(not issues, tuple(issues))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> StudySpec:
        result = cls(
            study_id=str(payload.get("study_id", "")),
            model_id=str(payload.get("model_id", "")),
            checkpoint_hash=str(payload.get("checkpoint_hash", "")),
            model_definition_hash=str(payload.get("model_definition_hash", payload.get("model_hash", ""))),
            target_ontology=str(payload.get("target_ontology", "")),
            target_ids_hash=str(payload.get("target_ids_hash", "")),
            class_map_hash=str(payload.get("class_map_hash", "")),
            preprocessing=_mapping(payload.get("preprocessing", {}), "preprocessing"),
            data_source=_mapping(payload.get("data_source", {}), "data_source"),
            route_schema=RouteCoordinateSchema.from_mapping(_mapping(payload.get("route_schema", {}), "route_schema")),
            seeds={str(key): int(value) for key, value in _mapping(payload.get("seeds", {}), "seeds").items()},
            notes=_mapping(payload.get("notes", {}), "notes"),
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
        )
        result.validate().require_valid()
        return result

    @classmethod
    def load(cls, path: str | Path) -> StudySpec:
        payload = json.loads(Path(path).read_text())
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path} did not contain a StudySpec object")
        return cls.from_mapping(payload)


@dataclass(frozen=True)
class ScorePlan(_ContentAddressed):
    """Frozen score-estimation protocol attached to one study."""

    schema_name: ClassVar[str] = "circuitflow_score_plan"

    plan_id: str
    study_hash: str
    estimator_id: str
    score_view: DataView
    target_evaluator: Mapping[str, Any]
    score_semantics: str
    score_heads: tuple[str, ...]
    random_seed: int
    status: str = "frozen"
    stability_windows: tuple[DataView, ...] = tuple()
    notes: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "study_hash": self.study_hash,
            "estimator_id": self.estimator_id,
            "score_view": self.score_view.to_dict(),
            "target_evaluator": _jsonable(self.target_evaluator),
            "score_semantics": self.score_semantics,
            "score_heads": list(self.score_heads),
            "random_seed": int(self.random_seed),
            "status": self.status,
            "stability_windows": [view.to_dict() for view in self.stability_windows],
            "notes": _jsonable(self.notes),
        }

    def validate(self, *, claim_bearing: bool = False) -> ValidationResult:
        issues: list[str] = []
        for name, value in (
            ("plan_id", self.plan_id),
            ("study_hash", self.study_hash),
            ("estimator_id", self.estimator_id),
            ("score_semantics", self.score_semantics),
        ):
            _nonempty(value, name, issues)
        if not self.score_heads:
            issues.append("score_heads must be nonempty")
        if not isinstance(self.target_evaluator, Mapping) or not self.target_evaluator:
            issues.append("target_evaluator must be nonempty")
        if self.status not in PLAN_STATUSES:
            issues.append("unsupported plan status")
        if claim_bearing and self.status != "frozen":
            issues.append("claim-bearing scoring requires status=frozen")
        issues.extend(self.score_view.validate().issues)
        for window in self.stability_windows:
            issues.extend(window.validate().issues)
        return ValidationResult(not issues, tuple(issues))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ScorePlan:
        result = cls(
            plan_id=str(payload.get("plan_id", "")),
            study_hash=str(payload.get("study_hash", "")),
            estimator_id=str(payload.get("estimator_id", "")),
            score_view=DataView.from_mapping(_mapping(payload.get("score_view", {}), "score_view")),
            target_evaluator=_mapping(payload.get("target_evaluator", {}), "target_evaluator"),
            score_semantics=str(payload.get("score_semantics", "")),
            score_heads=tuple(str(item) for item in payload.get("score_heads", ()) or ()),
            random_seed=int(payload.get("random_seed", 0)),
            status=str(payload.get("status", "frozen")),
            stability_windows=tuple(
                DataView.from_mapping(_mapping(item, "stability_window"))
                for item in payload.get("stability_windows", ()) or ()
            ),
            notes=_mapping(payload.get("notes", {}), "notes"),
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
        )
        result.validate(claim_bearing=result.status == "frozen").require_valid()
        return result

    @classmethod
    def load(cls, path: str | Path) -> ScorePlan:
        payload = json.loads(Path(path).read_text())
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path} did not contain a ScorePlan object")
        return cls.from_mapping(payload)


@dataclass(frozen=True)
class CircuitPlan(_ContentAddressed):
    """Frozen route-realization plan for classic or Thermobayes engines."""

    schema_name: ClassVar[str] = "circuitflow_circuit_plan"

    plan_id: str
    study_hash: str
    engine_id: str
    measurement_artifact_hash: str
    route_universe_hash: str
    optimization_view: DataView
    evaluation_view: DataView
    mask_budget: Mapping[str, Any]
    partition_policy: str
    matched_controls: tuple[str, ...]
    random_seed: int
    status: str = "frozen"
    ensemble_policy: Mapping[str, Any] = field(default_factory=dict)
    engine_config: Mapping[str, Any] = field(default_factory=dict)
    notes: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "study_hash": self.study_hash,
            "engine_id": self.engine_id,
            "measurement_artifact_hash": self.measurement_artifact_hash,
            "route_universe_hash": self.route_universe_hash,
            "optimization_view": self.optimization_view.to_dict(),
            "evaluation_view": self.evaluation_view.to_dict(),
            "mask_budget": _jsonable(self.mask_budget),
            "partition_policy": self.partition_policy,
            "matched_controls": list(self.matched_controls),
            "random_seed": int(self.random_seed),
            "status": self.status,
            "ensemble_policy": _jsonable(self.ensemble_policy),
            "engine_config": _jsonable(self.engine_config),
            "notes": _jsonable(self.notes),
        }

    def validate(self, *, claim_bearing: bool = False) -> ValidationResult:
        issues: list[str] = []
        for name, value in (
            ("plan_id", self.plan_id),
            ("study_hash", self.study_hash),
            ("measurement_artifact_hash", self.measurement_artifact_hash),
            ("route_universe_hash", self.route_universe_hash),
            ("partition_policy", self.partition_policy),
        ):
            _nonempty(value, name, issues)
        if self.engine_id not in ENGINE_IDS:
            issues.append(f"unsupported engine_id {self.engine_id!r}")
        if not isinstance(self.mask_budget, Mapping) or not self.mask_budget:
            issues.append("mask_budget must be a nonempty mapping")
        if not self.matched_controls:
            issues.append("matched_controls must be nonempty")
        if self.status not in PLAN_STATUSES:
            issues.append("unsupported plan status")
        if claim_bearing and self.status != "frozen":
            issues.append("claim-bearing circuit realization requires status=frozen")
        if claim_bearing and self.optimization_view.ids_hash == self.evaluation_view.ids_hash:
            issues.append("claim-bearing CircuitPlan requires held-out evaluation_view")
        issues.extend(self.optimization_view.validate().issues)
        issues.extend(self.evaluation_view.validate().issues)
        return ValidationResult(not issues, tuple(issues))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> CircuitPlan:
        result = cls(
            plan_id=str(payload.get("plan_id", "")),
            study_hash=str(payload.get("study_hash", "")),
            engine_id=str(payload.get("engine_id", "")),
            measurement_artifact_hash=str(payload.get("measurement_artifact_hash", "")),
            route_universe_hash=str(payload.get("route_universe_hash", "")),
            optimization_view=DataView.from_mapping(_mapping(payload.get("optimization_view", {}), "optimization_view")),
            evaluation_view=DataView.from_mapping(_mapping(payload.get("evaluation_view", {}), "evaluation_view")),
            mask_budget=_mapping(payload.get("mask_budget", {}), "mask_budget"),
            partition_policy=str(payload.get("partition_policy", "")),
            matched_controls=tuple(str(item) for item in payload.get("matched_controls", ()) or ()),
            random_seed=int(payload.get("random_seed", 0)),
            status=str(payload.get("status", "frozen")),
            ensemble_policy=_mapping(payload.get("ensemble_policy", {}), "ensemble_policy"),
            engine_config=_mapping(payload.get("engine_config", {}), "engine_config"),
            notes=_mapping(payload.get("notes", {}), "notes"),
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
        )
        result.validate(claim_bearing=result.status == "frozen").require_valid()
        return result

    @classmethod
    def load(cls, path: str | Path) -> CircuitPlan:
        payload = json.loads(Path(path).read_text())
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path} did not contain a CircuitPlan object")
        return cls.from_mapping(payload)


@dataclass(frozen=True)
class MechanismPlan(_ContentAddressed):
    """Frozen construction of a state or executed mechanism response basis."""

    schema_name: ClassVar[str] = "circuitflow_mechanism_plan"

    plan_id: str
    study_hash: str
    basis_kind: str
    parent_artifact_hashes: tuple[str, ...]
    stimulus_view: DataView
    execution_family: str
    query_axis_hash: str
    role_ids: tuple[str, ...]
    layer_ids: tuple[str, ...]
    pooling_policy: str
    storage_precision: str
    recombination_policy: str
    random_seed: int
    status: str = "frozen"
    notes: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "study_hash": self.study_hash,
            "basis_kind": self.basis_kind,
            "parent_artifact_hashes": list(self.parent_artifact_hashes),
            "stimulus_view": self.stimulus_view.to_dict(),
            "execution_family": self.execution_family,
            "query_axis_hash": self.query_axis_hash,
            "role_ids": list(self.role_ids),
            "layer_ids": list(self.layer_ids),
            "pooling_policy": self.pooling_policy,
            "storage_precision": self.storage_precision,
            "recombination_policy": self.recombination_policy,
            "random_seed": int(self.random_seed),
            "status": self.status,
            "notes": _jsonable(self.notes),
        }

    def validate(self, *, claim_bearing: bool = False) -> ValidationResult:
        issues: list[str] = []
        for name, value in (
            ("plan_id", self.plan_id),
            ("study_hash", self.study_hash),
            ("execution_family", self.execution_family),
            ("pooling_policy", self.pooling_policy),
            ("storage_precision", self.storage_precision),
            ("recombination_policy", self.recombination_policy),
        ):
            _nonempty(value, name, issues)
        if self.basis_kind not in RESPONSE_BASIS_KINDS:
            issues.append("unsupported basis_kind")
        if self.execution_family not in MECHANISM_EXECUTION_FAMILIES:
            issues.append("unsupported execution_family")
        if self.basis_kind == "state" and self.execution_family != "full_state":
            issues.append("state basis requires execution_family=full_state")
        if self.basis_kind == "state" and claim_bearing:
            family = str(self.notes.get("state_baseline_family", ""))
            scope = str(self.notes.get("state_baseline_scope", "model"))
            if family not in CANONICAL_STATE_BASELINE_FAMILIES:
                issues.append("frozen state basis requires a canonical state_baseline_family")
            if scope not in STATE_BASELINE_SCOPES:
                issues.append("unsupported state_baseline_scope")
        if self.basis_kind == "mechanism":
            if not self.parent_artifact_hashes:
                issues.append("mechanism basis requires parent circuit artifacts")
            _nonempty(self.query_axis_hash, "query_axis_hash", issues)
        if self.execution_family in {
            "local_signed_use",
            "local_additive_use",
            "route_coordinate_signed_use",
        }:
            if self.role_ids != ("positive", "negative", "unresolved", "aggregate"):
                issues.append("local additive use requires canonical signed roles")
            if self.recombination_policy != "linear_preadd_exact":
                issues.append("local additive use requires exact preadd recombination")
        if self.execution_family in {
            "endogenous_factorized_use",
            "endogenous_route_coordinate_use",
        }:
            if self.role_ids != ("positive", "negative", "unresolved", "aggregate"):
                issues.append("endogenous factorized use requires canonical signed roles")
            if self.recombination_policy != "masked_trajectory_linear_preadd_exact":
                issues.append(
                    "endogenous factorized use requires masked-trajectory exact preadd recombination"
                )
        if self.execution_family == "role_counterfactual":
            if self.role_ids != (
                "baseline",
                "positive_only",
                "negative_only",
                "aggregate",
                "interaction",
            ):
                issues.append("role counterfactual requires canonical nonlinear roles")
            if self.recombination_policy != "nonlinear_inclusion_exclusion":
                issues.append("role counterfactual requires nonlinear inclusion-exclusion")
        if self.execution_family == "interaction" and self.recombination_policy != "two_way_residual":
            issues.append("interaction execution requires two-way residual recombination")
        if not self.role_ids or not self.layer_ids:
            issues.append("role_ids and layer_ids must be nonempty")
        if self.status not in PLAN_STATUSES:
            issues.append("unsupported plan status")
        if claim_bearing and self.status != "frozen":
            issues.append("claim-bearing mechanism response requires status=frozen")
        issues.extend(self.stimulus_view.validate().issues)
        return ValidationResult(not issues, tuple(issues))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> MechanismPlan:
        result = cls(
            plan_id=str(payload.get("plan_id", "")),
            study_hash=str(payload.get("study_hash", "")),
            basis_kind=str(payload.get("basis_kind", "mechanism")),
            parent_artifact_hashes=tuple(str(value) for value in payload.get("parent_artifact_hashes", ()) or ()),
            stimulus_view=DataView.from_mapping(_mapping(payload.get("stimulus_view", {}), "stimulus_view")),
            execution_family=str(payload.get("execution_family", "")),
            query_axis_hash=str(payload.get("query_axis_hash", "")),
            role_ids=tuple(str(value) for value in payload.get("role_ids", ()) or ()),
            layer_ids=tuple(str(value) for value in payload.get("layer_ids", ()) or ()),
            pooling_policy=str(payload.get("pooling_policy", "")),
            storage_precision=str(payload.get("storage_precision", "")),
            recombination_policy=str(payload.get("recombination_policy", "")),
            random_seed=int(payload.get("random_seed", 0)),
            status=str(payload.get("status", "frozen")),
            notes=_mapping(payload.get("notes", {}), "notes"),
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
        )
        result.validate(claim_bearing=result.status == "frozen").require_valid()
        return result

    @classmethod
    def load(cls, path: str | Path) -> MechanismPlan:
        payload = json.loads(Path(path).read_text())
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path} did not contain a MechanismPlan object")
        return cls.from_mapping(payload)


@dataclass(frozen=True)
class KernelContractionSpec:
    """One coordinate-aware contraction of a response basis into a kernel."""

    spec_id: str
    comparison_axis: str
    integrated_axes: tuple[str, ...]
    role_block: str
    layer_aggregation: str
    layer_weights: Mapping[str, float]
    metric: str
    centering: str
    normalization: str
    contraction_type: str
    response_transform: str = "identity"
    group_weight_policy: str = "equal"
    transport_map_hash: str = ""
    source_object: str = "response_basis"
    primitive_retention: str = "aggregate_only"
    notes: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "spec_id": self.spec_id,
            "comparison_axis": self.comparison_axis,
            "integrated_axes": list(self.integrated_axes),
            "role_block": self.role_block,
            "layer_aggregation": self.layer_aggregation,
            "layer_weights": {str(key): float(value) for key, value in self.layer_weights.items()},
            "metric": self.metric,
            "centering": self.centering,
            "normalization": self.normalization,
            "contraction_type": self.contraction_type,
            "response_transform": self.response_transform,
            "group_weight_policy": self.group_weight_policy,
            "transport_map_hash": self.transport_map_hash,
            "notes": _jsonable(self.notes),
        }
        # ``response_basis`` was the only source in CircuitFlow v2 before
        # source-typed contractions. Keep its wire representation stable so
        # existing response plans and artifacts retain their content hashes.
        if self.source_object != "response_basis":
            payload["source_object"] = self.source_object
        # Primitive retention postdates the v2 wire contract. Preserve all
        # historical hashes when the backward-compatible default is used.
        if self.primitive_retention != "aggregate_only":
            payload["primitive_retention"] = self.primitive_retention
        return payload

    @property
    def spec_hash(self) -> str:
        return stable_digest(self.to_dict())

    def validate(self) -> ValidationResult:
        issues: list[str] = []
        for name, value in (
            ("spec_id", self.spec_id),
            ("role_block", self.role_block),
            ("layer_aggregation", self.layer_aggregation),
            ("metric", self.metric),
            ("centering", self.centering),
            ("normalization", self.normalization),
            ("group_weight_policy", self.group_weight_policy),
        ):
            _nonempty(value, name, issues)
        if self.comparison_axis not in KERNEL_COMPARISON_AXES:
            issues.append("unsupported comparison_axis")
        if self.source_object not in KERNEL_SOURCE_OBJECTS:
            issues.append("unsupported source_object")
        if self.contraction_type not in KERNEL_CONTRACTION_TYPES:
            issues.append("unsupported contraction_type")
        if self.contraction_type == "group_diagonal" and self.role_block in {
            "opposite_sign",
            "positive_negative",
            "negative_positive",
            "opponent",
            "opponent_positive_minus_negative",
        }:
            issues.append("group_diagonal requires a positive-semidefinite role block")
        if self.response_transform not in RESPONSE_TRANSFORMS:
            issues.append("unsupported response_transform")
        if self.metric not in KERNEL_METRICS:
            issues.append("unsupported kernel metric")
        elif self.source_object == "circuit_anatomy" and self.metric not in STRUCTURAL_KERNEL_METRICS:
            issues.append("circuit anatomy requires a structural kernel metric")
        elif self.source_object != "circuit_anatomy" and self.metric not in RESPONSE_KERNEL_METRICS:
            issues.append("response and state contractions require a response kernel metric")
        if self.centering not in KERNEL_CENTERING_POLICIES:
            issues.append("unsupported kernel centering policy")
        if self.normalization not in KERNEL_NORMALIZATION_POLICIES:
            issues.append("unsupported kernel normalization policy")
        if self.layer_aggregation not in LAYER_AGGREGATION_POLICIES:
            issues.append("unsupported layer aggregation policy")
        if self.primitive_retention not in KERNEL_PRIMITIVE_RETENTION_POLICIES:
            issues.append("unsupported kernel primitive retention policy")
        if self.comparison_axis == "interaction" and self.response_transform != "two_way_interaction":
            issues.append("interaction comparison requires response_transform=two_way_interaction")
        expected_axes = _INTEGRATED_AXES_BY_SOURCE_COMPARISON.get(
            (self.source_object, self.comparison_axis),
            _INTEGRATED_AXES_BY_COMPARISON.get(self.comparison_axis),
        )
        if not self.integrated_axes:
            issues.append("integrated_axes must be nonempty")
        elif expected_axes is not None:
            observed = frozenset(str(value) for value in self.integrated_axes)
            if len(observed) != len(self.integrated_axes) or observed != expected_axes:
                issues.append(
                    "integrated_axes must exactly match the declared source and comparison axis: "
                    + ",".join(sorted(expected_axes))
                )
        if self.layer_aggregation == "custom":
            weights = tuple(float(value) for value in self.layer_weights.values())
            if not weights or any(not math.isfinite(value) or value < 0 for value in weights):
                issues.append("custom layer_weights must be finite and nonnegative")
            elif not math.isclose(sum(weights), 1.0, rel_tol=1e-6, abs_tol=1e-6):
                issues.append("custom layer_weights must sum to one")
        if self.notes.get("cross_layer_inner_product") and not self.transport_map_hash:
            issues.append("cross-layer inner products require transport_map_hash")
        return ValidationResult(not issues, tuple(issues))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> KernelContractionSpec:
        result = cls(
            spec_id=str(payload.get("spec_id", "")),
            comparison_axis=str(payload.get("comparison_axis", "")),
            integrated_axes=tuple(str(value) for value in payload.get("integrated_axes", ()) or ()),
            role_block=str(payload.get("role_block", "all")),
            layer_aggregation=str(payload.get("layer_aggregation", "equal")),
            layer_weights={str(key): float(value) for key, value in _mapping(payload.get("layer_weights", {}), "layer_weights").items()},
            metric=str(payload.get("metric", "cosine")),
            centering=str(payload.get("centering", "none")),
            normalization=str(payload.get("normalization", "l2")),
            contraction_type=str(payload.get("contraction_type", "trace")),
            response_transform=str(payload.get("response_transform", "identity")),
            group_weight_policy=str(payload.get("group_weight_policy", "equal")),
            transport_map_hash=str(payload.get("transport_map_hash", "")),
            source_object=str(payload.get("source_object", "response_basis")),
            primitive_retention=str(
                payload.get("primitive_retention", "aggregate_only")
            ),
            notes=_mapping(payload.get("notes", {}), "notes"),
        )
        result.validate().require_valid()
        return result


@dataclass(frozen=True)
class GeometryPlan(_ContentAddressed):
    """Frozen kernel contractions over canonical circuit or response artifacts."""

    schema_name: ClassVar[str] = "circuitflow_geometry_plan"

    plan_id: str
    study_hashes: tuple[str, ...]
    source_artifact_hashes: tuple[str, ...]
    contractions: tuple[KernelContractionSpec, ...]
    status: str = "frozen"
    notes: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "study_hashes": list(self.study_hashes),
            "source_artifact_hashes": list(self.source_artifact_hashes),
            "contractions": [item.to_dict() for item in self.contractions],
            "status": self.status,
            "notes": _jsonable(self.notes),
        }

    def validate(self, *, claim_bearing: bool = False) -> ValidationResult:
        issues: list[str] = []
        _nonempty(self.plan_id, "plan_id", issues)
        if not self.study_hashes or not self.source_artifact_hashes or not self.contractions:
            issues.append("study_hashes, source_artifact_hashes, and contractions must be nonempty")
        if len({item.spec_id for item in self.contractions}) != len(self.contractions):
            issues.append("contraction spec_ids must be unique")
        for item in self.contractions:
            issues.extend(item.validate().issues)
        if self.status not in PLAN_STATUSES:
            issues.append("unsupported plan status")
        if claim_bearing and self.status != "frozen":
            issues.append("claim-bearing geometry requires status=frozen")
        return ValidationResult(not issues, tuple(issues))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> GeometryPlan:
        result = cls(
            plan_id=str(payload.get("plan_id", "")),
            study_hashes=tuple(str(value) for value in payload.get("study_hashes", ()) or ()),
            source_artifact_hashes=tuple(str(value) for value in payload.get("source_artifact_hashes", ()) or ()),
            contractions=tuple(KernelContractionSpec.from_mapping(item) for item in payload.get("contractions", ()) or ()),
            status=str(payload.get("status", "frozen")),
            notes=_mapping(payload.get("notes", {}), "notes"),
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
        )
        result.validate(claim_bearing=result.status == "frozen").require_valid()
        return result

    @classmethod
    def load(cls, path: str | Path) -> GeometryPlan:
        payload = json.loads(Path(path).read_text())
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path} did not contain a GeometryPlan object")
        return cls.from_mapping(payload)


@dataclass(frozen=True)
class ActionPlan(_ContentAddressed):
    """Prospective intervention plan attached to a circuit realization."""

    schema_name: ClassVar[str] = "circuitflow_action_plan"

    plan_id: str
    study_hash: str
    circuit_artifact_hash: str
    phenotype_family: str
    action_family: str
    doses: tuple[float, ...]
    matched_controls: tuple[str, ...]
    evaluation_view: DataView
    random_seed: int
    status: str = "frozen"
    notes: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "study_hash": self.study_hash,
            "circuit_artifact_hash": self.circuit_artifact_hash,
            "phenotype_family": self.phenotype_family,
            "action_family": self.action_family,
            "doses": list(self.doses),
            "matched_controls": list(self.matched_controls),
            "evaluation_view": self.evaluation_view.to_dict(),
            "random_seed": int(self.random_seed),
            "status": self.status,
            "notes": _jsonable(self.notes),
        }

    def validate(self, *, claim_bearing: bool = False) -> ValidationResult:
        issues: list[str] = []
        for name, value in (
            ("plan_id", self.plan_id),
            ("study_hash", self.study_hash),
            ("circuit_artifact_hash", self.circuit_artifact_hash),
            ("phenotype_family", self.phenotype_family),
            ("action_family", self.action_family),
        ):
            _nonempty(value, name, issues)
        if not self.doses or any(not math.isfinite(float(dose)) or float(dose) <= 0.0 for dose in self.doses):
            issues.append("doses must be finite positive values")
        if not self.matched_controls:
            issues.append("matched_controls must be nonempty")
        if self.status not in PLAN_STATUSES:
            issues.append("unsupported plan status")
        if claim_bearing and self.status != "frozen":
            issues.append("claim-bearing interventions require a frozen ActionPlan")
        issues.extend(self.evaluation_view.validate().issues)
        return ValidationResult(not issues, tuple(issues))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ActionPlan:
        result = cls(
            plan_id=str(payload.get("plan_id", "")),
            study_hash=str(payload.get("study_hash", "")),
            circuit_artifact_hash=str(payload.get("circuit_artifact_hash", "")),
            phenotype_family=str(payload.get("phenotype_family", "")),
            action_family=str(payload.get("action_family", "")),
            doses=tuple(float(value) for value in payload.get("doses", ()) or ()),
            matched_controls=tuple(str(value) for value in payload.get("matched_controls", ()) or ()),
            evaluation_view=DataView.from_mapping(_mapping(payload.get("evaluation_view", {}), "evaluation_view")),
            random_seed=int(payload.get("random_seed", 0)),
            status=str(payload.get("status", "frozen")),
            notes=_mapping(payload.get("notes", {}), "notes"),
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
        )
        result.validate(claim_bearing=result.status == "frozen").require_valid()
        return result

    @classmethod
    def load(cls, path: str | Path) -> ActionPlan:
        payload = json.loads(Path(path).read_text())
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path} did not contain an ActionPlan object")
        return cls.from_mapping(payload)


@dataclass(frozen=True)
class AnalysisPlan(_ContentAddressed):
    """Frozen inferential plan that consumes canonical artifact hashes only."""

    schema_name: ClassVar[str] = "circuitflow_analysis_plan"

    plan_id: str
    study_hashes: tuple[str, ...]
    input_artifact_hashes: tuple[str, ...]
    hypothesis: str
    estimand: str
    target_geometry: Mapping[str, Any]
    fold_strategy: str
    fold_assignments_hash: str
    controls: tuple[str, ...]
    outcomes: tuple[str, ...]
    multiplicity_family: str
    random_seed: int
    status: str = "frozen"
    selection_policy: Mapping[str, Any] = field(default_factory=dict)
    uncertainty_policy: Mapping[str, Any] = field(default_factory=dict)
    equivalence_policy: Mapping[str, Any] = field(default_factory=dict)
    notes: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "study_hashes": list(self.study_hashes),
            "input_artifact_hashes": list(self.input_artifact_hashes),
            "hypothesis": self.hypothesis,
            "estimand": self.estimand,
            "target_geometry": _jsonable(self.target_geometry),
            "fold_strategy": self.fold_strategy,
            "fold_assignments_hash": self.fold_assignments_hash,
            "controls": list(self.controls),
            "outcomes": list(self.outcomes),
            "multiplicity_family": self.multiplicity_family,
            "random_seed": int(self.random_seed),
            "status": self.status,
            "notes": _jsonable(self.notes),
        }
        if self.selection_policy:
            payload["selection_policy"] = _jsonable(self.selection_policy)
        if self.uncertainty_policy:
            payload["uncertainty_policy"] = _jsonable(self.uncertainty_policy)
        if self.equivalence_policy:
            payload["equivalence_policy"] = _jsonable(self.equivalence_policy)
        return payload

    def validate(self, *, claim_bearing: bool = False) -> ValidationResult:
        issues: list[str] = []
        for name, value in (
            ("plan_id", self.plan_id),
            ("hypothesis", self.hypothesis),
            ("estimand", self.estimand),
            ("fold_strategy", self.fold_strategy),
            ("fold_assignments_hash", self.fold_assignments_hash),
            ("multiplicity_family", self.multiplicity_family),
        ):
            _nonempty(value, name, issues)
        if not self.study_hashes:
            issues.append("study_hashes must be nonempty")
        if not self.input_artifact_hashes:
            issues.append("input_artifact_hashes must be nonempty")
        if not isinstance(self.target_geometry, Mapping) or not self.target_geometry:
            issues.append("target_geometry must be nonempty")
        if not self.controls or not self.outcomes:
            issues.append("controls and outcomes must be nonempty")
        if self.status not in PLAN_STATUSES:
            issues.append("unsupported plan status")
        if claim_bearing and self.status != "frozen":
            issues.append("claim-bearing analysis requires status=frozen")
        if self.selection_policy:
            if str(self.selection_policy.get("outer_folds", "")) != "frozen_assignment":
                issues.append("selection_policy requires frozen outer assignments")
            if str(self.selection_policy.get("inner_folds", "")) != "training_only_blocked_axis":
                issues.append("selection_policy requires training-only blocked inner folds")
            forbidden = {"heldout_outcomes", "heldout_neural_responses", "causal_outcomes"}
            if forbidden.intersection(self.selection_policy):
                issues.append("selection_policy contains forbidden held-out inputs")
        if self.uncertainty_policy:
            units = tuple(str(value) for value in self.uncertainty_policy.get("resampling_units", ()) or ())
            if not units or "rdm_pair" in units:
                issues.append("uncertainty_policy must resample independent units, never RDM pairs")
        if self.equivalence_policy:
            try:
                alignment = float(self.equivalence_policy.get("alignment_margin"))
                projector = float(self.equivalence_policy.get("projector_margin"))
            except (TypeError, ValueError):
                alignment, projector = 0.0, -1.0
            if not 0.0 < alignment <= 1.0 or not 0.0 <= projector < 1.0:
                issues.append("equivalence_policy margins are invalid")
        return ValidationResult(not issues, tuple(issues))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> AnalysisPlan:
        result = cls(
            plan_id=str(payload.get("plan_id", "")),
            study_hashes=tuple(str(value) for value in payload.get("study_hashes", ()) or ()),
            input_artifact_hashes=tuple(str(value) for value in payload.get("input_artifact_hashes", ()) or ()),
            hypothesis=str(payload.get("hypothesis", "")),
            estimand=str(payload.get("estimand", "")),
            target_geometry=_mapping(payload.get("target_geometry", {}), "target_geometry"),
            fold_strategy=str(payload.get("fold_strategy", "")),
            fold_assignments_hash=str(payload.get("fold_assignments_hash", "")),
            controls=tuple(str(value) for value in payload.get("controls", ()) or ()),
            outcomes=tuple(str(value) for value in payload.get("outcomes", ()) or ()),
            multiplicity_family=str(payload.get("multiplicity_family", "")),
            random_seed=int(payload.get("random_seed", 0)),
            status=str(payload.get("status", "frozen")),
            selection_policy=_mapping(payload.get("selection_policy", {}), "selection_policy"),
            uncertainty_policy=_mapping(payload.get("uncertainty_policy", {}), "uncertainty_policy"),
            equivalence_policy=_mapping(payload.get("equivalence_policy", {}), "equivalence_policy"),
            notes=_mapping(payload.get("notes", {}), "notes"),
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
        )
        result.validate(claim_bearing=result.status == "frozen").require_valid()
        return result

    @classmethod
    def load(cls, path: str | Path) -> AnalysisPlan:
        payload = json.loads(Path(path).read_text())
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path} did not contain an AnalysisPlan object")
        return cls.from_mapping(payload)


@dataclass(frozen=True)
class RuntimePlan(_ContentAddressed):
    """Non-scientific execution policy for a typed work packet."""

    schema_name: ClassVar[str] = "circuitflow_runtime_plan"

    plan_id: str
    job_kind: str
    partition_preference: tuple[str, ...] = ("kempner_h200", "kempner_h100", "kempner")
    cpus_per_gpu: int = 16
    memory_gb_per_gpu: float = 360.0
    walltime_minutes: float = 480.0
    max_active_gpus: int = 16
    cache_policy: Mapping[str, Any] = field(default_factory=dict)
    batch_policy: Mapping[str, Any] = field(default_factory=dict)
    log_policy: Mapping[str, Any] = field(default_factory=dict)
    replay_policy: Mapping[str, Any] = field(default_factory=dict)
    retry_policy: Mapping[str, Any] = field(default_factory=dict)
    notes: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "job_kind": self.job_kind,
            "partition_preference": list(self.partition_preference),
            "cpus_per_gpu": int(self.cpus_per_gpu),
            "memory_gb_per_gpu": float(self.memory_gb_per_gpu),
            "walltime_minutes": float(self.walltime_minutes),
            "max_active_gpus": int(self.max_active_gpus),
            "cache_policy": _jsonable(self.cache_policy),
            "batch_policy": _jsonable(self.batch_policy),
            "log_policy": _jsonable(self.log_policy),
            "replay_policy": _jsonable(self.replay_policy),
            "retry_policy": _jsonable(self.retry_policy),
            "notes": _jsonable(self.notes),
        }

    def validate(self) -> ValidationResult:
        issues: list[str] = []
        _nonempty(self.plan_id, "plan_id", issues)
        _nonempty(self.job_kind, "job_kind", issues)
        if not self.partition_preference or any(partition not in VALID_PARTITIONS for partition in self.partition_preference):
            issues.append("partition_preference contains unsupported partitions")
        if self.cpus_per_gpu <= 0 or self.memory_gb_per_gpu <= 0 or self.walltime_minutes <= 0:
            issues.append("resource requests must be positive")
        if not 1 <= self.max_active_gpus <= 16:
            issues.append("max_active_gpus must be between 1 and 16")
        initial = self.batch_policy.get("initial_batch_size") if isinstance(self.batch_policy, Mapping) else None
        ceiling = self.batch_policy.get("max_probe_batch_size") if isinstance(self.batch_policy, Mapping) else None
        if initial is not None and ceiling is not None and int(initial) > int(ceiling):
            issues.append("initial_batch_size exceeds max_probe_batch_size")
        return ValidationResult(not issues, tuple(issues))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> RuntimePlan:
        result = cls(
            plan_id=str(payload.get("plan_id", "")),
            job_kind=str(payload.get("job_kind", "")),
            partition_preference=tuple(
                str(value) for value in payload.get("partition_preference", ("kempner_h200", "kempner_h100", "kempner")) or ()
            ),
            cpus_per_gpu=int(payload.get("cpus_per_gpu", 16)),
            memory_gb_per_gpu=float(payload.get("memory_gb_per_gpu", 360.0)),
            walltime_minutes=float(payload.get("walltime_minutes", 480.0)),
            max_active_gpus=int(payload.get("max_active_gpus", 16)),
            cache_policy=_mapping(payload.get("cache_policy", {}), "cache_policy"),
            batch_policy=_mapping(payload.get("batch_policy", {}), "batch_policy"),
            log_policy=_mapping(payload.get("log_policy", {}), "log_policy"),
            replay_policy=_mapping(payload.get("replay_policy", {}), "replay_policy"),
            retry_policy=_mapping(payload.get("retry_policy", {}), "retry_policy"),
            notes=_mapping(payload.get("notes", {}), "notes"),
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
        )
        result.validate().require_valid()
        return result

    @classmethod
    def load(cls, path: str | Path) -> RuntimePlan:
        payload = json.loads(Path(path).read_text())
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path} did not contain a RuntimePlan object")
        return cls.from_mapping(payload)


def assert_compatible_parent_studies(*plans: _ContentAddressed, expected_study_hash: str) -> None:
    """Fail closed when a plan references a different scientific study."""

    for plan in plans:
        study_hash = getattr(plan, "study_hash", "")
        if study_hash and str(study_hash) != str(expected_study_hash):
            raise ValueError(
                f"{type(plan).__name__} references study {study_hash}, expected {expected_study_hash}"
            )


__all__ = [
    "ActionPlan",
    "AnalysisPlan",
    "CircuitPlan",
    "DataView",
    "ENGINE_IDS",
    "PLAN_STATUSES",
    "RouteCoordinateSchema",
    "RuntimePlan",
    "SCHEMA_VERSION",
    "ScorePlan",
    "StudySpec",
    "ValidationResult",
    "assert_compatible_parent_studies",
]
