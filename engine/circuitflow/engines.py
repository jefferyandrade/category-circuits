"""Common circuit-engine interface for classic pruning and Thermobayes.

The engines do not share an optimizer or posterior.  They share the public
meaning of a circuit realization: a point mask or a weighted collection of
mask states over the same stable route-coordinate schema.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .artifacts import ArtifactManifest, ArtifactType, PayloadRef
from .contracts import CircuitPlan, RuntimePlan

THERMOBAYES_PROGRAM_CLAIM_BINDING_KEYS = (
    "program_evidence_hash",
    "program_posterior_hash",
    "program_model_hash",
    "program_checkpoint_hash",
    "program_route_universe_hash",
    "program_path_law_hash",
    "program_ensemble_hash",
    "program_claim_decision_hash",
)


def validate_thermobayes_program_claim_bindings(
    plan: CircuitPlan,
    *,
    expected: Mapping[str, str],
) -> Mapping[str, str]:
    """Fail closed unless a frozen plan pins the exact program authority chain.

    A CircuitFlow plan is prospective scientific identity, not authority merely
    because it is frozen.  Claim-bearing Thermobayes imports therefore require
    the plan to name every upstream semantic object used by the final ensemble.
    """

    if plan.engine_id != ThermobayesProgramEngine.engine_id:
        raise ValueError("program claim bindings require a thermobayes_program CircuitPlan")
    required = set(THERMOBAYES_PROGRAM_CLAIM_BINDING_KEYS)
    if set(expected) != required:
        missing = sorted(required - set(expected))
        extra = sorted(set(expected) - required)
        raise ValueError(
            "Thermobayes expected claim binding keys are not exact: "
            f"missing={missing}, extra={extra}"
        )
    resolved: dict[str, str] = {}
    for key in THERMOBAYES_PROGRAM_CLAIM_BINDING_KEYS:
        declared = str(plan.engine_config.get(key, ""))
        authoritative = str(expected[key])
        if not declared:
            raise ValueError(f"Thermobayes claim-bearing CircuitPlan is missing engine_config.{key}")
        if declared != authoritative:
            raise ValueError(f"Thermobayes claim-bearing CircuitPlan engine_config.{key} mismatch")
        resolved[key] = declared
    if resolved["program_route_universe_hash"] != plan.route_universe_hash:
        raise ValueError("Thermobayes program route universe does not match CircuitPlan")
    return resolved


@dataclass(frozen=True)
class CircuitRealization:
    """A route realization usable by all downstream geometry producers."""

    realization_id: str
    realization_kind: str
    weight: float
    mask_payload_name: str
    route_universe_hash: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "realization_id": self.realization_id,
            "realization_kind": self.realization_kind,
            "weight": float(self.weight),
            "mask_payload_name": self.mask_payload_name,
            "route_universe_hash": self.route_universe_hash,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EngineCompilation:
    """Typed worker description; submission code may render it for Slurm."""

    engine_id: str
    circuit_plan_hash: str
    measurement_artifact_hash: str
    route_universe_hash: str
    work_units: tuple[str, ...]
    runtime_plan_hash: str
    worker_arguments: Mapping[str, Any]
    required_outputs: tuple[str, ...]
    preflight: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "circuit_plan_hash": self.circuit_plan_hash,
            "measurement_artifact_hash": self.measurement_artifact_hash,
            "route_universe_hash": self.route_universe_hash,
            "work_units": list(self.work_units),
            "runtime_plan_hash": self.runtime_plan_hash,
            "worker_arguments": dict(self.worker_arguments),
            "required_outputs": list(self.required_outputs),
            "preflight": dict(self.preflight),
        }


class CircuitEngine(Protocol):
    engine_id: str

    def plan(self, plan: CircuitPlan) -> Mapping[str, Any]:
        ...

    def compile(
        self,
        plan: CircuitPlan,
        runtime: RuntimePlan,
        *,
        work_units: Sequence[str],
    ) -> EngineCompilation:
        ...

    def execute(self, compilation: EngineCompilation) -> Mapping[str, Any]:
        ...

    def validate(self, manifest: ArtifactManifest, *, claim_bearing: bool = False) -> tuple[str, ...]:
        ...

    def materialize(
        self,
        plan: CircuitPlan,
        *,
        coordinate_schema_hash: str,
        payloads: Sequence[PayloadRef],
        unit_ids: Sequence[str],
        realizations: Sequence[CircuitRealization],
        valid_for_claims: bool,
        source_kind: str = "native",
        notes: Mapping[str, Any] | None = None,
    ) -> ArtifactManifest:
        ...


def _compile_preflight(runtime: RuntimePlan) -> dict[str, Any]:
    """Declare a safe preflight before a worker touches its large batch path."""

    runtime.validate().require_valid()
    batch = dict(runtime.batch_policy)
    return {
        "cache_policy": dict(runtime.cache_policy),
        "batch_policy": batch,
        "requires_safe_batch_preflight": True,
        "initial_batch_size": batch.get("initial_batch_size", batch.get("safe_batch_size", "")),
        "max_probe_batch_size": batch.get("max_probe_batch_size", batch.get("safe_batch_size", "")),
        "replay_policy": dict(runtime.replay_policy),
        "log_policy": dict(runtime.log_policy),
        "retry_policy": dict(runtime.retry_policy),
    }


class _BaseCircuitEngine:
    realization_kind: str

    def plan(self, plan: CircuitPlan) -> Mapping[str, Any]:
        """Validate and expose engine-neutral circuit semantics."""

        plan.validate(claim_bearing=plan.status == "frozen").require_valid()
        return {
            "engine_id": self.engine_id,
            "circuit_plan_hash": plan.content_hash,
            "measurement_artifact_hash": plan.measurement_artifact_hash,
            "route_universe_hash": plan.route_universe_hash,
            "optimization_view_hash": plan.optimization_view.view_hash,
            "evaluation_view_hash": plan.evaluation_view.view_hash,
            "partition_policy": plan.partition_policy,
            "mask_budget": dict(plan.mask_budget),
        }

    def execute(self, compilation: EngineCompilation) -> Mapping[str, Any]:
        """Return a typed handoff to the tested engine-specific worker."""

        return {
            "engine_id": self.engine_id,
            "execution_status": "worker_handoff_required",
            "worker_arguments": dict(compilation.worker_arguments),
            "required_outputs": list(compilation.required_outputs),
            "preflight": dict(compilation.preflight),
        }

    def validate(self, manifest: ArtifactManifest, *, claim_bearing: bool = False) -> tuple[str, ...]:
        issues = list(manifest.validate(claim_bearing=claim_bearing))
        if manifest.artifact_type != ArtifactType.CIRCUIT:
            issues.append("engine_validation_requires_circuit_artifact")
        if str(manifest.semantics.get("engine_id", "")) != self.engine_id:
            issues.append("circuit_engine_id_mismatch")
        if str(manifest.semantics.get("realization_kind", "")) != self.realization_kind:
            issues.append("circuit_realization_kind_mismatch")
        return tuple(sorted(set(issues)))

    def materialize(
        self,
        plan: CircuitPlan,
        *,
        coordinate_schema_hash: str,
        payloads: Sequence[PayloadRef],
        unit_ids: Sequence[str],
        realizations: Sequence[CircuitRealization],
        valid_for_claims: bool,
        source_kind: str = "native",
        notes: Mapping[str, Any] | None = None,
    ) -> ArtifactManifest:
        plan.validate(claim_bearing=valid_for_claims).require_valid()
        if not realizations:
            raise ValueError("a CircuitArtifact requires at least one realization")
        if self.realization_kind == "point":
            if len(realizations) != 1 or abs(float(realizations[0].weight) - 1.0) > 1e-9:
                raise ValueError("classic point realization requires exactly one weight-1 mask")
        else:
            total_weight = sum(float(item.weight) for item in realizations)
            if total_weight <= 0.0 or abs(total_weight - 1.0) > 1e-6:
                raise ValueError("ensemble realization weights must sum to one")
        if any(item.route_universe_hash != plan.route_universe_hash for item in realizations):
            raise ValueError("realization route universe mismatch")
        return ArtifactManifest(
            artifact_type=ArtifactType.CIRCUIT,
            study_hash=plan.study_hash,
            plan_hash=plan.content_hash,
            parent_hashes=(plan.measurement_artifact_hash,),
            axes={"query": tuple(str(unit) for unit in unit_ids), "route": (plan.route_universe_hash,)},
            coordinate_schema_hash=str(coordinate_schema_hash),
            semantics={
                "engine_id": self.engine_id,
                "realization_kind": self.realization_kind,
                "partition_policy": plan.partition_policy,
                "mask_budget": dict(plan.mask_budget),
                "matched_controls": list(plan.matched_controls),
                "optimization_view_hash": plan.optimization_view.view_hash,
                "evaluation_view_hash": plan.evaluation_view.view_hash,
                "ensemble_policy": dict(plan.ensemble_policy),
                "realizations": [realization.to_dict() for realization in realizations],
            },
            payloads=tuple(payloads),
            unit_ids=tuple(str(unit) for unit in unit_ids),
            source_kind=source_kind,
            valid_for_claims=bool(valid_for_claims),
            notes=dict(notes or {}),
        )


class ClassicFrontierEngine(_BaseCircuitEngine):
    """Adapter around the tested deterministic frontier pruning runtime."""

    engine_id = "classic_frontier"
    realization_kind = "point"

    def compile(
        self,
        plan: CircuitPlan,
        runtime: RuntimePlan,
        *,
        work_units: Sequence[str],
    ) -> EngineCompilation:
        if plan.engine_id != self.engine_id:
            raise ValueError("ClassicFrontierEngine received a non-classic CircuitPlan")
        plan.validate(claim_bearing=plan.status == "frozen").require_valid()
        return EngineCompilation(
            engine_id=self.engine_id,
            circuit_plan_hash=plan.content_hash,
            measurement_artifact_hash=plan.measurement_artifact_hash,
            route_universe_hash=plan.route_universe_hash,
            work_units=tuple(str(unit) for unit in work_units),
            runtime_plan_hash=runtime.content_hash,
            worker_arguments={
                "adapter": "legacy.pipelines.single_layer.session",
                "policy": plan.partition_policy,
                "budget": dict(plan.mask_budget),
                "engine_config": dict(plan.engine_config),
            },
            required_outputs=(
                "mask.pt",
                "metadata.json",
                "metrics.pt",
                "density_summary.json",
                "pruning_manifest.json",
                "unit_completion.json",
                "gpu_performance_receipt.json",
            ),
            preflight=_compile_preflight(runtime),
        )


class ThermobayesProgramEngine(_BaseCircuitEngine):
    """Adapter around program posterior/path-law circuit ensembles."""

    engine_id = "thermobayes_program"
    realization_kind = "ensemble"

    def compile(
        self,
        plan: CircuitPlan,
        runtime: RuntimePlan,
        *,
        work_units: Sequence[str],
    ) -> EngineCompilation:
        if plan.engine_id != self.engine_id:
            raise ValueError("ThermobayesProgramEngine received a non-Thermobayes CircuitPlan")
        plan.validate(claim_bearing=plan.status == "frozen").require_valid()
        return EngineCompilation(
            engine_id=self.engine_id,
            circuit_plan_hash=plan.content_hash,
            measurement_artifact_hash=plan.measurement_artifact_hash,
            route_universe_hash=plan.route_universe_hash,
            work_units=tuple(str(unit) for unit in work_units),
            runtime_plan_hash=runtime.content_hash,
            worker_arguments={
                "adapter": "engines.thermobayes.program",
                "posterior_config": dict(plan.engine_config),
                "ensemble_policy": dict(plan.ensemble_policy),
                "budget": dict(plan.mask_budget),
            },
            required_outputs=(
                "program_circuit_ensemble.json",
                "program_circuit_ensemble.json.manifest.json",
                "gpu_performance_receipt.json",
            ),
            preflight=_compile_preflight(runtime),
        )


def engine_for_plan(plan: CircuitPlan) -> CircuitEngine:
    if plan.engine_id == ClassicFrontierEngine.engine_id:
        return ClassicFrontierEngine()
    if plan.engine_id == ThermobayesProgramEngine.engine_id:
        return ThermobayesProgramEngine()
    raise ValueError(f"unsupported CircuitFlow engine {plan.engine_id!r}")


__all__ = [
    "CircuitEngine",
    "CircuitRealization",
    "ClassicFrontierEngine",
    "EngineCompilation",
    "THERMOBAYES_PROGRAM_CLAIM_BINDING_KEYS",
    "ThermobayesProgramEngine",
    "engine_for_plan",
    "validate_thermobayes_program_claim_bindings",
]
