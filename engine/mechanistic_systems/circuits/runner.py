"""Registered-model execution for canonical rate-distortion identification."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from types import SimpleNamespace

from circuitflow.artifacts import (
    ArtifactStore,
    ArtifactType,
    RuntimeReceipt,
)
from contracts.hashing import stable_digest
from mechanistic_systems.circuits.polarity import (
    TorchStructuralPolarityProvider,
    resolve_residual_route_partition_policy,
)
from mechanistic_systems.circuits.native_sampled import (
    NativeSampledPathPromotionReceipt,
    compile_sampled_path_route_groups,
    materialize_native_sampled_path_circuits,
    neyman_policy_from_profile,
    run_profiled_native_sampled_path_panel,
)
from mechanistic_systems.circuits.final_validation import (
    validate_native_sampled_path_final_masks,
)
from mechanistic_systems.circuits.dense_anchor import (
    derive_native_dense_anchor_receipt,
)
from mechanistic_systems.circuits.sampled_torch import (
    ContextAxisSampledPanelSource,
    TorchSampledPathEvaluationBackend,
)
from mechanistic_systems.backends.calibration_runtime import (
    RuntimeExecutionProfile,
)
from mechanistic_systems.circuits.search import (
    PartitionLabelProvider,
    materialize_identified_circuits,
)
from mechanistic_systems.circuits.torch_evaluator import (
    TorchCircuitEvaluator,
)
from mechanistic_systems.circuits.statistical_sequential import (
    StatisticalSequentialProfile,
    validate_plan_against_statistical_profile,
)
from mechanistic_systems.contracts import (
    CANONICAL_CALIBRATION_ANCHOR_PROBABILITY_EPSILON,
    CANONICAL_CALIBRATION_HESSIAN_RIDGE,
    CANONICAL_CALIBRATION_IDENTITY_ANCHOR_STRENGTHS,
    CANONICAL_CALIBRATION_KKT_TOLERANCE,
    CANONICAL_CALIBRATION_MAX_ITERATIONS,
    CANONICAL_CALIBRATION_OBJECTIVE_TOLERANCE,
    CANONICAL_CALIBRATION_REFERENCE_TOLERANCE,
    CANONICAL_CALIBRATION_SELECTION_BLOCKS,
    CANONICAL_PRUNING_CALIBRATION_ALGORITHM,
    CircuitPlan,
    MeasurementPlan,
    PRODUCTION_CIRCUIT_SEARCH_ALGORITHM,
    QuerySpec,
    SAMPLED_PATH_GROUP_REFERENCE_POLICY,
    StudySpec,
)
from mechanistic_systems.measurement.contexts import (
    PREPROCESSED_UINT8_BANK_LOADER_CONTRACT,
    ClassIndexedUint8ContextSource,
    ContextAxis,
    ImageFolderCircuitProvider,
    ImageFolderContextSource,
    PreprocessedUint8BankContextSource,
)
from mechanistic_systems.measurement.models import (
    LoadedRegisteredModel,
)
from mechanistic_systems.measurement.model_sources import (
    ModelSourceReplica,
    load_model_for_study,
)
from mechanistic_systems.measurement.native_panel_data_plane import (
    NativePanelDataPlane,
    bind_device_gather_run,
    NativePanelExecutionSpec,
    image_gather_worker_count,
)
from mechanistic_systems.measurement.runner import (
    validate_context_axis_for_study,
)
from mechanistic_systems.measurement.streaming import (
    SufficientStatisticsMeasurementReader,
)
from mechanistic_systems.measurement.storage import (
    BlockMeasurementReader,
)

try:  # Optional for import smoke; required by registered execution.
    import torch
except Exception:  # pragma: no cover - environment dependent.
    torch = None  # type: ignore[assignment]


def _apply_tf32_execution_policy(allow_tf32: bool) -> None:
    if not allow_tf32:
        return
    if torch is None:
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _sampled_path_calibration_options(
    profile: StatisticalSequentialProfile,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "binary_loss_weighting": str(profile.binary_loss_weighting),
        "enforce_positive_scale": bool(profile.calibration_enforce_positive_scale),
        "iters": int(profile.calibration_iterations),
        "lr": float(profile.calibration_learning_rate),
        "ws_min": float(profile.calibration_min_scale),
        "optimization_contract": profile.calibration_optimization_contract,
        "readout_columns": profile.readout_columns,
    }
    if (
        profile.calibration_optimization_contract
        == CANONICAL_PRUNING_CALIBRATION_ALGORITHM
    ):
        options.update(
            {
                "identity_anchor_strengths": (
                    CANONICAL_CALIBRATION_IDENTITY_ANCHOR_STRENGTHS
                ),
                "calibration_selection_blocks": (
                    CANONICAL_CALIBRATION_SELECTION_BLOCKS
                ),
                "anchor_probability_epsilon": (
                    CANONICAL_CALIBRATION_ANCHOR_PROBABILITY_EPSILON
                ),
                "iters": CANONICAL_CALIBRATION_MAX_ITERATIONS,
                "kkt_tolerance": CANONICAL_CALIBRATION_KKT_TOLERANCE,
                "hessian_ridge": CANONICAL_CALIBRATION_HESSIAN_RIDGE,
                "objective_tolerance": (CANONICAL_CALIBRATION_OBJECTIVE_TOLERANCE),
                "reference_tolerance": (CANONICAL_CALIBRATION_REFERENCE_TOLERANCE),
            }
        )
    return options


def _cuda_peak_gb(device: Any) -> float | None:
    if torch is None:
        return None
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated(resolved) / (1024**3))


def _reset_cuda_peak(device: Any) -> None:
    if torch is None:
        return
    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(resolved)


def naturalized_batch_size(
    loaded_model: Any,
    device: Any,
    *,
    default: int = 8,
) -> int:
    """The measured throughput-saturating batch for a loaded model.

    The continuous canonical lane treats batch size as a hardware fact, not
    configuration: callers opt in by passing the result as ``batch_size``.
    Claim-bearing replays keep their configured value because it enters
    work-packet hashes.
    """

    import torch

    from packages.loader_runtime.throughput import (
        measure_saturating_batch,
    )

    resolved = torch.device(device)
    if resolved.type != "cuda":
        return int(default)
    model = getattr(loaded_model, "model", loaded_model)
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        with torch.no_grad():
            return int(
                measure_saturating_batch(
                    model,
                    device=resolved,
                    fallback=int(default),
                )
            )
    finally:
        if was_training:
            model.train()


def _measurement_queries(store: ArtifactStore, plan_hash: str) -> tuple[QuerySpec, ...]:
    payload = store.load_contract("measurement_plan", plan_hash)
    plan = MeasurementPlan.from_mapping(payload)
    if plan.content_hash != plan_hash:
        raise ValueError(
            "stored MeasurementPlan content hash differs from its registry key"
        )
    return plan.queries


def bind_measurement_parent(
    *,
    template: CircuitPlan,
    parent: Any,
    freeze: bool,
) -> CircuitPlan:
    """Bind an immutable measurement parent (by content address) into a
    CircuitPlan template, optionally freezing it for claims.

    Factored out of the CLI so the same binding is reusable by the
    combined in-process ``score_and_prune`` path (and any other
    orchestration) without duplicating the notes / search-policy / freeze
    semantics. ``parent`` is the measurement ``ArtifactManifest``.
    """

    if parent.study_hash != template.study_hash:
        raise ValueError("measurement parent and CircuitPlan template studies differ")
    target_status = "frozen" if freeze else template.status
    if target_status == "frozen" and not parent.valid_for_claims:
        raise ValueError(
            "frozen CircuitPlan requires a claim-valid MeasurementArtifact parent"
        )

    notes = dict(template.notes)
    notes.pop("measurement_parent_placeholder", None)
    notes.update(
        {
            "measurement_parent_bound": True,
            "source_circuit_plan_template_hash": template.content_hash,
        }
    )
    if freeze:
        notes.pop("descriptive_only", None)
        notes["freeze_transition"] = "claim_valid_measurement_parent_binding_v1"

    search_policy = dict(template.search_policy)
    if str(search_policy.get("algorithm", "")) == (
        "statistical_sequential_symmetric_frontier"
    ):
        observed_reference_policy = str(
            search_policy.get("reference_anchor_policy", "")
        )
        if observed_reference_policy and observed_reference_policy != (
            SAMPLED_PATH_GROUP_REFERENCE_POLICY
        ):
            raise ValueError(
                "CircuitPlan template declares an unsupported sampled-path "
                "reference-anchor policy"
            )
        search_policy["reference_anchor_policy"] = (
            SAMPLED_PATH_GROUP_REFERENCE_POLICY
        )
        notes["reference_anchor_policy_bound_at_freeze"] = True

    plan = replace(
        template,
        measurement_artifact_hash=parent.artifact_hash,
        search_policy=search_policy,
        status=target_status,
        notes=notes,
    )
    plan.validate(claim_bearing=target_status == "frozen").require_valid()
    return plan


def materialize_registered_circuits(
    *,
    study: StudySpec,
    plan: CircuitPlan,
    optimization_axis: ContextAxis,
    evaluation_axis: ContextAxis,
    optimization_image_root: str | Path,
    evaluation_image_root: str | Path,
    optimization_uint8_cache_manifest: str | Path | None = None,
    optimization_uint8_cache_receipt: str | Path | None = None,
    evaluation_uint8_cache_manifest: str | Path | None = None,
    evaluation_uint8_cache_receipt: str | Path | None = None,
    store: ArtifactStore,
    staging_dir: str | Path,
    checkpoint_path: str | Path | None = None,
    allow_download: bool = False,
    device: Any = "cpu",
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = True,
    parity_receipt_hash: str = "",
    partition: str = "local",
    work_packet_hash: str = "",
    partition_label_provider: PartitionLabelProvider | None = None,
    loaded_model: LoadedRegisteredModel | None = None,
    model_source_replica: ModelSourceReplica | None = None,
    queries: Sequence[QuerySpec] | None = None,
) -> dict[str, Any]:
    """Run the exploratory native search for a registered model.

    Frozen sampled-path plans are intentionally rejected here. The native v1
    implementation is admitted only through its complete-trace parity gate;
    historical oracle execution is exposed administratively as CircuitFlow
    ``repair sealed-pruning-oracle-run``.
    """

    search_algorithm = str(plan.search_policy.get("algorithm", ""))
    if search_algorithm == PRODUCTION_CIRCUIT_SEARCH_ALGORITHM:
        raise ValueError(
            "sampled_path_frontier_v1 plans require the parity-gated native "
            "sampled-path runner; historical oracle execution is available "
            "only through CircuitFlow `repair sealed-pruning-oracle-run`"
        )
    if plan.status in {"frozen", "legacy_matched"}:
        raise ValueError(
            "claim-bearing or legacy-matched circuit plans require "
            "statistical_sequential_symmetric_frontier"
        )

    if plan.study_hash != study.content_hash:
        raise ValueError("CircuitPlan study hash differs from StudySpec")
    if plan.optimization_view.view_hash != optimization_axis.data_view().view_hash:
        raise ValueError("CircuitPlan optimization view differs from its context axis")
    if plan.evaluation_view.view_hash != evaluation_axis.data_view().view_hash:
        raise ValueError("CircuitPlan evaluation view differs from its context axis")
    validate_context_axis_for_study(optimization_axis, study)
    validate_context_axis_for_study(evaluation_axis, study)
    measurement_manifest = store.load(
        plan.measurement_artifact_hash, ArtifactType.MEASUREMENT
    )
    if measurement_manifest.study_hash != study.content_hash:
        raise ValueError("measurement parent and StudySpec differ")
    measurement_bundle = store.bundle_dir(measurement_manifest)
    measurement = SufficientStatisticsMeasurementReader(
        measurement_bundle / "payloads" / "measurement_sufficient_statistics.zarr"
    )
    query_axis = tuple(
        queries or _measurement_queries(store, measurement_manifest.plan_hash)
    )
    if tuple(query.query_id for query in query_axis) != tuple(measurement.query_ids):
        raise ValueError("query contract axis differs from MeasurementArtifact")
    loaded = loaded_model or load_model_for_study(
        study,
        source_replica=model_source_replica,
        checkpoint_path=checkpoint_path,
        allow_download=allow_download,
        device=device,
        claim_bearing=plan.status == "frozen",
    )
    loaded.validate_against_study(study, claim_bearing=plan.status == "frozen")
    resolved_partition_provider = partition_label_provider
    if (
        resolved_partition_provider is None
        and plan.partition_policy == "route_coordinate_sign"
        and str(plan.search_policy.get("algorithm", ""))
        == "reverse_hierarchical_equalized_marginal"
    ):
        resolved_partition_provider = TorchStructuralPolarityProvider(
            loaded.architecture,
            loaded.model,
            residual_route_policy=resolve_residual_route_partition_policy(
                loaded.architecture,
                plan.search_policy.get("residual_route_partition_policy"),
            ),
        )
    sources = {
        optimization_axis.data_view().view_hash: ImageFolderContextSource(
            optimization_axis,
            optimization_image_root,
            batch_size=int(batch_size),
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
        )
    }
    evaluation_hash = evaluation_axis.data_view().view_hash
    if evaluation_hash not in sources:
        sources[evaluation_hash] = ImageFolderContextSource(
            evaluation_axis,
            evaluation_image_root,
            batch_size=int(batch_size),
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
        )
    evaluator = TorchCircuitEvaluator.from_bound_architecture(
        loaded.architecture,
        model=loaded.model,
        queries=query_axis,
        batch_provider=ImageFolderCircuitProvider(sources),
        parity_receipt_hash=parity_receipt_hash,
        device=device,
    )
    store.seal_contract(
        contract_kind="study",
        content_hash=study.content_hash,
        payload=study.to_dict(),
    )
    store.seal_contract(
        contract_kind="circuit_plan",
        content_hash=plan.content_hash,
        payload=plan.to_dict(),
    )
    store.seal_contract(
        contract_kind="model_load_receipt",
        content_hash=loaded.receipt.content_hash,
        payload=loaded.receipt.to_dict(),
    )
    packet_hash = work_packet_hash or stable_digest(
        {
            "stage": "circuit_identification",
            "plan_hash": plan.content_hash,
            "optimization_axis_hash": optimization_axis.content_hash,
            "evaluation_axis_hash": evaluation_axis.content_hash,
            "model_load_receipt_hash": loaded.receipt.content_hash,
            "device": str(device),
            "batch_size": int(batch_size),
        }
    )
    _reset_cuda_peak(device)
    started = time.perf_counter()
    manifest, bundle, result = materialize_identified_circuits(
        plan,
        measurement_manifest,
        measurement,
        evaluator,
        store,
        staging_dir=staging_dir,
        partition_label_provider=resolved_partition_provider,
    )
    wall_seconds = float(time.perf_counter() - started)
    evaluation_count = len(optimization_axis.records) + len(evaluation_axis.records)
    candidate_count = sum(len(rows) for rows in result.candidates_by_query.values())
    receipt = RuntimeReceipt(
        artifact_hash=manifest.artifact_hash,
        work_packet_hash=packet_hash,
        stage="circuit_identification",
        partition=str(partition),
        wall_seconds=wall_seconds,
        throughput_items_per_second=(
            float(evaluation_count * max(1, candidate_count)) / wall_seconds
            if wall_seconds > 0.0
            else None
        ),
        peak_hbm_gb=_cuda_peak_gb(device),
        cache_ready=True,
        details={
            "evaluator_id": evaluator.evaluator_id,
            "optimization_axis_hash": optimization_axis.content_hash,
            "evaluation_axis_hash": evaluation_axis.content_hash,
            "model_load_receipt_hash": loaded.receipt.content_hash,
            "query_count": len(query_axis),
            "candidate_count": candidate_count,
            "selected_retained_routes": {
                query_id: candidate.retained_route_count
                for query_id, candidate in result.selected_by_query.items()
            },
            "candidate_execution": evaluator.candidate_execution_profile(),
        },
    )
    receipt_path = store.write_runtime_receipt(receipt)
    return {
        "schema_name": "mechanistic_registered_circuit_run_v1",
        "artifact_hash": manifest.artifact_hash,
        "bundle": str(bundle),
        "study_hash": study.content_hash,
        "plan_hash": plan.content_hash,
        "measurement_artifact_hash": measurement_manifest.artifact_hash,
        "optimization_axis_hash": optimization_axis.content_hash,
        "evaluation_axis_hash": evaluation_axis.content_hash,
        "model_load_receipt_hash": loaded.receipt.content_hash,
        "runtime_receipt": str(receipt_path),
        "wall_seconds": wall_seconds,
        "valid_for_claims": bool(manifest.valid_for_claims),
        "selected_retained_routes": {
            query_id: candidate.retained_route_count
            for query_id, candidate in result.selected_by_query.items()
        },
    }


def run_registered_sampled_path_group(
    *,
    evaluator: TorchCircuitEvaluator,
    panel_source: Any,
    queries: Sequence[QuerySpec],
    profile: Any,
    scores: Mapping[str, Any],
    signs: Mapping[str, Any],
    query_id: str,
    layer_id: str,
    route_group_kind: str,
    dense_anchor_loss: float,
    previous_group_masks: Mapping[str, Any] | None = None,
    parity_receipt: Any | None = None,
    claim_bearing: bool = False,
) -> Any:
    """Execute one parity-gated v1 layer group on the Torch backend.

    Full artifact materialization remains parity-gated until a supervised
    canary receipt promotes this path. This exposes canonical numerical
    ownership without silently changing production authority.
    """

    from mechanistic_systems.circuits.native_sampled import (
        neyman_policy_from_profile,
        run_profiled_native_sampled_path_group,
    )
    from mechanistic_systems.circuits.sampled_torch import (
        TorchSampledPathEvaluationBackend,
    )

    profile.validate()
    backend = TorchSampledPathEvaluationBackend(
        evaluator=evaluator,
        panel_source=panel_source,
        queries=queries,
        calibration_options=_sampled_path_calibration_options(profile),
        panel_policy=neyman_policy_from_profile(profile),
        runtime_execution_profile=evaluator.runtime_execution_profile,
    )
    return run_profiled_native_sampled_path_group(
        backend=backend,
        profile=profile,
        scores=scores,
        partition_labels={
            str(layer_id): np.where(np.asarray(value) >= 0, 0, 1).astype(np.int8)
            for layer_id, value in signs.items()
        },
        query_id=query_id,
        layer_id=layer_id,
        route_group_kind=route_group_kind,
        dense_anchor_loss=dense_anchor_loss,
        previous_group_masks=previous_group_masks,
        parity_receipt=parity_receipt,
        claim_bearing=claim_bearing,
    )


@dataclass(frozen=True)
class RegisteredCircuitRuntime:
    """Resolved registered-model runtime shared by circuit backends.

    This is the neutral public name for the runtime of the sampled-path v0
    entry point.  The private name below is an
    exact alias, rather than a subclass or wrapper, so old imports, type
    checks, construction, and replay behavior remain unchanged.
    """

    measurement_manifest: Any
    measurement: Any
    query_axis: tuple[QuerySpec, ...]
    loaded: LoadedRegisteredModel
    route_groups: tuple[Any, ...]
    partition_labels_by_layer: Mapping[str, np.ndarray]
    partition_label_provider_id: str
    partition_label_hash: str
    optimization_source: Any
    evaluation_source: Any
    optimization_data_plane: NativePanelDataPlane
    evaluation_data_plane: NativePanelDataPlane
    walk_device_cache_bytes: int
    evaluator: TorchCircuitEvaluator
    backend: TorchSampledPathEvaluationBackend


# Backward-compatibility boundary: private v0 consumers must continue to see
# exactly the same class object.
_RegisteredSampledPathRuntime = RegisteredCircuitRuntime


def resolve_registered_circuit_runtime(
    *,
    study: StudySpec,
    plan: CircuitPlan,
    profile: StatisticalSequentialProfile,
    promotion_receipt: NativeSampledPathPromotionReceipt | None,
    optimization_axis: ContextAxis,
    evaluation_axis: ContextAxis,
    optimization_image_root: str | Path,
    evaluation_image_root: str | Path,
    optimization_uint8_cache_manifest: str | Path | None,
    optimization_uint8_cache_receipt: str | Path | None,
    evaluation_uint8_cache_manifest: str | Path | None,
    evaluation_uint8_cache_receipt: str | Path | None,
    store: ArtifactStore,
    profiler: Any,
    checkpoint_path: str | Path | None,
    allow_download: bool,
    device: Any,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    loaded_model: LoadedRegisteredModel | None,
    model_source_replica: ModelSourceReplica | None,
    queries: Sequence[QuerySpec] | None,
    measurement_artifact_policy: str = "required",
    model_transform: Any | None = None,
    retained_inputs: dict[str, Any] | None = None,
) -> RegisteredCircuitRuntime:
    """Resolve one canonical backend shared by full and prefix execution.

    ``model_transform`` (a registered ``ModelTransformSpec``)
    is forwarded to the model loader, which re-derives and verifies the
    transformed state on the unchanged base Study.  A supplied
    ``loaded_model`` must carry exactly the requested transform's hash (and
    none when no transform is requested).  ``None`` is the v5 path.

    ``measurement_artifact_policy`` is ``"required"`` (default,
    byte-identical to the incumbent path) or ``"contracts_only"``.  The
    modern pilot re-measures every field it consumes in its own live
    census, and its only reads of the score-era MeasurementArtifact are
    the query-id axis and the route blocks — both of which the sealed
    contracts and the bound model already carry.  Under
    ``contracts_only`` the caller MUST pass the sealed ``queries``
    explicitly; route blocks come from the bound architecture (making
    the downstream measurement-vs-executable cross-check exact by
    construction), and the runtime discloses the policy where the
    artifact hash would appear.  No numerical measurement content is
    synthesized anywhere.
    """

    policy_value = str(measurement_artifact_policy)
    if policy_value not in {"required", "contracts_only"}:
        raise ValueError(
            "measurement_artifact_policy must be 'required' or "
            "'contracts_only'"
        )
    if policy_value == "contracts_only" and not queries:
        raise ValueError(
            "contracts_only resolution requires the sealed query axis"
        )
    measurement_manifest: Any = None
    measurement: Any = None
    if policy_value == "required":
        with profiler.phase("circuit.parent_resolution"):
            measurement_manifest = store.load(
                plan.measurement_artifact_hash,
                ArtifactType.MEASUREMENT,
            )
            if measurement_manifest.study_hash != study.content_hash:
                raise ValueError("measurement parent and StudySpec differ")
            payload_dir = store.bundle_dir(measurement_manifest) / "payloads"
            sufficient_statistics = payload_dir / "measurement_sufficient_statistics.zarr"
            aggregate_heads = payload_dir / "measurement_route_heads.zarr"
            if sufficient_statistics.exists():
                measurement = SufficientStatisticsMeasurementReader(
                    sufficient_statistics
                )
            elif aggregate_heads.exists():
                measurement = BlockMeasurementReader(aggregate_heads)
            else:
                raise ValueError(
                    "MeasurementArtifact has no supported canonical route payload"
                )
            query_axis = tuple(
                queries or _measurement_queries(store, measurement_manifest.plan_hash)
            )
            if tuple(query.query_id for query in query_axis) != tuple(
                measurement.query_ids
            ):
                raise ValueError("query contract axis differs from MeasurementArtifact")
    else:
        with profiler.phase("circuit.parent_resolution"):
            query_axis = tuple(queries)

    with profiler.phase("circuit.model_resolution"):
        if retained_inputs is not None and retained_inputs.get("loaded") is not None:
            loaded_model = retained_inputs["loaded"]
        supplied_transform_hash = ""
        if loaded_model is not None:
            from mechanistic_systems.measurement.models import ModelLoadReceipt

            supplied_receipt = getattr(loaded_model, "receipt", None)
            if isinstance(supplied_receipt, ModelLoadReceipt):
                supplied_transform_hash = supplied_receipt.model_transform_hash
        if loaded_model is not None and (
            model_transform is not None or supplied_transform_hash
        ):
            requested_transform_hash = (
                "" if model_transform is None else model_transform.content_hash
            )
            if supplied_transform_hash != requested_transform_hash:
                raise ValueError(
                    "supplied loaded model carries a different model transform "
                    "than the one requested"
                )
        loaded = loaded_model or load_model_for_study(
            study,
            source_replica=model_source_replica,
            checkpoint_path=checkpoint_path,
            allow_download=allow_download,
            device=device,
            claim_bearing=plan.status == "frozen",
            **(
                {}
                if model_transform is None
                else {"model_transform": model_transform}
            ),
        )
        loaded.validate_against_study(study, claim_bearing=plan.status == "frozen")
        if plan.route_universe_hash != loaded.architecture.circuit_route_schema_hash:
            raise ValueError("CircuitPlan route universe differs from the bound model")
    if measurement is None:
        # contracts_only: the identity surface the pilot actually
        # consumes, sourced from the sealed contracts and bound model.
        measurement = SimpleNamespace(
            query_ids=tuple(query.query_id for query in query_axis),
            route_blocks=tuple(loaded.architecture.circuit_route_blocks),
            schema_name="contracts_only_identity_view_A47",
        )
        measurement_manifest = SimpleNamespace(
            artifact_hash="measurement_artifact_policy_contracts_only_A47",
            study_hash=study.content_hash,
            plan_hash="",
        )
    with profiler.phase("circuit.route_group_compilation"):
        route_groups = compile_sampled_path_route_groups(
            profile=profile,
            bound_architecture=loaded.architecture,
            measurement_route_blocks=measurement.route_blocks,
        )
        if plan.partition_policy != "route_coordinate_sign":
            raise ValueError(
                "sampled_path_frontier_v1 requires structural route-coordinate polarity"
            )
        residual_partition_policy = resolve_residual_route_partition_policy(
            loaded.architecture,
            plan.search_policy.get("residual_route_partition_policy"),
        )
        partition_provider = TorchStructuralPolarityProvider(
            loaded.architecture,
            loaded.model,
            residual_route_policy=residual_partition_policy,
        )
        route_blocks = tuple(loaded.architecture.circuit_route_blocks)
        flat_partition_labels = np.asarray(
            partition_provider.labels(
                policy_id=plan.partition_policy,
                query_id="__structural__",
                query_index=0,
                score_mean=np.empty(0, dtype=np.float64),
                score_magnitude=np.empty(0, dtype=np.float64),
                route_blocks=route_blocks,
                random_seed=plan.random_seed,
            )
        )
        expected_route_count = sum(block.route_count for block in route_blocks)
        if (
            flat_partition_labels.shape != (expected_route_count,)
            or flat_partition_labels.dtype.kind not in {"i", "u", "b"}
            or not np.isin(flat_partition_labels, (0, 1)).all()
        ):
            raise ValueError(
                "structural polarity provider returned an invalid partition axis"
            )
        flat_partition_labels = np.ascontiguousarray(
            flat_partition_labels, dtype=np.int8
        )
        partition_labels_by_layer: dict[str, np.ndarray] = {}
        cursor = 0
        for block in route_blocks:
            stop = cursor + block.route_count
            partition_labels_by_layer[str(block.layer_id)] = (
                flat_partition_labels[cursor:stop].reshape(block.shape, order="C")
            )
            cursor = stop
        if cursor != expected_route_count:
            raise RuntimeError("structural partition compilation did not consume its axis")
        label_payload_hash = hashlib.sha256(
            memoryview(flat_partition_labels).cast("B")
        ).hexdigest()
        partition_label_hash = stable_digest(
            {
                "provider_id": partition_provider.provider_id,
                "partition_policy": plan.partition_policy,
                "residual_route_partition_policy": residual_partition_policy,
                "route_universe_hash": plan.route_universe_hash,
                "label_payload_hash": label_payload_hash,
            }
        )

    def context_source(
        axis: ContextAxis,
        image_root: str | Path,
        cache_manifest: str | Path | None,
        cache_receipt: str | Path | None,
        role: str,
    ) -> Any:
        if retained_inputs is not None:
            reused = retained_inputs.get(role)
            if reused is not None:
                if (
                    reused.axis is not axis
                    or int(reused.batch_size) != int(batch_size)
                    or str(reused.device) != str(device)
                    or bool(reused.pin_memory) != bool(pin_memory)
                ):
                    raise ValueError("retained bank source differs from this runtime")
                if cache_manifest is not None and (
                    Path(reused.source_cache_manifest_path).resolve()
                    != Path(cache_manifest).resolve()
                ):
                    raise ValueError("retained bank manifest differs from this runtime")
                if cache_manifest is None and (
                    Path(reused.root).resolve() != Path(image_root).resolve()
                    or int(reused.num_workers) != int(num_workers)
                ):
                    raise ValueError("retained image root differs from this runtime")
                # No staging or page warming was done for this item. Keep
                # the bank's identity and location, but retire the previous
                # item's construction timings before its new plane binds it.
                if hasattr(reused, "staging_elapsed_seconds"):
                    reused.staging_elapsed_seconds = 0.0
                    reused.page_warm_performed = False
                    reused.page_warm_elapsed_seconds = 0.0
                    staging = dict(reused.node_local_bank_staging)
                    for name in (
                        "copy_seconds", "lock_wait_seconds",
                        "capacity_lock_wait_seconds",
                    ):
                        if name in staging:
                            staging[name] = 0.0
                    if staging.get("staging_mode") == "staged":
                        staging["replica_reused"] = True
                    reused.node_local_bank_staging = staging
                return reused
        requires_bank = axis.selector.get("runtime_loader_contract") == (
            PREPROCESSED_UINT8_BANK_LOADER_CONTRACT
        )
        if (cache_manifest is None) != (cache_receipt is None):
            raise ValueError(
                "uint8 cache manifest and receipt must be supplied together"
            )
        if cache_manifest is not None and cache_receipt is not None:
            return PreprocessedUint8BankContextSource(
                axis,
                cache_manifest,
                cache_receipt,
                batch_size=int(batch_size),
                device=device,
                preload_to_ram=False,
                pin_memory=bool(pin_memory),
                verification_mode="certified_probes",
            )
        if requires_bank:
            raise ValueError(
                "sampled-path axis requires a certified preprocessed uint8 bank replica"
            )
        return ClassIndexedUint8ContextSource(
            axis,
            image_root,
            batch_size=int(batch_size),
            device=device,
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
        )

    with profiler.phase("circuit.context_source.optimization"):
        optimization_source = context_source(
            optimization_axis,
            optimization_image_root,
            optimization_uint8_cache_manifest,
            optimization_uint8_cache_receipt,
            "optimization_source",
        )
    with profiler.phase("circuit.context_source.evaluation"):
        evaluation_source = context_source(
            evaluation_axis,
            evaluation_image_root,
            evaluation_uint8_cache_manifest,
            evaluation_uint8_cache_receipt,
            "evaluation_source",
        )
    sources = {
        optimization_axis.data_view().view_hash: optimization_source,
        evaluation_axis.data_view().view_hash: evaluation_source,
    }
    runtime_execution_profile = RuntimeExecutionProfile.from_environment()
    _apply_tf32_execution_policy(runtime_execution_profile.allow_tf32)
    panel_execution_spec = NativePanelExecutionSpec(
        scoring_batch_rows=int(runtime_execution_profile.scoring_batch_rows),
        calibration_batch_rows=int(
            runtime_execution_profile.calibration_batch_rows
        ),
        calibration_oracle_fused_batch_rows=max(
            int(runtime_execution_profile.calibration_batch_rows),
            int(runtime_execution_profile.v1_candidate_oracle_batch_rows),
        ),
        v0_paired_oracle_batch_rows=int(
            runtime_execution_profile.v0_paired_oracle_batch_rows
        ),
        v1_candidate_oracle_batch_rows=int(
            runtime_execution_profile.v1_candidate_oracle_batch_rows
        ),
        terminal_batch_rows=int(runtime_execution_profile.terminal_batch_rows),
        prefetch_depth=int(runtime_execution_profile.sampled_prefetch_depth),
        producer_workers=(
            image_gather_worker_count(
                requested_workers=int(runtime_execution_profile.sampled_producer_workers)
            )
            if os.environ.get("IRG_IMAGE_GATHER", "0") == "1"
            else int(runtime_execution_profile.sampled_producer_workers)
        ),
        working_set_prefetch=bool(
            getattr(
                runtime_execution_profile,
                "working_set_prefetch",
                False,
            )
        ),
        # The census phase never allocates the device cache.  Its configured
        # byte budget is deferred to the v0 walk boundary through
        # ``_RegisteredSampledPathRuntime.walk_device_cache_bytes``.
        device_cache_bytes=0,
        require_optimized_bank=bool(
            runtime_execution_profile.require_optimized_native_panel
        ),
        unit_packing_degree=int(
            runtime_execution_profile.unit_packing_degree
        ),
        query_tile_size=int(runtime_execution_profile.query_tile_size),
    )
    optimization_data_plane = NativePanelDataPlane(
        optimization_source,
        spec=panel_execution_spec,
        device=device,
        model_state_hash=loaded.receipt.state_dict_hash,
    )
    evaluation_data_plane = NativePanelDataPlane(
        evaluation_source,
        spec=replace(panel_execution_spec, require_optimized_bank=False),
        device=device,
        model_state_hash=loaded.receipt.state_dict_hash,
    )
    bind_device_gather_run((optimization_data_plane, evaluation_data_plane))
    with profiler.phase("circuit.backend_compilation"):
        evaluator_parity_receipt_hash = (
            promotion_receipt.content_hash
            if promotion_receipt is not None
            else stable_digest(
                {
                    "schema_name": "registered_exploratory_evaluator_binding_v1",
                    "plan_hash": plan.content_hash,
                    "model_load_receipt_hash": loaded.receipt.content_hash,
                    "claim_bearing": False,
                }
            )
        )
        retained_evaluator = (
            None if retained_inputs is None else retained_inputs.get("evaluator")
        )
        if retained_evaluator is None:
            evaluator = TorchCircuitEvaluator.from_bound_architecture(
                loaded.architecture,
                model=loaded.model,
                queries=query_axis,
                batch_provider=ImageFolderCircuitProvider(sources),
                parity_receipt_hash=evaluator_parity_receipt_hash,
                device=device,
                require_fixed_readout=True,
                runtime_execution_profile=runtime_execution_profile,
            )
        else:
            if (
                retained_evaluator.source_model is not loaded.model
                or retained_evaluator.query_ids != tuple(q.query_id for q in query_axis)
                or retained_evaluator.route_schema_hash != plan.route_universe_hash
                or retained_evaluator.parity_receipt_hash != evaluator_parity_receipt_hash
                or retained_evaluator.runtime_execution_profile != runtime_execution_profile
            ):
                raise ValueError("retained evaluator differs from this runtime")
            retained_evaluator._gate_graph.reset()
            retained_evaluator._baseline_cache.clear()
            retained_evaluator._fully_pruned_cache.clear()
            # The driver restored process flags after the previous item.
            # Reapply the configured tier on this item's first forward.
            retained_evaluator._forward_tier_applied = False
            retained_evaluator.batch_provider = ImageFolderCircuitProvider(sources)
            from mechanistic_systems.backends.torch_runtime import (
                TorchCandidateExecutionBackend,
            )

            previous_backend = retained_evaluator.candidate_execution_backend
            fresh_backend = TorchCandidateExecutionBackend(
                runtime_execution_profile=runtime_execution_profile,
                compile_vmap=previous_backend.compile_vmap,
                forward_prefix_reuse=previous_backend.forward_prefix_reuse,
            )
            fresh_backend._prepared_single_runner_cache.update(
                previous_backend._prepared_single_runner_cache
            )
            fresh_backend._prepared_vmap_runner_cache.update(
                previous_backend._prepared_vmap_runner_cache
            )
            fresh_backend._prefix_split_cache.update(
                previous_backend._prefix_split_cache
            )
            fresh_backend._base_buffers_cache.update(
                previous_backend._base_buffers_cache
            )
            fresh_backend.carry_worker_walk_graphs(previous_backend)
            fresh_backend._cuda_graph_cache.update(previous_backend._cuda_graph_cache)
            fresh_backend.cuda_graphs_applied = previous_backend.cuda_graphs_applied
            # These flags describe the carried executable runners as well
            # as runners first compiled by this backend. Counters stay fresh.
            fresh_backend.compile_single_lane_applied = (
                previous_backend.compile_single_lane_applied
            )
            fresh_backend.compile_vmap_applied = previous_backend.compile_vmap_applied
            retained_evaluator.candidate_execution_backend = fresh_backend
            evaluator = retained_evaluator
        planes = (optimization_data_plane, evaluation_data_plane)
        bind_stem = getattr(evaluator, "bind_uint8_folded_stem", None)
        if callable(bind_stem):
            bind_stem(planes)
        elif any(getattr(plane, "_uint8_folded_stem_env", False) for plane in planes):
            raise ValueError("folded uint8 delivery requires an evaluator stem binding")
        panel_source = ContextAxisSampledPanelSource(
            optimization_axis,
            optimization_source,
        )
        backend = TorchSampledPathEvaluationBackend(
            evaluator=evaluator,
            panel_source=panel_source,
            queries=query_axis,
            calibration_options=_sampled_path_calibration_options(profile),
            panel_policy=neyman_policy_from_profile(profile),
            runtime_execution_profile=runtime_execution_profile,
        )
    return RegisteredCircuitRuntime(
        measurement_manifest=measurement_manifest,
        measurement=measurement,
        query_axis=query_axis,
        loaded=loaded,
        route_groups=route_groups,
        partition_labels_by_layer=partition_labels_by_layer,
        partition_label_provider_id=partition_provider.provider_id,
        partition_label_hash=partition_label_hash,
        optimization_source=optimization_source,
        evaluation_source=evaluation_source,
        optimization_data_plane=optimization_data_plane,
        evaluation_data_plane=evaluation_data_plane,
        walk_device_cache_bytes=int(
            runtime_execution_profile.native_panel_device_cache_bytes
        ),
        evaluator=evaluator,
        backend=backend,
    )


# Backward-compatibility boundary: keep the historical resolver as an exact
# alias so its signature, defaults, and returned object are byte-for-byte the
# incumbent path rather than a compatibility wrapper.
_resolve_registered_sampled_path_runtime = resolve_registered_circuit_runtime


def verify_registered_sampled_path_prefix(
    *,
    study: StudySpec,
    plan: CircuitPlan,
    profile: StatisticalSequentialProfile,
    promotion_receipt: NativeSampledPathPromotionReceipt,
    query_id: str,
    group_count: int,
    output_path: str | Path,
    dense_anchor_loss_by_query: Mapping[str, float] | None = None,
    dense_anchor_evidence_hash: str = "",
    optimization_axis: ContextAxis,
    evaluation_axis: ContextAxis,
    optimization_image_root: str | Path,
    evaluation_image_root: str | Path,
    optimization_uint8_cache_manifest: str | Path | None = None,
    optimization_uint8_cache_receipt: str | Path | None = None,
    evaluation_uint8_cache_manifest: str | Path | None = None,
    evaluation_uint8_cache_receipt: str | Path | None = None,
    store: ArtifactStore,
    checkpoint_path: str | Path | None = None,
    allow_download: bool = False,
    device: Any = "cpu",
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = True,
    partition: str = "local",
    work_packet_hash: str = "",
    loaded_model: LoadedRegisteredModel | None = None,
    model_source_replica: ModelSourceReplica | None = None,
    queries: Sequence[QuerySpec] | None = None,
) -> dict[str, Any]:
    """Run an immutable non-claim prefix through the production backend.

    Prefix verification keeps the full route universe dense outside the
    visited prefix. It deliberately skips final validation and circuit
    materialization, so its receipt can certify sequential semantics without
    being mistaken for a completed scientific artifact.
    """

    from mechanistic_systems.backends.profiling import PhaseProfiler

    if plan.status not in {"frozen", "legacy_matched"}:
        raise ValueError(
            "sampled-path prefix verification requires a frozen canonical plan "
            "or a legacy-matched diagnostic control"
        )
    claim_bearing = plan.status == "frozen"
    plan.validate(claim_bearing=claim_bearing).require_valid()
    if plan.study_hash != study.content_hash:
        raise ValueError("CircuitPlan study hash differs from StudySpec")
    validate_plan_against_statistical_profile(plan, profile)
    if plan.optimization_view.view_hash != optimization_axis.data_view().view_hash:
        raise ValueError("CircuitPlan optimization view differs from its context axis")
    if plan.evaluation_view.view_hash != evaluation_axis.data_view().view_hash:
        raise ValueError("CircuitPlan evaluation view differs from its context axis")
    validate_context_axis_for_study(optimization_axis, study)
    validate_context_axis_for_study(evaluation_axis, study)
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(
            f"immutable sampled-path prefix receipt exists: {destination}"
        )

    _reset_cuda_peak(device)
    started = time.perf_counter()
    profiler = PhaseProfiler()
    runtime = _resolve_registered_sampled_path_runtime(
        study=study,
        plan=plan,
        profile=profile,
        promotion_receipt=promotion_receipt,
        optimization_axis=optimization_axis,
        evaluation_axis=evaluation_axis,
        optimization_image_root=optimization_image_root,
        evaluation_image_root=evaluation_image_root,
        optimization_uint8_cache_manifest=optimization_uint8_cache_manifest,
        optimization_uint8_cache_receipt=optimization_uint8_cache_receipt,
        evaluation_uint8_cache_manifest=evaluation_uint8_cache_manifest,
        evaluation_uint8_cache_receipt=evaluation_uint8_cache_receipt,
        store=store,
        profiler=profiler,
        checkpoint_path=checkpoint_path,
        allow_download=allow_download,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        loaded_model=loaded_model,
        model_source_replica=model_source_replica,
        queries=queries,
    )
    normalized_query = str(query_id)
    query_by_id = {query.query_id: query for query in runtime.query_axis}
    if normalized_query not in query_by_id:
        raise ValueError("sampled-path prefix query is outside the measurement axis")
    if not 1 <= int(group_count) <= len(runtime.route_groups):
        raise ValueError("sampled-path prefix group count is outside the route axis")
    promotion_receipt.require_passed(
        profile_hash=profile.content_hash,
        query_id=normalized_query,
    )

    native_dense_anchor = None
    if dense_anchor_loss_by_query is None:
        with profiler.phase("circuit.dense_anchor"):
            dense_masks = {
                group.layer_id: np.ones(group.shape, dtype=bool)
                for group in runtime.route_groups
            }
            native_dense_anchor = derive_native_dense_anchor_receipt(
                backend=runtime.backend,
                profile=profile,
                queries=(query_by_id[normalized_query],),
                dense_masks=dense_masks,
                first_layer_id=runtime.route_groups[0].layer_id,
                plan_hash=plan.content_hash,
                optimization_axis_hash=optimization_axis.content_hash,
                model_load_receipt_hash=runtime.loaded.receipt.content_hash,
            )
        resolved_anchor = native_dense_anchor.dense_anchor_loss_by_query[
            normalized_query
        ]
        resolved_anchor_hash = native_dense_anchor.content_hash
        anchor_source = "canonical_native_stage0_crossfit"
    else:
        normalized_anchors = {
            str(key): float(value) for key, value in dense_anchor_loss_by_query.items()
        }
        if normalized_query not in normalized_anchors:
            raise ValueError("sampled-path dense anchors omit the prefix query")
        resolved_anchor = normalized_anchors[normalized_query]
        resolved_anchor_hash = str(dense_anchor_evidence_hash)
        anchor_source = "imported_certified_oracle"
    if len(resolved_anchor_hash) != 64:
        raise ValueError(
            "sampled-path dense anchors require content-addressed evidence"
        )

    with profiler.phase("circuit.sampled_path_prefix"):
        # The prefix stage is a parity-diagnostic surface (native v1 against
        # the sealed oracle), so it stays pinned to the grid engine rather
        # than resolving the canonical identification default.
        runs_by_query = run_profiled_native_sampled_path_panel(
            algorithm="statistical_sequential_symmetric_frontier",
            backend=runtime.backend,
            profile=profile,
            measurement=runtime.measurement,
            query_ids=(normalized_query,),
            route_groups=runtime.route_groups,
            partition_labels_by_layer=runtime.partition_labels_by_layer,
            dense_anchor_loss_by_query={normalized_query: resolved_anchor},
            promotion_receipt=promotion_receipt,
            claim_bearing=claim_bearing,
            max_group_count=int(group_count),
        )
    runs = runs_by_query[normalized_query]
    group_rows = []
    for group_index, run in enumerate(runs):
        if run.reference_anchor_loss is None:
            raise ValueError("sampled-path prefix group omits its reference loss")
        reference_hashes = (
            run.reference_anchor_evidence_hash,
            run.reference_anchor_mask_hash,
            run.reference_anchor_panel_hash,
        )
        if any(value is None or len(str(value)) != 64 for value in reference_hashes):
            raise ValueError("sampled-path prefix group reference is incomplete")
        group_rows.append(
            {
                "group_index": group_index,
                "layer_id": run.signature.layer_id,
                "reference_anchor_loss": float(run.reference_anchor_loss),
                "reference_anchor_evidence_hash": str(
                    run.reference_anchor_evidence_hash
                ),
                "reference_anchor_mask_hash": str(run.reference_anchor_mask_hash),
                "reference_anchor_panel_hash": str(run.reference_anchor_panel_hash),
                "trace_signature": run.signature.to_dict(),
                "trace_signature_hash": run.signature.content_hash,
            }
        )
    scientific_payload = {
        "schema_name": "mechanistic_native_sampled_path_prefix_receipt_v1",
        "algorithm_version": "sampled_path_frontier_v1",
        "reference_anchor_policy": SAMPLED_PATH_GROUP_REFERENCE_POLICY,
        "plan_hash": plan.content_hash,
        "profile_hash": profile.content_hash,
        "promotion_receipt_hash": promotion_receipt.content_hash,
        "measurement_artifact_hash": runtime.measurement_manifest.artifact_hash,
        "source_plan_status": plan.status,
        "partition_label_provider_id": runtime.partition_label_provider_id,
        "partition_label_hash": runtime.partition_label_hash,
        "optimization_axis_hash": optimization_axis.content_hash,
        "query_id": normalized_query,
        "requested_group_count": int(group_count),
        "full_group_count": len(runtime.route_groups),
        "dense_anchor_source": anchor_source,
        "dense_anchor_evidence_hash": resolved_anchor_hash,
        "dense_anchor_loss": float(resolved_anchor),
        "groups": group_rows,
        "complete": len(group_rows) == int(group_count),
        "valid_for_claims": False,
        "analysis_status": "diagnostic_pending_oracle_comparison",
    }
    receipt_hash = stable_digest(scientific_payload)
    wall_seconds = float(time.perf_counter() - started)
    packet_hash = work_packet_hash or stable_digest(
        {
            "stage": "sampled_path_frontier_v1_native_prefix",
            "receipt_hash": receipt_hash,
            "partition": str(partition),
        }
    )
    payload = {
        **scientific_payload,
        "receipt_hash": receipt_hash,
        "runtime": {
            **profiler.snapshot(),
            "work_packet_hash": packet_hash,
            "partition": str(partition),
            "wall_seconds": wall_seconds,
            "peak_hbm_gb": _cuda_peak_gb(device),
            "model_load_receipt_hash": runtime.loaded.receipt.content_hash,
            "partition_label_provider_id": runtime.partition_label_provider_id,
            "partition_label_hash": runtime.partition_label_hash,
            "optimization_context_source": runtime.optimization_source.source_id,
            "evaluation_context_source": runtime.evaluation_source.source_id,
            "response_cache": runtime.backend.response_cache_profile(),
            "candidate_execution": runtime.backend.runtime_profile(),
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**payload, "path": str(destination)}


def materialize_registered_sampled_path_circuits(
    *,
    study: StudySpec,
    plan: CircuitPlan,
    profile: StatisticalSequentialProfile,
    promotion_receipt: NativeSampledPathPromotionReceipt,
    dense_anchor_loss_by_query: Mapping[str, float] | None = None,
    dense_anchor_evidence_hash: str = "",
    optimization_axis: ContextAxis,
    evaluation_axis: ContextAxis,
    optimization_image_root: str | Path,
    evaluation_image_root: str | Path,
    optimization_uint8_cache_manifest: str | Path | None = None,
    optimization_uint8_cache_receipt: str | Path | None = None,
    evaluation_uint8_cache_manifest: str | Path | None = None,
    evaluation_uint8_cache_receipt: str | Path | None = None,
    store: ArtifactStore,
    staging_dir: str | Path,
    checkpoint_path: str | Path | None = None,
    allow_download: bool = False,
    device: Any = "cpu",
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = True,
    partition: str = "local",
    work_packet_hash: str = "",
    loaded_model: LoadedRegisteredModel | None = None,
    model_source_replica: ModelSourceReplica | None = None,
    queries: Sequence[QuerySpec] | None = None,
    selected_query_ids: Sequence[str] | None = None,
    glocal_terminal_parity_tolerance: float | None = None,
    glocal_terminal_alpha: float = 0.01,
    glocal_terminal_sampling_design: str | None = None,
    glocal_terminal_population_size: int | None = None,
) -> dict[str, Any]:
    """Run the complete parity-promoted native v1 query-by-layer panel.

    The resulting circuit artifact remains analysis-blocked until the separate
    untouched-final-validation receipt is attached. This function owns the
    accepted score-to-frontier execution and does not delegate numerical
    work to the legacy pruning namespace.
    """

    from mechanistic_systems.backends.profiling import PhaseProfiler

    if plan.status != "frozen":
        raise ValueError(
            "claim-bearing native sampled-path runner requires a frozen "
            "CircuitPlan; bind its claim-valid measurement parent with "
            "`identify bind-measurement --freeze`"
        )
    plan.validate(claim_bearing=True).require_valid()
    if plan.study_hash != study.content_hash:
        raise ValueError("CircuitPlan study hash differs from StudySpec")
    validate_plan_against_statistical_profile(plan, profile)
    if plan.optimization_view.view_hash != optimization_axis.data_view().view_hash:
        raise ValueError("CircuitPlan optimization view differs from its context axis")
    if plan.evaluation_view.view_hash != evaluation_axis.data_view().view_hash:
        raise ValueError("CircuitPlan evaluation view differs from its context axis")
    validate_context_axis_for_study(optimization_axis, study)
    validate_context_axis_for_study(evaluation_axis, study)

    _reset_cuda_peak(device)
    started = time.perf_counter()
    profiler = PhaseProfiler()
    runtime = _resolve_registered_sampled_path_runtime(
        study=study,
        plan=plan,
        profile=profile,
        promotion_receipt=promotion_receipt,
        optimization_axis=optimization_axis,
        evaluation_axis=evaluation_axis,
        optimization_image_root=optimization_image_root,
        evaluation_image_root=evaluation_image_root,
        optimization_uint8_cache_manifest=optimization_uint8_cache_manifest,
        optimization_uint8_cache_receipt=optimization_uint8_cache_receipt,
        evaluation_uint8_cache_manifest=evaluation_uint8_cache_manifest,
        evaluation_uint8_cache_receipt=evaluation_uint8_cache_receipt,
        store=store,
        profiler=profiler,
        checkpoint_path=checkpoint_path,
        allow_download=allow_download,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        loaded_model=loaded_model,
        model_source_replica=model_source_replica,
        queries=queries,
    )
    measurement_manifest = runtime.measurement_manifest
    measurement = runtime.measurement
    measurement_query_axis = runtime.query_axis
    measurement_query_ids = tuple(query.query_id for query in measurement_query_axis)
    query_ids = (
        measurement_query_ids
        if selected_query_ids is None
        else tuple(str(value) for value in selected_query_ids)
    )
    if not query_ids or len(set(query_ids)) != len(query_ids):
        raise ValueError("sampled-path query packet requires unique query IDs")
    if any(query_id not in measurement_query_ids for query_id in query_ids):
        raise ValueError("sampled-path query packet is outside its measurement axis")
    query_positions = tuple(
        measurement_query_ids.index(query_id) for query_id in query_ids
    )
    if query_positions != tuple(sorted(query_positions)):
        raise ValueError(
            "sampled-path query packet must preserve measurement query order"
        )
    query_by_id = {query.query_id: query for query in measurement_query_axis}
    query_axis = tuple(query_by_id[query_id] for query_id in query_ids)
    loaded = runtime.loaded
    route_groups = runtime.route_groups
    optimization_source = runtime.optimization_source
    evaluation_source = runtime.evaluation_source
    evaluator = runtime.evaluator
    backend = runtime.backend

    native_dense_anchor = None
    if dense_anchor_loss_by_query is None:
        with profiler.phase("circuit.dense_anchor"):
            dense_masks = {
                group.layer_id: np.ones(group.shape, dtype=bool)
                for group in route_groups
            }
            native_dense_anchor = derive_native_dense_anchor_receipt(
                backend=backend,
                profile=profile,
                queries=query_axis,
                dense_masks=dense_masks,
                first_layer_id=route_groups[0].layer_id,
                plan_hash=plan.content_hash,
                optimization_axis_hash=optimization_axis.content_hash,
                model_load_receipt_hash=loaded.receipt.content_hash,
            )
        resolved_dense_anchors = native_dense_anchor.dense_anchor_loss_by_query
        resolved_dense_anchor_hash = native_dense_anchor.content_hash
        dense_anchor_source = "canonical_native_stage0_crossfit"
    else:
        provided_dense_anchors = {
            str(key): float(value) for key, value in dense_anchor_loss_by_query.items()
        }
        missing_anchors = tuple(
            query_id
            for query_id in query_ids
            if query_id not in provided_dense_anchors
        )
        if missing_anchors:
            raise ValueError(
                "sampled-path dense anchors omit query packet: "
                + ",".join(missing_anchors)
            )
        resolved_dense_anchors = {
            query_id: provided_dense_anchors[query_id] for query_id in query_ids
        }
        resolved_dense_anchor_hash = str(dense_anchor_evidence_hash)
        dense_anchor_source = "imported_certified_oracle"
    if set(resolved_dense_anchors) != set(query_ids):
        raise ValueError("dense-anchor query axis differs from MeasurementArtifact")
    if len(resolved_dense_anchor_hash) != 64:
        raise ValueError(
            "sampled-path dense anchors require content-addressed evidence"
        )

    store.seal_contract(
        contract_kind="study",
        content_hash=study.content_hash,
        payload=study.to_dict(),
    )
    store.seal_contract(
        contract_kind="circuit_plan",
        content_hash=plan.content_hash,
        payload=plan.to_dict(),
    )
    store.seal_contract(
        contract_kind="sampled_path_profile",
        content_hash=profile.content_hash,
        payload=profile.to_dict(),
    )
    store.seal_contract(
        contract_kind="sampled_path_promotion",
        content_hash=promotion_receipt.content_hash,
        payload=promotion_receipt.to_dict(),
    )
    store.seal_contract(
        contract_kind="model_load_receipt",
        content_hash=loaded.receipt.content_hash,
        payload=loaded.receipt.to_dict(),
    )
    if native_dense_anchor is not None:
        store.seal_contract(
            contract_kind="sampled_path_dense_anchor",
            content_hash=native_dense_anchor.content_hash,
            payload=native_dense_anchor.to_dict(),
        )
    packet_hash = work_packet_hash or stable_digest(
        {
            "stage": "sampled_path_frontier_v1_native_panel",
            "plan_hash": plan.content_hash,
            "profile_hash": profile.content_hash,
            "promotion_receipt_hash": promotion_receipt.content_hash,
            "dense_anchor_evidence_hash": resolved_dense_anchor_hash,
            "optimization_axis_hash": optimization_axis.content_hash,
            "evaluation_axis_hash": evaluation_axis.content_hash,
            "model_load_receipt_hash": loaded.receipt.content_hash,
            "partition_label_provider_id": runtime.partition_label_provider_id,
            "partition_label_hash": runtime.partition_label_hash,
            "optimization_context_source": optimization_source.source_id,
            "evaluation_context_source": evaluation_source.source_id,
            "response_cache": backend.response_cache_profile(),
            "optimization_replica_receipt_hash": getattr(
                getattr(optimization_source, "replica_receipt", None),
                "content_hash",
                None,
            ),
            "evaluation_replica_receipt_hash": getattr(
                getattr(evaluation_source, "replica_receipt", None),
                "content_hash",
                None,
            ),
            "query_ids": list(query_ids),
            "device": str(device),
            "batch_size": int(batch_size),
        }
    )
    with profiler.phase("circuit.sampled_path_panel"):
        runs = run_profiled_native_sampled_path_panel(
            backend=backend,
            profile=profile,
            measurement=measurement,
            query_ids=query_ids,
            route_groups=route_groups,
            partition_labels_by_layer=runtime.partition_labels_by_layer,
            dense_anchor_loss_by_query=resolved_dense_anchors,
            promotion_receipt=promotion_receipt,
            claim_bearing=True,
        )
    with profiler.phase("circuit.final_validation"):
        selected_masks_by_query = {
            query_id: {
                str(layer_id): np.asarray(mask, dtype=bool).copy()
                for layer_id, mask in tuple(query_runs)[
                    -1
                ].result.committed_masks.items()
            }
            for query_id, query_runs in runs.items()
        }
        final_validation = validate_native_sampled_path_final_masks(
            plan_hash=plan.content_hash,
            profile_hash=profile.content_hash,
            promotion_receipt_hash=promotion_receipt.content_hash,
            backend=backend,
            evaluator=evaluator,
            queries=query_axis,
            selected_masks_by_query=selected_masks_by_query,
            optimization_axis=optimization_axis,
            evaluation_axis=evaluation_axis,
            evaluation_source=evaluation_source,
            sample_stage_per_class=int(profile.sample_stages_per_class[0]),
            fold_count=int(profile.crossfit_fold_count),
            random_seed=int(profile.statistical_sampling_seed),
            parity_tolerance=glocal_terminal_parity_tolerance,
            terminal_alpha=float(glocal_terminal_alpha),
            terminal_sampling_design=glocal_terminal_sampling_design,
            terminal_population_size=glocal_terminal_population_size,
        )
        validation_requirements = {
            "plan_hash": plan.content_hash,
            "profile_hash": profile.content_hash,
            "promotion_receipt_hash": promotion_receipt.content_hash,
            "query_ids": query_ids,
        }
        if glocal_terminal_parity_tolerance is None:
            final_validation.require_passed(**validation_requirements)
        else:
            final_validation.require_glocal_claim_ready(**validation_requirements)
        store.seal_contract(
            contract_kind="sampled_path_final_validation",
            content_hash=final_validation.content_hash,
            payload=final_validation.to_dict(),
        )
    with profiler.phase("circuit.artifact_materialization"):
        manifest, bundle = materialize_native_sampled_path_circuits(
            plan=plan,
            store=store,
            runs_by_query=runs,
            route_blocks=loaded.architecture.circuit_route_blocks,
            staging_dir=staging_dir,
            promotion_receipt=promotion_receipt,
            final_validation_receipt=final_validation,
            preservation_by_query=final_validation.preservation_by_query,
            valid_for_claims=True,
            partition_label_provider_id=runtime.partition_label_provider_id,
            partition_label_hash=runtime.partition_label_hash,
        )
        store.mark_analysis_ready(
            manifest,
            actor="mechanistic_systems.sampled_path_final_validation.v1",
        )
        store.mark_claim_ready(
            manifest,
            actor="mechanistic_systems.sampled_path_final_validation.v1",
        )
    wall_seconds = float(time.perf_counter() - started)
    proposal_count = sum(
        len(run.result.steps) for query_runs in runs.values() for run in query_runs
    )
    receipt = RuntimeReceipt(
        artifact_hash=manifest.artifact_hash,
        work_packet_hash=packet_hash,
        stage="sampled_path_frontier_v1_native_panel",
        partition=str(partition),
        wall_seconds=wall_seconds,
        throughput_items_per_second=(
            float(proposal_count) / wall_seconds if wall_seconds > 0.0 else None
        ),
        peak_hbm_gb=_cuda_peak_gb(device),
        cache_ready=True,
        details={
            **profiler.snapshot(),
            "algorithm_version": "sampled_path_frontier_v1",
            "profile_hash": profile.content_hash,
            "promotion_receipt_hash": promotion_receipt.content_hash,
            "dense_anchor_evidence_hash": resolved_dense_anchor_hash,
            "dense_anchor_source": dense_anchor_source,
            "query_count": len(query_ids),
            "query_ids": list(query_ids),
            "layer_group_count": len(route_groups),
            "proposal_count": proposal_count,
            "partition_label_provider_id": runtime.partition_label_provider_id,
            "partition_label_hash": runtime.partition_label_hash,
            "final_validation_receipt_hash": final_validation.content_hash,
            "optimization_context_source": optimization_source.source_id,
            "evaluation_context_source": evaluation_source.source_id,
            "optimization_replica_receipt_hash": getattr(
                getattr(optimization_source, "replica_receipt", None),
                "content_hash",
                None,
            ),
            "evaluation_replica_receipt_hash": getattr(
                getattr(evaluation_source, "replica_receipt", None),
                "content_hash",
                None,
            ),
            "response_cache": backend.response_cache_profile(),
            "candidate_execution": backend.runtime_profile(),
            "analysis_status": "claim_ready",
        },
    )
    receipt_path = store.write_runtime_receipt(receipt)
    return {
        "schema_name": "mechanistic_registered_sampled_path_panel_run_v1",
        "artifact_hash": manifest.artifact_hash,
        "bundle": str(bundle),
        "study_hash": study.content_hash,
        "plan_hash": plan.content_hash,
        "profile_hash": profile.content_hash,
        "promotion_receipt_hash": promotion_receipt.content_hash,
        "dense_anchor_evidence_hash": resolved_dense_anchor_hash,
        "dense_anchor_source": dense_anchor_source,
        "query_ids": list(query_ids),
        "final_validation_receipt_hash": final_validation.content_hash,
        "runtime_receipt": str(receipt_path),
        "wall_seconds": wall_seconds,
        "proposal_count": proposal_count,
        "partition_label_provider_id": runtime.partition_label_provider_id,
        "partition_label_hash": runtime.partition_label_hash,
        "response_cache_profile": backend.response_cache_profile(),
        "candidate_execution_profile": backend.runtime_profile(),
        "valid_for_claims": True,
        "analysis_status": "claim_ready",
    }


__all__ = [
    "RegisteredCircuitRuntime",
    "materialize_registered_circuits",
    "naturalized_batch_size",
    "materialize_registered_sampled_path_circuits",
    "resolve_registered_circuit_runtime",
    "run_registered_sampled_path_group",
    "verify_registered_sampled_path_prefix",
]
