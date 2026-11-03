"""Registered-model execution for canonical MeasurementArtifacts."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from circuitflow.artifacts import ArtifactStore, RuntimeReceipt
from contracts.hashing import stable_digest
from mechanistic_systems.contracts import MeasurementPlan, StudySpec
from mechanistic_systems.measurement.contexts import (
    SCRATCH_FOLDER_LOADER_CONTRACT,
    ContextAxis,
    ImageFolderMeasurementProvider,
    build_context_source,
)
from mechanistic_systems.measurement.estimators import ROUTE_GATE_V1
from mechanistic_systems.measurement.models import (
    LoadedRegisteredModel,
)
from mechanistic_systems.measurement.model_sources import (
    ModelSourceReplica,
    load_model_for_study,
)
from mechanistic_systems.measurement.production import (
    NativeLayerBlockedMeasurementAdapter,
)
from mechanistic_systems.core.numerics import (
    exact_float32_numerics,
)
from mechanistic_systems.measurement.scorestore import (
    lower_measurement_to_scorestores,
)
from mechanistic_systems.measurement.streaming import (
    materialize_streamed_measurement_artifact,
)
from mechanistic_systems.measurement.torch_adapter import (
    TorchModuleMeasurementAdapter,
)

try:  # Optional for import smoke; required by registered execution.
    import torch
except Exception:  # pragma: no cover - environment dependent.
    torch = None  # type: ignore[assignment]


def validate_context_axis_for_study(axis: ContextAxis, study: StudySpec) -> None:
    """Require every model-facing image axis to implement the Study operator."""

    axis.validate()
    study.validate().require_valid()
    issues: list[str] = []
    if axis.label_map_hash != study.label_map_hash:
        issues.append("label_map_hash")
    # Runtime replicas and cache formats may vary by stage, but the pixel-valued
    # scientific operator may not. Otherwise measurement and identification
    # refer to different systems.
    if axis.preprocessing.to_dict() != dict(study.preprocessing):
        issues.append("preprocessing")
    declared_dataset = str(study.data_source.get("dataset_id", ""))
    if declared_dataset and axis.dataset_id != declared_dataset:
        issues.append("dataset_id")
    if issues:
        raise ValueError("context axis differs from StudySpec: " + ",".join(issues))


def _cuda_peak_gb(device: Any) -> float | None:
    if torch is None:
        return None
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated(resolved) / (1024**3))


_CONTEXT_FACTOR_MODE_ALIASES = {
    "1": "sketch",
    "true": "sketch",
    "yes": "sketch",
    "sketch": "sketch",
    "marginal": "channel_marginal",
    "channel_marginal": "channel_marginal",
    "native": "native",
}


def _retention_flags() -> dict[str, Any]:
    """Pilot retention tiers, env-bridged (additive only).

    Each flag only ADDS datasets beside the pooled arrays (byte-identical
    with flags off).
    ``CIRCUIT_RETAIN_CONTEXT_FACTOR`` is MODE-VALUED (both
    large-R modes, sketch default): ``sketch`` (also
    1/true/yes), ``marginal``/``channel_marginal``, or ``native``;
    ``CIRCUIT_SKETCH_PROBES``/``CIRCUIT_SKETCH_SEED`` refine sketch mode.
    The covariance tee still REFUSES on block-mean production tiles —
    derive covariance from a NATIVE factor store instead.
    """

    import os

    def _on(name: str) -> bool:
        return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}

    flags: dict[str, Any] = {
        "retain_conditional_statistics": _on("CIRCUIT_RETAIN_CONDITIONAL"),
        "retain_covariance_blocks": _on("CIRCUIT_RETAIN_COVARIANCE"),
        "retain_context_factor": False,
    }
    raw = os.environ.get("CIRCUIT_RETAIN_CONTEXT_FACTOR", "").strip().lower()
    if raw:
        mode = _CONTEXT_FACTOR_MODE_ALIASES.get(raw)
        if mode is None:
            raise ValueError(
                "CIRCUIT_RETAIN_CONTEXT_FACTOR must be one of "
                "sketch|marginal|native (or 1/true/yes for the sketch "
                f"default); got {raw!r}"
            )
        flags["retain_context_factor"] = True
        flags["context_factor_mode"] = mode
        flags["sketch_probes"] = int(
            os.environ.get("CIRCUIT_SKETCH_PROBES", "64")
        )
        flags["sketch_seed"] = int(os.environ.get("CIRCUIT_SKETCH_SEED", "0"))
    return flags


def _adapter_retention_kwargs(flags: dict[str, Any]) -> dict[str, Any]:
    """Subset of retention flags the production adapter needs for capture."""

    if not flags.get("retain_context_factor"):
        return {}
    return {
        "retain_context_factor": True,
        "context_factor_mode": flags["context_factor_mode"],
        "sketch_probes": flags["sketch_probes"],
        "sketch_seed": flags["sketch_seed"],
    }


def _reset_cuda_peak(device: Any) -> None:
    if torch is None:
        return
    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(resolved)


def _fallback_reasons(details: Mapping[str, Any]) -> tuple[str, ...]:
    reasons: set[str] = set()
    for row in details.get("tiles", ()) or ():
        if not isinstance(row, Mapping):
            continue
        reasons.update(str(value) for value in row.get("fallback_reasons", ()) or () if value)
    return tuple(sorted(reasons))


def _write_new_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"immutable run report already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return destination


def _validate_route_gate_v1_registration(
    *,
    study: StudySpec,
    plan: MeasurementPlan,
    context_axis: ContextAxis,
    autograd_backend: str,
) -> None:
    if plan.status != "frozen":
        raise ValueError("registered route_gate_v1 execution requires a frozen MeasurementPlan")
    if plan.estimator_version != ROUTE_GATE_V1:
        raise ValueError("registered route-gate execution requires estimator_version=route_gate_v1")
    if autograd_backend != "is_grads_batched":
        raise ValueError("production route_gate_v1 requires the is_grads_batched backend")
    if (
        str(context_axis.selector.get("runtime_loader_contract", ""))
        != SCRATCH_FOLDER_LOADER_CONTRACT
    ):
        raise ValueError("route_gate_v1 requires the frozen scratch-folder uint8 loader contract")
    declared_images = plan.notes.get("score_images_per_class")
    if declared_images is not None and int(declared_images) == 16:
        if len(plan.context_block_ids) != 4:
            raise ValueError("the historical 16-image route_gate_v1 setting requires four blocks")
    if plan.study_hash != study.content_hash:
        raise ValueError("MeasurementPlan study hash differs from StudySpec")
    if plan.measurement_view.view_hash != context_axis.data_view().view_hash:
        raise ValueError("MeasurementPlan DataView differs from the context axis")
    if tuple(plan.context_block_ids) != context_axis.context_block_ids:
        raise ValueError("MeasurementPlan context blocks differ from the context axis")
    validate_context_axis_for_study(context_axis, study)


def materialize_registered_route_gate_v1(
    *,
    study: StudySpec,
    plan: MeasurementPlan,
    context_axis: ContextAxis,
    image_root: str | Path,
    store: ArtifactStore,
    staging_dir: str | Path,
    report_path: str | Path,
    checkpoint_path: str | Path | None = None,
    allow_download: bool = False,
    device: Any = "cpu",
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = True,
    query_tile_size: int = 2,
    autograd_backend: str = "is_grads_batched",
    partition: str = "local",
    work_packet_hash: str = "",
    scorestore_root: str | Path | None = None,
    cell_id: str = "",
    loaded_model: LoadedRegisteredModel | None = None,
    model_source_replica: ModelSourceReplica | None = None,
) -> dict[str, Any]:
    """Execute a frozen registered ``route_gate_v1`` measurement end to end.

    It evaluates the frozen image axis into four-block sufficient statistics.
    Runtime batching and
    accelerator choice are recorded but do not enter scientific identity.
    """

    if torch is None:
        raise RuntimeError("registered route_gate_v1 execution requires PyTorch")
    if int(batch_size) <= 0 or int(query_tile_size) <= 0:
        raise ValueError("batch and query tile sizes must be positive")
    if Path(report_path).exists():
        raise FileExistsError(f"immutable run report already exists: {report_path}")
    if (scorestore_root is None) != (not bool(cell_id)):
        raise ValueError("ScoreStore lowering requires both scorestore_root and cell_id")
    _validate_route_gate_v1_registration(
        study=study,
        plan=plan,
        context_axis=context_axis,
        autograd_backend=str(autograd_backend),
    )
    loaded = loaded_model or load_model_for_study(
        study,
        source_replica=model_source_replica,
        checkpoint_path=checkpoint_path,
        allow_download=allow_download,
        device=device,
        claim_bearing=True,
    )
    loaded.validate_against_study(study, claim_bearing=True)
    loaded.model.to(device=device, dtype=torch.float32)
    loaded.model.eval()
    source = build_context_source(
        context_axis,
        image_root,
        batch_size=int(batch_size),
        device=device,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
    )
    provider = ImageFolderMeasurementProvider(source)
    torch.manual_seed(int(plan.random_seed))
    with exact_float32_numerics() as numerical_policy:
        numerical_policy["canonical_execution_batch_size"] = int(batch_size)

        model_adapter = TorchModuleMeasurementAdapter.from_bound_architecture(
            loaded.architecture,
            model=loaded.model,
            batch_provider=provider,
            device=device,
            autocast_dtype=None,
        )
        retention = _retention_flags()
        adapter = NativeLayerBlockedMeasurementAdapter(
            model_adapter,
            query_tile_size=int(query_tile_size),
            autograd_backend="is_grads_batched",
            **_adapter_retention_kwargs(retention),
        )
        store.seal_contract(
            contract_kind="study",
            content_hash=study.content_hash,
            payload=study.to_dict(),
        )
        store.seal_contract(
            contract_kind="context_axis",
            content_hash=context_axis.content_hash,
            payload=context_axis.to_dict(),
        )
        store.seal_contract(
            contract_kind="model_load_receipt",
            content_hash=loaded.receipt.content_hash,
            payload=loaded.receipt.to_dict(),
        )
        packet_hash = work_packet_hash or stable_digest(
            {
                "stage": "route_gate_v1",
                "plan_hash": plan.content_hash,
                "context_axis_hash": context_axis.content_hash,
                "model_load_receipt_hash": loaded.receipt.content_hash,
                "device": str(device),
                "batch_size": int(batch_size),
                "query_tile_size": int(query_tile_size),
            }
        )
        _reset_cuda_peak(device)
        started = time.perf_counter()
        manifest, bundle = materialize_streamed_measurement_artifact(
            plan,
            adapter,
            store,
            staging_dir=staging_dir,
            **retention,
        )
        wall_seconds = float(time.perf_counter() - started)
    runtime_details = adapter.runtime_details()
    reasons = _fallback_reasons(runtime_details)
    if reasons:
        raise RuntimeError(
            "route_gate_v1 production backend used a forbidden fallback: "
            + ";".join(reasons)
        )
    measurement_zarr = (
        bundle / "payloads" / "measurement_sufficient_statistics.zarr"
    )
    scorestore_manifests: list[str] = []
    if scorestore_root is not None:
        lowered = lower_measurement_to_scorestores(
            measurement_zarr=measurement_zarr,
            scorestore_root=scorestore_root,
            architecture_family=loaded.receipt.architecture_family,
            cell_id=str(cell_id),
            source_metadata={
                "schema_name": "registered_route_gate_v1_scorestore_view_v1",
                "study_hash": study.content_hash,
                "measurement_plan_hash": plan.content_hash,
                "measurement_artifact_hash": manifest.artifact_hash,
                "model_state_hash": loaded.receipt.state_dict_hash,
                "topology_hash": loaded.architecture.topology_hash,
            },
        )
        scorestore_manifests = [str(item.manifest_path) for item in lowered]
    receipt = RuntimeReceipt(
        artifact_hash=manifest.artifact_hash,
        work_packet_hash=packet_hash,
        stage="measurement.route_gate_v1",
        partition=str(partition),
        wall_seconds=wall_seconds,
        throughput_items_per_second=(
            float(len(context_axis.records) * len(plan.queries)) / wall_seconds
            if wall_seconds > 0.0
            else None
        ),
        peak_hbm_gb=_cuda_peak_gb(device),
        fallback_reason="",
        replay_parity_passed=None,
        cache_ready=True,
        details={
            **runtime_details,
            "numerical_policy": numerical_policy,
            "context_axis_hash": context_axis.content_hash,
            "model_load_receipt_hash": loaded.receipt.content_hash,
            "context_count": len(context_axis.records),
            "query_count": len(plan.queries),
        },
    )
    receipt_path = store.write_runtime_receipt(receipt)
    context_payload = context_axis.to_dict()
    plan_payload = plan.to_dict()
    selector = dict(plan_payload.get("measurement_view", {}).get("selector", {}) or {})
    block_windows = tuple(selector.get("block_windows", ()) or ())
    samples_per_class = sum(int(row.get("count", 0)) for row in block_windows)
    target_ids = [str(query.query_id) for query in plan.queries]
    scorestore_query_ids = {target: target for target in target_ids}
    route_notes = dict(study.route_schema.notes)
    report = {
        "schema_name": "mechanistic_registered_route_gate_v1_run_v2",
        "status": "completed",
        "artifact_hash": manifest.artifact_hash,
        "measurement_artifact_hash": manifest.artifact_hash,
        "bundle": str(bundle),
        "artifact_root": str(store.root),
        "measurement_zarr": str(measurement_zarr),
        "study_hash": study.content_hash,
        "plan_hash": plan.content_hash,
        "measurement_plan_hash": plan.content_hash,
        "context_axis_hash": context_axis.content_hash,
        "model_load_receipt_hash": loaded.receipt.content_hash,
        "model_state_hash": loaded.receipt.state_dict_hash,
        "topology_hash": loaded.architecture.topology_hash,
        "route_schema_hash": loaded.receipt.route_schema_hash,
        "architecture_family": loaded.receipt.architecture_family,
        "preprocessing_hash": stable_digest(context_payload["preprocessing"]),
        "query_contract_hash": stable_digest([query.to_dict() for query in plan.queries]),
        "target_ids": target_ids,
        "scorestore_query_ids": scorestore_query_ids,
        "cell_id": str(cell_id),
        "scorestore_root": str(Path(scorestore_root).resolve())
        if scorestore_root is not None
        else "",
        "numerical_policy": numerical_policy,
        "loader": {
            "class_count": int(selector.get("class_count", 0)),
            "samples_per_class": int(samples_per_class),
            "observed_context_count": len(context_axis.records),
            "ordered_source_record_count": len(context_axis.records),
            "ordered_source_records_hash": stable_digest(context_payload["records"]),
        },
        "estimator": {
            "estimator_version": ROUTE_GATE_V1,
            "canonical_execution_batch_size": int(batch_size),
            "numerical_policy": numerical_policy,
        },
        "runtime": runtime_details,
        "profile_audit": {
            "fixed_readout_layers": tuple(
                str(value)
                for value in route_notes.get("fixed_measurement_layer_ids", ())
            )
        },
        "runtime_receipt": str(receipt_path),
        "scorestore_manifests": scorestore_manifests,
        "wall_seconds": wall_seconds,
        "valid_for_claims": bool(manifest.valid_for_claims),
    }
    destination = _write_new_json(report_path, report)
    return {**report, "report_path": str(destination)}


def materialize_registered_measurement(
    *,
    study: StudySpec,
    plan: MeasurementPlan,
    context_axis: ContextAxis,
    image_root: str | Path,
    store: ArtifactStore,
    staging_dir: str | Path,
    checkpoint_path: str | Path | None = None,
    allow_download: bool = False,
    device: Any = "cpu",
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = True,
    query_tile_size: int = 8,
    autograd_backend: str = "auto",
    parity_receipt_hash: str = "",
    autocast_dtype: Any = None,
    partition: str = "local",
    work_packet_hash: str = "",
    loaded_model: LoadedRegisteredModel | None = None,
    model_source_replica: ModelSourceReplica | None = None,
) -> dict[str, Any]:
    """Execute the registered exploratory scorer and seal sufficient statistics.

    Frozen ``route_gate_v1`` studies use the canonical ``measure
    run-route-gate-v1`` surface. Historical oracle execution is administrative
    Historical score-oracle execution is retired; new science uses
    ``measure run-route-gate-v1``.
    Keeping those authorities explicit prevents an exploratory VJP path from
    silently replacing the sealed data and reduction contract.
    """

    if plan.status in {"frozen", "legacy_matched"}:
        raise ValueError(
            "frozen route_gate_v1 measurements require `measure "
            "run-route-gate-v1`; legacy oracle execution requires CircuitFlow "
            "historical score-oracle execution is retired; use "
            "`measure run-route-gate-v1`"
        )

    if plan.study_hash != study.content_hash:
        raise ValueError("MeasurementPlan study hash differs from StudySpec")
    if plan.measurement_view.view_hash != context_axis.data_view().view_hash:
        raise ValueError("MeasurementPlan DataView differs from the context axis")
    if tuple(plan.context_block_ids) != context_axis.context_block_ids:
        raise ValueError("MeasurementPlan context blocks differ from the context axis")
    validate_context_axis_for_study(context_axis, study)
    loaded = loaded_model or load_model_for_study(
        study,
        source_replica=model_source_replica,
        checkpoint_path=checkpoint_path,
        allow_download=allow_download,
        device=device,
        claim_bearing=plan.status == "frozen",
    )
    loaded.validate_against_study(study, claim_bearing=plan.status == "frozen")
    source = build_context_source(
        context_axis,
        image_root,
        batch_size=int(batch_size),
        device=device,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
    )
    model_adapter = TorchModuleMeasurementAdapter.from_bound_architecture(
        loaded.architecture,
        model=loaded.model,
        batch_provider=ImageFolderMeasurementProvider(source),
        device=device,
        parity_receipt_hash=parity_receipt_hash,
        autocast_dtype=autocast_dtype,
    )
    retention = _retention_flags()
    adapter = NativeLayerBlockedMeasurementAdapter(
        model_adapter,
        query_tile_size=int(query_tile_size),
        autograd_backend=autograd_backend,  # type: ignore[arg-type]
        **_adapter_retention_kwargs(retention),
    )
    store.seal_contract(
        contract_kind="study",
        content_hash=study.content_hash,
        payload=study.to_dict(),
    )
    store.seal_contract(
        contract_kind="model_load_receipt",
        content_hash=loaded.receipt.content_hash,
        payload=loaded.receipt.to_dict(),
    )
    packet_hash = work_packet_hash or stable_digest(
        {
            "stage": "measurement",
            "plan_hash": plan.content_hash,
            "context_axis_hash": context_axis.content_hash,
            "model_load_receipt_hash": loaded.receipt.content_hash,
            "device": str(device),
            "batch_size": int(batch_size),
            "query_tile_size": int(query_tile_size),
            "autograd_backend": str(autograd_backend),
        }
    )
    _reset_cuda_peak(device)
    started = time.perf_counter()
    manifest, bundle = materialize_streamed_measurement_artifact(
        plan,
        adapter,
        store,
        staging_dir=staging_dir,
        **retention,
    )
    wall_seconds = float(time.perf_counter() - started)
    runtime_details = adapter.runtime_details()
    reasons = _fallback_reasons(runtime_details)
    context_count = len(context_axis.records)
    query_count = len(plan.queries)
    receipt = RuntimeReceipt(
        artifact_hash=manifest.artifact_hash,
        work_packet_hash=packet_hash,
        stage="measurement",
        partition=str(partition),
        wall_seconds=wall_seconds,
        throughput_items_per_second=(
            float(context_count * query_count) / wall_seconds if wall_seconds > 0.0 else None
        ),
        peak_hbm_gb=_cuda_peak_gb(device),
        fallback_reason=";".join(reasons),
        cache_ready=True,
        details={
            **runtime_details,
            "context_axis_hash": context_axis.content_hash,
            "model_load_receipt_hash": loaded.receipt.content_hash,
            "context_count": context_count,
            "query_count": query_count,
        },
    )
    receipt_path = store.write_runtime_receipt(receipt)
    return {
        "schema_name": "mechanistic_registered_measurement_run_v1",
        "artifact_hash": manifest.artifact_hash,
        "bundle": str(bundle),
        "study_hash": study.content_hash,
        "plan_hash": plan.content_hash,
        "context_axis_hash": context_axis.content_hash,
        "model_load_receipt_hash": loaded.receipt.content_hash,
        "runtime_receipt": str(receipt_path),
        "wall_seconds": wall_seconds,
        "valid_for_claims": bool(manifest.valid_for_claims),
    }


__all__ = [
    "materialize_registered_measurement",
    "materialize_registered_route_gate_v1",
    "validate_context_axis_for_study",
]
