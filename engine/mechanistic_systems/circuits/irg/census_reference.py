"""Campaign census: one dense forward, separate target VJPs, existing looks.

The intermediate bank stores forward captures once and each target's
cotangents, rather than context-by-route attribution rows for every target.
The ordinary scoring reducers and census controller consume that bank. Only
the six ordinary census fields are needed by a bound pruning run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass, replace
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from contracts.hashing import sha256_file
from mechanistic_systems.core.numerics import (
    CachedQueryVJP, CapturedRouteBlock, VJPTelemetry, exact_batched_vjp,
    numerics_record, run_forward_numerics,
)
from mechanistic_systems.measurement.production import CapturedMeasurementBatch
from mechanistic_systems.measurement.queries import evaluate_queries
from mechanistic_systems.measurement.optimized_irg_providers import (
    OptimizedScoringResponseProvider, alexnet_v1_conserving_activation_bindings,
    resnet_local_destination_transport, resnet_local_regime_capture_specs,
)
from mechanistic_systems.measurement.verified_once import read_verified_npz
from mechanistic_systems.measurement.pruning_loss import (
    CALIBRATED_OVR_BCE, CALIBRATED_SUBSTITUTED_SOFTMAX,
    PRUNING_GUIDING_LOSSES, dense_calibration_state_hash,
)
from .modern_v1_conserving_route_score import V1ConservingActivationCapture

CENSUS_MODE_ENV = "IRG_CENSUS_MODE"
CENSUS_REFERENCE_ENV = "IRG_CENSUS_REFERENCE"
CENSUS_REFERENCE_HASH_ENV = "IRG_CENSUS_REFERENCE_HASH"
SCHEMA = "campaign_census_reference_v2"
DEFAULT_PRUNING_GUIDING_LOSS_ID = CALIBRATED_OVR_BCE


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, SimpleNamespace):
        return _plain(vars(value))
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def reference_settings(path=None, reference_hash=None, mode=None):
    """Default to a supplied reference; one setting restores computation."""
    selected = str(os.environ.get(CENSUS_MODE_ENV, "reference") if mode is None else mode).strip()
    if selected not in {"reference", "per_category"}:
        raise ValueError(f"{CENSUS_MODE_ENV} must be reference or per_category")
    path = path or os.environ.get(CENSUS_REFERENCE_ENV)
    reference_hash = reference_hash or os.environ.get(CENSUS_REFERENCE_HASH_ENV)
    if selected == "per_category":
        return None, None
    if bool(path) != bool(reference_hash):
        raise ValueError("census reference requires both its path and content hash")
    if reference_hash and (len(reference_hash) != 64 or any(
        c not in "0123456789abcdef" for c in reference_hash
    )):
        raise ValueError("census reference content hash must be a SHA-256 digest")
    return path, reference_hash


def _require_dense_mask(mask):
    from .candidates import HardMask
    if not isinstance(mask, HardMask) or mask.retained_count != mask.route_count:
        raise ValueError("campaign census requires the full dense HardMask")
    return mask


def _source_identity(request):
    runtime = request["runtime"]
    mask = _require_dense_mask(request["dense_mask"])
    return {
        "model_state_hash": runtime.loaded.receipt.state_dict_hash,
        "topology_hash": runtime.evaluator.topology_hash,
        "route_schema_hash": runtime.evaluator.route_schema_hash,
        "context_axis_hash": request["optimization_axis"].content_hash,
        "dense_mask_hash": mask.mask_hash,
        "execution_profile_hash": _digest(getattr(runtime.evaluator, "runtime_execution_profile", None)),
        "device_type": torch.device(runtime.evaluator.device).type,
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "forward_precision": {
            "cuda_matmul_tf32": str(os.environ.get("CIRCUIT_ALLOW_TF32", "1")).strip() != "0",
            "cudnn_tf32": str(os.environ.get("CIRCUIT_ALLOW_TF32", "1")).strip() != "0",
        },
    }


def _census_materiality_needs_shared_forward(request):
    provider = request.get("anchor_provider")
    fit = request.get("dense_calibration_state")
    if provider is None or fit is None or not bool(getattr(fit, "full_census", False)):
        return False
    axis = provider._oracle_stratum_axis()
    key = (dense_calibration_state_hash(fit), axis.names)
    return (not provider.canonical_dense_reference_active
            and key not in getattr(provider, "_census_dense_loss_variance_cache", {}))


def _bind_census_materiality_resolution(request):
    """Bind the locked anchor's loss resolution before hashing or measuring it."""
    provider = request.get("anchor_provider")
    if (provider is not None and request.get("census_materiality_resolution") is None
        and not _census_materiality_needs_shared_forward(request)):
        request["census_materiality_resolution"] = provider.census_materiality_resolution(
            request["dense_mask"],
            confidence_level=float(request["sample_size_policy"].confidence_level),
            initial_total_count=max(
                4, int(request["runtime"].evaluator.runtime_execution_profile.v0_paired_oracle_batch_rows)
            ),
        )


def category_cache_key(request):
    from . import registered_resnet18_pilot as pilot
    names = (
        "runtime", "optimization_axis", "query", "dense_mask", "run_id",
        "random_seed", "sample_size_policy", "required_partition_views",
        "partition_family", "partition_random_seed", "prune_order_scope",
        "initial_partition", "dense_calibration_state",
    )
    _census_objective(request)
    _bind_census_materiality_resolution(request)
    return pilot._hard_partition_cache_key(
        **{name: request[name] for name in names},
        enable_local_regime_diagnostics=request.get("enable_local_regime_diagnostics", False),
        v1_ordering_key=request.get("v1_ordering_key", "abs_raw"),
        v0_attribution_order=request.get("v0_attribution_order", pilot.V0_DEFAULT_ATTRIBUTION_ORDER),
        pruning_guiding_loss_id=request.get("pruning_guiding_loss_id", DEFAULT_PRUNING_GUIDING_LOSS_ID),
        census_profile=request.get("census_profile"),
        census_proposal_route_cap=request.get("census_proposal_route_cap"),
        census_schedule_replay_schedule=request.get("census_schedule_replay_schedule"),
        census_materiality_resolution=request.get("census_materiality_resolution"),
        census_sample_design=request.get("census_sample_design"),
    )


def prepare_registered_census_requests(
    *, runtime, optimization_axis, dense_mask, queries, comparison_namespace,
    sample_size_policy, initial_partition, partition_family="local_attribution_sign",
    random_seed=0, partition_random_seed=0, v0_attribution_order=None,
    calibration_deployment_refit_anchor="selected", v0_oracle_minimum_per_stratum=2,
    v0_oracle_sampling_law_override=None, step_category_matrix=False,
    empty_certificate=None, enable_local_regime_diagnostics=False,
    pruning_guiding_loss_id=DEFAULT_PRUNING_GUIDING_LOSS_ID,
    census_profile=None, census_proposal_route_cap=None, census_schedule_replay_schedule=None,
):
    """Prepare many v0 categories using the run's actual namespaces/providers.

    The caller resolves one ordinary registered runtime and dense mask. No
    images are acquired here. Every consumer uses P's registered acquisition
    settings. Pass the same campaign, profile, policy, query and execution
    options to the pruning runs that will bind this reference.
    """
    from . import registered_resnet18_pilot as pilot
    from mechanistic_systems.experimental_design.sampling_identity import (
        binary_evidence_role_namespace, compile_binary_iid_sampling_design,
    )
    from mechanistic_systems.inference.sequential_bce import (
        SequentialCalibratedBCEResolutionSpec, V0_NORMALIZED_LOSS_SPAN_PER_CONNECTION_V1,
    )
    from mechanistic_systems.measurement.run_products import pruning_functional

    functional = pruning_functional()
    if functional == "substituted_softmax":
        pruning_guiding_loss_id = CALIBRATED_SUBSTITUTED_SOFTMAX
    family = pilot._canonical_partition_family(str(partition_family))
    views = pilot._required_partition_views(family)
    if not views:
        raise ValueError("this partition family does not acquire a census")
    policy = pilot._registered_precision_policy(
        "v0", sample_size_policy,
        oracle_sampling_law_override=v0_oracle_sampling_law_override,
    )
    route_ids, route_layers = pilot._global_route_axes(runtime)
    confidence = SequentialCalibratedBCEResolutionSpec(
        confidence_level=policy.confidence_level,
        estimand_policy=V0_NORMALIZED_LOSS_SPAN_PER_CONNECTION_V1,
        maximum_planned_looks=None, precision_only_iid=True,
    ).v0_anchor_confidence_level
    if functional == "target_logit":
        empty_certificate = empty_certificate or _registered_empty_certificate(
            runtime, dense_mask, len(route_ids), confidence)
    if empty_certificate is not None and float(empty_certificate.confidence_level) != float(confidence):
        raise ValueError("census empty anchor confidence differs from the registered risk law")
    _mode, table_path, population_path = pilot._resolved_dense_baseline_authority(
        architecture_family=runtime.loaded.receipt.architecture_family,
        model_id=getattr(runtime.loaded.receipt, "model_id", None))
    table = None
    if table_path:
        table = pilot._load_and_validate_canonical_dense_readout_table(
            receipt_path=table_path, model_state_hash=runtime.loaded.receipt.state_dict_hash,
            route_schema_hash=runtime.evaluator.route_schema_hash,
            topology_hash=runtime.evaluator.topology_hash, dense_mask_hash=dense_mask.mask_hash,
            context_axis_hash=optimization_axis.content_hash,
            context_axis_split=str(optimization_axis.split),
            forward_tier=int(runtime.evaluator.runtime_execution_profile.forward_tier))
    result = []
    for query in queries:
        target = pilot._target_class_id(query)
        run_id = pilot.content_hash({
            "comparison_namespace": str(comparison_namespace), "query_id": str(query.query_id),
            "optimization_axis_hash": optimization_axis.content_hash,
            "random_seed": int(random_seed),
        }, prefix="irg-matched-pilot-")
        namespace = binary_evidence_role_namespace(
            run_id=run_id, random_seed=random_seed, context_axis_hash=optimization_axis.content_hash)
        sampling = compile_binary_iid_sampling_design(
            optimization_axis, target_class_id=target, randomization_namespace=namespace)
        oracle_sampling = pilot._decision_oracle_sampling_design(
            optimization_axis, target_class_id=target,
            randomization_namespace=f"{namespace}:class-resolved-oracle",
            oracle_sampling_law=policy.oracle_sampling_law)
        provider = pilot._fixed_calibration_bce_provider(
            runtime=runtime, optimization_axis=optimization_axis, target_class_id=target,
            maximum_bce_increment=None, run_id=run_id, random_seed=random_seed,
            mask_resolver=lambda key: dense_mask if key == dense_mask.mask_hash else None,
            candidate_resolver=lambda key: None, sample_size_policy=policy,
            iid_sampling_design=sampling, oracle_iid_sampling_design=oracle_sampling,
            calibration_deployment_refit_anchor=calibration_deployment_refit_anchor,
            oracle_minimum_per_stratum=v0_oracle_minimum_per_stratum,
            calibration_fold_ownership=False, step_category_matrix=step_category_matrix)
        if str(pruning_guiding_loss_id) != provider.pruning_guiding_loss_id:
            raise ValueError("registered census loss must match its dense calibration decision law")
        if table is not None:
            provider.load_canonical_dense_reference(table.handle, identity={
                "receipt_path": table.receipt_path, "payload_sha256": table.payload_sha256,
                "execution_policy": table.execution_policy, "dense_mask_hash": table.dense_mask_hash})
        design = pilot.census_sample_design_setting()
        if design is not None and design.get("design") == pilot.CENSUS_SAMPLE_DESIGN_IMPORTANCE_BATCHES:
            raise ValueError("campaign census references are not built for the importance-batch census")
        request = dict(
            census_schedule_replay_schedule=census_schedule_replay_schedule,
            census_sample_design=design,
            census_profile=census_profile,
            census_proposal_route_cap=census_proposal_route_cap,
            runtime=runtime, optimization_axis=optimization_axis, dense_mask=dense_mask,
            query=query, target_class_id=target, route_ids=route_ids, route_layer_ids=route_layers,
            dense_calibration_state=None, run_id=run_id, random_seed=random_seed,
            pruning_guiding_loss_id=str(pruning_guiding_loss_id),
            sample_size_policy=policy, required_partition_views=views, partition_family=family,
            partition_random_seed=int(partition_random_seed or 0), prune_order_scope="layerwise",
            initial_partition=str(initial_partition),
            v0_attribution_order=v0_attribution_order or pilot.V0_DEFAULT_ATTRIBUTION_ORDER,
            enable_local_regime_diagnostics=bool(enable_local_regime_diagnostics),
            anchor_provider=provider, anchor_confidence_level=confidence,
            anchor_empty_certificate=empty_certificate,
        )
        closure = None
        if population_path:
            fit, certificate, closure = pilot._load_population_dense_calibration(
                population_path, table_binding=table, target_class_id=target,
                dense_mask_hash=dense_mask.mask_hash, confidence_level=confidence)
            provider.install_population_calibration(fit)
            if functional == "substituted_softmax":
                provider._decision_population_scale_zero_reference = dict(closure["scale_zero_reference"])
            request["dense_calibration_state"] = fit
            request["dense_anchor"] = {
                "request_hash": dense_anchor_request_hash(provider=provider, confidence_level=confidence,
                    **{name: request[name] for name in ("runtime", "optimization_axis", "dense_mask",
                        "target_class_id", "run_id", "random_seed", "sample_size_policy")},
                    pruning_guiding_loss_id=pruning_guiding_loss_id),
                "fit": fit, "certificate": certificate, "closure_receipt": closure,
            }
        if functional == "substituted_softmax":
            from .candidates import HardMask

            empty = HardMask.from_bool_array(np.zeros(len(route_ids), dtype=bool),
                                             route_axis_hash=dense_mask.route_axis_hash)
            category_empty, _reference = pilot._substituted_scale_zero_risk_certificate(
                provider=provider, reference_mask=empty, confidence_level=confidence,
                population_closure=closure)
            if (empty_certificate is not None
                    and empty_certificate.content_hash != category_empty.content_hash):
                raise ValueError("substituted census empty anchor differs from its category's scale-zero reference")
            request["anchor_empty_certificate"] = category_empty
            request["decision_fixed_offset_resolver"] = provider.decision_fixed_offsets
        result.append(pilot._p_census_settings(**request))
    return result


def _registered_empty_certificate(runtime, dense_mask, route_count, confidence):
    from . import registered_resnet18_pilot as pilot
    from .candidates import HardMask
    from mechanistic_systems.inference.sequential_bce import AnalyticChanceRiskCertificate
    empty = HardMask.from_bool_array(np.zeros(route_count, dtype=bool),
                                     route_axis_hash=dense_mask.route_axis_hash)
    proof = pilot._fully_pruned_constant_response_proof(runtime=runtime, fully_pruned_mask=empty)
    architecture = runtime.loaded.receipt.architecture_family.lower()
    code = ("registered_resnet18_balanced_binary_chance_anchor_v1"
            if architecture in {"resnet18", "alexnet", "alexnet_conv_kernel_weight"}
            else f"registered_{architecture}_balanced_binary_chance_anchor_v1")
    return AnalyticChanceRiskCertificate(mask_hash=empty.mask_hash,
        constant_response_proof_id=proof["content_hash"], code_identity=code, confidence_level=confidence)


def _algorithm_settings():
    from . import registered_resnet18_pilot as pilot
    from mechanistic_systems.core.numerics import resolve_abs_aggregation_mode
    return {
        "census_accumulator_mode": "v6",
        "abs_aggregation_mode": resolve_abs_aggregation_mode(),
        "census_materiality_rule": pilot._CENSUS_MATERIALITY_RULE,
        "census_executable_cut_rule": "count_changing_rungs_in_force_v1",
        "census_alpha_spending_rule": "alpha/((k+1)*(k+2))",
        "consequence_price_short_circuit": pilot._consequence_price_short_circuit_enabled(),
        "local_covariance_sketch_rank": pilot._local_covariance_sketch_rank(),
        **({} if pilot._census_producer() == pilot._CENSUS_ENGINE_ROWS
           else {"census_producer": pilot._census_producer()}),
    }


def _calibration_hash(calibration):
    return dense_calibration_state_hash(calibration)


def _census_objective(request):
    """Bind the loss to the actual dense affine before any census VJP."""
    mask = _require_dense_mask(request["dense_mask"])
    fit = request.get("dense_calibration_state")
    if fit is None:
        raise ValueError("census requires dense calibration before its first backward pass")
    if str(getattr(fit, "mask_hash", "")) != str(mask.mask_hash):
        raise ValueError("census calibration must belong to the dense mask")
    if not bool(getattr(fit, "numerical_converged", getattr(fit, "converged", False))):
        raise ValueError("census requires a converged dense calibration")
    target = int(request["target_class_id"])
    if int(getattr(fit, "target_class_id", target)) != target:
        raise ValueError("census calibration belongs to another target")
    loss_id = str(request.get("pruning_guiding_loss_id", DEFAULT_PRUNING_GUIDING_LOSS_ID))
    if loss_id not in PRUNING_GUIDING_LOSSES:
        raise ValueError(f"unsupported census pruning-guiding loss {loss_id!r}")
    return {"pruning_guiding_loss_id": loss_id,
            "dense_calibration_state_hash": _calibration_hash(fit),
            "dense_calibration_state_id": str(fit.calibration_state_id),
            "order_field": "loss_activity",
            **({"pruning_functional": "substituted_softmax", "loss_weighting": "per_image"}
               if request.get("decision_fixed_offset_resolver") is not None else {})}


def _anchor_acquisition_policy(provider, dense_mask):
    """Identify whether replacing forward execution can move a fit look."""
    fit = provider.calibration_fits.get(dense_mask.mask_hash)
    if fit is not None and fit.full_census:
        return {"law": "population_exhaustive", "fit": _calibration_hash(fit)}
    from mechanistic_systems.measurement import evidence_plane_v6 as v6
    from mechanistic_systems.backends.calibration_runtime import RuntimeExecutionProfile
    if (provider._device.type == "cuda"
        and provider.opportunistic_neyman_alignment
        and isinstance(provider.runtime_execution_profile, RuntimeExecutionProfile)):
        return {"law": "registered_calibration_alignment",
                "quantum": v6.REGISTERED_CALIBRATION_ALIGNMENT_QUANTUM}
    tuner = provider._execution_tuner(role="calibration_fit", lane_count=1)
    if provider._device.type == "cpu" and len(tuner.plans) == 1 and tuner.plans[0].batch_size == 1:
        return {"law": "fixed_single_row_execution", "quantum": 1}
    return {"law": "measured_execution_alignment",
            "plans": [asdict(plan) for plan in tuner.plans]}





def dense_anchor_request_hash(*, provider, runtime, optimization_axis, dense_mask,
                              target_class_id, run_id, random_seed, sample_size_policy,
                              confidence_level,
                              pruning_guiding_loss_id=DEFAULT_PRUNING_GUIDING_LOSS_ID):
    """Bind the actual calibration and risk draws, options, and precision law."""
    return _digest({
        "source": _source_identity(dict(runtime=runtime, optimization_axis=optimization_axis,
                                         dense_mask=dense_mask)),
        "pruning_guiding_loss_id": str(pruning_guiding_loss_id),
        "target_class_id": int(target_class_id), "run_id": run_id,
        "random_seed": int(random_seed), "sample_size_policy": sample_size_policy,
        "confidence_level": float(confidence_level),
        "calibration_policy": provider.calibration_policy,
        "calibration_options": provider.calibration_options,
        "calibration_orders": provider._calibration_orders,
        "oracle_orders": provider._oracle_orders,
        "iid_sampling_design": provider.iid_sampling_design,
        "oracle_iid_sampling_design": provider.oracle_iid_sampling_design,
        "oracle_minimum_per_stratum": provider.oracle_minimum_per_stratum,
        "oracle_sampling_id": provider.oracle_sampling_id,
        "bce_logit_clip": provider.bce_logit_clip,
        "readout_projection_hash": provider._readout_projection_hash,
        "frozen_category_readout_numerics": provider.frozen_category_readout_numerics,
        "fixed_offset_affine_convergence": provider.fixed_offset_affine_convergence,
        "anchor_acquisition_policy": _anchor_acquisition_policy(provider, dense_mask),
        "canonical_dense_reference": {name: (provider._canonical_dense_reference_identity or {}).get(name)
            for name in ("payload_sha256", "execution_policy", "dense_mask_hash")},
    })


def load_dense_anchor(*, reference_path, reference_hash, request_hash, provider, dense_mask,
                      include_binding_metadata=False):
    """Bind an ordinary dense risk certificate and its converged affine."""
    from mechanistic_systems.inference.sequential_bce import BaselineRiskCertificate
    from mechanistic_systems.measurement.native_irg_providers import NativeBCECalibrationFitState
    from mechanistic_systems.measurement.iid_streams import IIDContextDraw
    root, manifest = _read_manifest(reference_path, reference_hash)
    record = manifest.get("dense_anchors", {}).get(request_hash)
    if record is None:
        raise ValueError("census reference dense anchor request key mismatch")
    payload = (root / record["path"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise ValueError("census reference dense anchor payload hash mismatch")
    anchor = json.loads(payload)
    if anchor["request_hash"] != request_hash:
        raise ValueError("census reference dense anchor payload key mismatch")
    # Restore only the source rows already used by this affine/risk prefix.
    # An anchor with none (a population fit) restores no readouts. A sampled
    # anchor needs a working device before any provider state is installed.
    needed = {str(fields["source_context_id"]) for fields in anchor.get("iid_draws", {}).values()}
    needed.update(anchor["fit"]["context_ids"])
    if anchor.get("risk_evidence") is not None:
        needed.update(anchor["risk_evidence"]["context_ids"])
    if needed and provider._device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("census dense anchor source rows require an available CUDA runtime")
    fit_fields = dict(anchor["fit"])
    fit_fields["context_ids"] = tuple(fit_fields["context_ids"])
    fit_fields["allocation_receipts"] = tuple(fit_fields["allocation_receipts"])
    fit = NativeBCECalibrationFitState(**fit_fields)
    certificate = BaselineRiskCertificate(**anchor["certificate"])
    provider.install_census_dense_calibration(fit, certificate, dense_mask=dense_mask)
    if anchor.get("dense_loss_moments") is not None:
        moments = anchor["dense_loss_moments"]
        axis = provider._oracle_stratum_axis()
        fit_hash = dense_calibration_state_hash(fit)
        if (moments["dense_calibration_state_hash"] != fit_hash
            or set(moments["variances"]) != set(axis.names)
            or set(moments["counts"]) != set(axis.names)
            or any(int(moments["counts"][name]) < 2
                   or not np.isfinite(float(moments["variances"][name]))
                   or float(moments["variances"][name]) < 0 for name in axis.names)):
            raise ValueError("census dense loss moments differ from the frozen anchor")
        if not hasattr(provider, "_census_dense_loss_variance_cache"):
            provider._census_dense_loss_variance_cache = {}
        provider._census_dense_loss_variance_cache[(fit_hash, axis.names)] = (
            dict(moments["variances"]), dict(moments["counts"]), str(moments["source"])
        )
    for draw_id, fields in anchor.get("iid_draws", {}).items():
        draw = IIDContextDraw(**fields)
        if draw_id != draw.draw_id:
            raise ValueError("census dense anchor draw identity mismatch")
        provider._iid_draws[draw_id] = draw
    if anchor.get("risk_evidence") is not None:
        evidence = dict(anchor["risk_evidence"])
        for name in ("context_ids", "strata"):
            evidence[name] = tuple(evidence[name])
        provider._baseline_risk_evidence_by_mask[fit.mask_hash] = evidence
        provider._baseline_risk_look_count_by_mask[fit.mask_hash] = int(anchor["risk_look_count"])
    provider._calibration_context_history[fit.mask_hash] = set(anchor.get("calibration_context_history", fit.context_ids))
    if anchor.get("crossfit_disposition") is not None:
        provider._calibration_crossfit_disposition[fit.mask_hash] = anchor["crossfit_disposition"]
    # Separate small logits packets avoid reading dense activation captures.
    seen = set()
    packets = () if provider.canonical_dense_reference_active else manifest["readout_packets"]
    bulk_packets = []
    for packet in packets:
        positions = [i for i, context in enumerate(packet["context_ids"]) if context in needed]
        if not positions:
            continue
        loaded = read_verified_npz(root / packet["path"])
        if loaded.sha256 != packet["sha256"]:
            raise ValueError("census reference dense readout hash mismatch")
        ids = tuple(packet["context_ids"][i] for i in positions)
        values = loaded.arrays["responses"][positions]
        projection = provider._readout_projection
        if projection is not None:
            values = values[:, projection.column_ids]
        bulk_packets.append((ids, np.array(values, copy=True, order="C"),
                             loaded.arrays["labels"][positions]))
        seen.update(ids)
    if bulk_packets:
        from mechanistic_systems.measurement.phase_device_orchestration import preload_anchor_packets
        preload_anchor_packets(provider, dense_mask.mask_hash, bulk_packets)
    source_needed = {draw.source_context_id for draw in provider._iid_draws.values()}
    if provider.canonical_dense_reference_active and source_needed:
        ordered = tuple(sorted(source_needed))
        canonical = provider._canonical_dense_reference_rows(ordered)
        if canonical is None or canonical[0] != tuple(range(len(ordered))):
            raise ValueError("census dense anchor canonical source rows are incomplete")
    elif not source_needed <= seen:
        raise ValueError("census dense anchor source rows are incomplete")
    if include_binding_metadata:
        if anchor.get("span_closure_receipt") is None:
            raise ValueError("registered census reference lacks its positive dense-to-empty span")
        return certificate, dict(anchor["closure_receipt"]), {
            "initial_certificate": BaselineRiskCertificate(**anchor["initial_certificate"]),
            "span_closure_receipt": anchor["span_closure_receipt"],
            "empty_certificate_hash": anchor["empty_certificate_hash"],
        }
    return certificate, dict(anchor["closure_receipt"])


def bound_loss_span(*, binding, dense_certificate, empty_certificate):
    from mechanistic_systems.inference.sequential_bce import LossSpanCertificate
    if binding["empty_certificate_hash"] != empty_certificate.content_hash:
        raise ValueError("census reference fully-pruned anchor key mismatch")
    span = LossSpanCertificate(dense=dense_certificate, fully_pruned=empty_certificate)
    receipt = dict(binding["span_closure_receipt"])
    if (receipt["final_risk_certificate_hash"] != dense_certificate.content_hash
        or receipt["final_loss_span_certificate_hash"] != span.content_hash):
        raise ValueError("census reference loss-span certificate hash mismatch")
    span.validate_positive()
    return span, receipt


def _read_manifest(path, expected_hash):
    path = Path(path)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError("census reference hash mismatch")
    manifest = json.loads(payload)
    if manifest.get("schema") != SCHEMA:
        raise ValueError("census reference schema mismatch")
    if manifest.get("algorithm_settings") != _algorithm_settings():
        raise ValueError("census reference algorithm settings mismatch")
    return path.parent, manifest


def load_category_census(*, reference_path, reference_hash, cache_key, runtime,
                         optimization_axis, dense_mask, query, target_class_id,
                         dense_calibration_state, partition_family,
                         pruning_guiding_loss_id=DEFAULT_PRUNING_GUIDING_LOSS_ID):
    """Refuse a wrong category/key/hash, without a compute fallback."""
    from . import registered_resnet18_pilot as pilot
    objective = _census_objective(dict(dense_calibration_state=dense_calibration_state,
        dense_mask=dense_mask, target_class_id=target_class_id,
        pruning_guiding_loss_id=pruning_guiding_loss_id))
    root, manifest = _read_manifest(reference_path, reference_hash)
    actual = _source_identity(dict(runtime=runtime, optimization_axis=optimization_axis,
                                   dense_mask=dense_mask))
    if manifest["source_identity"] != actual:
        raise ValueError("census reference source key mismatch")
    entry = manifest["categories"].get(cache_key)
    if entry is None or (
        entry["query_hash"] != _digest(query)
        or entry["target_class_id"] != int(target_class_id)
        or entry["census_objective"] != objective
    ):
        raise ValueError("census reference category key, objective or calibration mismatch")
    # Exact same payload and reader as the run's existing partition cache.
    receipt_path = root / "categories" / f"{cache_key}.json"
    if sha256_file(receipt_path) != entry["receipt_sha256"]:
        raise ValueError("census reference category receipt hash mismatch")
    result = pilot._load_hard_partition_cache(
        payload_path=root / "categories" / f"{cache_key}.npz",
        receipt_path=receipt_path, cache_key=cache_key, runtime=runtime,
    )
    if result is None:
        raise ValueError("census reference category fields are absent or incompatible")
    result[5].update({
        "dense_field_cache_key": cache_key, "dense_field_cache_hit": True,
        "census_reference": {"hash": reference_hash, "category_key": cache_key,
                             "forward_capture_hash": manifest["forward_capture_hash"]},
    })
    return result


def _array(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _write_npz(path, arrays):
    with path.open("wb") as stream:
        np.savez(stream, **arrays)
    return {"path": path.name, "sha256": sha256_file(path)}


def _write_content_npz(directory, number, arrays):
    pending = directory / f"{number}.pending.npz"
    record = _write_npz(pending, arrays)
    destination = directory / f"{record['sha256']}.npz"
    pending.replace(destination)
    return {"path": f"{directory.name}/{destination.name}", "sha256": record["sha256"]}


class _FactorBank:
    """Bounded packet reads; images and target factors have separate files."""

    def __init__(self, root, packets, context_locations, block_descriptors, weights,
                 activation_descriptors, capture_batch_size):
        self.root, self.packets, self.locations = root, packets, context_locations
        self.blocks, self.weights = block_descriptors, weights
        self.activations = activation_descriptors
        self.capture_batch_size = capture_batch_size

    @lru_cache(maxsize=2)
    def read(self, relative_path, digest):
        loaded = read_verified_npz(self.root / relative_path)
        if loaded.sha256 != digest:
            raise ValueError("census forward/factor payload hash mismatch")
        return loaded.arrays

    def batches(self, provider, category, plan, context_block_id):
        selected = provider._batch_provider._active_context_ids
        device = provider.evaluator.device

        for start in range(0, len(selected), self.capture_batch_size):
            ids = selected[start:start + self.capture_batch_size]
            # Read each contributing physical packet once, then retain only
            # these requested image slices while assembling the sealed order.
            # A random prefix must not reread a whole packet for every member.
            gathered = {False: {}, True: {}}
            for packet in dict.fromkeys(self.locations[i][0] for i in ids):
                for gradients, role in ((False, "common"), (True, category)):
                    record = self.packets[packet][role]
                    arrays = self.read(record["path"], record["sha256"])
                    for context_id in ids:
                        number, position = self.locations[context_id]
                        if number == packet:
                            gathered[gradients][context_id] = {
                                name: value[position].copy() for name, value in arrays.items()
                            }

            def join(ids, member, *, gradients=False):
                return torch.as_tensor(np.stack([gathered[gradients][i][member] for i in ids]), device=device)

            response = join(ids, "responses").requires_grad_(True)
            labels = join(ids, "labels")
            blocks, activations = [], []
            route_grads, activation_grads = [], []
            for index, descriptor in enumerate(self.blocks):
                output = join(ids, f"output_{index}").requires_grad_(True)
                input_source = (join(ids, f"input_{index}") if descriptor["input"] else None)
                parameters = dict(descriptor["parameters"])
                for name in descriptor["tensor_parameters"]:
                    parameters[name] = join(ids, f"parameter_{index}_{name}")
                for name, value in descriptor["static_tensor_parameters"].items():
                    parameters[name] = torch.as_tensor(np.asarray(value["values"], dtype=value["dtype"]),
                                                        device=device)
                if descriptor["kind"] in {"conv_channel_pair", "conv_kernel_weight"}:
                    parameters["census_local_source_factor"] = join(ids, f"local_factor_{index}")
                blocks.append(CapturedRouteBlock(
                    route_block=provider.model_adapter._blocks[index],
                    reducer_kind=descriptor["kind"], output=output,
                    input_source=input_source, weight=self.weights[index],
                    scale=descriptor["scale"], abs_aggregation_mode=descriptor["abs_mode"],
                    reducer_parameters=parameters,
                ))
                route_grads.append(join(ids, f"route_{index}", gradients=True).unsqueeze(0))
                if self.activations and provider.scoring_backend.use_conserving_activation_attribution:
                    desc = self.activations[index]
                    activations.append(V1ConservingActivationCapture(
                        layer_id=desc["layer_id"], activation_kind=desc["activation_kind"],
                        post_activation_output=join(ids, f"post_{index}").requires_grad_(True),
                        bias_baseline=join(ids, f"bias_{index}"),
                        capture_source_identity=desc["capture_source_identity"],
                    ))
                    activation_grads.append(join(ids, f"activation_{index}", gradients=True).unsqueeze(0))
            record = self.packets[self.locations[ids[0]][0]][category]
            telemetry = replace(VJPTelemetry(**record["route_telemetry"]), context_count=len(ids))
            activation_telemetry = (replace(VJPTelemetry(**record["activation_telemetry"]),
                                           context_count=len(ids)) if activations else None)
            yield CapturedMeasurementBatch(
                context_block_id, response, labels, tuple(blocks), ids,
                tuple(activations) if activations else None,
                CachedQueryVJP((provider.queries[0].query_id,), tuple(route_grads), telemetry),
                (CachedQueryVJP((provider.queries[0].query_id,), tuple(activation_grads), activation_telemetry)
                 if activations else None),
            )

    def provider_factory(self, category):
        bank = self

        def factory(**kwargs):
            provider = OptimizedScoringResponseProvider(**kwargs)
            original = provider.model_adapter

            class Adapter:
                def __getattr__(self, name):
                    return getattr(original, name)

                def iter_captured_batches(self, plan, *, context_block_id, queries):
                    return bank.batches(provider, category, plan, context_block_id)

            # The scoring backend still owns query evaluation, reduction,
            # float32 rows, and the conserving positive-removal stream.
            provider.scoring_backend.model_adapter = Adapter()
            return provider

        return factory


def _prepare_dense_anchor(request):
    """Fit/install and close this dense calibration before census backprops."""
    from . import registered_resnet18_pilot as pilot
    mask = request["dense_mask"]
    anchor_provider = request.get("anchor_provider")
    if anchor_provider is not None and request.get("dense_anchor") is None:
        if str(request.get("pruning_guiding_loss_id", DEFAULT_PRUNING_GUIDING_LOSS_ID)) != anchor_provider.pruning_guiding_loss_id:
            raise ValueError("census loss must match its dense calibration decision law")
        confidence = float(request["anchor_confidence_level"])
        anchor_key = dense_anchor_request_hash(provider=anchor_provider, confidence_level=confidence,
            **{name: request[name] for name in ("runtime", "optimization_axis", "dense_mask",
                "target_class_id", "run_id", "random_seed", "sample_size_policy")},
            pruning_guiding_loss_id=request.get("pruning_guiding_loss_id", DEFAULT_PRUNING_GUIDING_LOSS_ID))
        required_sources = {i for order in (anchor_provider._calibration_orders,
                                            anchor_provider._oracle_orders)
                            for ids in order.values() for i in ids}
        if not required_sources <= set(request["optimization_axis"].context_ids):
            raise ValueError("dense anchor requests images outside the census axis")
        certificate = anchor_provider.certify_baseline_risk(mask,
            cumulative_context_count=max(2 * int(anchor_provider.oracle_minimum_per_stratum),
                int(request["sample_size_policy"].oracle_initial_packet)),
            confidence_level=confidence, maximum_planned_looks=None, precision_only_iid=True)
        certificate, closure = pilot._close_locked_mask_calibration_to_risk_resolution(
            provider=anchor_provider, mask=mask, confidence_level=confidence,
            cumulative_risk_context_count=int(certificate.context_count), certificate=certificate,
            certificate_role="dense_baseline_proposal_adjudication_reference", stream_namespace="dense-baseline",
            progress_callback=None)
        initial_certificate = certificate
        initial_fit_id = anchor_provider.calibration_fits[mask.mask_hash].calibration_state_id
        span_receipt = None
        empty_certificate = request.get("anchor_empty_certificate")
        if empty_certificate is not None:
            certificate, _span, span_receipt = pilot._close_positive_dense_to_empty_span(
                provider=anchor_provider, dense_mask=mask, empty_certificate=empty_certificate,
                dense_certificate=certificate, confidence_level=confidence,
                cumulative_risk_context_count=certificate.context_count, progress_callback=None)
        request["dense_calibration_state"] = anchor_provider.calibration_fits[mask.mask_hash]
        request["dense_anchor"] = {"request_hash": anchor_key,
            "fit": request["dense_calibration_state"], "certificate": certificate, "closure_receipt": closure,
            "initial_certificate": initial_certificate, "initial_fit_id": initial_fit_id,
            "span_closure_receipt": span_receipt,
            "empty_certificate_hash": empty_certificate.content_hash if empty_certificate is not None else None,
            "iid_draws": anchor_provider._iid_draws,
            "risk_evidence": anchor_provider._baseline_risk_evidence_by_mask.get(mask.mask_hash),
            "risk_look_count": anchor_provider._baseline_risk_look_count_by_mask.get(mask.mask_hash, 0),
            "calibration_context_history": sorted(anchor_provider._calibration_context_history.get(mask.mask_hash, ())),
            "crossfit_disposition": anchor_provider._calibration_crossfit_disposition.get(mask.mask_hash)}
    if request.get("dense_anchor") is not None:
        if (request["dense_anchor"].get("span_closure_receipt") is None
            and request.get("anchor_empty_certificate") is not None):
            from mechanistic_systems.inference.sequential_bce import BaselineRiskCertificate
            anchor = dict(request["dense_anchor"])
            certificate = anchor["certificate"]
            if not isinstance(certificate, BaselineRiskCertificate):
                certificate = BaselineRiskCertificate(**certificate)
            empty = request["anchor_empty_certificate"]
            from mechanistic_systems.inference.sequential_bce import LossSpanCertificate
            if LossSpanCertificate(dense=certificate, fully_pruned=empty).lower <= 0.0:
                raise ValueError("precomputed census dense anchor lacks a closed positive span")
            _, _span, closure = pilot._close_positive_dense_to_empty_span(
                provider=None, dense_mask=mask, empty_certificate=empty,
                dense_certificate=certificate, confidence_level=empty.confidence_level,
                cumulative_risk_context_count=certificate.context_count)
            anchor.update(initial_certificate=certificate, span_closure_receipt=closure,
                          empty_certificate_hash=empty.content_hash)
            request["dense_anchor"] = anchor
    _bind_census_materiality_resolution(request)
    _census_objective(request)


@run_forward_numerics()
def build_census_reference(*, requests: Sequence[Mapping[str, Any]], output_dir):
    """Build a campaign from resolved registered census requests.

    Consumer requests are normalized to P's registered census settings;
    requests sharing the category key acquire one field.
    Supply either each category's converged dense fit, or ``anchor_provider``
    and ``anchor_confidence_level``. The latter runs the existing dense-anchor
    closure before the shared census forward and fills the fit. A precomputed
    ``dense_anchor`` may carry the same fit/certificate/closure plus its
    ``request_hash`` (from :func:`dense_anchor_request_hash`).
    No terminal contexts or responses are acquired here.
    """
    from . import registered_resnet18_pilot as pilot
    from mechanistic_systems.measurement.run_products import pruning_functional
    if not requests:
        raise ValueError("campaign census requires categories")
    if any((request.get("census_sample_design") or {}).get("design")
           == pilot.CENSUS_SAMPLE_DESIGN_IMPORTANCE_BATCHES for request in requests):
        raise ValueError("campaign census references are not built for the importance-batch census")
    requests = [pilot._p_census_settings(**request) for request in requests]
    if pruning_functional() == "substituted_softmax":
        for request in requests:
            if request.get("pruning_guiding_loss_id") != CALIBRATED_SUBSTITUTED_SOFTMAX:
                raise ValueError("substituted census reference requires the substituted decision loss")
            if not callable(request.get("decision_fixed_offset_resolver")):
                raise ValueError("substituted census reference requires its resident dense table offset resolver")
            if getattr(request.get("anchor_empty_certificate"), "schema", None) != "substituted_scale_zero_training_risk_v1":
                raise ValueError("substituted census reference requires its category's scale-zero training anchor")
    first = requests[0]
    source = _source_identity(first)
    forward_precision = numerics_record()
    for request in requests:
        if pilot._target_class_id(request["query"]) != int(request["target_class_id"]):
            raise ValueError("campaign census query differs from its target category")
        if _source_identity(request) != source or request["runtime"] is not first["runtime"]:
            raise ValueError("campaign census must share one dense network and image axis")
        if request["sample_size_policy"].response_acquisition_envelope_seconds is not None:
            raise ValueError("campaign census needs a look policy without a wall-time acquisition envelope")
        provider = request.get("anchor_provider")
        if (provider is not None and request.get("dense_anchor") is None
            and _anchor_acquisition_policy(provider, request["dense_mask"])["law"]
            == "measured_execution_alignment"):
            raise ValueError("campaign census sampled anchors need fixed calibration acquisition sizes "
                             "or an already computed dense anchor")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    published = root / "reference.json"
    if published.exists():
        digest = sha256_file(published)
        for request in requests:
            if request.get("anchor_provider") is not None:
                anchor_key = dense_anchor_request_hash(
                    provider=request["anchor_provider"],
                    confidence_level=request["anchor_confidence_level"],
                    **{name: request[name] for name in ("runtime", "optimization_axis", "dense_mask",
                        "target_class_id", "run_id", "random_seed", "sample_size_policy")},
                    pruning_guiding_loss_id=request.get("pruning_guiding_loss_id", DEFAULT_PRUNING_GUIDING_LOSS_ID))
                load_dense_anchor(reference_path=published, reference_hash=digest,
                    request_hash=anchor_key, provider=request["anchor_provider"],
                    dense_mask=request["dense_mask"])
                request["dense_calibration_state"] = request["anchor_provider"].calibration_fits[
                    request["dense_mask"].mask_hash]
            load_category_census(reference_path=published, reference_hash=digest,
                cache_key=category_cache_key(request),
                pruning_guiding_loss_id=request.get("pruning_guiding_loss_id", DEFAULT_PRUNING_GUIDING_LOSS_ID),
                **{name: request[name] for name in (
                    "runtime", "optimization_axis", "dense_mask", "query", "target_class_id",
                    "dense_calibration_state", "partition_family")})
        return {"reference_path": str(published), "reference_hash": digest, "cache_hit": True}
    if any(root.iterdir()):
        raise ValueError("unpublished census output directory must be empty")
    started = time.perf_counter()
    for request in requests:
        _prepare_dense_anchor(request)
    requests_by_key = {}
    for request in requests:
        key = category_cache_key(request)
        previous = requests_by_key.get(key)
        if previous is not None and _digest(previous["query"]) != _digest(request["query"]):
            raise ValueError("campaign census category key has conflicting queries")
        requests_by_key.setdefault(key, request)
    requests = list(requests_by_key.values())
    keys = list(requests_by_key)
    from mechanistic_systems.inference.sequential_bce import RunningMoment
    deferred_loss_moments = {
        index: {name: RunningMoment.empty() for name in request["anchor_provider"]._oracle_stratum_axis().names}
        for index, request in enumerate(requests)
        if _census_materiality_needs_shared_forward(request)
    }
    common_root, factor_root = root / "forward", root / "factors"
    common_root.mkdir()
    factor_root.mkdir()
    (root / "categories").mkdir()
    (root / "anchors").mkdir()
    (root / "readouts").mkdir()
    for key in keys:
        (factor_root / key).mkdir()
    runtime, axis, mask = first["runtime"], first["optimization_axis"], first["dense_mask"]
    queries_by_id = {}
    for request in requests:
        query = request["query"]
        previous = queries_by_id.get(query.query_id)
        if previous is not None and _digest(previous) != _digest(query):
            raise ValueError("campaign query ID has conflicting queries")
        queries_by_id.setdefault(query.query_id, query)
    # P-rep's distinct run seed owns a separate census on the same query.
    queries = tuple(queries_by_id.values())
    corrected = any(request.get("v0_attribution_order", pilot.V0_DEFAULT_ATTRIBUTION_ORDER)
                    == "corrected_aumann_shapley_global_attribution" for request in requests)
    local_regime = any(request.get("enable_local_regime_diagnostics", False) for request in requests)
    scales, transport = resnet_local_destination_transport(runtime.evaluator)
    template = OptimizedScoringResponseProvider(
        evaluator=runtime.evaluator, measurement_view=axis.data_view(),
        context_ids=axis.context_ids, batch_factory=runtime.optimization_data_plane.role_batch_factory("scoring"),
        mask_resolver=lambda _: mask, calibration_resolver=lambda _: first["dense_calibration_state"],
        route_ids=first["route_ids"], queries=queries, response_spec_hash=_digest(source),
        response_metric_id="campaign-census", study_hash=axis.content_hash,
        local_destination_scales_by_layer=scales, local_destination_transport_hash=transport,
        local_regime_specs_by_layer=(resnet_local_regime_capture_specs(runtime.evaluator)
                                    if local_regime else {}),
        conserving_activation_bindings=(alexnet_v1_conserving_activation_bindings(
            runtime.evaluator.mask_bindings) if corrected else ()),
    )
    for layer, gate in template._masks_by_layer(mask).items():
        template.model_adapter.gate_graph.set_gate(layer, gate)
    template._active_request_id = _digest(source)
    block_id = "campaign-census-forward"
    template._batch_provider.activate(context_block_id=block_id, context_ids=axis.context_ids)
    plan = template._measurement_plan(context_block_id=block_id)
    packets, readout_packets, locations, descriptors, weights, activation_descriptors = [], [], {}, [], [], []
    forward_seconds, vjp_seconds, local_seconds = 0.0, 0.0, 0.0
    factor_write_seconds, common_write_seconds, readout_write_seconds = 0.0, 0.0, 0.0
    iterator = iter(template.model_adapter.iter_captured_batches(plan, context_block_id=block_id, queries=queries))
    capture_batch_size = 1
    while True:
        before = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        forward_seconds += time.perf_counter() - before
        capture_batch_size = max(capture_batch_size, len(batch.context_ids))
        packet_index = len(packets)
        for position, context_id in enumerate(batch.context_ids):
            if context_id in locations:
                raise ValueError("census forward repeated an image")
            locations[context_id] = (packet_index, position)
        common = {"responses": _array(batch.response_matrix), "labels": _array(batch.labels)}
        for index, moments in deferred_loss_moments.items():
            request = requests[index]
            provider = request["anchor_provider"]
            loss = provider._binary_bce_device_rows_for_mask(
                SimpleNamespace(device_values=batch.response_matrix, device_labels=batch.labels),
                mask_hash=request["dense_mask"].mask_hash,
                fit=request["dense_calibration_state"], fold_index_per_row=None,
                target_class_id=int(request["target_class_id"]),
            )
            rows = _array(loss).reshape(-1)
            labels = np.asarray(common["labels"]).reshape(-1)
            target = int(request["target_class_id"])
            strata = np.asarray([
                "on_target" if int(label) == target else (
                    "off_target" if "off_target" in moments else f"off_target:{int(label)}"
                ) for label in labels
            ])
            if len(rows) != len(labels) or not set(strata) <= set(moments):
                raise ValueError("shared census loss rows differ from the oracle strata")
            for name, moment in moments.items():
                selected = rows[strata == name]
                if selected.size:
                    moments[name] = moment.merge(RunningMoment.from_values(selected))
        before = time.perf_counter()
        readout = _write_content_npz(root / "readouts", packet_index, common)
        readout_write_seconds += time.perf_counter() - before
        readout_packets.append({"context_ids": list(batch.context_ids), **readout})
        for index, block in enumerate(batch.route_blocks):
            common[f"output_{index}"] = _array(block.output)
            if block.input_source is not None:
                common[f"input_{index}"] = _array(block.input_source)
            if block.reducer_kind in {"conv_channel_pair", "conv_kernel_weight"}:
                from mechanistic_systems.backends.torch_runtime import census_conv_local_source_factor
                before = time.perf_counter()
                common[f"local_factor_{index}"] = _array(census_conv_local_source_factor(
                    block.input_source, block.weight, output_shape=block.output.shape,
                    **{name: block.reducer_parameters[name] for name in ("stride", "padding", "dilation")}))
                local_seconds += time.perf_counter() - before
            tensor_names, parameters, static_parameters = [], {}, {}
            for name, value in block.reducer_parameters.items():
                if torch.is_tensor(value) or isinstance(value, np.ndarray):
                    array = _array(value)
                    if name == "local_regime_preactivation":
                        if array.ndim == 0 or array.shape[0] != len(batch.context_ids):
                            raise ValueError("census preactivation parameter differs from the image axis")
                        common[f"parameter_{index}_{name}"] = array
                        tensor_names.append(name)
                    else:
                        static_parameters[name] = {"values": array.tolist(), "dtype": str(array.dtype)}
                else:
                    parameters[name] = value
            if packet_index == 0:
                descriptors.append({"kind": block.reducer_kind, "scale": block.scale,
                    "abs_mode": block.abs_aggregation_mode, "input": block.input_source is not None,
                    "parameters": parameters, "tensor_parameters": tensor_names,
                    "static_tensor_parameters": static_parameters})
                weights.append(block.weight)
            elif descriptors[index]["static_tensor_parameters"] != static_parameters:
                raise ValueError("census static route parameter changed between image packets")
        if batch.conserving_activation_captures:
            for index, capture in enumerate(batch.conserving_activation_captures):
                common[f"post_{index}"] = _array(capture.post_activation_output)
                common[f"bias_{index}"] = np.broadcast_to(_array(capture.bias_baseline),
                                                          tuple(batch.route_blocks[index].output.shape))
                if packet_index == 0:
                    activation_descriptors.append({name: getattr(capture, name) for name in (
                        "layer_id", "activation_kind", "capture_source_identity")})
        packet = {"context_ids": list(batch.context_ids)}
        for category_index, (request, key) in enumerate(zip(requests, keys, strict=True)):
            query_batch = evaluate_queries(batch.response_matrix, (request["query"],), labels=batch.labels)
            last = category_index == len(requests) - 1
            category_corrected = (request.get("v0_attribution_order", pilot.V0_DEFAULT_ATTRIBUTION_ORDER)
                                  == "corrected_aumann_shapley_global_attribution")
            before = time.perf_counter()
            gradients, telemetry = exact_batched_vjp(batch.response_matrix,
                [block.output for block in batch.route_blocks], query_batch,
                backend="is_grads_batched", retain_graph=not last or category_corrected)
            factors = {f"route_{index}": _array(gradient[0]) for index, gradient in enumerate(gradients)}
            if category_corrected:
                post_gradients, activation_telemetry = exact_batched_vjp(batch.response_matrix,
                    [capture.post_activation_output for capture in batch.conserving_activation_captures],
                    query_batch, backend="is_grads_batched", retain_graph=not last)
                factors.update({f"activation_{index}": _array(gradient[0])
                                for index, gradient in enumerate(post_gradients)})
            vjp_seconds += time.perf_counter() - before
            before = time.perf_counter()
            record = _write_npz(factor_root / key / f"{packet_index}.npz", factors)
            factor_write_seconds += time.perf_counter() - before
            record["path"] = f"factors/{key}/{record['path']}"
            record["route_telemetry"] = asdict(telemetry)
            if category_corrected:
                record["activation_telemetry"] = asdict(activation_telemetry)
            packet[key] = record
        before = time.perf_counter()
        record = _write_content_npz(common_root, packet_index, common)
        common_write_seconds += time.perf_counter() - before
        packet["common"] = record
        packets.append(packet)
        del batch, gradients, factors, common
    if tuple(locations) != tuple(axis.context_ids):
        raise ValueError("campaign dense forward changed the census image order")
    for index, moments in deferred_loss_moments.items():
        request = requests[index]
        provider = request["anchor_provider"]
        stratum_axis = provider._oracle_stratum_axis()
        if any(moment.count != stratum_axis.population_by_stratum[name]
               or moment.count < 2 for name, moment in moments.items()):
            raise ValueError("shared census dense loss moments do not cover the oracle population")
        fit_hash = dense_calibration_state_hash(request["dense_calibration_state"])
        variances = {name: moment.sample_variance for name, moment in moments.items()}
        counts = {name: moment.count for name, moment in moments.items()}
        if not hasattr(provider, "_census_dense_loss_variance_cache"):
            provider._census_dense_loss_variance_cache = {}
        provider._census_dense_loss_variance_cache[(fit_hash, stratum_axis.names)] = (
            variances, counts, "shared_census_dense_forward"
        )
        _bind_census_materiality_resolution(request)
        request["dense_anchor"] = {**request["dense_anchor"], "dense_loss_moments": {
            "dense_calibration_state_hash": fit_hash, "variances": variances,
            "counts": counts, "source": "shared_census_dense_forward",
        }}
    final_keys = [category_cache_key(request) for request in requests]
    if len(set(final_keys)) != len(final_keys):
        raise ValueError("campaign census repeats a category key")
    for old, new in zip(keys, final_keys, strict=True):
        if old != new:
            (factor_root / old).rename(factor_root / new)
            for packet in packets:
                record = packet.pop(old)
                record["path"] = record["path"].replace(f"factors/{old}/", f"factors/{new}/", 1)
                packet[new] = record
    keys = final_keys
    forward_hash = _digest({"source": source, "packets": [p["common"] for p in packets],
                           "blocks": descriptors, "activation_blocks": activation_descriptors})
    bank = _FactorBank(root, packets, locations, descriptors, weights, activation_descriptors, capture_batch_size)
    manifest = {"schema": SCHEMA, "source_identity": source,
                "forward_precision": forward_precision,
                "algorithm_settings": _algorithm_settings(),
                "forward_capture_hash": forward_hash, "categories": {}, "dense_anchors": {},
                "readout_packets": readout_packets,
                "forward_packets": [{"context_ids": p["context_ids"], **p["common"]} for p in packets],
                "forward_block_descriptors": descriptors,
                "activation_descriptors": activation_descriptors,
                "algorithm": "dense_calibration_then_shared_forward_separate_target_vjps_loss_census_looks",
                "category_count": len(requests), "forward_context_count": len(locations),
                "forward_batch_count": len(packets), "target_vjp_count": len(packets) * sum(
                    2 if r.get("v0_attribution_order", pilot.V0_DEFAULT_ATTRIBUTION_ORDER)
                    == "corrected_aumann_shapley_global_attribution" else 1 for r in requests)}
    if pruning_functional() == "substituted_softmax":
        manifest["pruning_functional"] = "substituted_softmax"
    category_seconds = {}
    for request, key in zip(requests, keys, strict=True):
        category_started = time.perf_counter()
        if request.get("dense_anchor") is not None:
            anchor = _plain(request["dense_anchor"])
            if anchor["fit"] != _plain(request["dense_calibration_state"]):
                raise ValueError("dense anchor and census use different affines")
            anchor_path = root / "anchors" / f"{anchor['request_hash']}.json"
            anchor_path.write_text(json.dumps(anchor, sort_keys=True, indent=2) + "\n")
            manifest["dense_anchors"][anchor["request_hash"]] = {
                "path": f"anchors/{anchor_path.name}", "sha256": sha256_file(anchor_path)}
        inputs = {name: value for name, value in request.items()
                  if name not in {"dense_anchor", "anchor_provider", "anchor_confidence_level", "anchor_empty_certificate"}}
        fields = pilot._measure_dense_local_partition_geometry(
            **inputs, scoring_provider_factory=bank.provider_factory(key))
        fields[5]["campaign_census_execution"] = {
            "forward_precision": forward_precision,
            "forward_capture_hash": forward_hash,
            "shared_dense_forward": True, "target_vjps_separate": True,
            "local_source_factors_shared": True, "sealed_looks_category_specific": True,
        }
        if "v0_attribution_order" in fields[5]:
            fields[5]["v0_attribution_order"]["physical_execution"] = (
                "shared_dense_forward_separate_preactivation_and_postactivation_target_vjps")
        path = root / "categories" / key
        pilot._store_hard_partition_cache(payload_path=path.with_suffix(".npz"),
            receipt_path=path.with_suffix(".json"), cache_key=key, runtime=runtime,
            local_by_layer=fields[0], global_by_layer=fields[1], global_activity_by_layer=fields[2],
            ordering_statistics_by_layer=fields[3], flip_margin_geometry=fields[4], measurement_receipt=fields[5])
        manifest["categories"][key] = {"query_hash": _digest(request["query"]),
            "target_class_id": int(request["target_class_id"]),
            "census_objective": _census_objective(request),
            "receipt_sha256": sha256_file(path.with_suffix(".json")),
            "dense_anchor_request_hash": (request["dense_anchor"]["request_hash"]
                if request.get("dense_anchor") is not None else None)}
        category_seconds[key] = time.perf_counter() - category_started
        del fields
    manifest["cost"] = {"forward_seconds": forward_seconds, "target_vjp_seconds": vjp_seconds,
        "shared_local_rows_seconds": local_seconds, "factor_write_seconds": factor_write_seconds,
        "common_write_seconds": common_write_seconds, "readout_write_seconds": readout_write_seconds,
        "category_anchor_reduce_store_seconds": category_seconds,
        "total_seconds": time.perf_counter() - started}
    bank.read.cache_clear()
    pending = root / "reference.json.pending"
    pending.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    pending.replace(published)
    return {"reference_path": str(published), "reference_hash": sha256_file(published),
            "cache_hit": False, **manifest["cost"]}
