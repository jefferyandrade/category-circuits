"""Matched registered-model pilot for the authoritative modern queue-v0.

Historical v1/v2 helper definitions remain only as salvage-bound carrier
material until their byte-bound archive waves are applied; their workflow
assembly is absent.  The public entry point refuses those identities before
touching model, device, output, or artifact state.  Model loading, route gates,
score heads, masked execution, calibration, context loading, and custom route
reducers are inherited from the registered sampled-path runtime.
"""

from __future__ import annotations

from contracts.runtime_receipts import runtime_receipt_hash

import atexit
import errno
import json
import hashlib
import io
import math
import os
from pathlib import Path
import resource
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from zipfile import BadZipFile as zipfile_BadZipFile
from statistics import NormalDist
from dataclasses import asdict, dataclass, replace
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from collections.abc import Callable, Iterator, Mapping, Sequence

from mechanistic_systems.core.determinism import DETERMINISTIC_EXECUTION_CACHE_VERSION
from mechanistic_systems.core.numerics import numerics_record, resolve_abs_aggregation_mode

import numpy as np

from circuitflow.artifacts import ArtifactStore
from contracts.hashing import sha256_file, stable_digest
from mechanistic_systems.contracts.bce_evidence import (
    CalibrationPrecisionHorizonExhausted,
)
from mechanistic_systems.circuits.evidence_bce_precision import (
    DirectCandidateIncrementDecisionRule,
    V0MarginalConnectionDecisionRule,
)
from mechanistic_systems.circuits.irg.accepted_v0_trajectory import (
    AcceptedV0Trajectory,
    materialize_accepted_v0_trajectory,
)
from mechanistic_systems.measurement.lazy_attribute import LazyAttribute
from mechanistic_systems.measurement.run_products import effective_settings, calibration_penalty as _calibration_penalty, loss_weighting, pruning_functional
from mechanistic_systems.measurement.terminal_statistics import substituted_loss_window_statistics

robust_resnet50_checkpoint_proof_binding = LazyAttribute('mechanistic_systems.circuits.irg.checkpoint_qualified_proofs', 'robust_resnet50_checkpoint_proof_binding')
from mechanistic_systems.inference.sequential_bce import (
    IID_WITH_REPLACEMENT_SAMPLING_ID,
    ORACLE_SAMPLING_LAWS,
    SEALED_WOR_CLASS_STRATIFIED_SAMPLING_ID,
    SEQUENTIAL_CALIBRATED_BCE_V4,
    SEQUENTIAL_CALIBRATED_BCE_V5,
    V0_NORMALIZED_LOSS_SPAN_PER_CONNECTION_V1,
    AnalyticChanceRiskCertificate,
    BaselineRiskCertificate,
    LossSpanCertificate,
    PRECISION_ONLY_IID_SCOPE,
    SequentialCalibratedBCEResolutionSpec,
    guard_unvalidated_cumulative_forecast,
)
from mechanistic_systems.circuits.native_sampled import (
    BCE_PRECISION_ALGORITHM,
    run_profiled_native_sampled_path_panel,
)
from mechanistic_systems.circuits.native_continuous import (
    BackendNaturalizationFacts,
)
from mechanistic_systems.circuits.glocal_sampled_path import (
    GlocalSampledComparisonRequest,
)
from mechanistic_systems.circuits.runner import (
    _resolve_registered_sampled_path_runtime,
)
from mechanistic_systems.backends.torch_runtime import (
    prefix_reuse_backend,
)
from mechanistic_systems.circuits.torch_evaluator import (
    RuntimeReadoutProjectionSpec,
)
from mechanistic_systems.circuits.step_observables import (
    STEP_CONTEXT_DELTA_SIDECAR_FILENAME,
    STEP_CONTEXT_DELTA_SIDECAR_SCHEMA,
    StepContextDeltaAccumulator,
    step_observables_enabled,
)
from mechanistic_systems.contracts import CircuitPlan, StudySpec
from mechanistic_systems.measurement.model_sources import ModelSourceReplica
from mechanistic_systems.measurement.models import (
    ROBUSTBENCH_SALMAN2020DO_R50_MODEL_ID,
)
from mechanistic_systems.measurement.native_irg_providers import (
    FrozenCategoryCalibrationFitBundle,
    FrozenCategoryStructureCapture,
    FrozenCategoryValidationOpenToken,
    NativeBCECalibrationPolicy,
    NativeBCECalibrationFitState,
    NativeFixedCalibrationBCEPrecisionBackend,
    NativeTorchCalibratedBCEOracleProvider,
    calibration_folding_enabled,
    fused_calibration_oracle_batch_rows,
    frozen_category_bank_capture_status,
)
from mechanistic_systems.measurement.frozen_category_calibration import (
    CalibrationBankMode,
    CalibrationBankNotReadyError,
    CalibrationContractError,
    CalibrationFunctional,
    CategoryCalibrationIdentity,
    CategoryStratifiedEvaluationMatrices,
    FrozenAffineCalibrationState,
    FrozenCategoryCalibrationBank,
)

OptimizedScoringResponseProvider = LazyAttribute('mechanistic_systems.measurement.optimized_irg_providers', 'OptimizedScoringResponseProvider')
resnet_local_destination_transport = LazyAttribute('mechanistic_systems.measurement.optimized_irg_providers', 'resnet_local_destination_transport')
resnet_local_regime_capture_specs = LazyAttribute('mechanistic_systems.measurement.optimized_irg_providers', 'resnet_local_regime_capture_specs')

DeviceFrequentDirectionsTrustAccumulator = LazyAttribute('mechanistic_systems.measurement.device_route_moments', 'DeviceFrequentDirectionsTrustAccumulator')
DeviceTripartiteMomentAccumulator = LazyAttribute('mechanistic_systems.measurement.device_route_moments', 'DeviceTripartiteMomentAccumulator')
combine_stratified_route_decision_snapshots = LazyAttribute('mechanistic_systems.measurement.device_route_moments', 'combine_stratified_route_decision_snapshots')
combine_stratified_tripartite_moments = LazyAttribute('mechanistic_systems.measurement.device_route_moments', 'combine_stratified_tripartite_moments')
join_census_moment_hashes = LazyAttribute('mechanistic_systems.measurement.device_route_moments', 'join_census_moment_hashes')

DenseFlipMarginAccumulator = LazyAttribute('mechanistic_systems.measurement.flip_margin_geometry', 'DenseFlipMarginAccumulator')
from mechanistic_systems.measurement.flip_margin_geometry import MARGIN_NOT_CAPTURED, MARGIN_PERCENTILE_LEVELS
validate_flip_margin_geometry = LazyAttribute('mechanistic_systems.measurement.flip_margin_geometry', 'validate_flip_margin_geometry')
from mechanistic_systems.measurement.neyman_allocation import (
    UnifiedNeymanAllocator,
    binary_neyman_request,
)

LocalAttributionMomentTensor = LazyAttribute('mechanistic_systems.measurement.streaming_response_provider', 'LocalAttributionMomentTensor')
TripartitePullbackMomentTensor = LazyAttribute('mechanistic_systems.measurement.streaming_response_provider', 'TripartitePullbackMomentTensor')
from mechanistic_systems.measurement.iid_streams import (
    BinaryIIDSamplingDesign,
    StratifiedIIDSamplingDesign,
    class_stratified_binary_iid_sampling_design_from_axis,
    class_stratified_binary_sealed_without_replacement_design_from_axis,
)
from mechanistic_systems.measurement.gpu_runtime import GpuActivitySampler
from mechanistic_systems.measurement.pruning_loss import (
    CALIBRATED_OVR_BCE, CALIBRATED_SUBSTITUTED_SOFTMAX, dense_calibration_state_hash, census_loss_pullback_hash,
    census_loss_path_hash,
    resolve_pruning_guiding_loss_id, validate_pruning_guiding_loss_id,
)

# The raw heads remain in the payload; only these loss fields govern census readers.
CENSUS_LOSS_STATISTIC_NAMES = (
    "loss_mean", "loss_mean_standard_error", "loss_activity",
    "loss_activity_standard_error", "loss_gradient_activity",
    "loss_contextual_variance", "loss_second_moment",
    "loss_second_moment_standard_error", "loss_mean_estimator_variance",
)
# Verify-once receipts and single-pass verified reads.
from mechanistic_systems.measurement.verified_once import (
    VERIFIED_BY_RECEIPT as _VERIFY_ONCE_BY_RECEIPT,
    VerifiedNpzPrefetch,
    file_identity as _verified_file_identity,
    read_verified_npz,
    sha256_file_once,
)
# Compact run outputs and their formats.
from mechanistic_systems.measurement.run_products import (
    RunProductWriter,
    collect_run_product_sidecars as _collect_run_product_sidecars,
    source_manifest_verification as _source_manifest_verification,
    write_compact_run_outputs as _write_compact_run_outputs,
)

# Trust-walk collaborators are imported by FULL module path (no package
# ``__init__`` re-export).

scoped_field = LazyAttribute('mechanistic_systems.circuits.irg.scoped_field', None)

trust_walk = LazyAttribute('mechanistic_systems.circuits.irg.trust_walk', None)
from mechanistic_systems.core.axis_identity import (
    canonical_string_axis_content_hash,
    constructive_ordinal_route_axis,
    ordered_string_axis_hash,
)

from .candidates import HardMask
from .contracts import (
    GlobalTrustGeometrySpec,
    ContextSelector,
    MeasurementProvenance,
    MeasurementRequest,
    content_hash,
)
from .costs import CostEstimate
from .geometric_partitions import (
    CanonicalTwoPartitionTrustPolicy,
    canonical_global_two_sign_partitioning,
)

GlobalTrustGeometryState = LazyAttribute('mechanistic_systems.circuits.irg.trust_geometry', 'GlobalTrustGeometryState')
from .ordering_keys import (
    GlobalHeadOrderingField,
    ORDERING_FIELD_STATISTIC_NAMES,
    ORDERING_KEY_REGISTRY_VERSION,
    V1_ORDERING_KEYS,
    ordering_key_statistic,
)
from mechanistic_systems.experimental_design.sampling_identity import (
    BinaryEvidenceRoleDesign,
    binary_evidence_role_namespace,
    compile_binary_iid_sampling_design,
)
from .criterion_forecast import deciding_oracle_resolution_floor
from .criterion_forecast import expected_wrongsign_consequence_fraction
from .criterion_forecast import sigma_from_snapshot
from .criterion_forecast import smallest_count_meeting_tolerance
from .criterion_forecast import stratified_mean_standard_error_at
from .criterion_forecast import unresolved_activity_fraction
from .criterion_forecast import wrongsign_consequence_upper_bound_fraction
from .modern_v1_imagenet_population import (
    IMAGENET1K_CLASS_COUNT,
    IMAGENET1K_DATASET_ID,
    IMAGENET1K_TRAIN_CONTEXT_COUNT,
    IMAGENET1K_TRAIN_ROLE,
    IMAGENET1K_TRAIN_SPLIT,
    IMAGENET1K_VALIDATION_CONTEXT_COUNT,
    IMAGENET1K_VALIDATION_ROLE,
    IMAGENET1K_VALIDATION_SPLIT,
)
from .precision import IndependentSampleSizePolicy
from .precision import RESPONSE_FIELD_ENVELOPE_RECEIPT_V1
from .precision import ResponseFieldPrecisionEnvelopeExhausted
from .precision import simultaneous_normal_critical_value
from .response_acquisition import SealedUniformPermutationReceipt

from .schedule_replay import SCHEDULE_REPLAY_RANK_SOURCES
ScheduleReplayPolicy = LazyAttribute('mechanistic_systems.circuits.irg.schedule_replay', 'ScheduleReplayPolicy')
load_schedule_replay_spec = LazyAttribute('mechanistic_systems.circuits.irg.schedule_replay', 'load_schedule_replay_spec')
resolve_replay_rank_scores = LazyAttribute('mechanistic_systems.circuits.irg.schedule_replay', 'resolve_replay_rank_scores')
run_schedule_replay_walker = LazyAttribute('mechanistic_systems.circuits.irg.schedule_replay', 'run_schedule_replay_walker')
from .score_order import (
    LayerCodeAxis,
    certify_prune_order_uncertainty,
)
from .census_device_look import (
    certify_prune_order_stability_from_host_arrays as certify_prune_order_stability,
)
from .workflow_identity import resolve_current_workflow_identity

# A float64 tripartite accumulator with h heads, r routes, and sketch rank k
# occupies 8 * (h*r + r*h*h + r*h*h + 3*r + 3*r + k*h*r + k) bytes:
# sum_rows, the two outer-moment tensors, the two view-activity tensors,
# sketch_sum, and sketch_constant, respectively.
_TRIPARTITE_ACCUMULATOR_OFFLOAD_THRESHOLD_BYTES = 8 * (1024**3)


def _tripartite_accumulator_resident_bytes(
    *,
    head_count: int,
    route_count: int,
    sketch_rank: int,
) -> int:
    return 8 * (
        head_count * route_count
        + route_count * head_count * head_count
        + route_count * head_count * head_count
        + 3 * route_count
        + 3 * route_count
        + sketch_rank * head_count * route_count
        + sketch_rank
    )


def _tripartite_accumulator_exceeds_offload_threshold(
    accumulator: DeviceTripartiteMomentAccumulator,
) -> bool:
    return (
        _tripartite_accumulator_resident_bytes(
            head_count=len(accumulator.head_ids),
            route_count=len(accumulator.route_ids),
            sketch_rank=accumulator.sketch_rank,
        )
        > _TRIPARTITE_ACCUMULATOR_OFFLOAD_THRESHOLD_BYTES
    )


def _ensure_stratum_resident(
    accumulators: Mapping[str, DeviceTripartiteMomentAccumulator],
    active_stratum: str,
    device: Any,
) -> None:
    active_accumulator = accumulators[active_stratum]
    if not _tripartite_accumulator_exceeds_offload_threshold(active_accumulator):
        return
    # A census-scale stratum owns tens of GiB of persistent tensors.  Retire
    # every inactive device copy before restoring the requested stratum so a
    # stratum switch cannot transiently hold both full-width states on the
    # accelerator.  This ordering is also fail-closed: if an offload fails,
    # the restore never starts and the peak cannot grow further.
    for stratum, accumulator in accumulators.items():
        if stratum != active_stratum:
            accumulator.offload_to_host()
    active_accumulator.restore_to_device(device)


def _continuing_look_stratum_inputs(
    accumulator: DeviceTripartiteMomentAccumulator,
    *,
    population_size: int,
    count: int,
    sign_measurement_weights_by_head: Mapping[int, Any],
    route_chunk_size: int = 1_048_576,
) -> tuple[float, dict[int, np.ndarray]]:
    """Neyman allocation and sign forecasts from the selected label heads."""
    if accumulator.sample_count <= 0 or accumulator.total_weight <= 0.0:
        raise ValueError("cannot snapshot an empty device accumulator")
    if not accumulator.storage_allocated:
        raise RuntimeError("nonempty device accumulator has no storage")
    if len(accumulator.head_ids) < 3:
        raise ValueError("continuing-look inputs require the census loss head")
    population = int(population_size)
    if not 0 < int(accumulator.sample_count) <= population:
        raise ValueError("finite-population snapshot capacity is invalid")
    chunk_size = int(route_chunk_size)
    if chunk_size <= 0:
        raise ValueError("decision snapshot route chunk size must be positive")
    accumulator.drain_windows()
    route_count = len(accumulator.route_ids)
    heads = tuple(sorted(sign_measurement_weights_by_head))
    if not heads or any(head not in (0, 1, 2) for head in heads):
        raise ValueError("continuing-look sign head selection is invalid")
    variance = np.empty(route_count, dtype=np.float64)
    activity = np.empty(route_count, dtype=np.float64)
    standard_errors = {head: np.empty(route_count, dtype=np.float64) for head in heads}
    for start in range(0, route_count, chunk_size):
        stop = min(start + chunk_size, route_count)
        _, contextual, errors, magnitudes, _ = accumulator._decision_chunk(
            start=start, stop=stop, population_size=population,
            head_indices=heads, activity_indices=heads,
        )
        variance[start:stop] = contextual.sum(dim=0).detach().cpu().numpy()
        activity[start:stop] = magnitudes.sum(dim=0).detach().cpu().numpy()
        for index, head in enumerate(heads):
            standard_errors[head][start:stop] = errors[index].detach().cpu().numpy()
    denominator = max(float(np.sum(activity)), np.finfo(np.float64).tiny)
    allocation_sd = math.sqrt(max(float(np.sum(activity * variance) / denominator), 1e-12))
    return allocation_sd, {
        head: sigma_from_snapshot(
            standard_error=standard_errors[head], count=int(count), population=population,
        ) * sign_measurement_weights_by_head[head]
        for head in heads
    }


PILOT_SCHEMA = "irg_registered_resnet18_v0_v2_pilot_v3"
V0_TRAIN_ONLY_MASK_LOCK_RECEIPT_SCHEMA = "modern_v0_train_only_mask_lock_receipt_v1"
V0_ATTRIBUTION_ORDER_MODES = (
    "plain_global_attribution",
    "corrected_aumann_shapley_global_attribution",
)
V0_DEFAULT_ATTRIBUTION_ORDER = (
    "plain_global_attribution"
)
_V0_CORRECTED_ATTRIBUTION_ARCHITECTURE_FAMILIES = frozenset(
    {
        "alexnet",
        "alexnet_conv_kernel_weight",
    }
)
# v4 decision law: the proposal-adjudication oracle samples each
# class stratum WITHOUT replacement from a sealed permutation, the interval
# carries the exact per-stratum finite-population correction, and the census
# is a terminal.  That changes what a v0 arm certifies, so it changes the
# execution identity rather than being absorbed silently.  The
# with-replacement law remains selectable and replays under the _v3 /
# _ablation_v1 identities via ``v0_oracle_sampling_law_override``.
_V0_DEFAULT_EXECUTION_IDENTITY = (
    "irg_v0_layerwise_loss_span_per_connection_bce_precision_v4"
)
_V0_CORRECTED_ATTRIBUTION_EXECUTION_IDENTITY = (
    "irg_v0_layerwise_corrected_aumann_shapley_global_attribution_"
    "loss_span_per_connection_bce_precision_ablation_v2"
)
_V0_TRAIN_ONLY_MASK_LOCK_RECEIPT_PREFIX = "modern-v0-train-only-mask-lock-receipt-"
_V0_TRAIN_ONLY_ALGORITHM_CONFIGURATION_PREFIX = (
    "modern-v0-train-only-algorithm-configuration-"
)


def _v0_train_only_algorithm_configuration_hash(
    configuration: Mapping[str, Any],
) -> str:
    return stable_digest(
        {
            "domain": _V0_TRAIN_ONLY_ALGORITHM_CONFIGURATION_PREFIX,
            "configuration": dict(configuration),
        }
    )


def _json_receipt_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return an owned, finite, JSON-native receipt mapping."""

    try:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("mask-lock receipt fields must be finite JSON") from exc
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):  # pragma: no cover - mapping input above
        raise TypeError("mask-lock receipt field did not decode to a mapping")
    return decoded


def _required_receipt_mapping(
    value: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result = _json_receipt_copy(value)
    if not result:
        raise ValueError(f"{name} must be nonempty")
    return result


def _required_sha256(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a lowercase SHA-256")
    result = str(value)
    if (
        len(result) != 64
        or result != result.lower()
        or any(character not in "0123456789abcdef" for character in result)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return result


def _nonnegative_finite_seconds(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _validate_sealed_source_lineage(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    source = _required_receipt_mapping(value, name="source_code")
    if source.get("sealed") is not True:
        raise ValueError("train-only mask lock requires a sealed source tree")
    commit = source.get("source_commit")
    manifest = source.get("source_manifest_sha256")
    if commit is not None:
        if (
            type(commit) is not str
            or len(commit) not in {40, 64}
            or commit != commit.lower()
            or any(character not in "0123456789abcdef" for character in commit)
        ):
            raise ValueError("sealed source commit identity is invalid")
    elif manifest is not None:
        _required_sha256(manifest, name="sealed source manifest SHA-256")
    else:
        raise ValueError("sealed source identity lacks a commit or manifest")
    return source


@dataclass(frozen=True, slots=True)
class V0TrainOnlyMaskLockReceipt:
    """Typed authority for a completed v0 mask lock and its warm runtime."""

    model_source: Mapping[str, Any]
    source_code: Mapping[str, Any]
    train_population: Mapping[str, Any]
    sealed_validation_axis: Mapping[str, Any]
    route_axis: Mapping[str, Any]
    final_mask: Mapping[str, Any]
    accepted_trajectory: Mapping[str, Any]
    algorithm: Mapping[str, Any]
    warm_runtime: Mapping[str, Any]
    authority: Mapping[str, Any]
    phase_timing: Mapping[str, Any] | None = None
    schema_name: str = V0_TRAIN_ONLY_MASK_LOCK_RECEIPT_SCHEMA
    workflow_identity: str = "irg/v0"
    receipt_hash: str = ""

    def __post_init__(self) -> None:
        if self.schema_name != V0_TRAIN_ONLY_MASK_LOCK_RECEIPT_SCHEMA:
            raise ValueError("unsupported v0 train-only mask-lock receipt schema")
        if self.workflow_identity != "irg/v0":
            raise ValueError("train-only mask lock is restricted to irg/v0")
        for name in (
            "model_source",
            "source_code",
            "train_population",
            "sealed_validation_axis",
            "route_axis",
            "final_mask",
            "accepted_trajectory",
            "algorithm",
            "warm_runtime",
            "authority",
        ):
            object.__setattr__(
                self,
                name,
                _required_receipt_mapping(getattr(self, name), name=name),
            )

        model = self.model_source
        _required_sha256(
            model.get("model_load_receipt_hash"),
            name="model_load_receipt_hash",
        )
        _required_sha256(
            model.get("model_state_hash"),
            name="model_state_hash",
        )
        if (
            not model.get("model_id")
            or not model.get("architecture_family")
            or model.get("strict_load") is not True
            or model.get("claim_eligible_source") is not True
        ):
            raise ValueError("mask-lock model/source lineage is incomplete")

        _validate_sealed_source_lineage(self.source_code)

        train = self.train_population
        if (
            train.get("dataset_id") != IMAGENET1K_DATASET_ID
            or train.get("split") != IMAGENET1K_TRAIN_SPLIT
            or train.get("role") != IMAGENET1K_TRAIN_ROLE
            or train.get("complete_split") is not True
            or train.get("context_count") != IMAGENET1K_TRAIN_CONTEXT_COUNT
            or train.get("class_count") != IMAGENET1K_CLASS_COUNT
            or train.get("node_local_staged") is not True
            or train.get("optimized_bank") is not True
        ):
            raise ValueError(
                "mask-lock receipt requires complete node-local ImageNet train"
            )
        for name in (
            "context_axis_hash",
            "ordered_context_ids_hash",
            "label_map_hash",
            "bank_source_receipt_hash",
        ):
            _required_sha256(train.get(name), name=f"train_population.{name}")

        validation = self.sealed_validation_axis
        if (
            validation.get("dataset_id") != IMAGENET1K_DATASET_ID
            or validation.get("split") != IMAGENET1K_VALIDATION_SPLIT
            or validation.get("role") != IMAGENET1K_VALIDATION_ROLE
            or validation.get("complete_split") is not True
            or validation.get("context_count") != IMAGENET1K_VALIDATION_CONTEXT_COUNT
            or validation.get("outcomes_opened") is not False
            or validation.get("runtime_data_plane_constructed") is not False
        ):
            raise ValueError(
                "mask-lock receipt requires a sealed, unopened validation axis"
            )
        for name in ("context_axis_hash", "ordered_context_ids_hash"):
            _required_sha256(
                validation.get(name),
                name=f"sealed_validation_axis.{name}",
            )

        route = self.route_axis
        mask = self.final_mask
        route_count = route.get("route_count")
        if (
            isinstance(route_count, bool)
            or not isinstance(route_count, int)
            or route_count <= 0
            or not route.get("route_axis_hash")
        ):
            raise ValueError("mask-lock route axis is incomplete")
        _required_sha256(
            route.get("ordered_route_ids_hash"),
            name="ordered_route_ids_hash",
        )
        retained = mask.get("retained_route_count")
        if (
            mask.get("route_axis_hash") != route.get("route_axis_hash")
            or mask.get("route_count") != route_count
            or isinstance(retained, bool)
            or not isinstance(retained, int)
            or retained < 0
            or retained > route_count
            or mask.get("packed_byte_count") != (route_count + 7) // 8
            or not mask.get("mask_hash")
        ):
            raise ValueError("final mask differs from its bound route axis")
        _required_sha256(
            mask.get("packed_sha256"),
            name="final_mask.packed_sha256",
        )

        trajectory = self.accepted_trajectory
        if (
            trajectory.get("available") is not True
            or trajectory.get("root_mask_hash") != route.get("dense_mask_hash")
            or trajectory.get("terminal_mask_hash") != mask.get("mask_hash")
            or not trajectory.get("receipt_hash")
            or not isinstance(trajectory.get("entries"), list)
        ):
            raise ValueError("accepted v0 trajectory is absent or mask-inconsistent")
        if trajectory.get("accepted_count") != len(trajectory["entries"]):
            raise ValueError("accepted trajectory count differs from its entries")

        algorithm = self.algorithm
        configuration = algorithm.get("configuration")
        if not isinstance(configuration, Mapping):
            raise TypeError("v0 algorithm configuration must be a mapping")
        declared_configuration_hash = _required_sha256(
            algorithm.get("configuration_hash"),
            name="algorithm.configuration_hash",
        )
        observed_configuration_hash = _v0_train_only_algorithm_configuration_hash(
            configuration
        )
        if declared_configuration_hash != observed_configuration_hash:
            raise ValueError("v0 algorithm configuration differs from its hash")
        if algorithm.get("execution_identity") != configuration.get(
            "execution_identity"
        ):
            raise ValueError("v0 execution identity differs from its configuration")

        runtime = self.warm_runtime
        scoring = _nonnegative_finite_seconds(
            runtime.get("scoring_and_anchor_seconds"),
            name="scoring_and_anchor_seconds",
        )
        traversal = _nonnegative_finite_seconds(
            runtime.get("complete_traversal_seconds"),
            name="complete_traversal_seconds",
        )
        mask_lock = _nonnegative_finite_seconds(
            runtime.get("warm_mask_lock_seconds"),
            name="warm_mask_lock_seconds",
        )
        run_entry = _nonnegative_finite_seconds(
            runtime.get("run_entry_to_mask_lock_seconds"),
            name="run_entry_to_mask_lock_seconds",
        )
        if not math.isclose(
            mask_lock, scoring + traversal, rel_tol=1e-12, abs_tol=1e-9
        ):
            raise ValueError("warm mask-lock time does not equal its measured phases")
        if runtime.get("traversal_complete") is not True:
            raise ValueError("train-only mask lock requires a complete v0 traversal")
        if run_entry < mask_lock:
            raise ValueError("cold-inclusive mask-lock time is below its warm subset")
        if (
            runtime.get("scope")
            != "warm_scoring_anchor_plus_complete_traversal_to_mask_lock"
            or runtime.get("clock_start")
            != "after_bank_mapping_and_fixed_shape_runner_prewarm"
            or runtime.get("clock_stop")
            != "final_mask_and_accepted_trajectory_locked"
            or runtime.get("warm_runtime_comparison_authority") is not True
            or runtime.get("run_entry_time_is_cold_inclusive_diagnostic")
            is not True
        ):
            raise ValueError("mask-lock warm clock semantics are incomplete")
        if (
            runtime.get("terminal_preparation_included") is not False
            or runtime.get("terminal_validation_included") is not False
        ):
            raise ValueError("mask-lock runtime scope includes a terminal phase")

        authority = self.authority
        required_true = (
            "mask_generation_completed",
            "completed_mask_lock",
            "frozen_mask_comparator_authority",
            "accepted_trajectory_binding_authority",
            "warm_runtime_comparison_authority",
        )
        if any(authority.get(name) is not True for name in required_true):
            raise ValueError("completed mask-lock comparison authority is incomplete")
        required_false = (
            "new_pruning_commit_authority",
            "mask_mutation_authority",
            "terminal_preparation_authority",
            "terminal_calibration_authority",
            "terminal_validation_authority",
            "terminal_preparation_performed",
            "terminal_calibration_performed",
            "terminal_validation_opened",
            "evaluation_rows_materialized",
            "scientific_functionality_claim_authority",
            "pareto_superiority_claim",
        )
        if any(authority.get(name) is not False for name in required_false):
            raise ValueError(
                "train-only mask-lock receipt crossed its authority ceiling"
            )
        if authority.get("evaluation_data_plane_call_count") != 0:
            raise ValueError("evaluation data plane was touched before mask lock")

        declared_receipt_hash = str(self.receipt_hash)
        observed_receipt_hash = content_hash(
            self._hash_payload(),
            prefix=_V0_TRAIN_ONLY_MASK_LOCK_RECEIPT_PREFIX,
        )
        if declared_receipt_hash and declared_receipt_hash != observed_receipt_hash:
            raise ValueError("v0 train-only mask-lock receipt differs from its hash")
        object.__setattr__(self, "receipt_hash", observed_receipt_hash)

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "workflow_identity": self.workflow_identity,
            "model_source": self.model_source,
            "source_code": self.source_code,
            "train_population": self.train_population,
            "sealed_validation_axis": self.sealed_validation_axis,
            "route_axis": self.route_axis,
            "final_mask": self.final_mask,
            "accepted_trajectory": self.accepted_trajectory,
            "algorithm": self.algorithm,
            "warm_runtime": self.warm_runtime,
            "authority": self.authority,
            **({"loss_weighting": "per_image"}
               if (self.warm_runtime.get("effective_settings") or {}).get("IRG_LOSS_WEIGHTING") == "per_image"
               else {}),
            **(
                {} if self.phase_timing is None
                else {"phase_timing": self.phase_timing}
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return _json_receipt_copy(
            {**self._hash_payload(), "receipt_hash": self.receipt_hash,
             "effective_settings": effective_settings(resolved=self.warm_runtime.get("effective_settings"))}
        )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> "V0TrainOnlyMaskLockReceipt":
        if not isinstance(payload, Mapping):
            raise TypeError("v0 train-only mask-lock receipt must be a mapping")
        values = dict(payload)
        return cls(
            model_source=values.get("model_source", {}),
            source_code=values.get("source_code", {}),
            train_population=values.get("train_population", {}),
            sealed_validation_axis=values.get("sealed_validation_axis", {}),
            route_axis=values.get("route_axis", {}),
            final_mask=values.get("final_mask", {}),
            accepted_trajectory=values.get("accepted_trajectory", {}),
            algorithm=values.get("algorithm", {}),
            warm_runtime=values.get("warm_runtime", {}),
            authority=values.get("authority", {}),
            phase_timing=values.get("phase_timing"),
            schema_name=str(values.get("schema_name", "")),
            workflow_identity=str(values.get("workflow_identity", "")),
            receipt_hash=str(values.get("receipt_hash", "")),
        )


def _validate_complete_imagenet_axis_metadata(
    axis: Any,
    *,
    role: str,
) -> dict[str, Any]:
    """Validate a complete axis without opening one image or tensor row."""

    if role == "train":
        split = IMAGENET1K_TRAIN_SPLIT
        selector_role = IMAGENET1K_TRAIN_ROLE
        context_count = IMAGENET1K_TRAIN_CONTEXT_COUNT
    elif role == "validation":
        split = IMAGENET1K_VALIDATION_SPLIT
        selector_role = IMAGENET1K_VALIDATION_ROLE
        context_count = IMAGENET1K_VALIDATION_CONTEXT_COUNT
    else:  # pragma: no cover - private exact call sites
        raise ValueError("ImageNet axis role must be train or validation")
    selector = getattr(axis, "selector", None)
    records = getattr(axis, "records", None)
    if not isinstance(selector, Mapping) or records is None:
        raise TypeError(f"complete ImageNet {role} axis is not typed")
    class_order = selector.get("class_order")
    if (
        getattr(axis, "dataset_id", None) != IMAGENET1K_DATASET_ID
        or getattr(axis, "split", None) != split
        or selector.get("role") != selector_role
        or selector.get("complete_split") is not True
        or len(records) != context_count
        or selector.get("class_count") != IMAGENET1K_CLASS_COUNT
        or isinstance(class_order, (str, bytes))
        or not isinstance(class_order, Sequence)
        or len(class_order) != IMAGENET1K_CLASS_COUNT
        or len(set(map(str, class_order))) != IMAGENET1K_CLASS_COUNT
    ):
        raise ValueError(f"mask-lock mode requires complete ImageNet {role}")
    axis_hash = _required_sha256(
        getattr(axis, "content_hash", None),
        name=f"{role} context axis hash",
    )
    ids_hash = _required_sha256(
        getattr(axis, "ids_hash", None),
        name=f"{role} ordered context IDs hash",
    )
    label_map_hash = _required_sha256(
        getattr(axis, "label_map_hash", None),
        name=f"{role} label-map hash",
    )
    return {
        "schema": f"complete_imagenet_{role}_axis_binding_v1",
        "dataset_id": IMAGENET1K_DATASET_ID,
        "split": split,
        "role": selector_role,
        "complete_split": True,
        "context_count": context_count,
        "class_count": IMAGENET1K_CLASS_COUNT,
        "context_axis_hash": axis_hash,
        "ordered_context_ids_hash": ids_hash,
        "label_map_hash": label_map_hash,
    }


def _validate_v0_train_only_mask_lock_request(
    *,
    enabled: bool,
    version: str,
    comparison_validation_mode: str,
    maximum_commits: int | None,
    historical_mask_path: str | Path | None,
    v0_schedule_replay_spec: str | Path | None,
    terminal_namespace_salt: str | None,
    terminal_category_matrix: bool,
    step_category_matrix: bool,
) -> None:
    """Refuse noncanonical mask-lock requests before runtime side effects."""

    if not enabled:
        return
    if version != "v0":
        raise ValueError("train-only mask lock is restricted to irg/v0")
    incompatible: list[str] = []
    if comparison_validation_mode != "v0_marginal":
        incompatible.append("comparison_validation_mode")
    if maximum_commits is not None:
        incompatible.append("maximum_commits")
    if historical_mask_path is not None:
        incompatible.append("historical_mask_path")
    if v0_schedule_replay_spec is not None:
        incompatible.append("v0_schedule_replay_spec")
    if terminal_namespace_salt is not None:
        incompatible.append("terminal_namespace_salt")
    if terminal_category_matrix:
        incompatible.append("terminal_category_matrix")
    if step_category_matrix:
        incompatible.append("step_category_matrix")
    if incompatible:
        raise ValueError(
            "train-only v0 mask lock received incompatible modes: "
            + ",".join(incompatible)
        )


def _evaluation_data_plane_call_count(receipt: Mapping[str, Any]) -> int:
    roles = receipt.get("roles")
    if not isinstance(roles, Mapping):
        raise TypeError("evaluation data-plane receipt lacks role counters")
    count = 0
    for role, counters in roles.items():
        if not isinstance(counters, Mapping):
            raise TypeError(f"evaluation role {role!r} counters are invalid")
        calls = counters.get("calls")
        if isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
            raise ValueError(f"evaluation role {role!r} call count is invalid")
        count += int(calls)
    receipt_count = receipt.get("receipt_count")
    if (
        isinstance(receipt_count, bool)
        or not isinstance(receipt_count, int)
        or receipt_count < 0
    ):
        raise ValueError("evaluation receipt count is invalid")
    if count != int(receipt_count):
        raise ValueError("evaluation role calls differ from emitted panel receipts")
    return count


def _accepted_trajectory_mask_lock_binding(
    trajectory: AcceptedV0Trajectory,
) -> dict[str, Any]:
    if not isinstance(trajectory, AcceptedV0Trajectory):
        raise TypeError("train-only mask lock requires a typed accepted trajectory")
    receipt = trajectory.receipt
    return {
        "schema": "modern_v0_train_only_accepted_trajectory_binding_v1",
        "available": True,
        "receipt_hash": receipt.receipt_hash,
        "root_mask_hash": receipt.root_mask_hash,
        "terminal_mask_hash": receipt.terminal_mask_hash,
        "run_count": int(receipt.run_count),
        "total_decision_count": int(receipt.total_decision_count),
        "accepted_count": int(receipt.accepted_count),
        "rejected_count": int(receipt.rejected_count),
        "unresolved_count": int(receipt.unresolved_count),
        "run_manifest_hash": receipt.run_manifest_hash,
        "decision_manifest_hash": receipt.decision_manifest_hash,
        "nonaccepted_manifest_hash": receipt.nonaccepted_manifest_hash,
        "entry_hashes": list(receipt.entry_hashes),
        "entries": [
            {
                "trajectory_index": int(entry.trajectory_index),
                "run_index": int(entry.run_index),
                "proposal_index": int(entry.proposal_index),
                "query_id": entry.query_id,
                "layer_id": entry.layer_id,
                "partition_id": entry.partition_id,
                "candidate_id": entry.candidate_id,
                "parent_mask_hash": entry.parent_mask_hash,
                "resulting_mask_hash": entry.resulting_mask_hash,
                "provider_state_hash": entry.provider_state_hash,
                "decision_proposal_id": entry.decision_proposal_id,
                "decision_receipt_hash": entry.decision_receipt_hash,
                "decision_candidate_evidence_hash": (
                    entry.decision_candidate_evidence_hash
                ),
                "entry_hash": entry.entry_hash,
            }
            for entry in trajectory.entries
        ],
        "authority": trajectory.receipt.authority.to_dict(),
    }


def _build_v0_train_only_mask_lock_receipt(
    *,
    runtime: Any,
    study: StudySpec,
    plan: CircuitPlan,
    profile: Any,
    optimization_axis: Any,
    evaluation_axis: Any,
    comparison_namespace: str,
    requested_partition_family: str,
    partition_family: str,
    partition_random_seed: int | None,
    query_id: str,
    target_class_id: int,
    random_seed: int,
    sensitivity_tau: float,
    maximum_relative_bce_increase: float,
    v0_calibration_decision_estimand: str,
    v0_oracle_minimum_per_stratum: int,
    v0_accept_replication_gate: bool,
    v0_attribution_order: str,
    v0_oracle_sampling_law: str,
    v0_oracle_acquisition_envelope_seconds: float | None,
    v0_oracle_sampling_law_override: str | None,
    dense_baseline_mode: str,
    decision_sampling: Any,
    decision_oracle_sampling: Any,
    loss_span_certificate: LossSpanCertificate,
    hard_partition_measurement: Mapping[str, Any],
    workflow_details: Mapping[str, Any],
    dense_mask: HardMask,
    final_mask: HardMask,
    route_ids: Sequence[str],
    accepted_trajectory: AcceptedV0Trajectory,
    decision_provider: NativeTorchCalibratedBCEOracleProvider,
    scoring_and_anchor_seconds: float,
    traversal_seconds: float,
    run_entry_to_mask_lock_seconds: float,
    traversal_complete: bool,
    extraction_gpu_activity: Mapping[str, Any],
    model_source_replica: ModelSourceReplica | None,
    model_transform_block: Mapping[str, Any] | None = None,
    v0_decision_law_id: str = SEQUENTIAL_CALIBRATED_BCE_V4,
    phase_timing: Mapping[str, Any] | None = None,
) -> V0TrainOnlyMaskLockReceipt:
    """Build the authority-inert receipt before any terminal continuation."""

    if v0_decision_law_id not in {
        SEQUENTIAL_CALIBRATED_BCE_V4,
        SEQUENTIAL_CALIBRATED_BCE_V5,
    }:
        raise ValueError("unknown v0 decision law")

    train_axis = _validate_complete_imagenet_axis_metadata(
        optimization_axis,
        role="train",
    )
    validation_axis = _validate_complete_imagenet_axis_metadata(
        evaluation_axis,
        role="validation",
    )
    optimization_plane = runtime.optimization_data_plane.runtime_receipt()
    evaluation_plane = runtime.evaluation_data_plane.runtime_receipt()
    evaluation_calls = _evaluation_data_plane_call_count(evaluation_plane)
    if evaluation_calls != 0:
        raise RuntimeError("evaluation rows were materialized before v0 mask lock")
    if decision_provider.frozen_category_validation_is_open:
        raise RuntimeError("category validation opened before v0 mask lock")
    if evaluation_plane.get("axis_hash") != train_axis["context_axis_hash"]:
        raise RuntimeError(
            "train-only evaluation guard plane differs from ImageNet train"
        )
    if optimization_plane.get("axis_hash") != train_axis["context_axis_hash"]:
        raise RuntimeError("optimization data plane differs from full ImageNet train")
    train_bank_source_hash = optimization_plane.get("source_receipt_hash")
    _required_sha256(
        train_bank_source_hash,
        name="optimization bank source receipt hash",
    )
    if (
        optimization_plane.get("optimized_bank") is not True
        or optimization_plane.get("node_local_staged") is not True
    ):
        raise RuntimeError(
            "train-only mask lock requires the verified node-local ImageNet bank"
        )
    train_axis.update(
        {
            "bank_source_receipt_hash": train_bank_source_hash,
            "bank_source_id": str(optimization_plane.get("source_id", "")),
            "optimized_bank": True,
            "node_local_staged": True,
            "decision_sampling_design_hash": str(decision_sampling.content_hash),
            "decision_oracle_sampling_design_hash": str(
                decision_oracle_sampling.content_hash
            ),
            "decision_source_population_counts": {
                str(name): len(values)
                for name, values in (
                    decision_sampling.source_context_ids_by_stratum.items()
                )
            },
            "oracle_source_population_counts": {
                str(name): len(values)
                for name, values in (
                    decision_oracle_sampling.source_context_ids_by_stratum.items()
                )
            },
        }
    )
    validation_axis.update(
        {
            "outcomes_opened": False,
            "runtime_data_plane_constructed": False,
            "runtime_guard_plane_role": "complete_imagenet_train_alias",
            "runtime_guard_plane_axis_hash": evaluation_plane.get("axis_hash"),
            "runtime_guard_plane_call_count": 0,
        }
    )

    model_load_receipt = runtime.loaded.receipt
    model_load_receipt.validate()
    source_code = _sealed_source_identity()
    model_source = {
        "schema": "modern_v0_train_only_model_source_binding_v1",
        "model_id": model_load_receipt.model_id,
        "architecture_family": model_load_receipt.architecture_family,
        "model_state_hash": model_load_receipt.state_dict_hash,
        "model_load_receipt_hash": model_load_receipt.content_hash,
        "model_load_receipt": model_load_receipt.to_dict(),
        "strict_load": bool(model_load_receipt.strict_load),
        "claim_eligible_source": bool(model_load_receipt.claim_eligible_source),
        "source_replica_binding_hash": (
            None if model_source_replica is None else model_source_replica.binding_hash
        ),
        # Present only for a registered load-time transform.
        **(
            {}
            if model_transform_block is None
            else {"model_transform": dict(model_transform_block)}
        ),
    }

    final_mask.assert_same_axis(dense_mask)
    if final_mask.route_count != len(route_ids):
        raise ValueError("final mask route count differs from the physical route axis")
    trajectory_binding = _accepted_trajectory_mask_lock_binding(accepted_trajectory)
    if accepted_trajectory.root_mask != dense_mask:
        raise ValueError("accepted trajectory does not start at the dense mask")
    if accepted_trajectory.terminal_mask != final_mask:
        raise ValueError("accepted trajectory does not end at the final mask")
    route_axis = {
        "schema": "modern_v0_train_only_route_axis_binding_v1",
        "route_axis_hash": final_mask.route_axis_hash,
        "ordered_route_ids_hash": ordered_string_axis_hash(
            tuple(route_ids),
            name="v0 train-only physical route axis",
        ),
        "route_count": int(final_mask.route_count),
        "dense_mask_hash": dense_mask.mask_hash,
    }
    packed_sha256 = hashlib.sha256(final_mask.packed_bits).hexdigest()
    final_mask_binding = {
        "schema": "modern_v0_train_only_final_mask_binding_v1",
        "route_axis_hash": final_mask.route_axis_hash,
        "route_count": int(final_mask.route_count),
        "mask_hash": final_mask.mask_hash,
        "packed_sha256": packed_sha256,
        "packed_byte_count": len(final_mask.packed_bits),
        "packing": "numpy_packbits_big_endian_zero_padding_v1",
        "retained_route_count": int(final_mask.retained_count),
        "removed_route_count": int(final_mask.route_count - final_mask.retained_count),
        "retained_fraction": float(final_mask.retained_count / final_mask.route_count),
    }

    execution_identity = str(workflow_details.get("execution_identity", ""))
    attribution_order = str(v0_attribution_order)
    if attribution_order not in V0_ATTRIBUTION_ORDER_MODES:
        raise ValueError("mask-lock mode received an unregistered attribution order")
    expected_execution_identity = (
        _V0_CORRECTED_ATTRIBUTION_EXECUTION_IDENTITY
        if attribution_order == "corrected_aumann_shapley_global_attribution"
        else _V0_DEFAULT_EXECUTION_IDENTITY
    )
    if execution_identity != expected_execution_identity:
        raise ValueError("mask-lock mode received a noncanonical v0 algorithm")
    corrected_order_receipt = hard_partition_measurement.get(
        "v0_attribution_order"
    )
    if attribution_order == "corrected_aumann_shapley_global_attribution":
        if (
            not isinstance(corrected_order_receipt, Mapping)
            or corrected_order_receipt.get("mode") != attribution_order
            or corrected_order_receipt.get("selection_effect")
            != "route_activity_order_only"
            or corrected_order_receipt.get("local_sign_partition_labels_unchanged")
            is not True
            or corrected_order_receipt.get(
                "calibration_oracle_walker_semantics_unchanged"
            )
            is not True
            or corrected_order_receipt.get("incumbent_and_corrected_same_vjp")
            is not False
        ):
            raise ValueError(
                "corrected-order mask lock lacks its exact ablation receipt"
            )
    elif corrected_order_receipt is not None:
        raise ValueError("plain-order mask lock carries a corrected-order receipt")
    if str(v0_oracle_sampling_law) != str(
        _oracle_sampling_receipt_strings(decision_oracle_sampling)[
            "oracle_sampling_law"
        ]
    ):
        raise ValueError(
            "mask-lock mode received a sampling law that differs from the "
            "design the decisions were actually drawn from"
        )
    algorithm_configuration = {
        **({"loss_weighting": "per_image"} if loss_weighting() == "per_image" else {}),
        "schema": "modern_v0_train_only_algorithm_configuration_v1",
        "execution_identity": execution_identity,
        "study_hash": study.content_hash,
        "circuit_plan_hash": plan.content_hash,
        "profile_hash": profile.content_hash,
        "comparison_namespace": str(comparison_namespace),
        "query_id": str(query_id),
        "target_class_id": int(target_class_id),
        "requested_partition_family": str(requested_partition_family),
        "partition_family": str(partition_family),
        "partition_random_seed": partition_random_seed,
        "layer_execution_order": _resolved_layer_execution_order(),
        "random_seed": int(random_seed),
        "maximum_commits": None,
        "sensitivity_tau": float(sensitivity_tau),
        "maximum_relative_bce_increase": float(maximum_relative_bce_increase),
        "v0_calibration_decision_estimand": str(v0_calibration_decision_estimand),
        "v0_oracle_minimum_per_stratum": int(v0_oracle_minimum_per_stratum),
        "v0_accept_replication_gate": bool(v0_accept_replication_gate),
        **(
            {} if v0_decision_law_id == SEQUENTIAL_CALIBRATED_BCE_V4
            else {"v0_decision_law_id": str(v0_decision_law_id)}
        ),
        # v4 decision law.  The sampling law changes what the interval
        # certifies (exact finite-population correction, census terminal), so
        # it is part of the algorithm configuration rather than a runtime
        # option; the envelope is an operational fail-safe that enters no
        # statistic but does change which arms are comparable on runtime.
        "v0_oracle_sampling_law": str(v0_oracle_sampling_law),
        "v0_oracle_acquisition_envelope_seconds": (
            None
            if v0_oracle_acquisition_envelope_seconds is None
            else float(v0_oracle_acquisition_envelope_seconds)
        ),
        **(
            {}
            if v0_oracle_sampling_law_override is None
            else {
                "v0_oracle_sampling_law_override": str(
                    v0_oracle_sampling_law_override
                )
            }
        ),
        **(
            {}
            if attribution_order == "plain_global_attribution"
            else {
                "v0_attribution_order": attribution_order,
                "v0_attribution_order_receipt_hash": content_hash(
                    corrected_order_receipt,
                    prefix="v0-corrected-attribution-order-receipt-",
                ),
            }
        ),
        "dense_baseline_mode": str(dense_baseline_mode),
        "loss_span_certificate_hash": loss_span_certificate.content_hash,
        "accepted_trajectory_receipt_hash": (accepted_trajectory.receipt.receipt_hash),
        "hard_partition_measurement_hash": content_hash(
            hard_partition_measurement,
            prefix="modern-v0-train-only-hard-partition-measurement-",
        ),
        "workflow_details_hash": content_hash(
            workflow_details,
            prefix="modern-v0-train-only-workflow-details-",
        ),
    }
    algorithm_configuration_hash = _v0_train_only_algorithm_configuration_hash(
        algorithm_configuration
    )
    scoring = _nonnegative_finite_seconds(
        scoring_and_anchor_seconds,
        name="scoring_and_anchor_seconds",
    )
    traversal = _nonnegative_finite_seconds(
        traversal_seconds,
        name="traversal_seconds",
    )
    warm_runtime = {
        "schema": "modern_v0_train_only_mask_lock_runtime_scope_v1",
        "effective_settings": _run_effective_settings(runtime),
        "data_plane_effective_settings": _runtime_data_plane_effective_settings(runtime),
        "scope": "warm_scoring_anchor_plus_complete_traversal_to_mask_lock",
        "clock_start": "after_bank_mapping_and_fixed_shape_runner_prewarm",
        "clock_stop": "final_mask_and_accepted_trajectory_locked",
        "scoring_and_anchor_seconds": scoring,
        "complete_traversal_seconds": traversal,
        "warm_mask_lock_seconds": scoring + traversal,
        "run_entry_to_mask_lock_seconds": _nonnegative_finite_seconds(
            run_entry_to_mask_lock_seconds,
            name="run_entry_to_mask_lock_seconds",
        ),
        "warm_runtime_comparison_authority": True,
        "run_entry_time_is_cold_inclusive_diagnostic": True,
        "run_entry_clock_includes_runtime_resolution_and_cold_setup": True,
        "run_entry_clock_excludes_external_source_staging": True,
        "traversal_complete": bool(traversal_complete),
        "terminal_preparation_included": False,
        "terminal_calibration_included": False,
        "terminal_validation_included": False,
        "receipt_finalization_included": False,
        "cold_source_staging_included": False,
        "gpu_activity": dict(extraction_gpu_activity),
        "runtime_execution_profile": (
            runtime.evaluator.runtime_execution_profile.to_runtime_dict()
        ),
        "numerics_record": numerics_record(),
    }
    return V0TrainOnlyMaskLockReceipt(
        model_source=model_source,
        source_code=source_code,
        train_population=train_axis,
        sealed_validation_axis=validation_axis,
        route_axis=route_axis,
        final_mask=final_mask_binding,
        accepted_trajectory=trajectory_binding,
        algorithm={
            "schema": "modern_v0_train_only_algorithm_binding_v1",
            "execution_identity": execution_identity,
            "configuration": algorithm_configuration,
            "configuration_hash": algorithm_configuration_hash,
        },
        warm_runtime=warm_runtime,
        authority={
            "schema": "modern_v0_train_only_mask_lock_authority_v1",
            "artifact_claim_scope": (
                "completed_frozen_v0_mask_lock_accepted_trajectory_and_"
                "measured_warm_pre_finalization_runtime_only"
            ),
            "mask_generation_completed": True,
            "completed_mask_lock": True,
            "frozen_mask_comparator_authority": True,
            "accepted_trajectory_binding_authority": True,
            "warm_runtime_comparison_authority": True,
            "new_pruning_commit_authority": False,
            "mask_mutation_authority": False,
            "terminal_preparation_authority": False,
            "terminal_calibration_authority": False,
            "terminal_validation_authority": False,
            "terminal_preparation_performed": False,
            "terminal_calibration_performed": False,
            "terminal_validation_opened": False,
            "evaluation_rows_materialized": False,
            "evaluation_data_plane_call_count": 0,
            "scientific_functionality_claim_authority": False,
            "pareto_superiority_claim": False,
            "authorized_consumers": [
                "balanced_v0_v1_warm_runtime_comparison",
                "shared_terminal_validation_family_preparation",
            ],
        },
        phase_timing=phase_timing,
    )


def _write_exclusive_fsynced(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o640,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_install_directory_no_replace(
    staging: Path,
    destination: Path,
) -> None:
    """Atomically publish one complete directory without replacing a target."""

    import ctypes
    import errno
    import fcntl
    import platform

    def locked_directory_install_no_replace() -> None:
        """Filesystem-portable cooperative no-replace directory publish.

        Some shared filesystems reject Linux `renameat2(RENAME_NOREPLACE)` for
        directories with `EINVAL` even when ordinary directory rename is
        supported.  The fallback uses a parent-local advisory file lock,
        checks that the immutable destination is still absent while holding
        that lock, then performs a single directory rename.  The kernel
        releases the lock on process death, so a crash cannot strand a stale
        mkdir lease.  All campaign publishers use this routine, making the
        lock the shared no-clobber serialization point when native no-replace
        rename is absent.
        """

        lock = destination.parent / f".{destination.name}.publish-lock"
        descriptor = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(
                    errno.EEXIST,
                    os.strerror(errno.EEXIST),
                    str(destination),
                )
            os.rename(staging, destination)
            _fsync_directory(destination)
            _fsync_directory(destination.parent)
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    source_bytes = os.fsencode(staging)
    destination_bytes = os.fsencode(destination)
    libc = ctypes.CDLL(None, use_errno=True)
    system = platform.system()
    if system == "Linux":
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError(
                "atomic no-replace directory install requires renameat2"
            )
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,  # AT_FDCWD
            source_bytes,
            -100,
            destination_bytes,
            1,  # RENAME_NOREPLACE
        )
    elif system == "Darwin":
        renamex_np = getattr(libc, "renamex_np", None)
        if renamex_np is None:  # pragma: no cover - supported target OS API
            raise RuntimeError(
                "atomic no-replace directory install requires renamex_np"
            )
        renamex_np.argtypes = (
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renamex_np.restype = ctypes.c_int
        result = renamex_np(
            source_bytes,
            destination_bytes,
            0x00000004,  # RENAME_EXCL from <sys/stdio.h>
        )
    else:  # pragma: no cover - production supports Darwin and Linux only
        raise RuntimeError(
            "atomic no-replace directory install is unsupported on " + system
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if system == "Linux" and error_number == errno.EINVAL:
        locked_directory_install_no_replace()
        return
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            str(destination),
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        str(destination),
    )


def _persist_v0_train_only_mask_lock(
    *,
    destination: Path,
    receipt: V0TrainOnlyMaskLockReceipt,
    final_mask: HardMask,
    _before_atomic_install: Callable[[], None] | None = None,
) -> None:
    """Durably publish one complete, crash-atomic no-clobber directory."""

    if destination.exists():
        raise FileExistsError(f"immutable mask-lock output exists: {destination}")
    final_binding = receipt.final_mask
    if (
        final_mask.mask_hash != final_binding.get("mask_hash")
        or final_mask.route_axis_hash != final_binding.get("route_axis_hash")
        or final_mask.route_count != final_binding.get("route_count")
        or hashlib.sha256(final_mask.packed_bits).hexdigest()
        != final_binding.get("packed_sha256")
    ):
        raise ValueError("final-mask bytes differ from the mask-lock receipt")
    receipt_bytes = (
        json.dumps(
            receipt.to_dict(),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.mask-lock-stage-",
            dir=destination.parent,
        )
    )
    stage_mask = staging / "final_mask.packbits"
    stage_receipt = staging / "train_only_mask_lock_receipt.json"
    atomic_install_attempted = False
    atomic_install_completed = False
    try:
        _write_exclusive_fsynced(stage_mask, final_mask.packed_bits)
        _write_exclusive_fsynced(stage_receipt, receipt_bytes)
        _fsync_directory(staging)
        # Validate the complete staged object before its one publication
        # linearization point. A crash or injected failure before rename leaves
        # no destination; after rename the destination contains both files.
        observed = V0TrainOnlyMaskLockReceipt.from_mapping(
            json.loads(
                stage_receipt.read_text(encoding="utf-8")
            )
        )
        if observed.receipt_hash != receipt.receipt_hash:
            raise RuntimeError("staged mask-lock receipt failed readback")
        if (
            hashlib.sha256(stage_mask.read_bytes()).hexdigest()
            != receipt.final_mask["packed_sha256"]
        ):
            raise RuntimeError("staged final mask failed readback")
        if _before_atomic_install is not None:
            _before_atomic_install()
        atomic_install_attempted = True
        _atomic_install_directory_no_replace(staging, destination)
        atomic_install_completed = True
        _fsync_directory(destination.parent)
    finally:
        if atomic_install_completed:
            pass
        elif atomic_install_attempted and staging.exists():
            # Publication failures after a fully validated staging directory are
            # recoverability events, not partial-write events. Preserve the
            # staged mask and receipt for operator recovery and, more
            # importantly, do not mask the original filesystem error with a
            # cleanup failure after a multi-hour H200 run.
            try:
                _fsync_directory(staging)
                _fsync_directory(destination.parent)
            except OSError:
                pass
        else:
            for path in (stage_mask, stage_receipt):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            try:
                staging.rmdir()
            except FileNotFoundError:
                pass


def _finalize_v0_train_only_mask_lock(
    *,
    destination: Path,
    final_mask: HardMask,
    gpu_activity_sampler: GpuActivitySampler,
    receipt_kwargs: Mapping[str, Any],
    phase_timing=None,
) -> Mapping[str, Any]:
    """Stop extraction timing, export the mask lock, and grant no next phase."""

    extraction_gpu_activity = gpu_activity_sampler.stop()
    if phase_timing is not None and phase_timing.enabled:
        phase_timing.mark("receipt_write")
    receipt = _build_v0_train_only_mask_lock_receipt(
        **dict(receipt_kwargs),
        extraction_gpu_activity=extraction_gpu_activity,
        phase_timing=(
            None if phase_timing is None or not phase_timing.enabled
            else phase_timing.receipt_block()
        ),
    )
    _persist_v0_train_only_mask_lock(
        destination=destination,
        receipt=receipt,
        final_mask=final_mask,
    )
    payload = receipt.to_dict()
    _progress(
        "v0_train_only_mask_lock_completed",
        receipt_hash=receipt.receipt_hash,
        final_mask_hash=final_mask.mask_hash,
        retained_route_count=final_mask.retained_count,
        validation_opened=False,
    )
    return {
        **payload,
        "version_name": "v0",
        "output_dir": str(destination),
        "retained_route_count": int(final_mask.retained_count),
        "retained_fraction": float(final_mask.retained_count / final_mask.route_count),
        "final_mask_path": str(destination / "final_mask.packbits"),
        "mask_lock_receipt_path": str(
            destination / "train_only_mask_lock_receipt.json"
        ),
    }


# The pilot surface admits the registered architecture families;
# each family carries its own analytic fully-pruned proof, and receipts
# always disclose architecture_family from the model-load receipt.
# ResNet18 output remains byte-identical (its schema strings and proof
# are untouched).
_REGISTERED_PILOT_ARCHITECTURE_FAMILIES = frozenset(
    {
        "resnet18",
        "resnet18_conv_kernel_weight",
        "alexnet",
        "alexnet_conv_kernel_weight",
        "vgg11_conv_kernel_weight",
        "resnet50_conv_kernel_weight",
    }
)
_KERNEL_WEIGHT_ARCHITECTURE_FAMILIES = frozenset(
    {
        "resnet18_conv_kernel_weight",
        "vgg11_conv_kernel_weight",
        "resnet50_conv_kernel_weight",
    }
)


def _validate_architecture_feature_admission(
    *,
    architecture_family: str,
    enable_local_regime_diagnostics: bool,
    schedule_replay_requested: bool,
) -> None:
    """Refuse optional modes without a kernel-tap scientific contract."""

    family = str(architecture_family).lower()
    if family in _KERNEL_WEIGHT_ARCHITECTURE_FAMILIES:
        if enable_local_regime_diagnostics:
            raise ValueError(
                "kernel-tap architectures do not yet admit local activation-"
                "regime diagnostics"
            )
        if schedule_replay_requested:
            raise ValueError(
                "historical schedule replay does not admit kernel-tap "
                "architecture families"
            )


PilotVersion = str
PilotPartitionFamily = str
_VERSIONS = {"v0"}
_PARTITIONS = {
    "functional_atlas",
    "local_attribution_sign",
    "global_attribution_sign",
    "historical_spatial_weight_sum_sign",
    "randomized_count_matched",
    "single_population",
    # Replay-only aliases retained so historical launch receipts still parse.
    "structural_sign",
    "historical_weight_sign",
    "attributional_sign",
    "randomized_2_partition",
}


def _registered_precision_policy(
    version: str,
    policy: IndependentSampleSizePolicy,
    *,
    oracle_sampling_law_override: str | None = None,
) -> IndependentSampleSizePolicy:
    """Bind authoritative modern v0 to confidence-only stopping and the v4 law.

    The registered v0 identity owns three acquisition facts and takes them
    from nowhere else: precision-only stopping, no per-candidate context
    horizon, and the proposal-adjudication SAMPLING LAW.  The law is bound to
    the sealed without-replacement id because that is what makes the paired
    oracle terminate at census; ``oracle_sampling_law_override`` restores the
    with-replacement id for byte-identical historical replay and is itself
    receipted, so an overridden arm is never mistaken for a registered one.

    ``oracle_acquisition_envelope_seconds`` is caller policy (an operational
    fail-safe that enters no statistic) and is passed through untouched.
    """

    if str(version) != "v0":
        return policy
    if (
        oracle_sampling_law_override is not None
        and str(oracle_sampling_law_override) not in ORACLE_SAMPLING_LAWS
    ):
        raise ValueError(
            "v0 oracle sampling law override must be one of "
            f"{ORACLE_SAMPLING_LAWS!r}, got {oracle_sampling_law_override!r}"
        )
    return replace(
        policy,
        precision_only_iid=True,
        oracle_maximum_contexts_per_candidate=None,
        oracle_sampling_law=(
            SEALED_WOR_CLASS_STRATIFIED_SAMPLING_ID
            if oracle_sampling_law_override is None
            else str(oracle_sampling_law_override)
        ),
    )


def _decision_oracle_sampling_design(
    axis: Any,
    *,
    target_class_id: int,
    randomization_namespace: str,
    oracle_sampling_law: str,
) -> Any:
    """Compile the class-resolved adjudication design under a declared law.

    Both factories see byte-identical strata and estimand weights; only the
    sampling law differs, and the two designs hash differently on identical
    populations so a receipt cannot confuse them.
    """

    law = str(oracle_sampling_law)
    if law not in ORACLE_SAMPLING_LAWS:
        raise ValueError(
            f"unsupported oracle sampling law {law!r}; registered laws are "
            f"{ORACLE_SAMPLING_LAWS!r}"
        )
    factory = (
        class_stratified_binary_sealed_without_replacement_design_from_axis
        if law == SEALED_WOR_CLASS_STRATIFIED_SAMPLING_ID
        else class_stratified_binary_iid_sampling_design_from_axis
    )
    return factory(
        axis,
        target_class_id=int(target_class_id),
        randomization_namespace=str(randomization_namespace),
    )


def _oracle_sampling_receipt_strings(design: Any) -> dict[str, str]:
    """Derive the receipt vocabulary from the design that was actually bound.

    Nothing here is a free string: the law is read off the design, so a
    receipt cannot claim a law the provider did not sample under.
    """

    if design is None:
        return {
            "oracle_sampling": "fresh_proposal_empirical_iid_with_replacement",
            "proposal_adjudication_stream": (
                "fresh_proposal_empirical_iid_with_replacement"
            ),
            "oracle_sampling_law": IID_WITH_REPLACEMENT_SAMPLING_ID,
            "sampling_with_replacement": True,
        }
    if bool(design.sampling_with_replacement):
        return {
            "oracle_sampling": (
                "fresh_proposal_class_stratified_empirical_iid_with_replacement"
            ),
            "proposal_adjudication_stream": (
                "fresh_proposal_class_stratified_empirical_iid_with_replacement"
            ),
            "oracle_sampling_law": IID_WITH_REPLACEMENT_SAMPLING_ID,
            "sampling_with_replacement": True,
        }
    return {
        "oracle_sampling": (
            "fresh_proposal_class_stratified_sealed_without_replacement"
        ),
        "proposal_adjudication_stream": (
            "fresh_proposal_class_stratified_sealed_without_replacement"
        ),
        "oracle_sampling_law": SEALED_WOR_CLASS_STRATIFIED_SAMPLING_ID,
        "sampling_with_replacement": False,
    }


_PARTITION_ALIASES = {
    "structural_sign": "local_attribution_sign",
    "historical_weight_sign": "historical_spatial_weight_sum_sign",
    "attributional_sign": "global_attribution_sign",
    "randomized_2_partition": "randomized_count_matched",
}


def _validate_v0_attribution_order_mode(
    *,
    mode: str,
    version: str,
    architecture_family: str,
    partition_family: str,
    v1_ordering_key: str,
    v0_schedule_replay_spec: str | Path | None,
) -> None:
    """Fail closed around the corrected-order v0 ablation boundary.

    The opt-in arm changes exactly one estimand consumed by the historical
    walker: route activity inside the existing local-sign partitions.  It is
    therefore intentionally narrower than a general scoring-mode switch.
    """

    selected = str(mode)
    if selected not in V0_ATTRIBUTION_ORDER_MODES:
        raise ValueError(
            f"unsupported v0 attribution order mode {selected!r}; registered "
            f"modes are {V0_ATTRIBUTION_ORDER_MODES!r}"
        )
    if selected == "plain_global_attribution":
        return
    incompatible: list[str] = []
    if str(version) != "v0":
        incompatible.append("version")
    if (
        str(architecture_family).lower()
        not in _V0_CORRECTED_ATTRIBUTION_ARCHITECTURE_FAMILIES
    ):
        incompatible.append("architecture_family")
    if str(partition_family) != "local_attribution_sign":
        incompatible.append("partition_family")
    if str(v1_ordering_key) != "abs_raw":
        incompatible.append("v1_ordering_key")
    if v0_schedule_replay_spec is not None:
        incompatible.append("v0_schedule_replay_spec")
    if incompatible:
        raise ValueError(
            "corrected Aumann-Shapley v0 attribution order received "
            "incompatible modes: "
            + ",".join(incompatible)
        )


CENSUS_FIELD_STATISTIC_NAMES = ORDERING_FIELD_STATISTIC_NAMES + CENSUS_LOSS_STATISTIC_NAMES
from mechanistic_systems.measurement.census_sufficient_statistics import (  # noqa: E402
    ENGINE_ROWS as _CENSUS_ENGINE_ROWS,
    SUFFICIENT_STATISTICS as _CENSUS_SUFFICIENT_STATISTICS,
    accumulate_census_sufficient_statistics as _accumulate_census_sufficient_statistics,
    census_producer as _census_producer,
)


def _p_census_settings(**settings: Any) -> dict[str, Any]:
    """Select P's acquisition independently of the consuming arm's labels.

    Keep the registered panel, seed, loss, calibration and look controls.
    Attribution mode is an acquisition setting too: an explicitly selected
    P variant keeps its existing field and stopping rule.
    """
    # The v0 global walk consumes its layerwise census. A v1 cross-layer
    # census or a different ordering key cannot use that acquisition.
    scope = settings.get("prune_order_scope", "layerwise")
    ordering_key = settings.get("v1_ordering_key", "abs_raw")
    if scope not in {"layerwise", "global"}:
        raise ValueError(
            f"shared P census does not support prune_order_scope={scope!r}; "
            "acquisition requires layerwise scope (including v0 global walks)"
        )
    if ordering_key != "abs_raw":
        raise ValueError(
            f"shared P census does not support v1_ordering_key={ordering_key!r}; "
            "acquisition requires abs_raw"
        )
    return {
        **settings,
        "partition_family": "local_attribution_sign",
        "partition_random_seed": 0,
        "required_partition_views": ("local",),
        "prune_order_scope": "layerwise",
        "initial_partition": "positive",
        "v1_ordering_key": "abs_raw",
    }


def _hard_partition_cache_key(
    *,
    runtime: Any,
    optimization_axis: Any,
    query: Any,
    dense_mask: HardMask,
    dense_calibration_state: Any,
    pruning_guiding_loss_id: str = CALIBRATED_OVR_BCE,
    run_id: str,
    random_seed: int,
    sample_size_policy: IndependentSampleSizePolicy,
    enable_local_regime_diagnostics: bool,
    required_partition_views: tuple[str, ...],
    partition_family: PilotPartitionFamily,
    partition_random_seed: int,
    prune_order_scope: str,
    initial_partition: str,
    v1_ordering_key: str = "abs_raw",
    v0_attribution_order: str = V0_DEFAULT_ATTRIBUTION_ORDER,
    census_profile: Any | None = None,
    census_proposal_route_cap: int | None = None,
    census_materiality_resolution: Mapping[str, Any] | None = None,
    census_schedule_replay_schedule: Mapping[str, Any] | None = None,
    census_sample_design: Mapping[str, Any] | None = None,
) -> str:
    if str(v1_ordering_key) not in V1_ORDERING_KEYS:
        raise ValueError("unregistered v1 ordering key")
    if str(v1_ordering_key) != "abs_raw" and str(prune_order_scope) != ("cross_layer"):
        # Key selection is v1-only for now; a layerwise acquisition never
        # consumes the key, so accepting one here would silently mint a
        # cache identity for behavior that cannot occur.
        raise ValueError("ordering-key selection requires the cross_layer scope")
    if str(v0_attribution_order) not in V0_ATTRIBUTION_ORDER_MODES:
        raise ValueError("unregistered v0 attribution order mode")
    key_fields = {
        "schema": "dense_shared_p_loss_field_cache_key_v8",
        **({"loss_weighting": "per_image"} if loss_weighting() == "per_image" else {}),
        "execution_numerics_cache_version": DETERMINISTIC_EXECUTION_CACHE_VERSION,
        "per_layer_field_axis_alignment": "declared_route_block_axis_v1",
        "flip_margin_geometry": "fc_route_crossing_counts_and_layer_margins_v1",
        "model_state_hash": runtime.loaded.receipt.state_dict_hash,
        "route_schema_hash": runtime.evaluator.route_schema_hash,
        "topology_hash": runtime.evaluator.topology_hash,
        "optimization_axis_hash": optimization_axis.content_hash,
        "query_id": query.query_id,
        "query": query,
        "dense_mask_hash": dense_mask.mask_hash,
        "pruning_guiding_loss_id": validate_pruning_guiding_loss_id(pruning_guiding_loss_id),
        "dense_calibration_state_hash": dense_calibration_state_hash(dense_calibration_state),
        "dense_calibration_state_id": str(dense_calibration_state.calibration_state_id),
        "order_field": "loss_activity",
        "census_executable_cut_rule": "count_changing_rungs_in_force_v1",
        "census_rung_profile": _census_rung_profile_receipt(census_profile),
        "census_proposal_route_cap": census_proposal_route_cap,
        "census_schedule_replay_schedule": census_schedule_replay_schedule,
        "census_alpha_spending_rule": "alpha/((k+1)*(k+2))",
        "census_materiality_rule": _CENSUS_MATERIALITY_RULE,
        "census_materiality_resolution": census_materiality_resolution,
        "label_sign_stopping_rule": "registered_label_views_and_loss_order_v1",
        "run_id": run_id,
        "random_seed": int(random_seed),
        "sample_size_policy": sample_size_policy,
        "local_regime_diagnostics": bool(enable_local_regime_diagnostics),
        "census_stopping_surface": "registered_P_local_label_layerwise_order_v1",
        "implementation": "device_float64_census_v63",
        "census_decision_referenced": _census_decision_referenced(),
        "census_criterion": "simultaneous_wrongsign_consequence_upper_bound_bonferroni_blocks_v5",
        "census_epsilon": repr(_census_epsilon()),
        "analytic_sign_partition": _analytic_sign_mode(),
        "analytic_sign_scope": _analytic_sign_scope(),
        "v0_attribution_order": str(v0_attribution_order),
        # The registered census sample design (census fix) is part of the
        # field's identity; the legacy design (None) adds no field, so legacy
        # keys are byte-identical to the production engine's.
        **(
            {}
            if census_sample_design is None
            else {"census_sample_design": dict(census_sample_design)}
        ),
    }
    effective_producer = (
        _CENSUS_ENGINE_ROWS
        if str(v0_attribution_order) == "corrected_aumann_shapley_global_attribution"
        else _census_producer()
    )
    if effective_producer != _CENSUS_ENGINE_ROWS:
        key_fields["census_producer"] = effective_producer
    return content_hash(
        key_fields,
        prefix="irg-dense-partition-field-cache-",
    )


def _shared_category_census_key(
    *, category: str, dense_calibration_state: Any, **cache_settings: Any,
) -> str:
    """Bind the field shared across families to its complete census inputs.

    Family and partition seed select labels/certificates recomputed by each
    consumer. All field, sampling, numerical and acquisition settings remain
    in the binding, together with the full query and frozen calibration.
    """
    common = dict(cache_settings)
    common["dense_calibration_state"] = dense_calibration_state
    query = common["query"]
    common = _p_census_settings(**common)
    return content_hash({
        "schema": "irg_shared_category_census_binding_v4",
        "category": str(category),
        "field_dependency_key": _hard_partition_cache_key(**common),
        "query": query,
        "dense_calibration_state_hash": dense_calibration_state_hash(dense_calibration_state),
        "dense_calibration_state_id": str(dense_calibration_state.calibration_state_id),
        "analytic_sign_active": _census_analytic_sign_active("local_attribution_sign"),
        "census_materiality_rule": _CENSUS_MATERIALITY_RULE,
        "consequence_price_short_circuit": _consequence_price_short_circuit_enabled(),
        "local_covariance_sketch_rank": _local_covariance_sketch_rank(),
        "abs_aggregation_mode": resolve_abs_aggregation_mode(),
    }, prefix="irg-shared-category-census-")


def _load_hard_partition_cache(
    *,
    payload_path: Path,
    receipt_path: Path,
    cache_key: str,
    runtime: Any,
    prefetch: VerifiedNpzPrefetch | None = None,
) -> (
    tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, dict[str, np.ndarray]],
        dict[str, Any],
        dict[str, Any],
    ]
    | None
):
    if not payload_path.exists() or not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("schema") != "dense_hard_partition_field_cache_receipt_v7"
        or receipt.get("cache_key") != cache_key
    ):
        raise ValueError("dense hard-partition cache receipt is invalid")
    # One read of the payload, hashed in file order as
    # it lands in the member arrays (rather than twice: sha256_file,
    # then np.load).  The digest is the whole-file SHA-256 exactly, and it is
    # compared with the receipt before any array is used.  A prefetch started
    # before this lock is used only when the file identity observed here
    # (real path, size, mtime, ctime, inode, device) equals the prefetched one.
    verified_payload = None if prefetch is None else prefetch.take()
    try:
        if verified_payload is not None and (
            verified_payload.identity != _verified_file_identity(payload_path)
        ):
            verified_payload = None
        if verified_payload is None:
            verified_payload = read_verified_npz(payload_path)
    except OSError as error:
        raise ValueError("dense hard-partition cache receipt is invalid") from error
    if receipt.get("payload_sha256") != verified_payload.sha256:
        raise ValueError("dense hard-partition cache receipt is invalid")
    with verified_payload as payload:
        required_flip_margin_fields = {
            "flip_layer_axis_ids",
            "flip_layer_axis_route_counts",
            "flip_count_deactivate",
            "flip_count_activate",
            "margin_layer_axis_ids",
            "margin_capture_status",
            "margin_percentiles",
            "flip_margin_context_counts",
        }
        if not required_flip_margin_fields.issubset(payload.files):
            raise ValueError("dense hard-partition cache omits flip-margin geometry")
        local = np.asarray(payload["local_mean"], dtype=np.float64)
        global_values = np.asarray(payload["global_mean"], dtype=np.float64)
        global_activity = np.asarray(payload["global_activity_mean"], dtype=np.float64)
        statistics = {
            name: np.asarray(payload[name], dtype=np.float64)
            for name in CENSUS_FIELD_STATISTIC_NAMES
        }
        layer_axis_ids = tuple(str(value) for value in payload["layer_axis_ids"])
        layer_axis_route_counts = tuple(
            int(value) for value in payload["layer_axis_route_counts"]
        )
        flip_layer_axis_ids = tuple(
            str(value) for value in payload["flip_layer_axis_ids"]
        )
        flip_layer_axis_route_counts = tuple(
            int(value) for value in payload["flip_layer_axis_route_counts"]
        )
        flat_flip_deactivate = np.asarray(
            payload["flip_count_deactivate"], dtype=np.int32
        )
        flat_flip_activate = np.asarray(payload["flip_count_activate"], dtype=np.int32)
        margin_layer_axis_ids = tuple(
            str(value) for value in payload["margin_layer_axis_ids"]
        )
        margin_capture_status = tuple(
            str(value) for value in payload["margin_capture_status"]
        )
        margin_values = np.asarray(payload["margin_percentiles"], dtype=np.float64)
        flip_margin_context_counts = tuple(
            int(value) for value in payload["flip_margin_context_counts"]
        )
    expected = sum(
        int(np.prod(group.shape, dtype=np.int64)) for group in runtime.route_groups
    )
    if (
        local.shape != (expected,)
        or global_values.shape != (expected,)
        or global_activity.shape != (expected,)
        or np.any(global_activity < 0.0)
        or any(
            statistics[name].shape != (expected,)
            or not np.isfinite(statistics[name]).all()
            or (name != "loss_mean" and np.any(statistics[name] < 0.0))
            for name in CENSUS_FIELD_STATISTIC_NAMES
        )
    ):
        raise ValueError("dense hard-partition cache route axis is invalid")
    if layer_axis_ids != tuple(
        str(group.layer_id) for group in runtime.route_groups
    ) or layer_axis_route_counts != tuple(
        int(np.prod(group.shape, dtype=np.int64)) for group in runtime.route_groups
    ):
        raise ValueError("dense hard-partition cache layer axis is invalid")
    block_by_layer = {
        str(block.layer_id): block for block in runtime.measurement.route_blocks
    }
    expected_flip_layers = tuple(
        str(group.layer_id)
        for group in runtime.route_groups
        if str(getattr(block_by_layer.get(str(group.layer_id)), "route_kind", ""))
        == "linear_weight"
    )
    expected_flip_counts = tuple(
        int(np.prod(group.shape, dtype=np.int64))
        for group in runtime.route_groups
        if str(getattr(block_by_layer.get(str(group.layer_id)), "route_kind", ""))
        == "linear_weight"
    )
    if (
        flip_layer_axis_ids != expected_flip_layers
        or flip_layer_axis_route_counts != expected_flip_counts
        or flat_flip_deactivate.shape != (sum(expected_flip_counts),)
        or flat_flip_activate.shape != (sum(expected_flip_counts),)
        or np.any(flat_flip_deactivate < 0)
        or np.any(flat_flip_activate < 0)
        or len(flip_margin_context_counts) != len(expected_flip_layers)
        or any(value < 0 for value in flip_margin_context_counts)
    ):
        raise ValueError("dense hard-partition cache FC flip axis is invalid")
    expected_margin_layers = tuple(
        str(group.layer_id) for group in runtime.route_groups
    )
    if (
        margin_layer_axis_ids != expected_margin_layers
        or len(margin_capture_status) != len(expected_margin_layers)
        or margin_values.shape
        != (len(expected_margin_layers), len(MARGIN_PERCENTILE_LEVELS))
    ):
        raise ValueError("dense hard-partition cache margin axis is invalid")
    local_by_layer: dict[str, np.ndarray] = {}
    global_by_layer: dict[str, np.ndarray] = {}
    activity_by_layer: dict[str, np.ndarray] = {}
    statistics_by_layer: dict[str, dict[str, np.ndarray]] = {
        name: {} for name in CENSUS_FIELD_STATISTIC_NAMES
    }
    flip_deactivate_by_layer: dict[str, np.ndarray] = {}
    flip_activate_by_layer: dict[str, np.ndarray] = {}
    flip_cursor = 0
    group_by_layer = {str(group.layer_id): group for group in runtime.route_groups}
    for layer_id, count in zip(expected_flip_layers, expected_flip_counts, strict=True):
        stop = flip_cursor + count
        shape = tuple(group_by_layer[layer_id].shape)
        flip_deactivate_by_layer[layer_id] = flat_flip_deactivate[
            flip_cursor:stop
        ].reshape(shape)
        flip_activate_by_layer[layer_id] = flat_flip_activate[flip_cursor:stop].reshape(
            shape
        )
        flip_cursor = stop
    margins_by_layer: dict[str, np.ndarray | str] = {}
    for index, (layer_id, status) in enumerate(
        zip(expected_margin_layers, margin_capture_status, strict=True)
    ):
        if status == "captured":
            row = margin_values[index]
            if not np.isfinite(row).all():
                raise ValueError(
                    "dense hard-partition cache captured margins are invalid"
                )
            margins_by_layer[layer_id] = row
        elif status in {MARGIN_NOT_CAPTURED, "not_applicable"}:
            if not np.isnan(margin_values[index]).all():
                raise ValueError(
                    "dense hard-partition cache absent margins carry values"
                )
            margins_by_layer[layer_id] = status
        else:
            raise ValueError("dense hard-partition cache margin status is invalid")
    conv_layers = tuple(
        layer_id
        for layer_id in expected_margin_layers
        if str(getattr(block_by_layer.get(layer_id), "route_kind", ""))
        in {"conv_channel_pair", "conv_kernel_weight"}
    )
    flip_margin_geometry = {
        "schema": "dense_flip_margin_geometry_v1",
        "flip_count_deactivate": flip_deactivate_by_layer,
        "flip_count_activate": flip_activate_by_layer,
        "margin_percentiles": margins_by_layer,
        "margin_percentile_levels": MARGIN_PERCENTILE_LEVELS,
        "context_count_by_fc_layer": dict(
            zip(
                expected_flip_layers,
                flip_margin_context_counts,
                strict=True,
            )
        ),
        "conv_margins": {layer_id: MARGIN_NOT_CAPTURED for layer_id in conv_layers},
        "conv_flip_counters": "not_emitted",
    }
    validate_flip_margin_geometry(flip_margin_geometry)
    cursor = 0
    for group in runtime.route_groups:
        stop = cursor + int(np.prod(group.shape, dtype=np.int64))
        local_by_layer[group.layer_id] = local[cursor:stop].reshape(group.shape)
        global_by_layer[group.layer_id] = global_values[cursor:stop].reshape(
            group.shape
        )
        activity_by_layer[group.layer_id] = global_activity[cursor:stop].reshape(
            group.shape
        )
        for name in CENSUS_FIELD_STATISTIC_NAMES:
            statistics_by_layer[name][group.layer_id] = statistics[name][
                cursor:stop
            ].reshape(group.shape)
        cursor = stop
    metadata = dict(receipt["measurement_receipt"])
    metadata.update(
        {
            "dense_field_cache_key": cache_key,
            "dense_field_cache_hit": True,
            "dense_field_cache_payload_sha256": receipt["payload_sha256"],
        }
    )
    return (
        local_by_layer,
        global_by_layer,
        activity_by_layer,
        statistics_by_layer,
        flip_margin_geometry,
        metadata,
    )


def _write_field_npz_streaming_hashed(
    stream: Any,
    *,
    arrays: Mapping[str, np.ndarray],
    flat_chunked_arrays: Mapping[str, Sequence[np.ndarray]],
) -> str:
    """Write one deterministic uncompressed NPZ and hash its write stream.

    The wrapper intentionally refuses seeking, which makes ``zipfile`` emit
    data descriptors instead of returning to rewrite member headers.  Every
    archive byte is consequently observed exactly once by the SHA-256 sink;
    no post-write multi-gigabyte read is required.
    """

    import hashlib
    import zipfile

    class _HashingNonSeekableWriter:
        def __init__(self, raw: Any) -> None:
            self.raw = raw
            self.digest = hashlib.sha256()
            self.position = 0

        def write(self, payload: Any) -> int:
            written = int(self.raw.write(payload))
            self.digest.update(memoryview(payload)[:written])
            self.position += written
            return written

        def tell(self) -> int:
            return self.position

        def seek(self, *_args: Any, **_kwargs: Any) -> int:
            raise OSError("field sidecar hash stream is intentionally non-seekable")

        def seekable(self) -> bool:
            return False

        def writable(self) -> bool:
            return True

        def flush(self) -> None:
            self.raw.flush()

    overlap = set(arrays).intersection(flat_chunked_arrays)
    if overlap:
        raise ValueError(f"field sidecar members have two sources: {sorted(overlap)}")
    hashing_stream = _HashingNonSeekableWriter(stream)
    names = tuple(sorted({*arrays, *flat_chunked_arrays}))
    with zipfile.ZipFile(
        hashing_stream,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        for name in names:
            info = zipfile.ZipInfo(
                f"{name}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            with archive.open(info, "w", force_zip64=True) as member:
                if name in arrays:
                    np.lib.format.write_array(
                        member,
                        np.asarray(arrays[name]),
                        allow_pickle=False,
                    )
                    continue
                chunks = tuple(np.asarray(chunk) for chunk in flat_chunked_arrays[name])
                if not chunks:
                    raise ValueError(f"field sidecar member {name!r} has no chunks")
                dtype = chunks[0].dtype
                if any(chunk.dtype != dtype for chunk in chunks[1:]):
                    raise ValueError(f"field sidecar member {name!r} changes dtype")
                size = sum(int(chunk.size) for chunk in chunks)
                np.lib.format.write_array_header_1_0(
                    member,
                    {
                        "descr": np.lib.format.dtype_to_descr(dtype),
                        "fortran_order": False,
                        "shape": (size,),
                    },
                )
                for chunk in chunks:
                    values = np.ascontiguousarray(chunk).reshape(-1)
                    raw = memoryview(values).cast("B")
                    for start in range(0, len(raw), 8 << 20):
                        member.write(raw[start : start + (8 << 20)])
    hashing_stream.flush()
    return hashing_stream.digest.hexdigest()


def _store_hard_partition_cache(
    *,
    payload_path: Path,
    receipt_path: Path,
    cache_key: str,
    runtime: Any,
    local_by_layer: Mapping[str, np.ndarray],
    global_by_layer: Mapping[str, np.ndarray],
    global_activity_by_layer: Mapping[str, np.ndarray],
    ordering_statistics_by_layer: Mapping[str, Mapping[str, np.ndarray]],
    flip_margin_geometry: Mapping[str, Any],
    measurement_receipt: Mapping[str, Any],
) -> str:
    """Write the entry (payload, then receipt); return the payload's sha256.

    A v6 census receipt's pending moment hashes are joined after the payload
    write, so their background computation overlaps it.
    """

    payload_path.parent.mkdir(parents=True, exist_ok=True)
    local_chunks = tuple(
        np.asarray(local_by_layer[group.layer_id], dtype=np.float64).reshape(-1)
        for group in runtime.route_groups
    )
    global_chunks = tuple(
        np.asarray(global_by_layer[group.layer_id], dtype=np.float64).reshape(-1)
        for group in runtime.route_groups
    )
    global_activity_chunks = tuple(
        np.asarray(
            global_activity_by_layer[group.layer_id],
            dtype=np.float64,
        ).reshape(-1)
        for group in runtime.route_groups
    )
    route_count = sum(int(chunk.size) for chunk in local_chunks)
    if (
        sum(int(chunk.size) for chunk in global_chunks) != route_count
        or sum(int(chunk.size) for chunk in global_activity_chunks) != route_count
    ):
        raise ValueError("dense field cache route summaries have incompatible axes")
    if tuple(sorted(ordering_statistics_by_layer)) != tuple(
        sorted(CENSUS_FIELD_STATISTIC_NAMES)
    ):
        raise ValueError(
            "dense hard-partition cache requires the complete ordering "
            "field statistics"
        )
    statistic_chunks = {
        name: tuple(
            np.asarray(
                ordering_statistics_by_layer[name][group.layer_id],
                dtype=np.float64,
            ).reshape(-1)
            for group in runtime.route_groups
        )
        for name in CENSUS_FIELD_STATISTIC_NAMES
    }
    if any(
        sum(int(chunk.size) for chunk in chunks) != route_count
        for chunks in statistic_chunks.values()
    ):
        raise ValueError("dense field cache ordering statistics have wrong axes")
    validate_flip_margin_geometry(flip_margin_geometry)
    block_by_layer = {
        str(block.layer_id): block for block in runtime.measurement.route_blocks
    }
    flip_layers = tuple(
        str(group.layer_id)
        for group in runtime.route_groups
        if str(getattr(block_by_layer.get(str(group.layer_id)), "route_kind", ""))
        == "linear_weight"
    )
    flip_counts = tuple(
        int(np.prod(group.shape, dtype=np.int64))
        for group in runtime.route_groups
        if str(getattr(block_by_layer.get(str(group.layer_id)), "route_kind", ""))
        == "linear_weight"
    )
    deactivate_by_layer = flip_margin_geometry["flip_count_deactivate"]
    activate_by_layer = flip_margin_geometry["flip_count_activate"]
    if set(deactivate_by_layer) != set(flip_layers):
        raise ValueError("dense hard-partition cache requires every FC flip counter")
    flip_deactivate_parts: list[np.ndarray] = []
    flip_activate_parts: list[np.ndarray] = []
    group_by_layer = {str(group.layer_id): group for group in runtime.route_groups}
    for layer_id in flip_layers:
        shape = tuple(group_by_layer[layer_id].shape)
        deactivate = np.asarray(deactivate_by_layer[layer_id])
        activate = np.asarray(activate_by_layer[layer_id])
        if (
            deactivate.dtype != np.int32
            or activate.dtype != np.int32
            or deactivate.shape != shape
            or activate.shape != shape
        ):
            raise ValueError(
                "dense hard-partition cache FC flip counters have wrong shapes"
            )
        flip_deactivate_parts.append(deactivate.reshape(-1))
        flip_activate_parts.append(activate.reshape(-1))
    # An architecture without FC route blocks still needs a valid empty NPY
    # member; supply one empty chunk so the streaming member has a dtype.
    if not flip_deactivate_parts:
        flip_deactivate_parts.append(np.empty(0, dtype=np.int32))
        flip_activate_parts.append(np.empty(0, dtype=np.int32))
    margins_by_layer = flip_margin_geometry["margin_percentiles"]
    margin_layers = tuple(str(group.layer_id) for group in runtime.route_groups)
    if set(margins_by_layer) != set(margin_layers):
        raise ValueError(
            "dense hard-partition cache requires every layer margin status"
        )
    margin_status: list[str] = []
    margin_rows: list[np.ndarray] = []
    for layer_id in margin_layers:
        margin = margins_by_layer[layer_id]
        if isinstance(margin, str):
            if margin not in {MARGIN_NOT_CAPTURED, "not_applicable"}:
                raise ValueError("dense hard-partition cache margin marker is invalid")
            margin_status.append(margin)
            margin_rows.append(np.full(len(MARGIN_PERCENTILE_LEVELS), np.nan))
        else:
            row = np.asarray(margin, dtype=np.float64)
            if (
                row.shape != (len(MARGIN_PERCENTILE_LEVELS),)
                or not np.isfinite(row).all()
            ):
                raise ValueError(
                    "dense hard-partition cache margin percentiles are invalid"
                )
            margin_status.append("captured")
            margin_rows.append(row)
    context_counts = flip_margin_geometry.get("context_count_by_fc_layer", {})
    if set(context_counts) != set(flip_layers):
        raise ValueError("dense hard-partition cache requires every FC context count")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{cache_key}.",
        suffix=".npz",
        dir=payload_path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            payload_sha256 = _write_field_npz_streaming_hashed(
                stream,
                arrays={
                    # The layer axis makes the payload self-describing for
                    # offline consumers that have no runtime to rebuild
                    # per-layer boundaries from.
                    "layer_axis_ids": np.asarray(
                        [str(group.layer_id) for group in runtime.route_groups]
                    ),
                    "layer_axis_route_counts": np.asarray(
                        [
                            int(np.prod(group.shape, dtype=np.int64))
                            for group in runtime.route_groups
                        ],
                        dtype=np.int64,
                    ),
                    "flip_layer_axis_ids": np.asarray(flip_layers),
                    "flip_layer_axis_route_counts": np.asarray(
                        flip_counts, dtype=np.int64
                    ),
                    "margin_layer_axis_ids": np.asarray(margin_layers),
                    "margin_capture_status": np.asarray(margin_status),
                    "margin_percentiles": np.stack(margin_rows, axis=0),
                    "flip_margin_context_counts": np.asarray(
                        [int(context_counts[layer]) for layer in flip_layers],
                        dtype=np.int64,
                    ),
                },
                flat_chunked_arrays={
                    "local_mean": local_chunks,
                    "global_mean": global_chunks,
                    "global_activity_mean": global_activity_chunks,
                    "flip_count_deactivate": flip_deactivate_parts,
                    "flip_count_activate": flip_activate_parts,
                    **statistic_chunks,
                },
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, payload_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    join_census_moment_hashes(measurement_receipt)
    receipt = {
        "schema": "dense_hard_partition_field_cache_receipt_v7",
        "cache_key": cache_key,
        "payload_sha256": payload_sha256,
        "payload_serialization": "uncompressed_npz_streamed_sha256_v1",
        "payload_hash_passes": 1,
        "route_count": int(route_count),
        "measurement_receipt": dict(measurement_receipt),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{cache_key}.",
        suffix=".json",
        dir=receipt_path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({**receipt, "effective_settings": _run_effective_settings(runtime),
                       "data_plane_effective_settings": _runtime_data_plane_effective_settings(runtime)},
                      stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, receipt_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return payload_sha256


@dataclass(frozen=True)
class _DenseReadoutTableCacheLoad:
    valid: bool
    reason: str | None
    source_context_ids: tuple[str, ...]
    projected_values: np.ndarray
    labels: np.ndarray
    skipped_unknown_ids: int
    row_count: int
    payload_sha256: str | None


def _dense_readout_table_cache_key(
    *,
    runtime: Any,
    optimization_axis: Any,
    provider: NativeTorchCalibratedBCEOracleProvider,
    dense_mask: HardMask,
    target_class_id: int,
) -> str:
    import torch

    device = provider._device
    device_capability = (
        str(torch.cuda.get_device_capability(device))
        if device.type == "cuda" and torch.cuda.is_available()
        else "cpu"
    )
    key_fields: dict[str, Any] = {
        "schema": "dense_projected_readout_table_cache_key_v1",
        "execution_numerics_cache_version": DETERMINISTIC_EXECUTION_CACHE_VERSION,
        "model_state_hash": runtime.loaded.receipt.state_dict_hash,
        "route_schema_hash": runtime.evaluator.route_schema_hash,
        "topology_hash": runtime.evaluator.topology_hash,
        "dense_mask_hash": dense_mask.mask_hash,
        "readout_projection_hash": provider._readout_projection_hash,
        "target_class_id": int(target_class_id),
        "context_axis_hash": optimization_axis.content_hash,
        "context_axis_split": str(optimization_axis.split),
        "forward_tier": int(runtime.evaluator.runtime_execution_profile.forward_tier),
        "torch_version": str(torch.__version__),
        "cuda_version": str(getattr(torch.version, "cuda", None)),
        "device_capability": device_capability,
    }
    return content_hash(
        key_fields,
        prefix="irg-dense-projected-readout-table-",
    )


def _load_dense_readout_table_cache(
    *,
    payload_path: Path,
    receipt_path: Path,
    cache_key: str,
    dense_mask_hash: str,
    context_axis_hash: str,
    forward_tier: int,
    live_source_context_ids: Sequence[str],
    labels_by_context_id: Mapping[str, int],
) -> _DenseReadoutTableCacheLoad | None:
    """Load and validate a persistent dense projected-readout table.

    Invalid or corrupt artifacts are represented as a refusal result so a
    pilot can log the reason and continue with ordinary live forwards.
    """

    if not payload_path.exists() and not receipt_path.exists():
        return None
    payload_sha256: str | None = None

    def refused(reason: str) -> _DenseReadoutTableCacheLoad:
        return _DenseReadoutTableCacheLoad(
            valid=False,
            reason=str(reason),
            source_context_ids=(),
            projected_values=np.empty((0, 0), dtype=np.float32),
            labels=np.empty((0,), dtype=np.int64),
            skipped_unknown_ids=0,
            row_count=0,
            payload_sha256=payload_sha256,
        )

    if not payload_path.exists() or not receipt_path.exists():
        return refused("payload_or_receipt_missing")
    try:
        payload_sha256 = sha256_file(payload_path)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("schema") != "dense_projected_readout_table_cache_receipt_v1"
            or receipt.get("cache_key") != cache_key
            or receipt.get("payload_sha256") != payload_sha256
            or receipt.get("dense_mask_hash") != str(dense_mask_hash)
            or receipt.get("context_axis_hash") != str(context_axis_hash)
            or int(receipt.get("forward_tier", -1)) != int(forward_tier)
        ):
            return refused("receipt_identity_or_digest_mismatch")
        with np.load(payload_path, allow_pickle=False) as payload:
            source_context_ids = tuple(
                str(value) for value in payload["source_context_ids"]
            )
            projected_values = np.asarray(
                payload["projected_values"],
                dtype=np.float32,
            )
            labels = np.asarray(payload["labels"], dtype=np.int64)
        receipt_row_count = int(receipt.get("row_count", -1))
        receipt_readout_width = int(receipt.get("readout_width", -1))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return refused("payload_or_receipt_unreadable")
    row_count = len(source_context_ids)
    if (
        len(set(source_context_ids)) != row_count
        or projected_values.ndim != 2
        or labels.ndim != 1
        or projected_values.shape[0] != row_count
        or labels.shape != (row_count,)
        or receipt_row_count != row_count
        or receipt_readout_width != int(projected_values.shape[1])
        or not np.isfinite(projected_values).all()
    ):
        return refused("payload_axes_or_values_invalid")
    live_ids = {str(value) for value in live_source_context_ids}
    kept_positions = tuple(
        index
        for index, source_context_id in enumerate(source_context_ids)
        if source_context_id in live_ids
    )
    kept_source_context_ids = tuple(
        source_context_ids[index] for index in kept_positions
    )
    if any(
        source_context_ids[index] not in labels_by_context_id
        or int(labels[index]) != int(labels_by_context_id[source_context_ids[index]])
        for index in kept_positions
    ):
        return refused("context_axis_label_mismatch")
    return _DenseReadoutTableCacheLoad(
        valid=True,
        reason=None,
        source_context_ids=kept_source_context_ids,
        projected_values=np.asarray(
            projected_values[np.asarray(kept_positions, dtype=np.int64)],
            dtype=np.float32,
        ),
        labels=np.asarray(
            labels[np.asarray(kept_positions, dtype=np.int64)],
            dtype=np.int64,
        ),
        skipped_unknown_ids=row_count - len(kept_positions),
        row_count=row_count,
        payload_sha256=payload_sha256,
    )


def _store_dense_readout_table_cache(
    *,
    payload_path: Path,
    receipt_path: Path,
    cache_key: str,
    source_context_ids: Sequence[str],
    projected_values: Any,
    labels: Any,
    dense_mask_hash: str,
    context_axis_hash: str,
    forward_tier: int,
) -> None:
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    source_ids = tuple(str(value) for value in source_context_ids)
    values = np.asarray(projected_values, dtype=np.float32)
    label_values = np.asarray(labels, dtype=np.int64)
    if (
        len(set(source_ids)) != len(source_ids)
        or values.ndim != 2
        or values.shape[0] != len(source_ids)
        or label_values.shape != (len(source_ids),)
        or not np.isfinite(values).all()
    ):
        raise ValueError("dense readout table export is invalid")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{cache_key}.",
        suffix=".npz",
        dir=payload_path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(
                stream,
                source_context_ids=np.asarray(source_ids),
                projected_values=values,
                labels=label_values,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, payload_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    receipt = {
        "schema": "dense_projected_readout_table_cache_receipt_v1",
        "cache_key": cache_key,
        "payload_sha256": sha256_file(payload_path),
        "row_count": len(source_ids),
        "readout_width": int(values.shape[1]),
        "dense_mask_hash": str(dense_mask_hash),
        "context_axis_hash": str(context_axis_hash),
        "forward_tier": int(forward_tier),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{cache_key}.",
        suffix=".json",
        dir=receipt_path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({**receipt, "effective_settings": effective_settings()}, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, receipt_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _preload_dense_readout_table(
    *,
    provider: NativeTorchCalibratedBCEOracleProvider,
    dense_mask: HardMask,
    loaded: _DenseReadoutTableCacheLoad,
) -> int:
    """Install rows after the loader verified their identity and artifact hash."""

    if not loaded.valid:
        raise ValueError("preload requires a structurally valid table")
    if not loaded.source_context_ids:
        return 0
    return int(provider.preload_projected_readouts(
        dense_mask.mask_hash,
        loaded.source_context_ids,
        loaded.projected_values,
        loaded.labels,
    ))


# ---------------------------------------------------------------------------
# Canonical full-train, full-width dense-readout table.
# Modern-v0 defaults to this exhaustive training-population authority.  A
# sampled dense baseline remains available only as an explicitly named
# exploratory compatibility mode; absence of either canonical artifact in the
# default mode refuses before model/CUDA/data-plane setup.
#
# When ``IRG_CANONICAL_DENSE_TABLE`` names a receipt, its rows are preferred
# by the two observation-only consumers documented on
# ``NativeTorchCalibratedBCEOracleProvider.dense_anchored_step_reading`` and
# ``_StepCategoryStructureAccumulator._dense_exclusion_rows``.  Loading is a
# SEPARATE, provider-side, host/memmap-resident store -- it is never written
# into ``_readout_source_cache`` and never affects ``_readouts``.  Identity
# is REFUSED LOUDLY (never silently ignored or blended) on any mismatch.
# ---------------------------------------------------------------------------

IRG_CANONICAL_DENSE_TABLE_ENV = "IRG_CANONICAL_DENSE_TABLE"
# Exact dense-plane calibration authority.  The artifact is fit once over the
# complete canonical *training* readout table, then frozen before any
# validation response can be materialized.  If explicitly requested, failure
# to bind it is fatal; silently falling back to a finite-sample calibration
# changes both the v0 loss scale and the partition geometry.
IRG_CANONICAL_DENSE_POPULATION_CALIBRATION_ENV = (
    "IRG_CANONICAL_DENSE_POPULATION_CALIBRATION"
)
IRG_LOCAL_COVARIANCE_SKETCH_RANK_ENV = "IRG_LOCAL_COVARIANCE_SKETCH_RANK"
IRG_DENSE_BASELINE_MODE_ENV = "IRG_DENSE_BASELINE_MODE"
IRG_DENSE_BASELINE_REGISTRY_ENV = "IRG_DENSE_BASELINE_REGISTRY"
DENSE_BASELINE_POPULATION_EXACT = "population_exact_full_train"
DENSE_BASELINE_SAMPLED_EXPLORATORY = "sampled_training_exploratory"
IRG_LAYER_EXECUTION_ORDER_ENV = "IRG_LAYER_EXECUTION_ORDER"
_DENSE_BASELINE_REGISTRY_SCHEMA = "irg_dense_baseline_registry_v2"
_DENSE_BASELINE_HEALTH_SCHEMA = "canonical_dense_population_calibration_health_v1"
_DEFAULT_DENSE_BASELINE_REGISTRY = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "irg_v0_v2_pilots"
    / "modern_v0_cross_architecture_default100_20260813"
    / "dense_baseline_registry.json"
)


def _local_covariance_sketch_rank() -> int:
    """Resolve the rank-k covariance sketch retained by field accumulation.

    The sketch is a v1 targeting instrument and has no modern-v0 consumer.
    Rank zero removes its route-scale persistent slab without changing any v0
    estimand, decision, or mask.  The resolved rank remains receipt-bound.
    """

    raw = os.environ.get(IRG_LOCAL_COVARIANCE_SKETCH_RANK_ENV)
    if raw is None or not str(raw).strip():
        # Modern-v0 has no covariance-sketch consumer. Its registered default
        # is rank zero; modern-v1 proposal surfaces must opt into and
        # receipt-bind a positive rank explicitly.
        return 0
    rank = int(str(raw).strip())
    if rank < 0:
        raise ValueError("local covariance sketch rank must be nonnegative")
    return rank


def _automatic_dense_baseline_paths(
    *,
    architecture_family: str,
    model_id: str,
) -> tuple[str, str]:
    """Resolve one exact model entry from the source-bound dense registry."""

    declared = os.environ.get(IRG_DENSE_BASELINE_REGISTRY_ENV)
    registry_path = (
        Path(declared.strip()).expanduser()
        if declared is not None and declared.strip()
        else _DEFAULT_DENSE_BASELINE_REGISTRY
    )
    if not registry_path.is_file():
        raise ValueError(
            "automatic dense-baseline discovery requires the registered "
            f"manifest {registry_path}; override only with "
            f"{IRG_DENSE_BASELINE_REGISTRY_ENV}"
        )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("schema") != _DENSE_BASELINE_REGISTRY_SCHEMA:
        raise ValueError("automatic dense-baseline registry schema mismatch")
    payload = dict(registry)
    claimed = payload.pop("content_sha256", None)
    actual = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if claimed != actual:
        raise ValueError("automatic dense-baseline registry content hash mismatch")
    entries = registry.get("entries")
    if not isinstance(entries, list):
        raise ValueError("automatic dense-baseline registry entries are malformed")
    matches = [
        entry
        for entry in entries
        if isinstance(entry, Mapping)
        and entry.get("architecture_family") == str(architecture_family)
        and entry.get("model_id") == str(model_id)
    ]
    if len(matches) != 1:
        raise ValueError(
            "automatic dense-baseline discovery expected exactly one entry for "
            f"architecture={architecture_family!r}, model_id={model_id!r}; "
            f"found {len(matches)}"
        )
    entry = matches[0]
    if set(entry) != {
        "architecture_family",
        "model_id",
        "table_receipt_path",
        "table_receipt_sha256",
        "population_calibration_receipt_path",
        "population_calibration_receipt_sha256",
        "population_calibration_health_path",
        "population_calibration_health_sha256",
    }:
        raise ValueError(
            "automatic dense-baseline registry entry has unexpected fields"
        )
    resolved: list[str] = []
    for label, path_key, hash_key in (
        (
            "table",
            "table_receipt_path",
            "table_receipt_sha256",
        ),
        (
            "population calibration",
            "population_calibration_receipt_path",
            "population_calibration_receipt_sha256",
        ),
    ):
        path = Path(os.path.expandvars(str(entry[path_key]))).expanduser()
        expected = str(entry[hash_key])
        if len(expected) != 64 or not path.is_file():
            raise ValueError(
                f"automatic dense-baseline {label} artifact is unavailable: {path} "
                "(set the variable its path names, such as AUTHORS_STORE)"
            )
        if sha256_file(path) != expected:
            raise ValueError(f"automatic dense-baseline {label} receipt hash mismatch")
        resolved.append(str(path.resolve()))
    health_path = Path(os.path.expandvars(str(entry["population_calibration_health_path"]))).expanduser()
    expected_health_sha256 = str(entry["population_calibration_health_sha256"])
    if len(expected_health_sha256) != 64 or not health_path.is_file():
        raise ValueError(
            "automatic dense-baseline population calibration health artifact "
            "is unavailable"
        )
    if sha256_file(health_path) != expected_health_sha256:
        raise ValueError(
            "automatic dense-baseline population calibration health hash mismatch"
        )
    health = json.loads(health_path.read_text(encoding="utf-8"))
    health_content_hash = str(health.pop("content_hash", ""))
    if health_content_hash != stable_digest(health):
        raise ValueError(
            "automatic dense-baseline population calibration health content hash mismatch"
        )
    calibration_receipt = json.loads(Path(resolved[1]).read_text(encoding="utf-8"))
    if (
        health.get("schema") != _DENSE_BASELINE_HEALTH_SCHEMA
        or health.get("status") != "passed"
        or health.get("validation_responses_observed") is not False
        or health.get("solver_owner")
        != (
            "mechanistic_systems.backends.factored_calibration."
            "fit_batched_post_query_affine"
        )
        or Path(str(health.get("artifact_receipt_path", ""))).resolve()
        != Path(resolved[1])
        or health.get("artifact_receipt_file_sha256")
        != entry["population_calibration_receipt_sha256"]
        or health.get("artifact_receipt_content_hash")
        != calibration_receipt.get("content_hash")
        or health.get("class_count") != 1000
        or health.get("fit_cell_count") != 2000
    ):
        raise ValueError(
            "automatic dense-baseline population calibration health lineage mismatch"
        )
    return resolved[0], resolved[1]


def _load_pilot_model_transform(
    model_transform_path: str | Path | None,
    *,
    study: StudySpec,
) -> Any | None:
    """Load and admit a registered model transform, or None.

    Argument-only and outcome-independent: it reads the spec (not the payload),
    and refuses before any model, CUDA, store or output side effect when the
    transform names another base model or an unadmitted family.

    Every registered partition family orders routes from the shared dense
    census measured on the transformed model.
    """

    if model_transform_path is None:
        return None
    from mechanistic_systems.measurement.model_transforms import (
        MODEL_TRANSFORM_ADMITTED_FAMILIES,
        load_model_transform,
    )

    spec = load_model_transform(model_transform_path)
    if spec.base_model_id != str(study.model_id):
        raise ValueError(
            "model transform base model differs from the StudySpec model"
        )
    family = str(study.route_schema.architecture_family).lower()
    if family not in MODEL_TRANSFORM_ADMITTED_FAMILIES:
        raise ValueError(
            f"model transforms are admitted only for "
            f"{sorted(MODEL_TRANSFORM_ADMITTED_FAMILIES)!r}; got {family!r}"
        )
    if study.checkpoint_hash != spec.base_checkpoint_sha256:
        raise ValueError(
            "model transform base checkpoint differs from the StudySpec checkpoint"
        )
    return spec


def _validate_pilot_optin_settings() -> dict[str, str]:
    """Resolve both opt-in numerics settings at pilot entry (fail early)."""

    from mechanistic_systems.measurement.native_irg_providers import (
        fixed_offset_affine_convergence_setting,
        frozen_category_readout_numerics_setting,
    )

    return {
        "frozen_category_readout_numerics": frozen_category_readout_numerics_setting(),
        "fixed_offset_affine_convergence": fixed_offset_affine_convergence_setting(),
    }


def _refuse_transformed_arm_receipt(
    arm_receipt: Mapping[str, Any] | None,
    *,
    runtime: Any | None = None,
) -> None:
    """No silent base-model replay.

    Downstream tools that re-evaluate a stored arm mask (recensus, terminal-
    only evaluation, substituted capture, the calibration-closure diagnostic)
    resolve the BASE model.  An arm receipt that carries ``model_transform``
    was extracted on a derived model, so replaying its mask here would
    evaluate it on the wrong network.  Refuse unless the resolved runtime
    itself carries exactly that transform.
    """

    if not isinstance(arm_receipt, Mapping) or "model_transform" not in arm_receipt:
        return
    block = arm_receipt.get("model_transform")
    expected = str(block.get("content_hash", "")) if isinstance(block, Mapping) else ""
    loaded = (
        ""
        if runtime is None
        else str(getattr(runtime.loaded.receipt, "model_transform_hash", "") or "")
    )
    if not expected or loaded != expected:
        raise ValueError(
            "the source arm was extracted on a transformed model "
            f"(model_transform {expected or 'unbound'}); this tool resolved "
            f"{'the base model' if not loaded else 'another transform'} and "
            "refuses to replay the mask on it"
        )


def _refuse_transformed_arm_mask_source(
    mask_or_arm_path: str | Path,
    *,
    runtime: Any | None = None,
) -> None:
    """The file-path form -- read the arm's ``receipt.json`` if present."""

    path = Path(str(mask_or_arm_path))
    receipt_path = (path if path.is_dir() else path.parent) / "receipt.json"
    if not receipt_path.is_file():
        return
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    _refuse_transformed_arm_receipt(receipt, runtime=runtime)


def _require_runtime_carries_model_transform(runtime: Any, model_transform: Any) -> None:
    """Post-runtime re-check that the runtime carries the model transform.

    A transformed arm must run on a runtime whose loaded-model receipt binds
    the spec's content hash and derived state digest; a default arm (no
    transform) is never checked here.
    """

    if model_transform is None:
        return
    receipt = runtime.loaded.receipt
    if (
        getattr(receipt, "model_transform_hash", "") != model_transform.content_hash
        or receipt.state_dict_hash != model_transform.derived_state_dict_sha256
    ):
        raise RuntimeError("resolved runtime does not carry the requested model transform")


def _require_transform_dense_authority(
    model_transform: Any,
    *,
    canonical_dense_table_receipt_path: str | None,
) -> None:
    """A transformed arm needs a table of the DERIVED model.

    Refuses before any GPU work.  The later table identity check remains the
    authoritative, fail-closed binding; this only moves the refusal forward.
    """

    if canonical_dense_table_receipt_path is None:
        return
    receipt_path = Path(str(canonical_dense_table_receipt_path))
    if receipt_path.is_dir():
        receipt_path = receipt_path / "canonical_dense_readout_table_receipt.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(
            "transformed arm cannot read its canonical dense table receipt"
        ) from error
    declared = str(receipt.get("model_state_hash", ""))
    if declared != str(model_transform.derived_state_dict_sha256):
        raise ValueError(
            "transformed arm's canonical dense table was built for another model "
            f"state (table={declared or 'absent'}, derived="
            f"{model_transform.derived_state_dict_sha256}); supply the derived "
            f"model's registry through {IRG_DENSE_BASELINE_REGISTRY_ENV}"
        )


def _resolved_dense_baseline_authority(
    *,
    architecture_family: str | None = None,
    model_id: str | None = None,
) -> tuple[str, str | None, str | None]:
    """Resolve the shared modern dense-forward authority before setup.

    Dense *forward* quantities use the complete sealed training population by
    default.  This does not change attribution acquisition, which remains on
    its separately registered sampled training geometry.
    """

    raw_mode = os.environ.get(IRG_DENSE_BASELINE_MODE_ENV)
    mode = (
        DENSE_BASELINE_POPULATION_EXACT
        if raw_mode is None or not raw_mode.strip()
        else raw_mode.strip().lower()
    )
    if mode not in {
        DENSE_BASELINE_POPULATION_EXACT,
        DENSE_BASELINE_SAMPLED_EXPLORATORY,
    }:
        raise ValueError(
            f"{IRG_DENSE_BASELINE_MODE_ENV} must be "
            f"{DENSE_BASELINE_POPULATION_EXACT!r} or "
            f"{DENSE_BASELINE_SAMPLED_EXPLORATORY!r}, got {raw_mode!r}"
        )

    table_path = os.environ.get(IRG_CANONICAL_DENSE_TABLE_ENV)
    calibration_path = os.environ.get(IRG_CANONICAL_DENSE_POPULATION_CALIBRATION_ENV)
    table_path = table_path.strip() if table_path and table_path.strip() else None
    calibration_path = (
        calibration_path.strip()
        if calibration_path and calibration_path.strip()
        else None
    )
    if mode == DENSE_BASELINE_POPULATION_EXACT:
        if table_path is None and calibration_path is None:
            if architecture_family is not None and model_id is not None:
                table_path, calibration_path = _automatic_dense_baseline_paths(
                    architecture_family=str(architecture_family),
                    model_id=str(model_id),
                )
        missing = tuple(
            name
            for name, path in (
                (IRG_CANONICAL_DENSE_TABLE_ENV, table_path),
                (
                    IRG_CANONICAL_DENSE_POPULATION_CALIBRATION_ENV,
                    calibration_path,
                ),
            )
            if path is None
        )
        if missing:
            raise ValueError(
                "the default v0 dense baseline requires exhaustive full-train "
                "forward quantities; missing " + ", ".join(missing)
            )
    elif table_path is not None or calibration_path is not None:
        raise ValueError(
            f"{DENSE_BASELINE_SAMPLED_EXPLORATORY!r} cannot partially or "
            "silently consume canonical dense artifacts; select "
            f"{DENSE_BASELINE_POPULATION_EXACT!r} instead"
        )
    if calibration_path is not None and _calibration_penalty() == "firth":
        calibration_path = str(_population_calibration_receipt_for_penalty(calibration_path, "firth"))
    return mode, table_path, calibration_path


def _resolved_layer_execution_order() -> str:
    """'forward' (declared route-block order) or 'reverse'.

    Forward is the default, so an unset environment walks the declared
    route-block order. Reverse changes ONLY
    which layer group the walk visits first; the route axis, offsets, ids,
    axis hash, mask layout and within-layer score ordering are untouched, so
    masks stay comparable route-for-route across the two orders.
    """

    value = os.environ.get(IRG_LAYER_EXECUTION_ORDER_ENV)
    if value is None or not value.strip():
        return "forward"
    resolved = value.strip().lower()
    if resolved not in {"forward", "reverse"}:
        raise ValueError(
            f"{IRG_LAYER_EXECUTION_ORDER_ENV} must be 'forward' or 'reverse', "
            f"got {value!r}"
        )
    return resolved


def _resolved_prune_scope_setting(
    *,
    version: str,
    family: str,
    schedule_replay_requested: bool,
    v0_attribution_order: str,
    direct_candidate_comparison: bool,
    architecture_family: str,
    sensitivity_tau: float,
) -> Any:
    """The walk's prune scope and proposal grain, resolved at entry.

    ``IRG_PRUNE_SCOPE`` (``layerwise`` | ``global``),
    ``IRG_PROPOSAL_GRAIN`` (``nominal`` | ``layerwise_matched``), and
    ``IRG_PROPOSAL_ROUTE_CAP`` (``none`` | a positive integer). All unset
    means the registered layerwise walk on the nominal ladder: the returned
    setting is then the default and every scope statement below is skipped by
    its guard. Registered global walks require ``layerwise_matched`` and no
    cap; nominal and capped settings are diagnostic only. A variable that is
    set but blank is refused. Unknown values,
    and every combination the new walk is not built for (any tau other than
    the registered 5e-7, any architecture family other than AlexNet at
    kernel-weight granularity, and the modes ``validate_prune_scope_admission``
    lists), are refused here, before any model, CUDA, store or output side
    effect.  See ``prune_scope_walk``.
    """

    from .prune_scope_walk import (
        resolve_prune_scope_setting,
        validate_prune_scope_admission,
    )

    setting = resolve_prune_scope_setting()
    if setting.is_default:
        return setting
    validate_prune_scope_admission(
        setting,
        version=version,
        family=family,
        schedule_replay_requested=schedule_replay_requested,
        layer_execution_order=_resolved_layer_execution_order(),
        v0_attribution_order=v0_attribution_order,
        direct_candidate_comparison=direct_candidate_comparison,
        architecture_family=architecture_family,
        sensitivity_tau=sensitivity_tau,
    )
    return setting


_CANONICAL_DENSE_TABLE_VALIDATED_IDENTITY_FIELDS = (
    "model_state_hash",
    "route_schema_hash",
    "topology_hash",
    "dense_mask_hash",
    "context_axis_hash",
    "split",
    "forward_tier",
)

# Allocation-level verification of the canonical dense table.
# The campaign runner hashes the ~4.8 GB ``values`` payload ONCE per
# allocation and publishes a stage receipt; a query arm that finds a receipt
# whose content hash verifies, whose table-receipt binding matches the
# registry entry being loaded, and whose live ``values`` stat (size, inode,
# mtime_ns) matches skips the per-query ``sha256_file`` of that payload.  Any
# mismatch, or no receipt, is today's per-query hash, unchanged.
IRG_CANONICAL_DENSE_TABLE_STAGE_RECEIPT_ENV = "IRG_CANONICAL_DENSE_TABLE_STAGE_RECEIPT"
CANONICAL_DENSE_TABLE_STAGE_RECEIPT_SCHEMA = "modern_v0_canonical_dense_table_stage_v1"
CANONICAL_DENSE_TABLE_STAGE_RECEIPT_FIELDS = (
    "schema",
    "table_receipt_path",
    "table_receipt_sha256",
    "values_path",
    "values_sha256",
    "values_size_bytes",
    "values_inode",
    "values_mtime_ns",
)
CANONICAL_DENSE_TABLE_VERIFIED_BY_STAGE_RECEIPT = "allocation_stage_receipt"
CANONICAL_DENSE_TABLE_VERIFIED_BY_PER_QUERY = "per_query_sha256"
# The per-query hash is taken once per node boot and
# bound by a node-local receipt (size, mtime, ctime, inode, device, host, boot
# id); later category processes on that node reuse it.  See verified_once.py.
CANONICAL_DENSE_TABLE_VERIFIED_BY_NODE_RECEIPT = "node_verify_once_receipt"


def canonical_dense_table_stage_receipt_content_sha256(
    receipt: Mapping[str, Any],
) -> str:
    """``content_sha256`` of a stage receipt.

    SHA-256 over ``json.dumps`` of the eight bound fields
    (:data:`CANONICAL_DENSE_TABLE_STAGE_RECEIPT_FIELDS`) with ``sort_keys=True``,
    ``separators=(",", ":")`` and ``ensure_ascii=True``, encoded as UTF-8 --
    the repository's canonical-JSON convention.  A publisher must reproduce
    this exactly; a missing field raises ``KeyError``.
    """

    payload = {
        name: receipt[name] for name in CANONICAL_DENSE_TABLE_STAGE_RECEIPT_FIELDS
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_canonical_dense_table_stage_receipt(
    *,
    table_receipt_path: str | Path,
    values_path: str | Path,
    values_sha256: str,
) -> dict[str, Any]:
    """Publisher-side helper: bind an already computed ``values`` digest.

    The caller supplies the digest it computed (typically once per
    allocation); this function only records the identity coordinates the
    consumer re-checks and seals them with ``content_sha256``.
    """

    resolved_receipt = Path(table_receipt_path)
    resolved_values = Path(values_path)
    stat = os.stat(resolved_values)
    receipt: dict[str, Any] = {
        "schema": CANONICAL_DENSE_TABLE_STAGE_RECEIPT_SCHEMA,
        "table_receipt_path": str(resolved_receipt),
        "table_receipt_sha256": hashlib.sha256(
            resolved_receipt.read_bytes()
        ).hexdigest(),
        "values_path": str(resolved_values),
        "values_sha256": str(values_sha256),
        "values_size_bytes": int(stat.st_size),
        "values_inode": int(stat.st_ino),
        "values_mtime_ns": int(stat.st_mtime_ns),
    }
    receipt["content_sha256"] = canonical_dense_table_stage_receipt_content_sha256(
        receipt
    )
    return receipt


def _same_filesystem_path(declared: str, actual: Path) -> bool:
    try:
        return os.path.realpath(str(declared)) == os.path.realpath(str(actual))
    except (OSError, ValueError):
        return False


def _verify_canonical_dense_table_stage_receipt(
    *,
    stage_receipt_path: Path,
    table_receipt_path: Path,
    table_receipt_bytes: bytes,
    values_path: Path,
) -> tuple[str | None, str | None]:
    """Return ``(values_sha256, None)`` when the receipt vouches for the payload.

    Every refusal is ``(None, reason)`` and sends the caller to the per-query
    hash; nothing here can accept a payload the receipt does not bind.
    """

    try:
        stage = json.loads(stage_receipt_path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, "stage_receipt_unreadable"
    if not isinstance(stage, Mapping):
        return None, "stage_receipt_not_an_object"
    if stage.get("schema") != CANONICAL_DENSE_TABLE_STAGE_RECEIPT_SCHEMA:
        return None, "schema_mismatch"
    missing = [
        name
        for name in (*CANONICAL_DENSE_TABLE_STAGE_RECEIPT_FIELDS, "content_sha256")
        if name not in stage
    ]
    if missing:
        return None, "missing_fields:" + ",".join(missing)
    for name in ("values_size_bytes", "values_inode", "values_mtime_ns"):
        if isinstance(stage[name], bool) or not isinstance(stage[name], int):
            return None, f"field_type:{name}"
    for name in (
        "table_receipt_path",
        "table_receipt_sha256",
        "values_path",
        "values_sha256",
        "content_sha256",
    ):
        if not isinstance(stage[name], str):
            return None, f"field_type:{name}"
    if stage["content_sha256"] != canonical_dense_table_stage_receipt_content_sha256(
        stage
    ):
        return None, "content_sha256_mismatch"
    if not _same_filesystem_path(stage["table_receipt_path"], table_receipt_path):
        return None, "table_receipt_path_mismatch"
    if stage["table_receipt_sha256"] != hashlib.sha256(table_receipt_bytes).hexdigest():
        return None, "table_receipt_sha256_mismatch"
    if not _same_filesystem_path(stage["values_path"], values_path):
        return None, "values_path_mismatch"
    try:
        stat = os.stat(values_path)
    except OSError:
        return None, "values_stat_unavailable"
    if int(stat.st_size) != stage["values_size_bytes"]:
        return None, "values_size_mismatch"
    if int(stat.st_ino) != stage["values_inode"]:
        return None, "values_inode_mismatch"
    if int(stat.st_mtime_ns) != stage["values_mtime_ns"]:
        return None, "values_mtime_ns_mismatch"
    digest = stage["values_sha256"]
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        return None, "values_sha256_malformed"
    return digest, None


@dataclass(frozen=True)
class _CanonicalDenseReadoutTableBinding:
    """A validated canonical dense-readout table, ready to attach."""

    handle: Any
    payload_sha256: Mapping[str, str]
    table_sha256: str
    execution_policy: str
    receipt_path: str
    dense_mask_hash: str = ""
    verified_by: str = CANONICAL_DENSE_TABLE_VERIFIED_BY_PER_QUERY
    stage_receipt_path: str | None = None
    stage_receipt_refusal_reason: str | None = None
    verify_once: Mapping[str, Any] | None = None


def _load_and_validate_canonical_dense_readout_table(
    *,
    receipt_path: str | Path,
    model_state_hash: str,
    route_schema_hash: str,
    topology_hash: str,
    dense_mask_hash: str,
    context_axis_hash: str,
    context_axis_split: str,
    forward_tier: int,
    stage_receipt_path: str | Path | None = None,
) -> _CanonicalDenseReadoutTableBinding:
    """Load + validate the canonical dense-readout table receipt.

    Every declared field is compared against the *actual* runtime identity,
    recomputing the payload sha256 from the files on disk exactly as the
    per-arm dense-table cache does for its own receipt.  Any mismatch is a
    loud, immediate refusal: there is no silent fallback to the per-arm
    path, because the caller explicitly asked for this table via
    ``IRG_CANONICAL_DENSE_TABLE``.

    The ``values`` digest may instead come from an allocation-level stage
    receipt (``stage_receipt_path``, defaulting to the
    ``IRG_CANONICAL_DENSE_TABLE_STAGE_RECEIPT`` environment variable) when
    :func:`_verify_canonical_dense_table_stage_receipt` accepts it; the
    binding then records ``verified_by="allocation_stage_receipt"``.  The
    small ``ids`` sidecar is always hashed per query.
    """

    from experiments.irg_v0_v2_pilots.build_canonical_dense_readout_table import (
        IDS_FILENAME,
        VALUES_FILENAME,
        open_canonical_dense_readout_table,
    )

    resolved_receipt_path = Path(receipt_path)
    if not resolved_receipt_path.exists():
        raise FileNotFoundError(
            f"{IRG_CANONICAL_DENSE_TABLE_ENV} receipt does not exist: "
            f"{resolved_receipt_path}"
        )
    receipt_bytes = resolved_receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes.decode("utf-8"))
    table_dir = resolved_receipt_path.parent
    values_path = table_dir / VALUES_FILENAME
    ids_path = table_dir / IDS_FILENAME
    for path in (values_path, ids_path):
        if not path.exists():
            raise FileNotFoundError(
                "canonical dense-readout table payload is missing: " f"{path}"
            )
    stage_path_value = (
        os.environ.get(IRG_CANONICAL_DENSE_TABLE_STAGE_RECEIPT_ENV)
        if stage_receipt_path is None
        else stage_receipt_path
    )
    stage_path = (
        None
        if stage_path_value is None or not str(stage_path_value).strip()
        else Path(str(stage_path_value).strip())
    )
    values_sha256: str | None = None
    stage_refusal_reason: str | None = None
    if stage_path is not None:
        values_sha256, stage_refusal_reason = (
            _verify_canonical_dense_table_stage_receipt(
                stage_receipt_path=stage_path,
                table_receipt_path=resolved_receipt_path,
                table_receipt_bytes=receipt_bytes,
                values_path=values_path,
            )
        )
    verified_by = (
        CANONICAL_DENSE_TABLE_VERIFIED_BY_STAGE_RECEIPT
        if values_sha256 is not None
        else CANONICAL_DENSE_TABLE_VERIFIED_BY_PER_QUERY
    )
    verify_once_disclosure: dict[str, Any] | None = None
    if values_sha256 is None:
        # The miss path is the unchanged module-level ``sha256_file``.
        values_sha256, verify_once_disclosure = sha256_file_once(
            values_path, hasher=lambda path: sha256_file(path)
        )
        if verify_once_disclosure["verified_by"] == _VERIFY_ONCE_BY_RECEIPT:
            verified_by = CANONICAL_DENSE_TABLE_VERIFIED_BY_NODE_RECEIPT
    actual_payload_sha256 = {
        VALUES_FILENAME: values_sha256,
        IDS_FILENAME: sha256_file(ids_path),
    }
    expected: dict[str, Any] = {
        "model_state_hash": str(model_state_hash),
        "route_schema_hash": str(route_schema_hash),
        "topology_hash": str(topology_hash),
        "dense_mask_hash": str(dense_mask_hash),
        "context_axis_hash": str(context_axis_hash),
        "split": str(context_axis_split),
        "forward_tier": int(forward_tier),
    }
    mismatches: list[str] = []
    if receipt.get("payload_sha256") != actual_payload_sha256:
        mismatches.append("payload_sha256")
    for field in _CANONICAL_DENSE_TABLE_VALIDATED_IDENTITY_FIELDS:
        declared = receipt.get(field)
        if field == "forward_tier":
            declared = None if declared is None else int(declared)
        else:
            declared = None if declared is None else str(declared)
        if declared != expected[field]:
            mismatches.append(field)
    if mismatches:
        raise ValueError(
            f"{IRG_CANONICAL_DENSE_TABLE_ENV} refused: identity mismatch on "
            + ", ".join(sorted(mismatches))
            + f" (receipt={resolved_receipt_path})"
        )
    handle = open_canonical_dense_readout_table(table_dir)
    return _CanonicalDenseReadoutTableBinding(
        handle=handle,
        payload_sha256=dict(actual_payload_sha256),
        table_sha256=content_hash(
            actual_payload_sha256,
            prefix="irg-canonical-dense-readout-table-payload-",
        ),
        execution_policy=str(receipt.get("execution_policy")),
        receipt_path=str(resolved_receipt_path),
        dense_mask_hash=str(receipt.get("dense_mask_hash")),
        verified_by=verified_by,
        stage_receipt_path=None if stage_path is None else str(stage_path),
        stage_receipt_refusal_reason=stage_refusal_reason,
        verify_once=verify_once_disclosure,
    )


_POPULATION_CALIBRATION_SCHEMAS = frozenset(
    {
        "canonical_dense_population_calibration_receipt_v1",
        "canonical_dense_population_calibration_receipt_v2",
    }
)
_POPULATION_CALIBRATION_FUNCTIONAL = "raw_logit"


def _binary_loss_weights(populations: Mapping[str, Any]) -> dict[str, float]:
    """Population shares for per-image loss; preserve the balanced estimand."""
    if loss_weighting() == "balanced":
        if set(populations) == {"on_target", "off_target"}:
            return {"on_target": 0.5, "off_target": 0.5}
        off = tuple(name for name in populations if name != "on_target")
        return {"on_target": 0.5, **{name: 0.5 / len(off) for name in off}}
    sizes = {name: int(rows) if isinstance(rows, (int, np.integer)) else len(rows)
             for name, rows in populations.items()}
    total = sum(sizes.values())
    if total <= 0 or any(size <= 0 for size in sizes.values()):
        raise ValueError("per-image loss requires positive population stratum counts")
    return {name: size / float(total) for name, size in sizes.items()}


def _chance_risk(target_prior: float | None = None) -> float:
    """Unrestricted intercept-only population BCE in natural log units."""
    if loss_weighting() == "balanced":
        return math.log(2.0)
    if target_prior is None or not 0.0 < float(target_prior) < 1.0:
        raise ValueError("per-image chance risk requires a target population prior in (0, 1)")
    prior = float(target_prior)
    return -prior * math.log(prior) - (1.0 - prior) * math.log1p(-prior)


def _population_calibration_receipt_for_penalty(receipt_path, calibration_penalty):
    """Select the population artifact built for this penalty and estimand."""
    from mechanistic_systems.measurement.run_products import loss_weighting, pruning_functional

    path = Path(receipt_path)
    weighting = loss_weighting()
    if pruning_functional() == "substituted_softmax":
        if path.exists():
            artifact = json.loads(path.read_text(encoding="utf-8"))
            if (artifact.get("pruning_functional") == "substituted_softmax"
                    and artifact.get("calibration_penalty") == "firth"
                    and artifact.get("loss_weighting") == "per_image"):
                return path
        name = path.parent.name
        for previous in ("_firthps", "_per_image", "_firthp", "_firthd", "_firth"):
            if name.endswith(previous):
                name = name[:-len(previous)]
                break
        return path.parent.with_name(name + "_firthps") / path.name
    if calibration_penalty not in {"identity_anchor", "firth"}:
        raise ValueError("calibration_penalty must be identity_anchor or firth")
    if weighting == "per_image":
        if path.exists():
            artifact = json.loads(path.read_text(encoding="utf-8"))
            if (artifact.get("calibration_penalty", "identity_anchor") == calibration_penalty
                    and artifact.get("loss_weighting") == "per_image"
                    and artifact.get("pruning_functional", "target_logit") == "target_logit"):
                return path
        suffix = "_firthp" if calibration_penalty == "firth" else "_per_image"
        name = path.parent.name
        for previous in ("_firthps", "_per_image", "_firthp", "_firthd", "_firth"):
            if name.endswith(previous):
                name = name[:-len(previous)]
                break
        return path.parent.with_name(name + suffix) / path.name
    if calibration_penalty == "identity_anchor":
        return path
    if path.exists():
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if (artifact.get("calibration_penalty") == "firth"
                and artifact.get("loss_weighting", "balanced") == "balanced"
                and artifact.get("weighting") in {
                    "half_target_half_equal_represented_off_target_classes",
                    "class_balanced_metric", "class_balanced_binary_accuracy"}):
            return path
    name = path.parent.name
    for previous in ("_firthps", "_per_image", "_firthp", "_firthd", "_firth"):
        if name.endswith(previous):
            name = name[:-len(previous)]
            break
    return path.parent.with_name(name + "_firth") / path.name


def _load_population_dense_calibration(
    receipt_path: str | Path,
    *,
    table_binding: _CanonicalDenseReadoutTableBinding,
    target_class_id: int,
    dense_mask_hash: str,
    confidence_level: float,
    calibration_penalty: str | None = None,
) -> tuple[
    NativeBCECalibrationFitState,
    BaselineRiskCertificate,
    dict[str, Any],
]:
    """Bind one target's full-training-population affine and exact risk.

    The calibration receipt must name the same byte-verified canonical train
    table already attached to the provider.  This is deliberately a strict
    load-only path: no fit, selection, threshold update, or validation access
    is possible here.
    """

    from mechanistic_systems.measurement.run_products import pruning_functional

    penalty = _calibration_penalty() if calibration_penalty is None else calibration_penalty
    resolved_path = _population_calibration_receipt_for_penalty(receipt_path, penalty)
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"{IRG_CANONICAL_DENSE_POPULATION_CALIBRATION_ENV} receipt "
            f"does not exist: {resolved_path}"
        )
    artifact = json.loads(resolved_path.read_text(encoding="utf-8"))
    weighting = loss_weighting()
    functional = pruning_functional()
    if artifact.get("pruning_functional", "target_logit") != functional:
        raise ValueError("population calibration pruning_functional mismatch")
    expected_weighting = ("per_image_population_mean" if weighting == "per_image"
                          else "half_target_half_equal_represented_off_target_classes")
    if artifact.get("loss_weighting", "balanced") != weighting:
        raise ValueError("population calibration loss_weighting mismatch")
    if weighting == "per_image" and artifact.get("weighting") != expected_weighting:
        raise ValueError("per-image dense calibration requires a per-image population receipt")
    compatible_weighting = (artifact.get("weighting") == expected_weighting
                            or weighting == "balanced" and artifact.get("weighting") in {
                                "class_balanced_metric", "class_balanced_binary_accuracy"})
    if penalty == "firth" and (artifact.get("calibration_penalty") != "firth"
                               or not compatible_weighting):
        raise ValueError("Firth dense calibration requires a Firth metric-weighted population receipt")
    if penalty == "identity_anchor" and artifact.get("calibration_penalty", "identity_anchor") != "identity_anchor":
        raise ValueError("identity-anchor run cannot load a Firth population receipt")
    artifact_schema = str(artifact.get("schema", ""))
    if artifact_schema not in _POPULATION_CALIBRATION_SCHEMAS:
        raise ValueError(
            "population calibration receipt schema mismatch: "
            f"{artifact.get('schema')!r}"
        )
    if artifact_schema.endswith("_v2"):
        declared_content_hash = str(artifact.get("content_hash", ""))
        artifact_core = {
            key: value for key, value in artifact.items() if key != "content_hash"
        }
        if not declared_content_hash or declared_content_hash != stable_digest(
            artifact_core
        ):
            raise ValueError("population calibration v2 content hash mismatch")
        if artifact.get("status") != "passed":
            raise ValueError("population calibration v2 receipt is not passed")
        health = dict(artifact.get("health") or {})
        if not bool(health.get("passed")):
            raise ValueError("population calibration v2 build health is not passed")
        if artifact.get("solver_owner") != (
            "mechanistic_systems.backends.factored_calibration."
            "fit_batched_post_query_affine"
        ):
            raise ValueError("population calibration v2 solver owner mismatch")
    declared_table_payload = dict(
        (artifact.get("table_receipt_content") or {}).get("payload_sha256") or {}
    )
    if declared_table_payload != dict(table_binding.payload_sha256):
        raise ValueError(
            "population calibration refused: artifact binds a different "
            "canonical training-table payload than the attached table"
        )
    rows = [
        row
        for row in artifact.get("results", ())
        if row.get("functional") == _POPULATION_CALIBRATION_FUNCTIONAL
        and int(row.get("class_id", -1)) == int(target_class_id)
    ]
    if len(rows) != 1:
        raise ValueError(
            "population calibration refused: expected exactly one "
            f"{_POPULATION_CALIBRATION_FUNCTIONAL} row for class "
            f"{target_class_id}, found {len(rows)}"
        )
    row = rows[0]
    if functional == "substituted_softmax":
        if row.get("fixed_offsets") != "negative_dense_exclusion_lse":
            raise ValueError("substituted dense calibration requires fixed dense exclusion offsets")
        if not row.get("scale_zero_reference_converged"):
            raise ValueError("substituted dense calibration requires a converged scale-zero training reference")
        reference_risk = float(row["scale_zero_reference_risk"])
        if (not math.isfinite(reference_risk) or reference_risk < 0.0 or
                not math.isclose(reference_risk - float(row["bce_dense_population"]),
                                 float(row["span_dense_to_empty"]), rel_tol=1e-12, abs_tol=1e-14)):
            raise ValueError("substituted dense calibration scale-zero risk/span mismatch")
    if not bool(row.get("newton_converged")):
        raise ValueError(
            "population calibration refused: fit did not converge for "
            f"class {target_class_id}"
        )
    if artifact_schema.endswith("_v2") and (
        not bool(row.get("finite"))
        or not math.isfinite(float(row.get("gradient_kkt_residual", math.inf)))
        or float(row.get("gradient_kkt_residual", math.inf))
        > float((artifact.get("solver_options") or {}).get("tolerance", 0.0))
    ):
        raise ValueError(
            "population calibration refused: v2 numerical certificate failed "
            f"for class {target_class_id}"
        )
    row_count = int(artifact["row_count"])
    n_on = int(row["n_on_target"])
    n_off = int(row["n_off_target"])
    if row_count <= 0 or n_on <= 0 or n_off <= 0 or n_on + n_off != row_count:
        raise ValueError(
            "population calibration refused: positive stratum counts do not "
            "partition the sealed training bank"
        )
    artifact_payload_sha256 = dict(artifact.get("payload_sha256") or {})
    if not artifact_payload_sha256:
        raise ValueError("population calibration refused: payload_sha256 is absent")
    state_id = (
        "irg-native-bce-calibration-population-"
        + hashlib.sha256(
            json.dumps(
                {
                    "artifact_payload_sha256": artifact_payload_sha256,
                    "class_id": int(target_class_id),
                    "functional": _POPULATION_CALIBRATION_FUNCTIONAL,
                    "table_sha256": table_binding.table_sha256,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
    )
    fit = NativeBCECalibrationFitState(
        mask_hash=str(dense_mask_hash),
        calibration_state_id=state_id,
        scale=float(row["scale"]),
        offset=float(row["offset"]),
        input_center=0.0,
        parameter_covariance=np.zeros((2, 2), dtype=np.float64),
        context_ids=(),
        maximum_prediction_radius=0.0,
        maximum_bce_excess_risk_bound=0.0,
        numerical_converged=True,
        full_census=True,
        backend_id=(
            "population_exact_batched_newton_kkt_fp64_v2"
            if artifact_schema.endswith("_v2")
            else "population_exact_newton_fp64_v1"
        ),
        fit_receipt_hash=content_hash(
            artifact_payload_sha256,
            prefix="irg-population-calibration-payload-",
        ),
        uncertainty_backend_id="population_exact_zero_uncertainty_v1",
        sampled_stratum_counts={"on_target": n_on, "off_target": n_off},
    )
    bce = float(row["bce_dense_population"])
    span = float(row["span_dense_to_empty"])
    if not math.isfinite(bce) or bce < 0.0:
        raise ValueError("population calibration refused: invalid dense BCE")
    if not math.isfinite(span) or span <= 0.0:
        raise ValueError(
            "population calibration refused: dense-to-empty span is not "
            "strictly positive"
        )
    certificate = BaselineRiskCertificate(
        estimate=bce,
        lower=bce,
        upper=bce,
        context_count=row_count,
        calibration_state_id=state_id,
        sampling_design_id="sealed_train_bank_population_exhaustive_v1",
        data_role="dense_baseline_training_population_exact",
        code_identity=(
            "canonical_dense_population_calibration_batched_v2"
            if artifact_schema.endswith("_v2")
            else "canonical_dense_population_calibration_v1"
        ),
        output_identity=table_binding.table_sha256,
        confidence_level=float(confidence_level),
        confidence_scope=PRECISION_ONLY_IID_SCOPE,
        standard_error=0.0,
        degrees_of_freedom=None,
        look_index=0,
        mask_hash=str(dense_mask_hash),
        calibration_improvement_bound=0.0,
    )
    closure_receipt: dict[str, Any] = {
        "schema": "dense_baseline_training_population_exact_closure_v2",
        "population_exhaustive": True,
        "training_only": True,
        "validation_responses_observed": False,
        "functional": _POPULATION_CALIBRATION_FUNCTIONAL,
        "final_bce_estimate": bce,
        "final_independent_risk_confidence_radius": 0.0,
        "final_maximum_attainable_bce_improvement": 0.0,
        "risk_context_count": row_count,
        "confidence_level": float(confidence_level),
        "risk_certificate_hash": certificate.content_hash,
        "final_calibration_state_id": state_id,
        "artifact_receipt_path": str(resolved_path),
        "artifact_payload_sha256": artifact_payload_sha256,
        "table_sha256": table_binding.table_sha256,
        "span_dense_to_empty_exact": span,
    }
    if penalty == "firth":
        closure_receipt.update({"calibration_penalty": "firth",
                                "weighting": expected_weighting,
                                "effective_sample_size": float(row["effective_sample_size"])})
    if weighting == "per_image":
        closure_receipt.update(loss_weighting="per_image", weighting=expected_weighting)
    if functional == "substituted_softmax":
        closure_receipt.update(pruning_functional=functional,
            fixed_offsets="negative_dense_exclusion_lse",
            scale_zero_reference={
                "scale": 0.0, "offset": float(row["scale_zero_reference_offset"]),
                "per_image_training_bce": float(row["scale_zero_reference_risk"]),
                "converged": bool(row["scale_zero_reference_converged"]),
                "training_context_count": row_count,
                "training_membership_hash": table_binding.table_sha256,
                "pruning_functional": functional, "loss_weighting": "per_image",
                "calibration_penalty": "firth"})
    closure_receipt["content_hash"] = content_hash(
        closure_receipt,
        prefix="dense-baseline-training-population-exact-closure-",
    )
    return fit, certificate, closure_receipt


# ---------------------------------------------------------------------------
# Terminal category-structure matrix — sidecars, receipt block, and the
# shared dense-side cache.  The cache trio below is standalone (explicit
# inputs, no pilot state) so per-step walk-setup consumers can load the dense
# matrices + per-category calibration outside the terminal-validation scope,
# mirroring how the dense readout table cache is loaded at setup.
# ---------------------------------------------------------------------------

TERMINAL_CATEGORY_STRUCTURE_SCHEMA = "terminal_category_structure_v4"
TERMINAL_CATEGORY_SIDECAR_SCHEMA = "terminal_category_structure_sidecar_v3"
_TERMINAL_CATEGORY_CACHE_RECEIPT_SCHEMA = "terminal_category_structure_cache_receipt_v3"
_TERMINAL_CATEGORY_SIDECAR_ARRAYS = (
    "stratum_order",
    "logit_order",
    "counts",
    "moment_mean",
    "moment_m2",
    "ordinary_calibrated_logit_mean",
    "ordinary_probability_matrix",
    "bce_matrix",
    "accuracy_matrix",
    "sensitivity_matrix",
    "sensitivity_defined",
    "specificity_matrix",
    "specificity_defined",
    "calibration_scale",
    "calibration_offset",
    "calibration_input_center",
    "calibration_converged",
    "calibration_selected_strength",
    "calibration_state_id",
    "substituted_calibrated_logit_mean",
    "substituted_probability_matrix",
    "substituted_bce_matrix",
    "substituted_accuracy_matrix",
    "substituted_sensitivity_matrix",
    "substituted_sensitivity_defined",
    "substituted_specificity_matrix",
    "substituted_specificity_defined",
    "substituted_calibration_scale",
    "substituted_calibration_offset",
    "substituted_calibration_input_center",
    "substituted_calibration_converged",
    "substituted_calibration_selected_strength",
    "substituted_calibration_state_id",
)


TERMINAL_CATEGORY_SIDECAR_ENCODING = "npz_zip_stored_pinned_timestamp_v1"


def _npz_stored_bytes(members: Mapping[str, np.ndarray]) -> bytes:
    """Deterministic ``.npz`` bytes: ZIP_STORED, pinned timestamp, sorted members.

    ``np.savez_compressed`` stamps each member with the wall-clock time (second
    granularity) and its deflate output is library-version dependent, so two
    identical captures could yield different sidecar bytes.  Routing the same
    arrays through :func:`write_npz_streaming` makes the bytes a pure function
    of the arrays.  ``np.load`` reads stored and deflated archives alike, so
    loaders stay compatible with previously cached compressed payloads.
    """

    from mechanistic_systems.measurement.step_matrix_store import (
        write_npz_streaming,
    )

    with tempfile.TemporaryDirectory(prefix="terminal-category-sidecar-") as scratch:
        archive_path = Path(scratch) / "sidecar.npz"
        write_npz_streaming(str(archive_path), arrays=members, member_files={})
        return archive_path.read_bytes()


def _terminal_category_degradation(
    error: BaseException,
    *,
    stage: str,
) -> dict[str, Any]:
    """Receipt block for a per-category fit/capture that could not complete.

    ``unconverged_class_ids`` is the union over functionals of the states the
    bank reported unready (unavailable or unconverged); it is empty when the
    error carried no per-class detail, in which case ``error_message`` is the
    only diagnosis.
    """

    by_functional = getattr(error, "unready_class_ids_by_functional", None) or {}
    class_ids = tuple(
        sorted({int(class_id) for ids in by_functional.values() for class_id in ids})
    )
    return {
        "status": "degraded_unconverged_categories",
        "stage": str(stage),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "unconverged_count": int(len(class_ids)),
        "unconverged_class_ids": list(class_ids),
        "unconverged_class_ids_by_functional": {
            str(functional): [int(class_id) for class_id in ids]
            for functional, ids in by_functional.items()
        },
    }


def _terminal_category_sidecar_bytes(
    capture: FrozenCategoryStructureCapture,
    *,
    fit_bundle: FrozenCategoryCalibrationFitBundle,
    calibration_application: str = "mask_own",
    dense_token_hash: str | None = None,
) -> bytes:
    """Serialize only a typed, train-frozen category capture.

    Mapping-shaped v1/v2 captures were fit on terminal rows and are therefore
    provenance-only.  Refusing them here prevents a legacy payload from being
    relabelled as a v3 train-frozen validation artifact.  The archive is the
    deterministic stored form of :func:`_npz_stored_bytes`.
    """

    if not isinstance(capture, FrozenCategoryStructureCapture):
        raise TypeError("category sidecar v3 requires a typed frozen capture")
    if not isinstance(fit_bundle, FrozenCategoryCalibrationFitBundle):
        raise TypeError("typed category capture requires its frozen fit bundle")
    if (
        capture.fit_bundle_hash != fit_bundle.content_hash
        or capture.bank_hash != fit_bundle.bank.content_hash
    ):
        raise ValueError("typed category capture differs from its frozen bank")
    if calibration_application not in {"mask_own", "dense_frozen_final"}:
        raise ValueError("unknown category calibration application")
    if calibration_application == "dense_frozen_final" and (
        fit_bundle.mask_hash != capture.dense_reference_mask_hash
    ):
        raise ValueError("dense-frozen final capture requires the dense bank")
    ordinary_states = tuple(
        state
        for state in fit_bundle.bank.states
        if state.functional.value == "ordinary_ovr"
    )
    substituted_states = tuple(
        state
        for state in fit_bundle.bank.states
        if state.functional.value == "substituted_target_affine_dense_exclusion"
    )
    ordinary_scale, ordinary_offset, ordinary_center = fit_bundle.bank.affine_arrays(
        "ordinary_ovr"
    )
    # A tolerated FINAL bank reads its substituted affines
    # as NaN at exactly its carried unready classes; every other bank (and the
    # ordinary functional always) reads strictly, exactly as before.
    (
        substituted_scale,
        substituted_offset,
        substituted_center,
        substituted_unready,
    ) = fit_bundle.bank.evaluation_affine_arrays(
        "substituted_target_affine_dense_exclusion"
    )
    if tuple(capture.ordinary.unready_class_ids) or tuple(
        capture.substituted.unready_class_ids
    ) != tuple(substituted_unready):
        raise ValueError(
            "typed category capture carries unready classes its frozen bank does not"
        )
    ordinary_selected = tuple(
        str(value) for value in fit_bundle.ordinary_selected_strength
    )
    substituted_selected = tuple(
        str(value) for value in fit_bundle.substituted_selected_strength
    )
    arrays = {
        "stratum_order": capture.class_ids,
        "logit_order": capture.class_ids,
        "counts": capture.counts,
        # Raw-logit moments retain the published sidecar column names.
        "moment_mean": capture.raw_logit_mean,
        "moment_m2": capture.raw_logit_m2,
        "ordinary_calibrated_logit_mean": capture.ordinary.mean_calibrated_logit,
        "ordinary_probability_matrix": capture.ordinary.mean_probability,
        "bce_matrix": capture.ordinary.mean_bce,
        "accuracy_matrix": capture.ordinary.binary_accuracy,
        "sensitivity_matrix": capture.ordinary.sensitivity_component,
        "sensitivity_defined": capture.ordinary.sensitivity_defined,
        "specificity_matrix": capture.ordinary.specificity_component,
        "specificity_defined": capture.ordinary.specificity_defined,
        "calibration_scale": ordinary_scale,
        "calibration_offset": ordinary_offset,
        "calibration_input_center": ordinary_center,
        "calibration_converged": np.asarray(
            [state.converged for state in ordinary_states], dtype=bool
        ),
        "calibration_selected_strength": np.asarray(ordinary_selected),
        "calibration_state_id": np.asarray(
            [state.state_id for state in ordinary_states]
        ),
        "substituted_calibrated_logit_mean": capture.substituted.mean_calibrated_logit,
        "substituted_probability_matrix": capture.substituted.mean_probability,
        "substituted_bce_matrix": capture.substituted.mean_bce,
        "substituted_accuracy_matrix": capture.substituted.binary_accuracy,
        "substituted_sensitivity_matrix": (capture.substituted.sensitivity_component),
        "substituted_sensitivity_defined": (capture.substituted.sensitivity_defined),
        "substituted_specificity_matrix": (capture.substituted.specificity_component),
        "substituted_specificity_defined": (capture.substituted.specificity_defined),
        "substituted_calibration_scale": substituted_scale,
        "substituted_calibration_offset": substituted_offset,
        "substituted_calibration_input_center": substituted_center,
        "substituted_calibration_converged": np.asarray(
            [state.converged for state in substituted_states], dtype=bool
        ),
        "substituted_calibration_selected_strength": np.asarray(substituted_selected),
        "substituted_calibration_state_id": np.asarray(
            [state.state_id for state in substituted_states]
        ),
    }
    members: dict[str, np.ndarray] = {
        "schema": np.asarray(TERMINAL_CATEGORY_SIDECAR_SCHEMA),
        "capture_schema": np.asarray(str(capture.schema)),
        "mask_hash": np.asarray(str(capture.mask_hash)),
        "fit_bundle_hash": np.asarray(str(fit_bundle.content_hash)),
        "frozen_bank_hash": np.asarray(str(fit_bundle.bank.content_hash)),
        "validation_open_token_hash": np.asarray(
            str(capture.validation_open_token_hash if dense_token_hash is None else dense_token_hash)
        ),
        "substitution_reference_hash": np.asarray(
            str(capture.dense_reference_validation_identity)
        ),
        "census_membership_hash": np.asarray(
            str(capture.validation_partition.membership_identity)
        ),
        "calibration_context_membership_hash": np.asarray(
            str(fit_bundle.bank.identity.training_membership_identity)
        ),
        "calibration_sample_role": np.asarray("train_frozen_category_bank"),
        "bce_matrix_semantics": np.asarray(
            "ordinary_ovr_train_frozen_mean_bernoulli_bce_v1"
        ),
        "substituted_functional": np.asarray(
            "target_affine_then_dense_competitor_exclusion_v1"
        ),
        "substituted_bce_matrix_semantics": np.asarray(
            "substituted_target_affine_dense_exclusion_train_frozen_mean_"
            "bernoulli_bce_v1"
        ),
        "hybrid_top1_status": np.asarray(
            "not_claimed_independent_ovr_affines_are_not_joint_multiclass"
        ),
        "layout_content_hash": np.asarray(str(fit_bundle.layout_content_hash)),
        "layout_block_assignment_hash": np.asarray(
            str(fit_bundle.layout_block_assignment_hash)
        ),
        **{name: np.asarray(value) for name, value in arrays.items()},
    }
    if substituted_unready:
        # Present only on a partial capture, so every default-path sidecar is
        # byte-identical: the ids of the NaN substituted classifier columns.
        members["substituted_unready_class_ids"] = np.asarray(
            substituted_unready, dtype=np.int64
        )
        members["substituted_capture_status"] = np.asarray(
            "captured_substituted_partially_unconverged"
        )
    if calibration_application == "dense_frozen_final":
        members.update(_dense_frozen_sidecar_labels())
    return _npz_stored_bytes(members)


def _dense_frozen_sidecar_labels() -> dict[str, np.ndarray]:
    """Explain the calibration and shared raw-logit matrix of this capture."""

    return {
        "calibration_application": np.asarray(
            "dense_train_frozen_calibration_applied_to_final_circuit"
        ),
        "calibration_sample_role": np.asarray(
            "dense_train_frozen_calibration_applied_to_final_circuit"
        ),
        "bce_matrix_semantics": np.asarray(
            "ordinary_ovr_dense_train_frozen_calibration_applied_to_"
            "final_circuit_mean_bernoulli_bce_v1"
        ),
        "substituted_bce_matrix_semantics": np.asarray(
            "substituted_dense_train_frozen_calibration_applied_to_"
            "final_circuit_mean_bernoulli_bce_v1"
        ),
        "raw_logit_semantics": np.asarray(
            "moment_mean_is_final_mask_pre_affine_input_for_both_functionals"
        ),
    }


def _dense_frozen_sidecar_from_dense_bytes(dense_bytes: bytes) -> bytes:
    """The final and dense masks coincide; preserve the dense matrices exactly."""

    with np.load(io.BytesIO(dense_bytes), allow_pickle=False) as loaded:
        members = {name: np.asarray(loaded[name]) for name in loaded.files}
    members.update(_dense_frozen_sidecar_labels())
    return _npz_stored_bytes(members)


# The run's train-frozen dense and
# final category banks, saved beside the category matrices.  Evaluation needs
# nothing else from training, so a later process can reload the exact banks
# (their hashes include the solver's wall time, so a refit never matches) and
# recompute the run's terminal files bit for bit.
TERMINAL_FROZEN_BANKS_FILENAME = "terminal_category_frozen_banks.npz"
TERMINAL_FROZEN_BANKS_SCHEMA = "terminal_category_frozen_banks_v1"
# Members of the dense-cache layout that the product stores once for both
# roles (the context ids) or rebuilds exactly from the class counts (labels).
_FROZEN_BANKS_AXIS_MEMBERS = (
    "training_context_ids",
    "training_labels",
    "ordinary_labels",
    "substituted_labels",
)


def _encode_context_ids(values: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """UTF-8 bytes plus int64 offsets: about a quarter of numpy's UCS-4 form."""

    encoded = [str(value).encode("utf-8") for value in values]
    offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(item) for item in encoded], dtype=np.int64)
    return offsets, np.frombuffer(b"".join(encoded), dtype=np.uint8).copy()


def _decode_context_ids(offsets: np.ndarray, blob: np.ndarray) -> tuple[str, ...]:
    raw = np.asarray(blob, dtype=np.uint8).tobytes()
    bounds = np.asarray(offsets, dtype=np.int64).tolist()
    if not bounds or bounds[0] != 0 or bounds[-1] != len(raw):
        raise ValueError("saved training context ids are malformed")
    return tuple(
        raw[start:stop].decode("utf-8") for start, stop in zip(bounds, bounds[1:])
    )


def _terminal_frozen_banks_bytes(frozen_banks: Any) -> tuple[bytes, tuple[str, ...]]:
    """Serialize the run's banks; a bound reference stores the dense bank once.

    Each bundle is stored in the dense cache's member layout under a
    ``dense__`` or ``final__`` prefix, with the JSON fields that rebuild it
    exactly.  The training context ids are stored once when both roles share
    them (they always do: one training census fits both banks), and the
    label vectors are omitted because every bundle's labels equal its class
    ids repeated by its class counts.  The final bundle is absent when its
    fit degraded or when the final mask is the dense mask.
    """

    roles: list[tuple[str, Any]] = (
        [] if frozen_banks.dense_reference is not None
        else [("dense", frozen_banks.dense_fit)]
    )
    if frozen_banks.final_fit is not None and not frozen_banks.final_equals_dense:
        roles.append(("final", frozen_banks.final_fit))
    shared_ids = tuple((roles[0][1] if roles else frozen_banks.dense_fit).training_context_ids)
    offsets, blob = _encode_context_ids(shared_ids)
    members: dict[str, np.ndarray] = {
        "schema": np.asarray(TERMINAL_FROZEN_BANKS_SCHEMA),
        "roles": np.asarray([role for role, _bundle in roles]),
        "validation_open_token_hash": np.asarray(
            str(frozen_banks.validation_open_token.content_hash)
        ),
        "calibration_context_axis_hash": np.asarray(
            str(frozen_banks.calibration_context_axis_hash)
        ),
        "final_equals_dense": np.asarray(bool(frozen_banks.final_equals_dense)),
        "training_context_id_offsets": offsets,
        "training_context_id_utf8": blob,
    }
    for role, bundle in roles:
        prefix = f"{role}__"
        members[f"{prefix}metadata_json"] = np.asarray(
            json.dumps(
                _fit_bundle_payload_metadata(bundle),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        if tuple(bundle.training_context_ids) != shared_ids:
            (
                members[f"{prefix}training_context_id_offsets"],
                members[f"{prefix}training_context_id_utf8"],
            ) = _encode_context_ids(bundle.training_context_ids)
        for name, value in _fit_bundle_payload_arrays(bundle).items():
            if name not in _FROZEN_BANKS_AXIS_MEMBERS:
                members[f"{prefix}{name}"] = np.asarray(value)
    return _npz_stored_bytes(members), tuple(role for role, _bundle in roles)


def _load_terminal_frozen_banks(payload: bytes) -> dict[str, Any]:
    """Rebuild the saved bundles; each must match its recorded content hash."""

    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        if str(data["schema"]) != TERMINAL_FROZEN_BANKS_SCHEMA:
            raise ValueError("saved frozen category banks have an unknown schema")
        roles = tuple(str(value) for value in data["roles"])
        if roles not in ((), ("dense",), ("dense", "final"), ("final",)):
            raise ValueError("saved frozen category banks have invalid roles")
        loaded: dict[str, Any] = {
            "roles": roles,
            "validation_open_token_hash": str(data["validation_open_token_hash"]),
            "calibration_context_axis_hash": str(
                data["calibration_context_axis_hash"]
            ),
            "final_equals_dense": bool(data["final_equals_dense"]),
        }
        shared_ids = _decode_context_ids(
            data["training_context_id_offsets"], data["training_context_id_utf8"]
        )
        for role in roles:
            prefix = f"{role}__"
            metadata = json.loads(str(data[f"{prefix}metadata_json"]))
            members: dict[str, Any] = {
                name[len(prefix):]: np.asarray(data[name])
                for name in data.files
                if name.startswith(prefix)
            }
            ids = (
                _decode_context_ids(
                    members.pop("training_context_id_offsets"),
                    members.pop("training_context_id_utf8"),
                )
                if "training_context_id_offsets" in members
                else shared_ids
            )
            labels = np.repeat(
                np.asarray(members["ordinary_class_ids"], dtype=np.int64),
                np.asarray(members["training_counts"], dtype=np.int64),
            )
            members["training_context_ids"] = ids
            for name in _FROZEN_BANKS_AXIS_MEMBERS[1:]:
                members[name] = labels
            bundle = _fit_bundle_from_payload(
                members,
                bundle_meta=metadata["bundle"],
                identity_meta=metadata["identity"],
            )
            if bundle.content_hash != metadata["bundle_hash"]:
                raise ValueError(
                    f"saved {role} category bundle differs from its recorded hash"
                )
            loaded[role] = bundle
    return loaded


def _with_terminal_frozen_banks(
    block: dict[str, Any],
    sidecars: dict[str, bytes],
    frozen_banks: Any,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Add the frozen-bank run product to an enabled terminal leg's output.

    The leg's own category files and fields are unchanged; this appends one
    more product and its receipt entry (``sidecars.frozen_banks``). A bound
    reference supplies the dense bank. A failed
    save is recorded and costs only the later recompute, never the run.
    """

    import hashlib

    if not block.get("enabled") or frozen_banks is None:
        return block, sidecars
    reference_entry = block.get("dense_reference")
    reference = (
        frozen_banks.dense_reference
        if isinstance(reference_entry, Mapping) and reference_entry.get("status") == "bound"
        else None
    )
    # Setup may have bound a reference that diagnostic terminal evaluation
    # subsequently refused. Save the dense bank that evaluation actually used.
    banks_to_save = (
        frozen_banks if reference is not None or frozen_banks.dense_reference is None
        else replace(frozen_banks, dense_reference=None)
    )
    started = time.perf_counter()
    try:
        payload, roles = _terminal_frozen_banks_bytes(banks_to_save)
    except (OSError, ValueError, TypeError) as error:
        block = dict(block)
        block["sidecars"] = {
            **dict(block.get("sidecars") or {}),
            "frozen_banks": {
                "filename": None,
                "status": "not_saved",
                "reason": f"{type(error).__name__}: {error}",
            },
        }
        block["frozen_banks_sidecar"] = None
        return block, sidecars
    elapsed = time.perf_counter() - started
    block = dict(block)
    block["sidecars"] = {
        **dict(block.get("sidecars") or {}),
        "frozen_banks": {
            "filename": TERMINAL_FROZEN_BANKS_FILENAME,
            "status": "saved",
            "schema": TERMINAL_FROZEN_BANKS_SCHEMA,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": int(len(payload)),
            "roles": list(roles),
            **({} if reference is None else {
                "dense_reference_manifest_sha256": reference["manifest_sha256"],
                "dense_reference_bank_sha256": reference["manifest"]["products"][
                    TERMINAL_DENSE_REFERENCE_BANK
                ]["sha256"],
            }),
            "dense_fit_bundle_hash": frozen_banks.dense_fit.content_hash,
            "final_fit_bundle_hash": (
                frozen_banks.final_fit.content_hash if "final" in roles else None
            ),
            "elapsed_seconds": float(elapsed),
            "use": "evaluation_only_reload_for_terminal_recompute",
        },
    }
    block["frozen_banks_sidecar"] = TERMINAL_FROZEN_BANKS_FILENAME
    return block, {**sidecars, TERMINAL_FROZEN_BANKS_FILENAME: payload}


TERMINAL_DENSE_REFERENCE_SCHEMA = "terminal_dense_reference_v1"
TERMINAL_DENSE_SIDECAR_TOKEN_SEMANTICS = "dense_bundle_only_v1"
TERMINAL_DENSE_REFERENCE_MANIFEST = "manifest.json"
TERMINAL_DENSE_REFERENCE_BANK = "dense_frozen_bank.npz"
_TERMINAL_DENSE_CODE_MODULES = (
    "mechanistic_systems.circuits.irg.registered_resnet18_pilot",
    "mechanistic_systems.measurement.native_irg_providers",
    "mechanistic_systems.measurement.frozen_category_calibration",
    "mechanistic_systems.measurement.step_matrix_store",
    "mechanistic_systems.backends.calibration_backend",
    "mechanistic_systems.backends.factored_calibration",
)


def _terminal_dense_code_identity() -> str:
    """Content identity of the files that fit and serialize dense products."""

    import importlib

    files = {}
    for name in _TERMINAL_DENSE_CODE_MODULES:
        module = importlib.import_module(name)
        files[name] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    return content_hash(files, prefix="irg-terminal-dense-code-")


def _dense_category_token_hash(bundle: FrozenCategoryCalibrationFitBundle) -> str:
    """The token a dense-only terminal evaluation would issue for this fit."""

    return FrozenCategoryValidationOpenToken(
        provider_identity=bundle.provider_identity,
        model_state_hash=bundle.model_state_hash,
        frozen_bundle_hashes=(bundle.content_hash,),
    ).content_hash


def _terminal_dense_census_hash(populations: Mapping[int, Sequence[str]]) -> str:
    return content_hash(
        {
            "schema": "irg_category_validation_membership_v1",
            "populations": {
                str(class_id): tuple(sorted(rows))
                for class_id, rows in populations.items()
            },
        },
        prefix="irg-category-validation-membership-",
    )


def _terminal_dense_fit_semantic_hash(
    provider: Any,
    dense_mask: HardMask,
    dense_fit: FrozenCategoryCalibrationFitBundle,
    calibration_axis_hash: str,
    fit_chunk: int,
) -> str | None:
    method = getattr(provider, "frozen_category_fit_request", None)
    if not callable(method):
        return None
    populations: dict[int, list[str]] = {}
    for context_id, label in zip(
        dense_fit.training_context_ids, dense_fit.training_labels.tolist(), strict=True
    ):
        populations.setdefault(int(label), []).append(str(context_id))
    request = method(
        dense_mask,
        dense_mask,
        training_context_ids_by_class=populations,
        training_axis_source_identity=str(calibration_axis_hash),
        mode=dense_fit.bank.identity.mode,
        category_fit_chunk=int(fit_chunk),
    )
    return str(request["fit_semantic_request_hash"])


def _terminal_dense_numerics_profile(provider: Any) -> dict[str, Any]:
    """Settings affecting dense readouts, independent of campaign scheduling."""
    profile = getattr(provider, "runtime_execution_profile", None)
    if profile is None:
        profile = getattr(getattr(provider, "evaluator", None), "runtime_execution_profile", None)
    return {
        "execution_numerics_cache_version": DETERMINISTIC_EXECUTION_CACHE_VERSION,
        **{
            name: getattr(profile, name, None)
            for name in (
                "allow_tf32", "calibration_batch_rows", "terminal_batch_rows",
                "vmap_compile", "k1_compile", "compile_dynamic_batch", "cuda_graphs",
                "execution_batch_rows", "query_tile_size",
            )
        },
    }


def _terminal_dense_reference_key(
    *,
    provider: Any,
    dense_mask: HardMask,
    dense_fit: FrozenCategoryCalibrationFitBundle,
    model_state_hash: str,
    validation_axis_hash: str,
    census_membership_hash: str,
    calibration_axis_hash: str,
    category_fit_chunk: int,
    forward_tier: int,
) -> dict[str, Any]:
    import torch


    evaluation_bank_binding_hash = content_hash(
        {
            "axis_hash": str(validation_axis_hash),
            "census_membership_hash": str(census_membership_hash),
        },
        prefix="irg-terminal-evaluation-bank-",
    )
    return {
        "schema": "terminal_dense_reference_binding_v1",
        "dense_checkpoint_hash": str(model_state_hash),
        "executable_model_state_hash": str(dense_fit.model_state_hash),
        "evaluation_bank_binding_hash": evaluation_bank_binding_hash,
        "terminal_code_identity": _terminal_dense_code_identity(),
        "terminal_settings": {
            "dense_mask_hash": dense_mask.mask_hash,
            "route_axis_hash": dense_mask.route_axis_hash,
            "calibration_axis_hash": str(calibration_axis_hash),
            "calibration_membership_hash": (
                dense_fit.bank.identity.training_membership_identity
            ),
            "calibration_mode": dense_fit.bank.identity.mode.value,
            "calibration_contract": str(
                provider.calibration_options["optimization_contract"]
            ),
            "fit_options": _dense_fit_cache_optin_fields(provider),
            "category_fit_chunk": _terminal_category_fit_chunk(category_fit_chunk),
            "fit_semantic_request_hash": _terminal_dense_fit_semantic_hash(
                provider, dense_mask, dense_fit, calibration_axis_hash,
                _terminal_category_fit_chunk(category_fit_chunk),
            ),
            "forward_tier": int(forward_tier),
            "full_readout_projection_hash": (
                provider.terminal_full_readout_reference_hash
            ),
            "readout_numerics": _terminal_frozen_category_readout_numerics(provider),
            "validation_one_pass": _terminal_validation_one_pass_enabled(),
            "runtime_numerics_profile": _terminal_dense_numerics_profile(provider),
            "torch_version": str(torch.__version__),
            "cuda_version": str(getattr(torch.version, "cuda", None)),
        },
    }


def _load_terminal_dense_reference(
    directory: str | Path, expected_sha256: str
) -> dict[str, Any]:
    """Load all reference bytes and check the supplied manifest digest."""

    root = Path(directory)
    if len(expected_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in expected_sha256
    ):
        raise ValueError("terminal dense reference requires a manifest sha256")
    manifest_bytes = (root / TERMINAL_DENSE_REFERENCE_MANIFEST).read_bytes()
    actual = hashlib.sha256(manifest_bytes).hexdigest()
    if actual != expected_sha256:
        raise ValueError("terminal dense reference manifest hash differs")
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != TERMINAL_DENSE_REFERENCE_SCHEMA:
        raise ValueError("terminal dense reference manifest schema differs")
    if (
        manifest.get("dense_sidecar_validation_token_semantics")
        != TERMINAL_DENSE_SIDECAR_TOKEN_SEMANTICS
    ):
        raise ValueError("terminal dense reference dense-file token format differs")
    products = {}
    for name in ("category_matrix_dense.npz", TERMINAL_DENSE_REFERENCE_BANK):
        entry = manifest.get("products", {}).get(name)
        if not isinstance(entry, Mapping) or entry.get("filename") != name:
            raise ValueError(f"terminal dense reference missing {name}")
        payload = (root / name).read_bytes()
        if len(payload) != entry.get("bytes") or hashlib.sha256(payload).hexdigest() != entry.get("sha256"):
            raise ValueError(f"terminal dense reference altered {name}")
        products[name] = payload
    loaded = _load_terminal_frozen_banks(products[TERMINAL_DENSE_REFERENCE_BANK])
    if loaded["roles"] != ("dense",) or loaded["dense"].content_hash != manifest.get("dense_fit_bundle_hash"):
        raise ValueError("terminal dense reference bank differs from its manifest")
    if manifest.get("binding_key", {}).get("terminal_code_identity") != _terminal_dense_code_identity():
        raise ValueError("terminal dense reference code identity differs")
    if not isinstance(manifest.get("dense_fit_gate_summary"), (dict, type(None))):
        raise ValueError("terminal dense reference gate summary differs")
    return {
        "manifest": manifest,
        "manifest_sha256": actual,
        "dense_bytes": products["category_matrix_dense.npz"],
        "dense_fit": loaded["dense"],
    }


def build_terminal_dense_reference(
    directory: str | Path,
    *,
    provider: Any,
    dense_mask: HardMask,
    calibration_populations: Mapping[str, Sequence[str]],
    validation_populations: Mapping[str, Sequence[str]],
    target_class_id: int,
    calibration_batch_factory: Any,
    validation_batch_factory: Any,
    model_state_hash: str,
    calibration_context_axis_hash: str,
    validation_context_axis_hash: str,
    forward_tier: int,
    category_fit_chunk: int | None = None,
    calibration_bank_mode: CalibrationBankMode = CalibrationBankMode.CLAIM_BEARING_1000,
) -> dict[str, Any]:
    """Fit and evaluate the campaign's dense terminal products once."""

    root = Path(directory)
    if root.exists():
        raise FileExistsError("terminal dense reference directory already exists")
    representative_bits = np.ones(dense_mask.route_count, dtype=bool)
    representative_bits[-1] = False
    representative_mask = HardMask.from_bool_array(
        representative_bits, route_axis_hash=dense_mask.route_axis_hash
    )
    # Finish the representative mask's training calibration before opening
    # category validation below.
    provider.ensure_calibration(representative_mask)
    frozen = _freeze_terminal_category_banks(
        enabled=True,
        provider=provider,
        dense_mask=dense_mask,
        final_mask=dense_mask,
        calibration_populations=calibration_populations,
        target_class_id=target_class_id,
        calibration_batch_factory=calibration_batch_factory,
        calibration_context_axis_hash=calibration_context_axis_hash,
        category_fit_chunk=category_fit_chunk,
        calibration_bank_mode=calibration_bank_mode,
    )
    # A registered arm primes validation before category capture.  Use the
    # same provider path here, with a distinct representative final mask so
    # the dense lane is evaluated in the two-lane forward.
    prime = _prime_terminal_validation_one_pass(
        terminal_provider=provider,
        final_mask=representative_mask,
        dense_mask=dense_mask,
        validation_populations=validation_populations,
        target_class_id=target_class_id,
        batch_factory=validation_batch_factory,
        validation_axis_hash=validation_context_axis_hash,
    )
    try:
        if (
            prime is not None
            and prime["prime"].get("forward_lanes") != 2
            and prime["prime"].get("dense_source") != "dense_validation_table"
        ):
            raise ValueError("dense reference requires a two-lane validation prime")
        block, sidecars = _terminal_category_structure_leg(
            enabled=True,
            provider=provider,
            dense_mask=dense_mask,
            final_mask=dense_mask,
            validation_populations=validation_populations,
            target_class_id=target_class_id,
            batch_factory=validation_batch_factory,
            model_state_hash=model_state_hash,
            context_axis_hash=validation_context_axis_hash,
            forward_tier=forward_tier,
            cache_dir=None,
            category_fit_chunk=category_fit_chunk,
            frozen_banks=frozen,
        )
    finally:
        _release_terminal_validation_one_pass(terminal_provider=provider, one_pass=prime)
    dense_bytes = sidecars["category_matrix_dense.npz"]
    bank_bytes, roles = _terminal_frozen_banks_bytes(frozen)
    if roles != ("dense",):
        raise RuntimeError("dense reference unexpectedly contains a final bank")
    populations = _terminal_category_populations_by_class(
        validation_populations, target_class_id=target_class_id
    )
    key = _terminal_dense_reference_key(
        provider=provider,
        dense_mask=dense_mask,
        dense_fit=frozen.dense_fit,
        model_state_hash=model_state_hash,
        validation_axis_hash=validation_context_axis_hash,
        census_membership_hash=_terminal_dense_census_hash(populations),
        calibration_axis_hash=calibration_context_axis_hash,
        category_fit_chunk=_terminal_category_fit_chunk(category_fit_chunk),
        forward_tier=forward_tier,
    )
    products = {
        name: {"filename": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        for name, data in (
            ("category_matrix_dense.npz", dense_bytes),
            (TERMINAL_DENSE_REFERENCE_BANK, bank_bytes),
        )
    }
    manifest = {
        "schema": TERMINAL_DENSE_REFERENCE_SCHEMA,
        "binding_key": key,
        "dense_fit_bundle_hash": frozen.dense_fit.content_hash,
        "dense_capture_schema": block["dense_capture_schema"],
        "dense_sidecar_validation_token_semantics": TERMINAL_DENSE_SIDECAR_TOKEN_SEMANTICS,
        "dense_substitution_reference_hash": block["dense_substitution_reference_hash"],
        "dense_fit_gate_summary": _terminal_bank_gate_summary(provider, frozen.dense_fit),
        "validation_prime": None if prime is None else prime["prime"],
        "products": products,
    }
    manifest["effective_settings"] = effective_settings()
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    root.mkdir(parents=True)
    for name, data in (("category_matrix_dense.npz", dense_bytes), (TERMINAL_DENSE_REFERENCE_BANK, bank_bytes), (TERMINAL_DENSE_REFERENCE_MANIFEST, manifest_bytes)):
        (root / name).write_bytes(data)
    return {"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "binding_key": key, "manifest": manifest}


def _terminal_category_structure_cache_key(
    *,
    model_state_hash: str,
    dense_mask_hash: str,
    context_axis_hash: str,
    census_membership_hash: str,
    full_readout_projection_hash: str,
    calibration_contract: str,
    category_fit_chunk: int,
    forward_tier: int,
    calibration_context_membership_hash: str | None = None,
    frozen_bank_hash: str | None = None,
    final_frozen_bank_hash: str | None = None,
    validation_open_token_hash: str | None = None,
    frozen_category_readout_numerics: str | None = None,
) -> str:
    """Target-independent cache identity for the DENSE mask's matrices.

    The dense category structure depends on the model state, the dense mask,
    the evaluation axis and its class-partitioned census membership, the
    train calibration membership, the full-readout retention identity, and the
    registered calibration contract — never on ``target_class_id``, so every
    query of a family on the same axis shares one payload.
    """

    import torch

    return content_hash(
        {
            "schema": "terminal_category_structure_cache_key_v63",
            **({"loss_weighting": "per_image"} if loss_weighting() == "per_image" else {}),
            "execution_numerics_cache_version": DETERMINISTIC_EXECUTION_CACHE_VERSION,
            "frozen_category_readout_numerics": str(frozen_category_readout_numerics or "runtime"),
            "model_state_hash": str(model_state_hash),
            "dense_mask_hash": str(dense_mask_hash),
            "context_axis_hash": str(context_axis_hash),
            "census_membership_hash": str(census_membership_hash),
            "calibration_context_membership_hash": (
                ""
                if calibration_context_membership_hash is None
                else str(calibration_context_membership_hash)
            ),
            "frozen_bank_hash": (
                "" if frozen_bank_hash is None else str(frozen_bank_hash)
            ),
            "final_frozen_bank_hash": (
                "" if final_frozen_bank_hash is None else str(final_frozen_bank_hash)
            ),
            "validation_open_token_hash": (
                ""
                if validation_open_token_hash is None
                else str(validation_open_token_hash)
            ),
            "full_readout_projection_hash": str(full_readout_projection_hash),
            "calibration_contract": str(calibration_contract),
            "category_fit_chunk": _terminal_category_fit_chunk(category_fit_chunk),
            "forward_tier": int(forward_tier),
            "torch_version": str(torch.__version__),
            "cuda_version": str(getattr(torch.version, "cuda", None)),
        },
        prefix="irg-terminal-category-structure-",
    )


@dataclass(frozen=True)
class _TerminalCategoryStructureCacheLoad:
    valid: bool
    reason: str | None
    payload_sha256: str | None
    payload_bytes: bytes
    arrays: Mapping[str, np.ndarray]
    capture_schema: str
    substitution_reference_hash: str


def _load_terminal_category_structure_cache(
    *,
    payload_path: Path,
    receipt_path: Path,
    cache_key: str,
    dense_mask_hash: str,
    census_membership_hash: str,
    frozen_bank_hash: str,
    validation_open_token_hash: str,
) -> _TerminalCategoryStructureCacheLoad | None:
    """Standalone sha256-verified load of the shared dense-side matrices.

    Callable at walk-setup scope: only (cache paths, key inputs) are needed,
    never a terminal-validation context.  Invalid artifacts come back as a
    refusal with a recorded reason, mirroring the dense readout table cache.
    """

    import io

    if not payload_path.exists() and not receipt_path.exists():
        return None
    payload_sha256: str | None = None

    def refused(reason: str) -> _TerminalCategoryStructureCacheLoad:
        return _TerminalCategoryStructureCacheLoad(
            valid=False,
            reason=str(reason),
            payload_sha256=payload_sha256,
            payload_bytes=b"",
            arrays={},
            capture_schema="",
            substitution_reference_hash="",
        )

    if not payload_path.exists() or not receipt_path.exists():
        return refused("payload_or_receipt_missing")
    try:
        payload_bytes = payload_path.read_bytes()
        import hashlib

        payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("schema") != _TERMINAL_CATEGORY_CACHE_RECEIPT_SCHEMA
            or receipt.get("cache_key") != str(cache_key)
            or receipt.get("payload_sha256") != payload_sha256
            or receipt.get("dense_mask_hash") != str(dense_mask_hash)
            or receipt.get("census_membership_hash") != str(census_membership_hash)
            or receipt.get("frozen_bank_hash") != str(frozen_bank_hash)
            or receipt.get("validation_open_token_hash")
            != str(validation_open_token_hash)
        ):
            return refused("receipt_identity_or_digest_mismatch")
        with np.load(io.BytesIO(payload_bytes), allow_pickle=False) as data:
            if (
                str(data["schema"]) != TERMINAL_CATEGORY_SIDECAR_SCHEMA
                or str(data["mask_hash"]) != str(dense_mask_hash)
                or str(data["census_membership_hash"]) != str(census_membership_hash)
                or str(data["frozen_bank_hash"]) != str(frozen_bank_hash)
            ):
                return refused("payload_identity_mismatch")
            if str(data["validation_open_token_hash"]) != str(
                validation_open_token_hash
            ):
                return refused("payload_validation_token_mismatch")
            if str(data["bce_matrix_semantics"]) != (
                "ordinary_ovr_train_frozen_mean_bernoulli_bce_v1"
            ) or str(data["substituted_bce_matrix_semantics"]) != (
                "substituted_target_affine_dense_exclusion_train_frozen_"
                "mean_bernoulli_bce_v1"
            ):
                return refused("payload_bce_semantics_mismatch")
            arrays = {
                name: np.asarray(data[name])
                for name in _TERMINAL_CATEGORY_SIDECAR_ARRAYS
            }
            capture_schema = str(data["capture_schema"])
            substitution_reference_hash = str(data["substitution_reference_hash"])
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
        zipfile_BadZipFile,
    ):
        return refused("payload_or_receipt_unreadable")
    stratum_count = int(arrays["stratum_order"].shape[0])
    logit_count = int(arrays["logit_order"].shape[0])
    matrix_shape = (stratum_count, logit_count)
    if (
        arrays["counts"].shape != (stratum_count,)
        or any(
            arrays[name].shape != matrix_shape
            for name in (
                "moment_mean",
                "moment_m2",
                "ordinary_calibrated_logit_mean",
                "ordinary_probability_matrix",
                "bce_matrix",
                "accuracy_matrix",
                "sensitivity_matrix",
                "sensitivity_defined",
                "specificity_matrix",
                "specificity_defined",
                "substituted_calibrated_logit_mean",
                "substituted_probability_matrix",
                "substituted_bce_matrix",
                "substituted_accuracy_matrix",
                "substituted_sensitivity_matrix",
                "substituted_sensitivity_defined",
                "substituted_specificity_matrix",
                "substituted_specificity_defined",
            )
        )
        or any(
            arrays[name].shape != (logit_count,)
            for name in (
                "calibration_scale",
                "calibration_offset",
                "calibration_input_center",
                "calibration_converged",
                "calibration_selected_strength",
                "calibration_state_id",
                "substituted_calibration_scale",
                "substituted_calibration_offset",
                "substituted_calibration_input_center",
                "substituted_calibration_converged",
                "substituted_calibration_selected_strength",
                "substituted_calibration_state_id",
            )
        )
    ):
        return refused("payload_axes_invalid")
    return _TerminalCategoryStructureCacheLoad(
        valid=True,
        reason=None,
        payload_sha256=payload_sha256,
        payload_bytes=payload_bytes,
        arrays=arrays,
        capture_schema=capture_schema,
        substitution_reference_hash=substitution_reference_hash,
    )


def _store_terminal_category_structure_cache(
    *,
    payload_path: Path,
    receipt_path: Path,
    cache_key: str,
    payload_bytes: bytes,
    dense_mask_hash: str,
    census_membership_hash: str,
    frozen_bank_hash: str,
    validation_open_token_hash: str,
) -> str:
    """Atomically store the dense-side payload + sha-bound receipt."""

    import hashlib

    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{cache_key}.",
        suffix=".npz",
        dir=payload_path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, payload_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    receipt = {
        "schema": _TERMINAL_CATEGORY_CACHE_RECEIPT_SCHEMA,
        "cache_key": str(cache_key),
        "payload_sha256": payload_sha256,
        "dense_mask_hash": str(dense_mask_hash),
        "census_membership_hash": str(census_membership_hash),
        "frozen_bank_hash": str(frozen_bank_hash),
        "validation_open_token_hash": str(validation_open_token_hash),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{cache_key}.",
        suffix=".json",
        dir=receipt_path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({**receipt, "effective_settings": effective_settings()}, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, receipt_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return payload_sha256


def _terminal_category_populations_by_class(
    validation_populations: Mapping[str, Sequence[str]],
    *,
    target_class_id: int,
) -> dict[int, tuple[str, ...]]:
    """Reindex sealed terminal strata onto the target-independent class axis.

    Contexts within each class are sorted so the capture's row order (and
    therefore the dense sidecar bytes) is identical for every query of a
    family, making the shared dense cache exact rather than approximate.
    """

    populations: dict[int, tuple[str, ...]] = {}
    for stratum, rows in validation_populations.items():
        if stratum == "on_target":
            class_id = int(target_class_id)
        elif stratum.startswith("off_target:"):
            class_id = int(stratum.split(":", 1)[1])
        else:
            raise ValueError(
                f"unrecognized terminal stratum {stratum!r} for the "
                "category-structure capture"
            )
        if class_id in populations:
            raise ValueError("terminal category capture strata collide on one class")
        populations[class_id] = tuple(sorted(str(v) for v in rows))
    return populations


def _terminal_validation_one_pass_enabled() -> bool:
    """Select shared validation readouts for terminal consumers."""

    return os.environ.get("IRG_TERMINAL_VALIDATION_ONE_PASS", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _open_dense_validation_table(
    *,
    terminal_provider: Any,
    dense_mask: Any,
    validation_axis_hash: str,
) -> tuple[Any | None, dict[str, Any] | None]:
    """The sealed per-model dense validation table, when one is configured.

    ``IRG_V6_DENSE_VALIDATION_TABLE`` names the table directory.  Unset (the
    default): no table.  A configured table that is
    absent is built from this run's primed dense lane only when
    ``IRG_V6_DENSE_VALIDATION_TABLE_BUILD=1``; a present table that fails its
    seal, identity or payload check fails the run closed.
    """

    from mechanistic_systems.measurement import dense_validation_table as dvt

    directory = dvt.configured_table_directory()
    if directory is None:
        return None, None
    if not hasattr(terminal_provider, "dense_validation_table_identity"):
        return None, None
    identity = terminal_provider.dense_validation_table_identity(
        dense_mask, validation_axis_hash=str(validation_axis_hash)
    )
    if not (directory / dvt.RECEIPT_NAME).exists():
        return None, {
            "status": "absent",
            "directory": str(directory),
            "identity_hash": identity.content_hash,
            "build_requested": dvt.build_requested(),
        }
    table = dvt.DenseValidationTable.open(directory, expected_identity=identity)
    return table, {"status": "opened", "directory": str(directory)}


def _prime_terminal_validation_one_pass(
    *,
    terminal_provider: Any,
    final_mask: Any,
    dense_mask: Any,
    validation_populations: Mapping[str, Sequence[str]],
    target_class_id: int,
    batch_factory: Any,
    validation_axis_hash: str,
) -> dict[str, Any] | None:
    """Prime the shared post-lock forward.

    Returns the receipt block, or ``None`` when one pass is off or the
    provider has no prime (then every consumer runs its own forward).
    """

    if not _terminal_validation_one_pass_enabled() or not hasattr(
        terminal_provider, "prime_terminal_validation_readouts"
    ):
        return None
    table, table_status = _open_dense_validation_table(
        terminal_provider=terminal_provider,
        dense_mask=dense_mask,
        validation_axis_hash=validation_axis_hash,
    )
    block: dict[str, Any] = {
        "prime": terminal_provider.prime_terminal_validation_readouts(
            final_mask,
            dense_mask,
            context_ids=_terminal_validation_class_order(
                validation_populations, target_class_id=target_class_id
            ),
            batch_factory=batch_factory,
            dense_validation_table=table,
        )
    }
    if table_status is not None:
        block["dense_validation_table"] = {
            **table_status,
            **({} if table is None else table.receipt_block()),
        }
        if table is None and table_status.get("build_requested"):
            block["dense_validation_table"]["build"] = _build_dense_validation_table(
                terminal_provider=terminal_provider,
                dense_mask=dense_mask,
                validation_axis_hash=validation_axis_hash,
            )
    return block


def _build_dense_validation_table(
    *,
    terminal_provider: Any,
    dense_mask: Any,
    validation_axis_hash: str,
) -> dict[str, Any]:
    """Build mode: seal this run's primed dense lane as the model's table."""

    from mechanistic_systems.measurement import dense_validation_table as dvt

    primed = terminal_provider.primed_dense_validation_rows(dense_mask)
    if primed is None:
        return {"status": "not_built", "reason": "no_primed_dense_forward"}
    context_ids, rows, labels = primed
    identity = terminal_provider.dense_validation_table_identity(
        dense_mask, validation_axis_hash=str(validation_axis_hash)
    )
    try:
        receipt = dvt.build_dense_validation_table(
            dvt.configured_table_directory(),
            identity=identity,
            context_ids=context_ids,
            rows=rows,
            labels=labels,
            builder={
                "role": dvt.POST_LOCK_VALIDATION_ROLE,
                "source": "primed_pair_forward",
            },
        )
    except FileExistsError:
        return {"status": "not_built", "reason": "built_concurrently_by_another_task"}
    return {
        "status": "built",
        "seal": receipt["seal"],
        "payload_sha256": receipt["payload_sha256"],
    }


def _release_terminal_validation_one_pass(
    *,
    terminal_provider: Any,
    one_pass: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Release the primed rows; complete the one-pass receipt block."""

    if one_pass is None:
        return None
    return {
        **one_pass,
        "release": terminal_provider.release_terminal_validation_readouts(),
        # Gate note: a consumer served from the primed rows still advances the
        # provider's ``forward_passes`` counter, which therefore counts LOGICAL
        # evaluations here; the physical validation forwards of this window are
        # the prime plus ``release.own_forward_by_consumer``.
        "forward_passes_counter_semantics": "logical_evaluations",
    }


def _terminal_validation_class_order(
    validation_populations: Mapping[str, Sequence[str]],
    *,
    target_class_id: int,
) -> tuple[str, ...]:
    """The category captures' row order: class id, then sorted context id.

    Strata that are not per-class (a pooled design) keep their own order;
    consumers receive the primed rows permuted to theirs either way.
    """

    try:
        populations = _terminal_category_populations_by_class(
            validation_populations, target_class_id=target_class_id
        )
    except ValueError:
        return tuple(
            str(context_id)
            for rows in validation_populations.values()
            for context_id in rows
        )
    return tuple(
        context_id
        for class_id in sorted(populations)
        for context_id in populations[class_id]
    )


@dataclass(frozen=True)
class _TrainFrozenTerminalCategoryBanks:
    """Transient fit bundle frozen before terminal validation is opened.

    The provider owns the scientific fit and immutable bank identities.  This
    pilot-local wrapper exists only to make the lifecycle order impossible to
    obscure: construction consumes optimization/training batches; the
    terminal leg below can evaluate the returned mappings but has no fitting
    inputs or fit callback.
    """

    dense_fit: FrozenCategoryCalibrationFitBundle
    # ``None`` when the FINAL-mask fit soft-failed on unconverged categories;
    # ``final_fit_degraded`` then carries the receipted diagnosis and the leg
    # writes the dense sidecar only.
    final_fit: FrozenCategoryCalibrationFitBundle | None
    validation_open_token: FrozenCategoryValidationOpenToken
    calibration_by_class: Mapping[int, tuple[str, ...]]
    calibration_membership_hash: str
    calibration_context_axis_hash: str
    final_equals_dense: bool
    fit_elapsed_seconds: float
    final_fit_degraded: Mapping[str, Any] | None = None
    # The device hard-gate counters and missing-category
    # receipts of each bank fitted here ({"dense": ..., "final": ...}).  An
    # older v6 cache can carry the bank without its diagnostic summary; that
    # role then has an explicit unavailable marker instead of invented counts.
    fit_gate_summaries: Mapping[str, Any] | None = None
    dense_reference: Mapping[str, Any] | None = None
    dense_fit_chunk: int = 8
    # Shared-process worker: the category memo slot of this dense bank,
    # and the dense terminal products an earlier arm captured with it.
    dense_memo_key: tuple[str, str, str] | None = None
    dense_memo_products: Mapping[str, Any] | None = None


def _category_dense_bank_key(
    provider: Any,
    *,
    dense_mask: HardMask,
    calibration_by_class: Mapping[int, Sequence[str]],
    calibration_context_axis_hash: str,
    calibration_bank_mode: CalibrationBankMode,
    category_fit_chunk: int,
) -> tuple[str, str, str] | None:
    """The memo slot of a dense bank: the provider's exact fit request hash.

    The request binds the masks, training rows and order, model state,
    provider identity, solver options and runtime profile, so two arms with
    one request fit the same bank and the later one installs the earlier
    one's bundle. A provider without the request surface shares nothing.
    """

    request = getattr(provider, "frozen_category_fit_request_hash", None)
    if not callable(request):
        return None
    request_hash = str(
        request(
            dense_mask,
            dense_mask,
            training_context_ids_by_class=calibration_by_class,
            training_axis_source_identity=str(calibration_context_axis_hash),
            mode=calibration_bank_mode,
            category_fit_chunk=int(category_fit_chunk),
        )
    )
    from . import category_memo

    memo = category_memo.active()
    if memo is None:
        return None
    return memo.key(provider._frozen_category_model_state_hash(), request_hash)


def _terminal_bank_gate_summary(provider: Any, bundle: Any) -> Any:
    """The provider's v6 gate summary of a bank it fitted (None otherwise)."""

    method = getattr(provider, "frozen_category_fit_gate_summary", None)
    if bundle is None or not callable(method):
        return None
    return method(str(bundle.request_hash))


def _freeze_terminal_category_banks(
    *,
    enabled: bool,
    provider: Any,
    dense_mask: HardMask,
    final_mask: HardMask,
    calibration_populations: Mapping[str, Sequence[str]],
    target_class_id: int,
    calibration_batch_factory: Any,
    calibration_context_axis_hash: str,
    category_fit_chunk: int | None = None,
    calibration_bank_mode: CalibrationBankMode = (
        CalibrationBankMode.CLAIM_BEARING_1000
    ),
    pre_frozen_dense_fit: FrozenCategoryCalibrationFitBundle | None = None,
) -> _TrainFrozenTerminalCategoryBanks | None:
    """Fit dense/final category banks using training rows only.

    This function must run after the terminal mask is locked and before the
    first validation population or validation batch factory is materialized.
    It deliberately returns no validation authority and performs no terminal
    evaluation.
    """

    if not enabled:
        return None
    calibration_by_class = _terminal_category_populations_by_class(
        calibration_populations,
        target_class_id=int(target_class_id),
    )
    fit_started = time.perf_counter()
    expected_training_order = tuple(
        context_id
        for class_id in sorted(calibration_by_class)
        for context_id in calibration_by_class[class_id]
    )
    fit_chunk = 16 if pre_frozen_dense_fit is not None else _terminal_category_fit_chunk(category_fit_chunk)
    reference: Mapping[str, Any] | None = None
    reference_dir = os.environ.get("IRG_TERMINAL_DENSE_REFERENCE")
    reference_sha256 = os.environ.get("IRG_TERMINAL_DENSE_REFERENCE_SHA256")
    if reference_dir is not None or reference_sha256 is not None:
        try:
            if not reference_dir or not reference_sha256:
                raise ValueError("terminal dense reference needs both directory and manifest sha256")
            candidate = _load_terminal_dense_reference(reference_dir, reference_sha256)
            saved_fit = candidate["dense_fit"]
            saved_key = candidate["manifest"]["binding_key"]
            if (
                saved_fit.mask_hash != dense_mask.mask_hash
                or saved_fit.dense_reference_mask_hash != dense_mask.mask_hash
                or saved_fit.training_context_ids != expected_training_order
                or saved_fit.bank.identity.mode != CalibrationBankMode(calibration_bank_mode)
                or saved_key["terminal_settings"]["calibration_axis_hash"] != str(calibration_context_axis_hash)
                or saved_key["terminal_settings"]["category_fit_chunk"] != fit_chunk
                or saved_key["terminal_settings"].get("runtime_numerics_profile") != _terminal_dense_numerics_profile(provider)
                or saved_key["terminal_settings"]["fit_semantic_request_hash"] != _terminal_dense_fit_semantic_hash(
                    provider, dense_mask, saved_fit, str(calibration_context_axis_hash), fit_chunk
                )
                or provider._frozen_category_model_state_hash() != saved_fit.model_state_hash
                or (pre_frozen_dense_fit is not None and pre_frozen_dense_fit.content_hash != saved_fit.content_hash)
            ):
                raise ValueError("terminal dense reference training binding key differs")
            installed = provider.install_frozen_category_calibration_fit_bundle(saved_fit)
            gate_summary = candidate["manifest"]["dense_fit_gate_summary"]
            installer = getattr(provider, "install_frozen_category_fit_gate_summary", None)
            if gate_summary is not None and callable(installer):
                installer(installed.request_hash, gate_summary)
            reference = candidate
            pre_frozen_dense_fit = installed
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            raise ValueError(f"terminal dense reference refused: {error}") from error
    # In the shared-process worker the dense bank of a category is
    # fitted once; later arms with the same fit request install that bundle.
    from . import category_memo

    memo = (
        category_memo.active()
        if reference is None and reference_dir is None
        else None
    )
    memo_key = (
        None
        if memo is None
        else _category_dense_bank_key(
            provider,
            dense_mask=dense_mask,
            calibration_by_class=calibration_by_class,
            calibration_context_axis_hash=str(calibration_context_axis_hash),
            calibration_bank_mode=CalibrationBankMode(calibration_bank_mode),
            category_fit_chunk=fit_chunk,
        )
    )
    memo_entry = None if memo_key is None else memo.dense_banks.get(memo_key)
    if memo_entry is not None:
        installed = provider.install_frozen_category_calibration_fit_bundle(
            memo_entry["dense_fit"]
        )
        installer = getattr(provider, "install_frozen_category_fit_gate_summary", None)
        if memo_entry["gate_summary"] is not None and callable(installer):
            installer(installed.request_hash, memo_entry["gate_summary"])
        pre_frozen_dense_fit = installed
        memo.note(product="dense_bank", source="memo", key=memo_key)
    if pre_frozen_dense_fit is None:
        dense_fit = provider.fit_frozen_category_calibration_bank(
            dense_mask,
            dense_mask,
            training_context_ids_by_class=calibration_by_class,
            training_batch_factory=calibration_batch_factory,
            training_axis_source_identity=str(calibration_context_axis_hash),
            mode=calibration_bank_mode,
            category_fit_chunk=_terminal_category_fit_chunk(category_fit_chunk),
        )
        if memo_key is not None:
            memo.dense_banks[memo_key] = {
                "dense_fit": dense_fit,
                "gate_summary": _terminal_bank_gate_summary(provider, dense_fit),
                "products": None,
            }
            memo.note(product="dense_bank", source="fitted", key=memo_key)
    else:
        dense_fit = pre_frozen_dense_fit
        if (
            dense_fit.mask_hash != dense_mask.mask_hash
            or dense_fit.dense_reference_mask_hash != dense_mask.mask_hash
            or dense_fit.training_context_ids != expected_training_order
            or dense_fit.bank.identity.mode
            != CalibrationBankMode(calibration_bank_mode)
        ):
            raise RuntimeError(
                "pre-frozen dense category bundle differs from terminal "
                "training populations or dense-mask lineage"
            )
        if memo_key is not None and memo_entry is None:
            memo.dense_banks[memo_key] = {
                "dense_fit": dense_fit,
                "gate_summary": _terminal_bank_gate_summary(provider, dense_fit),
                "products": None,
            }
            memo.note(product="dense_bank", source="pre_frozen", key=memo_key)
    final_equals_dense = final_mask.mask_hash == dense_mask.mask_hash
    calibration_membership_hash = dense_fit.bank.identity.training_membership_identity
    final_fit: FrozenCategoryCalibrationFitBundle | None = dense_fit
    final_fit_degraded: dict[str, Any] | None = None
    if not final_equals_dense:
        # A FINAL-mask category whose affine did not converge
        # must not abort the arm.  The registered terminal metrics and the
        # mask were locked before this point and never read these banks; the
        # dense bank is already frozen; so the only thing lost is the final
        # sidecar, and that loss is receipted with the offending categories.
        # A category whose device hard gates fail after the device
        # fallback is MISSING, per cell, and the bank is carried with it
        # up to the provisional limit (``calibration_backend``).  Above the
        # limit the provider raises the typed
        # ``TerminalBankMissingCellThresholdError``: it degrades this
        # final-mask bank exactly like an unconverged refit, and the receipt
        # names every missing category and the threshold outcome.  Within
        # the limit, until the bank module declares ordinary carriage,
        # the bank is built strictly and refuses
        # (``CalibrationBankUnreadyStatesError``, caught below), with the
        # threshold receipt attached.  The dense bank's limit is 0, so any
        # missing dense category aborts the dense fit, never caught here.
        # ``CalibrationFailClosedError`` remains typed for the one v6 path
        # with no missing-column semantics (single-column owner emission).
        from mechanistic_systems.backends.calibration_backend import (
            TerminalBankMissingCellThresholdError,
        )
        from mechanistic_systems.backends.factored_calibration import (
            CalibrationFailClosedError,
        )

        try:
            final_fit = provider.fit_frozen_category_calibration_bank(
                final_mask,
                dense_mask,
                training_context_ids_by_class=calibration_by_class,
                training_batch_factory=calibration_batch_factory,
                training_axis_source_identity=str(calibration_context_axis_hash),
                mode=calibration_bank_mode,
                category_fit_chunk=_terminal_category_fit_chunk(category_fit_chunk),
            )
        except (
            CalibrationBankNotReadyError,
            CalibrationContractError,
            CalibrationFailClosedError,
            TerminalBankMissingCellThresholdError,
        ) as error:
            final_fit = None
            final_fit_degraded = _terminal_category_degradation(
                error, stage="final_mask_category_fit"
            )
            # Honest degradation receipt: the missing-category threshold
            # outcome and the bank's v6 gate summary (the provider attaches
            # both to the typed threshold abort and to the strict
            # refusal it keeps while the bank module cannot carry an
            # unavailable ordinary column), and, for a fail-closed error,
            # the failed query columns.
            threshold_receipt = getattr(error, "missing_cell_threshold", None)
            if threshold_receipt is None and isinstance(
                error, TerminalBankMissingCellThresholdError
            ):
                threshold_receipt = error.receipt
            if threshold_receipt is not None:
                final_fit_degraded["missing_cell_threshold"] = dict(
                    threshold_receipt
                )
            error_gate_summary = getattr(error, "v6_gate_summary", None)
            if error_gate_summary is not None:
                final_fit_degraded["v6_gate_summary"] = dict(error_gate_summary)
            failed_axes = getattr(error, "failed_query_axis_ids", None)
            if failed_axes:
                final_fit_degraded["failed_query_axis_ids"] = [
                    str(value) for value in failed_axes
                ]
                if not final_fit_degraded["unconverged_class_ids"]:
                    # A fail-closed fit stops at its first chunk with a
                    # missing column: its categories are named (not 0), and
                    # the scope says the list is a lower bound.
                    parsed = sorted(
                        {
                            int(str(value).rsplit(":col:", 1)[1])
                            for value in failed_axes
                            if ":col:" in str(value)
                            and str(value).rsplit(":col:", 1)[1].isdigit()
                        }
                    )
                    final_fit_degraded["unconverged_class_ids"] = parsed
                    final_fit_degraded["unconverged_count"] = int(len(parsed))
                    final_fit_degraded["unconverged_class_ids_by_functional"] = {
                        "ordinary_ovr": list(parsed)
                    }
                    final_fit_degraded["unconverged_class_ids_scope"] = (
                        "first_failing_chunk_lower_bound_fit_stopped"
                    )
    validation_open_token = provider.open_frozen_category_validation(
        frozen_bundles=(
            (dense_fit,)
            if final_equals_dense or final_fit is None
            else (dense_fit, final_fit)
        )
    )
    return _TrainFrozenTerminalCategoryBanks(
        dense_fit=dense_fit,
        final_fit=final_fit,
        validation_open_token=validation_open_token,
        calibration_by_class=calibration_by_class,
        calibration_membership_hash=calibration_membership_hash,
        calibration_context_axis_hash=str(calibration_context_axis_hash),
        final_equals_dense=bool(final_equals_dense),
        fit_elapsed_seconds=float(time.perf_counter() - fit_started),
        final_fit_degraded=final_fit_degraded,
        fit_gate_summaries={
            **(_terminal_bank_gate_summaries(
                provider,
                dense_fit=dense_fit,
                dense_fitted_here=pre_frozen_dense_fit is None,
                final_fit=None if final_equals_dense else final_fit,
                final_degraded=final_fit_degraded,
            ) or {}),
            **({"dense_source": "campaign_reference"} if reference is not None else {}),
            **({"dense_source": "category_memo"} if memo_entry is not None else {}),
        } or None,
        dense_reference=reference,
        dense_fit_chunk=fit_chunk,
        dense_memo_key=memo_key,
        dense_memo_products=None if memo_entry is None else memo_entry["products"],
    )


def _terminal_bank_gate_summaries(
    provider: Any,
    *,
    dense_fit: Any,
    dense_fitted_here: bool,
    final_fit: Any,
    final_degraded: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """The v6 gate summaries of the banks frozen by this call."""

    if not callable(getattr(provider, "frozen_category_fit_gate_summary", None)):
        return None
    dense_summary = _terminal_bank_gate_summary(provider, dense_fit)
    final_summary = _terminal_bank_gate_summary(provider, final_fit)
    if dense_summary is None and not dense_fitted_here:
        dense_summary = {
            "status": "unavailable",
            "reason": "cache written before counters were persisted",
        }
    if final_summary is None and final_degraded is not None:
        final_summary = final_degraded.get("v6_gate_summary") or final_degraded.get(
            "missing_cell_threshold"
        )
    if dense_summary is None and final_summary is None:
        return None
    return {
        "dense": dense_summary,
        "dense_source": "fitted" if dense_fitted_here else "pre_frozen_step_setup",
        "final": final_summary,
    }


def _terminal_frozen_category_readout_numerics(provider: Any) -> str:
    """The registered readout numerics this capture runs under.

    The provider owns the setting (resolved once from
    ``IRG_FROZEN_CATEGORY_READOUT_NUMERICS``); a provider that does not expose
    it (a test double) runs the default ``runtime`` numerics.
    """

    mode = getattr(provider, "frozen_category_readout_numerics", None)
    return mode if isinstance(mode, str) and mode else "runtime"


def _terminal_frozen_category_readout_numerics_block(
    provider: Any, readout_numerics: str
) -> dict[str, Any]:
    receipt_method = getattr(provider, "frozen_category_readout_numerics_receipt", None)
    receipt = receipt_method() if callable(receipt_method) else None
    return {
        "frozen_category_readout_numerics": (
            {"setting": readout_numerics} if receipt is None else dict(receipt)
        )
    }


def _terminal_category_structure_leg(
    *,
    enabled: bool,
    provider: Any,
    dense_mask: HardMask,
    final_mask: HardMask,
    validation_populations: Mapping[str, Sequence[str]],
    target_class_id: int,
    batch_factory: Any,
    model_state_hash: str,
    context_axis_hash: str,
    forward_tier: int,
    cache_dir: Path | None,
    category_fit_chunk: int | None = None,
    progress_callback: Callable[..., None] | None = None,
    frozen_banks: _TrainFrozenTerminalCategoryBanks | None = None,
    dense_frozen_final: bool = False,
    phase_timing=None,
    _phase_defer_output: bool = False,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Capture dense/final validation matrices with train-frozen banks.

    The category-matrix validation estimand is a frozen-training-bank estimand:
    per-category ordinary and substituted affines are fitted on optimization
    rows before terminal validation, and validation contexts only evaluate those
    frozen banks.  Legacy same-census fitting is refused by the provider unless
    an explicit provenance-only override is used outside this leg.
    """

    from mechanistic_systems.measurement.phase_device_orchestration import (
        DeferredPhaseValue, product_digest, resolve_products, state_for,
    )

    def encode_sidecar(*args, **kwargs):
        if _phase_defer_output:
            state_for(provider).calls["terminal_products_deferred"] += 1
            return DeferredPhaseValue(lambda: _terminal_category_sidecar_bytes(*args, **kwargs))
        return _terminal_category_sidecar_bytes(*args, **kwargs)

    def dense_frozen_sidecar(value):
        if _phase_defer_output:
            return DeferredPhaseValue(lambda: _dense_frozen_sidecar_from_dense_bytes(resolve_products(value)))
        return _dense_frozen_sidecar_from_dense_bytes(value)

    def store_cache(**kwargs):
        if not _phase_defer_output:
            return _store_terminal_category_structure_cache(**kwargs)
        pending = getattr(provider, "_phase_pending_cache_writes", None)
        if pending is None:
            pending = provider._phase_pending_cache_writes = []
        def write():
            import fcntl
            with kwargs["payload_path"].with_suffix(".lock").open("a+b") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                try:
                    _store_terminal_category_structure_cache(**resolve_products(kwargs))
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
        pending.append(write)

    def _notify(event: str, **details: object) -> None:
        if progress_callback is not None:
            progress_callback(event, **details)

    if not enabled:
        return (
            {
                "schema": TERMINAL_CATEGORY_STRUCTURE_SCHEMA,
                "enabled": False,
            },
            {},
        )
    if frozen_banks is None:
        raise ValueError(
            "terminal category structure requires banks frozen on training "
            "before validation is opened"
        )
    import hashlib
    from mechanistic_systems.backends.calibration_backend import (
        DENSE_REFERENCE_BANK_MAXIMUM_MISSING_CATEGORIES,
        FINAL_MASK_BANK_MAXIMUM_MISSING_CATEGORY_FRACTION,
        TERMINAL_BANK_MISSING_CELL_POLICY_STATUS,
    )

    populations_by_class = _terminal_category_populations_by_class(
        validation_populations,
        target_class_id=target_class_id,
    )
    calibration_by_class = frozen_banks.calibration_by_class

    census_membership_hash = content_hash(
        {
            "schema": "irg_category_validation_membership_v1",
            "populations": {
                str(class_id): tuple(sorted(rows))
                for class_id, rows in populations_by_class.items()
            },
        },
        prefix="irg-category-validation-membership-",
    )
    calibration_membership_hash = frozen_banks.calibration_membership_hash
    if tuple(sorted(populations_by_class)) != tuple(sorted(calibration_by_class)):
        raise ValueError(
            "terminal category validation and calibration class axes differ"
        )
    calibration_contract = str(provider.calibration_options["optimization_contract"])
    readout_numerics = _terminal_frozen_category_readout_numerics(provider)
    cache_key = _terminal_category_structure_cache_key(
        model_state_hash=model_state_hash,
        dense_mask_hash=dense_mask.mask_hash,
        context_axis_hash=context_axis_hash,
        census_membership_hash=census_membership_hash,
        full_readout_projection_hash=(provider.terminal_full_readout_reference_hash),
        calibration_contract=calibration_contract,
        category_fit_chunk=_terminal_category_fit_chunk(category_fit_chunk),
        forward_tier=int(forward_tier),
        calibration_context_membership_hash=calibration_membership_hash,
        frozen_bank_hash=frozen_banks.dense_fit.bank.content_hash,
        final_frozen_bank_hash=(
            None
            if frozen_banks.final_fit is None
            else frozen_banks.final_fit.bank.content_hash
        ),
        validation_open_token_hash=(frozen_banks.validation_open_token.content_hash),
        frozen_category_readout_numerics=readout_numerics,
    )

    dense_fit = frozen_banks.dense_fit
    final_fit = frozen_banks.final_fit
    final_equals_dense = frozen_banks.final_equals_dense
    train_fit_elapsed = frozen_banks.fit_elapsed_seconds
    final_degraded: dict[str, Any] | None = (
        None
        if frozen_banks.final_fit_degraded is None
        else dict(frozen_banks.final_fit_degraded)
    )

    dense_cache_hit = False
    dense_cache_reason: str | None = None
    dense_bytes: bytes | None = None
    dense_capture: FrozenCategoryStructureCapture | None = None
    dense_capture_schema = ""
    dense_substitution_reference_hash = ""
    dense_elapsed = 0.0

    dense_token_hash = _dense_category_token_hash(dense_fit)
    reference = frozen_banks.dense_reference
    if reference is not None:
        expected_key = _terminal_dense_reference_key(
            provider=provider,
            dense_mask=dense_mask,
            dense_fit=dense_fit,
            model_state_hash=model_state_hash,
            validation_axis_hash=context_axis_hash,
            census_membership_hash=census_membership_hash,
            calibration_axis_hash=frozen_banks.calibration_context_axis_hash,
            category_fit_chunk=frozen_banks.dense_fit_chunk,
            forward_tier=forward_tier,
        )
        if reference["manifest"].get("binding_key") != expected_key:
            raise ValueError("terminal dense reference binding key differs from run")

    def _capture_dense() -> FrozenCategoryStructureCapture:
        capture = provider.capture_frozen_category_structure(
            dense_mask,
            dense_mask,
            fit_bundle=dense_fit,
            validation_open_token=frozen_banks.validation_open_token,
            evaluation_context_ids_by_class=populations_by_class,
            evaluation_batch_factory=batch_factory,
            evaluation_axis_source_identity=str(context_axis_hash),
        )
        return capture

    # The entry is keyed by this arm's validation-open
    # token (its final bank), so no other category or arm can read it: 1,000
    # entries of ~117 MB would be written and never hit.  The launcher sets
    # IRG_TERMINAL_CATEGORY_STRUCTURE_CACHE=0, which skips the lock, the load
    # and the store; the dense capture then runs as on a miss.  Unset (the
    # default here) keeps the cache.
    cache_disabled = cache_dir is not None and os.environ.get(
        "IRG_TERMINAL_CATEGORY_STRUCTURE_CACHE", "1"
    ).strip().lower() in {"0", "false", "no", "off"}
    if phase_timing is not None:
        phase_timing.mark("terminal_dense_capture_start")
    # Dense products an earlier arm of this category captured with the
    # same dense bank, used only under an identical reference binding key.
    memo_products = frozen_banks.dense_memo_products
    memo_binding_key = None
    if reference is None and frozen_banks.dense_memo_key is not None:
        memo_binding_key = _terminal_dense_reference_key(
            provider=provider,
            dense_mask=dense_mask,
            dense_fit=dense_fit,
            model_state_hash=model_state_hash,
            validation_axis_hash=context_axis_hash,
            census_membership_hash=census_membership_hash,
            calibration_axis_hash=frozen_banks.calibration_context_axis_hash,
            category_fit_chunk=frozen_banks.dense_fit_chunk,
            forward_tier=forward_tier,
        )
        if memo_products is not None and memo_products["binding_key"] != memo_binding_key:
            memo_products = None
    else:
        memo_products = None
    if memo_products is not None:
        dense_bytes = memo_products["dense_bytes"]
        dense_capture_schema = str(memo_products["dense_capture_schema"])
        dense_substitution_reference_hash = str(
            memo_products["dense_substitution_reference_hash"]
        )
        dense_cache_reason = None
    elif reference is not None:
        dense_bytes = reference["dense_bytes"]
        dense_capture_schema = str(reference["manifest"]["dense_capture_schema"])
        dense_substitution_reference_hash = str(
            reference["manifest"]["dense_substitution_reference_hash"]
        )
        dense_cache_reason = None
    elif cache_dir is not None and not cache_disabled:
        import fcntl

        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        payload_path = cache_root / f"{cache_key}.npz"
        receipt_path = cache_root / f"{cache_key}.json"
        lock_path = cache_root / f"{cache_key}.lock"
        _notify(
            "terminal_category_structure_cache_waiting",
            cache_key=cache_key,
        )
        with lock_path.open("a+b") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            try:
                loaded = _load_terminal_category_structure_cache(
                    payload_path=payload_path,
                    receipt_path=receipt_path,
                    cache_key=cache_key,
                    dense_mask_hash=dense_mask.mask_hash,
                    census_membership_hash=census_membership_hash,
                    frozen_bank_hash=dense_fit.bank.content_hash,
                    validation_open_token_hash=dense_token_hash,
                )
                if loaded is not None and loaded.valid:
                    dense_cache_hit = True
                    dense_bytes = loaded.payload_bytes
                    dense_capture_schema = loaded.capture_schema
                    dense_substitution_reference_hash = (
                        loaded.substitution_reference_hash
                    )
                    _notify(
                        "terminal_category_structure_cache_hit",
                        cache_key=cache_key,
                    )
                else:
                    if loaded is not None:
                        dense_cache_reason = loaded.reason
                        _notify(
                            "terminal_category_structure_cache_refused",
                            cache_key=cache_key,
                            reason=loaded.reason,
                        )
                    dense_started = time.perf_counter()
                    dense_capture = _capture_dense()
                    dense_capture_schema = str(dense_capture.schema)
                    dense_substitution_reference_hash = str(
                        dense_capture.dense_reference_validation_identity
                    )
                    dense_elapsed = time.perf_counter() - dense_started
                    dense_bytes = encode_sidecar(
                        dense_capture,
                        fit_bundle=dense_fit,
                        dense_token_hash=dense_token_hash,
                    )
                    store_cache(
                        payload_path=payload_path,
                        receipt_path=receipt_path,
                        cache_key=cache_key,
                        payload_bytes=dense_bytes,
                        dense_mask_hash=dense_mask.mask_hash,
                        census_membership_hash=census_membership_hash,
                        frozen_bank_hash=dense_fit.bank.content_hash,
                        validation_open_token_hash=dense_token_hash,
                    )
            finally:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
    else:
        dense_cache_reason = (
            "cache_disabled_write_only_entry" if cache_disabled else "no_cache_dir"
        )
        dense_started = time.perf_counter()
        dense_capture = _capture_dense()
        dense_capture_schema = str(dense_capture.schema)
        dense_substitution_reference_hash = str(
            dense_capture.dense_reference_validation_identity
        )
        dense_elapsed = time.perf_counter() - dense_started
        dense_bytes = encode_sidecar(
            dense_capture,
            fit_bundle=dense_fit,
            dense_token_hash=dense_token_hash,
        )
    assert dense_bytes is not None
    if (
        memo_binding_key is not None
        and memo_products is None
    ):
        from . import category_memo

        memo = category_memo.active()
        slot = None if memo is None else memo.dense_banks.get(frozen_banks.dense_memo_key)
        if slot is not None and slot["dense_fit"].content_hash == dense_fit.content_hash:
            memo_source = "captured" if dense_capture is not None else "cache_entry_read"
            completed_memo_key = frozen_banks.dense_memo_key

            def publish_dense_products(payload):
                slot["products"] = {
                    "binding_key": memo_binding_key,
                    "dense_bytes": payload,
                    "dense_capture_schema": dense_capture_schema,
                    "dense_substitution_reference_hash": dense_substitution_reference_hash,
                    "dense_matrix_sha256": hashlib.sha256(payload).hexdigest(),
                }
                memo.note(product="dense_terminal", source=memo_source, key=completed_memo_key)
                return payload

            if isinstance(dense_bytes, DeferredPhaseValue):
                pending_dense_bytes = dense_bytes
                # Publish only completed host bytes. The deferred capture stays
                # item-owned and resolves before asynchronous writes begin.
                dense_bytes = DeferredPhaseValue(
                    lambda: publish_dense_products(resolve_products(pending_dense_bytes))
                )
            else:
                publish_dense_products(dense_bytes)
    if phase_timing is not None:
        phase_timing.mark("terminal_dense_capture_end")

    final_bytes: bytes | None
    # Set only when the FINAL capture succeeded under a
    # tolerated bank whose substituted functional carries NaN columns.
    final_carriage: dict[str, Any] | None = None
    if phase_timing is not None:
        phase_timing.mark(
            "terminal_final_capture_start",
            skipped=bool(final_fit is None or final_equals_dense),
        )
    if final_fit is None:
        # The FINAL fit soft-failed upstream; nothing to capture.
        final_bytes = None
        final_elapsed = 0.0
        final_capture_schema = ""
        final_substitution_reference_hash = ""
    elif final_equals_dense:
        final_bytes = dense_bytes
        final_elapsed = 0.0
        final_capture_schema = dense_capture_schema
        final_substitution_reference_hash = dense_substitution_reference_hash
    else:
        final_started = time.perf_counter()
        try:
            final_capture = provider.capture_frozen_category_structure(
                final_mask,
                dense_mask,
                fit_bundle=final_fit,
                validation_open_token=frozen_banks.validation_open_token,
                evaluation_context_ids_by_class=populations_by_class,
                evaluation_batch_factory=batch_factory,
                evaluation_axis_source_identity=str(context_axis_hash),
                **(
                    {"hold_readouts_for_dense_frozen": True}
                    if dense_frozen_final else {}
                ),
            )
            final_capture_schema = str(final_capture.schema)
            final_substitution_reference_hash = str(
                final_capture.dense_reference_validation_identity
            )
            final_bytes = encode_sidecar(
                final_capture,
                fit_bundle=final_fit,
            )
            carriage = frozen_category_bank_capture_status(final_fit.bank)
            if carriage["status"] != "complete":
                final_carriage = dict(carriage, stage="final_mask_category_capture")
        except (CalibrationBankNotReadyError, CalibrationContractError) as error:
            # Same soft-fail as the fit: an unready per-category affine on
            # the FINAL bank degrades the final sidecar, never the arm.
            final_bytes = None
            final_capture_schema = ""
            final_substitution_reference_hash = ""
            final_degraded = _terminal_category_degradation(
                error, stage="final_mask_category_capture"
            )
            # The frozen FINAL bank's own unready ids survive a failed capture.
            final_degraded["final_bank_unready_class_ids_by_functional"] = {
                str(functional): [int(class_id) for class_id in ids]
                for functional, ids in (
                    final_fit.bank.unready_class_ids_by_functional.items()
                )
            }
            if final_fit.bank.tolerate_unconverged:
                # A tolerated FINAL bank froze with unready
                # substituted states (a strict bank would degrade the arm
                # at ``final_mask_category_fit`` with those ids at top level).
                # When its capture then fails,
                # the capture error carries no per-class detail, so merge the
                # bank's unready ids into the top-level fields summarizers
                # count. A strict bank has no unready state, so this is
                # skipped and the degradation block is unchanged.
                bank_unready = final_fit.bank.unready_class_ids_by_functional
                merged = sorted(
                    {
                        int(class_id)
                        for class_id in final_degraded["unconverged_class_ids"]
                    }
                    | {
                        int(class_id)
                        for ids in bank_unready.values()
                        for class_id in ids
                    }
                )
                final_degraded["unconverged_class_ids"] = merged
                final_degraded["unconverged_count"] = int(len(merged))
                final_degraded["unconverged_class_ids_source"] = (
                    "capture_error_union_tolerated_final_bank_unready_v1"
                )
        final_elapsed = time.perf_counter() - final_started
    if phase_timing is not None:
        phase_timing.mark(
            "terminal_final_capture_end",
            skipped=bool(final_fit is None or final_equals_dense),
        )
    sidecars: dict[str, bytes] = {"category_matrix_dense.npz": dense_bytes}
    if final_bytes is not None:
        sidecars["category_matrix_final.npz"] = final_bytes
    dense_frozen_bytes: bytes | None = None
    dense_frozen_elapsed = 0.0
    # A failed third capture degrades
    # its own file exactly as a failed final capture does, never the arm.
    dense_frozen_degraded: dict[str, Any] | None = None
    if dense_frozen_final:
        if phase_timing is not None:
            phase_timing.mark("terminal_dense_frozen_capture_start")
        dense_frozen_started = time.perf_counter()
        if final_equals_dense:
            dense_frozen_bytes = dense_frozen_sidecar(dense_bytes)
        else:
            try:
                dense_frozen_capture = provider.capture_frozen_category_structure(
                    final_mask,
                    dense_mask,
                    fit_bundle=dense_fit,
                    validation_open_token=frozen_banks.validation_open_token,
                    evaluation_context_ids_by_class=populations_by_class,
                    evaluation_batch_factory=batch_factory,
                    evaluation_axis_source_identity=str(context_axis_hash),
                    calibration_application="dense_frozen_final",
                )
                dense_frozen_bytes = encode_sidecar(
                    dense_frozen_capture,
                    fit_bundle=dense_fit,
                    calibration_application="dense_frozen_final",
                )
            except (CalibrationBankNotReadyError, CalibrationContractError) as error:
                dense_frozen_bytes = None
                dense_frozen_degraded = _terminal_category_degradation(
                    error, stage="final_dense_frozen_capture"
                )
                # Rows held by the final capture are consumed at the start of
                # a dense-frozen capture; drop them if it refused earlier.
                if getattr(provider, "_dense_frozen_final_readouts", None) is not None:
                    provider._dense_frozen_final_readouts = None
        dense_frozen_elapsed = time.perf_counter() - dense_frozen_started
        if dense_frozen_bytes is not None:
            sidecars["category_matrix_final_dense_frozen.npz"] = dense_frozen_bytes
        if phase_timing is not None:
            phase_timing.mark("terminal_dense_frozen_capture_end")
    class_axis_complete = tuple(sorted(populations_by_class)) == tuple(
        range(int(provider._scientific_readout_width))
    )
    # Exactly one of ``final_degraded`` (final sidecar absent) and
    # ``final_carriage`` (final sidecar present, substituted NaN columns) can
    # be set; neither means the complete path.
    final_status_source = final_degraded if final_degraded is not None else final_carriage
    block = {
        "schema": TERMINAL_CATEGORY_STRUCTURE_SCHEMA,
        "enabled": True,
        "dense_sidecar_validation_token_semantics": TERMINAL_DENSE_SIDECAR_TOKEN_SEMANTICS,
        **({} if reference is None else {
            "dense_reference": {
                "status": "bound",
                "manifest_sha256": reference["manifest_sha256"],
                "binding_key": reference["manifest"]["binding_key"],
                "dense_matrix_sha256": product_digest(dense_bytes),
                "dense_fit_bundle_hash": dense_fit.content_hash,
            },
        }),
        # Status: ``complete``,
        # ``degraded_unconverged_categories`` (final sidecar absent, counts
        # and class ids receipted, dense sidecar and every registered
        # surface unaffected), or,
        # ``captured_substituted_partially_unconverged`` (final sidecar
        # present; ordinary matrices complete; substituted classifier columns
        # NaN at exactly ``substituted_carriage.unconverged_class_ids``).
        "status": (
            "complete"
            if final_status_source is None
            else str(final_status_source["status"])
        ),
        "unconverged_count": (
            0
            if final_status_source is None
            else int(final_status_source["unconverged_count"])
        ),
        "unconverged_class_ids": (
            []
            if final_status_source is None
            else list(final_status_source["unconverged_class_ids"])
        ),
        "final_sidecar": None if final_bytes is None else "category_matrix_final.npz",
        "degradation": final_degraded,
        "fit_gate_summaries": (
            None
            if frozen_banks.fit_gate_summaries is None
            else dict(frozen_banks.fit_gate_summaries)
        ),
        "fit_gate_policy_status": TERMINAL_BANK_MISSING_CELL_POLICY_STATUS,
        "fit_gate_policy": {
            "status": TERMINAL_BANK_MISSING_CELL_POLICY_STATUS,
            "dense_reference_maximum_missing_categories": (
                DENSE_REFERENCE_BANK_MAXIMUM_MISSING_CATEGORIES
            ),
            "final_mask_maximum_missing_category_fraction": (
                FINAL_MASK_BANK_MAXIMUM_MISSING_CATEGORY_FRACTION
            ),
            "search_fit_missing_policy": "never_missing",
        },
        "substituted_carriage": final_carriage,
        # Present only under a strict-float32 setting.
        **_terminal_frozen_category_readout_numerics_block(
            provider, readout_numerics
        ),
        "sidecar_encoding": TERMINAL_CATEGORY_SIDECAR_ENCODING,
        "sidecars": {
            "dense": {
                "filename": "category_matrix_dense.npz",
                "sha256": product_digest(dense_bytes),
                "elapsed_seconds": float(dense_elapsed),
                **({} if reference is None else {
                    "source": "campaign_reference",
                    "reference_manifest_sha256": reference["manifest_sha256"],
                    "reference_binding_key": reference["manifest"]["binding_key"],
                    "copy_mode": "byte_identical_copy",
                }),
                **({} if memo_products is None else {
                    "source": "category_memo",
                    "reference_binding_key": memo_products["binding_key"],
                    "copy_mode": "byte_identical_copy",
                }),
            },
            "final": (
                None
                if final_bytes is None
                else {
                    "filename": "category_matrix_final.npz",
                    "sha256": product_digest(final_bytes),
                    "elapsed_seconds": float(final_elapsed),
                }
            ),
            **(
                {}
                if dense_frozen_bytes is None
                else {
                    "final_dense_frozen": {
                        "filename": "category_matrix_final_dense_frozen.npz",
                        "status": "complete",
                        "sha256": product_digest(dense_frozen_bytes),
                        "elapsed_seconds": float(dense_frozen_elapsed),
                        "calibration_application": (
                            "dense_train_frozen_calibration_applied_to_final_circuit"
                        ),
                        "validation_readout_reuse": (
                            "dense_mask_identity"
                            if final_equals_dense else
                            "final_capture_when_available"
                        ),
                    }
                }
            ),
            **(
                {}
                if dense_frozen_degraded is None
                else {
                    # The third file is absent; the dense and final files
                    # and every registered surface are unaffected.
                    "final_dense_frozen": {
                        "filename": None,
                        "status": str(dense_frozen_degraded["status"]),
                        "reason": str(dense_frozen_degraded["error_message"]),
                        "elapsed_seconds": float(dense_frozen_elapsed),
                        "degradation": dense_frozen_degraded,
                    }
                }
            ),
        },
        # One of ``complete``, ``degraded_unconverged_categories`` (third file
        # absent, reason under ``sidecars.final_dense_frozen``) or
        # ``not_requested``.  ``status`` above keeps its final-file meaning.
        "final_dense_frozen_status": (
            "not_requested"
            if not dense_frozen_final
            else "complete"
            if dense_frozen_degraded is None
            else str(dense_frozen_degraded["status"])
        ),
        "final_dense_frozen_sidecar": (
            None
            if dense_frozen_bytes is None
            else "category_matrix_final_dense_frozen.npz"
        ),
        "calibration_policy": (
            "per_category_one_vs_rest_affines_fitted_on_train_axis_before_"
            "terminal_validation; validation_evaluates_frozen_ordinary_and_"
            "substituted_banks_only; no_validation_fit_or_threshold_tuning"
        ),
        "substituted_functional": ("target_affine_then_dense_competitor_exclusion_v1"),
        "ordinary_bce_matrix_semantics": (
            "ordinary_ovr_train_frozen_mean_bernoulli_bce_v1"
        ),
        "substituted_bce_matrix_semantics": (
            "substituted_target_affine_dense_exclusion_train_frozen_mean_"
            "bernoulli_bce_v1"
        ),
        "accuracy_policy": (
            "ovr_threshold_logit_ge_0; sensitivity_component_diagonal; "
            "specificity_component_off_diagonal"
        ),
        "hybrid_top1_status": (
            "retired_independent_ovr_affines_are_not_joint_multiclass_" "calibration"
        ),
        "calibration_contract": calibration_contract,
        "category_fit_chunk": _terminal_category_fit_chunk(category_fit_chunk),
        "subsample": "validation_census_train_frozen_calibration",
        "context_order": "class_ascending_context_id_sorted_v1",
        "census_membership_hash": census_membership_hash,
        "calibration_context_membership_hash": calibration_membership_hash,
        "validation_context_axis_hash": str(context_axis_hash),
        "calibration_context_axis_hash": (frozen_banks.calibration_context_axis_hash),
        "class_axis_complete": bool(class_axis_complete),
        "dense_cache_key": cache_key,
        "dense_cache_hit": bool(dense_cache_hit),
        "dense_cache_refusal_reason": dense_cache_reason,
        "train_fit_elapsed_seconds": float(train_fit_elapsed),
        "final_equals_dense": bool(final_equals_dense),
        "dense_capture_schema": dense_capture_schema,
        "final_capture_schema": final_capture_schema,
        "dense_substitution_reference_hash": dense_substitution_reference_hash,
        "final_substitution_reference_hash": final_substitution_reference_hash,
        "final_substitution_reference_available": bool(
            final_equals_dense or final_substitution_reference_hash
        ),
        "validation_open_token_hash": (frozen_banks.validation_open_token.content_hash),
        "dense_frozen_bank_hash": dense_fit.bank.content_hash,
        "final_frozen_bank_hash": (
            None if final_fit is None else final_fit.bank.content_hash
        ),
        "calibration_fit_preexisted_validation": True,
        "calibration_and_evaluation_disjoint": True,
    }
    return block, sidecars


TERMINAL_RECOMPUTE_RECEIPT_NAME = "terminal_recompute_receipt.json"
TERMINAL_RECOMPUTE_RECEIPT_SCHEMA = "terminal_category_recompute_receipt_v1"
# The modules whose code produces the terminal files; the recomputing tree
# must hold the exact bytes this process imported.
_TERMINAL_RECOMPUTE_ENGINE_MODULES = (
    "mechanistic_systems.circuits.irg.registered_resnet18_pilot",
    "mechanistic_systems.measurement.native_irg_providers",
    "mechanistic_systems.measurement.frozen_category_calibration",
)


def _recomputing_engine_seal(tree: str | Path) -> str:
    """The manifest seal of the tree whose code is running this recompute."""

    from mechanistic_systems.measurement.native_irg_providers import (
        source_manifest_seal,
    )

    root = Path(tree)
    try:
        seal = source_manifest_seal(root)
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(
            "recomputing source tree does not match its manifest"
        ) from error
    for module_name in _TERMINAL_RECOMPUTE_ENGINE_MODULES:
        module = sys.modules.get(module_name)
        running = None if module is None else getattr(module, "__file__", None)
        copy = root.joinpath(*module_name.split(".")).with_suffix(".py")
        if (
            running is None
            or not copy.is_file()
            or Path(running).read_bytes() != copy.read_bytes()
        ):
            raise ValueError(
                "the running engine is not the recomputing source tree "
                f"({module_name})"
            )
    return seal


def _reload_terminal_frozen_banks(
    root: Path,
    *,
    receipt: Mapping[str, Any],
    previous: Mapping[str, Any],
    provider: Any,
    dense_mask: HardMask,
    final_mask: HardMask,
    dense_reference: Mapping[str, Any] | None = None,
) -> _TrainFrozenTerminalCategoryBanks:
    """Install the run's saved banks on a fresh provider and reopen validation.

    Evaluation only: no training row is read and no fit runs.  The provider's
    executable model must be the one the banks were fitted under (compared
    with the hash the banks carry, which is the same kind of hash), and the
    reissued validation token must equal the run's recorded token.
    """

    import hashlib

    entry = (previous.get("sidecars") or {}).get("frozen_banks") or {}
    name = entry.get("filename")
    if name != TERMINAL_FROZEN_BANKS_FILENAME or not (root / name).is_file():
        raise ValueError(
            "completed run has no saved frozen category banks; its terminal "
            "files cannot be recomputed bit for bit"
        )
    payload = (root / name).read_bytes()
    if hashlib.sha256(payload).hexdigest() != str(entry.get("sha256")):
        raise ValueError("saved frozen category banks differ from the run's receipt")
    saved = _load_terminal_frozen_banks(payload)
    reference_entry = previous.get("dense_reference")
    if isinstance(reference_entry, Mapping) and reference_entry.get("status") == "bound":
        if dense_reference is None:
            raise ValueError("completed run requires its dense reference directory")
        if (
            dense_reference["manifest"]["products"][TERMINAL_DENSE_REFERENCE_BANK]["sha256"]
            != entry.get("dense_reference_bank_sha256")
            or dense_reference["manifest_sha256"] != entry.get("dense_reference_manifest_sha256")
            or saved["roles"] not in ((), ("final",))
        ):
            raise ValueError("completed run and dense reference banks differ")
        dense_fit = dense_reference["dense_fit"]
    else:
        if saved["roles"] not in (("dense",), ("dense", "final")):
            raise ValueError("completed run has no dense category bank")
        dense_fit = saved["dense"]
    final_fit = saved.get("final")
    final_equals_dense = bool(saved["final_equals_dense"])
    if final_equals_dense:
        final_fit = dense_fit
    if (
        dense_fit.content_hash != entry.get("dense_fit_bundle_hash")
        or (None if "final" not in saved else saved["final"].content_hash)
        != entry.get("final_fit_bundle_hash")
    ):
        raise ValueError("completed run and frozen category banks differ")
    if (
        saved["calibration_context_axis_hash"]
        != str(previous.get("calibration_context_axis_hash"))
        or saved["calibration_context_axis_hash"]
        != str(receipt.get("decision_context_axis_hash"))
    ):
        raise ValueError("completed run and context axes differ")
    if provider._frozen_category_model_state_hash() != dense_fit.model_state_hash:
        raise ValueError("completed run and provider model states differ")
    if (
        dense_fit.mask_hash != dense_mask.mask_hash
        or final_fit is not None
        and final_fit.mask_hash != final_mask.mask_hash
    ):
        raise ValueError("saved frozen banks differ from the run's masks")
    installed = [provider.install_frozen_category_calibration_fit_bundle(dense_fit)]
    if final_fit is not None and not final_equals_dense:
        installed.append(provider.install_frozen_category_calibration_fit_bundle(final_fit))
    token = provider.open_frozen_category_validation(frozen_bundles=tuple(installed))
    if token.content_hash != str(saved["validation_open_token_hash"]):
        raise ValueError("completed run and frozen category banks differ")
    calibration_by_class: dict[int, list[str]] = {}
    for context_id, label in zip(
        dense_fit.training_context_ids, dense_fit.training_labels.tolist(), strict=True
    ):
        calibration_by_class.setdefault(int(label), []).append(str(context_id))
    degradation = previous.get("degradation")
    return _TrainFrozenTerminalCategoryBanks(
        dense_fit=dense_fit,
        final_fit=final_fit,
        validation_open_token=token,
        calibration_by_class={
            class_id: tuple(rows) for class_id, rows in calibration_by_class.items()
        },
        calibration_membership_hash=(
            dense_fit.bank.identity.training_membership_identity
        ),
        calibration_context_axis_hash=str(saved["calibration_context_axis_hash"]),
        final_equals_dense=final_equals_dense,
        fit_elapsed_seconds=float(previous.get("train_fit_elapsed_seconds") or 0.0),
        final_fit_degraded=(
            dict(degradation)
            if isinstance(degradation, Mapping)
            and degradation.get("stage") == "final_mask_category_fit"
            else None
        ),
        fit_gate_summaries=previous.get("fit_gate_summaries"),
        dense_reference=dense_reference,
        dense_fit_chunk=(
            int(dense_reference["manifest"]["binding_key"]["terminal_settings"]["category_fit_chunk"])
            if dense_reference is not None else _terminal_category_fit_chunk(None)
        ),
    )


def recompute_terminal_category_matrices(
    run_dir: str | Path,
    *,
    output_dir: str | Path,
    source_tree: str | Path,
    recomputing_source_tree: str | Path,
    provider: Any,
    source_model_state_hash: str,
    validation_populations: Mapping[str, Sequence[str]],
    validation_batch_factory: Any,
    validation_context_axis_hash: str,
    expected_final_mask_hash: str | None = None,
    category_fit_chunk: int | None = None,
    frozen_banks: _TrainFrozenTerminalCategoryBanks | None = None,
    dense_reference_dir: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Recompute only the terminal files of a completed, identity-matched run.

    The run must have saved its frozen category banks (every terminal-on run
    of this engine does).  ``provider`` is a fresh provider over the run's
    registered model; the banks are installed on it and validation is
    reopened with the run's own token, so nothing is fitted and no walk,
    proposal or training row is used.  ``frozen_banks`` may instead pass the
    in-memory banks of the process that made the run.

    Anchors, each taken from the completed run: its source-tree seal, its
    source model hash (``source_model_state_hash`` is the loaded checkpoint's
    state-dict hash, the same kind the receipt records), its validation axis
    and census, its bank, bundle and token hashes, and the bytes of every
    terminal file it holds.  A recomputed file that the run holds must match
    it byte for byte, or nothing is written.  Files the run lacks are written
    to ``output_dir``, a new folder outside the run, with a recompute receipt
    naming both source seals, the bank hashes and every file's hash.
    """

    import hashlib

    from mechanistic_systems.measurement.run_products import load_run_receipt
    from mechanistic_systems.measurement.native_irg_providers import source_manifest_seal

    root = Path(run_dir).resolve()
    output = Path(output_dir).resolve()
    if output == root or root in output.parents:
        raise ValueError("recompute output must be a new folder outside the run")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("recompute output folder already holds files")
    receipt = load_run_receipt(root, verify_products=True)
    try:
        seal = source_manifest_seal(source_tree)
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError("completed run and source tree seals differ") from error
    source = receipt.get("sealed_source_identity") or {}
    if seal is None or source.get("source_manifest_sha256") != seal:
        raise ValueError("completed run and source tree seals differ")
    recomputing_seal = _recomputing_engine_seal(recomputing_source_tree)
    model_state_hash = str(receipt["model_state_hash"])
    if str(source_model_state_hash) != model_state_hash:
        raise ValueError("completed run and supplied source model states differ")
    previous = receipt.get("terminal_category_structure") or {}
    if not previous.get("enabled"):
        raise ValueError(
            "completed run ran without the terminal category matrix; nothing "
            "it recorded anchors a recompute"
        )
    if str(receipt["final_validation_context_axis_hash"]) != str(
        validation_context_axis_hash
    ) or str(previous.get("validation_context_axis_hash")) != str(
        validation_context_axis_hash
    ):
        raise ValueError("completed run and context axes differ")
    route_axis_hash = str(receipt["route_axis_hash"])
    route_count = int(receipt["route_count"])
    final_mask = HardMask(
        route_axis_hash=route_axis_hash,
        route_count=route_count,
        packed_bits=(root / "final_mask.packbits").read_bytes(),
    )
    if (
        expected_final_mask_hash is not None
        and final_mask.mask_hash != expected_final_mask_hash
    ):
        raise ValueError("completed run's final mask differs from its expected hash")
    dense_mask = HardMask.dense(route_count, route_axis_hash=route_axis_hash)
    target_class_id = int(receipt["target_class_id"])
    census_by_class = _terminal_category_populations_by_class(
        validation_populations, target_class_id=target_class_id
    )
    census_hash = content_hash(
        {
            "schema": "irg_category_validation_membership_v1",
            "populations": {
                str(class_id): tuple(sorted(rows))
                for class_id, rows in census_by_class.items()
            },
        },
        prefix="irg-category-validation-membership-",
    )
    if census_hash != previous.get("census_membership_hash"):
        raise ValueError("completed run and validation census differ")
    reference_entry = previous.get("dense_reference")
    reference = None
    if reference_entry is not None:
        if not isinstance(reference_entry, Mapping) or reference_entry.get("status") != "bound":
            raise ValueError("completed run has an invalid dense reference entry")
        if dense_reference_dir is None:
            raise ValueError("completed run requires its dense reference directory")
        try:
            reference = _load_terminal_dense_reference(
                dense_reference_dir, str(reference_entry.get("manifest_sha256"))
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(f"completed run's dense reference differs: {error}") from error
        if (
            reference["manifest"]["binding_key"] != reference_entry.get("binding_key")
            or reference["dense_fit"].content_hash != reference_entry.get("dense_fit_bundle_hash")
        ):
            raise ValueError("completed run and dense reference banks differ")
    frozen = (
        frozen_banks
        if frozen_banks is not None
        else _reload_terminal_frozen_banks(
            root,
            receipt=receipt,
            previous=previous,
            provider=provider,
            dense_mask=dense_mask,
            final_mask=final_mask,
            dense_reference=reference,
        )
    )
    if (
        frozen.dense_fit.mask_hash != dense_mask.mask_hash
        or frozen.final_fit is not None
        and frozen.final_fit.mask_hash != final_mask.mask_hash
    ):
        raise ValueError("supplied frozen banks differ from the run's masks")
    final_bank_hash = (
        None if frozen.final_fit is None else frozen.final_fit.bank.content_hash
    )
    if (
        frozen.dense_fit.bank.content_hash != previous.get("dense_frozen_bank_hash")
        or final_bank_hash != previous.get("final_frozen_bank_hash")
        or frozen.validation_open_token.content_hash
        != previous.get("validation_open_token_hash")
    ):
        raise ValueError("completed run and frozen category banks differ")
    if reference is not None and (
        frozen.dense_reference is None
        or frozen.dense_fit.content_hash != reference["dense_fit"].content_hash
    ):
        raise ValueError("supplied frozen banks differ from the dense reference")
    block, sidecars = _terminal_category_structure_leg(
        enabled=True,
        provider=provider,
        dense_mask=dense_mask,
        final_mask=final_mask,
        validation_populations=validation_populations,
        target_class_id=target_class_id,
        batch_factory=validation_batch_factory,
        model_state_hash=model_state_hash,
        context_axis_hash=validation_context_axis_hash,
        forward_tier=int(receipt["runtime_execution_profile"]["forward_tier"]),
        cache_dir=None,
        category_fit_chunk=category_fit_chunk,
        frozen_banks=frozen,
        dense_frozen_final=True,
    )
    block, sidecars = _with_terminal_frozen_banks(block, sidecars, frozen)
    files: dict[str, dict[str, Any]] = {}
    for name, payload in sidecars.items():
        path = root / name
        if path.exists() and path.read_bytes() != payload:
            raise ValueError(f"recomputed {name} differs from the completed run")
        files[name] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": int(len(payload)),
            "status": (
                "matched_completed_run" if path.exists() else "written_to_recompute_folder"
            ),
        }
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in sidecars.items():
        if files[name]["status"] == "written_to_recompute_folder":
            (output / name).write_bytes(payload)
    recompute_receipt: dict[str, Any] = {
        "schema": TERMINAL_RECOMPUTE_RECEIPT_SCHEMA,
        "completed_run": {
            "directory": str(root),
            "receipt_hash": receipt.get("receipt_hash"),
            "source_manifest_sha256": seal,
        },
        "recomputing_source_manifest_sha256": recomputing_seal,
        "source_model_state_hash": model_state_hash,
        "executable_model_state_hash": provider._frozen_category_model_state_hash(),
        "banks": {
            "source": "run_saved" if frozen_banks is None else "caller_in_memory",
            **({} if reference is None else {
                "dense_reference_manifest_sha256": reference["manifest_sha256"],
            }),
            "dense_fit_bundle_hash": frozen.dense_fit.content_hash,
            "final_fit_bundle_hash": (
                None if frozen.final_fit is None else frozen.final_fit.content_hash
            ),
            "dense_frozen_bank_hash": frozen.dense_fit.bank.content_hash,
            "final_frozen_bank_hash": final_bank_hash,
            "validation_open_token_hash": frozen.validation_open_token.content_hash,
        },
        "files": files,
        "terminal_category_structure": block,
    }
    recompute_receipt["effective_settings"] = effective_settings()
    recompute_receipt["receipt_hash"] = content_hash(
        recompute_receipt, prefix="irg-terminal-recompute-"
    )
    (output / TERMINAL_RECOMPUTE_RECEIPT_NAME).write_text(
        json.dumps(recompute_receipt, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return block, sidecars


# ---------------------------------------------------------------------------
# Per-step category-structure stack — setup leg, cluster family
# cache, sidecar serialization, and the additive receipt block.  The dense
# frozen frame and the tracker anchor ride the cache trio above (shared
# across arms); clusters derive ONCE per family from the cached dense matrix.
# ---------------------------------------------------------------------------

STEP_CATEGORY_SIDECAR_FILENAME = "step_category_structure.npz"
_STEP_CATEGORY_CLUSTER_RECEIPT_SCHEMA = "step_category_cluster_cache_receipt_v1"
_DENSE_CATEGORY_FIT_CACHE_RECEIPT_SCHEMA = "dense_category_fit_bundle_cache_receipt_v1"
# The one receipt key a writer with a fit-semantic request adds.
_DENSE_FIT_SEMANTIC_REQUEST_RECEIPT_KEY = "fit_semantic_request"
_DENSE_FIT_SEMANTIC_REQUEST_SCHEMA = "dense_category_fit_semantic_request_v1"


_CATEGORY_MATRIX_ARRAY_NAMES = (
    "class_ids",
    "labels",
    "stratum_counts",
    "mean_functional_input",
    "mean_calibrated_logit",
    "mean_probability",
    "mean_bce",
    "binary_accuracy",
    "sensitivity_component",
    "sensitivity_defined",
    "specificity_component",
    "specificity_defined",
)


def _category_matrix_cache_arrays(
    prefix: str,
    matrices: CategoryStratifiedEvaluationMatrices,
) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_{name}": np.asarray(getattr(matrices, name))
        for name in _CATEGORY_MATRIX_ARRAY_NAMES
    }


def _category_matrix_from_cache(
    data: Mapping[str, Any],
    *,
    prefix: str,
    evaluation_hash: str,
    functional: CalibrationFunctional,
) -> CategoryStratifiedEvaluationMatrices:
    return CategoryStratifiedEvaluationMatrices(
        evaluation_hash=str(evaluation_hash),
        functional=functional,
        **{
            name: np.asarray(data[f"{prefix}_{name}"])
            for name in _CATEGORY_MATRIX_ARRAY_NAMES
        },
    )


def _fit_bundle_payload_arrays(
    bundle: FrozenCategoryCalibrationFitBundle,
    *,
    fit_gate_summary: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """The npz members of one frozen fit bundle (the dense-cache layout).

    Shared by the architecture-wide dense cache and the per-run frozen-bank
    product; member names and order are the cache's historical ones.
    """

    states = bundle.bank.states
    arrays: dict[str, np.ndarray] = {
        "training_context_ids": np.asarray(bundle.training_context_ids),
        "training_labels": np.asarray(bundle.training_labels, dtype=np.int64),
        "training_counts": np.asarray(bundle.training_counts, dtype=np.int64),
        "training_raw_logit_mean": np.asarray(
            bundle.training_raw_logit_mean, dtype=np.float64
        ),
        "training_raw_logit_m2": np.asarray(
            bundle.training_raw_logit_m2, dtype=np.float64
        ),
        "state_id": np.asarray([state.state_id for state in states]),
        "state_class_id": np.asarray(
            [state.class_id for state in states], dtype=np.int64
        ),
        "state_functional": np.asarray([state.functional.value for state in states]),
        "state_scale": np.asarray([state.scale for state in states], dtype=np.float64),
        "state_offset": np.asarray(
            [state.offset for state in states], dtype=np.float64
        ),
        "state_input_center": np.asarray(
            [state.input_center for state in states], dtype=np.float64
        ),
        "state_available": np.asarray(
            [state.available for state in states], dtype=np.bool_
        ),
        "state_converged": np.asarray(
            [state.converged for state in states], dtype=np.bool_
        ),
        "ordinary_selected_strength": np.asarray(bundle.ordinary_selected_strength),
        "substituted_selected_strength": np.asarray(
            bundle.substituted_selected_strength
        ),
        "ordinary_fit_receipt_hashes": np.asarray(bundle.ordinary_fit_receipt_hashes),
        "substituted_fit_receipt_hashes": np.asarray(
            bundle.substituted_fit_receipt_hashes
        ),
    }
    if fit_gate_summary is not None:
        arrays["fit_gate_summary_json"] = np.asarray(
            json.dumps(fit_gate_summary, sort_keys=True, allow_nan=False)
        )
    arrays.update(
        _category_matrix_cache_arrays("ordinary", bundle.ordinary_training_matrices)
    )
    arrays.update(
        _category_matrix_cache_arrays(
            "substituted", bundle.substituted_training_matrices
        )
    )
    return arrays


def _fit_bundle_payload_metadata(
    bundle: FrozenCategoryCalibrationFitBundle,
) -> dict[str, Any]:
    """The JSON fields that, with the npz members, rebuild a bundle exactly."""

    identity = bundle.bank.identity
    return {
        "bundle_hash": bundle.content_hash,
        "bundle": {
            "request_hash": bundle.request_hash,
            "mask_hash": bundle.mask_hash,
            "dense_reference_mask_hash": bundle.dense_reference_mask_hash,
            "model_state_hash": bundle.model_state_hash,
            "provider_identity": bundle.provider_identity,
            "dense_reference_training_identity": (
                bundle.dense_reference_training_identity
            ),
            "mask_training_readout_hash": bundle.mask_training_readout_hash,
            "dense_reference_training_readout_hash": (
                bundle.dense_reference_training_readout_hash
            ),
            "layout_content_hash": bundle.layout_content_hash,
            "layout_block_assignment_hash": (bundle.layout_block_assignment_hash),
            "layout_stratification_rule": bundle.layout_stratification_rule,
            "ordinary_evaluation_hash": (
                bundle.ordinary_training_matrices.evaluation_hash
            ),
            "substituted_evaluation_hash": (
                bundle.substituted_training_matrices.evaluation_hash
            ),
        },
        "identity": {
            "mask_identity": identity.mask_identity,
            "reference_identity": identity.reference_identity,
            "model_identity": identity.model_identity,
            "ontology_identity": identity.ontology_identity,
            "ordered_class_identity": identity.ordered_class_identity,
            "training_axis_identity": identity.training_axis_identity,
            "training_membership_identity": (identity.training_membership_identity),
            "training_order_identity": identity.training_order_identity,
            "training_label_identity": identity.training_label_identity,
            "training_weight_identity": identity.training_weight_identity,
            "solver_identity": identity.solver_identity,
            "mode": identity.mode.value,
        },
    }


def _fit_bundle_from_payload(
    data: Any,
    *,
    bundle_meta: Mapping[str, Any],
    identity_meta: Mapping[str, Any],
) -> FrozenCategoryCalibrationFitBundle:
    """Rebuild one frozen fit bundle from its npz members and JSON fields."""

    contexts = tuple(str(value) for value in data["training_context_ids"])
    class_ids = tuple(int(value) for value in data["ordinary_class_ids"])
    identity = CategoryCalibrationIdentity(
        **{
            name: identity_meta[name]
            for name in (
                "mask_identity",
                "reference_identity",
                "model_identity",
                "ontology_identity",
                "ordered_class_identity",
                "training_axis_identity",
                "training_membership_identity",
                "training_order_identity",
                "training_label_identity",
                "training_weight_identity",
                "solver_identity",
            )
        },
        ordered_class_ids=class_ids,
        training_member_ids=contexts,
        mode=CalibrationBankMode(identity_meta["mode"]),
    )
    bank_identity_hash = identity.content_hash
    states = tuple(
        FrozenAffineCalibrationState(
            state_id=str(data["state_id"][index]),
            bank_identity_hash=bank_identity_hash,
            class_id=int(data["state_class_id"][index]),
            functional=CalibrationFunctional(
                str(data["state_functional"][index])
            ),
            scale=float(data["state_scale"][index]),
            offset=float(data["state_offset"][index]),
            input_center=float(data["state_input_center"][index]),
            available=bool(data["state_available"][index]),
            converged=bool(data["state_converged"][index]),
        )
        for index in range(int(data["state_id"].shape[0]))
    )
    bank = FrozenCategoryCalibrationBank(identity=identity, states=states)
    ordinary = _category_matrix_from_cache(
        data,
        prefix="ordinary",
        evaluation_hash=bundle_meta["ordinary_evaluation_hash"],
        functional=CalibrationFunctional.ORDINARY_OVR,
    )
    substituted = _category_matrix_from_cache(
        data,
        prefix="substituted",
        evaluation_hash=bundle_meta["substituted_evaluation_hash"],
        functional=(
            CalibrationFunctional.SUBSTITUTED_TARGET_AFFINE_DENSE_EXCLUSION
        ),
    )
    bundle = FrozenCategoryCalibrationFitBundle(
        bank=bank,
        request_hash=str(bundle_meta["request_hash"]),
        mask_hash=str(bundle_meta["mask_hash"]),
        dense_reference_mask_hash=str(bundle_meta["dense_reference_mask_hash"]),
        model_state_hash=str(bundle_meta["model_state_hash"]),
        provider_identity=str(bundle_meta["provider_identity"]),
        dense_reference_training_identity=str(
            bundle_meta["dense_reference_training_identity"]
        ),
        training_context_ids=contexts,
        training_labels=np.asarray(data["training_labels"]),
        training_counts=np.asarray(data["training_counts"]),
        training_raw_logit_mean=np.asarray(data["training_raw_logit_mean"]),
        training_raw_logit_m2=np.asarray(data["training_raw_logit_m2"]),
        ordinary_training_matrices=ordinary,
        substituted_training_matrices=substituted,
        mask_training_readout_hash=str(
            bundle_meta["mask_training_readout_hash"]
        ),
        dense_reference_training_readout_hash=str(
            bundle_meta["dense_reference_training_readout_hash"]
        ),
        layout_content_hash=str(bundle_meta["layout_content_hash"]),
        layout_block_assignment_hash=str(
            bundle_meta["layout_block_assignment_hash"]
        ),
        layout_stratification_rule=str(
            bundle_meta["layout_stratification_rule"]
        ),
        ordinary_selected_strength=tuple(
            str(value) for value in data["ordinary_selected_strength"]
        ),
        substituted_selected_strength=tuple(
            str(value) for value in data["substituted_selected_strength"]
        ),
        ordinary_fit_receipt_hashes=tuple(
            str(value) for value in data["ordinary_fit_receipt_hashes"]
        ),
        substituted_fit_receipt_hashes=tuple(
            str(value) for value in data["substituted_fit_receipt_hashes"]
        ),
    )
    return bundle


def _store_dense_category_fit_bundle_cache(
    *,
    payload_path: Path,
    receipt_path: Path,
    cache_key: str,
    bundle: FrozenCategoryCalibrationFitBundle,
    fit_semantic_request: Mapping[str, Any] | None = None,
    fit_gate_summary: Mapping[str, Any] | None = None,
) -> str:
    """Atomically persist one architecture-wide dense training bundle.

    The writer passes ``fit_semantic_request``,
    the request digest computed without the runtime execution profile, and
    the receipt records it under one extra top-level key.  The v5 loader
    ignores unknown keys, so sealed v5 readers still hit these entries; with
    ``None`` the receipt is byte-for-byte the v5 receipt.
    """

    import hashlib

    arrays = _fit_bundle_payload_arrays(bundle, fit_gate_summary=fit_gate_summary)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{payload_path.name}.",
        suffix=".npz",
        dir=payload_path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        payload_sha256 = sha256_file(temporary_name)
        os.replace(temporary_name, payload_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    receipt = {
        "schema": _DENSE_CATEGORY_FIT_CACHE_RECEIPT_SCHEMA,
        "cache_key": str(cache_key),
        "payload_sha256": payload_sha256,
        **_fit_bundle_payload_metadata(bundle),
    }
    if fit_semantic_request is not None:
        receipt[_DENSE_FIT_SEMANTIC_REQUEST_RECEIPT_KEY] = dict(fit_semantic_request)
    if fit_gate_summary is not None:
        receipt["fit_gate_summary"] = dict(fit_gate_summary)
    receipt["effective_settings"] = effective_settings()
    receipt_bytes = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{receipt_path.name}.",
        suffix=".json",
        dir=receipt_path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(receipt_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, receipt_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return payload_sha256


def _load_dense_category_fit_bundle_cache(
    *,
    payload_path: Path,
    receipt_path: Path,
    cache_key: str,
    fit_gate_summary_out: dict[str, Any] | None = None,
) -> FrozenCategoryCalibrationFitBundle | None:
    """Load and reconstruct a content-addressed dense training bundle."""

    if not payload_path.exists() or not receipt_path.exists():
        return None
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("schema") != _DENSE_CATEGORY_FIT_CACHE_RECEIPT_SCHEMA
            or receipt.get("cache_key") != str(cache_key)
        ):
            return None
        # One read, hashed in file order while it fills
        # the member arrays; compared with the receipt before any use.
        verified_payload = read_verified_npz(payload_path)
        if receipt.get("payload_sha256") != verified_payload.sha256:
            return None
        bundle_meta = dict(receipt["bundle"])
        identity_meta = dict(receipt["identity"])
        with verified_payload as data:
            if "fit_gate_summary_json" in data:
                payload_summary = json.loads(str(data["fit_gate_summary_json"].item()))
                if payload_summary != receipt.get("fit_gate_summary"):
                    return None
            else:
                payload_summary = None
            bundle = _fit_bundle_from_payload(
                data, bundle_meta=bundle_meta, identity_meta=identity_meta
            )
        if bundle.content_hash != receipt.get("bundle_hash"):
            return None
        if fit_gate_summary_out is not None and payload_summary is not None:
            fit_gate_summary_out.update(payload_summary)
        return bundle
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
        zipfile_BadZipFile,
    ):
        return None


def _step_category_sidecar_bytes(payload: Mapping[str, Any]) -> bytes:
    """Deterministic npz payload (sorted keys) for the per-step sidecar.

    IN-MEMORY PATH, used when step-matrix spilling is off. It materializes the
    whole archive twice -- once in the BytesIO and again in ``getvalue()`` --
    on top of the stacked planes, which is why the spilled path below exists.
    """

    import io

    arrays = {name: np.asarray(payload[name]) for name in sorted(payload)}
    stream = io.BytesIO()
    # Measured: zlib on this ~1.8 GB of dense float32 buys ~7%
    # size for ~45 s per arm ('pilot_completed' closing gap). Values are
    # identical either way; .npz container unchanged. Uncompressed write.
    np.savez(stream, **arrays)
    return stream.getvalue()


def _write_step_category_sidecar(
    destination: Any,
    payload: Mapping[str, Any],
    *,
    member_paths: Mapping[str, str],
) -> tuple[Any, str]:
    """Write the sidecar and return ``(path_or_bytes, sha256)``.

    Two paths, one contract. With spilling OFF the payload is fully in memory
    and the in-memory bytes path is used unchanged. With spilling ON the nine
    large families are already ``.npy`` files, so the archive is assembled by
    streaming them in and the digest is taken over the file -- neither the
    arrays nor the archive is ever held whole. That removes the dominant term
    in the ~19.9 GiB finalize peak (stacked planes + BytesIO + getvalue copy).
    """

    from mechanistic_systems.measurement.native_irg_providers import (
        split_spilled_payload,
    )
    from mechanistic_systems.measurement.step_matrix_store import (
        write_npz_streaming,
    )

    arrays, member_files = split_spilled_payload(payload, member_paths=member_paths)
    if not member_files:
        payload_bytes = _step_category_sidecar_bytes(payload)
        import hashlib

        return payload_bytes, hashlib.sha256(payload_bytes).hexdigest()
    target = str(destination)
    payload_sha256 = write_npz_streaming(
        target,
        arrays={name: np.asarray(value) for name, value in arrays.items()},
        member_files=dict(member_files),
    )
    return target, payload_sha256


def _load_step_category_cluster_cache(
    *,
    payload_path: Path,
    receipt_path: Path,
    dense_cache_key: str,
    dense_payload_sha256: str,
) -> dict[str, Any] | None:
    """sha256-verified family cluster-assignment load; refusal returns None."""

    import hashlib
    import io

    if not payload_path.exists() or not receipt_path.exists():
        return None
    try:
        payload_bytes = payload_path.read_bytes()
        payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("schema") != _STEP_CATEGORY_CLUSTER_RECEIPT_SCHEMA
            or receipt.get("dense_cache_key") != str(dense_cache_key)
            or receipt.get("dense_payload_sha256") != str(dense_payload_sha256)
            or receipt.get("payload_sha256") != payload_sha256
        ):
            return None
        with np.load(io.BytesIO(payload_bytes), allow_pickle=False) as data:
            assignment = np.asarray(data["assignment"], dtype=np.int64)
            spectrum = np.asarray(data["spectrum"], dtype=np.float64)
        derivation = dict(receipt.get("derivation", {}))
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
        zipfile_BadZipFile,
    ):
        return None
    if assignment.ndim != 1 or spectrum.shape != assignment.shape:
        return None
    derivation["assignment"] = assignment
    derivation["spectrum"] = spectrum
    derivation["cache_hit"] = True
    derivation["payload_sha256"] = payload_sha256
    return derivation


def _store_step_category_cluster_cache(
    *,
    payload_path: Path,
    receipt_path: Path,
    dense_cache_key: str,
    dense_payload_sha256: str,
    derivation: Mapping[str, Any],
) -> str:
    import hashlib
    import io

    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        assignment=np.asarray(derivation["assignment"], dtype=np.int64),
        spectrum=np.asarray(derivation["spectrum"], dtype=np.float64),
    )
    payload_bytes = stream.getvalue()
    payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{payload_path.name}.",
        suffix=".npz",
        dir=payload_path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, payload_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    receipt = {
        "schema": _STEP_CATEGORY_CLUSTER_RECEIPT_SCHEMA,
        "dense_cache_key": str(dense_cache_key),
        "dense_payload_sha256": str(dense_payload_sha256),
        "payload_sha256": payload_sha256,
        "derivation": {
            key: value
            for key, value in derivation.items()
            if key not in {"assignment", "spectrum"}
        },
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{receipt_path.name}.",
        suffix=".json",
        dir=receipt_path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({**receipt, "effective_settings": effective_settings()}, handle, sort_keys=True, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, receipt_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return payload_sha256


def _dense_fit_cache_optin_fields(provider: Any) -> dict[str, str]:
    """Effective numerical and solver settings of the dense fit cache key."""
    from mechanistic_systems.measurement.native_irg_providers import (
        frozen_category_fit_optin_fields,
    )
    return dict(frozen_category_fit_optin_fields(provider))


def _dense_fit_expected_request(
    provider: Any,
    *,
    dense_mask: HardMask,
    calibration_by_class: Mapping[int, Sequence[str]],
    context_axis_hash: str,
    bank_mode: CalibrationBankMode,
) -> dict[str, Any]:
    """Current fit identity and its profile-independent cache request."""
    request = provider.frozen_category_fit_request(
        dense_mask,
        dense_mask,
        training_context_ids_by_class=calibration_by_class,
        training_axis_source_identity=str(context_axis_hash),
        mode=bank_mode,
        category_fit_chunk=16,
    )
    return {
        "request_hash": str(request["request_hash"]),
        "fit_semantic_request_hash": str(request["fit_semantic_request_hash"]),
        "optin_fields": dict(request["optin_fields"]),
    }


def _dense_fit_semantic_request_receipt(
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Record the fit request and its effective numerical settings."""
    return {
        "schema": _DENSE_FIT_SEMANTIC_REQUEST_SCHEMA,
        "fit_semantic_request_hash": str(expected["fit_semantic_request_hash"]),
        "optin_fields": dict(expected["optin_fields"]),
    }


def _dense_fit_request_qualified_cache_key(
    dense_fit_cache_key: str, fit_semantic_request_hash: str
) -> str:
    """A cache entry qualified by its profile-independent fit request."""

    return content_hash(
        {
            "schema": "dense_category_fit_bundle_semantic_request_cache_key_v1",
            "execution_numerics_cache_version": DETERMINISTIC_EXECUTION_CACHE_VERSION,
            "dense_fit_cache_key": str(dense_fit_cache_key),
            "fit_semantic_request_hash": str(fit_semantic_request_hash),
        },
        prefix="irg-dense-category-fit-cache-",
    )


def _read_dense_category_fit_cache_receipt(receipt_path: Path) -> Mapping[str, Any]:
    """The JSON receipt of an entry that just loaded (read under its lock)."""

    try:
        receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return receipt if isinstance(receipt, Mapping) else {}


def _dense_fit_cached_entry_admission(
    *,
    receipt: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Reuse only a cache entry with the current fit-semantic request."""
    recorded = receipt.get(_DENSE_FIT_SEMANTIC_REQUEST_RECEIPT_KEY)
    cached_semantic = (
        recorded.get("fit_semantic_request_hash")
        if isinstance(recorded, Mapping)
        else None
    )
    if (
        isinstance(recorded, Mapping)
        and recorded.get("schema") == _DENSE_FIT_SEMANTIC_REQUEST_SCHEMA
        and str(cached_semantic) == str(expected["fit_semantic_request_hash"])
    ):
        return {"admitted": True, "basis": "fit_semantic_request_hash"}
    return {
        "admitted": False,
        "reason": "fit_semantic_request_differs",
        "cached_fit_semantic_request_hash": (
            None if cached_semantic is None else str(cached_semantic)
        ),
    }


def _step_category_structure_setup(
    *,
    enabled: bool,
    provider: Any,
    dense_mask: HardMask,
    training_populations: Mapping[str, Sequence[str]],
    target_class_id: int,
    batch_factory: Any,
    model_state_hash: str,
    context_axis_hash: str,
    forward_tier: int,
    cache_dir: Path | None,
    oracle_stratum_class_ids: Sequence[int],
    oracle_minimum_per_stratum: int,
    calibration_bank_mode: CalibrationBankMode = (
        CalibrationBankMode.CLAIM_BEARING_1000
    ),
    progress_callback: Callable[..., None] | None = None,
) -> Mapping[str, Any] | None:
    """Walk-setup leg: train-only dense frame + clusters + decay.

    The dense-side per-category calibrations and moment matrices are fitted on
    the optimization/training axis only.  This setup function must not touch the
    terminal validation axis or terminal batch factory; its output is the frozen
    frame consumed by per-step observation-only diagnostics.
    """

    if not enabled:
        return None
    from mechanistic_systems.measurement.native_irg_providers import (
        STEP_ANCHOR_INTERVAL,
        StepCategoryStructureConfig,
        derive_category_clusters,
        step_category_lambda_derivation,
    )

    calibration_by_class = _terminal_category_populations_by_class(
        training_populations,
        target_class_id=int(target_class_id),
    )
    calibration_membership_hash = content_hash(
        {
            "schema": "step_category_train_calibration_membership_v1",
            "populations": {
                str(class_id): tuple(sorted(rows))
                for class_id, rows in calibration_by_class.items()
            },
        }
    )
    calibration_contract = str(provider.calibration_options["optimization_contract"])
    resolved_bank_mode = CalibrationBankMode(calibration_bank_mode)
    # The key binds every effective numerical setting before fitting.
    dense_fit_optin_fields = _dense_fit_cache_optin_fields(provider)
    dense_fit_cache_key = content_hash(
        {
            "schema": "dense_category_fit_bundle_cache_key_v1",
            "execution_numerics_cache_version": DETERMINISTIC_EXECUTION_CACHE_VERSION,
            "model_state_hash": str(model_state_hash),
            "dense_mask_hash": dense_mask.mask_hash,
            "context_axis_hash": str(context_axis_hash),
            "calibration_membership_hash": calibration_membership_hash,
            "full_readout_projection_hash": (
                provider.terminal_full_readout_reference_hash
            ),
            "calibration_options": dict(provider.calibration_options),
            "bank_mode": resolved_bank_mode.value,
            "category_fit_chunk": 16,
            **dense_fit_optin_fields,
        },
        prefix="irg-dense-category-fit-cache-",
    )
    dense_fit_cache_hit = False
    dense_fit_reference_bound = False
    loaded_gate_summary: dict[str, Any] = {}
    dense_fit_request_mismatch: dict[str, Any] | None = None
    dense_fit: FrozenCategoryCalibrationFitBundle
    reference_dir = os.environ.get("IRG_TERMINAL_DENSE_REFERENCE")
    reference_sha256 = os.environ.get("IRG_TERMINAL_DENSE_REFERENCE_SHA256")
    reference = None
    from . import category_memo

    memo = category_memo.active() if reference_dir is None and reference_sha256 is None else None
    memo_key = None if memo is None else _category_dense_bank_key(
        provider, dense_mask=dense_mask, calibration_by_class=calibration_by_class,
        calibration_context_axis_hash=str(context_axis_hash),
        calibration_bank_mode=resolved_bank_mode, category_fit_chunk=16,
    )
    memo_entry = None if memo_key is None else memo.dense_banks.get(memo_key)
    if reference_dir is not None or reference_sha256 is not None:
        try:
            if not reference_dir or not reference_sha256:
                raise ValueError("terminal dense reference needs both directory and manifest sha256")
            reference = _load_terminal_dense_reference(reference_dir, reference_sha256)
            key = reference["manifest"]["binding_key"]
            settings = key["terminal_settings"]
            expected_order = tuple(
                context_id
                for class_id in sorted(calibration_by_class)
                for context_id in calibration_by_class[class_id]
            )
            fit = reference["dense_fit"]
            semantic_hash = _terminal_dense_fit_semantic_hash(
                provider, dense_mask, fit, str(context_axis_hash), 16
            )
            if (
                key["dense_checkpoint_hash"] != str(model_state_hash)
                or fit.mask_hash != dense_mask.mask_hash
                or fit.training_context_ids != expected_order
                or fit.model_state_hash != provider._frozen_category_model_state_hash()
                or settings["calibration_axis_hash"] != str(context_axis_hash)
                or settings["calibration_membership_hash"] != fit.bank.identity.training_membership_identity
                or settings["calibration_mode"] != resolved_bank_mode.value
                or settings["calibration_contract"] != calibration_contract
                or settings["fit_options"] != dense_fit_optin_fields
                or settings["category_fit_chunk"] != 16
                or settings.get("runtime_numerics_profile") != _terminal_dense_numerics_profile(provider)
                or settings["full_readout_projection_hash"] != provider.terminal_full_readout_reference_hash
                or settings["fit_semantic_request_hash"] != semantic_hash
            ):
                raise ValueError("terminal dense reference step setup binding differs")
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            raise ValueError(f"terminal dense reference refused: {error}") from error
    if reference is not None:
        dense_fit = provider.install_frozen_category_calibration_fit_bundle(
            reference["dense_fit"]
        )
        gate_summary = reference["manifest"]["dense_fit_gate_summary"]
        installer = getattr(provider, "install_frozen_category_fit_gate_summary", None)
        if gate_summary is not None and callable(installer):
            installer(dense_fit.request_hash, gate_summary)
        dense_fit_reference_bound = True
    elif memo_entry is not None:
        dense_fit = provider.install_frozen_category_calibration_fit_bundle(memo_entry["dense_fit"])
        installer = getattr(provider, "install_frozen_category_fit_gate_summary", None)
        if memo_entry["gate_summary"] is not None and callable(installer):
            installer(dense_fit.request_hash, memo_entry["gate_summary"])
        memo.note(product="dense_bank", source="memo", key=memo_key)
    elif cache_dir is None:
        dense_fit = provider.fit_frozen_category_calibration_bank(
            dense_mask,
            dense_mask,
            training_context_ids_by_class=calibration_by_class,
            training_batch_factory=batch_factory,
            training_axis_source_identity=str(context_axis_hash),
            mode=resolved_bank_mode,
            category_fit_chunk=16,
        )
    else:
        import fcntl

        fit_cache_root = Path(cache_dir)
        fit_cache_root.mkdir(parents=True, exist_ok=True)
        fit_payload_path = fit_cache_root / f"{dense_fit_cache_key}.fit_bundle.npz"
        fit_receipt_path = fit_cache_root / f"{dense_fit_cache_key}.fit_bundle.json"
        fit_lock_path = fit_cache_root / f"{dense_fit_cache_key}.fit_bundle.lock"
        with fit_lock_path.open("a+b") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            try:
                loaded_fit = _load_dense_category_fit_bundle_cache(
                    payload_path=fit_payload_path,
                    receipt_path=fit_receipt_path,
                    cache_key=dense_fit_cache_key,
                    fit_gate_summary_out=loaded_gate_summary,
                )
                if loaded_fit is not None:
                    expected_training_order = tuple(
                        context_id
                        for class_id in sorted(calibration_by_class)
                        for context_id in calibration_by_class[class_id]
                    )
                    if (
                        loaded_fit.mask_hash != dense_mask.mask_hash
                        or loaded_fit.dense_reference_mask_hash != dense_mask.mask_hash
                        or loaded_fit.training_context_ids != expected_training_order
                    ):
                        raise RuntimeError(
                            "dense category fit cache passed bytes but failed "
                            "runtime lineage revalidation"
                        )
                # Reuse across execution profiles requires the same fit request.
                expected_request = _dense_fit_expected_request(
                    provider,
                    dense_mask=dense_mask,
                    calibration_by_class=calibration_by_class,
                    context_axis_hash=str(context_axis_hash),
                    bank_mode=resolved_bank_mode,
                )
                store_key = dense_fit_cache_key
                store_payload_path = fit_payload_path
                store_receipt_path = fit_receipt_path
                admission = (
                    None
                    if loaded_fit is None
                    else _dense_fit_cached_entry_admission(
                        receipt=_read_dense_category_fit_cache_receipt(
                            fit_receipt_path
                        ),
                        expected=expected_request,
                    )
                )
                if admission is not None and not admission["admitted"]:
                    store_key = _dense_fit_request_qualified_cache_key(
                        dense_fit_cache_key,
                        expected_request["fit_semantic_request_hash"],
                    )
                    store_payload_path = fit_cache_root / f"{store_key}.fit_bundle.npz"
                    store_receipt_path = fit_cache_root / f"{store_key}.fit_bundle.json"
                    dense_fit_request_mismatch = {
                        "schema": "dense_category_fit_cache_request_mismatch_v2",
                        "reason": str(admission["reason"]),
                        "cached_request_hash": str(loaded_fit.request_hash),
                        "expected_request_hash": expected_request["request_hash"],
                        "cached_fit_semantic_request_hash": admission.get(
                            "cached_fit_semantic_request_hash"
                        ),
                        "expected_fit_semantic_request_hash": expected_request[
                            "fit_semantic_request_hash"
                        ],
                        "request_qualified_cache_key": store_key,
                        "request_qualified_cache_hit": False,
                    }
                    loaded_gate_summary.clear()
                    loaded_fit = _load_dense_category_fit_bundle_cache(
                        payload_path=store_payload_path,
                        receipt_path=store_receipt_path,
                        cache_key=store_key,
                        fit_gate_summary_out=loaded_gate_summary,
                    )
                    if loaded_fit is not None:
                        sibling_admission = _dense_fit_cached_entry_admission(
                            receipt=_read_dense_category_fit_cache_receipt(
                                store_receipt_path
                            ),
                                expected=expected_request,
                        )
                        if (
                            sibling_admission.get("basis")
                            != "fit_semantic_request_hash"
                            or loaded_fit.mask_hash != dense_mask.mask_hash
                            or loaded_fit.dense_reference_mask_hash
                            != dense_mask.mask_hash
                        ):
                            raise RuntimeError(
                                "request-qualified dense category fit cache entry "
                                "binds another request or lineage"
                            )
                    dense_fit_request_mismatch["request_qualified_cache_hit"] = (
                        loaded_fit is not None
                    )
                if loaded_fit is not None:
                    dense_fit = provider.install_frozen_category_calibration_fit_bundle(
                        loaded_fit
                    )
                    if loaded_gate_summary:
                        installer = getattr(
                            provider, "install_frozen_category_fit_gate_summary", None
                        )
                        if callable(installer):
                            installer(dense_fit.request_hash, loaded_gate_summary)
                    dense_fit_cache_hit = True
                else:
                    dense_fit = provider.fit_frozen_category_calibration_bank(
                        dense_mask,
                        dense_mask,
                        training_context_ids_by_class=calibration_by_class,
                        training_batch_factory=batch_factory,
                        training_axis_source_identity=str(context_axis_hash),
                        mode=resolved_bank_mode,
                        category_fit_chunk=16,
                    )
                    if (
                        str(dense_fit.request_hash)
                        != expected_request["request_hash"]
                    ):
                        raise RuntimeError(
                            "dense category fit bound a request other than the "
                            "one its cache entry was keyed for"
                        )
                    _store_dense_category_fit_bundle_cache(
                        payload_path=store_payload_path,
                        receipt_path=store_receipt_path,
                        cache_key=store_key,
                        bundle=dense_fit,
                        fit_semantic_request=_dense_fit_semantic_request_receipt(
                            expected_request
                        ),
                        fit_gate_summary=_terminal_bank_gate_summary(provider, dense_fit),
                    )
            finally:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
    if memo_key is not None and memo_entry is None:
        memo.dense_banks[memo_key] = {
            "dense_fit": dense_fit,
            "gate_summary": _terminal_bank_gate_summary(provider, dense_fit),
            "products": None,
        }
        memo.note(product="dense_bank", source="step_setup", key=memo_key)
    dense_cache_key = _terminal_category_structure_cache_key(
        model_state_hash=str(model_state_hash),
        dense_mask_hash=dense_mask.mask_hash,
        context_axis_hash=str(context_axis_hash),
        census_membership_hash=calibration_membership_hash,
        full_readout_projection_hash=(provider.terminal_full_readout_reference_hash),
        calibration_contract=calibration_contract,
        category_fit_chunk=16,
        forward_tier=int(forward_tier),
        calibration_context_membership_hash=calibration_membership_hash,
        frozen_bank_hash=dense_fit.bank.content_hash,
    )
    ordinary_scale, ordinary_offset, ordinary_center = dense_fit.bank.affine_arrays(
        "ordinary_ovr"
    )
    substituted_scale, substituted_offset, substituted_center = (
        dense_fit.bank.affine_arrays("substituted_target_affine_dense_exclusion")
    )
    dense_payload_sha256 = dense_fit.content_hash
    dense_arrays = {
        "stratum_order": dense_fit.bank.class_ids,
        "counts": dense_fit.training_counts,
        "moment_mean": dense_fit.training_raw_logit_mean,
        "moment_m2": dense_fit.training_raw_logit_m2,
        "calibration_scale": ordinary_scale,
        "calibration_offset": ordinary_offset,
        "calibration_input_center": ordinary_center,
        "calibration_converged": np.asarray(
            [
                state.converged
                for state in dense_fit.bank.states
                if state.functional.value == "ordinary_ovr"
            ],
            dtype=bool,
        ),
        "substituted_calibration_scale": substituted_scale,
        "substituted_calibration_offset": substituted_offset,
        "substituted_calibration_input_center": substituted_center,
        "substituted_calibration_converged": np.asarray(
            [
                state.converged
                for state in dense_fit.bank.states
                if state.functional.value == "substituted_target_affine_dense_exclusion"
            ],
            dtype=bool,
        ),
    }
    dense_scale = np.asarray(dense_arrays["calibration_scale"], dtype=np.float64)
    dense_offset = np.asarray(dense_arrays["calibration_offset"], dtype=np.float64)
    fitted_mask = (
        np.isfinite(dense_scale)
        & np.isfinite(dense_offset)
        & np.isfinite(
            np.asarray(dense_arrays["calibration_input_center"], dtype=np.float64)
        )
        & np.asarray(dense_arrays["calibration_converged"], dtype=bool)
    )
    substituted_dense_scale = np.asarray(
        dense_arrays["substituted_calibration_scale"], dtype=np.float64
    )
    substituted_dense_offset = np.asarray(
        dense_arrays["substituted_calibration_offset"], dtype=np.float64
    )
    substituted_dense_converged = np.asarray(
        dense_arrays["substituted_calibration_converged"], dtype=bool
    )
    substituted_dense_fitted_mask = (
        np.isfinite(substituted_dense_scale)
        & np.isfinite(substituted_dense_offset)
        & substituted_dense_converged
    )
    derivation: dict[str, Any] | None = None
    cluster_cache_hit = False
    if cache_dir is not None:
        import fcntl

        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        payload_path = cache_root / f"{dense_cache_key}.clusters.npz"
        receipt_path = cache_root / f"{dense_cache_key}.clusters.json"
        lock_path = cache_root / f"{dense_cache_key}.clusters.lock"
        with lock_path.open("a+b") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            try:
                loaded = _load_step_category_cluster_cache(
                    payload_path=payload_path,
                    receipt_path=receipt_path,
                    dense_cache_key=dense_cache_key,
                    dense_payload_sha256=dense_payload_sha256,
                )
                if loaded is not None:
                    derivation = loaded
                    cluster_cache_hit = True
                else:
                    derivation = dict(
                        derive_category_clusters(
                            dense_arrays["moment_mean"],
                            fitted_mask=fitted_mask,
                        )
                    )
                    derivation["payload_sha256"] = _store_step_category_cluster_cache(
                        payload_path=payload_path,
                        receipt_path=receipt_path,
                        dense_cache_key=dense_cache_key,
                        dense_payload_sha256=dense_payload_sha256,
                        derivation=derivation,
                    )
            finally:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
    else:
        derivation = dict(
            derive_category_clusters(
                dense_arrays["moment_mean"],
                fitted_mask=fitted_mask,
            )
        )
    assert derivation is not None
    lambda_receipt = step_category_lambda_derivation(
        dense_census_counts=tuple(int(value) for value in dense_arrays["counts"]),
        oracle_minimum_per_stratum=int(oracle_minimum_per_stratum),
        measurement_role_streams=2,
    )
    config = StepCategoryStructureConfig(
        target_class_id=int(target_class_id),
        stratum_class_ids=tuple(
            int(value) for value in sorted(oracle_stratum_class_ids)
        ),
        dense_scale=dense_scale,
        dense_offset=dense_offset,
        dense_input_center=np.asarray(
            dense_arrays["calibration_input_center"], dtype=np.float64
        ),
        dense_fitted_mask=fitted_mask,
        substituted_dense_scale=substituted_dense_scale,
        substituted_dense_offset=substituted_dense_offset,
        substituted_dense_input_center=np.asarray(substituted_center, dtype=np.float64),
        substituted_dense_fitted_mask=substituted_dense_fitted_mask,
        dense_moment_mean=np.asarray(dense_arrays["moment_mean"], dtype=np.float64),
        dense_moment_m2=np.asarray(dense_arrays["moment_m2"], dtype=np.float64),
        dense_counts=np.asarray(dense_arrays["counts"], dtype=np.int64),
        dense_stratum_class_ids=np.asarray(
            dense_arrays["stratum_order"], dtype=np.int64
        ),
        cluster_assignment=np.asarray(derivation["assignment"], dtype=np.int64),
        cluster_spectrum=np.asarray(derivation["spectrum"], dtype=np.float64),
        cluster_receipt={
            key: value
            for key, value in derivation.items()
            if key not in {"assignment", "spectrum"}
        },
        lambda_receipt=lambda_receipt,
        anchor_interval=int(STEP_ANCHOR_INTERVAL),
    )
    provider.enable_step_category_structure(config)
    return {
        "dense_cache_key": dense_cache_key,
        "dense_payload_sha256": dense_payload_sha256,
        "dense_frozen_bank_hash": dense_fit.bank.content_hash,
        "dense_fit_bundle_hash": dense_fit.content_hash,
        "dense_fit_request_hash": dense_fit.request_hash,
        "dense_cache_hit": bool(dense_fit_cache_hit),
        **({} if not dense_fit_reference_bound else {
            "dense_fit_source": "campaign_reference",
            "dense_reference_manifest_sha256": reference["manifest_sha256"],
        }),
        **({} if memo_entry is None else {"dense_fit_source": "category_memo"}),
        "dense_fit_cache_key": dense_fit_cache_key,
        # Present only when a shared entry was refused (another
        # fit-semantic request: an opt-in solver setting, never a runtime
        # profile alone).
        **(
            {}
            if dense_fit_request_mismatch is None
            else {"dense_fit_cache_request_mismatch": dense_fit_request_mismatch}
        ),
        "census_membership_hash": str(calibration_membership_hash),
        "calibration_context_membership_hash": str(calibration_membership_hash),
        "calibration_sample_role": "train",
        "cluster_cache_hit": bool(cluster_cache_hit),
        "cluster_receipt": {
            key: value
            for key, value in derivation.items()
            if key not in {"assignment", "spectrum"}
        },
        "lambda_receipt": dict(lambda_receipt),
        "anchor_interval": int(STEP_ANCHOR_INTERVAL),
    }


def _step_category_structure_finalize(
    *,
    enabled: bool,
    provider: Any,
    setup_receipt: Mapping[str, Any] | None,
    accepted_trajectory: AcceptedV0Trajectory | None = None,
    frozen_payload: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Build the accepted-trajectory ``step_category_structure_v4`` sidecar."""

    from mechanistic_systems.measurement.native_irg_providers import (
        STEP_CATEGORY_STRUCTURE_SCHEMA_V4,
    )

    if not enabled:
        return (
            {
                "schema": STEP_CATEGORY_STRUCTURE_SCHEMA_V4,
                "enabled": False,
                "accepted_trajectory_only": True,
            },
            {},
        )
    import hashlib

    if setup_receipt is None:
        raise RuntimeError(
            "step category structure finalize requires its setup receipt"
        )
    if accepted_trajectory is None:
        raise RuntimeError(
            "accepted step matrices require the immutable accepted trajectory"
        )
    payload = dict(
        provider.finalize_step_category_structure() if frozen_payload is None else frozen_payload
    )
    # Slab residency can change in terminal calibration. Keep this telemetry
    # in the receipt, so archive assembly can start at mask lock with the
    # immutable accepted matrices and produce the same bytes in either mode.
    slab_bytes = int(payload.pop("readout_slab_resident_bytes", 0))
    slab_host_primary = json.loads(str(payload.pop("readout_slab_host_primary", "[]")))
    if not isinstance(accepted_trajectory, AcceptedV0Trajectory):
        raise TypeError("accepted step matrices require a typed AcceptedV0Trajectory")
    entries = accepted_trajectory.entries
    exact_bindings = (
        tuple(int(value) for value in payload["step_trajectory_ordinal"])
        == tuple(entry.trajectory_index for entry in entries)
        and tuple(str(value) for value in payload["step_state_hash"])
        == tuple(entry.provider_state_hash for entry in entries)
        and tuple(str(value) for value in payload["step_query_id"])
        == tuple(entry.query_id for entry in entries)
        and tuple(str(value) for value in payload["step_layer_id"])
        == tuple(entry.layer_id for entry in entries)
        and tuple(str(value) for value in payload["step_partition_id"])
        == tuple(entry.partition_id for entry in entries)
        and tuple(int(value) for value in payload["step_proposal_index"])
        == tuple(entry.proposal_index for entry in entries)
        and tuple(str(value) for value in payload["step_parent_mask_hash"])
        == tuple(entry.parent_mask_hash for entry in entries)
        and tuple(str(value) for value in payload["step_candidate_mask_hash"])
        == tuple(entry.resulting_mask_hash for entry in entries)
        and tuple(str(value) for value in payload["step_decision_receipt_hash"])
        == tuple(entry.decision_receipt_hash for entry in entries)
        and tuple(str(value) for value in payload["step_action"])
        == ("accept",) * len(entries)
    )
    if not exact_bindings:
        raise RuntimeError(
            "accepted step matrices differ from the immutable commit trajectory"
        )
    if int(payload["matrix_evaluation_count"]) != len(entries):
        raise RuntimeError("accepted trajectory count differs from matrix evaluations")
    if (
        int(payload["accepted_proposal_count"])
        != accepted_trajectory.receipt.accepted_count
        or int(payload["discarded_proposal_count"])
        != accepted_trajectory.receipt.rejected_count
        + accepted_trajectory.receipt.unresolved_count
        or int(payload["resolved_proposal_count"])
        != accepted_trajectory.receipt.total_decision_count
    ):
        raise RuntimeError(
            "step proposal resolution counts differ from the accepted trajectory"
        )
    # With spilling active the nine large families are already .npy files;
    # the archive is assembled next to them and only its PATH travels onward,
    # so neither the planes nor the archive is held in memory through receipt
    # finalization. The staged file is moved into the arm directory later.
    member_paths = dict(getattr(provider, "step_category_spill_member_paths", {}))
    staging_directory = ""
    if member_paths:
        staging_directory = os.path.dirname(next(iter(member_paths.values())))
    payload_bytes, payload_sha256 = _write_step_category_sidecar(
        (
            os.path.join(staging_directory, STEP_CATEGORY_SIDECAR_FILENAME)
            if staging_directory
            else ""
        ),
        payload,
        member_paths=member_paths,
    )
    seconds = np.asarray(payload["step_flush_seconds"], dtype=np.float64)
    step_count = int(payload["step_count"])
    registered_scale = np.asarray(payload["target_registered_scale"], dtype=np.float64)
    registered_offset = np.asarray(
        payload["target_registered_offset"], dtype=np.float64
    )
    agreement: dict[str, Any] | None = None
    if step_count > 0:
        raw_scale = np.asarray(payload["target_tracker_scale_raw"], dtype=np.float64)
        raw_offset = np.asarray(payload["target_tracker_offset_raw"], dtype=np.float64)
        shrunk_scale = np.asarray(
            payload["target_tracker_scale_shrunk"], dtype=np.float64
        )
        shrunk_offset = np.asarray(
            payload["target_tracker_offset_shrunk"], dtype=np.float64
        )
        agreement = {
            "scale_max_abs_diff_raw": float(
                np.max(np.abs(raw_scale - registered_scale))
            ),
            "scale_max_abs_diff_shrunk": float(
                np.max(np.abs(shrunk_scale - registered_scale))
            ),
            "offset_max_abs_diff_raw": float(
                np.max(np.abs(raw_offset - registered_offset))
            ),
            "offset_max_abs_diff_shrunk": float(
                np.max(np.abs(shrunk_offset - registered_offset))
            ),
            "final_step_scale_diff_shrunk": float(
                shrunk_scale[-1] - registered_scale[-1]
            ),
            "final_step_offset_diff_shrunk": float(
                shrunk_offset[-1] - registered_offset[-1]
            ),
        }
    block = {
        "schema": STEP_CATEGORY_STRUCTURE_SCHEMA_V4,
        "enabled": True,
        "mode": "on",
        "registered_scope": "v0_accepted_mask_trajectory_only",
        "accepted_trajectory_only": bool(payload["accepted_trajectory_only"]),
        "accepted_proposal_count": int(payload["accepted_proposal_count"]),
        "discarded_proposal_count": int(payload["discarded_proposal_count"]),
        "matrix_evaluation_count": int(payload["matrix_evaluation_count"]),
        "accepted_trajectory_receipt_hash": (accepted_trajectory.receipt.receipt_hash),
        "accepted_trajectory_exact_binding_verified": True,
        "decision_projection": {
            "kind": "full_width_device_readout_projection",
            "hash": str(payload["decision_projection_hash"]),
        },
        "sidecars": {
            "steps": {
                "filename": STEP_CATEGORY_SIDECAR_FILENAME,
                "sha256": payload_sha256,
                # A streamed sidecar supplies its path; report the file size.
                "bytes": (
                    len(payload_bytes)
                    if isinstance(payload_bytes, (bytes, bytearray))
                    else int(os.path.getsize(payload_bytes))
                ),
            }
        },
        "step_count": step_count,
        "anchor_interval": int(setup_receipt["anchor_interval"]),
        "anchor_count": int(np.asarray(payload["anchor_step_indices"]).size),
        "decay_lambda": dict(setup_receipt["lambda_receipt"]),
        "clusters": {
            "k_clusters": int(payload["k_clusters"]),
            "cache_hit": bool(setup_receipt["cluster_cache_hit"]),
            **{
                key: value
                for key, value in dict(setup_receipt["cluster_receipt"]).items()
                if key != "cache_hit"
            },
        },
        "dense_reference": {
            "cache_key": str(setup_receipt["dense_cache_key"]),
            "payload_sha256": str(setup_receipt["dense_payload_sha256"]),
            "cache_hit": bool(setup_receipt["dense_cache_hit"]),
            "census_membership_hash": str(setup_receipt["census_membership_hash"]),
        },
        "per_step_overhead_seconds": {
            "total": float(seconds.sum()),
            "mean": float(seconds.mean()) if seconds.size else 0.0,
            "max": float(seconds.max()) if seconds.size else 0.0,
        },
        "target_column_agreement": agreement,
        # payload_bytes is the archive itself on the in-memory path and its
        # PATH on the streamed one, so the size is taken from the file there.
        "bytes": (
            len(payload_bytes)
            if isinstance(payload_bytes, (bytes, bytearray))
            else int(os.path.getsize(payload_bytes))
        ),
        # Disclose which storage path produced this sidecar, and why the
        # streamed one was not used when it was declared but unavailable.
        **dict(getattr(provider, "step_category_spill_telemetry", {})),
        "readout_slab_resident_bytes": slab_bytes,
        "readout_slab_host_primary": slab_host_primary,
        "readout_slab_telemetry_location": "receipt_after_terminal_evaluation_v1",
        "tracker_constants": {
            "weight_floor": float(payload["tracker_weight_floor"]),
            "delta_clamp": float(payload["tracker_delta_clamp"]),
            "scale_floor": float(payload["tracker_scale_floor"]),
        },
    }
    return block, {STEP_CATEGORY_SIDECAR_FILENAME: payload_bytes}




@contextmanager
def _step_archive_at_mask_lock(
    *, enabled: bool, destination: Path, provider: Any,
    setup_receipt: Mapping[str, Any] | None,
    accepted_trajectory: AcceptedV0Trajectory | None,
):
    """Assemble one immutable step archive on the existing run product writer."""

    writer = None
    assembled = None
    if enabled:
        # Close the step store and snapshot the accepted matrices on the walk
        # thread. Only archive assembly runs alongside terminal calibration.
        payload = provider.finalize_step_category_structure()
        writer = RunProductWriter(destination, persist_staged=_persist_sidecar)
        def assemble():
            nonlocal payload, assembled
            try:
                assembled = _step_category_structure_finalize(
                    enabled=True, provider=provider, setup_receipt=setup_receipt,
                    accepted_trajectory=accepted_trajectory, frozen_payload=payload,
                )
                return assembled
            finally:
                payload = None
        writer.defer(assemble)
        writer.start()
    try:
        yield writer
    finally:
        if writer is not None and not writer.completion_transferred:
            try:
                if sys.exc_info()[0] is None:
                    writer.join()
                else:
                    writer.cancel()
            finally:
                # A terminal failure may finish assembly but never move the
                # staged archive. Remove that archive so a retry can reuse
                # its spill directory as well as its still-absent run path.
                if assembled is not None:
                    for value in assembled[1].values():
                        if isinstance(value, (str, os.PathLike)):
                            Path(value).unlink(missing_ok=True)


def _step_category_structure_runtime_telemetry(block: dict[str, Any], provider: Any) -> None:
    """Sample the receipt's slab fields at the same post-terminal point in both modes."""

    if block.get("enabled"):
        block.update(provider.step_category_readout_telemetry())


def _pilot_receipt_content_hash(payload: Mapping[str, Any]) -> str:
    """Hash the receipt using its recorded choice without changing the payload."""

    return content_hash(payload, prefix="irg-resnet18-pilot-")


def _arm_run_products(
    *,
    final_mask: Any,
    terminal_category_structure_sidecars: Mapping[str, Any],
    step_category_structure_block: Mapping[str, Any],
    step_category_structure_sidecars: Mapping[str, Any],
    step_context_delta_sidecars: Mapping[str, Any],
    run_product_sidecars: Mapping[str, bytes],
) -> list[tuple[str, Any, str | None]]:
    """The arm's products in their fixed write order.

    Each entry is ``(file name, bytes or staged path, known sha256)``.  The
    step archive's digest is the one its streaming writer computed and bound
    into the step block (``sidecars.steps.sha256``), so it is not re-read.
    """

    products: list[tuple[str, Any, str | None]] = [
        ("final_mask.packbits", final_mask.packed_bits, None)
    ]
    for name, value in terminal_category_structure_sidecars.items():
        products.append((str(name), value, None))
    step_entry = (
        (step_category_structure_block.get("sidecars") or {}).get("steps")
        if isinstance(step_category_structure_block, Mapping)
        else None
    )
    for name, value in step_category_structure_sidecars.items():
        known = None
        if (
            isinstance(step_entry, Mapping)
            and step_entry.get("filename") == name
            and not isinstance(value, (bytes, bytearray))
        ):
            known = str(step_entry.get("sha256"))
        products.append((str(name), value, known))
    for name, value in step_context_delta_sidecars.items():
        products.append((str(name), value, None))
    for name, value in run_product_sidecars.items():
        products.append((str(name), value, None))
    return products


def _persist_sidecar(destination: Any, value: Any) -> None:
    """Write a sidecar that is either raw bytes or an already-staged file.

    The streamed step-matrix path assembles its archive beside the spilled
    ``.npy`` members and passes the PATH onward, so nothing holds a multi-GiB
    archive in memory through receipt finalization. Moving beats copying when
    the staging area and the arm directory share a filesystem, and falls back
    to a streaming copy when they do not.
    """

    import shutil

    if isinstance(value, (bytes, bytearray)):
        destination.write_bytes(value)
        return
    source = str(value)
    if not os.path.exists(source):
        raise RuntimeError(f"staged sidecar {source!r} is missing")
    shutil.move(source, str(destination))


def _bind_step_context_delta_sink(accumulator: Any, provider: Any) -> Any:
    """Bind the sidecar sink so it stores SOURCE identities, not draw IDs.

    The panel carries ``draw_id`` values, which hash in the draw index and so
    are unique per row by construction: they cannot compress, and no two
    decisions ever share one, which is exactly why draw-id-only sidecars
    cannot support any paired cross-decision estimator. ``_iid_draws`` already
    holds the draw -> source mapping in memory, so recording the source costs
    one dict lookup per context and makes those estimators possible.

    Unknown draws fall back to the draw ID itself, matching the convention
    ``NativeTorchCalibratedBCEOracleProvider._source_context_ids`` already
    uses, rather than dropping the row or raising.
    """

    if accumulator is None:
        return None
    draws = getattr(provider, "_iid_draws", None)
    if not isinstance(draws, dict):
        # No stream registry (synthetic or replayed providers): keep the
        # previous draw-identity behaviour rather than mislabel the column.
        return accumulator.append_decision

    def sink(*, context_ids: Sequence[str], **kwargs: Any) -> None:
        resolved = []
        for value in context_ids:
            key = str(value)
            draw = draws.get(key)
            resolved.append(key if draw is None else str(draw.source_context_id))
        accumulator.append_decision(
            context_ids=context_ids,
            source_context_ids=tuple(resolved),
            **kwargs,
        )

    return sink


def _step_context_delta_finalize(
    accumulator: StepContextDeltaAccumulator | None,
) -> tuple[Mapping[str, Any] | None, dict[str, bytes]]:
    """Flush the bounded per-context rows once after the v0 walk."""

    if accumulator is None:
        return None, {}
    import hashlib

    payload_bytes = accumulator.sidecar_bytes()
    payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    block = {
        "schema": STEP_CONTEXT_DELTA_SIDECAR_SCHEMA,
        "enabled": True,
        "write_strategy": "in_memory_batch_at_walk_finalize",
        "sidecar": {
            "filename": STEP_CONTEXT_DELTA_SIDECAR_FILENAME,
            "sha256": payload_sha256,
            "bytes": len(payload_bytes),
        },
        "row_count": int(accumulator.row_count),
        "truncated_row_count": int(accumulator.truncated_row_count),
        "truncated": bool(accumulator.truncated_row_count),
        "unavailable_row_count": int(accumulator.unavailable_row_count),
        "cap_bytes": int(accumulator.cap_bytes),
        "estimated_uncompressed_bytes": int(accumulator.estimated_uncompressed_bytes),
    }
    return block, {STEP_CONTEXT_DELTA_SIDECAR_FILENAME: payload_bytes}


def _current_resident_set_kib() -> int:
    """Current host RSS in KiB, or ``-1`` where ``/proc`` is unavailable.

    THE OBSERVABLE THAT CAN FALL. ``ru_maxrss`` is a HIGH-WATER MARK, monotone
    by construction, so it can never exhibit a release and therefore cannot
    separate accumulation from working set: in one measured run it showed zero
    decreases across 45 rows, a property of the field, not of the process.

    Paired with the peak, this makes the decomposition direct: the peak bounds
    the transient, and the trough of THIS series is the part never returned.
    """

    try:
        with open("/proc/self/statm", "rb") as handle:
            resident_pages = int(handle.read().split()[1])
    except Exception:  # pragma: no cover - non-Linux hosts have no /proc.
        return -1
    return resident_pages * (resource.getpagesize() // 1024)


class _PhaseTiming:
    """Execution-only phase observations, never read by a decision."""

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.events: list[dict[str, object]] = []
        self.host_output_path: str | None = None
        self.host_completion = False

    def detach_for_host_completion(self) -> None:
        """Stop sampling another item's CUDA counters during host writes."""
        self.host_completion = True

    def mark(self, phase: str, **coordinates: object) -> None:
        if not self.enabled:
            return
        rss_kib = _current_resident_set_kib()
        if rss_kib < 0:
            try:
                import psutil
            except ImportError:
                # The registered execution environment has /proc.  This
                # fallback keeps the phase line available on a host without
                # psutil, where resource exposes only the peak RSS.
                peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                rss_bytes = peak if sys.platform == "darwin" else peak * 1024
            else:
                rss_bytes = int(psutil.Process().memory_info().rss)
        else:
            rss_bytes = int(rss_kib * 1024)
        row: dict[str, object] = {
            "phase": str(phase),
            "monotonic_seconds": time.monotonic(),
            "utc_time": datetime.now(timezone.utc).isoformat(),
            "rss_bytes": rss_bytes,
            **coordinates,
        }
        torch_module = sys.modules.get("torch")
        if not self.host_completion and torch_module is not None and torch_module.cuda.is_available():
            row["cuda_allocated_bytes"] = int(torch_module.cuda.memory_allocated())
            row["cuda_peak_allocated_bytes"] = int(
                torch_module.cuda.max_memory_allocated()
            )
        self.events.append(row)
        line = "[phase] " + json.dumps(row, sort_keys=True)
        if self.host_completion and self.host_output_path is not None:
            with open(self.host_output_path, "a", encoding="utf-8") as output:
                print(line, file=output, flush=True)
        else:
            print(line, file=sys.stderr, flush=True)

    def receipt_block(self) -> Mapping[str, object]:
        return {"schema": "phase_timing_v1", "events": tuple(self.events)}


def _write_gpu_done_marker(
    destination: Path, *, run_id: str, mask_hash: str, cuda_reserved_bytes: int,
    resolved_settings: Mapping[str, Any] | None = None,
) -> None:
    """Publish the completed GPU boundary for a launcher by atomic rename."""

    body = {
        "schema": "irg_gpu_done_v1",
        "run_id": str(run_id),
        "final_mask_hash": str(mask_hash),
        "cuda_reserved_bytes": int(cuda_reserved_bytes),
    }
    body["effective_settings"] = effective_settings(resolved=resolved_settings)
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    body["content_sha256"] = hashlib.sha256(canonical).hexdigest()
    encoded = (json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n").encode()
    destination.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".gpu_done.", dir=destination)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination / "gpu_done.json")
        directory_fd = os.open(destination, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        (destination / "gpu_done.json").unlink(missing_ok=True)
        raise
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _gpu_done_marker_boundary(
    destination: Path, *, run_id: str, mask_hash: str, device: Any,
    phase_timing: _PhaseTiming,
    resolved_settings: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    """Publish after all host captures; remove the marker on a later failure.

    The marker records a completed device boundary, while receipt.json
    records completion of the run. Live tensors may still reserve memory;
    empty_cache releases unused allocator blocks and the remaining reserved
    bytes are disclosed for the launcher.
    """

    try:
        import torch

        reserved_bytes = 0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            reserved_bytes = int(torch.cuda.memory_reserved(device))
        phase_timing.mark("gpu_done", cuda_reserved_bytes=reserved_bytes)
        _write_gpu_done_marker(
            destination, run_id=run_id, mask_hash=mask_hash,
            cuda_reserved_bytes=reserved_bytes,
            resolved_settings=resolved_settings,
        )
        yield
    except BaseException:
        (destination / "gpu_done.json").unlink(missing_ok=True)
        raise


def _host_byte_census(
    *, top_n: int = 12, node_cap: int = 4_000_000
) -> Mapping[str, object]:
    """Name the host allocations holding the most bytes, by kind/dtype/shape.

    ATTRIBUTION BY MEASUREMENT, because attribution by reading has now failed
    five times on this same OOM. The reason it keeps failing is structural: on
    a doubling panel every candidate term scales together, so cumulative
    contexts, current panel, packet count and route-chunk updates are mutually
    collinear and no amount of arithmetic over look-level aggregates can
    separate them. Bytes-per-shape is not collinear with anything -- it names
    the object.

    numpy views are folded into their base so one buffer is never counted under
    two shapes, and torch tensors are deduplicated by storage pointer so
    aliases do not inflate the total.
    """

    import gc
    import warnings

    try:
        import torch
    except Exception:  # pragma: no cover - torch is optional for import smoke.
        torch = None  # type: ignore[assignment]

    totals: dict[str, list[int]] = {}
    seen_storages: set[int] = set()
    visited: set[int] = set()
    examined = 0
    truncated = False

    def _add(label: str, nbytes: int) -> None:
        row = totals.setdefault(label, [0, 0])
        row[0] += int(nbytes)
        row[1] += 1

    # A REFERENT WALK, not a flat sweep. `gc.get_objects()` returns only
    # GC-TRACKED objects, and a numpy array is not tracked unless its dtype is
    # object -- so the flat sweep returns every container and none of the
    # buffers, which is to say zero bytes and a confident empty answer. The
    # arrays are reachable FROM the tracked containers, so the walk follows
    # `gc.get_referents` outward to the untracked leaves.
    # Roots: tracked objects PLUS every live frame's locals and globals. A bare
    # local array is reachable only through its frame, and CPython materializes
    # frames lazily, so `gc.get_objects()` alone can miss the largest buffers on
    # the stack -- `contribution` and `preactivation` in the flip-margin chunk
    # loop are exactly that shape of object.
    import sys

    roots: list[object] = list(gc.get_objects())
    for frame in list(sys._current_frames().values()):
        cursor: object | None = frame
        while cursor is not None:
            roots.append(getattr(cursor, "f_locals", None))
            roots.append(getattr(cursor, "f_globals", None))
            cursor = getattr(cursor, "f_back", None)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frontier: list[object] = roots
        while frontier and not truncated:
            successors: list[object] = []
            for obj in frontier:
                identity = id(obj)
                if identity in visited:
                    continue
                visited.add(identity)
                examined += 1
                if examined >= node_cap:
                    truncated = True
                    break
                try:
                    if isinstance(obj, np.ndarray):
                        if obj.base is None:
                            _add(
                                f"np:{obj.dtype}:{tuple(obj.shape)}",
                                obj.nbytes,
                            )
                        continue
                    if torch is not None and torch.is_tensor(obj):
                        storage = obj.untyped_storage()
                        pointer = int(storage.data_ptr())
                        if pointer not in seen_storages:
                            seen_storages.add(pointer)
                            _add(
                                f"pt:{obj.device}:{obj.dtype}:" f"{tuple(obj.shape)}",
                                storage.nbytes(),
                            )
                        continue
                    successors.extend(gc.get_referents(obj))
                except Exception:  # pragma: no cover - foreign/dead objects.
                    continue
            frontier = successors

    ranked = sorted(totals.items(), key=lambda item: -item[1][0])[:top_n]
    return {
        "schema": "irg_host_byte_census_v1",
        "top": [
            {"label": label, "bytes": row[0], "count": row[1]} for label, row in ranked
        ],
        "distinct_shapes": len(totals),
        "total_bytes": sum(row[0] for row in totals.values()),
        "nodes_examined": examined,
        # A silent cap would read as "covered everything" when it did not.
        "truncated": truncated,
    }


_CONSEQUENCE_PRICE_SHORT_CIRCUIT_ENV = "IRG_CONSEQUENCE_PRICE_SHORT_CIRCUIT"
_CENSUS_MATERIALITY_RULE = "decision_look0_dense_loss_resolution_v1"


def _census_materiality_floor(resolution: Mapping[str, Any] | None) -> float | None:
    """Use the supplied dense-loss first-look resolution in loss units."""
    if resolution is None:
        return None
    value = float(resolution["delta"])
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("census materiality resolution must be finite and nonnegative")
    return value


_CENSUS_DECISION_REFERENCED_ENV = "IRG_CENSUS_DECISION_REFERENCED"
_CENSUS_EPSILON_ENV = "IRG_CENSUS_EPSILON"
_ANALYTIC_SIGN_ENV = "IRG_ANALYTIC_SIGN_PARTITION"
_ANALYTIC_SIGN_KINDS = frozenset({"conv_kernel_weight", "linear_weight"})
# ``all_supported_routes`` gives every supported group the sign
# of its effective weight, so the partition is exactly weight polarity in
# every layer.  For a group whose source can be negative (the raw input when
# the folded uint8 stem is not engaged) that is a registered DEFINITION, not
# the local-attribution identity; the receipt says which one applies.
_ANALYTIC_SIGN_SCOPE_ENV = "IRG_ANALYTIC_SIGN_SCOPE"
_ANALYTIC_SIGN_SCOPE_ALL = "all_supported_routes"
# Width of the terminal category-calibration chunk. The fit is
# per one-vs-rest column with one shared stratified selection layout, and its
# docstring (providers ``_fit_terminal_category_calibrations``) states results
# are independent of the chunk partition; a wider chunk only removes per-chunk
# Python/solver overhead (the warm-up spent 70% of its main thread here at
# width 8). Default 8 reproduces every historical run; launchers may widen it.
# Partition independence is bitwise on CPU; on GPU it holds up to reduction
# order and the call-global audit fallback (solver tolerance).
#
# DEVICE MEMORY WARNING. The proper-score selection solve
# holds float64 arrays of shape [20 selection lanes (4 blocks x 5 anchor
# strengths), N training rows, width] on the GPU, about 34.5 [20, N] slabs per
# category column at its peak. At N = 256,000 rows (1,000 classes x 256) that
# is ~1.32 GiB per column: ~10.5 GiB at the default width 8, ~42 GiB at 32.
# At width 1000 a SINGLE such array is ~38 GiB (40.96e9 B; a run OOMed
# allocating the third one) and the peak would be ~1.3 TiB. No automatic cap
# is applied: size the width against free device memory before widening it.
_TERMINAL_CATEGORY_FIT_CHUNK_ENV = "IRG_TERMINAL_CATEGORY_FIT_CHUNK"


def _terminal_category_fit_chunk(value: int | None) -> int:
    if value is not None:
        width = int(value)
    else:
        width = int(os.environ.get(_TERMINAL_CATEGORY_FIT_CHUNK_ENV, "8").strip() or "8")
    if width <= 0:
        raise ValueError(f"{_TERMINAL_CATEGORY_FIT_CHUNK_ENV} must be a positive integer")
    return width


def _enabled_environment_flag(name: str) -> bool:
    raw = os.environ.get(name, "0").strip().lower()
    if raw in {"", "0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(f"{name} must be a boolean token")


def _census_decision_referenced() -> bool:
    """Whether census closure uses the decision-scale consequence bound."""

    return _enabled_environment_flag(_CENSUS_DECISION_REFERENCED_ENV)


def _census_epsilon() -> float:
    raw = os.environ.get(_CENSUS_EPSILON_ENV, "0.005").strip()
    value = float(raw)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{_CENSUS_EPSILON_ENV} must lie in (0,1)")
    return value


def _analytic_sign_mode() -> bool:
    """Enable the exact post-activation local-attribution sign partition.

    For a Conv/Linear weight route whose source is post-activation,
    ``local_attr_r = effective_weight_r * mean(source_factor_r)`` and the
    factor is nonnegative. ``effective_weight`` includes the frozen signed
    destination affine transport. The sign is therefore its sign, including
    the declared nonnegative zero tie. Every supported group uses this sign;
    unsupported route kinds remain sampled.
    """

    return _enabled_environment_flag(_ANALYTIC_SIGN_ENV)


def _analytic_sign_scope() -> str:
    raw = os.environ.get(_ANALYTIC_SIGN_SCOPE_ENV, "").strip().lower()
    if raw in {"", _ANALYTIC_SIGN_SCOPE_ALL}:
        return _ANALYTIC_SIGN_SCOPE_ALL
    raise ValueError(
        f"{_ANALYTIC_SIGN_SCOPE_ENV} must be {_ANALYTIC_SIGN_SCOPE_ALL!r}, got {raw!r}"
    )


def _analytic_sign_group(route_kind: str) -> bool:
    return bool(
        _analytic_sign_mode()
        and str(route_kind) in _ANALYTIC_SIGN_KINDS
    )


def _census_simultaneous_family_size(
    *,
    route_count: int,
    measured_local_sign_route_count: int,
    required_views: Sequence[str],
) -> int:
    """Size the single joint family to the coordinates actually claimed."""

    total = int(route_count)
    measured_local = int(measured_local_sign_route_count)
    views = tuple(dict.fromkeys(map(str, required_views)))
    if total <= 0 or not 0 <= measured_local <= total:
        raise ValueError("invalid analytic-sign family cardinality")
    if not views or any(view not in {"local", "global", "loss"} for view in views):
        raise ValueError("invalid analytic-sign family views")
    # The order coordinate is always full-width.  Only local attribution sign
    # benefits from the theorem; global attribution remains full-width.
    return total + sum(measured_local if view == "local" else total for view in views)


def _census_simultaneous_block_critical_values(
    *,
    look_confidence: float,
    route_count: int,
    measured_local_sign_route_count: int,
    required_views: Sequence[str],
) -> tuple[float, int, int, float, float]:
    """Return a joint-valid sign/order block allocation for one look.

    Nonempty sign and order blocks receive equal shares of the look's
    error probability and are Bonferroni-corrected internally. If every
    label sign is analytic, the order block receives the whole look stake.  For the registered local-sign arm,
    this is algebraically identical to the fast historical correction over
    ``2 * measured`` sign coordinates and ``2 * route_count`` order
    coordinates, while making the joint coverage argument explicit.
    """

    confidence = float(look_confidence)
    if not 0.0 < confidence < 1.0:
        raise ValueError("look confidence must lie in (0,1)")
    total = int(route_count)
    measured = int(measured_local_sign_route_count)
    views = tuple(dict.fromkeys(map(str, required_views)))
    _census_simultaneous_family_size(
        route_count=total,
        measured_local_sign_route_count=measured,
        required_views=views,
    )
    sign_coordinates = sum(measured if view == "local" else total for view in views)
    # Retain the nonzero computational sentinel for the unused sign quantile.
    sign_family = max(sign_coordinates, 1)
    order_family = total
    per_block_confidence = (
        1.0 - (1.0 - confidence) / 2.0 if sign_coordinates else confidence
    )
    return (
        per_block_confidence,
        sign_family,
        order_family,
        simultaneous_normal_critical_value(
            per_block_confidence,
            family_size=sign_family,
        ),
        simultaneous_normal_critical_value(
            per_block_confidence,
            family_size=order_family,
        ),
    )


def _census_sign_criterion_value(
    *,
    mean: np.ndarray,
    standard_error: np.ndarray,
    activity: np.ndarray,
    critical: float,
) -> float:
    """Evaluate the exact same census criterion at runtime and forecast.

    The decision-referenced mode stops only on a simultaneous confidence
    upper bound.  The Gaussian expected-loss quantity is intentionally not
    used as authority; it is emitted separately as acquisition telemetry.
    """

    if _census_decision_referenced():
        return wrongsign_consequence_upper_bound_fraction(
            mean=mean,
            standard_error=standard_error,
            activity=activity,
            critical=critical,
        )
    return unresolved_activity_fraction(
        mean=mean,
        standard_error=standard_error,
        activity=activity,
        critical=critical,
    )


def _consequence_price_short_circuit_enabled() -> bool:
    """Resolve the decision-identical consequence-pricing optimization."""

    raw = os.environ.get(_CONSEQUENCE_PRICE_SHORT_CIRCUIT_ENV, "").strip().lower()
    if not raw or raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(f"{_CONSEQUENCE_PRICE_SHORT_CIRCUIT_ENV} must be a boolean token")


# One buffered runtime-only line per completed decision.
# Every v6 event passes through this sink; per-look and audit events are silent.
_V6_DECISION_PROGRESS_EVENT = "v0_proposal_decision_completed"
_V6_PROGRESS_RUNTIME_FIELDS = frozenset(
    {
        "looks",
        "panel_images",
        "requested_cumulative_images",
        "delivered_cumulative_images",
        "new_context_count",
        "elapsed_seconds",
        "wall_seconds",
    }
)
_V6_PROGRESS_BUFFER: list[str] = []
_V6_PROGRESS_FLUSH_LINES = 64
_V6_PROGRESS_FLUSH_SECONDS = 2.0
_V6_PROGRESS_LAST_FLUSH = [0.0]
# Provider callbacks may emit from worker threads: one lock orders the
# buffer, its flush and the unbuffered lines, so no line is lost or reordered.
_V6_PROGRESS_LOCK = threading.RLock()


def _v6_progress_flush() -> None:
    with _V6_PROGRESS_LOCK:
        if _V6_PROGRESS_BUFFER:
            print("\n".join(_V6_PROGRESS_BUFFER), flush=True)
            _V6_PROGRESS_BUFFER.clear()
        _V6_PROGRESS_LAST_FLUSH[0] = time.monotonic()


atexit.register(_v6_progress_flush)


def _v6_progress_scalar(key: str, value: object) -> bool:
    return key in _V6_PROGRESS_RUNTIME_FIELDS and (
        value is not None and isinstance(value, (int, float)) and not isinstance(value, bool)
    )


def _progress(event: str, **details: object) -> None:
    """Emit machine-readable pilot progress and host RSS.

    Both RSS observables are emitted: the peak (``ru_maxrss``) because it is
    what the OOM killer's threshold is compared against, and the current RSS
    because it is the only one of the two that can fall.

    Emit one buffered runtime line for each completed decision.
    """


    if str(event) != _V6_DECISION_PROGRESS_EVENT:
        return
    now = time.monotonic()
    line = (
        json.dumps(
            {
                "schema": "irg_resnet18_pilot_progress_v6_1",
                "event": str(event),
                "monotonic_seconds": now,
                "maximum_resident_set_kib": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                ),
                "resident_set_kib": _current_resident_set_kib(),
                **{
                    key: value
                    for key, value in details.items()
                    if _v6_progress_scalar(key, value)
                },
            },
            sort_keys=True,
            default=str,
        )
    )
    with _V6_PROGRESS_LOCK:
        _V6_PROGRESS_BUFFER.append(line)
        if (
            len(_V6_PROGRESS_BUFFER) >= _V6_PROGRESS_FLUSH_LINES
            or now - _V6_PROGRESS_LAST_FLUSH[0] >= _V6_PROGRESS_FLUSH_SECONDS
        ):
            _v6_progress_flush()
    return


def _canonical_partition_family(family: str) -> str:
    resolved = str(family)
    if resolved not in _PARTITIONS:
        raise ValueError("pilot partition family is unsupported")
    return _PARTITION_ALIASES.get(resolved, resolved)


def _target_class_id(query: Any) -> int:
    value = query.evaluator.get(
        "class_id",
        query.evaluator.get("unit", query.evaluator.get("coordinate")),
    )
    if value is None:
        raise ValueError("ResNet18 pilot requires a scalar class query")
    return int(value)


def _census_rung_profile_receipt(profile: Any | None) -> dict[str, float]:
    """The rung rule supplied to the walk, with today's v0 rule for direct calls."""
    defaults = {
        "target_preservation_conv_decay": 0.85,
        "target_preservation_conv_target_final_density": 1e-4,
        "target_preservation_linear_decay": 0.85,
        "target_preservation_linear_target_final_density": 1e-4,
    }
    return {
        name: float(value if profile is None else getattr(profile, name))
        for name, value in defaults.items()
    }


def _v0_executable_boundary_fractions(
    route_groups: Sequence[Any],
    *,
    profile: Any | None = None,
    partition_route_counts_by_layer: Mapping[str, Mapping[str, int]] | None = None,
    proposal_route_cap: int | None = None,
    census_schedule_replay_schedule: Mapping[str, Any] | None = None,
) -> dict[str, tuple[float, ...]]:
    """Audit every count-changing cut of the walk's layer/partition rung rule."""
    from mechanistic_systems.circuits.frontier_rungs import (
        boundary_fractions_from_retained_counts,
        retained_counts_for_density_schedule,
    )
    from .prune_scope_walk import executable_ladder_for_kinds, _refined_ladder

    if census_schedule_replay_schedule is not None:
        from .schedule_replay import schedule_replay_executable_retained_counts

        if partition_route_counts_by_layer is None:
            raise ValueError("replay census cuts require the current partition counts")
        replay_counts = schedule_replay_executable_retained_counts(
            census_schedule_replay_schedule, route_groups, partition_route_counts_by_layer
        )
        return {
            f"{layer}:partition:{label}": cuts[partition]
            for layer, retained in replay_counts.items()
            for cuts in (boundary_fractions_from_retained_counts(retained),)
            for label, partition in enumerate(("positive", "negative"))
        }
    rung_profile = SimpleNamespace(**_census_rung_profile_receipt(profile))
    result: dict[str, tuple[float, ...]] = {}
    eligible = tuple(
        group for group in route_groups
        if group.route_kind in {"linear_weight", "conv_channel_pair", "conv_kernel_weight"}
    )
    if not eligible:
        raise ValueError("hard-partition scorer has no executable connection group")
    for group in eligible:
        walk_kind = "linear_like" if group.route_kind == "linear_weight" else "conv_like"
        schedule, _floor = executable_ladder_for_kinds((walk_kind,), rung_profile)
        counts = (
            {"positive": int(np.prod(group.shape)), "negative": int(np.prod(group.shape))}
            if partition_route_counts_by_layer is None
            else partition_route_counts_by_layer[str(group.layer_id)]
        )
        retained = retained_counts_for_density_schedule(counts, schedule)
        if proposal_route_cap is not None:
            _, retained = _refined_ladder(schedule, retained, int(proposal_route_cap))
            cuts = boundary_fractions_from_retained_counts(retained)
        else:
            cuts = {
                partition: tuple(float(schedule[j]) for j in range(1, len(schedule))
                    if values[j] < values[j-1] and 0 < values[j] < values[0])
                for partition, values in retained.items()
            }
        for label, partition in enumerate(("positive", "negative")):
            result[f"{group.layer_id}:partition:{label}"] = cuts[partition]
    return result


def _census_partition_route_counts(
    labels: np.ndarray, offsets: Mapping[str, tuple[int, int]],
) -> dict[str, dict[str, int]]:
    return {
        layer: {
            "positive": int(np.count_nonzero(np.asarray(labels[start:stop]) == 0)),
            "negative": int(np.count_nonzero(np.asarray(labels[start:stop]) == 1)),
        }
        for layer, (start, stop) in offsets.items()
    }


# Registered v1 executable ladder (irg_v1_repair_program_preregistration_v1):
# dyadic global retained fractions n/32 for n in 10..31. Changing the ladder
# is a new registration.
_V1_EXECUTABLE_LADDER: tuple[float, ...] = tuple(n / 32.0 for n in range(10, 32))
# The registration pre-registers a per-decision look cap without a numeral;
# the codebase's registered planned-look horizon (BCEPrecisionPolicy /
# OracleSequentialPrecisionPolicy default) is 32.
_V1_DECISION_LOOK_CAP = 32
_V1_CENSUS_TERMINAL_RULE_ID = "census_complete_terminal_resolution_v1"

# Three-knob normalization: chart certification is
# proposal-side work — its fidelity demands scale with the trust
# tolerance (knob 3), never with the commit-grade decision confidence
# (knob 2), so the scoped class runs at a structural proposal-grade
# simultaneous confidence rather than the census look-split level.
_V1_CHART_PROPOSAL_CONFIDENCE = 0.95

# The trust walk IS v1.  The repair-era traversal and the v1 census path
# are not part of it; their receipts and the repair-era registration remain
# the immutable record.
# Scoped-field acquisition identity (era-separated from the census-grade
# v4 field cache; the v4 key function and receipts are byte-untouched).
_TRUST_WALK_CHART_FIELD_ROLE = "trust_walk_chart_field"
_TRUST_WALK_FANOUT_IMPLEMENTATION_ID = "pilot_tripartite_fd_fanout_v1"
_TRUST_WALK_SCOPED_FIELD_CACHE_KEY_SCHEMA = "trust_walk_scoped_field_cache_key_v1"
_TRUST_WALK_SCOPED_FIELD_CACHE_RECEIPT_SCHEMA = (
    "trust_walk_scoped_field_cache_receipt_v1"
)
_TRUST_WALK_RESPONSE_SEMANTICS = "raw_target_logit_for_trust_walk_chart_field_v1"
_TRUST_WALK_RESPONSE_METRIC_ID = "v1-trust-walk-chart-field-metric-v1"
# Registered trust-region refresh radius (registration ``trust_region``);
# the reuse radius is the registered chart leverage cap below.
_V1_TRUST_WALK_FD_REFRESH_THRESHOLD = 0.35
_V1_TRUST_WALK_FD_TRUST_ID = "irg_v1_trust_walk_chart_field_fd_v1"
# Chart-ACQUISITION constants, not TrustWalkPolicy fields.
# The unified trust-region walk has no fixed leverage cap and no
# geometric step grid in the WALK (its region radius is the learned
# high-water tau and its steps are continuous key-order prefixes); the
# decision-scoped MEASUREMENT class keeps them as acquisition-scoping
# constants: the leverage cap bounds the certified horizon prefix and the
# audit grid names the order-stability cut fractions. Values are
# byte-identical to the registered numerals (registration ``trust_region``
# and ``delta_schedule``) so scoped cache keys are unchanged.
_V1_TRUST_WALK_CHART_LEVERAGE_CAP = 0.1
_V1_TRUST_WALK_AUDIT_DELTA_MAX_FRACTION = 0.125
# The 64-route scale is not a TrustWalkPolicy field; it is used only
# as the acquisition audit-grid constant (measurement disclosure grid).
_V1_TRUST_WALK_AUDIT_DELTA_MIN_ROUTES = 64
_V1_TRUST_WALK_AUDIT_GRID_SCHEDULE = "geometric_halving_v1"


def _audit_delta_grid(*, delta_start: int, delta_min_routes: int) -> tuple[int, ...]:
    """Geometric audit-cut grid for the scoped order-stability family."""

    grid: list[int] = []
    value = int(delta_start)
    while True:
        grid.append(value)
        if value <= int(delta_min_routes):
            break
        value = max(math.ceil(value / 2), int(delta_min_routes))
    return tuple(grid)


# ``partition_ids[label]`` binding for the scoped class: label 0 is the
# nonnegative sign class (zero deterministically nonnegative), label 1
# the negative class — the canonical global sign partition ids.
CANONICAL_TRUST_WALK_PARTITION_IDS = (
    "global_nonnegative",
    "global_negative",
)
# v1 = v0's working chassis + the walk.
# The original v1's native session is never constructed on a v1 arm, so
# receipt fields whose only honest source was that session's census-era
# version state carry this literal disclosure — never a fabricated value.
_V1_SESSION_RETIREMENT_DISCLOSURE = "retired_with_the_v1_session_E2"


def _v1_global_retained_count(fraction: float, total_routes: int) -> int:
    """Global retained count at one ladder cut (audit and translation)."""

    return min(
        max(int(math.ceil(float(fraction) * int(total_routes))), 1),
        int(total_routes) - 1,
    )


_TRUST_WALK_RANDOMIZED_PARTITION_ROLE = (
    "trust_walk_randomized_count_matched_partition_v1"
)


def _trust_walk_randomized_partition_permutation(
    *,
    partition_random_seed: int,
    run_id: str,
    committed_mask_hash: str,
    route_count: int,
) -> np.ndarray:
    """The frozen per-chart route permutation of the randomized control.

    Seed material includes the committed mask hash, so the permutation is
    REDRAWN at every re-centering — matching the sign family's per-chart
    label refresh — while staying deterministic for a given (seed, run,
    anchor). Same PCG64 permutation algorithm as the census family
    (``numpy_pcg64_permutation_v1``).
    """

    material = content_hash(
        {
            "role": _TRUST_WALK_RANDOMIZED_PARTITION_ROLE,
            "partition_random_seed": int(partition_random_seed),
            "run_id": str(run_id),
            "committed_mask_hash": str(committed_mask_hash),
        }
    )
    return np.random.default_rng(int(material[-16:], 16)).permutation(int(route_count))


def _count_matched_random_labels(
    global_mean: np.ndarray, permutation: np.ndarray
) -> np.ndarray:
    """Byte labels of the randomized control at the current look.

    Exactly the census family's semantics at chart scope: the label-1
    block is COUNT-MATCHED to the current negative sign class while
    membership is the frozen random permutation's prefix — so per-look
    count drift keeps labels NESTED along one permutation instead of
    re-randomizing, and no sign information survives in membership.
    """

    mean = np.asarray(global_mean, dtype=np.float64).reshape(-1)
    if mean.size != permutation.size:
        raise ValueError("randomized-control permutation does not match the route axis")
    labels = np.zeros(permutation.size, dtype=np.int8)
    labels[permutation[: int(np.sum(mean < 0.0))]] = 1
    return labels


def _exact_ceil_fraction(count: int, size: int) -> float:
    """Return f in (0,1) with ceil(f * size) == count, exactly.

    ``_scope_receipt`` recovers audited counts through ceil(fraction *
    subset_size); floating rounding must not shift the audited boundary by
    one route.
    """

    retained = int(count)
    subset = int(size)
    if not 1 <= retained <= subset - 1:
        raise ValueError("exact ceil fraction requires an interior count")
    fraction = retained / subset
    if 0.0 < fraction < 1.0 and math.ceil(fraction * subset) == retained:
        return float(fraction)
    fraction = (retained - 0.5) / subset
    if math.ceil(fraction * subset) != retained:
        raise ValueError("ceil-exact fraction translation failed")
    return float(fraction)


def _translate_global_ladder_to_partition_fractions(
    *,
    global_activity: np.ndarray,
    partition_labels: np.ndarray,
    ladder: Sequence[float],
) -> tuple[dict[int, tuple[float, ...]], tuple[dict[str, Any], ...]]:
    """Map global retained cuts onto within-partition audited fractions.

    The registered ladder declares GLOBAL retained fractions while the order
    certificate audits within-partition subsets (the registered deciding
    risk).  A global cut is one activity threshold: the stable global top-K
    splits across the sign partitions by their activity composition, so each
    partition's audited count is its membership in the global top-K.
    """

    activity = np.asarray(global_activity, dtype=np.float64).reshape(-1)
    labels = np.asarray(partition_labels).reshape(-1)
    if (
        activity.shape != labels.shape
        or activity.size < 2
        or not np.isfinite(activity).all()
        or not np.isin(labels, (0, 1)).all()
    ):
        raise ValueError("ladder translation requires a labeled activity axis")
    order = np.argsort(-activity, kind="stable")
    labels_in_order = labels[order]
    total = int(activity.size)
    partition_sizes = {
        partition: int(np.count_nonzero(labels == partition)) for partition in (0, 1)
    }
    per_partition: dict[int, list[float]] = {0: [], 1: []}
    table: list[dict[str, Any]] = []
    for fraction in ladder:
        retained = _v1_global_retained_count(fraction, total)
        head = labels_in_order[:retained]
        entry: dict[str, Any] = {
            "global_retained_fraction": float(fraction),
            "global_retained_count": int(retained),
            "partitions": {},
        }
        for partition in (0, 1):
            size = partition_sizes[partition]
            count = int(np.count_nonzero(head == partition))
            if size == 0 or count <= 0 or count >= size:
                # The global cut lands outside this partition's interior:
                # no within-partition boundary exists, so no order claim.
                entry["partitions"][str(partition)] = {
                    "partition_route_count": size,
                    "retained_count": count,
                    "audited": False,
                    "reason": "no_within_partition_boundary",
                }
                continue
            partition_fraction = _exact_ceil_fraction(count, size)
            per_partition[partition].append(partition_fraction)
            entry["partitions"][str(partition)] = {
                "partition_route_count": size,
                "retained_count": count,
                "partition_retained_fraction": partition_fraction,
                "audited": True,
            }
        table.append(entry)
    return (
        {
            partition: tuple(dict.fromkeys(values))
            for partition, values in per_partition.items()
        },
        tuple(table),
    )


def _census_complete_terminal_receipt(
    *,
    strictly_resolved: bool,
    sign_resolved: bool,
    order_resolved: bool,
    existing: Mapping[str, int],
    capacity: Mapping[str, int],
    stable_looks: int,
    required_stable_looks: int,
    unresolved_activity_fraction: Mapping[str, float],
    required_partition_views: Sequence[str],
    maximum_permitted_unresolved_activity_fraction: float,
    order_certificate_receipt: Mapping[str, Any],
    frozen_calibration_state_id: str,
) -> dict[str, Any] | None:
    """Waive ONLY stable-looks when the panel is the complete census.

    Over the complete sealed census the estimand is exact, so consecutive
    sampling-stability looks are degenerate; resolution itself is NEVER
    waived — both the sign views and the order certificate must resolve on
    the census-complete look or the acquisition raise stands.
    """

    census_complete = all(
        int(existing.get(stratum, 0)) >= int(capacity[stratum]) for stratum in capacity
    )
    if (
        strictly_resolved
        or not census_complete
        or not sign_resolved
        or not order_resolved
    ):
        return None
    return {
        "schema": _V1_CENSUS_TERMINAL_RULE_ID,
        "waived": ("consecutive_stable_looks",),
        "resolution_waived": False,
        "stable_looks_observed": int(stable_looks),
        "stable_looks_required": int(required_stable_looks),
        "per_stratum_counts": {
            stratum: {
                "delivered": int(existing.get(stratum, 0)),
                "sealed_census_capacity": int(capacity[stratum]),
            }
            for stratum in capacity
        },
        "sign_resolved": True,
        "unresolved_activity_fraction_by_view": {
            str(key): float(value)
            for key, value in unresolved_activity_fraction.items()
        },
        "required_partition_views": tuple(
            str(view) for view in required_partition_views
        ),
        "maximum_permitted_unresolved_activity_fraction": float(
            maximum_permitted_unresolved_activity_fraction
        ),
        "order_certificate_resolved": True,
        "order_certificate_boundary_values": {
            key: order_certificate_receipt[key]
            for key in (
                "maximum_observed_boundary_activity_fraction",
                "maximum_permitted_boundary_activity_fraction",
                "maximum_observed_mask_disagreement_activity_fraction",
                "maximum_permitted_mask_disagreement_activity_fraction",
            )
            if key in order_certificate_receipt
        },
        # Build-time verified precondition: the acquisition's response
        # definition applies one frozen dense calibration through a fixed
        # loss-pullback identity; nothing refits per look, so the terminal
        # look measures the same estimand as every earlier look.
        "calibration_look_invariance": {
            "verified": True,
            "mechanism": (
                "single frozen dense calibration_resolver; accumulator "
                "loss_pullback_hash binds the calibration_state_id and "
                "raises on any drift"
            ),
            "frozen_calibration_state_id": str(frozen_calibration_state_id),
        },
    }


def _realized_terminal_span_fraction(
    terminal_dense_bce: float,
    terminal_final_bce: float,
    target_prior: float | None = None,
) -> dict[str, Any]:
    """Realized terminal span spend relative to the intercept-only BCE risk.

    The declared v1 budget bounds an optimization-side paired UCB, so
    realized terminal damage typically undershoots the declared fraction;
    this block makes the actual spend visible next to the budget.  The
    chance anchor ``ln 2`` is the balanced-binary constant-response risk —
    the same anchor the loss-span certificate uses.  A terminal dense risk
    at or above chance leaves the fraction undefined and is disclosed as
    such rather than sign-flipped.
    """

    dense = float(terminal_dense_bce)
    final = float(terminal_final_bce)
    chance = _chance_risk(target_prior)
    span = chance - dense
    if not math.isfinite(span) or span <= 0.0:
        return {
            "value": None,
            "state": "undefined_terminal_dense_not_below_chance",
            "terminal_dense_bce": dense,
            "terminal_final_bce": final,
            "chance_anchor": chance,
        }
    return {
        "value": float((final - dense) / span),
        "state": "measured",
        "terminal_dense_bce": dense,
        "terminal_final_bce": final,
        "chance_anchor": chance,
    }


def _query_signed_score_block(
    measurement: Any,
    *,
    layer_id: str,
    query_index: int,
) -> np.ndarray:
    reader = getattr(measurement, "source_image_aggregate_query_block", None)
    if callable(reader):
        values = reader(
            layer_id,
            query_index=int(query_index),
            score_head="route_scores",
        )
    else:
        values = measurement.source_image_aggregate_block(
            layer_id,
            score_head="route_scores",
        )[int(query_index)]
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(result).all():
        raise ValueError("attributional partition scores must be finite")
    return result


# The randomized count-matched control (RP)
# draws each layer's membership from a PCG64 permutation seeded by
# SeedSequence((partition_random_seed, category_id, route_group_index)).
# With (partition_random_seed, route_group_index) alone the category would
# never enter the seed material, and every category would share one
# partition per layer.  ``category_id`` is the arm's target class id (the
# category); the query index is not used because it is the query's position
# on the measurement axis and the census passes 0.
_RANDOMIZED_PARTITION_SEED_MATERIAL = (
    "numpy_seedsequence_partition_random_seed_category_id_route_group_index_v2"
)
# RP's per-layer negative-class count is taken from exactly the labels the
# local-attribution-sign family (P) uses (the effective-weight sign in every
# supported layer when the analytic partition is on), so RP differs from P
# only in membership.
_RANDOMIZED_PARTITION_COUNT_SOURCE_P_LABELS = (
    "local_attribution_sign_labels_under_active_analytic_sign_scope_v1"
)


def _census_analytic_sign_active(partition_family: str) -> bool:
    """Whether the census resolves supported groups' local signs analytically.

    P and RP use the same labels for per-layer counts. SA-H's uniform labels
    read no measured sign, and its census runs the same sign family to keep
    the local-vs-single controlled ablation.
    """

    _analytic_sign_scope()
    return bool(
        _analytic_sign_mode()
        and str(partition_family)
        in {"local_attribution_sign", "single_population", "randomized_count_matched"}
    )


# Census cache files.  ``1`` (default): the locked read/write cache
# below.  ``0``: the census reads and writes no file in the geometry folder
# (no entry, receipt or lock); every census is measured in the process, for a
# campaign that keeps the census in memory and never reads the files.
CENSUS_CACHE_FILES_ENVIRONMENT = "IRG_CENSUS_CACHE_FILES"


def _census_cache_files_enabled() -> bool:
    raw = str(os.environ.get(CENSUS_CACHE_FILES_ENVIRONMENT, "1")).strip().lower()
    if raw in {"", "1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{CENSUS_CACHE_FILES_ENVIRONMENT} must be a boolean token")


def _hard_partition_cache_keys(cache_key: str) -> tuple[tuple[str, ...], str, str | None]:
    """(keys to read, in order; key to write; phase namespace or None)."""

    from mechanistic_systems.measurement.device_route_moments import census_v6_cache_keys

    _, write_key = census_v6_cache_keys(cache_key)
    return (write_key,), write_key, None



def _load_or_measure_hard_partition_field(
    *,
    cache_key: str,
    partition_geometry_cache_dir: str | Path,
    runtime: Any,
    partition_family: str,
    measure: Callable[[], tuple[Any, ...]],
    prefetch: VerifiedNpzPrefetch | None = None,
) -> tuple[Any, Any, Any, Any, Any, dict[str, Any]]:
    """Load or measure the current dense census cache entry."""

    from mechanistic_systems.measurement.device_route_moments import (
        CENSUS_V6_CACHE_NAMESPACE,
        census_v6_cache_keys,
        deferred_census_moment_hashes,
    )

    if not _census_cache_files_enabled():
        # No file is read or written; the receipt's moment hashes stay
        # pending until the walk tail joins them (``_walk_tail_block``).
        if prefetch is not None:
            prefetch.abandon()
        with deferred_census_moment_hashes():
            result = measure()
        result[5].update(
            {
                "dense_field_cache_key": cache_key,
                "dense_field_cache_hit": False,
                "dense_field_cache_files": "off",
                "dense_field_cache_payload_sha256": None,
            }
        )
        return result
    _, write_key = census_v6_cache_keys(cache_key)
    if prefetch is not None:
        prefetch.discard()
    receipt_extra = {
        "dense_field_cache_key": write_key,
        "dense_field_cache_namespace": CENSUS_V6_CACHE_NAMESPACE,
    }
    return _load_or_measure_hard_partition_field_at_key(
        cache_key=write_key,
        partition_geometry_cache_dir=partition_geometry_cache_dir,
        runtime=runtime,
        partition_family=partition_family,
        measure=measure,
        prefetch=None,
        receipt_extra=receipt_extra,
    )


def _read_hard_partition_cache_entry(
    *,
    cache_key: str,
    partition_geometry_cache_dir: str | Path,
    runtime: Any,
    partition_family: str,
    miss_reasons: list[dict[str, Any]],
    prefetch: VerifiedNpzPrefetch | None = None,
) -> tuple[Any, Any, Any, Any, Any, dict[str, Any]] | None:
    """Read one current census entry under its lock without rewriting it."""

    import fcntl

    try:
        cache_root = Path(partition_geometry_cache_dir)
        payload_path = cache_root / f"{cache_key}.npz"
        receipt_path = cache_root / f"{cache_key}.json"
        if not payload_path.exists() or not receipt_path.exists():
            return None
        lock_path = cache_root / f"{cache_key}.lock"
        with lock_path.open("a+b") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            try:
                return _load_hard_partition_cache(
                    payload_path=payload_path,
                    receipt_path=receipt_path,
                    cache_key=cache_key,
                    runtime=runtime,
                    prefetch=prefetch,
                )
            finally:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
    finally:
        # An unconsumed prefetch releases its arrays before any fresh census.
        if prefetch is not None:
            prefetch.discard()


def _load_or_measure_hard_partition_field_at_key(
    *,
    cache_key: str,
    partition_geometry_cache_dir: str | Path,
    runtime: Any,
    partition_family: str,
    measure: Callable[[], tuple[Any, ...]],
    prefetch: VerifiedNpzPrefetch | None = None,
    receipt_extra: Mapping[str, Any] | None = None,
) -> tuple[Any, Any, Any, Any, Any, dict[str, Any]]:
    """Load or measure one current cache key under its file lock."""

    import fcntl

    from mechanistic_systems.measurement.device_route_moments import (
        deferred_census_moment_hashes,
    )

    cache_root = Path(partition_geometry_cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    payload_path = cache_root / f"{cache_key}.npz"
    receipt_path = cache_root / f"{cache_key}.json"
    lock_path = cache_root / f"{cache_key}.lock"
    _progress(
        "hard_partition_field_cache_waiting",
        dense_field_cache_key=cache_key,
        lock_path=str(lock_path),
    )
    with lock_path.open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        _progress(
            "hard_partition_field_cache_lock_acquired",
            dense_field_cache_key=cache_key,
        )
        try:
            cached = _load_hard_partition_cache(
                payload_path=payload_path,
                receipt_path=receipt_path,
                cache_key=cache_key,
                runtime=runtime,
                prefetch=prefetch,
            )
        finally:
            # Release a prefetch that the loader did not consume.
            if prefetch is not None:
                prefetch.discard()
        if cached is None:
            # The store joins the moment hashes after it writes the payload.
            with deferred_census_moment_hashes():
                (
                    partition_local_means,
                    partition_global_means,
                    partition_global_activity,
                    partition_ordering_statistics,
                    partition_flip_margin_geometry,
                    hard_partition_measurement,
                ) = measure()
            hard_partition_measurement.update(
                {
                    "dense_field_cache_key": cache_key,
                    "dense_field_cache_hit": False,
                    **dict(receipt_extra or {}),
                }
            )
            payload_sha256 = _store_hard_partition_cache(
                payload_path=payload_path,
                receipt_path=receipt_path,
                cache_key=cache_key,
                runtime=runtime,
                local_by_layer=partition_local_means,
                global_by_layer=partition_global_means,
                global_activity_by_layer=(partition_global_activity),
                ordering_statistics_by_layer=(partition_ordering_statistics),
                flip_margin_geometry=(partition_flip_margin_geometry),
                measurement_receipt=hard_partition_measurement,
            )
            # The store hashes every byte it writes; the file is not reread.
            hard_partition_measurement["dense_field_cache_payload_sha256"] = payload_sha256
            result = (
                partition_local_means,
                partition_global_means,
                partition_global_activity,
                partition_ordering_statistics,
                partition_flip_margin_geometry,
                hard_partition_measurement,
            )
        else:
            result = cached
            cached = None
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
    return result


CATEGORY_CENSUS_MEMO_SCHEMA = "irg_category_census_memo_v1"


def _measure_census_deferred(measure: Callable[[], tuple[Any, ...]]) -> tuple[Any, ...]:
    from mechanistic_systems.measurement.device_route_moments import deferred_census_moment_hashes

    with deferred_census_moment_hashes():
        return measure()


def _category_census_files_enabled() -> bool:
    return _census_cache_files_enabled()


def _join_pending_census_moment_hashes(receipt: dict[str, Any]) -> None:
    """Complete B's deferred census hashes before this receipt is serialized."""

    if "census_moment_hashes_pending" in receipt:
        from mechanistic_systems.measurement.device_route_moments import join_census_moment_hashes

        join_census_moment_hashes(receipt)


def _prefetch_category_census(
    cache_key: str, geometry_dir: str | Path | None, memo: Any, reference_path: Any,
) -> VerifiedNpzPrefetch | None:
    if (
        not _category_census_files_enabled()
        or geometry_dir is None
        or reference_path is not None
        or (memo is not None and memo.census)
    ):
        return None
    current_key = _hard_partition_cache_keys(cache_key)[0][0]
    return VerifiedNpzPrefetch(Path(geometry_dir) / f"{current_key}.npz")


def _shared_worker_process_context_block() -> dict[str, Any]:
    from . import category_memo

    context = category_memo.process_context()
    return {} if context is None else {"process_context": context}


def _category_memo_census(
    *,
    memo: Any,
    cache_key: str,
    shared_key: str,
    partition_family: str,
    partition_geometry_cache_dir: str | Path | None,
    runtime: Any,
    prefetch: VerifiedNpzPrefetch | None,
    measure: Callable[[], tuple[Any, ...]],
    family_view: Callable[[Mapping[str, Any]], tuple[tuple[Any, ...] | None, dict[str, Any]]],
) -> tuple[Any, Any, Any, Any, Any, dict[str, Any]]:
    """One census per category for every arm, held in memory.

    ``measure`` acquires P's registered census, including when another arm
    runs first. Only the complete shared dependency key selects acquisitions.
    Each consumer evaluates its labels and certificate on that fixed field;
    an unresolved certificate remains a diagnostic and never acquires more
    contexts. Existing common-key files may be read, but this worker writes
    no census files. With files off even the payload digest is omitted.
    """

    from . import category_memo

    if not isinstance(shared_key, str) or not shared_key:
        raise ValueError("shared census requires its dependency key")
    files_enabled = _category_census_files_enabled()
    memo_key = str(shared_key)

    def view(entry: Mapping[str, Any], source: str, **extra: Any) -> tuple[Any, ...]:
        result = entry["result"]
        if partition_family != "local_attribution_sign":
            consumer_result, rule = family_view(entry)
            # A partial certificate evaluator may lack an input, but the
            # registered walk still consumes the acquired field as it stands.
            if consumer_result is None:
                looks = result[5].get("panel_receipts") or ()
                rule = {**rule, "order_certificate_resolved": False,
                        "unresolved_at_shared_stop": True,
                        "shared_stop_look": int(looks[-1].get("look_index", len(looks) - 1)) if looks else None}
                certificate = {"schema": "irg_unresolved_arm_order_certificate_v1",
                               "decision_resolved": False, "reasons": list(rule.get("reasons") or ())}
                consumer_result = (*result[:5], {
                    **result[5], "prune_order_uncertainty": certificate,
                    "arm_census_certificate": {**rule, "certificate": certificate},
                })
            result = consumer_result
            extra["family_stopping_rule"] = rule
        result = category_memo.container_copy(result)
        receipt = dict(result[5])
        receipt.update(
            {
                "dense_field_cache_key": cache_key,
                "dense_field_cache_hit": files_enabled and source != "measured_in_memory",
                "dense_field_cache_payload_sha256": entry["payload_sha256"],
                **({} if files_enabled else {"dense_field_cache_files": "off"}),
                "category_census_memo": {
                    "schema": CATEGORY_CENSUS_MEMO_SCHEMA,
                    "source": source,
                    "producer_partition_family": entry["family"],
                    "producer_cache_key": entry["cache_key"],
                    "shared_census_key": str(shared_key),
                    "producer_source": entry["source"],
                    "payload_sha256": entry["payload_sha256"],
                    "payload_hash_kind": entry["payload_hash_kind"],
                    "files_written": False,
                    "stopping_surface": "P",
                    "shared_by_construction": [
                        "context_count", "stop_look", "stopping_reason", "moment_hash",
                    ],
                    **extra,
                },
            }
        )
        return (*result[:5], receipt)

    def discard_prefetch() -> None:
        if prefetch is not None:
            prefetch.discard()

    exact = memo.census.get(memo_key)
    if exact is not None:
        discard_prefetch()
        source = "memo_same_key" if partition_family == "local_attribution_sign" else "memo_shared_census"
        memo.note(product="census", source=source, family=partition_family)
        return view(exact, source)
    result = None
    source = "measured_in_memory"
    if files_enabled and partition_geometry_cache_dir is not None:
        read_keys, _, _ = _hard_partition_cache_keys(cache_key)
        miss_reasons: list[dict[str, Any]] = []
        for read_key in read_keys:
            result = _read_hard_partition_cache_entry(
                cache_key=read_key,
                partition_geometry_cache_dir=partition_geometry_cache_dir,
                runtime=runtime,
                partition_family="local_attribution_sign",
                miss_reasons=miss_reasons,
                prefetch=prefetch,
            )
            if result is not None:
                source = "cache_entry_read"
                break
    discard_prefetch()
    if result is None:
        from mechanistic_systems.measurement.device_route_moments import deferred_census_moment_hashes

        with deferred_census_moment_hashes():
            result = measure()
        payload_sha256 = category_memo.arrays_content_hash(result[:5]) if files_enabled else None
        payload_hash_kind = "in_memory_field_sha256" if files_enabled else None
    else:
        payload_sha256 = str(result[5]["dense_field_cache_payload_sha256"])
        payload_hash_kind = "cache_file_sha256"
    category_memo.freeze_arrays(result[:5])
    entry = {
        "family": "local_attribution_sign",
        "cache_key": cache_key,
        "shared_key": str(shared_key),
        "source": source,
        "result": result,
        "payload_sha256": payload_sha256,
        "payload_hash_kind": payload_hash_kind,
    }
    memo.census[memo_key] = entry
    memo.shared_census = entry
    memo.note(product="census", source=source, family=partition_family)
    return view(entry, source)


def _family_census_on_shared(
    shared: Mapping[str, Any],
    *,
    runtime: Any,
    family: str,
    partition_random_seed: int,
    target_class_id: int,
    sample_size_policy: IndependentSampleSizePolicy,
    required_partition_views: tuple[str, ...],
    v0_attribution_order: str,
    census_profile: Any | None = None,
    census_proposal_route_cap: int | None = None,
    census_schedule_replay_schedule: Mapping[str, Any] | None = None,
) -> tuple[tuple[Any, ...] | None, dict[str, Any]]:
    """Evaluate an arm's order certificate at P's fixed census stop.

    Labels and certificate are the arm's own. Acquisition counts, looks,
    stopping reason and moments remain P's. Insufficient certificate inputs
    or unresolved order uncertainty are reported without changing the field
    or preventing its use by the walk.
    """

    local, global_mean, activity, statistics, flips, receipt = shared["result"]
    looks = receipt.get("panel_receipts") or ()
    stop_look = int(looks[-1].get("look_index", len(looks) - 1)) if looks else None
    reasons: list[str] = []
    order_critical = (receipt.get("analytic_sign_partition") or {}).get("order_normal_critical_value")

    def finish(certificate_receipt: dict[str, Any], *, evaluated: bool) -> tuple[tuple[Any, ...], dict[str, Any]]:
        certificate_receipt = dict(certificate_receipt)
        if reasons:
            certificate_receipt["order_bound_resolved"] = bool(certificate_receipt.get("decision_resolved"))
            certificate_receipt["decision_resolved"] = False
            certificate_receipt["unresolved_reasons"] = list(reasons)
        resolved = bool(certificate_receipt.get("decision_resolved"))
        rule = {
            "evaluated": evaluated,
            "order_certificate_resolved": resolved,
            "order_certificate_hash": content_hash(certificate_receipt, prefix="irg-family-order-certificate-"),
            "shared_stop_look": stop_look,
            "reasons": list(reasons),
            "unresolved_at_shared_stop": not resolved,
        }
        view_receipt = dict(receipt)
        view_receipt["prune_order_uncertainty"] = certificate_receipt
        view_receipt["arm_census_certificate"] = {**rule, "certificate": certificate_receipt}
        return (local, global_mean, activity, statistics, flips, view_receipt), rule

    missing = [name for name in (
        "loss_activity", "loss_activity_standard_error", "loss_mean", "loss_mean_standard_error",
    ) if name not in statistics]
    if family in {"global_attribution_sign", "local_global_sign_alignment"} and "global_mean_standard_error" not in statistics:
        missing.append("global_mean_standard_error")
    if order_critical is None or missing:
        if order_critical is None:
            reasons.append("producer_receipt_lacks_order_critical_value")
        reasons.extend(f"shared_field_lacks_{name}" for name in missing)
        return finish({"schema": "irg_unresolved_arm_order_certificate_v1",
                       "decision_resolved": False, "reasons": list(reasons)}, evaluated=False)
    route_ids, layer_ids = _global_route_axes(runtime)
    offsets = _declared_route_axis_offsets(runtime, flat_length=len(route_ids))

    def flat(values_by_layer: Mapping[str, Any], dtype: Any = np.float64) -> np.ndarray:
        return _declared_flat_from_per_layer(values_by_layer, runtime=runtime, dtype=dtype, offsets=offsets)

    labels = _partition_labels_by_layer(
        runtime,
        measurement=runtime.measurement,
        query_index=0,
        family=family,
        random_seed=int(partition_random_seed),
        local_mean_by_layer=local,
        global_mean_by_layer=global_mean,
        local_destination_scales_by_layer=resnet_local_destination_transport(runtime.evaluator)[0],
        category_id=int(target_class_id),
    )
    producer_certificate = receipt.get("prune_order_uncertainty") or {}
    route_axis_hash = producer_certificate.get("route_axis_hash") or canonical_string_axis_content_hash(
        route_ids, name="prune-order route axis", prefix="prune-order-route-axis-",
    )
    current_activity = flat(statistics["loss_activity"])
    flat_labels = flat(labels, np.int8)
    sign_unresolved = np.zeros(current_activity.size, dtype=bool)
    if family in {"local_attribution_sign", "randomized_count_matched", "local_global_sign_alignment"}:
        measured_by_layer = {
            str(group.layer_id): np.full(group.shape, not (
                _census_analytic_sign_active(family) and _analytic_sign_group(group.route_kind)
            ), dtype=bool)
            for group in runtime.route_groups
        }
        measured = flat(measured_by_layer, bool)
        if np.any(measured):
            local_errors = statistics.get("local_mean_standard_error")
            counts = receipt.get("context_count_by_stratum") or {}
            populations = receipt.get("eligible_population_by_stratum") or {}
            complete = bool(populations) and all(
                name in counts and int(counts[name]) == int(population)
                for name, population in populations.items()
            )
            if local_errors is None and not complete:
                # The unchanged census payload does not expose local errors.
                # Keep those memberships uncertain rather than certifying
                # them from the producer's aggregate sign-risk fraction.
                sign_unresolved |= measured
                reasons.append("local_label_standard_errors_not_recorded")
            elif local_errors is not None:
                sign_critical = float((receipt.get("analytic_sign_partition") or {}).get(
                    "sign_normal_critical_value", order_critical))
                error = flat(local_errors)
                sign_unresolved |= measured & (error > 0) & (np.abs(flat(local)) <= sign_critical * error)
    if family in {"global_attribution_sign", "local_global_sign_alignment"}:
        error = flat(statistics["global_mean_standard_error"])
        # P's unused sign quantile can be a one-coordinate sentinel. A
        # measured global-label consumer claims its own full sign family at
        # the same fixed look, without extending the acquisition.
        final_look = looks[-1] if looks else {}
        alpha = float(receipt.get("alpha_k", (1.0 - float(sample_size_policy.confidence_level))
                                   / ((int(stop_look or 0) + 1) * (int(stop_look or 0) + 2))))
        _, _, _, sign_critical, order_critical = _census_simultaneous_block_critical_values(
            look_confidence=float(final_look.get("look_confidence_level", 1.0 - alpha)),
            route_count=current_activity.size,
            measured_local_sign_route_count=(current_activity.size if family == "local_global_sign_alignment" else 0),
            required_views=_required_partition_views(family),
        )
        sign_unresolved |= (error > 0) & (np.abs(flat(global_mean)) <= sign_critical * error)
    certificate = certify_prune_order_stability(
        device=getattr(runtime.evaluator, "device", "cpu"),
        current_global_activity=current_activity,
        previous_global_activity=None,
        current_partition_labels=flat_labels,
        previous_partition_labels=None,
        layer_ids=layer_ids,
        maximum_mask_disagreement_activity_fraction=float(
            sample_size_policy.maximum_prune_order_uncertainty_fraction
        ),
        scope="layerwise",
        route_axis_hash=route_axis_hash,
        audited_retained_fractions_by_scope=_v0_executable_boundary_fractions(
            runtime.route_groups,
            profile=census_profile,
            partition_route_counts_by_layer=_census_partition_route_counts(flat_labels, offsets),
            proposal_route_cap=census_proposal_route_cap,
            census_schedule_replay_schedule=census_schedule_replay_schedule,
        ),
        activity_normalization="global_route_field",
        score_standard_error=flat(statistics["loss_activity_standard_error"]),
        route_damage=flat(statistics["loss_mean"]),
        route_damage_standard_error=flat(statistics["loss_mean_standard_error"]),
        critical_value=float(order_critical),
        consequence_materiality_floor=_census_materiality_floor(
            receipt.get("census_materiality_resolution")
        ),
        consequence_gated=bool(sample_size_policy.prune_order_consequence_gated),
        partition_sign_unresolved=sign_unresolved,
        consequence_price_short_circuit_scope=None,
    )
    certificate_receipt = certificate.to_receipt()
    if not certificate.resolved:
        reasons.append("order_certificate_unresolved_at_shared_stop")
    return finish(certificate_receipt, evaluated=True)


def _randomized_partition_receipt(
    labels_by_layer: Mapping[str, np.ndarray],
    *,
    runtime: Any,
    partition_random_seed: int,
    category_id: int,
) -> dict[str, Any]:
    """How the RP labels were drawn: RNG key construction and count source."""

    return {
        "schema": "v0_randomized_count_matched_partition_receipt_v2",
        "randomization_algorithm": "numpy_pcg64_permutation_v1",
        "seed_material": _RANDOMIZED_PARTITION_SEED_MATERIAL,
        "seed_material_fields": [
            "partition_random_seed",
            "category_id",
            "route_group_index",
        ],
        "partition_random_seed": int(partition_random_seed),
        "category_id": int(category_id),
        "category_id_source": "target_class_id",
        "route_group_index_order": [
            str(group.layer_id) for group in runtime.route_groups
        ],
        "analytic_sign_scope": _analytic_sign_scope(),
        "analytic_sign_partition": bool(_analytic_sign_mode()),
        "count_source": _RANDOMIZED_PARTITION_COUNT_SOURCE_P_LABELS,
        "negative_count_by_layer": {
            str(group.layer_id): int(
                np.count_nonzero(np.asarray(labels_by_layer[str(group.layer_id)]) == 1)
            )
            for group in runtime.route_groups
        },
    }


def _partition_labels_by_layer(
    runtime: Any,
    *,
    measurement: Any,
    query_index: int,
    family: PilotPartitionFamily,
    random_seed: int,
    local_mean_by_layer: Mapping[str, np.ndarray] | None = None,
    global_mean_by_layer: Mapping[str, np.ndarray] | None = None,
    local_destination_scales_by_layer: Mapping[str, np.ndarray] | None = None,
    category_id: int | None = None,
) -> dict[str, np.ndarray]:
    structural = (
        {
            str(group.layer_id): np.where(
                np.asarray(local_mean_by_layer[str(group.layer_id)]).reshape(
                    group.shape
                )
                >= 0.0,
                0,
                1,
            ).astype(np.int8)
            for group in runtime.route_groups
        }
        if local_mean_by_layer is not None
        else {
            str(key): np.asarray(value, dtype=np.int8)
            for key, value in runtime.partition_labels_by_layer.items()
        }
    )
    if family == "single_population":
        # The k=1 family — every route in the nonnegative-byte
        # class; the walker's uniform-polarity path runs each layer as
        # a single queue and excludes the empty partition entirely.
        return {
            group.layer_id: np.zeros(group.shape, dtype=np.int8)
            for group in runtime.route_groups
        }
    if family == "local_attribution_sign":
        if local_mean_by_layer is None:
            raise ValueError(
                "local-attribution sign requires measured local attribution means"
            )
        if _analytic_sign_mode():
            _analytic_sign_scope()
            if local_destination_scales_by_layer is None:
                raise ValueError(
                    "analytic sign partition requires frozen destination scales"
                )
            model = getattr(getattr(runtime, "loaded", None), "model", None)
            if model is None:
                raise ValueError(
                    "analytic sign partition requires the bound loaded model"
                )
            labels: dict[str, np.ndarray] = {}
            for group in runtime.route_groups:
                layer_id = str(group.layer_id)
                if not _analytic_sign_group(group.route_kind):
                    labels[layer_id] = structural[layer_id].copy()
                    continue
                module = model.get_submodule(str(group.module_path))
                weight = getattr(module, "weight", None)
                if weight is None:
                    raise ValueError(
                        "analytic sign partition requires an explicit weight "
                        f"tensor for {layer_id}"
                    )
                values = weight.detach().cpu().numpy()
                if tuple(values.shape) != tuple(group.shape):
                    raise ValueError(
                        "analytic sign partition shape mismatch on "
                        f"{layer_id}: weight {tuple(values.shape)} versus "
                        f"route group {tuple(group.shape)}"
                    )
                if not np.isfinite(values).all():
                    raise ValueError(
                        f"analytic sign partition found nonfinite weights on {layer_id}"
                    )
                destination_scale = np.asarray(
                    local_destination_scales_by_layer.get(layer_id),
                    dtype=np.float64,
                )
                if destination_scale.shape != (int(values.shape[0]),) or not (
                    np.isfinite(destination_scale).all()
                ):
                    raise ValueError(
                        "analytic sign partition destination scale mismatch on "
                        f"{layer_id}"
                    )
                effective_weight = values * destination_scale.reshape(
                    (int(values.shape[0]),) + (1,) * (int(values.ndim) - 1)
                )
                labels[layer_id] = np.where(effective_weight >= 0.0, 0, 1).astype(
                    np.int8
                )
            return labels
        return {key: value.copy() for key, value in structural.items()}
    if family == "historical_spatial_weight_sum_sign":
        return {
            str(key): np.asarray(value, dtype=np.int8).copy()
            for key, value in runtime.partition_labels_by_layer.items()
        }
    if family == "global_attribution_sign":
        if global_mean_by_layer is None:
            raise ValueError(
                "global-attribution sign requires fused global response means"
            )
        return {
            group.layer_id: np.where(
                np.asarray(global_mean_by_layer[group.layer_id]).reshape(group.shape)
                >= 0.0,
                0,
                1,
            )
            .astype(np.int8)
            .reshape(group.shape)
            for group in runtime.route_groups
        }
    if family == "local_global_sign_alignment":
        if local_mean_by_layer is None or global_mean_by_layer is None:
            raise ValueError(
                "local-global sign alignment requires both measured local "
                "attribution means and fused global response means"
            )
        # Rung-1.5 of the commensurability ladder: the 2x2 refinement of
        # local x global orientation. Byte convention extends the pinned
        # k=2 rule (nonnegative -> 0) lexicographically by (local, global):
        #   0 = local>=0, global>=0   (aligned excitatory)
        #   1 = local>=0, global<0    (MISALIGNED: locally excitatory,
        #                              globally harmful -- the class the
        #                              theory distinguishes)
        #   2 = local<0,  global>=0   (misaligned mirror)
        #   3 = local<0,  global<0    (aligned inhibitory)
        # Both fields are already measured; only the labelling is new.
        #
        # NOT YET EXECUTABLE BY THE v0 WALK, and deliberately so. The walk
        # chassis is hard-wired to k=2 and fails CLOSED on anything else:
        # masking.py refuses labels outside {0, 1} ("partition labels must
        # be integer zero/one") and admits only partition_id "positive" or
        # "negative". These 4-class labels are therefore refused at mask
        # preparation rather than silently mis-partitioned -- the correct
        # failure mode. This family is registered here as the DECLARED
        # rung-1.5 object (the k-class contract that
        # PARTITION_COMMENSURABILITY_THEORY names as the designated
        # refinement, whose "misaligned" cells are its scientifically
        # distinguished class) so the measurement contract and the label
        # convention are pinned NOW, while both fields are already
        # co-measured for free. Executing it needs the k>2 walk contract
        # (chart ids {0..k-1}), which is v1 work under its
        # own registration, outside v0.
        alignment: dict[str, np.ndarray] = {}
        for group in runtime.route_groups:
            layer_id = str(group.layer_id)
            local_negative = (
                np.asarray(local_mean_by_layer[layer_id]).reshape(group.shape) < 0.0
            )
            global_negative = (
                np.asarray(global_mean_by_layer[layer_id]).reshape(group.shape) < 0.0
            )
            alignment[group.layer_id] = (
                2 * local_negative.astype(np.int8) + global_negative.astype(np.int8)
            ).astype(np.int8)
        return alignment
    if family != "randomized_count_matched":
        raise ValueError("unknown pilot partition family")
    if category_id is None:
        # v5: fail closed rather than fall back to the category-free key,
        # which gave every category the same partition.
        raise ValueError(
            "randomized_count_matched labels require the category id that keys "
            "each category's independent draw"
        )
    count_labels = _partition_labels_by_layer(
        runtime,
        measurement=measurement,
        query_index=query_index,
        family="local_attribution_sign",
        random_seed=random_seed,
        local_mean_by_layer=local_mean_by_layer,
        global_mean_by_layer=global_mean_by_layer,
        local_destination_scales_by_layer=local_destination_scales_by_layer,
    )
    randomized: dict[str, np.ndarray] = {}
    for group_index, group in enumerate(runtime.route_groups):
        original = count_labels[str(group.layer_id)].reshape(-1)
        negative_count = int(np.sum(original == 1))
        order = np.random.default_rng(
            np.random.SeedSequence(
                (int(random_seed), int(category_id), int(group_index))
            )
        ).permutation(original.size)
        labels = np.zeros(original.size, dtype=np.int8)
        labels[order[:negative_count]] = 1
        randomized[group.layer_id] = labels.reshape(group.shape)
    return randomized


# Registered dense-census sample design.
#
# The legacy design draws response_initial_packet (32) contexts per stratum in
# one look, the off-target 32 uniform over every other training image.  Its
# materiality floor is in whole-loss units, so the order certificate passes at
# that first look and the fc rankings are not reproducible across seeds.  The
# registered design measures the whole on-target stratum and a class-stratified
# off-target draw (m contexts from every other class, so off-target classes
# carry equal weight inside the off-target half), both read from the SAME
# sealed per-stratum permutations (run_id, stratum, seed), in one first look.
# The 0.5/0.5 stratum weights, the walk and its decision law are unchanged.
CENSUS_SAMPLE_DESIGN_ENV = "IRG_CENSUS_SAMPLE_DESIGN"
CENSUS_OFF_TARGET_PER_CLASS_ENV = "IRG_CENSUS_OFF_TARGET_PER_CLASS"
CENSUS_SAMPLE_DESIGN_LEGACY = "uniform_initial_packet_v1"
CENSUS_SAMPLE_DESIGN_FULL_TARGET_CLASS_STRATIFIED = (
    "full_target_class_stratified_off_target_v1"
)
CENSUS_OFF_TARGET_PER_CLASS_DEFAULT = 2
# Batch-sequential importance census
# (census_batches.py): all on-target images exact; off-target images in fixed
# batches of b* drawn with replacement with q_i proportional to a_i |p_i - y_i|
# (fixed for the whole run); Hansen-Hurwitz batch sums; stop over all scopes
# after every batch.
CENSUS_SAMPLE_DESIGN_IMPORTANCE_BATCHES = "importance_batches_v1"
CENSUS_BATCH_ENV = {
    "batch_size": "IRG_CENSUS_BATCH_SIZE",
    "logits_dir": "IRG_CENSUS_LOGITS_DIR",
    "logits_row_hashes": "IRG_CENSUS_LOGITS_ROW_HASHES",
    "stop_band_table": "IRG_CENSUS_STOP_BAND_TABLE",
    "eps": "IRG_CENSUS_PRECISION_EPS",
    "cut_ranges": "IRG_CENSUS_CUT_RANGES",
    "shared_categories": "IRG_CENSUS_SHARED_CATEGORIES",
}
# One confidence parameter: alpha = 1 - the walk's registered
# confidence level, bound per run; eps is the one precision knob.  The check
# schedule (first at 2,048 off-target draws, then every >= 10% growth) is fixed
# in advance, so alpha spending per check is valid.
CENSUS_BATCH_DEFAULTS = {
    "first_check_draws": 2048,
    "check_growth": 0.50,
    "eps": 0.5,
    # Exact switch: not passed when the draws reach N_off/3 ->
    # enumerate every off-target image exactly (worst case ~1.33 N, not 2 N).
    "exact_switch_fraction": 1.0 / 3.0,
}


def _census_batch_design_setting() -> dict[str, Any]:
    """The importance-batch design dict (cache key and receipts).

    Binds b*, the stop rule and its parameters, and the identities (sha256)
    of the class-major dense-logit copy, its row hashes and the stop-band
    table, so a change of any input mints a new census identity.
    """

    from .census_inputs import ClassMajorLogits

    def raw(name: str) -> str | None:
        value = os.environ.get(CENSUS_BATCH_ENV[name])
        return None if value is None or not str(value).strip() else str(value).strip()

    from .census_stop import IZ_STOP_RULE

    if raw("batch_size") is None:
        # b* is the kernel's saturation batch, registered per network and GPU
        # (memory caps it: AlexNet 128, VGG-11 64, ResNet-18 32 on H100); no default.
        raise ValueError("the importance-batch census needs IRG_CENSUS_BATCH_SIZE (b*, per network and GPU)")
    batch_size = int(raw("batch_size"))
    capture_rows = os.environ.get("IRG_CENSUS_CAPTURE_ROWS")
    if capture_rows is None or not str(capture_rows).strip() or int(capture_rows) < 1:
        # The physical capture packet changes TF32 kernel choice; it is registered per network and GPU (no default).
        raise ValueError("the importance-batch census needs IRG_CENSUS_CAPTURE_ROWS (per network and GPU)")
    eps = float(raw("eps") or CENSUS_BATCH_DEFAULTS["eps"])
    if batch_size < 1 or not eps >= 0.0:
        raise ValueError("census batch size must be >= 1 and the precision eps >= 0")
    logits_dir, row_hashes = raw("logits_dir"), raw("logits_row_hashes")
    if logits_dir is None or row_hashes is None:
        raise ValueError(
            "the importance-batch census needs IRG_CENSUS_LOGITS_DIR and IRG_CENSUS_LOGITS_ROW_HASHES"
        )
    logits = ClassMajorLogits.open(logits_dir, row_hashes)
    cut_ranges: dict[str, Any] | None = None
    shared_categories: dict[str, Any] | None = None
    if eps != 0.0:
        # The sampled stop's scopes and certified cuts; eps = 0 (exact census) reads neither, and they stay out
        # of its census identity.
        ranges_path, shared_path = raw("cut_ranges"), raw("shared_categories")
        if ranges_path is None or shared_path is None:
            raise ValueError(
                "the importance-batch census with eps > 0 needs IRG_CENSUS_CUT_RANGES and IRG_CENSUS_SHARED_CATEGORIES"
            )
        ranges_payload = Path(ranges_path).read_bytes()
        ranges_doc = json.loads(ranges_payload)
        shared_payload = Path(shared_path).read_bytes()
        shared = sorted({int(token) for token in shared_payload.decode().split()})
        cut_ranges = {
            "sha256": hashlib.sha256(ranges_payload).hexdigest(),
            "network": ranges_doc.get("network"),
            "arms": {
                arm: {layer: {part: {"min_kept": float(v["min_kept"])} for part, v in parts.items()}
                      for layer, parts in layers.items()}
                for arm, layers in dict(ranges_doc.get("arms") or {}).items()
            },
        }
        shared_categories = {"sha256": hashlib.sha256(shared_payload).hexdigest(), "ids": shared}
    return {
        "schema": "irg_census_sample_design_v3",
        "design": CENSUS_SAMPLE_DESIGN_IMPORTANCE_BATCHES,
        "on_target": "take_all_exact",
        "off_target": "with_replacement_batches_q_proportional_to_a_times_abs_p_minus_y",
        "class_weights": ("one_over_training_population_size" if loss_weighting() == "per_image"
                          else "on_0.5_over_N_on__off_0.5_over_C_off_N_class"),
        **({"loss_weighting": "per_image"} if loss_weighting() == "per_image" else {}),
        "weights": "fixed_for_the_run",
        "estimator": "hansen_hurwitz_batch_sums",
        "variance": "within_sample_over_all_draws_nu_n_minus_1",
        "draws": "philox_keyed_by_run_id_seed_batch_index",
        "batch_size": batch_size,
        "capture_rows": int(capture_rows),
        "check_schedule": {
            "first_check_draws": int(CENSUS_BATCH_DEFAULTS["first_check_draws"]),
            "check_growth": float(CENSUS_BATCH_DEFAULTS["check_growth"]),
        },
        "exact_switch_fraction": float(CENSUS_BATCH_DEFAULTS["exact_switch_fraction"]),
        "stop_rule": IZ_STOP_RULE,
        "stop_parameters": {
            "eps": eps,
            "delta": "eps * tau * positive_loss_span",
            "alpha": "1 - confidence_level (bound per run)",
            "alpha_spending": "alpha/((j+1)(j+2)) at check j",
            "tail": "student_t_nu_n_minus_1",
            "cuts": "executable walk-ladder cuts with k/N in [0.85 x min observed kept fraction, 1] per arm, layer, partition",
            "scopes": "P's sign partitions; on shared categories also SA-H's layer and RP1's partitions",
        },
        "scopes": "every_layer_whole_plus_analytic_sign_partitions",
        "population_limit": "draws_reach_off_target_population",
        # Content identities only: input paths are read from the environment
        # by the measurement and checked against these.  The stop-band table is
        # a receipt diagnostic only and stays out of the census identity; the cut
        # ranges and the shared-category list define the stop and are embedded.
        "logits": logits.identity(),
        "cut_ranges": cut_ranges,
        "shared_categories": shared_categories,
    }


def census_sample_design_setting() -> dict[str, Any] | None:
    """Resolve the registered census sample design from the environment.

    Default: the full-target, class-stratified design with m = 2.  ``None``
    means the legacy design; it adds nothing to the cache key or receipts, so
    a legacy run is byte-identical to the production engine.
    """

    raw = os.environ.get(CENSUS_SAMPLE_DESIGN_ENV)
    name = (
        CENSUS_SAMPLE_DESIGN_FULL_TARGET_CLASS_STRATIFIED
        if raw is None or not str(raw).strip()
        else str(raw).strip()
    )
    if name == CENSUS_SAMPLE_DESIGN_LEGACY:
        return None
    if name == CENSUS_SAMPLE_DESIGN_IMPORTANCE_BATCHES:
        return _census_batch_design_setting()
    if name != CENSUS_SAMPLE_DESIGN_FULL_TARGET_CLASS_STRATIFIED:
        raise ValueError(
            f"{CENSUS_SAMPLE_DESIGN_ENV} must be "
            f"{CENSUS_SAMPLE_DESIGN_IMPORTANCE_BATCHES!r}, "
            f"{CENSUS_SAMPLE_DESIGN_FULL_TARGET_CLASS_STRATIFIED!r} or "
            f"{CENSUS_SAMPLE_DESIGN_LEGACY!r}"
        )
    raw_m = os.environ.get(CENSUS_OFF_TARGET_PER_CLASS_ENV)
    per_class = (
        CENSUS_OFF_TARGET_PER_CLASS_DEFAULT
        if raw_m is None or not str(raw_m).strip()
        else int(str(raw_m).strip())
    )
    if per_class < 1:
        raise ValueError(f"{CENSUS_OFF_TARGET_PER_CLASS_ENV} must be >= 1")
    return {
        "schema": "irg_census_sample_design_v1",
        "design": name,
        "on_target": "whole_stratum",
        "off_target": ("uniform_without_replacement_over_images" if loss_weighting() == "per_image"
                       else "class_stratified_equal_count_per_class"),
        "off_target_per_class": int(per_class),
        "draw_source": "sealed_per_stratum_permutation_run_id_stratum_seed",
        "off_target_order": ("sealed_uniform_image_permutation" if loss_weighting() == "per_image"
                             else "within_class_permutation_rank_then_permutation_position"),
        "stratum_weights": ("training_population_shares" if loss_weighting() == "per_image"
                            else {"on_target": 0.5, "off_target": 0.5}),
        "off_target_class_weights": ("population_image_counts" if loss_weighting() == "per_image" else "equal"),
        **({"loss_weighting": "per_image"} if loss_weighting() == "per_image" else {}),
        "first_look": "whole_registered_draw",
    }


def _census_batch_design_guard(
    design: Mapping[str, Any] | None,
    *,
    family: str,
    proposal_route_cap: int | None,
    schedule_replay: Mapping[str, Any] | None,
    census_reference_path: Any,
) -> None:
    """Refuse, at run start, the configurations the importance-batch census does not implement.

    It serves P's analytic weight-sign partitions on the nominal walk ladder,
    computed in-run: measured-sign partitions, capped or replayed ladders, the
    global-mean label families, campaign census references and census file
    caching are refused rather than silently mis-served.
    """

    if design is None or design.get("design") != CENSUS_SAMPLE_DESIGN_IMPORTANCE_BATCHES:
        return
    problems = []
    if str(family) in {"global_attribution_sign", "local_global_sign_alignment"}:
        problems.append(f"family {family} labels from the global mean, which the batch census does not estimate fully")
    if not _analytic_sign_mode() or _analytic_sign_scope() != _ANALYTIC_SIGN_SCOPE_ALL:
        problems.append("the analytic weight-sign partition over all supported routes must be on")
    if proposal_route_cap is not None:
        problems.append("capped ladders (proposal route cap) are not certified")
    if schedule_replay is not None:
        problems.append("replayed census schedules are not certified")
    if census_reference_path is not None:
        problems.append("campaign census references are not built for the batch census")
    if _census_cache_files_enabled():
        problems.append("census file caching cannot store the batch census (IRG_CENSUS_CACHE_FILES must be 0)")
    if problems:
        raise ValueError("importance-batch census refused: " + "; ".join(problems))


def _census_design_for_run(
    design: Mapping[str, Any] | None,
    *,
    sensitivity_tau: float,
    loss_span_certificate: Any,
    confidence_level: float,
) -> dict[str, Any] | None:
    """Bind the run's tau and positive loss span into the importance-batch design.

    The indifference zone is delta = eps * tau * span, so tau_S belongs to the
    census's identity.  Other designs are returned unchanged (their cache keys
    and receipts do not move).
    """

    if design is None or design.get("design") != CENSUS_SAMPLE_DESIGN_IMPORTANCE_BATCHES:
        return None if design is None else dict(design)
    if loss_span_certificate is None:
        raise ValueError("the importance-batch census needs the run's positive loss span before the census")
    tau = float(sensitivity_tau)
    span = float(loss_span_certificate.estimate)
    alpha = 1.0 - float(confidence_level)
    if not (tau > 0.0 and span > 0.0 and 0.0 < alpha < 1.0):
        raise ValueError("the importance-batch census needs tau > 0, a positive loss span and alpha in (0, 1)")
    return {**dict(design), "tau": tau, "positive_loss_span": span, "tau_S": tau * span, "alpha": alpha}


def _census_context_labels(axis: Any) -> dict[str, int]:
    """Class label of every context on the census axis (context id -> label)."""

    return {str(record.context_id): int(record.label) for record in axis.records}


def _census_registered_draw_orders(
    permutations: Mapping[str, SealedUniformPermutationReceipt],
    *,
    label_by_context: Mapping[str, int],
    off_target_per_class: int,
) -> tuple[dict[str, tuple[str, ...]], dict[str, int], dict[str, Any]]:
    """Draw orders and first-look counts of the registered census design.

    Both strata keep their sealed permutation.  On-target: the permutation
    itself; the first look takes the whole stratum.  Off-target: the
    permutation reordered by (rank of the context inside its class, in
    permutation order; then permutation position).  Its first C*m entries are
    exactly the first m contexts of each of the C off-target classes in
    permutation order -- a uniform draw of m per class without replacement --
    and any later look would continue round by round from the same order.
    Returns ``(draw_order_by_stratum, first_look_count_by_stratum, receipt)``.
    """

    per_class = int(off_target_per_class)
    if per_class < 1:
        raise ValueError("census off-target draw needs at least one context per class")
    on_receipt = permutations["on_target"]
    off_receipt = permutations["off_target"]
    on_receipt.verify()
    off_receipt.verify()
    on_order = tuple(on_receipt.generated_order)
    off_order = off_receipt.generated_order
    if not on_order or not off_order:
        raise ValueError("census sample design has an empty stratum")
    try:
        on_labels = {int(label_by_context[context]) for context in on_order}
        off_labels = np.fromiter(
            (int(label_by_context[context]) for context in off_order),
            dtype=np.int64,
            count=len(off_order),
        )
    except KeyError as error:
        raise ValueError("census context has no class label") from error
    if len(on_labels) != 1:
        raise ValueError("census on-target stratum must hold exactly one class")
    # A stable sort by label keeps permutation order inside every class.
    by_class = np.argsort(off_labels, kind="stable")
    classes, class_starts, class_sizes = np.unique(
        off_labels[by_class], return_index=True, return_counts=True
    )
    if next(iter(on_labels)) in set(classes.tolist()):
        raise ValueError("census on-target class also appears off-target")
    if loss_weighting() == "per_image":
        first_off = min(len(off_order), int(len(classes) * per_class))
        first = {"on_target": len(on_order), "off_target": first_off}
        receipt = {
            "schema": "irg_census_registered_draw_receipt_v1",
            "loss_weighting": "per_image",
            "on_target_class": int(next(iter(on_labels))),
            "on_target_count": len(on_order), "on_target_population": len(on_order),
            "off_target_class_count": int(len(classes)),
            "off_target_per_class": per_class,
            "off_target_count": first_off, "off_target_population": len(off_order),
            "off_target_draw": "uniform_without_replacement_over_images",
            "stratum_weights": _binary_loss_weights({"on_target": on_order, "off_target": off_order}),
            "first_look_context_ids_hash": content_hash(
                {"on_target": on_order, "off_target": tuple(off_order[:first_off])},
                prefix="irg-census-registered-draw-"),
        }
        return {"on_target": on_order, "off_target": tuple(off_order)}, first, receipt
    if int(class_sizes.min()) < per_class:
        raise ValueError(
            "an off-target class has fewer contexts than the per-class draw"
        )
    rank_in_class = np.empty(len(off_order), dtype=np.int64)
    rank_in_class[by_class] = np.arange(len(off_order), dtype=np.int64) - np.repeat(
        class_starts, class_sizes
    )
    draw = np.lexsort((np.arange(len(off_order), dtype=np.int64), rank_in_class))
    off_draw = tuple(off_order[int(position)] for position in draw)
    first_off = int(len(classes) * per_class)
    first = {"on_target": len(on_order), "off_target": first_off}
    receipt = {
        "schema": "irg_census_registered_draw_receipt_v1",
        "on_target_class": int(next(iter(on_labels))),
        "on_target_count": len(on_order),
        "on_target_population": len(on_order),
        "off_target_class_count": int(len(classes)),
        "off_target_per_class": per_class,
        "off_target_count": first_off,
        "off_target_population": len(off_order),
        "off_target_class_size_min": int(class_sizes.min()),
        "off_target_class_size_max": int(class_sizes.max()),
        "first_look_context_ids_hash": content_hash(
            {"on_target": on_order, "off_target": off_draw[:first_off]},
            prefix="irg-census-registered-draw-",
        ),
    }
    return {"on_target": on_order, "off_target": off_draw}, first, receipt


# ---------------------------------------------------------------------------
# Importance-batch census: stop rules, scopes and the measurement hook.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CensusScopeSpec:
    """One stop scope: a layer's declared route block, optionally one partition.

    ``partition`` is None for the whole layer, else the analytic label value
    (0 = positive, 1 = negative) selected by ``labels`` (int8, the block's
    routes in declared order).
    """

    scope_id: str
    layer_id: str
    start: int
    stop: int
    partition: int | None
    labels: Any = None
    arm: str = "P"
    partition_name: str = "positive"

    @property
    def size(self) -> int:
        if self.partition is None:
            return int(self.stop - self.start)
        return int(np.count_nonzero(np.asarray(self.labels) == self.partition))


def _census_scope_specs(
    runtime: Any, *, route_count: int, target_class_id: int, shared_category: bool,
) -> dict[str, CensusScopeSpec]:
    """The stop's scopes: the partitions the arms that run on this category walk.

    P's weight-sign partitions on every category; on the shared-category list
    also SA-H's single population (the whole layer) and RP1's count-matched
    random partitions (partition seed 1, P's counts).  Only analytically signed
    connection groups are scopes; residual-stream gates have no executable cut.
    The set depends on the category and the network only, never on which arms
    a process happens to run, so one seed gives one census.
    """

    offsets = _declared_route_axis_offsets(runtime, flat_length=int(route_count))
    analytic = [
        group for group in runtime.route_groups
        if _census_analytic_sign_active("local_attribution_sign") and _analytic_sign_group(group.route_kind)
    ]
    specs: dict[str, CensusScopeSpec] = {}
    if not analytic:
        return specs
    zeros = {str(group.layer_id): np.zeros(group.shape, dtype=np.float64) for group in runtime.route_groups}
    scales = resnet_local_destination_transport(runtime.evaluator)[0]

    def labels(family: str, seed: int) -> dict[str, np.ndarray]:
        return _partition_labels_by_layer(
            runtime, measurement=runtime.measurement, query_index=0, family=family, random_seed=int(seed),
            local_mean_by_layer=zeros, global_mean_by_layer=None, local_destination_scales_by_layer=scales,
            category_id=int(target_class_id),
        )

    families = [("P", "local_attribution_sign", 0)]
    if shared_category:
        families += [("SAH", None, 0), ("RP1", "randomized_count_matched", 1)]
    for arm, family, seed in families:
        by_layer = None if family is None else labels(family, seed)
        for group in analytic:
            layer = str(group.layer_id)
            start, stop = (int(v) for v in offsets[layer])
            if by_layer is None:
                scope = f"{layer}|{arm}"
                specs[scope] = CensusScopeSpec(scope, layer, start, stop, None, None, arm, "positive")
                continue
            flat = np.asarray(by_layer[layer], dtype=np.int8).reshape(-1)
            for value, name in ((0, "positive"), (1, "negative")):
                if np.any(flat == value):
                    scope = f"{layer}|{arm}:{name}"
                    specs[scope] = CensusScopeSpec(scope, layer, start, stop, value, flat, arm, name)
    return specs


def _census_certified_cuts(
    all_cuts: Mapping[str, Sequence[int]],
    scopes: Mapping[str, CensusScopeSpec],
    ranges: Mapping[str, Any],
) -> tuple[dict[str, list[int]], dict[str, dict[str, Any]]]:
    """The cuts the walk can stop at: executable cuts with k/N in [0.85 x min observed kept, 1] per arm/layer/partition.

    A partition whose observed minimum is 1 (never pruned) keeps only its
    first block's cut.  A partition without observations keeps every
    executable cut (recorded as such).
    """

    cuts: dict[str, list[int]] = {}
    receipt: dict[str, dict[str, Any]] = {}
    for scope, spec in scopes.items():
        ladder = sorted({int(k) for k in all_cuts.get(scope, ())}, reverse=True)
        entry = (((ranges.get(spec.arm) or {}).get(spec.layer_id) or {}).get(spec.partition_name))
        if not ladder:
            cuts[scope] = []
            receipt[scope] = {"source": "no_executable_cut", "cuts": 0}
            continue
        if entry is None:
            cuts[scope] = ladder
            receipt[scope] = {"source": "no_observation_all_executable_cuts", "cuts": len(ladder)}
            continue
        minimum = float(entry["min_kept"])
        if minimum >= 1.0:
            selected = [ladder[0]]
        else:
            low = 0.85 * minimum
            selected = [k for k in ladder if k / spec.size >= low] or [ladder[0]]
        cuts[scope] = selected
        receipt[scope] = {"source": "observed_range", "min_kept": minimum, "f_lo": 0.85 * minimum,
                          "cuts": len(selected), "executable_cuts": len(ladder), "scope_size": int(spec.size),
                          "k_min": min(selected), "f_min": min(selected) / spec.size}
    return cuts, receipt


def _census_walk_cut_counts(
    runtime: Any,
    scopes: Mapping[str, CensusScopeSpec],
    *,
    profile: Any | None,
    proposal_route_cap: int | None,
) -> dict[str, list[int]]:
    """Every executable cut (retained count k, 0 < k < N_s) of the walk's rung ladder, per scope.

    The same rule as _v0_executable_boundary_fractions (the registered rung
    profile, ceil retained counts, no-op rungs removed, the proposal route
    cap's refinement), returned as counts.  Groups the walk does not prune as
    connections (residual-stream gates) have no cut.
    """

    from mechanistic_systems.circuits.frontier_rungs import retained_counts_for_density_schedule
    from .prune_scope_walk import _refined_ladder, executable_ladder_for_kinds

    rung_profile = SimpleNamespace(**_census_rung_profile_receipt(profile))
    kinds = {str(group.layer_id): str(group.route_kind) for group in runtime.route_groups}
    out: dict[str, list[int]] = {}
    for scope, spec in scopes.items():
        kind = kinds[spec.layer_id]
        if kind not in {"linear_weight", "conv_channel_pair", "conv_kernel_weight"}:
            out[scope] = []
            continue
        schedule, _floor = executable_ladder_for_kinds(
            ("linear_like" if kind == "linear_weight" else "conv_like",), rung_profile
        )
        retained = retained_counts_for_density_schedule({"scope": spec.size}, schedule)
        if proposal_route_cap is not None:
            _, retained = _refined_ladder(schedule, retained, int(proposal_route_cap))
        values = tuple(int(v) for v in retained["scope"])
        out[scope] = sorted(
            {later for earlier, later in zip(values, values[1:]) if later < earlier and 0 < later < values[0]},
            reverse=True,
        )
    return out


def _census_batch_stop_rule(design: Mapping[str, Any], cuts: Mapping[str, Sequence[int]]):
    """The registered one-parameter indifference-zone stop over all scopes and walk cuts."""

    from .census_stop import IZ_STOP_RULE, census_iz_stop

    if str(design["stop_rule"]) != IZ_STOP_RULE:
        raise ValueError(f"unregistered census stop rule {design['stop_rule']!r}")
    if "tau_S" not in design or "alpha" not in design:
        raise ValueError("the importance-batch census design lacks the run's tau_S or alpha")

    def stop(fields: Mapping[str, tuple[Any, Any]], *, look: int, n_draws: int,
             population_exhausted: bool) -> dict[str, Any]:
        return census_iz_stop(
            fields, cuts, tau_S=float(design["tau_S"]), alpha=float(design["alpha"]), look=int(look),
            n_draws=int(n_draws), eps=float(design["stop_parameters"]["eps"]),
            population_exhausted=bool(population_exhausted),
        )

    return stop


def _census_certified_range_check(
    measurement: Mapping[str, Any] | None,
    family: str,
    partition_random_seed: int | None,
    group_final_densities: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Flag a walk that stopped below the census's certified range for its arm (receipt only)."""

    design = dict((measurement or {}).get("census_sample_design") or {})
    if design.get("design") != CENSUS_SAMPLE_DESIGN_IMPORTANCE_BATCHES:
        return None
    if float((design.get("stop_parameters") or {}).get("eps", 1.0)) == 0.0:
        return {"applicable": False, "reason": "exact census (eps = 0): every cut is exact; no certified range"}
    ranges = dict((measurement or {}).get("certified_ranges") or {})
    if not ranges:
        return None
    from mechanistic_systems.circuits.masking import _keep_count
    arm = {"local_attribution_sign": "P", "single_population": "SAH"}.get(str(family))
    if str(family) == "randomized_count_matched":
        arm = f"RP{int(partition_random_seed or 0)}"
    outside = []
    checked = 0
    for group in group_final_densities:
        layer = str(group["layer_id"])
        parts = ("positive",) if arm == "SAH" else ("positive", "negative")
        for part in parts:
            scope = f"{layer}|SAH" if arm == "SAH" else f"{layer}|{arm}:{part}"
            entry = ranges.get(scope)
            if entry is None or entry.get("source") != "observed_range":
                continue
            checked += 1
            density = float(group[part])
            # Compare retained COUNTS: the walk reports its nominal rung density, whose
            # count is the engine's ceil rule; the certified range is a count.
            retained = _keep_count(int(entry["scope_size"]), density)
            if retained < int(entry["k_min"]):
                outside.append({"scope": scope, "final_density": density, "retained_count": int(retained),
                                "certified_k_min": int(entry["k_min"])})
    return {"applicable": True, "arm": arm, "scopes_checked": checked, "stopped_below_certified_range": outside,
            "all_inside": not outside, "comparison": "retained_counts"}


CENSUS_FIELD_DUMP_ENV = "IRG_CENSUS_FIELD_DUMP_DIR"


def _census_field_dump(
    ordering_statistics: Mapping[str, Mapping[str, Any]],
    local_means: Mapping[str, Any],
    *,
    run_id: str,
    random_seed: int,
    target_class_id: int,
    design: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Diagnostics only: write the census order field, its SE, the signed loss mean and the local mean per layer.

    Ground truth for evaluating the stop offline.  The folder
    must lie under a ``diagnostics`` directory; files are created exclusively
    and never overwritten.  Not part of the census identity.
    """

    root = os.environ.get(CENSUS_FIELD_DUMP_ENV)
    if root is None or not str(root).strip():
        return None
    folder = Path(root)
    if "diagnostics" not in folder.resolve().parts:
        raise ValueError(f"{CENSUS_FIELD_DUMP_ENV} must lie under a diagnostics folder")
    folder.mkdir(parents=True, exist_ok=True)
    stem = content_hash({"run_id": run_id, "random_seed": int(random_seed), "target": int(target_class_id),
                         "design": dict(design)}, prefix="census-fields-")
    path = folder / f"{stem}.npz"
    arrays: dict[str, np.ndarray] = {}
    for name in ("loss_activity", "loss_activity_standard_error", "loss_mean"):
        for layer, values in dict(ordering_statistics.get(name) or {}).items():
            arrays[f"{name}/{layer}"] = np.asarray(values)
    for layer, values in dict(local_means).items():
        arrays[f"local_mean/{layer}"] = np.asarray(values)
    with path.open("xb") as handle:
        np.savez(handle, **arrays)
    return {"path": str(path), "sha256": sha256_file(path), "arrays": len(arrays)}


def _census_exact_stop(fields: Any, *, look: int, n_draws: int, population_exhausted: bool) -> dict[str, Any]:
    """The eps = 0 stop: passes only on the exact census, which the controller enumerates from the start.

    It never reads the fields (no scope, sort or window is evaluated); a call
    before enumeration is a controller error.
    """

    if not population_exhausted:
        raise RuntimeError("the eps = 0 census stop was evaluated before exact enumeration")
    return {"rule": "exact_census_eps_0", "passed": True, "reason": "population", "statistic": 0.0, "E_total": 0.0,
            "binding_scope": None, "scopes": {}, "look": int(look), "n_draws": int(n_draws)}


def _census_band_diagnostic(report: Mapping[str, Any], scopes: Mapping[str, CensusScopeSpec]) -> dict[str, Any]:
    """Receipt diagnostic: the stop's expected count restricted to the stop-band table's range."""

    from .census_pps import load_band_table

    bands, digest = load_band_table(os.environ.get(CENSUS_BATCH_ENV["stop_band_table"]))
    if not bands:
        return {"table": None}
    restricted = {}
    for scope, record in dict(report.get("scopes") or {}).items():
        band = bands.get(scopes[scope].layer_id) if scope in scopes else None
        if band is None:
            continue
        restricted[scope] = float(sum(c["E"] for c in record["cuts"] if band[0] <= c["f"] <= band[1]))
    return {"table_sha256": digest, "E_by_scope_within_band": restricted,
            "E_total_within_band": float(sum(restricted.values()))}


def _census_batch_session_factory() -> Callable[..., Any]:
    """The kernel worker's weighted fast census (BatchCensusSession plus result())."""

    from mechanistic_systems.measurement import census_sufficient_statistics as css

    factory = getattr(css, "batch_census_session", None)
    if factory is None:
        raise RuntimeError(
            "the importance-batch census needs census_sufficient_statistics.batch_census_session "
            "(the weighted fast census); it is not in this tree"
        )
    return factory


def _measure_dense_batch_census(
    *,
    runtime: Any,
    optimization_axis: Any,
    query: Any,
    dense_mask: HardMask,
    route_ids: Any,
    route_layer_ids: Any,
    target_class_id: int,
    dense_calibration_state: Any,
    pruning_guiding_loss_id: str,
    run_id: str,
    random_seed: int,
    required_partition_views: tuple[str, ...],
    partition_family: str,
    census_sample_design: Mapping[str, Any],
    objective_receipt: Mapping[str, Any],
    census_profile: Any | None = None,
    census_proposal_route_cap: int | None = None,
    decision_fixed_offset_resolver: Callable[..., Any] | None = None,
) -> tuple[Any, ...]:
    """The dense census under the importance-batch design; returns the census 6-tuple."""

    from .census_batches import BatchSampler, run_batch_census
    from .census_inputs import ClassMajorLogits, importance_inputs

    design = dict(census_sample_design)
    logits = ClassMajorLogits.open(
        os.environ[CENSUS_BATCH_ENV["logits_dir"]], os.environ[CENSUS_BATCH_ENV["logits_row_hashes"]]
    )
    if logits.identity() != design["logits"]:
        raise ValueError("census dense logits differ from the registered design's identity")
    table_ids, table_labels = logits.ids_and_labels()
    records = tuple(optimization_axis.records)
    inputs = importance_inputs(
        context_ids=[str(record.context_id) for record in records],
        labels=[int(record.label) for record in records],
        target_logits_by_table_row=logits.target_row(int(target_class_id)),
        table_ids=table_ids, table_labels=table_labels,
        target_class_id=int(target_class_id), calibration=dense_calibration_state,
        **({"fixed_offsets": decision_fixed_offset_resolver(
            tuple(str(record.context_id) for record in records)).detach().cpu().numpy()}
           if decision_fixed_offset_resolver is not None else {}),
    )
    exact_census = float(design["stop_parameters"]["eps"]) == 0.0
    shared_category = (None if exact_census
                       else int(target_class_id) in set(design["shared_categories"]["ids"]))
    if exact_census:
        # eps = 0: the census is enumerated exactly from the start and passes by construction, so no stop scope,
        # cut set, window or label partition is built.
        scopes: dict[str, CensusScopeSpec] = {}
        all_cuts: dict[str, list[int]] = {}
        certified_cuts: dict[str, list[int]] = {}
        certified_ranges: dict[str, dict[str, Any]] = {}
    else:
        scopes = _census_scope_specs(
            runtime, route_count=len(route_ids), target_class_id=int(target_class_id),
            shared_category=shared_category,
        )
        all_cuts = _census_walk_cut_counts(
            runtime, scopes, profile=census_profile, proposal_route_cap=census_proposal_route_cap,
        )
        certified_cuts, certified_ranges = _census_certified_cuts(all_cuts, scopes, design["cut_ranges"]["arms"])
    session = _census_batch_session_factory()(
        runtime=runtime, optimization_axis=optimization_axis, query=query, dense_mask=dense_mask,
        route_ids=route_ids, route_layer_ids=route_layer_ids, target_class_id=int(target_class_id),
        dense_calibration_state=dense_calibration_state, pruning_guiding_loss_id=pruning_guiding_loss_id,
        scopes=scopes,
        **({"decision_fixed_offset_resolver": decision_fixed_offset_resolver}
           if decision_fixed_offset_resolver is not None else {}),
    )
    sampler = BatchSampler(
        inputs.off_target_ids, inputs.off_target_a, run_id=str(run_id), random_seed=int(random_seed),
        content_hash=content_hash,
    )
    loop = run_batch_census(
        session, sampler=sampler, off_target_u=inputs.off_target_u,
        on_target_ids=inputs.on_target_ids, on_target_a=inputs.on_target_a,
        batch_size=int(design["batch_size"]),
        stop_rule=(_census_exact_stop if exact_census else _census_batch_stop_rule(design, certified_cuts)),
        first_check_draws=int(design["check_schedule"]["first_check_draws"]),
        check_growth=float(design["check_schedule"]["check_growth"]),
        population_limit_draws=int(math.ceil(len(inputs.off_target_ids) * float(design["exact_switch_fraction"]))),
        # eps = 0 passes only on the exact census: enumerate from the start, no wasted draws.
        enumerate_exactly=float(design["stop_parameters"]["eps"]) == 0.0,
        refit=None,  # fixed weights for the run; the hook stays a documented no-op
        # _progress keeps only decision events; one stderr line per check makes the census observable.
        progress=lambda **facts: print("[census-check] " + json.dumps(facts, sort_keys=True, default=str),
                                       file=sys.stderr, flush=True),
    )
    # Receipt diagnostic: E over every executable cut of every scope at the final state (full sum).
    from .census_stop import census_iz_stop

    if exact_census:
        all_cut_diagnostic = {
            "E_total": 0.0, "E_by_scope": {},
            "basis": "exact census (eps = 0): SE = 0 for every route, so E = 0 by construction; not evaluated",
        }
    else:
        all_cut_report = census_iz_stop(
            session.scope_fields(), all_cuts, tau_S=float(design["tau_S"]), alpha=float(design["alpha"]),
            look=max(int(loop["checks"]) - 1, 0), n_draws=max(int(loop["draws"]), 2),
            eps=float(design["stop_parameters"]["eps"]), population_exhausted=True, forecast=False,
        )
        all_cut_diagnostic = {
            "E_total": float(all_cut_report["E_total"]),
            "E_by_scope": {scope: float(r["E"]) for scope, r in all_cut_report["scopes"].items()},
            "basis": "final fields; every executable cut; full sum",
        }
    means, global_means, global_activity, ordering_statistics, flip_margin_geometry, session_receipt = (
        session.result()
    )
    field_dump = _census_field_dump(
        ordering_statistics, means, run_id=str(run_id), random_seed=int(random_seed),
        target_class_id=int(target_class_id), design=design,
    )
    analytic_groups = [
        str(group.layer_id) for group in runtime.route_groups
        if _census_analytic_sign_active(partition_family) and _analytic_sign_group(group.route_kind)
    ]
    measured_route_count = sum(
        int(np.prod(group.shape, dtype=np.int64)) for group in runtime.route_groups
        if str(group.layer_id) not in set(analytic_groups)
    )
    (_, _, _, sign_critical, order_critical) = _census_simultaneous_block_critical_values(
        # The legacy census's first-look confidence (alpha 0.05 at look 0), for the
        # receipt-only per-arm order certificates of the shared census.
        look_confidence=1.0 - 0.05 / 2.0,
        route_count=len(route_ids),
        measured_local_sign_route_count=measured_route_count,
        required_views=tuple(dict.fromkeys(required_partition_views)),
    )
    passed = bool(loop["stop"].get("passed", False))
    try:
        import torch as _torch

        device = runtime.evaluator.device
        gpu = (_torch.cuda.get_device_name(device) if str(device).startswith("cuda") and _torch.cuda.is_available()
               else str(device))
        tf32 = {"cuda_matmul_allow_tf32": bool(_torch.backends.cuda.matmul.allow_tf32),
                "cudnn_allow_tf32": bool(_torch.backends.cudnn.allow_tf32)}
    except Exception as error:  # pragma: no cover - receipt only
        gpu, tf32 = f"unavailable: {type(error).__name__}", {}
    receipt = {
        **dict(objective_receipt),
        **dict(session_receipt),
        "schema": "dense_pruning_loss_census_receipt_v2",
        "census_sample_design": design,
        "batch_census": loop,
        # TF32 kernel choice depends on the batch and the GPU.
        "census_execution": {
            "batch_size": int(design["batch_size"]),
            "capture_rows": os.environ.get("IRG_CENSUS_CAPTURE_ROWS"),
            "gpu": gpu, **tf32,
        },
        "importance_inputs": inputs.receipt,
        "partition_label_views": tuple(dict.fromkeys(required_partition_views)),
        "required_partition_views": tuple(dict.fromkeys(required_partition_views)),
        "context_count_by_stratum": {
            "on_target": len(inputs.on_target_ids), "off_target": int(loop["distinct_off_target_images"]),
        },
        "eligible_population_by_stratum": {
            "on_target": len(inputs.on_target_ids), "off_target": len(inputs.off_target_ids),
        },
        "analytic_sign_partition": {
            "enabled": bool(analytic_groups), "analytic_sign_group_ids": analytic_groups,
            "sampled_sign_group_ids": [
                str(g.layer_id) for g in runtime.route_groups if str(g.layer_id) not in set(analytic_groups)
            ],
            "empirically_sampled_local_route_count": int(measured_route_count),
            "sign_normal_critical_value": float(sign_critical),
            "order_normal_critical_value": float(order_critical),
        },
        "panel_receipts": tuple(
            {"schema": "census_batch_look_v1", "look_index": int(b["batch"]),
             "cumulative_counts": {"on_target": len(inputs.on_target_ids),
                                   "off_target_draws": int((b["batch"] + 1) * int(design["batch_size"]))}}
            for b in loop["batches"]
        ),
        "prune_order_uncertainty": {
            "schema": "irg_census_order_certificate_receipt_only_v1",
            "decision_role": "receipt_diagnostic_not_a_stop_gate",
            "evaluated": False,
            "decision_resolved": bool(passed),
        },
        "decision_resolved": bool(passed),
        "stopping_reason": str(loop["stop_reason"]),
        "census_stopping_criterion": str(design["stop_rule"]),
        "stop_band_diagnostic": _census_band_diagnostic(loop.get("stop") or {}, scopes),
        "field_dump": field_dump,
        "shared_category": (None if shared_category is None else bool(shared_category)),
        "certified_ranges": certified_ranges,
        "all_executable_cuts_diagnostic": all_cut_diagnostic,
    }
    # Read by the run's progress record (scoring packet accounting); the
    # session may supply its own physical counts.
    receipt.setdefault("scoring_context_packet_count", int(loop["batch_count"]) + 1)
    receipt.setdefault("scoring_context_packet_size", int(design["batch_size"]))
    return means, global_means, global_activity, ordering_statistics, flip_margin_geometry, receipt


def _sealed_prefix_addition(
    receipt: SealedUniformPermutationReceipt,
    *,
    start: int,
    stop: int,
) -> tuple[str, ...]:
    """Return one verified nested-prefix extension from a sealed receipt."""

    receipt.verify()
    lower, upper = int(start), int(stop)
    if not 0 <= lower <= upper <= len(receipt.generated_order):
        raise ValueError("sealed prefix bounds are outside the population")
    return receipt.generated_order[lower:upper]


def _decision_bound_stable_looks(
    previous_stable_looks: int,
    *,
    unresolved_activity_fraction: Mapping[str, float],
    maximum_permitted_fraction: float,
) -> int:
    """Count consecutive looks that resolve the declared partition action."""

    values = tuple(float(value) for value in unresolved_activity_fraction.values())
    if (
        not values
        or not 0.0 <= float(maximum_permitted_fraction) <= 1.0
        or any(not math.isfinite(value) or value < 0.0 for value in values)
    ):
        raise ValueError("partition decision-bound inputs are invalid")
    if max(values) <= float(maximum_permitted_fraction):
        return int(previous_stable_looks) + 1
    return 0


# The partition functional each family realises, keyed by family.
#
# THIS LIVES BESIDE `_required_partition_views` ON PURPOSE: a family missing
# from this map raises `KeyError` at the very last step of the pilot -- after
# the census, the walk and the terminal, with every expensive thing already
# paid for and the receipt unwritten.  These keys must be exactly the
# families `_required_partition_views` accepts.
_PARTITION_FUNCTIONAL_BY_FAMILY: dict[str, str] = {
    "local_attribution_sign": (
        "signed_spatial_sum_immediate_destination_context_mean_v2"
    ),
    "global_attribution_sign": ("signed_spatial_sum_global_response_context_mean_v2"),
    "historical_spatial_weight_sum_sign": ("historical_spatial_weight_sum_sign_v1"),
    "randomized_count_matched": ("count_matched_to_local_attribution_sign_v1"),
    # The k=1 family: labels are the uniform nonnegative byte class and each
    # layer runs as a single queue under the incumbent uniform-polarity path.
    # There is no sign split, so there is no signed functional to name.
    "single_population": "uniform_nonnegative_single_queue_k1_v1",
    # The 2x2 of the two signed fields; both views must resolve before its
    # action is defined, which is why it is the most expensive family here.
    "local_global_sign_alignment": "local_global_sign_alignment_2x2_v1",
}


def _partition_functional(family: str) -> str:
    """Name the functional, or say which family is unregistered and where.

    A bare ``KeyError`` at receipt-writing time reads as a corrupt receipt
    rather than as a missing registration, which is how this cost two arms
    their completed science before anyone looked.
    """

    try:
        return _PARTITION_FUNCTIONAL_BY_FAMILY[family]
    except KeyError:
        raise KeyError(
            f"partition family {family!r} has no registered functional name; "
            "add it to _PARTITION_FUNCTIONAL_BY_FAMILY beside "
            "_required_partition_views, which already accepts it"
        ) from None


def _required_partition_views(
    family: PilotPartitionFamily,
) -> tuple[str, ...]:
    """Return the signed fields that define one hard partition family."""

    if family in {"local_attribution_sign", "randomized_count_matched"}:
        # The randomized control preserves the measured local-sign class
        # counts, so only the local orientation field defines its action.
        return ("local",)
    if family == "global_attribution_sign":
        return ("global",)
    if family == "local_global_sign_alignment":
        # Both recorded signed fields define this family's labels. Census
        # checks those label signs and certifies the loss-activity order.
        return ("local", "global")
    if family == "single_population":
        # Uniform labels need no signed field. Keeping the common recorded
        # local view matches the registered P sign family under the all scope.
        return ("local",)
    if family == "historical_spatial_weight_sum_sign":
        # Labels still derive from weights; this arm also needs the shared
        # dense loss-gradient activity for every removal order.
        return ("local",)
    raise ValueError(
        f"partition family {family!r} has no hard-field resolution contract"
    )


def _global_head_field_statistics(
    tripartite: TripartitePullbackMomentTensor,
    *,
    global_mean_standard_error: np.ndarray,
    global_activity_standard_error: np.ndarray,
) -> dict[str, np.ndarray]:
    """Extract the widened per-route ordering statistics (plumbing only).

    Every value is already measured: the second-moment/estimator terms are
    global-head slices of the fused tripartite tensor; the two standard
    errors are the loop's finite-population-corrected snapshot fusions.
    The contextual variance is the exact mixture identity
    ``sigma^2 = E[G^2] - mu^2`` on the fused moments, clamped at zero
    against representation noise (the tensor's own diagonal tolerance).
    """

    head = tripartite.head_ids.index(
        next(
            head_id for head_id in tripartite.head_ids if head_id.startswith("global:")
        )
    )
    mean = np.asarray(tripartite.mean[head], dtype=np.float64)
    second = np.maximum(
        np.asarray(
            tripartite.route_head_second_moment[:, head, head],
            dtype=np.float64,
        ),
        0.0,
    )
    mean_se = np.asarray(global_mean_standard_error, dtype=np.float64).reshape(-1)
    activity_se = np.asarray(global_activity_standard_error, dtype=np.float64).reshape(
        -1
    )
    if mean_se.shape != mean.shape or activity_se.shape != mean.shape:
        raise ValueError("ordering field standard errors are misaligned")
    statistics = {
        "global_mean_standard_error": mean_se,
        "global_contextual_variance": np.maximum(second - mean * mean, 0.0),
        "global_second_moment": second,
        "global_second_moment_standard_error": np.asarray(
            tripartite.second_moment_standard_error[:, head, head],
            dtype=np.float64,
        ),
        "global_mean_estimator_variance": np.maximum(
            np.asarray(
                tripartite.mean_estimator_covariance[:, head, head],
                dtype=np.float64,
            ),
            0.0,
        ),
        "global_activity_standard_error": activity_se,
    }
    statistics.update(_cross_head_field_statistics(tripartite, head))
    return statistics


# The heads the census accumulates, in the order the accumulator declares them.
# The tripartite tensor is (routes, heads, heads), so every pairing is already
# computed; only the global diagonal was ever extracted.
_LOCAL_HEAD_ID = "local_destination"
_LOSS_HEAD_ID = "loss:bce"


def _cross_head_field_statistics(
    tripartite: TripartitePullbackMomentTensor,
    global_head: int,
) -> dict[str, np.ndarray]:
    """Keep the head slices the global-only extraction discarded.

    NOTHING NEW IS MEASURED. ``route_head_second_moment`` and
    ``mean_estimator_covariance`` are ``(routes, heads, heads)`` and already
    populated for every pairing; the previous extraction took ``[:, g, g]`` and
    dropped the rest.

    WHY THIS IS NOT AN OPTIMIZATION. The global head is the local contribution
    contracted against the target cotangent (``local_rows`` is the contribution
    "before query cotangent contraction",
    ``backends/torch_runtime.py``). When a path is blocked that cotangent is
    ~0, so the global head vanishes BY MULTIPLICATION -- and with it every
    global-head statistic, mean and variance alike. A route whose function is
    to hold something else off is therefore invisible to `abs_raw`, K1, K2 and
    K3 simultaneously, and no refinement along the moment axis recovers it.
    The information survives only in the other heads, so a census that discards
    them cannot support any key that addresses gating.

    Emitted per route:
      * ``local_*`` -- the local diagonal, giving local sigma;
      * ``loss_*``  -- the loss:bce diagonal, nearest the decision's own units;
      * ``*_cross_covariance`` -- the off-diagonal pairings, which are what say
        whether a head carries information another does not.
    """

    ids = tuple(str(value) for value in tripartite.head_ids)
    second = np.asarray(tripartite.route_head_second_moment, dtype=np.float64)
    estimator = np.asarray(tripartite.mean_estimator_covariance, dtype=np.float64)
    means = np.asarray(tripartite.mean, dtype=np.float64)
    statistics: dict[str, np.ndarray] = {}
    for label, head_id in (
        ("local", _LOCAL_HEAD_ID),
        ("loss", _LOSS_HEAD_ID),
    ):
        if head_id not in ids:
            continue
        index = ids.index(head_id)
        head_mean = means[index]
        head_second = np.maximum(second[:, index, index], 0.0)
        statistics[f"{label}_head_mean"] = head_mean
        statistics[f"{label}_second_moment"] = head_second
        statistics[f"{label}_contextual_variance"] = np.maximum(
            head_second - head_mean * head_mean, 0.0
        )
        statistics[f"{label}_mean_estimator_variance"] = np.maximum(
            estimator[:, index, index], 0.0
        )
        # The cross term against the global head: low magnitude here with a
        # large local diagonal is the decoupling signature a blocked path
        # produces, and is the offline falsifier for the gating account.
        statistics[f"{label}_global_cross_covariance"] = second[:, index, global_head]
        statistics[f"{label}_global_estimator_covariance"] = estimator[
            :, index, global_head
        ]
    return statistics


def _declared_route_axis_offsets(
    runtime: Any,
    *,
    flat_length: int,
) -> dict[str, tuple[int, int]]:
    """One authority for the dense-field flat axis: DECLARED block order.

    The tripartite accumulator arrays (moments, activities, ordering
    statistics) are laid out on the physical route axis — the bound
    architecture's ``circuit_route_blocks`` in DECLARED order, carried by
    ``runtime.measurement.route_blocks``.  The profile's walk-order route
    groups may enumerate the same layers in a DIFFERENT order (the
    registered AlexNet profile is deep-first; ResNet's happens to
    coincide), so a per-layer view of any flat field array must be sliced
    by that layer's OWN declared offsets — never by a cursor advanced in
    group order.  Returned in declared order; every entry is validated
    against the route groups and the flat length, and any mismatch raises
    rather than guessing.
    """

    groups_by_layer = {str(group.layer_id): group for group in runtime.route_groups}
    blocks = tuple(
        block
        for block in runtime.measurement.route_blocks
        if str(block.layer_id) in groups_by_layer
    )
    block_layers = tuple(str(block.layer_id) for block in blocks)
    if len(set(block_layers)) != len(block_layers) or set(block_layers) != set(
        groups_by_layer
    ):
        raise ValueError(
            "declared measurement route blocks do not cover the route "
            f"groups exactly: blocks={sorted(set(block_layers))!r} "
            f"groups={sorted(groups_by_layer)!r}"
        )
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for block in blocks:
        layer_id = str(block.layer_id)
        group_shape = tuple(int(value) for value in groups_by_layer[layer_id].shape)
        if tuple(int(value) for value in block.shape) != group_shape:
            raise ValueError(
                "declared route block shape differs from the route group "
                f"for {layer_id!r}"
            )
        start = int(block.offset)
        stop = start + int(np.prod(group_shape, dtype=np.int64))
        if start != cursor:
            raise ValueError(
                "declared route blocks are not contiguous on the flat "
                f"field axis at {layer_id!r}: offset={start} expected={cursor}"
            )
        offsets[layer_id] = (start, stop)
        cursor = stop
    if cursor != int(flat_length):
        raise ValueError(
            "declared route blocks do not span the flat field axis: "
            f"covered={cursor} flat_length={int(flat_length)}"
        )
    return offsets


def _per_layer_fields_from_declared_flat(
    flat: np.ndarray,
    *,
    runtime: Any,
    offsets: Mapping[str, tuple[int, int]] | None = None,
) -> dict[str, np.ndarray]:
    """Slice one declared-axis flat field into per-layer group-shaped views."""

    values = np.asarray(flat)
    if values.ndim != 1:
        raise ValueError("declared flat field must be one-dimensional")
    resolved = (
        _declared_route_axis_offsets(runtime, flat_length=values.size)
        if offsets is None
        else offsets
    )
    shapes = {
        str(group.layer_id): tuple(int(value) for value in group.shape)
        for group in runtime.route_groups
    }
    return {
        layer_id: values[start:stop].reshape(shapes[layer_id])
        for layer_id, (start, stop) in resolved.items()
    }


def _declared_flat_from_per_layer(
    values_by_layer: Mapping[str, Any],
    *,
    runtime: Any,
    dtype: Any,
    offsets: Mapping[str, tuple[int, int]] | None = None,
) -> np.ndarray:
    """Assemble one declared-axis flat array from per-layer entries."""

    total = sum(
        int(np.prod(group.shape, dtype=np.int64)) for group in runtime.route_groups
    )
    resolved = (
        _declared_route_axis_offsets(runtime, flat_length=total)
        if offsets is None
        else offsets
    )
    parts: list[np.ndarray] = []
    for layer_id, (start, stop) in resolved.items():
        if layer_id not in values_by_layer:
            raise ValueError(
                f"per-layer values are missing declared layer {layer_id!r}"
            )
        part = np.asarray(values_by_layer[layer_id], dtype=dtype).reshape(-1)
        if part.size != stop - start:
            raise ValueError(f"per-layer values have the wrong size for {layer_id!r}")
        parts.append(part)
    return np.concatenate(parts)


def _response_field_binding_bar(
    *,
    required_views: Sequence[str],
    risks: Mapping[str, float],
    allowed_sign_unresolved_activity: float,
    order_resolved: bool,
    resolved: bool,
) -> str:
    """Which obligation is still outstanding: sign, order, or stable_looks.

    Reported because the two bars fail for different reasons and demand
    different responses. A sign bar short of tolerance wants more evidence; an
    order bar whose scope atom already exceeds the tolerance cannot be satisfied
    by any amount of it. Without this field an unresolved run cannot say whether
    it was short of evidence or blocked on something evidence cannot fix.

    Extracted rather than inlined because it was computed in two places -- the
    per-look receipt and the terminal exception -- and two transcriptions of a
    classification rule drift.
    """

    outstanding = tuple(
        str(view)
        for view in dict.fromkeys(required_views)
        if float(risks.get(str(view), 1.0)) > float(allowed_sign_unresolved_activity)
    )
    if outstanding:
        return "sign"
    if not bool(order_resolved):
        return "order"
    return "stable_looks" if not bool(resolved) else "none"


def _response_field_envelope_exceeded(
    *,
    elapsed_seconds: float,
    envelope_seconds: float | None,
    look_context_rates: Sequence[float],
    look_increment: int,
) -> dict[str, float] | None:
    """Forecast the next look's cost and decline it if the envelope cannot cover it.

    Returns the disclosure payload when the look is refused, else ``None``.

    Affordability is checked BEFORE the purchase: the point is to decline a look
    the envelope cannot cover, not to discover the overrun once it has been
    paid for. Returns ``None`` until at least one look has been measured, so the
    first look is never refused on a guess.

    The per-context rate is the MEDIAN over completed looks, not the last one,
    so a single slow look (node contention, a cold cache) cannot end an
    acquisition on its own. MEASURED: look cost is linear in the panel actually
    delivered, 0.072-0.084 s/context and stable across looks 7-10 of q504/cp.
    """

    if envelope_seconds is None or not len(look_context_rates):
        return None
    increment = max(int(look_increment), 0)
    predicted_rate = float(np.median(np.asarray(look_context_rates, dtype=np.float64)))
    predicted_seconds = predicted_rate * float(increment)
    if float(elapsed_seconds) + predicted_seconds <= float(envelope_seconds):
        return None
    return {
        "elapsed_seconds": float(elapsed_seconds),
        "envelope_seconds": float(envelope_seconds),
        "predicted_next_look_seconds": predicted_seconds,
        "predicted_seconds_per_context": predicted_rate,
        "declined_increment": float(increment),
    }


def _sealed_source_identity() -> Mapping[str, object]:
    """The commit that produced this receipt, read from the sealed worktree.

    **Without it a receipt cannot say what code made it.** Its identity
    fields name the model, the route axis and the context designs; the only
    `code_hash` is `content_hash({"implementation": "<a hardcoded string>"})`,
    which is identical across every seal. Meanwhile `SOURCE_COMMIT.txt`
    never enters Python at all: the launch script verifies
    `HEAD == SOURCE_COMMIT` and refuses a dirty tree, and then that verification
    is discarded.

    So code identity is guaranteed at LAUNCH and absent from the RECORD, and
    receipts from different seals are indistinguishable. Under
    receipt-only scoring that is not a small gap: reconstructing which code
    produced an arm means reading launch manifests, which live in git rather
    than beside the result.

    Absent worktree or marker yields ``None`` rather than raising -- a local or
    unsealed run is legitimate, and must be *disclosed as unsealed*, not blocked.
    """

    worktree = os.environ.get("IRG_WORKTREE", "").strip()
    commit: str | None = None
    manifest_sha256: str | None = None
    base_commit: str | None = None
    if worktree:
        marker = Path(worktree) / "SOURCE_COMMIT.txt"
        try:
            commit = marker.read_text().strip() or None
        except OSError:
            commit = None
        if commit is None:
            manifest = Path(worktree) / "SOURCE_MANIFEST.sha256"
            try:
                manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
            except OSError:
                manifest_sha256 = None
            provenance = Path(worktree) / "SOURCE_COMMIT.provenance.txt"
            try:
                base_commit = provenance.read_text().strip() or None
            except OSError:
                base_commit = None
    if commit is not None:
        return {
            "schema": "sealed_source_identity_v1",
            "source_commit": commit,
            "worktree": worktree or None,
            "sealed": True,
        }
    return {
        "schema": "sealed_source_identity_v2",
        "seal_kind": ("sha256_manifest" if manifest_sha256 is not None else "none"),
        "source_commit": None,
        "source_manifest_sha256": manifest_sha256,
        "base_commit": base_commit,
        "worktree": worktree or None,
        "sealed": manifest_sha256 is not None,
    }


def _accumulator_graph_state(accumulator: Any) -> Mapping[str, object]:
    """Whether a census accumulator is anchoring an autograd graph.

    `grad_fn` on ANY buffer means every context accumulated into it is still
    reachable, with its saved activations -- which is residency that no
    byte-count of the buffer itself would reveal, since the buffer keeps its
    declared size while holding the chain.

    `tracked_buffers` is therefore the diagnostic and `bytes` is context: if
    tracked_buffers is empty and residency still scales with panel, the
    retention is somewhere else and the graph hypothesis is refuted on this arm.
    """

    # torch is imported per-function in this module, never at module scope.
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is required in production
        return {"tracked_buffers": (), "buffer_bytes": 0, "context_ids": 0}

    tracked: list[str] = []
    total = 0
    for name, value in vars(accumulator).items():
        if not isinstance(value, torch.Tensor):
            continue
        total += int(value.numel()) * int(value.element_size())
        if value.grad_fn is not None or value.requires_grad:
            tracked.append(name)
    for name in ("sum_rows", "sketch_constant"):
        value = getattr(accumulator, name, None)
        if isinstance(value, torch.Tensor):
            total += int(value.numel()) * int(value.element_size())
            if value.grad_fn is not None or value.requires_grad:
                tracked.append(name)
    return {
        "tracked_buffers": tuple(sorted(set(tracked))),
        "buffer_bytes": total,
        "context_ids": len(getattr(accumulator, "context_ids", ()) or ()),
    }


def _residency_breakdown_armed() -> bool:
    """Whether to weigh the census's growing state at each look.

    Env: ``IRG_RESIDENCY_BREAKDOWN``. Off by default -- the breakdown walks
    every retained context id, which is cheap once per look and wasteful on an
    arm that is not being diagnosed.
    """

    return os.environ.get("IRG_RESIDENCY_BREAKDOWN", "").strip() in {
        "1",
        "true",
        "True",
    }


def _enrolled_response_field_target(
    *,
    total_existing: int,
    total_capacity: int,
    initial: int,
    response_packet: int,
    forecast_target: int | None,
) -> tuple[int, bool]:
    """Choose the next cumulative census panel under enrolled sizing.

    Look zero buys the occupancy-complete initial panel (``2 * initial``).
    Thereafter the sign forecast sizes the cumulative panel, guarded as an
    unvalidated cumulative forecast, while one response packet is the hard
    minimum increment of fresh contexts.  The sealed permutation prefix makes
    every such increment fresh.  No doubling or confirmation-growth schedule
    is retained as a hidden lower bound on the cumulative total.

    Returns ``(target_total, growth_guard_applied)``.
    """

    existing = int(total_existing)
    capacity_total = int(total_capacity)
    packet = int(response_packet)
    if existing < 0 or capacity_total < 0 or packet <= 0 or int(initial) <= 0:
        raise ValueError("invalid response-field sizing inputs")
    if existing == 0:
        return min(capacity_total, 2 * int(initial)), False
    if forecast_target is None:
        return min(capacity_total, existing + packet), False
    guarded, guard_applied = guard_unvalidated_cumulative_forecast(
        current_count=existing,
        analytic_target=max(int(forecast_target), existing),
        minimum_increment=packet,
        # One look's estimates can move this plug-in forecast by orders of
        # magnitude.  A realized look validates the unresolved fraction, not
        # those estimates' stability, so the same-quantity rule never earns a
        # validated-forecast exemption here.
        forecast_validated=False,
    )
    return (
        min(capacity_total, max(existing + packet, guarded)),
        bool(guard_applied),
    )


def _sign_bar_forecast_target(
    *,
    sigma_by_stratum: Mapping[str, Mapping[int, np.ndarray]],
    means: Mapping[int, np.ndarray],
    activities: Mapping[int, np.ndarray],
    capacity: Mapping[str, int],
    existing: Mapping[str, int],
    required_views: Sequence[str],
    critical: float,
    tolerance: float,
    total_capacity: int,
    minimum_increment: int,
) -> tuple[int | None, float]:
    """Smallest cumulative panel at which every required sign view resolves.

    Returns ``(target, predicted_fraction)``; ``target`` is ``None`` when the
    tolerance is not met even at the full sealed census, which cannot happen
    while any route carries a nonzero mean but is handled rather than assumed.

    The forecast holds the per-route means, activities and contextual sds at
    their current estimates and advances ONLY the standard errors, which is
    the part sampling actually buys. Strata advance along the ray the
    allocator has already realized (each keeps its current share of the
    panel), so the forecast describes the schedule that will actually be
    executed rather than an idealized reallocation.

    PLUG-IN BIAS, stated because it decides how the result may be used: the
    routes that dominate the forecast tail are those whose estimated mean
    landed near zero, and ``n`` grows as that estimate shrinks. The forecast
    is therefore optimistic about them. That direction is safe for SIZING --
    at worst it buys too little and the next look re-forecasts -- and unsafe
    for any futility claim, which is why this returns a size and never a
    verdict. The sign bar has no statistical futility to claim in any case:
    the finite-population correction drives every standard error to exactly
    zero at full census.
    """

    views = tuple(dict.fromkeys(required_views))
    heads = {"local": 0, "global": 1, "loss": 2}
    if any(view not in heads for view in views):
        raise ValueError("sign forecast views must select local/global/loss")
    delivered = sum(int(value) for value in existing.values())
    if delivered <= 0:
        return None, float("nan")
    shares = {
        stratum: float(int(existing[stratum])) / float(delivered)
        for stratum in existing
    }

    def _fused_errors(total: int, head: int) -> np.ndarray:
        variance = None
        for stratum, share in shares.items():
            population = int(capacity[stratum])
            # Never forecast beyond a stratum's own sealed population, and
            # never below what it already holds: targets are cumulative.
            count = int(
                min(
                    population,
                    max(int(existing[stratum]), round(share * float(total))),
                )
            )
            errors = stratified_mean_standard_error_at(
                sigma=sigma_by_stratum[stratum][head],
                count=count,
                population=population,
            )
            term = _binary_loss_weights(capacity)[stratum] ** 2 * np.square(errors)
            variance = term if variance is None else variance + term
        assert variance is not None
        return np.sqrt(variance)

    def _worst_fraction(total: int) -> float:
        return max(
            _census_sign_criterion_value(
                mean=means[heads[view]],
                standard_error=_fused_errors(total, heads[view]),
                activity=activities[heads[view]],
                critical=critical,
            )
            for view in views
        )

    target = smallest_count_meeting_tolerance(
        predicate=lambda total: _worst_fraction(total) <= float(tolerance),
        current_count=delivered,
        maximum_count=int(total_capacity),
        minimum_increment=int(minimum_increment),
    )
    return target, (float("nan") if target is None else _worst_fraction(target))


def _combine_v0_corrected_stratified_absolute_moments(
    *,
    accumulators: Mapping[str, Any],
    population_size_by_stratum: Mapping[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Combine the two registered class strata for exact ``E|G*phi*c|``.

    Each sink observes a sealed simple-random-sampling-without-replacement
    prefix with unit weights.  The registered binary estimand assigns each
    class stratum weight one half, so independent stratum variances combine
    with the squared weights.
    """

    means: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    for stratum in ("on_target", "off_target"):
        if stratum not in accumulators or stratum not in population_size_by_stratum:
            raise ValueError("corrected attribution strata are incomplete")
        mean, standard_error = accumulators[stratum].absolute_moment_snapshot(
            finite_population_size=int(population_size_by_stratum[stratum])
        )
        means.append(np.asarray(mean, dtype=np.float64))
        variances.append(np.square(np.asarray(standard_error, dtype=np.float64)))
    if loss_weighting() == "balanced":
        combined_mean = 0.5 * (means[0] + means[1])
        combined_standard_error = np.sqrt(0.25 * (variances[0] + variances[1]))
    else:
        weights = _binary_loss_weights(population_size_by_stratum)
        on, off = weights["on_target"], weights["off_target"]
        combined_mean = on * means[0] + off * means[1]
        combined_standard_error = np.sqrt(on ** 2 * variances[0] + off ** 2 * variances[1])
    if (
        combined_mean.shape != combined_standard_error.shape
        or not np.isfinite(combined_mean).all()
        or not np.isfinite(combined_standard_error).all()
        or np.any(combined_mean < 0.0)
        or np.any(combined_standard_error < 0.0)
    ):
        raise RuntimeError("corrected attribution order moments are invalid")
    return (
        np.ascontiguousarray(combined_mean, dtype=np.float64),
        np.ascontiguousarray(combined_standard_error, dtype=np.float64),
    )


class _PhaseClock:
    """Wall seconds per named host phase; receipt-only telemetry."""

    __slots__ = ("seconds", "_phase_started")

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}
        self._phase_started = time.perf_counter()

    def mark(self, phase: str) -> None:
        now = time.perf_counter()
        self.seconds[phase] = self.seconds.get(phase, 0.0) + float(
            now - self._phase_started
        )
        self._phase_started = now

    @property
    def total(self) -> float:
        return float(sum(self.seconds.values()))


def _align_census_host_threads() -> dict[str, Any]:
    """Observe census host threads. The launcher owns the OMP=1 setting."""

    import torch

    omp_value = os.environ.get("OMP_NUM_THREADS")
    omp_threads: int | None = None
    if omp_value is not None and omp_value.strip():
        try:
            omp_threads = int(omp_value.strip())
        except ValueError:
            omp_threads = None
    try:
        affinity_width = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        affinity_width = int(os.cpu_count() or 1)
    before = int(torch.get_num_threads())
    return {
        "affinity_width": int(affinity_width),
        "omp_num_threads": omp_threads,
        "torch_threads_before": before,
        "torch_threads_after": int(torch.get_num_threads()),
    }


def _check_cpu_thread_pools(
    *,
    device: Any,
    step_category_matrix: bool = False,
    optimization_bank: bool = True,
    evaluation_bank: bool = True,
    num_workers: int = 0,
) -> dict[str, Any]:
    """Record the run's thread pools at start and warn if CPU work can exceed the cores.

    This never raises and changes no pool or thread setting. Any problem is
    listed under ``warnings`` and reported once with ``warnings.warn``. If a
    warning filter turns that warning into an exception (``-W error``), the
    message is logged instead; if logging fails too (for example a log file
    that cannot be opened), the message stays only in the record. An error
    inside the check itself becomes a warning in a short record.
    """
    import logging
    import warnings

    try:
        report = _cpu_thread_pool_report(
            device=device,
            step_category_matrix=step_category_matrix,
            optimization_bank=optimization_bank,
            evaluation_bank=evaluation_bank,
            num_workers=num_workers,
        )
    except Exception as error:  # a diagnostic must not stop the run
        report = {
            "schema": "irg_cpu_thread_pool_check_v3",
            "checked_at": "run_start",
            "warnings": [f"the check itself failed: {type(error).__name__}: {error}"],
        }
    if report["warnings"]:
        message = "CPU pool check: " + "; ".join(report["warnings"])
        try:
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        except Exception:  # an "error" warning filter; report it without stopping
            try:
                logging.getLogger(__name__).warning(message)
            except Exception:  # the log destination failed; the record keeps the message
                pass
    return report


def _cpu_thread_pool_report(
    *,
    device: Any,
    step_category_matrix: bool,
    optimization_bank: bool,
    evaluation_bank: bool,
    num_workers: int,
) -> dict[str, Any]:
    """Record CPU work and waiting pools using the process affinity and quota.

    Decision work includes the calling thread and removal writers. Terminal
    work includes hashing and walk-log serialization. Image decoding adds
    work on bankless planes. Removal writers remain open until the walk tail,
    so the run-start count is a lower bound. V2_ARM_PROCS scales it by the
    number of processes sharing the task.
    """
    import torch
    from mechanistic_systems.contracts import content_address as ca

    problems: list[str] = []
    # The existing parsers fall back on a typo, which would hide it; report it.
    for name in (
        "SLURM_CPUS_PER_TASK", "CIRCUIT_PRODUCER_WORKERS",
        "IRG_PANEL_PRODUCER_WORKERS",
        "IRG_VERIFIED_READ_WORKERS", "IRG_BANK_STAGING_COPY_WORKERS",
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "V2_ARM_PROCS",
    ):
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value <= 0:
            problems.append(f"{name}={raw!r} is not a positive integer")
        elif name in {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"} and value != 1:
            problems.append(f"{name}={value}; the launcher sets 1")
    torch_threads = int(torch.get_num_threads())
    if torch_threads != 1:
        problems.append(f"torch uses {torch_threads} intra-op threads; the launcher sets 1")
    from mechanistic_systems.measurement.cpu_resources import _available_cpu_count

    available = _available_cpu_count()
    cores = available
    try:
        cores = min(available, max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", available))))
    except ValueError:
        pass
    arm_processes, arm_source = 1, "unset; taken as 1"
    try:
        if int(os.environ.get("V2_ARM_PROCS", "")) > 0:
            arm_processes, arm_source = int(os.environ["V2_ARM_PROCS"]), "V2_ARM_PROCS"
    except ValueError:
        pass
    hashing = max(
        int(ca._HASH_POOL_WORKERS),
        ca._concurrent_hash_worker_count(ca.CONCURRENT_ARRAY_HASH_MAX_WORKERS),
    )
    producers, profile = 1, None
    try:
        from mechanistic_systems.backends.calibration_runtime import RuntimeExecutionProfile

        profile = RuntimeExecutionProfile.from_environment()
        producers = int(profile.sampled_producer_workers)
    except Exception as error:
        problems.append(f"could not read the producer width: {type(error).__name__}: {error}")
    bankless_planes = sum(0 if bank else 1 for bank in (optimization_bank, evaluation_bank))
    loader_workers = max(0, int(num_workers))
    cpu_bound = {
        "caller": 1,
        "content_hashing": hashing,
        "sa_removal_writers": 1,
        "sa_walk_log_serializer": 1,
        "image_decode_workers": bankless_planes * producers * max(1, loader_workers),
    }
    phases = {
        "decision": 1 + cpu_bound["sa_removal_writers"],
        "terminal": (
            1 + hashing + cpu_bound["sa_walk_log_serializer"]
        ),
    }
    counted = cpu_bound["image_decode_workers"] + max(phases.values())
    task_counted = arm_processes * counted
    unbounded = {
        "sa_removal_writers": (
            "one writer thread per accepted group trace, each joined only at "
            "the walk tail; one is counted"
        ),
    }
    io_wait: dict[str, int] = {}
    try:
        from mechanistic_systems.measurement.bank_staging import _copy_settings
        from mechanistic_systems.measurement.verified_once import _read_workers

        if profile is None:
            raise RuntimeError("no runtime execution profile")
        for plane, bank in (("optimization", optimization_bank), ("evaluation", evaluation_bank)):
            io_wait.update({
                f"{plane}.gather": producers,
                f"{plane}.bank_shards": producers,
                f"{plane}.working_set_prefetch": (
                    producers if bank and profile.working_set_prefetch else 0
                ),
                f"{plane}.stage": 1,
                f"{plane}.feeders": 2,
            })
        io_wait.update({
            "verified_reads": _read_workers(None),
            "bank_copy": (
                _copy_settings(None, None)[0] if optimization_bank or evaluation_bank else 0
            ),
            "verified_npz_prefetch": 1,
            "readout_provenance": 1,
            "run_product_writer": 1,
            "step_matrix_writer": int(
                step_category_matrix
                and bool(os.environ.get("IRG_STEP_MATRIX_SPILL_DIR", "").strip())
            ),
            "torch_key_prefetch": int(
                torch.device(device).type == "cuda"
                and os.environ.get("IRG_TORCH_KEY_PREFETCH", "1").strip().lower()
                not in {"0", "false", "no", "off"}
            ),
            "gpu_activity_sampler": 1,
            "memory_probe": int(
                os.environ.get("IRG_MEMORY_PROBE", "").strip().lower()
                in {"1", "true", "yes", "on"}
            ),
        })
    except Exception as error:  # a diagnostic must not stop the run
        problems.append(f"could not read the waiting pools: {type(error).__name__}: {error}")
    if task_counted > cores:
        shared = (
            f"{arm_processes} arm processes x {counted} each; " if arm_processes > 1 else ""
        )
        problems.append(
            f"CPU-bound threads can reach {'at least ' if unbounded else ''}"
            f"{task_counted} on {cores} usable cores ("
            + shared
            + ", ".join(f"{name}={width}" for name, width in cpu_bound.items() if width)
            + ")"
        )
    if unbounded:
        problems.append(
            f"the count of {task_counted} CPU-bound threads is a lower bound: the S-A "
            "removal-log writers are not bounded at run start (one per accepted group "
            "trace, joined only at the walk tail)"
        )
    return {
        "schema": "irg_cpu_thread_pool_check_v3",
        "checked_at": "run_start",
        "available_cpu_count": available,
        "task_cores": int(cores),
        "torch_threads": torch_threads,
        "cpu_bound_threads": cpu_bound,
        "cpu_bound_pools_can_overlap": False,
        "phase_cpu_threads": phases,
        "image_decoding": {
            "bankless_planes": int(bankless_planes),
            "producer_threads_per_plane": int(producers),
            "loader_workers_per_producer": int(loader_workers),
            "decodes_on": (
                "producer_threads" if loader_workers == 0 else "loader_worker_processes"
            ),
        },
        "counted_cpu_threads": int(counted),
        "counted_cpu_threads_is_lower_bound": bool(unbounded),
        "unbounded_cpu_pools": unbounded,
        "arm_processes_sharing_task": int(arm_processes),
        "arm_processes_source": arm_source,
        "task_counted_cpu_threads": int(task_counted),
        "io_wait_threads": io_wait,
        "warnings": problems,
    }


def _record_cpu_thread_pools(
    measurement: dict[str, Any], cpu_thread_pools: Mapping[str, Any]
) -> None:
    """Put the run-start pool record into the census block's ``census_host_threads``.

    The census block is part of the receipt, so the receipt hash covers the
    record and every reader that hashes the whole receipt still verifies it.
    The nested mapping is replaced, never changed in place.
    """

    measurement["census_host_threads"] = {
        **dict(measurement.get("census_host_threads") or {}),
        "cpu_thread_pools": dict(cpu_thread_pools),
    }


def _measure_dense_local_partition_geometry(
    *,
    runtime: Any,
    optimization_axis: Any,
    query: Any,
    dense_mask: HardMask,
    route_ids: tuple[str, ...],
    route_layer_ids: tuple[str, ...] | LayerCodeAxis,
    target_class_id: int,
    dense_calibration_state: Any,
    pruning_guiding_loss_id: str = CALIBRATED_OVR_BCE,
    run_id: str,
    random_seed: int,
    sample_size_policy: IndependentSampleSizePolicy,
    required_partition_views: tuple[str, ...],
    partition_family: PilotPartitionFamily,
    partition_random_seed: int,
    prune_order_scope: str,
    initial_partition: str,
    enable_local_regime_diagnostics: bool = False,
    v1_ordering_key: str = "abs_raw",
    v0_attribution_order: str = V0_DEFAULT_ATTRIBUTION_ORDER,
    scoring_provider_factory: Callable[..., Any] | None = None,
    census_profile: Any | None = None,
    census_proposal_route_cap: int | None = None,
    census_materiality_resolution: Mapping[str, Any] | None = None,
    census_schedule_replay_schedule: Mapping[str, Any] | None = None,
    census_sample_design: Mapping[str, Any] | None = None,
    decision_fixed_offset_resolver: Callable[..., Any] | None = None,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, dict[str, np.ndarray]],
    dict[str, Any],
    dict[str, Any],
]:
    """Measure dense pruning-loss gradients and preserve registered labels.

    ``census_sample_design`` None is the legacy design (response_initial_packet
    per stratum); otherwise the registered design of
    :func:`census_sample_design_setting` (whole on-target stratum plus m per
    off-target class, one first look).
    """
    pruning_guiding_loss_id = validate_pruning_guiding_loss_id(pruning_guiding_loss_id)
    if dense_calibration_state is None:
        raise ValueError("dense census requires calibration before its first backward pass")
    calibration_hash = dense_calibration_state_hash(dense_calibration_state)
    if (
        dense_calibration_state.mask_hash != dense_mask.mask_hash
        or not bool(getattr(dense_calibration_state, "numerical_converged",
                            getattr(dense_calibration_state, "converged", False)))
    ):
        raise ValueError("dense census requires the converged canonical dense calibration state")
    if int(getattr(dense_calibration_state, "target_class_id", target_class_id)) != int(target_class_id):
        raise ValueError("dense census calibration belongs to another target")
    objective_receipt = {
        "schema": "dense_pruning_loss_census_objective_v2",
        "pruning_guiding_loss_id": pruning_guiding_loss_id,
        "dense_calibration_state_hash": calibration_hash,
        "dense_calibration_state_id": str(dense_calibration_state.calibration_state_id),
        "order_field": "loss_activity",
        "sign_field": "loss_mean",
        "calibration_before_census": True,
        **({"pruning_functional": "substituted_softmax",
            "fixed_offsets": "negative_dense_exclusion_lse",
            "loss_weighting": "per_image"}
           if decision_fixed_offset_resolver is not None else {}),
    }
    # Census producer: the registered setting is recorded in the receipt and,
    # when it is not the engine-rows reference, in the census cache key.
    # the effective producer: the corrected attribution order always runs the engine rows
    census_producer_id = (
        _CENSUS_ENGINE_ROWS
        if str(v0_attribution_order) == "corrected_aumann_shapley_global_attribution"
        else _census_producer()
    )
    objective_receipt["census_producer"] = census_producer_id

    materiality_floor = _census_materiality_floor(census_materiality_resolution)
    objective_receipt["census_materiality_resolution"] = census_materiality_resolution
    objective_receipt["consequence_materiality_floor"] = materiality_floor
    if "IRG_CONSEQUENCE_MATERIALITY_FLOOR" in os.environ:
        objective_receipt["consequence_materiality_warnings"] = [
            "IRG_CONSEQUENCE_MATERIALITY_FLOOR is retired and ignored; "
            "the census uses the supplied dense-loss first-look resolution"
        ]

    ordering_key = str(v1_ordering_key)
    if ordering_key not in V1_ORDERING_KEYS:
        raise ValueError("unregistered v1 ordering key")
    if ordering_key != "abs_raw":
        raise ValueError("dense census ordering is fixed to loss_activity")
    attribution_order = str(v0_attribution_order)
    if attribution_order not in V0_ATTRIBUTION_ORDER_MODES:
        raise ValueError("unregistered v0 attribution order mode")
    corrected_v0_order = (
        attribution_order == "corrected_aumann_shapley_global_attribution"
    )
    if corrected_v0_order and (
        str(prune_order_scope) != "layerwise"
        or ordering_key != "abs_raw"
        or str(partition_family) != "local_attribution_sign"
    ):
        raise ValueError(
            "corrected v0 attribution order requires layerwise local-sign "
            "abs_raw acquisition"
        )
    namespace = binary_evidence_role_namespace(
        run_id=run_id,
        random_seed=random_seed,
        context_axis_hash=optimization_axis.content_hash,
    )
    sampling_design = compile_binary_iid_sampling_design(
        optimization_axis,
        target_class_id=target_class_id,
        randomization_namespace=namespace,
    )
    populations = {
        stratum: tuple(
            map(
                str,
                sampling_design.source_context_ids_by_stratum[stratum],
            )
        )
        for stratum in ("on_target", "off_target")
    }
    if any(not population for population in populations.values()):
        raise ValueError("v0 local attribution has no response contexts")
    permutations = {
        stratum: SealedUniformPermutationReceipt.create(
            population,
            entropy_hex=content_hash(
                {
                    "run_id": run_id,
                    "stratum": stratum,
                    "role": "hard_partition_response_field",
                    "random_seed": int(random_seed),
                }
            )[-64:],
        )
        for stratum, population in populations.items()
    }
    if (
        census_sample_design is not None
        and census_sample_design.get("design") == CENSUS_SAMPLE_DESIGN_IMPORTANCE_BATCHES
    ):
        if corrected_v0_order:
            raise ValueError("the importance-batch census has no corrected-attribution order stream")
        return _measure_dense_batch_census(
            runtime=runtime, optimization_axis=optimization_axis, query=query, dense_mask=dense_mask,
            route_ids=route_ids, route_layer_ids=route_layer_ids, target_class_id=int(target_class_id),
            dense_calibration_state=dense_calibration_state, pruning_guiding_loss_id=pruning_guiding_loss_id,
            run_id=run_id, random_seed=int(random_seed), required_partition_views=required_partition_views,
            partition_family=str(partition_family), census_sample_design=census_sample_design,
            objective_receipt=objective_receipt, census_profile=census_profile,
            census_proposal_route_cap=census_proposal_route_cap,
            **({"decision_fixed_offset_resolver": decision_fixed_offset_resolver}
               if decision_fixed_offset_resolver is not None else {}),
        )
    census_draw_orders: dict[str, tuple[str, ...]] | None = None
    census_first_look_counts: dict[str, int] | None = None
    if census_sample_design is not None:
        if (
            census_sample_design.get("design")
            != CENSUS_SAMPLE_DESIGN_FULL_TARGET_CLASS_STRATIFIED
        ):
            raise ValueError("unregistered census sample design")
        census_draw_orders, census_first_look_counts, census_draw_receipt = (
            _census_registered_draw_orders(
                permutations,
                label_by_context=_census_context_labels(optimization_axis),
                off_target_per_class=int(census_sample_design["off_target_per_class"]),
            )
        )
        objective_receipt["census_sample_design"] = {
            **dict(census_sample_design),
            "draw": census_draw_receipt,
            **({"stratum_weights": _binary_loss_weights(populations)}
               if loss_weighting() == "per_image" else {}),
        }
    selected_population = tuple(
        context_id
        for stratum in ("on_target", "off_target")
        for context_id in populations[stratum]
    )
    destination_scales, transport_hash = resnet_local_destination_transport(
        runtime.evaluator
    )
    local_regime_specs = (
        resnet_local_regime_capture_specs(runtime.evaluator)
        if enable_local_regime_diagnostics
        else {}
    )

    provider_factory = scoring_provider_factory or OptimizedScoringResponseProvider
    provider = provider_factory(
        evaluator=runtime.evaluator,
        measurement_view=optimization_axis.data_view(),
        context_ids=selected_population,
        batch_factory=runtime.optimization_data_plane.role_batch_factory("scoring"),
        mask_resolver=lambda mask_hash: (
            dense_mask
            if str(mask_hash) == dense_mask.mask_hash
            else (_ for _ in ()).throw(KeyError(mask_hash))
        ),
        calibration_resolver=lambda mask: (
            dense_calibration_state
            if mask.mask_hash == dense_mask.mask_hash
            else (_ for _ in ()).throw(KeyError(mask.mask_hash))
        ),
        route_ids=route_ids,
        queries=(query,),
        response_spec_hash=content_hash(
            {
                "query_id": query.query_id,
                "semantics": "raw_target_logit_for_local_partition_v1",
            }
        ),
        response_metric_id="v0-local-destination-partition-metric-v1",
        study_hash=optimization_axis.content_hash,
        apply_affine_calibration_to_response=True,
        pruning_guiding_loss_id=pruning_guiding_loss_id,
        census_dense_calibration_state_hash=calibration_hash,
        **({"decision_fixed_offset_resolver": decision_fixed_offset_resolver}
           if decision_fixed_offset_resolver is not None else {}),
        local_destination_scales_by_layer=destination_scales,
        local_destination_transport_hash=transport_hash,
        local_covariance_sketch_rank=_local_covariance_sketch_rank(),
        local_covariance_sketch_seed=random_seed,
        local_regime_specs_by_layer=local_regime_specs,
    )
    corrected_order_provider: OptimizedScoringResponseProvider | None = None
    corrected_order_stream_identity: str | None = None
    if corrected_v0_order:
        # The incumbent tripartite producer owns local-sign labels and all
        # local/global/loss diagnostics.  Its generic row path
        # cannot simultaneously act as the exact conserving loss-only sink,
        # so the minimal safe seam is a second optimized provider invocation
        # on the IDENTICAL sealed stratum addition.  This is intentionally
        # receipted as a second VJP; it must never be described as same-VJP
        # with the incumbent v0 fields.
        from mechanistic_systems.measurement.optimized_irg_providers import (
            alexnet_v1_conserving_activation_bindings,
        )

        corrected_order_stream_identity = census_loss_path_hash(
            query_id=str(query.query_id), target_class_id=int(target_class_id),
            fit=dense_calibration_state, loss_id=pruning_guiding_loss_id,
        )
        corrected_order_provider = provider_factory(
            evaluator=runtime.evaluator,
            measurement_view=optimization_axis.data_view(),
            context_ids=selected_population,
            batch_factory=runtime.optimization_data_plane.role_batch_factory(
                "scoring"
            ),
            mask_resolver=lambda mask_hash: (
                dense_mask
                if str(mask_hash) == dense_mask.mask_hash
                else (_ for _ in ()).throw(KeyError(mask_hash))
            ),
            calibration_resolver=lambda mask: (
                dense_calibration_state
                if mask.mask_hash == dense_mask.mask_hash
                else (_ for _ in ()).throw(KeyError(mask.mask_hash))
            ),
            route_ids=route_ids,
            queries=(query,),
            response_spec_hash=content_hash(
                {
                    "query_id": query.query_id,
                    "semantics": (
                        "v0_corrected_aumann_shapley_dense_pruning_loss_order_v2"
                    ),
                }
            ),
            response_metric_id=(
                "v0-corrected-aumann-shapley-global-order-metric-v1"
            ),
            study_hash=optimization_axis.content_hash,
            apply_affine_calibration_to_response=True,
            pruning_guiding_loss_id=pruning_guiding_loss_id,
            census_dense_calibration_state_hash=calibration_hash,
        **({"decision_fixed_offset_resolver": decision_fixed_offset_resolver}
           if decision_fixed_offset_resolver is not None else {}),
            conserving_activation_bindings=(
                alexnet_v1_conserving_activation_bindings(
                    runtime.evaluator.mask_bindings
                )
            ),
        )

    from mechanistic_systems.measurement.phase_device_orchestration import state_for
    phase_state = state_for(provider)
    if corrected_order_provider is not None:
        corrected_order_provider._phase_device_state = phase_state

    calibration = dense_calibration_state
    loss_pullback_hash = census_loss_pullback_hash(
        query_id=query.query_id, target_class_id=int(target_class_id),
        fit=calibration, loss_id=pruning_guiding_loss_id,
    )
    head_ids = (
        "local_destination",
        f"global:{query.query_id}",
        f"loss:{pruning_guiding_loss_id}",
    )
    capacity = {stratum: len(population) for stratum, population in populations.items()}
    initial = int(sample_size_policy.response_initial_packet)
    if census_first_look_counts is None:
        if any(value < initial for value in capacity.values()):
            raise ValueError(
                "hard-partition response support cannot supply the initial packet"
            )
        census_minimum_by_stratum = {"on_target": initial, "off_target": initial}
    else:
        census_minimum_by_stratum = dict(census_first_look_counts)
    # Census strata, looks and final field close use the device path.
    from mechanistic_systems.measurement import device_route_moments as _census_drm

    census_v6_host_primary = _census_drm.census_v6_accumulator_host_primary(head_count=len(head_ids), route_count=len(route_ids), sketch_rank=_local_covariance_sketch_rank(), strata=2, device=runtime.evaluator.device)
    accumulators = {
        stratum: DeviceTripartiteMomentAccumulator(
            head_ids=head_ids,
            route_ids=route_ids,
            destination_transport_hash=transport_hash,
            loss_pullback_hash=loss_pullback_hash,
            sketch_rank=_local_covariance_sketch_rank(),
            sketch_seed=random_seed,
            device=runtime.evaluator.device,
            host_primary=census_v6_host_primary,
        )
        for stratum in ("on_target", "off_target")
    }
    census_v6_residency = _census_drm.census_v6_prepare_accumulators(accumulators)
    corrected_order_accumulators: dict[str, Any] = {}
    if corrected_v0_order:
        from .modern_v1_alexnet_backend import (
            _ExactConservingSalienceAccumulator,
        )

        assert corrected_order_stream_identity is not None
        corrected_order_accumulators = {
            stratum: _ExactConservingSalienceAccumulator(
                route_ids=route_ids,
                device=runtime.evaluator.device,
                source_by_context={},
                weight_by_context={},
                # This exact loss stream does not request the incumbent
                # local-destination head, so its declared transport is empty.
                destination_transport_hash="",
                loss_pullback_hash=corrected_order_stream_identity,
                positive_removal_stream_coordinate="frozen_affine_bce_loss",
                identity_draws_with_unit_weights=True,
                track_absolute_second_moment=True,
            )
            for stratum in ("on_target", "off_target")
        }
    # Absolute crossing counts pool the contexts actually observed in both
    # strata.  They deliberately do not use the 0.5/0.5 stratified averaging
    # that defines the existing local/global estimands.
    flip_margin_accumulator = DenseFlipMarginAccumulator()
    allocator = UnifiedNeymanAllocator()
    existing = {stratum: 0 for stratum in accumulators}
    allocation_sd = {stratum: 1.0 for stratum in accumulators}
    panel_receipts: list[Mapping[str, Any]] = []
    stable_looks = 0
    resolved = False
    stopping_reason = "response_horizon_exhausted_unresolved"
    confidence = float(sample_size_policy.confidence_level)
    look_index = 0
    previous_binding_price_scope: str | None = None
    label_views = tuple(dict.fromkeys(required_partition_views))
    required_views = label_views
    if not label_views or any(
        view not in {"local", "global"} for view in label_views
    ):
        raise ValueError("required partition views must select local/global")
    census_spending = SequentialCalibratedBCEResolutionSpec(
        **({"stratum_weights": tuple(sorted(_binary_loss_weights(populations).items()))}
           if loss_weighting() == "per_image" else {}),
        confidence_level=confidence,
        estimand_policy=V0_NORMALIZED_LOSS_SPAN_PER_CONNECTION_V1,
        maximum_planned_looks=None,
        precision_only_iid=True,
    )
    alpha_k = census_spending.look_alpha(look_index)
    look_confidence = 1.0 - alpha_k
    # Family-aware (RP under the P-label count source shares
    # P's sign family); identical to the P-only rule otherwise.
    analytic_sign_active = _census_analytic_sign_active(partition_family)
    analytically_resolved_route_count = sum(
        int(np.prod(group.shape, dtype=np.int64))
        for group in runtime.route_groups
        if analytic_sign_active and _analytic_sign_group(group.route_kind)
    )
    measured_sign_route_count = len(route_ids) - analytically_resolved_route_count
    # Label signs use the registered views and measured-label route count;
    # the loss-activity order coordinate covers every route.
    simultaneous_family_size = _census_simultaneous_family_size(
        route_count=len(route_ids),
        measured_local_sign_route_count=measured_sign_route_count,
        required_views=required_views,
    )
    (
        per_block_confidence,
        sign_simultaneous_family_size,
        order_simultaneous_family_size,
        sign_critical,
        order_critical,
    ) = _census_simultaneous_block_critical_values(
        look_confidence=look_confidence,
        route_count=len(route_ids),
        measured_local_sign_route_count=measured_sign_route_count,
        required_views=required_views,
    )
    allowed_order_disagreement_activity = float(
        sample_size_policy.maximum_prune_order_uncertainty_fraction
    )
    allowed_sign_unresolved_activity = float(
        sample_size_policy.maximum_partition_sign_uncertainty_fraction
    )
    if _census_decision_referenced():
        allowed_sign_unresolved_activity = _census_epsilon()
    final_risks = {"local": 1.0, "global": 1.0}
    final_loss_sign_diagnostic = 1.0
    final_order_certificate: Mapping[str, Any] | None = None
    final_order_source: np.ndarray | None = None
    final_order_standard_error: np.ndarray | None = None
    # Group-order/axis-order alignment: the accumulator flat axis is the
    # DECLARED route-block order, so every flat companion axis and every
    # per-layer view below must come from the same declared authority.
    if len(route_layer_ids) != len(route_ids):
        raise ValueError("route-layer companion differs from the route axis")
    # Resolve this once and retain it for every per-layer projection below.
    declared_axis_offsets = _declared_route_axis_offsets(
        runtime,
        flat_length=len(route_ids),
    )
    # The layer companion travels as int16 codes: a legacy
    # per-route string tuple is coerced once, and the per-block check below
    # is a vectorized code compare instead of a Python loop over 57M strings.
    # Scope ids the certificates emit are built from the axis's ``names`` and
    # are byte-identical to the string path.
    route_layer_ids = LayerCodeAxis.from_layer_ids(route_layer_ids)
    for layer_id, (start, stop) in declared_axis_offsets.items():
        if not route_layer_ids.block_matches(layer_id, start, stop):
            raise ValueError("route-layer companion differs from declared route blocks")
    flat_layer_ids = route_layer_ids
    census_host_threads = _align_census_host_threads()
    route_axis_hash = canonical_string_axis_content_hash(
        route_ids,
        name="prune-order route axis",
        prefix="prune-order-route-axis-",
    )
    if str(initial_partition) not in {"positive", "negative"}:
        raise ValueError("initial partition must be positive or negative")
    # Each look uses its unchanged label partition to lower the same rung
    # rule as the later walk, including ceil rounding and no-op positions.
    previous_global_activity: np.ndarray | None = None
    previous_partition_labels: np.ndarray | None = None
    measured_sign_mask = np.ones(len(route_ids), dtype=bool)
    analytic_sign_group_ids: list[str] = []
    if analytic_sign_active:
        for group in runtime.route_groups:
            if _analytic_sign_group(group.route_kind):
                start, stop = declared_axis_offsets[str(group.layer_id)]
                measured_sign_mask[start:stop] = False
                analytic_sign_group_ids.append(str(group.layer_id))
    sampled_sign_group_ids = [
        str(group.layer_id)
        for group in runtime.route_groups
        if str(group.layer_id) not in set(analytic_sign_group_ids)
    ]
    # The raw-input group is the first block of the DECLARED route axis
    # (smallest offset), independent of the profile's walk order.
    raw_input_group_id = min(
        declared_axis_offsets, key=lambda layer: int(declared_axis_offsets[layer][0])
    )
    raw_input_group_sampled = raw_input_group_id not in set(analytic_sign_group_ids)
    analytic_sign_scope = _analytic_sign_scope()
    # Receipt descriptors of the sign family, computed once for the per-look
    # and field-measurement receipts. RP's sampled local sign defines its
    # per-layer counts, not its membership.
    analytic_sign_scope_label = (
        "effective_weight_sign_all_supported_routes_v1"
        if analytic_sign_active
        else "disabled"
    )
    sign_definition = (
        "effective_weight_sign_all_supported_routes"
        if analytic_sign_active
        else "measured_local_attribution_sign"
    )
    randomized_partition_count_source: str | None = (
        _RANDOMIZED_PARTITION_COUNT_SOURCE_P_LABELS
        if str(partition_family) == "randomized_count_matched"
        else None
    )
    if randomized_partition_count_source is not None:
        sign_definition = (
            "randomized_count_matched_counts_from_effective_weight_sign_"
            "all_supported_routes"
            if analytic_sign_active
            else "randomized_count_matched_counts_from_measured_local_attribution_sign"
        )
    # SA-H's uniform labels read no sign; its census runs P's sign family
    # for parity, which the receipt names.
    census_sign_family_matched_to: str | None = (
        "local_attribution_sign"
        if str(partition_family) == "single_population"
        and analytic_sign_active
        else None
    )
    if census_sign_family_matched_to is not None:
        sign_definition = (
            "single_population_uniform_labels_census_sign_family_of_"
            "effective_weight_sign_all_supported_routes"
        )
    sign_measurement_weights_by_head = {
        0: measured_sign_mask.astype(np.float64),
        1: np.ones(len(route_ids), dtype=np.float64),
    }
    # Acquisition envelope bookkeeping. MEASURED on the census loop: look cost
    # is linear in the panel actually delivered (0.072-0.084 s/context, stable
    # across looks 7-10 of q504/cp), so a per-context rate learned from
    # completed looks predicts the next look's cost directly. Median over
    # completed looks rather than the last one, so a single slow look (node
    # contention, a cold cache) cannot end an acquisition on its own.
    acquisition_started = time.perf_counter()
    look_context_rates: list[float] = []
    look_increment = 0
    # Host post-processing seconds per look phase, summed over looks; the
    # per-look values ride the ``hard_partition_look_completed`` event and
    # the sums land in the field-measurement receipt.
    census_look_host_seconds: dict[str, float] = {}
    sign_forecast_state: dict[str, Any] | None = None
    envelope_seconds = (
        None
        if sample_size_policy.response_acquisition_envelope_seconds is None
        else float(sample_size_policy.response_acquisition_envelope_seconds)
    )
    envelope_forecast: dict[str, float] | None = None
    census_v6_close_receipt: dict[str, Any] = {}
    from .census_device_look import CensusDeviceLook

    census_device_look = CensusDeviceLook(
        device=runtime.evaluator.device,
        layer_axis=flat_layer_ids,
        partition_family=str(partition_family),
        analytic_route_mask=(
            ~measured_sign_mask
            if str(partition_family) == "local_attribution_sign"
            and _analytic_sign_mode()
            else None
        ),
        sign_measurement_weights_by_head=sign_measurement_weights_by_head,
    )
    census_device_look._phase_device_state = phase_state
    certify_layerwise_order = (
        (census_device_look.certify_prune_order_stability)
    )

    while True:
        alpha_k = census_spending.look_alpha(look_index)
        look_confidence = 1.0 - alpha_k
        (
            per_block_confidence,
            sign_simultaneous_family_size,
            order_simultaneous_family_size,
            sign_critical,
            order_critical,
        ) = _census_simultaneous_block_critical_values(
            look_confidence=look_confidence,
            route_count=len(route_ids),
            measured_local_sign_route_count=measured_sign_route_count,
            required_views=required_views,
        )
        total_existing = sum(existing.values())
        total_capacity = sum(capacity.values())
        sign_forecast_target: int | None = None
        sign_forecast_fraction = float("nan")
        if sign_forecast_state is not None and total_existing > 0:
            sign_forecast_target, sign_forecast_fraction = _sign_bar_forecast_target(
                sigma_by_stratum=sign_forecast_state["sigma_by_stratum"],
                means={head: sign_forecast_state["means"][head] for head in (0, 1)},
                activities={head: sign_forecast_state["activities"][head] for head in (0, 1)},
                capacity=capacity,
                existing=existing,
                required_views=required_views,
                critical=sign_critical,
                tolerance=allowed_sign_unresolved_activity,
                total_capacity=total_capacity,
                minimum_increment=int(sample_size_policy.response_packet),
            )
        (
            target_total,
            sign_forecast_guarded,
        ) = _enrolled_response_field_target(
            total_existing=int(total_existing),
            total_capacity=int(total_capacity),
            initial=int(initial),
            response_packet=int(sample_size_policy.response_packet),
            forecast_target=sign_forecast_target,
        )
        if census_first_look_counts is not None and int(total_existing) == 0:
            # Registered design: the first look is the whole registered draw.
            target_total = int(sum(census_first_look_counts.values()))
            sign_forecast_guarded = False
        schedule_target = target_total
        look_increment = int(target_total) - int(total_existing)
        # Affordability is checked BEFORE the purchase, never after: the point
        # is to decline a look the envelope cannot cover, not to discover the
        # overrun once it has been paid for. Skipped until at least one look
        # has been measured, so the first look is never refused on a guess.
        declined = _response_field_envelope_exceeded(
            elapsed_seconds=time.perf_counter() - acquisition_started,
            envelope_seconds=envelope_seconds,
            look_context_rates=look_context_rates,
            look_increment=look_increment,
        )
        if declined is not None:
            envelope_forecast = dict(declined)
            envelope_forecast["declined_cumulative_target"] = float(target_total)
            envelope_forecast["delivered_cumulative_count"] = float(total_existing)
            # EMIT THE EVIDENCE, NOT ONLY THE VERDICT. The exception carries
            # this payload, but only its message reaches the log, so without
            # this a refusal reads `acquisition_envelope_exhausted_unresolved`
            # and nothing else -- the rate, the increment and the margin would
            # have to be reconstructed by hand from panel timestamps to tell an
            # envelope that was too small from a field that genuinely could not
            # resolve. MEASURED in one run, those are one percent apart, so the
            # distinction is not academic.
            _progress(
                "response_field_envelope_declined_look",
                look_index=look_index,
                sign_forecast_target=(
                    None if sign_forecast_target is None else int(sign_forecast_target)
                ),
                sign_forecast_fraction=float(sign_forecast_fraction),
                **{str(k): float(v) for k, v in envelope_forecast.items()},
            )
            stopping_reason = "acquisition_envelope_exhausted_unresolved"
            break
        allocation = allocator.allocate(
            binary_neyman_request(
                total_count=target_total,
                available_by_stratum=capacity,
                minimum_by_stratum=census_minimum_by_stratum,
                existing_by_stratum=existing,
                sd_by_stratum=allocation_sd,
                weight_by_stratum=_binary_loss_weights(populations),
                role="hard_partition_response_field",
            )
        )
        look_started = time.perf_counter()
        delivered: dict[str, Mapping[str, object]] = {}
        for stratum in ("on_target", "off_target"):
            start = existing[stratum]
            stop = int(allocation.target_counts[stratum])
            addition = (
                _sealed_prefix_addition(
                    permutations[stratum],
                    start=start,
                    stop=stop,
                )
                if census_draw_orders is None
                else census_draw_orders[stratum][start:stop]
            )
            if not addition:
                continue
            _progress(
                "hard_partition_panel_started",
                look_index=look_index,
                stratum=stratum,
                previous_count=start,
                target_count=stop,
                addition_count=len(addition),
            )
            request = MeasurementRequest.create(
                kind="response_moments",
                state_hash=content_hash(
                    {
                        "run_id": run_id,
                        "mask_hash": dense_mask.mask_hash,
                        "stratum": stratum,
                        "look_index": look_index,
                    }
                ),
                version="v1",
                context_selector=ContextSelector(
                    selector_id=("sealed-hard-partition-response-addition-v2"),
                    requested_count=len(addition),
                    context_ids=addition,
                ),
                expected_cost=CostEstimate(
                    forward_passes=float(len(addition)),
                    backward_passes=float(len(addition)),
                ),
                expected_decision_value=0.0,
                provenance=MeasurementProvenance(
                    producer_id=("registered_resnet18_hard_partition_device_moments"),
                    producer_version="2",
                    code_hash=content_hash(
                        {
                            "implementation": (
                                "fused_u_g_d_device_sufficient_statistics_v2"
                            )
                        }
                    ),
                ),
                payload={
                    "mask_hash": dense_mask.mask_hash,
                    "whole_physical_route_axis": True,
                    "stratum_id": stratum,
                    "look_index": look_index,
                },
            )
            _ensure_stratum_resident(
                accumulators,
                stratum,
                runtime.evaluator.device,
            )
            if census_producer_id == _CENSUS_SUFFICIENT_STATISTICS and not corrected_v0_order:
                receipt = _accumulate_census_sufficient_statistics(
                    provider,
                    request,
                    accumulator=accumulators[stratum],
                    flip_margin_accumulator=flip_margin_accumulator,
                )
            else:
                receipt = provider.accumulate_device_tripartite(
                    request,
                    accumulator=accumulators[stratum],
                    flip_margin_accumulator=flip_margin_accumulator,
                )
            if corrected_v0_order:
                assert corrected_order_provider is not None
                assert corrected_order_stream_identity is not None
                corrected_request = MeasurementRequest.create(
                    kind="response_moments",
                    state_hash=content_hash(
                        {
                            "run_id": run_id,
                            "mask_hash": dense_mask.mask_hash,
                            "stratum": stratum,
                            "look_index": look_index,
                            "v0_attribution_order": attribution_order,
                        }
                    ),
                    version="v1",
                    context_selector=ContextSelector(
                        selector_id=(
                            "sealed-v0-corrected-attribution-response-addition-v1"
                        ),
                        requested_count=len(addition),
                        context_ids=addition,
                    ),
                    expected_cost=CostEstimate(
                        forward_passes=float(len(addition)),
                        backward_passes=float(len(addition)),
                    ),
                    expected_decision_value=0.0,
                    provenance=MeasurementProvenance(
                        producer_id=(
                            "registered-alexnet-v0-corrected-attribution-order"
                        ),
                        producer_version="2",
                        code_hash=content_hash(
                            {
                                "implementation": (
                                    "integrated_unit_activation_pruning_loss_"
                                    "second_same_context_vjp_v2"
                                )
                            }
                        ),
                    ),
                    payload={
                        "mask_hash": dense_mask.mask_hash,
                        "whole_physical_route_axis": True,
                        "stratum_id": stratum,
                        "look_index": look_index,
                        "positive_removal_stream_coordinate": (
                            "frozen_affine_bce_loss"
                        ),
                        "positive_removal_stream_identity_hash": (
                            corrected_order_stream_identity
                        ),
                        "affine_calibration_applied": True,
                        "context_specific_loss_pullback_applied": True,
                        "same_context_as_incumbent_v0_panel": True,
                        "same_vjp_as_incumbent_v0_panel": False,
                    },
                )
                corrected_receipt = (
                    corrected_order_provider.accumulate_device_tripartite(
                        corrected_request,
                        accumulator=corrected_order_accumulators[stratum],
                    )
                )
                if (
                    corrected_receipt.get(
                        "exact_conserving_single_pass_loss_path_streaming"
                    )
                    is not True
                    or corrected_receipt.get(
                        "positive_removal_stream_coordinate"
                    )
                    != "frozen_affine_bce_loss"
                    or corrected_receipt.get(
                        "positive_removal_stream_identity_hash"
                    )
                    != corrected_order_stream_identity
                    or int(corrected_receipt.get("context_count", 0))
                    != len(addition)
                    or corrected_receipt.get("full_tripartite_rows_materialized")
                    is not False
                ):
                    raise RuntimeError(
                        "corrected v0 attribution stream did not close exact "
                        "dense pruning-loss authority"
                    )
                delivered[stratum] = {
                    **dict(receipt),
                    "v0_corrected_attribution_extra_score_pass": {
                        **dict(corrected_receipt),
                        "same_context_as_incumbent_v0_panel": True,
                        "same_vjp_as_incumbent_v0_panel": False,
                        "selection_effect": "route_activity_order_only",
                    },
                }
            else:
                delivered[stratum] = receipt
            existing[stratum] = stop
            panel_attribution: dict[str, Any] = {}
            if _residency_breakdown_armed():
                # PER PANEL, not per look. In one measured run the decisive step
                # -- 97.6 -> 152.4 GiB, +54.8 GiB -- happened INSIDE a single
                # panel, so a per-look breakdown brackets the event it is meant
                # to catch and reports only its sum. Panels are tens per arm,
                # so the gc walk stays affordable.
                panel_attribution = {
                    "host_byte_census": _host_byte_census(),
                    "flip_margin_residency": (
                        flip_margin_accumulator.residency_bytes()
                    ),
                    "device_memory_breakdown": provider.residency_breakdown(
                        additional_roots={
                            "census_accumulators": accumulators,
                            "corrected_order_accumulators": corrected_order_accumulators,
                            "flip_margin_accumulator": flip_margin_accumulator,
                        }
                    )["device_memory"],
                }
            _progress(
                "hard_partition_panel_completed",
                look_index=look_index,
                stratum=stratum,
                cumulative_count=stop,
                **panel_attribution,
                **dict(receipt),
            )

        look_clock = _PhaseClock()
        decision_snapshot = combine_stratified_route_decision_snapshots(
            accumulators,
            weight_by_stratum=_binary_loss_weights(populations),
            population_size_by_stratum={
                stratum: len(population) for stratum, population in populations.items()
            },
            head_indices=(0, 1, 2),
            activity_indices=(0, 1, 2),
            prepare_stratum=lambda stratum: _ensure_stratum_resident(
                accumulators,
                stratum,
                runtime.evaluator.device,
            ),
        )
        means = {head: decision_snapshot.mean[head] for head in (0, 1, 2)}
        errors = {head: decision_snapshot.mean_standard_error[head] for head in (0, 1, 2)}
        activities = {head: decision_snapshot.activity_mean[head] for head in (0, 1, 2)}
        decision_snapshot_activity_errors_global = decision_snapshot.activity_mean_standard_error[1]
        activity_errors = decision_snapshot.activity_mean_standard_error[2]
        census_device_look.begin_look(decision_snapshot)
        look_clock.mark("decision_snapshot")
        corrected_order_statistic: tuple[np.ndarray, np.ndarray] | None = None
        if corrected_v0_order:
            corrected_order_statistic = (
                _combine_v0_corrected_stratified_absolute_moments(
                    accumulators=corrected_order_accumulators,
                    population_size_by_stratum=capacity,
                )
            )
        # Delegated to the forecast kernel so the bar and any forecast of it
        # share ONE implementation. Exact zero-variance ties are excluded
        # there as known zeros using the declared nonnegative tie class; they
        # are not statistically unresolved. Keeping a second transcription
        # here is precisely how a forecaster drifts from the criterion it is
        # supposed to be steering toward.
        risks: dict[str, float] = (
            (census_device_look.sign_criteria(
                means={head: means[head] for head in (0, 1)},
                errors={head: errors[head] for head in (0, 1)},
                activities={head: activities[head] for head in (0, 1)},
                sign_measurement_weights_by_head=sign_measurement_weights_by_head,
                critical=sign_critical,
                tolerance=allowed_sign_unresolved_activity,
                decision_referenced=_census_decision_referenced(),
            ))
        )
        expected_wrongsign_consequence = (
            (census_device_look.expected_wrongsign(
                means={head: means[head] for head in (0, 1)},
                errors={head: errors[head] for head in (0, 1)},
                activities={head: activities[head] for head in (0, 1)},
                sign_measurement_weights_by_head=sign_measurement_weights_by_head,
            ))
        )
        final_risks = risks
        # Preserve the loss-view receipt diagnostic. It has no role in label
        # resolution, family sizing, allocation, forecasting, or stopping.
        final_loss_sign_diagnostic = _census_sign_criterion_value(
            mean=means[2], standard_error=errors[2], activity=activities[2],
            critical=order_critical,
        )
        look_clock.mark("sign_criterion")

        def _host_flat_partition_labels() -> np.ndarray:
            local_by_layer = _per_layer_fields_from_declared_flat(
                means[0],
                runtime=runtime,
                offsets=declared_axis_offsets,
            )
            global_by_layer = _per_layer_fields_from_declared_flat(
                means[1],
                runtime=runtime,
                offsets=declared_axis_offsets,
            )
            labels_by_layer = _partition_labels_by_layer(
                runtime,
                measurement=runtime.measurement,
                query_index=0,
                family=partition_family,
                random_seed=int(partition_random_seed),
                local_mean_by_layer=local_by_layer,
                global_mean_by_layer=global_by_layer,
                local_destination_scales_by_layer=destination_scales,
                category_id=int(target_class_id),
            )
            return _declared_flat_from_per_layer(
                labels_by_layer,
                runtime=runtime,
                dtype=np.int8,
                offsets=declared_axis_offsets,
            )

        if census_device_look.labels_on_device:
            flat_labels = census_device_look.partition_labels(
                means=means,
                host_flat_labels=_host_flat_partition_labels,
            )
        else:
            flat_labels = _host_flat_partition_labels()
            census_device_look.note_host_labels()
        executable_boundary_fractions = _v0_executable_boundary_fractions(
            runtime.route_groups,
            profile=census_profile,
            partition_route_counts_by_layer=_census_partition_route_counts(flat_labels, declared_axis_offsets),
            proposal_route_cap=census_proposal_route_cap,
            census_schedule_replay_schedule=census_schedule_replay_schedule,
        )
        look_clock.mark("partition_labels")
        if corrected_order_statistic is not None:
            order_source, order_source_standard_error = corrected_order_statistic
        else:
            order_source = activities[2]
            order_source_standard_error = activity_errors
        final_order_source = order_source
        final_order_standard_error = order_source_standard_error
        look_clock.mark("ordering_statistic")
        v1_ladder_translation: (
            tuple[dict[int, tuple[float, ...]], tuple[dict[str, Any], ...]] | None
        ) = (
            None
            if str(prune_order_scope) == "layerwise"
            else _translate_global_ladder_to_partition_fractions(
                global_activity=order_source,
                partition_labels=flat_labels,
                ladder=_V1_EXECUTABLE_LADDER,
            )
        )
        order_certificate = (
            certify_layerwise_order(
                current_global_activity=order_source,
                previous_global_activity=previous_global_activity,
                current_partition_labels=flat_labels,
                previous_partition_labels=previous_partition_labels,
                layer_ids=flat_layer_ids,
                maximum_mask_disagreement_activity_fraction=(
                    allowed_order_disagreement_activity
                ),
                scope="layerwise",
                route_axis_hash=route_axis_hash,
                audited_retained_fractions_by_scope=(executable_boundary_fractions),
                # v0 commits are protected by exact paired BCE evidence.
                # Score-order uncertainty therefore measures the maximum
                # common-response activity whose route membership can change
                # in any executable proposal, not a scope-relative fraction
                # that lets one negligible route in a tiny layer dominate the
                # entire dense-field acquisition.
                activity_normalization="global_route_field",
                # Consequence damage and uncertainty use the same loss gradient.
                score_standard_error=order_source_standard_error,
                route_damage=means[2],
                route_damage_standard_error=errors[2],
                critical_value=order_critical,
                consequence_materiality_floor=materiality_floor,
                consequence_gated=bool(
                    sample_size_policy.prune_order_consequence_gated
                ),
                # Membership follows the measured local-label sign.
                # Registered analytic weight labels have exact membership.
                partition_sign_unresolved=(
                    (np.abs(means[0]) <= sign_critical * errors[0])
                    & (errors[0] > 0.0)
                    & measured_sign_mask
                ),
                consequence_price_short_circuit_scope=(
                    previous_binding_price_scope
                    if _consequence_price_short_circuit_enabled()
                    else None
                ),
            )
            if str(prune_order_scope) == "layerwise"
            else certify_prune_order_uncertainty(
                global_activity=order_source,
                global_activity_standard_error=order_source_standard_error,
                partition_labels=flat_labels,
                layer_ids=flat_layer_ids,
                confidence_level=per_block_confidence,
                normal_critical_value=order_critical,
                maximum_boundary_activity_fraction=(
                    allowed_order_disagreement_activity
                ),
                scope=str(prune_order_scope),
                route_axis_hash=route_axis_hash,
                # The certificate's claim narrows to order correctness
                # at the registered executable ladder cuts, translated from
                # global retained fractions to this look's per-partition
                # composition.
                audited_retained_fractions_by_scope={
                    "global": v1_ladder_translation[0]
                },
                order_statistic=(
                    None
                ),
                order_statistic_standard_error=(
                    None
                ),
            )
        )
        priced_scopes = [
            item
            for item in order_certificate.scope_receipts
            if not bool(getattr(item, "consequence_price_deferred", True))
            and int(getattr(item, "consequence_band_size", 0)) > 0
        ]
        if priced_scopes:
            previous_binding_price_scope = max(
                priced_scopes,
                key=lambda item: (
                    float(item.consequence_price) / float(item.consequence_band_size)
                ),
            ).scope_id
        final_order_certificate = order_certificate.to_receipt()
        look_clock.mark("order_certificate")
        if v1_ladder_translation is not None:
            final_order_certificate = {
                **final_order_certificate,
                "executable_ladder_scoping": {
                    "schema": "v1_executable_ladder_order_scope_v1",
                    "declared_global_ladder": list(_V1_EXECUTABLE_LADDER),
                    "ladder_registration": ("irg_v1_repair_program_preregistration_v1"),
                    "global_retained_count_rule": (
                        "ceil(fraction * total_route_count) clamped to the " "interior"
                    ),
                    "translation": list(v1_ladder_translation[1]),
                    "per_partition_audited_fractions": {
                        str(key): list(value)
                        for key, value in v1_ladder_translation[0].items()
                    },
                    "claim": (
                        "order correctness is certified only at the "
                        "declared executable ladder cuts translated per "
                        "partition; NO order claim exists at non-ladder cuts"
                    ),
                },
            }
        signature = content_hash(
            {
                "local_partition_labels": flat_labels,
                "global_sign": means[1] >= 0.0,
                "analytic_sign_partition": analytic_sign_active,
                **(
                    {}
                    if not corrected_v0_order
                    else {
                        "v0_attribution_order": attribution_order,
                        "corrected_order_score_hash": content_hash(
                            order_source,
                            prefix="v0-corrected-order-score-",
                        ),
                    }
                ),
            },
            prefix="hard-partition-decision-signature-",
        )
        look_clock.mark("decision_signature")
        sign_resolved = (
            max(risks[view] for view in required_views)
            <= allowed_sign_unresolved_activity
        )
        stable_looks = (
            stable_looks + 1 if sign_resolved and order_certificate.resolved else 0
        )
        strictly_resolved = (
            sign_resolved
            and order_certificate.resolved
            and stable_looks >= int(sample_size_policy.response_stable_looks)
        )
        # Over the complete sealed census the consecutive-stable-looks
        # requirement is degenerate; only that count is waivable, never the
        # resolution of either component.
        census_terminal_receipt = _census_complete_terminal_receipt(
            strictly_resolved=strictly_resolved,
            sign_resolved=sign_resolved,
            order_resolved=bool(order_certificate.resolved),
            existing=existing,
            capacity=capacity,
            stable_looks=stable_looks,
            required_stable_looks=int(sample_size_policy.response_stable_looks),
            unresolved_activity_fraction=risks,
            required_partition_views=required_views,
            maximum_permitted_unresolved_activity_fraction=(
                allowed_sign_unresolved_activity
            ),
            order_certificate_receipt=final_order_certificate,
            frozen_calibration_state_id=(calibration.calibration_state_id),
        )
        resolved = strictly_resolved or census_terminal_receipt is not None
        panel_receipts.append(
            {
                "schema": "hard_partition_sequential_look_receipt_v2",
                **objective_receipt,
                "schema": "hard_partition_sequential_look_receipt_v2",
                "look_index": look_index,
                "audited_executable_cuts_by_scope": executable_boundary_fractions,
                "census_rung_profile": _census_rung_profile_receipt(census_profile),
                "census_proposal_route_cap": census_proposal_route_cap,
                "census_schedule_replay_hash": (
                    None if census_schedule_replay_schedule is None
                    else content_hash(census_schedule_replay_schedule)
                ),
                "allocation": allocation.to_receipt(),
                "delivered_panels": delivered,
                "cumulative_counts": dict(existing),
                "decision_signature": signature,
                "stable_looks": stable_looks,
                "stable_look_semantics": (
                    "consecutive_label_sign_and_loss_activity_order_stability_v9"
                ),
                "confidence_level": confidence,
                "look_confidence_level": look_confidence,
                "alpha_k": alpha_k,
                "alpha_spending": {
                    "rule": "alpha/((k+1)*(k+2))",
                    "alpha": census_spending.error_probability,
                    "look_index": look_index,
                },
                "simultaneous_family_size": simultaneous_family_size,
                "simultaneous_block_allocation": {
                    "schema": "census_sign_order_bonferroni_blocks_v2",
                    "sign_block_empty": simultaneous_family_size == len(route_ids),
                    "order_alpha": 1.0 - per_block_confidence,
                    "sign_alpha": (
                        0.0 if simultaneous_family_size == len(route_ids)
                        else 1.0 - per_block_confidence
                    ),
                    "per_block_confidence_level": per_block_confidence,
                    "sign_family_size": sign_simultaneous_family_size,
                    "sign_normal_critical_value": sign_critical,
                    "order_family_size": order_simultaneous_family_size,
                    "order_normal_critical_value": order_critical,
                },
                "normal_critical_value": order_critical,
                "analytic_sign_partition": {
                    "enabled": analytic_sign_active,
                    "scope": analytic_sign_scope_label,
                    "analytic_sign_scope": analytic_sign_scope,
                    "analytic_sign_group_ids": list(analytic_sign_group_ids),
                    "sampled_sign_group_ids": list(sampled_sign_group_ids),
                    "raw_input_group_id": str(raw_input_group_id),
                    "raw_input_group_sampled": bool(raw_input_group_sampled),
                    "sign_definition": sign_definition,
                    # Key present only for RP under the P-label count source,
                    # so every other receipt keeps its exact bytes.
                    **(
                        {}
                        if randomized_partition_count_source is None
                        else {
                            "randomized_partition_count_source": (
                                randomized_partition_count_source
                            )
                        }
                    ),
                    # Key present only for SA-H under the all scope.
                    **(
                        {}
                        if census_sign_family_matched_to is None
                        else {
                            "census_sign_family_matched_to": (
                                census_sign_family_matched_to
                            )
                        }
                    ),
                    "analytically_resolved_route_count": int(
                        analytically_resolved_route_count
                    ),
                    "empirically_sampled_local_route_count": int(
                        measured_sign_route_count
                    ),
                    # Computed, not asserted: unsupported kinds are always
                    # sampled; the raw-input group is sampled only if no rule
                    # made it analytic.
                    "raw_input_and_unsupported_routes_remain_sampled": bool(
                        raw_input_group_sampled
                    ),
                    "zero_weight_tie_class": "nonnegative",
                },
                "census_stopping_criterion": (
                    "simultaneous_wrongsign_consequence_"
                    "upper_bound_bonferroni_blocks_v5"
                    if _census_decision_referenced()
                    else "unresolved_activity_fraction_v1"
                ),
                "expected_wrongsign_consequence_fraction": dict(
                    expected_wrongsign_consequence
                ),
                # The arm id carries family/query/granularity/direction and
                # nothing else, so two arms differing in a knob that changes
                # WHAT IS CERTIFIED -- consequence gating, the response
                # envelope, the audited cut ladder -- are indistinguishable in
                # the results tree. Receipt-only scoring then silently pools
                # them as replicates of one condition. This hash is what makes
                # that error detectable after the fact rather than never.
                "acquisition_policy_hash": (sample_size_policy.acquisition_policy_hash),
                "label_sign_unresolved_activity_fraction": dict(risks),
                "required_label_sign_views": required_views,
                "label_sign_fields": tuple(f"{view}_mean" for view in required_views),
                "loss_sign_diagnostic_only": True,
                "unresolved_activity_fraction": {"loss": final_loss_sign_diagnostic},
                "required_partition_views": ("loss",),
                "required_unresolved_activity_fraction": {
                    view: risks[view] for view in required_views
                },
                "maximum_permitted_unresolved_activity_fraction": (
                    allowed_sign_unresolved_activity
                ),
                "maximum_permitted_prune_order_disagreement_activity_fraction": (
                    allowed_order_disagreement_activity
                ),
                "prune_order_uncertainty": final_order_certificate,
                "decision_resolved": resolved,
                # Sizing disclosure: the sign forecast, its guarded delivered
                # target, and the enrolled fresh-context floor.  Which bar was
                # outstanding is disclosed alongside it below.
                "sign_bar_forecast": {
                    "schedule_target": int(schedule_target),
                    "forecast_target": (
                        None
                        if sign_forecast_target is None
                        else int(sign_forecast_target)
                    ),
                    "forecast_unresolved_fraction": (
                        None
                        if not math.isfinite(sign_forecast_fraction)
                        else float(sign_forecast_fraction)
                    ),
                    "growth_guard_applied": bool(sign_forecast_guarded),
                    "delivered_target": int(target_total),
                    "census_sizing": "enrolled_forecast_packet_floor_v1",
                    "fresh_minimum_per_look": int(sample_size_policy.response_packet),
                    "population_claim_scope": (
                        "enrolled_panel_only_pending_reuse_premise"
                    ),
                },
                "binding_bar": _response_field_binding_bar(
                    required_views=required_views,
                    risks=risks,
                    allowed_sign_unresolved_activity=(allowed_sign_unresolved_activity),
                    order_resolved=bool(order_certificate.resolved),
                    resolved=bool(resolved),
                ),
                "elapsed_seconds": float(time.perf_counter() - look_started),
                # Key present only when the census terminal rule fired, so
                # every ordinarily resolved acquisition keeps its exact
                # receipt bytes.
                **(
                    {}
                    if census_terminal_receipt is None
                    else {
                        "census_complete_terminal_resolution": (census_terminal_receipt)
                    }
                ),
            }
        )
        look_clock.mark("look_receipt")
        look_host_seconds = {**look_clock.seconds, "total": look_clock.total}
        for phase, seconds in look_host_seconds.items():
            census_look_host_seconds[phase] = (
                census_look_host_seconds.get(phase, 0.0) + float(seconds)
            )
        _progress(
            "hard_partition_look_completed",
            look_index=look_index,
            cumulative_counts=dict(existing),
            stable_looks=stable_looks,
            unresolved_activity_fraction=risks,
            required_partition_views=required_views,
            required_unresolved_activity_fraction={
                view: risks[view] for view in required_views
            },
            prune_order_uncertainty=final_order_certificate,
            decision_resolved=resolved,
            elapsed_seconds=float(time.perf_counter() - look_started),
            census_host_postprocessing_seconds=look_host_seconds,
        )
        if _residency_breakdown_armed():
            # ATTRIBUTION, per look. The census OOM (337.6 GiB, zero decisions)
            # resisted five successive localisations by reading, so the
            # dominant term is named by measurement
            # here instead. `tripartite` and `trust` are route-scale and fixed,
            # so the panel-scaled candidates are context_ids, request_ids and
            # fc_receipts, and this weighs all three against the panel that
            # produced them.
            #
            # PER LOOK, not per context: ten to twenty calls over an arm, each
            # O(entries + contexts). Disarmed by default so production arms pay
            # nothing.
            # MEASURE THE ACCUMULATORS THE CENSUS ACTUALLY USES, and measure
            # GRAPH RETENTION rather than buffer bytes.
            #
            # The first version called provider.residency_breakdown(), which
            # reports `_sequential_device_states` -- a structure the census does
            # not use: `accumulate_device_tripartite` takes the accumulator as an
            # ARGUMENT, and the census's accumulators are the local dict below.
            # It reported an empty dict, correctly, about the wrong object.
            #
            # And buffer bytes would not have shown the defect anyway. A
            # retained autograd graph is a REFERENCE CHAIN, not an allocation:
            # `sum_rows` keeps its declared size while anchoring every saved
            # activation behind it. `grad_fn` presence is the observable; bytes
            # are not.
            _progress(
                "census_residency_breakdown",
                look_index=look_index,
                cumulative_counts=dict(existing),
                accumulator_graph_retention={
                    stratum: _accumulator_graph_state(accumulator)
                    for stratum, accumulator in accumulators.items()
                },
                provider_states=provider.residency_breakdown(),
                # The enumeration above is INCOMPLETE, and its own docstring
                # says so wrongly: it reasons that "the panel-scaled terms can
                # only be context_ids, request_ids and fc_receipts", which is
                # true of the provider and false of the census. The flip-margin
                # accumulator is a local of this loop and retains a float32 row
                # per context for the whole census; the host census then weighs
                # everything neither of them knows about.
                flip_margin_residency=flip_margin_accumulator.residency_bytes(),
                host_byte_census=_host_byte_census(),
            )
        if resolved:
            stopping_reason = (
                "label_sign_and_all_loss_activity_boundaries_stable"
                if strictly_resolved
                else ("census_complete_terminal_resolution_sign_and_order_" "resolved")
            )
            break
        if sum(existing.values()) >= sum(capacity.values()):
            break
        # v5: PER-STRATUM inputs of the next look (Neyman allocation sd and
        # the sign forecast's sigma), rebuilt one stratum at a time from that
        # stratum's own accumulator. The fused look snapshot above is
        # stratum-free by design; these two consumers used to read a
        # per-stratum ``snapshots`` dict that commit 4872602a1 deleted, so
        # every continuing look raised NameError. Reached only here, after
        # both loop exits, so no resolved or census-exhausted look changes.
        # See _continuing_look_stratum_inputs for the exactness argument.
        sigma_by_stratum: dict[str, dict[int, np.ndarray]] = {}
        for stratum, accumulator in accumulators.items():
            _ensure_stratum_resident(
                accumulators,
                stratum,
                runtime.evaluator.device,
            )
            allocation_sd[stratum], sigma_by_stratum[stratum] = (
                _continuing_look_stratum_inputs(
                    accumulator,
                    population_size=len(populations[stratum]),
                    count=int(existing[stratum]),
                    sign_measurement_weights_by_head=(
                        sign_measurement_weights_by_head
                    ),
                )
            )
        if look_increment > 0:
            look_context_rates.append(
                float(time.perf_counter() - look_started) / float(look_increment)
            )
        sign_forecast_state = {
            "sigma_by_stratum": sigma_by_stratum,
            "means": {head: means[head] for head in (0, 1, 2)},
            "activities": {head: activities[head] for head in (0, 1, 2)},
        }
        previous_global_activity = np.array(
            order_source,
            dtype=np.float64,
            copy=True,
        )
        previous_partition_labels = np.array(flat_labels, dtype=np.int8, copy=True)
        census_device_look.carry_previous(
            previous_activity=previous_global_activity,
            previous_labels=previous_partition_labels,
            source_activity=order_source,
            source_labels=flat_labels,
        )
        look_index += 1

    if not resolved:
        # Which obligation was still outstanding. Reported because the two
        # bars fail for different reasons and demand different responses:
        # a sign bar short of tolerance wants more evidence, whereas an order
        # bar whose scope atom already exceeds the tolerance cannot be
        # satisfied by any amount of it (see the v3 atom diagnostic).
        order_outstanding = not (
            final_order_certificate is not None
            and bool(final_order_certificate.get("decision_resolved", False))
        )
        binding_bar = _response_field_binding_bar(
            required_views=required_views,
            risks=final_risks,
            allowed_sign_unresolved_activity=(allowed_sign_unresolved_activity),
            order_resolved=not order_outstanding,
            resolved=False,
        )
        sign_outstanding = tuple(
            str(view)
            for view in required_views
            if float(final_risks.get(str(view), 1.0)) > allowed_sign_unresolved_activity
        )
        infeasible_scopes = (
            ()
            if final_order_certificate is None
            else tuple(
                final_order_certificate.get("tolerance_infeasible_scope_ids", ())
            )
        )
        raise ResponseFieldPrecisionEnvelopeExhausted(
            {
                "schema": RESPONSE_FIELD_ENVELOPE_RECEIPT_V1,
                "stopping_reason": stopping_reason,
                "binding_bar": binding_bar,
                "decision_action": "no_field_commit",
                "required_partition_views": tuple(required_views),
                "unresolved_activity_fraction": {
                    str(view): float(value) for view, value in final_risks.items()
                },
                "maximum_permitted_unresolved_activity_fraction": float(
                    allowed_sign_unresolved_activity
                ),
                "sign_outstanding_views": tuple(sign_outstanding),
                "order_outstanding": bool(order_outstanding),
                # A nonempty tuple is the decisive read: the order bar is
                # unsatisfiable in those scopes by quantization, so more
                # evidence is not the remedy.
                "tolerance_infeasible_scope_ids": infeasible_scopes,
                "prune_order_uncertainty": final_order_certificate,
                "stable_looks": int(stable_looks),
                "required_stable_looks": int(sample_size_policy.response_stable_looks),
                "cumulative_counts": {
                    str(name): int(value) for name, value in existing.items()
                },
                "delivered_cumulative_count": int(sum(existing.values())),
                "population_count": int(sum(capacity.values())),
                "look_count": int(look_index),
                "acquisition_elapsed_seconds": float(
                    time.perf_counter() - acquisition_started
                ),
                "acquisition_envelope_forecast": envelope_forecast,
                "panel_receipts": tuple(panel_receipts),
            }
        )
    final_clock = _PhaseClock()
    census_prepare_stratum = lambda stratum: _ensure_stratum_resident(  # noqa: E731
        accumulators,
        stratum,
        runtime.evaluator.device,
    )
    decision_snapshot = None
    census_device_look.release_device_state()
    field_close = _census_drm.close_census_fields_v6(
        accumulators,
        weight_by_stratum=_binary_loss_weights(populations),
        prepare_stratum=census_prepare_stratum,
    )
    census_v6_close_receipt = dict(field_close.receipt)
    census_moment_hashes = field_close.moment_hashes
    covariance_factor_rows = int(field_close.receipt["covariance_factor_rows"])
    flat_local_mean = field_close.local_mean
    flat_global_mean = field_close.global_mean
    flat_global_activity = field_close.global_activity
    # The ``_global_head_field_statistics`` fields, the global-head
    # diagonal statistics computed on the device by the close.
    field_statistics = {
        "global_mean_standard_error": np.asarray(errors[1], dtype=np.float64).reshape(-1),
        "global_activity_standard_error": np.asarray(
            decision_snapshot_activity_errors_global, dtype=np.float64
        ).reshape(-1),
        **field_close.global_statistics,
        "loss_mean": field_close.loss_mean,
        "loss_mean_standard_error": np.asarray(errors[2], dtype=np.float64).reshape(-1),
        "loss_activity": final_order_source,
        "loss_activity_standard_error": final_order_standard_error,
        "loss_gradient_activity": field_close.loss_activity,
        **field_close.loss_statistics,
    }
    del field_close
    final_clock.mark("tripartite_finalize")
    means = _per_layer_fields_from_declared_flat(
        flat_local_mean,
        runtime=runtime,
        offsets=declared_axis_offsets,
    )
    global_means = _per_layer_fields_from_declared_flat(
        flat_global_mean,
        runtime=runtime,
        offsets=declared_axis_offsets,
    )
    if final_order_source is None or final_order_standard_error is None:
        raise RuntimeError("hard-partition acquisition closed without an order field")
    global_activity = _per_layer_fields_from_declared_flat(
        flat_global_activity, runtime=runtime, offsets=declared_axis_offsets,
    )
    ordering_statistics: dict[str, dict[str, np.ndarray]] = {
        name: _per_layer_fields_from_declared_flat(
            field_statistics[name],
            runtime=runtime,
            offsets=declared_axis_offsets,
        )
        for name in CENSUS_FIELD_STATISTIC_NAMES
    }
    final_clock.mark("per_layer_fields")
    flip_margin_geometry = flip_margin_accumulator.finalize()
    final_clock.mark("flip_margin_finalize")
    flip_margin_summary = {
        "schema": flip_margin_geometry["schema"],
        "margin_percentile_levels": list(MARGIN_PERCENTILE_LEVELS),
        "context_count_by_fc_layer": dict(
            flip_margin_geometry["context_count_by_fc_layer"]
        ),
        "margin_capture_status_by_layer": {
            layer_id: (value if isinstance(value, str) else "captured")
            for layer_id, value in flip_margin_geometry["margin_percentiles"].items()
        },
        "conv_margins": dict(flip_margin_geometry["conv_margins"]),
        "conv_flip_counters": flip_margin_geometry["conv_flip_counters"],
        "aggregation": "absolute_counts_summed_across_observed_strata",
        "decision_role": "receipts_observation_only",
    }
    phase_state = getattr(provider, "_phase_device_state", None)
    # The two moment digests hash ~13 GB of float64 arrays; they are hoisted
    # so the pass is timed and receipted, not buried in a dict literal.  The
    # v6 close computes them on a background thread: a caller that joins the
    # receipt later (``deferred_census_moment_hashes``) gets them pending.
    if _census_drm.census_moment_hashes_deferred():
        moment_hash_receipt = {
            _census_drm.CENSUS_MOMENT_HASHES_PENDING_KEY: census_moment_hashes,
        }
    else:
        moment_hash_receipt = census_moment_hashes.result()
    final_clock.mark("moment_content_hash")
    census_host_postprocessing_seconds: dict[str, float] = {
        **{
            f"look_{phase}": float(seconds)
            for phase, seconds in census_look_host_seconds.items()
        },
        **{phase: float(seconds) for phase, seconds in final_clock.seconds.items()},
    }
    return (
        means,
        global_means,
        global_activity,
        ordering_statistics,
        flip_margin_geometry,
        {
            **objective_receipt,
            "schema": "dense_pruning_loss_census_receipt_v2",
            "partition_label_views": label_views,
            **({} if phase_state is None else {
                "phase_device_scoring": phase_state.receipt(),
            }),
            **(
                {}
                if census_terminal_receipt is None
                else {"census_complete_terminal_resolution": (census_terminal_receipt)}
            ),
            **(
                {}
                if not corrected_v0_order
                else {
                    "v0_attribution_order": {
                        "schema": (
                            "v0_corrected_aumann_shapley_global_attribution_"
                            "order_receipt_v3"
                        ),
                        "mode": attribution_order,
                        "score_equation": "E_context_abs(-a*G_j*c_jr*integral_0_1((sigmoid(eta_j(t))-y)*indicator_path_j_open_dt))",
                        "relu_path_fraction": (
                            "phi_integral_0_1_indicator_b_plus_t_a_minus_b_positive_dt"
                        ),
                        "evidence_coordinate": "dense_pruning_guiding_loss",
                        "pruning_guiding_loss_id": pruning_guiding_loss_id,
                        "dense_calibration_state_hash": calibration_hash,
                        "dense_calibration_state_id": str(dense_calibration_state.calibration_state_id),
                        "loss_cotangent_path": "unitwise_bias_to_activation_frozen_target_cotangent",
                        "path_target_response": "z_t_dense+G_j*(activation_j(t)-activation_j_dense)",
                        "competitors": "frozen_at_dense_execution",
                        "path_integration": "exact_softplus_divided_difference",
                        "cotangent": "path_integrated_probability_residual_times_affine_scale",
                        "affine_calibration_applied": True,
                        "context_specific_loss_pullback_applied": True,
                        "positive_removal_orientation": True,
                        "sampling_estimand": (
                            ("population_share_weighted_per_image_" if loss_weighting() == "per_image" else
                             "one_half_on_target_plus_one_half_off_target_") +
                            "sealed_without_replacement_mean_absolute_score"
                        ),
                        "source_contexts_identical_to_incumbent_v0_field": True,
                        "incumbent_and_corrected_same_vjp": False,
                        "physical_execution": (
                            "optimized_second_same_context_forward_vjp_pass"
                        ),
                        "selection_effect": "route_activity_order_only",
                        "local_sign_partition_labels_unchanged": True,
                        "plain_local_global_loss_fields_unchanged": True,
                        "calibration_oracle_walker_semantics_unchanged": True,
                        "stream_identity_hash": corrected_order_stream_identity,
                        "final_score_hash": content_hash(
                            final_order_source,
                            prefix="v0-corrected-order-score-",
                        ),
                        "final_standard_error_hash": content_hash(
                            final_order_standard_error,
                            prefix="v0-corrected-order-standard-error-",
                        ),
                        "context_count_by_stratum": dict(existing),
                        "extra_score_context_packet_count": int(
                            sum(
                                int(
                                    panel.get(
                                        "v0_corrected_attribution_extra_score_pass",
                                        {},
                                    ).get("context_packet_count", 0)
                                )
                                for look in panel_receipts
                                for panel in look["delivered_panels"].values()
                            )
                        ),
                    }
                }
            ),
            **moment_hash_receipt,
            "census_device_v6": {
                "schema": "irg_v6_census_device_receipt_v2",
                "residency": census_v6_residency,
                "looks": census_device_look.close(),
                "close": dict(census_v6_close_receipt),
            },
            "census_host_postprocessing_seconds": census_host_postprocessing_seconds,
            "census_host_threads": census_host_threads,
            "destination_transport_hash": transport_hash,
            "context_count": int(sum(existing.values())),
            "context_count_by_stratum": dict(existing),
            "eligible_population_by_stratum": {
                key: len(value) for key, value in populations.items()
            },
            "sampling_resolution": (
                "sealed_binary_neyman_decision_specific_sequential_v2"
            ),
            "analytic_sign_partition": {
                "enabled": analytic_sign_active,
                "scope": analytic_sign_scope_label,
                "analytic_sign_scope": analytic_sign_scope,
                "analytic_sign_group_ids": list(analytic_sign_group_ids),
                "sampled_sign_group_ids": list(sampled_sign_group_ids),
                "sign_definition": sign_definition,
                "randomized_partition_count_source": randomized_partition_count_source,
                "census_sign_family_matched_to": census_sign_family_matched_to,
                "analytically_resolved_route_count": int(
                    analytically_resolved_route_count
                ),
                "empirically_sampled_local_route_count": int(measured_sign_route_count),
                "joint_simultaneous_family_size": int(simultaneous_family_size),
                "sign_simultaneous_family_size": int(sign_simultaneous_family_size),
                "order_simultaneous_family_size": int(order_simultaneous_family_size),
                "sign_normal_critical_value": float(sign_critical),
                "order_normal_critical_value": float(order_critical),
            },
            "census_stopping_criterion": (
                "simultaneous_wrongsign_consequence_" "upper_bound_bonferroni_blocks_v5"
                if _census_decision_referenced()
                else "unresolved_activity_fraction_v1"
            ),
            "stopping_reason": stopping_reason,
            "decision_resolved": resolved,
            "label_sign_unresolved_activity_fraction": final_risks,
            "required_label_sign_views": required_views,
            "label_sign_fields": tuple(f"{view}_mean" for view in required_views),
            "loss_sign_diagnostic_only": True,
            "unresolved_activity_fraction": {"loss": final_loss_sign_diagnostic},
            "prune_order_uncertainty": final_order_certificate,
            "audited_executable_cuts_by_scope": executable_boundary_fractions,
            "census_rung_profile": _census_rung_profile_receipt(census_profile),
            "census_proposal_route_cap": census_proposal_route_cap,
            "census_label_sign_family": "registered_partition_label_views",
            "pruning_strength_estimand": (
                "stratum_weighted_image_mean_absolute_dense_pruning_loss_gradient_v2"
            ),
            "prune_order_scope": str(prune_order_scope),
            "required_partition_views": ("loss",),
            "required_unresolved_activity_fraction": {
                view: final_risks[view] for view in required_views
            },
            "confidence_level": confidence,
            "alpha_k": alpha_k,
            "alpha_spending_rule": "alpha/((k+1)*(k+2))",
            "stable_looks": stable_looks,
            "sealed_permutation_hashes": {
                key: value.content_hash for key, value in permutations.items()
            },
            "covariance_sketch_rank": int(covariance_factor_rows),
            "scoring_context_packet_count": int(
                sum(
                    int(panel.get("context_packet_count", 0))
                    for look in panel_receipts
                    for panel in look["delivered_panels"].values()
                )
            ),
            "scoring_context_packet_size": provider.context_packet_size,
            "bounded_context_route_materialization": True,
            "device_sufficient_statistic_reduction": True,
            "host_context_route_rows_transferred": False,
            "local_regime_diagnostics_enabled": bool(enable_local_regime_diagnostics),
            "flip_margin_geometry": flip_margin_summary,
            "panel_receipts": tuple(panel_receipts),
        },
    )


def _global_route_axes(runtime: Any) -> tuple[tuple[str, ...], LayerCodeAxis]:
    route_count = sum(
        int(np.prod(group.shape, dtype=np.int64)) for group in runtime.route_groups
    )
    # The route-axis hash binds these compact ordinals to the physical block
    # layout; expanding coordinate JSON for every ResNet route would add no
    # identity and would dominate host memory.  The flat ordinal axis is the
    # physical (DECLARED route-block) axis, so the layer-id companion axis
    # must come from the declared authority, not the profile's group order.
    # The companion is a ``LayerCodeAxis`` (int16 codes + first-appearance
    # names): it compares equal to the per-route string tuple and
    # iterates as one, without materializing 57M Python strings per call.
    route_ids = constructive_ordinal_route_axis(route_count)
    layer_ids = LayerCodeAxis.from_blocks(
        (layer_id, stop - start)
        for layer_id, (start, stop) in _declared_route_axis_offsets(
            runtime,
            flat_length=route_count,
        ).items()
    )
    return route_ids, layer_ids


def _hard_mask_from_layers(
    masks: Mapping[str, np.ndarray],
    *,
    runtime: Any,
    route_axis_hash: str,
) -> HardMask:
    # Declared-axis assembly: the oracle providers expand a HardMask against
    # ``evaluator.route_blocks`` (the DECLARED route-block axis), so the flat
    # mask must be assembled on that same axis.  A group-order concatenation
    # would hand the providers a physically different mask wherever the
    # profile's walk order differs from the declared order (deep-first
    # AlexNet); on ResNet the orders coincide and the bytes are identical.
    incremental = getattr(masks, "_v6_hard_mask_for_layout", None)
    if incremental is not None:
        # A committed state or candidate view packs itself on
        # exactly this declared layout from its parent's cached bytes and
        # SHA states (``mask_hash`` seeded from the streamed digest of the
        # same bytes).  A layout it cannot serve returns None: full build.
        total = sum(
            int(np.prod(group.shape, dtype=np.int64)) for group in runtime.route_groups
        )
        layout = tuple(
            (layer_id, int(start), int(stop))
            for layer_id, (start, stop) in _declared_route_axis_offsets(
                runtime, flat_length=total
            ).items()
        )
        mask = incremental(route_axis_hash=route_axis_hash, layout=layout)
        if mask is not None:
            return mask
    values = _declared_flat_from_per_layer(
        masks,
        runtime=runtime,
        dtype=bool,
    )
    return HardMask.from_bool_array(values, route_axis_hash=route_axis_hash)


def _layers_from_hard_mask(
    mask: HardMask,
    *,
    runtime: Any,
) -> dict[str, np.ndarray]:
    """Invert declared-axis hard-mask assembly without group-order inference."""

    values = mask.to_bool_array()
    blocks = tuple(runtime.evaluator.route_blocks)
    expected = sum(int(np.prod(block.shape)) for block in blocks)
    if expected != values.size:
        raise ValueError("hard mask differs from the declared runtime route axis")
    offset = 0
    output: dict[str, np.ndarray] = {}
    for block in blocks:
        size = int(np.prod(block.shape))
        layer_id = str(block.layer_id)
        if layer_id in output:
            raise ValueError("declared runtime route blocks repeat a layer identity")
        output[layer_id] = values[offset : offset + size].reshape(block.shape)
        offset += size
    return output


# --------------------------------------------------------------------------
# Registered v1 trust-walk glue.
# The trust walk is the sole v1 traversal; everything below is the
# v1 path.
# --------------------------------------------------------------------------


class _TrustFanoutAccumulator:
    """Fan one device row stream into tripartite + FD trust accumulators.

    Duck-typed against ``accumulate_device_tripartite``'s consume hook,
    which reads ``route_ids``/``destination_transport_hash``/
    ``loss_pullback_hash`` (and ``head_ids`` for its panel receipt) via
    ``getattr`` and then calls ``update(rows, context_ids=...)``.  Rows
    arrive as ``[context, head, route]``; the Frequent-Directions sketch
    receives the global-head slice ``rows[:, 1:-1, :]`` — the runner-lane
    fan-out idiom — with balanced stratum weights ``0.5 /
    n_stratum_contexts`` per context, matching the fused 0.5/0.5
    estimand.  No provider file changes.
    """

    def __init__(
        self,
        *,
        tripartite: Any,
        trust: Any,
        n_stratum_contexts: int,
        stratum_weight: float = 0.5,
    ) -> None:
        contexts = int(n_stratum_contexts)
        if contexts <= 0:
            raise ValueError("trust fan-out requires a positive stratum context count")
        self.tripartite = tripartite
        self.trust = trust
        self.n_stratum_contexts = contexts
        self.fd_context_weight = float(stratum_weight) / float(contexts)

    @property
    def route_ids(self) -> tuple[str, ...]:
        return tuple(self.tripartite.route_ids)

    @property
    def head_ids(self) -> tuple[str, ...]:
        return tuple(self.tripartite.head_ids)

    @property
    def destination_transport_hash(self) -> str:
        return str(self.tripartite.destination_transport_hash)

    @property
    def loss_pullback_hash(self) -> str:
        return str(self.tripartite.loss_pullback_hash)

    def update(self, rows: Any, *, context_ids: Sequence[str]) -> None:
        ids = tuple(map(str, context_ids))
        self.tripartite.update(rows, context_ids=ids)
        self.trust.update(
            rows[:, 1:-1, :],
            context_ids=ids,
            weights=np.full(len(ids), self.fd_context_weight),
        )


def _trust_walk_scoped_field_cache_key(
    *,
    runtime: Any,
    optimization_axis: Any,
    query: Any,
    committed_mask_hash: str,
    run_id: str,
    random_seed: int,
    scoped_policy: Any,
    ordering_key: str,
    partition_family: str,
    walk_policy: Any,
    fd_reuse_threshold: float,
    fd_refresh_threshold: float,
    partition_random_seed: int | None = None,
) -> str:
    """Scoped-field cache identity (registration ``measurement_class.caching``).

    ``run_id`` and ``random_seed`` follow the census-key
    precedent: seed cells never share acquisitions.  The
    randomized control's partition seed is key material — cached results
    carry partition LABELS, so two randomized arms differing only in
    partition seed must never share an acquisition.  The census-grade
    v4 key function and its receipts are byte-untouched by this lane.
    """

    if str(ordering_key) not in V1_ORDERING_KEYS:
        raise ValueError("unregistered v1 ordering key")
    return content_hash(
        {
            "schema": _TRUST_WALK_SCOPED_FIELD_CACHE_KEY_SCHEMA,
            **({"loss_weighting": "per_image"} if loss_weighting() == "per_image" else {}),
            "execution_numerics_cache_version": DETERMINISTIC_EXECUTION_CACHE_VERSION,
            "model_state_hash": runtime.loaded.receipt.state_dict_hash,
            "route_schema_hash": runtime.evaluator.route_schema_hash,
            "topology_hash": runtime.evaluator.topology_hash,
            "optimization_axis_hash": optimization_axis.content_hash,
            "query_id": query.query_id,
            "run_id": str(run_id),
            "random_seed": int(random_seed),
            "measurement_mask_hash": str(committed_mask_hash),
            "measurement_class_id": scoped_field.SCOPED_FIELD_CLASS_ID,
            "measurement_class_numerals": scoped_policy.to_receipt(),
            "ordering_key": str(ordering_key),
            "ordering_key_registry": ORDERING_KEY_REGISTRY_VERSION,
            "partition_family": str(partition_family),
            "partition_random_seed": (
                None if partition_random_seed is None else int(partition_random_seed)
            ),
            "tranche_grid": {
                "delta_max_partition_fraction": float(
                    _V1_TRUST_WALK_AUDIT_DELTA_MAX_FRACTION
                ),
                "delta_min_routes": int(_V1_TRUST_WALK_AUDIT_DELTA_MIN_ROUTES),
                "schedule": str(_V1_TRUST_WALK_AUDIT_GRID_SCHEDULE),
            },
            "fd_spec": {
                "sketch_rank": int(scoped_policy.fd_sketch_rank),
                "maximum_rank": int(scoped_policy.fd_maximum_rank),
                "reuse_threshold": float(fd_reuse_threshold),
                "refresh_threshold": float(fd_refresh_threshold),
            },
            "fanout_implementation": _TRUST_WALK_FANOUT_IMPLEMENTATION_ID,
        },
        prefix="irg-trust-walk-scoped-field-cache-",
    )


_TRUST_WALK_EXTRA_STATISTIC_NAMES = tuple(
    name
    for name in ORDERING_FIELD_STATISTIC_NAMES
    if name not in ("global_mean_standard_error", "global_activity_standard_error")
)


def _store_trust_walk_scoped_field_cache(
    *,
    payload_path: Path,
    receipt_path: Path,
    cache_key: str,
    result: Any,
    measurement_receipt: Mapping[str, Any],
) -> None:
    """Persist one scoped acquisition: field arrays + FD pack + factor.

    The payload additionally carries ``partition_labels`` — for the
    local-family control the labels derive from the chart-fresh local
    head, which the ``FieldSnapshot`` deliberately does not carry, so a
    cache hit could not otherwise rebuild them.
    """

    snapshot = result.snapshot
    fd_values, fd_layout = snapshot.fd_state.pack()
    spec = fd_layout["spec"]
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    statistics = dict(snapshot.statistics)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{cache_key}.",
        suffix=".npz",
        dir=payload_path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(
                stream,
                global_mean=snapshot.global_mean,
                global_mean_standard_error=(snapshot.global_mean_standard_error),
                global_activity=snapshot.global_activity,
                global_activity_standard_error=(
                    snapshot.global_activity_standard_error
                ),
                **{
                    name: statistics[name] for name in _TRUST_WALK_EXTRA_STATISTIC_NAMES
                },
                fd_pack=fd_values,
                rademacher_global_factor=(snapshot.rademacher_global_factor),
                partition_labels=np.asarray(result.partition_labels, dtype=np.int8),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, payload_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    receipt = {
        "schema": _TRUST_WALK_SCOPED_FIELD_CACHE_RECEIPT_SCHEMA,
        "cache_key": cache_key,
        "payload_sha256": sha256_file(payload_path),
        "route_count": int(snapshot.route_count),
        "contexts_consumed": int(snapshot.contexts_consumed),
        "census_exhausted": bool(snapshot.census_exhausted),
        "looks_taken": int(result.looks_taken),
        "horizon": {
            str(key): int(value) for key, value in dict(result.horizon).items()
        },
        "fd_layout": {
            "mean_rows": int(fd_layout["mean_rows"]),
            "factor_rows": int(fd_layout["factor_rows"]),
            "residual_diagonal_rows": int(fd_layout["residual_diagonal_rows"]),
            "residual_spectral_bound": float(fd_layout["residual_spectral_bound"]),
            "total_weight": float(fd_layout["total_weight"]),
            "context_count": int(fd_layout["context_count"]),
            "context_axis_hash": str(fd_layout["context_axis_hash"]),
            "measured_mask_hash": str(fd_layout["measured_mask_hash"]),
            "calibration_state_id": str(fd_layout["calibration_state_id"]),
            "source_measurement_hashes": [
                str(value) for value in fd_layout["source_measurement_hashes"]
            ],
            "metadata": dict(fd_layout["metadata"]),
            "route_ids_count": len(snapshot.fd_state.route_ids),
            "route_ids_hash": ordered_string_axis_hash(
                snapshot.fd_state.route_ids,
                name="global trust route axis",
            ),
            "spec": {
                "trust_id": str(spec.trust_id),
                "initial_rank": int(spec.initial_rank),
                "maximum_rank": int(spec.maximum_rank),
                "reuse_threshold": float(spec.reuse_threshold),
                "refresh_threshold": float(spec.refresh_threshold),
                "residual_policy": str(spec.residual_policy),
                "metadata": dict(spec.metadata),
            },
        },
        "scoped_receipt": dict(result.receipt),
        "measurement_receipt": dict(measurement_receipt),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{cache_key}.",
        suffix=".json",
        dir=receipt_path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({**receipt, "effective_settings": effective_settings()}, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, receipt_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _load_trust_walk_scoped_field_cache(
    *,
    payload_path: Path,
    receipt_path: Path,
    cache_key: str,
    route_ids: tuple[str, ...],
    ordering_key: str,
) -> tuple[Any, dict[str, Any]] | None:
    """Rebuild one cached scoped acquisition, refusing foreign receipts.

    Mirrors the v3 census sidecar idiom: an absent cache is a miss, but a
    receipt with the wrong schema, key, or payload digest is corruption
    and refuses loudly (the census-grade v4 receipt schema lands here as
    a refusal, never a silent miss into the wrong measurement era).
    """

    if not payload_path.exists() or not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("schema") != _TRUST_WALK_SCOPED_FIELD_CACHE_RECEIPT_SCHEMA
        or receipt.get("cache_key") != cache_key
        or receipt.get("payload_sha256") != sha256_file(payload_path)
    ):
        raise ValueError("trust-walk scoped-field cache receipt is invalid")
    expected = len(route_ids)
    with np.load(payload_path, allow_pickle=False) as payload:
        arrays = {
            name: np.asarray(payload[name], dtype=np.float64)
            for name in (
                "global_mean",
                "global_mean_standard_error",
                "global_activity",
                "global_activity_standard_error",
                *_TRUST_WALK_EXTRA_STATISTIC_NAMES,
            )
        }
        fd_pack = np.asarray(payload["fd_pack"], dtype=np.float64)
        rademacher = np.asarray(payload["rademacher_global_factor"], dtype=np.float64)
        labels = np.asarray(payload["partition_labels"], dtype=np.int8)
    if (
        any(value.shape != (expected,) for value in arrays.values())
        or labels.shape != (expected,)
        or not np.isin(labels, (0, 1)).all()
        or rademacher.ndim != 2
        or rademacher.shape[1] != expected
    ):
        raise ValueError("trust-walk scoped-field cache route axis is invalid")
    fd_layout = dict(receipt["fd_layout"])
    spec_fields = dict(fd_layout["spec"])
    spec = GlobalTrustGeometrySpec(
        trust_id=str(spec_fields["trust_id"]),
        initial_rank=int(spec_fields["initial_rank"]),
        maximum_rank=int(spec_fields["maximum_rank"]),
        reuse_threshold=float(spec_fields["reuse_threshold"]),
        refresh_threshold=float(spec_fields["refresh_threshold"]),
        residual_policy=str(spec_fields["residual_policy"]),
        metadata=dict(spec_fields["metadata"]),
    )
    fd_state = GlobalTrustGeometryState.unpack(
        fd_pack,
        {
            "schema": "irg_global_trust_geometry_state_v1",
            "spec": spec,
            "axis_reference_schema": "ordered_string_axis_ref_v1",
            "route_ids_count": int(fd_layout["route_ids_count"]),
            "route_ids_hash": str(fd_layout["route_ids_hash"]),
            "mean_rows": int(fd_layout["mean_rows"]),
            "factor_rows": int(fd_layout["factor_rows"]),
            "residual_diagonal_rows": int(fd_layout["residual_diagonal_rows"]),
            "residual_spectral_bound": float(fd_layout["residual_spectral_bound"]),
            "total_weight": float(fd_layout["total_weight"]),
            "context_count": int(fd_layout["context_count"]),
            "context_axis_hash": str(fd_layout["context_axis_hash"]),
            "measured_mask_hash": str(fd_layout["measured_mask_hash"]),
            "calibration_state_id": str(fd_layout["calibration_state_id"]),
            "source_measurement_hashes": tuple(fd_layout["source_measurement_hashes"]),
            "metadata": dict(fd_layout["metadata"]),
        },
        route_ids=route_ids,
    )
    statistics = {
        "global_mean_standard_error": arrays["global_mean_standard_error"],
        "global_activity_standard_error": arrays["global_activity_standard_error"],
        **{name: arrays[name] for name in _TRUST_WALK_EXTRA_STATISTIC_NAMES},
    }
    snapshot = scoped_field.FieldSnapshot(
        global_mean=arrays["global_mean"],
        global_mean_standard_error=arrays["global_mean_standard_error"],
        global_activity=arrays["global_activity"],
        global_activity_standard_error=arrays["global_activity_standard_error"],
        statistics=statistics,
        fd_state=fd_state,
        rademacher_global_factor=rademacher,
        contexts_consumed=int(receipt["contexts_consumed"]),
        census_exhausted=bool(receipt["census_exhausted"]),
    )
    key_values, key_standard_errors = ordering_key_statistic(
        str(ordering_key),
        GlobalHeadOrderingField.from_statistics(
            global_mean=snapshot.global_mean,
            global_activity=snapshot.global_activity,
            statistics=snapshot.statistics,
        ),
    )
    scoped_receipt = dict(receipt["scoped_receipt"])
    result = scoped_field.ScopedFieldResult(
        snapshot=snapshot,
        partition_labels=labels,
        key_values=key_values,
        key_standard_errors=key_standard_errors,
        horizon=dict(receipt["horizon"]),
        order_certificate=dict(scoped_receipt["order_certificate"]),
        receipt=scoped_receipt,
        looks_taken=int(receipt["looks_taken"]),
    )
    metadata = dict(receipt["measurement_receipt"])
    metadata.update(
        {
            "trust_walk_scoped_field_cache_key": cache_key,
            "trust_walk_scoped_field_cache_hit": True,
            "trust_walk_scoped_field_cache_payload_sha256": receipt["payload_sha256"],
        }
    )
    return result, metadata


def _acquire_trust_walk_chart_field(
    *,
    reachable_budget: float | None = None,
    resolution_floor: float,
    resolution_floor_source: str,
    partition_random_seed: int | None = None,
    runtime: Any,
    optimization_axis: Any,
    query: Any,
    committed_mask: HardMask,
    committed_flat_retained: np.ndarray,
    route_ids: tuple[str, ...],
    target_class_id: int,
    decision_provider: Any,
    run_id: str,
    random_seed: int,
    sample_size_policy: IndependentSampleSizePolicy,
    partition_family: PilotPartitionFamily,
    ordering_key: str,
    walk_policy: Any,
    fd_sketch_rank: int,
    fd_maximum_rank: int,
    cache_dir: Path | None,
    chart_index: int,
    progress_callback: Callable[..., Any] = _progress,
) -> tuple[Any, dict[str, Any]]:
    """Measure (or load) one chart's decision-scoped field at the anchor.

    The measurement mirrors the census acquisition's machinery — sealed
    permutations, unified Neyman allocation, the device tripartite panel
    path, fused 0.5/0.5 stratum estimand — but is bound to the COMMITTED
    mask and its frozen accepted calibration fit, streams the global-head
    rows into the shared Frequent-Directions fan-out, and stops under the
    registered decision-scoped class instead of the census family.
    """

    frozen_fit = decision_provider.calibration_fits.get(committed_mask.mask_hash)
    if frozen_fit is None:
        raise RuntimeError(
            "trust-walk chart field requires the committed mask's frozen "
            "accepted calibration fit"
        )
    if str(frozen_fit.mask_hash) != committed_mask.mask_hash or not bool(
        frozen_fit.numerical_converged
    ):
        raise ValueError(
            "trust-walk chart scoring requires the converged frozen "
            "calibration fit of its anchor mask"
        )

    populations_namespace = binary_evidence_role_namespace(
        run_id=run_id,
        random_seed=random_seed,
        context_axis_hash=optimization_axis.content_hash,
    )
    sampling_design = compile_binary_iid_sampling_design(
        optimization_axis,
        target_class_id=target_class_id,
        randomization_namespace=populations_namespace,
    )
    populations = {
        stratum: tuple(
            map(
                str,
                sampling_design.source_context_ids_by_stratum[stratum],
            )
        )
        for stratum in ("on_target", "off_target")
    }
    if any(not population for population in populations.values()):
        raise ValueError("trust-walk chart field has no response contexts")
    capacity = {stratum: len(population) for stratum, population in populations.items()}
    initial = int(sample_size_policy.response_initial_packet)
    if any(value < initial for value in capacity.values()):
        raise ValueError(
            "trust-walk chart response support cannot supply the initial " "packet"
        )
    # The census look-split level on this axis (the C2 policy consumes it
    # directly as the per-look simultaneous confidence).
    maximum_looks = max(
        1,
        1
        + math.ceil(
            max(sum(capacity.values()) - 2 * initial, 0)
            / max(int(sample_size_policy.response_packet), 1)
        ),
    )
    look_confidence = 1.0 - (1.0 - _V1_CHART_PROPOSAL_CONFIDENCE) / float(maximum_looks)
    scoped_policy = scoped_field.ScopedFieldPolicy(
        fd_sketch_rank=int(fd_sketch_rank),
        fd_maximum_rank=int(fd_maximum_rank),
        confidence_level=float(look_confidence),
    )
    if (
        str(partition_family) == "randomized_count_matched"
        and partition_random_seed is None
    ):
        raise ValueError("randomized trust-walk charts require a frozen partition seed")
    cache_key = _trust_walk_scoped_field_cache_key(
        runtime=runtime,
        optimization_axis=optimization_axis,
        query=query,
        committed_mask_hash=committed_mask.mask_hash,
        run_id=run_id,
        random_seed=random_seed,
        scoped_policy=scoped_policy,
        ordering_key=ordering_key,
        partition_family=partition_family,
        walk_policy=walk_policy,
        fd_reuse_threshold=float(_V1_TRUST_WALK_CHART_LEVERAGE_CAP),
        fd_refresh_threshold=_V1_TRUST_WALK_FD_REFRESH_THRESHOLD,
        partition_random_seed=partition_random_seed,
    )

    def _measure() -> tuple[Any, dict[str, Any]]:
        permutations = {
            stratum: SealedUniformPermutationReceipt.create(
                population,
                entropy_hex=content_hash(
                    {
                        "run_id": run_id,
                        "stratum": stratum,
                        "role": _TRUST_WALK_CHART_FIELD_ROLE,
                        "mask_hash": committed_mask.mask_hash,
                    }
                )[-64:],
            )
            for stratum, population in populations.items()
        }
        selected_population = tuple(
            context_id
            for stratum in ("on_target", "off_target")
            for context_id in populations[stratum]
        )
        destination_scales, transport_hash = resnet_local_destination_transport(
            runtime.evaluator
        )
        provider = OptimizedScoringResponseProvider(
            evaluator=runtime.evaluator,
            measurement_view=optimization_axis.data_view(),
            context_ids=selected_population,
            batch_factory=(
                runtime.optimization_data_plane.role_batch_factory("scoring")
            ),
            mask_resolver=lambda mask_hash: (
                committed_mask
                if str(mask_hash) == committed_mask.mask_hash
                else (_ for _ in ()).throw(KeyError(mask_hash))
            ),
            calibration_resolver=lambda mask: (
                frozen_fit
                if mask.mask_hash == committed_mask.mask_hash
                else (_ for _ in ()).throw(KeyError(mask.mask_hash))
            ),
            route_ids=route_ids,
            queries=(query,),
            response_spec_hash=content_hash(
                {
                    "query_id": query.query_id,
                    "semantics": _TRUST_WALK_RESPONSE_SEMANTICS,
                }
            ),
            response_metric_id=_TRUST_WALK_RESPONSE_METRIC_ID,
            study_hash=optimization_axis.content_hash,
            apply_affine_calibration_to_response=True,
            local_destination_scales_by_layer=destination_scales,
            local_destination_transport_hash=transport_hash,
            local_covariance_sketch_rank=_local_covariance_sketch_rank(),
            local_covariance_sketch_seed=random_seed,
            local_regime_specs_by_layer={},
        )
        loss_pullback_hash = content_hash(
            {
                "schema": "frozen_affine_bce_loss_pullback_v1",
                "query_id": query.query_id,
                "target_class_id": int(target_class_id),
                "calibration_state_id": (frozen_fit.calibration_state_id),
            },
            prefix="irg-loss-pullback-",
        )
        head_ids = (
            "local_destination",
            f"global:{query.query_id}",
            "loss:bce",
        )
        shared_trust = DeviceFrequentDirectionsTrustAccumulator(
            route_ids=route_ids,
            sketch_rank=int(fd_sketch_rank),
            device=runtime.evaluator.device,
        )
        fanouts = {
            stratum: _TrustFanoutAccumulator(
                tripartite=DeviceTripartiteMomentAccumulator(
                    head_ids=head_ids,
                    route_ids=route_ids,
                    destination_transport_hash=transport_hash,
                    loss_pullback_hash=loss_pullback_hash,
                    sketch_rank=_local_covariance_sketch_rank(),
                    sketch_seed=random_seed,
                    device=runtime.evaluator.device,
                ),
                trust=shared_trust,
                n_stratum_contexts=capacity[stratum],
                stratum_weight=_binary_loss_weights(populations)[stratum],
            )
            for stratum in ("on_target", "off_target")
        }
        trust_spec = GlobalTrustGeometrySpec(
            trust_id=_V1_TRUST_WALK_FD_TRUST_ID,
            initial_rank=int(fd_sketch_rank),
            maximum_rank=int(fd_maximum_rank),
            reuse_threshold=float(_V1_TRUST_WALK_CHART_LEVERAGE_CAP),
            refresh_threshold=_V1_TRUST_WALK_FD_REFRESH_THRESHOLD,
        )
        allocator = UnifiedNeymanAllocator()
        existing = {stratum: 0 for stratum in fanouts}
        allocation_sd = {stratum: 1.0 for stratum in fanouts}
        request_ids: list[str] = []
        panel_receipts: list[Mapping[str, Any]] = []
        local_mean_cell: dict[str, np.ndarray] = {}

        def _deliver(look_index: int) -> None:
            total_existing = sum(existing.values())
            total_capacity = sum(capacity.values())
            if total_existing >= total_capacity:
                return
            if total_existing > 0:
                for stratum, fanout in fanouts.items():
                    snapshot_now = fanout.tripartite.decision_snapshot(
                        population_size=len(populations[stratum])
                    )
                    decision_variance = (
                        snapshot_now.contextual_variance[0]
                        + snapshot_now.contextual_variance[1]
                    )
                    decision_activity = (
                        snapshot_now.activity_mean[0] + snapshot_now.activity_mean[1]
                    )
                    denominator = max(
                        float(np.sum(decision_activity)),
                        np.finfo(np.float64).tiny,
                    )
                    allocation_sd[stratum] = math.sqrt(
                        max(
                            float(
                                np.sum(decision_activity * decision_variance)
                                / denominator
                            ),
                            1e-12,
                        )
                    )
            target_total = (
                2 * initial
                if total_existing == 0
                else min(
                    total_capacity,
                    max(
                        total_existing + int(sample_size_policy.response_packet),
                        2 * total_existing,
                    ),
                )
            )
            allocation = allocator.allocate(
                binary_neyman_request(
                    total_count=target_total,
                    available_by_stratum=capacity,
                    minimum_by_stratum={
                        "on_target": initial,
                        "off_target": initial,
                    },
                    existing_by_stratum=existing,
                    sd_by_stratum=allocation_sd,
                    weight_by_stratum=_binary_loss_weights(populations),
                    role=_TRUST_WALK_CHART_FIELD_ROLE,
                )
            )
            look_started = time.perf_counter()
            delivered: dict[str, Mapping[str, object]] = {}
            for stratum in ("on_target", "off_target"):
                start = existing[stratum]
                stop = int(allocation.target_counts[stratum])
                addition = _sealed_prefix_addition(
                    permutations[stratum],
                    start=start,
                    stop=stop,
                )
                if not addition:
                    continue
                progress_callback(
                    "trust_walk_chart_panel_started",
                    chart_index=int(chart_index),
                    look_index=int(look_index),
                    stratum=stratum,
                    previous_count=start,
                    target_count=stop,
                    addition_count=len(addition),
                )
                request = MeasurementRequest.create(
                    kind="response_moments",
                    state_hash=content_hash(
                        {
                            "run_id": run_id,
                            "mask_hash": committed_mask.mask_hash,
                            "stratum": stratum,
                            "look_index": int(look_index),
                            "role": _TRUST_WALK_CHART_FIELD_ROLE,
                        }
                    ),
                    version="v1",
                    context_selector=ContextSelector(
                        selector_id=("sealed-trust-walk-chart-field-addition-v1"),
                        requested_count=len(addition),
                        context_ids=addition,
                    ),
                    expected_cost=CostEstimate(
                        forward_passes=float(len(addition)),
                        backward_passes=float(len(addition)),
                    ),
                    expected_decision_value=0.0,
                    provenance=MeasurementProvenance(
                        producer_id=(
                            "registered_resnet18_trust_walk_chart_"
                            "field_device_moments"
                        ),
                        producer_version="1",
                        code_hash=content_hash(
                            {
                                "implementation": (
                                    "fused_u_g_d_device_sufficient_" "statistics_v2"
                                )
                            }
                        ),
                    ),
                    payload={
                        "mask_hash": committed_mask.mask_hash,
                        "whole_physical_route_axis": True,
                        "stratum_id": stratum,
                        "look_index": int(look_index),
                    },
                )
                receipt = provider.accumulate_device_tripartite(
                    request,
                    accumulator=fanouts[stratum],
                )
                request_ids.append(request.request_id)
                delivered[stratum] = receipt
                existing[stratum] = stop
                progress_callback(
                    "trust_walk_chart_panel_completed",
                    chart_index=int(chart_index),
                    look_index=int(look_index),
                    stratum=stratum,
                    cumulative_count=stop,
                    **dict(receipt),
                )
            panel_receipts.append(
                {
                    "schema": "trust_walk_chart_field_look_receipt_v1",
                    "look_index": int(look_index),
                    "allocation": allocation.to_receipt(),
                    "delivered_panels": delivered,
                    "cumulative_counts": dict(existing),
                    "elapsed_seconds": float(time.perf_counter() - look_started),
                }
            )

        def _snapshot() -> Any:
            snapshots = {
                stratum: fanout.tripartite.decision_snapshot(
                    population_size=len(populations[stratum])
                )
                for stratum, fanout in fanouts.items()
            }
            if loss_weighting() == "balanced":
                global_mean = 0.5 * (
                    snapshots["on_target"].mean[1] + snapshots["off_target"].mean[1]
                )
                global_mean_se = np.sqrt(
                    0.25
                    * (
                        snapshots["on_target"].mean_standard_error[1] ** 2
                        + snapshots["off_target"].mean_standard_error[1] ** 2
                    )
                )
                global_activity = 0.5 * (
                    snapshots["on_target"].activity_mean[1]
                    + snapshots["off_target"].activity_mean[1]
                )
                global_activity_se = np.sqrt(
                    0.25
                    * (
                        snapshots["on_target"].activity_mean_standard_error[1] ** 2
                        + snapshots["off_target"].activity_mean_standard_error[1] ** 2
                    )
                )
                local_mean_cell["value"] = 0.5 * (
                    snapshots["on_target"].mean[0] + snapshots["off_target"].mean[0]
                )
            else:
                weights = _binary_loss_weights(populations)
                on, off = weights["on_target"], weights["off_target"]
                global_mean = on * snapshots["on_target"].mean[1] + off * snapshots["off_target"].mean[1]
                global_mean_se = np.sqrt(on ** 2 * snapshots["on_target"].mean_standard_error[1] ** 2
                                         + off ** 2 * snapshots["off_target"].mean_standard_error[1] ** 2)
                global_activity = on * snapshots["on_target"].activity_mean[1] + off * snapshots["off_target"].activity_mean[1]
                global_activity_se = np.sqrt(on ** 2 * snapshots["on_target"].activity_mean_standard_error[1] ** 2
                                             + off ** 2 * snapshots["off_target"].activity_mean_standard_error[1] ** 2)
                local_mean_cell["value"] = on * snapshots["on_target"].mean[0] + off * snapshots["off_target"].mean[0]
            _moments, fused = combine_stratified_tripartite_moments(
                {
                    stratum: fanout.tripartite.finalize()
                    for stratum, fanout in fanouts.items()
                },
                weight_by_stratum=_binary_loss_weights(populations),
            )
            statistics = _global_head_field_statistics(
                fused,
                global_mean_standard_error=global_mean_se,
                global_activity_standard_error=global_activity_se,
            )
            global_head = fused.head_ids.index(
                next(
                    head_id
                    for head_id in fused.head_ids
                    if head_id.startswith("global:")
                )
            )
            fd_state = scoped_field.finalize_fd_snapshot(
                shared_trust,
                spec=trust_spec,
                measured_mask_hash=committed_mask.mask_hash,
                calibration_state_id=str(frozen_fit.calibration_state_id),
                source_measurement_hashes=tuple(request_ids),
                metadata={"requested_rank": int(fd_sketch_rank)},
            )
            return scoped_field.FieldSnapshot(
                global_mean=global_mean,
                global_mean_standard_error=global_mean_se,
                global_activity=global_activity,
                global_activity_standard_error=global_activity_se,
                statistics=statistics,
                fd_state=fd_state,
                rademacher_global_factor=fused.covariance_factor[:, global_head, :],
                contexts_consumed=int(sum(existing.values())),
                census_exhausted=bool(sum(existing.values()) >= sum(capacity.values())),
            )

        permutation_cell: dict[str, np.ndarray] = {}

        def _labeler(snap: Any) -> np.ndarray:
            if partition_family == "single_population":
                # The k=1 family — one queue; the walk's
                # single-queue mode consumes the lone partition.
                return np.zeros(int(np.asarray(snap.global_mean).size), dtype=np.int8)
            if partition_family == "global_attribution_sign":
                return np.where(np.asarray(snap.global_mean) >= 0.0, 0, 1).astype(
                    np.int8
                )
            if partition_family == "randomized_count_matched":
                # The registered randomized control at chart scope
                # — membership is a frozen per-chart permutation, the
                # label-1 block count-matched to the CURRENT negative
                # sign class each look (nested along the permutation).
                permutation = permutation_cell.get("value")
                if permutation is None:
                    permutation = _trust_walk_randomized_partition_permutation(
                        partition_random_seed=int(
                            partition_random_seed  # validated above
                        ),
                        run_id=run_id,
                        committed_mask_hash=committed_mask.mask_hash,
                        route_count=int(np.asarray(snap.global_mean).size),
                    )
                    permutation_cell["value"] = permutation
                return _count_matched_random_labels(snap.global_mean, permutation)
            local_mean = local_mean_cell.get("value")
            if local_mean is None:
                raise RuntimeError(
                    "trust-walk scoped labeler ran before its " "measurement stream"
                )
            return np.where(local_mean >= 0.0, 0, 1).astype(np.int8)

        def _tranche_grid(order: np.ndarray, horizon: int) -> tuple[int, ...]:
            indices = np.asarray(order, dtype=np.int64)
            retained_count = int(np.count_nonzero(committed_flat_retained[indices]))
            delta_start = max(
                int(_V1_TRUST_WALK_AUDIT_DELTA_MIN_ROUTES),
                math.floor(
                    float(_V1_TRUST_WALK_AUDIT_DELTA_MAX_FRACTION) * retained_count
                ),
            )
            grid = _audit_delta_grid(
                delta_start=delta_start,
                delta_min_routes=int(_V1_TRUST_WALK_AUDIT_DELTA_MIN_ROUTES),
            )
            return tuple(
                dict.fromkeys(
                    int(value) for value in grid if 1 <= int(value) <= int(horizon)
                )
            )

        result = scoped_field.measure_decision_scoped_field(
            deliver_panel=_deliver,
            snapshot=_snapshot,
            partition_labeler=_labeler,
            partition_ids=CANONICAL_TRUST_WALK_PARTITION_IDS,
            ordering_key=str(ordering_key),
            tranche_grid=_tranche_grid,
            leverage_cap=float(_V1_TRUST_WALK_CHART_LEVERAGE_CAP),
            resolution_floor=float(resolution_floor),
            reachable_budget=reachable_budget,
            policy=scoped_policy,
            progress=progress_callback,
        )
        measurement_receipt: dict[str, Any] = {
            "schema": "trust_walk_chart_field_acquisition_v1",
            "chart_index": int(chart_index),
            "anchor_mask_hash": committed_mask.mask_hash,
            "frozen_calibration_state_id": str(frozen_fit.calibration_state_id),
            "response_identity": {
                "response_spec_semantics": (_TRUST_WALK_RESPONSE_SEMANTICS),
                "response_metric_id": _TRUST_WALK_RESPONSE_METRIC_ID,
                "role": _TRUST_WALK_CHART_FIELD_ROLE,
            },
            "sampling_resolution": ("sealed_binary_neyman_decision_scoped_adaptive_v1"),
            "look_confidence_level": float(look_confidence),
            "maximum_planned_scoring_looks": int(maximum_looks),
            "resolution_floor": float(resolution_floor),
            "resolution_floor_source": str(resolution_floor_source),
            "partition_randomization": (
                None
                if partition_family != "randomized_count_matched"
                else {
                    "family": "randomized_count_matched",
                    "claim_status": "exploratory_ablation",
                    "randomization_algorithm": ("numpy_pcg64_permutation_v1"),
                    "count_matched_to_sign_partition": True,
                    "partition_random_seed": int(partition_random_seed),
                    "seed_material_role": (_TRUST_WALK_RANDOMIZED_PARTITION_ROLE),
                    "per_chart_redraw": ("committed_mask_hash_in_seed_material"),
                    "permutation_content_hash": (
                        None
                        if permutation_cell.get("value") is None
                        else content_hash(
                            permutation_cell["value"],
                            prefix=("trust-walk-randomized-partition-" "permutation-"),
                        )
                    ),
                }
            ),
            "fd_weighting": {
                "scheme": ("per_image_population_mean" if loss_weighting() == "per_image"
                           else "per_context_half_over_stratum_population_v1"),
                **({"loss_weighting": "per_image"} if loss_weighting() == "per_image" else {}),
                "per_context_weight_by_stratum": {
                    stratum: float(fanout.fd_context_weight)
                    for stratum, fanout in fanouts.items()
                },
                "exactness": (
                    "exact_population_shares_at_census_exhaustion; pre-exhaustion masses follow delivered fractions"
                    if loss_weighting() == "per_image" else
                    "exact_fused_0p5_0p5_at_census_exhaustion; "
                    "pre-exhaustion stratum masses tilt by the "
                    "delivered-fraction ratio (receipted counts)"
                ),
            },
            "context_count": int(sum(existing.values())),
            "context_count_by_stratum": dict(existing),
            "eligible_population_by_stratum": {
                key: len(value) for key, value in populations.items()
            },
            "panel_deliveries": tuple(panel_receipts),
            "fanout_implementation": (_TRUST_WALK_FANOUT_IMPLEMENTATION_ID),
            "trust_walk_scoped_field_cache_key": cache_key,
            "trust_walk_scoped_field_cache_hit": False,
        }
        return result, measurement_receipt

    if cache_dir is None:
        return _measure()

    import fcntl

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    payload_path = cache_root / f"{cache_key}.npz"
    receipt_path = cache_root / f"{cache_key}.json"
    lock_path = cache_root / f"{cache_key}.lock"
    progress_callback(
        "trust_walk_chart_field_cache_waiting",
        trust_walk_scoped_field_cache_key=cache_key,
        lock_path=str(lock_path),
    )
    with lock_path.open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        progress_callback(
            "trust_walk_chart_field_cache_lock_acquired",
            trust_walk_scoped_field_cache_key=cache_key,
        )
        try:
            cached = _load_trust_walk_scoped_field_cache(
                payload_path=payload_path,
                receipt_path=receipt_path,
                cache_key=cache_key,
                route_ids=route_ids,
                ordering_key=ordering_key,
            )
            if cached is None:
                result, measurement_receipt = _measure()
                _store_trust_walk_scoped_field_cache(
                    payload_path=payload_path,
                    receipt_path=receipt_path,
                    cache_key=cache_key,
                    result=result,
                    measurement_receipt=measurement_receipt,
                )
                measurement_receipt["trust_walk_scoped_field_cache_payload_sha256"] = (
                    sha256_file(payload_path)
                )
            else:
                result, measurement_receipt = cached
        finally:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
    return result, measurement_receipt


def _build_trust_walk_chart_builder(
    *,
    dense_baseline_certificate: Any,
    audit_population_images: int,
    runtime: Any,
    optimization_axis: Any,
    query: Any,
    route_ids: tuple[str, ...],
    route_layer_ids: tuple[str, ...],
    route_axis_hash: str,
    target_class_id: int,
    decision_provider: Any,
    run_id: str,
    random_seed: int,
    sample_size_policy: IndependentSampleSizePolicy,
    partition_family: PilotPartitionFamily,
    ordering_key: str,
    walk_policy: Any,
    cache_dir: str | Path | None,
    initial_partition: str,
    partition_random_seed: int | None = None,
    progress_callback: Callable[..., Any] = _progress,
) -> Callable[[Mapping[str, np.ndarray], int], Any]:
    """Close over the pilot runtime and return the walk's chart builder.

    The closure owns two pieces of walk-scoped mutable state: the FD
    sketch rank requested for the NEXT acquisition, and any pending
    rank-growth obligation (registered semantics: an obligation never
    blocks a step; it doubles the sketch rank, up to the maximum, at the
    next chart build and is receipted as honored).

    The walk passes its measured resolution floor (delta_min) into
    every build; before the first measured decision that floor is 0.0,
    and the builder seeds it from the dense baseline certificate's
    measured spread through the walk's own registered formula (z_c x
    sd / sqrt(panel rail) x finite-population correction).  The
    baseline (unpaired) spread dominates the paired spread, so the seed
    is loose exactly when the walk knows least and tightens the moment
    real paired evidence exists.
    """

    if str(initial_partition) not in {"positive", "negative"}:
        raise ValueError("initial partition must be positive or negative")
    base_policy = scoped_field.ScopedFieldPolicy()
    state: dict[str, Any] = {
        "requested_rank": int(base_policy.fd_sketch_rank),
        "pending_obligation": None,
    }
    resolved_cache_dir = None if cache_dir is None else Path(cache_dir)
    # One derivation, shared with the v0 census.  The kernel raises on a
    # non-positive spread.
    try:
        seed_resolution_floor = deciding_oracle_resolution_floor(
            context_standard_error=float(dense_baseline_certificate.standard_error),
            context_count=int(dense_baseline_certificate.context_count),
            panel_rail=int(walk_policy.panel_maximum_contexts),
            population_size=int(audit_population_images),
            confidence_level=float(sample_size_policy.confidence_level),
        )
    except ValueError as error:
        raise ValueError(
            "trust-walk chart builder requires a dense baseline "
            "certificate with a positive measured spread"
        ) from error

    def _chart_builder(
        committed_masks_by_layer: Mapping[str, np.ndarray],
        chart_index: int,
        *,
        remaining_budget: float | None = None,
        resolution_floor: float | None = None,
    ) -> Any:
        walk_floor = (
            float(resolution_floor)
            if resolution_floor is not None
            and math.isfinite(float(resolution_floor))
            and float(resolution_floor) > 0.0
            else None
        )
        effective_resolution_floor = (
            walk_floor if walk_floor is not None else float(seed_resolution_floor)
        )
        resolution_floor_source = (
            "walk_measured_paired_spread"
            if walk_floor is not None
            else "dense_baseline_certificate_spread_seed"
        )
        committed_mask = _hard_mask_from_layers(
            committed_masks_by_layer,
            runtime=runtime,
            route_axis_hash=route_axis_hash,
        )
        # Declared-axis assembly: the scoped-field partition orders index
        # the acquisition provider's flat route axis (declared route-block
        # order), and ``_tranche_grid`` reads this array at exactly those
        # indices — both sides must live on the declared axis.
        committed_flat_retained = _declared_flat_from_per_layer(
            committed_masks_by_layer,
            runtime=runtime,
            dtype=bool,
        )
        rank_growth_receipt: dict[str, Any] | None = None
        requested_rank = int(state["requested_rank"])
        pending = state["pending_obligation"]
        if pending is not None:
            maximum_rank = int(base_policy.fd_maximum_rank)
            at_maximum = requested_rank >= maximum_rank
            granted_rank = (
                requested_rank
                if at_maximum
                else min(
                    maximum_rank,
                    max(2 * requested_rank, requested_rank + 1),
                )
            )
            rank_growth_receipt = {
                "schema": "v1_trust_walk_rank_growth_honored_v1",
                "obligation": dict(pending),
                "previous_rank": int(requested_rank),
                "granted_rank": int(granted_rank),
                "maximum_rank": maximum_rank,
                "at_maximum_rank": bool(at_maximum),
                "honored_at_chart_index": int(chart_index),
            }
            requested_rank = granted_rank
            state["requested_rank"] = granted_rank
            state["pending_obligation"] = None
        result, acquisition_receipt = _acquire_trust_walk_chart_field(
            reachable_budget=remaining_budget,
            resolution_floor=effective_resolution_floor,
            resolution_floor_source=resolution_floor_source,
            partition_random_seed=partition_random_seed,
            runtime=runtime,
            optimization_axis=optimization_axis,
            query=query,
            committed_mask=committed_mask,
            committed_flat_retained=committed_flat_retained,
            route_ids=route_ids,
            target_class_id=target_class_id,
            decision_provider=decision_provider,
            run_id=run_id,
            random_seed=random_seed,
            sample_size_policy=sample_size_policy,
            partition_family=partition_family,
            ordering_key=ordering_key,
            walk_policy=walk_policy,
            fd_sketch_rank=requested_rank,
            fd_maximum_rank=int(base_policy.fd_maximum_rank),
            cache_dir=resolved_cache_dir,
            chart_index=int(chart_index),
            progress_callback=progress_callback,
        )
        obligation = dict(result.receipt.get("fd", {})).get("rank_growth_obligation")
        if obligation is not None:
            state["pending_obligation"] = dict(obligation)

        labels = np.asarray(result.partition_labels, dtype=np.int8).reshape(-1)
        if labels.size != len(route_ids):
            raise ValueError("trust-walk chart labels do not cover the route axis")
        key_values = np.asarray(result.key_values, dtype=np.float64)
        orders_by_label: dict[int, np.ndarray] = {}
        for label in (0, 1):
            members = np.flatnonzero(labels == label)
            orders_by_label[label] = members[
                np.argsort(key_values[members], kind="stable")
            ]
        sign_values = np.where(labels == 1, -1.0, 1.0)
        partitioning = canonical_global_two_sign_partitioning(
            route_axis_hash=route_axis_hash,
            layer_ids=route_layer_ids,
            sign_values=sign_values,
            partition_family=partition_family,
            trust=CanonicalTwoPartitionTrustPolicy(
                maximum_covariance_leverage=float(_V1_TRUST_WALK_CHART_LEVERAGE_CAP)
            ),
            source_geometry_hash=str(result.snapshot.fd_state.content_hash),
        )
        names_by_label = {
            0: CANONICAL_TRUST_WALK_PARTITION_IDS[0],
            1: CANONICAL_TRUST_WALK_PARTITION_IDS[1],
        }
        first_label = 1 if str(initial_partition) == "negative" else 0
        partition_order = {
            names_by_label[first_label]: orders_by_label[first_label],
            names_by_label[1 - first_label]: orders_by_label[1 - first_label],
        }
        # Uniform-label charts (single_population, or a sign
        # family that measured one empty class) present ONLY nonempty
        # partitions; the walk's single-queue mode consumes them.
        partition_order = {
            name: order
            for name, order in partition_order.items()
            if int(np.asarray(order).size) > 0
        }
        if not partition_order:
            raise ValueError("trust-walk chart has no nonempty partition")
        field_receipt: dict[str, Any] = {
            "schema": "v1_trust_walk_chart_field_receipt_v1",
            "acquisition": dict(acquisition_receipt),
            "scoped_field": dict(result.receipt),
            "initial_turn_partition": names_by_label[first_label],
        }
        if rank_growth_receipt is not None:
            field_receipt["rank_growth_honored"] = rank_growth_receipt
        return trust_walk.TrustWalkChart(
            chart_index=int(chart_index),
            anchor_mask_hash=committed_mask.mask_hash,
            partitioning=partitioning,
            partition_order=partition_order,
            key_id=str(ordering_key),
            key_values=result.key_values,
            key_standard_errors=result.key_standard_errors,
            global_mean=result.snapshot.global_mean,
            trust_state=result.snapshot.fd_state,
            rademacher_global_factor=(result.snapshot.rademacher_global_factor),
            horizon={
                str(key): int(value)
                for key, value in dict(result.horizon).items()
                if str(key) in partition_order
            },
            field_receipt=field_receipt,
            measurement_class_id=scoped_field.SCOPED_FIELD_CLASS_ID,
        )

    return _chart_builder


def _trust_walk_result_namespace(
    outcome: Mapping[str, Any],
) -> SimpleNamespace:
    """Adapt the walk outcome exactly as the ladder outcome is adapted."""

    return SimpleNamespace(
        committed_candidate_ids=tuple(
            str(value) for value in outcome["accepted_candidate_mask_hashes"]
        ),
        epoch_results=(),
    )


def _mount_v1_trust_walk_receipts(
    workflow_details: dict[str, Any],
    *,
    outcome_receipt: Mapping[str, Any],
    ordering_key: str,
) -> None:
    """Mount trust-walk identities (on-path only; off-path never calls this).

    The caller writes ``replaced_execution_identity`` BEFORE this mount,
    so the replaced-identity chain is preserved exactly as on the ladder
    path.  The ordering-key block is written UNCONDITIONALLY here: the
    registered key acts inside every stepper arm (chart orders and
    tranche boundaries), incumbent included — unlike the ladder path,
    where the incumbent key leaves the order source byte-identical and
    the block therefore stays conditional.
    """

    workflow_details["execution_identity"] = trust_walk.TRUST_WALK_ALGORITHM_VERSION
    workflow_details["v1_trust_walk"] = outcome_receipt
    workflow_details["v1_ordering_key"] = {
        "key_id": str(ordering_key),
        "registry_version": ORDERING_KEY_REGISTRY_VERSION,
        "removal_cost_convention": "c_r_equals_1",
        "order_source": ("registered_key_statistic_over_chart_scoped_field"),
        "c2_weighting_field": "incumbent_activity_envelope",
        "c3_status": ("unvalidated_no_measured_rank_association"),
    }


def _v1_walk_scoring_context_packet_size(
    chart_receipts: Sequence[Mapping[str, Any]],
) -> int | str:
    """Read the walk's scoring packet size from its own panel receipts.

    The chart-field acquisition delivers panels through
    ``OptimizedScoringResponseProvider.accumulate_device_tripartite``,
    whose per-panel receipt records ``context_packet_size`` — the
    bounded-materialization packet of the scoring provider the walk
    actually used.  If no delivered panel recorded one (e.g. a fully
    cache-served walk whose stored acquisition receipts predate the
    field), the value is the session retirement disclosure — never a
    fabricated number.
    """

    for chart_receipt in chart_receipts:
        field_receipt = dict(chart_receipt).get("field_receipt", {})
        acquisition = dict(field_receipt).get("acquisition", {})
        for look_receipt in dict(acquisition).get("panel_deliveries", ()):
            delivered = dict(look_receipt).get("delivered_panels", {})
            for panel_receipt in dict(delivered).values():
                packet_size = dict(panel_receipt).get("context_packet_size")
                if packet_size is not None:
                    return int(packet_size)
    return _V1_SESSION_RETIREMENT_DISCLOSURE


def _start_inductor_torch_key_prefetch(device: Any) -> dict[str, Any] | None:
    """Start ``torch._inductor.codecache.torch_key()`` on a helper thread.

    CUDA runs only (the prewarm compiles only there);
    ``IRG_TORCH_KEY_PREFETCH=0`` disables it.  The module is imported on this
    thread first, so the helper runs no import.  The key is Inductor's own
    in-process cache; a compile that asks before the helper finishes simply
    computes it itself, as before.
    """

    if os.environ.get("IRG_TORCH_KEY_PREFETCH", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return None
    try:
        import torch as _torch

        if _torch.device(device).type != "cuda":
            return None
        from torch._inductor import codecache as _codecache

        torch_key = _codecache.torch_key
    except Exception:
        return None
    import threading

    state: dict[str, Any] = {"started": time.perf_counter()}

    def run() -> None:
        try:
            torch_key()
            state["seconds"] = time.perf_counter() - state["started"]
        except Exception as error:  # the compile path recomputes it
            state["error"] = f"{type(error).__name__}"

    thread = threading.Thread(target=run, name="inductor-torch-key", daemon=True)
    state["thread"] = thread
    thread.start()
    return state


def _join_inductor_torch_key_prefetch(
    state: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if state is None:
        return None
    waited = time.perf_counter()
    state["thread"].join()
    receipt = {
        "schema": "irg_inductor_torch_key_prefetch_v1",
        "helper_seconds": (
            float(state["seconds"]) if "seconds" in state else None
        ),
        "join_wait_seconds": float(time.perf_counter() - waited),
        "error": state.get("error"),
    }
    return receipt




# Set on an evaluator whose every layer's prefix/suffix split was prewarmed
# (the shared-process worker retains the evaluator and its backend caches).
_LAYER_SPLITS_PREWARMED_ATTRIBUTE = "_v63_layer_splits_prewarmed"


def _preflight_registered_resident_images(
    *, runtime: Any, optimization_axis: Any, target_class_id: int,
    maximum_per_stratum: int, category_enabled: bool,
) -> dict[str, int]:
    """Admit the combined train and validation image sets before engine work.

    Training selection is the same sealed category prefix used by the frozen
    bank. Validation admission uses only its bank row count, before any
    validation label or response enters terminal evaluation.
    """
    if runtime.evaluator.device.type != "cuda":
        return {}
    from mechanistic_systems.measurement.native_panel_data_plane import preflight_resident_image_sets

    training_plan = {}
    if category_enabled:
        populations, _weights = _sealed_terminal_populations(
            optimization_axis, target_class_id=int(target_class_id),
            namespace=("architecture-dense-category-calibration:"
                       f"{runtime.loaded.receipt.state_dict_hash}:"
                       f"{optimization_axis.content_hash}"),
            maximum_per_stratum=int(maximum_per_stratum),
        )
        training_plan["category_calibration"] = tuple(
            context_id for population in populations.values() for context_id in population)
    evaluation_plane = runtime.evaluation_data_plane
    bank_inputs = evaluation_plane._bank_inputs
    return preflight_resident_image_sets((
        (runtime.optimization_data_plane, training_plan),
        (evaluation_plane, {"validation": 0 if bank_inputs is None else len(bank_inputs)}),
    ))


def _prewarm_registered_mask_runners(
    *,
    runtime: Any,
    dense_mask: HardMask,
    target_class_id: int,
    phase_timing: _PhaseTiming | None = None,
    oracle_provider: Any = None,
) -> Mapping[str, Any]:
    """Compile fixed single and every-layer paired runners at worker setup.

    The concrete gate values are runtime inputs to structural runners, so one
    dense K=1 call and one dense/one-route-removed K=2 call populate the same
    caches used by every later pruning mask.  The rows are training contexts
    used only for compilation; they update no calibration or oracle statistic.
    """

    k1_batch_rows, k2_batch_rows = _registered_runner_prewarm_shapes(runtime)
    if oracle_provider is not None:
        # F's first-look support plus its unchanged occupancy quantum. Warm
        # K's admitted physical shape without drawing a statistical panel.
        requested_rows = sum(
            min(len(population), int(oracle_provider.oracle_minimum_per_stratum))
            for population in oracle_provider._oracle_statistical_populations.values()
        ) + k2_batch_rows
        quantum = int(oracle_provider._logical_alignment_quantum(
            role="proposal_adjudication", lane_count=2,
            registered_quantum=k2_batch_rows,
        ))
        requested_rows = ((requested_rows + quantum - 1) // quantum) * quantum
        k2_batch_rows = oracle_provider._look_execution_batch_rows(
            requested_rows=requested_rows, configured_rows=k2_batch_rows,
            lane_count=2, role="proposal_adjudication",
        )
    if runtime.evaluator.device.type == "cuda":
        profile = runtime.evaluator.runtime_execution_profile
        single_rows = max(k1_batch_rows, int(profile.v1_candidate_oracle_batch_rows))
        pair_rows = k2_batch_rows
        if calibration_folding_enabled():
            single_rows = max(single_rows, fused_calibration_oracle_batch_rows(profile),
                              fused_calibration_oracle_batch_rows(profile, paired=True))
            pair_rows = max(pair_rows, fused_calibration_oracle_batch_rows(profile, paired=True))
        if ((runtime.evaluator.walk_graph_batch_capacity(1) or 0) >= single_rows
                and (runtime.evaluator.walk_graph_batch_capacity(2) or 0) >= pair_rows):
            return {"schema": "registered_mask_worker_graph_prewarm_v2", "status": "reused",
                    "k1_batch_rows": single_rows, "k2_batch_rows": pair_rows,
                    "total_seconds": 0.0, "cold_setup_only": True,
                    "statistical_rows_updated": 0,
                    "backend_cache_after": dict(runtime.evaluator.candidate_execution_backend.runtime_profile().metadata)}
        contexts = tuple(str(record.context_id) for record in
                         runtime.optimization_data_plane.axis.records[:max(single_rows, pair_rows)])
        if len(set(contexts)) != len(contexts):
            raise ValueError("runner prewarm requires unique training contexts")
        if len(contexts) < max(single_rows, pair_rows):
            raise ValueError("worker graph preparation requires its configured physical rows")
        backend = runtime.evaluator.candidate_execution_backend
        before = backend.runtime_profile()
        started = time.perf_counter()
        if phase_timing is not None:
            phase_timing.mark("compile_trace_start")
            phase_timing.mark("prewarm_first_batch_start")
        shape_receipts = []
        for role, rows, single, pairs in (
            ("calibration", single_rows, True, False),
            ("v0_paired_oracle", pair_rows, False, True),
        ):
            shape_started = time.perf_counter()
            for batch in runtime.optimization_data_plane.panel_batches(
                    contexts[:rows], role=role, batch_rows=rows):
                batch.validate()
                batch.wait_until_ready(runtime.evaluator.device)
                shape_receipts.append(runtime.evaluator.prewarm_walk_graphs(
                    batch.inputs, include_single=single, include_pairs=pairs,
                ))
                batch.mark_consumed(runtime.evaluator.device)
            shape_receipts[-1]["seconds"] = float(time.perf_counter() - shape_started)
            if phase_timing is not None and single:
                phase_timing.mark("prewarm_first_batch_end")
        if phase_timing is not None:
            phase_timing.mark("compile_trace_end")
        after = backend.runtime_profile()
        receipt = {
            "schema": "registered_mask_worker_graph_prewarm_v2",
            "k1_batch_rows": single_rows, "k2_batch_rows": pair_rows,
            "all_layer_graphs": tuple(shape_receipts),
            "total_seconds": float(time.perf_counter() - started),
            "backend_cache_before": dict(before.metadata),
            "backend_cache_after": dict(after.metadata),
            "cold_setup_only": True, "statistical_rows_updated": 0,
            "physical_tails_padded": True, "shared_graph_memory_pool": True,
        }
        receipt["content_hash"] = runtime_receipt_hash(receipt, prefix="irg-runner-prewarm-")
        return receipt
    extra_shapes: list[tuple[str, int, int]] = []
    if calibration_folding_enabled():
        profile = runtime.evaluator.runtime_execution_profile
        fused_rows = fused_calibration_oracle_batch_rows(profile, paired=True)
        candidate_only_rows = fused_calibration_oracle_batch_rows(profile)
        # Cover the paired union, the candidate's union after
        # lane-specific acquisition, and the parent's shorter oracle lane.
        seen_shapes = {(k1_batch_rows, 1), (k2_batch_rows, 2)}
        for role, rows, lanes in (
            ("calibration_oracle_fused", fused_rows, 2),
            ("calibration_oracle_fused", fused_rows, 1),
            ("calibration_oracle_fused", candidate_only_rows, 1),
            ("v0_paired_oracle", k2_batch_rows, 1),
        ):
            if (rows, lanes) not in seen_shapes:
                extra_shapes.append((role, rows, lanes))
                seen_shapes.add((rows, lanes))
    # Retain the original support check even when redundant warmups are
    # omitted; a short axis must take the same skipped-prewarm path.
    required_contexts = max(k1_batch_rows, k2_batch_rows, *(rows for _, rows, _ in extra_shapes))
    context_ids = tuple(
        str(record.context_id)
        for record in runtime.optimization_data_plane.axis.records[:required_contexts]
    )
    if len(set(context_ids)) != len(context_ids):
        raise ValueError("runner prewarm requires unique training contexts")
    if len(context_ids) < required_contexts:
        return {
            "schema": "registered_resnet18_mask_runner_prewarm_v1",
            "status": "skipped_insufficient_exploratory_support",
            "available_contexts": len(context_ids),
            "required_contexts": int(required_contexts),
            "k1_batch_rows": int(k1_batch_rows),
            "k2_batch_rows": int(k2_batch_rows),
            "cold_setup_only": True,
            "statistical_rows_updated": 0,
        }

    def masks_by_layer(mask: HardMask) -> dict[str, np.ndarray]:
        # Declared-axis expansion, mirroring the providers' own
        # ``masks_by_layer`` over ``evaluator.route_blocks``; the authority
        # validates coverage of the route axis and raises on any mismatch.
        return _per_layer_fields_from_declared_flat(
            mask.to_bool_array(),
            runtime=runtime,
        )

    synthetic_values = dense_mask.to_bool_array()
    synthetic_values[-1] = False
    synthetic_mask = HardMask.from_bool_array(
        synthetic_values,
        route_axis_hash=dense_mask.route_axis_hash,
    )
    dense_compiled = runtime.evaluator.compile_mask(masks_by_layer(dense_mask))
    synthetic_compiled = runtime.evaluator.compile_mask(masks_by_layer(synthetic_mask))
    projection = RuntimeReadoutProjectionSpec.coordinate(
        int(target_class_id),
        retain_host_values=False,
    )
    backend = runtime.evaluator.candidate_execution_backend
    before = backend.runtime_profile()
    started = time.perf_counter()
    if phase_timing is not None:
        phase_timing.mark("compile_trace_start")
        phase_timing.mark("prewarm_first_batch_start")
    runtime.evaluator.readout_responses_from_batches(
        runtime.optimization_data_plane.panel_batches(
            context_ids[:k1_batch_rows],
            role="calibration",
            batch_rows=k1_batch_rows,
        ),
        compiled_masks=(dense_compiled,),
        lane_keys=("prewarm:k1:dense",),
        readout_projection=projection,
    )
    if phase_timing is not None:
        phase_timing.mark("prewarm_first_batch_end")
    k1_completed = time.perf_counter()
    runtime.evaluator.readout_responses_from_batches(
        runtime.optimization_data_plane.panel_batches(
            context_ids[:k2_batch_rows],
            role="v0_paired_oracle",
            batch_rows=k2_batch_rows,
        ),
        compiled_masks=(dense_compiled, synthetic_compiled),
        lane_keys=("prewarm:k2:dense", "prewarm:k2:synthetic"),
        readout_projection=projection,
    )
    k2_completed = time.perf_counter()
    additional_receipts = []
    for role, rows, lanes in extra_shapes:
        shape_started = time.perf_counter()
        runtime.evaluator.readout_responses_from_batches(
            runtime.optimization_data_plane.panel_batches(
                context_ids[:rows], role=role, batch_rows=rows,
            ),
            compiled_masks=(dense_compiled, synthetic_compiled)[:lanes],
            lane_keys=tuple(f"prewarm:{role}:{rows}:lane:{lane}" for lane in range(lanes)),
            readout_projection=projection,
        )
        additional_receipts.append({
            "role": role, "batch_rows": rows, "lane_count": lanes,
            "seconds": float(time.perf_counter() - shape_started),
        })
    # Build and compile every walked layer's prefix/suffix graph pair
    # now, once per process (the split and its compiled prefix are cached by
    # the backend, and the shared-process worker carries both across arms),
    # instead of on each layer's first decision. One K=2 forward per layer
    # with one route of that layer removed forks the graph at that layer.
    layer_split_receipts: list[dict[str, Any]] = []
    layer_split_shapes = getattr(runtime.evaluator, _LAYER_SPLITS_PREWARMED_ATTRIBUTE, set())
    if getattr(backend, "forward_prefix_reuse", False) and k2_batch_rows not in layer_split_shapes:
        offsets = _declared_route_axis_offsets(runtime, flat_length=dense_mask.route_count)
        for group in runtime.route_groups:
            layer_started = time.perf_counter()
            values = dense_mask.to_bool_array()
            values[offsets[str(group.layer_id)][0]] = False
            layer_compiled = runtime.evaluator.compile_mask(
                masks_by_layer(
                    HardMask.from_bool_array(values, route_axis_hash=dense_mask.route_axis_hash)
                )
            )
            runtime.evaluator.readout_responses_from_batches(
                runtime.optimization_data_plane.panel_batches(
                    context_ids[:k2_batch_rows],
                    role="v0_paired_oracle",
                    batch_rows=k2_batch_rows,
                ),
                compiled_masks=(dense_compiled, layer_compiled),
                lane_keys=(
                    f"prewarm:layer:{group.layer_id}:dense",
                    f"prewarm:layer:{group.layer_id}:removed",
                ),
                readout_projection=projection,
            )
            layer_split_receipts.append(
                {
                    "layer_id": str(group.layer_id),
                    "seconds": float(time.perf_counter() - layer_started),
                }
            )
        layer_split_shapes.add(k2_batch_rows)
        setattr(runtime.evaluator, _LAYER_SPLITS_PREWARMED_ATTRIBUTE, layer_split_shapes)
    if runtime.evaluator.device.type == "cuda":
        import torch

        torch.cuda.synchronize(runtime.evaluator.device)
    if phase_timing is not None:
        phase_timing.mark("compile_trace_end")
    completed = time.perf_counter()
    after = backend.runtime_profile()
    receipt = {
        "schema": "registered_resnet18_mask_runner_prewarm_v1",
        "k1_batch_rows": int(k1_batch_rows),
        "k2_batch_rows": int(k2_batch_rows),
        "k1_seconds": float(k1_completed - started),
        "k2_seconds": float(k2_completed - k1_completed),
        "additional_shapes": tuple(additional_receipts),
        "total_seconds": float(completed - started),
        "dense_mask_hash": dense_mask.mask_hash,
        "synthetic_mask_hash": synthetic_mask.mask_hash,
        "target_class_id": int(target_class_id),
        "backend_cache_before": dict(before.metadata),
        "backend_cache_after": dict(after.metadata),
        "cold_setup_only": True,
        "statistical_rows_updated": 0,
    }
    if layer_split_receipts:
        receipt["layer_split_prewarm"] = tuple(layer_split_receipts)
    receipt["content_hash"] = runtime_receipt_hash(
        receipt,
        prefix="irg-runner-prewarm-",
    )
    return receipt


def _registered_runner_prewarm_shapes(runtime: Any) -> tuple[int, int]:
    """Return the exact role-bound K=1/K=2 shapes used by the live unit."""

    execution_profile = runtime.evaluator.runtime_execution_profile
    k1_batch_rows = int(execution_profile.calibration_batch_rows)
    k2_batch_rows = int(execution_profile.v0_paired_oracle_batch_rows)
    if k1_batch_rows <= 0 or k2_batch_rows <= 0:
        raise ValueError("runner prewarm batch rows must be positive")
    return k1_batch_rows, k2_batch_rows


def _bce_run_proposal_count(runs: Sequence[Any]) -> int:
    """Count canonical BCE decisions without consulting legacy grid steps."""

    count = 0
    for run in runs:
        decisions = getattr(getattr(run, "result", None), "decisions", None)
        if decisions is None:
            raise TypeError("BCE pilot run lacks its canonical decisions")
        count += len(decisions)
    return count


def _finite_receipt_statistic(value: float) -> tuple[float | None, str]:
    """Encode an expected open/undefined statistic without non-finite JSON.

    Sequential no-commit decisions legitimately carry open confidence bounds
    before a usable oracle look.  Content-addressed receipts prohibit NaN and
    infinities, so the scalar and its mathematical state are represented
    separately instead of weakening canonical hashing.
    """

    scalar = float(value)
    if math.isfinite(scalar):
        return scalar, "finite"
    if math.isnan(scalar):
        return None, "undefined"
    return None, ("positive_infinity" if scalar > 0.0 else "negative_infinity")


def _nonfinite_receipt_paths(
    value: Any,
    *,
    path: str = "$",
) -> tuple[str, ...]:
    """Return precise paths for accidental non-finite receipt values."""

    if isinstance(value, Mapping):
        return tuple(
            nested
            for key, item in value.items()
            for nested in _nonfinite_receipt_paths(
                item,
                path=f"{path}.{key}",
            )
        )
    if isinstance(value, (tuple, list)):
        return tuple(
            nested
            for index, item in enumerate(value)
            for nested in _nonfinite_receipt_paths(
                item,
                path=f"{path}[{index}]",
            )
        )
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return (path,)
    return ()


def _v0_performance_gate_receipt(
    *,
    pilot_receipt_hash: str,
    architecture_family: str = "resnet18",
    device_name: str,
    scoring_and_anchor_seconds: float,
    traversal_seconds: float,
    receipt_finalization_seconds: float,
    traversal_complete: bool,
    optimized_panel_eligible: bool,
    runtime: Any = None,
    resolved_settings: Mapping[str, str] | None = None,
    data_plane_settings: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Bind the three pre-audit H100 phase budgets to one pilot receipt."""

    scoring = float(scoring_and_anchor_seconds)
    traversal = float(traversal_seconds)
    finalization = float(receipt_finalization_seconds)
    if (
        not str(pilot_receipt_hash)
        or not str(device_name)
        or not all(
            math.isfinite(value) and value >= 0.0
            for value in (scoring, traversal, finalization)
        )
    ):
        raise ValueError("v0 performance gate inputs are invalid")
    total = scoring + traversal + finalization
    h100 = "H100" in str(device_name).upper()
    family = str(architecture_family).lower()
    # Preserve the established receipt bytes for already-admitted families.
    # New architecture families cannot inherit those H100 budgets.
    architecture_budget_validated = family in {
        "resnet18",
        "alexnet",
        "alexnet_conv_kernel_weight",
    }
    phase_passes = {
        "h100_hardware": h100,
        "optimized_panel_data_plane": bool(optimized_panel_eligible),
        "scoring_and_anchor": scoring <= 90.0,
        "complete_traversal": bool(traversal_complete) and traversal <= 450.0,
        "receipt_finalization": finalization <= 30.0,
        "total_under_600": total < 600.0,
    }
    if not architecture_budget_validated:
        phase_passes["architecture_specific_budget_validated"] = False
    gate_passed = all(phase_passes.values())
    receipt: dict[str, Any] = {
        "effective_settings": dict(resolved_settings) if resolved_settings is not None else _run_effective_settings(runtime),
        "data_plane_effective_settings": dict(data_plane_settings) if data_plane_settings is not None else _runtime_data_plane_effective_settings(runtime),
        "schema": (
            "resnet18_h100_v0_performance_gate_v1"
            if architecture_budget_validated
            else "registered_architecture_h100_v0_performance_gate_v1"
        ),
        "pilot_receipt_hash": str(pilot_receipt_hash),
        "scheduler_time_excluded": True,
        "terminal_validation_audit_excluded": True,
        "gate_receipt_write_excluded_from_own_timing": True,
        "device_name": str(device_name),
        "scoring_and_anchor_seconds": scoring,
        "scoring_and_anchor_budget_seconds": 90.0,
        "complete_traversal_seconds": traversal,
        "complete_traversal_budget_seconds": 450.0,
        "receipt_finalization_seconds": finalization,
        "receipt_finalization_budget_seconds": 30.0,
        "total_excluding_terminal_audit_seconds": total,
        "total_budget_seconds": 600.0,
        "traversal_complete": bool(traversal_complete),
        "optimized_panel_eligible": bool(optimized_panel_eligible),
        "phase_passes": phase_passes,
        "performance_gate_passed": gate_passed,
        "status": "passed" if gate_passed else "failed",
    }
    if not architecture_budget_validated:
        receipt["architecture_family"] = family
        receipt["architecture_specific_budget_validated"] = False
        receipt["status"] = "architecture_specific_budget_unvalidated"
    receipt["content_hash"] = runtime_receipt_hash(
        receipt,
        prefix=(
            "irg-resnet18-v0-performance-gate-"
            if architecture_budget_validated
            else "irg-registered-architecture-v0-performance-gate-"
        ),
    )
    return receipt


def _residual_observation_path_counts_block(provider: Any) -> dict[str, Any] | None:
    """Report the device and host arithmetic used for residual observables.

    These execution counts do not enter the statistical decision or its
    identity. No block is emitted when this architecture has no residual gate.
    """

    counts = getattr(provider, "residual_observation_path_counts", None)
    if not isinstance(counts, Mapping) or not counts:
        return None
    resolved = {str(path): int(count) for path, count in sorted(counts.items())}
    return {
        "schema": "irg_residual_observation_path_counts_v1",
        "execution_only": True,
        "counts": resolved,
        "device_fallback_to_host": int(resolved.get("device_fallback_to_host", 0)),
    }


def _unified_native_execution_receipt(
    *,
    version: str,
    optimization_data_plane: Mapping[str, Any],
    evaluation_data_plane: Mapping[str, Any],
    provider_resource_usage: Mapping[str, Any],
    gpu_activity: Mapping[str, Any],
    runtime_execution_profile: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Normalize the execution-plane facts required by the runtime contract."""

    optimization_roles = dict(optimization_data_plane.get("roles", {}))
    totals: dict[str, float] = {}
    for field in (
        "calls",
        "requested_physical_contexts",
        "delivered_physical_contexts",
        "input_cache_hits",
        "input_cache_misses",
        "bank_read_bytes",
        "host_to_device_bytes",
        "producer_wait_seconds",
        "panel_wall_seconds",
        "first_batch_latency_seconds",
        "batch_count",
    ):
        totals[field] = sum(
            float(dict(values).get(field, 0.0))
            for values in optimization_roles.values()
        )
    panel_wall = totals["panel_wall_seconds"]
    receipt: dict[str, Any] = {
        "schema": "irg_unified_native_execution_receipt_v1",
        "version": str(version),
        "data_source": {
            "optimized_bank": bool(
                optimization_data_plane.get("optimized_bank", False)
            ),
            "source_id": optimization_data_plane.get("source_id"),
            "source_receipt_hash": optimization_data_plane.get("source_receipt_hash"),
            "node_local_staged": bool(
                optimization_data_plane.get("node_local_staged", False)
            ),
            "staging_seconds": float(
                optimization_data_plane.get("bank_staging_seconds", 0.0)
            ),
            "staging_amortization_scope": optimization_data_plane.get(
                "bank_staging_amortization_scope"
            ),
        },
        "logical_and_physical_work": {
            "logical_draws": int(
                provider_resource_usage.get("statistical_draw_count", 0)
            ),
            "physical_context_state_evaluations": int(
                provider_resource_usage.get("physical_context_state_evaluations", 0)
            ),
            "source_context_state_cache_hits": int(
                provider_resource_usage.get("source_context_state_cache_hits", 0)
            ),
            "requested_physical_contexts": int(totals["requested_physical_contexts"]),
            "delivered_physical_contexts": int(totals["delivered_physical_contexts"]),
        },
        "data_plane_totals": totals,
        "producer_wait_fraction": (
            0.0 if panel_wall <= 0.0 else totals["producer_wait_seconds"] / panel_wall
        ),
        "role_specific_execution": optimization_roles,
        "runtime_execution_profile": dict(runtime_execution_profile),
        "numerics_record": numerics_record(),
        "device_input_cache": dict(optimization_data_plane.get("device_cache", {})),
        "provider": dict(provider_resource_usage),
        "gpu_activity": dict(gpu_activity),
        "evaluation_data_plane_hash": evaluation_data_plane.get("content_hash"),
        "scientific_outcomes_inspected_by_runtime_policy": False,
    }
    receipt["content_hash"] = runtime_receipt_hash(
        receipt,
        prefix="irg-unified-native-execution-",
    )
    return receipt


def _device_name(device: Any) -> str:
    """Return the physical CUDA model name when available."""

    try:
        import torch

        resolved = torch.device(device)
        if resolved.type == "cuda" and torch.cuda.is_available():
            return str(torch.cuda.get_device_name(resolved))
    except (RuntimeError, TypeError, ValueError):
        pass
    return str(device)


def _load_historical_hard_mask(
    path: str | Path,
    *,
    runtime: Any,
    route_axis_hash: str,
    prepared_payload: Mapping[str, Any] | None = None,
    prepared_sha256: str | None = None,
) -> tuple[HardMask, str, Mapping[str, Any]]:
    """Load a historical mask through the runtime's explicit route aliases."""

    import torch

    source = Path(path)
    payload = (torch.load(source, map_location="cpu", weights_only=True)
               if prepared_payload is None else prepared_payload)
    if not isinstance(payload, Mapping):
        raise TypeError("historical mask artifact must be a layer mapping")
    alias_to_canonical: dict[str, str] = {}
    for group in runtime.route_groups:
        canonical = str(group.layer_id)
        aliases = {
            canonical,
            str(getattr(group, "profile_group_name", "") or ""),
        }
        for alias in aliases - {""}:
            previous = alias_to_canonical.setdefault(alias, canonical)
            if previous != canonical:
                raise ValueError(f"historical route alias {alias!r} is not one-to-one")
    source_container_kind = "direct_complete_mask"
    selected_snapshot_key: str | None = None
    if payload and all(isinstance(value, Mapping) for value in payload.values()):
        complete_snapshots: list[tuple[str, Mapping[str, Any]]] = []
        expected_aliases = {
            str(group.profile_group_name) for group in runtime.route_groups
        }
        for snapshot_key, snapshot in payload.items():
            if set(map(str, snapshot)) == expected_aliases:
                complete_snapshots.append((str(snapshot_key), snapshot))
        if len(complete_snapshots) != 1:
            raise ValueError(
                "historical cumulative mask container must have exactly one "
                "complete terminal route-axis snapshot"
            )
        selected_snapshot_key, payload = complete_snapshots[0]
        source_container_kind = "cumulative_layerwise_snapshots"
    normalized: dict[str, Any] = {}
    source_to_canonical: dict[str, str] = {}
    unknown: list[str] = []
    for raw_key, value in payload.items():
        source_key = str(raw_key)
        canonical = alias_to_canonical.get(source_key)
        if canonical is None:
            unknown.append(source_key)
            continue
        if canonical in normalized:
            raise ValueError(
                f"historical mask repeats canonical route group {canonical!r}"
            )
        normalized[canonical] = value
        source_to_canonical[source_key] = canonical
    expected = {group.layer_id for group in runtime.route_groups}
    if set(normalized) != expected or unknown:
        missing = sorted(expected - set(normalized))
        raise ValueError(
            "historical mask route groups differ from the registered axis: "
            f"missing={missing}, extra={sorted(unknown)}"
        )
    masks: dict[str, np.ndarray] = {}
    for group in runtime.route_groups:
        raw = normalized[group.layer_id]
        if hasattr(raw, "detach"):
            raw = raw.detach().cpu().numpy()
        value = np.asarray(raw)
        if value.shape != group.shape or not np.isin(value, (0, 1)).all():
            raise ValueError(
                f"historical mask block {group.layer_id!r} has an invalid shape/value"
            )
        masks[group.layer_id] = np.asarray(value, dtype=bool)
    alias_receipt = {
        "schema_name": "irg_historical_mask_route_alias_receipt_v1",
        "source_to_canonical_layer_ids": dict(sorted(source_to_canonical.items())),
        # The disclosure names the axis of the produced flat mask: the
        # DECLARED route-block order (the ``_hard_mask_from_layers``
        # assembly axis), not the profile's walk-order group axis.
        "canonical_route_axis": tuple(
            _declared_route_axis_offsets(
                runtime,
                flat_length=sum(
                    int(np.prod(group.shape, dtype=np.int64))
                    for group in runtime.route_groups
                ),
            )
        ),
        "route_axis_hash": str(route_axis_hash),
        "source_container_kind": source_container_kind,
        "selected_snapshot_key": selected_snapshot_key,
    }
    alias_receipt["content_hash"] = content_hash(
        alias_receipt, prefix="irg-historical-mask-alias-"
    )
    return (
        _hard_mask_from_layers(
            masks,
            runtime=runtime,
            route_axis_hash=route_axis_hash,
        ),
        sha256_file(source) if prepared_sha256 is None else prepared_sha256,
        alias_receipt,
    )


# The sealed populations are a pure function of (axis
# content, target, namespace, cap).  Step setup and the terminal fit compute
# the category-calibration populations with identical arguments; at ImageNet
# scale each call orders 1.28M training records by a per-row content hash.
# Memoized only for a real ``ContextAxis`` (frozen; ``content_hash`` is its
# cached content address); every call returns fresh dicts whose values are
# immutable tuples, so no caller can alias or mutate a cached result.  The
# decorated function below is unchanged; ``__wrapped__`` is the uncached form.
_SEALED_TERMINAL_POPULATIONS_MEMO: dict[
    tuple[str, int, str, int],
    tuple[dict[str, tuple[str, ...]], dict[str, float]],
] = {}
_SEALED_TERMINAL_POPULATIONS_MEMO_ENTRIES = 4
# Guards the memo's lookup, eviction and insertion (gate fix).  The pure
# computation runs outside the lock; two concurrent misses compute the same
# value and the second insert replaces the first with equal content.
import threading as _populations_memo_threading  # noqa: E402

_SEALED_TERMINAL_POPULATIONS_MEMO_LOCK = _populations_memo_threading.Lock()


def _content_memoized_populations(function: Callable[..., Any]) -> Callable[..., Any]:
    import functools

    @functools.wraps(function)
    def memoized(
        axis: Any,
        *,
        target_class_id: int,
        namespace: str,
        maximum_per_stratum: int,
    ) -> tuple[dict[str, tuple[str, ...]], dict[str, float]]:
        from mechanistic_systems.measurement.contexts import ContextAxis

        key = None
        if isinstance(axis, ContextAxis):
            key = (
                str(axis.content_hash),
                int(target_class_id),
                str(namespace),
                int(maximum_per_stratum),
            )
            if loss_weighting() == "per_image":
                key += ("per_image",)
            with _SEALED_TERMINAL_POPULATIONS_MEMO_LOCK:
                cached = _SEALED_TERMINAL_POPULATIONS_MEMO.get(key)
            if cached is not None:
                return dict(cached[0]), dict(cached[1])
        populations, weights = function(
            axis,
            target_class_id=target_class_id,
            namespace=namespace,
            maximum_per_stratum=maximum_per_stratum,
        )
        if key is not None:
            with _SEALED_TERMINAL_POPULATIONS_MEMO_LOCK:
                memo = _SEALED_TERMINAL_POPULATIONS_MEMO
                memo.pop(key, None)
                while len(memo) >= _SEALED_TERMINAL_POPULATIONS_MEMO_ENTRIES:
                    memo.pop(next(iter(memo)))
                memo[key] = (dict(populations), dict(weights))
        return populations, weights

    return memoized


@_content_memoized_populations
def _sealed_terminal_populations(
    axis: Any,
    *,
    target_class_id: int,
    namespace: str,
    maximum_per_stratum: int,
) -> tuple[dict[str, tuple[str, ...]], dict[str, float]]:
    target = int(target_class_id)
    grouped: dict[int, list[str]] = {}
    for record in axis.records:
        grouped.setdefault(int(record.label), []).append(str(record.context_id))
    labels = tuple(sorted(grouped))
    off_target_labels = tuple(label for label in labels if label != target)
    if target not in grouped or not off_target_labels:
        raise ValueError("terminal evaluation requires target and off-target classes")
    populations = {
        "on_target": (target, tuple(grouped[target])),
        **{
            f"off_target:{label}": (label, tuple(grouped[label]))
            for label in off_target_labels
        },
    }
    from mechanistic_systems.measurement.sealed_sort_keys import (
        class_prefix_sort_keys,
        sorted_by_keys,
    )

    result: dict[str, tuple[str, ...]] = {}
    for stratum, (class_id, rows) in populations.items():
        # Selection is a property of the architecture/train class, not of
        # whichever class is the current query.  This makes the frozen dense
        # category bank exactly reusable across units while the OVR weights
        # below remain target-specific.  The per-row key
        # is content_hash({"algorithm": "sha256_class_prefix_v2", "namespace",
        # "class_id", "context_id"}) byte for byte, assembled directly and
        # checked against content_hash on every call.
        ordered = sorted_by_keys(
            rows,
            class_prefix_sort_keys(
                rows, namespace=str(namespace), class_id=int(class_id)
            ),
        )
        result[stratum] = ordered[: min(len(ordered), int(maximum_per_stratum))]
        if len(result[stratum]) < 2:
            raise ValueError(
                "terminal pilot needs at least two contexts per class stratum"
            )
    if loss_weighting() == "per_image":
        return result, _binary_loss_weights({name: rows for name, (_, rows) in populations.items()})
    off_weight = 0.5 / float(len(off_target_labels))
    return result, {
        "on_target": 0.5,
        **{f"off_target:{label}": off_weight for label in off_target_labels},
    }


def _fixed_calibration_bce_provider(
    *,
    runtime: Any,
    optimization_axis: Any,
    target_class_id: int,
    maximum_bce_increment: float | None,
    run_id: str,
    random_seed: int,
    mask_resolver: Any,
    candidate_resolver: Any,
    sample_size_policy: IndependentSampleSizePolicy,
    iid_sampling_design: BinaryIIDSamplingDesign | None = None,
    # Either sampling law; the design carries it and the provider reads it.
    oracle_iid_sampling_design: Any | None = None,
    evidence_roles: BinaryEvidenceRoleDesign | None = None,
    oracle_context_ids_by_stratum: Mapping[str, Sequence[str]] | None = None,
    oracle_batch_factory: Any | None = None,
    oracle_panel_role: str = "v0_paired_oracle",
    progress_callback: Callable[..., None] | None = None,
    calibration_deployment_refit_anchor: str = "selected",
    oracle_minimum_per_stratum: int = 2,
    calibration_fold_ownership: bool = False,
    step_category_matrix: bool = False,
) -> NativeTorchCalibratedBCEOracleProvider:
    """Build the one canonical calibration-fit/oracle role separation."""

    namespace = binary_evidence_role_namespace(
        run_id=run_id,
        random_seed=random_seed,
        context_axis_hash=optimization_axis.content_hash,
    )
    if iid_sampling_design is not None and evidence_roles is not None:
        raise ValueError("BCE provider accepts IID sampling or legacy roles")
    if oracle_iid_sampling_design is not None and str(
        sample_size_policy.oracle_sampling_law
    ) != _oracle_sampling_receipt_strings(oracle_iid_sampling_design)[
        "oracle_sampling_law"
    ]:
        # One law, declared once.  The policy states what the receipts
        # certify and the design states what the provider samples under; a
        # disagreement here is exactly the wiring fault the BCE frontier
        # refuses later, caught before any row is acquired.
        raise ValueError(
            "the declared oracle sampling law "
            f"{sample_size_policy.oracle_sampling_law!r} differs from the "
            "bound class-resolved adjudication design"
        )
    iid_design = (
        compile_binary_iid_sampling_design(
            optimization_axis,
            target_class_id=target_class_id,
            randomization_namespace=namespace,
        )
        if iid_sampling_design is None and evidence_roles is None
        else iid_sampling_design
    )
    roles = evidence_roles
    if iid_design is not None:
        if (
            iid_design.target_class_id != int(target_class_id)
            or iid_design.source_context_axis_hash != optimization_axis.content_hash
            or iid_design.randomization_namespace != namespace
        ):
            raise ValueError(
                "precomputed IID training design differs from the requested run"
            )
        calibration_populations = iid_design.source_context_ids_by_stratum
        default_oracle_populations = iid_design.source_context_ids_by_stratum
        minimum_capacity = max(
            2,
            int(sample_size_policy.calibration_maximum_contexts_without_support_cap)
            // 2,
        )
        maximum_capacity = minimum_capacity
    else:
        assert roles is not None
        if (
            roles.target_class_id != int(target_class_id)
            or roles.source_context_axis_hash != optimization_axis.content_hash
            or roles.randomization_namespace != namespace
        ):
            raise ValueError(
                "precomputed training evidence roles differ from the requested run"
            )
        calibration_populations = roles.calibration_contexts
        default_oracle_populations = roles.oracle_contexts
        capacities = tuple(map(len, calibration_populations.values()))
        minimum_capacity = min(capacities)
        maximum_capacity = max(capacities)
    initial = min(
        minimum_capacity,
        max(2, int(sample_size_policy.calibration_minimum_packet) // 2),
    )
    packet = min(minimum_capacity, max(1, initial))
    provider = NativeTorchCalibratedBCEOracleProvider(
        evaluator=runtime.evaluator,
        target_class_id=target_class_id,
        calibration_context_ids_by_stratum=calibration_populations,
        oracle_context_ids_by_stratum=(
            default_oracle_populations
            if oracle_context_ids_by_stratum is None
            else oracle_context_ids_by_stratum
        ),
        calibration_batch_factory=(
            runtime.optimization_data_plane.role_batch_factory("calibration")
        ),
        calibration_sized_batch_factory=(
            runtime.optimization_data_plane.sized_role_batch_factory("calibration")
        ),
        fused_calibration_oracle_batch_factory=(
            runtime.optimization_data_plane.role_batch_factory(
                "calibration_oracle_fused"
            )
        ),
        fused_calibration_oracle_sized_batch_factory=(
            runtime.optimization_data_plane.sized_role_batch_factory(
                "calibration_oracle_fused"
            )
        ),
        oracle_batch_factory=(
            (runtime.optimization_data_plane.role_batch_factory(oracle_panel_role))
            if oracle_batch_factory is None
            else oracle_batch_factory
        ),
        oracle_sized_batch_factory=(
            runtime.optimization_data_plane.sized_role_batch_factory(oracle_panel_role)
            if oracle_batch_factory is None
            else None
        ),
        candidate_resolver=candidate_resolver,
        mask_resolver=mask_resolver,
        maximum_bce_increment=(
            None if maximum_bce_increment is None else float(maximum_bce_increment)
        ),
        calibration_policy=NativeBCECalibrationPolicy(
            confidence_level=sample_size_policy.confidence_level,
            initial_per_class=initial,
            packet_per_class=packet,
            # The provider intersects this ceiling with each stratum's
            # physical support.  Capping both strata at the rarer population
            # prevented Neyman growth in the abundant stratum after the rare
            # stratum reached its complete census.
            maximum_per_class=maximum_capacity,
            precision_only_iid=bool(sample_size_policy.precision_only_iid),
        ),
        runtime_execution_profile=runtime.evaluator.runtime_execution_profile,
        randomization_namespace=namespace,
        iid_sampling_design=iid_design,
        oracle_iid_sampling_design=oracle_iid_sampling_design,
        context_orders_are_sealed=iid_design is None,
        oracle_execution_role=oracle_panel_role,
        progress_callback=progress_callback,
        summary_only_outcomes=True,
        oracle_minimum_per_stratum=int(oracle_minimum_per_stratum),
        calibration_fold_ownership=bool(calibration_fold_ownership),
        # Registered per-step category-structure mode.  OFF keeps every
        # projection identity, slab key, and decision float path
        # byte-identical to a provider without the mode.
        step_category_matrix=bool(step_category_matrix),
        # Conditional so the reference anchored path constructs the provider
        # with byte-identical options and calibration-state identities.
        calibration_options=(
            {
                "deployment_refit_anchor": str(calibration_deployment_refit_anchor),
            }
            if str(calibration_deployment_refit_anchor) != "selected"
            else None
        ),
    )
    # Pin the (already entry-validated) opt-in settings on the provider
    # at construction, before any fit, capture or cache key can read them.
    from mechanistic_systems.measurement.native_irg_providers import (
        pin_frozen_category_optin_settings,
    )

    pin_frozen_category_optin_settings(provider)
    return provider


def _resnet18_fully_pruned_constant_response_proof(
    *,
    runtime: Any,
    fully_pruned_mask: HardMask,
) -> Mapping[str, Any]:
    """Certify that the registered empty ResNet18 route mask is constant.

    Every raw image enters torchvision ResNet18 through ``conv1``.  The
    registered ``stem.conv`` route block contains every legal input/output
    channel pair of that operator.  The complete empty route mask therefore
    makes the stem output independent of the image. Eval BatchNorm may map
    that zero tensor to a nonzero constant tensor, but every later operation
    is then a deterministic function of a context-constant state. Residual
    skips do not reintroduce the raw image. Adaptive pooling and the fixed
    affine readout consequently remain context-constant.

    The proof is deliberately specific to the hash-bound canonical ResNet18
    topology. Other architectures must register an analogous complete
    input-dependence cutset or measure their fully-pruned anchor empirically.
    """

    architecture = runtime.loaded.architecture
    architecture.validate()
    if str(architecture.spec.family).lower() != "resnet18":
        raise ValueError(
            "analytic fully-pruned anchor is certified only for registered ResNet18"
        )
    if fully_pruned_mask.retained_count != 0:
        raise ValueError("analytic chance anchor requires the complete empty mask")
    route_groups = tuple(runtime.route_groups)
    circuit_bindings = tuple(architecture.circuit_bindings)
    fixed_bindings = tuple(
        binding
        for binding in architecture.bindings
        if binding.circuit_role == "fixed_readout"
    )
    bindings_by_layer = {str(binding.layer_id): binding for binding in circuit_bindings}
    groups_by_layer = {str(group.layer_id): group for group in route_groups}
    stem_binding = bindings_by_layer.get("stem.conv")
    stem_group = groups_by_layer.get("stem.conv")
    if (
        len(route_groups) != len(circuit_bindings)
        or len(bindings_by_layer) != len(circuit_bindings)
        or len(groups_by_layer) != len(route_groups)
        or set(groups_by_layer) != set(bindings_by_layer)
        or fully_pruned_mask.route_count
        != sum(int(np.prod(group.shape, dtype=np.int64)) for group in route_groups)
        or not circuit_bindings
        or stem_binding is None
        or stem_group is None
        or stem_binding.module_path != "conv1"
        or stem_binding.route_kind != "conv_channel_pair"
        or len(tuple(stem_group.shape)) != 2
        or any(int(value) <= 0 for value in stem_group.shape)
        or len(fixed_bindings) != 1
        or fixed_bindings[0].route_kind != "linear_weight"
        or fixed_bindings[0].module_path != "fc"
    ):
        raise ValueError(
            "registered ResNet18 route graph cannot prove a constant empty response"
        )
    proof = {
        "schema": "registered_resnet18_empty_route_constant_response_proof_v2",
        "architecture_id": architecture.architecture_id,
        "topology_hash": architecture.topology_hash,
        "circuit_route_schema_hash": architecture.circuit_route_schema_hash,
        "runtime_route_axis_hash": fully_pruned_mask.route_axis_hash,
        "empty_mask_hash": fully_pruned_mask.mask_hash,
        "input_dependence_cutset": {
            "layer_id": stem_binding.layer_id,
            "module_path": stem_binding.module_path,
            "route_kind": stem_binding.route_kind,
            "shape": tuple(stem_group.shape),
            "semantics": (
                "all_raw_image_paths_enter_through_zeroed_stem_channel_pairs"
            ),
        },
        "downstream_constant_map": {
            "normalization": (
                "eval_batchnorm_maps_context_constant_input_to_"
                "context_constant_output"
            ),
            "residuals": "no_raw_input_bypass_after_stem",
            "pool": "torchvision_resnet_adaptive_average_pool",
            "readout_layer_id": fixed_bindings[0].layer_id,
            "readout_module_path": fixed_bindings[0].module_path,
            "readout_semantics": "fixed_affine_bias_on_zero_feature",
        },
        "compiler_semantics": (
            "mechanistic_systems.torch_gate_graph.GatedResidualBlock"
        ),
    }
    proof["content_hash"] = content_hash(
        proof,
        prefix="registered-resnet18-empty-constant-proof-",
    )
    return proof


def _alexnet_fully_pruned_constant_response_proof(
    *,
    runtime: Any,
    fully_pruned_mask: HardMask,
) -> Mapping[str, Any]:
    """Certify that the registered empty AlexNet route mask is constant.

    Every raw image enters torchvision AlexNet through ``features.0``,
    and the registered ``features.stage1.conv`` route block contains
    every legal input/output channel pair of that operator.  The
    complete empty route mask therefore removes every image-dependent
    path at the first operator; the convolution bias maps that absence
    to a per-channel context-constant tensor, and the topology is
    STRICTLY SEQUENTIAL (no residual skips exist to reintroduce the raw
    image), so ReLU, pooling, the gated hidden affines (whose own empty
    route blocks reduce them to their biases), and the fixed affine
    readout each map a context-constant state to a context-constant
    state.  The proof is deliberately specific to the hash-bound
    canonical AlexNet topology, mirroring the ResNet18 proof's cutset
    structure.
    """

    architecture = runtime.loaded.architecture
    architecture.validate()
    if str(architecture.spec.family).lower() != "alexnet":
        raise ValueError(
            "analytic fully-pruned anchor is certified only for the "
            "registered AlexNet"
        )
    if fully_pruned_mask.retained_count != 0:
        raise ValueError("analytic chance anchor requires the complete empty mask")
    route_groups = tuple(runtime.route_groups)
    circuit_bindings = tuple(architecture.circuit_bindings)
    fixed_bindings = tuple(
        binding
        for binding in architecture.bindings
        if binding.circuit_role == "fixed_readout"
    )
    bindings_by_layer = {str(binding.layer_id): binding for binding in circuit_bindings}
    groups_by_layer = {str(group.layer_id): group for group in route_groups}
    stem_binding = bindings_by_layer.get("features.stage1.conv")
    stem_group = groups_by_layer.get("features.stage1.conv")
    if (
        len(route_groups) != len(circuit_bindings)
        or len(bindings_by_layer) != len(circuit_bindings)
        or len(groups_by_layer) != len(route_groups)
        or set(groups_by_layer) != set(bindings_by_layer)
        or fully_pruned_mask.route_count
        != sum(int(np.prod(group.shape, dtype=np.int64)) for group in route_groups)
        or not circuit_bindings
        or stem_binding is None
        or stem_group is None
        or stem_binding.module_path != "features.0"
        or stem_binding.route_kind != "conv_channel_pair"
        or len(tuple(stem_group.shape)) != 2
        or any(int(value) <= 0 for value in stem_group.shape)
        or len(fixed_bindings) != 1
        or fixed_bindings[0].route_kind != "linear_weight"
        or fixed_bindings[0].module_path != "classifier.6"
    ):
        raise ValueError(
            "registered AlexNet route graph cannot prove a constant " "empty response"
        )
    proof = {
        "schema": ("registered_alexnet_empty_route_constant_response_proof_v1"),
        "architecture_id": architecture.architecture_id,
        "topology_hash": architecture.topology_hash,
        "circuit_route_schema_hash": architecture.circuit_route_schema_hash,
        "runtime_route_axis_hash": fully_pruned_mask.route_axis_hash,
        "empty_mask_hash": fully_pruned_mask.mask_hash,
        "input_dependence_cutset": {
            "layer_id": stem_binding.layer_id,
            "module_path": stem_binding.module_path,
            "route_kind": stem_binding.route_kind,
            "shape": tuple(stem_group.shape),
            "semantics": (
                "all_raw_image_paths_enter_through_zeroed_stage1_" "channel_pairs"
            ),
        },
        "downstream_constant_map": {
            "normalization": (
                "conv_bias_maps_zeroed_channel_pairs_to_context_" "constant_output"
            ),
            "residuals": "strictly_sequential_topology_no_bypass",
            "pool": "torchvision_alexnet_adaptive_average_pool",
            "hidden_affines": (
                "empty_hidden_route_blocks_reduce_gated_affines_to_" "their_biases"
            ),
            "readout_layer_id": fixed_bindings[0].layer_id,
            "readout_module_path": fixed_bindings[0].module_path,
            "readout_semantics": "fixed_affine_bias_on_constant_feature",
        },
        "compiler_semantics": (
            "mechanistic_systems.torch_gate_graph.GatedResidualBlock"
        ),
    }
    proof["content_hash"] = content_hash(
        proof,
        prefix="registered-alexnet-empty-constant-proof-",
    )
    return proof


def _alexnet_conv_kernel_weight_fully_pruned_constant_response_proof(
    *,
    runtime: Any,
    fully_pruned_mask: HardMask,
) -> Mapping[str, Any]:
    """Certify the empty kernel-tap AlexNet route mask is constant.

    The tap-granularity route block partitions exactly the same
    ``features.0`` weight set as the channel-pair block, only finer:
    every legal (target, source, kh, kw) tap of the stem convolution is
    a route.  An empty tap mask therefore zeros a superset-equal weight
    set and the channel-pair cutset argument carries over strictly
    strengthened -- no raw-image path survives the first operator, the
    convolution bias maps that absence to a context-constant tensor,
    the topology is strictly sequential, and the gated hidden affines
    and the fixed affine readout preserve context-constancy.  The proof
    carries its own schema string and hash prefix so no receipt can
    conflate granularities.
    """

    architecture = runtime.loaded.architecture
    architecture.validate()
    if str(architecture.spec.family).lower() != "alexnet_conv_kernel_weight":
        raise ValueError(
            "analytic fully-pruned anchor variant is certified only for "
            "the registered kernel-tap AlexNet"
        )
    if fully_pruned_mask.retained_count != 0:
        raise ValueError("analytic chance anchor requires the complete empty mask")
    route_groups = tuple(runtime.route_groups)
    circuit_bindings = tuple(architecture.circuit_bindings)
    fixed_bindings = tuple(
        binding
        for binding in architecture.bindings
        if binding.circuit_role == "fixed_readout"
    )
    bindings_by_layer = {str(binding.layer_id): binding for binding in circuit_bindings}
    groups_by_layer = {str(group.layer_id): group for group in route_groups}
    stem_binding = bindings_by_layer.get("features.stage1.conv")
    stem_group = groups_by_layer.get("features.stage1.conv")
    if (
        len(route_groups) != len(circuit_bindings)
        or len(bindings_by_layer) != len(circuit_bindings)
        or len(groups_by_layer) != len(route_groups)
        or set(groups_by_layer) != set(bindings_by_layer)
        or fully_pruned_mask.route_count
        != sum(int(np.prod(group.shape, dtype=np.int64)) for group in route_groups)
        or not circuit_bindings
        or stem_binding is None
        or stem_group is None
        or stem_binding.module_path != "features.0"
        or stem_binding.route_kind != "conv_kernel_weight"
        or len(tuple(stem_group.shape)) != 4
        or any(int(value) <= 0 for value in stem_group.shape)
        or len(fixed_bindings) != 1
        or fixed_bindings[0].route_kind != "linear_weight"
        or fixed_bindings[0].module_path != "classifier.6"
    ):
        raise ValueError(
            "registered kernel-tap AlexNet route graph cannot prove a "
            "constant empty response"
        )
    proof = {
        "schema": (
            "registered_alexnet_empty_route_constant_response_proof_"
            "conv_kernel_weight_v1"
        ),
        "architecture_id": architecture.architecture_id,
        "topology_hash": architecture.topology_hash,
        "circuit_route_schema_hash": architecture.circuit_route_schema_hash,
        "runtime_route_axis_hash": fully_pruned_mask.route_axis_hash,
        "empty_mask_hash": fully_pruned_mask.mask_hash,
        "input_dependence_cutset": {
            "layer_id": stem_binding.layer_id,
            "module_path": stem_binding.module_path,
            "route_kind": stem_binding.route_kind,
            "shape": tuple(stem_group.shape),
            "semantics": (
                "all_raw_image_paths_enter_through_zeroed_stage1_" "kernel_taps"
            ),
        },
        "downstream_constant_map": {
            "normalization": (
                "conv_bias_maps_zeroed_kernel_taps_to_context_" "constant_output"
            ),
            "residuals": "strictly_sequential_topology_no_bypass",
            "pool": "torchvision_alexnet_adaptive_average_pool",
            "hidden_affines": (
                "empty_hidden_route_blocks_reduce_gated_affines_to_" "their_biases"
            ),
            "readout_layer_id": fixed_bindings[0].layer_id,
            "readout_module_path": fixed_bindings[0].module_path,
            "readout_semantics": "fixed_affine_bias_on_constant_feature",
        },
        "compiler_semantics": (
            "mechanistic_systems.torch_gate_graph.GatedResidualBlock"
        ),
    }
    proof["content_hash"] = content_hash(
        proof,
        prefix="registered-alexnet-conv-kernel-weight-empty-constant-proof-",
    )
    return proof


def _conv_kernel_weight_cutset_constant_response_proof(
    *,
    runtime: Any,
    fully_pruned_mask: HardMask,
    architecture_family: str,
    architecture_label: str,
    cutset_layer_id: str,
    cutset_module_path: str,
    fixed_readout_module_path: str,
    proof_schema: str,
    hash_prefix: str,
    cutset_semantics: str,
    normalization_semantics: str,
    residual_semantics: str,
    pool_semantics: str,
    compiler_semantics: str,
    hidden_affine_semantics: str | None = None,
    registered_model_source: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Build a family-specific empty-mask proof from a complete conv cutset.

    This helper shares only the structural verification grammar.  Each caller
    remains bound to an exact registered architecture family, cutset alias,
    proof schema, and hash namespace so evidence cannot pool across model
    topologies or route granularities.
    """

    architecture = runtime.loaded.architecture
    architecture.validate()
    if str(architecture.spec.family).lower() != architecture_family:
        raise ValueError(
            "analytic fully-pruned anchor variant is certified only for "
            f"the registered {architecture_label}"
        )
    if fully_pruned_mask.retained_count != 0:
        raise ValueError("analytic chance anchor requires the complete empty mask")
    source_model = getattr(runtime.loaded, "model", None)
    evaluator = getattr(runtime, "evaluator", None)
    compiled_model = getattr(evaluator, "model", None)
    if (
        source_model is None
        or compiled_model is None
        or bool(getattr(source_model, "training", True))
        or bool(getattr(compiled_model, "training", True))
    ):
        raise ValueError(
            "analytic fully-pruned anchor requires source and compiled "
            "models bound in eval mode"
        )
    route_groups = tuple(runtime.route_groups)
    circuit_bindings = tuple(architecture.circuit_bindings)
    fixed_bindings = tuple(
        binding
        for binding in architecture.bindings
        if binding.circuit_role == "fixed_readout"
    )
    bindings_by_layer = {str(binding.layer_id): binding for binding in circuit_bindings}
    groups_by_layer = {str(group.layer_id): group for group in route_groups}
    cutset_binding = bindings_by_layer.get(cutset_layer_id)
    cutset_group = groups_by_layer.get(cutset_layer_id)
    if (
        len(route_groups) != len(circuit_bindings)
        or len(bindings_by_layer) != len(circuit_bindings)
        or len(groups_by_layer) != len(route_groups)
        or set(groups_by_layer) != set(bindings_by_layer)
        or fully_pruned_mask.route_count
        != sum(int(np.prod(group.shape, dtype=np.int64)) for group in route_groups)
        or not circuit_bindings
        or cutset_binding is None
        or cutset_group is None
        or cutset_binding.module_path != cutset_module_path
        or cutset_binding.route_kind != "conv_kernel_weight"
        or len(tuple(cutset_group.shape)) != 4
        or any(int(value) <= 0 for value in cutset_group.shape)
        or len(fixed_bindings) != 1
        or fixed_bindings[0].route_kind != "linear_weight"
        or fixed_bindings[0].module_path != fixed_readout_module_path
    ):
        raise ValueError(
            f"registered {architecture_label} route graph cannot prove a "
            "constant empty response"
        )
    downstream_constant_map = {
        "normalization": normalization_semantics,
        "residuals": residual_semantics,
        "pool": pool_semantics,
        "readout_layer_id": fixed_bindings[0].layer_id,
        "readout_module_path": fixed_bindings[0].module_path,
        "readout_semantics": "fixed_affine_bias_on_constant_feature",
    }
    if hidden_affine_semantics is not None:
        downstream_constant_map["hidden_affines"] = hidden_affine_semantics
    proof = {
        "schema": proof_schema,
        "architecture_id": architecture.architecture_id,
        "topology_hash": architecture.topology_hash,
        "circuit_route_schema_hash": architecture.circuit_route_schema_hash,
        "runtime_route_axis_hash": fully_pruned_mask.route_axis_hash,
        "empty_mask_hash": fully_pruned_mask.mask_hash,
        "input_dependence_cutset": {
            "layer_id": cutset_binding.layer_id,
            "module_path": cutset_binding.module_path,
            "route_kind": cutset_binding.route_kind,
            "shape": tuple(cutset_group.shape),
            "semantics": cutset_semantics,
        },
        "downstream_constant_map": downstream_constant_map,
        "compiler_semantics": compiler_semantics,
    }
    if registered_model_source is not None:
        proof["registered_model_source"] = dict(registered_model_source)
    proof["content_hash"] = content_hash(proof, prefix=hash_prefix)
    return proof


def _vgg11_conv_kernel_weight_fully_pruned_constant_response_proof(
    *,
    runtime: Any,
    fully_pruned_mask: HardMask,
) -> Mapping[str, Any]:
    """Certify the empty kernel-tap VGG11 route mask is constant."""

    return _conv_kernel_weight_cutset_constant_response_proof(
        runtime=runtime,
        fully_pruned_mask=fully_pruned_mask,
        architecture_family="vgg11_conv_kernel_weight",
        architecture_label="kernel-tap VGG11",
        cutset_layer_id="features.stage3.conv1",
        cutset_module_path="features.6",
        fixed_readout_module_path="classifier.6",
        proof_schema=(
            "registered_vgg11_empty_route_constant_response_proof_"
            "conv_kernel_weight_v1"
        ),
        hash_prefix=("registered-vgg11-conv-kernel-weight-empty-constant-proof-"),
        cutset_semantics=(
            "all_raw_image_paths_enter_registered_stage3_through_zeroed_" "kernel_taps"
        ),
        normalization_semantics=(
            "conv_bias_maps_zeroed_kernel_taps_to_context_constant_output"
        ),
        residual_semantics="strictly_sequential_topology_no_bypass",
        pool_semantics="torchvision_vgg_adaptive_average_pool",
        hidden_affine_semantics=(
            "empty_hidden_route_blocks_reduce_gated_affines_to_their_biases"
        ),
        compiler_semantics=(
            "mechanistic_systems.backends.torch_gate_graph." "GatedConv2d+GatedLinear"
        ),
    )


def _resnet50_conv_kernel_weight_fully_pruned_constant_response_proof(
    *,
    runtime: Any,
    fully_pruned_mask: HardMask,
) -> Mapping[str, Any]:
    """Certify the empty kernel-tap ResNet50 route mask is constant."""

    receipt = getattr(getattr(runtime, "loaded", None), "receipt", None)
    if str(getattr(receipt, "source_kind", "")) == ("checkpoint_qualified_state_dict"):
        raise ValueError(
            "checkpoint-qualified ResNet50 sources require their registered "
            "source-specific fully-pruned proof"
        )

    return _conv_kernel_weight_cutset_constant_response_proof(
        runtime=runtime,
        fully_pruned_mask=fully_pruned_mask,
        architecture_family="resnet50_conv_kernel_weight",
        architecture_label="kernel-tap ResNet50",
        cutset_layer_id="stem.conv",
        cutset_module_path="conv1",
        fixed_readout_module_path="fc",
        proof_schema=(
            "registered_resnet50_empty_route_constant_response_proof_"
            "conv_kernel_weight_v1"
        ),
        hash_prefix=("registered-resnet50-conv-kernel-weight-empty-constant-proof-"),
        cutset_semantics=("all_raw_image_paths_enter_through_zeroed_stem_kernel_taps"),
        normalization_semantics=(
            "eval_batchnorm_maps_context_constant_input_to_context_" "constant_output"
        ),
        residual_semantics="no_raw_input_bypass_after_stem",
        pool_semantics="torchvision_resnet_adaptive_average_pool",
        compiler_semantics=(
            "mechanistic_systems.backends.torch_gate_graph.GatedConv2d+"
            "GatedResidualBlock"
        ),
    )


def _resnet18_conv_kernel_weight_fully_pruned_constant_response_proof(
    *,
    runtime: Any,
    fully_pruned_mask: HardMask,
) -> Mapping[str, Any]:
    """Certify the empty kernel-tap ResNet18 route mask is constant."""

    return _conv_kernel_weight_cutset_constant_response_proof(
        runtime=runtime,
        fully_pruned_mask=fully_pruned_mask,
        architecture_family="resnet18_conv_kernel_weight",
        architecture_label="kernel-tap ResNet18",
        cutset_layer_id="stem.conv",
        cutset_module_path="conv1",
        fixed_readout_module_path="fc",
        proof_schema=(
            "registered_resnet18_empty_route_constant_response_proof_"
            "conv_kernel_weight_v1"
        ),
        hash_prefix=("registered-resnet18-conv-kernel-weight-empty-constant-proof-"),
        cutset_semantics=("all_raw_image_paths_enter_through_zeroed_stem_kernel_taps"),
        normalization_semantics=(
            "eval_batchnorm_maps_context_constant_input_to_context_" "constant_output"
        ),
        residual_semantics="no_raw_input_bypass_after_stem",
        pool_semantics="torchvision_resnet_adaptive_average_pool",
        compiler_semantics=(
            "mechanistic_systems.backends.torch_gate_graph.GatedConv2d+"
            "GatedResidualBlock"
        ),
    )


def _robust_resnet50_conv_kernel_weight_fully_pruned_constant_response_proof(
    *,
    runtime: Any,
    fully_pruned_mask: HardMask,
) -> Mapping[str, Any]:
    """Certify the Salman2020Do_R50 empty mask with exact source binding."""

    source_binding = robust_resnet50_checkpoint_proof_binding(runtime)
    return _conv_kernel_weight_cutset_constant_response_proof(
        runtime=runtime,
        fully_pruned_mask=fully_pruned_mask,
        architecture_family="resnet50_conv_kernel_weight",
        architecture_label="checkpoint-qualified Salman2020Do_R50",
        cutset_layer_id="stem.conv",
        cutset_module_path="conv1",
        fixed_readout_module_path="fc",
        proof_schema=(
            "registered_salman2020do_r50_empty_route_constant_response_"
            "proof_conv_kernel_weight_v1"
        ),
        hash_prefix=(
            "registered-salman2020do-r50-conv-kernel-weight-empty-" "constant-proof-"
        ),
        cutset_semantics=("all_raw_image_paths_enter_through_zeroed_stem_kernel_taps"),
        normalization_semantics=(
            "one_external_imagenet_input_operator_precedes_a_bare_resnet;_"
            "zeroed_stem_taps_remove_dependence_on_its_output_and_eval_"
            "batchnorm_preserves_context_constancy"
        ),
        residual_semantics="no_raw_input_bypass_after_stem",
        pool_semantics="torchvision_resnet_adaptive_average_pool",
        compiler_semantics=(
            "mechanistic_systems.backends.torch_gate_graph.GatedConv2d+"
            "GatedResidualBlock"
        ),
        registered_model_source=source_binding,
    )


_FULLY_PRUNED_CONSTANT_RESPONSE_PROOFS = {
    "resnet18": _resnet18_fully_pruned_constant_response_proof,
    "resnet18_conv_kernel_weight": (
        _resnet18_conv_kernel_weight_fully_pruned_constant_response_proof
    ),
    "alexnet": _alexnet_fully_pruned_constant_response_proof,
    "alexnet_conv_kernel_weight": (
        _alexnet_conv_kernel_weight_fully_pruned_constant_response_proof
    ),
    "vgg11_conv_kernel_weight": (
        _vgg11_conv_kernel_weight_fully_pruned_constant_response_proof
    ),
    "resnet50_conv_kernel_weight": (
        _resnet50_conv_kernel_weight_fully_pruned_constant_response_proof
    ),
}


def _fully_pruned_constant_response_proof(
    *,
    runtime: Any,
    fully_pruned_mask: HardMask,
) -> Mapping[str, Any]:
    """Dispatch the analytic empty-mask proof by registered family."""

    family = str(runtime.loaded.architecture.spec.family).lower()
    model_id = str(getattr(getattr(runtime.loaded, "receipt", None), "model_id", ""))
    if model_id == ROBUSTBENCH_SALMAN2020DO_R50_MODEL_ID:
        return _robust_resnet50_conv_kernel_weight_fully_pruned_constant_response_proof(
            runtime=runtime,
            fully_pruned_mask=fully_pruned_mask,
        )
    proof = _FULLY_PRUNED_CONSTANT_RESPONSE_PROOFS.get(family)
    if proof is None:
        raise ValueError(
            "no registered analytic fully-pruned proof for architecture "
            f"family {family!r}; register a cutset proof or measure the "
            "anchor empirically"
        )
    return proof(runtime=runtime, fully_pruned_mask=fully_pruned_mask)


def _validated_terminal_calibration_settings(
    rule: Any,
    tolerance: Any | None,
) -> tuple[str, float | None]:
    """Validate terminal-only settings without changing a calibration state."""

    rule = str(rule).strip()
    if rule not in {"absolute_tolerance", "risk_radius", "walk_resolution"}:
        raise ValueError(
            "IRG_TERMINAL_CALIBRATION_RULE must be absolute_tolerance, "
            "risk_radius or walk_resolution"
        )
    if rule in {"risk_radius", "walk_resolution"}:
        if tolerance is not None:
            raise ValueError(
                "IRG_TERMINAL_CALIBRATION_TOLERANCE requires rule absolute_tolerance"
            )
        return rule, None
    try:
        parsed_tolerance = 1e-4 if tolerance is None else float(tolerance)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "IRG_TERMINAL_CALIBRATION_TOLERANCE must be a positive finite float"
        ) from exc
    if not math.isfinite(parsed_tolerance) or parsed_tolerance <= 0.0:
        raise ValueError(
            "IRG_TERMINAL_CALIBRATION_TOLERANCE must be a positive finite float"
        )
    return rule, parsed_tolerance


def _terminal_calibration_settings(
    environment: Mapping[str, str] | None = None,
) -> tuple[str, float | None]:
    """Read the terminal rule from campaign-exported engine settings."""

    source = os.environ if environment is None else environment
    return _validated_terminal_calibration_settings(
        source.get("IRG_TERMINAL_CALIBRATION_RULE", "absolute_tolerance"),
        source.get("IRG_TERMINAL_CALIBRATION_TOLERANCE"),
    )


def _walk_resolution_calibration_tolerance(
    *,
    decision_summaries: Sequence[Mapping[str, Any]],
    sensitivity_tau: float,
    positive_loss_span_estimate: float,
) -> Mapping[str, Any]:
    """Convert the finest accepted v0 step price back to weighted BCE units.

    The step statistic is delta_BCE / (loss_span * routes_removed), so the
    terminal bound's tolerance is tau * min_accepted_routes * loss_span.
    The caller supplies the final walk's current loss-span estimate so the
    normalized price is converted into the bound's loss units at mask lock.
    The loss span is the dense-to-empty risk difference; there is no division
    by the network's total connection count.  With no accepted step there is
    no positive walk resolution: use zero and retain only the numerical floor,
    population ceiling and stall stopping conditions.
    """

    tau = float(sensitivity_tau)
    span = float(positive_loss_span_estimate)
    if not math.isfinite(tau) or tau < 0.0:
        raise ValueError("walk resolution requires finite nonnegative sensitivity_tau")
    if not math.isfinite(span) or span <= 0.0:
        raise ValueError("walk resolution requires a positive finite loss span")
    accepted_routes = []
    for decision in decision_summaries:
        if decision.get("action") != "accept":
            continue
        routes = decision.get("removed_route_count")
        if isinstance(routes, bool) or not isinstance(routes, (int, np.integer)) or routes <= 0:
            raise ValueError("accepted walk steps require positive integer removed_route_count")
        accepted_routes.append(int(routes))
    smallest = min(accepted_routes) if accepted_routes else None
    tolerance = tau * float(smallest or 0) * span
    if not math.isfinite(tolerance):
        raise ValueError("walk resolution tolerance must be finite")
    return {
        "tolerance": tolerance,
        "sensitivity_tau": tau,
        "smallest_accepted_step_removed_route_count": smallest,
        "positive_loss_span_estimate": span,
        "accepted_step_count": len(accepted_routes),
        "tolerance_formula": (
            "sensitivity_tau * smallest_accepted_step_removed_route_count "
            "* positive_loss_span_estimate"
        ),
        "no_accepted_step_policy": (
            "zero_tolerance_numerical_floor_population_ceiling_or_stall"
            if smallest is None else None
        ),
    }


def _close_terminal_mask_calibration(
    *,
    provider: NativeTorchCalibratedBCEOracleProvider,
    mask: HardMask,
    confidence_level: float,
    cumulative_risk_context_count: int,
    certificate: Any | None = None,
    certificate_role: str,
    stream_namespace: str,
    progress_callback: Callable[..., None] | None = None,
    rule: str | None = None,
    tolerance: float | None = None,
    walk_resolution_context: Mapping[str, Any] | None = None,
) -> tuple[Any, Mapping[str, Any]]:
    """Select terminal closure; explicit settings support saved-mask rescoring.

    An explicit rule is independent of the process environment.  The legacy
    rule retains its original receipt, including its content hash.
    """

    if rule is None and tolerance is None:
        rule, tolerance = _terminal_calibration_settings()
    else:
        rule, tolerance = _validated_terminal_calibration_settings(
            os.environ.get("IRG_TERMINAL_CALIBRATION_RULE", "absolute_tolerance")
            if rule is None else rule,
            tolerance,
        )
    closure_kwargs = dict(
        provider=provider,
        mask=mask,
        confidence_level=confidence_level,
        cumulative_risk_context_count=cumulative_risk_context_count,
        certificate=certificate,
        certificate_role=certificate_role,
        stream_namespace=stream_namespace,
        progress_callback=progress_callback,
    )
    if rule == "absolute_tolerance":
        closure_kwargs["absolute_tolerance"] = tolerance
    elif rule == "walk_resolution":
        if walk_resolution_context is None:
            raise ValueError("walk_resolution requires the accepted walk's resolution context")
        closure_kwargs["walk_resolution_context"] = walk_resolution_context
    return _close_locked_mask_calibration_to_risk_resolution(**closure_kwargs)


def _close_locked_mask_calibration_to_risk_resolution(
    *,
    provider: NativeTorchCalibratedBCEOracleProvider,
    mask: HardMask,
    confidence_level: float,
    cumulative_risk_context_count: int,
    certificate: Any | None = None,
    certificate_role: str,
    stream_namespace: str,
    progress_callback: Callable[..., None] | None = None,
    absolute_tolerance: float | None = None,
    walk_resolution_context: Mapping[str, Any] | None = None,
) -> tuple[Any, Mapping[str, Any]]:
    """Close affine-fit uncertainty to the resolution of an independent risk audit.

    The oracle interval and calibration fit use independent logical IID
    streams.  Additional calibration rows are useful only while the maximum
    population-BCE improvement still attainable by refitting exceeds the
    confidence radius of the frozen oracle evidence.  This gives calibration
    a decision-relevant stopping rule, with calibration rows bounded by the
    size of their training population.  The terminal-only caller can instead
    select an absolute bound; omitting it preserves the risk-radius path and
    its receipt exactly.
    """

    if absolute_tolerance is not None:
        _, absolute_tolerance = _validated_terminal_calibration_settings(
            "absolute_tolerance", absolute_tolerance
        )
    walk_tolerance = None
    if walk_resolution_context is not None:
        if absolute_tolerance is not None:
            raise ValueError("walk resolution and absolute tolerance are mutually exclusive")
        walk_tolerance = float(walk_resolution_context["tolerance"])
        if not math.isfinite(walk_tolerance) or walk_tolerance < 0.0:
            raise ValueError("walk resolution tolerance must be finite and nonnegative")

    risk_certificate = (
        certificate
        if certificate is not None
        else provider.certify_baseline_risk(
            mask,
            cumulative_context_count=int(cumulative_risk_context_count),
            confidence_level=float(confidence_level),
            maximum_planned_looks=None,
            precision_only_iid=True,
            certificate_role=str(certificate_role),
            stream_namespace=str(stream_namespace),
        )
    )
    if risk_certificate.mask_hash != mask.mask_hash:
        raise RuntimeError("terminal calibration closure received the wrong mask")
    initial_fit = provider.calibration_fits[mask.mask_hash]
    initial_certificate = risk_certificate
    refinement_states: list[Mapping[str, Any]] = []
    stall_streak = 0
    while True:
        fit = provider.calibration_fits[mask.mask_hash]
        risk_radius = max(
            float(risk_certificate.estimate) - float(risk_certificate.lower),
            float(risk_certificate.upper) - float(risk_certificate.estimate),
        )
        if not math.isfinite(risk_radius) or risk_radius < 0.0:
            raise RuntimeError("terminal risk certificate has an invalid radius")
        calibration_bound = float(fit.maximum_bce_excess_risk_bound)
        # A structurally constant readout yields a deterministic loss, so its
        # independent risk sample legitimately has zero width while the fit
        # reports a bound at float64 rounding scale.  Resolve such bounds
        # against a scale-aware numerical floor instead of demanding a
        # strictly positive radius the evidence cannot possess.
        numerical_resolution_floor = (
            64.0
            * float(np.finfo(np.float64).eps)
            * max(1.0, abs(float(risk_certificate.estimate)))
        )
        if walk_tolerance is not None:
            stopping_resolution = (
                "walk_resolution"
                if calibration_bound <= walk_tolerance
                else (
                    "numerical_floor"
                    if calibration_bound <= numerical_resolution_floor
                    else None
                )
            )
        elif absolute_tolerance is None:
            stopping_resolution = (
                "risk_radius"
                if risk_radius > 0.0 and calibration_bound <= risk_radius
                else (
                    "numerical_floor"
                    if calibration_bound <= numerical_resolution_floor
                    else None
                )
            )
        else:
            stopping_resolution = (
                "absolute_tolerance"
                if calibration_bound <= absolute_tolerance
                else (
                    "numerical_floor"
                    if calibration_bound <= numerical_resolution_floor
                    else None
                )
            )
        refinement_states.append(
            {
                "calibration_state_id": fit.calibration_state_id,
                "calibration_context_count": int(
                    sum(fit.sampled_stratum_counts.values())
                ),
                "maximum_attainable_bce_improvement": calibration_bound,
                "independent_risk_confidence_radius": risk_radius,
                "numerical_resolution_floor": numerical_resolution_floor,
                "stopping_resolution": stopping_resolution,
            }
        )
        if stopping_resolution is not None:
            break
        if absolute_tolerance is None and walk_tolerance is None and risk_radius <= 0.0:
            raise RuntimeError(
                "zero-width risk evidence cannot resolve calibration "
                "uncertainty above the numerical resolution floor"
            )
        prior_state_id = fit.calibration_state_id
        prior_bound = calibration_bound
        try:
            provider.ensure_calibration(
                mask,
                target_loss_precision=(
                    max(walk_tolerance, numerical_resolution_floor)
                    if walk_tolerance is not None else (
                        risk_radius if absolute_tolerance is None else absolute_tolerance
                    )
                ),
            )
        except CalibrationPrecisionHorizonExhausted as exc:
            if exc.receipt.get("horizon_semantics") != "population_size_ceiling":
                raise
            ceiling_fit = provider.calibration_fits[mask.mask_hash]
            refinement_states.append(
                {
                    "calibration_state_id": ceiling_fit.calibration_state_id,
                    "calibration_context_count": int(
                        sum(ceiling_fit.sampled_stratum_counts.values())
                    ),
                    "maximum_attainable_bce_improvement": float(
                        ceiling_fit.maximum_bce_excess_risk_bound
                    ),
                    "independent_risk_confidence_radius": risk_radius,
                    "numerical_resolution_floor": numerical_resolution_floor,
                    "stopping_resolution": "population_ceiling",
                }
            )
            # Reapply the reached fit to the same frozen risk rows, so the
            # returned certificate carries the final calibration identity.
            # This refresh preserves their allocation and statistical look.
            risk_certificate = provider.refresh_baseline_calibration_certificate(
                risk_certificate
            )
            break
        refined_fit = provider.calibration_fits[mask.mask_hash]
        refinement_stalled = (
            refined_fit.calibration_state_id == prior_state_id
            or float(refined_fit.maximum_bce_excess_risk_bound) >= prior_bound
        )
        # An anchored fit's bound contracts strictly with rows, but an honest
        # unregularized bound can legitimately sit flat between looks. Two
        # consecutive non-improving refinements stop closure with an explicit
        # stalled receipt instead of failing the run.
        stall_streak = stall_streak + 1 if refinement_stalled else 0
        if stall_streak >= 2:
            stalled_fit = provider.calibration_fits[mask.mask_hash]
            refinement_states.append(
                {
                    "calibration_state_id": stalled_fit.calibration_state_id,
                    "calibration_context_count": int(
                        sum(stalled_fit.sampled_stratum_counts.values())
                    ),
                    "maximum_attainable_bce_improvement": float(
                        stalled_fit.maximum_bce_excess_risk_bound
                    ),
                    "independent_risk_confidence_radius": risk_radius,
                    "numerical_resolution_floor": numerical_resolution_floor,
                    "stopping_resolution": "refinement_stalled",
                }
            )
            if walk_tolerance is not None:
                risk_certificate = provider.refresh_baseline_calibration_certificate(
                    risk_certificate
                )
            break
        risk_certificate = provider.refresh_baseline_calibration_certificate(
            risk_certificate
        )

    final_fit = provider.calibration_fits[mask.mask_hash]
    receipt: dict[str, Any] = {
        "schema": "validation_independent_terminal_calibration_closure_v1",
        "mask_hash": mask.mask_hash,
        "certificate_role": str(certificate_role),
        "confidence_level": float(confidence_level),
        "stopping_rule": (
            "maximum_attainable_bce_improvement_not_greater_than_"
            "independent_training_risk_confidence_radius_or_float64_"
            "numerical_resolution_floor"
        ),
        "stopping_resolution": (
            refinement_states[-1]["stopping_resolution"] if refinement_states else None
        ),
        "validation_contexts_observed": False,
        "risk_context_count": int(risk_certificate.context_count),
        "risk_certificate_hash": risk_certificate.content_hash,
        # Promote the final locked-mask training-risk estimate itself, not
        # merely its hash/radius.  Campaign summaries must be able to report
        # dense and pruned train BCE side by side without replaying the model
        # or attempting to reverse an opaque certificate hash.
        "final_training_bce_estimate": float(risk_certificate.estimate),
        "final_training_bce_lower": float(risk_certificate.lower),
        "final_training_bce_upper": float(risk_certificate.upper),
        "initial_risk_certificate_hash": initial_certificate.content_hash,
        "initial_calibration_state_id": initial_fit.calibration_state_id,
        "final_calibration_state_id": final_fit.calibration_state_id,
        "initial_calibration_context_count": int(
            sum(initial_fit.sampled_stratum_counts.values())
        ),
        "final_calibration_context_count": int(
            sum(final_fit.sampled_stratum_counts.values())
        ),
        "final_maximum_attainable_bce_improvement": float(
            final_fit.maximum_bce_excess_risk_bound
        ),
        "final_independent_risk_confidence_radius": max(
            float(risk_certificate.estimate) - float(risk_certificate.lower),
            float(risk_certificate.upper) - float(risk_certificate.estimate),
        ),
        "refinement_states": tuple(refinement_states),
    }
    if absolute_tolerance is not None:
        receipt.update({
            "schema": "validation_independent_terminal_calibration_closure_v2",
            "rule": "absolute_tolerance",
            "tolerance": absolute_tolerance,
            "stopping_rule": (
                "maximum_attainable_bce_improvement_not_greater_than_"
                "absolute_tolerance_or_float64_numerical_resolution_floor"
            ),
        })
    if walk_resolution_context is not None:
        receipt.update({
            "schema": "validation_independent_terminal_calibration_closure_v3",
            "rule": "walk_resolution",
            **dict(walk_resolution_context),
            "stopping_rule": (
                "maximum_attainable_bce_improvement_not_greater_than_"
                "walk_resolution_or_float64_numerical_resolution_floor"
            ),
        })
    if getattr(provider, "pruning_functional", "target_logit") == "substituted_softmax":
        receipt.update(pruning_functional="substituted_softmax", loss_weighting="per_image",
                       loss_units="per_image_substituted_binary_natural_log_bce")
    receipt["content_hash"] = content_hash(
        receipt,
        prefix="terminal-calibration-closure-",
    )
    if progress_callback is not None:
        progress_callback(
            "validation_independent_terminal_calibration_closure_completed",
            **receipt,
        )
    return risk_certificate, receipt



def _substituted_scale_zero_risk_certificate(
    *, provider, reference_mask, confidence_level, population_closure=None,
):
    """Freeze the TERMSTATS2 scale-zero training risk used to price steps."""
    reference = dict((population_closure or {}).get("scale_zero_reference") or
                     provider.decision_scale_zero_reference())
    risk = float(reference["per_image_training_bce"])
    if not reference.get("converged") or not math.isfinite(risk) or risk < 0.0:
        raise ValueError("substituted scale-zero training reference did not converge")
    reference.update(pruning_functional="substituted_softmax", loss_weighting="per_image",
        definition="eta_i = b - LSE_{j != k} z_j(dense)(x_i); a=0; b fitted by per-image Firth on training rows")
    reference["content_hash"] = content_hash(reference, prefix="substituted-scale-zero-reference-")
    certificate = BaselineRiskCertificate(
        estimate=risk, lower=risk, upper=risk,
        context_count=int(reference["training_context_count"]),
        calibration_state_id=reference["content_hash"],
        sampling_design_id="fixed_training_empirical_scale_zero_reference_v1",
        data_role="scale_zero_substituted_training_reference",
        code_identity="termstats2_substituted_firth_per_image_scale_zero_v1",
        output_identity=reference["content_hash"],
        confidence_level=float(confidence_level), confidence_scope=PRECISION_ONLY_IID_SCOPE,
        mask_hash=reference_mask.mask_hash,
        schema="substituted_scale_zero_training_risk_v1")
    return certificate, reference


def _close_positive_dense_to_empty_span(
    *,
    provider: NativeTorchCalibratedBCEOracleProvider,
    dense_mask: HardMask,
    empty_certificate: AnalyticChanceRiskCertificate,
    dense_certificate: BaselineRiskCertificate,
    confidence_level: float,
    cumulative_risk_context_count: int,
    progress_callback: Callable[..., None] | None = None,
    calibration_closer: Callable[..., tuple[Any, Mapping[str, Any]]] | None = None,
) -> tuple[BaselineRiskCertificate, LossSpanCertificate, Mapping[str, Any]]:
    """Continue training-only dense-risk evidence until its span is signed.

    The v0 normalization requires a *certified* positive dense-to-empty BCE
    span.  A first finite IID panel can have a confidence interval wider than
    that span even when its point estimate is comfortably below chance.  That
    is unresolved evidence, not evidence that the architecture is invalid.

    Each continuation asks ``certify_baseline_risk`` for a doubled prefix of
    the same sealed, counter-addressed training stream.  That provider applies
    its precision-only alpha spending at every look.  Calibration is then
    closed against the narrower independent-risk radius before the span is
    tested again.  Validation remains unopened throughout.  A certificate
    whose most favorable admissible span is nonpositive refuses immediately;
    the positivity criterion itself is never weakened.
    """

    count = int(cumulative_risk_context_count)
    if count <= 0:
        raise ValueError("dense-to-empty span closure requires positive rows")
    certificate = dense_certificate
    looks: list[Mapping[str, Any]] = []
    while True:
        span = LossSpanCertificate(
            dense=certificate,
            fully_pruned=empty_certificate,
        )
        look = {
            "risk_context_count": int(certificate.context_count),
            "dense_risk_estimate": float(certificate.estimate),
            "dense_risk_lower": float(certificate.lower),
            "dense_risk_upper": float(certificate.upper),
            "loss_span_estimate": float(span.estimate),
            "loss_span_lower": float(span.lower),
            "loss_span_upper": float(span.upper),
            "risk_certificate_hash": certificate.content_hash,
        }
        looks.append(look)
        if span.lower > 0.0 and span.estimate > 0.0:
            span.validate_positive()
            break
        if span.upper <= 0.0:
            raise ValueError("v0 dense-to-empty BCE span is certified nonpositive")
        next_count = count * 2
        if next_count <= count:
            raise OverflowError("dense-to-empty span row target overflowed")
        count = next_count
        certificate = provider.certify_baseline_risk(
            dense_mask,
            cumulative_context_count=count,
            confidence_level=float(confidence_level),
            maximum_planned_looks=None,
            precision_only_iid=True,
            certificate_role=("dense_baseline_proposal_adjudication_reference"),
            stream_namespace="dense-baseline",
        )
        close_calibration = (
            _close_locked_mask_calibration_to_risk_resolution
            if calibration_closer is None
            else calibration_closer
        )
        certificate, _closure = close_calibration(
            provider=provider,
            mask=dense_mask,
            confidence_level=float(confidence_level),
            cumulative_risk_context_count=count,
            certificate=certificate,
            certificate_role=("dense_baseline_proposal_adjudication_reference"),
            stream_namespace="dense-baseline",
            progress_callback=progress_callback,
        )

    receipt: dict[str, Any] = {
        "schema": "positive_dense_to_empty_span_closure_v1",
        "sampling_role": "training_only_proposal_adjudication",
        "validation_contexts_observed": False,
        "continuation_rule": (
            "double_counter_addressed_iid_prefix_until_positive_or_"
            "certified_nonpositive"
        ),
        "look_count": len(looks),
        "looks": tuple(looks),
        "final_risk_certificate_hash": certificate.content_hash,
        "final_loss_span_certificate_hash": span.content_hash,
    }
    if getattr(provider, "pruning_functional", "target_logit") == "substituted_softmax":
        receipt.update(pruning_functional="substituted_softmax", loss_weighting="per_image",
                       reference="training_fitted_substituted_scale_zero_risk",
                       loss_units="per_image_substituted_binary_natural_log_bce")
    receipt["content_hash"] = content_hash(
        receipt,
        prefix="positive-dense-to-empty-span-closure-",
    )
    if progress_callback is not None:
        progress_callback(
            "positive_dense_to_empty_span_closure_completed",
            **receipt,
        )
    return certificate, span, receipt


def _a47_sealed_queries(
    measurement_artifact_policy: str,
    measurement_plan_path: str | Path | None,
):
    """Under contracts_only the sealed plan supplies the query axis."""

    if str(measurement_artifact_policy) != "contracts_only":
        return None
    if measurement_plan_path is None:
        raise ValueError("contracts_only requires the sealed measurement plan path")
    from mechanistic_systems.contracts import MeasurementPlan

    return MeasurementPlan.load(measurement_plan_path).queries


def _v6_walk_products(runs: Sequence[Any]) -> Any:
    """Start serializing the v6 walk logs of ``runs`` (None without any)."""

    from mechanistic_systems.circuits.v6_records import (
        V6_LEDGER_SCHEMA,
        V6WalkProducts,
    )

    results = []
    for item in runs:
        ledger = getattr(getattr(item, "result", None), "ledger_receipt", None)
        if isinstance(ledger, Mapping) and ledger.get("schema_name") == V6_LEDGER_SCHEMA:
            results.append(ledger["v6_walk_log"].result())
    return V6WalkProducts(results) if results else None


def _walk_tail_block(
    *,
    accepted_v0_trajectory,
    dense_mask,
    direct_candidate_comparison,
    family,
    final_mask,
    hard_partition_measurement,
    maximum_commits,
    partition_randomization_receipt,
    query_id,
    resolved_sensitivity_tau,
    runs,
    runtime,
    runtime_architecture_family,
    scoring_anchor_elapsed,
    traversal_started,
    v0_attribution_order,
    v0_candidate_registry,
    v0_decision_backend,
    v0_loss_span_certificate,
    v0_loss_span_closure_receipt,
    v0_mask_registry,
    v0_schedule_replay_run_receipt,
    v0_schedule_replay_spec_receipt,
    v0_sequential_spec,
    v1_span_budget_fraction,
    workflow_started,
    prune_scope_walk_receipt=None,
    phase_timing=None,
    partition_random_seed=None,
) -> tuple[Any, ...]:
    """Materialize the accepted v0 trajectory and assemble the walk ledger.

    The statements are the original walk-tail block
    of ``run_registered_resnet18_pilot``, unchanged and in the same order. Each
    local the block reads arrives as a keyword argument, and each local the pilot
    reads afterwards is returned, in the order the call site unpacks.

    Two statements are guarded by ``prune_scope_walk_receipt is
    not None`` (a non-default prune scope or proposal grain): the scope's own
    traversal rule and the ``prune_scope_walk`` receipt block.  With the
    default (``None``) neither runs.

    The census receipt's background moment hashes are joined here,
    after the walk and before the receipt is assembled.
    """

    if phase_timing is not None:
        phase_timing.mark("census_moment_hash_join_start")
    _join_pending_census_moment_hashes(hard_partition_measurement)
    if phase_timing is not None:
        phase_timing.mark("census_moment_hash_join_end")

    if v0_schedule_replay_run_receipt is None:
        if phase_timing is not None:
            phase_timing.mark("accepted_trajectory_start")
        accepted_v0_trajectory = materialize_accepted_v0_trajectory(
            runs,
            v0_candidate_registry=v0_candidate_registry,
            v0_mask_registry=v0_mask_registry,
            root_mask_hash=dense_mask.mask_hash,
            terminal_mask_hash=final_mask.mask_hash,
            adapter_id=v0_decision_backend.backend_id,
            query_id=str(query_id),
            hard_mask_to_layers=lambda mask: _layers_from_hard_mask(
                mask,
                runtime=runtime,
            ),
        )
        if phase_timing is not None:
            phase_timing.mark("accepted_trajectory_end")
    # The decision, mask-delta and removal logs of every
    # v6 group are serialized on a writer thread from here on, overlapped
    # with the terminal phase; the receipt writer joins it (``sidecars()``).
    if phase_timing is not None:
        phase_timing.mark("walk_log_setup_start")
    v6_walk_logs = _v6_walk_products(runs)
    if phase_timing is not None:
        phase_timing.mark("walk_log_setup_end")
    traversal_elapsed = time.perf_counter() - traversal_started
    extraction_elapsed = time.perf_counter() - workflow_started
    v0_traversal_complete = len(runs) == sum(
        group.route_kind
        in {"linear_weight", "conv_channel_pair", "conv_kernel_weight"}
        for group in runtime.route_groups
    )
    if prune_scope_walk_receipt is not None:
        # A global walk is complete as ONE run covering every eligible
        # layer; a capped layerwise walk keeps the rule above.  The scope's
        # own check (prune_scope_walk.PruneScopeWalk.traversal_complete)
        # decides.
        v0_traversal_complete = bool(prune_scope_walk_receipt["traversal_complete"])
    v0_scoring_anchor_elapsed = scoring_anchor_elapsed
    v0_traversal_elapsed = traversal_elapsed
    architecture_budget_validated = runtime_architecture_family in {
        "resnet18",
        "alexnet",
        "alexnet_conv_kernel_weight",
    }
    v0_performance_gate = {
        "schema": (
            "resnet18_h100_v0_performance_phase_snapshot_v1"
            if architecture_budget_validated
            else (
                "registered_architecture_h100_v0_performance_" "phase_snapshot_v1"
            )
        ),
        "scoring_and_anchor_seconds": scoring_anchor_elapsed,
        "scoring_and_anchor_budget_seconds": 90.0,
        "complete_traversal_seconds": traversal_elapsed,
        "complete_traversal_budget_seconds": 450.0,
        "total_pre_finalization_seconds": extraction_elapsed,
        "traversal_complete": v0_traversal_complete,
        "scoring_and_anchor_budget_passed": (scoring_anchor_elapsed <= 90.0),
        "traversal_budget_passed": traversal_elapsed <= 450.0,
        "total_pre_finalization_performance_target_seconds": 540.0,
        "total_pre_finalization_performance_target_passed": (
            extraction_elapsed <= 540.0
        ),
        "performance_target_is_stopping_authority": False,
        "authoritative_gate_receipt": "performance_gate.json",
        "status": (
            "pre_audit_phases_passed_pending_receipt_finalization"
            if v0_traversal_complete
            and scoring_anchor_elapsed <= 90.0
            and traversal_elapsed <= 450.0
            and extraction_elapsed <= 540.0
            else "failed"
        ),
    }
    if not architecture_budget_validated:
        v0_performance_gate.update(
            {
                "architecture_family": runtime_architecture_family,
                "status": "architecture_specific_budget_unvalidated",
            }
        )
    v0_decisions = tuple(
        decision for item in runs for decision in item.result.decisions
    )
    v0_normalization_receipts = tuple(
        dict(receipt)
        for item in runs
        for receipt in item.result.ledger_receipt.get(
            "sequential_bce_decision_receipts",
            (),
        )
    )
    accepted_v0_decisions = tuple(
        decision for decision in v0_decisions if decision.action == "accept"
    )
    accepted_removed_connections = int(
        sum(decision.removed_route_count for decision in accepted_v0_decisions)
    )
    accepted_point_loss_span_fraction = float(
        sum(
            decision.mean_sensitivity * float(decision.removed_route_count)
            for decision in accepted_v0_decisions
        )
    )
    accepted_upper_loss_span_fraction = float(
        sum(
            decision.certified_upper_bound * float(decision.removed_route_count)
            for decision in accepted_v0_decisions
        )
    )
    workflow_details = {
        "execution_identity": (
            _V0_CORRECTED_ATTRIBUTION_EXECUTION_IDENTITY
            if v0_attribution_order
            == "corrected_aumann_shapley_global_attribution"
            else _V0_DEFAULT_EXECUTION_IDENTITY
        ),
        "decision_evidence_role": (
            "independent_empirical_iid_calibration_and_proposal_oracle_v1"
        ),
        "layer_count": len(runs),
        "proposal_count": _bce_run_proposal_count(runs),
        "committed_candidate_count": int(len(accepted_v0_decisions)),
        "accepted_trajectory": (
            {
                "schema_name": "accepted_v0_trajectory_unavailable_v1",
                "available": False,
                "reason": (
                    "historical_schedule_replay_has_no_canonical_"
                    "action_candidate_registry"
                ),
                "claim_bearing": False,
                "campaign_launch": False,
                "pruning_commit": False,
                "promotion": False,
            }
            if accepted_v0_trajectory is None
            else {
                **accepted_v0_trajectory.receipt.to_dict(),
                "available": True,
                "entries": tuple(
                    {
                        "trajectory_index": entry.trajectory_index,
                        "run_index": entry.run_index,
                        "proposal_index": entry.proposal_index,
                        "query_id": entry.query_id,
                        "layer_id": entry.layer_id,
                        "partition_id": entry.partition_id,
                        "candidate_id": entry.candidate_id,
                        "parent_mask_hash": entry.parent_mask_hash,
                        "resulting_mask_hash": entry.resulting_mask_hash,
                        "decision_receipt_hash": entry.decision_receipt_hash,
                        "entry_hash": entry.entry_hash,
                    }
                    for entry in accepted_v0_trajectory.entries
                ),
            }
        ),
        "maximum_commits": (
            None if maximum_commits is None else int(maximum_commits)
        ),
        "maximum_commits_is_canary_override": (maximum_commits is not None),
        "sensitivity_tau": resolved_sensitivity_tau,
        "loss_span_certificate_hash": (v0_loss_span_certificate.content_hash),
        "loss_span_training_closure": dict(v0_loss_span_closure_receipt or {}),
        "decision_error_allocation": dict(
            v0_sequential_spec.component_error_allocation
        ),
        "loss_span_estimate": v0_loss_span_certificate.estimate,
        "loss_span_lower": v0_loss_span_certificate.lower,
        "loss_span_upper": v0_loss_span_certificate.upper,
        "decision_summaries": tuple(
            {
                "layer_id": item.signature.layer_id,
                "partition_id": decision.partition_id,
                "action": decision.action,
                "reason": decision.reason,
                "removed_route_count": int(decision.removed_route_count),
                "mean_loss_span_fraction_per_connection": (
                    _finite_receipt_statistic(decision.mean_sensitivity)[0]
                ),
                "mean_statistic_state": _finite_receipt_statistic(
                    decision.mean_sensitivity
                )[1],
                "lower_loss_span_fraction_per_connection": (
                    _finite_receipt_statistic(decision.certified_lower_bound)[0]
                ),
                "lower_bound_state": _finite_receipt_statistic(
                    decision.certified_lower_bound
                )[1],
                "upper_loss_span_fraction_per_connection": (
                    _finite_receipt_statistic(decision.certified_upper_bound)[0]
                ),
                "upper_bound_state": _finite_receipt_statistic(
                    decision.certified_upper_bound
                )[1],
                "sensitivity_tau": float(decision.threshold),
                "panel_images": int(decision.panel_images),
            }
            # Bind the owning layer per decision; the flattened
            # v0_decisions iteration silently reused a leaked loop
            # variable and stamped every summary with a stale layer_id.
            for item in runs
            for decision in item.result.decisions
        ),
        "normalization_audit": {
            "schema": "v0_marginal_normalization_audit_v1",
            "identity": (
                ("raw_per_image_bce_increment_divided_once_by_"
                 if loss_weighting() == "per_image" else "raw_class_balanced_bce_increment_divided_once_by_")
                + "positive_dense_to_empty_loss_span_and_once_by_exact_"
                "ablated_circuit_connections"
            ),
            "raw_risk_convention": (
                "population_share_weighted_per_image_bce" if loss_weighting() == "per_image"
                else "one_half_on_target_plus_one_half_off_target_bce"
            ),
            **({"loss_weighting": "per_image"} if loss_weighting() == "per_image" else {}),
            "fully_pruned_risk": float(
                v0_loss_span_certificate.fully_pruned.estimate
            ),
            "dense_risk": float(v0_loss_span_certificate.dense.estimate),
            "positive_loss_span": float(v0_loss_span_certificate.estimate),
            "accepted_removed_connections": (accepted_removed_connections),
            "accepted_point_loss_span_fraction_telescope": (
                accepted_point_loss_span_fraction
            ),
            "accepted_point_raw_bce_telescope": float(
                accepted_point_loss_span_fraction
                * v0_loss_span_certificate.estimate
            ),
            "accepted_upper_loss_span_fraction_sum": (
                accepted_upper_loss_span_fraction
            ),
            "threshold_times_accepted_connections": float(
                resolved_sensitivity_tau * accepted_removed_connections
            ),
            "circuit_level_simultaneous_risk_guarantee": False,
            "interpretation": (
                "per_decision marginal certificates are not a "
                "simultaneous terminal-circuit risk certificate"
            ),
            "decision_receipts": v0_normalization_receipts,
        },
        "group_stopping_reasons": tuple(
            item.result.stopping_reason for item in runs
        ),
        "performance_gate": v0_performance_gate,
        "group_final_densities": tuple(
            {
                "layer_id": item.signature.layer_id,
                "positive": float(item.result.positive_density),
                "negative": float(item.result.negative_density),
            }
            for item in runs
        ),
        "partition_functional": _partition_functional(family),
        "hard_partition_measurement": hard_partition_measurement,
    }
    census_range_check = _census_certified_range_check(
        hard_partition_measurement, family, partition_random_seed, workflow_details["group_final_densities"],
    )
    if census_range_check is not None:
        workflow_details["census_certified_range_check"] = census_range_check
    if partition_randomization_receipt is not None:
        workflow_details["partition_randomization"] = (
            partition_randomization_receipt
        )
    if v0_attribution_order != "plain_global_attribution":
        workflow_details["v0_attribution_order"] = {
            "mode": v0_attribution_order,
            "selection_effect": "route_activity_order_only",
            "local_sign_partition_labels_unchanged": True,
            "calibration_oracle_walker_semantics_unchanged": True,
            "hard_partition_measurement_receipt_hash": content_hash(
                hard_partition_measurement,
                prefix="v0-corrected-order-hard-partition-measurement-",
            ),
        }
    if v6_walk_logs is not None:
        workflow_details["v6_walk_logs"] = v6_walk_logs
        workflow_details["normalization_audit"]["decision_record_log"] = (
            "v6_walk_logs"
        )
    if direct_candidate_comparison:
        direct_threshold = float(
            v1_span_budget_fraction * v0_loss_span_certificate.estimate
        )
        workflow_details.update(
            {
                "execution_identity": (
                    "modern_v1_direct_candidate_increment_" "comparison_adapter_v1"
                ),
                "comparison_validation_only": True,
                "public_irg_v1_selector_enabled": False,
                "decision_estimand": "direct_candidate_bce_increment",
                "per_step_target_bce_increment": direct_threshold,
                "per_step_target_loss_span_fraction": float(
                    v1_span_budget_fraction
                ),
                "cross_step_alpha_spending": False,
                "cross_step_evidence_borrowing": False,
                "parent_is_statistical_arm": False,
                "normalization_audit": {
                    "schema": "modern_v1_direct_candidate_audit_v1",
                    "identity": (
                        "raw_paired_candidate_minus_current_bce_increment_"
                        "compared_directly_to_a_fixed_per_step_target"
                    ),
                    "loss_span_used_only_to_compile_target": True,
                    "positive_loss_span": float(v0_loss_span_certificate.estimate),
                    "target_loss_span_fraction": float(v1_span_budget_fraction),
                    "target_raw_bce_increment": direct_threshold,
                    "cumulative_endpoint_debit": False,
                    "trajectory_confidence_guarantee": False,
                },
                "decision_summaries": tuple(
                    {
                        "layer_id": item.signature.layer_id,
                        "partition_id": decision.partition_id,
                        "action": decision.action,
                        "reason": decision.reason,
                        "removed_route_count": int(decision.removed_route_count),
                        "mean_direct_candidate_bce_increment": (
                            _finite_receipt_statistic(decision.mean_sensitivity)[0]
                        ),
                        "lower_direct_candidate_bce_increment": (
                            _finite_receipt_statistic(
                                decision.certified_lower_bound
                            )[0]
                        ),
                        "upper_direct_candidate_bce_increment": (
                            _finite_receipt_statistic(
                                decision.certified_upper_bound
                            )[0]
                        ),
                        "per_step_target_bce_increment": direct_threshold,
                        "panel_images": int(decision.panel_images),
                    }
                    for item in runs
                    for decision in item.result.decisions
                ),
            }
        )
    if v0_schedule_replay_run_receipt is not None:
        # Gate-off receipts keep their exact bytes: replay fields exist
        # only when the walker actually replaced the tau traversal.
        workflow_details["execution_identity"] = (
            "irg_v0_historical_schedule_replay_walker_v1"
        )
        workflow_details["v0_schedule_replay"] = {
            "spec": dict(v0_schedule_replay_spec_receipt or {}),
            **dict(v0_schedule_replay_run_receipt),
        }
    if prune_scope_walk_receipt is not None:
        # Present only under a non-default prune scope or grain, so
        # every layerwise receipt keeps its exact key set.
        workflow_details["prune_scope_walk"] = dict(prune_scope_walk_receipt)
    _progress("v0_layerwise_pruning_completed")
    return (
        accepted_v0_trajectory,
        extraction_elapsed,
        v0_scoring_anchor_elapsed,
        v0_traversal_complete,
        v0_traversal_elapsed,
        workflow_details,
    )



def _target_logit_reporting_view(provider, masks, training_context_ids):
    """Fit the canonical balanced target-logit readout on shared training rows."""
    import copy

    view = copy.copy(provider)
    view.pruning_functional = "target_logit"
    view.loss_weighting = "balanced"
    view._stratum_weights = {"on_target": 0.5, "off_target": 0.5}
    view.calibration_options = dict(provider.calibration_options,
                                    binary_loss_weighting="design_weighted")
    view._calibration_fits = {}
    view._calibration_fold_fits = {}
    view._firth_calibration_diagnostics = {}
    for mask in masks:
        if mask is not None and mask.mask_hash not in view._calibration_fits:
            fit = provider.fit_target_logit_reporting_calibration(
                mask, training_context_ids=training_context_ids)
            view._calibration_fits[mask.mask_hash] = fit
            view._firth_calibration_diagnostics[mask.mask_hash] = dict(
                getattr(provider, "_target_logit_reporting_calibration_diagnostics", {}).get(mask.mask_hash, {}))
    return view


def _canonical_target_logit_evaluation_weights(populations):
    off = tuple(name for name in populations if name != "on_target")
    if "on_target" not in populations or not off:
        raise ValueError("canonical target-logit reporting requires both classes")
    return {"on_target": 0.5, **{name: 0.5 / len(off) for name in off}}


def _terminal_fit_block(
    *,
    dense_calibration_closure_receipt,
    dense_mask,
    evaluation_axis,
    final_mask,
    gpu_activity_sampler,
    historical_mask,
    optimization_axis,
    runtime,
    sample_size_policy,
    shared_run_id,
    step_category_setup_receipt,
    target_class_id,
    terminal_category_matrix,
    terminal_maximum_per_stratum,
    terminal_namespace_salt,
    v0_decision_backend,
    phase_timing=None,
    walk_workflow_details=None,
) -> tuple[Any, ...]:
    """Close the terminal calibrations, fit the targets and freeze the banks.

    The statements are the original terminal-fit block
    of ``run_registered_resnet18_pilot`` (closure, substituted target fits,
    sealed populations, category-bank freeze), in the same order. Only the
    locked final/historical mask closure selects the terminal tolerance rule.
    Each local the block reads arrives as a keyword argument, and each local the
    pilot reads afterwards is returned, in the order the call site unpacks.
    """

    from mechanistic_systems.measurement.run_products import pruning_functional

    decision_provider = v0_decision_backend.provider
    fit_ids_before_terminal = {
        mask_hash: fit.calibration_state_id
        for mask_hash, fit in decision_provider.calibration_fits.items()
    }
    for trajectory_mask in (dense_mask, final_mask):
        if trajectory_mask.mask_hash not in fit_ids_before_terminal:
            raise RuntimeError(
                "terminal validation refuses to refit a trajectory state"
            )
    terminal_risk_context_count = max(
        int(sample_size_policy.oracle_initial_packet),
        2 * int(terminal_maximum_per_stratum),
        2 * int(decision_provider.oracle_minimum_per_stratum),
    )
    terminal_calibration_closure_receipts: dict[str, Mapping[str, Any]] = {}
    terminal_closure_kwargs: dict[str, Any] = {}
    terminal_rule = _terminal_calibration_settings()[0]
    if terminal_rule == "walk_resolution":
        if walk_workflow_details is None:
            raise ValueError("walk_resolution requires the completed walk's decision summaries")
        final_walk_span = getattr(v0_decision_backend, "current_loss_span_certificate", None)
        terminal_closure_kwargs["walk_resolution_context"] = {
            **_walk_resolution_calibration_tolerance(
                decision_summaries=walk_workflow_details["decision_summaries"],
                sensitivity_tau=walk_workflow_details["sensitivity_tau"],
                positive_loss_span_estimate=(
                    final_walk_span.estimate if final_walk_span is not None
                    else walk_workflow_details["loss_span_estimate"]
                ),
            ),
            "positive_loss_span_estimate_source": (
                "walk_backend_current_loss_span_at_mask_lock"
                if final_walk_span is not None else "workflow_loss_span_estimate"
            ),
            "positive_loss_span_certificate_hash": (
                final_walk_span.content_hash if final_walk_span is not None else None
            ),
        }
    # Dense was closed before scoring/pruning for v0's loss-span normalization.
    # If the new walk rule ends at dense (no accepted step), continue that
    # final-mask fit only now, after mask lock, and retain its prewalk closure.
    if dense_calibration_closure_receipt is None:
        raise RuntimeError("dense calibration lacks its pre-pruning closure")
    terminal_calibration_closure_receipts[dense_mask.mask_hash] = (
        dense_calibration_closure_receipt
    )
    if final_mask.mask_hash != dense_mask.mask_hash or terminal_rule == "walk_resolution":
        if phase_timing is not None:
            phase_timing.mark("terminal_final_calibration_start")
        final_closure_kwargs = dict(terminal_closure_kwargs)
        if final_mask.mask_hash == dense_mask.mask_hash:
            final_closure_kwargs["walk_resolution_context"] = {
                **dict(terminal_closure_kwargs["walk_resolution_context"]),
                "prewalk_dense_calibration_closure": dict(dense_calibration_closure_receipt),
            }
        _, final_calibration_closure = (
            _close_terminal_mask_calibration(
                provider=decision_provider,
                mask=final_mask,
                confidence_level=float(sample_size_policy.confidence_level),
                cumulative_risk_context_count=terminal_risk_context_count,
                certificate_role=(
                    "terminal_final_mask_training_risk_resolution_reference"
                ),
                stream_namespace="terminal-final-calibration-resolution",
                progress_callback=_progress,
                **final_closure_kwargs,
            )
        )
        terminal_calibration_closure_receipts[final_mask.mask_hash] = (
            final_calibration_closure
        )
        if phase_timing is not None:
            phase_timing.mark("terminal_final_calibration_end")
    if (
        historical_mask is not None
        and historical_mask.mask_hash not in terminal_calibration_closure_receipts
    ):
        if phase_timing is not None:
            phase_timing.mark("terminal_historical_calibration_start")
        _, historical_calibration_closure = (
            _close_terminal_mask_calibration(
                provider=decision_provider,
                mask=historical_mask,
                confidence_level=float(sample_size_policy.confidence_level),
                cumulative_risk_context_count=terminal_risk_context_count,
                certificate_role=(
                    "historical_comparator_training_risk_resolution_reference"
                ),
                stream_namespace=("terminal-historical-calibration-resolution"),
                progress_callback=_progress,
                **terminal_closure_kwargs,
            )
        )
        terminal_calibration_closure_receipts[historical_mask.mask_hash] = (
            historical_calibration_closure
        )
        if phase_timing is not None:
            phase_timing.mark("terminal_historical_calibration_end")

    # Always-on scalar terminal enrichment.  Fit exactly one substituted
    # target affine per endpoint on the same frozen training membership before
    # validation is materialized.  This is independent of the optional
    # 1,000x1,000 category-matrix leg and therefore remains available when
    # both category-matrix flags are off.
    terminal_ordinary_fits = decision_provider.calibration_fits
    terminal_target_training_fit = terminal_ordinary_fits[final_mask.mask_hash]
    terminal_target_training_context_ids = tuple(
        terminal_target_training_fit.context_ids
    )
    if not terminal_target_training_context_ids:
        calibration_orders = decision_provider.calibration_context_ids_by_stratum
        sampled_counts = dict(terminal_target_training_fit.sampled_stratum_counts)
        terminal_target_training_context_ids = tuple(
            context_id
            for stratum, contexts in calibration_orders.items()
            for context_id in contexts[: max(1, int(sampled_counts.get(stratum, 256)))]
        )
    if phase_timing is not None:
        phase_timing.mark("terminal_substituted_dense_fit_start")
    target_substituted_dense_fit = (
        decision_provider.fit_frozen_target_substituted_calibration(
            dense_mask,
            dense_mask,
            training_context_ids=terminal_target_training_context_ids,
            training_batch_factory=decision_provider.calibration_batch_factory,
            training_axis_source_identity=optimization_axis.content_hash,
        )
    )
    if phase_timing is not None:
        phase_timing.mark("terminal_substituted_dense_fit_end")
    target_substituted_final_fit = target_substituted_dense_fit
    if final_mask.mask_hash != dense_mask.mask_hash:
        if phase_timing is not None:
            phase_timing.mark("terminal_substituted_final_fit_start")
        target_substituted_final_fit = (
            decision_provider.fit_frozen_target_substituted_calibration(
                final_mask,
                dense_mask,
                training_context_ids=terminal_target_training_context_ids,
                training_batch_factory=(decision_provider.calibration_batch_factory),
                training_axis_source_identity=optimization_axis.content_hash,
            )
        )
        if phase_timing is not None:
            phase_timing.mark("terminal_substituted_final_fit_end")
    target_substituted_historical_fit = None
    if historical_mask is not None:
        if phase_timing is not None:
            phase_timing.mark("terminal_substituted_historical_fit_start")
        target_substituted_historical_fit = (
            decision_provider.fit_frozen_target_substituted_calibration(
                historical_mask,
                dense_mask,
                training_context_ids=terminal_target_training_context_ids,
                training_batch_factory=(decision_provider.calibration_batch_factory),
                training_axis_source_identity=optimization_axis.content_hash,
            )
        )
        if phase_timing is not None:
            phase_timing.mark("terminal_substituted_historical_fit_end")

    if pruning_functional() == "substituted_softmax":
        decision_provider._terminal_target_logit_reporting_provider = _target_logit_reporting_view(
            decision_provider, (dense_mask, final_mask, historical_mask),
            terminal_target_training_context_ids)

    fit_ids_at_terminal_lock = {
        mask_hash: fit.calibration_state_id
        for mask_hash, fit in decision_provider.calibration_fits.items()
    }
    # Freeze the complete 1,000-category dense/final banks on the optimization
    # axis before even deriving the validation population.  This ordering is a
    # scientific boundary, not a timing preference: neither validation labels
    # nor validation response values may influence affine selection or refit.
    terminal_category_frozen_banks = None
    if terminal_category_matrix:
        if phase_timing is not None:
            phase_timing.mark("terminal_bank_populations_start")
        (
            category_calibration_populations,
            _category_calibration_weights,
        ) = _sealed_terminal_populations(
            optimization_axis,
            target_class_id=target_class_id,
            namespace=(
                "architecture-dense-category-calibration:"
                f"{runtime.loaded.receipt.state_dict_hash}:"
                f"{optimization_axis.content_hash}"
            ),
            maximum_per_stratum=terminal_maximum_per_stratum,
        )
        if phase_timing is not None:
            phase_timing.mark("terminal_bank_populations_end")
            phase_timing.mark("terminal_bank_freeze_start")
        terminal_category_frozen_banks = _freeze_terminal_category_banks(
            enabled=True,
            provider=decision_provider,
            dense_mask=dense_mask,
            final_mask=final_mask,
            calibration_populations=category_calibration_populations,
            target_class_id=target_class_id,
            calibration_batch_factory=(
                runtime.optimization_data_plane.role_batch_factory(
                    "category_calibration"
                )
            ),
            calibration_context_axis_hash=optimization_axis.content_hash,
            pre_frozen_dense_fit=(
                None
                if step_category_setup_receipt is None
                else decision_provider.frozen_category_calibration_fit_bundle_by_request(
                    str(step_category_setup_receipt["dense_fit_request_hash"])
                )
            ),
        )
        if phase_timing is not None:
            phase_timing.mark("terminal_bank_freeze_end")
    # Validation is materialized only after every extraction calibration state
    # is locked. It never supplies proposal, calibration, attribution, trust,
    # or candidate-selection evidence. A declared salt rotates the sealed
    # terminal panel identity so cumulative exposure from earlier import-only
    # readouts under the shared run identity never accumulates on a fresh run.
    validation_populations, validation_stratum_weights = _sealed_terminal_populations(
        evaluation_axis,
        target_class_id=target_class_id,
        namespace=(
            shared_run_id
            if terminal_namespace_salt is None
            else (f"{shared_run_id}:terminal-rotation:" f"{terminal_namespace_salt}")
        ),
        maximum_per_stratum=terminal_maximum_per_stratum,
    )
    validation_batch_factory = runtime.evaluation_data_plane.role_batch_factory(
        "terminal"
    )
    validation_mask_lookup: dict[str, HardMask] = {
        dense_mask.mask_hash: dense_mask,
        final_mask.mask_hash: final_mask,
    }
    if historical_mask is not None:
        validation_mask_lookup[historical_mask.mask_hash] = historical_mask
    terminal_provider = decision_provider
    extraction_gpu_activity_receipt = gpu_activity_sampler.stop()
    terminal_audit_started = time.perf_counter()
    return (
        decision_provider,
        extraction_gpu_activity_receipt,
        fit_ids_at_terminal_lock,
        fit_ids_before_terminal,
        target_substituted_dense_fit,
        target_substituted_final_fit,
        target_substituted_historical_fit,
        terminal_audit_started,
        terminal_calibration_closure_receipts,
        terminal_category_frozen_banks,
        terminal_provider,
        validation_batch_factory,
        validation_populations,
        validation_stratum_weights,
    )


def _category_dense_terminal_evaluation(
    *, provider, dense_mask, populations, weights, validation_axis_hash,
    calibration_state, kind, compute,
):
    """Reuse completed dense metrics while preserving their exact mapping."""

    from . import category_memo

    memo = category_memo.active()
    if memo is None:
        return compute()
    dense_state_digest = content_hash({
        "model_state_hash": provider._frozen_category_model_state_hash(),
        "dense_mask_hash": dense_mask.mask_hash,
        "target_class_id": int(provider.target_class_id),
        "calibration_state_id": calibration_state.calibration_state_id,
        "calibration_fit_receipt_hash": calibration_state.fit_receipt_hash,
        "calibration_options": dict(provider.calibration_options),
        **({"terminal_statistics_schema": "terminal_statistics_v1"}
           if getattr(provider, "loss_weighting", "balanced") == "per_image"
           or provider.calibration_options.get("calibration_penalty") == "firth" else {}),
        "fold_state_ids": tuple(
            fit.calibration_state_id for fit in provider._calibration_fold_fits.get(dense_mask.mask_hash, ())
        ),
        "runtime_execution_profile": asdict(provider.runtime_execution_profile),
    })
    panel_digest = content_hash({
        "validation_axis_hash": str(validation_axis_hash),
        "populations": {str(key): tuple(rows) for key, rows in populations.items()},
        "weights": dict(weights),
    })
    slot = memo.dense_panels.setdefault(memo.key(dense_state_digest, panel_digest), {})
    product_name = "terminal_metrics:" + str(kind)
    if product_name in slot:
        # The existing counter records one logical terminal evaluation even
        # when its immutable dense product was computed by the earlier arm.
        provider._forward_passes += 1
        memo.note(product=product_name, source="memo")
        return category_memo.container_copy(slot[product_name])
    result = compute()
    slot[product_name] = category_memo.freeze_arrays(category_memo.container_copy(result))
    memo.note(product=product_name, source="computed")
    return result


def _terminal_substituted_evaluation_weights(axis, populations, target_class_id):
    """Weight capped validation class panels by their full population shares."""
    counts = {str(key): 0 for key in populations}
    for record in axis.records:
        label = int(record.label)
        key = "on_target" if label == int(target_class_id) else "off_target:{}".format(label)
        if key not in counts:
            raise ValueError("substituted validation panel omits a population class")
        counts[key] += 1
    total = sum(counts.values())
    if not total or any(count <= 0 for count in counts.values()):
        raise ValueError("substituted validation requires positive full-population class counts")
    return {key: count / total for key, count in counts.items()}


def _terminal_evaluation_block(
    *,
    dense_mask,
    dense_readout_table_cache_dir,
    evaluation_axis,
    final_mask,
    historical_mask,
    runtime,
    target_class_id,
    target_substituted_dense_fit,
    target_substituted_final_fit,
    target_substituted_historical_fit,
    terminal_audit_started,
    terminal_category_frozen_banks,
    terminal_category_matrix,
    terminal_provider,
    validation_batch_factory,
    validation_populations,
    validation_stratum_weights,
    phase_timing=None,
) -> tuple[Any, ...]:
    """Evaluate the frozen masks on validation and capture the category structure.

    The statements are the original terminal-evaluation
    block of ``run_registered_resnet18_pilot``, unchanged and in the same order.
    Each local the block reads arrives as a keyword argument, and each local the
    pilot reads afterwards is returned, in the order the call site unpacks.
    """

    from mechanistic_systems.measurement.run_products import pruning_functional

    _progress("terminal_evaluation_started")
    terminal_populations = validation_populations
    terminal_batch_factory = validation_batch_factory
    ordinary_provider = getattr(terminal_provider, "_terminal_target_logit_reporting_provider", terminal_provider)
    ordinary_weights = (validation_stratum_weights if ordinary_provider is terminal_provider
                        else _canonical_target_logit_evaluation_weights(terminal_populations))
    if phase_timing is not None:
        phase_timing.mark("terminal_dense_start")
    dense_terminal = _category_dense_terminal_evaluation(
        provider=ordinary_provider, dense_mask=dense_mask,
        populations=terminal_populations, weights=ordinary_weights,
        validation_axis_hash=evaluation_axis.content_hash,
        calibration_state=ordinary_provider.calibration_fits[dense_mask.mask_hash],
        kind="ordinary_bce",
        compute=lambda: ordinary_provider.evaluate_frozen_mask_bce(
            dense_mask,
            evaluation_context_ids_by_stratum=terminal_populations,
            evaluation_stratum_weights=ordinary_weights,
            batch_factory=terminal_batch_factory,
            require_precalibrated=True,
        ),
    )
    if phase_timing is not None:
        phase_timing.mark("terminal_dense_end")
        phase_timing.mark("terminal_final_start")
    final_terminal = ordinary_provider.evaluate_frozen_mask_bce(
        final_mask,
        evaluation_context_ids_by_stratum=terminal_populations,
        evaluation_stratum_weights=ordinary_weights,
        batch_factory=terminal_batch_factory,
        require_precalibrated=True,
    )
    if phase_timing is not None:
        phase_timing.mark("terminal_final_end")
    if phase_timing is not None and historical_mask is not None:
        phase_timing.mark("terminal_historical_start")
    historical_terminal = (
        None
        if historical_mask is None
        else ordinary_provider.evaluate_frozen_mask_bce(
            historical_mask,
            evaluation_context_ids_by_stratum=terminal_populations,
            evaluation_stratum_weights=ordinary_weights,
            batch_factory=terminal_batch_factory,
            require_precalibrated=True,
        )
    )
    if phase_timing is not None and historical_mask is not None:
        phase_timing.mark("terminal_historical_end")
    # The substituted metrics below and
    # the category captures after this window read the same frozen model on
    # the same validation rows; one (final, dense) forward serves all of them.
    # The provider refuses to prime while any calibration is still open.
    terminal_validation_one_pass = _prime_terminal_validation_one_pass(
        terminal_provider=terminal_provider,
        final_mask=final_mask,
        dense_mask=dense_mask,
        validation_populations=validation_populations,
        target_class_id=target_class_id,
        batch_factory=terminal_batch_factory,
        validation_axis_hash=evaluation_axis.content_hash,
    )
    substituted_stratum_weights = validation_stratum_weights
    if getattr(terminal_provider, "substituted_loss_weighting", "follow") == "per_image":
        substituted_stratum_weights = _terminal_substituted_evaluation_weights(
            evaluation_axis, terminal_populations, target_class_id)
    dense_terminal_substituted = _category_dense_terminal_evaluation(
        provider=terminal_provider, dense_mask=dense_mask,
        populations=terminal_populations, weights=substituted_stratum_weights,
        validation_axis_hash=evaluation_axis.content_hash,
        calibration_state=target_substituted_dense_fit, kind="substituted_bce",
        compute=lambda: terminal_provider.evaluate_frozen_mask_substituted_bce(
            dense_mask,
            dense_mask,
            fit=target_substituted_dense_fit,
            evaluation_context_ids_by_stratum=terminal_populations,
            evaluation_stratum_weights=substituted_stratum_weights,
            batch_factory=terminal_batch_factory,
        ),
    )
    final_terminal_substituted = terminal_provider.evaluate_frozen_mask_substituted_bce(
        final_mask,
        dense_mask,
        fit=target_substituted_final_fit,
        evaluation_context_ids_by_stratum=terminal_populations,
        evaluation_stratum_weights=substituted_stratum_weights,
        batch_factory=terminal_batch_factory,
    )
    historical_terminal_substituted = (
        None
        if historical_mask is None
        else terminal_provider.evaluate_frozen_mask_substituted_bce(
            historical_mask,
            dense_mask,
            fit=target_substituted_historical_fit,
            evaluation_context_ids_by_stratum=terminal_populations,
            evaluation_stratum_weights=substituted_stratum_weights,
            batch_factory=terminal_batch_factory,
        )
    )
    final_terminal_substituted = substituted_loss_window_statistics(
        final_terminal_substituted, dense_terminal_substituted)
    if historical_terminal_substituted is not None:
        historical_terminal_substituted = substituted_loss_window_statistics(
            historical_terminal_substituted, dense_terminal_substituted)
    dense_terminal_substituted = substituted_loss_window_statistics(
        dense_terminal_substituted, dense_terminal_substituted)
    if pruning_functional() == "substituted_softmax":
        for metrics in (dense_terminal, final_terminal, historical_terminal):
            if metrics is not None:
                bce = float(metrics["class_balanced_binary_bce"])
                dense_bce = float(dense_terminal["class_balanced_binary_bce"])
                denominator = math.log(2.0) - dense_bce
                metrics.update(pruning_functional="substituted_softmax",
                    reporting_functional="target_logit", loss_weighting="balanced",
                    class_balanced_bce_share_vs_log2=bce / math.log(2.0),
                    class_balanced_loss_window_share=(
                        (bce - dense_bce) / denominator if denominator > 0.0 else None))
        for metrics in (dense_terminal_substituted, final_terminal_substituted,
                        historical_terminal_substituted):
            if metrics is not None:
                metrics.update(pruning_functional="substituted_softmax",
                               reporting_functional="substituted_softmax", loss_weighting="per_image")
    _progress("terminal_evaluation_completed")
    terminal_audit_elapsed = time.perf_counter() - terminal_audit_started
    # Capture the terminal category-structure matrices OUTSIDE the
    # registered terminal-audit timing window so the meaning of
    # ``terminal_validation_audit_seconds`` is unchanged.  The leg is purely
    # additive: it uses its own full-readout forwards, never mutates the
    # provider's registered fits or counters, and emits one new receipt
    # block plus two sidecar files next to receipt.json.
    (
        terminal_category_structure_block,
        terminal_category_structure_sidecars,
    ) = _terminal_category_structure_leg(
        enabled=bool(terminal_category_matrix),
        phase_timing=phase_timing,
        provider=terminal_provider,
        dense_mask=dense_mask,
        final_mask=final_mask,
        validation_populations=validation_populations,
        target_class_id=target_class_id,
        batch_factory=terminal_batch_factory,
        model_state_hash=runtime.loaded.receipt.state_dict_hash,
        context_axis_hash=evaluation_axis.content_hash,
        forward_tier=int(runtime.evaluator.runtime_execution_profile.forward_tier),
        cache_dir=(
            None
            if dense_readout_table_cache_dir is None
            else Path(dense_readout_table_cache_dir) / "shared_category_structure"
        ),
        progress_callback=_progress,
        frozen_banks=terminal_category_frozen_banks,
        dense_frozen_final=True,
        _phase_defer_output=True,
    )
    # Save the frozen banks with the run.
    terminal_provider._phase_pending_terminal_banks = terminal_category_frozen_banks
    terminal_validation_one_pass = _release_terminal_validation_one_pass(
        terminal_provider=terminal_provider,
        one_pass=terminal_validation_one_pass,
    )
    return (
        dense_terminal,
        dense_terminal_substituted,
        final_terminal,
        final_terminal_substituted,
        historical_terminal,
        historical_terminal_substituted,
        terminal_audit_elapsed,
        terminal_category_structure_block,
        terminal_category_structure_sidecars,
        terminal_validation_one_pass,
    )


def _pilot_receipt_hash_with_phase_timing(compact, phase_timing) -> str:
    if phase_timing is not None and phase_timing.enabled:
        phase_timing.mark("receipt_write")
        compact["phase_timing"] = phase_timing.receipt_block()
    return runtime_receipt_hash(compact, prefix="irg-resnet18-pilot-")


def _runtime_data_plane_effective_settings(runtime: Any) -> dict[str, dict[str, str]]:
    """Read bound delivery configuration without opening or serving its rows."""
    settings = {}
    for role in ("optimization", "evaluation"):
        plane = getattr(runtime, f"{role}_data_plane", None)
        disclose = getattr(plane, "_effective_delivery_settings", None)
        if callable(disclose):
            settings[role] = dict(disclose())
    return settings


def _run_effective_settings(runtime: Any) -> dict[str, str]:
    """Disclose the backend choice and active bound delivery components.

    The run-wide image setting records whether either bound plane uses device
    gather; each plane's own values remain in data_plane_effective_settings.
    Pinned pools share a process limit, whose largest bound value is recorded.
    """
    planes = tuple(_runtime_data_plane_effective_settings(runtime).values())
    resolved = {}
    if planes:
        resolved["IRG_DEVICE_GATHER"] = str(int(any(
            values.get("IRG_DEVICE_GATHER") == "1" for values in planes)))
        resolved["IRG_PINNED_POOL_MAX_BYTES"] = str(max(
            int(values["IRG_PINNED_POOL_MAX_BYTES"]) for values in planes))
    evaluator = getattr(runtime, "evaluator", None)
    backend = getattr(evaluator, "candidate_execution_backend", None)
    if backend is not None:
        resolved["CIRCUIT_FORWARD_PREFIX_REUSE"] = str(int(bool(backend.forward_prefix_reuse)))
    terminal_rule, terminal_tolerance = _terminal_calibration_settings()
    resolved["IRG_TERMINAL_CALIBRATION_RULE"] = terminal_rule
    if terminal_tolerance is not None:
        resolved["IRG_TERMINAL_CALIBRATION_TOLERANCE"] = str(terminal_tolerance)
    return effective_settings(resolved=resolved)


def _terminal_substituted_metric_delta(metrics, dense_metrics, metric):
    """Compare the substituted fit's own estimand when both fits are ready."""
    weighting = metrics.get("loss_weighting", loss_weighting())
    dense_weighting = dense_metrics.get("loss_weighting", weighting)
    key = ("per_image_binary_" if weighting == "per_image"
           else "class_balanced_binary_") + metric
    dense_key = ("per_image_binary_" if dense_weighting == "per_image"
                 else "class_balanced_binary_") + metric
    value, dense_value = metrics.get(key), dense_metrics.get(dense_key)
    if value is None or dense_value is None:
        return None
    return float(value - dense_value)


def _write_run_receipt(
    *,
    calibration_deployment_refit_anchor,
    canonical_dense_baseline_bce,
    canonical_dense_certificate,
    canonical_dense_table_active,
    canonical_dense_table_receipt_path,
    cold_runtime_setup_started,
    comparison_validation_mode,
    decision_oracle_sampling,
    decision_oracle_sampling_receipts,
    decision_provider,
    decision_sampling,
    dense_baseline_mode,
    dense_mask,
    dense_readout_table_telemetry,
    dense_terminal,
    dense_terminal_substituted,
    destination,
    evaluation_axis,
    extraction_gpu_activity_receipt,
    family,
    final_mask,
    final_terminal,
    final_terminal_substituted,
    fit_ids_at_terminal_lock,
    fit_ids_before_terminal,
    historical_mask,
    historical_mask_alias_receipt,
    historical_mask_sha256,
    historical_terminal,
    historical_terminal_substituted,
    maximum_relative_bce_increase,
    model_transform,
    optimization_axis,
    partition_random_seed,
    population_calibration_receipt_path,
    profiler,
    query_id,
    requested_family,
    residual_observation_path_counts,
    resolved_sensitivity_tau,
    retained,
    route_axis_hash,
    route_count,
    runner_prewarm_receipt,
    runtime,
    sample_size_policy,
    step_category_structure_block,
    step_category_structure_sidecars,
    step_context_delta_block,
    step_context_delta_sidecars,
    step_observables_effective,
    target_class_id,
    terminal_audit_elapsed,
    terminal_calibration_closure_receipts,
    terminal_category_structure_block,
    terminal_category_structure_sidecars,
    terminal_namespace_salt,
    terminal_provider,
    terminal_validation_one_pass,
    unit_gate_started,
    v0_accept_replication_gate,
    v0_calibration_decision_estimand,
    v0_oracle_minimum_per_stratum,
    version,
    workflow_details,
    before_receipt=None,
    output_submit=None,
) -> dict[str, Any]:
    """Assemble, hash and write the pilot receipt and its sidecars.

    The statements are the original receipt block of
    ``run_registered_resnet18_pilot``, unchanged and in the same order. Each local
    the block reads arrives as a keyword argument; the pilot reads only the
    payload afterwards.

    The receipt
    is the compact layout of ``run_products``: every product is written on a
    helper thread with its sha256 listed in the receipt, the retained execution
    rows move to a trace sidecar, and ``receipt.json`` is written last.
    """

    # Product sources that serialize on their own thread
    # (the walk logs in ``workflow_details``) are joined before anything
    # is hashed or serialized, because joining completes their descriptor.
    from mechanistic_systems.measurement.run_products import pruning_functional

    from mechanistic_systems.measurement.phase_device_orchestration import resolve_products, state_for
    terminal_category_structure_block = resolve_products(terminal_category_structure_block)
    terminal_category_structure_sidecars = resolve_products(terminal_category_structure_sidecars)
    for write in getattr(terminal_provider, "_phase_pending_cache_writes", ()):
        write()
    frozen = getattr(terminal_provider, "_phase_pending_terminal_banks", None)
    if frozen is not None:
        terminal_category_structure_block, terminal_category_structure_sidecars = _with_terminal_frozen_banks(
            terminal_category_structure_block, terminal_category_structure_sidecars, frozen)
    if terminal_provider is not None:
        terminal_provider._phase_pending_cache_writes = []
        terminal_provider._phase_pending_terminal_banks = None
        state_for(terminal_provider).calls["terminal_products_bulk_completed"] += 1
    phase_timing = getattr(profiler, "phase_timing", None)
    run_product_writer = getattr(profiler, "run_product_writer", None)
    run_product_sidecars = _collect_run_product_sidecars(workflow_details)
    for plane in (runtime.optimization_data_plane, runtime.evaluation_data_plane):
        if hasattr(plane, "settle_image_feed"):
            plane.settle_image_feed()
    decision_provider_usage = dict(decision_provider.resource_usage)
    optimization_data_plane_receipt = runtime.optimization_data_plane.runtime_receipt()
    evaluation_data_plane_receipt = runtime.evaluation_data_plane.runtime_receipt()
    runtime_execution_profile_receipt = (
        runtime.evaluator.runtime_execution_profile.to_runtime_dict()
    )
    unified_execution_receipt = _unified_native_execution_receipt(
        version=version,
        optimization_data_plane=optimization_data_plane_receipt,
        evaluation_data_plane=evaluation_data_plane_receipt,
        provider_resource_usage=decision_provider_usage,
        gpu_activity=extraction_gpu_activity_receipt,
        runtime_execution_profile=runtime_execution_profile_receipt,
    )
    phase_device_receipt = {}
    from mechanistic_systems.measurement.phase_device_orchestration import state_for
    phase_device_receipt["phase_device_orchestration"] = state_for(decision_provider).receipt()
    payload = {
        "schema_name": PILOT_SCHEMA,
        "effective_settings": _run_effective_settings(runtime),
        "data_plane_effective_settings": _runtime_data_plane_effective_settings(runtime),
        **phase_device_receipt,
        "status": "exploratory_non_claim_pilot",
        "version_name": version,
        "comparison_validation_mode": comparison_validation_mode,
        # What code produced this receipt. See _sealed_source_identity: the
        # sbatch verifies HEAD == SOURCE_COMMIT and then discarded the fact,
        # so f12's three seals were indistinguishable in their results.
        "sealed_source_identity": _sealed_source_identity(),
        "partition_family": family,
        # Which layer order this arm actually walked. The field is emitted
        # always so arms of either order are distinguishable from the
        # receipt alone.
        "layer_execution_order": _resolved_layer_execution_order(),
        "requested_partition_family": requested_family,
        "partition_random_seed": partition_random_seed,
        "query_id": str(query_id),
        "target_class_id": target_class_id,
        "model_id": runtime.loaded.receipt.model_id,
        "model_state_hash": runtime.loaded.receipt.state_dict_hash,
        # Present only when a registered transform is set.
        **(
            {}
            if model_transform is None
            else {"model_transform": model_transform.receipt_block()}
        ),
        # Present only under a strict-float32 readout setting.
        **_terminal_frozen_category_readout_numerics_block(
            terminal_provider,
            _terminal_frozen_category_readout_numerics(terminal_provider),
        ),
        "route_axis_hash": route_axis_hash,
        "route_count": route_count,
        "retained_route_count": retained,
        "removed_route_count": route_count - retained,
        "retained_fraction": float(retained / route_count),
        "sparsity": float(1.0 - retained / route_count),
        "canonical_dense_baseline_bce": canonical_dense_baseline_bce,
        "dense_baseline_authority": {
            "schema": "v0_dense_baseline_authority_v1",
            "mode": dense_baseline_mode,
            "training_population_exhaustive": bool(
                dense_baseline_mode == DENSE_BASELINE_POPULATION_EXACT
            ),
            "dense_forward_scope": (
                "complete_sealed_training_population"
                if dense_baseline_mode == DENSE_BASELINE_POPULATION_EXACT
                else "sampled_training_exploratory"
            ),
            "attribution_scope_unchanged": True,
            "canonical_table_receipt_path": (
                None
                if canonical_dense_table_receipt_path is None
                else str(canonical_dense_table_receipt_path)
            ),
            "population_calibration_receipt_path": (
                None
                if population_calibration_receipt_path is None
                else str(population_calibration_receipt_path)
            ),
        },
        "terminal_calibration_lineage": {
            "schema": "terminal_calibration_lineage_v2",
            "dense_calibration_state_id": fit_ids_at_terminal_lock[
                dense_mask.mask_hash
            ],
            "final_calibration_state_id": fit_ids_at_terminal_lock[
                final_mask.mask_hash
            ],
            "dense_decision_calibration_state_id": (
                canonical_dense_certificate.calibration_state_id
                if canonical_dense_certificate is not None
                else fit_ids_before_terminal[dense_mask.mask_hash]
            ),
            "final_decision_calibration_state_id": (
                fit_ids_before_terminal[final_mask.mask_hash]
            ),
            "terminal_closure_receipts": {
                mask_hash: dict(receipt)
                for mask_hash, receipt in (
                    terminal_calibration_closure_receipts.items()
                )
            },
            "terminal_closure_completed_before_validation_materialization": True,
            "dense_refit_on_validation": False,
            "final_refit_on_validation": False,
            "validation_rows_used_for_calibration": False,
            "historical_comparator_training_fit_state_id": (
                None
                if historical_mask is None
                else fit_ids_at_terminal_lock[historical_mask.mask_hash]
            ),
        },
        "decision_context_axis_hash": optimization_axis.content_hash,
        "decision_context_design_hash": decision_sampling.content_hash,
        "decision_oracle_context_design_hash": (
            decision_sampling.content_hash
            if decision_oracle_sampling is None
            else decision_oracle_sampling.content_hash
        ),
        "decision_evidence_streams": {
            "calibration_fit": ("mask_specific_empirical_iid_with_replacement"),
            "attribution_geometry": (
                "independent_sealed_without_replacement_training_geometry"
            ),
            "proposal_adjudication": (
                decision_oracle_sampling_receipts["proposal_adjudication_stream"]
            ),
            "terminal_validation": "evaluation_split_accessed_after_mask_lock",
        },
        "decision_source_population_counts": {
            "shared_training_distribution": {
                key: len(value)
                for key, value in (
                    decision_sampling.source_context_ids_by_stratum.items()
                )
            },
            "proposal_adjudication_strata": {
                key: len(value)
                for key, value in (
                    (
                        decision_sampling
                        if decision_oracle_sampling is None
                        else decision_oracle_sampling
                    ).source_context_ids_by_stratum.items()
                )
            },
        },
        "decision_static_role_partition": False,
        "decision_sampling_with_replacement": {
            "calibration_fit": True,
            "attribution_geometry": False,
            "proposal_adjudication": (
                decision_oracle_sampling_receipts["sampling_with_replacement"]
            ),
        },
        "decision_oracle_sampling_law": (
            decision_oracle_sampling_receipts["oracle_sampling_law"]
        ),
        "final_validation_context_axis_hash": evaluation_axis.content_hash,
        "final_validation_context_design_hash": dense_terminal["context_design_hash"],
        "final_validation_accessed_before_mask_lock": False,
        "terminal_validation_audit_seconds": terminal_audit_elapsed,
        "terminal_dense_bce": dense_terminal,
        "terminal_final_bce": final_terminal,
        "terminal_bce_delta": float(
            final_terminal["per_image_binary_bce" if loss_weighting() == "per_image" and pruning_functional() != "substituted_softmax" else "class_balanced_binary_bce"]
            - dense_terminal["per_image_binary_bce" if loss_weighting() == "per_image" and pruning_functional() != "substituted_softmax" else "class_balanced_binary_bce"]
        ),
        "terminal_dense_substituted_softmax": dense_terminal_substituted,
        "terminal_final_substituted_softmax": final_terminal_substituted,
        "terminal_substituted_softmax_bce_delta": _terminal_substituted_metric_delta(
            final_terminal_substituted, dense_terminal_substituted, "bce"
        ),
        "terminal_substituted_softmax_accuracy_delta": _terminal_substituted_metric_delta(
            final_terminal_substituted, dense_terminal_substituted, "accuracy"
        ),
        "terminal_category_structure": terminal_category_structure_block,
        # Execution-only disclosure of the shared
        # post-lock validation forward (absent when it is switched off).
        **(
            {}
            if terminal_validation_one_pass is None
            else {"terminal_validation_one_pass": terminal_validation_one_pass}
        ),
        "step_category_structure": step_category_structure_block,
        "step_observables_enabled": bool(step_observables_effective),
        **(
            {}
            if step_context_delta_block is None
            else {"step_context_deltas": dict(step_context_delta_block)}
        ),
        "historical_control": (
            None
            if historical_mask is None
            else {
                "source_sha256": historical_mask_sha256,
                "route_alias_receipt": historical_mask_alias_receipt,
                "mask_hash": historical_mask.mask_hash,
                "retained_route_count": historical_mask.retained_count,
                "retained_fraction": float(
                    historical_mask.retained_count / route_count
                ),
                "sparsity": float(1.0 - historical_mask.retained_count / route_count),
                "terminal_bce": historical_terminal,
                "terminal_bce_delta": float(
                    historical_terminal["per_image_binary_bce" if loss_weighting() == "per_image" and pruning_functional() != "substituted_softmax" else "class_balanced_binary_bce"]
                    - dense_terminal["per_image_binary_bce" if loss_weighting() == "per_image" and pruning_functional() != "substituted_softmax" else "class_balanced_binary_bce"]
                ),
                "terminal_substituted_softmax": (historical_terminal_substituted),
                "terminal_substituted_softmax_bce_delta": _terminal_substituted_metric_delta(
                    historical_terminal_substituted, dense_terminal_substituted, "bce"
                ),
                "terminal_substituted_softmax_accuracy_delta": _terminal_substituted_metric_delta(
                    historical_terminal_substituted, dense_terminal_substituted, "accuracy"
                ),
            }
        ),
        "confidence_level": sample_size_policy.confidence_level,
        "maximum_relative_bce_increase": maximum_relative_bce_increase,
        "calibration_deployment_refit_anchor": str(calibration_deployment_refit_anchor),
        "v0_calibration_decision_estimand": str(v0_calibration_decision_estimand),
        "v0_oracle_minimum_per_stratum": int(v0_oracle_minimum_per_stratum),
        "v0_accept_replication_gate": bool(v0_accept_replication_gate),
        "terminal_namespace_salt": (
            None if terminal_namespace_salt is None else str(terminal_namespace_salt)
        ),
        "sensitivity_tau": resolved_sensitivity_tau,
        "brier_used": False,
        "partition_topology": "two_partitions_per_physical_layer",
        "covariance_trust": "none",
        "workflow": workflow_details,
        "runtime_profile": profiler.snapshot(),
        "cold_runtime_setup_seconds": float(
            unit_gate_started - cold_runtime_setup_started
        ),
        "fixed_shape_runner_prewarm": dict(runner_prewarm_receipt),
        "runtime_execution_profile": (runtime_execution_profile_receipt),
        "numerics_record": unified_execution_receipt["numerics_record"],
        "extraction_gpu_activity": dict(extraction_gpu_activity_receipt),
        "decision_provider_resource_usage": dict(decision_provider_usage),
        "dense_readout_table_source": (
            "canonical" if canonical_dense_table_active else "per_arm"
        ),
        "dense_readout_table": dict(dense_readout_table_telemetry),
        "optimization_data_plane": optimization_data_plane_receipt,
        "evaluation_data_plane": evaluation_data_plane_receipt,
        "unified_native_execution_receipt": unified_execution_receipt,
        "performance_gate_receipt_path": "performance_gate.json",
        # Execution-only, present only when a residual observation
        # ran (never on AlexNet/VGG); see _residual_observation_path_counts_block.
        **(
            {}
            if residual_observation_path_counts is None
            else {"residual_observation_path_counts": residual_observation_path_counts}
        ),
        "source_manifest_verification": _source_manifest_verification(),
        # The shared-process worker's identity and item order (absent
        # in a fresh-process arm, whose receipt keeps its exact fields).
        **_shared_worker_process_context_block(),
    }
    if loss_weighting() == "per_image":
        payload["loss_weighting"] = "per_image"
    if pruning_functional() == "substituted_softmax":
        payload["pruning_functional"] = "substituted_softmax"
    write_arguments = dict(
        destination=destination,
        payload=payload,
        products=_arm_run_products(
            final_mask=final_mask,
            terminal_category_structure_sidecars=(
                terminal_category_structure_sidecars
            ),
            step_category_structure_block=step_category_structure_block,
            step_category_structure_sidecars=step_category_structure_sidecars,
            step_context_delta_sidecars=step_context_delta_sidecars,
            run_product_sidecars=run_product_sidecars,
        ),
        nonfinite_paths=_nonfinite_receipt_paths,
        receipt_hash=lambda compact: _pilot_receipt_hash_with_phase_timing(
            compact, phase_timing
        ),
        products_write_start=(
            None if phase_timing is None or not phase_timing.enabled
            else lambda: phase_timing.mark("step_archive_write_start")
        ),
        products_write_end=(
            None if phase_timing is None or not phase_timing.enabled
            else lambda: phase_timing.mark("step_archive_write_end")
        ),
        persist_staged=_persist_sidecar,
        writer=run_product_writer,
        before_receipt=before_receipt,
        environment=dict(os.environ),
    )

    def discard_staged():
        if run_product_writer is not None and not run_product_writer._finished:
            run_product_writer.cancel()
        for _, value, _ in write_arguments["products"]:
            if isinstance(value, (str, os.PathLike)):
                Path(value).unlink(missing_ok=True)

    def complete_write():
        try:
            return _write_compact_run_outputs(**write_arguments)
        finally:
            discard_staged()

    if output_submit is None:
        return complete_write()
    from .category_memo import container_copy

    # Detach nested telemetry mappings before retained runtime resources are
    # reset. Bytes and item-owned arrays remain shared with this completion.
    write_arguments["payload"] = container_copy(write_arguments["payload"])
    if phase_timing is not None:
        phase_timing.detach_for_host_completion()
    if run_product_writer is not None:
        run_product_writer.transfer_completion()
    try:
        return output_submit(destination, complete_write, abort=discard_staged)
    except BaseException:
        discard_staged()
        raise


def run_registered_resnet18_pilot(
    *,
    version_name: PilotVersion,
    partition_family: PilotPartitionFamily,
    study: StudySpec,
    plan: CircuitPlan,
    profile: Any,
    promotion_receipt: Any | None,
    query_id: str,
    comparison_namespace: str,
    optimization_axis: Any,
    evaluation_axis: Any,
    optimization_image_root: str | Path,
    evaluation_image_root: str | Path,
    store: ArtifactStore,
    output_dir: str | Path,
    historical_mask_path: str | Path | None = None,
    measurement_artifact_policy: str = "required",
    measurement_plan_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    model_source_replica: ModelSourceReplica | None = None,
    optimization_uint8_cache_manifest: str | Path | None = None,
    optimization_uint8_cache_receipt: str | Path | None = None,
    evaluation_uint8_cache_manifest: str | Path | None = None,
    evaluation_uint8_cache_receipt: str | Path | None = None,
    device: Any = "cuda:0",
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool = True,
    random_seed: int = 0,
    partition_random_seed: int | None = None,
    maximum_commits: int | None = None,
    maximum_relative_bce_increase: float = 0.20,
    sensitivity_tau: float | None = None,
    terminal_maximum_per_stratum: int = 256,
    sample_size_policy: IndependentSampleSizePolicy = IndependentSampleSizePolicy(),
    enable_local_regime_diagnostics: bool = False,
    partition_geometry_cache_dir: str | Path | None = None,
    dense_readout_table_cache_dir: str | Path | None = None,
    calibration_deployment_refit_anchor: str = "selected",
    terminal_namespace_salt: str | None = None,
    v0_calibration_decision_estimand: str = ("frozen_extraction_calibration_risk_v1"),
    v0_oracle_minimum_per_stratum: int = 2,
    v0_accept_replication_gate: bool = False,
    # Historical replay only.  ``None`` binds the registered v4 law (sealed
    # without replacement); passing the with-replacement id reproduces a
    # pre-v4 arm byte for byte and is recorded in the mask-lock
    # ``algorithm_configuration`` so an overridden arm is self-identifying.
    v0_oracle_sampling_law_override: str | None = None,
    v0_schedule_replay_spec: str | Path | None = None,
    v0_schedule_replay_rank_source: str = "family",
    v0_attribution_order: str = V0_DEFAULT_ATTRIBUTION_ORDER,
    v1_ordering_key: str = "abs_raw",
    # registered accept_rule.span_budget_fraction
    v1_span_budget_fraction: float = 0.066145,
    v1_calibration_fold_ownership: bool = True,
    # Terminal category-structure matrix sidecars. ON is the registered
    # default; OFF emits only the {enabled: false} receipt block.
    terminal_category_matrix: bool = True,
    # Per-step category-structure stack.  The RAW pilot function
    # default is OFF so every existing fixture call keeps its exact decision
    # float path and receipt bytes; the registered arm runner CLI defaults it
    # ON for the current registered v0 arms. Developing and reserved workflow
    # identities refuse before this function performs any side effect.
    step_category_matrix: bool = False,
    comparison_validation_mode: str = "v0_marginal",
    train_only_mask_lock: bool = False,
    # A registered, hash-bound load-time model transform
    # (zero_routes_v1, positive_unit_rescale_v1) applied to the
    # unchanged base Study's checkpoint.  ``None`` loads the base model.
    model_transform_path: str | Path | None = None,
    phase_timing_enabled: bool = True,
    phase_timing: _PhaseTiming | None = None,
    v0_decision_law_id: str = SEQUENTIAL_CALIBRATED_BCE_V5,
    census_reference_path: str | Path | None = None,
    census_reference_hash: str | None = None,
    census_mode: str | None = None,
    retained_runtime_inputs: dict[str, Any] | None = None,
    prepared_host_inputs: Any = None,
    output_submit: Callable[..., Any] | None = None,
) -> Mapping[str, Any]:
    """Execute one immutable matched ResNet18 pilot arm."""

    # This compatibility argument has its own fail-closed parse boundary.  It
    # must reject before workflow normalization or any study/model access so a
    # malformed key cannot be masked by an unrelated invalid fixture field.
    v1_ordering_key = str(v1_ordering_key)
    if v1_ordering_key not in V1_ORDERING_KEYS:
        raise ValueError(
            f"unsupported v1 ordering key {v1_ordering_key!r}; registered "
            f"keys are {V1_ORDERING_KEYS!r}"
        )
    if v1_ordering_key != "abs_raw":
        raise ValueError("ordering-key selection is a v1-only execution mode")
    v0_attribution_order = str(v0_attribution_order)
    if v0_decision_law_id not in {
        SEQUENTIAL_CALIBRATED_BCE_V4,
        SEQUENTIAL_CALIBRATED_BCE_V5,
    }:
        raise ValueError("unknown v0 decision law")
    # Refuse malformed terminal settings before loading a model or starting
    # extraction. The selected policy is used only after the walk is complete.
    _terminal_calibration_settings()
    _calibration_penalty()
    if v0_attribution_order not in V0_ATTRIBUTION_ORDER_MODES:
        raise ValueError(
            f"unsupported v0 attribution order mode {v0_attribution_order!r}; "
            f"registered modes are {V0_ATTRIBUTION_ORDER_MODES!r}"
        )

    # Workflow resolution is the first operation after argument-only parsing.
    # Developing or reserved names must refuse before callers can create
    # directories, load a model, touch CUDA, open an ArtifactStore, or expose
    # held-out material.
    workflow = resolve_current_workflow_identity(str(version_name))
    version = workflow.canonical_identity.rsplit("/", 1)[-1]
    from .modern_v0_runtime_profile import resolve_modern_v0_sensitivity_tau

    resolved_sensitivity_tau = resolve_modern_v0_sensitivity_tau(sensitivity_tau)
    comparison_validation_mode = str(comparison_validation_mode)
    if comparison_validation_mode not in {
        "v0_marginal",
        "modern_v1_direct_candidate",
    }:
        raise ValueError("unsupported comparison validation mode")
    if comparison_validation_mode != "v0_marginal" and version != "v0":
        raise ValueError(
            "the modern-v1 direct-candidate comparison adapter requires the "
            "registered v0 execution substrate"
        )
    direct_candidate_comparison = (
        comparison_validation_mode == "modern_v1_direct_candidate"
    )
    if type(train_only_mask_lock) is not bool:
        raise TypeError("train_only_mask_lock must be an exact boolean")
    if (
        os.environ.get("IRG_TERMINAL_DENSE_REFERENCE") is not None
        or os.environ.get("IRG_TERMINAL_DENSE_REFERENCE_SHA256") is not None
    ) and not terminal_category_matrix:
        raise ValueError("terminal dense reference requires terminal category matrices")
    _validate_v0_train_only_mask_lock_request(
        enabled=train_only_mask_lock,
        version=version,
        comparison_validation_mode=comparison_validation_mode,
        maximum_commits=maximum_commits,
        historical_mask_path=historical_mask_path,
        v0_schedule_replay_spec=v0_schedule_replay_spec,
        terminal_namespace_salt=terminal_namespace_salt,
        terminal_category_matrix=terminal_category_matrix,
        step_category_matrix=step_category_matrix,
    )
    if train_only_mask_lock:
        # Metadata certification is outcome-independent and happens before model,
        # CUDA, data-plane, store, or output side effects. The same axes are
        # re-bound into the terminal receipt after extraction.
        _validate_complete_imagenet_axis_metadata(
            optimization_axis,
            role="train",
        )
        _validate_complete_imagenet_axis_metadata(
            evaluation_axis,
            role="validation",
        )
        _validate_sealed_source_lineage(_sealed_source_identity())
    # Records the thread pools and warns on CPU oversubscription; never raises.
    cpu_thread_pools = _check_cpu_thread_pools(
        device=device,
        step_category_matrix=bool(step_category_matrix),
        optimization_bank=optimization_uint8_cache_manifest is not None,
        evaluation_bank=(
            optimization_uint8_cache_manifest is not None if train_only_mask_lock
            else evaluation_uint8_cache_manifest is not None
        ),
        num_workers=num_workers,
    )
    if prepared_host_inputs is not None:
        if (prepared_host_inputs.query_id != str(query_id)
                or prepared_host_inputs.optimization_axis_hash != optimization_axis.content_hash
                or prepared_host_inputs.random_seed != int(random_seed)
                or prepared_host_inputs.comparison_namespace != str(comparison_namespace)
                or prepared_host_inputs.partition_family != str(partition_family)):
            raise ValueError("prepared host inputs belong to another item")
        model_transform = prepared_host_inputs.model_transform
    else:
        model_transform = _load_pilot_model_transform(
            model_transform_path, study=study,
        )
    # Resolve and validate both opt-in numerics settings here, before
    # any model, CUDA, store or output side effect, so a typo cannot surface
    # only at first use (step setup, or the terminal phase after the walk).
    _validate_pilot_optin_settings()
    if pruning_functional() == "substituted_softmax" and version != "v0":
        raise ValueError("substituted_softmax pruning requires the registered v0 BCE walk")
    from .census_reference import (
        reference_settings, load_category_census, dense_anchor_request_hash, load_dense_anchor,
        bound_loss_span,
    )

    census_reference_path, census_reference_hash = reference_settings(
        census_reference_path, census_reference_hash, census_mode
    )
    (
        dense_baseline_mode,
        canonical_dense_table_receipt_path,
        population_calibration_receipt_path,
    ) = _resolved_dense_baseline_authority(
        architecture_family=str(study.route_schema.architecture_family).lower(),
        model_id=str(study.model_id),
    )
    if model_transform is not None:
        _require_transform_dense_authority(
            model_transform,
            canonical_dense_table_receipt_path=canonical_dense_table_receipt_path,
        )
    loss_weighting()  # Validate the loss estimand before allocating runtime resources.
    pruning_functional()  # Validate the decision functional and coherent combination.
    requested_family = str(partition_family)
    if version not in _VERSIONS or requested_family not in _PARTITIONS:
        raise ValueError("pilot version or partition family is unsupported")
    declared_architecture_family = str(study.route_schema.architecture_family).lower()
    if declared_architecture_family not in _REGISTERED_PILOT_ARCHITECTURE_FAMILIES:
        raise ValueError(
            "this pilot surface is locked to the registered families "
            f"{sorted(_REGISTERED_PILOT_ARCHITECTURE_FAMILIES)!r}"
        )
    _validate_architecture_feature_admission(
        architecture_family=declared_architecture_family,
        enable_local_regime_diagnostics=enable_local_regime_diagnostics,
        schedule_replay_requested=v0_schedule_replay_spec is not None,
    )
    family = _canonical_partition_family(requested_family)
    legal_families = {
        "local_attribution_sign",
        "global_attribution_sign",
        "historical_spatial_weight_sum_sign",
        "randomized_count_matched",
        "single_population",
    }
    if family not in legal_families:
        raise ValueError(
            f"{version} does not admit partition family {family!r}; "
            f"legal families are {tuple(sorted(legal_families))!r}"
        )
    _validate_v0_attribution_order_mode(
        mode=v0_attribution_order,
        version=version,
        architecture_family=declared_architecture_family,
        partition_family=family,
        v1_ordering_key=str(v1_ordering_key),
        v0_schedule_replay_spec=v0_schedule_replay_spec,
    )
    if not str(comparison_namespace):
        raise ValueError("matched pilots require a shared comparison namespace")
    # The modern registered identities do not inherit compatibility horizons
    # from a caller-created policy. Their IID sample count and number of looks
    # are determined only by confidence and the pruning boundary.
    sample_size_policy = _registered_precision_policy(
        version,
        sample_size_policy,
        oracle_sampling_law_override=v0_oracle_sampling_law_override,
    )
    if v0_calibration_decision_estimand not in {
        "frozen_extraction_calibration_risk_v1",
        "accepted_state_frozen_candidate_refinable_v1",
        "population_optimal_affine_risk_v1",
    }:
        raise ValueError(
            "unsupported v0 calibration decision estimand "
            f"{v0_calibration_decision_estimand!r}"
        )
    if int(v0_oracle_minimum_per_stratum) < 2:
        raise ValueError("v0 oracle minimum per stratum must be at least two")
    if v0_schedule_replay_rank_source not in SCHEDULE_REPLAY_RANK_SOURCES:
        raise ValueError(
            "unsupported v0 schedule replay rank source "
            f"{v0_schedule_replay_rank_source!r}"
        )
    # v1 ladder acceptance budget, declared in LOSS-SPAN units (matching
    # v0's span-normalized tau convention); consumed by the v1 path only.
    # ``maximum_relative_bce_increase`` remains untouched for the v0/tau
    # surfaces and the provider's per-decision increment bound.
    v1_span_budget_fraction = float(v1_span_budget_fraction)
    if (
        not math.isfinite(v1_span_budget_fraction)
        or not 0.0 < v1_span_budget_fraction < 1.0
    ):
        raise ValueError("v1 span budget fraction must lie in (0,1)")
    # The two count-matched conventions (census geometry
    # matches the GLOBAL dense sign split once per run; the hard-field
    # views contract matches the LOCAL split) do not conflict on the
    # v1 walk, because the walk-scope convention is registered: the
    # label-1 block is count-matched to the GLOBAL sign split of the
    # CURRENT chart snapshot — the walk's damage-bearing head — with
    # membership a per-chart frozen permutation (redrawn at every
    # re-centering via the committed-mask hash in the seed material) and
    # per-look counts nested along that permutation. The census and
    # views conventions stay byte-untouched on their own (v0) paths.
    # Registered scope: the widened decision lanes and per-step stack
    # apply to the current v0 paired-oracle surface. The workflow resolver
    # above refuses every developing or reserved identity before this mode is
    # evaluated, so no non-v0 receipt is emitted.
    step_category_matrix_effective = bool(step_category_matrix)
    v0_schedule_replay_schedule: Mapping[str, Any] | None = None
    v0_schedule_replay_spec_receipt: Mapping[str, Any] | None = None
    if v0_schedule_replay_spec is not None:
        if step_category_matrix_effective:
            raise ValueError(
                "accepted-only step category matrices are unavailable for "
                "the schedule-replay walker"
            )
        # Load and validate the sealed schedule before any device work so a
        # bad spec path fails closed at zero cost.
        if prepared_host_inputs is None:
            v0_schedule_replay_schedule, v0_schedule_replay_spec_receipt = load_schedule_replay_spec(v0_schedule_replay_spec)
        else:
            v0_schedule_replay_schedule = prepared_host_inputs.v0_schedule_replay_schedule
            v0_schedule_replay_spec_receipt = prepared_host_inputs.v0_schedule_replay_spec_receipt
    if family == "randomized_count_matched" and partition_random_seed is None:
        raise ValueError("randomized pilot arms require a frozen partition seed")
    if family != "randomized_count_matched" and partition_random_seed is not None:
        raise ValueError("nonrandom pilot arms may not carry a partition seed")
    # Prune scope and proposal grain: resolved and validated before any
    # side effect.  The default (both variables unset) is the registered
    # layerwise walk and reaches none of the guarded scope statements below.
    prune_scope_setting = _resolved_prune_scope_setting(
        version=version,
        family=family,
        schedule_replay_requested=v0_schedule_replay_spec is not None,
        v0_attribution_order=v0_attribution_order,
        direct_candidate_comparison=direct_candidate_comparison,
        architecture_family=declared_architecture_family,
        sensitivity_tau=resolved_sensitivity_tau,
    )
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"immutable pilot output exists: {destination}")
    workflow_started = time.perf_counter()
    if phase_timing is None:
        phase_timing = _PhaseTiming(enabled=phase_timing_enabled)
    phase_timing.mark("setup_start")

    from mechanistic_systems.backends.cuda_graphs import initialize_cuda_graph_state
    initialize_cuda_graph_state(device)

    from mechanistic_systems.backends.profiling import PhaseProfiler

    _progress(
        "pilot_start",
        version=version,
        partition_family=family,
        output_dir=str(destination),
        dense_baseline_mode=dense_baseline_mode,
    )
    profiler = PhaseProfiler(phase_timing=phase_timing)
    # Inductor's torch key (a sha256 over every torch
    # source module, ~14 s of NFS metadata walks in a smoke-test profile) is
    # computed on a helper thread while the runtime resolves, instead of
    # inside the first compile of the prewarm.  Same function, same bytes.
    if os.environ.get("IRG_V6_RUNTIME_ITEMS", "0") == "1":
        phase_timing.mark("torch_key_start")
    inductor_torch_key_prefetch = _start_inductor_torch_key_prefetch(device)
    if retained_runtime_inputs is not None:
        for source_role in ("optimization_source", "evaluation_source"):
            source = retained_runtime_inputs.get(source_role)
            if source is not None:
                source._bundle_cache.clear()
                if hasattr(source, "exit_raw_panel_delivery"):
                    source.exit_raw_panel_delivery()
                bank = getattr(source, "_inputs", None)
                if bank is not None and hasattr(bank, "rows_read"):
                    bank.rows_read = 0
                    bank.bytes_read = 0
                    bank.read_calls = 0
    runtime = _resolve_registered_sampled_path_runtime(
        study=study,
        plan=plan,
        profile=profile,
        promotion_receipt=promotion_receipt,
        optimization_axis=optimization_axis,
        # Train-only mask locking must not even construct or certified-probe
        # a validation source. Bind the runtime's otherwise-unused evaluation
        # slot to the same complete train replica; the real validation axis
        # remains metadata-only and is bound separately into the receipt.
        evaluation_axis=(
            optimization_axis if train_only_mask_lock else evaluation_axis
        ),
        optimization_image_root=optimization_image_root,
        evaluation_image_root=(
            optimization_image_root
            if train_only_mask_lock
            else evaluation_image_root
        ),
        optimization_uint8_cache_manifest=optimization_uint8_cache_manifest,
        optimization_uint8_cache_receipt=optimization_uint8_cache_receipt,
        evaluation_uint8_cache_manifest=(
            optimization_uint8_cache_manifest
            if train_only_mask_lock
            else evaluation_uint8_cache_manifest
        ),
        evaluation_uint8_cache_receipt=(
            optimization_uint8_cache_receipt
            if train_only_mask_lock
            else evaluation_uint8_cache_receipt
        ),
        store=store,
        profiler=profiler,
        checkpoint_path=checkpoint_path,
        allow_download=False,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        loaded_model=None,
        model_source_replica=model_source_replica,
        queries=(_a47_sealed_queries(measurement_artifact_policy, measurement_plan_path)
                 if prepared_host_inputs is None else prepared_host_inputs.sealed_queries),
        measurement_artifact_policy=str(measurement_artifact_policy),
        **({} if model_transform is None else {"model_transform": model_transform}),
        **(
            {}
            if retained_runtime_inputs is None
            else {"retained_inputs": retained_runtime_inputs}
        ),
    )
    if retained_runtime_inputs is not None:
        retained_runtime_inputs.update(
            loaded=runtime.loaded,
            optimization_source=runtime.optimization_source,
            evaluation_source=runtime.evaluation_source,
            evaluator=runtime.evaluator,
            active_planes=(runtime.optimization_data_plane, runtime.evaluation_data_plane),
        )
    _require_runtime_carries_model_transform(runtime, model_transform)
    # The backend resolves the prefix setting at construction and preserves
    # worker executable caches. K=1 requests remain ordinary functional calls.
    cold_runtime_setup_started = time.perf_counter()
    gpu_activity_sampler = GpuActivitySampler(
        device=device,
        interval_seconds=0.5,
    ).start()
    _progress("registered_runtime_resolved")
    runtime_architecture_family = runtime.loaded.receipt.architecture_family.lower()
    if runtime_architecture_family not in _REGISTERED_PILOT_ARCHITECTURE_FAMILIES:
        raise ValueError(
            "this pilot surface is locked to the registered families "
            f"{sorted(_REGISTERED_PILOT_ARCHITECTURE_FAMILIES)!r}"
        )
    if runtime_architecture_family != declared_architecture_family:
        raise ValueError("loaded model architecture family differs from the StudySpec")
    # A non-default prune scope or grain runs only on AlexNet's seven
    # registered connection groups; checked before the census and the walk.
    prune_scope_route_group_registration: dict[str, Any] | None = None
    if not prune_scope_setting.is_default:
        from .prune_scope_walk import validate_registered_prune_scope_route_groups

        prune_scope_route_group_registration = (
            validate_registered_prune_scope_route_groups(
                prune_scope_setting,
                runtime.route_groups,
                architecture_family=runtime_architecture_family,
            )
        )
    query_by_id = {query.query_id: query for query in runtime.query_axis}
    if str(query_id) not in query_by_id:
        raise ValueError("pilot query is outside the measurement artifact")
    query = query_by_id[str(query_id)]
    query_index = tuple(runtime.measurement.query_ids).index(str(query_id))
    target_class_id = _target_class_id(query)
    resident_image_preflight = _preflight_registered_resident_images(
        runtime=runtime, optimization_axis=optimization_axis,
        target_class_id=target_class_id, maximum_per_stratum=terminal_maximum_per_stratum,
        category_enabled=bool(terminal_category_matrix or step_category_matrix_effective),
    )
    route_axis_hash = str(plan.route_universe_hash)
    route_count = sum(
        int(np.prod(group.shape, dtype=np.int64)) for group in runtime.route_groups
    )
    dense_mask = HardMask.dense(route_count, route_axis_hash=route_axis_hash)
    _progress("fixed_shape_runner_prewarm_started")
    if os.environ.get("IRG_V6_RUNTIME_ITEMS", "0") == "1":
        phase_timing.mark("torch_key_join_start")
    torch_key_prefetch_receipt = _join_inductor_torch_key_prefetch(
        inductor_torch_key_prefetch
    )
    if os.environ.get("IRG_V6_RUNTIME_ITEMS", "0") == "1":
        phase_timing.mark(
            "torch_key_end",
            helper_seconds=(
                None if torch_key_prefetch_receipt is None
                else torch_key_prefetch_receipt["helper_seconds"]
            ),
            join_wait_seconds=(
                None if torch_key_prefetch_receipt is None
                else torch_key_prefetch_receipt["join_wait_seconds"]
            ),
        )
        phase_timing.mark("prewarm_start")
    runner_prewarm_receipt = (
        _prewarm_registered_mask_runners(
            runtime=runtime,
            dense_mask=dense_mask,
            target_class_id=target_class_id,
            phase_timing=(
                phase_timing
                if os.environ.get("IRG_V6_RUNTIME_ITEMS", "0") == "1"
                else None
            ),
        )
        if runtime.evaluator.device.type == "cuda"
        else {
            "schema": "registered_resnet18_mask_runner_prewarm_v1",
            "status": "skipped_non_cuda_or_non_v0_v1",
            "cold_setup_only": True,
            "statistical_rows_updated": 0,
        }
    )
    # Captured graphs and persistent forward buffers now occupy the process
    # allowance. Recheck both resident sets before acquiring census or walk
    # rows; validation admission still reads only the bank's row count.
    resident_image_preflight = _preflight_registered_resident_images(
        runtime=runtime, optimization_axis=optimization_axis,
        target_class_id=target_class_id, maximum_per_stratum=terminal_maximum_per_stratum,
        category_enabled=bool(terminal_category_matrix or step_category_matrix_effective),
    )
    if resident_image_preflight:
        runner_prewarm_receipt = {
            **runner_prewarm_receipt, "resident_image_preflight_bytes": resident_image_preflight,
        }
    if torch_key_prefetch_receipt is not None:
        runner_prewarm_receipt = {
            **runner_prewarm_receipt,
            "inductor_torch_key_prefetch": torch_key_prefetch_receipt,
        }
    _progress(
        "fixed_shape_runner_prewarm_completed",
        **runner_prewarm_receipt,
    )
    if os.environ.get("IRG_V6_RUNTIME_ITEMS", "0") == "1":
        phase_timing.mark("prewarm_end")
    # The warm-unit performance gate starts after the immutable bank is mapped
    # and the fixed K=1/K=2 structural runners are compiled.  Cold setup is
    # retained separately in the final runtime receipt and campaign wall time.
    unit_gate_started = time.perf_counter()
    historical_mask: HardMask | None = None
    historical_mask_sha256: str | None = None
    historical_mask_alias_receipt: Mapping[str, Any] | None = None
    if historical_mask_path is not None:
        (
            historical_mask,
            historical_mask_sha256,
            historical_mask_alias_receipt,
        ) = _load_historical_hard_mask(
            historical_mask_path,
            runtime=runtime,
            route_axis_hash=route_axis_hash,
            prepared_payload=(None if prepared_host_inputs is None else prepared_host_inputs.historical_mask_payload),
            prepared_sha256=(None if prepared_host_inputs is None else prepared_host_inputs.historical_mask_sha256),
        )
    if prepared_host_inputs is None:
        shared_run_id = content_hash(
            {
                "comparison_namespace": str(comparison_namespace),
                "query_id": str(query_id),
                "optimization_axis_hash": optimization_axis.content_hash,
                "random_seed": int(random_seed),
            },
            prefix="irg-matched-pilot-",
        )
        decision_namespace = binary_evidence_role_namespace(
            run_id=shared_run_id,
            random_seed=random_seed,
            context_axis_hash=optimization_axis.content_hash,
        )
        decision_sampling = compile_binary_iid_sampling_design(
            optimization_axis,
            target_class_id=target_class_id,
            randomization_namespace=decision_namespace,
        )
        # v0 makes many sequential, layer-local decisions.  A pooled off-target
        # packet is unbiased for the binary risk, but a small packet can entirely
        # miss most competitor classes and allow the adaptive trajectory to
        # specialize to the few classes it happened to observe.  Resolve the same
        # one-half target / one-half off-target estimand over explicit class
        # strata.  The provider supplies two initial observations per stratum,
        # adds one occupancy-complete quantum, and then applies the ordinary
        # cost-aware Neyman allocator.  This recovers the competitor coverage of
        # the historical per-category panel without its fixed 16-row-per-class
        # cost at every proposal.
        decision_oracle_sampling = _decision_oracle_sampling_design(
            optimization_axis,
            target_class_id=target_class_id,
            randomization_namespace=f"{decision_namespace}:class-resolved-oracle",
            oracle_sampling_law=str(sample_size_policy.oracle_sampling_law),
        )
        decision_oracle_sampling_receipts = _oracle_sampling_receipt_strings(
            decision_oracle_sampling
        )
    else:
        shared_run_id = prepared_host_inputs.shared_run_id
        decision_namespace = prepared_host_inputs.decision_namespace
        decision_sampling = prepared_host_inputs.decision_sampling
        decision_oracle_sampling = prepared_host_inputs.decision_oracle_sampling
        decision_oracle_sampling_receipts = dict(prepared_host_inputs.decision_oracle_sampling_receipts)
    v0_mask_registry: dict[str, HardMask] = {
        dense_mask.mask_hash: dense_mask,
    }
    v0_candidate_registry: dict[str, Any] = {}
    canonical_decision_provider: NativeTorchCalibratedBCEOracleProvider | None = None
    canonical_dense_certificate: Any | None = None
    canonical_dense_calibration: Any | None = None
    dense_calibration_closure_receipt: Mapping[str, Any] | None = None
    dense_readout_table_cache_paths: tuple[Path, Path, Path] | None = None
    dense_readout_table_live_source_ids: tuple[str, ...] = ()
    dense_readout_table_labels_by_context_id: dict[str, int] = {}
    dense_readout_table_refused_payload_sha256: str | None = None
    dense_readout_table_telemetry: dict[str, Any] = {
        "cache_key": None,
        "hit": False,
        "preloaded_rows": 0,
        "skipped_unknown_ids": 0,
        "implementation_comparisons": "removed",
        "stored_rows": 0,
        "source": "per_arm",
        "canonical_payload_sha256": None,
        "canonical_table_sha256": None,
        "canonical_execution_policy": None,
        "canonical_receipt_path": None,
    }
    canonical_dense_table_active = False
    census_dense_anchor_binding = None
    v0_loss_span_certificate: LossSpanCertificate | None = None
    v0_loss_span_closure_receipt: Mapping[str, Any] | None = None
    v0_empty_certificate: AnalyticChanceRiskCertificate | None = None
    empty_response_proof: Mapping[str, Any] | None = None
    step_category_setup_receipt: Mapping[str, Any] | None = None
    step_observables_effective = bool(step_observables_enabled())
    step_context_delta_accumulator = (
        StepContextDeltaAccumulator() if step_observables_effective else None
    )
    if version == "v0":
        canonical_decision_provider = _fixed_calibration_bce_provider(
            runtime=runtime,
            optimization_axis=optimization_axis,
            target_class_id=target_class_id,
            # The boundary is bound from the measured dense certificate below;
            # there is no provisional legacy or chance-risk scale.
            maximum_bce_increment=None,
            run_id=shared_run_id,
            random_seed=random_seed,
            mask_resolver=(lambda mask_hash: v0_mask_registry[str(mask_hash)]),
            candidate_resolver=(
                lambda candidate_id: v0_candidate_registry[str(candidate_id)]
            ),
            sample_size_policy=sample_size_policy,
            iid_sampling_design=decision_sampling,
            # The weights, rather than the stratum count, define the same
            # balanced binary estimand.  v0 uses class-resolved off-target
            # strata for sequential competitor coverage; v1 retains the
            # pooled binary design pending its separate promotion work.
            oracle_iid_sampling_design=decision_oracle_sampling,
            oracle_panel_role="v0_paired_oracle",
            progress_callback=_progress,
            calibration_deployment_refit_anchor=(calibration_deployment_refit_anchor),
            oracle_minimum_per_stratum=int(v0_oracle_minimum_per_stratum),
            # K-fold cross-fitted ownership of the calibration affine is
            # a v1 trust-walk execution mode.  The v0 construction stays
            # byte-identical (default off).
            calibration_fold_ownership=False,
            step_category_matrix=step_category_matrix_effective,
        )
        if runtime.evaluator.device.type == "cuda":
            full_look_prewarm = _prewarm_registered_mask_runners(
                runtime=runtime, dense_mask=dense_mask, target_class_id=target_class_id,
                oracle_provider=canonical_decision_provider,
                phase_timing=(phase_timing if os.environ.get("IRG_V6_RUNTIME_ITEMS", "0") == "1" else None),
            )
            resident_image_preflight = _preflight_registered_resident_images(
                runtime=runtime, optimization_axis=optimization_axis,
                target_class_id=target_class_id, maximum_per_stratum=terminal_maximum_per_stratum,
                category_enabled=bool(terminal_category_matrix or step_category_matrix_effective),
            )
            runner_prewarm_receipt = {
                **runner_prewarm_receipt, "full_look_prewarm": full_look_prewarm,
                "resident_image_preflight_bytes": resident_image_preflight,
            }
            # Both physical warmups are cold setup and update no evidence.
            unit_gate_started = time.perf_counter()
        # An explicit env-named receipt takes over the dense-side
        # readout reference for the two observation-only consumers.  This is
        # loaded and VALIDATED here at arm setup -- a SEPARATE, provider-side
        # store (never ``_readout_source_cache``) -- and refuses loudly on
        # any identity mismatch rather than silently falling back.
        canonical_dense_table_active = bool(canonical_dense_table_receipt_path)
        if canonical_dense_table_active:
            canonical_dense_table_binding = (
                _load_and_validate_canonical_dense_readout_table(
                    receipt_path=canonical_dense_table_receipt_path,
                    model_state_hash=runtime.loaded.receipt.state_dict_hash,
                    route_schema_hash=runtime.evaluator.route_schema_hash,
                    topology_hash=runtime.evaluator.topology_hash,
                    dense_mask_hash=dense_mask.mask_hash,
                    context_axis_hash=optimization_axis.content_hash,
                    context_axis_split=str(optimization_axis.split),
                    forward_tier=int(
                        runtime.evaluator.runtime_execution_profile.forward_tier
                    ),
                )
            )
            canonical_decision_provider.load_canonical_dense_reference(
                canonical_dense_table_binding.handle,
                identity={
                    "receipt_path": canonical_dense_table_binding.receipt_path,
                    "payload_sha256": (canonical_dense_table_binding.payload_sha256),
                    "execution_policy": (
                        canonical_dense_table_binding.execution_policy
                    ),
                    "dense_mask_hash": (canonical_dense_table_binding.dense_mask_hash),
                },
            )
            dense_readout_table_telemetry.update(
                {
                    "source": "canonical",
                    "canonical_payload_sha256": dict(
                        canonical_dense_table_binding.payload_sha256
                    ),
                    "canonical_table_sha256": (
                        canonical_dense_table_binding.table_sha256
                    ),
                    "canonical_execution_policy": (
                        canonical_dense_table_binding.execution_policy
                    ),
                    "canonical_receipt_path": (
                        canonical_dense_table_binding.receipt_path
                    ),
                    "verified_by": canonical_dense_table_binding.verified_by,
                    "stage_receipt_path": (
                        canonical_dense_table_binding.stage_receipt_path
                    ),
                    "stage_receipt_refusal_reason": (
                        canonical_dense_table_binding.stage_receipt_refusal_reason
                    ),
                    # v6: present only when the per-query hash path ran.
                    **(
                        {}
                        if canonical_dense_table_binding.verify_once is None
                        else {
                            "verify_once": dict(
                                canonical_dense_table_binding.verify_once
                            )
                        }
                    ),
                }
            )
            _progress(
                "canonical_dense_readout_table_loaded",
                receipt_path=canonical_dense_table_binding.receipt_path,
                table_sha256=canonical_dense_table_binding.table_sha256,
                execution_policy=canonical_dense_table_binding.execution_policy,
                row_count=len(canonical_dense_table_binding.handle.source_context_ids),
            )
        dense_resolution_spec = SequentialCalibratedBCEResolutionSpec(
            **({"stratum_weights": tuple(sorted(_binary_loss_weights(
                decision_sampling.source_context_ids_by_stratum).items()))}
               if loss_weighting() == "per_image" else {}),
            confidence_level=sample_size_policy.confidence_level,
            estimand_policy=V0_NORMALIZED_LOSS_SPAN_PER_CONNECTION_V1,
            maximum_planned_looks=None,
            precision_only_iid=True,
        )
        dense_readout_table_cache_key = _dense_readout_table_cache_key(
            runtime=runtime,
            optimization_axis=optimization_axis,
            provider=canonical_decision_provider,
            dense_mask=dense_mask,
            target_class_id=target_class_id,
        )
        dense_readout_table_telemetry["cache_key"] = dense_readout_table_cache_key
        dense_readout_table_live_source_ids = tuple(
            dict.fromkeys(
                str(source_context_id)
                for values in decision_sampling.source_context_ids_by_stratum.values()
                for source_context_id in values
            )
        )
        dense_readout_table_labels_by_context_id = {
            str(record.context_id): int(record.label)
            for record in optimization_axis.records
        }
        if dense_readout_table_cache_dir is not None and canonical_dense_table_active:
            # The canonical table subsumes the per-arm load/probe
            # entirely (it refuses under TF32 essentially always anyway --
            # see the IRG_DENSE_READOUT_TABLE_STORE=0 note below).  This is
            # the load/probe-side counterpart of that store-side knob.
            dense_readout_table_telemetry["load_probe_skipped"] = (
                "canonical_dense_table_active"
            )
            _progress(
                "dense_readout_table_cache_load_probe_skipped",
                reason="canonical_dense_table_active",
            )
        elif dense_readout_table_cache_dir is not None:
            import fcntl

            cache_root = Path(dense_readout_table_cache_dir)
            cache_root.mkdir(parents=True, exist_ok=True)
            payload_path = cache_root / f"{dense_readout_table_cache_key}.npz"
            receipt_path = cache_root / f"{dense_readout_table_cache_key}.json"
            lock_path = cache_root / f"{dense_readout_table_cache_key}.lock"
            dense_readout_table_cache_paths = (
                payload_path,
                receipt_path,
                lock_path,
            )
            _progress(
                "dense_readout_table_cache_waiting",
                cache_key=dense_readout_table_cache_key,
                lock_path=str(lock_path),
            )
            with lock_path.open("a+b") as lock_stream:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
                try:
                    _progress(
                        "dense_readout_table_cache_lock_acquired",
                        cache_key=dense_readout_table_cache_key,
                    )
                    loaded_dense_readout_table = _load_dense_readout_table_cache(
                        payload_path=payload_path,
                        receipt_path=receipt_path,
                        cache_key=dense_readout_table_cache_key,
                        dense_mask_hash=dense_mask.mask_hash,
                        context_axis_hash=optimization_axis.content_hash,
                        forward_tier=int(
                            runtime.evaluator.runtime_execution_profile.forward_tier
                        ),
                        live_source_context_ids=(dense_readout_table_live_source_ids),
                        labels_by_context_id=(dense_readout_table_labels_by_context_id),
                    )
                    if loaded_dense_readout_table is not None:
                        if not loaded_dense_readout_table.valid:
                            dense_readout_table_refused_payload_sha256 = (
                                loaded_dense_readout_table.payload_sha256
                            )
                            _progress(
                                "dense_readout_table_cache_refused",
                                cache_key=dense_readout_table_cache_key,
                                reason=loaded_dense_readout_table.reason,
                            )
                        else:
                            dense_readout_table_telemetry["skipped_unknown_ids"] = int(
                                loaded_dense_readout_table.skipped_unknown_ids
                            )
                            preloaded_rows = _preload_dense_readout_table(
                                provider=canonical_decision_provider,
                                dense_mask=dense_mask,
                                loaded=loaded_dense_readout_table,
                            )
                            dense_readout_table_telemetry.update(
                                {"hit": True, "preloaded_rows": preloaded_rows}
                            )
                finally:
                    fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        _dense_baseline_confidence_level = (
            dense_resolution_spec.v0_anchor_confidence_level
        )
        population_dense_certificate: BaselineRiskCertificate | None = None
        population_dense_closure: dict[str, Any] | None = None
        if population_calibration_receipt_path:
            if not canonical_dense_table_active:
                raise ValueError(
                    f"{IRG_CANONICAL_DENSE_POPULATION_CALIBRATION_ENV} "
                    "requires the canonical dense training table "
                    f"({IRG_CANONICAL_DENSE_TABLE_ENV})"
                )
            (
                population_dense_fit,
                population_dense_certificate,
                population_dense_closure,
            ) = _load_population_dense_calibration(
                population_calibration_receipt_path,
                table_binding=canonical_dense_table_binding,
                target_class_id=int(canonical_decision_provider.target_class_id),
                dense_mask_hash=dense_mask.mask_hash,
                confidence_level=_dense_baseline_confidence_level,
            )
            canonical_decision_provider.install_population_calibration(
                population_dense_fit
            )
            if pruning_functional() == "substituted_softmax":
                canonical_decision_provider._decision_population_scale_zero_reference = dict(
                    population_dense_closure["scale_zero_reference"])
            if _calibration_penalty() == "firth":
                diagnostics = getattr(canonical_decision_provider, "_firth_calibration_diagnostics", None)
                if diagnostics is None:
                    diagnostics = canonical_decision_provider._firth_calibration_diagnostics = {}
                diagnostics[dense_mask.mask_hash] = {
                    "calibration_penalty": "firth",
                    "calibration_effective_sample_size": population_dense_closure["effective_sample_size"],
                    "calibration_weighting": ("per_image_population_mean" if loss_weighting() == "per_image" else "half_target_half_equal_represented_off_target_classes"),
                    **({"loss_weighting": "per_image"} if loss_weighting() == "per_image" else {}),
                    "calibration_off_target_class_count": 999,
                    "calibration_missing_off_target_class_count": 0,
                }
            _progress(
                "dense_population_calibration_bound",
                calibration_state_id=(population_dense_fit.calibration_state_id),
                bce_dense_population=float(population_dense_certificate.estimate),
                span_dense_to_empty_exact=float(
                    population_dense_closure["span_dense_to_empty_exact"]
                ),
                risk_context_count=int(population_dense_certificate.context_count),
                artifact_receipt_path=str(
                    population_dense_closure["artifact_receipt_path"]
                ),
                training_only=True,
                validation_responses_observed=False,
            )

        if census_reference_path is not None:
            anchor_request_hash = dense_anchor_request_hash(
                provider=canonical_decision_provider,
                runtime=runtime,
                optimization_axis=optimization_axis,
                dense_mask=dense_mask,
                target_class_id=target_class_id,
                run_id=shared_run_id,
                random_seed=random_seed,
                sample_size_policy=sample_size_policy,
                confidence_level=_dense_baseline_confidence_level,
            )
            canonical_dense_certificate, dense_calibration_closure_receipt, census_dense_anchor_binding = load_dense_anchor(
                reference_path=census_reference_path,
                reference_hash=census_reference_hash,
                request_hash=anchor_request_hash,
                provider=canonical_decision_provider,
                dense_mask=dense_mask,
                include_binding_metadata=True,
            )
        elif population_dense_certificate is not None:
            canonical_dense_certificate = population_dense_certificate
            dense_calibration_closure_receipt = population_dense_closure
        else:
            canonical_dense_certificate = (
                canonical_decision_provider.certify_baseline_risk(
                    dense_mask,
                    cumulative_context_count=max(
                        2 * int(canonical_decision_provider.oracle_minimum_per_stratum),
                        int(sample_size_policy.oracle_initial_packet),
                    ),
                    confidence_level=_dense_baseline_confidence_level,
                    maximum_planned_looks=None,
                    precision_only_iid=True,
                )
            )
            (
                canonical_dense_certificate,
                dense_calibration_closure_receipt,
            ) = _close_locked_mask_calibration_to_risk_resolution(
                provider=canonical_decision_provider,
                mask=dense_mask,
                confidence_level=_dense_baseline_confidence_level,
                cumulative_risk_context_count=int(
                    canonical_dense_certificate.context_count
                ),
                certificate=canonical_dense_certificate,
                certificate_role=("dense_baseline_proposal_adjudication_reference"),
                stream_namespace="dense-baseline",
                progress_callback=_progress,
            )
        dense_boundary_certificate = (
            canonical_dense_certificate if census_dense_anchor_binding is None
            else census_dense_anchor_binding["initial_certificate"]
        )
        canonical_decision_provider.bind_maximum_bce_increment(
            float(dense_boundary_certificate.estimate)
            * float(maximum_relative_bce_increase),
            reference_certificate_hash=dense_boundary_certificate.content_hash,
        )
        canonical_dense_calibration = canonical_decision_provider.calibration_fits[
            dense_mask.mask_hash
        ]
        if (
            canonical_dense_calibration.calibration_state_id
            != canonical_dense_certificate.calibration_state_id
        ):
            raise RuntimeError(
                "dense calibration differs from the locked risk certificate"
            )
        if version == "v0":
            fully_pruned_mask = HardMask.from_bool_array(
                np.zeros(route_count, dtype=bool),
                route_axis_hash=route_axis_hash,
            )
            if pruning_functional() == "substituted_softmax":
                v0_empty_certificate, empty_response_proof = _substituted_scale_zero_risk_certificate(
                    provider=canonical_decision_provider, reference_mask=fully_pruned_mask,
                    confidence_level=dense_resolution_spec.v0_anchor_confidence_level,
                    population_closure=population_dense_closure)
            else:
                empty_response_proof = _fully_pruned_constant_response_proof(
                    runtime=runtime,
                    fully_pruned_mask=fully_pruned_mask,
                )
                v0_empty_certificate = AnalyticChanceRiskCertificate(
                    mask_hash=fully_pruned_mask.mask_hash,
                    constant_response_proof_id=str(empty_response_proof["content_hash"]),
                    code_identity=(
                        f"registered_{runtime_architecture_family}_per_image_binary_chance_anchor_v1"
                        if loss_weighting() == "per_image" else (
                            "registered_resnet18_balanced_binary_chance_anchor_v1"
                            if runtime_architecture_family
                            in {
                                "resnet18",
                                "alexnet",
                                "alexnet_conv_kernel_weight",
                            }
                            else "registered_"
                            f"{runtime_architecture_family}_balanced_binary_"
                            "chance_anchor_v1"
                        )
                    ),
                    confidence_level=(dense_resolution_spec.v0_anchor_confidence_level),
                    **({"schema": "analytic_per_image_binary_natural_log_bce_chance_risk_v1",
                        "stratum_weights": tuple(sorted(_binary_loss_weights(
                            decision_sampling.source_context_ids_by_stratum).items()))}
                       if loss_weighting() == "per_image" else {}),
                )
            if census_dense_anchor_binding is not None:
                v0_loss_span_certificate, v0_loss_span_closure_receipt = bound_loss_span(
                    binding=census_dense_anchor_binding,
                    dense_certificate=canonical_dense_certificate,
                    empty_certificate=v0_empty_certificate,
                )
            else:
                (
                    canonical_dense_certificate,
                    v0_loss_span_certificate,
                    v0_loss_span_closure_receipt,
                ) = _close_positive_dense_to_empty_span(
                    provider=canonical_decision_provider,
                    dense_mask=dense_mask,
                    empty_certificate=v0_empty_certificate,
                    dense_certificate=canonical_dense_certificate,
                    confidence_level=(dense_resolution_spec.v0_anchor_confidence_level),
                    cumulative_risk_context_count=int(
                        canonical_dense_certificate.context_count
                    ),
                    progress_callback=_progress,
                )
            canonical_dense_calibration = canonical_decision_provider.calibration_fits[
                dense_mask.mask_hash
            ]
            if (
                canonical_dense_calibration.calibration_state_id
                != canonical_dense_certificate.calibration_state_id
            ):
                raise RuntimeError("dense calibration differs after loss-span closure")

    canonical_dense_baseline_bce = (
        None
        if canonical_dense_certificate is None
        else float(canonical_dense_certificate.estimate)
    )
    _progress(
        "optimization_training_reference_prepared",
        canonical_dense_baseline_bce=canonical_dense_baseline_bce,
        decision_dense_baseline_status=(
            "not_applicable_v2"
            if canonical_dense_certificate is None
            else "locked_common_mask_calibration_and_risk_certificate"
        ),
        dense_calibration_state_id=(
            None
            if canonical_dense_calibration is None
            else canonical_dense_calibration.calibration_state_id
        ),
        dense_baseline_certificate_hash=(
            None
            if canonical_dense_certificate is None
            else canonical_dense_certificate.content_hash
        ),
        decision_sampling_design_hash=decision_sampling.content_hash,
        decision_oracle_sampling_design_hash=(
            decision_sampling.content_hash
            if decision_oracle_sampling is None
            else decision_oracle_sampling.content_hash
        ),
        source_population_counts={
            key: len(value)
            for key, value in (decision_sampling.source_context_ids_by_stratum.items())
        },
        oracle_source_population_counts={
            key: len(value)
            for key, value in (
                (
                    decision_sampling
                    if decision_oracle_sampling is None
                    else decision_oracle_sampling
                ).source_context_ids_by_stratum.items()
            )
        },
        calibration_sampling="independent_empirical_iid_with_replacement",
        response_sampling="independent_sealed_without_replacement_geometry",
        oracle_sampling=decision_oracle_sampling_receipts["oracle_sampling"],
        static_role_partition=False,
        final_validation_accessed=False,
    )

    partition_local_means: dict[str, np.ndarray] | None = None
    partition_global_means: dict[str, np.ndarray] | None = None
    partition_global_activity: dict[str, np.ndarray] | None = None
    partition_ordering_statistics: dict[str, dict[str, np.ndarray]] | None = None
    partition_flip_margin_geometry: dict[str, Any] | None = None
    hard_partition_measurement: dict[str, Any] = {}
    v0_route_ids: tuple[str, ...] | None = None
    v0_route_layer_ids: tuple[str, ...] | None = None
    required_partition_views = _required_partition_views(family)
    phase_timing.mark("setup_end")
    if retained_runtime_inputs is not None:
        from mechanistic_systems.measurement.memory_probe import worker_memory_boundary

        retained_runtime_inputs["worker_setup_memory"] = worker_memory_boundary(
            retained_runtime_inputs, label="post_setup",
        )
    if required_partition_views:
        phase_timing.mark("census_start")
        _progress("hard_partition_field_measurement_started")
        pruning_guiding_loss_id = (CALIBRATED_SUBSTITUTED_SOFTMAX
            if pruning_functional() == "substituted_softmax"
            else resolve_pruning_guiding_loss_id(v0_decision_law_id))
        if canonical_decision_provider.pruning_guiding_loss_id != pruning_guiding_loss_id:
            raise ValueError("census objective differs from the pruning decision law")
        census_materiality_resolution = canonical_decision_provider.census_materiality_resolution(
            dense_mask,
            confidence_level=float(sample_size_policy.confidence_level),
            initial_total_count=max(
                4, int(runtime.evaluator.runtime_execution_profile.v0_paired_oracle_batch_rows)
            ),
        )
        _census_batch_design_guard(
            census_sample_design_setting(), family=str(family),
            proposal_route_cap=prune_scope_setting.proposal_route_cap,
            schedule_replay=v0_schedule_replay_schedule, census_reference_path=census_reference_path,
        )
        census_cache_settings = _p_census_settings(
            census_schedule_replay_schedule=v0_schedule_replay_schedule,
            census_profile=profile,
            census_proposal_route_cap=prune_scope_setting.proposal_route_cap,
            census_materiality_resolution=census_materiality_resolution,
            census_sample_design=_census_design_for_run(
                census_sample_design_setting(),
                sensitivity_tau=resolved_sensitivity_tau,
                loss_span_certificate=v0_loss_span_certificate,
                confidence_level=float(sample_size_policy.confidence_level),
            ),
            runtime=runtime,
            optimization_axis=optimization_axis,
            query=query,
            dense_mask=dense_mask,
            dense_calibration_state=canonical_dense_calibration,
            pruning_guiding_loss_id=pruning_guiding_loss_id,
            run_id=shared_run_id,
            random_seed=random_seed,
            sample_size_policy=sample_size_policy,
            enable_local_regime_diagnostics=enable_local_regime_diagnostics,
            required_partition_views=required_partition_views,
            partition_family=family,
            partition_random_seed=int(partition_random_seed or 0),
            prune_order_scope="layerwise",
            initial_partition=str(profile.initial_partition),
            v1_ordering_key=v1_ordering_key,
            v0_attribution_order=v0_attribution_order,
        )
        cache_key = _hard_partition_cache_key(**census_cache_settings)
        # The cached field payload is read
        # and hashed on a helper thread (positional reads and hashlib release
        # the GIL) while this thread materializes the route axis (GIL-bound).
        # The loader below consumes it under the cache lock only after the
        # receipt digest and the file identity both match; otherwise it reads
        # again.  No value, order or check changes.
        # The shared-process worker holds one census per category in
        # memory (category_memo); a fresh-process arm has no active memo.
        from . import category_memo

        census_memo = None if census_reference_path is not None else category_memo.active()
        partition_payload_prefetch = _prefetch_category_census(
            cache_key, partition_geometry_cache_dir, census_memo, census_reference_path,
        )
        v0_route_ids, v0_route_layer_ids = _global_route_axes(runtime)

        def measure_census() -> tuple[Any, ...]:
            return _measure_dense_local_partition_geometry(
                **census_cache_settings,
                route_ids=v0_route_ids,
                route_layer_ids=v0_route_layer_ids,
                target_class_id=target_class_id,
                **({"decision_fixed_offset_resolver": canonical_decision_provider.decision_fixed_offsets}
                   if pruning_functional() == "substituted_softmax" else {}),
            )

        def family_census_view(shared: Mapping[str, Any]) -> tuple[tuple[Any, ...] | None, dict[str, Any]]:
            return _family_census_on_shared(
                shared,
                census_schedule_replay_schedule=v0_schedule_replay_schedule,
                census_profile=profile,
                census_proposal_route_cap=prune_scope_setting.proposal_route_cap,
                runtime=runtime,
                family=family,
                partition_random_seed=int(partition_random_seed or 0),
                target_class_id=int(target_class_id),
                sample_size_policy=sample_size_policy,
                required_partition_views=required_partition_views,
                v0_attribution_order=v0_attribution_order,
            )

        if census_reference_path is not None:
            (
                partition_local_means,
                partition_global_means,
                partition_global_activity,
                partition_ordering_statistics,
                partition_flip_margin_geometry,
                hard_partition_measurement,
            ) = load_category_census(
                reference_path=census_reference_path,
                reference_hash=census_reference_hash,
                cache_key=cache_key,
                runtime=runtime,
                optimization_axis=optimization_axis,
                dense_mask=dense_mask,
                query=query,
                target_class_id=target_class_id,
                dense_calibration_state=canonical_dense_calibration,
                pruning_guiding_loss_id=pruning_guiding_loss_id,
                partition_family=family,
            )
        elif census_memo is not None:
            (
                partition_local_means,
                partition_global_means,
                partition_global_activity,
                partition_ordering_statistics,
                partition_flip_margin_geometry,
                hard_partition_measurement,
            ) = _category_memo_census(
                memo=census_memo,
                cache_key=cache_key,
                shared_key=_shared_category_census_key(
                    category=census_memo.category,
                    **census_cache_settings,
                ),
                partition_family=family,
                partition_geometry_cache_dir=partition_geometry_cache_dir,
                runtime=runtime,
                prefetch=partition_payload_prefetch,
                measure=measure_census,
                family_view=family_census_view,
            )
        elif partition_geometry_cache_dir is None:
            (
                partition_local_means,
                partition_global_means,
                partition_global_activity,
                partition_ordering_statistics,
                partition_flip_margin_geometry,
                hard_partition_measurement,
            ) = _measure_census_deferred(measure_census)
            hard_partition_measurement.update(
                {
                    "dense_field_cache_key": cache_key,
                    "dense_field_cache_hit": False,
                }
            )
        else:
            # The locked cache block lives in
            # _load_or_measure_hard_partition_field so
            # its receipt contract can be tested directly.
            cached = _load_or_measure_hard_partition_field(
                cache_key=cache_key,
                partition_geometry_cache_dir=partition_geometry_cache_dir,
                runtime=runtime,
                partition_family="local_attribution_sign",
                prefetch=partition_payload_prefetch,
                measure=measure_census,
            )
            (
                partition_local_means,
                partition_global_means,
                partition_global_activity,
                partition_ordering_statistics,
                partition_flip_margin_geometry,
                hard_partition_measurement,
            ) = cached
            # The unpacked fields now own every required reference; do not
            # let the returned tuple pin all route arrays until function exit
            # after the phase-local names retire.
            cached = None
        if census_memo is None and family != "local_attribution_sign":
            consumer_fields, _ = family_census_view({
                "family": "local_attribution_sign",
                "cache_key": cache_key,
                "result": (
                    partition_local_means, partition_global_means,
                    partition_global_activity, partition_ordering_statistics,
                    partition_flip_margin_geometry, hard_partition_measurement,
                ),
            })
            (
                partition_local_means, partition_global_means,
                partition_global_activity, partition_ordering_statistics,
                partition_flip_margin_geometry, hard_partition_measurement,
            ) = consumer_fields
        # After the dense-field cache store, so a cache hit records this run's
        # pools and no cache entry carries them.
        _record_cpu_thread_pools(hard_partition_measurement, cpu_thread_pools)
        _progress(
            "hard_partition_field_measurement_completed",
            scoring_context_packet_count=hard_partition_measurement[
                "scoring_context_packet_count"
            ],
            scoring_context_packet_size=hard_partition_measurement[
                "scoring_context_packet_size"
            ],
            census_host_postprocessing_seconds=hard_partition_measurement.get(
                "census_host_postprocessing_seconds"
            ),
        )
        phase_timing.mark("census_done")

    partition_global_activity = None
    if partition_ordering_statistics is not None:
        partition_loss_activity = partition_ordering_statistics["loss_activity"]
    else:
        raise RuntimeError("walk requires the dense census loss_activity field")

    workflow_details: dict[str, Any]
    v0_scoring_anchor_elapsed: float | None = None
    v0_traversal_elapsed: float | None = None
    v0_traversal_complete: bool | None = None
    accepted_v0_trajectory: AcceptedV0Trajectory | None = None
    residual_observation_path_counts: dict[str, Any] | None = None
    # Set only by the guarded prune-scope branch of the walk below.
    prune_scope_walk: Any = None
    prune_scope_walk_receipt: dict[str, Any] | None = None
    if version == "v0":
        previous_walk_layer: str | None = None

        def _timed_group_transition(**facts: object) -> None:
            nonlocal previous_walk_layer
            if previous_walk_layer is not None:
                phase_timing.mark("layer_walk_end", layer_id=previous_walk_layer)
            previous_walk_layer = str(facts["layer_id"])
            phase_timing.mark("layer_walk_start", layer_id=previous_walk_layer)
            runtime.evaluator.candidate_execution_backend.invalidate_activation_prefix_cache(
                "v0_layer_group_transition:" + previous_walk_layer
            )

        layer_execution_order = _resolved_layer_execution_order()
        _progress(
            "v0_layerwise_pruning_started",
            layer_execution_order=layer_execution_order,
        )
        if not prune_scope_setting.is_default:
            _progress("v0_prune_scope_selected", **prune_scope_setting.to_dict())
        if runtime.walk_device_cache_bytes > 0:
            runtime.optimization_data_plane.set_device_cache_bytes(
                runtime.walk_device_cache_bytes
            )
            _progress(
                "walk_device_cache_enabled",
                device_cache_bytes=int(runtime.walk_device_cache_bytes),
            )
        labels = _partition_labels_by_layer(
            runtime,
            measurement=runtime.measurement,
            query_index=query_index,
            family=family,
            random_seed=int(partition_random_seed or 0),
            local_mean_by_layer=partition_local_means,
            global_mean_by_layer=partition_global_means,
            local_destination_scales_by_layer=(
                resnet_local_destination_transport(runtime.evaluator)[0]
            ),
            category_id=int(target_class_id),
        )
        # v5: how the randomized control's labels were drawn (RNG key and
        # count source), receipted in workflow_details below.
        partition_randomization_receipt = (
            _randomized_partition_receipt(
                labels,
                runtime=runtime,
                partition_random_seed=int(partition_random_seed or 0),
                category_id=int(target_class_id),
            )
            if family == "randomized_count_matched"
            else None
        )
        # Labels are now an owned byte copy, so the signed-field inputs that
        # defined them have reached their last consumer.  Keep only the local
        # field needed by the optional local-attribution schedule replay and
        # the activity field needed to construct the walk ordering below.
        # Rebinding (rather than allocator trimming or spilling) releases the
        # owning mappings without adding synchronization, I/O, or numerical
        # work to the scientific path.
        # The global walk's reporting-only cross-layer order diagnostic
        # reads the stored loss activity, its standard error, and the signed
        # loss-gradient mean with its standard error (the
        # arguments the layerwise census passes to its order certificate).
        prune_scope_ordering_field = None
        if (
            not prune_scope_setting.is_default
            and prune_scope_setting.scope == "global"
            and partition_ordering_statistics is not None
        ):
            prune_scope_ordering_field = {
                "loss_activity_standard_error": partition_ordering_statistics["loss_activity_standard_error"],
                "loss_mean": partition_ordering_statistics["loss_mean"],
                "loss_mean_standard_error": partition_ordering_statistics["loss_mean_standard_error"],
            }
        partition_global_means = None
        partition_ordering_statistics = None
        partition_flip_margin_geometry = None
        partition_local_means = None
        if v0_route_ids is None:
            v0_route_ids, v0_route_layer_ids = _global_route_axes(runtime)
        if (
            canonical_decision_provider is None
            or canonical_dense_certificate is None
            or canonical_dense_calibration is None
            or v0_loss_span_certificate is None
            or v0_empty_certificate is None
            or empty_response_proof is None
        ):
            raise RuntimeError("modern v0 canonical dense state is incomplete")
        v0_decision_provider = canonical_decision_provider
        v0_dense_certificate = canonical_dense_certificate
        v0_sequential_spec = dense_resolution_spec
        v0_canonical_dense_baseline_bce = float(canonical_dense_certificate.estimate)
        _progress(
            "decision_training_reference_locked",
            canonical_dense_baseline_bce=v0_canonical_dense_baseline_bce,
            dense_mask_hash=dense_mask.mask_hash,
            dense_calibration_state_id=(v0_dense_certificate.calibration_state_id),
            dense_baseline_certificate_hash=(v0_dense_certificate.content_hash),
            calibration_contract=(
                v0_decision_provider.calibration_options["optimization_contract"]
            ),
            dense_is_trajectory_state_zero=True,
        )
        _progress(
            "v0_loss_span_certified",
            dense_bce=v0_dense_certificate.estimate,
            dense_lower=v0_dense_certificate.lower,
            dense_upper=v0_dense_certificate.upper,
            fully_pruned_bce=v0_empty_certificate.estimate,
            fully_pruned_lower=v0_empty_certificate.lower,
            fully_pruned_upper=v0_empty_certificate.upper,
            loss_span=v0_loss_span_certificate.estimate,
            loss_span_lower=v0_loss_span_certificate.lower,
            loss_span_upper=v0_loss_span_certificate.upper,
            loss_span_certificate_hash=(v0_loss_span_certificate.content_hash),
            fully_pruned_anchor_kind=v0_empty_certificate.schema,
            fully_pruned_constant_response_proof=empty_response_proof,
        )
        # Arm the per-step category-structure stack at walk setup —
        # dense frozen frame via the cache trio, once-per-family cluster
        # derivation, and the derived decay — strictly after the dense
        # calibration froze and strictly before the first paired decision.
        # The sealed terminal populations are a pure function of
        # (axis, namespace, cap), so this setup-scope computation is
        # byte-identical to the terminal one below.
        if step_category_matrix_effective:
            assert decision_oracle_sampling is not None
            _progress("step_category_structure_setup_started")
            setup_populations, _setup_weights = _sealed_terminal_populations(
                optimization_axis,
                target_class_id=target_class_id,
                namespace=(
                    "architecture-dense-category-calibration:"
                    f"{runtime.loaded.receipt.state_dict_hash}:"
                    f"{optimization_axis.content_hash}"
                ),
                maximum_per_stratum=terminal_maximum_per_stratum,
            )
            step_category_setup_receipt = _step_category_structure_setup(
                enabled=True,
                provider=v0_decision_provider,
                dense_mask=dense_mask,
                training_populations=setup_populations,
                target_class_id=target_class_id,
                batch_factory=(
                    runtime.optimization_data_plane.role_batch_factory(
                        "category_calibration"
                    )
                ),
                model_state_hash=runtime.loaded.receipt.state_dict_hash,
                context_axis_hash=optimization_axis.content_hash,
                forward_tier=int(
                    runtime.evaluator.runtime_execution_profile.forward_tier
                ),
                cache_dir=(
                    None
                    if dense_readout_table_cache_dir is None
                    else Path(dense_readout_table_cache_dir)
                    / "shared_category_structure"
                ),
                oracle_stratum_class_ids=tuple(
                    (
                        int(target_class_id)
                        if stratum == "on_target"
                        else int(stratum.split(":", 1)[1])
                    )
                    for stratum in (
                        decision_oracle_sampling.source_context_ids_by_stratum
                    )
                ),
                oracle_minimum_per_stratum=int(v0_oracle_minimum_per_stratum),
                progress_callback=_progress,
            )
            _progress(
                "step_category_structure_setup_completed",
                decay_lambda=(
                    step_category_setup_receipt["lambda_receipt"]["decay_lambda"]
                ),
                k_clusters=(
                    step_category_setup_receipt["cluster_receipt"]["k_realized"]
                ),
            )
        v0_decision_backend = NativeFixedCalibrationBCEPrecisionBackend(
            provider=v0_decision_provider,
            route_ids=v0_route_ids,
            hard_mask_from_layers=lambda masks: _hard_mask_from_layers(
                masks,
                runtime=runtime,
                route_axis_hash=route_axis_hash,
            ),
            mask_registry=v0_mask_registry,
            candidate_registry=v0_candidate_registry,
            dense_baseline_bce=v0_canonical_dense_baseline_bce,
            loss_span_certificate=v0_loss_span_certificate,
            context_axis=optimization_axis,
            calibration_decision_estimand=(
                v0_calibration_decision_estimand
            ),
        )
        # Naturalization chooses only the first physical packet. Precision,
        # not this execution fact, determines the eventual IID sample count.
        oracle_context_count = int(sample_size_policy.oracle_initial_packet)
        v0_naturalization = BackendNaturalizationFacts(
            class_count=2,
            bank_depth_per_class=max(
                2,
                int(math.ceil(oracle_context_count / 2.0)),
            ),
            # The statistical packet is the throughput-optimal physical K=2
            # execution quantum, not the 32-row controller
            # default. Neyman allocation still determines the on/off-target
            # composition; occupancy rounding adds genuine IID observations.
            gpu_batch_size=max(
                4,
                int(
                    runtime.evaluator.runtime_execution_profile.v0_paired_oracle_batch_rows
                ),
            ),
        )
        scoring_anchor_elapsed = time.perf_counter() - unit_gate_started
        traversal_started = time.perf_counter()
        v0_schedule_replay_run_receipt: Mapping[str, Any] | None = None
        if v0_schedule_replay_schedule is not None:
            # Pre-registered schedule probe: the recorded historical group
            # schedule replaces the tau traversal; every other decision
            # surface (provider, fits, paired audit, downstream receipts)
            # is the baseline v0 machinery.
            runs, v0_schedule_replay_run_receipt = run_schedule_replay_walker(
                schedule=v0_schedule_replay_schedule,
                policy=ScheduleReplayPolicy(
                    schedule_replay_spec=str(v0_schedule_replay_spec),
                    schedule_replay_rank_source=str(v0_schedule_replay_rank_source),
                    maximum_relative_bce_increase=float(maximum_relative_bce_increase),
                    confidence_level=float(sample_size_policy.confidence_level),
                    oracle_initial_packet=int(sample_size_policy.oracle_initial_packet),
                    random_seed=int(random_seed),
                    maximum_commits=(
                        None if maximum_commits is None else int(maximum_commits)
                    ),
                ),
                backend=v0_decision_backend,
                query_id=str(query_id),
                route_groups=runtime.route_groups,
                partition_labels_by_layer=labels,
                rank_scores_by_layer=resolve_replay_rank_scores(
                    rank_source=str(v0_schedule_replay_rank_source),
                    route_groups=runtime.route_groups,
                    partition_loss_activity=partition_loss_activity,
                ),
                dense_baseline_bce=v0_canonical_dense_baseline_bce,
                positive_loss_span_estimate=float(v0_loss_span_certificate.estimate),
                profile_hash=str(profile.content_hash),
                # v3 dense-lane disclosure input: the locked dense
                # certificate CI, never a decision quantity.
                dense_certificate_bounds=(
                    float(v0_dense_certificate.lower),
                    float(v0_dense_certificate.upper),
                ),
                group_transition_callback=_timed_group_transition,
                progress_callback=_progress,
            )
        elif not prune_scope_setting.is_default:
            # The explicitly selected prune scope / proposal grain.  The
            # inputs are those of the registered panel call below; only the
            # group structure, the lowerer and the ladder differ (see
            # prune_scope_walk).  The admission check at entry refused every
            # other version, family, order and comparison mode.
            from .prune_scope_walk import run_prune_scope_panel

            prune_scope_walk = run_prune_scope_panel(
                setting=prune_scope_setting,
                backend=v0_decision_backend,
                profile=profile,
                measurement=runtime.measurement,
                layer_execution_order=layer_execution_order,
                route_activity_scores_by_layer={
                    group.layer_id: np.asarray(
                        partition_loss_activity[group.layer_id], dtype=np.float64,
                    ).reshape(1, -1)
                    for group in runtime.route_groups
                },
                query_ids=(str(query_id),),
                route_groups=runtime.route_groups,
                partition_labels_by_layer=labels,
                dense_anchor_loss_by_query={
                    str(query_id): v0_canonical_dense_baseline_bce
                },
                promotion_receipt=promotion_receipt,
                claim_bearing=False,
                algorithm=BCE_PRECISION_ALGORITHM,
                maximum_commits=maximum_commits,
                accept_replication_gate=bool(v0_accept_replication_gate),
                bce_decision_rule_by_query={
                    str(query_id): V0MarginalConnectionDecisionRule(
                        confidence_level=sample_size_policy.confidence_level,
                        sensitivity_tau=resolved_sensitivity_tau,
                    )
                },
                loss_span_certificate_by_query={
                    str(query_id): (v0_decision_backend.current_loss_span_certificate)
                },
                group_transition_callback=_timed_group_transition,
                naturalization_facts=v0_naturalization,
                oracle_sampling_law=str(sample_size_policy.oracle_sampling_law),
                oracle_acquisition_envelope_seconds=(
                    sample_size_policy.oracle_acquisition_envelope_seconds
                ),
                decision_law_id=v0_decision_law_id,
                progress_callback=_progress,
                step_context_delta_sink=_bind_step_context_delta_sink(
                    step_context_delta_accumulator,
                    v0_decision_provider,
                ),
            )
            runs = prune_scope_walk.runs_by_query[str(query_id)]
            prune_scope_walk_receipt = prune_scope_walk.receipt(
                str(query_id),
                route_group_registration=prune_scope_route_group_registration,
                score_source="census_loss_activity",
                census={
                    "pruning_guiding_loss_id": hard_partition_measurement["pruning_guiding_loss_id"],
                    "dense_calibration_state_hash": hard_partition_measurement["dense_calibration_state_hash"],
                    "dense_calibration_state_id": hard_partition_measurement["dense_calibration_state_id"],
                    "order_field": "loss_activity",
                    "partition_family": str(family),
                    "dense_field_cache_key": hard_partition_measurement.get(
                        "dense_field_cache_key"
                    ),
                    "dense_field_cache_hit": hard_partition_measurement.get(
                        "dense_field_cache_hit"
                    ),
                },
            )
        else:
            runs = run_profiled_native_sampled_path_panel(
                backend=v0_decision_backend,
                profile=profile,
                measurement=runtime.measurement,
                layer_execution_order=layer_execution_order,
                route_activity_scores_by_layer={
                    group.layer_id: np.asarray(
                        partition_loss_activity[group.layer_id], dtype=np.float64,
                    ).reshape(1, -1)
                    for group in runtime.route_groups
                },
                query_ids=(str(query_id),),
                route_groups=runtime.route_groups,
                partition_labels_by_layer=labels,
                dense_anchor_loss_by_query={
                    str(query_id): v0_canonical_dense_baseline_bce
                },
                promotion_receipt=promotion_receipt,
                claim_bearing=False,
                algorithm=BCE_PRECISION_ALGORITHM,
                maximum_commits=maximum_commits,
                accept_replication_gate=bool(v0_accept_replication_gate),
                bce_decision_rule_by_query={
                    str(query_id): (
                        DirectCandidateIncrementDecisionRule(
                            confidence_level=sample_size_policy.confidence_level,
                            maximum_candidate_increment=(
                                v1_span_budget_fraction
                                * v0_loss_span_certificate.estimate
                            ),
                        )
                        if direct_candidate_comparison
                        else V0MarginalConnectionDecisionRule(
                            confidence_level=sample_size_policy.confidence_level,
                            sensitivity_tau=resolved_sensitivity_tau,
                        )
                    )
                },
                loss_span_certificate_by_query={
                    str(query_id): (v0_decision_backend.current_loss_span_certificate)
                },
                group_transition_callback=_timed_group_transition,
                naturalization_facts=v0_naturalization,
                # v4 decision law: the receipts certify the law the provider
                # actually samples under.
                oracle_sampling_law=str(sample_size_policy.oracle_sampling_law),
                oracle_acquisition_envelope_seconds=(
                    sample_size_policy.oracle_acquisition_envelope_seconds
                ),
                decision_law_id=v0_decision_law_id,
                progress_callback=_progress,
                step_context_delta_sink=_bind_step_context_delta_sink(
                    step_context_delta_accumulator,
                    v0_decision_provider,
                ),
            )[str(query_id)]
        if previous_walk_layer is not None:
            phase_timing.mark("layer_walk_end", layer_id=previous_walk_layer)
        # Both walkers consume their ordering arrays synchronously.  No
        # terminal-calibration, validation, or receipt path reads these
        # full-width fields, so retire the final references before those
        # phases allocate their own category matrices and sidecars.
        partition_local_means = None
        partition_loss_activity = None
        del labels
        if not runs:
            raise RuntimeError("v0 pilot produced no layerwise pruning runs")
        phase_timing.mark("final_mask_pack_start")
        final_mask = _hard_mask_from_layers(
            runs[-1].result.committed_masks,
            runtime=runtime,
            route_axis_hash=route_axis_hash,
        )
        phase_timing.mark("final_mask_pack_end")
        phase_timing.mark("walk_tail_start")
        (
            accepted_v0_trajectory,
            extraction_elapsed,
            v0_scoring_anchor_elapsed,
            v0_traversal_complete,
            v0_traversal_elapsed,
            workflow_details,
        ) = _walk_tail_block(
            accepted_v0_trajectory=accepted_v0_trajectory,
            dense_mask=dense_mask,
            direct_candidate_comparison=direct_candidate_comparison,
            family=family,
            partition_random_seed=partition_random_seed,
            final_mask=final_mask,
            hard_partition_measurement=hard_partition_measurement,
            maximum_commits=maximum_commits,
            partition_randomization_receipt=partition_randomization_receipt,
            query_id=query_id,
            resolved_sensitivity_tau=resolved_sensitivity_tau,
            runs=runs,
            runtime=runtime,
            runtime_architecture_family=runtime_architecture_family,
            scoring_anchor_elapsed=scoring_anchor_elapsed,
            traversal_started=traversal_started,
            v0_attribution_order=v0_attribution_order,
            v0_candidate_registry=v0_candidate_registry,
            v0_decision_backend=v0_decision_backend,
            v0_loss_span_certificate=v0_loss_span_certificate,
            v0_loss_span_closure_receipt=v0_loss_span_closure_receipt,
            v0_mask_registry=v0_mask_registry,
            v0_schedule_replay_run_receipt=v0_schedule_replay_run_receipt,
            v0_schedule_replay_spec_receipt=v0_schedule_replay_spec_receipt,
            v0_sequential_spec=v0_sequential_spec,
            v1_span_budget_fraction=v1_span_budget_fraction,
            workflow_started=workflow_started,
            prune_scope_walk_receipt=prune_scope_walk_receipt,
            phase_timing=phase_timing,
        )
        phase_timing.mark("walk_tail_end")
        if prune_scope_walk is not None:
            # Reporting only, and after the traversal clock stopped: the
            # cross-layer order certificate call at every committed
            # global cut, and a descriptive band labelled as not a
            # certificate.  No decision reads it.
            from .prune_scope_walk import (
                census_order_critical_value,
                cross_layer_order_diagnostic,
            )

            phase_timing.mark("cross_layer_order_diagnostic_start")
            workflow_details["prune_scope_walk"]["cross_layer_order_diagnostic"] = (
                cross_layer_order_diagnostic(
                    prune_scope_walk,
                    str(query_id),
                    ordering_field=prune_scope_ordering_field,
                    critical_value=census_order_critical_value(
                        hard_partition_measurement
                    ),
                    maximum_mask_disagreement_activity_fraction=float(
                        sample_size_policy.maximum_prune_order_uncertainty_fraction
                    ),
                )
            )
            phase_timing.mark("cross_layer_order_diagnostic_end")
            prune_scope_walk = None
            prune_scope_ordering_field = None
        # Once per walk, execution-only.
        residual_observation_path_counts = _residual_observation_path_counts_block(
            v0_decision_backend.provider
        )
        if residual_observation_path_counts is not None:
            _progress(
                "residual_observation_path_counts",
                execution_only=True,
                counts=residual_observation_path_counts["counts"],
                device_fallback_to_host=(
                    residual_observation_path_counts["device_fallback_to_host"]
                ),
            )
        if runtime.walk_device_cache_bytes > 0:
            phase_timing.mark("walk_device_cache_release_start")
            runtime.optimization_data_plane.release_device_cache()
            phase_timing.mark("walk_device_cache_release_end")
            _progress("walk_device_cache_released")
    if train_only_mask_lock:
        # This is the exact continuation cut: final_mask, workflow_details,
        # and the accepted trajectory have been locked, while no terminal
        # calibration closure, evaluation batch factory, or validation row
        # has been requested. Returning here is the only authority granted by
        # the opt-in mode.
        if (
            version != "v0"
            or accepted_v0_trajectory is None
            or v0_route_ids is None
            or v0_scoring_anchor_elapsed is None
            or v0_traversal_elapsed is None
            or v0_traversal_complete is None
            or v0_loss_span_certificate is None
            or decision_oracle_sampling is None
        ):
            raise RuntimeError("v0 mask-lock lineage is incomplete")
        return _finalize_v0_train_only_mask_lock(
            destination=destination,
            final_mask=final_mask,
            gpu_activity_sampler=gpu_activity_sampler,
            phase_timing=phase_timing,
            receipt_kwargs={
                "runtime": runtime,
                "study": study,
                "plan": plan,
                "profile": profile,
                "optimization_axis": optimization_axis,
                "evaluation_axis": evaluation_axis,
                "comparison_namespace": comparison_namespace,
                "requested_partition_family": requested_family,
                "partition_family": family,
                "partition_random_seed": partition_random_seed,
                "query_id": query_id,
                "target_class_id": target_class_id,
                "random_seed": random_seed,
                "sensitivity_tau": resolved_sensitivity_tau,
                "maximum_relative_bce_increase": (maximum_relative_bce_increase),
                "v0_calibration_decision_estimand": (v0_calibration_decision_estimand),
                "v0_oracle_minimum_per_stratum": (v0_oracle_minimum_per_stratum),
                "v0_accept_replication_gate": v0_accept_replication_gate,
                "v0_decision_law_id": v0_decision_law_id,
                "v0_attribution_order": v0_attribution_order,
                "v0_oracle_sampling_law": str(
                    sample_size_policy.oracle_sampling_law
                ),
                "v0_oracle_acquisition_envelope_seconds": (
                    sample_size_policy.oracle_acquisition_envelope_seconds
                ),
                "v0_oracle_sampling_law_override": (
                    v0_oracle_sampling_law_override
                ),
                "dense_baseline_mode": dense_baseline_mode,
                "decision_sampling": decision_sampling,
                "decision_oracle_sampling": decision_oracle_sampling,
                "loss_span_certificate": v0_loss_span_certificate,
                "hard_partition_measurement": hard_partition_measurement,
                "workflow_details": workflow_details,
                "dense_mask": dense_mask,
                "final_mask": final_mask,
                "route_ids": v0_route_ids,
                "accepted_trajectory": accepted_v0_trajectory,
                "decision_provider": v0_decision_backend.provider,
                "scoring_and_anchor_seconds": v0_scoring_anchor_elapsed,
                "traversal_seconds": v0_traversal_elapsed,
                "run_entry_to_mask_lock_seconds": extraction_elapsed,
                "traversal_complete": v0_traversal_complete,
                "model_source_replica": model_source_replica,
                **(
                    {}
                    if model_transform is None
                    else {"model_transform_block": model_transform.receipt_block()}
                ),
            },
        )
    phase_timing.mark("step_archive_setup_start")
    with _step_archive_at_mask_lock(
        enabled=step_category_matrix_effective,
        destination=destination,
        provider=v0_decision_backend.provider if step_category_matrix_effective else None,
        setup_receipt=step_category_setup_receipt,
        accepted_trajectory=accepted_v0_trajectory,
    ) as run_product_writer:
        phase_timing.mark("step_archive_setup_end")
        profiler.run_product_writer = run_product_writer
        # The complete physical mask is now locked. Before the validation axis
        # is materialized, close the final/historical affine fits on their
        # extraction-training streams to the selected terminal tolerance
        # (default 1e-4), or the legacy independent training-risk radius. Dense
        # retains its pre-pruning fit. Intermediate proposal calibration keeps
        # its existing policy; terminal closure changes no mask and observes no
        # validation row.
        (
            decision_provider,
            extraction_gpu_activity_receipt,
            fit_ids_at_terminal_lock,
            fit_ids_before_terminal,
            target_substituted_dense_fit,
            target_substituted_final_fit,
            target_substituted_historical_fit,
            terminal_audit_started,
            terminal_calibration_closure_receipts,
            terminal_category_frozen_banks,
            terminal_provider,
            validation_batch_factory,
            validation_populations,
            validation_stratum_weights,
        ) = _terminal_fit_block(
            dense_calibration_closure_receipt=dense_calibration_closure_receipt,
            dense_mask=dense_mask,
            evaluation_axis=evaluation_axis,
            final_mask=final_mask,
            gpu_activity_sampler=gpu_activity_sampler,
            historical_mask=historical_mask,
            optimization_axis=optimization_axis,
            runtime=runtime,
            sample_size_policy=sample_size_policy,
            shared_run_id=shared_run_id,
            step_category_setup_receipt=step_category_setup_receipt,
            target_class_id=target_class_id,
            terminal_category_matrix=terminal_category_matrix,
            terminal_maximum_per_stratum=terminal_maximum_per_stratum,
            terminal_namespace_salt=terminal_namespace_salt,
            v0_decision_backend=v0_decision_backend,
            phase_timing=phase_timing,
            walk_workflow_details=workflow_details,
        )
        phase_timing.mark("terminal_evaluation_start")
        (
            dense_terminal,
            dense_terminal_substituted,
            final_terminal,
            final_terminal_substituted,
            historical_terminal,
            historical_terminal_substituted,
            terminal_audit_elapsed,
            terminal_category_structure_block,
            terminal_category_structure_sidecars,
            terminal_validation_one_pass,
        ) = _terminal_evaluation_block(
            dense_mask=dense_mask,
            dense_readout_table_cache_dir=dense_readout_table_cache_dir,
            evaluation_axis=evaluation_axis,
            final_mask=final_mask,
            historical_mask=historical_mask,
            runtime=runtime,
            target_class_id=target_class_id,
            target_substituted_dense_fit=target_substituted_dense_fit,
            target_substituted_final_fit=target_substituted_final_fit,
            target_substituted_historical_fit=target_substituted_historical_fit,
            terminal_audit_started=terminal_audit_started,
            terminal_category_frozen_banks=terminal_category_frozen_banks,
            terminal_category_matrix=terminal_category_matrix,
            terminal_provider=terminal_provider,
            validation_batch_factory=validation_batch_factory,
            validation_populations=validation_populations,
            validation_stratum_weights=validation_stratum_weights,
            phase_timing=phase_timing,
        )
        phase_timing.mark("terminal_evaluation_end")
        # Close the per-step stack after every controller outcome has been
        # resolved at its commit/no-commit branch, bind the accepted-only sidecar
        # bytes to their sha256, and build the additive receipt block.
        # Mode OFF (or a non-v0 arm) records {enabled: false} and nothing else.
        if run_product_writer is None:
            (
                step_category_structure_block,
                step_category_structure_sidecars,
            ) = _step_category_structure_finalize(
                enabled=bool(step_category_matrix_effective),
                provider=decision_provider,
                setup_receipt=step_category_setup_receipt,
                accepted_trajectory=accepted_v0_trajectory,
            )
        else:
            (
                step_category_structure_block,
                step_category_structure_sidecars,
            ) = run_product_writer.deferred_result()
        _step_category_structure_runtime_telemetry(step_category_structure_block, decision_provider)
        (
            step_context_delta_block,
            step_context_delta_sidecars,
        ) = _step_context_delta_finalize(step_context_delta_accumulator)
        receipt_finalization_started = time.perf_counter()
        retained = final_mask.retained_count
        if (
            canonical_decision_provider is not None
            and dense_readout_table_cache_paths is not None
        ):
            import fcntl

            payload_path, receipt_path, lock_path = dense_readout_table_cache_paths
            cache_key = str(dense_readout_table_telemetry["cache_key"])
            (
                exported_source_context_ids,
                exported_projected_values,
                exported_labels,
            ) = canonical_decision_provider.export_projected_readouts(dense_mask.mask_hash)
            _progress(
                "dense_readout_table_cache_store_waiting",
                cache_key=cache_key,
                lock_path=str(lock_path),
            )
            if os.environ.get("IRG_DENSE_READOUT_TABLE_STORE", "1").strip() == "0":
                # Skip persistence when the caller disables this cache store.
                dense_readout_table_telemetry["stored_rows"] = 0
                dense_readout_table_telemetry["store_skipped"] = (
                    "IRG_DENSE_READOUT_TABLE_STORE=0"
                )
                _progress(
                    "dense_readout_table_cache_store_skipped",
                    cache_key=cache_key,
                )
            else:
                with lock_path.open("a+b") as lock_stream:
                    fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
                    try:
                        current_dense_readout_table = _load_dense_readout_table_cache(
                            payload_path=payload_path,
                            receipt_path=receipt_path,
                            cache_key=cache_key,
                            dense_mask_hash=dense_mask.mask_hash,
                            context_axis_hash=optimization_axis.content_hash,
                            forward_tier=int(
                                runtime.evaluator.runtime_execution_profile.forward_tier
                            ),
                            live_source_context_ids=dense_readout_table_live_source_ids,
                            labels_by_context_id=(dense_readout_table_labels_by_context_id),
                        )
                        current_is_refused_payload = bool(
                            current_dense_readout_table is not None
                            and dense_readout_table_refused_payload_sha256 is not None
                            and current_dense_readout_table.payload_sha256
                            == dense_readout_table_refused_payload_sha256
                        )
                        should_store = bool(
                            current_dense_readout_table is None
                            or not current_dense_readout_table.valid
                            or current_is_refused_payload
                            or len(exported_source_context_ids)
                            > current_dense_readout_table.row_count
                        )
                        if should_store:
                            _store_dense_readout_table_cache(
                                payload_path=payload_path,
                                receipt_path=receipt_path,
                                cache_key=cache_key,
                                source_context_ids=exported_source_context_ids,
                                projected_values=exported_projected_values,
                                labels=exported_labels,
                                dense_mask_hash=dense_mask.mask_hash,
                                context_axis_hash=optimization_axis.content_hash,
                                forward_tier=int(
                                    runtime.evaluator.runtime_execution_profile.forward_tier
                                ),
                            )
                            stored_rows = len(exported_source_context_ids)
                        else:
                            stored_rows = current_dense_readout_table.row_count
                        dense_readout_table_telemetry["stored_rows"] = int(stored_rows)
                    finally:
                        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        elif (
            canonical_decision_provider is not None
            and dense_readout_table_cache_dir is not None
            and canonical_dense_table_active
        ):
            # The canonical table is the authoritative reference and already
            # supplies these readouts, so the (potentially TF32-tainted) live
            # slab is not exported to the per-arm table.
            dense_readout_table_telemetry["store_skipped"] = "canonical_dense_table_active"
            _progress(
                "dense_readout_table_cache_store_skipped",
                reason="canonical_dense_table_active",
            )
        # Both step-category tensors and the dense readout slab are now on the
        # host. Every remaining write runs inside the boundary so a failed output
        # or receipt write removes the marker.
        with _gpu_done_marker_boundary(
            destination, run_id=shared_run_id, mask_hash=final_mask.mask_hash,
            device=runtime.evaluator.device, phase_timing=phase_timing,
            resolved_settings=_run_effective_settings(runtime),
        ):
            performance_gate_receipt: Mapping[str, Any] | None = None
            gate_settings = _run_effective_settings(runtime)
            gate_plane_settings = _runtime_data_plane_effective_settings(runtime)
            gate_panel_eligible = runtime.optimization_data_plane.performance_gate_eligible
            gate_device_name = _device_name(device)
            def _write_gate_before_receipt(receipt_payload: Mapping[str, Any]) -> None:
                nonlocal performance_gate_receipt
                if version != "v0":
                    return
                assert v0_scoring_anchor_elapsed is not None
                assert v0_traversal_elapsed is not None
                assert v0_traversal_complete is not None
                performance_gate_receipt = _v0_performance_gate_receipt(
                    resolved_settings=gate_settings,
                    data_plane_settings=gate_plane_settings,
                    pilot_receipt_hash=str(receipt_payload["receipt_hash"]),
                    architecture_family=runtime_architecture_family,
                    device_name=gate_device_name,
                    scoring_and_anchor_seconds=v0_scoring_anchor_elapsed,
                    traversal_seconds=v0_traversal_elapsed,
                    receipt_finalization_seconds=(
                        time.perf_counter() - receipt_finalization_started
                    ),
                    traversal_complete=v0_traversal_complete,
                    optimized_panel_eligible=(
                        gate_panel_eligible
                    ),
                )
                (destination / "performance_gate.json").write_text(
                    json.dumps(performance_gate_receipt, indent=2, sort_keys=True, default=str)
                    + "\n",
                    encoding="utf-8",
                )
                with (destination / "performance_gate.json").open("rb") as stream:
                    os.fsync(stream.fileno())

            before_receipt = _write_gate_before_receipt
            payload = _write_run_receipt(
                calibration_deployment_refit_anchor=calibration_deployment_refit_anchor,
                canonical_dense_baseline_bce=canonical_dense_baseline_bce,
                canonical_dense_certificate=canonical_dense_certificate,
                canonical_dense_table_active=canonical_dense_table_active,
                canonical_dense_table_receipt_path=canonical_dense_table_receipt_path,
                cold_runtime_setup_started=cold_runtime_setup_started,
                comparison_validation_mode=comparison_validation_mode,
                decision_oracle_sampling=decision_oracle_sampling,
                decision_oracle_sampling_receipts=decision_oracle_sampling_receipts,
                decision_provider=decision_provider,
                decision_sampling=decision_sampling,
                dense_baseline_mode=dense_baseline_mode,
                dense_mask=dense_mask,
                dense_readout_table_telemetry=dense_readout_table_telemetry,
                dense_terminal=dense_terminal,
                dense_terminal_substituted=dense_terminal_substituted,
                destination=destination,
                evaluation_axis=evaluation_axis,
                extraction_gpu_activity_receipt=extraction_gpu_activity_receipt,
                family=family,
                final_mask=final_mask,
                final_terminal=final_terminal,
                final_terminal_substituted=final_terminal_substituted,
                fit_ids_at_terminal_lock=fit_ids_at_terminal_lock,
                fit_ids_before_terminal=fit_ids_before_terminal,
                historical_mask=historical_mask,
                historical_mask_alias_receipt=historical_mask_alias_receipt,
                historical_mask_sha256=historical_mask_sha256,
                historical_terminal=historical_terminal,
                historical_terminal_substituted=historical_terminal_substituted,
                maximum_relative_bce_increase=maximum_relative_bce_increase,
                model_transform=model_transform,
                optimization_axis=optimization_axis,
                partition_random_seed=partition_random_seed,
                population_calibration_receipt_path=population_calibration_receipt_path,
                profiler=profiler,
                query_id=query_id,
                requested_family=requested_family,
                residual_observation_path_counts=residual_observation_path_counts,
                resolved_sensitivity_tau=resolved_sensitivity_tau,
                retained=retained,
                route_axis_hash=route_axis_hash,
                route_count=route_count,
                runner_prewarm_receipt=runner_prewarm_receipt,
                runtime=runtime,
                sample_size_policy=sample_size_policy,
                step_category_structure_block=step_category_structure_block,
                step_category_structure_sidecars=step_category_structure_sidecars,
                step_context_delta_block=step_context_delta_block,
                step_context_delta_sidecars=step_context_delta_sidecars,
                step_observables_effective=step_observables_effective,
                target_class_id=target_class_id,
                terminal_audit_elapsed=terminal_audit_elapsed,
                terminal_calibration_closure_receipts=terminal_calibration_closure_receipts,
                terminal_category_structure_block=terminal_category_structure_block,
                terminal_category_structure_sidecars=terminal_category_structure_sidecars,
                terminal_namespace_salt=terminal_namespace_salt,
                terminal_provider=terminal_provider,
                terminal_validation_one_pass=terminal_validation_one_pass,
                unit_gate_started=unit_gate_started,
                v0_accept_replication_gate=v0_accept_replication_gate,
                v0_calibration_decision_estimand=v0_calibration_decision_estimand,
                v0_oracle_minimum_per_stratum=v0_oracle_minimum_per_stratum,
                version=version,
                workflow_details=workflow_details,
                before_receipt=before_receipt,
                output_submit=output_submit,
            )
            if output_submit is not None:
                return payload
            _progress(
                "pilot_completed",
                receipt_hash=payload["receipt_hash"],
                retained_fraction=payload["retained_fraction"],
            )
            return {
                **payload,
                "output_dir": str(destination),
                "performance_gate_receipt": performance_gate_receipt,
            }


CLOSURE_DIAGNOSTIC_SCHEMA = "registered_resnet18_calibration_closure_diagnostic_v1"


def _certificate_interval(certificate: Any) -> dict[str, float]:
    return {
        "estimate": float(certificate.estimate),
        "lower": float(certificate.lower),
        "upper": float(certificate.upper),
        "context_count": int(certificate.context_count),
        "content_hash": str(certificate.content_hash),
    }


def _calibration_fit_identity(fit: Any) -> dict[str, Any]:
    return {
        "calibration_state_id": str(fit.calibration_state_id),
        "calibration_context_count": int(sum(fit.sampled_stratum_counts.values())),
        "maximum_attainable_bce_improvement": float(fit.maximum_bce_excess_risk_bound),
        "first_order_excess_risk_bound": float(
            getattr(fit, "first_order_excess_risk_bound", 0.0)
        ),
        "second_order_excess_risk_bound": float(
            getattr(fit, "second_order_excess_risk_bound", 0.0)
        ),
        "affine_scale": float(fit.scale),
        "affine_offset": float(fit.offset),
        "affine_input_center": float(fit.input_center),
    }


def _closure_diagnostic_leg(
    *,
    provider: NativeTorchCalibratedBCEOracleProvider,
    mask: HardMask,
    mask_role: str,
    confidence_level: float,
    cumulative_risk_context_count: int,
    certificate_role: str,
    stream_namespace: str,
) -> dict[str, Any]:
    """Measure one locked mask's risk before and after calibration closure.

    The pre-closure certificate is the deployed-state readout the registered
    frozen-calibration policy would report.  The closure then refines only the
    extraction-training affine fit under the risk-resolution stopping rule, so
    the pre/post difference is exactly the attainable-improvement recovery the
    forward-work hypothesis predicts.  No validation context participates.
    """

    leg_started = time.perf_counter()
    pre_certificate = provider.certify_baseline_risk(
        mask,
        cumulative_context_count=int(cumulative_risk_context_count),
        confidence_level=float(confidence_level),
        maximum_planned_looks=None,
        precision_only_iid=True,
        certificate_role=str(certificate_role),
        stream_namespace=str(stream_namespace),
    )
    pre_fit = provider.calibration_fits[mask.mask_hash]
    predicted_recovery_bound = float(pre_fit.maximum_bce_excess_risk_bound)
    pre_risk_width = float(pre_certificate.upper) - float(pre_certificate.lower)
    if pre_risk_width <= 0.0 and predicted_recovery_bound > (
        64.0
        * float(np.finfo(np.float64).eps)
        * max(1.0, abs(float(pre_certificate.estimate)))
    ):
        # A collapsed (constant-prediction) fit on an uninformative readout
        # yields a deterministic loss sample while still reporting a material
        # local improvement bound.  Closure cannot resolve that contradiction
        # with more rows; record the degenerate state instead of failing the
        # other legs.  Terminal evaluation still reads the frozen fit.
        post_certificate = pre_certificate
        closure_receipt: Mapping[str, Any] = {
            "schema": "validation_independent_terminal_calibration_closure_v1",
            "mask_hash": mask.mask_hash,
            "certificate_role": str(certificate_role),
            "status": "skipped_degenerate_zero_width_risk_evidence",
            "stopping_resolution": None,
            "validation_contexts_observed": False,
        }
    else:
        post_certificate, closure_receipt = (
            _close_locked_mask_calibration_to_risk_resolution(
                provider=provider,
                mask=mask,
                confidence_level=float(confidence_level),
                cumulative_risk_context_count=int(cumulative_risk_context_count),
                certificate=pre_certificate,
                certificate_role=str(certificate_role),
                stream_namespace=str(stream_namespace),
                progress_callback=_progress,
            )
        )
    post_fit = provider.calibration_fits[mask.mask_hash]
    realized_training_recovery = float(pre_certificate.estimate) - float(
        post_certificate.estimate
    )
    return {
        "mask_role": str(mask_role),
        "mask_hash": mask.mask_hash,
        "retained_fraction": (float(mask.retained_count) / float(mask.route_count)),
        "certificate_role": str(certificate_role),
        "stream_namespace": str(stream_namespace),
        "pre_closure_risk": _certificate_interval(pre_certificate),
        "pre_closure_calibration": _calibration_fit_identity(pre_fit),
        "predicted_recovery_bound": predicted_recovery_bound,
        "closure_receipt": dict(closure_receipt),
        "post_closure_risk": _certificate_interval(post_certificate),
        "post_closure_calibration": _calibration_fit_identity(post_fit),
        "realized_training_risk_recovery": realized_training_recovery,
        "elapsed_seconds": time.perf_counter() - leg_started,
    }


def run_registered_resnet18_calibration_closure_diagnostic(
    *,
    study: StudySpec,
    plan: CircuitPlan,
    profile: Any,
    promotion_receipt: Any | None,
    query_id: str,
    comparison_namespace: str,
    optimization_axis: Any,
    evaluation_axis: Any,
    optimization_image_root: str | Path,
    evaluation_image_root: str | Path,
    store: ArtifactStore,
    output_dir: str | Path,
    final_mask_packbits_path: str | Path,
    expected_final_mask_hash: str | None = None,
    historical_mask_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    model_source_replica: ModelSourceReplica | None = None,
    optimization_uint8_cache_manifest: str | Path | None = None,
    optimization_uint8_cache_receipt: str | Path | None = None,
    evaluation_uint8_cache_manifest: str | Path | None = None,
    evaluation_uint8_cache_receipt: str | Path | None = None,
    device: Any = "cuda:0",
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool = True,
    random_seed: int = 0,
    terminal_maximum_per_stratum: int = 256,
    sample_size_policy: IndependentSampleSizePolicy = IndependentSampleSizePolicy(),
    reference_terminal_bce: Mapping[str, float] | None = None,
    count_matched_probe_seed: int | None = None,
    probe_permutation_fraction: float | None = None,
    calibration_deployment_refit_anchor: str = "selected",
) -> Mapping[str, Any]:
    """Audit already-locked masks under risk-resolution calibration closure.

    This is validation-independent falsification for the calibration-under-resolution
    hypothesis: it never proposes, scores, or changes a physical mask.  It
    reproduces the registered v0 decision provider on the identical shared run
    identity, reads each locked mask's deployed-state risk under the cheap
    pre-closure fit, closes each fit on independent extraction-training
    streams, and only then materializes the sealed terminal panel for an
    import-only readout of dense, final-modern, and historical masks.
    """

    if not str(comparison_namespace):
        raise ValueError("the diagnostic requires a shared comparison namespace")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"immutable diagnostic output exists: {destination}")
    sample_size_policy = _registered_precision_policy(
        "v0",
        sample_size_policy,
    )
    workflow_started = time.perf_counter()

    from mechanistic_systems.backends.profiling import PhaseProfiler

    _progress(
        "closure_diagnostic_start",
        output_dir=str(destination),
    )
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
        allow_download=False,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        loaded_model=None,
        model_source_replica=model_source_replica,
        queries=None,
    )
    runtime.evaluator.candidate_execution_backend = prefix_reuse_backend(
        runtime.evaluator.candidate_execution_backend
    )
    if runtime.loaded.receipt.architecture_family.lower() not in (
        _REGISTERED_PILOT_ARCHITECTURE_FAMILIES
    ):
        raise ValueError(
            "this diagnostic surface is locked to the registered families "
            f"{sorted(_REGISTERED_PILOT_ARCHITECTURE_FAMILIES)!r}"
        )
    query_by_id = {query.query_id: query for query in runtime.query_axis}
    if str(query_id) not in query_by_id:
        raise ValueError("diagnostic query is outside the measurement artifact")
    query = query_by_id[str(query_id)]
    target_class_id = _target_class_id(query)
    route_axis_hash = str(plan.route_universe_hash)
    route_count = sum(
        int(np.prod(group.shape, dtype=np.int64)) for group in runtime.route_groups
    )
    dense_mask = HardMask.dense(route_count, route_axis_hash=route_axis_hash)
    if runtime.evaluator.device.type == "cuda":
        _prewarm_registered_mask_runners(
            runtime=runtime,
            dense_mask=dense_mask,
            target_class_id=target_class_id,
        )
    _refuse_transformed_arm_mask_source(final_mask_packbits_path, runtime=runtime)
    final_mask_bytes = Path(final_mask_packbits_path).read_bytes()
    final_mask = HardMask(
        route_axis_hash=route_axis_hash,
        route_count=route_count,
        packed_bits=final_mask_bytes,
    )
    if expected_final_mask_hash is not None and final_mask.mask_hash != str(
        expected_final_mask_hash
    ):
        raise ValueError(
            "final mask bytes do not reproduce the expected mask hash: "
            f"{final_mask.mask_hash} != {expected_final_mask_hash}"
        )
    if final_mask.mask_hash == dense_mask.mask_hash:
        raise ValueError(
            "the diagnostic requires a pruned final mask distinct from dense"
        )
    audited_mask = final_mask
    audited_mask_role = "final_modern"
    probe_layer_retained_counts: dict[str, int] | None = None
    if count_matched_probe_seed is not None:
        # A layer-count-matched randomized control: permute the selected
        # mask's retained pattern within every layer segment so per-layer
        # budgets match exactly while route-level selection is destroyed.
        # A fractional dose permutes only that share of each layer's routes,
        # giving the dose-response curve that separates selection-specific
        # validation fragility from sparsity-conditional distribution shift
        # (the full permutation is dead at chance, so only partial doses are
        # informative).
        resolved_probe_fraction = (
            1.0
            if probe_permutation_fraction is None
            else float(probe_permutation_fraction)
        )
        if not 0.0 < resolved_probe_fraction <= 1.0:
            raise ValueError("probe permutation fraction must lie in (0, 1]")
        generator = np.random.Generator(np.random.PCG64(int(count_matched_probe_seed)))
        selected_values = final_mask.to_bool_array()
        probe_values = np.empty_like(selected_values)
        probe_layer_retained_counts = {}
        # Declared-axis segmentation: ``final_mask.packbits`` lives on the
        # declared route-block axis (the ``_hard_mask_from_layers`` producer
        # convention), so every per-layer permutation window must be that
        # layer's OWN declared offsets; the authority validates coverage of
        # the probe mask route axis and raises on any mismatch.
        probe_axis_offsets = _declared_route_axis_offsets(
            runtime,
            flat_length=route_count,
        )
        for layer_id, (start, stop) in probe_axis_offsets.items():
            size = stop - start
            segment = selected_values[start:stop]
            if resolved_probe_fraction >= 1.0:
                permuted = generator.permutation(segment)
            else:
                permuted = segment.copy()
                subset_size = int(round(resolved_probe_fraction * size))
                if subset_size >= 2:
                    chosen = generator.choice(
                        size,
                        size=subset_size,
                        replace=False,
                    )
                    permuted[chosen] = generator.permutation(segment[chosen])
            probe_values[start:stop] = permuted
            probe_layer_retained_counts[str(layer_id)] = int(segment.sum())
        audited_mask = HardMask.from_bool_array(
            probe_values,
            route_axis_hash=route_axis_hash,
        )
        audited_mask_role = "count_matched_random_probe"
        if audited_mask.retained_count != final_mask.retained_count:
            raise ValueError(
                "count-matched probe changed the retained connection count"
            )
    historical_mask: HardMask | None = None
    historical_mask_sha256: str | None = None
    if historical_mask_path is not None:
        historical_mask, historical_mask_sha256, _ = _load_historical_hard_mask(
            historical_mask_path,
            runtime=runtime,
            route_axis_hash=route_axis_hash,
        )
    shared_run_id = content_hash(
        {
            "comparison_namespace": str(comparison_namespace),
            "query_id": str(query_id),
            "optimization_axis_hash": optimization_axis.content_hash,
            "random_seed": int(random_seed),
        },
        prefix="irg-matched-pilot-",
    )
    decision_namespace = binary_evidence_role_namespace(
        run_id=shared_run_id,
        random_seed=random_seed,
        context_axis_hash=optimization_axis.content_hash,
    )
    decision_sampling = compile_binary_iid_sampling_design(
        optimization_axis,
        target_class_id=target_class_id,
        randomization_namespace=decision_namespace,
    )
    decision_oracle_sampling = _decision_oracle_sampling_design(
        optimization_axis,
        target_class_id=target_class_id,
        randomization_namespace=f"{decision_namespace}:class-resolved-oracle",
        oracle_sampling_law=str(sample_size_policy.oracle_sampling_law),
    )
    mask_registry: dict[str, HardMask] = {
        dense_mask.mask_hash: dense_mask,
        audited_mask.mask_hash: audited_mask,
    }
    if historical_mask is not None:
        mask_registry[historical_mask.mask_hash] = historical_mask
    provider = _fixed_calibration_bce_provider(
        runtime=runtime,
        optimization_axis=optimization_axis,
        target_class_id=target_class_id,
        maximum_bce_increment=None,
        run_id=shared_run_id,
        random_seed=random_seed,
        mask_resolver=(lambda mask_hash: mask_registry[str(mask_hash)]),
        candidate_resolver=(
            lambda candidate_id: (_ for _ in ()).throw(KeyError(candidate_id))
        ),
        sample_size_policy=sample_size_policy,
        iid_sampling_design=decision_sampling,
        oracle_iid_sampling_design=decision_oracle_sampling,
        oracle_panel_role="v0_paired_oracle",
        progress_callback=_progress,
        calibration_deployment_refit_anchor=(calibration_deployment_refit_anchor),
    )
    dense_resolution_spec = SequentialCalibratedBCEResolutionSpec(
        **({"stratum_weights": tuple(sorted(_binary_loss_weights(
            decision_sampling.source_context_ids_by_stratum).items()))}
           if loss_weighting() == "per_image" else {}),
        confidence_level=sample_size_policy.confidence_level,
        estimand_policy=V0_NORMALIZED_LOSS_SPAN_PER_CONNECTION_V1,
        maximum_planned_looks=None,
        precision_only_iid=True,
    )
    dense_risk_context_count = max(
        2 * int(provider.oracle_minimum_per_stratum),
        int(sample_size_policy.oracle_initial_packet),
    )
    terminal_risk_context_count = max(
        int(sample_size_policy.oracle_initial_packet),
        2 * int(terminal_maximum_per_stratum),
        2 * int(provider.oracle_minimum_per_stratum),
    )
    legs: list[dict[str, Any]] = []
    legs.append(
        _closure_diagnostic_leg(
            provider=provider,
            mask=dense_mask,
            mask_role="dense",
            confidence_level=(dense_resolution_spec.v0_anchor_confidence_level),
            cumulative_risk_context_count=dense_risk_context_count,
            certificate_role=("dense_baseline_proposal_adjudication_reference"),
            stream_namespace="dense-baseline",
        )
    )
    legs.append(
        _closure_diagnostic_leg(
            provider=provider,
            mask=audited_mask,
            mask_role=audited_mask_role,
            confidence_level=float(sample_size_policy.confidence_level),
            cumulative_risk_context_count=terminal_risk_context_count,
            certificate_role=("terminal_final_mask_training_risk_resolution_reference"),
            stream_namespace="terminal-final-calibration-resolution",
        )
    )
    if historical_mask is not None:
        legs.append(
            _closure_diagnostic_leg(
                provider=provider,
                mask=historical_mask,
                mask_role="historical",
                confidence_level=float(sample_size_policy.confidence_level),
                cumulative_risk_context_count=terminal_risk_context_count,
                certificate_role=(
                    "historical_comparator_training_risk_resolution_reference"
                ),
                stream_namespace=("terminal-historical-calibration-resolution"),
            )
        )
    fit_ids_at_terminal_lock = {
        mask_hash: fit.calibration_state_id
        for mask_hash, fit in provider.calibration_fits.items()
    }
    # Validation is materialized only after every extraction calibration state
    # is locked, exactly as in the registered pilot's terminal audit.
    validation_populations, validation_stratum_weights = _sealed_terminal_populations(
        evaluation_axis,
        target_class_id=target_class_id,
        namespace=shared_run_id,
        maximum_per_stratum=terminal_maximum_per_stratum,
    )
    validation_batch_factory = runtime.evaluation_data_plane.role_batch_factory(
        "terminal"
    )
    _progress("closure_diagnostic_terminal_evaluation_started")
    terminal_bce: dict[str, Mapping[str, Any]] = {}
    for leg in legs:
        mask = mask_registry[leg["mask_hash"]]
        terminal_bce[leg["mask_role"]] = provider.evaluate_frozen_mask_bce(
            mask,
            evaluation_context_ids_by_stratum=validation_populations,
            evaluation_stratum_weights=validation_stratum_weights,
            batch_factory=validation_batch_factory,
            require_precalibrated=True,
        )
    _progress("closure_diagnostic_terminal_evaluation_completed")
    fit_ids_after_validation = {
        mask_hash: fit.calibration_state_id
        for mask_hash, fit in provider.calibration_fits.items()
    }
    if fit_ids_after_validation != fit_ids_at_terminal_lock:
        raise RuntimeError("terminal validation mutated a locked calibration state")
    reference_comparison: dict[str, Any] | None = None
    if reference_terminal_bce is not None:
        reference_comparison = {}
        for leg in legs:
            role = leg["mask_role"]
            if role not in reference_terminal_bce:
                continue
            reference_value = float(reference_terminal_bce[role])
            diagnostic_value = float(terminal_bce[role]["per_image_binary_bce" if loss_weighting() == "per_image" else "class_balanced_binary_bce"])
            reference_comparison[role] = {
                "reference_terminal_bce": reference_value,
                "diagnostic_terminal_bce": diagnostic_value,
                "realized_validation_recovery": (reference_value - diagnostic_value),
                "predicted_recovery_bound": float(leg["predicted_recovery_bound"]),
                # The reference run's deployed fit may differ in lineage and
                # selected face from this diagnostic's fresh fit on the same
                # stream identity, so this difference mixes closure effect
                # with fit-identity spread; it is not a pure closure effect.
                "recovery_includes_fit_identity_spread": True,
            }
    payload: dict[str, Any] = {
        "schema": CLOSURE_DIAGNOSTIC_SCHEMA,
        "comparison_namespace": str(comparison_namespace),
        "query_id": str(query_id),
        "target_class_id": int(target_class_id),
        "shared_run_id": shared_run_id,
        "decision_namespace": decision_namespace,
        "optimization_axis_hash": optimization_axis.content_hash,
        "evaluation_axis_hash": evaluation_axis.content_hash,
        "route_axis_hash": route_axis_hash,
        "route_count": int(route_count),
        "random_seed": int(random_seed),
        "confidence_level": float(sample_size_policy.confidence_level),
        "calibration_deployment_refit_anchor": str(calibration_deployment_refit_anchor),
        "terminal_maximum_per_stratum": int(terminal_maximum_per_stratum),
        "final_mask_packbits_path": str(final_mask_packbits_path),
        "final_mask_hash": final_mask.mask_hash,
        "audited_mask_role": audited_mask_role,
        "audited_mask_hash": audited_mask.mask_hash,
        "count_matched_probe_seed": (
            None if count_matched_probe_seed is None else int(count_matched_probe_seed)
        ),
        "probe_permutation_fraction": (
            None
            if count_matched_probe_seed is None
            else (
                1.0
                if probe_permutation_fraction is None
                else float(probe_permutation_fraction)
            )
        ),
        "probe_layer_retained_counts": probe_layer_retained_counts,
        "historical_mask_path": (
            None if historical_mask_path is None else str(historical_mask_path)
        ),
        "historical_mask_sha256": historical_mask_sha256,
        "validation_contexts_observed_before_closure": False,
        "closure_legs": legs,
        "terminal_bce": {role: dict(result) for role, result in terminal_bce.items()},
        "reference_comparison": reference_comparison,
        "numerics_record": numerics_record(),
        "elapsed_seconds": time.perf_counter() - workflow_started,
    }
    nonfinite_paths = _nonfinite_receipt_paths(payload)
    if nonfinite_paths:
        raise ValueError(
            "closure diagnostic receipt contains non-finite values at "
            + ", ".join(nonfinite_paths[:16])
        )
    payload["effective_settings"] = _run_effective_settings(runtime)
    payload["data_plane_effective_settings"] = _runtime_data_plane_effective_settings(runtime)
    payload["receipt_hash"] = runtime_receipt_hash(
        payload,
        prefix="irg-resnet18-closure-diagnostic-",
    )
    # Immutability is owned by the entry-time exists() refusal; the receipt
    # write must tolerate a destination created during the run by launch
    # tooling.
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "receipt.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    _progress(
        "closure_diagnostic_completed",
        receipt_hash=payload["receipt_hash"],
    )
    return {**payload, "output_dir": str(destination)}


__all__ = [
    "PILOT_SCHEMA",
    "V0_ATTRIBUTION_ORDER_MODES",
    "V0_DEFAULT_ATTRIBUTION_ORDER",
    "CLOSURE_DIAGNOSTIC_SCHEMA",
    "_alexnet_fully_pruned_constant_response_proof",
    "_fully_pruned_constant_response_proof",
    "_robust_resnet50_conv_kernel_weight_fully_pruned_constant_response_proof",
    "_resnet50_conv_kernel_weight_fully_pruned_constant_response_proof",
    "_resnet18_conv_kernel_weight_fully_pruned_constant_response_proof",
    "_resnet18_fully_pruned_constant_response_proof",
    "_vgg11_conv_kernel_weight_fully_pruned_constant_response_proof",
    "run_registered_resnet18_pilot",
    "run_registered_resnet18_calibration_closure_diagnostic",
]
