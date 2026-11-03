"""Torch execution backend for the canonical sampled-path experiment.

The backend deliberately separates panel allocation, exact masked execution,
off-fold calibration, and held-out loss construction.  A committed/candidate
pair is always evaluated on identical ordered contexts before either result is
used by the sequential decision rule.
"""

from __future__ import annotations

import hashlib
import math
import os
import queue
import threading
import time
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol

import numpy as np
from mechanistic_systems.measurement.lazy_attribute import LazyAttribute

minimize = LazyAttribute("scipy.optimize", "minimize")

from mechanistic_systems.backends.calibration_backend import (
    CanonicalCalibrationBackend,
)
from mechanistic_systems.backends.calibration_runtime import (
    RuntimeExecutionSession,
    RuntimeExecutionProfile,
)
from mechanistic_systems.backends.profiling import PhaseProfiler
from mechanistic_systems.contracts.calibration_compilation import (
    coordinate_calibration_contracts,
    coordinate_calibration_contracts as _coordinate_calibration_contracts,
)
from mechanistic_systems.circuits.glocal_sampled_path import (
    FrozenBrierCalibration,
    PairedBrierEvidence,
    request_cumulative_target_images,
    request_maximum_new_contexts,
    request_per_stratum_target_images,
)
from mechanistic_systems.circuits.sampled_path import (
    CrossfitPanel,
    PairedCandidateEvidence,
    SampledReferenceEvidence,
    SampledComparisonRequest,
    fold_warm_start,
    sampled_path_mask_hash,
)
from mechanistic_systems.circuits.neyman import (
    BinaryCostOptimalNeymanPanelPlanner,
    CostOptimalNeymanPolicy,
    NeymanEvidenceState,
)
from mechanistic_systems.circuits.torch_evaluator import (
    ReadoutResponse,
    TorchCircuitEvaluator,
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
    CalibrationProblemSpec,
    PROPER_SCORE_SHRINKAGE_AFFINE_V2,
    QueryOperatorSpec,
    QuerySpec,
    ReadoutProjectionSpec,
    SAMPLED_PATH_V1_CALIBRATION_ALGORITHM,
)
from mechanistic_systems.measurement.contexts import ContextAxis
from mechanistic_systems.measurement.torch_adapter import (
    MeasurementInputBatch,
)

try:  # Optional for import smoke.
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - environment dependent.
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


DEFAULT_RESPONSE_CACHE_MAX_BYTES = 512 * 1024 * 1024
GLOCAL_RUNTIME_RESOURCE_SCHEMA = "mechanistic_glocal_runtime_resources_v1"
GLOCAL_RUNTIME_CALL_RECEIPT_SCHEMA = "mechanistic_glocal_runtime_call_receipt_v1"
_COMPILED_MASK_CACHE_CAPACITY = 6


# Mirrors _BoundedCompiledMaskCache in
# mechanistic_systems/measurement/native_irg_providers.py; importing it would
# create a sampled_torch<->native_irg_providers import cycle.
class _BoundedCompiledMaskCache:
    """Small LRU over compiled device masks."""

    def __init__(self, capacity: int = _COMPILED_MASK_CACHE_CAPACITY) -> None:
        if int(capacity) < 2:
            raise ValueError("compiled-mask cache must hold parent plus candidate")
        self._capacity = int(capacity)
        self._entries: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str) -> Any:
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
        return entry

    def put(self, key: str, value: Any) -> None:
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self._capacity:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)


@dataclass(frozen=True)
class GlocalRuntimeSnapshot:
    """Runtime-only cumulative counters at one claim-mode call boundary."""

    unique_contexts: int
    unique_decision_contexts: int
    unique_context_state_requests: int
    unique_context_state_evaluations: int
    high_level_forward_calls: int
    failed_high_level_forward_calls: int
    physical_forward_calls: int
    physical_forward_calls_observed: bool
    backward_calls: int
    backward_calls_observed: bool
    host_to_device_bytes: int
    device_to_host_bytes: int
    calibration_fit_calls: int
    glocal_calls: int
    glocal_wall_seconds: float
    glocal_cuda_device_seconds: float
    absolute_peak_device_memory_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "unique_contexts": int(self.unique_contexts),
            "unique_decision_contexts": int(self.unique_decision_contexts),
            "unique_context_state_requests": int(self.unique_context_state_requests),
            "unique_context_state_evaluations": int(
                self.unique_context_state_evaluations
            ),
            "high_level_forward_calls": int(self.high_level_forward_calls),
            "failed_high_level_forward_calls": int(
                self.failed_high_level_forward_calls
            ),
            "physical_forward_calls": int(self.physical_forward_calls),
            "physical_forward_calls_observed": bool(
                self.physical_forward_calls_observed
            ),
            "backward_calls": int(self.backward_calls),
            "backward_calls_observed": bool(self.backward_calls_observed),
            "host_to_device_bytes": int(self.host_to_device_bytes),
            "device_to_host_bytes": int(self.device_to_host_bytes),
            "calibration_fit_calls": int(self.calibration_fit_calls),
            "glocal_calls": int(self.glocal_calls),
            "glocal_wall_seconds": float(self.glocal_wall_seconds),
            "glocal_cuda_device_seconds": float(self.glocal_cuda_device_seconds),
            "absolute_peak_device_memory_bytes": int(
                self.absolute_peak_device_memory_bytes
            ),
        }

    def delta_from(self, prior: "GlocalRuntimeSnapshot") -> dict[str, Any]:
        """Return additive deltas while preserving observation qualifiers."""

        current = self.to_dict()
        previous = prior.to_dict()
        result: dict[str, Any] = {}
        qualifiers = {
            "physical_forward_calls_observed",
            "backward_calls_observed",
        }
        for key, value in current.items():
            if key in qualifiers:
                result[key] = bool(value)
            elif key == "absolute_peak_device_memory_bytes":
                # A peak is an absolute high-water mark, not additive usage.
                result[key] = int(value)
            else:
                result[key] = value - previous[key]
        return result


@dataclass(frozen=True)
class GlocalRuntimeCallReceipt:
    """Operational receipt excluded from calibration/evidence identities."""

    operation: str
    query_id: str
    role: str
    proposal_index: int
    stage_index: int
    panel_hash: str
    succeeded: bool
    error_type: str | None
    before: GlocalRuntimeSnapshot
    after: GlocalRuntimeSnapshot
    schema_name: str = GLOCAL_RUNTIME_CALL_RECEIPT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "operation": self.operation,
            "query_id": self.query_id,
            "role": self.role,
            "proposal_index": int(self.proposal_index),
            "stage_index": int(self.stage_index),
            "panel_hash": self.panel_hash,
            "succeeded": bool(self.succeeded),
            "error_type": self.error_type,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "delta": self.after.delta_from(self.before),
        }


@dataclass(frozen=True)
class GlocalCalibrationLogitRows:
    """Exact role-bound logits over one sealed calibration prefix."""

    query_id: str
    role: str
    mask_hash: str
    panel: CrossfitPanel
    logits: np.ndarray
    binary_labels: np.ndarray
    positive_class_id: int

    def __post_init__(self) -> None:
        logits = np.asarray(self.logits, dtype=np.float64).reshape(-1).copy()
        labels = np.asarray(self.binary_labels, dtype=np.uint8).reshape(-1).copy()
        if (
            not self.query_id
            or self.role != "calibration_fit"
            or len(self.mask_hash) != 64
            or logits.size < 1
            or labels.size != logits.size
            or logits.size != len(self.panel.context_ids)
            or not np.isfinite(logits).all()
            or not np.all(np.isin(labels, (0, 1)))
        ):
            raise ValueError("glocal calibration-logit rows are invalid")
        if len(self.panel.fold_ids) != len(self.panel.context_ids) or any(
            int(fold) < 0 for fold in self.panel.fold_ids
        ):
            raise ValueError("glocal calibration-logit rows have an invalid fold axis")
        logits.setflags(write=False)
        labels.setflags(write=False)
        object.__setattr__(self, "logits", logits)
        object.__setattr__(self, "binary_labels", labels)
        object.__setattr__(self, "positive_class_id", int(self.positive_class_id))

    @property
    def evidence_hash(self) -> str:
        digest = hashlib.sha256()
        for value in (
            self.query_id,
            self.role,
            self.mask_hash,
            self.panel.panel_hash,
            str(self.positive_class_id),
        ):
            digest.update(str(value).encode("utf-8"))
            digest.update(b"\x1f")
        digest.update(np.asarray(self.logits, dtype="<f8").tobytes(order="C"))
        digest.update(np.asarray(self.binary_labels, dtype=np.uint8).tobytes(order="C"))
        return digest.hexdigest()


def _require_torch() -> None:
    if torch is None or F is None:
        raise RuntimeError("sampled-path Torch evaluation requires PyTorch")


def _seed(*parts: Any) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _numpy_sigmoid(values: np.ndarray) -> np.ndarray:
    value = np.asarray(values, dtype=np.float64)
    result = np.empty_like(value)
    positive = value >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def fit_class_balanced_brier_affine(
    logits: Sequence[float] | np.ndarray,
    binary_labels: Sequence[int] | np.ndarray,
    *,
    query_id: str,
    role: str,
    mask_hash: str,
    calibration_context_ids: Sequence[str],
    positive_class_id: int,
    maximum_iterations: int = 256,
    gradient_tolerance: float = 1e-9,
    identity_ridge: float = 1e-8,
    warm_start: FrozenBrierCalibration | None = None,
) -> FrozenBrierCalibration:
    """Fit a positive affine by class-balanced bounded Brier risk.

    The tiny frozen identity ridge resolves otherwise flat finite samples.  It
    is part of the algorithm identity and never enters a safety bound.
    """

    x = np.asarray(logits, dtype=np.float64).reshape(-1)
    y = np.asarray(binary_labels, dtype=np.float64).reshape(-1)
    contexts = tuple(str(value) for value in calibration_context_ids)
    if (
        x.size < 4
        or y.size != x.size
        or len(contexts) != x.size
        or len(set(contexts)) != x.size
        or not np.isfinite(x).all()
        or not np.all(np.isin(y, (0.0, 1.0)))
    ):
        raise ValueError("Brier calibration requires finite unique aligned binary rows")
    positive_count = int(np.count_nonzero(y == 1.0))
    negative_count = int(y.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Brier calibration requires both binary classes")
    maximum = int(maximum_iterations)
    tolerance = float(gradient_tolerance)
    ridge = float(identity_ridge)
    if maximum <= 0 or tolerance <= 0.0 or ridge < 0.0:
        raise ValueError("Brier calibration numerical controls are invalid")
    weights = np.where(
        y == 1.0,
        0.5 / positive_count,
        0.5 / negative_count,
    )
    center = float(np.sum(weights * x))
    centered = x - center
    if warm_start is None:
        initial = np.asarray([0.0, 0.0], dtype=np.float64)
        warm_hash = None
    else:
        warm_start.validate()
        initial_scale = float(warm_start.scale)
        initial_offset = float(
            warm_start.offset
            + initial_scale * (center - float(warm_start.input_center))
        )
        initial = np.asarray([math.log(initial_scale), initial_offset])
        warm_hash = warm_start.output_hash

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        log_scale, offset = map(float, parameters)
        scale = math.exp(log_scale)
        natural = scale * centered + offset
        probabilities = _numpy_sigmoid(natural)
        residual = probabilities - y
        brier = float(np.sum(weights * residual * residual))
        regularizer = 0.5 * ridge * ((scale - 1.0) ** 2 + offset * offset)
        derivative = 2.0 * weights * residual * probabilities * (1.0 - probabilities)
        gradient = np.asarray(
            [
                float(np.sum(derivative * scale * centered))
                + ridge * (scale - 1.0) * scale,
                float(np.sum(derivative)) + ridge * offset,
            ],
            dtype=np.float64,
        )
        return brier + regularizer, gradient

    result = minimize(
        lambda value: objective(value),
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=((-12.0, 12.0), (-40.0, 40.0)),
        options={
            "maxiter": maximum,
            "ftol": min(tolerance, 1e-12),
            "gtol": tolerance,
            "maxls": 40,
        },
    )
    final_value, final_gradient = objective(np.asarray(result.x, dtype=np.float64))
    scale = math.exp(float(result.x[0]))
    offset = float(result.x[1])
    probabilities = _numpy_sigmoid(scale * centered + offset)
    probability_slope = probabilities * (1.0 - probabilities)
    design = np.column_stack((centered, np.ones_like(centered)))
    information_weights = 2.0 * weights * probability_slope * probability_slope
    information = design.T @ (information_weights[:, None] * design)
    information += max(ridge, 1e-12) * np.eye(2)
    covariance = np.linalg.pinv(information, hermitian=True)
    covariance = 0.5 * (covariance + covariance.T)
    converged = bool(
        result.success
        and np.isfinite(final_value)
        and np.isfinite(final_gradient).all()
        and float(np.max(np.abs(final_gradient))) <= max(10.0 * tolerance, 1e-7)
        and -12.0 + 1e-8 < float(result.x[0]) < 12.0 - 1e-8
        and -40.0 + 1e-8 < offset < 40.0 - 1e-8
    )
    fit = FrozenBrierCalibration(
        query_id=str(query_id),
        role=str(role),
        mask_hash=str(mask_hash),
        calibration_context_ids=contexts,
        positive_class_id=int(positive_class_id),
        scale=scale,
        offset=offset,
        input_center=center,
        covariance=tuple(tuple(float(value) for value in row) for row in covariance),
        converged=converged,
        iterations=max(1, int(getattr(result, "nit", 0))),
        objective_value=float(final_value),
        algorithm_id="class_balanced_brier_positive_affine_v1",
        algorithm_options={
            "maximum_iterations": maximum,
            "gradient_tolerance": tolerance,
            "identity_ridge": ridge,
            "scale_parameterization": "exp_log_scale",
            "scale_log_bounds": (-12.0, 12.0),
            "offset_bounds": (-40.0, 40.0),
        },
        warm_start_parent_hash=warm_hash,
        diagnostics={
            "optimizer_success": bool(result.success),
            "optimizer_status": int(result.status),
            "optimizer_message": str(result.message),
            "gradient_max_abs": float(np.max(np.abs(final_gradient))),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "covariance_role": "allocation_only_never_safety_bound",
            "covariance_method": "inverse_brier_gauss_newton_v1",
        },
    )
    fit.validate()
    return fit


def _query_class_id(query: QuerySpec) -> int:
    value = query.evaluator.get(
        "class_id",
        query.evaluator.get("unit", query.evaluator.get("coordinate")),
    )
    if value is None:
        raise ValueError("v1 sampled-path C2 calibration requires a class query")
    return int(value)


class SampledPanelSource(Protocol):
    source_id: str
    context_axis: ContextAxis
    records_by_label: Mapping[int, tuple[Any, ...]]

    def iter_panel_batches(
        self,
        context_ids: Sequence[str],
        *,
        query_id: str,
    ) -> Iterable[MeasurementInputBatch]: ...


@dataclass(frozen=True)
class ContextAxisSampledPanelSource:
    """Expose ordered panel reads over an immutable ContextAxis source."""

    context_axis: ContextAxis
    source: Any
    source_id: str = "context-axis-sampled-panel-source-v1"
    records_by_label: Mapping[int, tuple[Any, ...]] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        self.context_axis.validate()
        if self.source.axis.content_hash != self.context_axis.content_hash:
            raise ValueError("sampled panel source differs from its ContextAxis")
        if not callable(getattr(self.source, "panel_batches", None)):
            raise TypeError("sampled panel source must implement panel_batches")
        grouped: dict[int, list[Any]] = {}
        for record in self.context_axis.records:
            grouped.setdefault(int(record.label), []).append(record)
        object.__setattr__(
            self,
            "records_by_label",
            {label: tuple(records) for label, records in grouped.items()},
        )

    def iter_panel_batches(
        self,
        context_ids: Sequence[str],
        *,
        query_id: str,
    ) -> Iterable[MeasurementInputBatch]:
        del query_id
        return self.source.panel_batches(tuple(context_ids))


@dataclass(frozen=True)
class FittedSampledReferenceCalibration:
    """Full-panel affine frozen for application to untouched responses."""

    query_id: str
    role: str
    mask_hash: str
    panel: CrossfitPanel
    fit: Any
    metadata: Mapping[str, Any]

    def validate(self, *, fold_count: int) -> None:
        self.panel.validate(fold_count=fold_count)
        if not self.query_id or not self.role or len(self.mask_hash) != 64:
            raise ValueError("sampled reference calibration identity is invalid")
        if self.metadata.get("panel_hash") != self.panel.panel_hash:
            raise ValueError("sampled reference calibration panel hash differs")
        if self.metadata.get("mask_hash") != self.mask_hash:
            raise ValueError("sampled reference calibration mask hash differs")


class PermutationMemo(OrderedDict):
    """LRU memo for full-depth per-(base_seed, label) panel permutations.

    The panel permutation is stage-independent: it depends only on
    (seed, query, layer, partition, proposal, class), and a look only changes
    the prefix length taken from it. Memoizing the full permutation and
    slicing per look is therefore bit-identical to redrawing — it removes
    ~class_count numpy Generator constructions + permutations per look.
    """

    def __init__(self, limit: int = 8192) -> None:
        super().__init__()
        self.limit = int(limit)
        self.hits = 0
        self.misses = 0

    def lookup(self, key: tuple[int, int]) -> np.ndarray | None:
        value = super().get(key)
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        self.move_to_end(key)
        return value

    def store(self, key: tuple[int, int], value: np.ndarray) -> None:
        self[key] = value
        while len(self) > self.limit:
            self.popitem(last=False)


def draw_nested_class_balanced_panel(
    source: SampledPanelSource,
    request: SampledComparisonRequest,
    *,
    permutation_memo: PermutationMemo | None = None,
) -> CrossfitPanel:
    """Draw one nested, class-balanced panel without replacement.

    The permutation is proposal-specific but stage-independent.  Increasing a
    stage therefore reveals a strict extension of the same sampled path.
    Folds are stratified within class so every held-out fold retains positives
    and negatives whenever the declared stage is at least the fold count.
    """

    axis = source.context_axis
    records_by_label = getattr(source, "records_by_label", None)
    if records_by_label is None:
        grouped: dict[int, list[Any]] = {}
        for record in axis.records:
            grouped.setdefault(int(record.label), []).append(record)
        records_by_label = {label: tuple(records) for label, records in grouped.items()}
    labels_in_order = tuple(sorted(records_by_label))
    exact_total = request_cumulative_target_images(request)
    if request_per_stratum_target_images(request):
        raise ValueError("per-stratum exact targets require the Neyman panel allocator")
    if exact_total is None:
        requested_by_label = {
            label: int(request.stage_per_class) for label in labels_in_order
        }
    else:
        total = int(exact_total)
        if total <= 0:
            raise ValueError("cumulative panel target must be positive")
        quotient, remainder = divmod(total, max(1, len(labels_in_order)))
        requested_by_label = {
            label: quotient + int(index < remainder)
            for index, label in enumerate(labels_in_order)
        }
    if any(
        requested < int(request.fold_count) for requested in requested_by_label.values()
    ):
        raise ValueError("sampled panel target must cover every fold within each class")
    selected: list[tuple[Any, int, float]] = []
    base_seed = _seed(
        request.random_seed,
        request.query_id,
        request.layer_id,
        request.partition_id,
        request.proposal_index,
        "sampled-path-panel-v1",
    )
    for label in labels_in_order:
        records = records_by_label[label]
        requested = requested_by_label[label]
        if len(records) < requested:
            raise ValueError(
                f"class {label} has {len(records)} contexts, fewer than stage {requested}"
            )
        full_order: np.ndarray | None = None
        memo_key = (int(base_seed), int(label))
        if permutation_memo is not None:
            full_order = permutation_memo.lookup(memo_key)
        if full_order is None:
            rng = np.random.default_rng(_seed(base_seed, label))
            full_order = rng.permutation(len(records))
            if permutation_memo is not None:
                permutation_memo.store(memo_key, full_order)
        order = full_order[:requested]
        for rank, index in enumerate(order.tolist()):
            selected.append(
                (
                    records[int(index)],
                    int(rank % request.fold_count),
                    float(len(records)) / float(requested),
                )
            )
    selected.sort(key=lambda item: int(item[0].ordinal))
    panel = CrossfitPanel(
        context_ids=tuple(item[0].context_id for item in selected),
        fold_ids=tuple(item[1] for item in selected),
        sampling_weights=tuple(item[2] for item in selected),
        labels=tuple(int(item[0].label) for item in selected),
        draw_id=hashlib.sha256(str(base_seed).encode("ascii")).hexdigest(),
        stage_index=int(request.stage_index),
        allocation_metadata={
            "allocation_rule": "diagnostic_class_balanced_prefix",
            **(
                {
                    "requested_cumulative_target_images": int(exact_total),
                    "delivered_class_counts": {
                        str(label): int(requested_by_label[label])
                        for label in labels_in_order
                    },
                }
                if exact_total is not None
                else {}
            ),
        },
    )
    panel.validate(fold_count=request.fold_count)
    if request.prior_panel is not None:
        panel.assert_nested_extension(request.prior_panel)
    return panel


def _fold_warm_start(
    warm_start: Mapping[str, Any] | None, fold: int
) -> Mapping[str, Any]:
    if not warm_start:
        return {}
    by_fold = warm_start.get("fits_by_fold")
    if isinstance(by_fold, Mapping):
        fit = by_fold.get(str(int(fold)), by_fold.get(int(fold)))
        if isinstance(fit, Mapping):
            return fit
    if int(warm_start.get("fold", -1)) == int(fold):
        return warm_start
    return {}


def _fit_receipt(
    fit: Any,
    *,
    fold: int,
    panel: CrossfitPanel,
    calibration_indices: np.ndarray,
    extraction_indices: np.ndarray,
) -> dict[str, Any]:
    scale = getattr(fit, "readout_scale", getattr(fit, "scale", None))
    offset = getattr(fit, "readout_offset", getattr(fit, "offset", None))
    if scale is None or offset is None:
        raise TypeError("calibration fit does not expose affine parameters")
    payload = {
        "family": str(getattr(fit, "family", "c2_shared_diagonal_affine")),
        "backend": str(
            getattr(fit, "calibration_backend", getattr(fit, "backend", "unknown"))
        ),
        "fold": int(fold),
        "panel_hash": panel.panel_hash,
        "calibration_context_ids": [
            panel.context_ids[int(i)] for i in calibration_indices
        ],
        "extraction_context_ids": [
            panel.context_ids[int(i)] for i in extraction_indices
        ],
        "readout_scale": np.asarray(
            scale.detach().to(device="cpu", dtype=torch.float32).numpy()
        )
        .reshape(-1)
        .tolist(),
        "readout_offset": np.asarray(
            offset.detach().to(device="cpu", dtype=torch.float32).numpy()
        )
        .reshape(-1)
        .tolist(),
    }
    receipt = getattr(fit, "receipt", None)
    factorization = getattr(fit, "factorization", None)
    if receipt is not None:
        payload["calibration_fit_receipt"] = receipt.to_dict()
    if factorization is not None:
        payload["calibration_dependency_hash"] = factorization.dependency_hash
        payload["calibration_factorization_hash"] = factorization.factorization_hash
    return payload


def _resolved_proper_score_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve one numerically complete v2 policy for contracts and execution."""

    return {
        "identity_anchor_strengths": tuple(
            float(value)
            for value in options.get(
                "identity_anchor_strengths",
                CANONICAL_CALIBRATION_IDENTITY_ANCHOR_STRENGTHS,
            )
        ),
        "calibration_selection_blocks": int(
            options.get(
                "calibration_selection_blocks",
                CANONICAL_CALIBRATION_SELECTION_BLOCKS,
            )
        ),
        "anchor_probability_epsilon": float(
            options.get(
                "anchor_probability_epsilon",
                CANONICAL_CALIBRATION_ANCHOR_PROBABILITY_EPSILON,
            )
        ),
        "max_iterations": int(
            options.get("iters", CANONICAL_CALIBRATION_MAX_ITERATIONS)
        ),
        "kkt_tolerance": float(
            options.get("kkt_tolerance", CANONICAL_CALIBRATION_KKT_TOLERANCE)
        ),
        "hessian_ridge": float(
            options.get("hessian_ridge", CANONICAL_CALIBRATION_HESSIAN_RIDGE)
        ),
        "objective_tolerance": float(
            options.get(
                "objective_tolerance",
                CANONICAL_CALIBRATION_OBJECTIVE_TOLERANCE,
            )
        ),
        "reference_tolerance": float(
            options.get(
                "reference_tolerance",
                CANONICAL_CALIBRATION_REFERENCE_TOLERANCE,
            )
        ),
    }


def _query_readout_center(
    readout_bias: Any,
    *,
    class_id: int,
    response_width: int,
    device: Any,
) -> Any:
    center = torch.as_tensor(readout_bias, device=device, dtype=torch.float32).reshape(
        -1
    )
    if int(center.numel()) == 1:
        return center
    if int(center.numel()) != int(response_width):
        raise ValueError("fixed-readout bias differs from the response axis")
    return center[class_id : class_id + 1]


class TorchSampledPathEvaluationBackend:
    """Exact paired masked execution with a versioned calibration policy."""

    backend_id = "mechanistic-sampled-path-crossfit-calibration"

    def __init__(
        self,
        *,
        evaluator: TorchCircuitEvaluator,
        panel_source: SampledPanelSource,
        queries: Sequence[QuerySpec],
        calibration_backend: CanonicalCalibrationBackend | None = None,
        calibration_options: Mapping[str, Any] | None = None,
        panel_policy: CostOptimalNeymanPolicy | None = None,
        response_cache_max_bytes: int | None = None,
        runtime_execution_profile: RuntimeExecutionProfile | None = None,
    ) -> None:
        _require_torch()
        self.evaluator = evaluator
        self.panel_source = panel_source
        self._query_by_id = {query.query_id: query for query in queries}
        if not self._query_by_id or len(self._query_by_id) != len(tuple(queries)):
            raise ValueError("sampled-path backend requires unique queries")
        self.runtime_execution_profile = RuntimeExecutionProfile.resolve(
            runtime_execution_profile
        )
        self.calibration_backend = calibration_backend or CanonicalCalibrationBackend(
            runtime_profile=self.runtime_execution_profile
        )
        self.runtime_execution_session = RuntimeExecutionSession(
            self.runtime_execution_profile,
            candidate_backend=getattr(evaluator, "candidate_execution_backend", None),
        )
        self.panel_policy = panel_policy
        supplied_options = dict(calibration_options or {})
        if "scalar_target_semantics" in supplied_options:
            raise ValueError(
                "scalar_target_semantics was removed; calibration topology is "
                "compiled from readout, query, parameter, and loss dependencies"
            )
        requested_contract = str(
            supplied_options.get(
                "optimization_contract",
                CANONICAL_PRUNING_CALIBRATION_ALGORITHM,
            )
        )
        if requested_contract not in {
            SAMPLED_PATH_V1_CALIBRATION_ALGORITHM,
            PROPER_SCORE_SHRINKAGE_AFFINE_V2,
            "finite_mle",
        }:
            raise ValueError(
                f"unsupported sampled-path calibration contract {requested_contract!r}"
            )
        self.backend_id = f"mechanistic-sampled-path-crossfit:{requested_contract}"
        v1_defaults = {
            "binary_loss_weighting": "pos_weighted",
            "enforce_positive_scale": True,
            "iters": 64,
            "lr": 0.25,
            "ws_min": 1e-8,
            "optimization_contract": requested_contract,
        }
        if requested_contract == PROPER_SCORE_SHRINKAGE_AFFINE_V2:
            v1_defaults.update(
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
        if (
            panel_policy is not None
            and requested_contract == SAMPLED_PATH_V1_CALIBRATION_ALGORITHM
        ):
            conflicts = [
                key
                for key, expected in v1_defaults.items()
                if key in supplied_options and supplied_options[key] != expected
            ]
            if conflicts:
                raise ValueError(
                    "sampled_path_frontier_v1 calibration semantics differ: "
                    + ",".join(sorted(conflicts))
                )
        self.calibration_options = {**v1_defaults, **supplied_options}
        # Readout-column policy: the sealed v1 fitter is 2-D-only by design,
        # so its runs pin to the decision column; explicit multi-column intent
        # under the sealed contract fails closed. Every other contract
        # defaults to "all" (the estimability-resolved cross-category set).
        requested_columns = self.calibration_options.get("readout_columns", "all")
        if requested_contract == SAMPLED_PATH_V1_CALIBRATION_ALGORITHM:
            if isinstance(requested_columns, (tuple, list)):
                raise ValueError(
                    "the sealed v1 calibration contract is single-column; "
                    "explicit readout_columns are not supported"
                )
            self.calibration_options["readout_columns"] = "target_only"
        else:
            self.calibration_options.setdefault("readout_columns", "all")
        if "readout_bias" not in self.calibration_options:
            readout_bias = getattr(evaluator, "readout_bias", None)
            if readout_bias is not None:
                self.calibration_options["readout_bias"] = readout_bias
        if panel_policy is not None and "readout_bias" not in self.calibration_options:
            raise ValueError(
                "sampled_path_frontier_v1 requires the declared fixed-readout bias"
            )
        self._panel_planners: dict[str, BinaryCostOptimalNeymanPanelPlanner] = {}
        if response_cache_max_bytes is None:
            # Runtime-only budget resolved from the execution profile
            # (CIRCUIT_SAMPLED_RESPONSE_CACHE_BYTES); an explicit argument
            # keeps precedence. Cache hits return stored exact responses, so
            # the size can never change a decision.
            response_cache_max_bytes = (
                self.runtime_execution_profile.sampled_response_cache_bytes
            )
        if int(response_cache_max_bytes) < 0:
            raise ValueError("sampled-path response-cache budget cannot be negative")
        self.response_cache_max_bytes = int(response_cache_max_bytes)
        self._response_cache: OrderedDict[
            tuple[str, str], tuple[ReadoutResponse, int]
        ] = OrderedDict()
        self._response_cache_bytes = 0
        self._response_cache_peak_bytes = 0
        self._response_cache_hits = 0
        self._response_cache_misses = 0
        self._response_cache_evictions = 0
        self._panel_draws = 0
        self._panel_draw_repeats = 0
        self._panel_draw_counts: dict[str, int] = {}
        # Identity-keyed panel memo (frozen panels; draws are
        # deterministic in the full identity, so hits are bit-identical).
        self._panel_draw_memo: "OrderedDict[str, Any]" = OrderedDict()
        self._panel_draw_memo_hits = 0
        # Device-resident panel batches, LRU-bounded BY BYTES — a working
        # set of active panels, never the bank. Reuse is structural: the CRN
        # pair consumes every panel at least twice, and sealed-prefix panels
        # are content-identical across proposals, so the cache is keyed by
        # the context-id sequence (content), not the panel hash (which
        # carries draw metadata and would defeat cross-proposal reuse).
        # First touches stream through a background pipeline (load hides
        # under the panel's own forwards) and the prefetch worker fills
        # future panels during the current decision's compute.
        self._panel_batch_cache: OrderedDict[str, tuple[list, int]] = OrderedDict()
        self._panel_batch_cache_byte_limit = 8 << 30
        self._panel_batch_cache_bytes = 0
        self._panel_batch_cache_hits = 0
        self._panel_batch_cache_misses = 0
        self._panel_cache_lock = threading.Lock()
        self._prefetch_inflight: set[str] = set()
        self._prefetch_queue: "queue.Queue | None" = None
        self._prefetch_thread: "threading.Thread | None" = None
        self._producer_pool_fallback_reason = ""
        self.profiler = PhaseProfiler(
            cuda_events_enabled=self.runtime_execution_profile.cuda_events
        )
        # Opt-in evaluation reuse (adopt behind a parity receipt).
        # Incremental responses: nested-look panels re-forward only contexts
        # not already evaluated under the same mask (eval-mode readouts are
        # per-context deterministic). Crossfit cache: identical (mask, panel,
        # fold-structure) fits reuse their losses — the committed side of
        # every CRN pair re-fits byte-identical inputs across proposals.
        self.incremental_responses = (
            self.runtime_execution_profile.incremental_responses
        )
        self.crossfit_cache_enabled = self.runtime_execution_profile.sampled_fit_cache
        self._crossfit_cache: OrderedDict[tuple, tuple] = OrderedDict()
        self._crossfit_cache_limit = 512
        self._crossfit_cache_hits = 0
        self._crossfit_cache_misses = 0
        # Panel-derived keys (content sha256, fold-structure sha256) are pure
        # functions of the panel; both lanes of every CRN pair and the panel
        # batch path need them, so memoize per panel identity.
        self._panel_key_memo: OrderedDict[str, tuple[str, str]] = OrderedDict()
        self._panel_key_memo_limit = 64
        # Panel-derived per-fit assemblies. Readout-column resolution,
        # the [fold, sample] validity stack, and the per-fold context-id
        # lists are pure functions of (panel content, fold structure,
        # fold count, class, policy); every lane of every look re-derived
        # them from the same frozen panel. Values are immutable by
        # construction (tuples; the cached validity ndarray is marked
        # read-only and its torch view is what the fit consumes).
        self._readout_columns_memo: OrderedDict[tuple, tuple[int, ...]] = OrderedDict()
        self._fold_validity_memo: OrderedDict[tuple, tuple[Any, Any]] = OrderedDict()
        self._fold_context_ids_memo: OrderedDict[tuple, tuple] = OrderedDict()
        self._panel_assembly_memo_limit = 64
        # Full-depth panel permutations are stage-independent per
        # (proposal, class); each look re-slices the same permutation, so the
        # ~class_count numpy permutations per look collapse to memo lookups.
        self._permutation_memo = PermutationMemo()
        self._incremental_rows_reused = 0
        self._incremental_rows_forwarded = 0
        self._contract_memo: dict[tuple, tuple] = {}
        # Per-mask incremental row store: each (mask, context) readout row is
        # ingested exactly once; assemblies are vectorized takes on integer
        # indices (the first implementation rebuilt row maps by scanning the
        # panel-keyed response cache per assembly — receipt-proven Python
        # wall). float32: every consumer casts readouts to float32 anyway.
        self._mask_row_store: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._mask_row_store_bytes = 0
        self._mask_row_store_byte_limit = int(
            self.runtime_execution_profile.sampled_row_store_bytes
        )
        # Coalesce ragged run-split batches to this many rows before the
        # forward (0 = off). Receipt-proven: increment forwards averaged 69
        # rows per call, paying full per-call overhead each.
        self._coalesce_rows = int(self.runtime_execution_profile.sampled_coalesce_rows)
        # The execution batch is the row count each masked forward actually
        # processes. Historically it silently equaled the loader chunk (8-32
        # rows) even though the throughput-saturating batch was measured and
        # reported — the single largest per-forward overhead multiplier.
        # Runtime never touches identity: batch size is recorded in telemetry
        # and cannot change a scientific plan hash.
        self._execution_batch_rows: int | None = None
        self._execution_batch_provenance = "loader_default"
        if self.runtime_execution_profile.execution_batch_rows is not None:
            self.adopt_execution_batch(
                self.runtime_execution_profile.execution_batch_rows,
                provenance=(
                    "env" if runtime_execution_profile is None else "runtime_profile"
                ),
            )
        self._compiled_mask_cache = _BoundedCompiledMaskCache()
        self._resource_lock = threading.Lock()
        # One campaign owns one serial claim stream.  Refusing overlap keeps
        # each runtime receipt's global before/after delta attributable to
        # exactly one public glocal operation.
        self._glocal_call_lock = threading.Lock()
        self._resource_unique_context_ids: set[str] = set()
        self._resource_unique_decision_context_ids: set[str] = set()
        self._resource_context_state_requests: set[tuple[str, str, str]] = set()
        self._resource_context_state_evaluations: set[tuple[str, str, str]] = set()
        self._resource_panel_deliveries = 0
        self._resource_panel_context_rows = 0
        # ``forward_calls`` is the historical high-level evaluator invocation
        # count.  Physical candidate-backend calls are tracked independently:
        # one fused evaluator invocation can execute many physical batches.
        self._resource_forward_calls = 0
        self._resource_forward_rows = 0
        self._resource_failed_high_level_forward_calls = 0
        self._resource_physical_forward_calls = 0
        self._resource_physical_forward_calls_observed = False
        self._resource_backward_calls = 0
        self._resource_backward_calls_observed = False
        self._resource_host_to_device_bytes = 0
        self._resource_device_to_host_bytes = 0
        self._resource_glocal_calls = 0
        self._resource_concurrent_glocal_rejections = 0
        self._resource_glocal_wall_seconds = 0.0
        self._resource_glocal_cuda_device_seconds = 0.0
        self._resource_absolute_peak_device_memory_bytes = 0
        self._resource_calibration_fit_calls = 0
        self._resource_prefetch_requests = 0
        self._resource_prefetch_scheduled = 0
        self._resource_prefetch_cache_skips = 0
        self._resource_prefetch_inflight_skips = 0
        self._resource_prefetch_completed = 0
        self._resource_prefetch_failed = 0
        self._resource_glocal_prefetch_unreserved_rejections = 0
        self._resource_glocal_prefetch_reservations_consumed: set[str] = set()
        self._resource_prefetched_context_ids: set[str] = set()
        self._resource_materialized_response_ids: set[int] = set()
        self._resource_panel_prefixes: list[dict[str, Any]] = []
        self._resource_glocal_call_receipts: list[GlocalRuntimeCallReceipt] = []
        self._compiled_mask_cache_hits = 0
        self._compiled_mask_cache_misses = 0

    def _ensure_resource_accounting(self) -> None:
        """Initialize counters on lightweight test shells made via __new__."""

        if hasattr(self, "_resource_lock"):
            return
        self._resource_lock = threading.Lock()
        self._glocal_call_lock = threading.Lock()
        self._resource_unique_context_ids = set()
        self._resource_unique_decision_context_ids = set()
        self._resource_context_state_requests = set()
        self._resource_context_state_evaluations = set()
        self._resource_panel_deliveries = 0
        self._resource_panel_context_rows = 0
        self._resource_forward_calls = 0
        self._resource_forward_rows = 0
        self._resource_failed_high_level_forward_calls = 0
        self._resource_physical_forward_calls = 0
        self._resource_physical_forward_calls_observed = False
        self._resource_backward_calls = 0
        self._resource_backward_calls_observed = False
        self._resource_host_to_device_bytes = 0
        self._resource_device_to_host_bytes = 0
        self._resource_glocal_calls = 0
        self._resource_concurrent_glocal_rejections = 0
        self._resource_glocal_wall_seconds = 0.0
        self._resource_glocal_cuda_device_seconds = 0.0
        self._resource_absolute_peak_device_memory_bytes = 0
        self._resource_calibration_fit_calls = 0
        self._resource_prefetch_requests = 0
        self._resource_prefetch_scheduled = 0
        self._resource_prefetch_cache_skips = 0
        self._resource_prefetch_inflight_skips = 0
        self._resource_prefetch_completed = 0
        self._resource_prefetch_failed = 0
        self._resource_glocal_prefetch_unreserved_rejections = 0
        self._resource_glocal_prefetch_reservations_consumed = set()
        self._resource_prefetched_context_ids = set()
        self._resource_materialized_response_ids = set()
        self._resource_panel_prefixes = []
        self._resource_glocal_call_receipts = []
        self._compiled_mask_cache_hits = 0
        self._compiled_mask_cache_misses = 0

    def _record_panel_delivery(
        self,
        panel: CrossfitPanel,
        *,
        request: SampledComparisonRequest,
        role: str,
        mask_hashes: Sequence[str],
    ) -> None:
        self._ensure_resource_accounting()
        contexts = tuple(str(value) for value in panel.context_ids)
        with self._resource_lock:
            new_contexts = set(contexts) - self._resource_unique_context_ids
            if role == "terminal":
                evaluated_contexts = {
                    context_id
                    for _query_id, context_id, _mask_hash in (
                        self._resource_context_state_evaluations
                    )
                }
                prior_contexts = self._resource_unique_context_ids | evaluated_contexts
                if set(contexts).intersection(prior_contexts):
                    raise ValueError(
                        "terminal audit contexts must be untouched by every earlier role"
                    )
            new_decision = (
                set(contexts) - self._resource_unique_decision_context_ids
                if role == "decision"
                else set()
            )
            cap = request_maximum_new_contexts(request)
            if cap is not None and int(cap) < 0:
                raise ValueError("new-context budget must be nonnegative")
            if cap is not None and len(new_contexts) > int(cap):
                raise ValueError(
                    "sampled panel exceeds the declared new-context budget"
                )
            self._resource_unique_context_ids.update(contexts)
            if role == "decision":
                self._resource_unique_decision_context_ids.update(contexts)
            self._resource_context_state_requests.update(
                (str(request.query_id), context_id, str(mask_hash))
                for mask_hash in mask_hashes
                for context_id in contexts
            )
            self._resource_panel_deliveries += 1
            self._resource_panel_context_rows += len(contexts)
            self._resource_panel_prefixes.append(
                {
                    "role": str(role),
                    "query_id": str(request.query_id),
                    "proposal_index": int(request.proposal_index),
                    "stage_index": int(request.stage_index),
                    "requested_cumulative_target_images": (
                        int(request_cumulative_target_images(request))
                        if request_cumulative_target_images(request) is not None
                        else None
                    ),
                    "requested_per_stratum_target_images": dict(
                        request_per_stratum_target_images(request)
                    ),
                    "delivered_images": len(contexts),
                    "new_contexts": len(new_contexts),
                    "new_decision_images": len(new_decision),
                    "panel_hash": panel.panel_hash,
                }
            )

    @staticmethod
    def _validate_explicit_glocal_panel_request(
        panel: CrossfitPanel,
        request: SampledComparisonRequest,
        *,
        target_class_id: int,
    ) -> None:
        """Bind an explicit claim-mode panel to its predictable request.

        The generic sampled-path methods may still draw a panel internally.
        Glocal-v3 instead supplies the exact cumulative sealed prefix, so a
        backend must refuse any adapter that silently delivers a different
        total, stratum allocation, or nested history.
        """

        delivered = len(panel.context_ids)
        cumulative_target = request_cumulative_target_images(request)
        requested_targets = request_per_stratum_target_images(request)
        if cumulative_target is not None and delivered != int(cumulative_target):
            raise ValueError(
                "explicit glocal panel size differs from its cumulative target"
            )
        if request.prior_panel is not None:
            panel.assert_nested_extension(request.prior_panel)
        if not requested_targets:
            return
        if not panel.labels:
            raise ValueError("explicit glocal stratum targets require panel labels")
        declared = {str(key): int(value) for key, value in requested_targets.items()}
        if any(value < 0 for value in declared.values()):
            raise ValueError("explicit glocal stratum targets must be nonnegative")
        labels = np.asarray(panel.labels, dtype=np.int64)
        observed: dict[str, int] = {
            "positive": int(np.count_nonzero(labels == int(target_class_id)))
        }
        for class_id in np.unique(labels):
            if int(class_id) == int(target_class_id):
                continue
            observed[f"neg:{int(class_id)}"] = int(
                np.count_nonzero(labels == int(class_id))
            )
        observed = {key: value for key, value in observed.items() if value > 0}
        declared = {key: value for key, value in declared.items() if value > 0}
        if observed != declared:
            raise ValueError(
                "explicit glocal panel strata differ from their cumulative targets"
            )
        if sum(declared.values()) != delivered:
            raise ValueError("explicit glocal stratum targets do not exhaust the panel")

    def _record_forward(
        self,
        *,
        mask_hashes: Sequence[str],
        context_ids: Sequence[str],
        calls: int = 1,
        physical_calls: int | None = None,
        backward_calls: int | None = None,
        query_id: str | None = None,
    ) -> None:
        self._ensure_resource_accounting()
        contexts = tuple(str(value) for value in context_ids)
        hashes = tuple(str(value) for value in mask_hashes)
        if query_id is None:
            available = tuple(getattr(self, "_query_by_id", {}))
            query_id = (
                str(available[0])
                if len(available) == 1
                else "__legacy_unscoped_query__"
            )
        high_level_calls = int(calls)
        if high_level_calls < 0:
            raise ValueError("high-level forward-call delta cannot be negative")
        if physical_calls is not None and int(physical_calls) < 0:
            raise ValueError("physical forward-call delta cannot be negative")
        if backward_calls is not None and int(backward_calls) < 0:
            raise ValueError("backward-call delta cannot be negative")
        with self._resource_lock:
            self._resource_forward_calls += high_level_calls
            self._resource_forward_rows += len(contexts) * len(hashes)
            if physical_calls is not None:
                self._resource_physical_forward_calls_observed = True
                self._resource_physical_forward_calls += int(physical_calls)
            if backward_calls is not None:
                self._resource_backward_calls_observed = True
                self._resource_backward_calls += int(backward_calls)
            self._resource_context_state_evaluations.update(
                (str(query_id), context_id, mask_hash)
                for mask_hash in hashes
                for context_id in contexts
            )

    def _candidate_backend_counters(self) -> tuple[int | None, int | None]:
        """Read exact physical counters exposed by the candidate backend."""

        evaluator = getattr(self, "evaluator", None)
        backend = getattr(evaluator, "candidate_execution_backend", None)

        def value(name: str) -> int | None:
            raw = getattr(backend, name, None)
            if isinstance(raw, (int, np.integer)) and int(raw) >= 0:
                return int(raw)
            return None

        return value("calls"), value("backward_calls")

    def _record_failed_forward_attempt(
        self,
        before: tuple[int | None, int | None],
    ) -> None:
        """Account backend work performed before a high-level forward failed."""

        after = self._candidate_backend_counters()
        physical_calls = self._counter_delta(before[0], after[0])
        backward_calls = self._counter_delta(before[1], after[1])
        with self._resource_lock:
            self._resource_forward_calls += 1
            self._resource_failed_high_level_forward_calls += 1
            if physical_calls is not None:
                self._resource_physical_forward_calls_observed = True
                self._resource_physical_forward_calls += int(physical_calls)
            if backward_calls is not None:
                self._resource_backward_calls_observed = True
                self._resource_backward_calls += int(backward_calls)

    @staticmethod
    def _counter_delta(
        before: int | None,
        after: int | None,
    ) -> int | None:
        if before is None or after is None:
            return None
        if after < before:
            # A counter reset makes the interval unobservable; never invent a
            # negative or replacement count.
            return None
        return int(after - before)

    def _record_transfer(
        self,
        *,
        host_to_device_bytes: int = 0,
        device_to_host_bytes: int = 0,
    ) -> None:
        self._ensure_resource_accounting()
        h2d = int(host_to_device_bytes)
        d2h = int(device_to_host_bytes)
        if h2d < 0 or d2h < 0:
            raise ValueError("runtime transfer byte counts cannot be negative")
        with self._resource_lock:
            self._resource_host_to_device_bytes += h2d
            self._resource_device_to_host_bytes += d2h

    @staticmethod
    def _tensor_nbytes(value: Any) -> int:
        if torch is not None and torch.is_tensor(value):
            return int(value.numel()) * int(value.element_size())
        return int(np.asarray(value).nbytes)

    @staticmethod
    def _is_host_to_device_move(value: Any, destination: Any) -> bool:
        if torch is None or not torch.is_tensor(value):
            source_type = "cpu"
        else:
            source_type = str(value.device.type)
        try:
            destination_type = str(torch.device(destination).type)
        except Exception:
            return False
        return source_type == "cpu" and destination_type != "cpu"

    def _record_response_materialization(
        self,
        response: ReadoutResponse,
        *,
        include_labels: bool,
    ) -> None:
        """Account one actual device-to-host response materialization.

        Native evaluator responses retain ``device_values`` even after eager
        materialization.  That lets the outer backend account both eager and
        lazy paths exactly once without changing the response's scientific
        representation.  CPU-only stubs correctly contribute zero bytes.
        """

        device_values = getattr(response, "device_values", None)
        if (
            torch is None
            or not torch.is_tensor(device_values)
            or str(device_values.device.type) == "cpu"
        ):
            return
        identity = id(response)
        self._ensure_resource_accounting()
        with self._resource_lock:
            if identity in self._resource_materialized_response_ids:
                return
            self._resource_materialized_response_ids.add(identity)
        try:
            # Access performs the one lazy D2H transfer when required.
            values = np.asarray(response.values)
            transferred = int(values.nbytes)
            if include_labels:
                transferred += int(np.asarray(response.labels).nbytes)
        except Exception:
            with self._resource_lock:
                self._resource_materialized_response_ids.discard(identity)
            raise
        self._record_transfer(device_to_host_bytes=transferred)

    def _probe_absolute_peak_device_memory(self) -> int:
        if torch is None or not torch.cuda.is_available():
            return 0
        device = getattr(getattr(self, "evaluator", None), "device", None)
        try:
            resolved = torch.device(device)
        except Exception:
            return 0
        if resolved.type != "cuda":
            return 0
        try:
            return int(torch.cuda.max_memory_allocated(resolved))
        except Exception:
            return 0

    def _resource_snapshot(self) -> GlocalRuntimeSnapshot:
        self._ensure_resource_accounting()
        observed_peak = self._probe_absolute_peak_device_memory()
        with self._resource_lock:
            self._resource_absolute_peak_device_memory_bytes = max(
                int(self._resource_absolute_peak_device_memory_bytes),
                observed_peak,
            )
            return GlocalRuntimeSnapshot(
                unique_contexts=len(self._resource_unique_context_ids),
                unique_decision_contexts=len(
                    self._resource_unique_decision_context_ids
                ),
                unique_context_state_requests=len(
                    self._resource_context_state_requests
                ),
                unique_context_state_evaluations=len(
                    self._resource_context_state_evaluations
                ),
                high_level_forward_calls=int(self._resource_forward_calls),
                failed_high_level_forward_calls=int(
                    self._resource_failed_high_level_forward_calls
                ),
                physical_forward_calls=int(self._resource_physical_forward_calls),
                physical_forward_calls_observed=bool(
                    self._resource_physical_forward_calls_observed
                ),
                backward_calls=int(self._resource_backward_calls),
                backward_calls_observed=bool(self._resource_backward_calls_observed),
                host_to_device_bytes=int(self._resource_host_to_device_bytes),
                device_to_host_bytes=int(self._resource_device_to_host_bytes),
                calibration_fit_calls=int(self._resource_calibration_fit_calls),
                glocal_calls=int(self._resource_glocal_calls),
                glocal_wall_seconds=float(self._resource_glocal_wall_seconds),
                glocal_cuda_device_seconds=float(
                    self._resource_glocal_cuda_device_seconds
                ),
                absolute_peak_device_memory_bytes=int(
                    self._resource_absolute_peak_device_memory_bytes
                ),
            )

    @contextmanager
    def _glocal_runtime_call(
        self,
        *,
        operation: str,
        request: SampledComparisonRequest,
        role: str,
        panel: CrossfitPanel,
    ) -> Iterator[None]:
        """Receipt one public glocal call without entering scientific hashes."""

        self._ensure_resource_accounting()
        if not self._glocal_call_lock.acquire(blocking=False):
            with self._resource_lock:
                self._resource_concurrent_glocal_rejections += 1
            raise RuntimeError(
                "concurrent claim-mode calls on one backend are unsupported; "
                "a campaign must serialize its predictable evidence stream"
            )
        try:
            before = self._resource_snapshot()
            started = time.perf_counter()
            cuda_start = cuda_end = None
            if torch is not None and torch.cuda.is_available():
                device = getattr(getattr(self, "evaluator", None), "device", None)
                try:
                    if torch.device(device).type == "cuda":
                        cuda_start = torch.cuda.Event(enable_timing=True)
                        cuda_end = torch.cuda.Event(enable_timing=True)
                        cuda_start.record()
                except Exception:
                    cuda_start = cuda_end = None
            succeeded = False
            error_type: str | None = None
            try:
                yield
                succeeded = True
            except BaseException as error:
                error_type = type(error).__name__
                raise
            finally:
                cuda_seconds = 0.0
                if cuda_start is not None and cuda_end is not None:
                    try:
                        cuda_end.record()
                        cuda_end.synchronize()
                        cuda_seconds = float(cuda_start.elapsed_time(cuda_end)) / 1000.0
                    except Exception:
                        cuda_seconds = 0.0
                elapsed = float(time.perf_counter() - started)
                peak = self._probe_absolute_peak_device_memory()
                with self._resource_lock:
                    self._resource_glocal_calls += 1
                    self._resource_glocal_wall_seconds += elapsed
                    self._resource_glocal_cuda_device_seconds += cuda_seconds
                    self._resource_absolute_peak_device_memory_bytes = max(
                        int(self._resource_absolute_peak_device_memory_bytes),
                        peak,
                    )
                after = self._resource_snapshot()
                receipt = GlocalRuntimeCallReceipt(
                    operation=str(operation),
                    query_id=str(request.query_id),
                    role=str(role),
                    proposal_index=int(request.proposal_index),
                    stage_index=int(request.stage_index),
                    panel_hash=panel.panel_hash,
                    succeeded=succeeded,
                    error_type=error_type,
                    before=before,
                    after=after,
                )
                with self._resource_lock:
                    self._resource_glocal_call_receipts.append(receipt)
        finally:
            self._glocal_call_lock.release()

    @property
    def glocal_runtime_receipts(self) -> tuple[GlocalRuntimeCallReceipt, ...]:
        self._ensure_resource_accounting()
        with self._resource_lock:
            return tuple(self._resource_glocal_call_receipts)

    @property
    def last_glocal_runtime_receipt(
        self,
    ) -> GlocalRuntimeCallReceipt | None:
        receipts = self.glocal_runtime_receipts
        return receipts[-1] if receipts else None

    @property
    def decision_images_consumed(self) -> int:
        """Unique decision contexts introduced, excluding hints and replays."""

        self._ensure_resource_accounting()
        with self._resource_lock:
            return len(self._resource_unique_decision_context_ids)

    @property
    def resource_usage(self) -> Mapping[str, Any]:
        """Backend-owned, runtime-only acquisition and execution ledger."""

        self._ensure_resource_accounting()
        snapshot = self._resource_snapshot()
        with self._resource_lock:
            query_ids = sorted(
                {
                    query_id
                    for query_id, _context_id, _mask_hash in (
                        self._resource_context_state_requests
                        | self._resource_context_state_evaluations
                    )
                }
            )
            by_query = {
                query_id: {
                    "unique_context_state_requests": sum(
                        1
                        for key in self._resource_context_state_requests
                        if key[0] == query_id
                    ),
                    "unique_context_state_evaluations": sum(
                        1
                        for key in self._resource_context_state_evaluations
                        if key[0] == query_id
                    ),
                }
                for query_id in query_ids
            }
            return {
                "schema_name": "sampled_path_runtime_resource_ledger_v1",
                "runtime_extension_schema": GLOCAL_RUNTIME_RESOURCE_SCHEMA,
                "unique_contexts": len(self._resource_unique_context_ids),
                "unique_decision_contexts": len(
                    self._resource_unique_decision_context_ids
                ),
                "unique_context_state_requests": len(
                    self._resource_context_state_requests
                ),
                "unique_context_state_evaluations": len(
                    self._resource_context_state_evaluations
                ),
                "panel_deliveries": self._resource_panel_deliveries,
                "panel_context_rows": self._resource_panel_context_rows,
                # Compatibility name: one high-level evaluator invocation.
                "forward_calls": self._resource_forward_calls,
                "high_level_forward_calls": self._resource_forward_calls,
                "failed_high_level_forward_calls": (
                    self._resource_failed_high_level_forward_calls
                ),
                "forward_rows": self._resource_forward_rows,
                "physical_forward_calls": (self._resource_physical_forward_calls),
                "physical_forward_calls_observed": (
                    self._resource_physical_forward_calls_observed
                ),
                "backward_calls": self._resource_backward_calls,
                "backward_calls_observed": (self._resource_backward_calls_observed),
                "host_to_device_bytes": (self._resource_host_to_device_bytes),
                "device_to_host_bytes": (self._resource_device_to_host_bytes),
                "absolute_peak_device_memory_bytes": (
                    snapshot.absolute_peak_device_memory_bytes
                ),
                "calibration_fit_calls": self._resource_calibration_fit_calls,
                "glocal": {
                    "calls": self._resource_glocal_calls,
                    "concurrent_call_rejections": (
                        self._resource_concurrent_glocal_rejections
                    ),
                    "wall_seconds": self._resource_glocal_wall_seconds,
                    "cuda_device_seconds": (self._resource_glocal_cuda_device_seconds),
                    "call_receipts": tuple(
                        receipt.to_dict()
                        for receipt in self._resource_glocal_call_receipts
                    ),
                },
                "by_query": by_query,
                "prefetch": {
                    "requests": self._resource_prefetch_requests,
                    "scheduled": self._resource_prefetch_scheduled,
                    "cache_skips": self._resource_prefetch_cache_skips,
                    "inflight_skips": self._resource_prefetch_inflight_skips,
                    "completed": self._resource_prefetch_completed,
                    "failed": self._resource_prefetch_failed,
                    "unique_contexts_loaded": len(
                        self._resource_prefetched_context_ids
                    ),
                    "glocal_unreserved_rejections": (
                        self._resource_glocal_prefetch_unreserved_rejections
                    ),
                    "glocal_reservations_consumed": len(
                        self._resource_glocal_prefetch_reservations_consumed
                    ),
                },
                "response_cache": {
                    "hits": int(getattr(self, "_response_cache_hits", 0)),
                    "misses": int(getattr(self, "_response_cache_misses", 0)),
                    "evictions": int(getattr(self, "_response_cache_evictions", 0)),
                    "resident_bytes": int(getattr(self, "_response_cache_bytes", 0)),
                    "peak_bytes": int(getattr(self, "_response_cache_peak_bytes", 0)),
                },
                "crossfit_cache": {
                    "hits": int(getattr(self, "_crossfit_cache_hits", 0)),
                    "misses": int(getattr(self, "_crossfit_cache_misses", 0)),
                },
                "panel_batch_cache": {
                    "hits": int(getattr(self, "_panel_batch_cache_hits", 0)),
                    "misses": int(getattr(self, "_panel_batch_cache_misses", 0)),
                    "resident_bytes": int(getattr(self, "_panel_batch_cache_bytes", 0)),
                },
                "compiled_mask_cache": {
                    "hits": int(self._compiled_mask_cache_hits),
                    "misses": int(self._compiled_mask_cache_misses),
                    "entries": len(getattr(self, "_compiled_mask_cache", {})),
                },
                "panel_prefixes": tuple(
                    dict(value) for value in self._resource_panel_prefixes
                ),
            }

    def adopt_execution_batch(self, rows: int, *, provenance: str) -> None:
        """Adopt ``rows`` as the execution chunk for masked forwards.

        Sets the stream coalesce target (covering ragged loader chunks and
        small increment forwards alike) and clears the device panel-batch
        cache, whose entries are shaped by the previous chunking. Recorded in
        ``runtime_profile()`` with its provenance (measured | env | pinned).
        """

        rows = int(rows)
        if rows < 1:
            raise ValueError("execution batch must be a positive row count")
        self._execution_batch_rows = rows
        self._execution_batch_provenance = str(provenance)
        self._coalesce_rows = rows
        with self._panel_cache_lock:
            self._panel_batch_cache.clear()
            self._panel_batch_cache_bytes = 0

    @staticmethod
    def _response_payload_bytes(response: ReadoutResponse) -> int:
        return int(
            np.asarray(response.values).nbytes + np.asarray(response.labels).nbytes
        )

    def _cache_response(
        self,
        key: tuple[str, str],
        response: ReadoutResponse,
    ) -> None:
        # The cache budget is host memory; retaining the transient device
        # tensor here would silently pin HBM for every cached panel.
        # ReadoutResponse is not a dataclass (lazy host values), so
        # strip by explicit construction; touching .values here forces the
        # one-time host materialization a durable cache entry must carry.
        if getattr(response, "device_values", None) is not None:
            response = ReadoutResponse(
                context_ids=response.context_ids,
                values=response.values,
                labels=response.labels,
                device_values=None,
            )
        payload_bytes = self._response_payload_bytes(response)
        if (
            self.response_cache_max_bytes == 0
            or payload_bytes > self.response_cache_max_bytes
        ):
            return
        prior = self._response_cache.pop(key, None)
        if prior is not None:
            self._response_cache_bytes -= int(prior[1])
        while (
            self._response_cache
            and self._response_cache_bytes + payload_bytes
            > self.response_cache_max_bytes
        ):
            _, (_, evicted_bytes) = self._response_cache.popitem(last=False)
            self._response_cache_bytes -= int(evicted_bytes)
            self._response_cache_evictions += 1
        self._response_cache[key] = (response, payload_bytes)
        self._response_cache_bytes += payload_bytes
        self._response_cache_peak_bytes = max(
            self._response_cache_peak_bytes,
            self._response_cache_bytes,
        )

    def response_cache_profile(self) -> dict[str, int]:
        """Return runtime-only cache telemetry excluded from scientific identity."""

        return {
            "max_bytes": self.response_cache_max_bytes,
            "resident_bytes": self._response_cache_bytes,
            "peak_bytes": self._response_cache_peak_bytes,
            "entry_count": len(self._response_cache),
            "hits": self._response_cache_hits,
            "misses": self._response_cache_misses,
            "evictions": self._response_cache_evictions,
            "panel_draws": self._panel_draws,
            "panel_draw_distinct": len(self._panel_draw_counts),
            "panel_draw_repeats": self._panel_draw_repeats,
            "panel_draw_memo_hits": self._panel_draw_memo_hits,
            "panel_draw_memo_entries": len(self._panel_draw_memo),
        }

    def runtime_profile(self) -> dict[str, Any]:
        """Return candidate, calibration, and cache telemetry for receipts."""

        return {
            "backend_id": self.backend_id,
            "scientific_identity": "excluded",
            "runtime_execution_profile": self.runtime_execution_profile.to_runtime_dict(),
            "runtime_execution_session": self.runtime_execution_session.runtime_receipt(),
            "resource_usage": dict(self.resource_usage),
            **self.profiler.snapshot(),
            "response_cache": self.response_cache_profile(),
            "evaluation_reuse": {
                "incremental_responses": self.incremental_responses,
                "incremental_rows_reused": self._incremental_rows_reused,
                "incremental_rows_forwarded": self._incremental_rows_forwarded,
                "crossfit_cache_enabled": self.crossfit_cache_enabled,
                "crossfit_cache_hits": self._crossfit_cache_hits,
                "crossfit_cache_misses": self._crossfit_cache_misses,
            },
            "permutation_memo": {
                "hits": self._permutation_memo.hits,
                "misses": self._permutation_memo.misses,
                "entries": len(self._permutation_memo),
            },
            "execution_batch": {
                "rows": self._execution_batch_rows,
                "provenance": self._execution_batch_provenance,
                "coalesce_rows": self._coalesce_rows,
            },
            "candidate_execution_backend": (
                self.evaluator.candidate_execution_profile()
            ),
        }

    def _fit_calibration_with_runtime_context(self, **kwargs: Any) -> Any:
        """Fit one exact calibration input under the run's shared profile.

        A prepared calibration state is valid only for its exact ordered
        panel/readout tensor. Each fit therefore receives a fresh context;
        the enclosing session aggregates operational evidence without turning
        that state into a scientific or cross-run cache.
        """

        context = self.runtime_execution_session.new_calibration_context()
        self._ensure_resource_accounting()
        with self._resource_lock:
            self._resource_calibration_fit_calls += 1
        started = time.perf_counter()
        try:
            return self.calibration_backend.fit_problem(
                execution_context=context,
                **kwargs,
            )
        finally:
            context.record_phase(
                "calibration.fit_problem", time.perf_counter() - started
            )
            self.runtime_execution_session.record_calibration_context(context)

    def _compile_mask_cached(
        self, mask_hash: str, masks: Mapping[str, np.ndarray]
    ) -> Any:
        compiled = self._compiled_mask_cache.get(mask_hash)
        if compiled is not None:
            self._compiled_mask_cache_hits += 1
            return compiled
        self._compiled_mask_cache_misses += 1
        compiled = self.evaluator.compile_mask(masks)
        self._compiled_mask_cache.put(mask_hash, compiled)
        return compiled

    def _panel_draw_memo_key(self, request: SampledComparisonRequest) -> str:
        from contracts.hashing import stable_digest

        prior = getattr(request.prior_panel, "panel_hash", None)
        return stable_digest(
            {
                "query_id": request.query_id,
                "layer_id": request.layer_id,
                "partition_id": request.partition_id,
                "previous_density": float(request.previous_density),
                "proposed_density": float(request.proposed_density),
                "stage_index": int(request.stage_index),
                "stage_per_class": int(request.stage_per_class),
                "cumulative_target_images": request_cumulative_target_images(request),
                "per_stratum_target_images": dict(
                    request_per_stratum_target_images(request)
                ),
                "proposal_index": int(request.proposal_index),
                "random_seed": int(request.random_seed),
                "fold_count": int(request.fold_count),
                "prior_panel_hash": prior,
                # Over-keying is safe (hit-rate cost only); under-keying
                # is not — every mapping that could steer a draw is in.
                "committed_warm_start": stable_digest(
                    dict(request.committed_warm_start or {})
                ),
                "candidate_warm_start": stable_digest(
                    dict(request.candidate_warm_start or {})
                ),
                "allocation_evidence": stable_digest(
                    dict(request.allocation_evidence or {})
                ),
                "policy_active": self.panel_policy is not None,
            }
        )

    def _draw_panel(self, request: SampledComparisonRequest) -> CrossfitPanel:
        query = self._query_by_id.get(request.query_id)
        if query is None:
            raise KeyError(request.query_id)
        memo_limit = int(self.runtime_execution_profile.panel_draw_memo_entries)
        memo_key: str | None = None
        if memo_limit > 0:
            memo_key = self._panel_draw_memo_key(request)
            hit = self._panel_draw_memo.get(memo_key)
            if hit is not None:
                self._panel_draw_memo.move_to_end(memo_key)
                self._panel_draw_memo_hits += 1
                self._panel_draws += 1
                self._panel_draw_repeats += 1
                return hit
        if self.panel_policy is None:
            panel = draw_nested_class_balanced_panel(
                self.panel_source, request, permutation_memo=self._permutation_memo
            )
        else:
            planner = self._panel_planners.get(request.query_id)
            if planner is None:
                planner = BinaryCostOptimalNeymanPanelPlanner(
                    self.panel_source.context_axis,
                    query_id=request.query_id,
                    target_class_id=_query_class_id(query),
                    policy=self.panel_policy,
                )
                self._panel_planners[request.query_id] = planner
            panel = planner.draw(
                request,
                previous_state=NeymanEvidenceState.from_mapping(
                    request.allocation_evidence
                ),
            ).panel
        # Sharing telemetry only: prices panel-draw memoization.
        # Counting is decision-inert; the drawn panel is returned unchanged.
        panel_hash = getattr(panel, "panel_hash", None)
        if panel_hash is not None:
            seen = self._panel_draw_counts.get(panel_hash, 0) + 1
            self._panel_draw_counts[panel_hash] = seen
            self._panel_draws += 1
            if seen > 1:
                self._panel_draw_repeats += 1
        if memo_key is not None:
            self._panel_draw_memo[memo_key] = panel
            self._panel_draw_memo.move_to_end(memo_key)
            while len(self._panel_draw_memo) > memo_limit:
                self._panel_draw_memo.popitem(last=False)
        return panel

    def _execute(
        self,
        masks: Mapping[str, np.ndarray],
        panel: CrossfitPanel,
        *,
        query_id: str,
        mask_hash: str | None = None,
    ) -> ReadoutResponse:
        # Callers that already hold the (loop-invariant) mask identity pass it in;
        # recomputing the hash per call was measurable per-look Python overhead.
        if mask_hash is None:
            mask_hash = sampled_path_mask_hash(masks)
        key = (mask_hash, panel.panel_hash)
        with self.profiler.phase("candidate.cache_lookup"):
            cached = self._response_cache.get(key)
        if cached is not None:
            self._response_cache_hits += 1
            self._response_cache.move_to_end(key)
            return cached[0]
        self._response_cache_misses += 1
        if self.incremental_responses:
            with self.profiler.phase("candidate.incremental_assembly"):
                assembled = self._assemble_incremental_response(
                    mask_hash, panel, masks=masks, query_id=query_id
                )
            if assembled is not None:
                with self.profiler.phase("candidate.cache_store"):
                    self._cache_response(key, assembled)
                return assembled
        with self.profiler.phase("candidate.mask_compile"):
            compiled = self._compile_mask_cached(mask_hash, masks)
        batches_iterable = self._panel_batches_iterable(panel, query_id=query_id)
        counters_before = self._candidate_backend_counters()
        try:
            with self.profiler.phase("candidate.masked_forward"):
                response = self.evaluator.readout_response_from_batches(
                    batches_iterable,
                    compiled_mask=compiled,
                )
        except BaseException:
            self._record_failed_forward_attempt(counters_before)
            raise
        counters_after = self._candidate_backend_counters()
        self._record_forward(
            query_id=query_id,
            mask_hashes=(mask_hash,),
            context_ids=panel.context_ids,
            physical_calls=self._counter_delta(counters_before[0], counters_after[0]),
            backward_calls=self._counter_delta(counters_before[1], counters_after[1]),
        )
        if response.context_ids != panel.context_ids:
            raise ValueError("sampled mask execution changed the frozen panel order")
        self._record_response_materialization(response, include_labels=True)
        with self.profiler.phase("candidate.cache_store"):
            self._cache_response(key, response)
        if self.incremental_responses:
            self._row_store_ingest(mask_hash, response)
        return response

    def _panel_batches_iterable(
        self, panel: CrossfitPanel, *, query_id: str
    ) -> Iterable[MeasurementInputBatch]:
        with self.profiler.phase("candidate.panel_batches"):
            content_key = self._panel_keys(panel)[0]
            with self._panel_cache_lock:
                cache_entry = self._panel_batch_cache.get(content_key)
                if cache_entry is not None:
                    self._panel_batch_cache_hits += 1
                    self._panel_batch_cache.move_to_end(content_key)
            if cache_entry is not None:
                return cache_entry[0]
            self._panel_batch_cache_misses += 1
            # First touch streams: gather and host-to-device issue run on
            # a background thread while this panel's own forwards
            # consume, and the completed device-resident list publishes
            # to the cache for every replay (CRN second mask, staged
            # re-looks, content-identical proposal panels — all HBM
            # reads, zero host or PCIe involvement).
            from packages.loader_runtime.throughput import (
                BackgroundIterator,
            )

            return BackgroundIterator(
                self._device_batch_stream(
                    panel.context_ids, query_id=query_id, content_key=content_key
                ),
                depth=int(self.runtime_execution_profile.sampled_prefetch_depth),
            )

    def _execute_lanes(
        self,
        lanes: Sequence[tuple[str, Mapping[str, np.ndarray]]],
        panel: CrossfitPanel,
        *,
        query_id: str,
    ) -> tuple[ReadoutResponse, ...]:
        """Evaluate K mask lanes over one shared panel pass.

        Cache and incremental-assembly hits resolve per lane; every lane that
        still needs a forward joins ONE fused pass over the panel stream (the
        CRN pair's committed+candidate share each input batch), so a pair
        costs at most one stream traversal instead of two.
        """

        results: dict[int, ReadoutResponse] = {}
        missing: list[int] = []
        for index, (mask_hash, masks) in enumerate(lanes):
            key = (mask_hash, panel.panel_hash)
            with self.profiler.phase("candidate.cache_lookup"):
                cached = self._response_cache.get(key)
            if cached is not None:
                self._response_cache_hits += 1
                self._response_cache.move_to_end(key)
                results[index] = cached[0]
                continue
            self._response_cache_misses += 1
            if self.incremental_responses:
                with self.profiler.phase("candidate.incremental_assembly"):
                    assembled = self._assemble_incremental_response(
                        mask_hash, panel, masks=masks, query_id=query_id
                    )
                if assembled is not None:
                    with self.profiler.phase("candidate.cache_store"):
                        self._cache_response(key, assembled)
                    results[index] = assembled
                    continue
            missing.append(index)
        if missing:
            with self.profiler.phase("candidate.mask_compile"):
                compiled = [
                    self._compile_mask_cached(lanes[i][0], lanes[i][1]) for i in missing
                ]
            fused = getattr(self.evaluator, "readout_responses_from_batches", None)
            if fused is not None:
                batches_iterable = self._panel_batches_iterable(
                    panel, query_id=query_id
                )
                counters_before = self._candidate_backend_counters()
                try:
                    with self.profiler.phase("candidate.masked_forward"):
                        responses = fused(
                            batches_iterable,
                            compiled_masks=compiled,
                            lane_keys=[lanes[i][0] for i in missing],
                        )
                except BaseException:
                    self._record_failed_forward_attempt(counters_before)
                    raise
                counters_after = self._candidate_backend_counters()
                self._record_forward(
                    query_id=query_id,
                    mask_hashes=tuple(lanes[i][0] for i in missing),
                    context_ids=panel.context_ids,
                    physical_calls=self._counter_delta(
                        counters_before[0], counters_after[0]
                    ),
                    backward_calls=self._counter_delta(
                        counters_before[1], counters_after[1]
                    ),
                )
            else:
                # Evaluator stubs (tests) implement only the single-mask
                # surface; run per-lane passes with the historical semantics.
                responses = []
                for lane_offset, lane_index in enumerate(missing):
                    batches_iterable = self._panel_batches_iterable(
                        panel, query_id=query_id
                    )
                    counters_before = self._candidate_backend_counters()
                    try:
                        with self.profiler.phase("candidate.masked_forward"):
                            responses.append(
                                self.evaluator.readout_response_from_batches(
                                    batches_iterable,
                                    compiled_mask=compiled[lane_offset],
                                )
                            )
                    except BaseException:
                        self._record_failed_forward_attempt(counters_before)
                        raise
                    counters_after = self._candidate_backend_counters()
                    self._record_forward(
                        query_id=query_id,
                        mask_hashes=(lanes[lane_index][0],),
                        context_ids=panel.context_ids,
                        physical_calls=self._counter_delta(
                            counters_before[0], counters_after[0]
                        ),
                        backward_calls=self._counter_delta(
                            counters_before[1], counters_after[1]
                        ),
                    )
            for response_offset, (lane_index, response) in enumerate(
                zip(missing, responses, strict=True)
            ):
                if response.context_ids != panel.context_ids:
                    raise ValueError(
                        "sampled mask execution changed the frozen panel order"
                    )
                self._record_response_materialization(
                    response,
                    include_labels=response_offset == 0,
                )
                mask_hash = lanes[lane_index][0]
                with self.profiler.phase("candidate.cache_store"):
                    self._cache_response((mask_hash, panel.panel_hash), response)
                if self.incremental_responses:
                    self._row_store_ingest(mask_hash, response)
                results[lane_index] = response
        return tuple(results[index] for index in range(len(lanes)))

    @staticmethod
    def _panel_content_key(context_ids: Sequence[str]) -> str:
        return hashlib.sha256("\x1f".join(context_ids).encode("utf-8")).hexdigest()

    def _panel_keys(self, panel: CrossfitPanel) -> tuple[str, str]:
        """Memoized (content_key, fold_structure_key) for a panel.

        Both are pure functions of the panel; recomputing the sha256 digests
        for each lane of every pair was measurable per-look Python overhead.
        """

        memo_key = getattr(panel, "panel_hash", None)
        if memo_key is not None:
            cached = self._panel_key_memo.get(memo_key)
            if cached is not None:
                self._panel_key_memo.move_to_end(memo_key)
                return cached
        content_key = self._panel_content_key(panel.context_ids)
        fold_structure = hashlib.sha256(
            "\x1f".join(
                (
                    ",".join(str(value) for value in panel.fold_ids),
                    ",".join(
                        f"{float(value):.12g}"
                        for value in (panel.sampling_weights or ())
                    ),
                )
            ).encode("utf-8")
        ).hexdigest()
        keys = (content_key, fold_structure)
        if memo_key is not None:
            self._panel_key_memo[memo_key] = keys
            while len(self._panel_key_memo) > self._panel_key_memo_limit:
                self._panel_key_memo.popitem(last=False)
        return keys

    def _fold_validity_stack(
        self, panel: CrossfitPanel, *, fold_count: int
    ) -> tuple[np.ndarray, Any]:
        """Memoized [fold, sample] validity matrix for a panel.

        Returns ``(array, tensor)`` — one read-only float64 ndarray plus the
        CPU torch view sharing its storage. The tensor is what fits consume
        (``torch.as_tensor`` is an identity on CPU and a device copy on
        CUDA); the read-only flag turns any accidental host-side mutation of
        the shared cache entry into an error instead of silent corruption.
        """

        _, fold_structure = self._panel_keys(panel)
        memo_key = (fold_structure, int(fold_count))
        cached = self._fold_validity_memo.get(memo_key)
        if cached is not None:
            self._fold_validity_memo.move_to_end(memo_key)
            return cached
        fold_ids_host = np.asarray(panel.fold_ids)
        stack = np.stack(
            [
                (fold_ids_host != fold).astype(np.float64)
                for fold in range(int(fold_count))
            ]
        )
        tensor = torch.from_numpy(stack)
        stack.setflags(write=False)
        value = (stack, tensor)
        self._fold_validity_memo[memo_key] = value
        while len(self._fold_validity_memo) > self._panel_assembly_memo_limit:
            self._fold_validity_memo.popitem(last=False)
        return value

    def _fold_context_id_lists(
        self, panel: CrossfitPanel, *, fold_count: int
    ) -> tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]:
        """Memoized per-fold (calibration, extraction) context-id tuples."""

        content_key, fold_structure = self._panel_keys(panel)
        memo_key = (content_key, fold_structure, int(fold_count))
        cached = self._fold_context_ids_memo.get(memo_key)
        if cached is not None:
            self._fold_context_ids_memo.move_to_end(memo_key)
            return cached
        lists = tuple(
            (
                tuple(
                    panel.context_ids[int(index)]
                    for index in panel.calibration_indices(fold)
                ),
                tuple(
                    panel.context_ids[int(index)]
                    for index in panel.extraction_indices(fold)
                ),
            )
            for fold in range(int(fold_count))
        )
        self._fold_context_ids_memo[memo_key] = lists
        while len(self._fold_context_ids_memo) > self._panel_assembly_memo_limit:
            self._fold_context_ids_memo.popitem(last=False)
        return lists

    def _row_store_ingest(self, mask_hash: str, response: ReadoutResponse) -> None:
        entry = self._mask_row_store.get(mask_hash)
        if entry is None:
            entry = {
                "index": {},
                "count": 0,
                "bytes": 0,
                "flat": None,
                "flat_labels": None,
            }
            self._mask_row_store[mask_hash] = entry
        self._mask_row_store.move_to_end(mask_hash)
        index = entry["index"]
        fresh_positions = [
            position
            for position, context_id in enumerate(response.context_ids)
            if context_id not in index
        ]
        if not fresh_positions:
            return
        values = np.asarray(response.values, dtype=np.float32)
        labels = np.asarray(response.labels).reshape(-1)
        block = values[fresh_positions]
        label_block = labels[fresh_positions]
        base = int(entry["count"])
        grown_count = base + len(fresh_positions)
        # Amortized-growth flat store: rows are written in place at ingest so
        # assembly never re-concatenates the whole store (in a production
        # receipt the invalidate-and-reconcatenate flat dominated assembly
        # wall).  Row contents are identical to block-wise concatenation, so
        # gathers stay byte-for-byte identical.
        flat = entry["flat"]
        needs_grow = flat is None or flat.shape[0] < grown_count
        labels_dtype = (
            label_block.dtype
            if flat is None
            else np.promote_types(entry["flat_labels"].dtype, label_block.dtype)
        )
        if not needs_grow and labels_dtype != entry["flat_labels"].dtype:
            needs_grow = True
        if needs_grow:
            capacity = max(grown_count, 2 * (flat.shape[0] if flat is not None else 0))
            new_flat = np.empty((capacity, values.shape[1]), dtype=np.float32)
            new_labels = np.empty((capacity,), dtype=labels_dtype)
            if base:
                new_flat[:base] = entry["flat"][:base]
                new_labels[:base] = entry["flat_labels"][:base]
            entry["flat"] = flat = new_flat
            entry["flat_labels"] = new_labels
        flat[base:grown_count] = block
        entry["flat_labels"][base:grown_count] = label_block
        for offset, position in enumerate(fresh_positions):
            index[response.context_ids[position]] = base + offset
        entry["count"] = grown_count
        grown = int(block.nbytes + label_block.nbytes)
        entry["bytes"] += grown
        self._mask_row_store_bytes += grown
        while (
            self._mask_row_store_bytes > self._mask_row_store_byte_limit
            and len(self._mask_row_store) > 1
        ):
            evicted_hash, evicted = self._mask_row_store.popitem(last=False)
            if evicted_hash == mask_hash:
                self._mask_row_store[evicted_hash] = evicted
                self._mask_row_store.move_to_end(evicted_hash, last=False)
                break
            self._mask_row_store_bytes -= int(evicted["bytes"])

    def _assemble_incremental_response(
        self,
        mask_hash: str,
        panel: CrossfitPanel,
        *,
        masks: Mapping[str, np.ndarray],
        query_id: str,
    ) -> ReadoutResponse | None:
        """Rebuild a panel response from the mask's row store plus the increment.

        Eval-mode readouts are per-context deterministic, so rows evaluated
        under the same mask on an earlier (nested) panel are exact. Returns
        None when nothing is reusable — the caller takes the full path.
        """

        entry = self._mask_row_store.get(mask_hash)
        if entry is None:
            return None
        index = entry["index"]
        missing = tuple(
            context_id for context_id in panel.context_ids if context_id not in index
        )
        if len(missing) == len(panel.context_ids):
            return None
        if missing:
            with self.profiler.phase("candidate.mask_compile"):
                compiled = self._compile_mask_cached(mask_hash, masks)
            from packages.loader_runtime.throughput import (
                BackgroundIterator,
            )

            increment_key = self._panel_content_key(missing)
            counters_before = self._candidate_backend_counters()
            try:
                with self.profiler.phase("candidate.masked_forward"):
                    fresh = self.evaluator.readout_response_from_batches(
                        BackgroundIterator(
                            self._device_batch_stream(
                                missing,
                                query_id=query_id,
                                content_key=increment_key,
                            ),
                            depth=int(
                                self.runtime_execution_profile.sampled_prefetch_depth
                            ),
                        ),
                        compiled_mask=compiled,
                    )
            except BaseException:
                self._record_failed_forward_attempt(counters_before)
                raise
            counters_after = self._candidate_backend_counters()
            self._record_forward(
                query_id=query_id,
                mask_hashes=(mask_hash,),
                context_ids=missing,
                physical_calls=self._counter_delta(
                    counters_before[0], counters_after[0]
                ),
                backward_calls=self._counter_delta(
                    counters_before[1], counters_after[1]
                ),
            )
            self._record_response_materialization(fresh, include_labels=True)
            self._row_store_ingest(mask_hash, fresh)
        self._incremental_rows_reused += len(panel.context_ids) - len(missing)
        self._incremental_rows_forwarded += len(missing)
        # Pure host-side gather (separately spanned so receipts can split the
        # enclosing incremental_assembly phase from its nested increment
        # forwards, which double-count into candidate.masked_forward above).
        with self.profiler.phase("candidate.assembly_gather"):
            count = int(entry["count"])
            take = np.fromiter(
                (index[context_id] for context_id in panel.context_ids),
                dtype=np.int64,
                count=len(panel.context_ids),
            )
            return ReadoutResponse(
                context_ids=tuple(panel.context_ids),
                values=entry["flat"][:count].take(take, axis=0),
                labels=entry["flat_labels"][:count].take(take, axis=0),
            )

    def _host_panel_batches(self, context_ids: tuple[str, ...], *, query_id: str):
        """Panel batches in canonical order, optionally pool-staged.

        With ``sampled_producer_workers > 1``, context IDs are chunked into
        batch_size-aligned spans and each span's gather/decode runs on a
        worker pool; spans EMIT strictly in canonical order, so the batch
        sequence and content are the plain path's exactly (the span
        partition is a pure function of the ID tuple — contexts._batches).
        Missing batch-size discovery falls back to the plain path with a
        recorded reason rather than guessing.
        """

        workers = int(self.runtime_execution_profile.sampled_producer_workers)
        if workers <= 1:
            return self.panel_source.iter_panel_batches(context_ids, query_id=query_id)
        batch_size = getattr(
            getattr(self.panel_source, "source", None), "batch_size", None
        )
        if not isinstance(batch_size, int) or batch_size < 1:
            self._producer_pool_fallback_reason = "batch_size_undiscoverable"
            return self.panel_source.iter_panel_batches(context_ids, query_id=query_id)
        self._producer_pool_fallback_reason = ""
        spans = [
            context_ids[start : start + batch_size]
            for start in range(0, len(context_ids), batch_size)
        ]

        def _stage(span: tuple[str, ...]) -> list:
            return list(self.panel_source.iter_panel_batches(span, query_id=query_id))

        from packages.loader_runtime.throughput import PooledBackgroundIterator

        depth = int(self.runtime_execution_profile.sampled_prefetch_depth)
        pooled = PooledBackgroundIterator(
            iter(spans), _stage, workers=workers, depth=depth
        )

        def _flatten():
            for staged in pooled:
                yield from staged

        return _flatten()

    def _device_batch_stream(
        self,
        context_ids: Sequence[str],
        *,
        query_id: str,
        content_key: str,
    ):
        """Yield device-resident batches, publishing the panel on completion."""

        device = self.evaluator.device
        collected: list[MeasurementInputBatch] = []
        pending: list[MeasurementInputBatch] = []
        pending_rows = 0

        def _flush() -> MeasurementInputBatch | None:
            nonlocal pending, pending_rows
            if not pending:
                return None
            if len(pending) == 1:
                merged = pending[0]
            else:
                merged = MeasurementInputBatch(
                    context_ids=tuple(
                        context_id
                        for item in pending
                        for context_id in item.context_ids
                    ),
                    inputs=torch.cat([item.inputs for item in pending], dim=0),
                    labels=(
                        torch.cat([item.labels for item in pending], dim=0)
                        if pending[0].labels is not None
                        else None
                    ),
                )
            pending = []
            pending_rows = 0
            collected.append(merged)
            return merged

        for batch in self._host_panel_batches(tuple(context_ids), query_id=query_id):
            input_moves_h2d = torch.is_tensor(
                batch.inputs
            ) and self._is_host_to_device_move(batch.inputs, device)
            labels_tensor = (
                torch.as_tensor(batch.labels) if batch.labels is not None else None
            )
            label_moves_h2d = (
                labels_tensor is not None
                and self._is_host_to_device_move(labels_tensor, device)
            )
            moved = MeasurementInputBatch(
                context_ids=batch.context_ids,
                inputs=(
                    batch.inputs.to(device=device, non_blocking=True)
                    if torch.is_tensor(batch.inputs)
                    else batch.inputs
                ),
                labels=(
                    labels_tensor.to(device=device, non_blocking=True)
                    if labels_tensor is not None
                    else None
                ),
            )
            transferred = 0
            if input_moves_h2d:
                transferred += self._tensor_nbytes(batch.inputs)
            if label_moves_h2d:
                transferred += self._tensor_nbytes(labels_tensor)
            if transferred:
                self._record_transfer(host_to_device_bytes=transferred)
            if self._coalesce_rows <= 1 or not torch.is_tensor(moved.inputs):
                collected.append(moved)
                yield moved
                continue
            pending.append(moved)
            pending_rows += len(moved.context_ids)
            if pending_rows >= self._coalesce_rows:
                merged = _flush()
                if merged is not None:
                    yield merged
        merged = _flush()
        if merged is not None:
            yield merged
        self._publish_panel_entry(content_key, collected)

    def _publish_panel_entry(
        self, content_key: str, batches: list[MeasurementInputBatch]
    ) -> None:
        entry_bytes = sum(
            int(batch.inputs.nbytes)
            for batch in batches
            if torch.is_tensor(batch.inputs)
        )
        with self._panel_cache_lock:
            if content_key in self._panel_batch_cache:
                return
            self._panel_batch_cache[content_key] = (batches, entry_bytes)
            self._panel_batch_cache_bytes += entry_bytes
            while (
                self._panel_batch_cache_bytes > self._panel_batch_cache_byte_limit
                and len(self._panel_batch_cache) > 1
            ):
                _evicted_key, (_evicted, evicted_bytes) = (
                    self._panel_batch_cache.popitem(last=False)
                )
                self._panel_batch_cache_bytes -= evicted_bytes

    def _ensure_prefetch_worker(self) -> None:
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            return
        self._prefetch_queue = queue.Queue()

        def _worker() -> None:
            while True:
                job = self._prefetch_queue.get()
                if job is None:
                    return
                content_key, context_ids, query_id = job
                try:
                    for _batch in self._device_batch_stream(
                        context_ids, query_id=query_id, content_key=content_key
                    ):
                        pass
                except Exception:
                    self._ensure_resource_accounting()
                    with self._resource_lock:
                        self._resource_prefetch_failed += 1
                    pass  # prefetch is advisory; the demand path re-reads
                else:
                    self._ensure_resource_accounting()
                    with self._resource_lock:
                        self._resource_prefetch_completed += 1
                        self._resource_prefetched_context_ids.update(context_ids)
                finally:
                    with self._panel_cache_lock:
                        self._prefetch_inflight.discard(content_key)

        self._prefetch_thread = threading.Thread(target=_worker, daemon=True)
        self._prefetch_thread.start()

    def _schedule_prefetch_panel(
        self,
        panel: CrossfitPanel,
        request: SampledComparisonRequest,
    ) -> bool:
        """Schedule one already-resolved panel after caller-side authorization."""

        content_key = self._panel_keys(panel)[0]
        with self._panel_cache_lock:
            if content_key in self._panel_batch_cache:
                with self._resource_lock:
                    self._resource_prefetch_cache_skips += 1
                return False
            if content_key in self._prefetch_inflight:
                with self._resource_lock:
                    self._resource_prefetch_inflight_skips += 1
                return False
            self._prefetch_inflight.add(content_key)
        with self._resource_lock:
            self._resource_prefetch_scheduled += 1
        self._ensure_prefetch_worker()
        self._prefetch_queue.put(
            (content_key, tuple(panel.context_ids), request.query_id)
        )
        return True

    def prefetch_request(self, request: SampledComparisonRequest) -> bool:
        """Historical v1/v2 advisory prefetch for a backend-drawn panel.

        Panels are mask-independent, so the engines can hint the next look,
        the next proposal, or the warm-start panel at job start (concurrent
        with model setup and autotune) — the fill runs during current
        compute, and a mispredicted hint costs nothing beyond cached bytes.

        This surface intentionally retains historical semantics. Claim-mode
        glocal execution must use :meth:`prefetch_glocal_panel`, whose exact
        panel and resource reservation are explicit.
        """

        self._ensure_resource_accounting()
        with self._resource_lock:
            self._resource_prefetch_requests += 1
        try:
            with self.profiler.phase("candidate.panel_draw"):
                panel = self._draw_panel(request)
        except Exception:
            with self._resource_lock:
                self._resource_prefetch_failed += 1
            return False
        # Warms the panel-key memo for the demand path (the prefetch request
        # draws the next look's panel before evaluate_pair does).
        return self._schedule_prefetch_panel(panel, request)

    def prefetch_glocal_panel(
        self,
        request: SampledComparisonRequest,
        *,
        panel: CrossfitPanel,
        reservation_token: str,
    ) -> bool:
        """Prefetch an exact claim panel under a one-shot resource reservation.

        A controller, not the backend, mints the opaque token after reserving
        the panel's costs.  Tokens are consumed even when the panel is already
        cached: a claim-bearing planner may never recover budget by learning
        runtime cache state after making its predictable acquisition choice.
        """

        self._ensure_resource_accounting()
        token = str(reservation_token).strip()
        if not token:
            with self._resource_lock:
                self._resource_glocal_prefetch_unreserved_rejections += 1
            raise ValueError(
                "claim-mode prefetch requires an explicit resource reservation token"
            )
        query = self._query_by_id.get(request.query_id)
        if query is None:
            raise KeyError(request.query_id)
        panel.validate(fold_count=request.fold_count)
        self._validate_explicit_glocal_panel_request(
            panel,
            request,
            target_class_id=_query_class_id(query),
        )
        token_hash = hashlib.sha256(
            "\x1f".join(
                (
                    token,
                    str(request.query_id),
                    str(request.proposal_index),
                    str(request.stage_index),
                    panel.panel_hash,
                )
            ).encode("utf-8")
        ).hexdigest()
        with self._resource_lock:
            self._resource_prefetch_requests += 1
            if token_hash in self._resource_glocal_prefetch_reservations_consumed:
                self._resource_glocal_prefetch_unreserved_rejections += 1
                raise ValueError(
                    "claim-mode prefetch reservation token was already consumed"
                )
            self._resource_glocal_prefetch_reservations_consumed.add(token_hash)
        return self._schedule_prefetch_panel(panel, request)

    # The t=0 warmup alias: call at job start, concurrent with model
    # placement and the batch autotune, so the first decision begins with
    # its panel already device-resident.
    start_warmup_prefetch = prefetch_request

    def _crossfit_losses_cached(
        self,
        response: ReadoutResponse,
        panel: CrossfitPanel,
        *,
        mask_hash: str,
        query: QuerySpec,
        class_id: int,
        fold_count: int,
        warm_start: Mapping[str, Any] | None,
    ) -> tuple[np.ndarray, tuple[Mapping[str, Any], ...]]:
        """Reuse fits for byte-identical (mask, panel, fold-structure) inputs.

        The committed side of every CRN pair re-fits the same problem across
        proposals; identical inputs converge to the same optimum, so the
        cached losses are exact (warm starts only change the path, and only
        when the fitter early-stops).
        """

        if not self.crossfit_cache_enabled:
            return self._crossfit_losses(
                response,
                panel,
                query=query,
                class_id=class_id,
                fold_count=fold_count,
                warm_start=warm_start,
            )
        content_key, fold_structure = self._panel_keys(panel)
        cache_key = (
            mask_hash,
            content_key,
            fold_structure,
            int(fold_count),
            int(class_id),
            str(query.query_id),
        )
        cached = self._crossfit_cache.get(cache_key)
        if cached is not None:
            self._crossfit_cache_hits += 1
            self._crossfit_cache.move_to_end(cache_key)
            losses, fits = cached
            return np.array(losses, copy=True), fits
        self._crossfit_cache_misses += 1
        losses, fits = self._crossfit_losses(
            response,
            panel,
            query=query,
            class_id=class_id,
            fold_count=fold_count,
            warm_start=warm_start,
        )
        self._crossfit_cache[cache_key] = (np.array(losses, copy=True), fits)
        while len(self._crossfit_cache) > self._crossfit_cache_limit:
            self._crossfit_cache.popitem(last=False)
        return losses, fits

    def _readout_columns_policy_key(self) -> Any:
        policy = self.calibration_options.get("readout_columns", "target_only")
        if isinstance(policy, str):
            return policy
        return tuple(int(value) for value in policy)

    def _resolve_readout_columns(
        self, panel: CrossfitPanel, *, class_id: int, fold_count: int
    ) -> tuple[int, ...]:
        """Memoized column resolution (pure function of panel/class/policy)."""

        content_key, fold_structure = self._panel_keys(panel)
        memo_key = (
            content_key,
            fold_structure,
            int(fold_count),
            int(class_id),
            self._readout_columns_policy_key(),
        )
        cached = self._readout_columns_memo.get(memo_key)
        if cached is not None:
            self._readout_columns_memo.move_to_end(memo_key)
            return cached
        columns = self._resolve_readout_columns_direct(
            panel, class_id=class_id, fold_count=fold_count
        )
        self._readout_columns_memo[memo_key] = columns
        while len(self._readout_columns_memo) > self._panel_assembly_memo_limit:
            self._readout_columns_memo.popitem(last=False)
        return columns

    def _resolve_readout_columns_direct(
        self, panel: CrossfitPanel, *, class_id: int, fold_count: int
    ) -> tuple[int, ...]:
        """Resolve the consumed readout column set from the declared policy.

        ``target_only`` is the single decision column. ``all`` means all
        ESTIMABLE columns: every fold's calibration split must retain at
        least ``CANONICAL_CALIBRATION_SELECTION_BLOCKS`` contexts of the
        column's class and of its complement (guaranteeing both classes in
        every inner selection block). An explicit tuple fails closed on any
        unsupported requested column. The resolution is deterministic in the
        panel and receipted through the crossfit-cache key.
        """

        policy = self.calibration_options.get("readout_columns", "target_only")
        if policy == "target_only":
            return (int(class_id),)
        labels_host = np.asarray(panel.labels)
        fold_ids_host = np.asarray(panel.fold_ids)
        total = int(labels_host.shape[0])
        minimum = int(CANONICAL_CALIBRATION_SELECTION_BLOCKS)

        def _estimable(column: int) -> bool:
            positives = labels_host == int(column)
            for fold in range(int(fold_count)):
                in_fold = fold_ids_host == fold
                fit_pos = int((positives & ~in_fold).sum())
                fit_total = int((~in_fold).sum())
                if fit_pos < minimum or (fit_total - fit_pos) < minimum:
                    return False
            return True

        if policy == "all":
            observed = sorted(int(value) for value in set(labels_host.tolist()))
            columns = tuple(column for column in observed if _estimable(column))
            if int(class_id) not in columns:
                # The decision column must always be present (its estimability
                # is enforced fail-closed by the fit itself).
                columns = tuple(sorted({*columns, int(class_id)}))
            return columns
        requested = tuple(int(value) for value in policy)
        if int(class_id) not in requested:
            raise ValueError("readout column set must include the target class")
        for column in requested:
            if column != int(class_id) and not _estimable(column):
                raise ValueError(
                    f"requested readout column {column} lacks per-fold support"
                )
        return tuple(sorted(set(requested)))

    def _crossfit_losses(
        self,
        response: ReadoutResponse,
        panel: CrossfitPanel,
        *,
        query: QuerySpec,
        class_id: int,
        fold_count: int,
        warm_start: Mapping[str, Any] | None,
    ) -> tuple[np.ndarray, tuple[Mapping[str, Any], ...]]:
        with self.profiler.phase("candidate.readout_stage"):
            # Fresh forwards carry the readout on device; only cache-replayed
            # responses pay the host->device upload.
            readout = getattr(response, "device_values", None)
            if readout is None:
                readout = torch.as_tensor(
                    response.values,
                    device=self.evaluator.device,
                    dtype=torch.float32,
                )
            labels = torch.as_tensor(
                response.labels, device=self.evaluator.device, dtype=torch.long
            )
        if class_id < 0 or class_id >= int(readout.shape[1]):
            raise ValueError("query class coordinate lies outside the model readout")
        contract = str(self.calibration_options["optimization_contract"])
        if contract != SAMPLED_PATH_V1_CALIBRATION_ALGORITHM:
            return self._crossfit_losses_batched(
                response,
                panel,
                readout=readout,
                labels=labels,
                query=query,
                class_id=class_id,
                fold_count=fold_count,
                warm_start=warm_start,
            )
        return self._crossfit_losses_sequential(
            response,
            panel,
            readout=readout,
            labels=labels,
            query=query,
            class_id=class_id,
            fold_count=fold_count,
            warm_start=warm_start,
        )

    def _crossfit_losses_batched(
        self,
        response: ReadoutResponse,
        panel: CrossfitPanel,
        *,
        readout: Any,
        labels: Any,
        query: QuerySpec,
        class_id: int,
        fold_count: int,
        warm_start: Mapping[str, Any] | None,
    ) -> tuple[np.ndarray, tuple[Mapping[str, Any], ...]]:
        """One two-axis batched fit per (mask, look): folds x readout columns.

        The Python fold loop is gone — fold membership enters the fit as a
        [fold, sample] validity matrix (zero-weight rows contribute nothing),
        the consumed column set enters as the query axis, and the decision
        losses are the target column of the held-out gather. Non-target
        columns are enrichment: fitted, receipted, never gating.
        """

        with self.profiler.phase("candidate.readout_stage"):
            column_ids = self._resolve_readout_columns(
                panel, class_id=class_id, fold_count=fold_count
            )
            target_position = column_ids.index(int(class_id))
            fold_ids_host = np.asarray(panel.fold_ids)
            # Memoized [fold, sample] validity; the torch view is what
            # the fit consumes, the ndarray is read-only by contract.
            _, fold_validity = self._fold_validity_stack(panel, fold_count=fold_count)
            # Host-side support backstop for the decision column (enrichment
            # columns were pre-filtered by the resolution policy).
            labels_host = np.asarray(response.labels)
            positives_host = labels_host == int(class_id)
            for fold in range(int(fold_count)):
                in_fold = fold_ids_host == fold
                fit_positive = int((positives_host & ~in_fold).sum())
                if fit_positive in {0, int((~in_fold).sum())}:
                    raise ValueError(
                        "each calibration fold must contain target and "
                        "nontarget contexts"
                    )
            column_index = torch.as_tensor(
                column_ids, device=readout.device, dtype=torch.long
            )
            query_values = readout.index_select(1, column_index)
            targets = (labels.unsqueeze(1) == column_index.unsqueeze(0)).to(
                dtype=torch.float32
            )
            panel_weights = (
                torch.as_tensor(
                    panel.sampling_weights,
                    device=readout.device,
                    dtype=torch.float32,
                )
                if panel.sampling_weights
                else None
            )
        fit_options = dict(self.calibration_options)
        fit_options["diagnostic_context"] = (
            f"sampled_path:{panel.panel_hash}:batched_folds"
        )
        proper_score_options = _resolved_proper_score_options(fit_options)
        columns_signature = ",".join(str(value) for value in column_ids)
        contract_key = (
            query.query_id,
            query.source_hash,
            int(readout.shape[1]),
            columns_signature,
            str(fit_options.get("binary_loss_weighting", "pos_weighted")),
            float(fit_options.get("ws_min", 1e-8)),
            int(fold_count),
            str(fit_options["optimization_contract"]),
            tuple(sorted(proper_score_options.items())),
        )
        contracts = self._contract_memo.get(contract_key)
        if contracts is None:
            contracts = _coordinate_calibration_contracts(
                query,
                response_width=int(readout.shape[1]),
                class_id=class_id,
                column_ids=column_ids,
                weighting=str(fit_options.get("binary_loss_weighting", "pos_weighted")),
                minimum_scale=float(fit_options.get("ws_min", 1e-8)),
                fold_ids=tuple(
                    f"calibration_without_fold:{fold}"
                    for fold in range(int(fold_count))
                ),
                optimization_contract=str(fit_options["optimization_contract"]),
                **proper_score_options,
            )
            if len(self._contract_memo) >= 256:
                self._contract_memo.pop(next(iter(self._contract_memo)))
            self._contract_memo[contract_key] = contracts
        readout_spec, query_spec, problem_spec = contracts
        required_mask = np.zeros(len(column_ids), dtype=bool)
        required_mask[target_position] = True
        if isinstance(self.calibration_options.get("readout_columns"), (tuple, list)):
            # Explicitly requested columns are declared intent: fail closed.
            required_mask[:] = True
        audit_seed = int.from_bytes(bytes.fromhex(panel.panel_hash[:16]), "big")
        # Warm starts only shorten the solver path; assemble per-fold priors
        # for the decision column (finite_mle consumes them; v2 ignores).
        initial_scale = None
        initial_offset = None
        if warm_start:
            scale_grid = np.ones((int(fold_count), len(column_ids)))
            offset_grid = np.zeros((int(fold_count), len(column_ids)))
            found = False
            for fold in range(int(fold_count)):
                prior = _fold_warm_start(warm_start, fold)
                if prior.get("readout_scale") is not None:
                    values = np.asarray(prior["readout_scale"], dtype=np.float64)
                    scale_grid[fold, target_position] = float(values.reshape(-1)[0])
                    found = True
                if prior.get("readout_offset") is not None:
                    values = np.asarray(prior["readout_offset"], dtype=np.float64)
                    offset_grid[fold, target_position] = float(values.reshape(-1)[0])
                    found = True
            if found:
                initial_scale = scale_grid
                initial_offset = offset_grid
        # phase() (not record()) so cuda-event timing surfaces the fit's
        # GPU-vs-wall split: the double-buffer decision needs to know whether
        # calibration is GPU-bound or CPU/GIL-bound. Timing only — the fit is
        # unchanged. (When the fit later moves to a worker thread, route it
        # through the session calibration_profiler, not self.profiler.)
        with self.profiler.phase("candidate.calibration_fit"):
            fit = self._fit_calibration_with_runtime_context(
                readout=readout_spec,
                query=query_spec,
                problem=problem_spec,
                query_values=query_values,
                targets=targets,
                sample_weights=panel_weights,
                initial_scale=initial_scale,
                initial_offset=initial_offset,
                fold_validity=fold_validity,
                row_validity=fold_validity,
                required_query_mask=required_mask,
                reference_audit_policy=(
                    "required_plus_sampled" if len(column_ids) > 1 else "full"
                ),
                audit_seed=audit_seed,
                selection_context_ids=panel.context_ids,
                iters=int(
                    fit_options.get("iters", fit_options.get("num_iterations", 64))
                ),
                **proper_score_options,
                require_convergence=True,
            )
        with self.profiler.phase("candidate.readout_stage"):
            # Per-fold application: the 3-D apply branch broadcasts each
            # fold's own (scale, offset) over the shared panel readout.
            calibrated = fit.apply(
                query_values.unsqueeze(0).expand(int(fold_count), -1, -1)
            )
            fold_assignment = torch.as_tensor(
                fold_ids_host, device=readout.device, dtype=torch.long
            )
            # Decision estimand: fold m's parameters score exactly the rows
            # fold m never fitted (its own held-out rows).
            gathered = calibrated.gather(
                0,
                fold_assignment.reshape(1, -1, 1).expand(1, -1, calibrated.shape[2]),
            )[0]
            target_calibrated = gathered[:, target_position]
            target_binary = (labels == int(class_id)).to(torch.float32)
            row_loss = torch.where(
                target_binary > 0.5,
                F.softplus(-target_calibrated),
                F.softplus(target_calibrated),
            )
            losses_host = (
                row_loss.detach().to(device="cpu", dtype=torch.float64).numpy()
            )
            # Enrichment: the cross-category response profile — held-out
            # calibrated loss per column (mean + class-balanced mean).
            profile: dict[str, Any] | None = None
            if len(column_ids) > 1:
                column_binary = targets
                column_loss = torch.where(
                    column_binary > 0.5,
                    F.softplus(-gathered),
                    F.softplus(gathered),
                )
                mean_loss = column_loss.mean(dim=0)
                positives_mass = column_binary.sum(dim=0).clamp_min(1.0)
                negatives_mass = (
                    float(column_binary.shape[0]) - column_binary.sum(dim=0)
                ).clamp_min(1.0)
                positive_mean = (column_loss * column_binary).sum(dim=0) / (
                    positives_mass
                )
                negative_mean = (column_loss * (1.0 - column_binary)).sum(
                    dim=0
                ) / negatives_mass
                balanced = 0.5 * (positive_mean + negative_mean)
                scale_host = np.asarray(
                    fit.scale.detach().to(device="cpu", dtype=torch.float32)
                )
                offset_host = np.asarray(
                    fit.offset.detach().to(device="cpu", dtype=torch.float32)
                )
                profile = {
                    "schema_name": "mechanistic_cross_category_readout_profile",
                    "schema_version": "1",
                    "panel_hash": panel.panel_hash,
                    "column_ids": list(column_ids),
                    "target_class_id": int(class_id),
                    "fold_count": int(fold_count),
                    "scale_by_fold_column": scale_host.tolist(),
                    "offset_by_fold_column": offset_host.tolist(),
                    "heldout_mean_loss_by_column": [
                        float(value) for value in mean_loss.tolist()
                    ],
                    "heldout_balanced_loss_by_column": [
                        float(value) for value in balanced.tolist()
                    ],
                }
        receipts = []
        receipt_payload = getattr(fit, "receipt", None)
        factorization = getattr(fit, "factorization", None)
        shared_receipt = (
            receipt_payload.to_dict() if receipt_payload is not None else None
        )
        scale_all = fit.scale.detach().to(device="cpu", dtype=torch.float32)
        offset_all = fit.offset.detach().to(device="cpu", dtype=torch.float32)
        fold_context_ids = self._fold_context_id_lists(panel, fold_count=fold_count)
        for fold in range(int(fold_count)):
            calibration_ids, extraction_ids = fold_context_ids[fold]
            payload = {
                "family": str(getattr(fit, "family", "c2_shared_diagonal_affine")),
                "backend": str(
                    getattr(
                        fit,
                        "calibration_backend",
                        getattr(fit, "backend", "unknown"),
                    )
                ),
                "fold": int(fold),
                "panel_hash": panel.panel_hash,
                "calibration_context_ids": list(calibration_ids),
                "extraction_context_ids": list(extraction_ids),
                # Target-column scalars keep warm starts and evidence
                # consumers shape-stable.
                "readout_scale": [
                    float(scale_all[min(fold, scale_all.shape[0] - 1), target_position])
                ],
                "readout_offset": [
                    float(
                        offset_all[min(fold, offset_all.shape[0] - 1), target_position]
                    )
                ],
                "batched_fold_axis": True,
            }
            if shared_receipt is not None and fold == 0:
                payload["calibration_fit_receipt"] = shared_receipt
            if factorization is not None:
                payload["calibration_dependency_hash"] = factorization.dependency_hash
                payload["calibration_factorization_hash"] = (
                    factorization.factorization_hash
                )
            if profile is not None and fold == 0:
                payload["cross_category_readout_profile"] = profile
            receipts.append(payload)
        return losses_host, tuple(receipts)

    def _crossfit_losses_sequential(
        self,
        response: ReadoutResponse,
        panel: CrossfitPanel,
        *,
        readout: Any,
        labels: Any,
        query: QuerySpec,
        class_id: int,
        fold_count: int,
        warm_start: Mapping[str, Any] | None,
    ) -> tuple[np.ndarray, tuple[Mapping[str, Any], ...]]:
        """Historical per-fold path — retained verbatim for the sealed v1
        calibration contract (2-D-only fitter, replay surface)."""
        losses = torch.empty(
            (readout.shape[0],), device=readout.device, dtype=torch.float32
        )
        panel_weights = (
            torch.as_tensor(
                panel.sampling_weights,
                device=readout.device,
                dtype=torch.float32,
            )
            if panel.sampling_weights
            else None
        )
        receipts: list[Mapping[str, Any]] = []
        # Fold support is checkable host-side from the response labels; the
        # previous device-tensor sum() forced one CUDA sync per fold per lane.
        labels_host = np.asarray(response.labels)
        positives_host = labels_host == int(class_id)
        for fold in range(int(fold_count)):
            calibration_indices = panel.calibration_indices(fold)
            extraction_indices = panel.extraction_indices(fold)
            fit_positive = int(positives_host[list(calibration_indices)].sum())
            if fit_positive in {0, len(calibration_indices)}:
                raise ValueError(
                    "each calibration fold must contain target and nontarget contexts"
                )
            fit_index = torch.as_tensor(
                calibration_indices, device=readout.device, dtype=torch.long
            )
            test_index = torch.as_tensor(
                extraction_indices, device=readout.device, dtype=torch.long
            )
            fit_labels = labels.index_select(0, fit_index)
            binary_fit = (
                (fit_labels == int(class_id)).to(dtype=torch.float32).unsqueeze(1)
            )
            prior = _fold_warm_start(warm_start, fold)
            fit_weights = (
                None
                if panel_weights is None
                else panel_weights.index_select(0, fit_index)
            )
            fit_options = dict(self.calibration_options)
            fit_options["diagnostic_context"] = (
                f"sampled_path:{panel.panel_hash}:fold={fold}"
            )
            proper_score_options = _resolved_proper_score_options(fit_options)
            # Spec construction is deterministic in these arguments and sits
            # on the per-fit hot path — memoize per (query, geometry, fold).
            contract_key = (
                query.query_id,
                query.source_hash,
                int(readout.shape[1]),
                int(class_id),
                str(fit_options.get("binary_loss_weighting", "pos_weighted")),
                float(fit_options.get("ws_min", 1e-8)),
                int(fold),
                str(fit_options["optimization_contract"]),
                tuple(sorted(proper_score_options.items())),
            )
            contracts = self._contract_memo.get(contract_key)
            if contracts is None:
                contracts = _coordinate_calibration_contracts(
                    query,
                    response_width=int(readout.shape[1]),
                    class_id=class_id,
                    weighting=str(
                        fit_options.get("binary_loss_weighting", "pos_weighted")
                    ),
                    minimum_scale=float(fit_options.get("ws_min", 1e-8)),
                    fold_ids=(f"calibration_without_fold:{fold}",),
                    optimization_contract=str(fit_options["optimization_contract"]),
                    **proper_score_options,
                )
                if len(self._contract_memo) >= 256:
                    self._contract_memo.pop(next(iter(self._contract_memo)))
                self._contract_memo[contract_key] = contracts
            readout_spec, query_spec, problem_spec = contracts
            with self.profiler.phase("candidate.calibration_fit"):
                fit = self._fit_calibration_with_runtime_context(
                    readout=readout_spec,
                    query=query_spec,
                    problem=problem_spec,
                    query_values=readout.index_select(0, fit_index)[
                        :, class_id : class_id + 1
                    ],
                    targets=binary_fit,
                    sample_weights=fit_weights,
                    initial_scale=prior.get("readout_scale"),
                    initial_offset=prior.get("readout_offset"),
                    input_center=_query_readout_center(
                        fit_options.get("readout_bias", 0.0),
                        class_id=class_id,
                        response_width=int(readout.shape[1]),
                        device=readout.device,
                    ),
                    selection_context_ids=tuple(
                        panel.context_ids[int(index)] for index in calibration_indices
                    ),
                    iters=int(
                        fit_options.get("iters", fit_options.get("num_iterations", 64))
                    ),
                    **proper_score_options,
                    require_convergence=True,
                )
            calibrated = fit.apply(
                readout.index_select(0, test_index)[:, class_id : class_id + 1]
            )[:, 0]
            targets = (labels.index_select(0, test_index) == int(class_id)).to(
                torch.float32
            )
            # The Neyman estimator is stratified after the fold is scored.
            # Positive weighting belongs to calibration, not to held-out rows.
            row_loss = torch.where(
                targets > 0.5, F.softplus(-calibrated), F.softplus(calibrated)
            )
            losses.index_copy_(0, test_index, row_loss)
            receipts.append(
                _fit_receipt(
                    fit,
                    fold=fold,
                    panel=panel,
                    calibration_indices=calibration_indices,
                    extraction_indices=extraction_indices,
                )
            )
        return (
            losses.detach().to(device="cpu", dtype=torch.float64).numpy(),
            tuple(receipts),
        )

    def fit_reference_calibration(
        self,
        masks: Mapping[str, np.ndarray],
        request: SampledComparisonRequest,
        *,
        role: str,
        panel: CrossfitPanel | None = None,
    ) -> FittedSampledReferenceCalibration:
        """Fit the state-specific online affine on one extraction-only panel.

        Candidate decisions remain cross-fitted. Historical final validation
        additionally fitted one online affine on the complete sampled panel and
        then applied it unchanged to the untouched validation split. This is
        that second, explicitly typed operation.
        """

        query = self._query_by_id.get(request.query_id)
        if query is None:
            raise KeyError(request.query_id)
        if panel is None:
            with self.profiler.phase("candidate.panel_draw"):
                resolved_panel = self._draw_panel(request)
        else:
            resolved_panel = panel
        resolved_panel.validate(fold_count=request.fold_count)
        mask_hash = sampled_path_mask_hash(masks)
        self._record_panel_delivery(
            resolved_panel,
            request=request,
            role="calibration",
            mask_hashes=(mask_hash,),
        )
        response = self._execute(
            masks,
            resolved_panel,
            query_id=request.query_id,
            mask_hash=mask_hash,
        )
        readout = getattr(response, "device_values", None)
        if readout is None:
            readout = torch.as_tensor(
                response.values,
                device=self.evaluator.device,
                dtype=torch.float32,
            )
        labels = torch.as_tensor(
            response.labels,
            device=self.evaluator.device,
            dtype=torch.long,
        )
        class_id = _query_class_id(query)
        if class_id < 0 or class_id >= int(readout.shape[1]):
            raise ValueError("query class coordinate lies outside the model readout")
        binary = (labels == int(class_id)).to(dtype=torch.float32).unsqueeze(1)
        # Host-side support check (response labels are host numpy); avoids a
        # device sync on the reference path.
        positive_count = int((np.asarray(response.labels) == int(class_id)).sum())
        if positive_count in {0, len(response.labels)}:
            raise ValueError(
                "reference calibration requires target and nontarget contexts"
            )
        sample_weights = (
            None
            if not resolved_panel.sampling_weights
            else torch.as_tensor(
                resolved_panel.sampling_weights,
                device=readout.device,
                dtype=torch.float32,
            )
        )
        options = dict(self.calibration_options)
        options["diagnostic_context"] = (
            f"sampled_path_final_validation:{resolved_panel.panel_hash}:{role}"
        )
        proper_score_options = _resolved_proper_score_options(options)
        readout_spec, query_spec, problem_spec = _coordinate_calibration_contracts(
            query,
            response_width=int(readout.shape[1]),
            class_id=class_id,
            weighting=str(options.get("binary_loss_weighting", "pos_weighted")),
            minimum_scale=float(options.get("ws_min", 1e-8)),
            fold_ids=("full_sampled_optimization_panel",),
            optimization_contract=str(options["optimization_contract"]),
            **proper_score_options,
        )
        fit = self._fit_calibration_with_runtime_context(
            readout=readout_spec,
            query=query_spec,
            problem=problem_spec,
            query_values=readout[:, class_id : class_id + 1],
            targets=binary,
            sample_weights=sample_weights,
            input_center=_query_readout_center(
                options.get("readout_bias", 0.0),
                class_id=class_id,
                response_width=int(readout.shape[1]),
                device=readout.device,
            ),
            selection_context_ids=resolved_panel.context_ids,
            iters=int(options.get("iters", options.get("num_iterations", 64))),
            **proper_score_options,
            require_convergence=True,
        )
        scale = getattr(fit, "readout_scale", getattr(fit, "scale", None))
        offset = getattr(fit, "readout_offset", getattr(fit, "offset", None))
        if scale is None or offset is None:
            raise TypeError("reference calibration fit lacks affine parameters")
        metadata = {
            "schema_name": "sampled_path_full_panel_reference_calibration_v1",
            "query_id": request.query_id,
            "role": str(role),
            "mask_hash": mask_hash,
            "panel_hash": resolved_panel.panel_hash,
            "panel_context_count": len(resolved_panel.context_ids),
            "panel_context_ids_hash": hashlib.sha256(
                "\x1f".join(resolved_panel.context_ids).encode("utf-8")
            ).hexdigest(),
            "class_id": int(class_id),
            "positive_count": positive_count,
            "negative_count": int(binary.numel()) - positive_count,
            "family": str(getattr(fit, "family", "c2_shared_diagonal_affine")),
            "backend": str(
                getattr(
                    fit,
                    "calibration_backend",
                    getattr(fit, "backend", "unknown"),
                )
            ),
            "readout_projection_hash": readout_spec.content_hash,
            "query_operator_hash": query_spec.content_hash,
            "calibration_problem_hash": problem_spec.content_hash,
            "calibration_dependency_hash": fit.factorization.dependency_hash,
            "calibration_factorization_hash": fit.factorization.factorization_hash,
            "calibration_fit_receipt": fit.receipt.to_dict(),
            "readout_scale": np.asarray(
                scale.detach().to(device="cpu", dtype=torch.float32).numpy()
            )
            .reshape(-1)
            .tolist(),
            "readout_offset": np.asarray(
                offset.detach().to(device="cpu", dtype=torch.float32).numpy()
            )
            .reshape(-1)
            .tolist(),
            "calibration_source": "full_sampled_optimization_panel",
            "validation_refit": False,
        }
        result = FittedSampledReferenceCalibration(
            query_id=request.query_id,
            role=str(role),
            mask_hash=mask_hash,
            panel=resolved_panel,
            fit=fit,
            metadata=metadata,
        )
        result.validate(fold_count=request.fold_count)
        return result

    def fit_glocal_brier_calibration(
        self,
        masks: Mapping[str, np.ndarray],
        request: SampledComparisonRequest,
        *,
        role: str,
        panel: CrossfitPanel,
        warm_start: FrozenBrierCalibration | None = None,
        maximum_iterations: int = 256,
        gradient_tolerance: float = 1e-9,
        identity_ridge: float = 1e-8,
    ) -> FrozenBrierCalibration:
        """Fit claim-mode Brier calibration and retain a runtime receipt."""

        with self._glocal_runtime_call(
            operation="fit_glocal_brier_calibration",
            request=request,
            role=role,
            panel=panel,
        ):
            return self._fit_glocal_brier_calibration_impl(
                masks,
                request,
                role=role,
                panel=panel,
                warm_start=warm_start,
                maximum_iterations=maximum_iterations,
                gradient_tolerance=gradient_tolerance,
                identity_ridge=identity_ridge,
            )

    def evaluate_glocal_calibration_logits(
        self,
        masks: Mapping[str, np.ndarray],
        request: SampledComparisonRequest,
        *,
        panel: CrossfitPanel,
    ) -> GlocalCalibrationLogitRows:
        """Execute one sealed calibration prefix without changing the fit."""

        with self._glocal_runtime_call(
            operation="evaluate_glocal_calibration_logits",
            request=request,
            role="calibration_fit",
            panel=panel,
        ):
            query = self._query_by_id.get(request.query_id)
            if query is None:
                raise KeyError(request.query_id)
            panel.validate(fold_count=request.fold_count)
            class_id = _query_class_id(query)
            self._validate_explicit_glocal_panel_request(
                panel,
                request,
                target_class_id=class_id,
            )
            mask_hash = sampled_path_mask_hash(masks)
            self._record_panel_delivery(
                panel,
                request=request,
                role="calibration_fit",
                mask_hashes=(mask_hash,),
            )
            response = self._execute(
                masks,
                panel,
                query_id=request.query_id,
                mask_hash=mask_hash,
            )
            values = np.asarray(response.values, dtype=np.float64)
            if values.ndim != 2 or class_id < 0 or class_id >= values.shape[1]:
                raise ValueError(
                    "glocal calibration response lacks the query coordinate"
                )
            rows = GlocalCalibrationLogitRows(
                query_id=request.query_id,
                role="calibration_fit",
                mask_hash=mask_hash,
                panel=panel,
                logits=values[:, class_id],
                binary_labels=(np.asarray(response.labels) == class_id).astype(
                    np.uint8
                ),
                positive_class_id=class_id,
            )
            return rows

    def _fit_glocal_brier_calibration_impl(
        self,
        masks: Mapping[str, np.ndarray],
        request: SampledComparisonRequest,
        *,
        role: str,
        panel: CrossfitPanel,
        warm_start: FrozenBrierCalibration | None = None,
        maximum_iterations: int = 256,
        gradient_tolerance: float = 1e-9,
        identity_ridge: float = 1e-8,
    ) -> FrozenBrierCalibration:
        """Scientific implementation behind the runtime-only call envelope."""

        query = self._query_by_id.get(request.query_id)
        if query is None:
            raise KeyError(request.query_id)
        panel.validate(fold_count=request.fold_count)
        class_id = _query_class_id(query)
        self._validate_explicit_glocal_panel_request(
            panel,
            request,
            target_class_id=class_id,
        )
        mask_hash = sampled_path_mask_hash(masks)
        self._record_panel_delivery(
            panel,
            request=request,
            role="calibration",
            mask_hashes=(mask_hash,),
        )
        response = self._execute(
            masks,
            panel,
            query_id=request.query_id,
            mask_hash=mask_hash,
        )
        values = np.asarray(response.values, dtype=np.float64)
        if values.ndim != 2 or class_id < 0 or class_id >= values.shape[1]:
            raise ValueError("glocal calibration response lacks the query coordinate")
        labels = (np.asarray(response.labels) == class_id).astype(np.uint8)
        self._ensure_resource_accounting()
        with self._resource_lock:
            self._resource_calibration_fit_calls += 1
        return fit_class_balanced_brier_affine(
            values[:, class_id],
            labels,
            query_id=request.query_id,
            role=role,
            mask_hash=mask_hash,
            calibration_context_ids=panel.context_ids,
            positive_class_id=class_id,
            maximum_iterations=maximum_iterations,
            gradient_tolerance=gradient_tolerance,
            identity_ridge=identity_ridge,
            warm_start=warm_start,
        )

    def evaluate_glocal_brier_pair(
        self,
        parent_masks: Mapping[str, np.ndarray],
        candidate_masks: Mapping[str, np.ndarray],
        request: SampledComparisonRequest,
        *,
        panel: CrossfitPanel,
        parent_calibration: FrozenBrierCalibration,
        candidate_calibration: FrozenBrierCalibration,
    ) -> PairedBrierEvidence:
        """Evaluate a frozen state pair and retain a runtime receipt."""

        with self._glocal_runtime_call(
            operation="evaluate_glocal_brier_pair",
            request=request,
            role="proposal_adjudication",
            panel=panel,
        ):
            return self._evaluate_glocal_brier_pair_impl(
                parent_masks,
                candidate_masks,
                request,
                panel=panel,
                parent_calibration=parent_calibration,
                candidate_calibration=candidate_calibration,
                delivery_role="decision",
            )

    def evaluate_glocal_terminal_pair(
        self,
        parent_masks: Mapping[str, np.ndarray],
        candidate_masks: Mapping[str, np.ndarray],
        request: SampledComparisonRequest,
        *,
        panel: CrossfitPanel,
        parent_calibration: FrozenBrierCalibration,
        candidate_calibration: FrozenBrierCalibration,
    ) -> PairedBrierEvidence:
        """Evaluate one untouched terminal panel with frozen state fits."""

        with self._glocal_runtime_call(
            operation="evaluate_glocal_terminal_pair",
            request=request,
            role="terminal_audit",
            panel=panel,
        ):
            return self._evaluate_glocal_brier_pair_impl(
                parent_masks,
                candidate_masks,
                request,
                panel=panel,
                parent_calibration=parent_calibration,
                candidate_calibration=candidate_calibration,
                delivery_role="terminal",
            )

    def _evaluate_glocal_brier_pair_impl(
        self,
        parent_masks: Mapping[str, np.ndarray],
        candidate_masks: Mapping[str, np.ndarray],
        request: SampledComparisonRequest,
        *,
        panel: CrossfitPanel,
        parent_calibration: FrozenBrierCalibration,
        candidate_calibration: FrozenBrierCalibration,
        delivery_role: str,
    ) -> PairedBrierEvidence:
        """Scientific implementation behind the runtime-only call envelope."""

        if delivery_role not in {"decision", "terminal"}:
            raise ValueError("glocal pair delivery role is invalid")
        query = self._query_by_id.get(request.query_id)
        if query is None:
            raise KeyError(request.query_id)
        panel.validate(fold_count=request.fold_count)
        parent_calibration.validate()
        candidate_calibration.validate()
        parent_hash = sampled_path_mask_hash(parent_masks)
        candidate_hash = sampled_path_mask_hash(candidate_masks)
        if (
            parent_calibration.mask_hash != parent_hash
            or candidate_calibration.mask_hash != candidate_hash
            or parent_calibration.query_id != request.query_id
            or candidate_calibration.query_id != request.query_id
        ):
            raise ValueError(
                "glocal calibration does not bind the requested state pair"
            )
        class_id = _query_class_id(query)
        self._validate_explicit_glocal_panel_request(
            panel,
            request,
            target_class_id=class_id,
        )
        adjudication_contexts = set(panel.context_ids)
        if adjudication_contexts.intersection(
            parent_calibration.calibration_context_ids
        ) or adjudication_contexts.intersection(
            candidate_calibration.calibration_context_ids
        ):
            raise ValueError(
                "calibration-fit rows must be disjoint from glocal evaluation rows"
            )
        self._record_panel_delivery(
            panel,
            request=request,
            role=delivery_role,
            mask_hashes=(parent_hash, candidate_hash),
        )
        parent, candidate = self._execute_lanes(
            (
                (parent_hash, parent_masks),
                (candidate_hash, candidate_masks),
            ),
            panel,
            query_id=request.query_id,
        )
        if parent.context_ids != candidate.context_ids or not np.array_equal(
            parent.labels, candidate.labels
        ):
            raise ValueError("glocal state pair did not use common random numbers")

        def probabilities(
            response: ReadoutResponse,
            calibration: FrozenBrierCalibration,
        ) -> np.ndarray:
            values = np.asarray(response.values, dtype=np.float64)
            if values.ndim != 2 or class_id < 0 or class_id >= values.shape[1]:
                raise ValueError(
                    "glocal adjudication response lacks the query coordinate"
                )
            natural = (
                calibration.scale * (values[:, class_id] - calibration.input_center)
                + calibration.offset
            )
            return _numpy_sigmoid(natural)

        binary_labels = (np.asarray(parent.labels) == class_id).astype(np.uint8)
        evidence = PairedBrierEvidence(
            panel=panel,
            parent_probabilities=probabilities(parent, parent_calibration),
            candidate_probabilities=probabilities(candidate, candidate_calibration),
            binary_labels=binary_labels,
            parent_calibration_output_hash=parent_calibration.output_hash,
            candidate_calibration_output_hash=candidate_calibration.output_hash,
            parent_mask_hash=parent_hash,
            candidate_mask_hash=candidate_hash,
            common_random_numbers=True,
        )
        evidence.validate(fold_count=request.fold_count)
        return evidence

    def calibrated_target_logits(
        self,
        response: ReadoutResponse,
        calibration: FittedSampledReferenceCalibration,
    ) -> np.ndarray:
        """Apply a frozen extraction-panel affine to an ordered response."""

        query = self._query_by_id.get(calibration.query_id)
        if query is None:
            raise KeyError(calibration.query_id)
        response.validate()
        readout = getattr(response, "device_values", None)
        if readout is None:
            readout = torch.as_tensor(
                response.values,
                device=self.evaluator.device,
                dtype=torch.float32,
            )
        class_id = _query_class_id(query)
        calibrated = calibration.fit.apply(readout[:, class_id : class_id + 1])
        return calibrated[:, 0].detach().to(device="cpu", dtype=torch.float64).numpy()

    def evaluate_pair(
        self,
        committed_masks: Mapping[str, np.ndarray],
        candidate_masks: Mapping[str, np.ndarray],
        request: SampledComparisonRequest,
    ) -> PairedCandidateEvidence:
        query = self._query_by_id.get(request.query_id)
        if query is None:
            raise KeyError(request.query_id)
        with self.profiler.phase("candidate.panel_draw"):
            panel = self._draw_panel(request)
        # Mask identities are per-proposal invariants: hash each side once and
        # thread the value through execution, crossfit caching, and evidence.
        committed_hash = sampled_path_mask_hash(committed_masks)
        candidate_hash = sampled_path_mask_hash(candidate_masks)
        self._record_panel_delivery(
            panel,
            request=request,
            role="decision",
            mask_hashes=(committed_hash, candidate_hash),
        )
        committed, candidate = self._execute_lanes(
            (
                (committed_hash, committed_masks),
                (candidate_hash, candidate_masks),
            ),
            panel,
            query_id=request.query_id,
        )
        if committed.context_ids != candidate.context_ids or not np.array_equal(
            committed.labels, candidate.labels
        ):
            raise ValueError("paired masks were not evaluated on common random numbers")
        class_id = _query_class_id(query)
        committed_loss, committed_fits = self._crossfit_losses_cached(
            committed,
            panel,
            mask_hash=committed_hash,
            query=query,
            class_id=class_id,
            fold_count=request.fold_count,
            warm_start=request.committed_warm_start,
        )
        candidate_loss, candidate_fits = self._crossfit_losses_cached(
            candidate,
            panel,
            mask_hash=candidate_hash,
            query=query,
            class_id=class_id,
            fold_count=request.fold_count,
            warm_start=request.candidate_warm_start,
        )
        evidence = PairedCandidateEvidence(
            panel=panel,
            committed_loss=committed_loss,
            candidate_loss=candidate_loss,
            committed_calibration_by_fold=committed_fits,
            candidate_calibration_by_fold=candidate_fits,
            committed_mask_hash=committed_hash,
            candidate_mask_hash=candidate_hash,
            common_random_numbers=True,
        )
        evidence.validate(fold_count=request.fold_count)
        return evidence

    def evaluate_dense_reference(
        self,
        masks: Mapping[str, np.ndarray],
        request: SampledComparisonRequest,
    ) -> SampledReferenceEvidence:
        """Evaluate the native v1 committed reference without a dummy candidate.

        This is the canonical equivalent of the production
        ``evaluate_reference`` call at a layer-group boundary: draw the first
        extraction panel, fit each calibration fold off-fold, and retain every
        held-out row loss. It deliberately does not inspect final validation.
        """

        query = self._query_by_id.get(request.query_id)
        if query is None:
            raise KeyError(request.query_id)
        with self.profiler.phase("candidate.panel_draw"):
            panel = self._draw_panel(request)
        mask_hash = sampled_path_mask_hash(masks)
        self._record_panel_delivery(
            panel,
            request=request,
            role="reference",
            mask_hashes=(mask_hash,),
        )
        response = self._execute(
            masks,
            panel,
            query_id=request.query_id,
            mask_hash=mask_hash,
        )
        losses, fits = self._crossfit_losses(
            response,
            panel,
            query=query,
            class_id=_query_class_id(query),
            fold_count=request.fold_count,
            warm_start=request.committed_warm_start,
        )
        evidence = SampledReferenceEvidence(
            query_id=request.query_id,
            panel=panel,
            losses=losses,
            calibration_by_fold=fits,
            mask_hash=mask_hash,
        )
        evidence.validate(fold_count=request.fold_count)
        return evidence


__all__ = [
    "ContextAxisSampledPanelSource",
    "coordinate_calibration_contracts",
    "FittedSampledReferenceCalibration",
    "GLOCAL_RUNTIME_CALL_RECEIPT_SCHEMA",
    "GLOCAL_RUNTIME_RESOURCE_SCHEMA",
    "GlocalCalibrationLogitRows",
    "GlocalRuntimeCallReceipt",
    "GlocalRuntimeSnapshot",
    "SampledReferenceEvidence",
    "SampledPanelSource",
    "TorchSampledPathEvaluationBackend",
    "draw_nested_class_balanced_panel",
    "fit_class_balanced_brier_affine",
    "fold_warm_start",
]
