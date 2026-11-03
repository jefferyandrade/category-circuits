"""Convex output-coordinate calibration for factorized query problems."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from contracts.hashing import stable_digest
from mechanistic_systems.measurement.run_products import loss_weighting

from mechanistic_systems.backends.calibration_problem import (
    CalibrationFactorization,
)
from mechanistic_systems.backends.contracts import BackendProfile
from mechanistic_systems.backends.calibration_statistics import (
    calibration_statistics,
)

try:  # Optional for import smoke; required by executable calibration.
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - environment dependent.
    torch = F = None  # type: ignore[assignment]


# Runtime-only execution knobs. Never part of scientific identity;
# recorded in the backend profile and named as future RuntimeExecutionProfile
# fields.
#
# NEWTON_ACTIVE_POLL_EVERY: host polls of device-side "any cell still
# active?" run every N solver iterations instead of every iteration. Skipped
# polls only ever add masked no-op iterations (frozen cells' updates are
# `where`-masked), so any cadence is bit-identical to per-iteration polling.
#
# LINE_SEARCH_ALPHA_CHUNK: the Armijo backtracking line search evaluates this
# many 2^-k alpha candidates per batched statistics call (first-accept
# semantics preserved; chunk 1 IS the sequential search). One host
# sync per chunk boundary replaces one per candidate step.
NEWTON_ACTIVE_POLL_EVERY = 4
LINE_SEARCH_ALPHA_CHUNK = 8


# The production fit uses device float64 projected Newton statistics.
TERMINAL_FIT_EXECUTION_V6 = "v6"
GPU_HARD_GATES_AUDIT_POLICY = "device_float64_convergence_v63"
V6_NONFINITE_DIAGNOSTIC_SENTINEL = float.fromhex("0x1.fffffffffffffp+1023")
V6_MISSING_CELL_RULE = "missing_unconverged_cell_v63"

_GPU_FALLBACK_ITERATION_FACTOR = 4
_GPU_FALLBACK_LINE_SEARCH_LIMIT = 60
_V6_GATE_NAMES = ("none", "nonfinite", "objective_worsened", "line_search_failed", "iteration_limit")


class CalibrationConvergenceError(RuntimeError):
    def __init__(self, message: str, receipt: "CalibrationFitReceipt") -> None:
        super().__init__(message)
        self.receipt = receipt


class CalibrationFailClosedError(CalibrationConvergenceError):
    """Cells failed their device hard gates after the fallback.

    Per-cell failures do not abort a fit: the
    proper-score core marks the affected (mask, query) column MISSING and
    records it (``proper_score_shrinkage._v6_gate_record``).  A consumer that
    can carry a missing column requests ``missing_column_policy="carry"``
    (the terminal bank provider, which applies the registered bank-level
    threshold of ``calibration_backend``).  Every other consumer keeps the
    default ``"fail_closed"``: after the WHOLE fit this typed error names
    every missing column, so no NaN affine reaches a reader that cannot
    carry one.  The single-column differential-owner emission always raises
    it.  ``failed_query_axis_ids`` names the missing query columns, in query
    order; ``gate_summary`` (when set) is the fit's full v6 gate record.
    """

    def __init__(
        self,
        message: str,
        receipt: "CalibrationFitReceipt",
        *,
        failed_query_axis_ids: Sequence[str] = (),
    ) -> None:
        super().__init__(message, receipt)
        self.failed_query_axis_ids = tuple(str(value) for value in failed_query_axis_ids)


@dataclass(frozen=True)
class CalibrationFitReceipt:
    dependency_hash: str
    factorization_hash: str
    component_hashes: tuple[str, ...]
    backend: str
    initial_objective: tuple[float, ...]
    final_objective: tuple[float, ...]
    identity_objective: tuple[float, ...]
    intercept_only_objective: tuple[float, ...]
    iterations: tuple[int, ...]
    line_search_steps: tuple[int, ...]
    gradient_kkt_residual: tuple[float, ...]
    termination_reason: tuple[str, ...]
    finite: tuple[bool, ...]
    converged: tuple[bool, ...]
    fallback_reason: str = ""
    backend_profile: Mapping[str, Any] = field(default_factory=dict)
    # Convergence gating for multi-column fits: required cells (the decision
    # column plus explicitly requested columns) stay fail-closed; enrichment
    # cells are reported, never silently repaired. None = no declared
    # required set (single-column callers; the full-strict `passed` governs).
    required_passed: bool | None = None
    def validate(self) -> None:
        count = len(self.final_objective)
        aligned = (
            self.initial_objective,
            self.identity_objective,
            self.intercept_only_objective,
            self.iterations,
            self.line_search_steps,
            self.gradient_kkt_residual,
            self.termination_reason,
            self.finite,
            self.converged,
        )
        if (
            count == 0
            or len(self.component_hashes) != count
            or any(len(values) != count for values in aligned)
        ):
            raise ValueError("calibration receipt fit axes are empty or misaligned")
        if len(self.dependency_hash) != 64 or len(self.factorization_hash) != 64:
            raise ValueError("calibration receipt hashes are invalid")
        numeric = (
            *self.initial_objective,
            *self.final_objective,
            *self.identity_objective,
            *self.intercept_only_objective,
            *self.gradient_kkt_residual,
        )
        if any(not math.isfinite(float(value)) for value in numeric):
            raise ValueError("calibration receipt contains non-finite diagnostics")

    @property
    def passed(self) -> bool:
        self.validate()
        return all(self.finite) and all(self.converged)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_name": "mechanistic_calibration_fit_receipt",
            **({"loss_weighting": "per_image"} if self.backend_profile.get("metadata", {}).get("loss_weighting") == "per_image" else {}),
            "schema_version": "1",
            "dependency_hash": self.dependency_hash,
            "factorization_hash": self.factorization_hash,
            "component_hashes": list(self.component_hashes),
            "backend": self.backend,
            "initial_objective": list(self.initial_objective),
            "final_objective": list(self.final_objective),
            "identity_objective": list(self.identity_objective),
            "intercept_only_objective": list(self.intercept_only_objective),
            "iterations": list(self.iterations),
            "line_search_steps": list(self.line_search_steps),
            "gradient_kkt_residual": list(self.gradient_kkt_residual),
            "termination_reason": list(self.termination_reason),
            "finite": list(self.finite),
            "converged": list(self.converged),
            "fallback_reason": self.fallback_reason,
            "backend_profile": dict(self.backend_profile),
            "passed": self.passed,
            "required_passed": (
                None if self.required_passed is None else bool(self.required_passed)
            ),
            **({"calibration_penalty": "firth",
                "effective_sample_size": list(self.backend_profile["metadata"]["effective_sample_size"])}
               if self.backend_profile.get("metadata", {}).get("calibration_penalty") == "firth"
               else {}),
        }


@dataclass(frozen=True)
class FactoredCalibrationFit:
    scale: Any
    offset: Any
    factorization: CalibrationFactorization
    receipt: CalibrationFitReceipt
    input_was_batched: bool
    family: str = "overlap_factored_post_query_affine"

    @property
    def backend(self) -> str:
        return self.receipt.backend

    @property
    def calibration_backend(self) -> str:
        return self.receipt.backend

    def apply(self, values: Any) -> Any:
        tensor = torch.as_tensor(values)
        tensor = tensor.to(dtype=torch.promote_types(tensor.dtype, self.scale.dtype))
        if tensor.ndim == 2:
            return tensor * self.scale[0].to(tensor).unsqueeze(0) + self.offset[0].to(tensor).unsqueeze(0)
        if tensor.ndim == 3:
            return tensor * self.scale.to(tensor).unsqueeze(1) + self.offset.to(tensor).unsqueeze(1)
        raise ValueError("calibration values must have shape [sample,query] or [mask,sample,query]")


class CalibrationDifferentialOwnerError(ValueError):
    """The fitted calibrator lies on an unresolved differential face."""

    def __init__(self, code: str, message: str) -> None:
        self.code = str(code); super().__init__(f"{self.code}: {message}")


SCALAR_V2R_REVISION4_SPEC_SHA256 = (
    "e958c4ad5a2cddd4fc50ff60e59cb87f501ab76fef720d86330320a50d5f1aa1"
)


@dataclass(frozen=True)
class PositiveAffineComponentFace:
    mask_index: int; query_index: int
    raw_scale: float; raw_offset: float
    conditioned_scale: float; conditioned_offset: float
    weighted_location: float; raw_spread: float; effective_spread: float; minimum_raw_scale: float
    minimum_conditioned_scale: float; anchor_strength: float; weight_sum: float
    identifiability_tau: float; identifiability_face_tau: float
    identifiability_class: str; positive_scale_class: str
    free_raw_coordinates: tuple[int, ...]; raw_gradient: tuple[float, float]
    raw_multiplier: float; primal_distance: float; stationarity_residual: float; complementarity_residual: float
    tolerances: tuple[float, float, float, float]
    raw_hessian: tuple[tuple[float, float], tuple[float, float]]
    clamp_branch_hash: str

    def validate(self) -> None:
        names = ("raw_scale", "raw_offset", "conditioned_scale", "conditioned_offset", "weighted_location", "raw_spread", "effective_spread", "minimum_raw_scale", "minimum_conditioned_scale", "anchor_strength", "weight_sum", "identifiability_tau", "identifiability_face_tau", "raw_multiplier", "primal_distance", "stationarity_residual", "complementarity_residual")
        numbers = tuple(float(getattr(self, name)) for name in names) + self.raw_gradient + self.tolerances + tuple(value for row in self.raw_hessian for value in row)
        if not all(math.isfinite(value) for value in numbers): raise ValueError("positive-affine face contains nonfinite state")
        tau_primal, tau_dual, tau_stationarity, tau_complementarity = self.tolerances
        if min(self.effective_spread, self.weight_sum, self.identifiability_tau, self.identifiability_face_tau, *self.tolerances) <= 0.0 or not self.identifiability_face_tau < self.identifiability_tau: raise ValueError("positive-affine face tolerances are invalid")
        identified = self.raw_spread > self.identifiability_tau + self.identifiability_face_tau
        quotient = self.raw_spread < self.identifiability_tau - self.identifiability_face_tau
        if self.identifiability_class == "identified" and not identified: raise ValueError("identified affine face violates its strict spread margin")
        if self.identifiability_class == "quotient_fixed_scale" and not quotient: raise ValueError("quotient affine face violates its strict spread margin")
        if self.identifiability_class not in {"identified", "quotient_fixed_scale"}: raise ValueError("positive-affine identifiability class is invalid")
        primal = self.raw_scale - self.minimum_raw_scale
        ga, gc = self.raw_gradient
        if self.positive_scale_class == "interior":
            valid = identified and primal > tau_primal and self.free_raw_coordinates == (0, 1) and abs(self.raw_multiplier) <= tau_dual
            stationarity = max(abs(ga), abs(gc))
        elif self.positive_scale_class == "certified_active":
            valid = identified and abs(primal) <= tau_primal and self.raw_multiplier > tau_dual and self.free_raw_coordinates == (1,)
            stationarity = max(abs(ga - self.raw_multiplier), abs(gc))
        elif self.positive_scale_class == "quotient_fixed_scale":
            valid = quotient and self.free_raw_coordinates == (1,) and abs(self.raw_multiplier) <= tau_dual
            stationarity = abs(gc)
        else:
            valid = False; stationarity = math.inf
        if not valid or self.raw_scale < self.minimum_raw_scale - tau_primal: raise ValueError("positive-affine scale face is invalid")
        closure_tau = 64.0 * math.ulp(max(1.0, abs(primal), abs(stationarity)))
        if not math.isclose(self.primal_distance, primal, rel_tol=1e-12, abs_tol=closure_tau) or not math.isclose(self.stationarity_residual, stationarity, rel_tol=1e-12, abs_tol=closure_tau) or not math.isclose(self.complementarity_residual, abs(self.raw_multiplier * primal), rel_tol=1e-12, abs_tol=closure_tau): raise ValueError("positive-affine KKT fields are not closed over their raw state")
        chart_valid = math.isclose(self.conditioned_scale, self.raw_scale * self.effective_spread, rel_tol=1e-12, abs_tol=1e-12) and math.isclose(self.conditioned_offset, self.raw_offset + self.raw_scale * self.weighted_location, rel_tol=1e-12, abs_tol=1e-12) and math.isclose(self.minimum_conditioned_scale, self.minimum_raw_scale * self.effective_spread, rel_tol=1e-12, abs_tol=1e-12)
        if not chart_valid: raise ValueError("positive-affine conditioned/raw chart identity is invalid")
        if self.stationarity_residual > tau_stationarity or self.complementarity_residual > tau_complementarity: raise ValueError("positive-affine KKT face exceeds its certified tolerance")
        if len(self.clamp_branch_hash) != 64 or any(character not in "0123456789abcdef" for character in self.clamp_branch_hash) or self.raw_hessian[0][1] != self.raw_hessian[1][0] or min(self.raw_hessian[0][0], self.raw_hessian[1][1]) < 0.0 or self.raw_hessian[0][0] * self.raw_hessian[1][1] - self.raw_hessian[0][1] ** 2 < -1e-12 * (1.0 + abs(self.raw_hessian[0][0] * self.raw_hessian[1][1])): raise ValueError("positive-affine curvature identity is invalid")

    def to_dict(self) -> dict[str, Any]:
        self.validate(); return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class PositiveAffineDifferentialOwnerReceipt:
    revision_identity: str; solver_owner_identity: str
    factorization_hash: str; fit_receipt_hash: str
    logits_hash: str; targets_hash: str; row_weights_hash: str
    zero_weight_rows_hash: str; optimization_state_hash: str; context_ids_hash: str
    optimization_backend: str
    source_logit_dtype: str; source_logit_epsilon: float
    anchor_probability_epsilon: float; objective_reduction: str
    component_faces: tuple[PositiveAffineComponentFace, ...]

    def validate(self) -> None:
        hash_names = ("revision_identity", "solver_owner_identity", "factorization_hash", "fit_receipt_hash", "logits_hash", "targets_hash", "row_weights_hash", "zero_weight_rows_hash", "optimization_state_hash", "context_ids_hash")
        if any(len(getattr(self, name)) != 64 or any(character not in "0123456789abcdef" for character in getattr(self, name)) for name in hash_names) or not self.component_faces: raise ValueError("positive-affine differential owner identity is invalid")
        if self.revision_identity != SCALAR_V2R_REVISION4_SPEC_SHA256 or self.objective_reduction != "weighted_mean_over_sample_axis_v1": raise ValueError("positive-affine objective reduction is invalid")
        if not all(math.isfinite(value) and value > 0.0 for value in (self.source_logit_epsilon, self.anchor_probability_epsilon)): raise ValueError("positive-affine owner tolerances are invalid")
        if not self.optimization_backend: raise ValueError("positive-affine backend lineage is incomplete")
        coordinates = tuple((face.mask_index, face.query_index) for face in self.component_faces)
        if any(left < 0 or right < 0 for left, right in coordinates) or len(set(coordinates)) != len(coordinates): raise ValueError("positive-affine component axis is invalid")
        expected = tuple((left, right) for left in range(max(value[0] for value in coordinates) + 1) for right in range(max(value[1] for value in coordinates) + 1))
        if coordinates != expected: raise ValueError("positive-affine component axis is incomplete or noncanonical")
        for face in self.component_faces: face.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = {name: getattr(self, name) for name in self.__dataclass_fields__ if name != "component_faces"}
        return {"schema_name": "mechanistic_positive_affine_differential_owner_receipt_v1", **payload, "component_faces": [value.to_dict() for value in self.component_faces]}

    @property
    def content_hash(self) -> str: return stable_digest(self.to_dict())


@dataclass(frozen=True)
class PositiveAffineOwnedFit:
    fit: FactoredCalibrationFit; differential_owner: PositiveAffineDifferentialOwnerReceipt


def _multiclass_target_indices(
    targets: Any,
    *,
    sample_count: int,
    class_count: int,
    device: Any,
) -> Any:
    values = torch.as_tensor(targets, device=device)
    if values.ndim == 1:
        indices = values.to(torch.int64)
    elif values.ndim == 2 and tuple(values.shape) == (sample_count, class_count):
        finite = torch.isfinite(values).all()
        nonnegative = (values >= 0).all()
        normalized = torch.isclose(
            values.to(torch.float64).sum(dim=1),
            torch.ones(sample_count, device=device, dtype=torch.float64),
            atol=1e-10,
            rtol=1e-10,
        ).all()
        if not bool((finite & nonnegative & normalized).item()):
            raise ValueError("multiclass calibration targets must be one-hot rows")
        indices = values.argmax(dim=1).to(torch.int64)
    else:
        raise ValueError(
            "multiclass calibration targets must be class indices or one-hot rows"
        )
    if tuple(indices.shape) != (sample_count,):
        raise ValueError("multiclass calibration targets differ from the sample axis")
    if bool(((indices < 0) | (indices >= class_count)).any().item()):
        raise ValueError("multiclass calibration target index is outside the class axis")
    return indices


def _shared_temperature_weights(
    sample_weights: Any | None,
    *,
    weighting: str,
    mask_count: int,
    sample_count: int,
    device: Any,
) -> Any:
    if weighting == "pos_weighted":
        raise ValueError(
            "positive-class weighting is undefined for shared-temperature "
            "multiclass calibration"
        )
    if weighting == "design_weighted" and sample_weights is None:
        raise ValueError(
            "design-weighted shared-temperature calibration requires sample weights"
        )
    if weighting not in {"unweighted", "design_weighted"}:
        raise ValueError(f"unsupported multiclass calibration weighting {weighting!r}")
    if sample_weights is None:
        return torch.ones(
            (mask_count, sample_count), device=device, dtype=torch.float64
        )
    weights = torch.as_tensor(sample_weights, device=device, dtype=torch.float64)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1).expand(mask_count, -1)
    elif tuple(weights.shape) == (sample_count, 1):
        weights = weights.reshape(1, -1).expand(mask_count, -1)
    elif tuple(weights.shape) != (mask_count, sample_count):
        raise ValueError(
            "multiclass calibration weights differ from mask/sample axes"
        )
    if not bool(torch.isfinite(weights).all().item()) or bool((weights < 0).any().item()):
        raise ValueError("multiclass calibration weights must be finite and nonnegative")
    if bool((weights.sum(dim=1) <= 0).any().item()):
        raise ValueError("every multiclass calibration mask requires positive weight")
    return weights


def _shared_temperature_initial_scale(
    value: Any | None,
    *,
    mask_count: int,
    class_count: int,
    minimum_scale: float,
    device: Any,
) -> Any:
    if value is None:
        return torch.ones(mask_count, device=device, dtype=torch.float64)
    scale = torch.as_tensor(value, device=device, dtype=torch.float64)
    if int(scale.numel()) == 1:
        scale = scale.reshape(1).expand(mask_count)
    elif tuple(scale.shape) == (mask_count,):
        pass
    elif tuple(scale.shape) == (mask_count, 1):
        scale = scale[:, 0]
    elif tuple(scale.shape) == (class_count,):
        if not bool(
            torch.isclose(
                scale,
                scale[:1],
                atol=0.0,
                rtol=0.0,
            ).all().item()
        ):
            raise ValueError(
                "shared-temperature initialization must be identical across classes"
            )
        scale = scale[:1].expand(mask_count)
    elif tuple(scale.shape) == (mask_count, class_count):
        if not bool(
            torch.isclose(
                scale,
                scale[:, :1],
                atol=0.0,
                rtol=0.0,
            ).all().item()
        ):
            raise ValueError(
                "shared-temperature initialization must be identical across classes"
            )
        scale = scale[:, 0]
    else:
        raise ValueError("shared-temperature initialization differs from the mask axis")
    if not bool(torch.isfinite(scale).all().item()):
        raise ValueError("shared-temperature initialization must be finite")
    return scale.clamp_min(float(minimum_scale))


def _shared_temperature_statistics(
    logits: Any,
    target_indices: Any,
    weights: Any,
    scale: Any,
) -> tuple[Any, Any, Any]:
    scaled = logits * scale.reshape(-1, 1, 1)
    log_partition = torch.logsumexp(scaled, dim=2)
    target_logits = logits.gather(
        2,
        target_indices.reshape(1, -1, 1).expand(logits.shape[0], -1, 1),
    ).squeeze(2)
    probabilities = torch.softmax(scaled, dim=2)
    mean_logit = (probabilities * logits).sum(dim=2)
    variance = (
        probabilities * (logits - mean_logit.unsqueeze(2)).square()
    ).sum(dim=2)
    denominator = weights.sum(dim=1)
    objective = (
        weights * (log_partition - scale.reshape(-1, 1) * target_logits)
    ).sum(dim=1) / denominator
    gradient = (weights * (mean_logit - target_logits)).sum(dim=1) / denominator
    hessian = (weights * variance).sum(dim=1) / denominator
    return objective, gradient, hessian




def fit_shared_temperature_multiclass(
    values: Any,
    targets: Any,
    *,
    factorization: CalibrationFactorization,
    weighting: str = "unweighted",
    sample_weights: Any | None = None,
    minimum_scale: float = 1e-8,
    initial_scale: Any | None = None,
    max_iterations: int = 80,
    tolerance: float = 1e-9,
    hessian_ridge: float = 1e-12,
    objective_tolerance: float = 1e-10,
    reference_tolerance: float = 1e-8,
    maximum_scale: float = 1e6,
    require_convergence: bool = True,
    row_validity: Any | None = None,
) -> FactoredCalibrationFit:
    """Fit one positive inverse temperature per mask for multiclass logits.

    The multiclass loss is convex in the shared inverse temperature. A shared
    additive offset is deliberately absent because softmax makes it
    unidentifiable. Candidate masks are batched computationally while the
    factor graph retains one scientific component per mask.
    """

    _require_torch()
    wall_start = time.perf_counter()
    if len(factorization.components) != 1:
        raise ValueError("shared-temperature calibration requires one joint component")
    if factorization.components[0].parameter_axis_ids != ("shared:temperature",):
        raise ValueError("shared-temperature factorization has an unexpected parameter axis")
    x, input_was_batched = _as_batched_values(values)
    if not x.is_floating_point():
        x = x.to(torch.float32)
    membership = None
    if row_validity is not None:
        membership = torch.as_tensor(
            row_validity, device=x.device, dtype=torch.float64
        )
        if membership.ndim != 2 or int(membership.shape[1]) != int(x.shape[1]):
            raise ValueError("shared-temperature row validity must be [fold, sample]")
        if not bool(((membership == 0.0) | (membership == 1.0)).all().item()):
            raise ValueError("shared-temperature row validity must be binary")
        if int(x.shape[0]) == 1:
            x = x.expand(int(membership.shape[0]), -1, -1)
        elif int(x.shape[0]) != int(membership.shape[0]):
            raise ValueError(
                "shared-temperature row validity differs from the batched mask axis"
            )
    if int(x.shape[2]) < 2:
        raise ValueError("multiclass calibration requires at least two classes")
    device = x.device
    x64 = x.detach().to(device=device, dtype=torch.float64)
    mask_count, sample_count, class_count = map(int, x64.shape)
    labels = _multiclass_target_indices(
        targets,
        sample_count=sample_count,
        class_count=class_count,
        device=device,
    )
    weights = _shared_temperature_weights(
        sample_weights,
        weighting=weighting,
        mask_count=mask_count,
        sample_count=sample_count,
        device=device,
    )
    if membership is not None:
        weights = weights * membership
        if bool((weights.sum(dim=1) <= 0.0).any().item()):
            raise ValueError(
                "every shared-temperature outer fold requires positive row weight"
            )
    initial = _shared_temperature_initial_scale(
        initial_scale,
        mask_count=mask_count,
        class_count=class_count,
        minimum_scale=minimum_scale,
        device=device,
    )
    if not math.isfinite(float(maximum_scale)) or maximum_scale <= minimum_scale:
        raise ValueError("maximum shared-temperature scale must exceed its lower bound")

    initial_objective = _shared_temperature_statistics(
        x64, labels, weights, initial
    )[0]
    identity = torch.ones(mask_count, device=device, dtype=torch.float64)
    identity_objective = _shared_temperature_statistics(
        x64, labels, weights, identity
    )[0]
    uniform = torch.zeros(mask_count, device=device, dtype=torch.float64)
    intercept_only_objective = _shared_temperature_statistics(
        x64, labels, weights, uniform
    )[0]

    lower = torch.full_like(initial, float(minimum_scale))
    _, lower_gradient, _ = _shared_temperature_statistics(
        x64, labels, weights, lower
    )
    boundary = lower_gradient >= -float(tolerance)
    target_logits = x64.gather(
        2,
        labels.reshape(1, -1, 1).expand(mask_count, -1, 1),
    ).squeeze(2)
    asymptotic_gap = x64.amax(dim=2) - target_logits
    asymptotic_separation = (~boundary) & (
        ((asymptotic_gap <= 1e-12) | (weights <= 0.0)).all(dim=1)
    )
    upper = torch.maximum(initial, torch.ones_like(initial))
    upper = upper.clamp_max(float(maximum_scale))
    upper = torch.where(
        asymptotic_separation,
        torch.full_like(upper, float(maximum_scale)),
        upper,
    )
    _, upper_gradient, _ = _shared_temperature_statistics(
        x64, labels, weights, upper
    )
    expansion_steps = torch.zeros(mask_count, device=device, dtype=torch.int64)
    for expansion_index in range(64):
        unresolved = (~boundary) & (~asymptotic_separation) & (upper_gradient < 0.0) & (
            upper < float(maximum_scale)
        )
        # Resolved masks make every update below a masked no-op, so polling
        # on a cadence is bit-identical to polling every doubling.
        if expansion_index % NEWTON_ACTIVE_POLL_EVERY == 0 and not bool(
            unresolved.any().item()
        ):
            break
        upper = torch.where(
            unresolved,
            (2.0 * upper).clamp_max(float(maximum_scale)),
            upper,
        )
        expansion_steps = expansion_steps + unresolved.to(torch.int64)
        _, upper_gradient, _ = _shared_temperature_statistics(
            x64, labels, weights, upper
        )
    separated = asymptotic_separation | ((~boundary) & (upper_gradient < 0.0))
    interior = (~boundary) & (~separated)
    scale = torch.minimum(torch.maximum(initial, lower), upper)
    scale = torch.where(boundary, lower, torch.where(separated, upper, scale))
    converged = boundary.clone()
    iterations = torch.zeros(mask_count, device=device, dtype=torch.int64)
    line_search_steps = torch.zeros(mask_count, device=device, dtype=torch.int64)

    for _iteration in range(1, max(1, int(max_iterations)) + 1):
        active = interior & (~converged)
        # Converged masks make every update below a masked no-op, so host
        # polls run on a cadence instead of per iteration.
        if _iteration % NEWTON_ACTIVE_POLL_EVERY == 0 and not bool(
            active.any().item()
        ):
            break
        objective, gradient, hessian = _shared_temperature_statistics(
            x64, labels, weights, scale
        )
        iterations = iterations + active.to(torch.int64)
        gradient_converged = active & (gradient.abs() <= float(tolerance))
        converged = converged | gradient_converged
        active = active & (~gradient_converged)
        lower = torch.where(active & (gradient < 0.0), scale, lower)
        upper = torch.where(active & (gradient >= 0.0), scale, upper)
        midpoint = 0.5 * (lower + upper)
        newton = scale - gradient / hessian.clamp_min(float(hessian_ridge))
        usable_newton = (
            torch.isfinite(newton)
            & (newton > lower)
            & (newton < upper)
            & (hessian > float(hessian_ridge))
        )
        proposal = torch.where(usable_newton, newton, midpoint)
        proposal_objective = _shared_temperature_statistics(
            x64, labels, weights, proposal
        )[0]
        for _search in range(24):
            worsened = active & (
                proposal_objective > objective + float(objective_tolerance)
            )
            if _search % NEWTON_ACTIVE_POLL_EVERY == 0 and not bool(
                worsened.any().item()
            ):
                break
            proposal = torch.where(worsened, 0.5 * (proposal + scale), proposal)
            line_search_steps = line_search_steps + worsened.to(torch.int64)
            proposal_objective = _shared_temperature_statistics(
                x64, labels, weights, proposal
            )[0]
        scale = torch.where(active, proposal, scale)
        bracket_converged = active & (
            (upper - lower)
            <= float(tolerance) * torch.maximum(torch.ones_like(scale), scale.abs())
        )
        converged = converged | bracket_converged

    final_objective, final_gradient, _ = _shared_temperature_statistics(
        x64, labels, weights, scale
    )
    at_boundary = scale <= float(minimum_scale) + max(float(tolerance), 1e-12)
    final_kkt = torch.where(
        at_boundary,
        torch.clamp(-final_gradient, min=0.0),
        final_gradient.abs(),
    )
    finite = (
        torch.isfinite(scale)
        & torch.isfinite(final_objective)
        & torch.isfinite(final_kkt)
    )
    objective_worsened = final_objective > initial_objective + float(
        objective_tolerance
    )
    converged = (
        converged
        & finite
        & (~objective_worsened)
        & (~separated)
        & (final_kkt <= max(float(tolerance), 1e-12) * 10.0)
    )

    reasons = tuple(
        "complete_or_quasi_separation"
        if bool(separated[index].item())
        else "objective_worsened"
        if bool(objective_worsened[index].item())
        else "nonfinite"
        if not bool(finite[index].item())
        else "kkt_boundary"
        if bool(boundary[index].item()) and bool(converged[index].item())
        else "kkt_converged"
        if bool(converged[index].item())
        else "iteration_limit"
        for index in range(mask_count)
    )
    profile = BackendProfile(
        backend_id="mechanistic_systems.overlap_factored_calibration.v1",
        capability="shared_temperature_multiclass_calibration",
        implementation="torch_vectorized_float64_bracketed_newton",
        device_type=str(device.type),
        numerical_mode="float64_direct_positive_inverse_temperature",
        calls=int((iterations + expansion_steps).sum().item()),
        wall_time_s=max(0.0, time.perf_counter() - wall_start),
        peak_hbm_bytes=(
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        processed_contexts=int(mask_count * sample_count),
        processed_queries=int(mask_count * class_count),
        metadata={
            "parameter_semantics": "positive_inverse_temperature",
            "shared_offset": "omitted_as_softmax_gauge",
            "separated_component_count": int(separated.sum().item()),
            "row_validity_batched": membership is not None,
        },
    )
    component_hash = factorization.components[0].component_id
    receipt = CalibrationFitReceipt(
        dependency_hash=factorization.dependency_hash,
        factorization_hash=factorization.factorization_hash,
        component_hashes=tuple(component_hash for _ in range(mask_count)),
        backend="torch_vectorized_float64_bracketed_newton",
        initial_objective=_flat_tuple(initial_objective),
        final_objective=_flat_tuple(final_objective),
        identity_objective=_flat_tuple(identity_objective),
        intercept_only_objective=_flat_tuple(intercept_only_objective),
        iterations=_flat_tuple(iterations + expansion_steps, int),
        line_search_steps=_flat_tuple(line_search_steps, int),
        gradient_kkt_residual=_flat_tuple(final_kkt),
        termination_reason=reasons,
        finite=_flat_tuple(finite, bool),
        converged=_flat_tuple(converged, bool),
        backend_profile=profile.to_runtime_dict(),
    )
    expanded_scale = scale.reshape(-1, 1).expand(-1, class_count)
    fit = FactoredCalibrationFit(
        scale=expanded_scale.to(dtype=x.dtype).detach(),
        offset=torch.zeros_like(expanded_scale, dtype=x.dtype).detach(),
        factorization=factorization,
        receipt=receipt,
        input_was_batched=input_was_batched,
        family="overlap_factored_shared_temperature",
    )
    if require_convergence and not receipt.passed:
        raise CalibrationConvergenceError(
            "shared-temperature calibration failed its convergence receipt: "
            f"reasons={receipt.termination_reason!r} "
            f"max_kkt={max(receipt.gradient_kkt_residual):.6g} "
            f"finite={receipt.finite}",
            receipt,
        )
    return fit


def _require_torch() -> None:
    if torch is None or F is None:
        raise RuntimeError("factorized calibration requires PyTorch")


def _as_batched_values(values: Any) -> tuple[Any, bool]:
    tensor = torch.as_tensor(values)
    if tensor.ndim == 2:
        return tensor.unsqueeze(0), False
    if tensor.ndim == 3:
        return tensor, True
    raise ValueError("calibration values must have shape [sample,query] or [mask,sample,query]")


def _row_weights(
    labels: Any,
    *,
    weighting: str,
    sample_weights: Any | None,
    dtype: Any,
    device: Any,
    row_validity: Any | None = None,
) -> Any:
    """Per-row calibration weights.

    Without ``row_validity`` this returns the ``[sample, query]``
    weights. With ``row_validity`` (a binary ``[mask, sample]`` membership
    matrix — e.g. one row-mask per crossfit fold) it returns
    ``[mask, sample, query]`` weights where every statistic that shapes the
    weights (class support, positive ratios, design normalization) is
    computed within the valid rows of each mask. Zero-validity rows carry
    exactly zero weight, so the per-(mask, query) argmin equals the
    corresponding subset fit.
    """

    if loss_weighting() == "per_image" and weighting == "pos_weighted":
        weighting = "design_weighted" if sample_weights is not None else "unweighted"
    y = torch.as_tensor(labels, device=device, dtype=dtype)
    sample_count, query_count = map(int, y.shape)
    if row_validity is not None:
        validity = torch.as_tensor(row_validity, device=device, dtype=dtype)
        if validity.ndim != 2 or int(validity.shape[1]) != sample_count:
            raise ValueError("row validity must have shape [mask, sample]")
        if not bool(((validity == 0.0) | (validity == 1.0)).all().item()):
            raise ValueError("row validity must be binary membership")
        valid = validity.unsqueeze(2)  # [M, N, 1]
        positives = (valid * y.unsqueeze(0)).sum(dim=1)  # [M, K]
        negatives = valid.sum(dim=1) - positives  # [M, K] (broadcast [M,1])
        if bool(((positives <= 0) | (negatives <= 0)).any().item()):
            raise ValueError(
                "every calibration query requires positive and negative "
                "support within every row mask"
            )
        result = valid.expand(-1, sample_count, query_count).clone()
        if weighting == "pos_weighted":
            ratio = (negatives / positives).unsqueeze(1)  # [M, 1, K]
            result = torch.where(y.unsqueeze(0) > 0.5, result * ratio, result)
        elif weighting == "design_weighted":
            if sample_weights is None:
                raise ValueError("design-weighted calibration requires sample weights")
        elif weighting != "unweighted":
            raise ValueError(f"unsupported calibration weighting {weighting!r}")
        if sample_weights is not None:
            design = torch.as_tensor(sample_weights, device=device, dtype=dtype)
            if design.ndim == 1:
                design = design.unsqueeze(1)
            if tuple(design.shape) not in {(sample_count, 1), (sample_count, query_count)}:
                raise ValueError("calibration sample weights differ from sample/query axes")
            if not bool(torch.isfinite(design).all().item()) or bool((design < 0).any().item()):
                raise ValueError("calibration sample weights must be finite and nonnegative")
            design_rows = design.unsqueeze(0)  # [1, N, 1|K]
            valid_mean = (valid * design_rows).sum(dim=1) / valid.sum(dim=1).clamp_min(
                1e-12
            )  # [M, 1|K]
            design_rows = design_rows / valid_mean.unsqueeze(1).clamp_min(1e-12)
            result = result * design_rows
        if loss_weighting() == "per_image":
            # Statistics divide by the physical sample count, including
            # excluded fold rows; each cell must still report a mean.
            result = result * sample_count / result.sum(dim=1, keepdim=True).clamp_min(1e-300)
        return result
    positives = y.sum(dim=0)
    negatives = float(sample_count) - positives
    if bool(((positives <= 0) | (negatives <= 0)).any().item()):
        raise ValueError("every calibration query requires positive and negative support")
    result = torch.ones((sample_count, query_count), device=device, dtype=dtype)
    if weighting == "pos_weighted":
        result = torch.where(y > 0.5, (negatives / positives).unsqueeze(0), result)
    elif weighting == "design_weighted":
        if sample_weights is None:
            raise ValueError("design-weighted calibration requires sample weights")
    elif weighting != "unweighted":
        raise ValueError(f"unsupported calibration weighting {weighting!r}")
    if sample_weights is not None:
        design = torch.as_tensor(sample_weights, device=device, dtype=dtype)
        if design.ndim == 1:
            design = design.unsqueeze(1)
        if tuple(design.shape) not in {(sample_count, 1), (sample_count, query_count)}:
            raise ValueError("calibration sample weights differ from sample/query axes")
        if not bool(torch.isfinite(design).all().item()) or bool((design < 0).any().item()):
            raise ValueError("calibration sample weights must be finite and nonnegative")
        design = design / design.mean(dim=0, keepdim=True).clamp_min(1e-12)
        result = result * design
    return result


def _stats(values: Any, labels: Any, weights: Any, scale: Any, offset: Any) -> tuple[Any, ...]:
    return calibration_statistics(
        values, labels, weights, scale, offset, backend="torch"
    ).values()


def _flat_tuple(values: Any, cast: Any = float) -> tuple[Any, ...]:
    return tuple(cast(value) for value in values.detach().cpu().reshape(-1).tolist())










def _termination_reasons(
    *,
    scale_identifiable: Any,
    separated: Any,
    finite: Any,
    objective_worsened: Any,
    failed_line_search: Any,
    converged: Any,
) -> list[str]:
    """Row-major per-cell termination reasons, one host transfer per flag.

    Byte-identical strings and ordering to a per-cell `.item()` loop.
    """

    flags = [
        tensor.detach().reshape(-1).to(device="cpu", dtype=torch.bool).tolist()
        for tensor in (
            scale_identifiable,
            separated,
            finite,
            objective_worsened,
            failed_line_search,
            converged,
        )
    ]
    return _termination_reasons_from_host_flags(*flags)


def _termination_reasons_from_host_flags(
    scale_identifiable: Sequence[Any],
    separated: Sequence[Any],
    finite: Sequence[Any],
    objective_worsened: Sequence[Any],
    failed_line_search: Sequence[Any],
    converged: Sequence[Any],
) -> list[str]:
    """``_termination_reasons`` over flag lists already on the host."""

    reasons: list[str] = []
    for identifiable, is_separated, is_finite, worsened, failed, ok in zip(
        scale_identifiable,
        separated,
        finite,
        objective_worsened,
        failed_line_search,
        converged,
        strict=True,
    ):
        if not identifiable:
            reasons.append("intercept_only_scale_unidentifiable")
        elif is_separated:
            reasons.append("complete_or_quasi_separation")
        elif not is_finite:
            reasons.append("nonfinite")
        elif worsened:
            reasons.append("objective_worsened")
        elif failed:
            reasons.append("line_search_failed")
        elif ok:
            reasons.append("kkt_converged")
        else:
            reasons.append("iteration_limit")
    return reasons


def calibration_tensor_content_hash(value: Any) -> str:
    tensor = torch.as_tensor(value).detach().to(device="cpu").contiguous()
    return stable_digest({"dtype": str(tensor.dtype), "shape": tuple(int(v) for v in tensor.shape), "sha256": hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest()})


def _scientific_fit_hash(receipt: CalibrationFitReceipt) -> str:
    payload = receipt.to_dict(); payload.pop("backend_profile", None); return stable_digest(payload)


def _build_positive_affine_owner(
    *, x64_raw: Any, target: Any, weights: Any, optimization_target: Any,
    optimization_weights: Any, anchor_strength: Any, identity_probability: Any,
    location: Any, raw_spread: Any, spread: Any, minimum_scale: float,
    minimum_conditioned_scale: Any,
    scale_identifiability_tolerance: float, source_dtype: Any, source_epsilon: float,
    probability_epsilon: float, tolerance: float, factorization: CalibrationFactorization,
    fit: FactoredCalibrationFit, context_ids: Sequence[Any],
) -> PositiveAffineDifferentialOwnerReceipt:
    ids = tuple(str(value) for value in context_ids)
    if len(ids) != int(x64_raw.shape[1]) or len(set(ids)) != len(ids): raise ValueError("differential calibration context IDs must be unique and ordered")
    raw_scale = fit.scale.detach().to(device=x64_raw.device, dtype=torch.float64)
    raw_offset = fit.offset.detach().to(device=x64_raw.device, dtype=torch.float64)
    scale, offset = raw_scale * spread, raw_offset + raw_scale * location
    conditioned = (x64_raw - location.unsqueeze(1)) / spread.unsqueeze(1)
    probability = torch.sigmoid(scale.unsqueeze(1) * conditioned + offset.unsqueeze(1))
    residual = optimization_weights * (probability - optimization_target) / float(x64_raw.shape[1])
    curvature = optimization_weights * probability * (1.0 - probability) / float(x64_raw.shape[1])
    grad_scale = (residual * conditioned).sum(dim=1); grad_offset = residual.sum(dim=1)
    hessian_scale = (curvature * conditioned.square()).sum(dim=1)
    hessian_cross = (curvature * conditioned).sum(dim=1); hessian_offset = curvature.sum(dim=1)
    raw_grad_scale = spread * grad_scale + location * grad_offset
    raw_grad_offset = grad_offset
    tau_id = float(scale_identifiability_tolerance)
    tau_id_face = min(max(64.0 * float(torch.finfo(torch.float64).eps), 0.1 * tau_id), 0.5 * tau_id)
    boundary_tolerance = max(float(tolerance), 1e-12)
    unclamped = torch.sigmoid(x64_raw)
    clamp_tau = 64.0 * float(torch.finfo(torch.float64).eps)
    near_clamp = ((unclamped - probability_epsilon).abs() <= clamp_tau) | ((unclamped - (1.0 - probability_epsilon)).abs() <= clamp_tau)
    if bool(near_clamp.any().item()):
        raise CalibrationDifferentialOwnerError("ANCHOR_CLAMP_FACE_UNRESOLVED", "anchor probability is within the clamp transition margin")
    clamp_code = torch.where(unclamped <= probability_epsilon, torch.zeros_like(unclamped, dtype=torch.uint8), torch.where(unclamped >= 1.0 - probability_epsilon, torch.full_like(unclamped, 2, dtype=torch.uint8), torch.ones_like(unclamped, dtype=torch.uint8)))
    faces = []
    mask_count, _, query_count = map(int, x64_raw.shape)
    for mask_index in range(mask_count):
        for query_index in range(query_count):
            raw_sigma = float(raw_spread[mask_index, query_index].item())
            effective_sigma = float(spread[mask_index, query_index].item())
            if raw_sigma > tau_id + tau_id_face:
                identifiability = "identified"
            elif raw_sigma < tau_id - tau_id_face:
                identifiability = "quotient_fixed_scale"
            else:
                raise CalibrationDifferentialOwnerError("SCALE_IDENTIFIABILITY_FACE_UNRESOLVED", "weighted logit spread lies in the identifiability transition band")
            a = float(raw_scale[mask_index, query_index].item()); ga = float(raw_grad_scale[mask_index, query_index].item()); gc = float(raw_grad_offset[mask_index, query_index].item())
            primal = a - float(minimum_scale)
            tau_primal = boundary_tolerance / max(effective_sigma, 1e-30)
            tau_dual = max(float(tolerance), 64.0 * float(source_epsilon))
            tau_stationarity = tau_dual * (1.0 + abs(float(location[mask_index, query_index].item())) + effective_sigma)
            tau_complementarity = max(tau_primal * tau_dual, 64.0 * float(torch.finfo(torch.float64).eps))
            multiplier = 0.0
            if identifiability == "quotient_fixed_scale":
                scale_class, free = "quotient_fixed_scale", (1,)
                stationarity = abs(gc)
            elif primal > tau_primal:
                scale_class, free = "interior", (0, 1)
                stationarity = max(abs(ga), abs(gc))
            elif abs(primal) <= tau_primal:
                multiplier = ga
                if abs(multiplier) <= tau_dual:
                    raise CalibrationDifferentialOwnerError("POSITIVE_SCALE_FACE_UNRESOLVED", "scale lies on a degenerate bound face")
                if multiplier < -tau_dual:
                    raise CalibrationDifferentialOwnerError("POSITIVE_SCALE_FACE_INVALID", "lower-bound multiplier is negative")
                scale_class, free = "certified_active", (1,)
                stationarity = max(abs(ga - multiplier), abs(gc))
            else:
                raise CalibrationDifferentialOwnerError("POSITIVE_SCALE_FACE_INVALID", "raw affine scale is infeasible")
            complementarity = abs(multiplier * primal)
            if stationarity > tau_stationarity or complementarity > tau_complementarity:
                raise CalibrationDifferentialOwnerError("POSITIVE_SCALE_FACE_INVALID", "raw-coordinate KKT residual exceeds its owner tolerance")
            hss = float(hessian_scale[mask_index, query_index].item()); hso = float(hessian_cross[mask_index, query_index].item()); hoo = float(hessian_offset[mask_index, query_index].item())
            mu = float(location[mask_index, query_index].item())
            h_aa = effective_sigma**2 * hss + 2.0 * effective_sigma * mu * hso + mu**2 * hoo
            h_ac = effective_sigma * hso + mu * hoo
            code_hash = calibration_tensor_content_hash(clamp_code[mask_index, :, query_index])
            faces.append(PositiveAffineComponentFace(
                mask_index, query_index, a, float(raw_offset[mask_index, query_index].item()),
                float(scale[mask_index, query_index].item()), float(offset[mask_index, query_index].item()),
                mu, raw_sigma, effective_sigma, float(minimum_scale),
                float(minimum_conditioned_scale[mask_index, query_index].item()),
                float(anchor_strength[mask_index, query_index].item()),
                float(weights[:, query_index].sum().item()), tau_id, tau_id_face,
                identifiability, scale_class, free, (ga, gc), multiplier, primal,
                stationarity, complementarity,
                (tau_primal, tau_dual, tau_stationarity, tau_complementarity),
                ((h_aa, h_ac), (h_ac, hoo)), code_hash,
            ))
    state_hash = stable_digest({
        "optimization_target": calibration_tensor_content_hash(optimization_target),
        "optimization_weights": calibration_tensor_content_hash(optimization_weights),
        "anchor_strength": calibration_tensor_content_hash(anchor_strength),
        "identity_probability": calibration_tensor_content_hash(identity_probability),
    })
    result = PositiveAffineDifferentialOwnerReceipt(
        revision_identity=SCALAR_V2R_REVISION4_SPEC_SHA256,
        solver_owner_identity=stable_digest("factored_calibration.projected_newton_owner.v1"),
        factorization_hash=factorization.factorization_hash,
        fit_receipt_hash=_scientific_fit_hash(fit.receipt),
        logits_hash=calibration_tensor_content_hash(x64_raw), targets_hash=calibration_tensor_content_hash(target),
        row_weights_hash=calibration_tensor_content_hash(weights),
        zero_weight_rows_hash=calibration_tensor_content_hash(weights == 0.0),
        optimization_state_hash=state_hash, context_ids_hash=stable_digest(ids),
        optimization_backend=fit.receipt.backend,
        source_logit_dtype=str(source_dtype),
        source_logit_epsilon=float(source_epsilon), anchor_probability_epsilon=float(probability_epsilon),
        objective_reduction="weighted_mean_over_sample_axis_v1", component_faces=tuple(faces),
    )
    result.validate(); return result


def _torch_statistics_objective(
    values: Any, labels: Any, weights: Any, scale: Any, offset: Any
) -> Any:
    """``torch_calibration_statistics(...).objective`` without the other five.

    The same torch ops in the same order as the canonical reducer's objective
    (``calibration_statistics.torch_calibration_statistics``), so the value is
    that reducer's objective bit for bit; the
    residual, curvature and their five reductions are not computed.
    """

    eta = values * scale.unsqueeze(1) + offset.unsqueeze(1)
    y = (
        labels.unsqueeze(0).expand(int(eta.shape[0]), -1, -1)
        if labels.ndim == 2
        else labels
    )
    w = (
        weights.unsqueeze(0).expand(int(eta.shape[0]), -1, -1)
        if weights.ndim == 2
        else weights
    )
    if tuple(y.shape) != tuple(eta.shape) or tuple(w.shape) != tuple(eta.shape):
        raise ValueError(
            "calibration labels and weights must broadcast exactly over "
            "[mask,sample,query]"
        )
    loss = (F.softplus(eta) - y * eta) * w
    return loss.sum(dim=1) / float(values.shape[1])


def _guarded_projected_newton(
    *,
    stats: Callable[[Any, Any], tuple[Any, ...]],
    stats_objective_stack: Callable[[Any, Any], Any] | None,
    scale: Any,
    offset: Any,
    minimum_conditioned_scale: Any,
    max_iterations: int,
    tolerance: float,
    hessian_ridge: float,
    objective_tolerance: float,
    line_search_limit: int = 30,
    alpha_chunk: int | None = None,
    proposal_objective_fn: Callable[[Any, Any], Any] | None = None,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """The production guarded projected Newton over a ``[mask, query]`` grid.

    ``stats(scale, offset)`` returns the six anchored statistics of every
    cell and ``stats_objective_stack`` the objective of a ``[width, mask,
    query]`` candidate stack (used only for chunks wider than one).  The body
    is the loop of ``_fit_batched_post_query_affine_core``
    (cadence polls; chunked first-accept Armijo over the 2^-k ladder);
    the defaults are the production ladder (30 halvings, chunk
    ``LINE_SEARCH_ALPHA_CHUNK``).  The v6 device fallback re-runs the same
    loop with a larger iteration bound and a longer sequential ladder.
    ``proposal_objective_fn`` (v6) evaluates a width-one proposal's objective
    alone; it must equal ``stats(...)[0]`` exactly
    (``_torch_statistics_objective``).

    Returns ``(scale, offset, converged, failed_line_search, iterations,
    line_steps)``; the closing KKT pass is ``_newton_final_gates``.
    """

    device = scale.device
    converged = torch.zeros(tuple(scale.shape), device=device, dtype=torch.bool)
    failed_line_search = torch.zeros_like(converged)
    iterations = torch.zeros(tuple(scale.shape), device=device, dtype=torch.int64)
    line_steps = torch.zeros_like(iterations)
    search_limit = int(line_search_limit)
    chunk_width = max(
        1, int(LINE_SEARCH_ALPHA_CHUNK if alpha_chunk is None else alpha_chunk)
    )
    for iteration in range(1, max(1, int(max_iterations)) + 1):
        objective, grad_s, grad_o, h_ss, h_so, h_oo = stats(scale, offset)
        at_boundary = scale <= minimum_conditioned_scale + max(
            float(tolerance), 1e-12
        )
        kkt_scale = torch.where(at_boundary, torch.clamp(-grad_s, min=0.0), grad_s.abs())
        kkt = torch.maximum(kkt_scale, grad_o.abs())
        newly_converged = kkt <= float(tolerance)
        converged |= newly_converged
        active = ~(converged | failed_line_search)
        # Frozen cells' updates below are `where`-masked no-ops, so a cadence
        # poll is bit-identical to per-iteration polling. On CPU the
        # poll is free — keep the per-iteration early exit there.
        active_poll_every = (
            NEWTON_ACTIVE_POLL_EVERY if device.type == "cuda" else 1
        )
        if (
            active_poll_every == 1 or iteration % active_poll_every == 1
        ) and not bool(active.any().item()):
            break
        iterations[active] = iteration

        h_ss = h_ss + float(hessian_ridge)
        h_oo = h_oo + float(hessian_ridge)
        determinant = h_ss * h_oo - h_so.square()
        stable = determinant > max(float(hessian_ridge) ** 2, 1e-30)
        determinant = determinant.clamp_min(max(float(hessian_ridge) ** 2, 1e-30))
        coupled_step_s = (h_oo * grad_s - h_so * grad_o) / determinant
        coupled_step_o = (h_ss * grad_o - h_so * grad_s) / determinant
        diagonal_step_s = grad_s / h_ss.clamp_min(float(hessian_ridge))
        diagonal_step_o = grad_o / h_oo.clamp_min(float(hessian_ridge))
        step_s = torch.where(stable, coupled_step_s, diagonal_step_s)
        step_o = torch.where(stable, coupled_step_o, diagonal_step_o)
        # A boundary optimum cannot move toward an infeasible negative scale.
        boundary_active = at_boundary & (grad_s >= 0.0)
        step_s = torch.where(
            boundary_active,
            torch.zeros_like(step_s),
            step_s,
        )
        step_o = torch.where(
            boundary_active,
            diagonal_step_o,
            step_o,
        )
        direction_s = -step_s
        direction_o = -step_o

        # Chunked alpha-grid backtracking: the same 2^-k candidate
        # ladder and Armijo predicate as the sequential search,
        # evaluated a chunk at a time with device-side first-accept, so one
        # host poll per chunk boundary replaces one per candidate. The first
        # chunk has width one — the overwhelmingly common accept-at-alpha=1
        # case pays no extra reducer work and keeps the [mask, query]
        # statistics shapes bit-exactly.
        accepted = torch.zeros_like(active)
        search_base = 0
        while search_base < search_limit:
            width = (
                1
                if search_base == 0
                else min(chunk_width, search_limit - search_base)
            )
            alphas = torch.tensor(
                [0.5**k for k in range(search_base, search_base + width)],
                device=device,
                dtype=torch.float64,
            ).reshape(width, 1, 1)
            proposal_scale = torch.maximum(
                scale.unsqueeze(0) + alphas * direction_s.unsqueeze(0),
                minimum_conditioned_scale.unsqueeze(0),
            )
            proposal_offset = offset.unsqueeze(0) + alphas * direction_o.unsqueeze(0)
            if width == 1:
                proposal_objective = (
                    stats(proposal_scale[0], proposal_offset[0])[0]
                    if proposal_objective_fn is None
                    else proposal_objective_fn(proposal_scale[0], proposal_offset[0])
                ).unsqueeze(0)
            else:
                proposal_objective = stats_objective_stack(
                    proposal_scale, proposal_offset
                )
            delta_s = proposal_scale - scale.unsqueeze(0)
            delta_o = proposal_offset - offset.unsqueeze(0)
            directional = grad_s.unsqueeze(0) * delta_s + grad_o.unsqueeze(0) * delta_o
            sufficient = (
                proposal_objective
                <= objective.unsqueeze(0)
                + 1e-4 * directional
                + float(objective_tolerance)
            ) & torch.isfinite(proposal_objective)
            any_sufficient = sufficient.any(dim=0)
            first_index = sufficient.to(torch.int64).argmax(dim=0)
            take = active & ~accepted & any_sufficient
            gather_index = first_index.unsqueeze(0)
            chosen_scale = proposal_scale.gather(0, gather_index).squeeze(0)
            chosen_offset = proposal_offset.gather(0, gather_index).squeeze(0)
            scale = torch.where(take, chosen_scale, scale)
            offset = torch.where(take, chosen_offset, offset)
            line_steps[take] = (search_base + first_index + 1)[take]
            accepted |= take
            if bool((accepted | ~active).all().item()):
                break
            search_base += width
        failed_line_search |= active & ~accepted
    return scale, offset, converged, failed_line_search, iterations, line_steps


def _newton_final_gates(
    *,
    stats: Callable[[Any, Any], tuple[Any, ...]],
    scale: Any,
    offset: Any,
    minimum_conditioned_scale: Any,
    tolerance: float,
    objective_tolerance: float,
    initial_objective: Any,
    converged: Any,
    failed_line_search: Any,
    separated: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Closing KKT pass and the per-cell hard gates of the production solve.

    Updates ``converged`` in place and
    returns ``(final_objective, grad_scale, grad_offset, final_kkt,
    converged, finite, objective_worsened)``.
    """

    final_objective, grad_s, grad_o, *_ = stats(scale, offset)
    at_boundary = scale <= minimum_conditioned_scale + max(
        float(tolerance), 1e-12
    )
    final_kkt = torch.maximum(
        torch.where(at_boundary, torch.clamp(-grad_s, min=0.0), grad_s.abs()),
        grad_o.abs(),
    )
    converged |= final_kkt <= float(tolerance)
    finite = (
        torch.isfinite(scale)
        & torch.isfinite(offset)
        & torch.isfinite(final_objective)
        & torch.isfinite(final_kkt)
    )
    objective_worsened = final_objective > initial_objective + float(objective_tolerance)
    converged &= finite & ~objective_worsened & ~failed_line_search & ~separated
    return (
        final_objective,
        grad_s,
        grad_o,
        final_kkt,
        converged,
        finite,
        objective_worsened,
    )






def _finish_post_query_affine_fit(
    *,
    scale: Any,
    offset: Any,
    spread: Any,
    location: Any,
    x_dtype: Any,
    factorization: CalibrationFactorization,
    receipt: CalibrationFitReceipt,
    input_was_batched: bool,
    required_bool: Any | None,
    required_passed: bool | None,
    require_convergence: bool,
    build_owner: Callable[[FactoredCalibrationFit], Any] | None,
) -> tuple[FactoredCalibrationFit, PositiveAffineDifferentialOwnerReceipt | None]:
    """Map conditioned parameters to raw ones, gate, and emit the owner."""

    raw_scale = scale / spread
    raw_offset = offset - raw_scale * location
    fit = FactoredCalibrationFit(
        scale=raw_scale.to(dtype=x_dtype).detach(),
        offset=raw_offset.to(dtype=x_dtype).detach(),
        factorization=factorization,
        receipt=receipt,
        input_was_batched=input_was_batched,
    )
    # With a declared required set, only required cells gate the fit; other
    # columns are enrichment and report through the receipt.
    gate_passed = receipt.passed if required_bool is None else bool(required_passed)
    if require_convergence and not gate_passed:
        raise CalibrationConvergenceError(
            "factorized affine calibration failed its convergence receipt: "
            f"reasons={receipt.termination_reason!r} "
            f"max_kkt={max(receipt.gradient_kkt_residual):.6g} "
            f"required_passed={receipt.required_passed}",
            receipt,
        )
    owner = None if build_owner is None else build_owner(fit)
    return fit, owner


def _host_cell_vectors(cells: Sequence[Any], scalars: Sequence[Any]) -> tuple[list[list[float]], list[float]]:
    """ONE device-to-host transfer for the receipt's per-cell and scalar values.

    Every value is widened to float64 on the device (exact for float64, for
    booleans and for integer counts below 2**53) and copied once, instead of
    one blocking copy per receipt field and per diagnostic.
    """

    flat_cells = torch.stack(
        [value.detach().reshape(-1).to(torch.float64) for value in cells]
    )
    width = int(flat_cells.shape[1])
    flat_scalars = (
        torch.stack(
            [torch.as_tensor(value).detach().reshape(()).to(device=flat_cells.device, dtype=torch.float64) for value in scalars]
        )
        if len(scalars)
        else torch.empty((0,), device=flat_cells.device, dtype=torch.float64)
    )
    host = torch.cat((flat_cells.reshape(-1), flat_scalars)).to(device="cpu").tolist()
    rows = [host[index * width : (index + 1) * width] for index in range(len(cells))]
    return rows, host[len(cells) * width :]


def _recover_newton_cells(
    *, positions, x64, optimization_target, optimization_weights,
    anchor_objective_constant, initial_conditioned_scale,
    initial_conditioned_offset, minimum_conditioned_scale,
    initial_objective, max_iterations, tolerance, hessian_ridge, objective_tolerance,
):
    """Continue the solver algorithm with its extended Newton budget."""

    masks, samples, queries = map(int, x64.shape)
    def rows(value):
        return value.expand_as(x64).permute(0, 2, 1).reshape(masks * queries, samples).index_select(0, positions).unsqueeze(-1)
    def cells(value):
        return value.expand(masks, queries).reshape(-1).index_select(0, positions).unsqueeze(-1)
    values, targets, weights = rows(x64), rows(optimization_target), rows(optimization_weights)
    constant = cells(anchor_objective_constant)
    minimum = cells(minimum_conditioned_scale)
    scale, offset = cells(initial_conditioned_scale), cells(initial_conditioned_offset)
    start = cells(initial_objective)
    def stats(a, b):
        result = list(calibration_statistics(values, targets, weights, a, b, backend="torch").values())
        result[0] = result[0] - constant
        return tuple(result)
    def objective(a, b):
        return _torch_statistics_objective(values, targets, weights, a, b) - constant
    scale, offset, converged, failed, iterations, steps = _guarded_projected_newton(
        stats=stats, stats_objective_stack=None, scale=scale, offset=offset,
        minimum_conditioned_scale=minimum,
        max_iterations=max(1, int(max_iterations)) * _GPU_FALLBACK_ITERATION_FACTOR,
        tolerance=tolerance, hessian_ridge=hessian_ridge,
        objective_tolerance=objective_tolerance,
        line_search_limit=_GPU_FALLBACK_LINE_SEARCH_LIMIT,
        alpha_chunk=1, proposal_objective_fn=objective,
    )
    final, _, _, kkt, converged, finite, worsened = _newton_final_gates(
        stats=stats, scale=scale, offset=offset, minimum_conditioned_scale=minimum,
        tolerance=tolerance, objective_tolerance=objective_tolerance,
        initial_objective=start, converged=converged, failed_line_search=failed,
        separated=torch.zeros_like(converged),
    )
    converged = converged & (kkt <= float(tolerance)) & (scale >= minimum)
    return {name: value.reshape(-1) for name, value in (
        ("scale", scale), ("offset", offset), ("final_objective", final),
        ("final_kkt", kkt), ("converged", converged), ("finite", finite),
        ("objective_worsened", worsened), ("failed_line_search", failed),
        ("iterations", iterations), ("line_steps", steps),
    )}


def _finish_with_device_hard_gates_v6(
    *, x_dtype: Any, x64: Any, scale: Any, offset: Any,
    optimization_target: Any, optimization_weights: Any,
    anchor_objective_constant: Any, initial_conditioned_scale: Any,
    initial_conditioned_offset: Any, max_iterations: int,
    hessian_ridge: float, objective_tolerance: float,
    final_objective: Any, final_kkt: Any, finite: Any, converged: Any,
    objective_worsened: Any, failed_line_search: Any, iterations: Any,
    line_steps: Any, separated: Any, scale_identifiable: Any,
    minimum_conditioned_scale: Any, initial_objective: Any,
    identity_objective: Any, intercept_only_objective: Any,
    location: Any, spread: Any, raw_spread: Any, anchor_strength: Any,
    row_mass_min: Any, row_validity_provided: bool, tolerance: float,
    scale_identifiability_tolerance: float, source_dtype: Any,
    source_epsilon: float, probability_epsilon: float,
    resolved_statistics_backend: str, required_query_mask: Any | None,
    factorization: CalibrationFactorization, input_was_batched: bool,
    require_convergence: bool, wall_start: float,
    build_owner: Callable[[FactoredCalibrationFit], Any] | None,
) -> tuple[FactoredCalibrationFit, PositiveAffineDifferentialOwnerReceipt | None]:
    """Materialize the single solve's convergence diagnostics and affine fit."""

    device = scale.device
    mask_count, query_count = map(int, scale.shape)
    required_bool = None
    if required_query_mask is not None:
        required_bool = torch.as_tensor(required_query_mask, dtype=torch.bool).reshape(-1)
        if int(required_bool.shape[0]) != query_count:
            raise ValueError("required query mask differs from the query axis")
    converged = (
        converged & finite & (final_kkt <= float(tolerance))
        & (scale >= minimum_conditioned_scale)
    )
    retry = ~(finite & converged) & ~separated
    positions = retry.reshape(-1).nonzero(as_tuple=False).reshape(-1)
    recovery_status = torch.zeros_like(converged, dtype=torch.int8)
    production_flags = (finite, converged, objective_worsened, failed_line_search)
    recovered = None
    if positions.numel():
        recovered = _recover_newton_cells(
            positions=positions, x64=x64, optimization_target=optimization_target,
            optimization_weights=optimization_weights,
            anchor_objective_constant=anchor_objective_constant,
            initial_conditioned_scale=initial_conditioned_scale,
            initial_conditioned_offset=initial_conditioned_offset,
            minimum_conditioned_scale=minimum_conditioned_scale,
            initial_objective=initial_objective, max_iterations=max_iterations,
            tolerance=tolerance, hessian_ridge=hessian_ridge,
            objective_tolerance=objective_tolerance,
        )
        adopt = recovered["finite"] & recovered["converged"]
        adopted = positions[adopt]
        refused = positions[~adopt]
        updated = []
        for target, name in (
            (scale, "scale"), (offset, "offset"), (final_objective, "final_objective"),
            (final_kkt, "final_kkt"), (converged, "converged"), (finite, "finite"),
            (objective_worsened, "objective_worsened"), (failed_line_search, "failed_line_search"),
            (iterations, "iterations"), (line_steps, "line_steps"),
        ):
            result = target.clone(memory_format=torch.contiguous_format)
            result.view(-1)[adopted] = recovered[name][adopt]
            updated.append(result)
        (scale, offset, final_objective, final_kkt, converged, finite,
         objective_worsened, failed_line_search, iterations, line_steps) = updated
        recovery_status.view(-1)[adopted] = 1
        recovery_status.view(-1)[refused] = 2
    cell_rows, scalars = _host_cell_vectors(
        (initial_objective, final_objective, identity_objective,
         intercept_only_objective, iterations, line_steps, final_kkt,
         finite, converged, scale_identifiable, separated,
         objective_worsened, failed_line_search, recovery_status, *production_flags),
        (location.abs().max(), spread.min(), spread.max(), raw_spread.min(),
         raw_spread.max(), (~scale_identifiable).sum(), separated.sum(),
         anchor_strength.min(), anchor_strength.max(), row_mass_min, iterations.sum()),
    )
    (initial_host, final_host, identity_host, intercept_host, iterations_host,
     line_steps_host, kkt_host, finite_host, converged_host, identifiable_host,
     separated_host, worsened_host, failed_host, recovery_host,
     production_finite, production_converged, production_worsened,
     production_failed) = cell_rows
    (location_abs_max, spread_min, spread_max, raw_spread_min, raw_spread_max,
     unidentifiable_count, separated_count, anchor_min, anchor_max,
     row_mass_min, iteration_total) = scalars
    finite_flags = tuple(bool(value) for value in finite_host)
    converged_flags = tuple(bool(value) for value in converged_host)
    reasons = _termination_reasons_from_host_flags(
        [bool(value) for value in identifiable_host],
        [bool(value) for value in separated_host], list(finite_flags),
        [bool(value) for value in worsened_host],
        [bool(value) for value in failed_host], list(converged_flags),
    )
    reasons = tuple(("gpu_fallback_" + reason) if int(status) == 1 else
                    ("gpu_fallback_failed_" + reason) if int(status) == 2 else reason
                    for reason, status in zip(reasons, recovery_host, strict=True))
    missing_positions = tuple(i for i, ok in enumerate(converged_flags) if not ok)
    rescued_cells, refused_cells, failed_gate_counts = [], [], {}
    if recovered is not None:
        recovered_fields = ("final_objective", "final_kkt", "finite", "converged", "iterations", "line_steps")
        recovered_rows, _ = _host_cell_vectors(tuple(recovered[name] for name in recovered_fields), ())
        for index, position in enumerate(positions.tolist()):
            gate = _v6_failed_gate(
                finite=bool(production_finite[position]),
                objective_worsened=bool(production_worsened[position]),
                failed_line_search=bool(production_failed[position]),
                converged=bool(production_converged[position]),
            )
            failed_gate_counts[gate] = failed_gate_counts.get(gate, 0) + 1
            final, kkt, cell_finite, cell_converged, cell_iterations, cell_steps = (
                row[index] for row in recovered_rows
            )
            cell = {
                "position": int(position), "mask_index": int(position) // query_count,
                "query_index": int(position) % query_count, "production_failed_gate": gate,
                "rerun": {
                    "final_objective": float(final) if math.isfinite(final) else None,
                    "gradient_kkt_residual": float(kkt) if math.isfinite(kkt) else None,
                    "finite": bool(cell_finite), "converged": bool(cell_converged),
                    "iterations": int(cell_iterations), "line_search_steps": int(cell_steps),
                },
            }
            (rescued_cells if int(recovery_host[position]) == 1 else refused_cells).append(cell)
    nonfinite_positions = {}

    def receipt_floats(name, values):
        bad = tuple(i for i, value in enumerate(values) if not math.isfinite(float(value)))
        if bad:
            nonfinite_positions[name] = bad
        return tuple(float(value) if math.isfinite(float(value))
                     else V6_NONFINITE_DIAGNOSTIC_SENTINEL for value in values)

    initial_receipt = receipt_floats("initial_objective", initial_host)
    final_receipt = receipt_floats("final_objective", final_host)
    identity_receipt = receipt_floats("identity_objective", identity_host)
    intercept_receipt = receipt_floats("intercept_only_objective", intercept_host)
    kkt_receipt = receipt_floats("gradient_kkt_residual", kkt_host)
    required_passed = None
    if required_bool is not None:
        columns = required_bool.tolist()
        required_passed = all(converged_flags[m * query_count + q]
                              for m in range(mask_count) for q in range(query_count) if columns[q])
    profile = BackendProfile(
        backend_id="mechanistic_systems.overlap_factored_calibration.v1",
        capability="singleton_positive_affine_calibration",
        implementation=f"{resolved_statistics_backend}_float64_projected_newton",
        device_type=str(device.type),
        numerical_mode="float64_accumulation_direct_constrained_parameters",
        calls=int(iteration_total), wall_time_s=max(0.0, time.perf_counter() - wall_start),
        peak_hbm_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        processed_contexts=int(mask_count * x64.shape[1]),
        processed_queries=int(mask_count * query_count),
        metadata={
            **({"loss_weighting": "per_image", "weighting": "per_image_population_mean"} if loss_weighting() == "per_image" else {}),
            "terminal_fit_execution": TERMINAL_FIT_EXECUTION_V6,
            "affine_conditioning": "weighted_center_and_rms_exact_reparameterization",
            "kkt_coordinate_system": "conditioned_direct_affine",
            "conditioning_location_abs_max": float(location_abs_max),
            "conditioning_spread_min": float(spread_min),
            "conditioning_spread_max": float(spread_max),
            "raw_conditioning_spread_min": float(raw_spread_min),
            "raw_conditioning_spread_max": float(raw_spread_max),
            "scale_identifiability_tolerance": scale_identifiability_tolerance,
            "source_logit_dtype": str(source_dtype), "source_logit_epsilon": source_epsilon,
            "scale_unidentifiable_component_count": int(unidentifiable_count),
            "separated_component_count": int(separated_count),
            "gpu_hard_gate_failed_cell_count": int(positions.numel()),
            "gpu_hard_gate_failed_gate_counts": failed_gate_counts,
            "gpu_fallback_rescued_cells": rescued_cells,
            "gpu_fallback_fail_closed_cells": refused_cells,
            "gpu_fallback_rescued_cell_count": sum(int(v) == 1 for v in recovery_host),
            "gpu_fallback_fail_closed_cell_count": sum(int(v) == 2 for v in recovery_host),
            "gpu_hard_gate_missing_cell_count": len(missing_positions),
            "gpu_hard_gate_missing_cell_positions": missing_positions,
            "gpu_hard_gate_separated_missing_cell_count": sum(bool(separated_host[i]) for i in missing_positions),
            "missing_cell_rule": V6_MISSING_CELL_RULE,
            "nonfinite_diagnostic_sentinel": V6_NONFINITE_DIAGNOSTIC_SENTINEL,
            "nonfinite_diagnostic_cell_positions": nonfinite_positions,
            "final_statistics_backend": resolved_statistics_backend,
            "identity_anchor_strength_min": float(anchor_min),
            "identity_anchor_strength_max": float(anchor_max),
            "anchor_probability_epsilon": probability_epsilon,
            "regularization_geometry": "bernoulli_predictive_kl_to_identity",
            "newton_active_poll_every": NEWTON_ACTIVE_POLL_EVERY,
            "line_search_alpha_chunk": LINE_SEARCH_ALPHA_CHUNK,
            "row_validity_provided": row_validity_provided, "row_mass_min": float(row_mass_min),
        },
    )
    receipt = CalibrationFitReceipt(
        dependency_hash=factorization.dependency_hash,
        factorization_hash=factorization.factorization_hash,
        component_hashes=tuple(c.component_id for _ in range(mask_count) for c in factorization.components),
        backend=f"{resolved_statistics_backend}_vectorized_float64_projected_newton",
        initial_objective=initial_receipt, final_objective=final_receipt,
        identity_objective=identity_receipt, intercept_only_objective=intercept_receipt,
        iterations=tuple(int(v) for v in iterations_host),
        line_search_steps=tuple(int(v) for v in line_steps_host),
        gradient_kkt_residual=kkt_receipt, termination_reason=tuple(reasons),
        finite=finite_flags, converged=converged_flags,
        backend_profile=profile.to_runtime_dict(), required_passed=required_passed,
    )
    return _finish_post_query_affine_fit(
        scale=scale, offset=offset, spread=spread, location=location,
        x_dtype=torch.float64, factorization=factorization, receipt=receipt,
        input_was_batched=input_was_batched, required_bool=required_bool,
        required_passed=required_passed, require_convergence=require_convergence,
        build_owner=build_owner,
    )


def _v6_failed_gate(*, finite: bool, objective_worsened: bool,
                    failed_line_search: bool, converged: bool) -> str:
    """The single solve's first unmet convergence condition."""
    if not finite:
        return "nonfinite"
    if objective_worsened:
        return "objective_worsened"
    if failed_line_search:
        return "line_search_failed"
    if not converged:
        return "iteration_limit"
    return "none"


def v6_missing_cell_positions(receipt: "CalibrationFitReceipt") -> tuple[int, ...]:
    """Flat ``mask * query + query`` positions of unconverged cells."""

    return tuple(
        position for position, ok in enumerate(receipt.converged) if not bool(ok)
    )


def class_balanced_calibration_weights(binary_targets: Any, class_ids: Any, *,
                                       row_validity: Any | None = None,
                                       population_class_counts: Any | None = None,
                                       weighting: str | None = None) -> Any:
    """Half target mass and equal off-target class masses, rows equal per class.

    Each row-validity mask uses its represented classes. Per-image loss keeps
    population image masses, correcting class-stratified panels when supplied.
    """
    y = torch.as_tensor(binary_targets)
    try:
        classes = torch.as_tensor(class_ids, device=y.device).reshape(-1)
    except (TypeError, ValueError):
        lookup: dict[str, int] = {}
        codes = [lookup.setdefault(str(value), len(lookup)) for value in class_ids]
        classes = torch.tensor(codes, device=y.device, dtype=torch.int64)
    if classes.numel() != y.shape[0]:
        raise ValueError("calibration class IDs differ from the sample axis")
    _, inverse = torch.unique(classes, sorted=True, return_inverse=True)
    count = int(inverse.max().item()) + 1
    validity = (torch.ones((1, y.shape[0]), device=y.device, dtype=torch.float64)
                if row_validity is None else torch.as_tensor(row_validity, device=y.device,
                                                            dtype=torch.float64))
    mode = loss_weighting() if weighting is None else str(weighting)
    if mode not in {"balanced", "per_image"}:
        raise ValueError("loss weighting must be balanced or per_image")
    if mode == "per_image":
        if population_class_counts is None:
            mass = validity / validity.sum(-1, keepdim=True).clamp_min(1e-300)
        else:
            counts_by_class = (dict(population_class_counts) if isinstance(population_class_counts, Mapping)
                               else dict(enumerate(population_class_counts)))
            counts_by_class = {int(k): float(v) for k, v in counts_by_class.items()}
            total = sum(counts_by_class.values())
            if total <= 0 or any(not math.isfinite(v) or v < 0 for v in counts_by_class.values()):
                raise ValueError("population class counts must be finite and nonnegative with positive total")
            observed_classes = set(int(v) for v in torch.unique(classes).detach().cpu().tolist())
            if any(value > 0 and key not in observed_classes for key, value in counts_by_class.items()):
                raise ValueError("class-stratified calibration lacks a population class")
            mass = torch.zeros_like(validity)
            for class_id in torch.unique(classes).detach().cpu().tolist():
                if class_id not in counts_by_class or counts_by_class[class_id] <= 0:
                    raise ValueError("sample class is absent from the population")
                selected = classes == class_id
                class_rows = validity * selected.unsqueeze(0)
                if bool((class_rows.sum(-1) <= 0).any()):
                    raise ValueError("class-stratified calibration lacks class support within a row mask")
                mass += class_rows * (counts_by_class[class_id] / total) / class_rows.sum(-1, keepdim=True).clamp_min(1e-300)
        result = mass.unsqueeze(-1).expand(-1, -1, y.shape[1])
        return result[0] if row_validity is None else result
    result = torch.zeros((validity.shape[0], *y.shape), device=y.device, dtype=torch.float64)
    for query in range(y.shape[1]):
        positive = (y[:, query] > 0.5).to(torch.float64).unsqueeze(0) * validity
        negative = (y[:, query] <= 0.5).to(torch.float64).unsqueeze(0) * validity
        counts = torch.zeros((validity.shape[0], count), device=y.device, dtype=torch.float64)
        counts.scatter_add_(1, inverse.unsqueeze(0).expand(validity.shape[0], -1), negative)
        observed = (counts > 0.0).sum(-1).clamp_min(1)
        per_row = counts.gather(1, inverse.unsqueeze(0).expand(validity.shape[0], -1))
        result[:, :, query] = (0.5 * positive / positive.sum(-1, keepdim=True).clamp_min(1e-300)
                               + 0.5 * negative / observed.unsqueeze(1) / per_row.clamp_min(1e-300))
    return result[0] if row_validity is None else result


def _fit_batched_firth_affine(x, target, *, factorization, sample_weights, class_ids,
                            fixed_offsets, row_validity, minimum_scale, initial_scale,
                            initial_offset, max_iterations, tolerance, hessian_ridge,
                            objective_tolerance, reference_tolerance, require_convergence,
                            required_query_mask, input_was_batched, wall_start,
                            firth_convergence="absolute", metric_loss_weighting=None):
    from .device_calibration_fit import CELL_FIELDS, fit_cells

    masks, rows, queries = map(int, x.shape)
    if not math.isfinite(float(minimum_scale)) or float(minimum_scale) < 0.0:
        raise ValueError("minimum calibration scale must be finite and nonnegative")
    validity = (torch.ones((masks, rows), dtype=torch.float64, device=x.device)
                if row_validity is None else torch.as_tensor(row_validity, device=x.device,
                                                            dtype=torch.float64))
    if tuple(validity.shape) != (masks, rows) or not bool(((validity == 0) | (validity == 1)).all().item()):
        raise ValueError("row validity must be binary with shape [mask, sample]")
    effective_loss_weighting = loss_weighting() if metric_loss_weighting is None else metric_loss_weighting
    if effective_loss_weighting not in {"balanced", "per_image"}:
        raise ValueError(f"unsupported metric loss weighting {effective_loss_weighting!r}")
    if class_ids is not None and not (effective_loss_weighting == "per_image" and sample_weights is not None):
        supplied = class_balanced_calibration_weights(target, class_ids, row_validity=validity)
    elif sample_weights is not None:
        supplied = torch.as_tensor(sample_weights, dtype=torch.float64, device=x.device)
        if supplied.ndim == 1:
            supplied = supplied.unsqueeze(-1)
        if supplied.ndim == 2:
            supplied = supplied.unsqueeze(0)
        try:
            supplied = supplied.expand(masks, rows, queries)
        except RuntimeError as error:
            raise ValueError("metric sample weights differ from mask/sample/query axes") from error
    else:
        raise ValueError("Firth calibration requires metric sample weights or class IDs")
    if not bool(torch.isfinite(supplied).all().item()) or bool((supplied < 0.0).any().item()):
        raise ValueError("metric sample weights must be finite and nonnegative")

    def cells(value, default):
        tensor = torch.as_tensor(default if value is None else value, device=x.device,
                                 dtype=torch.float64)
        try:
            return tensor.expand(masks, queries).reshape(-1).clone()
        except RuntimeError as error:
            raise ValueError("calibration initialization differs from mask/query axes") from error

    offsets = None
    if fixed_offsets is not None:
        offsets = torch.as_tensor(fixed_offsets, dtype=torch.float64, device=x.device)
        try:
            offsets = offsets.expand(masks, rows, queries).permute(0, 2, 1).reshape(-1, rows)
        except RuntimeError as error:
            raise ValueError("fixed offsets differ from mask/sample/query axes") from error
    result = fit_cells(x.permute(0, 2, 1).reshape(-1, rows),
        target.T.unsqueeze(0).expand(masks, -1, -1).reshape(-1, rows),
        validity.unsqueeze(1).expand(-1, queries, -1).reshape(-1, rows),
        torch.zeros(masks * queries, dtype=torch.float64, device=x.device),
        initial_scale=cells(initial_scale, 1.0), initial_offset=cells(initial_offset, 0.0),
        enabled=torch.ones(masks * queries, dtype=torch.bool, device=x.device),
        weighting="design_weighted", minimum_scale=minimum_scale, iterations=max_iterations,
        tolerance=tolerance, hessian_ridge=hessian_ridge, objective_tolerance=objective_tolerance,
        reference_tolerance=reference_tolerance, probability_epsilon=1e-12,
        source_epsilon=torch.finfo(x.dtype).eps, calibration_penalty="firth",
        sample_weights=supplied.permute(0, 2, 1).reshape(-1, rows), fixed_offsets=offsets,
        firth_convergence=firth_convergence)
    diagnostic_rows = result.diagnostics.detach().cpu().tolist()
    diagnostics = dict(zip(CELL_FIELDS, diagnostic_rows))
    size = tuple(float(value) for value in result.effective_sample_size.detach().cpu().tolist())
    valid = result.inputs_valid.detach().cpu().tolist()
    finite = tuple(bool(a) and bool(b) for a, b in zip(diagnostics["finite"], valid))
    converged = tuple(bool(a) and bool(b) for a, b in zip(diagnostics["converged"], valid))
    success_reason = ("fisher_standardized_kkt_tolerance"
                      if firth_convergence == "fisher_standardized" else "kkt_tolerance")
    reasons = tuple(success_reason if done else "invalid_inputs" if not okay else
                    "line_search_failed" if failed else "iteration_limit"
                    for done, okay, failed in zip(converged, valid, diagnostics["failed_line_search"]))
    required = None
    required_passed = None
    if required_query_mask is not None:
        required = torch.as_tensor(required_query_mask, dtype=torch.bool).reshape(-1)
        if required.numel() != queries:
            raise ValueError("required query mask differs from the query axis")
        required_passed = all(converged[m * queries + q] for m in range(masks)
                              for q in range(queries) if bool(required[q]))
    profile = BackendProfile(
        backend_id="mechanistic_systems.overlap_factored_calibration.firth.v1",
        capability="singleton_positive_affine_calibration",
        implementation="torch_float64_firth_fisher_scoring", device_type=str(x.device.type),
        numerical_mode="normalized_metric_weighted_firth_positive_affine",
        calls=sum(int(value) for value in diagnostics["iterations"]),
        wall_time_s=max(0.0, time.perf_counter() - wall_start),
        processed_contexts=masks * rows, processed_queries=masks * queries,
        metadata={"calibration_penalty": "firth", "effective_sample_size": size,
                  "regularization_geometry": "jeffreys_information_log_determinant",
                  "weighting": "per_image_population_mean" if effective_loss_weighting == "per_image" else "class_balanced_metric",
                  **({"loss_weighting": "per_image"} if effective_loss_weighting == "per_image" else {}), "selection_rule": "single_firth_fit",
                  "kkt_coordinate_system": ("fisher_standardized_conditioned_modified_score"
                                            if firth_convergence == "fisher_standardized"
                                            else "conditioned_modified_score"),
                  "scale_unidentifiable_component_count": sum(not bool(v) for v in diagnostics["scale_identifiable"]),
                  "constant_readout_rule": "intercept_only_firth_quotient",
                  "fixed_offsets_present": fixed_offsets is not None})

    def floats(name):
        return tuple(float(v) if math.isfinite(float(v)) else V6_NONFINITE_DIAGNOSTIC_SENTINEL
                     for v in diagnostics[name])

    receipt = CalibrationFitReceipt(
        dependency_hash=factorization.dependency_hash, factorization_hash=factorization.factorization_hash,
        component_hashes=tuple(c.component_id for _ in range(masks) for c in factorization.components),
        backend="torch_vectorized_float64_firth_fisher_scoring",
        initial_objective=floats("initial_objective"), final_objective=floats("final_objective"),
        identity_objective=floats("identity_objective"), intercept_only_objective=floats("intercept_only_objective"),
        iterations=tuple(int(v) for v in diagnostics["iterations"]),
        line_search_steps=tuple(int(v) for v in diagnostics["line_search_steps"]),
        gradient_kkt_residual=floats("gradient_kkt_residual"), termination_reason=reasons,
        finite=finite, converged=converged, backend_profile=profile.to_runtime_dict(),
        required_passed=required_passed)
    receipt.validate()
    if require_convergence and not (receipt.passed if required is None else required_passed):
        raise CalibrationConvergenceError(
            "Firth affine calibration failed convergence: " + repr(reasons), receipt)
    return FactoredCalibrationFit(result.scale.reshape(masks, queries).detach(),
        result.offset.reshape(masks, queries).detach(), factorization, receipt, input_was_batched,
        family="firth_positive_affine_v1")


def _fit_batched_post_query_affine_core(
    values: Any,
    binary_targets: Any,
    *,
    factorization: CalibrationFactorization,
    weighting: str = "pos_weighted",
    sample_weights: Any | None = None,
    minimum_scale: float = 1e-8,
    initial_scale: Any | None = None,
    initial_offset: Any | None = None,
    max_iterations: int = 80,
    tolerance: float = 1e-8,
    hessian_ridge: float = 1e-10,
    objective_tolerance: float = 1e-10,
    reference_tolerance: float = 1e-8,
    statistics_backend: str = "auto",
    require_convergence: bool = True,
    source_logit_dtype: Any | None = None,
    identity_anchor_strength: Any | None = None,
    calibration_penalty: str = "identity_anchor",
    class_ids: Any | None = None,
    fixed_offsets: Any | None = None,
    firth_convergence: str = "absolute",
    metric_loss_weighting: str | None = None,
    anchor_probability_epsilon: float = 1e-12,
    row_validity: Any | None = None,
    required_query_mask: Any | None = None,
    reference_audit_policy: str = GPU_HARD_GATES_AUDIT_POLICY,
    audit_seed: int | None = None,
    _differential_context_ids: Sequence[Any] | None = None,
) -> tuple[FactoredCalibrationFit, PositiveAffineDifferentialOwnerReceipt | None]:
    """Fit all independent positive affine queries with guarded Newton steps.

    The optimization is convex in direct ``(scale, offset)`` coordinates. The
    scale constraint uses projected active-set steps, preserving convexity and
    avoiding a non-linear softplus parameterization of the scale.
    """

    _require_torch()
    wall_start = time.perf_counter()
    if calibration_penalty not in {"identity_anchor", "firth"}:
        raise ValueError(f"unsupported calibration penalty {calibration_penalty!r}")
    if not factorization.is_fully_factorized:
        raise ValueError("post-query affine solver requires singleton calibration components")
    x, input_was_batched = _as_batched_values(values)
    if not x.is_floating_point():
        x = x.to(torch.float32)
    source_dtype = x.dtype if source_logit_dtype is None else source_logit_dtype
    try:
        source_epsilon = float(torch.finfo(source_dtype).eps)
    except (TypeError, ValueError) as error:
        raise ValueError("source logit dtype must be floating point") from error
    device = x.device
    x64_raw = x.detach().to(device=device, dtype=torch.float64)
    target = torch.as_tensor(binary_targets, device=device, dtype=torch.float64)
    if target.ndim != 2 or tuple(target.shape) != (
        int(x64_raw.shape[1]),
        int(x64_raw.shape[2]),
    ):
        raise ValueError("binary calibration targets must have shape [sample,query]")
    if not bool(torch.isfinite(x64_raw).all().item()) or not bool(
        torch.isfinite(target).all().item()
    ):
        raise ValueError("calibration inputs and targets must be finite")
    if not bool(((target == 0.0) | (target == 1.0)).all().item()):
        raise ValueError("binary calibration targets must contain only zero and one")
    mask_count, _, query_count = map(int, x64_raw.shape)
    expected_queries = sum(len(component.query_axis_ids) for component in factorization.components)
    if query_count != expected_queries:
        raise ValueError("factorization query axis differs from calibration values")
    if row_validity is not None and int(torch.as_tensor(row_validity).shape[0]) != mask_count:
        raise ValueError("row validity mask axis differs from calibration values")
    if calibration_penalty == "firth":
        if _differential_context_ids is not None:
            raise CalibrationDifferentialOwnerError(
                "firth_differential_owner_unavailable",
                "the predictive-KL differential owner does not describe a Firth fit")
        return _fit_batched_firth_affine(
            x64_raw, target, factorization=factorization, sample_weights=sample_weights,
            class_ids=class_ids, fixed_offsets=fixed_offsets, row_validity=row_validity,
            minimum_scale=minimum_scale, initial_scale=initial_scale, initial_offset=initial_offset,
            max_iterations=max_iterations, tolerance=tolerance, hessian_ridge=hessian_ridge,
            objective_tolerance=objective_tolerance, reference_tolerance=reference_tolerance,
            require_convergence=require_convergence, required_query_mask=required_query_mask,
            input_was_batched=input_was_batched, wall_start=wall_start,
            firth_convergence=firth_convergence, metric_loss_weighting=metric_loss_weighting), None
    weights = _row_weights(
        target,
        weighting=weighting,
        sample_weights=sample_weights,
        dtype=torch.float64,
        device=device,
        row_validity=row_validity,
    )
    # Normalize to a broadcastable [1|M, N, K] row-weight view once; every
    # sample-axis reduction below is dim=1 against this view.
    row_weights = weights.unsqueeze(0) if weights.ndim == 2 else weights
    weighted_positive = (row_weights * target.unsqueeze(0)).sum(dim=1)
    weighted_negative = (row_weights * (1.0 - target.unsqueeze(0))).sum(dim=1)
    if bool(((weighted_positive <= 0.0) | (weighted_negative <= 0.0)).any().item()):
        raise ValueError(
            "every calibration query requires positive-weight positive and negative support"
        )
    min_scale = float(minimum_scale)
    if not math.isfinite(min_scale) or min_scale < 0.0:
        raise ValueError("minimum calibration scale must be finite and nonnegative")

    def _parameter(value: Any | None, default: float) -> Any:
        if value is None:
            return torch.full((mask_count, query_count), default, device=device, dtype=torch.float64)
        tensor = torch.as_tensor(value, device=device, dtype=torch.float64)
        if tensor.ndim == 0:
            tensor = tensor.reshape(1, 1).expand(mask_count, query_count)
        elif tensor.ndim == 1:
            tensor = tensor.unsqueeze(0).expand(mask_count, -1)
        if tuple(tensor.shape) != (mask_count, query_count):
            raise ValueError("calibration initialization differs from mask/query axes")
        return tensor.detach().clone()

    raw_initial_scale = _parameter(initial_scale, 1.0).clamp_min(min_scale)
    raw_initial_offset = _parameter(initial_offset, 0.0)
    anchor_strength = _parameter(identity_anchor_strength, 0.0)
    if not bool(torch.isfinite(anchor_strength).all().item()) or bool(
        (anchor_strength < 0.0).any().item()
    ):
        raise ValueError("identity-anchor strengths must be finite and nonnegative")
    probability_epsilon = float(anchor_probability_epsilon)
    if not 0.0 < probability_epsilon < 0.5:
        raise ValueError("anchor probability epsilon must lie in (0,0.5)")

    # The Bernoulli predictive-KL anchor is, up to an immutable constant, the
    # same canonical proper loss evaluated at a soft target p0=sigmoid(z).
    # This preserves the existing Newton sufficient statistics while making
    # the objective coercive under separation.  The anchor uses the same row
    # measure as the supervised loss, so lambda is an equivalent-sample-mass
    # ratio rather than a coordinate-dependent parameter penalty.
    identity_probability = torch.sigmoid(x64_raw).clamp(
        min=probability_epsilon,
        max=1.0 - probability_epsilon,
    )
    anchor = anchor_strength.unsqueeze(1)
    optimization_target = (target.unsqueeze(0) + anchor * identity_probability) / (
        1.0 + anchor
    )
    optimization_weights = row_weights * (1.0 + anchor)
    anchor_objective_constant = (
        anchor
        * row_weights
        * (F.softplus(x64_raw) - identity_probability * x64_raw)
    ).sum(dim=1) / float(x64_raw.shape[1])

    # Solve in an exactly equivalent affine chart. Candidate pruning can leave
    # logits with a large common offset and very little residual spread. The
    # raw (a, c) Hessian is then numerically singular even when the convex MLE
    # is finite. With u=(x-mu)/sigma, beta=a*sigma and gamma=c+a*mu,
    # beta*u+gamma == a*x+c and beta >= minimum_scale*sigma. This is pure
    # preconditioning: it neither regularizes nor caps a separating solution.
    weight_sum = row_weights.sum(dim=1).clamp_min(1e-30)
    location = (x64_raw * row_weights).sum(dim=1) / weight_sum
    centered = x64_raw - location.unsqueeze(1)
    active_rows = row_weights > 0.0
    positive_min = torch.where(
        (target.unsqueeze(0) > 0.5) & active_rows,
        x64_raw,
        torch.full_like(x64_raw, float("inf")),
    ).amin(dim=1)
    negative_max = torch.where(
        (target.unsqueeze(0) <= 0.5) & active_rows,
        x64_raw,
        torch.full_like(x64_raw, float("-inf")),
    ).amax(dim=1)
    active_min = torch.where(
        active_rows,
        x64_raw,
        torch.full_like(x64_raw, float("inf")),
    ).amin(dim=1)
    active_max = torch.where(
        active_rows,
        x64_raw,
        torch.full_like(x64_raw, float("-inf")),
    ).amax(dim=1)
    raw_spread = torch.sqrt(
        (centered.square() * row_weights).sum(dim=1) / weight_sum
    )
    # Below the declared solver/reference resolution, affine scale is not an
    # identifiable scientific parameter: ``a*x+c`` has only one estimable
    # intercept coordinate.  Without this quotient, CUDA-level variation in a
    # mathematically constant fully-pruned output can look perfectly separated
    # and drive an unbounded, meaningless scale.  The threshold is the existing
    # numerical contract, not a data-dependent overlap heuristic.
    scale_identifiability_tolerance = max(
        float(tolerance),
        float(reference_tolerance),
    )
    scale_identifiable = raw_spread > scale_identifiability_tolerance
    separated = (
        (negative_max <= positive_min) & scale_identifiable & (anchor_strength <= 0.0)
    )
    spread = torch.where(
        scale_identifiable,
        raw_spread,
        torch.ones_like(raw_spread),
    )
    x64 = centered / spread.unsqueeze(1)
    minimum_conditioned_scale = spread * min_scale
    initial_conditioned_scale = raw_initial_scale * spread
    initial_conditioned_offset = raw_initial_offset + raw_initial_scale * location
    scale = initial_conditioned_scale.clone()
    offset = initial_conditioned_offset.clone()
    resolved_statistics_backend = str(statistics_backend)
    if resolved_statistics_backend.strip().lower() == "auto":
        # Canonical torch float64 reductions under the v6 gates.
        resolved_statistics_backend = "torch"

    def stats(current_scale: Any, current_offset: Any) -> tuple[Any, ...]:
        nonlocal resolved_statistics_backend
        reduced = calibration_statistics(
            x64,
            optimization_target,
            optimization_weights,
            current_scale,
            current_offset,
            backend=resolved_statistics_backend,
        )
        resolved_statistics_backend = (
            "triton" if reduced.backend.startswith("triton") else "torch"
        )
        values = list(reduced.values())
        values[0] = values[0] - anchor_objective_constant
        return tuple(values)

    def stats_objective(current_scale: Any, current_offset: Any) -> Any:
        """``stats(...)[0]`` (v6): the torch reducer's objective ops alone."""

        if resolved_statistics_backend != "torch":
            return stats(current_scale, current_offset)[0]
        return (
            _torch_statistics_objective(
                x64,
                optimization_target,
                optimization_weights,
                current_scale,
                current_offset,
            )
            - anchor_objective_constant
        )

    row_mass_min_v6 = None
    # The receipt's row-mass scalar, taken now so the row weights can go.
    row_mass_min_v6 = (
        (row_weights > 0.0).to(torch.float64).mean(dim=1).min()
        if row_validity is not None
        else torch.ones((), dtype=torch.float64)
    )
    if _differential_context_ids is None:
        # Release the [mask, sample, query] setup tensors
        # the solve never reads again (the raw float64 logits, the
        # identity probabilities, the centered logits, the row weights
        # and the active-row mask).  The solve reads only the conditioned
        # logits, the anchored target and the anchored weights, so every
        # value is unchanged; the live float64 lane tensors of a call drop
        # from about 14 to about 10 at the peak of a statistics pass,
        # which is what bounds the logical chunks per physical call.
        x64_raw = identity_probability = centered = None
        weights = row_weights = active_rows = None
    objective_of = stats_objective

    stacked_line_search_inputs: dict[int, tuple[Any, Any, Any]] = {}

    def stats_objective_stack(scale_stack: Any, offset_stack: Any) -> Any:
        """Objective of ``stats`` for a [width, mask, query] candidate stack.

        One reducer call evaluates a whole chunk of line-search alpha
        candidates; the anchor constant broadcasts over the stack axis
        with the same elementwise IEEE-exact arithmetic as ``stats``.
        """

        nonlocal resolved_statistics_backend
        width = int(scale_stack.shape[0])
        if resolved_statistics_backend == "torch":
            # v6: each candidate's objective alone (the torch reducer's
            # objective ops), without materializing the width-fold copies
            # of x, target and weights (3 x width x [mask, sample, query]
            # float64, ~7.9 GB for a terminal selection call at width 8) or
            # computing the five unused reductions.  Same ladder, same
            # Armijo predicate, same first-accept rule.
            return torch.stack(
                [
                    _torch_statistics_objective(
                        x64,
                        optimization_target,
                        optimization_weights,
                        scale_stack[index],
                        offset_stack[index],
                    )
                    for index in range(width)
                ]
            ) - anchor_objective_constant.unsqueeze(0)
        sample_count = int(x64.shape[1])
        cached = stacked_line_search_inputs.get(width)
        if cached is None:
            cached = tuple(
                tensor.unsqueeze(0)
                .expand(width, -1, -1, -1)
                .reshape(width * mask_count, sample_count, query_count)
                for tensor in (x64, optimization_target, optimization_weights)
            )
            stacked_line_search_inputs[width] = cached
        x_stack, target_stack, weight_stack = cached
        reduced = calibration_statistics(
            x_stack,
            target_stack,
            weight_stack,
            scale_stack.reshape(width * mask_count, query_count),
            offset_stack.reshape(width * mask_count, query_count),
            backend=resolved_statistics_backend,
        )
        resolved_statistics_backend = (
            "triton" if reduced.backend.startswith("triton") else "torch"
        )
        objective_stack = reduced.values()[0].reshape(
            width, mask_count, query_count
        )
        return objective_stack - anchor_objective_constant.unsqueeze(0)

    initial_objective = objective_of(scale, offset)
    identity_objective = objective_of(
        spread,
        location,
    )
    intercept_null = torch.log(
        weighted_positive.clamp_min(1e-30) / weighted_negative.clamp_min(1e-30)
    )
    if int(intercept_null.shape[0]) == 1:
        intercept_null = intercept_null.expand(mask_count, -1)
    intercept_only_objective = objective_of(
        torch.zeros_like(scale),
        intercept_null,
    )

    (
        scale,
        offset,
        converged,
        failed_line_search,
        iterations,
        line_steps,
    ) = _guarded_projected_newton(
        stats=stats,
        stats_objective_stack=stats_objective_stack,
        scale=scale,
        offset=offset,
        minimum_conditioned_scale=minimum_conditioned_scale,
        max_iterations=max_iterations,
        tolerance=tolerance,
        hessian_ridge=hessian_ridge,
        objective_tolerance=objective_tolerance,
        proposal_objective_fn=stats_objective,
    )
    (
        final_objective,
        grad_s,
        grad_o,
        final_kkt,
        converged,
        finite,
        objective_worsened,
    ) = _newton_final_gates(
        stats=stats,
        scale=scale,
        offset=offset,
        minimum_conditioned_scale=minimum_conditioned_scale,
        tolerance=tolerance,
        objective_tolerance=objective_tolerance,
        initial_objective=initial_objective,
        converged=converged,
        failed_line_search=failed_line_search,
        separated=separated,
    )

    build_owner: Callable[[FactoredCalibrationFit], Any] | None = None
    if _differential_context_ids is not None:

        def _owner(fit: FactoredCalibrationFit) -> Any:
            return _build_positive_affine_owner(
                x64_raw=x64_raw, target=target, weights=weights,
                optimization_target=optimization_target,
                optimization_weights=optimization_weights,
                anchor_strength=anchor_strength,
                identity_probability=identity_probability,
                location=location, raw_spread=raw_spread, spread=spread,
                minimum_scale=min_scale,
                minimum_conditioned_scale=minimum_conditioned_scale,
                scale_identifiability_tolerance=scale_identifiability_tolerance,
                source_dtype=source_dtype, source_epsilon=source_epsilon,
                probability_epsilon=probability_epsilon, tolerance=tolerance,
                factorization=factorization, fit=fit,
                context_ids=_differential_context_ids,
            )

        build_owner = _owner

    return _finish_with_device_hard_gates_v6(
        x_dtype=x.dtype,
        x64=x64,
        optimization_target=optimization_target, optimization_weights=optimization_weights,
        anchor_objective_constant=anchor_objective_constant,
        initial_conditioned_scale=initial_conditioned_scale,
        initial_conditioned_offset=initial_conditioned_offset,
        max_iterations=max_iterations, hessian_ridge=hessian_ridge,
        objective_tolerance=objective_tolerance,
        scale=scale,
        offset=offset,
        final_objective=final_objective,
        final_kkt=final_kkt,
        finite=finite,
        converged=converged,
        objective_worsened=objective_worsened,
        failed_line_search=failed_line_search,
        iterations=iterations,
        line_steps=line_steps,
        separated=separated,
        scale_identifiable=scale_identifiable,
        minimum_conditioned_scale=minimum_conditioned_scale,
        initial_objective=initial_objective,
        identity_objective=identity_objective,
        intercept_only_objective=intercept_only_objective,
        location=location,
        spread=spread,
        raw_spread=raw_spread,
        anchor_strength=anchor_strength,
        row_mass_min=row_mass_min_v6,
        row_validity_provided=row_validity is not None,
        tolerance=tolerance,
        scale_identifiability_tolerance=scale_identifiability_tolerance,
        source_dtype=source_dtype,
        source_epsilon=source_epsilon,
        probability_epsilon=probability_epsilon,
        resolved_statistics_backend=resolved_statistics_backend,
        required_query_mask=required_query_mask,
        factorization=factorization,
        input_was_batched=input_was_batched,
        require_convergence=require_convergence,
        wall_start=wall_start,
        build_owner=build_owner,
    )

def fit_batched_post_query_affine(
    values: Any,
    binary_targets: Any,
    *,
    factorization: CalibrationFactorization,
    weighting: str = "pos_weighted",
    sample_weights: Any | None = None,
    minimum_scale: float = 1e-8,
    initial_scale: Any | None = None,
    initial_offset: Any | None = None,
    max_iterations: int = 80,
    tolerance: float = 1e-8,
    hessian_ridge: float = 1e-10,
    objective_tolerance: float = 1e-10,
    reference_tolerance: float = 1e-8,
    statistics_backend: str = "auto",
    require_convergence: bool = True,
    source_logit_dtype: Any | None = None,
    identity_anchor_strength: Any | None = None,
    calibration_penalty: str = "identity_anchor",
    class_ids: Any | None = None,
    fixed_offsets: Any | None = None,
    firth_convergence: str = "absolute",
    metric_loss_weighting: str | None = None,
    anchor_probability_epsilon: float = 1e-12,
    row_validity: Any | None = None,
    required_query_mask: Any | None = None,
    reference_audit_policy: str = GPU_HARD_GATES_AUDIT_POLICY,
    audit_seed: int | None = None,
) -> FactoredCalibrationFit:
    """Fit all independent positive affine queries with guarded Newton steps."""

    return _fit_batched_post_query_affine_core(
        values, binary_targets, factorization=factorization, weighting=weighting,
        sample_weights=sample_weights, minimum_scale=minimum_scale,
        initial_scale=initial_scale, initial_offset=initial_offset,
        max_iterations=max_iterations, tolerance=tolerance,
        hessian_ridge=hessian_ridge, objective_tolerance=objective_tolerance,
        reference_tolerance=reference_tolerance,
        statistics_backend=statistics_backend, require_convergence=require_convergence,
        source_logit_dtype=source_logit_dtype,
        identity_anchor_strength=identity_anchor_strength,
        calibration_penalty=calibration_penalty, class_ids=class_ids,
        fixed_offsets=fixed_offsets,
        firth_convergence=firth_convergence,
        metric_loss_weighting=metric_loss_weighting,
        anchor_probability_epsilon=anchor_probability_epsilon,
        row_validity=row_validity,
        required_query_mask=required_query_mask,
        reference_audit_policy=reference_audit_policy,
        audit_seed=audit_seed,
    )[0]


def fit_batched_post_query_affine_with_differential_owner(
    values: Any,
    binary_targets: Any,
    *,
    factorization: CalibrationFactorization,
    context_ids: Sequence[Any],
    **fit_options: Any,
) -> PositiveAffineOwnedFit:
    """Fit through the canonical owner and expose its stable differential face."""

    fit, owner = _fit_batched_post_query_affine_core(
        values, binary_targets, factorization=factorization,
        _differential_context_ids=context_ids, **fit_options,
    )
    if owner is None:  # pragma: no cover - construction invariant.
        raise RuntimeError("positive-affine differential owner receipt was not emitted")
    return PositiveAffineOwnedFit(fit=fit, differential_owner=owner)


__all__ = [
    "CalibrationDifferentialOwnerError",
    "CalibrationConvergenceError",
    "CalibrationFailClosedError",
    "CalibrationFitReceipt",
    "V6_MISSING_CELL_RULE",
    "V6_NONFINITE_DIAGNOSTIC_SENTINEL",
    "FactoredCalibrationFit",
    "PositiveAffineComponentFace",
    "PositiveAffineDifferentialOwnerReceipt",
    "PositiveAffineOwnedFit",
    "calibration_tensor_content_hash",
    "fit_batched_post_query_affine",
    "fit_batched_post_query_affine_with_differential_owner",
    "fit_shared_temperature_multiclass",
    "v6_missing_cell_positions",
]
