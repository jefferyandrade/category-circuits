"""Factor-compiled calibration for declared query problems."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from contracts.hashing import stable_digest
from mechanistic_systems.contracts import (
    CANONICAL_CALIBRATION_ANCHOR_PROBABILITY_EPSILON,
    CANONICAL_CALIBRATION_HESSIAN_RIDGE,
    CANONICAL_CALIBRATION_IDENTITY_ANCHOR_STRENGTHS,
    CANONICAL_CALIBRATION_KKT_TOLERANCE,
    CANONICAL_CALIBRATION_MAX_ITERATIONS,
    CANONICAL_CALIBRATION_OBJECTIVE_TOLERANCE,
    CANONICAL_CALIBRATION_REFERENCE_TOLERANCE,
    CANONICAL_CALIBRATION_SELECTION_BLOCKS,
    CalibrationProblemSpec,
    PROPER_SCORE_SHRINKAGE_AFFINE_V2,
    QueryOperatorSpec,
    ReadoutProjectionSpec,
)

from packages.numeric_backends.calibration import (
    fit_batched_shared_readout_diagonal_affine_calibration,
    fit_shared_readout_diagonal_affine_calibration,
)
from .calibration_problem import CalibrationFactorization, CalibrationProblemCompiler
from .calibration_runtime import (
    CalibrationExecutionContext,
    CrossFitLayout,
    RuntimeExecutionProfile,
)
from .contracts import BackendProfile
from .factored_calibration import (
    GPU_HARD_GATES_AUDIT_POLICY,
    CalibrationConvergenceError,
    CalibrationFitReceipt,
    FactoredCalibrationFit,
    fit_batched_post_query_affine,
    fit_shared_temperature_multiclass,
)
from .sampled_path_v1_calibration import (
    SAMPLED_PATH_V1_CALIBRATION_ALGORITHM,
    fit_sampled_path_v1_coordinate_affine,
)
from .proper_score_shrinkage import (
    ProperScoreShrinkageFit,
    _fit_proper_score_shrinkage_affine_core,
    fit_proper_score_shrinkage_affine,
)

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - import smoke without Torch.
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


# The fold identity the provider declares on every terminal category fit
# (``NativeTorchCalibratedBCEOracleProvider._fit_terminal_category_calibrations``
# and the per-category terminal structure it serves).  It scopes the v6
# terminal-fit execution to exactly those problems.
TERMINAL_CATEGORY_STRUCTURE_FOLD_ID = "terminal_category_structure_v1"


# The REGISTERED missing-category thresholds of a
# terminal category calibration bank.  A category whose ordinary one-vs-rest
# fit has a selection or deployment cell that failed its device hard gates
# after the device fallback is MISSING (never substituted, never reported
# converged).  The bank aborts only above its limit:
#
# * the DENSE reference bank: 0.  Every final-mask bank is read against it
#   (the dense convention, Y_kj = C_j(M_k) - C_j(dense), and the dense
#   head's bank-rule affine behind the dense pair similarity s_kj), so one
#   missing dense category j would remove j from every arm of every dyadic
#   slot at once.  It is fitted once per model, so
#   a failure is diagnosed and rerun rather than carried.
# * a FINAL-mask bank: at most floor(0.5% x categories), i.e. 5 of 1,000.
#   The analyses' infrastructure-missing rules absorb a handful of missing
#   outcomes (mean slots: the supremum p over the logical bounds, exact
#   vertex enumeration up to 20 missing values; regression and dyadic slots:
#   drop the category from both axes and count, p = 1 above 20 missing
#   categories).  On a convex,
#   anchor-regularized solve that already had a 4x device re-run, more than
#   5 missing categories in one bank is a systemic numerical defect of that
#   arm, so the bank is degraded as a whole (typed, receipted with every
#   missing category), never partially trusted.
#
# The substituted functional is not device-gated (its fixed-offset fitter
# reports convergence) and keeps its v5 carriage rule.
# These limits are provisional until heavily lesioned acceptance masks
# measure the missing-category rate. Search fits never carry missing cells.
TERMINAL_BANK_MISSING_CELL_POLICY_ID = "terminal_bank_missing_category_threshold_v6"
TERMINAL_BANK_MISSING_CELL_POLICY_STATUS = "PROVISIONAL"
DENSE_REFERENCE_BANK_MAXIMUM_MISSING_CATEGORIES = 0
FINAL_MASK_BANK_MAXIMUM_MISSING_CATEGORY_FRACTION = 0.005


def terminal_bank_missing_category_limit(
    *, dense_reference_bank: bool, category_count: int
) -> int:
    """The provisional missing-category limit of one terminal bank."""

    count = int(category_count)
    if count <= 0:
        raise ValueError("a terminal bank has at least one category")
    if bool(dense_reference_bank):
        return int(DENSE_REFERENCE_BANK_MAXIMUM_MISSING_CATEGORIES)
    return int(
        math.floor(FINAL_MASK_BANK_MAXIMUM_MISSING_CATEGORY_FRACTION * count + 1e-9)
    )


class TerminalBankMissingCellThresholdError(RuntimeError):
    """A terminal bank has more missing categories than its provisional limit.

    ``unready_class_ids_by_functional`` uses the name the pilot's degradation
    receipt reads, so a degraded final-mask bank names every missing
    category; ``receipt`` is the full threshold receipt.
    """

    def __init__(self, message: str, *, receipt: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.receipt = dict(receipt)
        self.unready_class_ids_by_functional = {
            str(functional): tuple(int(value) for value in class_ids)
            for functional, class_ids in dict(
                receipt.get("missing_class_ids_by_functional", {})
            ).items()
        }


def terminal_bank_missing_cell_receipt(
    *,
    dense_reference_bank: bool,
    category_count: int,
    missing_class_ids_by_functional: Mapping[str, Sequence[int]],
    gated_functionals: Sequence[str] = ("ordinary_ovr",),
    reasons_by_class: Mapping[int, Sequence[str]] | None = None,
    carriage_supported: bool = True,
) -> dict[str, Any]:
    """Apply the threshold to a bank's missing categories; raise above it.

    Only ``gated_functionals`` (the device-gated fits) count toward the
    limit; every functional's missing ids are recorded.  Above the limit the
    typed ``TerminalBankMissingCellThresholdError`` is raised, whatever the
    carriage.  Within the limit the outcome is ``carried_missing_categories``
    (the bank carries them as unavailable NaN columns) when
    ``carriage_supported``; otherwise (the bank module cannot carry
    an unavailable ORDINARY column) it is
    ``strict_refusal_carriage_unsupported``: nothing is raised here, and the
    caller builds the bank strictly, so it refuses an unready ordinary state
    exactly as a strict bank does (``CalibrationBankUnreadyStatesError``, the
    same message and ids), with this receipt attached.
    """

    limit = terminal_bank_missing_category_limit(
        dense_reference_bank=dense_reference_bank, category_count=category_count
    )
    by_functional = {
        str(functional): sorted({int(value) for value in class_ids})
        for functional, class_ids in missing_class_ids_by_functional.items()
    }
    counted = sorted(
        {
            class_id
            for functional in gated_functionals
            for class_id in by_functional.get(str(functional), ())
        }
    )
    within = len(counted) <= limit
    carried = bool(counted) and within and bool(carriage_supported)
    receipt = {
        "policy": TERMINAL_BANK_MISSING_CELL_POLICY_ID,
        "policy_status": TERMINAL_BANK_MISSING_CELL_POLICY_STATUS,
        "search_fit_missing_policy": "never_missing",
        "bank_role": "dense_reference" if dense_reference_bank else "final_mask",
        "category_count": int(category_count),
        "limit": int(limit),
        "limit_rule": (
            "dense_reference_bank_complete"
            if dense_reference_bank
            else f"floor({FINAL_MASK_BANK_MAXIMUM_MISSING_CATEGORY_FRACTION!r}*categories)"
        ),
        "gated_functionals": [str(value) for value in gated_functionals],
        "missing_count": int(len(counted)),
        "missing_rate": float(len(counted)) / float(category_count),
        "missing_class_ids": counted,
        "missing_class_ids_by_functional": by_functional,
        "missing_reasons_by_class": {
            str(class_id): sorted({str(reason) for reason in reasons})
            for class_id, reasons in dict(reasons_by_class or {}).items()
        },
        "carriage_supported": bool(carriage_supported),
        "within_limit": bool(within),
        "carried": bool(carried),
        "outcome": (
            "complete"
            if not counted
            else "aborted_above_limit"
            if not within
            else "carried_missing_categories"
            if carried
            else "strict_refusal_carriage_unsupported"
        ),
        "missing_class": "infrastructure_missing_numerical_fit_failure",
    }
    if not within:
        raise TerminalBankMissingCellThresholdError(
            f"terminal {receipt['bank_role']} bank has {len(counted)} missing "
            f"categories (PROVISIONAL limit {limit}; outcome "
            f"{receipt['outcome']}): class_ids={counted!r}",
            receipt=receipt,
        )
    return receipt


@dataclass(frozen=True)
class AffineCalibrationFit:
    family: str
    scale: Any
    offset: Any
    backend: str
    input_center: Any | None = None
    target_class_id: int | None = None
    factorization: CalibrationFactorization | None = None
    receipt: CalibrationFitReceipt | None = None


@dataclass(frozen=True)
class ReadoutAffineCalibrationFit:
    family: str
    readout_scale: Any
    readout_offset: Any
    calibration_backend: str
    input_center: Any
    factorization: CalibrationFactorization | None = None
    receipt: CalibrationFitReceipt | None = None


def _joint_primitive_receipt(
    *,
    values: Any,
    coefficients: Any,
    labels: Any,
    intercepts: Any | None,
    query: QueryOperatorSpec,
    factorization: CalibrationFactorization,
    optimized_scale: Any,
    optimized_offset: Any,
    initial_scale: Any | None,
    initial_offset: Any | None,
    minimum_scale: float,
    max_iterations: int,
    lr: float,
    kkt_tolerance: float,
    objective_tolerance: float,
    backend: str,
) -> CalibrationFitReceipt:
    """Record the coupled fit's objective and convergence diagnostics."""

    started = time.perf_counter()
    x = torch.as_tensor(values, dtype=torch.float64)
    input_was_batched = x.ndim == 3
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim != 3:
        raise ValueError("joint primitive values must be [sample,primitive] or [mask,sample,primitive]")
    device = x.device
    u = torch.as_tensor(coefficients, device=device, dtype=torch.float64)
    y = torch.as_tensor(labels, device=device, dtype=torch.float64)
    if y.ndim != 2 or tuple(y.shape) != (int(x.shape[1]), int(u.shape[0])):
        raise ValueError("joint primitive labels differ from sample/query axes")
    if not bool(((y == 0.0) | (y == 1.0)).all().item()):
        raise ValueError("joint primitive calibration requires binary targets")
    positive = y.sum(dim=0)
    negative = float(y.shape[0]) - positive
    if bool(((positive <= 0) | (negative <= 0)).any().item()):
        raise ValueError("every joint calibration query requires positive and negative support")
    positive_weight = negative / positive
    query_weights = torch.where(
        y > 0.5,
        positive_weight.reshape(1, -1),
        torch.ones_like(y),
    )
    bias = (
        torch.zeros((u.shape[0],), device=device, dtype=torch.float64)
        if intercepts is None
        else torch.as_tensor(intercepts, device=device, dtype=torch.float64).reshape(-1)
    )
    mask_count, _sample_count, primitive_count = map(int, x.shape)

    def parameter(value: Any | None, default: float) -> Any:
        if value is None:
            return torch.full(
                (mask_count, primitive_count), default, device=device, dtype=torch.float64
            )
        tensor = torch.as_tensor(value, device=device, dtype=torch.float64)
        if tensor.ndim == 1:
            tensor = tensor.reshape(1, -1).expand(mask_count, -1)
        if tuple(tensor.shape) != (mask_count, primitive_count):
            raise ValueError("joint calibration initialization differs from mask/primitive axes")
        return tensor.detach().clone()

    initial_s = parameter(initial_scale, 1.0).clamp_min(float(minimum_scale))
    initial_o = parameter(initial_offset, 0.0)
    final_s = parameter(optimized_scale, 1.0)
    final_o = parameter(optimized_offset, 0.0)
    primitive_index = {
        axis: index for index, axis in enumerate(query.primitive_axis_ids)
    }
    query_index = {axis: index for index, axis in enumerate(query.query_axis_ids)}
    component_query_indices = tuple(
        tuple(query_index[axis] for axis in component.query_axis_ids)
        for component in factorization.components
    )
    component_primitive_indices = tuple(
        tuple(
            sorted(
                {
                    primitive_index[axis]
                    for query_axis in component.query_axis_ids
                    for axis in query.support[query_index[query_axis]]
                }
            )
        )
        for component in factorization.components
    )

    def query_logits(scale: Any, offset: Any) -> Any:
        calibrated = x * scale.unsqueeze(1) + offset.unsqueeze(1)
        return torch.einsum("mnp,qp->mnq", calibrated, u) + bias.reshape(1, 1, -1)

    def component_objectives(scale: Any, offset: Any) -> Any:
        logits = query_logits(scale, offset)
        rows = F.binary_cross_entropy_with_logits(
            logits,
            y.reshape(1, *y.shape).expand_as(logits),
            reduction="none",
        ) * query_weights.reshape(1, *query_weights.shape)
        per_query = rows.mean(dim=1)
        return torch.stack(
            [per_query[:, indices].mean(dim=1) for indices in component_query_indices],
            dim=1,
        )

    initial_objective = component_objectives(initial_s, initial_o)
    identity_objective = component_objectives(
        torch.ones_like(initial_s), torch.zeros_like(initial_o)
    )
    weighted_positive = (query_weights * y).sum(dim=0)
    weighted_negative = (query_weights * (1.0 - y)).sum(dim=0)
    null_logits = torch.log(
        weighted_positive.clamp_min(1e-30) / weighted_negative.clamp_min(1e-30)
    )
    null_rows = F.binary_cross_entropy_with_logits(
        null_logits.reshape(1, -1).expand_as(y), y, reduction="none"
    ) * query_weights
    null_per_query = null_rows.mean(dim=0)
    intercept_only = torch.stack(
        [null_per_query[list(indices)].mean() for indices in component_query_indices]
    ).reshape(1, -1).expand(mask_count, -1)

    final_s_grad = final_s.detach().clone().requires_grad_(True)
    final_o_grad = final_o.detach().clone().requires_grad_(True)
    final_objective_graph = component_objectives(final_s_grad, final_o_grad)
    kkt_rows = torch.empty_like(final_objective_graph)
    for component_index, primitive_indices in enumerate(component_primitive_indices):
        grad_s, grad_o = torch.autograd.grad(
            final_objective_graph[:, component_index].sum(),
            (final_s_grad, final_o_grad),
            retain_graph=True,
        )
        selected_s = grad_s[:, list(primitive_indices)]
        selected_o = grad_o[:, list(primitive_indices)]
        selected_scale = final_s_grad[:, list(primitive_indices)]
        at_boundary = selected_scale <= float(minimum_scale) + 1e-12
        scale_kkt = torch.where(
            at_boundary,
            torch.clamp(-selected_s, min=0.0),
            selected_s.abs(),
        )
        kkt_rows[:, component_index] = torch.maximum(
            scale_kkt.amax(dim=1), selected_o.abs().amax(dim=1)
        )
    final_objective = final_objective_graph.detach()

    finite = (
        torch.isfinite(final_objective)
        & torch.isfinite(kkt_rows)
        & torch.isfinite(final_s).all(dim=1, keepdim=True)
        & torch.isfinite(final_o).all(dim=1, keepdim=True)
    )
    objective_worsened = final_objective > initial_objective + float(
        objective_tolerance
    )
    converged = (
        finite
        & ~objective_worsened
        & (kkt_rows <= float(kkt_tolerance))
        & (final_s.amin(dim=1, keepdim=True) >= float(minimum_scale))
    )
    reasons = tuple(
        "kkt_converged"
        if bool(converged.reshape(-1)[index].item())
        else "objective_worsened"
        if bool(objective_worsened.reshape(-1)[index].item())
        else "nonfinite"
        if not bool(finite.reshape(-1)[index].item())
        else "kkt_not_converged"
        for index in range(mask_count * len(factorization.components))
    )
    profile = BackendProfile(
        backend_id="mechanistic_systems.overlap_factored_calibration.v1",
        capability="joint_primitive_affine_calibration",
        implementation=str(backend),
        device_type=str(device.type),
        numerical_mode="direct_float64_kkt_diagnostics",
        calls=int(max_iterations),
        wall_time_s=max(0.0, time.perf_counter() - started),
        peak_hbm_bytes=(
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        processed_contexts=int(mask_count * x.shape[1]),
        processed_queries=int(mask_count * u.shape[0]),
    )
    component_hashes = tuple(
        component.component_id
        for _mask in range(mask_count)
        for component in factorization.components
    )
    return CalibrationFitReceipt(
        dependency_hash=factorization.dependency_hash,
        factorization_hash=factorization.factorization_hash,
        component_hashes=component_hashes,
        backend=str(backend),
        initial_objective=tuple(float(value) for value in initial_objective.reshape(-1).tolist()),
        final_objective=tuple(float(value) for value in final_objective.reshape(-1).tolist()),
        identity_objective=tuple(float(value) for value in identity_objective.reshape(-1).tolist()),
        intercept_only_objective=tuple(float(value) for value in intercept_only.reshape(-1).tolist()),
        iterations=tuple(int(max_iterations) for _ in component_hashes),
        line_search_steps=tuple(0 for _ in component_hashes),
        gradient_kkt_residual=tuple(float(value) for value in kkt_rows.reshape(-1).tolist()),
        termination_reason=reasons,
        finite=tuple(bool(value) for value in finite.reshape(-1).tolist()),
        converged=tuple(bool(value) for value in converged.reshape(-1).tolist()),
        backend_profile=profile.to_runtime_dict(),
    )


class CanonicalCalibrationBackend:
    """Compile calibration coupling from structural dependencies.

    The canonical entry point is :meth:`fit_problem`. ``fit`` remains only as
    a lowering edge for sealed v1 callers while those manifests are imported.
    Architecture names and target-type labels never participate in dispatch.
    """

    backend_id = "mechanistic_systems.overlap_factored_calibration.v1"
    supported_families = frozenset(
        {
            "none",
            "overlap_factored_affine",
            "probe_logit_v1",
            "probe_margin_v2",
            "primitive_coherent_affine",
            "shared_affine",
            "classwise_diagonal_affine",
            "c2_shared_diagonal_affine",
        }
    )

    def __init__(
        self, *, runtime_profile: RuntimeExecutionProfile | None = None
    ) -> None:
        self.problem_compiler = CalibrationProblemCompiler()
        # Ambient settings are bridged once when a legacy caller constructs a
        # backend. New CLI/session callers pass an explicit immutable profile.
        self.runtime_profile = RuntimeExecutionProfile.resolve(runtime_profile)

    def fit_problem(
        self,
        *,
        readout: ReadoutProjectionSpec,
        query: QueryOperatorSpec,
        problem: CalibrationProblemSpec,
        query_values: Any,
        targets: Any,
        sample_weights: Any | None = None,
        initial_scale: Any | None = None,
        initial_offset: Any | None = None,
        execution_context: CalibrationExecutionContext | None = None,
        **options: Any,
    ) -> Any:
        """Compile exact independence and dispatch by mathematical component.

        ``query_values`` are evaluated *after* the immutable readout projection
        and query operator. Fixed probe intercepts therefore enter exactly once
        before this method. Primitive-space repair must be explicitly requested
        and provide its own primitive values and coefficient map.
        """

        if torch is None or F is None:
            raise RuntimeError("calibration requires PyTorch")
        declared_layout = options.pop("crossfit_layout", None)
        runtime_input_provenance = options.pop("runtime_input_provenance", None)
        if declared_layout is not None and not isinstance(
            declared_layout, CrossFitLayout
        ):
            raise TypeError("crossfit_layout must be a CrossFitLayout")
        declared_fold_validity = options.get("fold_validity")
        declared_row_validity = options.get("row_validity")
        if (
            declared_fold_validity is not None
            and declared_row_validity is not None
        ):
            fold_rows = torch.as_tensor(declared_fold_validity, device="cpu")
            row_rows = torch.as_tensor(declared_row_validity, device="cpu")
            if not bool(torch.equal(fold_rows, row_rows)):
                raise ValueError(
                    "fold_validity and row_validity must be identical when both are declared"
                )
        # Compilation is a pure function of the three content-addressed
        # specs and is invoked once per fit on the sampled-path hot path —
        # memoize it (bounded; identical problems recur thousands of times).
        memo = getattr(self, "_factorization_memo", None)
        if memo is None:
            memo = {}
            self._factorization_memo = memo
        memo_key = (
            readout.content_hash,
            query.content_hash,
            problem.content_hash,
        )
        factorization = memo.get(memo_key)
        if factorization is None:
            factorization = self.problem_compiler.compile(
                readout=readout,
                query=query,
                problem=problem,
            )
            if len(memo) >= 256:
                memo.pop(next(iter(memo)))
            memo[memo_key] = factorization

        optimization_contract = str(
            problem.notes.get("optimization_contract", "finite_mle")
        )
        runtime_layout = declared_layout
        context_ids = options.get("selection_context_ids")
        if execution_context is not None:
            # Proper-score selection needs a full layout (including its local
            # inner folds). Other solvers only receive a layout when the caller
            # explicitly supplies cross-fit row validity and ordered contexts.
            if (
                runtime_layout is None
                and optimization_contract == PROPER_SCORE_SHRINKAGE_AFFINE_V2
                and options.get("calibration_penalty", "identity_anchor") != "firth"
            ):
                if context_ids is None:
                    raise ValueError(
                        "proper-score execution context requires immutable context IDs"
                    )
                runtime_layout = CrossFitLayout.from_binary_targets(
                    targets,
                    context_ids,
                    fold_validity=options.get("fold_validity"),
                    requested_blocks=int(
                        problem.notes.get(
                            "calibration_selection_blocks",
                            CANONICAL_CALIBRATION_SELECTION_BLOCKS,
                        )
                    ),
                )
            elif runtime_layout is None and options.get("fold_validity") is not None:
                if context_ids is None:
                    raise ValueError(
                        "cross-fit execution context requires ordered context IDs"
                    )
                binary_targets = (
                    targets
                    if problem.loss_family == "independent_binary_bce"
                    else None
                )
                runtime_layout = CrossFitLayout.from_outer_fit_membership(
                    context_ids,
                    options["fold_validity"],
                    binary_targets=binary_targets,
                )
            prepared = execution_context.bind(
                factorization=factorization,
                query_values=query_values,
                targets=targets,
                sample_weights=sample_weights,
                context_ids=context_ids,
                layout=runtime_layout,
                problem=problem,
                input_provenance=runtime_input_provenance,
            )
            query_values = prepared.query_values
            targets = prepared.targets
            sample_weights = prepared.sample_weights
            if (
                options.get("fold_validity") is None
                and prepared.layout is not None
                and prepared.layout.fold_count > 1
            ):
                options["fold_validity"] = prepared.outer_fold_validity()
        if (options.get("calibration_penalty") == "firth"
                and optimization_contract == SAMPLED_PATH_V1_CALIBRATION_ALGORITHM):
            if execution_context is not None:
                execution_context.record_solver_call("firth_positive_affine")
            return fit_batched_post_query_affine(
                query_values, targets, factorization=factorization, weighting="design_weighted",
                sample_weights=sample_weights, calibration_penalty="firth",
                fixed_offsets=options.get("fixed_offsets"), minimum_scale=problem.minimum_scale,
                firth_convergence=str(options.get("firth_convergence", "absolute")),
                metric_loss_weighting=options.get("metric_loss_weighting"),
                initial_scale=initial_scale, initial_offset=initial_offset,
                max_iterations=int(options.get("iters", 64)),
                tolerance=float(options.get("kkt_tolerance", 1e-8)),
                hessian_ridge=float(options.get("hessian_ridge", 1e-10)),
                objective_tolerance=float(options.get("objective_tolerance", 1e-10)),
                reference_tolerance=float(options.get("reference_tolerance", 1e-8)),
                statistics_backend=str(options.get("statistics_backend", "auto")),
                require_convergence=True)
        if optimization_contract == SAMPLED_PATH_V1_CALIBRATION_ALGORITHM:
            if (
                problem.parameterization != "post_query_positive_affine"
                or problem.loss_family != "independent_binary_bce"
            ):
                raise ValueError(
                    "sampled_path_frontier_v1 calibration requires independent "
                    "post-query binary affine factors"
                )
            input_center = options.get("input_center")
            if input_center is None:
                raise ValueError(
                    "sampled_path_frontier_v1 calibration requires its fixed-readout center"
                )
            if execution_context is not None:
                execution_context.record_solver_call(
                    "sampled_path_coordinate_affine"
                )
            return fit_sampled_path_v1_coordinate_affine(
                query_values,
                targets,
                factorization=factorization,
                input_center=input_center,
                weighting=problem.weighting,
                sample_weights=sample_weights,
                minimum_scale=problem.minimum_scale,
                configured_iterations=int(
                    options.get("iters", options.get("num_iterations", 64))
                ),
                learning_rate=float(options.get("lr", 0.25)),
                damping_init=float(options.get("damping_init", 1e-4)),
                objective_tolerance=float(
                    options.get("objective_tolerance", 1e-6)
                ),
            )
        if optimization_contract == PROPER_SCORE_SHRINKAGE_AFFINE_V2:
            if (
                problem.parameterization != "post_query_positive_affine"
                or problem.loss_family != "independent_binary_bce"
            ):
                raise ValueError(
                    "proper-score shrinkage calibration requires independent "
                    "post-query binary affine factors"
                )
            context_ids = options.get("selection_context_ids")
            if context_ids is None:
                raise ValueError(
                    "proper-score shrinkage calibration requires immutable "
                    "calibration context IDs"
                )
            declared_strengths = tuple(
                float(value)
                for value in problem.notes.get(
                    "identity_anchor_strengths",
                    CANONICAL_CALIBRATION_IDENTITY_ANCHOR_STRENGTHS,
                )
            )
            runtime_strengths = tuple(
                float(value)
                for value in options.get(
                    "identity_anchor_strengths", declared_strengths
                )
            )
            declared_blocks = int(
                problem.notes.get(
                    "calibration_selection_blocks",
                    CANONICAL_CALIBRATION_SELECTION_BLOCKS,
                )
            )
            declared_epsilon = float(
                problem.notes.get(
                    "anchor_probability_epsilon",
                    CANONICAL_CALIBRATION_ANCHOR_PROBABILITY_EPSILON,
                )
            )
            declared_iterations = int(
                problem.notes.get(
                    "max_iterations", CANONICAL_CALIBRATION_MAX_ITERATIONS
                )
            )
            declared_numerics = {
                "kkt_tolerance": float(
                    problem.notes.get(
                        "kkt_tolerance", CANONICAL_CALIBRATION_KKT_TOLERANCE
                    )
                ),
                "hessian_ridge": float(
                    problem.notes.get(
                        "hessian_ridge", CANONICAL_CALIBRATION_HESSIAN_RIDGE
                    )
                ),
                "objective_tolerance": float(
                    problem.notes.get(
                        "objective_tolerance",
                        CANONICAL_CALIBRATION_OBJECTIVE_TOLERANCE,
                    )
                ),
                "reference_tolerance": float(
                    problem.notes.get(
                        "reference_tolerance",
                        CANONICAL_CALIBRATION_REFERENCE_TOLERANCE,
                    )
                ),
            }
            if runtime_strengths != declared_strengths:
                raise ValueError(
                    "proper-score runtime strength grid differs from its "
                    "scientific problem"
                )
            if (
                int(options.get("calibration_selection_blocks", declared_blocks))
                != declared_blocks
            ):
                raise ValueError(
                    "proper-score runtime block count differs from its "
                    "scientific problem"
                )
            if (
                float(options.get("anchor_probability_epsilon", declared_epsilon))
                != declared_epsilon
            ):
                raise ValueError(
                    "proper-score runtime probability epsilon differs from its "
                    "scientific problem"
                )
            runtime_iterations = int(
                options.get(
                    "max_iterations",
                    options.get(
                        "iters", options.get("num_iterations", declared_iterations)
                    ),
                )
            )
            if runtime_iterations != declared_iterations:
                raise ValueError(
                    "proper-score runtime iteration bound differs from its "
                    "scientific problem"
                )
            for key, declared_value in declared_numerics.items():
                if float(options.get(key, declared_value)) != declared_value:
                    raise ValueError(
                        f"proper-score runtime {key} differs from its scientific problem"
                    )
            reference_audit_policy = GPU_HARD_GATES_AUDIT_POLICY
            statistics_backend = str(options.get("statistics_backend", "torch"))
            proper_options = dict(
                factorization=factorization,
                context_ids=context_ids,
                weighting=problem.weighting,
                sample_weights=sample_weights,
                minimum_scale=problem.minimum_scale,
                initial_scale=initial_scale,
                initial_offset=initial_offset,
                candidate_strengths=declared_strengths,
                selection_block_count=declared_blocks,
                max_iterations=declared_iterations,
                tolerance=declared_numerics["kkt_tolerance"],
                hessian_ridge=declared_numerics["hessian_ridge"],
                objective_tolerance=declared_numerics["objective_tolerance"],
                reference_tolerance=declared_numerics["reference_tolerance"],
                statistics_backend=statistics_backend,
                source_logit_dtype=options.get("source_logit_dtype"),
                anchor_probability_epsilon=declared_epsilon,
                fold_validity=options.get("fold_validity"),
                required_query_mask=options.get("required_query_mask"),
                reference_audit_policy=reference_audit_policy,
                audit_seed=options.get("audit_seed"),
                deployment_refit_anchor=str(
                    options.get("deployment_refit_anchor", "selected")
                ),
            )
            if options.get("calibration_penalty", "identity_anchor") == "firth":
                proper_options["calibration_penalty"] = "firth"
                proper_options["fixed_offsets"] = options.get("fixed_offsets")
                proper_options["firth_convergence"] = str(options.get("firth_convergence", "absolute"))
                proper_options["metric_loss_weighting"] = options.get("metric_loss_weighting")
            if "missing_column_policy" in options:
                # A consumer that can carry a missing column (the
                # terminal bank provider) requests it explicitly.
                proper_options["missing_column_policy"] = str(
                    options["missing_column_policy"]
                )
            if execution_context is None:
                proper_options.pop("initial_scale", None)
                proper_options.pop("initial_offset", None)
                return fit_proper_score_shrinkage_affine(
                    query_values,
                    targets,
                    **proper_options,
                )
            execution_context.record_solver_call("proper_score_positive_affine")
            return _fit_proper_score_shrinkage_affine_core(
                query_values,
                targets,
                _crossfit_layout=runtime_layout,
                **proper_options,
            )[0]
        if optimization_contract != "finite_mle":
            raise ValueError(
                f"unsupported calibration optimization contract {optimization_contract!r}"
            )
        if (
            problem.parameterization == "post_query_positive_affine"
            and problem.loss_family == "independent_binary_bce"
        ):
            if execution_context is not None:
                execution_context.record_solver_call(
                    "independent_post_query_positive_affine"
                )
            return fit_batched_post_query_affine(
                query_values,
                targets,
                factorization=factorization,
                weighting=problem.weighting,
                sample_weights=sample_weights,
                minimum_scale=problem.minimum_scale,
                initial_scale=initial_scale,
                initial_offset=initial_offset,
                max_iterations=int(options.get("iters", options.get("num_iterations", 80))),
                tolerance=float(options.get("kkt_tolerance", 1e-8)),
                hessian_ridge=float(options.get("hessian_ridge", 1e-10)),
                objective_tolerance=float(options.get("objective_tolerance", 1e-10)),
                reference_tolerance=float(options.get("reference_tolerance", 1e-8)),
                statistics_backend=str(options.get("statistics_backend", "auto")),
                require_convergence=bool(options.get("require_convergence", True)),
                source_logit_dtype=options.get("source_logit_dtype"),
                identity_anchor_strength=options.get("identity_anchor_strength"),
                **({"calibration_penalty": "firth", "fixed_offsets": options.get("fixed_offsets"),
                    "firth_convergence": str(options.get("firth_convergence", "absolute")),
                    "metric_loss_weighting": options.get("metric_loss_weighting")}
                   if options.get("calibration_penalty", "identity_anchor") == "firth" else {}),
                anchor_probability_epsilon=float(
                    options.get("anchor_probability_epsilon", 1e-12)
                ),
                row_validity=options.get("row_validity"),
                required_query_mask=options.get("required_query_mask"),
                reference_audit_policy=str(
                    options.get("reference_audit_policy", GPU_HARD_GATES_AUDIT_POLICY)
                ),
                audit_seed=options.get("audit_seed"),
            )
        if (
            problem.parameterization == "shared_temperature"
            and problem.loss_family == "multiclass_cross_entropy"
        ):
            if query.operator_kind != "coordinate":
                raise ValueError(
                    "shared-temperature multiclass calibration requires declared "
                    "coordinate logits"
                )
            if initial_offset is not None and bool(
                torch.as_tensor(initial_offset).abs().max().item() > 0.0
            ):
                raise ValueError(
                    "shared-temperature calibration has no identifiable additive offset"
                )
            if execution_context is not None:
                execution_context.record_solver_call(
                    "shared_temperature_multiclass"
                )
            return fit_shared_temperature_multiclass(
                query_values,
                targets,
                factorization=factorization,
                weighting=problem.weighting,
                sample_weights=sample_weights,
                minimum_scale=problem.minimum_scale,
                initial_scale=initial_scale,
                max_iterations=int(
                    options.get("iters", options.get("num_iterations", 80))
                ),
                tolerance=float(options.get("kkt_tolerance", 1e-9)),
                hessian_ridge=float(options.get("hessian_ridge", 1e-12)),
                objective_tolerance=float(
                    options.get("objective_tolerance", 1e-10)
                ),
                reference_tolerance=float(
                    options.get("reference_tolerance", 1e-8)
                ),
                maximum_scale=float(options.get("maximum_scale", 1e6)),
                require_convergence=bool(options.get("require_convergence", True)),
                row_validity=options.get(
                    "row_validity", options.get("fold_validity")
                ),
            )
        if problem.parameterization != "primitive_diagonal_affine":
            raise NotImplementedError(
                "the declared calibration component requires a joint solver that "
                "has not been promoted for this loss family"
            )
        if (
            options.get("fold_validity") is not None
            or options.get("row_validity") is not None
        ):
            raise NotImplementedError(
                "primitive-diagonal affine has no certified folded objective/gradient "
                "path for a folded row measure"
            )
        if execution_context is not None:
            execution_context.record_solver_call(
                "primitive_diagonal_affine"
            )
        primitive_values = options.get("primitive_values")
        query_coefficients = options.get("query_coefficients")
        if primitive_values is None or query_coefficients is None:
            raise ValueError(
                "primitive_coherent_affine requires primitive_values and "
                "query_coefficients"
            )
        if problem.loss_family != "independent_binary_bce":
            raise NotImplementedError(
                "primitive coherent affine currently supports independent binary BCE"
            )
        if problem.weighting != "pos_weighted" or sample_weights is not None:
            raise NotImplementedError(
                "primitive coherent affine requires positive-weighted BCE without "
                "design weights until a weighted joint reference is promoted"
            )
        values = torch.as_tensor(primitive_values, dtype=torch.float32)
        coefficients = torch.as_tensor(
            query_coefficients,
            device=values.device,
            dtype=torch.float32,
        )
        labels = torch.as_tensor(targets, device=values.device, dtype=torch.float32)
        intercepts = options.get("query_intercepts")
        intercepts = (
            None
            if intercepts is None
            else torch.as_tensor(intercepts, device=values.device, dtype=torch.float32)
        )
        common = {
            "init_readout_scale": initial_scale,
            "init_readout_offset": initial_offset,
            "num_iterations": int(options.get("iters", options.get("num_iterations", 80))),
            "enforce_positive_scale": bool(options.get("enforce_positive_scale", True)),
            "positive_scale_min": float(options.get("ws_min", problem.minimum_scale)),
            "lr": float(options.get("lr", 0.25)),
            "identity_l2": float(options.get("identity_l2", 0.0)),
            "offset_l2": float(options.get("offset_l2", 0.0)),
            "shared_readout_backend": str(options.get("shared_readout_backend", "auto")),
            "use_triton_grad_hess": bool(options.get("use_triton_grad_hess", True)),
            "diagnostic_context": options.get("diagnostic_context"),
        }
        if values.ndim == 3:
            fit = fit_batched_shared_readout_diagonal_affine_calibration(
                values,
                coefficients,
                labels,
                target_intercepts=intercepts,
                shared_readout_solver=str(options.get("shared_readout_solver", "lbfgs")),
                **common,
            )
        elif values.ndim == 2:
            if intercepts is not None:
                # The legacy single-fit entry point lacks target intercepts;
                # route through the batched implementation and retain its
                # leading singleton mask axis rather than dropping the bias.
                batched = fit_batched_shared_readout_diagonal_affine_calibration(
                    values.unsqueeze(0),
                    coefficients,
                    labels,
                    target_intercepts=intercepts,
                    shared_readout_solver=str(options.get("shared_readout_solver", "lbfgs")),
                    **common,
                )
                fit = batched
            else:
                fit = fit_shared_readout_diagonal_affine_calibration(
                    values,
                    coefficients,
                    labels,
                    **common,
                )
        else:
            raise ValueError("primitive calibration values must be [sample,primitive] or [mask,sample,primitive]")
        fit_scale = fit.readout_scale
        fit_offset = fit.readout_offset
        if values.ndim == 2 and fit_scale.ndim == 2:
            fit_scale = fit_scale[0]
            fit_offset = fit_offset[0]
        receipt = _joint_primitive_receipt(
            values=values,
            coefficients=coefficients,
            labels=labels,
            intercepts=intercepts,
            query=query,
            factorization=factorization,
            optimized_scale=fit_scale,
            optimized_offset=fit_offset,
            initial_scale=initial_scale,
            initial_offset=initial_offset,
            minimum_scale=float(options.get("ws_min", problem.minimum_scale)),
            max_iterations=int(options.get("iters", options.get("num_iterations", 80))),
            lr=float(options.get("lr", 0.25)),
            kkt_tolerance=float(options.get("joint_kkt_tolerance", 1e-4)),
            objective_tolerance=float(options.get("objective_tolerance", 1e-8)),
            backend=str(fit.calibration_backend),
        )
        if bool(options.get("require_convergence", True)) and not receipt.passed:
            raise CalibrationConvergenceError(
                "joint primitive calibration failed its convergence receipt",
                receipt,
            )
        return ReadoutAffineCalibrationFit(
            family="primitive_coherent_affine",
            readout_scale=fit_scale,
            readout_offset=fit_offset,
            calibration_backend=fit.calibration_backend,
            input_center=torch.zeros(
                values.shape[-1], device=values.device, dtype=values.dtype
            ),
            factorization=factorization,
            receipt=receipt,
        )

    @staticmethod
    def _nonuniform_sample_weights(weights: Any, *, count: int, device: Any) -> Any | None:
        if weights is None:
            return None
        result = torch.as_tensor(weights, device=device, dtype=torch.float32).reshape(-1)
        if int(result.numel()) != int(count) or float(result.sum().item()) <= 0.0:
            raise ValueError("calibration sample weights differ from the fit axis")
        result = result.clamp_min(0.0)
        result = result / result.sum().clamp_min(1e-12)
        if int(result.numel()) <= 1 or torch.allclose(
            result, result[:1].expand_as(result), rtol=1e-5, atol=1e-8
        ):
            return None
        return result

    def _fit_scalar_target_affine(
        self,
        values: Any,
        targets: Any,
        sample_weights: Any | None,
        *,
        class_id: int,
        input_center: Any,
        options: dict[str, Any],
    ) -> AffineCalibrationFit:
        class_id = int(class_id)
        if values.ndim != 2 or not 0 <= class_id < int(values.shape[1]):
            raise ValueError("weighted target calibration requires a valid readout class coordinate")
        labels = torch.as_tensor(targets, device=values.device, dtype=torch.float32).reshape(
            int(values.shape[0]), -1
        )
        if int(labels.shape[1]) != 1:
            raise ValueError("weighted target calibration requires one binary query")
        labels = labels[:, 0]
        center = torch.as_tensor(input_center, device=values.device, dtype=torch.float32).reshape(-1)
        if int(center.numel()) == 1:
            center_value = center[0]
        elif int(center.numel()) == int(values.shape[1]):
            center_value = center[class_id]
        else:
            raise ValueError("readout bias does not match the calibration coordinate")
        logits = values[:, class_id] - center_value
        weights = (
            None
            if sample_weights is None
            else torch.as_tensor(
                sample_weights, device=values.device,
                dtype=torch.float64 if options.get("calibration_penalty") == "firth" else torch.float32
            ).reshape(-1)
        )
        enforce_positive = bool(options.get("enforce_positive_scale", False))
        minimum = float(options.get("ws_min", options.get("positive_scale_min", 1e-8)))
        if enforce_positive:
            primitive_axis = (f"legacy_centered_output:{class_id}",)
            readout = ReadoutProjectionSpec(
                projection_id="legacy-centered-output-compatibility",
                projection_kind="identity",
                source_axis_ids=primitive_axis,
                primitive_axis_ids=primitive_axis,
                source_hash=stable_digest(
                    {
                        "compatibility_boundary": "sealed_v1_centered_output",
                        "class_id": class_id,
                        "readout_width": int(values.shape[1]),
                    }
                ),
            )
            query = QueryOperatorSpec.coordinate(
                operator_id=f"legacy-coordinate:{class_id}",
                primitive_axis_ids=primitive_axis,
                source_hash=readout.content_hash,
            )
            problem = CalibrationProblemSpec(
                problem_id=f"legacy-positive-affine:{class_id}",
                readout_projection_hash=readout.content_hash,
                query_operator_hash=query.content_hash,
                weighting=str(options.get("binary_loss_weighting", "pos_weighted")),
                minimum_scale=minimum,
                notes={"compatibility_only": True},
            )

            def _selected_initial(name: str) -> Any | None:
                raw = options.get(name)
                if raw is None:
                    return None
                tensor = torch.as_tensor(raw, device=values.device, dtype=torch.float64).reshape(-1)
                if int(tensor.numel()) == 1:
                    return tensor
                if int(tensor.numel()) == int(values.shape[1]):
                    return tensor[class_id : class_id + 1]
                raise ValueError(f"{name} differs from the legacy readout axis")

            solver_options = {
                key: value
                for key, value in options.items()
                if key
                not in {
                    "sample_weights",
                    "init_readout_scale",
                    "init_readout_offset",
                    "target_vectors",
                    "class_id",
                    "readout_bias",
                    "scalar_target_semantics",
                }
            }
            fit = self.fit_problem(
                readout=readout,
                query=query,
                problem=problem,
                query_values=logits.reshape(-1, 1),
                targets=labels.reshape(-1, 1),
                sample_weights=weights,
                initial_scale=_selected_initial("init_readout_scale"),
                initial_offset=_selected_initial("init_readout_offset"),
                **solver_options,
            )
            return AffineCalibrationFit(
                family="overlap_factored_post_query_affine",
                scale=fit.scale[0, 0].detach().reshape(1),
                offset=fit.offset[0, 0].detach().reshape(1),
                backend=fit.backend,
                input_center=center_value.detach().reshape(1),
                target_class_id=class_id,
                factorization=fit.factorization,
                receipt=fit.receipt,
            )
        raw_scale = torch.tensor(
            0.54132485 if enforce_positive else 1.0,
            device=values.device,
            dtype=torch.float32,
            requires_grad=True,
        )
        offset = torch.tensor(0.0, device=values.device, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            (raw_scale, offset),
            lr=float(options.get("lr", 0.25)),
            max_iter=max(1, int(options.get("iters", options.get("num_iterations", 50)))),
            line_search_fn="strong_wolfe",
        )
        weighting = str(options.get("binary_loss_weighting", "pos_weighted"))
        positives = labels.sum()
        total = torch.as_tensor(
            labels.numel(), device=labels.device, dtype=labels.dtype
        )
        negatives = total - positives
        positive_weight = torch.where(
            (positives > 0) & (negatives > 0),
            negatives / positives.clamp_min(1e-12),
            torch.ones_like(positives),
        )

        def closure() -> Any:
            optimizer.zero_grad(set_to_none=True)
            scale = F.softplus(raw_scale) + minimum if enforce_positive else raw_scale
            eta = scale * logits + offset
            row_loss = F.binary_cross_entropy_with_logits(
                eta,
                labels,
                pos_weight=(positive_weight if weighting == "pos_weighted" else None),
                reduction="none",
            )
            loss = row_loss.mean() if weights is None else (row_loss * weights).sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            scale = F.softplus(raw_scale) + minimum if enforce_positive else raw_scale
        return AffineCalibrationFit(
            family=(
                "design_weighted_target_affine"
                if weights is not None
                else "scalar_target_affine"
            ),
            scale=scale.detach().reshape(1),
            offset=offset.detach().reshape(1),
            backend=(
                "torch_lbfgs_weighted_profiled_logit"
                if weights is not None
                else "torch_lbfgs_profiled_logit"
            ),
            input_center=center_value.detach().reshape(1),
            target_class_id=class_id,
        )

    def fit(self, family: str, logits: Any, targets: Any, **options: Any) -> Any:
        if torch is None or F is None:
            raise RuntimeError("calibration requires PyTorch")
        token = str(family)
        if token not in self.supported_families:
            raise ValueError(f"unsupported calibration family: {token!r}")
        values = torch.as_tensor(logits, dtype=torch.float32)
        if token == "none":
            width = int(values.shape[-1]) if values.ndim > 1 else 1
            return AffineCalibrationFit(
                token,
                torch.ones(width, device=values.device),
                torch.zeros(width, device=values.device),
                "identity",
            )
        if token in {"overlap_factored_affine", "probe_logit_v1", "probe_margin_v2"}:
            readout_spec = options.get("readout_projection_spec")
            query_spec = options.get("query_operator_spec")
            problem_spec = options.get("calibration_problem_spec")
            if not isinstance(readout_spec, ReadoutProjectionSpec):
                raise ValueError(f"{token} requires ReadoutProjectionSpec")
            if not isinstance(query_spec, QueryOperatorSpec):
                raise ValueError(f"{token} requires QueryOperatorSpec")
            if not isinstance(problem_spec, CalibrationProblemSpec):
                raise ValueError(f"{token} requires CalibrationProblemSpec")
            return self.fit_problem(
                readout=readout_spec,
                query=query_spec,
                problem=problem_spec,
                query_values=values,
                targets=targets,
                sample_weights=options.get("sample_weights"),
                initial_scale=options.get("initial_scale"),
                initial_offset=options.get("initial_offset"),
                **{
                    key: value
                    for key, value in options.items()
                    if key
                    not in {
                        "readout_projection_spec",
                        "query_operator_spec",
                        "calibration_problem_spec",
                        "sample_weights",
                        "initial_scale",
                        "initial_offset",
                    }
                },
            )
        if token in {
            "primitive_coherent_affine",
            "classwise_diagonal_affine",
            "c2_shared_diagonal_affine",
        }:
            target_vectors = options.get("target_vectors")
            if target_vectors is None:
                raise ValueError("C2 calibration requires target_vectors")
            sample_weights = (
                torch.as_tensor(options["sample_weights"], device=values.device, dtype=torch.float64).reshape(-1)
                if options.get("calibration_penalty") == "firth" else
                self._nonuniform_sample_weights(options.get("sample_weights"),
                    count=int(values.shape[0]), device=values.device))
            readout_center = options.get("readout_bias")
            if readout_center is None:
                readout_center = torch.zeros(
                    int(values.shape[1]), device=values.device, dtype=torch.float32
                )
            readout_center = torch.as_tensor(
                readout_center, device=values.device, dtype=torch.float32
            ).reshape(-1)
            if bool(options.get("scalar_target_semantics", False)):
                return self._fit_scalar_target_affine(
                    values,
                    targets,
                    sample_weights,
                    class_id=int(options.get("class_id", -1)),
                    input_center=readout_center,
                    options=dict(options),
                )
            vectors = torch.as_tensor(
                target_vectors, device=values.device, dtype=torch.float32
            )
            if vectors.ndim != 2 or int(vectors.shape[1]) != int(values.shape[1]):
                raise ValueError("target vectors differ from the declared readout width")
            if token == "primitive_coherent_affine":
                centered = values - (
                    readout_center[0]
                    if int(readout_center.numel()) == 1
                    else readout_center.reshape(1, -1)
                )
                primitive_axes = tuple(
                    f"latent_readout:{index}" for index in range(values.shape[1])
                )
                query_axes = tuple(
                    f"latent_query:{index}" for index in range(vectors.shape[0])
                )
                declared_support = options.get("query_support")
                support = (
                    tuple(primitive_axes for _ in query_axes)
                    if declared_support is None
                    else tuple(
                        tuple(str(axis) for axis in row)
                        for row in declared_support
                    )
                )
                readout = ReadoutProjectionSpec(
                    projection_id="explicit-primitive-coherent-readout-repair",
                    projection_kind="identity",
                    source_axis_ids=primitive_axes,
                    primitive_axis_ids=primitive_axes,
                    source_hash=str(
                        options.get(
                            "readout_source_hash",
                            stable_digest(
                                {
                                    "compatibility_boundary": "explicit_primitive_coherent_affine",
                                    "readout_width": int(values.shape[1]),
                                }
                            ),
                        )
                    ),
                    notes={"ordinary_category_calibration": False},
                )
                query = QueryOperatorSpec(
                    operator_id="explicit-latent-feature-queries",
                    operator_kind="linear_combination",
                    primitive_axis_ids=primitive_axes,
                    query_axis_ids=query_axes,
                    support=support,
                    source_hash=readout.content_hash,
                )
                problem = CalibrationProblemSpec(
                    problem_id="explicit-primitive-coherent-affine",
                    readout_projection_hash=readout.content_hash,
                    query_operator_hash=query.content_hash,
                    parameterization="primitive_diagonal_affine",
                    loss_family="independent_binary_bce",
                    weighting=str(options.get("binary_loss_weighting", "pos_weighted")),
                    minimum_scale=float(
                        options.get("ws_min", options.get("positive_scale_min", 1e-8))
                    ),
                    notes={"ordinary_category_calibration": False},
                )
                return self.fit_problem(
                    readout=readout,
                    query=query,
                    problem=problem,
                    query_values=centered.matmul(vectors.transpose(0, 1)),
                    targets=targets,
                    sample_weights=sample_weights,
                    initial_scale=options.get("init_readout_scale"),
                    initial_offset=options.get("init_readout_offset"),
                    primitive_values=centered,
                    query_coefficients=vectors,
                    query_intercepts=options.get("target_intercepts"),
                    **{
                        key: value
                        for key, value in options.items()
                        if key
                        not in {
                            "target_vectors",
                            "target_intercepts",
                            "sample_weights",
                            "init_readout_scale",
                            "init_readout_offset",
                            "query_support",
                            "readout_source_hash",
                        }
                    },
                )
            if token != "primitive_coherent_affine":
                intercepts = options.get("target_intercepts")
                intercepts = (
                    torch.zeros(
                        vectors.shape[0], device=values.device, dtype=values.dtype
                    )
                    if intercepts is None
                    else torch.as_tensor(
                        intercepts, device=values.device, dtype=values.dtype
                    ).reshape(-1)
                )
                if int(intercepts.numel()) != int(vectors.shape[0]):
                    raise ValueError("target intercepts differ from the query axis")
                projected = values.matmul(vectors.transpose(0, 1)) + intercepts.reshape(1, -1)
                source_axes = tuple(f"readout_feature:{index}" for index in range(values.shape[1]))
                primitive_axes = tuple(f"projected_query:{index}" for index in range(vectors.shape[0]))

                def _tensor_hash(tensor: Any) -> str:
                    array = tensor.detach().to(device="cpu").contiguous().numpy()
                    digest = hashlib.sha256()
                    digest.update(str(array.dtype).encode("ascii"))
                    digest.update(str(tuple(array.shape)).encode("ascii"))
                    digest.update(array.tobytes(order="C"))
                    return digest.hexdigest()

                readout = ReadoutProjectionSpec(
                    projection_id="legacy-fixed-projection-lowering",
                    projection_kind="fixed_linear",
                    source_axis_ids=source_axes,
                    primitive_axis_ids=primitive_axes,
                    source_hash=str(
                        options.get(
                            "readout_source_hash",
                            stable_digest(
                                {
                                    "compatibility_boundary": "fixed_projection_lowering",
                                    "source_width": int(values.shape[1]),
                                }
                            ),
                        )
                    ),
                    weight_hash=_tensor_hash(vectors),
                    intercept_hash=_tensor_hash(intercepts),
                    notes={"legacy_family": token},
                )
                query = QueryOperatorSpec.coordinate(
                    operator_id="legacy-projected-coordinate-queries",
                    primitive_axis_ids=primitive_axes,
                    source_hash=readout.content_hash,
                )
                problem = CalibrationProblemSpec(
                    problem_id="legacy-c2-lowered-to-output-factorization",
                    readout_projection_hash=readout.content_hash,
                    query_operator_hash=query.content_hash,
                    weighting=str(options.get("binary_loss_weighting", "pos_weighted")),
                    minimum_scale=float(
                        options.get("ws_min", options.get("positive_scale_min", 1e-8))
                    ),
                    notes={"compatibility_family": token},
                )
                return self.fit_problem(
                    readout=readout,
                    query=query,
                    problem=problem,
                    query_values=projected,
                    targets=targets,
                    sample_weights=options.get("sample_weights"),
                    initial_scale=options.get("init_classifier_scale"),
                    initial_offset=options.get("init_classifier_offset"),
                    **{
                        key: value
                        for key, value in options.items()
                        if key
                        not in {
                            "target_vectors",
                            "target_intercepts",
                            "sample_weights",
                            "init_classifier_scale",
                            "init_classifier_offset",
                        }
                    },
                )
            raise AssertionError("unreachable primitive calibration lowering")
        # The v2 shared family is deliberately only two parameters. This is
        # the simplest nonidentity family in extraction-only inner selection.
        labels = torch.as_tensor(targets, device=values.device, dtype=torch.float32)
        if labels.shape != values.shape:
            labels = labels.reshape(values.shape)
        raw_scale = torch.nn.Parameter(torch.zeros((), device=values.device))
        offset = torch.nn.Parameter(torch.zeros((), device=values.device))
        optimizer = torch.optim.LBFGS(
            (raw_scale, offset),
            lr=float(options.get("lr", 0.25)),
            max_iter=max(1, int(options.get("num_iterations", 80))),
            line_search_fn="strong_wolfe",
        )

        def closure() -> Any:
            optimizer.zero_grad(set_to_none=True)
            scale = F.softplus(raw_scale) + 1e-8
            loss = F.binary_cross_entropy_with_logits(scale * values + offset, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return AffineCalibrationFit(
            token,
            (F.softplus(raw_scale) + 1e-8).detach(),
            offset.detach(),
            "torch_lbfgs_shared_affine",
        )

    def apply(self, fit: Any, logits: Any) -> Any:
        values = torch.as_tensor(logits)
        if isinstance(fit, (FactoredCalibrationFit, ProperScoreShrinkageFit)):
            return fit.apply(values)
        if isinstance(fit, AffineCalibrationFit):
            input_dtype = values.dtype
            # Preserve the float64 point certified by the factored core.
            # Fits from the legacy or unconstrained branches keep their dtype.
            if fit.receipt is not None and fit.scale.dtype == torch.float64:
                values = values.to(dtype=torch.float64)
            if fit.target_class_id is not None:
                result = values.clone()
                class_id = int(fit.target_class_id)
                center = (
                    torch.zeros(1, device=values.device, dtype=values.dtype)
                    if fit.input_center is None
                    else torch.as_tensor(
                        fit.input_center, device=values.device, dtype=values.dtype
                    )
                )
                result[:, class_id] = (
                    (values[:, class_id] - center.reshape(-1)[0])
                    * fit.scale.to(values).reshape(-1)[0]
                    + fit.offset.to(values).reshape(-1)[0]
                )
                # Round only after evaluating the certified affine.
                return result.to(dtype=input_dtype)
            return values * fit.scale.to(values) + fit.offset.to(values)
        if isinstance(fit, ReadoutAffineCalibrationFit):
            center = fit.input_center.to(values)
            centered = values - (
                center.reshape(-1)[0]
                if int(center.numel()) == 1
                else center.reshape(1, 1, -1)
                if values.ndim == 3
                else center.reshape(1, -1)
            )
            scale = fit.readout_scale.to(values)
            offset = fit.readout_offset.to(values)
            if values.ndim == 3 and scale.ndim == 2:
                return centered * scale.unsqueeze(1) + offset.unsqueeze(1)
            return centered * scale + offset
        if hasattr(fit, "readout_scale") and hasattr(fit, "readout_offset"):
            return values * fit.readout_scale.to(values) + fit.readout_offset.to(values)
        raise TypeError("unsupported calibration fit object")


__all__ = [
    "AffineCalibrationFit",
    "CanonicalCalibrationBackend",
    "ReadoutAffineCalibrationFit",
]
