"""Pure target-batched VJP and route-reduction numerics.

This module is the scientific numerical core of native measurement.  It owns
neither model construction nor hooks, data loading, artifact writing, Slurm,
nor legacy score heads.  Architecture adapters supply differentiable layer
outputs and replayable layer inputs; this module returns one signed route
score tensor per declared route block.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
import math
import os
from typing import Any, Literal

from mechanistic_systems.core.determinism import (
    configure_deterministic_backend,
    deterministic_backend_record,
)
from mechanistic_systems.core.storage import RouteBlockSpec
from mechanistic_systems.core.operators import (
    RouteReducerKind,
    route_gate_operator,
)
from mechanistic_systems.measurement.queries import QueryCotangentBatch

try:  # Optional for import smoke; required by native measurement execution.
    import torch
except Exception:  # pragma: no cover - environment dependent.
    torch = None  # type: ignore[assignment]


AutogradBackend = Literal["auto", "is_grads_batched", "shared_forward_scalar_loop"]


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("native target-batched measurement requires PyTorch")


def numerics_record() -> dict[str, Any]:
    """Read the effective process precision flags without applying a policy.

    This is an observation at receipt assembly, including any surrounding
    strict context. It does not describe earlier calls under other contexts
    and is excluded from scientific plan and execution-profile identity.
    """

    _require_torch()
    try:
        mkldnn_precision = torch.backends.mkldnn.fp32_precision
    except (AttributeError, RuntimeError):  # older PyTorch builds
        mkldnn_precision = None
    return {
        "schema": "irg_numerics_record_v1",
        "cuda_matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        **deterministic_backend_record(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "mkldnn_fp32_precision": mkldnn_precision,
    }


@contextmanager
def run_forward_numerics() -> Iterator[dict[str, Any]]:
    """Apply the run's TF32 choice to scoring and restore the caller's flags."""

    _require_torch()
    configure_deterministic_backend()
    allow_tf32 = str(os.environ.get("CIRCUIT_ALLOW_TF32", "1")).strip() != "0"
    prior_matmul_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    prior_cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
    prior_matmul_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        yield numerics_record()
    finally:
        torch.set_float32_matmul_precision(prior_matmul_precision)
        torch.backends.cuda.matmul.allow_tf32 = prior_matmul_tf32
        torch.backends.cudnn.allow_tf32 = prior_cudnn_tf32


@contextmanager
def exact_float32_numerics() -> Iterator[dict[str, Any]]:
    """Freeze the accepted ``route_gate_v1`` floating-point policy.

    The route-gate estimand is invariant to runtime packetization, but its
    concrete realization must not silently opt into reduced-precision TF32 or
    autocast. The returned mapping is runtime evidence and is excluded from
    scientific plan identity.
    """

    _require_torch()
    configure_deterministic_backend()
    prior_matmul_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    prior_cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
    prior_matmul_precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    policy = {
        "model_and_gradient_dtype": "float32",
        "autocast_enabled": False,
        "cuda_matmul_tf32": False,
        "cudnn_tf32": False,
        **deterministic_backend_record(),
        "float32_matmul_precision": "highest",
        "host_accumulation_dtype": "float64",
        "durable_dtype": "float32",
    }
    try:
        yield policy
    finally:
        torch.set_float32_matmul_precision(prior_matmul_precision)
        torch.backends.cuda.matmul.allow_tf32 = prior_matmul_tf32
        torch.backends.cudnn.allow_tf32 = prior_cudnn_tf32


@dataclass(frozen=True)
class VJPTelemetry:
    requested_backend: str
    backend_used: str
    query_count: int
    context_count: int
    layer_count: int
    fallback_reason: str = ""


@dataclass(frozen=True)
class CachedQueryVJP:
    """A separate target VJP retained by the campaign census producer."""

    query_ids: tuple[str, ...]
    gradients: tuple[Any, ...]
    telemetry: VJPTelemetry

    def for_query(self, response_matrix: Any, outputs: Sequence[Any], query_batch: Any):
        query_batch.validate(response_matrix)
        if (
            self.query_ids != tuple(query_batch.query_ids)
            or len(outputs) != len(self.gradients)
        ):
            raise ValueError("cached census VJP differs from the query or layer axis")
        for output, gradient in zip(outputs, self.gradients, strict=True):
            if (
                gradient is None
                or tuple(gradient.shape) != (len(self.query_ids), *output.shape)
                or gradient.device != output.device
                or gradient.dtype != output.dtype
            ):
                raise ValueError("cached census VJP differs from its forward capture")
        return self.gradients, replace(
            self.telemetry, context_count=int(response_matrix.shape[0])
        )


ABS_AGGREGATION_MODE_ENV = "IRG_ABS_AGGREGATION_MODE"
DEFAULT_ABS_AGGREGATION_MODE = "exact_abs"

# The two families the absolute score head can belong to. This is the
# |E[aG]| vs E[|aG|] question directly: ``exact_abs`` sums |g|*|a|*|w| PER
# CONTEXT (a first absolute moment, spread-sensitive), while the frontier
# modes take |sum| (a pure mean functional). The walk's ordering key reads
# this head, so the choice is a scientific one, not a runtime detail.
EXACT_ABS_AGGREGATION_MODES = frozenset({"exact_abs", "exact", "sum_abs"})
FRONTIER_ABS_AGGREGATION_MODES = frozenset(
    {
        "frontier_aggregate_abs",
        "canonical_frontier_abs",
        "frontier_abs",
        "frontier_abs_of_sum",
        "abs_of_sum",
    }
)
SUPPORTED_ABS_AGGREGATION_MODES = (
    EXACT_ABS_AGGREGATION_MODES | FRONTIER_ABS_AGGREGATION_MODES
)


def resolve_abs_aggregation_mode(environ: Mapping[str, str] | None = None) -> str:
    """Resolve the absolute-score aggregation mode, defaulting to exact_abs.

    Registered as an ABLATION HANDLE: it selects between two different
    estimators of route activity, so an unrecognized value must fail loudly
    rather than fall back to the default. A silent fallback here would change
    the ordering key with no disclosure -- the same failure shape that makes an
    absent cross-fit family silently revert to the pooled estimator.

    Callers that must pin the mode regardless of environment still pass
    ``abs_aggregation_mode=`` explicitly; only the default consults the
    environment.
    """

    source = os.environ if environ is None else environ
    raw = str(source.get(ABS_AGGREGATION_MODE_ENV, "")).strip()
    if not raw:
        return DEFAULT_ABS_AGGREGATION_MODE
    token = raw.lower()
    if token not in SUPPORTED_ABS_AGGREGATION_MODES:
        raise ValueError(
            f"{ABS_AGGREGATION_MODE_ENV}={raw!r} is not a supported absolute "
            f"aggregation mode; expected one of "
            f"{sorted(SUPPORTED_ABS_AGGREGATION_MODES)}"
        )
    return token


@dataclass(frozen=True)
class CapturedRouteBlock:
    """Architecture-neutral differentiable state for one scored route block."""

    route_block: RouteBlockSpec
    reducer_kind: RouteReducerKind
    output: Any
    input_source: Any
    weight: Any
    scale: float = 1.0
    abs_aggregation_mode: str = field(
        default_factory=lambda: resolve_abs_aggregation_mode()
    )
    reducer_parameters: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require_torch()
        self.route_block.validate()
        if not torch.is_tensor(self.output) or not self.output.requires_grad:
            raise ValueError("captured route output must be a differentiable torch tensor")
        if float(self.scale) == 0.0 or not math.isfinite(float(self.scale)):
            raise ValueError("captured route scale must be finite and nonzero")
        route_gate_operator(self.reducer_kind).validate_capture(self)


@dataclass(frozen=True)
class ReducedRouteScores:
    route_block: RouteBlockSpec
    query_ids: tuple[str, ...]
    signed_scores: Any
    absolute_scores: Any
    reducer_kind: str
    # 4.8 capture tier: optional [context, query, *route_shape] values whose
    # batch sum is bitwise-equal to signed_scores (operator-guaranteed).
    per_image_scores: Any | None = None
    # Query-independent signed contribution to the route's immediate
    # destination, shape [context, *route_shape].  Architecture adapters may
    # subsequently apply a declared downstream affine transport (for example
    # eval-mode BatchNorm's linear scale).
    local_per_image_scores: Any | None = None
    # Optional route-axis ReLU-regime sufficient statistics accumulated from
    # the ephemeral spatial local-drive field before its canonical spatial
    # reduction. These diagnostics never alter the score heads.
    local_regime_statistics: Mapping[str, Any] | None = None

    def validate(self) -> None:
        _require_torch()
        expected = (len(self.query_ids), *self.route_block.shape)
        if tuple(self.signed_scores.shape) != expected:
            raise ValueError(
                f"signed route scores have shape {tuple(self.signed_scores.shape)}, expected {expected}"
            )
        if tuple(self.absolute_scores.shape) != expected:
            raise ValueError(
                f"absolute route scores have shape {tuple(self.absolute_scores.shape)}, expected {expected}"
            )
        if not bool(torch.isfinite(self.signed_scores).all().item()):
            raise ValueError("signed route scores contain non-finite values")
        if not bool(torch.isfinite(self.absolute_scores).all().item()):
            raise ValueError("absolute route scores contain non-finite values")
        if self.per_image_scores is not None:
            expected_rows = (
                int(self.per_image_scores.shape[0]),
                len(self.query_ids),
                *self.route_block.shape,
            )
            if tuple(self.per_image_scores.shape) != expected_rows:
                raise ValueError(
                    "per-image global route rows use an incompatible route/query axis"
                )
        if self.local_per_image_scores is not None:
            expected_local = (
                int(self.local_per_image_scores.shape[0]),
                *self.route_block.shape,
            )
            if tuple(self.local_per_image_scores.shape) != expected_local:
                raise ValueError(
                    "local destination rows use an incompatible route axis"
                )
            if not bool(torch.isfinite(self.local_per_image_scores).all().item()):
                raise ValueError("local destination rows contain non-finite values")
        if self.local_regime_statistics is not None:
            from mechanistic_systems.measurement.local_regime import (
                validate_local_regime_block_statistics,
            )

            validate_local_regime_block_statistics(
                self.local_regime_statistics,
                route_shape=tuple(self.route_block.shape),
            )


def exact_batched_vjp(
    response_matrix: Any,
    layer_outputs: tuple[Any, ...] | list[Any],
    query_batch: QueryCotangentBatch,
    *,
    retain_graph: bool = False,
    backend: AutogradBackend = "auto",
) -> tuple[tuple[Any | None, ...], VJPTelemetry]:
    """Differentiate an ordered query tile against all captured layer outputs."""

    _require_torch()
    query_batch.validate(response_matrix)
    outputs = tuple(layer_outputs)
    if not outputs:
        raise ValueError("target-batched VJP requires at least one captured layer")
    if any(not torch.is_tensor(value) or not value.requires_grad for value in outputs):
        raise ValueError("every captured layer output must be a differentiable tensor")
    requested = str(backend or "auto").strip().lower()
    if requested not in {"auto", "is_grads_batched", "shared_forward_scalar_loop"}:
        raise ValueError(f"unsupported target-batched autograd backend {backend!r}")
    fallback_reason = ""
    if requested in {"auto", "is_grads_batched"}:
        try:
            gradients = torch.autograd.grad(
                response_matrix,
                outputs,
                grad_outputs=query_batch.cotangents,
                retain_graph=retain_graph,
                allow_unused=True,
                is_grads_batched=True,
            )
            telemetry = VJPTelemetry(
                requested_backend=requested,
                backend_used="torch_autograd_is_grads_batched",
                query_count=len(query_batch.query_ids),
                context_count=int(response_matrix.shape[0]),
                layer_count=len(outputs),
            )
            return tuple(gradients), telemetry
        except TypeError as error:
            fallback_reason = f"is_grads_batched_unavailable:{type(error).__name__}"
        except RuntimeError as error:
            message = str(error).lower()
            if not any(token in message for token in ("is_grads_batched", "vmap", "batched")):
                raise
            fallback_reason = f"is_grads_batched_runtime_fallback:{type(error).__name__}"
        if requested == "is_grads_batched":
            raise RuntimeError(
                "explicit is_grads_batched backend failed and may not fall back: "
                + (fallback_reason or "unknown_failure")
            )

    stacked: list[list[Any | None]] = [[] for _ in outputs]
    query_count = len(query_batch.query_ids)
    for query_index in range(query_count):
        keep = bool(retain_graph or query_index < query_count - 1)
        gradients = torch.autograd.grad(
            response_matrix,
            outputs,
            grad_outputs=query_batch.cotangents[query_index],
            retain_graph=keep,
            allow_unused=True,
        )
        for layer_index, gradient in enumerate(gradients):
            stacked[layer_index].append(gradient)
    result: list[Any | None] = []
    for gradients in stacked:
        if any(gradient is None for gradient in gradients):
            result.append(None)
        else:
            result.append(torch.stack(tuple(gradients), dim=0))
    telemetry = VJPTelemetry(
        requested_backend=requested,
        backend_used="shared_forward_scalar_vjp_loop",
        query_count=query_count,
        context_count=int(response_matrix.shape[0]),
        layer_count=len(outputs),
        fallback_reason=fallback_reason,
    )
    return tuple(result), telemetry


def reduce_route_gradients(
    captured: CapturedRouteBlock,
    gradients: Any,
    *,
    query_ids: tuple[str, ...],
    reducer_backend: Any | None = None,
    per_image: bool = False,
    local_attribution: bool = False,
    local_regime: bool = False,
) -> ReducedRouteScores:
    """Reduce exact layer-output gradients into canonical signed route scores."""

    _require_torch()
    captured.validate()
    if not torch.is_tensor(gradients):
        raise ValueError("route gradients must be a torch tensor")
    if int(gradients.shape[0]) != len(query_ids):
        raise ValueError("route gradient query axis does not match the declared query tile")
    if tuple(gradients.shape[1:]) != tuple(captured.output.shape):
        raise ValueError(
            f"route gradients have shape {tuple(gradients.shape[1:])}, "
            f"expected {tuple(captured.output.shape)}"
        )
    resolved_backend = reducer_backend
    if resolved_backend is not None:
        resolve_backend = getattr(resolved_backend, "resolve", None)
        if callable(resolve_backend):
            resolved_backend = resolve_backend(
                capability="route_reducer",
                route_family=captured.reducer_kind,
                device_type=str(gradients.device.type),
                dtype=str(gradients.dtype).removeprefix("torch."),
            )
    parameters = {
        **dict(captured.reducer_parameters),
        "scale": float(captured.scale),
        "abs_aggregation_mode": captured.abs_aggregation_mode,
        "retain_local_attribution": bool(local_attribution),
    }
    per_image_scores = None
    local_per_image_scores = None
    local_regime_statistics = None
    if local_regime:
        if captured.reducer_kind != "conv_channel_pair":
            raise ValueError(
                "local activation-regime diagnostics currently require "
                "convolutional channel-pair routes"
            )
        try:
            baseline = parameters["local_regime_preactivation"]
            route_scale = parameters["local_regime_route_scale"]
            geometry_hash = str(parameters["local_regime_geometry_hash"])
        except KeyError as error:
            raise ValueError(
                "local regime diagnostics require architecture-bound "
                "destination preactivation geometry"
            ) from error
        from mechanistic_systems.measurement.local_regime import make_local_regime_tee

        parameters["local_regime_tee"] = make_local_regime_tee(
            baseline_preactivation=baseline,
            route_scale=route_scale,
            route_shape=tuple(map(int, captured.route_block.shape)),
            geometry_hash=geometry_hash,
            activation_kind=str(
                parameters.get("local_regime_activation_kind", "relu")
            ),
            local_active_tolerance=float(
                parameters.get("local_regime_active_tolerance", 0.0)
            ),
            boundary_epsilon=float(
                parameters.get("local_regime_boundary_epsilon", 1e-8)
            ),
            route_chunk_size=int(
                parameters.get("local_regime_route_chunk_size", 16)
            ),
        )
    fused_context_reducer = (
        None
        if resolved_backend is None
        else getattr(resolved_backend, "reduce_vjp_with_context", None)
    )
    if per_image and callable(fused_context_reducer):
        fused_result = fused_context_reducer(
            route_family=captured.reducer_kind,
            activation=(
                captured.output
                if captured.reducer_kind == "residual_stream_gate"
                else captured.input_source
            ),
            output_gradient=gradients,
            weight=captured.weight,
            metadata=parameters,
        )
        if len(fused_result) not in {4, 5}:
            raise RuntimeError("fused route reducer returned an invalid result")
        (
            signed_scores,
            absolute_scores,
            per_image_scores,
            fused_local_rows,
        ) = fused_result[:4]
        if len(fused_result) == 5:
            local_regime_statistics = fused_result[4]
        if local_regime and local_regime_statistics is None:
            raise RuntimeError(
                "fused reducer omitted requested local regime statistics"
            )
        if local_attribution:
            if fused_local_rows is None:
                raise RuntimeError(
                    "fused reducer omitted requested local-attribution rows"
                )
            local_per_image_scores = fused_local_rows
    elif resolved_backend is None:
        signed_scores, absolute_scores = route_gate_operator(
            captured.reducer_kind
        ).reduce_vjp(captured, gradients)
    else:
        signed_scores, absolute_scores = resolved_backend.reduce_vjp(
            route_family=captured.reducer_kind,
            activation=(
                captured.output
                if captured.reducer_kind == "residual_stream_gate"
                else captured.input_source
            ),
            output_gradient=gradients,
            weight=captured.weight,
            metadata=parameters,
        )
    if per_image and per_image_scores is None:
        per_image_scores = route_gate_operator(
            captured.reducer_kind
        ).reduce_vjp_per_image(captured, gradients)
        if local_attribution:
            local_per_image_scores = route_gate_operator(
                captured.reducer_kind
            ).local_target_contribution_per_context(captured)
    reduced = ReducedRouteScores(
        route_block=captured.route_block,
        query_ids=tuple(query_ids),
        signed_scores=signed_scores,
        absolute_scores=absolute_scores,
        reducer_kind=captured.reducer_kind,
        per_image_scores=per_image_scores,
        local_per_image_scores=local_per_image_scores,
        local_regime_statistics=local_regime_statistics,
    )
    reduced.validate()
    return reduced


def score_captured_route_blocks(
    response_matrix: Any,
    query_batch: QueryCotangentBatch,
    captured_blocks: tuple[CapturedRouteBlock, ...] | list[CapturedRouteBlock],
    *,
    backend: AutogradBackend = "auto",
    retain_graph: bool = False,
    reducer_backend: Any | None = None,
    profiler: Any | None = None,
    per_image: bool = False,
    local_attribution: bool = False,
    local_regime: bool = False,
    cached_vjp: CachedQueryVJP | None = None,
) -> tuple[tuple[ReducedRouteScores, ...], VJPTelemetry]:
    """Run one exact query-tile VJP and reduce every captured route block."""

    blocks = tuple(captured_blocks)
    if not blocks:
        raise ValueError("route measurement requires at least one captured block")
    if len({block.route_block.layer_id for block in blocks}) != len(blocks):
        raise ValueError("captured route layer IDs must be unique within a model pass")
    with (
        profiler.phase("measurement.autograd_vjp")
        if profiler is not None
        else nullcontext()
    ):
        if cached_vjp is None:
            gradients, telemetry = exact_batched_vjp(
                response_matrix,
                [block.output for block in blocks],
                query_batch,
                retain_graph=retain_graph,
                backend=backend,
            )
        else:
            gradients, telemetry = cached_vjp.for_query(
                response_matrix, [block.output for block in blocks], query_batch
            )
    reduced: list[ReducedRouteScores] = []
    for block, gradient in zip(blocks, gradients, strict=True):
        if gradient is None:
            raise RuntimeError(
                f"query response is disconnected from route block {block.route_block.layer_id!r}"
            )
        with (
            profiler.phase(f"measurement.route_reduce.{block.reducer_kind}")
            if profiler is not None
            else nullcontext()
        ):
            try:
                reduced.append(
                    reduce_route_gradients(
                        block,
                        gradient,
                        query_ids=query_batch.query_ids,
                        reducer_backend=reducer_backend,
                        per_image=per_image,
                        local_attribution=local_attribution,
                        local_regime=(
                            local_regime
                            and "local_regime_preactivation"
                            in block.reducer_parameters
                        ),
                    )
                )
            except NotImplementedError:
                # This route kind has no per-image reducer (e.g. the
                # linear_weight readout). Per-image retention gracefully
                # OMITS such a block from the factor rather than
                # approximating or crashing — the pooled scores are still
                # exact; per_image_scores stays None and the factor simply
                # does not cover this layer. Only reached when per_image
                # is requested; the reduced-only path never raises here.
                if not per_image:
                    raise
                reduced.append(
                    reduce_route_gradients(
                        block,
                        gradient,
                        query_ids=query_batch.query_ids,
                        reducer_backend=reducer_backend,
                        per_image=False,
                        local_attribution=False,
                    )
                )
    return tuple(reduced), telemetry


__all__ = [
    "AutogradBackend",
    "CapturedRouteBlock",
    "CachedQueryVJP",
    "ReducedRouteScores",
    "RouteReducerKind",
    "VJPTelemetry",
    "exact_float32_numerics",
    "run_forward_numerics",
    "exact_batched_vjp",
    "reduce_route_gradients",
    "score_captured_route_blocks",
]
