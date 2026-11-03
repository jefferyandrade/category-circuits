"""Target-batched route scoring with exact conserving activation geometry.

This is the numerical bridge between captured post-activation cotangents and
the established architecture-specific route reducers.  It deliberately does
not infer an activation from a layer name: every affine route block must carry
an explicit, aligned activation capture.  ReLU blocks replace the incumbent
preactivation cotangent by ``G * phi(a,b)``; identity blocks use their output
cotangent unchanged.

The returned route rows remain *current-contribution* oriented.  A pruning
consumer applies the registered negative sign to obtain positive-removal work
and applies the current hard-route gate before aggregation.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Any, Final, Literal, Protocol

from mechanistic_systems.core.numerics import (
    AutogradBackend,
    CapturedRouteBlock,
    CachedQueryVJP,
    ReducedRouteScores,
    VJPTelemetry,
    exact_batched_vjp,
    reduce_route_gradients,
)
from mechanistic_systems.core.storage import RouteBlockSpec
from mechanistic_systems.measurement.queries import QueryCotangentBatch

from .modern_v1_conserving_attribution import (
    CONSERVING_ATTRIBUTION_EQUATION_ID,
    POSITIVE_REMOVAL_ORIENTATION_ID,
    conserving_pruning_loss_cotangent,
    conserving_relu_cotangent,
)
from .modern_v1_gate_disclosure import (
    GATE_DISCLOSURE_ROLE,
    GATE_DISCLOSURE_SCHEMA,
    V1GateDisclosureChunk,
    V1GateDisclosureMemoryReceipt,
    compute_frozen_gate_removal_disclosure,
)
from mechanistic_systems.measurement.pruning_loss import FrozenPruningLossPath


CONSERVING_ROUTE_SCORE_SCHEMA: Final = (
    "partition_free_v1_conserving_route_score_v1"
)
CONSERVING_ROUTE_SCORE_ID: Final = (
    "post_activation_cotangent_relu_path_fraction_route_contraction_v1"
)
PLAIN_UNCORRECTED_ROUTE_SCORE_ID: Final = (
    "post_activation_cotangent_plain_route_contraction_v1"
)
LITERAL_GATE_JACOBIAN_ROUTE_SCORE_ID: Final = (
    "post_activation_cotangent_current_relu_derivative_route_contraction_v1"
)
LITERAL_GATE_JACOBIAN_EQUATION_ID: Final = (
    "negative_G_times_indicator_preactivation_positive_times_route_drive_v1"
)
RELU_ZERO_SUBGRADIENT_POLICY_ID: Final = (
    "pytorch_relu_derivative_zero_at_preactivation_zero_v1"
)

ActivationKind = Literal["relu", "identity"]


def _torch() -> Any:
    try:
        import torch
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
        raise RuntimeError(
            "conserving route scoring requires PyTorch"
        ) from exc
    return torch


def literal_relu_gate_jacobian_cotangent(
    post_activation_cotangent: Any,
    preactivation: Any,
) -> Any:
    """Return the literal current-chart ReLU gate cotangent.

    The conserving Aumann--Shapley cotangent ``G*phi`` allocates finite
    evidence from the bias baseline to the current activation.  It is not, in
    general, the derivative of the raw response with respect to an
    infinitesimal route gate at the current chart.  The latter is
    ``G*1[a>0]``.  PyTorch's registered ReLU subgradient at ``a == 0`` is zero;
    this function makes that boundary convention explicit and testable.
    """

    torch = _torch()
    if (
        not isinstance(post_activation_cotangent, torch.Tensor)
        or not isinstance(preactivation, torch.Tensor)
        or tuple(post_activation_cotangent.shape[2:])
        != tuple(preactivation.shape[1:])
        or int(post_activation_cotangent.shape[1]) != int(preactivation.shape[0])
        or post_activation_cotangent.dtype != preactivation.dtype
        or post_activation_cotangent.device != preactivation.device
        or not bool(torch.isfinite(post_activation_cotangent).all().item())
        or not bool(torch.isfinite(preactivation).all().item())
    ):
        raise ValueError(
            "literal gate-Jacobian cotangent must be finite and capture aligned"
        )
    active = preactivation.gt(0).unsqueeze(0)
    return torch.where(
        active,
        post_activation_cotangent,
        torch.zeros((), dtype=post_activation_cotangent.dtype, device=post_activation_cotangent.device),
    )


@dataclass(frozen=True, slots=True)
class V1ConservingActivationCapture:
    """One immediate post-activation output and its bias-only baseline."""

    layer_id: str
    activation_kind: ActivationKind
    post_activation_output: Any
    bias_baseline: Any
    capture_source_identity: str
    schema: str = CONSERVING_ROUTE_SCORE_SCHEMA

    def __post_init__(self) -> None:
        if type(self.layer_id) is not str or not self.layer_id:
            raise ValueError("activation capture layer_id must be nonempty")
        if self.activation_kind not in {"relu", "identity"}:
            raise ValueError("unsupported conserving activation kind")
        if (
            type(self.capture_source_identity) is not str
            or not self.capture_source_identity
        ):
            raise ValueError(
                "activation capture source identity must be nonempty"
            )
        if self.schema != CONSERVING_ROUTE_SCORE_SCHEMA:
            raise ValueError("unsupported conserving route-score schema")

    def validate_for(self, block: CapturedRouteBlock) -> None:
        torch = _torch()
        block.validate()
        post = self.post_activation_output
        bias = self.bias_baseline
        pre = block.output
        if self.layer_id != block.route_block.layer_id:
            raise ValueError(
                "activation capture and route block layer IDs differ"
            )
        if (
            not isinstance(post, torch.Tensor)
            or not isinstance(bias, torch.Tensor)
            or not post.requires_grad
            or post.shape != pre.shape
            or post.dtype != pre.dtype
            or post.device != pre.device
            or bias.dtype != pre.dtype
            or bias.device != pre.device
            or not bool(torch.isfinite(post).all().item())
            or not bool(torch.isfinite(bias).all().item())
        ):
            raise ValueError(
                "activation capture must be finite, differentiable, and "
                "aligned "
                "with its preactivation"
            )
        try:
            torch.broadcast_shapes(tuple(pre.shape), tuple(bias.shape))
        except RuntimeError as exc:
            raise ValueError(
                "bias baseline does not broadcast over the preactivation"
            ) from exc
        expected = torch.relu(pre) if self.activation_kind == "relu" else pre
        if not bool(torch.equal(post, expected)):
            raise ValueError(
                "activation capture is not the exact immediate activation of "
                "the captured affine output"
            )


@dataclass(frozen=True, slots=True)
class V1GateDisclosureBatchDescriptor:
    """Typed identity for one transactional target-scale disclosure batch."""

    context_ids: tuple[str, ...]
    query_ids: tuple[str, ...]
    route_blocks: tuple[RouteBlockSpec, ...]
    target_scale_identity: str
    route_count: int
    ordering_key_eligible: bool = False
    semantic_role: str = GATE_DISCLOSURE_ROLE
    schema: str = GATE_DISCLOSURE_SCHEMA

    def __post_init__(self) -> None:
        if (
            not self.context_ids
            or len(set(self.context_ids)) != len(self.context_ids)
            or not self.query_ids
            or len(set(self.query_ids)) != len(self.query_ids)
            or not self.route_blocks
            or type(self.target_scale_identity) is not str
            or not self.target_scale_identity
            or isinstance(self.route_count, bool)
            or int(self.route_count) <= 0
        ):
            raise ValueError("gate-disclosure batch identity is incomplete")
        expected_offset = 0
        for block in self.route_blocks:
            if not isinstance(block, RouteBlockSpec):
                raise TypeError(
                    "gate-disclosure batch routes must be typed block specs"
                )
            block.validate()
            if int(block.offset) != expected_offset:
                raise ValueError(
                    "gate-disclosure route blocks are not globally contiguous"
                )
            expected_offset = int(block.stop)
        if sum(block.route_count for block in self.route_blocks) != int(
            self.route_count
        ):
            raise ValueError("gate-disclosure route blocks do not close")
        if self.ordering_key_eligible:
            raise PermissionError(
                "gate-removal disclosure cannot claim ordering-key authority"
            )
        if self.semantic_role != GATE_DISCLOSURE_ROLE:
            raise ValueError("unsupported gate-disclosure semantic role")
        if self.schema != GATE_DISCLOSURE_SCHEMA:
            raise ValueError("unsupported gate-disclosure schema")


@dataclass(frozen=True, slots=True)
class V1TargetScaleGateDisclosureChunk:
    """Bounded target-scale disclosure with explicit axis identity."""

    context_ids: tuple[str, ...]
    query_ids: tuple[str, ...]
    block_index: int
    global_start: int
    global_stop: int
    target_scale_identity: str
    chunk: V1GateDisclosureChunk
    ordering_key_eligible: bool = False
    semantic_role: str = GATE_DISCLOSURE_ROLE
    schema: str = GATE_DISCLOSURE_SCHEMA

    def __post_init__(self) -> None:
        values = self.chunk.values
        if (
            not self.context_ids
            or len(set(self.context_ids)) != len(self.context_ids)
            or not self.query_ids
            or len(set(self.query_ids)) != len(self.query_ids)
            or isinstance(self.block_index, bool)
            or int(self.block_index) < 0
            or isinstance(self.global_start, bool)
            or isinstance(self.global_stop, bool)
            or int(self.global_start) < 0
            or int(self.global_stop) <= int(self.global_start)
            or int(self.global_stop) - int(self.global_start)
            != int(self.chunk.flat_stop) - int(self.chunk.flat_start)
            or int(self.global_start)
            != int(self.chunk.route_block.offset) + int(self.chunk.flat_start)
            or int(self.global_stop)
            != int(self.chunk.route_block.offset) + int(self.chunk.flat_stop)
            or int(values.shape[0]) != len(self.context_ids)
            or int(values.shape[1]) != len(self.query_ids)
            or type(self.target_scale_identity) is not str
            or not self.target_scale_identity
        ):
            raise ValueError("target-scale gate-disclosure chunk axes differ")
        if self.ordering_key_eligible:
            raise PermissionError(
                "gate-removal disclosure cannot claim ordering-key authority"
            )
        if self.semantic_role != GATE_DISCLOSURE_ROLE:
            raise ValueError("unsupported gate-disclosure semantic role")
        if self.schema != GATE_DISCLOSURE_SCHEMA:
            raise ValueError("unsupported gate-disclosure schema")


class V1GateDisclosureStreamConsumer(Protocol):
    """Transactional consumer for ephemeral target-scale disclosure chunks."""

    def begin_gate_disclosure_batch(
        self,
        descriptor: V1GateDisclosureBatchDescriptor,
    ) -> None: ...

    def consume_gate_disclosure_chunk(
        self,
        chunk: V1TargetScaleGateDisclosureChunk,
    ) -> None: ...

    def end_gate_disclosure_batch(
        self,
        descriptor: V1GateDisclosureBatchDescriptor,
        memory_receipts: tuple[V1GateDisclosureMemoryReceipt, ...],
    ) -> None: ...

    def abort_gate_disclosure_batch(
        self,
        descriptor: V1GateDisclosureBatchDescriptor,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class V1GateDisclosureRequest:
    """Request a separate frozen-cotangent diagnostic from the same VJP.

    ``context_query_multipliers`` transforms the raw query cotangent to the
    declared target scale.  It does not affect the conserving score rows.
    """

    context_ids: tuple[str, ...]
    context_query_multipliers: Any
    target_scale_identity: str
    consumer: V1GateDisclosureStreamConsumer
    route_chunk_size: int = 4096
    linear_route_chunk_size: int | None = None
    ordering_key_eligible: bool = False
    semantic_role: str = GATE_DISCLOSURE_ROLE
    schema: str = GATE_DISCLOSURE_SCHEMA

    def validate_for(
        self,
        query_batch: QueryCotangentBatch,
        captured_blocks: Sequence[CapturedRouteBlock],
    ) -> None:
        torch = _torch()
        blocks = tuple(captured_blocks)
        multipliers = self.context_query_multipliers
        context_count = int(blocks[0].output.shape[0]) if blocks else 0
        query_count = len(query_batch.query_ids)
        if (
            not blocks
            or len(self.context_ids) != context_count
            or len(set(self.context_ids)) != len(self.context_ids)
            or not isinstance(multipliers, torch.Tensor)
            or tuple(multipliers.shape) != (query_count, context_count)
            or multipliers.dtype != blocks[0].output.dtype
            or multipliers.device != blocks[0].output.device
            or not bool(torch.isfinite(multipliers).all().item())
            or type(self.target_scale_identity) is not str
            or not self.target_scale_identity
            or isinstance(self.route_chunk_size, bool)
            or int(self.route_chunk_size) <= 0
            or (
                self.linear_route_chunk_size is not None
                and (
                    isinstance(self.linear_route_chunk_size, bool)
                    or int(self.linear_route_chunk_size) <= 0
                )
            )
        ):
            raise ValueError("gate-disclosure request is not capture aligned")
        for block in blocks:
            if (
                int(block.output.shape[0]) != context_count
                or block.output.dtype != multipliers.dtype
                or block.output.device != multipliers.device
            ):
                raise ValueError(
                    "gate-disclosure blocks differ in context, dtype, or "
                    "device"
                )
        required = (
            "begin_gate_disclosure_batch",
            "consume_gate_disclosure_chunk",
            "end_gate_disclosure_batch",
            "abort_gate_disclosure_batch",
        )
        if any(
            not callable(getattr(self.consumer, name, None))
            for name in required
        ):
            raise TypeError(
                "gate-disclosure consumer lacks the transactional protocol"
            )
        if self.ordering_key_eligible:
            raise PermissionError(
                "gate-removal disclosure cannot claim ordering-key authority"
            )
        if self.semantic_role != GATE_DISCLOSURE_ROLE:
            raise ValueError("unsupported gate-disclosure semantic role")
        if self.schema != GATE_DISCLOSURE_SCHEMA:
            raise ValueError("unsupported gate-disclosure schema")


@dataclass(frozen=True, slots=True)
class V1ConservingRouteScoreResult:
    """Ephemeral corrected route scores plus typed numerical semantics."""

    scores: tuple[ReducedRouteScores, ...]
    telemetry: VJPTelemetry
    activation_kinds: tuple[ActivationKind, ...]
    plain_uncorrected_scores: tuple[ReducedRouteScores, ...] | None = None
    literal_jacobian_scores: tuple[ReducedRouteScores, ...] | None = None
    gate_disclosure_memory_receipts: (
        tuple[V1GateDisclosureMemoryReceipt, ...] | None
    ) = None
    route_score_id: str = CONSERVING_ROUTE_SCORE_ID
    attribution_equation_id: str = CONSERVING_ATTRIBUTION_EQUATION_ID
    positive_removal_orientation_id: str = POSITIVE_REMOVAL_ORIENTATION_ID
    schema: str = CONSERVING_ROUTE_SCORE_SCHEMA

    def __post_init__(self) -> None:
        if not self.scores or len(self.scores) != len(self.activation_kinds):
            raise ValueError("conserving route-score result axes do not close")
        if (
            self.plain_uncorrected_scores is not None
            and len(self.plain_uncorrected_scores) != len(self.scores)
        ):
            raise ValueError(
                "plain and conserving route-score result axes do not close"
            )
        if (
            self.literal_jacobian_scores is not None
            and len(self.literal_jacobian_scores) != len(self.scores)
        ):
            raise ValueError(
                "literal-Jacobian and conserving route-score axes do not close"
            )
        if (
            self.gate_disclosure_memory_receipts is not None
            and len(self.gate_disclosure_memory_receipts) != len(self.scores)
        ):
            raise ValueError(
                "gate-disclosure receipts do not cover the route blocks"
            )
        if self.route_score_id != CONSERVING_ROUTE_SCORE_ID:
            raise ValueError("unsupported conserving route-score identity")
        if self.attribution_equation_id != CONSERVING_ATTRIBUTION_EQUATION_ID:
            raise ValueError("unsupported conserving attribution equation")
        if (
            self.positive_removal_orientation_id
            != POSITIVE_REMOVAL_ORIENTATION_ID
        ):
            raise ValueError("unsupported positive-removal orientation")
        if self.schema != CONSERVING_ROUTE_SCORE_SCHEMA:
            raise ValueError("unsupported conserving route-score schema")


def _profile_phase(profiler: Any | None, name: str) -> Any:
    if profiler is None:
        return nullcontext()
    return profiler.phase(name)


def _prefixed_query_ids(role: str, query_ids: Sequence[str]) -> tuple[str, ...]:
    return tuple(f"{role}:{query_id}" for query_id in query_ids)


def _slice_reduced_query_axis(
    reduced: ReducedRouteScores,
    *,
    start: int,
    stop: int,
    query_ids: Sequence[str],
    retain_local_rows: bool,
) -> ReducedRouteScores:
    """Return one ordinary score object from a fused independent query tile."""

    per_image_scores = (
        None
        if reduced.per_image_scores is None
        else reduced.per_image_scores[:, start:stop]
    )
    sliced = ReducedRouteScores(
        route_block=reduced.route_block,
        query_ids=tuple(query_ids),
        signed_scores=reduced.signed_scores[start:stop],
        absolute_scores=reduced.absolute_scores[start:stop],
        reducer_kind=reduced.reducer_kind,
        per_image_scores=per_image_scores,
        local_per_image_scores=(
            reduced.local_per_image_scores if retain_local_rows else None
        ),
        local_regime_statistics=(
            reduced.local_regime_statistics if retain_local_rows else None
        ),
    )
    sliced.validate()
    return sliced


def _captured_with_profiler(
    block: CapturedRouteBlock,
    profiler: Any | None,
) -> CapturedRouteBlock:
    if profiler is None:
        return block
    return replace(
        block,
        reducer_parameters={
            **dict(block.reducer_parameters),
            "profiler": profiler,
        },
    )


def score_captured_conserving_route_blocks(
    response_matrix: Any,
    query_batch: QueryCotangentBatch,
    captured_blocks: tuple[CapturedRouteBlock, ...] | list[CapturedRouteBlock],
    activation_captures: tuple[V1ConservingActivationCapture, ...]
    | list[V1ConservingActivationCapture],
    *,
    backend: AutogradBackend = "auto",
    retain_graph: bool = False,
    reducer_backend: Any | None = None,
    profiler: Any | None = None,
    per_image: bool = False,
    local_attribution: bool = False,
    gate_disclosure: V1GateDisclosureRequest | None = None,
    include_plain_uncorrected: bool = False,
    include_literal_jacobian: bool = False,
    cached_vjp: CachedQueryVJP | None = None,
    loss_path: FrozenPruningLossPath | None = None,
) -> V1ConservingRouteScoreResult:
    """Score all route blocks from one target-batched post-activation VJP.

    ``include_plain_uncorrected`` is a diagnostic-only dual reduction.  It
    contracts the *same* raw post-activation cotangent from the one VJP with
    the same route contribution ``c`` before the conserving ``phi`` factor is
    applied.  Thus per-context rows expose the controlled pair ``G*c`` and
    ``G*phi*c`` without a second forward, capture, query evaluation, or VJP.
    The option does not alter the canonical conserving ``scores`` field.
    ``loss_path`` composes the frozen dense calibrated loss with each unit's
    registered bias-to-activation target-affine path before route contraction.
    """

    if type(include_plain_uncorrected) is not bool:
        raise TypeError("include_plain_uncorrected must be bool")
    if type(include_literal_jacobian) is not bool:
        raise TypeError("include_literal_jacobian must be bool")
    if include_plain_uncorrected and local_attribution:
        raise ValueError(
            "diagnostic plain-score pairing does not implement local attribution"
        )
    blocks = tuple(captured_blocks)
    captures = tuple(activation_captures)
    if not blocks or len(blocks) != len(captures):
        raise ValueError(
            "every captured route block requires one activation capture"
        )
    if len({capture.layer_id for capture in captures}) != len(captures):
        raise ValueError(
            "conserving activation capture layer IDs must be unique"
        )
    for block, capture in zip(blocks, captures, strict=True):
        capture.validate_for(block)
    if gate_disclosure is not None:
        gate_disclosure.validate_for(query_batch, blocks)
    with _profile_phase(profiler, "measurement.conserving_autograd_vjp"):
        gradients, telemetry = (exact_batched_vjp(
            response_matrix,
            [capture.post_activation_output for capture in captures],
            query_batch,
            retain_graph=retain_graph,
            backend=backend,
        ) if cached_vjp is None else cached_vjp.for_query(
            response_matrix,
            [capture.post_activation_output for capture in captures],
            query_batch,
        ))
    reduced: list[ReducedRouteScores] = []
    plain_reduced: list[ReducedRouteScores] | None = (
        [] if include_plain_uncorrected else None
    )
    jacobian_reduced: list[ReducedRouteScores] | None = (
        [] if include_literal_jacobian else None
    )
    post_gradients: list[Any] = []
    for block, capture, post_gradient in zip(
        blocks,
        captures,
        gradients,
        strict=True,
    ):
        if post_gradient is None:
            raise RuntimeError(
                f"query response is disconnected from post-activation layer "
                f"{capture.layer_id!r}"
            )
        post_gradients.append(post_gradient)
        reduce_block = _captured_with_profiler(block, profiler)
        with _profile_phase(
            profiler,
            f"measurement.dual_chart_cotangent_prepare.{capture.activation_kind}",
        ):
            literal_cotangent = (
                literal_relu_gate_jacobian_cotangent(
                    post_gradient,
                    block.output,
                )
                if capture.activation_kind == "relu"
                else post_gradient
            )
            if loss_path is not None:
                corrected_gradient = conserving_pruning_loss_cotangent(
                    post_gradient, block.output, capture.bias_baseline, loss_path,
                    activation_kind=capture.activation_kind,
                )
            elif capture.activation_kind == "relu":
                corrected_gradient = conserving_relu_cotangent(
                    post_gradient,
                    block.output,
                    capture.bias_baseline,
                )
            else:
                corrected_gradient = post_gradient

        stream_specs: list[tuple[str, Any, bool]] = []
        if plain_reduced is not None:
            stream_specs.append(("plain_uncorrected", post_gradient, False))
        if jacobian_reduced is not None:
            stream_specs.append(("literal_jacobian", literal_cotangent, False))
        stream_specs.append(("corrected", corrected_gradient, local_attribution))

        if len(stream_specs) > 1:
            combined_query_ids: list[str] = []
            tensors = []
            for role, tensor, _retain_local_rows in stream_specs:
                combined_query_ids.extend(
                    _prefixed_query_ids(role, query_batch.query_ids)
                )
                tensors.append(tensor)
            torch = _torch()
            combined_gradient = torch.cat(tuple(tensors), dim=0)
            with _profile_phase(
                profiler,
                f"measurement.fused_dual_chart_route_reduce.{block.reducer_kind}",
            ):
                combined_reduced = reduce_route_gradients(
                    reduce_block,
                    combined_gradient,
                    query_ids=tuple(combined_query_ids),
                    reducer_backend=reducer_backend,
                    per_image=per_image,
                    local_attribution=local_attribution,
                    local_regime=False,
                )
            query_count = len(query_batch.query_ids)
            cursor = 0
            for role, _tensor, retain_local_rows in stream_specs:
                part = _slice_reduced_query_axis(
                    combined_reduced,
                    start=cursor,
                    stop=cursor + query_count,
                    query_ids=query_batch.query_ids,
                    retain_local_rows=retain_local_rows,
                )
                if role == "plain_uncorrected":
                    assert plain_reduced is not None
                    plain_reduced.append(part)
                elif role == "literal_jacobian":
                    assert jacobian_reduced is not None
                    jacobian_reduced.append(part)
                else:
                    reduced.append(part)
                cursor += query_count
        else:
            with _profile_phase(
                profiler,
                f"measurement.conserving_route_reduce.{block.reducer_kind}",
            ):
                reduced.append(
                    reduce_route_gradients(
                        reduce_block,
                        corrected_gradient,
                        query_ids=query_batch.query_ids,
                        reducer_backend=reducer_backend,
                        per_image=per_image,
                        local_attribution=local_attribution,
                        local_regime=False,
                    )
                )
    disclosure_receipts: (
        tuple[V1GateDisclosureMemoryReceipt, ...] | None
    ) = None
    if gate_disclosure is not None:
        route_blocks = tuple(block.route_block for block in blocks)
        descriptor = V1GateDisclosureBatchDescriptor(
            context_ids=tuple(gate_disclosure.context_ids),
            query_ids=tuple(query_batch.query_ids),
            route_blocks=route_blocks,
            target_scale_identity=gate_disclosure.target_scale_identity,
            route_count=sum(block.route_count for block in route_blocks),
        )
        consumer = gate_disclosure.consumer
        receipts: list[V1GateDisclosureMemoryReceipt] = []
        begin_attempted = False
        try:
            begin_attempted = True
            consumer.begin_gate_disclosure_batch(descriptor)
            global_offset = 0
            multipliers = gate_disclosure.context_query_multipliers
            for block_index, (block, capture, post_gradient) in enumerate(
                zip(blocks, captures, post_gradients, strict=True)
            ):
                expanded_multipliers = multipliers.reshape(
                    int(multipliers.shape[0]),
                    int(multipliers.shape[1]),
                    *([1] * (post_gradient.ndim - 2)),
                )
                target_scale_cotangent = post_gradient * expanded_multipliers

                def consume_chunk(
                    chunk: V1GateDisclosureChunk,
                    *,
                    _block_index: int = block_index,
                    _global_offset: int = global_offset,
                ) -> None:
                    consumer.consume_gate_disclosure_chunk(
                        V1TargetScaleGateDisclosureChunk(
                            context_ids=descriptor.context_ids,
                            query_ids=descriptor.query_ids,
                            block_index=_block_index,
                            global_start=(
                                _global_offset + int(chunk.flat_start)
                            ),
                            global_stop=_global_offset + int(chunk.flat_stop),
                            target_scale_identity=(
                                descriptor.target_scale_identity
                            ),
                            chunk=chunk,
                        )
                    )

                moments = compute_frozen_gate_removal_disclosure(
                    block,
                    target_scale_cotangent,
                    block.output,
                    activation_kind=capture.activation_kind,
                    route_chunk_size=(
                        int(gate_disclosure.linear_route_chunk_size)
                        if block.reducer_kind == "linear_weight"
                        and gate_disclosure.linear_route_chunk_size is not None
                        else gate_disclosure.route_chunk_size
                    ),
                    consumer=consume_chunk,
                    aggregate_dtype=post_gradient.dtype,
                    retain_moments=False,
                )
                receipts.append(moments.memory_receipt)
                global_offset += block.route_block.route_count
            disclosure_receipts = tuple(receipts)
            consumer.end_gate_disclosure_batch(
                descriptor,
                disclosure_receipts,
            )
        except Exception:
            if begin_attempted:
                consumer.abort_gate_disclosure_batch(descriptor)
            raise
    return V1ConservingRouteScoreResult(
        scores=tuple(reduced),
        telemetry=telemetry,
        activation_kinds=tuple(
            capture.activation_kind for capture in captures
        ),
        plain_uncorrected_scores=(
            None if plain_reduced is None else tuple(plain_reduced)
        ),
        literal_jacobian_scores=(
            None if jacobian_reduced is None else tuple(jacobian_reduced)
        ),
        gate_disclosure_memory_receipts=disclosure_receipts,
    )


__all__ = [
    "CONSERVING_ROUTE_SCORE_ID",
    "CONSERVING_ROUTE_SCORE_SCHEMA",
    "LITERAL_GATE_JACOBIAN_EQUATION_ID",
    "LITERAL_GATE_JACOBIAN_ROUTE_SCORE_ID",
    "PLAIN_UNCORRECTED_ROUTE_SCORE_ID",
    "RELU_ZERO_SUBGRADIENT_POLICY_ID",
    "V1ConservingActivationCapture",
    "V1ConservingRouteScoreResult",
    "V1GateDisclosureBatchDescriptor",
    "V1GateDisclosureRequest",
    "V1GateDisclosureStreamConsumer",
    "V1TargetScaleGateDisclosureChunk",
    "literal_relu_gate_jacobian_cotangent",
    "score_captured_conserving_route_blocks",
]
