from __future__ import annotations

"""Exact target-batched frontier reducers.

These reducers add a leading target axis to the canonical fast score math:

    scalar gradients: [B, ...] -> scores [route...]
    target gradients: [T, B, ...] -> scores [T, route...]

They intentionally expose a compact dictionary API for the four canonical
production quantities instead of the legacy 40+ slot tuple.
"""

from dataclasses import dataclass
import os
from typing import Any, Dict, Optional, Union

import torch
import torch.nn.functional as F

from packages.salience.ops.abs_mode import (
    is_frontier_aggregate_abs_mode,
    normalize_abs_aggregation_mode,
)
from packages.salience.ops.conv_oldfast_fused import (
    _build_conv_channel_virtual_weight,
    _compute_conv_channel_virtual_preadd,
    _frontier_aggregate_conv_block_size,
)
from packages.salience.ops.linear_emit import (
    _choose_frontier_aggregate_linear_block_size,
)
from packages.salience.utils.replay import TensorReplayProvider, fetch_replay_slice, replay_batch_size


ScoreDict = Dict[str, torch.Tensor]
LEGACY_OLDFAST_ABS_SCORE_HEAD = "legacy_oldfast_abs_scores"
PER_CONTEXT_SCORE_HEAD = "per_context_scores"
LOCAL_DESTINATION_CONTEXT_HEAD = "local_destination_context_scores"

# Census fused backward (IRG_FUSED_CENSUS_BACKWARD=1, default OFF): with a
# ``per_context_consumer`` armed, the with-context reducers stream whole-
# context route-row chunks to the consumer instead of materializing the
# persistent [batch x targets x route-axis] per-context head — the K=2
# census OOM class.  Peak live rows: one fp32 chunk.
FUSED_CENSUS_BACKWARD_TARGET_MB_ENVIRONMENT = (
    "IRG_FUSED_CENSUS_BACKWARD_TARGET_MB"
)


def _debug_assert_finite_enabled() -> bool:
    raw = os.environ.get("SCORING_TARGET_BATCHED_ASSERT_FINITE", "0")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _census_output_row_chunk(
    *, batch: int, targets: int, row_width: int, out_rows: int
) -> int:
    """Output-row chunking for streamed census emission.

    Bounds the transient [batch, targets, o_chunk, row_width] fp32 rows
    under a runtime budget (default 512 MiB), mirroring the reducer-budget
    knobs above.
    """

    target_mb = max(
        1,
        int(os.getenv(FUSED_CENSUS_BACKWARD_TARGET_MB_ENVIRONMENT, "512")),
    )
    bytes_per_row = max(1, batch * targets * row_width * 4)
    return max(1, min(out_rows, target_mb * 1024 * 1024 // bytes_per_row))


def _assert_finite_tensor(tensor: torch.Tensor, *, role: str) -> None:
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"Non-finite tensor encountered during target-batched reduction: {role}.")


@dataclass(frozen=True)
class TargetBatchedReductionResult:
    target_ids: tuple[int, ...]
    layer_name: str
    route_family: str
    scores: ScoreDict
    metadata: dict


def _resolve_scale(scale: Union[float, int, torch.Tensor], *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.is_tensor(scale):
        value = float(scale.detach().to(device="cpu").item())
    else:
        value = float(scale)
    if abs(value) < 1e-12:
        value = 1.0
    scale_tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    abs_scale_tensor = torch.as_tensor(abs(value), dtype=torch.float32, device=device)
    return scale_tensor, abs_scale_tensor


def _init_linear_scores(
    *,
    target_count: int,
    out_features: int,
    in_features: int,
    device: torch.device,
    compute_legacy_oldfast_abs: bool = False,
) -> ScoreDict:
    scores = {
        "all_scores": torch.zeros((target_count, out_features, in_features), dtype=torch.float32, device=device),
        "all_abs_scores": torch.zeros((target_count, out_features, in_features), dtype=torch.float32, device=device),
        "neuron_scores": torch.zeros((target_count, out_features), dtype=torch.float32, device=device),
        "neuron_abs_scores": torch.zeros((target_count, out_features), dtype=torch.float32, device=device),
    }
    if compute_legacy_oldfast_abs:
        scores[LEGACY_OLDFAST_ABS_SCORE_HEAD] = torch.zeros(
            (target_count, out_features, in_features),
            dtype=torch.float32,
            device=device,
        )
    return scores


def _linear_frontier_aggregate_streamed_census(
    inputs_source: Union[torch.Tensor, TensorReplayProvider],
    target_neuron_gradients: torch.Tensor,
    *,
    weights: torch.Tensor,
    scale: Union[float, int, torch.Tensor],
    device: torch.device,
    abs_aggregation_mode: str,
    per_context_consumer: Any,
) -> ScoreDict:
    """Linear census reduction streamed by output-row route chunks.

    Instead of allocating the persistent ``[B, T, O, I]`` per-context head
    (for the fc census blocks this IS the measured ~30 GiB/lane burst),
    each ``[B, T, o_chunk, I]`` fp32 chunk plus its query-independent local
    rows are handed to ``per_context_consumer(route_offset, rows, local)``
    and freed.  The pooled signed/abs heads keep the per-context branch's
    arithmetic (rows summed over contexts per chunk); relative to the
    retained path only the ROUTE-axis chunking of that sum changes, so
    agreement is reduction-order-equivalent at fp32 accumulation, not
    bitwise.  The full input activations ``[B, I]`` are fetched once —
    they are orders of magnitude smaller than one route chunk.
    """

    weights = weights.to(device=device, dtype=torch.float32)
    target_neuron_gradients = target_neuron_gradients.to(
        device=device, dtype=torch.float32
    )
    target_count, batch_size, out_features = target_neuron_gradients.shape
    weight_out, in_features = weights.shape
    if int(weight_out) != int(out_features):
        raise ValueError(
            f"Linear gradient out_features mismatch: grads={out_features}, weights={weight_out}."
        )
    source_batch = replay_batch_size(inputs_source, batch_size)
    if int(source_batch) != int(batch_size):
        raise ValueError(
            f"Linear input batch mismatch: inputs={source_batch}, grads={batch_size}."
        )
    scores = _init_linear_scores(
        target_count=int(target_count),
        out_features=int(out_features),
        in_features=int(in_features),
        device=device,
    )
    abs_mode = normalize_abs_aggregation_mode(abs_aggregation_mode)
    frontier_abs = is_frontier_aggregate_abs_mode(abs_mode)
    scale_tensor, abs_scale_tensor = _resolve_scale(scale, device=device)
    inputs = fetch_replay_slice(
        inputs_source, 0, int(batch_size), device=device, dtype=torch.float32
    )
    if inputs.dim() > 2:
        inputs = inputs.reshape(inputs.size(0), -1)
    row_chunk = _census_output_row_chunk(
        batch=int(batch_size),
        targets=int(target_count),
        row_width=int(in_features),
        out_rows=int(out_features),
    )
    for start in range(0, int(out_features), row_chunk):
        end = min(int(out_features), start + row_chunk)
        grads_chunk = target_neuron_gradients[:, :, start:end] / scale_tensor
        rows = torch.einsum("tbo,bi->btoi", grads_chunk, inputs)
        rows.mul_(
            weights[start:end].reshape(1, 1, end - start, int(in_features))
        )
        scores["all_scores"][:, start:end].add_(rows.sum(dim=0))
        if not frontier_abs:
            scores["all_abs_scores"][:, start:end].add_(
                torch.einsum(
                    "tbo,bi->toi",
                    target_neuron_gradients[:, :, start:end].abs()
                    / abs_scale_tensor,
                    inputs.abs(),
                )
                * weights[start:end].abs().unsqueeze(0)
            )
        local_chunk = inputs.unsqueeze(1) * weights[start:end].unsqueeze(0)
        local_chunk.mul_(scale_tensor)
        per_context_consumer(start * int(in_features), rows, local_chunk)
        del rows, local_chunk
    if frontier_abs:
        scores["all_abs_scores"] = scores["all_scores"].abs()
    scores["neuron_scores"].zero_()
    scores["neuron_abs_scores"].zero_()
    return scores


def _linear_frontier_aggregate_inbackward_census(
    inputs_source: Union[torch.Tensor, TensorReplayProvider],
    target_neuron_gradients: torch.Tensor,
    *,
    weights: torch.Tensor,
    scale: Union[float, int, torch.Tensor],
    device: torch.device,
    abs_aggregation_mode: str,
    per_context_consumer: Any,
    retain_local_destination: bool = False,
) -> ScoreDict:
    """Linear census reduction with IN-REGISTER rows (full in-backward form).

    The graded streamed variant above still materializes each
    ``[B, T, o_chunk, I]`` fp32 row chunk for the consumer; here the
    consumer receives the backward's OWN tensors — the fp32 inputs, the
    per-o-chunk gradient slice, and the weight slice — and forms the rows
    inside its Triton kernel: the row tensor never exists at any width.
    The pooled score heads are computed by direct ``tbo,bi->toi``
    contractions (no [B, ...] head either); relative to the graded form
    only fp32 REDUCTION ORDER changes (the weight factor distributes over
    the batch sum exactly in real arithmetic), the same documented
    relation the graded form holds to the retained path.  Single-query
    (T == 1) packets only — the census sink's own contract.

    ``retain_local_destination`` (production handoff): the query-
    independent local head is RETAINED — elementwise ``(x*w)*scale``,
    bitwise the retained path's values — because the fc nonnegative-input
    receipts, the input-covariance accumulator, and the flip-margin
    geometry consume it downstream.  Only the [B, T, O, I] GLOBAL head
    (the measured census burst) is dissolved.
    """

    weights = weights.to(device=device, dtype=torch.float32)
    target_neuron_gradients = target_neuron_gradients.to(
        device=device, dtype=torch.float32
    )
    target_count, batch_size, out_features = target_neuron_gradients.shape
    weight_out, in_features = weights.shape
    if int(weight_out) != int(out_features):
        raise ValueError(
            f"Linear gradient out_features mismatch: grads={out_features}, weights={weight_out}."
        )
    if int(target_count) != 1:
        raise ValueError(
            "in-backward census streaming requires single-query packets"
        )
    source_batch = replay_batch_size(inputs_source, batch_size)
    if int(source_batch) != int(batch_size):
        raise ValueError(
            f"Linear input batch mismatch: inputs={source_batch}, grads={batch_size}."
        )
    scores = _init_linear_scores(
        target_count=int(target_count),
        out_features=int(out_features),
        in_features=int(in_features),
        device=device,
    )
    abs_mode = normalize_abs_aggregation_mode(abs_aggregation_mode)
    frontier_abs = is_frontier_aggregate_abs_mode(abs_mode)
    scale_tensor, abs_scale_tensor = _resolve_scale(scale, device=device)
    inputs = fetch_replay_slice(
        inputs_source, 0, int(batch_size), device=device, dtype=torch.float32
    )
    if inputs.dim() > 2:
        inputs = inputs.reshape(inputs.size(0), -1)
    if retain_local_destination:
        # Bitwise the retained path's local head: pure elementwise
        # ``(x*w)*scale`` (blocking-invariant), computed once.
        local_head = torch.empty(
            (int(batch_size), int(out_features), int(in_features)),
            dtype=torch.float32,
            device=device,
        )
        torch.mul(inputs.unsqueeze(1), weights.unsqueeze(0), out=local_head)
        local_head.mul_(scale_tensor)
        scores[LOCAL_DESTINATION_CONTEXT_HEAD] = local_head
    row_chunk = _census_output_row_chunk(
        batch=int(batch_size),
        targets=int(target_count),
        row_width=int(in_features),
        out_rows=int(out_features),
    )
    for start in range(0, int(out_features), row_chunk):
        end = min(int(out_features), start + row_chunk)
        grads_chunk = target_neuron_gradients[:, :, start:end] / scale_tensor
        scores["all_scores"][:, start:end].add_(
            torch.einsum("tbo,bi->toi", grads_chunk, inputs)
            * weights[start:end].unsqueeze(0)
        )
        if not frontier_abs:
            scores["all_abs_scores"][:, start:end].add_(
                torch.einsum(
                    "tbo,bi->toi",
                    target_neuron_gradients[:, :, start:end].abs()
                    / abs_scale_tensor,
                    inputs.abs(),
                )
                * weights[start:end].abs().unsqueeze(0)
            )
        per_context_consumer.consume_linear_backward(
            start * int(in_features),
            inputs=inputs,
            grads=target_neuron_gradients[0, :, start:end].contiguous(),
            layer_weight=weights[start:end].contiguous(),
            producer_scale=float(scale_tensor.item()),
        )
    if frontier_abs:
        scores["all_abs_scores"] = scores["all_scores"].abs()
    scores["neuron_scores"].zero_()
    scores["neuron_abs_scores"].zero_()
    return scores


def linear_frontier_aggregate_target_batched(
    inputs_source: Union[torch.Tensor, TensorReplayProvider],
    target_neuron_gradients: torch.Tensor,
    *,
    weights: torch.Tensor,
    scale: Union[float, int, torch.Tensor],
    device: torch.device,
    abs_aggregation_mode: str = "exact_abs",
    compute_neuron_metrics: bool = True,
    compute_legacy_oldfast_abs: bool = False,
    retain_per_context: bool = False,
    retain_local_destination: bool = False,
    per_context_consumer: Any | None = None,
) -> ScoreDict:
    """Compute exact Linear frontier scores for multiple target cotangents.

    Args:
        inputs_source: Layer inputs with batch axis first, or replay provider.
        target_neuron_gradients: Tensor shaped ``[T, B, O]``.
        weights: Linear weight matrix shaped ``[O, I]``.
        per_context_consumer: Sink for bounded context-row census reduction.
    """

    if per_context_consumer is not None:
        if not (retain_per_context and retain_local_destination):
            raise ValueError(
                "census-backward streaming requires per-context and "
                "local-destination retention semantics"
            )
        if compute_neuron_metrics or compute_legacy_oldfast_abs:
            raise ValueError(
                "census-backward streaming supports only the canonical "
                "four-head census reduction"
            )
        if target_neuron_gradients.ndim != 3:
            raise ValueError(
                "target_neuron_gradients for Linear must have shape [targets, batch, out_features]."
            )
        if (
            callable(
                getattr(per_context_consumer, "consume_linear_backward", None)
            )
            and bool(
                getattr(per_context_consumer, "wants_linear_inbackward", False)
            )
        ):
            return _linear_frontier_aggregate_inbackward_census(
                inputs_source,
                target_neuron_gradients,
                weights=weights,
                scale=scale,
                device=device,
                abs_aggregation_mode=abs_aggregation_mode,
                per_context_consumer=per_context_consumer,
                retain_local_destination=bool(retain_local_destination),
            )
        if not bool(getattr(per_context_consumer, "wants_linear_rows", True)):
            # Capability negotiation failed (an in-backward capture sink
            # never accepts materialized graded streams): keep the retained
            # path so every downstream row consumer stays intact — the
            # provider detects the uncovered span and tile-feeds it.
            per_context_consumer = None
        if per_context_consumer is not None:
            return _linear_frontier_aggregate_streamed_census(
                inputs_source,
                target_neuron_gradients,
                weights=weights,
                scale=scale,
                device=device,
                abs_aggregation_mode=abs_aggregation_mode,
                per_context_consumer=per_context_consumer,
            )
    if target_neuron_gradients.ndim != 3:
        raise ValueError(
            "target_neuron_gradients for Linear must have shape [targets, batch, out_features]."
        )
    weights = weights.to(device=device, dtype=torch.float32)
    target_neuron_gradients = target_neuron_gradients.to(device=device, dtype=torch.float32)
    target_count, batch_size, out_features = target_neuron_gradients.shape
    weight_out, in_features = weights.shape
    if int(weight_out) != int(out_features):
        raise ValueError(
            f"Linear gradient out_features mismatch: grads={out_features}, weights={weight_out}."
        )
    source_batch = replay_batch_size(inputs_source, batch_size)
    if int(source_batch) != int(batch_size):
        raise ValueError(
            f"Linear input batch mismatch: inputs={source_batch}, grads={batch_size}."
        )

    block_size = _choose_frontier_aggregate_linear_block_size(
        batch_size=int(batch_size),
        in_features=int(in_features),
        out_features=int(out_features),
    )
    scores = _init_linear_scores(
        target_count=int(target_count),
        out_features=int(out_features),
        in_features=int(in_features),
        device=device,
        compute_legacy_oldfast_abs=compute_legacy_oldfast_abs,
    )
    abs_mode = normalize_abs_aggregation_mode(abs_aggregation_mode)
    frontier_abs = is_frontier_aggregate_abs_mode(abs_mode)
    scale_tensor, abs_scale_tensor = _resolve_scale(scale, device=device)
    if retain_per_context:
        scores[PER_CONTEXT_SCORE_HEAD] = torch.empty(
            (int(batch_size), int(target_count), int(out_features), int(in_features)),
            dtype=torch.float32,
            device=device,
        )
    if retain_local_destination:
        scores[LOCAL_DESTINATION_CONTEXT_HEAD] = torch.empty(
            (int(batch_size), int(out_features), int(in_features)),
            dtype=torch.float32,
            device=device,
        )

    for start in range(0, int(batch_size), int(block_size)):
        end = min(int(batch_size), start + int(block_size))
        inputs_chunk = fetch_replay_slice(
            inputs_source,
            start,
            end,
            device=device,
            dtype=torch.float32,
        )
        if inputs_chunk.dim() > 2:
            inputs_chunk = inputs_chunk.reshape(inputs_chunk.size(0), -1)
        grads_chunk = target_neuron_gradients[:, start:end, :] / scale_tensor
        if _debug_assert_finite_enabled():
            _assert_finite_tensor(inputs_chunk, role="linear inputs_chunk")
            _assert_finite_tensor(grads_chunk, role="linear grads_chunk")
        per_context_scores = None
        if retain_per_context:
            per_context_scores = torch.einsum("tbo,bi->btoi", grads_chunk, inputs_chunk)
            per_context_scores.mul_(
                weights.reshape(1, 1, int(out_features), int(in_features))
            )
            signed_scores = per_context_scores.sum(dim=0)
        else:
            # Preserve the historical fast-path contraction and its floating
            # reduction order when the auxiliary tee is disabled.
            signed_scores = (
                torch.einsum("tbo,bi->toi", grads_chunk, inputs_chunk)
                * weights.unsqueeze(0)
            )
        scores["all_scores"].add_(signed_scores)
        if retain_per_context:
            assert per_context_scores is not None
            scores[PER_CONTEXT_SCORE_HEAD][start:end].copy_(per_context_scores)
            del per_context_scores
        if retain_local_destination:
            # The local head is the signed contribution to the immediate
            # destination coordinate before the query cotangent contracts it.
            # It is query-independent and is therefore emitted once per
            # context, not once per target.
            local_destination_slice = scores[LOCAL_DESTINATION_CONTEXT_HEAD][start:end]
            torch.mul(
                inputs_chunk.unsqueeze(1),
                weights.unsqueeze(0),
                out=local_destination_slice,
            )
            local_destination_slice.mul_(scale_tensor)
        legacy_abs_scores = None
        if compute_legacy_oldfast_abs or not frontier_abs:
            legacy_abs_scores = (
                torch.einsum(
                    "tbo,bi->toi",
                    target_neuron_gradients[:, start:end, :].abs() / abs_scale_tensor,
                    inputs_chunk.abs(),
                )
                * weights.abs().unsqueeze(0)
            )
        if compute_legacy_oldfast_abs:
            scores[LEGACY_OLDFAST_ABS_SCORE_HEAD].add_(legacy_abs_scores)
        if frontier_abs:
            # Frontier abs is applied after the full signed accumulation.
            pass
        else:
            scores["all_abs_scores"].add_(legacy_abs_scores)
        if compute_neuron_metrics:
            scores["neuron_scores"].add_(signed_scores.sum(dim=-1))
            if not frontier_abs:
                scores["neuron_abs_scores"].add_(legacy_abs_scores.sum(dim=-1))

    if frontier_abs:
        scores["all_abs_scores"] = scores["all_scores"].abs()
        if compute_neuron_metrics:
            scores["neuron_abs_scores"] = scores["neuron_scores"].abs()
    elif not compute_neuron_metrics:
        scores["neuron_abs_scores"].zero_()
    if not compute_neuron_metrics:
        scores["neuron_scores"].zero_()
        scores["neuron_abs_scores"].zero_()
    return scores


def _init_conv_scores(
    *,
    target_count: int,
    out_channels: int,
    in_per_group: int,
    device: torch.device,
    compute_legacy_oldfast_abs: bool = False,
) -> ScoreDict:
    scores = {
        "all_scores": torch.zeros((target_count, out_channels, in_per_group), dtype=torch.float32, device=device),
        "all_abs_scores": torch.zeros((target_count, out_channels, in_per_group), dtype=torch.float32, device=device),
        "neuron_scores": torch.zeros((target_count, out_channels), dtype=torch.float32, device=device),
        "neuron_abs_scores": torch.zeros((target_count, out_channels), dtype=torch.float32, device=device),
    }
    if compute_legacy_oldfast_abs:
        scores[LEGACY_OLDFAST_ABS_SCORE_HEAD] = torch.zeros(
            (target_count, out_channels, in_per_group),
            dtype=torch.float32,
            device=device,
        )
    return scores


def conv_channel_virtual_frontier_aggregate_target_batched(
    inputs_source: Union[torch.Tensor, TensorReplayProvider],
    target_neuron_gradients: torch.Tensor,
    *,
    weights: torch.Tensor,
    scale: Union[float, int, torch.Tensor],
    stride,
    padding,
    dilation,
    groups: int,
    device: torch.device,
    abs_aggregation_mode: str = "exact_abs",
    compute_neuron_metrics: bool = True,
    compute_legacy_oldfast_abs: bool = False,
    retain_per_context: bool = False,
    retain_local_destination: bool = False,
    local_regime_tee: Any | None = None,
    per_context_consumer: Any | None = None,
) -> ScoreDict:
    """Compute exact channel-virtual Conv2d frontier scores for target blocks.

    ``target_neuron_gradients`` must be shaped ``[T, B, O, H, W]``.

    With ``per_context_consumer``, the per-context and local heads are
    assembled for THIS block as usual — a conv block's rows are orders of
    magnitude below the fc burst and stay bounded by the block — then
    emitted to the consumer as whole-context output-channel chunks and
    freed, so nothing multi-block accumulates and the returned dictionary
    omits the retained heads.
    """

    if per_context_consumer is not None and (
        not bool(getattr(per_context_consumer, "wants_conv_rows", True))
    ):
        per_context_consumer = None
    if per_context_consumer is not None:
        if not (retain_per_context and retain_local_destination):
            raise ValueError(
                "census-backward streaming requires per-context and "
                "local-destination retention semantics"
            )
        if compute_neuron_metrics or compute_legacy_oldfast_abs:
            raise ValueError(
                "census-backward streaming supports only the canonical "
                "four-head census reduction"
            )
    if target_neuron_gradients.ndim != 5:
        raise ValueError(
            "target_neuron_gradients for Conv2d must have shape [targets, batch, out_channels, H, W]."
        )
    weights = weights.to(device=device, dtype=torch.float32)
    target_neuron_gradients = target_neuron_gradients.to(device=device, dtype=torch.float32)
    target_count, batch_size, out_channels, out_h, out_w = target_neuron_gradients.shape
    if int(weights.size(0)) != int(out_channels):
        raise ValueError(
            f"Conv gradient out_channels mismatch: grads={out_channels}, weights={weights.size(0)}."
        )
    source_batch = replay_batch_size(inputs_source, batch_size)
    if int(source_batch) != int(batch_size):
        raise ValueError(
            f"Conv input batch mismatch: inputs={source_batch}, grads={batch_size}."
        )
    preadd_weight, in_per_group, out_per_group = _build_conv_channel_virtual_weight(
        weights,
        groups=int(groups),
    )
    preadd_weight = preadd_weight.to(device=device, dtype=torch.float32)
    block_size = _frontier_aggregate_conv_block_size(
        batch_size=int(batch_size),
        out_channels=int(out_channels),
        in_per_group=int(in_per_group),
        output_h=int(out_h),
        output_w=int(out_w),
    )
    scores = _init_conv_scores(
        target_count=int(target_count),
        out_channels=int(out_channels),
        in_per_group=int(in_per_group),
        device=device,
        compute_legacy_oldfast_abs=compute_legacy_oldfast_abs,
    )
    abs_mode = normalize_abs_aggregation_mode(abs_aggregation_mode)
    frontier_abs = is_frontier_aggregate_abs_mode(abs_mode)
    scale_tensor, abs_scale_tensor = _resolve_scale(scale, device=device)
    if retain_per_context:
        scores[PER_CONTEXT_SCORE_HEAD] = torch.empty(
            (
                int(batch_size),
                int(target_count),
                int(out_channels),
                int(in_per_group),
            ),
            dtype=torch.float32,
            device=device,
        )
    if retain_local_destination:
        scores[LOCAL_DESTINATION_CONTEXT_HEAD] = torch.empty(
            (int(batch_size), int(out_channels), int(in_per_group)),
            dtype=torch.float32,
            device=device,
        )

    for start in range(0, int(batch_size), int(block_size)):
        end = min(int(batch_size), start + int(block_size))
        inputs_chunk = fetch_replay_slice(
            inputs_source,
            start,
            end,
            device=device,
            dtype=torch.float32,
        )
        grads_chunk = target_neuron_gradients[:, start:end, :, :, :]
        if _debug_assert_finite_enabled():
            _assert_finite_tensor(inputs_chunk, role="conv inputs_chunk")
            _assert_finite_tensor(grads_chunk, role="conv grads_chunk")
        preadd_chunk = _compute_conv_channel_virtual_preadd(
            inputs_chunk,
            preadd_weight,
            groups=int(groups),
            in_per_group=int(in_per_group),
            out_per_group=int(out_per_group),
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        if _debug_assert_finite_enabled():
            _assert_finite_tensor(preadd_chunk, role="conv channel-virtual preadd_chunk")
        per_sample_scores = torch.einsum(
            "boihw,tbohw->tboi",
            preadd_chunk,
            grads_chunk / scale_tensor,
        )
        signed_scores = per_sample_scores.sum(dim=1)
        scores["all_scores"].add_(signed_scores)
        if retain_per_context:
            scores[PER_CONTEXT_SCORE_HEAD][start:end].copy_(
                per_sample_scores.transpose(0, 1)
            )
        if retain_local_destination:
            # preadd_chunk is already the exact route-resolved contribution
            # to each destination channel and spatial coordinate.  Taking its
            # signed spatial sum here is a reduction-only tee: no unfold, replay,
            # convolution, or autograd pass is added.
            scores[LOCAL_DESTINATION_CONTEXT_HEAD][start:end].copy_(
                preadd_chunk.sum(dim=(-1, -2)) * scale_tensor
            )
        if local_regime_tee is not None:
            # Optional gate-geometry tee. ``preadd_chunk`` is the same
            # ephemeral route-resolved field used by the canonical global and
            # local heads; the tee consumes it before spatial reduction and
            # retains only route-axis sufficient statistics.
            local_regime_tee.update(start, preadd_chunk)
        exact_abs_scores = None
        if compute_legacy_oldfast_abs or not frontier_abs:
            exact_abs_scores = per_sample_scores.abs().sum(dim=1)
        if compute_legacy_oldfast_abs:
            scores[LEGACY_OLDFAST_ABS_SCORE_HEAD].add_(exact_abs_scores)
        if not frontier_abs:
            scores["all_abs_scores"].add_(exact_abs_scores)
        if compute_neuron_metrics:
            scores["neuron_scores"].add_(signed_scores.sum(dim=-1))
            if not frontier_abs:
                scores["neuron_abs_scores"].add_(exact_abs_scores.sum(dim=-1))

        # Drop references before the next chunk so CUDA memory follows the
        # chunk plan instead of the full batch shape.
        del per_sample_scores, signed_scores, exact_abs_scores

    if frontier_abs:
        scores["all_abs_scores"] = scores["all_scores"].abs()
        if compute_neuron_metrics:
            scores["neuron_abs_scores"] = scores["neuron_scores"].abs()
    if not compute_neuron_metrics:
        scores["neuron_scores"].zero_()
        scores["neuron_abs_scores"].zero_()
    if per_context_consumer is not None:
        rows_buffer = scores.pop(PER_CONTEXT_SCORE_HEAD)
        local_buffer = scores.pop(LOCAL_DESTINATION_CONTEXT_HEAD)
        row_chunk = _census_output_row_chunk(
            batch=int(batch_size),
            targets=int(target_count),
            row_width=int(in_per_group),
            out_rows=int(out_channels),
        )
        for start in range(0, int(out_channels), row_chunk):
            end = min(int(out_channels), start + row_chunk)
            per_context_consumer(
                start * int(in_per_group),
                rows_buffer[:, :, start:end],
                local_buffer[:, start:end],
            )
        del rows_buffer, local_buffer
    return scores


def linear_route_measurement_target_batched(
    inputs_source: Union[torch.Tensor, TensorReplayProvider],
    target_neuron_gradients: torch.Tensor,
    *,
    weights: torch.Tensor,
    scale: Union[float, int, torch.Tensor],
    device: torch.device,
    abs_aggregation_mode: str = "exact_abs",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Canonical signed and absolute Linear route measurements.

    This compact API keeps legacy score-head names and optional compatibility
    quantities outside the mechanistic-systems namespace.
    """

    result = linear_frontier_aggregate_target_batched(
        inputs_source,
        target_neuron_gradients,
        weights=weights,
        scale=scale,
        device=device,
        abs_aggregation_mode=abs_aggregation_mode,
        compute_neuron_metrics=False,
        compute_legacy_oldfast_abs=False,
    )
    return result["all_scores"], result["all_abs_scores"]


def linear_route_measurement_with_context_target_batched(
    inputs_source: Union[torch.Tensor, TensorReplayProvider],
    target_neuron_gradients: torch.Tensor,
    *,
    weights: torch.Tensor,
    scale: Union[float, int, torch.Tensor],
    device: torch.device,
    abs_aggregation_mode: str = "exact_abs",
    retain_local_destination: bool = True,
    per_context_consumer: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Fused global rows and query-independent local destination rows.

    With an armed census-backward ``per_context_consumer`` the retained
    heads stream to the consumer and the tuple carries ``None`` in their
    places; unarmed calls are unchanged.
    """

    result = linear_frontier_aggregate_target_batched(
        inputs_source,
        target_neuron_gradients,
        weights=weights,
        scale=scale,
        device=device,
        abs_aggregation_mode=abs_aggregation_mode,
        compute_neuron_metrics=False,
        compute_legacy_oldfast_abs=False,
        retain_per_context=True,
        retain_local_destination=bool(retain_local_destination),
        per_context_consumer=per_context_consumer,
    )
    return (
        result["all_scores"],
        result["all_abs_scores"],
        result.get(PER_CONTEXT_SCORE_HEAD),
        (
            result.get(LOCAL_DESTINATION_CONTEXT_HEAD)
            if retain_local_destination
            else None
        ),
    )


def conv_channel_route_measurement_target_batched(
    inputs_source: Union[torch.Tensor, TensorReplayProvider],
    target_neuron_gradients: torch.Tensor,
    *,
    weights: torch.Tensor,
    scale: Union[float, int, torch.Tensor],
    stride,
    padding,
    dilation,
    groups: int,
    device: torch.device,
    abs_aggregation_mode: str = "exact_abs",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Canonical signed and absolute channel-pair Conv route measurements."""

    result = conv_channel_virtual_frontier_aggregate_target_batched(
        inputs_source,
        target_neuron_gradients,
        weights=weights,
        scale=scale,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        device=device,
        abs_aggregation_mode=abs_aggregation_mode,
        compute_neuron_metrics=False,
        compute_legacy_oldfast_abs=False,
    )
    return result["all_scores"], result["all_abs_scores"]


def conv_channel_route_measurement_with_context_target_batched(
    inputs_source: Union[torch.Tensor, TensorReplayProvider],
    target_neuron_gradients: torch.Tensor,
    *,
    weights: torch.Tensor,
    scale: Union[float, int, torch.Tensor],
    stride,
    padding,
    dilation,
    groups: int,
    device: torch.device,
    abs_aggregation_mode: str = "exact_abs",
    retain_local_destination: bool = True,
    per_context_consumer: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Fused Conv global rows and immediate-destination local rows.

    With an armed census-backward ``per_context_consumer`` the retained
    heads stream to the consumer and the tuple carries ``None`` in their
    places; unarmed calls are unchanged.
    """

    result = conv_channel_virtual_frontier_aggregate_target_batched(
        inputs_source,
        target_neuron_gradients,
        weights=weights,
        scale=scale,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        device=device,
        abs_aggregation_mode=abs_aggregation_mode,
        compute_neuron_metrics=False,
        compute_legacy_oldfast_abs=False,
        retain_per_context=True,
        retain_local_destination=bool(retain_local_destination),
        per_context_consumer=per_context_consumer,
    )
    return (
        result["all_scores"],
        result["all_abs_scores"],
        result.get(PER_CONTEXT_SCORE_HEAD),
        (
            result.get(LOCAL_DESTINATION_CONTEXT_HEAD)
            if retain_local_destination
            else None
        ),
    )


def conv_channel_route_measurement_with_context_and_regime_target_batched(
    inputs_source: Union[torch.Tensor, TensorReplayProvider],
    target_neuron_gradients: torch.Tensor,
    *,
    weights: torch.Tensor,
    scale: Union[float, int, torch.Tensor],
    stride,
    padding,
    dilation,
    groups: int,
    device: torch.device,
    local_regime_tee: Any,
    abs_aggregation_mode: str = "exact_abs",
    retain_local_destination: bool = True,
    per_context_consumer: Any | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    dict[str, Any],
]:
    """Canonical four heads plus an optional pre-reduction regime tee.

    The ordinary four-value wrapper above remains byte-for-byte authoritative
    when regime diagnostics are disabled.
    """

    if local_regime_tee is None:
        raise ValueError("regime-aware Conv reduction requires a regime tee")
    result = conv_channel_virtual_frontier_aggregate_target_batched(
        inputs_source,
        target_neuron_gradients,
        weights=weights,
        scale=scale,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        device=device,
        abs_aggregation_mode=abs_aggregation_mode,
        compute_neuron_metrics=False,
        compute_legacy_oldfast_abs=False,
        retain_per_context=True,
        retain_local_destination=bool(retain_local_destination),
        local_regime_tee=local_regime_tee,
        per_context_consumer=per_context_consumer,
    )
    return (
        result["all_scores"],
        result["all_abs_scores"],
        result.get(PER_CONTEXT_SCORE_HEAD),
        (
            result.get(LOCAL_DESTINATION_CONTEXT_HEAD)
            if retain_local_destination
            else None
        ),
        local_regime_tee.finalize(),
    )


__all__ = [
    "FUSED_CENSUS_BACKWARD_ENVIRONMENT",
    "LEGACY_OLDFAST_ABS_SCORE_HEAD",
    "LOCAL_DESTINATION_CONTEXT_HEAD",
    "PER_CONTEXT_SCORE_HEAD",
    "TargetBatchedReductionResult",
    "conv_channel_virtual_frontier_aggregate_target_batched",
    "conv_channel_route_measurement_target_batched",
    "conv_channel_route_measurement_with_context_and_regime_target_batched",
    "conv_channel_route_measurement_with_context_target_batched",
    "linear_frontier_aggregate_target_batched",
    "linear_route_measurement_target_batched",
    "linear_route_measurement_with_context_target_batched",
]
