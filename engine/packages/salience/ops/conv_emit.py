from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Union
import os

import torch
import torch.nn.functional as F
import torch._dynamo

from packages.salience.ops.abs_mode import (
    is_frontier_aggregate_abs_mode,
    normalize_abs_aggregation_mode,
)
from packages.salience.ops.linear_utils import safe_pearson_corr
from packages.salience.utils import (
    TensorReplayProvider,
    replay_batch_size as _replay_batch_size,
    fetch_replay_slice as _fetch_replay_slice,
    fetch_replay_indices as _fetch_replay_indices,
    pack_boolean_rows,
)
from packages.salience.core.frontier_utils import (
    COMPUTE_BITPACK,
    DEBUG_SCORING,
    VALIDATE_FRONTIER,
    choose_chunk_size,
)
from packages.salience.utils.layer_logging import (
    flatten_objective_vec as _flatten_objective_vec,
    finalize_layer_corr_logging as _finalize_layer_corr_logging,
    maybe_record_layer_corr_sample as _maybe_record_layer_corr_sample,
)


def _normalize_conv_score_basis(score_basis: Optional[str]) -> str:
    raw = str(score_basis or os.environ.get("SCORING_CONV_SCORE_BASIS", "channel_virtual")).strip().lower()
    aliases = {
        "": "kernel_weight",
        "weight": "kernel_weight",
        "kernel": "kernel_weight",
        "kernel_weight": "kernel_weight",
        "channel": "channel_virtual",
        "virtual": "channel_virtual",
        "channel_virtual": "channel_virtual",
        "conv_channel_virtual": "channel_virtual",
    }
    basis = aliases.get(raw)
    if basis is None:
        raise ValueError(
            f"Unsupported conv score basis {score_basis!r}. "
            "Expected one of: kernel_weight, channel_virtual."
        )
    return basis


def _resolve_scale_tensors(
    scale: float,
    *,
    device: torch.device,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    if isinstance(scale, torch.Tensor):
        scale_value = float(scale.detach().cpu())
    else:
        scale_value = float(scale)
    if abs(scale_value) < 1e-12:
        scale_value = 1.0
    scale_tensor = torch.as_tensor(scale_value, device=device, dtype=torch.float32)
    abs_scale_tensor = torch.as_tensor(abs(scale_value), device=device, dtype=torch.float32)
    return scale_value, scale_tensor, abs_scale_tensor


def _frontier_aggregate_conv_reduction_mode(
    *,
    device: torch.device,
    locations: int,
) -> str:
    raw = str(
        os.environ.get(
            "SCORING_FRONTIER_AGGREGATE_CONV_REDUCTION",
            os.environ.get("SCORING_OLDFAST_CONV_REDUCTION", "auto"),
        )
    ).strip().lower()
    if raw in {"direct", "lowmem", "low_memory"}:
        return "direct"
    if raw in {"sample_sum", "sample-then-sum", "two_stage", "twostage"}:
        return "sample_sum"
    if raw not in {"", "auto", "default"}:
        return "auto"
    if device.type == "cuda" and locations > 1:
        return "sample_sum"
    return "direct"


def _choose_conv_exact_sample_block_size(
    *,
    groups: int,
    out_per_group: int,
    in_per_group: int,
    kernel_elements: int,
    locations: int,
    batch_size: int,
) -> int:
    target_elements = max(
        1,
        int(os.environ.get("SCORING_CONV_EXACT_TARGET_ELEMENTS", str(16 * 1024 * 1024))),
    )
    max_samples = max(1, int(os.environ.get("SCORING_CONV_EXACT_MAX_SAMPLES", "32")))
    per_sample_patch = max(1, groups * in_per_group * kernel_elements * locations)
    per_sample_grad = max(1, groups * out_per_group * in_per_group * kernel_elements)
    per_sample_footprint = per_sample_patch + per_sample_grad
    sample_block = max(1, target_elements // per_sample_footprint)
    return max(1, min(batch_size, max_samples, sample_block))


def _build_conv_channel_virtual_weight(
    weights: torch.Tensor,
    *,
    groups: int,
) -> Tuple[torch.Tensor, int, int]:
    out_channels, in_per_group, kernel_h, kernel_w = weights.shape
    if out_channels % groups != 0:
        raise ValueError(
            f"Conv channel-virtual basis requires out_channels divisible by groups; "
            f"got out_channels={out_channels}, groups={groups}."
        )
    out_per_group = out_channels // groups
    preadd_weight = (
        weights.view(groups, out_per_group, in_per_group, kernel_h, kernel_w)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
        .view(groups * in_per_group * out_per_group, 1, kernel_h, kernel_w)
    )
    return preadd_weight, in_per_group, out_per_group


def _compute_conv_channel_virtual_preadd(
    inputs_chunk: torch.Tensor,
    preadd_weight: torch.Tensor,
    *,
    groups: int,
    in_per_group: int,
    out_per_group: int,
    stride,
    padding,
    dilation,
) -> torch.Tensor:
    preadd_flat = F.conv2d(
        inputs_chunk,
        preadd_weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups * in_per_group,
    )
    batch_size, _, out_h, out_w = preadd_flat.shape
    preadd_grouped = preadd_flat.view(batch_size, groups, in_per_group, out_per_group, out_h, out_w)
    return (
        preadd_grouped.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(batch_size, groups * out_per_group, in_per_group, out_h, out_w)
    )


def _conv_channel_virtual_reduce_exact_chunk(
    preadd_chunk: torch.Tensor,
    grads_chunk: torch.Tensor,
    *,
    scale: float,
    outputs_chunk: Optional[torch.Tensor] = None,
    compute_neuron_metrics: bool = True,
    compute_product_metrics: bool = True,
    abs_aggregation_mode: str = "exact_abs",
    need_per_sample_scores: bool = True,
) -> Dict[str, Optional[torch.Tensor]]:
    abs_mode = normalize_abs_aggregation_mode(abs_aggregation_mode)
    frontier_abs = is_frontier_aggregate_abs_mode(abs_mode)
    _, scale_tensor, abs_scale_tensor = _resolve_scale_tensors(scale, device=preadd_chunk.device)

    grads_scaled = grads_chunk / scale_tensor
    per_sample_scores: Optional[torch.Tensor] = None
    locations = int(preadd_chunk.size(-1) * preadd_chunk.size(-2))
    use_sample_sum_reduction = bool(
        frontier_abs
        and not need_per_sample_scores
        and _frontier_aggregate_conv_reduction_mode(
            device=preadd_chunk.device,
            locations=locations,
        )
        == "sample_sum"
    )
    if need_per_sample_scores or use_sample_sum_reduction:
        per_sample_scores = torch.einsum("boihw,bohw->boi", preadd_chunk, grads_scaled)
        kernel_scores_sum = per_sample_scores.sum(dim=0)
    else:
        kernel_scores_sum = torch.einsum("boihw,bohw->oi", preadd_chunk, grads_scaled)

    if frontier_abs:
        abs_scores_sum = kernel_scores_sum.abs()
        activating_scores_sum = torch.relu(kernel_scores_sum)
        inhibiting_scores_sum = -torch.relu(-kernel_scores_sum)
        signed_acts_scores = abs_scores_sum.clone()
    else:
        if per_sample_scores is None:
            per_sample_scores = torch.einsum("boihw,bohw->boi", preadd_chunk, grads_scaled)
        abs_scores_sum = per_sample_scores.abs().sum(dim=0)
        activating_scores_sum = torch.relu(per_sample_scores).sum(dim=0)
        inhibiting_scores_sum = (-torch.relu(-per_sample_scores)).sum(dim=0)
        signed_acts_scores = torch.einsum(
            "boihw,bohw->oi",
            preadd_chunk,
            grads_chunk.abs() / scale_tensor,
        )

    spatial_grad_sum = (grads_chunk.sum(dim=(-1, -2)) / scale_tensor).sum(dim=0)
    gradients_sum = spatial_grad_sum.unsqueeze(1).expand(-1, preadd_chunk.size(2)).clone()
    if frontier_abs:
        abs_gradients_sum = gradients_sum.abs()
    else:
        spatial_abs_grad_sum = (grads_chunk.abs().sum(dim=(-1, -2)) / abs_scale_tensor).sum(dim=0)
        abs_gradients_sum = spatial_abs_grad_sum.unsqueeze(1).expand(-1, preadd_chunk.size(2)).clone()

    sample_sums = (
        per_sample_scores.sum(dim=2)
        if need_per_sample_scores and per_sample_scores is not None
        else None
    )
    sample_abs_sums = sample_sums.abs() if sample_sums is not None else None

    neuron_scores_sum: Optional[torch.Tensor] = None
    abs_neuron_scores_sum: Optional[torch.Tensor] = None
    neuron_scores_sq_sum: Optional[torch.Tensor] = None
    abs_neuron_scores_sq_sum: Optional[torch.Tensor] = None
    product_scores_sum: Optional[torch.Tensor] = None
    abs_product_scores_sum: Optional[torch.Tensor] = None

    if outputs_chunk is not None:
        neuron_scores_per_image = (outputs_chunk * grads_chunk).sum(dim=(-2, -1)) / scale_tensor
        neuron_scores_sum = neuron_scores_per_image.sum(dim=0)
        if frontier_abs:
            abs_neuron_scores_sum = neuron_scores_sum.abs()
            abs_neuron_scores_sq_sum = abs_neuron_scores_sum.square()
        else:
            abs_neuron_scores_per_image = (
                outputs_chunk * grads_chunk
            ).abs().sum(dim=(-2, -1)) / abs_scale_tensor
            abs_neuron_scores_sum = abs_neuron_scores_per_image.sum(dim=0)
            abs_neuron_scores_sq_sum = abs_neuron_scores_per_image.square().sum(dim=0)
        neuron_scores_sq_sum = neuron_scores_per_image.square().sum(dim=0)
        if compute_product_metrics:
            if per_sample_scores is None:
                per_sample_scores = torch.einsum("boihw,bohw->boi", preadd_chunk, grads_scaled)
            product_scores_sum = (per_sample_scores * neuron_scores_per_image.unsqueeze(-1)).sum(dim=0)
            if frontier_abs:
                abs_product_scores_sum = product_scores_sum.abs()
            else:
                abs_product_scores_sum = (
                    per_sample_scores.abs() * abs_neuron_scores_per_image.unsqueeze(-1)
                ).sum(dim=0)

    return {
        "per_sample_scores": per_sample_scores,
        "kernel_scores_sum": kernel_scores_sum,
        "activating_scores_sum": activating_scores_sum,
        "inhibiting_scores_sum": inhibiting_scores_sum,
        "scores_signed_acts_sum": signed_acts_scores,
        "abs_scores_sum": abs_scores_sum,
        "gradients_sum": gradients_sum,
        "abs_gradients_sum": abs_gradients_sum,
        "sample_sums": sample_sums,
        "sample_abs_sums": sample_abs_sums,
        "neuron_scores_sum": neuron_scores_sum,
        "abs_neuron_scores_sum": abs_neuron_scores_sum,
        "neuron_scores_sq_sum": neuron_scores_sq_sum,
        "abs_neuron_scores_sq_sum": abs_neuron_scores_sq_sum,
        "product_scores_sum": product_scores_sum,
        "abs_product_scores_sum": abs_product_scores_sum,
    }


def _conv_reduce_kernel_exact_chunk(
    inputs_chunk: torch.Tensor,
    grads_chunk: torch.Tensor,
    weights: torch.Tensor,
    *,
    scale: float,
    stride,
    padding,
    dilation,
    groups: int,
    outputs_chunk: Optional[torch.Tensor] = None,
    compute_neuron_metrics: bool = True,
    target_unit_acts_chunk: Optional[torch.Tensor] = None,
    return_sample_sums: bool = True,
    emit_bitpack: bool = True,
    abs_aggregation_mode: str = "exact_abs",
) -> Dict[str, Optional[torch.Tensor]]:
    abs_mode = normalize_abs_aggregation_mode(abs_aggregation_mode)
    frontier_abs = is_frontier_aggregate_abs_mode(abs_mode)
    device = weights.device
    _, scale_tensor, abs_scale_tensor = _resolve_scale_tensors(scale, device=device)

    out_channels, in_per_group, kernel_h, kernel_w = weights.shape
    if out_channels % groups != 0:
        raise ValueError(
            f"Conv kernel-weight basis requires out_channels divisible by groups; "
            f"got out_channels={out_channels}, groups={groups}."
        )
    out_per_group = out_channels // groups
    kernel_elements = kernel_h * kernel_w
    batch_size = inputs_chunk.size(0)
    outputs_h = grads_chunk.size(-2)
    outputs_w = grads_chunk.size(-1)
    locations = outputs_h * outputs_w
    grads_flat = grads_chunk.view(batch_size, groups, out_per_group, locations).contiguous()
    sample_block_size = _choose_conv_exact_sample_block_size(
        groups=groups,
        out_per_group=out_per_group,
        in_per_group=in_per_group,
        kernel_elements=kernel_elements,
        locations=locations,
        batch_size=batch_size,
    )

    score_weight = weights / scale_tensor
    abs_score_weight = weights.abs() / abs_scale_tensor
    pos_score_weight = torch.relu(score_weight)
    neg_score_weight_mag = torch.relu(-score_weight)
    score_weight_grouped = score_weight.view(groups, out_per_group, in_per_group, kernel_h, kernel_w)
    abs_score_weight_grouped = abs_score_weight.view(groups, out_per_group, in_per_group, kernel_h, kernel_w)
    pos_score_weight_grouped = pos_score_weight.view(groups, out_per_group, in_per_group, kernel_h, kernel_w)
    neg_score_weight_mag_grouped = neg_score_weight_mag.view(groups, out_per_group, in_per_group, kernel_h, kernel_w)

    signed_grad_total_grouped = torch.zeros_like(score_weight_grouped, dtype=torch.float32, device=device)
    abs_grad_total_grouped = (
        None if frontier_abs else torch.zeros_like(score_weight_grouped, dtype=torch.float32, device=device)
    )
    pos_grad_total_grouped = (
        None if frontier_abs else torch.zeros_like(score_weight_grouped, dtype=torch.float32, device=device)
    )
    neg_grad_mag_total_grouped = (
        None if frontier_abs else torch.zeros_like(score_weight_grouped, dtype=torch.float32, device=device)
    )

    need_sample_rows = bool(return_sample_sums or target_unit_acts_chunk is not None or emit_bitpack)
    sample_sums = (
        torch.zeros((batch_size, out_channels), dtype=torch.float32, device=device)
        if (return_sample_sums or target_unit_acts_chunk is not None)
        else None
    )
    sample_abs_sums = torch.zeros_like(sample_sums) if sample_sums is not None else None
    sample_score_totals = (
        torch.zeros((batch_size,), dtype=torch.float32, device=device)
        if target_unit_acts_chunk is not None
        else None
    )
    bitpacked: List[torch.Tensor] = []

    neuron_scores_sum: Optional[torch.Tensor] = None
    abs_neuron_scores_sum: Optional[torch.Tensor] = None
    neuron_scores_sq_sum: Optional[torch.Tensor] = None
    abs_neuron_scores_sq_sum: Optional[torch.Tensor] = None
    product_scores_sum: Optional[torch.Tensor] = None
    abs_product_scores_sum: Optional[torch.Tensor] = None
    corr: Optional[torch.Tensor] = None

    neuron_scores_per_image: Optional[torch.Tensor] = None
    abs_neuron_scores_per_image: Optional[torch.Tensor] = None
    grad_times_neuron_total: Optional[torch.Tensor] = None
    abs_grad_times_abs_neuron_total: Optional[torch.Tensor] = None

    if outputs_chunk is not None:
        neuron_scores_per_image = (outputs_chunk * grads_chunk).sum(dim=(-2, -1)) / scale_tensor
        neuron_scores_sum = neuron_scores_per_image.sum(dim=0)
        abs_neuron_scores_sum = (
            neuron_scores_sum.abs()
            if frontier_abs
            else neuron_scores_per_image.abs().sum(dim=0)
        )
        neuron_scores_sq_sum = neuron_scores_per_image.square().sum(dim=0)
        abs_neuron_scores_sq_sum = (
            abs_neuron_scores_sum.square()
            if frontier_abs and abs_neuron_scores_sum is not None
            else neuron_scores_per_image.abs().square().sum(dim=0)
        )
        if compute_neuron_metrics:
            grad_times_neuron_total = torch.zeros_like(score_weight_grouped, dtype=torch.float32, device=device)
            if not frontier_abs:
                abs_neuron_scores_per_image = neuron_scores_per_image.abs()
                abs_grad_times_abs_neuron_total = torch.zeros_like(
                    score_weight_grouped,
                    dtype=torch.float32,
                    device=device,
                )

    for start in range(0, batch_size, sample_block_size):
        end = min(start + sample_block_size, batch_size)
        block_size = end - start
        unfolded_block = F.unfold(
            inputs_chunk[start:end],
            kernel_size=(kernel_h, kernel_w),
            dilation=dilation,
            padding=padding,
            stride=stride,
        )
        patches_block = unfolded_block.view(
            block_size,
            groups,
            in_per_group * kernel_elements,
            locations,
        ).contiguous()
        grads_block = grads_flat[start:end]
        grad_sample_grouped = torch.bmm(
            grads_block.reshape(block_size * groups, out_per_group, locations),
            patches_block.reshape(block_size * groups, in_per_group * kernel_elements, locations).transpose(1, 2),
        )
        grad_sample_grouped = grad_sample_grouped.view(
            block_size,
            groups,
            out_per_group,
            in_per_group,
            kernel_h,
            kernel_w,
        )
        signed_grad_total_grouped.add_(grad_sample_grouped.sum(dim=0))
        if not frontier_abs:
            abs_grad_sample_grouped = grad_sample_grouped.abs()
            abs_grad_total_grouped.add_(abs_grad_sample_grouped.sum(dim=0))
            pos_grad_total_grouped.add_(torch.relu(grad_sample_grouped).sum(dim=0))
            neg_grad_mag_total_grouped.add_(torch.relu(-grad_sample_grouped).sum(dim=0))

        if need_sample_rows:
            sample_channel_scores = (
                score_weight_grouped.unsqueeze(0) * grad_sample_grouped
            ).sum(dim=(-1, -2))
            sample_rows = sample_channel_scores.sum(dim=-1).reshape(block_size, out_channels)
            if sample_sums is not None:
                sample_sums[start:end] = sample_rows
                if sample_abs_sums is not None:
                    sample_abs_sums[start:end] = sample_rows.abs()
            if sample_score_totals is not None:
                sample_score_totals[start:end] = sample_rows.sum(dim=1)
            if emit_bitpack:
                packed_rows = pack_boolean_rows(
                    sample_channel_scores.abs().reshape(block_size * out_channels, -1) > 0
                )
                bitpacked.extend(row.cpu() for row in packed_rows)

        if (
            compute_neuron_metrics
            and grad_times_neuron_total is not None
            and neuron_scores_per_image is not None
        ):
            neuron_scale = neuron_scores_per_image[start:end].view(
                block_size,
                groups,
                out_per_group,
                1,
                1,
                1,
            )
            grad_times_neuron_total.add_((grad_sample_grouped * neuron_scale).sum(dim=0))
            if (
                not frontier_abs
                and abs_grad_times_abs_neuron_total is not None
                and abs_neuron_scores_per_image is not None
            ):
                abs_neuron_scale = abs_neuron_scores_per_image[start:end].view(
                    block_size,
                    groups,
                    out_per_group,
                    1,
                    1,
                    1,
                )
                abs_grad_times_abs_neuron_total.add_(
                    (grad_sample_grouped.abs() * abs_neuron_scale).sum(dim=0)
                )

    if target_unit_acts_chunk is not None and sample_score_totals is not None:
        totals = sample_score_totals
        corr = safe_pearson_corr(
            totals,
            target_unit_acts_chunk.to(dtype=torch.float32, device=device),
        )

    signed_grad_total = signed_grad_total_grouped.view(out_channels, in_per_group, kernel_h, kernel_w)
    kernel_scores_sum = score_weight * signed_grad_total
    gradients_sum = signed_grad_total / scale_tensor
    if frontier_abs:
        activating_scores_sum = torch.relu(kernel_scores_sum)
        inhibiting_scores_sum = -torch.relu(-kernel_scores_sum)
        scores_signed_acts_sum = score_weight * signed_grad_total.abs()
        abs_scores_sum = kernel_scores_sum.abs()
        abs_gradients_sum = gradients_sum.abs()
    else:
        abs_grad_total = abs_grad_total_grouped.view(out_channels, in_per_group, kernel_h, kernel_w)
        pos_grad_total = pos_grad_total_grouped.view(out_channels, in_per_group, kernel_h, kernel_w)
        neg_grad_mag_total = neg_grad_mag_total_grouped.view(out_channels, in_per_group, kernel_h, kernel_w)
        activating_scores_sum = (
            pos_score_weight * pos_grad_total
            + neg_score_weight_mag * neg_grad_mag_total
        )
        inhibiting_scores_sum = -(
            pos_score_weight * neg_grad_mag_total
            + neg_score_weight_mag * pos_grad_total
        )
        scores_signed_acts_sum = score_weight * abs_grad_total
        abs_scores_sum = abs_score_weight * abs_grad_total
        abs_gradients_sum = abs_grad_total / abs_scale_tensor

    if (
        compute_neuron_metrics
        and grad_times_neuron_total is not None
    ):
        grad_times_neuron_total = grad_times_neuron_total.view(out_channels, in_per_group, kernel_h, kernel_w)
        product_scores_sum = score_weight * grad_times_neuron_total
        if frontier_abs:
            abs_product_scores_sum = product_scores_sum.abs()
        else:
            abs_grad_times_abs_neuron_total = abs_grad_times_abs_neuron_total.view(
                out_channels,
                in_per_group,
                kernel_h,
                kernel_w,
            )
            abs_product_scores_sum = abs_score_weight * abs_grad_times_abs_neuron_total

    return {
        "kernel_scores_sum": kernel_scores_sum,
        "activating_scores_sum": activating_scores_sum,
        "inhibiting_scores_sum": inhibiting_scores_sum,
        "scores_signed_acts_sum": scores_signed_acts_sum,
        "abs_scores_sum": abs_scores_sum,
        "gradients_sum": gradients_sum,
        "abs_gradients_sum": abs_gradients_sum,
        "sample_sums": sample_sums,
        "sample_abs_sums": sample_abs_sums,
        "neuron_scores_sum": neuron_scores_sum,
        "abs_neuron_scores_sum": abs_neuron_scores_sum,
        "neuron_scores_sq_sum": neuron_scores_sq_sum,
        "abs_neuron_scores_sq_sum": abs_neuron_scores_sq_sum,
        "product_scores_sum": product_scores_sum,
        "abs_product_scores_sum": abs_product_scores_sum,
        "corr": corr,
        "bitpacked": bitpacked,
    }


@torch._dynamo.disable
def _conv_compute_channel_virtual_score_aggregated(
    inputs_source: Union[torch.Tensor, TensorReplayProvider],
    saved_neuron_gradients: torch.Tensor,
    weights: torch.Tensor,
    scale: float,
    stride,
    padding,
    dilation,
    groups,
    device: torch.device,
    compute_neuron_metrics: bool = True,
    *,
    bitpack_sink: Optional[Callable[[torch.Tensor], None]] = None,
    bias: Optional[torch.Tensor] = None,
    validate_frontier: bool = False,
    target_acts: Optional[torch.Tensor] = None,
    stats_callback: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    return_sample_sums: bool = False,
    debug_layer_name: Optional[str] = None,
    emit_bitpack: bool = True,
    abs_aggregation_mode: str = "exact_abs",
    preadd_weight: Optional[torch.Tensor] = None,
    in_per_group: Optional[int] = None,
    out_per_group: Optional[int] = None,
) -> Tuple:
    if (
        bitpack_sink is None
        and bias is None
        and not validate_frontier
        and target_acts is None
        and stats_callback is None
        and not return_sample_sums
        and not debug_layer_name
        and not emit_bitpack
    ):
        from packages.salience.ops.conv_frontier_aggregate import (
            conv_compute_frontier_aggregate_channel_virtual_preadd,
        )

        return conv_compute_frontier_aggregate_channel_virtual_preadd(
            inputs_source=inputs_source,
            grads=saved_neuron_gradients,
            weights=weights,
            scale=scale,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            device=device,
            abs_aggregation_mode=abs_aggregation_mode,
            compute_neuron_metrics=compute_neuron_metrics,
            # The preadd fast path must emit the same payload as the slow path
            # for identical inputs. The slow path (and dissected_Conv2d /
            # frontier_conv) populate product_scores whenever neuron metrics are
            # requested, so forward the product flag too rather than silently
            # dropping product_scores/abs_product_scores on this branch.
            compute_product_metrics=compute_neuron_metrics,
            preadd_weight=preadd_weight,
            in_per_group=in_per_group,
            out_per_group=out_per_group,
        )
    with torch.no_grad():
        frontier_abs = is_frontier_aggregate_abs_mode(abs_aggregation_mode)
        scale_value, scale_tensor, _ = _resolve_scale_tensors(scale, device=device)
        basis = _normalize_conv_score_basis("channel_virtual")
        del basis

        out_features = weights.size(0)
        in_features = weights.size(1)
        chunk_size = choose_chunk_size(out_features, in_features, 1)
        batch_size = _replay_batch_size(inputs_source, saved_neuron_gradients.size(0))
        if preadd_weight is None or in_per_group is None or out_per_group is None:
            preadd_weight, in_per_group, out_per_group = _build_conv_channel_virtual_weight(weights, groups=groups)
        preadd_weight = preadd_weight.to(device=device, dtype=torch.float32)
        in_per_group = int(in_per_group)
        out_per_group = int(out_per_group)

        kernel_scores_total = torch.zeros((out_features, in_features), dtype=torch.float32, device=device)
        activating_scores_total = torch.zeros_like(kernel_scores_total)
        inhibiting_scores_total = torch.zeros_like(kernel_scores_total)
        scores_signed_acts_total = torch.zeros_like(kernel_scores_total)
        abs_kernel_scores_total = torch.zeros_like(kernel_scores_total)
        gradients_total = torch.zeros_like(kernel_scores_total)
        abs_gradients_total = torch.zeros_like(kernel_scores_total)

        sample_sum_accum = (
            torch.zeros((batch_size, out_features), dtype=torch.float32, device=device)
            if return_sample_sums
            else None
        )
        sample_abs_sum_accum = (
            torch.zeros((batch_size, out_features), dtype=torch.float32, device=device)
            if return_sample_sums
            else None
        )
        layer_corr_scores: Optional[List[torch.Tensor]] = [] if debug_layer_name else None
        layer_corr_objectives: Optional[List[torch.Tensor]] = [] if debug_layer_name else None

        if bias is not None:
            bias = bias.to(device=device, dtype=torch.float32)

        if compute_neuron_metrics:
            neuron_scores_total = torch.zeros(out_features, dtype=torch.float32, device=device)
            abs_neuron_scores_total = torch.zeros_like(neuron_scores_total)
            neuron_scores_sq_total = torch.zeros_like(neuron_scores_total)
            abs_neuron_scores_sq_total = torch.zeros_like(neuron_scores_total)
            product_scores_total = torch.zeros_like(kernel_scores_total)
            abs_product_scores_total = torch.zeros_like(kernel_scores_total)
        else:
            neuron_scores_total = None
            abs_neuron_scores_total = None
            neuron_scores_sq_total = None
            abs_neuron_scores_sq_total = None
            product_scores_total = None
            abs_product_scores_total = None

        frontier_lhs = torch.zeros((), device=device, dtype=torch.float64) if validate_frontier else None
        frontier_rhs = torch.zeros_like(frontier_lhs) if validate_frontier else None
        frontier_bias = torch.zeros_like(frontier_lhs) if (validate_frontier and bias is not None) else None

        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            inputs_chunk = _fetch_replay_slice(
                inputs_source,
                start,
                end,
                device=device,
                dtype=torch.float32,
            )
            grads_chunk = saved_neuron_gradients[start:end].to(device=device, dtype=torch.float32)
            target_chunk = None
            if target_acts is not None:
                target_chunk = target_acts[start:end].to(device=device, dtype=torch.float32)

            preadd_chunk = _compute_conv_channel_virtual_preadd(
                inputs_chunk,
                preadd_weight,
                groups=groups,
                in_per_group=in_per_group,
                out_per_group=out_per_group,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )

            need_outputs = compute_neuron_metrics or validate_frontier
            outputs_chunk = preadd_chunk.sum(dim=2) if need_outputs else None

            chunk_stats = _conv_channel_virtual_reduce_exact_chunk(
                preadd_chunk,
                grads_chunk,
                scale=scale,
                outputs_chunk=outputs_chunk,
                compute_neuron_metrics=compute_neuron_metrics,
                abs_aggregation_mode=abs_aggregation_mode,
                need_per_sample_scores=bool(
                    return_sample_sums
                    or target_chunk is not None
                    or emit_bitpack
                    or compute_neuron_metrics
                    or debug_layer_name
                    or stats_callback is not None
                ),
            )
            kernel_scores_total += chunk_stats["kernel_scores_sum"]
            activating_scores_total += chunk_stats["activating_scores_sum"]
            inhibiting_scores_total += chunk_stats["inhibiting_scores_sum"]
            scores_signed_acts_total += chunk_stats["scores_signed_acts_sum"]
            abs_kernel_scores_total += chunk_stats["abs_scores_sum"]
            gradients_total += chunk_stats["gradients_sum"]
            abs_gradients_total += chunk_stats["abs_gradients_sum"]

            sample_sums = chunk_stats["sample_sums"]
            sample_abs_sums = chunk_stats["sample_abs_sums"]
            if compute_neuron_metrics:
                neuron_scores_total += chunk_stats["neuron_scores_sum"]
                abs_neuron_scores_total += chunk_stats["abs_neuron_scores_sum"]
                neuron_scores_sq_total += chunk_stats["neuron_scores_sq_sum"]
                abs_neuron_scores_sq_total += chunk_stats["abs_neuron_scores_sq_sum"]
                product_scores_total += chunk_stats["product_scores_sum"]
                abs_product_scores_total += chunk_stats["abs_product_scores_sum"]

            if debug_layer_name:
                per_sample_scores = sample_sums.sum(dim=1)
                objective_vec = _flatten_objective_vec(target_chunk)
                _maybe_record_layer_corr_sample(
                    layer_corr_scores,
                    layer_corr_objectives,
                    per_sample_scores,
                    objective_vec,
                )

            if return_sample_sums and sample_sum_accum is not None:
                sample_sum_accum[start:end] = sample_sums
                if sample_abs_sum_accum is not None:
                    sample_abs_sum_accum[start:end] = sample_abs_sums

            if stats_callback is not None:
                phi_rows = sample_sums
                if bias is not None and not _EXCLUDE_BIAS_FROM_SAMPLE_SUMS:
                    phi_rows = phi_rows + (grads_chunk.sum(dim=(2, 3)) * bias.view(1, -1)) / scale_tensor
                stats_callback(phi_rows, target_chunk)

            if validate_frontier and outputs_chunk is not None:
                frontier_lhs += chunk_stats["kernel_scores_sum"].sum().to(torch.float64)
                frontier_rhs += (outputs_chunk * (grads_chunk / scale_tensor)).sum().to(torch.float64)
                if frontier_bias is not None:
                    frontier_bias += (bias * (grads_chunk / scale_tensor).sum(dim=(0, 2, 3))).sum().to(torch.float64)

        _finalize_layer_corr_logging(debug_layer_name, layer_corr_scores, layer_corr_objectives)

        if frontier_abs:
            abs_kernel_scores_total = kernel_scores_total.abs()
            activating_scores_total = torch.relu(kernel_scores_total)
            inhibiting_scores_total = -torch.relu(-kernel_scores_total)
            scores_signed_acts_total = abs_kernel_scores_total.clone()
            abs_gradients_total = gradients_total.abs()
            if compute_neuron_metrics:
                abs_neuron_scores_total = neuron_scores_total.abs()
                abs_neuron_scores_sq_total = abs_neuron_scores_total.square()
                abs_product_scores_total = product_scores_total.abs()

        if validate_frontier and frontier_lhs is not None and frontier_rhs is not None:
            lhs_total = frontier_lhs
            rhs_total = frontier_rhs
            if frontier_bias is not None:
                lhs_total = lhs_total + frontier_bias
                rhs_total = rhs_total + frontier_bias
            if not torch.allclose(lhs_total, rhs_total, rtol=1e-3, atol=1e-3):
                raise AssertionError(
                    f"Conv channel-virtual completeness failed: phi={lhs_total.item():.6f}, "
                    f"z·m={rhs_total.item():.6f}"
                )

        grad_non_zeroes = torch.tensor(torch.count_nonzero(gradients_total).item(), device=device)
        flat_kernel = abs_kernel_scores_total.reshape(abs_kernel_scores_total.size(0), -1)
        if emit_bitpack:
            packed_rows = pack_boolean_rows(flat_kernel > 0)
            if bitpack_sink is not None and packed_rows.numel() > 0:
                bitpack_sink(packed_rows.cpu())
                bitpacked = []
            else:
                bitpacked = [row.cpu() for row in packed_rows]
        else:
            bitpacked = []

        if compute_neuron_metrics:
            neuron_scores = neuron_scores_total
            abs_neuron_scores = abs_neuron_scores_total
            neuron_scores_squared = neuron_scores_sq_total
            abs_neuron_scores_squared = abs_neuron_scores_sq_total
            product_scores = product_scores_total
            abs_product_scores = abs_product_scores_total
        else:
            neuron_scores = None
            abs_neuron_scores = None
            neuron_scores_squared = None
            abs_neuron_scores_squared = None
            product_scores = None
            abs_product_scores = None

        dummy = torch.zeros(1, device=device)
        sample_sum_payload = sample_sum_accum if sample_sum_accum is not None else None
        sample_abs_payload = sample_abs_sum_accum if sample_abs_sum_accum is not None else None

        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            sample_sum_payload,
            sample_abs_payload,
            dummy,
            dummy,
            dummy,
            kernel_scores_total,
            activating_scores_total,
            inhibiting_scores_total,
            scores_signed_acts_total,
            abs_kernel_scores_total,
            gradients_total,
            abs_gradients_total,
            None,
            None,
            None,
            None,
            None,
            None,
            neuron_scores,
            abs_neuron_scores,
            neuron_scores_squared,
            abs_neuron_scores_squared,
            product_scores,
            abs_product_scores,
            dummy,
            dummy,
            weights,
            grad_non_zeroes,
            None,
            None,
            bitpacked,
        )


@torch._dynamo.disable
def conv_compute_kernel_score_aggregated(
    inputs_source: Union[torch.Tensor, TensorReplayProvider],
    saved_neuron_gradients: torch.Tensor,
    weights: torch.Tensor,
    scale: float,
    stride,
    padding,
    dilation,
    groups,
    device: torch.device,
    compute_neuron_metrics: bool = True,
    *,
    bitpack_sink: Optional[Callable[[torch.Tensor], None]] = None,
    bias: Optional[torch.Tensor] = None,
    validate_frontier: bool = False,
    target_acts: Optional[torch.Tensor] = None,
    stats_callback: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    return_sample_sums: bool = False,
    debug_layer_name: Optional[str] = None,
    score_basis: str = "kernel_weight",
    dual_channel_virtual: bool = False,
    auxiliary_payload: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    emit_bitpack: bool = True,
    abs_aggregation_mode: str = "exact_abs",
    preadd_weight: Optional[torch.Tensor] = None,
    in_per_group: Optional[int] = None,
    out_per_group: Optional[int] = None,
) -> Tuple:
    basis = _normalize_conv_score_basis(score_basis)
    if basis == "channel_virtual":
        return _conv_compute_channel_virtual_score_aggregated(
            inputs_source=inputs_source,
            saved_neuron_gradients=saved_neuron_gradients,
            weights=weights,
            scale=scale,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            device=device,
            compute_neuron_metrics=compute_neuron_metrics,
            bitpack_sink=bitpack_sink,
            bias=bias,
            validate_frontier=validate_frontier,
            target_acts=target_acts,
            stats_callback=stats_callback,
            return_sample_sums=return_sample_sums,
            debug_layer_name=debug_layer_name,
            emit_bitpack=emit_bitpack,
            abs_aggregation_mode=abs_aggregation_mode,
            preadd_weight=preadd_weight,
            in_per_group=in_per_group,
            out_per_group=out_per_group,
        )
    with torch.no_grad():
        frontier_abs = is_frontier_aggregate_abs_mode(abs_aggregation_mode)
        scale_value, _, _ = _resolve_scale_tensors(scale, device=device)

        out_features = weights.size(0)
        in_features = weights.size(1)
        spatial_elements = weights.shape[2] * weights.shape[3] if weights.dim() == 4 else 1
        chunk_size = choose_chunk_size(out_features, in_features, spatial_elements)
        batch_size = _replay_batch_size(inputs_source, saved_neuron_gradients.size(0))

        sample_sum_accum = (
            torch.zeros((batch_size, out_features), dtype=torch.float32, device=device)
            if return_sample_sums
            else None
        )
        sample_abs_sum_accum = (
            torch.zeros((batch_size, out_features), dtype=torch.float32, device=device)
            if return_sample_sums
            else None
        )
        layer_corr_scores: Optional[List[torch.Tensor]] = [] if debug_layer_name else None
        layer_corr_objectives: Optional[List[torch.Tensor]] = [] if debug_layer_name else None

        kernel_scores_total = torch.zeros_like(weights, dtype=torch.float32, device=device)
        activating_scores_total = torch.zeros_like(weights, dtype=torch.float32, device=device)
        inhibiting_scores_total = torch.zeros_like(weights, dtype=torch.float32, device=device)
        scores_signed_acts_total = torch.zeros_like(weights, dtype=torch.float32, device=device)
        abs_kernel_scores_total = torch.zeros_like(weights, dtype=torch.float32, device=device)
        gradients_total = torch.zeros_like(weights, dtype=torch.float32, device=device)
        abs_gradients_total = torch.zeros_like(weights, dtype=torch.float32, device=device)
        bitpacked: List[torch.Tensor] = []

        if bias is not None:
            bias = bias.to(device=device, dtype=torch.float32)

        if compute_neuron_metrics:
            neuron_scores_total = torch.zeros(weights.size(0), dtype=torch.float32, device=device)
            abs_neuron_scores_total = torch.zeros_like(neuron_scores_total)
            neuron_scores_sq_total = torch.zeros_like(neuron_scores_total)
            abs_neuron_scores_sq_total = torch.zeros_like(neuron_scores_total)
            product_scores_total = torch.zeros_like(weights, dtype=torch.float32, device=device)
            abs_product_scores_total = torch.zeros_like(weights, dtype=torch.float32, device=device)
        else:
            neuron_scores_total = None
            abs_neuron_scores_total = None
            neuron_scores_sq_total = None
            abs_neuron_scores_sq_total = None
            product_scores_total = None
            abs_product_scores_total = None

        frontier_lhs = torch.zeros((), device=device, dtype=torch.float64) if validate_frontier else None
        frontier_rhs = torch.zeros_like(frontier_lhs) if validate_frontier else None
        frontier_bias = torch.zeros_like(frontier_lhs) if (validate_frontier and bias is not None) else None
        dual_enabled = bool(dual_channel_virtual)
        dual_preadd_weight = None
        dual_in_per_group = 0
        dual_out_per_group = 0
        dual_kernel_scores_total = None
        dual_scores_signed_acts_total = None
        dual_abs_scores_total = None
        dual_gradients_total = None
        dual_abs_gradients_total = None
        if dual_enabled:
            dual_preadd_weight, dual_in_per_group, dual_out_per_group = _build_conv_channel_virtual_weight(
                weights,
                groups=groups,
            )
            dual_preadd_weight = dual_preadd_weight.to(device=device, dtype=torch.float32)
            dual_shape = (out_features, in_features)
            dual_kernel_scores_total = torch.zeros(dual_shape, dtype=torch.float32, device=device)
            dual_scores_signed_acts_total = torch.zeros_like(dual_kernel_scores_total)
            dual_abs_scores_total = torch.zeros_like(dual_kernel_scores_total)
            dual_gradients_total = torch.zeros_like(dual_kernel_scores_total)
            dual_abs_gradients_total = torch.zeros_like(dual_kernel_scores_total)

        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            inputs_chunk = _fetch_replay_slice(
                inputs_source,
                start,
                end,
                device=device,
                dtype=torch.float32,
            )
            grads_chunk = saved_neuron_gradients[start:end].to(device=device, dtype=torch.float32)
            target_chunk = None
            if target_acts is not None:
                target_chunk = target_acts[start:end].to(device=device, dtype=torch.float32)

            need_outputs = (
                compute_neuron_metrics
                or validate_frontier
                or bool(debug_layer_name)
            )
            outputs_chunk = None
            if need_outputs:
                outputs_chunk = torch.nn.functional.conv2d(
                    inputs_chunk,
                    weights,
                    bias=None,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )

            chunk_stats = _conv_reduce_kernel_exact_chunk(
                inputs_chunk,
                grads_chunk,
                weights,
                scale=scale_value,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                outputs_chunk=outputs_chunk,
                compute_neuron_metrics=compute_neuron_metrics,
                target_unit_acts_chunk=target_chunk if debug_layer_name else None,
                return_sample_sums=return_sample_sums,
                emit_bitpack=emit_bitpack,
                abs_aggregation_mode=abs_aggregation_mode,
            )
            if dual_enabled and dual_preadd_weight is not None:
                dual_preadd_chunk = _compute_conv_channel_virtual_preadd(
                    inputs_chunk,
                    dual_preadd_weight,
                    groups=groups,
                    in_per_group=dual_in_per_group,
                    out_per_group=dual_out_per_group,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                )
                dual_chunk_stats = _conv_channel_virtual_reduce_exact_chunk(
                    dual_preadd_chunk,
                    grads_chunk,
                    scale=scale_value,
                    outputs_chunk=None,
                    compute_neuron_metrics=False,
                    abs_aggregation_mode=abs_aggregation_mode,
                    need_per_sample_scores=False,
                )
                dual_kernel_scores_total += dual_chunk_stats["kernel_scores_sum"]
                dual_scores_signed_acts_total += dual_chunk_stats["scores_signed_acts_sum"]
                dual_abs_scores_total += dual_chunk_stats["abs_scores_sum"]
                dual_gradients_total += dual_chunk_stats["gradients_sum"]
                dual_abs_gradients_total += dual_chunk_stats["abs_gradients_sum"]

            kernel_scores_total += chunk_stats["kernel_scores_sum"]
            activating_scores_total += chunk_stats["activating_scores_sum"]
            inhibiting_scores_total += chunk_stats["inhibiting_scores_sum"]
            scores_signed_acts_total += chunk_stats["scores_signed_acts_sum"]
            abs_kernel_scores_total += chunk_stats["abs_scores_sum"]
            gradients_total += chunk_stats["gradients_sum"]
            abs_gradients_total += chunk_stats["abs_gradients_sum"]

            if compute_neuron_metrics and chunk_stats["neuron_scores_sum"] is not None:
                neuron_scores_total += chunk_stats["neuron_scores_sum"]
                abs_neuron_scores_total += chunk_stats["abs_neuron_scores_sum"]
                neuron_scores_sq_total += chunk_stats["neuron_scores_sq_sum"]
                abs_neuron_scores_sq_total += chunk_stats["abs_neuron_scores_sq_sum"]
                product_scores_total += chunk_stats["product_scores_sum"]
                abs_product_scores_total += chunk_stats["abs_product_scores_sum"]

            chunk_sample_sums = chunk_stats["sample_sums"]
            chunk_sample_abs_sums = chunk_stats["sample_abs_sums"]
            if debug_layer_name and chunk_sample_sums is not None:
                per_sample_scores = chunk_sample_sums.sum(dim=1)
                objective_vec = _flatten_objective_vec(target_chunk)
                _maybe_record_layer_corr_sample(
                    layer_corr_scores,
                    layer_corr_objectives,
                    per_sample_scores,
                    objective_vec,
                )
            if return_sample_sums and sample_sum_accum is not None:
                sample_sum_accum[start:end] = chunk_sample_sums
                if sample_abs_sum_accum is not None:
                    sample_abs_sum_accum[start:end] = chunk_sample_abs_sums
            if stats_callback is not None and chunk_sample_sums is not None:
                phi_rows = chunk_sample_sums
                if bias is not None and not _EXCLUDE_BIAS_FROM_SAMPLE_SUMS:
                    phi_rows = phi_rows + (grads_chunk.sum(dim=(2, 3)) * bias.view(1, -1)) / scale_value
                stats_callback(phi_rows, target_chunk)
            if bitpack_sink is None:
                bitpacked.extend(chunk_stats["bitpacked"])
            elif chunk_stats["bitpacked"]:
                packed_tensor = torch.stack(chunk_stats["bitpacked"], dim=0)
                if packed_tensor.numel() > 0:
                    bitpack_sink(packed_tensor.cpu())

            if validate_frontier:
                frontier_lhs += chunk_sample_sums.sum().to(torch.float64)
                frontier_rhs += (outputs_chunk * (grads_chunk / scale_value)).sum().to(torch.float64)
                if frontier_bias is not None:
                    frontier_bias += (
                        bias * (grads_chunk.sum(dim=(0, 2, 3)) / scale_value)
                    ).sum().to(torch.float64)

        _finalize_layer_corr_logging(
            debug_layer_name,
            layer_corr_scores,
            layer_corr_objectives,
        )

        if frontier_abs:
            abs_kernel_scores_total = kernel_scores_total.abs()
            activating_scores_total = torch.relu(kernel_scores_total)
            inhibiting_scores_total = -torch.relu(-kernel_scores_total)
            scale_sign = -1.0 if scale_value < 0 else 1.0
            scores_signed_acts_total = (
                weights.to(dtype=torch.float32, device=device)
                * gradients_total.abs()
                * scale_sign
            )
            abs_gradients_total = gradients_total.abs()
            if compute_neuron_metrics:
                abs_neuron_scores_total = neuron_scores_total.abs()
                abs_neuron_scores_sq_total = abs_neuron_scores_total.square()
                abs_product_scores_total = product_scores_total.abs()

        if validate_frontier and frontier_lhs is not None and frontier_rhs is not None:
            lhs_total = frontier_lhs
            rhs_total = frontier_rhs
            if frontier_bias is not None:
                lhs_total = lhs_total + frontier_bias
                rhs_total = rhs_total + frontier_bias
            if not torch.allclose(lhs_total, rhs_total, rtol=1e-3, atol=1e-3):
                raise AssertionError(
                    f"Conv frontier completeness failed: phi={lhs_total.item():.6f}, z·m={rhs_total.item():.6f}"
                )

        grad_non_zeroes = torch.tensor(
            int(torch.count_nonzero(gradients_total).item()),
            device=device,
            dtype=torch.int64,
        )

        if compute_neuron_metrics:
            neuron_scores = neuron_scores_total
            abs_neuron_scores = abs_neuron_scores_total
            neuron_scores_squared = neuron_scores_sq_total
            abs_neuron_scores_squared = abs_neuron_scores_sq_total
            product_scores = product_scores_total
            abs_product_scores = abs_product_scores_total
        else:
            neuron_scores = None
            abs_neuron_scores = None
            neuron_scores_squared = None
            abs_neuron_scores_squared = None
            product_scores = None
            abs_product_scores = None

        dummy = torch.zeros(1, device=device)
        sample_sum_payload = sample_sum_accum if sample_sum_accum is not None else None
        sample_abs_payload = sample_abs_sum_accum if sample_abs_sum_accum is not None else None

        if (
            dual_enabled
            and auxiliary_payload is not None
            and dual_kernel_scores_total is not None
            and dual_scores_signed_acts_total is not None
            and dual_abs_scores_total is not None
            and dual_gradients_total is not None
            and dual_abs_gradients_total is not None
        ):
            auxiliary_payload["channel_virtual"] = {
                "kernel_scores_sum": dual_kernel_scores_total.detach().cpu(),
                "scores_signed_acts_sum": dual_scores_signed_acts_total.detach().cpu(),
                "abs_scores_sum": dual_abs_scores_total.detach().cpu(),
                "gradients_sum": dual_gradients_total.detach().cpu(),
                "abs_gradients_sum": dual_abs_gradients_total.detach().cpu(),
            }

        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            sample_sum_payload,
            sample_abs_payload,
            dummy,
            dummy,
            dummy,
            kernel_scores_total,
            activating_scores_total,
            inhibiting_scores_total,
            scores_signed_acts_total,
            abs_kernel_scores_total,
            gradients_total,
            abs_gradients_total,
            None,
            None,
            None,
            None,
            None,
            None,
            neuron_scores,
            abs_neuron_scores,
            neuron_scores_squared,
            abs_neuron_scores_squared,
            product_scores,
            abs_product_scores,
            dummy,
            dummy,
            weights,
            grad_non_zeroes,
            None,
            None,
            bitpacked,
        )


def _per_sample_conv2d_weight(
    inputs_chunk: torch.Tensor,
    grads_chunk: torch.Tensor,
    weight_shape: torch.Size,
    *,
    stride,
    padding,
    dilation,
    groups,
) -> torch.Tensor:
    """
    Compute per-sample conv2d weight gradients using vmap when available, falling
    back to a simple loop otherwise. Returns a tensor of shape
    [B, Cout, Cin/groups, kH, kW].
    """
    batch = inputs_chunk.size(0)

    inputs_chunk = inputs_chunk.contiguous()
    grads_chunk = grads_chunk.contiguous()

    try:
        try:
            from torch.func import vmap  # PyTorch >= 2.0
        except ImportError:
            from functorch import vmap  # type: ignore

        def single_grad(input_sample: torch.Tensor, grad_sample: torch.Tensor) -> torch.Tensor:
            input_sample = input_sample.unsqueeze(0)
            grad_sample = grad_sample.unsqueeze(0)
            return torch.nn.grad.conv2d_weight(
                input_sample,
                weight_size=weight_shape,
                grad_output=grad_sample,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )

        grad_weight = vmap(single_grad)(inputs_chunk, grads_chunk)
    except Exception:
        grad_list = []
        for idx in range(batch):
            grad_single = torch.nn.grad.conv2d_weight(
                inputs_chunk[idx : idx + 1],
                weight_size=weight_shape,
                grad_output=grads_chunk[idx : idx + 1],
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
            grad_list.append(grad_single)
        grad_weight = torch.stack(grad_list, dim=0)

    return grad_weight

__all__ = ['conv_compute_kernel_score_aggregated', '_per_sample_conv2d_weight']
# When set to 1, per-sample sums returned by conv scoring exclude the bias
_EXCLUDE_BIAS_FROM_SAMPLE_SUMS = bool(int(os.environ.get("SCORING_EXCLUDE_BIAS_FROM_SAMPLE_SUMS", "0")))
