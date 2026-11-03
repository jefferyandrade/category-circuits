from __future__ import annotations

import math
import os
from typing import Dict, Optional, Tuple, Union

import torch

from packages.salience.ops.abs_mode import is_frontier_aggregate_abs_mode
from packages.salience.utils import (
    TensorReplayProvider,
    fetch_replay_slice as _fetch_replay_slice,
    replay_batch_size as _replay_batch_size,
)
from packages.salience.ops.conv_emit import (
    _build_conv_channel_virtual_weight,
    _compute_conv_channel_virtual_preadd,
    _conv_channel_virtual_reduce_exact_chunk,
)

_FUSED_BACKEND = None
_FUSED_BACKEND_ERROR: Optional[Exception] = None


_SUPPORTED_KERNEL_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw not in {"0", "false", "False", "no", "No", ""}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return max(1, int(raw))
    except Exception:
        return default


def _resolve_scale(scale: float) -> Tuple[float, float]:
    if isinstance(scale, torch.Tensor):
        value = float(scale.detach().cpu())
    else:
        value = float(scale)
    if abs(value) < 1e-12:
        value = 1.0
    return value, abs(value)


def _cuda_bf16_supported(device: torch.device) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_bf16_supported())
    except Exception:
        return False


def _resolve_kernel_dtype(
    device: torch.device,
    inputs: torch.Tensor,
    grads: torch.Tensor,
    weights: torch.Tensor,
) -> torch.dtype:
    mode = str(os.environ.get("SCORING_OLDFAST_CONV_KERNEL_DTYPE", "float32")).strip().lower()
    if mode in _SUPPORTED_KERNEL_DTYPES:
        return _SUPPORTED_KERNEL_DTYPES[mode]
    if mode == "input":
        return inputs.dtype
    if mode == "auto":
        if device.type == "cuda":
            if _cuda_bf16_supported(device):
                return torch.bfloat16
            return torch.float16
        return torch.float32
    return torch.float32


def _dense_group_block_size(batch_size: int, out_channels: int, in_channels: int) -> int:
    max_block = _env_int(
        "SCORING_FRONTIER_AGGREGATE_CONV_GROUP_BLOCK",
        _env_int("SCORING_OLDFAST_CONV_GROUP_BLOCK", 256),
    )
    budget_mb = _env_int(
        "SCORING_FRONTIER_AGGREGATE_CONV_GROUP_BUDGET_MB",
        _env_int("SCORING_OLDFAST_CONV_GROUP_BUDGET_MB", 384),
    )
    bytes_per_sample = max(1, out_channels * in_channels * torch.float32.itemsize * 2)
    budget_bytes = max(1, budget_mb) * 1024 * 1024
    budget_block = max(1, budget_bytes // bytes_per_sample)
    dense_max_k = _env_int("MASKSTATS_CONV_GROUP_DENSE_MAX_K", 256)
    return max(1, min(batch_size, max_block, dense_max_k, budget_block))


def _frontier_aggregate_conv_impl() -> str:
    raw = str(
        os.environ.get(
            "SCORING_FRONTIER_AGGREGATE_CONV_IMPL",
            os.environ.get("SCORING_OLDFAST_CONV_IMPL", "preadd"),
        )
    ).strip().lower()
    if raw in {"", "default", "auto"}:
        return "preadd"
    if raw in {"preadd", "fused", "reference"}:
        return raw
    return "preadd"


def _oldfast_conv_impl() -> str:
    return _frontier_aggregate_conv_impl()


def _frontier_aggregate_conv_block_size(
    *,
    batch_size: int,
    out_channels: int,
    in_per_group: int,
    output_h: int,
    output_w: int,
) -> int:
    target_mb = _env_int(
        "SCORING_FRONTIER_AGGREGATE_CONV_TARGET_MB",
        _env_int("SCORING_OLDFAST_CONV_TARGET_MB", 768),
    )
    max_samples = _env_int(
        "SCORING_FRONTIER_AGGREGATE_CONV_MAX_SAMPLES",
        _env_int("SCORING_OLDFAST_CONV_MAX_SAMPLES", 16),
    )
    bytes_per_sample = max(
        1,
        out_channels * in_per_group * output_h * output_w * torch.float32.itemsize,
    )
    budget_bytes = max(1, target_mb) * 1024 * 1024
    block_size = max(1, budget_bytes // bytes_per_sample)
    return max(1, min(batch_size, max_samples, block_size))


def _oldfast_conv_block_size(
    *,
    batch_size: int,
    out_channels: int,
    in_per_group: int,
    output_h: int,
    output_w: int,
) -> int:
    return _frontier_aggregate_conv_block_size(
        batch_size=batch_size,
        out_channels=out_channels,
        in_per_group=in_per_group,
        output_h=output_h,
        output_w=output_w,
    )


def _build_frontier_aggregate_result_tuple(
    *,
    kernel_scores_total: torch.Tensor,
    activating_scores_total: torch.Tensor,
    inhibiting_scores_total: torch.Tensor,
    signed_acts_total: torch.Tensor,
    abs_scores_total: torch.Tensor,
    gradients_total: torch.Tensor,
    abs_gradients_total: torch.Tensor,
    weights: torch.Tensor,
    neuron_scores: Optional[torch.Tensor] = None,
    abs_neuron_scores: Optional[torch.Tensor] = None,
    neuron_scores_squared: Optional[torch.Tensor] = None,
    abs_neuron_scores_squared: Optional[torch.Tensor] = None,
    product_scores: Optional[torch.Tensor] = None,
    abs_product_scores: Optional[torch.Tensor] = None,
) -> Tuple:
    grad_non_zeroes = torch.tensor(
        int(torch.count_nonzero(gradients_total).item()),
        device=kernel_scores_total.device,
        dtype=torch.int64,
    )
    dummy = torch.zeros(1, device=kernel_scores_total.device, dtype=torch.float32)
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
        None,
        None,
        dummy,
        dummy,
        dummy,
        kernel_scores_total,
        activating_scores_total,
        inhibiting_scores_total,
        signed_acts_total,
        abs_scores_total,
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
        [],
    )


def _load_fused_backend():
    global _FUSED_BACKEND, _FUSED_BACKEND_ERROR
    if _FUSED_BACKEND is not None:
        return _FUSED_BACKEND
    if _FUSED_BACKEND_ERROR is not None:
        raise _FUSED_BACKEND_ERROR
    try:
        from engines.thermobayes.models.fused_kernels.fused_masked_conv_kernels import (
            fused_masked_conv2d_available,
            fused_masked_conv_stats,
        )
    except Exception as exc:  # pragma: no cover - optional Triton dependency
        _FUSED_BACKEND_ERROR = exc
        raise
    _FUSED_BACKEND = (fused_masked_conv2d_available, fused_masked_conv_stats)
    return _FUSED_BACKEND


def fused_frontier_aggregate_channel_virtual_available() -> bool:
    if not _env_flag(
        "SCORING_FRONTIER_AGGREGATE_USE_FUSED_CONV",
        _env_flag("SCORING_OLDFAST_USE_FUSED_CONV", True),
    ):
        return False
    try:
        fused_masked_conv2d_available, _ = _load_fused_backend()
    except Exception:
        return False
    try:
        return bool(fused_masked_conv2d_available())
    except Exception:
        return False


def fused_oldfast_channel_virtual_available() -> bool:
    return fused_frontier_aggregate_channel_virtual_available()


def _materialize_group_scores(group_scores: object) -> torch.Tensor:
    if isinstance(group_scores, torch.Tensor):
        return group_scores
    materialize = getattr(group_scores, "materialize_dense", None)
    if callable(materialize):
        dense = materialize()
        if dense.dim() == 5:
            return dense[..., 0, 0]
        return dense
    raise TypeError(f"Unsupported grouped stats payload type: {type(group_scores)!r}")


def _run_grouped_conv_stats_block(
    *,
    inputs: torch.Tensor,
    grads: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
    stride,
    padding,
    dilation,
    groups: int,
) -> torch.Tensor:
    _, fused_masked_conv_stats = _load_fused_backend()
    block = inputs.size(0)
    fid = torch.arange(block, device=inputs.device, dtype=torch.int64)
    _, _, grad_sum_k, _, _ = fused_masked_conv_stats(
        grad_output=grads,
        input=inputs,
        weight=weights,
        mask=mask,
        fid=fid,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        compute_weight_stats=False,
    )
    return _materialize_group_scores(grad_sum_k).to(device=inputs.device, dtype=torch.float32)


def conv_compute_frontier_aggregate_channel_virtual_fused(
    *,
    inputs: torch.Tensor,
    grads: torch.Tensor,
    weights: torch.Tensor,
    scale: float,
    stride,
    padding,
    dilation,
    groups: int,
    device: torch.device,
    abs_aggregation_mode: str = "exact_abs",
) -> Optional[Tuple]:
    if not fused_frontier_aggregate_channel_virtual_available():
        return None
    if device.type != "cuda" or inputs.device.type != "cuda" or grads.device.type != "cuda":
        return None

    frontier_abs = is_frontier_aggregate_abs_mode(abs_aggregation_mode)
    scale_value, abs_scale_value = _resolve_scale(scale)
    kernel_dtype = _resolve_kernel_dtype(device, inputs, grads, weights)

    out_channels, in_per_group = weights.shape[:2]
    block_size = _dense_group_block_size(int(inputs.size(0)), out_channels, in_per_group)
    mask = torch.ones((out_channels, in_per_group), device=device, dtype=kernel_dtype)

    weight_kernel = weights.detach().to(device=device, dtype=kernel_dtype).contiguous()
    inputs_kernel = inputs.detach().to(device=device, dtype=kernel_dtype).contiguous()
    grads_kernel = grads.detach().to(device=device, dtype=kernel_dtype).contiguous()

    kernel_scores_total = torch.zeros((out_channels, in_per_group), dtype=torch.float32, device=device)
    abs_scores_total = torch.zeros_like(kernel_scores_total)
    activating_scores_total = torch.zeros_like(kernel_scores_total)
    inhibiting_scores_total = torch.zeros_like(kernel_scores_total)
    signed_acts_total = torch.zeros_like(kernel_scores_total)

    for start in range(0, inputs.size(0), block_size):
        end = min(start + block_size, inputs.size(0))
        block_scores = _run_grouped_conv_stats_block(
            inputs=inputs_kernel[start:end],
            grads=grads_kernel[start:end],
            weights=weight_kernel,
            mask=mask,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        scaled_block = block_scores / scale_value
        kernel_scores_total += scaled_block.sum(dim=0)
        if frontier_abs:
            activating_scores_total += torch.relu(scaled_block.sum(dim=0))
            inhibiting_scores_total += -torch.relu(-scaled_block.sum(dim=0))
        else:
            activating_scores_total += torch.relu(scaled_block).sum(dim=0)
            inhibiting_scores_total += -torch.relu(-scaled_block).sum(dim=0)
            abs_scores_total += scaled_block.abs().sum(dim=0)
            abs_grad_block = grads_kernel[start:end].abs()
            signed_block = _run_grouped_conv_stats_block(
                inputs=inputs_kernel[start:end],
                grads=abs_grad_block,
                weights=weight_kernel,
                mask=mask,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
            signed_acts_total += signed_block.sum(dim=0) / scale_value

    if frontier_abs:
        abs_scores_total = kernel_scores_total.abs()
        activating_scores_total = torch.relu(kernel_scores_total)
        inhibiting_scores_total = -torch.relu(-kernel_scores_total)
        signed_acts_total = abs_scores_total.clone()

    gradients_total = (
        (grads.detach().to(device=device, dtype=torch.float32).sum(dim=(-1, -2)) / scale_value).sum(dim=0)
    ).unsqueeze(1).expand(-1, in_per_group).clone()
    abs_gradients_total = (
        (grads.detach().to(device=device, dtype=torch.float32).abs().sum(dim=(-1, -2)) / abs_scale_value).sum(dim=0)
    ).unsqueeze(1).expand(-1, in_per_group).clone()
    if frontier_abs:
        abs_gradients_total = gradients_total.abs()

    return _build_frontier_aggregate_result_tuple(
        kernel_scores_total=kernel_scores_total,
        activating_scores_total=activating_scores_total,
        inhibiting_scores_total=inhibiting_scores_total,
        signed_acts_total=signed_acts_total,
        abs_scores_total=abs_scores_total,
        gradients_total=gradients_total,
        abs_gradients_total=abs_gradients_total,
        weights=weights,
    )


def conv_compute_oldfast_channel_virtual_fused(
    *,
    inputs: torch.Tensor,
    grads: torch.Tensor,
    weights: torch.Tensor,
    scale: float,
    stride,
    padding,
    dilation,
    groups: int,
    device: torch.device,
    abs_aggregation_mode: str = "exact_abs",
) -> Optional[Tuple]:
    return conv_compute_frontier_aggregate_channel_virtual_fused(
        inputs=inputs,
        grads=grads,
        weights=weights,
        scale=scale,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        device=device,
        abs_aggregation_mode=abs_aggregation_mode,
    )


def conv_compute_frontier_aggregate_channel_virtual_preadd(
    *,
    inputs_source: Union[torch.Tensor, TensorReplayProvider],
    grads: torch.Tensor,
    weights: torch.Tensor,
    scale: float,
    stride,
    padding,
    dilation,
    groups: int,
    device: torch.device,
    abs_aggregation_mode: str = "exact_abs",
    compute_neuron_metrics: bool = False,
    compute_product_metrics: bool = False,
    preadd_weight: Optional[torch.Tensor] = None,
    in_per_group: Optional[int] = None,
    out_per_group: Optional[int] = None,
) -> Tuple:
    if preadd_weight is None or in_per_group is None or out_per_group is None:
        preadd_weight, in_per_group, out_per_group = _build_conv_channel_virtual_weight(
            weights,
            groups=groups,
        )
    preadd_weight = preadd_weight.to(device=device, dtype=torch.float32)

    batch_size = _replay_batch_size(inputs_source, grads.size(0))
    output_h = int(grads.size(-2))
    output_w = int(grads.size(-1))
    block_size = _frontier_aggregate_conv_block_size(
        batch_size=batch_size,
        out_channels=int(weights.size(0)),
        in_per_group=int(in_per_group),
        output_h=output_h,
        output_w=output_w,
    )

    kernel_scores_total = torch.zeros(
        (weights.size(0), weights.size(1)),
        dtype=torch.float32,
        device=device,
    )
    activating_scores_total = torch.zeros_like(kernel_scores_total)
    inhibiting_scores_total = torch.zeros_like(kernel_scores_total)
    signed_acts_total = torch.zeros_like(kernel_scores_total)
    abs_scores_total = torch.zeros_like(kernel_scores_total)
    gradients_total = torch.zeros_like(kernel_scores_total)
    abs_gradients_total = torch.zeros_like(kernel_scores_total)
    if compute_neuron_metrics:
        neuron_scores_total = torch.zeros(weights.size(0), dtype=torch.float32, device=device)
        abs_neuron_scores_total = torch.zeros_like(neuron_scores_total)
        neuron_scores_sq_total = torch.zeros_like(neuron_scores_total)
        abs_neuron_scores_sq_total = torch.zeros_like(neuron_scores_total)
        product_scores_total = (
            torch.zeros_like(kernel_scores_total)
            if compute_product_metrics
            else None
        )
        abs_product_scores_total = (
            torch.zeros_like(kernel_scores_total)
            if compute_product_metrics
            else None
        )
    else:
        neuron_scores_total = None
        abs_neuron_scores_total = None
        neuron_scores_sq_total = None
        abs_neuron_scores_sq_total = None
        product_scores_total = None
        abs_product_scores_total = None

    frontier_abs = is_frontier_aggregate_abs_mode(abs_aggregation_mode)
    need_per_sample_scores = not frontier_abs

    for start in range(0, batch_size, block_size):
        end = min(start + block_size, batch_size)
        inputs_chunk = _fetch_replay_slice(
            inputs_source,
            start,
            end,
            device=device,
            dtype=torch.float32,
        )
        grads_chunk = grads[start:end].to(device=device, dtype=torch.float32)
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
        outputs_chunk = preadd_chunk.sum(dim=2) if compute_neuron_metrics else None
        chunk_stats = _conv_channel_virtual_reduce_exact_chunk(
            preadd_chunk,
            grads_chunk,
            scale=scale,
            outputs_chunk=outputs_chunk,
            compute_neuron_metrics=compute_neuron_metrics,
            compute_product_metrics=compute_product_metrics,
            abs_aggregation_mode=abs_aggregation_mode,
            need_per_sample_scores=need_per_sample_scores,
        )
        kernel_scores_total += chunk_stats["kernel_scores_sum"]
        activating_scores_total += chunk_stats["activating_scores_sum"]
        inhibiting_scores_total += chunk_stats["inhibiting_scores_sum"]
        signed_acts_total += chunk_stats["scores_signed_acts_sum"]
        abs_scores_total += chunk_stats["abs_scores_sum"]
        gradients_total += chunk_stats["gradients_sum"]
        abs_gradients_total += chunk_stats["abs_gradients_sum"]
        if compute_neuron_metrics:
            neuron_scores_total += chunk_stats["neuron_scores_sum"]
            abs_neuron_scores_total += chunk_stats["abs_neuron_scores_sum"]
            neuron_scores_sq_total += chunk_stats["neuron_scores_sq_sum"]
            abs_neuron_scores_sq_total += chunk_stats["abs_neuron_scores_sq_sum"]
            if compute_product_metrics:
                product_scores_total += chunk_stats["product_scores_sum"]
                abs_product_scores_total += chunk_stats["abs_product_scores_sum"]

    if frontier_abs:
        abs_scores_total = kernel_scores_total.abs()
        activating_scores_total = torch.relu(kernel_scores_total)
        inhibiting_scores_total = -torch.relu(-kernel_scores_total)
        signed_acts_total = abs_scores_total.clone()
        abs_gradients_total = gradients_total.abs()
        if compute_neuron_metrics:
            abs_neuron_scores_total = neuron_scores_total.abs()
            abs_neuron_scores_sq_total = abs_neuron_scores_total.square()
            if compute_product_metrics:
                abs_product_scores_total = product_scores_total.abs()

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

    return _build_frontier_aggregate_result_tuple(
        kernel_scores_total=kernel_scores_total,
        activating_scores_total=activating_scores_total,
        inhibiting_scores_total=inhibiting_scores_total,
        signed_acts_total=signed_acts_total,
        abs_scores_total=abs_scores_total,
        gradients_total=gradients_total,
        abs_gradients_total=abs_gradients_total,
        weights=weights,
        neuron_scores=neuron_scores,
        abs_neuron_scores=abs_neuron_scores,
        neuron_scores_squared=neuron_scores_squared,
        abs_neuron_scores_squared=abs_neuron_scores_squared,
        product_scores=product_scores,
        abs_product_scores=abs_product_scores,
    )


def conv_compute_oldfast_channel_virtual_preadd(
    *,
    inputs_source: Union[torch.Tensor, TensorReplayProvider],
    grads: torch.Tensor,
    weights: torch.Tensor,
    scale: float,
    stride,
    padding,
    dilation,
    groups: int,
    device: torch.device,
    abs_aggregation_mode: str = "exact_abs",
    compute_neuron_metrics: bool = False,
    compute_product_metrics: bool = False,
    preadd_weight: Optional[torch.Tensor] = None,
    in_per_group: Optional[int] = None,
    out_per_group: Optional[int] = None,
) -> Tuple:
    return conv_compute_frontier_aggregate_channel_virtual_preadd(
        inputs_source=inputs_source,
        grads=grads,
        weights=weights,
        scale=scale,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        device=device,
        abs_aggregation_mode=abs_aggregation_mode,
        compute_neuron_metrics=compute_neuron_metrics,
        compute_product_metrics=compute_product_metrics,
        preadd_weight=preadd_weight,
        in_per_group=in_per_group,
        out_per_group=out_per_group,
    )


__all__ = [
    "conv_compute_frontier_aggregate_channel_virtual_preadd",
    "conv_compute_frontier_aggregate_channel_virtual_fused",
    "fused_frontier_aggregate_channel_virtual_available",
    "_frontier_aggregate_conv_impl",
    "_frontier_aggregate_conv_block_size",
    "conv_compute_oldfast_channel_virtual_preadd",
    "conv_compute_oldfast_channel_virtual_fused",
    "fused_oldfast_channel_virtual_available",
    "_oldfast_conv_impl",
]
