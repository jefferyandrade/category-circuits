"""Adaptive average pooling with fixed reductions and deterministic autograd."""

from __future__ import annotations

import operator
from types import MethodType
from typing import Any


def _output_shape(output_size: Any, height: int, width: int) -> tuple[int, int]:
    if isinstance(output_size, (tuple, list)):
        if len(output_size) != 2:
            raise ValueError("adaptive average pooling requires two output dimensions")
        sizes = output_size
    else:
        size = operator.index(output_size)
        sizes = (size, size)
    out_height = height if sizes[0] is None else operator.index(sizes[0])
    out_width = width if sizes[1] is None else operator.index(sizes[1])
    if out_height < 0 or out_width < 0:
        raise ValueError("adaptive average pooling output dimensions must be nonnegative")
    return out_height, out_width


def _pool_last_axis(inputs: Any, input_size: int, output_size: int) -> Any:
    import torch

    rows = torch.arange(output_size, device=inputs.device, dtype=torch.int64)
    columns = torch.arange(input_size, device=inputs.device, dtype=torch.int64)
    starts = (rows * input_size) // output_size
    ends = ((rows + 1) * input_size + output_size - 1) // output_size
    selected = (columns[None, :] >= starts[:, None]) & (columns[None, :] < ends[:, None])
    members = torch.where(selected, inputs.unsqueeze(-2), 0.0)
    return members.sum(dim=-1) / (ends - starts).to(dtype=inputs.dtype)


def deterministic_adaptive_avg_pool2d(inputs: Any, output_size: Any) -> Any:
    """Apply PyTorch's floor/ceil adaptive bins without atomic CUDA backward.

    Equal-size axes are identities. Nonoverlapping uniform bins use one mean;
    other bins use separable masked sums over fixed axes. Their backward
    sums the broadcast axis without scattering, atomics or TF32. No random
    state or mutable cache is used.
    """

    import torch

    if inputs.ndim not in (3, 4):
        raise ValueError("adaptive average pooling requires CHW or NCHW inputs")
    height, width = inputs.shape[-2:]
    if height <= 0 or width <= 0:
        raise ValueError("adaptive average pooling requires nonempty spatial dimensions")
    out_height, out_width = _output_shape(output_size, height, width)
    if (height, width) == (out_height, out_width):
        return inputs.clone()
    if out_height == 0 or out_width == 0:
        return inputs.sum(dim=(-2, -1), keepdim=True).expand(*inputs.shape[:-2], out_height, out_width)
    values = inputs.to(dtype=torch.float32) if inputs.dtype in (torch.float16, torch.bfloat16) else inputs
    if height % out_height == 0 and width % out_width == 0:
        result = values.reshape(
            *values.shape[:-2], out_height, height // out_height, out_width, width // out_width
        ).mean(dim=(-3, -1))
    else:
        result = values
        if width != out_width:
            result = _pool_last_axis(result, width, out_width)
        if height != out_height:
            result = _pool_last_axis(result.transpose(-2, -1), height, out_height).transpose(-2, -1)
    return result.to(dtype=inputs.dtype)


def _adaptive_avg_pool_forward(module: Any, inputs: Any) -> Any:
    return deterministic_adaptive_avg_pool2d(inputs, module.output_size)


def install_deterministic_adaptive_avg_pool2d(model: Any) -> int:
    """Replace native pooling execution while retaining module/state identity."""

    import torch

    changed = 0
    for module in model.modules():
        if type(module) is torch.nn.AdaptiveAvgPool2d:
            if getattr(module.forward, "__func__", None) is not _adaptive_avg_pool_forward:
                module.forward = MethodType(_adaptive_avg_pool_forward, module)
                changed += 1
    return changed
