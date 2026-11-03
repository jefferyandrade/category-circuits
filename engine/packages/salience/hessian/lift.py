"""Helpers to lift activation-space Hessian information into weight space."""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from packages.salience.core.frontier_wrappers import HookedMultiheadAttention, HookedSwiGLU


def lift_linear_hessian(
    hvp_fn: Callable[[torch.Tensor], torch.Tensor],
    activations: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    """Lift an activation-space HVP into the weight space of a linear layer.

    Args:
        hvp_fn: Callable that consumes a delta in activation space (same shape as
            the linear pre-activation ``z``) and returns ``H_z @ delta``.
        activations: Input activations ``A`` with shape ``(batch, in_features)``.
        direction: Weight perturbation ``V`` with shape ``(out_features, in_features)``.

    Returns:
        Tensor representing ``H_W @ V`` with the same shape as ``direction``.
    """

    if activations.ndim != 2:
        raise ValueError("Expected activations to have shape (batch, in_features)")
    if direction.ndim != 2:
        raise ValueError("Expected direction to have shape (out_features, in_features)")

    delta_z = activations @ direction.t()
    hvp_z = hvp_fn(delta_z)

    if hvp_z.shape != delta_z.shape:
        raise ValueError("hvp_fn returned tensor with unexpected shape")

    lifted = torch.einsum("bo,bi->oi", hvp_z, activations)
    return lifted


def _solve_linear_direction(
    activations: torch.Tensor, delta: torch.Tensor
) -> torch.Tensor:
    """Solve for the weight perturbation that induces ``delta``."""

    if activations.ndim != 2:
        raise ValueError("activations must have shape (batch, in_features)")
    if delta.ndim != 2:
        raise ValueError("delta must have shape (batch, out_features)")

    A = activations.to(dtype=torch.float32)
    B = delta.to(dtype=torch.float32)
    solution = torch.linalg.lstsq(A, B).solution  # (in_features, out_features)
    return solution.transpose(0, 1).contiguous()


def lift_linear_block(
    activations: torch.Tensor,
    deltas: Sequence[torch.Tensor],
    responses: Sequence[torch.Tensor],
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Lift a collection of activation-space probes into weight space."""

    if len(deltas) != len(responses):
        raise ValueError("deltas and responses must have the same length")
    A = activations.to(dtype=torch.float32)
    weight_vectors: List[torch.Tensor] = []
    weight_responses: List[torch.Tensor] = []
    for delta, resp in zip(deltas, responses):
        delta_2d = delta.to(dtype=torch.float32).view(delta.size(0), -1)
        resp_2d = resp.to(dtype=torch.float32).view(resp.size(0), -1)
        weight_vectors.append(_solve_linear_direction(A, delta_2d))
        weight_responses.append(_solve_linear_direction(A, resp_2d))
    return weight_vectors, weight_responses


def lift_convolution_hessian(
    hvp_fn: Callable[[torch.Tensor], torch.Tensor],
    activations: torch.Tensor,
    direction: torch.Tensor,
    kernel_shape: Tuple[int, ...],
) -> torch.Tensor:
    """Lift convolution activation HVPs into weight space via grad.conv2d_weight.

    Args:
        hvp_fn: Callable that maps a delta output tensor to ``H_z @ delta`` with the
            same shape as the convolution output.
        activations: Input activations tensor ``X`` with shape ``(batch, in_channels, H, W)``.
        direction: Weight perturbation ``V`` with the same shape as the convolution
            weights ``(out_channels, in_channels/groups, kH, kW)``.
        kernel_shape: Tuple containing ``(stride, padding, dilation, groups)``.

    Returns:
        Tensor representing ``H_W @ V`` with the same shape as ``direction``.
    """

    stride, padding, dilation, groups = kernel_shape
    delta_z = torch.nn.functional.conv2d(
        activations,
        direction,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    hvp_z = hvp_fn(delta_z)
    lifted = torch.nn.grad.conv2d_weight(
        activations,
        direction.shape,
        hvp_z,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    return lifted


def _prepare_conv_design_matrix(
    activations: torch.Tensor,
    *,
    kernel_size: Tuple[int, int],
    dilation: Tuple[int, int],
    padding: Tuple[int, int],
    stride: Tuple[int, int],
) -> Tuple[torch.Tensor, int]:
    unfolded = F.unfold(
        activations,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )  # (B, in_dim, L)
    design = unfolded.permute(0, 2, 1).reshape(-1, unfolded.size(1))  # (B*L, in_dim)
    locations = unfolded.size(0) * unfolded.size(2)
    return design, locations


def lift_conv_block(
    activations: torch.Tensor,
    deltas: Sequence[torch.Tensor],
    responses: Sequence[torch.Tensor],
    *,
    weight_shape: Tuple[int, int, int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    groups: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Lift convolution activation probes into weight space (groups=1 only)."""

    if groups != 1:
        raise NotImplementedError("lift_conv_block currently supports groups=1 only.")
    if len(deltas) != len(responses):
        raise ValueError("deltas and responses must have the same length")

    out_channels, in_channels, kH, kW = weight_shape
    design, _ = _prepare_conv_design_matrix(
        activations.to(dtype=torch.float32),
        kernel_size=(kH, kW),
        dilation=dilation,
        padding=padding,
        stride=stride,
    )  # (B*L, in_channels*kH*kW)

    weight_vectors: List[torch.Tensor] = []
    weight_responses: List[torch.Tensor] = []

    for delta, resp in zip(deltas, responses):
        delta_flat = delta.to(dtype=torch.float32).permute(0, 2, 3, 1).reshape(-1, out_channels)
        resp_flat = resp.to(dtype=torch.float32).permute(0, 2, 3, 1).reshape(-1, out_channels)

        weights_solution = torch.linalg.lstsq(design, delta_flat).solution  # (in_dim, out_channels)
        response_solution = torch.linalg.lstsq(design, resp_flat).solution

        weight_vectors.append(
            weights_solution.transpose(0, 1).contiguous().view(out_channels, in_channels, kH, kW)
        )
        weight_responses.append(
            response_solution.transpose(0, 1).contiguous().view(out_channels, in_channels, kH, kW)
        )

    return weight_vectors, weight_responses


def lift_attention_block(
    layer: HookedMultiheadAttention,
    deltas: Sequence[torch.Tensor],
    responses: Sequence[torch.Tensor],
) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, List[torch.Tensor]]]:
    caches = layer.attn_buffers
    if caches is None or not deltas:
        return {}, {}

    attn_weights = caches.attn_weights
    context = caches.context
    dtype = torch.float32

    vectors: Dict[str, List[torch.Tensor]] = {}
    responses_dict: Dict[str, List[torch.Tensor]] = {}

    context_flat = context.to(dtype=dtype).reshape(-1, context.size(-1))
    out_deltas: List[torch.Tensor] = []
    out_responses: List[torch.Tensor] = []

    if attn_weights is None or attn_weights.numel() == 0:
        for delta, response in zip(deltas, responses):
            out_deltas.append(delta.to(dtype=dtype).reshape(-1, delta.size(-1)))
            out_responses.append(response.to(dtype=dtype).reshape(-1, response.size(-1)))
        vectors["W_o"] = [
            torch.einsum("bo,bi->oi", delta_flat, context_flat) for delta_flat in out_deltas
        ]
        responses_dict["W_o"] = [
            torch.einsum("bo,bi->oi", resp_flat, context_flat) for resp_flat in out_responses
        ]
        return vectors, responses_dict

    batch_size, tgt_len, embed_dim = context.shape
    num_heads = layer.num_heads
    head_dim = layer.head_dim
    weights = attn_weights.to(dtype=dtype).view(batch_size, num_heads, tgt_len, -1)

    query_inputs_flat = caches.query_input.to(dtype=dtype).reshape(-1, caches.query_input.size(-1))
    key_inputs_flat = caches.key_input.to(dtype=dtype).reshape(-1, caches.key_input.size(-1))
    value_inputs_flat = caches.value_input.to(dtype=dtype).reshape(-1, caches.value_input.size(-1))

    query_proj = caches.query_proj.to(dtype=dtype)
    key_proj = caches.key_proj.to(dtype=dtype)
    value_proj = caches.value_proj.to(dtype=dtype)

    query_deltas: List[torch.Tensor] = []
    key_deltas: List[torch.Tensor] = []
    value_deltas: List[torch.Tensor] = []
    query_responses: List[torch.Tensor] = []
    key_responses: List[torch.Tensor] = []
    value_responses: List[torch.Tensor] = []

    def _reshape_heads(tensor: torch.Tensor) -> torch.Tensor:
        return (
            tensor.view(batch_size, tgt_len, num_heads, head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    def _merge_heads(tensor: torch.Tensor) -> torch.Tensor:
        return (
            tensor.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, tgt_len, embed_dim)
        )

    def _to_context(tensor: torch.Tensor) -> torch.Tensor:
        if hasattr(layer.original, "out_proj"):
            weight = layer.original.out_proj.weight.to(dtype=tensor.dtype)
            return torch.matmul(tensor, weight)
        return tensor

    value_heads = _reshape_heads(value_proj)
    query_heads = _reshape_heads(query_proj)
    key_heads = _reshape_heads(key_proj)

    for delta, response in zip(deltas, responses):
        delta_tensor = delta.to(dtype=dtype)
        response_tensor = response.to(dtype=dtype)

        out_deltas.append(delta_tensor.reshape(-1, delta_tensor.size(-1)))
        out_responses.append(response_tensor.reshape(-1, response_tensor.size(-1)))

        delta_context = _reshape_heads(_to_context(delta_tensor))
        response_context = _reshape_heads(_to_context(response_tensor))

        delta_weights = torch.matmul(delta_context, value_heads.transpose(-2, -1))
        response_weights = torch.matmul(response_context, value_heads.transpose(-2, -1))

        dot_delta = (delta_weights * weights).sum(dim=-1, keepdim=True)
        dot_response = (response_weights * weights).sum(dim=-1, keepdim=True)

        delta_scores = (delta_weights - dot_delta) * weights
        response_scores = (response_weights - dot_response) * weights

        delta_query_heads = torch.matmul(delta_scores, key_heads) / (head_dim ** 0.5)
        delta_key_heads = torch.matmul(delta_scores.transpose(-2, -1), query_heads) / (
            head_dim ** 0.5
        )
        delta_value_heads = torch.matmul(weights.transpose(-2, -1), delta_context)

        response_query_heads = torch.matmul(response_scores, key_heads) / (head_dim ** 0.5)
        response_key_heads = torch.matmul(
            response_scores.transpose(-2, -1), query_heads
        ) / (head_dim ** 0.5)
        response_value_heads = torch.matmul(weights.transpose(-2, -1), response_context)

        query_deltas.append(_merge_heads(delta_query_heads).reshape(-1, embed_dim))
        key_deltas.append(_merge_heads(delta_key_heads).reshape(-1, embed_dim))
        value_deltas.append(_merge_heads(delta_value_heads).reshape(-1, embed_dim))

        query_responses.append(_merge_heads(response_query_heads).reshape(-1, embed_dim))
        key_responses.append(_merge_heads(response_key_heads).reshape(-1, embed_dim))
        value_responses.append(_merge_heads(response_value_heads).reshape(-1, embed_dim))

    vectors["W_o"] = [
        torch.einsum("bo,bi->oi", delta_flat, context_flat) for delta_flat in out_deltas
    ]
    responses_dict["W_o"] = [
        torch.einsum("bo,bi->oi", resp_flat, context_flat) for resp_flat in out_responses
    ]
    vectors["W_q"] = [
        torch.einsum("bo,bi->oi", delta_flat, query_inputs_flat) for delta_flat in query_deltas
    ]
    responses_dict["W_q"] = [
        torch.einsum("bo,bi->oi", resp_flat, query_inputs_flat) for resp_flat in query_responses
    ]
    vectors["W_k"] = [
        torch.einsum("bo,bi->oi", delta_flat, key_inputs_flat) for delta_flat in key_deltas
    ]
    responses_dict["W_k"] = [
        torch.einsum("bo,bi->oi", resp_flat, key_inputs_flat) for resp_flat in key_responses
    ]
    vectors["W_v"] = [
        torch.einsum("bo,bi->oi", delta_flat, value_inputs_flat) for delta_flat in value_deltas
    ]
    responses_dict["W_v"] = [
        torch.einsum("bo,bi->oi", resp_flat, value_inputs_flat) for resp_flat in value_responses
    ]
    return vectors, responses_dict


def lift_swiglu_block(
    layer: HookedSwiGLU,
    deltas: Sequence[torch.Tensor],
    responses: Sequence[torch.Tensor],
) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, List[torch.Tensor]]]:
    caches = layer.caches
    if caches is None or not deltas:
        return {}, {}

    dtype = torch.float32
    inputs_flat = caches.inputs.to(dtype=dtype).reshape(-1, caches.inputs.size(-1))
    hidden_flat = caches.hidden.to(dtype=dtype).reshape(-1, caches.hidden.size(-1))
    activated_gate = caches.activated_gate.to(dtype=dtype)
    up_pre = caches.up_pre.to(dtype=dtype)
    gate_pre = caches.gate_pre.to(dtype=dtype)
    activation_deriv = layer._activation_derivative(gate_pre).to(dtype=dtype)

    vectors: Dict[str, List[torch.Tensor]] = {}
    responses_dict: Dict[str, List[torch.Tensor]] = {}

    out_deltas: List[torch.Tensor] = []
    out_responses: List[torch.Tensor] = []
    gate_deltas: List[torch.Tensor] = []
    gate_responses: List[torch.Tensor] = []
    up_deltas: List[torch.Tensor] = []
    up_responses: List[torch.Tensor] = []

    weight = None
    if hasattr(layer.original, "down_proj"):
        weight = layer.original.down_proj.weight.to(dtype=dtype)
    elif hasattr(layer.original, "wo"):
        weight = layer.original.wo.weight.to(dtype=dtype)

    for delta, response in zip(deltas, responses):
        delta_tensor = delta.to(dtype=dtype)
        response_tensor = response.to(dtype=dtype)

        out_deltas.append(delta_tensor.reshape(-1, delta_tensor.size(-1)))
        out_responses.append(response_tensor.reshape(-1, response_tensor.size(-1)))

        delta_hidden = delta_tensor if weight is None else torch.matmul(delta_tensor, weight)
        response_hidden = response_tensor if weight is None else torch.matmul(response_tensor, weight)

        delta_up = delta_hidden * activated_gate
        response_up = response_hidden * activated_gate

        delta_gate = delta_hidden * up_pre * activation_deriv
        response_gate = response_hidden * up_pre * activation_deriv

        gate_deltas.append(delta_gate.reshape(-1, delta_gate.size(-1)))
        gate_responses.append(response_gate.reshape(-1, response_gate.size(-1)))
        up_deltas.append(delta_up.reshape(-1, delta_up.size(-1)))
        up_responses.append(response_up.reshape(-1, response_up.size(-1)))

    vectors["W_down"] = [
        torch.einsum("bo,bi->oi", delta_flat, hidden_flat) for delta_flat in out_deltas
    ]
    responses_dict["W_down"] = [
        torch.einsum("bo,bi->oi", resp_flat, hidden_flat) for resp_flat in out_responses
    ]
    vectors["W_gate"] = [
        torch.einsum("bo,bi->oi", delta_flat, inputs_flat) for delta_flat in gate_deltas
    ]
    responses_dict["W_gate"] = [
        torch.einsum("bo,bi->oi", resp_flat, inputs_flat) for resp_flat in gate_responses
    ]
    vectors["W_up"] = [
        torch.einsum("bo,bi->oi", delta_flat, inputs_flat) for delta_flat in up_deltas
    ]
    responses_dict["W_up"] = [
        torch.einsum("bo,bi->oi", resp_flat, inputs_flat) for resp_flat in up_responses
    ]
    return vectors, responses_dict
