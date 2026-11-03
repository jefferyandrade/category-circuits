"""Closed-form Integrated Gradients rules for attention-style blocks."""
from __future__ import annotations

from typing import Dict, Optional

import torch


def _reshape_to_heads(
    tensor: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    return (
        tensor.view(batch_size, seq_len, num_heads, head_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def _merge_from_heads(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.permute(0, 2, 1, 3).contiguous().view(tensor.size(0), tensor.size(2), -1)


def compute_attention_branch_multipliers(
    *,
    grad_output: torch.Tensor,
    query_proj: torch.Tensor,
    key_proj: torch.Tensor,
    value_proj: torch.Tensor,
    attn_weights: Optional[torch.Tensor],
    out_proj_weight: Optional[torch.Tensor],
    num_heads: int,
) -> Dict[str, torch.Tensor]:
    """
    Propagate Integrated Gradients multipliers through a multi-head attention block.

    Args:
        grad_output: Tensor of shape ``(B, T, D)`` containing the multiplier arriving at the attention output.
        query_proj / key_proj / value_proj: Linear projections with the same orientation as
            ``grad_output`` (batch-first).
        attn_weights: Attention probabilities of shape ``(B, num_heads, T, S)`` (or
            ``(B, T, S)`` if averaged across heads). Required to propagate scores.
        out_proj_weight: Optional output projection weight ``W_O`` with shape ``(D, D)``.
        num_heads: Number of attention heads.

    Returns:
        Dictionary with keys ``{"query", "key", "value"}`` mapping to tensors with the same
        orientation as the provided projections (typically batch-first).
    """

    if attn_weights is None or attn_weights.numel() == 0:
        return {"query": torch.zeros_like(query_proj), "key": torch.zeros_like(key_proj), "value": torch.zeros_like(value_proj)}

    grad = grad_output
    batch_size, tgt_len, embed_dim = grad.shape
    head_dim = embed_dim // num_heads

    grad_context = grad
    if out_proj_weight is not None:
        grad_context = torch.matmul(grad, out_proj_weight)

    value_heads = _reshape_to_heads(
        value_proj,
        batch_size=batch_size,
        seq_len=tgt_len,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    query_heads = _reshape_to_heads(
        query_proj,
        batch_size=batch_size,
        seq_len=tgt_len,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    key_heads = _reshape_to_heads(
        key_proj,
        batch_size=batch_size,
        seq_len=tgt_len,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    grad_context_heads = _reshape_to_heads(
        grad_context,
        batch_size=batch_size,
        seq_len=tgt_len,
        num_heads=num_heads,
        head_dim=head_dim,
    )

    weights = attn_weights
    if weights.dim() == 3:
        weights = weights.unsqueeze(1)
    weights = weights.view(batch_size, num_heads, tgt_len, -1)

    grad_weights = torch.matmul(
        grad_context_heads,
        value_heads.transpose(-2, -1),
    )

    dot = (grad_weights * weights).sum(dim=-1, keepdim=True)
    grad_scores = (grad_weights - dot) * weights

    scale = float(head_dim) ** 0.5
    grad_query_heads = torch.matmul(grad_scores, key_heads) / scale
    grad_key_heads = torch.matmul(grad_scores.transpose(-2, -1), query_heads) / scale
    grad_value_heads = torch.matmul(weights.transpose(-2, -1), grad_context_heads)

    return {
        "query": _merge_from_heads(grad_query_heads),
        "key": _merge_from_heads(grad_key_heads),
        "value": _merge_from_heads(grad_value_heads),
    }


__all__ = ["compute_attention_branch_multipliers"]
