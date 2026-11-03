"""
Unified frontier wrappers for linear, convolutional, attention, and SwiGLU modules.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from math import sqrt
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_rules import compute_attention_branch_multipliers
from .frontier_conv import HookedConv2d
from .frontier_linear import HookedLinear
from .frontier_utils import ENABLE_ATTENTION_WRAPPER, ENABLE_SWIGLU_WRAPPER

logger = logging.getLogger(__name__)

__all__ = [
    "HookedLinear",
    "HookedConv2d",
    "HookedMultiheadAttention",
    "HookedSwiGLU",
    "convert_linear",
    "convert_conv2d",
    "convert_multihead_attention",
    "convert_swiglu",
    "register_frontier_wrappers",
]


def convert_linear(module: nn.Module, **config: Any) -> HookedLinear:
    """Convert an ``nn.Linear`` (or already hooked module) to ``HookedLinear``."""

    if isinstance(module, HookedLinear):
        return module
    if not isinstance(module, nn.Linear):
        raise TypeError(f"convert_linear expected nn.Linear, got {type(module)!r}")
    linear_cfg = dict(config)
    linear_cfg.setdefault("collect_integrated", True)
    linear_cfg.setdefault("collect_stats", True)
    linear_cfg.setdefault("collect_hvp", True)
    return HookedLinear(module, **linear_cfg)


def convert_conv2d(module: nn.Module, **config: Any) -> HookedConv2d:
    """Convert an ``nn.Conv2d`` (or already hooked module) to ``HookedConv2d``."""

    if isinstance(module, HookedConv2d):
        return module
    if not isinstance(module, nn.Conv2d):
        raise TypeError(f"convert_conv2d expected nn.Conv2d, got {type(module)!r}")
    conv_cfg = dict(config)
    conv_cfg.setdefault("collect_integrated", True)
    conv_cfg.setdefault("collect_stats", True)
    conv_cfg.setdefault("collect_hvp", True)
    return HookedConv2d(module, **conv_cfg)


def _is_swiglu_module(module: nn.Module) -> bool:
    if hasattr(module, "gate_proj") and hasattr(module, "up_proj"):
        return True
    if hasattr(module, "wi"):
        return True
    return False


def register_frontier_wrappers(
    model: nn.Module,
    *,
    enable_linear: bool = True,
    enable_conv: bool = True,
    enable_attention: Optional[bool] = None,
    enable_swiglu: Optional[bool] = None,
    linear_config: Optional[Dict[str, Any]] = None,
    conv_config: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """
    Recursively replace standard modules with their frontier-aware counterparts.
    """

    enable_attention = ENABLE_ATTENTION_WRAPPER if enable_attention is None else enable_attention
    enable_swiglu = ENABLE_SWIGLU_WRAPPER if enable_swiglu is None else enable_swiglu
    linear_kwargs = dict(linear_config or {})
    conv_kwargs = dict(conv_config or {})

    def _transform(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            replaced = child
            if enable_linear and isinstance(child, nn.Linear) and not isinstance(child, HookedLinear):
                replaced = convert_linear(child, **linear_kwargs)
            elif enable_conv and isinstance(child, nn.Conv2d) and not isinstance(child, HookedConv2d):
                replaced = convert_conv2d(child, **conv_kwargs)
            elif enable_attention and isinstance(child, nn.MultiheadAttention):
                replaced = convert_multihead_attention(child)
            elif enable_swiglu and _is_swiglu_module(child):
                replaced = convert_swiglu(child)

            if replaced is not child:
                parent.add_module(name, replaced)

            if isinstance(replaced, (HookedLinear, HookedConv2d, HookedMultiheadAttention, HookedSwiGLU)):
                continue
            if isinstance(replaced, nn.Module):
                _transform(replaced)

    _transform(model)
    return model


@dataclass
class AttentionBuffers:
    query_proj: torch.Tensor
    key_proj: torch.Tensor
    value_proj: torch.Tensor
    context: torch.Tensor
    output: torch.Tensor
    attn_weights: Optional[torch.Tensor]
    query_input: torch.Tensor
    key_input: torch.Tensor
    value_input: torch.Tensor


class HookedMultiheadAttention(nn.Module):
    """Wrapper around ``nn.MultiheadAttention`` that exposes salience-friendly caches."""

    def __init__(self, original: nn.MultiheadAttention, *, store_inputs: bool = True) -> None:
        super().__init__()
        self.original = original
        self.embed_dim = original.embed_dim
        self.num_heads = original.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.batch_first = bool(getattr(original, "batch_first", False))
        self.store_inputs = bool(store_inputs)

        # Runtime flags mirrored from the linear/conv frontier wrappers.
        self.calc_kernels = False
        self.compute_scores = False
        self.hook_in = False
        self.ig_capture_output = True

        self.saved_integrated_multipliers: Optional[torch.Tensor] = None
        self.branch_multipliers: Dict[str, torch.Tensor] = {}
        self._cache: Optional[AttentionBuffers] = None
        self._output_tensor: Optional[torch.Tensor] = None
        self._cached_query_cpu: Optional[torch.Tensor] = None
        self._cached_key_cpu: Optional[torch.Tensor] = None
        self._cached_value_cpu: Optional[torch.Tensor] = None
        self._staged_stats: Dict[str, Any] = {}
        self.stage_stats_enabled: bool = True
        self.stream_flush_enabled: bool = False
        self.stream_flush_replay_device: Optional[torch.device] = None
        self._stream_results_ready: bool = False
        self._stream_context = None
        self.frontier_layer_name: Optional[str] = None
        self.streaming_reductions: bool = False
        self.out_proj_bias_scores: Optional[torch.Tensor] = None
        self._value_transport_route_mask: Optional[torch.Tensor] = None
        self._qk_logit_route_mask: Optional[torch.Tensor] = None
        self._probability_route_mask: Optional[torch.Tensor] = None
        self._value_row_mask: Optional[torch.Tensor] = None
        self._out_column_mask: Optional[torch.Tensor] = None
        self._qk_row_pair_mask: Optional[torch.Tensor] = None
        self._ov_mode_mask: Optional[torch.Tensor] = None
        self._qk_mode_mask: Optional[torch.Tensor] = None
        self._head_mask: Optional[torch.Tensor] = None
        self._attention_route_mask_metadata: Dict[str, Any] = {}

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        if key is None:
            key = query
        if value is None:
            value = key

        attn_kwargs = dict(kwargs)
        need_weights_was_explicit = "need_weights" in attn_kwargs
        return_weights = bool(attn_kwargs.get("need_weights", False))
        average_requested = bool(attn_kwargs.get("average_attn_weights", True))
        attn_kwargs["need_weights"] = True
        attn_kwargs["average_attn_weights"] = False

        if self.has_attention_route_masks:
            attn_output, attn_weights = self._forward_with_attention_route_masks(
                query,
                key,
                value,
                attn_kwargs=attn_kwargs,
            )
        else:
            attn_output, attn_weights = self.original(
                query,
                key,
                value,
                **attn_kwargs,
            )
        if attn_output.requires_grad:
            attn_output = attn_output.clone()
            attn_output.retain_grad()
        self._output_tensor = attn_output

        query_in, key_in, value_in = query, key, value
        if not self.batch_first:
            query_in = query.transpose(0, 1).contiguous()
            key_in = key.transpose(0, 1).contiguous()
            value_in = value.transpose(0, 1).contiguous()

        q_proj, k_proj, v_proj = self._project_qkv(query_in, key_in, value_in)
        context = self._compute_context(attn_weights, v_proj)
        self._cache_buffers(
            query_in,
            key_in,
            value_in,
            q_proj,
            k_proj,
            v_proj,
            attn_output.detach(),
            context.detach(),
            attn_weights.detach() if attn_weights is not None else None,
        )

        returned_weights = attn_weights
        if average_requested and attn_weights is not None:
            returned_weights = attn_weights.mean(dim=1)

        if return_weights:
            return attn_output, returned_weights
        if need_weights_was_explicit:
            return attn_output, None
        return attn_output

    @property
    def has_attention_route_masks(self) -> bool:
        return any(
            mask is not None
            for mask in (
                self._value_transport_route_mask,
                self._qk_logit_route_mask,
                self._probability_route_mask,
                self._ov_mode_mask,
                self._qk_mode_mask,
                self._head_mask,
            )
        )

    def set_attention_route_masks(
        self,
        *,
        value_transport: Optional[torch.Tensor] = None,
        qk_logit: Optional[torch.Tensor] = None,
        probability: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Install exact hard route masks for native attention execution.

        Masks are over the architecture-native route axes ``head x query x key``.
        A mask with one query row is interpreted as a CLS-query-only mask: query
        0 is masked and all other query-token routes remain dense. This matches
        the default ViT pruning score object while preserving full-sequence
        execution for the rest of the model.
        """

        self._value_transport_route_mask = (
            value_transport.detach().to(dtype=torch.bool).cpu() if value_transport is not None else None
        )
        self._qk_logit_route_mask = (
            qk_logit.detach().to(dtype=torch.bool).cpu() if qk_logit is not None else None
        )
        self._probability_route_mask = (
            probability.detach().to(dtype=torch.bool).cpu() if probability is not None else None
        )
        self._attention_route_mask_metadata = dict(metadata or {})

    def set_attention_structural_masks(
        self,
        *,
        value_row: Optional[torch.Tensor] = None,
        out_column: Optional[torch.Tensor] = None,
        qk_row_pair: Optional[torch.Tensor] = None,
        vout_row_column_pair: Optional[torch.Tensor] = None,
        ov_mode: Optional[torch.Tensor] = None,
        qk_mode: Optional[torch.Tensor] = None,
        head: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Install static structural attention masks.

        New canonical names are ``value_row``, ``out_column``,
        ``vout_row_column_pair`` and ``qk_row_pair``.  ``ov_mode`` and
        ``qk_mode`` remain compatibility aliases for value-output row/column
        pairs and QK row pairs. These masks are independent of the input image
        and are therefore static parameter-sparse ViT pruning probes.
        """

        pair = vout_row_column_pair if vout_row_column_pair is not None else ov_mode
        qk_pair = qk_row_pair if qk_row_pair is not None else qk_mode
        self._value_row_mask = value_row.detach().to(dtype=torch.bool).cpu() if value_row is not None else None
        self._out_column_mask = out_column.detach().to(dtype=torch.bool).cpu() if out_column is not None else None
        self._ov_mode_mask = pair.detach().to(dtype=torch.bool).cpu() if pair is not None else None
        self._qk_row_pair_mask = qk_pair.detach().to(dtype=torch.bool).cpu() if qk_pair is not None else None
        self._qk_mode_mask = self._qk_row_pair_mask
        self._head_mask = head.detach().to(dtype=torch.bool).cpu() if head is not None else None
        if metadata:
            self._attention_route_mask_metadata.update(dict(metadata))

    def clear_attention_route_masks(self) -> None:
        self._value_transport_route_mask = None
        self._qk_logit_route_mask = None
        self._probability_route_mask = None
        self._value_row_mask = None
        self._out_column_mask = None
        self._qk_row_pair_mask = None
        self._ov_mode_mask = None
        self._qk_mode_mask = None
        self._head_mask = None
        self._attention_route_mask_metadata = {}

    def attention_route_mask_metadata(self) -> Dict[str, Any]:
        return dict(self._attention_route_mask_metadata)

    def _forward_with_attention_route_masks(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attn_kwargs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        unsupported = {
            name: attn_kwargs.get(name)
            for name in ("attn_mask", "key_padding_mask")
            if attn_kwargs.get(name) is not None
        }
        if unsupported:
            raise RuntimeError(
                "Native attention route masking currently requires explicit "
                f"route masks without PyTorch attn_mask/key_padding_mask; got {sorted(unsupported)}."
            )
        if bool(attn_kwargs.get("is_causal", False)):
            raise RuntimeError("Native attention route masking does not silently combine with is_causal=True.")

        dropout_p = float(getattr(self.original, "dropout", 0.0) or 0.0)
        if bool(self.original.training) and dropout_p > 0.0:
            raise RuntimeError("Native attention route masking is production-eval only when dropout is nonzero.")

        query_in, key_in, value_in = query, key, value
        if not self.batch_first:
            query_in = query.transpose(0, 1).contiguous()
            key_in = key.transpose(0, 1).contiguous()
            value_in = value.transpose(0, 1).contiguous()

        q_proj, k_proj, v_proj = self._project_qkv(query_in, key_in, value_in)
        batch_size, target_tokens, _ = q_proj.shape
        source_tokens = k_proj.shape[1]
        q_heads = self._split_heads(q_proj)
        k_heads = self._split_heads(k_proj)
        v_heads = self._split_heads(v_proj)
        qk_pair = self._expanded_mode_mask(self._qk_row_pair_mask, device=q_heads.device, dtype=q_heads.dtype)
        if qk_pair is not None:
            q_heads = q_heads * qk_pair
            k_heads = k_heads * qk_pair
        value_row = self._expanded_mode_mask(self._value_row_mask, device=v_heads.device, dtype=v_heads.dtype)
        if value_row is not None:
            v_heads = v_heads * value_row
        vout_pair = self._expanded_mode_mask(self._ov_mode_mask, device=v_heads.device, dtype=v_heads.dtype)
        if vout_pair is not None:
            v_heads = v_heads * vout_pair
        head_mask = self._expanded_head_mask(self._head_mask, device=v_heads.device, dtype=v_heads.dtype)
        if head_mask is not None:
            q_heads = q_heads * head_mask
            k_heads = k_heads * head_mask
            v_heads = v_heads * head_mask

        scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / sqrt(float(self.head_dim))
        qk_mask = self._expanded_route_mask(
            self._qk_logit_route_mask,
            batch_size=batch_size,
            target_tokens=target_tokens,
            source_tokens=source_tokens,
            device=scores.device,
        )
        if qk_mask is not None:
            active = qk_mask.any(dim=-1, keepdim=True)
            masked_scores = scores.masked_fill(~qk_mask, torch.finfo(scores.dtype).min)
            scores = torch.where(active, masked_scores, torch.zeros_like(scores))
        probs = torch.softmax(scores, dim=-1)
        if qk_mask is not None:
            probs = torch.where(qk_mask.any(dim=-1, keepdim=True), probs, torch.zeros_like(probs))

        probability_mask = self._expanded_route_mask(
            self._probability_route_mask,
            batch_size=batch_size,
            target_tokens=target_tokens,
            source_tokens=source_tokens,
            device=probs.device,
        )
        if probability_mask is not None:
            probs = probs * probability_mask.to(dtype=probs.dtype)

        value_mask = self._expanded_route_mask(
            self._value_transport_route_mask,
            batch_size=batch_size,
            target_tokens=target_tokens,
            source_tokens=source_tokens,
            device=probs.device,
        )
        if value_mask is not None:
            probs = probs * value_mask.to(dtype=probs.dtype)

        context_heads = torch.matmul(probs, v_heads)
        context = (
            context_heads.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, target_tokens, self.embed_dim)
        )
        out_column = self._expanded_mode_mask(self._out_column_mask, device=context.device, dtype=context.dtype)
        if out_column is not None:
            context = context * out_column.reshape(1, 1, self.embed_dim)
        if vout_pair is not None:
            context = context * vout_pair.reshape(1, 1, self.embed_dim)
        attn_output = F.linear(context, self.original.out_proj.weight, self.original.out_proj.bias)
        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1).contiguous()

        returned_weights: Optional[torch.Tensor] = probs
        if bool(attn_kwargs.get("average_attn_weights", True)):
            returned_weights = probs.mean(dim=1)
        return attn_output, returned_weights

    def _expanded_mode_mask(
        self,
        mask: Optional[torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        mode = mask.to(device=device, dtype=dtype)
        if mode.ndim == 1:
            if int(mode.numel()) != int(self.head_dim):
                raise RuntimeError(f"Attention mode mask has {mode.numel()} dims, expected {self.head_dim}.")
            mode = mode.view(1, self.head_dim).expand(self.num_heads, self.head_dim)
        if mode.ndim != 2 or tuple(mode.shape) != (self.num_heads, self.head_dim):
            raise RuntimeError(
                f"Attention mode mask must have shape {(self.num_heads, self.head_dim)}, got {tuple(mode.shape)}."
            )
        return mode.view(1, self.num_heads, 1, self.head_dim)

    def _expanded_head_mask(
        self,
        mask: Optional[torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        head = mask.to(device=device, dtype=dtype)
        if head.ndim != 1 or int(head.numel()) != int(self.num_heads):
            raise RuntimeError(f"Attention head mask must have shape ({self.num_heads},), got {tuple(head.shape)}.")
        return head.view(1, self.num_heads, 1, 1)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, tokens, _ = tensor.shape
        return (
            tensor.view(batch_size, tokens, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    def _expanded_route_mask(
        self,
        mask: Optional[torch.Tensor],
        *,
        batch_size: int,
        target_tokens: int,
        source_tokens: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        route = mask.to(device=device, dtype=torch.bool)
        if route.ndim == 1:
            if route.numel() != self.num_heads:
                raise RuntimeError(
                    f"Attention head mask has {route.numel()} heads, expected {self.num_heads}."
                )
            return route.view(1, self.num_heads, 1, 1).expand(batch_size, self.num_heads, target_tokens, source_tokens)
        if route.ndim == 2:
            if tuple(route.shape) != (target_tokens, source_tokens):
                raise RuntimeError(
                    f"Attention token-route mask has shape {tuple(route.shape)}, "
                    f"expected {(target_tokens, source_tokens)}."
                )
            return route.view(1, 1, target_tokens, source_tokens).expand(
                batch_size, self.num_heads, target_tokens, source_tokens
            )
        if route.ndim == 3:
            if route.shape[0] != self.num_heads or route.shape[-1] != source_tokens:
                raise RuntimeError(
                    f"Attention route mask has shape {tuple(route.shape)}, "
                    f"expected head axis {self.num_heads} and key axis {source_tokens}."
                )
            if route.shape[1] == target_tokens:
                expanded = route
            elif route.shape[1] == 1:
                expanded = torch.ones(
                    (self.num_heads, target_tokens, source_tokens),
                    device=device,
                    dtype=torch.bool,
                )
                expanded[:, :1, :] = route
            else:
                raise RuntimeError(
                    f"Attention route mask query axis {route.shape[1]} is neither 1 nor {target_tokens}."
                )
            return expanded.view(1, self.num_heads, target_tokens, source_tokens).expand(
                batch_size, self.num_heads, target_tokens, source_tokens
            )
        if route.ndim == 4:
            expected = (batch_size, self.num_heads, target_tokens, source_tokens)
            if tuple(route.shape) != expected:
                raise RuntimeError(f"Attention route mask has shape {tuple(route.shape)}, expected {expected}.")
            return route
        raise RuntimeError("Attention route masks must have shape H, T x S, H x T x S, or B x H x T x S.")

    def _project_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weight = self.original.in_proj_weight
        bias = self.original.in_proj_bias
        if weight is None:
            raise RuntimeError("MultiheadAttention must expose in_proj_weight to be wrapped.")

        if query.data_ptr() == key.data_ptr() == value.data_ptr():
            proj = F.linear(query, weight, bias)
            return proj.chunk(3, dim=-1)

        w_q, w_k, w_v = weight.chunk(3, dim=0)
        if bias is not None:
            b_q, b_k, b_v = bias.chunk(3)
        else:
            b_q = b_k = b_v = None
        q_proj = F.linear(query, w_q, b_q)
        k_proj = F.linear(key, w_k, b_k)
        v_proj = F.linear(value, w_v, b_v)
        return q_proj, k_proj, v_proj

    def _compute_context(self, attn_weights: torch.Tensor, value_proj: torch.Tensor) -> torch.Tensor:
        batch_size, tgt_len, _ = value_proj.shape
        value_heads = value_proj.view(batch_size, tgt_len, self.num_heads, self.head_dim)
        value_heads = value_heads.permute(0, 2, 1, 3)
        weights = attn_weights
        if weights.dim() == 3:
            weights = weights.unsqueeze(1)
        context_heads = torch.matmul(weights, value_heads)
        context = (
            context_heads.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, tgt_len, self.embed_dim)
        )
        return context

    def _cache_buffers(
        self,
        query_in: torch.Tensor,
        key_in: torch.Tensor,
        value_in: torch.Tensor,
        q_proj: torch.Tensor,
        k_proj: torch.Tensor,
        v_proj: torch.Tensor,
        output: torch.Tensor,
        context: torch.Tensor,
        attn_weights: Optional[torch.Tensor],
    ) -> None:
        with torch.no_grad():
            self._cache = AttentionBuffers(
                query_proj=q_proj,
                key_proj=k_proj,
                value_proj=v_proj,
                context=context,
                output=output,
                attn_weights=attn_weights,
                query_input=query_in.detach(),
                key_input=key_in.detach(),
                value_input=value_in.detach(),
            )
            if self.store_inputs:
                self._cached_query_cpu = query_in.detach().cpu()
                self._cached_key_cpu = key_in.detach().cpu()
                self._cached_value_cpu = value_in.detach().cpu()
            self._stage_stat("context", context.detach())

    def _detach_structure_cpu(self, value):
        if value is None:
            return None
        if torch.is_tensor(value):
            tensor = value.detach()
            if tensor.is_floating_point() and tensor.dtype != torch.float32:
                tensor = tensor.to(dtype=torch.float32)
            try:
                return tensor.cpu()
            except RuntimeError:
                return tensor.detach()
        if isinstance(value, dict):
            return {k: self._detach_structure_cpu(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._detach_structure_cpu(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._detach_structure_cpu(v) for v in value)
        if hasattr(value, "__dataclass_fields__"):
            kwargs = {
                field: self._detach_structure_cpu(getattr(value, field))
                for field in value.__dataclass_fields__.keys()
            }
            return value.__class__(**kwargs)
        return value

    def collect_integrated_multipliers(self) -> Optional[torch.Tensor]:
        tensor = self._output_tensor
        self._output_tensor = None
        if tensor is None:
            return None
        grad = tensor.grad
        if grad is None:
            return None
        result = grad.detach()
        tensor.grad = None
        self.saved_integrated_multipliers = result
        return result

    def prepare_for_scoring(
        self,
        *,
        cross_entropy_loss=None,
        full_acts=None,
        pre_softmax_vector=None,
        in_cat_indices_list=None,
        should_score: bool,
    ) -> None:
        self.calc_kernels = bool(should_score)
        self.compute_scores = bool(should_score)
        if not should_score:
            self._cached_query_cpu = None
            self._cached_key_cpu = None
            self._cached_value_cpu = None

    def before_backward(self) -> None:
        self.hook_in = True
        self._stream_results_ready = False

    def after_backward(self) -> None:
        self.hook_in = False
        if getattr(self, "streaming_reductions", False):
            try:
                self.flush_results()
            except Exception:
                logger.exception(
                    "[stream] Attention layer %s streaming finalize failed",
                    getattr(self, "frontier_layer_name", "unknown"),
                )
        elif getattr(self, "stream_flush_enabled", False):
            try:
                self.flush_results()
            except Exception:
                logger.exception(
                    "[stream-flush] Attention layer %s flush failed",
                    getattr(self, "frontier_layer_name", "unknown"),
                )

    def clear_state(self) -> None:
        self.saved_integrated_multipliers = None
        self.branch_multipliers = {}
        self._cache = None
        self._output_tensor = None
        self._cached_query_cpu = None
        self._cached_key_cpu = None
        self._cached_value_cpu = None
        self._staged_stats.clear()
        self._stream_results_ready = False
        self.out_proj_bias_scores = None

    def flush_results(self) -> None:
        if not (
            getattr(self, "stream_flush_enabled", False)
            or getattr(self, "streaming_reductions", False)
        ):
            return
        if getattr(self, "_stream_results_ready", False):
            return
        if self.saved_integrated_multipliers is not None:
            self.saved_integrated_multipliers = self.saved_integrated_multipliers.detach().cpu()
        if self.out_proj_bias_scores is not None:
            self.out_proj_bias_scores = self.out_proj_bias_scores.detach().cpu()
        if self.branch_multipliers:
            self.branch_multipliers = {
                k: v.detach().cpu() for k, v in self.branch_multipliers.items() if v is not None
            }
        if self._cache is not None:
            self._cache = self._detach_structure_cpu(self._cache)
        context = getattr(self, "_stream_context", None)
        if context is not None and getattr(context, "enabled", False):
            layer_name = getattr(self, "frontier_layer_name", None)
            if layer_name:
                context.submit(layer_name, None)
        self._stream_results_ready = True

    def get_boundary_inputs(self, *, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        if self._cached_query_cpu is None:
            return None
        target_device = device or self._cached_query_cpu.device
        tensor = self._cached_query_cpu
        if tensor.device != target_device:
            tensor = tensor.to(device=target_device, non_blocking=True)
        return tensor
    
    def get_inputs_for_hvp(self, *, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        return self.get_boundary_inputs(device=device)

    def get_key_inputs_for_hvp(
        self, *, device: Optional[torch.device] = None
    ) -> Optional[torch.Tensor]:
        if self._cached_key_cpu is None:
            return None
        target_device = device or self._cached_key_cpu.device
        tensor = self._cached_key_cpu
        if tensor.device != target_device:
            tensor = tensor.to(device=target_device, non_blocking=True)
        return tensor

    def get_value_inputs_for_hvp(
        self, *, device: Optional[torch.device] = None
    ) -> Optional[torch.Tensor]:
        if self._cached_value_cpu is None:
            return None
        target_device = device or self._cached_value_cpu.device
        tensor = self._cached_value_cpu
        if tensor.device != target_device:
            tensor = tensor.to(device=target_device, non_blocking=True)
        return tensor

    @property
    def attn_buffers(self) -> Optional[AttentionBuffers]:
        return self._cache

    def pop_frontier_stats(self) -> Dict[str, Any]:
        stats = self._staged_stats
        self._staged_stats = {}
        return stats

    def apply_local_ig(self, grad_output: torch.Tensor) -> None:
        cache = self._cache
        if cache is None:
            return

        grad = grad_output.detach()
        if not self.batch_first:
            grad = grad.transpose(0, 1).contiguous()
        scale = getattr(self, "scale", 1.0)
        if isinstance(scale, torch.Tensor):
            scale = float(scale.detach().cpu())
        scale = float(scale)
        if abs(scale) < 1e-12:
            scale = 1.0
        grad_factor = grad / scale

        context = cache.context
        if torch.is_tensor(context):
            out_contrib = torch.einsum("bti,btj->bij", grad_factor, context)
            self.saved_integrated_multipliers = out_contrib.detach()
            self.out_proj_bias_scores = grad_factor.sum(dim=(0, 1)).detach()
        else:
            self.saved_integrated_multipliers = grad_factor.detach()
            self.out_proj_bias_scores = grad_factor.sum(dim=(0, 1)).detach()

        branch_raw = compute_attention_branch_multipliers(
            grad_output=grad_factor,
            query_proj=cache.query_proj,
            key_proj=cache.key_proj,
            value_proj=cache.value_proj,
            attn_weights=cache.attn_weights,
            out_proj_weight=getattr(self.original, "out_proj", None).weight  # type: ignore[arg-type]
            if hasattr(self.original, "out_proj")
            else None,
            num_heads=self.num_heads,
        )
        branch: Dict[str, torch.Tensor] = {}
        query_input = cache.query_input
        key_input = cache.key_input
        value_input = cache.value_input
        if torch.is_tensor(query_input):
            branch["query"] = torch.einsum("bti,btj->bij", branch_raw["query"], query_input).detach()
            branch["query_bias"] = branch_raw["query"].sum(dim=(0, 1)).detach()
        else:
            branch["query"] = branch_raw["query"].detach()
        if torch.is_tensor(key_input):
            branch["key"] = torch.einsum("bti,btj->bij", branch_raw["key"], key_input).detach()
            branch["key_bias"] = branch_raw["key"].sum(dim=(0, 1)).detach()
        else:
            branch["key"] = branch_raw["key"].detach()
        if torch.is_tensor(value_input):
            branch["value"] = torch.einsum("bti,btj->bij", branch_raw["value"], value_input).detach()
            branch["value_bias"] = branch_raw["value"].sum(dim=(0, 1)).detach()
        else:
            branch["value"] = branch_raw["value"].detach()
        self.branch_multipliers = branch
        self._stage_stat("branch_multipliers", branch)

    def _stage_stat(self, key: str, value: Any) -> None:
        if not bool(getattr(self, "stage_stats_enabled", True)):
            return
        if isinstance(value, torch.Tensor):
            self._staged_stats[key] = value.detach().cpu()
        elif isinstance(value, dict):
            self._staged_stats[key] = {k: v.detach().cpu() for k, v in value.items()}
        else:
            self._staged_stats[key] = value

@dataclass
class SwiGLUCaches:
    gate_pre: torch.Tensor
    up_pre: torch.Tensor
    activated_gate: torch.Tensor
    hidden: torch.Tensor
    output: torch.Tensor
    inputs: torch.Tensor


class HookedSwiGLU(nn.Module):
    """Wrapper for SwiGLU-style blocks capturing branch activations and IG multipliers."""

    def __init__(self, module: nn.Module, *, activation: str = "silu") -> None:
        super().__init__()
        self.original = module
        self.activation = activation

        self.calc_kernels = False
        self.compute_scores = False
        self.hook_in = False
        self.ig_capture_output = True

        self.saved_integrated_multipliers: Optional[torch.Tensor] = None
        self.branch_multipliers: Dict[str, torch.Tensor] = {}
        self._caches: Optional[SwiGLUCaches] = None
        self._output_tensor: Optional[torch.Tensor] = None
        self._cached_inputs_cpu: Optional[torch.Tensor] = None
        self._staged_stats: Dict[str, Any] = {}
        self.stream_flush_enabled: bool = False
        self.stream_flush_replay_device: Optional[torch.device] = None
        self._stream_results_ready: bool = False
        self._stream_context = None
        self.frontier_layer_name: Optional[str] = None
        self.down_proj_bias_scores: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_pre, up_pre = self._project_branches(x)
        activated_gate = self._activate(gate_pre)
        hidden = activated_gate * up_pre
        output = self._project_output(hidden)
        if output.requires_grad:
            output = output.clone()
            output.retain_grad()
        self._output_tensor = output
        self._cache(
            gate_pre=gate_pre.detach(),
            up_pre=up_pre.detach(),
            activated_gate=activated_gate.detach(),
            hidden=hidden.detach(),
            output=output.detach(),
            inputs=x,
        )
        return output

    def _detach_structure_cpu(self, value):
        if value is None:
            return None
        if torch.is_tensor(value):
            tensor = value.detach()
            if tensor.is_floating_point() and tensor.dtype != torch.float32:
                tensor = tensor.to(dtype=torch.float32)
            try:
                return tensor.cpu()
            except RuntimeError:
                return tensor.detach()
        if isinstance(value, dict):
            return {k: self._detach_structure_cpu(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._detach_structure_cpu(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._detach_structure_cpu(v) for v in value)
        if hasattr(value, "__dataclass_fields__"):
            kwargs = {
                field: self._detach_structure_cpu(getattr(value, field))
                for field in value.__dataclass_fields__.keys()
            }
            return value.__class__(**kwargs)
        return value

    def _project_branches(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        module = self.original
        if hasattr(module, "gate_proj") and hasattr(module, "up_proj"):
            gate = module.gate_proj(x)
            up = module.up_proj(x)
            return gate, up
        if hasattr(module, "wi"):
            proj = module.wi(x)
            gate, up = proj.chunk(2, dim=-1)
            return gate, up
        raise TypeError(
            "Unsupported SwiGLU module; expected gate_proj/up_proj or fused wi projection."
        )

    def _project_output(self, hidden: torch.Tensor) -> torch.Tensor:
        module = self.original
        if hasattr(module, "down_proj"):
            return module.down_proj(hidden)
        if hasattr(module, "wo"):
            return module.wo(hidden)
        return hidden

    def _activate(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.activation == "silu":
            return F.silu(tensor)
        if self.activation == "gelu":
            return F.gelu(tensor)
        raise ValueError(f"Unsupported SwiGLU activation {self.activation!r}")

    def _cache(
        self,
        *,
        gate_pre: torch.Tensor,
        up_pre: torch.Tensor,
        activated_gate: torch.Tensor,
        hidden: torch.Tensor,
        output: torch.Tensor,
        inputs: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            self._caches = SwiGLUCaches(
                gate_pre=gate_pre,
                up_pre=up_pre,
                activated_gate=activated_gate,
                hidden=hidden,
                output=output,
                inputs=inputs.detach(),
            )
            self._cached_inputs_cpu = inputs.detach().cpu()
            self._stage_stat("preactivations", {"gate": gate_pre, "up": up_pre})

    def collect_integrated_multipliers(self) -> Optional[torch.Tensor]:
        tensor = self._output_tensor
        self._output_tensor = None
        if tensor is None:
            return None
        grad = tensor.grad
        if grad is None:
            return None
        self.saved_integrated_multipliers = grad.detach()
        tensor.grad = None
        return self.saved_integrated_multipliers

    def prepare_for_scoring(
        self,
        *,
        cross_entropy_loss=None,
        full_acts=None,
        pre_softmax_vector=None,
        in_cat_indices_list=None,
        should_score: bool,
    ) -> None:
        self.calc_kernels = bool(should_score)
        self.compute_scores = bool(should_score)
        if not should_score:
            self._cached_inputs_cpu = None

    def before_backward(self) -> None:
        self.hook_in = True
        self._stream_results_ready = False

    def after_backward(self) -> None:
        self.hook_in = False
        if getattr(self, "streaming_reductions", False):
            try:
                self.flush_results()
            except Exception:
                logger.exception(
                    "[stream] SwiGLU layer %s streaming finalize failed",
                    getattr(self, "frontier_layer_name", "unknown"),
                )
        elif getattr(self, "stream_flush_enabled", False):
            try:
                self.flush_results()
            except Exception:
                logger.exception(
                    "[stream-flush] SwiGLU layer %s flush failed",
                    getattr(self, "frontier_layer_name", "unknown"),
                )

    def clear_state(self) -> None:
        self.saved_integrated_multipliers = None
        self.branch_multipliers = {}
        self._caches = None
        self._output_tensor = None
        self._cached_inputs_cpu = None
        self._staged_stats.clear()
        self._stream_results_ready = False
        self.down_proj_bias_scores = None

    def flush_results(self) -> None:
        if not (
            getattr(self, "stream_flush_enabled", False)
            or getattr(self, "streaming_reductions", False)
        ):
            return
        if getattr(self, "_stream_results_ready", False):
            return
        if self.saved_integrated_multipliers is not None:
            self.saved_integrated_multipliers = self.saved_integrated_multipliers.detach().cpu()
        if self.down_proj_bias_scores is not None:
            self.down_proj_bias_scores = self.down_proj_bias_scores.detach().cpu()
        if self.branch_multipliers:
            self.branch_multipliers = {
                k: v.detach().cpu() for k, v in self.branch_multipliers.items() if v is not None
            }
        if self._caches is not None:
            self._caches = self._detach_structure_cpu(self._caches)
        if self._cached_inputs_cpu is None and self._caches is not None:
            inputs = getattr(self._caches, "inputs", None)
            if torch.is_tensor(inputs):
                self._cached_inputs_cpu = inputs.detach().cpu()
        self._stream_results_ready = True

    def get_boundary_inputs(self, *, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        if self._cached_inputs_cpu is None:
            return None
        target_device = device or self._cached_inputs_cpu.device
        tensor = self._cached_inputs_cpu
        if tensor.device != target_device:
            tensor = tensor.to(device=target_device, non_blocking=True)
        return tensor
    
    def get_inputs_for_hvp(self, *, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        return self.get_boundary_inputs(device=device)

    @property
    def caches(self) -> Optional[SwiGLUCaches]:
        return self._caches

    def pop_frontier_stats(self) -> Dict[str, Any]:
        stats = self._staged_stats
        self._staged_stats = {}
        return stats

    def apply_local_ig(self, grad_output: torch.Tensor) -> None:
        caches = self._caches
        if caches is None:
            return

        grad = grad_output.detach()
        scale = getattr(self, "scale", 1.0)
        if isinstance(scale, torch.Tensor):
            scale = float(scale.detach().cpu())
        scale = float(scale)
        if abs(scale) < 1e-12:
            scale = 1.0
        grad_factor = grad / scale

        weight = None
        if hasattr(self.original, "down_proj"):
            weight = self.original.down_proj.weight
        elif hasattr(self.original, "wo"):
            weight = self.original.wo.weight

        if torch.is_tensor(caches.hidden):
            down_contrib = torch.einsum("bti,btj->bij", grad_factor, caches.hidden).detach()
            self.saved_integrated_multipliers = down_contrib
            self.down_proj_bias_scores = grad_factor.sum(dim=(0, 1)).detach()
        else:
            self.saved_integrated_multipliers = grad_factor.detach()
            self.down_proj_bias_scores = grad_factor.sum(dim=(0, 1)).detach()

        grad_hidden = grad_factor
        if weight is not None:
            grad_hidden = torch.matmul(grad_factor, weight)

        grad_up = grad_hidden * caches.activated_gate
        grad_gate_act = grad_hidden * caches.up_pre
        grad_gate_pre = grad_gate_act * self._activation_derivative(caches.gate_pre)

        branch = {
            "gate": torch.einsum("bti,btj->bij", grad_gate_pre, caches.inputs).detach()
            if torch.is_tensor(caches.inputs)
            else grad_gate_pre.detach(),
            "up": torch.einsum("bti,btj->bij", grad_up, caches.inputs).detach()
            if torch.is_tensor(caches.inputs)
            else grad_up.detach(),
        }
        branch["gate_bias"] = grad_gate_pre.sum(dim=(0, 1)).detach()
        branch["up_bias"] = grad_up.sum(dim=(0, 1)).detach()
        self.branch_multipliers = branch
        self._stage_stat("branch_multipliers", branch)

    def _activation_derivative(self, pre: torch.Tensor) -> torch.Tensor:
        if self.activation == "silu":
            sig = torch.sigmoid(pre)
            return sig * (1 + pre * (1 - sig))
        if self.activation == "gelu":
            pre_clone = pre.detach().requires_grad_(True)
            out = F.gelu(pre_clone)
            ones = torch.ones_like(out)
            grad = torch.autograd.grad(
                out, pre_clone, ones, retain_graph=False, create_graph=False
            )[0]
            return grad.detach()
        raise ValueError(f"Unsupported SwiGLU activation {self.activation!r}")

    def _stage_stat(self, key: str, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            self._staged_stats[key] = value.detach().cpu()
        elif isinstance(value, dict):
            self._staged_stats[key] = {k: v.detach().cpu() for k, v in value.items()}
        else:
            self._staged_stats[key] = value

def convert_multihead_attention(module: nn.Module) -> HookedMultiheadAttention:
    """Convert an attention module into its hooked counterpart."""

    if isinstance(module, HookedMultiheadAttention):
        return module
    if not isinstance(module, nn.MultiheadAttention):
        raise TypeError("convert_multihead_attention expects an nn.MultiheadAttention module")
    return HookedMultiheadAttention(module)


def convert_swiglu(module: nn.Module) -> HookedSwiGLU:
    """Convert a SwiGLU-style module into a hooked wrapper."""

    if isinstance(module, HookedSwiGLU):
        return module
    return HookedSwiGLU(module)
