from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, Set
import logging
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from packages.salience.ops.linear_utils import safe_pearson_corr
from packages.salience.utils.layer_logging import (
    finalize_layer_bie_corr_logging as _finalize_layer_bie_corr_logging,
    finalize_layer_corr_logging as _finalize_layer_corr_logging,
    flatten_objective_vec as _flatten_objective_vec,
    maybe_record_layer_corr_sample as _maybe_record_layer_corr_sample,
)
from packages.salience.utils import (
    compile_if_available,
    pack_boolean_rows,
    vectorised_bitpack,
    TensorReplayProvider,
    replay_batch_size as _replay_batch_size,
    replay_matches_tensor as _replay_matches_tensor,
    fetch_replay_slice as _fetch_replay_slice,
    fetch_replay_indices as _fetch_replay_indices,
)
from packages.salience.ops.conv_emit import (
    conv_compute_kernel_score_aggregated,
    _build_conv_channel_virtual_weight,
    _compute_conv_channel_virtual_preadd,
    _conv_channel_virtual_reduce_exact_chunk,
    _conv_reduce_kernel_exact_chunk,
    _normalize_conv_score_basis,
    _per_sample_conv2d_weight,
)
from packages.salience.ops.conv_frontier_aggregate import (
    _frontier_aggregate_conv_block_size,
    _frontier_aggregate_conv_impl,
    conv_compute_frontier_aggregate_channel_virtual_preadd,
    conv_compute_frontier_aggregate_channel_virtual_fused,
    fused_frontier_aggregate_channel_virtual_available,
)
from packages.salience.ops.abs_mode import resolve_abs_aggregation_mode
from packages.salience.core.reducers import ReducerBundle
from packages.salience.analysis.extraction import FeatureExtractionMixin

from .frontier_utils import (
    COMPUTE_BITPACK,
    COMPUTE_LOSS_DELTA,
    DEFAULT_TARGET_CHUNK_ELEMENTS,
    MAX_CHUNK_SAMPLES,
    REPLAY_STORAGE_DEVICE,
    VALIDATE_FRONTIER,
    GPUStats,
    current_replay_provider,
    debug_print,
    pop_replay_provider,
    push_replay_provider,
    choose_chunk_size,
)
logger = logging.getLogger(__name__)
OLDFAST_TRACE = bool(int(os.environ.get("SCORING_OLDFAST_TRACE", "0")))
__all__ = ["HookedConv2d", "conv_compute_kernel_score_aggregated"]


class HookedConv2d(FeatureExtractionMixin, nn.Module):
    """
    Optimized Conv2d wrapper that computes frontier scores using the weight-gradient
    identity instead of constructing per-(Cin,Cout) grouped convolutions. It supports
    both aggregated and per-sample scoring while maintaining the legacy public API.
    """

    def __init__(
        self,
        from_conv: nn.Conv2d,
        absolute_score: bool = True,
        score_neg: bool = True,
        normalize_features_per_image: bool = False,
        weight_norm_type: str = "sum_abs",
        score_post_af: bool = False,
        store_per_sample: bool = True,
        *,
        collect_integrated: bool = False,
        collect_stats: bool = False,
        collect_hvp: bool = False,
        conv_score_basis: Optional[str] = None,
        abs_aggregation_mode: Optional[str] = None,
    ):
        super().__init__()

        # Static layer metadata
        self.in_features = from_conv.in_channels
        self.out_features = from_conv.out_channels
        self.kernel_size = from_conv.kernel_size
        self.stride = from_conv.stride
        self.padding = from_conv.padding
        self.dilation = from_conv.dilation
        self.groups = from_conv.groups

        # Trainable parameters cloned from the source conv
        self.weight = nn.Parameter(from_conv.weight.detach().clone())
        if from_conv.bias is not None:
            self.bias = nn.Parameter(from_conv.bias.detach().clone())
            self.original_bias = self.bias.detach().clone()
        else:
            self.register_parameter("bias", None)
            self.original_bias = None

        # Scoring configuration
        self.absolute = absolute_score
        self.score_neg = score_neg
        self.normalize_features_per_image = normalize_features_per_image
        self.weight_norm_type = weight_norm_type
        self.score_post_af = score_post_af
        self.store_per_sample = bool(store_per_sample)
        self.conv_score_basis = _normalize_conv_score_basis(conv_score_basis)
        self.score_basis = self.conv_score_basis
        self.abs_aggregation_mode = resolve_abs_aggregation_mode(
            "exact_abs",
            explicit_mode=abs_aggregation_mode,
        )
        self.capture_sample_sums: bool = False
        self.capture_bitpack: bool = COMPUTE_BITPACK
        self.collect_score_correlation: bool = True
        self.compute_neuron_metrics: bool = True
        self.dual_channel_virtual: bool = False
        self.collect_integrated = bool(collect_integrated)
        self.collect_stats = bool(collect_stats)
        self.collect_hvp = bool(collect_hvp)
        self.collect_bie: bool = False
        self._chunk_records: Dict[int, Dict[str, torch.Tensor]] = {}
        self._chunk_capture_failed: bool = False
        self._pending_chunk_ids: Set[int] = set()
        self.bie_logging_enabled: bool = False
        self._stream_bie_success: bool = False

        # Runtime state used by the pipeline
        self.calc_kernels = False
        self.compute_scores = False
        self.get_act_weights = True
        self.mask = None
        self.bias_mask = None
        self.original_weights = self.weight.detach().clone()
        self.scale = 1.0
        self.results = None
        self._oldfast_results_cache = None
        self.full_acts = None
        self.pre_softmax_vector = None
        self.cross_entropy_loss = None
        self.target_unit = None
        self.target_unit_acts = None
        self.final_layer = False
        self.label = None
        self.grads_to_final_acts = None
        self.final_acts = None
        self.original_layer_activations = None
        self.pre_sum_output = None
        self.post_sum_output = None
        self.saved_weight_gradients = None
        self.hook_in = False
        self.bitpack_sink: Optional[Callable[[torch.Tensor], None]] = None

        # Buffers for gradient capture and replay
        self.saved_neuron_gradients: Optional[torch.Tensor] = None
        self.saved_integrated_multipliers: Optional[torch.Tensor] = None
        self._added_out: Optional[torch.Tensor] = None
        self._saved_output_buffer: Optional[torch.Tensor] = None
        self._output_hook: Optional[torch.utils.hooks.RemovableHandle] = None
        self._pending_replay_provider: Optional[object] = None
        self._active_replay_provider: Optional[object] = None
        self._warned_replay_batch_mismatch = False
        self.ig_capture_output = True
        self.z_cache: Optional[torch.Tensor] = None
        self.retain_z_cache_until_reset: bool = False

        # Running statistics for mean activations
        self.running_activation_sum: Optional[torch.Tensor] = None
        self.running_activation_count: int = 0
        self._pending_activation_sum: Optional[torch.Tensor] = None
        self._pending_activation_count: int = 0

        self.device = self.weight.device

        # Frontier diagnostics
        self.frontier_stats: Optional[Dict[str, float]] = None
        self.analysis_payload: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
        self.gpu_stats: Optional[GPUStats] = GPUStats(device=self.device)
        self.stream_flush_enabled: bool = False
        self.stream_flush_replay_device: Optional[torch.device] = None
        self._stream_results_ready: bool = False
        self._stream_context = None
        self.frontier_layer_name: Optional[str] = None

        # Streaming reducers
        self.streaming_reductions: bool = False
        self._reducers: Optional[ReducerBundle] = None
        self._stream_cache: Dict[str, Optional[torch.Tensor]] = {}
        self._stream_category_indices: Optional[torch.Tensor] = None
        self._stream_processed: bool = False
        self._last_stream_chunk_size: Optional[int] = None
        self._active_stream_chunk_size: Optional[int] = None
        self._stream_sample_sums: List[torch.Tensor] = []
        self._stream_sample_abs_sums: List[torch.Tensor] = []
        self._sample_sum_payload: Optional[torch.Tensor] = None
        self._sample_abs_sum_payload: Optional[torch.Tensor] = None
        self.oldfast_mode: bool = False
        self._oldfast_inputs: Optional[torch.Tensor] = None
        self._preserve_score_capture_state: bool = False
        self._oldfast_preadd_cache_key: Optional[Tuple[object, ...]] = None
        self._oldfast_preadd_weight: Optional[torch.Tensor] = None
        self._oldfast_in_per_group: Optional[int] = None
        self._oldfast_out_per_group: Optional[int] = None

        # Retained payload diagnostics
        self.retain_backward_payloads: bool = False
        self.results_ready: Dict[str, List[Dict[str, object]]] = {}
        self.retained_payloads: Dict[str, Dict[str, object]] = {}
        self.enable_layer_corr_logging: bool = False

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _effective_weight(
        self,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        weight = self.weight
        if dtype is not None and weight.dtype != dtype:
            weight = weight.to(dtype=dtype)
        if device is not None and weight.device != device:
            weight = weight.to(device=device)
        mask = getattr(self, "mask", None)
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise TypeError("mask must be a torch.Tensor when provided.")
            mask_tensor = mask.to(device=weight.device, dtype=weight.dtype)
            if mask_tensor.dim() == 1:
                expected = self.out_features * (self.in_features // self.groups)
                if mask_tensor.numel() == expected:
                    mask_tensor = mask_tensor.view(self.out_features, -1)
                mask_tensor = mask_tensor.unsqueeze(-1).unsqueeze(-1)
            elif mask_tensor.dim() == 2:
                mask_tensor = mask_tensor.unsqueeze(-1).unsqueeze(-1)
            elif mask_tensor.dim() == 3:
                mask_tensor = mask_tensor.unsqueeze(-1)
            elif mask_tensor.dim() > 4:
                raise ValueError("mask tensor dimensionality is unsupported.")
            weight = weight * mask_tensor
        return weight

    def _effective_bias(
        self,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Optional[torch.Tensor]:
        bias = self.bias
        if bias is None:
            return None
        if dtype is not None and bias.dtype != dtype:
            bias = bias.to(dtype=dtype)
        if device is not None and bias.device != device:
            bias = bias.to(device=device)
        bias_mask = getattr(self, "bias_mask", None)
        if bias_mask is not None:
            bias = bias * bias_mask.to(device=bias.device, dtype=bias.dtype)
        return bias

    def _oldfast_preadd_cache_signature(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[object, ...]:
        mask = getattr(self, "mask", None)
        mask_ptr = int(mask.data_ptr()) if isinstance(mask, torch.Tensor) else None
        mask_version = int(getattr(mask, "_version", 0)) if isinstance(mask, torch.Tensor) else None
        return (
            device.type,
            device.index,
            str(dtype),
            int(self.weight.data_ptr()),
            int(getattr(self.weight, "_version", 0)),
            mask_ptr,
            mask_version,
        )

    def _get_oldfast_preadd_payload(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        weight = self._effective_weight(device=device, dtype=dtype)
        cache_key = self._oldfast_preadd_cache_signature(dtype=dtype, device=device)
        if self._oldfast_preadd_cache_key != cache_key or self._oldfast_preadd_weight is None:
            preadd_weight, in_per_group, out_per_group = _build_conv_channel_virtual_weight(
                weight,
                groups=self.groups,
            )
            self._oldfast_preadd_weight = preadd_weight.to(device=device, dtype=dtype).contiguous()
            self._oldfast_in_per_group = int(in_per_group)
            self._oldfast_out_per_group = int(out_per_group)
            self._oldfast_preadd_cache_key = cache_key
        return (
            weight,
            self._oldfast_preadd_weight,
            int(self._oldfast_in_per_group),
            int(self._oldfast_out_per_group),
        )

    def _detach_structure_cpu(self, value):
        if value is None:
            return None
        if torch.is_tensor(value):
            tensor = value.detach()
            if tensor.is_floating_point() and tensor.dtype != torch.float32:
                tensor = tensor.to(dtype=torch.float32)
            use_async = bool(getattr(self, "_async_pinned_flush", False))
            if use_async and tensor.is_cuda:
                try:
                    pinned = torch.empty_like(tensor, device="cpu", pin_memory=True)
                    pinned.copy_(tensor, non_blocking=True)
                    return pinned
                except Exception:
                    pass
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
        return value

    # Streaming helpers -------------------------------------------------
    def _streaming_enabled(self) -> bool:
        return bool(self.streaming_reductions)

    def _use_channel_virtual_basis(self) -> bool:
        return self.conv_score_basis == "channel_virtual"

    def _supports_conv_bie(self) -> bool:
        return not self._use_channel_virtual_basis()

    def _effective_collect_bie(self) -> bool:
        return bool(self.collect_bie and self._supports_conv_bie())

    def set_retain_backward_payloads(self, enabled: bool) -> None:
        """Enable or disable retention of finalized backward payloads."""
        self.retain_backward_payloads = bool(enabled)
        if not enabled:
            self.results_ready.clear()
            self.retained_payloads.clear()

    def _estimate_bytes(self, obj: object) -> int:
        if torch.is_tensor(obj):
            tensor = obj  # type: ignore[assignment]
            return tensor.element_size() * tensor.numel()
        if isinstance(obj, dict):
            return sum(self._estimate_bytes(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(self._estimate_bytes(v) for v in obj)
        return 0

    def _store_results_ready(self, result_tuple: Tuple[object, ...], duration_s: float) -> None:
        if not self.retain_backward_payloads:
            return
        layer_key = getattr(self, "frontier_layer_name", None) or self.label or "unknown"
        total_bytes = self._estimate_bytes(result_tuple)
        unit_slice = getattr(self, "current_unit_slice", None)
        logger.info(
            "[payload-finalize] layer=%s unit_slice=%s time_ms=%.2f retained_bytes=%.3fMB",
            layer_key,
            unit_slice,
            duration_s * 1e3,
            total_bytes / (1024 * 1024) if total_bytes else 0.0,
        )
        payload = {
            "results": result_tuple,
            "unit_slice": unit_slice,
            "duration_s": duration_s,
            "bytes": total_bytes,
        }
        self.results_ready.setdefault(layer_key, []).append(payload)

    def _detach_to_cpu(self, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        try:
            return tensor.detach().cpu()
        except Exception:
            logger.debug(
                "[payload-retain] Failed to move tensor to CPU for layer=%s",
                getattr(self, "frontier_layer_name", "unknown"),
                exc_info=True,
            )
            return tensor.detach()

    def _record_retained_payload(
        self,
        *,
        sample_sum: Optional[torch.Tensor],
        sample_abs_sum: Optional[torch.Tensor],
        all_payload: Mapping[str, Optional[torch.Tensor]],
        cat_payload: Optional[Mapping[str, Optional[torch.Tensor]]],
        bitpacked: Sequence[torch.Tensor],
        grad_non_zeroes: Optional[torch.Tensor],
        weight_out: Optional[torch.Tensor],
    ) -> None:
        if not self.retain_backward_payloads:
            return
        layer_key = getattr(self, "frontier_layer_name", None) or self.label or "unknown"
        store: Dict[str, object] = {
            "sample_sums": self._detach_to_cpu(sample_sum),
            "sample_abs_sums": self._detach_to_cpu(sample_abs_sum),
            "all_payload": {
                key: self._detach_to_cpu(value) for key, value in all_payload.items()
            },
            "cat_payload": (
                {key: self._detach_to_cpu(value) for key, value in cat_payload.items()}
                if isinstance(cat_payload, Mapping)
                else {}
            ),
            "bitpacked": [bp.detach().cpu() for bp in bitpacked],
            "grad_non_zeroes": self._detach_to_cpu(grad_non_zeroes),
        }
        if weight_out is not None:
            store["weight_out"] = self._detach_to_cpu(weight_out)

        def _tensor_bytes(obj: Optional[torch.Tensor]) -> int:
            if obj is None or not torch.is_tensor(obj):
                return 0
            return obj.element_size() * obj.numel()

        total_bytes = 0
        for value in store.values():
            if isinstance(value, dict):
                for tensor in value.values():
                    total_bytes += _tensor_bytes(tensor if torch.is_tensor(tensor) else None)
            elif isinstance(value, list):
                for tensor in value:
                    total_bytes += _tensor_bytes(tensor if torch.is_tensor(tensor) else None)
            else:
                total_bytes += _tensor_bytes(value if torch.is_tensor(value) else None)

        logger.info(
            "[payload-retain] layer=%s retained_payload_bytes=%.3fMB",
            layer_key,
            total_bytes / (1024 * 1024) if total_bytes else 0.0,
        )
        self.retained_payloads[layer_key] = store

    def _reset_stream_state(self) -> None:
        self._commit_running_activation_stats()
        self._reducers = None
        self._stream_cache = {}
        self._stream_category_indices = None
        self._stream_processed = False
        self._last_stream_chunk_size = None
        self._active_stream_chunk_size = None
        self._pending_activation_sum = None
        self._pending_activation_count = 0
        self._stream_sample_sums = []
        self._stream_sample_abs_sums = []
        self._sample_sum_payload = None
        self._sample_abs_sum_payload = None
        self._pending_chunk_ids.clear()
        self._stream_bie_success = False
        self._chunk_capture_failed = False

    def _reset_chunk_capture(self) -> None:
        self._chunk_capture_failed = False

    def _mark_chunk_capture_failed(self) -> None:
        self._chunk_capture_failed = True

    def _capture_stream_cache(self, *, full_acts, pre_softmax_vector, cross_entropy_loss) -> None:
        def _to_cpu(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if t is None:
                return None
            return t.detach().cpu()

        self._stream_cache = {
            "full_acts": _to_cpu(full_acts),
            "pre_softmax": _to_cpu(pre_softmax_vector),
            "cross_entropy": _to_cpu(cross_entropy_loss),
            "target_unit_acts": _to_cpu(self.target_unit_acts),
            "labels": _to_cpu(self.label),
        }

    def _fetch_stream_cache_slice(
        self,
        key: str,
        start: int,
        end: int,
        *,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        source = self._stream_cache.get(key)
        if source is None:
            return None
        slice_tensor = source[start:end]
        if slice_tensor.numel() == 0:
            return slice_tensor.to(device=device)
        if slice_tensor.device == device:
            return slice_tensor
        return slice_tensor.to(device=device, non_blocking=True)

    def _gather_stream_cache(
        self,
        key: str,
        indices: torch.Tensor,
        *,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        source = self._stream_cache.get(key)
        if source is None:
            return None
        gathered = source.index_select(0, indices.to(device=source.device))
        if gathered.numel() == 0:
            return gathered.to(device=device)
        if gathered.device == device:
            return gathered
        return gathered.to(device=device, non_blocking=True)

    def _initialize_reducers(self) -> None:
        track_categories = (
            self._stream_category_indices is not None
            and self._stream_category_indices.numel() > 0
        )
        self._reducers = ReducerBundle(
            device=self.weight.device,
            track_categories=track_categories,
            compute_neuron_metrics=self.compute_neuron_metrics,
            collect_bitpack=bool(getattr(self, "capture_bitpack", COMPUTE_BITPACK)),
            collect_bie=self._effective_collect_bie(),
        )
        self._reducers.reset({"compute_neuron_metrics": self.compute_neuron_metrics})
        self._stream_processed = False

    def _set_stream_category_indices(self, indices: Optional[Sequence[int]]) -> None:
        if indices is None:
            self._stream_category_indices = None
            return
        if isinstance(indices, torch.Tensor):
            tensor = indices.detach().to(device="cpu", dtype=torch.long)
        else:
            tensor = torch.as_tensor(list(indices), dtype=torch.long)
        self._stream_category_indices = tensor

    def _oldfast_score_requested(self) -> bool:
        return bool(self.oldfast_mode and (self.compute_scores or self.calc_kernels))

    def _compute_oldfast_results(self, grad: torch.Tensor) -> None:
        if not self._oldfast_score_requested():
            return
        if not self._use_channel_virtual_basis():
            raise RuntimeError("Oldfast conv scoring requires channel_virtual basis.")
        inputs = self._oldfast_inputs
        if inputs is None and self._active_replay_provider is not None:
            try:
                inputs = self._active_replay_provider.get(
                    0,
                    grad.size(0),
                    device=self.weight.device,
                    dtype=self.weight.dtype,
                )
            except Exception:
                inputs = None
        if inputs is None:
            raise RuntimeError("Oldfast conv scoring requires saved forward inputs.")
        device = self.weight.device
        scale = self.scale if self.scale is not None else 1.0
        if isinstance(scale, torch.Tensor):
            scale = float(scale.item())
        if abs(scale) < 1e-12:
            scale = 1.0
        grad_detached = grad.detach().to(device=device, dtype=torch.float32)
        inputs_source = inputs.detach().to(device=device)
        weight, preadd_weight, in_per_group, out_per_group = self._get_oldfast_preadd_payload(
            device=device,
            dtype=torch.float32,
        )
        impl = _frontier_aggregate_conv_impl()
        fused_results = None
        can_use_specialized_oldfast = not self.compute_neuron_metrics
        if (
            can_use_specialized_oldfast
            and impl == "fused"
            and fused_frontier_aggregate_channel_virtual_available()
        ):
            fused_results = conv_compute_frontier_aggregate_channel_virtual_fused(
                inputs=inputs_source,
                grads=grad_detached,
                weights=weight,
                scale=scale,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                device=device,
                abs_aggregation_mode=self.abs_aggregation_mode,
            )
        self.results = (
            fused_results
            if fused_results is not None
            else (
                conv_compute_kernel_score_aggregated(
                    inputs_source=inputs_source,
                    saved_neuron_gradients=grad_detached,
                    weights=weight,
                    scale=scale,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                    device=device,
                    compute_neuron_metrics=self.compute_neuron_metrics,
                    bitpack_sink=None,
                    bias=None,
                    validate_frontier=False,
                    target_acts=None,
                    stats_callback=None,
                    return_sample_sums=False,
                    debug_layer_name=None,
                    score_basis="channel_virtual",
                    dual_channel_virtual=False,
                    auxiliary_payload=None,
                    emit_bitpack=False,
                    abs_aggregation_mode=self.abs_aggregation_mode,
                    preadd_weight=preadd_weight,
                    in_per_group=in_per_group,
                    out_per_group=out_per_group,
                )
                if (impl == "reference" or self.compute_neuron_metrics)
                else conv_compute_frontier_aggregate_channel_virtual_preadd(
                    inputs_source=inputs_source,
                    grads=grad_detached,
                    weights=weight,
                    scale=scale,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                    device=device,
                    abs_aggregation_mode=self.abs_aggregation_mode,
                    preadd_weight=preadd_weight,
                    in_per_group=in_per_group,
                    out_per_group=out_per_group,
                )
            )
        )
        self._oldfast_results_cache = self.results
        context = getattr(self, "_stream_context", None)
        layer_name = getattr(self, "frontier_layer_name", None)
        if context is not None and getattr(context, "enabled", False) and layer_name:
            context.submit(layer_name, self.results)
        self._stream_results_ready = True
        if OLDFAST_TRACE:
            logger.warning(
                "[oldfast-trace:hook] layer=%s id=%s results_set=%s tuple_len=%s grad_shape=%s",
                getattr(self, "frontier_layer_name", "unknown"),
                id(self),
                self.results is not None,
                len(self.results) if isinstance(self.results, tuple) else None,
                tuple(grad.shape),
            )
        self.analysis_payload = None
        self.saved_neuron_gradients = None
        self.saved_integrated_multipliers = None
        self._oldfast_inputs = None

    def _register_output_hook(self, output: torch.Tensor) -> None:
        if not output.requires_grad:
            self._teardown_output_hook()
            self.saved_neuron_gradients = None
            self._saved_output_buffer = None
            return
        if self._output_hook is not None:
            self._output_hook.remove()
            self._output_hook = None
        try:
            output.retain_grad()
            self._saved_output_buffer = output
        except Exception:
            self._saved_output_buffer = None

        def _save(grad: torch.Tensor) -> None:
            with torch.no_grad():
                if self._oldfast_score_requested() and self._use_channel_virtual_basis():
                    self._compute_oldfast_results(grad)
                    self._saved_output_buffer = None
                    return
                self.saved_neuron_gradients = grad.detach().to(dtype=torch.float32)
                self._saved_output_buffer = None

        self._output_hook = output.register_hook(_save)

    def _update_gpu_stats(self, phi_rows: torch.Tensor, target_rows: Optional[torch.Tensor] = None) -> None:
        if phi_rows is None:
            return
        if self.gpu_stats is None:
            self.gpu_stats = GPUStats(device=phi_rows.device)
        self.gpu_stats.update(phi_rows, target_rows)

    def iter_neuron_gradient_chunks(
        self,
        *,
        chunk_size: Optional[int] = None,
        target_device: Optional[torch.device] = None,
        detach: bool = True,
    ) -> Iterable[Tuple[int, int, torch.Tensor]]:
        """
        Yield per-sample neuron gradient slices and clear the cached tensor afterward.

        Args:
            chunk_size: Optional max number of samples per chunk.
            target_device: Optional device to move each chunk onto.
            detach: Whether to detach the chunk from the autograd graph.
        """
        grads = getattr(self, "saved_neuron_gradients", None)
        if grads is None:
            return

        if chunk_size is None or chunk_size <= 0:
            chunk_size = grads.size(0)

        device = target_device if target_device is not None else grads.device

        total = grads.size(0)
        with torch.no_grad():
            for start in range(0, total, chunk_size):
                end = min(start + chunk_size, total)
                slice_tensor = grads.narrow(0, start, end - start)
                if detach:
                    slice_tensor = slice_tensor.detach()
                if device is not None and slice_tensor.device != device:
                    slice_tensor = slice_tensor.to(device=device, non_blocking=True)
                else:
                    slice_tensor = slice_tensor.contiguous()
                yield (start, end, slice_tensor)
        # Clear cached gradients once the iterator is exhausted.
        self.saved_neuron_gradients = None

    def iter_weight_gradient_chunks(
        self,
        *,
        chunk_size: Optional[int] = None,
        target_device: Optional[torch.device] = None,
        detach: bool = True,
    ) -> Iterable[Tuple[int, int, torch.Tensor]]:
        gradients = getattr(self, "saved_weight_gradients", None)
        if gradients is None:
            return

        if chunk_size is None or chunk_size <= 0:
            chunk_size = gradients.size(0)
        device = target_device if target_device is not None else gradients.device

        total = gradients.size(0)
        with torch.no_grad():
            for start in range(0, total, chunk_size):
                end = min(start + chunk_size, total)
                chunk = gradients.narrow(0, start, end - start)
                if detach:
                    chunk = chunk.detach()
                if device is not None and chunk.device != device:
                    chunk = chunk.to(device=device, non_blocking=True)
                else:
                    chunk = chunk.contiguous()
                yield (start, end, chunk)
        self.saved_weight_gradients = None

    def _setup_replay_provider(self, inputs: torch.Tensor) -> None:
        provider = self._pending_replay_provider or current_replay_provider()
        self._pending_replay_provider = None

        if provider is not None:
            expected = getattr(provider, "batch_size", None)
            if expected is not None and expected != inputs.size(0):
                if expected > inputs.size(0):
                    if not self._warned_replay_batch_mismatch:
                        print(
                            "[HookedConv2d] Replay provider batch_size mismatch: "
                            f"expected {expected}, got {inputs.size(0)}. "
                            "Falling back to tensor-backed replay."
                        )
                        self._warned_replay_batch_mismatch = True
                    provider = None
                else:
                    raise ValueError(
                        "[HookedConv2d] Replay provider batch_size mismatch: "
                        f"expected {expected}, got {inputs.size(0)}"
                    )
            elif not _replay_matches_tensor(provider, inputs):
                provider = None

        override_device = getattr(self, "stream_flush_replay_device", None)
        if provider is None:
            storage_device = REPLAY_STORAGE_DEVICE
            if getattr(self, "stream_flush_enabled", False) and override_device is not None:
                storage_device = override_device
            provider = TensorReplayProvider(inputs.detach(), storage_device=storage_device)
        else:
            if getattr(self, "stream_flush_enabled", False) and override_device is not None:
                storage = getattr(provider, "_storage", None)
                try:
                    if storage is not None and storage.device != override_device:
                        provider = TensorReplayProvider(storage, storage_device=override_device)
                except Exception:
                    logger.debug(
                        "[stream-flush] Unable to migrate replay cache for %s",
                        getattr(self, "frontier_layer_name", "unknown"),
                    )

        self._active_replay_provider = provider
        self.hook_in = True

    def flush_results(self) -> None:
        if self._streaming_enabled():
            if not getattr(self, "_stream_results_ready", False):
                try:
                    self._finalize_streaming_results()
                except Exception:
                    logger.exception(
                        "[stream] Conv layer %s flush finalize failed",
                        getattr(self, "frontier_layer_name", "unknown"),
                    )
                    return
            self._stream_results_ready = True
            return
        if not getattr(self, "stream_flush_enabled", False):
            return
        if getattr(self, "_stream_results_ready", False):
            return
        if self.results is not None and not getattr(self, "compute_scores", False):
            self._stream_results_ready = True
            return

        # Require gradients and replay data to be available before attempting flush.
        if self.saved_neuron_gradients is None and self.saved_integrated_multipliers is None:
            return
        replay_source = getattr(self, "_active_replay_provider", None)
        if replay_source is None and self.pre_sum_output is None:
            return

        cat_list = getattr(self, "in_cat_indices_list", None)
        if isinstance(cat_list, torch.Tensor):
            cat_indices = cat_list.tolist()
        elif isinstance(cat_list, list):
            cat_indices = cat_list
        elif cat_list is None:
            cat_indices = []
        else:
            try:
                cat_indices = list(cat_list)
            except TypeError:
                cat_indices = []

        try:
            results = self.compute_kernel_scores(cat_indices, inputs=None)
        except Exception as exc:
            logger.debug(
                "[stream-flush] Falling back to legacy aggregation for %s: %s",
                getattr(self, "frontier_layer_name", "unknown"),
                exc,
            )
            return

        if not isinstance(results, tuple):
            results = tuple(results)
        self.results = self._detach_structure_cpu(results)

        provider = getattr(self, "_active_replay_provider", None)
        if provider is not None:
            free = getattr(provider, "free", None)
            if callable(free):
                try:
                    free()
                except Exception:
                    logger.debug(
                        "[stream-flush] Unable to free replay cache for %s",
                        getattr(self, "frontier_layer_name", "unknown"),
                    )
        self._active_replay_provider = None

        context = getattr(self, "_stream_context", None)
        if context is not None and getattr(context, "enabled", False):
            layer_name = getattr(self, "frontier_layer_name", None)
            if layer_name:
                context.submit(layer_name, self.results)

        self._stream_results_ready = True

    def _teardown_output_hook(self) -> None:
        if self._output_hook is not None:
            self._output_hook.remove()
            self._output_hook = None

    def get_boundary_inputs(self, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        source = self._cached_inputs_for_hvp
        if source is None:
            return None
        target_device = device or self.weight.device
        dtype = torch.float32
        if isinstance(source, torch.Tensor):
            tensor = source
            if tensor.device != target_device or tensor.dtype != dtype:
                tensor = tensor.to(device=target_device, dtype=dtype, non_blocking=True)
            return tensor
        getter = getattr(source, "get", None)
        if callable(getter):
            batch_size = getattr(source, "batch_size", None)
            if batch_size is None:
                return None
            return getter(0, batch_size, device=target_device, dtype=dtype)
        return None
    
    def get_inputs_for_hvp(self, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        return self.get_boundary_inputs(device=device)

    def _update_running_stats(self, activations: torch.Tensor) -> None:
        with torch.no_grad():
            batch_sum = activations.detach().sum(dim=0)
            if self._pending_activation_sum is None:
                self._pending_activation_sum = batch_sum
            else:
                self._pending_activation_sum.add_(batch_sum)
            self._pending_activation_count += activations.shape[0]

    def _commit_running_activation_stats(self) -> None:
        pending = self._pending_activation_sum
        if pending is None:
            return
        with torch.no_grad():
            batch_sum_cpu = pending.detach().to(device="cpu")
            if self.running_activation_sum is None:
                self.running_activation_sum = batch_sum_cpu
            else:
                self.running_activation_sum.add_(batch_sum_cpu)
            self.running_activation_count += self._pending_activation_count
        self._pending_activation_sum = None
        self._pending_activation_count = 0

    def _finalize_frontier_stats(self) -> None:
        if self.gpu_stats is None:
            self.frontier_stats = None
            return
        summary = self.gpu_stats.export_analysis()
        if not isinstance(summary, dict) or not summary:
            self.frontier_stats = None
            return
        global_stats = summary.get("global")
        if isinstance(global_stats, dict) and global_stats:
            self.frontier_stats = global_stats
        else:
            self.frontier_stats = None

    def _effective_weight(
        self,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        weight = self.weight
        if dtype is not None and weight.dtype != dtype:
            weight = weight.to(dtype=dtype)
        if device is not None and weight.device != device:
            weight = weight.to(device=device)
        mask = getattr(self, "mask", None)
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise TypeError("mask must be a torch.Tensor when provided.")
            mask_tensor = mask.to(device=weight.device, dtype=weight.dtype)
            if mask_tensor.dim() == 1:
                expected = self.out_features * (self.in_features // self.groups)
                if mask_tensor.numel() == expected:
                    mask_tensor = mask_tensor.view(self.out_features, -1)
                mask_tensor = mask_tensor.unsqueeze(-1).unsqueeze(-1)
            elif mask_tensor.dim() == 2:
                mask_tensor = mask_tensor.unsqueeze(-1).unsqueeze(-1)
            elif mask_tensor.dim() == 3:
                mask_tensor = mask_tensor.unsqueeze(-1)
            elif mask_tensor.dim() > 4:
                raise ValueError("mask tensor dimensionality is unsupported.")
            weight = weight * mask_tensor
        return weight

    def _effective_bias(
        self,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Optional[torch.Tensor]:
        bias = self.bias
        if bias is None:
            return None
        if dtype is not None and bias.dtype != dtype:
            bias = bias.to(dtype=dtype)
        if device is not None and bias.device != device:
            bias = bias.to(device=device)
        bias_mask = getattr(self, "bias_mask", None)
        if bias_mask is not None:
            bias = bias * bias_mask.to(device=bias.device, dtype=bias.dtype)
        return bias

    def _setup_replay_provider(self, inputs: torch.Tensor) -> None:
        provider = self._pending_replay_provider or current_replay_provider()
        self._pending_replay_provider = None

        if provider is not None:
            expected = getattr(provider, "batch_size", None)
            if expected is not None and expected != inputs.size(0):
                if expected > inputs.size(0):
                    if not self._warned_replay_batch_mismatch:
                        print(
                            "[HookedConv2d] Replay provider batch_size mismatch: "
                            f"expected {expected}, got {inputs.size(0)}. "
                            "Falling back to tensor-backed replay."
                        )
                        self._warned_replay_batch_mismatch = True
                    provider = None
                else:
                    raise ValueError(
                        "[HookedConv2d] Replay provider batch_size mismatch: "
                        f"expected {expected}, got {inputs.size(0)}"
                    )
            elif not _replay_matches_tensor(provider, inputs):
                provider = None

        if provider is None:
            provider = TensorReplayProvider(
                inputs.detach(), storage_device=REPLAY_STORAGE_DEVICE
            )

        self._active_replay_provider = provider
        self.hook_in = True

    def _teardown_output_hook(self) -> None:
        if self._output_hook is not None:
            self._output_hook.remove()
            self._output_hook = None

    def _reset_after_scoring(self) -> None:
        self._teardown_output_hook()
        if self._active_replay_provider is not None:
            free_fn = getattr(self._active_replay_provider, "free", None)
            if callable(free_fn):
                try:
                    free_fn()
                except Exception:
                    logger.debug("[frontier_conv] replay provider free() failed", exc_info=True)
        self._active_replay_provider = None
        self._pending_replay_provider = None
        self.saved_neuron_gradients = None
        self.saved_integrated_multipliers = None
        self._pre_bias_output = None
        self._saved_output_buffer = None
        self._added_out = None
        self._oldfast_inputs = None
        if not self.retain_z_cache_until_reset:
            self.z_cache = None
        self.full_acts = None
        self.pre_softmax_vector = None
        self.cross_entropy_loss = None
        self.target_unit_acts = None
        self.label = None
        self._commit_running_activation_stats()
        if torch.is_tensor(self.running_activation_sum):
            try:
                self.running_activation_sum = self.running_activation_sum.detach().cpu()
            except Exception:
                self.running_activation_sum = None
        if self.running_activation_sum is None:
            self.running_activation_count = 0
        self._pending_activation_sum = None
        self._pending_activation_count = 0
        self._cached_inputs_for_hvp = None
        if isinstance(getattr(self, "_stream_cache", None), dict):
            self._stream_cache.clear()
        self.hook_in = False

    def get_boundary_inputs(self, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        source = self._cached_inputs_for_hvp
        if source is None:
            return None
        target_device = device or self.weight.device
        dtype = torch.float32
        if isinstance(source, torch.Tensor):
            tensor = source
            if tensor.device != target_device or tensor.dtype != dtype:
                tensor = tensor.to(device=target_device, dtype=dtype, non_blocking=True)
            return tensor
        getter = getattr(source, "get", None)
        if callable(getter):
            batch_size = getattr(source, "batch_size", None)
            if batch_size is None:
                return None
            return getter(0, batch_size, device=target_device, dtype=dtype)
        return None
    
    def get_inputs_for_hvp(self, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        return self.get_boundary_inputs(device=device)

    # ------------------------------------------------------------------
    # Public API mirroring the legacy implementation
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._effective_weight(device=x.device, dtype=self.weight.dtype)
        bias = self._effective_bias(device=x.device, dtype=self.weight.dtype)
        output = F.conv2d(
            x,
            weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        capture_kernels = bool(self.calc_kernels or getattr(self, "compute_scores", False))
        preserve_capture_state = bool(getattr(self, "_preserve_score_capture_state", False))
        if capture_kernels and not output.requires_grad:
            output = output.detach().requires_grad_(True)
        self.z_cache = output.detach()

        # Track running statistics for mean activations
        self._update_running_stats(output)

        forward_output = output
        if capture_kernels:
            if self.oldfast_mode and self._use_channel_virtual_basis():
                self._oldfast_inputs = x.detach()
                self._active_replay_provider = TensorReplayProvider(
                    x.detach(), storage_device=x.device
                )
                self._pending_replay_provider = None
            else:
                self._setup_replay_provider(x)
                self._oldfast_inputs = None
            self._register_output_hook(output)
            # Keep the scored route tensor stable even when the next module is an
            # in-place nonlinearity, as in torchvision AlexNet/VGG.  Downstream
            # computation sees identical values through the clone, while scalar
            # hooks and target-batched VJPs both differentiate with respect to the
            # pre-inplace Conv output.
            forward_output = output.clone()
        else:
            if not preserve_capture_state:
                self._pending_replay_provider = None
                self._active_replay_provider = None
                self._teardown_output_hook()
                self._oldfast_inputs = None

        self.hook_in = capture_kernels

        return forward_output


    def collect_integrated_multipliers(self) -> Optional[torch.Tensor]:
        grad = getattr(self, "saved_neuron_gradients", None)
        if grad is None:
            return None
        result = grad.detach().to(dtype=torch.float32)
        self.saved_integrated_multipliers = result
        return result

    def before_backward(self) -> None:
        if OLDFAST_TRACE and getattr(self, "oldfast_mode", False):
            logger.warning(
                "[oldfast-trace:before-backward] layer=%s id=%s clearing_results=%s clearing_cache=%s",
                getattr(self, "frontier_layer_name", "unknown"),
                id(self),
                self.results is not None,
                getattr(self, "_oldfast_results_cache", None) is not None,
            )
        self.results = None
        self._oldfast_results_cache = None
        self.analysis_payload = None
        self._stream_results_ready = False
        if self._streaming_enabled():
            self._initialize_reducers()

    def after_backward(self) -> None:
        self._commit_running_activation_stats()
        if self._streaming_enabled():
            try:
                self._finalize_streaming_results()
            except Exception:
                logger.exception(
                    "[stream] Conv layer %s finalize failed",
                    getattr(self, "frontier_layer_name", "unknown"),
                )
        elif getattr(self, "stream_flush_enabled", False):
            try:
                self.flush_results()
            except Exception:
                logger.exception(
                    "[stream-flush] Conv layer %s flush failed",
                    getattr(self, "frontier_layer_name", "unknown"),
                )
        self.compute_scores = False

    def prepare_for_scoring(
        self,
        *,
        cross_entropy_loss,
        full_acts,
        pre_softmax_vector,
        in_cat_indices_list,
        should_score: bool,
    ) -> None:
        if OLDFAST_TRACE and getattr(self, "oldfast_mode", False):
            logger.warning(
                "[oldfast-trace:prepare] layer=%s id=%s should_score=%s clearing_results=%s clearing_cache=%s",
                getattr(self, "frontier_layer_name", "unknown"),
                id(self),
                should_score,
                self.results is not None,
                getattr(self, "_oldfast_results_cache", None) is not None,
            )
        self.calc_kernels = bool(should_score)
        self.compute_scores = bool(should_score)
        self.results = None
        self._oldfast_results_cache = None
        self.frontier_stats = None
        self.frontier_stats = None
        self.analysis_payload = None
        self._stream_results_ready = False
        self.in_cat_indices_list = in_cat_indices_list

        if self._streaming_enabled():
            self._reset_stream_state()
            self._set_stream_category_indices(in_cat_indices_list)
            self._capture_stream_cache(
                full_acts=full_acts,
                pre_softmax_vector=pre_softmax_vector,
                cross_entropy_loss=cross_entropy_loss,
            )
            self.cross_entropy_loss = self._stream_cache.get("cross_entropy")
            self.full_acts = None
            self.pre_softmax_vector = None
            cached_target = self._stream_cache.get("target_unit_acts")
            if cached_target is not None:
                self.target_unit_acts = cached_target
            cached_labels = self._stream_cache.get("labels")
            if cached_labels is not None:
                self.label = cached_labels
        else:
            self.cross_entropy_loss = cross_entropy_loss
            self.full_acts = full_acts
            self.pre_softmax_vector = pre_softmax_vector
        self.in_cat_indices_list = in_cat_indices_list

    def pop_running_mean_activation(self) -> Optional[torch.Tensor]:
        self._commit_running_activation_stats()
        if self.running_activation_sum is None or self.running_activation_count == 0:
            return None
        mean = (self.running_activation_sum / float(self.running_activation_count)).detach().cpu()
        self.running_activation_sum = None
        self.running_activation_count = 0
        return mean

    def set_compute_neuron_metrics(self, enabled: bool) -> None:
        self.compute_neuron_metrics = bool(enabled)

    def set_replay_provider(self, provider: Optional[object]) -> None:
        self._pending_replay_provider = provider

    def capture_gradients_hook_neurons(self, grad: torch.Tensor) -> None:
        with torch.no_grad():
            self.saved_neuron_gradients = grad.detach().to(dtype=torch.float32)
            self._saved_output_buffer = None

    def _compute_conv_chunk_bie(
        self,
        inputs_chunk: torch.Tensor,
        preactivations_chunk: Optional[torch.Tensor],
        grads_chunk: torch.Tensor,
        weight_matrix: torch.Tensor,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        if not self._effective_collect_bie():
            return None, None, None, None, None, None
        if preactivations_chunk is None:
            self._mark_chunk_capture_failed()
            return None, None, None, None, None, None
        if self.groups != 1:
            self._mark_chunk_capture_failed()
            return None, None, None, None, None, None

        try:
            cols = F.unfold(
                inputs_chunk.detach(),
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
            z_flat = preactivations_chunk.detach().to(dtype=torch.float32).flatten(2)
            delta_flat = grads_chunk.detach().to(dtype=torch.float32).flatten(2)
            if cols.size(2) != z_flat.size(2) or delta_flat.size(2) != z_flat.size(2):
                self._mark_chunk_capture_failed()
                return None, None, None, None

            weight_flat = weight_matrix.detach().to(dtype=torch.float32).view(weight_matrix.size(0), -1)
            tile = 1024
            base_sum = torch.zeros_like(weight_flat)
            base_abs_sum = torch.zeros_like(weight_flat)
            boundary_sum = torch.zeros_like(weight_flat)
            boundary_abs_sum = torch.zeros_like(weight_flat)
            base_sample_totals = torch.zeros(z_flat.size(0), device=weight_flat.device, dtype=torch.float32)
            boundary_sample_totals = torch.zeros_like(base_sample_totals)
            on_mask = z_flat > 0

            for i0 in range(0, weight_flat.size(1), tile):
                i1 = min(weight_flat.size(1), i0 + tile)
                Ai = cols[:, i0:i1, :]  # [B, Ti, L]
                Wi = weight_flat[:, i0:i1]  # [O, Ti]
                base_plane = -Ai.unsqueeze(1) * Wi.unsqueeze(0).unsqueeze(-1)  # [B, O, Ti, L]
                hold_mask = (z_flat.unsqueeze(2) - (-base_plane)) > 0
                active_mask = (on_mask.unsqueeze(2) & hold_mask).to(base_plane.dtype)
                bie_state = base_plane * active_mask
                contrib = torch.einsum("bjl,bjil->ji", delta_flat, bie_state)
                base_sum[:, i0:i1] += contrib
                base_abs_sum[:, i0:i1] += contrib.abs()
                base_sample_totals += torch.einsum("bjl,bjil->b", delta_flat, bie_state)

                overshoot = (
                    on_mask.unsqueeze(2)
                    & ((z_flat.unsqueeze(2) - Ai.unsqueeze(1) * Wi.unsqueeze(0).unsqueeze(-1)) <= 0)
                ).to(base_plane.dtype)
                corr_state = (-z_flat.unsqueeze(2)) * overshoot
                corr_contrib = torch.einsum("bjl,bjil->ji", delta_flat, corr_state)
                boundary_sum[:, i0:i1] += corr_contrib
                boundary_abs_sum[:, i0:i1] += corr_contrib.abs()
                boundary_sample_totals += torch.einsum("bjl,bjil->b", delta_flat, corr_state)

            base_sum = base_sum.view_as(weight_matrix)
            base_abs_sum = base_abs_sum.view_as(weight_matrix)
            boundary_sum = boundary_sum.view_as(weight_matrix)
            boundary_abs_sum = boundary_abs_sum.view_as(weight_matrix)
            return (
                base_sum,
                base_abs_sum,
                boundary_sum,
                boundary_abs_sum,
                base_sample_totals.detach(),
                boundary_sample_totals.detach(),
            )
        except Exception:
            self._mark_chunk_capture_failed()
            logger.debug(
                "[bie-conv] Failed to compute chunk corrections for layer=%s",
                getattr(self, "frontier_layer_name", "unknown"),
                exc_info=True,
            )
            return None, None, None, None, None, None

    def _reduce_phi_chunk(
        self,
        phi_chunk: torch.Tensor,
        grad_weight_chunk: torch.Tensor,
        grads_chunk: torch.Tensor,
        outputs_chunk: Optional[torch.Tensor],
        *,
        inputs_chunk: Optional[torch.Tensor],
        weight_matrix: Optional[torch.Tensor],
        full_acts_chunk: Optional[torch.Tensor],
        labels_chunk: Optional[torch.Tensor],
        pre_softmax_chunk: Optional[torch.Tensor],
        target_unit_acts_chunk: Optional[torch.Tensor],
        cross_entropy_loss_chunk: Optional[torch.Tensor],
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Reduce a per-sample phi tensor (and its corresponding grad-weight tensor)
        into the summary statistics required by the scoring pipeline.
        """
        scale_value = self.scale if self.scale is not None else 1.0
        if isinstance(scale_value, torch.Tensor):
            scale_value = scale_value.item()
        scale = float(scale_value)
        if abs(scale) < 1e-12:
            scale = 1.0

        phi_scaled = phi_chunk / scale
        grad_weight_scaled = grad_weight_chunk / scale

        kernel_scores_sum = phi_scaled.sum(dim=0)
        activating_scores_sum = torch.relu(phi_scaled).sum(dim=0)
        inhibiting_scores_sum = (-torch.relu(-phi_scaled)).sum(dim=0)
        abs_scores_sum = phi_scaled.abs().sum(dim=0)
        scores_signed_acts_sum = (phi_scaled * grad_weight_chunk.sign()).sum(dim=0)
        gradients_sum = grad_weight_scaled.sum(dim=0)
        abs_gradients_sum = grad_weight_scaled.abs().sum(dim=0)

        compute_loss_delta = (
            self.score_post_af
            and COMPUTE_LOSS_DELTA
            and full_acts_chunk is not None
            and labels_chunk is not None
        )
        need_kernel_rows = bool(
            getattr(self, "capture_sample_sums", False)
            or target_unit_acts_chunk is not None
            or bool(getattr(self, "capture_bitpack", COMPUTE_BITPACK))
            or compute_loss_delta
        )
        kernel_scores_no_spatial: Optional[torch.Tensor] = None
        sample_sum_matrix: Optional[torch.Tensor] = None
        sample_sum_cpu: Optional[torch.Tensor] = None
        sample_abs_cpu: Optional[torch.Tensor] = None
        if need_kernel_rows:
            kernel_scores_no_spatial = phi_scaled.sum(dim=(-1, -2))
            if getattr(self, "capture_sample_sums", False) or target_unit_acts_chunk is not None:
                sample_sum_matrix = kernel_scores_no_spatial.sum(dim=2)
            if sample_sum_matrix is not None and (
                getattr(self, "capture_sample_sums", False) or target_unit_acts_chunk is not None
            ):
                sample_sum_cpu = sample_sum_matrix.detach().to(device="cpu", dtype=torch.float32)
                sample_abs_cpu = sample_sum_cpu.abs()

        loss_change_sum: Optional[torch.Tensor] = None
        abs_loss_change_sum: Optional[torch.Tensor] = None

        if compute_loss_delta:
            if kernel_scores_no_spatial is None:
                raise RuntimeError("Conv loss-delta requested kernel row scores but none were computed.")
            flat_scores = kernel_scores_no_spatial.reshape(kernel_scores_no_spatial.size(0), -1)
            loss_diff = compute_loss_difference_triton_optimized(
                flattened_base_scores=flat_scores,
                full_acts=full_acts_chunk,
                labels=labels_chunk,
                target_unit=self.target_unit if self.target_unit is not None else 0,
                original_loss=cross_entropy_loss_chunk,
                out_dtype=torch.float32,
            )
            loss_diff = loss_diff.view(kernel_scores_no_spatial.shape)
            loss_change_sum = loss_diff.sum(dim=0)
            abs_loss_change_sum = loss_diff.abs().sum(dim=0)

        neuron_scores_sum: Optional[torch.Tensor] = None
        abs_neuron_scores_sum: Optional[torch.Tensor] = None
        neuron_scores_sq_sum: Optional[torch.Tensor] = None
        abs_neuron_scores_sq_sum: Optional[torch.Tensor] = None
        product_scores_sum: Optional[torch.Tensor] = None
        abs_product_scores_sum: Optional[torch.Tensor] = None

        if outputs_chunk is not None:
            neuron_scores_per_image = (outputs_chunk * grads_chunk).sum(dim=(-2, -1)) / scale
            abs_neuron_scores_per_image = neuron_scores_per_image.abs()
            neuron_scores_sum = neuron_scores_per_image.sum(dim=0)
            abs_neuron_scores_sum = abs_neuron_scores_per_image.sum(dim=0)
            neuron_scores_sq_sum = (neuron_scores_per_image ** 2).sum(dim=0)
            abs_neuron_scores_sq_sum = (abs_neuron_scores_per_image ** 2).sum(dim=0)

            view_scores = neuron_scores_per_image.view(neuron_scores_per_image.size(0), neuron_scores_per_image.size(1), 1, 1, 1)
            view_abs_scores = abs_neuron_scores_per_image.view(abs_neuron_scores_per_image.size(0), abs_neuron_scores_per_image.size(1), 1, 1, 1)
            product_scores_sum = (phi_scaled * view_scores).sum(dim=0)
            abs_product_scores_sum = (phi_scaled.abs() * view_abs_scores).sum(dim=0)

        corr = torch.tensor(0.0, device=phi_chunk.device)
        if target_unit_acts_chunk is not None and sample_sum_matrix is not None:
            target_unit_acts_flat = target_unit_acts_chunk.to(dtype=torch.float32, device=phi_chunk.device)
            totals = kernel_scores_no_spatial.sum(dim=(1, 2))
            corr = safe_pearson_corr(totals, target_unit_acts_flat)

        bitpacked: List[torch.Tensor] = []
        if bool(getattr(self, "capture_bitpack", COMPUTE_BITPACK)) and kernel_scores_no_spatial is not None:
            for sample_matrix in kernel_scores_no_spatial:
                packed_rows = pack_boolean_rows(sample_matrix.abs().reshape(sample_matrix.size(0), -1) > 0)
                for row in packed_rows:
                    bitpacked.append(row.cpu())

        bie_base_sum: Optional[torch.Tensor] = None
        bie_base_abs_sum: Optional[torch.Tensor] = None
        bie_boundary_sum: Optional[torch.Tensor] = None
        bie_boundary_abs_sum: Optional[torch.Tensor] = None
        bie_base_samples: Optional[torch.Tensor] = None
        bie_boundary_samples: Optional[torch.Tensor] = None
        if (
            self._effective_collect_bie()
            and inputs_chunk is not None
            and weight_matrix is not None
            and outputs_chunk is not None
        ):
            (
                bie_base_sum,
                bie_base_abs_sum,
                bie_boundary_sum,
                bie_boundary_abs_sum,
                bie_base_samples,
                bie_boundary_samples,
            ) = self._compute_conv_chunk_bie(
                inputs_chunk,
                outputs_chunk,
                grads_chunk,
                weight_matrix,
            )
            if torch.is_tensor(bie_base_sum):
                bie_base_sum = bie_base_sum / scale
            if torch.is_tensor(bie_base_abs_sum):
                bie_base_abs_sum = bie_base_abs_sum / scale
            if torch.is_tensor(bie_boundary_sum):
                bie_boundary_sum = bie_boundary_sum / scale
            if torch.is_tensor(bie_boundary_abs_sum):
                bie_boundary_abs_sum = bie_boundary_abs_sum / scale
            if torch.is_tensor(bie_base_samples):
                bie_base_samples = bie_base_samples.detach().to(device="cpu", dtype=torch.float32) / scale
            if torch.is_tensor(bie_boundary_samples):
                bie_boundary_samples = bie_boundary_samples.detach().to(device="cpu", dtype=torch.float32) / scale

        def _to_cpu(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if torch.is_tensor(t):
                return t.detach().to(device="cpu")
            return t

        return {
            "kernel_scores_sum": _to_cpu(kernel_scores_sum),
            "activating_scores_sum": _to_cpu(activating_scores_sum),
            "inhibiting_scores_sum": _to_cpu(inhibiting_scores_sum),
            "scores_signed_acts_sum": _to_cpu(scores_signed_acts_sum),
            "abs_scores_sum": _to_cpu(abs_scores_sum),
            "gradients_sum": _to_cpu(gradients_sum),
            "abs_gradients_sum": _to_cpu(abs_gradients_sum),
            "loss_change_sum": _to_cpu(loss_change_sum),
            "abs_loss_change_sum": _to_cpu(abs_loss_change_sum),
            "product_scores_sum": _to_cpu(product_scores_sum),
            "abs_product_scores_sum": _to_cpu(abs_product_scores_sum),
            "neuron_scores_sum": _to_cpu(neuron_scores_sum),
            "abs_neuron_scores_sum": _to_cpu(abs_neuron_scores_sum),
            "neuron_scores_sq_sum": _to_cpu(neuron_scores_sq_sum),
            "abs_neuron_scores_sq_sum": _to_cpu(abs_neuron_scores_sq_sum),
            "corr": _to_cpu(corr),
            "bitpacked": bitpacked,
            "sample_sums": sample_sum_cpu,
            "sample_abs_sums": sample_abs_cpu,
            "bie_base_sum": _to_cpu(bie_base_sum),
            "bie_base_abs_sum": _to_cpu(bie_base_abs_sum),
            "bie_boundary_sum": _to_cpu(bie_boundary_sum),
            "bie_boundary_abs_sum": _to_cpu(bie_boundary_abs_sum),
            "bie_base_sample_totals": bie_base_samples,
            "bie_boundary_sample_totals": bie_boundary_samples,
        }

    def _reduce_channel_virtual_chunk(
        self,
        preadd_chunk: torch.Tensor,
        grads_chunk: torch.Tensor,
        outputs_chunk: Optional[torch.Tensor],
        *,
        target_unit_acts_chunk: Optional[torch.Tensor],
        return_sample_sums: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        chunk_stats = _conv_channel_virtual_reduce_exact_chunk(
            preadd_chunk,
            grads_chunk,
            scale=self.scale if self.scale is not None else 1.0,
            outputs_chunk=outputs_chunk,
            compute_neuron_metrics=self.compute_neuron_metrics,
            abs_aggregation_mode=self.abs_aggregation_mode,
            need_per_sample_scores=bool(
                return_sample_sums
                or getattr(self, "capture_sample_sums", False)
                or target_unit_acts_chunk is not None
                or getattr(self, "capture_bitpack", COMPUTE_BITPACK)
                or self.compute_neuron_metrics
            ),
        )
        per_sample_scores = chunk_stats["per_sample_scores"]
        sample_sum_matrix = chunk_stats["sample_sums"]
        sample_sum_cpu = None
        sample_abs_cpu = None
        if bool(return_sample_sums or getattr(self, "capture_sample_sums", False) or target_unit_acts_chunk is not None):
            sample_sum_cpu = sample_sum_matrix.detach().to(device="cpu", dtype=torch.float32)
            sample_abs_cpu = sample_sum_cpu.abs()

        corr = torch.tensor(0.0, device=preadd_chunk.device)
        if target_unit_acts_chunk is not None:
            target_unit_flat = target_unit_acts_chunk.to(dtype=torch.float32, device=preadd_chunk.device)
            totals = sample_sum_matrix.sum(dim=1)
            corr = safe_pearson_corr(totals, target_unit_flat)

        bitpacked: List[torch.Tensor] = []
        if bool(getattr(self, "capture_bitpack", COMPUTE_BITPACK)):
            for sample_matrix in per_sample_scores:
                packed_rows = pack_boolean_rows(sample_matrix.ne(0).reshape(sample_matrix.size(0), -1))
                for row in packed_rows:
                    bitpacked.append(row.cpu())

        def _to_cpu(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if torch.is_tensor(t):
                return t.detach().to(device="cpu")
            return t

        return {
            "kernel_scores_sum": _to_cpu(chunk_stats["kernel_scores_sum"]),
            "activating_scores_sum": _to_cpu(chunk_stats["activating_scores_sum"]),
            "inhibiting_scores_sum": _to_cpu(chunk_stats["inhibiting_scores_sum"]),
            "scores_signed_acts_sum": _to_cpu(chunk_stats["scores_signed_acts_sum"]),
            "abs_scores_sum": _to_cpu(chunk_stats["abs_scores_sum"]),
            "gradients_sum": _to_cpu(chunk_stats["gradients_sum"]),
            "abs_gradients_sum": _to_cpu(chunk_stats["abs_gradients_sum"]),
            "loss_change_sum": None,
            "abs_loss_change_sum": None,
            "product_scores_sum": _to_cpu(chunk_stats["product_scores_sum"]),
            "abs_product_scores_sum": _to_cpu(chunk_stats["abs_product_scores_sum"]),
            "neuron_scores_sum": _to_cpu(chunk_stats["neuron_scores_sum"]),
            "abs_neuron_scores_sum": _to_cpu(chunk_stats["abs_neuron_scores_sum"]),
            "neuron_scores_sq_sum": _to_cpu(chunk_stats["neuron_scores_sq_sum"]),
            "abs_neuron_scores_sq_sum": _to_cpu(chunk_stats["abs_neuron_scores_sq_sum"]),
            "corr": _to_cpu(corr),
            "bitpacked": bitpacked,
            "sample_sums": sample_sum_cpu,
            "sample_abs_sums": sample_abs_cpu,
            "bie_base_sum": None,
            "bie_base_abs_sum": None,
            "bie_boundary_sum": None,
            "bie_boundary_abs_sum": None,
        }

    def _run_streaming_reduction(self) -> None:
        if not self._streaming_enabled():
            return
        if self._reducers is None:
            self._initialize_reducers()
        if self._stream_processed:
            return

        saved_grads = self.saved_integrated_multipliers
        if saved_grads is None:
            saved_grads = self.saved_neuron_gradients
        if saved_grads is None:
            raise RuntimeError("Saved neuron gradients are unavailable for streaming.")
        effective_collect_bie = self._effective_collect_bie()
        if effective_collect_bie:
            self._reset_chunk_capture()

        inputs_source = self._active_replay_provider or current_replay_provider()
        if inputs_source is None:
            raise RuntimeError("Replay source unavailable for streaming reduction.")

        device = saved_grads.device
        weight = self._effective_weight(device=device, dtype=torch.float32)
        bias = self._effective_bias(device=device, dtype=torch.float32)
        grads = saved_grads.to(dtype=torch.float32, device=device)

        scale_value = self.scale if self.scale is not None else 1.0
        if isinstance(scale_value, torch.Tensor):
            scale_value = scale_value.item()
        scale = float(scale_value)
        if abs(scale) < 1e-12:
            scale = 1.0

        batch_size = _replay_batch_size(inputs_source, grads.size(0))
        use_channel_virtual = self._use_channel_virtual_basis()
        spatial_elements = weight.shape[2] * weight.shape[3] if weight.dim() == 4 else 1
        base_chunk = choose_chunk_size(
            weight.size(0),
            weight.size(1),
            spatial_elements,
            target_elements=DEFAULT_TARGET_CHUNK_ELEMENTS,
            max_samples=MAX_CHUNK_SAMPLES,
        )
        chunk_size = max(1, min(base_chunk, batch_size if batch_size > 0 else base_chunk))
        override_size = getattr(self, "streaming_chunk_size", None)
        if override_size is not None:
            override_int = max(1, int(override_size))
            if batch_size > 0:
                override_int = min(override_int, batch_size)
            chunk_size = min(chunk_size, override_int)

        self._active_stream_chunk_size = chunk_size
        self._last_stream_chunk_size = 0
        stats_callback = getattr(self, "_update_gpu_stats", None) if self.collect_stats else None
        validate_frontier = VALIDATE_FRONTIER
        self._stream_sample_sums = []
        self._stream_sample_abs_sums = []
        debug_layer_name = (
            getattr(self, "frontier_layer_name", None)
            if getattr(self, "enable_layer_corr_logging", False)
            else None
        )
        need_target_rows = bool(
            callable(stats_callback)
            or debug_layer_name
            or getattr(self, "collect_score_correlation", True)
        )
        request_sample_sums = bool(self.capture_sample_sums or need_target_rows)

        weight_shape = weight.shape
        preadd_weight = None
        in_per_group = 0
        out_per_group = 0
        if use_channel_virtual:
            preadd_weight, in_per_group, out_per_group = _build_conv_channel_virtual_weight(
                weight,
                groups=self.groups,
            )
            preadd_weight = preadd_weight.to(device=device, dtype=torch.float32)
            frontier_only_channel_virtual = bool(
                not self._effective_collect_bie()
                and not self.compute_neuron_metrics
                and not validate_frontier
                and not request_sample_sums
                and not need_target_rows
            )
            if frontier_only_channel_virtual:
                chunk_size = _frontier_aggregate_conv_block_size(
                    batch_size=batch_size,
                    out_channels=int(weight.size(0)),
                    in_per_group=int(in_per_group),
                    output_h=int(grads.size(-2)),
                    output_w=int(grads.size(-1)),
                )
                override_size = getattr(self, "streaming_chunk_size", None)
                if override_size is not None:
                    chunk_size = min(chunk_size, max(1, int(override_size)))
                self._active_stream_chunk_size = chunk_size
        dual_enabled = bool(self.dual_channel_virtual and not use_channel_virtual)
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
                weight,
                groups=self.groups,
            )
            dual_preadd_weight = dual_preadd_weight.to(device=device, dtype=torch.float32)
            dual_shape = (weight.shape[0], weight.shape[1])
            dual_kernel_scores_total = torch.zeros(dual_shape, dtype=torch.float32, device=device)
            dual_scores_signed_acts_total = torch.zeros_like(dual_kernel_scores_total)
            dual_abs_scores_total = torch.zeros_like(dual_kernel_scores_total)
            dual_gradients_total = torch.zeros_like(dual_kernel_scores_total)
            dual_abs_gradients_total = torch.zeros_like(dual_kernel_scores_total)
        for chunk_idx, start in enumerate(range(0, batch_size, chunk_size)):
            end = min(start + chunk_size, batch_size)
            slice_size = end - start
            if slice_size > (self._last_stream_chunk_size or 0):
                self._last_stream_chunk_size = slice_size

            inputs_chunk = _fetch_replay_slice(
                inputs_source,
                start,
                end,
                device=device,
                dtype=torch.float32,
            )
            grads_chunk = grads[start:end].to(device=device, dtype=torch.float32)

            full_acts_chunk = self._fetch_stream_cache_slice("full_acts", start, end, device=device)
            labels_chunk = self._fetch_stream_cache_slice("labels", start, end, device=device)
            pre_softmax_chunk = self._fetch_stream_cache_slice("pre_softmax", start, end, device=device)
            target_unit_chunk = (
                self._fetch_stream_cache_slice("target_unit_acts", start, end, device=device)
                if need_target_rows
                else None
            )
            cross_entropy_chunk = self._fetch_stream_cache_slice("cross_entropy", start, end, device=device)
            needs_materialized_phi = bool(
                (not use_channel_virtual)
                and self.score_post_af
                and COMPUTE_LOSS_DELTA
                and full_acts_chunk is not None
                and labels_chunk is not None
            )
            need_outputs = (
                effective_collect_bie
                or self.compute_neuron_metrics
                or validate_frontier
                or needs_materialized_phi
            )
            outputs_chunk = (
                F.conv2d(
                    inputs_chunk,
                    weight,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
                if need_outputs
                else None
            )

            if use_channel_virtual:
                del full_acts_chunk, labels_chunk, pre_softmax_chunk, cross_entropy_chunk
                preadd_chunk = _compute_conv_channel_virtual_preadd(
                    inputs_chunk,
                    preadd_weight,
                    groups=self.groups,
                    in_per_group=in_per_group,
                    out_per_group=out_per_group,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                )
                chunk_payload = self._reduce_channel_virtual_chunk(
                    preadd_chunk,
                    grads_chunk,
                    outputs_chunk,
                    target_unit_acts_chunk=target_unit_chunk,
                    return_sample_sums=request_sample_sums,
                )
            elif needs_materialized_phi:
                grad_weight_chunk = _per_sample_conv2d_weight(
                    inputs_chunk,
                    grads_chunk,
                    weight_shape,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
                phi_chunk = grad_weight_chunk * weight.unsqueeze(0)
                chunk_payload = self._reduce_phi_chunk(
                    phi_chunk,
                    grad_weight_chunk,
                    grads_chunk,
                    outputs_chunk,
                    inputs_chunk=inputs_chunk,
                    weight_matrix=weight,
                    full_acts_chunk=full_acts_chunk,
                    labels_chunk=labels_chunk,
                    pre_softmax_chunk=pre_softmax_chunk,
                    target_unit_acts_chunk=target_unit_chunk,
                    cross_entropy_loss_chunk=cross_entropy_chunk,
                )
            else:
                del full_acts_chunk, labels_chunk, pre_softmax_chunk, cross_entropy_chunk
                chunk_payload = _conv_reduce_kernel_exact_chunk(
                    inputs_chunk,
                    grads_chunk,
                    weight,
                    scale=scale,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                    outputs_chunk=outputs_chunk,
                    compute_neuron_metrics=self.compute_neuron_metrics,
                    target_unit_acts_chunk=target_unit_chunk,
                    return_sample_sums=request_sample_sums,
                    emit_bitpack=bool(getattr(self, "capture_bitpack", COMPUTE_BITPACK)),
                    abs_aggregation_mode=self.abs_aggregation_mode,
                )
                if dual_enabled and dual_preadd_weight is not None:
                    dual_preadd_chunk = _compute_conv_channel_virtual_preadd(
                        inputs_chunk,
                        dual_preadd_weight,
                        groups=self.groups,
                        in_per_group=dual_in_per_group,
                        out_per_group=dual_out_per_group,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                    )
                    dual_chunk = _conv_channel_virtual_reduce_exact_chunk(
                        dual_preadd_chunk,
                        grads_chunk,
                        scale=self.scale if self.scale is not None else 1.0,
                        outputs_chunk=None,
                        compute_neuron_metrics=False,
                        abs_aggregation_mode=self.abs_aggregation_mode,
                        need_per_sample_scores=False,
                    )
                    dual_kernel_scores_total += dual_chunk["kernel_scores_sum"]
                    dual_scores_signed_acts_total += dual_chunk["scores_signed_acts_sum"]
                    dual_abs_scores_total += dual_chunk["abs_scores_sum"]
                    dual_gradients_total += dual_chunk["gradients_sum"]
                    dual_abs_gradients_total += dual_chunk["abs_gradients_sum"]
            sample_sums = chunk_payload.pop("sample_sums", None)
            sample_abs_sums = chunk_payload.pop("sample_abs_sums", None)
            if isinstance(sample_sums, torch.Tensor):
                self._stream_sample_sums.append(sample_sums.detach().to(device="cpu", dtype=torch.float32))
            if isinstance(sample_abs_sums, torch.Tensor):
                self._stream_sample_abs_sums.append(sample_abs_sums.detach().to(device="cpu", dtype=torch.float32))
            self._reducers.update_all(chunk_payload)

            if callable(stats_callback) and sample_sums is not None:
                try:
                    phi_rows = sample_sums
                    if phi_rows.device != device:
                        phi_rows = phi_rows.to(device=device)
                    if bias is not None and not _EXCLUDE_BIAS_FROM_SAMPLE_SUMS:
                        bias_term = grads_chunk.sum(dim=(2, 3)) * bias.view(1, -1)
                        phi_rows = phi_rows + (bias_term / scale)
                    stats_callback(phi_rows, target_unit_chunk)
                except Exception:
                    logger.debug(
                        "[stream] GPU stats update failed for %s",
                        getattr(self, "frontier_layer_name", "unknown"),
                        exc_info=True,
                    )

            if use_channel_virtual:
                del inputs_chunk, grads_chunk, preadd_chunk, outputs_chunk
            elif needs_materialized_phi:
                del inputs_chunk, grads_chunk, grad_weight_chunk, phi_chunk, outputs_chunk
            else:
                del inputs_chunk, grads_chunk, outputs_chunk

        if isinstance(inputs_source, torch.Tensor):
            self._cached_inputs_for_hvp = inputs_source.detach().to(device="cpu")
        else:
            self._cached_inputs_for_hvp = inputs_source

        self._stream_process_categories(
            inputs_source=inputs_source,
            saved_neuron_grads=grads,
            weight=weight,
            bias=bias,
            device=device,
            use_channel_virtual=use_channel_virtual,
        )

        weight_cpu = weight.detach().cpu()
        scale_value = self.scale if self.scale is not None else 1.0
        if isinstance(scale_value, torch.Tensor):
            scale_value = float(scale_value.detach().cpu())
        self._reducers.record_metadata(
            weight_out=weight_cpu,
            score_basis=self.conv_score_basis,
            scale_value=float(scale_value),
            abs_aggregation_mode=self.abs_aggregation_mode,
        )
        self._stream_cache["weight_out"] = weight_cpu
        if (
            dual_enabled
            and dual_kernel_scores_total is not None
            and dual_scores_signed_acts_total is not None
            and dual_abs_scores_total is not None
            and dual_gradients_total is not None
            and dual_abs_gradients_total is not None
        ):
            self.analysis_payload = {
                "channel_virtual": {
                    "kernel_scores_sum": dual_kernel_scores_total.detach().cpu(),
                    "scores_signed_acts_sum": dual_scores_signed_acts_total.detach().cpu(),
                    "abs_scores_sum": dual_abs_scores_total.detach().cpu(),
                    "gradients_sum": dual_gradients_total.detach().cpu(),
                    "abs_gradients_sum": dual_abs_gradients_total.detach().cpu(),
                }
            }
        else:
            self.analysis_payload = None
        self._stream_processed = True
        self._active_stream_chunk_size = None
        if self._stream_sample_sums:
            self._sample_sum_payload = torch.cat(self._stream_sample_sums, dim=0)
        else:
            self._sample_sum_payload = None
        if self._stream_sample_abs_sums:
            self._sample_abs_sum_payload = torch.cat(self._stream_sample_abs_sums, dim=0)
        else:
            self._sample_abs_sum_payload = None
        self._stream_sample_sums = []
        self._stream_sample_abs_sums = []

    def _stream_process_categories(
        self,
        *,
        inputs_source,
        saved_neuron_grads: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        device: torch.device,
        use_channel_virtual: bool = False,
    ) -> None:
        if self._reducers is None:
            return
        if self._stream_category_indices is None or self._stream_category_indices.numel() == 0:
            return

        weight_shape = weight.shape
        preadd_weight = None
        in_per_group = 0
        out_per_group = 0
        if use_channel_virtual:
            preadd_weight, in_per_group, out_per_group = _build_conv_channel_virtual_weight(
                weight,
                groups=self.groups,
            )
            preadd_weight = preadd_weight.to(device=device, dtype=torch.float32)
        total = self._stream_category_indices.numel()
        cat_chunk = max(1, min(total, MAX_CHUNK_SAMPLES))
        for offset in range(0, total, cat_chunk):
            idx_slice = self._stream_category_indices[offset : offset + cat_chunk]
            inputs_cat = _fetch_replay_indices(
                inputs_source,
                idx_slice,
                device=device,
                dtype=torch.float32,
            )
            grads_cat = saved_neuron_grads[idx_slice].to(device=device, dtype=torch.float32)

            full_acts_cat = self._gather_stream_cache("full_acts", idx_slice, device=device)
            labels_cat = self._gather_stream_cache("labels", idx_slice, device=device)
            pre_softmax_cat = self._gather_stream_cache("pre_softmax", idx_slice, device=device)
            target_cat = self._gather_stream_cache("target_unit_acts", idx_slice, device=device)
            cross_entropy_cat = self._gather_stream_cache("cross_entropy", idx_slice, device=device)
            needs_materialized_phi = bool(
                (not use_channel_virtual)
                and self.score_post_af
                and COMPUTE_LOSS_DELTA
                and full_acts_cat is not None
                and labels_cat is not None
            )
            need_outputs = self.compute_neuron_metrics or needs_materialized_phi
            outputs_cat = (
                F.conv2d(
                    inputs_cat,
                    weight,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
                if need_outputs
                else None
            )

            if use_channel_virtual:
                del full_acts_cat, labels_cat, pre_softmax_cat, cross_entropy_cat
                preadd_cat = _compute_conv_channel_virtual_preadd(
                    inputs_cat,
                    preadd_weight,
                    groups=self.groups,
                    in_per_group=in_per_group,
                    out_per_group=out_per_group,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                )
                cat_payload = self._reduce_channel_virtual_chunk(
                    preadd_cat,
                    grads_cat,
                    outputs_cat,
                    target_unit_acts_chunk=target_cat,
                    return_sample_sums=False,
                )
            elif needs_materialized_phi:
                grad_weight_cat = _per_sample_conv2d_weight(
                    inputs_cat,
                    grads_cat,
                    weight_shape,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
                phi_cat = grad_weight_cat * weight.unsqueeze(0)
                cat_payload = self._reduce_phi_chunk(
                    phi_cat,
                    grad_weight_cat,
                    grads_cat,
                    outputs_cat,
                    inputs_chunk=inputs_cat,
                    weight_matrix=weight,
                    full_acts_chunk=full_acts_cat,
                    labels_chunk=labels_cat,
                    pre_softmax_chunk=pre_softmax_cat,
                    target_unit_acts_chunk=target_cat,
                    cross_entropy_loss_chunk=cross_entropy_cat,
                )
            else:
                del full_acts_cat, labels_cat, pre_softmax_cat, cross_entropy_cat
                cat_payload = _conv_reduce_kernel_exact_chunk(
                    inputs_cat,
                    grads_cat,
                    weight,
                    scale=self.scale if self.scale is not None else 1.0,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                    outputs_chunk=outputs_cat,
                    compute_neuron_metrics=self.compute_neuron_metrics,
                    target_unit_acts_chunk=target_cat,
                    emit_bitpack=False,
                    abs_aggregation_mode=self.abs_aggregation_mode,
                )
            cat_payload.pop("sample_sums", None)
            cat_payload.pop("sample_abs_sums", None)
            self._reducers.update_categories(cat_payload)

            if use_channel_virtual:
                del inputs_cat, grads_cat, preadd_cat, outputs_cat
            elif needs_materialized_phi:
                del inputs_cat, grads_cat, grad_weight_cat, phi_cat, outputs_cat
            else:
                del inputs_cat, grads_cat, outputs_cat

    def _assemble_stream_results(
        self,
        *,
        all_payload: Dict[str, Optional[torch.Tensor]],
        cat_payload: Optional[Dict[str, Optional[torch.Tensor]]],
        bitpacked: List[torch.Tensor],
        weight_out: torch.Tensor,
        grad_non_zeroes: torch.Tensor,
    ) -> Tuple:
        def _get(payload: Optional[Dict[str, Optional[torch.Tensor]]], key: str):
            if payload is None:
                return None
            return payload.get(key)

        dummy = torch.zeros(0, dtype=torch.float32)
        sample_sum_payload = self._sample_sum_payload
        sample_abs_sum_payload = self._sample_abs_sum_payload

        if self.retain_backward_payloads:
            self._record_retained_payload(
                sample_sum=sample_sum_payload,
                sample_abs_sum=sample_abs_sum_payload,
                all_payload=all_payload,
                cat_payload=cat_payload,
                bitpacked=bitpacked,
                grad_non_zeroes=grad_non_zeroes,
                weight_out=weight_out,
            )
            sample_sum_payload = None
            sample_abs_sum_payload = None
            self._sample_sum_payload = None
            self._sample_abs_sum_payload = None

        return (
            _get(cat_payload, "kernel_scores_sum"),
            _get(cat_payload, "activating_scores_sum"),
            _get(cat_payload, "inhibiting_scores_sum"),
            _get(cat_payload, "scores_signed_acts_sum"),
            _get(cat_payload, "abs_scores_sum"),
            _get(cat_payload, "gradients_sum"),
            _get(cat_payload, "abs_gradients_sum"),
            _get(cat_payload, "loss_change_sum"),
            _get(cat_payload, "abs_loss_change_sum"),
            _get(cat_payload, "bie_base_sum"),
            _get(cat_payload, "bie_base_abs_sum"),
            _get(cat_payload, "bie_boundary_sum"),
            _get(cat_payload, "bie_boundary_abs_sum"),
            sample_sum_payload if sample_sum_payload is not None else None,
            sample_abs_sum_payload if sample_abs_sum_payload is not None else None,
            dummy,
            dummy,
            dummy,
            all_payload.get("kernel_scores_sum"),
            all_payload.get("activating_scores_sum"),
            all_payload.get("inhibiting_scores_sum"),
            all_payload.get("scores_signed_acts_sum"),
            all_payload.get("abs_scores_sum"),
            all_payload.get("gradients_sum"),
            all_payload.get("abs_gradients_sum"),
            all_payload.get("loss_change_sum"),
            all_payload.get("abs_loss_change_sum"),
            all_payload.get("bie_base_sum"),
            all_payload.get("bie_base_abs_sum"),
            all_payload.get("bie_boundary_sum"),
            all_payload.get("bie_boundary_abs_sum"),
            all_payload.get("neuron_scores_sum"),
            all_payload.get("abs_neuron_scores_sum"),
            all_payload.get("neuron_scores_sq_sum"),
            all_payload.get("abs_neuron_scores_sq_sum"),
            all_payload.get("product_scores_sum"),
            all_payload.get("abs_product_scores_sum"),
            dummy,
            dummy,
            weight_out,
            grad_non_zeroes,
            None,
            None,
            bitpacked,
        )

    def _finalize_streaming_results(self) -> None:
        if not self._streaming_enabled():
            return
        if self._reducers is None:
            self._initialize_reducers()
        self._run_streaming_reduction()

        bundle = self._reducers.finalize()
        all_payload = dict(bundle.get("all", {}))
        cat_payload = bundle.get("categories")
        bitpacked = bundle.get("bitpack", [])
        metadata = bundle.get("metadata", {})
        weight_out = metadata.get("weight_out")
        if weight_out is None:
            weight_out = self._stream_cache.get("weight_out")
        if weight_out is None:
            weight_out = self._effective_weight(device=self.weight.device, dtype=torch.float32).detach().cpu()
        elif torch.is_tensor(weight_out):
            weight_out = weight_out.detach().cpu()

        gradients_sum = all_payload.get("gradients_sum")
        if gradients_sum is not None:
            grad_non_zeroes = torch.tensor(
                int(torch.count_nonzero(gradients_sum).item()),
                dtype=torch.int64,
            )
        else:
            grad_non_zeroes = torch.tensor(0, dtype=torch.int64)

        results_tuple = self._assemble_stream_results(
            all_payload=all_payload,
            cat_payload=cat_payload if isinstance(cat_payload, dict) else None,
            bitpacked=bitpacked,
            weight_out=weight_out,
            grad_non_zeroes=grad_non_zeroes,
        )

        self.results = results_tuple
        self._stream_results_ready = True
        self._stream_bie_success = bool(self._effective_collect_bie() and not self._chunk_capture_failed)

        self._reset_after_scoring()
        self._reset_stream_state()

    def compute_kernel_scores(
        self,
        in_cat_list,
        *,
        inputs: Optional[torch.Tensor] = None,
    ):
        if self.oldfast_mode:
            cached_results = self.results
            if not isinstance(cached_results, tuple):
                cached_results = getattr(self, "_oldfast_results_cache", None)
                if isinstance(cached_results, tuple):
                    self.results = cached_results
            if isinstance(cached_results, tuple):
                if OLDFAST_TRACE:
                    logger.warning(
                        "[oldfast-trace:compute-return-cached] layer=%s id=%s tuple_len=%s cache_attr=%s",
                        getattr(self, "frontier_layer_name", "unknown"),
                        id(self),
                        len(cached_results),
                        isinstance(getattr(self, "_oldfast_results_cache", None), tuple),
                    )
                return cached_results
        saved_grads = self.saved_integrated_multipliers
        if saved_grads is None:
            saved_grads = self.saved_neuron_gradients
        if saved_grads is None:
            if OLDFAST_TRACE:
                logger.warning(
                    "[oldfast-trace:compute-missing] layer=%s id=%s results_is_tuple=%s compute_scores=%s oldfast=%s replay=%s saved_output=%s",
                    getattr(self, "frontier_layer_name", "unknown"),
                    id(self),
                    isinstance(self.results, tuple),
                    getattr(self, "compute_scores", None),
                    getattr(self, "oldfast_mode", None),
                    self._active_replay_provider is not None,
                    self._saved_output_buffer is not None,
                )
            raise RuntimeError("Saved neuron gradients are unavailable for scoring.")
        if self._effective_collect_bie():
            self._reset_chunk_capture()

        compute_device = self.weight.device
        if self.retain_backward_payloads:
            self.results_ready.clear()
            self.retained_payloads.clear()
        device = compute_device
        scale = self.scale if self.scale is not None else 1.0
        if isinstance(scale, torch.Tensor):
            scale = scale.item()
        if abs(scale) < 1e-12:
            scale = 1.0

        inputs_source = self._active_replay_provider or inputs
        if inputs_source is None:
            raise RuntimeError("Replay source unavailable for aggregated scoring.")

        grads_detached = saved_grads.detach()
        use_channel_virtual = self._use_channel_virtual_basis()
        use_per_sample_path = bool(self.store_per_sample and not use_channel_virtual)
        if use_per_sample_path:
            grads_source = grads_detached.to(dtype=torch.float32, device="cpu")
        else:
            grads_source = grads_detached.to(dtype=torch.float32, device=compute_device, non_blocking=True)

        weight = self._effective_weight(device=compute_device, dtype=torch.float32)
        bias_eff = self._effective_bias(device=compute_device, dtype=torch.float32)
        debug_layer_name = (
            getattr(self, "frontier_layer_name", None) if getattr(self, "enable_layer_corr_logging", False) else None
        )
        request_sample_sums = bool(
            self.capture_sample_sums
            or self.collect_stats
            or debug_layer_name
            or getattr(self, "collect_score_correlation", True)
        )

        if use_per_sample_path:
            results = self._compute_per_sample_scores(
                in_cat_list,
                inputs_source,
                grads_source,
                weight,
                scale,
                compute_device,
                bias=bias_eff,
                validate_frontier=VALIDATE_FRONTIER,
                debug_layer_name=debug_layer_name,
            )
            self._finalize_frontier_stats()
            self._reset_after_scoring()
            del grads_source
            return results

        if hasattr(inputs_source, "get"):
            prepared_inputs = inputs_source
        else:
            prepared_inputs = inputs_source.to(dtype=torch.float32, device=compute_device)

        if isinstance(inputs_source, torch.Tensor):
            self._cached_inputs_for_hvp = inputs_source.detach().to(device="cpu")
        else:
            self._cached_inputs_for_hvp = inputs_source

        target_acts = (
            self.target_unit_acts
            if bool(self.collect_stats or debug_layer_name or getattr(self, "collect_score_correlation", True))
            else None
        )
        stats_cb = self._update_gpu_stats if self.collect_stats else None

        aux_payload: Dict[str, Dict[str, torch.Tensor]] = {}
        results = conv_compute_kernel_score_aggregated(
            inputs_source=prepared_inputs,
            saved_neuron_gradients=grads_source,
            weights=weight,
            scale=scale,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            device=compute_device,
            compute_neuron_metrics=self.compute_neuron_metrics,
            bitpack_sink=getattr(self, "bitpack_sink", None),
            bias=bias_eff,
            validate_frontier=VALIDATE_FRONTIER,
            target_acts=target_acts,
            stats_callback=stats_cb,
            return_sample_sums=request_sample_sums,
            debug_layer_name=debug_layer_name,
            score_basis=self.conv_score_basis,
            dual_channel_virtual=bool(self.dual_channel_virtual and not self._use_channel_virtual_basis()),
            auxiliary_payload=aux_payload,
            emit_bitpack=bool(getattr(self, "capture_bitpack", COMPUTE_BITPACK)),
            abs_aggregation_mode=self.abs_aggregation_mode,
        )
        self.analysis_payload = aux_payload or None

        self._finalize_frontier_stats()
        self._reset_after_scoring()
        del grads_source
        return results

    def _compute_per_sample_scores(
        self,
        in_cat_list,
        inputs_source: Union[torch.Tensor, TensorReplayProvider],
        saved_neuron_grads: torch.Tensor,
        weight: torch.Tensor,
        scale: float,
        device: torch.device,
        bias: Optional[torch.Tensor] = None,
        validate_frontier: bool = False,
        debug_layer_name: Optional[str] = None,
    ):
        if isinstance(inputs_source, torch.Tensor):
            self._cached_inputs_for_hvp = inputs_source.detach().to(device="cpu")
        else:
            self._cached_inputs_for_hvp = inputs_source

        weight_shape = weight.shape
        out_features = weight_shape[0]
        in_features = weight_shape[1]
        spatial_elements = weight_shape[2] * weight_shape[3] if weight.dim() == 4 else 1
        chunk_size = choose_chunk_size(out_features, in_features, spatial_elements)
        batch_size = saved_neuron_grads.size(0)

        final_shape = weight_shape
        final_shape_no_spatial = (weight_shape[0], weight_shape[1])

        zeros_like_weight = torch.zeros(final_shape, dtype=torch.float32, device=device)

        if bias is not None:
            bias = bias.to(device=device, dtype=torch.float32)

        frontier_lhs = torch.zeros((), device=device, dtype=torch.float64) if validate_frontier else None
        frontier_rhs = torch.zeros_like(frontier_lhs) if validate_frontier else None
        frontier_bias = torch.zeros_like(frontier_lhs) if (validate_frontier and bias is not None) else None
        all_accumulators: Dict[str, Optional[torch.Tensor]] = {
            "scores": zeros_like_weight.clone(),
            "activating_scores": zeros_like_weight.clone(),
            "inhibiting_scores": zeros_like_weight.clone(),
            "scores_signed_acts": zeros_like_weight.clone(),
            "abs_scores": zeros_like_weight.clone(),
            "gradient_component": zeros_like_weight.clone(),
            "abs_gradient_component": zeros_like_weight.clone(),
            "loss_contribution": None,
            "abs_loss_contribution": None,
            "product_scores": zeros_like_weight.clone() if self.compute_neuron_metrics else None,
            "abs_product_scores": zeros_like_weight.clone() if self.compute_neuron_metrics else None,
        }

        if not self.compute_neuron_metrics:
            all_neuron_scores = None
            all_abs_neuron_scores = None
            all_neuron_scores_sq = None
            all_abs_neuron_scores_sq = None
        else:
            all_neuron_scores = torch.zeros(out_features, dtype=torch.float32, device=device)
            all_abs_neuron_scores = torch.zeros_like(all_neuron_scores)
            all_neuron_scores_sq = torch.zeros_like(all_neuron_scores)
            all_abs_neuron_scores_sq = torch.zeros_like(all_neuron_scores)

        all_bitpacked: List[torch.Tensor] = []

        def _slice_optional(t: Optional[torch.Tensor], start: int, end: int) -> Optional[torch.Tensor]:
            if t is None:
                return None
            return t[start:end]

        # When debugging row sums, accumulate per-sample sums across chunks so
        # downstream correlation can align samples to the objective vector.
        capture_sample_sums = bool(self.capture_sample_sums)
        sample_sum_accum_cpu: Optional[torch.Tensor] = None
        sample_abs_sum_accum_cpu: Optional[torch.Tensor] = None
        if capture_sample_sums and batch_size > 0:
            sample_sum_accum_cpu = torch.zeros(
                (batch_size, out_features), dtype=torch.float32, device="cpu"
            )
            sample_abs_sum_accum_cpu = torch.zeros_like(sample_sum_accum_cpu)
        if not debug_layer_name and getattr(self, "enable_layer_corr_logging", False):
            debug_layer_name = getattr(self, "frontier_layer_name", None)
        layer_corr_scores: Optional[List[torch.Tensor]] = [] if debug_layer_name else None
        layer_corr_objectives: Optional[List[torch.Tensor]] = [] if debug_layer_name else None
        bie_logging_enabled = bool(self._effective_collect_bie() and self.bie_logging_enabled and debug_layer_name)
        bie_layer_corr_scores: Optional[List[torch.Tensor]] = [] if bie_logging_enabled else None
        bie_layer_corr_objectives: Optional[List[torch.Tensor]] = [] if bie_logging_enabled else None
        bie_base_total = 0.0
        bie_boundary_total = 0.0

        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            inputs_chunk = _fetch_replay_slice(
                inputs_source,
                start,
                end,
                device=device,
                dtype=torch.float32,
            )
            grads_chunk = saved_neuron_grads[start:end].to(device=device, dtype=torch.float32)

            grad_weight_chunk = _per_sample_conv2d_weight(
                inputs_chunk,
                grads_chunk,
                weight_shape,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            phi_chunk = grad_weight_chunk * weight.unsqueeze(0)

            need_outputs = self._effective_collect_bie() or self.compute_neuron_metrics or validate_frontier
            outputs_chunk = (
                F.conv2d(
                    inputs_chunk,
                    weight,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
                if need_outputs
                else None
            )

            if validate_frontier:
                frontier_lhs += (weight * grad_weight_chunk.sum(dim=0)).sum().to(torch.float64)
                frontier_rhs += (outputs_chunk * grads_chunk).sum().to(torch.float64)
                if frontier_bias is not None:
                    frontier_bias += (bias * grads_chunk.sum(dim=(0, 2, 3))).sum().to(torch.float64)

            target_chunk = _slice_optional(self.target_unit_acts, start, end)
            chunk_results = self._reduce_phi_chunk(
                phi_chunk,
                grad_weight_chunk,
                grads_chunk,
                outputs_chunk,
                inputs_chunk=inputs_chunk,
                weight_matrix=weight,
                full_acts_chunk=_slice_optional(self.full_acts, start, end),
                labels_chunk=_slice_optional(self.label, start, end),
                pre_softmax_chunk=_slice_optional(self.pre_softmax_vector, start, end),
                target_unit_acts_chunk=target_chunk,
                cross_entropy_loss_chunk=_slice_optional(self.cross_entropy_loss, start, end),
            )
            bie_base_samples = chunk_results.pop("bie_base_sample_totals", None)
            bie_boundary_samples = chunk_results.pop("bie_boundary_sample_totals", None)

            kernel_sum = chunk_results["kernel_scores_sum"].to(device=device)
            activating_sum = chunk_results["activating_scores_sum"].to(device=device)
            inhibiting_sum = chunk_results["inhibiting_scores_sum"].to(device=device)
            signed_sum = chunk_results["scores_signed_acts_sum"].to(device=device)
            abs_scores_sum = chunk_results["abs_scores_sum"].to(device=device)
            grad_sum = chunk_results["gradients_sum"].to(device=device)
            abs_grad_sum = chunk_results["abs_gradients_sum"].to(device=device)

            all_accumulators["scores"] += kernel_sum
            all_accumulators["activating_scores"] += activating_sum
            all_accumulators["inhibiting_scores"] += inhibiting_sum
            all_accumulators["scores_signed_acts"] += signed_sum
            all_accumulators["abs_scores"] += abs_scores_sum
            all_accumulators["gradient_component"] += grad_sum
            all_accumulators["abs_gradient_component"] += abs_grad_sum

            if chunk_results["loss_change_sum"] is not None:
                loss_sum = chunk_results["loss_change_sum"].to(device=device)
                abs_loss_sum = (
                    chunk_results["abs_loss_change_sum"].to(device=device)
                    if chunk_results["abs_loss_change_sum"] is not None
                    else None
                )
                if all_accumulators["loss_contribution"] is None:
                    all_accumulators["loss_contribution"] = loss_sum.clone()
                    if abs_loss_sum is not None:
                        all_accumulators["abs_loss_contribution"] = abs_loss_sum.clone()
                else:
                    all_accumulators["loss_contribution"] += loss_sum
                    if abs_loss_sum is not None:
                        if all_accumulators["abs_loss_contribution"] is None:
                            all_accumulators["abs_loss_contribution"] = abs_loss_sum.clone()
                        else:
                            all_accumulators["abs_loss_contribution"] += abs_loss_sum

            if self.compute_neuron_metrics and chunk_results["product_scores_sum"] is not None:
                prod_sum = chunk_results["product_scores_sum"].to(device=device)
                abs_prod_sum = (
                    chunk_results["abs_product_scores_sum"].to(device=device)
                    if chunk_results["abs_product_scores_sum"] is not None
                    else None
                )
                all_accumulators["product_scores"] += prod_sum
                if all_accumulators["abs_product_scores"] is None:
                    all_accumulators["abs_product_scores"] = abs_prod_sum
                else:
                    if abs_prod_sum is not None:
                        all_accumulators["abs_product_scores"] += abs_prod_sum

            if self.compute_neuron_metrics and all_neuron_scores is not None:
                all_neuron_scores += chunk_results["neuron_scores_sum"].to(device=device)
                all_abs_neuron_scores += chunk_results["abs_neuron_scores_sum"].to(device=device)
                all_neuron_scores_sq += chunk_results["neuron_scores_sq_sum"].to(device=device)
                all_abs_neuron_scores_sq += chunk_results["abs_neuron_scores_sq_sum"].to(device=device)

            all_bitpacked.extend(chunk_results["bitpacked"])

            chunk_sample = chunk_results.get("sample_sums")
            if (
                debug_layer_name
                and torch.is_tensor(chunk_sample)
                and chunk_sample.numel() > 0
            ):
                per_sample_scores = chunk_sample.sum(dim=1)
                objective_vec = _flatten_objective_vec(target_chunk)
                _maybe_record_layer_corr_sample(
                    layer_corr_scores,
                    layer_corr_objectives,
                    per_sample_scores,
                    objective_vec,
                )

            if bie_logging_enabled:
                bie_per_sample: Optional[torch.Tensor] = None
                if torch.is_tensor(bie_base_samples):
                    bie_base_total += float(bie_base_samples.sum().item())
                    bie_per_sample = (
                        bie_base_samples
                        if bie_per_sample is None
                        else bie_per_sample + bie_base_samples
                    )
                if torch.is_tensor(bie_boundary_samples):
                    bie_boundary_total += float(bie_boundary_samples.sum().item())
                    bie_per_sample = (
                        bie_boundary_samples
                        if bie_per_sample is None
                        else bie_per_sample + bie_boundary_samples
                    )
                if bie_per_sample is not None:
                    objective_vec = _flatten_objective_vec(target_chunk)
                    _maybe_record_layer_corr_sample(
                        bie_layer_corr_scores,
                        bie_layer_corr_objectives,
                        bie_per_sample,
                        objective_vec,
                    )

            if (
                capture_sample_sums
                and sample_sum_accum_cpu is not None
                and torch.is_tensor(chunk_sample)
                and chunk_sample.numel() > 0
            ):
                try:
                    # chunk_sample arrives on CPU and has shape [slice, Cout]
                    slice_len = end - start
                    if chunk_sample.shape[0] != slice_len:
                        # If shape is unexpected, attempt a best-effort view
                        flat = chunk_sample.reshape(-1, out_features)
                        if flat.shape[0] == slice_len:
                            chunk_sample = flat
                    sample_sum_accum_cpu[start:end] = chunk_sample.to(
                        device="cpu", dtype=torch.float32
                    )
                    abs_chunk = chunk_sample.abs()
                    if torch.is_tensor(sample_abs_sum_accum_cpu):
                        sample_abs_sum_accum_cpu[start:end] = abs_chunk
                except Exception:
                    # Do not fail the scoring path due to diagnostics
                    pass
        _finalize_layer_corr_logging(
            debug_layer_name,
            layer_corr_scores,
            layer_corr_objectives,
        )
        if bie_logging_enabled:
            _finalize_layer_bie_corr_logging(
                debug_layer_name,
                bie_layer_corr_scores,
                bie_layer_corr_objectives,
                bie_base_total,
                bie_boundary_total,
            )

        grad_non_zeroes = torch.tensor(
            torch.count_nonzero(all_accumulators["gradient_component"]).item(),
            device=device,
        )

        in_cat_indices = torch.as_tensor(in_cat_list, dtype=torch.long, device=device) if in_cat_list else None
        cat_accumulators: Optional[Dict[str, Optional[torch.Tensor]]] = None

        if in_cat_indices is not None and in_cat_indices.numel() > 0:
            cat_accumulators = {k: (torch.zeros_like(v) if v is not None else None) for k, v in all_accumulators.items()}
            cat_chunk = max(1, chunk_size)

            for start in range(0, in_cat_indices.numel(), cat_chunk):
                end = min(start + cat_chunk, in_cat_indices.numel())
                idx_slice = in_cat_indices[start:end]

                inputs_cat = _fetch_replay_indices(
                    inputs_source,
                    idx_slice,
                    device=device,
                    dtype=torch.float32,
                )
                grads_cat = saved_neuron_grads[idx_slice].to(device=device, dtype=torch.float32)

                grad_weight_cat = _per_sample_conv2d_weight(
                    inputs_cat,
                    grads_cat,
                    weight_shape,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
                phi_cat = grad_weight_cat * weight.unsqueeze(0)

                need_outputs = self._effective_collect_bie() or self.compute_neuron_metrics
                outputs_cat = (
                    F.conv2d(
                        inputs_cat,
                        weight,
                        bias=None,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=self.groups,
                    )
                    if need_outputs
                    else None
                )

                chunk_cat_results = self._reduce_phi_chunk(
                    phi_cat,
                    grad_weight_cat,
                    grads_cat,
                    outputs_cat,
                    inputs_chunk=inputs_cat,
                    weight_matrix=weight,
                    full_acts_chunk=self.full_acts[idx_slice] if self.full_acts is not None else None,
                    labels_chunk=self.label[idx_slice] if self.label is not None else None,
                    pre_softmax_chunk=self.pre_softmax_vector[idx_slice] if self.pre_softmax_vector is not None else None,
                    target_unit_acts_chunk=self.target_unit_acts[idx_slice] if self.target_unit_acts is not None else None,
                    cross_entropy_loss_chunk=self.cross_entropy_loss[idx_slice] if self.cross_entropy_loss is not None else None,
                )

                cat_accumulators["scores"] += chunk_cat_results["kernel_scores_sum"].to(device=device)
                cat_accumulators["activating_scores"] += chunk_cat_results["activating_scores_sum"].to(device=device)
                cat_accumulators["inhibiting_scores"] += chunk_cat_results["inhibiting_scores_sum"].to(device=device)
                cat_accumulators["scores_signed_acts"] += chunk_cat_results["scores_signed_acts_sum"].to(device=device)
                cat_accumulators["abs_scores"] += chunk_cat_results["abs_scores_sum"].to(device=device)
                cat_accumulators["gradient_component"] += chunk_cat_results["gradients_sum"].to(device=device)
                cat_accumulators["abs_gradient_component"] += chunk_cat_results["abs_gradients_sum"].to(device=device)

                if chunk_cat_results["loss_change_sum"] is not None:
                    cat_loss_sum = chunk_cat_results["loss_change_sum"].to(device=device)
                    cat_abs_loss_sum = (
                        chunk_cat_results["abs_loss_change_sum"].to(device=device)
                        if chunk_cat_results["abs_loss_change_sum"] is not None
                        else None
                    )
                    if cat_accumulators["loss_contribution"] is None:
                        cat_accumulators["loss_contribution"] = cat_loss_sum.clone()
                        if cat_abs_loss_sum is not None:
                            cat_accumulators["abs_loss_contribution"] = cat_abs_loss_sum.clone()
                    else:
                        cat_accumulators["loss_contribution"] += cat_loss_sum
                        if cat_abs_loss_sum is not None:
                            if cat_accumulators["abs_loss_contribution"] is None:
                                cat_accumulators["abs_loss_contribution"] = cat_abs_loss_sum.clone()
                            else:
                                cat_accumulators["abs_loss_contribution"] += cat_abs_loss_sum

                if self.compute_neuron_metrics and chunk_cat_results["product_scores_sum"] is not None:
                    cat_prod_sum = chunk_cat_results["product_scores_sum"].to(device=device)
                    cat_abs_prod_sum = (
                        chunk_cat_results["abs_product_scores_sum"].to(device=device)
                        if chunk_cat_results["abs_product_scores_sum"] is not None
                        else None
                    )
                    cat_accumulators["product_scores"] += cat_prod_sum
                    if cat_accumulators["abs_product_scores"] is None:
                        cat_accumulators["abs_product_scores"] = cat_abs_prod_sum
                    else:
                        if cat_abs_prod_sum is not None:
                            cat_accumulators["abs_product_scores"] += cat_abs_prod_sum

        dummy = torch.zeros(0, device=device)

        if self.compute_neuron_metrics and all_neuron_scores is not None:
            neuron_scores = all_neuron_scores
            abs_neuron_scores = all_abs_neuron_scores
            neuron_scores_sq = all_neuron_scores_sq
            abs_neuron_scores_sq = all_abs_neuron_scores_sq
        else:
            neuron_scores = None
            abs_neuron_scores = None
            neuron_scores_sq = None
            abs_neuron_scores_sq = None

        if validate_frontier and frontier_lhs is not None and frontier_rhs is not None:
            lhs_total = frontier_lhs
            rhs_total = frontier_rhs
            if frontier_bias is not None:
                lhs_total = lhs_total + frontier_bias
                rhs_total = rhs_total + frontier_bias
            if not torch.allclose(lhs_total, rhs_total, rtol=1e-3, atol=1e-3):
                raise AssertionError(
                    f"Conv (per-sample) frontier completeness failed: phi={lhs_total.item():.6f}, z·m={rhs_total.item():.6f}"
                )

        cat_scores = cat_accumulators["scores"] if cat_accumulators is not None else None
        cat_activating = cat_accumulators["activating_scores"] if cat_accumulators is not None else None
        cat_inhibiting = cat_accumulators["inhibiting_scores"] if cat_accumulators is not None else None
        cat_signed = cat_accumulators["scores_signed_acts"] if cat_accumulators is not None else None
        cat_abs_scores = cat_accumulators["abs_scores"] if cat_accumulators is not None else None
        cat_gradients = cat_accumulators["gradient_component"] if cat_accumulators is not None else None
        cat_abs_gradients = cat_accumulators["abs_gradient_component"] if cat_accumulators is not None else None
        cat_loss = cat_accumulators["loss_contribution"] if cat_accumulators is not None else None
        cat_abs_loss = cat_accumulators["abs_loss_contribution"] if cat_accumulators is not None else None

        if cat_accumulators is not None and self.compute_neuron_metrics:
            cat_product_scores = cat_accumulators["product_scores"]
            cat_abs_product_scores = cat_accumulators["abs_product_scores"]
        else:
            cat_product_scores = None
            cat_abs_product_scores = None

        all_loss = all_accumulators["loss_contribution"]
        all_abs_loss = all_accumulators["abs_loss_contribution"]

        all_product_scores = all_accumulators["product_scores"]
        all_abs_product_scores = all_accumulators["abs_product_scores"]

        weights_out = weight

        finalize_start = time.perf_counter()

        result_tuple = (
            cat_scores,
            cat_activating,
            cat_inhibiting,
            cat_signed,
            cat_abs_scores,
            cat_gradients,
            cat_abs_gradients,
            cat_loss,
            cat_abs_loss,
            sample_sum_accum_cpu if sample_sum_accum_cpu is not None else dummy,
            sample_abs_sum_accum_cpu if sample_abs_sum_accum_cpu is not None else dummy,
            dummy,
            dummy,
            dummy,
            all_accumulators["scores"],
            all_accumulators["activating_scores"],
            all_accumulators["inhibiting_scores"],
            all_accumulators["scores_signed_acts"],
            all_accumulators["abs_scores"],
            all_accumulators["gradient_component"],
            all_accumulators["abs_gradient_component"],
            all_loss,
            all_abs_loss,
            neuron_scores,
            abs_neuron_scores,
            neuron_scores_sq,
            abs_neuron_scores_sq,
            all_product_scores,
            all_abs_product_scores,
            dummy,
            dummy,
            weights_out,
            grad_non_zeroes,
            None,
            None,
            all_bitpacked,
        )

        cpu_result_tuple = tuple(self._detach_structure_cpu(item) for item in result_tuple)
        finalize_duration = time.perf_counter() - finalize_start
        self._store_results_ready(cpu_result_tuple, finalize_duration)

        # Release GPU references before returning.
        del all_accumulators
        if cat_accumulators is not None:
            del cat_accumulators
        del weights_out
        del weight

        return cpu_result_tuple


_COMPILE_CONV = bool(int(os.environ.get("SCORING_COMPILE_CONV", "0")))
if _COMPILE_CONV:
    conv_compute_kernel_score_aggregated = compile_if_available(conv_compute_kernel_score_aggregated)
