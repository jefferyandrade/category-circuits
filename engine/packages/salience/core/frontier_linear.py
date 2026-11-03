from __future__ import annotations

import gc
import math
import random
import logging
import time
from collections import OrderedDict
from packages.salience.ops.linear_utils import safe_pearson_corr
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, Set

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import LogNorm

from packages.salience.utils import (
    compile_if_available,
    pack_boolean_rows,
    vectorised_bitpack,
    TensorReplayProvider,
    ReplayTensorProxy as _ReplayTensorProxy,
    replay_batch_size as _replay_batch_size,
    replay_matches_tensor as _replay_matches_tensor,
    fetch_replay_slice as _fetch_replay_slice,
    fetch_replay_indices as _fetch_replay_indices,
)
from packages.salience.ops.linear_utils import (
    _prepare_softmax_stats,
    _loss_diff_bn_kernel,
    compute_loss_difference_triton_optimized,
    dissected_conv_tiled_kernel,
    fixed_pre_sum_kernel,
    masked_linear,
    analyze_base_scores_vs_loss_change_efficient,
    check_zeros_relationship,
    compute_sparsity_detailed,
    create_sparse_output_tensor_dict,
    update_sparse_cross_products,
    save_row_distributions_2d,
    save_row_distributions,
    batch_pearsonr,
    batch_spearmanr,
    safe_pearson_corr,
    save_average_distributions,
    zscore_standardize_scores_dissect,
    minmax_norm_scores_dissect,
    _compute_chunk_scores,
    _compute_chunk_bie,
)
from packages.salience.core.reducers import ReducerBundle
from packages.salience.analysis.extraction import FeatureExtractionMixin
from packages.salience.ops.linear_emit import (
    _choose_frontier_aggregate_linear_block_size,
    _linear_reduce_exact_chunk,
    _resolve_scale_tensors,
    linear_compute_frontier_aggregate_score,
    linear_compute_kernel_score,
    linear_compute_kernel_score_aggregated,
)
from packages.salience.ops.abs_mode import resolve_abs_aggregation_mode
from packages.salience.core.frontier_utils import (
    COMPUTE_BITPACK,
    COMPUTE_LOSS_DELTA,
    DEFAULT_TARGET_CHUNK_ELEMENTS,
    MAX_CHUNK_SAMPLES,
    REPLAY_STORAGE_DEVICE,
    VALIDATE_FRONTIER,
    GPUStats,
    debug_print,
    current_replay_provider,
    pop_replay_provider,
    push_replay_provider,
    choose_chunk_size,
)
logger = logging.getLogger(__name__)

__all__ = [
    "TensorReplayProvider",
    "choose_chunk_size",
    "pack_boolean_rows",
    "vectorised_bitpack",
    "linear_compute_kernel_score",
    "linear_compute_kernel_score_aggregated",
    "HookedLinear",
]


def debug_print(*args, **kwargs):
    if DEBUG_SCORING:
        print(*args, **kwargs)

    
# Helpful for ceiling division

# ==========
# Precompute per-batch scalars (one-time, O(B·M))
# ==========

class HookedLinear(FeatureExtractionMixin, nn.Module):
    """
    An optimized version of the HookedLinear module for memory and speed,
    preserving the original functionality.

    Attributes:
        calc_kernels (bool): Flag to determine if kernel calculations are performed.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        absolute_score: bool = True,
        score_neg: bool = True,
        normalize_features_per_image: bool = False,
        weight_norm_type: str = 'sum_abs',
        score_post_af = False,
        store_per_sample: bool = True,
        *,
        collect_integrated: bool = False,
        collect_stats: bool = False,
        collect_hvp: bool = False,
        abs_aggregation_mode: Optional[str] = None,
    ):
        """
        Initializes the HookedLinear module.

        Args:
            original_linear (nn.Linear): The original linear layer to hook into.
            absolute_score (bool, optional): Whether to use absolute scores. Defaults to True.
            score_neg (bool, optional): Whether to include negative scores. Defaults to True.
        """
        super().__init__()

        # Extend the original linear layer with Backpack to get per-sample gradients
        self.original_linear: nn.Linear = original_linear
        self.score_post_af = score_post_af
        self.in_features: int = self.original_linear.in_features
        self.out_features: int = self.original_linear.out_features
        self.absolute: bool = absolute_score
        self.score_neg: bool = score_neg
        self.kernel_scores: Optional[torch.Tensor] = None
        self.capture_inputs: bool = True
        self.capture_gradients: bool = False
        self.in_cat_indices_list: Optional[List[int]] = None
        self.full_acts = None
        self.hook_in = False
        self.pre_softmax_vector = None
        self.compute_scores: bool = False
        self.normalize_features_per_image: bool = normalize_features_per_image
        self.weight_norm_type: str = weight_norm_type
        self.scale = None
        self.original_layer_activations = None
        self.calc_kernels = False
        self.running_activation_sum: Optional[torch.Tensor] = None
        self.running_activation_count: int = 0
        self.results = None
        self._oldfast_results_cache = None
        self.enable_layer_corr_logging: bool = False
        self.saved_integrated_multipliers: Optional[torch.Tensor] = None
        self.compute_neuron_metrics: bool = True
        self.store_per_sample = store_per_sample
        self.capture_sample_sums: bool = False
        self.capture_bitpack: bool = COMPUTE_BITPACK
        self.collect_score_correlation: bool = True
        self.collect_integrated = bool(collect_integrated)
        self.collect_stats = bool(collect_stats)
        self.collect_hvp = bool(collect_hvp)
        self.abs_aggregation_mode = resolve_abs_aggregation_mode(
            "exact_abs",
            explicit_mode=abs_aggregation_mode,
        )
        self.collect_bie: bool = False
        self._chunk_records: Dict[int, Dict[str, torch.Tensor]] = {}
        self._chunk_capture_failed: bool = False
        self._pending_chunk_ids: Set[int] = set()
        self.bie_logging_enabled: bool = False
        self._stream_bie_success: bool = False
        self.saved_weight_gradients = None
        self.device = self.original_linear.weight.device
        self.bitpack_sink: Optional[Callable[[torch.Tensor], None]] = None
        self._pending_replay_provider: Optional[object] = None
        self._active_replay_provider: Optional[object] = None
        self._warned_replay_batch_mismatch: bool = False
        # self.original_linear.register_forward_hook(self.capture_input_hook)
        self.get_act_weights = True
        self.linear_hook_neurons = None
        self.target_unit = None
        self.weight_tensor = torch.zeros((0,), device=self.device)
        self.mask = None
        self.original_weights=None
        self.target_unit_acts=None
        self.cross_entropy_loss = None
        self.saved_neuron_gradients = None
        self.saved_integrated_multipliers: Optional[torch.Tensor] = None
        self._pre_bias_output: Optional[torch.Tensor] = None
        self.z_cache: Optional[torch.Tensor] = None
        self.retain_z_cache_until_reset: bool = False
        self.ig_capture_output = True
        self.frontier_stats: Optional[Dict[str, float]] = None
        self.gpu_stats: Optional[GPUStats] = GPUStats(device=self.device)
        self._cached_inputs_for_hvp: Optional[Union[torch.Tensor, TensorReplayProvider]] = None
        self._cached_inputs_for_hvp: Optional[Union[torch.Tensor, "TensorReplayProvider"]] = None
        self.stream_flush_enabled: bool = False
        self.stream_flush_replay_device: Optional[torch.device] = None
        self._stream_results_ready: bool = False
        self.streaming_reductions: bool = False
        self._reducers: Optional[ReducerBundle] = None
        self._stream_cache: Dict[str, Optional[torch.Tensor]] = {}
        self._stream_category_indices: Optional[torch.Tensor] = None
        self._stream_processed: bool = False
        self._last_stream_chunk_size: Optional[int] = None
        self._active_stream_chunk_size: Optional[int] = None
        self._pending_activation_sum: Optional[torch.Tensor] = None
        self._pending_activation_count: int = 0
        self._stream_sample_sums: List[torch.Tensor] = []
        self._stream_sample_abs_sums: List[torch.Tensor] = []
        self._sample_sum_payload: Optional[torch.Tensor] = None
        self._sample_abs_sum_payload: Optional[torch.Tensor] = None
        self.oldfast_mode: bool = False
        self._oldfast_inputs: Optional[torch.Tensor] = None
        self._preserve_score_capture_state: bool = False
        self.retain_backward_payloads: bool = False
        self.results_ready: Dict[str, List[Dict[str, object]]] = {}
        self.retained_payloads: Dict[str, Dict[str, object]] = {}

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

    def _oldfast_score_requested(self) -> bool:
        return bool(self.oldfast_mode and (self.compute_scores or self.calc_kernels))

    def _compute_oldfast_results(self, grad: torch.Tensor) -> None:
        if not self._oldfast_score_requested():
            return
        inputs = self._oldfast_inputs
        if inputs is None and self._active_replay_provider is not None:
            try:
                inputs = self._active_replay_provider.get(
                    0,
                    int(self._active_replay_provider.batch_size),
                    device=self.original_linear.weight.device,
                    dtype=torch.float32,
                )
            except Exception:
                inputs = None
        if inputs is None:
            raise RuntimeError("Oldfast linear scoring requires saved forward inputs.")
        device = self.original_linear.weight.device
        scale = self.scale if self.scale is not None else 1.0
        if isinstance(scale, torch.Tensor):
            scale = float(scale.item())
        if abs(scale) < 1e-12:
            scale = 1.0
        inputs_detached = inputs.detach().to(device=device)
        if inputs_detached.dim() > 2:
            inputs_detached = inputs_detached.reshape(inputs_detached.size(0), -1)
        grad_detached = grad.detach().to(device=device, dtype=torch.float32)
        weight = self.original_linear.weight.detach().to(device=device, dtype=torch.float32)
        if self.mask is not None:
            weight = weight * self.mask.to(device=device, dtype=torch.float32)
        bias = self.original_linear.bias
        if bias is not None:
            bias = bias.detach().to(device=device, dtype=torch.float32)
        if self.compute_neuron_metrics:
            self.results = linear_compute_kernel_score_aggregated(
                inputs_source=inputs_detached,
                saved_neuron_gradients=grad_detached,
                scale=scale,
                weights=weight,
                device=device,
                compute_neuron_metrics=True,
                bitpack_sink=None,
                bias=bias,
                validate_frontier=False,
                target_acts=None,
                stats_callback=None,
                return_sample_sums=False,
                debug_layer_name=None,
                emit_bitpack=False,
                abs_aggregation_mode=self.abs_aggregation_mode,
            )
        else:
            self.results = linear_compute_frontier_aggregate_score(
                inputs_source=inputs_detached,
                saved_neuron_gradients=grad_detached,
                scale=scale,
                weights=weight,
                device=device,
                abs_aggregation_mode=self.abs_aggregation_mode,
            )
        self._oldfast_results_cache = self.results
        context = getattr(self, "_stream_context", None)
        layer_name = getattr(self, "frontier_layer_name", None)
        if context is not None and getattr(context, "enabled", False) and layer_name:
            context.submit(layer_name, self.results)
        self._stream_results_ready = True
        self.saved_neuron_gradients = None
        self.saved_integrated_multipliers = None
        self._pre_bias_output = None
        self.post_sum_output = None
        self._oldfast_inputs = None

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

    def set_retain_backward_payloads(self, enabled: bool) -> None:
        self.retain_backward_payloads = bool(enabled)
        if not enabled:
            self.results_ready.clear()
            self.retained_payloads.clear()

    def set_compute_neuron_metrics(self, enabled: bool) -> None:
        self.compute_neuron_metrics = bool(enabled)

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
        layer_key = getattr(self, "frontier_layer_name", None) or "unknown"
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
    ) -> None:
        if not self.retain_backward_payloads:
            return
        layer_key = getattr(self, "frontier_layer_name", None) or "unknown"
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
        def _tensor_bytes(obj: Optional[torch.Tensor]) -> int:
            if obj is None or not torch.is_tensor(obj):
                return 0
            return obj.element_size() * obj.numel()
        total_bytes = 0
        for value in store.values():
            if isinstance(value, dict):
                for tensor in value.values():
                    total_bytes += _tensor_bytes(tensor)
            elif isinstance(value, list):
                for tensor in value:
                    total_bytes += _tensor_bytes(tensor)
            else:
                total_bytes += _tensor_bytes(value if torch.is_tensor(value) else None)
        logger.info(
            "[payload-retain] layer=%s retained_payload_bytes=%.3fMB",
            layer_key,
            total_bytes / (1024 * 1024) if total_bytes else 0.0,
        )
        self.retained_payloads[layer_key] = store

    def _streaming_enabled(self) -> bool:
        return bool(self.streaming_reductions)

    def _reset_stream_state(self) -> None:
        self._commit_running_activation_stats()
        self._reducers = None
        self._stream_cache = {}
        self._stream_category_indices = None
        self._stream_processed = False
        self._pending_activation_sum = None
        self._pending_activation_count = 0
        self._stream_sample_sums = []
        self._stream_sample_abs_sums = []
        self._sample_sum_payload = None
        self._sample_abs_sum_payload = None
        self._chunk_records.clear()
        self._chunk_capture_failed = False

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

    def _capture_stream_cache(
        self,
        *,
        full_acts,
        pre_softmax_vector,
        cross_entropy_loss,
        grad_to_final_acts,
        final_acts,
        label,
    ) -> None:
        def _to_cpu(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if t is None:
                return None
            return t.detach().cpu()

        self._stream_cache = {
            "full_acts": _to_cpu(full_acts),
            "pre_softmax": _to_cpu(pre_softmax_vector),
            "cross_entropy": _to_cpu(cross_entropy_loss),
            "grad_to_final": _to_cpu(grad_to_final_acts),
            "final_acts": _to_cpu(final_acts),
            "labels": _to_cpu(label),
            "target_unit_acts": _to_cpu(self.target_unit_acts),
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
            device=self.device,
            track_categories=track_categories,
            compute_neuron_metrics=self.compute_neuron_metrics,
            collect_bitpack=bool(getattr(self, "capture_bitpack", COMPUTE_BITPACK)),
            collect_bie=self.collect_bie,
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

    def flush_results(self) -> None:
        """
        Aggregate scores immediately after backward when streaming is enabled.
        Results are moved to CPU and intermediate GPU caches are released.
        """
        if self._streaming_enabled():
            if getattr(self, "_stream_results_ready", False):
                return
            try:
                self._finalize_streaming_results()
            except Exception:
                logger.exception(
                    "[stream] Linear layer %s finalize failed",
                    getattr(self, "frontier_layer_name", "unknown"),
                )
                return
            context = getattr(self, "_stream_context", None)
            layer_name = getattr(self, "frontier_layer_name", None)
            if context is not None and getattr(context, "enabled", False) and layer_name:
                context.submit(layer_name, self.results)
            self._stream_results_ready = True
            return

        if not getattr(self, "stream_flush_enabled", False):
            return
        if getattr(self, "_stream_results_ready", False):
            return
        if self.results is not None and not getattr(self, "compute_scores", False):
            self._stream_results_ready = True
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

        results = self.compute_kernel_scores(cat_indices, inputs=None)
        if not isinstance(results, tuple):
            results = tuple(results)
        if len(results) < 36:
            results_list = list(results) + [None] * (36 - len(results))
        else:
            results_list = list(results)
        if results_list[32] is None:
            results_list[32] = torch.tensor(0.0)
        if results_list[-1] is None:
            results_list[-1] = []
        processed = tuple(results_list)
        self.results = self._detach_structure_cpu(processed)
        self.saved_integrated_multipliers = None
        self.saved_neuron_gradients = None
        self.post_sum_output = None
        self._pre_bias_output = None
        self.target_unit_acts = None
        if not self.retain_z_cache_until_reset:
            self.z_cache = None
        self.in_cat_indices_list = None
        provider = getattr(self, "_active_replay_provider", None)
        if provider is not None:
            free = getattr(provider, "free", None)
            if callable(free):
                try:
                    free()
                except Exception:
                    logger.debug(
                        "Failed to free replay provider for layer %s",
                        getattr(self, "frontier_layer_name", "unknown"),
                    )
        self._active_replay_provider = None
        context = getattr(self, "_stream_context", None)
        layer_name = getattr(self, "frontier_layer_name", None)
        if context is not None and getattr(context, "enabled", False) and layer_name:
            context.submit(layer_name, self.results)
        self._stream_results_ready = True

    def prepare_for_scoring(
        self,
        *,
        cross_entropy_loss,
        full_acts,
        pre_softmax_vector,
        in_cat_indices_list,
        should_score: bool,
    ) -> None:
        self.calc_kernels = bool(should_score)
        self.compute_scores = bool(should_score)
        self.results = None
        self._oldfast_results_cache = None
        self.frontier_stats = None
        self._stream_results_ready = False
        self.in_cat_indices_list = in_cat_indices_list

        if self._streaming_enabled():
            self._reset_stream_state()
            self._set_stream_category_indices(in_cat_indices_list)
            self._capture_stream_cache(
                full_acts=full_acts,
                pre_softmax_vector=pre_softmax_vector,
                cross_entropy_loss=cross_entropy_loss,
                grad_to_final_acts=getattr(self, "grads_to_final_acts", None),
                final_acts=getattr(self, "final_acts", None),
                label=getattr(self, "label", None),
            )
            self.cross_entropy_loss = self._stream_cache.get("cross_entropy")
            self.full_acts = None
            self.pre_softmax_vector = None

            cached_target = self._stream_cache.get("target_unit_acts")
            if cached_target is not None:
                self.target_unit_acts = cached_target

            cached_grad_final = self._stream_cache.get("grad_to_final")
            if cached_grad_final is not None:
                self.grads_to_final_acts = cached_grad_final

            cached_final = self._stream_cache.get("final_acts")
            if cached_final is not None:
                self.final_acts = cached_final

            cached_labels = self._stream_cache.get("labels")
            if cached_labels is not None:
                self.label = cached_labels
        else:
            self.cross_entropy_loss = cross_entropy_loss
            self.full_acts = full_acts
            self.pre_softmax_vector = pre_softmax_vector

    def _reduce_linear_chunk(
        self,
        grads_chunk: torch.Tensor,
        *,
        inputs_chunk: torch.Tensor,
        weight_matrix: torch.Tensor,
        outputs_chunk: Optional[torch.Tensor],
        full_acts_chunk: Optional[torch.Tensor],
        labels_chunk: Optional[torch.Tensor],
        pre_softmax_chunk: Optional[torch.Tensor],
        target_unit_acts_chunk: Optional[torch.Tensor],
        cross_entropy_loss_chunk: Optional[torch.Tensor],
    ) -> Dict[str, Optional[torch.Tensor]]:
        scale_value = self.scale if self.scale is not None else 1.0
        scale, _, _ = _resolve_scale_tensors(scale_value, device=weight_matrix.device)

        chunk_stats = _linear_reduce_exact_chunk(
            inputs_chunk,
            grads_chunk,
            weight_matrix,
            scale=scale,
            outputs_chunk=outputs_chunk,
            compute_neuron_metrics=self.compute_neuron_metrics,
            abs_aggregation_mode=self.abs_aggregation_mode,
        )
        kernel_scores_sum = chunk_stats["kernel_scores_sum"]
        activating_scores_sum = chunk_stats["activating_scores_sum"]
        inhibiting_scores_sum = chunk_stats["inhibiting_scores_sum"]
        abs_scores_sum = chunk_stats["abs_scores_sum"]
        scores_signed_acts_sum = chunk_stats["scores_signed_acts_sum"]
        gradients_sum = chunk_stats["gradients_sum"]
        abs_gradients_sum = chunk_stats["abs_gradients_sum"]
        sample_sum_matrix = chunk_stats["sample_sums"]
        if sample_sum_matrix is None and (
            self.capture_sample_sums or target_unit_acts_chunk is not None or self.compute_neuron_metrics
        ):
            raise RuntimeError("Linear exact reducer expected `outputs_chunk` for sample-level reductions.")
        sample_sum_cpu: Optional[torch.Tensor] = None
        sample_abs_cpu: Optional[torch.Tensor] = None
        if sample_sum_matrix is not None:
            sample_sum_cpu = sample_sum_matrix.detach().to(device="cpu", dtype=torch.float32)
            sample_abs_cpu = sample_sum_cpu.abs()

        loss_change_sum: Optional[torch.Tensor] = None
        abs_loss_change_sum: Optional[torch.Tensor] = None
        if (
            self.score_post_af
            and COMPUTE_LOSS_DELTA
            and full_acts_chunk is not None
            and labels_chunk is not None
        ):
            grad_factor = grads_chunk / float(scale)
            kernel_scores_no_spatial = (
                torch.einsum("bo,bi->boi", grad_factor, inputs_chunk) * weight_matrix.unsqueeze(0)
            )
            flat_scores = kernel_scores_no_spatial.reshape(kernel_scores_no_spatial.size(0), -1)
            loss_diff = compute_loss_difference_triton_optimized(
                flattened_base_scores=flat_scores,
                full_acts=full_acts_chunk,
                labels=labels_chunk,
                target_unit=self.target_unit if self.target_unit is not None else 0,
                original_loss=cross_entropy_loss_chunk,
                out_dtype=torch.float32,
            )
            loss_diff = loss_diff.view_as(kernel_scores_no_spatial)
            loss_change_sum = loss_diff.sum(dim=0)
            abs_loss_change_sum = loss_diff.abs().sum(dim=0)

        neuron_scores_sum: Optional[torch.Tensor] = None
        abs_neuron_scores_sum: Optional[torch.Tensor] = None
        neuron_scores_sq_sum: Optional[torch.Tensor] = None
        abs_neuron_scores_sq_sum: Optional[torch.Tensor] = None
        product_scores_sum: Optional[torch.Tensor] = None
        abs_product_scores_sum: Optional[torch.Tensor] = None

        if outputs_chunk is not None:
            neuron_scores_sum = chunk_stats["neuron_scores_sum"]
            abs_neuron_scores_sum = chunk_stats["abs_neuron_scores_sum"]
            neuron_scores_sq_sum = chunk_stats["neuron_scores_sq_sum"]
            abs_neuron_scores_sq_sum = chunk_stats["abs_neuron_scores_sq_sum"]
            product_scores_sum = chunk_stats["product_scores_sum"]
            abs_product_scores_sum = chunk_stats["abs_product_scores_sum"]

        corr = torch.tensor(0.0, device=weight_matrix.device)
        if target_unit_acts_chunk is not None:
            target_unit_flat = target_unit_acts_chunk.to(dtype=torch.float32, device=weight_matrix.device)
            totals = sample_sum_matrix.sum(dim=1)
            corr = safe_pearson_corr(totals, target_unit_flat)

        bitpacked: List[torch.Tensor] = []
        if bool(getattr(self, "capture_bitpack", COMPUTE_BITPACK)):
            weight_nonzero = weight_matrix.ne(0)
            inputs_nonzero = inputs_chunk.ne(0)
            grads_nonzero = grads_chunk.ne(0)
            for sample_idx in range(grads_nonzero.size(0)):
                sample_mask = (
                    weight_nonzero
                    & grads_nonzero[sample_idx].unsqueeze(1)
                    & inputs_nonzero[sample_idx].unsqueeze(0)
                )
                packed_rows = pack_boolean_rows(sample_mask.reshape(sample_mask.size(0), -1))
                for row in packed_rows:
                    bitpacked.append(row.cpu())

        def _to_cpu(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if torch.is_tensor(t):
                return t.detach().to(device="cpu")
            return t

        bie_base_sum: Optional[torch.Tensor] = None
        bie_base_abs_sum: Optional[torch.Tensor] = None
        bie_boundary_sum: Optional[torch.Tensor] = None
        bie_boundary_abs_sum: Optional[torch.Tensor] = None
        if self.collect_bie and torch.is_tensor(outputs_chunk):
            bie_base_chunk, bie_boundary_chunk = _compute_chunk_bie(
                inputs_chunk,
                outputs_chunk,
                grads_chunk,
                weight_matrix,
            )
            if torch.is_tensor(bie_base_chunk):
                bie_base_scaled = bie_base_chunk / scale
                bie_base_sum = bie_base_scaled.sum(dim=0)
                bie_base_abs_sum = bie_base_scaled.abs().sum(dim=0)
            if torch.is_tensor(bie_boundary_chunk):
                bie_boundary_scaled = bie_boundary_chunk / scale
                bie_boundary_sum = bie_boundary_scaled.sum(dim=0)
                bie_boundary_abs_sum = bie_boundary_scaled.abs().sum(dim=0)

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

        inputs_source = self._active_replay_provider or current_replay_provider()
        if inputs_source is None:
            raise RuntimeError("Replay source unavailable for streaming reduction.")

        device = self.original_linear.weight.device
        if self.retain_backward_payloads:
            self.results_ready.clear()
        saved_grads = saved_grads.to(dtype=torch.float32, device=device)

        weight = self.original_linear.weight.detach().to(dtype=torch.float32, device=device)
        if self.mask is not None:
            weight = weight * self.mask.to(device=device, dtype=torch.float32)
        bias = self.original_linear.bias
        if bias is not None:
            bias = bias.to(device=device, dtype=torch.float32)

        batch_size = _replay_batch_size(inputs_source, saved_grads.size(0))
        stats_callback = getattr(self, "_update_gpu_stats", None) if self.collect_stats else None
        debug_layer_name = (
            getattr(self, "frontier_layer_name", None)
            if getattr(self, "enable_layer_corr_logging", False)
            else None
        )
        need_target_rows = bool(callable(stats_callback) or debug_layer_name or getattr(self, "collect_score_correlation", True))
        request_sample_sums = bool(self.capture_sample_sums or need_target_rows)
        need_outputs = (
            self.collect_bie
            or self.compute_neuron_metrics
            or request_sample_sums
            or bool(need_target_rows)
            or callable(stats_callback)
        )
        if need_outputs:
            base_chunk = choose_chunk_size(weight.size(0), weight.size(1))
        else:
            base_chunk = _choose_frontier_aggregate_linear_block_size(
                batch_size=batch_size,
                in_features=weight.size(1),
                out_features=weight.size(0),
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
        self._stream_sample_sums = []
        self._stream_sample_abs_sums = []

        for start in range(0, batch_size, chunk_size):
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
            if inputs_chunk.dim() > 2:
                inputs_chunk = inputs_chunk.view(inputs_chunk.size(0), -1)

            grads_chunk = saved_grads[start:end].to(device=device, dtype=torch.float32)
            full_acts_chunk = self._fetch_stream_cache_slice("full_acts", start, end, device=device)
            labels_chunk = self._fetch_stream_cache_slice("labels", start, end, device=device)
            pre_softmax_chunk = self._fetch_stream_cache_slice("pre_softmax", start, end, device=device)
            target_chunk = (
                self._fetch_stream_cache_slice("target_unit_acts", start, end, device=device)
                if need_target_rows
                else None
            )
            cross_entropy_chunk = self._fetch_stream_cache_slice("cross_entropy", start, end, device=device)
            outputs_chunk = F.linear(inputs_chunk, weight, bias=None) if need_outputs else None

            chunk_payload = self._reduce_linear_chunk(
                grads_chunk,
                inputs_chunk=inputs_chunk,
                weight_matrix=weight,
                outputs_chunk=outputs_chunk,
                full_acts_chunk=full_acts_chunk,
                labels_chunk=labels_chunk,
                pre_softmax_chunk=pre_softmax_chunk,
                target_unit_acts_chunk=target_chunk,
                cross_entropy_loss_chunk=cross_entropy_chunk,
            )
            sample_sums = chunk_payload.pop("sample_sums", None)
            sample_abs_sums = chunk_payload.pop("sample_abs_sums", None)
            if isinstance(sample_sums, torch.Tensor):
                self._stream_sample_sums.append(sample_sums)
            if isinstance(sample_abs_sums, torch.Tensor):
                self._stream_sample_abs_sums.append(sample_abs_sums)
            self._reducers.update_all(chunk_payload)

            if callable(stats_callback) and outputs_chunk is not None:
                try:
                    phi_rows = outputs_chunk * grads_chunk
                    if bias is not None:
                        phi_rows = phi_rows + grads_chunk * bias.view(1, -1)
                    stats_callback(phi_rows, target_chunk)
                except Exception:
                    logger.debug(
                        "[stream] GPU stats update failed for %s",
                        getattr(self, "frontier_layer_name", "unknown"),
                        exc_info=True,
                    )

            del inputs_chunk, grads_chunk, outputs_chunk

        self._cache_inputs_for_hvp(inputs_source)

        self._stream_process_categories(
            inputs_source=inputs_source,
            saved_neuron_grads=saved_grads,
            weight=weight,
            device=device,
        )

        weight_cpu = weight.detach().cpu()
        scale_value = self.scale if self.scale is not None else 1.0
        if isinstance(scale_value, torch.Tensor):
            scale_value = float(scale_value.detach().cpu())
        self._reducers.record_metadata(
            weight_out=weight_cpu,
            scale_value=float(scale_value),
            abs_aggregation_mode=self.abs_aggregation_mode,
        )
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
        device: torch.device,
    ) -> None:
        if self._reducers is None:
            return
        if self._stream_category_indices is None or self._stream_category_indices.numel() == 0:
            return

        total = self._stream_category_indices.numel()
        cat_chunk = max(1, min(total, MAX_CHUNK_SAMPLES))
        for offset in range(0, total, cat_chunk):
            idx_slice = self._stream_category_indices[offset : offset + cat_chunk]
            if idx_slice.numel() == 0:
                continue

            inputs_cat = _fetch_replay_indices(
                inputs_source,
                idx_slice,
                device=device,
                dtype=torch.float32,
            )
            if inputs_cat.dim() > 2:
                inputs_cat = inputs_cat.view(inputs_cat.size(0), -1)

            grads_cat = saved_neuron_grads[idx_slice].to(device=device, dtype=torch.float32)

            full_acts_cat = self._gather_stream_cache("full_acts", idx_slice, device=device)
            labels_cat = self._gather_stream_cache("labels", idx_slice, device=device)
            pre_softmax_cat = self._gather_stream_cache("pre_softmax", idx_slice, device=device)
            target_cat = self._gather_stream_cache("target_unit_acts", idx_slice, device=device)
            cross_entropy_cat = self._gather_stream_cache("cross_entropy", idx_slice, device=device)

            outputs_cat = (
                F.linear(inputs_cat, weight, bias=None)
                if (self.collect_bie or self.compute_neuron_metrics or target_cat is not None)
                else None
            )

            chunk_cat_payload = self._reduce_linear_chunk(
                grads_cat,
                inputs_chunk=inputs_cat,
                weight_matrix=weight,
                outputs_chunk=outputs_cat,
                full_acts_chunk=full_acts_cat,
                labels_chunk=labels_cat,
                pre_softmax_chunk=pre_softmax_cat,
                target_unit_acts_chunk=target_cat,
                cross_entropy_loss_chunk=cross_entropy_cat,
            )
            chunk_cat_payload.pop("sample_sums", None)
            chunk_cat_payload.pop("sample_abs_sums", None)
            self._reducers.update_categories(chunk_cat_payload)

            del inputs_cat, grads_cat, outputs_cat

    def _assemble_stream_results(
        self,
        *,
        all_payload: Dict[str, Optional[torch.Tensor]],
        cat_payload: Optional[Dict[str, Optional[torch.Tensor]]],
        bitpacked: List[torch.Tensor],
        grad_non_zeroes: torch.Tensor,
    ) -> Tuple:
        def _get(payload: Optional[Dict[str, Optional[torch.Tensor]]], key: str):
            if payload is None:
                return None
            return payload.get(key)

        dummy = torch.zeros(0, dtype=torch.float32)
        sample_sum_payload = self._sample_sum_payload
        sample_abs_sum_payload = self._sample_abs_sum_payload

        cat_bie_base_sum = _get(cat_payload, "bie_base_sum")
        cat_bie_base_abs_sum = _get(cat_payload, "bie_base_abs_sum")
        cat_bie_boundary_sum = _get(cat_payload, "bie_boundary_sum")
        cat_bie_boundary_abs_sum = _get(cat_payload, "bie_boundary_abs_sum")

        all_bie_base_sum = all_payload.get("bie_base_sum")
        all_bie_base_abs_sum = all_payload.get("bie_base_abs_sum")
        all_bie_boundary_sum = all_payload.get("bie_boundary_sum")
        all_bie_boundary_abs_sum = all_payload.get("bie_boundary_abs_sum")

        if self.retain_backward_payloads:
            self._record_retained_payload(
                sample_sum=sample_sum_payload,
                sample_abs_sum=sample_abs_sum_payload,
                all_payload=all_payload,
                cat_payload=cat_payload,
                bitpacked=bitpacked,
                grad_non_zeroes=grad_non_zeroes,
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
            cat_bie_base_sum,
            cat_bie_base_abs_sum,
            cat_bie_boundary_sum,
            cat_bie_boundary_abs_sum,
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
            all_bie_base_sum,
            all_bie_base_abs_sum,
            all_bie_boundary_sum,
            all_bie_boundary_abs_sum,
            all_payload.get("neuron_scores_sum"),
            all_payload.get("abs_neuron_scores_sum"),
            all_payload.get("neuron_scores_sq_sum"),
            all_payload.get("abs_neuron_scores_sq_sum"),
            all_payload.get("product_scores_sum"),
            all_payload.get("abs_product_scores_sum"),
            dummy,
            dummy,
            dummy,
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
            grad_non_zeroes=grad_non_zeroes,
        )

        self.results = results_tuple
        self._stream_results_ready = True
        self._stream_bie_success = bool(self.collect_bie and not self._chunk_capture_failed)

        self.post_sum_output = None
        self.saved_neuron_gradients = None
        self.saved_integrated_multipliers = None
        self._pre_bias_output = None
        if not self.retain_z_cache_until_reset:
            self.z_cache = None

        self._reset_stream_state()
    # def capture_input_hook(self, module: nn.Module, inputs: Tuple[torch.Tensor], outputs: torch.Tensor) -> None:
    #     """
    #     Captures the inputs during the forward pass.

    #     Args:
    #         module (nn.Module): The module where the hook is registered.
    #         inputs (Tuple[torch.Tensor]): The input tensors to the module.
    #         outputs (torch.Tensor): The output tensor from the module.
    #     """
    #     if self.capture_inputs:
    #         module.saved_inputs = inputs[0].detach()
            # if self.get_act_weights:
            #     act_weights_temp = None
            #     # Fixed batch size for processing
            #     fixed_batch_size = 5 if self.out_features > 2000 and self.in_features > 5000 else (
            #         25 if self.out_features > 2000 else 250
            #     )
            #     for start_idx in range(0, module.saved_inputs.shape[0], fixed_batch_size):
            #         end_idx = min(start_idx + fixed_batch_size, module.saved_inputs.shape[0])
            #         # Extract batch data
            #         saved_inputs_batch = module.saved_inputs[start_idx:end_idx]
            #         if act_weights_temp == None:
            #             act_weights_temp = (saved_inputs_batch.unsqueeze(1) * module.weight.detach()).sum(dim=0)
            #         else:
            #             act_weights_temp += (saved_inputs_batch.unsqueeze(1) * module.weight.detach()).sum(dim=0)
            #     self.act_weights = act_weights_temp
            #     save_dir = '<a local folder>'
            #     save_average_distributions(torch.flatten((act_weights_temp.sum(dim=0)), start_dim=0, end_dim=1), save_dir, self.in_features, self.out_features)
            #     print('act weights', self.act_weights.shape)
            # print('saved inputs', module.saved_inputs.shape)
    


    # def capture_gradients_hook(self, module: nn.Module, grad_inputs: Tuple[torch.Tensor], grad_outputs: Tuple[torch.Tensor]) -> None:
    #     """
    #     Captures the gradients during the backward pass.

    #     Args:
    #         module (nn.Module): The module where the hook is registered.
    #         grad_inputs (Tuple[torch.Tensor]): The gradients w.r.t. the inputs.
    #         grad_outputs (Tuple[torch.Tensor]): The gradients w.r.t. the outputs.
    #     """
    #     if self.capture_gradients:
    #         # Access per-sample gradients using Backpack's BatchGrad extension
    #         self.saved_gradients = module.weight.grad_batch.detach()
    #         # Clear grad_batch to free memory
    #         module.weight.grad_batch = None

    #         if self.compute_scores:
    #             self.results = self.compute_kernel_scores(self.in_cat_indices_list)

    def _save_neuron_grads(self, grad: torch.Tensor) -> None:
        if self._oldfast_score_requested():
            with torch.no_grad():
                self._compute_oldfast_results(grad)
            return
        needs_score_gradient = bool(
            self.compute_scores
            or self._streaming_enabled()
            or getattr(self, "stream_flush_enabled", False)
        )
        if not self.compute_neuron_metrics and not needs_score_gradient:
            return
        with torch.no_grad():
            self.saved_neuron_gradients = grad.detach().to(
                dtype=torch.float32
            )

    def _update_gpu_stats(self, phi_rows: torch.Tensor, target_rows: Optional[torch.Tensor] = None) -> None:
        if phi_rows is None:
            return
        if self.gpu_stats is None:
            self.gpu_stats = GPUStats(device=phi_rows.device)
        self.gpu_stats.update(phi_rows, target_rows)

    def collect_integrated_multipliers(self) -> Optional[torch.Tensor]:
        tensor = self._pre_bias_output
        if tensor is None:
            return None
        grad = tensor.grad
        self._pre_bias_output = None
        if grad is None:
            return None
        result = grad.detach().to(dtype=torch.float32)
        try:
            tensor.grad = None
        except AttributeError:
            pass
        return result

    def compute_kernel_scores(
        self,
        in_cat_list: Optional[List[int]],
        *,
        inputs: Optional[torch.Tensor] = None,
    ) -> Tuple:
        if self.oldfast_mode:
            cached_results = self.results
            if not isinstance(cached_results, tuple):
                cached_results = getattr(self, "_oldfast_results_cache", None)
                if isinstance(cached_results, tuple):
                    self.results = cached_results
            if isinstance(cached_results, tuple):
                return cached_results

        device = self.original_linear.weight.device
        if self.retain_backward_payloads:
            self.retained_payloads.clear()
        if device != self.device:
            self.device = device
            self.weight_tensor = self.weight_tensor.to(device)
        scale = self.scale if self.scale is not None else 1.0
        if isinstance(scale, torch.Tensor):
            scale = scale.item()
        if abs(scale) < 1e-12:
            scale = 1.0
        if self._should_stage_boundary_chunks():
            self._reset_chunk_capture()

        saved_neuron_grads = self.saved_integrated_multipliers
        if saved_neuron_grads is None:
            saved_neuron_grads = self.collect_integrated_multipliers()
        if saved_neuron_grads is None:
            saved_neuron_grads = self.saved_neuron_gradients
        if saved_neuron_grads is None:
            raise RuntimeError("Saved neuron gradients are unavailable for score computation.")

        grads_detached = saved_neuron_grads.detach()

        finalize_start = time.perf_counter()

        debug_layer_name = getattr(self, "frontier_layer_name", None) if getattr(self, "enable_layer_corr_logging", False) else None
        request_sample_sums = bool(
            self.capture_sample_sums
            or self.collect_stats
            or debug_layer_name
            or getattr(self, "collect_score_correlation", True)
        )

        if self.store_per_sample:
            saved_neuron_grads = grads_detached.to(dtype=torch.float32, device="cpu")
        else:
            saved_neuron_grads = grads_detached.to(dtype=torch.float32, device=device)
        neuron_outputs = (
            self.post_sum_output.detach().to(dtype=torch.float32, device=device)
            if self.compute_neuron_metrics
            else None
        )
        weight = self.original_linear.weight.detach().to(dtype=torch.float32, device=device)

        if self.original_layer_activations is not None and self.target_unit is not None:
            original_layer_means = self.original_layer_activations.detach()[self.target_unit].unsqueeze(0)
        else:
            original_layer_means = None

        streaming_inputs = None
        inputs_provider = None
        weight_acts_tensor = None

        replay_source = self._active_replay_provider
        if replay_source is None and inputs is not None:
            candidate = inputs
            if isinstance(candidate, torch.Tensor):
                candidate = candidate.detach()
                if candidate.dim() > 2:
                    candidate = candidate.reshape(candidate.size(0), -1)
                if candidate.size(1) != self.in_features:
                    raise RuntimeError(
                        f"HookedLinear aggregated scoring requires inputs with {self.in_features} features; "
                        f"got tensor with shape {tuple(inputs.shape)}"
                    )
                replay_source = TensorReplayProvider(candidate, storage_device=REPLAY_STORAGE_DEVICE)
            else:
                replay_source = candidate


        if self.store_per_sample:
            if self.pre_sum_output is None:
                if replay_source is None:
                    raise RuntimeError("Per-sample activations are unavailable while store_per_sample=True.")
                if hasattr(replay_source, "get"):
                    inputs_provider = replay_source  # streaming via provider
                else:
                    streaming_inputs = replay_source.to(device=device, dtype=torch.float32)
            else:
                weight_acts_tensor = self.pre_sum_output.to(device=device, dtype=torch.float32)
            chunk_recorder = self if self._should_stage_boundary_chunks() else None
            results = linear_compute_kernel_score(
                weight_acts_tensor,
                None,
                in_cat_list,
                self.score_neg,
                self.absolute,
                self.normalize_features_per_image,
                self.full_acts,
                self.pre_softmax_vector.unsqueeze(dim=1).unsqueeze(dim=2),
                self.weight_norm_type,
                device,
                scale,
                original_layer_means,
                weight,
                self.grads_to_final_acts,
                self.final_acts,
                self.label,
                self.target_unit,
                neuron_outputs,
                saved_neuron_grads,
                self.target_unit_acts,
                self.cross_entropy_loss,
                self.score_post_af,
                inputs=streaming_inputs,
                inputs_provider=inputs_provider,
                saved_neuron_gradients=saved_neuron_grads,
                bitpack_sink=getattr(self, "bitpack_sink", None),
                bias=self.original_linear.bias,
                validate_frontier=VALIDATE_FRONTIER,
                stats_callback=self._update_gpu_stats if self.collect_stats else None,
                return_sample_sums=request_sample_sums,
                debug_layer_name=debug_layer_name,
                chunk_recorder=chunk_recorder,
                bie_logging=bool(getattr(self, "bie_logging_enabled", False)),
            )
            self._finalize_frontier_stats()
        else:
            if replay_source is None:
                raise RuntimeError("Replay source unavailable for aggregated scoring.")
            if hasattr(replay_source, "get"):
                inputs_source = replay_source
            else:
                inputs_source = replay_source.to(device=device, dtype=torch.float32)
            target_acts = self.target_unit_acts if bool(debug_layer_name or self.collect_stats or self.collect_score_correlation) else None
            stats_cb = self._update_gpu_stats if self.collect_stats else None
            results = linear_compute_kernel_score_aggregated(
                inputs_source=inputs_source,
                saved_neuron_gradients=saved_neuron_grads,
                scale=scale,
                weights=weight,
                device=device,
                compute_neuron_metrics=self.compute_neuron_metrics,
                bitpack_sink=getattr(self, "bitpack_sink", None),
                bias=self.original_linear.bias,
                validate_frontier=VALIDATE_FRONTIER,
                target_acts=target_acts,
                stats_callback=stats_cb,
                return_sample_sums=request_sample_sums,
                debug_layer_name=debug_layer_name,
                emit_bitpack=bool(getattr(self, "capture_bitpack", COMPUTE_BITPACK)),
                abs_aggregation_mode=self.abs_aggregation_mode,
            )
            self._finalize_frontier_stats()

        cache_source = streaming_inputs
        if cache_source is None:
            cache_source = inputs_provider if inputs_provider is not None else replay_source
        self._cache_inputs_for_hvp(cache_source)

        self.post_sum_output = None
        self.saved_neuron_gradients = None
        self.saved_integrated_multipliers = None
        self.pre_sum_output = None
        self._oldfast_inputs = None
        if not self.retain_z_cache_until_reset:
            self.z_cache = None
        self._active_replay_provider = None
        del saved_neuron_grads

        results_cpu = self._detach_structure_cpu(results)
        finalize_duration = time.perf_counter() - finalize_start
        if isinstance(results_cpu, tuple):
            self._store_results_ready(results_cpu, finalize_duration)
        else:
            logger.debug(
                "[payload-finalize] Unexpected results type for layer=%s: %s",
                getattr(self, "frontier_layer_name", None) or "unknown",
                type(results_cpu).__name__,
            )
        return results_cpu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hooked linear layer.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_features].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, out_features].
        """


        # pre_sum_output = torch.einsum('bi,oi->boi', x, self.original_linear.weight)
        # if x.abs().max() > 65500:
        #     if x.abs().min()*(65500/x.abs().max()) < 0.00006:
        #         print('big problem with rescaling')
        #     else:
        #         print('recasting')
        #         x = (x*(65500/x.abs().max())).to(dtype=torch.float16)
        # else:
        #     if x.abs().min() < 0.00006:
        #         if x.abs().max()*(0.00006/x.abs().min()) > 65500:
        #             print('big problem with rescaling')
        #         else:
        #             print('recasting')
        #             x = (x*(0.00006/x.abs().min())).to(dtype=torch.float16)
        #     else:
        #         x = x.to(dtype=torch.float16)
        
        # if self.original_linear.weight.abs().max() > 65500:
        #     if self.original_linear.weight.abs().min()*(65500/self.original_linear.weight.abs().max()) < 0.00006:
        #         print('big problem with rescaling')
        #     else:
        #         print('recasting')
        #         self.original_linear.weight = (self.original_linear.weight*(65500/self.original_linear.weight.abs().max())).to(dtype=torch.float16)
        # else:
        #     if self.original_linear.weight.abs().min() < 0.00006:
        #         if self.original_linear.weight.abs().max()*(0.00006/self.original_linear.weight.abs().min()) > 65500:
        #             print('big problem with rescaling')
        #         else:
        #             print('recasting')
        #             self.original_linear.weight = (self.original_linear.weight*(0.00006/self.original_linear.weight.abs().min())).to(dtype=torch.float16)
        #     else:
        #         self.original_linear.weight = self.original_linear.weight.to(dtype=torch.float16)

        if not hasattr(self, "running_activation_sum"):
            self.running_activation_sum = None
            self.running_activation_count = 0
        if not hasattr(self, "results"):
            self.results = None
        weight = self.original_linear.weight
        device = weight.device
        if device != self.device:
            self.device = device
            self.weight_tensor = self.weight_tensor.to(device)
        x = x.to(device)
        if self.mask is not None:
            weight = weight * self.mask

        provider = self._pending_replay_provider or current_replay_provider()
        capture_kernels = bool(self.calc_kernels or getattr(self, "compute_scores", False))
        preserve_capture_state = bool(getattr(self, "_preserve_score_capture_state", False))
        if capture_kernels:
            provider_to_use = provider
            if provider_to_use is not None:
                expected = getattr(provider_to_use, "batch_size", None)
                if expected is not None and expected != x.size(0):
                    if expected > x.size(0):
                        if not self._warned_replay_batch_mismatch:
                            print(
                                "[HookedLinear] Replay provider batch_size mismatch: "
                                f"expected {expected}, got {x.size(0)}. "
                                "Falling back to per-batch tensor cache."
                            )
                            self._warned_replay_batch_mismatch = True
                        print(f"[HookedLinear] caching fallback tensor with shape {tuple(x.shape)}")
                        provider_to_use = None
                    else:
                        raise ValueError(
                            f"Replay provider batch_size mismatch: expected {expected}, got {x.size(0)}"
                        )
                elif not _replay_matches_tensor(provider_to_use, x):
                    provider_to_use = None
            storage_device = REPLAY_STORAGE_DEVICE
            override_device = getattr(self, "stream_flush_replay_device", None)
            if getattr(self, "stream_flush_enabled", False) and override_device is not None:
                storage_device = override_device
            if self.oldfast_mode:
                self._oldfast_inputs = x.detach()
                if provider_to_use is not None:
                    self._active_replay_provider = provider_to_use
                else:
                    self._active_replay_provider = TensorReplayProvider(x, storage_device=storage_device)
            elif provider_to_use is not None:
                self._active_replay_provider = provider_to_use
            else:
                self._active_replay_provider = TensorReplayProvider(x, storage_device=storage_device)
            self._pending_replay_provider = None
        else:
            if not preserve_capture_state:
                self._active_replay_provider = None
                self._pending_replay_provider = None
                self._oldfast_inputs = None

        self.pre_sum_output = None

        pre_bias_output = F.linear(x, weight, bias=None)
        if capture_kernels and not pre_bias_output.requires_grad:
            pre_bias_output = pre_bias_output.detach().requires_grad_(True)
        self.z_cache = pre_bias_output.detach()

        with torch.no_grad():
            batch_sum = pre_bias_output.detach().sum(dim=0)
            if self._pending_activation_sum is None:
                self._pending_activation_sum = batch_sum
            else:
                self._pending_activation_sum.add_(batch_sum)
            self._pending_activation_count += pre_bias_output.shape[0]

        if self.linear_hook_neurons is not None and not preserve_capture_state:
            self.linear_hook_neurons.remove()
            self.linear_hook_neurons = None

        if capture_kernels and pre_bias_output.requires_grad:
            if self.linear_hook_neurons is not None:
                self.linear_hook_neurons.remove()
                self.linear_hook_neurons = None
            pre_bias_output.retain_grad()
            self._pre_bias_output = pre_bias_output
            self.linear_hook_neurons = pre_bias_output.register_hook(self._save_neuron_grads)
        else:
            if not preserve_capture_state:
                self._pre_bias_output = None

        self.post_sum_output = pre_bias_output.detach()

        if self.original_linear.bias is None:
            output = pre_bias_output
        else:
            output = pre_bias_output + self.original_linear.bias

        self.hook_in = capture_kernels

        return output
    
    def iter_neuron_gradient_chunks(
        self,
        *,
        chunk_size: Optional[int] = None,
        target_device: Optional[torch.device] = None,
        detach: bool = True,
    ) -> Iterable[Tuple[int, int, torch.Tensor]]:
        """
        Yield cached per-sample neuron gradient slices and clear storage afterwards.

        Args:
            chunk_size: Optional maximum number of samples per chunk.
            target_device: Optional device each chunk should reside on.
            detach: Whether to detach the chunk from autograd.
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


    def get_boundary_inputs(self, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        source = self._cached_inputs_for_hvp
        if source is None:
            return None
        target_device = device or self.device
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

    def _should_cache_inputs_for_hvp(self) -> bool:
        return bool((not self.streaming_reductions) or self.collect_hvp)

    def _cache_inputs_for_hvp(self, source: Optional[Union[torch.Tensor, TensorReplayProvider]]) -> None:
        if not self._should_cache_inputs_for_hvp() or source is None:
            self._cached_inputs_for_hvp = None
            return
        if isinstance(source, torch.Tensor):
            self._cached_inputs_for_hvp = source.detach().to(device="cpu")
        else:
            self._cached_inputs_for_hvp = source

    def _reset_chunk_capture(self) -> None:
        self._chunk_records.clear()
        self._chunk_capture_failed = False
        self._pending_chunk_ids.clear()
        self._stream_bie_success = False

    def _mark_chunk_capture_failed(self) -> None:
        self._chunk_capture_failed = True

    def _should_stage_boundary_chunks(self) -> bool:
        return bool(self.collect_bie and self.collect_integrated)

    def _stage_boundary_chunk_inputs(
        self,
        chunk_id: int,
        start: int,
        end: int,
        inputs_chunk: torch.Tensor,
        preactivations: Optional[torch.Tensor],
    ) -> None:
        if not self._should_stage_boundary_chunks():
            return
        try:
            record = self._chunk_records.setdefault(chunk_id, {"start": start, "end": end})
            self._pending_chunk_ids.add(chunk_id)
            record["a"] = inputs_chunk.detach().to(device="cpu", dtype=torch.float32)
            if preactivations is None:
                self._mark_chunk_capture_failed()
            else:
                record["z"] = preactivations.detach().to(device="cpu", dtype=torch.float32)
        except Exception:
            self._mark_chunk_capture_failed()

    def _stage_boundary_chunk_grads(
        self,
        chunk_id: int,
        start: int,
        end: int,
        grad_chunk: torch.Tensor,
    ) -> None:
        if not self._should_stage_boundary_chunks():
            return
        try:
            record = self._chunk_records.setdefault(chunk_id, {"start": start, "end": end})
            record["delta"] = grad_chunk.detach().to(device="cpu", dtype=torch.float32)
            if "a" not in record or "z" not in record:
                self._mark_chunk_capture_failed()
        except Exception:
            self._mark_chunk_capture_failed()

    def get_boundary_chunk_payload(
        self,
        chunk_id: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if not self._should_stage_boundary_chunks():
            return None
        record = self._chunk_records.get(chunk_id)
        if record is None:
            return None
        required = ("a", "z", "delta")
        if any(key not in record or not torch.is_tensor(record[key]) for key in required):
            self._mark_chunk_capture_failed()
            return None
        return record["a"], record["z"], record["delta"]

    def release_boundary_chunk(self, chunk_id: int) -> None:
        if not self._should_stage_boundary_chunks():
            return
        self._chunk_records.pop(chunk_id, None)
        self._pending_chunk_ids.discard(chunk_id)

    def pop_running_mean_activation(self) -> Optional[torch.Tensor]:
        self._commit_running_activation_stats()
        if self.running_activation_sum is None or self.running_activation_count == 0:
            return None
        mean = (self.running_activation_sum / float(self.running_activation_count)).detach().cpu()
        self.running_activation_sum = None
        self.running_activation_count = 0
        return mean

    def before_backward(self) -> None:
        """
        To be called before the backward pass to enable gradient capturing.
        """
        self.capture_gradients = True
        self.results = None
        self._oldfast_results_cache = None
        self._stream_results_ready = False
        if self._streaming_enabled():
            self._initialize_reducers()

    def after_backward(self) -> None:
        """
        To be called after the backward pass to disable gradient capturing.
        """
        self._commit_running_activation_stats()
        self.capture_gradients = False
        if self._streaming_enabled() or getattr(self, "stream_flush_enabled", False):
            try:
                self.flush_results()
            except Exception:
                logger.exception(
                    "Stream flush failed for layer %s", getattr(self, "frontier_layer_name", "unknown")
                )
        self.compute_scores = False
