from __future__ import annotations
import logging
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn.functional as F

from packages.salience.utils import (
    TensorReplayProvider,
    ReplayTensorProxy as _ReplayTensorProxy,
    replay_batch_size as _replay_batch_size,
    fetch_replay_slice as _fetch_replay_slice,
    fetch_replay_indices as _fetch_replay_indices,
    vectorised_bitpack,
    pack_boolean_rows,
)
from packages.salience.core.frontier_utils import (
    COMPUTE_BITPACK,
    COMPUTE_LOSS_DELTA,
    DEBUG_SCORING,
    VALIDATE_FRONTIER,
    choose_chunk_size,
)
from packages.salience.ops import fused_scores
from packages.salience.ops.abs_mode import (
    is_frontier_aggregate_abs_mode,
    normalize_abs_aggregation_mode,
)
from packages.salience.ops.linear_utils import (
    _compute_chunk_scores,
    create_sparse_output_tensor_dict,
    analyze_base_scores_vs_loss_change_efficient,
    save_row_distributions,
    save_row_distributions_2d,
    compute_sparsity_detailed,
    update_sparse_cross_products,
    batch_spearmanr,
)
from packages.salience.utils.layer_logging import (
    flatten_objective_vec as _flatten_objective_vec,
    finalize_layer_corr_logging as _finalize_layer_corr_logging,
    finalize_layer_bie_corr_logging as _finalize_layer_bie_corr_logging,
    maybe_record_layer_corr_sample as _maybe_record_layer_corr_sample,
)

logger = logging.getLogger(__name__)


def _safe_chunk_stage(recorder: Optional[object], method: str, *args) -> None:
    if recorder is None:
        return
    fn = getattr(recorder, method, None)
    if not callable(fn):
        return
    try:
        fn(*args)
    except Exception:
        fail = getattr(recorder, "_mark_chunk_capture_failed", None)
        if callable(fail):
            fail()


def _fetch_boundary_chunk_tensors(
    recorder: Optional[object],
    chunk_idx: int,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if recorder is None:
        return None, None, None
    getter = getattr(recorder, "get_boundary_chunk_payload", None)
    if not callable(getter):
        return None, None, None
    try:
        payload = getter(chunk_idx)
    except Exception:
        fail = getattr(recorder, "_mark_chunk_capture_failed", None)
        if callable(fail):
            fail()
        return None, None, None
    if not isinstance(payload, (tuple, list)) or len(payload) != 3:
        return None, None, None
    tensors: List[torch.Tensor] = []
    fail = getattr(recorder, "_mark_chunk_capture_failed", None)
    for tensor in payload:
        if not torch.is_tensor(tensor):
            if callable(fail):
                fail()
            return None, None, None
        tensors.append(tensor.to(device=device, dtype=torch.float32))
    return tensors[0], tensors[1], tensors[2]


def _release_boundary_chunk_tensors(recorder: Optional[object], chunk_idx: int) -> None:
    if recorder is None:
        return
    releaser = getattr(recorder, "release_boundary_chunk", None)
    if not callable(releaser):
        return
    try:
        releaser(chunk_idx)
    except Exception:
        fail = getattr(recorder, "_mark_chunk_capture_failed", None)
        if callable(fail):
            fail()


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


def _choose_frontier_aggregate_linear_block_size(
    *,
    batch_size: int,
    in_features: int,
    out_features: int,
) -> int:
    target_mb_raw = os.environ.get(
        "SCORING_FRONTIER_DIRECT_LINEAR_TARGET_MB",
        os.environ.get("SCORING_OLDFAST_LINEAR_TARGET_MB", "512"),
    )
    max_samples_raw = os.environ.get(
        "SCORING_FRONTIER_DIRECT_LINEAR_MAX_SAMPLES",
        os.environ.get("SCORING_OLDFAST_LINEAR_MAX_SAMPLES", "4096"),
    )
    try:
        target_mb = max(1, int(target_mb_raw))
    except Exception:
        target_mb = 512
    try:
        max_samples = max(1, int(max_samples_raw))
    except Exception:
        max_samples = 4096
    bytes_per_sample = max(1, (in_features + out_features) * torch.float32.itemsize * 3)
    budget_bytes = target_mb * 1024 * 1024
    block_size = max(1, budget_bytes // bytes_per_sample)
    return max(1, min(batch_size, max_samples, block_size))


def linear_compute_frontier_aggregate_score(
    inputs_source: Union[torch.Tensor, TensorReplayProvider],
    saved_neuron_gradients: torch.Tensor,
    scale: float,
    weights: torch.Tensor,
    device: torch.device,
    *,
    abs_aggregation_mode: str = "exact_abs",
    compute_neuron_metrics: bool = False,
    compute_product_metrics: bool = False,
) -> Tuple:
    """
    Canonical score-only frontier aggregate linear scorer.

    This keeps the frontier aggregate semantics but uses a block heuristic tuned for
    GEMM throughput rather than the generic frontier chunk heuristic, which
    collapses large FC layers to batch_size=1 chunks.
    """
    with torch.no_grad():
        out_features, in_features = weights.shape
        batch_size = _replay_batch_size(inputs_source, saved_neuron_gradients.size(0))
        block_size = _choose_frontier_aggregate_linear_block_size(
            batch_size=batch_size,
            in_features=in_features,
            out_features=out_features,
        )
        frontier_abs = is_frontier_aggregate_abs_mode(abs_aggregation_mode)

        kernel_scores_total = torch.zeros_like(weights, dtype=torch.float32, device=device)
        activating_scores_total = torch.zeros_like(weights, dtype=torch.float32, device=device)
        inhibiting_scores_total = torch.zeros_like(weights, dtype=torch.float32, device=device)
        scores_signed_acts_total = torch.zeros_like(weights, dtype=torch.float32, device=device)
        abs_kernel_scores_total = torch.zeros_like(weights, dtype=torch.float32, device=device)
        gradients_total = torch.zeros_like(weights, dtype=torch.float32, device=device)
        abs_gradients_total = torch.zeros_like(weights, dtype=torch.float32, device=device)
        if compute_neuron_metrics:
            neuron_scores_total = torch.zeros(weights.size(0), dtype=torch.float32, device=device)
            abs_neuron_scores_total = torch.zeros_like(neuron_scores_total)
            neuron_scores_sq_total = torch.zeros_like(neuron_scores_total)
            abs_neuron_scores_sq_total = torch.zeros_like(neuron_scores_total)
            product_scores_total = (
                torch.zeros_like(weights, dtype=torch.float32, device=device)
                if compute_product_metrics
                else None
            )
            abs_product_scores_total = (
                torch.zeros_like(weights, dtype=torch.float32, device=device)
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

        for start in range(0, batch_size, block_size):
            end = min(start + block_size, batch_size)
            inputs_chunk = _fetch_replay_slice(
                inputs_source,
                start,
                end,
                device=device,
                dtype=torch.float32,
            )
            if inputs_chunk.dim() > 2:
                inputs_chunk = inputs_chunk.reshape(inputs_chunk.size(0), -1)
            grads_chunk = saved_neuron_gradients[start:end].to(device=device, dtype=torch.float32)
            outputs_chunk = (
                torch.nn.functional.linear(inputs_chunk, weights, bias=None)
                if compute_neuron_metrics
                else None
            )
            chunk_stats = _linear_reduce_exact_chunk(
                inputs_chunk,
                grads_chunk,
                weights,
                scale=scale,
                outputs_chunk=outputs_chunk,
                compute_neuron_metrics=compute_neuron_metrics,
                compute_product_metrics=compute_product_metrics,
                abs_aggregation_mode=abs_aggregation_mode,
            )
            kernel_scores_total += chunk_stats["kernel_scores_sum"]
            activating_scores_total += chunk_stats["activating_scores_sum"]
            inhibiting_scores_total += chunk_stats["inhibiting_scores_sum"]
            scores_signed_acts_total += chunk_stats["scores_signed_acts_sum"]
            abs_kernel_scores_total += chunk_stats["abs_scores_sum"]
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
            abs_kernel_scores_total = kernel_scores_total.abs()
            activating_scores_total = torch.relu(kernel_scores_total)
            inhibiting_scores_total = -torch.relu(-kernel_scores_total)
            scale_value, scale_tensor, _ = _resolve_scale_tensors(scale, device=device)
            scale_sign = -1.0 if scale_value < 0 else 1.0
            scores_signed_acts_total = weights * gradients_total.abs() * scale_sign
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

        grad_non_zeroes = torch.tensor(torch.count_nonzero(gradients_total).item(), device=device)
        dummy = torch.zeros(1, device=device)
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
            [],
        )


def _choose_oldfast_linear_block_size(
    *,
    batch_size: int,
    in_features: int,
    out_features: int,
) -> int:
    return _choose_frontier_aggregate_linear_block_size(
        batch_size=batch_size,
        in_features=in_features,
        out_features=out_features,
    )


def linear_compute_oldfast_score_aggregated(
    inputs_source: Union[torch.Tensor, TensorReplayProvider],
    saved_neuron_gradients: torch.Tensor,
    scale: float,
    weights: torch.Tensor,
    device: torch.device,
    *,
    abs_aggregation_mode: str = "exact_abs",
) -> Tuple:
    return linear_compute_frontier_aggregate_score(
        inputs_source=inputs_source,
        saved_neuron_gradients=saved_neuron_gradients,
        scale=scale,
        weights=weights,
        device=device,
        abs_aggregation_mode=abs_aggregation_mode,
    )


def _linear_reduce_exact_chunk(
    inputs_chunk: torch.Tensor,
    grads_chunk: torch.Tensor,
    weights: torch.Tensor,
    *,
    scale: float,
    outputs_chunk: Optional[torch.Tensor] = None,
    compute_neuron_metrics: bool = True,
    compute_product_metrics: bool = True,
    abs_aggregation_mode: str = "exact_abs",
) -> Dict[str, Optional[torch.Tensor]]:
    abs_mode = normalize_abs_aggregation_mode(abs_aggregation_mode)
    frontier_abs = is_frontier_aggregate_abs_mode(abs_mode)
    _, scale_tensor, abs_scale_tensor = _resolve_scale_tensors(scale, device=weights.device)

    signed_grad_unscaled = grads_chunk.t().matmul(inputs_chunk)
    gradients_sum = signed_grad_unscaled / scale_tensor
    kernel_scores_sum = weights * gradients_sum
    if frontier_abs:
        abs_grad_unscaled = None
        abs_gradients_sum = gradients_sum.abs()
        abs_scores_sum = kernel_scores_sum.abs()
        activating_scores_sum = torch.relu(kernel_scores_sum)
        inhibiting_scores_sum = -torch.relu(-kernel_scores_sum)
        scores_signed_acts_sum = (weights * signed_grad_unscaled.abs()) / scale_tensor
    else:
        abs_grad_unscaled = grads_chunk.abs().t().matmul(inputs_chunk.abs())
        abs_gradients_sum = abs_grad_unscaled / abs_scale_tensor
        abs_scores_sum = weights.abs() * abs_gradients_sum
        activating_scores_sum = 0.5 * (abs_scores_sum + kernel_scores_sum)
        inhibiting_scores_sum = 0.5 * (kernel_scores_sum - abs_scores_sum)
        scores_signed_acts_sum = (weights * abs_grad_unscaled) / scale_tensor

    sample_sums: Optional[torch.Tensor] = None
    sample_abs_sums: Optional[torch.Tensor] = None
    neuron_scores_sum: Optional[torch.Tensor] = None
    abs_neuron_scores_sum: Optional[torch.Tensor] = None
    neuron_scores_sq_sum: Optional[torch.Tensor] = None
    abs_neuron_scores_sq_sum: Optional[torch.Tensor] = None
    product_scores_sum: Optional[torch.Tensor] = None
    abs_product_scores_sum: Optional[torch.Tensor] = None

    if outputs_chunk is not None:
        grad_factor = grads_chunk / scale_tensor
        sample_sums = outputs_chunk * grad_factor
        sample_abs_sums = sample_sums.abs()

        if compute_neuron_metrics:
            neuron_scores_sum = sample_sums.sum(dim=0)
            abs_neuron_scores_sum = (
                neuron_scores_sum.abs() if frontier_abs else sample_abs_sums.sum(dim=0)
            )
            neuron_scores_sq_sum = sample_sums.square().sum(dim=0)
            abs_neuron_scores_sq_sum = (
                abs_neuron_scores_sum.square()
                if frontier_abs and abs_neuron_scores_sum is not None
                else neuron_scores_sq_sum.clone()
            )

            if compute_product_metrics:
                product_row_factor = sample_sums * grad_factor
                product_scores_sum = product_row_factor.t().matmul(inputs_chunk) * weights

                if frontier_abs:
                    abs_product_scores_sum = product_scores_sum.abs()
                else:
                    abs_grad_factor = grads_chunk.abs() / abs_scale_tensor
                    abs_product_row_factor = sample_abs_sums * abs_grad_factor
                    abs_product_scores_sum = (
                        abs_product_row_factor.t().matmul(inputs_chunk.abs()) * weights.abs()
                    )

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
    }


def linear_compute_kernel_score(
    weight_acts: Optional[torch.Tensor],
    saved_actweight_gradients: Optional[torch.Tensor],
        in_cat_list: Optional[List[int]],
    score_neg: bool,
    absolute: bool,
    normalize_features_per_image: bool,
    full_acts,
    pre_softmax_vector,
    weight_norm_type: str,
    device: torch.device,
    scale,
    original_activations_mean,
    weights,
    grad_to_final_acts,
    final_acts,
    label,
    target_unit,
    neuron_outputs,
    neuron_gradients,
    target_unit_acts,
    cross_entropy_loss,
    score_post_act_function,
    *,
    inputs: Optional[torch.Tensor] = None,
    inputs_provider: Optional[TensorReplayProvider] = None,
    saved_neuron_gradients: Optional[torch.Tensor] = None,
    bitpack_sink: Optional[Callable[[torch.Tensor], None]] = None,
    bias: Optional[torch.Tensor] = None,
    validate_frontier: bool = False,
    stats_callback: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    return_sample_sums: bool = False,
    debug_layer_name: Optional[str] = None,
    chunk_recorder: Optional[object] = None,
    bie_logging: bool = False,
) -> Tuple:
    
    # print('starting linear compute')
    with torch.no_grad():
        
        if inputs is None and inputs_provider is not None:
            inputs = _ReplayTensorProxy(inputs_provider, device=device, dtype=torch.float32)

        streaming = weight_acts is None or saved_actweight_gradients is None
        if streaming:
            if inputs is None or saved_neuron_gradients is None:
                raise ValueError("Streaming mode requires `inputs` and `saved_neuron_gradients`.")
            sample_dim = inputs.size(0)
            out_features, in_features = weights.shape
        else:
            sample_dim, out_features, in_features = weight_acts.shape
        if saved_neuron_gradients is not None:
            saved_neuron_gradients = saved_neuron_gradients.to(dtype=torch.float32)

        # Convert in_cat_list to tensor
        in_cat_indices = torch.tensor(in_cat_list, dtype=torch.long, device=device) if in_cat_list else torch.tensor([], dtype=torch.long, device=device)


        # Initialize accumulators for all_scores
        zero_kernel = torch.zeros((out_features, in_features), device=device, dtype=torch.float32)
        all_kernel_scores = zero_kernel.clone()
        all_loss_contribution = zero_kernel.clone()
        all_abs_loss_contribution = zero_kernel.clone() if absolute else None
        all_self_products = torch.zeros((out_features, in_features), device=device, dtype=torch.float32)
        all_cross_products = create_sparse_output_tensor_dict(out_features, in_features, device)
        all_bie_base_scores: Optional[torch.Tensor] = None
        all_bie_base_abs_scores: Optional[torch.Tensor] = None
        all_bie_boundary_scores: Optional[torch.Tensor] = None
        all_bie_boundary_abs_scores: Optional[torch.Tensor] = None
        cat_bie_base_scores: Optional[torch.Tensor] = None
        cat_bie_base_abs_scores: Optional[torch.Tensor] = None
        cat_bie_boundary_scores: Optional[torch.Tensor] = None
        cat_bie_boundary_abs_scores: Optional[torch.Tensor] = None
        all_activating_kernel_scores = zero_kernel.clone()
        all_inhibiting_kernel_scores = zero_kernel.clone()
        all_kernel_scores_signed_acts = zero_kernel.clone()
        all_abs_kernel_scores = zero_kernel.clone() if absolute else None
        all_gradients_component = zero_kernel.clone()
        all_abs_gradients_component = zero_kernel.clone() if absolute else None
        grad_non_zeroes = zero_kernel.clone()
        # empty tensor
        chosen_weight_tensor = torch.zeros((0,), device=device)
        if return_sample_sums:
            sample_sum_accum = torch.zeros((sample_dim, out_features), device=device, dtype=torch.float32)
            sample_abs_sum_accum = torch.zeros((sample_dim, out_features), device=device, dtype=torch.float32)
        else:
            sample_sum_accum = None
            sample_abs_sum_accum = None

        if bias is not None:
            bias = bias.to(device=device, dtype=torch.float32)

        frontier_lhs = torch.zeros((), device=device, dtype=torch.float64) if validate_frontier else None
        frontier_rhs = torch.zeros_like(frontier_lhs) if validate_frontier else None
        frontier_bias = torch.zeros_like(frontier_lhs) if (validate_frontier and bias is not None) else None

        compute_neuron_metrics = neuron_outputs is not None and neuron_gradients is not None

        if compute_neuron_metrics:
            neuron_outputs = neuron_outputs.to(device=device, dtype=torch.float32)
            neuron_gradients = neuron_gradients.to(device=device, dtype=torch.float32)

        scale_value = float(scale) if not isinstance(scale, (float, int)) else float(scale)
        if abs(scale_value) < 1e-12:
            scale_value = 1.0
        scale_tensor = torch.as_tensor(scale_value, device=device, dtype=torch.float32)
        layer_corr_scores: Optional[List[torch.Tensor]] = [] if debug_layer_name else None
        layer_corr_objectives: Optional[List[torch.Tensor]] = [] if debug_layer_name else None
        bie_logging_enabled = bool(bie_logging and debug_layer_name)
        bie_layer_corr_scores: Optional[List[torch.Tensor]] = [] if bie_logging_enabled else None
        bie_layer_corr_objectives: Optional[List[torch.Tensor]] = [] if bie_logging_enabled else None
        bie_base_total = 0.0
        bie_boundary_total = 0.0

        weight_shape = weights.shape

        if compute_neuron_metrics:
            neuron_scores = (neuron_outputs * neuron_gradients / scale_tensor).sum(dim=1).unsqueeze(dim=1)
            neuron_scores_squared = neuron_scores * neuron_scores
            abs_neuron_scores = torch.abs(neuron_scores)
            abs_neuron_scores_squared = abs_neuron_scores * abs_neuron_scores
            all_product_scores = torch.zeros(weight_shape, dtype=torch.float32, device=device)
            all_abs_product_scores = (
                torch.zeros(weight_shape, dtype=torch.float32, device=device) if absolute else None
            )
        else:
            neuron_scores = None
            neuron_scores_squared = None
            abs_neuron_scores = None
            abs_neuron_scores_squared = None
            all_product_scores = None
            all_abs_product_scores = None




    
        chunk_size = choose_chunk_size(out_features, in_features)
        corrs = []
        list_of_bitpacked = []
        for chunk_idx, i in enumerate(range(0, sample_dim, chunk_size)):
            end = min(i + chunk_size, sample_dim)

            grad_batch = saved_neuron_gradients[i:end].to(dtype=torch.float32, device=device)

            if streaming:
                inputs_batch = inputs[i:end].to(dtype=torch.float32, device=device)
                weight_acts_batch = torch.einsum('bi,oi->boi', inputs_batch, weights)
            else:
                weight_acts_batch = weight_acts[i:end].to(dtype=torch.float32, device=device)
                if inputs is not None:
                    inputs_batch = inputs[i:end].to(dtype=torch.float32, device=device)
                else:
                    weight_expanded = weights.unsqueeze(0).expand_as(weight_acts_batch)
                    mask = weight_expanded != 0
                    divided = torch.zeros_like(weight_acts_batch)
                    divided[mask] = weight_acts_batch[mask] / weight_expanded[mask]
                    counts = mask.float().sum(dim=1).clamp_min(1.0)
                    inputs_batch = divided.sum(dim=1) / counts


            if compute_neuron_metrics:
                neuron_scores_batch = neuron_scores[i:end].to(dtype=torch.float32, device=device)
                abs_neuron_scores_batch = abs_neuron_scores[i:end].to(dtype=torch.float32, device=device)
            else:
                neuron_scores_batch = None
                abs_neuron_scores_batch = None
            # neuron_scores_squared_batch = neuron_scores_squared[i:end].to(dtype=torch.float32, device=device)
            # abs_neuron_scores_squared_batch = abs_neuron_scores_squared[i:end].to(dtype=torch.float32, device=device)

            need_outputs = (
                compute_neuron_metrics
                or validate_frontier
                or stats_callback is not None
                or chunk_recorder is not None
            )
            outputs_linear = (
                torch.nn.functional.linear(inputs_batch, weights, bias=None) if need_outputs else None
            )

            if chunk_recorder is not None:
                _safe_chunk_stage(
                    chunk_recorder,
                    "_stage_boundary_chunk_inputs",
                    chunk_idx,
                    i,
                    end,
                    inputs_batch,
                    outputs_linear,
                )

            if chunk_recorder is not None:
                _safe_chunk_stage(
                    chunk_recorder,
                    "_stage_boundary_chunk_grads",
                    chunk_idx,
                    i,
                    end,
                    grad_batch,
                )

            (
                boundary_inputs_chunk,
                boundary_pre_chunk,
                boundary_grads_chunk,
            ) = _fetch_boundary_chunk_tensors(chunk_recorder, chunk_idx, device)

            grad_to_final_acts_batch = grad_to_final_acts[i:end] if grad_to_final_acts is not None else None
            final_acts_batch = final_acts[i:end] if final_acts is not None else None
            label_batch = label[i:end] if label is not None else None
            unit_act_batch = full_acts[i:end] if full_acts is not None else None
            pre_softmax_vector_batch = pre_softmax_vector[i:end] if pre_softmax_vector is not None else None
            target_unit_acts_batch = target_unit_acts[i:end] if target_unit_acts is not None else None
            cross_entropy_loss_batch = cross_entropy_loss[i:end] if cross_entropy_loss is not None else None

            # Compute chunk scores
            chunk_scores = _compute_chunk_scores(
                weight_acts_batch,
                grad_batch,
                inputs_batch,
                normalize_features_per_image,
                unit_act_batch,
                pre_softmax_vector_batch,
                weight_norm_type,
                original_activations_mean,
                weights,
                scale,
                grad_to_final_acts_batch,
                final_acts_batch,
                label_batch,
                target_unit,
                target_unit_acts_batch,
                cross_entropy_loss_batch,
                score_post_act_function,
                boundary_inputs_chunk=boundary_inputs_chunk,
                boundary_pre_chunk=boundary_pre_chunk,
                boundary_grads_chunk=boundary_grads_chunk,
            )

            _release_boundary_chunk_tensors(chunk_recorder, chunk_idx)

            base_scores_chunk = chunk_scores.get("base_scores")
            if debug_layer_name and torch.is_tensor(base_scores_chunk):
                per_sample_scores = base_scores_chunk.reshape(base_scores_chunk.size(0), -1).sum(dim=1)
                objective_vec = _flatten_objective_vec(target_unit_acts_batch)
                _maybe_record_layer_corr_sample(
                    layer_corr_scores,
                    layer_corr_objectives,
                    per_sample_scores,
                    objective_vec,
                )
            else:
                objective_vec = _flatten_objective_vec(target_unit_acts_batch)

            bie_base_chunk = chunk_scores.get("bie_base_scores")
            bie_boundary_chunk = chunk_scores.get("bie_overshoot_scores")
            if bie_logging_enabled and objective_vec is not None:
                bie_per_sample: Optional[torch.Tensor] = None
                if torch.is_tensor(bie_base_chunk):
                    bie_base_total += float(bie_base_chunk.sum().item())
                    bie_per_sample = bie_base_chunk.sum(dim=(1, 2))
                if torch.is_tensor(bie_boundary_chunk):
                    bie_boundary_total += float(bie_boundary_chunk.sum().item())
                    boundary_totals = bie_boundary_chunk.sum(dim=(1, 2))
                    bie_per_sample = (
                        boundary_totals
                        if bie_per_sample is None
                        else (bie_per_sample + boundary_totals)
                    )
                if bie_per_sample is not None:
                    _maybe_record_layer_corr_sample(
                        bie_layer_corr_scores,
                        bie_layer_corr_objectives,
                        bie_per_sample,
                        objective_vec,
                    )

            # chosen_weight_tensor = torch.cat([chosen_weight_tensor, chunk_scores['tracked_weight']], dim=0)
            chosen_weight_tensor = None

            if stats_callback is not None and target_unit_acts_batch is not None:
                phi_totals = chunk_scores['base_scores'].sum(dim=(1, 2))
                if bias is not None:
                    phi_totals = phi_totals + (bias * grad_batch).sum(dim=1)
                stats_callback(phi_totals, target_unit_acts_batch)

            grad_w_chunk = torch.einsum('bo,bi->boi', grad_batch / scale_tensor, inputs_batch)

            bie_base_chunk = chunk_scores.get("bie_base_scores")
            if torch.is_tensor(bie_base_chunk):
                bie_base_scaled = bie_base_chunk / scale_tensor
                bie_base_sum = bie_base_scaled.sum(dim=0)
                bie_base_abs = bie_base_scaled.abs().sum(dim=0)
                if all_bie_base_scores is None:
                    all_bie_base_scores = bie_base_sum.clone()
                    all_bie_base_abs_scores = bie_base_abs.clone()
                else:
                    all_bie_base_scores += bie_base_sum
                    if all_bie_base_abs_scores is not None:
                        all_bie_base_abs_scores += bie_base_abs
            bie_boundary_chunk = chunk_scores.get("bie_overshoot_scores")
            if torch.is_tensor(bie_boundary_chunk):
                bie_boundary_scaled = bie_boundary_chunk / scale_tensor
                bie_boundary_sum = bie_boundary_scaled.sum(dim=0)
                bie_boundary_abs = bie_boundary_scaled.abs().sum(dim=0)
                if all_bie_boundary_scores is None:
                    all_bie_boundary_scores = bie_boundary_sum.clone()
                    all_bie_boundary_abs_scores = bie_boundary_abs.clone()
                else:
                    all_bie_boundary_scores += bie_boundary_sum
                    if all_bie_boundary_abs_scores is not None:
                        all_bie_boundary_abs_scores += bie_boundary_abs

            # Accumulate scores based on score_neg flag
            if score_neg:
                corrs.append(chunk_scores['corr'])
                base_scores_chunk = chunk_scores['base_scores']
                # print(((chunk_scores['base_scores']) != 0).sum(dim=(1,2))/((chunk_scores['base_scores']).shape[1]*(chunk_scores['base_scores']).shape[2]))
                # print(((chunk_scores['base_scores']) != 0).sum(dim=(0,1,2)))#/
                # print((chunk_scores['base_scores']).numel())
                # print(((chunk_scores['base_scores'])<0).sum(dim=(1,2))/((chunk_scores['base_scores']).shape[1]*(chunk_scores['base_scores']).shape[2]))
                # print(((chunk_scores['base_scores'])<0).sum(dim=(0,1,2))/(chunk_scores['base_scores']).numel())

                # save_dir = '<a local folder>'
                # if start_idx == 0:
                #     scores = chunk_scores['base_scores'][0]
                #     scores_mask = (scores != 0).any(dim=1)

                #     nonzero_scores = scores_mask.nonzero(as_tuple=True)[0]
                        
                #     if nonzero_scores.numel() == 0:
                #         raise ValueError("No common output channels satisfying the condition were found.")
                #     save_row_distributions_2d(chunk_scores['activations'], chunk_scores['gradients'], save_dir, start_idx, in_features, out_features, unit = random.choice(nonzero_scores.tolist()))
                    # save_row_distributions(chunk_scores['base_scores'], save_dir, start_idx, in_features, out_features, unit = random.choice(nonzero_scores.tolist()))
                    # del scores, scores_mask, nonzero_scores   

                # print('nonzero fraction of chunk_scores', ((chunk_scores['base_scores']) != 0).sum(dim=(0,1,2))/(chunk_scores['base_scores']).numel())
                # all_self_products += ((chunk_scores['base_scores'])**2).sum(dim=0)
                # print('cross products')
                # all_cross_products = update_sparse_cross_products(all_cross_products, chunk_scores['base_scores'], device)
                # for out_dict_key, out_dict in all_cross_products.items():
                #     if out_dict_key == 0:
                #         sparsity_dict = compute_sparsity_detailed(out_dict)
                #         print(out_dict_key, out_dict.shape, sparsity_dict)

                if COMPUTE_BITPACK:
                    flat_mask = base_scores_chunk.abs().reshape(base_scores_chunk.size(0), -1) > 0
                    packed_rows = pack_boolean_rows(flat_mask)
                    for row in packed_rows:
                        list_of_bitpacked.append(row.cpu())

                if absolute and all_abs_kernel_scores is not None:
                    all_abs_kernel_scores += torch.sum(base_scores_chunk.abs(), dim=0)
                    if compute_neuron_metrics and all_abs_product_scores is not None and abs_neuron_scores_batch is not None:
                        all_abs_product_scores += torch.sum(
                            base_scores_chunk.abs() * abs_neuron_scores_batch.unsqueeze(-1), dim=0
                        )
                all_activating_kernel_scores += torch.sum(torch.relu(base_scores_chunk), dim=0)
                all_inhibiting_kernel_scores += torch.sum(-torch.relu(-base_scores_chunk), dim=0)
                grad_sign = grad_w_chunk.sign()
                all_kernel_scores_signed_acts += torch.sum(base_scores_chunk * grad_sign, dim=0)
                
                # neuron_scores_squared += torch.sum(chunk_scores['base_scores']**2, dim=0)
                all_kernel_scores += torch.sum(base_scores_chunk, dim=0)
                if compute_neuron_metrics and all_product_scores is not None and neuron_scores_batch is not None:
                    all_product_scores += torch.sum(base_scores_chunk * neuron_scores_batch.unsqueeze(-1), dim=0)
                # del chunk_scores['base_scores']
                if chunk_scores['loss_change'] is not None:
                    all_loss_contribution += chunk_scores['loss_change'].sum(dim=0)
                    if absolute and all_abs_loss_contribution is not None:
                        all_abs_loss_contribution += chunk_scores['loss_change'].abs().sum(dim=0)
                else:
                    all_loss_contribution = None
                    all_abs_loss_contribution = None

                if sample_sum_accum is not None:
                    sample_summary = base_scores_chunk.sum(dim=2)
                    sample_sum_accum[i:end] = sample_summary
                    if sample_abs_sum_accum is not None:
                        sample_abs_sum_accum[i:end] = sample_summary.abs()
            # else:
            #     # all_kernel_scores += (chunk_scores['relu_scores']).sum(dim=0)


            # Accumulate other components
            all_gradients_component += grad_w_chunk.sum(dim=0)
            if absolute and all_abs_gradients_component is not None:
                all_abs_gradients_component += grad_w_chunk.abs().sum(dim=0)

            if validate_frontier:
                grad_weight_true = torch.einsum('bo,bi->oi', grad_batch, inputs_batch)
                outputs_for_check = outputs_linear if outputs_linear is not None else torch.nn.functional.linear(inputs_batch, weights, bias=None)
                frontier_lhs += (weights * grad_weight_true).sum().to(torch.float64)
                frontier_rhs += (outputs_for_check * grad_batch).sum().to(torch.float64)
                if frontier_bias is not None:
                    frontier_bias += (bias * grad_batch.sum(dim=0)).sum().to(torch.float64)

            grad_non_zeroes += torch.sum(chunk_scores['grad_nonzero'], dim=0)
            del chunk_scores['grad_nonzero']

            # print('mean weights', all_kernel_scores/(end-i))
            # print('mean neurons', neuron_scores_batch.sum(dim=0)/(end-i))
            # print(neuron_scores_batch.sum(dim=0).shape)
            # cov_xY = torch.sum(chunk_scores['base_scores']*neuron_scores_batch.unsqueeze(-1), dim=0)/(scale*(end-i)) - (torch.sum(chunk_scores['base_scores'], dim=0)/(scale*(end-i)))*(neuron_scores_batch.sum(dim=0).unsqueeze(-1)/(end-i))
            # # print(cov_xY.shape)
            # # print('covaraince x,Y', cov_xY)
            # var_y = neuron_scores_squared_batch.sum(dim=0)/(end-i) - (neuron_scores_batch.sum(dim=0)/(end-i))**2
            # print(var_y.shape)
            # print('variance of Y', var_y)
            # if var_y[1] != 0:
            #     # print('max fraction of varaince', (cov_xY/(var_y.unsqueeze(-1)))[1].abs().max())
            #     # print('min fraction of varaince', (cov_xY/(var_y.unsqueeze(-1)))[1].abs().min())
            #     # print('number of zero fractions', ((cov_xY/(var_y.unsqueeze(-1))) == 0).sum(dim=1)/((cov_xY/(var_y.unsqueeze(-1))).shape[1]))
            #     print('kernel means', (all_kernel_scores[1]).mean())
            #     print('only zero covariance means', (all_kernel_scores[1][((cov_xY/(var_y.unsqueeze(-1)))[1] == 0)]).mean())
            #     print('kernel scores at zero points', (all_kernel_scores[1][((cov_xY/(var_y.unsqueeze(-1)))[1] == 0)]).sum())

            # else:
            #     print('nan!')
            # print('sum of variance fractions', (cov_xY/(var_y.unsqueeze(-1))).sum(dim=1))
            del chunk_scores['base_scores']
        # print('all_gradients_component',(all_gradients_component[target_unit]/65536.0078)[(all_gradients_component[target_unit]/65536.0078).abs()!=1])
        # print('abs loss contribution', all_abs_loss_contribution[target_unit])
        _finalize_layer_corr_logging(debug_layer_name, layer_corr_scores, layer_corr_objectives)
        if bie_logging_enabled:
            _finalize_layer_bie_corr_logging(
                debug_layer_name,
                bie_layer_corr_scores,
                bie_layer_corr_objectives,
                bie_base_total,
                bie_boundary_total,
            )

        if DEBUG_SCORING and corrs:
            logger.debug(
                "[linear-debug] corr average=%.6f",
                torch.stack(corrs).mean().item(),
            )
        # Initialize category-specific results
        if in_cat_indices.numel() > 0:
            cat_kernel_scores = torch.zeros(weight_shape, dtype=torch.float32, device=device)
            cat_product_scores = (
                torch.zeros(weight_shape, dtype=torch.float32, device=device) if compute_neuron_metrics else None
            )
            cat_loss_contribution = torch.zeros(weight_shape, dtype=torch.float32, device=device)
            cat_abs_loss_contribution = (
                torch.zeros(weight_shape, dtype=torch.float32, device=device) if absolute else None
            )
            cat_activating_kernel_scores = torch.zeros(weight_shape, dtype=torch.float32, device=device)
            cat_inhibiting_kernel_scores = torch.zeros(weight_shape, dtype=torch.float32, device=device)
            cat_kernel_scores_signed_acts = torch.zeros(weight_shape, dtype=torch.float32, device=device)
            cat_abs_kernel_scores = (
                torch.zeros(weight_shape, dtype=torch.float32, device=device) if absolute else None
            )
            cat_abs_product_scores = (
                torch.zeros(weight_shape, dtype=torch.float32, device=device)
                if (absolute and compute_neuron_metrics)
                else None
            )
            cat_gradients_component = torch.zeros(weight_shape, dtype=torch.float32, device=device)
            cat_abs_gradients_component = (
                torch.zeros(weight_shape, dtype=torch.float32, device=device) if absolute else None
            )

            cat_chunk = max(1, choose_chunk_size(out_features, in_features))

            for batch_indices in torch.split(in_cat_indices, cat_chunk):
                # Extract batch data
                grad_batch = saved_neuron_gradients[batch_indices].to(dtype=torch.float32, device=device)

                if inputs is not None:
                    inputs_batch = _fetch_replay_indices(
                        inputs,
                        batch_indices,
                        device=device,
                        dtype=torch.float32,
                    )
                    weight_acts_batch = torch.einsum('bi,oi->boi', inputs_batch, weights)
                else:
                    weight_acts_batch = weight_acts[batch_indices].to(dtype=torch.float32, device=device)
                    weight_expanded = weights.unsqueeze(0).expand_as(weight_acts_batch)
                    mask = weight_expanded != 0
                    divided = torch.zeros_like(weight_acts_batch)
                    divided[mask] = weight_acts_batch[mask] / weight_expanded[mask]
                    counts = mask.float().sum(dim=1).clamp_min(1.0)
                    inputs_batch = divided.sum(dim=1) / counts

                neuron_scores_batch = neuron_scores[batch_indices] if neuron_scores is not None else None
                abs_neuron_scores_batch = abs_neuron_scores[batch_indices] if abs_neuron_scores is not None else None

                grad_to_final_acts_batch = grad_to_final_acts[batch_indices] if grad_to_final_acts is not None else None
                final_acts_batch = final_acts[batch_indices] if final_acts is not None else None
                label_batch = label[batch_indices] if label is not None else None
                unit_act_batch = full_acts[batch_indices] if full_acts is not None else None
                pre_softmax_vector_batch = pre_softmax_vector[batch_indices] if pre_softmax_vector is not None else None
                target_unit_acts_batch = target_unit_acts[batch_indices] if target_unit_acts is not None else None
                cross_entropy_loss_batch = cross_entropy_loss[batch_indices] if cross_entropy_loss is not None else None

                # Compute chunk scores
                chunk_scores = _compute_chunk_scores(
                    weight_acts_batch,
                    grad_batch,
                    inputs_batch,
                    normalize_features_per_image,
                    unit_act_batch,
                    pre_softmax_vector_batch,
                    weight_norm_type,
                    original_activations_mean,
                    weights,
                    scale,
                    grad_to_final_acts_batch,
                    final_acts_batch,
                    label_batch,
                    target_unit,
                    target_unit_acts_batch,
                    cross_entropy_loss_batch,
                    score_post_act_function
                )

                # Accumulate scores based on score_neg flag
                if score_neg:
                    base_scores_chunk = chunk_scores['base_scores']
                    if absolute and cat_abs_kernel_scores is not None:
                        cat_abs_kernel_scores += torch.sum(base_scores_chunk.abs(), dim=0)
                        if compute_neuron_metrics and cat_abs_product_scores is not None and abs_neuron_scores_batch is not None:
                            cat_abs_product_scores += torch.sum(
                                base_scores_chunk.abs() * abs_neuron_scores_batch.unsqueeze(-1), dim=0
                            )
                    cat_activating_kernel_scores += torch.sum(torch.relu(base_scores_chunk), dim=0)
                    cat_inhibiting_kernel_scores += torch.sum(-torch.relu(-base_scores_chunk), dim=0)

                    grad_w_chunk_cat = torch.einsum('bo,bi->boi', grad_batch / scale_tensor, inputs_batch)
                    cat_kernel_scores_signed_acts += torch.sum(base_scores_chunk * grad_w_chunk_cat.sign(), dim=0)
                    cat_kernel_scores += torch.sum(base_scores_chunk, dim=0)
                    if absolute and cat_abs_gradients_component is not None:
                        cat_abs_gradients_component += grad_w_chunk_cat.abs().sum(dim=0)
                    if compute_neuron_metrics and cat_product_scores is not None and neuron_scores_batch is not None:
                        cat_product_scores += torch.sum(base_scores_chunk * neuron_scores_batch.unsqueeze(-1), dim=0)
                    if chunk_scores['loss_change'] is not None:
                        cat_loss_contribution += chunk_scores['loss_change'].sum(dim=0)
                        if absolute and cat_abs_loss_contribution is not None:
                            cat_abs_loss_contribution += chunk_scores['loss_change'].abs().sum(dim=0)

                cat_gradients_component += torch.einsum('bo,bi->oi', grad_batch / scale_tensor, inputs_batch)

                del chunk_scores['base_scores']
                del chunk_scores['grad_nonzero']






        else:
            cat_kernel_scores = cat_loss_contribution = cat_abs_loss_contribution = cat_activating_kernel_scores = cat_inhibiting_kernel_scores = cat_kernel_scores_signed_acts = None
            cat_abs_kernel_scores = cat_gradients_component = None
            cat_abs_gradients_component = None

        # Create dummy tensor
        dummy = torch.zeros(1, device=device)#

        # Combine category-specific and out-of-category scores
        if in_cat_indices.numel() > 0:
            out_kernel_scores =  all_kernel_scores - cat_kernel_scores
            out_abs_kernel_scores = (-cat_abs_kernel_scores + all_abs_kernel_scores) if absolute else None
            out_gradients_component = -cat_gradients_component + all_gradients_component
            out_abs_gradients_component = (-cat_abs_gradients_component + all_abs_gradients_component) if absolute else None
        else:
            out_kernel_scores = all_kernel_scores
            out_abs_kernel_scores = all_abs_kernel_scores if absolute else None
            out_gradients_component = all_gradients_component
            out_abs_gradients_component = all_abs_gradients_component if absolute else None
        



        if cat_abs_kernel_scores is not None:
            if compute_neuron_metrics and cat_product_scores is not None:
                cat_product_scores = cat_product_scores
            else:
                cat_product_scores = None
            if compute_neuron_metrics and absolute and cat_abs_product_scores is not None:
                cat_abs_product_scores = cat_abs_product_scores
            else:
                cat_abs_product_scores = None
            cat_loss_contribution = cat_loss_contribution if cat_loss_contribution is not None else None
            cat_abs_loss_contribution = cat_abs_loss_contribution if cat_abs_loss_contribution is not None else None
            cat_activating_kernel_scores = cat_activating_kernel_scores
            cat_inhibiting_kernel_scores = cat_inhibiting_kernel_scores
            cat_kernel_scores_signed_acts = cat_kernel_scores_signed_acts
            cat_abs_kernel_scores = cat_abs_kernel_scores
            cat_gradients_component = cat_gradients_component
            if absolute and cat_abs_gradients_component is not None:
                cat_abs_gradients_component = cat_abs_gradients_component
            else:
                cat_abs_gradients_component = None

        if absolute and all_abs_kernel_scores is None:
            all_abs_kernel_scores = None
        if compute_neuron_metrics and all_product_scores is None:
            all_product_scores = None
        if compute_neuron_metrics and absolute and all_abs_product_scores is None:
            all_abs_product_scores = None
        all_loss_contribution = all_loss_contribution if all_loss_contribution is not None else None
        all_abs_loss_contribution = all_abs_loss_contribution if all_abs_loss_contribution is not None else None
        if absolute and all_abs_gradients_component is None:
            all_abs_gradients_component = None
        out_kernel_scores = out_kernel_scores
        if absolute and out_abs_kernel_scores is None:
            out_abs_kernel_scores = None
        out_gradients_component = out_gradients_component
        if absolute and out_abs_gradients_component is None:
            out_abs_gradients_component = None
        grad_non_zeroes = grad_non_zeroes

        sample_sum_payload = sample_sum_accum if sample_sum_accum is not None else None
        sample_abs_sum_payload = sample_abs_sum_accum if sample_abs_sum_accum is not None else None
        if sample_abs_sum_payload is None and isinstance(sample_sum_payload, torch.Tensor):
            sample_abs_sum_payload = sample_sum_payload.abs()

        # for out_dict_key, out_dict in all_cross_products.items():
        #     all_cross_products[out_dict_key] = (out_dict/scale**2)
        # all_self_products = (all_self_products/scale**2)
        all_self_products = None
        all_cross_products = None

        if compute_neuron_metrics:
            abs_neuron_scores = abs_neuron_scores.sum(dim=0)
            neuron_scores = neuron_scores.sum(dim=0)
            neuron_scores_squared = neuron_scores_squared.sum(dim=0)
            abs_neuron_scores_squared = abs_neuron_scores_squared.sum(dim=0)
        else:
            abs_neuron_scores = None
            neuron_scores = None
            neuron_scores_squared = None
            abs_neuron_scores_squared = None
            all_product_scores = None
            all_abs_product_scores = None

        if validate_frontier and frontier_lhs is not None and frontier_rhs is not None:
            lhs_total = frontier_lhs
            rhs_total = frontier_rhs
            if frontier_bias is not None:
                lhs_total = lhs_total + frontier_bias
                rhs_total = rhs_total + frontier_bias
            if not torch.allclose(lhs_total, rhs_total, rtol=1e-3, atol=1e-3):
                raise AssertionError(
                    f"Linear (per-sample) frontier completeness failed: phi={lhs_total.item():.6f}, z·m={rhs_total.item():.6f}"
                )

        return (
            cat_kernel_scores,
            cat_activating_kernel_scores,
            cat_inhibiting_kernel_scores,
            cat_kernel_scores_signed_acts,
            cat_abs_kernel_scores,
            cat_gradients_component,
            cat_abs_gradients_component,
            cat_loss_contribution,
            cat_abs_loss_contribution,
            cat_bie_base_scores,
            cat_bie_base_abs_scores,
            cat_bie_boundary_scores,
            cat_bie_boundary_abs_scores,
            sample_sum_payload,
            sample_abs_sum_payload,
            dummy,
            dummy,
            dummy,
            all_kernel_scores,
            all_activating_kernel_scores,
            all_inhibiting_kernel_scores,
            all_kernel_scores_signed_acts,
            all_abs_kernel_scores,
            all_gradients_component,
            all_abs_gradients_component,
            all_loss_contribution,
            all_abs_loss_contribution,
            all_bie_base_scores,
            all_bie_base_abs_scores,
            all_bie_boundary_scores,
            all_bie_boundary_abs_scores,
            neuron_scores,
            abs_neuron_scores,
            neuron_scores_squared,
            abs_neuron_scores_squared,
            all_product_scores,
            all_abs_product_scores,
            dummy,
            dummy,
            chosen_weight_tensor,
            grad_non_zeroes,
            all_self_products,
            all_cross_products,
            list_of_bitpacked,
        )


def linear_compute_kernel_score_aggregated(
    inputs_source: Union[torch.Tensor, TensorReplayProvider],
    saved_neuron_gradients: torch.Tensor,
    scale: float,
    weights: torch.Tensor,
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
) -> Tuple:
    """
    Aggregated frontier scoring without storing full per-sample tensors.
    Results match the per-sample accumulation while keeping memory bounded.
    """
    if (
        bitpack_sink is None
        and not validate_frontier
        and target_acts is None
        and stats_callback is None
        and not return_sample_sums
        and not debug_layer_name
        and not emit_bitpack
    ):
        return linear_compute_frontier_aggregate_score(
            inputs_source=inputs_source,
            saved_neuron_gradients=saved_neuron_gradients,
            scale=scale,
            weights=weights,
            device=device,
            abs_aggregation_mode=abs_aggregation_mode,
            compute_neuron_metrics=compute_neuron_metrics,
        )
    with torch.no_grad():
        frontier_abs = is_frontier_aggregate_abs_mode(abs_aggregation_mode)
        _, scale_tensor, _ = _resolve_scale_tensors(scale, device=device)

        out_features, in_features = weights.shape
        chunk_size = choose_chunk_size(out_features, in_features)
        batch_size = _replay_batch_size(inputs_source, saved_neuron_gradients.size(0))

        kernel_scores_total = torch.zeros_like(weights, dtype=torch.float32, device=device)
        activating_scores_total = torch.zeros_like(weights, dtype=torch.float32, device=device)
        inhibiting_scores_total = torch.zeros_like(weights, dtype=torch.float32, device=device)
        scores_signed_acts_total = torch.zeros_like(weights, dtype=torch.float32, device=device)
        abs_kernel_scores_total = torch.zeros_like(weights, dtype=torch.float32, device=device)
        gradients_total = torch.zeros_like(weights, dtype=torch.float32, device=device)
        abs_gradients_total = torch.zeros_like(weights, dtype=torch.float32, device=device)
        sample_sum_accum = None
        sample_abs_sum_accum = None

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

        sample_sum_accum = None
        sample_abs_sum_accum = None
        if return_sample_sums:
            sample_sum_accum = torch.zeros((batch_size, out_features), dtype=torch.float32, device=device)
            sample_abs_sum_accum = torch.zeros((batch_size, out_features), dtype=torch.float32, device=device)
        layer_corr_scores: Optional[List[torch.Tensor]] = [] if debug_layer_name else None
        layer_corr_objectives: Optional[List[torch.Tensor]] = [] if debug_layer_name else None

        for chunk_idx, start in enumerate(range(0, batch_size, chunk_size)):
            end = min(start + chunk_size, batch_size)
            inputs_chunk = _fetch_replay_slice(
                inputs_source,
                start,
                end,
                device=device,
                dtype=torch.float32,
            )
            if inputs_chunk.dim() > 2:
                inputs_chunk = inputs_chunk.reshape(inputs_chunk.size(0), -1)
            if inputs_chunk.size(1) != in_features:
                raise ValueError(
                    f"linear_compute_kernel_score_aggregated: expected input features {in_features}, "
                    f"got {inputs_chunk.size(1)}"
                )
            grads_chunk = saved_neuron_gradients[start:end].to(device=device, dtype=torch.float32)
            target_chunk = None
            if target_acts is not None:
                target_chunk = target_acts[start:end].to(device=device, dtype=torch.float32)

            need_outputs = (
                compute_neuron_metrics
                or validate_frontier
                or stats_callback is not None
                or return_sample_sums
                or bool(debug_layer_name)
            )
            outputs_chunk = torch.nn.functional.linear(inputs_chunk, weights, bias=None) if need_outputs else None

            chunk_stats = _linear_reduce_exact_chunk(
                inputs_chunk,
                grads_chunk,
                weights,
                scale=scale,
                outputs_chunk=outputs_chunk,
                compute_neuron_metrics=compute_neuron_metrics,
                abs_aggregation_mode=abs_aggregation_mode,
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

            if stats_callback is not None and outputs_chunk is not None:
                phi_rows = sample_sums
                if bias is not None:
                    phi_rows = phi_rows + (grads_chunk / scale_tensor) * bias.unsqueeze(0)
                stats_callback(phi_rows, target_chunk)

            if sample_sums is not None:
                if debug_layer_name:
                    per_sample_scores = sample_sums.sum(dim=1)
                    objective_vec = _flatten_objective_vec(target_chunk)
                    _maybe_record_layer_corr_sample(
                        layer_corr_scores,
                        layer_corr_objectives,
                        per_sample_scores,
                        objective_vec,
                    )
                if sample_sum_accum is not None:
                    sample_sum_accum[start:end] = sample_sums
                    if sample_abs_sum_accum is not None:
                        sample_abs_sum_accum[start:end] = sample_abs_sums

            if validate_frontier and outputs_chunk is not None:
                grad_weight_true = chunk_stats["gradients_sum"]
                frontier_lhs += (weights * grad_weight_true).sum().to(torch.float64)
                frontier_rhs += (outputs_chunk * (grads_chunk / scale_tensor)).sum().to(torch.float64)
                if frontier_bias is not None:
                    frontier_bias += (bias * (grads_chunk / scale_tensor).sum(dim=0)).sum().to(torch.float64)

        _finalize_layer_corr_logging(debug_layer_name, layer_corr_scores, layer_corr_objectives)

        if frontier_abs:
            abs_kernel_scores_total = kernel_scores_total.abs()
            activating_scores_total = torch.relu(kernel_scores_total)
            inhibiting_scores_total = -torch.relu(-kernel_scores_total)
            scale_value = float(scale_tensor.detach().cpu())
            scale_sign = -1.0 if scale_value < 0 else 1.0
            scores_signed_acts_total = weights * gradients_total.abs() * scale_sign
            abs_gradients_total = gradients_total.abs()
            if compute_neuron_metrics:
                abs_neuron_scores_total = neuron_scores_total.abs()
                abs_neuron_scores_sq_total = abs_neuron_scores_total.square()
                abs_product_scores_total = product_scores_total.abs()

        grad_non_zeroes = torch.tensor(torch.count_nonzero(gradients_total).item(), device=device)
        flat_kernel = abs_kernel_scores_total.reshape(abs_kernel_scores_total.size(0), -1)
        if emit_bitpack:
            bitpacked_tensor = pack_boolean_rows(flat_kernel > 0)
            if bitpack_sink is not None and bitpacked_tensor.numel() > 0:
                bitpack_sink(bitpacked_tensor.cpu())
                bitpacked = []
            else:
                bitpacked = [row.cpu() for row in bitpacked_tensor]
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

        if validate_frontier and frontier_lhs is not None and frontier_rhs is not None:
            lhs_total = frontier_lhs
            rhs_total = frontier_rhs
            if frontier_bias is not None:
                lhs_total = lhs_total + frontier_bias
                rhs_total = rhs_total + frontier_bias
            if not torch.allclose(lhs_total, rhs_total, rtol=1e-3, atol=1e-3):
                raise AssertionError(
                    f"Linear frontier completeness failed: phi={lhs_total.item():.6f}, z·m={rhs_total.item():.6f}"
                )

        dummy = torch.zeros(1, device=device)
        zero_vector = torch.zeros((0,), device=device)
        sample_sum_payload = sample_sum_accum if sample_sum_accum is not None else None
        sample_abs_payload = sample_abs_sum_accum if sample_abs_sum_accum is not None else None
        if sample_abs_payload is None and isinstance(sample_sum_payload, torch.Tensor):
            sample_abs_payload = sample_sum_payload.abs()

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
) -> Tuple:
    with torch.no_grad():
        if isinstance(scale, torch.Tensor):
            scale_value = float(scale.detach().cpu())
        else:
            scale_value = float(scale)
        if abs(scale_value) < 1e-12:
            scale_value = 1.0

        out_features = weights.size(0)
        in_features = weights.size(1)
        spatial_elements = weights.shape[2] * weights.shape[3] if weights.dim() == 4 else 1
        chunk_size = choose_chunk_size(out_features, in_features, spatial_elements)
        batch_size = _replay_batch_size(inputs_source, saved_neuron_gradients.size(0))

        weight_shape = weights.shape
        grad_weight_total = torch.zeros(weight_shape, dtype=torch.float32, device=device)

        if bias is not None:
            bias = bias.to(device=device, dtype=torch.float32)

        if compute_neuron_metrics:
            neuron_scores_total = torch.zeros(weights.size(0), dtype=torch.float32, device=device)
            abs_neuron_scores_total = torch.zeros_like(neuron_scores_total)
            neuron_scores_sq_total = torch.zeros_like(neuron_scores_total)
            abs_neuron_scores_sq_total = torch.zeros_like(neuron_scores_total)
        else:
            neuron_scores_total = None
            abs_neuron_scores_total = None
            neuron_scores_sq_total = None
            abs_neuron_scores_sq_total = None

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

            grad_weight_chunk = torch.nn.grad.conv2d_weight(
                inputs_chunk,
                weight_shape,
                grads_chunk,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
            grad_weight_total += grad_weight_chunk

            need_outputs = compute_neuron_metrics or validate_frontier or stats_callback is not None
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

            if compute_neuron_metrics and outputs_chunk is not None:
                sums, abs_sums, sq_sums = fused_scores.reduce_mul(
                    outputs_chunk,
                    grads_chunk,
                    reduce_dims=(0, 2, 3),
                    compute_abs=True,
                    compute_square=True,
                )
                neuron_scores_total += (sums / scale_value).to(device=neuron_scores_total.device)
                if abs_sums is not None:
                    abs_neuron_scores_total += (abs_sums / scale_value).to(device=abs_neuron_scores_total.device)
                else:
                    abs_neuron_scores_total += (sums.abs() / scale_value).to(device=abs_neuron_scores_total.device)
                if sq_sums is not None:
                    sq_vals = (sq_sums / (scale_value ** 2)).to(device=neuron_scores_sq_total.device)
                    neuron_scores_sq_total += sq_vals
                    abs_neuron_scores_sq_total += sq_vals
                else:
                    per_sample_sq = (outputs_chunk * grads_chunk) ** 2
                    reductions = per_sample_sq.sum(dim=(0, 2, 3)) / (scale_value ** 2)
                    neuron_scores_sq_total += reductions.to(device=neuron_scores_sq_total.device)
                    abs_neuron_scores_sq_total += reductions.to(device=abs_neuron_scores_sq_total.device)

            if stats_callback is not None and outputs_chunk is not None:
                phi_rows = (outputs_chunk * grads_chunk).sum(dim=(2, 3)) / scale_value
                if bias is not None:
                    bias_term = grads_chunk.sum(dim=(2, 3)) * bias.view(1, -1) / scale_value
                    phi_rows = phi_rows + bias_term
                stats_callback(phi_rows, target_chunk)

            if validate_frontier:
                frontier_lhs += (weights * grad_weight_chunk).sum().to(torch.float64)
                frontier_rhs += (outputs_chunk * grads_chunk).sum().to(torch.float64)
                if frontier_bias is not None:
                    frontier_bias += (bias * grads_chunk.sum(dim=(0, 2, 3))).sum().to(torch.float64)

        grad_weight = grad_weight_total / scale_value

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

        kernel_scores_sum = weights * grad_weight

        activating_scores = torch.relu(kernel_scores_sum)
        inhibiting_scores = -torch.relu(-kernel_scores_sum)
        abs_kernel_scores = torch.abs(kernel_scores_sum)

        gradients_component = grad_weight
        abs_gradients_component = torch.abs(grad_weight)

        grad_non_zeroes = torch.tensor(torch.count_nonzero(grad_weight).item(), device=device)
        flat_kernel = kernel_scores_sum.abs().reshape(kernel_scores_sum.size(0), -1)
        packed_rows = pack_boolean_rows(flat_kernel > 0)
        if bitpack_sink is not None and packed_rows.numel() > 0:
            bitpack_sink(packed_rows.cpu())
            bitpacked = []
        else:
            bitpacked = [row.cpu() for row in packed_rows]

        if compute_neuron_metrics:
            neuron_scores = neuron_scores_total
            abs_neuron_scores = abs_neuron_scores_total
            neuron_scores_squared = neuron_scores_sq_total
            abs_neuron_scores_squared = abs_neuron_scores_sq_total
            product_scores = kernel_scores_sum * neuron_scores.view(-1, 1, 1, 1)
            abs_product_scores = abs_kernel_scores * abs_neuron_scores.view(-1, 1, 1, 1)
        else:
            neuron_scores = None
            abs_neuron_scores = None
            neuron_scores_squared = None
            abs_neuron_scores_squared = None
            product_scores = None
            abs_product_scores = None

        dummy = torch.zeros(1, device=device)
        zero_vector = torch.zeros((0,), device=device)

        return (
            None, None, None, None, None, None, None, None, None,
            None, None, None, None,
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            kernel_scores_sum,
            activating_scores,
            inhibiting_scores,
            kernel_scores_sum,
            abs_kernel_scores,
            gradients_component,
            abs_gradients_component,
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

__all__ = [
    'linear_compute_kernel_score',
    'linear_compute_kernel_score_aggregated',
    'linear_compute_frontier_aggregate_score',
    'linear_compute_oldfast_score_aggregated',
    '_choose_frontier_aggregate_linear_block_size',
    '_choose_oldfast_linear_block_size',
]
