from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch.nn import functional as F

from packages.numeric_backends.trust_region import (
    TRITON_AVAILABLE,
    trust_region_step_fused,
)
_TRITON_IMPORT_ENABLED = str(
    os.getenv("MECHANISTIC_SYSTEMS_TRITON_IMPORT_ENABLED", os.getenv("CIRCUIT_TRITON_IMPORT_ENABLED", "1"))
).strip().lower() not in {"0", "false", "no", "off"}
_CUDA_AVAILABLE_FOR_TRITON_IMPORT = bool(torch.cuda.is_available())

try:  # pragma: no cover - optional dependency
    if not _TRITON_IMPORT_ENABLED:
        raise ImportError("Triton import disabled by MECHANISTIC_SYSTEMS_TRITON_IMPORT_ENABLED")
    if not _CUDA_AVAILABLE_FOR_TRITON_IMPORT:
        raise ImportError("Triton classwise affine backend requires an active CUDA driver")
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False


__all__ = [
    "ClasswiseAffineFitResult",
    "ReadoutSpaceAffineFitResult",
    "BatchedReadoutSpaceAffineFitResult",
    "classwise_triton_available",
    "shared_readout_triton_available",
    "compute_classwise_pos_weight",
    "compute_binary_target_matrix_pos_weight",
    "compute_selected_pos_weight",
    "fit_binary_target_direction_affine_calibration",
    "fit_shared_readout_diagonal_affine_calibration",
    "fit_batched_shared_readout_diagonal_affine_calibration",
    "fit_shared_readout_low_rank_plus_diagonal_affine_calibration",
    "evaluate_binary_target_matrix_losses",
    "evaluate_classwise_binary_losses",
    "evaluate_selected_binary_losses",
    "fit_binary_target_matrix_affine_calibration",
    "fit_single_binarized_softmax_affine_calibration",
    "fit_classwise_binary_affine_calibration",
    "fit_selected_binary_affine_calibration",
    "fit_single_binary_affine_calibration",
    "functional_selectivity_contrast_from_damages",
    "functional_selectivity_contrast_from_losses",
    "normalize_classwise_damage",
]


def classwise_triton_available() -> bool:
    return bool(_TRITON_AVAILABLE and TRITON_AVAILABLE)


def shared_readout_triton_available() -> bool:
    return bool(_TRITON_AVAILABLE and TRITON_AVAILABLE)


def _should_use_triton_classwise_backend(
    *,
    use_triton_grad_hess: bool,
    is_cuda: bool,
    num_slots: int,
    min_triton_slots: int,
) -> bool:
    return bool(
        use_triton_grad_hess
        and classwise_triton_available()
        and is_cuda
        and int(num_slots) >= int(min_triton_slots)
    )


def _safe_float_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _inverse_softplus_positive(value: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    value = value.clamp_min(float(eps))
    return value + torch.log(-torch.expm1(-value))


def _joint_scale_identifiability_mask(
    xs_class_major: torch.Tensor,
    *,
    eps_abs: float = 1e-8,
    eps_rel: float = 1e-6,
    diagnostic_context: Optional[str] = None,
) -> torch.Tensor:
    if xs_class_major.ndim != 2:
        raise ValueError("xs_class_major must have shape [C, N].")
    compute_dtype = _safe_float_dtype(xs_class_major.dtype)
    xs_f = xs_class_major.to(dtype=compute_dtype)
    slot_min = xs_f.amin(dim=1)
    slot_max = xs_f.amax(dim=1)
    finite_mask = torch.isfinite(slot_min) & torch.isfinite(slot_max)
    if not bool(finite_mask.all().item()):
        bad_slot_tensor = torch.nonzero(~finite_mask, as_tuple=False).view(-1)
        bad_slots = bad_slot_tensor[:8].tolist()
        sample_summaries = []
        nonfinite_samples = ~torch.isfinite(xs_f)
        for slot_idx in bad_slots[:4]:
            bad_positions = (
                torch.nonzero(nonfinite_samples[int(slot_idx)], as_tuple=False)
                .view(-1)[:8]
                .tolist()
            )
            finite_vals = xs_f[int(slot_idx)][torch.isfinite(xs_f[int(slot_idx)])]
            if int(finite_vals.numel()) > 0:
                finite_min = float(finite_vals.min().item())
                finite_max = float(finite_vals.max().item())
            else:
                finite_min = float("nan")
                finite_max = float("nan")
            sample_summaries.append(
                f"{int(slot_idx)}:bad_positions={bad_positions},"
                f"finite_min={finite_min:.6g},finite_max={finite_max:.6g}"
            )
        raise RuntimeError(
            "Non-finite classwise affine fit logits detected while checking scale identifiability; "
            f"context={diagnostic_context!r}, "
            f"bad_slots={bad_slots}, bad_slot_count={int((~finite_mask).sum().item())}, "
            f"sample_details=[{'; '.join(sample_summaries)}]."
        )
    slot_span = (slot_max - slot_min).abs()
    scale_ref = torch.maximum(slot_max.abs(), slot_min.abs()).clamp_min(1.0)
    threshold = torch.as_tensor(float(eps_abs), device=slot_span.device, dtype=slot_span.dtype)
    threshold = threshold + (
        torch.as_tensor(float(eps_rel), device=slot_span.device, dtype=slot_span.dtype)
        * scale_ref
    )
    return slot_span > threshold


def _raise_on_nonfinite_fit_result(
    *,
    classifier_scale: torch.Tensor,
    classifier_offset: torch.Tensor,
    classifier_scale_optimized: torch.Tensor,
    classifier_offset_optimized: torch.Tensor,
    bias_only_loss: torch.Tensor,
    optimized_loss: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    def _bad_indices(tensor: torch.Tensor) -> list[int]:
        mask = valid_mask & (~torch.isfinite(tensor))
        return torch.nonzero(mask, as_tuple=False).view(-1)[:8].tolist()

    bad_fields = []
    for field_name, tensor in (
        ("classifier_scale", classifier_scale),
        ("classifier_offset", classifier_offset),
        ("classifier_scale_optimized", classifier_scale_optimized),
        ("classifier_offset_optimized", classifier_offset_optimized),
        ("bias_only_loss", bias_only_loss),
        ("optimized_loss", optimized_loss),
    ):
        if bool((valid_mask & (~torch.isfinite(tensor))).any().item()):
            bad_fields.append(f"{field_name}={_bad_indices(tensor)}")
    if bad_fields:
        raise RuntimeError(
            "Non-finite classwise affine fit result detected after optimization; "
            + ", ".join(bad_fields)
        )


def compute_classwise_pos_weight(
    labels: torch.Tensor,
    *,
    num_classes: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    labels_long = labels.to(device=device, dtype=torch.long)
    counts = torch.bincount(labels_long, minlength=int(num_classes)).to(device=device, dtype=torch.float32)
    total = torch.tensor(float(labels_long.numel()), device=device, dtype=torch.float32)
    pos = counts.clamp_min(1.0)
    neg = (total - counts).clamp_min(1.0)
    pos_weight = neg / pos
    return pos_weight.to(dtype=dtype)


def compute_selected_pos_weight(
    labels: torch.Tensor,
    *,
    selected_class_ids: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    labels_long = labels.to(device=device, dtype=torch.long).view(-1)
    class_ids_long = selected_class_ids.to(device=device, dtype=torch.long).view(-1)
    hits = labels_long.unsqueeze(0).eq(class_ids_long.unsqueeze(1))
    counts = hits.sum(dim=1).to(dtype=torch.float32)
    total = torch.tensor(float(labels_long.numel()), device=device, dtype=torch.float32)
    pos = counts.clamp_min(1.0)
    neg = (total - counts).clamp_min(1.0)
    return (neg / pos).to(dtype=dtype)


def compute_binary_target_matrix_pos_weight(
    binary_targets: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if binary_targets.ndim != 2:
        raise ValueError("binary_targets must have shape [N, K].")
    targets = binary_targets.to(device=device, dtype=torch.float32)
    counts = targets.sum(dim=0)
    total = torch.tensor(float(targets.shape[0]), device=device, dtype=torch.float32)
    pos = counts.clamp_min(1.0)
    neg = (total - counts).clamp_min(1.0)
    return (neg / pos).to(dtype=dtype)


if _TRITON_AVAILABLE:

    @triton.jit
    def _classwise_grad_hess_kernel(
        xs_ptr,
        labels_ptr,
        class_ids_ptr,
        sample_mask_ptr,
        ws_ptr,
        bs_ptr,
        pos_w_ptr,
        loss_ptr,
        g_w_ptr,
        g_b_ptr,
        H_ww_ptr,
        H_wb_ptr,
        H_bb_ptr,
        count_ptr,
        C: tl.constexpr,
        N: tl.constexpr,
        HAS_SAMPLE_MASK: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        slot_id = tl.program_id(0)
        if slot_id >= C:
            return

        target_class = tl.load(class_ids_ptr + slot_id).to(tl.int32)
        w = tl.load(ws_ptr + slot_id)
        b = tl.load(bs_ptr + slot_id)
        pos_w = tl.load(pos_w_ptr + slot_id)

        loss_acc = 0.0
        g_w_acc = 0.0
        g_b_acc = 0.0
        H_ww_acc = 0.0
        H_wb_acc = 0.0
        H_bb_acc = 0.0
        count_acc = 0.0

        for start in range(0, N, BLOCK_N):
            offsets = start + tl.arange(0, BLOCK_N)
            sample_valid = offsets < N

            x = tl.load(xs_ptr + slot_id * N + offsets, mask=sample_valid, other=0.0).to(tl.float32)
            labels = tl.load(labels_ptr + offsets, mask=sample_valid, other=0).to(tl.int32)
            y = labels == target_class
            y_f = y.to(tl.float32)

            if HAS_SAMPLE_MASK:
                sample_mask = tl.load(sample_mask_ptr + offsets, mask=sample_valid, other=0.0).to(tl.float32)
            else:
                sample_mask = tl.where(sample_valid, 1.0, 0.0)

            logits = x * w + b
            p = tl.sigmoid(logits)
            s = p * (1.0 - p)

            alpha = tl.where(y, pos_w, 1.0) * sample_mask

            # softplus(-z) for positives, softplus(z) for negatives
            loss_pos = tl.where(
                logits >= 0,
                tl.log(1.0 + tl.exp(-logits)),
                -logits + tl.log(1.0 + tl.exp(logits)),
            )
            loss_neg = tl.where(
                logits >= 0,
                logits + tl.log(1.0 + tl.exp(-logits)),
                tl.log(1.0 + tl.exp(logits)),
            )
            loss_elem = tl.where(y, pos_w * loss_pos, loss_neg) * sample_mask

            delta = (p - y_f) * alpha
            s_weight = s * alpha

            loss_acc += tl.sum(loss_elem, axis=0)
            g_w_acc += tl.sum(delta * x, axis=0)
            g_b_acc += tl.sum(delta, axis=0)
            H_ww_acc += tl.sum(s_weight * x * x, axis=0)
            H_wb_acc += tl.sum(s_weight * x, axis=0)
            H_bb_acc += tl.sum(s_weight, axis=0)
            count_acc += tl.sum(sample_mask, axis=0)

        tl.store(loss_ptr + slot_id, loss_acc)
        tl.store(g_w_ptr + slot_id, g_w_acc)
        tl.store(g_b_ptr + slot_id, g_b_acc)
        tl.store(H_ww_ptr + slot_id, H_ww_acc)
        tl.store(H_wb_ptr + slot_id, H_wb_acc)
        tl.store(H_bb_ptr + slot_id, H_bb_acc)
        tl.store(count_ptr + slot_id, count_acc)


def _classwise_grad_hess_triton_prepacked(
    xs_class_major: torch.Tensor,
    labels_c: torch.Tensor,
    class_ids_c: torch.Tensor,
    ws: torch.Tensor,
    bs: torch.Tensor,
    pos_w_c: torch.Tensor,
    *,
    sample_mask_c: torch.Tensor,
    has_sample_mask: bool,
    out_buffers: Optional[Dict[str, torch.Tensor]] = None,
):
    if not classwise_triton_available():
        raise RuntimeError("Classwise Triton backend is unavailable.")
    if not xs_class_major.is_cuda:
        raise RuntimeError("Classwise Triton backend requires CUDA tensors.")
    if xs_class_major.ndim != 2:
        raise ValueError("xs_class_major must have shape [C, N].")

    C, N = xs_class_major.shape
    device = xs_class_major.device
    out_dtype = torch.float32

    def _buffer(name: str) -> torch.Tensor:
        if out_buffers is None:
            return torch.empty((C,), device=device, dtype=out_dtype)
        existing = out_buffers.get(name)
        if (
            existing is None
            or existing.shape != (C,)
            or existing.device != device
            or existing.dtype != out_dtype
        ):
            existing = torch.empty((C,), device=device, dtype=out_dtype)
            out_buffers[name] = existing
        return existing

    xs_c = xs_class_major.contiguous()
    ws_c = ws.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
    bs_c = bs.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()

    loss = _buffer("loss")
    g_w = _buffer("g_w")
    g_b = _buffer("g_b")
    H_ww = _buffer("H_ww")
    H_wb = _buffer("H_wb")
    H_bb = _buffer("H_bb")
    count = _buffer("count")

    grid = (int(C),)
    _classwise_grad_hess_kernel[grid](
        xs_c,
        labels_c,
        class_ids_c,
        sample_mask_c,
        ws_c,
        bs_c,
        pos_w_c,
        loss,
        g_w,
        g_b,
        H_ww,
        H_wb,
        H_bb,
        count,
        C=int(C),
        N=int(N),
        HAS_SAMPLE_MASK=has_sample_mask,
        BLOCK_N=256,
        num_warps=4,
        num_stages=2,
    )

    denom = count.clamp_min(1e-12)
    return (
        loss / denom,
        g_w / denom,
        g_b / denom,
        H_ww / denom,
        H_wb / denom,
        H_bb / denom,
        count,
    )


def _classwise_grad_hess_triton(
    xs_class_major: torch.Tensor,
    labels: torch.Tensor,
    class_ids: torch.Tensor,
    ws: torch.Tensor,
    bs: torch.Tensor,
    pos_w: torch.Tensor,
    *,
    sample_mask: Optional[torch.Tensor] = None,
    out_buffers: Optional[Dict[str, torch.Tensor]] = None,
):
    device = xs_class_major.device
    labels_c = labels.to(device=device, dtype=torch.int32, non_blocking=True).contiguous()
    class_ids_c = class_ids.to(device=device, dtype=torch.int32, non_blocking=True).contiguous()
    pos_w_c = pos_w.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
    if sample_mask is not None:
        sample_mask_c = sample_mask.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
    else:
        sample_mask_c = torch.empty((0,), device=device, dtype=torch.float32)
    return _classwise_grad_hess_triton_prepacked(
        xs_class_major,
        labels_c,
        class_ids_c,
        ws,
        bs,
        pos_w_c,
        sample_mask_c=sample_mask_c,
        has_sample_mask=sample_mask is not None,
        out_buffers=out_buffers,
    )


def _classwise_grad_hess_torch(
    xs_class_major: torch.Tensor,
    labels: torch.Tensor,
    class_ids: torch.Tensor,
    ws: torch.Tensor,
    bs: torch.Tensor,
    pos_w: torch.Tensor,
    *,
    sample_mask: Optional[torch.Tensor] = None,
    class_chunk_size: int = 128,
):
    if xs_class_major.ndim != 2:
        raise ValueError("xs_class_major must have shape [C, N].")

    device = xs_class_major.device
    C, N = xs_class_major.shape
    compute_dtype = _safe_float_dtype(xs_class_major.dtype)
    labels_long = labels.to(device=device, dtype=torch.long).view(-1)
    sample_mask_f = (
        sample_mask.to(device=device, dtype=compute_dtype).view(-1)
        if sample_mask is not None
        else None
    )
    denom = (
        sample_mask_f.sum().clamp_min(1.0)
        if sample_mask_f is not None
        else torch.tensor(float(N), device=device, dtype=compute_dtype)
    )

    losses = torch.empty((C,), device=device, dtype=compute_dtype)
    g_w = torch.empty((C,), device=device, dtype=compute_dtype)
    g_b = torch.empty((C,), device=device, dtype=compute_dtype)
    H_ww = torch.empty((C,), device=device, dtype=compute_dtype)
    H_wb = torch.empty((C,), device=device, dtype=compute_dtype)
    H_bb = torch.empty((C,), device=device, dtype=compute_dtype)
    count = torch.full((C,), float(denom.item()), device=device, dtype=compute_dtype)

    class_ids_long = class_ids.to(device=device, dtype=torch.long).view(-1, 1)
    labels_expanded = labels_long.unsqueeze(0)
    for start in range(0, int(C), max(1, int(class_chunk_size))):
        end = min(start + int(class_chunk_size), int(C))
        class_ids_chunk = class_ids_long[start:end]
        x_chunk = xs_class_major[start:end].to(dtype=compute_dtype)
        w_chunk = ws[start:end].to(device=device, dtype=compute_dtype).unsqueeze(1)
        b_chunk = bs[start:end].to(device=device, dtype=compute_dtype).unsqueeze(1)
        pos_chunk = pos_w[start:end].to(device=device, dtype=compute_dtype).unsqueeze(1)
        y_chunk = labels_expanded.eq(class_ids_chunk)
        y_f = y_chunk.to(dtype=compute_dtype)
        logits = x_chunk * w_chunk + b_chunk
        p = torch.sigmoid(logits)
        s = p * (1.0 - p)
        alpha = torch.where(y_chunk, pos_chunk, torch.ones_like(pos_chunk))
        if sample_mask_f is not None:
            alpha = alpha * sample_mask_f.unsqueeze(0)
        losses[start:end] = torch.where(
            y_chunk,
            pos_chunk * F.softplus(-logits),
            F.softplus(logits),
        ).mul(sample_mask_f.unsqueeze(0) if sample_mask_f is not None else 1.0).sum(dim=1) / denom
        delta = (p - y_f) * alpha
        g_w[start:end] = delta.mul(x_chunk).sum(dim=1) / denom
        g_b[start:end] = delta.sum(dim=1) / denom
        s_weight = s * alpha
        H_ww[start:end] = s_weight.mul(x_chunk.square()).sum(dim=1) / denom
        H_wb[start:end] = s_weight.mul(x_chunk).sum(dim=1) / denom
        H_bb[start:end] = s_weight.sum(dim=1) / denom

    return losses, g_w, g_b, H_ww, H_wb, H_bb, count


def _optimize_wb_logistic_classwise(
    xs_class_major: torch.Tensor,
    labels: torch.Tensor,
    class_ids: torch.Tensor,
    ws0: torch.Tensor,
    bs0: torch.Tensor,
    pos_w: torch.Tensor,
    *,
    optimize_scale: bool,
    iters: int = 50,
    damping_init: float = 1e-4,
    step_tol: float = 1e-9,
    patience: int = 10,
    conv_tol: float = 1e-8,
    min_iters: int = 3,
    early_stop_grad_tol: float = 1e-7,
    class_chunk_size: int = 128,
    use_triton_grad_hess: bool = True,
    min_triton_slots: int = 8,
    compact_active_slots: bool = True,
    prior_lambda_w: Optional[torch.Tensor] = None,
    prior_lambda_b: Optional[torch.Tensor] = None,
    prior_center_w: float = 1.0,
    prior_center_b: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = xs_class_major.device
    dtype = _safe_float_dtype(xs_class_major.dtype)
    C = int(xs_class_major.shape[0])

    ws = ws0.to(device=device, dtype=dtype).clone()
    bs = bs0.to(device=device, dtype=dtype).clone()
    if prior_lambda_w is None:
        prior_lambda_w = torch.zeros((C,), device=device, dtype=dtype)
    else:
        prior_lambda_w = prior_lambda_w.to(device=device, dtype=dtype).view(-1)
    if prior_lambda_b is None:
        prior_lambda_b = torch.zeros((C,), device=device, dtype=dtype)
    else:
        prior_lambda_b = prior_lambda_b.to(device=device, dtype=dtype).view(-1)
    prior_center_w_t = torch.as_tensor(float(prior_center_w), device=device, dtype=dtype)
    prior_center_b_t = torch.as_tensor(float(prior_center_b), device=device, dtype=dtype)
    lam = torch.full((C,), float(damping_init), device=device, dtype=dtype)
    step_history = torch.full((C,), float("inf"), device=device, dtype=dtype)
    patience_ctr = torch.zeros((C,), device=device, dtype=torch.long)
    best_loss = torch.full((C,), float("inf"), device=device, dtype=dtype)
    best_ws = ws.clone()
    best_bs = bs.clone()
    active_all = torch.ones((C,), device=device, dtype=torch.bool)

    use_triton = _should_use_triton_classwise_backend(
        use_triton_grad_hess=use_triton_grad_hess,
        is_cuda=bool(xs_class_major.is_cuda),
        num_slots=C,
        min_triton_slots=min_triton_slots,
    )
    triton_buffers: Optional[Dict[str, torch.Tensor]] = {} if use_triton else None
    triton_step_buffers: Optional[Dict[str, torch.Tensor]] = {} if (use_triton and optimize_scale) else None
    triton_subset_buffers: Optional[Dict[str, torch.Tensor]] = {} if use_triton else None
    ws_min_tensor = torch.full((C,), -float("inf"), device=device, dtype=dtype) if (use_triton and optimize_scale) else None
    if use_triton:
        labels_triton = labels.to(device=device, dtype=torch.int32, non_blocking=True).contiguous()
        class_ids_triton = class_ids.to(device=device, dtype=torch.int32, non_blocking=True).contiguous()
        pos_w_triton = pos_w.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
        sample_mask_triton = torch.empty((0,), device=device, dtype=torch.float32)
        has_sample_mask = False
    else:
        labels_triton = None
        class_ids_triton = None
        pos_w_triton = None
        sample_mask_triton = None
        has_sample_mask = False

    def _apply_prior_terms(
        loss_curr: torch.Tensor,
        g_w_curr: torch.Tensor,
        g_b_curr: torch.Tensor,
        H_ww_curr: torch.Tensor,
        H_wb_curr: torch.Tensor,
        H_bb_curr: torch.Tensor,
        w_curr: torch.Tensor,
        b_curr: torch.Tensor,
        lambda_w_curr: torch.Tensor,
        lambda_b_curr: torch.Tensor,
    ):
        loss_with_prior = loss_curr.clone()
        if optimize_scale:
            loss_with_prior = loss_with_prior + 0.5 * lambda_w_curr * (w_curr - prior_center_w_t).square()
            g_w_curr = g_w_curr + lambda_w_curr * (w_curr - prior_center_w_t)
            H_ww_curr = H_ww_curr + lambda_w_curr
        loss_with_prior = loss_with_prior + 0.5 * lambda_b_curr * (b_curr - prior_center_b_t).square()
        g_b_curr = g_b_curr + lambda_b_curr * (b_curr - prior_center_b_t)
        H_bb_curr = H_bb_curr + lambda_b_curr
        return loss_with_prior, g_w_curr, g_b_curr, H_ww_curr, H_wb_curr, H_bb_curr

    def _objective(w_curr: torch.Tensor, b_curr: torch.Tensor):
        if use_triton:
            loss_curr, g_w_curr, g_b_curr, H_ww_curr, H_wb_curr, H_bb_curr = _classwise_grad_hess_triton_prepacked(
                xs_class_major,
                labels_triton,
                class_ids_triton,
                w_curr,
                b_curr,
                pos_w_triton,
                sample_mask_c=sample_mask_triton,
                has_sample_mask=has_sample_mask,
                out_buffers=triton_buffers,
            )[:6]
        else:
            loss_curr, g_w_curr, g_b_curr, H_ww_curr, H_wb_curr, H_bb_curr = _classwise_grad_hess_torch(
            xs_class_major,
            labels,
            class_ids,
            w_curr,
            b_curr,
            pos_w,
            class_chunk_size=class_chunk_size,
            )[:6]
        return _apply_prior_terms(
            loss_curr,
            g_w_curr,
            g_b_curr,
            H_ww_curr,
            H_wb_curr,
            H_bb_curr,
            w_curr,
            b_curr,
            prior_lambda_w,
            prior_lambda_b,
        )

    def _objective_subset(active_idx: torch.Tensor, w_curr: torch.Tensor, b_curr: torch.Tensor):
        if int(active_idx.numel()) == C:
            return _objective(w_curr, b_curr)
        xs_subset = xs_class_major.index_select(0, active_idx)
        labels_subset = labels_triton if use_triton else labels
        class_ids_subset = (
            class_ids_triton.index_select(0, active_idx)
            if use_triton
            else class_ids.index_select(0, active_idx)
        )
        pos_w_subset = (
            pos_w_triton.index_select(0, active_idx)
            if use_triton
            else pos_w.index_select(0, active_idx)
        )
        lambda_w_subset = prior_lambda_w.index_select(0, active_idx)
        lambda_b_subset = prior_lambda_b.index_select(0, active_idx)
        if use_triton:
            loss_curr, g_w_curr, g_b_curr, H_ww_curr, H_wb_curr, H_bb_curr = _classwise_grad_hess_triton_prepacked(
                xs_subset,
                labels_subset,
                class_ids_subset,
                w_curr,
                b_curr,
                pos_w_subset,
                sample_mask_c=sample_mask_triton,
                has_sample_mask=has_sample_mask,
                out_buffers=triton_subset_buffers,
            )[:6]
        else:
            loss_curr, g_w_curr, g_b_curr, H_ww_curr, H_wb_curr, H_bb_curr = _classwise_grad_hess_torch(
            xs_subset,
            labels_subset,
            class_ids_subset,
            w_curr,
            b_curr,
            pos_w_subset,
            class_chunk_size=class_chunk_size,
            )[:6]
        return _apply_prior_terms(
            loss_curr,
            g_w_curr,
            g_b_curr,
            H_ww_curr,
            H_wb_curr,
            H_bb_curr,
            w_curr,
            b_curr,
            lambda_w_subset,
            lambda_b_subset,
        )

    loss, g_w, g_b, H_ww, H_wb, H_bb = _objective(ws, bs)
    best_loss = loss.clone()

    for iteration in range(max(1, int(iters))):
        grad_norm = torch.sqrt((g_w if optimize_scale else 0.0) ** 2 + g_b ** 2)
        converged = (grad_norm <= conv_tol) & (step_history <= step_tol)
        converged = converged | (patience_ctr >= int(patience))
        done = converged
        if iteration >= int(min_iters) and done.all():
            break
        if iteration >= int(min_iters) and grad_norm.max() < float(early_stop_grad_tol):
            break

        active = active_all & (~done)
        if not active.any():
            break
        active_idx = torch.nonzero(active, as_tuple=False).view(-1)
        use_active_subset = bool(
            compact_active_slots
            and int(active_idx.numel()) < C
        )

        if optimize_scale and use_triton:
            lam_avg = lam
            delta_w, delta_b, cand_ws, cand_bs, pred_red = trust_region_step_fused(
                g_w,
                g_b,
                H_ww,
                H_wb,
                H_bb,
                lam_avg,
                ws,
                bs,
                ws_min_tensor,
                active,
                False,
                out_buffers=triton_step_buffers,
            )
        elif optimize_scale:
            a = H_ww + lam
            d = H_bb + lam
            b = H_wb
            det = (a * d - b * b).clamp_min(1e-12)
            inv_det = 1.0 / det
            inv_00 = d * inv_det
            inv_01 = -b * inv_det
            inv_11 = a * inv_det
            delta_w = -(inv_00 * g_w + inv_01 * g_b) * active
            delta_b = -(inv_01 * g_w + inv_11 * g_b) * active
            cand_ws = ws + delta_w
            cand_bs = bs + delta_b
            quad = (
                delta_w * (H_ww * delta_w + H_wb * delta_b)
                + delta_b * (H_wb * delta_w + H_bb * delta_b)
            )
            pred_red = -(g_w * delta_w + g_b * delta_b) - 0.5 * quad
        else:
            delta_w = torch.zeros_like(ws)
            delta_b = -(g_b / (H_bb + lam).clamp_min(1e-12)) * active
            cand_ws = ws
            cand_bs = bs + delta_b
            pred_red = -(g_b * delta_b) - 0.5 * (H_bb * delta_b * delta_b)

        if use_active_subset:
            cand_loss = loss.clone()
            cand_loss_subset, _, _, _, _, _ = _objective_subset(
                active_idx,
                cand_ws.index_select(0, active_idx),
                cand_bs.index_select(0, active_idx),
            )
            cand_loss[active_idx] = cand_loss_subset
        else:
            cand_loss, _, _, _, _, _ = _objective(cand_ws, cand_bs)
        act_red = loss - cand_loss
        rho = torch.where(pred_red.abs() > 1e-12, act_red / pred_red, torch.zeros_like(act_red))
        accept = active & (rho > 1e-3) & torch.isfinite(cand_loss)

        ws = torch.where(accept, cand_ws, ws)
        bs = torch.where(accept, cand_bs, bs)
        loss = torch.where(accept, cand_loss, loss)

        decrease_factor = (1.0 - (2.0 * rho - 1.0).clamp(-1.0, 1.0) ** 3).clamp_min(1.0 / 3.0)
        increase_factor = 2.0 + (0.25 - rho.clamp(0.0, 0.25)) * 4.0
        lam = torch.where(accept & (rho > 0.25), lam * decrease_factor, lam)
        lam = torch.where(accept & (rho <= 0.25), lam * increase_factor, lam)
        lam = torch.where(active & (~accept), lam * 2.0, lam)
        lam = lam.clamp(1e-12, 1e8)

        step_sq = delta_w.square() + delta_b.square()
        step_history = torch.where(accept, torch.sqrt(step_sq), step_history)

        improved = loss < (best_loss - conv_tol)
        best_loss = torch.where(improved, loss, best_loss)
        best_ws = torch.where(improved, ws, best_ws)
        best_bs = torch.where(improved, bs, best_bs)
        patience_ctr = torch.where(improved, torch.zeros_like(patience_ctr), patience_ctr + 1)
        if (patience_ctr >= int(patience)).all():
            break

        if use_active_subset:
            loss_subset, g_w_subset, g_b_subset, H_ww_subset, H_wb_subset, H_bb_subset = _objective_subset(
                active_idx,
                ws.index_select(0, active_idx),
                bs.index_select(0, active_idx),
            )
            loss[active_idx] = loss_subset
            g_w[active_idx] = g_w_subset
            g_b[active_idx] = g_b_subset
            H_ww[active_idx] = H_ww_subset
            H_wb[active_idx] = H_wb_subset
            H_bb[active_idx] = H_bb_subset
        else:
            loss, g_w, g_b, H_ww, H_wb, H_bb = _objective(ws, bs)

    return best_ws, best_bs, best_loss


@dataclass
class ClasswiseAffineFitResult:
    classifier_offset: torch.Tensor
    classifier_scale: torch.Tensor
    classifier_offset_optimized: torch.Tensor
    classifier_scale_optimized: torch.Tensor
    pos_weight: torch.Tensor
    bias_only_loss: torch.Tensor
    optimized_loss: torch.Tensor


@dataclass(frozen=True)
class ReadoutSpaceAffineFitResult:
    """Shared readout-space diagonal affine calibration.

    ``readout_scale`` and ``readout_offset`` live in the shared readout feature
    coordinates.  Target logits are obtained as
    ``(readout * readout_scale) @ target_vectors.T + target_vectors @ readout_offset``.
    This is the coherent SSL/prototype/adaptive generalization of old-sparse
    per-unit affine calibration: for one-hot target vectors it reduces to
    independent affine calibration of the corresponding output coordinates.
    """

    readout_scale: torch.Tensor
    readout_offset: torch.Tensor
    target_vectors: torch.Tensor
    pos_weight: torch.Tensor
    bias_only_loss: torch.Tensor
    optimized_loss: torch.Tensor
    target_intercepts: Optional[torch.Tensor] = None
    calibration_family: str = "c2_shared_diagonal_affine"
    calibration_backend: str = "shared_readout_lbfgs_reference"
    low_rank_left: Optional[torch.Tensor] = None
    low_rank_right: Optional[torch.Tensor] = None
    low_rank_rank: int = 0


@dataclass(frozen=True)
class BatchedReadoutSpaceAffineFitResult:
    """Independent shared-readout C2 fits for a candidate-mask batch.

    The leading dimension indexes masks, usually committed plus candidate masks
    evaluated on the same statistical panel.  Each slice is the exact same C2
    fit that ``fit_shared_readout_diagonal_affine_calibration`` would produce
    for that mask/readout panel.
    """

    readout_scale: torch.Tensor
    readout_offset: torch.Tensor
    target_vectors: torch.Tensor
    pos_weight: torch.Tensor
    bias_only_loss: torch.Tensor
    optimized_loss: torch.Tensor
    target_intercepts: Optional[torch.Tensor] = None
    calibration_family: str = "c2_shared_diagonal_affine"
    calibration_backend: str = "batched_shared_readout_torch_analytic"


def _one_hot_target_vector_columns(target_vectors: torch.Tensor) -> torch.Tensor | None:
    if target_vectors.ndim != 2 or int(target_vectors.shape[1]) == 0:
        return None
    u = target_vectors.detach()
    abs_u = u.abs()
    max_values, columns = abs_u.max(dim=1)
    nonzero_counts = (abs_u > 1e-7).sum(dim=1)
    if not bool((nonzero_counts == 1).all().item()):
        return None
    if not bool(torch.allclose(max_values, torch.ones_like(max_values), atol=1e-6, rtol=1e-6)):
        return None
    selected_values = u.gather(1, columns.view(-1, 1)).view(-1)
    if not bool(torch.allclose(selected_values, torch.ones_like(selected_values), atol=1e-6, rtol=1e-6)):
        return None
    return columns.to(device=target_vectors.device, dtype=torch.long)


if _TRITON_AVAILABLE:

    @triton.jit
    def _shared_readout_residual_kernel(
        readout_ptr,
        target_ptr,
        labels_ptr,
        scale_ptr,
        offset_ptr,
        pos_weight_ptr,
        residual_ptr,
        loss_blocks_ptr,
        N: tl.constexpr,
        D: tl.constexpr,
        K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for d_start in range(0, D, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            r = tl.load(
                readout_ptr + offs_n[:, None] * D + offs_d[None, :],
                mask=(offs_n[:, None] < N) & (offs_d[None, :] < D),
                other=0.0,
            ).to(tl.float32)
            s = tl.load(scale_ptr + offs_d, mask=offs_d < D, other=1.0).to(tl.float32)
            b = tl.load(offset_ptr + offs_d, mask=offs_d < D, other=0.0).to(tl.float32)
            u = tl.load(
                target_ptr + offs_k[:, None] * D + offs_d[None, :],
                mask=(offs_k[:, None] < K) & (offs_d[None, :] < D),
                other=0.0,
            ).to(tl.float32)
            acc += tl.dot(r * s[None, :] + b[None, :], tl.trans(u))

        valid = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        y = tl.load(
            labels_ptr + offs_n[:, None] * K + offs_k[None, :],
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        pos_w = tl.load(pos_weight_ptr + offs_k, mask=offs_k < K, other=1.0).to(tl.float32)
        y_pos = y > 0.5
        p = tl.sigmoid(acc)
        alpha = tl.where(y_pos, pos_w[None, :], 1.0)
        residual = (p - y) * alpha
        loss_pos = tl.where(
            acc >= 0.0,
            tl.log(1.0 + tl.exp(-acc)),
            -acc + tl.log(1.0 + tl.exp(acc)),
        )
        loss_neg = tl.where(
            acc >= 0.0,
            acc + tl.log(1.0 + tl.exp(-acc)),
            tl.log(1.0 + tl.exp(acc)),
        )
        loss_elem = tl.where(y_pos, pos_w[None, :] * loss_pos, loss_neg)
        residual = tl.where(valid, residual, 0.0)
        loss_elem = tl.where(valid, loss_elem, 0.0)
        tl.store(
            residual_ptr + offs_n[:, None] * K + offs_k[None, :],
            residual,
            mask=valid,
        )
        loss_sum = tl.sum(loss_elem, axis=0)
        tl.store(loss_blocks_ptr + pid_n * K + offs_k, loss_sum, mask=offs_k < K)


def _resolve_shared_readout_backend(backend: object, *, readout: torch.Tensor) -> str:
    token = str(backend or "auto").strip().lower().replace("-", "_")
    aliases = {
        "": "auto",
        "fast": "auto",
        "fused": "auto",
        "cuda": "triton",
        "lbfgs": "reference",
        "legacy": "reference",
        "pytorch_autograd": "reference",
        "shared_readout_lbfgs_reference": "reference",
        "torch": "torch_analytic",
        "torch_fused": "torch_analytic",
        "analytic": "torch_analytic",
        "shared_readout_torch_analytic": "torch_analytic",
        "shared_readout_triton": "triton",
    }
    token = aliases.get(token, token)
    if token == "auto":
        if readout.is_cuda and shared_readout_triton_available():
            return "triton"
        return "reference"
    if token == "triton":
        if not readout.is_cuda or not shared_readout_triton_available():
            raise RuntimeError("shared_readout_triton backend requires CUDA tensors and Triton.")
        return "triton"
    if token in {"torch_analytic", "reference"}:
        return token
    raise ValueError(f"Unsupported shared readout C2 backend {backend!r}.")


def _shared_readout_loss_grad_torch(
    r: torch.Tensor,
    u: torch.Tensor,
    y: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    pos_weight: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    denom = torch.tensor(float(r.shape[0]), device=r.device, dtype=r.dtype).clamp_min(1.0)
    eta = (r * scale.unsqueeze(0)).matmul(u.t()) + u.matmul(offset).unsqueeze(0)
    weight = torch.where(y > 0.5, pos_weight.unsqueeze(0), torch.ones_like(y))
    loss = torch.where(y > 0.5, F.softplus(-eta), F.softplus(eta)) * weight
    losses = loss.sum(dim=0) / denom
    residual = (torch.sigmoid(eta) - y) * weight / denom
    valid_count = valid_mask.to(dtype=r.dtype).sum().clamp_min(1.0)
    residual = residual * valid_mask.to(dtype=r.dtype).unsqueeze(0) / valid_count
    readout_grad = residual.matmul(u)
    grad_scale = (readout_grad * r).sum(dim=0)
    grad_offset = readout_grad.sum(dim=0)
    return losses, grad_scale, grad_offset


def _shared_readout_loss_grad_triton(
    r: torch.Tensor,
    u: torch.Tensor,
    y: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    pos_weight: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not shared_readout_triton_available() or not r.is_cuda:
        raise RuntimeError("Shared readout Triton backend is unavailable.")
    r_c = r.contiguous()
    u_c = u.contiguous()
    y_c = y.contiguous()
    scale_c = scale.to(device=r.device, dtype=torch.float32).contiguous()
    offset_c = offset.to(device=r.device, dtype=torch.float32).contiguous()
    pos_c = pos_weight.to(device=r.device, dtype=torch.float32).contiguous()
    n, d = int(r_c.shape[0]), int(r_c.shape[1])
    k = int(u_c.shape[0])
    block_n = 32
    block_k = 16
    block_d = 32
    grid_n = triton.cdiv(n, block_n)
    residual = torch.empty((n, k), device=r.device, dtype=torch.float32)
    loss_blocks = torch.empty((grid_n, k), device=r.device, dtype=torch.float32)
    _shared_readout_residual_kernel[(grid_n, triton.cdiv(k, block_k))](
        r_c,
        u_c,
        y_c,
        scale_c,
        offset_c,
        pos_c,
        residual,
        loss_blocks,
        N=n,
        D=d,
        K=k,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
    )
    denom = torch.tensor(float(n), device=r.device, dtype=torch.float32).clamp_min(1.0)
    losses = loss_blocks.sum(dim=0) / denom
    valid_count = valid_mask.to(device=r.device, dtype=torch.float32).sum().clamp_min(1.0)
    residual = residual / denom
    residual = residual * valid_mask.to(device=r.device, dtype=torch.float32).unsqueeze(0) / valid_count
    readout_grad = residual.matmul(u_c)
    grad_scale = (readout_grad * r_c).sum(dim=0)
    grad_offset = readout_grad.sum(dim=0)
    return losses.to(dtype=r.dtype), grad_scale.to(dtype=r.dtype), grad_offset.to(dtype=r.dtype)


def _shared_readout_loss_grad(
    r: torch.Tensor,
    u: torch.Tensor,
    y: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    pos_weight: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    backend: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if backend == "triton":
        return _shared_readout_loss_grad_triton(r, u, y, scale, offset, pos_weight, valid_mask)
    return _shared_readout_loss_grad_torch(r, u, y, scale, offset, pos_weight, valid_mask)


def _shared_readout_loss_grad_torch_batched(
    r: torch.Tensor,
    u: torch.Tensor,
    y: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    pos_weight: torch.Tensor,
    valid_mask: torch.Tensor,
    target_intercepts: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if r.ndim != 3:
        raise ValueError(f"batched readout must have shape [M, N, D], got {tuple(r.shape)}.")
    denom = torch.tensor(float(r.shape[1]), device=r.device, dtype=r.dtype).clamp_min(1.0)
    valid_count = valid_mask.to(device=r.device, dtype=r.dtype).sum().clamp_min(1.0)
    mask_count = torch.tensor(float(r.shape[0]), device=r.device, dtype=r.dtype).clamp_min(1.0)
    calibrated_readout = r * scale.unsqueeze(1) + offset.unsqueeze(1)
    eta = torch.einsum("mnd,kd->mnk", calibrated_readout, u)
    if target_intercepts is not None:
        eta = eta + target_intercepts.to(device=r.device, dtype=r.dtype).view(1, 1, -1)
    y_b = y.unsqueeze(0)
    weight = torch.where(y_b > 0.5, pos_weight.view(1, 1, -1), torch.ones_like(y_b))
    loss = torch.where(y_b > 0.5, F.softplus(-eta), F.softplus(eta)) * weight
    losses = loss.sum(dim=1) / denom
    residual = (torch.sigmoid(eta) - y_b) * weight / (denom * valid_count * mask_count)
    residual = residual * valid_mask.to(dtype=r.dtype).view(1, 1, -1)
    readout_grad = torch.einsum("mnk,kd->mnd", residual, u)
    grad_scale = (readout_grad * r).sum(dim=1)
    grad_offset = readout_grad.sum(dim=1)
    return losses, grad_scale, grad_offset


def _shared_readout_losses_torch_batched(
    r: torch.Tensor,
    u: torch.Tensor,
    y: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    pos_weight: torch.Tensor,
    target_intercepts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    denom = torch.tensor(float(r.shape[1]), device=r.device, dtype=r.dtype).clamp_min(1.0)
    calibrated_readout = r * scale.unsqueeze(1) + offset.unsqueeze(1)
    eta = torch.einsum("mnd,kd->mnk", calibrated_readout, u)
    if target_intercepts is not None:
        eta = eta + target_intercepts.to(device=r.device, dtype=r.dtype).view(1, 1, -1)
    y_b = y.unsqueeze(0)
    weight = torch.where(y_b > 0.5, pos_weight.view(1, 1, -1), torch.ones_like(y_b))
    loss = torch.where(y_b > 0.5, F.softplus(-eta), F.softplus(eta)) * weight
    return loss.sum(dim=1) / denom


def _compute_selected_effective_n(
    labels: torch.Tensor,
    class_ids: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    labels_long = labels.to(device=device, dtype=torch.long).view(-1)
    class_ids_long = class_ids.to(device=device, dtype=torch.long).view(-1, 1)
    total = torch.tensor(float(labels_long.numel()), device=device, dtype=dtype)
    num_pos = labels_long.unsqueeze(0).eq(class_ids_long).sum(dim=1).to(dtype=dtype)
    num_neg = total - num_pos
    pos_weight = torch.where(
        num_pos > 0,
        num_neg / num_pos.clamp_min(1.0),
        torch.ones_like(num_pos),
    )
    sample_weight_sum = num_neg + num_pos * pos_weight
    return torch.where(
        (num_pos > 0) & (num_neg > 0),
        sample_weight_sum.clamp_min(1.0),
        torch.ones_like(sample_weight_sum),
    )


def _compute_log_evidence_classwise(
    *,
    obj_hat: torch.Tensor,
    H_ww: torch.Tensor,
    H_wb: torch.Tensor,
    H_bb: torch.Tensor,
    lambda_w: torch.Tensor,
    lambda_b: torch.Tensor,
    n_eff: torch.Tensor,
    optimize_scale: bool,
    ridge: float,
) -> torch.Tensor:
    ridge_t = torch.as_tensor(float(ridge), device=obj_hat.device, dtype=obj_hat.dtype)
    if optimize_scale:
        det = (H_ww * H_bb - H_wb.square()).clamp_min(ridge_t)
        logdet = torch.log(det)
        log_lambda_term = 0.5 * (
            torch.log(lambda_w.clamp_min(ridge_t))
            + torch.log(lambda_b.clamp_min(ridge_t))
        )
    else:
        logdet = torch.log(H_bb.clamp_min(ridge_t))
        log_lambda_term = 0.5 * torch.log(lambda_b.clamp_min(ridge_t))
    return -n_eff * obj_hat - 0.5 * logdet + log_lambda_term


def fit_selected_binary_affine_calibration(
    logits_selected: torch.Tensor,
    labels: torch.Tensor,
    *,
    selected_class_ids: torch.Tensor,
    init_classifier_offset: Optional[torch.Tensor] = None,
    init_classifier_scale: Optional[torch.Tensor] = None,
    optimize_classifier_scale_joint: bool = True,
    num_iterations: int = 80,
    damping_init: float = 1e-4,
    class_chunk_size: int = 128,
    use_triton_grad_hess: bool = True,
    min_triton_slots: Optional[int] = None,
    compact_active_slots: bool = True,
    use_map_prior: bool = False,
    map_prior_c: float = 0.1,
    map_prior_eps: float = 1e-6,
    map_prior_mode: str = "off",
    map_prior_grid_size: int = 7,
    map_prior_grid_span: float = 2.0,
    map_prior_grid_max_iters: int = 200,
    map_prior_hessian_ridge: float = 1e-8,
    affine_calibration_mode: str = "hessian_minimization",
    diagnostic_context: Optional[str] = None,
) -> ClasswiseAffineFitResult:
    if logits_selected.ndim != 2:
        raise ValueError("logits_selected must have shape [N, K].")
    device = logits_selected.device
    compute_dtype = _safe_float_dtype(logits_selected.dtype)
    num_slots = int(logits_selected.shape[1])
    class_ids = selected_class_ids.to(device=device, dtype=torch.long).view(-1)
    if int(class_ids.numel()) != num_slots:
        raise ValueError("selected_class_ids must have shape [K].")
    xs_class_major = logits_selected.transpose(0, 1).contiguous()
    labels_long = labels.to(device=device, dtype=torch.long).view(-1)
    total = float(labels_long.numel())
    counts = labels_long.unsqueeze(0).eq(class_ids.unsqueeze(1)).sum(dim=1)
    valid_mask = (counts > 0) & (counts < int(total))
    pos_weight = compute_selected_pos_weight(
        labels,
        selected_class_ids=class_ids,
        device=device,
        dtype=compute_dtype,
    )
    if init_classifier_scale is None:
        init_classifier_scale = torch.ones((num_slots,), device=device, dtype=compute_dtype)
    else:
        init_classifier_scale = init_classifier_scale.to(device=device, dtype=compute_dtype).view(-1)
    if init_classifier_offset is None:
        init_classifier_offset = torch.zeros((num_slots,), device=device, dtype=compute_dtype)
    else:
        init_classifier_offset = init_classifier_offset.to(device=device, dtype=compute_dtype).view(-1)
    resolved_affine_mode = str(affine_calibration_mode).strip().lower()
    if resolved_affine_mode == "":
        resolved_affine_mode = "hessian_minimization"
    resolved_map_prior_mode = (
        str(map_prior_mode).strip().lower()
        if use_map_prior
        else "off"
    )
    if resolved_affine_mode == "lobo_hierarchical_empirical_bayes":
        raise NotImplementedError(
            "LOBO hierarchical empirical Bayes is not implemented in the optimized classwise backend. "
            "Caller should fall back to optimize_affine_transformation()."
        )

    scale_bias = init_classifier_scale.clone()
    offset_bias = init_classifier_offset.clone()
    scale_joint = init_classifier_scale.clone()
    offset_joint = init_classifier_offset.clone()
    loss_bias = torch.full((num_slots,), float("inf"), device=device, dtype=compute_dtype)
    loss_joint = torch.full((num_slots,), float("inf"), device=device, dtype=compute_dtype)

    if valid_mask.any():
        resolved_min_triton_slots = (
            int(os.getenv("CIRCUIT_AFFINE_TRITON_MIN_SLOTS", "8"))
            if min_triton_slots is None
            else int(min_triton_slots)
        )
        valid_xs = xs_class_major[valid_mask]
        valid_class_ids = class_ids[valid_mask]
        valid_pos_weight = pos_weight[valid_mask]
        valid_init_scale = init_classifier_scale[valid_mask]
        valid_init_offset = init_classifier_offset[valid_mask]
        valid_nonfinite_mask = ~torch.isfinite(valid_xs)
        if bool(valid_nonfinite_mask.any().item()):
            bad_local_slots = torch.nonzero(valid_nonfinite_mask.any(dim=1), as_tuple=False).view(-1)
            sample_summaries = []
            for local_slot_idx in bad_local_slots[:4]:
                local_slot = int(local_slot_idx.item())
                slot_values = valid_xs[local_slot]
                bad_positions = (
                    torch.nonzero(valid_nonfinite_mask[local_slot], as_tuple=False)
                    .view(-1)[:8]
                    .tolist()
                )
                finite_values = slot_values[torch.isfinite(slot_values)]
                if int(finite_values.numel()) > 0:
                    finite_min = float(finite_values.min().item())
                    finite_max = float(finite_values.max().item())
                else:
                    finite_min = float("nan")
                    finite_max = float("nan")
                sample_summaries.append(
                    "slot_local="
                    f"{local_slot},class_id={int(valid_class_ids[local_slot].item())},"
                    f"label_count={int(counts[valid_mask][local_slot].item())},"
                    f"init_scale={float(valid_init_scale[local_slot].item()):.6g},"
                    f"init_offset={float(valid_init_offset[local_slot].item()):.6g},"
                    f"pos_weight={float(valid_pos_weight[local_slot].item()):.6g},"
                    f"bad_positions={bad_positions},"
                    f"finite_min={finite_min:.6g},finite_max={finite_max:.6g}"
                )
            raise RuntimeError(
                "Non-finite classwise affine fit logits detected before scale identifiability check; "
                f"context={diagnostic_context!r}, "
                f"logits_shape={tuple(logits_selected.shape)}, "
                f"valid_slot_count={int(valid_mask.sum().item())}, "
                f"selected_class_ids={valid_class_ids[:8].tolist()}, "
                f"sample_details=[{'; '.join(sample_summaries)}]."
            )
        valid_joint_scale_identifiable = _joint_scale_identifiability_mask(
            valid_xs,
            diagnostic_context=diagnostic_context,
        )
        valid_bias_init_scale = torch.where(
            valid_joint_scale_identifiable,
            valid_init_scale,
            torch.zeros_like(valid_init_scale),
        )

        if not use_map_prior:
            scale_bias_valid, offset_bias_valid, loss_bias_valid = _optimize_wb_logistic_classwise(
                valid_xs,
                labels_long,
                valid_class_ids,
                valid_bias_init_scale,
                valid_init_offset,
                valid_pos_weight,
                optimize_scale=False,
                iters=num_iterations,
                damping_init=damping_init,
                class_chunk_size=class_chunk_size,
                use_triton_grad_hess=use_triton_grad_hess,
                min_triton_slots=resolved_min_triton_slots,
                compact_active_slots=compact_active_slots,
            )
            scale_bias[valid_mask] = scale_bias_valid
            offset_bias[valid_mask] = offset_bias_valid
            loss_bias[valid_mask] = loss_bias_valid

            if optimize_classifier_scale_joint:
                scale_joint_valid = scale_bias_valid.clone()
                offset_joint_valid = offset_bias_valid.clone()
                loss_joint_valid = loss_bias_valid.clone()
                if bool(valid_joint_scale_identifiable.any().item()):
                    joint_xs = valid_xs[valid_joint_scale_identifiable]
                    joint_class_ids = valid_class_ids[valid_joint_scale_identifiable]
                    joint_pos_weight = valid_pos_weight[valid_joint_scale_identifiable]
                    joint_scale_init = scale_bias_valid[valid_joint_scale_identifiable]
                    joint_offset_init = offset_bias_valid[valid_joint_scale_identifiable]
                    (
                        joint_scale_valid,
                        joint_offset_valid,
                        joint_loss_valid,
                    ) = _optimize_wb_logistic_classwise(
                        joint_xs,
                        labels_long,
                        joint_class_ids,
                        joint_scale_init,
                        joint_offset_init,
                        joint_pos_weight,
                        optimize_scale=True,
                        iters=num_iterations,
                        damping_init=damping_init,
                        class_chunk_size=class_chunk_size,
                        use_triton_grad_hess=use_triton_grad_hess,
                        min_triton_slots=resolved_min_triton_slots,
                        compact_active_slots=compact_active_slots,
                    )
                    scale_joint_valid[valid_joint_scale_identifiable] = joint_scale_valid
                    offset_joint_valid[valid_joint_scale_identifiable] = joint_offset_valid
                    loss_joint_valid[valid_joint_scale_identifiable] = joint_loss_valid
                scale_joint[valid_mask] = scale_joint_valid
                offset_joint[valid_mask] = offset_joint_valid
                loss_joint[valid_mask] = loss_joint_valid
            else:
                scale_joint = scale_bias.clone()
                offset_joint = offset_bias.clone()
                loss_joint = loss_bias.clone()
        else:
            valid_global_idx = torch.nonzero(valid_mask, as_tuple=False).view(-1)

            def _compute_slot_raw_objective(
                slot_xs: torch.Tensor,
                slot_class_id: torch.Tensor,
                slot_w: torch.Tensor,
                slot_b: torch.Tensor,
                slot_pos_weight: torch.Tensor,
            ):
                use_triton_slot = _should_use_triton_classwise_backend(
                    use_triton_grad_hess=use_triton_grad_hess,
                    is_cuda=bool(slot_xs.is_cuda),
                    num_slots=int(slot_xs.shape[0]),
                    min_triton_slots=resolved_min_triton_slots,
                )
                if use_triton_slot:
                    labels_slot = labels_long.to(device=device, dtype=torch.int32, non_blocking=True).contiguous()
                    class_ids_slot = slot_class_id.to(device=device, dtype=torch.int32, non_blocking=True).contiguous()
                    pos_w_slot = slot_pos_weight.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
                    empty_mask = torch.empty((0,), device=device, dtype=torch.float32)
                    return _classwise_grad_hess_triton_prepacked(
                        slot_xs,
                        labels_slot,
                        class_ids_slot,
                        slot_w,
                        slot_b,
                        pos_w_slot,
                        sample_mask_c=empty_mask,
                        has_sample_mask=False,
                        out_buffers=None,
                    )[:6]
                return _classwise_grad_hess_torch(
                    slot_xs,
                    labels_long,
                    slot_class_id,
                    slot_w,
                    slot_b,
                    slot_pos_weight,
                    class_chunk_size=1,
                )[:6]

            def _run_single_stage(
            slot_xs: torch.Tensor,
            slot_class_id: torch.Tensor,
            slot_init_scale: torch.Tensor,
            slot_init_offset: torch.Tensor,
            slot_pos_weight: torch.Tensor,
            *,
            optimize_scale_stage: bool,
        ):
                n_eff = _compute_selected_effective_n(
                labels_long,
                slot_class_id,
                device=device,
                dtype=compute_dtype,
                ).view(-1)
                x_std = slot_xs.transpose(0, 1).contiguous().std(dim=0, unbiased=True).clamp_min(float(map_prior_eps))
                sigma_w = torch.as_tensor(float(map_prior_c), device=device, dtype=compute_dtype) / x_std
                sigma_b = torch.full_like(x_std, float(map_prior_c))
                fixed_lambda_w = (
                    1.0 / (n_eff * sigma_w.square()).clamp_min(float(map_prior_eps))
                    if optimize_scale_stage
                    else torch.zeros_like(x_std)
                )
                fixed_lambda_b = 1.0 / (n_eff * sigma_b.square()).clamp_min(float(map_prior_eps))
                if resolved_map_prior_mode == "empirical_bayes":
                    ridge = float(map_prior_hessian_ridge)
                    _, _, _, base_H_ww, _, base_H_bb = _compute_slot_raw_objective(
                        slot_xs,
                        slot_class_id,
                        torch.ones_like(slot_init_scale),
                        torch.zeros_like(slot_init_offset),
                        slot_pos_weight,
                    )
                    base_lambda_w = (
                        base_H_ww.abs().clamp_min(ridge)
                        if optimize_scale_stage
                        else torch.zeros_like(base_H_bb)
                    )
                    base_lambda_b = base_H_bb.abs().clamp_min(ridge)
                    exp_offsets = torch.linspace(
                        -float(map_prior_grid_span),
                        float(map_prior_grid_span),
                        steps=max(1, int(map_prior_grid_size)),
                        device=device,
                        dtype=compute_dtype,
                    )
                    grid_scale = torch.pow(torch.tensor(10.0, device=device, dtype=compute_dtype), exp_offsets)
                    best_log_evid = None
                    best_lambda_w = fixed_lambda_w.clone()
                    best_lambda_b = fixed_lambda_b.clone()
                    grid_iters = max(1, min(int(num_iterations), int(map_prior_grid_max_iters)))
                    for scale_factor in grid_scale:
                        cand_lambda_w = (
                            base_lambda_w * scale_factor
                            if optimize_scale_stage
                            else torch.zeros_like(base_lambda_b)
                        )
                        for bias_scale_factor in grid_scale:
                            cand_lambda_b = base_lambda_b * bias_scale_factor
                            cand_w, cand_b, cand_loss = _optimize_wb_logistic_classwise(
                                slot_xs,
                                labels_long,
                                slot_class_id,
                                slot_init_scale,
                                slot_init_offset,
                                slot_pos_weight,
                                optimize_scale=optimize_scale_stage,
                                iters=grid_iters,
                                damping_init=damping_init,
                                class_chunk_size=1,
                                use_triton_grad_hess=use_triton_grad_hess,
                                min_triton_slots=resolved_min_triton_slots,
                                compact_active_slots=compact_active_slots,
                                prior_lambda_w=cand_lambda_w,
                                prior_lambda_b=cand_lambda_b,
                            )
                            _, _, _, H_ww_hat_raw, H_wb_hat_raw, H_bb_hat_raw = _compute_slot_raw_objective(
                                slot_xs,
                                slot_class_id,
                                cand_w,
                                cand_b,
                                slot_pos_weight,
                            )
                            H_ww_hat = H_ww_hat_raw + cand_lambda_w if optimize_scale_stage else H_ww_hat_raw
                            H_wb_hat = H_wb_hat_raw
                            H_bb_hat = H_bb_hat_raw + cand_lambda_b
                            log_evid = _compute_log_evidence_classwise(
                                obj_hat=cand_loss,
                                H_ww=H_ww_hat,
                                H_wb=H_wb_hat,
                                H_bb=H_bb_hat,
                                lambda_w=cand_lambda_w,
                                lambda_b=cand_lambda_b,
                                n_eff=n_eff,
                                optimize_scale=optimize_scale_stage,
                                ridge=ridge,
                            )
                            if best_log_evid is None or log_evid.item() > best_log_evid:
                                best_log_evid = log_evid.item()
                                best_lambda_w = cand_lambda_w
                                best_lambda_b = cand_lambda_b
                    prior_lambda_w = best_lambda_w
                    prior_lambda_b = best_lambda_b
                else:
                    prior_lambda_w = fixed_lambda_w
                    prior_lambda_b = fixed_lambda_b

                return _optimize_wb_logistic_classwise(
                    slot_xs,
                    labels_long,
                    slot_class_id,
                    slot_init_scale,
                    slot_init_offset,
                    slot_pos_weight,
                    optimize_scale=optimize_scale_stage,
                    iters=num_iterations,
                    damping_init=damping_init,
                    class_chunk_size=1,
                    use_triton_grad_hess=use_triton_grad_hess,
                    min_triton_slots=resolved_min_triton_slots,
                    compact_active_slots=compact_active_slots,
                    prior_lambda_w=prior_lambda_w,
                    prior_lambda_b=prior_lambda_b,
                )

            for slot_idx in range(int(valid_xs.shape[0])):
                global_idx = int(valid_global_idx[slot_idx].item())
                slot_slice = slice(slot_idx, slot_idx + 1)
                slot_joint_scale_identifiable = bool(
                    valid_joint_scale_identifiable[slot_idx].item()
                )
                slot_init_scale_bias = (
                    valid_init_scale[slot_slice]
                    if slot_joint_scale_identifiable
                    else torch.zeros_like(valid_init_scale[slot_slice])
                )
                slot_scale_bias, slot_offset_bias, slot_loss_bias = _run_single_stage(
                    valid_xs[slot_slice],
                    valid_class_ids[slot_slice],
                    slot_init_scale_bias,
                    valid_init_offset[slot_slice],
                    valid_pos_weight[slot_slice],
                    optimize_scale_stage=False,
                )
                scale_bias[global_idx] = slot_scale_bias[0]
                offset_bias[global_idx] = slot_offset_bias[0]
                loss_bias[global_idx] = slot_loss_bias[0]
                if optimize_classifier_scale_joint and slot_joint_scale_identifiable:
                    slot_scale_joint, slot_offset_joint, slot_loss_joint = _run_single_stage(
                        valid_xs[slot_slice],
                        valid_class_ids[slot_slice],
                        slot_scale_bias,
                        slot_offset_bias,
                        valid_pos_weight[slot_slice],
                        optimize_scale_stage=True,
                    )
                    scale_joint[global_idx] = slot_scale_joint[0]
                    offset_joint[global_idx] = slot_offset_joint[0]
                    loss_joint[global_idx] = slot_loss_joint[0]
                else:
                    scale_joint[global_idx] = slot_scale_bias[0]
                    offset_joint[global_idx] = slot_offset_bias[0]
                    loss_joint[global_idx] = slot_loss_bias[0]

    _raise_on_nonfinite_fit_result(
        classifier_scale=scale_bias,
        classifier_offset=offset_bias,
        classifier_scale_optimized=scale_joint,
        classifier_offset_optimized=offset_joint,
        bias_only_loss=loss_bias,
        optimized_loss=loss_joint,
        valid_mask=valid_mask,
    )

    return ClasswiseAffineFitResult(
        classifier_offset=offset_bias,
        classifier_scale=scale_bias,
        classifier_offset_optimized=offset_joint,
        classifier_scale_optimized=scale_joint,
        pos_weight=pos_weight,
        bias_only_loss=loss_bias,
        optimized_loss=loss_joint,
    )


def fit_single_binary_affine_calibration(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    positive_class_id: int,
    init_classifier_offset: Optional[torch.Tensor] = None,
    init_classifier_scale: Optional[torch.Tensor] = None,
    optimize_classifier_scale_joint: bool = True,
    num_iterations: int = 80,
    damping_init: float = 1e-4,
    use_triton_grad_hess: bool = True,
    min_triton_slots: Optional[int] = None,
    use_map_prior: bool = False,
    map_prior_c: float = 0.1,
    map_prior_eps: float = 1e-6,
    map_prior_mode: str = "off",
    map_prior_grid_size: int = 7,
    map_prior_grid_span: float = 2.0,
    map_prior_grid_max_iters: int = 200,
    map_prior_hessian_ridge: float = 1e-8,
    affine_calibration_mode: str = "hessian_minimization",
    diagnostic_context: Optional[str] = None,
) -> ClasswiseAffineFitResult:
    """Compatibility wrapper for the historical single-neuron affine API.

    New route-native pruning/evaluation code should use
    :func:`fit_binary_target_direction_affine_calibration`, which treats a
    class unit as the one-column special case of the generalized target
    direction objective.  This wrapper stays public because the frozen
    old-sparse parity path and archived audit scripts import it by name.
    """

    logits_matrix = logits.view(-1, 1)
    class_ids = torch.tensor([int(positive_class_id)], device=logits_matrix.device, dtype=torch.long)
    return fit_selected_binary_affine_calibration(
        logits_matrix,
        labels,
        selected_class_ids=class_ids,
        init_classifier_offset=init_classifier_offset,
        init_classifier_scale=init_classifier_scale,
        optimize_classifier_scale_joint=optimize_classifier_scale_joint,
        num_iterations=num_iterations,
        damping_init=damping_init,
        class_chunk_size=1,
        use_triton_grad_hess=use_triton_grad_hess,
        min_triton_slots=min_triton_slots,
        use_map_prior=use_map_prior,
        map_prior_c=map_prior_c,
        map_prior_eps=map_prior_eps,
        map_prior_mode=map_prior_mode,
        map_prior_grid_size=map_prior_grid_size,
        map_prior_grid_span=map_prior_grid_span,
        map_prior_grid_max_iters=map_prior_grid_max_iters,
        map_prior_hessian_ridge=map_prior_hessian_ridge,
        affine_calibration_mode=affine_calibration_mode,
        diagnostic_context=diagnostic_context,
    )


def fit_binary_target_direction_affine_calibration(
    logits: torch.Tensor,
    binary_targets: torch.Tensor,
    *,
    init_classifier_offset: Optional[torch.Tensor] = None,
    init_classifier_scale: Optional[torch.Tensor] = None,
    optimize_classifier_scale_joint: bool = True,
    num_iterations: int = 80,
    damping_init: float = 1e-4,
    class_chunk_size: int = 128,
    enforce_positive_scale: bool = False,
    positive_scale_min: float = 1e-8,
    lr: float = 0.25,
    diagnostic_context: Optional[str] = None,
) -> ClasswiseAffineFitResult:
    """Canonical affine calibration for one or many binary target directions.

    ``logits`` is the scalar target signal ``z_t(x)`` for one target with shape
    ``[N]`` or the target-signal matrix ``Z = R U^T`` with shape ``[N, K]``.
    ``binary_targets`` has the same shape and declares the positive samples for
    each direction.  A neuron-aligned class unit is exactly the one-column
    special case where ``binary_targets[:, 0] = labels == unit``.
    """

    logits_matrix = logits.view(-1, 1) if logits.ndim == 1 else logits
    target_matrix = binary_targets.view(-1, 1) if binary_targets.ndim == 1 else binary_targets
    if not bool(enforce_positive_scale):
        return fit_binary_target_matrix_affine_calibration(
            logits_matrix,
            target_matrix,
            init_classifier_offset=init_classifier_offset,
            init_classifier_scale=init_classifier_scale,
            optimize_classifier_scale_joint=optimize_classifier_scale_joint,
            num_iterations=num_iterations,
            damping_init=damping_init,
            class_chunk_size=class_chunk_size,
            diagnostic_context=diagnostic_context,
        )

    bias_fit = fit_binary_target_matrix_affine_calibration(
        logits_matrix,
        target_matrix,
        init_classifier_offset=init_classifier_offset,
        init_classifier_scale=init_classifier_scale,
        optimize_classifier_scale_joint=False,
        num_iterations=num_iterations,
        damping_init=damping_init,
        class_chunk_size=class_chunk_size,
        diagnostic_context=diagnostic_context,
    )
    if not bool(optimize_classifier_scale_joint):
        return bias_fit

    if logits_matrix.ndim != 2:
        raise ValueError("logits must have shape [N] or [N, K].")
    if target_matrix.shape != logits_matrix.shape:
        raise ValueError("binary_targets must have the same shape as logits.")

    with torch.inference_mode(False):
        device = logits_matrix.device
        compute_dtype = _safe_float_dtype(logits_matrix.dtype)
        x_all = logits_matrix.detach().clone().to(device=device, dtype=compute_dtype)
        y_all = target_matrix.detach().clone().to(device=device, dtype=compute_dtype).clamp(0.0, 1.0)
        num_samples, num_slots = int(x_all.shape[0]), int(x_all.shape[1])
        pos_counts = y_all.sum(dim=0)
        valid_mask = (pos_counts > 0) & (pos_counts < float(num_samples))
        pos_weight = compute_binary_target_matrix_pos_weight(
            y_all,
            device=device,
            dtype=compute_dtype,
        )

        scale_joint = bias_fit.classifier_scale_optimized.detach().clone().to(device=device, dtype=compute_dtype)
        offset_joint = bias_fit.classifier_offset_optimized.detach().clone().to(device=device, dtype=compute_dtype)
        loss_joint = torch.full((num_slots,), float("inf"), device=device, dtype=compute_dtype)
        denom = torch.tensor(float(num_samples), device=device, dtype=compute_dtype).clamp_min(1.0)
        ws_min_t = torch.tensor(float(positive_scale_min), device=device, dtype=compute_dtype).clamp_min(1e-12)

        for start in range(0, num_slots, max(1, int(class_chunk_size))):
            end = min(start + int(class_chunk_size), num_slots)
            valid = valid_mask[start:end]
            if not bool(valid.any().item()):
                continue
            x = x_all[:, start:end][:, valid]
            y = y_all[:, start:end][:, valid]
            pw = pos_weight[start:end][valid]
            init_scale_chunk = (
                init_classifier_scale.to(device=device, dtype=compute_dtype).view(-1)[start:end][valid]
                if init_classifier_scale is not None
                else torch.ones((int(valid.sum().item()),), device=device, dtype=compute_dtype)
            ).clamp_min(float(positive_scale_min) + 1e-8)
            init_offset_chunk = offset_joint[start:end][valid]
            raw_scale = torch.nn.Parameter(
                _inverse_softplus_positive(init_scale_chunk - ws_min_t, eps=1e-8).detach().clone()
            )
            offset = torch.nn.Parameter(init_offset_chunk.detach().clone())
            optimizer = torch.optim.LBFGS(
                [raw_scale, offset],
                lr=float(lr),
                max_iter=max(1, int(num_iterations)),
                line_search_fn="strong_wolfe",
            )

            def closure() -> torch.Tensor:
                optimizer.zero_grad(set_to_none=True)
                scale = F.softplus(raw_scale) + ws_min_t
                eta = x * scale.unsqueeze(0) + offset.unsqueeze(0)
                weight = torch.where(y > 0.5, pw.unsqueeze(0), torch.ones_like(y))
                loss = (
                    torch.where(y > 0.5, F.softplus(-eta), F.softplus(eta)) * weight
                ).sum(dim=0) / denom
                total = loss.sum()
                total.backward()
                return total

            try:
                optimizer.step(closure)
            except Exception as exc:
                context = f" context={diagnostic_context!r}" if diagnostic_context else ""
                raise RuntimeError(f"Positive target-direction affine optimization failed.{context}") from exc

            with torch.no_grad():
                scale_opt = F.softplus(raw_scale.detach()) + ws_min_t
                offset_opt = offset.detach()
                eta = x * scale_opt.unsqueeze(0) + offset_opt.unsqueeze(0)
                weight = torch.where(y > 0.5, pw.unsqueeze(0), torch.ones_like(y))
                loss_opt = (
                    torch.where(y > 0.5, F.softplus(-eta), F.softplus(eta)) * weight
                ).sum(dim=0) / denom
                local_scales = scale_joint[start:end].clone()
                local_offsets = offset_joint[start:end].clone()
                local_losses = loss_joint[start:end].clone()
                local_scales[valid] = scale_opt
                local_offsets[valid] = offset_opt
                local_losses[valid] = loss_opt
                scale_joint[start:end] = local_scales
                offset_joint[start:end] = local_offsets
                loss_joint[start:end] = local_losses

    _raise_on_nonfinite_fit_result(
        classifier_scale=bias_fit.classifier_scale,
        classifier_offset=bias_fit.classifier_offset,
        classifier_scale_optimized=scale_joint,
        classifier_offset_optimized=offset_joint,
        bias_only_loss=bias_fit.bias_only_loss,
        optimized_loss=loss_joint,
        valid_mask=valid_mask,
    )
    return ClasswiseAffineFitResult(
        classifier_offset=bias_fit.classifier_offset,
        classifier_scale=bias_fit.classifier_scale,
        classifier_offset_optimized=offset_joint,
        classifier_scale_optimized=scale_joint,
        pos_weight=pos_weight,
        bias_only_loss=bias_fit.bias_only_loss,
        optimized_loss=loss_joint,
    )


def fit_shared_readout_diagonal_affine_calibration(
    readout: torch.Tensor,
    target_vectors: torch.Tensor,
    binary_targets: torch.Tensor,
    *,
    init_readout_offset: Optional[torch.Tensor] = None,
    init_readout_scale: Optional[torch.Tensor] = None,
    optimize_readout_scale_joint: bool = True,
    num_iterations: int = 80,
    enforce_positive_scale: bool = True,
    positive_scale_min: float = 1e-8,
    lr: float = 0.25,
    identity_l2: float = 0.0,
    offset_l2: float = 0.0,
    shared_readout_backend: str = "auto",
    use_triton_grad_hess: bool = True,
    diagnostic_context: Optional[str] = None,
) -> ReadoutSpaceAffineFitResult:
    """Fit coherent shared diagonal calibration in readout-feature space.

    Parameters
    ----------
    readout:
        Dense/pruned readout representation ``R`` with shape ``[N, D]``.
    target_vectors:
        Target modes ``U`` with shape ``[K, D]``.  For supervised class units,
        ``U`` is one-hot and this objective exactly reduces to per-output-unit
        affine calibration.
    binary_targets:
        Binary target labels with shape ``[N, K]``.
    """

    if readout.ndim != 2:
        raise ValueError(f"readout must have shape [N, D], got {tuple(readout.shape)}.")
    if target_vectors.ndim != 2:
        raise ValueError(f"target_vectors must have shape [K, D], got {tuple(target_vectors.shape)}.")
    if binary_targets.ndim != 2:
        raise ValueError(f"binary_targets must have shape [N, K], got {tuple(binary_targets.shape)}.")
    if int(readout.shape[1]) != int(target_vectors.shape[1]):
        raise ValueError(
            f"readout width {int(readout.shape[1])} does not match target vector width {int(target_vectors.shape[1])}."
        )
    if int(readout.shape[0]) != int(binary_targets.shape[0]) or int(target_vectors.shape[0]) != int(binary_targets.shape[1]):
        raise ValueError(
            "binary_targets shape must match readout/target vectors: expected "
            f"[{int(readout.shape[0])}, {int(target_vectors.shape[0])}], got {tuple(binary_targets.shape)}."
        )

    with torch.inference_mode(False):
        device = readout.device
        compute_dtype = _safe_float_dtype(readout.dtype)
        r = readout.detach().clone().to(device=device, dtype=compute_dtype)
        u = target_vectors.detach().clone().to(device=device, dtype=compute_dtype)
        y = binary_targets.detach().clone().to(device=device, dtype=compute_dtype).clamp(0.0, 1.0)
        num_samples, readout_width = int(r.shape[0]), int(r.shape[1])
        num_targets = int(u.shape[0])
        pos_counts = y.sum(dim=0)
        valid_mask = (pos_counts > 0) & (pos_counts < float(num_samples))
        pos_weight = compute_binary_target_matrix_pos_weight(y, device=device, dtype=compute_dtype)

        if init_readout_scale is None:
            init_scale = torch.ones((readout_width,), device=device, dtype=compute_dtype)
        else:
            init_scale = init_readout_scale.to(device=device, dtype=compute_dtype).view(-1)
            if int(init_scale.numel()) != readout_width:
                raise ValueError("init_readout_scale width does not match readout width.")
        if init_readout_offset is None:
            init_offset = torch.zeros((readout_width,), device=device, dtype=compute_dtype)
        else:
            init_offset = init_readout_offset.to(device=device, dtype=compute_dtype).view(-1)
            if int(init_offset.numel()) != readout_width:
                raise ValueError("init_readout_offset width does not match readout width.")

        one_hot_columns = _one_hot_target_vector_columns(u)
        if (
            one_hot_columns is not None
            and float(identity_l2) == 0.0
            and float(offset_l2) == 0.0
            and len(set(int(value) for value in one_hot_columns.detach().cpu().tolist())) == int(one_hot_columns.numel())
        ):
            selected_fit = fit_binary_target_matrix_affine_calibration(
                r.index_select(1, one_hot_columns),
                y,
                init_classifier_offset=init_offset.index_select(0, one_hot_columns),
                init_classifier_scale=init_scale.index_select(0, one_hot_columns),
                optimize_classifier_scale_joint=bool(optimize_readout_scale_joint),
                num_iterations=int(num_iterations),
                diagnostic_context=diagnostic_context,
            )
            optimized_scale = torch.ones((readout_width,), device=device, dtype=compute_dtype)
            optimized_offset = torch.zeros((readout_width,), device=device, dtype=compute_dtype)
            optimized_scale.index_copy_(
                0,
                one_hot_columns,
                selected_fit.classifier_scale_optimized.to(device=device, dtype=compute_dtype),
            )
            optimized_offset.index_copy_(
                0,
                one_hot_columns,
                selected_fit.classifier_offset_optimized.to(device=device, dtype=compute_dtype),
            )
            return ReadoutSpaceAffineFitResult(
                readout_scale=optimized_scale.detach().clone(),
                readout_offset=optimized_offset.detach().clone(),
                target_vectors=u.detach().clone(),
                pos_weight=selected_fit.pos_weight.detach().clone(),
                bias_only_loss=selected_fit.bias_only_loss.detach().clone(),
                optimized_loss=selected_fit.optimized_loss.detach().clone(),
                calibration_backend="classwise_triton_or_torch_one_hot_reduction",
            )

        denom = torch.tensor(float(num_samples), device=device, dtype=compute_dtype).clamp_min(1.0)
        identity_l2_t = torch.tensor(float(identity_l2), device=device, dtype=compute_dtype).clamp_min(0.0)
        offset_l2_t = torch.tensor(float(offset_l2), device=device, dtype=compute_dtype).clamp_min(0.0)
        ws_min_t = torch.tensor(float(positive_scale_min), device=device, dtype=compute_dtype).clamp_min(1e-12)
        requested_backend = str(shared_readout_backend or os.environ.get("CIRCUIT_SHARED_READOUT_C2_BACKEND", "auto"))
        resolved_backend = _resolve_shared_readout_backend(requested_backend, readout=r)
        if not bool(use_triton_grad_hess) and resolved_backend == "triton":
            resolved_backend = "torch_analytic"

        def _reference_target_loss(scale: torch.Tensor, offset_value: torch.Tensor) -> torch.Tensor:
            eta = (r * scale.unsqueeze(0)).matmul(u.t()) + u.matmul(offset_value).unsqueeze(0)
            weight = torch.where(y > 0.5, pos_weight.unsqueeze(0), torch.ones_like(y))
            loss = torch.where(y > 0.5, F.softplus(-eta), F.softplus(eta)) * weight
            return loss.sum(dim=0) / denom

        def _run_reference_fit() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
            if enforce_positive_scale:
                raw_scale_ref = torch.nn.Parameter(
                    _inverse_softplus_positive(init_scale.clamp_min(float(positive_scale_min) + 1e-8) - ws_min_t)
                    .detach()
                    .clone()
                )
            else:
                raw_scale_ref = torch.nn.Parameter(init_scale.detach().clone())
            offset_ref = torch.nn.Parameter(init_offset.detach().clone())
            params_ref = [offset_ref] + ([raw_scale_ref] if optimize_readout_scale_joint else [])
            optimizer_ref = torch.optim.LBFGS(
                params_ref,
                lr=float(lr),
                max_iter=max(1, int(num_iterations)),
                line_search_fn="strong_wolfe",
            )

            def _scale_ref() -> torch.Tensor:
                if not optimize_readout_scale_joint:
                    return init_scale
                return F.softplus(raw_scale_ref) + ws_min_t if enforce_positive_scale else raw_scale_ref

            def closure_ref() -> torch.Tensor:
                optimizer_ref.zero_grad(set_to_none=True)
                scale_ref = _scale_ref()
                losses_ref = _reference_target_loss(scale_ref, offset_ref)
                valid_losses_ref = losses_ref[valid_mask]
                total_ref = valid_losses_ref.mean() if valid_losses_ref.numel() else losses_ref.sum() * 0.0
                if bool((identity_l2_t > 0).item()) and optimize_readout_scale_joint:
                    total_ref = total_ref + 0.5 * identity_l2_t * (scale_ref - 1.0).square().mean()
                if bool((offset_l2_t > 0).item()):
                    total_ref = total_ref + 0.5 * offset_l2_t * offset_ref.square().mean()
                total_ref.backward()
                return total_ref

            optimizer_ref.step(closure_ref)
            with torch.no_grad():
                scale_ref = _scale_ref().detach().clone()
                offset_value_ref = offset_ref.detach().clone()
                return (
                    scale_ref,
                    offset_value_ref,
                    _reference_target_loss(torch.ones_like(scale_ref), offset_value_ref),
                    _reference_target_loss(scale_ref, offset_value_ref),
                    "shared_readout_lbfgs_reference",
                )

        def _run_fused_fit(backend_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
            if enforce_positive_scale:
                raw_scale_fused = torch.nn.Parameter(
                    _inverse_softplus_positive(init_scale.clamp_min(float(positive_scale_min) + 1e-8) - ws_min_t)
                    .detach()
                    .clone()
                )
            else:
                raw_scale_fused = torch.nn.Parameter(init_scale.detach().clone())
            offset_fused = torch.nn.Parameter(init_offset.detach().clone())
            params_fused = [offset_fused] + ([raw_scale_fused] if optimize_readout_scale_joint else [])
            optimizer_fused = torch.optim.LBFGS(
                params_fused,
                lr=float(lr),
                max_iter=max(1, int(num_iterations)),
                line_search_fn="strong_wolfe",
            )

            def _scale_fused() -> torch.Tensor:
                if not optimize_readout_scale_joint:
                    return init_scale
                return F.softplus(raw_scale_fused) + ws_min_t if enforce_positive_scale else raw_scale_fused

            def closure_fused() -> torch.Tensor:
                optimizer_fused.zero_grad(set_to_none=True)
                scale_fused = _scale_fused()
                losses_fused, grad_scale, grad_offset = _shared_readout_loss_grad(
                    r,
                    u,
                    y,
                    scale_fused,
                    offset_fused,
                    pos_weight,
                    valid_mask,
                    backend=backend_name,
                )
                valid_losses_fused = losses_fused[valid_mask]
                total_fused = valid_losses_fused.mean() if valid_losses_fused.numel() else losses_fused.sum() * 0.0
                if bool((identity_l2_t > 0).item()) and optimize_readout_scale_joint:
                    total_fused = total_fused + 0.5 * identity_l2_t * (scale_fused - 1.0).square().mean()
                    grad_scale = grad_scale + identity_l2_t * (scale_fused - 1.0) / max(float(readout_width), 1.0)
                if bool((offset_l2_t > 0).item()):
                    total_fused = total_fused + 0.5 * offset_l2_t * offset_fused.square().mean()
                    grad_offset = grad_offset + offset_l2_t * offset_fused / max(float(readout_width), 1.0)
                if optimize_readout_scale_joint:
                    if enforce_positive_scale:
                        raw_scale_fused.grad = grad_scale * torch.sigmoid(raw_scale_fused.detach())
                    else:
                        raw_scale_fused.grad = grad_scale.detach().clone()
                offset_fused.grad = grad_offset.detach().clone()
                return total_fused.detach().clone().requires_grad_(True)

            optimizer_fused.step(closure_fused)
            with torch.no_grad():
                scale_fused = _scale_fused().detach().clone()
                offset_value_fused = offset_fused.detach().clone()
                return (
                    scale_fused,
                    offset_value_fused,
                    _reference_target_loss(torch.ones_like(scale_fused), offset_value_fused),
                    _reference_target_loss(scale_fused, offset_value_fused),
                    f"shared_readout_{backend_name}",
                )

        try:
            if resolved_backend == "reference":
                optimized_scale, optimized_offset, bias_only_loss, optimized_loss, calibration_backend = _run_reference_fit()
            else:
                optimized_scale, optimized_offset, bias_only_loss, optimized_loss, calibration_backend = _run_fused_fit(resolved_backend)
        except Exception as exc:
            if str(requested_backend).strip().lower().replace("-", "_") in {"auto", "fast", "fused", ""}:
                optimized_scale, optimized_offset, bias_only_loss, optimized_loss, calibration_backend = _run_reference_fit()
                calibration_backend = "shared_readout_lbfgs_reference_fallback"
            else:
                context = f" context={diagnostic_context!r}" if diagnostic_context else ""
                raise RuntimeError(
                    f"Shared readout diagonal affine optimization failed with backend={resolved_backend!r}.{context}"
                ) from exc

    bad_parts = []
    if not bool(torch.isfinite(optimized_scale).all().item()):
        bad_parts.append("readout_scale")
    if not bool(torch.isfinite(optimized_offset).all().item()):
        bad_parts.append("readout_offset")
    if bool((valid_mask & (~torch.isfinite(bias_only_loss))).any().item()):
        bad_parts.append("bias_only_loss")
    if bool((valid_mask & (~torch.isfinite(optimized_loss))).any().item()):
        bad_parts.append("optimized_loss")
    if bad_parts:
        context = f" context={diagnostic_context!r}" if diagnostic_context else ""
        raise RuntimeError(
            "Non-finite shared readout diagonal affine fit result "
            f"for fields {bad_parts}.{context}"
        )
    return ReadoutSpaceAffineFitResult(
        readout_scale=optimized_scale,
        readout_offset=optimized_offset,
        target_vectors=u.detach().clone(),
        pos_weight=pos_weight,
        bias_only_loss=bias_only_loss,
        optimized_loss=optimized_loss,
        calibration_backend=calibration_backend,
    )


def fit_batched_shared_readout_diagonal_affine_calibration(
    readouts: torch.Tensor,
    target_vectors: torch.Tensor,
    binary_targets: torch.Tensor,
    *,
    target_intercepts: Optional[torch.Tensor] = None,
    init_readout_offset: Optional[torch.Tensor] = None,
    init_readout_scale: Optional[torch.Tensor] = None,
    optimize_readout_scale_joint: bool = True,
    num_iterations: int = 80,
    enforce_positive_scale: bool = True,
    positive_scale_min: float = 1e-8,
    lr: float = 0.25,
    identity_l2: float = 0.0,
    offset_l2: float = 0.0,
    shared_readout_backend: str = "auto",
    shared_readout_solver: str = "lbfgs",
    use_triton_grad_hess: bool = True,
    diagnostic_context: Optional[str] = None,
) -> BatchedReadoutSpaceAffineFitResult:
    """Fit independent C2 shared-readout calibrations for a mask batch.

    ``readouts[m]`` is fitted independently, but the implementation batches the
    objective and gradient evaluation.  The result should match looping over
    ``fit_shared_readout_diagonal_affine_calibration(readouts[m], ...)`` within
    ordinary float32/LBFGS tolerance.
    """

    if readouts.ndim != 3:
        raise ValueError(f"readouts must have shape [M, N, D], got {tuple(readouts.shape)}.")
    if target_vectors.ndim != 2:
        raise ValueError(f"target_vectors must have shape [K, D], got {tuple(target_vectors.shape)}.")
    if binary_targets.ndim != 2:
        raise ValueError(f"binary_targets must have shape [N, K], got {tuple(binary_targets.shape)}.")
    if int(readouts.shape[2]) != int(target_vectors.shape[1]):
        raise ValueError(
            f"readout width {int(readouts.shape[2])} does not match target vector width "
            f"{int(target_vectors.shape[1])}."
        )
    if int(readouts.shape[1]) != int(binary_targets.shape[0]) or int(target_vectors.shape[0]) != int(binary_targets.shape[1]):
        raise ValueError(
            "binary_targets shape must match readouts/target vectors: expected "
            f"[{int(readouts.shape[1])}, {int(target_vectors.shape[0])}], got {tuple(binary_targets.shape)}."
        )

    solver = str(shared_readout_solver or "lbfgs").strip().lower().replace("-", "_")
    if solver in {"auto", "default", "torch_analytic"}:
        solver = "lbfgs"
    if solver not in {"lbfgs", "batched_newton_irls", "newton_irls", "irls"}:
        raise ValueError(f"Unsupported shared_readout_solver={shared_readout_solver!r}.")

    with torch.inference_mode(False):
        device = readouts.device
        compute_dtype = _safe_float_dtype(readouts.dtype)
        r = readouts.detach().clone().to(device=device, dtype=compute_dtype)
        u = target_vectors.detach().clone().to(device=device, dtype=compute_dtype)
        y = binary_targets.detach().clone().to(device=device, dtype=compute_dtype).clamp(0.0, 1.0)
        target_intercepts_t = None
        if target_intercepts is not None:
            target_intercepts_t = target_intercepts.detach().clone().to(device=device, dtype=compute_dtype).reshape(-1)
            if int(target_intercepts_t.numel()) != int(u.shape[0]):
                raise ValueError(
                    "target_intercepts must have one value per target vector: "
                    f"expected {int(u.shape[0])}, got {int(target_intercepts_t.numel())}."
                )
        num_masks, num_samples, readout_width = int(r.shape[0]), int(r.shape[1]), int(r.shape[2])
        pos_counts = y.sum(dim=0)
        valid_mask = (pos_counts > 0) & (pos_counts < float(num_samples))
        pos_weight = compute_binary_target_matrix_pos_weight(y, device=device, dtype=compute_dtype)

        one_hot_columns = _one_hot_target_vector_columns(u)
        if (
            one_hot_columns is not None
            and (target_intercepts_t is None or bool(torch.allclose(target_intercepts_t, torch.zeros_like(target_intercepts_t))))
            and float(identity_l2) == 0.0
            and float(offset_l2) == 0.0
            and len(set(int(value) for value in one_hot_columns.detach().cpu().tolist())) == int(one_hot_columns.numel())
        ):
            fits = [
                fit_shared_readout_diagonal_affine_calibration(
                    r[mask_index],
                    u,
                    y,
                    init_readout_offset=None
                    if init_readout_offset is None
                    else init_readout_offset[mask_index]
                    if init_readout_offset.ndim == 2
                    else init_readout_offset,
                    init_readout_scale=None
                    if init_readout_scale is None
                    else init_readout_scale[mask_index]
                    if init_readout_scale.ndim == 2
                    else init_readout_scale,
                    optimize_readout_scale_joint=optimize_readout_scale_joint,
                    num_iterations=num_iterations,
                    enforce_positive_scale=enforce_positive_scale,
                    positive_scale_min=positive_scale_min,
                    lr=lr,
                    identity_l2=identity_l2,
                    offset_l2=offset_l2,
                    shared_readout_backend=shared_readout_backend,
                    use_triton_grad_hess=use_triton_grad_hess,
                    diagnostic_context=diagnostic_context,
                )
                for mask_index in range(num_masks)
            ]
            return BatchedReadoutSpaceAffineFitResult(
                readout_scale=torch.stack([fit.readout_scale for fit in fits], dim=0),
                readout_offset=torch.stack([fit.readout_offset for fit in fits], dim=0),
                target_vectors=u.detach().clone(),
                target_intercepts=target_intercepts_t.detach().clone() if target_intercepts_t is not None else None,
                pos_weight=fits[0].pos_weight.detach().clone(),
                bias_only_loss=torch.stack([fit.bias_only_loss for fit in fits], dim=0),
                optimized_loss=torch.stack([fit.optimized_loss for fit in fits], dim=0),
                calibration_backend="batched_classwise_one_hot_reduction",
            )

        if init_readout_scale is None:
            init_scale = torch.ones((num_masks, readout_width), device=device, dtype=compute_dtype)
        else:
            init_scale = init_readout_scale.to(device=device, dtype=compute_dtype)
            if init_scale.ndim == 1:
                init_scale = init_scale.view(1, -1).expand(num_masks, -1).clone()
            if tuple(init_scale.shape) != (num_masks, readout_width):
                raise ValueError("init_readout_scale must have shape [D] or [M, D].")
        if init_readout_offset is None:
            init_offset = torch.zeros((num_masks, readout_width), device=device, dtype=compute_dtype)
        else:
            init_offset = init_readout_offset.to(device=device, dtype=compute_dtype)
            if init_offset.ndim == 1:
                init_offset = init_offset.view(1, -1).expand(num_masks, -1).clone()
            if tuple(init_offset.shape) != (num_masks, readout_width):
                raise ValueError("init_readout_offset must have shape [D] or [M, D].")

        identity_l2_t = torch.tensor(float(identity_l2), device=device, dtype=compute_dtype).clamp_min(0.0)
        offset_l2_t = torch.tensor(float(offset_l2), device=device, dtype=compute_dtype).clamp_min(0.0)
        ws_min_t = torch.tensor(float(positive_scale_min), device=device, dtype=compute_dtype).clamp_min(1e-12)

        def _scale_from_raw(raw_scale_value: torch.Tensor) -> torch.Tensor:
            if not optimize_readout_scale_joint:
                return init_scale
            return F.softplus(raw_scale_value) + ws_min_t if enforce_positive_scale else raw_scale_value

        if enforce_positive_scale:
            raw_scale = torch.nn.Parameter(
                _inverse_softplus_positive(init_scale.clamp_min(float(positive_scale_min) + 1e-8) - ws_min_t)
                .detach()
                .clone()
            )
        else:
            raw_scale = torch.nn.Parameter(init_scale.detach().clone())
        offset = torch.nn.Parameter(init_offset.detach().clone())

        def _objective(scale_value: torch.Tensor, offset_value: torch.Tensor) -> torch.Tensor:
            losses = _shared_readout_losses_torch_batched(
                r,
                u,
                y,
                scale_value,
                offset_value,
                pos_weight,
                target_intercepts_t,
            )
            valid_losses = losses[:, valid_mask]
            total = valid_losses.mean() if valid_losses.numel() else losses.sum() * 0.0
            if bool((identity_l2_t > 0).item()) and optimize_readout_scale_joint:
                total = total + 0.5 * identity_l2_t * (scale_value - 1.0).square().mean()
            if bool((offset_l2_t > 0).item()):
                total = total + 0.5 * offset_l2_t * offset_value.square().mean()
            return total

        if solver == "lbfgs":
            params = [offset] + ([raw_scale] if optimize_readout_scale_joint else [])
            optimizer = torch.optim.LBFGS(
                params,
                lr=float(lr),
                max_iter=max(1, int(num_iterations)),
                line_search_fn="strong_wolfe",
            )

            def closure() -> torch.Tensor:
                optimizer.zero_grad(set_to_none=True)
                scale_value = _scale_from_raw(raw_scale)
                losses, grad_scale, grad_offset = _shared_readout_loss_grad_torch_batched(
                    r,
                    u,
                    y,
                    scale_value,
                    offset,
                    pos_weight,
                    valid_mask,
                    target_intercepts_t,
                )
                valid_losses = losses[:, valid_mask]
                total = valid_losses.mean() if valid_losses.numel() else losses.sum() * 0.0
                if bool((identity_l2_t > 0).item()) and optimize_readout_scale_joint:
                    total = total + 0.5 * identity_l2_t * (scale_value - 1.0).square().mean()
                    grad_scale = grad_scale + identity_l2_t * (scale_value - 1.0) / max(float(num_masks * readout_width), 1.0)
                if bool((offset_l2_t > 0).item()):
                    total = total + 0.5 * offset_l2_t * offset.square().mean()
                    grad_offset = grad_offset + offset_l2_t * offset / max(float(num_masks * readout_width), 1.0)
                if optimize_readout_scale_joint:
                    if enforce_positive_scale:
                        raw_scale.grad = grad_scale * torch.sigmoid(raw_scale.detach())
                    else:
                        raw_scale.grad = grad_scale.detach().clone()
                offset.grad = grad_offset.detach().clone()
                return total.detach().clone().requires_grad_(True)

            optimizer.step(closure)
            calibration_backend = "batched_shared_readout_torch_analytic"
        else:
            # Diagnostic/off-default solver.  It uses a diagonal Fisher/IRLS
            # approximation and guarded line search, so it optimizes the same
            # objective but is not the production default until canaries promote it.
            calibration_backend = "batched_newton_irls_diagnostic"
            for _ in range(max(1, int(num_iterations))):
                with torch.no_grad():
                    scale_value = _scale_from_raw(raw_scale)
                    losses, grad_scale, grad_offset = _shared_readout_loss_grad_torch_batched(
                        r,
                        u,
                        y,
                        scale_value,
                        offset,
                        pos_weight,
                        valid_mask,
                        target_intercepts_t,
                    )
                    calibrated = r * scale_value.unsqueeze(1) + offset.unsqueeze(1)
                    eta = torch.einsum("mnd,kd->mnk", calibrated, u)
                    if target_intercepts_t is not None:
                        eta = eta + target_intercepts_t.view(1, 1, -1)
                    prob = torch.sigmoid(eta)
                    y_b = y.unsqueeze(0)
                    weight = torch.where(y_b > 0.5, pos_weight.view(1, 1, -1), torch.ones_like(y_b))
                    curvature = prob * (1.0 - prob) * weight
                    curvature = curvature * valid_mask.to(dtype=compute_dtype).view(1, 1, -1)
                    h_readout = torch.einsum("mnk,kd->mnd", curvature, u.square()) / max(float(num_samples), 1.0)
                    h_scale = (h_readout * r.square()).sum(dim=1).clamp_min(1e-6)
                    h_offset = h_readout.sum(dim=1).clamp_min(1e-6)
                    if bool((identity_l2_t > 0).item()) and optimize_readout_scale_joint:
                        grad_scale = grad_scale + identity_l2_t * (scale_value - 1.0) / max(float(num_masks * readout_width), 1.0)
                        h_scale = h_scale + identity_l2_t / max(float(num_masks * readout_width), 1.0)
                    if bool((offset_l2_t > 0).item()):
                        grad_offset = grad_offset + offset_l2_t * offset / max(float(num_masks * readout_width), 1.0)
                        h_offset = h_offset + offset_l2_t / max(float(num_masks * readout_width), 1.0)
                    old_obj = _objective(scale_value, offset).detach()
                    step = 1.0
                    accepted = False
                    for _ls in range(8):
                        next_offset = offset - step * grad_offset / h_offset
                        if optimize_readout_scale_joint:
                            next_scale_value = (scale_value - step * grad_scale / h_scale).clamp_min(float(positive_scale_min))
                            if enforce_positive_scale:
                                next_raw = _inverse_softplus_positive(next_scale_value - ws_min_t)
                            else:
                                next_raw = next_scale_value
                        else:
                            next_raw = raw_scale
                        next_obj = _objective(_scale_from_raw(next_raw), next_offset).detach()
                        if bool((next_obj <= old_obj + 1e-7).item()):
                            offset.copy_(next_offset)
                            if optimize_readout_scale_joint:
                                raw_scale.copy_(next_raw)
                            accepted = True
                            break
                        step *= 0.5
                    if not accepted:
                        break

        with torch.no_grad():
            optimized_scale = _scale_from_raw(raw_scale).detach().clone()
            optimized_offset = offset.detach().clone()
            bias_only_loss = _shared_readout_losses_torch_batched(
                r,
                u,
                y,
                torch.ones_like(optimized_scale),
                optimized_offset,
                pos_weight,
                target_intercepts_t,
            )
            optimized_loss = _shared_readout_losses_torch_batched(
                r,
                u,
                y,
                optimized_scale,
                optimized_offset,
                pos_weight,
                target_intercepts_t,
            )

    bad_parts = []
    if not bool(torch.isfinite(optimized_scale).all().item()):
        bad_parts.append("readout_scale")
    if not bool(torch.isfinite(optimized_offset).all().item()):
        bad_parts.append("readout_offset")
    if bool((valid_mask.view(1, -1) & (~torch.isfinite(bias_only_loss))).any().item()):
        bad_parts.append("bias_only_loss")
    if bool((valid_mask.view(1, -1) & (~torch.isfinite(optimized_loss))).any().item()):
        bad_parts.append("optimized_loss")
    if bad_parts:
        context = f" context={diagnostic_context!r}" if diagnostic_context else ""
        raise RuntimeError(
            "Non-finite batched shared readout diagonal affine fit result "
            f"for fields {bad_parts}.{context}"
        )
    return BatchedReadoutSpaceAffineFitResult(
        readout_scale=optimized_scale,
        readout_offset=optimized_offset,
        target_vectors=u.detach().clone(),
        target_intercepts=target_intercepts_t.detach().clone() if target_intercepts_t is not None else None,
        pos_weight=pos_weight,
        bias_only_loss=bias_only_loss,
        optimized_loss=optimized_loss,
        calibration_backend=calibration_backend,
    )


def fit_shared_readout_low_rank_plus_diagonal_affine_calibration(
    readout: torch.Tensor,
    target_vectors: torch.Tensor,
    binary_targets: torch.Tensor,
    *,
    rank: int = 4,
    num_iterations: int = 80,
    lr: float = 0.1,
    positive_scale_min: float = 1e-8,
    enforce_positive_scale: bool = True,
    identity_l2: float = 0.0,
    offset_l2: float = 0.0,
    low_rank_l2: float = 1e-4,
    diagnostic_context: Optional[str] = None,
) -> ReadoutSpaceAffineFitResult:
    """Diagnostic low-rank-plus-diagonal C2 extension.

    This adds readout-repair capacity beyond canonical C2 and is intentionally
    off by default.  It remains useful as a controlled diagnostic/future mode.
    """

    if readout.ndim != 2 or target_vectors.ndim != 2 or binary_targets.ndim != 2:
        raise ValueError("low-rank shared readout calibration expects [N,D], [K,D], [N,K] tensors.")
    if int(readout.shape[1]) != int(target_vectors.shape[1]):
        raise ValueError("readout and target vector widths must match.")
    if int(binary_targets.shape[0]) != int(readout.shape[0]) or int(binary_targets.shape[1]) != int(target_vectors.shape[0]):
        raise ValueError("binary target matrix shape must match readout/targets.")
    rank = max(1, min(int(rank), int(readout.shape[1])))

    with torch.inference_mode(False):
        device = readout.device
        compute_dtype = _safe_float_dtype(readout.dtype)
        r = readout.detach().clone().to(device=device, dtype=compute_dtype)
        u = target_vectors.detach().clone().to(device=device, dtype=compute_dtype)
        y = binary_targets.detach().clone().to(device=device, dtype=compute_dtype).clamp(0.0, 1.0)
        n, d = int(r.shape[0]), int(r.shape[1])
        pos_counts = y.sum(dim=0)
        valid_mask = (pos_counts > 0) & (pos_counts < float(n))
        pos_weight = compute_binary_target_matrix_pos_weight(y, device=device, dtype=compute_dtype)
        ws_min_t = torch.tensor(float(positive_scale_min), device=device, dtype=compute_dtype).clamp_min(1e-12)

        raw_scale = torch.nn.Parameter(
            _inverse_softplus_positive(torch.ones((d,), device=device, dtype=compute_dtype) - ws_min_t)
            if enforce_positive_scale
            else torch.ones((d,), device=device, dtype=compute_dtype)
        )
        offset = torch.nn.Parameter(torch.zeros((d,), device=device, dtype=compute_dtype))
        low_left = torch.nn.Parameter(torch.zeros((d, rank), device=device, dtype=compute_dtype))
        low_right = torch.nn.Parameter(torch.zeros((d, rank), device=device, dtype=compute_dtype))
        torch.nn.init.normal_(low_left, mean=0.0, std=1e-3)
        torch.nn.init.normal_(low_right, mean=0.0, std=1e-3)
        optimizer = torch.optim.LBFGS(
            [raw_scale, offset, low_left, low_right],
            lr=float(lr),
            max_iter=max(1, int(num_iterations)),
            line_search_fn="strong_wolfe",
        )

        def _scale() -> torch.Tensor:
            return F.softplus(raw_scale) + ws_min_t if enforce_positive_scale else raw_scale

        def _losses(scale_value: torch.Tensor) -> torch.Tensor:
            low_rank_delta = (r.matmul(low_right)).matmul(low_left.t())
            transformed = r * scale_value.unsqueeze(0) + low_rank_delta + offset.unsqueeze(0)
            eta = transformed.matmul(u.t())
            weight = torch.where(y > 0.5, pos_weight.unsqueeze(0), torch.ones_like(y))
            loss = torch.where(y > 0.5, F.softplus(-eta), F.softplus(eta)) * weight
            return loss.sum(dim=0) / max(float(n), 1.0)

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            scale_value = _scale()
            losses = _losses(scale_value)
            valid_losses = losses[valid_mask]
            total = valid_losses.mean() if valid_losses.numel() else losses.sum() * 0.0
            if float(identity_l2) > 0.0:
                total = total + 0.5 * float(identity_l2) * (scale_value - 1.0).square().mean()
            if float(offset_l2) > 0.0:
                total = total + 0.5 * float(offset_l2) * offset.square().mean()
            if float(low_rank_l2) > 0.0:
                total = total + 0.5 * float(low_rank_l2) * (low_left.square().mean() + low_right.square().mean())
            total.backward()
            return total

        optimizer.step(closure)
        with torch.no_grad():
            optimized_scale = _scale().detach().clone()
            optimized_offset = offset.detach().clone()
            optimized_loss = _losses(optimized_scale).detach().clone()
            bias_only_loss = _shared_readout_loss_grad_torch(
                r,
                u,
                y,
                torch.ones_like(optimized_scale),
                optimized_offset,
                pos_weight,
                valid_mask,
            )[0].detach().clone()
            low_left_value = low_left.detach().clone()
            low_right_value = low_right.detach().clone()

    bad_parts = []
    for name, tensor in (
        ("readout_scale", optimized_scale),
        ("readout_offset", optimized_offset),
        ("low_rank_left", low_left_value),
        ("low_rank_right", low_right_value),
        ("optimized_loss", optimized_loss[valid_mask]),
    ):
        if not bool(torch.isfinite(tensor).all().item()):
            bad_parts.append(name)
    if bad_parts:
        context = f" context={diagnostic_context!r}" if diagnostic_context else ""
        raise RuntimeError(f"Non-finite low-rank shared readout fit result for {bad_parts}.{context}")
    return ReadoutSpaceAffineFitResult(
        readout_scale=optimized_scale,
        readout_offset=optimized_offset,
        target_vectors=u.detach().clone(),
        pos_weight=pos_weight,
        bias_only_loss=bias_only_loss,
        optimized_loss=optimized_loss,
        calibration_family="c2_shared_diagonal_affine_low_rank",
        calibration_backend="shared_readout_low_rank_lbfgs_diagnostic",
        low_rank_left=low_left_value,
        low_rank_right=low_right_value,
        low_rank_rank=rank,
    )


def _contextual_binary_grad_hess_torch(
    target_logits_adjusted: torch.Tensor,
    other_lse: torch.Tensor,
    y_binary: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    compute_dtype = _safe_float_dtype(target_logits_adjusted.dtype)
    x = target_logits_adjusted.to(dtype=compute_dtype).view(-1)
    c = other_lse.to(device=x.device, dtype=compute_dtype).view(-1)
    y = y_binary.to(device=x.device, dtype=compute_dtype).view(-1)
    denom = torch.tensor(float(y.numel()), device=x.device, dtype=compute_dtype).clamp_min(1.0)
    logits = x * w + b - c
    p = torch.sigmoid(logits)
    s = p * (1.0 - p)
    losses = F.binary_cross_entropy_with_logits(logits, y, reduction="mean")
    delta = (p - y) / denom
    g_w = delta.mul(x).sum()
    g_b = delta.sum()
    s_weight = s / denom
    H_ww = s_weight.mul(x.square()).sum()
    H_wb = s_weight.mul(x).sum()
    H_bb = s_weight.sum()
    return losses, g_w, g_b, H_ww, H_wb, H_bb, denom


def _optimize_wb_contextual_binary_single(
    target_logits_adjusted: torch.Tensor,
    other_lse: torch.Tensor,
    y_binary: torch.Tensor,
    *,
    init_scale: torch.Tensor,
    init_offset: torch.Tensor,
    optimize_scale: bool,
    iters: int = 50,
    damping_init: float = 1e-4,
    step_tol: float = 1e-9,
    patience: int = 10,
    conv_tol: float = 1e-8,
    min_iters: int = 3,
    early_stop_grad_tol: float = 1e-7,
    prior_lambda_w: Optional[torch.Tensor] = None,
    prior_lambda_b: Optional[torch.Tensor] = None,
    prior_center_w: float = 1.0,
    prior_center_b: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # This optimized path is invoked from the legacy intermix runtime while it is
    # often inside a broader torch.inference_mode() region. LBFGS requires a real
    # autograd-tracked closure, so we temporarily escape inference mode and clone
    # fresh normal tensors for the single-slot contextual optimization.
    with torch.inference_mode(False):
        device = target_logits_adjusted.device
        dtype = _safe_float_dtype(target_logits_adjusted.dtype)
        lambda_w = (
            torch.zeros((), device=device, dtype=dtype)
            if prior_lambda_w is None
            else prior_lambda_w.detach().clone().to(device=device, dtype=dtype).view(())
        )
        lambda_b = (
            torch.zeros((), device=device, dtype=dtype)
            if prior_lambda_b is None
            else prior_lambda_b.detach().clone().to(device=device, dtype=dtype).view(())
        )
        prior_center_w_t = torch.as_tensor(float(prior_center_w), device=device, dtype=dtype)
        prior_center_b_t = torch.as_tensor(float(prior_center_b), device=device, dtype=dtype)
        x = target_logits_adjusted.detach().clone().to(device=device, dtype=dtype).view(-1)
        c = other_lse.detach().clone().to(device=device, dtype=dtype).view(-1)
        y = y_binary.detach().clone().to(device=device, dtype=dtype).view(-1)

        init_scale_local = init_scale.detach().clone().to(device=device, dtype=dtype).view(())
        init_offset_local = init_offset.detach().clone().to(device=device, dtype=dtype).view(())
        if optimize_scale:
            w_param = torch.nn.Parameter(init_scale_local)
        else:
            w_fixed = init_scale_local
        b_param = torch.nn.Parameter(init_offset_local)
        params = [b_param] + ([w_param] if optimize_scale else [])

        optimizer = torch.optim.LBFGS(
            params,
            max_iter=max(1, int(iters)),
            tolerance_grad=float(early_stop_grad_tol),
            tolerance_change=float(step_tol),
            line_search_fn="strong_wolfe",
        )

        def _objective_value() -> torch.Tensor:
            w_curr = w_param if optimize_scale else w_fixed
            contextual_logit = x * w_curr + b_param - c
            loss_curr = F.binary_cross_entropy_with_logits(
                contextual_logit,
                y,
                reduction="mean",
            )
            if optimize_scale:
                loss_curr = loss_curr + 0.5 * lambda_w * (w_curr - prior_center_w_t).square()
            loss_curr = loss_curr + 0.5 * lambda_b * (b_param - prior_center_b_t).square()
            return loss_curr

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss_curr = _objective_value()
            loss_curr.backward()
            return loss_curr

        optimizer.step(closure)
        with torch.no_grad():
            final_w = (w_param if optimize_scale else w_fixed).detach().clone()
            final_b = b_param.detach().clone()
            final_loss = _objective_value().detach().clone()

        return final_w, final_b, final_loss


def fit_single_binarized_softmax_affine_calibration(
    target_logits_adjusted: torch.Tensor,
    other_lse: torch.Tensor,
    labels: torch.Tensor,
    *,
    positive_class_id: int,
    init_classifier_offset: Optional[torch.Tensor] = None,
    init_classifier_scale: Optional[torch.Tensor] = None,
    optimize_classifier_scale_joint: bool = True,
    num_iterations: int = 80,
    damping_init: float = 1e-4,
    use_triton_grad_hess: bool = True,
    min_triton_slots: Optional[int] = None,
    use_map_prior: bool = False,
    map_prior_c: float = 0.1,
    map_prior_eps: float = 1e-6,
    map_prior_mode: str = "off",
    map_prior_grid_size: int = 7,
    map_prior_grid_span: float = 2.0,
    map_prior_grid_max_iters: int = 200,
    map_prior_hessian_ridge: float = 1e-8,
    affine_calibration_mode: str = "hessian_minimization",
    diagnostic_context: Optional[str] = None,
) -> ClasswiseAffineFitResult:
    del use_triton_grad_hess, min_triton_slots

    if target_logits_adjusted.ndim != 1:
        raise ValueError("target_logits_adjusted must have shape [N].")
    if other_lse.ndim != 1:
        raise ValueError("other_lse must have shape [N].")
    if int(target_logits_adjusted.numel()) != int(other_lse.numel()):
        raise ValueError("target_logits_adjusted and other_lse must have the same length.")

    device = target_logits_adjusted.device
    compute_dtype = _safe_float_dtype(target_logits_adjusted.dtype)
    x = target_logits_adjusted.to(device=device, dtype=compute_dtype).view(-1)
    c = other_lse.to(device=device, dtype=compute_dtype).view(-1)
    labels_long = labels.to(device=device, dtype=torch.long).view(-1)
    if int(labels_long.numel()) != int(x.numel()):
        raise ValueError("labels must have the same length as target_logits_adjusted.")

    if not bool(torch.isfinite(x).all().item()):
        raise RuntimeError(
            "Non-finite contextual softmax affine fit target logits detected before optimization; "
            f"context={diagnostic_context!r}."
        )
    if not bool(torch.isfinite(c).all().item()):
        raise RuntimeError(
            "Non-finite contextual softmax affine fit other_lse detected before optimization; "
            f"context={diagnostic_context!r}."
        )

    y_binary = labels_long.eq(int(positive_class_id)).to(dtype=compute_dtype)
    num_positive = y_binary.sum()
    num_negative = torch.tensor(float(y_binary.numel()), device=device, dtype=compute_dtype) - num_positive
    if bool((num_positive <= 0).item()) or bool((num_negative <= 0).item()):
        pos_weight = torch.tensor(1.0, device=device, dtype=compute_dtype)
    else:
        pos_weight = (num_negative / num_positive).to(dtype=compute_dtype)

    if init_classifier_scale is None:
        init_scale = torch.tensor(1.0, device=device, dtype=compute_dtype)
    else:
        init_scale = init_classifier_scale.to(device=device, dtype=compute_dtype).view(())
    if init_classifier_offset is None:
        init_offset = torch.tensor(0.0, device=device, dtype=compute_dtype)
    else:
        init_offset = init_classifier_offset.to(device=device, dtype=compute_dtype).view(())

    resolved_affine_mode = str(affine_calibration_mode).strip().lower() or "hessian_minimization"
    if resolved_affine_mode == "lobo_hierarchical_empirical_bayes":
        raise NotImplementedError(
            "LOBO hierarchical empirical Bayes is not implemented in the optimized contextual softmax backend. "
            "Caller should fall back to optimize_affine_transformation()."
        )

    joint_scale_identifiable = bool(
        _joint_scale_identifiability_mask(x.view(1, -1), diagnostic_context=diagnostic_context)[0].item()
    )
    bias_init_scale = init_scale if joint_scale_identifiable else init_scale.detach().clone()
    bias_optimize_scale = False

    n_eff = torch.tensor(float(y_binary.numel()), device=device, dtype=compute_dtype).clamp_min(1.0)
    resolved_map_prior_mode = str(map_prior_mode).strip().lower() if use_map_prior else "off"
    x_std = x.std(unbiased=True).clamp_min(float(map_prior_eps))
    sigma_w = torch.as_tensor(float(map_prior_c), device=device, dtype=compute_dtype) / x_std
    sigma_b = torch.as_tensor(float(map_prior_c), device=device, dtype=compute_dtype)
    fixed_lambda_w = (
        1.0 / (n_eff * sigma_w.square()).clamp_min(float(map_prior_eps))
        if joint_scale_identifiable
        else torch.zeros((), device=device, dtype=compute_dtype)
    )
    fixed_lambda_b = 1.0 / (n_eff * sigma_b.square()).clamp_min(float(map_prior_eps))

    def _compute_prior_lambdas(*, optimize_scale_stage: bool):
        if not use_map_prior:
            return (
                torch.zeros_like(fixed_lambda_w),
                torch.zeros_like(fixed_lambda_b),
            )
        if resolved_map_prior_mode == "empirical_bayes":
            ridge = float(map_prior_hessian_ridge)
            _, _, _, base_H_ww, _, base_H_bb, _ = _contextual_binary_grad_hess_torch(
                x,
                c,
                y_binary,
                torch.tensor(1.0, device=device, dtype=compute_dtype),
                torch.tensor(0.0, device=device, dtype=compute_dtype),
            )
            base_lambda_w = (
                base_H_ww.abs().clamp_min(ridge)
                if optimize_scale_stage
                else torch.zeros_like(base_H_bb)
            )
            base_lambda_b = base_H_bb.abs().clamp_min(ridge)
            exp_offsets = torch.linspace(
                -float(map_prior_grid_span),
                float(map_prior_grid_span),
                steps=max(1, int(map_prior_grid_size)),
                device=device,
                dtype=compute_dtype,
            )
            grid_scale = torch.pow(torch.tensor(10.0, device=device, dtype=compute_dtype), exp_offsets)
            best_log_evid = None
            best_lambda_w = fixed_lambda_w if optimize_scale_stage else torch.zeros_like(fixed_lambda_w)
            best_lambda_b = fixed_lambda_b.clone()
            grid_iters = max(1, min(int(num_iterations), int(map_prior_grid_max_iters)))
            for scale_factor in grid_scale:
                cand_lambda_w = (
                    base_lambda_w * scale_factor
                    if optimize_scale_stage
                    else torch.zeros_like(base_lambda_b)
                )
                for bias_scale_factor in grid_scale:
                    cand_lambda_b = base_lambda_b * bias_scale_factor
                    cand_w, cand_b, cand_loss = _optimize_wb_contextual_binary_single(
                        x,
                        c,
                        y_binary,
                        init_scale=bias_init_scale if not optimize_scale_stage else init_scale,
                        init_offset=init_offset,
                        optimize_scale=optimize_scale_stage,
                        iters=grid_iters,
                        damping_init=damping_init,
                        prior_lambda_w=cand_lambda_w,
                        prior_lambda_b=cand_lambda_b,
                    )
                    _, _, _, H_ww_hat_raw, H_wb_hat_raw, H_bb_hat_raw, _ = _contextual_binary_grad_hess_torch(
                        x,
                        c,
                        y_binary,
                        cand_w,
                        cand_b,
                    )
                    H_ww_hat = H_ww_hat_raw + (cand_lambda_w if optimize_scale_stage else torch.zeros_like(H_ww_hat_raw))
                    H_wb_hat = H_wb_hat_raw
                    H_bb_hat = H_bb_hat_raw + cand_lambda_b
                    log_evid = _compute_log_evidence_classwise(
                        obj_hat=cand_loss.view(1),
                        H_ww=H_ww_hat.view(1),
                        H_wb=H_wb_hat.view(1),
                        H_bb=H_bb_hat.view(1),
                        lambda_w=(cand_lambda_w if optimize_scale_stage else torch.zeros_like(cand_lambda_b)).view(1),
                        lambda_b=cand_lambda_b.view(1),
                        n_eff=n_eff.view(1),
                        optimize_scale=optimize_scale_stage,
                        ridge=ridge,
                    )
                    if best_log_evid is None or float(log_evid.item()) > best_log_evid:
                        best_log_evid = float(log_evid.item())
                        best_lambda_w = cand_lambda_w
                        best_lambda_b = cand_lambda_b
            return best_lambda_w, best_lambda_b
        return (
            fixed_lambda_w if optimize_scale_stage else torch.zeros_like(fixed_lambda_w),
            fixed_lambda_b,
        )

    bias_lambda_w, bias_lambda_b = _compute_prior_lambdas(optimize_scale_stage=False)
    scale_bias, offset_bias, loss_bias = _optimize_wb_contextual_binary_single(
        x,
        c,
        y_binary,
        init_scale=bias_init_scale,
        init_offset=init_offset,
        optimize_scale=bias_optimize_scale,
        iters=num_iterations,
        damping_init=damping_init,
        prior_lambda_w=bias_lambda_w,
        prior_lambda_b=bias_lambda_b,
    )

    if optimize_classifier_scale_joint and joint_scale_identifiable:
        joint_lambda_w, joint_lambda_b = _compute_prior_lambdas(optimize_scale_stage=True)
        scale_joint, offset_joint, loss_joint = _optimize_wb_contextual_binary_single(
            x,
            c,
            y_binary,
            init_scale=scale_bias,
            init_offset=offset_bias,
            optimize_scale=True,
            iters=num_iterations,
            damping_init=damping_init,
            prior_lambda_w=joint_lambda_w,
            prior_lambda_b=joint_lambda_b,
        )
    else:
        scale_joint = scale_bias.clone()
        offset_joint = offset_bias.clone()
        loss_joint = loss_bias.clone()

    for name, tensor in (
        ("classifier_scale", scale_bias),
        ("classifier_offset", offset_bias),
        ("classifier_scale_optimized", scale_joint),
        ("classifier_offset_optimized", offset_joint),
        ("bias_only_loss", loss_bias),
        ("optimized_loss", loss_joint),
    ):
        if not bool(torch.isfinite(tensor).item()):
            raise RuntimeError(
                "Non-finite contextual softmax affine fit result detected after optimization; "
                f"field={name}, context={diagnostic_context!r}."
            )

    return ClasswiseAffineFitResult(
        classifier_offset=offset_bias.view(1),
        classifier_scale=scale_bias.view(1),
        classifier_offset_optimized=offset_joint.view(1),
        classifier_scale_optimized=scale_joint.view(1),
        pos_weight=pos_weight.view(1),
        bias_only_loss=loss_bias.view(1),
        optimized_loss=loss_joint.view(1),
    )


def fit_classwise_binary_affine_calibration(
    logits_full: torch.Tensor,
    labels: torch.Tensor,
    *,
    init_classifier_offset: Optional[torch.Tensor] = None,
    init_classifier_scale: Optional[torch.Tensor] = None,
    optimize_classifier_scale_joint: bool = True,
    num_iterations: int = 80,
    damping_init: float = 1e-4,
    class_chunk_size: int = 128,
    use_triton_grad_hess: bool = True,
    min_triton_slots: Optional[int] = None,
    compact_active_slots: bool = True,
    diagnostic_context: Optional[str] = None,
) -> ClasswiseAffineFitResult:
    if logits_full.ndim != 2:
        raise ValueError("logits_full must have shape [N, C].")
    class_ids = torch.arange(int(logits_full.shape[1]), device=logits_full.device, dtype=torch.long)
    return fit_selected_binary_affine_calibration(
        logits_full,
        labels,
        selected_class_ids=class_ids,
        init_classifier_offset=init_classifier_offset,
        init_classifier_scale=init_classifier_scale,
        optimize_classifier_scale_joint=optimize_classifier_scale_joint,
        num_iterations=num_iterations,
        damping_init=damping_init,
        class_chunk_size=class_chunk_size,
        use_triton_grad_hess=use_triton_grad_hess,
        min_triton_slots=min_triton_slots,
        compact_active_slots=compact_active_slots,
        diagnostic_context=diagnostic_context,
    )


def fit_binary_target_matrix_affine_calibration(
    logits_selected: torch.Tensor,
    binary_targets: torch.Tensor,
    *,
    init_classifier_offset: Optional[torch.Tensor] = None,
    init_classifier_scale: Optional[torch.Tensor] = None,
    optimize_classifier_scale_joint: bool = True,
    num_iterations: int = 80,
    damping_init: float = 1e-4,
    class_chunk_size: int = 128,
    diagnostic_context: Optional[str] = None,
) -> ClasswiseAffineFitResult:
    """Fit independent affine calibrations for arbitrary binary target columns."""

    if logits_selected.ndim != 2:
        raise ValueError("logits_selected must have shape [N, K].")
    if binary_targets.shape != logits_selected.shape:
        raise ValueError("binary_targets must have the same shape as logits_selected.")
    device = logits_selected.device
    compute_dtype = _safe_float_dtype(logits_selected.dtype)
    num_samples, num_slots = int(logits_selected.shape[0]), int(logits_selected.shape[1])
    if num_samples <= 0 or num_slots <= 0:
        raise ValueError("logits_selected must have non-empty [N, K] dimensions.")
    targets = binary_targets.to(device=device, dtype=compute_dtype).clamp(0.0, 1.0)
    pos_counts = targets.sum(dim=0)
    valid_mask = (pos_counts > 0) & (pos_counts < float(num_samples))
    pos_weight = compute_binary_target_matrix_pos_weight(
        targets,
        device=device,
        dtype=compute_dtype,
    )

    if init_classifier_scale is None:
        init_classifier_scale = torch.ones((num_slots,), device=device, dtype=compute_dtype)
    else:
        init_classifier_scale = init_classifier_scale.to(device=device, dtype=compute_dtype).view(-1)
    if init_classifier_offset is None:
        init_classifier_offset = torch.zeros((num_slots,), device=device, dtype=compute_dtype)
    else:
        init_classifier_offset = init_classifier_offset.to(device=device, dtype=compute_dtype).view(-1)

    scale_bias = torch.ones_like(init_classifier_scale)
    offset_bias = init_classifier_offset.clone()
    scale_joint = init_classifier_scale.clone()
    offset_joint = init_classifier_offset.clone()
    loss_bias = torch.full((num_slots,), float("inf"), device=device, dtype=compute_dtype)
    loss_joint = torch.full((num_slots,), float("inf"), device=device, dtype=compute_dtype)
    denom = torch.tensor(float(num_samples), device=device, dtype=compute_dtype).clamp_min(1.0)
    damping = torch.tensor(float(damping_init), device=device, dtype=compute_dtype).clamp_min(1e-12)

    def _loss_for_chunk(
        x: torch.Tensor,
        y: torch.Tensor,
        pw: torch.Tensor,
        scale: torch.Tensor,
        offset: torch.Tensor,
    ) -> torch.Tensor:
        eta = x * scale.unsqueeze(0) + offset.unsqueeze(0)
        weight = torch.where(y > 0.5, pw.unsqueeze(0), torch.ones_like(y))
        return (torch.where(y > 0.5, F.softplus(-eta), F.softplus(eta)) * weight).sum(dim=0) / denom

    for start in range(0, num_slots, max(1, int(class_chunk_size))):
        end = min(start + int(class_chunk_size), num_slots)
        valid = valid_mask[start:end]
        if not bool(valid.any().item()):
            continue
        x = logits_selected[:, start:end].to(dtype=compute_dtype)[:, valid]
        y = targets[:, start:end][:, valid]
        pw = pos_weight[start:end][valid]

        b = offset_bias[start:end][valid].clone()
        for _ in range(max(1, int(num_iterations))):
            eta = x + b.unsqueeze(0)
            p = torch.sigmoid(eta)
            weight = torch.where(y > 0.5, pw.unsqueeze(0), torch.ones_like(y))
            diff = (p - y) * weight / denom
            curv = p * (1.0 - p) * weight / denom
            grad = diff.sum(dim=0)
            hess = curv.sum(dim=0).clamp_min(damping)
            b = b - grad / hess
        local_offsets = offset_bias[start:end].clone()
        local_offsets[valid] = b
        offset_bias[start:end] = local_offsets

        if optimize_classifier_scale_joint:
            s = scale_joint[start:end][valid].clone()
            b_joint = offset_joint[start:end][valid].clone()
            for _ in range(max(1, int(num_iterations))):
                eta = x * s.unsqueeze(0) + b_joint.unsqueeze(0)
                p = torch.sigmoid(eta)
                weight = torch.where(y > 0.5, pw.unsqueeze(0), torch.ones_like(y))
                diff = (p - y) * weight / denom
                curv = p * (1.0 - p) * weight / denom
                g_s = (diff * x).sum(dim=0)
                g_b = diff.sum(dim=0)
                h_ss = (curv * x.square()).sum(dim=0) + damping
                h_sb = (curv * x).sum(dim=0)
                h_bb = curv.sum(dim=0) + damping
                det = (h_ss * h_bb - h_sb.square()).clamp_min(damping)
                step_s = (h_bb * g_s - h_sb * g_b) / det
                step_b = (-h_sb * g_s + h_ss * g_b) / det
                s = s - step_s
                b_joint = b_joint - step_b
            local_scales = scale_joint[start:end].clone()
            local_offsets_joint = offset_joint[start:end].clone()
            local_scales[valid] = s
            local_offsets_joint[valid] = b_joint
            scale_joint[start:end] = local_scales
            offset_joint[start:end] = local_offsets_joint
        else:
            scale_joint[start:end] = scale_bias[start:end]
            offset_joint[start:end] = offset_bias[start:end]

        loss_bias_chunk = loss_bias[start:end].clone()
        loss_joint_chunk = loss_joint[start:end].clone()
        loss_bias_chunk[valid] = _loss_for_chunk(x, y, pw, torch.ones_like(b), b)
        loss_joint_chunk[valid] = _loss_for_chunk(
            x,
            y,
            pw,
            scale_joint[start:end][valid],
            offset_joint[start:end][valid],
        )
        loss_bias[start:end] = loss_bias_chunk
        loss_joint[start:end] = loss_joint_chunk

    _raise_on_nonfinite_fit_result(
        classifier_scale=scale_bias,
        classifier_offset=offset_bias,
        classifier_scale_optimized=scale_joint,
        classifier_offset_optimized=offset_joint,
        bias_only_loss=loss_bias,
        optimized_loss=loss_joint,
        valid_mask=valid_mask,
    )

    return ClasswiseAffineFitResult(
        classifier_offset=offset_bias,
        classifier_scale=scale_bias,
        classifier_offset_optimized=offset_joint,
        classifier_scale_optimized=scale_joint,
        pos_weight=pos_weight,
        bias_only_loss=loss_bias,
        optimized_loss=loss_joint,
    )


def evaluate_binary_target_matrix_losses(
    logits_selected: torch.Tensor,
    binary_targets: torch.Tensor,
    *,
    classifier_scale: torch.Tensor,
    classifier_offset: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
    class_chunk_size: int = 128,
) -> torch.Tensor:
    if logits_selected.ndim != 2:
        raise ValueError("logits_selected must have shape [N, K].")
    if binary_targets.shape != logits_selected.shape:
        raise ValueError("binary_targets must have the same shape as logits_selected.")
    device = logits_selected.device
    compute_dtype = _safe_float_dtype(logits_selected.dtype)
    targets = binary_targets.to(device=device, dtype=compute_dtype)
    num_slots = int(logits_selected.shape[1])
    if pos_weight is None:
        pos_weight = compute_binary_target_matrix_pos_weight(
            targets,
            device=device,
            dtype=compute_dtype,
        )
    else:
        pos_weight = pos_weight.to(device=device, dtype=compute_dtype).view(-1)
    losses = torch.empty((num_slots,), device=device, dtype=compute_dtype)
    denom = torch.tensor(float(logits_selected.shape[0]), device=device, dtype=compute_dtype).clamp_min(1.0)
    for start in range(0, num_slots, max(1, int(class_chunk_size))):
        end = min(start + int(class_chunk_size), num_slots)
        x = logits_selected[:, start:end].to(dtype=compute_dtype)
        y = targets[:, start:end]
        eta = (
            classifier_scale[start:end].to(device=device, dtype=compute_dtype).unsqueeze(0) * x
            + classifier_offset[start:end].to(device=device, dtype=compute_dtype).unsqueeze(0)
        )
        weight = torch.where(y > 0.5, pos_weight[start:end].unsqueeze(0), torch.ones_like(y))
        losses[start:end] = (
            torch.where(y > 0.5, F.softplus(-eta), F.softplus(eta)) * weight
        ).sum(dim=0) / denom
    return losses


def evaluate_selected_binary_losses(
    logits_selected: torch.Tensor,
    labels: torch.Tensor,
    *,
    selected_class_ids: torch.Tensor,
    classifier_scale: torch.Tensor,
    classifier_offset: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
    class_chunk_size: int = 128,
) -> torch.Tensor:
    if logits_selected.ndim != 2:
        raise ValueError("logits_selected must have shape [N, K].")
    device = logits_selected.device
    compute_dtype = _safe_float_dtype(logits_selected.dtype)
    labels_long = labels.to(device=device, dtype=torch.long)
    class_ids = selected_class_ids.to(device=device, dtype=torch.long).view(-1)
    num_slots = int(logits_selected.shape[1])
    if int(class_ids.numel()) != num_slots:
        raise ValueError("selected_class_ids must have shape [K].")
    if pos_weight is None:
        pos_weight = compute_selected_pos_weight(
            labels_long,
            selected_class_ids=class_ids,
            device=device,
            dtype=compute_dtype,
        )
    else:
        pos_weight = pos_weight.to(device=device, dtype=compute_dtype).view(-1)

    xs_class_major = logits_selected.transpose(0, 1).contiguous()
    losses = torch.empty((num_slots,), device=device, dtype=compute_dtype)
    labels_expanded = labels_long.unsqueeze(0)
    denom = torch.tensor(float(labels_long.numel()), device=device, dtype=compute_dtype).clamp_min(1.0)

    class_ids_expanded = class_ids.unsqueeze(1)
    for start in range(0, num_slots, max(1, int(class_chunk_size))):
        end = min(start + int(class_chunk_size), num_slots)
        y_chunk = labels_expanded.eq(class_ids_expanded[start:end])
        x_chunk = xs_class_major[start:end].to(dtype=compute_dtype)
        logits_chunk = (
            classifier_scale[start:end].to(device=device, dtype=compute_dtype).unsqueeze(1) * x_chunk
            + classifier_offset[start:end].to(device=device, dtype=compute_dtype).unsqueeze(1)
        )
        pos_chunk = pos_weight[start:end].unsqueeze(1)
        losses[start:end] = torch.where(
            y_chunk,
            pos_chunk * F.softplus(-logits_chunk),
            F.softplus(logits_chunk),
        ).sum(dim=1) / denom

    return losses


def evaluate_classwise_binary_losses(
    logits_full: torch.Tensor,
    labels: torch.Tensor,
    *,
    classifier_scale: torch.Tensor,
    classifier_offset: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
    class_chunk_size: int = 128,
) -> torch.Tensor:
    if logits_full.ndim != 2:
        raise ValueError("logits_full must have shape [N, C].")
    class_ids = torch.arange(int(logits_full.shape[1]), device=logits_full.device, dtype=torch.long)
    return evaluate_selected_binary_losses(
        logits_full,
        labels,
        selected_class_ids=class_ids,
        classifier_scale=classifier_scale,
        classifier_offset=classifier_offset,
        pos_weight=pos_weight,
        class_chunk_size=class_chunk_size,
    )


def functional_selectivity_contrast_from_losses(
    losses: torch.Tensor,
    target_class: int,
    *,
    pruning_frontier_mode: str,
    unpruned_losses: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    if losses.ndim != 1:
        raise ValueError("losses must be rank-1 [C].")
    losses_f = losses.to(dtype=torch.float32)
    if unpruned_losses is None:
        baseline_losses = losses_f
    else:
        baseline_losses = unpruned_losses.to(device=losses.device, dtype=torch.float32)
        if baseline_losses.shape != losses_f.shape:
            raise ValueError("unpruned_losses must share shape [C].")

    target_loss = losses_f[int(target_class)]
    loss_deltas = losses_f - baseline_losses
    target_loss_delta = loss_deltas[int(target_class)]
    if int(losses_f.numel()) > 1:
        off_target_mask = torch.ones(int(losses_f.numel()), device=losses.device, dtype=torch.bool)
        off_target_mask[int(target_class)] = False
        off_target_losses = losses_f[off_target_mask]
        off_target_loss_deltas = loss_deltas[off_target_mask]
        off_target_loss = off_target_losses.mean()
        off_target_loss_delta = off_target_loss_deltas.mean()
    else:
        off_target_loss = torch.zeros_like(target_loss)
        off_target_loss_delta = torch.zeros_like(target_loss)
    loss_contrast = target_loss_delta - off_target_loss_delta
    return {
        "loss_contrast": loss_contrast,
        "target_loss": target_loss,
        "off_target_loss": off_target_loss,
        "target_loss_delta": target_loss_delta,
        "off_target_loss_delta": off_target_loss_delta,
        # Backward-compatible alias for older report plumbing.
        "off_target_harm": off_target_loss_delta,
    }


def normalize_classwise_damage(
    losses: torch.Tensor,
    *,
    unpruned_losses: torch.Tensor,
    fully_pruned_losses: torch.Tensor,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    if losses.ndim != 1:
        raise ValueError("losses must be rank-1 [C].")
    losses_f = losses.to(dtype=torch.float32)
    unpruned_f = unpruned_losses.to(device=losses.device, dtype=torch.float32)
    fully_pruned_f = fully_pruned_losses.to(device=losses.device, dtype=torch.float32)
    if losses_f.shape != unpruned_f.shape or losses_f.shape != fully_pruned_f.shape:
        raise ValueError("loss tensors must share shape [C].")

    full_delta = fully_pruned_f - unpruned_f
    loss_delta = losses_f - unpruned_f
    active = full_delta > float(eps)
    damages = torch.zeros_like(losses_f)
    damages[active] = loss_delta[active].clamp_min(0.0) / full_delta[active].clamp_min(float(eps))
    return {
        "damage": damages,
        "loss_delta": loss_delta,
        "full_delta": full_delta,
        "active_mask": active,
    }


def functional_selectivity_contrast_from_damages(
    damages: torch.Tensor,
    target_class: int,
    *,
    pruning_frontier_mode: str,
) -> Dict[str, torch.Tensor]:
    if damages.ndim != 1:
        raise ValueError("damages must be rank-1 [C].")
    damages_f = damages.to(dtype=torch.float32)
    target_damage = damages_f[int(target_class)]
    if int(damages_f.numel()) > 1:
        off_target_mask = torch.ones(int(damages_f.numel()), device=damages.device, dtype=torch.bool)
        off_target_mask[int(target_class)] = False
        off_target_damage_values = damages_f[off_target_mask]
        off_target_damage = off_target_damage_values.mean()
    else:
        off_target_damage = torch.zeros_like(target_damage)
    damage_contrast = target_damage - off_target_damage
    return {
        "damage_contrast": damage_contrast,
        "target_damage": target_damage,
        "off_target_damage": off_target_damage,
    }
