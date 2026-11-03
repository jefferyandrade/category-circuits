from __future__ import annotations

import math
import os
import random
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:  # pragma: no cover - optional dependency
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; using Triton decorator stubs.")
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
    _HAS_TRITON = True
except Exception:  # pragma: no cover - fallback
    _HAS_TRITON = False

    class _TritonStub:  # minimal decorator shim
        @staticmethod
        def jit(fn=None, **_kwargs):
            def _wrap(f):
                return f

            return _wrap(fn) if fn is not None else _wrap

    class _TLStub:
        @staticmethod
        def constexpr(x=None):  # pragma: no cover - metadata-only stub
            return x

    triton = _TritonStub()  # type: ignore
    tl = _TLStub()  # type: ignore

from packages.salience.core.frontier_utils import (
    COMPUTE_LOSS_DELTA,
    COMPUTE_BITPACK,
    DEBUG_SCORING,
    VALIDATE_FRONTIER,
    debug_print,
)
from packages.salience.utils import pack_boolean_rows, vectorised_bitpack

def _prepare_softmax_stats(full_acts: torch.Tensor,
                           labels: torch.Tensor,
                           target_unit: int,
                           original_loss: torch.Tensor | None = None):
    """
    Returns tensors of shape [B] needed by the fused kernel:

      z_t    : target logit
      z_y    : correct-class logit
      m_all  : max_j z_j   (for numerical stability)
      Sx     : sum_{j != t} exp(z_j - m_all)   (scaled by m_all)
      L0     : original loss per sample
      is_ty  : 1 if label==t else 0  (int32)
    """
    assert full_acts.is_cuda and labels.is_cuda
    B, M = full_acts.shape
    logits = full_acts.to(torch.float32)  # keep math in fp32

    m_all = logits.max(dim=1).values                            # [B]
    lse_all = torch.logsumexp(logits, dim=1)                    # [B]
    z_t = logits[:, target_unit]                                # [B]
    # contribution of the target class inside S_all (in the shifted scale)
    S_all = torch.exp(lse_all - m_all)                          # [B]
    A_t = torch.exp(z_t - m_all)                                # [B]
    Sx = S_all - A_t                                            # [B], excludes target

    y = labels.long().view(-1)
    z_y = logits.gather(1, y[:, None]).squeeze(1)               # [B]
    if original_loss is None:
        L0 = lse_all - z_y                                      # CE(logits, y)
    else:
        L0 = original_loss.to(torch.float32)

    is_ty = (y == target_unit).to(torch.int32)                  # [B]

    return z_t, z_y, m_all, Sx, L0, is_ty


# ==========
# Fused O(B·N) persistent-B kernel
# ==========

@triton.jit
def _loss_diff_bn_kernel(
    # pointers
    base_ptr, out_ptr,
    zt_ptr, zy_ptr, m_ptr, Sx_ptr, L0_ptr, isty_ptr,
    # sizes/strides
    B, N,
    sbs_b, sbs_n, sout_b, sout_n,
    # constexpr
    BLOCK_N: tl.constexpr,
    EPS: tl.constexpr,
):
    b = tl.program_id(0)  # one persistent CTA per batch row

    # guard (shouldn't trigger, but cheap)
    if b >= B:
        return

    # cache per-batch scalars in registers (fp32)
    zt   = tl.load(zt_ptr + b).to(tl.float32)
    zy   = tl.load(zy_ptr + b).to(tl.float32)
    m0   = tl.load(m_ptr  + b).to(tl.float32)
    Sx   = tl.load(Sx_ptr + b).to(tl.float32)   # sum_{j!=t} exp(z_j - m0)
    L0   = tl.load(L0_ptr + b).to(tl.float32)
    isty = tl.load(isty_ptr + b)                # int32 0/1

    # sweep N dimension in tiles
    for n0 in range(0, N, BLOCK_N):
        offs = n0 + tl.arange(0, BLOCK_N)
        mask = offs < N

        # vectorized load of base_scores[b, offs] (bf16/fp16/fp32 → fp32)
        s = tl.load(base_ptr + b * sbs_b + offs * sbs_n, mask=mask, other=0).to(tl.float32)

        # New target logit after subtracting score
        zt_abl = zt - s

        # Stable log-sum-exp of { zt_abl } ∪ { others } using common baseline m = max(m0, zt_abl)
        m = tl.maximum(m0, zt_abl)
        # exp of non-target classes (already scaled by m0): Sx * exp(m0 - m)
        se_others = Sx * tl.exp(m0 - m)
        # exp of ablated target: exp(zt_abl - m)
        se_target = tl.exp(zt_abl - m)
        se = se_others + se_target

        new_logsumexp = tl.log(se + EPS) + m
        # correct-class logit (depends on whether label == target)
        corr = tl.where(isty != 0, zt_abl, zy)
        new_loss = new_logsumexp - corr

        diff = L0 - new_loss  # loss difference

        tl.store(out_ptr + b * sout_b + offs * sout_n, diff, mask=mask)


# ==========
# Public wrapper
# ==========

def compute_loss_difference_triton_optimized(
    flattened_base_scores: torch.Tensor,   # [B, N], any float type
    full_acts: torch.Tensor,               # [B, M], logits
    labels: torch.Tensor,                  # [B], int64/int32
    target_unit: int,
    original_loss: torch.Tensor | None = None,  # [B] optional
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Compute per-(b,n) loss difference when subtracting base_scores[b,n] from the
    target logit z_t[b]. O(B·N), single-pass, bandwidth-bound.

    Returns: [B, N] tensor in `out_dtype`.
    """
    if not _HAS_TRITON:
        # CPU / non-triton fallback using torch ops.
        logits = full_acts.to(torch.float32)
        base = flattened_base_scores.to(torch.float32)
        y = labels.to(torch.long)
        if logits.ndim != 2 or base.ndim != 2:
            raise ValueError("Expected logits and base scores to be 2D tensors.")
        z_t = logits[:, target_unit]
        lse_all = torch.logsumexp(logits, dim=1)
        z_y = logits.gather(1, y.view(-1, 1)).squeeze(1)
        L0 = (lse_all - z_y) if original_loss is None else original_loss.to(torch.float32)
        # For each base score, compute new logit and loss.
        zt_abl = z_t[:, None] - base
        # Recompute logsumexp with ablated target using stable two-term logsumexp.
        # others = logsumexp(logits_except_target)
        mask = torch.ones(logits.shape[1], device=logits.device, dtype=torch.bool)
        mask[target_unit] = False
        lse_others = torch.logsumexp(logits[:, mask], dim=1)
        lse_new = torch.logsumexp(torch.stack([zt_abl, lse_others[:, None].expand_as(zt_abl)], dim=-1), dim=-1)
        new_loss = lse_new - torch.where(y == target_unit, zt_abl, z_y[:, None])
        diff = L0[:, None] - new_loss
        return diff.to(dtype=out_dtype, device=flattened_base_scores.device)

    assert flattened_base_scores.is_cuda and full_acts.is_cuda and labels.is_cuda
    assert flattened_base_scores.dim() == 2 and full_acts.dim() == 2
    B, N = flattened_base_scores.shape
    assert full_acts.shape[0] == B, "Batch mismatch between base_scores and full_acts"
    assert 0 <= target_unit < full_acts.shape[1], "target_unit out of range"

    # Best throughput with contiguous N dimension
    base = flattened_base_scores.contiguous()

    # Precompute per-batch stats
    z_t, z_y, m_all, Sx, L0, is_ty = _prepare_softmax_stats(full_acts, labels, target_unit, original_loss)

    # Output buffer (you can choose bf16 for half the write bandwidth)
    out = torch.empty_like(base, dtype=out_dtype)

    # Launch a persistent kernel: one CTA per batch row
    grid = (B,)

    _loss_diff_bn_kernel[grid](
        base_ptr=base,
        out_ptr=out,
        zt_ptr=z_t,
        zy_ptr=z_y,
        m_ptr=m_all,
        Sx_ptr=Sx,
        L0_ptr=L0,
        isty_ptr=is_ty,
        B=B, N=N,
        sbs_b=base.stride(0), sbs_n=base.stride(1),
        sout_b=out.stride(0), sout_n=out.stride(1),
        BLOCK_N=256,
        num_warps=4,
        num_stages=2,
        EPS=1e-9,          # constexpr
    )

    return out


@triton.jit
def dissected_conv_tiled_kernel(
    X_ptr, W_ptr, OUT_ptr,
    B, C_in, H_in, W_in, C_out, KH, KW, H_out, W_out,
    stride_xb, stride_xcin, stride_xh, stride_xw,
    stride_wcout, stride_wcin, stride_wkh, stride_wkw,
    stride_ob, stride_ocout, stride_ocin, stride_oh, stride_ow,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PADDING_H: tl.constexpr, PADDING_W: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
):
    """
    Triton kernel for dissected convolution using a tiled algorithm for high performance.
    """
    pid_channel_combo = tl.program_id(0)
    pid_spatial_block = tl.program_id(1)

    GRID_W = tl.cdiv(W_out, BLOCK_SIZE_W)
    block_h_idx = pid_spatial_block // GRID_W
    block_w_idx = pid_spatial_block % GRID_W
    
    c_in_idx = pid_channel_combo % C_in
    pid_remaining = pid_channel_combo // C_in
    c_out_idx = pid_remaining % C_out
    b_idx = pid_remaining // C_out

    offs_h_out = tl.arange(0, BLOCK_SIZE_H)
    offs_w_out = tl.arange(0, BLOCK_SIZE_W)
    h_out_base = block_h_idx * BLOCK_SIZE_H + offs_h_out
    w_out_base = block_w_idx * BLOCK_SIZE_W + offs_w_out
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    for kh in range(0, KH):
        for kw in range(0, KW):
            h_in_base = h_out_base[:, None] * STRIDE_H - PADDING_H + kh
            w_in_base = w_out_base[None, :] * STRIDE_W - PADDING_W + kw

            load_mask = (h_in_base >= 0) & (h_in_base < H_in) & (w_in_base >= 0) & (w_in_base < W_in)

            x_ptr_base = (X_ptr + b_idx * stride_xb + c_in_idx * stride_xcin +
                          h_in_base * stride_xh + w_in_base * stride_xw)
            x_tile = tl.load(x_ptr_base, mask=load_mask, other=0.0)

            w_ptr = (W_ptr + c_out_idx * stride_wcout + c_in_idx * stride_wcin +
                     kh * stride_wkh + kw * stride_wkw)
            weight_val = tl.load(w_ptr)

            acc += x_tile * weight_val

    store_mask = (h_out_base[:, None] < H_out) & (w_out_base[None, :] < W_out)
    out_ptr_base = (OUT_ptr + b_idx * stride_ob + c_out_idx * stride_ocout + c_in_idx * stride_ocin +
                    h_out_base[:, None] * stride_oh + w_out_base[None, :] * stride_ow)
    tl.store(out_ptr_base, acc, mask=store_mask)





# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
#         triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
#     ],
#     key=['M', 'N', 'K'],
# )


@triton.jit
def fixed_pre_sum_kernel(
    x_ptr, weight_ptr, output_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on, stride_ok,
    BLOCK_SIZE: tl.constexpr
):
    """
    Fixed-block kernel for computing pre-sum output: x.unsqueeze(1) * weight.unsqueeze(0)
    
    X: (M, K)
    W: (N, K)
    Output: (M, N, K)
    """
    # Program IDs
    pid_m = tl.program_id(0)  # batch dimension
    pid_n = tl.program_id(1)  # output features dimension
    
    # Check if we're in bounds
    if pid_m >= M or pid_n >= N:
        return
    
    # Cast base pointers to float32
    x_ptr = tl.cast(x_ptr, tl.pointer_type(tl.float32))
    weight_ptr = tl.cast(weight_ptr, tl.pointer_type(tl.float32))
    output_ptr = tl.cast(output_ptr, tl.pointer_type(tl.float32))
    
    
    for k_start in range(0, K, BLOCK_SIZE):
        # Create fixed-size offset tensor
        k_offsets = k_start + tl.arange(0, BLOCK_SIZE)
        # Create mask for valid K indices
        k_mask = k_offsets < K
        
        # Load x[m, k] for the current m and block of k
        x_offsets = pid_m * stride_xm + k_offsets * stride_xk
        x_values = tl.load(x_ptr + x_offsets, mask=k_mask, other=0.0)
        
        # Load weight[n, k] for the current n and block of k
        w_offsets = pid_n * stride_wn + k_offsets * stride_wk
        w_values = tl.load(weight_ptr + w_offsets, mask=k_mask, other=0.0)
        
        # Compute pre-sum output: x[m, k] * weight[n, k]
        pre_output = x_values * w_values
        
        # Store to output tensor
        out_offsets = pid_m * stride_om + pid_n * stride_on + k_offsets * stride_ok
        tl.store(output_ptr + out_offsets, pre_output, mask=k_mask)

def masked_linear(x: torch.Tensor, weight: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """
    Computes the pre-sum output tensor which is x.unsqueeze(1) * (weight*mask).unsqueeze(0)
    
    Args:
        x: Input tensor of shape (batch_size, in_features), i.e. (M, K)
        weight: Weight tensor of shape (out_features, in_features), i.e. (N, K)
        mask: Optional mask tensor of shape (out_features, in_features), i.e. (N, K)
        
    Returns:
        Pre-sum output tensor of shape (batch_size, out_features, in_features), i.e. (M, N, K)
    """
    assert x.is_cuda and weight.is_cuda, "Input tensors must be on CUDA device"
    assert x.dtype == weight.dtype, "Input and weight must have the same dtype"
    
    if mask is not None:
        assert mask.is_cuda, "Mask tensor must be on CUDA device"
        assert mask.shape == weight.shape, f"Mask shape {mask.shape} must match weight shape {weight.shape}"
        assert mask.dtype == weight.dtype, f"Mask dtype {mask.dtype} must match weight dtype {weight.dtype}"
        # Apply mask to weight
        masked_weight = weight * mask
    else:
        masked_weight = weight
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    masked_weight = masked_weight.contiguous()
    
    M, K = x.shape
    N, K_w = masked_weight.shape
    assert K == K_w, f"Input features {K} must match weight features {K_w}"
    
    # Allocate pre-sum output buffer
    pre_output = torch.empty((M, N, K), device=x.device, dtype=x.dtype)

    
    # Grid dimensions - we'll process each (m, n) pair and let the kernel handle the K dimension
    grid = (triton.cdiv(M, 1), triton.cdiv(N, 1))
    
    # Launch kernel for computing the pre-sum output tensor
    fixed_pre_sum_kernel[grid](
        x_ptr=x.data_ptr(),
        weight_ptr=masked_weight.data_ptr(),
        output_ptr=pre_output.data_ptr(),
        M=M, N=N, K=K,
        stride_xm=x.stride(0), stride_xk=x.stride(1),
        stride_wn=masked_weight.stride(0), stride_wk=masked_weight.stride(1),
        stride_om=pre_output.stride(0), stride_on=pre_output.stride(1), stride_ok=pre_output.stride(2),
        BLOCK_SIZE=32,  # Adjust as needed
    )
    
    return pre_output

def analyze_base_scores_vs_loss_change_efficient(base_scores, loss_change, save_dir):
    """
    Analyze the relationship between base_scores and loss_change using efficient PyTorch correlation functions.
    
    Args:
        base_scores: Tensor of shape [B, X, Y]
        loss_change: Tensor of shape [B, X, Y]
        save_dir: Directory to save analysis plots
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Get dimensions
    B, X, Y = base_scores.shape
    
    # Flatten X and Y dimensions for each batch element
    flattened_base_scores = base_scores.reshape(B, X * Y)
    flattened_loss_change = loss_change.reshape(B, X * Y)
    
    # Calculate correlations using efficient batch functions
    pearson_corrs = batch_pearsonr(flattened_base_scores, flattened_loss_change, dim=1)
    spearman_corrs = batch_spearmanr(flattened_base_scores, flattened_loss_change, dim=1)
    
    # Move to CPU for plotting
    pearson_corrs_np = pearson_corrs.detach().cpu().numpy()
    spearman_corrs_np = spearman_corrs.detach().cpu().numpy()
    
    # Print summary statistics
    debug_print("\nCorrelation Analysis:")
    debug_print(f"Mean Pearson correlation: {pearson_corrs_np.mean():.4f}")
    debug_print(f"Mean Spearman correlation: {spearman_corrs_np.mean():.4f}")
    debug_print(f"Min Spearman correlation: {spearman_corrs_np.min():.4f}")
    debug_print(f"Max Spearman correlation: {spearman_corrs_np.max():.4f}")
    
    # Create visualizations
    for b in range(10):
        # Get data for this batch element
        scores_flat = base_scores[b].detach().cpu().flatten().numpy()
        loss_flat = loss_change[b].detach().cpu().flatten().numpy()
        
        # Create visualization
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Scatter plot of base_scores vs loss_change
        plt.subplot(1, 3, 1)
        plt.scatter(scores_flat, loss_flat, alpha=0.3, s=1)
        plt.xlabel('Base Scores')
        plt.ylabel('Loss Change')
        plt.title(f'Batch {b}: Base Scores vs Loss Change\nPearson r={pearson_corrs_np[b]:.4f}, Spearman r={spearman_corrs_np[b]:.4f}')
        
        # Plot 2: Histogram of base_scores
        plt.subplot(1, 3, 2)
        plt.hist(scores_flat, bins=500)
        plt.xlabel('Base Scores')
        plt.ylabel('Count')
        plt.title(f'Distribution of Base Scores')
        
        # Plot 3: Histogram of loss_change
        plt.subplot(1, 3, 3)
        plt.hist(loss_flat, bins=500)
        plt.xlabel('Loss Change')
        plt.ylabel('Count')
        plt.title(f'Distribution of Loss Change')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'batch_{b}_analysis.png'), dpi=150)
        plt.close()
        
        # # Create a heatmap of the loss change
        # plt.figure(figsize=(12, 10))
        # plt.subplot(2, 1, 1)
        # plt.imshow(base_scores[b].detach().cpu().numpy(), cmap='viridis')
        # plt.colorbar(label='Base Scores')
        # plt.title(f'Batch {b}: Base Scores Heatmap')
        
        # plt.subplot(2, 1, 2)
        # plt.imshow(loss_change[b].detach().cpu().numpy(), cmap='viridis')
        # plt.colorbar(label='Loss Change')
        # plt.title(f'Batch {b}: Loss Change Heatmap')
        
        # plt.tight_layout()
        # plt.savefig(os.path.join(save_dir, f'batch_{b}_heatmaps.png'), dpi=150)
        # plt.close()
    
    # Create summary plot for all batches
    plt.figure(figsize=(10, 6))
    x = np.arange(B)
    width = 0.35
    
    plt.bar(x - width/2, pearson_corrs_np, width, label='Pearson')
    plt.bar(x + width/2, spearman_corrs_np, width, label='Spearman')
    
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Batch Index')
    plt.ylabel('Correlation Coefficient')
    plt.title('Correlation: Base Scores vs Loss Change')
    plt.ylim(-1.1, 1.1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add correlation values as text
    for i, (p_corr, s_corr) in enumerate(zip(pearson_corrs_np, spearman_corrs_np)):
        plt.text(i - width/2, p_corr + 0.05 * np.sign(p_corr), f'{p_corr:.2f}', 
                 ha='center', va='center', rotation=90, fontsize=8)
        plt.text(i + width/2, s_corr + 0.05 * np.sign(s_corr), f'{s_corr:.2f}', 
                 ha='center', va='center', rotation=90, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'correlation_summary.png'), dpi=150)
    plt.close()
    
    # Also check if zeros in base_scores correspond to zeros in loss_change
    zero_analysis = check_zeros_relationship(base_scores, loss_change)
    
    # Save zero analysis results
    with open(os.path.join(save_dir, 'zero_analysis.txt'), 'w') as f:
        f.write(zero_analysis)
    
    return pearson_corrs_np, spearman_corrs_np

def check_zeros_relationship(base_scores, loss_change):
    """Check if loss_change is zero when base_scores is zero"""
    result_text = "Zero Analysis Results:\n" + "-" * 50 + "\n"
    
    for b in range(base_scores.shape[0]):
        zero_mask = base_scores[b] == 0
        if zero_mask.any():
            zero_loss_diffs = loss_change[b][zero_mask]
            max_abs_diff = zero_loss_diffs.abs().max().item()
            mean_abs_diff = zero_loss_diffs.abs().mean().item()
            num_zeros = zero_mask.sum().item()
            
            result_text += f"Batch {b}:\n"
            result_text += f"  Number of zeros in base_scores: {num_zeros}\n"
            result_text += f"  Max absolute loss diff at zero positions: {max_abs_diff:.8f}\n"
            result_text += f"  Mean absolute loss diff at zero positions: {mean_abs_diff:.8f}\n"
            
            # Check if all values are close to zero
            is_close_to_zero = (zero_loss_diffs.abs() < 1e-5).all().item()
            result_text += f"  All zero positions have loss diff close to zero: {is_close_to_zero}\n\n"
        else:
            result_text += f"Batch {b}: No zeros found in base_scores\n\n"
    
    return result_text

def compute_sparsity_detailed(sparse_tensor):
    # Get dimensions
    total_elements = sparse_tensor.size(0) * sparse_tensor.size(1)
    nonzero_elements = sparse_tensor._nnz()
    sparsity = 1.0 - (nonzero_elements / total_elements)
    
    return {
        'sparsity': sparsity,
        'total_elements': total_elements,
        'nonzero_elements': nonzero_elements,
        'zero_elements': total_elements - nonzero_elements
    }



def create_sparse_output_tensor_dict(out_features, in_features, device):
    """
    Initializes a dictionary to store separate sparse tensors for each out_channel.
    
    Returns:
        all_cross_products: dict {out_channel -> sparse tensor (in_features, in_features)}
    """
    all_cross_products = {}
    
    # Initialize an empty sparse tensor for each out_channel
    for out_idx in range(out_features):
        indices = torch.zeros((2, 0), dtype=torch.long, device=device)
        values = torch.zeros(0, device=device)
        
        all_cross_products[out_idx] = torch.sparse_coo_tensor(
            indices, values, 
            size=(in_features, in_features), 
            device=device
        )
    
    return all_cross_products



def update_sparse_cross_products(all_cross_products, chunk_scores, device):
    batch_size, out_features, in_features = chunk_scores.shape
    
    for b in range(batch_size):
        scores = chunk_scores[b]
        min_score = 1
        nonzero_mask = torch.logical_or(scores < -min_score, scores > min_score)
        nonzero_indices = torch.nonzero(nonzero_mask)
        nonzero_values = scores[nonzero_mask]
        
        if nonzero_indices.shape[0] == 0:
            continue
            
        # Group by out_idx
        unique_out_indices = torch.unique(nonzero_indices[:, 0])  # Get unique out_channels
        
        for out_idx in unique_out_indices:
            mask = nonzero_indices[:, 0] == out_idx
            in_indices = nonzero_indices[mask, 1]
            values = nonzero_values[mask]
            
            if values.shape[0] == 0:
                continue
            
            # Compute outer product within this out_idx
            values_outer = torch.outer(values, values)
            
            # Generate indices within (in_features, in_features)
            rows = in_indices.unsqueeze(1).expand_as(values_outer).flatten()
            cols = in_indices.unsqueeze(0).expand_as(values_outer).flatten()
            
            indices = torch.stack([rows, cols], dim=0)
            sparse_block = torch.sparse_coo_tensor(
                indices, values_outer.flatten(), 
                size=(in_features, in_features), 
                device=device
            ).coalesce()

            # Store in dictionary (one sparse tensor per out_idx)
            if out_idx.item() not in all_cross_products:
                all_cross_products[out_idx.item()] = sparse_block
            else:
                all_cross_products[out_idx.item()] = (all_cross_products[out_idx.item()] + sparse_block).coalesce()
    
    return all_cross_products



def save_row_distributions_2d(acts, grads, save_dir, loop, in_features, out_features, unit=None):
    
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        if acts.shape != grads.shape:
            raise ValueError("acts and grads must have the same shape.")
        
        num_rows = acts.shape[0]
        for idx in range(num_rows):
            if idx != 0:
                continue  # Adjust if needed

            # Clear the previous figure
            plt.clf()
            
            # Create a new figure with a dark gray background
            fig, ax = plt.subplots(facecolor='#2B2B2B')  # Dark gray figure background
            ax.set_facecolor('#2B2B2B')  # Dark gray axis background
            
            # Extract the current row from both tensors and move to CPU
            if unit is not None:
                acts_row = acts[idx][unit].detach().cpu().numpy()
                grads_row = grads[idx][unit].detach().cpu().numpy()
            else:
                acts_row = acts[idx][0].detach().cpu().numpy()
                grads_row = grads[idx][0].detach().cpu().numpy()

            # Filter out zero values
            nonzero_acts = acts_row != 0
            nonzero_grads = grads_row != 0
            both_nonzero = np.logical_and(nonzero_acts, nonzero_grads)
            acts_row = acts_row[both_nonzero]
            grads_row = grads_row[both_nonzero]


            # print('numer of simultaneously nozero values', acts_row.shape)

            if not np.any(nonzero_acts) and not np.any(nonzero_grads):
                debug_print(f"Row {idx} has all zero values. Skipping.")
                continue

            # Compute symmetric log threshold
            linthresh = min(
                np.abs(acts_row[acts_row != 0]).min()/10,
                np.abs(grads_row[grads_row != 0]).min()/10
            )

            # Define symlog transformation
            def symlog_transform(x, linthresh):
                return np.sign(x) * (np.log10(np.abs(x) / linthresh + 1))

            # Apply symlog transformation
            acts_trans = symlog_transform(acts_row, linthresh)
            grads_trans = symlog_transform(grads_row, linthresh)


            # Define number of bins
            num_bins = max(len(acts_row), 2000)
            cmap = plt.cm.cool.copy()
            cmap.set_under(color='none')  # Transparent zero-count bins


            # Ensure values greater than linthresh exist
            pos_acts = acts_row[acts_row > 0]
            neg_acts = acts_row[acts_row < 0]

            pos_grads = grads_row[grads_row > 0]
            neg_grads = grads_row[grads_row < 0]

            # Find absolute max values for acts and grads
            if len(pos_acts) > 0:
                if len(neg_acts) > 0:  
                    max_abs_acts = max(pos_acts.max(), abs(neg_acts).max())
                else:
                    max_abs_acts = pos_acts.max()
            else:
                max_abs_acts = abs(neg_acts).max()
            if len(pos_grads) > 0:
                if len(neg_grads) > 0:
                    max_abs_grads = max(pos_grads.max(), abs(neg_grads).max())
                else:
                    max_abs_grads = pos_grads.max()
            else:
                max_abs_grads = abs(neg_grads).max()

           # Choose binning range based on the larger absolute max
            max_range_acts = max_abs_acts
            max_range_grads = max_abs_grads

            # Compute positive bins using geomspace()
            bin_edges_original_pos_acts = np.geomspace(np.abs(acts_row[acts_row != 0]).min()/10, 10*np.abs(acts_row[acts_row != 0]).max(), num_bins // 2) #if max_range_acts > np.abs(acts_row[acts_row != 0]).min() else np.array([np.abs(acts_row[acts_row != 0]).min()])
            bin_edges_original_pos_grads = np.geomspace(np.abs(grads_row[grads_row != 0]).min()/10, 10*np.abs(grads_row[grads_row != 0]).max(), num_bins // 2) #if max_range_grads > np.abs(grads_row[grads_row != 0]).min() else np.array([np.abs(grads_row[grads_row != 0]).min()])

            # Mirror positive bins for negative values using the same range
            bin_edges_original_neg_acts = -bin_edges_original_pos_acts
            bin_edges_original_neg_grads = -bin_edges_original_pos_grads

            # Combine negative, zero, and positive bins
            bin_edges_original_acts = np.sort(np.concatenate((bin_edges_original_neg_acts, [0], bin_edges_original_pos_acts)))
            bin_edges_original_grads = np.sort(np.concatenate((bin_edges_original_neg_grads, [0], bin_edges_original_pos_grads)))

            # Transform bin edges into symlog space
            bin_edges_symlog_acts = np.sort(symlog_transform(bin_edges_original_acts, linthresh))
            bin_edges_symlog_grads = np.sort(symlog_transform(bin_edges_original_grads, linthresh))

            # print(bin_edges_original_acts)
            # print(bin_edges_original_grads)

            # print(acts_row)
            # print(grads_row)


            # Create histogram with transformed data
            hist, xedges, yedges, image = plt.hist2d(
                acts_row, grads_row,
                bins=[bin_edges_original_acts, bin_edges_original_grads],
                cmap=cmap,
                vmin = 0.5,
            ) 
       
            plt.xscale('symlog', linthresh=np.abs(acts_row[acts_row != 0]).min()/10)
            plt.yscale('symlog', linthresh=np.abs(grads_row[grads_row != 0]).min()/10)

            # print("Peak count over all bins:", hist.max())
            # print(hist[hist!=0].min())

            # Set custom tick labels to represent original data scale
            def symlog_inverse_transform(ticks, linthresh):
                return np.sign(ticks) * (linthresh * (10**np.abs(ticks) - 1))
            
            # # Get and transform ticks
            # xticks = plt.xticks()[0]
            # yticks = plt.yticks()[0]
            # plt.xticks(xticks, [f"{symlog_inverse_transform(t, linthresh):.2e}" for t in xticks])
            # plt.yticks(yticks, [f"{symlog_inverse_transform(t, linthresh):.2e}" for t in yticks])

            # Add colorbar with white labels for readability
            cbar = plt.colorbar(image)
            cbar.set_label('Density', color='white')

            # Set labels and title with white text for visibility
            plt.xlabel('acts Value (symlog scale)', color='white')
            plt.ylabel('grads Value (symlog scale)', color='white')
            plt.title(f'2D Distribution for Row {idx + idx * loop}', color='white')

            # Update axis tick colors for readability
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')

            # Save the plot
            save_path = os.path.join(
                save_dir,
                f'row_{idx + idx * loop}_distribution_{out_features}by{in_features}.png'
            )
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())

            # print(f"Saved 2D histogram for row {idx + idx * loop} to {save_path}")

            plt.close()




def save_row_distributions(data, save_dir, loop, in_features, out_features, unit=None):
    """
    Create and save histograms for each row in the data using logarithmic bins.
    
    Args:
        data: 2D tensor/array (GPU)
        save_dir: Directory to save the histograms
        loop: Loop counter
    """
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        # summed = data.sum(dim=0)
        for idx, row in enumerate(data):
            if idx == 0:
                plt.clf()
                
                # Move data to CPU for histogram calculation

                if unit is not None:
                    row_cpu = row[unit].detach().cpu().numpy()
                else:
                    row_cpu = row[0].detach().cpu().numpy()
                debug_print('row_cpu shape', row_cpu.shape)
                
                # Create masks for positive and negative values
                pos_mask = row_cpu > 0
                neg_mask = row_cpu < 0
                
                # Get positive and negative values
                pos_values = row_cpu[pos_mask]
                debug_print('pos_values shape', pos_values.shape)
                pos_values_num = pos_values.shape[0]
                neg_values = np.abs(row_cpu[neg_mask])  # Take absolute value for logarithmic binning
                debug_print('neg_values shape', neg_values.shape)
                neg_values_num = neg_values.shape[0]
                nonzero_values_fraction = (pos_values_num + neg_values_num)/row_cpu.shape[0]
                debug_print(in_features, out_features, 'nonzero_values_fraction', nonzero_values_fraction)
                
                if len(pos_values) > 0:
                    # Create logarithmic bins for positive values
                    pos_bins = np.logspace(np.log10(pos_values.min()), np.log10(pos_values.max()), 10000)
                    pos_counts, _, _ = plt.hist(pos_values, bins=pos_bins, alpha=0.5, label='Positive', color='blue')
                
                if len(neg_values) > 0:
                    # Create logarithmic bins for negative values
                    neg_bins = -np.logspace(np.log10(neg_values.min()), np.log10(neg_values.max()), 10000)[::-1]
                    neg_counts, _, _ = plt.hist(-neg_values, bins=neg_bins, alpha=0.5, label='Negative', color='red')
                
                # Set x-axis to symmetric log scale
                plt.xscale('symlog', linthresh=np.min(np.abs(row_cpu[row_cpu != 0]))/10)
                
                # Add labels and title
                plt.xlabel('Value (log scale)')
                plt.ylabel('Count') 
                plt.xlim(row_cpu.min(), row_cpu.max())
                plt.ylim(0,np.max([np.max(pos_counts), np.max(neg_counts)]))
                plt.title(f'Distribution of Values for Row {idx + idx*loop}')
                plt.legend()
                
                # Save the plot
                save_path = os.path.join(save_dir, f'row_{idx + idx*loop}_distribution_{out_features}by{in_features}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                debug_print(f"Saved histogram for row {idx + idx*loop} to {save_path}")
                plt.close()
def batch_pearsonr(X, Y, dim=1):
    """
    Compute Pearson correlation coefficient batch-wise along a given dimension.

    Args:
        X, Y: Input tensors (torch.Tensor) of the same shape. (batch_size, num_features)
        dim: Dimension along which to compute correlation (usually features dimension).

    Returns:
        Tensor containing Pearson correlation coefficients for each element in the batch. (batch_size,)
    """
    X_mean = torch.mean(X, dim=dim, keepdim=True)
    Y_mean = torch.mean(Y, dim=dim, keepdim=True)
    Xm = X - X_mean
    Ym = Y - Y_mean
    r_num = torch.sum(Xm * Ym, dim=dim)
    # Use linalg.norm for stability and correctness
    Xm_norm = torch.linalg.norm(Xm, ord=2, dim=dim)
    Ym_norm = torch.linalg.norm(Ym, ord=2, dim=dim)
    r_den = Xm_norm * Ym_norm
    # Add small epsilon where denominator is zero (or close to it)
    # A small value like 1e-8 or 1e-6
    eps = torch.finfo(r_den.dtype).eps # Machine epsilon for the dtype
    r = r_num / torch.clamp(r_den, min=eps) # Clamp denominator to avoid division by zero/very small numbers
    # Clamp final result to valid range [-1, 1] due to potential floating point inaccuracies
    return torch.clamp(r, -1.0, 1.0)

def batch_spearmanr(X, Y, dim=1):
    """
    Compute Spearman rank correlation coefficient batch-wise along a given dimension.

    Args:
        X, Y: Input tensors (torch.Tensor) of the same shape. (batch_size, num_features)
        dim: Dimension along which to compute correlation (usually features dimension).

    Returns:
        Tensor containing Spearman correlation coefficients for each element in the batch. (batch_size,)
    """
    # Get ranks along the specified dimension using argsort twice
    X_rank = torch.argsort(torch.argsort(X, dim=dim), dim=dim).float()
    Y_rank = torch.argsort(torch.argsort(Y, dim=dim), dim=dim).float()

    # Compute Pearson correlation on the ranks
    return batch_pearsonr(X_rank, Y_rank, dim=dim)

def safe_pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute a scalar Pearson correlation in float32, returning 0 when the
    vectors are too short or nearly constant.
    """
    x = x.flatten().to(torch.float32)
    y = y.flatten().to(torch.float32)

    zero = torch.zeros((), device=x.device, dtype=torch.float32)
    if x.numel() < 2 or y.numel() < 2:
        return zero

    if torch.std(x, unbiased=False) < eps or torch.std(y, unbiased=False) < eps:
        return zero

    return batch_pearsonr(x.unsqueeze(0), y.unsqueeze(0), dim=1)[0]


def save_average_distributions(data, save_dir, in_features, out_features):
    """
    Create and save histograms for each row in the data using logarithmic bins.
    
    Args:
        data: 2D tensor/array (GPU)
        save_dir: Directory to save the histograms
        loop: Loop counter
    """
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
            plt.clf()
            
            # Move data to CPU for histogram calculation
            data_cpu = data.detach().cpu().numpy()
            
            # Create masks for positive and negative values
            pos_mask = data_cpu > 0
            neg_mask = data_cpu < 0
            non_zero_mask = data_cpu != 0
            
            # Get positive and negative values
            pos_values = data_cpu[pos_mask]
            neg_values = np.abs(data_cpu[neg_mask])  # Take absolute value for logarithmic binning
            
            if len(pos_values) > 0:
                # Create logarithmic bins for positive values
                pos_bins = np.logspace(np.log10(pos_values.min()/10), np.log10(pos_values.max()), 1000)
                pos_counts, _, _= plt.hist(pos_values, bins=pos_bins, alpha=0.5, label='Positive', color='blue')
            
            if len(neg_values) > 0:
                # Create logarithmic bins for negative values
                neg_bins = -np.logspace(np.log10(neg_values.min()/10), np.log10(neg_values.max()), 1000)[::-1]
                neg_counts, _, _ = plt.hist(-neg_values, bins=neg_bins, alpha=0.5, label='Negative', color='red')
            
            # Set x-axis to symmetric log scale
            plt.xscale('symlog', linthresh=np.min(np.abs(data_cpu[non_zero_mask]))/10)
            # print(data_cpu)
            # print('data contains nan or inf', np.isnan(data_cpu).any(), np.isinf(data_cpu).any())
            debug_print(np.min(data_cpu), np.max(data_cpu))
            plt.xlim(np.min(data_cpu), np.max(data_cpu))
            plt.ylim(0,np.max([np.max(pos_counts), np.max(neg_counts)]))
            
            # Add labels and title
            plt.xlabel('Value (log scale)')
            plt.ylabel('Count')
            plt.title(f'Distribution of Values')
            plt.legend()
            
            # Save the plot
            save_path = os.path.join(save_dir, f'distribution_{out_features}by{in_features}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            debug_print(f"Saved histogram to {save_path}")
            debug_print('number of nonzero values', (data_cpu != 0).sum())
            debug_print('number of zero values', (data_cpu == 0).sum())
            
            plt.close()



def zscore_standardize_scores_dissect(scores):
    """
    Perform z-score standardization on the scores for each layer.
    
    Args:
        scores (dict): A dictionary where keys are layer names and values are tensors of scores.
    
    Returns:
        OrderedDict: A dictionary with z-score standardized scores.
    """
    mean = torch.mean(scores)
    std = torch.std(scores)
    if std == 0:  # Avoid division by zero
        return scores - mean  # Subtract mean, but no scaling
    else:
        return (scores - mean) / std



def minmax_norm_scores_dissect(scores, min=0, max=1):
    old_min = torch.min(torch.abs(scores))
    old_max = torch.max(torch.abs(scores))
    if old_min == old_max:
        return torch.zeros_like(scores)
    
    pos_scores = torch.relu(scores)
    neg_scores = -torch.relu(-scores)
    pos_scores_minmaxed = torch.relu((pos_scores - old_min)/(old_max - old_min)*(max - min) + min)
    neg_scores_minmaxed = -torch.relu(-(neg_scores + old_min)/(old_max - old_min)*(max - min) + min)

    return pos_scores_minmaxed + neg_scores_minmaxed


def _compute_chunk_bie(
    boundary_inputs_chunk: torch.Tensor,
    boundary_pre_chunk: torch.Tensor,
    boundary_grads_chunk: torch.Tensor,
    weights: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Compute boundary integral estimators for a streaming chunk using local tensors.

    Returns:
        Tuple of (bie_base_scores, bie_overshoot_scores), each shaped [B, O, I], or (None, None)
        when inputs are invalid.
    """
    tensors = (boundary_inputs_chunk, boundary_pre_chunk, boundary_grads_chunk, weights)
    if not all(torch.is_tensor(t) for t in tensors):
        return None, None
    if boundary_inputs_chunk.dim() != 2 or boundary_pre_chunk.dim() != 2 or boundary_grads_chunk.dim() != 2:
        return None, None

    B, in_features = boundary_inputs_chunk.shape
    _, out_features = boundary_pre_chunk.shape
    if boundary_grads_chunk.shape != (B, out_features) or weights.shape != (out_features, in_features):
        return None, None

    target_device = weights.device
    dtype = torch.float32
    a = boundary_inputs_chunk.to(device=target_device, dtype=dtype)
    z = boundary_pre_chunk.to(device=target_device, dtype=dtype)
    delta = boundary_grads_chunk.to(device=target_device, dtype=dtype)
    W = weights.to(device=target_device, dtype=dtype)

    base_plane = -a.unsqueeze(1) * W.unsqueeze(0)  # [B, O, I]
    z_diff = z.unsqueeze(-1) - a.unsqueeze(1) * W.unsqueeze(0)
    active_mask = (z > 0).unsqueeze(-1)

    hold_mask = active_mask & (z_diff > 0)
    bie_state = base_plane * hold_mask.to(dtype)
    bie_base = delta.unsqueeze(-1) * bie_state

    overshoot_mask = active_mask & (z_diff <= 0)
    bie_correction = delta.unsqueeze(-1) * ((-z.unsqueeze(-1)) * overshoot_mask.to(dtype))
    return bie_base, bie_correction


def _compute_chunk_scores(
    chunk_weight_acts,
    chunk_neuron_grads,
    chunk_inputs,
    normalize_features_per_image,
    full_acts,
    pre_softmax_vector,
    weight_norm_type,
    original_activations_mean,
    weights,
    scale,
    grad_to_final_acts,
    final_acts,
    label,
    target_unit,
    target_unit_acts,
    cross_entropy_loss,
    score_post_act_function,
    boundary_inputs_chunk=None,
    boundary_pre_chunk=None,
    boundary_grads_chunk=None,
):
    """Static computation without control flow"""
    
    #(chunk_weight_acts.unsqueeze(1)*weights.unsqueeze(0)).to(dtype=torch.bfloat16, device=device)


    #OLD METRIC chunk_grad_actweights = 2 * activation_weight * activation_weight * chunk_grad_actweights
    #OLD grad_activation_weight = 2 * activation_weight.abs() * chunk_grad_actweights


    compute_loss_delta = score_post_act_function and COMPUTE_LOSS_DELTA

    if scale is None:
        scale_value = 1.0
    elif isinstance(scale, torch.Tensor):
        scale_value = float(scale.detach().cpu())
    else:
        scale_value = float(scale)
    if abs(scale_value) < 1e-12:
        scale_value = 1.0

    grad_weight = chunk_neuron_grads.unsqueeze(-1) * chunk_inputs.unsqueeze(1)
    grad_factor = grad_weight / scale_value
    weight_delta = weights.unsqueeze(0)
    base_scores = grad_factor * weight_delta

    if original_activations_mean is None:
        if base_scores.shape[1] == 1000:
            base_scores = base_scores[:,target_unit].unsqueeze(1)
        # print('base scores shape', base_scores.shape)
        abs_scores = torch.abs(base_scores)
        relu_scores = torch.relu(base_scores)

        if compute_loss_delta:
            with torch.no_grad():
                # 1. Prepare dimensions and flatten base scores
                B, X, Y = base_scores.shape
                N = X * Y
                M = full_acts.shape[1]
                
                flattened_base_scores = base_scores.flatten(start_dim=1).contiguous().to(device) # Shape: [B, N]
                full_acts = full_acts.to(device)
                label = label.to(device)
                # Pre-calculate original loss (important!)
                original_loss = cross_entropy_loss #torch.nn.functional.cross_entropy(full_acts, labels, reduction='none')

                # start_event.record()
                loss_change = compute_loss_difference_triton_optimized(
                    flattened_base_scores,
                    full_acts,
                    label,
                    target_unit,
                    original_loss
                )
                loss_change = loss_change.view(B, X, Y).contiguous()
                
                # print('loss_change positives', (loss_change.sum(dim=0) > 0).sum()/(loss_change.shape[1]*loss_change.shape[2]))
                # end_event.record()
                torch.cuda.synchronize() # Wait for the kernel to finish
        else:
            loss_change = None
        corr_with_true = safe_pearson_corr(base_scores.sum(dim=(1, 2)), target_unit_acts)
        # save_dir = '<a local folder>'
        # # print((base_scores[:,0,0]!=0).any(dim=0))
        # if (base_scores[:,0,0]!=0).any(dim=0):
        #     save_average_distributions(base_scores[:,0,0], save_dir, chunk_weight_acts.shape[2], chunk_weight_acts.shape[1])
        # tracked_weight = base_scores[:,0,0]
        if (
            DEBUG_SCORING
            and compute_loss_delta
            and grad_to_final_acts is not None
            and final_acts is not None
        ):
            output_length = final_acts.shape[1]

            # Get dimensions
            batch_size = chunk_weight_acts.shape[0]
            dim1_size = chunk_weight_acts.shape[1]
            dim2_size = chunk_weight_acts.shape[2]
            # print('layer of dim:', dim1_size, dim2_size)
            original_loss = F.cross_entropy(final_acts, label, reduction='none')
            linear_estimated_contribution = base_scores
            # Flatten M and K dimensions for ablation
            total_ablations = dim1_size*dim2_size

            # Reshape target_contributions to [N, M*K]
            target_contributions_flat = linear_estimated_contribution.view(batch_size, -1)  # [N, M*K]
            loss_fn = nn.CrossEntropyLoss(reduction='none')  # To compute loss per sample
            
            # Define chunk size (number of ablations to process at once)
            chunk_size = 100000 if dim1_size > 2000 and dim2_size > 5000 else (
            300000 if dim1_size > 2000 else 15000
        )

            # Calculate the number of chunks
            num_chunks = (total_ablations + chunk_size - 1) // chunk_size
            loss_per_weight = torch.zeros(batch_size, dim1_size, dim2_size, device=device)
            for chunk in range(num_chunks):
                # Determine the start and end indices for the current chunk
                start = chunk * chunk_size
                end = min((chunk + 1) * chunk_size, total_ablations)
                current_chunk_size = end - start

                # Extract the current chunk of target contributions
                # Shape: [N, current_chunk_size]
                current_contributions = target_contributions_flat[:, start:end]  # [N, current_chunk_size]

                # Compute new target logits after ablation
                # Shape: [N, current_chunk_size]
                original_target_logits = final_acts[:, target_unit].unsqueeze(1)  # [N, 1]
                new_target_logits = original_target_logits - current_contributions  # [N, current_chunk_size]

                # Expand original logits to accommodate the current chunk
                # Shape: [N, current_chunk_size, num_classes]
                expanded_logits = final_acts.unsqueeze(1).repeat(1, current_chunk_size, 1)  # [N, current_chunk_size, 1000]

                # Replace the target_unit logit with new_target_logits
                # Create a mask to zero out the target_unit
                mask = torch.ones_like(expanded_logits)
                mask[:, :, target_unit] = 0  # Zero out the target_unit across all ablations in the chunk

                # Apply mask and add new_target_logits
                # Shape: [N, current_chunk_size, 1000]
                modified_logits = expanded_logits * mask + new_target_logits.unsqueeze(2)  # [N, current_chunk_size, 1000]

                # Reshape modified_logits for loss computation
                # Shape: [N * current_chunk_size, 1000]
                modified_logits_flat = modified_logits.view(-1, output_length)  # [N*current_chunk_size, 1000]

                # Expand targets for the current chunk
                # Shape: [N * current_chunk_size]
                targets_expanded = label.unsqueeze(1).repeat(1, current_chunk_size).view(-1)  # [N*current_chunk_size]

                # Compute loss for the current chunk
                # Shape: [N * current_chunk_size]
                loss_flat = loss_fn(modified_logits_flat, targets_expanded)  # [N * current_chunk_size]

                # Reshape loss to [N, current_chunk_size]
                loss_chunk = loss_flat.view(batch_size, current_chunk_size)  # [N, current_chunk_size]

                # Assign the computed loss to the appropriate slice in loss_per_weight
                # Compute the corresponding M and K indices
                m_indices = torch.arange(dim1_size).unsqueeze(1).repeat(1, dim2_size).view(-1)[start:end]
                k_indices = torch.arange(dim2_size).repeat(dim1_size)[start:end]

                # Assign the loss to loss_per_weight
                loss_per_weight[:, m_indices, k_indices] = loss_chunk  # [N, M, K]
            loss_change = original_loss.unsqueeze(-1).unsqueeze(-1) - loss_per_weight
        else:
            loss_change = None


        # Initialize the output tensor
        # old_logits = final_acts.unsqueeze(-1).unsqueeze(-1)
        # dim1_batch_size = 200
        # dim2_batch_size = dim1_batch_size
        # # Process in batches for both dimensions
        # for dim1_start in range(0, dim1_size, dim1_batch_size):
        #     print('dim1_start', dim1_start)
        #     dim1_end = min(dim1_start + dim1_batch_size, dim1_size)
            
        #     for dim2_start in range(0, dim2_size, dim2_batch_size):
        #         dim2_end = min(dim2_start + dim2_batch_size, dim2_size)
                
        #         # Calculate base scores for current batch
        #         base_scores_batch = base_scores[:, dim1_start:dim1_end, dim2_start:dim2_end]
        #         base_scores_scaled_expanded_batch = (base_scores_batch/scale).unsqueeze(1)
                
        #         # Process grad_to_final_acts
        #         grad_to_final_acts_reshaped = grad_to_final_acts.unsqueeze(2).unsqueeze(3)
                
        #         # Compute contribution for current batch
        #         linear_estimated_contribution_batch = base_scores_scaled_expanded_batch * grad_to_final_acts_reshaped
        #         del grad_to_final_acts_reshaped, base_scores_scaled_expanded_batch
                
        #         # Compute new logits for current batch
        #         new_logits = old_logits - linear_estimated_contribution_batch

        #         del linear_estimated_contribution_batch
                
        #         if grad_to_final_acts is not None and final_acts is not None:
        #             # Compute original loss once
        #             if dim1_start == 0 and dim2_start == 0:
        #                 original_loss = F.cross_entropy(final_acts, label, reduction='none')
                    
        #             # Process the current batch for loss computation
        #             current_logits_flat = new_logits.permute(0, 2, 3, 1).reshape(-1, new_logits.shape[1])
                    
        #             # Expand labels for current batch
        #             current_batch_size = (dim1_end - dim1_start) * (dim2_end - dim2_start)
        #             current_labels = label.unsqueeze(1).repeat(1, current_batch_size).view(-1)
                    
        #             # Compute new loss for current batch
        #             new_loss_batch = F.cross_entropy(current_logits_flat, current_labels, reduction='none')
        #             new_loss_batch = new_loss_batch.view(batch_size, dim1_end - dim1_start, dim2_end - dim2_start)
                    
        #             # Compute and store loss change for current batch
        #             loss_change[:, dim1_start:dim1_end, dim2_start:dim2_end] = (
        #                 original_loss.unsqueeze(-1).unsqueeze(-1) - new_loss_batch
        #             )

        #         else:
        #             loss_change = None

                
        #         # Clean up intermediate tensors
        #         if 'current_logits_flat' in locals():
        #             del current_logits_flat
        #         if 'new_loss_batch' in locals():
        #             del new_loss_batch
        #         torch.cuda.empty_cache()


    else:
        original_activations_mean = original_activations_mean.unsqueeze(0)
        # print('chunk_weight_acts shape', chunk_weight_acts.shape)
        # print('original_activations_mean shape', original_activations_mean.shape)
        base_scores = grad_weight / scale_value
        abs_scores = torch.abs(base_scores)
        relu_scores = torch.relu(base_scores)


    if normalize_features_per_image:
        if weight_norm_type == 'sum':
            base_scores = base_scores/(1.0e-10 + ((base_scores)).sum(dim=(1,2), keepdim=True).abs())
            abs_scores = abs_scores/(abs_scores).sum(dim=(1,2), keepdim=True)
            relu_scores = relu_scores/(relu_scores).sum(dim=(1,2), keepdim=True)

        elif weight_norm_type == 'sum_abs':
            base_scores = base_scores/(1.0e-10 + ((base_scores)).abs().sum(dim=(1,2), keepdim=True).abs())
            abs_scores = abs_scores/(abs_scores).abs().sum(dim=(1,2), keepdim=True)
            relu_scores = relu_scores/((relu_scores).abs().sum(dim=(1,2), keepdim=True).abs())

        elif weight_norm_type == 'minmax':
            base_scores = minmax_norm_scores_dissect(base_scores)
            abs_scores = minmax_norm_scores_dissect(abs_scores)
            relu_scores = minmax_norm_scores_dissect(relu_scores)
        
        elif weight_norm_type == 'zscore':
            base_scores = zscore_standardize_scores_dissect(base_scores)
            abs_scores = zscore_standardize_scores_dissect(abs_scores)
            relu_scores = zscore_standardize_scores_dissect(relu_scores)

        elif weight_norm_type == 'output_magnitude':
            mask = (base_scores == 0) & (target_unit_acts == 0)
            # Perform division with masking
            base_scores = torch.where(mask, torch.tensor(0.0), base_scores / target_unit_acts)
            # Perform division with masking
            mask = (abs_scores == 0) & (target_unit_acts == 0)
            abs_scores = torch.where(mask, torch.tensor(0.0), abs_scores / target_unit_acts)

        elif weight_norm_type == 'fraction_pre_softmax':
            base_scores = base_scores/(pre_softmax_vector)
            abs_scores = abs_scores/(pre_softmax_vector)

        elif weight_norm_type == 'fraction_pre_softmax_times_output':
            base_scores = target_unit_acts*base_scores/(pre_softmax_vector)
            abs_scores = target_unit_acts*abs_scores/(pre_softmax_vector)
        
    bie_base_chunk: Optional[torch.Tensor] = None
    bie_correction_chunk: Optional[torch.Tensor] = None
    if (
        boundary_inputs_chunk is not None
        and boundary_pre_chunk is not None
        and boundary_grads_chunk is not None
    ):
        bie_base_chunk, bie_correction_chunk = _compute_chunk_bie(
            boundary_inputs_chunk,
            boundary_pre_chunk,
            boundary_grads_chunk,
            weights,
        )

    # Return all computations
    return {
        # 'tracked_weight': tracked_weight,
        'base_scores': base_scores,
        'loss_change': loss_change,
        'grad_nonzero': base_scores,#torch.count_nonzero(chunk_grad_actweights, dim=0),
        'corr': corr_with_true,
        'bie_base_scores': bie_base_chunk,
        'bie_overshoot_scores': bie_correction_chunk,
    }

# @torch.compile(backend='aot_eager')

__all__ = [
    '_prepare_softmax_stats',
    '_loss_diff_bn_kernel',
    'compute_loss_difference_triton_optimized',
    'dissected_conv_tiled_kernel',
    'fixed_pre_sum_kernel',
    'masked_linear',
    'analyze_base_scores_vs_loss_change_efficient',
    'check_zeros_relationship',
    'compute_sparsity_detailed',
    'create_sparse_output_tensor_dict',
    'update_sparse_cross_products',
    'save_row_distributions_2d',
    'save_row_distributions',
    'batch_pearsonr',
    'batch_spearmanr',
    'safe_pearson_corr',
    'save_average_distributions',
    'zscore_standardize_scores_dissect',
    'minmax_norm_scores_dissect',
    '_compute_chunk_bie',
    '_compute_chunk_scores',
]
