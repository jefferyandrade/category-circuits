"""
Bitpacking helpers shared across salience frontier modules.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
import torch
try:  # pragma: no cover - optional dependency
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
    _HAS_TRITON = True
except Exception:  # pragma: no cover - fallback
    triton = None  # type: ignore
    tl = None  # type: ignore
    _HAS_TRITON = False

from packages.salience.core.frontier_utils import COMPUTE_BITPACK


if _HAS_TRITON:
    @triton.jit
    def _pack_bits_kernel(
        src_ptr,
        dst_ptr,
        n_cols,
        src_stride_row,
        src_stride_col,
        dst_stride_row,
        dst_stride_chunk,
        CHUNK: tl.constexpr,
    ):
        row = tl.program_id(0)
        chunk = tl.program_id(1)

        col_start = chunk * CHUNK
        offs = col_start + tl.arange(0, CHUNK)
        mask = offs < n_cols

        src_offsets = row * src_stride_row + offs * src_stride_col
        bits = tl.load(src_ptr + src_offsets, mask=mask, other=0).to(tl.uint64)
        shifts = tl.arange(0, CHUNK).to(tl.uint64)
        word = tl.sum(bits << shifts, axis=0)

        dst_offset = row * dst_stride_row + chunk * dst_stride_chunk
        tl.store(dst_ptr + dst_offset, word)


def vectorised_bitpack(
    kernel_scores_sum: torch.Tensor,
    threshold: float = 0.0,
    *,
    sink: Optional[Callable[[torch.ByteTensor], None]] = None,
) -> Optional[List[torch.ByteTensor]]:
    """
    Convert boolean masks derived from kernel scores into packed byte tensors.

    Args:
        kernel_scores_sum: Tensor of shape ``(B, ...)`` whose entries will be thresholded.
        threshold: Values with absolute magnitude greater than this threshold are treated as ``True``.
        sink: Optional callable that receives the packed rows on CPU. When provided the packed tensor
            is passed to the sink and ``None`` is returned.

    Returns:
        ``None`` if bitpacking is disabled or the sink handled the packed rows; otherwise a list of
        CPU tensors (one per batch row) containing the packed bits.
    """
    if not COMPUTE_BITPACK:
        return None
    with torch.no_grad():
        flat = (kernel_scores_sum.abs() > threshold).reshape(kernel_scores_sum.shape[0], -1)
        if flat.numel() == 0:
            return []
        flat_uint8 = flat.to(torch.uint8)
        if hasattr(torch, "packbits"):
            byte_rows = torch.packbits(flat_uint8, dim=1)
            byte_rows = byte_rows.to(flat.device)
        else:
            byte_rows_np = np.packbits(flat_uint8.cpu().numpy(), axis=1)
            byte_rows = torch.from_numpy(byte_rows_np).to(flat.device)
        if sink is not None:
            sink(byte_rows.cpu())
            return None
        return [row.cpu() for row in byte_rows]


def pack_boolean_rows(tensor: torch.Tensor) -> torch.Tensor:
    """
    Pack a 2D boolean/binary matrix into uint64 bitsets per row.

    Uses Triton when available; falls back to a torch/numpy implementation otherwise.
    """
    if tensor.numel() == 0:
        return torch.zeros((tensor.size(0), 0), dtype=torch.uint64, device=tensor.device)

    bool_tensor = tensor.to(dtype=torch.bool, device=tensor.device)
    if not bool_tensor.is_contiguous():
        bool_tensor = bool_tensor.contiguous()

    n_rows, n_cols = bool_tensor.shape
    n_chunks = (n_cols + 63) // 64
    if n_chunks == 0:
        return torch.zeros((n_rows, 0), dtype=torch.uint64, device=bool_tensor.device)

    if _HAS_TRITON and bool_tensor.is_cuda:
        packed = torch.empty((n_rows, n_chunks), dtype=torch.uint64, device=bool_tensor.device)
        grid = (n_rows, n_chunks)
        _pack_bits_kernel[grid](
            bool_tensor,
            packed,
            n_cols,
            bool_tensor.stride(0),
            bool_tensor.stride(1),
            packed.stride(0),
            packed.stride(1),
            CHUNK=64,
        )
        return packed

    # Fallback: use torch.packbits or numpy packbits on CPU then reshape.
    flat = bool_tensor.reshape(n_rows, n_cols).to(torch.uint8)
    if hasattr(torch, "packbits"):
        packed_bytes = torch.packbits(flat, dim=1)
        packed_bytes = packed_bytes.to(bool_tensor.device)
    else:
        packed_np = np.packbits(flat.cpu().numpy(), axis=1)
        packed_bytes = torch.from_numpy(packed_np).to(bool_tensor.device)

    packed_np = np.zeros((n_rows, n_chunks), dtype=np.uint64)
    flat_np = flat.cpu().numpy().astype(np.uint8, copy=False)
    for chunk_idx in range(n_chunks):
        start = chunk_idx * 64
        end = min(start + 64, n_cols)
        word = np.zeros(n_rows, dtype=np.uint64)
        for bit_idx in range(start, end):
            word |= flat_np[:, bit_idx].astype(np.uint64) << np.uint64(bit_idx - start)
        packed_np[:, chunk_idx] = word
    return torch.from_numpy(packed_np).to(bool_tensor.device)


__all__ = ["vectorised_bitpack", "pack_boolean_rows"]
