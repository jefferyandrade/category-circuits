from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import torch

try:  # pragma: no cover - optional compiled extension
	from . import fused_scores_kernel as _fused_scores_cuda  # type: ignore[attr-defined]
	_HAS_FUSED_KERNEL = True
except Exception:  # pragma: no cover - optional dependency
	_fused_scores_cuda = None  # type: ignore[assignment]
	_HAS_FUSED_KERNEL = False


def has_compiled_kernel() -> bool:
	"""Return True when the CUDA fused score extension was successfully imported."""
	return bool(_HAS_FUSED_KERNEL and _fused_scores_cuda is not None)


def _env_enabled() -> bool:
	return bool(int(os.environ.get("SCORING_USE_FUSED", "0")))


def _python_reduce(
	lhs: torch.Tensor,
	rhs: torch.Tensor,
	reduce_dims: Sequence[int],
	*,
	compute_abs: bool,
	compute_square: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
	product = lhs * rhs
	sum_tensor = product.sum(dim=reduce_dims)
	abs_tensor = product.abs().sum(dim=reduce_dims) if compute_abs else None
	sq_tensor = (product * product).sum(dim=reduce_dims) if compute_square else None
	return sum_tensor, abs_tensor, sq_tensor


def _prepare_segments(
	lhs: torch.Tensor,
	rhs: torch.Tensor,
	reduce_dims: Sequence[int],
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, ...]]]:
	dims = tuple(sorted(reduce_dims))
	if dims == (0,) and lhs.dim() == 2:
		lhs2 = lhs.transpose(0, 1).contiguous()
		rhs2 = rhs.transpose(0, 1).contiguous()
		num_segments = lhs2.size(0)
		segment_len = lhs2.size(1)
		offsets = torch.arange(
			0,
			(num_segments + 1) * segment_len,
			segment_len,
			device=lhs.device,
			dtype=torch.int64,
		)
		return lhs2.reshape(-1), rhs2.reshape(-1), offsets, (lhs2.size(0),)
	if dims == (0, 2, 3) and lhs.dim() == 4:
		lhs2 = lhs.permute(1, 0, 2, 3).contiguous().view(lhs.size(1), -1)
		rhs2 = rhs.permute(1, 0, 2, 3).contiguous().view(rhs.size(1), -1)
		num_segments = lhs2.size(0)
		segment_len = lhs2.size(1)
		offsets = torch.arange(
			0,
			(num_segments + 1) * segment_len,
			segment_len,
			device=lhs.device,
			dtype=torch.int64,
		)
		return lhs2.reshape(-1), rhs2.reshape(-1), offsets, (lhs2.size(0),)
	return None


def reduce_mul(
	lhs: torch.Tensor,
	rhs: torch.Tensor,
	*,
	reduce_dims: Sequence[int],
	compute_abs: bool,
	compute_square: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
	if not torch.is_tensor(lhs) or not torch.is_tensor(rhs):
		raise TypeError("lhs and rhs must be tensors.")
	if lhs.shape != rhs.shape:
		raise ValueError("lhs and rhs must have matching shapes.")

	if not _env_enabled():
		return _python_reduce(lhs, rhs, reduce_dims, compute_abs=compute_abs, compute_square=compute_square)

	prepared = _prepare_segments(lhs, rhs, reduce_dims)
	if prepared is None:
		return _python_reduce(lhs, rhs, reduce_dims, compute_abs=compute_abs, compute_square=compute_square)

	lhs_flat, rhs_flat, offsets, keep_shape = prepared
	num_segments = int(offsets.numel() - 1)
	if num_segments <= 0:
		return _python_reduce(lhs, rhs, reduce_dims, compute_abs=compute_abs, compute_square=compute_square)

	use_kernel = (
		has_compiled_kernel()
		and torch.cuda.is_available()
		and lhs.device.type == "cuda"
		and rhs.device.type == "cuda"
		and lhs.dtype == torch.float32
		and rhs.dtype == torch.float32
	)

	if use_kernel:
		try:
			segment_len = int(offsets[1].item() - offsets[0].item()) if offsets.numel() >= 2 else 0
			if segment_len <= 0:
				return _python_reduce(lhs, rhs, reduce_dims, compute_abs=compute_abs, compute_square=compute_square)
			block = 256 if segment_len >= 256 else segment_len
			if block <= 0:
				block = 1
			sum_out = torch.zeros(num_segments, device=lhs.device, dtype=torch.float32)
			abs_out = torch.zeros_like(sum_out)
			sq_out = torch.zeros_like(sum_out)
			_fused_scores_cuda.fused_scores_kernel(  # type: ignore[attr-defined]
				lhs_flat,
				rhs_flat,
				sum_out,
				abs_out,
				sq_out,
				offsets,
				num_segments,
				int(compute_abs),
				int(compute_square),
				grid=(num_segments, 1, 1),
				block=(block, 1, 1),
			)
			sum_tensor = sum_out.reshape(keep_shape)
			abs_tensor = abs_out.reshape(keep_shape) if compute_abs else None
			sq_tensor = sq_out.reshape(keep_shape) if compute_square else None
			return sum_tensor, abs_tensor, sq_tensor
		except Exception:
			pass

	return _python_reduce(lhs, rhs, reduce_dims, compute_abs=compute_abs, compute_square=compute_square)


__all__ = [
	"has_compiled_kernel",
	"reduce_mul",
]
