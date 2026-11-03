from __future__ import annotations

import logging
from typing import List, Optional

import torch

from packages.salience.ops.linear_utils import safe_pearson_corr

logger = logging.getLogger(__name__)


def flatten_objective_vec(obj: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
	if obj is None or not torch.is_tensor(obj):
		return None
	if obj.dim() == 1:
		flattened = obj
	else:
		leading = obj.shape[0]
		flattened = obj.reshape(leading, -1).sum(dim=1)
	return flattened.detach().to(device="cpu", dtype=torch.float32)


def log_layer_debug(
	layer_name: Optional[str],
	per_sample_scores: torch.Tensor,
	objective_vec: torch.Tensor,
) -> None:
	if layer_name is None:
		return
	vector = per_sample_scores.to(device="cpu", dtype=torch.float32).reshape(-1)
	target_vec = objective_vec.to(device="cpu", dtype=torch.float32).reshape(-1)
	if vector.numel() == 0 or target_vec.numel() == 0:
		return
	samples = min(vector.numel(), target_vec.numel())
	if samples == 0:
		return
	vector = vector[:samples]
	target_vec = target_vec[:samples]
	score_sum = float(vector.sum().item())
	objective_sum = float(target_vec.sum().item())
	corr_val = float("nan")
	if samples > 1:
		corr_val = float(safe_pearson_corr(vector, target_vec).item())
	logger.info(
		"[layer-debug] layer=%s samples=%d score_sum=%.6f objective_sum=%.6f corr=%.6f",
		layer_name or "unknown",
		samples,
		score_sum,
		objective_sum,
		corr_val,
	)


def log_bie_layer_debug(
    layer_name: Optional[str],
    base_sum: float,
    boundary_sum: float,
    corrected_sum: float,
    sample_total: float,
    *,
    corr: Optional[float] = None,
) -> None:
    if layer_name is None:
        return
    corr_value = float("nan") if corr is None else float(corr)
    logger.info(
        "[layer-debug-bie] layer=%s samples=%.0f score_sum=%.6f base_sum=%.6f boundary_sum=%.6f boundary_frac=%.6f corr=%.6f",
        layer_name,
        sample_total,
        corrected_sum,
        base_sum,
        boundary_sum,
        0.0 if corrected_sum == 0.0 else boundary_sum / corrected_sum,
        corr_value,
    )


def maybe_record_layer_corr_sample(
	score_chunks: Optional[List[torch.Tensor]],
	objective_chunks: Optional[List[torch.Tensor]],
	per_sample_scores: Optional[torch.Tensor],
	objective_vec: Optional[torch.Tensor],
) -> None:
	if (
		score_chunks is None
		or objective_chunks is None
		or per_sample_scores is None
		or objective_vec is None
	):
		return
	if not torch.is_tensor(per_sample_scores) or not torch.is_tensor(objective_vec):
		return
	if per_sample_scores.shape[0] != objective_vec.shape[0]:
		return
	score_chunks.append(per_sample_scores.detach().to(device="cpu", dtype=torch.float32))
	objective_chunks.append(objective_vec.detach().to(device="cpu", dtype=torch.float32))


def finalize_layer_corr_logging(
	layer_name: Optional[str],
	score_chunks: Optional[List[torch.Tensor]],
	objective_chunks: Optional[List[torch.Tensor]],
) -> None:
	if (
		layer_name is None
		or not score_chunks
		or not objective_chunks
	):
		return
	score_vec = torch.cat(score_chunks, dim=0)
	objective_vec = torch.cat(objective_chunks, dim=0)
	if score_vec.numel() == 0 or objective_vec.numel() == 0:
		return
	samples = min(score_vec.shape[0], objective_vec.shape[0])
	if samples == 0:
		return
	log_layer_debug(layer_name, score_vec[:samples], objective_vec[:samples])


def finalize_layer_bie_corr_logging(
	layer_name: Optional[str],
	score_chunks: Optional[List[torch.Tensor]],
	objective_chunks: Optional[List[torch.Tensor]],
	base_sum: float,
	boundary_sum: float,
) -> None:
	if (
		layer_name is None
		or not score_chunks
		or not objective_chunks
	):
		return
	score_vec = torch.cat(score_chunks, dim=0)
	objective_vec = torch.cat(objective_chunks, dim=0)
	if score_vec.numel() == 0 or objective_vec.numel() == 0:
		return
	samples = min(score_vec.shape[0], objective_vec.shape[0])
	if samples == 0:
		return
	score_vec = score_vec[:samples]
	objective_vec = objective_vec[:samples]
	corr_val = float("nan")
	if samples > 1:
		corr_val = float(safe_pearson_corr(score_vec, objective_vec).item())
	corrected_sum = float(score_vec.sum().item())
	reference_sum = float(objective_vec.sum().item())

	log_bie_layer_debug(
		layer_name,
		float(base_sum),
		float(boundary_sum),
		float(base_sum + boundary_sum),
		float(samples),
		corr=corr_val,
	)
	logger.info(
		"[layer-debug-bie-corr] layer=%s samples=%d score_sum=%.6f objective_sum=%.6f",
		layer_name,
		samples,
		corrected_sum,
		reference_sum,
	)
