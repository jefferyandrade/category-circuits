"""
Skeleton utilities for low-rank Hessian sketches.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import torch


def power_iteration(
    hvp_fn: Callable[[torch.Tensor], torch.Tensor],
    initial_vector: torch.Tensor,
    num_steps: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Approximate the dominant eigenpair of a symmetric operator via power iteration."""

    vec = initial_vector.detach().clone()
    vec = vec / (vec.norm() + 1e-12)

    eigenvalue = torch.tensor(0.0, device=vec.device, dtype=vec.dtype)

    for _ in range(max(num_steps, 1)):
        hv = hvp_fn(vec)
        norm = hv.norm()
        if norm.item() == 0:
            break
        vec = hv / norm
        eigenvalue = torch.dot(vec.flatten(), hvp_fn(vec).flatten())

    return eigenvalue, vec


def subspace_principal_components(
    vectors: Sequence[torch.Tensor],
    responses: Sequence[torch.Tensor],
    *,
    top_k: int = 2,
    fp16: bool = False,
    max_iters: int = 10,
    conv_tol: float = 1e-3,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[Optional[torch.Tensor], List[torch.Tensor], Optional[torch.Tensor]]:
    """Extract dominant modes from a set of (v, Hv) pairs.

    Args:
        vectors: Sequence of direction tensors ``v_i``.
        responses: Sequence of Hessian responses ``H v_i`` matching ``vectors``.
        top_k: Number of principal components to return.

    Returns:
        A tuple ``(eigenvalues, modes, H_sub)`` where ``eigenvalues`` is a tensor
        of the leading eigenvalues (largest magnitude), ``modes`` are the
        corresponding eigenvectors expressed in the original tensor space, and
        ``H_sub`` is the small matrix ``VᵀHV`` used for the decomposition.
    """

    if not vectors or not responses:
        return None, [], None

    count = min(len(vectors), len(responses))
    if count == 0:
        return None, [], None

    flat_vecs = torch.stack([vectors[i].reshape(-1) for i in range(count)], dim=1)
    flat_resp = torch.stack([responses[i].reshape(-1) for i in range(count)], dim=1)
    h_sub = flat_vecs.t().matmul(flat_resp)
    h_sub = 0.5 * (h_sub + h_sub.t())

    if h_sub.numel() == 0:
        return None, [], h_sub

    work_dtype = torch.float16 if fp16 else h_sub.dtype
    h_work = h_sub.to(dtype=work_dtype)
    eigvals, eigvecs = torch.linalg.eigh(h_work)
    if log_fn is not None:
        try:
            log_fn(f"eigh dtype={h_work.dtype} eigvals_min={float(eigvals.min().item()):.3e} eigvals_max={float(eigvals.max().item()):.3e}")
        except Exception:
            pass
    if eigvals.numel() == 0:
        return None, [], h_sub

    k = max(1, min(top_k, eigvals.numel()))
    idx = eigvals.argsort()
    top_indices = idx[-k:]
    top_vals = eigvals[top_indices]

    modes: List[torch.Tensor] = []
    basis = eigvecs[:, top_indices]
    for column in basis.unbind(dim=1):
        mode = torch.zeros_like(vectors[0])
        for coeff, vec in zip(column, vectors):
            mode = mode + coeff.item() * vec
        norm = mode.norm()
        if norm > 0:
            mode = mode / norm
        modes.append(mode)

    return top_vals, modes, h_sub
