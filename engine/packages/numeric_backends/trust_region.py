from __future__ import annotations

import os

import torch

__all__ = [
    "TRITON_AVAILABLE",
    "inv_sym2x2_batch",
    "trust_region_step_fused",
]

_TRITON_IMPORT_ENABLED = str(
    os.getenv("MECHANISTIC_SYSTEMS_TRITON_IMPORT_ENABLED", os.getenv("CIRCUIT_TRITON_IMPORT_ENABLED", "1"))
).strip().lower() not in {"0", "false", "no", "off"}

try:  # pragma: no cover - optional dependency
    if not _TRITON_IMPORT_ENABLED:
        raise ImportError("Triton import disabled by MECHANISTIC_SYSTEMS_TRITON_IMPORT_ENABLED")
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    TRITON_AVAILABLE = False


def inv_sym2x2_batch(M: torch.Tensor, eps: float = 1e-18) -> torch.Tensor:
    """
    Invert batched symmetric 2x2 matrices M [..., 2, 2].
    Uses analytic inverse with determinant clamping for stability.
    """
    if M.dim() == 2:
        M = M.unsqueeze(0)

    a = M[..., 0, 0]
    b = M[..., 0, 1]
    c = M[..., 1, 1]

    det = (a * c - b * b).clamp_min(eps)

    inv = torch.empty_like(M)
    inv[..., 0, 0] = c / det
    inv[..., 0, 1] = -b / det
    inv[..., 1, 0] = -b / det
    inv[..., 1, 1] = a / det

    return inv


if TRITON_AVAILABLE:

    @triton.jit
    def _trust_region_step_kernel(
        g_w_ptr,
        g_b_ptr,
        H_ww_ptr,
        H_wb_ptr,
        H_bb_ptr,
        lam_ptr,
        ws_ptr,
        bs_ptr,
        ws_min_ptr,
        active_ptr,
        delta_w_ptr,
        delta_b_ptr,
        cand_ws_ptr,
        cand_bs_ptr,
        pred_red_ptr,
        enforce_positive_ws: tl.constexpr,
        B: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < B

        g_w = tl.load(g_w_ptr + offsets, mask=mask)
        g_b = tl.load(g_b_ptr + offsets, mask=mask)
        H_ww = tl.load(H_ww_ptr + offsets, mask=mask)
        H_wb = tl.load(H_wb_ptr + offsets, mask=mask)
        H_bb = tl.load(H_bb_ptr + offsets, mask=mask)
        lam = tl.load(lam_ptr + offsets, mask=mask)
        ws = tl.load(ws_ptr + offsets, mask=mask)
        bs = tl.load(bs_ptr + offsets, mask=mask)
        ws_min = tl.load(ws_min_ptr + offsets, mask=mask)
        active = tl.load(active_ptr + offsets, mask=mask).to(tl.float32)

        H_ww_d = H_ww + lam
        H_bb_d = H_bb + lam
        det = H_ww_d * H_bb_d - H_wb * H_wb
        det = tl.maximum(det, 1e-12)
        det_inv = 1.0 / det

        delta_w = -(H_bb_d * g_w - H_wb * g_b) * det_inv
        delta_b = -(-H_wb * g_w + H_ww_d * g_b) * det_inv

        cand_ws = ws + delta_w
        cand_bs = bs + delta_b
        if enforce_positive_ws:
            cand_ws = tl.maximum(cand_ws, ws_min)

        quad = (
            delta_w * (H_ww * delta_w + H_wb * delta_b)
            + delta_b * (H_wb * delta_w + H_bb * delta_b)
        )
        pred_red = -(g_w * delta_w + g_b * delta_b) - 0.5 * quad

        delta_w = delta_w * active
        delta_b = delta_b * active
        cand_ws = cand_ws * active + ws * (1 - active)
        cand_bs = cand_bs * active + bs * (1 - active)
        pred_red = pred_red * active

        tl.store(delta_w_ptr + offsets, delta_w, mask=mask)
        tl.store(delta_b_ptr + offsets, delta_b, mask=mask)
        tl.store(cand_ws_ptr + offsets, cand_ws, mask=mask)
        tl.store(cand_bs_ptr + offsets, cand_bs, mask=mask)
        tl.store(pred_red_ptr + offsets, pred_red, mask=mask)


def trust_region_step_fused(
    g_w: torch.Tensor,
    g_b: torch.Tensor,
    H_ww: torch.Tensor,
    H_wb: torch.Tensor,
    H_bb: torch.Tensor,
    lam: torch.Tensor,
    ws: torch.Tensor,
    bs: torch.Tensor,
    ws_min_tensor: torch.Tensor,
    active: torch.Tensor,
    enforce_positive_ws: bool,
    out_buffers: dict[str, torch.Tensor] | None = None,
):
    """Fused trust region step using Triton kernel."""
    B = g_w.shape[0]
    device, dtype = g_w.device, g_w.dtype

    def _buffer(name: str) -> torch.Tensor:
        if out_buffers is None:
            return torch.empty(B, device=device, dtype=dtype)
        existing = out_buffers.get(name)
        if (
            existing is None
            or existing.shape != (B,)
            or existing.device != device
            or existing.dtype != dtype
        ):
            existing = torch.empty(B, device=device, dtype=dtype)
            out_buffers[name] = existing
        return existing

    delta_w = _buffer("delta_w")
    delta_b = _buffer("delta_b")
    cand_ws = _buffer("cand_ws")
    cand_bs = _buffer("cand_bs")
    pred_red = _buffer("pred_red")

    if ws_min_tensor is None:
        ws_min_tensor = torch.full((B,), 1e-8, device=device, dtype=dtype)

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton not available for fused trust region step.")

    BLOCK_SIZE = 256
    grid = ((B + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _trust_region_step_kernel[grid](
        g_w,
        g_b,
        H_ww,
        H_wb,
        H_bb,
        lam,
        ws,
        bs,
        ws_min_tensor,
        active.to(torch.int32),
        delta_w,
        delta_b,
        cand_ws,
        cand_bs,
        pred_red,
        enforce_positive_ws=enforce_positive_ws,
        B=B,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return delta_w, delta_b, cand_ws, cand_bs, pred_red
