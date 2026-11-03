"""Bounded literal-J row tiles for the unweighted context Gram.

The tensor workspace excludes resident source factors and vendor BLAS/allocator
workspaces. Real-number semantics equal J J.T; reduction order differs from the
basis-adjoint reference. This module issues no numerical enclosure or authority.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .operators import ChartOperators


@dataclass(frozen=True)
class GramLayerPlan:
    name: str
    fibers: int
    inputs: int
    positions: int
    fiber_tile: int
    input_tile: int
    context_tile: int
    row_tile_bytes: int
    masked_weight_bytes: int
    factor_copy_bytes: int
    explicit_peak_bytes: int


@dataclass(frozen=True)
class ResponseGramPlan:
    context_count: int
    element_size: int
    workspace_bytes: int
    output_bytes: int
    symmetrization_bytes: int
    explicit_peak_bytes: int
    layers: tuple[GramLayerPlan, ...]

    def to_json(self) -> dict:
        payload = asdict(self)
        payload['layers'] = [asdict(layer) for layer in self.layers]
        return dict(schema='tractable_v1_tiled_response_gram_plan_v1', **payload,
                    scope='explicit_tensors_including_output_excluding_source_factors_and_vendor_workspace',
                    unbounded_context_route_matrix_retained=False)


def plan_response_gram(ops: ChartOperators, *, workspace_bytes: int = 256 << 20) -> ResponseGramPlan:
    """Shape-only tensor allocation plan; refuse before allocating a Gram."""
    if type(workspace_bytes) is not int or workspace_bytes <= 0:
        raise ValueError('positive integer response Gram workspace_bytes required')
    B = ops.context_count
    if type(B) is not int or B < 1 or ops.dtype not in (torch.float32, torch.float64):
        raise ValueError('positive contexts and float32/float64 factors required')
    element = torch.finfo(ops.dtype).bits // 8
    output = B * B * element
    available = (workspace_bytes - 2 * output) // element
    if available <= 0:
        raise ValueError('response Gram output and symmetrization exceed workspace')
    plans = []
    for layer in ops.factors.layers:
        F, S = layer.W.shape
        if layer.X.ndim != 3 or layer.D.ndim != 3:
            raise ValueError('response Gram requires three-dimensional X and D factors')
        P = layer.X.shape[-1]
        if (F < 1 or S < 1 or P < 1 or tuple(layer.X.shape) != (B, S, P)
                or tuple(layer.D.shape) != (B, F, P) or tuple(layer.mask.shape) != (F, S)
                or F != layer.geometry.fiber_count or S != layer.geometry.routes_per_fiber):
            raise ValueError('response Gram factor geometry differs from the captured axis')
        for value in (layer.X, layer.D, layer.W, layer.mask):
            if value.dtype != ops.dtype or value.device != ops.device or value.requires_grad:
                raise ValueError('response Gram requires same-device/dtype detached captured factors')
        if P == 1:
            capacity = available // (B + 1)
            if capacity < 1:
                raise ValueError('one response Gram route tile exceeds workspace')
            st = min(S, capacity)
            ft = min(F, capacity // st)
            ct = B
            copies = 0
        else:
            # Keep at most half the provisional budget for copied factors;
            # choose route dimensions using the remaining exact inequality.
            ct = min(B, max(1, available // (2 * P * (F + S))))
            st = min(S, (available - ct * P) // (B + 1 + ct * P))
            if st < 1:
                raise ValueError('one context/route response Gram contraction exceeds workspace')
            ft = min(F, (available - ct * P * st) // ((B + 1) * st + ct * P))
            if ft < 1:
                raise ValueError('one response Gram fiber tile exceeds workspace')
            copies = ct * P * (ft + st) * element
        rows = B * ft * st * element
        weight = ft * st * element
        peak = 2 * output + rows + weight + copies
        if peak > workspace_bytes:
            raise ValueError('response Gram plan exceeds its explicit tensor workspace')
        plans.append(GramLayerPlan(layer.geometry.name, F, S, P, ft, st, ct, rows, weight, copies, peak))
    if not plans:
        raise ValueError('response Gram requires at least one captured layer')
    return ResponseGramPlan(B, element, workspace_bytes, output, output,
                            max(p.explicit_peak_bytes for p in plans), tuple(plans))


def tiled_response_gram(ops: ChartOperators, *, workspace_bytes: int = 256 << 20) -> torch.Tensor:
    """Return unweighted J J.T without a full context-by-route matrix.

    ``R[c,f,j] = -(W*m)[f,j] sum_p D[c,f,p] X[c,j,p]``. Each
    disjoint route tile contributes ``R_tile @ R_tile.T`` exactly in real
    arithmetic. No context weights, corrected cotangents or centering enter.
    """
    if torch.is_autocast_enabled('cpu') or torch.is_autocast_enabled('cuda'):
        raise ValueError('response Gram refuses ambient autocast')
    plan = plan_response_gram(ops, workspace_bytes=workspace_bytes)
    B = plan.context_count
    gram = torch.zeros((B, B), device=ops.device, dtype=ops.dtype)
    for layer, lp in zip(ops.factors.layers, plan.layers, strict=True):
        for f0 in range(0, lp.fibers, lp.fiber_tile):
            f1 = min(f0 + lp.fiber_tile, lp.fibers)
            for s0 in range(0, lp.inputs, lp.input_tile):
                s1 = min(s0 + lp.input_tile, lp.inputs)
                wm = layer.W[f0:f1, s0:s1] * layer.mask[f0:f1, s0:s1]
                wm.neg_()
                rows = torch.empty((B, f1 - f0, s1 - s0), device=ops.device, dtype=ops.dtype)
                if lp.positions == 1:
                    torch.mul(layer.D[:, f0:f1, 0, None], layer.X[:, None, s0:s1, 0], out=rows)
                else:
                    for c0 in range(0, B, lp.context_tile):
                        c1 = min(c0 + lp.context_tile, B)
                        d = layer.D[c0:c1, f0:f1].contiguous()
                        x = layer.X[c0:c1, s0:s1].contiguous()
                        torch.bmm(d, x.transpose(1, 2), out=rows[c0:c1])
                        del d, x
                rows.mul_(wm)
                flat = rows.view(B, -1)
                gram.addmm_(flat, flat.T)
                del flat, rows, wm
    symmetric = gram + gram.T
    del gram
    symmetric.mul_(0.5)
    if not bool(torch.isfinite(symmetric).all()):
        raise ValueError('response Gram is nonfinite')
    return symmetric
