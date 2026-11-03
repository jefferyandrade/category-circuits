"""Ordinary per-cell CUDA kernels for one guarded Newton iteration.

The captured fit launches a fixed number of iterations. Each launch leaves
converged and failed cells frozen, and skips their row reductions. This is
the same projected Newton step and first-sufficient backtracking ladder as
the tensor implementation; no kernel owns the decision walk.
"""
from __future__ import annotations

from functools import lru_cache

import torch

# Bound the row vectors independently of the calibration population. A cell
# can visit any number of these tiles without enlarging a Triton block.
# Tile width: 2**16 rows. Every fit of at most 65,536 rows runs as one tile, exactly the pre-tiling single-block
# arithmetic, so ordinary closures are unchanged across seals; only population-scale fits (CLOSURE-CEIL) are tiled.
ROW_BLOCK = 65536

try:  # CUDA workers supply Triton; importing the CPU runtime needs none.
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:
    triton = None
    tl = None
    libdevice = None


if triton is not None:
    @triton.jit
    def maximum(a, b):
        return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)

    @triton.jit
    def objective(x, target, weight, scale, offset):
        eta = x * scale + offset
        softplus = tl.where(eta > 20.0, eta, libdevice.log1p(libdevice.exp(eta)))
        return tl.sum((softplus - target * eta) * weight, 0)

    @triton.jit
    def statistics(x, target, weight, scale, offset):
        eta = x * scale + offset
        probability = 1.0 / (1.0 + libdevice.exp(-eta))
        residual = (probability - target) * weight
        curvature = probability * (1.0 - probability) * weight
        loss = objective(x, target, weight, scale, offset)
        return (loss, tl.sum(residual * x, 0), tl.sum(residual, 0),
                tl.sum(curvature * (x * x), 0),
                tl.sum(curvature * x, 0), tl.sum(curvature, 0))

    @triton.jit
    def row_objective(X, T, W, scale, offset, N: tl.constexpr, BLOCK: tl.constexpr,
                      X1: tl.constexpr, T1: tl.constexpr, W1: tl.constexpr):
        loss = tl.full((), 0.0, tl.float64)
        for tile in range((N + BLOCK - 1) // BLOCK):
            row = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
            valid = row < N
            x = tl.load(X + row * X1, valid, 0.0).to(tl.float64)
            target = tl.load(T + row * T1, valid, 0.0).to(tl.float64)
            weight = tl.load(W + row * W1, valid, 0.0).to(tl.float64)
            loss += objective(x, target, weight, scale, offset)
        return loss / N

    @triton.jit
    def row_statistics(X, T, W, scale, offset, N: tl.constexpr, BLOCK: tl.constexpr,
                       X1: tl.constexpr, T1: tl.constexpr, W1: tl.constexpr):
        loss = tl.full((), 0.0, tl.float64)
        gs = tl.full((), 0.0, tl.float64)
        go = tl.full((), 0.0, tl.float64)
        hss = tl.full((), 0.0, tl.float64)
        hso = tl.full((), 0.0, tl.float64)
        hoo = tl.full((), 0.0, tl.float64)
        for tile in range((N + BLOCK - 1) // BLOCK):
            row = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
            valid = row < N
            x = tl.load(X + row * X1, valid, 0.0).to(tl.float64)
            target = tl.load(T + row * T1, valid, 0.0).to(tl.float64)
            weight = tl.load(W + row * W1, valid, 0.0).to(tl.float64)
            part_loss, part_gs, part_go, part_hss, part_hso, part_hoo = statistics(
                x, target, weight, scale, offset)
            loss += part_loss
            gs += part_gs
            go += part_go
            hss += part_hss
            hso += part_hso
            hoo += part_hoo
        # Normalize once by the original physical row count, including rows
        # whose weight is zero. Tail padding never changes that denominator.
        return loss / N, gs / N, go / N, hss / N, hso / N, hoo / N

    @triton.jit(do_not_specialize=["iteration"])
    def step(X, T, W, Constant, Scale, Offset, Minimum, Live,
             Converged, Failed, Counts, Steps, iteration,
             N: tl.constexpr, BLOCK: tl.constexpr, FIRST: tl.constexpr,
             X0: tl.constexpr, X1: tl.constexpr, T0: tl.constexpr, T1: tl.constexpr,
             W0: tl.constexpr, W1: tl.constexpr,
             TOLERANCE: tl.constexpr, RIDGE: tl.constexpr,
             FLOOR: tl.constexpr, SLACK: tl.constexpr,
             OBJECTIVE_TOLERANCE: tl.constexpr):
        cell = tl.program_id(0).to(tl.int64)
        converged = tl.load(Converged + cell)
        failed = tl.load(Failed + cell)
        live = tl.load(Live + cell)
        # Disabled cells keep their initial parameters. Their KKT result is
        # therefore constant, and only the first iteration needs to form it.
        if (~converged) & (~failed) & (live | FIRST):
            scale = tl.load(Scale + cell)
            offset = tl.load(Offset + cell)
            minimum = tl.load(Minimum + cell)
            constant = tl.load(Constant + cell)
            loss, gs, go, hss, hso, hoo = row_statistics(
                X + cell * X0, T + cell * T0, W + cell * W0,
                scale, offset, N, BLOCK, X1, T1, W1)
            loss = loss - constant
            boundary = scale <= minimum + SLACK
            scale_kkt = tl.where(boundary, maximum(-gs, 0.0), tl.abs(gs))
            kkt = maximum(scale_kkt, tl.abs(go))
            converged = kkt <= TOLERANCE
            tl.store(Converged + cell, converged)
            if live & (~converged):
                tl.store(Counts + cell, iteration)
                hss = hss + RIDGE
                hoo = hoo + RIDGE
                determinant = hss * hoo - hso * hso
                stable = determinant > FLOOR
                determinant = maximum(determinant, FLOOR)
                diagonal_o = go / maximum(hoo, RIDGE)
                step_s = tl.where(stable, (hoo * gs - hso * go) / determinant,
                                  gs / maximum(hss, RIDGE))
                step_o = tl.where(stable, (hss * go - hso * gs) / determinant, diagonal_o)
                bound = boundary & (gs >= 0.0)
                step_s = tl.where(bound, 0.0, step_s)
                step_o = tl.where(bound, diagonal_o, step_o)
                found = False
                chosen_scale, chosen_offset = scale, offset
                chosen_step = 0
                alpha = tl.full((), 1.0, tl.float64)
                # This bounded ladder evaluates only until its first
                # sufficient step. Later ladder entries cannot change it.
                for backtrack in range(30):
                    if not found:
                        proposed_scale = maximum(scale + alpha * (-step_s), minimum)
                        proposed_offset = offset + alpha * (-step_o)
                        proposed = row_objective(
                            X + cell * X0, T + cell * T0, W + cell * W0,
                            proposed_scale, proposed_offset, N, BLOCK, X1, T1, W1) - constant
                        directional = gs * (proposed_scale - scale) + go * (proposed_offset - offset)
                        sufficient = ((proposed <= loss + 1.0e-4 * directional + OBJECTIVE_TOLERANCE)
                                      & (tl.abs(proposed) < float("inf")))
                        if sufficient:
                            found = True
                            chosen_scale = proposed_scale
                            chosen_offset = proposed_offset
                            chosen_step = backtrack + 1
                    alpha = alpha * 0.5
                if found:
                    tl.store(Scale + cell, chosen_scale)
                    tl.store(Offset + cell, chosen_offset)
                    tl.store(Steps + cell, chosen_step)
                else:
                    tl.store(Failed + cell, True)

    @triton.jit
    def pending(X, T, W, Scale, Offset, Minimum, Live, Converged, Failed, Pending,
                N: tl.constexpr, BLOCK: tl.constexpr,
                X0: tl.constexpr, X1: tl.constexpr, T0: tl.constexpr, T1: tl.constexpr,
                W0: tl.constexpr, W1: tl.constexpr,
                TOLERANCE: tl.constexpr, SLACK: tl.constexpr):
        cell = tl.program_id(0).to(tl.int64)
        active = tl.load(Live + cell) & (~tl.load(Converged + cell)) & (~tl.load(Failed + cell))
        result = False
        if active:
            scale = tl.load(Scale + cell)
            offset = tl.load(Offset + cell)
            minimum = tl.load(Minimum + cell)
            _, gs, go, _, _, _ = row_statistics(
                X + cell * X0, T + cell * T0, W + cell * W0,
                scale, offset, N, BLOCK, X1, T1, W1)
            boundary = scale <= minimum + SLACK
            kkt = maximum(tl.where(boundary, maximum(-gs, 0.0), tl.abs(gs)), tl.abs(go))
            # Match the eager pending predicate for NaNs as well: it negates
            # `kkt <= tolerance`, rather than testing `kkt > tolerance`.
            result = ~(kkt <= TOLERANCE)
        tl.store(Pending + cell, result)

    @triton.jit
    def firth_statistics(X, T, W, Fixed, scale, offset, inverse_size, identifiable,
                         N: tl.constexpr, BLOCK: tl.constexpr,
                         X1: tl.constexpr, T1: tl.constexpr, W1: tl.constexpr,
                         F1: tl.constexpr, SCORE: tl.constexpr):
        loss = tl.full((), 0.0, tl.float64)
        gs = tl.full((), 0.0, tl.float64)
        go = tl.full((), 0.0, tl.float64)
        hss = tl.full((), 0.0, tl.float64)
        hso = tl.full((), 0.0, tl.float64)
        hoo = tl.full((), 0.0, tl.float64)
        for tile in range((N + BLOCK - 1) // BLOCK):
            row = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
            valid = row < N
            u = tl.load(X + row * X1, valid, 0.0).to(tl.float64)
            y = tl.load(T + row * T1, valid, 0.0).to(tl.float64)
            w = tl.load(W + row * W1, valid, 0.0).to(tl.float64)
            fixed = tl.load(Fixed + row * F1, valid, 0.0).to(tl.float64)
            eta = u * scale + offset + fixed
            p = 1.0 / (1.0 + libdevice.exp(-eta))
            q = 1.0 / (1.0 + libdevice.exp(eta))
            v = w * p * q
            softplus = tl.where(eta > 20.0, eta, libdevice.log1p(libdevice.exp(eta)))
            loss += tl.sum((softplus - y * eta) * w, 0)
            gs += tl.sum(w * (p - y) * u, 0)
            go += tl.sum(w * (p - y), 0)
            hss += tl.sum(v * u * u, 0)
            hso += tl.sum(v * u, 0)
            hoo += tl.sum(v, 0)
        determinant = tl.where(identifiable, hss * hoo - hso * hso, hoo)
        safe_det = maximum(determinant, 1.0e-300)
        loss = tl.where(determinant > 0.0,
                        loss - 0.5 * inverse_size * libdevice.log(safe_det), float("inf"))
        if SCORE:
            correction_s = tl.full((), 0.0, tl.float64)
            correction_o = tl.full((), 0.0, tl.float64)
            for tile in range((N + BLOCK - 1) // BLOCK):
                row = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
                valid = row < N
                u = tl.load(X + row * X1, valid, 0.0).to(tl.float64)
                w = tl.load(W + row * W1, valid, 0.0).to(tl.float64)
                fixed = tl.load(Fixed + row * F1, valid, 0.0).to(tl.float64)
                eta = u * scale + offset + fixed
                p = 1.0 / (1.0 + libdevice.exp(-eta))
                q = 1.0 / (1.0 + libdevice.exp(eta))
                leverage = w * p * q * tl.where(identifiable,
                    (hoo * u * u - 2.0 * hso * u + hss) / safe_det,
                    1.0 / maximum(hoo, 1.0e-300))
                adjustment = leverage * (0.5 - p)
                correction_s += tl.sum(adjustment * u, 0)
                correction_o += tl.sum(adjustment, 0)
            gs = tl.where(identifiable, gs - inverse_size * correction_s, 0.0)
            go = go - inverse_size * correction_o
        return loss, gs, go, hss, hso, hoo

    @triton.jit(do_not_specialize=["iteration"])
    def firth_step(X, T, W, Fixed, InverseSize, Identifiable, Scale, Offset, Minimum, Live,
                   Converged, Failed, Counts, Steps, iteration,
                   N: tl.constexpr, BLOCK: tl.constexpr,
                   X0: tl.constexpr, X1: tl.constexpr, T0: tl.constexpr, T1: tl.constexpr,
                   W0: tl.constexpr, W1: tl.constexpr, F0: tl.constexpr, F1: tl.constexpr,
                   TOLERANCE: tl.constexpr, RIDGE: tl.constexpr,
                   FLOOR: tl.constexpr, SLACK: tl.constexpr,
                   OBJECTIVE_TOLERANCE: tl.constexpr,
                   FISHER_STANDARDIZED: tl.constexpr):
        cell = tl.program_id(0).to(tl.int64)
        live = tl.load(Live + cell)
        if live & (~tl.load(Converged + cell)) & (~tl.load(Failed + cell)):
            scale, offset = tl.load(Scale + cell), tl.load(Offset + cell)
            minimum = tl.load(Minimum + cell)
            inverse_size = tl.load(InverseSize + cell)
            identifiable = tl.load(Identifiable + cell)
            loss, gs, go, hss, hso, hoo = firth_statistics(
                X + cell * X0, T + cell * T0, W + cell * W0, Fixed + cell * F0,
                scale, offset, inverse_size, identifiable, N, BLOCK, X1, T1, W1, F1, True)
            boundary = scale <= minimum + SLACK
            if FISHER_STANDARDIZED:
                # Match _firth_projected_kkt before adding the scoring ridge:
                # normalize the projected score by the weighted Fisher scale.
                scale_score = tl.where(boundary, maximum(-gs, 0.0), tl.abs(gs))
                scale_score = scale_score / libdevice.sqrt(maximum(hss * hoo, 1.0e-300))
                offset_score = tl.abs(go) / maximum(hoo, 1.0e-300)
                kkt = maximum(scale_score, offset_score)
            else:
                kkt = maximum(tl.where(boundary, maximum(-gs, 0.0), tl.abs(gs)), tl.abs(go))
            converged = kkt <= TOLERANCE
            tl.store(Converged + cell, converged)
            if not converged:
                tl.store(Counts + cell, iteration)
                hss, hoo = hss + RIDGE, hoo + RIDGE
                determinant = maximum(hss * hoo - hso * hso, FLOOR)
                ds = (hoo * gs - hso * go) / determinant
                do = (hss * go - hso * gs) / determinant
                bound = (boundary & (gs >= 0.0)) | (~identifiable)
                ds = tl.where(bound, 0.0, ds)
                do = tl.where(bound, go / maximum(hoo, maximum(RIDGE, 1.0e-30)), do)
                found = False
                chosen_scale, chosen_offset = scale, offset
                chosen_step = 0
                alpha = tl.full((), 1.0, tl.float64)
                for backtrack in range(30):
                    if not found:
                        proposed_scale = maximum(scale - alpha * ds, minimum)
                        proposed_offset = offset - alpha * do
                        proposed, _, _, _, _, _ = firth_statistics(
                            X + cell * X0, T + cell * T0, W + cell * W0, Fixed + cell * F0,
                            proposed_scale, proposed_offset, inverse_size, identifiable,
                            N, BLOCK, X1, T1, W1, F1, False)
                        directional = gs * (proposed_scale - scale) + go * (proposed_offset - offset)
                        sufficient = ((proposed <= loss + 1.0e-4 * directional + OBJECTIVE_TOLERANCE)
                                      & (tl.abs(proposed) < float("inf")))
                        if sufficient:
                            found = True
                            chosen_scale, chosen_offset = proposed_scale, proposed_offset
                            chosen_step = backtrack + 1
                    alpha = alpha * 0.5
                if found:
                    tl.store(Scale + cell, chosen_scale)
                    tl.store(Offset + cell, chosen_offset)
                    tl.store(Steps + cell, chosen_step)
                else:
                    tl.store(Failed + cell, True)



@lru_cache(maxsize=1)
def _kernels():
    if triton is None:
        raise RuntimeError("CUDA calibration Newton steps require Triton")
    return step, pending


def guarded_newton(x, t, w, constant, scale, offset, minimum, live, *, iterations,
                   tolerance, hessian_ridge, objective_tolerance, start=0):
    """Fixed launches of per-cell Newton steps, with no device-to-host reads."""
    if x.device.type != "cuda":
        raise ValueError("the CUDA Newton step requires CUDA tensors")
    if x.ndim != 2 or x.shape[1] <= 0 or t.shape != x.shape or w.shape != x.shape:
        raise ValueError("Newton rows must be aligned [cells, rows] matrices")
    cells, rows = map(int, x.shape)
    arguments = (x,t,w,constant,scale,offset,minimum,live)
    if any(value.device != x.device for value in arguments):
        raise ValueError("Newton tensors must share the CUDA device")
    if any(value.dtype != torch.float64 for value in arguments[:-1]):
        raise ValueError("Newton numerical tensors must use float64")
    if any(value.shape != (cells,) for value in arguments[3:]):
        raise ValueError("Newton parameters must have one value per cell")
    if live.dtype != torch.bool:
        raise ValueError("Newton live cells must be boolean")
    step, pending_kernel = _kernels()
    constant,minimum,live = constant.contiguous(),minimum.contiguous(),live.contiguous()
    scale,offset = scale.clone().contiguous(),offset.clone().contiguous()
    converged,failed = torch.zeros_like(live),torch.zeros_like(live)
    counts = torch.zeros((cells,),dtype=torch.int64,device=x.device)
    steps = torch.zeros_like(counts)
    pending = torch.empty_like(live)
    ridge = float(hessian_ridge)
    block = min(ROW_BLOCK, triton.next_power_of_2(rows))
    options = dict(N=rows,BLOCK=block,X0=x.stride(0),X1=x.stride(1),
                   T0=t.stride(0),T1=t.stride(1),W0=w.stride(0),W1=w.stride(1),
                   TOLERANCE=float(tolerance),
                   SLACK=max(float(tolerance),1e-12),num_warps=4 if block <= 2048 else 8,
                   enable_fp_fusion=False)
    for iteration in range(int(start)+1,int(start)+int(iterations)+1):
        step[(cells,)](x,t,w,constant,scale,offset,minimum,live,
            converged,failed,counts,steps,iteration,FIRST=iteration == int(start)+1,
            RIDGE=ridge,FLOOR=max(ridge*ridge,1e-30),
            OBJECTIVE_TOLERANCE=float(objective_tolerance),**options)
    pending_kernel[(cells,)](x,t,w,scale,offset,minimum,live,converged,failed,pending,**options)
    return scale,offset,converged,failed,counts,steps,pending


def guarded_firth_newton(x, y, w, fixed_offsets, inverse_size, identifiable,
                         scale, offset, minimum, live, *, iterations,
                         tolerance, hessian_ridge, objective_tolerance,
                         convergence="absolute"):
    """Firth scoring on CUDA with device-only absolute or Fisher KKT tests.

    Both modes skip frozen cells and unused backtracks; a fixed launch bound
    permits graph capture without reading convergence decisions to the host.
    """
    if convergence not in {"absolute", "fisher_standardized"}:
        raise ValueError(f"unsupported Firth convergence test {convergence!r}")
    if triton is None:
        raise RuntimeError("CUDA Firth calibration requires Triton")
    if x.ndim != 2:
        raise ValueError("Firth CUDA rows must have shape [cells, rows]")
    cells, rows = map(int, x.shape)
    y = y.expand_as(x)
    w = w.expand_as(x)
    fixed = (torch.zeros((1, 1), dtype=x.dtype, device=x.device).expand_as(x)
             if fixed_offsets is None else fixed_offsets.expand_as(x))
    scale, offset = scale.clone().contiguous(), offset.clone().contiguous()
    inverse_size, identifiable = inverse_size.contiguous(), identifiable.contiguous()
    minimum, live = minimum.contiguous(), live.contiguous()
    converged, failed = torch.zeros_like(live), torch.zeros_like(live)
    counts = torch.zeros((cells,), dtype=torch.int64, device=x.device)
    steps = torch.zeros_like(counts)
    block = min(ROW_BLOCK, triton.next_power_of_2(rows))
    options = dict(N=rows, BLOCK=block, X0=x.stride(0), X1=x.stride(1),
        T0=y.stride(0), T1=y.stride(1), W0=w.stride(0), W1=w.stride(1),
        F0=fixed.stride(0), F1=fixed.stride(1), TOLERANCE=float(tolerance),
        RIDGE=float(hessian_ridge), FLOOR=max(float(hessian_ridge) ** 2, 1e-30),
        SLACK=max(float(tolerance), 1e-12), OBJECTIVE_TOLERANCE=float(objective_tolerance),
        FISHER_STANDARDIZED=convergence == "fisher_standardized",
        num_warps=4 if block <= 2048 else 8, enable_fp_fusion=False)
    for iteration in range(1, max(1, int(iterations)) + 1):
        firth_step[(cells,)](x, y, w, fixed, inverse_size, identifiable, scale, offset,
            minimum, live, converged, failed, counts, steps, iteration, **options)
    return scale, offset, converged, failed, counts, steps


__all__ = ["guarded_newton", "guarded_firth_newton"]
