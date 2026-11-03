"""Fused census-consumer kernel: heads, pullback, moments in one pass.

The census fused backward, consumer half
(COMPILATION_FUSION_KERNEL_SURVEY_20260811 Tier 1, re-scoped). The
capture backend still produces per-block raw sensitivity rows; this
module fuses EVERYTHING downstream of capture for one route chunk —

    head formation   l = local_raw * dest_scale * gate
                     g = global_raw * affine_scale * gate
                     s = (p - y) * g                     (loss pullback)
    moment update    sum_rows, sum_outer, sum_outer_squared,
                     view activity (+ squares)
    sketch epilogue  rank-K FrequentDirections projections (v1 form;
                     K is a compile-time constant, K=0 for v0)

— into a single Triton kernel that keeps every intermediate in
registers and writes only the persistent accumulator slabs. No
[contexts x routes] head tensor is ever materialized, and the ~10-kernel
per-chunk launch chain collapses to one.

Determinism: the kernel reduces contexts in a SEQUENTIAL in-kernel loop
(a defined order, identical for every route lane). The einsum path's
BLAS tilings are shape-dependent, so kernel-vs-einsum agreement is
reduction-order-equivalent (fp64 allclose), not bitwise — the same
relation the streamed protocol already documents for the sketch. The
CPU reference below mirrors the kernel's exact order and is the parity
authority for the GPU test.

The fused kernel is the production route-moment path.
"""

# Triton kernel locals use the conventional mathematical head name ``l``.
# ruff: noqa: E741

from __future__ import annotations

from typing import Any

import numpy as np

try:  # pragma: no cover - exercised only where torch exists
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - triton exists only on CUDA builds
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


def fused_kernel_available() -> bool:
    return torch is not None and triton is not None and torch.cuda.is_available()


if triton is not None:

    @triton.jit
    def _add_owned(pointer, value, mask):
        # Every route coordinate belongs to one program. Chunk launches are
        # ordered on the stream, so these updates need no floating atomics.
        prior = tl.load(pointer, mask=mask, other=0.0)
        tl.store(pointer, prior + value, mask=mask)

    @triton.jit
    def _fused_route_moment_kernel(
        local_raw_ptr,  # [n, r] fp64 raw local rows
        global_raw_ptr,  # [n, r] fp64 raw global rows
        dest_scale_ptr,  # [r] fp64 destination scale
        gate_ptr,  # [r] fp64 active-route gate (0/1)
        pullback_ptr,  # [n] fp64 (p - y)
        weight_ptr,  # [n] fp64 context weights
        sign_ptr,  # [K, n] fp64 sketch signs (unused when K == 0)
        sum_rows_ptr,  # [3, R_total] fp64 (slab base, route offset applied)
        sum_outer_ptr,  # [R_total, 3, 3] fp64
        sum_outer_sq_ptr,  # [R_total, 3, 3] fp64
        activity_ptr,  # [3, R_total] fp64
        activity_sq_ptr,  # [3, R_total] fp64
        sketch_ptr,  # [K, 3, R_total] fp64
        per_image_ptr,  # [n, 3, SCORE_BLOCKS] fp64 uniquely owned partials
        finite_flag_ptr,  # [1] int32, 1 == all rows finite (epilogue)
        n_contexts,
        r_chunk,
        route_offset,
        route_total,
        affine_scale_ptr,  # [1] fp64 — scalar args type as fp32 in Triton
        BLOCK_R: tl.constexpr,
        SKETCH_RANK: tl.constexpr,
        SCORE_EPILOGUE: tl.constexpr,
        SCORE_BLOCKS: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        lane = pid * BLOCK_R + tl.arange(0, BLOCK_R)
        in_chunk = lane < r_chunk
        route = route_offset + lane

        affine_scale = tl.load(affine_scale_ptr)
        dest = tl.load(dest_scale_ptr + lane, mask=in_chunk, other=0.0)
        gate = tl.load(gate_ptr + lane, mask=in_chunk, other=0.0)

        s_l = tl.zeros([BLOCK_R], dtype=tl.float64)
        s_g = tl.zeros([BLOCK_R], dtype=tl.float64)
        s_s = tl.zeros([BLOCK_R], dtype=tl.float64)
        o_ll = tl.zeros([BLOCK_R], dtype=tl.float64)
        o_lg = tl.zeros([BLOCK_R], dtype=tl.float64)
        o_ls = tl.zeros([BLOCK_R], dtype=tl.float64)
        o_gg = tl.zeros([BLOCK_R], dtype=tl.float64)
        o_gs = tl.zeros([BLOCK_R], dtype=tl.float64)
        o_ss = tl.zeros([BLOCK_R], dtype=tl.float64)
        q_ll = tl.zeros([BLOCK_R], dtype=tl.float64)
        q_lg = tl.zeros([BLOCK_R], dtype=tl.float64)
        q_ls = tl.zeros([BLOCK_R], dtype=tl.float64)
        q_gg = tl.zeros([BLOCK_R], dtype=tl.float64)
        q_gs = tl.zeros([BLOCK_R], dtype=tl.float64)
        q_ss = tl.zeros([BLOCK_R], dtype=tl.float64)
        a_l = tl.zeros([BLOCK_R], dtype=tl.float64)
        a_g = tl.zeros([BLOCK_R], dtype=tl.float64)
        a_s = tl.zeros([BLOCK_R], dtype=tl.float64)
        a2_l = tl.zeros([BLOCK_R], dtype=tl.float64)
        a2_g = tl.zeros([BLOCK_R], dtype=tl.float64)
        a2_s = tl.zeros([BLOCK_R], dtype=tl.float64)
        bad_total = 0.0

        for context in range(0, n_contexts):
            w = tl.load(weight_ptr + context)
            pb = tl.load(pullback_ptr + context)
            base = context * r_chunk
            lraw = tl.load(local_raw_ptr + base + lane, mask=in_chunk, other=0.0)
            graw = tl.load(global_raw_ptr + base + lane, mask=in_chunk, other=0.0)
            l = lraw * dest * gate
            g = graw * affine_scale * gate
            s = pb * g

            if SCORE_EPILOGUE > 0:
                # Per-image (per-context) head sums over this chunk's routes;
                # masked lanes carry exact zeros (dest/gate load as 0.0).
                tl.store(per_image_ptr + (context * 3 + 0) * SCORE_BLOCKS + pid, tl.sum(l))
                tl.store(per_image_ptr + (context * 3 + 1) * SCORE_BLOCKS + pid, tl.sum(g))
                tl.store(per_image_ptr + (context * 3 + 2) * SCORE_BLOCKS + pid, tl.sum(s))
                bad_total += tl.sum(
                    tl.where(
                        in_chunk
                        & (
                            (lraw != lraw)
                            | (tl.abs(lraw) == float("inf"))
                            | (graw != graw)
                            | (tl.abs(graw) == float("inf"))
                        ),
                        1.0,
                        0.0,
                    )
                )

            s_l += w * l
            s_g += w * g
            s_s += w * s
            o_ll += w * l * l
            o_lg += w * l * g
            o_ls += w * l * s
            o_gg += w * g * g
            o_gs += w * g * s
            o_ss += w * s * s
            l2 = l * l
            g2 = g * g
            s2 = s * s
            q_ll += w * l2 * l2
            q_lg += w * l2 * g2
            q_ls += w * l2 * s2
            q_gg += w * g2 * g2
            q_gs += w * g2 * s2
            q_ss += w * s2 * s2
            a_l += w * tl.abs(l)
            a_g += w * tl.abs(g)
            a_s += w * tl.abs(s)
            a2_l += w * l * l
            a2_g += w * g * g
            a2_s += w * s * s
            if SKETCH_RANK > 0:
                root = tl.sqrt(w)
                for probe in range(0, SKETCH_RANK):
                    sign = tl.load(sign_ptr + probe * n_contexts + context)
                    coefficient = sign * root
                    _add_owned(
                        sketch_ptr + probe * 3 * route_total + 0 * route_total + route,
                        coefficient * l,
                        mask=in_chunk,
                    )
                    _add_owned(
                        sketch_ptr + probe * 3 * route_total + 1 * route_total + route,
                        coefficient * g,
                        mask=in_chunk,
                    )
                    _add_owned(
                        sketch_ptr + probe * 3 * route_total + 2 * route_total + route,
                        coefficient * s,
                        mask=in_chunk,
                    )

        _add_owned(sum_rows_ptr + 0 * route_total + route, s_l, mask=in_chunk)
        _add_owned(sum_rows_ptr + 1 * route_total + route, s_g, mask=in_chunk)
        _add_owned(sum_rows_ptr + 2 * route_total + route, s_s, mask=in_chunk)

        outer_base = route * 9
        _add_owned(sum_outer_ptr + outer_base + 0, o_ll, mask=in_chunk)
        _add_owned(sum_outer_ptr + outer_base + 1, o_lg, mask=in_chunk)
        _add_owned(sum_outer_ptr + outer_base + 2, o_ls, mask=in_chunk)
        _add_owned(sum_outer_ptr + outer_base + 3, o_lg, mask=in_chunk)
        _add_owned(sum_outer_ptr + outer_base + 4, o_gg, mask=in_chunk)
        _add_owned(sum_outer_ptr + outer_base + 5, o_gs, mask=in_chunk)
        _add_owned(sum_outer_ptr + outer_base + 6, o_ls, mask=in_chunk)
        _add_owned(sum_outer_ptr + outer_base + 7, o_gs, mask=in_chunk)
        _add_owned(sum_outer_ptr + outer_base + 8, o_ss, mask=in_chunk)

        _add_owned(sum_outer_sq_ptr + outer_base + 0, q_ll, mask=in_chunk)
        _add_owned(sum_outer_sq_ptr + outer_base + 1, q_lg, mask=in_chunk)
        _add_owned(sum_outer_sq_ptr + outer_base + 2, q_ls, mask=in_chunk)
        _add_owned(sum_outer_sq_ptr + outer_base + 3, q_lg, mask=in_chunk)
        _add_owned(sum_outer_sq_ptr + outer_base + 4, q_gg, mask=in_chunk)
        _add_owned(sum_outer_sq_ptr + outer_base + 5, q_gs, mask=in_chunk)
        _add_owned(sum_outer_sq_ptr + outer_base + 6, q_ls, mask=in_chunk)
        _add_owned(sum_outer_sq_ptr + outer_base + 7, q_gs, mask=in_chunk)
        _add_owned(sum_outer_sq_ptr + outer_base + 8, q_ss, mask=in_chunk)

        _add_owned(activity_ptr + 0 * route_total + route, a_l, mask=in_chunk)
        _add_owned(activity_ptr + 1 * route_total + route, a_g, mask=in_chunk)
        _add_owned(activity_ptr + 2 * route_total + route, a_s, mask=in_chunk)
        _add_owned(activity_sq_ptr + 0 * route_total + route, a2_l, mask=in_chunk)
        _add_owned(activity_sq_ptr + 1 * route_total + route, a2_g, mask=in_chunk)
        _add_owned(activity_sq_ptr + 2 * route_total + route, a2_s, mask=in_chunk)

        if SCORE_EPILOGUE > 0:
            # Deferred finiteness verdict: no host sync here; the packet
            # finalizer reads the flag exactly once.
            tl.atomic_xchg(finite_flag_ptr, 0, mask=bad_total > 0.0)

    @triton.jit
    def _census_linear_inbackward_kernel(
        inputs_ptr,  # [n, I] fp32 layer inputs (the backward's saved input)
        grads_ptr,  # [n, o_chunk] fp32 target-gradient slice
        layer_weight_ptr,  # [o_chunk, I] fp32 layer-weight slice (row-contiguous)
        dest_scale_ptr,  # [r_chunk] fp64 destination scale (chunk-relative)
        gate_ptr,  # [r_chunk] fp64 active-route gate (0/1)
        pullback_ptr,  # [n] fp64 (p - y)
        ctx_weight_ptr,  # [n] fp64 context weights (unit on the census path)
        sign_ptr,  # [K, n] fp64 sketch signs (unused when K == 0)
        sum_rows_ptr,  # [3, R_total] fp64 (slab base, route offset applied)
        sum_outer_ptr,  # [R_total, 3, 3] fp64
        sum_outer_sq_ptr,  # [R_total, 3, 3] fp64
        activity_ptr,  # [3, R_total] fp64
        activity_sq_ptr,  # [3, R_total] fp64
        sketch_ptr,  # [K, 3, R_total] fp64
        n_contexts,
        r_chunk,
        route_offset,
        route_total,
        in_features,
        o_chunk,
        producer_scale_ptr,  # [1] fp32 — the producer's fp32 layer scale
        affine_scale_ptr,  # [1] fp64
        BLOCK_R: tl.constexpr,
        SKETCH_RANK: tl.constexpr,
    ):
        """Full in-backward census form: rows live only in registers.

        The graded form's producer computes fp32 row chunks
        ``((grad/scale)*input)*weight`` and streams them; this kernel
        replays those EXACT fp32 operations per (context, route) lane —
        route = o*I + i over the chunk's flattened [o_chunk, I] axis —
        widens to fp64, forms the tripartite heads in the SAME order as
        ``_fused_route_moment_kernel`` (l = raw_local*dest*gate,
        g = raw_global*affine*gate, s = pullback*g), and accumulates every
        moment slab in the same sequential context order.  The composed
        CPU authority is therefore ``census_inbackward_reference`` ==
        fp32 row mirror + ``fused_reference_update`` — the established
        parity pattern.  The [n, o_chunk, I] row tensor NEVER exists.
        """

        pid = tl.program_id(axis=0)
        lane = pid * BLOCK_R + tl.arange(0, BLOCK_R)
        in_chunk = lane < r_chunk
        route = route_offset + lane
        o_lane = lane // in_features
        i_lane = lane - o_lane * in_features

        producer_scale = tl.load(producer_scale_ptr)
        affine_scale = tl.load(affine_scale_ptr)
        dest = tl.load(dest_scale_ptr + lane, mask=in_chunk, other=0.0)
        gate = tl.load(gate_ptr + lane, mask=in_chunk, other=0.0)
        w32 = tl.load(layer_weight_ptr + lane, mask=in_chunk, other=0.0)

        s_l = tl.zeros([BLOCK_R], dtype=tl.float64)
        s_g = tl.zeros([BLOCK_R], dtype=tl.float64)
        s_s = tl.zeros([BLOCK_R], dtype=tl.float64)
        o_ll = tl.zeros([BLOCK_R], dtype=tl.float64)
        o_lg = tl.zeros([BLOCK_R], dtype=tl.float64)
        o_ls = tl.zeros([BLOCK_R], dtype=tl.float64)
        o_gg = tl.zeros([BLOCK_R], dtype=tl.float64)
        o_gs = tl.zeros([BLOCK_R], dtype=tl.float64)
        o_ss = tl.zeros([BLOCK_R], dtype=tl.float64)
        q_ll = tl.zeros([BLOCK_R], dtype=tl.float64)
        q_lg = tl.zeros([BLOCK_R], dtype=tl.float64)
        q_ls = tl.zeros([BLOCK_R], dtype=tl.float64)
        q_gg = tl.zeros([BLOCK_R], dtype=tl.float64)
        q_gs = tl.zeros([BLOCK_R], dtype=tl.float64)
        q_ss = tl.zeros([BLOCK_R], dtype=tl.float64)
        a_l = tl.zeros([BLOCK_R], dtype=tl.float64)
        a_g = tl.zeros([BLOCK_R], dtype=tl.float64)
        a_s = tl.zeros([BLOCK_R], dtype=tl.float64)
        a2_l = tl.zeros([BLOCK_R], dtype=tl.float64)
        a2_g = tl.zeros([BLOCK_R], dtype=tl.float64)
        a2_s = tl.zeros([BLOCK_R], dtype=tl.float64)

        for context in range(0, n_contexts):
            w = tl.load(ctx_weight_ptr + context)
            pb = tl.load(pullback_ptr + context)
            x32 = tl.load(
                inputs_ptr + context * in_features + i_lane,
                mask=in_chunk,
                other=0.0,
            )
            g32 = tl.load(
                grads_ptr + context * o_chunk + o_lane,
                mask=in_chunk,
                other=0.0,
            )
            # The graded producer's exact fp32 operation order:
            #   rows  = ((grad / scale) * input) * weight
            #   local = (input * weight) * scale
            r32 = ((g32 / producer_scale) * x32) * w32
            l32 = (x32 * w32) * producer_scale
            graw = r32.to(tl.float64)
            lraw = l32.to(tl.float64)
            l = lraw * dest * gate
            g = graw * affine_scale * gate
            s = pb * g

            s_l += w * l
            s_g += w * g
            s_s += w * s
            o_ll += w * l * l
            o_lg += w * l * g
            o_ls += w * l * s
            o_gg += w * g * g
            o_gs += w * g * s
            o_ss += w * s * s
            l2 = l * l
            g2 = g * g
            s2 = s * s
            q_ll += w * l2 * l2
            q_lg += w * l2 * g2
            q_ls += w * l2 * s2
            q_gg += w * g2 * g2
            q_gs += w * g2 * s2
            q_ss += w * s2 * s2
            a_l += w * tl.abs(l)
            a_g += w * tl.abs(g)
            a_s += w * tl.abs(s)
            a2_l += w * l * l
            a2_g += w * g * g
            a2_s += w * s * s
            if SKETCH_RANK > 0:
                root = tl.sqrt(w)
                for probe in range(0, SKETCH_RANK):
                    sign = tl.load(sign_ptr + probe * n_contexts + context)
                    coefficient = sign * root
                    _add_owned(
                        sketch_ptr + probe * 3 * route_total + 0 * route_total + route,
                        coefficient * l,
                        mask=in_chunk,
                    )
                    _add_owned(
                        sketch_ptr + probe * 3 * route_total + 1 * route_total + route,
                        coefficient * g,
                        mask=in_chunk,
                    )
                    _add_owned(
                        sketch_ptr + probe * 3 * route_total + 2 * route_total + route,
                        coefficient * s,
                        mask=in_chunk,
                    )

        _add_owned(sum_rows_ptr + 0 * route_total + route, s_l, mask=in_chunk)
        _add_owned(sum_rows_ptr + 1 * route_total + route, s_g, mask=in_chunk)
        _add_owned(sum_rows_ptr + 2 * route_total + route, s_s, mask=in_chunk)

        outer_base = route * 9
        _add_owned(sum_outer_ptr + outer_base + 0, o_ll, mask=in_chunk)
        _add_owned(sum_outer_ptr + outer_base + 1, o_lg, mask=in_chunk)
        _add_owned(sum_outer_ptr + outer_base + 2, o_ls, mask=in_chunk)
        _add_owned(sum_outer_ptr + outer_base + 3, o_lg, mask=in_chunk)
        _add_owned(sum_outer_ptr + outer_base + 4, o_gg, mask=in_chunk)
        _add_owned(sum_outer_ptr + outer_base + 5, o_gs, mask=in_chunk)
        _add_owned(sum_outer_ptr + outer_base + 6, o_ls, mask=in_chunk)
        _add_owned(sum_outer_ptr + outer_base + 7, o_gs, mask=in_chunk)
        _add_owned(sum_outer_ptr + outer_base + 8, o_ss, mask=in_chunk)

        _add_owned(sum_outer_sq_ptr + outer_base + 0, q_ll, mask=in_chunk)
        _add_owned(sum_outer_sq_ptr + outer_base + 1, q_lg, mask=in_chunk)
        _add_owned(sum_outer_sq_ptr + outer_base + 2, q_ls, mask=in_chunk)
        _add_owned(sum_outer_sq_ptr + outer_base + 3, q_lg, mask=in_chunk)
        _add_owned(sum_outer_sq_ptr + outer_base + 4, q_gg, mask=in_chunk)
        _add_owned(sum_outer_sq_ptr + outer_base + 5, q_gs, mask=in_chunk)
        _add_owned(sum_outer_sq_ptr + outer_base + 6, q_ls, mask=in_chunk)
        _add_owned(sum_outer_sq_ptr + outer_base + 7, q_gs, mask=in_chunk)
        _add_owned(sum_outer_sq_ptr + outer_base + 8, q_ss, mask=in_chunk)

        _add_owned(activity_ptr + 0 * route_total + route, a_l, mask=in_chunk)
        _add_owned(activity_ptr + 1 * route_total + route, a_g, mask=in_chunk)
        _add_owned(activity_ptr + 2 * route_total + route, a_s, mask=in_chunk)
        _add_owned(activity_sq_ptr + 0 * route_total + route, a2_l, mask=in_chunk)
        _add_owned(activity_sq_ptr + 1 * route_total + route, a2_g, mask=in_chunk)
        _add_owned(activity_sq_ptr + 2 * route_total + route, a2_s, mask=in_chunk)


def fused_reference_update(
    *,
    local_raw: np.ndarray,
    global_raw: np.ndarray,
    dest_scale: np.ndarray,
    gate: np.ndarray,
    pullback: np.ndarray,
    weights: np.ndarray,
    affine_scale: float,
    signs: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """CPU mirror of the kernel's exact arithmetic and context order.

    The parity authority for the GPU test, and the local validation
    surface. Head semantics are identical to the legacy consumer path:
    l = local_raw*dest*gate, g = global_raw*scale*gate, s = (p-y)*g;
    the middle activity view reduces to |g| for the single-query head
    layout (heads = local, global, loss).
    """

    n, r = local_raw.shape
    heads = np.empty((n, 3, r), dtype=np.float64)
    heads[:, 0, :] = local_raw * dest_scale[None, :] * gate[None, :]
    heads[:, 1, :] = global_raw * float(affine_scale) * gate[None, :]
    heads[:, 2, :] = pullback[:, None] * heads[:, 1, :]

    sum_rows = np.zeros((3, r))
    sum_outer = np.zeros((r, 3, 3))
    sum_outer_sq = np.zeros((r, 3, 3))
    activity = np.zeros((3, r))
    activity_sq = np.zeros((3, r))
    rank = 0 if signs is None else signs.shape[0]
    sketch = np.zeros((rank, 3, r))
    for context in range(n):
        w = float(weights[context])
        row = heads[context]
        sum_rows += w * row
        sum_outer += w * np.einsum("hr,jr->rhj", row, row)
        squared = row * row
        sum_outer_sq += w * np.einsum("hr,jr->rhj", squared, squared)
        view = np.stack((np.abs(row[0]), np.abs(row[1]), np.abs(row[2])))
        activity += w * view
        activity_sq += w * view * view
        if rank:
            root = np.sqrt(w)
            for probe in range(rank):
                sketch[probe] += signs[probe, context] * root * row
    return {
        "sum_rows": sum_rows,
        "sum_outer": sum_outer,
        "sum_outer_squared": sum_outer_sq,
        "sum_view_activity": activity,
        "sum_view_activity_squared": activity_sq,
        "sketch_sum": sketch,
        # Score-epilogue mirror: UNWEIGHTED per-context head sums over this
        # chunk's routes, plus the raw-row finiteness verdict the epilogue
        # defers to packet finalization.
        "per_image_scores": np.einsum("nhr->nh", heads),
        "row_finite": bool(np.isfinite(local_raw).all() and np.isfinite(global_raw).all()),
    }


def census_inbackward_raw_rows(
    *,
    inputs: np.ndarray,
    grads: np.ndarray,
    layer_weight: np.ndarray,
    producer_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """CPU mirror of the in-backward kernel's fp32 row formation.

    Per (context, o, i) element, in IEEE fp32 with the graded producer's
    exact operation order (pure multiply/divide chains — no reductions,
    so element values are order-free):

        raw_global = ((grad / scale) * input) * weight
        raw_local  = (input * weight) * scale

    Returns the flattened ``[n, o*I]`` raw rows widened exactly to fp64 —
    the inputs ``fused_reference_update`` (the shared sequential-order
    authority) consumes.
    """

    x = np.asarray(inputs, dtype=np.float32)
    g = np.asarray(grads, dtype=np.float32)
    w = np.asarray(layer_weight, dtype=np.float32)
    if x.ndim != 2 or g.ndim != 2 or w.ndim != 2:
        raise ValueError("in-backward rows require 2-D inputs/grads/weight")
    scale = np.float32(producer_scale)
    raw_global = ((g / scale)[:, :, None] * x[:, None, :]) * w[None, :, :]
    raw_local = (x[:, None, :] * w[None, :, :]) * scale
    contexts = int(x.shape[0])
    return (
        raw_local.reshape(contexts, -1).astype(np.float64),
        raw_global.reshape(contexts, -1).astype(np.float64),
    )


def census_inbackward_reference(
    *,
    inputs: np.ndarray,
    grads: np.ndarray,
    layer_weight: np.ndarray,
    dest_scale: np.ndarray,
    gate: np.ndarray,
    pullback: np.ndarray,
    weights: np.ndarray,
    producer_scale: float,
    affine_scale: float,
    signs: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """CPU parity authority for ``_census_linear_inbackward_kernel``.

    Composes the fp32 row mirror with ``fused_reference_update`` — the
    same sequential context order and fp64 arithmetic the kernel runs —
    so the GPU test gates on EXACT equality against these numbers, the
    established pattern for this surface.
    """

    raw_local, raw_global = census_inbackward_raw_rows(
        inputs=inputs,
        grads=grads,
        layer_weight=layer_weight,
        producer_scale=producer_scale,
    )
    return fused_reference_update(
        local_raw=raw_local,
        global_raw=raw_global,
        dest_scale=np.asarray(dest_scale, dtype=np.float64),
        gate=np.asarray(gate, dtype=np.float64),
        pullback=np.asarray(pullback, dtype=np.float64),
        weights=np.asarray(weights, dtype=np.float64),
        affine_scale=float(affine_scale),
        signs=signs,
    )


def _torch_inbackward_raw_rows(
    inputs: Any,
    grads: Any,
    layer_weight: Any,
    producer_scale: float,
) -> tuple[Any, Any]:
    """Torch twin of :func:`census_inbackward_raw_rows` (any device).

    Elementwise IEEE fp32 multiply/divide chains — value-identical to the
    numpy mirror and to the kernel's in-register formation.  Used by the
    graded-degradation fallback, which materializes ONE bounded chunk of
    raw rows and hands them to the established chunk-update entry.
    """

    x = inputs.to(dtype=torch.float32)
    g = grads.to(dtype=torch.float32)
    w = layer_weight.to(dtype=torch.float32)
    scale = torch.tensor(float(producer_scale), dtype=torch.float32, device=x.device)
    raw_global = ((g / scale)[:, :, None] * x[:, None, :]) * w[None, :, :]
    raw_local = (x[:, None, :] * w[None, :, :]) * scale
    contexts = int(x.shape[0])
    return (
        raw_local.reshape(contexts, -1).to(dtype=torch.float64),
        raw_global.reshape(contexts, -1).to(dtype=torch.float64),
    )


class ScoreEpilogueState:
    """Per-packet device-side score epilogue (IRG_FUSED_SCORE_EPILOGUE).

    Accumulates, without ANY host synchronization per chunk:

    - ``per_image_scores`` [contexts, 3] fp64 — the per-image (per-context)
      head-sum reduction (the ``nhr->nh`` einsum) folded into the fused
      region, never materializing the [contexts x heads x route-axis]
      intermediate; and
    - ``finite_flag`` [1] int32 — the raw-row finiteness verdict, flipped to
      0 device-side if any chunk carried a non-finite row.

    ``finalize_packet`` performs the packet's SINGLE host sync and raises
    the exact error class and message the per-chunk guard raised, so a
    non-finite row still fails the packet identically.
    """

    def __init__(self, *, contexts: int, device: Any) -> None:
        if torch is None:
            raise RuntimeError("score epilogue requires PyTorch")
        self.contexts = int(contexts)
        if self.contexts <= 0:
            raise ValueError("score epilogue requires a positive context axis")
        self.device = torch.device(device)
        self.per_image_scores = torch.zeros(
            (self.contexts, 3), dtype=torch.float64, device=self.device
        )
        self._route_block_partials = None
        self.finite_flag = torch.ones((1,), dtype=torch.int32, device=self.device)
        self.chunk_count = 0
        self.max_chunk_width = 0
        self.transient_bytes_peak = 0
        self.kernel_paths: set[str] = set()
        self.finalized = False

    def route_block_partials(self, blocks: int) -> Any:
        """Scratch with one writer per route block and a fixed reduction axis."""
        shape = (self.contexts, 3, int(blocks))
        if self._route_block_partials is None or tuple(self._route_block_partials.shape) != shape:
            self._route_block_partials = torch.empty(shape, dtype=torch.float64, device=self.device)
        return self._route_block_partials

    def note_chunk(self, *, width: int, kernel_path: str, transient_bytes: int) -> None:
        self.chunk_count += 1
        self.max_chunk_width = max(self.max_chunk_width, int(width))
        self.transient_bytes_peak = max(self.transient_bytes_peak, int(transient_bytes))
        self.kernel_paths.add(str(kernel_path))

    def observe_reference_chunk(
        self,
        *,
        local_raw: Any,
        global_raw: Any,
        per_image_chunk: np.ndarray,
    ) -> None:
        """CPU-authority chunk observation: device ops only, no host sync."""

        finite = torch.isfinite(local_raw).all() & torch.isfinite(global_raw).all()
        self.finite_flag.mul_(finite.to(torch.int32))
        self.per_image_scores.add_(
            torch.as_tensor(per_image_chunk, dtype=torch.float64, device=self.device)
        )

    def finalize_packet(self) -> dict[str, Any]:
        if self.finalized:
            raise RuntimeError("score epilogue packet already finalized")
        self.finalized = True
        # THE one host sync per packet.
        if int(self.finite_flag.item()) == 0:
            raise ValueError("device tripartite rows/context axis is invalid")
        return {
            "engaged": True,
            "kernel_path": tuple(sorted(self.kernel_paths)),
            "chunk_count": int(self.chunk_count),
            "max_chunk_width": int(self.max_chunk_width),
            "transient_bytes_estimate": int(self.transient_bytes_peak),
            "context_count": int(self.contexts),
            "host_syncs_per_packet": 1,
            "per_chunk_host_syncs_eliminated": 2 * int(self.chunk_count),
            "per_image_reduction": "ordered_route_block_partials",
            "floating_atomics": False,
        }


def _reference_chunk_update_inplace(
    accumulator: Any,
    *,
    local_raw: Any,
    global_raw: Any,
    dest_scale: Any,
    gate: Any,
    pullback: Any,
    weights: Any,
    affine_scale: float,
    route_start: int,
    signs: Any | None,
    score_state: ScoreEpilogueState | None,
) -> None:
    """CPU-authority form of the fused chunk update, applied in place.

    Arithmetic and context order are exactly ``fused_reference_update`` —
    the kernel's parity authority — so the CPU fused path IS the reference.
    """

    def as_numpy(value: Any) -> np.ndarray:
        return value.detach().cpu().numpy().astype(np.float64, copy=False)

    reference = fused_reference_update(
        local_raw=as_numpy(local_raw),
        global_raw=as_numpy(global_raw),
        dest_scale=as_numpy(dest_scale),
        gate=as_numpy(gate),
        pullback=as_numpy(pullback),
        weights=as_numpy(weights),
        affine_scale=float(affine_scale),
        signs=None if signs is None else as_numpy(signs),
    )
    start = int(route_start)
    stop = start + int(local_raw.shape[1])

    def to_acc(value: Any) -> Any:
        return torch.as_tensor(value, dtype=torch.float64, device=accumulator.sum_rows.device)

    accumulator.sum_rows[:, start:stop].add_(to_acc(reference["sum_rows"]))
    accumulator.sum_outer[start:stop].add_(to_acc(reference["sum_outer"]))
    accumulator.sum_outer_squared[start:stop].add_(to_acc(reference["sum_outer_squared"]))
    accumulator.sum_view_activity[:, start:stop].add_(to_acc(reference["sum_view_activity"]))
    accumulator.sum_view_activity_squared[:, start:stop].add_(
        to_acc(reference["sum_view_activity_squared"])
    )
    if signs is not None and int(signs.shape[0]):
        accumulator.sketch_sum[:, :, start:stop].add_(to_acc(reference["sketch_sum"]))
    if score_state is not None:
        score_state.observe_reference_chunk(
            local_raw=local_raw,
            global_raw=global_raw,
            per_image_chunk=reference["per_image_scores"],
        )
        score_state.note_chunk(
            width=stop - start,
            kernel_path="cpu_reference",
            transient_bytes=int(local_raw.numel()) * 8 * 3,
        )


class _HostPrimaryWindowTarget:
    """Present one acquired route window as accumulator slab attributes.

    The fused consumer's slab pointers are just the window tensors: the
    kernel (or its CPU parity authority) writes into the window at
    route offset 0 and the caller commits the window afterwards.
    """

    def __init__(self, windows: dict[str, Any]) -> None:
        self.sum_rows = windows["sum_rows"]
        self.sum_outer = windows["sum_outer"]
        self.sum_outer_squared = windows["sum_outer_squared"]
        self.sum_view_activity = windows["sum_view_activity"]
        self.sum_view_activity_squared = windows["sum_view_activity_squared"]
        self.sketch_sum = windows.get("sketch_sum")


def fused_route_chunk_update(
    accumulator: Any,
    *,
    local_raw: Any,
    global_raw: Any,
    dest_scale: Any,
    gate: Any,
    pullback: Any,
    weights: Any,
    affine_scale: float,
    route_start: int,
    signs: Any | None = None,
    block_r: int = 128,
    score_state: ScoreEpilogueState | None = None,
    initialize: bool = False,
) -> None:
    """Apply one fused chunk update directly onto the accumulator slabs.

    The v1 form passes ``signs`` ([K, n]); the v0 form passes None and
    the sketch epilogue compiles out (SKETCH_RANK=0 specialization).
    ``score_state`` (IRG_FUSED_SCORE_EPILOGUE) additionally accumulates
    per-image head sums and a deferred finiteness flag in the same fused
    region (SCORE_EPILOGUE=1 specialization; compiles out when absent).

    On a CPU accumulator the update runs through the CPU parity authority
    (``fused_reference_update``) in place — the same numbers the GPU
    kernel is certified against — so the fused consumer is exercisable
    without CUDA.

    HOST-PRIMARY accumulators (IRG_HOST_PRIMARY_ACCUMULATORS): the
    chunk obtains ``windowed_slabs(start, stop)`` device views instead
    of full-slab slices, the kernel writes into the window (route
    offset 0, route total = window width — the pointers ARE the window
    tensors), and the window commits after the chunk.  Where bytes live
    changes; no arithmetic operation or its order does.
    """

    initialize = bool(initialize or not bool(getattr(accumulator, "storage_allocated", True)))
    ensure_storage = getattr(accumulator, "_ensure_storage", None)
    if callable(ensure_storage):
        ensure_storage()
    if bool(getattr(accumulator, "host_primary", False)):
        start = int(route_start)
        stop = start + int(local_raw.shape[1])
        windows = accumulator.windowed_slabs(start, stop)
        target = _HostPrimaryWindowTarget(windows)
        if initialize:
            _zero_route_target(target, 0, stop - start)
        _apply_route_chunk_update(
            target,
            local_raw=local_raw,
            global_raw=global_raw,
            dest_scale=dest_scale,
            gate=gate,
            pullback=pullback,
            weights=weights,
            affine_scale=affine_scale,
            route_start=0,
            signs=signs,
            block_r=block_r,
            score_state=score_state,
        )
        accumulator.commit_window(start, stop)
        return
    if initialize:
        _zero_route_target(
            accumulator, int(route_start), int(route_start) + int(local_raw.shape[1])
        )
    _apply_route_chunk_update(
        accumulator,
        local_raw=local_raw,
        global_raw=global_raw,
        dest_scale=dest_scale,
        gate=gate,
        pullback=pullback,
        weights=weights,
        affine_scale=affine_scale,
        route_start=route_start,
        signs=signs,
        block_r=block_r,
        score_state=score_state,
    )


def _zero_route_target(target: Any, start: int, stop: int) -> None:
    """Zero exactly one route window before first-packet accumulation."""

    target.sum_rows[:, start:stop].zero_()
    target.sum_outer[start:stop].zero_()
    target.sum_outer_squared[start:stop].zero_()
    target.sum_view_activity[:, start:stop].zero_()
    target.sum_view_activity_squared[:, start:stop].zero_()
    sketch = getattr(target, "sketch_sum", None)
    if sketch is not None and int(sketch.shape[0]):
        sketch[:, :, start:stop].zero_()


def _apply_route_chunk_update(
    accumulator: Any,
    *,
    local_raw: Any,
    global_raw: Any,
    dest_scale: Any,
    gate: Any,
    pullback: Any,
    weights: Any,
    affine_scale: float,
    route_start: int,
    signs: Any | None,
    block_r: int,
    score_state: ScoreEpilogueState | None,
) -> None:
    """Kernel/reference dispatch against real slabs or one window target."""

    device = accumulator.sum_rows.device
    if device.type != "cuda":
        _reference_chunk_update_inplace(
            accumulator,
            local_raw=local_raw,
            global_raw=global_raw,
            dest_scale=dest_scale,
            gate=gate,
            pullback=pullback,
            weights=weights,
            affine_scale=affine_scale,
            route_start=route_start,
            signs=signs,
            score_state=score_state,
        )
        return
    if not fused_kernel_available():  # pragma: no cover - GPU only
        raise RuntimeError("fused route-moment kernel requires CUDA triton")
    n, r_chunk = local_raw.shape
    rank = 0 if signs is None else int(signs.shape[0])
    route_total = int(accumulator.sum_rows.shape[1])
    grid = ((r_chunk + block_r - 1) // block_r,)
    epilogue = 0 if score_state is None else 1
    score_partials = None
    if score_state is not None:  # pragma: no cover - GPU only
        if int(score_state.contexts) != int(n):
            raise ValueError("score epilogue context axis mismatch")
        score_partials = score_state.route_block_partials(grid[0])
        score_state.note_chunk(
            width=int(r_chunk),
            kernel_path="triton",
            transient_bytes=int(local_raw.numel()) * 8 * 2 + int(score_partials.numel()) * 8,
        )
    _fused_route_moment_kernel[grid](
        local_raw.contiguous(),
        global_raw.contiguous(),
        dest_scale.contiguous(),
        gate.contiguous(),
        pullback.contiguous(),
        weights.contiguous(),
        (
            signs.contiguous()
            if signs is not None
            else torch.empty((1,), dtype=torch.float64, device=device)
        ),
        accumulator.sum_rows,
        accumulator.sum_outer,
        accumulator.sum_outer_squared,
        accumulator.sum_view_activity,
        accumulator.sum_view_activity_squared,
        (accumulator.sketch_sum if rank else torch.empty((1,), dtype=torch.float64, device=device)),
        (
            score_partials
            if score_partials is not None
            else torch.empty((1,), dtype=torch.float64, device=device)
        ),
        (
            score_state.finite_flag
            if score_state is not None
            else torch.ones((1,), dtype=torch.int32, device=device)
        ),
        n,
        r_chunk,
        int(route_start),
        route_total,
        torch.tensor([float(affine_scale)], dtype=torch.float64, device=device),
        BLOCK_R=block_r,
        SKETCH_RANK=rank,
        SCORE_EPILOGUE=epilogue,
        SCORE_BLOCKS=grid[0],
    )
    if score_partials is not None:
        score_state.per_image_scores.add_(score_partials.sum(dim=-1))


def fused_inbackward_chunk_update(
    accumulator: Any,
    *,
    inputs: Any,
    grads: Any,
    layer_weight: Any,
    producer_scale: float,
    dest_scale: Any,
    gate: Any,
    pullback: Any,
    weights: Any,
    affine_scale: float,
    route_start: int,
    signs: Any | None = None,
    block_r: int = 128,
    initialize: bool = False,
) -> str:
    """Apply one full in-backward chunk update onto the accumulator slabs.

    CUDA: ``_census_linear_inbackward_kernel`` — row formation, head
    formation, and every moment slab in one launch; the ``[n, o, I]`` row
    tensor never exists.  CPU: the composed parity authority
    (``census_inbackward_reference`` order) applied in place, so the CPU
    fused path IS the reference.  HOST-PRIMARY accumulators window the
    slab writes exactly as ``fused_route_chunk_update`` does.  Returns
    the kernel-path label for receipts.  Raises on any kernel problem —
    the sink owns fail-soft degradation.
    """

    initialize = bool(initialize or not bool(getattr(accumulator, "storage_allocated", True)))
    ensure_storage = getattr(accumulator, "_ensure_storage", None)
    if callable(ensure_storage):
        ensure_storage()
    o_chunk = int(grads.shape[1])
    in_features = int(inputs.shape[1])
    width = o_chunk * in_features
    if bool(getattr(accumulator, "host_primary", False)):
        start = int(route_start)
        stop = start + width
        windows = accumulator.windowed_slabs(start, stop)
        target = _HostPrimaryWindowTarget(windows)
        if initialize:
            _zero_route_target(target, 0, width)
        label = _apply_inbackward_chunk_update(
            target,
            inputs=inputs,
            grads=grads,
            layer_weight=layer_weight,
            producer_scale=producer_scale,
            dest_scale=dest_scale,
            gate=gate,
            pullback=pullback,
            weights=weights,
            affine_scale=affine_scale,
            route_start=0,
            signs=signs,
            block_r=block_r,
        )
        accumulator.commit_window(start, stop)
        return label
    if initialize:
        _zero_route_target(accumulator, int(route_start), int(route_start) + width)
    return _apply_inbackward_chunk_update(
        accumulator,
        inputs=inputs,
        grads=grads,
        layer_weight=layer_weight,
        producer_scale=producer_scale,
        dest_scale=dest_scale,
        gate=gate,
        pullback=pullback,
        weights=weights,
        affine_scale=affine_scale,
        route_start=route_start,
        signs=signs,
        block_r=block_r,
    )


def _apply_inbackward_chunk_update(
    accumulator: Any,
    *,
    inputs: Any,
    grads: Any,
    layer_weight: Any,
    producer_scale: float,
    dest_scale: Any,
    gate: Any,
    pullback: Any,
    weights: Any,
    affine_scale: float,
    route_start: int,
    signs: Any | None,
    block_r: int,
) -> str:
    device = accumulator.sum_rows.device
    if device.type != "cuda":
        raw_local, raw_global = _torch_inbackward_raw_rows(
            inputs, grads, layer_weight, producer_scale
        )
        _reference_chunk_update_inplace(
            accumulator,
            local_raw=raw_local,
            global_raw=raw_global,
            dest_scale=dest_scale,
            gate=gate,
            pullback=pullback,
            weights=weights,
            affine_scale=affine_scale,
            route_start=route_start,
            signs=signs,
            score_state=None,
        )
        return "cpu_reference_inbackward"
    if not fused_kernel_available():  # pragma: no cover - GPU only
        raise RuntimeError("in-backward census kernel requires CUDA triton")
    contexts, o_chunk = (int(grads.shape[0]), int(grads.shape[1]))
    in_features = int(inputs.shape[1])
    width = o_chunk * in_features
    rank = 0 if signs is None else int(signs.shape[0])
    route_total = int(accumulator.sum_rows.shape[1])
    grid = ((width + block_r - 1) // block_r,)
    _census_linear_inbackward_kernel[grid](  # pragma: no cover - GPU only
        inputs.to(dtype=torch.float32).contiguous(),
        grads.to(dtype=torch.float32).contiguous(),
        layer_weight.to(dtype=torch.float32).contiguous(),
        dest_scale.contiguous(),
        gate.contiguous(),
        pullback.contiguous(),
        weights.contiguous(),
        (
            signs.contiguous()
            if signs is not None
            else torch.empty((1,), dtype=torch.float64, device=device)
        ),
        accumulator.sum_rows,
        accumulator.sum_outer,
        accumulator.sum_outer_squared,
        accumulator.sum_view_activity,
        accumulator.sum_view_activity_squared,
        (accumulator.sketch_sum if rank else torch.empty((1,), dtype=torch.float64, device=device)),
        contexts,
        width,
        int(route_start),
        route_total,
        in_features,
        o_chunk,
        torch.tensor([float(producer_scale)], dtype=torch.float32, device=device),
        torch.tensor([float(affine_scale)], dtype=torch.float64, device=device),
        BLOCK_R=block_r,
        SKETCH_RANK=rank,
    )
    return "triton_inbackward"


class CensusBackwardMomentSink:
    """Chunk-streamed census backward producer sink (IRG_FUSED_CENSUS_BACKWARD).

    The with-context salience reducers, when armed, hand each route-row
    chunk here as whole-context ``[contexts, 1, ...route-chunk]`` fp32
    global rows plus the query-independent ``[contexts, ...route-chunk]``
    local-destination rows — and never allocate the persistent
    ``[contexts x full-route-axis]`` per-context head (the K=2 census OOM
    class).  This sink forms the tripartite heads for one route chunk at a
    time (fp64, elementwise order identical to the eager provider branch:
    ``l = (local*dest)*gate``, ``g = (raw*scale)*gate``,
    ``s = (pullback*(raw*scale))*gate``) and streams them into the
    ESTABLISHED v1 accumulator batch protocol —
    ``begin_streamed_batch`` / ``update_streamed_route_block`` /
    ``end_streamed_batch`` — which enforces exact route-axis coverage and
    commits packet counts once.  Peak transient is O(one chunk).

    Numerics: heads and every moment reduction are fp64.  Relative to the
    current unfused pipeline the CONTEXT reduction order changes (the
    streamed protocol's ``"n,nhr->hr"`` contraction versus the chunked
    provider's route-batched form), so agreement is reduction-order-
    equivalent at fp64, not bitwise — the same documented relation the
    streamed protocol already holds to the monolithic ``update``.
    """

    def __init__(
        self,
        *,
        accumulator: Any,
        context_ids: Any,
        dest_scale: Any,
        gate: Any,
        pullback: Any,
        affine_scale: float,
        route_chunk_size: int,
        route_start: int = 0,
    ) -> None:
        if torch is None:
            raise RuntimeError("census backward sink requires PyTorch")
        self._accumulator = accumulator
        self._context_ids = tuple(map(str, context_ids))
        self._contexts = len(self._context_ids)
        # The COMPUTE device.  For every pre-existing accumulator state
        # this equals ``sum_rows.device`` exactly (residency moves keep
        # them in lock-step); under host-primary the slabs live on host
        # while compute stays on ``accumulator.device``.
        device = getattr(accumulator, "device", None)
        if device is None:
            device = accumulator.sum_rows.device
        device = torch.device(device)
        self._device = device

        def as_f64(value: Any) -> Any:
            return torch.as_tensor(value, dtype=torch.float64, device=device).detach()

        self._dest_scale = as_f64(dest_scale).reshape(-1)
        self._gate = as_f64(gate).reshape(-1)
        self._pullback = as_f64(pullback).reshape(-1)
        self._affine_scale = float(affine_scale)
        self._route_chunk_size = int(route_chunk_size)
        self._route_start = int(route_start)
        route_count = int(self._dest_scale.shape[0])
        if (
            self._contexts <= 0
            or route_count <= 0
            or self._route_chunk_size <= 0
            or tuple(self._gate.shape) != (route_count,)
            or tuple(self._pullback.shape) != (self._contexts,)
            or not float(self._affine_scale) > 0.0
        ):
            raise ValueError("census backward sink axes are invalid")
        self._route_count = route_count
        self.chunk_count = 0
        self.max_chunk_width = 0
        self.transient_bytes_peak = 0
        # Direct in-backward packet state.
        self._inbackward_open = False
        self._inbackward_covered: Any | None = None
        self._inbackward_weights: Any | None = None
        self._inbackward_signs: Any | None = None
        self._inbackward_initialize = False
        self._kernel_paths: set[str] = set()

    # Capability negotiation flags (producer handoff).  Defaults keep
    # every pre-handoff consumer accepting all emissions; a capture-time
    # in-backward sink flips them so materialized streams are declined and
    # conv blocks stay on retained tiles (tile-fed by the provider).
    wants_linear_rows = True
    wants_conv_rows = True

    @property
    def wants_linear_inbackward(self) -> bool:
        """Producer capability probe: fc chunks may skip row materialization."""

        return True

    def covers_span(self, start: int, stop: int) -> bool:
        """True when the in-backward packet already ingested [start, stop).

        Sink-relative route offsets.  The provider uses this to skip
        tile-feeding blocks the capture-time producer handoff streamed.
        """

        if self._inbackward_covered is None:
            return False
        if not (0 <= int(start) < int(stop) <= self._route_count):
            return False
        return bool(self._inbackward_covered[int(start) : int(stop)].all().item())

    # ------------------------------------------------------------------ #
    # Full in-backward form
    # ------------------------------------------------------------------ #

    def _begin_inbackward(self) -> None:
        """Open the packet with the direct-bookkeeping pattern.

        Mirrors ``begin_streamed_batch`` validation and the provider-side
        ``_RouteChunkedTripartiteUpdate`` bookkeeping (the established
        direct-slab-write pattern): unit context weights, deterministic
        sketch signs from the accumulator, sink-local coverage.
        """

        accumulator = self._accumulator
        ids = self._context_ids
        if len(set(ids)) != len(ids) or set(ids).intersection(accumulator.context_ids):
            raise ValueError("streamed batch context axis is invalid")
        self._inbackward_initialize = int(accumulator.sample_count) == 0
        ensure_storage = getattr(accumulator, "_ensure_storage", None)
        if callable(ensure_storage):
            ensure_storage()
        wait = getattr(accumulator, "wait_transfers", None)
        if wait is not None:
            wait(torch.cuda.current_stream() if self._device.type == "cuda" else None)
        self._inbackward_weights = torch.ones(
            self._contexts, dtype=torch.float64, device=self._device
        )
        self._inbackward_signs = (
            accumulator._signs(ids) if int(getattr(accumulator, "sketch_rank", 0)) else None
        )
        self._inbackward_covered = torch.zeros(self._route_count, dtype=torch.bool, device="cpu")
        self._inbackward_open = True

    def _mark_covered(self, offset: int, width: int) -> None:
        if bool(self._inbackward_covered[offset : offset + width].any().item()):
            raise ValueError("streamed route block overlaps prior coverage")
        self._inbackward_covered[offset : offset + width] = True

    def consume_linear_backward(
        self,
        route_offset: int,
        *,
        inputs: Any,
        grads: Any,
        layer_weight: Any,
        producer_scale: float,
    ) -> None:
        """Consume one fc chunk WITHOUT materialized rows (in-backward form).

        ``inputs`` [contexts, in_features] and ``grads`` [contexts,
        o_chunk] are the backward pass's own tensors; ``layer_weight``
        is the [o_chunk, in_features] weight slice.  The kernel forms
        the fp32 rows in-register and accumulates every moment slab in
        one launch; on CPU the composed parity authority applies in
        place.  Fail-soft: any kernel failure degrades THIS chunk to the
        graded form (one bounded raw-row materialization through the
        established chunk-update entry) with a receipted reason — never
        an arm failure.
        """

        if (
            inputs is None
            or grads is None
            or layer_weight is None
            or int(inputs.ndim) != 2
            or int(grads.ndim) != 2
            or int(layer_weight.ndim) != 2
            or int(inputs.shape[0]) != self._contexts
            or int(grads.shape[0]) != self._contexts
            or int(layer_weight.shape[0]) != int(grads.shape[1])
            or int(layer_weight.shape[1]) != int(inputs.shape[1])
        ):
            raise ValueError("in-backward census chunk axes are invalid")
        offset = int(route_offset)
        width = int(grads.shape[1]) * int(inputs.shape[1])
        if width <= 0 or offset < 0 or offset + width > self._route_count:
            raise ValueError("in-backward census chunk axes are invalid")
        if not bool(
            (
                torch.isfinite(inputs).all()
                & torch.isfinite(grads).all()
                & torch.isfinite(layer_weight).all()
            ).item()
        ):
            raise ValueError("streamed route block is invalid")
        if not self._inbackward_open:
            self._begin_inbackward()
        self._mark_covered(offset, width)
        span = slice(offset, offset + width)
        absolute_start = self._route_start + offset
        label = fused_inbackward_chunk_update(
            self._accumulator,
            inputs=inputs,
            grads=grads,
            layer_weight=layer_weight,
            producer_scale=float(producer_scale),
            dest_scale=self._dest_scale[span],
            gate=self._gate[span],
            pullback=self._pullback,
            weights=self._inbackward_weights,
            affine_scale=self._affine_scale,
            route_start=absolute_start,
            signs=self._inbackward_signs,
            initialize=self._inbackward_initialize,
        )
        self._kernel_paths.add(label)
        self.chunk_count += 1
        self.max_chunk_width = max(self.max_chunk_width, width)
        transient = int(
            inputs.element_size() * inputs.numel()
            + grads.element_size() * grads.numel()
            + layer_weight.element_size() * layer_weight.numel()
        )
        self.transient_bytes_peak = max(self.transient_bytes_peak, transient)

    def _consume_rows_inbackward(
        self,
        route_offset: int,
        per_context_rows: Any,
        local_rows: Any,
    ) -> None:
        """In-backward ingestion of materialized row chunks (conv blocks).

        Conv emissions stay block-bounded and arrive as raw rows; they go
        through the established raw-row chunk update so the whole packet
        shares ONE accumulation order (the sequential authority on CPU,
        the consumer kernel on CUDA).
        """

        rows = per_context_rows.reshape(
            int(per_context_rows.shape[0]), int(per_context_rows.shape[1]), -1
        )
        if int(rows.shape[0]) != self._contexts or int(rows.shape[1]) != 1:
            raise ValueError(
                "census backward streaming requires whole-context single-query row chunks"
            )
        rows = rows[:, 0, :]
        local = local_rows.reshape(self._contexts, -1)
        width = int(rows.shape[1])
        offset = int(route_offset)
        if (
            width <= 0
            or int(local.shape[1]) != width
            or offset < 0
            or offset + width > self._route_count
        ):
            raise ValueError("census backward route chunk axes are invalid")
        if not bool((torch.isfinite(rows).all() & torch.isfinite(local).all()).item()):
            raise ValueError("streamed route block is invalid")
        if not self._inbackward_open:
            self._begin_inbackward()
        self._mark_covered(offset, width)
        transient = int(rows.element_size() * rows.numel() + local.element_size() * local.numel())
        for start in range(0, width, self._route_chunk_size):
            stop = min(width, start + self._route_chunk_size)
            span = slice(offset + start, offset + stop)
            fused_route_chunk_update(
                self._accumulator,
                local_raw=local[:, start:stop].to(device=self._device, dtype=torch.float64),
                global_raw=rows[:, start:stop].to(device=self._device, dtype=torch.float64),
                dest_scale=self._dest_scale[span],
                gate=self._gate[span],
                pullback=self._pullback,
                weights=self._inbackward_weights,
                affine_scale=self._affine_scale,
                route_start=self._route_start + offset + start,
                signs=self._inbackward_signs,
                initialize=self._inbackward_initialize,
            )
            self.chunk_count += 1
            self.max_chunk_width = max(self.max_chunk_width, stop - start)
            self.transient_bytes_peak = max(self.transient_bytes_peak, transient)
        # Tile-fed (materialized) rows — receipted distinctly from the
        # row-free producer-fed path, so the receipt says which one ran.
        self._kernel_paths.add(
            "tile_rows_reference" if self._device.type != "cuda" else "tile_rows_consumer_kernel"
        )

    def __call__(
        self,
        route_offset: int,
        per_context_rows: Any,
        local_rows: Any,
    ) -> None:
        """Consume one whole-context route-row chunk from the producer.

        ``per_context_rows``: [contexts, targets(=1), *chunk-route-shape]
        (the exact values the retained per-context head would have held for
        these routes); ``local_rows``: [contexts, *chunk-route-shape].
        ``route_offset`` indexes the block's flattened route axis.
        """

        if local_rows is None:
            raise ValueError("census backward streaming requires local-destination rows")
        self._consume_rows_inbackward(route_offset, per_context_rows, local_rows)

    def finalize(self) -> dict[str, Any]:
        """Close the packet after complete route-axis coverage."""
        return self._finalize_inbackward()

    def _finalize_inbackward(self) -> dict[str, Any]:
        accumulator = self._accumulator
        if not self._inbackward_open:
            raise RuntimeError("census backward sink saw no route chunks for this packet")
        axis = len(
            getattr(accumulator, "route_ids", ()) or range(int(accumulator.sum_rows.shape[1]))
        )
        if (
            self._route_start != 0
            or self._route_count != axis
            or not bool(self._inbackward_covered.all().item())
        ):
            raise ValueError("streamed batch closed without covering the full route axis")
        weights = self._inbackward_weights
        accumulator.total_weight += float(weights.sum().item())
        accumulator.sum_weight_squared += float(torch.sum(weights * weights).item())
        accumulator.sample_count += self._contexts
        if int(getattr(accumulator, "sketch_rank", 0)):
            write = "copy_" if self._inbackward_initialize else "add_"
            getattr(accumulator.sketch_constant, write)(
                torch.einsum("kn,n->k", self._inbackward_signs, torch.sqrt(weights))
            )
        elif self._inbackward_initialize:
            accumulator.sketch_constant.zero_()
        accumulator._context_ids.extend(self._context_ids)
        self._inbackward_open = False
        self._inbackward_weights = None
        self._inbackward_signs = None
        self._inbackward_covered = None
        self._inbackward_initialize = False
        return {
            "engaged": True,
            "form": "inbackward",
            "kernel_path": tuple(sorted(self._kernel_paths)),
            "chunk_count": int(self.chunk_count),
            "max_chunk_width": int(self.max_chunk_width),
            "route_chunk_size": int(self._route_chunk_size),
            "transient_bytes_estimate": int(self.transient_bytes_peak),
            "context_count": int(self._contexts),
            "route_count": int(self._route_count),
        }
