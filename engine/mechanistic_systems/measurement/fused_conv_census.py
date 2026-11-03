"""Fused per-image conv weight-gradient census.

README
======
What it computes.  For one Conv2d (groups = 1) and a packet of images i, per route r = (o, c, kh, kw):
    g0[i, r] = (sum_p patch_i(c, kh, kw, p) * (delta_i(o, p) / sp)) * W[o, c, kh, kw]   fp32 products and sum (TF32 under
                                                                                      the run's matmul policy)
    g[i, r]  = fp64(g0[i, r]) * a
and per-route fp64 sums over images  S = sum_i coef_i * value(g[i, r]),  value in {g, |g|, g^2, (g^2)^2},
coef_i in {h_i, h_i c_i, h_i |c_i|, h_i c_i^2, h_i (c_i^2)^2} (statistics g1 ga g2 g4 s1 sa s2 s4), each statistic with
its own per-image weights h; plus the local head l1 = sum_i h_i l_i, l2 = sum_i h_i l_i^2 with
l_i = fp64(fp32(fp32(patch_sum_i * W) * scale)) * dest[o].  No per-image [b, R] quantity is written to global memory.
Entry points (torch + triton only): row_reducer (the FastBatchCensusSession slot; returns None for unsupported blocks
and single-image packets, adds into caller-supplied fp64 views), conv_stats (drop-in for fast_census.conv_stats),
conv_row_sums (the core).

Design.  (1) Staging: a Triton im2col kernel writes the patches [b, K, P16] (F.unfold's values; rows padded to 16
positions); under TF32 it rounds them to nearest TF32 (cvt.rna semantics: Triton's tensor cores would otherwise
truncate, cuBLAS rounds) and emits fp64 partial row sums of the unrounded values for the local head; delta is copied
and rounded the same way.  Under true fp32 the patches are staged tap-contiguous for the FMA path and the local head
uses the engine's own unfold(...).sum(-1).  (2) The fused kernel: one program owns a tile of routes (BLOCK_O x
BLOCK_N) and a contiguous range of images; per image the correlation tile delta_i @ patch_i^T is computed on chip (wgmma
under TF32, FMA under fp32; one flat loop over (image, position chunk) so loads pipeline across images), then
g = fp64(corr * W) * a and every requested coef_i * value(g) is added into an fp64 register accumulator, all statistics
in one pass.  Programs over disjoint image ranges write fp64 partials summed in a fixed order (deterministic, no
atomics).  (3) A small fused local-head kernel.  Tiles per (accumulator count, layer shape) come from H200 sweeps
(_CONFIG_TABLE).  All launches go to the caller's current stream; inputs are record_stream'ed on it.
The census walk requests exactly two fp64 slots per batch (sa, z2_s;
s1/g1 only on the first look's draws), which allows 128x64 tiles; four or more slots force 64x64 tiles (four 128x64
fp64 tiles would fill the whole 256 KB register file).

Measured, H200 (all conv layers, TF32 policy; runs/ JSON named per job in the authors' run log).
Census session workload vs the session's operator path (per-image rows -> fp64 [b, R] -> GEMV per statistic), img/s:
    network (batch)    walk: sa,z2_s + l1,z2_l       old walk: + s1,g1 (4 slots)    full (8 slots)
    AlexNet (256)      3,921 -> 32,713  (8.3x)       3,785 -> 24,355 (6.4x)         3,112 -> 15,805 (5.1x)
    VGG-11  (64)       1,182 ->  3,995  (3.4x)       1,133 ->  2,988 (2.6x)           909 ->  2,046 (2.3x)
    ResNet-18 (96)     1,039 ->  6,946  (6.7x)         993 ->  5,619 (5.7x)           785 ->  3,515 (4.5x)
    ResNet-50 (64)       460 ->  2,555  (5.6x)         437 ->  2,181 (5.0x)           351 ->  1,434 (4.1x)
Kernel-only b* (smallest batch at >= 95% of plateau, walk workload): AlexNet 512 (plateau 35.5k img/s), VGG-11 128
(4.3k), ResNet-18 512 (9.3k), ResNet-50 512 (4.7k); staging is chunked to <= 2 GB, so the batch is limited by
throughput, not memory.  ResNet-50 capture: 3.13k img/s (reducer 1.23x one capture pass).
Census capture (forward + backward with route hooks, TF32) on the same GPU: AlexNet 33.7k, VGG-11 3.66k, ResNet-18
9.40k img/s, i.e. the conv reducer now costs 1.03x / 0.92x / 1.35x one capture pass.
Tuning rounds (walk img/s AlexNet / VGG-11 / ResNet-18): r0 16.8k / 2.63k / 4.46k; r1 two-stream overlap slower,
reverted; r2 Triton im2col +3 / +9 / +6%; r3 per-shape tiles +10 / 0 / +13%; r4 operands pre-rounded at staging
+57 / +30 / +24%; r5 re-swept tiles +6 / +6 / +2%; r6 staging-kernel tiles ~1% (not kept); r7 fewer small ops
+4 / +1 / +4%: two successive rounds < 5%, tuning stopped.

Per layer (H200, walk workload; ms per batch: staging / kernel / local head; kernel TFLOP/s and % of the 495 TF32
peak; staging bytes/s and % of the 4.8 TB/s HBM peak):
    AlexNet  conv1 64x3x11x11 P=3025  0.75/0.37/0.05   97 TF/s 20%  2.3 TB/s 47%
             conv2 192x64x5x5 P=729   0.99/0.91/0.07  126      25%  1.6      32%
             conv3-5 3x3 P=169        0.31-0.56/0.67-1.12/0.12-0.17   68-76 14-15%  1.4-1.5 29-32%
    VGG-11   conv1 64x3x3x3 P=50176   0.55/0.45/0.04   25       5%  3.7      77%
             conv2-6 P=12544..784     0.40-1.45/0.89-2.32/0.04-0.10  102-137 21-28%  1.7-3.2 35-68%
             conv7,8 512x512 P=196    0.21/0.88/0.10   67      14%  1.6      33%
    ResNet-18 stem 7x7 P=12544        0.58/0.31/0.05   73      15%  2.4      50%
             3x3 P=3136..784          0.16-0.56/0.14-0.25/0.04  79-99 16-20%  1.6-2.1 34-44%
             3x3 P=196 / P=49         0.07-0.16/0.18-0.55/0.05-0.14  32-64 7-13%  1.3-1.9 28-39%
             1x1 downsample           0.07-0.08/0.06/0.04-0.05  21-22 4%  0.6-2.2 14-47%
Binding limits: no layer reaches either roof.  Large-P layers (VGG conv2-6, AlexNet conv1-2): one CTA per SM (two fp64
accumulator tiles use half the register file) and the once-per-image fp64 epilogue interrupting the MMA stream keep the
kernel at 20-28% of TF32 peak.  Small-P layers (P <= 196): the epilogue is amortized over only 1-7 MMA steps per image
(epilogue/latency-bound).  1x1 and C=3 layers: tiny reductions; launch- and memory-bound but small in total.  Staging:
gather-bound (index arithmetic and unaligned loads; a plain im2col takes ~3x a memset of the same buffer), at 30-77%
of HBM.  Against the target (reducer no costlier than the backward's own weight-gradient arithmetic): not met;
the kernel FLOPs equal a batched wgrad's, but run at 67-137 TF/s where a tuned TF32 GEMM reaches ~280 TF/s (cuBLAS,
measured on H100), and staging plus local head add ~40% (not measured: a cuDNN wgrad of these shapes).

What a CUTLASS 3 wgrad with a custom fp64 epilogue would buy (estimate, not built): TMA im2col removes the staging
buffer (staging minus delta rounding and patch sums: ~25% of VGG-11's and ~30% of AlexNet's and ResNet-18's time);
warp-specialized wgmma with the fp64 accumulators in shared memory (2 slots x 128x64 fp64 = 128 KB) frees the
registers that now cap the tile and occupancy, plausibly 2-2.5x on the kernel for large-P layers, 1.5-2x for small-P.
Conv reducer: AlexNet 32.7k -> ~70k, VGG-11 4.0k -> ~10k, ResNet-18 6.9k -> ~16k img/s (about 2.2-2.6x).  End to end
with the census capture above (fc GEMMs not included): AlexNet 16.6k -> ~23k, VGG-11 1.9k -> ~2.7k, ResNet-18
4.0k -> ~5.9k img/s (about 1.4-1.5x), since the capture then dominates.  Uncertainty about +-40%; several days of work;
the fp64-in-shared-memory epilogue is unproven.

Equivalence and what is verified (H100 and H200):
  - True fp32: <= 1e-15 relative vs the reference (fast_census.conv_stats and the engine operator path) on every real
    layer of the three networks and on random shapes (stride, padding, dilation, signed producer scale, image
    weights): the per-image fp32 sums are bitwise those of cuBLAS's batched GEMM.  For a single-image packet cuBLAS
    runs a non-batched GEMM with a different summation order (up to 4.6e-6), so row_reducer returns None and the
    session's operator path handles that image.
  - TF32: against an fp64 truth, the fused/reference RMS error ratio over 512 layer x statistic cases has median
    1.000 (0.983-1.033); acceptance RMS <= 1.05x (1.10x for g4, s4), worst route <= 1.25x.  Where cuBLAS ran true fp32
    despite the TF32 policy (one real layer at test batch: ResNet layer4.0.downsample), the fused error is 1.01x the
    ideal TF32 error.  Local head under TF32: patch sums correctly rounded (<= 0.5 ulp; engine's fp32 sum up to 1.2
    ulp), so not bitwise the engine but never less precise; under fp32 exact.
  - Route order: no inversion beyond the two paths' observed rounding on any layer or statistic.
  - Session level: FastBatchCensusSession with this reducer equals the operator path (estimates 1e-12, SEs 1e-10) in
    walk and full mode.
  - Memory safety: canary-guarded shared buffers for all 23 real conv shapes of AlexNet / VGG-11 / ResNet-18 and the
    26 of ResNet-50 (torchvision bottlenecks plus the original-V1 strided 1x1s) at the in-arm b* (1,024 / 512 / 1,024
    / 1,024) and ragged batches, untouched; compute-sanitizer memcheck over all 49 shapes, both policies: 0 errors.
  Not verified: H100 numbers for the final module (H100 runs were for earlier rounds); in-arm end-to-end census
  throughput (the kernels worker's runs); a cuDNN wgrad baseline; groups != 1 (returns None).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # no Triton (e.g. the Mac): the module imports for reading; the entry points raise
    triton = None
    tl = None

ROW_STATS = ("g1", "ga", "g2", "g4", "s1", "sa", "s2", "s4")
LOCAL_STATS = ("l1", "l2")
VALUE_KIND = {"g1": 0, "s1": 0, "ga": 1, "sa": 1, "g2": 2, "s2": 2, "g4": 3, "s4": 3}
STAT_ORDER = {"g1": "first", "ga": "first", "s1": "first", "sa": "first", "l1": "first",
              "g2": "second", "s2": "second", "l2": "second", "g4": "fourth", "s4": "fourth"}
MAX_SLOTS = 8  # fp64 accumulators per pass
IM2COL_BLOCK_MAX, IM2COL_ELEMS, IM2COL_WARPS = 1024, 2048, 4  # TF32 staging kernel tile (positions, elements, warps)
STAGE_BUDGET_BYTES = 2 * 2**30  # staging copies (im2col patches, padded delta) per image chunk


def coefficient(stat: str, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Per-image fp64 coefficient of one statistic (the expressions of census_sufficient_statistics._weighted_rows
    and fast_census.conv_stats)."""
    if stat in ("g1", "ga", "g2", "g4"):
        return h
    if stat == "s1":
        return h * c
    if stat == "sa":
        return h * c.abs()
    if stat == "s2":
        return h * c * c
    if stat == "s4":
        return h * (c * c) ** 2
    raise ValueError(stat)


def _round16(v: int) -> int:
    return (int(v) + 15) // 16 * 16


def _out_size(size: int, k: int, s: int, p: int, d: int) -> int:
    return (size + 2 * p - d * (k - 1) - 1) // s + 1


if triton is not None:

    @triton.jit
    def _value(g, VT: tl.constexpr):
        if VT == 0:
            r = g
        elif VT == 1:
            r = tl.abs(g)
        elif VT == 2:
            r = g * g
        else:
            g2 = g * g
            r = g2 * g2
        return r

    @triton.jit
    def _tf32_rn(v):
        # fp32 -> TF32 by round-to-nearest (ties away), as cvt.rna.tf32.f32; the tensor core would otherwise drop
        # the 13 low mantissa bits (truncation, a biased rounding that cuBLAS does not use)
        bits = v.to(tl.int32, bitcast=True)
        bits = (bits + 0x1000) & -8192
        return bits.to(tl.float32, bitcast=True)

    @triton.jit
    def _acc(acc, coef_ptr, slot, s_coef, i, g, VT: tl.constexpr):
        if VT >= 0:
            acc += tl.load(coef_ptr + slot * s_coef + i) * _value(g, VT)
        return acc

    @triton.jit
    def _store(out_base, slot, s_oj, optr, acc, tile_mask, VT: tl.constexpr):
        if VT >= 0:
            tl.store(out_base + slot * s_oj + optr, acc, mask=tile_mask)

    @triton.jit
    def _fused_conv_census_kernel(
            pat_ptr, del_ptr, w_ptr, a_ptr, coef_ptr, out_ptr,
            n_img, O, K, P, ipp,
            s_pn, s_pk, s_pp, s_dn, s_do,
            s_coef, s_os, s_oj,
            VT0: tl.constexpr, VT1: tl.constexpr, VT2: tl.constexpr, VT3: tl.constexpr,
            VT4: tl.constexpr, VT5: tl.constexpr, VT6: tl.constexpr, VT7: tl.constexpr,
            BLOCK_O: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_P: tl.constexpr, TF32: tl.constexpr):
        """pat: patches [n, K, P] (k = c, kh, kw; strided); delta: [n, O, P] (positions contiguous); w: [O, K];
        P is a multiple of 16 with zero tails in both operands.  Program = (tile of O x K routes, range of images)."""
        pid_t = tl.program_id(0)
        pid_s = tl.program_id(1)
        n_nt = tl.cdiv(K, BLOCK_N)
        po = pid_t // n_nt
        pn = pid_t % n_nt
        offs_o = po * BLOCK_O + tl.arange(0, BLOCK_O)
        offs_n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
        mo = offs_o < O
        mn = offs_n < K
        a = tl.load(a_ptr)
        acc0 = tl.zeros([BLOCK_O, BLOCK_N], dtype=tl.float64)
        acc1 = tl.zeros([BLOCK_O, BLOCK_N], dtype=tl.float64)
        acc2 = tl.zeros([BLOCK_O, BLOCK_N], dtype=tl.float64)
        acc3 = tl.zeros([BLOCK_O, BLOCK_N], dtype=tl.float64)
        acc4 = tl.zeros([BLOCK_O, BLOCK_N], dtype=tl.float64)
        acc5 = tl.zeros([BLOCK_O, BLOCK_N], dtype=tl.float64)
        acc6 = tl.zeros([BLOCK_O, BLOCK_N], dtype=tl.float64)
        acc7 = tl.zeros([BLOCK_O, BLOCK_N], dtype=tl.float64)
        i0 = pid_s * ipp
        i1 = tl.minimum(n_img, i0 + ipp)
        offs_p0 = tl.arange(0, BLOCK_P)
        n_pc = tl.cdiv(P, BLOCK_P)
        corr = tl.zeros([BLOCK_O, BLOCK_N], dtype=tl.float32)
        # one flat loop over (image, spatial chunk) so the load pipeline runs across image boundaries;
        # the per-image epilogue (fp64 accumulation) runs on the image's last chunk
        for t in range(0, (i1 - i0) * n_pc):
            il = t // n_pc
            pc = t - il * n_pc
            i = i0 + il
            offs_p = pc * BLOCK_P + offs_p0
            mp = offs_p < P
            d = tl.load(del_ptr + i.to(tl.int64) * s_dn + offs_o[:, None] * s_do + offs_p[None, :],
                        mask=mo[:, None] & mp[None, :], other=0.0)
            xt = tl.load(pat_ptr + i.to(tl.int64) * s_pn + offs_n[:, None] * s_pk + offs_p[None, :] * s_pp,
                         mask=mn[:, None] & mp[None, :], other=0.0)
            if TF32:  # the run's matmul policy; the operands were rounded to nearest TF32 when staged
                corr = tl.dot(d, tl.trans(xt), corr, input_precision="tf32")
            else:
                corr = tl.dot(d, tl.trans(xt), corr, input_precision="ieee")
            if pc == n_pc - 1:
                w = tl.load(w_ptr + offs_o[:, None] * K + offs_n[None, :], mask=mo[:, None] & mn[None, :],
                            other=0.0)
                g = (corr * w).to(tl.float64) * a
                acc0 = _acc(acc0, coef_ptr, 0, s_coef, i, g, VT0)
                acc1 = _acc(acc1, coef_ptr, 1, s_coef, i, g, VT1)
                acc2 = _acc(acc2, coef_ptr, 2, s_coef, i, g, VT2)
                acc3 = _acc(acc3, coef_ptr, 3, s_coef, i, g, VT3)
                acc4 = _acc(acc4, coef_ptr, 4, s_coef, i, g, VT4)
                acc5 = _acc(acc5, coef_ptr, 5, s_coef, i, g, VT5)
                acc6 = _acc(acc6, coef_ptr, 6, s_coef, i, g, VT6)
                acc7 = _acc(acc7, coef_ptr, 7, s_coef, i, g, VT7)
                corr = tl.zeros([BLOCK_O, BLOCK_N], dtype=tl.float32)
        out_base = out_ptr + pid_s.to(tl.int64) * s_os
        optr = offs_o[:, None] * K + offs_n[None, :]
        tile_mask = mo[:, None] & mn[None, :]
        _store(out_base, 0, s_oj, optr, acc0, tile_mask, VT0)
        _store(out_base, 1, s_oj, optr, acc1, tile_mask, VT1)
        _store(out_base, 2, s_oj, optr, acc2, tile_mask, VT2)
        _store(out_base, 3, s_oj, optr, acc3, tile_mask, VT3)
        _store(out_base, 4, s_oj, optr, acc4, tile_mask, VT4)
        _store(out_base, 5, s_oj, optr, acc5, tile_mask, VT5)
        _store(out_base, 6, s_oj, optr, acc6, tile_mask, VT6)
        _store(out_base, 7, s_oj, optr, acc7, tile_mask, VT7)

    @triton.jit
    def _local_head_kernel(ps_ptr, w_ptr, dest_ptr, sc_ptr, h1_ptr, h2_ptr, out_ptr,
                           n_img, O, K, ipp, s_psn, s_os, s_oj,
                           HAS1: tl.constexpr, HAS2: tl.constexpr, BLOCK_O: tl.constexpr, BLOCK_N: tl.constexpr):
        """l_i = fp64(fp32(fp32(ps_i[k] * w[o, k]) * sc)) * dest[o];  out = (sum_i h1_i l_i, sum_i h2_i l_i^2)."""
        pid_t = tl.program_id(0)
        pid_s = tl.program_id(1)
        n_nt = tl.cdiv(K, BLOCK_N)
        po = pid_t // n_nt
        pn = pid_t % n_nt
        offs_o = po * BLOCK_O + tl.arange(0, BLOCK_O)
        offs_n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
        mo = offs_o < O
        mn = offs_n < K
        tile_mask = mo[:, None] & mn[None, :]
        optr = offs_o[:, None] * K + offs_n[None, :]
        w = tl.load(w_ptr + optr, mask=tile_mask, other=0.0)
        dest = tl.load(dest_ptr + offs_o, mask=mo, other=0.0)
        sc = tl.load(sc_ptr)
        acc1 = tl.zeros([BLOCK_O, BLOCK_N], dtype=tl.float64)
        acc2 = tl.zeros([BLOCK_O, BLOCK_N], dtype=tl.float64)
        i0 = pid_s * ipp
        i1 = tl.minimum(n_img, i0 + ipp)
        for i in range(i0, i1):
            ps = tl.load(ps_ptr + i.to(tl.int64) * s_psn + offs_n, mask=mn, other=0.0)
            l32 = ps[None, :] * w
            l32 = l32 * sc
            lv = l32.to(tl.float64) * dest[:, None]
            if HAS1:
                acc1 += tl.load(h1_ptr + i) * lv
            if HAS2:
                acc2 += tl.load(h2_ptr + i) * (lv * lv)
        out_base = out_ptr + pid_s.to(tl.int64) * s_os
        if HAS1:
            tl.store(out_base + optr, acc1, mask=tile_mask)
        if HAS2:
            tl.store(out_base + s_oj + optr, acc2, mask=tile_mask)


if triton is not None:

    @triton.jit
    def _im2col_kernel(x_ptr, pat_ptr, psum_ptr, C, H, W, OW, P, P_STORE, KW, KHW, n_rows, s_xn, s_xc, s_xh, s_xw,
                       s_pr, sh, sw, ph, pw, dh, dw, n_pb,
                       ROWS: tl.constexpr, BLOCK: tl.constexpr, ROUND: tl.constexpr, PSUM: tl.constexpr):
        """pat[row, p] = x[i, c, oh*sh - ph + kh*dh, ow*sw - pw + kw*dw] (0 outside), row = (i*C + c)*KHW + kh*KW + kw:
        F.unfold's values (ROUND: rounded to nearest TF32 for the tensor cores); PSUM: fp64 partial sums of the
        unrounded row over this block of positions.  One program per (ROWS rows, block of positions)."""
        rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        mr = rows < n_rows
        i = rows // (C * KHW)
        rem = rows % (C * KHW)
        c = rem // KHW
        tap = rem % KHW
        kh = tap // KW
        kw = tap % KW
        pb = tl.program_id(1)
        offs = pb * BLOCK + tl.arange(0, BLOCK)
        mp = offs < P
        oh = offs // OW
        ow = offs - oh * OW
        ih = oh[None, :] * sh - ph + (kh * dh)[:, None]
        iw = ow[None, :] * sw - pw + (kw * dw)[:, None]
        m = mr[:, None] & mp[None, :]
        inside = m & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
        base = i.to(tl.int64) * s_xn + c * s_xc
        v = tl.load(x_ptr + base[:, None] + ih * s_xh + iw * s_xw, mask=inside, other=0.0)
        if PSUM:
            tl.store(psum_ptr + rows.to(tl.int64) * n_pb + pb, tl.sum(v.to(tl.float64), axis=1), mask=mr)
        if ROUND:
            v = _tf32_rn(v)
        # positions P <= p < P_STORE (the zero tail up to the 16-aligned row length) are written as zeros
        tl.store(pat_ptr + rows.to(tl.int64)[:, None] * s_pr + offs[None, :], v,
                 mask=mr[:, None] & (offs < P_STORE)[None, :])

    @triton.jit
    def _round_rows_kernel(src_ptr, dst_ptr, n_rows, P, Pst, s_sr, s_dr, ROWS: tl.constexpr, BLOCK: tl.constexpr):
        """dst[row, p] = TF32 round-to-nearest of src[row, p] for p < P, 0 for P <= p < Pst."""
        rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mr = rows < n_rows
        v = tl.load(src_ptr + rows.to(tl.int64)[:, None] * s_sr + offs[None, :],
                    mask=mr[:, None] & (offs < P)[None, :], other=0.0)
        tl.store(dst_ptr + rows.to(tl.int64)[:, None] * s_dr + offs[None, :], _tf32_rn(v),
                 mask=mr[:, None] & (offs < Pst)[None, :])


def _sm_count(device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


# Tuned tile configurations per (accumulator bucket, O, K, P) for the conv shapes of AlexNet, VGG-11 and ResNet-18:
# (BLOCK_O, BLOCK_N, BLOCK_P, num_warps, num_stages).  2-slot entries from the H200 sweep,
# 8-slot entries from the H100 sweep (the H100 4-slot entries were slower on H200 and are
# not used); shapes not listed use the rule below.
_A = (64, 32, 64, 4, 3)    # small tiles, 4 warps
_B = (64, 64, 64, 8, 3)
_C = (64, 64, 32, 4, 3)
_D = (128, 64, 32, 8, 3)
_E = (64, 64, 32, 8, 3)
_F = (64, 64, 64, 4, 3)
_G = (128, 64, 64, 8, 4)
_CONFIG_TABLE = {
    # 2 slots (the walk pass: sa, z2_s); round-5 H200 sweep with operands pre-rounded at staging
    (2, 64, 363, 3025): _F, (2, 192, 1600, 729): _F, (2, 384, 1728, 169): _F,           # AlexNet
    (2, 256, 3456, 169): _G, (2, 256, 2304, 169): _G,
    (2, 128, 576, 12544): _G, (2, 256, 1152, 3136): _G, (2, 256, 2304, 3136): _G,        # VGG-11
    (2, 512, 2304, 784): _G, (2, 512, 4608, 784): _F, (2, 512, 4608, 196): _C,
    (2, 64, 147, 12544): _A, (2, 64, 576, 3136): _F, (2, 128, 576, 784): _F,             # ResNet-18
    (2, 128, 1152, 784): _F, (2, 128, 64, 784): (64, 128, 64, 8, 3), (2, 256, 1152, 196): _C,
    (2, 256, 2304, 196): _G, (2, 256, 128, 196): _A, (2, 512, 2304, 49): _A, (2, 512, 4608, 49): _A,
    (2, 512, 256, 49): _A,
    # 8 slots
    (8, 128, 576, 12544): _D, (8, 256, 1152, 3136): _D, (8, 256, 2304, 3136): _D, (8, 512, 2304, 784): _D,
    (8, 64, 27, 50176): (64, 64, 32, 4, 3), (8, 64, 363, 3025): (64, 64, 32, 4, 3), (8, 64, 147, 12544): (64, 64, 32, 4, 3),
    (8, 64, 576, 3136): (64, 64, 32, 4, 3), (8, 256, 128, 196): _A, (8, 512, 256, 49): _A,
}


def pick_config(O: int, K: int, P: int, nslot: int) -> dict:
    """Tile choice per layer shape and accumulator count: the tuned table, else a rule from the same sweeps."""
    bucket = 2 if nslot <= 2 else 4 if nslot <= 4 else 8
    hit = _CONFIG_TABLE.get((bucket, int(O), int(K), int(P)))
    if hit is None:
        bn = 32 if K <= 48 else 64
        bp = 64 if P >= 1024 else 32
        if bucket == 2:
            hit = (128, bn, bp, 8, 3) if O >= 128 else (64, bn, bp, 4, 3)
        elif bucket == 4:
            hit = (64, bn, bp, 8, 3)
        else:
            hit = (64, 64, 32, 8, 3)
    bo, bn, bp, nw, ns = hit
    return dict(BLOCK_O=bo, BLOCK_N=bn, BLOCK_P=bp, num_warps=nw, num_stages=ns)


def pick_slots_per_pass(nslot: int) -> int:
    """fp64 accumulators per pass for the session reducer (H200 measurements in gpu/bench_fused.py --session)."""
    return MAX_SLOTS


def stage_patches(x: torch.Tensor, KH: int, KW: int, stride, padding, dilation,
                  positions_contiguous: bool = True, unpadded: bool = False) -> torch.Tensor:
    """im2col of x as a logical [b, K, Pst] fp32 view (k = c, kh, kw as F.unfold; Pst = P rounded up to 16, zeros
    beyond P): one strided copy from a window view of the zero-padded input (the same values F.unfold produces, no
    arithmetic).  positions_contiguous: [b, K, Pst] storage (TF32 tensor-core operand); otherwise [b, Pst, K16]
    storage (fp32 FMA operand).  unpadded (positions_contiguous only): F.unfold's own [b, K, P] layout instead."""
    b, C, H, W = (int(v) for v in x.shape)
    (sh, sw), (ph, pw), (dh, dw) = stride, padding, dilation
    OH, OW = _out_size(H, KH, sh, ph, dh), _out_size(W, KW, sw, pw, dw)
    P, K = OH * OW, C * KH * KW
    Pst = _round16(P)
    if positions_contiguous and triton is not None and x.is_cuda:
        xf = x if x.dtype == torch.float32 else x.to(torch.float32)
        if unpadded:  # F.unfold's own layout, 16 slack elements (see below)
            flat = torch.empty(b * K * P + 16, dtype=torch.float32, device=x.device)
            flat[b * K * P:].zero_()
            pat, row_stride, buf = flat[:b * K * P].view(b, K, P), P, flat
        else:
            buf = torch.empty((b, K, Pst), dtype=torch.float32, device=x.device)
            if Pst > P:
                buf[:, :, P:].zero_()
            pat, row_stride = buf, Pst
        block = 1024 if P >= 1024 else triton.next_power_of_2(P)
        rows = max(1, 2048 // block)  # short rows: several per program
        _im2col_kernel[(triton.cdiv(b * K, rows), triton.cdiv(P, block))](
            xf, buf, buf, C, H, W, OW, P, P, KW, KH * KW, b * K, xf.stride(0), xf.stride(1), xf.stride(2),
            xf.stride(3), row_stride, sh, sw, ph, pw, dh, dw, 1, ROWS=rows, BLOCK=block, ROUND=False, PSUM=False,
            num_warps=4)
        return pat
    xp = F.pad(x.to(torch.float32), (pw, pw, ph, ph)).contiguous()
    s0, s1, s2, s3 = xp.stride()
    window = xp.as_strided((b, C, KH, KW, OH, OW), (s0, s1, dh * s2, dw * s3, sh * s2, sw * s3))
    if positions_contiguous and unpadded:
        # exactly F.unfold's layout [b, K, P] (so .sum(-1) is bitwise the engine's patch sum), 16 slack elements so a
        # kernel loop over Pst positions can read past the last row (those products meet a zero delta tail)
        flat = torch.empty(b * K * P + 16, dtype=torch.float32, device=x.device)
        flat[b * K * P:].zero_()
        pat = flat[:b * K * P].view(b, K, P)
        pat.view(b, C, KH, KW, OH, OW).copy_(window)
        return pat
    if positions_contiguous:
        buf = torch.empty((b, K, Pst), dtype=torch.float32, device=x.device)
        if Pst > P:
            buf[:, :, P:].zero_()
        buf[:, :, :P].view(b, C, KH, KW, OH, OW).copy_(window)
        return buf
    buf = torch.empty((b, Pst, _round16(K)), dtype=torch.float32, device=x.device)
    if Pst > P:
        buf[:, P:, :].zero_()
    buf[:, :P, :K].view(b, OH, OW, C, KH, KW).copy_(window.permute(0, 4, 5, 1, 2, 3))
    return buf[:, :, :K].transpose(1, 2)


def stage_tf32(x: torch.Tensor, delta: torch.Tensor, KH: int, KW: int, stride, padding, dilation,
               want_psum: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """TF32-policy staging: im2col patches [b, K, Pst] and delta [b, O, Pst], both rounded to nearest TF32 (what the
    tensor cores then multiply exactly; zero tails beyond P), and, with want_psum, the local head's patch sums [b, K]
    as fp32 roundings of fp64 sums of the unrounded patch values (deterministic: per-block fp64 partials, then a
    fixed-order fp64 sum)."""
    b, C, H, W = (int(v) for v in x.shape)
    O = int(delta.shape[1])
    (sh, sw), (ph, pw), (dh, dw) = stride, padding, dilation
    OH, OW = _out_size(H, KH, sh, ph, dh), _out_size(W, KW, sw, pw, dw)
    P, K = OH * OW, C * KH * KW
    Pst = _round16(P)
    xf = x if x.dtype == torch.float32 else x.to(torch.float32)
    pat = torch.empty((b, K, Pst), dtype=torch.float32, device=x.device)  # every element written by the kernel
    block = min(IM2COL_BLOCK_MAX, triton.next_power_of_2(Pst))
    rows = max(1, IM2COL_ELEMS // block)
    n_pb = triton.cdiv(Pst, block)
    psum = torch.empty((b * K, n_pb), dtype=torch.float64, device=x.device) if want_psum else pat
    _im2col_kernel[(triton.cdiv(b * K, rows), n_pb)](
        xf, pat, psum, C, H, W, OW, P, Pst, KW, KH * KW, b * K, xf.stride(0), xf.stride(1), xf.stride(2),
        xf.stride(3), Pst, sh, sw, ph, pw, dh, dw, n_pb, ROWS=rows, BLOCK=block, ROUND=True, PSUM=want_psum,
        num_warps=IM2COL_WARPS)
    if want_psum:
        ps = (psum[:, 0] if n_pb == 1 else psum.sum(dim=1)).to(torch.float32).reshape(b, K)
    else:
        ps = None
    d = delta.reshape(b * O, P)
    if d.stride(1) != 1:
        d = d.contiguous()
    dp = torch.empty((b, O, Pst), dtype=torch.float32, device=x.device)
    dblock = 1024 if Pst >= 1024 else triton.next_power_of_2(Pst)
    drows = max(1, 2048 // dblock)
    _round_rows_kernel[(triton.cdiv(b * O, drows), triton.cdiv(Pst, dblock))](
        d, dp, b * O, P, Pst, d.stride(0), Pst, ROWS=drows, BLOCK=dblock, num_warps=4)
    return pat, dp, ps


def _launch_rows(pat, dp, w2d, a_t, coef, kinds, config, programs_target):
    """One pass over an image chunk: fp64 [len(kinds), O*K] sums over the chunk's images."""
    b, K = int(pat.shape[0]), int(pat.shape[1])
    P = int(dp.shape[2])  # loop bound: a multiple of 16; delta is zero beyond the real positions
    O = int(w2d.shape[0])
    bo, bn, bp = config["BLOCK_O"], config["BLOCK_N"], config["BLOCK_P"]
    tiles = triton.cdiv(O, bo) * triton.cdiv(K, bn)
    splits = max(1, min(b, programs_target // tiles))
    ipp = triton.cdiv(b, splits)
    splits = triton.cdiv(b, ipp)
    ns = len(kinds)
    out = torch.empty((splits, ns, O * K), dtype=torch.float64, device=pat.device)
    vt = list(kinds) + [-1] * (MAX_SLOTS - ns)
    _fused_conv_census_kernel[(tiles, splits)](
        pat, dp, w2d, a_t, coef, out,
        b, O, K, P, ipp,
        pat.stride(0), pat.stride(1), pat.stride(2), dp.stride(0), dp.stride(1),
        coef.stride(0), out.stride(0), out.stride(1),
        VT0=vt[0], VT1=vt[1], VT2=vt[2], VT3=vt[3], VT4=vt[4], VT5=vt[5], VT6=vt[6], VT7=vt[7],
        BLOCK_O=bo, BLOCK_N=bn, BLOCK_P=bp, TF32=bool(torch.backends.cuda.matmul.allow_tf32),
        num_warps=config["num_warps"], num_stages=config["num_stages"])
    return out[0] if splits == 1 else out.sum(dim=0)


def _launch_local(ps, w2d, dest, sc_t, h1, h2, programs_target):
    """Local head over an image chunk: fp64 [2, O*K] (sum h1 l, sum h2 l^2; a missing weight gives zeros)."""
    b, K = (int(v) for v in ps.shape)
    O = int(w2d.shape[0])
    bo, bn = 64, 64
    tiles = triton.cdiv(O, bo) * triton.cdiv(K, bn)
    splits = max(1, min(b, programs_target // tiles))
    ipp = triton.cdiv(b, splits)
    splits = triton.cdiv(b, ipp)
    alloc = torch.empty if (h1 is not None and h2 is not None) else torch.zeros  # both slots written by the kernel
    out = alloc((splits, 2, O * K), dtype=torch.float64, device=ps.device)
    dummy = dest
    _local_head_kernel[(tiles, splits)](
        ps, w2d, dest, sc_t, h1 if h1 is not None else dummy, h2 if h2 is not None else dummy, out,
        b, O, K, ipp, ps.stride(0), out.stride(0), out.stride(1),
        HAS1=h1 is not None, HAS2=h2 is not None, BLOCK_O=bo, BLOCK_N=bn, num_warps=4, num_stages=1)
    return out[0] if splits == 1 else out.sum(dim=0)


def conv_row_sums(x: torch.Tensor, delta: torch.Tensor, weight: torch.Tensor, *, scale: Any, stride, padding,
                  dilation, a: float, slots: Sequence[tuple[int, torch.Tensor]] = (),
                  local: Mapping[str, Any] | None = None, out: Sequence[torch.Tensor] | None = None,
                  local_out: Mapping[str, torch.Tensor] | None = None, config: dict | None = None,
                  max_slots: int = MAX_SLOTS, programs_per_sm: int = 4) -> tuple[list, dict]:
    """The core.  slots: [(value kind 0..3, per-image fp64 coefficients [n])]; local: {"dest": [O] fp64,
    "h1": [n] or None, "h2": [n] or None} for the local head (scale is the raw producer scale).
    Returns ([O*K] fp64 sum per slot, {"l1": [O*K], "l2": [O*K]} for the requested local sums); with out /
    local_out the sums are added into those caller-supplied fp64 buffers ([O*K] views) and returned."""
    if triton is None:
        raise RuntimeError("fused_conv_census needs Triton and a CUDA device")
    assert x.is_cuda and delta.is_cuda, "fused conv census runs on the GPU"
    O, C, KH, KW = (int(v) for v in weight.shape)
    K = C * KH * KW
    n = int(x.shape[0])
    device = x.device
    stride, padding, dilation = (tuple(int(v) for v in t) for t in (stride, padding, dilation))
    H, W = int(x.shape[2]), int(x.shape[3])
    OH = _out_size(H, KH, stride[0], padding[0], dilation[0])
    OW = _out_size(W, KW, stride[1], padding[1], dilation[1])
    P, Pst = OH * OW, _round16(OH * OW)
    if tuple(delta.shape) != (n, O, OH, OW):
        raise ValueError(f"delta shape {tuple(delta.shape)} does not match the layer output {(n, O, OH, OW)}")
    raw = float(scale.detach().cpu().item()) if torch.is_tensor(scale) else float(scale)
    sp = raw if abs(raw) >= 1e-12 else 1.0
    sums = list(out) if out is not None else [torch.zeros(O * K, dtype=torch.float64, device=device) for _ in slots]
    want_local = local is not None and (local.get("h1") is not None or local.get("h2") is not None)
    lsums = dict(local_out) if local_out is not None else {}
    if want_local:
        for key, name in (("h1", "l1"), ("h2", "l2")):
            if local.get(key) is not None and name not in lsums:
                lsums[name] = torch.zeros(O * K, dtype=torch.float64, device=device)
    if n == 0 or (not slots and not want_local):
        return sums, lsums
    w2d = weight.detach().to(device=device, dtype=torch.float32).reshape(O, K).contiguous()
    programs_target = programs_per_sm * _sm_count(device)
    tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    if slots:
        coef_all = torch.stack([torch.as_tensor(cf, dtype=torch.float64, device=device).reshape(-1)
                                for _, cf in slots])
        if coef_all.shape[1] != n:
            raise ValueError("per-image coefficients differ from the packet's image axis")
        kinds_all = [int(k) for k, _ in slots]
        groups = [list(range(s, min(len(slots), s + max_slots))) for s in range(0, len(slots), max_slots)]
        cfgs = [config or pick_config(O, K, P, len(idx)) for idx in groups]
        a_t = torch.full((1,), float(a), dtype=torch.float64, device=device)  # no host copy (no stream sync)
    if want_local:
        dest = torch.as_tensor(local["dest"], dtype=torch.float64, device=device).reshape(-1).contiguous()
        sc_t = torch.full((1,), raw, dtype=torch.float32, device=device)
        h1 = None if local.get("h1") is None else torch.as_tensor(local["h1"], dtype=torch.float64, device=device)
        h2 = None if local.get("h2") is None else torch.as_tensor(local["h2"], dtype=torch.float64, device=device)
    per_image = 4 * (_round16(K) + O) * Pst
    chunk = max(1, min(n, STAGE_BUDGET_BYTES // per_image))
    bounds = [(s0, min(n, s0 + chunk)) for s0 in range(0, n, chunk)]

    def stage(start, stop):
        """Staging copies of the same fp32 values: im2col patches; delta / sp, both over Pst = P rounded up to 16
        positions with zero tails (delta is used in place when P is already a multiple of 16); the engine-exact
        patch sums of the local head.  With the local head on a layer whose P is not a multiple of 16, one unpadded
        F.unfold-layout buffer serves both the kernel and the patch sums."""
        b = stop - start
        xs = x[start:stop]
        pat = dp = ps = None
        if slots and tf32:
            d = delta[start:stop].to(torch.float32)
            if sp != 1.0:
                d = d / torch.full((), sp, dtype=torch.float32, device=device)  # as the engine: tensor divisor
            pat, dp, ps = stage_tf32(xs, d, KH, KW, stride, padding, dilation, want_psum=want_local)
            return pat, dp, ps
        if slots:
            unpadded = want_local and tf32 and Pst != P
            pat = stage_patches(xs, KH, KW, stride, padding, dilation, positions_contiguous=tf32, unpadded=unpadded)
            d = delta[start:stop].to(torch.float32)
            if sp != 1.0:
                d = d / torch.full((), sp, dtype=torch.float32, device=device)  # as the engine: tensor divisor
            d = d.reshape(b, O, P)
            if Pst == P and d.stride(2) == 1 and d.stride(1) == P:
                dp = d
            else:
                dp = torch.empty((b, O, Pst), dtype=torch.float32, device=device)
                dp[:, :, P:].zero_()
                dp[:, :, :P].copy_(d)
        if want_local:
            if pat is not None and tf32 and pat.shape[2] == P and pat.is_contiguous():
                ps = pat.sum(dim=-1)  # the engine's unfold(...).sum(-1): same values, shape and layout
            else:
                ps = F.unfold(xs.to(torch.float32), kernel_size=(KH, KW), dilation=dilation, padding=padding,
                              stride=stride).reshape(b, 1, C, KH * KW, P).sum(dim=-1).reshape(b, K)
            ps = ps.contiguous()
        return pat, dp, ps

    def consume(start, stop, pat, dp, ps):
        if slots:
            for idx, cfg in zip(groups, cfgs):
                coef = coef_all[idx[0]:idx[-1] + 1, start:stop].contiguous()
                part = _launch_rows(pat, dp, w2d, a_t, coef, kinds_all[idx[0]:idx[-1] + 1], cfg, programs_target)
                for j, row in zip(idx, part):
                    sums[j].add_(row)
        if want_local:
            part = _launch_local(ps, w2d, dest, sc_t, None if h1 is None else h1[start:stop].contiguous(),
                                 None if h2 is None else h2[start:stop].contiguous(), programs_target)
            if h1 is not None:
                lsums["l1"].add_(part[0])
            if h2 is not None:
                lsums["l2"].add_(part[1])

    for start, stop in bounds:
        consume(start, stop, *stage(start, stop))
    return sums, lsums


# ----------------------------------------------------------------------------------------------------------------
# The FastBatchCensusSession slot
# ----------------------------------------------------------------------------------------------------------------


WALK_CRITICAL = ("sa", "z2_s")  # the order field and its per-image square (the walk's SE): first pass


def row_reducer(blk: Any, grad: torch.Tensor, c: torch.Tensor, weights: Mapping[str, torch.Tensor],
                stats: Mapping[str, str], a: float, dest: torch.Tensor,
                out: Mapping[str, torch.Tensor] | None = None,
                max_slots: int | None = None) -> dict[str, torch.Tensor]:
    """census_sufficient_statistics.FastBatchCensusSession row_reducer: {name: [R_layer] fp64} for one conv block.

    blk: the engine's captured conv block (input_source, weight, scale, reducer_parameters); grad: [b, O, H, W];
    c: [b] loss multipliers; weights: {name: [b] fp64}; stats: {name: stat in g1 ga g2 g4 s1 sa s2 s4 l1 l2};
    a: the affine scale; dest: [O] local destination scale.  out (optional): {name: [R_layer] fp64} buffers the sums
    are added into (returned in place of new tensors).
    Returns None, before touching any buffer, for a block it does not support (groups != 1, no Triton/CUDA, a
    statistic outside the list, a local statistic requested twice) and for a single-image packet: for one image the
    engine's einsum runs a non-batched cuBLAS GEMM whose spatial sum takes a different (split) order, while from two
    images on its batched GEMM sums in the kernel's sequential order (measured bitwise equal); the session's operator
    path then gives the engine's own numbers for that image."""
    params = getattr(blk, "reducer_parameters", None) or {}
    x = getattr(blk, "input_source", None)
    weight = getattr(blk, "weight", None)
    if (triton is None or not torch.is_tensor(x) or not torch.is_tensor(weight) or not x.is_cuda
            or not grad.is_cuda or x.ndim != 4 or weight.ndim != 4 or int(params.get("groups", 0)) != 1
            or any(key not in params for key in ("stride", "padding", "dilation")) or int(grad.shape[0]) < 2):
        return None
    local_seen = [stat for stat in stats.values() if stat in LOCAL_STATS]
    if any(stat not in VALUE_KIND and stat not in LOCAL_STATS for stat in stats.values()) or \
            len(local_seen) != len(set(local_seen)):
        return None
    pair = lambda v: tuple(int(t) for t in v) if isinstance(v, (tuple, list)) else (int(v), int(v))  # noqa: E731
    device = grad.device
    # every launch goes to the caller's current stream (ordered after whatever produced the inputs on it); if the
    # inputs were allocated on another stream, the allocator must not hand their memory to that stream again until
    # this stream's reads are done
    stream = torch.cuda.current_stream(device)
    for t in (x, grad, weight):
        if t.is_cuda:
            t.record_stream(stream)
    cc = torch.as_tensor(c, dtype=torch.float64, device=device).reshape(-1)
    slot_names, slots, local = [], [], {"dest": dest, "h1": None, "h2": None}
    local_names = {}
    ordered = sorted(stats, key=lambda nm: (0, WALK_CRITICAL.index(nm)) if nm in WALK_CRITICAL else (1, 0))
    for name in ordered:
        stat = stats[name]
        h = torch.as_tensor(weights[name], dtype=torch.float64, device=device).reshape(-1)
        if stat in VALUE_KIND:
            slot_names.append(name)
            slots.append((VALUE_KIND[stat], coefficient(stat, h, cc)))
        else:
            local["h1" if stat == "l1" else "h2"] = h
            local_names[stat] = name
    R = int(weight.numel())
    slot_out = None if out is None else [out[name].reshape(-1) for name in slot_names]
    local_out = None if out is None else {stat: out[name].reshape(-1) for stat, name in local_names.items()}
    sums, lsums = conv_row_sums(x, grad, weight, scale=blk.scale, stride=pair(params["stride"]),
                                padding=pair(params["padding"]), dilation=pair(params["dilation"]), a=a,
                                slots=slots, local=local if local_names else None, out=slot_out,
                                local_out=local_out, max_slots=max_slots or pick_slots_per_pass(len(slots)))
    if out is not None:  # the caller's own buffer objects, now holding their previous value plus these sums
        return {name: out[name] for name in stats}
    result = {name: s for name, s in zip(slot_names, sums)}
    result.update({name: lsums[stat] for stat, name in local_names.items()})
    for name in result:
        assert result[name].numel() == R
    return {name: result[name] for name in stats}


# ----------------------------------------------------------------------------------------------------------------
# Drop-in for fast_census.conv_stats (duck-typed RouteLayer)
# ----------------------------------------------------------------------------------------------------------------


def _image_weight(image_weights, name, n, device):
    """Per-image weight of one statistic by its order (fast_census convention; None = unit weights)."""
    if image_weights is None or STAT_ORDER[name] not in image_weights:
        return None
    value = torch.as_tensor(image_weights[STAT_ORDER[name]], dtype=torch.float64, device=device).reshape(-1)
    if value.shape[0] != n:
        raise ValueError("image weights differ from the packet's image axis")
    return value


def conv_stats(layer: Any, x: torch.Tensor, delta: torch.Tensor, c: torch.Tensor, a: float,
               stats: Sequence[str], image_chunk: int = 128,
               image_weights: Mapping[str, torch.Tensor] | None = None, *,
               out: Mapping[str, torch.Tensor] | None = None, config: dict | None = None,
               max_slots: int = MAX_SLOTS, programs_per_sm: int = 4) -> dict[str, torch.Tensor]:
    """fast_census.conv_stats on the fused kernels (same arguments and return; image_chunk is ignored: staging is
    chunked by memory).  layer: weight, producer_scale, stride, padding, dilation, groups, dest_scale.
    out (optional): caller-supplied fp64 buffers {stat: weight-shaped tensor}; the sums are added into them."""
    if int(layer.groups) != 1:
        raise ValueError("fused conv census supports groups = 1 only")
    shape = tuple(int(v) for v in layer.weight.shape)
    n = int(x.shape[0])
    device = x.device
    if out is not None:
        for name in stats:
            buf = out[name]
            if buf.dtype != torch.float64 or tuple(buf.shape) != shape or buf.device != device:
                raise ValueError(f"out[{name!r}] must be fp64 of shape {shape} on {device}")
        target = dict(out)
    else:
        target = {name: torch.zeros(shape, dtype=torch.float64, device=device) for name in stats}
    cc = torch.as_tensor(c, dtype=torch.float64, device=device).reshape(-1)
    ones = torch.ones_like(cc)
    row = [name for name in stats if name in VALUE_KIND]
    for name in stats:
        if name not in VALUE_KIND and name not in ("l1",):
            raise ValueError(name)
    slots = []
    for name in row:
        h = _image_weight(image_weights, name, n, device)
        slots.append((VALUE_KIND[name], coefficient(name, ones if h is None else h, cc)))
    local = None
    if "l1" in stats:
        h = _image_weight(image_weights, "l1", n, device)
        dest = (torch.ones(shape[0], dtype=torch.float64, device=device) if layer.dest_scale is None else
                torch.as_tensor(layer.dest_scale, dtype=torch.float64, device=device).reshape(-1))
        local = {"dest": dest, "h1": ones if h is None else h, "h2": None}
    conv_row_sums(x, delta, layer.weight, scale=layer.producer_scale, stride=layer.stride, padding=layer.padding,
                  dilation=layer.dilation, a=a, slots=slots, local=local,
                  out=[target[name].view(-1) for name in row],
                  local_out=None if local is None else {"l1": target["l1"].view(-1)},
                  config=config, max_slots=max_slots, programs_per_sm=programs_per_sm)
    return {name: target[name] for name in stats}
