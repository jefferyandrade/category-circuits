"""Animacy and category-overlap statistics of AlexNet's kept connections ("routes") on all 1,000 categories (Fig. 5),
read from a category incidence library (``incidence_library``).

For each layer (``statistics_from_library``): the animate-inanimate specificity of each kept connection with its
label-permutation null (5,000 permutations, 10,000 at fc6 and fc7, in ``nproc`` streams seeded by network and layer),
the participation curves with their max-T envelope and the joint count grid H[k_A, k_I], and the Jaccard overlap of
every category pair against its exact size-matched hypergeometric reference (``extras=True`` also writes the grids and
the overlap by sign). ``exact_J0`` is the size-matched overlap reference, also used by
``analysis/stats/overlap_nulls.py``.

Outputs (the ``semantic`` files hold the animacy statistics): ``semantic_layer_stats_alexnet1000.parquet`` (one row
per layer: observed and null mean |s|, their difference ``delta_pi_abs_s``, the permutation p value ``p_perm`` and
its number of permutations ``B_perms``), ``semantic_curves_alexnet1000.npz`` (participation curves, null and
envelope), ``domain_grids_alexnet1000.npz`` (H[k_A, k_I] of each layer), ``meta_alexnet1000.json`` (the group sizes
and bins), and the overlap statistics (``overlap_layer_stats``, ``pairwise_JJ0``, ``polarity_*``).
"""
from __future__ import annotations

import json
import multiprocessing
import zlib
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from scipy.special import gammaln
from scipy.stats import norm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import prepare_dir, publish_file, write_file

EPS = 1e-9
NSIGMA = 14.0
QEDGES = np.array([0., .05, .1, .15, .2, .3, .4, .5, .6, .8, 1.000001])
NB = len(QEDGES) - 1
MINW = 1000.0
POP8 = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)


def hyp_pmf(xs, E, a, b):
    lg = lambda n: gammaln(np.asarray(n, dtype=np.float64) + 1.0)
    logp = (lg(a) - lg(xs) - lg(a - xs) + lg(E - a) - lg(b - xs)
            - lg(E - a - b + xs) - (lg(E) - lg(b) - lg(E - b)))
    return np.exp(logp)


def pmf_window(E, a, b):
    E, a, b = int(E), int(a), int(b)
    if a == 0 or b == 0:
        return np.array([0]), np.array([1.0])
    mean = a * (b / E)
    var = mean * (1 - a / E) * ((E - b) / max(E - 1, 1))
    half = max(8.0, NSIGMA * np.sqrt(max(var, 1e-12)))
    lo = max(int(np.floor(mean - half)), max(0, a + b - E))
    hi = min(int(np.ceil(mean + half)), min(a, b))
    xs = np.arange(lo, hi + 1)
    pmf = hyp_pmf(xs, E, a, b)
    s = pmf.sum()
    return xs, pmf / s if s > 0 else pmf


@lru_cache(maxsize=300000)
def exact_J0(a, b, E):
    """Windowed exact hypergeometric expectation, including empty sets."""
    a, b, E = int(a), int(b), int(E)
    if a + b == 0:
        return np.nan
    if a == 0 or b == 0:
        return 0.0
    xs, pmf = pmf_window(E, a, b)
    denom = a + b - xs
    vals = np.where(denom > 0, xs / np.maximum(denom, 1), 1.0)
    return float((pmf * vals).sum())


@lru_cache(maxsize=300000)
def J0_sign(ap, bp, Ep, an, bn, En):
    ap, bp, an, bn = int(ap), int(bp), int(an), int(bn)
    a, b = ap + an, bp + bn
    if a + b == 0:
        return np.nan
    if a == 0 or b == 0:
        return 0.0
    xp, pp = pmf_window(Ep, ap, bp)
    xn, pn = pmf_window(En, an, bn)
    if len(xp) == 1:
        xs, pmf = xn + xp[0], pn
    elif len(xn) == 1:
        xs, pmf = xp + xn[0], pp
    else:
        pmf = fftconvolve(pp, pn)
        pmf = np.clip(pmf, 0, None)
        pmf /= pmf.sum()
        xs = np.arange(xp[0] + xn[0], xp[0] + xn[0] + len(pmf))
    denom = a + b - xs
    vals = np.where(denom > 0, xs / np.maximum(denom, 1), 1.0)
    return float((pmf * vals).sum())


def _j0_upper(a, E, iu):
    # Pairs of the same sizes share one exact integral, computed once.
    if not len(iu[0]):
        return np.empty(0, np.float64)
    if E <= 20000:
        sizes = np.column_stack((a[iu[0]], a[iu[1]]))
        uniq, inv = np.unique(sizes, axis=0, return_inverse=True)
        values = np.empty(len(uniq))
        for lo in range(0, len(uniq), 2048):
            sub = uniq[lo:lo + 2048]
            xs, pmf = _pmf_rows(E, sub[:, 0], sub[:, 1])
            denom = sub[:, 0, None] + sub[:, 1, None] - xs
            vals = np.where(denom > 0, xs / np.maximum(denom, 1), 1.)
            v = (pmf * vals).sum(axis=1)
            v[(sub[:, 0] + sub[:, 1]) == 0] = np.nan
            values[lo:lo + len(sub)] = v
        return values[inv]
    return np.fromiter((exact_J0(int(a[i]), int(a[j]), int(E))
                        for i, j in zip(*iu)), dtype=np.float64, count=len(iu[0]))


def _j0_sign_upper(ap, an, Ep, En, iu):
    if Ep + En <= 20000:
        sizes = np.column_stack((ap[iu[0]], ap[iu[1]], an[iu[0]], an[iu[1]]))
        values = np.empty(len(sizes))
        for lo in range(0, len(sizes), 1024):
            sub = sizes[lo:lo + 1024]
            xp, pp = _pmf_rows(Ep, sub[:, 0], sub[:, 1])
            xn, pn = _pmf_rows(En, sub[:, 2], sub[:, 3])
            pmf = fftconvolve(pp, pn, axes=1)
            pmf = np.clip(pmf, 0, None)
            pmf /= pmf.sum(axis=1, keepdims=True)
            xs = xp[:, 0, None] + xn[:, 0, None] + np.arange(pmf.shape[1])[None, :]
            denom = sub.sum(axis=1)[:, None] - xs
            vals = np.where(denom > 0, xs / np.maximum(denom, 1), 1.)
            v = (pmf * vals).sum(axis=1)
            v[sub.sum(axis=1) == 0] = np.nan
            values[lo:lo + len(sub)] = v
        return values
    return np.fromiter((J0_sign(int(ap[i]), int(ap[j]), int(Ep),
                               int(an[i]), int(an[j]), int(En))
                        for i, j in zip(*iu)), dtype=np.float64, count=len(iu[0]))


def _pmf_rows(E, a, b):
    """Batch the same exact scalar windows with zero padding on the right."""
    a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
    if E == 0:
        return np.zeros((len(a), 1), np.int64), np.ones((len(a), 1))
    mean = a * (b / E)
    var = mean * (1 - a / E) * ((E - b) / max(E - 1, 1))
    half = np.maximum(8., NSIGMA * np.sqrt(np.maximum(var, 1e-12)))
    lo = np.maximum(np.floor(mean - half).astype(np.int64), np.maximum(0, a + b - E))
    hi = np.minimum(np.ceil(mean + half).astype(np.int64), np.minimum(a, b))
    empty = (a == 0) | (b == 0)
    lo[empty], hi[empty] = 0, 0
    xs = lo[:, None] + np.arange(int((hi - lo + 1).max()))[None, :]
    valid = xs <= hi[:, None]
    # Evaluate padded points at the valid lower edge to avoid invalid factorials.
    pmf = hyp_pmf(np.where(valid, xs, lo[:, None]), E, a[:, None], b[:, None])
    pmf[~valid] = 0.
    total = pmf.sum(axis=1, keepdims=True)
    pmf /= np.where(total > 0, total, 1.)
    return xs, pmf


def _jackknife(v, iu, K):
    tots, cnts = np.zeros(K), np.zeros(K)
    valid = ~np.isnan(v)
    for arr in iu:
        np.add.at(tots, arr[valid], v[valid])
        np.add.at(cnts, arr[valid], 1)
    loo = (np.nansum(v) - tots) / np.maximum(valid.sum() - cnts, 1)
    return (float(np.nanmean(v)),
            float(np.sqrt((K - 1) / K * ((loo - loo.mean()) ** 2).sum())))


def _holm(frame):
    p = frame.p_two.to_numpy()
    m = np.sum(~np.isnan(p))
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    adj, running = np.full_like(p, np.nan), 0.0
    for rank, idx in enumerate(order):
        if np.isnan(p[idx]):
            continue
        running = max(running, min((m - rank) * p[idx], 1.0))
        adj[idx] = running
    frame['p_holm'] = adj
    return frame


def _animacy(cm):
    col = 'animate_inanimate' if 'animate_inanimate' in cm else 'animate'
    values = cm[col].to_numpy()
    return values.astype(bool) if values.dtype.kind == 'b' else values == 'animate'


def _category_ids(cm):
    col = 'category_id_global' if 'category_id_global' in cm else 'category_id'
    return cm[col].to_numpy(dtype=np.int64)


def _overlap(L, lm, a, ap, an, I, anim, model):
    K = len(a)
    iu = np.triu_indices(K, k=1)
    E, Ep, En = (int(lm[k]) for k in ('eligible_route_count',
                                               'eligible_positive_count', 'eligible_negative_count'))
    S = a[:, None] + a[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        J = np.where(S - I > 0, I / (S - I), np.nan)
    j0u = _j0_upper(a, E, iu)
    j0su = _j0_sign_upper(ap, an, Ep, En, iu)
    ju, exc = J[iu], J[iu] - j0u
    with np.errstate(divide='ignore', invalid='ignore'):
        rbar = np.where(1 - j0u > 0, exc / (1 - j0u), np.nan)
    U, Use = _jackknife(exc, iu, K)
    R, Rse = _jackknife(rbar, iu, K)
    Us, Usse = _jackknife(ju - j0su, iu, K)
    z = U / Use if Use > 0 else np.nan
    cond = {}
    for name, mk in (('within_animate', np.outer(anim, anim)),
                     ('within_inanimate', np.outer(~anim, ~anim)),
                     ('between', np.outer(anim, ~anim) | np.outer(~anim, anim))):
        cond[name] = _jackknife(np.where(mk[iu], exc, np.nan), iu, K)
    row = dict(model=model, layer=L, stage=lm['stage'], eligible=E,
               mean_retained=float(a.mean()), mean_density=float((a / E).mean()),
               mean_J=float(np.nanmean(ju)), median_J=float(np.nanmedian(ju)),
               mean_J0=float(np.nanmean(j0u)), mean_J0_signmatched=float(np.nanmean(j0su)),
               U=U, U_se=Use, z=z, p_two=float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan,
               Rbar=R, Rbar_se=Rse, U_signmatched=Us, U_signmatched_se=Usse,
               ratio=float(np.nanmean(ju) / np.nanmean(j0u)) if np.nanmean(j0u) > 0 else np.nan,
               U_within_animate=cond['within_animate'][0], U_within_animate_se=cond['within_animate'][1],
               U_within_inanimate=cond['within_inanimate'][0], U_within_inanimate_se=cond['within_inanimate'][1],
               U_between=cond['between'][0], U_between_se=cond['between'][1],
               n_pairs_valid=int(np.isfinite(exc).sum()))
    J0 = np.zeros((K, K), np.float32)
    J0[iu] = j0u
    J0 += J0.T
    return row, J.astype(np.float32), J0


def _polarity_overlap(L, pol, E, I):
    a, K = np.diag(I).astype(np.int64), len(I)
    iu = np.triu_indices(K, k=1)
    S = a[:, None] + a[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        J = np.where(S - I > 0, I / (S - I), np.nan)
    j0u = _j0_upper(a, E, iu)
    ju = J[iu]
    U, Use = _jackknife(ju - j0u, iu, K)
    J0 = np.zeros((K, K), np.float32)
    J0[iu] = j0u
    J0 += J0.T
    row = dict(layer=L, polarity=pol, eligible=E, mean_retained=float(a.mean()),
               mean_density=float((a / E).mean()), mean_J=float(np.nanmean(ju)),
               mean_J0=float(np.nanmean(j0u)), U=U, U_se=Use,
               n_pairs_valid=int(np.isfinite(ju - j0u).sum()))
    return row, J.astype(np.float32), J0


@lru_cache(maxsize=4096)
def _struct_null_second_moment(t, nA, nI, eps):
    """Exact cell second moment, shared by repeated participation totals."""
    xs = np.arange(max(0, t - nI), min(nA, t) + 1)
    pmf = hyp_pmf(xs, nA + nI, nA, t)
    pmf /= max(pmf.sum(), 1e-300)
    pAx, pIx = xs / nA, (t - xs) / nI
    sx = (pAx - pIx) / (pAx + pIx + eps)
    return float((pmf * sx * sx).sum())


def _observed_semantic(kA, tot, nA, nI):
    kA, tot = np.asarray(kA, np.float32), np.asarray(tot, np.float32)
    pA, pI = kA / nA, (tot - kA) / nI
    s = (pA - pI) / (pA + pI + EPS)
    q, d = .5 * (pA + pI), np.abs(pA - pI)
    bi = np.clip(np.digitize(q, QEDGES) - 1, 0, NB - 1)
    wsum = np.bincount(bi, minlength=NB).astype(np.float64)
    dsum = np.bincount(bi, weights=d.astype(np.float64), minlength=NB)
    curve = np.where(wsum >= MINW, dsum / np.maximum(wsum, 1e-30), np.nan)
    es2 = np.zeros(len(tot))
    for t in np.unique(tot.astype(np.int64)):
        es2[tot == t] = _struct_null_second_moment(int(t), int(nA), int(nI), EPS)
    struct = float(np.sqrt(max(0.0, float((s ** 2).mean()) - float(es2.mean()))))
    return dict(obs_mean_abs_s=float(np.abs(s).mean()), obs_mean_signed_s=float(s.mean()),
                struct=struct), curve, wsum


def _observed_grid(H, nA, nI):
    # The observed statistics from the joint count grid, without the route order: the per-route quantities are
    # float32 as in the per-route computation, and the counts are accumulated in float64.
    ka, ki = np.nonzero(H)
    w = H[ka, ki].astype(np.float64)
    ka, tot = ka.astype(np.float32), (ka + ki).astype(np.float32)
    pA, pI = ka / nA, (tot - ka) / nI
    s = (pA - pI) / (pA + pI + EPS)
    bi = np.clip(np.digitize(.5 * (pA + pI), QEDGES) - 1, 0, NB - 1)
    wsum = np.bincount(bi, weights=w, minlength=NB)
    dsum = np.bincount(bi, weights=np.abs(pA - pI).astype(np.float64) * w, minlength=NB)
    curve = np.where(wsum >= MINW, dsum / np.maximum(wsum, 1e-30), np.nan)
    exp_sum = 0.
    for t in np.unique(tot.astype(np.int64)):
        exp_sum += _struct_null_second_moment(int(t), int(nA), int(nI), EPS) * w[tot == t].sum()
    mass = w.sum()
    result = dict(obs_mean_abs_s=float((np.abs(s).astype(np.float64) * w).sum() / mass),
                  obs_mean_signed_s=float((s.astype(np.float64) * w).sum() / mass),
                  struct=float(np.sqrt(max(0., ((s*s).astype(np.float64)*w).sum()/mass - exp_sum/mass))))
    return result, curve, wsum


def _semantic_row(model, L, stage, nroutes, nA, nI, observed, seed):
    return dict(model=model, layer=L, stage=stage, n_rows=int(nroutes), total_mult=float(nroutes),
                B_perms=10000 if 'classifier' in L else 5000, seed=int(seed), **observed,
                null_mean_abs_s=np.nan, null_sd_abs_s=np.nan, delta_pi_abs_s=np.nan,
                p_perm=np.nan, mc_se=np.nan, jk_se_obs=np.nan, jk_se_delta=np.nan,
                maxT_crit=np.nan, n_exceed_bins=0)


def row_popcount(sig, wmask=None, chunk=2_000_000):
    out = np.empty(len(sig), dtype=np.int64)
    for lo in range(0, len(sig), chunk):
        m = sig[lo:lo + chunk] if wmask is None else sig[lo:lo + chunk] & wmask[None, :]
        u8 = m.view(np.uint8)
        pc = np.bitwise_count(u8) if hasattr(np, 'bitwise_count') else POP8[u8]
        out[lo:lo + chunk] = pc.reshape(len(m), -1).sum(axis=1, dtype=np.int64)
    return out


def _words(indicator):
    w = np.zeros((len(indicator) + 63) // 64, dtype=np.uint64)
    for i in np.flatnonzero(indicator):
        w[i // 64] |= np.uint64(1) << np.uint64(i % 64)
    return w


def _incidence_chunks(library, L, E, chunk=400_000):
    packed = library.layer_packed(L)
    for lo in range(0, E, chunk):
        hi = min(lo + chunk, E)
        bits = np.unpackbits(packed[:, lo // 8:(hi + 7) // 8], axis=1)[:, :hi - lo]
        yield lo, hi, bits


def _summaries_library(library, L, lm, anim):
    E = int(lm['eligible_route_count'])
    K = len(anim)
    ip, ine = np.zeros((K, K), np.float64), np.zeros((K, K), np.float64)
    sign = library.polarity(L).astype(bool)
    H = np.zeros((int(anim.sum()) + 1, int((~anim).sum()) + 1), np.int64)
    sig = np.zeros((E, (K + 63) // 64), dtype=np.uint64)
    # Routes stay in the layer's weight order, which the float32 sums depend on.
    for lo, hi, bits in _incidence_chunks(library, L, E):
        pos = sign[lo:hi]
        for mk, dest in ((pos, ip), (~pos, ine)):
            f = bits[:, mk].astype(np.float32)
            dest += (f @ f.T).astype(np.float64)
        ka = bits[anim].sum(axis=0, dtype=np.int64)
        ki = bits[~anim].sum(axis=0, dtype=np.int64)
        ret = ka + ki > 0
        np.add.at(H, (ka[ret], ki[ret]), 1)
        # Pack the category bits in little-endian uint64 words.
        padded = np.zeros((hi - lo, sig.shape[1] * 8), dtype=np.uint8)
        cat_bytes = np.packbits(bits.T, bitorder='little', axis=1)
        padded[:, :cat_bytes.shape[1]] = cat_bytes
        sig[lo:hi] = padded.view(np.uint64)
    return np.rint(ip).astype(np.int64), np.rint(ine).astype(np.int64), H, sig


def _permutation_streams(B, seed, generator, nproc):
    """The (seed, number of permutations) of each stream: one stream, or nproc streams of near-equal size."""
    if generator == 'single':
        streams = [(seed, B)]
    elif generator == 'blocks':
        if nproc < 1:
            raise ValueError('nproc must be positive')
        streams = [(seed + 1000 + w, B // nproc + int(w < B % nproc)) for w in range(nproc)]
    else:
        raise ValueError("generator must be 'blocks' or 'single'")
    for stream_seed, count in streams:
        if count:
            yield stream_seed, count


def _permutation_seeds(K, nA, B, seed, generator, nproc):
    for stream_seed, count in _permutation_streams(B, seed, generator, nproc):
        rng = np.random.default_rng(stream_seed)
        for _ in range(count):
            p = np.zeros(K, dtype=np.float32)
            p[rng.permutation(K)[:nA]] = 1
            yield p


def _null_bits(sig, K):
    if len(sig) * K <= 40_000_000:
        return np.unpackbits(sig.view(np.uint8), bitorder='little', axis=1)[:, :K].astype(np.float32)
    return None


def _evaluate_semantic_null(sig, tot, nA, nI, draws, N, generator, bits):
    mabs = np.empty(N)
    curves = np.full((N, NB), np.nan)
    # For small libraries matrix multiplication is exact for these 0/1 sums
    # (at most K). It changes no random draw or per-route float32 arithmetic.
    batch = 128 if bits is not None else 1
    edges = QEDGES[1:-1].astype(np.float32) if generator == 'single' else None
    for start in range(0, N, batch):
        indicators = np.stack([next(draws) for _ in range(min(batch, N - start))], axis=1)
        if bits is not None:
            ka = bits @ indicators
        else:
            ka = row_popcount(sig, _words(indicators[:, 0].astype(bool))).astype(np.float32)[:, None]
        pA = ka / nA
        pI = (tot[:, None] - ka) / nI
        ss = pA + pI
        d = np.abs(pA - pI)
        sval = d / (ss + EPS)
        if generator == 'blocks':
            # Each draw's mean is taken over a contiguous float32 copy in route
            # order. A strided axis reduction chooses a different NumPy
            # summation path and can change the last few bits.
            means = np.array([np.ascontiguousarray(sval[:, j]).mean(dtype=np.float32)
                              for j in range(indicators.shape[1])])
        else:
            means = sval.sum(axis=0, dtype=np.float64) / len(tot)
        mabs[start:start + indicators.shape[1]] = means
        q = .5 * ss
        for j in range(indicators.shape[1]):
            bi = (np.searchsorted(edges, q[:, j], side='right') if edges is not None
                  else np.clip(np.digitize(q[:, j], QEDGES) - 1, 0, NB - 1))
            wsum = np.bincount(bi, minlength=NB).astype(np.float64)
            dsum = np.bincount(bi, weights=d[:, j].astype(np.float64), minlength=NB)
            curves[start + j] = np.where(wsum >= MINW, dsum / np.maximum(wsum, 1e-30), np.nan)
    return mabs, curves


def _semantic_null_serial(sig, tot, nA, nI, B, seed, generator, nproc):
    """The null in one process, the streams in order (also the single stream)."""
    draws = _permutation_seeds(nA + nI, nA, B, seed, generator, nproc)
    return _evaluate_semantic_null(sig, tot, nA, nI, draws, B, generator,
                                   _null_bits(sig, nA + nI))


_NULL_WORKER_STATE = None


def _init_null_worker(sig, tot, bits, nA, nI):
    # Explicit fork inherits these arrays once, with shared read-only pages.
    # Only the per-draw arrays are allocated independently in each worker.
    global _NULL_WORKER_STATE
    _NULL_WORKER_STATE = sig, tot, bits, nA, nI


def _semantic_null_block(stream):
    sig, tot, bits, nA, nI = _NULL_WORKER_STATE
    seed, count = stream
    draws = _permutation_seeds(nA + nI, nA, count, seed, 'single', 1)
    return _evaluate_semantic_null(sig, tot, nA, nI, draws, count, 'blocks', bits)


def _semantic_null(sig, tot, nA, nI, B, seed, generator, nproc):
    if generator != 'blocks' or nproc == 1:
        return _semantic_null_serial(sig, tot, nA, nI, B, seed, generator, nproc)
    streams = list(_permutation_streams(B, seed, generator, nproc))
    if not streams:
        return np.empty(0), np.full((0, NB), np.nan)
    bits = _null_bits(sig, nA + nI)
    with ProcessPoolExecutor(max_workers=min(nproc, len(streams)),
                             mp_context=multiprocessing.get_context('fork'),
                             initializer=_init_null_worker,
                             initargs=(sig, tot, bits, nA, nI)) as pool:
        # map yields in input order and propagates worker exceptions; a failed
        # stream must never become a shorter permutation population.
        results = list(pool.map(_semantic_null_block, streams))
    return np.concatenate([r[0] for r in results]), np.concatenate([r[1] for r in results])


def _semantic_jackknife(sig, kA, tot, anim, nA, nI, generator):
    K = len(anim)
    loo = np.empty(K)
    for c in range(K):
        bc = ((sig[:, c // 64] >> np.uint64(c % 64)) & np.uint64(1)).astype(np.float32)
        kc = kA - bc if anim[c] else kA
        nAc, nIc = nA - int(anim[c]), nI - int(not anim[c])
        totc = tot - bc
        keep = totc > 0
        pA, pI = kc[keep] / nAc, (totc[keep] - kc[keep]) / nIc
        s = np.abs(pA - pI) / (pA + pI + EPS)
        loo[c] = float(s.mean()) if generator == 'blocks' else float(s.sum(dtype=np.float64) / keep.sum())
    return float(np.sqrt((K - 1) / K * ((loo - loo.mean()) ** 2).sum()))


def _library_semantic(sig, anim, model, L, stage, generator, nproc):
    nA, nI = int(anim.sum()), int((~anim).sum())
    ka = row_popcount(sig, _words(anim)).astype(np.float32)
    tot = row_popcount(sig).astype(np.float32)
    ret = tot > 0
    if not ret.all():
        sig, ka, tot = np.ascontiguousarray(sig[ret]), ka[ret], tot[ret]
    seed_text = f'{model}|{L}'
    seed = zlib.crc32(seed_text.encode()) & 0xffffffff
    # The permutation counts are _semantic_row's: 10,000 at the classifier layers, 5,000 elsewhere.
    absent = []
    if not len(sig):
        absent.append('No retained routes in this layer')
    if not nA:
        absent.append('No usable animate categories')
    if not nI:
        absent.append('No usable inanimate categories')
    if absent:
        observed = dict(obs_mean_abs_s=np.nan, obs_mean_signed_s=np.nan, struct=np.nan)
        row = _semantic_row(model, L, stage, len(sig), nA, nI, observed, seed)
        row.update(semantic_status='NOT_FORMED', semantic_reason='; '.join(absent),
                   semantic_jackknife_status='NOT_FORMED', semantic_jackknife_reason='; '.join(absent))
        N = row['B_perms']
        curves = {f'{L}|obs_curve': np.full(NB, np.nan), f'{L}|obs_wsum': np.zeros(NB),
                  f'{L}|null_curve_mean': np.full(NB, np.nan), f'{L}|null_curve_sd': np.full(NB, np.nan),
                  f'{L}|envelope_bound': np.full(NB, np.nan), f'{L}|exceed': np.zeros(NB, np.int8),
                  f'{L}|null_mabs': np.full(N, np.nan, np.float32)}
        return row, curves
    observed, curve, wsum = _observed_semantic(ka, tot, nA, nI)
    row = _semantic_row(model, L, stage, len(sig), nA, nI, observed, seed)
    B = row['B_perms']
    nm, nc = _semantic_null(sig, tot, nA, nI, B, seed, generator, nproc)
    row.update(semantic_status='FORMED', semantic_reason='', semantic_jackknife_status='FORMED',
               semantic_jackknife_reason='')
    if nA < 2 or nI < 2:
        row.update(semantic_jackknife_status='NOT_FORMED',
                   semantic_jackknife_reason='The semantic jackknife requires at least two usable categories in each '
                                             'animacy domain')
    else:
        row['jk_se_obs'] = _semantic_jackknife(sig, ka, tot, anim, nA, nI, generator)
        if not np.isfinite(row['jk_se_obs']):
            row.update(semantic_jackknife_status='NOT_FORMED',
                       semantic_jackknife_reason='A delete-one-category library has no retained routes')
    nmc, nsc = np.nanmean(nc, axis=0), np.nanstd(nc, axis=0, ddof=1)
    bound, exceed = np.full(NB, np.nan), np.zeros(NB, dtype=bool)
    null_mean, null_sd = float(nm.mean()), float(nm.std(ddof=1))
    p = float((1 + (nm >= observed['obs_mean_abs_s']).sum()) / (B + 1))
    row.update(null_mean_abs_s=null_mean, null_sd_abs_s=null_sd,
               delta_pi_abs_s=observed['obs_mean_abs_s'] - null_mean, p_perm=p,
               mc_se=float(np.sqrt(p * (1 - p) / B)))
    valid = np.isfinite(nmc) & (nsc > 0) & np.isfinite(curve)
    if valid.any():
        Tm = (nc[:, valid] - nmc[valid]) / nsc[valid]
        crit = float(np.nanquantile(np.nanmax(Tm, axis=1), .95))
        bound = nmc + crit * nsc
        exceed = valid & (curve > bound)
        row['maxT_crit'] = crit
    row['n_exceed_bins'] = int(exceed.sum())
    curves = {f'{L}|obs_curve': curve, f'{L}|obs_wsum': wsum,
              f'{L}|null_curve_mean': nmc, f'{L}|null_curve_sd': nsc,
              f'{L}|envelope_bound': bound, f'{L}|exceed': exceed.astype(np.int8),
              f'{L}|null_mabs': nm.astype(np.float32)}
    return row, curves


def statistics_from_library(library_dir, out_dir, *, model_tag, generator, nproc=40,
                            layers=None, extras=False):
    """Write the statistics of every layer of the incidence library in ``library_dir`` to ``out_dir``."""
    from incidence_library import load_library
    out = Path(out_dir)
    prepare_dir(out, [library_dir])
    library = load_library(library_dir)
    all1000 = model_tag == 'alexnet1000'
    suffix = '_alexnet1000' if all1000 else ''
    lm = library.layer_manifest.copy()
    cm = library.category_manifest.copy()
    layers = list(layers) if layers is not None else list(lm.layer)
    anim, cats = _animacy(cm), _category_ids(cm)
    counts = library.counts.copy()
    counts['model'] = 'alexnet' if all1000 else model_tag
    lm['model'] = 'alexnet' if all1000 else model_tag
    lm = lm[['model', 'layer', 'stage', 'eligible_route_count', 'eligible_positive_count', 'eligible_negative_count']]
    overlaps, semantics, polarities = [], [], []
    pairs, ppairs, gstore, curves = {}, {}, {}, {'qedges': QEDGES}
    for L in layers:
        row = lm[lm.layer == L].iloc[0]
        c = counts[counts.layer == L].set_index('category_id').loc[cats]
        a, ap, an = (c[k].to_numpy(dtype=np.int64) for k in
                    ('retained_count', 'retained_positive_count', 'retained_negative_count'))
        ip, ine, H, sig = _summaries_library(library, L, row, anim)
        if not np.array_equal(a, np.diag(ip + ine)):
            raise ValueError(f'{L}: incidence counts differ from category counts')
        ov, J, J0 = _overlap(L, row, a, ap, an, ip + ine, anim, model_tag)
        overlaps.append(ov)
        pairs.update({f'{L}|J': J, f'{L}|J0': J0})
        se, cs = _library_semantic(sig, anim, model_tag, L, row['stage'], generator, nproc)
        del sig
        semantics.append(se)
        curves.update(cs)
        if extras:
            gstore[f'{L}|H'] = H
            for pol, I, key in (('pos', ip, 'eligible_positive_count'), ('neg', ine, 'eligible_negative_count')):
                pr, pJ, pJ0 = _polarity_overlap(L, pol, int(row[key]), I)
                polarities.append(pr)
                ppairs.update({f'{L}|I_{pol}': I, f'{L}|J_{pol}': pJ, f'{L}|J0_{pol}': pJ0})
    with write_file(out / f'overlap_layer_stats{suffix}.parquet', 'wb') as stream:
        _holm(pd.DataFrame(overlaps)).to_parquet(stream)
    with write_file(out / f'semantic_layer_stats{suffix}.parquet', 'wb') as stream:
        pd.DataFrame(semantics).to_parquet(stream)
    with write_file(out / f'pairwise_JJ0{suffix}.npz', "wb") as stream:
        np.savez_compressed(stream, **pairs)
    with write_file(out / f'semantic_curves{suffix}.npz', "wb") as stream:
        np.savez_compressed(stream, **curves)
    with write_file(out / f'category_layer_counts{suffix}.parquet', 'wb') as stream:
        counts[counts.layer.isin(layers)].to_parquet(stream)
    with write_file(out / f'layer_manifest{suffix}.csv', 'w', newline='') as stream:
        lm[lm.layer.isin(layers)].to_csv(stream, index=False)
    with write_file(out / f'category_manifest{suffix}.csv', 'w', newline='') as stream:
        cm[['category_id_global', 'animate_inanimate']].to_csv(stream, index=False)
    meta = dict(K=len(anim), n_animate=int(anim.sum()), n_inanimate=int((~anim).sum()),
                q_edges=QEDGES.tolist(), eps=EPS, min_bin_weight=MINW,
                tag=library.metadata.get('tag', model_tag),
                animacy_source=library.metadata.get('animacy_source', ''),
                note='multiplicity=1 per route (no signature grouping at K=1000)')
    if not all1000:
        meta['arm'] = model_tag.removeprefix('alexnet_')
    if all1000 or not extras:
        publish_file(json.dumps(meta, indent=1) + '\n', out / (f'meta{suffix}.json' if all1000 else 'meta.json'))
    if extras:
        with write_file(out / 'domain_grids_alexnet1000.npz', "wb") as stream:
            np.savez_compressed(stream, **gstore)
        with write_file(out / 'polarity_pairwise_alexnet1000.npz', "wb") as stream:
            np.savez_compressed(stream, **ppairs)
        with write_file(out / 'polarity_overlap_stats_alexnet1000.parquet', 'wb') as stream:
            pd.DataFrame(polarities).to_parquet(stream)
        extra = dict(K=len(anim), n_animate=int(anim.sum()), n_inanimate=int((~anim).sum()),
                     tag=meta['tag'], layers=layers, animacy_source=meta['animacy_source'],
                     note='grid H[kA,kI] counts retained routes by animate/inanimate category counts; '
                          'polarity split uses sign(sum kernel weights) >= 0')
        if library.axis == 'kernel_weight':
            extra['note'] = ('grid H[kA,kI] counts retained routes by animate/inanimate category counts; '
                             'polarity split uses each weight >= 0')
        publish_file(json.dumps(extra, indent=1) + '\n', out / 'meta.json')
    return out
