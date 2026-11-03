"""Animate-inanimate specificity of each layer's kept connections on the shared categories (Fig. 5, cross-network
panel), with its label-permutation null and delete-one-category jackknife, computed from cell histograms of the
incidence signatures: 5,000 permutations, 10,000 at the classifier layers and the ResNets' last stage, seeded by
network and layer."""
import zlib
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from incidence_package import words_from_flags
from cell_histograms import cell_histograms, ordered_sum

EPS = 1e-9
QEDGES = np.array([0., .05, .1, .15, .2, .3, .4, .5, .6, .8, 1.000001])
MINW = 1000.


def cell_values(K, nA, dtype=np.float64):
    t, a = np.meshgrid(np.arange(K + 1), np.arange(nA + 1), indexing='ij')
    pA = a.astype(dtype) / nA
    pI = (t - a).astype(dtype) / (K - nA)
    s = (pA - pI) / (pA + pI + EPS)
    d = np.abs(pA - pI)
    bi = np.clip(np.digitize(.5 * (pA + pI), QEDGES) - 1, 0, len(QEDGES) - 2)
    return s, d, bi


def summarize(H, K, A, dtype=np.float64):
    s, d, bi = cell_values(K, A, dtype)
    W = H.sum()
    wsum = np.bincount(bi.ravel(), weights=H.ravel(), minlength=10)
    dsum = np.bincount(bi.ravel(), weights=(H * d).ravel(), minlength=10)
    curve = np.where(wsum >= MINW, dsum / np.maximum(wsum, 1e-30), np.nan)
    return float((H * np.abs(s)).sum() / W), float((H * s).sum() / W), curve, wsum


def permutation_words(rng, K, A, B, *, columns=None, word_count=None):
    columns = np.arange(K) if columns is None else np.asarray(columns)
    NW = (K + 63) // 64 if word_count is None else word_count
    out = np.zeros((B, NW), np.uint64)
    for b in range(B):
        for c in columns[rng.permutation(K)[:A]]:
            out[b, c // 64] |= np.uint64(1) << np.uint64(c % 64)
    return out


def null_statistics(words, mult, total, K, A, B, rng, *, columns=None, curves=True, ordered=True):
    mabs = np.empty(B)
    cc = np.empty((B, 10)) if curves else None
    s, d, bi = cell_values(K, A, np.float32)
    flat_s = np.abs(s).ravel().astype(np.float64)
    bin_indices = [np.flatnonzero(bi.ravel() == b) for b in range(10)]
    W = mult[total > 0].sum()
    for lo in range(0, B, 128):
        P = permutation_words(rng, K, A, min(128, B - lo), columns=columns, word_count=words.shape[1])
        H = cell_histograms(words, mult, total, P, K, A).reshape(len(P), -1) if curves or not ordered else None
        # The null mean |s| adds multiplicity * |s| in float32, in signature row
        # order; on millions of rows that rounding is measurable, so the order is
        # kept. The curves use the integer cell histogram.
        if ordered:
            mabs[lo:lo + len(P)] = ordered_sum(words, mult, total, P, A, np.abs(s)).astype(np.float64) / W
        else:
            mabs[lo:lo + len(P)] = H @ flat_s / W
        if curves:
            df = d.ravel().astype(np.float64)
            for b, ix in enumerate(bin_indices):
                w = H[:, ix].sum(axis=1)
                v = H[:, ix] @ df[ix]
                cc[lo:lo + len(P), b] = np.where(w >= MINW, v / np.maximum(w, 1e-30), np.nan)
    return mabs, cc


def animacy_specificity(lib, layers=None):
    """The layer statistics, curves and metadata of one network (``incidence_package.IncidencePackage``)."""
    if layers is not None and set(layers) - set(lib.layers()):
        raise ValueError(f'unknown layers: {sorted(set(layers)-set(lib.layers()))}')
    K = len(lib.category_ids)
    anim = np.asarray(lib.animate, bool)
    A, I = int(anim.sum()), int((~anim).sum())
    # ResNet-18's null means are summed from the float64 multiplicity histogram;
    # the other networks sum in float32 in signature order. Both evaluate the
    # same float32 per-cell specificity.
    ordered = lib.axis == 'kernel_level' and lib.model != 'resnet18'
    if A < 2 or I < 2:
        raise ValueError('the jackknife needs at least two categories in each domain')
    rows, store = [], {'qedges': QEDGES.copy()}
    for lr in lib.layer_manifest.itertuples(index=False):
        L, stage = lr.layer, lr.stage
        if layers is not None and L not in layers:
            continue
        words, mult, positive = lib.signatures(L)
        total = np.bitwise_count(words).sum(axis=1).astype(np.int64)
        kA = np.bitwise_count(words & words_from_flags(anim)).sum(axis=1).astype(np.int64)
        H = np.bincount(total * (A + 1) + kA, weights=mult,
                        minlength=(K + 1) * (A + 1)).reshape(K + 1, A + 1)
        H[0] = 0
        W = float(H.sum())
        obs, signed, oc, ow = summarize(H, K, A)
        s, _, _ = cell_values(K, A)
        es2 = np.zeros(K + 1)
        for t in np.unique(total):
            xs = np.arange(max(0, t - I), min(A, t) + 1)
            pmf = hypergeom.pmf(xs, K, A, t)
            es2[t] = (pmf * s[t, xs] ** 2).sum() / max(pmf.sum(), 1e-300)
        struct = float(np.sqrt(max(0., (H * s**2).sum() / W - H.sum(axis=1) @ es2 / W)))
        B = 10000 if stage in {'classifier', 'layer4'} else 5000
        seed = zlib.crc32(f'{lib.model}|{L}'.encode()) & 0xFFFFFFFF
        null, nc = null_statistics(words, mult, total, K, A, B, np.random.default_rng(seed), ordered=ordered)
        nm, ns = float(null.mean()), float(null.std(ddof=1))
        delta = obs - nm
        p = float((1 + (null >= obs).sum()) / (B + 1))
        mcse = float(np.sqrt(p * (1 - p) / B))
        nmc, nsc = np.nanmean(nc, axis=0), np.nanstd(nc, axis=0, ddof=1)
        valid = np.isfinite(nmc) & (nsc > 0) & np.isfinite(oc)
        if valid.any():
            maxT = np.nanmax((nc[:, valid] - nmc[valid]) / nsc[valid], axis=1)
            crit = float(np.nanquantile(maxT, .95))
            bound = nmc + crit * nsc
            exceed = valid & (oc > bound)
        else:
            crit, bound, exceed = np.nan, np.full(10, np.nan), np.zeros(10, bool)
        loo, loo_delta = np.zeros(K), np.zeros(K)
        rng2 = np.random.default_rng(seed ^ 0xA5A5A5)
        for c in range(K):
            col = ((words[:, c // 64] >> np.uint64(c % 64)) & np.uint64(1)).astype(np.int64)
            totc = total - col
            ac = kA - col if anim[c] else kA
            Ac = A - int(anim[c])
            keep = totc > 0
            hc = np.bincount(totc[keep] * (Ac + 1) + ac[keep], weights=mult[keep],
                             minlength=K * (Ac + 1)).reshape(K, Ac + 1)
            loo[c] = summarize(hc, K - 1, Ac)[0]
            if stage in {'classifier', 'layer4'}:
                nv, _ = null_statistics(words, mult, totc, K - 1, Ac, 200, rng2,
                                        columns=np.delete(np.arange(K), c), curves=False, ordered=ordered)
                loo_delta[c] = loo[c] - nv.mean()
        jk = float(np.sqrt((K - 1) / K * ((loo - loo.mean())**2).sum()))
        jkd = (float(np.sqrt((K - 1) / K * ((loo_delta - loo_delta.mean())**2).sum()))
               if stage in {'classifier', 'layer4'} else np.nan)
        rows.append(dict(model=lib.model, layer=L, stage=stage,
                         n_rows=int(W) if lib.axis == 'kernel_weight' else len(words), total_mult=W,
                         B_perms=B, seed=seed, obs_mean_abs_s=obs, obs_mean_signed_s=signed,
                         struct=struct, null_mean_abs_s=nm, null_sd_abs_s=ns, delta_pi_abs_s=delta,
                         p_perm=p, mc_se=mcse, jk_se_obs=jk, jk_se_delta=jkd, maxT_crit=crit,
                         n_exceed_bins=int(exceed.sum())))
        for suffix, value in [('obs_curve', oc), ('obs_wsum', ow), ('null_curve_mean', nmc),
                              ('null_curve_sd', nsc), ('envelope_bound', bound),
                              ('exceed', exceed.astype(np.int8)), ('null_mabs', null.astype(np.float32))]:
            store[f'{L}|{suffix}'] = value
        print(f'[{lib.model}] animacy {L}: delta={delta:+.6f}, p={p:.5g}', flush=True)
    return dict(stats=pd.DataFrame(rows), curves=store, meta=dict(model=lib.model, n_layers=len(rows),
                K=K, q_edges=QEDGES.tolist(), n_animate=A, n_inanimate=I, eps=EPS, min_bin_weight=MINW))
