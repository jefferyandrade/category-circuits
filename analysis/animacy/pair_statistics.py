"""Category-pair statistics of the shared-category overlap (``shared_overlap.py``).

  exact_J0       exact hypergeometric expectation of the Jaccard overlap of two random sets of the given sizes
                 (windowed, machine precision)
  jackknife      delete-one-category jackknife of the mean of a pair statistic
  holm           Holm step-down adjustment
"""
import numpy as np

NSIGMA = 14.0          # pmf window half-width; tail mass beyond 14 sigma < 1e-40

def _hyp_pmf(xs, E, a, b):
    """Hypergeometric pmf via gammaln (scipy's boost backend is pathologically
    slow for mid-size E with large a,b). Exact to ~1e-14 relative."""
    from scipy.special import gammaln
    xs = np.asarray(xs, dtype=np.float64)
    lg = lambda n: gammaln(n + 1.0)
    logp = (lg(a) - lg(xs) - lg(a - xs)
            + lg(E - a) - lg(b - xs) - lg(E - a - b + xs)
            - (lg(E) - lg(b) - lg(E - b)))
    return np.exp(logp)

def _pmf_window(E, a, b):
    """Support window and pmf for I ~ Hypergeom(E, a, b)."""
    if a == 0 or b == 0:
        return np.array([0]), np.array([1.0])
    mean = a * (b / E)
    var = mean * (1 - a / E) * ((E - b) / max(E - 1, 1))
    half = max(8.0, NSIGMA * np.sqrt(max(var, 1e-12)))
    lo = max(int(np.floor(mean - half)), max(0, a + b - E))
    hi = min(int(np.ceil(mean + half)), min(a, b))
    xs = np.arange(lo, hi + 1)
    pmf = _hyp_pmf(xs, E, a, b)
    s = pmf.sum()
    return xs, (pmf / s if s > 0 else pmf)

def exact_J0(a, b, E):
    """E[ I/(a+b-I) ] under the density-matched hypergeometric null."""
    a, b, E = int(a), int(b), int(E)
    if a + b == 0:
        return np.nan
    if a == 0 or b == 0:
        return 0.0
    xs, pmf = _pmf_window(E, a, b)
    denom = a + b - xs
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = np.where(denom > 0, xs / np.maximum(denom, 1), 1.0)
    return float((pmf * vals).sum())

def pair_upper(M):
    iu = np.triu_indices(M.shape[0], k=1)
    return M[iu], iu

def jackknife_pair_mean(pair_vals, iu, K):
    """Delete-one-category jackknife for the mean of an order-2 pair statistic.
    Returns (mean, jackknife SE)."""
    full_mean = float(np.nanmean(pair_vals))
    i_idx, j_idx = iu
    tots = np.zeros(K)
    cnts = np.zeros(K)
    valid = ~np.isnan(pair_vals)
    for arr in (i_idx, j_idx):
        np.add.at(tots, arr[valid], pair_vals[valid])
        np.add.at(cnts, arr[valid], 1)
    total = np.nansum(pair_vals)
    nvalid = valid.sum()
    loo = (total - tots) / np.maximum(nvalid - cnts, 1)
    se = float(np.sqrt((K - 1) / K * np.sum((loo - loo.mean()) ** 2)))
    return full_mean, se

def holm(pvals):
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    m = np.sum(~np.isnan(p))
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    adj = np.full_like(p, np.nan)
    running = 0.0
    for rank, idx in enumerate(order):
        if np.isnan(p[idx]):
            continue
        val = (m - rank) * p[idx]
        running = max(running, min(val, 1.0))
        adj[idx] = running
    return adj
