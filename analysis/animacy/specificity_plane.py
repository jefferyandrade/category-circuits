"""The participation-specificity plane of Fig. 5 and its chance contours.

Every kept connection of a layer is placed by the categories whose circuits keep it: k_A of the n_A animate and
k_I of the n_I inanimate categories give

  participation  q = (p_A + p_I) / 2,           p_A = k_A / n_A,  p_I = k_I / n_I   (0 narrow, 1 broad)
  specificity    s = (p_A - p_I) / (p_A + p_I)                                    (-1 inanimate, +1 animate)

``plane(H)`` bins the connections of the joint count grid H[k_A, k_I] on a 40 x 40 (q, s) grid and returns
the observed counts, the counts expected under the label-permutation null and the participation marginal.
Under that null a connection kept by k categories keeps k_A ~ Hypergeometric(K, n_A, k) animate ones, so the
null redistributes each total k of the grid by that probability mass function (``hyp_pmf``).

The figure draws the null as contours on the bin-centre grid at NULL_CONTOUR_FRACTIONS of its peak
(levels = NULL_CONTOUR_FRACTIONS * null.max(), drawn when the null has positive mass).

``panel_tables.write_animacy_tables`` loads ``hyp_pmf`` and ``plane`` from this file and evaluates them with the
population's n_A and n_I.
"""
import numpy as np
from scipy.special import gammaln

# The AlexNet population of the paper: 401 animate and 599 inanimate categories.
nA, nI = 401, 599
K = nA + nI
EPS = 1e-9

# plane binning
NQ, NS = 40, 40
qb = np.linspace(0, 1, NQ + 1)
sb = np.linspace(-1, 1, NS + 1)

# Chance contours: the null counts at 1%, 10% and 50% of their peak.
NULL_CONTOUR_FRACTIONS = np.array([0.01, 0.1, 0.5])


def hyp_pmf(xs, E, a, b):
    lg = lambda n: gammaln(np.asarray(n, dtype=np.float64) + 1.0)
    return np.exp(lg(a) - lg(xs) - lg(a - xs) + lg(E - a) - lg(b - xs) - lg(E - a - b + xs)
                  - (lg(E) - lg(b) - lg(E - b)))


def plane(H):
    """observed and null (label-permutation mean) counts on the (q, s) grid, and the
    q-marginal of retained routes"""
    kA = np.arange(H.shape[0])[:, None]; kI = np.arange(H.shape[1])[None, :]
    pA, pI = kA / nA, kI / nI
    q = 0.5 * (pA + pI); s = (pA - pI) / (pA + pI + EPS)
    H = H.copy(); H[0, 0] = 0
    # null: for each total k, redistribute the k-marginal by the hypergeometric pmf
    tot = (kA + kI)
    Hn = np.zeros_like(H)
    for k in np.unique(tot[H > 0]):
        n_k = H[tot == k].sum()
        xs = np.arange(max(0, k - nI), min(nA, k) + 1)
        pmf = hyp_pmf(xs, K, nA, k); pmf /= pmf.sum()
        Hn[xs, k - xs] = n_k * pmf
    qi = np.clip(np.digitize(q, qb) - 1, 0, NQ - 1)
    si = np.clip(np.digitize(s, sb) - 1, 0, NS - 1)
    obs = np.zeros((NS, NQ)); nul = np.zeros((NS, NQ))
    np.add.at(obs, (si, qi), H); np.add.at(nul, (si, qi), Hn)
    qm = obs.sum(axis=0)
    return obs, nul, qm
