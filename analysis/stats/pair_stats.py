"""Helpers for statistics of category-pair matrices.

A pair statistic uses the unordered pairs above the diagonal whose values are finite. The helpers give the pair
values, their mean and Pearson correlation, the delete-one-category jackknife of a statistic of category indices
(through category_resampling), label-permutation distributions and result rows. An undefined value stays undefined; it is never
dropped or filled in.
"""
from __future__ import annotations

import numpy as np
import category_resampling as PS


def pair_values(matrix):
    m = np.asarray(matrix, dtype=float)
    return m[np.triu_indices(m.shape[0], 1)]


def pair_mean(matrix):
    values = pair_values(matrix)
    good = np.isfinite(values)
    return float(values[good].mean()) if good.any() else np.nan


def correlation(x, y):
    a, b = pair_values(x), pair_values(y)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def indexed_jackknife(statistic, blocks=None, unit="category", n_categories=None):
    """Delete-one-unit jackknife of statistic(category indices), through category_resampling.block_jackknife. If it is undefined,
    the estimate and interval are NaN and the reason is returned."""
    n = len(blocks) if blocks is not None else n_categories
    if n is None:
        raise ValueError("category count or labels required")
    ids = np.diag(np.arange(n, dtype=float))
    try:
        return PS.block_jackknife(ids, blocks,
            statistic=lambda reduced: statistic(np.diag(reduced).astype(int)), unit=unit)
    except ValueError as error:
        return {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "replicates": [], "status": "not formed", "reason": str(error)}


def label_distribution(labels, blocks, statistic, rng, n_resamples, unit="category", constraints=None):
    """The statistic under n_resamples random relabellings of the categories (category_resampling.permuted_labels)."""
    return np.asarray([statistic(PS.permuted_labels(labels, blocks, rng,
                       constraints=constraints, unit=unit))
                       for _ in range(n_resamples)], float)


def result_row(estimate, **kwargs):
    """category_resampling.result_row, with non-finite values left empty."""
    clean = lambda value: None if value is None or not np.isfinite(value) else float(value)
    for key in ("ci_low", "ci_high", "p", "p_holm"):
        if key in kwargs:
            kwargs[key] = clean(kwargs[key])
    return PS.result_row(clean(estimate), **kwargs)
