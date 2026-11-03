"""Category resampling and Holm adjustment behind the paper's intervals and tests.

Categories are the resampling and label-permutation unit: the percentile category bootstrap (10,000 resamples), the
sign-flip test of a mean, label permutations, and the delete-one-category jackknife of category-pair statistics, which
drops every pair that contains the deleted category and uses a Student t interval. A taxonomy group (block50,
superordinate or domain) can take the place of the category. A statistic that is undefined in a resample raises an
error; it is never dropped from the resampling distribution. Holm adjustment is applied once per family of tests.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import numpy as np
from scipy.stats import t

N_RESAMPLES = 10000
FIELDS = ("estimate", "ci_low", "ci_high", "p", "p_holm", "family", "unit", "scope",
          "n_categories", "n_units", "n_blocks", "method")


UNITS = ("category", "block50", "superordinate", "domain")


def unit_labels(frame, unit="category"):
    """The resampling unit of each row: its category, or its taxonomy group at the level named by unit."""
    if unit not in UNITS:
        raise ValueError("unit must be one of %s" % (UNITS,))
    column = "category" if unit == "category" else unit
    if unit == "domain" and column not in frame and "animate" in frame:
        column = "animate"
    if column not in frame:
        raise ValueError("%s labels are required for unit=%s" % (column, unit))
    return np.asarray(frame[column])


def _blocks(blocks, n, unit="category"):
    if unit not in UNITS:
        raise ValueError("unit must be one of %s" % (UNITS,))
    if n < 2:
        raise ValueError("at least two categories are required")
    if unit == "category":
        if blocks is not None and np.asarray(blocks).shape != (n,):
            raise ValueError("one aligned label per category is required")
        return np.arange(n), np.arange(n)
    b = np.asarray(blocks)
    if b.ndim != 1 or len(b) != n:
        raise ValueError("one %s label per category is required" % unit)
    if any(x is None or (isinstance(x, (float, np.floating)) and np.isnan(x)) for x in b):
        raise ValueError("missing %s labels" % unit)
    unique, inv = np.unique(b, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("inference requires at least two %s units" % unit)
    return unique, inv


def jackknife_method(unit="category"):
    return "delete-one-%s jackknife (drop either pair endpoint; t 95%%)" % unit


def bootstrap_method(unit="category", n_resamples=N_RESAMPLES):
    return "percentile %s bootstrap (%d resamples)" % (unit, n_resamples)


def _draws(n_resamples):
    if isinstance(n_resamples, bool) or int(n_resamples) != n_resamples or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    return int(n_resamples)


def _finite_stat(x):
    v = float(x)
    if not np.isfinite(v):
        raise ValueError("statistic is not finite; a resample with an undefined statistic is not dropped")
    return v


def block_bootstrap(values, blocks=None, statistic=np.mean, n_resamples=N_RESAMPLES, seed=0,
                    return_distribution=False, unit="category"):
    """Percentile bootstrap: resample units with replacement, each with all its categories, weighting categories
    equally."""
    x = np.asarray(values, dtype=float)
    if x.ndim == 0 or not np.isfinite(x).all():
        raise ValueError("finite category values are required")
    u, inv = _blocks(blocks, len(x), unit)
    n_draw = _draws(n_resamples)
    rng = np.random.default_rng(seed)
    estimate = _finite_stat(statistic(x))
    dist = np.empty(n_draw)
    if statistic is np.mean and x.ndim == 1:
        sums = np.bincount(inv, weights=x, minlength=len(u))
        counts = np.bincount(inv, minlength=len(u))
        for start in range(0, n_draw, 512):
            chosen = rng.integers(0, len(u), (min(512, n_draw-start), len(u)))
            dist[start:start+len(chosen)] = sums[chosen].sum(1) / counts[chosen].sum(1)
    else:
        members = [np.flatnonzero(inv == g) for g in range(len(u))]
        for i in range(n_draw):
            idx = np.concatenate([members[g] for g in rng.integers(0, len(u), len(u))])
            dist[i] = _finite_stat(statistic(x[idx]))
    lo, hi = np.quantile(dist, [.025, .975])
    out = dict(estimate=estimate, ci_low=float(lo), ci_high=float(hi),
               n_categories=len(x), n_units=len(u), n_blocks=len(u), n_resamples=n_draw,
               resampling_unit=unit, method=bootstrap_method(unit, n_draw))
    if return_distribution:
        out["distribution"] = dist
    return out


def block_sign_flip(values, blocks=None, n_resamples=N_RESAMPLES, seed=0, unit="category"):
    """Two-sided sign-flip test of a mean of per-category differences: one random sign per unit.

    Monte Carlo p = (1 + number of draws at least as extreme) / (B + 1), ties included.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("finite per-category differences are required")
    u, inv = _blocks(blocks, len(x), unit)
    n_draw = _draws(n_resamples)
    sums = np.bincount(inv, weights=x, minlength=len(u))
    obs = abs(x.sum())
    rng = np.random.default_rng(seed)
    extreme = 0
    tolerance = 16*np.finfo(float).eps*max(1., np.abs(x).sum())
    for start in range(0, n_draw, 512):
        flips = 2*rng.integers(0, 2, (min(512, n_draw-start), len(u)), dtype=np.int8)-1
        null = (flips * sums).sum(1)
        extreme += np.count_nonzero(np.abs(null) >= obs-tolerance)
    return float((1+extreme)/(n_draw+1))


def label_permutation(labels, blocks=None, statistic=None, constraints=None,
                      n_resamples=N_RESAMPLES, seed=0, vectorized=False, unit="category"):
    """Two-sided label-permutation test: shuffle the category labels within constraints, keeping label counts fixed.

    With a taxonomy unit, units of equal size exchange their whole label vectors.
    A vectorized statistic accepts a B x N label array and returns B values.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be a category vector")
    u, inv = _blocks(blocks, len(labels), unit)
    members = [np.flatnonzero(inv == g) for g in range(len(u))]
    if constraints is None:
        constraints = np.zeros(len(labels), dtype=int)
    constraints = np.asarray(constraints)
    if constraints.ndim != 1 or len(constraints) != len(labels):
        raise ValueError("constraints require one value per category")
    groups = defaultdict(list)
    for g, idx in enumerate(members):
        if len(np.unique(constraints[idx])) != 1:
            raise ValueError("permutation constraints must be constant within each unit")
        groups[(len(idx), str(constraints[idx[0]]))].append(g)
    groups = [np.asarray(gs) for gs in groups.values() if len(gs) > 1]
    mobile = [gs for gs in groups if any(not np.array_equal(labels[members[gs[0]]], labels[members[g]])
                                        for g in gs[1:])]
    if not mobile:
        raise ValueError("label permutation undefined: no exchange of units changes the labels")
    n_draw = _draws(n_resamples)
    estimate = _finite_stat(statistic(labels))
    rng = np.random.default_rng(seed)
    extreme = 0
    tol = 16*np.finfo(float).eps*max(1., abs(estimate))
    for start in range(0, n_draw, 128):
        size = min(128, n_draw-start)
        perm = np.broadcast_to(labels, (size, len(labels))).copy()
        for gs in mobile:
            source = gs[np.argsort(rng.random((size, len(gs))), axis=1)]
            targets = np.stack([members[g] for g in gs])
            source_positions = np.searchsorted(gs, source)
            perm[:, targets] = labels[targets[source_positions]]
        null = np.asarray(statistic(perm) if vectorized else [statistic(v) for v in perm], dtype=float)
        if null.shape != (size,) or not np.isfinite(null).all():
            raise ValueError("undefined permutation statistic")
        extreme += np.count_nonzero(np.abs(null) >= abs(estimate)-tol)
    return dict(estimate=estimate, p=float((1+extreme)/(n_draw+1)), n_resamples=n_draw,
                n_categories=len(labels), n_units=len(u), n_blocks=len(u), resampling_unit=unit,
                n_permutable_units=sum(map(len, mobile)), n_permutable_blocks=sum(map(len, mobile)),
                method=("single-category label permutation within declared constraint" if unit == "category"
                        else "%s label permutation within equal unit size and declared constraint" % unit))


def pair_mean(matrix):
    m = np.asarray(matrix, dtype=float)
    values = m[np.triu_indices(len(m), 1)]
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("no defined category pairs")
    return float(values.mean())


def joint_block_jackknife(matrices, blocks=None, statistic=None, unit="category"):
    """Delete-one-unit jackknife of a statistic of several aligned category-pair matrices, deleting each unit from
    all of them at once; t interval."""
    if statistic is None:
        statistic = pair_mean
    matrices = [np.asarray(m, dtype=float) for m in matrices]
    if not matrices or any(m.ndim != 2 or m.shape != matrices[0].shape for m in matrices):
        raise ValueError("aligned square category matrices required")
    n = len(matrices[0])
    if matrices[0].shape != (n, n):
        raise ValueError("square category matrices required")
    u, inv = _blocks(blocks, n, unit)
    estimate = _finite_stat(statistic(*matrices))
    deleted = []
    for g in range(len(u)):
        idx = np.flatnonzero(inv != g)
        if len(idx) < 2:
            raise ValueError("deleting a unit leaves fewer than two categories")
        deleted.append(_finite_stat(statistic(*[m[np.ix_(idx, idx)] for m in matrices])))
    deleted = np.asarray(deleted)
    se = np.sqrt((len(u)-1)/len(u)*np.square(deleted-deleted.mean()).sum())
    half = t.ppf(.975, len(u)-1)*se
    return dict(estimate=estimate, ci_low=float(estimate-half), ci_high=float(estimate+half),
                se=float(se), n_categories=n, n_units=len(u), n_blocks=len(u),
                resampling_unit=unit, method=jackknife_method(unit),
                deleted=deleted, replicates=deleted)


def block_jackknife(matrix, blocks=None, statistic=None, unit="category"):
    """Delete-one-unit jackknife of the mean of a category-pair matrix (or of another statistic of it); t interval."""
    if statistic is None or statistic is pair_mean:
        m = np.asarray(matrix, dtype=float)
        if m.ndim != 2 or m.shape != (len(m), len(m)):
            raise ValueError("square category matrix required")
        u, inv = _blocks(blocks, len(m), unit)
        ii, jj = np.triu_indices(len(m), 1)
        keep = np.isfinite(m[ii, jj])
        ii, jj, values = ii[keep], jj[keep], m[ii[keep], jj[keep]]
        moments = _deleted_pair_sums(np.column_stack([np.ones(len(values)), values]), inv[ii], inv[jj], len(u))
        if np.any(moments[:, 0] <= 0):
            raise ValueError("deleting a unit leaves no defined pairs")
        return _jackknife_summary(moments[0,1]/moments[0,0], moments[1:,1]/moments[1:,0], len(m), len(u), unit)
    return joint_block_jackknife([matrix], blocks, pair_mean if statistic is None else statistic, unit=unit)


def _deleted_pair_sums(values, left, right, groups):
    """Sums over pairs after deleting each unit (every pair with an endpoint in it); a pair within one unit is
    subtracted once."""
    total = values.sum(0)
    incident = np.empty((groups, values.shape[1]))
    cross = left != right
    for j in range(values.shape[1]):
        incident[:,j] = (np.bincount(left, weights=values[:,j], minlength=groups)
            + np.bincount(right[cross], weights=values[cross,j], minlength=groups))
    return np.vstack([total, total-incident])


def _jackknife_summary(estimate, deleted, n, groups, unit="category"):
    deleted = np.asarray(deleted)
    if not np.isfinite(deleted).all() or not np.isfinite(estimate):
        raise ValueError("undefined unit-deletion statistic")
    se = np.sqrt((groups-1)/groups*np.square(deleted-deleted.mean()).sum())
    half = t.ppf(.975, groups-1)*se
    return dict(estimate=float(estimate), ci_low=float(estimate-half), ci_high=float(estimate+half),
                se=float(se), n_categories=n, n_units=groups, n_blocks=groups,
                resampling_unit=unit, method=jackknife_method(unit),
                deleted=deleted, replicates=deleted)


def holm(p_values, unit="category"):
    """Holm adjustment over one family of tests. A missing test (NaN) counts as p = 1 and stays missing."""
    if unit not in UNITS:
        raise ValueError("unknown resampling unit: %s" % unit)
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or np.any((p[np.isfinite(p)] < 0) | (p[np.isfinite(p)] > 1)) or np.isinf(p).any():
        raise ValueError("p-values must be in [0,1] or missing")
    if not len(p):
        return p.copy()
    missing = np.isnan(p)
    ranked = np.where(missing, 1., p)
    order = np.argsort(ranked, kind="stable")
    adjusted = np.minimum(1., np.maximum.accumulate((len(p)-np.arange(len(p)))*ranked[order]))
    out = np.empty_like(p)
    out[order] = adjusted
    out[missing] = np.nan
    return out


def log_ratio_to_steps(log_ratio, unit="category"):
    """A log ratio of kept fractions in pruning steps, each step keeping a factor 0.85 (Methods)."""
    if unit not in UNITS:
        raise ValueError("unknown resampling unit: %s" % unit)
    return np.asarray(log_ratio)/np.log(1./.85)


def result_row(estimate, ci_low=None, ci_high=None, p=None, p_holm=None, *, family,
               unit="category", scope="G", n_categories, n_units=None, n_blocks=None, method, **extra):
    """One checked result row: estimate, interval, p values, the counts behind them and the method. scope is D for a
    value exact over the categories present and G for an interval that generalizes to other categories; a family
    named "descriptive" is never Holm-adjusted."""
    estimate, ci_low, ci_high, p, p_holm = [None if v is None or np.isnan(v) else float(v)
        for v in (estimate, ci_low, ci_high, p, p_holm)]
    if scope not in ("D", "G") or not family or not unit or not method:
        raise ValueError("scope D/G, family, unit, and method are required")
    if n_units is None:
        n_units = n_categories if unit == "category" else (0 if n_blocks is None else n_blocks)
    if int(n_categories) != n_categories or int(n_units) != n_units or not 0 <= n_units <= n_categories:
        raise ValueError("invalid category or unit counts")
    # n_blocks, a column of the result tables, is the number of units unless given.
    n_blocks = n_units if n_blocks is None else n_blocks
    if int(n_blocks) != n_blocks or not 0 <= n_blocks <= n_categories:
        raise ValueError("invalid block count")
    if family == "descriptive":
        p_holm = None
    for name, value in (("estimate", estimate), ("ci_low", ci_low), ("ci_high", ci_high), ("p", p), ("p_holm", p_holm)):
        if value is not None and not np.isfinite(value):
            raise ValueError("%s must be finite or explicitly absent" % name)
    if (ci_low is None) != (ci_high is None) or (ci_low is not None and ci_low > ci_high):
        raise ValueError("both ordered interval limits are required")
    if any(v is not None and not 0 <= v <= 1 for v in (p, p_holm)):
        raise ValueError("p-values must lie in [0,1]")
    return dict(estimate=estimate, ci_low=ci_low, ci_high=ci_high, p=p, p_holm=p_holm,
                family=family, unit=unit, scope=scope, n_categories=int(n_categories),
                n_units=int(n_units), n_blocks=int(n_blocks), method=method, **extra)


def adjust_families(rows):
    """Holm-adjust the p values within each network and family; rows of one hypothesis_id count as one test."""
    rows = [dict(r) for r in rows]
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        if row["family"] == "descriptive":
            row["p_holm"] = None
            continue
        groups[(row.get("network"), row["family"])].append(i)
    for indices in groups.values():
        hypotheses = defaultdict(list)
        for i in indices:
            hypotheses[rows[i].get("hypothesis_id", i)].append(i)
        tests = []
        for same in hypotheses.values():
            values = [np.nan if rows[i].get("p") is None else rows[i]["p"] for i in same]
            if not np.allclose(values, values[0], equal_nan=True, rtol=0, atol=0):
                raise ValueError("two units of the same hypothesis have different p-values")
            tests.append(values[0])
        for same, p in zip(hypotheses.values(), holm(tests)):
            for i in same:
                rows[i]["p_holm"] = None if np.isnan(p) else float(p)
    return rows


def write_results(rows, path):
    """Check the result rows, apply Holm once per family and write them to a new CSV file."""
    import pandas as pd
    from atomic_io import publish_file
    rows = adjust_families(rows)
    for row in rows:
        row.setdefault("n_units", row.get("n_blocks", row.get("n_categories", 0)))
        row.setdefault("n_blocks", row["n_units"])
        row.setdefault("partial", False)
    if not rows:
        raise ValueError("no result rows to write")
    for row in rows:
        missing = set(FIELDS)-set(row)
        if missing:
            raise ValueError("result rows lack the fields %s" % sorted(missing))
        result_row(**row)
    df = pd.DataFrame(rows)
    df = df[list(FIELDS)+[c for c in df if c not in FIELDS]]
    publish_file(df, Path(path))
    return df


def permuted_labels(labels, blocks, rng, constraints=None, unit="category"):
    """One random relabelling of the categories within constraints; with a taxonomy unit, equal-size units
    exchange their labels."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) < 2:
        raise ValueError("at least two category labels are required")
    if unit == "category":
        if blocks is not None and np.asarray(blocks).shape != labels.shape:
            raise ValueError("one aligned label per category is required")
        if constraints is None:
            return rng.permutation(labels)
        constraints = np.asarray(constraints)
        if constraints.shape != labels.shape:
            raise ValueError("one constraint per category required")
        out = labels.copy()
        for constraint in np.unique(constraints):
            idx = np.flatnonzero(constraints == constraint)
            out[idx] = rng.permutation(labels[idx])
        return out
    u, inv = _blocks(blocks, len(labels), unit)
    constraints = np.zeros(len(labels), dtype=int) if constraints is None else np.asarray(constraints)
    if constraints.shape != labels.shape:
        raise ValueError("one constraint per category required")
    members = [np.flatnonzero(inv == g) for g in range(len(u))]
    groups = defaultdict(list)
    for g, idx in enumerate(members):
        if len(np.unique(constraints[idx])) != 1:
            raise ValueError("constraint varies within a unit")
        groups[(len(idx), str(constraints[idx[0]]))].append(g)
    out = labels.copy()
    for gs in groups.values():
        for dst, src in zip(gs, rng.permutation(gs)):
            out[members[dst]] = labels[members[src]]
    return out
