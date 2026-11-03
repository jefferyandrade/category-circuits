"""Fig. 6: the category-tree structure of the wiring (W) and of the mean category representations (A) at one layer,
plain and each beyond the other, optionally with five untrained networks (U).

At each level of the category tree (domain, superordinate, block50), the chance-corrected tree share of W and of A,
and the partial shares W|A and A|W (the share of one beyond the other, tree_contrast.partial_share). W is the excess
overlap over the sign- and weight-size-matched reference, A the cross-validated similarity of the dense network's
category means at the layer's activation site (any pair matrix in that format can stand in for A, as in the
readout-weight and output-geometry controls). All lenses share one relabelling of the categories per draw. Each
share is chance-corrected with its mean over relabellings of all categories, held fixed in every deletion of its
delete-one-category jackknife (t interval). With --untrained-dirs, the five untrained networks' shares are averaged,
and the interval of that mean adds the between-network variance divided by five to the jackknife variance.

With --wiring-minus-representations, the table adds at each level the difference W - A with its paired jackknife
interval (difference_interval); section 2.3 prints it at fc6 and fc7 on all 1,000 categories
(data/tree_share/wiring_minus_representations.tsv). With --exclude-superordinate GROUP, every matrix first loses the
categories of that superordinate group, and the difference is added as well: section 2.3 leaves out the mammals
(mammal.n.01, the largest group, 218 categories) at fc6 and fc7, which keeps 782 categories
(data/robustness/wiring_without_mammals.tsv).

Usage: tree_contrast.py --layer L --gram-dir DIR --pairs-dir DIR --trained-dir DIR --site-map JSON --taxonomy CSV
           --out CSV [--untrained-dirs DIR1,...,DIR5] [--wiring-minus-representations]
           [--exclude-superordinate GROUP]
       tree_contrast.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tree_share as TS  # noqa: E402
import tree_infer as TI  # noqa: E402
import pair_stats as PST  # noqa: E402
import category_resampling as PS  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import publish_file, check_outputs  # noqa: E402


def partial_share(y: np.ndarray, x: np.ndarray, cell: np.ndarray, ncell: int) -> float:
    """The partial share of y beyond x: the partial R^2 of the cell indicators after a linear regression of y on x
    (with intercept), over the finite pairs."""
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < 3:
        return np.nan
    y, x, c = y[ok], x[ok], cell[ok]
    xc, yc = x - x.mean(), y - y.mean()
    sxx = float(xc @ xc)
    if sxx <= 0:
        return np.nan
    ssr_x = float(yc @ yc) - float(xc @ yc) ** 2 / sxx
    n = np.bincount(c, minlength=ncell).astype(np.float64)
    if (n > 0).sum() < 2 or ssr_x <= 0:
        return np.nan
    nz = np.maximum(n, 1)
    xm, ym = np.bincount(c, x, ncell) / nz, np.bincount(c, y, ncell) / nz
    xw, yw = x - xm[c], y - ym[c]
    swxx = float(xw @ xw)
    ssr_full = float(yw @ yw) - (float(xw @ yw) ** 2 / swxx if swxx > 0 else 0.0)
    v = (ssr_x - ssr_full) / ssr_x
    return float(v) if np.isfinite(v) else np.nan


def stat(kind: str, y: np.ndarray, x: np.ndarray | None, iu, codes: np.ndarray) -> float:
    """The plain share of y, or its partial share beyond x."""
    if kind == "plain":
        return TI.formed_share(y, iu, codes)
    G = int(codes.max()) + 1
    return partial_share(y, x, TS.cell_index(codes, iu, G), G * G)


def perm(kind, y, x, iu, codes, rng, nperm, blocks=None, unit="category"):
    """The observed share and its values under nperm relabellings of the categories."""
    obs = stat(kind, y, x, iu, codes)
    dist = (np.array([stat(kind, y, x, iu, rng.permutation(codes)) for _ in range(nperm)])
            if blocks is None else PST.label_distribution(codes, blocks,
                lambda c: stat(kind, y, x, iu, c), rng, nperm, unit=unit))
    return obs, dist


def load(a, layer, categories=None):
    """The layer's pair matrices on the categories they all define: W (wiring), A (--trained-dir), U_s0..U_s4 and
    their mean U (--untrained-dirs). Given categories, every matrix is first cut to those of them."""
    sites = json.loads(Path(a.site_map).read_text())
    if layer not in sites or not isinstance(sites[layer], str) or not sites[layer].strip():
        raise ValueError("site map must supply an activation site for layer %s" % layer)
    site = sites[layer]
    z = np.load(Path(a.gram_dir) / f"gram_{layer}.npz")
    pz = np.load(Path(a.pairs_dir) / f"pairs_{layer}.npz")
    cats = z["categories"]
    if not np.array_equal(cats, pz["categories"]):
        raise ValueError("gram and pairs categories must be aligned for layer %s" % layer)
    K = cats.size
    I = (z["I_all"] if "I_all" in z.files else z["I"]).astype(np.float64)
    J0 = pz["J0_magnitude"].astype(np.float64)
    if any(M.ndim != 2 or M.shape[0] != M.shape[1] or M.shape[0] < K for M in (I, J0)):
        raise ValueError("gram and pairs matrices must contain a square P block for layer %s" % layer)
    I = I[:K, :K]
    d = np.diag(I)
    S = d[:, None] + d[None, :]
    J = np.where(S - I > 0, I / np.where(S - I > 0, S - I, 1), np.nan)
    mats = {"W": (cats, J - J0[:K, :K])}

    def act(dirpath):
        m = np.load(Path(dirpath) / f"matrices_{site}.npz")
        ids = np.asarray(m["class_ids"], np.int64)
        M = np.asarray(m["S|cv:all"], np.float64)
        if ids.ndim != 1 or M.shape != (ids.size, ids.size):
            raise ValueError("activation matrix and class_ids must be aligned for site %s" % site)
        return ids, M

    mats["A"] = act(a.trained_dir)
    seeds = []
    untrained_dirs = getattr(a, "untrained_dirs", None)
    for k, dp in enumerate(untrained_dirs.split(",") if untrained_dirs else []):
        mats[f"U_s{k}"] = act(dp)
        seeds.append(f"U_s{k}")
    if categories is not None:
        for name, (c, M) in list(mats.items()):
            inside = np.isin(c, categories)
            mats[name] = (c[inside], M[np.ix_(inside, inside)])
    common = None
    for c, M in mats.values():
        if c.ndim != 1 or not np.all(c[1:] > c[:-1]):
            raise ValueError("category ids must be sorted and unique for layer %s, site %s" % (layer, site))
        M0 = M.copy(); np.fill_diagonal(M0, 0.0)
        bad = (~np.isfinite(M0)).sum(1) > M0.shape[0] // 2
        good = c[~bad]
        common = good if common is None else np.intersect1d(common, good)
    out = {}
    for name, (c, M) in mats.items():
        ii = np.searchsorted(c, common)
        assert np.array_equal(c[ii], common)
        R = M[np.ix_(ii, ii)].copy()
        np.fill_diagonal(R, np.nan)
        out[name] = R
    if seeds:
        out["U"] = np.mean([out[s] for s in seeds], axis=0)
    return common, out, seeds, site


def seed_mean_interval(estimates, deletions):
    """The mean over the five untrained networks, with a normal 95% interval whose variance is the jackknife variance
    of the mean plus the between-network variance divided by five."""
    estimates, deletions = np.asarray(estimates, float), np.asarray(deletions, float)
    if len(estimates) != 5:
        raise ValueError("untrained-network control requires exactly five random-init seeds")
    if not np.isfinite(estimates).all() or not np.isfinite(deletions).all() or deletions.shape[0] != 5:
        raise ValueError("all five seed estimates and aligned deletions must be formed")
    means = deletions.mean(axis=0)
    jk_variance = (len(means)-1)/len(means)*np.square(means-means.mean()).sum()
    seed_variance = estimates.var(ddof=1)/5
    estimate, se = estimates.mean(), np.sqrt(jk_variance + seed_variance)
    return dict(estimate=float(estimate), ci_low=float(estimate-1.96*se),
        ci_high=float(estimate+1.96*se), se=float(se),
        jackknife_variance=float(jk_variance), between_seed_variance_over_5=float(seed_variance))


def difference_interval(left, right):
    """left minus right, two shares from joint_adjusted_inference, with the paired jackknife interval: the variance of
    the n deleted differences is (n - 1)/n times their sum of squared deviations from their mean, and the interval is
    the difference +/- t(.975, n - 1) times its square root."""
    from scipy.stats import t
    estimate = left["share_adjusted"] - right["share_adjusted"]
    deleted = np.asarray(left["replicates"]) - np.asarray(right["replicates"])
    n = len(deleted)
    variance = (n - 1) / n * np.square(deleted - deleted.mean()).sum()
    se = np.sqrt(variance)
    formed = n > 1 and np.isfinite(estimate) and np.isfinite(deleted).all()
    width = t.ppf(.975, n - 1) * se if formed else np.nan
    return dict(estimate=estimate, se=se, jackknife_variance=variance, ci_low=estimate - width,
                ci_high=estimate + width)


def joint_statistics(matrices, specs):
    """statistic(indices, labels) giving every share in specs at once, from node-adjusted pair vectors that are
    computed once per set of categories and shared by all lenses."""
    cache = {}
    names = list(matrices)
    needed = set((name,name) for name in names)
    needed.update((x,y) for kind,y,x in specs if x is not None)
    def statistic(idx, labels):
        key = tuple(idx)
        if key not in cache:
            if len(cache) >= 2:
                cache.pop(next(iter(cache)))
            ij = np.triu_indices(len(idx), 1)
            vectors = {}
            for name in names:
                m = matrices[name][np.ix_(idx,idx)]
                adjusted, ok = TI.backfit_adjust(m, ~np.eye(len(idx),dtype=bool), return_convergence=True)
                vectors[name] = adjusted[ij] if ok else np.full(len(ij[0]),np.nan)
            if all(np.isfinite(vector).all() for vector in vectors.values()):
                centred = {name:vector-vector.mean() for name,vector in vectors.items()}
                covariance = {(left,right):float(np.sum(centred[left]*centred[right]))
                    for left,right in needed}
                cache[key] = (ij, vectors, centred, covariance)
            else:
                cache[key] = (ij, vectors, None, None)
        ij,vectors,centred,covariance = cache[key]
        if centred is None:
            return np.asarray([stat(kind,vectors[y],vectors[x] if x else None,ij,labels)
                               for kind,y,x in specs])
        groups = int(np.max(labels))+1
        cells = TS.cell_index(labels,ij,groups)
        counts = np.bincount(cells,minlength=groups*groups)
        if np.count_nonzero(counts) < 2:
            return np.full(len(specs),np.nan)
        counts = np.maximum(counts,1)
        sums = {name:np.bincount(cells,weights=centred[name],minlength=groups*groups)
                for name in names}
        between = {(left,right):float(np.sum(sums[left]*sums[right]/counts))
                   for left,right in needed}
        out=[]
        for kind,y,x in specs:
            syy=covariance[(y,y)]
            if kind == "plain":
                out.append(between[(y,y)]/syy if syy>0 else np.nan)
                continue
            sxx,sxy=covariance[(x,x)],covariance[(x,y)]
            residual=syy-sxy*sxy/sxx if sxx>0 else np.nan
            within_x=sxx-between[(x,x)]
            within_xy=sxy-between[(x,y)]
            within_y=syy-between[(y,y)]
            full_residual=within_y-(within_xy*within_xy/within_x if within_x>0 else 0.)
            out.append((residual-full_residual)/residual if residual>0 else np.nan)
        return np.asarray(out)
    return statistic


def joint_adjusted_inference(statistic, labels, blocks, nperm=10000, seed=0, unit="category",
                             check_deletions=20, check_draws=2000):
    """Chance-corrected shares of all lenses, with one relabelling per draw shared by all of them, and their
    jackknife intervals over the same deletions."""
    from scipy.stats import t
    labels,blocks=np.asarray(labels),np.asarray(blocks)
    groups=np.arange(len(labels)) if unit=="category" else blocks
    all_idx=np.arange(len(labels))
    observed=statistic(all_idx,labels)
    def distribution(idx,draws,draw_seed):
        rng=np.random.default_rng(draw_seed)
        return np.asarray([statistic(idx,PS.permuted_labels(labels[idx],blocks[idx],rng,unit=unit))
                           for _ in range(draws)])
    full=distribution(all_idx,nperm,seed)
    means=full.mean(axis=0)
    estimates=(observed-means)/(1-means)
    deleted=np.asarray([(statistic(np.flatnonzero(groups!=group),labels[groups!=group])-means)/(1-means)
                        for group in np.unique(groups)])
    se=np.sqrt((len(deleted)-1)/len(deleted)*np.square(deleted-deleted.mean(axis=0)).sum(axis=0))
    changes=[]
    chosen=np.random.default_rng(seed+741).permutation(np.unique(groups))[:check_deletions]
    for group in chosen:
        idx=np.flatnonzero(groups!=group)
        changes.append(np.abs(distribution(idx,check_draws,seed+1907).mean(axis=0)-means))
    maximum=np.max(changes,axis=0) if changes else np.full(len(estimates),np.nan)
    ratio=np.divide(maximum,se,out=np.full_like(maximum,np.inf),where=se>0)
    ratio[(maximum==0)&(se==0)]=0.
    results=[]
    for i,estimate in enumerate(estimates):
        summary=TI.permutation_summary(observed[i],full[:,i],strict=True)
        formed=np.isfinite(deleted[:,i]).all() and len(deleted)>1
        width=t.ppf(.975,len(deleted)-1)*se[i]
        summary.update(share_adjusted=estimate,estimate=estimate,
            ci_low=estimate-width if formed else np.nan,ci_high=estimate+width if formed else np.nan,
            replicates=deleted[:,i],se=se[i],perm_mean_held_fixed=True,
            perm_mean_check_deletions=len(chosen),perm_mean_check_draws=check_draws,
            perm_mean_check_max_change=maximum[i],perm_mean_check_ratio=ratio[i],
            perm_mean_check_flag=bool(not np.isfinite(ratio[i]) or ratio[i]>.1),
            perm_mean_check_seed=seed+741)
        results.append(summary)
    return results


def run(a) -> int:
    """All rows of one layer."""
    if a.nperm_test < 1000:
        raise ValueError("nperm-test must be >= 1000")
    out = Path(a.out)
    check_outputs(files=[out], inputs=[a.taxonomy, a.site_map, a.gram_dir, a.pairs_dir, a.trained_dir,
        *filter(None, (getattr(a, "untrained_dirs", None) or "").split(","))])
    tax = TS.load_tax(a.taxonomy)
    excluded = getattr(a, "exclude_superordinate", None)
    if excluded is not None and excluded not in set(tax.superordinate):
        raise ValueError("the taxonomy has no superordinate group %s" % excluded)
    kept = None if excluded is None else tax.index[tax.superordinate != excluded].to_numpy()
    with_difference = getattr(a, "wiring_minus_representations", False) or excluded is not None
    cats, matrices, seeds, site = load(a, a.layer, kept)
    if seeds and len(seeds) != 5:
        raise ValueError("untrained-network control requires exactly five random-init seeds")
    unit, network = getattr(a, "unit", "category"), getattr(a, "network", "alexnet")
    blocks = TI.category_units(tax, cats, unit)
    n_units = len(cats) if unit == "category" else len(np.unique(blocks))
    codes = {lvl: pd.factorize(tax.loc[cats, col].astype(str).values)[0] for lvl, col in TS.LEVELS}
    specs = [("plain", "A", None), ("plain", "W", None)]
    specs += [("plain", seed, None) for seed in seeds]
    specs += [("partial", "A", "W"), ("partial", "W", "A")]
    specs += [("partial", "A", seed) for seed in seeds]
    # wiring beyond image statistics: the tree share of W given each untrained network's similarity
    # (descriptive like every row here)
    specs += [("partial", "W", seed) for seed in seeds]
    source={name:matrix for name,matrix in matrices.items() if name != "U"}
    statistic=joint_statistics(source,specs)
    rows,results=[],{}
    for li,(level,labels) in enumerate(codes.items()):
        joint=joint_adjusted_inference(statistic,labels,blocks,a.nperm_test,a.seed+li,unit,
            getattr(a,"check_deletions",20),getattr(a,"check_draws",2000))
        for (kind,y,x),result in zip(specs,joint):
            name=y if x is None else "%s|%s"%(y,x)
            results[(name,level)]=result
            rows.append({"layer":a.layer,"network":network,"site":site,"quantity":kind,
                "lens":name,"level":level,"n_seeds":len(seeds),
                **{k:v for k,v in result.items() if k not in ("replicates","deleted")},
                **PST.result_row(result["share_adjusted"],ci_low=result.get("ci_low"),
                    ci_high=result.get("ci_high"),family="descriptive",unit=unit,
                    effect_unit="fraction",scope="G" if np.isfinite(result.get("ci_low",np.nan)) else "D",
                    n_categories=len(cats),n_units=n_units,
                    method="delete-one-%s jackknife (t interval); full-data permutation mean held fixed"%unit)})
        if with_difference:
            d=difference_interval(results[("W",level)],results[("A",level)])
            rows.append({"layer":a.layer,"network":network,"site":site,"quantity":"difference","lens":"W-A",
                "level":level,"n_seeds":len(seeds),"share_adjusted":d["estimate"],"se":d["se"],
                "jackknife_variance":d["jackknife_variance"],"perm_mean_held_fixed":True,
                "status":"FORMED" if np.isfinite(d["estimate"]) else "NOT_FORMED",
                **PST.result_row(d["estimate"],ci_low=d["ci_low"],ci_high=d["ci_high"],family="descriptive",
                    unit=unit,effect_unit="fraction",scope="G" if np.isfinite(d["ci_low"]) else "D",
                    n_categories=len(cats),n_units=n_units,
                    method="paired delete-one-%s jackknife (t interval); full-data permutation means held fixed"%unit)})
    for level in (codes if seeds else ()):
        for lens, names in (("U", seeds), ("A|U", ["A|%s" % seed for seed in seeds]), ("W|U", ["W|%s" % seed for seed in seeds])):
            estimates = np.asarray([results[(name, level)]["share_adjusted"] for name in names])
            deletions = np.asarray([results[(name, level)].get("replicates", []) for name in names])
            try:
                interval = seed_mean_interval(estimates, deletions)
            except ValueError:
                interval = dict(estimate=np.nan, ci_low=np.nan, ci_high=np.nan,
                                jackknife_variance=np.nan, between_seed_variance_over_5=np.nan)
            row = {"network": network, "layer": a.layer, "site": site, "quantity": "seed_mean", "lens": lens,
                "level": level, "n_seeds": 5, "share_adjusted": interval["estimate"],
                "seed_min": float(estimates.min()), "seed_max": float(estimates.max()),
                "seed_variance_in_primary_interval": True, "perm_mean_held_fixed": True,
                "perm_mean_check_ratio": max(results[(name, level)]["perm_mean_check_ratio"] for name in names),
                "perm_mean_check_flag": any(results[(name, level)]["perm_mean_check_flag"] for name in names),
                "jackknife_variance": interval["jackknife_variance"],
                "between_seed_variance_over_5": interval["between_seed_variance_over_5"],
                **PST.result_row(interval["estimate"], ci_low=interval["ci_low"], ci_high=interval["ci_high"],
                    family="descriptive", unit=unit, effect_unit="fraction", scope="G", n_categories=len(cats),
                    n_units=n_units, method="delete-one-%s jackknife variance of five-seed mean + between-seed variance / 5; normal 95%%" % unit)}
            rows.append(row)
            if lens == "U":
                trained = results[("A", level)]
                difference_estimates = trained["share_adjusted"] - estimates
                difference_deletions = np.asarray(trained.get("replicates", []))[None, :] - deletions
                try:
                    difference = seed_mean_interval(difference_estimates, difference_deletions)
                except ValueError:
                    difference = dict(estimate=np.nan, ci_low=np.nan, ci_high=np.nan,
                        jackknife_variance=np.nan,between_seed_variance_over_5=np.nan)
                rows.append({**row, "quantity": "difference", "lens": "A-U", "share_adjusted": difference["estimate"],
                    "seed_min":float(difference_estimates.min()), "seed_max":float(difference_estimates.max()),
                    "jackknife_variance":difference["jackknife_variance"],
                    "between_seed_variance_over_5":difference["between_seed_variance_over_5"],
                    **PST.result_row(difference["estimate"], ci_low=difference["ci_low"], ci_high=difference["ci_high"],
                        family="descriptive", unit=unit, effect_unit="fraction", scope="G", n_categories=len(cats),
                        n_units=n_units, method="paired delete-one-%s jackknife variance + between-seed variance / 5; normal 95%%" % unit)})
    for row in rows:
        if not seeds:
            row["control"] = "untrained control not supplied"
        row.setdefault("seed_variance_in_primary_interval", False)
        row.update(runtime_n_categories=len(cats), runtime_nperm=a.nperm_test,
                   runtime_check_draws=getattr(a, "check_draws", 2000))
    publish_file(pd.DataFrame(PS.adjust_families(rows)), out)
    print("wrote %s; layer %s" % (out, a.layer))
    return 0


def selftest() -> int:
    rng = np.random.default_rng(5)
    K, G = 60, 6
    codes = np.repeat(np.arange(G), K // G)
    iu = np.triu_indices(K, 1)
    cell = TS.cell_index(codes, iu, G)
    tree = rng.normal(size=G * G)[cell]
    xonly = rng.normal(size=iu[0].size)
    # (1) y = x + noise: the tree's partial share given x is ~ its chance level; plain share of y is also chance
    y = xonly + 0.1 * rng.normal(size=xonly.size)
    p = partial_share(y, xonly, cell, G * G)
    o, dist = perm("partial", y, xonly, iu, codes, rng, 200)
    assert abs(TI.adjusted_share(o, dist.mean())) < 0.05, (o, dist.mean())
    # (2) y = tree + x: the partial share recovers the tree part (adjusted share clearly > 0)
    y2 = tree + xonly
    o2, d2 = perm("partial", y2, xonly, iu, codes, rng, 200)
    assert TI.adjusted_share(o2, d2.mean()) > 0.5
    # (3) closed form equals a direct least-squares fit of y ~ x + cell dummies
    D = np.zeros((xonly.size, G * G)); D[np.arange(xonly.size), cell] = 1
    D = D[:, D.sum(0) > 0]
    full = np.linalg.lstsq(np.c_[xonly, D], y2, rcond=None)[0]
    ssr_full = float(((y2 - np.c_[xonly, D] @ full) ** 2).sum())
    xx = np.c_[np.ones_like(xonly), xonly]
    rx = y2 - xx @ np.linalg.lstsq(xx, y2, rcond=None)[0]
    direct = (float(rx @ rx) - ssr_full) / float(rx @ rx)
    assert abs(partial_share(y2, xonly, cell, G * G) - direct) < 1e-10, (partial_share(y2, xonly, cell, G * G), direct)
    # (4) x carrying the tree exactly removes it: y = tree + noise, x = tree -> partial share near chance.
    #     (With x = tree + independent noise the cells still explain the tree part that x's noise hides: the partial
    #     share depends on the covariate's own measurement noise, which is why it is an upper bound.)
    o3, d3 = perm("partial", tree + 0.3 * rng.normal(size=tree.size), tree, iu, codes, rng, 200)
    assert abs(TI.adjusted_share(o3, d3.mean())) < 0.05, TI.adjusted_share(o3, d3.mean())
    print("selftest: all checks passed")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="run the checks on planted data and exit")
    ap.add_argument("--network", default="alexnet")
    for k in ("layer", "gram-dir", "pairs-dir", "trained-dir", "site-map", "taxonomy", "out"):
        ap.add_argument(f"--{k}")
    ap.add_argument("--untrained-dirs", help="comma-separated folders of the five untrained networks' matrices")
    ap.add_argument("--wiring-minus-representations", action="store_true",
                    help="add the difference W - A at each level, with its paired jackknife interval")
    ap.add_argument("--exclude-superordinate", metavar="GROUP",
                    help="leave out the categories of this superordinate group (section 2.3: mammal.n.01); implies "
                         "--wiring-minus-representations")
    ap.add_argument("--nperm-test", type=int, default=1000, help="relabellings for the null mean and the p value")
    ap.add_argument("--unit", choices=("category", "block50", "superordinate", "domain"), default="category",
                    help="the deletion and relabelling unit (default: category, as in the paper)")
    ap.add_argument("--check-deletions", type=int, default=20)
    ap.add_argument("--check-draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.nperm_test < 1000:
        ap.error("--nperm-test must be >= 1000")
    return run(a)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error))
