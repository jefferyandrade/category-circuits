"""Tree shares of the wiring and of the mean category representations at one layer, with their intervals, and the
crossover of the two across AlexNet's layers. tree_contrast.py (Fig. 6) and the Fig. 7 code use its estimators.

The wiring is the excess overlap over the sign- and weight-size-matched reference (from the layer's gram and
overlap_nulls.py's pairs); the representations are the cross-validated similarity of the dense network's category
means (S|cv:all) at the layer's activation site. A tree share is chance-corrected with its mean over relabellings of
all categories (the null mean), and these relabellings also give its permutation p value. Its delete-one-category
jackknife interval refits the node adjustment in every deletion and holds the null mean fixed; 20 deletions, with
2,000 relabellings each, measure how far the null mean would move (the perm_mean_check columns).

Usage: tree_infer.py --layer L --gram-dir DIR --pairs-dir DIR --matrices-dir DIR --site-map JSON --taxonomy CSV
           --out CSV
       tree_infer.py --crossover --gram-dir DIR --pairs-dir DIR --matrices-dir DIR --site-map JSON --taxonomy CSV
           --out CSV
       tree_infer.py --selftest
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
import category_resampling as PS  # noqa: E402
import pair_stats as PST  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import publish_file, validate_paths, check_outputs  # noqa: E402


NORMAL_EQUATION_RTOL = 1e-10


def backfit_adjust(M: np.ndarray, valid: np.ndarray, iters: int = 200,
                   return_convergence: bool = False):
    """The node adjustment: alpha + u_k + u_j removed by least squares over the valid off-diagonal pairs
    (backfitting). With a complete off-diagonal this equals tree_share.node_adjust. The fit has converged when the
    relative infinity-norm residual of the least-squares normal equations is at most NORMAL_EQUATION_RTOL (1e-10). A
    backfit that stalls is solved directly from the same normal equations; a fit that fails returns NaN."""
    V = valid & ~np.eye(M.shape[0], dtype=bool)
    Z = np.where(V, M, 0.0)
    n = V.sum(1).astype(np.float64)
    u = np.zeros(M.shape[0])
    if not V.any() or not np.isfinite(M[V]).all():
        out = np.full_like(M, np.nan, dtype=np.float64)
        return (out, False) if return_convergence else out
    alpha = Z.sum() / V.sum()
    rhs = np.r_[Z.sum() / 2, Z.sum(1)]
    scale = float(np.max(np.abs(rhs)))

    def converged(residual):
        r = np.where(V, residual, 0.0)
        normal_residual = np.r_[r.sum() / 2, r.sum(1)]
        return bool(np.isfinite(normal_residual).all() and
                    np.max(np.abs(normal_residual)) <= NORMAL_EQUATION_RTOL * scale)

    if M.shape[0] >= 3 and V.sum() == M.shape[0] * (M.shape[0] - 1):
        out = TS.node_adjust(M)
        ok = converged(out)
        return (out, ok) if return_convergence else out

    for _ in range(iters):
        alpha = (Z - (u[:, None] + u[None, :]) * V).sum() / V.sum()
        u_new = (Z - (alpha + u[None, :]) * V).sum(1) / np.maximum(n, 1)
        u_new -= u_new.mean()
        done = np.max(np.abs(u_new - u)) < 1e-12
        u = u_new
        if done:
            break
    alpha = (Z - (u[:, None] + u[None, :]) * V).sum() / V.sum()
    out = M - alpha - u[:, None] - u[None, :]
    ok = converged(out)
    if not ok:
        # Intercept and endpoint indicators: each unordered valid pair has unit weight.
        normal = np.empty((M.shape[0] + 1, M.shape[0] + 1), dtype=np.float64)
        normal[0, 0] = V.sum() / 2
        normal[0, 1:] = normal[1:, 0] = n
        normal[1:, 1:] = V.astype(np.float64) + np.diag(n)
        try:
            coef = np.linalg.lstsq(normal, rhs, rcond=None)[0]
            out = M - coef[0] - coef[1:, None] - coef[None, 1:]
            ok = converged(out)
        except np.linalg.LinAlgError:
            ok = False
    if not ok:
        out[:] = np.nan
    out[~V] = np.nan
    return (out, ok) if return_convergence else out


def formed_share(e: np.ndarray, iu, codes: np.ndarray) -> float:
    """The tree share (R^2 of the pair values explained by their cells), or NaN unless there are three categories, the
    pair values vary and at least two cells are occupied."""
    if codes.size < 3 or codes.size == 0:
        return np.nan
    ok = np.isfinite(e)
    if ok.sum() < 2 or np.unique(np.r_[iu[0][ok], iu[1][ok]]).size < 3:
        return np.nan
    v = e[ok]
    if np.all(v == v[0]):
        return np.nan
    G = int(codes.max()) + 1
    cell = TS.cell_index(codes, iu, G)
    if np.unique(cell[ok]).size < 2:
        return np.nan
    share = TS.between_cell_share(e, cell, G * G)
    return float(share) if np.isfinite(share) else np.nan


def adjusted_share(obs: float, pm: float) -> float:
    """The chance-corrected share (obs - pm) / (1 - pm)."""
    if not (np.isfinite(obs) and np.isfinite(pm)) or 1 - pm == 0:
        return np.nan
    value = (obs - pm) / (1 - pm)
    return float(value) if np.isfinite(value) else np.nan


def permutation_summary(obs: float, dist: np.ndarray, strict: bool = False) -> dict:
    """The permutation summary of a share: null mean, quantiles, p value and chance-corrected share. Only defined
    relabellings count, and the test needs at least 1,000 of them."""
    defined = dist[np.isfinite(dist)]
    pm = float(defined.mean()) if defined.size else np.nan
    adj = adjusted_share(obs, pm)
    variable = defined.size > 1 and np.ptp(defined) > 64*np.finfo(float).eps*max(1., abs(pm))
    formed = np.isfinite(obs) and np.isfinite(adj) and defined.size >= 1000 and variable
    if strict:
        formed = formed and defined.size == dist.size
    return {"status": "FORMED" if formed else "NOT_FORMED", "share": obs,
            "perm_mean": pm, "perm_q95": float(np.quantile(defined, .95)) if defined.size else np.nan,
            "perm_q99": float(np.quantile(defined, .99)) if defined.size else np.nan,
            "perm_max": float(defined.max()) if defined.size else np.nan, "n_perm": dist.size,
            "n_perm_defined": defined.size, "n_perm_undefined": dist.size - defined.size,
            "permutation_varies": bool(variable),
            "p_perm": (1 + int((defined >= obs).sum())) / (1 + defined.size) if formed else np.nan,
            "share_adjusted": adj if formed else np.nan}


def shares(e: np.ndarray, iu, codes_by_level: dict, rng, nperm: int, keep_dist: bool = False, blocks=None, unit="category"):
    """Observed share per level, permutation mean (and distribution if asked)."""
    out = {}
    for level, codes in codes_by_level.items():
        obs = formed_share(e, iu, codes)
        dist = (np.array([formed_share(e, iu, rng.permutation(codes)) for _ in range(nperm)])
                if blocks is None else PST.label_distribution(
                    codes, blocks, lambda labels: formed_share(e, iu, labels), rng, nperm, unit=unit))
        defined = dist[np.isfinite(dist)]
        out[level] = (obs, dist if keep_dist else float(defined.mean()) if defined.size == dist.size and defined.size else np.nan)
    return out


def lens_matrices(a, layer, tax):
    z = np.load(Path(a.gram_dir) / f"gram_{layer}.npz")
    pz = np.load(Path(a.pairs_dir) / f"pairs_{layer}.npz")
    cats = z["categories"]
    assert np.array_equal(cats, pz["categories"])
    K = cats.size
    I = (z["I_all"] if "I_all" in z.files else z["I"]).astype(np.float64)[:K, :K]
    d = np.diag(I)
    S = d[:, None] + d[None, :]
    J = np.where(S - I > 0, I / np.where(S - I > 0, S - I, 1), np.nan)
    W = J - pz["J0_magnitude"].astype(np.float64)[:K, :K]
    site = json.loads(Path(a.site_map).read_text())[layer]
    m = np.load(Path(a.matrices_dir) / f"matrices_{site}.npz")
    acats = np.asarray(m["class_ids"], np.int64)
    A = np.asarray(m["S|cv:all"], np.float64)
    A0 = A.copy(); np.fill_diagonal(A0, 0.0)
    abad = (~np.isfinite(A0)).sum(1) > A0.shape[0] // 2
    common = np.intersect1d(cats, acats[~abad])
    wi, ai = np.searchsorted(cats, common), np.searchsorted(acats, common)
    W, A = W[np.ix_(wi, wi)], A[np.ix_(ai, ai)]
    for M in (W, A):
        np.fill_diagonal(M, np.nan)
    return common, {"wiring": W, "act_cv": A}, site


def category_units(tax, categories, unit="category"):
    """The deletion unit of each category: the category itself, or its taxonomy group at the level named by unit."""
    if unit == "category":
        return np.arange(len(categories))
    column = "animate" if unit == "domain" and "domain" not in tax else unit
    if column not in tax:
        raise ValueError("taxonomy is missing %s" % column)
    return tax.loc[categories, column].astype(str).to_numpy()


def adjusted_inference(statistic, labels, blocks=None, nperm=10000, seed=0,
                       unit="category", check_deletions=20, check_draws=2000):
    """The chance-corrected tree share with its permutation test and jackknife interval.

    statistic(indices, labels) returns the uncorrected share of the categories at indices under labels, refitting the
    node adjustment. The null mean comes from nperm relabellings of all categories and is held fixed in every
    deletion; check_deletions deletions with check_draws relabellings each measure how far it would move.
    """
    labels = np.asarray(labels)
    n = len(labels)
    blocks = np.arange(n) if blocks is None else np.asarray(blocks)
    groups = np.arange(n) if unit == "category" else blocks
    full = np.arange(n)
    obs = statistic(full, labels)
    distribution = PST.label_distribution(labels, blocks,
        lambda lab: statistic(full, lab), np.random.default_rng(seed), nperm, unit=unit)
    summary = permutation_summary(obs, distribution, strict=True)
    pm = summary["perm_mean"]
    def fixed_mean(idx):
        return adjusted_share(statistic(idx, labels[idx]), pm)
    jack = PST.indexed_jackknife(fixed_mean, blocks, unit=unit)
    checked, changes = [], []
    chosen = np.random.default_rng(seed + 741).permutation(np.unique(groups))[:check_deletions]
    for group in chosen:
        idx = np.flatnonzero(groups != group)
        dist = PST.label_distribution(labels[idx], blocks[idx],
            lambda lab: statistic(idx, lab), np.random.default_rng(seed + 1907),
            check_draws, unit=unit)
        delta = abs(float(dist.mean()) - pm) if np.isfinite(dist).all() else np.nan
        checked.append(group.item() if hasattr(group, "item") else group)
        changes.append(delta)
    se = jack.get("se", np.nan)
    maximum = max(changes) if changes and np.isfinite(changes).all() else np.nan
    ratio = maximum / se if np.isfinite(se) and se > 0 else (0. if maximum == 0 and se == 0 else np.inf)
    summary.update(jack)
    summary["status"] = "FORMED" if np.isfinite(summary.get("ci_low", np.nan)) else "NOT_FORMED"
    summary.update(share_adjusted=adjusted_share(obs, pm), lo95_G=jack.get("ci_low"),
        hi95_G=jack.get("ci_high"), perm_mean_held_fixed=True,
        perm_mean_check_deletions=len(checked), perm_mean_check_draws=check_draws,
        perm_mean_check_max_change=maximum, perm_mean_check_ratio=ratio,
        perm_mean_check_flag=bool(not np.isfinite(ratio) or ratio > .1),
        perm_mean_check_seed=seed + 741)
    return summary


def _prepared_share(values, ij):
    """The share as a function of the labels, with the centred pair values computed once for all relabellings."""
    values = np.asarray(values, float)
    finite = np.isfinite(values)
    endpoints = (ij[0][finite], ij[1][finite])
    values = values[finite]
    formed = len(values) >= 2 and np.unique(np.r_[endpoints[0], endpoints[1]]).size >= 3
    centred = values-values.mean() if len(values) else values
    denominator = float(np.square(centred).sum())
    def share(labels):
        if not formed or denominator == 0 or len(labels) < 3:
            return np.nan
        groups = int(np.max(labels))+1
        cell = TS.cell_index(labels, endpoints, groups)
        counts = np.bincount(cell, minlength=groups*groups)
        if np.count_nonzero(counts) < 2:
            return np.nan
        sums = np.bincount(cell, weights=centred, minlength=groups*groups)
        return float(np.sum(sums*sums/np.maximum(counts,1))/denominator)
    return share


def _matrix_share_statistic(source, adjusted=True):
    """statistic(indices, labels) for one pair matrix, caching the node-adjusted matrix of the last two sets of
    categories, since relabellings reuse it."""
    cache = {}
    def statistic(idx, labels):
        key = tuple(idx)
        if key not in cache:
            if len(cache) >= 2:
                cache.pop(next(iter(cache)))
            matrix = source[np.ix_(idx, idx)]
            if adjusted:
                matrix, ok = backfit_adjust(matrix, ~np.eye(len(idx), dtype=bool), return_convergence=True)
                if not ok:
                    cache[key] = None
                    return np.nan
            ij = np.triu_indices(len(idx), 1)
            cache[key] = _prepared_share(matrix[ij], ij)
        return np.nan if cache[key] is None else cache[key](labels)
    return statistic


def declared_families(rows, network, layers, unit="category"):
    """Holm-adjust the primary variants (node-adjusted wiring and representations) over levels and layers. A layer
    whose input is missing keeps an empty row, so the family has the same size."""
    # Rows read from other layers' outputs can include empty rows of absent layers; an estimate replaces the empty
    # row of the same layer, variant and level.
    merged = {}
    for row in rows:
        key = (row.get("layer"),row.get("variant", "lens:%s|%s" % (row.get("lens"),row.get("quantity"))),
               row.get("level"),row.get("unit",unit))
        previous = merged.get(key)
        if previous is None or previous.get("estimate") is None or pd.isna(previous.get("estimate")):
            merged[key] = row
        elif row.get("estimate") is not None and not pd.isna(row.get("estimate")):
            raise ValueError("a layer, variant and level appear twice")
    rows = list(merged.values())
    family = "%s: primary variants x levels x sites" % network
    primary = ("wiring_mag_adj", "act_cv_adj")
    present = set()
    for row in rows:
        if row.get("network", network) != network:
            raise ValueError("the family's rows must all be of the network named")
        row["network"] = network
        key = (row.get("layer"), row.get("variant"), row.get("level"))
        is_primary = key[1] in primary
        row["family"] = family if is_primary else "descriptive"
        if not is_primary:
            row["p"] = row["p_holm"] = None
        else:
            present.add(key)
    for layer in layers:
        for variant in primary:
            for level, _ in TS.LEVELS:
                if (layer, variant, level) not in present:
                    rows.append(PST.result_row(None, family=family, unit=unit, scope="D",
                        n_categories=0, n_units=0, method="not formed: site input not supplied",
                        network=network, layer=layer, variant=variant, level=level,
                        status="NOT_FORMED", effect_unit="fraction"))
    adjusted = PS.adjust_families(rows)
    for row in adjusted:
        row["family_planned_tests"] = len(layers)*len(primary)*len(TS.LEVELS) if row["family"] == family else 0
    return adjusted


def crossover_contrast(layer_matrices, blocks, labels, nperm=10000, seed=0,
                       unit="category", network="alexnet"):
    """The crossover: the wiring-minus-representation difference of chance-corrected tree shares averaged over fc6
    and fc7, minus its average over conv3-conv5, with a joint delete-one-category jackknife interval. Each layer's
    and measure's null mean is computed once on all categories and held fixed."""
    required = ("conv3", "conv4", "conv5", "fc6", "fc7")
    if set(layer_matrices) != set(required):
        raise ValueError("crossover requires conv3-conv5 and fc6-fc7 on common categories")
    functions, means = {}, {}
    all_idx = np.arange(len(labels))
    for layer in required:
        for lens in ("wiring", "act_cv"):
            fn = _matrix_share_statistic(layer_matrices[layer][lens])
            functions[(layer, lens)] = fn
            d = PST.label_distribution(labels, blocks, lambda lab: fn(all_idx, lab),
                np.random.default_rng(seed), nperm, unit=unit)
            means[(layer, lens)] = float(d.mean())
    def statistic(idx):
        gaps = {}
        for layer in required:
            estimates = [adjusted_share(functions[(layer, lens)](idx, np.asarray(labels)[idx]),
                                       means[(layer, lens)]) for lens in ("wiring", "act_cv")]
            gaps[layer] = estimates[0] - estimates[1]
        return float(np.mean([gaps[l] for l in ("fc6", "fc7")]) -
                     np.mean([gaps[l] for l in ("conv3", "conv4", "conv5")]))
    jk = PST.indexed_jackknife(statistic, blocks, unit=unit)
    return PST.result_row(jk.get("estimate"), ci_low=jk.get("ci_low"), ci_high=jk.get("ci_high"),
        family="%s: crossover" % network, unit=unit, effect_unit="fraction", scope="G",
        n_categories=len(labels), n_units=len(labels) if unit == "category" else len(np.unique(blocks)),
        method="joint delete-one-%s jackknife; full-data permutation means held fixed" % unit,
        perm_mean_held_fixed=True, contrast="mean(fc6,fc7)-mean(conv3,conv4,conv5), wiring minus activations")


def run_crossover(a):
    """The crossover on AlexNet's conv3-fc7, on the categories every layer has."""
    check_outputs(files=[a.out], inputs=[a.taxonomy, a.site_map, a.gram_dir, a.pairs_dir, a.matrices_dir])
    tax = TS.load_tax(a.taxonomy)
    loaded = {layer: lens_matrices(a, layer, tax) for layer in ("conv3", "conv4", "conv5", "fc6", "fc7")}
    categories = loaded["conv3"][0]
    for cats, _, _ in loaded.values():
        categories = np.intersect1d(categories, cats)
    matrices = {}
    for layer, (cats, pair_matrices, _) in loaded.items():
        ii = np.searchsorted(cats, categories)
        matrices[layer] = {lens: m[np.ix_(ii, ii)] for lens, m in pair_matrices.items()}
    unit = getattr(a, "unit", "category")
    blocks = category_units(tax, categories, unit)
    rows = []
    for level, column in TS.LEVELS:
        labels = pd.factorize(tax.loc[categories, column].astype(str))[0]
        rows.append({"network": getattr(a, "network", "alexnet"), "level": level,
            **crossover_contrast(matrices, blocks, labels, a.nperm_test, a.seed, unit,
                                  getattr(a, "network", "alexnet"))})
    publish_file(pd.DataFrame(PS.adjust_families(rows)), Path(a.out))
    return 0


def run(a) -> int:
    """Tree shares of the wiring and the representations at one layer, and their differences, Holm-adjusted."""
    if a.nperm_test < 1000:
        raise ValueError("nperm-test must be >= 1000")
    out = Path(a.out)
    inputs = [a.taxonomy, a.site_map, Path(a.gram_dir) / ("gram_%s.npz" % a.layer),
              Path(a.pairs_dir) / ("pairs_%s.npz" % a.layer)]
    check_outputs(files=[out], inputs=[*inputs, a.gram_dir, a.pairs_dir, a.matrices_dir,
        *filter(None, (getattr(a, "family_inputs", None) or "").split(","))])
    tax = TS.load_tax(a.taxonomy)
    cats, base, site = lens_matrices(a, a.layer, tax)
    validate_paths([out], inputs + [Path(a.matrices_dir) / ("matrices_%s.npz" % site)])
    unit, network = getattr(a, "unit", "category"), getattr(a, "network", "alexnet")
    blocks = category_units(tax, cats, unit)
    codes = {lvl: pd.factorize(tax.loc[cats, col].astype(str).values)[0] for lvl, col in TS.LEVELS}
    rows, inferred = [], {}
    for lens, source in base.items():
        for adjusted in (True, False):
            name = ("wiring_mag" if lens == "wiring" else "act_cv") + ("_adj" if adjusted else "_raw")
            statistic = _matrix_share_statistic(source, adjusted)
            for li, (level, labels) in enumerate(codes.items()):
                result = adjusted_inference(statistic, labels, blocks, a.nperm_test, a.seed + li, unit,
                    getattr(a, "check_deletions", 20), getattr(a, "check_draws", 2000))
                inferred[(name, level)] = result
                rows.append({"network": network, "layer": a.layer, "site": site, "variant": name,
                    "level": level, **{k:v for k,v in result.items() if k not in ("deleted", "replicates")},
                    **PST.result_row(result["share_adjusted"], ci_low=result.get("ci_low"),
                        ci_high=result.get("ci_high"), p=result["p_perm"], family="descriptive", unit=unit,
                        effect_unit="fraction", scope="G" if np.isfinite(result.get("ci_low", np.nan)) else "D",
                        n_categories=len(cats), n_units=len(cats) if unit == "category" else len(np.unique(blocks)),
                        method="delete-one-%s jackknife (t interval); %s label permutation; full-data null mean fixed" % (unit, unit))})
    for kind in ("adj", "raw"):
        for level in codes:
            w, act = inferred[("wiring_mag_" + kind, level)], inferred[("act_cv_" + kind, level)]
            dw, da = np.asarray(w.get("replicates", [])), np.asarray(act.get("replicates", []))
            value = w["share_adjusted"] - act["share_adjusted"]
            delta = dw - da
            if len(delta) > 1 and np.isfinite(delta).all():
                from scipy.stats import t
                se = np.sqrt((len(delta)-1)/len(delta)*np.square(delta-delta.mean()).sum())
                width = t.ppf(.975, len(delta)-1)*se
                low, high = value-width, value+width
            else:
                low = high = np.nan
            rows.append({"network": network, "layer": a.layer, "site": site, "level": level,
                "variant": "wiring_minus_activation_" + kind, "share_adjusted": value,
                "status": "FORMED" if np.isfinite(value) else "NOT_FORMED",
                "n_perm_undefined": 0 if np.isfinite(value) else a.nperm_test,
                "p_perm": np.nan, "lo95_G": low, "hi95_G": high,
                **PST.result_row(value, ci_low=low, ci_high=high, family="descriptive", unit=unit,
                    effect_unit="fraction", scope="G" if np.isfinite(low) else "D", n_categories=len(cats),
                    n_units=len(cats) if unit == "category" else len(np.unique(blocks)),
                    method="paired delete-one-%s jackknife; full-data null means fixed" % unit)})
    if getattr(a, "family_inputs", None):
        for path in a.family_inputs.split(","):
            rows.extend(pd.read_csv(path).to_dict("records"))
    layers = list(json.loads(Path(a.site_map).read_text())) if Path(a.site_map).is_file() else [a.layer]
    rows = declared_families(rows, network, layers, unit)
    for row in rows:
        row.update(runtime_n_categories=len(cats), runtime_nperm=a.nperm_test,
                   runtime_check_draws=getattr(a, "check_draws", 2000),
                   runtime_check_deletions=min(getattr(a, "check_deletions", 20), len(np.unique(blocks))))
    publish_file(pd.DataFrame(rows), out)
    print("wrote %s; layer %s" % (out, a.layer))
    return 0


def selftest() -> int:
    rng = np.random.default_rng(5)
    K = 60
    u = rng.normal(size=K)
    M = 0.2 + u[:, None] + u[None, :] + 0.01 * rng.normal(size=(K, K))
    M = (M + M.T) / 2
    A = TS.node_adjust(M)
    B = backfit_adjust(M, np.ones((K, K), bool))
    assert np.nanmax(np.abs(A - B)) < 1e-8, np.nanmax(np.abs(A - B))
    # permutation p under the null is not small; under a planted effect it is minimal
    cats = np.arange(200)
    codes = {"g": cats // 10}
    noise = rng.normal(size=(200, 200)); noise = (noise + noise.T) / 2
    iu = np.triu_indices(200, 1)
    obs, dist = shares(noise[iu], iu, codes, rng, 200, keep_dist=True)["g"]
    p0 = (1 + (dist >= obs).sum()) / 201
    planted = noise + 1.0 * ((cats[:, None] // 10) == (cats[None, :] // 10))
    obs1, dist1 = shares(planted[iu], iu, codes, rng, 200, keep_dist=True)["g"]
    p1 = (1 + (dist1 >= obs1).sum()) / 201
    assert p0 > 0.01 and p1 == 1 / 201, (p0, p1)
    print("selftest: all checks passed")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="run the checks on planted data and exit")
    ap.add_argument("--network", default="alexnet")
    ap.add_argument("--crossover", action="store_true", help="the crossover across conv3-fc7 instead of one layer")
    ap.add_argument("--family-inputs", help="comma-separated outputs of other layers, Holm-adjusted together with "
                                            "this layer's rows")
    ap.add_argument("--unit", choices=("category", "block50", "superordinate", "domain"), default="category",
                    help="the deletion and relabelling unit (default: category, as in the paper)")
    ap.add_argument("--check-deletions", type=int, default=20)
    ap.add_argument("--check-draws", type=int, default=2000)
    for k in ("layer", "gram-dir", "pairs-dir", "matrices-dir", "site-map", "taxonomy", "out"):
        ap.add_argument(f"--{k}")
    ap.add_argument("--nperm-test", type=int, default=1000, help="relabellings for the null mean and the p value")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.nperm_test < 1000:
        ap.error("--nperm-test must be >= 1000")
    return selftest() if a.selftest else run_crossover(a) if a.crossover else run(a)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error))
