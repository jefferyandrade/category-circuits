"""Category-tree structure of category-pair similarity: the share explained by the category tree, at each level of
the tree and each layer, for the wiring and for the mean category representations, with the reliability of each.
Section 2.3 uses the wiring's superordinate share at every layer; tree_contrast.py builds Fig. 6 on these estimators.

Pair values (higher = more similar):
  wiring      e_kj = J_kj - J0_kj: J the Jaccard overlap of two circuits' kept weights in the layer (from its gram),
              J0 the sign- and weight-size-matched reference ("magnitude", the paper's) or the size-matched one
              ("uniform"), from overlap_nulls.py's pairs_<layer>.npz;
  activation  S_kj, the similarity of the dense network's mean category representations over the training images
              (matrices_<site>.npz): 'cv' the cross-validated similarity of the means of two image halves (the
              paper's), 'rho' the cosine similarity of the means after subtracting their grand mean.
Node adjustment (the paper's): a fit of alpha + u_k + u_j by least squares over the off-diagonal pairs is removed, so
that a category's average similarity to all others does not count as tree structure. Unadjusted values are given
beside it.
Levels: domain (animate or inanimate, 2 groups), superordinate (32 groups), block50 (131 subgroups).
share = the R^2 of the pair values explained by their cell (the unordered pair of the two categories' groups);
permuted = its mean over NPERM relabellings of the categories that keep the group sizes;
adjusted = (share - permuted) / (1 - permuted), the chance-corrected share.
Reliability: the Pearson correlation, over cells with at least MIN_PAIRS pairs, of the cell means from two
independent measurements:
  wiring      P against its repeat extraction P-redraw (on the shared categories, where the gram holds both);
  activation  one image half against the other (all categories, and the shared categories).
Scope: D (exact over the categories present); the reliability measures extraction noise (wiring) or image noise
(activation).

Usage:
  tree_share.py wiring --network NET --gram-dir DIR --pairs-dir DIR --taxonomy CSV --out CSV [--layers ...]
      (DIR holds gram_<layer>.npz; pairs-dir holds pairs_<layer>.npz from overlap_nulls.py, in the same row order)
  tree_share.py activation --network NET --matrices-dir DIR --site-map JSON --taxonomy CSV --out CSV
  tree_share.py selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import publish_file, check_outputs

LEVELS = (("domain", "animate"), ("superordinate", "superordinate"), ("block50", "block50"))
NPERM, MIN_PAIRS, SEED = 50, 3, 0


def node_adjust(S: np.ndarray) -> np.ndarray:
    """Residual of the off-diagonal of symmetric S after alpha + u_k + u_j (least squares, sum u = 0)."""
    S = np.asarray(S, dtype=np.float64)
    K = S.shape[0]
    off = ~np.eye(K, dtype=bool)
    if not np.isfinite(S[off]).all():
        raise ValueError("node adjustment needs a complete off-diagonal")
    Z = np.where(off, S, 0.0)
    alpha = Z.sum() / (K * (K - 1))
    R = Z.sum(1)
    u = (R - (K - 1) * alpha) / (K - 2)
    out = S - alpha - u[:, None] - u[None, :]
    out[~off] = np.nan
    return out


def cell_index(codes: np.ndarray, iu, G: int) -> np.ndarray:
    """The cell of each pair: the unordered pair of its two categories' groups, as one index."""
    a, b = codes[iu[0]], codes[iu[1]]
    return np.minimum(a, b) * G + np.maximum(a, b)


def between_cell_share(e: np.ndarray, cell: np.ndarray, ncell: int) -> float:
    """The share of the variance of the finite pair values explained by their cells (R^2)."""
    ok = np.isfinite(e)
    e, cell = e[ok], cell[ok]
    n = np.bincount(cell, minlength=ncell)
    s = np.bincount(cell, e, minlength=ncell)
    grand = e.mean()
    m = np.where(n > 0, s / np.maximum(n, 1), 0.0)
    return float((n * (m - grand) ** 2).sum() / ((e - grand) ** 2).sum())


def cell_means(e: np.ndarray, cell: np.ndarray, ncell: int):
    ok = np.isfinite(e)
    n = np.bincount(cell[ok], minlength=ncell)
    s = np.bincount(cell[ok], e[ok], minlength=ncell)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan), n


def tree_rows(M: np.ndarray, cats: np.ndarray, tax: pd.DataFrame, rng) -> tuple[list, dict]:
    """Shares for one symmetric pair matrix M over categories cats. Returns rows and the cell means per level."""
    iu = np.triu_indices(len(cats), 1)
    e = M[iu]
    rows, cells = [], {}
    for level, col in LEVELS:
        codes = pd.factorize(tax.loc[cats, col].astype(str).values)[0]
        G = int(codes.max()) + 1
        cell = cell_index(codes, iu, G)
        share = between_cell_share(e, cell, G * G)
        perm = float(np.mean([between_cell_share(e, cell_index(rng.permutation(codes), iu, G), G * G)
                              for _ in range(NPERM)]))
        within = codes[iu[0]] == codes[iu[1]]
        rows.append({"level": level, "n_groups": G, "share": share, "share_permuted": perm,
                     "share_adjusted": (share - perm) / (1 - perm),
                     "contrast_within_minus_between": float(np.nanmean(e[within]) - np.nanmean(e[~within]))})
        cells[level] = cell_means(e, cell, G * G)
    cells["pairs"] = (e, np.ones_like(e))
    return rows, cells


def reliability(c1: dict, c2: dict) -> dict:
    """Pearson correlation of two measurements' cell means at each level (and of the pair values), with the count."""
    out = {}
    for key in list(c1):
        m1, n1 = c1[key]
        m2, n2 = c2[key]
        ok = np.isfinite(m1) & np.isfinite(m2) & (n1 >= MIN_PAIRS if key != "pairs" else True) & \
            (n2 >= MIN_PAIRS if key != "pairs" else True)
        out[key] = (float(np.corrcoef(m1[ok], m2[ok])[0, 1]) if ok.sum() > 3 else np.nan, int(ok.sum()))
    return out


def load_tax(path) -> pd.DataFrame:
    return pd.read_csv(path).set_index("category")


def run_wiring(a) -> pd.DataFrame:
    tax = load_tax(a.taxonomy)
    rng = np.random.default_rng(SEED)
    rows = []
    layers = a.layers or [p.stem[len("gram_"):] for p in sorted(Path(a.gram_dir).glob("gram_*.npz"))]
    for L in layers:
        z = np.load(Path(a.gram_dir) / f"gram_{L}.npz")
        pz = np.load(Path(a.pairs_dir) / f"pairs_{L}.npz")
        cats = z["categories"]
        has_r = "I_all" in z.files and "redraw_categories" in z.files and z["redraw_categories"].size > 0
        I = (z["I_all"] if has_r else z["I"]).astype(np.float64)
        rc = z["redraw_categories"] if has_r else np.zeros(0, np.int64)
        if "redraw_categories" in z.files and z["redraw_categories"].size and not has_r:
            raise ValueError(f"{L}: redraw declared without I_all")
        if not (np.all(np.diff(cats) > 0) and np.isin(rc, cats).all()):
            raise ValueError(f"{L}: categories not sorted/unique or redraws not a subset")
        if not (np.array_equal(cats, pz["categories"]) and np.array_equal(rc, pz["redraw_categories"])):
            raise ValueError(f"{L}: gram and pairs rows differ")
        a_ = np.diag(I)
        S = a_[:, None] + a_[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            J = np.where(S - I > 0, I / (S - I), np.nan)
        K = cats.size
        for null in ("magnitude", "uniform"):
            E = J - pz[f"J0_{null}"].astype(np.float64)
            for adj in (True, False):
                lens = f"wiring_{null}"
                sets = {"all": (np.arange(K), cats)}
                if rc.size:
                    pidx = np.searchsorted(cats, rc)
                    sets.update({"shared100_P": (pidx, rc), "shared100_Predraw": (K + np.arange(rc.size), rc)})
                cells = {}
                for sname, (idx, cc) in sets.items():
                    M = E[np.ix_(idx, idx)]
                    M = node_adjust(M) if adj else M
                    r, cells[sname] = tree_rows(M, cc, tax, rng)
                    rows += [{"network": a.network, "lens": lens, "layer": L, "node_adjusted": adj, "set": sname,
                              "n_categories": len(cc), **x} for x in r]
                if rc.size:
                    for key, (v, n) in reliability(cells["shared100_P"], cells["shared100_Predraw"]).items():
                        rows.append({"network": a.network, "lens": lens, "layer": L, "node_adjusted": adj,
                                     "set": "shared100_P_vs_Predraw", "level": key, "reliability": v,
                                     "n_cells": n, "n_categories": rc.size})
        print(f"wiring {L} done", flush=True)
    return pd.DataFrame(rows)


def run_activation(a) -> pd.DataFrame:
    tax = load_tax(a.taxonomy)
    rng = np.random.default_rng(SEED)
    site_map = json.loads(Path(a.site_map).read_text())       # {"conv1": "features.1", ...}
    shared = np.array(sorted(int(x) for x in Path(a.shared100).read_text().split())) if a.shared100 else None
    rows = []
    for L, site in site_map.items():
        z = np.load(Path(a.matrices_dir) / f"matrices_{site}.npz")
        cats = np.asarray(z["class_ids"], dtype=np.int64)
        for kind in ("cv", "rho"):
            for adj in (True, False):
                lens = f"activation_{kind}"
                def prep(key, idx):
                    M = np.asarray(z[f"S|{kind}:{key}"], dtype=np.float64)[np.ix_(idx, idx)].copy()
                    np.fill_diagonal(M, np.nan)
                    return node_adjust(M) if adj else M
                # Drop, one at a time, the category with the most undefined pairs in any of the three matrices of
                # this kind (all images and each half) until the rest is complete, so that the full estimate and the
                # halves use the same categories; dropped_undefined counts the categories dropped
                bad = np.zeros(cats.size, bool)
                nf = np.zeros((cats.size, cats.size), bool)
                for key in ("all", "img_half0", "img_half1"):
                    M0 = np.asarray(z[f"S|{kind}:{key}"], dtype=np.float64)
                    np.fill_diagonal(M0, 0.0)
                    nf |= ~np.isfinite(M0)
                while True:
                    sub = nf[np.ix_(~bad, ~bad)]
                    if not sub.any():
                        break
                    worst = np.flatnonzero(~bad)[np.argmax(sub.sum(1))]
                    bad[worst] = True
                keep = np.flatnonzero(~bad)
                for key in ("all", "img_half0", "img_half1"):
                    M0 = np.asarray(z[f"S|{kind}:{key}"], dtype=np.float64)[np.ix_(keep, keep)]
                    np.fill_diagonal(M0, 0.0)
                    if not np.isfinite(M0).all():
                        raise ValueError(f"{site} {kind}:{key}: scattered undefined pairs after dropping rows")
                sets = [("all", keep)]
                if shared is not None:
                    pos = {int(c): i for i, c in enumerate(cats)}           # explicit membership
                    missing = [int(c) for c in shared if int(c) not in pos]
                    if missing:
                        raise ValueError(f"shared categories absent from the capture: {missing[:5]}")
                    sets.append(("shared100", np.intersect1d([pos[int(c)] for c in shared], keep)))
                for sname, idx in sets:
                    cc = cats[idx]
                    r, _ = tree_rows(prep("all", idx), cc, tax, rng)
                    rows += [{"network": a.network, "lens": lens, "layer": L, "site": site, "node_adjusted": adj,
                              "set": sname, "n_categories": len(cc), "dropped_undefined": int(bad.sum()), **x}
                             for x in r]
                    r0, c0 = tree_rows(prep("img_half0", idx), cc, tax, rng)
                    r1, c1 = tree_rows(prep("img_half1", idx), cc, tax, rng)
                    for hname, rr in (("img_half0", r0), ("img_half1", r1)):
                        rows += [{"network": a.network, "lens": lens, "layer": L, "site": site,
                                  "node_adjusted": adj, "set": f"{sname}_{hname}", "n_categories": len(cc), **x}
                                 for x in rr]
                    for key, (v, n) in reliability(c0, c1).items():
                        rows.append({"network": a.network, "lens": lens, "layer": L, "site": site,
                                     "node_adjusted": adj, "set": f"{sname}_half0_vs_half1", "level": key,
                                     "reliability": v, "n_cells": n, "n_categories": len(cc)})
        print(f"activation {L} ({site}) done", flush=True)
    return pd.DataFrame(rows)


def selftest() -> int:
    rng = np.random.default_rng(3)
    # node adjustment removes a pure additive structure exactly
    K = 40
    u = rng.normal(size=K)
    S = 0.3 + u[:, None] + u[None, :]
    R = node_adjust(S)
    assert np.nanmax(np.abs(R)) < 1e-12
    # planted tree effect: groups of 5 within superordinates of 20; share recovered, permuted near the small-cell bias
    cats = np.arange(200)
    tax = pd.DataFrame({"category": cats, "animate": (cats < 100).astype(int), "superordinate": cats // 20,
                        "block50": cats // 5}).set_index("category")
    noise = rng.normal(size=(200, 200))
    noise = (noise + noise.T) / 2
    within_b = (cats[:, None] // 5) == (cats[None, :] // 5)
    M = noise + 2.0 * within_b
    rows, _ = tree_rows(M, cats, tax, rng)
    r = {x["level"]: x for x in rows}
    assert 0.10 < r["block50"]["share_adjusted"] < 0.17 and abs(r["block50"]["contrast_within_minus_between"] - 2) < 0.1, r["block50"]  # expected ~0.135
    assert abs(r["domain"]["share_adjusted"]) < 0.02, r["domain"]
    # no structure: adjusted shares near 0 at every level
    rows0, _ = tree_rows(noise, cats, tax, rng)
    assert all(abs(x["share_adjusted"]) < 0.03 for x in rows0), rows0
    # reliability: two noisy copies of the same planted structure correlate at the block level
    _, c1 = tree_rows(M, cats, tax, rng)
    n2 = rng.normal(size=(200, 200))
    _, c2 = tree_rows(2.0 * within_b + (n2 + n2.T) / 2, cats, tax, rng)
    rel = reliability(c1, c2)
    assert rel["block50"][0] > 0.8 and abs(rel["pairs"][0]) < 0.9, rel
    print("selftest: all checks passed")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("wiring", "activation", "selftest"))
    ap.add_argument("--network", default="alexnet")
    ap.add_argument("--gram-dir")
    ap.add_argument("--pairs-dir")
    ap.add_argument("--layers", nargs="*")
    ap.add_argument("--matrices-dir")
    ap.add_argument("--site-map")
    ap.add_argument("--shared100")
    ap.add_argument("--taxonomy")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.mode == "selftest":
        return selftest()
    out = Path(a.out)
    check_outputs(files=[out], inputs=[a.gram_dir, a.pairs_dir, a.matrices_dir, a.site_map, a.shared100, a.taxonomy])
    df = run_wiring(a) if a.mode == "wiring" else run_activation(a)
    publish_file(df, out)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error))
