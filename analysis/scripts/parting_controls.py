"""Section 2.3, controls: does the relation between how far circuits part (a layer's kept fraction) and how much of
the category tree its wiring carries (the node-adjusted superordinate tree share) survive (a) each layer's size (its
number of prunable weights) and (b) how reliably its wiring is measured (the Pearson r of the cell means between the
P circuits and their repeat extraction on the shared 100: superordinate, node-adjusted, lens wiring_magnitude, the
lens of parting_tree.py's tree share, from tree_share.py wiring)?
Descriptive, as parting_tree_stats.py: Spearman correlations and rank partial correlations (ranks residualized on the
ranked covariates by least squares), pooled over the layers of four networks (main path, and parted layers) and within
each network; layers within a network are not independent, so the p values are descriptive.
Usage: parting_controls.py --parting parting_tree.csv --reliability CSV [CSV ...] --out OUT.csv (OUT.csv must not
exist; --reliability takes tree_share.py wiring outputs)."""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import check_outputs, write_file

ALEX = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7"]
VGG = ["conv3_1", "conv3_2", "conv4_1", "conv4_2", "conv5_1", "conv5_2", "fc6", "fc7"]


def order_key(net, layer):
    if net == "alexnet":
        return (ALEX.index(layer),)
    if net == "vgg11":
        return (VGG.index(layer),)
    if layer.startswith("stem"):
        return (0, 0, 0, 0)
    m = re.match(r"stage(\d+)\.block(\d+)\.(conv(\d+)|projection)", layer)
    return (1, int(m.group(1)), int(m.group(2)), int(m.group(4)) if m.group(4) else 0)


def with_depth(d):
    parts = []
    for net, g in d.groupby("network"):
        g = g.copy()
        order = sorted(g.layer, key=lambda l: order_key(net, l))
        g["depth"] = [order.index(l) / (len(order) - 1) for l in g.layer]
        parts.append(g)
    return pd.concat(parts)


def rank_partial(x, y, covariates):
    """Pearson r of the ranks of x and y after least-squares removal of the ranked covariates (with intercept).
    The p value is the partial-correlation t test with df = n - rank(design) - 1 (design = intercept + ranked
    covariates); undefined (nan) when a ranked response lies in the covariates' span."""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    z = np.column_stack([np.ones(len(rx))] + [stats.rankdata(c) for c in covariates])
    scale = [np.linalg.norm(v - v.mean()) for v in (rx, ry)]
    if min(scale) == 0:  # a constant response lies in the intercept's span
        return float("nan"), float("nan")
    rx = rx - z @ np.linalg.lstsq(z, rx, rcond=None)[0]
    ry = ry - z @ np.linalg.lstsq(z, ry, rcond=None)[0]
    if min(np.linalg.norm(rx) / scale[0], np.linalg.norm(ry) / scale[1]) < 1e-9:
        return float("nan"), float("nan")
    r = float(np.dot(rx, ry) / (np.linalg.norm(rx) * np.linalg.norm(ry)))
    df = len(rx) - np.linalg.matrix_rank(z) - 1
    if df < 1:
        return r, float("nan")
    t = r * np.sqrt(df / max(1e-300, 1.0 - r * r))
    return r, float(2 * stats.t.sf(abs(t), df))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parting", required=True)
    ap.add_argument("--reliability", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    check_outputs(files=[a.out], inputs=[a.parting, *a.reliability])
    if os.path.lexists(a.out):
        raise FileExistsError(a.out)
    d = pd.read_csv(a.parting)
    rel = pd.concat([pd.read_csv(f) for f in a.reliability], ignore_index=True)
    rel = rel[(rel.set == "shared100_P_vs_Predraw") & (rel.level == "superordinate") & (rel.lens == "wiring_magnitude")
              & (rel.node_adjusted.astype(str) == "True")
              & rel.reliability.notna()][["network", "layer", "reliability"]]
    if rel.duplicated(["network", "layer"]).any():
        raise ValueError("duplicate reliability rows")
    d = d.merge(rel, on=["network", "layer"], how="left", validate="one_to_one")
    if d.reliability.isna().any():
        raise ValueError("layers without a recorded reliability: %s" % d[d.reliability.isna()][["network", "layer"]].values.tolist())
    d = with_depth(d)
    d["log_routes"] = np.log10(d.n_routes.astype(float))
    d["projection"] = d.layer.str.contains("projection")
    rows = []
    sets = [("main path", d[~d.projection]), ("parted (kept < 0.8), main path", d[(~d.projection) & (d.kept_fraction < 0.8)])]
    sets += [("within %s, main path" % net, g[~g.projection]) for net, g in d.groupby("network")]
    # Reliability may itself follow from parting (shared wiring leaves a small, noisy category pattern), so partialling
    # it out can over-correct; the complementary test keeps only layers whose wiring is measured reliably.
    sets += [("reliability >= %.1f, main path" % f, d[(~d.projection) & (d.reliability >= f)]) for f in (0.5, 0.8)]
    controls = [("none", []), ("layer size", ["log_routes"]), ("reliability", ["reliability"]),
                ("depth", ["depth"]), ("size and reliability", ["log_routes", "reliability"]),
                ("size, reliability and depth", ["log_routes", "reliability", "depth"])]
    for name, sub in sets:
        for cname, cols in controls:
            if len(sub) < len(cols) + 4:
                continue
            r, p = rank_partial(sub.kept_fraction.values, sub.tree_share.values, [sub[c].values for c in cols])
            rows.append(dict(layers=name, n=len(sub), relation="tree share ~ kept fraction", controls=cname, r=r, p=p))
        for var in ("log_routes", "reliability"):
            r, p = stats.spearmanr(sub.kept_fraction, sub[var])
            rows.append(dict(layers=name, n=len(sub), relation="kept fraction ~ " + var, controls="none", r=float(r), p=float(p)))
            r, p = stats.spearmanr(sub.tree_share, sub[var])
            rows.append(dict(layers=name, n=len(sub), relation="tree share ~ " + var, controls="none", r=float(r), p=float(p)))
    out = pd.DataFrame(rows)
    with write_file(a.out) as handle:
        out.to_csv(handle, index=False)
    pd.set_option("display.width", 200)
    print(out[out.relation == "tree share ~ kept fraction"].pivot_table(index=["layers", "n"], columns="controls", values="r",
                                                                      sort=False).round(2).to_string())
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error))
