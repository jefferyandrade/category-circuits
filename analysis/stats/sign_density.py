"""Figs. 3C and 4A: the kept fraction of weights by sign, for each AlexNet layer and pruning method.

From the circuits' masks (--from-masks): per category, layer and method (P, its repeat extraction P-redraw, SA-H and
RP1), the kept positive and negative weights, as counts and as fractions of each sign's weights (the sign from the
checkpoint; a zero weight counts as positive, as in the engine). Summaries per layer and method:
- the means of the two fractions, with percentile category bootstrap intervals; the mean and quantiles of the
  positive-to-negative ratio over the finite ratios, with the numbers of finite, infinite and undefined ratios, and
  the share of all categories whose ratio is within 5% of one (D);
- the correlation over categories of the positive with the negative fraction, Pearson and Spearman (D), with
  percentile category bootstrap intervals (G);
- P minus SA-H, minus RP1 and minus P-redraw, per sign, on the categories both have (D, with G intervals);
- the ratio statistics described next, without the trends over depth.
With --gram-dir, the P masks read must be, in order, the masks the grams were computed from (their sha256).

From the per-category table (the default; per_category_sign_density.csv, as in data/sign_density/): the
negative-to-positive density ratio of each layer and method as a geometric mean over categories, also in pruning
steps, with its percentile category bootstrap interval and sign-flip p value, Holm-adjusted over P's seven layers;
P minus P-redraw in pruning steps; and the trends over depth of each fraction and of the ratio in pruning steps
(depth_trends.py).

Usage: sign_density.py --from-masks --checkpoint PTH --out NEW_DIR [--gram-dir DIR]
       sign_density.py --out NEW_DIR [--input-csv per_category_sign_density.csv]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import check_outputs, publish_dir, publish_file, validate_paths
from arms import load_config, arm_record, input_path, declared_cohort, validate_cohort
from category_resampling import (block_bootstrap, block_sign_flip, result_row, adjust_families,
                     write_results, log_ratio_to_steps, unit_labels)

ARMS = ("P", "P_redraw", "SAH", "RP1")        # the methods read from masks; their folders are in arms_config.json
LAYERS = ("conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7")


def sign_statistics(frame, n_resamples=10000, seed=0, unit="category", network="alexnet"):
    """The negative-to-positive density ratio of each layer and method, and the same in pruning steps.

    The estimate is exp(mean(log(neg / pos))) over categories, evaluated in that order, with its percentile category
    bootstrap interval and sign-flip p value. A zero density makes the log ratio undefined: the row is kept, with no
    estimate, and no category is dropped.
    """
    needed = {"category", "arm", "layer", "dens_pos", "dens_neg"}
    if not needed.issubset(frame.columns):
        raise ValueError("sign input missing columns: " + ", ".join(sorted(needed - set(frame.columns))))
    if "network" in frame and frame.network.nunique() > 1:
        return [row for net, d in frame.groupby("network", sort=False)
                for row in sign_statistics(d, n_resamples, seed, unit, str(net))]
    if frame.category.isna().any() or frame.duplicated(["category", "arm", "layer"]).any():
        raise ValueError("missing or duplicate category-arm-layer observations")
    if unit != "category":
        unit_labels(frame, unit)
        if frame[unit].isna().any() or frame.groupby("category")[unit].nunique(dropna=False).max() != 1:
            raise ValueError("each category requires one " + unit)
    if "network" in frame and len(frame):
        network = str(frame.network.iloc[0])
    density = frame[["dens_pos", "dens_neg"]].to_numpy(float)
    if not np.isfinite(density).all() or np.any(density < 0) or np.any(density > 1):
        raise ValueError("sign densities must be finite fractions in [0,1]")
    ratios, steps, anchors = [], [], []
    for (arm, layer), d in frame.groupby(["arm", "layer"], sort=False):
        labels = unit_labels(d, unit)
        family = network + ": sign density: P x layers" if arm == "P" else "descriptive"
        meta = dict(arm=arm, layer=layer, network=network, n_categories=len(d), n_units=len(np.unique(labels)),
                    n_blocks=d.block50.nunique() if "block50" in d else len(d), unit=unit,
                    partial=bool(d.partial.any()) if "partial" in d else False,
                    family=family, scope="G", hypothesis_id=str(arm) + ":" + str(layer))
        invalid = (d.dens_pos <= 0) | (d.dens_neg <= 0)
        if invalid.any():
            r = result_row(estimate=None, method="log ratio undefined for zero densities", effect_unit="negative/positive density ratio",
                status="not formed", n_zero_density=int(invalid.sum()), **{**meta, "scope": "D"})
            ratios.append(r)
            steps.append({**r, "effect_unit": "pruning steps"})
            continue
        log_ratio = np.log(d.dens_neg.values / d.dens_pos.values)
        ci = block_bootstrap(log_ratio, labels, n_resamples=n_resamples, seed=seed, unit=unit)
        p = block_sign_flip(log_ratio, labels, n_resamples=n_resamples, seed=seed, unit=unit)
        method = ci["method"] + "; sign flip over " + unit + " units of category log ratios"
        common = dict(p=p, method=method, mean_log_ratio_neg_over_pos=float(log_ratio.mean()),
            mean_pos=float(d.dens_pos.mean()), mean_neg=float(d.dens_neg.mean()),
            share_neg_higher=float(np.mean(log_ratio > 0)), share_equal=float(np.mean(log_ratio == 0)), **meta)
        ratios.append(result_row(**{k: float(np.exp(ci[k])) for k in ("estimate", "ci_low", "ci_high")},
            effect_unit="negative/positive density ratio", **common))
        steps.append(result_row(**{k: float(log_ratio_to_steps(ci[k])) for k in ("estimate", "ci_low", "ci_high")},
            effect_unit="pruning steps", **common))
    # P's family of tests is its seven layers: a layer missing from the input keeps an empty row. The other
    # methods' rows are descriptive.
    present = set(frame.loc[frame.arm == "P", "layer"])
    for layer in LAYERS:
        if layer in present:
            continue
        missing = result_row(estimate=None, family=network + ": sign density: P x layers",
            unit=unit, effect_unit="negative/positive density ratio", scope="G", n_categories=0,
            n_units=0, n_blocks=0, network=network, arm="P", layer=layer,
            hypothesis_id="P:" + layer, partial=bool(frame.partial.any()) if "partial" in frame else False,
            status="not formed", method="layer absent from the input")
        ratios.append(missing)
        steps.append({**missing, "effect_unit": "pruning steps"})
    # Ratio and steps express the same test and count only once in Holm.
    ratios = adjust_families(ratios)
    for r, s in zip(ratios, steps):
        s["p_holm"] = r["p_holm"]
    p_frame = frame[frame.arm == "P"].set_index(["category", "layer"])
    redraw = frame[frame.arm.isin(["P_redraw", "P-redraw"])].set_index(["category", "layer"])
    joined = p_frame.join(redraw, lsuffix="_P", rsuffix="_R", how="inner")
    if not joined.empty:
        for layer, d in joined.groupby(level="layer", sort=False):
            categories = d.index.get_level_values("category").to_numpy()
            labels = categories if unit == "category" else d[unit + "_P"].to_numpy()
            anchor_meta = dict(family="descriptive", unit=unit, effect_unit="pruning steps", network=network,
                scope="G", n_categories=len(d), n_units=len(np.unique(labels)), layer=layer,
                n_blocks=d.block50_P.nunique() if "block50_P" in d else len(d), contrast="P minus P-redraw",
                partial=bool(d.partial_P.any() or d.partial_R.any()) if "partial_P" in d else False)
            values = d[["dens_pos_P", "dens_neg_P", "dens_pos_R", "dens_neg_R"]].to_numpy(float)
            if np.any(values <= 0):
                anchors.append(result_row(estimate=None, **anchor_meta,
                    method="log ratio undefined for zero densities", status="not formed"))
                continue
            diff = log_ratio_to_steps(np.log(values[:, 1] / values[:, 0]) - np.log(values[:, 3] / values[:, 2]))
            ci = {k: v for k, v in block_bootstrap(diff, labels, n_resamples=n_resamples, seed=seed, unit=unit).items()
                  if k in ("estimate", "ci_low", "ci_high")}
            anchors.append(result_row(**ci, **anchor_meta,
                mean_absolute_step_difference=float(np.mean(np.abs(diff))),
                method="paired shared-category retest; percentile bootstrap over " + unit + " units"))
    return ratios + steps + adjust_families(anchors)


def run_from_table(a):
    """The statistics from the per-category table (--input-csv, or the config's sign_per_category input); needs no
    masks or checkpoint."""
    import pandas as pd
    from depth_trends import depth_rows
    check_outputs(dirs=[a.out], inputs=[getattr(a, "input_csv", None), getattr(a, "config", None)])
    config = load_config(getattr(a, "config", None))
    source = Path(a.input_csv) if getattr(a, "input_csv", None) else input_path(config, "sign_per_category")
    out = Path(a.out)
    validate_paths([out], [source])
    frame = pd.read_csv(source)
    if "partial" in frame and frame.partial.any():
        frame["partial"] = True
    unit = getattr(a, "unit", "category")
    network = getattr(a, "network", "alexnet")
    if "network" not in frame:
        frame["network"] = network
    rows = sign_statistics(frame, n_resamples=a.n_resamples, seed=a.seed, unit=unit, network=network)
    trends = []
    for (net, arm), d in frame.groupby(["network", "arm"], sort=False):
        d = d.copy()
        metrics = ["dens_pos", "dens_neg"]
        if (d[["dens_pos", "dens_neg"]] > 0).all().all():
            d["sign_steps"] = log_ratio_to_steps(np.log(d.dens_neg.values / d.dens_pos.values))
            metrics.append("sign_steps")
        for metric in metrics:
            layers = [l for l in LAYERS if l in set(d.layer)]
            if len(layers) > 1:
                contrasts = [("fc7", "fc6")] if {"fc6", "fc7"}.issubset(layers) else []
                trends.extend(depth_rows(d, metric, layers, "depth trends",
                    contrasts=contrasts, n_resamples=a.n_resamples, seed=a.seed, arm=arm,
                    unit=unit, network=str(net)))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=out.parent))
    try:
        write_results(rows, tmp / "sign_density_stats.csv")
        write_results(adjust_families(trends), tmp / "depth_trends.csv")
        publish_dir(tmp, out)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    return 0


def boot_corr(x, y, blocks=None, n=10000, seed=0, unit="category"):
    """Pearson and Spearman correlations of x and y over categories, each with a percentile bootstrap interval."""
    from scipy.stats import spearmanr
    idx = np.arange(len(x))
    pe = block_bootstrap(idx, blocks,
        statistic=lambda i: np.corrcoef(x[i.astype(int)], y[i.astype(int)])[0, 1], n_resamples=n, seed=seed, unit=unit)
    sp = block_bootstrap(idx, blocks,
        statistic=lambda i: spearmanr(x[i.astype(int)], y[i.astype(int)]).correlation, n_resamples=n, seed=seed, unit=unit)
    return tuple(float(d[k]) for d in (pe, sp) for k in ("estimate", "ci_low", "ci_high"))


def _bootstrap_mean(values, blocks=None, n_resamples=10000, unit="category"):
    """The mean and its percentile bootstrap interval, as (mean, low, high, number of categories)."""
    d = block_bootstrap(values, blocks, n_resamples=n_resamples, unit=unit)
    return d["estimate"], d["ci_low"], d["ci_high"], len(values)


def ratio_summary(d):
    """The positive-to-negative ratio over the finite ratios, and the share of all categories within 5% of one."""
    ratios = d.ratio_pos_over_neg.to_numpy()
    finite = np.isfinite(ratios)
    vals = ratios[finite]
    zero_neg = d.dens_neg.to_numpy() == 0
    positive_pos = d.dens_pos.to_numpy() > 0
    r = {"n_ratio_inf": int(np.count_nonzero(zero_neg & positive_pos)),
         "n_ratio_undefined": int(np.count_nonzero(zero_neg & (d.dens_pos.to_numpy() == 0))),
         "n_ratio_finite": int(finite.sum()), "ratio_summary_population": "finite ratios",
         "ratio_mean": float(vals.mean()) if vals.size else None,
         "share_ratio_within_5pct_of_1": float(np.count_nonzero(finite & (np.abs(ratios - 1) <= .05)) / len(d))
         if len(d) else None}
    r.update({f"ratio_q{int(q*100):02d}": float(np.quantile(vals, q)) if vals.size else None
              for q in (.05, .25, .5, .75, .95)})
    return r


def verify_gram_masks(gram_dir, shas):
    """Check that the P masks read are, category by category, those each gram records (mask_sha256)."""
    grams = sorted(Path(gram_dir).glob("gram_*.npz"))
    if not grams:
        raise ValueError(f"no gram_*.npz files in {gram_dir}")
    checks = {}
    for gram in grams:
        with np.load(gram) as z:
            cats, hashes = z["categories"], z["mask_sha256"]
            if cats.ndim != 1 or hashes.ndim != 1 or len(cats) != len(hashes):
                raise ValueError(f"{gram}: require exactly one P mask hash per category")
            if len(set(map(int, cats))) != len(cats) or len(shas) != len(cats) or set(shas) != set(map(int, cats)):
                raise ValueError(f"{gram}: loaded P categories differ from the complete Gram category array")
            got = [shas.get(int(c)) for c in cats]
            want = [v.decode() if isinstance(v, bytes) else str(v) for v in hashes]
            if any(v is None for v in got) or got != want:
                raise ValueError(f"{gram}: P masks differ from the complete ordered Gram hashes")
            checks[gram.name] = {"n": len(got), "mismatch": 0}
    return checks


def run_from_masks(a) -> int:
    """The per-category kept fractions by sign from the circuits' masks, and their summaries."""
    import pandas as pd
    import circuit_masks as RT
    out = Path(a.out)
    initial_inputs = [a.root, a.taxonomy]
    if a.gram_dir:
        initial_inputs.append(a.gram_dir)
    if a.checkpoint:
        initial_inputs.append(a.checkpoint)
    validate_paths([out], initial_inputs)
    check_outputs(dirs=[out], inputs=initial_inputs)
    layout = RT.tree_layout("alexnet")
    ck = Path(a.checkpoint) if a.checkpoint else input_path(load_config(getattr(a, "config", None)), "alexnet_checkpoint")
    neg, ck_sha = RT.weight_signs("alexnet", layout, ck)       # checks the checkpoint identity
    unit = getattr(a, "unit", "category")
    blocks_of = RT.load_blocks(Path(a.taxonomy))
    if unit in ("category", "block50"):
        labels_of = {c: c if unit == "category" else b for c, b in blocks_of.items()}
    else:
        taxonomy = pd.read_csv(a.taxonomy)
        labels_of = dict(zip(taxonomy.category, unit_labels(taxonomy, unit)))
    rows, shas = [], {}
    config = load_config(getattr(a, "config", None))
    records = {arm: arm_record(config, "alexnet", arm, price=getattr(a, "price", "v66")) for arm in ARMS}
    configured_arms = {arm: record["arm_folder"] for arm, record in records.items()}
    cohorts = {arm: declared_cohort(config, "alexnet", arm, record=record) for arm, record in records.items()}
    roots = {arm: RT.results_dir(record["results_root"]) for arm, record in records.items()}
    validate_paths([out], [*initial_inputs, ck, *roots.values()])
    found = {arm: [] for arm in records}
    mask_hashes = {arm: {} for arm in records}
    for c in sorted(set().union(*map(set, cohorts.values()))):
        for arm, record in records.items():
            if c not in cohorts[arm]:
                continue
            p = roots[arm] / record["category_pattern"].format(category=c) / record["arm_folder"]
            if not ((p / "final_mask.packbits").is_file() and (p / "receipt.json").is_file()):  # finished circuits only
                continue
            rec = json.loads((p / "receipt.json").read_text())
            bits = RT.load_verified_mask(p / "final_mask.packbits", rec, layout)
            found[arm].append(c)
            mask_hashes[arm][c] = hashlib.sha256((p / "final_mask.packbits").read_bytes()).hexdigest()
            if arm == "P":
                shas[c] = mask_hashes[arm][c]
            for s in layout.layers:
                b = bits[s.offset:s.offset + s.size]
                n = neg[s.layer_id]
                kp, kn = int((b & ~n).sum()), int((b & n).sum())
                ap, an = int((~n).sum()), int(n.sum())
                rows.append({"category": c, "arm": arm, "layer": RT.short_name("alexnet", s.layer_id),
                             "kept_pos": kp, "kept_neg": kn, "avail_pos": ap, "avail_neg": an,
                             "dens_pos": kp / ap, "dens_neg": kn / an})
        if c % 100 == 0:
            print(f"  {c} categories", flush=True)
    df = pd.DataFrame(rows)
    cohort_meta = {arm: validate_cohort(found[arm], cohorts[arm], getattr(a, "allow_partial", False)) for arm in records}
    shared = set(mask_hashes["P"]) & set(mask_hashes["P_redraw"])
    if shared and all(mask_hashes["P"][c] == mask_hashes["P_redraw"][c] for c in shared):
        raise ValueError("P and P-redraw masks are byte-identical on every shared category")
    if df.empty:
        raise ValueError("no category masks found in declared arms")
    for key in ("partial", "n_declared_categories", "n_missing_categories"):
        df[key] = df.arm.map({arm: meta[key] for arm, meta in cohort_meta.items()})
    df["arm_partial"] = df.partial
    df["partial"] = any(meta["partial"] for meta in cohort_meta.values())
    df["ratio_pos_over_neg"] = df.dens_pos / df.dens_neg
    df["block50"] = df.category.map(blocks_of)
    summ = []
    for (arm, L), d in df.groupby(["arm", "layer"], sort=False):
        b = [labels_of[int(c)] for c in d.category]
        mp, lp, hp, _ = _bootstrap_mean(d.dens_pos.values, b, getattr(a, "n_resamples", 10000), unit=unit)
        mn, ln, hn, _ = _bootstrap_mean(d.dens_neg.values, b, getattr(a, "n_resamples", 10000), unit=unit)
        r = {"arm": arm, "layer": L, "n_categories": len(d), "unit": unit,
             "n_units": len(set(b)), "scope": "G", "method": "percentile bootstrap over " + unit + " units",
             **cohort_meta[arm], "partial": bool(df.partial.any()), "dens_pos": mp, "dens_pos_lo_G": lp, "dens_pos_hi_G": hp,
             "dens_neg": mn, "dens_neg_lo_G": ln, "dens_neg_hi_G": hn,
             **ratio_summary(d),
             "kept_pos_mean": float(d.kept_pos.mean()), "kept_neg_mean": float(d.kept_neg.mean())}
        if d.dens_pos.std() > 0 and d.dens_neg.std() > 0:
            try:
                pe, pl, ph, sp, sl, sh = boot_corr(d.dens_pos.values, d.dens_neg.values, b, getattr(a, "n_resamples", 10000), unit=unit)
                r.update({"pearson": pe, "pearson_lo_G": pl, "pearson_hi_G": ph, "spearman": sp, "spearman_lo_G": sl,
                          "spearman_hi_G": sh})
            except ValueError as exc:
                # A nearly fully kept layer (conv1: most categories keep every weight of both signs) makes the
                # correlation undefined in some category resamples; the full-sample value is still defined. Report it
                # and leave the interval empty with the reason, rather than dropping undefined resamples (category_resampling
                # refuses to) or failing the whole table.
                if "not finite" not in str(exc):
                    raise
                from scipy.stats import spearmanr
                r.update({"pearson": float(np.corrcoef(d.dens_pos.values, d.dens_neg.values)[0, 1]),
                          "pearson_lo_G": float("nan"), "pearson_hi_G": float("nan"),
                          "spearman": float(spearmanr(d.dens_pos.values, d.dens_neg.values).correlation),
                          "spearman_lo_G": float("nan"), "spearman_hi_G": float("nan"),
                          "correlation_interval_status": "not formed: correlation undefined in some " + unit + " resamples"})
        summ.append(r)
    paired = []
    P = df[df.arm == "P"].set_index(["category", "layer"])
    for other in ("SAH", "RP1", "P_redraw"):
        O = df[df.arm == other].set_index(["category", "layer"])
        j = P.join(O, lsuffix="_P", rsuffix="_O", how="inner")
        for L, d in j.groupby(level="layer", sort=False):
            b = [labels_of[int(c)] for c in d.index.get_level_values("category")]
            for sgn in ("pos", "neg"):
                diff = (d[f"dens_{sgn}_P"] - d[f"dens_{sgn}_O"]).values
                m, lo, hi, n = _bootstrap_mean(diff, b, getattr(a, "n_resamples", 10000), unit=unit)
                paired.append({"comparison": f"P_minus_{other}", "layer": L, "sign": sgn, "n_categories": n,
                               "unit": unit, "n_units": len(set(b)), "scope": "G",
                               "partial": bool(df.partial.any()),
                               "mean_diff": m, "lo_G": lo, "hi_G": hi,
                               "share_P_higher": float((diff > 0).mean()), "median_diff": float(np.median(diff))})
    gram_check = {}
    if a.gram_dir:
        gram_check = verify_gram_masks(a.gram_dir, shas)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=out.parent))
    try:
        publish_file(df, tmp / "per_category_sign_density.csv")
        publish_file(pd.DataFrame(summ), tmp / "sign_density_by_layer.csv")
        publish_file(pd.DataFrame(paired), tmp / "sign_density_paired.csv")
        if unit != "category":
            df[unit] = df.category.map(labels_of)
        write_results(sign_statistics(df, n_resamples=getattr(a, "n_resamples", 10000), seed=getattr(a, "seed", 0), unit=unit),
                      tmp / "sign_density_stats.csv")
        publish_file(json.dumps({"arms": configured_arms, "checkpoint_sha256": ck_sha, "gram_mask_check": gram_check,
                                 "script": str(Path(__file__).resolve())}, indent=1).encode(),
                     tmp / "meta.json")
        publish_dir(tmp, out)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    print(json.dumps(gram_check))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="the P circuits' results folder (default: from arms_config.json); "
                                   "implies --from-masks")
    ap.add_argument("--taxonomy", help="the category tree table (default: from arms_config.json)")
    ap.add_argument("--out", required=True, help="the new output folder")
    ap.add_argument("--gram-dir", help="check the P masks against the hashes in this folder's gram_<layer>.npz")
    ap.add_argument("--checkpoint", help="the AlexNet checkpoint the circuits were extracted from")
    ap.add_argument("--config", "--arms-config", dest="config",
                    help="where the circuits are (default: arms_config.json)")
    ap.add_argument("--price", default="v66", help="which circuits to read (default: v66, the paper's)")
    ap.add_argument("--unit", choices=("category", "block50", "superordinate", "domain"), default="category",
                    help="the resampling unit (default: category, as in the paper)")
    ap.add_argument("--network", default="alexnet")
    ap.add_argument("--allow-partial", action=argparse.BooleanOptionalAction, default=True,
                    help="accept a method whose cohort is incomplete and mark the rows as partial (the default)")
    ap.add_argument("--input-csv", help="the per-category table (default: the config's sign_per_category input)")
    ap.add_argument("--from-masks", action="store_true",
                    help="read the circuits' masks instead of the per-category table")
    ap.add_argument("--n-resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    check_outputs(dirs=[a.out], inputs=[a.config, a.input_csv, a.checkpoint, a.gram_dir, a.root, a.taxonomy])
    if a.from_masks or a.root:
        config = load_config(a.config)
        if not a.root:
            a.root = str(arm_record(config, "alexnet", "P", price=a.price)["results_root"])
        if not a.taxonomy:
            a.taxonomy = str(input_path(config, "taxonomy"))
        return run_from_masks(a)
    return run_from_table(a)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error))
