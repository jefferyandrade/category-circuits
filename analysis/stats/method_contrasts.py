"""Section 2.1 and Fig. 3A: the pruning methods ("arms") compared on AlexNet's 1,000 categories.

Each measure is read from the circuits' receipts through arms_config.json's receipt_fields: the kept fraction of
weights (native_density), the corrected-accuracy drop in points, and the increase of the held-out loss. For each
measure, each method's paired difference from P (SA-H - P, SA-G - P, SG - P, SF - P) and the interaction
(SA-G - SG) - (SA-H - P), which asks whether separating the signs matters differently for global and layer-by-layer
pruning, are given as the mean over categories with a percentile category bootstrap interval (10,000 resamples,
seed 0) and a sign-flip p value, Holm-adjusted over the network's measures and contrasts. A measure the receipts do
not record gets a row with the reason and no estimate.

Usage: python method_contrasts.py --out OUT.csv [--network alexnet] [--price v66]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from arms import InputNotAvailable, add_config_arguments, load_config, load_receipts
from category_resampling import adjust_families, block_bootstrap, block_sign_flip, unit_labels, write_results

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import check_outputs

METRICS = {"native_density": "fraction", "kernel_unit_density": "fraction",
           "corrected_accuracy_drop": "corrected-accuracy points",
           "heldout_delta_bce": "held-out delta BCE", "loss_window_share": "fraction"}
ARMS = ("P", "SA-H", "SA-G", "SG", "SF")
FAMILY = "method_comparison:arms_x_metrics"


def unavailable_row(metric, contrast, reason, family=FAMILY, network="alexnet", unit="category"):
    """The row of a contrast whose inputs are not available, with the reason."""
    return dict(estimate=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan, p_holm=np.nan,
                family=family if family == "descriptive" else network + ":" + family,
                unit=unit, effect_unit=METRICS.get(metric, metric), scope="G", n_categories=0,
                n_units=0, n_blocks=0, method="percentile %s bootstrap; %s sign-flip" % (unit, unit),
                metric=metric, contrast=contrast, network=network,
                status="not in reach", reason=str(reason))


def contrast_row(frame, values, metric, contrast, n_resamples=10000, seed=0,
                 family=FAMILY, network="alexnet", unit="category"):
    """The mean of per-category differences, its percentile bootstrap interval and its sign-flip p value."""
    values = np.asarray(values, dtype=float)
    blocks = unit_labels(frame, unit)
    if len(frame) < 2 or len(pd.unique(blocks)) < 2 or not np.isfinite(values).all():
        raise ValueError("a contrast requires finite paired categories in at least two units")
    interval = block_bootstrap(values, blocks, n_resamples=n_resamples, seed=seed, unit=unit)
    return dict(estimate=interval["estimate"], ci_low=interval["ci_low"], ci_high=interval["ci_high"],
                p=block_sign_flip(values, blocks, n_resamples=n_resamples, seed=seed, unit=unit),
                p_holm=np.nan, family=family if family == "descriptive" else network + ":" + family,
                unit=unit, effect_unit=METRICS.get(metric, metric), scope="G",
                n_categories=len(frame), n_units=len(pd.unique(blocks)),
                n_blocks=frame.block50.nunique() if "block50" in frame else len(frame),
                partial=bool(frame.get("partial", pd.Series([False])).any()),
                method="percentile %s bootstrap; %s sign-flip" % (unit, unit),
                metric=metric, contrast=contrast, network=network, status="done", reason="")


def paired_frames(frames, arms, metric, unit="category"):
    """Each method's rows on the categories all of them have, in category order; their taxonomy blocks must agree."""
    common = set(frames[arms[0]].category)
    for arm in arms:
        frame = frames[arm]
        if frame.category.duplicated().any() or frame[["category", "block50", metric]].isna().any().any():
            raise ValueError("duplicate category or undefined paired input in %s" % arm)
        common &= set(frame.category)
    if not common:
        raise InputNotAvailable("arms have no common categories")
    categories = sorted(common)
    paired = {arm: frames[arm].set_index("category").loc[categories].reset_index() for arm in arms}
    blocks = paired[arms[0]].block50.to_numpy()
    for arm in arms[1:]:
        if not np.array_equal(blocks, paired[arm].block50.to_numpy()):
            raise ValueError("taxonomy blocks disagree between arms")
    partial = any(bool(frames[arm].get("partial", pd.Series([False])).any()) for arm in arms)
    for frame in paired.values():
        frame["partial"] = partial
        if unit != "category":
            expected = unit_labels(paired[arms[0]], unit)
            if not np.array_equal(expected, unit_labels(frame, unit)):
                raise ValueError("taxonomy units disagree between arms")
    return paired


def analyze(config, network="alexnet", price="v66", run="primary", n_resamples=10000, seed=0,
            unit="category", allow_partial=False):
    """Every contrast row of one network, Holm-adjusted."""
    rows = []
    any_partial = False
    for metric in METRICS:
        frames, failures = {}, {}
        for arm in ARMS:
            try:
                frames[arm] = load_receipts(config, network, arm, price, run, metrics=[metric], allow_partial=allow_partial)
                any_partial = any_partial or bool(frames[arm].partial.any())
            except InputNotAvailable as exc:
                failures[arm] = str(exc)
        for arm in ARMS[1:]:
            contrast = arm + " - P"
            try:
                if arm in failures or "P" in failures:
                    raise InputNotAvailable(failures.get(arm, failures.get("P")))
                paired = paired_frames(frames, ("P", arm), metric, unit=unit)
                difference = paired[arm][metric].to_numpy() - paired["P"][metric].to_numpy()
                rows.append(contrast_row(paired["P"], difference, metric, contrast,
                                         n_resamples, seed, network=network, unit=unit))
            except InputNotAvailable as exc:
                rows.append(unavailable_row(metric, contrast, exc, network=network, unit=unit))
        contrast = "(SA-G - SG) - (SA-H - P)"
        try:
            needed = ("P", "SA-H", "SA-G", "SG")
            if any(arm in failures for arm in needed):
                raise InputNotAvailable("; ".join(failures[arm] for arm in needed if arm in failures))
            paired = paired_frames(frames, needed, metric, unit=unit)
            difference = ((paired["SA-G"][metric] - paired["SG"][metric]) -
                          (paired["SA-H"][metric] - paired["P"][metric])).to_numpy()
            rows.append(contrast_row(paired["P"], difference, metric, contrast,
                                     n_resamples, seed, network=network, unit=unit))
        except InputNotAvailable as exc:
            rows.append(unavailable_row(metric, contrast, exc, network=network, unit=unit))
    for row in rows:
        row["partial"] = any_partial
    return adjust_families(rows)


def main(argv=None):
    parser = add_config_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--network", default="alexnet")
    parser.add_argument("--out", required=True, type=Path, help="the result table (CSV) to write; it must not exist")
    args = parser.parse_args(argv)
    check_outputs(files=[args.out], inputs=[args.arms_config])
    rows = analyze(load_config(args.arms_config), args.network, args.price, args.run,
                   unit=args.unit, allow_partial=args.allow_partial)
    write_results(rows, args.out)
    print("%s: %d of %d rows computed" % (args.out, sum(r["status"] == "done" for r in rows), len(rows)))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error))
