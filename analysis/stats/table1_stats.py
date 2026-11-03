"""Table 1: each network's kept fraction of weights and corrected-accuracy drop on the shared categories.

For the P circuits of AlexNet, VGG-11, ResNet-18 and the two ResNet-50 versions, on the shared categories: the mean
over categories with a percentile category bootstrap interval (10,000 resamples, seed 0), and the paired difference
between ResNet-50 V1 and V2 with its sign-flip p value. The measures are read from the circuits' receipts through
arms_config.json. A network whose circuits are not available gets a row with the reason and no estimate.

Usage: python table1_stats.py --out OUT.csv [--price v66]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from arms import InputNotAvailable, add_config_arguments, load_config, load_receipts, shared_categories
from method_contrasts import METRICS, contrast_row, paired_frames, unavailable_row
from category_resampling import adjust_families, block_bootstrap, unit_labels, write_results

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import check_outputs

NETWORKS = ("alexnet", "vgg11", "resnet18", "resnet50v1", "resnet50v2")
FAMILY = "table1:resnet50v1_minus_resnet50v2:density_accuracy"
TABLE_METRICS = ("native_density", "corrected_accuracy_drop")


def analyze(config, price="v66", run="primary", n_resamples=10000, seed=0, unit="category", allow_partial=False):
    """Every row of the table, Holm-adjusted."""
    rows, frames = [], {}
    try:
        categories = shared_categories(config)
        cohort_error = None
    except InputNotAvailable as exc:
        categories, cohort_error = None, exc
    for metric in TABLE_METRICS:
        for network in NETWORKS:
            try:
                if cohort_error:
                    raise cohort_error
                frame = load_receipts(config, network, "P", price, run,
                                      metrics=[metric], categories=categories, allow_partial=allow_partial)
                labels = unit_labels(frame, unit)
                if len(set(labels)) < 2:
                    raise ValueError("Table 1 requires at least two resampling units")
                frames[(network, metric)] = frame
                interval = block_bootstrap(frame[metric].to_numpy(), labels,
                                           n_resamples=n_resamples, seed=seed, unit=unit)
                # A network mean is reported with its interval; it is not tested against zero.
                rows.append(dict(estimate=interval["estimate"], ci_low=interval["ci_low"],
                                 ci_high=interval["ci_high"], p=np.nan, p_holm=np.nan,
                                 family="descriptive", unit=unit, effect_unit=METRICS[metric], scope="G",
                                 n_categories=len(frame), n_units=len(set(labels)), n_blocks=frame.block50.nunique(),
                                 partial=bool(frame.get("partial", pd.Series([False])).any()),
                                 method="percentile %s bootstrap" % unit, metric=metric,
                                 contrast="network mean", network=network, status="done", reason=""))
            except InputNotAvailable as exc:
                rows.append(unavailable_row(metric, "network mean", exc, "descriptive", network, unit=unit))
        network = "resnet50v1 - resnet50v2"
        try:
            names = ("resnet50v1", "resnet50v2")
            if not all((name, metric) in frames for name in names):
                raise InputNotAvailable("both ResNet-50 versions require the complete declared shared cohort")
            paired = paired_frames({name: frames[(name, metric)] for name in names}, names, metric, unit=unit)
            values = paired[names[0]][metric].to_numpy() - paired[names[1]][metric].to_numpy()
            rows.append(contrast_row(paired[names[0]], values, metric, network,
                                     n_resamples, seed, FAMILY, network, unit=unit))
        except InputNotAvailable as exc:
            rows.append(unavailable_row(metric, network, exc, FAMILY, network, unit=unit))
    any_partial = any(bool(frame.get("partial", pd.Series([False])).any()) for frame in frames.values())
    for row in rows:
        row["partial"] = any_partial
    return adjust_families(rows)


def main(argv=None):
    parser = add_config_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--out", type=Path, required=True, help="the result table (CSV) to write; it must not exist")
    args = parser.parse_args(argv)
    check_outputs(files=[args.out], inputs=[args.arms_config])
    rows = analyze(load_config(args.arms_config), args.price, args.run, unit=args.unit, allow_partial=args.allow_partial)
    write_results(rows, args.out)
    print("%s: %d of %d rows computed" % (args.out, sum(r["status"] == "done" for r in rows), len(rows)))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error))
