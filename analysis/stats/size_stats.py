"""Real-world size (big against small objects): the Holm adjustment of the size estimates.

The estimates come from the size analysis's export, through size_adapter.py, with their delete-one-subgroup
intervals and raw p values; they are not recomputed here, and no p value is taken from a rounded table. Three
estimands: delta_size, the separation of big from small objects (beyond animacy when conditioned on it), in standard
deviations of the pair similarity, the one the paper reports; delta3, the split between animals, big objects and
small objects; beta_tree, size beyond the category tree. Three lenses: wiring_Cphi (the correlation between two
circuits' masks), activation_Rcv (the mean category representations, cross-validated) and
wiring_minus_activation_Rcv (their difference). Holm's adjustment runs within each network and lens over the
3 estimands x 7 layers (21 tests); a test missing from the input, such as one at a layer where size is not estimated,
still counts. A row the input marks "not formed" keeps its reason and counts, with no estimate, interval or p value.

Required columns: estimand, layer, lens, estimate, ci_low, ci_high, p, n_categories, n_blocks, unit. The optional
conditioning column says what an estimate is conditioned on; conv5_beyond_animacy marks conv5's size beyond animacy.

Usage: size_stats.py --input-csv primary.csv --out OUT.csv [--network alexnet]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from arms import load_config, input_path, InputNotAvailable
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import validate_paths, check_outputs
from category_resampling import result_row, adjust_families, write_results

FAMILY = "real-world size: estimands x layers"
ESTIMANDS = ("delta3", "delta_size", "beta_tree")
LAYERS = ("conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7")
LENSES = ("wiring_Cphi", "activation_Rcv", "wiring_minus_activation_Rcv")
PRIMARY_PLAN = tuple((e, layer, lens) for e in ESTIMANDS for layer in LAYERS for lens in LENSES)


def _missing_rows(keys, reason, network="alexnet", unit="category"):
    """Empty rows, with the reason, for tests missing from the input."""
    return [result_row(estimate=None, family=network + ":" + FAMILY + ":" + lens, unit=unit, effect_unit="units of the size export", scope="D",
        n_categories=0, n_units=0, n_blocks=0, method="block_jackknife_upstream", status="not in reach",
        reason=reason, network=network, estimand=e, layer=layer, lens=lens, hypothesis_id=":".join((e, layer, lens)),
        conv5_beyond_animacy=False, conditioning_mapping="not formed: exact source conditioning absent")
        for e, layer, lens in keys]


def size_rows(frame, plan=None, network="alexnet", unit="category"):
    """The Holm-adjusted rows of each network in frame. plan is the 63 tests (estimand x layer x lens) unless a
    smaller one is given, as in tests on small inputs."""
    plan = PRIMARY_PLAN if plan is None else tuple(plan)
    if len(set(plan)) != len(plan) or not plan:
        raise ValueError("the plan needs distinct (estimand, layer, lens) tests")
    required = {"estimand", "layer", "lens", "estimate", "ci_low", "ci_high", "p", "n_categories", "n_blocks", "unit"}
    if not required.issubset(frame.columns):
        raise ValueError("size input lacks the columns: " + ", ".join(sorted(required - set(frame.columns))))
    keys = ["estimand", "layer", "lens"]
    if "network" in frame:
        result = []
        for net, part in frame.groupby("network", sort=False):
            result.extend(size_rows(part.drop(columns="network"), plan, str(net), unit))
        return result
    if frame.duplicated(keys).any():
        raise ValueError("size input repeats an (estimand, layer, lens) row")
    if frame.empty:
        raise ValueError("size input has no rows")
    rows = []
    observed = set()
    # Keep the input's other columns, but never in place of the result fields.
    analysis_fields = {"estimate", "ci_low", "ci_high", "p", "p_holm", "family",
        "unit", "scope", "n_categories", "n_units", "n_blocks", "method", "network",
        "requested_unit", "status", "hypothesis_id", "conv5_beyond_animacy", "conditioning_mapping"}
    for _, source in frame.iterrows():
        key = tuple(str(source[k]) for k in keys)
        if key not in plan:
            raise ValueError("size input row is not one of the planned tests: " + str(key))
        observed.add(key)
        unformed = isinstance(source.get("status"), str) and source.status == "not formed"
        effect_fields = ("estimate", "ci_low", "ci_high", "p")
        if unformed:
            if not all(pd.isna(source[k]) for k in effect_fields):
                raise ValueError("a row marked not formed must have no estimate, interval or p value")
            reason = source.get("reason", None)
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("a row marked not formed needs its reason")
            numeric = [source[k] for k in ("n_categories", "n_blocks")]
        else:
            numeric = [source[k] for k in effect_fields + ("n_categories", "n_blocks")]
        if not np.isfinite(np.asarray(numeric, dtype=float)).all():
            raise ValueError("size input rows need finite estimates, intervals, raw p values and counts")
        extra = {k: source[k] for k in frame.columns if k not in analysis_fields}
        if "status" in frame and "source_status" not in extra:
            extra["source_status"] = source.status
        conditioning = source.get("conditioning", None)
        has_conditioning = conditioning is not None and pd.notna(conditioning) and bool(str(conditioning).strip())
        # effect_unit comes from the input's unit column.
        extra.pop("effect_unit", None)
        rows.append(result_row(estimate=None if unformed else float(source.estimate),
            ci_low=None if unformed else float(source.ci_low), ci_high=None if unformed else float(source.ci_high),
            p=None if unformed else float(source.p), family=network + ":" + FAMILY + ":" + str(source.lens),
            unit="block50", effect_unit=str(source.unit), scope="D" if unformed else "G", n_categories=source.n_categories,
            n_units=source.n_blocks, n_blocks=source.n_blocks, method="block_jackknife_upstream", network=network,
            requested_unit=unit,
            status="not formed" if unformed else "formed from exact source rows", hypothesis_id=":".join(key),
            conv5_beyond_animacy=bool(not unformed and source.layer == "conv5" and source.estimand == "delta_size" and has_conditioning and conditioning == "animacy"),
            conditioning_mapping="explicit source conditioning" if has_conditioning else "not formed: exact source conditioning absent",
            **extra))
    rows.extend(_missing_rows([key for key in plan if key not in observed], "test missing from the size input", network, unit))
    rows = adjust_families(rows)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-csv", type=Path,
                        help="size_adapter.py's main output (default: the config's s2as_lenses input)")
    parser.add_argument("--network", default="alexnet")
    parser.add_argument("--unit", default="category", choices=("category", "block50", "superordinate", "domain"))
    parser.add_argument("--config", "--arms-config", dest="config")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    check_outputs(files=[args.out], inputs=[args.input_csv, args.config])
    config = load_config(args.config)
    try:
        source = args.input_csv if args.input_csv else input_path(config, "s2as_lenses")
        validate_paths([args.out], [source])
        rows = size_rows(pd.read_csv(source, float_precision="round_trip"), network=args.network, unit=args.unit)
    except InputNotAvailable as error:
        rows = _missing_rows(PRIMARY_PLAN, str(error), args.network, args.unit)
    write_results(rows, args.out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error))
