"""Real-world size: turn the size analysis's per-layer exports (lenses_<network>_<layer>/S2AS_LENSES.csv) into the
input of size_stats.py.

The rows that size_stats.py adjusts (three lenses, three estimands, seven layers; variant "primary", frame
"as delivered") go to the main output, with size_stats.py's lens names; every other row keeps its names and fields
and goes to a descriptive table. Estimates and intervals are taken as they are, not refitted, in units of the
prepared pair similarity (standard deviations of its off-diagonal values). Each row records how its one-sided p value
was made two-sided; a row the export marks "not formed" may hold a placeholder p, which is converted (p_two_sided)
but not used (p is empty). A conditioning value is kept only where the export's field is not blank. size_stats.py
adds the tests missing from the export.

Usage: size_adapter.py --lenses DIR --network alexnet --out primary.csv [--descriptive-out descriptive.csv]
The descriptive table defaults to <out stem>_descriptive.csv. Both are written together, and neither replaces an
existing file. An unknown lens or estimand, or a malformed row, raises ValueError before anything is written.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import publish_files, validate_paths, check_outputs
from size_stats import ESTIMANDS, LAYERS

LENS_MAP = {
    "C-phi-prepared": "wiring_Cphi",
    "R-cv": "activation_Rcv",
    "C-phi-prepared minus R-cv": "wiring_minus_activation_Rcv",
}
# size_stats.py uses the export's estimand names.
ESTIMAND_MAP = {name: name for name in ESTIMANDS}
EXTRA_LENSES = ("R-rho", "R-xn", "C-phi-prepared minus R-rho",
                "C-phi-prepared minus R-xn")
EXTRA_ESTIMANDS = ("beta_tree_cont", "delta_domain", "domain_organization",
    "organization_animal", "organization_big", "organization_small",
    "pair_AA", "pair_AB", "pair_AS", "pair_BB", "pair_BS", "pair_SS")
P_CONVERSION = "min(1, 2 * min(p_one_sided, 1 - p_one_sided))"
EFFECT_UNIT = "prepared pair similarity (off-diagonal SD units)"
SOURCE_COLUMNS = ("network", "lens", "frame", "layer", "estimand", "variant",
    "value", "lo95", "hi95", "se", "df", "absence_value", "n_categories",
    "n_pairs", "n_blocks", "n_identifying_pairs", "n_identifying_nodes",
    "n_excluded", "p_one_sided", "status", "reason")
PROVENANCE_COLUMNS = ("source_csv", "source_row", "source_lens", "source_estimand",
    "source_status", "source_frame", "source_variant", "p", "p_two_sided",
    "p_conversion", "p_conversion_status")
PRIMARY_COLUMNS = SOURCE_COLUMNS + PROVENANCE_COLUMNS + (
    "estimate", "ci_low", "ci_high", "unit")
DESCRIPTIVE_COLUMNS = SOURCE_COLUMNS + PROVENANCE_COLUMNS + (
    "family", "p_holm", "descriptive_reason")


def _absent(value):
    return pd.isna(value) or (isinstance(value, str) and not value.strip())


def _number(value, name, optional=False):
    if _absent(value):
        if optional:
            return float("nan")
        raise ValueError("missing " + name)
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid numeric " + name) from error
    if not math.isfinite(number):
        raise ValueError("nonfinite " + name)
    return number


def _columns(records, declared):
    columns = list(dict.fromkeys(declared))
    columns.extend(c for row in records for c in row if c not in columns)
    return pd.DataFrame(records).reindex(columns=list(dict.fromkeys(columns)))


def adapt_frames(frames, network="alexnet"):
    """The main and descriptive tables from (source_csv, table) pairs; source_row is the row's line in its CSV (>= 2).

    The folder of each source_csv names its network and layer. Missing tests and conditioning labels are not filled in.
    """
    primary, descriptive, observed = [], [], set()
    for source_csv, frame in frames:
        missing = set(SOURCE_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError("size export lacks the columns: " + ", ".join(sorted(missing)))
        if frame.columns.duplicated().any():
            raise ValueError("size export repeats a column")
        for line, source in enumerate(frame.to_dict("records"), start=2):
            context = f"{source_csv}:{line}"
            try:
                if source["network"] != network:
                    raise ValueError("the row's network is not the one given")
                layer, lens, estimand = (source[k] for k in ("layer", "lens", "estimand"))
                if Path(source_csv).parent.name != f"lenses_{network}_{layer}":
                    raise ValueError("directory layer/network differs from source row")
                if lens not in (*LENS_MAP, *EXTRA_LENSES):
                    raise ValueError("unmapped lens: " + str(lens))
                if estimand not in (*ESTIMANDS, *EXTRA_ESTIMANDS):
                    raise ValueError("unmapped estimand: " + str(estimand))
                for name in ("layer", "frame", "variant"):
                    if _absent(source[name]):
                        raise ValueError("missing " + name)
                identity = tuple(source[k] for k in ("network", "lens", "frame", "layer", "estimand", "variant"))
                if identity in observed:
                    raise ValueError("size export repeats a row")
                observed.add(identity)
                status = source["status"]
                if status not in ("formed", "not formed"):
                    raise ValueError("unsupported source status: " + str(status))
                unformed = status == "not formed"
                is_primary = (lens in LENS_MAP and estimand in ESTIMANDS
                    and source["variant"] == "primary" and source["frame"] == "as delivered"
                    and layer in LAYERS)
                effects = [_number(source[k], k, optional=unformed)
                           for k in ("value", "lo95", "hi95")]
                if unformed:
                    if not all(math.isnan(v) for v in effects):
                        raise ValueError("a row marked not formed must have no estimate or interval")
                    if _absent(source["reason"]):
                        raise ValueError("a row marked not formed needs its reason")
                elif effects[1] > effects[2]:
                    raise ValueError("interval limits are reversed")
                counts = {k: _number(source[k], k) for k in ("n_categories", "n_blocks")}
                # Pair means can count more subgroups than categories: they keep the subgroups of the node
                # adjustment fitted on all categories. Only the main rows must not.
                if any(v < 0 or not v.is_integer() for v in counts.values()) or (is_primary and counts["n_blocks"] > counts["n_categories"]):
                    raise ValueError("invalid category or block counts")
                q = _number(source["p_one_sided"], "p_one_sided", optional=True)
                if not math.isnan(q) and not 0 <= q <= 1:
                    raise ValueError("p_one_sided must lie in [0,1]")
                if is_primary and not unformed and math.isnan(q):
                    raise ValueError("formed primary row requires p_one_sided")
                two_sided = float("nan") if math.isnan(q) else min(1., 2 * min(q, 1 - q))
                row = dict(source)
                row.update(source_csv=str(source_csv), source_row=line,
                    source_lens=lens, source_estimand=estimand, source_status=status,
                    source_frame=source["frame"], source_variant=source["variant"],
                    p=float("nan") if unformed else two_sided, p_two_sided=two_sided,
                    p_conversion=P_CONVERSION,
                    p_conversion_status=("not formed: upstream p is not an inferential test" if unformed
                        else "source p absent" if math.isnan(q) else "converted to two-sided"))
                if is_primary:
                    row.update(lens=LENS_MAP[lens], estimand=ESTIMAND_MAP[estimand],
                        estimate=effects[0], ci_low=effects[1], ci_high=effects[2], unit=EFFECT_UNIT,
                        **{k: int(v) for k, v in counts.items()})
                    if "conditioning" in row and _absent(row["conditioning"]):
                        row["conditioning"] = None
                    primary.append(row)
                else:
                    reasons = []
                    for excluded, reason in ((lens not in LENS_MAP, "additional lens"),
                        (estimand not in ESTIMANDS, "additional estimand"),
                        (source["variant"] != "primary", "nonprimary variant"),
                        (source["frame"] != "as delivered", "other frame"),
                        (layer not in LAYERS, "layer outside the seven")):
                        if excluded:
                            reasons.append(reason)
                    row.update(family="descriptive", p_holm=float("nan"),
                        descriptive_reason="; ".join(reasons))
                    descriptive.append(row)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{context}: {error}") from error
    return _columns(primary, PRIMARY_COLUMNS), _columns(descriptive, DESCRIPTIVE_COLUMNS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lenses", type=Path, required=True)
    parser.add_argument("--network", default="alexnet")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--descriptive-out", type=Path)
    args = parser.parse_args(argv)
    descriptive_out = args.descriptive_out or args.out.with_name(args.out.stem + "_descriptive.csv")
    check_outputs(files=[args.out, descriptive_out], inputs=[args.lenses])
    sources = sorted(args.lenses.glob(f"lenses_{args.network}_*/S2AS_LENSES.csv"))
    if not sources:
        raise ValueError("no per-layer S2AS_LENSES.csv files for this network")
    validate_paths([args.out, descriptive_out], [args.lenses, *sources])
    frames = ((str(path), pd.read_csv(path, keep_default_na=False, float_precision="round_trip"))
              for path in sources)
    primary, descriptive = adapt_frames(frames, network=args.network)
    if primary.empty:
        raise ValueError("no main rows in the size export for this network")
    publish_files([(primary, args.out), (descriptive, descriptive_out)])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error))
