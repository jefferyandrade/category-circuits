"""Summary table of the readout-weight control, SUMMARY_estimates.tsv, from the per-task tree_contrast outputs.

For each network's last two read layers and each lens (the absolute and the signed readout geometry, and the recorded
output-geometry control), at the superordinate and domain levels: the wiring's chance-corrected tree share (W), the
lens's own share (A), the wiring's share beyond the lens (W|A), the lens's share beyond the wiring (A|W), each as
"estimate [95% delete-one-category jackknife interval]", and W|A / W. The planted control (AlexNet fc7, readout rows
shuffled across categories) closes the table, with the lens's permutation p in place of the ratio.

Usage: summarize.py --readout-out DIR --outgeom-out DIR --out FILE
  DIR for --readout-out holds tc_<inputs>_<layer>_<abs|signed>/tree_contrast_<layer>.csv (run_readout_control.sh);
  DIR for --outgeom-out holds outgeom_tc_<network>_<layer>/tree_contrast_<layer>.csv (the output-geometry control).
"""
import argparse
from pathlib import Path

import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import check_outputs, publish_file

LAYERS = (("alexnet", "conv5"), ("alexnet", "fc6"), ("alexnet", "fc7"), ("vgg11", "fc6"), ("vgg11", "fc7"),
          ("resnet18", "stage4.block0.conv2"), ("resnet18", "stage4.block1.conv2"),
          ("resnet50v1", "stage4.block1.conv3"), ("resnet50v1", "stage4.block2.conv3"))
LEVELS = ("superordinate", "domain")
COLUMNS = ("network", "layer", "lens", "level", "wiring", "lens_alone", "wiring_beyond_lens", "lens_beyond_wiring",
           "beyond/alone")


def read(path):
    d = pd.read_csv(path)
    if not (d["status"] == "FORMED").all():
        raise ValueError("%s has rows that did not form" % path)
    return d


def cell(d, level, lens):
    r = d[(d["level"] == level) & (d["lens"] == lens)]
    if len(r) != 1:
        raise ValueError("expected one %s row at level %s, found %d" % (lens, level, len(r)))
    return r.iloc[0]


def fmt(r):
    return "%.2f [%.2f, %.2f]" % (r["estimate"], r["ci_low"], r["ci_high"])


def rows(d, net, layer, name, planted=False):
    for level in LEVELS:
        w, a, wa, aw = (cell(d, level, k) for k in ("W", "A", "W|A", "A|W"))
        if planted:
            yield (net, layer, name, level, fmt(w), fmt(a) + " p=%.2f" % a["p_perm"], fmt(wa), fmt(aw), "")
        else:
            yield (net, layer, name, level, fmt(w), fmt(a), fmt(wa), fmt(aw), "%.2f" % (wa["estimate"] / w["estimate"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("readout-out", "outgeom-out", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args(argv)
    check_outputs(files=[a.out], inputs=[a.readout_out, a.outgeom_out])
    ro, og = Path(a.readout_out), Path(a.outgeom_out)
    out = []
    for net, layer in LAYERS:
        for kind in ("abs", "signed"):
            d = read(ro / ("tc_%s_%s_%s" % (net, layer, kind)) / ("tree_contrast_%s.csv" % layer))
            out.extend(rows(d, net, layer, "readout_" + kind))
        d = read(og / ("outgeom_tc_%s_%s" % (net, layer)) / ("tree_contrast_%s.csv" % layer))
        out.extend(rows(d, net, layer, "output_geometry"))
    d = read(ro / "tc_alexnet_shuf_fc7_abs" / "tree_contrast_fc7.csv")
    out.extend(rows(d, "alexnet", "fc7", "SHUFFLED_abs", planted=True))
    text = "\t".join(COLUMNS) + "\n" + "".join("\t".join(r) + "\n" for r in out)
    publish_file(text, a.out)
    print("wrote %d rows to %s" % (len(out), a.out))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error))
