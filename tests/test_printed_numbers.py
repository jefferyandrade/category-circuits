"""Recompute numbers the paper prints from the summary tables in data/, with the statistics code that produced them where
the paper's numbers came from a statistic (means, percentile category bootstraps, Holm adjustments).

Printed values come from data/printed_numbers.tsv (the numbers as printed) and, for Table 1, whose
cells are typed in the table, from the table as printed. A recomputed value must round (half up) to the printed text.
"""
import csv, sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
sys.path.insert(0, str(REPO / "analysis" / "stats"))
import method_contrasts as mc  # noqa: E402
from category_resampling import block_bootstrap  # noqa: E402

with open(DATA / "printed_numbers.tsv", newline="") as fh:
    PRINTED = {r["macro"]: r["printed"] for r in csv.DictReader(fh, delimiter="\t")}


def same_as_printed(value, text):
    """True when value, rounded half up to the printed number of decimals, reads as the printed text."""
    text = text.replace(",", "").rstrip("%")
    places = len(text.split(".")[1]) if "." in text else 0
    q = Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    return q == Decimal(text.lstrip("+")) and (not text.startswith("+") or q > 0)


def check(macro, value):
    assert same_as_printed(value, PRINTED[macro]), (macro, value, PRINTED[macro])


def bootstrap(values, labels):
    return block_bootstrap(np.asarray(values), np.asarray(labels), n_resamples=10000, seed=0, unit="category")


def close(a, b, tol=1e-12):
    return abs(float(a) - float(b)) <= tol


# Fig. 3A and the abstract: AlexNet, all 1,000 categories, five pruning methods ------------------------------------

@pytest.fixture(scope="module")
def method_frames():
    pc = pd.read_csv(DATA / "method_comparison" / "per_category_endpoints.csv")
    return {a: pc[pc.arm == a].reset_index(drop=True) for a in mc.ARMS}


def test_method_means_and_intervals(method_frames):
    """absolute_means.csv, recomputed from the per-category endpoints."""
    ref = pd.read_csv(DATA / "method_comparison" / "absolute_means.csv")
    got = {}
    for metric in ("native_density", "corrected_accuracy_drop"):
        paired = mc.paired_frames(method_frames, mc.ARMS, metric, unit="category")
        for arm in mc.ARMS:
            iv = bootstrap(paired[arm][metric], paired[arm].category)
            r = ref[(ref.arm == arm) & (ref.metric == metric)].iloc[0]
            assert all(close(iv[k], r[k]) for k in ("estimate", "ci_low", "ci_high")), (arm, metric)
            got[arm, metric] = iv
    kept, drop = got["P", "native_density"], got["P", "corrected_accuracy_drop"]
    check("CoreAlexKeptPct", 100 * kept["estimate"])
    check("CoreAlexKeptPctLo", 100 * kept["ci_low"])
    check("CoreAlexKeptPctHi", 100 * kept["ci_high"])
    check("CoreAlexAccDrop", drop["estimate"])
    check("CoreAlexAccDropLo", drop["ci_low"])
    check("CoreAlexAccDropHi", drop["ci_high"])
    check("MCAccDropSAH", got["SA-H", "corrected_accuracy_drop"]["estimate"])
    check("MCAccDropSAG", got["SA-G", "corrected_accuracy_drop"]["estimate"])


def test_method_contrasts(method_frames):
    """Paired differences from E/I-separate pruning (method_contrasts.csv rows), recomputed."""
    ref = pd.read_csv(DATA / "method_comparison" / "method_contrasts.csv")
    m = "corrected_accuracy_drop"
    paired = mc.paired_frames(method_frames, mc.ARMS, m, unit="category")
    rows = {}
    for arm in ("SA-H", "SA-G", "SG", "SF"):
        row = mc.contrast_row(paired["P"], paired[arm][m].to_numpy() - paired["P"][m].to_numpy(), m, arm + " - P",
                              n_resamples=10000, seed=0, unit="category")
        r = ref[(ref.metric == m) & (ref.contrast == arm + " - P")].iloc[0]
        assert all(close(row[k], r[k]) for k in ("estimate", "ci_low", "ci_high", "p")), arm
        rows[arm] = row
    for arm, name in (("SA-H", "SAH"), ("SA-G", "SAG")):
        check("MCDiffAcc%sAbs" % name, rows[arm]["estimate"])
        check("MCDiffAcc%sLo" % name, rows[arm]["ci_low"])
        check("MCDiffAcc%sHi" % name, rows[arm]["ci_high"])
    check("MCDiffAccSG", rows["SG"]["estimate"])
    check("MCDiffAccSGLo", rows["SG"]["ci_low"])
    check("MCDiffAccSGHi", rows["SG"]["ci_high"])


# Table 1 -------------------------------------------------------------------------------------------------------------

TABLE1 = {  # as printed: prunable weights, kept % [95%], dense, circuit, accuracy lost [95%]
    "alexnet": (56994496, (8.9, 8.3, 9.4), 85.1, 80.4, (4.7, 3.7, 5.7)),
    "vgg11": (128679936, (5.0, 4.9, 5.2), 90.2, 85.2, (5.0, 3.8, 6.2)),
    "resnet18": (11166912, (27.2, 26.4, 28.0), 92.2, 88.1, (4.1, 3.4, 4.8)),
    "resnet50v1": (23454912, (30.5, 29.8, 31.3), 94.4, 89.4, (5.0, 4.2, 5.8)),
}


@pytest.mark.parametrize("net", sorted(TABLE1))
def test_table1(net):
    weights, kept, dense, circuit, lost = TABLE1[net]
    d = DATA / "table1"
    manifest = pd.read_csv(d / "layer_manifest.csv")
    model = {"resnet50v1": "resnet50"}.get(net, net)
    assert int(manifest[manifest.model == model].eligible_route_count.sum()) == weights
    f = pd.read_csv(d / ("fidelity_%s.csv" % net)).sort_values("category_id")
    assert same_as_printed(100 * f.dense_acc.mean(), str(dense))
    assert same_as_printed(100 * f.circuit_acc.mean(), str(circuit))
    iv = bootstrap(100 * (f.dense_acc - f.circuit_acc), f.category_id)
    t1 = pd.read_csv(d / "table1.csv")
    sel = lambda metric: t1[(t1.network == net) & (t1.metric == metric) & (t1.contrast == "network mean")].iloc[0]
    r = sel("corrected_accuracy_drop")
    assert all(close(iv[k], r[k], 1e-9) for k in ("estimate", "ci_low", "ci_high"))
    assert all(same_as_printed(iv[k], str(v)) for k, v in zip(("estimate", "ci_low", "ci_high"), lost))
    r = sel("native_density")
    assert all(same_as_printed(100 * r[k], str(v)) for k, v in zip(("estimate", "ci_low", "ci_high"), kept))


def test_table1_alexnet_kept_from_per_category_rows():
    """AlexNet's kept fraction on the 100 shared categories, recomputed from all-category per-category rows."""
    blocks = pd.read_csv(REPO / "taxonomy" / "inputs" / "CATEGORY_BLOCKS_cap50.tsv", sep="\t")
    shared = set(blocks[blocks.shared100 == 1].class_index)
    pc = pd.read_csv(DATA / "method_comparison" / "per_category_endpoints.csv")
    p = pc[(pc.arm == "P") & pc.category.isin(shared)].sort_values("category")
    assert len(p) == 100
    iv = bootstrap(p.native_density, p.category)
    t1 = pd.read_csv(DATA / "table1" / "table1.csv")
    r = t1[(t1.network == "alexnet") & (t1.metric == "native_density") & (t1.contrast == "network mean")].iloc[0]
    assert all(close(iv[k], r[k], 1e-12) for k in ("estimate", "ci_low", "ci_high"))


# Figs. 3B, 3C and 4 --------------------------------------------------------------------------------------------------

def test_fig3c_and_fig4a_sign_density():
    d = pd.read_csv(DATA / "sign_density" / "per_category_sign_density.csv")
    p6, s6 = d[(d.arm == "P") & (d.layer == "fc6")], d[(d.arm == "SAH") & (d.layer == "fc6")]
    assert len(p6) == len(s6) == 1000
    check("SignRatioMeansSeparateFcSix", p6.dens_neg.mean() / p6.dens_pos.mean())
    check("SignRatioMeansPooledFcSix", s6.dens_neg.mean() / s6.dens_pos.mean())
    for layer, macro in (("fc6", "DensMeanFcSix"), ("conv1", "DensMeanConvOne")):
        x = d[(d.arm == "P") & (d.layer == layer)]
        check(macro, ((x.kept_pos + x.kept_neg) / (x.avail_pos + x.avail_neg)).mean())


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p, kind="mergesort")
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(adj)
    out[order] = np.minimum(adj, 1.0)
    return out


def test_fig4d_sharing_by_sign():
    """Holm within each network, recomputed from the two-sided p values, and the counts the text prints."""
    d = pd.read_csv(DATA / "sign_sharing" / "sign_sharing_four_networks_holm.tsv", sep="\t")
    for net, g in d.groupby("network"):
        assert np.allclose(holm(g.p_two_sided), g.p_holm, rtol=1e-9, atol=0), net
    for net, name in (("vgg11", "Vgg"), ("resnet18", "ResEighteen"), ("resnet50v1", "ResFiftyVOne")):
        g = d[d.network == net]
        inh = g[(g.p_holm < 0.05) & (g.ie > 0)]
        check("ShareSignLayers" + name, len(g))
        check("ShareSignHolmInh" + name, len(inh))
    g = d[d.network == "resnet50v1"]
    inh = g[(g.p_holm < 0.05) & (g.ie > 0)]
    check("ShareSignHolmInhLessKeptResFiftyVOne", int((inh.dens_inh < inh.dens_exc).sum()))


def test_fig3a_frontier_and_fig3b_resnet50():
    f = pd.read_csv(DATA / "frontier" / "pairing.csv")
    for arm, name in (("SA-H", "SAH"), ("RP1", "RPOne")):
        r = f[(f.arm == arm) & (f.metric == "accuracy")].iloc[0]
        assert close(r.estimate, r.new_mean - r.comparator_mean, 1e-9)
        check("ResAFrontier%sAccuracyDiff" % name, r.estimate)
        check("ResAFrontier%sAccuracyMargLo" % name, r.ci_low)
        check("ResAFrontier%sAccuracyMargHi" % name, r.ci_high)
    v = pd.read_csv(DATA / "resnet50_pooled" / "values_fig3B_resnet50.csv")
    drop = v[v.metric == "corrected_accuracy_drop"].set_index("arm_or_contrast")
    assert close(drop.estimate["SA-H - P"], drop.estimate["SA-H"] - drop.estimate["P"], 1e-9)
    check("ResAResNetFiftyVOneAccDiff", drop.estimate["SA-H - P"])
    check("ResAResNetFiftyVOneN", drop.n_categories["SA-H - P"])


# Fig. 6 and the readout-weight control ---------------------------------------------------------------------------------

def test_fig6_alexnet_tree_shares():
    def row(layer, lens):
        d = pd.read_csv(DATA / "tree_share" / ("tree_contrast_alexnet_%s.csv" % layer))
        return d[(d.level == "superordinate") & (d.lens == lens)].iloc[0]
    check("CapWireFcSix", row("fc6", "W").estimate)
    check("CapActFcSix", row("fc6", "A").estimate)
    for layer, name in (("fc6", "FcSix"), ("fc7", "FcSeven")):
        check("CapDiff" + name, row(layer, "W").estimate - row(layer, "A").estimate)
        wa, aw = row(layer, "W|A"), row(layer, "A|W")
        check("CapPartWire" + name, wa.estimate)
        check("CapPartWire%sLo" % name, wa.ci_low)
        check("CapPartWire%sHi" % name, wa.ci_high)
        check("CapPartAct" + name, aw.estimate)


def test_readout_control_macros():
    d = DATA / "readout_control" / "readout"
    last = {"alexnet": "fc7", "vgg11": "fc7", "resnet18": "stage4.block1.conv2", "resnet50v1": "stage4.block2.conv3"}
    beyond = {}
    for net, layer in last.items():
        t = pd.read_csv(d / ("tc_%s_%s_abs" % (net, layer)) / ("tree_contrast_%s.csv" % layer))
        beyond[net] = t[(t.level == "superordinate") & (t.lens == "W|A")].iloc[0].estimate
    check("ResDAlexTreeBeyondReadoutFcSeven", beyond["alexnet"])
    check("ResDTreeBeyondReadoutLastLoVOne", min(beyond.values()))
    check("ResDTreeBeyondReadoutLastHiVOne", max(beyond.values()))


def test_table1_thresholds_from_the_pilots():
    """The other networks' thresholds, matched to AlexNet's loss share by the pilot rule (Methods)."""
    import subprocess
    r = subprocess.run([sys.executable, "-I", str(REPO / "analysis" / "scripts" / "price_match.py"),
                        str(DATA / "thresholds" / "pilots_round1.csv"), str(DATA / "thresholds" / "pilots_round2.csv")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    chosen = {}
    for line in r.stdout.splitlines():
        if line.strip().startswith("->"):
            net, rest = line.strip()[2:].split(":", 1)
            chosen[net.strip()] = float(rest.split("price")[1].split()[0])
    assert "anchor AlexNet 1.5e-7" in r.stdout and PRINTED["MethPriceAlex"] == "1.5e-7"
    assert chosen == {"vgg11": 1.768e-07, "resnet18": 5e-07, "resnet50v2": 4.033e-07}, chosen


def test_resnet50v1_threshold_check():
    """ResNet-50 (IMAGENET1K_V1) took the threshold piloted on its IMAGENET1K_V2 checkpoint (the pilots above). Checked
    after the extraction on the same 21 pilot categories, its circuits' loss share (0.100) is within one standard
    error of AlexNet's (0.108) under the same rule (Methods)."""
    import re
    import subprocess
    r = subprocess.run([sys.executable, "-I", str(REPO / "analysis" / "scripts" / "price_match.py"),
                        str(DATA / "thresholds" / "pilots_round1.csv"), str(DATA / "thresholds" / "pilots_round2.csv"),
                        str(DATA / "thresholds" / "resnet50v1_check.csv")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    anchor = float(re.search(r"anchor AlexNet 1\.5e-7: share ([0-9.]+)", r.stdout).group(1))
    v1 = re.search(r"-> resnet50v1_campaign: piloted price ([0-9.e-]+) \(share ([0-9.]+), within one SE", r.stdout)
    assert v1 and float(v1.group(1)) == 4.033e-07, r.stdout
    assert same_as_printed(anchor, "0.108") and same_as_printed(float(v1.group(2)), "0.100")


# Section 2.3 -------------------------------------------------------------------------------------------------------------

def test_section_2_3_sparsity_against_tree_structure():
    """Across the networks' main-path layers, sparser wiring carries more category-tree structure (section 2.3)."""
    import re
    import subprocess
    r = subprocess.run([sys.executable, "-I", str(REPO / "analysis" / "scripts" / "parting_tree_stats.py"),
                        str(DATA / "parting" / "parting_tree.csv")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    main = next(line for line in r.stdout.splitlines() if line.startswith("main path"))
    check("ResCPartingLayersVOne", int(re.search(r"n=(\d+)", main).group(1)))
    check("ResCPartingRhoVOne", float(re.search(r"tree~kept rho (-?[0-9.]+)", main).group(1)))
    check("ResCPartingRhoBeyondDepthVOne", float(re.search(r"tree~kept \| depth r (-?[0-9.]+)", main).group(1)))
    controls = pd.read_csv(DATA / "parting" / "parting_controls.csv")
    size = controls[(controls.layers == "main path") & (controls.relation == "kept fraction ~ log_routes")
                    & (controls.controls == "none")].iloc[0]
    check("ResCPartingRhoLayerSizeVOne", size.r)
