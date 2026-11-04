"""Fig. 5 (animacy): recompute plotted values from data/figures/values_fig5_animacy.csv with the code in
analysis/animacy/, and run that code end to end on small synthetic circuits.

From the shipped table: the Holm-adjusted p values from the permutation p values, the observed-minus-null mean |s|,
the permutation counts behind each p value, the participation curves' exceedance of their max-T envelope, the plane's
participation counts against the curves' weights, the cross-network rows (excess, within-network Holm, significance
marks) through the collector that built them, and the ResNets' block means. From the joint count grids
(data/animacy/domain_grids_alexnet1000.npz): the plane and its hypergeometric null, and the observed participation
curves. Tests that write or read parquet files need pyarrow and are skipped without it.
"""
import csv
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import hypergeom

REPO = Path(__file__).resolve().parents[1]
ANIMACY = REPO / "analysis" / "animacy"
TABLE = REPO / "data" / "figures" / "values_fig5_animacy.csv"
GRIDS = REPO / "data" / "animacy" / "domain_grids_alexnet1000.npz"
if str(ANIMACY) not in sys.path:
    sys.path.insert(0, str(ANIMACY))

import library_statistics  # noqa: E402

LABELS = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7"]


def load(name):
    """Import a module of analysis/animacy under a name of its own (other folders' modules share sys.modules)."""
    spec = importlib.util.spec_from_file_location("animacy_" + name, ANIMACY / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STATS = load("animacy_statistics")
PLANE = load("specificity_plane")
COLLECT = load("collect_shared_layers")

with open(TABLE, newline="") as fh:
    ROWS = list(csv.DictReader(fh))
TEST = {(r["layer"], r["quantity"]): r["value"] for r in ROWS if r["panel"] == "test"}
NATIVE = {label: native for native, label in STATS.LAYER_NAMES.items()}
CROSS_NETWORK = [r for r in ROWS if r["panel"] == "xnet-test" and r["quantity"] == ""]
BLOCKS = [r for r in ROWS if r["panel"] == "xnet-test" and r["quantity"] == "block_mean"]


@pytest.fixture(scope="module")
def table():
    return pd.read_csv(TABLE, float_precision="round_trip", low_memory=False)


def test_modules_import():
    """Every module of analysis/animacy imports."""
    for name in ("animacy_statistics", "specificity_plane", "panel_tables", "collect_shared_layers",
                 "run_shared_animacy", "run_alexnet_animacy", "read_shared_layers"):
        load(name)
    import pair_statistics  # noqa: F401
    import incidence_package  # noqa: F401
    import cell_histograms  # noqa: F401
    import shared_overlap  # noqa: F401
    import shared_specificity  # noqa: F401
    import incidence_library  # noqa: F401


# AlexNet, all 1,000 categories: the layer test -----------------------------------------------------------------

def test_holm_adjusted_p_values():
    """p_holm: the permutation p values Holm-adjusted over the seven layers."""
    frame = pd.DataFrame([dict(model="alexnet1000", layer=NATIVE[label], p_perm=float(TEST[label, "p_perm"]))
                          for label in LABELS])
    adjusted = STATS.layer_holm(frame)
    assert list(adjusted.index) == [NATIVE[label] for label in LABELS]
    for label in LABELS:
        assert adjusted.loc[NATIVE[label], "p_holm"] == float(TEST[label, "p_holm"]), label
    assert list(STATS.holm_pvalues([0.01, 0.04, 0.03])) == pytest.approx([0.03, 0.06, 0.06])


def test_observed_minus_null():
    """The plotted layer test: observed mean |s| minus the label-permutation null mean, at full precision."""
    stats = [dict(model="alexnet1000", layer=NATIVE[label], obs_mean_abs_s=TEST[label, "obs_mean_abs_s"],
                  null_mean_abs_s=TEST[label, "null_mean_abs_s"]) for label in LABELS]
    excess = STATS.layer_excess(stats)
    assert [float(v) for v in excess] == [float(TEST[label, "obs_minus_null_mean_abs_s"]) for label in LABELS]


def test_permutation_counts_behind_p_values():
    """p = (1 + #null >= observed) / (B + 1), with B = 10,000 at fc6 and fc7 and 5,000 elsewhere."""
    for label in LABELS:
        B = library_statistics._semantic_row("alexnet1000", NATIVE[label], "", 0, 0, 0, {}, 0)["B_perms"]
        assert B == (10000 if label in ("fc6", "fc7") else 5000)
        count = float(TEST[label, "p_perm"]) * (B + 1)
        assert count == pytest.approx(round(count), abs=1e-6) and round(count) >= 1, label


# AlexNet: participation curves and the participation-specificity plane ----------------------------------------------

def test_participation_envelope_and_exceedance(table):
    """A bin exceeds when its curve is above the max-T envelope, null mean + crit * null SD, with one crit per layer."""
    rows = table[table.panel == "participation"]
    for layer, g in rows.groupby("layer", sort=False):
        g = g.sort_values("participation_bin")
        curve, mean, sd, bound = (g[c].to_numpy(float) for c in
                                  ("obs_curve", "null_curve_mean", "null_curve_sd", "envelope_bound"))
        valid = np.isfinite(mean) & (sd > 0) & np.isfinite(curve)
        assert np.array_equal(valid & (curve > bound), g.exceed.to_numpy(int).astype(bool)), layer
        crit = (bound[valid] - mean[valid]) / sd[valid]
        assert np.allclose(crit, crit[0], rtol=1e-12, atol=0), layer


def test_plane_participation_counts_match_curve_weights(table):
    """The plane's participation counts, gathered into the curves' ten participation bins, are the curves' weights."""
    plane = table[(table.panel == "plane") & (table.specificity_bin == 0)]
    curves = table[table.panel == "participation"]
    for layer, g in plane.groupby("layer", sort=False):
        g = g.sort_values("participation_bin")
        centre = 0.5 * (g.q_low.to_numpy() + g.q_high.to_numpy())
        bins = np.clip(np.digitize(centre, library_statistics.QEDGES) - 1, 0, library_statistics.NB - 1)
        weights = np.bincount(bins, weights=g.participation_route_count.to_numpy(), minlength=library_statistics.NB)
        shipped = curves[curves.layer == layer].sort_values("participation_bin").obs_wsum.to_numpy()
        assert np.array_equal(weights, shipped), layer


def test_plane_null_keeps_the_connection_count_and_has_its_contours(table):
    """The null redistributes each layer's connections, so its total is the observed total; contours at 1, 10, 50%."""
    assert PLANE.NULL_CONTOUR_FRACTIONS.tolist() == [0.01, 0.1, 0.5]
    plane = table[table.panel == "plane"]
    for layer, g in plane.groupby("layer", sort=False):
        observed, null = g.observed_route_count.to_numpy(), g.null_route_count.to_numpy()
        assert null.sum() == pytest.approx(observed.sum(), rel=1e-9), layer
        levels = PLANE.NULL_CONTOUR_FRACTIONS * null.max()
        assert all((null > level).any() for level in levels), layer


def test_plane_null_is_hypergeometric():
    """On a synthetic grid, plane()'s null puts each total's connections on k_A by the hypergeometric distribution."""
    assert PLANE.hyp_pmf(np.arange(0, 11), 1000, 401, 10) == pytest.approx(hypergeom.pmf(np.arange(0, 11), 1000, 401, 10),
                                                                           rel=1e-10)
    H = np.zeros((PLANE.nA + 1, PLANE.nI + 1))
    H[20, 30], H[10, 40], H[1, 2] = 7, 5, 4
    observed, null, marginal = PLANE.plane(H)
    expected = np.zeros(PLANE.NS)
    for k, n_k in ((50, 12), (3, 4)):
        kA = np.arange(max(0, k - PLANE.nI), min(PLANE.nA, k) + 1)
        pA, pI = kA / PLANE.nA, (k - kA) / PLANE.nI
        s = (pA - pI) / (pA + pI + PLANE.EPS)
        si = np.clip(np.digitize(s, PLANE.sb) - 1, 0, PLANE.NS - 1)
        np.add.at(expected, si, n_k * hypergeom.pmf(kA, PLANE.K, PLANE.nA, k))
    assert null.sum(axis=1) == pytest.approx(expected, rel=1e-9, abs=1e-12)
    assert observed.sum() == 16 and marginal.sum() == 16


def test_exact_overlap_reference():
    """exact_J0, the size-matched overlap reference: E[I / (a + b - I)] for I ~ Hypergeometric(E, a, b)."""
    for a, b, E in ((5, 7, 40), (30, 25, 100), (3, 400, 2000)):
        x = np.arange(0, min(a, b) + 1)
        exact = float((hypergeom.pmf(x, E, a, b) * x / (a + b - x)).sum())
        assert library_statistics.exact_J0(a, b, E) == pytest.approx(exact, rel=1e-12)


def test_plane_and_curves_from_the_joint_count_grids(table):
    """From the grids H[k_A, k_I]: the plane, its null and the observed participation curves, as the figure has them."""
    grids = np.load(GRIDS)
    plane = table[table.panel == "plane"]
    curves = table[table.panel == "participation"]
    for key in grids.files:
        layer, H = key[:-2], grids[key]
        observed, null, marginal = PLANE.plane(H.astype(float))
        g = plane[plane.layer == layer].sort_values(["specificity_bin", "participation_bin"])
        assert np.array_equal(observed.ravel(), g.observed_route_count.to_numpy()), layer
        np.testing.assert_allclose(null.ravel(), g.null_route_count.to_numpy(), rtol=1e-9, atol=1e-9)
        assert np.array_equal(np.tile(marginal, PLANE.NS), g.participation_route_count.to_numpy()), layer
        # The run took float32 means over single connections; the grid sums the same values in float64.
        result, curve, weights = library_statistics._observed_grid(H, PLANE.nA, PLANE.nI)
        c = curves[curves.layer == layer].sort_values("participation_bin")
        assert np.array_equal(weights, c.obs_wsum.to_numpy()), layer
        assert np.array_equal(curve, c.obs_curve.to_numpy(), equal_nan=True), layer
        label = STATS.LAYER_NAMES[layer]
        assert result["obs_mean_abs_s"] == pytest.approx(float(TEST[label, "obs_mean_abs_s"]), rel=1e-6), layer


# Four networks, shared categories: the cross-network panel ---------------------------------------------------------

VGG = dict(zip(["conv3_1", "conv3_2", "conv4_1", "conv4_2", "conv5_1", "conv5_2", "fc6", "fc7"],
               ["features.6", "features.8", "features.11", "features.13", "features.16", "features.18",
                "classifier.0", "classifier.3"]))


def native_layer(network, layer):
    """The circuit library's name of a layer the table names as the paper does."""
    if network == "alexnet":
        return NATIVE[layer]
    if network == "vgg11":
        return VGG[layer]
    if layer == "stem.conv":
        return "conv1"
    return re.sub(r"^stage([1-4])\.block([0-9]+)\.", r"layer\1.\2.", layer).replace("projection", "downsample.0")


def stage(network, native):
    if network == "alexnet" and native == "features.0":
        return "stem"
    if network.startswith("resnet"):
        return "stem" if native == "conv1" else native.split(".")[0]
    return native.split(".")[0]


def stored_layers():
    """The per-layer statistics behind the panel, rebuilt from the table: one item per network, as stored."""
    items = []
    for network in ("alexnet", "vgg11", "resnet18", "resnet50v1"):
        rows = sorted((r for r in CROSS_NETWORK if r["network"] == network), key=lambda r: int(r["layer_index"]))
        model = "resnet50" if network == "resnet50v1" else network
        stats = [dict(model=model, layer=native_layer(network, r["layer"]),
                      stage=stage(network, native_layer(network, r["layer"])), total_mult=1.0,
                      B_perms=int(r["n_permutations"]), obs_mean_abs_s=float(r["observed"]),
                      null_mean_abs_s=float(r["null"]), delta_pi_abs_s=float(r["observed"]) - float(r["null"]),
                      p_perm=float(r["p"])) for r in rows]
        items.append(dict(network=model, source="stored", layers=[s["layer"] for s in stats], stats=stats,
                          meta=dict(n_animate=int(rows[0]["n_animate"]), n_inanimate=int(rows[0]["n_inanimate"])),
                          density=[dict(layer=s["layer"], n_categories=int(r["n_categories"]))
                                   for s, r in zip(stats, rows)], rows=rows))
    return items


FIELDS = ["network", "layer", "layer_index", "n_layers", "n_categories", "n_animate", "n_inanimate", "observed",
          "null", "excess", "p", "p_holm", "n_permutations"]


def test_cross_network_rows_excess_and_within_network_holm():
    """The collector's rows from the stored statistics: excess = observed - null, Holm within each network."""
    for item in stored_layers():
        built = COLLECT.build_rows(item)
        assert len(built) == len(item["rows"])
        for got, shipped in zip(built, item["rows"]):
            for field in FIELDS:
                assert str(got[field]) == shipped[field], (item["network"], shipped["layer"], field)
    assert COLLECT.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_cross_network_marks_permutations_and_depths():
    """Filled points are Holm p < 0.05; B is 10,000 at every layer."""
    assert len(CROSS_NETWORK) == 88
    for r in CROSS_NETWORK:
        B = 10000
        assert int(r["n_permutations"]) == B, (r["network"], r["layer"])
        count = float(r["p"]) * (B + 1)
        assert count == pytest.approx(round(count), abs=1e-6)
        assert r["filled"] == str(float(r["p_holm"]) < 0.05)
        depth = (int(r["layer_index"]) - 1) / (int(r["n_layers"]) - 1)
        assert r["relative_depth"] == "%.6f" % depth


def test_resnet_block_means():
    """The ResNets' block means of the excess, joined in the panel, from the per-layer rows."""
    rows = [dict(r, value=r["excess"]) for r in CROSS_NETWORK]
    means = STATS.block_means(rows, "xnet-test")
    assert [(m["network"], m["layer"], m["value"], m["relative_depth"]) for m in means] == \
           [(b["network"], b["layer"], b["value"], b["relative_depth"]) for b in BLOCKS]


TABLE1_KEPT = {"alexnet": 89, "vgg11": 50, "resnet18": 272, "resnet50": 305}   # per 1,000 connections


def test_cross_network_rows_through_the_collectors(tmp_path):
    """read_shared_layers.py and collect_shared_layers.py, run on the stored statistics laid out as the run wrote them,
    give the table's 88 rows, and the checks file compares the kept percentages with Table 1."""
    pytest.importorskip("pyarrow")
    run = tmp_path / "run"
    manifests, counts = {}, {}
    for item in stored_layers():
        root = run
        (root / "finalization_pkg/semantic").mkdir(parents=True, exist_ok=True)
        (root / "handoff").mkdir(parents=True, exist_ok=True)
        model = item["network"]
        pd.DataFrame([{k: v for k, v in s.items()} for s in item["stats"]]).to_parquet(
            root / f"finalization_pkg/semantic/semantic_layer_stats_{model}.parquet")
        (root / f"finalization_pkg/semantic/semantic_meta_{model}.json").write_text(json.dumps(item["meta"]))
        kept = TABLE1_KEPT[model]
        for s, d in zip(item["stats"], item["density"]):
            manifests.setdefault(root, []).append(dict(model=model, layer=s["layer"], stage=s["stage"],
                                                       eligible_route_count=1000, eligible_positive_count=500,
                                                       eligible_negative_count=500))
            counts.setdefault(root, []).extend(dict(model=model, layer=s["layer"], category_id=c, retained_count=kept,
                                                    retained_positive_count=kept // 2,
                                                    retained_negative_count=kept - kept // 2)
                                               for c in range(d["n_categories"]))
    for root in manifests:
        pd.DataFrame(manifests[root]).to_csv(root / "handoff/layer_manifest.csv", index=False)
        pd.DataFrame(counts[root]).to_parquet(root / "handoff/category_layer_counts.parquet")
    env = dict(os.environ, CC_SHARED_RUN=str(run))
    layers = subprocess.run([sys.executable, str(ANIMACY / "read_shared_layers.py")], env=env, capture_output=True,
                            text=True, check=True).stdout
    (tmp_path / "layers.json").write_text(layers)
    (tmp_path / "density.tsv").write_text("network\tlayer\tdens_pos\tdens_neg\n")
    subprocess.run([sys.executable, str(ANIMACY / "collect_shared_layers.py"), "--layers", tmp_path / "layers.json",
                    "--density", tmp_path / "density.tsv", "--out", tmp_path / "rows.csv",
                    "--checks", tmp_path / "checks.json"], capture_output=True, text=True, check=True)
    with open(tmp_path / "rows.csv", newline="") as fh:
        got = list(csv.DictReader(fh))
    assert [[g[f] for f in FIELDS] for g in got] == [[s[f] for f in FIELDS] for s in CROSS_NETWORK]
    checks = json.loads((tmp_path / "checks.json").read_text())
    assert all(v["kept_percent_matches_table1"] for v in checks["networks"].values())
    assert {n: v["table1_kept_percent"] for n, v in checks["networks"].items()} == \
           {"alexnet": 8.9, "vgg11": 5.0, "resnet18": 27.2, "resnet50": 30.5}


# The code end to end, on small synthetic circuits -------------------------------------------------------------------

ALEXNET = [("features.0", "stem", 40), ("features.3", "features", 48), ("features.6", "features", 32),
           ("features.8", "features", 32), ("features.10", "features", 32), ("classifier.1", "classifier", 56),
           ("classifier.4", "classifier", 40)]


def test_alexnet_statistics_end_to_end(tmp_path):
    """Incidence library -> statistics (permutation null, curves, grids) -> plotted tables and the layer test."""
    pytest.importorskip("pyarrow")
    from incidence_library import build_library
    rng = np.random.default_rng(7)
    K, nA = 8, 3
    masks = {L: rng.random((K, n)) < 0.45 for L, _, n in ALEXNET}
    signs = {L: rng.random(n) < 0.5 for L, _, n in ALEXNET}
    categories = pd.DataFrame(dict(category_id_global=np.arange(K) * 10,
                                   animate_inanimate=["animate"] * nA + ["inanimate"] * (K - nA)))
    layers = pd.DataFrame([dict(model="alexnet", layer=L, stage=s, eligible_route_count=n,
                                eligible_positive_count=int(signs[L].sum()),
                                eligible_negative_count=int((~signs[L]).sum())) for L, s, n in ALEXNET])
    build_library(tmp_path / "library", model="alexnet", arm="P", axis="kernel_weight",
                  category_manifest=categories, layer_manifest=layers,
                  fidelity=pd.DataFrame(dict(category_id=categories.category_id_global)),
                  packed_reader=lambda L, lo, hi: np.packbits(masks[L][:, lo:hi], axis=1, bitorder="big"),
                  polarities=signs, metadata={"tag": "P", "scope": "alexnet1000"})
    env = dict(os.environ)
    script = ANIMACY / "run_alexnet_animacy.py"
    subprocess.run([sys.executable, str(script), "statistics", "--library", tmp_path / "library",
                    "--out", tmp_path / "statistics", "--nproc", "1"], env=env, capture_output=True, check=True)
    subprocess.run([sys.executable, str(script), "tables", "--statistics", tmp_path / "statistics",
                    "--out", tmp_path / "tables"], env=env, capture_output=True, check=True)
    stats = pd.read_parquet(tmp_path / "statistics/semantic_layer_stats_alexnet1000.parquet").set_index("layer")
    grids = np.load(tmp_path / "statistics/domain_grids_alexnet1000.npz")
    for L, _, n in ALEXNET:
        kA, kI = masks[L][:nA].sum(axis=0), masks[L][nA:].sum(axis=0)
        kept = kA + kI > 0
        pA, pI = kA[kept] / nA, kI[kept] / (K - nA)
        assert stats.loc[L, "obs_mean_abs_s"] == pytest.approx(np.mean(np.abs(pA - pI) / (pA + pI + 1e-9)), rel=1e-6)
        assert stats.loc[L, "B_perms"] == (10000 if L.startswith("classifier") else 5000)
        count = stats.loc[L, "p_perm"] * (stats.loc[L, "B_perms"] + 1)
        assert count == pytest.approx(round(count), abs=1e-6)
        H = np.zeros((nA + 1, K - nA + 1), np.int64)
        np.add.at(H, (kA[kept], kI[kept]), 1)
        assert np.array_equal(grids[L + "|H"], H), L
    # The plane table is plane(H) evaluated with this population's group sizes.
    namespace = {}
    exec(compile(Path(PLANE.__file__).read_text(), str(PLANE.__file__), "exec"), namespace)
    namespace.update(nA=nA, nI=K - nA, K=K)
    plane = pd.read_csv(tmp_path / "tables/animacy_plane.csv", float_precision="round_trip")
    for L, _, _ in ALEXNET:
        observed, null, _ = namespace["plane"](grids[L + "|H"].astype(float))
        g = plane[plane.layer == L].sort_values(["specificity_bin", "participation_bin"])
        assert np.array_equal(observed.ravel(), g.observed_route_count.to_numpy())
        assert np.allclose(null.ravel(), g.null_route_count.to_numpy(), rtol=1e-12, atol=1e-12)
    tests = pd.read_csv(tmp_path / "tables/animacy_layer_tests.csv", float_precision="round_trip").set_index("layer")
    assert list(tests.layer_label) == LABELS
    assert np.array_equal(tests.p_holm.to_numpy(), STATS.holm_pvalues(stats.loc[tests.index, "p_perm"].to_numpy()))
    assert np.array_equal(tests.obs_minus_null_mean_abs_s.to_numpy(),
                          (stats.loc[tests.index, "obs_mean_abs_s"] - stats.loc[tests.index, "null_mean_abs_s"]).to_numpy())


def test_shared_category_statistics_end_to_end(tmp_path, monkeypatch):
    """Shared-category incidence package -> overlap and animacy statistics of one network (run_shared_animacy.py)."""
    pytest.importorskip("pyarrow")
    import cell_histograms
    # The cell-histogram kernel compiles beside its module; keep the build in the test's folder.
    monkeypatch.setattr(cell_histograms, "SCRATCH", tmp_path / "scratch")
    rng = np.random.default_rng(3)
    K, nA, model = 7, 3, "alexnet"
    spec = [("features.0", "stem", 30), ("features.3", "features", 36), ("classifier.1", "classifier", 44)]
    package = tmp_path / "package"
    (package / "incidence_signatures").mkdir(parents=True)
    pd.DataFrame(dict(category_index=np.arange(K), category_id_global=np.arange(K) * 10,
                      animate_inanimate=["animate"] * nA + ["inanimate"] * (K - nA))).to_csv(
        package / "category_manifest.csv", index=False)
    layers, counts, signatures, intersections, masks = [], [], [], {}, {}
    for L, st, n in spec:
        m, positive = rng.random((K, n)) < 0.4, rng.random(n) < 0.5
        masks[L] = m
        layers.append(dict(model=model, layer=L, stage=st, eligible_route_count=n,
                           eligible_positive_count=int(positive.sum()), eligible_negative_count=int((~positive).sum())))
        counts += [dict(model=model, category_id=c * 10, layer=L, retained_count=int(m[c].sum()),
                        retained_positive_count=int(m[c][positive].sum()),
                        retained_negative_count=int(m[c][~positive].sum()), eligible_count=n) for c in range(K)]
        kept = m.any(axis=0)
        words = np.zeros((int(kept.sum()), 8), np.uint8)
        packed = np.packbits(m[:, kept].T, axis=1, bitorder="little")
        words[:, :packed.shape[1]] = packed
        rows = pd.DataFrame(dict(bits_0_63=words.view("<u8")[:, 0], structural_sign=np.where(positive[kept], "+", "-")))
        grouped = rows.groupby(["bits_0_63", "structural_sign"]).size().reset_index(name="multiplicity")
        signatures.append(grouped.assign(layer=L))
        intersections[f"{model}|{L}"] = m.astype(np.int64) @ m.astype(np.int64).T
    pd.DataFrame(layers).to_csv(package / "layer_manifest.csv", index=False)
    pd.DataFrame(counts).to_parquet(package / "category_layer_counts.parquet")
    table = pd.concat(signatures, ignore_index=True)
    table["bits_0_63"] = table.bits_0_63.astype(np.uint64)
    table.to_parquet(package / f"incidence_signatures/{model}.parquet")
    np.savez(package / "pairwise_intersections.npz", **intersections)
    shared = load("run_shared_animacy")
    shared.main(["--incidence-package", str(package), "--out", str(tmp_path / "out"), "--models", model])
    stats = pd.read_parquet(tmp_path / f"out/semantic/semantic_layer_stats_{model}.parquet").set_index("layer")
    for L, st, n in spec:
        m = masks[L]
        kA, kI = m[:nA].sum(axis=0), m[nA:].sum(axis=0)
        kept = kA + kI > 0
        pA, pI = kA[kept] / nA, kI[kept] / (K - nA)
        assert stats.loc[L, "obs_mean_abs_s"] == pytest.approx(np.mean(np.abs(pA - pI) / (pA + pI + 1e-9)), rel=1e-12)
        assert stats.loc[L, "B_perms"] == 10000
        count = stats.loc[L, "p_perm"] * (stats.loc[L, "B_perms"] + 1)
        assert count == pytest.approx(round(count), abs=1e-6)
    assert (tmp_path / f"out/overlap/overlap_layer_stats_{model}.parquet").is_file()
    assert not (ANIMACY / "scratch").exists()
