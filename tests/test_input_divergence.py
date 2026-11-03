"""Fig. 4C, input divergence: the code imports from this repository, its estimators recover planted values, and the
shipped table's block means, differences and intervals are recomputed from its own per-layer rows."""
import csv
import importlib
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
DIVERGENCE = REPO / "analysis" / "input_divergence"
COMMON = REPO / "analysis" / "common"
for path in (DIVERGENCE, REPO / "analysis" / "stats", COMMON):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

FIG4 = REPO / "data" / "figures" / "values_fig4_sharing.csv"
SCRIPTS = ["divergence_math", "run_four_networks", "network_sources", "alexnet_references", "alexnet_intervals",
           "census", "publish"]
SHARED = ["populations", "architectures", "route_layout", "strata", "block_means"]


def panel_c():
    with FIG4.open(newline="") as handle:
        return [r for r in csv.DictReader(handle) if r["panel"] == "C"]


@pytest.mark.parametrize("name", SCRIPTS)
def test_scripts_import_from_this_repository(name):
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().parent == DIVERGENCE


@pytest.mark.parametrize("name", SHARED)
def test_shared_modules_import_from_this_repository(name):
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().parent == COMMON


def test_layer_layout_and_populations_come_from_this_repository():
    import network_sources
    layouts, populations = network_sources._imports()
    assert Path(layouts.__file__).resolve() == REPO / "analysis" / "stats" / "circuit_masks.py"
    assert Path(populations.__file__).resolve() == COMMON / "populations.py"


def brute_pair_sums(mask):
    """Divergence numerator and maximum of every category pair, unit by unit, with Python sets."""
    q, units, n = mask.shape
    num = np.zeros((q, q), dtype=np.int64)
    den = np.zeros((q, q), dtype=np.int64)
    for a, b in itertools.permutations(range(q), 2):
        for u in range(units):
            A = set(np.flatnonzero(mask[a, u]))
            B = set(np.flatnonzero(mask[b, u]))
            low = min(len(A), len(B))
            top = low - max(0, len(A) + len(B) - n)
            if top > 0:
                num[a, b] += low - len(A & B)
                den[a, b] += top
    return num, den


@pytest.mark.parametrize("inputs", [9, 40])
def test_pair_sums_match_enumeration_and_the_census(inputs):
    """The cross-network estimator, the census (both of its code paths) and set enumeration agree exactly."""
    import divergence_math as dm
    import census
    rng = np.random.default_rng(inputs)
    q, units = 6, 7
    density = rng.uniform(0.05, 0.95, size=(q, 1, 1))
    mask = rng.random((q, units, inputs)) < density
    mask[0, 0] = True                    # a full unit and an empty one contribute nothing
    mask[1, 1] = False
    num, den = brute_pair_sums(mask)
    totals = dm.PairTotals(q)
    totals.update(mask, dm.decile_labels(rng.random((units, inputs))))
    raw = totals.finalize()
    assert np.array_equal(raw["numerator"], num) and np.array_equal(raw["denominator"], den)
    c_num, c_den, _ = census.group_totals(census.to_groups(mask, "unit_fanin"))
    assert np.array_equal(c_num, num) and np.array_equal(c_den, den)


def test_references_are_exact_expected_overlaps():
    """E[I] for random kept sets of the same sizes, by enumeration: c1 c2 / n overall, summed per stratum."""
    import divergence_math as dm
    n = 6
    strata = np.array([[0, 0, 0, 1, 1, 1]], dtype=np.uint8)
    masks = np.array([[[1, 0, 1, 1, 0, 0]], [[0, 1, 1, 0, 1, 1]]], dtype=bool)
    counts = masks.sum(axis=2)
    sizes = np.array([[3, 3]])
    per_stratum = np.stack([(masks & (strata[None] == s)).sum(axis=2) for s in range(2)], axis=2)
    low, uniform, matched = dm.reference_totals(counts, per_stratum, sizes)
    c1, c2 = counts[:, 0]

    def mean_overlap(choices_a, choices_b):
        return np.mean([len(set(a) & set(b)) for a in choices_a for b in choices_b])

    every = range(n)
    expected_uniform = mean_overlap(itertools.combinations(every, c1), itertools.combinations(every, c2))
    assert uniform[0, 1] == pytest.approx(min(c1, c2) - expected_uniform, abs=1e-12)

    def stratified(k):
        a_lo, a_hi = per_stratum[k, 0]
        return [lo + hi for lo in itertools.combinations(range(3), a_lo)
                for hi in itertools.combinations(range(3, 6), a_hi)]

    expected_matched = mean_overlap(stratified(0), stratified(1))
    assert matched[0, 1] == pytest.approx(min(c1, c2) - expected_matched, abs=1e-12)
    assert low[0, 1] == min(c1, c2)


def test_jackknife_deletes_every_category():
    import alexnet_references
    import divergence_math as dm
    rng = np.random.default_rng(3)
    q = 9
    x = rng.random((q, q))
    x = (x + x.T) / 2
    x[0, 5] = x[5, 0] = np.nan
    np.fill_diagonal(x, np.nan)
    result = dm.jackknife(x)
    iu = np.triu_indices(q, 1)
    point = np.nanmean(x[iu])
    deleted = []
    for k in range(q):
        keep = [i for i in range(q) if i != k]
        sub = x[np.ix_(keep, keep)]
        deleted.append(np.nanmean(sub[np.triu_indices(q - 1, 1)]))
    deleted = np.array(deleted)
    se = np.sqrt((q - 1) / q * np.sum((deleted - deleted.mean()) ** 2))
    assert result["point"] == pytest.approx(point, abs=1e-15)
    assert result["se"] == pytest.approx(se, rel=1e-12)
    from scipy.stats import t
    assert result["upper"] - result["point"] == pytest.approx(t.ppf(0.975, q - 1) * se, rel=1e-12)
    other = alexnet_references.jackknife(x)
    assert other["point"] == pytest.approx(result["point"], abs=1e-15)
    assert other["se"] == pytest.approx(result["se"], rel=1e-12)


def test_census_values_select_the_leading_p_block():
    import run_four_networks
    num = np.arange(25).reshape(5, 5) % 4
    den = np.full((5, 5), 4)
    values = run_four_networks.census_p_values({"categories": np.array([7, 3, 9]), "num": num, "den": den})
    assert values.shape == (3, 3) and np.isnan(np.diag(values)).all()
    assert values[0, 1] == num[0, 1] / 4


def test_census_exact_summary():
    import census
    x = np.array([[np.nan, .2, .4], [.2, np.nan, .6], [.4, .6, np.nan]])
    out = census.exact_summary(x, np.array([0, 0, 1]), np.ones((3, 3), dtype=np.int64))
    assert out["mean_all"] == pytest.approx(0.4) and out["mean_within_block"] == pytest.approx(0.2)
    assert out["mean_cross_block"] == pytest.approx(0.5)
    x[0, 1] = x[1, 0] = np.nan
    out = census.exact_summary(x, np.array([0, 0, 1]), np.ones((3, 3), dtype=np.int64))
    assert out["mean_all"] is None and out["mean_over_defined_pairs_only"] == pytest.approx(0.5)


def test_block_means_reproduce_the_shipped_panel():
    import block_means
    rows = panel_c()
    shipped = {(r["network"], r["layer"]): r for r in rows if r["quantity"] == "block_mean"}
    means = block_means.block_means(rows, "C")
    assert len(means) == len(shipped) == 15
    for m in means:
        r = shipped[(m["network"], m["layer"])]
        assert m["value"] == r["value"] and m["relative_depth"] == r["relative_depth"]


def test_shipped_differences_and_intervals_are_consistent():
    """Each difference from a reference is the observed mean minus the reference mean over the same pairs, and every
    jackknife interval is centred on its point."""
    rows = [r for r in panel_c() if r["quantity"] != "block_mean"]
    assert len(rows) == 55
    for r in rows:
        f = lambda key: float(r[key])
        if r["network"] == "alexnet":
            pairs = [("uniform_reference", "observed_minus_uniform", "uniform_difference"),
                     ("weight_size_matched_reference", "observed_minus_weight_size_matched", "weight_size_difference")]
            for ref, diff, ci in pairs:
                assert f("observed") - f(ref) == pytest.approx(f(diff), abs=1e-12)
                assert (f(ci + "_lower") + f(ci + "_upper")) / 2 == pytest.approx(f(diff), abs=1e-12)
            assert f("reference") == f("weight_size_matched_reference")
        else:
            for ref, diff in [("ref_uniform", "obs_minus_uniform"), ("ref_weight", "obs_minus_weight")]:
                assert f("observed") - f(ref) == pytest.approx(f(diff), abs=1e-12)
                assert (f(diff + "_ci_low") + f(diff + "_ci_high")) / 2 == pytest.approx(f(diff), abs=1e-12)
            assert f("reference") == f("ref_weight")
            # The figure leaves out layers where fewer than half of the pairs are defined.
            assert 2 * f("n_pairs_defined") >= f("n_pairs")
        assert (f("ci_low") + f("ci_high")) / 2 == pytest.approx(f("value"), abs=1e-12)


def test_alexnet_intervals_from_a_census(tmp_path, monkeypatch):
    """alexnet_intervals.py on census pair sums of 1,000 P categories followed by repeat extractions: each layer's
    point and interval are the jackknife of the leading P block, for the five layers the figure plots."""
    import alexnet_intervals
    import divergence_math as dm
    rng = np.random.default_rng(4)
    q = 1002
    den = np.triu(rng.integers(1, 30, size=(q, q)), 1)
    den[3, 7] = 0                        # one undefined pair
    num = np.triu(rng.integers(0, 30, size=(q, q)), 1) % (den + 1)
    den, num = (den + den.T).astype(np.uint8), (num + num.T).astype(np.uint8)
    folders = [lid + "_" + grain for name, lid, grain in alexnet_intervals.ALEX if name in alexnet_intervals.LAYERS]
    for folder in folders:
        (tmp_path / folder).mkdir()
        np.savez(tmp_path / folder / "pair_totals.npz", num=num, den=den, categories=np.arange(1000),
                 redraw_categories=np.array([0, 10]))
    monkeypatch.setattr(alexnet_intervals, "CENSUS", tmp_path)
    monkeypatch.setattr(sys, "argv", ["alexnet_intervals.py", "--out", str(tmp_path / "out")])
    alexnet_intervals.main()
    with (tmp_path / "out" / "values_fig4C_alexnet_all1000.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = dm.jackknife(dm.pair_values(num[:1000, :1000], den[:1000, :1000]))
    assert [r["layer"] for r in rows] == alexnet_intervals.LAYERS == ["conv3", "conv4", "conv5", "fc6", "fc7"]
    for r in rows:
        assert float(r["value"]) == expected["point"] and float(r["ci_low"]) == expected["lower"]
        assert float(r["ci_high"]) == expected["upper"] and int(r["n_delete_defined"]) == 1000


class _Weight:
    """A checkpoint tensor stand-in: detach().cpu().numpy() gives the array."""

    def __init__(self, array):
        self.array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


def test_four_network_layer_from_masks(tmp_path):
    """run_four_networks.layer_task reads a layer's masks by bit offset, groups a convolution's inputs by input channel,
    checks the kept counts against the receipts, the Gram file and the kept fractions by sign, and gives the pair sums
    and values of divergence_math on the grouped masks; a kept count that differs from the receipts stops it."""
    import divergence_math as dm
    import run_four_networks
    from route_layout import LayerSpec
    rng = np.random.default_rng(8)
    cats = [0, 10, 20, 30, 40, 50]
    conv = LayerSpec("stage1.block0.conv1", "layer1.0.conv1", "conv_kernel_weight", (12, 8, 3, 3), 0, 864)
    linear = LayerSpec("fc", "fc", "linear_weight", (16, 24), 864, 384)
    # Sparse convolution masks, so that circuits keep some but not all input channels of a unit.
    density = np.concatenate([np.full(864, 0.06), np.full(384, 0.4)]) * rng.uniform(0.5, 1.5, size=(len(cats), 1))
    masks = rng.random((len(cats), 1248)) < density
    paths = []
    for c, m in zip(cats, masks):
        paths.append(tmp_path / f"mask{c}.packbits")
        np.packbits(m, bitorder="big").tofile(paths[-1])
    weights = {s.module_name: rng.normal(size=s.shape) for s in (conv, linear)}
    state = {name + ".weight": _Weight(w) for name, w in weights.items()}
    state["layer1.0.bn1.weight"] = _Weight(rng.uniform(0.5, 1.5, size=12))
    receipts = [{s.layer_id: int(m[s.offset:s.stop].sum()) for s in (conv, linear)} for m in masks]
    densities = {}
    for s in (conv, linear):
        negative, kept = weights[s.module_name].reshape(-1) < 0, masks[:, s.offset:s.stop]
        densities[s.layer_id] = {"dens_pos": "%.4f" % ((kept & ~negative).sum() / (len(cats) * (~negative).sum())),
                                 "dens_neg": "%.4f" % ((kept & negative).sum() / (len(cats) * negative.sum()))}
    for spec in (conv, linear):
        item = dict(layer=spec.layer_id, spec=spec, layer_index=1, source="test",
                    expected_counts={c: r[spec.layer_id] for c, r in zip(cats, receipts)})
        ctx = dict(network="resnet18", categories=cats, state=state, mask_paths=paths, receipt_counts=receipts,
                   densities=densities, layers=[item], mask_root=tmp_path, arm="P", checkpoint="checkpoint",
                   checkpoint_sha256=None)
        result = run_four_networks.layer_task(ctx, item, tmp_path / "out")
        layer = masks[:, spec.offset:spec.stop].reshape((len(cats),) + spec.shape)
        grouped = layer.reshape(len(cats), spec.shape[0], spec.shape[1], -1).any(-1) if layer.ndim == 5 else layer
        scores = np.abs(weights[spec.module_name])
        scores = scores.sum(axis=(2, 3)) if scores.ndim == 4 else scores
        totals = dm.PairTotals(len(cats))
        totals.update(grouped, dm.decile_labels(scores))
        expected = dm.summarize(totals.finalize())
        assert np.isfinite(expected["observed"])
        assert result["values"]["observed"] == expected["observed"]
        assert result["values"]["ref_weight"] == expected["ref_weight"]
        assert result["kept"]["retained_count_total"] == int(masks[:, spec.offset:spec.stop].sum())
        stored = np.load(result["pair_totals_path"])
        assert np.array_equal(stored["numerator"], totals.finalize()["numerator"])
    receipts[2] = dict(receipts[2], fc=receipts[2]["fc"] + 1)
    item = dict(layer="fc", spec=linear, layer_index=1, source="test", expected_counts=None)
    ctx.update(receipt_counts=receipts, layers=[item])
    with pytest.raises(ValueError, match="receipts"):
        run_four_networks.layer_task(ctx, item, tmp_path / "out")


def test_defaults_are_the_paper_settings():
    import census
    import alexnet_references
    args = census.parser().parse_args(["--network", "alexnet", "--layer", "readout.hidden1", "--grain",
                                       "unit_fanin", "--out", "x", "--masks-root", "y"])
    assert args.p_arm == "v0_local_attribution_sign_plain_t1p5e7_v64"
    assert args.r_arm == "v0_local_attribution_sign_plain_p_redraw_t1p5e7_v64"
    assert [layer for layer, _, _ in alexnet_references.LAYERS] == ["conv3", "conv4", "conv5", "fc6", "fc7"]
