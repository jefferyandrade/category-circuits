"""Fig. 7 and section 2.5, circuit activations: import the analysis code in analysis/circuit_activations/ and
recompute the plotted values in data/figures/values_fig7_running.csv from data/.

Panel D is rebuilt through the code that produced it: per-layer results are collected into the per-layer table
(collect_layer_table.py), turned into plotted points (fig7_values.running), the wiring crossovers are selected from the
paired wiring-minus-prototype tests shipped with Fig. 6 (fig7_values.crossovers), and the ResNet block means are
recomputed (fig7_values.block_means). The intervals of the crossover tests are recomputed from their estimates and
standard errors with wiring_minus_prototypes.paired_interval. The statistics and the command lines run on planted
inputs: shard sums, activation files, run plans and a whole-circuit extraction made here.
"""
import ast
import csv
import importlib
import json
import math
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
CODE = REPO / "analysis" / "circuit_activations"
STATS = REPO / "analysis" / "stats"
TAXONOMY = REPO / "taxonomy" / "inputs" / "category_taxonomy.csv"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

MODULES = ["fc_all_images", "fc_statistics", "fig7_values", "layer_activations", "layer_statistics",
           "partial_tree_shares", "whole_circuits", "wiring_minus_prototypes"]
LEVELS = ("domain", "superordinate", "block50")


def rows(name, delimiter=","):
    with open(name, newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


FIG7 = rows(DATA / "figures" / "values_fig7_running.csv")
FIG6 = rows(DATA / "figures" / "values_fig6_tree.csv")


def module(name):
    loaded = importlib.import_module(name)
    assert Path(loaded.__file__).resolve().parent == CODE, (name, loaded.__file__)
    return loaded


@pytest.mark.parametrize("name", MODULES)
def test_every_module_imports(name):
    """Every module imports with analysis/circuit_activations on the path (no GPU or torch needed)."""
    module(name)


def test_estimators_come_from_analysis_stats():
    """The tree-share, partial-share and wiring estimators are analysis/stats' modules in this repository."""
    assert Path(module("fc_statistics").TS.__file__).resolve() == STATS / "tree_share.py"
    assert Path(module("whole_circuits").TS.__file__).resolve() == STATS / "tree_share.py"
    shares = module("partial_tree_shares")
    assert Path(shares.TI.__file__).resolve() == STATS / "tree_infer.py"
    assert Path(sys.modules[shares.partial_share.__module__].__file__).resolve() == STATS / "tree_contrast.py"
    assert Path(module("wiring_minus_prototypes").TC.__file__).resolve() == STATS / "tree_contrast.py"


def panel(name):
    return [r for r in FIG7 if r["panel"] == name]


def test_panel_d_block_means_are_the_mean_of_each_blocks_layers():
    """Each ResNet curve point of panel D is the mean of its block's plotted layer values, at their mean depth."""
    f = module("fig7_values")
    layers = [r for r in panel("D") if r["quantity"] != "block_mean"]
    shipped = [r for r in panel("D") if r["quantity"] == "block_mean"]
    means = f.block_means(layers, "D")
    assert [(m["network"], m["layer"], m["value"], m["relative_depth"]) for m in means] == \
        [(r["network"], r["layer"], r["value"], r["relative_depth"]) for r in shipped]


def test_panel_d_points_are_circuit_minus_prototypes():
    """Each point is the circuits' tree share minus the prototypes'; AlexNet fc6 and fc7 exactly so, the others
    within the rounding of three-decimal values."""
    for r in panel("D"):
        if r["quantity"] == "block_mean":
            continue
        if r["network"] == "alexnet" and r["layer"] in ("fc6", "fc7"):
            assert r["value"] == str(Decimal(r["circuit"]) - Decimal(r["prototypes"]))
        else:
            assert abs(float(r["value"]) - (float(r["circuit"]) - float(r["prototypes"]))) <= .0011, r


def per_layer_result(r):
    """A per-site result as layer_activations.py writes it, holding the plotted point's values (superordinate level)."""
    level = {"computation": float(r["circuit"]), "image": float(r["prototypes"]), "difference": float(r["value"]),
             "lo": float(r["ci_low"]), "hi": float(r["ci_high"]), "p": float(r["p"])}
    share = lambda value: {"tree": {lv: {"share_adjusted": value} for lv in LEVELS},
                           "confusion_spearman": {"rho": 0.0}}
    return {"schema": "comprep-x-aggregation-v1", "network": r["network"], "population": "shared",
            "site": {"name": r["layer"], "reduction": "gap", "response": "post-ReLU"},
            "n_categories": int(r["n_categories"]),
            "families": {"computation": share(level["computation"]), "image": share(level["image"])},
            "computation_minus_image": {lv: {"difference": level["difference"],
                                             "jackknife": {"ci_low": level["lo"], "ci_high": level["hi"]},
                                             "permutation": {"p_two_sided": level["p"]}} for lv in LEVELS}}


def test_panel_d_rebuilt_from_per_layer_results(tmp_path, monkeypatch):
    """Per-layer results -> collect_layer_table.py -> fig7_values.running, crossovers and block_means reproduce
    every shipped panel D row, column for column. The crossover input is the table of paired wiring-minus-prototype
    tests that wiring_minus_prototypes.py wrote, shipped as Fig. 6's C-test rows."""
    results = tmp_path / "aggregation"
    results.mkdir()
    for r in panel("D"):
        if r["quantity"] == "block_mean" or (r["network"] == "alexnet" and r["layer"] in ("fc6", "fc7")):
            continue
        name = "%s_shared_%s.json" % (r["network"], r["layer"])
        (results / name).write_text(json.dumps(per_layer_result(r)))
    collected = subprocess.run([sys.executable, "-I", str(CODE / "collect_layer_table.py")], cwd=results,
                               capture_output=True, text=True)
    assert collected.returncode == 0, collected.stderr
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "site_tree_shares.tsv").write_text(collected.stdout)
    (inputs / "sign_sharing_four_networks_holm.tsv").write_bytes(
        (DATA / "sign_sharing" / "sign_sharing_four_networks_holm.tsv").read_bytes())
    tests = [r for r in FIG6 if r["panel"] == "C-test"]
    with open(inputs / "values_fig6C_difference.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tests[0]))
        writer.writeheader()
        writer.writerows(tests)
    f = module("fig7_values")
    monkeypatch.setattr(f, "R", inputs)
    rebuilt = f.running() + f.crossovers()
    rebuilt += f.block_means(rebuilt, "D")
    shipped = [r for r in FIG7 if r["panel"] in ("D", "D-crossover")]
    assert len(rebuilt) == len(shipped) == 111
    for got, want in zip(rebuilt, shipped):
        for key, value in want.items():
            assert str(got.get(key, "")) == value, (want["network"], want["layer"], key, got.get(key), value)


@pytest.mark.parametrize("row", [r for r in FIG7 if r["panel"] == "D-crossover" and r["quantity"]] +
                         [r for r in FIG6 if r["panel"] == "C-test"],
                         ids=lambda r: r["network"] + ":" + r["layer"])
def test_crossover_intervals_from_estimate_and_se(row):
    """paired_interval: difference +/- t(.975, n - 1) x jackknife SE, with the asterisk where it excludes zero.
    Deletion replicates with the shipped jackknife variance are rebuilt and passed through the function. SciPy
    versions differ in the eleventh digit of the t quantile (the analyses ran SciPy 1.17.0), hence 1e-9."""
    w = module("wiring_minus_prototypes")
    n, variance = int(row["n_categories"]), float(row["jackknife_variance"])
    a = math.sqrt(variance * n / (2 * (n - 1)))
    left = np.zeros(n)
    left[:2] = a, -a
    result = w.paired_interval(float(row["difference"]), left, np.zeros(n))
    assert math.isclose(result["se"], float(row["se"]), rel_tol=1e-12)
    assert math.isclose(result["se"] ** 2, variance, rel_tol=1e-12)
    for key in ("ci_low", "ci_high"):
        assert math.isclose(result[key], float(row[key]), rel_tol=0, abs_tol=1e-9), key
    assert str(result["asterisk"]) == row["asterisk"]


def test_panel_b_difference_and_interval():
    """Panel B: circuit share minus prototype share, all 1,000 categories, with a normal interval (difference +/-
    1.96 x jackknife SE over 100 groups of ten categories)."""
    b = {r["label"]: r for r in panel("B")}
    for layer in ("fc6", "fc7"):
        left = float(b[layer + " weight tree share"]["value"])
        right = float(b[layer + " image tree share"]["value"])
        diff = b[layer + " weight_minus_image"]
        assert left - right == float(diff["value"])
        assert abs((float(diff["lo"]) + float(diff["hi"])) / 2 - float(diff["value"])) <= 1e-15


def test_panel_c_rows_from_partial_shares(tmp_path):
    """Panel C: fig7_values.figure_rows reads the four partial shares at fc6 and fc7 from partial_tree_shares.py's
    output; each interval is the adjusted share +/- 1.96 x jackknife SE."""
    shipped = panel("C")
    document = {"layers": {}}
    for r in shipped:
        cell = document["layers"].setdefault(r["layer"], {}).setdefault(r["quantity"], {})
        cell["superordinate"] = {"adjusted": float(r["value"]), "lo": float(r["lo"]), "hi": float(r["hi"])}
        assert abs((float(r["lo"]) + float(r["hi"])) / 2 - float(r["value"])) <= 1e-15
        assert (r["ci_low"], r["ci_high"]) == (r["lo"], r["hi"])
    path = tmp_path / "partial_tree_shares.json"
    path.write_text(json.dumps(document))
    f = module("fig7_values")
    got = f.figure_rows(f.Source(str(path)))
    assert [(g["panel"], g["label"], repr(g["value"]), repr(g["lo"]), repr(g["hi"]), g["key"]) for g in got] == \
        [(r["panel"], r["label"], r["value"], r["lo"], r["hi"], r["key"]) for r in shipped]


def planted_tree():
    rng = np.random.default_rng(1)
    categories = list(range(0, 120, 10))
    groups = np.repeat(["a", "b", "c", "d"], 3)
    tree = {"category": categories, "animate": np.where(np.isin(groups, ["a", "b"]), "yes", "no").tolist(),
            "superordinate": groups.tolist(), "block50": [g + str(i % 2) for i, g in enumerate(groups)]}
    centers = {g: rng.normal(size=16) for g in "abcd"}
    structured = np.stack([centers[g] + .3 * rng.normal(size=16) for g in groups])
    return rng, categories, tree, structured, rng.normal(size=(12, 16))


def test_layer_statistics_planted_null_and_signal():
    """layer_statistics.analyze_pair, the per-layer statistic: identical representations give a zero difference, a
    zero-width interval and p = 1; a tree-structured circuit representation against an unstructured one gives an
    interval above zero."""
    statistics = module("layer_statistics")
    rng, categories, tree, structured, unstructured = planted_tree()
    counts = rng.integers(0, 40, size=(12, 1000))
    null = statistics.analyze_pair(unstructured, unstructured, categories, tree, counts,
                                   seed=0, permutations=50)["computation_minus_image"]["superordinate"]
    assert null["difference"] == 0 and null["jackknife"]["ci_low"] == null["jackknife"]["ci_high"] == 0
    assert null["permutation"]["p_two_sided"] == 1
    signal = statistics.analyze_pair(structured, unstructured, categories, tree, counts,
                                     seed=0, permutations=50)["computation_minus_image"]["superordinate"]
    assert signal["difference"] > 0 and signal["jackknife"]["ci_low"] > 0
    assert signal["permutation"]["p_two_sided"] == 1 / 51


def test_fc_paired_difference_planted_null_and_signal():
    """fc_statistics.paired_difference, the panel B statistic, on the same planted representations."""
    resampling = module("fc_statistics")
    rng, categories, tree, structured, unstructured = planted_tree()
    taxonomy = pd.DataFrame(tree)
    confusion = rng.random((12, 12))
    confusion = (confusion + confusion.T) / 2
    np.fill_diagonal(confusion, 0)
    positive = unstructured - unstructured.min() + .1
    null = resampling.paired_difference(positive, positive, np.asarray(categories), taxonomy, confusion,
                                        n_permutations=20)["metrics"]["tree_superordinate"]
    assert null["difference"] == 0 and null["jackknife"]["lo"] == null["jackknife"]["hi"] == 0
    signal = resampling.paired_difference(structured - structured.min() + .1, positive, np.asarray(categories),
                                          taxonomy, confusion, n_permutations=20)["metrics"]["tree_superordinate"]
    assert signal["difference"] == signal["left"] - signal["right"] and signal["jackknife"]["lo"] > 0


def test_partial_share_planted_controls():
    """partial_tree_shares.py's planted-value checks (run before any input is read) pass with analysis/stats'
    estimators."""
    shares = module("partial_tree_shares")
    controls = shares.self_test()
    assert controls["passed"]
    for name in ("cell_only", "negative_cell_only", "cell_beyond_input"):
        assert format(controls["plants"][name]["result"]["observed"], ".3f") == "1.000"


def option_defaults(path):
    defaults = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument" and node.args:
            keywords = {k.arg: k.value for k in node.keywords}
            if "default" in keywords and isinstance(node.args[0], ast.Constant):
                defaults[node.args[0].value] = ast.unparse(keywords["default"])
    return defaults


def test_defaults_are_the_papers_settings():
    layers = option_defaults(CODE / "layer_activations.py")
    assert (layers["--nshards"], layers["--permutations"], layers["--seed"], layers["--population"]) == \
        ("64", "1000", "0", "'shared'")
    shares = option_defaults(CODE / "partial_tree_shares.py")
    assert (shares["--nperm"], shares["--jackknife"]) == ("50", "'ten'")
    statistics = module("layer_statistics")
    assert (statistics.TREE_PERMUTATIONS, statistics.TREE_SEED) == (50, 0)
    assert module("fc_all_images").aggregate.__defaults__ == (None, 1000)
    assert len(module("fc_statistics").jackknife_groups(np.arange(1000))) == 100


def tree_categories(per_block=4):
    """Categories of two animate and two inanimate block50 groups of the category tree."""
    tax = pd.read_csv(TAXONOMY)
    blocks = tax.groupby("block50").category.apply(list)
    animate = tax.set_index("category").animate
    chosen = [b for b in blocks.index if len(blocks[b]) >= per_block and animate[blocks[b][0]] == 1][:2] + \
        [b for b in blocks.index if len(blocks[b]) >= per_block and animate[blocks[b][0]] == 0][:2]
    return sorted(c for b in chosen for c in blocks[b][:per_block]), tax.set_index("category").block50


def test_layer_merge_and_statistics_on_planted_shards(tmp_path):
    """layer_activations.py merge and aggregate-all, on shard sums planted from per-image responses: each circuit's
    activation is the mean of its category means over all 1,000 categories, the prototypes pass the dense check, and
    collect_layer_table.py reads the result. Prototype files that disagree with the shards stop aggregate."""
    layers = module("layer_activations")
    rng = np.random.default_rng(4)
    shared, block = tree_categories()
    counts = rng.integers(2, 5, size=1000)
    circuits = [dict(key="%s:%d" % (arm, c), arm=arm, category=c, walk={"deep_first": True})
                for arm in ("primary", "redraw") for c in shared]
    site = dict(name="fc6", input_module="classifier.1", output_module="classifier.2", mask_modules=["classifier.1"],
                path_modules=["classifier.1", "classifier.2"], kind="plain", output_call=0, reduction="flat",
                response="post-ReLU")
    identity = dict(state_hash="s", axis_hash="a", all_training_membership_sha256="m")
    centers = {b: rng.normal(size=5) for b in set(block[shared])}
    dense = [rng.normal(size=(n, 5)) for n in counts]
    responses = [np.stack([rng.normal(size=(n, 5)) * .3 + centers[block[k["category"]]] for k in circuits])
                 for n in counts]
    nshards, base = 2, tmp_path / "out/run/alexnet"
    for shard in range(nshards):
        sums, seen, dense_sums = np.zeros((3, len(circuits), 5)), np.zeros((1000, 3), dtype=np.int64), np.zeros((3, 1000, 5))
        for k, n in enumerate(counts):
            images = np.arange(n)[np.arange(n) % nshards == shard]
            for half, (selected, denominator) in enumerate(((images, n), (images[images % 2 == 0], (n + 1) // 2),
                                                            (images[images % 2 == 1], n // 2))):
                sums[half] += responses[k][:, selected].sum(1) / denominator
                dense_sums[half, k] += dense[k][selected].sum(0) / denominator
                seen[k, half] = len(selected)
        directory = base / "shards" / ("%05d" % shard)
        directory.mkdir(parents=True)
        np.savez(directory / "fc6.npz", sums=sums, counts=counts, seen=seen, halves=np.asarray("odd_even"),
                 dense=dense_sums, circuit_keys=np.asarray([k["key"] for k in circuits]))
        (directory / "COMPLETE.json").write_text(json.dumps(dict(
            schema=layers.SHARD_SCHEMA, network="alexnet", shard=shard, nshards=nshards, identity=identity,
            populations={"shared": shared}, categories=list(range(1000)), counts=counts.tolist(),
            circuit_keys=[k["key"] for k in circuits], circuits=circuits, sites=[site], batch_size=256,
            half_definition="even/odd image ordinals")))
    prototypes = tmp_path / "dense"
    prototypes.mkdir()
    top1 = np.zeros((1000, 1000), dtype=np.int64)
    for k, n in enumerate(counts):
        np.add.at(top1[k], rng.choice(shared, size=n), 1)
    np.savez(prototypes / "confusion.npz", top1=top1, stratum_order=np.arange(1000), logit_order=np.arange(1000))

    def write_prototypes(shift=0.):
        np.savez(prototypes / "comprep_dense_fc6.npz", class_ids=np.arange(1000), counts=counts,
                 membership_sha256="m", state_sha256="s", axis_hash="a", site_metadata=json.dumps(site),
                 A_counts=(counts + 1) // 2, B_counts=counts // 2,
                 mean=np.stack([x.mean(0) for x in dense]) + shift, A=np.stack([x[0::2].mean(0) for x in dense]) + shift,
                 B=np.stack([x[1::2].mean(0) for x in dense]) + shift)

    write_prototypes()
    out = ["--network", "alexnet", "--out", str(tmp_path / "out")]
    layers.main(["merge", *out, "--nshards", "2", "--prototypes", str(prototypes)])
    assert json.loads((base / "MERGED.json").read_text())["verdict"] == "pass"
    with np.load(base / "shared" / ("primary_%d.npz" % shared[0])) as archive:
        planted = np.mean([r[0].mean(0) for r in responses], axis=0)
        assert np.allclose(archive["fc6__mean"], planted, rtol=1e-12, atol=1e-12)
    layers.main(["aggregate-all", *out, "--taxonomy", str(TAXONOMY), "--permutations", "20"])
    result = json.loads((tmp_path / "out/run/aggregation/alexnet_shared_fc6.json").read_text())
    assert result["schema"] == layers.RESULT_SCHEMA and result["n_categories"] == len(shared)
    assert result["computation_minus_image"]["superordinate"]["permutation"]["permutations"] == 20
    collected = subprocess.run([sys.executable, "-I", str(CODE / "collect_layer_table.py")],
                               cwd=tmp_path / "out/run/aggregation", capture_output=True, text=True)
    assert collected.returncode == 0 and collected.stdout.splitlines()[1].startswith("alexnet\tshared\tfc6\t16\t")
    finished = (base / "MERGED.json").read_bytes()
    write_prototypes(shift=1e-3)
    layers.main(["merge", *out, "--nshards", "2", "--prototypes", str(prototypes)])
    assert (base / "MERGED.json").read_bytes() == finished
    import shutil
    shifted = tmp_path / "shifted"
    shutil.copytree(base / "shards", shifted / "run/alexnet/shards")
    shifted_out = ["--network", "alexnet", "--out", str(shifted)]
    layers.main(["merge", *shifted_out, "--nshards", "2", "--prototypes", str(prototypes)])
    assert json.loads((shifted / "run/alexnet/MERGED.json").read_text())["verdict"] == "fail"
    with pytest.raises(ValueError, match="differ from the prototype files"):
        layers.main(["aggregate-all", *shifted_out, "--taxonomy", str(TAXONOMY), "--permutations", "20"])


def planted_activation_files(directory, categories, run, rng, image=None):
    """An extraction folder in the layout fc_all_images.load_run reads."""
    directory.mkdir(parents=True)
    tax = pd.read_csv(TAXONOMY).set_index("category")
    images = np.full(len(categories), 9)
    image = image or {}
    for c in categories:
        record = {"category": np.asarray(c), "categories": np.asarray(categories)}
        for layer in ("fc6", "fc7"):
            signal = np.abs(np.sin(np.arange(6) * (1 + hash(tax.superordinate[c]) % 7)))
            halves = {h: np.maximum(0, signal + rng.normal(size=6) * .3) for h in "AB"}
            image.setdefault((c, layer), {h: np.maximum(0, .5 * signal + rng.normal(size=6)) for h in "AB"})
            prototype = dict(image[(c, layer)], mean=(5 * image[(c, layer)]["A"] + 4 * image[(c, layer)]["B"]) / 9)
            for h, value in dict(halves, mean=(halves["A"] + halves["B"]) / 2).items():
                record["computation__%s__%s" % (layer, h)] = value
                record["image__%s__%s" % (layer, h)] = prototype[h]
        np.savez(directory / ("category_%d.npz" % c), **record)
    confusion = np.zeros((len(categories), 1000), dtype=np.int64)
    for i in range(len(categories)):
        np.add.at(confusion[i], np.random.default_rng(i).choice(categories, size=9), 1)
    np.save(directory / "confusion_counts.npy", confusion)
    manifest = dict(schema="comprep-n-v66-1", source_schema="comprep-all-v66-1", categories=categories,
                    extracted_categories=categories, population="shared100", input_identity={"bank": "planted"},
                    panel_sha256="0", probe=False, slice_counts=[9] * len(categories), image_counts=images.tolist(),
                    label="P", arm_kind="weight", run=run)
    (directory / "EXTRACT.json").write_text(json.dumps(manifest))
    (directory / "COMPLETE.json").write_text(json.dumps(dict(schema="comprep-n-v66-1", categories=categories)))
    return image


def test_fc_all_images_on_planted_activation_files(tmp_path):
    """fc_all_images.py on planted activation files of the shared 100: the circuits' activations against the
    prototypes, with the single-category jackknife and the category-swap test. A P-redraw run must carry the same
    prototypes and be labelled as the second run."""
    fc = module("fc_all_images")
    rng = np.random.default_rng(6)
    categories = sorted(np.random.default_rng(2).choice(1000, size=100, replace=False).tolist())
    image = planted_activation_files(tmp_path / "first", categories, "first", rng)
    planted_activation_files(tmp_path / "second", categories, "second", rng, image)
    planted_activation_files(tmp_path / "other", categories, "second", rng)
    result = fc.aggregate(tmp_path / "first", TAXONOMY, n_permutations=20)
    for layer in ("fc6", "fc7"):
        cell = result["differences"][layer]["weight_minus_image"]["metrics"]["tree_superordinate"]
        assert cell["difference"] == cell["left"] - cell["right"]
        assert cell["jackknife"]["n_replicates"] == 100 and cell["category_swap"]["n_permutations"] == 20
        assert result["differences"][layer]["weight_minus_weight_second"]["status"] == "undefined"
    json.dumps(module("fc_statistics").json_clean(result), allow_nan=False)
    with pytest.raises(ValueError, match="extraction role"):
        fc.aggregate(tmp_path / "second", TAXONOMY)
    with pytest.raises(ValueError, match="paired dense image representations differ"):
        fc.aggregate(tmp_path / "first", TAXONOMY, tmp_path / "other")


def test_wiring_minus_prototypes_on_planted_plan(tmp_path):
    """wiring_minus_prototypes.py from a run plan, the tree_contrast inputs it names and a panel table: the
    difference is the panel's W minus A share, with the paired jackknife interval."""
    wiring = module("wiring_minus_prototypes")
    rng = np.random.default_rng(8)
    categories = np.asarray(sorted(rng.choice(1000, size=100, replace=False)))
    same = (lambda codes: codes[:, None] == codes[None, :])(
        pd.factorize(pd.read_csv(TAXONOMY).set_index("category").superordinate[categories])[0])
    intersections = rng.integers(100, 500, size=(100, 100))
    intersections = np.minimum(intersections, intersections.T) + same * 150
    np.fill_diagonal(intersections, rng.integers(2000, 4000, size=100))
    np.savez(tmp_path / "gram_fc6.npz", categories=categories, I=intersections)
    np.savez(tmp_path / "pairs_fc6.npz", categories=categories, J0_magnitude=np.full((100, 100), .01))
    activation = rng.normal(size=(100, 100)) * .1 + same * .3
    np.savez(tmp_path / "matrices_classifier.0.npz", class_ids=categories, **{"S|cv:all": (activation + activation.T) / 2})
    (tmp_path / "sites.json").write_text(json.dumps({"fc6": "classifier.0"}))
    with open(tmp_path / "tree.csv", "w", newline="") as handle:
        handle.write("quantity,level,lens,perm_mean\nplain,superordinate,W,0.039\nplain,superordinate,A,0.038\n")
    argv = ["--network", "vgg11", "--layer", "fc6", "--gram-dir", str(tmp_path), "--pairs-dir", str(tmp_path),
            "--trained-dir", str(tmp_path), "--site-map", str(tmp_path / "sites.json"), "--taxonomy", str(TAXONOMY)]
    (tmp_path / "plan.json").write_text(json.dumps({"tasks": [
        {"id": "tree_vgg11_fc6_train", "argv": argv, "output": str(tmp_path / "tree.csv")}]}))
    (tmp_path / "resnet50_plan.json").write_text(json.dumps({"tasks": []}))
    tasks = wiring.task_plan(tmp_path / "plan.json", tmp_path / "resnet50_plan.json")
    shares = wiring.compute_task(tasks[("vgg11", "fc6")])
    with open(tmp_path / "panel.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["panel", "network", "layer", "selector", "value", "source"])
        writer.writeheader()
        for lens, name in (("W", "recomputed_wiring"), ("A", "recomputed_prototypes")):
            writer.writerow(dict(panel="C", network="vgg11", layer="fc6", selector=json.dumps({"lens": lens}),
                                 value=repr(shares[name]), source="planted"))
    wiring.main(["--plan", str(tmp_path / "plan.json"), "--resnet50-plan", str(tmp_path / "resnet50_plan.json"),
                 "--panel", str(tmp_path / "panel.csv"), "--out", str(tmp_path / "out")])
    row = rows(tmp_path / "out" / "values_fig6C_difference.csv")[0]
    assert float(row["difference"]) == shares["recomputed_wiring"] - shares["recomputed_prototypes"]
    assert (row["n_categories"], row["lens"], row["activation_site"]) == ("100", "W-A", "classifier.0")
    assert math.isclose(float(row["ci_high"]) - float(row["difference"]), float(row["difference"]) - float(row["ci_low"]))


@pytest.mark.filterwarnings("ignore:.*encountered in matmul:RuntimeWarning")  # stale BLAS floating-point flags
def test_whole_circuits_on_planted_extraction(tmp_path):
    """whole_circuits.py on a planted 32-image extraction: tree shares for every representation; the dense
    all-image reference, the same for every category, has none."""
    whole = module("whole_circuits")
    rng = np.random.default_rng(12)
    categories = sorted(np.random.default_rng(3).choice(1000, size=100, replace=False).tolist())
    tax = pd.read_csv(TAXONOMY).set_index("category")
    names = list(whole.REQUIRED_VARIANTS) + list(whole.REFERENCES)
    dense = {layer: {h: rng.normal(size=6) for h in "AB"} for layer in whole.LAYERS}
    trajectory = {}
    for c in categories:
        signal = np.sin(np.arange(6) * (1 + hash(tax.superordinate[c]) % 7))
        record = {"category": np.asarray(c), "categories": np.asarray(categories)}
        for layer in whole.LAYERS:
            values = {name: {h: signal * rng.random() + rng.normal(size=6) for h in "AB"} for name in names}
            for name in ("dense_all_image", whole.FC7_WALK[0]) + ((("fc7_only",) + whole.FC7_WALK) if layer == "fc6" else ()):
                values[name] = dense[layer]
            values[whole.FC7_WALK[-1]] = values["fc7_only"]
            values[whole.LATER_WALK[-1]] = values["full"]
            for name, halves in values.items():
                for h, value in dict(halves, mean=(halves["A"] + halves["B"]) / 2).items():
                    record["%s__%s__%s" % (name, layer, h)] = value
        np.savez(tmp_path / ("category_%d.npz" % c), **record)
        trajectory[str(c)] = [dict(name=name, accepted_steps=index, retained_fraction=1 - index / 10)
                              for index, name in enumerate(whole.FC7_WALK)] + [dict(name=n) for n in whole.LATER_WALK]
    (tmp_path / "EXTRACT.json").write_text(json.dumps(dict(
        schema="comprep-v66-1", categories=categories, extracted_categories=categories, slice_categories=categories,
        slice_images_per_category=32, image_images_per_category=256, references=list(whole.REFERENCES),
        layers=list(whole.LAYERS), endpoint_equal={str(c): True for c in categories},
        trajectory_by_category=trajectory, probe=False, variants=list(whole.REQUIRED_VARIANTS))))
    (tmp_path / "COMPLETE.json").write_text(json.dumps(dict(schema="comprep-v66-1", categories=categories)))
    result = whole.aggregate(tmp_path, TAXONOMY)
    for layer in whole.LAYERS:
        assert result["layers"][layer]["dense_all_image"]["tree"]["superordinate"]["status"] == "undefined"
        for name in ("full", "image"):
            tree = result["layers"][layer][name]["tree"]
            assert set(tree) == set(LEVELS) and math.isfinite(tree["superordinate"]["share_adjusted"])
