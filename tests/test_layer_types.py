"""Section 2.5, the layer-type tests (width, pooling, the skip path and the ReLU): check the printed numbers against
data/layer_types/, recompute every interval from the category deletions, and run analysis/circuit_activations/
layer_type_tests.py on planted inputs.

The table's values come from layer_type_tests.py run on the saved activations, which are not in this repository; here
the printed values are read from the table, and the standard errors and intervals are recomputed from the deletions.
"""
import csv
import hashlib
import json
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "layer_types"
CODE = REPO / "analysis" / "circuit_activations"
TAXONOMY = REPO / "taxonomy" / "inputs" / "category_taxonomy.csv"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import layer_type_tests as L  # noqa: E402
from layer_activations import read_taxonomy  # noqa: E402

# The macro of each printed value: the stem names the difference, Lo and Hi its interval.
MACROS = {("width", "vgg11"): "ResEWidthVggFcSixLead",
          ("pooling", "vgg11"): "ResEPoolVggConvFiveTwoLocal",
          ("skip", "resnet18"): "ResESkipResNetEighteenLast",
          ("before_relu", "alexnet"): "ResEPreReluAlexFcSix",
          ("after_relu", "alexnet"): "ResEPostReluAlexFcSixShared",
          ("before_relu", "vgg11"): "ResEPreReluVggFcSix"}


def rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


TABLE = rows(DATA / "layer_type_tests.tsv")
DELETIONS = rows(DATA / "category_deletions.tsv")
with open(REPO / "data" / "printed_numbers.tsv", newline="") as handle:
    PRINTED = {r["macro"]: r["printed"] for r in csv.DictReader(handle, delimiter="\t")}


def same_as_printed(value, text):
    """True when value, rounded half up to the printed number of decimals, reads as the printed text."""
    places = len(text.split(".")[1]) if "." in text else 0
    rounded = Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    return rounded == Decimal(text.lstrip("+")) and (not text.startswith("+") or rounded > 0)


def test_printed_numbers_are_the_tables_rows():
    """All 18 printed numbers of the layer-type tests: each test's difference and interval, in the order of section 2.5."""
    assert [(r["test"], r["network"]) for r in TABLE] == list(L.TEST_ORDER) == list(MACROS)
    group = [m for m, r in PRINTED.items() if m.startswith(tuple(MACROS.values()))]
    assert len(group) == 18
    for r in TABLE:
        stem = MACROS[(r["test"], r["network"])]
        for suffix, column in (("", "difference"), ("Lo", "ci_low"), ("Hi", "ci_high")):
            assert same_as_printed(r[column], PRINTED[stem + suffix]), (stem + suffix, r[column])


def test_intervals_from_the_category_deletions():
    """Each interval, recomputed from the 100 delete-one values: the difference plus or minus 1.96 jackknife SEs
    for AlexNet's two readings and the width contrast, t(.975, 99) SEs for the others."""
    shared = list(L.shared100())
    t99 = L.S.student_t_critical(99)
    for r in TABLE:
        deleted = [d for d in DELETIONS if (d["test"], d["network"]) == (r["test"], r["network"])]
        assert [int(d["deleted_category"]) for d in deleted] == shared and int(r["n_categories"]) == 100
        critical = float(r["critical_value"])
        if r["network"] == "alexnet" or r["test"] == "width":
            assert critical == 1.96
        else:
            assert abs(critical - t99) < 1e-12
        interval = L.jackknife_interval(float(r["difference"]), [float(d["difference"]) for d in deleted], critical)
        for column in ("se", "ci_low", "ci_high"):
            assert abs(interval[column] - float(r[column])) < 1e-12, (r["test"], r["network"], column)
        if r["circuit"]:
            assert abs(float(r["circuit"]) - float(r["prototypes"]) - float(r["difference"])) < 1e-15


def test_images_per_category_are_the_methods():
    """The captures' images per category (Methods: 64 for pooling and VGG-11 before the ReLU, 16 for the skip path);
    the other readings use every training image."""
    images = {(r["test"], r["network"]): r["images_per_category"] for r in TABLE}
    assert images == {("width", "vgg11"): "all", ("pooling", "vgg11"): "64", ("skip", "resnet18"): "16",
                      ("before_relu", "alexnet"): "all", ("after_relu", "alexnet"): "all",
                      ("before_relu", "vgg11"): "64"}
    assert L.CAPTURE_IMAGES == {"pooling": 64, "skip": 16, "relu-vgg11": 64}
    arguments = L.command_line().parse_args(["relu-alexnet", "--pool5", "p", "--activations", "a", "--out", "o"])
    assert arguments.price == "v66" and arguments.taxonomy == TAXONOMY


def test_width_units_are_one_fixed_draw():
    """fc6's 512 units: unique, sorted, within fc6's 4,096, and always the same draw."""
    units = L.width_units()
    assert len(units) == 512 and len(set(units.tolist())) == 512 and np.all(np.diff(units) > 0)
    assert 0 <= units[0] and units[-1] < 4096
    assert hashlib.sha256(units.astype("<i8").tobytes()).hexdigest()[:16] == "4699c19324cce622"


def planted_categories(groups=3, per_group=4):
    """Categories of the first superordinate groups of the category tree with at least per_group categories."""
    tax = pd.read_csv(TAXONOMY)
    members = tax.groupby("superordinate").category.apply(sorted)
    chosen = [g for g in members.index if len(members[g]) >= per_group][:groups]
    categories = sorted(c for g in chosen for c in members[g][:per_group])
    return categories, tax.set_index("category").superordinate[categories].to_numpy()


def structured(rng, groups, features, noise=.3):
    centers = {g: rng.normal(size=features) for g in sorted(set(groups))}
    return np.stack([centers[g] + noise * rng.normal(size=features) for g in groups])


def test_statistics_on_planted_values():
    """Identical circuits and prototypes differ by exactly zero with a zero-width interval; circuit activations with
    superordinate structure exceed unstructured prototypes with an interval above zero. The similarities over
    positions reduce to the plain one when every position is the same, and the width contrast is the difference of
    its two readings."""
    rng = np.random.default_rng(3)
    categories, groups = planted_categories()
    tree = read_taxonomy(TAXONOMY)
    noise = rng.normal(size=(len(categories), 24))
    same = L.circuits_minus_prototypes(L.pearson(noise), L.pearson(noise), categories, tree, 1.96)
    assert same["difference"] == 0 and same["se"] == 0 and same["ci_low"] == same["ci_high"] == 0
    signal = L.circuits_minus_prototypes(L.pearson(structured(rng, groups, 24)), L.pearson(noise), categories, tree,
                                         L.t_critical(len(categories)))
    assert signal["difference"] > 0 and signal["ci_low"] > 0 and len(signal["deleted_differences"]) == len(categories)
    one_position = structured(rng, groups, 8)
    spatial = np.repeat(one_position[:, :, None, None], 3, axis=2).repeat(2, axis=3)
    assert np.allclose(L.mean_pearson_over_positions(spatial), L.pearson(one_position), atol=1e-12)
    varied = spatial + rng.normal(size=spatial.shape)
    assert np.allclose(L.pearson_of_channel_means(varied), L.pearson(varied.mean(axis=(2, 3))), atol=1e-12)
    fc6 = (structured(rng, groups, 600), rng.normal(size=(len(categories), 600)))
    conv = (structured(rng, groups, 16, noise=1.), rng.normal(size=(len(categories), 16)))
    contrast = L.width_contrast(fc6, conv, categories, tree)
    units = L.width_units(600)
    narrow = L.circuits_minus_prototypes(L.pearson(fc6[0][:, units]), L.pearson(fc6[1][:, units]), categories, tree,
                                         1.96)
    assert contrast["fc6_difference"] == narrow["difference"]
    assert contrast["difference"] == narrow["difference"] - contrast["conv5_2_difference"]


def test_before_relu_means_are_the_mean_preactivation():
    """fc6's circuit activations and prototypes before the ReLU, from the category means of fc6's input, equal the
    mean pre-activation over planted images: over every category's images with equal category weights for a circuit
    (its own mask, dense bias), over the category's own images for its prototype."""
    rng = np.random.default_rng(5)
    weight, bias = rng.normal(size=(5, 7)), rng.normal(size=5)
    images = [rng.normal(size=(n, 7)) for n in (3, 6, 2, 5)]
    masks = {c: rng.random((5, 7)) < .4 for c in range(4)}
    means = np.stack([x.mean(axis=0) for x in images])
    circuits, prototypes = L.before_relu_means(means, weight, bias, [1, 3], masks.get, block_rows=2)
    for i, c in enumerate([1, 3]):
        each_category = [(x @ (weight * masks[c]).T + bias).mean(axis=0) for x in images]
        assert np.allclose(circuits[i], np.mean(each_category, axis=0), atol=1e-12)
        assert np.allclose(prototypes[i], (images[c] @ weight.T + bias).mean(axis=0), atol=1e-12)
    with pytest.raises(ValueError, match="boolean"):
        L.before_relu_means(means, weight, bias, [1], lambda c: masks[c].astype(int))


def write_capture(folder, k, site, field, categories, circuits, prototypes):
    """A capture folder in the layout capture_means reads."""
    index = {"additive": False, "k": k, "category_ids": [str(c) for c in categories], "subsets": {"K%d" % k: {}}}
    for arm, values in (("comp", circuits), ("image", prototypes)):
        relative = Path("arrays") / ("K%d" % k) / site / arm / (field + ".npy")
        (folder / relative).parent.mkdir(parents=True, exist_ok=True)
        np.save(folder / relative, values)
        index["subsets"]["K%d" % k].setdefault(site, {}).setdefault(arm, {})[field] = str(relative)
    path = folder / "means_index.json"
    if path.exists():
        old = json.loads(path.read_text())
        for name, arms in index["subsets"]["K%d" % k].items():
            old["subsets"]["K%d" % k][name] = arms
        index = old
    path.write_text(json.dumps(index))


def write_fig7_folder(folder, categories, sites):
    """A network folder in the layout layer_activations.py merge writes: sites maps a site to (circuits, prototypes)."""
    (folder / "cache" / "shared").mkdir(parents=True)
    (folder / "shared").mkdir()
    (folder / "cache" / "COMPLETE.json").write_text(json.dumps({"populations": {"shared": categories}}))
    np.savez(folder / "cache" / "shared" / "dense.npz", categories=np.asarray(categories),
             **{site + "__mean": values[1] for site, values in sites.items()})
    for i, c in enumerate(categories):
        np.savez(folder / "shared" / ("primary_%d.npz" % c), category=np.asarray(c), categories=np.asarray(categories),
                 **{site + "__mean": values[0][i] for site, values in sites.items()})


def test_command_line_on_planted_captures(tmp_path, monkeypatch):
    """width, pooling, skip and relu-vgg11 on planted folders: each command's row is its statistic on the planted
    arrays, a capture with another number of images per category is refused, and table writes every test's row in
    order with its deletions."""
    rng = np.random.default_rng(7)
    categories, groups = planted_categories()
    monkeypatch.setattr(L, "shared100", lambda: tuple(categories))
    tree = read_taxonomy(TAXONOMY)
    n = len(categories)
    conv = (np.abs(structured(rng, groups, 4 * 9)).reshape(n, 4, 3, 3), np.abs(rng.normal(size=(n, 4, 3, 3))))
    fc6 = (structured(rng, groups, 6)[:, :, None], rng.normal(size=(n, 6, 1)))
    block = (np.abs(rng.normal(size=(n, 4, 2, 2))), np.abs(structured(rng, groups, 16)).reshape(n, 4, 2, 2))
    write_capture(tmp_path / "c64", 64, "vgg11.conv5_2", "post", categories, *conv)
    write_capture(tmp_path / "c64", 64, "vgg11.fc6", "pre", categories, *fc6)
    write_capture(tmp_path / "c16", 16, "resnet18.stage4.block1.conv2", "alpha0", categories, *block)
    write_fig7_folder(tmp_path / "vgg11", categories,
                      {"fc6": (structured(rng, groups, 600), rng.normal(size=(n, 600))),
                       "conv5_2": (rng.normal(size=(n, 8)), structured(rng, groups, 8))})
    out = tmp_path / "out"
    out.mkdir()
    for command, flag, folder in (("pooling", "--capture", "c64"), ("relu-vgg11", "--capture", "c64"),
                                  ("skip", "--capture", "c16"), ("width", "--activations", "vgg11")):
        assert L.main([command, flag, str(tmp_path / folder), "--out", str(out / (command + ".json"))]) == 0
    results = {name: json.loads((out / (name + ".json")).read_text())["rows"][0]
               for name in ("pooling", "relu-vgg11", "skip", "width")}
    critical = L.t_critical(n)
    expected = {"pooling": L.circuits_minus_prototypes(L.mean_pearson_over_positions(conv[0]),
                                                       L.mean_pearson_over_positions(conv[1]), categories, tree,
                                                       critical),
                "skip": L.circuits_minus_prototypes(L.pearson_of_channel_means(block[0]),
                                                    L.pearson_of_channel_means(block[1]), categories, tree, critical)}
    for name, value in expected.items():
        for key in ("circuit", "prototypes", "difference", "se", "ci_low", "ci_high"):
            assert abs(results[name][key] - value[key]) < 1e-12, (name, key)
    assert results["pooling"]["images_per_category"] == "64" and results["skip"]["images_per_category"] == "16"
    assert abs(results["relu-vgg11"]["circuit"] - L.tree_share(L.pearson(fc6[0]), categories, tree)) < 1e-12
    assert results["width"]["circuit"] is None and results["width"]["critical_value"] == 1.96
    with pytest.raises(ValueError, match="64 images per category"):
        L.main(["pooling", "--capture", str(tmp_path / "c16"), "--out", str(out / "refused.json")])
    with pytest.raises(FileExistsError):
        L.main(["skip", "--capture", str(tmp_path / "c16"), "--out", str(out / "skip.json")])
    alexnet = {"rows": [dict(results["skip"], test=test, network="alexnet", layer="fc6")
                        for test in ("before_relu", "after_relu")]}
    (out / "relu_alexnet.json").write_text(json.dumps(alexnet))
    names = ["width", "pooling", "skip", "relu_alexnet", "relu-vgg11"]
    L.main(["table", *[str(out / (name + ".json")) for name in names], "--out", str(out / "tables")])
    table = rows(out / "tables" / "layer_type_tests.tsv")
    assert [(r["test"], r["network"]) for r in table] == list(L.TEST_ORDER)
    assert list(table[0]) == list(L.TABLE_COLUMNS) and table[0]["circuit"] == ""
    assert len(rows(out / "tables" / "category_deletions.tsv")) == 6 * n
    with pytest.raises(ValueError, match="every test once"):
        L.main(["table", *[str(out / (name + ".json")) for name in names[:4]], "--out", str(out / "other")])


def planted_activation_run(directory, categories, rng):
    """AlexNet activation files of the shared categories in the layout fc_all_images.load_run reads."""
    directory.mkdir()
    for c in categories:
        record = {"category": np.asarray(c), "categories": np.asarray(categories)}
        for layer in ("fc6", "fc7"):
            for family in ("computation", "image"):
                halves = {h: np.abs(rng.normal(size=5)) for h in "AB"}
                mean = (halves["A"] + halves["B"]) / 2
                for half, value in dict(halves, mean=mean).items():
                    record["%s__%s__%s" % (family, layer, half)] = value
        np.savez(directory / ("category_%d.npz" % c), **record)
    confusion = np.zeros((len(categories), 1000), dtype=np.int64)
    confusion[np.arange(len(categories)), categories] = 4
    np.save(directory / "confusion_counts.npy", confusion)
    manifest = dict(schema=L.fc_all_images.SCHEMA, source_schema=L.fc_all_images.SOURCE_SCHEMA, categories=categories,
                    extracted_categories=categories, population="shared100", input_identity={"bank": "planted"},
                    panel_sha256="0", probe=False, slice_counts=[4] * len(categories),
                    image_counts=[4] * len(categories), label="P", arm_kind="weight", run="first")
    (directory / "EXTRACT.json").write_text(json.dumps(manifest))
    (directory / "COMPLETE.json").write_text(json.dumps(dict(schema=L.fc_all_images.SCHEMA, categories=categories)))


def test_relu_alexnet_on_planted_inputs(tmp_path, monkeypatch):
    """relu-alexnet on the shared categories, with planted category means of fc6's input, fc6 weights and masks in
    place of the checkpoint and the circuit library: the row before the ReLU scores before_relu_means' output, the row
    after it the activation files' fc6 means, both with 1.96 SEs."""
    rng = np.random.default_rng(9)
    categories = list(L.shared100())
    weight, bias = rng.normal(size=(5, 8)), rng.normal(size=5)
    masks = {c: np.random.default_rng(c).random((5, 8)) < .5 for c in categories}
    monkeypatch.setattr(L, "alexnet_fc6_weights", lambda checkpoint: (weight, bias))
    monkeypatch.setattr(L, "alexnet_fc6_masks", lambda config, price: masks.get)
    pool5 = np.abs(rng.normal(size=(1000, 8)))
    np.savez(tmp_path / "pool5.npz", category_mean=pool5, image_categories=np.arange(1000))
    planted_activation_run(tmp_path / "run", categories, rng)
    out = tmp_path / "relu.json"
    L.main(["relu-alexnet", "--pool5", str(tmp_path / "pool5.npz"), "--activations", str(tmp_path / "run"),
            "--out", str(out)])
    before, after = json.loads(out.read_text())["rows"]
    tree = read_taxonomy(TAXONOMY)
    circuits, prototypes = L.before_relu_means(pool5, weight, bias, categories, masks.get)
    assert (before["test"], after["test"]) == ("before_relu", "after_relu")
    assert abs(before["circuit"] - L.tree_share(L.pearson(circuits), categories, tree)) < 1e-12
    assert abs(before["prototypes"] - L.tree_share(L.pearson(prototypes), categories, tree)) < 1e-12
    image = np.stack([np.load(tmp_path / "run" / ("category_%d.npz" % c))["image__fc6__mean"] for c in categories])
    assert abs(after["prototypes"] - L.tree_share(L.pearson(image), categories, tree)) < 1e-12
    assert before["critical_value"] == after["critical_value"] == 1.96 and len(after["deleted_differences"]) == 100
