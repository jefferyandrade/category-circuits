"""AlexNet fc6 and fc7: the circuits' activations against the category prototypes, from every training image (Fig. 7B;
the AlexNet fc6 and fc7 points of Fig. 7D).

For each layer, the category-tree share (node- and permutation-adjusted, at the domain, superordinate and block50
levels) and the confusion Spearman correlation of each representation: the circuits' activations (weight), the
P-redraw circuits' activations (weight_second) and the dense network's category means, the prototypes (image). Each
paired difference (weight_minus_image, weight_minus_weight_second) gets a delete-category jackknife interval
(difference +/- 1.96 SE; single categories at the shared 100, 100 fixed random groups of ten at all 1,000) and a
permutation test of 1,000 paired category swaps, seed 0 (fc_statistics.paired_difference).

With --population held_out, the same comparison of the circuits' activations with the prototypes on the 900 of the
1,000 categories that are not among the shared 100 (section 2.5). The jackknife then deletes 90 fixed random groups of
ten of them (seed 0), and the output is a table with one row per layer at the superordinate level
(data/robustness/populations.tsv).

Usage: fc_all_images.py --weight RUN [--weight-second RUN] --taxonomy category_taxonomy.csv --out OUT.json
       fc_all_images.py --weight RUN --taxonomy category_taxonomy.csv --population held_out --out OUT.tsv
Panel B is --weight on the circuits of all 1,000 categories; the AlexNet fc6 and fc7 points of panel D are --weight on
the shared 100, with --weight-second the P-redraw circuits; the held-out categories are drawn from the circuits of all
1,000 categories. Each RUN folder holds one extraction's activation files:
EXTRACT.json (its manifest: categories, image counts, input identity), COMPLETE.json (the categories written),
category_<c>.npz (category c's circuit activations, computation, and its prototype, image, at fc6 and fc7, each as the
mean and the means of its two halves) and confusion_counts.npy (the dense network's top-1 counts over 1,000 outputs).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import fc_statistics as S

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from populations import shared100  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import check_outputs, write_file

# Schema labels of the extraction's EXTRACT.json and COMPLETE.json.
SCHEMA = "comprep-n-v66-1"
SOURCE_SCHEMA = "comprep-all-v66-1"
LAYERS = ("fc6", "fc7")
ARMS = ("weight", "weight_second")
HALVES = ("mean", "A", "B")
POPULATIONS = ("held_out",)
TABLE_COLUMNS = ("population", "n_categories", "layer", "circuit", "prototypes", "difference", "se", "ci_low",
                 "ci_high", "p", "n_jackknife_groups", "n_category_swaps")


def _categories(values, name):
    result = S.category_ids(values, name)
    if any(value >= 1000 for value in result):
        raise ValueError(name + " must use AlexNet's 1,000 output category indices")
    return result


def _counts(values, categories, name):
    if isinstance(values, dict):
        values = [values.get(str(category), values.get(category)) for category in categories]
    if not isinstance(values, list) or len(values) != len(categories):
        raise ValueError(name + " must state the count for every category")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in values):
        raise ValueError(name + " must contain positive integers")
    return np.asarray(values, dtype=np.int64)


def confusion_pairs(counts, categories, image_counts):
    """Mean of unconditional reciprocal top-1 rates, including zero pairs."""
    counts = np.asarray(counts)
    image_counts = np.asarray(image_counts)
    n = len(categories)
    if (counts.shape != (n, 1000) or not np.issubdtype(counts.dtype, np.integer)
            or np.issubdtype(counts.dtype, np.bool_) or (counts < 0).any()):
        raise ValueError("confusion counts must be nonnegative integer rows over 1,000 outputs")
    if image_counts.shape != (n,) or (image_counts <= 0).any():
        raise ValueError("image counts must align with confusion categories")
    if not np.array_equal(counts.astype(np.int64).sum(1), image_counts):
        raise ValueError("confusion row totals differ from panel image counts")
    rates = counts[:, categories].astype(np.float64) / image_counts[:, None]
    pair_rate = (rates + rates.T) / 2.
    np.fill_diagonal(pair_rate, 0.)
    return {"pair_rate": pair_rate, "directed_shared_rates": rates,
            "images_per_category": image_counts, "n_unordered_pairs": n * (n - 1) // 2,
            "normalization": "mean of the two unconditional 1,000-way top-1 rates; zero pairs included; diagonal excluded"}


def _image_halves(manifest, categories, image_counts):
    representation_counts = _counts(
        manifest.get("image_representation_counts", manifest["image_counts"]),
        categories, "image_representation_counts")
    if (representation_counts > image_counts).any() or (representation_counts < 2).any():
        raise ValueError("image representation counts must be between 2 and the image counts")
    expected = np.column_stack(((representation_counts + 1) // 2,
                                representation_counts // 2))
    supplied = manifest.get("image_half_counts")
    if supplied is not None:
        actual = np.asarray(supplied)
        if (actual.shape != expected.shape or actual.dtype.kind not in "iu"
                or not np.array_equal(actual, expected)):
            raise ValueError("image_half_counts differ from odd/even representation counts")
    return expected


def load_run(run):
    """Read one extraction: (manifest, categories, arrays[layer][computation or image][half], confusion).

    The categories have different image counts. Image halves reconstruct their mean using their own counts. Circuit
    halves each average category means with equal category weights; when the categories have different odd counts, no
    single pair of scalar weights reconstructs the overall circuit mean from its two stored halves.
    """
    run = Path(run)
    manifest = json.loads((run / "EXTRACT.json").read_text())
    complete = json.loads((run / "COMPLETE.json").read_text())
    if (manifest.get("schema") != SCHEMA or complete.get("schema") != SCHEMA
            or manifest.get("source_schema") != SOURCE_SCHEMA):
        raise ValueError("unsupported extraction schema")
    categories = _categories(manifest.get("categories"), "categories")
    extracted = _categories(manifest.get("extracted_categories"), "extracted_categories")
    if len(categories) not in (100, 1000):
        raise ValueError("declared population must be the shared 100 or all 1,000")
    if len(categories) == 1000 and categories != list(range(1000)):
        raise ValueError("all-category population must contain every AlexNet output id")
    if manifest.get("population") != ("shared100" if len(categories) == 100 else "all1000"):
        raise ValueError("population label differs from declared category count")
    if not isinstance(manifest.get("input_identity"), dict) or not manifest["input_identity"]:
        raise ValueError("extraction requires a nonempty input_identity")
    if not isinstance(manifest.get("panel_sha256"), str) or not manifest["panel_sha256"]:
        raise ValueError("extraction requires panel_sha256")
    if not isinstance(manifest.get("probe"), bool):
        raise ValueError("extraction must state whether it is partial (probe)")
    if not set(extracted).issubset(categories) or complete.get("categories") != extracted:
        raise ValueError("completed categories differ from extracted categories")
    if not manifest["probe"] and extracted != categories:
        raise ValueError("a complete extraction is missing categories or out of order")
    slices = _counts(manifest.get("slice_counts"), categories, "slice_counts")
    images = _counts(manifest.get("image_counts"), categories, "image_counts")
    if (slices > images).any():
        raise ValueError("slice counts exceed image counts")
    image_halves = _image_halves(manifest, categories, images)
    positions = {category: index for index, category in enumerate(categories)}
    rows = {layer: {family: {half: [] for half in HALVES}
                    for family in ("computation", "image")} for layer in LAYERS}
    dimensions = {}
    for category in extracted:
        with np.load(run / ("category_%d.npz" % category), allow_pickle=False) as record:
            if ("category" not in record or record["category"].shape != ()
                    or record["category"].dtype.kind not in "iu"
                    or int(record["category"]) != category):
                raise ValueError("category record id differs from its filename")
            if "categories" not in record or not np.array_equal(record["categories"], categories):
                raise ValueError("category record slice order differs from declared categories")
            for layer in LAYERS:
                for family in ("computation", "image"):
                    fields = {}
                    for half in HALVES:
                        key = "%s__%s__%s" % (family, layer, half)
                        if key not in record:
                            raise ValueError("missing representation: " + key)
                        vector = np.asarray(record[key], dtype=np.float64)
                        if (vector.ndim != 1 or len(vector) < 2
                                or not np.isfinite(vector).all() or (vector < 0).any()):
                            raise ValueError(key + " must be a finite nonnegative post-ReLU feature vector")
                        if layer in dimensions and dimensions[layer] != len(vector):
                            raise ValueError("representation feature dimensions differ")
                        dimensions[layer] = len(vector)
                        fields[half] = vector
                        rows[layer][family][half].append(vector)
                    if family == "image":
                        counts = image_halves[positions[category]]
                        expected = (counts[0] * fields["A"] + counts[1] * fields["B"]) / counts.sum()
                        # Independent float32 reductions of up to 1,300 images
                        # accumulate different rounding in full and half means.
                        if not np.allclose(fields["mean"], expected, rtol=2e-5, atol=2e-5):
                            raise ValueError(key + " mean differs from count-weighted odd/even image halves")
    arrays = {layer: {family: {half: np.stack(values) for half, values in fields.items()}
                      for family, fields in families.items()} for layer, families in rows.items()}
    selected = [positions[category] for category in extracted]
    raw_counts = np.load(run / "confusion_counts.npy", allow_pickle=False)
    if raw_counts.ndim != 2:
        raise ValueError("confusion counts must be a matrix")
    if raw_counts.shape[0] == len(categories) and extracted != categories:
        raw_counts = raw_counts[selected]
    confusion = confusion_pairs(raw_counts, extracted, images[selected])
    saved = run / "confusion_probabilities.npy"
    if saved.exists():
        probabilities = np.load(saved, allow_pickle=False)
        if probabilities.shape == (len(categories), len(categories)) and extracted != categories:
            probabilities = probabilities[np.ix_(selected, selected)]
        directed = confusion["directed_shared_rates"]
        expected = directed.astype(np.float32)
        if probabilities.shape != expected.shape or not np.allclose(probabilities, expected, rtol=2e-6, atol=2e-7):
            raise ValueError("saved confusion probabilities differ from raw top-1 counts")
    return manifest, extracted, arrays, confusion


def aggregate(weight, taxonomy, weight_second=None, n_permutations=1000):
    paths = {"weight": weight, "weight_second": weight_second}
    runs = {arm: load_run(path) for arm, path in paths.items() if path is not None}
    # The label, kind and run that each extraction's manifest records for its circuits.
    roles = {"weight": ("P", "weight", "first"), "weight_second": ("P", "weight", "second")}
    for arm, (item, _, _, _) in runs.items():
        label, kind, run = roles[arm]
        if item.get("label") != label or item.get("arm_kind") != kind or (run is not None and item.get("run") != run):
            raise ValueError("extraction role differs from the arm it is given as: " + arm)
    manifest, categories, _, confusion = runs["weight"]
    taxonomy = pd.read_csv(taxonomy) if isinstance(taxonomy, (str, Path)) else taxonomy.copy()
    for arm, (other, ids, arrays, counts) in runs.items():
        if ids != categories or other["categories"] != manifest["categories"]:
            raise ValueError("paired runs differ in population or category order: " + arm)
        for key in ("slice_counts", "image_counts", "panel_sha256", "input_identity"):
            if other.get(key) != manifest.get(key):
                raise ValueError("paired runs differ in image inputs: " + key)
        if not np.array_equal(counts["directed_shared_rates"], confusion["directed_shared_rates"]):
            raise ValueError("paired runs differ in dense confusion counts")
        for layer in LAYERS:
            for half in HALVES:
                if not np.allclose(arrays[layer]["image"][half], runs["weight"][2][layer]["image"][half], rtol=2e-6, atol=2e-7):
                    raise ValueError("paired dense image representations differ")
    layers, differences = {}, {}
    for layer in LAYERS:
        plan = S.TreePlan(categories, taxonomy) if len(categories) >= 4 else None
        vectors = {arm: item[2][layer]["computation"]["mean"] for arm, item in runs.items()}
        vectors["image"] = runs["weight"][2][layer]["image"]["mean"]
        layers[layer] = {arm: S.representation_metrics(values, confusion["pair_rate"], categories, taxonomy, plan)
                         for arm, values in vectors.items()}
        pairs = [("weight", "image")]
        if "weight_second" in vectors:
            pairs.append(("weight", "weight_second"))
        differences[layer] = {}
        for left, right in pairs:
            differences[layer][left + "_minus_" + right] = S.paired_difference(
                vectors[left], vectors[right], categories, taxonomy, confusion["pair_rate"],
                n_permutations=n_permutations, plan=plan)
        for arm in ARMS:
            if arm not in vectors:
                layers[layer][arm] = {"tree": {level: S.undefined("arm extraction is not supplied") for level, _ in S.TREE_LEVELS},
                                      "confusion_spearman": dict(S.undefined("arm extraction is not supplied"), rho=None),
                                      "pearson_similarity": None}
        for left, right in [("weight", "image"), ("weight", "weight_second")]:
            key = left + "_minus_" + right
            if key not in differences[layer]:
                differences[layer][key] = dict(S.undefined("requires both arm extractions"),
                    metrics={metric: dict(S.undefined("requires both arm extractions"), difference=None,
                                          jackknife=S.undefined("requires both arm extractions"),
                                          category_swap=S.undefined("requires both arm extractions")) for metric in S.METRICS})
        del plan
    return {"schema": SCHEMA, "categories": categories,
            "declared_categories": manifest["categories"], "n_categories": len(categories),
            "input_manifests": {arm: item[0] for arm, item in runs.items()},
            "estimators": {"tree": "tree_share's node adjustment and 50 permutations of the category labels, seed 0",
                           "confusion": "Spearman correlation over all unordered off-diagonal category pairs, "
                                        "average ranks, zero pairs included",
                           "jackknife": "delete-one-category at the shared 100; 100 fixed random groups of ten at all "
                                        "1,000, seed 0; node adjustment and label permutations refitted on each "
                                        "deletion",
                           "permutation": "paired category-vector swaps, probability 1/2, %d draws, seed 0; "
                                          "two-sided plus-one p" % n_permutations,
                           "interval": "difference +/- 1.96 sqrt((g-1)/g sum((delete-group difference - "
                                       "replicate mean)^2))"},
            "confusion": confusion, "layers": layers, "differences": differences}


def population_categories(name):
    """held_out: the 900 categories that are not among the shared 100."""
    if name == "held_out":
        return np.setdiff1d(np.arange(1000), shared100()).tolist()
    raise ValueError("unknown population: " + str(name))


def compare_population(weight, taxonomy, categories, n_permutations=1000):
    """At fc6 and fc7, the P circuits' activations against the prototypes on some of an extraction's categories
    (fc_statistics.paired_difference), with the jackknife over fixed random groups of ten of them."""
    manifest, extracted, arrays, confusion = load_run(weight)
    if manifest.get("label") != "P" or manifest.get("arm_kind") != "weight" or manifest.get("run") != "first":
        raise ValueError("a population is compared on the first extraction of the P circuits")
    positions = {category: index for index, category in enumerate(extracted)}
    if any(category not in positions for category in categories):
        raise ValueError("the extraction lacks some of the population's categories")
    rows = [positions[category] for category in categories]
    taxonomy = pd.read_csv(taxonomy) if isinstance(taxonomy, (str, Path)) else taxonomy.copy()
    plan = S.TreePlan(categories, taxonomy)
    pair_rate = confusion["pair_rate"][np.ix_(rows, rows)]
    return {layer: S.paired_difference(arrays[layer]["computation"]["mean"][rows], arrays[layer]["image"]["mean"][rows],
                                       categories, taxonomy, pair_rate, n_permutations=n_permutations, plan=plan,
                                       groups_of_ten=True)
            for layer in LAYERS}


def population_rows(population, differences):
    """The table rows of one population: at each layer, the superordinate tree shares of the circuits' activations and
    of the prototypes, their difference with its jackknife SE and interval, and the category-swap p."""
    rows = []
    for layer in LAYERS:
        cell = differences[layer]["metrics"]["tree_superordinate"]
        jackknife, swaps = cell["jackknife"], cell["category_swap"]
        rows.append(dict(population=population, n_categories=differences[layer]["n_categories"], layer=layer,
                         circuit=cell["left"], prototypes=cell["right"], difference=cell["difference"],
                         se=jackknife.get("se"), ci_low=jackknife.get("lo"), ci_high=jackknife.get("hi"),
                         p=swaps.get("p_two_sided"), n_jackknife_groups=jackknife["n_replicates"],
                         n_category_swaps=swaps["n_permutations"]))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weight", required=True, type=Path, help="the activation files of the P circuits")
    parser.add_argument("--weight-second", type=Path, help="the activation files of the P-redraw circuits")
    parser.add_argument("--taxonomy", required=True, type=Path, help="taxonomy/inputs/category_taxonomy.csv")
    parser.add_argument("--population", nargs="+", choices=POPULATIONS,
                        help="compare on these populations of the 1,000 categories (held_out: the 900 outside the "
                             "shared 100) and write their table")
    parser.add_argument("--out", required=True, type=Path,
                        help="the JSON to write, or with --population the table (must not exist)")
    args = parser.parse_args(argv)
    if args.population and args.weight_second:
        parser.error("--population compares the P circuits with the prototypes only")
    check_outputs(files=[args.out], inputs=[args.weight, args.weight_second, args.taxonomy])
    if args.population:
        taxonomy = pd.read_csv(args.taxonomy)
        rows = []
        for population in args.population:
            differences = compare_population(args.weight, taxonomy, population_categories(population))
            rows += population_rows(population, differences)
        with write_file(args.out, newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=TABLE_COLUMNS, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        return 0
    result = aggregate(args.weight, args.taxonomy, args.weight_second)
    with write_file(args.out) as stream:
        json.dump(S.json_clean(result), stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error))
