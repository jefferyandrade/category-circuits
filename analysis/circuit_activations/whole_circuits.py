"""The whole-circuit comparison at AlexNet fc6 and fc7 (section 2.5).

An exploratory comparison on the shared 100 categories ran each circuit whole, with every layer's mask applied, on 32
images of each shared category, and the dense network on 256 of each category's own images for its category means.
For every representation in that extraction and each layer, this script gives the category-tree shares (node- and
permutation-adjusted, 50 label permutations, seed 0; tree_share.tree_rows) of the Pearson similarity between
categories. Section 2.5 reports the fc7 superordinate shares of the whole circuits (full) and of the category means
(image). The extraction also holds partial circuits (fc7's mask alone, every mask but fc7's, and points along the
pruning walk) and two other references; their shares are computed in the same way.

Usage: whole_circuits.py --run DIR --taxonomy category_taxonomy.csv --out OUT.json
where DIR holds EXTRACT.json (the manifest), COMPLETE.json and category_<c>.npz (each representation at fc6 and fc7,
as the mean and the means of the two halves of the images).
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from fc_statistics import REL_TOL, category_ids, centered_scaled, json_clean, undefined

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stats"))
import tree_share as TS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import check_outputs, write_file


LAYERS = ("fc6", "fc7")
# The representations of the extraction, by the names its files use. The pruning walk runs from fc7 down to conv1:
# eight points along fc7's own part of it (from dense to fc7_only), then the circuit after each further layer
# (walk_after_conv1 is the whole circuit, full).
FC7_WALK = tuple("walk_fc7_%02d" % index for index in range(8))
LATER_WALK = tuple("walk_after_" + layer for layer in
                   ("fc6", "conv5", "conv4", "conv3", "conv2", "conv1"))
REQUIRED_VARIANTS = ("full", "fc7_only", "all_but_fc7") + FC7_WALK + LATER_WALK
REFERENCES = ("image", "decision_weighted_image", "dense_all_image")
TREE_LEVELS = ("domain", "superordinate", "block50")


def _finite_number(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value)


def check_trajectories(trajectory_by_category, categories):
    """Every category's walk record lists the 14 points in order, and fc7's walk starts dense and only removes."""
    for category in categories:
        records = trajectory_by_category.get(str(category))
        if not isinstance(records, list) or [record.get("name") for record in records] != list(FC7_WALK + LATER_WALK):
            raise ValueError("trajectory metadata must retain all 14 points in order for every category")
        steps, fractions = [], []
        for record in records[:8]:
            step, fraction = record.get("accepted_steps"), record.get("retained_fraction")
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                raise ValueError("trajectory accepted_steps must be nonnegative integers")
            if not _finite_number(fraction) or not 0 <= fraction <= 1:
                raise ValueError("trajectory retained fractions must be finite and between zero and one")
            steps.append(step)
            fractions.append(fraction)
        if steps[0] != 0 or fractions[0] != 1.0 or steps != sorted(steps) or fractions != sorted(fractions, reverse=True):
            raise ValueError("fc7 trajectory must start dense and move monotonically through accepted states")


def load_run(run):
    """Read the extraction, checking its row order, every half and the walk's endpoints."""
    run = Path(run)
    manifest = json.loads((run / "EXTRACT.json").read_text())
    complete = json.loads((run / "COMPLETE.json").read_text())
    # The schema label of the extraction's files.
    if manifest.get("schema") != "comprep-v66-1" or complete.get("schema") != "comprep-v66-1":
        raise ValueError("unsupported extraction schema")
    categories = category_ids(manifest.get("categories"), "categories")
    extracted = category_ids(manifest.get("extracted_categories"), "extracted_categories")
    if complete.get("categories") != extracted:
        raise ValueError("COMPLETE.json categories differ from the extraction's")
    if len(categories) != 100:
        raise ValueError("categories must be the shared 100")
    if not set(extracted).issubset(categories):
        raise ValueError("extracted categories differ from the shared 100")
    if manifest.get("slice_categories") != categories:
        raise ValueError("slice category order differs from the shared 100")
    if manifest.get("slice_images_per_category") != 32 or manifest.get("image_images_per_category") != 256:
        raise ValueError("extraction must use 32 slice and 256 image-reference images per category")
    if manifest.get("references") != list(REFERENCES) or manifest.get("layers") != list(LAYERS):
        raise ValueError("extraction references or layers differ")
    if any(manifest.get("endpoint_equal", {}).get(str(c)) is not True for c in extracted):
        raise ValueError("every category must have bit-for-bit reconstructed endpoint equality")
    check_trajectories(manifest.get("trajectory_by_category", {}), extracted)
    probe = manifest.get("probe")
    if not isinstance(probe, bool):
        raise ValueError("manifest must state whether the extraction is partial (probe)")
    if not probe and extracted != categories:
        raise ValueError("a complete extraction is missing categories or out of order")
    variants = manifest.get("variants")
    if not isinstance(variants, list) or len(variants) != len(set(variants)):
        raise ValueError("manifest variants must be a unique ordered list")
    if set(variants) != set(REQUIRED_VARIANTS):
        raise ValueError("manifest must contain all three mask variants and all 14 walk points")
    names = list(variants) + list(REFERENCES)
    rows = {layer: {name: {half: [] for half in ("mean", "A", "B")} for name in names}
            for layer in LAYERS}
    dimensions = {}
    for category in extracted:
        with np.load(run / ("category_%d.npz" % category), allow_pickle=False) as record:
            if "category" not in record or np.asarray(record["category"]).shape != () or int(record["category"]) != category:
                raise ValueError("category record id is absent or differs from its filename")
            if "categories" not in record or not np.array_equal(record["categories"], categories):
                raise ValueError("category record slice category order differs from the shared 100")
            for layer in LAYERS:
                for name in names:
                    fields = {}
                    for half in ("mean", "A", "B"):
                        key = name + "__" + layer + "__" + half
                        if key not in record:
                            raise ValueError("missing representation: " + key)
                        value = np.asarray(record[key], dtype=np.float64)
                        if value.ndim != 1 or len(value) < 2 or not np.isfinite(value).all():
                            raise ValueError(key + " must be a finite feature vector")
                        if layer in dimensions and len(value) != dimensions[layer]:
                            raise ValueError("representation feature dimensions differ")
                        dimensions[layer] = len(value)
                        fields[half] = value
                        rows[layer][name][half].append(value)
                    if not np.allclose(fields["mean"], (fields["A"] + fields["B"]) / 2,
                                       rtol=2e-6, atol=2e-7):
                        raise ValueError(key + " mean differs from the two equally weighted halves")
    arrays = {layer: {name: {half: np.stack(values) for half, values in fields.items()}
                      for name, fields in families.items()} for layer, families in rows.items()}
    for layer in LAYERS:
        for half in ("mean", "A", "B"):
            dense = arrays[layer]["dense_all_image"][half]
            if not np.allclose(dense, dense[:1], rtol=2e-6, atol=2e-7):
                raise ValueError("dense all-image reference must be the same for every circuit category")
            for first, second in ((FC7_WALK[0], "dense_all_image"),
                                  (FC7_WALK[-1], "fc7_only"),
                                  (LATER_WALK[-1], "full")):
                if not np.allclose(arrays[layer][first][half], arrays[layer][second][half],
                                   rtol=2e-6, atol=2e-7):
                    raise ValueError("representation trajectory endpoint mismatch: " + first + " / " + second)
    for name in ("fc7_only",) + FC7_WALK:
        for half in ("mean", "A", "B"):
            if not np.allclose(arrays["fc6"][name][half],
                               arrays["fc6"]["dense_all_image"][half], rtol=2e-6, atol=2e-7):
                raise ValueError("an fc7-only mask changed the fc6 all-image response")
    return manifest, extracted, arrays


def pearson_rows(m):
    c = centered_scaled(m)
    c = c / np.linalg.norm(c, axis=1, keepdims=True)
    return c @ c.T


def _validate_matrix(matrix, k, name):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (k, k):
        raise ValueError(name + " must align with category matrix dimensions")
    off = ~np.eye(k, dtype=bool)
    if not np.isfinite(matrix[off]).all():
        raise ValueError(name + " must have a complete finite off-diagonal")
    if not np.allclose(matrix, matrix.T, rtol=0, atol=1e-12, equal_nan=True):
        raise ValueError(name + " must be symmetric")
    return matrix


def tree_shares(similarity, categories, taxonomy, name):
    """Node- and permutation-adjusted tree shares of one category similarity matrix, by level."""
    categories = np.asarray(categories)
    if categories.ndim != 1 or len(categories) < 4 or len(np.unique(categories)) != len(categories):
        raise ValueError("at least four unique category ids required")
    if not isinstance(taxonomy, pd.DataFrame):
        taxonomy = pd.DataFrame(taxonomy)
    if "category" in taxonomy.columns:
        taxonomy = taxonomy.set_index("category")
    if not taxonomy.index.is_unique:
        raise ValueError("taxonomy category ids must be unique")
    if not np.isin(categories, taxonomy.index).all():
        raise ValueError("categories absent from taxonomy")
    for _, column in TS.LEVELS:
        if column not in taxonomy.columns or taxonomy.loc[categories, column].isna().any():
            raise ValueError("missing taxonomy labels: " + column)
    matrix = _validate_matrix(similarity, len(categories), name)
    adjusted = TS.node_adjust(matrix)
    iu = np.triu_indices(len(categories), 1)
    if np.ptp(adjusted[iu]) <= np.finfo(float).eps * max(1, np.max(np.abs(matrix[iu]))):
        return {level: dict(level=level, status="undefined", reason="no variation after node adjustment")
                for level, _ in TS.LEVELS}
    with np.errstate(divide="ignore", invalid="ignore"):
        rows, _ = TS.tree_rows(adjusted, categories, taxonomy, np.random.default_rng(TS.SEED))
    for row in rows:
        if not np.isfinite(row["share_adjusted"]):
            row.update(status="undefined", reason="permutation adjustment is undefined")
    return {row["level"]: row for row in rows}


def representation_shares(fields, categories, taxonomy, name):
    """The Pearson similarity between categories of one representation's means, and its tree shares."""
    n = len(categories)
    result = {"n_categories": n, "feature_dimension": fields["mean"].shape[1]}
    try:
        similarity = pearson_rows(fields["mean"])
        result["pearson_similarity"] = similarity
        vectors = np.asarray(fields["mean"], dtype=np.float64)
        scale = max(float(np.abs(vectors).max()), np.finfo(float).tiny)
        if np.abs(vectors - vectors[:1]).max() <= REL_TOL * scale:
            result["tree"] = {level: undefined("representation is identical across categories (variation at roundoff)")
                              for level in TREE_LEVELS}
        elif n < 4:
            result["tree"] = {level: undefined("tree share requires at least four categories") for level in TREE_LEVELS}
        else:
            try:
                result["tree"] = tree_shares(similarity, categories, taxonomy, name)
            except (ZeroDivisionError, FloatingPointError):
                # A saturated level (e.g. all singleton blocks) has permutation
                # share one, and tree_share.tree_rows divides Python floats.
                # Retry each level alone to retain the other levels. Repeating
                # its labels in all three columns keeps that level's
                # exact RNG position: every permutation still consumes n rows.
                result["tree"] = {}
                for level, column in (("domain", "animate"), ("superordinate", "superordinate"), ("block50", "block50")):
                    standalone = taxonomy.copy()
                    for target in ("animate", "superordinate", "block50"):
                        standalone[target] = taxonomy[column]
                    try:
                        result["tree"][level] = tree_shares(similarity, categories, standalone, name)[level]
                    except (ZeroDivisionError, FloatingPointError) as error:
                        result["tree"][level] = undefined("permutation adjustment is undefined: " + str(error))
    except (ValueError, np.linalg.LinAlgError) as error:
        result["pearson_similarity"] = None
        result["tree"] = {level: undefined(error) for level in TREE_LEVELS}
    return result


def aggregate(run, taxonomy):
    manifest, categories, arrays = load_run(run)
    taxonomy = pd.read_csv(taxonomy) if isinstance(taxonomy, (str, Path)) else taxonomy.copy()
    if "category" not in taxonomy or taxonomy.category.duplicated().any():
        raise ValueError("taxonomy must contain unique category ids")
    for label in ("block50", "superordinate", "animate"):
        if label not in taxonomy or taxonomy.set_index("category").reindex(categories)[label].isna().any():
            raise ValueError("missing taxonomy labels: " + label)
    layers = {layer: {name: representation_shares(fields, categories, taxonomy, name)
                      for name, fields in arrays[layer].items()} for layer in LAYERS}
    return {"categories": categories, "shared_categories": manifest["categories"], "n_categories": len(categories),
            "variants": manifest["variants"], "references": list(REFERENCES),
            "trajectory_by_category": manifest.get("trajectory_by_category", {}),
            "estimators": {"tree": "tree_share.tree_rows on the Pearson similarity between categories",
                           "tree_adjustment": "node and permutation adjusted",
                           "tree_permutations": 50, "tree_seed": 0},
            "layers": layers}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, type=Path, help="the whole-circuit extraction folder")
    parser.add_argument("--taxonomy", required=True, type=Path, help="taxonomy/inputs/category_taxonomy.csv")
    parser.add_argument("--out", required=True, type=Path, help="the JSON to write (must not exist)")
    args = parser.parse_args(argv)
    check_outputs(files=[args.out], inputs=[args.run, args.taxonomy])
    result = aggregate(args.run, args.taxonomy)
    with write_file(args.out) as stream:
        json.dump(json_clean(result), stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error))
