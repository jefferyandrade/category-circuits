"""The category-tree structure each measure carries beyond another, at AlexNet fc6 and fc7 (Fig. 7C).

For all 1,000 categories, the circuits' activations (computation, from every training image), the category prototypes
(image, the dense network's category means) and the wiring similarity are compared through their category-tree
shares: each alone, and the partial share of one beyond another (computation beyond image, computation beyond wiring,
wiring beyond computation, image beyond computation), at the domain, superordinate and block50 levels. Each share is
node-adjusted and adjusted by its mean over 50 label permutations (seed 0, drawn independently for each cell, with the
full-data permutation mean held fixed in the deletions); its interval is the share plus or minus 1.96 times the
jackknife SE over 100 seed-0 random groups of ten categories (--jackknife ten, the paper's setting; single-category
deletions with --jackknife single). Planted-value checks of the estimator run before any input is read.

Usage: partial_tree_shares.py --run P_ALL_1000 --wiring WIRING_DIR --taxonomy category_taxonomy.csv --out OUT.json
where --run holds the circuits' all-image activation files (fc_all_images.load_run) and --wiring the per-layer wiring
matrices (gram_<layer>.npz and pairs_<layer>.npz, or <layer>_matrices.npz). Completed cells are kept in OUT.json.cells/;
a run that stops resumes with the same arguments. OUT.json is written once when every cell is complete.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import fc_all_images

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stats"))
import tree_share as TS  # noqa: E402
import tree_infer as TI  # noqa: E402
from tree_contrast import partial_share  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import check_outputs, publish_file, validate_paths

LEVELS = (("domain", "animate"), ("superordinate", "superordinate"), ("block50", "block50"))
COMPARISONS = (("computation", None), ("image", None), ("wiring", None),
               ("computation", "image"), ("computation", "wiring"),
               ("wiring", "computation"), ("image", "computation"))


def comparison_key(pair):
    y_name, x_name = pair
    return y_name if x_name is None else y_name + "_beyond_" + x_name


def atomic_json(path, result):
    publish_file(json.dumps(result, indent=1) + "\n", path)


def fingerprint(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def pearson(vectors):
    v = vectors - vectors.mean(1, keepdims=True)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v @ v.T


def category_indices(source, categories):
    """The positions of categories in the category order source."""
    source = np.asarray(source)
    if source.ndim != 1 or len(set(source.tolist())) != len(source):
        raise ValueError("source categories must be unique and one-dimensional")
    positions = {int(c): i for i, c in enumerate(source)}
    if any(int(c) not in positions for c in categories):
        raise ValueError("a category is missing from the source")
    return np.asarray([positions[int(c)] for c in categories], dtype=np.int64)


def load_wiring(root, layer, categories):
    root = Path(root)
    cube = root / (layer + "_matrices.npz")
    if cube.exists():
        with np.load(cube, allow_pickle=False) as z:
            ids = z["categories"]
            matrix = np.asarray(z["similarity__circuits"], dtype=np.float64)
        sources = [fingerprint(cube)]
    else:
        # The wiring arithmetic of tree_contrast.load.
        gp, pp = root / ("gram_" + layer + ".npz"), root / ("pairs_" + layer + ".npz")
        with np.load(gp, allow_pickle=False) as z, np.load(pp, allow_pickle=False) as pz:
            ids = z["categories"]
            if not np.array_equal(ids, pz["categories"]):
                raise ValueError("gram and pairs categories are not aligned")
            K = ids.size
            I = (z["I_all"] if "I_all" in z.files else z["I"]).astype(np.float64)
            J0 = pz["J0_magnitude"].astype(np.float64)
        if any(M.ndim != 2 or M.shape[0] != M.shape[1] or M.shape[0] < K for M in (I, J0)):
            raise ValueError("gram and pairs matrices need the square P block")
        I = I[:K, :K]
        d = np.diag(I)
        S = d[:, None] + d[None, :]
        J = np.where(S - I > 0, I / np.where(S - I > 0, S - I, 1), np.nan)
        matrix = J - J0[:K, :K]
        sources = [fingerprint(gp), fingerprint(pp)]
    if matrix.shape != (len(ids), len(ids)):
        raise ValueError("wiring matrix and categories have different shapes")
    idx = category_indices(ids, categories)
    return matrix[np.ix_(idx, idx)], sources


def deletion_groups(n, kind):
    if kind == "single":
        return [np.asarray([i]) for i in range(n)]
    order = np.random.default_rng(0).permutation(n)
    return [np.sort(g) for g in np.array_split(order, int(np.ceil(n / 10)))]


class TreeShareCell:
    """The tree share of y (x None) or its partial share beyond x, at one level, with its jackknife interval."""
    def __init__(self, y, x, labels):
        self.y, self.x, self.labels = y, x, labels
        self.cache = {}

    def raw(self, idx, rng=None):
        key = tuple(idx)
        if key not in self.cache:
            if len(self.cache) >= 2:
                self.cache.pop(next(iter(self.cache)))
            iu = np.triu_indices(len(idx), 1)
            y = TS.node_adjust(self.y[np.ix_(idx, idx)])
            x = None if self.x is None else TS.node_adjust(self.x[np.ix_(idx, idx)])[iu]
            self.cache[key] = (y, x, iu)
        y, x, iu = self.cache[key]
        codes = pd.factorize(self.labels[idx])[0]
        if rng is not None:
            codes = rng.permutation(codes)
        if x is None:
            return TI.formed_share(y[iu], iu, codes)
        G = int(codes.max()) + 1
        return partial_share(y[iu], x, TS.cell_index(codes, iu, G), G * G)

    def compute(self, nperm, groups):
        full = np.arange(len(self.labels))
        rng = np.random.default_rng(0)
        pm = float(np.nanmean([self.raw(full, rng) for _ in range(nperm)]))
        observed = self.raw(full)
        adjusted = TI.adjusted_share(observed, pm)
        replicates = np.asarray([TI.adjusted_share(self.raw(np.setdiff1d(full, group)), pm)
                                 for group in groups])
        G = len(groups)
        se = float(np.sqrt((G - 1) / G * np.nansum((replicates - np.nanmean(replicates)) ** 2)))
        return dict(observed=observed, permutation_mean=pm, adjusted=adjusted, se=se,
                    lo=adjusted - 1.96 * se, hi=adjusted + 1.96 * se)


def self_test():
    """Planted values: a pair matrix that is exactly its tree cells has share 1.000, alone or beyond an unrelated
    matrix, and every share agrees at three decimals with an independent least-squares fit."""
    labels = np.repeat(np.arange(4), 2)
    y = np.where(labels[:, None] == labels[None, :], 6.0, -1.0)
    np.fill_diagonal(y, 0.0)  # off-diagonal row sums are exactly zero
    x = np.arange(64, dtype=np.float64).reshape(8, 8) ** 2
    x = x + x.T
    full, groups = np.arange(8), deletion_groups(8, "single")
    plants = {}
    for name, target, control in (("cell_only", y, None), ("negative_cell_only", -y, None),
                                   ("cell_beyond_input", y, x), ("noisy_cells", y + x / 16, None),
                                   ("noisy_cells_beyond_input", y + x / 16, x ** 2)):
        result = TreeShareCell(target, control, labels).compute(50, groups)
        # The analytic planted value is 1. A separate dense OLS route checks the
        # statistic at the reported three-decimal precision.
        iu = np.triu_indices(8, 1)
        cell = np.minimum(labels[iu[0]], labels[iu[1]]) * 4 + np.maximum(labels[iu[0]], labels[iu[1]])
        design = np.column_stack([np.ones(len(iu[0]))] +
                                 [(cell == c).astype(float) for c in np.unique(cell)])
        nodes = np.zeros((len(iu[0]), 8))
        nodes[np.arange(len(iu[0])), iu[0]] = 1
        nodes[np.arange(len(iu[0])), iu[1]] = 1
        yy = target[iu] - nodes @ np.linalg.lstsq(nodes, target[iu], rcond=None)[0]
        base = np.ones((len(yy), 1))
        if control is not None:
            # Independent explicit node-effect OLS, then full cell OLS.
            xx = control[iu] - nodes @ np.linalg.lstsq(nodes, control[iu], rcond=None)[0]
            base = np.column_stack((base, xx))
            design = np.column_stack((design, xx))
        residual = yy - base @ np.linalg.lstsq(base, yy, rcond=None)[0]
        full_residual = yy - design @ np.linalg.lstsq(design, yy, rcond=None)[0]
        oracle = 1.0 - float(full_residual @ full_residual) / float(residual @ residual)
        if format(result["observed"], ".3f") != format(oracle, ".3f"):
            raise AssertionError("independent planted-cell recovery failed")
        analytic = None if name.startswith("noisy") else 1.0
        if analytic is not None and format(result["observed"], ".3f") != "1.000":
            raise AssertionError("analytic cell-only plant failed")
        if analytic is not None and not result["lo"] <= analytic <= result["hi"]:
            raise AssertionError("planted value lies outside its interval")
        if analytic is None and (not 0 < oracle < 1 or format(oracle, ".3f") == "1.000"):
            raise AssertionError("noisy plant must test a nontrivial share")
        plants[name] = {"analytic_raw_share": analytic, "ols_raw_share": oracle, "result": result}
    null = TreeShareCell(np.zeros((8, 8)), None, labels).raw(full)
    if np.isfinite(null):
        raise AssertionError("a constant pair vector must be undefined")
    ten = deletion_groups(1000, "ten")
    if len(ten) != 100 or any(len(g) != 10 for g in ten) or not np.array_equal(np.sort(np.concatenate(ten)), np.arange(1000)):
        raise AssertionError("groups of ten do not partition all 1000 categories")
    return {"passed": True, "plants": plants, "zero_vector_is_undefined": True,
            "group_partition": "100 disjoint groups of ten covering 0..999",
            "accuracy_rule": "the estimator and an independent dense least-squares fit agree at three decimals; "
                             "cell-only plants recover 1.000"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="the circuits' all-image activation files, all 1,000 categories")
    ap.add_argument("--wiring", required=True, help="the folder of the per-layer wiring matrices")
    ap.add_argument("--taxonomy", required=True, help="taxonomy/inputs/category_taxonomy.csv")
    ap.add_argument("--out", required=True, help="the completed JSON; an interrupted run resumes from its .cells folder")
    ap.add_argument("--nperm", type=int, default=50, help="label permutations per share")
    ap.add_argument("--jackknife", choices=("single", "ten"), default="ten",
                    help="delete single categories, or 100 random groups of ten")
    a = ap.parse_args(argv)
    out = Path(a.out)
    progress = out.with_name(out.name + ".cells")
    check_outputs(files=[out], inputs=[a.run, a.wiring, a.taxonomy])
    validate_paths([progress], [a.run, a.wiring, a.taxonomy, out])
    records = [progress / "configuration.json"]
    records += [progress / (layer + "__" + comparison_key(pair) + "__" + level + ".json")
                for layer in ("fc6", "fc7") for pair in COMPARISONS for level, _ in LEVELS]
    validate_paths(records, [a.run, a.wiring, a.taxonomy, out])
    controls = self_test()  # the planted-value checks run before the inputs are read
    if a.nperm < 1:
        ap.error("--nperm must be positive")
    manifest, all_categories, arrays, _ = fc_all_images.load_run(a.run)
    all_categories = np.asarray(all_categories)
    if not np.array_equal(all_categories, np.arange(1000)):
        raise ValueError("the extraction must contain exactly categories 0..999 in order")
    categories = all_categories
    idx = category_indices(all_categories, categories)
    tax = pd.read_csv(a.taxonomy).set_index("category")
    layers = ["fc6", "fc7"]
    groups = deletion_groups(len(categories), a.jackknife)
    configuration = {"inputs": {name: str(Path(getattr(a, name)).resolve()) for name in ("run", "wiring", "taxonomy")},
                     "categories": categories.tolist(), "layers": layers, "levels": [x[0] for x in LEVELS],
                     "comparisons": [comparison_key(x) for x in COMPARISONS],
                     "nperm": a.nperm, "seed": 0, "jackknife": a.jackknife,
                     "deletion_groups_indices": [g.tolist() for g in groups], "normal_multiplier": 1.96}
    progress.mkdir(parents=True, exist_ok=True)
    config_path = progress / "configuration.json"
    if config_path.exists():
        saved_configuration = json.loads(config_path.read_text())
    else:
        atomic_json(config_path, configuration)
        saved_configuration = configuration
    result = {"configuration": saved_configuration, "layers": {}}
    if result["configuration"] != configuration:
        raise ValueError("the cell folder holds cells computed with other inputs or settings; use another output path")
    result.update(controls=controls, extraction_manifest=manifest,
                  estimators="full-data permutation mean held fixed in the deletions; normal 95% interval")
    result.setdefault("wiring_sources", {})
    for layer in layers:
        wiring, sources = load_wiring(a.wiring, layer, categories)
        result["wiring_sources"][layer] = sources
        mats = {name: pearson(np.asarray(arrays[layer][name]["mean"], dtype=np.float64)[idx])
                for name in ("computation", "image")}
        mats["wiring"] = wiring
        cells = result["layers"].setdefault(layer, {})
        for level, column in LEVELS:
            labels = tax.loc[categories, column].to_numpy()
            for y_name, x_name in COMPARISONS:
                key = comparison_key((y_name, x_name))
                cell_path = progress / (layer + "__" + key + "__" + level + ".json")
                if cell_path.exists():
                    cells.setdefault(key, {})[level] = json.loads(cell_path.read_text())
                    continue
                cell = TreeShareCell(mats[y_name], mats[x_name] if x_name else None, labels)
                cells.setdefault(key, {})[level] = cell.compute(a.nperm, groups)
                atomic_json(cell_path, cells[key][level])
                print(json.dumps({"completed": "/".join((layer, key, level)), **cells[key][level]}), flush=True)
    atomic_json(out, result)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error))
