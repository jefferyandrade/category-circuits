"""Fig. 6E: the adjusted agreement of AlexNet's wiring with each other network's, beyond the category tree and the mean
category representations (partial correlations).

For AlexNet's conv5, fc6 and fc7 and every layer of another network: the correlation over category pairs of the two
wirings' excess overlap, raw (r_raw), beyond the cell means of each level of the category tree (r_beyond_tree), beyond
both networks' representation similarities (r_beyond_activations), and beyond both at the superordinate level
(r_beyond_both). Each wiring matrix is node-adjusted on its own finite pairs before the pairs are intersected, as in
cross_network_agreement.py; each statistic uses the pairs finite in all its inputs (the two wirings, and for the
activation rows both representation matrices). The representation similarities are the cross-validated S|cv:all,
node-adjusted the same way, so that they remove pair-specific similarity only. Every deletion of the joint
delete-one-category jackknife (t 95%) refits the node adjustments, cell means and regression coefficients. The rows
are descriptive, with no multiplicity correction.

With --pairs within_kind or between_kind, only pairs of categories in the same superordinate group, or in different
ones, are used, after the node adjustment on all pairs. Within-kind rows remove the groups' cell means from the
responses and covariates, together with any tree cells. Every row is kept: in the nested category tree, within-kind
r_raw and r_beyond_tree at the domain and superordinate levels coincide, as do r_beyond_activations and
r_beyond_both. Between-kind rows keep the group-level structure unless tree cells are removed. Every row records its
pair subset and number of pairs.

Usage: cross_network_adjusted.py --alex-grams DIR --alex-pairs DIR --other DIR --alex-act-dir DIR --other-act-dir DIR
           --alex-site-map JSON --other-site-map JSON --taxonomy CSV --out CSV [--pairs within_kind] [--categories FILE]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import category_resampling as PS  # noqa: E402
import pair_stats as PST  # noqa: E402
import tree_share as TS  # noqa: E402
import cross_network_agreement as agreement  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import publish_file, validate_paths  # noqa: E402


COLUMNS = ("quantity", "network", "alexnet_layer", "layer", "level", "estimate",
           "ci_low", "ci_high", "n_categories", "method", "status", "reason",
           "pairs", "n_pairs")
SLOTS = (("r_raw", "none"),) + tuple(("r_beyond_tree", level) for level, _ in TS.LEVELS) + (
    ("r_beyond_activations", "none"), ("r_beyond_both", "superordinate"))
METHOD = ("descriptive; joint delete-one-category jackknife (t 95%); "
          "node adjustment and nuisance effects refitted per deletion")


def _validate_matrix(categories, matrix):
    categories = np.asarray(categories)
    matrix = np.asarray(matrix, dtype=float)
    if (categories.ndim != 1 or categories.dtype.kind not in "iu" or
            len(np.unique(categories)) != len(categories)):
        raise ValueError("category ids must be unique integers")
    if matrix.shape != (len(categories), len(categories)):
        raise ValueError("matrix shape and category ids are misaligned")
    off = ~np.eye(len(categories), dtype=bool)
    if not np.allclose(matrix[off], matrix.T[off], rtol=1e-10, atol=1e-12, equal_nan=True):
        raise ValueError("category-pair matrices must be symmetric")
    return categories, matrix


def align_matrices(*datasets):
    """Align labelled matrices explicitly to their sorted category intersection."""
    if not datasets:
        raise ValueError("at least one labelled matrix is required")
    checked = [_validate_matrix(*data) for data in datasets]
    common = checked[0][0]
    for categories, _ in checked[1:]:
        common = np.intersect1d(common, categories)
    common = np.sort(common)
    aligned = []
    for categories, matrix in checked:
        position = {int(category): i for i, category in enumerate(categories)}
        idx = np.asarray([position[int(category)] for category in common], dtype=int)
        aligned.append(matrix[np.ix_(idx, idx)].copy())
    return common, aligned


def load_activation(directory, site):
    """The categories and the similarity of their mean representations (S|cv:all) at one site, diagonal NaN."""
    with np.load(Path(directory) / ("matrices_%s.npz" % site)) as archive:
        categories = np.asarray(archive["class_ids"])
        matrix = np.asarray(archive["S|cv:all"], dtype=float).copy()
    categories, matrix = _validate_matrix(categories, matrix)
    np.fill_diagonal(matrix, np.nan)
    return categories, matrix


def _demean_cells(values, cells):
    means, _ = TS.cell_means(values, cells, int(cells.max()) + 1)
    return values - means[cells]


def _validate_pair_subset(pairs, kind_codes, n):
    if pairs not in (None, "within_kind", "between_kind"):
        raise ValueError("pairs must be None, 'within_kind', or 'between_kind'")
    if pairs is None:
        if kind_codes is not None:
            raise ValueError("kind_codes require a pair subset")
        return None
    if kind_codes is None:
        raise ValueError("kind_codes are required when pairs is set")
    codes = np.asarray(kind_codes)
    if codes.shape != (n,) or codes.dtype.kind not in "iu" or np.any(codes < 0):
        raise ValueError("kind_codes must be non-negative integers aligned with the category matrices")
    return codes


def _pair_mask(inputs, iu, pairs=None, kind_codes=None):
    common = np.logical_and.reduce([np.isfinite(values) for values in inputs])
    if pairs is not None:
        same = kind_codes[iu[0]] == kind_codes[iu[1]]
        common &= same if pairs == "within_kind" else ~same
    return common


def pair_correlation(x, y, alex_activation=None, other_activation=None, tree_codes=None,
                     pairs=None, kind_codes=None, return_n=False):
    """The (partial) correlation of two wiring matrices, node-adjusting each before the joint pair mask.

    With tree cells and activations together, demeaning both responses AND
    covariates implements the joint least-squares projection by the
    Frisch--Waugh--Lovell identity, including a cell-specific intercept.
    Within-kind rows always include kind-cell intercepts; with tree codes,
    the cells are the explicit joint kind and tree pair cells. return_n adds
    the number of jointly finite selected pairs to the returned correlation.
    """
    kind_codes = _validate_pair_subset(pairs, kind_codes, len(x))
    unadjusted_x, unadjusted_y = np.asarray(x), np.asarray(y)
    x, y = agreement.fit_matrix(x), agreement.fit_matrix(y)
    use_activations = alex_activation is not None or other_activation is not None
    if use_activations and (alex_activation is None or other_activation is None):
        raise ValueError("both activation similarity matrices are required")
    if use_activations:
        # covariates in the same pair-specific space as the wiring: node effects removed from the activation
        # similarities too, as tree_share's node-adjusted activation lens does
        alex_activation = agreement.fit_matrix(np.asarray(alex_activation, float))
        other_activation = agreement.fit_matrix(np.asarray(other_activation, float))
    if tree_codes is None and not use_activations and pairs is None:
        r = PST.correlation(x, y)
        if return_n:
            iu = np.triu_indices(len(x), 1)
            return r, int(_pair_mask([x[iu], y[iu]], iu).sum())
        return r
    iu = np.triu_indices(len(x), 1)
    inputs = [x[iu], y[iu]]
    if use_activations:
        inputs += [np.asarray(alex_activation)[iu], np.asarray(other_activation)[iu]]
    common = _pair_mask(inputs, iu, pairs, kind_codes)
    if common.sum() < 3:
        raise ValueError("fewer than three jointly defined category pairs")
    values = np.column_stack([v[common] for v in inputs])
    # Centering first also supplies the intercept for the activation-only fit.
    scale = np.linalg.norm(values[:, :2] - values[:, :2].mean(axis=0), axis=0)
    unadjusted = np.column_stack([unadjusted_x[iu][common], unadjusted_y[iu][common]])
    unadjusted_scale = np.linalg.norm(unadjusted - unadjusted.mean(axis=0), axis=0)
    if np.any(scale <= 1e-12 * unadjusted_scale) or np.any(scale == 0):
        raise ValueError("no residual wiring variation after node adjustment")
    cells = None
    if tree_codes is not None:
        codes = np.asarray(tree_codes, dtype=int)
        if codes.shape != (len(x),) or np.any(codes < 0):
            raise ValueError("tree codes must align with the category matrices")
        cells = TS.cell_index(codes, iu, int(codes.max()) + 1)[common]
    if pairs == "within_kind":
        # Compress occupied kind labels before crossing with tree pair cells;
        # labels need not be contiguous and taxonomy levels need not nest.
        _, kind_cells = np.unique(kind_codes[iu[0]][common], return_inverse=True)
        if cells is None:
            cells = kind_cells
        else:
            _, cells = np.unique(np.column_stack([kind_cells, cells]), axis=0,
                                 return_inverse=True)
    if cells is not None:
        values = np.column_stack([_demean_cells(v, cells) for v in values.T])
    else:
        values = values - values.mean(axis=0)
    responses = values[:, :2]
    if use_activations:
        covariates = values[:, 2:]
        # Scaling preserves the projection when the two similarities have very
        # different units; redundant or constant covariates are permitted.
        norms = np.linalg.norm(covariates, axis=0)
        covariates = covariates / np.where(norms > 0, norms, 1.0)
        coefficients = np.linalg.lstsq(covariates, responses, rcond=None)[0]
        responses = responses - np.einsum("ij,jk->ik", covariates, coefficients)
    responses = responses - responses.mean(axis=0)
    if np.any(np.linalg.norm(responses, axis=0) <= 1e-12 * scale) or np.any(scale == 0):
        raise ValueError("no residual wiring variation after adjustment")
    r = float(np.corrcoef(responses.T)[0, 1])
    return (r, int(common.sum())) if return_n else r


def _row(quantity, level, n, network, alexnet_layer, layer, pairs=None):
    return dict(quantity=quantity, network=network, alexnet_layer=alexnet_layer,
                layer=layer, level=level, estimate=None, ci_low=None, ci_high=None,
                n_categories=n, method=METHOD, status="not formed", reason="",
                pairs="all" if pairs is None else pairs, n_pairs=0)


def analyze_pair_matrices(x, y, alex_activation, other_activation, categories, tax,
                          network="other", alexnet_layer="", layer="", pairs=None):
    """The rows of one layer pair, r_raw, r_beyond_tree at each level, r_beyond_activations and r_beyond_both,
    each with a joint delete-one-category jackknife interval."""
    categories = np.asarray(categories)
    matrices = [_validate_matrix(categories, matrix)[1]
                for matrix in (x, y, alex_activation, other_activation)]
    if not tax.index.is_unique:
        raise ValueError("taxonomy category ids must be unique")
    if not np.isin(categories, tax.index).all():
        raise ValueError("categories absent from taxonomy")
    codes_by_level = {}
    for level, column in TS.LEVELS:
        labels = tax.loc[categories, column]
        if labels.isna().any():
            raise ValueError("missing taxonomy labels at %s" % level)
        codes_by_level[level] = pd.factorize(labels.astype(str))[0]
    kind_codes = _validate_pair_subset(pairs, codes_by_level[TS.LEVELS[1][0]]
                                       if pairs is not None else None, len(categories))
    iu = np.triu_indices(len(categories), 1)
    rows = []
    for quantity, level in SLOTS:
        row = _row(quantity, level, len(categories), network, alexnet_layer, layer, pairs)
        use_activations = quantity in ("r_beyond_activations", "r_beyond_both")
        use_tree = quantity in ("r_beyond_tree", "r_beyond_both")
        inputs = matrices[:] if use_activations else matrices[:2]
        # Node adjustment preserves each matrix's finite-pair mask. Count the
        # full sample here so the count survives an undefined deletion too.
        row["n_pairs"] = int(_pair_mask([m[iu] for m in inputs], iu, pairs, kind_codes).sum())
        tree_index = len(inputs) if use_tree else None
        if use_tree:
            # joint_block_jackknife slices square matrices only. The diagonal
            # carries labels through exactly the same category deletions and
            # is never included in the finite-pair intersection.
            inputs = inputs + [np.diag(codes_by_level[level])]
        kind_index = len(inputs) if pairs is not None else None
        if pairs is not None:
            inputs = inputs + [np.diag(kind_codes)]

        def statistic(*reduced):
            codes = np.diag(reduced[tree_index]).astype(int) if use_tree else None
            kinds = np.diag(reduced[kind_index]).astype(int) if pairs is not None else None
            act = reduced[2:4] if use_activations else (None, None)
            return pair_correlation(reduced[0], reduced[1], *act, tree_codes=codes,
                                    pairs=pairs, kind_codes=kinds)

        try:
            result = PS.joint_block_jackknife(inputs, statistic=statistic, unit="category")
            row.update(estimate=result["estimate"], ci_low=result["ci_low"],
                       ci_high=result["ci_high"], status="formed")
        except (ValueError, np.linalg.LinAlgError) as error:
            row["reason"] = str(error)
        rows.append(row)
    return rows


def _site_map(path):
    mapping = json.loads(Path(path).read_text())
    if not isinstance(mapping, dict) or any(
            not isinstance(layer, str) or not isinstance(site, str) or not site
            for layer, site in mapping.items()):
        raise ValueError("site map must map layer names to activation site names")
    return mapping


def _load_excess(gram, pairs):
    try:
        return _validate_matrix(*agreement.excess(gram, pairs))
    except AssertionError as error:
        raise ValueError("gram and pairs category ids are misaligned") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("alex-grams", "alex-pairs", "other", "alex-act-dir", "other-act-dir",
                 "alex-site-map", "other-site-map", "taxonomy", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--pairs", choices=("all", "within_kind", "between_kind"), default="all")
    parser.add_argument("--categories", help="file of category ids to keep (whitespace-separated); others are dropped "
                        "from every matrix before node adjustment, as confusion_wiring.py --categories does")
    args = parser.parse_args(argv)
    pairs = None if args.pairs == "all" else args.pairs
    out = Path(args.out)
    if os.path.lexists(out):
        raise FileExistsError(out)
    validate_paths([out], [args.alex_grams, args.alex_pairs,
        args.other, args.alex_act_dir, args.other_act_dir, args.alex_site_map,
        args.other_site_map, args.taxonomy] + ([args.categories] if args.categories else []))
    keep = None
    if args.categories:
        keep = np.array(sorted({int(x) for x in Path(args.categories).read_text().split()}), dtype=np.int64)
        if keep.size < 3:
            raise ValueError("--categories keeps fewer than three categories")
    tax = TS.load_tax(args.taxonomy)
    alex_map, other_map = _site_map(args.alex_site_map), _site_map(args.other_site_map)
    grams = sorted(Path(args.other).glob("*__gram_*.npz"))
    if not grams:
        raise ValueError("no other-network wiring matrices found")
    alex_wiring, alex_activation, other_activation = {}, {}, {}
    rows = []
    for gram in grams:
        network, other_layer = gram.name.split("__gram_", 1)
        other_layer = other_layer[:-4]
        other_wiring = None
        for alex_layer in ("conv5", "fc6", "fc7"):
            missing = []
            if alex_layer not in alex_map:
                missing.append("activation site not declared for AlexNet layer %s" % alex_layer)
            if other_layer not in other_map:
                missing.append("activation site not declared for %s layer %s" % (network, other_layer))
            if missing:
                for quantity, level in SLOTS:
                    row = _row(quantity, level, 0, network, alex_layer, other_layer, pairs)
                    row["reason"] = "; ".join(missing)
                    rows.append(row)
                continue
            if alex_layer not in alex_wiring:
                alex_wiring[alex_layer] = _load_excess(
                    Path(args.alex_grams) / ("gram_%s.npz" % alex_layer),
                    Path(args.alex_pairs) / ("pairs_%s.npz" % alex_layer))
            if other_wiring is None:
                other_wiring = _load_excess(gram, Path(args.other) / (
                    "%s__pairs_%s.npz" % (network, other_layer)))
            alex_site, other_site = alex_map[alex_layer], other_map[other_layer]
            if alex_site not in alex_activation:
                alex_activation[alex_site] = load_activation(args.alex_act_dir, alex_site)
            if other_site not in other_activation:
                other_activation[other_site] = load_activation(args.other_act_dir, other_site)
            common, aligned = align_matrices(alex_wiring[alex_layer], other_wiring,
                alex_activation[alex_site], other_activation[other_site])
            if keep is not None:
                index = np.flatnonzero(np.isin(common, keep))
                common, aligned = common[index], [m[np.ix_(index, index)] for m in aligned]
            rows.extend(analyze_pair_matrices(*aligned, common, tax, network=network,
                                             alexnet_layer=alex_layer, layer=other_layer, pairs=pairs))
    publish_file(pd.DataFrame(rows, columns=COLUMNS), out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error))
