"""Does the wiring predict which categories the network confuses, beyond its representations? Its confusability
matrix is also the output geometry of the output-geometry control (section 2.4; make_logit_geometry.py).

Confusability: the dense network's mean logit of class j on images of class i (`moment_mean`, the default), or the
one-vs-all loss of logit j on those images (`bce_matrix`), from the category_matrix_dense.npz of any one of the
network's circuits (the dense network is the same for every circuit; 50 training images per class). Both are
symmetrized, without the diagonal, with higher values meaning more confusable. Node effects are removed inside
cross_network_adjusted.pair_correlation, as for every pair matrix there.
For each layer with a wiring matrix and an activation site, two rows from
cross_network_adjusted.analyze_pair_matrices (with the covariate passed in both covariate slots):
  predictor=wiring:      x = wiring excess overlap, y = confusability, covariate = representation similarity
  predictor=activations: x = representation similarity, y = confusability, covariate = wiring excess overlap
so r_beyond_activations of the two rows compares each predictor beyond the other (r_raw is the plain correlation;
r_beyond_tree also removes the category tree). Descriptive; joint delete-one-category jackknife; the categories are
those in --categories (the shared 100) when given, and the pairs those of --pairs. Rows record the matrix key and the
pair subset.

Usage: confusion_wiring.py --network NET --gram-dir DIR --pairs-dir DIR --act-dir DIR --site-map JSON
           --category-matrix category_matrix_dense.npz --taxonomy CSV --out CSV [--categories FILE]
           [--pairs within_kind]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tree_share as TS  # noqa: E402
import cross_network_adjusted as adjusted  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import publish_file, check_outputs  # noqa: E402


def confusability(path, matrix_key="moment_mean"):
    """The categories and the symmetrized confusability matrix, diagonal NaN."""
    if matrix_key not in ("moment_mean", "bce_matrix"):
        raise ValueError("matrix_key must be 'moment_mean' or 'bce_matrix'")
    with np.load(path) as z:
        rows, cols, M = np.asarray(z["stratum_order"]), np.asarray(z["logit_order"]), np.asarray(z[matrix_key], float)
    if not (np.array_equal(np.sort(rows), np.sort(cols)) and M.shape == (rows.size, cols.size)):
        raise ValueError("category matrix rows and logit columns differ")
    order = np.argsort(rows)
    M = M[np.ix_(order, np.argsort(cols))]
    C = (M + M.T) / 2
    np.fill_diagonal(C, np.nan)
    return np.sort(rows).astype(np.int64), C


def restrict(data, keep):
    """The categories and matrix restricted to the categories in keep."""
    cats, M = data
    idx = np.flatnonzero(np.isin(cats, keep))
    return cats[idx], M[np.ix_(idx, idx)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("network", "gram-dir", "pairs-dir", "act-dir", "site-map", "category-matrix", "taxonomy", "out"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--categories", help="file of category ids to keep (e.g. the shared 100)")
    ap.add_argument("--pairs", choices=("all", "within_kind", "between_kind"), default="all")
    ap.add_argument("--matrix-key", choices=("moment_mean", "bce_matrix"), default="moment_mean")
    a = ap.parse_args(argv)
    pair_subset = None if a.pairs == "all" else a.pairs
    out = Path(a.out)
    check_outputs(files=[out], inputs=[a.gram_dir, a.pairs_dir, a.act_dir, a.site_map, a.category_matrix, a.taxonomy, a.categories])
    site_map = json.loads(Path(a.site_map).read_text())
    tax = TS.load_tax(a.taxonomy)
    keep = np.array(sorted(int(x) for x in Path(a.categories).read_text().split())) if a.categories else None
    conf = confusability(a.category_matrix, matrix_key=a.matrix_key)
    rows = []
    for layer, site in site_map.items():
        gram, pairs = Path(a.gram_dir) / ("gram_%s.npz" % layer), Path(a.pairs_dir) / ("pairs_%s.npz" % layer)
        if not (gram.is_file() and pairs.is_file()):
            continue
        data = [adjusted._load_excess(gram, pairs), adjusted.load_activation(a.act_dir, site), conf]
        if keep is not None:
            data = [restrict(d, keep) for d in data]
        common, (w, act, c) = adjusted.align_matrices(*data)
        for predictor, x, cov in (("wiring", w, act), ("activations", act, w)):
            result = adjusted.analyze_pair_matrices(x, c, cov, cov, common, tax, network=a.network,
                                                    alexnet_layer="", layer=layer, pairs=pair_subset)
            for row in result:
                row.update(predictor=predictor, site=site, matrix_key=a.matrix_key, pairs=a.pairs)
            rows.extend(result)
        print("%s done (%d categories)" % (layer, len(common)), flush=True)
    if not rows:
        raise ValueError("no layer has both a wiring matrix and an activation site")
    publish_file(pd.DataFrame(rows).drop(columns=["alexnet_layer"]), out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error))
