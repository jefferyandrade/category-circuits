"""Section 2.3: AlexNet's wiring minus its mean category representations in superordinate tree share (W - A), with the
paired delete-one-category jackknife interval (tree_contrast.py --wiring-minus-representations).

The printed differences at fc6 and fc7 on all 1,000 categories and their intervals are read from
data/tree_share/wiring_minus_representations.tsv, with each interval recomputed from the row's estimate and standard
error; the table's W and A rows are those of Fig. 6's tables in data/tree_share. On planted matrices, the option adds
the W - A rows and nothing else, and --exclude-superordinate (the check without the mammals) is the same analysis on
the categories outside one superordinate group.
"""
import csv
import json
import math
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
TAXONOMY = REPO / "taxonomy" / "inputs" / "category_taxonomy.csv"
if str(REPO / "analysis" / "stats") not in sys.path:
    sys.path.insert(0, str(REPO / "analysis" / "stats"))
import tree_contrast as TC  # noqa: E402
import tree_share as TS  # noqa: E402

MAMMALS = "mammal.n.01"   # the mammals' superordinate group in the category tree


def rows(path, delimiter="\t"):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


DIFFERENCE = rows(DATA / "tree_share" / "wiring_minus_representations.tsv")
PRINTED = {r["macro"]: r["printed"] for r in rows(DATA / "printed_numbers.tsv")}


def same_as_printed(value, text):
    """True when value, rounded half up to the printed number of decimals, reads as the printed text."""
    text = text.replace(",", "")
    places = len(text.split(".")[1]) if "." in text else 0
    q = Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    return q == Decimal(text.lstrip("+")) and (not text.startswith("+") or q > 0)


def test_printed_differences_and_intervals():
    """At fc6 and fc7, on all 1,000 categories: the difference is the W row minus the A row, its interval is the
    difference +/- t(.975, 999) SE, and both round to the printed numbers (an interval above zero, as the asterisks of
    Fig. 6A show)."""
    assert [(r["layer"], r["lens"], r["level"]) for r in DIFFERENCE] == [
        (layer, lens, "superordinate") for layer in ("fc6", "fc7") for lens in ("A", "W", "W-A")]
    for layer, name in (("fc6", "FcSix"), ("fc7", "FcSeven")):
        share = {r["lens"]: r for r in DIFFERENCE if r["layer"] == layer}
        assert {r["n_categories"] for r in share.values()} == {PRINTED["CoreNAlex"].replace(",", "")}
        d = share["W-A"]
        difference, se, n = float(d["estimate"]), float(d["se"]), int(d["n_categories"])
        assert difference == float(share["W"]["estimate"]) - float(share["A"]["estimate"])
        width = t.ppf(.975, n - 1) * se
        assert math.isclose(float(d["ci_low"]), difference - width, rel_tol=1e-12)
        assert math.isclose(float(d["ci_high"]), difference + width, rel_tol=1e-12)
        for suffix, value in (("", difference), ("Lo", difference - width), ("Hi", difference + width)):
            assert same_as_printed(value, PRINTED["CapDiff" + name + suffix]), (name + suffix, value)
        assert float(d["ci_low"]) > 0


def test_shares_are_fig6s():
    """The W and A rows are those of Fig. 6's tables at fc6 and fc7, column for column."""
    for layer in ("fc6", "fc7"):
        fig6 = {(r["quantity"], r["lens"], r["level"]): r
                for r in rows(DATA / "tree_share" / ("tree_contrast_alexnet_%s.csv" % layer), delimiter=",")}
        for r in DIFFERENCE:
            if r["layer"] == layer and r["lens"] in ("W", "A"):
                reference = fig6[(r["quantity"], r["lens"], r["level"])]
                assert {k: r[k] for k in reference} == reference


def planted_layer(directory, categories, intersections, A, untrained):
    """tree_contrast.py's inputs for fc6 on the given categories: the circuits' kept-weight intersections (a gram),
    a constant reference overlap of 0.01, A as the mean representations' similarity and five untrained
    similarities."""
    directory.mkdir()
    k = len(categories)
    np.savez(directory / "gram_fc6.npz", categories=categories, I=intersections)
    np.savez(directory / "pairs_fc6.npz", categories=categories, J0_magnitude=np.full((k, k), .01))
    np.savez(directory / "matrices_classifier.2.npz", class_ids=categories, **{"S|cv:all": A})
    for s, matrix in enumerate(untrained):
        (directory / ("u%d" % s)).mkdir()
        np.savez(directory / ("u%d" % s) / "matrices_classifier.2.npz", class_ids=categories, **{"S|cv:all": matrix})
    (directory / "sites.json").write_text(json.dumps({"fc6": "classifier.2"}))
    inputs = SimpleNamespace(gram_dir=str(directory), pairs_dir=str(directory), trained_dir=str(directory),
                             site_map=str(directory / "sites.json"),
                             untrained_dirs=",".join(str(directory / ("u%d" % s)) for s in range(5)))
    argv = ["--layer", "fc6", "--taxonomy", str(TAXONOMY), "--check-deletions", "2", "--check-draws", "10"]
    for name, value in vars(inputs).items():
        argv += ["--" + name.replace("_", "-"), value]
    return argv, inputs


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    """30 mammals and 20 categories of four other superordinate groups, with the groups planted in the wiring and the
    representations; one of the 20 has no defined similarity to the mammals in A. tree_contrast.py runs on all 50 with
    --exclude-superordinate, and on inputs that hold only the 20 with and without --wiring-minus-representations."""
    folder = tmp_path_factory.mktemp("tree_difference")
    taxonomy = TS.load_tax(TAXONOMY)
    rng = np.random.default_rng(4)
    mammals = rng.choice(taxonomy.index[taxonomy.superordinate == MAMMALS], 30, replace=False)
    others = np.concatenate([rng.choice(taxonomy.index[taxonomy.superordinate == group], 5, replace=False)
                             for group in ("device.n.01", "conveyance.n.03", "commodity.n.01", "invertebrate.n.01")])
    categories = np.sort(np.r_[mammals, others])
    kept = np.isin(categories, others)
    codes = pd.factorize(taxonomy.superordinate[categories])[0]

    def similarity(strength):
        m = rng.normal(size=(50, 50)) * .1 + (codes[:, None] == codes[None, :]) * strength
        return (m + m.T) / 2

    intersections = rng.integers(200, 600, size=(50, 50))
    intersections = np.minimum(intersections, intersections.T) + (codes[:, None] == codes[None, :]) * 150
    np.fill_diagonal(intersections, rng.integers(3000, 5000, size=50))
    A, untrained = similarity(.3), [similarity(.05) for _ in range(5)]
    first = np.flatnonzero(kept)[0]
    A[first, ~kept] = A[~kept, first] = np.nan
    argv, inputs = planted_layer(folder / "all", categories, intersections, A, untrained)
    cut, _ = planted_layer(folder / "cut", categories[kept], intersections[np.ix_(kept, kept)],
                           A[np.ix_(kept, kept)], [u[np.ix_(kept, kept)] for u in untrained])
    for name, options in (("without", argv + ["--exclude-superordinate", MAMMALS]),
                          ("difference", cut + ["--wiring-minus-representations"]), ("default", cut)):
        assert TC.main(options + ["--out", str(folder / (name + ".csv"))]) == 0
    tables = {name: pd.read_csv(folder / (name + ".csv"), float_precision="round_trip")
              for name in ("without", "difference", "default")}
    gram = intersections[np.ix_(kept, kept)].astype(float)
    diagonal = np.diag(gram)
    matrices = {"W": gram / (diagonal[:, None] + diagonal[None, :] - gram) - .01, "A": A[np.ix_(kept, kept)]}
    labels = pd.factorize(taxonomy.superordinate[categories[kept]])[0]
    return dict(tables, argv=argv, inputs=inputs, excluded_category=categories[first], matrices=matrices,
                labels=labels, folder=folder)


def recomputed_difference(matrices, labels, null_means):
    """W - A at the superordinate level from the matrices, with tree_share's node adjustment and between-cell share,
    each share chance-corrected with its recorded null mean, and its paired delete-one-category jackknife interval."""
    n = len(labels)

    def share(name, keep):
        iu = np.triu_indices(keep.sum(), 1)
        values = TS.node_adjust(matrices[name][np.ix_(keep, keep)])[iu]
        groups = labels[keep].max() + 1
        observed = TS.between_cell_share(values, TS.cell_index(labels[keep], iu, groups), groups * groups)
        return (observed - null_means[name]) / (1 - null_means[name])

    deleted = np.asarray([share("W", np.arange(n) != i) - share("A", np.arange(n) != i) for i in range(n)])
    difference = share("W", np.ones(n, dtype=bool)) - share("A", np.ones(n, dtype=bool))
    se = np.sqrt((n - 1) / n * np.square(deleted - deleted.mean()).sum())
    width = t.ppf(.975, n - 1) * se
    return dict(estimate=difference, se=se, ci_low=difference - width, ci_high=difference + width)


def test_wiring_minus_representations_on_planted_matrices(planted):
    """The option adds one W - A row at each level and leaves every other row as the default run writes it; the default
    table keeps the format of data/tree_share. The superordinate W - A row is recomputed from the planted matrices."""
    with_difference, default = planted["difference"], planted["default"]
    added = with_difference.lens == "W-A"
    assert list(with_difference.level[added]) == ["domain", "superordinate", "block50"]
    assert set(with_difference.quantity[added]) == {"difference"} and "W-A" not in set(default.lens)
    pd.testing.assert_frame_equal(with_difference[~added].reset_index(drop=True), default[with_difference.columns],
                                  check_exact=True, check_dtype=False)
    assert list(default.columns) == list(pd.read_csv(DATA / "tree_share" / "tree_contrast_alexnet_fc6.csv").columns)
    assert list(with_difference.columns) == list(DIFFERENCE[0])
    row = with_difference[with_difference.level == "superordinate"].set_index("lens")
    expected = recomputed_difference(planted["matrices"], planted["labels"], row.perm_mean)
    for column, value in expected.items():
        assert math.isclose(row.loc["W-A", column], value, rel_tol=1e-9), (column, row.loc["W-A", column], value)
    assert row.loc["W-A", "estimate"] == row.loc["W", "estimate"] - row.loc["A", "estimate"]


def test_exclude_superordinate_on_planted_matrices(planted):
    """--exclude-superordinate cuts every matrix to the other categories before tree_contrast.load keeps the categories
    that every matrix defines: the category whose similarity to the 30 mammals is undefined is kept, while all 50
    categories would drop it. Its table is that of --wiring-minus-representations on inputs that hold only the 20
    other categories, row for row; a group the category tree lacks is refused."""
    assert planted["excluded_category"] not in TC.load(planted["inputs"], "fc6")[0]
    without = planted["without"]
    assert set(without.n_categories) == {20}
    pd.testing.assert_frame_equal(without, planted["difference"], check_exact=True)
    assert list(without.columns) == list(rows(DATA / "robustness" / "wiring_without_mammals.tsv")[0])
    with pytest.raises(ValueError, match="no superordinate group"):
        TC.main(planted["argv"] + ["--exclude-superordinate", "mammal", "--out", str(planted["folder"] / "typo.csv")])
