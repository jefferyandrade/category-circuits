"""The robustness checks of sections 2.3 and 2.5.

Section 2.5's held-out check: Fig. 7B's comparison of the circuits' activations with the prototypes on the 900
categories outside the shared 100 (fc_all_images.py --population held_out). The printed numbers are read from
data/robustness/populations.tsv, whose rows are differences with normal jackknife intervals; the population and its
deletion groups are recomputed; and the comparison and its command line run on a planted extraction.

Section 2.3's check without the mammals: the superordinate tree share of the wiring minus that of the mean category
representations at fc6 and fc7, on the 782 categories outside the mammals' superordinate group (tree_contrast.py
--exclude-superordinate mammal.n.01). The printed count and the claim that both differences stay above zero are read
from data/robustness/wiring_without_mammals.tsv; test_tree_difference.py runs the option on planted matrices.
"""
import csv
import functools
import json
import math
import sys
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
CODE = REPO / "analysis" / "circuit_activations"
TAXONOMY = REPO / "taxonomy" / "inputs" / "category_taxonomy.csv"
for folder in (CODE, REPO / "analysis" / "common"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
import fc_all_images as F  # noqa: E402
import fc_statistics as S  # noqa: E402
from populations import shared100  # noqa: E402

MAMMALS = "mammal.n.01"   # the mammals' superordinate group in the category tree


def rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


TABLE = rows(DATA / "robustness" / "populations.tsv")
WIRING = rows(DATA / "robustness" / "wiring_without_mammals.tsv")
with open(DATA / "printed_numbers.tsv", newline="") as handle:
    PRINTED = {r["macro"]: r["printed"] for r in csv.DictReader(handle, delimiter="\t")}


def same_as_printed(value, text):
    """True when value, rounded half up to the printed number of decimals, reads as the printed text."""
    places = len(text.split(".")[1]) if "." in text else 0
    q = Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    return q == Decimal(text.lstrip("+")) and (not text.startswith("+") or q > 0)


def row(population, layer):
    [found] = [r for r in TABLE if (r["population"], r["layer"]) == (population, layer)]
    return found


def test_printed_numbers_come_from_the_table():
    """Section 2.5's held-out differences with their intervals, and the population's size."""
    for layer, name in (("fc6", "FcSix"), ("fc7", "FcSeven")):
        r = row("held_out", layer)
        for suffix, column in (("", "difference"), ("Lo", "ci_low"), ("Hi", "ci_high")):
            macro = "ResERobustHeldOut" + name + suffix
            assert same_as_printed(r[column], PRINTED[macro]), (macro, r[column], PRINTED[macro])
        assert r["n_categories"] == PRINTED["ResERobustHeldOutN"]


def test_table_rows_are_differences_with_normal_intervals():
    """Each row: the circuits' superordinate tree share minus the prototypes', +/- 1.96 jackknife SE, with the
    category-swap p from 1,000 swaps (a multiple of 1/1,001)."""
    assert list(TABLE[0]) == list(F.TABLE_COLUMNS)
    assert [(r["population"], r["layer"]) for r in TABLE] == [(p, layer) for p in F.POPULATIONS for layer in F.LAYERS]
    for r in TABLE:
        difference, se = float(r["difference"]), float(r["se"])
        assert difference == float(r["circuit"]) - float(r["prototypes"])
        assert (float(r["ci_low"]), float(r["ci_high"])) == (difference - 1.96 * se, difference + 1.96 * se)
        swaps = int(r["n_category_swaps"])
        extreme = float(r["p"]) * (swaps + 1) - 1
        assert swaps == 1000 and abs(extreme - round(extreme)) < 1e-9 and round(extreme) >= 0


def test_held_out_population():
    """held_out is every category outside the shared 100."""
    held_out = F.population_categories("held_out")
    assert len(held_out) == 900 and held_out == sorted(held_out)
    assert sorted(held_out + list(shared100())) == list(range(1000))
    assert {int(r["n_categories"]) for r in TABLE if r["population"] == "held_out"} == {len(held_out)}
    with pytest.raises(ValueError, match="unknown population"):
        F.population_categories("shared")


def test_deletion_groups():
    """Fixed random groups of ten (seed 0) that partition the population: 90 groups of ten for the held-out
    categories. At all 1,000 categories they are Fig. 7B's 100 groups of ten, and the shared 100 still delete single
    categories."""
    categories = F.population_categories("held_out")
    groups = S.jackknife_groups(categories, 0, groups_of_ten=True)
    assert Counter(map(len, groups)) == {10: 90}
    assert np.array_equal(np.sort(np.concatenate(groups)), np.arange(len(categories)))
    assert {int(r["n_jackknife_groups"]) for r in TABLE if r["population"] == "held_out"} == {len(groups)}
    order = np.random.default_rng(0).permutation(1000)
    for groups in (S.jackknife_groups(np.arange(1000)), S.jackknife_groups(np.arange(1000), 0, groups_of_ten=True)):
        assert len(groups) == 100 and all(np.array_equal(g, np.sort(r)) for g, r in zip(groups, order.reshape(100, 10)))
    assert [g.tolist() for g in S.jackknife_groups(np.arange(100))] == [[i] for i in range(100)]


def planted_extraction(directory, categories):
    """An extraction of the P circuits in the layout fc_all_images.load_run reads: the circuits' activations follow the
    category tree's superordinate groups, the prototypes do not."""
    directory.mkdir()
    rng = np.random.default_rng(5)
    superordinate = pd.read_csv(TAXONOMY).set_index("category").superordinate
    centers = {group: 3 * rng.random(8) for group in set(superordinate[categories])}
    for c in categories:
        record = {"category": np.asarray(c), "categories": np.asarray(categories)}
        for layer in F.LAYERS:
            circuit = {h: np.maximum(0, centers[superordinate[c]] + .3 * rng.normal(size=8)) for h in "AB"}
            prototype = {h: rng.random(8) for h in "AB"}
            circuit["mean"] = (circuit["A"] + circuit["B"]) / 2
            prototype["mean"] = (5 * prototype["A"] + 4 * prototype["B"]) / 9
            for h in F.HALVES:
                record["computation__%s__%s" % (layer, h)] = circuit[h]
                record["image__%s__%s" % (layer, h)] = prototype[h]
        np.savez(directory / ("category_%d.npz" % c), **record)
    confusion = np.zeros((len(categories), 1000), dtype=np.int64)
    for i in range(len(categories)):
        np.add.at(confusion[i], rng.choice(categories, size=9), 1)
    np.save(directory / "confusion_counts.npy", confusion)
    manifest = dict(schema=F.SCHEMA, source_schema=F.SOURCE_SCHEMA, categories=categories,
                    extracted_categories=categories, population="shared100", input_identity={"images": "planted"},
                    panel_sha256="0", probe=False, slice_counts=[9] * len(categories),
                    image_counts=[9] * len(categories), label="P", arm_kind="weight", run="first")
    (directory / "EXTRACT.json").write_text(json.dumps(manifest))
    (directory / "COMPLETE.json").write_text(json.dumps(dict(schema=F.SCHEMA, categories=categories)))


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    """A planted extraction of 100 categories and, as the population, those of them outside the shared 100 (92, in
    groups of nine or ten)."""
    categories = sorted(np.random.default_rng(2).choice(1000, size=100, replace=False).tolist())
    directory = tmp_path_factory.mktemp("planted") / "run"
    planted_extraction(directory, categories)
    shared = set(shared100())
    population = [c for c in categories if c not in shared]
    return directory, population, F.compare_population(directory, TAXONOMY, population, n_permutations=20)


def test_compare_population_on_a_planted_extraction(planted):
    """compare_population is paired_difference on the population's rows of the extraction, with groups of ten; the
    tree-structured circuits' activations exceed the unstructured prototypes, with an interval above zero."""
    directory, population, result = planted
    _, extracted, arrays, confusion = F.load_run(directory)
    taxonomy = pd.read_csv(TAXONOMY)
    selected = [extracted.index(c) for c in population]
    for layer in F.LAYERS:
        expected = S.paired_difference(arrays[layer]["computation"]["mean"][selected],
                                       arrays[layer]["image"]["mean"][selected], population, taxonomy,
                                       confusion["pair_rate"][np.ix_(selected, selected)], n_permutations=20,
                                       groups_of_ten=True)
        assert S.json_clean(result[layer]) == S.json_clean(expected)
        groups = result[layer]["deleted_category_groups"]
        assert len(groups) == math.ceil(len(population) / 10) and sorted(sum(groups, [])) == population
        sizes = " or ".join(map(str, sorted({len(g) for g in groups})))
        assert result[layer]["jackknife_unit"] == "%d fixed random groups of %s categories" % (len(groups), sizes)
        cell = result[layer]["metrics"]["tree_superordinate"]
        assert cell["difference"] == cell["left"] - cell["right"] and cell["jackknife"]["lo"] > 0
        assert cell["category_swap"]["n_permutations"] == 20
    missing = next(c for c in range(1000) if c not in extracted)
    with pytest.raises(ValueError, match="lacks some of the population"):
        F.compare_population(directory, TAXONOMY, sorted(population + [missing]), n_permutations=1)


def test_population_command_line_writes_the_table(planted, tmp_path, monkeypatch):
    """fc_all_images.py --population writes one row per population and layer with the shipped table's columns (here
    the planted population and 20 category swaps); it does not take the P-redraw circuits or overwrite a table."""
    directory, population, result = planted
    monkeypatch.setattr(F, "population_categories", lambda name: population)
    monkeypatch.setattr(F, "compare_population", functools.partial(F.compare_population, n_permutations=20))
    out = tmp_path / "populations.tsv"
    argv = ["--weight", str(directory), "--taxonomy", str(TAXONOMY), "--out", str(out), "--population", "held_out"]
    assert F.main(argv) == 0
    written = rows(out)
    assert list(written[0]) == list(F.TABLE_COLUMNS)
    assert written == [{k: str(v) for k, v in r.items()} for r in F.population_rows("held_out", result)]
    with pytest.raises(FileExistsError):
        F.main(argv)
    with pytest.raises(SystemExit):
        F.main(argv[:-4] + ["--out", str(tmp_path / "other.tsv"), "--weight-second", str(directory),
                            "--population", "held_out"])


# Section 2.3: the wiring against the mean category representations without the mammals ------------------------------

def test_wiring_difference_without_mammals_stays_above_zero():
    """At fc6 and fc7, on the printed 782 categories (all but the 218 of the mammals, the largest superordinate group),
    the wiring's superordinate tree share minus the mean category representations' is the W row minus the A row, with
    the paired jackknife interval (difference +/- t(.975, 781) SE) above zero."""
    groups = pd.read_csv(TAXONOMY).superordinate.value_counts()
    assert groups.index[0] == MAMMALS and groups.iloc[0] == 218 > groups.iloc[1]
    assert int(PRINTED["ResERobustNoMammalN"]) == 1000 - 218
    assert [(r["layer"], r["lens"]) for r in WIRING] == [(layer, lens) for layer in ("fc6", "fc7")
                                                        for lens in ("A", "W", "W-A")]
    for layer in ("fc6", "fc7"):
        share = {r["lens"]: r for r in WIRING if r["layer"] == layer}
        assert {(r["level"], r["n_categories"]) for r in share.values()} == {
            ("superordinate", PRINTED["ResERobustNoMammalN"])}
        d = share["W-A"]
        difference, se = float(d["estimate"]), float(d["se"])
        assert difference == float(share["W"]["estimate"]) - float(share["A"]["estimate"])
        width = t.ppf(.975, int(d["n_categories"]) - 1) * se
        assert math.isclose(float(d["ci_low"]), difference - width, rel_tol=1e-12)
        assert math.isclose(float(d["ci_high"]), difference + width, rel_tol=1e-12)
        assert float(d["ci_low"]) > 0
