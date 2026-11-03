"""Fig. 3B: ResNet-50, E/I-separate (P) against E/I-pooled (SA-H) pruning, recomputed from the per-category table, and
the step that reads that table from the circuits' receipts, run on a small planted library."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "resnet50_pooled"
sys.path.insert(0, str(REPO / "analysis" / "stats"))
import resnet50_pooled as rp  # noqa: E402


def test_fig3b_table_recomputes_byte_for_byte(tmp_path):
    out = tmp_path / "values_fig3B_resnet50.csv"
    assert rp.main(["table", "--per-category", str(DATA / "per_category.csv"), "--out", str(out)]) == 0
    assert out.read_bytes() == (DATA / "values_fig3B_resnet50.csv").read_bytes()


def test_per_category_table_covers_the_96_complete_pairs():
    by_arm = rp.read_per_category(DATA / "per_category.csv")
    assert sorted(by_arm["P"]) == [c for c in rp.CATEGORIES if c not in (20, 30, 333)]
    assert len(by_arm["SA-H"]) == 96
    # The dense network is the same for both methods.
    assert all(by_arm["P"][c]["dense_balanced_accuracy"] == by_arm["SA-H"][c]["dense_balanced_accuracy"]
               for c in by_arm["P"])


def plant(results, arm, category, kept, dense, circuit, tau=rp.TAU, written_last=True):
    folder = results / rp.CATEGORY_FOLDER.format(category=category) / rp.METHOD_FOLDERS[arm]
    folder.mkdir(parents=True)
    receipt = {"query_id": str(category), "sensitivity_tau": tau, "partition_family": rp.PARTITIONS[arm],
               "retained_fraction": kept, "run_products": {"receipt_written_last": written_last},
               "terminal_dense_bce": {"class_balanced_binary_accuracy": dense},
               "terminal_final_bce": {"class_balanced_binary_accuracy": circuit}}
    (folder / "receipt.json").write_text(json.dumps(receipt))


def test_receipts_step_reads_complete_pairs(tmp_path):
    p, sah = tmp_path / "p", tmp_path / "sah"
    plant(p, "P", 0, 0.25, 0.99, 0.97)
    plant(sah, "SA-H", 0, 0.375, 0.99, 0.875)
    plant(p, "P", 10, 0.5, 0.75, 0.625)
    plant(p, "P", 20, 0.25, 0.99, 0.97)  # no SA-H circuit
    plant(sah, "SA-H", 10, 0.5, 0.75, 0.5, written_last=False)  # an extraction that did not finish
    rows, missing = rp.per_category_rows({"P": p, "SA-H": sah})
    assert rows == [dict(category=0, arm="P", kept_fraction=0.25, dense_balanced_accuracy=0.99,
                         circuit_balanced_accuracy=0.97),
                    dict(category=0, arm="SA-H", kept_fraction=0.375, dense_balanced_accuracy=0.99,
                         circuit_balanced_accuracy=0.875)]
    assert missing[10] == ["SA-H"] and missing[20] == ["SA-H"] and missing[40] == ["P", "SA-H"]
    out = tmp_path / "per_category.csv"
    rp.write_csv(out, rp.PER_CATEGORY_FIELDS, rows)
    assert rp.read_per_category(out)["SA-H"][0]["circuit_balanced_accuracy"] == 0.875


def test_receipts_step_refuses_a_receipt_of_another_threshold(tmp_path):
    plant(tmp_path / "p", "P", 0, 0.25, 0.99, 0.97, tau=1.21e-6)
    with pytest.raises(ValueError, match="threshold"):
        rp.per_category_rows({"P": tmp_path / "p", "SA-H": tmp_path / "sah"})
