"""Finished analysis outputs survive a rerun, including a mistaken input alias."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for folder in ("common", "stats", "input_divergence", "circuit_activations", "animacy"):
    sys.path.insert(0, str(REPO / "analysis" / folder))


def test_alexnet_intervals_refuses_before_reading(tmp_path, monkeypatch):
    import alexnet_intervals as script
    out = tmp_path / "out"
    out.mkdir()
    table = out / "values_fig4C_alexnet_all1000.csv"
    table.write_bytes(b"finished\n")
    monkeypatch.setattr(script, "CENSUS", tmp_path / "missing")
    monkeypatch.setattr(sys, "argv", ["alexnet_intervals.py", "--out", str(out)])
    with pytest.raises(FileExistsError, match=str(out)):
        script.main()
    assert table.read_bytes() == b"finished\n"


def test_resnet50_pooled_rerun_preserves_table(tmp_path):
    import resnet50_pooled as script
    table = tmp_path / "per_category.csv"
    script.write_csv(table, script.PER_CATEGORY_FIELDS, [])
    before = table.read_bytes()
    with pytest.raises(FileExistsError, match=str(table)):
        script.main(["receipts", "--p-results", str(tmp_path / "missing-p"),
                     "--sah-results", str(tmp_path / "missing-sah"), "--out", str(table)])
    assert table.read_bytes() == before


def test_fig7_values_rerun_preserves_table(tmp_path):
    import fig7_values as script
    table = tmp_path / "values.csv"
    script.write(table, [dict(panel="C", value=0.25)])
    before = table.read_bytes()
    with pytest.raises(FileExistsError, match=str(table)):
        script.main(["C", "--partial-shares", str(tmp_path / "missing.json"), "--out", str(table)])
    assert table.read_bytes() == before


@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
def test_block_means_refuses_input_alias(tmp_path, alias):
    import block_means as script
    source = tmp_path / "values.csv"
    source.write_text("panel,network,layer,value,relative_depth\nC,resnet18,stem.conv,0.5,0\n")
    out = source if alias == "same" else tmp_path / "alias.csv"
    if alias == "symlink":
        out.symlink_to(source)
    elif alias == "hardlink":
        import os
        os.link(source, out)
    before = source.read_bytes()
    with pytest.raises(FileExistsError, match=str(out)):
        script.main(["--values", str(source), "--out", str(out)])
    assert source.read_bytes() == before


def test_block_means_new_output(tmp_path):
    import block_means as script
    source, out = tmp_path / "values.csv", tmp_path / "means.csv"
    source.write_text("panel,network,layer,value,relative_depth\nC,resnet18,stem.conv,0.5,0\n")
    script.main(["--values", str(source), "--out", str(out)])
    with out.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["value"] == "0.5"


def test_publish_refuses_a_file_created_during_writing(tmp_path):
    from atomic_io import write_file
    out = tmp_path / "result.csv"
    with pytest.raises(FileExistsError):
        with write_file(out) as stream:
            stream.write("new result\n")
            out.write_bytes(b"finished by another run\n")
    assert out.read_bytes() == b"finished by another run\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == [out.name]


def test_failed_write_leaves_no_final_or_temporary_file(tmp_path):
    from atomic_io import write_file
    out = tmp_path / "result.csv"
    with pytest.raises(RuntimeError, match="interrupted"):
        with write_file(out) as stream:
            stream.write("unfinished\n")
            raise RuntimeError("interrupted")
    assert list(tmp_path.iterdir()) == []


def test_empty_folder_and_overlapping_destinations(tmp_path):
    from atomic_io import check_outputs, prepare_dir, publish_file
    folder = tmp_path / "out"
    folder.mkdir()
    prepare_dir(folder)
    publish_file("finished\n", folder / "table.csv")
    with pytest.raises(FileExistsError, match=str(folder)):
        prepare_dir(folder)
    with pytest.raises(SystemExit, match="overlap"):
        check_outputs(files=[tmp_path / "new.csv", tmp_path / "new.csv"])
    with pytest.raises(SystemExit, match="input"):
        check_outputs(dirs=[tmp_path / "input"], inputs=[tmp_path / "input" / "table.csv"])


@pytest.mark.parametrize("script,options,folder", [
    ("common/block_means.py", ["--values", "missing.csv"], False),
    ("stats/resnet50_pooled.py", ["table", "--per-category", "missing.csv"], False),
    ("circuit_activations/fig7_values.py", ["C", "--partial-shares", "missing.json"], False),
    ("input_divergence/alexnet_intervals.py", [], True),
    ("input_divergence/alexnet_references.py", [], True),
    ("input_divergence/census.py", ["--network", "alexnet", "--layer", "readout.hidden1",
                                   "--grain", "unit_fanin", "--masks-root", "missing"], True),
    ("animacy/run_alexnet_animacy.py", ["tables", "--statistics", "missing"], True),
    ("animacy/run_shared_animacy.py", ["--incidence-package", "missing"], True),
    ("circuit_activations/fc_all_images.py", ["--weight", "missing", "--taxonomy", "missing.csv"], False),
    ("circuit_activations/whole_circuits.py", ["--run", "missing", "--taxonomy", "missing.csv"], False),
    ("circuit_activations/partial_tree_shares.py", ["--run", "missing", "--wiring", "missing",
                                                 "--taxonomy", "missing.csv"], False),
    ("readout_control/summarize.py", ["--readout-out", "missing", "--outgeom-out", "missing"], False),
    ("stats/table1_stats.py", [], False),
    ("stats/method_contrasts.py", [], False),
    ("stats/size_stats.py", [], False),
    ("stats/size_adapter.py", ["--lenses", "missing"], False),
    ("stats/sign_density.py", ["--input-csv", "missing.csv"], True),
    ("stats/sign_magnitude.py", ["--checkpoint", "missing.pth"], True),
    ("scripts/parting_controls.py", ["--parting", "missing.csv", "--reliability", "missing.csv"], False),
    ("scripts/frontier_analysis.py", ["--inputs", "missing"], True),
    ("animacy/collect_shared_layers.py", ["--layers", "missing.json", "--density", "missing.tsv",
                                        "--checks", "missing-checks.json"], False),
    ("circuit_activations/layer_type_tests.py", ["width", "--activations", "missing"], False),
    ("circuit_activations/wiring_minus_prototypes.py", ["--plan", "missing.json", "--resnet50-plan",
                                                     "missing.json", "--panel", "missing.csv"], True),
    ("readout_control/make_readout_geometry.py", ["--checkpoint", "missing.pth", "--key", "fc.weight",
                                               "--like", "missing.npz", "--gram-dir", "missing"], True),
    ("stats/make_logit_geometry.py", ["--category-matrix", "missing.npz", "--gram-dir", "missing"], True),
    ("stats/overlap_nulls.py", [], True),
    ("stats/tree_share.py", ["wiring"], False),
    ("stats/tree_contrast.py", ["--layer", "fc6", "--gram-dir", "missing", "--pairs-dir", "missing",
                                "--trained-dir", "missing", "--site-map", "missing.json",
                                "--taxonomy", "missing.csv"], False),
    ("stats/tree_infer.py", ["--layer", "fc6", "--gram-dir", "missing", "--pairs-dir", "missing",
                            "--matrices-dir", "missing", "--site-map", "missing.json",
                            "--taxonomy", "missing.csv"], False),
    ("stats/confusion_wiring.py", ["--network", "alexnet", "--gram-dir", "missing", "--pairs-dir", "missing",
                                  "--act-dir", "missing", "--site-map", "missing.json",
                                  "--category-matrix", "missing.npz", "--taxonomy", "missing.csv"], False),
    ("stats/cross_network_agreement.py", ["--alex-grams", "missing", "--alex-pairs", "missing",
                                         "--other", "missing", "--taxonomy", "missing.csv"], False),
    ("stats/cross_network_adjusted.py", ["--alex-grams", "missing", "--alex-pairs", "missing",
                                        "--other", "missing", "--alex-act-dir", "missing",
                                        "--other-act-dir", "missing", "--alex-site-map", "missing.json",
                                        "--other-site-map", "missing.json", "--taxonomy", "missing.csv"], False),
])
def test_command_refuses_before_reading_missing_inputs(tmp_path, script, options, folder):
    out = tmp_path / "out"
    if folder:
        out.mkdir()
        result = out / "finished.csv"
    else:
        result = out
    result.write_bytes(b"finished\n")
    run = subprocess.run([sys.executable, str(REPO / "analysis" / script), *options, "--out", str(out)],
                         capture_output=True, text=True)
    assert run.returncode != 0
    assert str(out) in run.stderr
    assert "Traceback" not in run.stderr
    assert result.read_bytes() == b"finished\n"


def test_collect_shared_layers_checks_both_outputs_before_reading(tmp_path, monkeypatch):
    import collect_shared_layers as script
    table, checks = tmp_path / "new.csv", tmp_path / "checks.json"
    checks.write_text("finished\n")
    monkeypatch.setattr(sys, "argv", ["collect_shared_layers.py", "--layers", "missing.json",
                                     "--density", "missing.tsv", "--out", str(table), "--checks", str(checks)])
    with pytest.raises(FileExistsError, match=str(checks)):
        script.main()
    assert not table.exists()
    assert checks.read_text() == "finished\n"


def test_partial_tree_shares_resumes_cells_and_publishes_once(tmp_path, monkeypatch):
    import numpy as np
    import partial_tree_shares as script
    calls = []

    class Cell:
        def __init__(self, *args):
            pass

        def compute(self, *args):
            calls.append(len(calls))
            if len(calls) == 2:
                raise RuntimeError("interrupted")
            return {"adjusted": 0.25, "lo": 0.125, "hi": 0.375}

    arrays = {layer: {name: {"mean": np.zeros((1000, 2))} for name in ("computation", "image")}
              for layer in ("fc6", "fc7")}
    monkeypatch.setattr(script, "self_test", lambda: {"passed": True})
    monkeypatch.setattr(script.fc_all_images, "load_run", lambda path: ({}, np.arange(1000), arrays, None))
    monkeypatch.setattr(script, "load_wiring", lambda *args: (np.eye(1000), []))
    monkeypatch.setattr(script, "pearson", lambda array: np.eye(1000))
    monkeypatch.setattr(script, "TreeShareCell", Cell)
    out = tmp_path / "shares.json"
    argv = ["--run", str(tmp_path / "run"), "--wiring", str(tmp_path / "wiring"),
            "--taxonomy", str(REPO / "taxonomy/inputs/category_taxonomy.csv"), "--out", str(out)]
    with pytest.raises(RuntimeError, match="interrupted"):
        script.main(argv)
    assert not out.exists()
    progress = out.with_name(out.name + ".cells")
    saved = {p: p.read_bytes() for p in progress.glob("*.json")}
    assert len(saved) == 2  # configuration and one completed cell
    assert script.main(argv) == 0
    assert all(p.read_bytes() == value for p, value in saved.items())
    assert len(calls) == 43  # 42 cells and the interrupted attempt
    assert len(json.loads(out.read_text())["layers"]) == 2
    before = out.read_bytes()
    with pytest.raises(FileExistsError, match=str(out)):
        script.main(argv)
    assert out.read_bytes() == before


@pytest.mark.parametrize("alias", ["same", "hardlink"])
def test_resumable_layer_refuses_a_result_that_is_an_input(tmp_path, alias):
    import layer_activations as script
    source = tmp_path / "taxonomy.csv"
    source.write_bytes(b"input\n")
    out = source
    if alias == "hardlink":
        import os
        out = tmp_path / "alias.json"
        os.link(source, out)
    with pytest.raises(ValueError, match="input"):
        script.main(["aggregate", "--network", "alexnet", "--out", str(tmp_path / "out"),
                     "--site", "fc6", "--taxonomy", str(source), "--result", str(out)])
    assert source.read_bytes() == b"input\n"


def test_resumable_merge_refuses_a_result_that_is_an_input(tmp_path):
    import layer_activations as script
    source = tmp_path / "run/alexnet/MERGED.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"input\n")
    with pytest.raises(ValueError, match="input"):
        script.main(["merge", "--network", "alexnet", "--out", str(tmp_path), "--prototypes", str(source)])
    assert source.read_bytes() == b"input\n"


def test_resumable_cell_refuses_an_input_alias_before_calculation(tmp_path, monkeypatch):
    import partial_tree_shares as script
    source, out = tmp_path / "taxonomy.csv", tmp_path / "out.json"
    source.write_bytes(b"input\n")
    progress = out.with_name(out.name + ".cells")
    progress.mkdir()
    (progress / "fc6__computation__domain.json").symlink_to(source)
    monkeypatch.setattr(script, "self_test", lambda: pytest.fail("calculation started"))
    with pytest.raises(ValueError, match="input"):
        script.main(["--run", str(tmp_path / "run"), "--wiring", str(tmp_path / "wiring"),
                     "--taxonomy", str(source), "--out", str(out)])
    assert source.read_bytes() == b"input\n"
    assert not out.exists()
