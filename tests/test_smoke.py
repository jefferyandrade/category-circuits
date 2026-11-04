"""Rebuild the category tree and the readout-weight control's summary table from the repository alone, byte for byte
(the sha256 values are those of the files the paper's analyses used)."""
import hashlib, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(*args):
    r = subprocess.run([sys.executable, "-I", *map(str, args)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_category_tree_rebuilds_byte_for_byte(tmp_path):
    """The WordNet category tree (32 superordinate groups, 131 subgroups) the analyses read."""
    out = tmp_path / "category_taxonomy.csv"
    run(REPO / "taxonomy" / "build_taxonomy.py", "--out", out)
    assert sha256(out) == "414494fff85fd3179c220d703187f5ff1e5990f94a593aa762d5057be5da217c"
    assert out.read_bytes() == (REPO / "taxonomy" / "inputs" / "category_taxonomy.csv").read_bytes()


def test_readout_control_summary_rebuilds_byte_for_byte(tmp_path):
    """The readout-weight control's table (SUMMARY_estimates.tsv) from its per-task outputs."""
    out = tmp_path / "SUMMARY_estimates.tsv"
    d = DATA / "readout_control"
    run(REPO / "analysis" / "readout_control" / "summarize.py", "--readout-out", d / "readout",
        "--outgeom-out", d / "output_geometry", "--out", out)
    assert sha256(out) == "7636b7055e0f2d2ff10d8bc254d96e5486220e571cbbfd8904e2efe5f4b785f1"
    assert out.read_bytes() == (d / "SUMMARY_estimates.tsv").read_bytes()


def test_frontier_analysis_loads_the_papers_settings():
    """The frontier comparison (Fig. 3A) loads its helper module and the shipped settings, checking their sha256."""
    script = REPO / "analysis" / "scripts" / "frontier_analysis.py"
    out = run("-c", "import importlib.util as u; s = u.spec_from_file_location('fa', %r); m = u.module_from_spec(s); "
                    "s.loader.exec_module(m); settings, categories = m.read_settings(m.SETTINGS, m.SETTINGS_SHA256); "
                    "print(len(categories), settings['bootstrap']['n_resamples'], settings['bootstrap']['seed'])" % str(script))
    assert out.split() == ["1000", "10000", "20261024"]
