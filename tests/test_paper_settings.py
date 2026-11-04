"""The extraction's defaults are the settings of the paper's circuits.

The engine's entry points fill in every setting the environment leaves unset from
engine/experiments/irg_v0_v2_pilots/paper_settings.py, and their option defaults select the paper's method.
These tests need no GPU and do not import the engine (which needs torch): the option defaults are read from the
source.
"""
import ast
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PILOTS = REPO / "engine" / "experiments" / "irg_v0_v2_pilots"

spec = importlib.util.spec_from_file_location("paper_settings", PILOTS / "paper_settings.py")
paper_settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper_settings)

NETWORKS = {
    "alexnet": "torchvision.alexnet.conv_kernel_weight.imagenet1k_v1",
    "vgg11": "torchvision.vgg11.conv_kernel_weight.imagenet1k_v1",
    "resnet18": "torchvision.resnet18.conv_kernel_weight.imagenet1k_v1",
    "resnet50v1": "torchvision.resnet50.conv_kernel_weight.imagenet1k_v1",
}
PILOTED_ON = {"resnet50v1": "resnet50v2"}  # ResNet-50 took the threshold piloted on its IMAGENET1K_V2 checkpoint (Methods)


def test_unset_settings_are_filled_and_explicit_ones_kept():
    environment = {"IRG_CALIBRATION_PENALTY": "identity_anchor"}
    filled = paper_settings.apply(environment, NETWORKS["vgg11"])
    assert environment["IRG_CALIBRATION_PENALTY"] == "identity_anchor"
    assert "IRG_CALIBRATION_PENALTY" not in filled
    assert environment["IRG_CENSUS_PRECISION_EPS"] == "0"
    assert environment["IRG_SENSITIVITY_TAU"] == "1.768e-07"
    assert environment["IRG_CENSUS_BATCH_SIZE"] == "512"
    assert paper_settings.apply(environment, NETWORKS["vgg11"]) == []


def test_the_methods_settings():
    shared = paper_settings.SHARED
    assert shared["IRG_CALIBRATION_PENALTY"] == "firth"
    assert shared["IRG_LOSS_WEIGHTING"] == "balanced"
    assert shared["IRG_CENSUS_PRECISION_EPS"] == "0"
    assert shared["IRG_TERMINAL_CALIBRATION_RULE"] == "walk_resolution"


def test_default_thresholds_are_table_1s():
    """Each network's default threshold is the one price_match.py selects from the pilot tables."""
    r = subprocess.run([sys.executable, "-I", str(REPO / "analysis" / "scripts" / "price_match.py"),
                        str(REPO / "data" / "thresholds" / "pilots_round1.csv"),
                        str(REPO / "data" / "thresholds" / "pilots_round2.csv")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "anchor AlexNet 1.5e-7" in r.stdout
    chosen = {"alexnet": 1.5e-7}
    for line in r.stdout.splitlines():
        if line.strip().startswith("->"):
            net, rest = line.strip()[2:].split(":", 1)
            chosen[net.strip()] = float(rest.split("price")[1].split()[0])
    for net, model_id in NETWORKS.items():
        assert float(paper_settings.PER_NETWORK[model_id]["IRG_SENSITIVITY_TAU"]) == chosen[PILOTED_ON.get(net, net)], net


def _option_defaults(path):
    defaults = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument" and node.args:
            keywords = {k.arg: k.value for k in node.keywords}
            if "default" in keywords and isinstance(node.args[0], ast.Constant):
                defaults[node.args[0].value] = ast.unparse(keywords["default"])
    return defaults


def test_entry_point_defaults_select_the_papers_method():
    defaults = _option_defaults(PILOTS / "run_registered_resnet18_arm.py")
    assert defaults["--version"] == "'v0'"
    assert defaults["--partition-family"] == "'local_attribution_sign'"
    assert defaults["--step-category-matrix"] == "'off'"
    assert defaults["--require-optimized-native-panel"] == "True"
    assert defaults["--v0-attribution-order"] == "V0_DEFAULT_ATTRIBUTION_ORDER"
    assert defaults["--prune-order-consequence-gated"] == "True"
    assert defaults["--measurement-artifact-policy"] == "'contracts_only'"
    pilot = (REPO / "engine" / "mechanistic_systems" / "circuits" / "irg" / "registered_resnet18_pilot.py").read_text()
    assert re.search(r'V0_DEFAULT_ATTRIBUTION_ORDER = \(\s*"plain_global_attribution"\s*\)', pilot)


def test_analysis_defaults_are_the_papers_runs():
    """The statistics read the paper's circuits and use the paper's resampling settings by default."""
    stats = REPO / "analysis" / "stats"
    for script in ("arms.py", "sign_density.py", "cross_network_agreement.py"):
        defaults = _option_defaults(stats / script)
        assert defaults["--price"] == "'v66'", script
        assert defaults["--allow-partial"] == "True", script
    for script in ("tree_contrast.py", "tree_infer.py"):
        defaults = _option_defaults(stats / script)
        assert (defaults["--nperm-test"], defaults["--seed"], defaults["--check-deletions"],
                defaults["--check-draws"]) == ("1000", "0", "20", "2000"), script
