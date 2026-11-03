"""Where the circuits are, and which receipt fields the statistics read.

arms_config.json names, for each network, pruning method ("arm") and set of circuits (`price`), the circuit library's
results folder and each category's circuit folder, and the categories the method was run on (its cohort). The
`price` v66 selects the paper's circuits, extracted at each network's loss threshold tau (Table 1); v64 and v65 are
earlier extractions. The config also maps each measure to a receipt field (a circuit's kept fraction, its accuracy
drop, ...). This module reads circuits' receipts into per-category tables, checks that each method's cohort is
complete, and names the pair of runs (P and its repeat extraction) that measures extraction noise. Masks are read
by circuit_masks.load_verified_mask.
"""
from __future__ import annotations

import argparse
import filecmp
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CONFIG = Path(__file__).with_name("arms_config.json")
ALIASES = {"SAH": "SA-H", "P_redraw": "P-redraw"}       # the names the sign-density tables use


class InputNotAvailable(ValueError):
    """An input named in arms_config.json is absent, or the config does not name it."""


def _inside(path):
    """The absolute path of an input path from the config."""
    return Path(path).expanduser().resolve()


def load_config(path=None):
    """Read arms_config.json (or another config of the same form)."""
    path = _inside(DEFAULT_CONFIG if path is None else path)
    config = json.loads(path.read_text())
    if config.get("version") != 1 or not isinstance(config.get("networks"), dict):
        raise ValueError("arms configuration requires version 1 and a networks object")
    root = config.get("workspace_root", ".")
    root = Path(root).expanduser()
    config["_root"] = _inside(root if root.is_absolute() else path.parent / root)
    config["_path"] = path
    return config


def local_path(config, value, require_exists=True):
    if not value:
        raise InputNotAvailable("local input path is not configured")
    value = Path(value).expanduser()
    path = _inside(value if value.is_absolute() else config["_root"] / value)
    if require_exists and not path.exists():
        raise InputNotAvailable("local input is absent: %s" % path)
    return path


def input_path(config, key, require_exists=True):
    try:
        value = config["inputs"][key]
    except KeyError:
        raise InputNotAvailable("input %r is not configured" % key)
    return local_path(config, value, require_exists)


def arm_record(config, network, arm, price="v66", run="primary"):
    """The config's entry for one run of a method: its results folder, circuit folder name and cohort."""
    arm = ALIASES.get(arm, arm)
    try:
        net = config["networks"][network]
        record = dict(net["arms"][arm]["runs"][price][run])
    except KeyError:
        raise InputNotAvailable("arm is not configured: %s/%s/%s/%s" %
                                (network, arm, price, run))
    if not record.get("arm_folder"):
        raise InputNotAvailable("arm folder is not supplied: %s/%s/%s/%s" %
                                (network, arm, price, run))
    folder = Path(record["arm_folder"])
    if folder.is_absolute() or len(folder.parts) != 1 or folder.name in (".", ".."):
        raise ValueError("arm_folder must be a single local folder name")
    record["cohort"] = record.get("cohort", net["arms"][arm].get("cohort", net.get("cohort", "all")))
    record["results_root"] = local_path(config, record.get("results_root"), False)
    record["category_pattern"] = net.get("category_pattern", network + "_tap_query{category}")
    return record


def resolve_arm(config, network, arm, price="v66", run="primary", category=None, require_exists=True):
    """The results folder of a run, or with a category the folder of that category's circuit."""
    record = arm_record(config, network, arm, price, run)
    path = record["results_root"]
    if category is not None:
        path = _inside(path / record["category_pattern"].format(category=int(category)) /
                       record["arm_folder"])
    if require_exists and not path.is_dir():
        raise InputNotAvailable("local arm directory is absent: %s" % path)
    return path


def add_config_arguments(parser):
    """The command-line options shared by the statistics that read circuits through arms_config.json."""
    parser.add_argument("--arms-config", type=Path, default=DEFAULT_CONFIG,
                        help="where the circuits are (default: arms_config.json beside this file)")
    parser.add_argument("--price", default="v66", choices=("v64", "v65", "v66"),
                        help="which circuits to read: v66, the default, is the paper's circuits, at each network's "
                             "loss threshold (Table 1); v64 and v65 are earlier extractions")
    parser.add_argument("--run", default="primary", help="which run of each method to read (default: primary)")
    parser.add_argument("--unit", choices=("category", "block50", "superordinate", "domain"), default="category",
                        help="the resampling unit (default: category, as in the paper)")
    parser.add_argument("--allow-partial", action=argparse.BooleanOptionalAction, default=True,
                        help="accept a method whose cohort is incomplete and mark every result row as partial "
                             "(the default)")
    return parser


def load_categories(config):
    """The category tree (taxonomy) table, one row per category, sorted by category."""
    frame = pd.read_csv(input_path(config, "taxonomy"))
    if not {"category", "block50"} <= set(frame):
        raise ValueError("taxonomy requires category and block50")
    if frame[["category", "block50"]].isna().any().any() or frame.category.duplicated().any():
        raise ValueError("taxonomy has missing values or duplicate categories")
    categories = pd.to_numeric(frame.category, errors="raise").to_numpy(float)
    if (not np.isfinite(categories).all() or np.any(categories < 0) or
            np.any(categories != np.floor(categories))):
        raise ValueError("taxonomy category IDs must be nonnegative integers")
    frame["category"] = categories.astype(int)
    if frame.category.duplicated().any():
        raise ValueError("taxonomy has duplicate numeric category IDs")
    return frame.sort_values("category").reset_index(drop=True)


def shared_categories(config):
    """The shared categories (the 100 extracted in every network), sorted."""
    path = input_path(config, "shared_categories")
    if path.suffix == ".csv":
        frame = pd.read_csv(path)
        if "category" not in frame:
            raise ValueError("shared-category CSV requires category")
        values = frame.category.tolist()
    else:
        values = path.read_text().replace(",", " ").split()
    categories = []
    for value in values:
        category = int(value)
        if float(value) != category or category < 0:
            raise ValueError("shared category IDs must be nonnegative integers")
        categories.append(category)
    if not categories or len(set(categories)) != len(categories):
        raise ValueError("shared cohort is empty or has duplicate categories")
    return sorted(categories)


def declared_cohort(config, network, arm="P", price="v66", run="primary", record=None):
    """The categories a method was run on, as the config names them (not the receipts found)."""
    if record is None:
        record = arm_record(config, network, arm, price, run)
    cohort = record.get("cohort", "all")
    if isinstance(cohort, str):
        if cohort == "all":
            values = load_categories(config).category.tolist()
        elif cohort in ("all1000", "all_1000"):
            values = list(range(1000))
        elif cohort in ("shared100", "shared_100"):
            values = shared_categories(config)
        else:
            try:
                values = config["cohorts"][cohort]
            except KeyError:
                raise InputNotAvailable("named category cohort is not configured: %s" % cohort)
    else:
        values = cohort
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("declared cohort must be a nonempty category list or configured name")
    ids = []
    for value in values:
        if isinstance(value, bool) or int(value) != float(value) or int(value) < 0:
            raise ValueError("declared cohort IDs must be nonnegative integers")
        ids.append(int(value))
    if len(set(ids)) != len(ids):
        raise ValueError("declared cohort contains duplicate categories")
    return sorted(ids)


def validate_cohort(found, expected, allow_partial=False):
    """Check that the categories found are the expected cohort; with allow_partial, missing ones are counted."""
    def category_ids(values):
        result = []
        for value in values:
            if isinstance(value, (bool, np.bool_)) or int(value) != float(value) or int(value) < 0:
                raise ValueError("category IDs must be nonnegative integers")
            result.append(int(value))
        if len(set(result)) != len(result):
            raise ValueError("duplicate categories in declared cohort validation")
        return set(result)
    found, expected = category_ids(found), category_ids(expected)
    if not found <= expected:
        raise ValueError("present categories exceed the declared cohort")
    missing = sorted(expected-found)
    if missing and not allow_partial:
        raise InputNotAvailable("declared cohort incomplete: %d categories missing (first: %s); use --allow-partial to override" %
                                (len(missing), missing[:8]))
    return dict(partial=bool(missing), n_declared_categories=len(expected),
                n_missing_categories=len(missing))


def retest_pair(config, network, price="v66"):
    """The two runs whose agreement measures extraction noise (P and its repeat extraction), as the config names
    them for this network and price."""
    try:
        pair = dict(config["networks"][network]["retest_pairs"][price])
        first, second = pair["primary"], pair["retest"]
        for entry in (first, second):
            if not isinstance(entry, dict) or not entry.get("arm") or not entry.get("run"):
                raise ValueError("retest pair entries require arm and run")
        if first == second:
            raise ValueError("retest pair names must identify distinct arm/run entries")
        pair.setdefault("cohort", "shared100")
        return pair
    except KeyError:
        raise InputNotAvailable("named retest pair is not configured: %s/%s" % (network, price))


def validate_retest_pair(config, network, price="v66", categories=None, allow_partial=False):
    """Check that both runs of the retest pair are present and that their masks differ in at least one category."""
    pair = retest_pair(config, network, price)
    first, second = pair["primary"], pair["retest"]
    expected = declared_cohort(config, network, record={"cohort": pair["cohort"]})
    if categories is not None:
        chosen = set(map(int, categories))
        if not chosen <= set(expected):
            raise ValueError("the categories given are not all in the retest pair's cohort")
        expected = sorted(chosen)
    found, identical = [], True
    for category in expected:
        paths = [resolve_arm(config, network, entry["arm"], price, entry["run"],
                             category=category, require_exists=False)/"final_mask.packbits"
                 for entry in (first, second)]
        if not all(path.is_file() for path in paths):
            continue
        found.append(category)
        if not filecmp.cmp(str(paths[0]), str(paths[1]), shallow=False):
            identical = False
    metadata = validate_cohort(found, expected, allow_partial)
    if not found:
        raise InputNotAvailable("no masks form the declared retest pair")
    if identical:
        raise ValueError("retest masks are byte-identical for every category")
    return dict(metadata, pair=pair, categories=found)


def _field(receipt, path):
    """A finite number from a receipt, at a dotted field path."""
    if not path:
        raise InputNotAvailable("receipt field is not configured")
    value = receipt
    try:
        for component in path.split("."):
            value = value[component]
    except (KeyError, TypeError):
        raise InputNotAvailable("receipt field is absent: %s" % path)
    if isinstance(value, bool):
        raise ValueError("receipt field is boolean: %s" % path)
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("receipt field is not finite: %s" % path)
    return value


def load_receipts(config, network, arm, price="v66", run="primary", metrics=None,
                  categories=None, allow_partial=False):
    """One row per category of the method's cohort, with the measures read from its circuit's receipt.

    The config's receipt_fields map each measure to a dotted receipt field, or to {difference: [a, b], scale: s} for
    (a - b) x s. An incomplete cohort raises unless allow_partial.
    """
    record = arm_record(config, network, arm, price, run)
    root = resolve_arm(config, network, arm, price, run)
    taxonomy = load_categories(config)
    fields = dict(config.get("receipt_fields", {}))
    fields.update(record.get("receipt_fields", {}))
    metrics = list(fields) if metrics is None else list(metrics)
    for metric in metrics:
        if not fields.get(metric):
            raise InputNotAvailable("%s has no documented receipt mapping" % metric)
    declared = declared_cohort(config, network, arm, price, run, record)
    taxonomy_ids = set(taxonomy.category)
    if not set(declared) <= taxonomy_ids:
        raise InputNotAvailable("declared cohort categories are absent from taxonomy")
    present = []
    for category in declared:
        folder = root / record["category_pattern"].format(category=category) / record["arm_folder"]
        if (folder/"receipt.json").is_file():
            present.append(category)
    metadata = validate_cohort(present, declared, allow_partial)
    wanted = set(declared) if categories is None else set(map(int, categories))
    if not wanted <= set(declared):
        raise ValueError("the categories given are not all in the method's cohort")
    if categories is not None:
        metadata.update(validate_cohort(set(present)&wanted, wanted, allow_partial))
        # The method stays partial if its cohort is incomplete, even when every category given is present.
        metadata["partial"] = len(present) != len(declared)
        metadata["n_declared_categories"] = len(declared)
        metadata["n_missing_categories"] = len(declared)-len(present)
    taxonomy = taxonomy[taxonomy.category.isin(wanted & set(present))]
    rows = []
    for item in taxonomy.to_dict("records"):
        folder = _inside(root / record["category_pattern"].format(category=int(item["category"])) /
                         record["arm_folder"])
        receipt_path = folder / "receipt.json"
        if not receipt_path.is_file():
            raise InputNotAvailable("declared category receipt disappeared while reading: %s" % receipt_path)
        receipt = json.loads(receipt_path.read_text())
        row = dict(item, arm=ALIASES.get(arm, arm), price=price,
                   seed=receipt.get("random_seed"), **metadata)
        for metric in metrics:
            mapping = fields[metric]
            if isinstance(mapping, str):
                value = _field(receipt, mapping)
            elif isinstance(mapping, dict) and len(mapping.get("difference", [])) == 2:
                first, second = mapping["difference"]
                value = (_field(receipt, first) - _field(receipt, second)) * float(mapping.get("scale", 1))
            else:
                raise ValueError("invalid receipt mapping for %s" % metric)
            if not np.isfinite(value):
                raise ValueError("non-finite metric %s" % metric)
            if metric in ("native_density", "kernel_unit_density", "loss_window_share") and not 0 <= value <= 1:
                raise ValueError("%s must be a fraction in [0, 1]" % metric)
            # corrected accuracy = 2 x balanced accuracy - 1 lies in [-1, 1], so its drop in points lies in [-200, 200]
            # (a circuit below chance drops more than 100); the guard still catches a fraction-vs-points mix-up.
            if metric == "corrected_accuracy_drop" and not -200 <= value <= 200:
                raise ValueError("corrected_accuracy_drop must be in accuracy points [-200, 200]")
            if metric in ("hit_loss", "rejection_loss") and not -1 <= value <= 1:
                raise ValueError("%s must be an accuracy-rate loss in [-1, 1]" % metric)
            row[metric] = value
        rows.append(row)
    if not rows:
        raise InputNotAvailable("no category receipts for %s/%s" % (network, arm))
    return pd.DataFrame(rows).sort_values("category").reset_index(drop=True)
