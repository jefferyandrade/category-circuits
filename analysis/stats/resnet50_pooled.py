#!/usr/bin/env python3
"""Fig. 3B: ResNet-50 (V2 weights), E/I-separate (P) against E/I-pooled (SA-H) pruning.

On the shared categories where both circuits are complete, the table gives each method's mean kept percentage and
mean corrected-accuracy drop, and the paired SA-H - P difference in corrected-accuracy drop (the extra accuracy lost
with pooling), each with a 95% percentile interval from 10,000 category bootstrap resamples, seed 0
(`category_resampling.block_bootstrap`, as for the other method comparisons).

Per category and method, from the circuit's receipt:
- kept % is 100 x `retained_fraction`, the circuit's kept connections over ResNet-50's 23,454,944 prunable
  connections: every convolution weight (the stem and the projection shortcuts included) and, in each of the 16
  bottleneck blocks, the two connections that carry the main branch and the shortcut into the block's sum. The
  classifier is not pruned.
- corrected accuracy is 2 x class-balanced binary accuracy - 1, in points; its drop is the dense network's minus the
  circuit's on held-out images, 200 x (dense - circuit class-balanced binary accuracy), from the receipt's
  `terminal_dense_bce` and `terminal_final_bce`.

Categories: ResNet-50's 99 shared categories (the 100 shared categories less 262, whose P extraction did not
converge). There are no SA-H circuits for 20 and 30 (their calibration fit did not converge) or for 333 (its
extraction did not finish), which leaves 96 pairs.

Two steps: `receipts` reads the circuits' receipts into the per-category table, and `table` computes Fig. 3B's rows
from it.

    python analysis/stats/resnet50_pooled.py receipts --p-results P_RESULTS --sah-results SAH_RESULTS \\
        --out data/resnet50_pooled/per_category.csv
    python analysis/stats/resnet50_pooled.py table --per-category data/resnet50_pooled/per_category.csv \\
        --out data/resnet50_pooled/values_fig3B_resnet50.csv

P_RESULTS and SAH_RESULTS are the circuit library's folders that hold
`resnet50v2_tap_query<category>/<method folder>/receipt.json` for the P and the SA-H circuits (for P, the
`results_root` that arms_config.json gives ResNet-50 V2's P circuits at price v66).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from category_resampling import block_bootstrap
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import check_outputs, write_file

NETWORK = 'resnet50v2'
TAU = 4.033e-7  # ResNet-50's loss threshold (Table 1)
# The 100 shared categories (the shared100 column of taxonomy/inputs/CATEGORY_BLOCKS_cap50.tsv).
SHARED_CATEGORIES = [round(i * 999 / 99) for i in range(100)]
CATEGORIES = [category for category in SHARED_CATEGORIES if category != 262]
# The circuit library's folder names: one folder per category, holding one folder per method at this threshold.
CATEGORY_FOLDER = 'resnet50v2_tap_query{category}'
METHOD_FOLDERS = {'P': 'v0_local_attribution_sign_resnet50v2_plain_t4p03e7_v64',
                  'SA-H': 'v0_single_population_resnet50v2_plain_sah_t4p03e7_v64'}
# The receipt's name for each method's partition of connections into pruning queues.
PARTITIONS = {'P': 'local_attribution_sign', 'SA-H': 'single_population'}
DESCRIPTIONS = {'P': 'E/I-separate', 'SA-H': 'E/I-pooled within each layer'}
PER_CATEGORY_FIELDS = ('category', 'arm', 'kept_fraction', 'dense_balanced_accuracy', 'circuit_balanced_accuracy')
TABLE_FIELDS = ('network', 'arm_or_contrast', 'description', 'metric', 'unit', 'estimate', 'ci_low', 'ci_high',
                'n_categories', 'ci_level', 'bootstrap_unit')


def read_receipt(folder, category, arm):
    """The receipt's per-category values, or None when the circuit is not complete (no receipt, or a receipt not
    written after the run's other products). A receipt of another category, threshold or method is an error."""
    path = Path(folder) / 'receipt.json'
    if not path.is_file():
        return None
    receipt = json.loads(path.read_text())
    if receipt.get('run_products', {}).get('receipt_written_last') is not True:
        return None
    if str(receipt.get('query_id')) != str(category):
        raise ValueError('%s: receipt of category %r, expected %d' % (path, receipt.get('query_id'), category))
    if receipt.get('sensitivity_tau') != TAU:
        raise ValueError('%s: receipt threshold %r, expected %r' % (path, receipt.get('sensitivity_tau'), TAU))
    if receipt.get('partition_family') != PARTITIONS[arm]:
        raise ValueError('%s: receipt of method %r, expected %s' % (path, receipt.get('partition_family'), arm))
    return dict(category=category, arm=arm, kept_fraction=float(receipt['retained_fraction']),
                dense_balanced_accuracy=float(receipt['terminal_dense_bce']['class_balanced_binary_accuracy']),
                circuit_balanced_accuracy=float(receipt['terminal_final_bce']['class_balanced_binary_accuracy']))


def per_category_rows(results):
    """The rows of the categories whose P and SA-H circuits are both complete, and the methods without a complete
    circuit in each other category; results maps each method to its library folder."""
    rows, missing = [], {}
    for category in CATEGORIES:
        pair = [read_receipt(Path(results[arm]) / CATEGORY_FOLDER.format(category=category) / METHOD_FOLDERS[arm],
                             category, arm) for arm in METHOD_FOLDERS]
        absent = [arm for arm, row in zip(METHOD_FOLDERS, pair) if row is None]
        if absent:
            missing[category] = absent
        else:
            rows.extend(pair)
    return rows, missing


def write_csv(path, fields, rows):
    with write_file(path, newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_per_category(path):
    """The per-category table as {method: {category: values}}; both methods must cover the same categories."""
    by_arm = {arm: {} for arm in METHOD_FOLDERS}
    with open(path, newline='') as stream:
        for row in csv.DictReader(stream):
            category = int(row['category'])
            if category in by_arm[row['arm']]:
                raise ValueError('category %d appears twice for %s' % (category, row['arm']))
            by_arm[row['arm']][category] = {field: float(row[field]) for field in PER_CATEGORY_FIELDS[2:]}
    if set(by_arm['P']) != set(by_arm['SA-H']):
        raise ValueError('P and SA-H rows must cover the same categories')
    return by_arm


def interval_row(arm_or_contrast, description, metric, unit, values, categories):
    interval = block_bootstrap(values, categories, n_resamples=10000, seed=0)
    return dict(network=NETWORK, arm_or_contrast=arm_or_contrast, description=description, metric=metric, unit=unit,
                estimate=interval['estimate'], ci_low=interval['ci_low'], ci_high=interval['ci_high'],
                n_categories=len(values), ci_level=0.95, bootstrap_unit='category')


def fig3b_rows(by_arm):
    categories = np.array(sorted(by_arm['P']))
    drops, rows = {}, []
    for arm, description in DESCRIPTIONS.items():
        column = {field: np.array([by_arm[arm][category][field] for category in categories])
                  for field in PER_CATEGORY_FIELDS[2:]}
        kept_pct = 100 * column['kept_fraction']
        drops[arm] = (column['dense_balanced_accuracy'] - column['circuit_balanced_accuracy']) * 200
        rows.append(interval_row(arm, description, 'kept_pct', 'percent', kept_pct, categories))
        rows.append(interval_row(arm, description, 'corrected_accuracy_drop', 'percentage_points', drops[arm],
                                 categories))
    rows.append(interval_row('SA-H - P', 'Paired extra corrected-accuracy loss with pooling', 'corrected_accuracy_drop',
                             'percentage_points', drops['SA-H'] - drops['P'], categories))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = parser.add_subparsers(dest='step', required=True)
    receipts = steps.add_parser('receipts', help='read the circuits\' receipts into the per-category table')
    receipts.add_argument('--p-results', type=Path, required=True, help='the library folder of the P circuits')
    receipts.add_argument('--sah-results', type=Path, required=True, help='the library folder of the SA-H circuits')
    receipts.add_argument('--out', type=Path, required=True)
    table = steps.add_parser('table', help='compute Fig. 3B\'s rows from the per-category table')
    table.add_argument('--per-category', type=Path, required=True)
    table.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    inputs = [args.p_results, args.sah_results] if args.step == 'receipts' else [args.per_category]
    check_outputs(files=[args.out], inputs=inputs)
    if args.step == 'receipts':
        rows, missing = per_category_rows({'P': args.p_results, 'SA-H': args.sah_results})
        write_csv(args.out, PER_CATEGORY_FIELDS, rows)
        print('%d categories with both circuits; left out (method without a complete circuit): %s' %
              (len(rows) // 2, ', '.join('%d (%s)' % (c, '/'.join(arms)) for c, arms in missing.items()) or 'none'))
    else:
        write_csv(args.out, TABLE_FIELDS, fig3b_rows(read_per_category(args.per_category)))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error))
