"""Wiring minus prototypes: the paired difference of two category-tree shares at one layer (Fig. 7D crossovers;
Fig. 6, C-test rows).

For each network layer of Fig. 6C (VGG-11 and the ResNets on the shared 100, 99 in ResNet-50 V2), the superordinate
tree share of the wiring (W) minus that of the category prototypes (A), with a joint delete-one-category jackknife
interval: each deletion removes the same category from both matrices and refits the node effects, while the
full-data permutation means that tree_contrast.py recorded stay fixed, as in tree_contrast. The adjusted share is
(observed - permutation mean) / (1 - permutation mean); the jackknife variance is (n - 1)/n times the sum of squared
deviations of the deleted differences; the interval is the difference plus or minus t(.975, n - 1) times its square
root. An asterisk marks an interval that excludes zero.

Each layer's matrices are read with tree_contrast.load, using the arguments of that layer's tree_contrast.py run as
its run plan records them; the permutation means come from that run's output table.

Usage: wiring_minus_prototypes.py --plan PLAN.json --resnet50-plan PLAN.json --panel FIG6_VALUES.csv --out DIR
writes DIR/values_fig6C_difference.csv, which fig7_values.py reads for the crossovers.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stats"))
import tree_contrast as TC  # noqa: E402
import tree_share as TS  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import prepare_dir, write_file


def read_rows(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def paired_interval(estimate, left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.ndim != 1 or left.shape != right.shape or len(left) < 2 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('paired category deletion replicates must be aligned and finite')
    deleted = left - right
    count = len(deleted)
    variance = (count - 1) / count * np.square(deleted - deleted.mean()).sum()
    se = np.sqrt(variance)
    width = t.ppf(.975, count - 1) * se
    low, high = float(estimate - width), float(estimate + width)
    return dict(difference=float(estimate), ci_low=low, ci_high=high,
                se=float(se), jackknife_variance=float(variance),
                n_categories=count, asterisk=bool(low > 0 or high < 0))


def jackknife_difference(statistic, labels, means, point=None):
    """W minus A with its interval, the permutation means held fixed in every deletion."""
    labels, means = np.asarray(labels), np.asarray(means, dtype=float)
    indices = np.arange(len(labels))
    full = (statistic(indices, labels) - means) / (1 - means)
    deleted = np.asarray([(statistic(indices[indices != omitted], labels[indices != omitted]) - means) / (1 - means)
                          for omitted in indices])
    result = paired_interval(full[0] - full[1] if point is None else point, deleted[:, 0], deleted[:, 1])
    result.update(recomputed_wiring=float(full[0]), recomputed_prototypes=float(full[1]))
    return result


def arg_values(argv):
    names = ('network', 'layer', 'gram-dir', 'pairs-dir', 'trained-dir', 'site-map', 'taxonomy', 'untrained-dirs')
    return {name.replace('-', '_'): argv[argv.index('--' + name) + 1]
            for name in names if '--' + name in argv}


def task_plan(plan, resnet50_plan):
    """Each (network, layer)'s tree_contrast.py arguments and output table, from the run plans.

    A plan is a JSON object whose tasks each record an id, the argv of one run and its output. The tree_contrast runs
    on the training images are the tasks with ids tree_<network>_<layer>_train; ResNet-50 V2's are in its own plan.
    """
    tasks = {}
    for row in json.loads(Path(plan).read_text())['tasks']:
        if row['id'].startswith('tree_') and row['id'].endswith('_train'):
            args = arg_values(row['argv'])
            tasks[(args['network'], args['layer'])] = dict(args=args, source_csv=row['output'])
    for row in json.loads(Path(resnet50_plan).read_text())['tasks']:
        args = arg_values(row['argv'])
        if args.get('network') == 'resnet50v2' and '--trained-dir' in row['argv']:
            key = ('resnet50v2', args['layer'])
            if key not in tasks:
                tasks[key] = dict(args=args, source_csv=row['output'])
    return tasks


def select_plain(path):
    rows = read_rows(path)
    out = {row['lens']: row for row in rows if row['quantity'] == 'plain' and row['level'] == 'superordinate' and row['lens'] in ('W', 'A')}
    if set(out) != {'W', 'A'}:
        raise ValueError('expected W and A superordinate rows in %s' % path)
    return out


def compute_task(task, point=None):
    args = SimpleNamespace(**task['args'])
    cats, matrices, seeds, site = TC.load(args, args.layer)
    # tree_contrast.load keeps the categories that every matrix it reads (untrained networks included) covers. Only W
    # and A enter this difference; their node fits do not depend on the other matrices.
    specs = [('plain', 'W', None), ('plain', 'A', None)]
    statistic = TC.joint_statistics({name: matrices[name] for name in ('W', 'A')}, specs)
    tax = TS.load_tax(args.taxonomy)
    labels = pd.factorize(tax.loc[cats, 'superordinate'].astype(str).values)[0]
    recorded = select_plain(task['source_csv'])
    means = [float(recorded[name]['perm_mean']) for name in ('W', 'A')]
    result = jackknife_difference(statistic, labels, means, point=point)
    result.update(network=args.network, layer=args.layer, activation_site=site,
                  wiring_perm_mean=means[0], prototypes_perm_mean=means[1],
                  source_csv=task['source_csv'], input_args=json.dumps(task['args'], sort_keys=True))
    return result


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with write_file(path, newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def differences(tasks, panel_path, out):
    """The W - A difference of every network layer that the panel table's C rows plot."""
    panel = {}
    for row in read_rows(panel_path):
        if row['panel'] == 'C':
            lens = json.loads(row['selector'])['lens']
            panel.setdefault((row['network'], row['layer']), {})[lens] = row
    results = []
    for key, rows in panel.items():
        point = float(rows['W']['value']) - float(rows['A']['value'])
        result = compute_task(tasks[key], point=point)
        for lens, name in [('W', 'wiring'), ('A', 'prototypes')]:
            result[name] = float(rows[lens]['value'])
            result[name + '_matrix_minus_panel'] = result['recomputed_' + name] - result[name]
            # The plotted value is a CSV round trip of the share; the share recomputed from the matrices must agree
            # with it to 14 decimal places.
            if round(result['recomputed_' + name], 14) != round(result[name], 14):
                raise AssertionError('share recomputed from the matrices differs from the plotted value: %s %s'
                                     % (key, lens))
        if result['difference'] != result['wiring'] - result['prototypes']:
            raise AssertionError('the difference is not the plotted W share minus the plotted A share')
        expected_categories = 99 if key[0] == 'resnet50v2' else 100
        if result['n_categories'] != expected_categories:
            raise AssertionError('category count differs from the shared categories: %s' % (key,))
        result.update(panel='C', quantity='difference', lens='W-A', level='superordinate',
                      interval='joint delete-one-category jackknife; t(n-1), unadjusted 95%',
                      panel_source=str(panel_path), source_wiring=rows['W']['source'], source_prototypes=rows['A']['source'])
        results.append(result)
        print(json.dumps({k: result[k] for k in ('network', 'layer', 'difference', 'ci_low', 'ci_high', 'asterisk')}), flush=True)
    write_csv(out / 'values_fig6C_difference.csv', results)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--plan', type=Path, required=True,
                        help='the run plan of the tree_contrast.py runs (tasks tree_<network>_<layer>_train)')
    parser.add_argument('--resnet50-plan', type=Path, required=True,
                        help="the run plan of ResNet-50 V2's tree_contrast.py runs")
    parser.add_argument('--panel', type=Path, required=True,
                        help='the plotted Fig. 6C values: network, layer, selector (JSON with the lens), value, source')
    parser.add_argument('--out', type=Path, required=True, help='the folder to write values_fig6C_difference.csv in')
    args = parser.parse_args(argv)
    prepare_dir(args.out, [args.plan, args.resnet50_plan, args.panel])
    differences(task_plan(args.plan, args.resnet50_plan), args.panel, args.out)


if __name__ == '__main__':
    try:
        main()
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error))
