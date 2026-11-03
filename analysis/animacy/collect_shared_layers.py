"""Fig. 5, the cross-network panel: one row per network and layer from the stored per-layer animacy statistics.

Each row has the observed and null mean |s| of the layer's kept connections on the shared categories, their
difference (the plotted excess), the permutation p value, its Holm adjustment over the network's layers and the
number of permutations. The checks file compares the incidence package's kept connections with the paper's circuits:
each network's kept percentage with Table 1's, and each layer's kept fractions by sign with DENSITY.tsv.

    python analysis/animacy/collect_shared_layers.py --layers LAYERS.json --density DENSITY.tsv
        --out ROWS.csv --checks CHECKS.json

LAYERS.json is the output of ``read_shared_layers.py``.
"""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import check_outputs, publish_file, write_file

FIELDS = ['network', 'layer', 'layer_index', 'n_layers', 'n_categories',
          'n_animate', 'n_inanimate', 'observed', 'null', 'excess', 'p',
          'p_holm', 'n_permutations', 'source']
# Each network's kept percentage as Table 1 prints it.
TABLE_DENSITY = {'alexnet': 8.9, 'vgg11': 5.0, 'resnet18': 27.2, 'resnet50': 31.4}


def rounded_equal(value, expected, places):
    return format(value, '.' + str(places) + 'f') == format(expected, '.' + str(places) + 'f')


def holm(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    adjusted = [None] * len(values)
    previous = 0.0
    for rank, i in enumerate(order):
        previous = max(previous, min(1.0, values[i] * (len(values) - rank)))
        adjusted[i] = previous
    return adjusted


def label(network, native):
    maps = {
        'alexnet': dict(zip(['features.0', 'features.3', 'features.6', 'features.8', 'features.10', 'classifier.1', 'classifier.4'],
                           ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7'])),
        'vgg11': dict(zip(['features.6', 'features.8', 'features.11', 'features.13', 'features.16', 'features.18', 'classifier.0', 'classifier.3'],
                         ['conv3_1', 'conv3_2', 'conv4_1', 'conv4_2', 'conv5_1', 'conv5_2', 'fc6', 'fc7'])),
    }
    if network in maps:
        return maps[network][native]
    if native == 'conv1':
        return 'stem.conv'
    return re.sub(r'^layer([1-4])\.([0-9]+)\.', r'stage\1.block\2.', native).replace('downsample.0', 'projection')


def build_rows(item):
    stats = {r['layer']: r for r in item['stats']}
    missing = [x for x in item['layers'] if x not in stats]
    if missing:
        raise ValueError(item['network'] + ' missing layer statistics: ' + ', '.join(missing))
    fields = ['obs_mean_abs_s', 'null_mean_abs_s', 'delta_pi_abs_s', 'p_perm', 'B_perms']
    for layer in item['layers']:
        absent = [f for f in fields if stats[layer].get(f) is None]
        if absent:
            raise ValueError(item['network'] + ' ' + layer + ' missing: ' + ', '.join(absent))
    adjusted = holm([stats[x]['p_perm'] for x in item['layers']])
    meta = item['meta']
    densities = {r['layer']: r for r in item['density']}
    rows = []
    for i, layer in enumerate(item['layers']):
        row = stats[layer]
        rows.append(dict(zip(FIELDS, [
            'resnet50v2' if item['network'] == 'resnet50' else item['network'], label(item['network'], layer), i + 1, len(item['layers']),
            densities[layer]['n_categories'], meta['n_animate'], meta['n_inanimate'],
            row['obs_mean_abs_s'], row['null_mean_abs_s'],
            row['obs_mean_abs_s'] - row['null_mean_abs_s'], row['p_perm'], adjusted[i],
            row['B_perms'], item['source'] + '#layer=' + layer,
        ])))
    return rows


def arguments():
    parser = argparse.ArgumentParser(description='Rows of the cross-network animacy panel from the stored '
                                                 'per-layer statistics, with checks of their inputs.')
    parser.add_argument('--layers', type=Path, required=True,
                        help='the stored per-layer statistics, as printed by read_shared_layers.py (JSON)')
    parser.add_argument('--density', type=Path, required=True,
                        help='kept fractions by sign per network and layer on the shared categories '
                             '(tab-separated: network, layer, dens_pos, dens_neg), for the checks')
    parser.add_argument('--out', type=Path, required=True, help='the panel rows (CSV)')
    parser.add_argument('--checks', type=Path, required=True, help='the checks (JSON)')
    return parser.parse_args()


def main():
    args = arguments()
    check_outputs(files=[args.out, args.checks], inputs=[args.layers, args.density])
    evidence = json.loads(args.layers.read_text())
    density_ref = {(r['network'], r['layer']): r for r in csv.DictReader(
        args.density.open(), delimiter='\t')}
    summary = {'networks': {}}
    output = []
    for item in evidence:
        network = item['network']
        rows = build_rows(item)
        output.extend(rows)
        density = sum(r['mean_retained'] for r in item['density']) / sum(r['eligible_count'] for r in item['density'])
        per_layer = []
        for r in item['density']:
            key = ('resnet50v2' if network == 'resnet50' else network, label(network, r['layer']))
            if key not in density_ref:
                per_layer.append({'layer': label(network, r['layer']), 'reference': 'absent'})
            else:
                ref = density_ref[key]
                per_layer.append({'layer': key[1], 'dens_pos': r['dens_pos'], 'dens_neg': r['dens_neg'],
                                  'reference_pos': float(ref['dens_pos']), 'reference_neg': float(ref['dens_neg']),
                                  'matches_reference': all(rounded_equal(r[k], float(ref[k]), 4) for k in ['dens_pos', 'dens_neg'])})
        summary['networks'][network] = {
            'n_layers': len(rows), 'n_categories': sorted(set(r['n_categories'] for r in rows)),
            'domain_counts': [item['meta']['n_animate'], item['meta']['n_inanimate']],
            'permutations': dict(Counter(r['n_permutations'] for r in rows)),
            'kept_percent': 100 * density,
            'table1_kept_percent': TABLE_DENSITY[network],
            'kept_percent_matches_table1': rounded_equal(100 * density, TABLE_DENSITY[network], 1),
            'per_layer_density': per_layer,
            'source_hashes': item['hashes'],
        }
    with write_file(args.out, newline='') as stream:
        writer = csv.DictWriter(stream, FIELDS)
        writer.writeheader()
        writer.writerows(output)
    publish_file(json.dumps(summary, indent=2, sort_keys=True) + '\n', args.checks)
    print('Collected %d stored layer rows; %d of %d kept percentages match Table 1.' % (
        len(output), sum(v['kept_percent_matches_table1'] for v in summary['networks'].values()),
        len(summary['networks'])))


if __name__ == '__main__':
    try:
        main()
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error))
