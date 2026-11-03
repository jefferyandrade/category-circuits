"""AlexNet's input divergence on all 1,000 categories with its interval (the AlexNet points of Fig. 4C).

For conv3, conv4, conv5, fc6 and fc7, the census's integer pair sums (``pair_totals.npz`` in each layer's census
folder below CC_CENSUS) give each pair's divergence over the P circuits; the point is the mean over the defined pairs
and the interval a delete-one-category jackknife over all 1,000 categories, point +/- t(0.975, 999) x SE
(``run_four_networks.census_p_values`` and ``divergence_math.jackknife``). Every category deletion must leave a
defined mean. Writes values_fig4C_alexnet_all1000.csv in --out:

    CC_CENSUS=CENSUS python analysis/input_divergence/alexnet_intervals.py --out INTERVALS
"""
import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
from run_four_networks import ALEX, CENSUS, census_p_values
from divergence_math import jackknife
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import prepare_dir, write_file

LAYERS = ['conv3', 'conv4', 'conv5', 'fc6', 'fc7']
FIELDS = ['network', 'layer', 'value', 'ci_low', 'ci_high', 'se',
          'n_categories', 'n_delete_defined', 'source', 'selector']


def check_result(result, q):
    """All q categories are present and every category deletion is defined."""
    if result['n_categories'] != q or result['n_delete_defined'] != q:
        raise ValueError('%d categories with %d defined deletions; expected %d of each' %
                         (result['n_categories'], result['n_delete_defined'], q))


def evaluate_file(path, critical=None):
    """The jackknife summary of one layer's census pair sums over all of its P categories."""
    with np.load(path, allow_pickle=False) as raw:
        observed = census_p_values(raw)
    return jackknife(observed, critical=critical)


def main():
    """Compute the intervals of the five layers and write the table."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out', type=Path, required=True, help='output folder')
    args = parser.parse_args()
    sources = [CENSUS / (lid + '_' + grain) / 'pair_totals.npz'
               for name, lid, grain in ALEX if name in LAYERS]
    prepare_dir(args.out, sources)
    output_csv = args.out / 'values_fig4C_alexnet_all1000.csv'
    specs = {name: (lid, grain) for name, lid, grain in ALEX}
    rows = []
    for name in LAYERS:
        lid, grain = specs[name]
        source = CENSUS / (lid + '_' + grain) / 'pair_totals.npz'
        full = evaluate_file(source)
        check_result(full, 1000)
        rows.append(dict(network='alexnet', layer=name, value=full['point'],
            ci_low=full['lower'], ci_high=full['upper'], se=full['se'],
            n_categories=full['n_categories'], n_delete_defined=full['n_delete_defined'],
            source=str(source), selector='leading P block [:1000,:1000]; all categories in stored order; defined unordered pairs; delete every category'))
        print(json.dumps(dict(layer=name, value=full['point'], ci_low=full['lower'], ci_high=full['upper'])),
              flush=True)
    with write_file(output_csv, newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    try:
        main()
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error))
