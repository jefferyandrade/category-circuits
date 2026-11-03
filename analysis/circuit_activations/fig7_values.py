"""Assemble the plotted values of Fig. 7C and 7D from the analysis outputs; no statistic is computed here.

Panel D: running() reads the per-layer table written by collect_layer_table.py (site_tree_shares.tsv) and turns each
shared-population row into one plotted point: the circuits' superordinate tree share (circuit), the prototypes'
(prototypes) and their difference with its interval and permutation p (value, ci_low, ci_high, p), all at the table's
three-decimal precision, with each layer's normalized depth from sign_sharing_four_networks_holm.tsv (shipped in
data/sign_sharing/). The AlexNet fc6 and fc7 points come from fc_all_images.py on the shared 100, at three decimals.
block_means() gives the ResNet curves: the mean of the per-layer values within each residual block (projections and the
unpruned classifier excluded) at the block's mean depth. crossovers() marks, for VGG-11 and the ResNets, the shallowest
layer whose wiring-minus-prototypes interval lies above zero, read from the table written by
wiring_minus_prototypes.py (values_fig6C_difference.csv); AlexNet's mark is fc6, read from Fig. 6A (all 1,000
categories).

Panel C: figure_rows() reads the four partial shares (computation beyond image, computation beyond wiring, wiring
beyond computation, image beyond computation) at fc6 and fc7, superordinate level, from the JSON written by
partial_tree_shares.py. The paper's table (data/figures/values_fig7_running.csv) also gives each row's layer and
quantity, and repeats lo and hi as ci_low and ci_high.

The three input tables are read from the folder in CC_FIGURE_INPUTS (default: the current folder).
Usage: fig7_values.py D --out values_D.csv
       fig7_values.py C --partial-shares partial_tree_shares.json --out values_C.csv
"""
from pathlib import Path
import csv
import json
import os
import re
from decimal import Decimal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from block_means import block_means  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import check_outputs, write_file

# The folder holding the three input tables, and their file names.
R = Path(os.environ.get('CC_FIGURE_INPUTS', '.'))
DEPTH = 'sign_sharing_four_networks_holm.tsv'
STAGE = 'site_tree_shares.tsv'
DIFFERENCE = 'values_fig6C_difference.csv'
NETWORKS = ('alexnet', 'vgg11', 'resnet18', 'resnet50v2')
LAYERS = ("fc6", "fc7")
PARTIAL_SHARES = ("computation_beyond_image", "computation_beyond_wiring",
                  "wiring_beyond_computation", "image_beyond_computation")


def read(path):
    path = Path(path)
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t' if path.suffix == '.tsv' else ','))


def write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with write_file(path, newline='') as handle:
        w = csv.DictWriter(handle, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def depths():
    return {(r['network'], r['layer']): r['relative_depth'] for r in read(R / DEPTH)}


def parse_difference(text):
    match = re.fullmatch(r'([+-]?\d+\.\d+) \[([+-]?\d+\.\d+),([+-]?\d+\.\d+)\] p=([\d.]+)', text)
    if match is None:
        raise ValueError('Malformed supplied difference: ' + text)
    return dict(zip(('value', 'ci_low', 'ci_high', 'p'), match.groups()))


def running():
    depth = depths()
    output = []
    # The interval labels are those of the paper's table: the interval of the per-layer table's superordinate
    # difference column (supD), kept at its three decimals; and, for AlexNet fc6 and fc7, the interval of the
    # all-image fc6/fc7 comparison (fc_all_images.py).
    for lineno, r in enumerate(read(R / STAGE), 2):
        if r['pop'] != 'shared':
            continue
        output.append(dict(parse_difference(r['supD']), panel='D', network=r['net'], layer=r['site'],
            circuit=r['supC'], prototypes=r['supI'], n_categories=r['n'],
            relative_depth=depth[r['net'], r['site']], source=STAGE, source_row=str(lineno),
            source_column='supD', source_circuit_column='supC', source_prototypes_column='supI',
            interval='paired delete-one-category jackknife; t(n-1) 95%; three decimals of the per-layer table'))
    for layer, c, p, lo, hi, prob in [('fc6', '0.626', '0.436', '0.151', '0.229', '0.001'),
                                      ('fc7', '0.531', '0.461', '-0.006', '0.146', '0.056')]:
        output.append(dict(panel='D', network='alexnet', layer=layer, circuit=c, prototypes=p,
            value=str(Decimal(c) - Decimal(p)), ci_low=lo, ci_high=hi, p=prob, n_categories='100',
            relative_depth=depth['alexnet', layer], source='fc_all_images.py, shared 100',
            interval='paired delete-one-category jackknife; difference +/- 1.96 SE; all-image fc6 and fc7 comparison'))
    return sorted(output, key=lambda r: (NETWORKS.index(r['network']), float(r['relative_depth'])))


def crossovers():
    output = [dict(panel='D-crossover', network='alexnet', layer='fc6',
        relative_depth=depths()['alexnet', 'fc6'], source='Fig. 6A', n_categories='1000')]
    expected = {'vgg11': 'conv5_2', 'resnet18': 'stage4.block0.conv2', 'resnet50v2': 'stage4.block1.conv3'}
    source = read(R / DIFFERENCE)
    depth = depths()
    for net, layer in expected.items():
        positive = sorted([r for r in source if r['network'] == net and float(r['ci_low']) > 0],
                          key=lambda r: float(depth[net, r['layer']]))
        assert positive[0]['layer'] == layer
        r = positive[0]
        output.append(dict(r, panel='D-crossover', relative_depth=depth[net, layer],
            source=DIFFERENCE, source_row=str(source.index(r) + 2)))
    return output


class Source:
    def __init__(self, filename):
        self.filename = filename  # written into each row as given on the command line

    def document(self):
        with Path(self.filename).open() as stream:
            return json.load(stream)

    def read(self, selector):
        value = self.document()
        for key in selector.split("/") if selector else ():
            value = value[key]
        return value


def figure_rows(source):
    rows = []
    for cell in PARTIAL_SHARES:
        for layer in LAYERS:
            key = "/".join(("layers", layer, cell, "superordinate"))
            rows.append(dict(panel="C", label=layer + " " + cell, value=source.read(key + "/adjusted"),
                             lo=source.read(key + "/lo"), hi=source.read(key + "/hi"),
                             source=source.filename, key=key + "/adjusted"))
    return rows


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Write the plotted rows of Fig. 7C or 7D as CSV.')
    parser.add_argument('panel', choices=('C', 'D'))
    parser.add_argument('--partial-shares', help='the JSON written by partial_tree_shares.py (panel C)')
    parser.add_argument('--out', required=True, help='the CSV to write')
    args = parser.parse_args(argv)
    inputs = [args.partial_shares] if args.panel == 'C' else [R / DEPTH, R / STAGE, R / DIFFERENCE]
    check_outputs(files=[args.out], inputs=inputs)
    if args.panel == 'C':
        if not args.partial_shares:
            parser.error('panel C needs --partial-shares')
        rows = figure_rows(Source(args.partial_shares))
    else:
        rows = running() + crossovers()
        rows += block_means(rows, 'D')
    write(args.out, rows)


if __name__ == '__main__':
    try:
        main()
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error))
