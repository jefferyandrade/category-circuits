#!/usr/bin/env python3
"""The ResNets' block means in Figs. 4C, 5 and 7: the mean of a per-layer value over each residual block's
main-path convolutions (projection shortcuts excluded; the stem convolution is a block of its own), at the mean of
their relative depths.

``block_means`` computes them from a figure table's per-layer rows; ``main`` writes them for one panel of a figure
table (default: the Fig. 4 table, panel C). The sums are exactly rounded (``math.fsum``), so the means do not depend
on the Python version.
"""
import argparse
import csv
import math
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from atomic_io import check_outputs, write_file

RESNETS=('resnet18','resnet50v2')


def block_means(rows, panel):
    result=[]
    for net in RESNETS:
        groups={}
        for row in rows:
            if row['panel']!=panel or row.get('network')!=net: continue
            layer=row['layer']
            if layer=='stem.conv': key=layer
            elif re.fullmatch(r'stage\d+\.block\d+\.conv\d+',layer): key=layer.rsplit('.',1)[0]
            else: continue  # Projections and the unpruned classifier do not enter a block.
            groups.setdefault(key,[]).append(row)
        for key,parts in groups.items():
            result.append(dict(panel=panel,network=net,layer='block_mean:'+key,quantity='block_mean',
                value=repr(math.fsum(float(q['value']) for q in parts)/len(parts)),
                relative_depth=repr(math.fsum(float(q['relative_depth']) for q in parts)/len(parts))))
    return result


def main(argv=None):
    """Write the block-mean rows of one panel of a figure table as CSV."""
    parser = argparse.ArgumentParser(description='Block means of the ResNets in a figure table.')
    parser.add_argument('--values', default=str(Path(__file__).resolve().parents[2] / 'data/figures/values_fig4_sharing.csv'),
                        help='figure table with panel, network, layer, value and relative_depth columns')
    parser.add_argument('--panel', default='C', help='panel whose per-layer rows are averaged')
    parser.add_argument('--out', help='output CSV (default: standard output)')
    args = parser.parse_args(argv)
    check_outputs(files=[args.out], inputs=[args.values])
    with open(args.values, newline='') as handle:
        rows = list(csv.DictReader(handle))
    means = block_means(rows, args.panel)
    fields = ['panel', 'network', 'layer', 'quantity', 'value', 'relative_depth']
    def write(handle):
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(means)
    if args.out:
        with write_file(args.out, newline='') as handle:
            write(handle)
    else:
        write(sys.stdout)


if __name__ == '__main__':
    try:
        main()
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error))
