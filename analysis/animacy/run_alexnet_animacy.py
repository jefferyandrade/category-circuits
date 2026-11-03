"""Fig. 5, AlexNet on all 1,000 categories: the animacy statistics, from the circuits' category incidence library.

statistics  For each layer: the animate-inanimate specificity s of every kept connection and the layer's mean |s|
            against its label-permutation null (5,000 permutations; 10,000 at fc6 and fc7), the participation curves
            with their max-T envelope, the joint count grid H[k_A, k_I] and the category-pair overlap, with the
            paper's settings (``library_statistics.statistics_from_library``). The incidence library is the one
            ``incidence_library.build_library`` writes from the P circuits' masks.
tables      From those outputs, the plotted tables: the participation-specificity plane with its hypergeometric null,
            the participation curves, the joint counts and the layer statistics (``panel_tables``, evaluating the
            plane helper ``specificity_plane.py``), and the layer test: the observed-minus-null mean |s| and the
            permutation p values Holm-adjusted over the seven layers (``animacy_statistics``).

    python analysis/animacy/run_alexnet_animacy.py statistics --library LIBRARY --out STATISTICS
    python analysis/animacy/run_alexnet_animacy.py tables --statistics STATISTICS --out TABLES
"""
import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import prepare_dir, write_file

HERE = Path(__file__).resolve().parent


def statistics(args):
    from library_statistics import statistics_from_library
    statistics_from_library(args.library, args.out, model_tag='alexnet1000', generator='blocks',
                            nproc=args.nproc, extras=True)


def tables(args):
    import pandas as pd
    import animacy_statistics
    import panel_tables
    out = args.out
    prepare_dir(out, [args.statistics])
    paths = panel_tables.write_animacy_tables(out / 'animacy', args.statistics, HERE / 'specificity_plane.py')
    stats = pd.read_csv(out / 'animacy_layer_statistics.csv', float_precision='round_trip')
    tests = animacy_statistics.layer_holm(stats)
    tests['obs_minus_null_mean_abs_s'] = animacy_statistics.layer_excess(stats.to_dict('records'))
    tests['layer_label'] = list(animacy_statistics.LAYER_NAMES.values())
    with write_file(out / 'animacy_layer_tests.csv', 'w', newline='') as stream:
        tests.to_csv(stream)
    for path in paths + [out / 'animacy_layer_tests.csv']:
        print(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description='AlexNet animacy statistics on all 1,000 categories (Fig. 5).')
    steps = parser.add_subparsers(dest='step', required=True)
    first = steps.add_parser('statistics', help='the per-layer statistics, null and grids from an incidence library')
    first.add_argument('--library', type=Path, required=True, help='the category incidence library of the P circuits')
    first.add_argument('--out', type=Path, required=True, help='output folder')
    first.add_argument('--nproc', type=int, default=40,
                       help='permutation streams, one worker process each; the paper used 40, and another value '
                            'draws different permutations (default: 40)')
    second = steps.add_parser('tables', help='the plotted tables and the layer test from the statistics')
    second.add_argument('--statistics', type=Path, required=True, help='the statistics step\'s output folder')
    second.add_argument('--out', type=Path, required=True, help='output folder')
    args = parser.parse_args(argv)
    (statistics if args.step == 'statistics' else tables)(args)


if __name__ == '__main__':
    try:
        main()
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error))
