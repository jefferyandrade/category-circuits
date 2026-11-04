"""Fig. 5, the cross-network panel: animacy statistics of each network's circuits on the shared categories.

For each network, ``network_statistics`` reads the shared-category incidence package (``incidence_package``),
computes the category-pair overlap of each layer, which also checks the package (``shared_overlap.overlap``), and the
animate-inanimate specificity of each layer's kept connections with its label-permutation null
(``shared_specificity``: 10,000 permutations at every layer, seeded by network and layer). It writes
``semantic/semantic_layer_stats_<network>.parquet`` with the curves and metadata beside it, and the
overlap statistics in ``overlap/``. A layer whose circuits keep no connection gets a row with status NOT_FORMED and no
statistics.

The paper's shared population has 100 categories (40 animate, 60 inanimate): the categories extracted in every
network.

    python analysis/animacy/run_shared_animacy.py --incidence-package PACKAGE --out OUT
        [--models alexnet vgg11 resnet18 resnet50]

``read_shared_layers.py`` and ``collect_shared_layers.py`` then turn the per-layer statistics into the panel's rows.
"""
import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import prepare_dir, publish_file, write_file


def write_json(path, value):
    publish_file(json.dumps(value, indent=2, default=lambda x: x.item()) + '\n', path)


def network_statistics(incidence_package, out, model):
    """Overlap and animacy statistics of one network on the shared categories, from the incidence package in the
    folder ``incidence_package``, written to the folder ``out`` (``overlap/`` and ``semantic/``)."""
    from incidence_package import load_incidence_package
    from shared_overlap import overlap
    from shared_specificity import animacy_specificity
    out = Path(out)
    library = load_incidence_package(incidence_package, model)
    overlap_result = overlap(library)
    with write_file(out / f'overlap/overlap_layer_stats_{model}.parquet', 'wb') as stream:
        overlap_result['stats'].to_parquet(stream)
    with write_file(out / f'overlap/pairwise_JJ0_{model}.npz', "wb") as stream:
        np.savez_compressed(stream, **overlap_result['pairwise'])
    with write_file(out / f'overlap/sign_split_intersections_{model}.npz', "wb") as stream:
        np.savez_compressed(stream, **overlap_result['sign_split'])
    layers = library.layers()
    missing = [L for L in layers
               if int(library.counts.loc[library.counts.layer == L, 'retained_count'].sum()) == 0]
    semantic_result = animacy_specificity(library, layers=[L for L in layers if L not in missing])
    if missing:
        rows, unavailable = [], []
        numeric = ('obs_mean_abs_s', 'obs_mean_signed_s', 'struct', 'null_mean_abs_s',
                   'null_sd_abs_s', 'delta_pi_abs_s', 'p_perm', 'mc_se', 'jk_se_obs',
                   'jk_se_delta', 'maxT_crit')
        for L in missing:
            stage = library.layer_manifest.set_index('layer').loc[L, 'stage']
            reason = 'No retained routes in this layer; animate–inanimate specificity is undefined'
            row = dict(model=model, layer=L, stage=stage, n_rows=0, total_mult=0.,
                       B_perms=0, B_planned=10000,
                       seed=zlib.crc32(f'{model}|{L}'.encode()) & 0xFFFFFFFF,
                       n_exceed_bins=0, semantic_status='NOT_FORMED', semantic_reason=reason)
            row.update({name: np.nan for name in numeric})
            rows.append(row)
            unavailable.append(dict(item=f'semantic:{model}:{L}', scope='shared100', reason=reason))
            for suffix in ('obs_curve', 'null_curve_mean', 'null_curve_sd', 'envelope_bound'):
                semantic_result['curves'][f'{L}|{suffix}'] = np.full(10, np.nan)
            semantic_result['curves'][f'{L}|obs_wsum'] = np.zeros(10)
            semantic_result['curves'][f'{L}|exceed'] = np.zeros(10, np.int8)
            semantic_result['curves'][f'{L}|null_mabs'] = np.empty(0, np.float32)
        stats = pd.concat([semantic_result['stats'], pd.DataFrame(rows)], ignore_index=True)
        stats['semantic_status'] = stats.semantic_status.fillna('FORMED')
        stats['semantic_reason'] = stats.semantic_reason.fillna('')
        semantic_result['stats'] = stats.set_index('layer').loc[layers].reset_index()
        semantic_result['meta'].update(n_layers=len(layers), unavailable_layers=unavailable)
    with write_file(out / f'semantic/semantic_layer_stats_{model}.parquet', 'wb') as stream:
        semantic_result['stats'].to_parquet(stream)
    with write_file(out / f'semantic/semantic_curves_{model}.npz', "wb") as stream:
        np.savez_compressed(stream, **semantic_result['curves'])
    write_json(out / f'semantic/semantic_meta_{model}.json', semantic_result['meta'])


def main(argv=None):
    parser = argparse.ArgumentParser(description='Animacy statistics of each network on the shared categories.')
    parser.add_argument('--incidence-package', type=Path, required=True,
                        help='the shared-category incidence package (category_manifest.csv, layer_manifest.csv, '
                             'category_layer_counts.parquet, incidence_signatures/, pairwise_intersections.npz)')
    parser.add_argument('--out', type=Path, required=True, help='output folder')
    parser.add_argument('--models', nargs='+', default=['alexnet', 'vgg11', 'resnet18', 'resnet50'],
                        help='networks, as named in the package (default: all four)')
    args = parser.parse_args(argv)
    prepare_dir(args.out, [args.incidence_package])
    for name in ('overlap', 'semantic'):
        (args.out / name).mkdir(parents=True, exist_ok=True)
    for model in args.models:
        network_statistics(args.incidence_package.resolve(), args.out.resolve(), model)


if __name__ == '__main__':
    try:
        main()
    except (FileExistsError, ValueError) as error:
        raise SystemExit(str(error))
