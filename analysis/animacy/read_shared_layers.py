"""Print the stored per-layer animacy statistics of the four networks on the shared categories as JSON.

``CC_SHARED_RUN`` names the folder of the stored run for AlexNet, VGG-11 and ResNet-18; ResNet-50 V2's sits in the
sibling folder with the suffix ``_V2``. In each, ``handoff/`` holds the shared-category incidence package and
``finalization_pkg/`` the output of ``run_shared_animacy.py`` (its ``--out``). For each network this prints its
layer statistics (``finalization_pkg/semantic/semantic_layer_stats_<network>.parquet``), the layer manifest and
per-category kept-connection counts of its incidence package and the statistics' metadata, with each file's sha256.
No permutation is computed here.

    CC_SHARED_RUN=RUN python analysis/animacy/read_shared_layers.py > LAYERS.json
"""
import hashlib
import json
import math
import os
from pathlib import Path

for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[variable] = '1'

import pandas as pd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base = Path(os.environ['CC_SHARED_RUN'])
    output = []
    for network in ['alexnet', 'vgg11', 'resnet18', 'resnet50']:
        suffix = '_V2' if network == 'resnet50' else ''
        root = base.with_name(base.name + suffix)
        stats_path = root / 'finalization_pkg/semantic' / ('semantic_layer_stats_' + network + '.parquet')
        manifest_path = root / 'handoff/layer_manifest.csv'
        counts_path = root / 'handoff/category_layer_counts.parquet'
        meta_path = root / 'finalization_pkg/semantic' / ('semantic_meta_' + network + '.json')
        manifest = pd.read_csv(manifest_path)
        manifest = manifest[manifest.model == network]
        stats = pd.read_parquet(stats_path)
        counts = pd.read_parquet(counts_path)
        counts = counts[counts.model == network]
        density = []
        for row in manifest.to_dict(orient='records'):
            frame = counts[counts.layer == row['layer']]
            density.append({
                'layer': row['layer'], 'stage': row['stage'],
                'n_categories': int(frame.category_id.nunique()),
                'category_ids': sorted(int(x) for x in frame.category_id.unique()),
                'eligible_count': int(row['eligible_route_count']),
                'mean_retained': float(frame.retained_count.mean()),
                'dens_pos': float(frame.retained_positive_count.mean()) / row['eligible_positive_count'],
                'dens_neg': float(frame.retained_negative_count.mean()) / row['eligible_negative_count'],
            })
        output.append({
            'network': network, 'source': str(stats_path),
            'layers': list(manifest.layer), 'stats': [{k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in row.items()} for row in stats.to_dict(orient='records')],
            'density': density, 'meta': json.loads(meta_path.read_text()),
            'hashes': {str(p): digest(p) for p in [stats_path, manifest_path, counts_path, meta_path]},
        })
    print(json.dumps(output, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
