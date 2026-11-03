"""A category incidence library on disk: for each layer, which categories' circuits keep each connection (one bit per
category and connection, in the network's weight order), each connection's weight sign, and each category's kept
counts. ``build_library`` writes one from any mask reader; ``library_statistics`` reads it with ``load_library``.

Files: ``library.json`` (network, method, connection grain, layers, categories), ``category_manifest.csv``,
``layer_manifest.csv``, ``fidelity.csv``, ``category_layer_counts.parquet``, ``incidence/layer_<i>.npy`` (categories x
packed connections, big-endian bits) and ``polarity/layer_<i>.npy`` (True where the weight is positive or zero).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from atomic_io import prepare_dir, publish_file, write_file, staged_file

COUNTS_COLUMNS = ['model', 'category_id', 'layer', 'retained_count', 'retained_positive_count',
                  'retained_negative_count', 'eligible_count']


class DiskLibrary:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / 'library.json').read_text())
        self.model = self.metadata['model']
        self.arm = self.metadata['arm']
        self.axis = self.metadata['axis']
        self.category_manifest = pd.read_csv(self.directory / 'category_manifest.csv')
        self.category_ids = self.category_manifest.category_id_global.to_numpy(np.int64)
        self.animate = (self.category_manifest.animate_inanimate == 'animate').to_numpy()
        self.layer_manifest = pd.read_csv(self.directory / 'layer_manifest.csv')
        self.counts = pd.read_parquet(self.directory / 'category_layer_counts.parquet')
        self.fidelity = pd.read_csv(self.directory / 'fidelity.csv')

    def layers(self):
        return self.metadata['layers']

    def _index(self, layer):
        return self.layers().index(layer)

    def layer_packed(self, layer):
        return np.load(self.directory / 'incidence' / f'layer_{self._index(layer)}.npy', mmap_mode='r')

    def polarity(self, layer):
        return np.load(self.directory / 'polarity' / f'layer_{self._index(layer)}.npy', mmap_mode='r')


def load_library(directory):
    return DiskLibrary(directory)


def build_library(directory, *, model, arm, axis, category_manifest, layer_manifest, fidelity,
                  packed_reader, polarities, metadata=None):
    """Write the packed incidence rows, signs and exact counts of every layer from a mask reader:
    ``packed_reader(layer, lo, hi)`` returns the categories' masks of connections lo..hi of the layer, packed
    big-endian, one row per category in the order of ``category_manifest``."""
    directory = Path(directory)
    prepare_dir(directory)
    (directory / 'incidence').mkdir(exist_ok=True)
    (directory / 'polarity').mkdir(exist_ok=True)
    cm = category_manifest.copy().reset_index(drop=True)
    cats = cm.category_id_global.to_numpy(np.int64)
    with write_file(directory / 'category_manifest.csv', 'w', newline='') as stream:
        cm.to_csv(stream, index=False)
    with write_file(directory / 'layer_manifest.csv', 'w', newline='') as stream:
        layer_manifest.to_csv(stream, index=False)
    with write_file(directory / 'fidelity.csv', 'w', newline='') as stream:
        fidelity.to_csv(stream, index=False)
    layers = layer_manifest.layer.tolist()
    rows = []
    for li, r in enumerate(layer_manifest.itertuples()):
        n = int(r.eligible_route_count)
        sign = np.asarray(polarities[r.layer], bool)
        if len(sign) != n:
            raise ValueError(f'{r.layer}: polarity length differs from the layer manifest')
        with write_file(directory / 'polarity' / f'layer_{li}.npy', "wb") as stream:
            np.save(stream, sign)
        with staged_file(directory / 'incidence' / f'layer_{li}.npy') as temporary:
            data = np.lib.format.open_memmap(temporary, mode='w+',
                                            dtype=np.uint8, shape=(len(cats), (n + 7) // 8))
            ret, pos = np.zeros(len(cats), np.int64), np.zeros(len(cats), np.int64)
            for lo in range(0, n, 32768):
                hi = min(n, lo + 32768)
                p = np.asarray(packed_reader(r.layer, lo, hi), np.uint8)
                b = np.unpackbits(p, axis=1, bitorder='big')[:, :hi - lo]
                # Repacking also clears padding inherited from the following axis segment.
                data[:, lo // 8:(hi + 7) // 8] = np.packbits(b, axis=1, bitorder='big')
                ret += b.sum(axis=1, dtype=np.int64)
                pos += b[:, sign[lo:hi]].sum(axis=1, dtype=np.int64)
            data.flush()
            del data
        for i, c in enumerate(cats):
            rows.append(dict(model=model, category_id=int(c), layer=r.layer,
                             retained_count=int(ret[i]), retained_positive_count=int(pos[i]),
                             retained_negative_count=int(ret[i] - pos[i]), eligible_count=n))
    counts = pd.DataFrame(rows, columns=COUNTS_COLUMNS)
    with write_file(directory / 'category_layer_counts.parquet', 'wb') as stream:
        counts.to_parquet(stream, index=False)
    meta = dict(model=model, arm=arm, axis=axis, layers=layers,
                category_ids=cats.tolist(), animate=(cm.animate_inanimate == 'animate').tolist(),
                missing=[], **(metadata or {}))
    publish_file(json.dumps(meta, indent=2) + '\n', directory / 'library.json')
    return directory
