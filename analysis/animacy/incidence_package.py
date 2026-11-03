"""Read one network from the shared-category incidence package: the category and layer manifests, the kept-connection
counts, the packed incidence signatures (one row per distinct set of categories that keep a connection, and sign) and
the category-pair intersections of kept connections."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

MODELS = ('alexnet', 'vgg11', 'resnet18', 'resnet50')


def words_from_flags(flags):
    flags = np.asarray(flags, bool)
    out = np.zeros((flags.size + 63) // 64, np.uint64)
    for i in np.flatnonzero(flags):
        out[i // 64] |= np.uint64(1) << np.uint64(i % 64)
    return out


def expand_words(words, K):
    return np.unpackbits(np.ascontiguousarray(words, dtype='<u8').view(np.uint8),
                         axis=1, bitorder='little')[:, :K]


@dataclass
class IncidencePackage:
    """One network's view of the package (``axis`` is the package's connection grain, ``kernel_level``)."""
    model: str
    axis: str
    category_ids: np.ndarray
    animate: np.ndarray
    layer_manifest: pd.DataFrame
    counts: pd.DataFrame
    directory: Path

    def layers(self):
        return self.layer_manifest.layer.tolist()

    def signatures(self, layer):
        df = pd.read_parquet(self.directory / f'incidence_signatures/{self.model}.parquet',
                             filters=[('layer', '=', layer)])
        cols = ['bits_0_63'] + (['bits_64_99'] if len(self.category_ids) > 64 else [])
        return (np.ascontiguousarray(df[cols].to_numpy(np.uint64)),
                df.multiplicity.to_numpy(np.int64), (df.structural_sign == '+').to_numpy())

    def intersections(self, layer, polarity=None):
        if polarity is None and self.axis == 'kernel_level':
            with np.load(self.directory / 'pairwise_intersections.npz') as z:
                return z[f'{self.model}|{layer}']
        if polarity not in (None, 'pos', 'neg'):
            raise ValueError('polarity must be pos, neg, or None')
        words, mult, positive = self.signatures(layer)
        if polarity is not None:
            use = positive if polarity == 'pos' else ~positive
            words, mult = words[use], mult[use]
        K = len(self.category_ids)
        M = np.zeros((K, K), np.float64)
        for lo in range(0, len(words), 32768):
            B = expand_words(words[lo:lo + 32768], K).astype(np.float64)
            M += (B * mult[lo:lo + 32768, None]).T @ B
        return np.rint(M).astype(np.int64)


def load_incidence_package(directory, model):
    """The package in ``directory`` for the network ``model`` (alexnet, vgg11, resnet18 or resnet50)."""
    directory = Path(directory)
    if model not in MODELS:
        raise ValueError(f'unknown architecture {model}')
    cm = pd.read_csv(directory / 'category_manifest.csv').sort_values('category_index')
    lm = pd.read_csv(directory / 'layer_manifest.csv')
    cnt = pd.read_parquet(directory / 'category_layer_counts.parquet')
    return IncidencePackage(model, 'kernel_level', cm.category_id_global.to_numpy(np.int64),
                            (cm.animate_inanimate == 'animate').to_numpy(), lm[lm.model == model].reset_index(drop=True),
                            cnt[cnt.model == model].reset_index(drop=True), directory)
