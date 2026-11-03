"""The layer test of Fig. 5B and the block means of Fig. 5C, from the layer statistics.

- ``holm_pvalues`` and ``layer_holm``: the Holm adjustment of the AlexNet layers' permutation p values, over the
  seven layers (the significance marks of the layer test).
- ``layer_excess``: each AlexNet layer's observed mean |s| minus its label-permutation null mean (the plotted
  layer test).
- ``block_means`` (from ``analysis/common/block_means.py``): the ResNets' block means of a per-layer value, over each
  residual block's convolutions (projections excluded; the stem convolution is a block of its own); the
  cross-network panel joins these means.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from block_means import block_means  # noqa: E402

# AlexNet's layers in the circuit library's names and in the paper's.
LAYER_NAMES = {"features.0": "conv1", "features.3": "conv2",
               "features.6": "conv3", "features.8": "conv4",
               "features.10": "conv5", "classifier.1": "fc6",
               "classifier.4": "fc7"}
NATIVE = ['features.0', 'features.3', 'features.6', 'features.8', 'features.10', 'classifier.1', 'classifier.4']
def holm_pvalues(pvalues):
    """Holm step-down adjusted p-values in the supplied layer order."""
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('invalid permutation p-values')
    order = np.argsort(p, kind='stable')
    adjusted = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    result = np.empty_like(p)
    result[order] = adjusted
    return result


def layer_holm(animacy):
    """The layer statistics table (one row per AlexNet layer, model ``alexnet1000``) in layer order, with
    ``p_holm``: the permutation p values Holm-adjusted over the seven layers."""
    animacy = animacy[animacy.model == 'alexnet1000'].set_index('layer')
    if animacy.index.duplicated().any() or set(animacy.index) != set(LAYER_NAMES):
        raise ValueError('Expected one animacy row per AlexNet layer')
    animacy = animacy.reindex(LAYER_NAMES).copy()
    animacy['p_holm'] = holm_pvalues(animacy.p_perm.to_numpy(float))
    return animacy


def layer_excess(stats):
    """Observed mean |s| minus the null mean |s| of each AlexNet layer, in layer order, from the layer
    statistics rows (mappings with ``model``, ``layer``, ``obs_mean_abs_s`` and ``null_mean_abs_s``)."""
    # Full precision: the small negative conv1 excess is printed in the paper as 0.000.
    by_layer = {r['layer']: r for r in stats if r['model'] == 'alexnet1000'}
    excess = np.array([float(by_layer[n]['obs_mean_abs_s']) -
                       float(by_layer[n]['null_mean_abs_s']) for n in NATIVE])
    return excess


