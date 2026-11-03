"""Trends over depth, for sign_density.py (Figs. 3C, 4A).

For each category, the Spearman correlation of a quantity with the order of the layers; its mean over categories
with a percentile category bootstrap interval and a sign-flip p value; and named paired layer contrasts (fc7 minus
fc6). A category whose values are equal at every layer has no correlation and is counted (n_constant_profiles). The
caller gives the layer order and the contrasts.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from category_resampling import block_bootstrap, block_sign_flip, result_row, adjust_families, unit_labels


def depth_rows(frame, quantity, layers, family, contrasts=(), n_resamples=10000, seed=0, quantity_unit=None,
               unit="category", network="alexnet", **labels):
    """The mean per-category Spearman rho over the layers, and the paired layer contrasts.

    Every category needs a value at every layer: varying depth ranges would change the quantity estimated. A constant
    profile counts in the contrasts but has no rho.
    """
    if quantity_unit is None:
        quantity_unit = {"dens_pos": "fraction", "dens_neg": "fraction", "sign_steps": "pruning steps"}.get(quantity)
    if not quantity_unit:
        raise ValueError("give quantity_unit for a quantity without a known unit")
    needed = {"category", "layer", quantity}
    if not needed.issubset(frame.columns):
        raise ValueError("depth input missing columns: " + ", ".join(sorted(needed - set(frame.columns))))
    if frame.duplicated(["category", "layer"]).any():
        raise ValueError("duplicate category-layer observations")
    if unit != "category":
        unit_labels(frame, unit)
        if frame.groupby("category")[unit].nunique(dropna=False).max() != 1:
            raise ValueError("each category requires one " + unit)
    table = frame.pivot(index="category", columns="layer", values=quantity).reindex(columns=layers)
    if table.empty or not np.isfinite(table.to_numpy(float)).all():
        raise ValueError("depth inference requires a finite value at every declared layer for each category")
    ordered = frame.drop_duplicates("category").set_index("category").loc[table.index].reset_index()
    blocks = unit_labels(ordered, unit)
    subgroups = ordered.block50.to_numpy() if "block50" in ordered else blocks
    family = network + ": " + family if not family.startswith(network + ":") else family
    values = table.to_numpy(float)
    nonconstant = np.ptp(values, axis=1) > 0
    rhos = np.array([spearmanr(np.arange(len(layers)), v).correlation for v in values[nonconstant]])
    rows = []
    meta = dict(quantity=quantity, layers=",".join(layers), network=network, unit=unit,
        partial=bool(frame.partial.any()) if "partial" in frame else False, **labels)
    if rhos.size:
        ci = {k: v for k, v in block_bootstrap(rhos, blocks[nonconstant], n_resamples=n_resamples, seed=seed, unit=unit).items()
              if k in ("estimate", "ci_low", "ci_high")}
        rows.append(result_row(**ci, p=block_sign_flip(rhos, blocks[nonconstant], n_resamples=n_resamples, seed=seed, unit=unit),
            family=family, effect_unit="Spearman rho", scope="G", n_categories=int(rhos.size),
            n_units=int(np.unique(blocks[nonconstant]).size), n_blocks=int(np.unique(subgroups[nonconstant]).size),
            method="mean per-category Spearman; percentile bootstrap and sign flip over " + unit + " units",
            contrast="depth", n_constant_profiles=int((~nonconstant).sum()), **meta))
    else:
        rows.append(result_row(estimate=None, family=family, effect_unit="Spearman rho", scope="D",
            n_categories=len(values), n_units=len(np.unique(blocks)), n_blocks=len(np.unique(subgroups)), method="per-category Spearman",
            contrast="depth", status="not formed: all category profiles are constant",
            n_constant_profiles=len(values), **meta))
    for upper, lower in contrasts:
        if upper not in layers or lower not in layers:
            raise ValueError("named layer contrast is outside the declared order")
        delta = table[upper].to_numpy() - table[lower].to_numpy()
        ci = {k: v for k, v in block_bootstrap(delta, blocks, n_resamples=n_resamples, seed=seed, unit=unit).items()
              if k in ("estimate", "ci_low", "ci_high")}
        rows.append(result_row(**ci, p=block_sign_flip(delta, blocks, n_resamples=n_resamples, seed=seed, unit=unit),
            family=family, effect_unit=quantity_unit, scope="G", n_categories=len(delta),
            n_units=len(np.unique(blocks)), n_blocks=len(np.unique(subgroups)),
            method="paired category contrast; percentile bootstrap and sign flip over " + unit + " units",
            contrast=upper + " minus " + lower, **meta))
    return adjust_families(rows)
