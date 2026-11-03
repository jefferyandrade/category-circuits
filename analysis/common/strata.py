"""Weight strata of the sign- and weight-size-matched overlap reference (Fig. 4B2, the within-sign comparison of
Fig. 4D and the wiring similarity of Fig. 6).

A layer's connections are divided by the sign of their weight and, within each sign, by decile of |w|. The weights
are the checkpoint's, so the strata depend on the network alone and not on any circuit.

* **Sign.** ``w >= 0`` is class 0 ("positive"), ``w < 0`` class 1. Exact zeros go with the positive class
  (``zero_policy="positive"``); AlexNet's checkpoint has no exact zeros in a pruned layer, so there the rule never
  applies.
* **|w| decile** (``scope="within_polarity"``, the default): within one layer and one sign, a connection's decile is
  ``floor(10 * first_rank / n)``, where ``first_rank`` is the 0-based rank of the first connection with the same |w|
  in ascending order. Exact ties therefore always share a decile, and the decile depends on the weight values only,
  not on the order of the connections. ``scope="layer"`` ranks |w| over the whole layer.
* **Stratum label** ``s = sign * n_deciles + decile``, so a layer has 20 strata by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

N_DECILES = 10
POLARITY_POSITIVE = 0
POLARITY_NEGATIVE = 1
STRATA_RULE = "sign by |w| decile within sign"


@dataclass(frozen=True)
class RouteStrata:
    """The stratum label of every connection of one layer, in the layer's weight order."""

    layer_id: str
    labels: np.ndarray  # (n_routes,) smallest unsigned dtype in [0, n_strata)
    n_strata: int
    definition: dict = field(default_factory=dict)

    def __post_init__(self):
        labels = np.asarray(self.labels)
        if (not isinstance(self.n_strata, (int, np.integer)) or self.n_strata < 1
                or labels.ndim != 1 or labels.dtype.kind not in "iu"
                or np.any(labels < 0) or np.any(labels >= self.n_strata)):
            raise ValueError("strata need integer route labels in [0, n_strata)")
        object.__setattr__(self, "labels", np.asarray(labels, dtype=smallest_unsigned_dtype(self.n_strata - 1)))


def smallest_unsigned_dtype(maximum: int) -> np.dtype:
    """Smallest unsigned storage type covering the inclusive integer maximum."""
    if not isinstance(maximum, (int, np.integer)) or maximum < 0:
        raise ValueError("an unsigned storage maximum must be a nonnegative integer")
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if maximum <= np.iinfo(dtype).max:
            return np.dtype(dtype)
    raise ValueError("integer maximum exceeds uint64")


def polarity_class(weight: np.ndarray, zero_policy: str = "positive") -> np.ndarray:
    """0 for w >= 0 (zero_policy 'positive'), 1 for w < 0; flat, in weight order."""

    w = np.asarray(weight).reshape(-1)
    if zero_policy != "positive":
        raise ValueError("only zero_policy='positive' is implemented")
    return (w < 0).astype(np.uint8)


def tie_preserving_deciles(values: np.ndarray, n_bins: int = N_DECILES) -> np.ndarray:
    """``floor(n_bins * first_rank / n)``, ties sharing their first rank; unsigned labels."""

    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    dtype = smallest_unsigned_dtype(n_bins - 1)
    n = v.size
    if n == 0:
        return np.zeros(0, dtype=dtype)
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite weight magnitude")
    sv = np.sort(v, kind="stable")
    first = np.searchsorted(sv, v, side="left").astype(np.int64)
    out = (first * int(n_bins)) // n
    return np.minimum(out, n_bins - 1).astype(dtype)


def route_strata(
    weight: np.ndarray,
    *,
    layer_id: str = "",
    n_deciles: int = N_DECILES,
    scope: str = "within_polarity",
    zero_policy: str = "positive",
) -> RouteStrata:
    """Stratum label of every connection: sign by |w| decile (see the module docstring)."""

    w = np.asarray(weight, dtype=np.float64).reshape(-1)
    pol = polarity_class(w, zero_policy)
    absw = np.abs(w)
    dec = np.zeros(w.size, dtype=smallest_unsigned_dtype(n_deciles - 1))
    if scope == "within_polarity":
        for cls in (POLARITY_POSITIVE, POLARITY_NEGATIVE):
            idx = np.flatnonzero(pol == cls)
            if idx.size:
                dec[idx] = tie_preserving_deciles(absw[idx], n_deciles)
    elif scope == "layer":
        dec = tie_preserving_deciles(absw, n_deciles)
    else:
        raise ValueError(f"unknown decile scope {scope!r}")
    labels = (pol.astype(np.int64) * n_deciles + dec).astype(smallest_unsigned_dtype(2 * n_deciles - 1))
    definition = {
        "rule": STRATA_RULE,
        "polarity": "sign>=0 -> 0, sign<0 -> 1",
        "zero_policy": zero_policy,
        "n_deciles": int(n_deciles),
        "decile_scope": scope,
        "decile_rule": "floor(n_deciles * first_rank / n), ties share first rank",
        "label": "polarity * n_deciles + decile",
    }
    return RouteStrata(layer_id, labels, 2 * int(n_deciles), definition)


__all__ = [
    "N_DECILES",
    "RouteStrata",
    "polarity_class",
    "route_strata",
    "smallest_unsigned_dtype",
    "tie_preserving_deciles",
]
