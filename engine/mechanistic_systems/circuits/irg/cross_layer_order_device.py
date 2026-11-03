"""Tensor kernels of the global walk's cross-layer order diagnostic.

``prune_scope_walk.cross_layer_order_diagnostic`` reports three quantities after
a global walk; no decision reads any of them.  As host numpy over the pooled
route field (57M routes on AlexNet) they cost 105-148 s per SG or SA-G run on
the main thread with the GPU idle.  This module holds the same calculations
as torch kernels on one device:

* :func:`certify_cross_layer_order_stability`, a port of
  ``score_order.certify_prune_order_stability`` for the walk's one call (scope
  ``cross_layer``, one look, every pricing input supplied, no unresolved-sign
  routes), with every audited fraction of a scope priced against one shared
  set of sorts;
* :func:`queue_boundary_uncertainty`, a linear algorithm for the queue's
  boundary uncertainty (prefix and suffix interval extremes, and one
  run of cuts per route and side);
* :func:`pointwise_band`, the descriptive band at every cut in one pass over the
  queue.  A route is counted on one contiguous run of cuts per side, so the
  per-cut counts come from difference arrays instead of one window per cut.

Every comparison is the host's (the same float expressions, the same strict and
non-strict sides), so routes, counts and bands are the host's; sums of
activity, damage and variance differ only by summation order.  The kernels use
no float cumsum and no weighted bincount on the device: neither has a
deterministic CUDA kernel, and the pilot may run with deterministic algorithms
on.  Difference arrays (a few hundred values) come back to the host and are
accumulated there with ``np.cumsum``.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

_TINY = float(np.finfo(np.float64).tiny)


def resolve_device(device: Any = None) -> torch.device:
    """The device the diagnostic runs on: the one given, else the current CUDA device, else the CPU."""

    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def as_device_vector(values: Any, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """One flat array on ``device``.  A host array is copied once; a tensor already there is used as is."""

    if torch.is_tensor(values):
        return values.detach().reshape(-1).to(device=device, dtype=dtype)
    array = np.ascontiguousarray(np.asarray(values)).reshape(-1)
    with warnings.catch_warnings():
        # Read-only host arrays (the lowerer's queues) are never written here.
        warnings.simplefilter("ignore", UserWarning)
        host = torch.from_numpy(array)
    return host.to(device=device, dtype=dtype)


def stable_argsort(values: torch.Tensor, *, descending: bool) -> torch.Tensor:
    """``score_order._stable_argsort``: the unique stable permutation (``-0.0`` folded into ``+0.0``)."""

    keyed = (-values if descending else values) + 0.0
    return torch.sort(keyed, stable=True).indices


# ---------------------------------------------------------------------------
# (a) The consequence price: certify_prune_order_stability, scope cross_layer.
# ---------------------------------------------------------------------------


def consequence_price_over_band(
    *,
    score: torch.Tensor,
    standard_error: torch.Tensor,
    route_damage: torch.Tensor,
    route_damage_standard_error: torch.Tensor,
    retained_fractions: Sequence[float],
    critical_value: float,
) -> tuple[float, int, float, float]:
    """``score_order.consequence_price_over_band`` with every route a certain member.

    The cross-layer call has no unresolved-sign routes, so at fraction ``f`` the
    cut is the one count ``k = ceil(f * size)`` (the host's ``low_count`` and
    ``high_count`` coincide).  The two interval sorts and the damage order are
    shared by every fraction; each fraction is a few passes over the scope with
    no host read, and one read brings back every fraction's band, free slots,
    price and price variance.  The binding fraction is then chosen on the host
    exactly as the host does: the first maximum of ``(price, band_size)``.
    Returns ``(price, band_size, retained_fraction, price_standard_error)``.
    """

    size = int(score.numel())
    if size < 2:
        return 0.0, 0, 0.0, 0.0
    fractions: list[float] = []
    counts: list[int] = []
    for fraction in dict.fromkeys(float(value) for value in retained_fractions):
        count = int(math.ceil(fraction * size))
        if count <= 0 or count >= size:  # the host's high_count <= 0 or low_count >= size
            continue
        fractions.append(fraction)
        counts.append(count)
    if not counts:
        return 0.0, 0, 0.0, 0.0
    device = score.device
    critical = float(critical_value)
    lower = score - critical * standard_error
    upper = score + critical * standard_error
    sorted_upper = torch.sort(upper).values
    sorted_lower = torch.sort(lower).values
    order = stable_argsort(route_damage, descending=False)
    damage = route_damage[order]
    variance = torch.square(route_damage_standard_error)[order]
    lower_by_damage = lower[order]
    upper_by_damage = upper[order]
    del lower, upper, order
    count = torch.tensor(counts, dtype=torch.int64, device=device)
    # minimum_overtakers = count (>= 1); retained_count = count.
    threshold_upper = sorted_upper[size - count - 1]
    threshold_lower = sorted_lower[size - count]
    # #{upper >= threshold_lower}: the certainly retained routes plus the band.
    upper_at_least = size - torch.searchsorted(sorted_upper, threshold_lower, side="left")
    rows = []
    for index in range(len(counts)):
        band = (lower_by_damage < threshold_upper[index]) & (
            upper_by_damage >= threshold_lower[index]
        )
        rank = torch.cumsum(band, 0, dtype=torch.int32)  # 1-based rank within the band
        band_size = rank[-1].to(torch.int64)
        free = count[index] - (upper_at_least[index] - band_size)
        # The band's ``free`` smallest and largest damages, in damage order.
        low = band & (rank <= free)
        high = band & (rank > band_size - free)
        # The two sets overlap when free > band_size / 2; as on the host, an
        # overlapping route cancels in the price and counts twice in the variance.
        price = torch.where(high, damage, 0.0).sum() - torch.where(low, damage, 0.0).sum()
        price_variance = torch.where(high, variance, 0.0).sum() + torch.where(low, variance, 0.0).sum()
        rows.append(
            torch.stack(
                (band_size.to(torch.float64), free.to(torch.float64), price, price_variance)
            )
        )
    table = torch.stack(rows).cpu().numpy()
    best_price, best_band, best_fraction, best_price_se = 0.0, 0, 0.0, 0.0
    for fraction, (band_value, free_value, price_value, variance_value) in zip(fractions, table):
        band_size_value, free_slots = int(band_value), int(free_value)
        if band_size_value == 0 or free_slots <= 0 or free_slots >= band_size_value:
            continue
        if (float(price_value), band_size_value) > (best_price, best_band):
            best_price = float(price_value)
            best_band = band_size_value
            best_fraction = float(fraction)
            best_price_se = float(math.sqrt(float(variance_value)))
    return best_price, best_band, best_fraction, best_price_se


def decision_functional_by_fraction(
    *, score: torch.Tensor, route_damage: torch.Tensor, retained_fractions: Sequence[float]
) -> tuple[tuple[float, float], ...]:
    """D(M_k) at every audited fraction: the damage summed over the top ``ceil(f * size)`` scores.

    The host's cumulative sum over one stable descending score order, read at
    each cut; here each cut is the sum of one prefix of that order.
    """

    size = int(score.numel())
    damage = route_damage[stable_argsort(score, descending=True)]
    kept: list[float] = []
    sums = []
    for fraction in dict.fromkeys(float(value) for value in retained_fractions):
        retained = int(math.ceil(fraction * size))
        if retained <= 0 or retained > size:
            continue
        kept.append(fraction)
        sums.append(damage[:retained].sum())
    if not sums:
        return ()
    values = torch.stack(sums).cpu().numpy()
    return tuple((float(fraction), float(value)) for fraction, value in zip(kept, values))


def certify_cross_layer_order_stability(
    *,
    current_global_activity: torch.Tensor,
    current_partition_labels: torch.Tensor,
    score_standard_error: torch.Tensor,
    route_damage: torch.Tensor,
    route_damage_standard_error: torch.Tensor,
    critical_value: float,
    audited_retained_fractions_by_scope: Mapping[str, Sequence[float]],
    maximum_mask_disagreement_activity_fraction: float,
    route_axis_hash: str,
) -> Any:
    """``score_order.certify_prune_order_stability`` for the walk's cross-layer call, on the device.

    The call it ports is the one ``prune_scope_walk.cross_layer_order_certificate``
    describes: ``scope="cross_layer"`` (scopes ``global:partition:0`` and
    ``global:partition:1``), no previous look, ``activity_normalization=
    "global_route_field"``, every pricing input supplied, no unresolved-sign
    routes, no materiality floor or short circuit.  Returns the same
    ``PruneOrderStabilityCertificate``.  With no previous look the
    mask-disagreement fields hold the host's no-previous-look values (1.0) and
    the partition never changes, so the price and the decision functional are
    the only per-scope computations.
    """

    from .score_order import PruneOrderStabilityCertificate, PruneOrderStabilityScopeReceipt

    current = current_global_activity.reshape(-1).to(torch.float64)
    labels = current_partition_labels.reshape(-1)
    errors = score_standard_error.reshape(-1)
    damage = route_damage.reshape(-1)
    damage_errors = route_damage_standard_error.reshape(-1)
    if not (labels.shape == errors.shape == damage.shape == damage_errors.shape == current.shape):
        raise ValueError("prune-order stability axes are incompatible")
    members = (labels == 0, labels == 1)
    # One read: the host's input check, the activity denominator, and each
    # partition's size and largest route activity.
    summary = torch.stack(
        (
            (
                torch.isfinite(current).all()
                & (current >= 0.0).all()
                & (members[0] | members[1]).all()
            ).to(torch.float64),
            current.sum(),
            members[0].sum().to(torch.float64),
            members[1].sum().to(torch.float64),
            torch.where(members[0], current, -math.inf).max(),
            torch.where(members[1], current, -math.inf).max(),
        )
    ).cpu().numpy()
    if not summary[0]:
        raise ValueError("prune-order stability axes are incompatible")
    global_activity_denominator = max(float(summary[1]), _TINY)
    receipts = []
    consequence_price_evaluated = False
    for partition in (0, 1):
        scope_id = f"global:partition:{partition}"
        if scope_id not in audited_retained_fractions_by_scope:
            continue
        scope_fractions = tuple(
            dict.fromkeys(float(value) for value in audited_retained_fractions_by_scope[scope_id])
        )
        if any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in scope_fractions):
            raise ValueError("scope-specific audited retained fractions must lie in (0,1)")
        route_count = int(summary[2 + partition])
        if route_count == 0:
            continue
        maximum_route_activity_fraction = float(summary[4 + partition] / global_activity_denominator)
        price, price_band, price_fraction, price_standard_error = 0.0, 0, 0.0, 0.0
        functional: tuple[tuple[float, float], ...] = ()
        if scope_fractions:
            consequence_price_evaluated = True
            eligible = torch.nonzero(members[partition]).reshape(-1)  # axis order
            scope_scores = current[eligible]
            scope_damage = damage[eligible]
            price, price_band, price_fraction, price_standard_error = consequence_price_over_band(
                score=scope_scores,
                standard_error=errors[eligible],
                route_damage=scope_damage,
                route_damage_standard_error=damage_errors[eligible],
                retained_fractions=scope_fractions,
                critical_value=float(critical_value),
            )
            functional = decision_functional_by_fraction(
                score=scope_scores, route_damage=scope_damage, retained_fractions=scope_fractions
            )
            del eligible, scope_scores, scope_damage
        receipts.append(
            PruneOrderStabilityScopeReceipt(
                scope_id=scope_id,
                route_count=route_count,
                partition_changed_route_count=0,
                audited_boundary_count=(
                    len(scope_fractions) if scope_fractions else max(route_count - 1, 0)
                ),
                maximum_mask_disagreement_route_fraction=0.0,
                maximum_mask_disagreement_activity_fraction=1.0,
                mean_mask_disagreement_activity_fraction=1.0,
                maximum_route_activity_fraction=maximum_route_activity_fraction,
                minimum_feasible_tolerance=float(2.0 * maximum_route_activity_fraction),
                membership_churn_activity_fraction=0.0,
                score_order_activity_fraction=0.0,
                consequence_price=float(price),
                consequence_price_standard_error=float(price_standard_error),
                consequence_band_size=int(price_band),
                consequence_binding_retained_fraction=float(price_fraction),
                consequence_decision_functional_by_fraction=functional,
                consequence_price_deferred=False,
            )
        )
    if not receipts:
        raise ValueError("prune-order stability certificate has no scope")
    return PruneOrderStabilityCertificate(
        maximum_permitted_mask_disagreement_activity_fraction=float(
            maximum_mask_disagreement_activity_fraction
        ),
        maximum_observed_mask_disagreement_activity_fraction=1.0,
        audited_retained_fractions=(),
        scope_receipts=tuple(receipts),
        route_axis_hash=str(route_axis_hash),
        previous_look_available=False,
        activity_normalization="global_route_field",
        consequence_price_evaluated=bool(consequence_price_evaluated),
    )


# ---------------------------------------------------------------------------
# Runs of cuts: difference arrays shared by (b) and (c).
# ---------------------------------------------------------------------------


def _run_differences(
    first: torch.Tensor,
    stop: torch.Tensor,
    active: torch.Tensor,
    layers: torch.Tensor,
    weight: torch.Tensor,
    width: int,
    layer_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Difference arrays of routes counted on the run of cuts ``[first, stop)``.

    Returns per ``(cut, layer)`` counts (``width * layer_count``, int64) and per
    cut weight sums (``width``, float64), +1 / +weight at ``first`` and -1 /
    -weight at ``stop``.  Compact the active runs before histogramming: sending
    millions of inactive routes to one dummy bin serializes CUDA histogram
    atomics and needlessly enlarges deterministic ``index_add_`` reductions.
    """

    bins = width * layer_count
    selected = torch.nonzero(active, as_tuple=True)[0]
    starts = first[selected]
    stops = stop[selected]
    selected_layers = layers[selected]
    start_bin = starts * layer_count + selected_layers
    stop_bin = stops * layer_count + selected_layers
    counts = torch.bincount(start_bin, minlength=bins) - torch.bincount(
        stop_bin, minlength=bins
    )
    weights = weight[selected]
    sums = torch.zeros(width, dtype=torch.float64, device=weight.device)
    sums.index_add_(0, starts, weights)
    sums.index_add_(0, stops, -weights)
    return counts, sums


def _accumulate(
    counts: np.ndarray, sums: np.ndarray, width: int, layer_count: int, cut_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cut counts by layer and weight sums from their difference arrays (host)."""

    by_layer = np.cumsum(counts.reshape(width, layer_count), axis=0)[:cut_count]
    total = np.cumsum(sums)[:cut_count]
    # A cut with no counted route holds exactly zero weight; the difference
    # arrays could leave a rounding residue there.
    total = np.where(by_layer.sum(axis=1) > 0, np.maximum(total, 0.0), 0.0)
    return by_layer, total


# ---------------------------------------------------------------------------
# (b) Boundary-uncertain activity on the non-floor queue.
# ---------------------------------------------------------------------------


def queue_boundary_uncertainty(
    *,
    activity: torch.Tensor,
    standard_error: torch.Tensor,
    layers: torch.Tensor,
    cuts: np.ndarray,
    critical_value: float,
    layer_count: int,
) -> dict[str, Any]:
    """The queue boundary uncertainty on device tensors in queue order.

    ``activity``, ``standard_error`` and ``layers`` follow the queue (descending
    score); ``cuts`` holds distinct kept counts ``1 <= c <= size - 1`` in
    ascending order.  At cut ``c`` the kept routes are positions ``[0, c)``.  A
    kept route is uncertain when ``max(a - z*se, 0)`` is strictly below the
    largest removed ``a + z*se``; a removed route when ``a + z*se`` is strictly
    above the smallest kept ``max(a - z*se, 0)``.  Those extremes are a suffix
    maximum and a prefix minimum read at the cuts; both are monotone in the
    cut, so each route is uncertain on one run of cuts per side.  Returns the
    host function's dictionary (numpy arrays and the queue's activity).
    """

    device = activity.device
    size = int(activity.numel())
    count = int(len(cuts))
    width = count + 1
    z = float(critical_value)
    upper = activity + z * standard_error
    lower = torch.clamp_min(activity - z * standard_error, 0.0)
    cut = torch.as_tensor(np.asarray(cuts, dtype=np.int64), device=device)
    # At cut k: the largest upper end over positions >= cuts[k] (removed) and
    # the smallest lower end over positions < cuts[k] (kept); both non-increasing.
    removed_upper = torch.cummax(upper.flip(0), 0).values.flip(0)[cut]
    kept_lower = torch.cummin(lower, 0).values[cut - 1]
    position = torch.arange(size, dtype=torch.int64, device=device)
    # Number of cuts at or below each position: the route is kept at cut k
    # exactly when k >= this number, and removed when k < it.
    segment = torch.searchsorted(cut, position, right=True)
    del position
    # Kept side: uncertain while removed_upper[k] > lower (a prefix of k).
    kept_stop = torch.searchsorted(-removed_upper, -lower, side="left")
    # Removed side: uncertain once kept_lower[k] < upper (a suffix of k).
    removed_first = torch.searchsorted(-kept_lower, -upper, side="right")
    del lower, upper
    kept_counts, kept_sums = _run_differences(
        segment, kept_stop, segment < kept_stop, layers, activity, width, layer_count
    )
    removed_counts, removed_sums = _run_differences(
        removed_first, segment, removed_first < segment, layers, activity, width, layer_count
    )
    del segment, kept_stop, removed_first
    host = [
        tensor.cpu().numpy()
        for tensor in (
            kept_counts, kept_sums, removed_counts, removed_sums, removed_upper, kept_lower,
            activity.sum().reshape(1),
        )
    ]
    kept_by_layer, kept_activity = _accumulate(host[0], host[1], width, layer_count, count)
    removed_by_layer, removed_activity = _accumulate(host[2], host[3], width, layer_count, count)
    return {
        "kept_counts_by_layer": kept_by_layer,
        "removed_counts_by_layer": removed_by_layer,
        "kept_activity": kept_activity,
        "removed_activity": removed_activity,
        "largest_removed_upper": host[4],
        "smallest_kept_lower": host[5],
        "queue_activity": float(host[6][0]),
    }


# ---------------------------------------------------------------------------
# (c) The descriptive pointwise band.
# ---------------------------------------------------------------------------


def pointwise_band(
    *,
    scores: torch.Tensor,
    standard_error: torch.Tensor,
    layers: torch.Tensor,
    cuts: np.ndarray,
    critical_value: float,
    layer_count: int,
) -> dict[str, np.ndarray]:
    """The descriptive band at every cut of one queue, all cuts in one pass.

    ``scores`` (descending), ``standard_error`` and ``layers`` follow the queue;
    ``cuts`` holds distinct kept counts ``1 <= c <= size - 1`` in ascending
    order.  At cut ``c`` the threshold is ``t = 0.5 * (s[c-1] + s[c])`` and the
    window is ``[low, high)`` with ``low = #{s > t + z*se_max}`` and
    ``high = #{s >= t - z*se_max}``.  A route at position ``q`` is counted when
    ``low <= q < high`` and, kept (``q < c``), ``s - z*se <= t``, or removed
    (``q >= c``), ``s + z*se >= t``.  ``t``, ``low`` and ``high`` are monotone in
    the cut, so each condition holds on a prefix or a suffix of the cuts and a
    route is counted on one run of cuts per side.  Returns per cut: the
    threshold, the kept and removed counts by layer, and the counted activity.
    """

    device = scores.device
    size = int(scores.numel())
    count = int(len(cuts))
    width = count + 1
    z = float(critical_value)
    cut = torch.as_tensor(np.asarray(cuts, dtype=np.int64), device=device)
    threshold = 0.5 * (scores[cut - 1] + scores[cut])  # non-increasing in the cut
    spread = z * standard_error.max()
    keys = -scores  # ascending
    low = torch.searchsorted(keys, -(threshold + spread), side="left")  # non-decreasing
    high = torch.searchsorted(keys, -(threshold - spread), side="right")  # non-decreasing
    negative_threshold = -threshold  # ascending
    position = torch.arange(size, dtype=torch.int64, device=device)
    below = torch.searchsorted(cut, position, right=True)  # #{k : c_k <= q}
    inside_low = torch.searchsorted(low, position, right=True)  # #{k : low_k <= q}
    outside_high = torch.searchsorted(high, position, right=True)  # #{k : high_k <= q}
    del position
    # #{k : t_k >= s - z*se}: kept routes are counted on a prefix of the cuts.
    kept_value = torch.searchsorted(
        negative_threshold, -(scores - z * standard_error), side="right"
    )
    # #{k : t_k > s + z*se}: removed routes are counted on a suffix of the cuts.
    removed_value = torch.searchsorted(
        negative_threshold, -(scores + z * standard_error), side="left"
    )
    # Kept at k: k >= below (q < c_k), k >= outside_high (q < high_k),
    # k < inside_low (low_k <= q), k < kept_value.
    kept_first = torch.maximum(below, outside_high)
    kept_stop = torch.minimum(inside_low, kept_value)
    # Removed at k: k < below (q >= c_k), k < inside_low, k >= outside_high,
    # k >= removed_value.
    removed_first = torch.maximum(outside_high, removed_value)
    removed_stop = torch.minimum(below, inside_low)
    del below, inside_low, outside_high, kept_value, removed_value
    kept_counts, kept_sums = _run_differences(
        kept_first, kept_stop, kept_first < kept_stop, layers, scores, width, layer_count
    )
    removed_counts, removed_sums = _run_differences(
        removed_first, removed_stop, removed_first < removed_stop, layers, scores, width, layer_count
    )
    del kept_first, kept_stop, removed_first, removed_stop
    host = [
        tensor.cpu().numpy()
        for tensor in (kept_counts, kept_sums, removed_counts, removed_sums, threshold)
    ]
    kept_by_layer, kept_activity = _accumulate(host[0], host[1], width, layer_count, count)
    removed_by_layer, removed_activity = _accumulate(host[2], host[3], width, layer_count, count)
    return {
        "threshold": host[4],
        "kept_counts_by_layer": kept_by_layer,
        "removed_counts_by_layer": removed_by_layer,
        "activity": kept_activity + removed_activity,
    }


__all__ = [
    "as_device_vector",
    "certify_cross_layer_order_stability",
    "consequence_price_over_band",
    "decision_functional_by_fraction",
    "pointwise_band",
    "queue_boundary_uncertainty",
    "resolve_device",
    "stable_argsort",
]
