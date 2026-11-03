"""Bounded tensor batches for the census's retained-fraction readouts.

All bands share their sorts and rank vectors. Fixed route tiles replace
per-band nonzero compaction, pairwise plans, launches and scalar reads.
"""
from __future__ import annotations

import math
import torch

_ROW_TILE = 16384


def consequence(*, score, standard_error, route_damage, retained_fractions,
                critical_value, route_damage_standard_error, certain_member,
                state, decision_band=0.0, decision_flags=None):
    phase = "census_consequence_bands"
    values, errors, damage = score.reshape(-1).to(torch.float64), standard_error.reshape(-1).to(torch.float64), route_damage.reshape(-1).to(torch.float64)
    size = values.numel()
    fractions = tuple(dict.fromkeys(float(f) for f in retained_fractions))
    if size < 2 or not fractions:
        return 0.0, 0, 0.0, 0.0
    if (errors.shape != values.shape or damage.shape != values.shape
            or not fractions or any(not math.isfinite(f) or not 0 <= f <= 1 for f in fractions)
            or (certain_member is not None and certain_member.numel() != size)
            or (route_damage_standard_error is not None and route_damage_standard_error.numel() != size)):
        raise ValueError(f"{phase} input axes differ")
    device = values.device
    f = torch.tensor(fractions, device=device, dtype=torch.float64)
    lower, upper = values - critical_value * errors, values + critical_value * errors
    sorted_lower, sorted_upper = torch.sort(lower).values, torch.sort(upper).values
    # Stable C-order tie breaking matches the incumbent damage sort.
    order = torch.argsort(damage, stable=True)
    dd, lo, hi = damage[order], lower[order], upper[order]
    vv = None if route_damage_standard_error is None else route_damage_standard_error.reshape(-1).to(torch.float64)[order].square()
    count = (torch.tensor(size, device=device) if certain_member is None
             else certain_member.reshape(-1).count_nonzero())
    low_count, high_count = torch.ceil(f * count).long(), torch.ceil(f * size).long()
    minimum_overtakers = low_count.clamp_min(1)
    threshold_upper = sorted_upper[(size - minimum_overtakers - 1).clamp(0, size - 1)]
    threshold_lower = sorted_lower[(size - high_count).clamp(0, size - 1)]
    bands = torch.zeros(len(fractions), device=device, dtype=torch.int64)
    for start in range(0, size, _ROW_TILE):
        stop = min(start + _ROW_TILE, size)
        band = (lo[start:stop].unsqueeze(0) < threshold_upper.unsqueeze(1)) & (
            hi[start:stop].unsqueeze(0) >= threshold_lower.unsqueeze(1))
        bands += band.sum(1)
    certainly_retained = size - torch.searchsorted(sorted_upper, threshold_lower, side="left") - bands
    slots = high_count - certainly_retained
    valid_band = ((high_count > 0) & (low_count < size) & (size - minimum_overtakers - 1 >= 0)
                  & (bands > 0) & (slots > 0) & (slots < bands))
    low_sum, high_sum = torch.zeros_like(f), torch.zeros_like(f)
    low_var, high_var = torch.zeros_like(f), torch.zeros_like(f)
    prior = torch.zeros_like(bands)
    for start in range(0, size, _ROW_TILE):
        stop = min(start + _ROW_TILE, size)
        band = (lo[start:stop].unsqueeze(0) < threshold_upper.unsqueeze(1)) & (
            hi[start:stop].unsqueeze(0) >= threshold_lower.unsqueeze(1))
        rank = band.cumsum(1) + prior.unsqueeze(1)
        low = band & (rank <= slots.unsqueeze(1))
        high = band & (rank > (bands - slots).unsqueeze(1))
        low_sum += torch.where(low, dd[start:stop], 0).sum(1)
        high_sum += torch.where(high, dd[start:stop], 0).sum(1)
        if vv is not None:
            low_var += torch.where(low, vv[start:stop], 0).sum(1)
            high_var += torch.where(high, vv[start:stop], 0).sum(1)
        prior += band.sum(1)
    prices = high_sum - low_sum
    errors = (high_var + low_var).sqrt()
    finite = torch.isfinite(lower).all() & torch.isfinite(upper).all() & torch.isfinite(damage).all()
    if vv is not None:
        finite &= torch.isfinite(vv).all()
    finite &= torch.isfinite(prices).all() & torch.isfinite(errors).all()
    n = len(fractions)
    indices = torch.arange(n + 1, device=device)
    candidate_prices = torch.cat((prices.new_zeros(1), torch.where(valid_band, prices, -torch.inf)))
    candidate_bands = torch.cat((bands.new_zeros(1), bands))
    maximum_price = candidate_prices.max()
    maximum_band = torch.where(candidate_prices == maximum_price, candidate_bands, -1).max()
    winner = torch.where((candidate_prices == maximum_price) & (candidate_bands == maximum_band),
                         indices, n + 1).min()
    result = torch.stack((maximum_price, maximum_band.to(torch.float64),
        torch.cat((f.new_zeros(1), f))[winner], torch.cat((errors.new_zeros(1), errors))[winner]))
    packed = torch.cat((result, finite.to(torch.float64).reshape(1))).detach().cpu().numpy()
    state.reads[phase] += 1
    if not packed[-1]:
        raise ValueError(f"{phase} contains nonfinite values")
    best = (float(packed[0]), int(packed[1]), float(packed[2]), float(packed[3]))
    state.calls[phase] += 1
    return best


def disagreement(*, previous_score, current_score, activity_weight, previous_member,
                 current_member, retained_fractions, activity_denominator, state):
    phase = "census_disagreement_bands"
    previous, current, weight = previous_score.reshape(-1).to(torch.float64), current_score.reshape(-1).to(torch.float64), activity_weight.reshape(-1).to(torch.float64)
    p_in, c_in = previous_member.reshape(-1).bool(), current_member.reshape(-1).bool()
    fractions = tuple(dict.fromkeys(float(f) for f in retained_fractions))
    size = current.numel()
    if not fractions or any(not math.isfinite(f) or not 0 < f < 1 for f in fractions):
        raise ValueError("retained fractions must lie strictly between 0 and 1")
    if any(v.numel() != size for v in (previous, weight, p_in, c_in)):
        raise ValueError(f"{phase} input axes differ")
    # Keep full fixed axes; nonmembers sort after members and receive no rank.
    device = current.device
    sentinel = torch.tensor(float("-inf"), device=device)
    p_order = torch.argsort(torch.where(p_in, previous, sentinel), descending=True, stable=True)
    c_order = torch.argsort(torch.where(c_in, current, sentinel), descending=True, stable=True)
    p_rank, c_rank = torch.empty_like(p_order), torch.empty_like(c_order)
    arange = torch.arange(size, device=device)
    p_rank.scatter_(0, p_order, arange)
    c_rank.scatter_(0, c_order, arange)
    f = torch.tensor(fractions, device=device, dtype=torch.float64)
    p_count, c_count = torch.ceil(f * p_in.sum()).long(), torch.ceil(f * c_in.sum()).long()
    union, stable = p_in | c_in, p_in & c_in
    mass = (torch.where(union, weight, 0).sum() if activity_denominator is None
            else torch.tensor(activity_denominator, device=device, dtype=torch.float64))
    total = mass.clamp_min(torch.finfo(torch.float64).tiny)
    routes = torch.zeros_like(f)
    activity, churn, order = torch.zeros_like(f), torch.zeros_like(f), torch.zeros_like(f)
    for start in range(0, size, _ROW_TILE):
        stop = min(start + _ROW_TILE, size)
        p_kept = p_in[start:stop] & (p_rank[start:stop].unsqueeze(0) < p_count.unsqueeze(1))
        c_kept = c_in[start:stop] & (c_rank[start:stop].unsqueeze(0) < c_count.unsqueeze(1))
        changed = (p_kept ^ c_kept) & union[start:stop]
        routes += changed.sum(1)
        activity += torch.where(changed, weight[start:stop], 0).sum(1)
        churn += torch.where(changed & ~stable[start:stop], weight[start:stop], 0).sum(1)
        order += torch.where(changed & stable[start:stop], weight[start:stop], 0).sum(1)
    finite = (torch.isfinite(previous).all() & torch.isfinite(current).all()
              & torch.isfinite(weight).all() & torch.isfinite(mass)
              & torch.isfinite(activity / total).all())
    peak, binding_tensor = activity.max(), activity.argmax()
    churn_fraction, order_fraction = churn / total, order / total
    result = torch.stack((routes.max(), peak / total, (activity / total).mean(),
                          f.new_tensor(len(fractions)), churn_fraction[binding_tensor],
                          order_fraction[binding_tensor]))
    packed = torch.cat((result, finite.to(torch.float64).reshape(1))).detach().cpu().numpy()
    state.reads[phase] += 1
    if not packed[-1]:
        raise ValueError(f"{phase} contains nonfinite values")
    actual = (int(packed[0]), float(packed[1]), float(packed[2]),
              int(packed[3]), float(packed[4]), float(packed[5]))
    state.calls[phase] += 1
    return actual
