"""Batched rejection screens enclosing the retained scalar reduction.

The eager pointwise expressions match the scalar screen. Only reduction layout
changes. A floating-sum enclosure covers either addition order; interval
operations then follow the scalar dtype's monotone rounded arithmetic. Only
proved scalar rejections may bypass scalar screening. All other pairs use the
retained scalar implementation, including every returned candidate.

This is a numerical comparison with that implementation, not a bound on true
population work, nonlinear intervention validity, or statistical acceptance.
"""
from __future__ import annotations

import math

import torch


def sum_reference_enclosure(terms: torch.Tensor):
    """Enclose a same-dtype scalar sum of each contiguous row's exact terms.

    For n finite terms of magnitude at most M, either addition order differs
    from the real sum by at most gamma_n*n*M, absent overflow. Twice this bound
    encloses the other order around the batched sum. An extra 8*n*tiny covers
    gradual underflow or flushing subnormal inputs/results for both reductions;
    another 8*n*tiny protects bound construction itself against flushing.
    We require an outward n*M below max/2, so intermediate partial sums cannot
    overflow. Unsupported/unsafe rows never authorize a rejection.
    """
    if terms.ndim != 2 or terms.shape[1] < 1 or terms.dtype not in (torch.float32, torch.float64):
        raise ValueError('sum enclosure requires nonempty FP32/FP64 rows')
    n = int(terms.shape[1])
    info = torch.finfo(terms.dtype)
    inf = math.inf
    nu = math.nextafter(n * (info.eps/2), inf)
    if nu >= .5:
        return (torch.full_like(terms[:, 0], -inf), torch.full_like(terms[:, 0], inf),
                torch.zeros_like(terms[:, 0], dtype=torch.bool))
    denominator = math.nextafter(1.-nu, -inf)
    gamma = math.nextafter(nu/denominator, inf)
    coefficient = math.nextafter(math.nextafter(2.*gamma, inf)*n, inf)
    underflow = math.nextafter(16.*n*info.tiny, inf)
    maximum = terms.abs().amax(dim=1).to(torch.float64)
    upper_inf = torch.full_like(maximum, inf)
    magnitude = torch.nextafter(maximum*n, upper_inf)
    error = torch.nextafter(maximum*coefficient, upper_inf)
    error = torch.nextafter(error+underflow, upper_inf)
    fast = terms.sum(dim=1)
    lower64 = torch.nextafter(fast.to(torch.float64)-error, -upper_inf)
    upper64 = torch.nextafter(fast.to(torch.float64)+error, upper_inf)
    # nextafter alone need not move a flushed zero into the subnormal range.
    # A smallest-normal allowance also protects subtraction/cast endpoints.
    lower = torch.nextafter(lower64.to(terms.dtype)-info.tiny, torch.full_like(fast, -inf))
    upper = torch.nextafter(upper64.to(terms.dtype)+info.tiny, torch.full_like(fast, inf))
    safe = torch.isfinite(maximum) & torch.isfinite(fast) & (magnitude < info.max/2)
    return lower, upper, safe


def _scale(lower, upper, value):
    # The scalar operand is converted by the same tensor-dtype rule as in the
    # retained expression. Min/max handles either coefficient sign.
    a, b = lower*value, upper*value
    return torch.minimum(a, b), torch.maximum(a, b)


def reference_screen_rejections(*, dvals, avals, jd, ja, db, ab, dw, aw,
                                response, weights, diag_delta, same, legal,
                                base_work, base_energy, alpha, total_mass, radius):
    """Return only pairs that cannot pass the retained scalar F/Q screen.

    All row arrays are [pairs, contexts], with each row contiguous. The returned
    Boolean vector is not a list of feasible candidates; false means scalar
    screening remains necessary. No comparison against surrogate priority is
    used. Returned candidates always retain scalar work/energy metadata.
    """
    dlo, dhi, dsafe = sum_reference_enclosure(dw*((db-dvals).abs()-db.abs()))
    alo, ahi, asafe = sum_reference_enclosure(aw*((ab+avals).abs()-ab.abs()))
    slo, shi, ssafe = sum_reference_enclosure(dw*((db-dvals+avals).abs()-db.abs()))
    wlo = base_work + torch.where(same, slo, dlo+alo)
    whi = base_work + torch.where(same, shi, dhi+ahi)
    work_safe = torch.where(same, ssafe, dsafe & asafe)
    delta = -jd+ja
    elo, ehi, esafe = sum_reference_enclosure(weights*((response+delta)**2-response**2))
    elo, ehi = _scale(elo, ehi, 1.-alpha)
    # diag_delta and its scaled term use unchanged eager pointwise arithmetic.
    diagonal = alpha*diag_delta
    elo, ehi = elo+diagonal, ehi+diagonal
    elo, ehi = _scale(elo, ehi, total_mass)
    elo, ehi = base_energy+elo, base_energy+ehi
    work_safe = work_safe & torch.isfinite(wlo) & torch.isfinite(whi)
    esafe = esafe & torch.isfinite(elo) & torch.isfinite(ehi)
    return (~legal) | (work_safe & (wlo >= base_work)) | (esafe & (.5*elo > radius))
