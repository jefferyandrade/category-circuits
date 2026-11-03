"""Device solve of the per-candidate proper-score calibration.

The algorithm is the host route's, unchanged: ``_fit_proper_score_shrinkage_affine_core``
over ``fit_batched_post_query_affine`` under the v6 device hard gates, for one
binary readout column (the decision fit).  Inner-block x strength selection
cells are solved together, the one-SE rule picks the largest eligible anchor,
the deployment refit uses the selected strength (identity is fixed at (1, 0)),
and the returned parameters pass the v6 hard gates.  The two-parameter
stratified uncertainty is computed in the same pass.

The projected Newton solve runs a fixed number of masked iterations; a
converged or failed cell's parameters remain frozen. CPU execution evaluates
the backtracking ladder as tensors, and CUDA execution uses one per-cell
kernel per iteration. Both select the first sufficient step. The packet
retains fit parameters, uncertainty and convergence diagnostics on device.
During a CUDA look, input and solve failures join its control word and end
the execution; numerical records are materialized after control completes.
"""

from __future__ import annotations

import math
from mechanistic_systems.measurement.run_products import loss_weighting

from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import torch

from .cuda_graphs import initialize_cuda_graph_state
import torch.nn.functional as F

from .device_newton_step import ROW_BLOCK

# The host ladder (_guarded_projected_newton's line_search_limit default).
LINE_SEARCH_LIMIT = 30
# Initial masked iteration chunk for explicit fits outside a device look.
# Device looks use the declared full iteration bound without host polling.
CHUNK_ITERATIONS = 8

# Packed status words (float64, zero means failure). Uncertainty uses 1 for
# the original analytic route, 2 for inverse recovery, 3 for covariance
# recovery and 4 for both; all other words remain 0/1.
STATUS_FIELDS = (
    "inputs_valid",       # host route's input preconditions hold
    "selection_done",     # no selection cell still iterating at the bound used
    "deployment_done",    # same for the deployment refit
    "certified",          # every enabled cell passes the v6 hard gates
    "uncertainty_valid",  # analytic spectra resolved, or recorded Firth recovery
    "labels_match",       # delivered labels equal the source design's labels
)


def _reduce(x, t, w, scale, offset, *, row_block=None):
    """``torch_calibration_statistics`` with an optional row partition.

    CPU Newton supplies the tile width. Other fit diagnostics keep Torch's
    full-row reductions, including the diagnostics in a captured CUDA fit.
    """
    block = max(1, x.shape[-1]) if row_block is None else row_block
    sums = None
    for start in range(0, max(1, x.shape[-1]), block):
        row = slice(start, start + block)
        values, target, weight = x[..., row], t[..., row], w[..., row]
        eta = values * scale.unsqueeze(-1) + offset.unsqueeze(-1)
        loss = (F.softplus(eta) - target * eta) * weight
        probability = torch.sigmoid(eta)
        residual = (probability - target) * weight
        curvature = probability * (1.0 - probability) * weight
        parts = (loss.sum(-1), (residual * values).sum(-1), residual.sum(-1),
                 (curvature * values.square()).sum(-1), (curvature * values).sum(-1),
                 curvature.sum(-1))
        sums = parts if sums is None else tuple(total + part for total, part in zip(sums, parts))
    n = float(x.shape[-1])
    return tuple(total / n for total in sums)


def _objective(x, t, w, scale, offset, *, row_block=None):
    """Row-partitioned objective; scale/offset may carry a ladder axis."""
    block = max(1, x.shape[-1]) if row_block is None else row_block
    loss = None
    for start in range(0, max(1, x.shape[-1]), block):
        row = slice(start, start + block)
        eta = x[..., row] * scale.unsqueeze(-1) + offset.unsqueeze(-1)
        part = ((F.softplus(eta) - t[..., row] * eta) * w[..., row]).sum(-1)
        loss = part if loss is None else loss + part
    return loss / float(x.shape[-1])


def guarded_newton(x, t, w, constant, scale, offset, minimum, live, *, iterations,
                   tolerance, hessian_ridge, objective_tolerance, start=0):
    """``_guarded_projected_newton`` without host polls.

    Inputs are ``[cells, n]`` rows and ``[cells]`` parameters.  ``start`` is the
    number of iterations already run (a continuation repeats from zero, so it
    is always zero here; it keeps iteration counts aligned with the host's).
    Returns ``(scale, offset, converged, failed_line_search, iterations,
    line_steps, pending)`` where ``pending`` marks cells still active after the
    last iteration, which the host loop would have continued.
    """
    if x.device.type == "cuda":
        from .device_newton_step import guarded_newton as cuda_guarded_newton
        return cuda_guarded_newton(x,t,w,constant,scale,offset,minimum,live,
            iterations=iterations,tolerance=tolerance,hessian_ridge=hessian_ridge,
            objective_tolerance=objective_tolerance,start=start)
    ridge = float(hessian_ridge)
    floor = max(ridge * ridge, 1e-30)
    tol = float(tolerance)
    boundary_slack = max(tol, 1e-12)
    alphas = torch.pow(torch.full((LINE_SEARCH_LIMIT, 1), 0.5, dtype=torch.float64,
                                  device=scale.device),
                       torch.arange(LINE_SEARCH_LIMIT, dtype=torch.float64,
                                    device=scale.device).unsqueeze(1))
    converged = torch.zeros_like(live)
    failed = torch.zeros_like(live)
    counts = torch.zeros(scale.shape, dtype=torch.int64, device=scale.device)
    steps = torch.zeros_like(counts)
    for iteration in range(start + 1, start + int(iterations) + 1):
        objective, gs, go, hss, hso, hoo = _reduce(x, t, w, scale, offset, row_block=ROW_BLOCK)
        objective = objective - constant
        at_boundary = scale <= minimum + boundary_slack
        kkt = torch.maximum(torch.where(at_boundary, torch.clamp(-gs, min=0.0), gs.abs()), go.abs())
        converged = converged | (kkt <= tol)
        active = live & ~(converged | failed)
        counts = torch.where(active, torch.full_like(counts, iteration), counts)
        hss = hss + ridge
        hoo = hoo + ridge
        determinant = hss * hoo - hso.square()
        stable = determinant > floor
        determinant = determinant.clamp_min(floor)
        diagonal_o = go / hoo.clamp_min(ridge)
        step_s = torch.where(stable, (hoo * gs - hso * go) / determinant, gs / hss.clamp_min(ridge))
        step_o = torch.where(stable, (hss * go - hso * gs) / determinant, diagonal_o)
        bound = at_boundary & (gs >= 0.0)
        step_s = torch.where(bound, torch.zeros_like(step_s), step_s)
        step_o = torch.where(bound, diagonal_o, step_o)
        proposal_s = torch.maximum(scale + alphas * (-step_s), minimum)
        proposal_o = offset + alphas * (-step_o)
        proposal = _objective(x, t, w, proposal_s, proposal_o, row_block=ROW_BLOCK) - constant
        directional = gs * (proposal_s - scale) + go * (proposal_o - offset)
        sufficient = (proposal <= objective + 1e-4 * directional + float(objective_tolerance)) \
            & torch.isfinite(proposal)
        any_sufficient = sufficient.any(0)
        first = sufficient.to(torch.int64).argmax(0)
        take = active & any_sufficient
        scale = torch.where(take, proposal_s.gather(0, first.unsqueeze(0)).squeeze(0), scale)
        offset = torch.where(take, proposal_o.gather(0, first.unsqueeze(0)).squeeze(0), offset)
        steps = torch.where(take, first + 1, steps)
        failed = failed | (active & ~any_sufficient)
    # The host's next iteration would re-test KKT before stepping.
    objective, gs, go, *_ = _reduce(x, t, w, scale, offset, row_block=ROW_BLOCK)
    at_boundary = scale <= minimum + boundary_slack
    kkt = torch.maximum(torch.where(at_boundary, torch.clamp(-gs, min=0.0), gs.abs()), go.abs())
    pending = live & ~(converged | (kkt <= tol) | failed)
    return scale, offset, converged, failed, counts, steps, pending


@dataclass
class CellFit:
    """Per-cell results; tensors only (no host values)."""
    scale: Any            # raw float64 scale
    offset: Any           # raw float64 offset
    certified: Any        # every v6 hard gate holds (or the cell is disabled)
    pending: Any          # still iterating at the bound used
    inputs_valid: Any     # host route's input preconditions hold
    diagnostics: Any      # [len(CELL_FIELDS), cells] float64
    effective_sample_size: Any = None  # Kish size, Firth fits only


CELL_FIELDS = ("initial_objective", "final_objective", "identity_objective",
               "intercept_only_objective", "iterations", "line_search_steps",
               "gradient_kkt_residual", "finite", "converged", "scale_identifiable",
               "separated", "objective_worsened", "failed_line_search")


def _firth_statistics(x, y, w, scale, offset, inverse_sample_size, identifiable,
                      fixed_offsets=None, *, score=True):
    """Normalized weighted Firth objective and modified-score Fisher matrix.

    All operations stay on the tensors' device. The intercept-only quotient
    is used for a numerically constant readout, as in the ordinary solver.
    The Jeffreys determinant is never regularized; the Newton solve alone
    uses the declared numerical ridge.
    """
    loss = hss = hso = hoo = gs = go = None
    block = ROW_BLOCK
    for start in range(0, x.shape[-1], block):
        row = slice(start, start + block)
        u, target, weight = x[..., row], y[..., row], w[..., row]
        eta = u * scale.unsqueeze(-1) + offset.unsqueeze(-1)
        if fixed_offsets is not None:
            eta = eta + fixed_offsets[..., row]
        p = torch.sigmoid(eta)
        v = weight * p * torch.sigmoid(-eta)
        residual = weight * (p - target)
        parts = (((F.softplus(eta) - target * eta) * weight).sum(-1),
                 (v * u.square()).sum(-1), (v * u).sum(-1), v.sum(-1),
                 (residual * u).sum(-1), residual.sum(-1))
        if loss is None:
            loss, hss, hso, hoo, gs, go = parts
        else:
            loss, hss, hso, hoo, gs, go = tuple(
                a + b for a, b in zip((loss, hss, hso, hoo, gs, go), parts))
    determinant = torch.where(identifiable, hss * hoo - hso.square(), hoo)
    # Clamping only protects intermediate divisions. A singular information
    # matrix has infinite objective and cannot pass the fit's finite gate.
    safe_det = determinant.clamp_min(1e-300)
    objective = loss - 0.5 * inverse_sample_size * torch.log(safe_det)
    objective = torch.where(determinant > 0.0, objective,
                            torch.full_like(objective, math.inf))
    if not score:
        return objective
    correction_s = torch.zeros_like(scale)
    correction_o = torch.zeros_like(offset)
    for start in range(0, x.shape[-1], block):
        row = slice(start, start + block)
        u, weight = x[..., row], w[..., row]
        eta = u * scale.unsqueeze(-1) + offset.unsqueeze(-1)
        if fixed_offsets is not None:
            eta = eta + fixed_offsets[..., row]
        p = torch.sigmoid(eta)
        v = weight * p * torch.sigmoid(-eta)
        leverage = v * torch.where(
            identifiable.unsqueeze(-1),
            (hoo.unsqueeze(-1) * u.square() - 2.0 * hso.unsqueeze(-1) * u
             + hss.unsqueeze(-1)) / safe_det.unsqueeze(-1),
            1.0 / hoo.clamp_min(1e-300).unsqueeze(-1))
        adjustment = leverage * (0.5 - p)
        correction_s = correction_s + (adjustment * u).sum(-1)
        correction_o = correction_o + adjustment.sum(-1)
    gs = torch.where(identifiable, gs - inverse_sample_size * correction_s,
                     torch.zeros_like(gs))
    go = go - inverse_sample_size * correction_o
    return objective, gs, go, hss, hso, hoo


def fixed_offset_mean_probability_intercept(binary_targets, sample_weights, fixed_offsets):
    """Initialize a fixed-offset intercept at the weighted target prevalence.

    The last axis contains rows; preceding axes identify independent fits.
    This is only a starting value, without a Firth or identity-anchor penalty.
    The monotone probability equation is bracketed analytically and bisected
    in float64, so large dense competitor logits do not distort its start.
    """
    fixed = torch.as_tensor(fixed_offsets, dtype=torch.float64)
    target = torch.as_tensor(binary_targets, device=fixed.device, dtype=torch.float64)
    weights = torch.as_tensor(sample_weights, device=fixed.device, dtype=torch.float64)
    target, weights, fixed = torch.broadcast_tensors(target, weights, fixed)
    if not bool((torch.isfinite(target) & torch.isfinite(weights) & torch.isfinite(fixed)
                 & (weights >= 0.0)).all().item()):
        raise ValueError("fixed-offset initialization requires finite rows and nonnegative weights")
    mass = weights.sum(-1)
    positive = (weights * target).sum(-1)
    if not bool(((mass > 0.0) & (positive > 0.0) & (positive < mass)).all().item()):
        raise ValueError("fixed-offset initialization requires both target outcomes")
    weights = weights / mass.unsqueeze(-1)
    prevalence = positive / mass
    log_odds = torch.log(prevalence) - torch.log1p(-prevalence)
    lower = log_odds - fixed.amax(-1)
    upper = log_odds - fixed.amin(-1)
    for _ in range(64):
        middle = lower + 0.5 * (upper - lower)
        mean = (weights * torch.sigmoid(middle.unsqueeze(-1) + fixed)).sum(-1)
        lower = torch.where(mean < prevalence, middle, lower)
        upper = torch.where(mean >= prevalence, middle, upper)
    return lower + 0.5 * (upper - lower)


def _firth_projected_kkt(scale, minimum, gs, go, hss, hoo, *, tolerance,
                         convergence="absolute"):
    """Projected score, optionally relative to the weighted Fisher scale.

    Centered unit-spread coordinates make both diagonal information terms
    comparable. Dividing the scale score by sqrt(I_ss I_oo) and the intercept
    score by I_oo cancels the O(prevalence) information scale. Thus the opt-in
    test retains its tolerance for rare targets instead of accepting a small
    absolute score merely because the target population share is small.
    """
    boundary = scale <= minimum + max(float(tolerance), 1e-12)
    scale_score = torch.where(boundary, (-gs).clamp_min(0.0), gs.abs())
    if convergence == "fisher_standardized":
        scale_score = scale_score / torch.sqrt((hss * hoo).clamp_min(1e-300))
        offset_score = go.abs() / hoo.clamp_min(1e-300)
        return torch.maximum(scale_score, offset_score)
    return torch.maximum(scale_score, go.abs())


def _guarded_firth_newton(x, y, w, scale, offset, minimum, live,
                          inverse_sample_size, identifiable, fixed_offsets, *,
                          iterations, tolerance, hessian_ridge, objective_tolerance,
                          convergence="absolute"):
    """Fisher scoring with the existing projected Armijo ladder and KKT test.

    CUDA uses per-cell kernels that skip frozen cells and unused backtracks;
    no row vectors or fit decisions are read to host. CPU stops its masked
    iterations and ladder as soon as all cells finish.
    """
    if x.device.type == "cuda":
        from .device_newton_step import guarded_firth_newton
        return guarded_firth_newton(x, y, w, fixed_offsets, inverse_sample_size,
            identifiable, scale, offset, minimum, live, iterations=iterations,
            tolerance=tolerance, hessian_ridge=hessian_ridge,
            objective_tolerance=objective_tolerance, convergence=convergence)
    converged, failed = torch.zeros_like(live), torch.zeros_like(live)
    counts = torch.zeros_like(scale, dtype=torch.int64)
    steps = torch.zeros_like(counts)
    ridge = float(hessian_ridge)
    for iteration in range(1, max(1, int(iterations)) + 1):
        objective, gs, go, hss, hso, hoo = _firth_statistics(
            x, y, w, scale, offset, inverse_sample_size, identifiable, fixed_offsets)
        boundary = scale <= minimum + max(float(tolerance), 1e-12)
        kkt = _firth_projected_kkt(scale, minimum, gs, go, hss, hoo,
                                  tolerance=tolerance, convergence=convergence)
        converged = converged | (kkt <= float(tolerance))
        active = live & ~(converged | failed)
        if x.device.type == "cpu" and not bool(active.any().item()):
            break
        counts = torch.where(active, torch.full_like(counts, iteration), counts)
        hss, hoo = hss + ridge, hoo + ridge
        det = (hss * hoo - hso.square()).clamp_min(max(ridge * ridge, 1e-30))
        ds = (hoo * gs - hso * go) / det
        do = (hss * go - hso * gs) / det
        bound = (boundary & (gs >= 0.0)) | ~identifiable
        ds = torch.where(bound, torch.zeros_like(ds), ds)
        do = torch.where(bound, go / hoo.clamp_min(max(ridge, 1e-30)), do)
        accepted = torch.zeros_like(active)
        chosen_s, chosen_o = scale, offset
        for backtrack in range(LINE_SEARCH_LIMIT):
            alpha = 0.5 ** backtrack
            proposal_s = torch.maximum(scale - alpha * ds, minimum)
            proposal_o = offset - alpha * do
            proposal = _firth_statistics(
                x, y, w, proposal_s, proposal_o, inverse_sample_size,
                identifiable, fixed_offsets, score=False)
            directional = gs * (proposal_s - scale) + go * (proposal_o - offset)
            sufficient = (proposal <= objective + 1e-4 * directional
                          + float(objective_tolerance)) & torch.isfinite(proposal)
            take = active & ~accepted & sufficient
            chosen_s = torch.where(take, proposal_s, chosen_s)
            chosen_o = torch.where(take, proposal_o, chosen_o)
            steps = torch.where(take, torch.full_like(steps, backtrack + 1), steps)
            accepted = accepted | take
            if x.device.type == "cpu" and bool((accepted | ~active).all().item()):
                break
        scale, offset = chosen_s, chosen_o
        failed = failed | (active & ~accepted)
    return scale, offset, converged, failed, counts, steps


def _fit_firth_cells(x, y, validity, *, sample_weights, fixed_offsets,
                     initial_scale, initial_offset, enabled, minimum_scale,
                     iterations, tolerance, hessian_ridge, objective_tolerance,
                     reference_tolerance, convergence="absolute"):
    """One finite, metric-weighted Firth fit per enabled cell."""
    weights = validity.to(torch.float64)
    if sample_weights is None:
        raise ValueError("Firth calibration requires metric sample weights or class IDs")
    supplied = torch.as_tensor(sample_weights, dtype=torch.float64, device=x.device)
    weights = weights * supplied
    mass = weights.sum(-1)
    weights = weights / mass.clamp_min(1e-300).unsqueeze(-1)
    inverse_size = weights.square().sum(-1)
    weighted_positive = (weights * y).sum(-1)
    weighted_negative = (weights * (1.0 - y)).sum(-1)
    inputs_valid = ((mass > 0.0) & (weighted_positive > 0.0) & (weighted_negative > 0.0)
                    & torch.isfinite(weights).all(-1) & (weights >= 0.0).all(-1)
                    & torch.isfinite(initial_scale) & torch.isfinite(initial_offset))
    inputs_valid = inputs_valid & torch.isfinite(x).all(-1) & ((y == 0.0) | (y == 1.0)).all(-1)
    if fixed_offsets is not None:
        fixed_offsets = torch.as_tensor(fixed_offsets, dtype=torch.float64, device=x.device)
        inputs_valid = inputs_valid & torch.isfinite(fixed_offsets).all(-1)
    location = (x * weights).sum(-1)
    centered = x - location.unsqueeze(-1)
    raw_spread = torch.sqrt((centered.square() * weights).sum(-1))
    identifiable = raw_spread > max(float(tolerance), float(reference_tolerance))
    spread = torch.where(identifiable, raw_spread, torch.ones_like(raw_spread))
    u = centered / spread.unsqueeze(-1)
    minimum = spread * float(minimum_scale)
    a0 = initial_scale.clamp_min(float(minimum_scale))
    beta0, gamma0 = a0 * spread, initial_offset + a0 * location
    initial = _firth_statistics(u, y, weights, beta0, gamma0, inverse_size,
                                identifiable, fixed_offsets, score=False)
    # Extremely large warm starts may have zero floating-point information.
    # Restart only such cells at a finite intercept, without capping the fit.
    null = torch.log(weighted_positive.clamp_min(1e-300) / weighted_negative.clamp_min(1e-300))
    bad_start = ~torch.isfinite(initial)
    beta0 = torch.where(bad_start, minimum, beta0)
    gamma0 = torch.where(bad_start, null, gamma0)
    initial = _firth_statistics(u, y, weights, beta0, gamma0, inverse_size,
                                identifiable, fixed_offsets, score=False)
    identity = _firth_statistics(u, y, weights, spread, location, inverse_size,
                                 identifiable, fixed_offsets, score=False)
    intercept = _firth_statistics(u, y, weights, torch.zeros_like(beta0), null,
                                  inverse_size, identifiable, fixed_offsets, score=False)
    live = enabled & inputs_valid
    beta, gamma, converged, failed, counts, steps = _guarded_firth_newton(
        u, y, weights, beta0, gamma0, minimum, live, inverse_size, identifiable,
        fixed_offsets, iterations=iterations, tolerance=tolerance,
        hessian_ridge=hessian_ridge, objective_tolerance=objective_tolerance,
        convergence=convergence)
    final, gs, go, hss, _, hoo = _firth_statistics(
        u, y, weights, beta, gamma, inverse_size, identifiable, fixed_offsets)
    kkt = _firth_projected_kkt(beta, minimum, gs, go, hss, hoo,
                              tolerance=tolerance, convergence=convergence)
    finite = torch.isfinite(beta) & torch.isfinite(gamma) & torch.isfinite(final) & torch.isfinite(kkt)
    worsened = final > initial + float(objective_tolerance)
    converged = finite & ~worsened & ~failed & (kkt <= float(tolerance)) & (beta >= minimum)
    pending = live & ~converged & ~failed
    separated = torch.zeros_like(live)  # Firth has no unanchored-MLE separation gate.
    raw_scale = beta / spread
    raw_offset = gamma - raw_scale * location
    diagnostics = torch.stack(tuple(v.to(torch.float64) for v in (
        initial, final, identity, intercept, counts, steps, kkt, finite, converged,
        identifiable, separated, worsened, failed)))
    return CellFit(raw_scale, raw_offset, ~live | converged, pending, inputs_valid | ~enabled,
                   diagnostics, 1.0 / inverse_size.clamp_min(1e-300))


def fit_cells(x, y, validity, strength, *, initial_scale, initial_offset, enabled,
              weighting, minimum_scale, iterations, tolerance, hessian_ridge,
              objective_tolerance, reference_tolerance, probability_epsilon,
              source_epsilon, normalization=1.0, calibration_penalty="identity_anchor",
              sample_weights=None, fixed_offsets=None, firth_convergence="absolute"):
    """One ``fit_batched_post_query_affine`` call under the v6 gates.

    ``x`` [n] float64 raw logits, ``y`` [n] binary, ``validity`` [cells, n]
    binary row membership, ``strength`` [cells] anchor strengths,
    ``initial_*`` [cells] raw warm starts, ``enabled`` [cells] cells to solve.
    """
    if calibration_penalty not in {"identity_anchor", "firth"}:
        raise ValueError(f"unsupported calibration penalty {calibration_penalty!r}")
    if firth_convergence not in {"absolute", "fisher_standardized"}:
        raise ValueError(f"unsupported Firth convergence test {firth_convergence!r}")
    if calibration_penalty == "firth":
        return _fit_firth_cells(x, y, validity, sample_weights=sample_weights,
            fixed_offsets=fixed_offsets, initial_scale=initial_scale,
            initial_offset=initial_offset, enabled=enabled, minimum_scale=minimum_scale,
            iterations=iterations, tolerance=tolerance, hessian_ridge=hessian_ridge,
            objective_tolerance=objective_tolerance, reference_tolerance=reference_tolerance,
            convergence=firth_convergence)
    n = float(x.shape[-1])
    positives = (validity * y).sum(-1)
    negatives = validity.sum(-1) - positives
    weights = validity
    if weighting == "pos_weighted":
        weights = torch.where(y > 0.5, validity * (negatives / positives.clamp_min(1e-300)).unsqueeze(-1), validity)
    elif weighting == "design_weighted":
        if sample_weights is None:
            raise ValueError("design-weighted calibration requires sample weights")
        supplied = torch.as_tensor(sample_weights, dtype=torch.float64, device=x.device)
        weights = validity * supplied
        # Use a weighted mean within every selection/deployment cell.
        weights = weights * n / weights.sum(-1, keepdim=True).clamp_min(1e-300)
    elif weighting != "unweighted":
        raise ValueError(f"unsupported calibration weighting {weighting!r}")
    # Statistics normalize by the physical row count. Padded rows carry no
    # mass; retain the original logical normalization, ridge and KKT scale.
    weights = weights * (1.0 if weighting == "design_weighted" else normalization)
    probability = torch.sigmoid(x).clamp(min=probability_epsilon, max=1.0 - probability_epsilon)
    anchor = strength.unsqueeze(-1)
    target = (y + anchor * probability) / (1.0 + anchor)
    anchored = weights * (1.0 + anchor)
    constant = (anchor * weights * (F.softplus(x) - probability * x)).sum(-1) / n
    weighted_positive = (weights * y).sum(-1)
    weighted_negative = (weights * (1.0 - y)).sum(-1)
    inputs_valid = ((positives > 0.0) & (negatives > 0.0) & (weighted_positive > 0.0)
                    & (weighted_negative > 0.0) & torch.isfinite(strength) & (strength >= 0.0)
                    & torch.isfinite(initial_scale) & torch.isfinite(initial_offset))
    inputs_valid = inputs_valid & torch.isfinite(x).all() & ((y == 0.0) | (y == 1.0)).all()
    if weighting == "design_weighted":
        inputs_valid = inputs_valid & torch.isfinite(weights).all(-1) & (weights >= 0.0).all(-1)
    mass = weights.sum(-1).clamp_min(1e-30)
    location = (x * weights).sum(-1) / mass
    centered = x - location.unsqueeze(-1)
    active_rows = weights > 0.0
    positive_min = torch.where((y > 0.5) & active_rows, x, torch.full_like(centered, math.inf)).amin(-1)
    negative_max = torch.where((y <= 0.5) & active_rows, x, torch.full_like(centered, -math.inf)).amax(-1)
    raw_spread = torch.sqrt((centered.square() * weights).sum(-1) / mass)
    identifiable = raw_spread > max(float(tolerance), float(reference_tolerance))
    separated = (negative_max <= positive_min) & identifiable & (strength <= 0.0)
    spread = torch.where(identifiable, raw_spread, torch.ones_like(raw_spread))
    u = centered / spread.unsqueeze(-1)
    minimum = spread * float(minimum_scale)
    a0 = initial_scale.clamp_min(float(minimum_scale))
    beta0, gamma0 = a0 * spread, initial_offset + a0 * location
    initial = _objective(u, target, anchored, beta0, gamma0) - constant
    identity = _objective(u, target, anchored, spread, location) - constant
    null = torch.log(weighted_positive.clamp_min(1e-30) / weighted_negative.clamp_min(1e-30))
    intercept = _objective(u, target, anchored, torch.zeros_like(beta0), null) - constant
    live = enabled & inputs_valid
    beta, gamma, converged, failed, counts, steps, pending = guarded_newton(
        u, target, anchored, constant, beta0, gamma0, minimum, live,
        iterations=iterations, tolerance=tolerance, hessian_ridge=hessian_ridge,
        objective_tolerance=objective_tolerance)
    # _newton_final_gates and the v6 gates of _finish_with_device_hard_gates_v6.
    final, gs, go, *_ = _reduce(u, target, anchored, beta, gamma)
    final = final - constant
    at_boundary = beta <= minimum + max(float(tolerance), 1e-12)
    kkt = torch.maximum(torch.where(at_boundary, torch.clamp(-gs, min=0.0), gs.abs()), go.abs())
    converged = converged | (kkt <= float(tolerance))
    finite = torch.isfinite(beta) & torch.isfinite(gamma) & torch.isfinite(final) & torch.isfinite(kkt)
    worsened = final > initial + float(objective_tolerance)
    converged = converged & finite & ~worsened & ~failed & ~separated
    converged = converged & (kkt <= float(tolerance)) & (beta >= minimum)
    certified = ~live | (finite & converged)
    raw_scale = beta / spread
    raw_offset = gamma - raw_scale * location
    diagnostics = torch.stack(tuple(value.to(torch.float64) for value in (
        initial, final, identity, intercept, counts, steps, kkt, finite, converged,
        identifiable, separated, worsened, failed)))
    return CellFit(raw_scale, raw_offset, certified, pending & live, inputs_valid | ~enabled, diagnostics)


def _balanced_loss(logits, y, validity, sample_weights=None):
    """``_class_balanced_binary_loss`` with row validity, ``[cells, n]``."""
    positives = (validity * y).sum(-1)
    negatives = validity.sum(-1) - positives
    weights = torch.where(y > 0.5, (negatives / positives.clamp_min(1e-300)).unsqueeze(-1),
                          torch.ones_like(logits)) * validity
    if sample_weights is not None:
        weights = validity * sample_weights
        weights = weights * validity.sum(-1, keepdim=True) / weights.sum(-1, keepdim=True).clamp_min(1e-300)
    rows = F.softplus(logits) - y * logits
    return (rows * weights).sum(-1) / validity.sum(-1).clamp_min(1e-30), (positives > 0.0) & (negatives > 0.0)


def fit_family(x, y, block_ids, strength_values, *, block_count, initial_scale, initial_offset,
               weighting, minimum_scale, iterations, tolerance, hessian_ridge,
               objective_tolerance, reference_tolerance, probability_epsilon, source_epsilon,
               calibration_penalty="identity_anchor", sample_weights=None, fixed_offsets=None,
               firth_convergence="absolute"):
    """Selection grid, one-SE rule and deployment refit for one column.

    ``x`` [n] raw logits (any float dtype), ``y`` [n] binary, ``block_ids`` [n]
    inner selection blocks, ``strength_values`` [s] float64 anchor strengths,
    ``initial_*`` 0-d or [1] tensors (or floats).  Returns tensors only;
    ``pack`` concatenates them.
    """
    x = x.detach().to(torch.float64)
    y = y.detach().to(torch.float64)
    b, s = int(block_count), int(strength_values.shape[0])
    if calibration_penalty not in {"identity_anchor", "firth"}:
        raise ValueError(f"unsupported calibration penalty {calibration_penalty!r}")
    if calibration_penalty == "firth":
        one = torch.ones(1, dtype=torch.float64, device=x.device)
        deployment = fit_cells(x, y, (block_ids >= 0).to(torch.float64).unsqueeze(0),
            torch.zeros_like(one), initial_scale=one * initial_scale,
            initial_offset=one * initial_offset,
            enabled=torch.ones(1, dtype=torch.bool, device=x.device),
            weighting="design_weighted", minimum_scale=minimum_scale, iterations=iterations,
            tolerance=tolerance, hessian_ridge=hessian_ridge,
            objective_tolerance=objective_tolerance, reference_tolerance=reference_tolerance,
            probability_epsilon=probability_epsilon, source_epsilon=source_epsilon,
            calibration_penalty="firth", sample_weights=sample_weights, fixed_offsets=fixed_offsets,
            firth_convergence=firth_convergence)
        # Keep the transport packet's static dimensions; these zero padding
        # lanes are never fits or candidates and the Firth receipt omits them.
        zeros = torch.zeros(s + 1, dtype=torch.float64, device=x.device)
        zero_index = torch.zeros((), dtype=torch.int64, device=x.device)
        status = torch.stack((deployment.inputs_valid.all(),
                              torch.ones((), dtype=torch.bool, device=x.device),
                              ~deployment.pending.any(), deployment.certified.all()))
        return dict(scale=deployment.scale, offset=deployment.offset,
            selected=zero_index, best=zero_index, means=zeros, errors=zeros,
            losses=torch.zeros((s + 1, b), dtype=torch.float64, device=x.device),
            eligible=torch.zeros(s + 1, dtype=torch.bool, device=x.device),
            selection=torch.zeros((len(CELL_FIELDS), b * s), dtype=torch.float64, device=x.device),
            deployment=deployment.diagnostics, status=status,
            effective_sample_size=deployment.effective_sample_size)
    held = block_ids.unsqueeze(0) == torch.arange(b, device=x.device).unsqueeze(1)    # [b, n]
    training = ((~held) & (block_ids >= 0).unsqueeze(0)).to(torch.float64)
    testing = held.to(torch.float64)
    grid_valid = training.unsqueeze(1).expand(b, s, -1).reshape(b * s, -1)       # lane = block*s + strength
    grid_strength = strength_values.repeat(b)
    ones = torch.ones(b * s, dtype=torch.float64, device=x.device)
    options = dict(weighting=weighting, minimum_scale=minimum_scale, iterations=iterations,
                   tolerance=tolerance, hessian_ridge=hessian_ridge,
                   objective_tolerance=objective_tolerance, reference_tolerance=reference_tolerance,
                   probability_epsilon=probability_epsilon, source_epsilon=source_epsilon,
                   normalization=float(x.shape[-1]) / (block_ids >= 0).sum().to(torch.float64).clamp_min(1.),
                   **({"sample_weights": sample_weights} if weighting == "design_weighted" else {}))
    selection = fit_cells(x, y, grid_valid, grid_strength, initial_scale=ones * initial_scale,
                          initial_offset=ones * initial_offset,
                          enabled=torch.ones(b * s, dtype=torch.bool, device=x.device), **options)
    eta = x * selection.scale.unsqueeze(-1) + selection.offset.unsqueeze(-1)
    held_valid = testing.unsqueeze(1).expand(b, s, -1).reshape(b * s, -1)
    finite_losses, finite_support = _balanced_loss(eta, y, held_valid, sample_weights if weighting == "design_weighted" else None)
    identity_losses, identity_support = _balanced_loss(x.expand(b, -1), y, testing, sample_weights if weighting == "design_weighted" else None)
    losses = torch.cat((finite_losses.reshape(b, s).T, identity_losses.unsqueeze(0)), 0)  # [s+1, b]
    means = losses.mean(-1)
    errors = losses.std(-1, unbiased=True) / math.sqrt(float(b))
    best = means.argmin()
    # Gathers, not 0-d tensor subscripts: those read the index to the host.
    threshold = means.gather(0, best.reshape(1)) + errors.gather(0, best.reshape(1))
    eligible = means <= threshold
    order = torch.arange(s + 1, device=x.device)
    selected = torch.where(eligible, order, torch.full_like(order, -1)).amax()
    identity = selected == s
    # A non-finite selection row (inputs_valid false) still indexes in range.
    selected_strength = strength_values.index_select(0, selected.clamp(min=0, max=s - 1).reshape(1))
    full = (block_ids >= 0).to(torch.float64).unsqueeze(0)
    one = torch.ones(1, dtype=torch.float64, device=x.device)
    deployment = fit_cells(x, y, full, selected_strength, initial_scale=one * initial_scale,
                           initial_offset=one * initial_offset, enabled=~identity.reshape(1), **options)
    scale = torch.where(identity, torch.ones_like(deployment.scale), deployment.scale)
    offset = torch.where(identity, torch.zeros_like(deployment.offset), deployment.offset)
    inputs_valid = (selection.inputs_valid.all() & deployment.inputs_valid.all()
                    & finite_support.all() & identity_support.all() & torch.isfinite(losses).all())
    status = torch.stack((inputs_valid, ~selection.pending.any(), ~deployment.pending.any(),
                          selection.certified.all() & deployment.certified.all()))
    return dict(scale=scale, offset=offset, selected=selected, best=best, means=means,
                errors=errors, losses=losses, eligible=eligible, selection=selection.diagnostics,
                deployment=deployment.diagnostics, status=status)


def _symmetric_2x2(matrix, operation):
    """Analytic spectral functions of a symmetric 2x2 (no eigensolver sync).

    The inverse accepts a well-resolved positive spectrum (or a diagonal one
    above the 2*eps rank cutoff of the host's ``pinv``); otherwise ``valid``
    is false. Firth uncertainty can then use the recorded recovery below.
    """
    a, b, d = matrix[0, 0], matrix[0, 1], matrix[1, 1]
    half_difference = (a - d) * 0.5
    radius = torch.hypot(half_difference, b)
    middle = (a + d) * 0.5
    high = middle + radius
    low = torch.where(high > 0.0, (a * d - b.square()) / high.clamp_min(1e-300), middle - radius)
    valid = torch.isfinite(matrix).all() & torch.isfinite(low) & torch.isfinite(high)
    eps = torch.finfo(matrix.dtype).eps
    if operation == "inverse":
        diagonal = b == 0.0
        resolved = diagonal & (torch.minimum(a, d) > 2.0 * eps * torch.maximum(a.abs(), d.abs()))
        valid = valid & ((low > math.sqrt(eps) * high.abs()) | resolved)
        determinant = a * d - b.square()
        inverse = torch.stack((torch.stack((d, -b)), torch.stack((-b, a)))) / determinant
        diagonal_inverse = torch.diag(torch.stack((a.reciprocal(), d.reciprocal())))
        return torch.where(diagonal, diagonal_inverse, inverse), valid
    denominator = radius.clamp_min(1e-300)
    p00 = 0.5 * (1.0 + half_difference / denominator)
    p01 = 0.5 * b / denominator
    projector = torch.stack((torch.stack((p00, p01)), torch.stack((p01, 1.0 - p00))))
    eye = torch.eye(2, device=matrix.device, dtype=matrix.dtype)
    if operation == "positive":
        f_low, f_high = low.clamp_min(0.0), high.clamp_min(0.0)
    else:  # "sqrt_positive"
        f_low, f_high = low.clamp_min(0.0).sqrt(), high.clamp_min(0.0).sqrt()
    return f_low * (eye - projector) + f_high * projector, valid


def _normalized_symmetric_2x2(matrix):
    """Finite-range eigenvalues and high-eigenvalue projector, on device."""
    magnitude = matrix.abs().amax()
    magnitude = torch.where(magnitude > 0.0, magnitude, torch.ones_like(magnitude))
    normalized = matrix / magnitude
    a, b, d = normalized[0, 0], normalized[0, 1], normalized[1, 1]
    half_difference = (a - d) * 0.5
    radius = torch.hypot(half_difference, b)
    middle = (a + d) * 0.5
    high = middle + radius
    low = torch.where(high > 0.0, (a * d - b.square()) / high.clamp_min(1e-300),
                      middle - radius)
    denominator = radius.clamp_min(1e-300)
    p00 = 0.5 * (1.0 + half_difference / denominator)
    p01 = 0.5 * b / denominator
    projector = torch.stack((torch.stack((p00, p01)), torch.stack((p01, 1.0 - p00))))
    valid = torch.isfinite(matrix).all() & torch.isfinite(low) & torch.isfinite(high)
    return magnitude, low, high, projector, valid


def _symmetric_2x2_fallback(matrix, operation):
    """Scale-normalized spectra for a failed Firth uncertainty calculation.

    The inverse uses the host pseudoinverse's 2*eps relative rank cutoff,
    retaining positive eigenvalues only. A materially negative Hessian or a
    nonfinite source/reconstruction remains invalid. Covariance eigenvalues
    are projected onto the positive semidefinite cone, as on the host.
    Tensor operations only: this also runs inside a captured device fit.
    """
    magnitude, low, high, projector, valid = _normalized_symmetric_2x2(matrix)
    eye = torch.eye(2, device=matrix.device, dtype=matrix.dtype)
    if operation == "inverse":
        cutoff = 2.0 * torch.finfo(matrix.dtype).eps * torch.maximum(low.abs(), high.abs())
        valid = valid & (high > 0.0) & (low >= -cutoff)
        inverse_low = torch.where(low > cutoff, low, torch.ones_like(low)).reciprocal()
        inverse_low = torch.where(low > cutoff, inverse_low, torch.zeros_like(low))
        inverse_high = torch.where(high > cutoff, high, torch.ones_like(high)).reciprocal()
        inverse_high = torch.where(high > cutoff, inverse_high, torch.zeros_like(high))
        result = (inverse_low * (eye - projector) + inverse_high * projector) / magnitude
    else:
        f_low, f_high = low.clamp_min(0.0), high.clamp_min(0.0)
        multiplier = magnitude
        if operation == "sqrt_positive":
            f_low, f_high = f_low.sqrt(), f_high.sqrt()
            multiplier = magnitude.sqrt()
        result = (f_low * (eye - projector) + f_high * projector) * multiplier
    return result, valid & torch.isfinite(result).all()


def _resolved_symmetric_2x2(matrix, operation, allow_fallback):
    """Retain every valid analytic result, including its floating-point bits."""
    result, valid = _symmetric_2x2(matrix, operation)
    if not allow_fallback:
        return result, valid, torch.zeros_like(valid)
    # Do not send an accepted matrix through potentially overflowing fallback
    # arithmetic. Selection uses the original result without recomputation.
    eye = torch.eye(2, device=matrix.device, dtype=matrix.dtype)
    replacement, recovered = _symmetric_2x2_fallback(torch.where(valid, eye, matrix), operation)
    used = ~valid & recovered
    return torch.where(used, replacement, result), valid | used, used


def _recovered_influence_covariance(x, score, row_curvature, membership, sample_counts,
                                    risk_hessian, inverse_hessian, ridge):
    """Sandwich factors without cancellation through a nearly singular inverse.

    For full numerical rank, a centered Schur solve is algebraically H^-1
    applied to each score row, with the original isotropic ridge unchanged.
    If the spectral cutoff drops a direction, use the spectral pseudoinverse.
    Center the resulting influences within strata before summing squares.
    """
    mass = row_curvature.sum()
    safe_mass = torch.where(mass > 0.0, mass, torch.ones_like(mass))
    anchor = x[0]
    mean = anchor + (row_curvature * (x - anchor)).sum() / safe_mass
    centered_x = x - mean
    variance = (row_curvature * centered_x.square()).sum()
    intercept_curvature = mass + ridge
    schur = variance + ridge + mass / intercept_curvature * ridge * mean.square()
    u = score[:, 1]
    slope = u * (centered_x + mean * (ridge / intercept_curvature)) / schur
    intercept = u * ((variance + ridge) - mass * mean * centered_x) / (intercept_curvature * schur)
    solved = torch.stack((slope, intercept), dim=1)
    _, low, high, _, _ = _normalized_symmetric_2x2(risk_hessian)
    cutoff = 2.0 * torch.finfo(x.dtype).eps * torch.maximum(low.abs(), high.abs())
    influence = torch.where(low > cutoff, solved, score @ inverse_hessian)
    stratum_means = (membership @ influence) / sample_counts.unsqueeze(1)
    centered = influence.unsqueeze(0) - stratum_means.unsqueeze(1)
    return torch.einsum("hni,hnj->hij", membership.unsqueeze(2) * centered, centered) / (
        sample_counts - 1.0).reshape(-1, 1, 1)


def affine_uncertainty(values, labels, strata, weights, populations, scale, offset, *,
                       confidence_level, confidence_parameters=None, hessian_ridge=1e-10,
                       finite_population_correction=True, sample_weights=None, fixed_offsets=None,
                       allow_spectral_fallback=False):
    """``stratified_affine_bce_uncertainty`` as tensors, for the device batch.

    The same formulas; the 2x2 inverse, the covariance projection and square
    root and the largest curvature eigenvalue are analytic.  Returns the host
    function's quantities in its packet order plus validity and recovery flags.
    Recovery is opt-in and leaves every accepted analytic result untouched.
    """
    x = values.to(torch.float64).reshape(-1)
    y = labels.to(torch.float64).reshape(-1)
    h = strata.to(torch.int64).reshape(-1)
    weights = weights.to(torch.float64).reshape(-1)
    populations = populations.to(torch.float64).reshape(-1)
    count = int(weights.numel())
    scale = torch.as_tensor(scale, dtype=torch.float64, device=x.device).reshape(())
    offset = torch.as_tensor(offset, dtype=torch.float64, device=x.device).reshape(())
    design = torch.stack((x, torch.ones_like(x)), dim=1)
    eta = scale * x + offset
    if fixed_offsets is not None:
        eta = eta + fixed_offsets.to(device=x.device, dtype=torch.float64).reshape(-1)
    probability = torch.sigmoid(eta)
    residual = probability - y
    curvature = probability * (1.0 - probability)
    score = residual.unsqueeze(1) * design
    membership = (h.unsqueeze(0) == torch.arange(count, device=x.device).unsqueeze(1)).to(torch.float64)
    sample_counts = membership.sum(dim=1)
    metric_weights_valid = torch.ones((), dtype=torch.bool, device=x.device)
    if sample_weights is not None:
        supplied = torch.as_tensor(sample_weights, device=x.device, dtype=torch.float64).reshape(-1)
        active = (h >= 0) & (h < count)
        normalized = torch.where(active, supplied, torch.zeros_like(supplied))
        mass = normalized.sum()
        metric_weights_valid = torch.isfinite(supplied).all() & (supplied >= 0.0).all() & (mass > 0.0)
        normalized = normalized / mass.clamp_min(1e-300)
        safe_h = h.clamp(min=0, max=count - 1)
        # The design keeps binary Neyman strata, while each score/Hessian
        # row estimates the class-balanced metric under those strata.
        factor = normalized * sample_counts.gather(0, safe_h) / weights.gather(0, safe_h).clamp_min(1e-300)
        factor = torch.where(active, factor, torch.zeros_like(factor))
        score = score * factor.unsqueeze(1)
        curvature = curvature * factor
    stratum_score_means = (membership @ score) / sample_counts.unsqueeze(1)
    risk_gradient = weights @ stratum_score_means
    centered_score = score.unsqueeze(0) - stratum_score_means.unsqueeze(1)
    score_covariance = torch.einsum("hni,hnj->hij", membership.unsqueeze(2) * centered_score,
                                    centered_score) / (sample_counts - 1.0).reshape(-1, 1, 1)
    stratum_hessian = torch.einsum("hni,nj->hij", (membership * curvature).unsqueeze(2) * design,
                                   design) / sample_counts.reshape(-1, 1, 1)
    risk_hessian = torch.einsum("h,hij->ij", weights, stratum_hessian) + float(hessian_ridge) * torch.eye(
        2, device=x.device, dtype=torch.float64)
    inverse_hessian, valid, inverse_fallback = _resolved_symmetric_2x2(
        risk_hessian, "inverse", allow_spectral_fallback)
    fpc = (torch.clamp(1.0 - sample_counts / populations, min=0.0)
           if finite_population_correction else torch.ones_like(sample_counts))
    gradient_covariance = torch.einsum("h,hij->ij", weights.square() * fpc / sample_counts, score_covariance)
    covariance = inverse_hessian @ gradient_covariance @ inverse_hessian
    influence_covariance = torch.einsum("ij,hjk,lk->hil", inverse_hessian, score_covariance, inverse_hessian)
    if allow_spectral_fallback:
        row_metric = (weights / sample_counts) @ membership
        row_curvature = row_metric * curvature
        recovered_influence = _recovered_influence_covariance(
            x, score, row_curvature, membership, sample_counts, risk_hessian, inverse_hessian,
            float(hessian_ridge))
        recovered_covariance = torch.einsum(
            "h,hij->ij", weights.square() * fpc / sample_counts, recovered_influence)
        covariance = torch.where(inverse_fallback, recovered_covariance, covariance)
        influence_covariance = torch.where(inverse_fallback, recovered_influence, influence_covariance)
    covariance = 0.5 * (covariance + covariance.T)
    root, valid_root, root_fallback = _resolved_symmetric_2x2(
        covariance, "sqrt_positive", allow_spectral_fallback)
    covariance, valid_positive, positive_fallback = _resolved_symmetric_2x2(
        covariance, "positive", allow_spectral_fallback)
    covariance = 0.5 * (covariance + covariance.T)
    valid = valid & valid_root & valid_positive
    allocation_standard_deviations = torch.sqrt(torch.clamp(
        torch.diagonal(influence_covariance, dim1=1, dim2=2).sum(dim=1), min=0.0))
    if confidence_parameters is None:
        radius_squared = -2.0 * math.log(1.0 - float(confidence_level))
        radius = math.sqrt(radius_squared)
        normal_critical = NormalDist().inv_cdf(1.0 - (1.0 - float(confidence_level)) / 4.0)
    else:
        radius, radius_squared, normal_critical = confidence_parameters.unbind()
    first_order_coefficient = torch.sqrt(torch.clamp(risk_gradient @ covariance @ risk_gradient, min=0.0))
    curvature_matrix = root @ risk_hessian @ root
    if allow_spectral_fallback:
        # Factor forms avoid cancellation in quadratic forms of the recovered
        # covariance. No accepted analytic result uses these replacements.
        transformed_design = design @ root
        recovered_gradient = (row_metric.unsqueeze(1) * score[:, 1:2] * transformed_design).sum(0)
        first_order_coefficient = torch.where(
            inverse_fallback, torch.linalg.vector_norm(recovered_gradient), first_order_coefficient)
        recovered_curvature = (transformed_design.T @ (row_curvature.unsqueeze(1) * transformed_design)
                               + float(hessian_ridge) * (root @ root))
        curvature_matrix = torch.where(inverse_fallback, recovered_curvature, curvature_matrix)
    maximum_curvature = 0.5 * (curvature_matrix[0, 0] + curvature_matrix[1, 1]) + torch.hypot(
        0.5 * (curvature_matrix[0, 0] - curvature_matrix[1, 1]),
        0.5 * (curvature_matrix[0, 1] + curvature_matrix[1, 0]))
    second_order_coefficient = 0.5 * torch.clamp(maximum_curvature, min=0.0)
    first_order = radius * first_order_coefficient
    second_order = radius_squared * second_order_coefficient
    prediction_variance = ((design @ covariance) * design).sum(dim=1)
    if allow_spectral_fallback:
        prediction_variance = torch.where(
            inverse_fallback, transformed_design.square().sum(1), prediction_variance)
    radius = normal_critical * torch.sqrt(torch.clamp(
        torch.where(h >= 0, prediction_variance, torch.zeros_like(prediction_variance)).max(), min=0.0))
    excess = first_order + second_order
    valid = valid & (sample_counts >= 2.0).all() & metric_weights_valid
    if finite_population_correction:
        valid = valid & (sample_counts <= populations).all()
    spectral_fallback = inverse_fallback.to(torch.int64) + 2 * (root_fallback | positive_fallback).to(torch.int64)
    # Recovery must not turn overflowing final bounds into a valid decision.
    # Accepted legacy calculations retain their original validity predicate.
    recovered_finite = (torch.isfinite(covariance).all() & torch.isfinite(risk_gradient).all()
                        & torch.isfinite(allocation_standard_deviations).all()
                        & torch.isfinite(radius) & torch.isfinite(excess)
                        & torch.isfinite(first_order_coefficient) & torch.isfinite(second_order_coefficient))
    valid = valid & ((spectral_fallback == 0) | recovered_finite)
    return dict(parameter_covariance=covariance, allocation_standard_deviations=allocation_standard_deviations,
                maximum_prediction_radius=radius, maximum_bce_excess_risk_bound=excess,
                first_order_excess_risk_bound=first_order, second_order_excess_risk_bound=second_order,
                first_order_excess_risk_coefficient=first_order_coefficient,
                second_order_excess_risk_coefficient=second_order_coefficient,
                sample_counts=sample_counts, risk_gradient=risk_gradient, risk_hessian=risk_hessian,
                valid=valid, spectral_fallback=spectral_fallback)


UNCERTAINTY_SCALARS = ("maximum_prediction_radius", "maximum_bce_excess_risk_bound",
                       "first_order_excess_risk_bound", "second_order_excess_risk_bound",
                       "first_order_excess_risk_coefficient", "second_order_excess_risk_coefficient")


def pack(family, uncertainty, labels_match):
    """One float64 vector for the single device-to-host read."""
    uncertainty_status = uncertainty["valid"].to(torch.float64) * (1 + uncertainty.get("spectral_fallback", 0))
    status = torch.cat((family["status"], torch.stack((uncertainty_status, labels_match))))
    parts = (status, family["scale"], family["offset"], family["selected"], family["best"],
             family["means"], family["errors"], family["losses"], family["eligible"],
             family["selection"], family["deployment"],
             uncertainty["parameter_covariance"], uncertainty["allocation_standard_deviations"],
             torch.stack(tuple(uncertainty[name] for name in UNCERTAINTY_SCALARS)),
             uncertainty["sample_counts"])
    return torch.cat(tuple(part.reshape(-1).to(torch.float64) for part in parts))


def unpack(host, *, strength_count, block_count, stratum_count):
    """Split the completed host packet into named NumPy views."""
    result = packet_views(host, strength_count=strength_count,
                          block_count=block_count, stratum_count=stratum_count)
    fallback = int(result["status"][STATUS_FIELDS.index("uncertainty_valid")]) - 1
    if fallback > 0:
        result["uncertainty_fallback"] = {
            **({"inverse": "scale_normalized_spectral_pseudoinverse_2eps"} if fallback & 1 else {}),
            **({"sandwich": "centered_influence_rows"} if fallback & 1 else {}),
            **({"covariance": "scale_normalized_positive_spectrum"} if fallback & 2 else {}),
        }
    result["status"] = dict(zip(STATUS_FIELDS, (bool(v) for v in result["status"])))
    return result


def packet_views(packet, *, strength_count, block_count, stratum_count):
    """Static slices of a packet, retaining device tensors without a read."""
    s1, b, cells = strength_count + 1, block_count, block_count * strength_count
    shapes = (("status", (len(STATUS_FIELDS),)), ("scale", (1,)), ("offset", (1,)),
              ("selected", ()), ("best", ()), ("means", (s1,)), ("errors", (s1,)),
              ("losses", (s1, b)), ("eligible", (s1,)),
              ("selection", (len(CELL_FIELDS), cells)), ("deployment", (len(CELL_FIELDS), 1)),
              ("parameter_covariance", (2, 2)), ("allocation_standard_deviations", (stratum_count,)),
              ("scalars", (len(UNCERTAINTY_SCALARS),)), ("sample_counts", (stratum_count,)))
    result, position = {}, 0
    for name, shape in shapes:
        size = math.prod(shape)
        result[name] = packet[position:position + size].reshape(shape)
        position += size
    if position != packet.shape[0]:
        raise ValueError("device calibration packet has an unexpected length")
    return result


def calibration_packet(x, center, labels, design, block_ids, strata, strength_values,
                       initial_scale, initial_offset, weights, populations, *, block_count,
                       confidence_level, finite_population_correction, hessian_ridge,
                       confidence_parameters=None, **options):
    """The whole decision fit as one device packet (``pack`` layout).

    ``x`` [n] the target readout column in its source dtype, ``center`` [1]
    the readout bias the uncertainty is centred on, ``labels`` [n] delivered
    binary labels, ``design`` [n] the source design's binary classes (they
    choose the inner blocks on the host), ``strata`` [n] binary Neyman
    stratum codes.  The fit reads the raw column and the uncertainty the
    centred one with its own default ridge, exactly as ``_fit_one`` does.
    Firth uses raw coordinates and permits recorded spectral recovery only
    when the original analytic spectrum calculation fails.
    """
    family = fit_family(x, labels, block_ids, strength_values, block_count=block_count,
                        initial_scale=initial_scale, initial_offset=initial_offset,
                        hessian_ridge=hessian_ridge, **options)
    firth = options.get("calibration_penalty") == "firth"
    uncertainty = affine_uncertainty(x if firth else x - center.reshape(()), labels, strata, weights, populations,
                                     family["scale"], family["offset"],
                                     confidence_level=confidence_level,
                                     confidence_parameters=confidence_parameters,
                                     finite_population_correction=finite_population_correction,
                                     allow_spectral_fallback=firth,
                                     **({"fixed_offsets": options["fixed_offsets"]} if options.get("fixed_offsets") is not None else {}),
                                     **({"sample_weights": options.get("sample_weights")} if firth or options.get("weighting") == "design_weighted" else {}))
    labels_match = (labels.to(torch.float64) == design.to(torch.float64)).all()
    return pack(family, uncertainty, labels_match)


class _Graph:
    """One captured ``calibration_packet`` with static input buffers."""

    def __init__(self, inputs, static):
        initialize_cuda_graph_state(next(iter(inputs.values())).device)
        with torch.inference_mode(False):
            self.inputs = {name: value.clone() for name, value in inputs.items()}
        stream = torch.cuda.Stream(device=next(iter(inputs.values())).device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):  # warm the allocator and the kernels outside capture
                calibration_packet(**self.inputs, **static)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        self.pool = torch.cuda.graph_pool_handle()
        # Thread-local capture: a fit may run on the overlapping fit thread
        # while the main thread queues adjudication work.
        # Keep stream restoration outside capture finalization: if either
        # the packet or capture_end raises, the caller still gets its stream
        # back. An unsuccessful capture must never enter the replay cache.
        with torch.cuda.stream(stream):
            try:
                self.graph.capture_begin(pool=self.pool, capture_error_mode="thread_local")
                self.output = calibration_packet(**self.inputs, **static)
            except BaseException as error:
                try:
                    self._capture_end()
                except BaseException as cleanup_error:
                    # Preserve the packet failure when CUDA also reports an
                    # invalidated capture while ending it.
                    raise error from cleanup_error
                raise
            else:
                self._capture_end()

    def _capture_end(self):
        try:
            self.graph.capture_end()
        except BaseException:
            # An invalidated CUDA capture can make PyTorch's capture_end
            # throw before it removes the allocator's active pool. End that
            # allocation scope as well, or empty_cache in between-arm reset
            # will still see captures_underway. The binding is versioned
            # with PyTorch's memory-pool API; older builds may lack it.
            end_pool = getattr(torch.cuda.memory, "_cuda_endAllocateToPool", None)
            if end_pool is None:
                end_pool = getattr(torch._C, "_cuda_endAllocateToPool", None)
            if end_pool is not None:
                try:
                    device = next(iter(self.inputs.values())).device
                    index = device.index if device.index is not None else torch.cuda.current_device()
                    end_pool(index, self.pool)
                except BaseException:
                    # Keep the original capture error even when this
                    # PyTorch build already ended its allocation scope.
                    pass
            try:
                self.graph.reset()
            except BaseException:
                pass
            raise

    def __call__(self, inputs):
        for name, value in inputs.items():
            self.inputs[name].copy_(value, non_blocking=True)
        self.graph.replay()
        return self.output.clone()


class CalibrationFitRunner:
    """Queues ``calibration_packet``: eager on CPU, one CUDA-graph replay on CUDA.

    A graph is captured once per input shape and static configuration (the
    decision fit's row count is fixed by the registered calibration packet,
    so a run holds a few graphs).  Replay runs the same kernels as eager
    execution. CUDA capture errors end the execution.
    """

    def __init__(self):
        self.graphs = {}
        self.counts = {"graph_replays": 0, "eager_calls": 0, "captures": 0}

    @staticmethod
    def _key(inputs, static):
        device = next(iter(inputs.values())).device
        return (str(device), tuple((name, tuple(value.shape), str(value.dtype))
                                   for name, value in inputs.items()),
                tuple(sorted(static.items())))

    def prewarm(self, inputs, static):
        """Create a worker lane once; an existing lane needs no warm replay."""
        if next(iter(inputs.values())).device.type != "cuda":
            raise ValueError("fit graph prewarming requires CUDA inputs")
        key = self._key(inputs, static)
        graph = self.graphs.get(key)
        if graph is None:
            graph = self.graphs[key] = _Graph(inputs, static)
            self.counts["captures"] += 1
        return graph

    def __call__(self, inputs, static):
        device = next(iter(inputs.values())).device
        if device.type != "cuda":
            self.counts["eager_calls"] += 1
            return calibration_packet(**inputs, **static)
        graph = self.prewarm(inputs, static)
        self.counts["graph_replays"] += 1
        return graph(inputs)


RUNNER = CalibrationFitRunner()


class DeviceFitReceipt:
    """The device fit's receipt: the proper-score receipt's fields, built from the packet."""

    passed = True

    def __init__(self, payload):
        self.payload = payload

    def to_dict(self):
        return dict(self.payload)


def receipt_payload(result, *, tokens, block_hash, block_count, problem_hashes,
                    initialization, device_type, iterations,
                    calibration_penalty="identity_anchor", effective_sample_size=None,
                    metric_loss_weighting=None):
    """The proper-score receipt layout (``_materialize_selection_rows``) from host values."""
    if calibration_penalty not in {"identity_anchor", "firth"}:
        raise ValueError(f"unsupported calibration penalty {calibration_penalty!r}")
    metric = loss_weighting() if metric_loss_weighting is None else metric_loss_weighting
    if metric not in {"balanced", "per_image"}:
        raise ValueError(f"unsupported metric loss weighting {metric!r}")
    if calibration_penalty == "firth":
        deployment = dict(zip(CELL_FIELDS, result["deployment"][:, 0].tolist()))
        size = effective_sample_size
        if size is None and "effective_sample_size" in result:
            size = float(result["effective_sample_size"].reshape(-1)[0])
        if size is None:
            raise ValueError("Firth device receipt requires Kish effective sample size")
        return {
            "schema_name": "mechanistic_proper_score_shrinkage_receipt", "schema_version": "1",
            **({"loss_weighting": "per_image"} if metric == "per_image" else {}),
            **problem_hashes, "algorithm_id": "firth_positive_affine_v1",
            "calibration_penalty": "firth", "effective_sample_size": [float(size)],
            "candidate_strengths": [], "selected_strengths": ["firth"],
            "component_rows": [{"mask_index": 0, "query_index": 0,
                                "selected_strength": "firth", "effective_sample_size": float(size)}],
            "block_assignment_hash": "", "block_count": 0, "selection_rule": "single_firth_fit",
            "fit_receipt": {"schema_name": "device_firth_positive_affine_fit_receipt",
                            "schema_version": "1", "passed": bool(deployment["converged"]),
                            "calibration_penalty": "firth", "effective_sample_size": [float(size)],
                            "iteration_bound": int(iterations), **deployment},
            "backend_profile": {"backend_id": "mechanistic_systems.device_calibration_fit.firth.v1",
                "device_type": str(device_type), "metadata": {
                    "solver_backend": "device_float64_firth_fisher_scoring",
                    "calibration_penalty": "firth", "effective_sample_size": [float(size)],
                    "regularization_geometry": "jeffreys_information_log_determinant",
                    "weighting": "per_image_population_mean" if metric == "per_image" else "class_balanced_metric",
                    **({"loss_weighting": "per_image"} if metric == "per_image" else {}), "initialization": initialization,
                    "constant_readout_rule": "intercept_only_firth_quotient"}},
            **({"uncertainty_fallback": dict(result["uncertainty_fallback"])} if "uncertainty_fallback" in result else {}),
            "passed": bool(deployment["converged"])}
    selected, best = int(result["selected"]), int(result["best"])
    losses = result["losses"].tolist()
    eligible = [tokens[i] for i, flag in enumerate(result["eligible"].tolist()) if flag]
    row = {"mask_index": 0, "query_index": 0, "selected_strength": tokens[selected],
           "best_strength": tokens[best], "mean_losses": result["means"].tolist(),
           "standard_errors": result["errors"].tolist(), "fold_losses": losses,
           "eligible_strengths": eligible}
    identity = selected == len(tokens) - 1
    deployment = dict(zip(CELL_FIELDS, result["deployment"][:, 0].tolist()))
    selection = {name: values for name, values in zip(CELL_FIELDS, result["selection"].tolist())}
    fit_receipt = ({"schema_name": "mechanistic_identity_calibration_receipt", "schema_version": "1",
                    "passed": True, "reason": "identity_selected_within_one_standard_error"}
                   if identity else
                   {"schema_name": "device_v63_positive_affine_fit_receipt", "schema_version": "1",
                    "passed": True,
                    "iteration_bound": int(iterations), **deployment})
    return {
        "schema_name": "mechanistic_proper_score_shrinkage_receipt", "schema_version": "1",
            **({"loss_weighting": "per_image"} if metric == "per_image" else {}),
        **problem_hashes,
        "algorithm_id": "proper_score_identity_shrinkage_affine_v2",
        "candidate_strengths": list(tokens), "selected_strengths": [tokens[selected]],
        "component_rows": [row], "block_assignment_hash": str(block_hash),
        "block_count": int(block_count),
        "selection_rule": "largest_identity_anchor_within_one_se_of_best",
        "fit_receipt": fit_receipt,
        # Diagnostics only: the backend profile is not part of the fit identity.
        "backend_profile": {"backend_id": "mechanistic_systems.device_calibration_fit.v63",
                            "device_type": str(device_type),
                            "metadata": {**({"loss_weighting": "per_image", "weighting": "per_image_population_mean"} if metric == "per_image" else {}),
                                         "solver_backend": "identity" if identity
                                         else "device_masked_projected_newton_v63",
                                         "initialization": initialization,
                                         "selection_cells": selection}},
        "passed": True,
    }


__all__ = ["CELL_FIELDS", "CHUNK_ITERATIONS", "RUNNER", "STATUS_FIELDS", "CalibrationFitRunner",
           "DeviceFitReceipt", "affine_uncertainty", "calibration_packet", "fit_cells",
           "fit_family", "guarded_newton", "pack", "receipt_payload", "unpack"]
