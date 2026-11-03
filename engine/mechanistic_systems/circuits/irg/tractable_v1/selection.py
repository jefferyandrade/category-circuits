"""Production-usable binary-selection primitives for tractable-v1.

This module intentionally stops below full controller policy.  It provides the
bounded arithmetic operations a policy needs without claiming global binary
optimality, population authority, bootstrap authority, or C/K/O admission.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence
import math

import torch

from .operators import ChartOperators
from .solvers import DiagonalShrunkGeometry

SelectionMode = Literal["logit_aligned", "loss_aligned_ablation", "diagnostic_lagrange"]


@dataclass(frozen=True)
class CandidateRowWorkspacePlan:
    candidate_count: int
    retained_bytes: int
    workspace_bytes: int
    max_retained_bytes: int
    temp_bytes_per_context_candidate_upper: int
    principal_temporaries: tuple[str, ...]
    law: str = "candidate_rows_explicit_workspace_budget_v2"


@dataclass(frozen=True)
class CandidateCellRows:
    candidates: torch.Tensor
    fiber_ids: torch.Tensor
    values: torch.Tensor
    workspace_plan: CandidateRowWorkspacePlan
    law: str = "tractable_v1_candidate_corrected_cell_rows_v3"


@dataclass(frozen=True)
class CandidateWorkMarginals:
    candidates: torch.Tensor
    marginal_work: torch.Tensor
    base_work: float
    candidate_rows: CandidateCellRows
    law: str = "tractable_v1_exact_finite_native_work_marginals_from_candidate_rows_v3"


@dataclass(frozen=True)
class CandidateMarginalBatch:
    candidates: torch.Tensor
    delta_q: torch.Tensor
    delta_f: torch.Tensor
    legal_order: torch.Tensor
    mode: str
    authority: str = "proposal_primitive_only_observed_chart"
    optimality_authority: bool = False
    population_certified: bool = False
    law: str = "tractable_v1_bounded_candidate_marginal_batch_v2"


def _positive_int(name: str, value: int) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def _flat_tensor(name: str, x: torch.Tensor, n: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    raw = torch.as_tensor(x)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional tensor")
    if raw.numel() != n:
        raise ValueError(f"{name} must have one entry per route")
    if not bool(torch.isfinite(raw).all()):
        raise ValueError(f"{name} must be finite")
    return raw.to(device=device, dtype=dtype)


def _binary_tensor(name: str, x: torch.Tensor, n: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    raw = torch.as_tensor(x)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional tensor")
    if raw.numel() != n:
        raise ValueError(f"{name} must have one entry per route")
    if not bool(torch.isfinite(raw).all()):
        raise ValueError(f"{name} must be finite")
    if not bool(((raw == 0) | (raw == 1)).all()):
        raise ValueError(f"{name} must be binary")
    return raw.to(device=device, dtype=dtype)


def _candidate_tensor(candidates: torch.Tensor | Sequence[int], n: int, *, device: torch.device) -> torch.Tensor:
    raw = torch.as_tensor(candidates)
    if raw.ndim != 1:
        raise ValueError("candidate routes must be a one-dimensional tensor")
    if raw.numel() == 0:
        return torch.empty(0, device=device, dtype=torch.long)
    if raw.is_floating_point() and not bool(torch.isfinite(raw).all()):
        raise ValueError("candidate routes must be finite")
    if raw.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("candidate routes must have integer dtype")
    cand_cpu = raw.to(dtype=torch.long)
    if bool((cand_cpu < 0).any()) or bool((cand_cpu >= n).any()):
        raise ValueError("candidate route out of bounds")
    if cand_cpu.unique().numel() != cand_cpu.numel():
        raise ValueError("candidate routes must be unique")
    return cand_cpu.to(device=device)


def validate_selection_inputs(ops: ChartOperators, z: torch.Tensor, candidate_eligible: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate binary action and candidate mask separately from physical liveness."""

    zz = _binary_tensor("z", z, ops.route_count, device=ops.device, dtype=ops.dtype)
    parent_live = _binary_tensor("ops.eligible()", ops.eligible(), ops.route_count, device=ops.device, dtype=ops.dtype)
    if bool(((zz > 0) & (parent_live <= 0)).any()):
        raise ValueError("z selects a route that is not live in the physical parent")
    physical_live_after_z = ((parent_live > 0) & (zz <= 0)).to(ops.dtype)
    if candidate_eligible is None:
        cand = physical_live_after_z
    else:
        cand = _binary_tensor("candidate_eligible", candidate_eligible, ops.route_count, device=ops.device, dtype=ops.dtype)
        if bool(((cand > 0) & (physical_live_after_z <= 0)).any()):
            raise ValueError("candidate_eligible must be a subset of currently live unselected routes")
    return zz, cand


def _require_live_candidates(name: str, mask: torch.Tensor, candidates: torch.Tensor) -> None:
    if candidates.numel() and bool((mask[candidates] <= 0).any()):
        raise ValueError(f"{name} includes dead or phase-ineligible candidates")


def q_add_marginal(ops: ChartOperators, geom: DiagonalShrunkGeometry, z: torch.Tensor,
                   candidates: torch.Tensor | Sequence[int], candidate_eligible: torch.Tensor | None = None) -> torch.Tensor:
    """Exact finite ``Q(z + e_j) - Q(z)`` for live unselected binary candidates."""

    zz, cand_mask = validate_selection_inputs(ops, z, candidate_eligible)
    cand = _candidate_tensor(candidates, ops.route_count, device=ops.device)
    if cand.numel() == 0:
        return torch.empty(0, device=ops.device, dtype=ops.dtype)
    if bool((zz[cand] != 0).any()):
        raise ValueError("q_add_marginal only supports adding currently unselected routes")
    _require_live_candidates("q_add_marginal", cand_mask, cand)
    return geom.self_term()[cand] + 2.0 * geom.cross_term(ops, zz)[cand]


def work_subgradient_diagnostic(ops: ChartOperators, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``A^T[omega sign(Az)]`` and zero-fiber mask; no singleton substitution."""

    zz, _ = validate_selection_inputs(ops, z)
    cells = ops.a_forward(zz)
    y = ops.work_weights() * torch.sign(cells)
    return ops.a_adjoint(y), (cells == 0).all(dim=0)


def singleton_finite_work_screen(ops: ChartOperators) -> torch.Tensor:
    """Exact empty-action singleton ``F(e_j)-F(0)``, including native fiber measure."""

    return ops.singleton_price_magnitude() * ops.fiber_measure()[ops.fiber_of_route()]


def forecast_candidate_row_workspace(ops: ChartOperators, candidates: torch.Tensor | Sequence[int], *,
                                     max_candidates: int, workspace_bytes: int,
                                     max_retained_bytes: int) -> CandidateRowWorkspacePlan:
    """Forecast retained output and principal per context-candidate temporary budget.

    Principal temporaries are gathered ``Gt`` [P], gathered ``X`` [P], and their
    product/reduction [P].  Scalar gathered ``W`` and ``mask`` are accounted in
    slack; no full layer ``W * mask`` is materialized.
    """

    max_candidates = _positive_int("max_candidates", max_candidates)
    workspace_bytes = _positive_int("workspace_bytes", workspace_bytes)
    max_retained_bytes = _positive_int("max_retained_bytes", max_retained_bytes)
    cand = _candidate_tensor(candidates, ops.route_count, device=ops.device)
    if cand.numel() > max_candidates:
        raise ValueError("candidate count exceeds explicit max_candidates")
    elem = torch.empty((), device=ops.device, dtype=ops.dtype).element_size()
    retained = int(cand.numel()) * (int(ops.context_count) * elem + 16)
    if retained > max_retained_bytes:
        raise ValueError("candidate row retained bytes exceed explicit max_retained_bytes")
    max_p = max(int(layer.X.shape[-1]) for layer in ops.factors.layers)
    per_pair = elem * (3 * max_p + 4)
    if per_pair > workspace_bytes:
        raise ValueError("one context-candidate row exceeds explicit workspace_bytes")
    return CandidateRowWorkspacePlan(candidate_count=int(cand.numel()), retained_bytes=retained,
                                     workspace_bytes=workspace_bytes, max_retained_bytes=max_retained_bytes,
                                     temp_bytes_per_context_candidate_upper=per_pair,
                                     principal_temporaries=("gathered_Gt[P]", "gathered_X[P]", "product_or_reduction[P]"))


def _chunk_shape(*, contexts: int, candidates: int, positions: int, elem: int, workspace_bytes: int) -> tuple[int, int]:
    per_pair = elem * (3 * int(positions) + 4)
    if per_pair > workspace_bytes:
        raise ValueError("one context-candidate row exceeds explicit workspace_bytes")
    max_pairs = max(1, workspace_bytes // max(1, per_pair))
    if max_pairs >= contexts:
        return contexts, max(1, min(candidates, max_pairs // max(1, contexts)))
    return max(1, min(contexts, max_pairs)), 1


def candidate_corrected_cell_rows(ops: ChartOperators, candidates: torch.Tensor | Sequence[int], *,
                                  max_candidates: int, workspace_bytes: int,
                                  max_retained_bytes: int,
                                  candidate_eligible: torch.Tensor | None = None) -> CandidateCellRows:
    """Build corrected A coefficients only for live requested candidates with explicit tiling."""

    _zz, cand_mask = validate_selection_inputs(ops, torch.zeros(ops.route_count, device=ops.device, dtype=ops.dtype), candidate_eligible)
    cand = _candidate_tensor(candidates, ops.route_count, device=ops.device)
    _require_live_candidates("candidate_corrected_cell_rows", cand_mask, cand)
    plan = forecast_candidate_row_workspace(ops, cand, max_candidates=max_candidates,
                                            workspace_bytes=workspace_bytes,
                                            max_retained_bytes=max_retained_bytes)
    values = torch.empty((ops.context_count, cand.numel()), device=ops.device, dtype=ops.dtype)
    fibers = torch.empty((cand.numel(),), device=ops.device, dtype=torch.long)
    if cand.numel() == 0:
        return CandidateCellRows(candidates=cand, fiber_ids=fibers, values=values, workspace_plan=plan)
    elem = values.element_size()
    offset_f = 0
    for layer, axis_layer in zip(ops.factors.layers, ops.axis.layers, strict=True):
        lo = axis_layer.route_offset
        hi = lo + axis_layer.route_count
        sel = torch.nonzero((cand >= lo) & (cand < hi), as_tuple=False).reshape(-1)
        if sel.numel() == 0:
            offset_f += axis_layer.fiber_count
            continue
        context_chunk, candidate_chunk = _chunk_shape(contexts=ops.context_count, candidates=int(sel.numel()),
                                                      positions=int(layer.X.shape[-1]), elem=elem,
                                                      workspace_bytes=workspace_bytes)
        for ks in range(0, int(sel.numel()), candidate_chunk):
            pos = sel[ks:ks + candidate_chunk]
            local = cand[pos] - lo
            f_local = local // axis_layer.routes_per_fiber
            j_local = local % axis_layer.routes_per_fiber
            wm = layer.W[f_local, j_local] * layer.mask[f_local, j_local]
            fibers[pos] = offset_f + f_local
            for cs in range(0, ops.context_count, context_chunk):
                ce = min(ops.context_count, cs + context_chunk)
                prod = layer.Gt[cs:ce][:, f_local, :] * layer.X[cs:ce][:, j_local, :]
                values[cs:ce, pos] = -wm[None, :] * prod.sum(dim=2)
                del prod
        offset_f += axis_layer.fiber_count
    return CandidateCellRows(candidates=cand, fiber_ids=fibers, values=values, workspace_plan=plan)


def _expected_fiber_ids(ops: ChartOperators, candidates: torch.Tensor) -> torch.Tensor:
    expected = torch.empty((candidates.numel(),), device=ops.device, dtype=torch.long)
    offset_f = 0
    for axis_layer in ops.axis.layers:
        lo = axis_layer.route_offset
        hi = lo + axis_layer.route_count
        sel = (candidates >= lo) & (candidates < hi)
        if bool(sel.any()):
            local = candidates[sel] - lo
            expected[sel] = offset_f + (local // axis_layer.routes_per_fiber)
        offset_f += axis_layer.fiber_count
    return expected


def _validate_candidate_rows(ops: ChartOperators, candidate_rows: CandidateCellRows) -> None:
    cand = _candidate_tensor(candidate_rows.candidates, ops.route_count, device=ops.device)
    fibers = torch.as_tensor(candidate_rows.fiber_ids)
    if fibers.ndim != 1 or fibers.numel() != cand.numel():
        raise ValueError("candidate row fiber ids have incompatible shape")
    if fibers.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("candidate row fiber ids must be integer")
    fibers = fibers.to(device=ops.device, dtype=torch.long)
    if fibers.numel() and (bool((fibers < 0).any()) or bool((fibers >= ops.fiber_count).any())):
        raise ValueError("candidate row fiber ids out of range")
    expected_fibers = _expected_fiber_ids(ops, cand)
    if fibers.numel() and not bool(torch.equal(fibers, expected_fibers)):
        raise ValueError("candidate row fiber ids do not match candidate route ownership")
    values = torch.as_tensor(candidate_rows.values)
    if values.ndim != 2 or values.shape != (ops.context_count, cand.numel()):
        raise ValueError("candidate row values have incompatible shape")
    if values.device != ops.device:
        raise ValueError("candidate row values must remain on the operator device")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("candidate row values must be finite")
    if candidate_rows.workspace_plan.candidate_count != cand.numel():
        raise ValueError("candidate row workspace plan candidate count mismatch")


def exact_work_marginals_from_cells(ops: ChartOperators, base_cells: torch.Tensor,
                                    candidate_rows: CandidateCellRows) -> CandidateWorkMarginals:
    """Exact finite ``F(z+e_j)-F(z)`` from cached base cells and validated candidate rows."""

    _validate_candidate_rows(ops, candidate_rows)
    cells_raw = torch.as_tensor(base_cells)
    if cells_raw.ndim != 2 or cells_raw.shape != (ops.context_count, ops.fiber_count):
        raise ValueError("base_cells must have shape [context_count, fiber_count]")
    if cells_raw.device != ops.device:
        raise ValueError("base_cells must remain on the operator device")
    if not bool(torch.isfinite(cells_raw).all()):
        raise ValueError("base_cells must be finite")
    cells = cells_raw.to(dtype=ops.dtype)
    fibers = candidate_rows.fiber_ids.to(device=ops.device, dtype=torch.long)
    values = candidate_rows.values.to(dtype=ops.dtype)
    base = cells[:, fibers]
    omega = ops.work_weights()[:, fibers]
    delta = (omega * ((base + values).abs() - base.abs())).sum(dim=0)
    base_work = float((ops.work_weights() * cells.abs()).sum().item())
    return CandidateWorkMarginals(candidates=candidate_rows.candidates.to(device=ops.device), marginal_work=delta,
                                  base_work=base_work, candidate_rows=candidate_rows)


def exact_work_marginal_replay(ops: ChartOperators, z: torch.Tensor, candidates: torch.Tensor | Sequence[int], *,
                               base_cells: torch.Tensor | None = None, max_candidates: int,
                               workspace_bytes: int, max_retained_bytes: int,
                               candidate_eligible: torch.Tensor | None = None) -> CandidateWorkMarginals:
    """Exact candidate-local native-work marginals with at most one base ``A z``."""

    zz, cand_mask = validate_selection_inputs(ops, z, candidate_eligible)
    cand = _candidate_tensor(candidates, ops.route_count, device=ops.device)
    if bool((zz[cand] != 0).any()):
        raise ValueError("work replay only supports adding currently unselected routes")
    _require_live_candidates("work replay", cand_mask, cand)
    cells = ops.a_forward(zz) if base_cells is None else base_cells
    rows = candidate_corrected_cell_rows(ops, cand, max_candidates=max_candidates,
                                         workspace_bytes=workspace_bytes,
                                         max_retained_bytes=max_retained_bytes,
                                         candidate_eligible=cand_mask)
    return exact_work_marginals_from_cells(ops, cells, rows)


def legalize_candidate_order(order: torch.Tensor | Sequence[int], *, candidate_eligible: torch.Tensor,
                             physical_live: torch.Tensor, fiber_of_route: torch.Tensor,
                             fiber_count: int) -> torch.Tensor:
    """Vectorized queue legality with full physical survivor counts."""

    fiber_count = _positive_int("fiber_count", fiber_count)
    device = fiber_of_route.device
    ord_t = _candidate_tensor(order, fiber_of_route.numel(), device=device)
    cand = _binary_tensor("candidate_eligible", candidate_eligible, fiber_of_route.numel(), device=device, dtype=torch.float64)
    live = _binary_tensor("physical_live", physical_live, fiber_of_route.numel(), device=device, dtype=torch.float64)
    usable = ord_t[(cand[ord_t] > 0) & (live[ord_t] > 0)]
    if usable.numel() == 0:
        return usable
    live_per_fiber = torch.bincount(fiber_of_route[live > 0], minlength=fiber_count).to(torch.long)
    fibers = fiber_of_route[usable]
    fiber_sorted, perm = torch.sort(fibers, stable=True)
    counts = torch.bincount(fiber_sorted, minlength=fiber_count)
    starts = torch.cumsum(counts, 0) - counts
    ranks = torch.empty_like(perm)
    ranks[perm] = torch.arange(perm.numel(), device=device, dtype=perm.dtype) - starts[fiber_sorted]
    keep = ranks < (live_per_fiber[fibers] - 1)
    return usable[keep]


def lexicographic_queue_order(candidates: torch.Tensor | Sequence[int], keys: Sequence[torch.Tensor], *,
                              route_count: int) -> torch.Tensor:
    """Stable lexicographic order from queue-aligned keys, then route index."""

    if not keys:
        raise ValueError("at least one ranking key is required")
    first = torch.as_tensor(keys[0])
    if first.ndim != 1:
        raise ValueError("ranking keys must be one-dimensional")
    cand = _candidate_tensor(candidates, int(route_count), device=first.device)
    if cand.numel() != first.numel():
        raise ValueError("queue-aligned ranking keys must match candidate count")
    order = torch.arange(cand.numel(), device=first.device)
    keyed = []
    for key in keys:
        raw = torch.as_tensor(key)
        if raw.ndim != 1 or raw.numel() != cand.numel():
            raise ValueError("queue-aligned ranking keys must match candidate count")
        k = raw.to(device=first.device, dtype=torch.float64)
        if not bool(torch.isfinite(k).all()):
            raise ValueError("ranking keys must be finite")
        keyed.append(k)
    route_key = cand.to(torch.float64)
    for key in reversed([*keyed, route_key]):
        order = order[torch.argsort(key[order], stable=True)]
    return cand[order]


def lexicographic_candidate_order(candidates: torch.Tensor | Sequence[int], keys: Sequence[torch.Tensor]) -> torch.Tensor:
    """Stable lexicographic order from full route-length keys, then route index."""

    if not keys:
        raise ValueError("at least one ranking key is required")
    first = torch.as_tensor(keys[0])
    if first.ndim != 1:
        raise ValueError("ranking keys must be one-dimensional")
    cand = _candidate_tensor(candidates, first.numel(), device=first.device)
    queue_keys = []
    for key in keys:
        raw = torch.as_tensor(key)
        if raw.ndim != 1 or raw.numel() != first.numel():
            raise ValueError("ranking keys must be one-dimensional and route-length")
        queue_keys.append(raw.to(device=first.device, dtype=torch.float64)[cand])
    return lexicographic_queue_order(cand, queue_keys, route_count=first.numel())


def bounded_candidate_marginals(ops: ChartOperators, geom: DiagonalShrunkGeometry, *, z: torch.Tensor,
                                candidate_queue: torch.Tensor | Sequence[int], candidate_eligible: torch.Tensor | None = None,
                                mode: SelectionMode = "logit_aligned", loss_price: torch.Tensor | None = None,
                                lambda_work_per_energy: float | None = None, max_candidates: int,
                                workspace_bytes: int, max_retained_bytes: int) -> CandidateMarginalBatch:
    """Compute exact finite ΔQ/ΔF and a legal order for one bounded queue."""

    if mode not in ("logit_aligned", "loss_aligned_ablation", "diagnostic_lagrange"):
        raise ValueError("unknown selection mode")
    if mode == "loss_aligned_ablation" and loss_price is None:
        raise ValueError("loss_aligned_ablation requires loss_price")
    if mode == "diagnostic_lagrange" and (lambda_work_per_energy is None or not math.isfinite(float(lambda_work_per_energy))):
        raise ValueError("diagnostic_lagrange requires a finite work/energy lambda")
    zz, cand_mask = validate_selection_inputs(ops, z, candidate_eligible)
    queue = _candidate_tensor(candidate_queue, ops.route_count, device=ops.device)
    if queue.numel() > _positive_int("max_candidates", max_candidates):
        raise ValueError("candidate queue exceeds explicit max_candidates")
    if bool((zz[queue] != 0).any()):
        raise ValueError("candidate queue includes already selected routes")
    _require_live_candidates("candidate queue", cand_mask, queue)
    dq = q_add_marginal(ops, geom, zz, queue, candidate_eligible=cand_mask)
    base_cells = ops.a_forward(zz)
    replay = exact_work_marginal_replay(ops, zz, queue, base_cells=base_cells,
                                        max_candidates=max_candidates, workspace_bytes=workspace_bytes,
                                        max_retained_bytes=max_retained_bytes, candidate_eligible=cand_mask)
    if mode == "loss_aligned_ablation":
        loss = _flat_tensor("loss_price", loss_price, ops.route_count, device=ops.device, dtype=ops.dtype)
        raw_order = lexicographic_queue_order(queue, [loss[queue], dq, replay.marginal_work], route_count=ops.route_count)
    elif mode == "diagnostic_lagrange":
        local = replay.marginal_work + float(lambda_work_per_energy) * dq
        raw_order = lexicographic_queue_order(queue, [local, replay.marginal_work, dq], route_count=ops.route_count)
    else:
        raw_order = lexicographic_queue_order(queue, [replay.marginal_work, dq], route_count=ops.route_count)
    parent_live = _binary_tensor("ops.eligible()", ops.eligible(), ops.route_count, device=ops.device, dtype=ops.dtype)
    physical_live = ((parent_live > 0) & (zz <= 0)).to(ops.dtype)
    legal = legalize_candidate_order(raw_order, candidate_eligible=cand_mask, physical_live=physical_live,
                                     fiber_of_route=ops.fiber_of_route(), fiber_count=ops.fiber_count)
    return CandidateMarginalBatch(candidates=queue, delta_q=dq, delta_f=replay.marginal_work,
                                  legal_order=legal, mode=mode)

