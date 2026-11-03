"""Bounded controller policy built on tractable-v1 selection primitives.

This module owns operator-based binary proposal generation. It uses partial
queues, exact chart endpoint replay and, by default, accumulated frozen-input
ReLU checks. The last check is an observed local constraint, not a proof about
the joint nonlinear network, pooling, unseen contexts or intermediate doses.
It grants no global binary optimality, population, bootstrap, or C/K/O authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Any, Literal
import math
import time

import torch

from .operators import ChartOperators
from .chart_safeguards import local_action_kinks
from .solvers import DiagonalShrunkGeometry
from .selection import (
    CandidateCellRows,
    candidate_corrected_cell_rows,
    exact_work_marginals_from_cells,
    exact_work_marginal_replay,
    forecast_candidate_row_workspace,
    legalize_candidate_order,
    lexicographic_queue_order,
    singleton_finite_work_screen,
    validate_selection_inputs,
)
from .selection_refinement import FixedResourceRefinementConfig, refine_frontier_fixed_resource

PolicyMode = Literal["logit_work", "expected_damage"]
StageMode = Literal["frontier", "least_work", "expected_damage"]
SELECTION_POLICY_SCHEMA = "tractable_v1_selection_policy_config_v9"


@dataclass(frozen=True)
class SelectionPolicyConfig:
    min_candidates: int = 256
    max_candidates: int = 4096
    queue_multiplier: int = 4
    max_block: int = 4096
    # None exhausts the finite greedy path. An explicit integer is a bounded
    # diagnostic/ablation budget, never a proof that its frontier was reached.
    max_evaluations: int | None = None
    max_refinement_exchanges: int | None = None
    scan_chunk_routes: int = 1 << 20
    candidate_workspace_bytes: int = 64 << 20
    candidate_retained_bytes: int = 256 << 20
    frontier_tolerance: float = 0.0
    rerank_blocks: bool = True
    require_local_phase_stability: bool = True
    phase_workspace_bytes: int = 256 << 20

    def __post_init__(self) -> None:
        for name in (
            "min_candidates", "max_candidates", "queue_multiplier", "max_block",
            "scan_chunk_routes", "candidate_workspace_bytes",
            "candidate_retained_bytes",
            "phase_workspace_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive int")
        if self.max_evaluations is not None and (type(self.max_evaluations) is not int or self.max_evaluations <= 0):
            raise ValueError('max_evaluations must be None or a positive int')
        if self.max_refinement_exchanges is not None and (type(self.max_refinement_exchanges) is not int or self.max_refinement_exchanges <= 0):
            raise ValueError('max_refinement_exchanges must be None or a positive int')
        if self.min_candidates > self.max_candidates:
            raise ValueError("min_candidates must not exceed max_candidates")
        tol = float(self.frontier_tolerance)
        if not math.isfinite(tol) or tol < 0.0 or tol >= 1.0:
            raise ValueError("frontier_tolerance must be finite in [0, 1)")
        if type(self.rerank_blocks) is not bool:
            raise ValueError("rerank_blocks must be a bool")
        if type(self.require_local_phase_stability) is not bool:
            raise ValueError("require_local_phase_stability must be a bool")
        if self.phase_workspace_bytes < 1024:
            raise ValueError("phase_workspace_bytes must be at least 1024")

    def to_json(self) -> dict[str, Any]:
        return dict(schema=SELECTION_POLICY_SCHEMA, **self.__dict__)


@dataclass
class SelectionPolicyResult:
    order: torch.Tensor
    z: torch.Tensor
    resource: int
    energy: float
    half_energy: float
    work: float
    first_order: float
    predicted_damage: float
    key: str
    stage: str
    frontier_resource: int
    frontier_work: float | None
    frontier_energy: float | None
    least_work_status: str | None
    evaluations: int
    seconds: float
    operator_calls: dict[str, int] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    authority: str = "proposal_only_observed_chart"
    optimality_authority: bool = False
    population_certified: bool = False
    bootstrap_certified: bool = False
    production_commit_authority: bool = False

    def to_action_dict(self) -> dict[str, Any]:
        return dict(
            order=self.order,
            resource=int(self.resource),
            z=self.z,
            energy=float(self.energy),
            half_energy=float(self.half_energy),
            predicted_damage=float(self.predicted_damage),
            first_order=float(self.first_order),
            work=float(self.work),
            key=self.key,
            stage=self.stage,
            frontier_resource=int(self.frontier_resource),
            frontier_work=self.frontier_work,
            frontier_energy=self.frontier_energy,
            least_work_status=self.least_work_status,
            lam=None,
            evaluations=int(self.evaluations),
            operator_calls=dict(self.operator_calls),
            history=self.history[-64:],
            authority=self.authority,
            optimality_authority=self.optimality_authority,
            population_certified=self.population_certified,
            bootstrap_certified=self.bootstrap_certified,
            production_commit_authority=self.production_commit_authority,
            seconds=float(self.seconds),
        )


@dataclass
class _Endpoint:
    stats: dict[str, float]
    cells: torch.Tensor


@dataclass
class _QueueReplay:
    queue: torch.Tensor
    legal: torch.Tensor
    dq_legal: torch.Tensor
    candidate_eligible: torch.Tensor
    primary: torch.Tensor
    delta_q_full: torch.Tensor
    qsize: int
    work_key: torch.Tensor
    candidate_rows: CandidateCellRows | None


def _new_counters() -> dict[str, int]:
    return dict(a_forward=0, a_adjoint=0, geom_energy=0, geom_cross=0, queue_replays=0,
                endpoint_replays=0, queue_primary_cache_reuses=0,
                queue_candidate_routes_replayed=0, queue_candidate_routes_reused=0,
                queue_candidate_rows_peak_bytes=0, queue_row_cache_budget_fallbacks=0)


def _merge_counters(*items: dict[str, int]) -> dict[str, int]:
    keys: set[str] = set()
    for item in items:
        keys.update(item)
    return {key: (max if key.endswith("_peak_bytes") else sum)(int(item.get(key, 0)) for item in items)
            for key in sorted(keys)}


def _positive_radius(radius: float) -> float:
    r = float(radius)
    if not math.isfinite(r) or r <= 0.0:
        raise ValueError("radius must be finite and positive")
    return r


def _require_chart(chart: dict[str, Any]) -> tuple[ChartOperators, DiagonalShrunkGeometry, torch.Tensor, torch.Tensor]:
    try:
        ops = chart["ops"]
        geom = chart["geom"]
        eligible = chart["eligible"]
        g = chart["g"]
    except KeyError as exc:
        raise ValueError(f"chart missing required key {exc.args[0]}") from exc
    if not isinstance(ops, ChartOperators):
        raise ValueError("chart['ops'] must be a ChartOperators")
    if not isinstance(geom, DiagonalShrunkGeometry):
        raise ValueError("chart['geom'] must be a DiagonalShrunkGeometry")
    return ops, geom, eligible, g


def _finite_route_vector(name: str, value: torch.Tensor, route_count: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    raw = torch.as_tensor(value)
    if raw.ndim != 1 or raw.numel() != route_count:
        raise ValueError(f"{name} must be a one-dimensional route vector")
    if not bool(torch.isfinite(raw).all()):
        raise ValueError(f"{name} must be finite")
    return raw.to(device=device, dtype=dtype)


def _partial_smallest_by_primary(primary: torch.Tensor, candidate_mask: torch.Tensor, *, k: int,
                                 chunk_routes: int) -> torch.Tensor:
    """Return a bounded primary-key queue with chunked top-k, not a global sort."""

    if primary.ndim != 1 or candidate_mask.ndim != 1 or primary.numel() != candidate_mask.numel():
        raise ValueError("primary and candidate_mask must be aligned route vectors")
    if not bool(torch.isfinite(primary).all()):
        raise ValueError("primary key must be finite")
    if not bool(((candidate_mask == 0) | (candidate_mask == 1)).all()):
        raise ValueError("candidate_mask must be binary")
    k = min(int(k), int(candidate_mask.sum().item()))
    if k <= 0:
        return torch.empty(0, device=primary.device, dtype=torch.long)
    chunk_routes = int(chunk_routes)
    if chunk_routes <= 0:
        raise ValueError("chunk_routes must be positive")
    best_values: torch.Tensor | None = None
    best_indices: torch.Tensor | None = None
    n = primary.numel()
    for start in range(0, n, chunk_routes):
        stop = min(n, start + chunk_routes)
        local_mask = candidate_mask[start:stop] > 0
        if not bool(local_mask.any()):
            continue
        local_idx = torch.nonzero(local_mask, as_tuple=False).reshape(-1)
        local_values = primary[start:stop][local_idx]
        take = min(k, int(local_values.numel()))
        values, pos = torch.topk(local_values, take, largest=False, sorted=False)
        indices = local_idx[pos] + start
        if best_values is None:
            best_values, best_indices = values, indices
        else:
            merged_values = torch.cat([best_values, values])
            merged_indices = torch.cat([best_indices, indices])
            take = min(k, int(merged_values.numel()))
            best_values, pos = torch.topk(merged_values, take, largest=False, sorted=False)
            best_indices = merged_indices[pos]
    if best_indices is None:
        return torch.empty(0, device=primary.device, dtype=torch.long)
    # Deterministic queue order within the retained partial set; ties outside the retained top-k are not a global claim.
    return lexicographic_queue_order(best_indices, [primary[best_indices]], route_count=primary.numel())


def _queue_size(config: SelectionPolicyConfig, block: int) -> int:
    return min(config.max_candidates, max(config.min_candidates, config.queue_multiplier * max(1, int(block))))


def _block_from_budget(delta_q_ordered: torch.Tensor, *, remaining_half_q: float, max_block: int,
                       target_remaining: int | None) -> int:
    if delta_q_ordered.numel() == 0:
        return 0
    positive_half = torch.clamp(delta_q_ordered, min=0.0) * 0.5
    cumulative = torch.cumsum(positive_half.to(torch.float64), dim=0)
    count = int((cumulative <= max(0.0, float(remaining_half_q))).sum().item())
    count = max(1, count)
    count = min(count, int(max_block), int(delta_q_ordered.numel()))
    if target_remaining is not None:
        count = min(count, max(0, int(target_remaining)))
    return count


def _queue_values(queue: torch.Tensor, values: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
    if queue.numel() == 0:
        return torch.empty(0, device=values.device, dtype=values.dtype)
    sorted_routes, perm = torch.sort(routes)
    pos = torch.searchsorted(sorted_routes, queue)
    if bool((pos >= sorted_routes.numel()).any()) or not bool(torch.equal(sorted_routes[pos], queue)):
        raise ValueError("queue values missing route")
    return values[perm[pos]]


def _removable_candidate_mask(ops: ChartOperators, z: torch.Tensor, candidate_eligible: torch.Tensor) -> torch.Tensor:
    """Candidate subset whose fibers have a physical survivor after one more removal."""

    zz, cand = validate_selection_inputs(ops, z, candidate_eligible)
    physical_live = validate_selection_inputs(ops, zz)[1]
    fib = ops.fiber_of_route()
    live_per_fiber = torch.bincount(fib[physical_live > 0], minlength=ops.fiber_count).to(torch.long)
    removable = live_per_fiber[fib] > 1
    return ((cand > 0) & removable).to(ops.dtype)


def _current_delta_q_full(ops: ChartOperators, geom: DiagonalShrunkGeometry, z: torch.Tensor,
                          counters: dict[str, int] | None = None) -> torch.Tensor:
    if int(z.sum().item()) == 0:
        return geom.self_term()
    if counters is not None:
        counters["geom_cross"] = counters.get("geom_cross", 0) + 1
    return geom.self_term() + 2.0 * geom.cross_term(ops, z)


def _endpoint_stats(chart: dict[str, Any], z: torch.Tensor, *, cells: torch.Tensor | None = None,
                    counters: dict[str, int] | None = None) -> _Endpoint:
    ops, geom, _eligible, g = _require_chart(chart)
    resource = int(z.sum().item())
    response = None
    if cells is None:
        if resource <= 0:
            cells = torch.zeros((ops.context_count, ops.fiber_count), device=ops.device, dtype=ops.dtype)
        else:
            if counters is not None:
                counters["a_j_forward"] = counters.get("a_j_forward", 0) + 1
            cells, response = ops.a_j_forward(z)
    if resource <= 0:
        return _Endpoint(cells=cells, stats=dict(resource=0, energy=0.0, half_energy=0.0, work=0.0,
                                                 first_order=0.0, predicted_damage=0.0))
    if counters is not None:
        counters["geom_energy"] = counters.get("geom_energy", 0) + 1
    energy = float((geom.energy(ops, z) if response is None else geom.energy_from_response(z, response)).item())
    first_order = float((ops.context_weights * g.to(ops.dtype) * cells.sum(dim=1)).sum().item())
    work = float((ops.work_weights() * cells.abs()).sum().item())
    return _Endpoint(cells=cells, stats=dict(resource=resource, energy=energy, half_energy=0.5 * energy,
                                             work=work, first_order=first_order,
                                             predicted_damage=first_order + 0.5 * energy))


def _work_subgradient_from_cells(ops: ChartOperators, cells: torch.Tensor, counters: dict[str, int] | None = None) -> torch.Tensor:
    if cells.ndim != 2 or cells.shape != (ops.context_count, ops.fiber_count):
        raise ValueError("cached cells must have shape [context_count, fiber_count]")
    if counters is not None:
        counters["a_adjoint"] = counters.get("a_adjoint", 0) + 1
    return ops.a_adjoint(ops.work_weights() * torch.sign(cells))


def _stage_primary(chart: dict[str, Any], z: torch.Tensor, stage: StageMode, delta_q_full: torch.Tensor,
                   *, cells: torch.Tensor, config: SelectionPolicyConfig,
                   counters: dict[str, int] | None = None) -> torch.Tensor:
    ops, geom, _eligible, _g = _require_chart(chart)
    if stage == "expected_damage":
        if "d" not in chart:
            raise ValueError("expected_damage stage requires chart['d'] loss price")
        return _finite_route_vector("chart['d']", chart["d"], ops.route_count, device=ops.device, dtype=ops.dtype)
    if not config.rerank_blocks:
        if stage == "least_work":
            return singleton_finite_work_screen(ops)
        return geom.self_term()
    if stage == "least_work":
        if int(z.sum().item()) == 0:
            return singleton_finite_work_screen(ops)
        return _work_subgradient_from_cells(ops, cells, counters)
    return delta_q_full


def _rank_stage_queue(chart: dict[str, Any], z: torch.Tensor, queue: torch.Tensor, stage: StageMode,
                      config: SelectionPolicyConfig, delta_q_full: torch.Tensor, *, base_cells: torch.Tensor,
                      candidate_eligible: torch.Tensor, primary: torch.Tensor | None = None,
                      qsize: int | None = None, counters: dict[str, int] | None = None,
                      unchanged_endpoint_cache: _QueueReplay | None = None) -> _QueueReplay:
    ops, geom, _eligible, _g = _require_chart(chart)
    dq = delta_q_full[queue]
    if config.rerank_blocks:
        # Reuse rows only at an unchanged endpoint. Recompute the full-width
        # work reduction: reducing just the new columns can change floating
        # reduction order (including FP64), and hence ties near a boundary.
        reused = torch.zeros(queue.numel(), device=ops.device, dtype=torch.bool)
        old_rows = None if unchanged_endpoint_cache is None else unchanged_endpoint_cache.candidate_rows
        if old_rows is not None and unchanged_endpoint_cache.queue.numel():
            old_routes, perm = torch.sort(unchanged_endpoint_cache.queue)
            positions = torch.searchsorted(old_routes, queue)
            safe = positions.clamp(max=old_routes.numel() - 1)
            reused = (positions < old_routes.numel()) & (old_routes[safe] == queue)
        missing = queue[~reused]
        row_limits = dict(max_candidates=config.max_candidates,
            workspace_bytes=config.candidate_workspace_bytes,
            max_retained_bytes=config.candidate_retained_bytes)
        full_plan = forecast_candidate_row_workspace(ops, queue, **row_limits)
        missing_plan = forecast_candidate_row_workspace(ops, missing, **row_limits)
        # Reserve one full-row equivalent for indexing/copy scratch as well.
        cache_peak = 2 * full_plan.retained_bytes + missing_plan.retained_bytes
        if old_rows is not None:
            cache_peak += old_rows.workspace_plan.retained_bytes
        if old_rows is not None and cache_peak <= config.candidate_retained_bytes:
            values = torch.empty((ops.context_count, queue.numel()), device=ops.device, dtype=ops.dtype)
            fibers = torch.empty(queue.numel(), device=ops.device, dtype=torch.long)
            old_positions = perm[safe[reused]]
            values[:, reused] = old_rows.values[:, old_positions]
            fibers[reused] = old_rows.fiber_ids[old_positions]
            if missing.numel():
                new_rows = candidate_corrected_cell_rows(ops, missing,
                    candidate_eligible=candidate_eligible, **row_limits)
                values[:, ~reused] = new_rows.values
                fibers[~reused] = new_rows.fiber_ids
                del new_rows
            rows = CandidateCellRows(candidates=queue, fiber_ids=fibers, values=values, workspace_plan=full_plan)
            replay = exact_work_marginals_from_cells(ops, base_cells, rows)
        else:
            # The retained-row budget covers old + assembled + newly built
            # rows and copy scratch. Drop old storage before a full fallback.
            if old_rows is not None and counters is not None:
                counters["queue_row_cache_budget_fallbacks"] = counters.get("queue_row_cache_budget_fallbacks", 0) + 1
            old_rows = None
            if unchanged_endpoint_cache is not None:
                unchanged_endpoint_cache.candidate_rows = None
            missing = queue
            reused.zero_()
            replay = exact_work_marginal_replay(
                ops, z, queue, base_cells=base_cells,
                candidate_eligible=candidate_eligible, **row_limits,
            )
        work_key = replay.marginal_work
        candidate_rows = replay.candidate_rows
        if missing.numel():
            if counters is not None:
                counters["queue_replays"] = counters.get("queue_replays", 0) + 1
                counters["queue_candidate_routes_replayed"] = counters.get("queue_candidate_routes_replayed", 0) + int(missing.numel())
        if counters is not None:
            counters["queue_candidate_routes_reused"] = counters.get("queue_candidate_routes_reused", 0) + int(reused.sum().item())
            retained_peak = cache_peak if old_rows is not None else full_plan.retained_bytes
            counters["queue_candidate_rows_peak_bytes"] = max(counters.get("queue_candidate_rows_peak_bytes", 0), retained_peak)
        q_key = dq
    else:
        candidate_rows = None
        if counters is not None:
            counters["frozen_queue_rankings"] = counters.get("frozen_queue_rankings", 0) + 1
        work_key = singleton_finite_work_screen(ops)[queue]
        q_key = geom.self_term()[queue]
    if stage == "frontier":
        raw_order = lexicographic_queue_order(queue, [q_key, work_key], route_count=ops.route_count)
    elif stage == "least_work":
        raw_order = lexicographic_queue_order(queue, [work_key, q_key], route_count=ops.route_count)
    else:
        loss = _finite_route_vector("chart['d']", chart["d"], ops.route_count, device=ops.device, dtype=ops.dtype)
        raw_order = lexicographic_queue_order(queue, [loss[queue], q_key, work_key], route_count=ops.route_count)
    physical_live = validate_selection_inputs(ops, z)[1]
    legal = legalize_candidate_order(raw_order, candidate_eligible=candidate_eligible, physical_live=physical_live,
                                     fiber_of_route=ops.fiber_of_route(), fiber_count=ops.fiber_count)
    primary_source = delta_q_full if primary is None else primary
    return _QueueReplay(queue=queue, legal=legal, dq_legal=_queue_values(legal, dq, queue),
                        candidate_eligible=candidate_eligible.clone(), primary=primary_source,
                        delta_q_full=delta_q_full, qsize=int(queue.numel() if qsize is None else qsize),
                        work_key=work_key, candidate_rows=candidate_rows)


def _refresh_queue_cache_after_singleton_rejection(chart: dict[str, Any], z: torch.Tensor, cache: _QueueReplay,
                                                   rejected: torch.Tensor, stage: StageMode,
                                                   config: SelectionPolicyConfig, *, block: int,
                                                   base_cells: torch.Tensor,
                                                   counters: dict[str, int] | None = None) -> _QueueReplay | None:
    """Rebuild queue membership from cached primary data after unchanged-endpoint rejection.

    A rejected singleton changes only the candidate mask.  It does not change
    ``z``, cached ``A z`` cells, ``Q(z)``, the full primary vector, or the full
    ``Delta Q`` vector.  Recomputing the partial queue from those cached vectors
    preserves the existing partial-topk/tie semantics without paying another
    full ``A^T`` or ``geom.cross`` construction. Within the retained-row budget,
    only newly entering routes need candidate-row construction; the complete
    work reduction is replayed in the new queue layout. Any accepted action
    discards this entire cache.
    """

    ops, _geom, _eligible, _g = _require_chart(chart)
    candidate_eligible = cache.candidate_eligible.clone()
    if rejected.numel():
        candidate_eligible[rejected.to(device=ops.device, dtype=torch.long)] = 0.0
    desired = min(int(candidate_eligible.sum().item()), _queue_size(config, block))
    if desired <= 0:
        return None
    if counters is not None:
        counters["queue_primary_cache_reuses"] = counters.get("queue_primary_cache_reuses", 0) + 1
    queue = _partial_smallest_by_primary(cache.primary, candidate_eligible, k=desired,
                                         chunk_routes=config.scan_chunk_routes)
    return _rank_stage_queue(chart, z, queue, stage, config, cache.delta_q_full,
                             base_cells=base_cells, candidate_eligible=candidate_eligible,
                             primary=cache.primary, qsize=desired, counters=counters,
                             unchanged_endpoint_cache=cache)


def _empty_result(chart: dict[str, Any], stage: StageMode, *, started: float, counters: dict[str, int]) -> SelectionPolicyResult:
    ops, _geom, _eligible, _g = _require_chart(chart)
    z = torch.zeros(ops.route_count, device=ops.device, dtype=ops.dtype)
    final = _endpoint_stats(chart, z, counters=counters).stats
    return SelectionPolicyResult(order=torch.empty(0, device=ops.device, dtype=torch.long), z=z, resource=0,
                                 energy=0.0, half_energy=0.0, work=0.0, first_order=0.0,
                                 predicted_damage=0.0, key=stage, stage="no_candidate_routes",
                                 frontier_resource=0, frontier_work=None, frontier_energy=None,
                                 least_work_status="no_candidate_routes", evaluations=0,
                                 seconds=time.perf_counter() - started, operator_calls=dict(counters),
                                 history=[dict(stage=stage, reason="no_candidate_routes",
                                               resource=final["resource"])])


def _grow_stage(chart: dict[str, Any], radius: float, *, stage: StageMode, config: SelectionPolicyConfig,
                target_resource: int | None = None) -> SelectionPolicyResult:
    started = time.perf_counter()
    counters = _new_counters()
    ops, _geom, eligible, _g = _require_chart(chart)
    r = _positive_radius(radius)
    z = torch.zeros(ops.route_count, device=ops.device, dtype=ops.dtype)
    remaining = validate_selection_inputs(ops, z, eligible)[1].clone()
    total_live = int(remaining.sum().item())
    if total_live <= 0:
        return _empty_result(chart, stage, started=started, counters=counters)
    block = min(config.max_block, max(1, min(total_live, target_resource if target_resource is not None else total_live)))
    evaluations = 0
    # Retain bounded diagnostics even when the finite path takes many trials.
    history = deque(maxlen=64)
    taken: list[torch.Tensor] = []
    terminal_reason = "evaluation_budget_exhausted"
    current_endpoint = _endpoint_stats(chart, z, counters=counters)
    queue_cache: _QueueReplay | None = None

    while config.max_evaluations is None or evaluations < config.max_evaluations:
        current = current_endpoint.stats
        if target_resource is not None and current["resource"] >= target_resource:
            terminal_reason = "target_resource_reached"
            break
        remaining_half_q = r - current["half_energy"]
        if remaining_half_q < 0.0:
            terminal_reason = "current_endpoint_exceeds_radius"
            break
        live_left = int(remaining.sum().item())
        if live_left <= 0:
            terminal_reason = "candidate_queue_exhausted"
            break
        if queue_cache is None:
            screen = _removable_candidate_mask(ops, z, remaining)
            removable_left = int(screen.sum().item())
            if removable_left <= 0:
                terminal_reason = "no_legal_candidate_routes"
                break
            qsize = min(removable_left, _queue_size(config, block))
            delta_q_full = _current_delta_q_full(ops, chart["geom"], z, counters)
            primary = _stage_primary(chart, z, stage, delta_q_full, cells=current_endpoint.cells, config=config, counters=counters)
            queue = _partial_smallest_by_primary(primary, screen, k=qsize, chunk_routes=config.scan_chunk_routes)
            if queue.numel() == 0:
                terminal_reason = "no_candidate_in_partial_queue"
                break
            queue_cache = _rank_stage_queue(chart, z, queue, stage, config, delta_q_full,
                                            base_cells=current_endpoint.cells, candidate_eligible=screen,
                                            primary=primary, qsize=qsize, counters=counters)
        if queue_cache.legal.numel() == 0:
            terminal_reason = "no_legal_candidate_in_partial_queue"
            break
        target_remaining = None if target_resource is None else max(0, target_resource - int(current["resource"]))
        take_count = _block_from_budget(queue_cache.dq_legal, remaining_half_q=remaining_half_q,
                                        max_block=min(block, config.max_block), target_remaining=target_remaining)
        if take_count <= 0:
            terminal_reason = "budget_admits_no_candidate"
            break
        take = queue_cache.legal[:take_count]
        trial = z.clone()
        trial[take] = 1.0
        counters["endpoint_replays"] = counters.get("endpoint_replays", 0) + 1
        trial_endpoint = _endpoint_stats(chart, trial, counters=counters)
        stats = trial_endpoint.stats
        evaluations += 1
        energy_ok = stats["half_energy"] <= r
        phase = None
        if energy_ok and config.require_local_phase_stability:
            # A null tangent does not prevent a finite removal from opening a
            # gate. Replay the accumulated trial, including previously taken
            # blocks, before it becomes the next selection endpoint.
            phase = local_action_kinks(ops, trial, workspace_bytes=config.phase_workspace_bytes)
            counters["local_phase_replays"] = counters.get("local_phase_replays", 0) + 1
        phase_ok = phase is None or phase["local_phase_unchanged"]
        accepted = energy_ok and phase_ok
        record = dict(stage=stage, resource_before=int(current["resource"]), queue=int(queue_cache.queue.numel()),
                            legal=int(queue_cache.legal.numel()), block=int(take.numel()), accepted=bool(accepted),
                            half_energy=stats["half_energy"], radius=r)
        if config.require_local_phase_stability:
            record.update(energy_ok=bool(energy_ok), local_phase=phase,
                          refusal_reason=None if accepted else ("response_energy" if not energy_ok else "local_phase_crossing"))
        history.append(record)
        if accepted:
            z = trial
            current_endpoint = trial_endpoint
            taken.append(take)
            # Interference changes with z. A singleton rejected at the previous
            # endpoint is not permanently ineligible at the new endpoint.
            remaining = ((eligible > 0) & (z <= 0)).to(ops.dtype)
            queue_cache = None
            if target_resource is not None and int(z.sum().item()) >= target_resource:
                terminal_reason = "target_resource_reached"
                break
            block = min(config.max_block, max(1, int(take.numel()) * 2))
        else:
            if take_count == 1:
                remaining[take] = 0.0
                block = 1
                if config.max_evaluations is None or evaluations < config.max_evaluations:
                    queue_cache = _refresh_queue_cache_after_singleton_rejection(
                        chart, z, queue_cache, take, stage, config, block=block,
                        base_cells=current_endpoint.cells, counters=counters,
                    )
                else:
                    queue_cache = None
            else:
                block = max(1, take_count // 2)
    final = current_endpoint.stats
    order = torch.cat(taken) if taken else torch.empty(0, device=ops.device, dtype=torch.long)
    return SelectionPolicyResult(order=order, z=z, resource=int(final["resource"]), energy=final["energy"],
                                 half_energy=final["half_energy"], work=final["work"], first_order=final["first_order"],
                                 predicted_damage=final["predicted_damage"], key=stage, stage=terminal_reason,
                                 frontier_resource=int(final["resource"]), frontier_work=None, frontier_energy=None,
                                 least_work_status=None, evaluations=evaluations,
                                 seconds=time.perf_counter() - started, operator_calls=dict(counters), history=list(history))


def _frontier_target_resource(frontier_resource: int, tolerance: float) -> int:
    if frontier_resource <= 0:
        return 0
    return max(1, min(int(frontier_resource), int(math.ceil((1.0 - float(tolerance)) * int(frontier_resource)))))


def choose_action(chart: dict[str, Any], radius: float, *, mode: PolicyMode,
                  config: SelectionPolicyConfig = SelectionPolicyConfig(),
                  target_resource: int | None = None) -> dict[str, Any]:
    """Choose a bounded binary proposal for the canonical controller."""

    started = time.perf_counter()
    if mode not in ("logit_work", "expected_damage"):
        raise ValueError("unknown policy mode")
    if target_resource is not None and (type(target_resource) is not int or target_resource <= 0):
        raise ValueError("target_resource must be None or a positive int")
    if mode == "expected_damage":
        result = _grow_stage(chart, radius, stage="expected_damage", config=config, target_resource=target_resource)
        data = result.to_action_dict()
        data.update(selection_policy=config.to_json(), requested_resource=target_resource, least_work_status="not_applicable_expected_damage_ablation",
                    growth_termination_reason=result.stage,
                    growth_computation_exhausted=result.stage == 'evaluation_budget_exhausted',
                    total_seconds=time.perf_counter() - started, total_evaluations=result.evaluations,
                    total_operator_calls=dict(result.operator_calls))
        return data

    frontier = _grow_stage(chart, radius, stage="frontier", config=config, target_resource=target_resource)
    if frontier.resource <= 0:
        data = frontier.to_action_dict()
        if frontier.stage in ("no_candidate_routes", "no_legal_candidate_routes"):
            final_stage = frontier.stage
            final_status = frontier.stage
        else:
            final_stage = "frontier_only"
            final_status = "no_frontier_resource"
        data.update(selection_policy=config.to_json(), requested_resource=target_resource, stage=final_stage, least_work_status=final_status,
                    growth_termination_reason=frontier.stage,
                    growth_computation_exhausted=frontier.stage == 'evaluation_budget_exhausted',
                    total_seconds=time.perf_counter() - started, total_evaluations=frontier.evaluations,
                    total_operator_calls=dict(frontier.operator_calls))
        return data
    requested_resource = target_resource
    target_resource = _frontier_target_resource(frontier.resource, config.frontier_tolerance)
    warm = refine_frontier_fixed_resource(
        chart, radius, frontier.to_action_dict(),
        config=FixedResourceRefinementConfig.from_selection_policy(config),
    )
    use_warm = warm.improved and int(warm.action.get("resource", 0)) >= target_resource and warm.action.get("work", math.inf) <= frontier.work
    data = dict(warm.action if use_warm else frontier.to_action_dict())
    merged_calls = _merge_counters(frontier.operator_calls, warm.counters)
    warm_evaluations = int(warm.counters.get("endpoint_replays", 0))
    data.update(selection_policy=config.to_json(), requested_resource=requested_resource, frontier_resource=frontier.resource,
                growth_termination_reason=frontier.stage,
                growth_computation_exhausted=frontier.stage == 'evaluation_budget_exhausted',
                frontier_work=frontier.work, frontier_energy=frontier.energy,
                frontier_target_resource=target_resource, total_seconds=time.perf_counter() - started,
                total_evaluations=int(frontier.evaluations + warm_evaluations),
                total_operator_calls=merged_calls)
    data.update(refinement_endpoint_budget_exhausted=warm.endpoint_budget_exhausted,
                refinement_search_scope=warm.search_scope,
                refinement_screened_endpoints=warm.screened_endpoints,
                refinement_unattempted_endpoints=warm.unattempted_endpoints,
                refinement_unvisited_pairs=warm.unvisited_pairs,
                refinement_exchanges=warm.exchanges,
                refinement_exchange_budget_exhausted=warm.exchange_budget_exhausted,
                refinement_search_exhausted=warm.search_exhausted,
                refinement_ranking=warm.action.get('refinement_ranking'))
    for name in ('refinement_termination_reason', 'refinement_neighborhood_exhausted',
                 'refinement_next_neighborhood_unopened', 'refinement_last_search_epoch',
                 'refinement_last_search_is_returned_action', 'refinement_total_screened_endpoints',
                 'refinement_descent_history'):
        data[name] = warm.action.get(name)
    data["authority"] = "proposal_only_observed_chart"
    if use_warm:
        data.update(stage="fixed_resource_refined_at_frontier", least_work_status=warm.status,
                    fixed_resource_refinement_authority=warm.authority)
    else:
        data.update(stage="frontier_only_fixed_resource_no_improvement", least_work_status=warm.status,
                    fixed_resource_refinement_authority=warm.authority)
    data["least_work_candidate"] = dict(resource=int(warm.action.get("resource", frontier.resource)),
                                         work=warm.action.get("work"), energy=warm.action.get("energy"),
                                         status=warm.status, evaluations=warm_evaluations,
                                         operator_calls=dict(warm.counters))
    data["fixed_resource_candidate"] = dict(resource=int(warm.action.get("resource", frontier.resource)),
                                             work=warm.action.get("work"), energy=warm.action.get("energy"),
                                             status=warm.status, improved=bool(warm.improved),
                                             evaluations=warm_evaluations, operator_calls=dict(warm.counters))
    data["frontier_candidate"] = dict(resource=frontier.resource, work=frontier.work, energy=frontier.energy,
                                       status=frontier.stage, evaluations=frontier.evaluations,
                                       operator_calls=dict(frontier.operator_calls))
    data["evaluations"] = int(frontier.evaluations + warm_evaluations)
    data["history"] = (frontier.history + warm.history)[-64:]
    return data
