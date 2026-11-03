"""Fixed-resource warm refinement for tractable-v1 controller proposals.

This module refines an already feasible frontier binary action by same-resource
exchanges. The default lazily traverses the complete eligible Cartesian domain
in surrogate work order, then screens exact finite F/Q and replays full actions.
Explicit count limits retain the earlier finite-screen diagnostic. Accepted
exchanges also preserve captured frozen-input ReLU phases by
default. This does not certify joint multilayer or pooling behavior. It does
not claim global binary optimality, population authority,
bootstrap authority, C/K/O authority, nonlinear phase validity, finite-dose
validity, or production admission.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections import deque, OrderedDict
from typing import Any, Mapping, Sequence
import math
import time
from itertools import islice
from contextlib import closing

import torch
import numpy as np

from .operators import ChartOperators
from .chart_safeguards import local_action_kinks
from .selection import (
    CandidateCellRows,
    CandidateRowWorkspacePlan,
    _binary_tensor,
    _candidate_tensor,
    _chunk_shape,
    _positive_int,
    candidate_corrected_cell_rows,
    forecast_candidate_row_workspace,
    singleton_finite_work_screen,
    validate_selection_inputs,
)
from .solvers import DiagonalShrunkGeometry
from .selection_pair_stream import RankedPairPage, RankedPairSpool
from .selection_exchange_order import OrderedExchangeStream
from .selection_pair_batch import reference_screen_rejections


@dataclass(frozen=True)
class FixedResourceRefinementConfig:
    """Memory settings and explicit diagnostic bounds for fixed-resource descent."""

    # None traverses the complete eligible single-exchange domain lazily.
    # Explicit screening counts select the retained finite-screen diagnostic.
    max_drop_candidates: int | None = None
    max_add_candidates: int | None = None
    max_pair_evaluations: int | None = None
    # None exhausts the finite screened list until a strict improvement is
    # accepted. Positive integers are explicit diagnostic computation budgets.
    max_endpoint_replays: int | None = None
    max_exchanges: int | None = None
    scan_chunk_routes: int = 1 << 20
    candidate_workspace_bytes: int = 64 << 20
    candidate_retained_bytes: int = 256 << 20
    pair_workspace_bytes: int = 64 << 20
    rerank_blocks: bool = True
    require_local_phase_stability: bool = True
    phase_workspace_bytes: int = 256 << 20

    def __post_init__(self) -> None:
        for name in (
            "scan_chunk_routes",
            "candidate_workspace_bytes",
            "candidate_retained_bytes",
            "pair_workspace_bytes",
            "phase_workspace_bytes",
        ):
            _positive_int(name, getattr(self, name))
        for name in ('max_drop_candidates', 'max_add_candidates', 'max_pair_evaluations'):
            if getattr(self, name) is not None:
                _positive_int(name, getattr(self, name))
        if self.max_endpoint_replays is not None:
            _positive_int("max_endpoint_replays", self.max_endpoint_replays)
        if self.max_exchanges is not None:
            _positive_int('max_exchanges', self.max_exchanges)
        if type(self.rerank_blocks) is not bool:
            raise ValueError("rerank_blocks must be a bool")
        if type(self.require_local_phase_stability) is not bool:
            raise ValueError("require_local_phase_stability must be a bool")
        if self.phase_workspace_bytes < 1024:
            raise ValueError("phase_workspace_bytes must be at least 1024")

    @classmethod
    def from_selection_policy(cls, policy: Any) -> "FixedResourceRefinementConfig":
        # Queue memory settings do not truncate the refinement domain. An
        # explicit diagnostic evaluation budget limits replay, not eligibility.
        replay_limit = policy.max_evaluations
        return cls(
            max_endpoint_replays=replay_limit,
            max_exchanges=policy.max_refinement_exchanges,
            scan_chunk_routes=int(policy.scan_chunk_routes),
            candidate_workspace_bytes=int(policy.candidate_workspace_bytes),
            candidate_retained_bytes=int(policy.candidate_retained_bytes),
            pair_workspace_bytes=int(policy.candidate_workspace_bytes),
            rerank_blocks=bool(policy.rerank_blocks),
            require_local_phase_stability=policy.require_local_phase_stability,
            phase_workspace_bytes=policy.phase_workspace_bytes,
        )

    def to_json(self) -> dict[str, Any]:
        return dict(schema="tractable_v1_fixed_resource_refinement_config_v6", **self.__dict__)


@dataclass(frozen=True)
class CandidateResponseRows:
    candidates: torch.Tensor
    values: torch.Tensor
    workspace_plan: CandidateRowWorkspacePlan
    law: str = "tractable_v1_candidate_literal_j_response_rows_v2"


@dataclass(frozen=True)
class FixedResourceRefinementResult:
    action: dict[str, Any]
    improved: bool
    status: str
    counters: dict[str, int]
    seconds: float
    history: list[dict[str, Any]]
    endpoint_budget_exhausted: bool = False
    screened_endpoints: int = 0
    unattempted_endpoints: int = 0
    search_scope: str = "bounded_screened_pair_list_at_frontier"
    authority: str = "proposal_only_observed_chart_fixed_resource_refinement"
    optimality_authority: bool = False
    population_certified: bool = False
    bootstrap_certified: bool = False
    production_commit_authority: bool = False
    unvisited_pairs: int = 0
    exchanges: int = 0
    exchange_budget_exhausted: bool = False
    search_exhausted: bool = False


@dataclass(frozen=True)
class _CompactPair:
    work: float
    energy: float
    drop_position: int
    add_route: int
    drop_route: int

    @property
    def sort_key(self) -> tuple[float, float, int, int, int]:
        return (self.work, self.energy, self.drop_position, self.add_route, self.drop_route)


@dataclass(frozen=True)
class _SurvivorState:
    parent_live: torch.Tensor
    survivor_counts: torch.Tensor
    parent_counts: torch.Tensor
    fiber_of_route: torch.Tensor


def _new_counters() -> dict[str, int]:
    return dict(
        a_forward=0,
        a_adjoint=0,
        j_forward=0,
        geom_energy=0,
        geom_cross=0,
        work_weight_reads=0,
        singleton_screens=0,
        candidate_a_row_batches=0,
        candidate_a_rows=0,
        candidate_j_row_batches=0,
        candidate_j_rows=0,
        pair_tiles=0,
        pair_local_evaluations=0,
        endpoint_replays=0,
    )


def merge_operator_counters(*items: Mapping[str, int]) -> dict[str, int]:
    keys: set[str] = set()
    for item in items:
        keys.update(item)
    return {key: (max if key.endswith("_peak_bytes") else sum)(int(item.get(key, 0)) for item in items)
            for key in sorted(keys)}


def _require_chart(chart: Mapping[str, Any]) -> tuple[ChartOperators, DiagonalShrunkGeometry, torch.Tensor, torch.Tensor]:
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


def _positive_radius(radius: float) -> float:
    r = float(radius)
    if not math.isfinite(r) or r <= 0.0:
        raise ValueError("radius must be finite and positive")
    return r


def _frontier_z(ops: ChartOperators, frontier_action: Mapping[str, Any]) -> torch.Tensor:
    if "z" not in frontier_action:
        raise ValueError("frontier action missing z")
    z = _binary_tensor("frontier z", torch.as_tensor(frontier_action["z"]), ops.route_count,
                       device=ops.device, dtype=ops.dtype)
    resource = int(z.sum().item())
    if int(frontier_action.get("resource", resource)) != resource:
        raise ValueError("frontier resource does not match z")
    return z


def _frontier_order(ops: ChartOperators, frontier_action: Mapping[str, Any], z: torch.Tensor) -> torch.Tensor:
    selected_routes = torch.nonzero(z > 0, as_tuple=False).reshape(-1)
    if "order" not in frontier_action:
        return selected_routes
    order = _candidate_tensor(torch.as_tensor(frontier_action["order"]), ops.route_count, device=ops.device)
    if order.numel() < selected_routes.numel():
        raise ValueError("frontier order is shorter than resource")
    selected_prefix = order[:selected_routes.numel()]
    if selected_prefix.numel() and not bool((z[selected_prefix] > 0).all()):
        raise ValueError("frontier order prefix does not match selected z")
    if int(selected_prefix.unique().numel()) != int(selected_routes.numel()):
        raise ValueError("frontier order prefix must list each selected route exactly once")
    if selected_routes.numel() and not bool(torch.equal(torch.sort(selected_prefix).values, torch.sort(selected_routes).values)):
        raise ValueError("frontier order prefix does not equal selected z")
    return selected_prefix


def _endpoint(ops: ChartOperators, geom: DiagonalShrunkGeometry, g: torch.Tensor, z: torch.Tensor,
              counters: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    counters["geom_energy"] += 1
    counters["a_j_forward"] = counters.get("a_j_forward", 0) + 1
    cells, response = ops.a_j_forward(z)
    energy = float(geom.energy_from_response(z, response).item())
    counters["work_weight_reads"] += 1
    omega = ops.work_weights()
    work = float((omega * cells.abs()).sum().item())
    first_order = float((ops.context_weights * g.to(ops.dtype) * cells.sum(dim=1)).sum().item())
    stats = dict(resource=int(z.sum().item()), energy=energy, half_energy=0.5 * energy, work=work,
                 first_order=first_order, predicted_damage=first_order + 0.5 * energy)
    if not all(math.isfinite(float(v)) for v in stats.values()):
        raise ValueError("endpoint replay produced nonfinite stats")
    return cells, response, stats


def _survivor_state(ops: ChartOperators, z: torch.Tensor, parent_live: torch.Tensor) -> _SurvivorState:
    fibers = ops.fiber_of_route()
    live_after = (parent_live > 0) & (z <= 0)
    parent_counts = torch.bincount(fibers[parent_live > 0], minlength=ops.fiber_count).to(torch.long)
    survivor_counts = torch.bincount(fibers[live_after], minlength=ops.fiber_count).to(torch.long)
    if bool((survivor_counts[parent_counts > 0] <= 0).any()):
        raise ValueError("frontier action leaves an empty physical survivor fiber")
    return _SurvivorState(parent_live=parent_live, survivor_counts=survivor_counts,
                          parent_counts=parent_counts, fiber_of_route=fibers)


def _partial_smallest(primary: torch.Tensor, mask: torch.Tensor, *, k: int, chunk_routes: int) -> torch.Tensor:
    from .selection_policy import _partial_smallest_by_primary

    return _partial_smallest_by_primary(primary, mask, k=k, chunk_routes=chunk_routes)


def _screening_keys(ops: ChartOperators, base_cells: torch.Tensor,
                    config: FixedResourceRefinementConfig,
                    counters: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    if config.rerank_blocks:
        counters["work_weight_reads"] += 1
        counters["a_adjoint"] += 1
        subgrad = ops.a_adjoint(ops.work_weights() * torch.sign(base_cells))
        return subgrad, -subgrad
    counters["singleton_screens"] += 1
    singleton = singleton_finite_work_screen(ops)
    return singleton, -singleton


def _add_candidate_mask(ops: ChartOperators, z: torch.Tensor, parent_live: torch.Tensor,
                        drop_queue: torch.Tensor, survivor: _SurvivorState) -> torch.Tensor:
    drop_fibers = torch.zeros(ops.fiber_count, device=ops.device, dtype=torch.bool)
    if drop_queue.numel():
        drop_fibers[survivor.fiber_of_route[drop_queue]] = True
    add_fibers = survivor.fiber_of_route
    legal_for_some_drop = (survivor.survivor_counts[add_fibers] > 1) | drop_fibers[add_fibers]
    return ((parent_live > 0) & (z <= 0) & legal_for_some_drop).to(ops.dtype)


def candidate_response_rows(ops: ChartOperators, candidates: torch.Tensor | Sequence[int], *,
                            max_candidates: int, workspace_bytes: int, max_retained_bytes: int,
                            candidate_eligible: torch.Tensor | None = None) -> CandidateResponseRows:
    """Build literal-J response coefficients only for requested parent-live routes."""

    _zz, cand_mask = validate_selection_inputs(ops, torch.zeros(ops.route_count, device=ops.device, dtype=ops.dtype),
                                               candidate_eligible)
    cand = _candidate_tensor(candidates, ops.route_count, device=ops.device)
    if cand.numel() and bool((cand_mask[cand] <= 0).any()):
        raise ValueError("response rows include routes outside candidate_eligible")
    plan = forecast_candidate_row_workspace(ops, cand, max_candidates=max_candidates,
                                            workspace_bytes=workspace_bytes,
                                            max_retained_bytes=max_retained_bytes)
    out = torch.empty((ops.context_count, cand.numel()), device=ops.device, dtype=ops.dtype)
    if cand.numel() == 0:
        return CandidateResponseRows(candidates=cand, values=out, workspace_plan=plan)
    elem = out.element_size()
    for layer, axis_layer in zip(ops.factors.layers, ops.axis.layers, strict=True):
        lo = axis_layer.route_offset
        hi = lo + axis_layer.route_count
        sel = torch.nonzero((cand >= lo) & (cand < hi), as_tuple=False).reshape(-1)
        if sel.numel() == 0:
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
            for cs in range(0, ops.context_count, context_chunk):
                ce = min(ops.context_count, cs + context_chunk)
                prod = layer.D[cs:ce][:, f_local, :] * layer.X[cs:ce][:, j_local, :]
                out[cs:ce, pos] = -wm[None, :] * prod.sum(dim=2)
                del prod
    if not bool(torch.isfinite(out).all()):
        raise ValueError("response row values must be finite")
    return CandidateResponseRows(candidates=cand, values=out, workspace_plan=plan)


def _row_positions(routes: torch.Tensor, wanted: torch.Tensor) -> torch.Tensor:
    sorted_routes, perm = torch.sort(routes)
    pos = torch.searchsorted(sorted_routes, wanted)
    if bool((pos >= sorted_routes.numel()).any()) or not bool(torch.equal(sorted_routes[pos], wanted)):
        raise ValueError("candidate rows missing requested route")
    return perm[pos]


def _replace_order(order: torch.Tensor, drop: int, add: int) -> torch.Tensor:
    kept = order[order != int(drop)]
    return torch.cat([kept, torch.tensor([int(add)], device=order.device, dtype=torch.long)])


def _pair_tile_shape(contexts: int, drop_count: int, add_count: int, elem: int, workspace_bytes: int) -> tuple[int, int]:
    # Principal pair temporaries include several [B,D,A] arrays for same-fiber work and Q deltas.
    per_pair = max(1, int(contexts)) * int(elem) * 12
    if int(workspace_bytes) < per_pair:
        raise ValueError("pair_workspace_bytes cannot hold one full-context candidate pair")
    capacity = int(workspace_bytes) // per_pair
    d = max(1, min(int(drop_count), int(math.sqrt(capacity))))
    a = max(1, min(int(add_count), max(1, capacity // d)))
    return d, a


def _row_pair_bytes(ops: ChartOperators, count: int) -> int:
    # Both retained row plans, including their conservative index allowances.
    return 2 * int(count) * (ops.context_count * torch.empty((), dtype=ops.dtype).element_size() + 16)


def _candidate_row_pair(ops: ChartOperators, candidates: torch.Tensor,
                        config: FixedResourceRefinementConfig, counters: dict[str, int], *,
                        candidate_eligible: torch.Tensor) -> tuple[CandidateCellRows, CandidateResponseRows]:
    count = int(candidates.numel())
    if _row_pair_bytes(ops, count) > config.candidate_retained_bytes:
        raise ValueError('combined A/J candidate rows exceed retained byte budget')
    limits = dict(max_candidates=count, workspace_bytes=config.candidate_workspace_bytes,
                  max_retained_bytes=config.candidate_retained_bytes, candidate_eligible=candidate_eligible)
    a_rows = candidate_corrected_cell_rows(ops, candidates, **limits)
    j_rows = candidate_response_rows(ops, candidates, **limits)
    counters['candidate_a_row_batches'] += 1
    counters['candidate_a_rows'] += count
    counters['candidate_j_row_batches'] += 1
    counters['candidate_j_rows'] += count
    return a_rows, j_rows


def _compact_pairs(ops: ChartOperators, geom: DiagonalShrunkGeometry, base_cells: torch.Tensor,
                   base_response: torch.Tensor, base_work: float, base_energy: float,
                   a_rows: CandidateCellRows | None, j_rows: CandidateResponseRows | None, drop_queue: torch.Tensor,
                   add_queue: torch.Tensor, drop_positions: torch.Tensor, survivor: _SurvivorState,
                   radius: float, config: FixedResourceRefinementConfig,
                   counters: dict[str, int], *, page: RankedPairPage | RankedPairSpool | None = None,
                   row_provider=None) -> list[_CompactPair]:
    if drop_queue.numel() == 0 or add_queue.numel() == 0:
        return []
    if (a_rows is None) != (j_rows is None):
        raise ValueError('A/J candidate rows must both be supplied or both streamed')
    streamed = a_rows is None
    if not streamed and a_rows.values.shape != (ops.context_count, a_rows.candidates.numel()):
        raise ValueError("A candidate rows have incompatible shape")
    if not streamed and j_rows.values.shape != (ops.context_count, j_rows.candidates.numel()):
        raise ValueError("J candidate rows have incompatible shape")
    if not streamed and not bool(torch.equal(a_rows.candidates, j_rows.candidates)):
        raise ValueError("A/J candidate rows must share candidate coordinates")
    elem = base_cells.element_size()
    drop_tile, add_tile = _pair_tile_shape(ops.context_count, int(drop_queue.numel()), int(add_queue.numel()),
                                           elem, config.pair_workspace_bytes)
    if streamed:
        peak_rows = _row_pair_bytes(ops, drop_tile + add_tile)
        if peak_rows > config.candidate_retained_bytes:
            raise ValueError('pair tile A/J rows exceed retained byte budget; reduce pair workspace')
        counters['candidate_rows_peak_bytes'] = max(counters.get('candidate_rows_peak_bytes', 0), peak_rows)
        counters['candidate_row_streamed_scans'] = counters.get('candidate_row_streamed_scans', 0) + 1
    omega = ops.work_weights()
    counters["work_weight_reads"] += 1
    if streamed:
        drop_fibers_all = survivor.fiber_of_route[drop_queue]
        add_fibers_all = survivor.fiber_of_route[add_queue]
    else:
        a_dpos = _row_positions(a_rows.candidates, drop_queue)
        a_apos = _row_positions(a_rows.candidates, add_queue)
        j_dpos = _row_positions(j_rows.candidates, drop_queue)
        j_apos = _row_positions(j_rows.candidates, add_queue)
        drop_fibers_all = a_rows.fiber_ids[a_dpos].to(device=ops.device, dtype=torch.long)
        add_fibers_all = a_rows.fiber_ids[a_apos].to(device=ops.device, dtype=torch.long)
    drop_pos_cpu = drop_positions.detach().cpu()
    out: list[_CompactPair] = []
    max_pairs = (int(drop_queue.numel()) * int(add_queue.numel()) if config.max_pair_evaluations is None
                 else int(config.max_pair_evaluations))
    row_builder = _candidate_row_pair if row_provider is None else row_provider
    add_total = int(add_queue.numel())
    for ds in range(0, int(drop_queue.numel()), drop_tile):
        if ds * add_total >= max_pairs:
            break
        de = min(int(drop_queue.numel()), ds + drop_tile)
        df = drop_fibers_all[ds:de]
        if streamed:
            drop_a, drop_j = row_builder(ops, drop_queue[ds:de], config, counters,
                                                candidate_eligible=survivor.parent_live)
            dvals, jd = drop_a.values, drop_j.values
        else:
            dvals = a_rows.values[:, a_dpos[ds:de]]
            jd = j_rows.values[:, j_dpos[ds:de]]
        db = base_cells[:, df]
        dw = omega[:, df]
        for astart in range(0, int(add_queue.numel()), add_tile):
            ae = min(int(add_queue.numel()), astart + add_tile)
            pair_start = ds * add_total + astart
            if pair_start >= max_pairs:
                break
            af = add_fibers_all[astart:ae]
            if streamed:
                add_a, add_j = row_builder(ops, add_queue[astart:ae], config, counters,
                                                  candidate_eligible=survivor.parent_live)
                avals, ja = add_a.values, add_j.values
            else:
                avals = a_rows.values[:, a_apos[astart:ae]]
                ja = j_rows.values[:, j_apos[astart:ae]]
            ab = base_cells[:, af]
            aw = omega[:, af]
            d_marg = (dw * ((db - dvals).abs() - db.abs())).sum(dim=0)[:, None]
            a_marg = (aw * ((ab + avals).abs() - ab.abs())).sum(dim=0)[None, :]
            diff_work = d_marg + a_marg
            same_base = db[:, :, None]
            same = df[:, None] == af[None, :]
            same_work = (dw[:, :, None] * ((same_base - dvals[:, :, None] + avals[:, None, :]).abs() - same_base.abs())).sum(dim=0)
            work = base_work + torch.where(same, same_work, diff_work)
            delta = -jd[:, :, None] + ja[:, None, :]
            interference = (geom.weights[:, None, None] * ((base_response[:, None, None] + delta) ** 2 -
                                                           base_response[:, None, None] ** 2)).sum(dim=0)
            diag_delta = -geom.diagonal[drop_queue[ds:de]][:, None] + geom.diagonal[add_queue[astart:ae]][None, :]
            energy = base_energy + geom.total_mass * ((1.0 - geom.alpha) * interference + geom.alpha * diag_delta)
            pair_numbers = (torch.arange(ds, de, device=ops.device)[:, None] * add_total +
                            torch.arange(astart, ae, device=ops.device)[None, :])
            add_legal = survivor.survivor_counts[af] > 1
            legal = same | add_legal[None, :]
            within_cap = pair_numbers < max_pairs
            finite = torch.isfinite(work) & torch.isfinite(energy)
            useful = work < float(base_work)
            feasible = 0.5 * energy <= radius
            keep = legal & within_cap & finite & useful & feasible
            counters["pair_local_evaluations"] += int(within_cap.sum().item())
            counters["pair_legal_evaluations"] = counters.get("pair_legal_evaluations", 0) + int((within_cap & legal).sum().item())
            counters["pair_tiles"] += 1
            if bool(keep.any()):
                coords = torch.nonzero(keep, as_tuple=False).detach().cpu()
                work_cpu = work.detach().cpu()
                energy_cpu = energy.detach().cpu()
                drop_cpu = drop_queue[ds:de].detach().cpu()
                add_cpu = add_queue[astart:ae].detach().cpu()
                for local_d, local_a in coords.tolist():
                    item = _CompactPair(
                        work=float(work_cpu[local_d, local_a].item()),
                        energy=float(energy_cpu[local_d, local_a].item()),
                        drop_position=int(drop_pos_cpu[ds + local_d].item()),
                        add_route=int(add_cpu[local_a].item()),
                        drop_route=int(drop_cpu[local_d].item()),
                    )
                    if page is None:
                        out.append(item)
                    else:
                        page.offer(item.sort_key)
            del d_marg, a_marg, diff_work, same_work, delta, interference, diag_delta, energy, work
            if streamed:
                del avals, ja, add_a, add_j
        if streamed:
            del dvals, jd, drop_a, drop_j
    out.sort(key=lambda item: item.sort_key)
    return out


class _CompactPairStream:
    """Single-screen ranked delivery, with private bounded spill when needed."""
    def __init__(self, *args):
        self._counters = args[-1]
        config = args[-2]
        maximum = args[8].numel() * args[9].numel()
        if config.max_pair_evaluations is not None:
            maximum = min(config.max_pair_evaluations, maximum)
        self._spool = RankedPairSpool(payload_bytes=config.pair_workspace_bytes, maximum_pairs=maximum)
        self._consumed = False
        self._closed = False
        self._iterator = None
        try:
            _compact_pairs(*args, page=self._spool)
            self._spool.finish()
            self.screened_endpoints = self._spool.seen
            self._update_counters()
        except BaseException:
            self.close()
            raise

    def _update_counters(self):
        self._counters.update(pair_scan_passes=1,
            pair_output_peak_bytes=self._spool.payload_peak_bytes,
            pair_spill_runs=self._spool.run_count, pair_spill_merges=self._spool.merge_count,
            pair_spill_bytes_written=self._spool.bytes_written,
            pair_spill_bytes_read=self._spool.bytes_read,
            pair_scratch_peak_bytes=self._spool.scratch_peak_bytes)

    def __iter__(self):
        if self._consumed or self._closed:
            raise ValueError('pair stream is single-use')
        self._consumed = True
        self._iterator = self._spool.keys()
        for key in self._iterator:
            self._update_counters()
            yield _CompactPair(work=key[0], energy=key[1], drop_position=key[2],
                               add_route=key[3], drop_route=key[4])

    def close(self):
        if self._closed:
            return
        if self._iterator is not None:
            self._iterator.close()
        self._update_counters()
        self._spool.close()
        self._closed = True


class _ProgressivePairStream:
    """Complete surrogate order; exact finite screen and replay remain separate."""
    search_scope = 'complete_eligible_single_exchange_domain'

    def __init__(self, *args, drop_primary, add_primary):
        self._args = args
        ops, config = args[0], args[-2]
        drops, adds, positions = args[8:11]
        self._order = OrderedExchangeStream(
            drop_routes=drops.detach().cpu().numpy(), drop_positions=positions.detach().cpu().numpy(),
            drop_priorities=drop_primary[drops].detach().cpu().numpy(),
            add_routes=adds.detach().cpu().numpy(), add_priorities=add_primary[adds].detach().cpu().numpy())
        self._capacity = config.candidate_retained_bytes // _row_pair_bytes(ops, 1)
        if self._capacity < 2:
            self._order.close()
            raise ValueError('pair tile A/J rows exceed retained byte budget; two cached rows required')
        self._rows = OrderedDict()
        self._counters = args[-1]
        self._counters['exchange_order_peak_bytes'] = self._order.retained_payload_bytes
        self._counters['exchange_domain_pairs'] = self._order.domain_size
        self.screened_endpoints = 0
        self._iterator = None
        self._closed = False
        self._visited = 0

    @property
    def unvisited_pairs(self):
        return self._order.domain_size - self._visited

    def _row_provider(self, ops, candidates, config, counters, *, candidate_eligible):
        if candidates.numel() != 1:
            raise ValueError('progressive row cache requires singleton coordinates')
        route = int(candidates.item())
        return self._row_for_id(ops, route, config, counters, candidate_eligible=candidate_eligible)

    def _row_for_id(self, ops, route, config, counters, *, candidate_eligible):
        if route in self._rows:
            counters['candidate_row_cache_hits'] = counters.get('candidate_row_cache_hits', 0) + 1
            self._rows.move_to_end(route)
            return self._rows[route]
        if len(self._rows) == self._capacity:
            self._rows.popitem(last=False)
        candidates = torch.tensor([route], device=ops.device, dtype=torch.long)
        result = _candidate_row_pair(ops, candidates, config, counters, candidate_eligible=candidate_eligible)
        self._rows[route] = result
        counters['candidate_rows_peak_bytes'] = max(counters.get('candidate_rows_peak_bytes', 0),
                                                     _row_pair_bytes(ops, len(self._rows)))
        return result

    def _batch_rejections(self, records):
        ops, geom, cells, response, work, energy = self._args[:6]
        survivor, radius, config, counters = self._args[11:]
        routes = list(dict.fromkeys(int(r) for r in records[:, :2].reshape(-1)))
        if len(routes) > self._capacity:
            raise ValueError('pair batch exceeds protected row-cache capacity')
        for route in routes:
            self._row_for_id(ops, route, config, counters, candidate_eligible=survivor.parent_live)
        # At most capacity distinct requested rows: LRU insertion evicts only
        # older non-batch rows. Stack/gather copies belong to pair workspace.
        a_bank = torch.stack([self._rows[r][0].values[:, 0] for r in routes])
        j_bank = torch.stack([self._rows[r][1].values[:, 0] for r in routes])
        lookup = {r:i for i,r in enumerate(routes)}
        di = torch.tensor([lookup[int(r)] for r in records[:, 0]], device=ops.device)
        ai = torch.tensor([lookup[int(r)] for r in records[:, 1]], device=ops.device)
        dvals, avals, jd, ja = a_bank[di], a_bank[ai], j_bank[di], j_bank[ai]
        del a_bank, j_bank
        drops = torch.as_tensor(records[:, 0].copy(), device=ops.device)
        adds = torch.as_tensor(records[:, 1].copy(), device=ops.device)
        df, af = survivor.fiber_of_route[drops], survivor.fiber_of_route[adds]
        omega = ops.work_weights()
        counters['work_weight_reads'] += 1
        same = df == af
        rejected = reference_screen_rejections(dvals=dvals, avals=avals, jd=jd, ja=ja,
            db=cells[:, df].T.contiguous(), ab=cells[:, af].T.contiguous(),
            dw=omega[:, df].T.contiguous(), aw=omega[:, af].T.contiguous(),
            response=response[None, :], weights=geom.weights[None, :],
            diag_delta=-geom.diagonal[drops]+geom.diagonal[adds], same=same,
            legal=same | (survivor.survivor_counts[af] > 1),
            base_work=work, base_energy=energy, alpha=geom.alpha,
            total_mass=geom.total_mass, radius=radius)
        counters['pair_batch_evaluations'] = counters.get('pair_batch_evaluations', 0)+len(records)
        counters['pair_batches'] = counters.get('pair_batches', 0)+1
        return rejected.detach().cpu().numpy()

    def _generate(self):
        ops, geom, cells, response, work, energy = self._args[:6]
        survivor, radius, config, counters = self._args[11:]
        per_pair = 32*ops.context_count*cells.element_size()+128
        max_batch = max(1, min(self._capacity//2, config.pair_workspace_bytes//per_pair))
        if ops.dtype not in (torch.float32, torch.float64) or ops.device.type not in ('cpu', 'cuda'):
            max_batch = 1
        batch_size = 1
        while self._visited < self._order.domain_size:
            capacity = min(batch_size, self._order.domain_size-self._order.visited)
            records = np.empty((capacity, 3), dtype=np.int64)
            count = 0
            for pair in islice(self._order, capacity):
                records[count] = (pair.drop_route, pair.add_route, pair.drop_position)
                count += 1
            if not count:
                break
            records = records[:count]
            counters['exchange_pairs_prefetched'] = self._order.visited
            counters['exchange_order_comparisons'] = self._order.comparisons
            if count > 1:
                counters['pair_batch_workspace_peak_bytes'] = max(
                    counters.get('pair_batch_workspace_peak_bytes', 0), capacity*per_pair)
                rejected = self._batch_rejections(records)
            else:
                rejected = np.zeros(count, dtype=bool)
            for (drop, add, position), skip in zip(records, rejected, strict=True):
                self._visited += 1
                counters['exchange_pairs_visited'] = self._visited
                if skip:
                    counters['pair_batch_scalar_skips'] = counters.get('pair_batch_scalar_skips', 0)+1
                    continue
                drops = torch.tensor([int(drop)], device=ops.device, dtype=torch.long)
                adds = torch.tensor([int(add)], device=ops.device, dtype=torch.long)
                positions = torch.tensor([int(position)], device=ops.device, dtype=torch.long)
                counters['pair_scalar_reference_screens'] = counters.get('pair_scalar_reference_screens', 0)+1
                items = _compact_pairs(ops, geom, cells, response, work, energy, None, None,
                    drops, adds, positions, survivor, radius, config, counters, row_provider=self._row_provider)
                if items:
                    self.screened_endpoints += 1
                    yield items[0]
            del records, rejected
            batch_size = min(max_batch, batch_size*2)

    def __iter__(self):
        if self._iterator is not None or self._closed:
            raise ValueError('pair stream is single-use')
        self._iterator = self._generate()
        return self._iterator

    def close(self):
        if self._closed:
            return
        if self._iterator is not None:
            self._iterator.close()
        self._rows.clear()
        self._order.close()
        self._closed = True


def _refine_frontier_exchange(chart: Mapping[str, Any], radius: float, frontier_action: Mapping[str, Any], *,
                                   config: FixedResourceRefinementConfig = FixedResourceRefinementConfig()
                                   ) -> FixedResourceRefinementResult:
    """Attempt one strict same-resource improvement from a feasible frontier."""

    started = time.perf_counter()
    counters = _new_counters()
    ops, geom, eligible, g = _require_chart(chart)
    r = _positive_radius(radius)
    z = _frontier_z(ops, frontier_action)
    order = _frontier_order(ops, frontier_action, z)
    resource = int(z.sum().item())
    history = deque(maxlen=64)
    progressive = all(getattr(config, name) is None for name in
                      ('max_drop_candidates', 'max_add_candidates', 'max_pair_evaluations'))
    search_scope = ('complete_eligible_single_exchange_domain' if progressive
                    else 'bounded_screened_pair_list_at_frontier')
    ranking = (('work_subgradient' if config.rerank_blocks else 'singleton_work')
               + '_cartesian_first_improvement' if progressive else 'finite_work_sorted_bounded_screen')
    if resource <= 0:
        action = dict(frontier_action, stage="fixed_resource_no_frontier_resource",
                      least_work_status="fixed_resource_no_frontier_resource",
                      refinement_replayed_frontier_work=0., refinement_replayed_frontier_energy=0.)
        return FixedResourceRefinementResult(action=action, improved=False, status="fixed_resource_no_frontier_resource",
                                             counters=counters, seconds=time.perf_counter() - started, history=list(history),
                                             search_scope=search_scope)
    base_cells, base_response, base_stats = _endpoint(ops, geom, g, z, counters)
    replayed_frontier = dict(refinement_replayed_frontier_work=base_stats['work'],
                             refinement_replayed_frontier_energy=base_stats['energy'])
    if base_stats["half_energy"] > r:
        raise ValueError("frontier action is not feasible under the supplied radius")
    physical_parent = _binary_tensor("ops.eligible()", ops.eligible(), ops.route_count,
                                     device=ops.device, dtype=ops.dtype)
    chart_candidate = _binary_tensor("chart['eligible']", torch.as_tensor(eligible), ops.route_count,
                                     device=ops.device, dtype=ops.dtype)
    validate_selection_inputs(ops, z)
    if bool(((chart_candidate > 0) & (physical_parent <= 0)).any()):
        raise ValueError("chart eligible includes routes outside the physical parent")
    if bool(((z > 0) & (chart_candidate <= 0)).any()):
        raise ValueError("frontier action selects a route outside chart eligibility")
    survivor = _survivor_state(ops, z, physical_parent)
    if config.require_local_phase_stability:
        phase = local_action_kinks(ops, z, workspace_bytes=config.phase_workspace_bytes)
        counters["local_phase_replays"] = counters.get("local_phase_replays", 0) + 1
        if not phase["local_phase_unchanged"]:
            raise ValueError("frontier action crosses a captured frozen-input local phase")
    add_primary, drop_primary = _screening_keys(ops, base_cells, config, counters)
    selected_mask = z.to(dtype=ops.dtype)
    drop_limit = min(resource, config.max_drop_candidates or resource, config.max_pair_evaluations or resource)
    # The lazy stream sorts its frozen queues once on the host. Do not first
    # sort the complete queues on the operator device as well.
    drop_queue = (torch.nonzero(selected_mask > 0, as_tuple=False).reshape(-1) if progressive else
                  _partial_smallest(drop_primary, selected_mask, k=drop_limit, chunk_routes=config.scan_chunk_routes))
    if drop_queue.numel() <= 0:
        action = dict(frontier_action, stage="fixed_resource_no_exchange_candidates",
                      least_work_status="fixed_resource_no_exchange_candidates", **replayed_frontier)
        return FixedResourceRefinementResult(action=action, improved=False, status="fixed_resource_no_exchange_candidates",
                                             counters=counters, seconds=time.perf_counter() - started, history=list(history),
                                             search_scope=search_scope)
    order_rank = torch.full((ops.route_count,), ops.route_count, device=ops.device, dtype=torch.long)
    order_rank[order] = torch.arange(order.numel(), device=ops.device, dtype=torch.long)
    drop_positions = order_rank[drop_queue]
    add_mask = _add_candidate_mask(ops, z, chart_candidate, drop_queue, survivor)
    add_available = int(add_mask.sum().item())
    add_limit = min(config.max_add_candidates or add_available,
                    max(1, config.max_pair_evaluations // max(1, int(drop_queue.numel())))
                    if config.max_pair_evaluations is not None else add_available)
    add_queue = (torch.nonzero(add_mask > 0, as_tuple=False).reshape(-1) if progressive else
                 _partial_smallest(add_primary, add_mask, k=add_limit, chunk_routes=config.scan_chunk_routes))
    if add_queue.numel() <= 0:
        action = dict(frontier_action, stage="fixed_resource_no_exchange_candidates",
                      least_work_status="fixed_resource_no_exchange_candidates", **replayed_frontier)
        return FixedResourceRefinementResult(action=action, improved=False, status="fixed_resource_no_exchange_candidates",
                                             counters=counters, seconds=time.perf_counter() - started, history=list(history),
                                             search_scope=search_scope)
    a_rows = j_rows = None
    if not progressive:
        row_candidates = torch.cat([drop_queue, add_queue]).unique(sorted=True)
        explicit_row_cap = drop_limit + add_limit
        if int(row_candidates.numel()) > explicit_row_cap:
            raise ValueError("fixed-resource row candidate count exceeds explicit drop/add cap")
        # The retained finite-screen diagnostic still uses whole banks when
        # they fit jointly, otherwise its existing pair tile geometry streams.
        if _row_pair_bytes(ops, int(row_candidates.numel())) <= config.candidate_retained_bytes:
            a_rows, j_rows = _candidate_row_pair(ops, row_candidates, config, counters,
                                                candidate_eligible=physical_parent)
            counters['candidate_rows_peak_bytes'] = _row_pair_bytes(ops, int(row_candidates.numel()))
    stream_args = (ops, geom, base_cells, base_response, base_stats["work"], base_stats["energy"],
                   a_rows, j_rows, drop_queue, add_queue, drop_positions, survivor, r, config, counters)
    stream = (_ProgressivePairStream(*stream_args, drop_primary=drop_primary, add_primary=add_primary)
              if progressive else _CompactPairStream(*stream_args))
    search_scope = getattr(stream, 'search_scope', 'bounded_screened_pair_list_at_frontier')
    with closing(stream) as compact:
        for item in islice(compact, config.max_endpoint_replays):
            counters["endpoint_replays"] += 1
            trial_z = z.clone()
            trial_z[item.drop_route] = 0.0
            trial_z[item.add_route] = 1.0
            trial_order = _replace_order(order, item.drop_route, item.add_route)
            cells, _response, stats = _endpoint(ops, geom, g, trial_z, counters)
            del cells
            replay_survivor_ok = True
            try:
                _survivor_state(ops, trial_z, physical_parent)
            except ValueError:
                replay_survivor_ok = False
            accepted = (stats["resource"] == resource and stats["half_energy"] <= r and
                        stats["work"] < base_stats["work"] and replay_survivor_ok)
            phase = None
            if accepted and config.require_local_phase_stability:
                phase = local_action_kinks(ops, trial_z, workspace_bytes=config.phase_workspace_bytes)
                counters["local_phase_replays"] = counters.get("local_phase_replays", 0) + 1
                accepted = phase["local_phase_unchanged"]
            record = dict(drop=item.drop_route, add=item.add_route, accepted=bool(accepted),
                                local_work=item.work, replay_work=stats["work"], local_energy=item.energy,
                                replay_energy=stats["energy"], drop_position=item.drop_position)
            if config.require_local_phase_stability:
                record.update(local_phase=phase,
                              local_phase_refused=phase is not None and not phase["local_phase_unchanged"])
            history.append(record)
            if not accepted:
                continue
            compact.close()
            action = dict(order=trial_order, z=trial_z, resource=resource, energy=stats["energy"],
                          half_energy=stats["half_energy"], predicted_damage=stats["predicted_damage"],
                          first_order=stats["first_order"], work=stats["work"], key="fixed_resource_least_work",
                          stage="fixed_resource_refined", frontier_resource=resource,
                          frontier_work=base_stats["work"], frontier_energy=base_stats["energy"],
                          least_work_status="fixed_resource_strict_work_improvement",
                          lam=None, evaluations=int(counters["endpoint_replays"]), operator_calls=dict(counters),
                          history=list(history), authority="proposal_only_observed_chart_fixed_resource_refinement",
                          refinement_endpoint_budget_exhausted=False,
                          refinement_screened_endpoints=compact.screened_endpoints,
                          refinement_unattempted_endpoints=compact.screened_endpoints-counters["endpoint_replays"],
                          refinement_search_scope=search_scope,
                          refinement_unvisited_pairs=getattr(compact, 'unvisited_pairs', 0),
                          refinement_ranking=ranking,
                          optimality_authority=False, population_certified=False, bootstrap_certified=False,
                          production_commit_authority=False, seconds=time.perf_counter() - started,
                          refinement_config=config.to_json(), **replayed_frontier)
            return FixedResourceRefinementResult(action=action, improved=True,
                                                 status="fixed_resource_strict_work_improvement",
                                                 counters=dict(counters), seconds=time.perf_counter() - started,
                                                 history=list(history), screened_endpoints=compact.screened_endpoints,
                                                 unattempted_endpoints=compact.screened_endpoints-counters["endpoint_replays"],
                                                 unvisited_pairs=getattr(compact, 'unvisited_pairs', 0),
                                                 search_scope=search_scope)
        compact.close()
        remaining_endpoints = compact.screened_endpoints - counters["endpoint_replays"]
        unvisited_pairs = getattr(compact, 'unvisited_pairs', 0)
        budget_exhausted = remaining_endpoints > 0 or unvisited_pairs > 0
        status = ("fixed_resource_endpoint_budget_exhausted" if budget_exhausted
                  else "fixed_resource_no_strict_work_improvement")
        action = dict(frontier_action, stage="frontier_only_fixed_resource_no_improvement",
                      least_work_status=status,
                      refinement_endpoint_budget_exhausted=budget_exhausted,
                      refinement_unattempted_endpoints=remaining_endpoints,
                      refinement_screened_endpoints=compact.screened_endpoints,
                      refinement_search_scope=search_scope,
                      refinement_unvisited_pairs=unvisited_pairs,
                      refinement_ranking=ranking,
                      fixed_resource_refinement_candidate_count=compact.screened_endpoints,
                      fixed_resource_refinement_operator_calls=dict(counters),
                      fixed_resource_refinement_history=list(history), refinement_config=config.to_json(), **replayed_frontier)
        return FixedResourceRefinementResult(action=action, improved=False,
                                             status=status,
                                             counters=dict(counters), seconds=time.perf_counter() - started,
                                             history=list(history), endpoint_budget_exhausted=budget_exhausted,
                                             screened_endpoints=compact.screened_endpoints, unattempted_endpoints=remaining_endpoints,
                                             unvisited_pairs=unvisited_pairs, search_scope=search_scope)


def refine_frontier_fixed_resource(chart: Mapping[str, Any], radius: float, frontier_action: Mapping[str, Any], *,
                                   config: FixedResourceRefinementConfig = FixedResourceRefinementConfig()
                                   ) -> FixedResourceRefinementResult:
    """Descend by replayed strict improvements until the declared search stops.

    Every pass rebuilds its work priorities and survivor state at the new action.
    The physical parent, chart, radius and released resource remain fixed. With
    deterministic frozen replay, strict work descent cannot revisit a binary
    mask, so the finite path terminates without an exchange-count default. This
    is no polynomial-runtime or mathematical local/global optimality guarantee.
    """
    started = time.perf_counter()
    current = dict(frontier_action)
    counters: dict[str, int] = {}
    history = deque(maxlen=64)
    descent_history = deque(maxlen=64)
    total_screened = 0
    exchanges = 0
    passes = 0
    previous_work = previous_energy = None
    initial_work = initial_energy = None
    exchange_exhausted = False
    endpoint_exhausted = False
    pending_neighborhood = False
    while True:
        used = counters.get('endpoint_replays', 0)
        remaining = None if config.max_endpoint_replays is None else config.max_endpoint_replays-used
        # A zero remaining budget is handled after the prior accepted pass;
        # no invalid zero-budget request or new search is opened here.
        if remaining is not None and remaining <= 0:
            raise RuntimeError('refinement opened a pass without replay budget')
        last = _refine_frontier_exchange(chart, radius, current,
            config=replace(config, max_endpoint_replays=remaining))
        replay_work = last.action['refinement_replayed_frontier_work']
        replay_energy = last.action['refinement_replayed_frontier_energy']
        if previous_work is not None and (replay_work != previous_work or replay_energy != previous_energy):
            raise RuntimeError('frozen refinement frontier changed on replay')
        if initial_work is None:
            initial_work, initial_energy = replay_work, replay_energy
        counters = merge_operator_counters(counters, last.counters)
        total_screened += last.screened_endpoints
        for record in last.history:
            history.append(dict(record, exchange_epoch=passes))
        passes += 1
        if not last.improved:
            endpoint_exhausted = last.endpoint_budget_exhausted
            termination = ('endpoint_budget_exhausted' if endpoint_exhausted else last.status)
            break
        if not last.action['work'] < replay_work:
            raise RuntimeError('refinement exchange did not strictly reduce replayed work')
        if current.get('resource', last.action['resource']) != last.action['resource']:
            raise RuntimeError('refinement exchange changed released resource')
        current = dict(last.action)
        previous_work, previous_energy = current['work'], current['energy']
        descent_history.append(dict(exchange=exchanges, work_before=replay_work,
            work_after=previous_work, energy_before=replay_energy, energy_after=previous_energy,
            resource=current['resource'], drop=last.history[-1]['drop'], add=last.history[-1]['add']))
        exchanges += 1
        endpoint_exhausted = (config.max_endpoint_replays is not None and
                              counters.get('endpoint_replays', 0) >= config.max_endpoint_replays)
        exchange_exhausted = config.max_exchanges is not None and exchanges >= config.max_exchanges
        if endpoint_exhausted or exchange_exhausted:
            pending_neighborhood = True
            termination = ('endpoint_budget_exhausted' if endpoint_exhausted else 'exchange_budget_exhausted')
            break
    improved = exchanges > 0
    search_exhausted = not last.improved and not last.endpoint_budget_exhausted
    status = 'fixed_resource_strict_work_improvement' if improved else last.status
    action = dict(current if improved else last.action)
    counters.update(refinement_passes=passes, refinement_accepted_exchanges=exchanges)
    elapsed = time.perf_counter()-started
    action.update(least_work_status=status, frontier_work=initial_work, frontier_energy=initial_energy,
        refinement_config=config.to_json(), refinement_exchanges=exchanges,
        refinement_termination_reason=termination, refinement_search_exhausted=search_exhausted,
        refinement_neighborhood_exhausted=(search_exhausted and last.search_scope == 'complete_eligible_single_exchange_domain'),
        refinement_next_neighborhood_unopened=pending_neighborhood,
        refinement_last_search_epoch=passes-1, refinement_last_search_is_returned_action=not last.improved,
        refinement_total_screened_endpoints=total_screened, refinement_descent_history=list(descent_history),
        refinement_endpoint_budget_exhausted=endpoint_exhausted,
        refinement_exchange_budget_exhausted=exchange_exhausted,
        refinement_search_scope=last.search_scope, refinement_screened_endpoints=last.screened_endpoints,
        refinement_unattempted_endpoints=last.unattempted_endpoints, refinement_unvisited_pairs=last.unvisited_pairs,
        refinement_last_search_status=last.status,
        evaluations=counters.get('endpoint_replays', 0), operator_calls=dict(counters),
        fixed_resource_refinement_operator_calls=dict(counters), history=list(history),
        fixed_resource_refinement_history=list(history), seconds=elapsed)
    return FixedResourceRefinementResult(action=action, improved=improved, status=status,
        counters=counters, seconds=elapsed, history=list(history), endpoint_budget_exhausted=endpoint_exhausted,
        screened_endpoints=last.screened_endpoints, unattempted_endpoints=last.unattempted_endpoints,
        unvisited_pairs=last.unvisited_pairs, search_scope=last.search_scope, exchanges=exchanges,
        exchange_budget_exhausted=exchange_exhausted, search_exhausted=search_exhausted)
