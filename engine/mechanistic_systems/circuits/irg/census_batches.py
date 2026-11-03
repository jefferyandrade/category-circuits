"""Batch-sequential importance census.

One global stream of off-target images, drawn in fixed batches of b* from the
current importance-weight state; every sampled image is scored for all layers
on the same forward and backward; after every batch each scope's stop
statistic is updated; sampling continues until every scope passes the stop
rule or the population limit is reached.  On-target images (and any certainty
set) enter once, exactly.

Estimator.  Off-target images are drawn
with replacement from fixed probabilities q_i proportional to a_i |p_i - y_i|
(Hansen-Hurwitz).  Each draw d of image i gives z_dr = a_i y_ir / q_i, unbiased
for the off-target total.  After n draws the census estimate is
``exact + (1/n) sum_d z_d`` and its within-sample variance is
``(sum_d z_d**2 / n - zbar**2) / (n - 1)`` (nu = n - 1).  A session keeps,
per route, the exact sum, S1 = sum_d z_d and S2 = sum_d z_d**2: per unique
image in a batch, ``first = m_i a_i / q_i`` multiplies y and
``second = m_i (a_i / q_i)**2`` multiplies y**2 (m_i = draws of i).  The
batch-to-batch dispersion is a diagnostic only.  An image with b* q_i >= 1
(it would be drawn more than once per batch) is enumerated exactly instead and
removed from q (the certainty set).

The stop rule is a pluggable callable (the indifference-zone rule tied to tau
is in ``census_stop``).  This module decides only which
images enter each batch and when the census stops; the weighted census itself
is computed by :class:`BatchCensusSession`.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .census_pps import class_weights, size_measure  # noqa: F401  (re-exported inputs)

CENSUS_SAMPLE_DESIGN_BATCHES = "importance_batches_v1"
BATCH_DRAW_ROLE = "census_batch_draw_v1"
STOP_PASSED = "all_scopes_passed"
STOP_POPULATION = "population_limit"
STOP_EXACT_SWITCH = "exact_switch"
STOP_EXACT_START = "exact_from_start"


# ---------------------------------------------------------------------------
# Weight state and the batch sampler.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeightState:
    """Normalized draw probabilities q over the off-target population.

    ``version`` counts refits; ``source`` names what the sizes came from.
    """

    q: np.ndarray
    version: int
    source: str

    @classmethod
    def from_sizes(cls, sizes: np.ndarray, *, source: str, version: int = 0) -> "WeightState":
        values = np.asarray(sizes, dtype=np.float64).reshape(-1)
        if values.size == 0 or not np.isfinite(values).all() or np.any(values < 0.0) or not values.sum() > 0.0:
            raise ValueError("census importance sizes must be finite, nonnegative and not all zero")
        return cls(q=values / values.sum(), version=int(version), source=str(source))

    def receipt(self) -> dict[str, Any]:
        q = self.q
        return {
            "version": self.version,
            "source": self.source,
            "support": int(np.count_nonzero(q)),
            "kish_effective_draw_size": float(1.0 / np.square(q).sum()),
            "q_hash": hashlib.sha256(np.ascontiguousarray(q).tobytes()).hexdigest(),
        }


def initial_weight_state(off_target_a: np.ndarray, off_target_u: np.ndarray) -> WeightState:
    """First implementation: q_i proportional to a_i * |p_i - y_i|."""

    return WeightState.from_sizes(
        np.asarray(off_target_a, dtype=np.float64) * np.asarray(off_target_u, dtype=np.float64),
        source="a_times_abs_p_minus_y",
    )


def certainty_split(a: np.ndarray, u: np.ndarray, batch_size: int) -> tuple[np.ndarray, WeightState | None]:
    """Images with b* q_i >= 1 (iterated after renormalising) and the weight state of the rest.

    Returns (certainty mask, state over the population with the certainty set
    at q = 0); the state is None when every image is in the certainty set.
    """

    x = np.asarray(a, dtype=np.float64) * np.asarray(u, dtype=np.float64)
    certain = np.zeros(x.size, dtype=bool)
    while True:
        rest = np.where(certain, 0.0, x)
        total = rest.sum()
        if not total > 0.0:
            return certain, None
        grow = (~certain) & (int(batch_size) * rest / total >= 1.0)
        if not grow.any():
            return certain, WeightState.from_sizes(rest, source="a_times_abs_p_minus_y_without_certainty_set")
        certain |= grow


@dataclass(frozen=True)
class Batch:
    index: int
    context_ids: tuple[str, ...]      # unique images, first-draw order
    multiplicity: np.ndarray          # draws per unique image
    first_weights: np.ndarray         # m_i a_i / q_i   (multiplies y)
    second_weights: np.ndarray        # m_i (a_i/q_i)^2 (multiplies y^2)
    positions: np.ndarray             # indices into the off-target population
    weight_state_version: int
    draws: int


class BatchSampler:
    """Draws batch k with replacement from q, reproducibly for (run_id, seed, k)."""

    def __init__(self, off_target_ids: Sequence[str], off_target_a: np.ndarray, *, run_id: str,
                 random_seed: int, content_hash: Callable[..., str]):
        self.ids = np.asarray([str(v) for v in off_target_ids])
        self.a = np.asarray(off_target_a, dtype=np.float64).reshape(-1)
        if self.ids.size != self.a.size or self.ids.size == 0:
            raise ValueError("census batch sampler inputs differ in length or are empty")
        self.run_id = str(run_id)
        self.random_seed = int(random_seed)
        self._content_hash = content_hash

    def _uniforms(self, index: int, count: int) -> np.ndarray:
        digest = self._content_hash({
            "run_id": self.run_id, "random_seed": self.random_seed, "role": BATCH_DRAW_ROLE,
            "batch_index": int(index),
        })
        hexdigest = "".join(ch for ch in str(digest) if ch in "0123456789abcdef")[-32:]
        key = np.array([int(hexdigest[:16], 16), int(hexdigest[16:], 16)], dtype=np.uint64)
        return np.random.Generator(np.random.Philox(key=key)).random(int(count))

    def draw(self, state: WeightState, batch_size: int, index: int) -> Batch:
        b = int(batch_size)
        if b < 1:
            raise ValueError("census batch size must be positive")
        if state.q.size != self.ids.size:
            raise ValueError("census weight state does not cover the off-target population")
        cdf = np.cumsum(state.q)
        cdf[-1] = 1.0
        positions = np.searchsorted(cdf, self._uniforms(index, b), side="right")
        positions = np.minimum(positions, self.ids.size - 1)
        unique, first, counts = np.unique(positions, return_index=True, return_counts=True)
        keep = np.argsort(first, kind="stable")
        unique, counts = unique[keep], counts[keep]
        ratio = self.a[unique] / state.q[unique]
        return Batch(
            index=int(index), context_ids=tuple(self.ids[unique].tolist()), multiplicity=counts.astype(np.int64),
            first_weights=counts * ratio, second_weights=counts * ratio * ratio, positions=unique,
            weight_state_version=state.version, draws=b,
        )


# ---------------------------------------------------------------------------
# The session protocol (kernel worker) and a NumPy reference.
# ---------------------------------------------------------------------------


class BatchCensusSession(Protocol):
    """The weighted fast census as the batch loop drives it (every route, every layer).

    ``add_exact(ids, weights)``: exact images (on-target, certainty set,
    exhausted population): exact_r += w_i y_ir.
    ``add_batch(ids, first_weights, second_weights, draws, index)``: one batch
    of with-replacement draws: S1_r += sum_i first_i y_ir,
    S2_r += sum_i second_i y_ir**2, n += draws (and, as a diagnostic, the
    batch total T = S1 increment / draws folded into sum_T, sum_T2).
    ``drop_sampled()``: forget S1, S2, n (used when the population is
    enumerated exactly instead).
    ``scope_fields()``: per scope (estimate, within-sample SE) in the scope's
    route order: estimate = exact + S1/n; SE = sqrt(max(S2/n - (S1/n)**2, 0) / (n - 1)).
    ``result()``: the census 6-tuple (the fields of every head, the walk's
    ordering statistics, flip margins, the session's receipt).
    """

    def add_exact(self, context_ids: Sequence[str], *, weights: np.ndarray) -> None: ...

    def add_batch(self, context_ids: Sequence[str], *, first_weights: np.ndarray, second_weights: np.ndarray,
                  draws: int, index: int) -> None: ...

    def drop_sampled(self) -> None: ...

    def scope_fields(self) -> Mapping[str, tuple[np.ndarray, np.ndarray]]: ...


@dataclass
class ReferenceBatchSession:
    rows: Mapping[str, np.ndarray]
    scope_slices: Mapping[str, slice]

    def __post_init__(self) -> None:
        width = len(next(iter(self.rows.values())))
        self.exact = np.zeros(width)
        self.drop_sampled()

    def drop_sampled(self) -> None:
        width = self.exact.size
        self.s1 = np.zeros(width)
        self.s2 = np.zeros(width)
        self.sum_t = np.zeros(width)
        self.sum_t2 = np.zeros(width)
        self.n = 0
        self.batches = 0

    def add_exact(self, context_ids, *, weights):
        for cid, w in zip(context_ids, weights, strict=True):
            self.exact += float(w) * np.asarray(self.rows[cid], dtype=np.float64)

    def add_batch(self, context_ids, *, first_weights, second_weights, draws, index):
        inc = np.zeros_like(self.s1)
        for cid, w1, w2 in zip(context_ids, first_weights, second_weights, strict=True):
            y = np.asarray(self.rows[cid], dtype=np.float64)
            inc += float(w1) * y
            self.s2 += float(w2) * y * y
        self.s1 += inc
        t = inc / int(draws)
        self.sum_t += t
        self.sum_t2 += t * t
        self.n += int(draws)
        self.batches += 1

    def fields(self) -> tuple[np.ndarray, np.ndarray]:
        n = self.n
        if n == 0:
            return self.exact.copy(), np.zeros_like(self.exact)
        mean = self.s1 / n
        se = (np.sqrt(np.maximum(self.s2 / n - mean * mean, 0.0) / (n - 1)) if n > 1
              else np.full_like(mean, np.inf))
        return self.exact + mean, se

    def scope_fields(self):
        estimate, se = self.fields()
        return {scope: (estimate[sl], se[sl]) for scope, sl in self.scope_slices.items()}


# ---------------------------------------------------------------------------
# The loop.
# ---------------------------------------------------------------------------

StopRule = Callable[..., Mapping[str, Any]]
RefitHook = Callable[[WeightState, Any, int], "WeightState | None"]


def _ids_hash(ids: Sequence[str]) -> str:
    h = hashlib.sha256()
    for value in ids:
        h.update(str(value).encode())
        h.update(b"\0")
    return h.hexdigest()


def run_batch_census(
    session: BatchCensusSession,
    *,
    sampler: BatchSampler,
    off_target_u: np.ndarray,
    on_target_ids: Sequence[str],
    on_target_a: np.ndarray,
    batch_size: int,
    stop_rule: StopRule,
    first_check_draws: int = 2048,
    check_growth: float = 0.10,
    population_limit_draws: int | None = None,
    refit: RefitHook | None = None,
    clock: Callable[[], float] = time.perf_counter,
    progress: Callable[..., None] | None = None,
    enumerate_exactly: bool = False,
) -> dict[str, Any]:
    """Draw batches of b* until the stop rule passes or the population limit binds.

    On-target images (and the certainty set, b* q_i >= 1) enter exactly
    before the first batch.  ``stop_rule(fields, look=j, n_draws=n,
    population_exhausted=False)`` returns a report with ``passed`` (bool),
    ``binding_scope`` and per-scope entries.  Checks follow a schedule fixed
    in advance: the first once the off-target draws reach
    ``first_check_draws``, then whenever they have grown by ``check_growth``
    since the last check; ``look`` is the check index j (alpha is spent per
    check).  When the draws reach
    ``population_limit_draws`` (default: the off-target population size) the
    remaining population is enumerated exactly (``drop_sampled`` then
    ``add_exact`` of every off-target image), which is the exact census, and
    the stop is evaluated once more with ``population_exhausted=True``.
    A ``population_limit_draws`` below the population is the registered exact
    switch: the remaining census is enumerated as soon as
    the draws reach it (reason ``exact_switch``).
    ``enumerate_exactly`` (eps = 0, which can pass only on the exact census)
    skips sampling and enumerates every off-target image from the start.
    ``refit`` is a documented no-op hook (weights are fixed for the
    run).  Wall time is recorded only.
    """

    b = int(batch_size)
    n_off = int(sampler.ids.size)
    limit = int(n_off if population_limit_draws is None else population_limit_draws)
    started = clock()
    certain, state = certainty_split(sampler.a, off_target_u, b)
    session.add_exact(tuple(map(str, on_target_ids)), weights=np.asarray(on_target_a, dtype=np.float64))
    if certain.any():
        session.add_exact(tuple(sampler.ids[certain].tolist()), weights=sampler.a[certain])
    states = [] if state is None else [state.receipt()]
    batches: list[dict[str, Any]] = []
    distinct = np.zeros(n_off, dtype=bool)
    report: Mapping[str, Any] = {}
    reason = None
    k = 0
    draws = 0
    checks = 0
    next_check = int(first_check_draws)
    exhausted = state is None or bool(enumerate_exactly)
    while not exhausted:
        batch = sampler.draw(state, b, k)
        session.add_batch(batch.context_ids, first_weights=batch.first_weights,
                          second_weights=batch.second_weights, draws=batch.draws, index=k)
        distinct[batch.positions] = True
        draws += batch.draws
        record = {
            "batch": batch.index, "weight_state_version": batch.weight_state_version,
            "unique_images": len(batch.context_ids), "draws": batch.draws, "cumulative_draws": draws,
            "ids_hash": _ids_hash(batch.context_ids), "elapsed_wall_seconds": clock() - started,
        }
        k += 1
        if draws >= limit:
            # At the population limit or the exact switch the census is enumerated exactly BEFORE any stop is
            # evaluated: a sampled field never passes at the limit.
            batches.append(record)
            exhausted = True
            break
        if draws >= next_check:
            report = stop_rule(session.scope_fields(), look=checks, n_draws=draws, population_exhausted=False)
            record["check"] = checks
            checks += 1
            next_check = int(math.ceil(draws * (1.0 + float(check_growth))))
            record["passed"] = bool(report["passed"])
            record["binding_scope"] = report.get("binding_scope")
            record["statistic"] = report.get("statistic")
            record["check_wall_seconds"] = clock() - started - record["elapsed_wall_seconds"]
            if progress is not None:
                progress(check=int(record["check"]), draws=int(draws), batches=int(k),
                         statistic=float(report.get("statistic", math.nan)),
                         threshold=float(report.get("alpha_check", math.nan)),
                         binding_scope=report.get("binding_scope"), forecast=report.get("B_forecast"),
                         passed=bool(report["passed"]), elapsed_wall_seconds=float(clock() - started))
            if report["passed"]:
                reason = STOP_PASSED
        batches.append(record)
        if reason is not None:
            break
        if refit is not None:
            new_state = refit(state, session, k)
            if new_state is not None:
                state = new_state
                states.append(state.receipt())
    if exhausted and reason is None:
        session.drop_sampled()
        rest = ~certain
        if rest.any():
            session.add_exact(tuple(sampler.ids[rest].tolist()), weights=sampler.a[rest])
        distinct[:] = True
        # eps = 0 (enumerate_exactly) passes by construction: no field is formed for the stop.
        fields = {} if enumerate_exactly else session.scope_fields()
        report = stop_rule(fields, look=checks, n_draws=draws, population_exhausted=True)
        reason = (STOP_EXACT_START if enumerate_exactly else
                  (STOP_EXACT_SWITCH if limit < n_off else STOP_POPULATION))
    return {
        "schema": "irg_batch_census_receipt_v2",
        "design": CENSUS_SAMPLE_DESIGN_BATCHES,
        "batch_size": b,
        "batch_count": k,
        "checks": checks,
        "check_schedule": {"first_check_draws": int(first_check_draws), "check_growth": float(check_growth)},
        "draws": draws,
        "certainty_set_images": int(certain.sum()),
        "exact_switch_draws": (int(limit) if limit < n_off else None),
        "exact_enumeration_at_draws": (int(draws) if reason in (STOP_POPULATION, STOP_EXACT_SWITCH, STOP_EXACT_START)
                                       else None),
        "distinct_off_target_images": int(distinct.sum()),
        "off_target_population": n_off,
        "on_target_count": len(on_target_ids),
        "stop_reason": reason,
        "stop": dict(report),
        "binding_scope": report.get("binding_scope") if report else None,
        "weight_states": states,
        "batches": batches,
        "elapsed_wall_seconds": clock() - started,
        "wall_clock_role": "recorded_only",
    }


__all__ = [
    "BATCH_DRAW_ROLE", "Batch", "BatchCensusSession", "BatchSampler", "CENSUS_SAMPLE_DESIGN_BATCHES",
    "certainty_split",
    "ReferenceBatchSession", "STOP_EXACT_START", "STOP_EXACT_SWITCH", "STOP_PASSED", "STOP_POPULATION", "WeightState",
    "initial_weight_state",
    "run_batch_census",
]
