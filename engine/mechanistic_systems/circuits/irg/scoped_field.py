"""Adaptive decision-scoped field measurement class for the trust walk.

Implements the registered measurement class
``decision_scoped_sign_order_fd_field_v1``.  The scoped family — the set of
simultaneous statistical claims a look must resolve — is the walk's DECISION
SURFACE, not the route census:

* sign resolution is demanded only within the reachable horizon ``H_t``
  (per partition, the budget-priced admissible prefix, clipped to the
  partition size, with no margin);
* order stability is audited only at the tranche-grid cuts within the
  horizon plus the horizon edge itself, each cut's crossing set PRICED
  with the registered D2 model against the caller's resolution floor
  (absolute damage units, no fraction language);
* the sign bar is the quadrature mu-UCB + sketch energy price of the
  horizon's ambiguous set against the same floor;
* the Frequent Directions energy bracket on each partition's horizon
  prefix probe action must reach the registered coverage fraction.

Stopping is adaptive with a registered look cap; a cap hit is disclosed
as an uncertified bound rail.  Family size tracks the horizon, so the
per-claim Bonferroni critical values self-tune with it.

Claim language (binding).  The family is DATA-SELECTED
(the horizon comes from the same measurement stream; shrink keeps what
resolved), so every receipt claims per-claim normal intervals at the
stated Bonferroni level over the final ENUMERATED family, with the
family disclosed as data-selected and coverage CONDITIONAL ON SELECTION.
Receipts never claim unconditional simultaneous coverage over an
a-priori family, never signs for unresolved routes, never order at
unaudited cuts, and no alpha transferable to arm-level conclusions.  The
class is proposal-side disclosure; commit validity rests exclusively on
the walk's exact paired evidence.

Provider-agnostic by construction: panel delivery and array snapshots
are injected callables, so the statistics run identically on stubs and
on the device acquisition path.  Estimator roles: the FD
state exclusively owns horizon computation and sketch coverage; the
Rademacher global-head factor is carried through untouched (the walk
prices with it) and is never used for horizon or coverage here.

Vocabulary: accept / refuse / resolved / unresolved.  Raising is
reserved for contract violations; statistical non-resolution always
returns a receipt with ``resolved: false``.

FD snapshot strategy (verified, binding here).  Reading
``trust_geometry._compress``/``finalize`` shows ``finalize`` is NOT
snapshot-safe for continued streaming: it passes the accumulator's live
``_mean`` buffer into ``GlobalTrustGeometryState``, whose read-only
coercion freezes that same object in place, so the next ``update`` on
the accumulator raises ``ValueError`` MID-update — after the weighted
second-moment diagonal has already been advanced — leaving a corrupted
sketch for any caller that continues.  ``finalize_fd_snapshot``
therefore deep-copies the accumulator and finalizes the copy; the
original stream is byte-identical to a never-snapshotted accumulator.
That choice is receipted under ``fd.snapshot_strategy``.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .contracts import GlobalTrustGeometrySpec, content_hash
from .ordering_keys import (
    GlobalHeadOrderingField,
    ORDERING_FIELD_STATISTIC_NAMES,
    ORDERING_KEY_REGISTRY_VERSION,
    V1_ORDERING_KEYS,
    ordering_key_statistic,
)
from .precision import simultaneous_normal_critical_value
from .trust_geometry import (
    FrequentDirectionsTrustAccumulator,
    GlobalTrustGeometryState,
)

SCOPED_FIELD_CLASS_ID = "decision_scoped_sign_order_fd_field_v1"

# Block width for the vectorized reachable-prefix pricing sweep.
_PRICE_BLOCK_ROUTES = 4096
_ADMISSIBLE_PREFIX_LENGTH_ROUTE_CEILING = 262_144
SCOPED_FIELD_RECEIPT_SCHEMA = "trust_walk_scoped_field_receipt_v1"

# Selection-disclosure receipt values (binding).
SCOPED_FIELD_FAMILY_SELECTION = "enumerated_family_fixed_at_acquisition"
SCOPED_FIELD_COVERAGE_CLAIM = (
    "per_claim_bonferroni_over_final_enumerated_family_conditional_on_selection"
)

# Verified strategy for mid-stream FD snapshots — see the module
# docstring and ``finalize_fd_snapshot``.
FD_SNAPSHOT_STRATEGY = "deepcopy_before_finalize_v1"
_FD_SNAPSHOT_REASON = (
    "finalize passes the accumulator's live mean buffer into the frozen "
    "state, whose read-only coercion freezes that same object; a "
    "subsequent update raises mid-update after the second-moment "
    "diagonal already advanced, corrupting the stream — so snapshots "
    "finalize a deep copy and the live accumulator is never finalized "
    "mid-stream"
)

# The scoped class measures exactly one signed partition view (the
# global head carried by ``FieldSnapshot.global_mean``); the census
# loop's ``len(required_views)`` specializes to 1 here.
_SIGN_VIEW_COUNT = 1

_TINY = float(np.finfo(np.float64).tiny)


def _readonly_flat_float64(value: Any, *, name: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True).reshape(-1)
    if array.size <= 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite and nonempty")
    array = np.ascontiguousarray(array)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class ScopedFieldPolicy:
    """Registered numerals of the scoped measurement class.

    ``confidence_level`` is the program's per-look simultaneous
    confidence (the census loop's look-split level), consumed directly
    by the per-look Bonferroni critical value — this module performs no
    further look splitting.  The SIGN and ORDER bars and the
    materiality floor are not policy numerals: both bars are absolute
    damage PRICES demanded at or below the caller's
    ``resolution_floor`` (the smallest damage increment the deciding
    oracle can resolve), because ambiguity whose induced damage sits
    below that floor cannot change any decision the field feeds.
    """

    # Structural stability-stop — the chart stops after this many
    # looks once FD coverage passes; sign/order status is disclosed.
    look_cap: int = 16
    fd_sketch_rank: int = 64
    fd_maximum_rank: int = 64
    sketch_coverage_fraction: float = 0.5
    # Default = the registered decision confidence c, not a census-grade
    # numeral that an unset override would silently inherit.
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        value = float(self.sketch_coverage_fraction)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                "scoped-field sketch_coverage_fraction must lie in [0,1]"
            )
        if int(self.look_cap) <= 0:
            raise ValueError("scoped-field look_cap must be positive")
        if int(self.fd_sketch_rank) <= 0:
            raise ValueError("scoped-field fd_sketch_rank must be positive")
        if int(self.fd_maximum_rank) < int(self.fd_sketch_rank):
            raise ValueError(
                "scoped-field fd_maximum_rank must be >= fd_sketch_rank"
            )
        confidence = float(self.confidence_level)
        if not 0.5 < confidence < 1.0:
            raise ValueError(
                "scoped-field confidence_level must lie in (0.5,1)"
            )

    def to_receipt(self) -> dict[str, object]:
        return {
            "look_cap": int(self.look_cap),
            "fd_sketch_rank": int(self.fd_sketch_rank),
            "fd_maximum_rank": int(self.fd_maximum_rank),
            "sketch_coverage_fraction": float(self.sketch_coverage_fraction),
            "confidence_level": float(self.confidence_level),
        }


@dataclass(frozen=True, eq=False, slots=True)
class FieldSnapshot:
    """One look's injected view of the accumulated measurement.

    Every array is float64 and flat over the WHOLE physical route axis:
    scoping changes the enumerated family and stopping rule, never the
    measured arrays.  ``fd_state`` is the finalize-snapshot of the FD
    accumulator at this look (build it with ``finalize_fd_snapshot``);
    ``rademacher_global_factor`` is carried through untouched for the
    walk's tranche pricing and is never consumed here.
    """

    global_mean: np.ndarray
    global_mean_standard_error: np.ndarray
    global_activity: np.ndarray
    global_activity_standard_error: np.ndarray
    statistics: Mapping[str, np.ndarray]
    fd_state: GlobalTrustGeometryState
    rademacher_global_factor: np.ndarray
    contexts_consumed: int
    census_exhausted: bool

    def __post_init__(self) -> None:
        mean = _readonly_flat_float64(
            self.global_mean, name="snapshot global_mean"
        )
        arrays = {
            "global_mean_standard_error": self.global_mean_standard_error,
            "global_activity": self.global_activity,
            "global_activity_standard_error": (
                self.global_activity_standard_error
            ),
        }
        coerced: dict[str, np.ndarray] = {}
        for name, value in arrays.items():
            array = _readonly_flat_float64(value, name=f"snapshot {name}")
            if array.shape != mean.shape:
                raise ValueError(
                    "snapshot arrays must share one flat route axis"
                )
            if np.any(array < 0.0):
                raise ValueError(f"snapshot {name} must be nonnegative")
            coerced[name] = array
        statistics: dict[str, np.ndarray] = {}
        for name in ORDERING_FIELD_STATISTIC_NAMES:
            if name not in self.statistics:
                raise ValueError(
                    f"snapshot statistics are incomplete: missing {name!r}"
                )
        for name, value in dict(self.statistics).items():
            array = _readonly_flat_float64(
                value, name=f"snapshot statistic {name}"
            )
            if array.shape != mean.shape:
                raise ValueError(
                    "snapshot statistics must share the route axis"
                )
            statistics[str(name)] = array
        # Coherence: the two standard errors that appear both as
        # top-level fields and as ordering statistics must be the same
        # measurement (the fused snapshot supplies both from one sum).
        for name in (
            "global_mean_standard_error",
            "global_activity_standard_error",
        ):
            if not np.array_equal(statistics[name], coerced[name]):
                raise ValueError(
                    f"snapshot statistic {name} disagrees with the "
                    "top-level field of the same name"
                )
        if not isinstance(self.fd_state, GlobalTrustGeometryState):
            raise ValueError("snapshot fd_state must be a trust geometry")
        if len(self.fd_state.route_ids) != mean.size:
            raise ValueError(
                "snapshot fd_state route axis disagrees with the arrays"
            )
        rademacher = np.array(
            self.rademacher_global_factor, dtype=np.float64, copy=True
        )
        if rademacher.size <= 0 or not np.isfinite(rademacher).all():
            raise ValueError(
                "snapshot rademacher_global_factor must be finite"
            )
        rademacher.setflags(write=False)
        if int(self.contexts_consumed) < 0:
            raise ValueError("snapshot contexts_consumed must be >= 0")
        object.__setattr__(self, "global_mean", mean)
        for name, array in coerced.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "statistics", statistics)
        object.__setattr__(self, "rademacher_global_factor", rademacher)
        object.__setattr__(
            self, "contexts_consumed", int(self.contexts_consumed)
        )
        object.__setattr__(
            self, "census_exhausted", bool(self.census_exhausted)
        )

    @property
    def route_count(self) -> int:
        return int(self.global_mean.size)


@dataclass(frozen=True, eq=False, slots=True)
class ScopedFieldResult:
    """Final snapshot plus the scoped decision surface and its receipt."""

    snapshot: FieldSnapshot
    partition_labels: np.ndarray
    key_values: np.ndarray
    key_standard_errors: np.ndarray
    horizon: Mapping[str, int]
    order_certificate: Mapping[str, Any]
    receipt: Mapping[str, Any]
    looks_taken: int

    def __post_init__(self) -> None:
        labels = np.array(self.partition_labels, dtype=np.int8, copy=True)
        labels = labels.reshape(-1)
        labels.setflags(write=False)
        object.__setattr__(self, "partition_labels", labels)
        object.__setattr__(
            self,
            "key_values",
            _readonly_flat_float64(self.key_values, name="result key values"),
        )
        object.__setattr__(
            self,
            "key_standard_errors",
            _readonly_flat_float64(
                self.key_standard_errors, name="result key standard errors"
            ),
        )
        object.__setattr__(
            self,
            "horizon",
            {str(key): int(value) for key, value in dict(self.horizon).items()},
        )
        object.__setattr__(self, "order_certificate", dict(self.order_certificate))
        object.__setattr__(self, "receipt", dict(self.receipt))
        object.__setattr__(self, "looks_taken", int(self.looks_taken))
        if self.looks_taken <= 0:
            raise ValueError("scoped-field result requires at least one look")


def finalize_fd_snapshot(
    accumulator: FrequentDirectionsTrustAccumulator,
    *,
    spec: GlobalTrustGeometrySpec,
    measured_mask_hash: str,
    calibration_state_id: str,
    source_measurement_hashes: Sequence[str],
    metadata: Mapping[str, Any] | None = None,
) -> GlobalTrustGeometryState:
    """Finalize a DEEP COPY of the live accumulator (verified necessity).

    ``FrequentDirectionsTrustAccumulator.finalize`` is not snapshot-safe:
    the returned state's read-only coercion freezes the accumulator's
    live mean buffer in place, so the next ``update`` raises mid-update
    and leaves the sketch partially advanced.  Finalizing a deep copy
    leaves the original stream byte-identical to a never-snapshotted
    accumulator; the snapshot itself equals a direct finalize of an
    equally fed accumulator.  This choice is receipted under
    ``fd.snapshot_strategy`` as ``deepcopy_before_finalize_v1``.
    """

    clone = copy.deepcopy(accumulator)
    return clone.finalize(
        spec=spec,
        measured_mask_hash=measured_mask_hash,
        calibration_state_id=calibration_state_id,
        source_measurement_hashes=source_measurement_hashes,
        metadata=metadata,
    )


def _price_reachable_prefix(
    *,
    order: np.ndarray,
    global_mean: np.ndarray,
    factor: np.ndarray,
    reachable_budget: float,
) -> int:
    """Longest prefix whose D2 price fits the reachable budget.

    The walk can never legitimately travel further than the remaining
    span budget within a chart era, so the measurement horizon derives
    from the budget itself — the last acquisition-side preset radius
    retires.  Incremental O(rank)-per-route sweep of the joint price
    (sum mu)^2 + ||F_G 1_S||^2 along the prune-first order.
    """

    if order.size == 0 or reachable_budget <= 0.0:
        return 0
    # Runtime only: identical arithmetic, blocked and vectorized.  A
    # per-route Python loop would dominate wall-clock (~90 percent of a
    # 48-minute canary arm), because a large early
    # budget walks this prefix through hundreds of thousands of routes
    # (measured admissible prefix: 407,065) at one rank-64 dot product per
    # route. Blocked np.cumsum with carried offsets is the same sequential
    # accumulation the walk's own sweep already uses.
    budget = float(reachable_budget)
    columns = np.asarray(order, dtype=np.int64)
    positions = int(columns.size)
    run_mu = 0.0
    run_slice = np.zeros(factor.shape[0], dtype=np.float64)
    length = 0
    for start in range(0, positions, _PRICE_BLOCK_ROUTES):
        stop = min(start + _PRICE_BLOCK_ROUTES, positions)
        block = columns[start:stop]
        mu_run = run_mu + np.cumsum(
            np.asarray(global_mean, dtype=np.float64)[block]
        )
        slice_run = run_slice[:, None] + np.cumsum(
            np.asarray(factor, dtype=np.float64)[:, block], axis=1
        )
        price = mu_run * mu_run + np.sum(slice_run * slice_run, axis=0)
        exceeded = np.flatnonzero(price > budget)
        if exceeded.size:
            return length + int(exceeded[0])
        length = stop
        run_mu = float(mu_run[-1])
        run_slice = slice_run[:, -1]
    return length


def _admissible_prefix_length(
    *,
    order: np.ndarray,
    fd_state: GlobalTrustGeometryState,
    leverage_cap: float,
) -> int:
    """Longest prefix whose removal action stays inside the leverage cap.

    No-budget fallback arm only (walk-path horizons are
    budget-derived): a plain
    incremental scan of the exact envelope, mirroring the budget arm's
    linear sweep, stopping at the first prefix whose
    ``sqrt(upper energy)`` exceeds the cap.  The returned length always
    satisfies its own cap; where the mean/factor cross terms make the
    energy locally non-monotone the scan is conservative.
    """

    length = int(order.size)
    if length > _ADMISSIBLE_PREFIX_LENGTH_ROUTE_CEILING:
        raise ValueError(
            f"no-budget fallback horizon length {length} exceeds "
            f"{_ADMISSIBLE_PREFIX_LENGTH_ROUTE_CEILING}; supply reachable_budget "
            "to use the walk path's budget-derived horizon"
        )
    if length == 0:
        return 0
    route_count = len(fd_state.route_ids)
    action = np.zeros(route_count, dtype=np.float64)
    admissible = 0
    for position in range(length):
        action[int(order[position])] = -1.0
        _lower, upper = fd_state.action_energy_bounds(action)
        if math.sqrt(max(float(upper), 0.0)) > float(leverage_cap):
            break
        admissible = position + 1
    return admissible

def _partition_orders(
    labels: np.ndarray, key_values: np.ndarray
) -> dict[int, np.ndarray]:
    """Pruned-first orders: LOWEST key value first, per partition.

    Direction convention (defined once, binding for this class): the
    walk prunes ascending in the registered key, so a partition's order
    lists its route indices by ascending key value; exact key ties keep
    route-axis ascending order (stable sort), matching the order
    certificate's stable route-axis exact-tie policy.
    """

    orders: dict[int, np.ndarray] = {}
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        orders[label] = indices[
            np.argsort(key_values[indices], kind="stable")
        ]
    return orders


def _audited_cuts(
    *,
    order: np.ndarray,
    horizon: int,
    tranche_grid: Callable[[np.ndarray, int], tuple[int, ...]],
) -> tuple[int, ...]:
    """Tranche-grid cut counts within the horizon plus the horizon edge.

    Cuts are pruned-prefix sizes.  Only INTERIOR cuts (``1 <= cut <=
    partition size - 1``) carry an order claim: retaining everything or
    nothing does not depend on score order (the census loop's declared
    interior-only precedent), so edge cuts at 0 or at the partition size
    are dropped from the audited family rather than audited vacuously.
    """

    size = int(order.size)
    if horizon <= 0 or size == 0:
        return ()
    raw = tranche_grid(order, int(horizon))
    cuts: list[int] = []
    for value in tuple(raw):
        cut = int(value)
        if cut != value or cut < 1 or cut > horizon:
            raise ValueError(
                "tranche grid must return integer cut counts within "
                "[1, horizon]"
            )
        cuts.append(cut)
    cuts.append(int(horizon))
    interior = tuple(
        sorted(dict.fromkeys(cut for cut in cuts if 1 <= cut <= size - 1))
    )
    return interior


@dataclass(frozen=True, eq=False)
class _FamilyEvaluation:
    horizon_by_label: dict[int, int]
    cuts_by_label: dict[int, tuple[int, ...]]
    family_size: int
    boundaries_total: int
    critical: float | None
    ambiguous: np.ndarray | None
    sign_price: float | None
    sign_pass: bool
    order_receipt: dict[str, Any]
    order_price: float | None
    order_pass: bool
    coverage_by_label: dict[int, float]
    coverage_pass: bool
    # Materiality disclosure (defaults serve the empty-family path).
    materiality_floor: float | None = None
    immaterial_route_count: int = 0
    immaterial_activity_fraction: float = 0.0
    immaterial_mu_ucb_total: float = 0.0

    @property
    def all_pass(self) -> bool:
        return (
            self.family_size > 0
            and self.sign_pass
            and self.order_pass
            and self.coverage_pass
        )


def _order_price_receipt(
    *,
    resolved: bool,
    price_floor: float | None,
    maximum_cut_price: float | None,
    per_cut: tuple[dict[str, Any], ...] = (),
    note: str = "",
) -> dict[str, Any]:
    """Order receipt: audited cuts priced in absolute damage units."""

    return {
        "schema": "trust_walk_scoped_field_order_priced_v1",
        "audited_cut_count": len(per_cut),
        "decision_resolved": bool(resolved),
        "maximum_observed_cut_price": (
            None if maximum_cut_price is None else float(maximum_cut_price)
        ),
        "price_floor": (
            None if price_floor is None else float(price_floor)
        ),
        "per_cut": list(per_cut),
        "note": str(note),
    }


def _damage_price(
    indices: np.ndarray,
    *,
    global_mean: np.ndarray,
    factor: np.ndarray,
) -> tuple[float, float]:
    """Registered D2 price of removing ``indices`` together.

    Returns ``(price, energy)`` with price = (sum mu_hat)^2 + energy and
    energy = ||F_G 1_S||^2 from the Rademacher global-head slice — the
    sketch's registered pricing role, respecting cancellation in
    the signed mean term.
    """

    if indices.size == 0:
        return 0.0, 0.0
    mean_term = float(np.sum(global_mean[indices])) ** 2
    rows = (
        factor[:, indices]
        if factor.ndim == 2
        else np.asarray(factor, dtype=np.float64)[np.newaxis, indices]
    )
    energy = float(np.sum(np.sum(rows, axis=1) ** 2))
    return mean_term + energy, energy


def _cut_crossing_indices(
    member_indices: np.ndarray,
    *,
    key_values: np.ndarray,
    key_standard_errors: np.ndarray,
    cut: int,
    critical: float,
) -> np.ndarray:
    """Route indices whose key interval straddles one audited cut.

    Byte-identical interval conventions to the retired certificate
    branch: descending stable order, zero-clamped lower bounds, strict
    crossing comparisons.  ``cut`` is the pruned-prefix count.
    """

    values = key_values[member_indices]
    errors = key_standard_errors[member_indices]
    order = np.argsort(-values, kind="stable")
    ordered_indices = member_indices[order]
    ordered = values[order]
    ordered_error = errors[order]
    lower = np.maximum(ordered - float(critical) * ordered_error, 0.0)
    upper = ordered + float(critical) * ordered_error
    retained = int(member_indices.size) - int(cut)
    if retained <= 0 or retained >= int(member_indices.size):
        return ordered_indices[:0]
    largest_unselected_upper = float(np.max(upper[retained:]))
    smallest_selected_lower = float(np.min(lower[:retained]))
    uncertain = np.concatenate(
        (
            lower[:retained] < largest_unselected_upper,
            upper[retained:] > smallest_selected_lower,
        )
    )
    return ordered_indices[uncertain]


def _evaluate_family(
    *,
    snap: FieldSnapshot,
    labels: np.ndarray,
    orders: Mapping[int, np.ndarray],
    horizon_by_label: Mapping[int, int],
    key_values: np.ndarray,
    key_standard_errors: np.ndarray,
    policy: ScopedFieldPolicy,
    tranche_grid: Callable[[np.ndarray, int], tuple[int, ...]],
    route_axis_hash: str,
    resolution_floor: float,
) -> _FamilyEvaluation:
    """Evaluate family size, critical value, and the three bars once.

    ``resolution_floor`` is the smallest damage increment the deciding
    oracle can resolve (the walk's delta_min).  Both derived tolerances
    convert it through the registered price convention — mu-sums live
    in sqrt-damage units ((sum mu)^2 is commensurate with the budget) —
    so ``sqrt(resolution_floor)`` is the mu-mass whose worst-case
    induced damage equals the floor.
    """

    horizon = {label: int(horizon_by_label[label]) for label in (0, 1)}
    cuts_by_label = {
        label: _audited_cuts(
            order=orders[label],
            horizon=horizon[label],
            tranche_grid=tranche_grid,
        )
        for label in (0, 1)
    }
    boundaries_total = sum(len(cuts) for cuts in cuts_by_label.values())
    horizon_total = sum(horizon.values())
    # Family arithmetic mirrors the census construction (route count
    # times sign views plus the ordering coordinate) scoped to the
    # horizon, plus one claim per audited order boundary:
    #   census: len(route_ids) * (len(required_views) + 1)
    #   scoped: |H| * (_SIGN_VIEW_COUNT + 1) + audited boundaries.
    family_size = (
        horizon_total * (_SIGN_VIEW_COUNT + 1) + boundaries_total
    )
    if family_size == 0:
        # No claims exist; an empty family can never resolve, and no
        # critical value is defined for it.
        return _FamilyEvaluation(
            horizon_by_label=horizon,
            cuts_by_label=cuts_by_label,
            family_size=0,
            boundaries_total=0,
            critical=None,
            ambiguous=None,
            sign_price=None,
            sign_pass=False,
            order_receipt=_order_price_receipt(
                resolved=False,
                price_floor=None,
                maximum_cut_price=None,
                note=(
                    "empty enumerated family: no claims exist and the "
                    "look cannot resolve"
                ),
            ),
            order_price=None,
            order_pass=False,
            coverage_by_label={0: 1.0, 1: 1.0},
            coverage_pass=False,
        )
    critical = simultaneous_normal_critical_value(
        policy.confidence_level, family_size=family_size
    )

    # MATERIALITY — a horizon route whose |mu| UCB sits at
    # or below the floor is DECISION-RESOLVED BY IRRELEVANCE: the
    # union-coherence violation of a misassigned member is bounded by
    # |mu_r|, so the aggregate worst-case misassignment damage of every
    # exempt route together is bounded by (sum of exempt mu UCBs)^2 <=
    # resolution_floor — BELOW WHAT THE DECIDING ORACLE CAN RESOLVE, so
    # the exemption cannot change any accept/refuse outcome.  The
    # derived yardstick takes the place of a chosen f_material fraction.
    # The full-family critical (the largest) sets the UCB,
    # so the exemption is conservative.  Materiality exempts the
    # MEAN-term certification only; variance, trust, and coverage paths
    # are untouched.
    resolution_quantum = math.sqrt(max(float(resolution_floor), 0.0))
    materiality_floor = resolution_quantum / float(max(horizon_total, 1))
    mu_ucb = (
        np.abs(snap.global_mean)
        + critical * snap.global_mean_standard_error
    )
    material = mu_ucb > materiality_floor
    horizon_mask = np.zeros(snap.route_count, dtype=bool)
    for label in (0, 1):
        horizon_mask[orders[label][: horizon[label]]] = True
    immaterial_mask = horizon_mask & ~material
    immaterial_route_count = int(np.sum(immaterial_mask))
    immaterial_mu_ucb_total = float(np.sum(mu_ucb[immaterial_mask]))
    if immaterial_mu_ucb_total > resolution_quantum + 1e-12:
        raise RuntimeError(
            "materiality accounting violated its registered bound"
        )

    # WEIGHTING — both bars weigh routes by the DAMAGE-BEARING
    # magnitude mu_ucb (the K1-matched weighting), not the incumbent
    # activity envelope.  The
    # bars' tolerances are mu-mass bounds (a misassigned or misordered
    # route misplaces at most |mu_r|), and activity dominates |mu| by
    # Jensen exactly on cancellation-heavy routes — measured on
    # a canary run: the activity denominator made the
    # derived order tolerance ~16x TIGHTER than a fixed 0.001, both
    # bars failed all 12 looks, and panels escalated geometrically to a
    # 391,916-context request.  Weighting by mu_ucb makes tolerance and
    # observed mass share one unit system; using the UCB (not the point
    # estimate) keeps statistically unresolved routes fully weighted.
    ambiguous = (
        np.abs(snap.global_mean)
        <= critical * snap.global_mean_standard_error
    )
    ambiguous &= snap.global_mean_standard_error > 0.0
    ambiguous &= material
    horizon_activity = float(np.sum(snap.global_activity[horizon_mask]))
    immaterial_activity_fraction = (
        0.0
        if horizon_activity <= _TINY
        else float(np.sum(snap.global_activity[immaterial_mask]))
        / horizon_activity
    )
    damage_weight = mu_ucb * material

    # SIGN BAR — the quadrature price of the ambiguous set:
    # sign statistic = sum of squared mu UCBs
    # over the horizon's ambiguous material routes PLUS the sketch
    # energy of that set, demanded <= the resolution floor.  The signed
    # sum is not used (it cancels toward zero exactly when
    # signs are unknown — a statistic that passes by noise); the
    # linear sum was measured unattainable (z * sum(SE), ~700k routes).
    # Quadrature kills the linear blowup while SE inflation still
    # inflates the statistic, so a noisy field cannot pass vacuously.
    # Disclosed as a CONVERGENCE certificate, not an adversarial-sign
    # worst case.  Exact zero-variance ties are known zero under the
    # declared nonnegative tie class; they are not statistically
    # unresolved.
    ambiguous_indices = np.flatnonzero(horizon_mask & ambiguous)
    _, sign_energy = _damage_price(
        ambiguous_indices,
        global_mean=snap.global_mean,
        factor=snap.rademacher_global_factor,
    )
    sign_price = (
        float(np.sum(mu_ucb[ambiguous_indices] ** 2)) + sign_energy
    )
    sign_pass = sign_price <= float(resolution_floor)
    material_horizon_weight = float(
        np.sum(damage_weight[horizon_mask])
    )

    # ORDER BAR — each audited cut's crossing
    # set is extracted by the retired certificate's exact interval
    # conventions and PRICED with the registered D2 model:
    # (sum mu_hat over the set)^2 + ||F_G 1_S||^2, demanded <= the
    # resolution floor RAW.  The price respects cancellation in the
    # signed mean term, the sketch term prices the covariance mass, and
    # the yardstick is delta_min itself — no fraction language, no
    # denominator.  Noise cannot pass vacuously: mu_hat is unbiased and
    # larger SEs ENLARGE the crossing set and its sketch term.  The
    # only understatement channel (a downward mean-term fluctuation)
    # has scale z^2 * energy / n — constant-factor slack, disclosed per
    # cut as price noise.  Crossing sets are material-only (immaterial
    # misplacement is already covered by the materiality budget).
    # Early exit on the first failing cut keeps early looks cheap.
    if boundaries_total == 0:
        order_receipt = _order_price_receipt(
            resolved=True,
            price_floor=float(resolution_floor),
            maximum_cut_price=0.0,
            note=(
                "no audited interior boundary exists within the "
                "horizon; no order claim is made at any cut"
            ),
        )
        order_price = 0.0
        order_pass = True
    elif material_horizon_weight <= _TINY:
        order_receipt = _order_price_receipt(
            resolved=True,
            price_floor=float(resolution_floor),
            maximum_cut_price=0.0,
            note=(
                "all-immaterial horizon: every audited cut is "
                "separated-by-immateriality (v1.2-A4); no material "
                "order claim exists"
            ),
        )
        order_price = 0.0
        order_pass = True
    else:
        contexts = max(int(snap.contexts_consumed), 1)
        per_cut: list[dict[str, Any]] = []
        order_pass = True
        order_price = 0.0
        for label in (0, 1):
            members = orders[label]
            members = members[material[members]]
            if members.size == 0:
                continue
            for cut in cuts_by_label[label]:
                crossing = _cut_crossing_indices(
                    members,
                    key_values=key_values,
                    key_standard_errors=key_standard_errors,
                    cut=int(cut),
                    critical=critical,
                )
                price, energy = _damage_price(
                    crossing,
                    global_mean=snap.global_mean,
                    factor=snap.rademacher_global_factor,
                )
                per_cut.append(
                    {
                        "partition_label": int(label),
                        "cut": int(cut),
                        "crossing_route_count": int(crossing.size),
                        "price": float(price),
                        "energy": float(energy),
                        "price_noise": float(
                            critical
                            * math.sqrt(max(energy, 0.0) / contexts)
                        ),
                    }
                )
                order_price = max(order_price, float(price))
                if price > float(resolution_floor):
                    order_pass = False
                    break
            if not order_pass:
                break
        order_receipt = _order_price_receipt(
            resolved=order_pass,
            price_floor=float(resolution_floor),
            maximum_cut_price=float(order_price),
            per_cut=tuple(per_cut),
            note=(
                "priced with the registered D2 model over "
                "material-only crossing sets; early exit on first "
                "failing cut (later cuts unaudited that look)"
            ),
        )

    # FD COVERAGE — energy bracket on each partition's horizon-prefix
    # probe action; the empty probe's 0/0 is declared resolved.
    coverage_by_label: dict[int, float] = {}
    for label in (0, 1):
        prefix = horizon[label]
        if prefix <= 0:
            coverage_by_label[label] = 1.0
            continue
        action = np.zeros(snap.route_count, dtype=np.float64)
        action[orders[label][:prefix]] = 1.0
        lower, upper = snap.fd_state.action_energy_bounds(action)
        coverage_by_label[label] = (
            1.0 if upper <= 0.0 else float(lower / upper)
        )
    coverage_pass = all(
        value >= float(policy.sketch_coverage_fraction)
        for value in coverage_by_label.values()
    )

    return _FamilyEvaluation(
        horizon_by_label=horizon,
        cuts_by_label=cuts_by_label,
        family_size=int(family_size),
        boundaries_total=int(boundaries_total),
        critical=float(critical),
        ambiguous=ambiguous,
        sign_price=float(sign_price),
        sign_pass=bool(sign_pass),
        order_receipt=order_receipt,
        order_price=float(order_price),
        order_pass=bool(order_pass),
        coverage_by_label=coverage_by_label,
        coverage_pass=bool(coverage_pass),
        materiality_floor=float(materiality_floor),
        immaterial_route_count=immaterial_route_count,
        immaterial_activity_fraction=float(immaterial_activity_fraction),
        immaterial_mu_ucb_total=float(immaterial_mu_ucb_total),
    )

def _trace_entry(
    *,
    look_index: int,
    phase: str,
    evaluation: _FamilyEvaluation,
    partition_ids: tuple[str, str],
) -> dict[str, Any]:
    return {
        "look_index": int(look_index),
        "phase": str(phase),
        "horizon": {
            partition_ids[label]: int(
                evaluation.horizon_by_label[label]
            )
            for label in (0, 1)
        },
        "family_size": int(evaluation.family_size),
        "materiality_floor": evaluation.materiality_floor,
        "immaterial_route_count": int(evaluation.immaterial_route_count),
        "immaterial_activity_fraction": float(
            evaluation.immaterial_activity_fraction
        ),
        "immaterial_mu_ucb_total": float(
            evaluation.immaterial_mu_ucb_total
        ),
        "normal_critical_value": (
            None
            if evaluation.critical is None
            else float(evaluation.critical)
        ),
        "audited_boundary_count": int(evaluation.boundaries_total),
        "sign_ambiguity_price": (
            None
            if evaluation.sign_price is None
            else float(evaluation.sign_price)
        ),
        "sign_bar_passed": bool(evaluation.sign_pass),
        "order_maximum_cut_price": (
            None
            if evaluation.order_price is None
            else float(evaluation.order_price)
        ),
        "order_bar_passed": bool(evaluation.order_pass),
        "sketch_coverage_minimum": float(
            min(evaluation.coverage_by_label.values())
        ),
        "coverage_bar_passed": bool(evaluation.coverage_pass),
        "resolved": bool(evaluation.all_pass),
    }


def measure_decision_scoped_field(
    *,
    deliver_panel: Callable[[int], None],
    snapshot: Callable[[], FieldSnapshot],
    partition_labeler: Callable[[FieldSnapshot], np.ndarray],
    partition_ids: tuple[str, str],
    ordering_key: str,
    tranche_grid: Callable[[np.ndarray, int], tuple[int, ...]],
    leverage_cap: float,
    policy: ScopedFieldPolicy,
    resolution_floor: float,
    reachable_budget: float | None = None,
    progress: Callable[..., None] | None = None,
) -> ScopedFieldResult:
    """Run the adaptive decision-scoped acquisition loop.

    Injected roles: ``deliver_panel(look_index)`` requests the look's
    additional contexts (the injector owns Neyman allocation, nesting
    and integrity receipts; this module never sees raw rows; look
    indices are 1-based, so ``look_cap`` is the maximum number of
    looks); ``snapshot()`` returns the accumulated ``FieldSnapshot``;
    ``partition_labeler`` labels routes from the CURRENT global-mean
    signs (byte convention: nonnegative -> 0, negative -> 1);
    ``partition_ids[label]`` names each partition (canonical global
    sign ids: ``global_nonnegative`` for label 0, ``global_negative``
    for label 1); ``tranche_grid(order, horizon)`` returns the walk's
    registered Delta-grid cut COUNTS (pruned-prefix sizes) within the
    horizon, receiving the partition's pruned-first route-index order.

    Per-look loop (binding order): panel -> snapshot -> key statistic ->
    labels/orders -> horizon from the reachable budget -> family size
    and critical -> sign bar -> order bar (tolerance derived per look
    from ``resolution_floor``) -> FD coverage -> stop or next look.
    ``resolution_floor`` is the smallest damage increment the deciding
    oracle can resolve (the walk's delta_min; callers without a
    measured paired spread seed it from the dense baseline
    certificate's spread).  A look-cap hit is disclosed as an
    uncertified bound rail and never raises.  Determinism: identical
    injected snapshots produce identical results (this module holds no
    RNG).

    All receipt coverage statements are conditional on selection over
    the final enumerated family; see the module
    docstring.
    """

    policy.validate()
    if str(ordering_key) not in V1_ORDERING_KEYS:
        raise ValueError("unregistered scoped-field ordering key")
    identifiers = tuple(map(str, partition_ids))
    if len(identifiers) != 2 or len(set(identifiers)) != 2 or not all(
        identifiers
    ):
        raise ValueError(
            "scoped-field partition_ids must be two distinct names"
        )
    cap = float(leverage_cap)
    if not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("scoped-field leverage_cap must be positive")

    trace: list[dict[str, Any]] = []
    resolved = False
    stopping_reason = "unresolved"
    terminal_reason: str | None = None

    snap: FieldSnapshot | None = None
    labels: np.ndarray | None = None
    orders: dict[int, np.ndarray] | None = None
    key_values: np.ndarray | None = None
    key_errors: np.ndarray | None = None
    final_evaluation: _FamilyEvaluation | None = None
    looks_taken = 0
    # Runtime only: the route axis is fixed for the whole acquisition,
    # and hashing 1.39M route ids canonicalizes ~2.3s per look at
    # cluster scale (measured).  Hash
    # once and reuse while the ids tuple is the SAME OBJECT — identity
    # implies equal bytes, and a fresh object recomputes exactly.
    hashed_route_ids: Any | None = None
    route_axis_hash: str | None = None

    floor = float(resolution_floor)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError(
            "scoped-field resolution_floor must be positive"
        )
    for look_index in range(1, int(policy.look_cap) + 1):
        looks_taken = look_index
        deliver_panel(look_index)
        snap = snapshot()
        if not isinstance(snap, FieldSnapshot):
            raise TypeError(
                "scoped-field snapshot callable must return a "
                "FieldSnapshot"
            )
        if (
            route_axis_hash is None
            or snap.fd_state.route_ids is not hashed_route_ids
        ):
            hashed_route_ids = snap.fd_state.route_ids
            route_axis_hash = content_hash(
                hashed_route_ids, prefix="prune-order-route-axis-"
            )
        key_values, key_errors = ordering_key_statistic(
            str(ordering_key),
            GlobalHeadOrderingField.from_statistics(
                global_mean=snap.global_mean,
                global_activity=snap.global_activity,
                statistics=snap.statistics,
            ),
        )
        raw_labels = np.asarray(partition_labeler(snap)).reshape(-1)
        if (
            raw_labels.size != snap.route_count
            or raw_labels.dtype.kind not in {"i", "u", "b"}
            or not np.isin(raw_labels, (0, 1)).all()
        ):
            raise ValueError(
                "scoped-field partition labels must be 0/1 on the "
                "route axis"
            )
        labels = raw_labels.astype(np.int8)
        orders = _partition_orders(labels, key_values)

        horizon_by_label: dict[int, int] = {}
        for label in (0, 1):
            if reachable_budget is not None:
                # On the walk path the horizon derives from the
                # budget ALONE — the walk can never legitimately travel
                # further than the remaining budget within a chart era.
                # The FD-leverage arm (and its 0.1 constant) survives
                # only as the no-budget fallback for stub tests.
                admissible = _price_reachable_prefix(
                    order=orders[label],
                    global_mean=snap.global_mean,
                    factor=snap.rademacher_global_factor,
                    reachable_budget=float(reachable_budget),
                )
            else:
                admissible = _admissible_prefix_length(
                    order=orders[label],
                    fd_state=snap.fd_state,
                    leverage_cap=cap,
                )
            # The horizon IS the admissible prefix, with no margin:
            # with the budget-priced prefix recomputed every look, a
            # margin's only effect would be doubling the enumerated family.
            horizon_by_label[label] = min(
                int(admissible), int(orders[label].size)
            )

        evaluation = _evaluate_family(
            snap=snap,
            labels=labels,
            orders=orders,
            horizon_by_label=horizon_by_label,
            key_values=key_values,
            key_standard_errors=key_errors,
            policy=policy,
            tranche_grid=tranche_grid,
            route_axis_hash=route_axis_hash,
            resolution_floor=floor,
        )
        trace.append(
            _trace_entry(
                look_index=look_index,
                phase="primary",
                evaluation=evaluation,
                partition_ids=identifiers,
            )
        )
        if progress is not None:
            # Per-bar disclosure in the LOG, not only the receipt: a
            # cancelled run can leave hours of logs and no
            # receipts, and which bar was binding would be unrecoverable.
            progress(
                "scoped_field_look_completed",
                look_index=look_index,
                phase="primary",
                family_size=evaluation.family_size,
                resolved=evaluation.all_pass,
                sign_bar_passed=evaluation.sign_pass,
                order_bar_passed=evaluation.order_pass,
                coverage_bar_passed=evaluation.coverage_pass,
                sign_ambiguity_price=evaluation.sign_price,
                order_maximum_cut_price=evaluation.order_price,
                price_floor=resolution_floor,
            )
        # ACQUIRE UNTIL THE ORDER IS CERTIFIED WHERE IT IS CONSUMED.
        # The field exists to RANK routes, so its precision requirement is
        # a statement about ranking, not a context count: the tranche
        # boundaries inside the budget-reachable horizon must separate at
        # the registered confidence.  That is the same buy-until-resolved
        # discipline every other decision in the walk follows, and this
        # loop was the one place that ignored it — it stopped after a
        # structural three looks and merely DISCLOSED the certificate it
        # had already computed.  Consequence, measured: charts were
        # acquired on 384 contexts, so the ranking fed to every proposal
        # carried large per-route error, and v1's damage-vs-depth curve sat
        # ABOVE v0's despite relinearizing 16-28 times.  Freshness only
        # beats a stale dense-state field if each refresh is trustworthy.
        # Contexts now rise and fall on their own: cheap where routes are
        # well separated, expensive near the frontier where misranking
        # actually costs damage.  look_cap remains the rail, and a bound
        # rail is disclosed rather than silently treated as certified.
        final_evaluation = evaluation
        if evaluation.all_pass:
            resolved = True
            stopping_reason = "order_certified_over_reachable_horizon"
            break

    if not resolved:
        # The rail bound before the ordering certified. The chart is still
        # usable — validity lives in the walk's per-commit exact evidence —
        # but the receipt must say the ordering is UNCERTIFIED rather than
        # imply the acquisition converged.
        resolved = True
        stopping_reason = "look_cap_reached_order_uncertified"
    assert snap is not None and final_evaluation is not None
    assert labels is not None and orders is not None
    assert key_values is not None and key_errors is not None

    horizon_receipt = {
        identifiers[label]: int(final_evaluation.horizon_by_label[label])
        for label in (0, 1)
    }
    partition_sizes = {
        identifiers[label]: int(orders[label].size) for label in (0, 1)
    }
    coverage_observed = {
        identifiers[label]: float(
            final_evaluation.coverage_by_label[label]
        )
        for label in (0, 1)
    }
    fd_metadata = snap.fd_state.metadata
    fd_receipt: dict[str, Any] = {
        "rank": int(snap.fd_state.rank),
        "requested_rank": int(
            fd_metadata.get("requested_rank", snap.fd_state.rank)
        ),
        "residual_spectral_bound": float(
            snap.fd_state.residual_spectral_bound
        ),
        "residual_diagonal_bound_available": (
            snap.fd_state.residual_diagonal_bound is not None
        ),
        "residual_diagonal_bound_maximum": (
            None
            if snap.fd_state.residual_diagonal_bound is None
            else float(np.max(snap.fd_state.residual_diagonal_bound))
        ),
        "context_count": int(snap.fd_state.context_count),
        "measured_mask_hash": str(snap.fd_state.measured_mask_hash),
        "route_count": int(len(snap.fd_state.route_ids)),
        "snapshot_strategy": {
            "strategy": FD_SNAPSHOT_STRATEGY,
            "reason": _FD_SNAPSHOT_REASON,
        },
    }
    if not final_evaluation.coverage_pass:
        # Coverage deficiency is a disclosure at the stability stop
        # and becomes a rank-growth obligation for the next chart —
        # never a gate (the deterministic envelope already makes low
        # coverage conservative, not unsafe).
        next_rank = snap.fd_state.next_decision_relevant_rank()
        fd_receipt["rank_growth_obligation"] = {
            "reason": (
                "sketch_coverage_below_required_at_stop"
            ),
            "current_rank": int(
                fd_metadata.get("requested_rank", snap.fd_state.rank)
            ),
            "requested_rank": (
                None if next_rank is None else int(next_rank)
            ),
            "maximum_rank": int(snap.fd_state.spec.maximum_rank),
            "at_maximum_rank": next_rank is None,
        }

    receipt: dict[str, Any] = {
        "schema": SCOPED_FIELD_RECEIPT_SCHEMA,
        "measurement_class_id": SCOPED_FIELD_CLASS_ID,
        "policy": policy.to_receipt(),
        "leverage_cap": float(cap),
        "ordering_key": str(ordering_key),
        "ordering_key_registry": ORDERING_KEY_REGISTRY_VERSION,
        "partition_ids": list(identifiers),
        "partition_sizes": partition_sizes,
        "looks_taken": int(looks_taken),
        "stopping_rule": "chart_stability_v1_3",
        "certification_disclosure": {
            "sign_bar_passed": (
                None
                if final_evaluation is None
                else bool(final_evaluation.sign_pass)
            ),
            "order_bar_passed": (
                None
                if final_evaluation is None
                else bool(final_evaluation.order_pass)
            ),
            "note": (
                "proposal-side disclosure only: chart stopping is "
                "stability + FD coverage; commit validity rests on the "
                "walk's exact paired evidence"
            ),
        },
        "look_cap": int(policy.look_cap),
        "look_indices_one_based": True,
        "resolution_floor": float(floor),
        "standalone_vs_marginal_interaction_bound": (
            None
            if reachable_budget is None
            else float(
                2.0 * math.sqrt(max(float(reachable_budget), 0.0) * floor)
            )
        ),
        "family_trace": trace,
        "horizon": horizon_receipt,
        "sign_ambiguity_price": (
            None
            if final_evaluation.sign_price is None
            else float(final_evaluation.sign_price)
        ),
        # The priced-bars design: both
        # bars are absolute damage prices demanded <= resolution_floor.
        # Sign = quadrature mu-UCB energy + sketch energy of the
        # ambiguous set (a CONVERGENCE certificate, not an
        # adversarial-sign worst case); order = registered D2 price of
        # each audited cut's crossing set.
        "bar_semantics": "priced_bars_resolution_floor_v1",
        "order_price_scope_note": (
            "price(S) bounds the crossing set's own damage; interaction "
            "with the committed base is bounded separately (see "
            "standalone_vs_marginal_interaction_bound) and is tolerable "
            "solely because commits are oracle-adjudicated"
        ),
        "fd_coverage_scope_note": (
            "FD sketch coverage brackets the FD envelope only, not the "
            "Rademacher factor inside the price"
        ),
        "materiality_rule": "sign_or_order_exemption_mean_term_only_v1",
        "materiality_floor": final_evaluation.materiality_floor,
        "immaterial_route_count": int(
            final_evaluation.immaterial_route_count
        ),
        "immaterial_activity_fraction": float(
            final_evaluation.immaterial_activity_fraction
        ),
        "immaterial_mu_ucb_total": float(
            final_evaluation.immaterial_mu_ucb_total
        ),
        "order_maximum_cut_price": (
            None
            if final_evaluation.order_price is None
            else float(final_evaluation.order_price)
        ),
        "audited_boundary_count": int(
            final_evaluation.boundaries_total
        ),
        "order_certificate": final_evaluation.order_receipt,
        "sketch_coverage": {
            "required_fraction": float(
                policy.sketch_coverage_fraction
            ),
            "observed_by_partition": coverage_observed,
            "minimum_observed": float(min(coverage_observed.values())),
        },
        "contexts_consumed": int(snap.contexts_consumed),
        "census_exhausted": bool(snap.census_exhausted),
        "fd": fd_receipt,
        "rademacher_global_factor_hash": content_hash(
            snap.rademacher_global_factor,
            prefix="trust-walk-rademacher-global-factor-",
        ),
        "rademacher_global_factor_role": (
            "carried_untouched_for_walk_pricing_never_horizon_or_"
            "coverage"
        ),
        "resolved": bool(resolved),
        "stopping_reason": str(stopping_reason),
        "family_selection": SCOPED_FIELD_FAMILY_SELECTION,
        "coverage": SCOPED_FIELD_COVERAGE_CLAIM,
        "claim": (
            "per-claim normal intervals at the stated Bonferroni level "
            "over the final enumerated family; the family is "
            "data-selected and coverage is conditional on selection; no "
            "unconditional simultaneous coverage over any a-priori "
            "family, no signs for unresolved routes, no order at "
            "unaudited cuts, and no alpha transfers to arm-level "
            "conclusions"
        ),
        "role": "proposal_side_disclosure",
        "era_separation": (
            "scoped-class receipts never pool with census-grade "
            "measurement receipts"
        ),
    }

    if progress is not None:
        progress(
            "scoped_field_measurement_completed",
            resolved=resolved,
            looks_taken=looks_taken,
            stopping_reason=stopping_reason,
        )

    return ScopedFieldResult(
        snapshot=snap,
        partition_labels=labels,
        key_values=key_values,
        key_standard_errors=key_errors,
        horizon=horizon_receipt,
        order_certificate=final_evaluation.order_receipt,
        receipt=receipt,
        looks_taken=looks_taken,
    )


__all__ = [
    "FD_SNAPSHOT_STRATEGY",
    "FieldSnapshot",
    "SCOPED_FIELD_CLASS_ID",
    "SCOPED_FIELD_COVERAGE_CLAIM",
    "SCOPED_FIELD_FAMILY_SELECTION",
    "SCOPED_FIELD_RECEIPT_SCHEMA",
    "ScopedFieldPolicy",
    "ScopedFieldResult",
    "finalize_fd_snapshot",
    "measure_decision_scoped_field",
]
