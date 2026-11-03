"""Uncertainty certificates for activity-ranked pruning orders.

The hard-partition scorer has two separate obligations:

* a signed local/global mean determines the hard partition label;
* the metric-global activity ``E[||G_r||]`` determines route order inside a
  partition.

This module owns the second obligation. It never materializes route-pair
matrices and it does not demand that nearly equal interior routes be totally
ordered. The current operational certificate compares successive physical
proposal masks at declared retained-density cutoffs. The older simultaneous
interval certificate remains available for provenance and focused audits.
Both certificates control mask membership rather than a scientifically
meaningless total order among routes that remain on the same side of a
proposal boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np

from .contracts import content_hash


PRUNE_ORDER_CERTIFICATE_V2 = "prune_order_uncertainty_certificate_v2"
PRUNE_ORDER_STABILITY_CERTIFICATE_V1 = (
    "prune_order_mask_stability_certificate_v1"
)
PRUNE_ORDER_STABILITY_CERTIFICATE_V2 = (
    "prune_order_active_boundary_stability_certificate_v2"
)
# v3 adds the per-scope atom diagnostic. The disagreement measure sums the
# activity of WHOLE routes, so its smallest nonzero value in a scope is one
# route's activity share -- the scope's "atom". When that atom exceeds the
# declared tolerance the bar is unsatisfiable by quantization rather than by
# sampling, and no additional evidence can change that. MEASURED at coarse
# (cp) granularity: features.stage1.conv holds 192 routes carrying 3.69% of
# global activity, and its largest single route is 0.001423 -- 142% of the
# 0.001 budget on its own. At kernel (kt) granularity the same layer holds
# 23,232 routes, the largest is 1.05e-4, and the budget first admits 13.
# Reporting only: nothing gates on these fields.
PRUNE_ORDER_STABILITY_CERTIFICATE_V3 = (
    "prune_order_active_boundary_stability_certificate_v3"
)
STABLE_TIE_POLICY_V1 = "route_axis_ascending_exact_tie_v1"

_LAYER_CODE_DTYPE = np.int16
_LAYER_CODE_CAPACITY = int(np.iinfo(_LAYER_CODE_DTYPE).max) + 1
_FLOAT64_SIGN_FLIP_MASK = np.int64(0x7FFFFFFFFFFFFFFF)


@dataclass(frozen=True, slots=True, eq=False)
class LayerCodeAxis:
    """Layer-id companion of a route axis, stored as compact integer codes.

    ``codes[i]`` is the position in ``names`` of route ``i``'s layer id, and
    ``names`` lists the distinct layer ids in order of FIRST APPEARANCE on
    the axis -- exactly the order ``dict.fromkeys(map(str, layer_ids))``
    yields -- so scope keys and scope ids built from ``names`` are
    byte-identical to a string-object path.  Per-scope layer
    membership is ``codes == code`` on an int16 array instead of
    ``layers == layer`` on a 57M-element object array, which would be the
    dominant host cost of every census look.

    The class is a read-only sequence of layer-id strings for legacy callers
    (``len``, iteration, indexing, equality with a string tuple), but every
    hot consumer reads ``codes``/``names`` directly.
    """

    codes: np.ndarray
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        codes = np.asarray(self.codes)
        if codes.ndim != 1 or codes.dtype != _LAYER_CODE_DTYPE:
            raise ValueError("layer code axis requires a rank-1 int16 code array")
        names = tuple(str(name) for name in self.names)
        if len(set(names)) != len(names):
            raise ValueError("layer code axis names must be unique")
        if len(names) > _LAYER_CODE_CAPACITY:
            raise ValueError("layer code axis exceeds int16 code capacity")
        if codes.size:
            if int(codes.min()) < 0 or int(codes.max()) >= len(names):
                raise ValueError("layer code axis codes fall outside the name table")
            # First-appearance order is the invariant that makes ``names``
            # equal to ``dict.fromkeys`` over the expanded strings.  Run
            # boundaries suffice: a code's first appearance is the start of
            # its first run.
            boundary = np.empty(codes.size, dtype=bool)
            boundary[0] = True
            np.not_equal(codes[1:], codes[:-1], out=boundary[1:])
            run_codes = codes[boundary]
            first_seen = tuple(dict.fromkeys(int(code) for code in run_codes))
            if first_seen != tuple(range(len(names))):
                raise ValueError(
                    "layer code axis names must be listed in first-appearance order"
                )
        elif names:
            raise ValueError("an empty layer code axis cannot name layers")
        if not codes.flags.c_contiguous:
            codes = np.ascontiguousarray(codes)
        codes = codes.view()
        codes.setflags(write=False)
        object.__setattr__(self, "codes", codes)
        object.__setattr__(self, "names", names)

    # -- constructors ---------------------------------------------------

    @classmethod
    def from_layer_ids(cls, layer_ids: "Sequence[str] | LayerCodeAxis") -> "LayerCodeAxis":
        """Coerce a legacy per-route string sequence (or pass an axis through)."""

        if isinstance(layer_ids, cls):
            return layer_ids
        values = np.asarray(tuple(map(str, layer_ids)), dtype=object).reshape(-1)
        if values.size == 0:
            return cls(codes=np.zeros(0, dtype=_LAYER_CODE_DTYPE), names=())
        boundary = np.empty(values.size, dtype=bool)
        boundary[0] = True
        np.not_equal(values[1:], values[:-1], out=boundary[1:])
        starts = np.flatnonzero(boundary)
        run_names = tuple(str(value) for value in values[starts])
        run_lengths = np.diff(np.append(starts, values.size))
        return cls.from_blocks(zip(run_names, run_lengths.tolist(), strict=True))

    @classmethod
    def from_blocks(cls, blocks: Iterable[tuple[str, int]]) -> "LayerCodeAxis":
        """Build from ``(layer_id, route_count)`` blocks in axis order."""

        names: dict[str, int] = {}
        block_codes: list[int] = []
        block_counts: list[int] = []
        for layer_id, count in blocks:
            name = str(layer_id)
            size = int(count)
            if size < 0:
                raise ValueError("layer code axis block counts must be nonnegative")
            if size == 0:
                continue
            block_codes.append(names.setdefault(name, len(names)))
            block_counts.append(size)
        if len(names) > _LAYER_CODE_CAPACITY:
            raise ValueError("layer code axis exceeds int16 code capacity")
        codes = np.repeat(
            np.asarray(block_codes, dtype=_LAYER_CODE_DTYPE),
            np.asarray(block_counts, dtype=np.int64),
        )
        return cls(codes=codes, names=tuple(names))

    # -- hot-path accessors ---------------------------------------------

    @property
    def route_count(self) -> int:
        return int(self.codes.size)

    def code_of(self, name: str) -> int:
        try:
            return self.names.index(str(name))
        except ValueError as error:
            raise KeyError(str(name)) from error

    def membership(self, name: str) -> np.ndarray:
        """Boolean route mask of ``name``; all-False for an unknown layer."""

        if str(name) not in self.names:
            return np.zeros(self.codes.size, dtype=bool)
        return self.codes == _LAYER_CODE_DTYPE(self.code_of(name))

    def block_matches(self, name: str, start: int, stop: int) -> bool:
        """True when every route in ``[start, stop)`` carries layer ``name``."""

        if str(name) not in self.names:
            return int(stop) <= int(start)
        code = _LAYER_CODE_DTYPE(self.code_of(name))
        return bool(np.all(self.codes[int(start) : int(stop)] == code))

    # -- legacy string-sequence view --------------------------------------

    def as_strings(self) -> tuple[str, ...]:
        if not self.codes.size:
            return ()
        table = np.asarray(self.names, dtype=object)
        return tuple(table[self.codes])

    def __len__(self) -> int:
        return int(self.codes.size)

    def __iter__(self):
        table = np.asarray(self.names, dtype=object)
        return iter(table[self.codes].tolist()) if self.codes.size else iter(())

    def __getitem__(self, item):
        if isinstance(item, slice):
            if not self.codes.size:
                return ()
            table = np.asarray(self.names, dtype=object)
            return tuple(table[self.codes[item]])
        return self.names[int(self.codes[item])]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LayerCodeAxis):
            return self.names == other.names and np.array_equal(self.codes, other.codes)
        if isinstance(other, (tuple, list)):
            if len(other) != self.codes.size:
                return False
            return self == LayerCodeAxis.from_layer_ids(other)
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]


def _stable_argsort(values: np.ndarray, *, descending: bool) -> np.ndarray:
    """Exact replacement for ``np.argsort(values, kind="stable")``.

    Invariant: the stable sorting permutation of a NaN-free float axis is
    UNIQUE, so any stable algorithm returns identical indices.  The values
    are mapped to a monotone int64 key (the classic sign-flip transform,
    after ``+ 0.0`` folds ``-0.0`` into ``+0.0`` so equal floats have equal
    keys) and sorted with ``torch.sort(stable=True)``, whose 1-D int64 path
    is the fastest stable sort available on the census host (parallel radix
    on FBGEMM builds; measured 3.6 s vs numpy's 6.4 s for 57M routes on the
    development machine).  ``descending=True`` mirrors
    ``np.argsort(-values, kind="stable")`` exactly: negate, then ascending.
    Any NaN (never present after the callers' finiteness checks) falls back
    to numpy so NaN placement stays numpy's.
    """

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size <= 1:
        return np.arange(array.size, dtype=np.int64)
    keyed = (-array if descending else array) + 0.0
    if np.isnan(keyed).any():
        return np.argsort(keyed, kind="stable")
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a runtime dependency here
        return np.argsort(keyed, kind="stable")
    key = keyed.view(np.int64)
    negative = key < 0
    key[negative] ^= _FLOAT64_SIGN_FLIP_MASK
    _, indices = torch.sort(torch.from_numpy(key), stable=True)
    return indices.numpy()


@dataclass(frozen=True, slots=True)
class PruneOrderScopeReceipt:
    """Order-uncertainty summary for one executable partition scope."""

    scope_id: str
    route_count: int
    uncertain_boundary_count: int
    uncertain_route_count: int
    exact_tie_block_count: int
    maximum_boundary_uncertain_route_count: int
    maximum_boundary_route_fraction: float
    maximum_boundary_activity_fraction: float
    mean_boundary_activity_fraction: float
    audited_retained_fractions: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.scope_id or int(self.route_count) <= 0:
            raise ValueError("prune-order scope identity is invalid")
        integer_values = (
            self.uncertain_boundary_count,
            self.uncertain_route_count,
            self.exact_tie_block_count,
            self.maximum_boundary_uncertain_route_count,
        )
        if any(int(value) < 0 for value in integer_values):
            raise ValueError("prune-order scope counts must be nonnegative")
        fractions = (
            self.maximum_boundary_route_fraction,
            self.maximum_boundary_activity_fraction,
            self.mean_boundary_activity_fraction,
        )
        if any(
            not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
            for value in fractions
        ):
            raise ValueError("prune-order scope fractions must lie in [0,1]")
        if any(
            not math.isfinite(float(value)) or not 0.0 < float(value) < 1.0
            for value in self.audited_retained_fractions
        ):
            raise ValueError(
                "audited prune-order retained fractions must lie in (0,1)"
            )


@dataclass(frozen=True, slots=True)
class PruneOrderUncertaintyCertificate:
    """Bound the mask instability induced by uncertain activity ordering."""

    confidence_level: float
    normal_critical_value: float
    maximum_permitted_boundary_activity_fraction: float
    maximum_observed_boundary_activity_fraction: float
    scope_receipts: tuple[PruneOrderScopeReceipt, ...]
    route_axis_hash: str
    tie_policy_id: str = STABLE_TIE_POLICY_V1
    schema: str = PRUNE_ORDER_CERTIFICATE_V2

    def __post_init__(self) -> None:
        if (
            not 0.5 < float(self.confidence_level) < 1.0
            or not math.isfinite(float(self.normal_critical_value))
            or float(self.normal_critical_value) <= 0.0
            or not 0.0
            <= float(self.maximum_permitted_boundary_activity_fraction)
            <= 1.0
            or not 0.0
            <= float(self.maximum_observed_boundary_activity_fraction)
            <= 1.0
            or not self.scope_receipts
            or not self.route_axis_hash
            or not self.tie_policy_id
        ):
            raise ValueError("prune-order uncertainty certificate is invalid")

    @property
    def resolved(self) -> bool:
        return (
            self.maximum_observed_boundary_activity_fraction
            <= self.maximum_permitted_boundary_activity_fraction
        )

    @property
    def content_hash(self) -> str:
        return content_hash(self, prefix="irg-prune-order-uncertainty-")

    def to_receipt(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "confidence_level": float(self.confidence_level),
            "normal_critical_value": float(self.normal_critical_value),
            "maximum_permitted_boundary_activity_fraction": float(
                self.maximum_permitted_boundary_activity_fraction
            ),
            "maximum_observed_boundary_activity_fraction": float(
                self.maximum_observed_boundary_activity_fraction
            ),
            "decision_resolved": self.resolved,
            "route_axis_hash": self.route_axis_hash,
            "tie_policy_id": self.tie_policy_id,
            "scope_receipts": tuple(
                {
                    "scope_id": item.scope_id,
                    "route_count": int(item.route_count),
                    "uncertain_boundary_count": int(
                        item.uncertain_boundary_count
                    ),
                    "uncertain_route_count": int(item.uncertain_route_count),
                    "exact_tie_block_count": int(
                        item.exact_tie_block_count
                    ),
                    "maximum_boundary_uncertain_route_count": int(
                        item.maximum_boundary_uncertain_route_count
                    ),
                    "maximum_boundary_route_fraction": float(
                        item.maximum_boundary_route_fraction
                    ),
                    "maximum_boundary_activity_fraction": float(
                        item.maximum_boundary_activity_fraction
                    ),
                    "mean_boundary_activity_fraction": float(
                        item.mean_boundary_activity_fraction
                    ),
                    "audited_retained_fractions": tuple(
                        float(value)
                        for value in item.audited_retained_fractions
                    ),
                }
                for item in self.scope_receipts
            ),
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True, slots=True)
class PruneOrderStabilityScopeReceipt:
    """Successive-look mask disagreement at audited executable cutoffs."""

    scope_id: str
    route_count: int
    partition_changed_route_count: int
    audited_boundary_count: int
    maximum_mask_disagreement_route_fraction: float
    maximum_mask_disagreement_activity_fraction: float
    mean_mask_disagreement_activity_fraction: float
    # Atom diagnostic (v3). ``maximum_route_activity_fraction`` is the largest
    # single route's share of this scope's disagreement denominator, i.e. the
    # smallest nonzero step the measure above can take. A boundary SWAP moves
    # one route out and one in, so twice the atom is the smallest tolerance a
    # single swap can respect -- reported as ``minimum_feasible_tolerance``.
    maximum_route_activity_fraction: float = 0.0
    minimum_feasible_tolerance: float = 0.0
    # Churn attribution (v3): a disjoint split of
    # ``maximum_mask_disagreement_activity_fraction`` at the binding cut.
    # Membership churn is driven by unresolved SIGNS (a route changed
    # partition), which the sign bar licenses at a 50x looser tolerance;
    # score-order movement is the instability this bar is actually about.
    #
    # CAVEAT, so the split is not over-read: attribution is BY ROUTE, not by
    # cause. A membership change also alters the partition's cardinality, so
    # ``ceil(fraction * size)`` can move the cut by one and push a route with
    # stable membership across it. That route is counted as score-order
    # movement although churn caused it, so the churn term is a LOWER bound
    # on churn's contribution. Measured on a synthetic pure-churn panel
    # (identical scores, labels perturbed): churn 0.0202, order 0.00095.
    # The converse is clean -- with labels held fixed the churn term is
    # exactly zero.
    membership_churn_activity_fraction: float = 0.0
    score_order_activity_fraction: float = 0.0
    # Consequence price (v3, reporting only). The metric counterpart to the
    # mass-based disagreement above: how much the DAMAGE could differ across
    # orderings the data cannot distinguish, in the decision's own units,
    # alongside its own uncertainty so it can be judged against a DERIVED
    # floor rather than a declared numeral. Zero when the caller supplies no
    # damage field.
    consequence_price: float = 0.0
    consequence_price_standard_error: float = 0.0
    consequence_band_size: int = 0
    consequence_binding_retained_fraction: float = 0.0
    # D(M_k) at every audited fraction: the damage of the POINT-ESTIMATE mask
    # (top-k by score), so the realized consequence |D(M_k) - D(M_{k-1})| can
    # be formed across looks at matched fractions.
    # Reporting only. Empty when pricing was skipped or unsupplied.
    consequence_decision_functional_by_fraction: tuple[
        tuple[float, float], ...
    ] = ()
    # Deferred per-look pricing. True when the
    # price was SKIPPED this look because the sign field was far from
    # resolution -- the bands would span most of the axis and the certificate
    # would price noise, expensively. Disclosed, never silent.
    consequence_price_deferred: bool = False


@dataclass(frozen=True, slots=True)
class PruneOrderStabilityCertificate:
    """Activity-weighted stability at the controller's audited cutoffs."""

    maximum_permitted_mask_disagreement_activity_fraction: float
    maximum_observed_mask_disagreement_activity_fraction: float
    audited_retained_fractions: tuple[float, ...]
    scope_receipts: tuple[PruneOrderStabilityScopeReceipt, ...]
    route_axis_hash: str
    previous_look_available: bool
    activity_normalization: str = "scope"
    schema: str = PRUNE_ORDER_STABILITY_CERTIFICATE_V3
    # Consequence gating (v4). When enabled, `resolved` tests the summed
    # consequence price against its OWN derived floor instead of the activity
    # mass fraction against a declared numeral. Default False keeps every
    # existing arm byte-identical, so enabling it is an explicit act.
    # DEPRECATED (default off). Do not arm.
    #
    # MEASURED in production on five arms spanning both granularities and
    # four families: consequence_price_total 114.5-118.8 against a
    # consequence_floor of 0.008-0.024, consequence_resolved FALSE on every one,
    # while mass_resolved was TRUE at kt. Arming it was STRICTLY WORSE than the
    # criterion it replaces -- at kt the incumbent resolved and this blocked it,
    # at cp neither resolved, and nothing completed.
    #
    # THE DEFECT IS THE FLOOR, NOT THE PRICE. `consequence_resolved` tests
    # price <= critical * standard_error, and that floor SHRINKS as sampling
    # improves while the price -- a genuine damage difference, not noise -- does
    # not. The gate therefore gets HARDER with more data and resolves only when
    # the price is exactly zero. It asks "is there any consequence at all?"
    # where the decision needs "is the consequence small enough to ignore?".
    #
    # The PRICE is retained and still receipted: it is measured precisely
    # (SE ~0.0008 on ~115) and varies by family at completion, so the campaign
    # supplies the distribution needed to derive a MATERIALITY tolerance. That
    # tolerance is a numeral fixed in advance.
    # Replacement: a scale-free floor, price/total_damage against a declared
    # tolerance, parallel to the mass measure.
    consequence_gated: bool = False
    consequence_floor_critical_value: float = 1.959963984540054
    consequence_price_deferred: bool = False
    # DEFAULT-OFF. Delta is the deciding oracle's
    # resolvable increment (pooled p50 = 3.5e-3 over 5,571 walk decisions),
    # compared on the PER-DECISION MEAN basis (price/|B| <= delta per
    # scope). None = unarmed: the census-precision noise floor decides,
    # byte-for-byte unchanged.
    consequence_materiality_floor: float | None = None
    # True iff the price block actually ran this look (damage supplied and
    # not deferred) -- so an unpriced look can never read as resolved.
    consequence_price_evaluated: bool = False
    # While the price is deferred, a CHEAP CONSERVATIVE bound on the
    # per-decision mean price the deferral is skipping: 2x the largest
    # single-route |damage| (each undetermined decision's contribution to
    # the band price is bounded by one in/out swap of the largest route).
    # Discloses that the omitted price could not already bite. None when never deferred or no damage supplied.
    consequence_deferred_per_decision_bound: float | None = None

    @property
    def consequence_price_total(self) -> float:
        """Summed price over scopes -- a SUM, not a max (see below).

        The aggregate is the diameter of the decision functional over the whole
        mask. For damage additive across scopes that is a sum; a max over
        scopes maximises quantities that were never commensurable. Summing is
        also the conservative choice while additivity is unmeasured, since it
        can only over-state relative to a max.
        """

        return float(
            sum(float(item.consequence_price) for item in self.scope_receipts)
        )

    @property
    def consequence_price_standard_error(self) -> float:
        """Root-sum-square of the per-scope price errors.

        Scopes are disjoint route sets, so their price errors carry no shared
        route; the calibration-common component that would NOT add in
        quadrature does not arise here, because `d_r` is a directly measured
        head mean rather than a fitted projection.
        """

        return float(
            math.sqrt(
                sum(
                    float(item.consequence_price_standard_error) ** 2
                    for item in self.scope_receipts
                )
            )
        )

    @property
    def consequence_resolved(self) -> bool:
        """Is the ordering ambiguity distinguishable from none at all?

        The DERIVED alternative to a declared tolerance. A tie prices exactly
        zero and can never exceed its own uncertainty, so the census stops
        rather than chasing an ambiguity with no consequence -- which is the
        configuration a fixed activity-mass numeral can render permanently
        infeasible.

        Requires a positive standard error: with no damage field supplied the
        price is identically zero and this must not read as resolution.
        """

        if self.consequence_materiality_floor is not None:
            # Materiality regime: a FIXED consequence scale on the
            # per-decision mean basis, monotone-easier as evidence
            # accumulates.
            # Deferred or unevaluated looks can never read as resolved.
            if not self.consequence_price_evaluated:
                return False
            floor = float(self.consequence_materiality_floor)
            for item in self.scope_receipts:
                if item.consequence_price_deferred:
                    return False
                band = int(item.consequence_band_size)
                if band > 0 and (
                    float(item.consequence_price) / float(band) > floor
                ):
                    return False
            return True
        error = self.consequence_price_standard_error
        if not (error > 0.0):
            return False
        return self.consequence_price_total <= (
            float(self.consequence_floor_critical_value) * error
        )

    @property
    def tolerance_infeasible_scope_ids(self) -> tuple[str, ...]:
        """Scopes whose own atom exceeds the declared tolerance.

        Diagnostic only -- it does not enter ``resolved``. A nonempty tuple
        says the bar cannot be satisfied in those scopes by any amount of
        additional evidence, because a single boundary swap already costs
        more than the whole budget. That is a granularity or tolerance
        finding, not a sampling one.
        """

        return tuple(
            item.scope_id
            for item in self.scope_receipts
            if item.minimum_feasible_tolerance
            > self.maximum_permitted_mask_disagreement_activity_fraction
        )

    @property
    def resolved(self) -> bool:
        """Whether the pruning order is certified.

        Two regimes, selected by ``consequence_gated``:

        * **mass** (default, unchanged) -- the look-to-look activity fraction
          against a declared numeral. Requires a previous look, so the first is
          structurally unresolved, and at coarse granularity it is unsatisfiable
          by arithmetic whenever one route's activity exceeds the whole budget.
        * **consequence** -- the summed price against its own derived floor.
          Needs NO previous look, because it is a within-look statement about
          what the data cannot distinguish rather than a comparison of two
          realisations. That is also why it is monotone in ``n`` where the mass
          measure is not.
        """

        if self.consequence_gated:
            return self.consequence_resolved
        return (
            self.previous_look_available
            and self.maximum_observed_mask_disagreement_activity_fraction
            <= self.maximum_permitted_mask_disagreement_activity_fraction
        )

    @property
    def content_hash(self) -> str:
        return content_hash(self, prefix="irg-prune-order-stability-")

    def to_receipt(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "maximum_permitted_mask_disagreement_activity_fraction": float(
                self.maximum_permitted_mask_disagreement_activity_fraction
            ),
            "maximum_observed_mask_disagreement_activity_fraction": float(
                self.maximum_observed_mask_disagreement_activity_fraction
            ),
            "previous_look_available": bool(self.previous_look_available),
            "decision_resolved": self.resolved,
            # Which regime decided it, and both quantities, always -- so an
            # arm's receipt shows what the alternative gate would have said.
            "gating_regime": (
                "consequence_price_vs_materiality_floor"
                if self.consequence_gated
                and self.consequence_materiality_floor is not None
                else "consequence_price_vs_derived_floor"
                if self.consequence_gated
                else "activity_mass_vs_declared_tolerance"
            ),
            "consequence_materiality_floor": (
                None
                if self.consequence_materiality_floor is None
                else float(self.consequence_materiality_floor)
            ),
            "consequence_materiality_basis": "per_decision_mean",
            "consequence_price_evaluated": bool(
                self.consequence_price_evaluated
            ),
            "consequence_deferred_per_decision_bound": (
                None
                if self.consequence_deferred_per_decision_bound is None
                else float(self.consequence_deferred_per_decision_bound)
            ),
            "consequence_price_total": self.consequence_price_total,
            "consequence_price_standard_error": (
                self.consequence_price_standard_error
            ),
            "consequence_floor": (
                float(self.consequence_floor_critical_value)
                * self.consequence_price_standard_error
            ),
            "consequence_resolved": self.consequence_resolved,
            "consequence_price_deferred": bool(
                self.consequence_price_deferred
            ),
            "mass_resolved": bool(
                self.previous_look_available
                and self.maximum_observed_mask_disagreement_activity_fraction
                <= self.maximum_permitted_mask_disagreement_activity_fraction
            ),
            "tolerance_infeasible_scope_ids": (
                self.tolerance_infeasible_scope_ids
            ),
            "route_axis_hash": self.route_axis_hash,
            "activity_normalization": self.activity_normalization,
            "audited_retained_fractions": tuple(
                float(value) for value in self.audited_retained_fractions
            ),
            "scope_receipts": tuple(
                {
                    "scope_id": item.scope_id,
                    "route_count": int(item.route_count),
                    "partition_changed_route_count": int(
                        item.partition_changed_route_count
                    ),
                    "audited_boundary_count": int(
                        item.audited_boundary_count
                    ),
                    "maximum_mask_disagreement_route_fraction": float(
                        item.maximum_mask_disagreement_route_fraction
                    ),
                    "maximum_mask_disagreement_activity_fraction": float(
                        item.maximum_mask_disagreement_activity_fraction
                    ),
                    "mean_mask_disagreement_activity_fraction": float(
                        item.mean_mask_disagreement_activity_fraction
                    ),
                    "maximum_route_activity_fraction": float(
                        item.maximum_route_activity_fraction
                    ),
                    "minimum_feasible_tolerance": float(
                        item.minimum_feasible_tolerance
                    ),
                    "membership_churn_activity_fraction": float(
                        item.membership_churn_activity_fraction
                    ),
                    "score_order_activity_fraction": float(
                        item.score_order_activity_fraction
                    ),
                    "consequence_price": float(item.consequence_price),
                    "consequence_price_standard_error": float(
                        item.consequence_price_standard_error
                    ),
                    "consequence_band_size": int(item.consequence_band_size),
                    "consequence_binding_retained_fraction": float(
                        item.consequence_binding_retained_fraction
                    ),
                    "consequence_decision_functional_by_fraction": tuple(
                        (float(fraction), float(value))
                        for fraction, value in (
                            item.consequence_decision_functional_by_fraction
                        )
                    ),
                    "consequence_price_deferred": bool(
                        item.consequence_price_deferred
                    ),
                    "consequence_per_decision_price": (
                        float(item.consequence_price)
                        / float(item.consequence_band_size)
                        if int(item.consequence_band_size) > 0
                        else 0.0
                    ),
                    # The DERIVED gate: is this ordering ambiguity
                    # distinguishable from no ambiguity at the census's own
                    # precision? A tie prices exactly zero and can never be,
                    # which is the whole point -- under the declared activity
                    # numeral the same configuration could be permanently
                    # infeasible. Reporting only; nothing gates on it yet.
                    "consequence_price_significant": bool(
                        item.consequence_price
                        > 1.959963984540054
                        * item.consequence_price_standard_error
                        and item.consequence_price_standard_error > 0.0
                    ),
                    "tolerance_admits_single_swap": bool(
                        item.minimum_feasible_tolerance
                        <= self.
                        maximum_permitted_mask_disagreement_activity_fraction
                    ),
                }
                for item in self.scope_receipts
            ),
            "content_hash": self.content_hash,
        }


def _maximum_prefix_disagreement(
    *,
    previous_score: np.ndarray,
    current_score: np.ndarray,
    activity_weight: np.ndarray,
) -> tuple[int, float, float]:
    """Return maximum route/mass disagreement over every top-k cutoff."""

    previous = np.asarray(previous_score, dtype=np.float64).reshape(-1)
    current = np.asarray(current_score, dtype=np.float64).reshape(-1)
    weight = np.asarray(activity_weight, dtype=np.float64).reshape(-1)
    if (
        previous.shape != current.shape
        or current.shape != weight.shape
        or not np.isfinite(previous).all()
        or not np.isfinite(current).all()
        or not np.isfinite(weight).all()
        or np.any(previous < 0.0)
        or np.any(current < 0.0)
        or np.any(weight < 0.0)
    ):
        raise ValueError("prune-order stability arrays are invalid")
    count = int(current.size)
    if count <= 1:
        return 0, 0.0, 0.0
    previous_order = _stable_argsort(previous, descending=True)
    current_order = _stable_argsort(current, descending=True)
    previous_position = np.empty(count, dtype=np.int64)
    current_position = np.empty(count, dtype=np.int64)
    previous_position[previous_order] = np.arange(count, dtype=np.int64)
    current_position[current_order] = np.arange(count, dtype=np.int64)
    first = np.minimum(previous_position, current_position) + 1
    last = np.maximum(previous_position, current_position) + 1
    moved = first < last
    if not np.any(moved):
        return 0, 0.0, 0.0
    activity_delta = np.zeros(count + 1, dtype=np.float64)
    route_delta = np.zeros(count + 1, dtype=np.int64)
    # ``np.add.at`` is kept deliberately: the only bit-identical bincount
    # form is one call over the concatenated (first, last) index/weight
    # pairs (a split ``bincount(first) + bincount(last)`` re-associates the
    # float sums), and numpy>=1.25's unbuffered fast path makes ``add.at``
    # faster than that form (0.65 s vs 0.93 s for 55M moved routes).
    np.add.at(activity_delta, first[moved], weight[moved])
    np.add.at(activity_delta, last[moved], -weight[moved])
    np.add.at(route_delta, first[moved], 1)
    np.add.at(route_delta, last[moved], -1)
    boundary_activity = np.maximum(
        np.cumsum(activity_delta)[1:count],
        0.0,
    )
    boundary_routes = np.maximum(
        np.cumsum(route_delta)[1:count],
        0,
    )
    total_activity = max(
        float(np.sum(weight)),
        np.finfo(np.float64).tiny,
    )
    return (
        int(np.max(boundary_routes, initial=0)),
        float(np.max(boundary_activity, initial=0.0) / total_activity),
        float(
            (
                np.mean(boundary_activity) / total_activity
                if boundary_activity.size
                else 0.0
            )
        ),
    )


def consequence_price_over_band(
    *,
    score: np.ndarray,
    standard_error: np.ndarray,
    route_damage: np.ndarray,
    retained_fractions: Sequence[float],
    critical_value: float,
    route_damage_standard_error: np.ndarray | None = None,
    certain_member: np.ndarray | None = None,
) -> tuple[float, int, float, float]:
    """Worst-case damage swing over orderings the data cannot distinguish.

    Returns ``(price, band_size, retained_fraction, price_standard_error)`` at
    the binding cut. The standard error is ``0.0`` when
    ``route_damage_standard_error`` is not supplied.

    **The price's own uncertainty is what makes the floor derivable.** The
    price is a difference of sums of per-route damages, so its variance is the
    sum of those routes' estimator variances, and a census can ask the only
    question it is equipped to answer: *is this ordering ambiguity
    distinguishable from no ambiguity at all, at my own precision?* Certifying
    an ordering more finely than the measurement can resolve certifies noise.

    This is deliberately NOT the deciding oracle's resolvable increment, which
    is unavailable here: the
    census runs before any paired evidence exists, so the oracle's spread is
    not yet measured. The census's own precision is TIGHTER than the oracle's
    (it reads a far larger panel than the oracle's rail), so gating on it is
    conservative -- it certifies more finely than the decision strictly needs,
    which is safe rather than optimal. The looser, optimal floor becomes
    available once the census is a term rather than a gate.

    This is the metric replacement for the mass-based disagreement measure.
    That measure counts whether a top-k mask CHANGED and charges the change by
    the routes' ACTIVITY; this one asks how much the DAMAGE could differ across
    orderings consistent with the confidence intervals -- so a pair of routes
    that are genuinely tied costs zero, because pruning either one costs the
    same.

    Band membership is a COUNTING condition, not a comparison against an
    extreme:

        excludable(r) = |{s != r : upper_s > lower_r}| >= k
        includable(r) = |{s != r : lower_s > upper_r}| <  k

    The alternative rule -- ``lower_r < max(upper of the unselected)`` -- flags a
    route whenever a SINGLE other route could overtake it, regardless of its
    distance from the cut. MEASURED on the real coarse-granularity field, that
    rule marks **99.94%** of routes uncertain and is near-vacuous at census
    scale, because ``max(upper)`` over millions of noisy intervals is an
    extreme value. The counting condition gives 90.4% over all cuts and, at the
    binding cut of the smallest scope, **2 routes**.

    Reporting only: nothing gates on this yet. Comparing it against a floor
    requires the deciding oracle's resolvable increment, which is not in scope
    where this is computed.
    """

    values = np.asarray(score, dtype=np.float64).reshape(-1)
    errors = np.asarray(standard_error, dtype=np.float64).reshape(-1)
    damage = np.asarray(route_damage, dtype=np.float64).reshape(-1)
    damage_variance = (
        None
        if route_damage_standard_error is None
        else np.square(
            np.asarray(route_damage_standard_error, dtype=np.float64).reshape(-1)
        )
    )
    if (
        values.shape != errors.shape
        or values.shape != damage.shape
        or (damage_variance is not None and damage_variance.shape != values.shape)
    ):
        raise ValueError("consequence price axes are incompatible")
    size = values.size
    if size < 2:
        return 0.0, 0, 0.0, 0.0
    critical = float(critical_value)
    lower = values - critical * errors
    upper = values + critical * errors
    sorted_upper = np.sort(upper)
    sorted_lower = np.sort(lower)
    # No per-route count vectors: band membership at a
    # cut is EXACTLY two threshold comparisons via the searchsorted
    # identities (tie semantics preserved):
    #   overtakers_r >= L  <=>  lower_r <  sorted_upper[size-L-1]
    #   certain_r    <  K  <=>  upper_r >= sorted_lower[size-K]
    # so each fraction needs two scalar order statistics and one O(n)
    # mask instead of two full-axis searchsorted passes -- the same
    # numbers, computed on the estimand's own support.
    # Partition membership may itself be undetermined. A route whose sign is
    # not resolved is admissibly in EITHER partition, so it is eligible for
    # this scope's cut but not certainly a member -- which makes the scope's
    # cardinality, and hence the cut position k = ceil(f * |P|), a range rather
    # than a number. Treating that range as part of the admissible
    # configuration is what stops the cardinality shift appearing as a separate
    # contaminant charged to score-order instability.
    if certain_member is None:
        certain_count = size
    else:
        certain_mask = np.asarray(certain_member, dtype=bool).reshape(-1)
        if certain_mask.shape != values.shape:
            raise ValueError("consequence price membership axis is incompatible")
        certain_count = int(np.count_nonzero(certain_mask))
    best_price = 0.0
    best_band = 0
    best_fraction = 0.0
    best_price_se = 0.0
    # ONE damage argsort per scope, shared by every fraction (measured on a
    # canary: a per-fraction argpartition path costs 410-642 s
    # per certificate on the 57M-route kt axis -- ~87 fractions x O(band)
    # selections; with the shared presort each fraction is one boolean gather
    # plus slices, and the selected extremes are consumed in the SAME
    # damage-ascending order as a per-fraction selection, so the float
    # summation order and the emitted sums are the same).
    damage_order = _stable_argsort(damage, descending=False)
    damage_sorted = damage[damage_order]
    lower_damage_order = lower[damage_order]
    upper_damage_order = upper[damage_order]
    variance_sorted = (
        None if damage_variance is None else damage_variance[damage_order]
    )
    for retained_fraction in dict.fromkeys(
        float(value) for value in retained_fractions
    ):
        # The cut position spans the admissible scope sizes. Excludability is
        # hardest at the largest k and inclusion easiest there, so the widest
        # admissible band takes each at its most permissive end.
        low_count = int(math.ceil(retained_fraction * certain_count))
        high_count = int(math.ceil(retained_fraction * size))
        if high_count <= 0 or low_count >= size:
            continue
        retained_count = high_count
        minimum_overtakers = max(low_count, 1)
        if size - minimum_overtakers - 1 < 0:
            continue
        threshold_upper = sorted_upper[size - minimum_overtakers - 1]
        threshold_lower = sorted_lower[size - high_count]
        band_sorted = (lower_damage_order < threshold_upper) & (
            upper_damage_order >= threshold_lower
        )
        band_size = int(np.count_nonzero(band_sorted))
        if band_size == 0:
            continue
        # Certainly-retained routes fill the cut first; the band supplies the
        # remainder, and the adversary picks its best and worst members.
        # #{certain_r < high_count} = #{upper_r >= threshold_lower}.
        certainly_retained = (
            size
            - int(np.searchsorted(sorted_upper, threshold_lower, side="left"))
            - band_size
        )
        free_slots = retained_count - certainly_retained
        if free_slots <= 0 or free_slots >= band_size:
            continue
        # SELECT the extremes; do not SORT the band. Only the ``free_slots``
        # largest and smallest are ever read, and the bands are enormous:
        # MEASURED over one run's 98 scope-receipts, band sizes run min 6,
        # mean 3,221,662, max 37,557,506 -- the largest is 68.6% of the whole
        # 54,776,000-route axis. A full `argsort` of that costs 2.323 s against
        # 0.313 s for partition-based selection (7.42x; 5.60x at 3,086,299),
        # and this runs 28 fractions x 14 scopes = 392 times per certificate,
        # which is 56% of an arm's look budget.
        band_positions = np.flatnonzero(band_sorted)
        low_positions = band_positions[:free_slots]
        high_positions = band_positions[band_positions.size - free_slots :]
        # The presorted axis is damage-ascending, so these slices ARE the
        # band's smallest/largest damages already in ascending order -- the
        # exact order the previous selection consumed them in.
        price = float(
            damage_sorted[high_positions].sum()
            - damage_sorted[low_positions].sum()
        )
        price_se = 0.0
        if variance_sorted is not None:
            # The price is a difference of sums over two DISJOINT sets of
            # ``free_slots`` routes -- the band's best and worst. Its variance
            # is the sum of those routes' estimator variances; no covariance
            # term is added because the estimators' cross-covariance is not
            # measured, and omitting it is the same convention the registered
            # K3 interval already uses.
            price_se = float(
                math.sqrt(
                    float(variance_sorted[high_positions].sum())
                    + float(variance_sorted[low_positions].sum())
                )
            )
        # Rank by (price, band) rather than price alone. A price of exactly
        # zero over a LARGE band is a real and distinct state -- routes are
        # contested but tied, so the ordering genuinely does not matter -- and
        # ranking on price alone would report it as an empty band, collapsing
        # it with "nothing was undetermined". Those two say opposite things
        # about whether more evidence would help.
        if (price, band_size) > (best_price, best_band):
            best_price = price
            best_band = band_size
            best_fraction = float(retained_fraction)
            best_price_se = price_se
    return best_price, best_band, best_fraction, best_price_se


def _audited_partition_disagreement(
    *,
    previous_score: np.ndarray,
    current_score: np.ndarray,
    activity_weight: np.ndarray,
    previous_member: np.ndarray,
    current_member: np.ndarray,
    retained_fractions: Sequence[float],
    activity_denominator: float | None = None,
) -> tuple[int, float, float, int, float, float]:
    """Compare actual selected masks at declared retained fractions.

    Returns ``(routes, maximum_activity, mean_activity, audited_cuts,
    membership_churn_activity, score_order_activity)``. The last two are a
    disjoint, exhaustive split of ``maximum_activity`` at the binding cut.
    """

    previous = np.asarray(previous_score, dtype=np.float64).reshape(-1)
    current = np.asarray(current_score, dtype=np.float64).reshape(-1)
    weight = np.asarray(activity_weight, dtype=np.float64).reshape(-1)
    previous_in = np.asarray(previous_member, dtype=bool).reshape(-1)
    current_in = np.asarray(current_member, dtype=bool).reshape(-1)
    if not (
        previous.shape
        == current.shape
        == weight.shape
        == previous_in.shape
        == current_in.shape
    ):
        raise ValueError("audited partition axes are incompatible")
    fractions = tuple(dict.fromkeys(float(value) for value in retained_fractions))
    if not fractions or any(
        not np.isfinite(value) or not 0.0 < value < 1.0
        for value in fractions
    ):
        raise ValueError("audited retained fractions must lie in (0,1)")

    previous_indices = np.flatnonzero(previous_in)
    current_indices = np.flatnonzero(current_in)
    union = previous_in | current_in
    total_activity = max(
        (
            float(np.sum(weight[union]))
            if activity_denominator is None
            else float(activity_denominator)
        ),
        np.finfo(np.float64).tiny,
    )
    route_disagreements: list[int] = []
    activity_disagreements: list[float] = []
    # Churn attribution: a disagreeing route is selected in exactly one look,
    # and selection is a subset of membership, so it lies in the membership
    # union. Split it by whether its PARTITION MEMBERSHIP was stable. Routes
    # whose membership changed did not move at the score boundary at all --
    # they changed partition, which happens whenever a route's signed mean is
    # not sign-resolved, and the sign bar explicitly licenses a fraction of
    # activity to stay unresolved. Charging that churn to score-order
    # instability conflates two different failures, so report them apart.
    stable_membership = previous_in & current_in
    churn_disagreements: list[float] = []
    order_disagreements: list[float] = []
    # Hoisted: `members` is fixed across the fraction loop, so gathering the
    # member scores inside it repeated identical work 28 times per score.
    previous_member_values = previous[previous_indices]
    current_member_values = current[current_indices]
    # ONE stable descending argsort per side replaces two np.partition calls
    # per fraction (measured ~150 s per confirmation look at 57M routes even
    # with pricing deferred). A stable argsort of the NEGATED member values
    # realizes exactly `_stable_top_k_indices`' selection set -- descending
    # value with ties broken by ascending route index -- so top-k at any
    # fraction is a rank comparison. All per-fraction work then happens on
    # the membership UNION (disagreeing routes are selected in exactly one
    # look, hence lie in the union), never the full route axis.
    previous_rank_order = _stable_argsort(previous_member_values, descending=True)
    current_rank_order = _stable_argsort(current_member_values, descending=True)
    union_indices = np.flatnonzero(union)
    previous_rank = np.full(current.size, np.iinfo(np.int64).max, np.int64)
    previous_rank[previous_indices[previous_rank_order]] = np.arange(
        previous_indices.size, dtype=np.int64
    )
    current_rank = np.full(current.size, np.iinfo(np.int64).max, np.int64)
    current_rank[current_indices[current_rank_order]] = np.arange(
        current_indices.size, dtype=np.int64
    )
    union_previous_rank = previous_rank[union_indices]
    union_current_rank = current_rank[union_indices]
    union_weight = weight[union_indices]
    union_stable = stable_membership[union_indices]
    for retained_fraction in fractions:
        previous_count = int(
            np.ceil(retained_fraction * previous_indices.size)
        )
        current_count = int(
            np.ceil(retained_fraction * current_indices.size)
        )
        union_disagreement = (
            (union_previous_rank < previous_count)
            ^ (union_current_rank < current_count)
        )
        route_disagreements.append(
            int(np.count_nonzero(union_disagreement))
        )
        disagreeing_weight = union_weight[union_disagreement]
        disagreeing_stable = union_stable[union_disagreement]
        activity_disagreements.append(
            float(np.sum(disagreeing_weight) / total_activity)
        )
        churn_disagreements.append(
            float(
                np.sum(disagreeing_weight[~disagreeing_stable])
                / total_activity
            )
        )
        order_disagreements.append(
            float(
                np.sum(disagreeing_weight[disagreeing_stable])
                / total_activity
            )
        )
    # Report the two components AT THE CUT THAT BINDS, so they sum to the
    # reported maximum rather than to an unattainable combination of cuts.
    binding = (
        int(np.argmax(activity_disagreements))
        if activity_disagreements
        else None
    )
    return (
        max(route_disagreements, default=0),
        max(activity_disagreements, default=0.0),
        (
            float(np.mean(activity_disagreements))
            if activity_disagreements
            else 0.0
        ),
        len(fractions),
        0.0 if binding is None else churn_disagreements[binding],
        0.0 if binding is None else order_disagreements[binding],
    )


def _stable_top_k_indices(
    score: np.ndarray,
    member_indices: np.ndarray,
    retained_count: int,
    member_values: np.ndarray | None = None,
) -> np.ndarray:
    """Select stable-axis top-k routes without fully sorting the partition.

    ``member_values`` is the caller's chance to hoist ``score[member_indices]``
    out of a loop. The audited-boundary loop calls this once per retained
    fraction with the SAME members and only ``retained_count`` varying, so the
    gather was repeated 28 times per score per scope where one would do:
    784 gathers across a certificate, of which 28 are distinct.

    MEASURED, and smaller than it looks: **1.11x** on 28 calls over a 2M-route
    partition. The gather is real but it is not what dominates here --
    ``np.partition`` is -- so this is worth roughly 2% of the certificate, not
    a headline. Recorded at its measured size rather than its rhetorical one.

    Passing it changes no result: it is the same array by construction, and the
    parameter defaults to None so every other caller is untouched.
    """

    values = np.asarray(score, dtype=np.float64).reshape(-1)
    members = np.asarray(member_indices, dtype=np.int64).reshape(-1)
    count = int(retained_count)
    if count <= 0 or members.size == 0:
        return np.empty(0, dtype=np.int64)
    if count >= members.size:
        return members.copy()
    member_values = (
        values[members] if member_values is None else np.asarray(member_values)
    )
    if member_values.shape != members.shape:
        raise ValueError("precomputed member values differ from the member axis")
    threshold = float(
        np.partition(member_values, member_values.size - count)[
            member_values.size - count
        ]
    )
    strictly_above = members[member_values > threshold]
    tied = members[member_values == threshold]
    needed = count - int(strictly_above.size)
    if needed < 0 or needed > tied.size:
        raise RuntimeError("stable top-k threshold accounting failed")
    # ``members`` is in route-axis order, so the tie prefix is exactly the
    # stable descending-order tie rule without an O(R log R) full sort.
    return np.concatenate((strictly_above, tied[:needed]))


def certify_prune_order_stability(
    *,
    current_global_activity: Sequence[float] | np.ndarray,
    previous_global_activity: Sequence[float] | np.ndarray | None,
    current_partition_labels: Sequence[int] | np.ndarray,
    previous_partition_labels: Sequence[int] | np.ndarray | None,
    layer_ids: Sequence[str] | LayerCodeAxis,
    maximum_mask_disagreement_activity_fraction: float,
    scope: str,
    route_axis_hash: str,
    audited_retained_fractions: Sequence[float] | None = None,
    audited_retained_fractions_by_scope: Mapping[
        str, Sequence[float]
    ] | None = None,
    activity_normalization: str = "scope",
    score_standard_error: Sequence[float] | np.ndarray | None = None,
    route_damage: Sequence[float] | np.ndarray | None = None,
    route_damage_standard_error: Sequence[float] | np.ndarray | None = None,
    critical_value: float | None = None,
    consequence_gated: bool = False,
    consequence_floor_critical_value: float = 1.959963984540054,
    consequence_price_deferred: bool = False,
    consequence_materiality_floor: float | None = None,
    partition_sign_unresolved: Sequence[bool] | np.ndarray | None = None,
    consequence_price_short_circuit_scope: str | None = None,
) -> PruneOrderStabilityCertificate:
    """Audit successive proposal masks without treating them as commit evidence.

    The four optional arguments enable the **consequence price** alongside the
    incumbent mass-based disagreement. Supplying them changes no decision --
    the price is reporting only -- but it makes both measures visible on the
    same scopes, which is what lets the metric formulation be judged against
    the topological one on real fields rather than in principle.
    """

    current = np.asarray(current_global_activity, dtype=np.float64).reshape(-1)
    labels = np.asarray(current_partition_labels).reshape(-1)
    # Layer membership as int16 codes.  A legacy per-route string sequence is
    # coerced once here; the scope order (``names``) and every ``in_layer``
    # mask are identical to an object-array path by construction.
    layer_axis = LayerCodeAxis.from_layer_ids(layer_ids)
    if (
        labels.shape != current.shape
        or layer_axis.route_count != current.size
        or labels.dtype.kind not in {"i", "u", "b"}
        or not np.isin(labels, (0, 1)).all()
        or not np.isfinite(current).all()
        or np.any(current < 0.0)
        or scope not in {"layerwise", "cross_layer"}
        or activity_normalization not in {"scope", "global_route_field"}
    ):
        raise ValueError("prune-order stability axes are incompatible")
    global_activity_denominator = max(
        float(np.sum(current)),
        np.finfo(np.float64).tiny,
    )
    previous_available = (
        previous_global_activity is not None
        and previous_partition_labels is not None
    )
    previous = (
        np.asarray(previous_global_activity, dtype=np.float64).reshape(-1)
        if previous_available
        else current
    )
    previous_labels = (
        np.asarray(previous_partition_labels).reshape(-1)
        if previous_available
        else labels
    )
    if (
        previous.shape != current.shape
        or previous_labels.shape != current.shape
        or not np.isfinite(previous).all()
        or np.any(previous < 0.0)
        or previous_labels.dtype.kind not in {"i", "u", "b"}
        or not np.isin(previous_labels, (0, 1)).all()
    ):
        raise ValueError("previous prune-order stability axes are incompatible")
    audited_fractions = (
        ()
        if audited_retained_fractions is None
        else tuple(
            dict.fromkeys(float(value) for value in audited_retained_fractions)
        )
    )
    if audited_fractions and any(
        not np.isfinite(value) or not 0.0 < value < 1.0
        for value in audited_fractions
    ):
        raise ValueError("audited retained fractions must lie in (0,1)")
    scope_keys = (
        tuple(
            (str(layer), int(partition))
            for layer in layer_axis.names
            for partition in (0, 1)
        )
        if scope == "layerwise"
        else tuple(("global", partition) for partition in (0, 1))
    )
    receipts: list[PruneOrderStabilityScopeReceipt] = []
    consequence_price_evaluated = False

    def _evaluate_scope(
        layer: str,
        partition: int,
        *,
        price_allowed: bool,
    ) -> PruneOrderStabilityScopeReceipt | None:
        # Escape-velocity short-circuit (ALGORITHMIC_OPTIMIZATION_RECORDS
        # V0-2): the whole scope body, extracted verbatim so a hinted
        # binding scope can be priced FIRST; when its per-decision price
        # already exceeds the floor, the other scopes skip pricing
        # (price_allowed=False -> per-scope deferral, which
        # consequence_resolved already reads as unresolved). Gate
        # outcomes are identical by construction: a look can only PASS
        # via full evaluation. Unarmed (hint None), every scope prices
        # exactly as before, in the same order, bit-identically.
        nonlocal consequence_price_evaluated
        scope_id = f"{layer}:partition:{partition}"
        if (
            audited_retained_fractions_by_scope is not None
            and scope_id not in audited_retained_fractions_by_scope
        ):
            return None
        scope_fractions = (
            audited_fractions
            if audited_retained_fractions_by_scope is None
            else tuple(
                dict.fromkeys(
                    float(value)
                    for value in audited_retained_fractions_by_scope[
                        scope_id
                    ]
                )
            )
        )
        if scope_fractions and any(
            not np.isfinite(value) or not 0.0 < value < 1.0
            for value in scope_fractions
        ):
            raise ValueError(
                "scope-specific audited retained fractions must lie in (0,1)"
            )
        no_executable_cuts = (
            audited_retained_fractions_by_scope is not None and not scope_fractions
        )
        in_layer = (
            layer_axis.membership(layer)
            if scope == "layerwise"
            else np.ones(current.size, dtype=bool)
        )
        current_member = in_layer & (labels == partition)
        previous_member = in_layer & (previous_labels == partition)
        union = current_member | previous_member
        if not np.any(union):
            return None
        common = current_member & previous_member
        changed = union & ~common
        # The atom is priced against the SAME denominator the disagreement
        # uses, so the two are directly comparable against one tolerance.
        scope_activity_denominator = (
            global_activity_denominator
            if activity_normalization == "global_route_field"
            else max(
                float(np.sum(current[union])),
                np.finfo(np.float64).tiny,
            )
        )
        maximum_route_activity_fraction = float(
            np.max(current[union]) / scope_activity_denominator
        )
        minimum_feasible_tolerance = float(
            2.0 * maximum_route_activity_fraction
        )
        membership_churn_activity = 0.0
        score_order_activity = 0.0
        # Consequence price over this scope's own routes and cuts. Needs no
        # previous look -- it is a WITHIN-look statement about what the data
        # cannot distinguish, which is why it is monotone in n where the
        # look-to-look measure is not.
        price = 0.0
        price_standard_error = 0.0
        price_band = 0
        price_fraction = 0.0
        price_decision_functional: tuple[tuple[float, float], ...] = ()
        # Routes whose partition sign is unresolved are admissibly in EITHER
        # partition, so they are eligible for this scope's cut without being
        # certain members of it. That is the whole of the partition coordinate:
        # sign and order become two coordinates of one admissible-configuration
        # statement rather than two bars auditing the same hyperplane
        # arrangement while independent of each other.
        if partition_sign_unresolved is None:
            price_eligible = union
            price_certain = union
        else:
            unresolved = np.asarray(
                partition_sign_unresolved, dtype=bool
            ).reshape(-1)
            if unresolved.shape != current.shape:
                raise ValueError("partition sign axis is incompatible")
            price_eligible = union | (in_layer & unresolved)
            price_certain = union & ~unresolved
        if (
            score_standard_error is not None
            and route_damage is not None
            and critical_value is not None
            and scope_fractions
            and not consequence_price_deferred
            and price_allowed
        ):
            consequence_price_evaluated = True
            score_errors = np.asarray(
                score_standard_error, dtype=np.float64
            ).reshape(-1)
            damage_values = np.asarray(route_damage, dtype=np.float64).reshape(-1)
            damage_errors = (
                None
                if route_damage_standard_error is None
                else np.asarray(
                    route_damage_standard_error, dtype=np.float64
                ).reshape(-1)[price_eligible]
            )
            (
                price,
                price_band,
                price_fraction,
                price_standard_error,
            ) = consequence_price_over_band(
                score=current[price_eligible],
                standard_error=score_errors[price_eligible],
                route_damage=damage_values[price_eligible],
                route_damage_standard_error=damage_errors,
                retained_fractions=scope_fractions,
                critical_value=float(critical_value),
                certain_member=price_certain[price_eligible],
            )
            # D(M_k) at every audited fraction: ONE descending-score argsort
            # per scope supplies all fractions via a cumulative damage sum.
            # The summation order (descending score, stable ties) is fixed, so
            # the floats are deterministic run-to-run. This rides the deferral
            # above, so its argsort cost lands only on the late looks where
            # the price itself is computed.
            eligible_scores = current[price_eligible]
            eligible_damage = damage_values[price_eligible]
            descending = _stable_argsort(eligible_scores, descending=True)
            damage_cumulative = np.cumsum(eligible_damage[descending])
            functional_rows: list[tuple[float, float]] = []
            for fraction in dict.fromkeys(
                float(value) for value in scope_fractions
            ):
                retained = int(
                    math.ceil(fraction * int(eligible_scores.size))
                )
                if retained <= 0 or retained > int(eligible_scores.size):
                    continue
                functional_rows.append(
                    (
                        float(fraction),
                        float(damage_cumulative[retained - 1]),
                    )
                )
            price_decision_functional = tuple(functional_rows)
        if no_executable_cuts:
            # An explicit empty cut set is an executed rung rule with no
            # score-dependent proposal, so its order obligation is vacuous.
            maximum_routes, maximum_activity, mean_activity = 0, 0.0, 0.0
            audited_boundary_count = 0
            maximum_route_activity_fraction = 0.0
            minimum_feasible_tolerance = 0.0
            if (
                score_standard_error is not None
                and route_damage is not None
                and critical_value is not None
            ):
                consequence_price_evaluated = True
        elif not previous_available:
            maximum_routes, maximum_activity, mean_activity = 0, 1.0, 1.0
            audited_boundary_count = (
                len(scope_fractions)
                if scope_fractions
                else max(int(np.count_nonzero(union)) - 1, 0)
            )
        elif scope_fractions:
            (
                maximum_routes,
                maximum_activity,
                mean_activity,
                audited_boundary_count,
                membership_churn_activity,
                score_order_activity,
            ) = _audited_partition_disagreement(
                previous_score=previous,
                current_score=current,
                activity_weight=current,
                previous_member=previous_member,
                current_member=current_member,
                retained_fractions=scope_fractions,
                activity_denominator=(
                    global_activity_denominator
                    if activity_normalization == "global_route_field"
                    else None
                ),
            )
        else:
            union_activity = max(
                float(np.sum(current[union])),
                np.finfo(np.float64).tiny,
            )
            changed_activity_fraction = float(
                np.sum(current[changed]) / union_activity
            )
            max_routes, max_activity, mean_activity = (
                _maximum_prefix_disagreement(
                    previous_score=previous[common],
                    current_score=current[common],
                    activity_weight=current[common],
                )
                if np.any(common)
                else (0, 0.0, 0.0)
            )
            common_activity_fraction = float(
                np.sum(current[common]) / union_activity
            )
            # The unaudited branch already carries the split explicitly:
            # ``changed`` is the membership-churn term and the common-prefix
            # term is score-order movement among stable-membership routes.
            membership_churn_activity = float(changed_activity_fraction)
            score_order_activity = float(
                common_activity_fraction * max_activity
            )
            maximum_activity = min(
                changed_activity_fraction
                + common_activity_fraction * max_activity,
                1.0,
            )
            mean_activity = min(
                changed_activity_fraction
                + common_activity_fraction * mean_activity,
                1.0,
            )
            maximum_routes = int(
                max_routes + np.count_nonzero(changed)
            )
            audited_boundary_count = max(
                int(np.count_nonzero(common)) - 1,
                0,
            )
        return PruneOrderStabilityScopeReceipt(
                scope_id=scope_id,
                route_count=int(np.count_nonzero(union)),
                partition_changed_route_count=int(
                    np.count_nonzero(changed)
                ),
                audited_boundary_count=int(audited_boundary_count),
                maximum_mask_disagreement_route_fraction=float(
                    maximum_routes / max(int(np.count_nonzero(union)), 1)
                ),
                maximum_mask_disagreement_activity_fraction=(
                    float(maximum_activity)
                ),
                mean_mask_disagreement_activity_fraction=(
                    float(mean_activity)
                ),
                maximum_route_activity_fraction=(
                    maximum_route_activity_fraction
                ),
                minimum_feasible_tolerance=minimum_feasible_tolerance,
                membership_churn_activity_fraction=float(
                    membership_churn_activity
                ),
                score_order_activity_fraction=float(score_order_activity),
                consequence_price=float(price),
                consequence_price_standard_error=float(price_standard_error),
                consequence_band_size=int(price_band),
                consequence_binding_retained_fraction=float(price_fraction),
                consequence_decision_functional_by_fraction=(
                    price_decision_functional
                ),
                consequence_price_deferred=bool(
                    (consequence_price_deferred or not price_allowed)
                    and not no_executable_cuts
                ),
            )

    short_circuit_hint = consequence_price_short_circuit_scope
    short_circuit_engaged = False
    hint_receipt: PruneOrderStabilityScopeReceipt | None = None
    if (
        short_circuit_hint is not None
        and consequence_materiality_floor is not None
        and score_standard_error is not None
        and route_damage is not None
        and critical_value is not None
        and not consequence_price_deferred
    ):
        for layer, partition in scope_keys:
            if f"{layer}:partition:{partition}" != short_circuit_hint:
                continue
            hint_receipt = _evaluate_scope(
                layer, partition, price_allowed=True
            )
            if hint_receipt is not None:
                band = int(hint_receipt.consequence_band_size)
                if band > 0 and (
                    float(hint_receipt.consequence_price) / float(band)
                    > float(consequence_materiality_floor)
                ):
                    # Every scope must clear the floor for the gate to
                    # pass; this scope already exceeds it, so the other
                    # scopes' prices cannot change this look's outcome.
                    short_circuit_engaged = True
            break
    for layer, partition in scope_keys:
        if (
            hint_receipt is not None
            and f"{layer}:partition:{partition}" == short_circuit_hint
        ):
            receipts.append(hint_receipt)
            continue
        item = _evaluate_scope(
            layer,
            partition,
            price_allowed=not short_circuit_engaged,
        )
        if item is not None:
            receipts.append(item)
    if not receipts:
        raise ValueError("prune-order stability certificate has no scope")
    maximum = max(
        item.maximum_mask_disagreement_activity_fraction
        for item in receipts
    )
    return PruneOrderStabilityCertificate(
        maximum_permitted_mask_disagreement_activity_fraction=float(
            maximum_mask_disagreement_activity_fraction
        ),
        maximum_observed_mask_disagreement_activity_fraction=float(maximum),
        audited_retained_fractions=audited_fractions,
        scope_receipts=tuple(receipts),
        route_axis_hash=str(route_axis_hash),
        previous_look_available=bool(previous_available),
        activity_normalization=str(activity_normalization),
        consequence_gated=bool(consequence_gated),
        consequence_floor_critical_value=float(
            consequence_floor_critical_value
        ),
        consequence_price_deferred=bool(consequence_price_deferred),
        consequence_materiality_floor=(
            None
            if consequence_materiality_floor is None
            else float(consequence_materiality_floor)
        ),
        consequence_price_evaluated=bool(consequence_price_evaluated),
        consequence_deferred_per_decision_bound=(
            None
            if not consequence_price_deferred or route_damage is None
            else 2.0
            * float(
                np.max(np.abs(np.asarray(route_damage, dtype=np.float64)))
            )
        ),
    )


def _scope_receipt(
    *,
    scope_id: str,
    score: np.ndarray,
    standard_error: np.ndarray,
    critical_value: float,
    audited_retained_fractions: Sequence[float] | None = None,
    activity_weight: np.ndarray | None = None,
) -> PruneOrderScopeReceipt:
    """Audit executable top-k mask boundaries under score intervals.

    ``activity_weight`` optionally separates the mass field from the
    ordered statistic: membership uncertainty is decided by the score
    intervals while ambiguous mass is weighed by the supplied activity
    (the C2 weighting field is the incumbent activity envelope for every
    key).  ``None`` keeps the score as the weight, byte-identically.
    """

    values = np.asarray(score, dtype=np.float64).reshape(-1)
    errors = np.asarray(standard_error, dtype=np.float64).reshape(-1)
    if (
        values.shape != errors.shape
        or values.size <= 0
        or not np.isfinite(values).all()
        or not np.isfinite(errors).all()
        or np.any(values < 0.0)
        or np.any(errors < 0.0)
    ):
        raise ValueError("prune-order scores and uncertainties are invalid")
    order = np.argsort(-values, kind="stable")
    ordered = values[order]
    ordered_error = errors[order]
    if activity_weight is None:
        ordered_weight = ordered
    else:
        weight = np.asarray(activity_weight, dtype=np.float64).reshape(-1)
        if (
            weight.shape != values.shape
            or not np.isfinite(weight).all()
            or np.any(weight < 0.0)
        ):
            raise ValueError("prune-order activity weights are invalid")
        ordered_weight = weight[order]
    lower = np.maximum(ordered - float(critical_value) * ordered_error, 0.0)
    upper = ordered + float(critical_value) * ordered_error

    # The stable route-axis rule orders exact, zero-uncertainty ties without
    # inventing a scientific magnitude distinction. Nonzero-width crossings
    # remain uncertain and are never silently broken by route index.
    exact_tie_boundary = (
        (ordered_error[:-1] == 0.0)
        & (ordered_error[1:] == 0.0)
        & (ordered[:-1] == ordered[1:])
    )

    exact_tie_blocks = 0
    if exact_tie_boundary.size:
        tie_start: int | None = None
        for index, tied in enumerate(exact_tie_boundary):
            if tied and tie_start is None:
                tie_start = index
            if tie_start is not None and (
                not tied or index == exact_tie_boundary.size - 1
            ):
                exact_tie_blocks += 1
                tie_start = None

    retained_fractions = (
        ()
        if audited_retained_fractions is None
        else tuple(
            dict.fromkeys(
                float(value) for value in audited_retained_fractions
            )
        )
    )
    if retained_fractions and any(
        not math.isfinite(value) or not 0.0 < value < 1.0
        for value in retained_fractions
    ):
        raise ValueError("audited retained fractions must lie in (0,1)")

    total_activity = float(np.sum(ordered_weight))
    denominator = max(total_activity, np.finfo(np.float64).tiny)
    route_denominator = max(int(values.size), 1)
    boundary_count = max(int(values.size) - 1, 0)
    if retained_fractions and boundary_count > 0:
        retained_counts = tuple(
            dict.fromkeys(
                min(
                    max(int(np.ceil(fraction * values.size)), 1),
                    values.size - 1,
                )
                for fraction in retained_fractions
            )
        )
        boundary_activity_values: list[float] = []
        boundary_route_values: list[int] = []
        route_is_uncertain = np.zeros(values.size, dtype=bool)
        for retained_count in retained_counts:
            largest_unselected_upper = float(
                np.max(upper[retained_count:])
            )
            smallest_selected_lower = float(
                np.min(lower[:retained_count])
            )
            selected_uncertain = (
                lower[:retained_count] < largest_unselected_upper
            )
            unselected_uncertain = (
                upper[retained_count:] > smallest_selected_lower
            )
            uncertain = np.concatenate(
                (selected_uncertain, unselected_uncertain)
            )
            route_is_uncertain |= uncertain
            boundary_route_values.append(int(np.count_nonzero(uncertain)))
            boundary_activity_values.append(
                float(np.sum(ordered_weight[uncertain]))
            )
        boundary_activity = np.asarray(
            boundary_activity_values,
            dtype=np.float64,
        )
        boundary_routes = np.asarray(
            boundary_route_values,
            dtype=np.int64,
        )
    elif boundary_count == 0:
        boundary_activity = np.zeros(0, dtype=np.float64)
        boundary_routes = np.zeros(0, dtype=np.int64)
        route_is_uncertain = np.zeros(values.size, dtype=bool)
    else:
        # Boundary b retains ordered[0:b+1]. A selected route can change
        # membership iff its lower bound is strictly below the greatest
        # unselected upper bound. An unselected route can change membership
        # iff its upper bound is strictly above the least selected lower
        # bound. Strict comparisons leave exact point ties to the stable
        # route-axis policy.
        suffix_unselected_upper = np.maximum.accumulate(upper[::-1])[::-1][1:]
        prefix_selected_lower = np.minimum.accumulate(lower)[:-1]

        activity_delta = np.zeros(boundary_count + 1, dtype=np.float64)
        route_delta = np.zeros(boundary_count + 1, dtype=np.int64)
        route_is_uncertain = np.zeros(values.size, dtype=bool)

        selected_indices = np.arange(boundary_count, dtype=np.int64)
        selected_last = (
            np.searchsorted(
                -suffix_unselected_upper,
                -lower[:-1],
                side="left",
            )
            - 1
        )
        selected_valid = selected_last >= selected_indices
        if np.any(selected_valid):
            starts = selected_indices[selected_valid]
            stops = selected_last[selected_valid] + 1
            weights = ordered_weight[:-1][selected_valid]
            np.add.at(activity_delta, starts, weights)
            np.add.at(activity_delta, stops, -weights)
            np.add.at(route_delta, starts, 1)
            np.add.at(route_delta, stops, -1)
            route_is_uncertain[:-1] |= selected_valid

        unselected_indices = np.arange(1, values.size, dtype=np.int64)
        unselected_first = np.searchsorted(
            -prefix_selected_lower,
            -upper[1:],
            side="right",
        )
        unselected_last = unselected_indices - 1
        unselected_valid = unselected_first <= unselected_last
        if np.any(unselected_valid):
            starts = unselected_first[unselected_valid]
            stops = unselected_last[unselected_valid] + 1
            weights = ordered_weight[1:][unselected_valid]
            np.add.at(activity_delta, starts, weights)
            np.add.at(activity_delta, stops, -weights)
            np.add.at(route_delta, starts, 1)
            np.add.at(route_delta, stops, -1)
            route_is_uncertain[1:] |= unselected_valid

        boundary_activity = np.maximum(
            np.cumsum(activity_delta[:-1]),
            0.0,
        )
        boundary_routes = np.maximum(
            np.cumsum(route_delta[:-1]),
            0,
        )

    maximum_boundary_routes = int(
        np.max(boundary_routes, initial=0)
    )
    maximum_boundary_activity = float(
        np.max(boundary_activity, initial=0.0) / denominator
    )
    mean_boundary_activity = float(
        (
            np.mean(boundary_activity) / denominator
            if boundary_activity.size
            else 0.0
        )
    )
    return PruneOrderScopeReceipt(
        scope_id=str(scope_id),
        route_count=int(values.size),
        uncertain_boundary_count=int(np.count_nonzero(boundary_routes)),
        uncertain_route_count=int(np.count_nonzero(route_is_uncertain)),
        exact_tie_block_count=int(exact_tie_blocks),
        maximum_boundary_uncertain_route_count=maximum_boundary_routes,
        maximum_boundary_route_fraction=float(
            maximum_boundary_routes / route_denominator
        ),
        maximum_boundary_activity_fraction=float(
            min(maximum_boundary_activity, 1.0)
        ),
        mean_boundary_activity_fraction=float(
            min(mean_boundary_activity, 1.0)
        ),
        audited_retained_fractions=retained_fractions,
    )


def certify_prune_order_uncertainty(
    *,
    global_activity: Sequence[float] | np.ndarray,
    global_activity_standard_error: Sequence[float] | np.ndarray,
    partition_labels: Sequence[int] | np.ndarray,
    layer_ids: Sequence[str] | LayerCodeAxis,
    confidence_level: float,
    normal_critical_value: float,
    maximum_boundary_activity_fraction: float,
    scope: str,
    route_axis_hash: str,
    audited_retained_fractions_by_scope: Mapping[
        str, Sequence[float] | Mapping[int, Sequence[float]]
    ] | None = None,
    order_statistic: Sequence[float] | np.ndarray | None = None,
    order_statistic_standard_error: Sequence[float] | np.ndarray | None = None,
) -> PruneOrderUncertaintyCertificate:
    """Certify layerwise-v0 or cross-layer-v1 pruning-order uncertainty.

    A scope entry may be one shared fraction sequence (the layerwise-v0
    contract, unchanged) or a mapping from partition label to that
    partition's own fractions: a GLOBAL executable ladder cut lands at
    partition-specific within-subset counts, so the caller translates and
    supplies each partition's fractions explicitly.  A partition whose
    entry is empty has no interior executable cut and contributes no order
    claim (and no scope receipt).

    ``order_statistic`` (with its standard error) optionally replaces the
    activity as the ORDERED statistic — the registered ordering-key arms
    certify the selected key's order — while ``global_activity`` remains
    the declared C2 weighting field for ambiguous mass.  Omitted, the
    activity is both order and weight, byte-identically to an audit without it.
    """

    activity = np.asarray(global_activity, dtype=np.float64).reshape(-1)
    errors = np.asarray(
        global_activity_standard_error, dtype=np.float64
    ).reshape(-1)
    labels = np.asarray(partition_labels).reshape(-1)
    layer_axis = LayerCodeAxis.from_layer_ids(layer_ids)
    if (
        activity.shape != errors.shape
        or labels.shape != activity.shape
        or layer_axis.route_count != activity.size
        or labels.dtype.kind not in {"i", "u", "b"}
        or not np.isin(labels, (0, 1)).all()
        or scope not in {"layerwise", "cross_layer"}
    ):
        raise ValueError("prune-order certificate axes are incompatible")
    if (order_statistic is None) != (order_statistic_standard_error is None):
        raise ValueError(
            "order statistic and its standard error travel together"
        )
    if order_statistic is None:
        score = activity
        score_errors = errors
        weight: np.ndarray | None = None
    else:
        score = np.asarray(order_statistic, dtype=np.float64).reshape(-1)
        score_errors = np.asarray(
            order_statistic_standard_error, dtype=np.float64
        ).reshape(-1)
        if score.shape != activity.shape or score_errors.shape != activity.shape:
            raise ValueError(
                "order statistic axes differ from the activity axis"
            )
        weight = activity
    scope_keys = (
        tuple(
            (str(layer), int(partition))
            for layer in layer_axis.names
            for partition in (0, 1)
        )
        if scope == "layerwise"
        else tuple(("global", partition) for partition in (0, 1))
    )
    receipts: list[PruneOrderScopeReceipt] = []
    for layer, partition in scope_keys:
        if (
            audited_retained_fractions_by_scope is not None
            and str(layer) not in audited_retained_fractions_by_scope
        ):
            continue
        audited = (
            None
            if audited_retained_fractions_by_scope is None
            else audited_retained_fractions_by_scope.get(str(layer))
        )
        if isinstance(audited, Mapping):
            audited = audited.get(int(partition))
            if not audited:
                # No interior executable cut in this partition: nothing to
                # certify, and auditing every boundary here would silently
                # widen the claim past the declared ladder.
                continue
        selected = labels == partition
        if scope == "layerwise":
            selected &= layer_axis.membership(layer)
        indices = np.flatnonzero(selected)
        if not indices.size:
            continue
        receipts.append(
            _scope_receipt(
                scope_id=f"{layer}:partition:{partition}",
                score=score[indices],
                standard_error=score_errors[indices],
                critical_value=float(normal_critical_value),
                audited_retained_fractions=audited,
                activity_weight=(
                    None if weight is None else weight[indices]
                ),
            )
        )
    if not receipts:
        raise ValueError("prune-order certificate has no nonempty scope")
    maximum = max(
        item.maximum_boundary_activity_fraction for item in receipts
    )
    return PruneOrderUncertaintyCertificate(
        confidence_level=float(confidence_level),
        normal_critical_value=float(normal_critical_value),
        maximum_permitted_boundary_activity_fraction=float(
            maximum_boundary_activity_fraction
        ),
        maximum_observed_boundary_activity_fraction=float(maximum),
        scope_receipts=tuple(receipts),
        route_axis_hash=str(route_axis_hash),
    )


__all__ = [
    "PRUNE_ORDER_CERTIFICATE_V2",
    "PRUNE_ORDER_STABILITY_CERTIFICATE_V1",
    "PRUNE_ORDER_STABILITY_CERTIFICATE_V2",
    "PRUNE_ORDER_STABILITY_CERTIFICATE_V3",
    "consequence_price_over_band",
    "STABLE_TIE_POLICY_V1",
    "LayerCodeAxis",
    "PruneOrderScopeReceipt",
    "PruneOrderStabilityCertificate",
    "PruneOrderStabilityScopeReceipt",
    "PruneOrderUncertaintyCertificate",
    "certify_prune_order_stability",
    "certify_prune_order_uncertainty",
]
