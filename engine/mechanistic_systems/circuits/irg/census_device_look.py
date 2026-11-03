"""The census look on the device (sign criterion, labels, order certificate).

The census look of ``_measure_dense_local_partition_geometry`` reads the fused
decision snapshot and evaluates, per look, the sign criterion, the partition
labels and the layerwise prune-order stability certificate.  As host numpy
all of it runs over the whole route axis (57M routes on AlexNet, 128.7M on
VGG-11).  This module evaluates the same quantities on the census compute
device, in float64, with the host expressions:

* sums are ordinary device reductions (equal to the host's to rounding);
* sorts use the host's stable int64 key transform, whose stable permutation is
  unique, so every order statistic, rank and cut is the host's;
* the order certificate prices every retained fraction of a scope in one
  batched pass and reads each scope's scalars back once.

Every decision uses float64 device arithmetic, including near ties. Receipt
flags describe near-threshold results without recomputing them. Nonfinite
reductions raise ValueError.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

try:  # Optional for import smoke.
    import torch
except Exception:  # pragma: no cover - environment dependent.
    torch = None  # type: ignore[assignment]


# Scalar decision band (relative): a decision scalar this close to its
# threshold is reported in the look receipt as a near tie (telemetry only).
SCALAR_DECISION_BAND = 1e-12
_FLOAT64_SIGN_FLIP_MASK = 0x7FFFFFFFFFFFFFFF
_TINY = float(np.finfo(np.float64).tiny)


def _host_float(value: Any) -> float:
    return float(value.item()) if hasattr(value, "item") else float(value)


def _stable_order(values: Any, *, descending: bool) -> Any:
    """``score_order._stable_argsort`` on the device: the same unique permutation.

    No host read: callers validate finiteness with their own results (for
    NaN-free input the stable permutation of the int64 keys is the host's).
    """

    keyed = (-values.reshape(-1).to(torch.float64) if descending else values.reshape(-1).to(torch.float64)) + 0.0
    key = keyed.contiguous().view(torch.int64)
    key = torch.where(key < 0, key ^ _FLOAT64_SIGN_FLIP_MASK, key)
    return torch.sort(key, stable=True).indices


def _near(value: float, threshold: float) -> bool:
    scale = max(abs(float(value)), abs(float(threshold)), _TINY)
    return abs(float(value) - float(threshold)) <= SCALAR_DECISION_BAND * scale


def _numpy_maximum_zero(values: Any) -> Any:
    """``np.maximum(values, 0.0)`` with NumPy's NaN propagation."""

    return torch.maximum(values, torch.zeros_like(values))


def _read(*values: Any) -> list[float]:
    """One device-to-host read of several 0-dim or 1-dim device values."""

    return torch.cat([torch.as_tensor(value).reshape(-1).to(torch.float64) for value in values]).cpu().tolist()


# ---------------------------------------------------------------------------
# Sign criterion (criterion_forecast.py) on the device.
# ---------------------------------------------------------------------------


def unresolved_activity_fraction_device(
    *, mean: Any, standard_error: Any, activity: Any, critical: float
) -> float:
    """``criterion_forecast.unresolved_activity_fraction`` (no floor)."""

    ambiguous = (torch.abs(mean) <= float(critical) * standard_error) & (standard_error > 0.0)
    mass, ambiguous_mass = _read(activity.sum(), torch.where(ambiguous, activity, 0.0).sum())
    if not math.isfinite(mass):
        raise ValueError("nonfinite_activity_mass")
    if mass <= _TINY:
        return 0.0
    if not math.isfinite(ambiguous_mass):
        raise ValueError("nonfinite_ambiguous_mass")
    return float(ambiguous_mass / mass)


def wrongsign_consequence_upper_bound_fraction_device(
    *, mean: Any, standard_error: Any, activity: Any, critical: float
) -> float:
    """``criterion_forecast.wrongsign_consequence_upper_bound_fraction``."""

    magnitude = torch.abs(mean)
    wrong_side = _numpy_maximum_zero(float(critical) * standard_error - magnitude)
    consequence_mass, wrong_mass = _read(
        (activity * magnitude).sum(), (2.0 * activity * wrong_side).sum()
    )
    if not math.isfinite(consequence_mass):
        raise ValueError("nonfinite_consequence_mass")
    if consequence_mass <= _TINY:
        return 0.0
    if not math.isfinite(wrong_mass):
        raise ValueError("nonfinite_wrong_mass")
    return wrong_mass / consequence_mass


def expected_wrongsign_consequence_fraction_device(
    *, mean: Any, standard_error: Any, activity: Any
) -> float:
    """``expected_wrongsign_consequence_fraction`` on the device (telemetry).

    Receipt-only acquisition telemetry (never a stopping authority).  The
    wrong-side term ``se*phi(z) - |mean|*Q(z)`` is evaluated as
    ``se*exp(-z^2/2)*(1/sqrt(2*pi) - z*erfcx(z/sqrt(2))/2)``, whose relative
    error grows only like ``z^2`` ulps (z < 39 before the factor underflows),
    so no z needs the host.
    """

    magnitude = torch.abs(mean)
    live = standard_error > 0.0
    live_error = torch.where(live, standard_error, 1.0)
    z = magnitude / live_error
    bracket = 1.0 / math.sqrt(2.0 * math.pi) - 0.5 * z * torch.special.erfcx(z / math.sqrt(2.0))
    wrong_side = live_error * torch.exp(-0.5 * z * z) * bracket
    # z = inf (a subnormal error): both tails vanish, as on the host.
    wrong_side = torch.where(torch.isinf(z) | ~live, 0.0, wrong_side)
    wrong_side = _numpy_maximum_zero(wrong_side)
    consequence_mass, expected_wrong_mass = _read(
        (activity * magnitude).sum(), (2.0 * activity * wrong_side).sum()
    )
    if not math.isfinite(consequence_mass):
        raise ValueError("nonfinite_consequence_mass")
    if consequence_mass <= _TINY:
        return 0.0
    if not math.isfinite(expected_wrong_mass):
        raise ValueError("nonfinite_expected_wrong_mass")
    return float(expected_wrong_mass / consequence_mass)


# ---------------------------------------------------------------------------
# Layerwise prune-order stability certificate (score_order.py).
# ---------------------------------------------------------------------------


def _consequence_price_over_band_device(
    *, score: Any, standard_error: Any, route_damage: Any,
    retained_fractions: Sequence[float], critical_value: float,
    route_damage_standard_error: Any | None, certain_member: Any | None,
    decision_flags: list[str],
) -> tuple[float, int, float, float]:
    """Price every retained-fraction uncertainty band on the device."""
    from mechanistic_systems.measurement.phase_device_orchestration import active_state, PhaseDeviceState
    from mechanistic_systems.measurement.phase_census_program import consequence
    return consequence(score=score, standard_error=standard_error, route_damage=route_damage,
        retained_fractions=retained_fractions, critical_value=critical_value,
        route_damage_standard_error=route_damage_standard_error, certain_member=certain_member,
        state=active_state() or PhaseDeviceState(), decision_band=SCALAR_DECISION_BAND,
        decision_flags=decision_flags)


def _decision_functional_device(
    *, score: Any, route_damage: Any, retained_fractions: Sequence[float]
) -> tuple[tuple[float, float], ...]:
    """D(M_k) per fraction: the damage of the top-k-by-score mask (receipt only).

    Sum only the requested prefixes in the stable descending score order.
    Each fixed-shape reduction has deterministic CUDA execution; a floating
    CUDA cumulative scan has no deterministic implementation.
    """

    size = int(score.numel())
    rows = []
    for fraction in dict.fromkeys(float(value) for value in retained_fractions):
        retained = int(math.ceil(fraction * size))
        if 0 < retained <= size:
            rows.append((fraction, retained))
    if not rows:
        return ()
    ordered_damage = route_damage.reshape(-1)[_stable_order(score, descending=True)]
    damage_sums = torch.stack([ordered_damage[:retained].sum() for _, retained in rows])
    finite = torch.isfinite(ordered_damage).all() & torch.isfinite(damage_sums).all()
    values = _read(finite, damage_sums)
    if not values[0]:
        raise ValueError("nonfinite_damage_cumulative")
    return tuple((fraction, float(value)) for (fraction, _), value in zip(rows, values[1:], strict=True))


def _audited_partition_disagreement_device(
    *, previous_score: Any, current_score: Any, activity_weight: Any,
    previous_member: Any, current_member: Any, retained_fractions: Sequence[float],
    activity_denominator: float | None,
) -> tuple[int, float, float, int, float, float]:
    """Measure mask disagreement at every executable retained fraction."""
    from mechanistic_systems.measurement.phase_device_orchestration import active_state, PhaseDeviceState
    from mechanistic_systems.measurement.phase_census_program import disagreement
    return disagreement(previous_score=previous_score, current_score=current_score,
        activity_weight=activity_weight, previous_member=previous_member,
        current_member=current_member, retained_fractions=retained_fractions,
        activity_denominator=activity_denominator, state=active_state() or PhaseDeviceState())


def certify_prune_order_stability_device(
    *,
    current_global_activity: Any,
    previous_global_activity: Any | None,
    current_partition_labels: Any,
    previous_partition_labels: Any | None,
    layer_codes: Any,
    layer_names: tuple[str, ...],
    maximum_mask_disagreement_activity_fraction: float,
    scope: str,
    route_axis_hash: str,
    audited_retained_fractions: Sequence[float] | None = None,
    audited_retained_fractions_by_scope: Mapping[str, Sequence[float]] | None = None,
    activity_normalization: str = "scope",
    score_standard_error: Any | None = None,
    route_damage: Any | None = None,
    route_damage_standard_error: Any | None = None,
    critical_value: float | None = None,
    consequence_gated: bool = False,
    consequence_floor_critical_value: float = 1.959963984540054,
    consequence_price_deferred: bool = False,
    consequence_materiality_floor: float | None = None,
    partition_sign_unresolved: Any | None = None,
    consequence_price_short_circuit_scope: str | None = None,
    decision_flags: list[str] | None = None,
) -> Any:
    """``score_order.certify_prune_order_stability`` on device tensors.

    Returns the same ``PruneOrderStabilityCertificate`` dataclass.  Each
    layer's routes are gathered once (one stable sort of the layer codes);
    each scope prices all its retained fractions in one batched pass and
    reads its scalars back in a few reads.  Raises
    :class:`ValueError` on a nonfinite field and for the one branch
    it does not port (an audited-fraction-free scope with a previous look,
    never reached by the census: every census scope carries its executable
    boundary fractions).  ``decision_flags`` collects scalar decisions that
    landed within ``SCALAR_DECISION_BAND`` of their thresholds (reported,
    never re-decided).
    """

    from .score_order import (
        PruneOrderStabilityCertificate,
        PruneOrderStabilityScopeReceipt,
    )

    flags = decision_flags if decision_flags is not None else []
    current = current_global_activity.reshape(-1).to(torch.float64)
    labels = current_partition_labels.reshape(-1)
    codes = layer_codes.reshape(-1)
    integer_labels = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool)
    if (
        labels.shape != current.shape
        or codes.numel() != current.numel()
        or labels.dtype not in integer_labels
        or scope not in {"layerwise", "cross_layer"}
        or activity_normalization not in {"scope", "global_route_field"}
    ):
        raise ValueError("prune-order stability axes are incompatible")
    previous_available = (
        previous_global_activity is not None and previous_partition_labels is not None
    )
    previous = (
        previous_global_activity.reshape(-1).to(torch.float64) if previous_available else current
    )
    previous_labels = previous_partition_labels.reshape(-1) if previous_available else labels
    if previous.shape != current.shape or previous_labels.shape != current.shape:
        raise ValueError("previous prune-order stability axes are incompatible")
    # One read: the axis checks and the route field's activity mass.
    checks = _read(
        ((labels == 0) | (labels == 1)).all(),
        torch.isfinite(current).all(),
        (current >= 0.0).all(),
        current.sum(),
        ((previous_labels == 0) | (previous_labels == 1)).all(),
        torch.isfinite(previous).all(),
        (previous >= 0.0).all(),
    )
    if not all(checks[:3]):
        raise ValueError("prune-order stability axes are incompatible")
    global_activity_denominator = max(float(checks[3]), _TINY)
    if not math.isfinite(global_activity_denominator):
        raise ValueError("nonfinite_activity_mass")
    if not all(checks[4:]):
        raise ValueError("previous prune-order stability axes are incompatible")
    audited_fractions = (
        ()
        if audited_retained_fractions is None
        else tuple(dict.fromkeys(float(value) for value in audited_retained_fractions))
    )
    if audited_fractions and any(
        not np.isfinite(value) or not 0.0 < value < 1.0 for value in audited_fractions
    ):
        raise ValueError("audited retained fractions must lie in (0,1)")
    scope_keys = (
        tuple((str(layer), int(partition)) for layer in layer_names for partition in (0, 1))
        if scope == "layerwise"
        else tuple(("global", partition) for partition in (0, 1))
    )
    unresolved = (
        None if partition_sign_unresolved is None else partition_sign_unresolved.reshape(-1).to(torch.bool)
    )
    if unresolved is not None and unresolved.shape != current.shape:
        raise ValueError("partition sign axis is incompatible")
    pricing = (
        score_standard_error is not None
        and route_damage is not None
        and critical_value is not None
        and not consequence_price_deferred
    )
    fields = {
        "current": current,
        "previous": previous,
        "labels": labels,
        "previous_labels": previous_labels,
        "unresolved": unresolved,
        "score_errors": None if score_standard_error is None else score_standard_error.reshape(-1),
        "damage": None if route_damage is None else route_damage.reshape(-1),
        "damage_errors": (
            None if route_damage_standard_error is None else route_damage_standard_error.reshape(-1)
        ),
    }
    # Each layer's routes in route order: one stable sort of the layer codes.
    layer_views: dict[str, dict[str, Any]] = {}
    if scope == "layerwise":
        code_values = codes.to(torch.int64)
        by_layer = torch.sort(code_values, stable=True).indices
        sizes = torch.bincount(code_values, minlength=len(layer_names))[: len(layer_names)]
        bounds = np.concatenate(([0], np.cumsum(sizes.cpu().numpy()))).astype(np.int64)
        for index, name in enumerate(layer_names):
            if str(name) in layer_views:
                continue
            routes = by_layer[int(bounds[index]) : int(bounds[index + 1])]
            layer_views[str(name)] = {
                key: None if value is None else value.index_select(0, routes)
                for key, value in fields.items()
            }
    else:
        layer_views["global"] = fields
    receipts: list[Any] = []
    state = {"consequence_price_evaluated": False}

    def _evaluate_scope(layer: str, partition: int, *, price_allowed: bool) -> Any | None:
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
                    float(value) for value in audited_retained_fractions_by_scope[scope_id]
                )
            )
        )
        if scope_fractions and any(
            not np.isfinite(value) or not 0.0 < value < 1.0 for value in scope_fractions
        ):
            raise ValueError("scope-specific audited retained fractions must lie in (0,1)")
        no_executable_cuts = (
            audited_retained_fractions_by_scope is not None and not scope_fractions
        )
        view = layer_views.get(str(layer))
        if view is None or view["current"].numel() == 0:
            return None
        activity = view["current"]
        current_member = view["labels"] == partition
        previous_member = view["previous_labels"] == partition
        union = current_member | previous_member
        changed = union & ~(current_member & previous_member)
        union_activity = torch.where(union, activity, 0.0)
        union_count, changed_count, union_mass, union_maximum = _read(
            union.sum(), changed.sum(), union_activity.sum(), union_activity.max()
        )
        union_count = int(union_count)
        if union_count == 0:
            return None
        scope_activity_denominator = (
            global_activity_denominator
            if activity_normalization == "global_route_field"
            else max(float(union_mass), _TINY)
        )
        if not math.isfinite(scope_activity_denominator):
            raise ValueError("nonfinite_scope_activity_mass")
        maximum_route_activity_fraction = float(union_maximum / scope_activity_denominator)
        minimum_feasible_tolerance = float(2.0 * maximum_route_activity_fraction)
        membership_churn_activity = 0.0
        score_order_activity = 0.0
        price = 0.0
        price_standard_error = 0.0
        price_band = 0
        price_fraction = 0.0
        price_decision_functional: tuple[tuple[float, float], ...] = ()
        if pricing and scope_fractions and price_allowed:
            state["consequence_price_evaluated"] = True
            if view["unresolved"] is None:
                eligible, certain = union, union
            else:
                eligible = union | view["unresolved"]
                certain = union & ~view["unresolved"]
            routes = torch.nonzero(eligible, as_tuple=False).reshape(-1)
            eligible_scores = activity.index_select(0, routes)
            eligible_damage = view["damage"].index_select(0, routes)
            (
                price,
                price_band,
                price_fraction,
                price_standard_error,
            ) = _consequence_price_over_band_device(
                score=eligible_scores,
                standard_error=view["score_errors"].index_select(0, routes),
                route_damage=eligible_damage,
                route_damage_standard_error=(
                    None
                    if view["damage_errors"] is None
                    else view["damage_errors"].index_select(0, routes)
                ),
                retained_fractions=scope_fractions,
                critical_value=float(critical_value),
                certain_member=certain.index_select(0, routes),
                decision_flags=flags,
            )
            price_decision_functional = _decision_functional_device(
                score=eligible_scores,
                route_damage=eligible_damage,
                retained_fractions=scope_fractions,
            )
        if no_executable_cuts:
            maximum_routes, maximum_activity, mean_activity = 0, 0.0, 0.0
            audited_boundary_count = 0
            maximum_route_activity_fraction = 0.0
            minimum_feasible_tolerance = 0.0
            if (
                score_standard_error is not None
                and route_damage is not None
                and critical_value is not None
            ):
                state["consequence_price_evaluated"] = True
        elif not previous_available:
            maximum_routes, maximum_activity, mean_activity = 0, 1.0, 1.0
            audited_boundary_count = (
                len(scope_fractions) if scope_fractions else max(union_count - 1, 0)
            )
        elif scope_fractions:
            (
                maximum_routes,
                maximum_activity,
                mean_activity,
                audited_boundary_count,
                membership_churn_activity,
                score_order_activity,
            ) = _audited_partition_disagreement_device(
                previous_score=view["previous"],
                current_score=activity,
                activity_weight=activity,
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
            raise ValueError("unaudited_scope_prefix_disagreement")
        if not all(
            math.isfinite(value)
            for value in (
                maximum_route_activity_fraction, minimum_feasible_tolerance,
                maximum_activity, mean_activity, membership_churn_activity,
                score_order_activity, price, price_standard_error,
            )
        ):
            raise ValueError("nonfinite_order_certificate_field")
        return PruneOrderStabilityScopeReceipt(
            scope_id=scope_id,
            route_count=union_count,
            partition_changed_route_count=int(changed_count),
            audited_boundary_count=int(audited_boundary_count),
            maximum_mask_disagreement_route_fraction=float(
                maximum_routes / max(union_count, 1)
            ),
            maximum_mask_disagreement_activity_fraction=float(maximum_activity),
            mean_mask_disagreement_activity_fraction=float(mean_activity),
            maximum_route_activity_fraction=maximum_route_activity_fraction,
            minimum_feasible_tolerance=minimum_feasible_tolerance,
            membership_churn_activity_fraction=float(membership_churn_activity),
            score_order_activity_fraction=float(score_order_activity),
            consequence_price=float(price),
            consequence_price_standard_error=float(price_standard_error),
            consequence_band_size=int(price_band),
            consequence_binding_retained_fraction=float(price_fraction),
            consequence_decision_functional_by_fraction=price_decision_functional,
            consequence_price_deferred=bool(
                (consequence_price_deferred or not price_allowed) and not no_executable_cuts
            ),
        )

    short_circuit_hint = consequence_price_short_circuit_scope
    short_circuit_engaged = False
    hint_receipt: Any | None = None
    if short_circuit_hint is not None and consequence_materiality_floor is not None and pricing:
        for layer, partition in scope_keys:
            if f"{layer}:partition:{partition}" != short_circuit_hint:
                continue
            hint_receipt = _evaluate_scope(layer, partition, price_allowed=True)
            if hint_receipt is not None:
                band = int(hint_receipt.consequence_band_size)
                if band > 0:
                    ratio = float(hint_receipt.consequence_price) / float(band)
                    if _near(ratio, float(consequence_materiality_floor)):
                        flags.append("short_circuit_floor_near_tie")
                    if ratio > float(consequence_materiality_floor):
                        short_circuit_engaged = True
            break
    for layer, partition in scope_keys:
        if hint_receipt is not None and f"{layer}:partition:{partition}" == short_circuit_hint:
            receipts.append(hint_receipt)
            continue
        item = _evaluate_scope(layer, partition, price_allowed=not short_circuit_engaged)
        if item is not None:
            receipts.append(item)
    if not receipts:
        raise ValueError("prune-order stability certificate has no scope")
    maximum = max(item.maximum_mask_disagreement_activity_fraction for item in receipts)
    certificate = PruneOrderStabilityCertificate(
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
        consequence_floor_critical_value=float(consequence_floor_critical_value),
        consequence_price_deferred=bool(consequence_price_deferred),
        consequence_materiality_floor=(
            None if consequence_materiality_floor is None else float(consequence_materiality_floor)
        ),
        consequence_price_evaluated=bool(state["consequence_price_evaluated"]),
        consequence_deferred_per_decision_bound=(
            None
            if not consequence_price_deferred or route_damage is None
            else 2.0 * _host_float(torch.max(torch.abs(route_damage.reshape(-1).to(torch.float64))))
        ),
    )
    flags.extend(_certificate_decision_band_flags(certificate))
    return certificate


def _certificate_decision_band_flags(certificate: Any) -> list[str]:
    """Scalar decision comparisons of ``resolved`` sitting inside the band."""

    flags: list[str] = []
    permitted = float(certificate.maximum_permitted_mask_disagreement_activity_fraction)
    if certificate.previous_look_available and _near(
        float(certificate.maximum_observed_mask_disagreement_activity_fraction), permitted
    ):
        flags.append("mass_tolerance_near_tie")
    if certificate.consequence_materiality_floor is not None:
        floor = float(certificate.consequence_materiality_floor)
        for item in certificate.scope_receipts:
            band = int(item.consequence_band_size)
            if band > 0 and _near(float(item.consequence_price) / float(band), floor):
                flags.append("materiality_floor_near_tie")
                break
    elif certificate.consequence_gated:
        error = certificate.consequence_price_standard_error
        if _near(
            certificate.consequence_price_total,
            float(certificate.consequence_floor_critical_value) * error,
        ):
            flags.append("derived_floor_near_tie")
    ratios = sorted(
        (
            float(item.consequence_price) / float(item.consequence_band_size)
            for item in certificate.scope_receipts
            if not bool(getattr(item, "consequence_price_deferred", True))
            and int(getattr(item, "consequence_band_size", 0)) > 0
        ),
        reverse=True,
    )
    if len(ratios) >= 2 and ratios[0] != ratios[1] and _near(ratios[0], ratios[1]):
        flags.append("binding_scope_near_tie")
    return flags


# ---------------------------------------------------------------------------
# The census's per-look device state.
# ---------------------------------------------------------------------------

DEVICE_LABEL_FAMILIES = frozenset(
    {
        "local_attribution_sign",
        "single_population",
        "global_attribution_sign",
        "local_global_sign_alignment",
    }
)


def certify_prune_order_stability_from_host_arrays(*, device: Any, **kwargs: Any) -> Any:
    """Evaluate a fixed shared census with the registered device certificate.

    Shared census fields are retained as host arrays between worker items.
    Transfer them to the consumer's compute device and reuse the ordinary
    census evaluator, including its batched executable-fraction pricing.
    """
    from .score_order import LayerCodeAxis

    look = CensusDeviceLook(
        device=device,
        layer_axis=LayerCodeAxis.from_layer_ids(kwargs["layer_ids"]),
        partition_family="single_population",
        analytic_route_mask=None,
        sign_measurement_weights_by_head={},
    )
    return look.certify_prune_order_stability(**kwargs)


class CensusDeviceLook:
    """Device twins of one census's look arrays and the v6 look evaluations.

    The census keeps its host arrays (receipts, hashes, the sign
    forecast and every host consumer read them).  This object maps each such
    array to its device twin (the snapshot's device fields, the labels it
    computed, the carried previous-look arrays, the per-census sign weights)
    and evaluates the look's sign criterion, labels and layerwise order
    certificate there.
    """

    def __init__(
        self,
        *,
        device: Any,
        layer_axis: Any,
        partition_family: str,
        analytic_route_mask: np.ndarray | None,
        sign_measurement_weights_by_head: Mapping[int, np.ndarray],
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("the v6 census look requires PyTorch")
        self.device = torch.device(device)
        self.layer_names = tuple(str(name) for name in layer_axis.names)
        self.layer_codes = torch.as_tensor(
            np.asarray(layer_axis.codes, dtype=np.int64), device=self.device
        )
        self.route_count = int(np.asarray(layer_axis.codes).size)
        self.partition_family = str(partition_family)
        self._census_twins: list[tuple[np.ndarray, Any]] = []
        self._carried_twins: list[tuple[np.ndarray, Any]] = []
        self._look_twins: list[tuple[np.ndarray, Any]] = []
        for values in sign_measurement_weights_by_head.values():
            array = np.asarray(values)
            self._census_twins.append(
                (array, torch.as_tensor(np.ascontiguousarray(array), device=self.device))
            )
        self._analytic_mask = (
            None
            if analytic_route_mask is None or not bool(np.any(analytic_route_mask))
            else torch.as_tensor(np.asarray(analytic_route_mask, dtype=bool), device=self.device)
        )
        self._analytic_labels: Any | None = None
        self.labels_on_device = self.partition_family in DEVICE_LABEL_FAMILIES
        self._look: dict[str, Any] = {}
        self.look_receipts: list[dict[str, Any]] = []

    # -- twins ----------------------------------------------------------------

    def _register(self, host: np.ndarray, device_tensor: Any) -> Any:
        self._look_twins.append((np.asarray(host), device_tensor))
        return device_tensor

    @staticmethod
    def _match(array: np.ndarray, base: np.ndarray, tensor: Any) -> Any | None:
        if array is base:
            return tensor
        if array.dtype != base.dtype or base.size == 0 or array.size == 0:
            return None
        pointer = array.__array_interface__["data"][0]
        base_pointer = base.__array_interface__["data"][0]
        if array.shape == base.shape and array.strides == base.strides and pointer == base_pointer:
            return tensor
        if (
            base.ndim == 2
            and array.ndim == 1
            and array.shape[0] == base.shape[1]
            and array.strides[0] == base.strides[1]
        ):
            offset = pointer - base_pointer
            if offset >= 0 and offset % base.strides[0] == 0 and offset // base.strides[0] < base.shape[0]:
                return tensor[int(offset // base.strides[0])]
        return None

    def twin(self, value: Any) -> Any:
        """The device tensor holding ``value``'s bytes (H2D when unknown)."""

        if torch.is_tensor(value):
            return value.to(self.device)
        array = np.asarray(value)
        for base, tensor in (*self._look_twins, *self._carried_twins, *self._census_twins):
            found = self._match(array, base, tensor)
            if found is not None:
                return found
        if self._look:
            self._look["host_to_device_arrays"] = int(self._look.get("host_to_device_arrays", 0)) + 1
        return torch.as_tensor(np.ascontiguousarray(array), device=self.device)

    # -- per look -------------------------------------------------------------

    def begin_look(self, snapshot: Any) -> None:
        """Register the look snapshot's host arrays against its device fields."""

        if self._look:
            self.look_receipts.append(dict(self._look))
        fields = getattr(snapshot, "device_fields", None)
        if fields is None:
            raise RuntimeError("the v6 census look needs a v6 device snapshot")
        self._look_twins = []
        self._phase_expected = None
        for name in ("mean", "mean_standard_error", "activity_mean", "activity_mean_standard_error"):
            self._register(getattr(snapshot, name), fields[name])
        self._look = {
            "schema": "irg_v6_census_device_look_v3",
            "snapshot": dict(getattr(snapshot, "v6_receipt", None) or {}),
            "order_field": "loss_activity",
            "sign_criterion_near_tolerance_views": [],
            "labels": None,
            "order_certificate": None,
        }

    def carry_previous(
        self,
        *,
        previous_activity: np.ndarray,
        previous_labels: np.ndarray,
        source_activity: np.ndarray,
        source_labels: np.ndarray,
    ) -> None:
        """Keep the next look's previous-look arrays on the device."""

        self._carried_twins = [
            (np.asarray(previous_activity), self.twin(source_activity).clone()),
            (np.asarray(previous_labels), self.twin(source_labels).clone()),
        ]

    def release_device_state(self) -> None:
        """Drop the device twins (the look is over; the receipts stay)."""

        self._census_twins = []
        self._carried_twins = []
        self._look_twins = []
        self._analytic_labels = None
        self._analytic_mask = None

    def close(self) -> list[dict[str, Any]]:
        if self._look:
            if getattr(self, "_phase_device_state", None) is not None:
                self._look["phase_device_orchestration"] = self._phase_device_state.receipt()
            self.look_receipts.append(dict(self._look))
            self._look = {}
        return list(self.look_receipts)

    # -- sign criterion -------------------------------------------------------

    def sign_criteria(
        self, *, means: Mapping[int, np.ndarray], errors: Mapping[int, np.ndarray],
        activities: Mapping[int, np.ndarray], sign_measurement_weights_by_head: Mapping[int, np.ndarray],
        critical: float, tolerance: float, decision_referenced: bool, host_criterion: Any = None,
    ) -> dict[str, float]:
        """Evaluate sign criteria for the explicitly supplied census heads."""
        from mechanistic_systems.measurement.phase_device_orchestration import census_sign_batch, state_for
        heads = self._sign_heads(means, errors, activities, sign_measurement_weights_by_head)
        names = tuple({0: "local", 1: "global", 2: "loss"}[head] for head in heads)
        risks, expected = census_sign_batch(
            torch.stack([self.twin(means[h]) for h in heads]),
            torch.stack([self.twin(errors[h]) * self.twin(sign_measurement_weights_by_head[h]) for h in heads]),
            torch.stack([self.twin(activities[h]) for h in heads]),
            critical=critical, decision_referenced=decision_referenced,
            state=state_for(self))
        self._phase_expected = dict(zip(names, map(float, expected), strict=True))
        result = dict(zip(names, map(float, risks), strict=True))
        self._look["sign_criterion_heads"] = list(names)
        self._look["sign_criterion_near_tolerance_views"] = [
            name for name, value in result.items() if _near(value, tolerance)]
        return result

    @staticmethod
    def _sign_heads(means, errors, activities, weights) -> tuple[int, ...]:
        heads = tuple(sorted(means))
        if (
            not heads
            or any(head not in (0, 1, 2) for head in heads)
            or set(errors) != set(heads)
            or set(activities) != set(heads)
            or any(head not in weights for head in heads)
        ):
            raise ValueError("census sign head selection is invalid")
        return heads

    def expected_wrongsign(
        self, *, means: Mapping[int, np.ndarray], errors: Mapping[int, np.ndarray],
        activities: Mapping[int, np.ndarray], sign_measurement_weights_by_head: Mapping[int, np.ndarray],
    ) -> dict[str, float]:
        heads = self._sign_heads(means, errors, activities, sign_measurement_weights_by_head)
        names = tuple({0: "local", 1: "global", 2: "loss"}[head] for head in heads)
        if (
            getattr(self, "_phase_expected", None) is not None
            and set(self._phase_expected) == set(names)
        ):
            return dict(self._phase_expected)
        return {name: expected_wrongsign_consequence_fraction_device(
            mean=self.twin(means[head]),
            standard_error=self.twin(errors[head]) * self.twin(sign_measurement_weights_by_head[head]),
            activity=self.twin(activities[head])) for name, head in zip(names, heads, strict=True)}


    # -- labels ---------------------------------------------------------------

    def partition_labels(
        self,
        *,
        means: Mapping[int, np.ndarray],
        host_flat_labels: Any,
    ) -> np.ndarray:
        """Flat int8 partition labels computed on the device.

        The measured labels are the host thresholds of the fused means.  The
        analytic routes (effective-weight signs, measurement-free) take the
        host label code's values, fetched once on the first look and
        kept on the device.  Nonfinite means use the host label code.
        """

        family = self.partition_family
        local = self.twin(means[0])
        global_ = self.twin(means[1])
        if family == "single_population":
            labels = torch.zeros(self.route_count, dtype=torch.int8, device=self.device)
        elif family == "local_attribution_sign":
            labels = torch.where(local >= 0.0, 0, 1).to(torch.int8)
        elif family == "global_attribution_sign":
            labels = torch.where(global_ >= 0.0, 0, 1).to(torch.int8)
        elif family == "local_global_sign_alignment":
            labels = (2 * (local < 0.0).to(torch.int8) + (global_ < 0.0).to(torch.int8)).to(
                torch.int8
            )
        else:  # pragma: no cover - guarded by labels_on_device
            raise RuntimeError(f"no device labels for {family!r}")
        fetched = False
        if self._analytic_mask is not None:
            if self._analytic_labels is None:
                host_labels = np.ascontiguousarray(host_flat_labels(), dtype=np.int8)
                self._analytic_labels = torch.as_tensor(host_labels, device=self.device)
                fetched = True
            labels = torch.where(self._analytic_mask, self._analytic_labels, labels)
        self._look["labels"] = {
            "source": "device",
            "analytic_labels_from_host_this_look": bool(fetched),
        }
        host = np.ascontiguousarray(labels.cpu().numpy(), dtype=np.int8)
        self._register(host, labels)
        return host

    def note_host_labels(self) -> None:
        self._look["labels"] = {"source": "host_v52"}

    # -- order certificate ----------------------------------------------------

    def certify_prune_order_stability(self, **kwargs: Any) -> Any:
        """Drop-in for ``score_order.certify_prune_order_stability``."""

        from .score_order import LayerCodeAxis
        from mechanistic_systems.measurement.phase_device_orchestration import call_with_state, state_for
        layer_axis = LayerCodeAxis.from_layer_ids(kwargs["layer_ids"])
        if tuple(layer_axis.names) != self.layer_names or layer_axis.route_count != self.route_count:
            raise ValueError("census order layer axis differs")
        flags: list[str] = []
        certificate = call_with_state(state_for(self), certify_prune_order_stability_device,
                current_global_activity=self.twin(kwargs["current_global_activity"]),
                previous_global_activity=(
                    None
                    if kwargs.get("previous_global_activity") is None
                    else self.twin(kwargs["previous_global_activity"])
                ),
                current_partition_labels=self.twin(kwargs["current_partition_labels"]),
                previous_partition_labels=(
                    None
                    if kwargs.get("previous_partition_labels") is None
                    else self.twin(kwargs["previous_partition_labels"])
                ),
                layer_codes=self.layer_codes,
                layer_names=self.layer_names,
                maximum_mask_disagreement_activity_fraction=kwargs[
                    "maximum_mask_disagreement_activity_fraction"
                ],
                scope=kwargs["scope"],
                route_axis_hash=kwargs["route_axis_hash"],
                audited_retained_fractions=kwargs.get("audited_retained_fractions"),
                audited_retained_fractions_by_scope=kwargs.get(
                    "audited_retained_fractions_by_scope"
                ),
                activity_normalization=kwargs.get("activity_normalization", "scope"),
                score_standard_error=(
                    None
                    if kwargs.get("score_standard_error") is None
                    else self.twin(kwargs["score_standard_error"])
                ),
                route_damage=(
                    None
                    if kwargs.get("route_damage") is None
                    else self.twin(kwargs["route_damage"])
                ),
                route_damage_standard_error=(
                    None
                    if kwargs.get("route_damage_standard_error") is None
                    else self.twin(kwargs["route_damage_standard_error"])
                ),
                critical_value=kwargs.get("critical_value"),
                consequence_gated=bool(kwargs.get("consequence_gated", False)),
                consequence_floor_critical_value=kwargs.get(
                    "consequence_floor_critical_value", 1.959963984540054
                ),
                consequence_price_deferred=bool(kwargs.get("consequence_price_deferred", False)),
                consequence_materiality_floor=kwargs.get("consequence_materiality_floor"),
                partition_sign_unresolved=(
                    None
                    if kwargs.get("partition_sign_unresolved") is None
                    else self.twin(np.asarray(kwargs["partition_sign_unresolved"], dtype=bool))
                ),
                consequence_price_short_circuit_scope=kwargs.get(
                    "consequence_price_short_circuit_scope"
                ),
                decision_flags=flags,
            )
        # Near ties are decided here and recorded, never re-decided on the host.
        self._look["order_certificate"] = {"source": "device", "band_flags": list(flags)}
        return certificate
