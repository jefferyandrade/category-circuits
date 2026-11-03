"""Prune scope and proposal grain for the modern-v0 walk.

The registered v0 walk, ``run_profiled_native_sampled_path_panel``, prunes one
layer group at a time: a rung of density ``d`` keeps ``ceil(n_{l,p} * d)`` of
each layer's own best-scored routes in each partition, on the profile's nominal
geometric ladder.  That path is unchanged and remains the default.  This module
adds explicitly selected scope and grain settings:

``IRG_PRUNE_SCOPE`` (``layerwise`` | ``global``; unset = ``layerwise``)
    ``global`` walks ONE group made of every eligible connection layer.  On
    the older nominal diagnostic ladder, a rung of pooled density ``d`` keeps,
    in each partition ``p``,

        retained_p(d) = TopK_p(ceil(N_p * d))  union  Floor_p

    where ``TopK_p`` ranks the partition's routes of all layers together by the
    same activity score the layerwise walk ranks within each layer (pooled
    without rescaling), and ``Floor_p`` is each layer's own best
    ``ceil(rho_min * n_{l,p})`` routes (``rho_min`` = the ladder's terminal
    positive rung, the layerwise arm's own terminal kept count).  Kept sets are
    nested as ``d`` falls, no layer can fall below its floor, and with one
    eligible layer the rule is exactly the layerwise lowering.  Ties go to the
    lower declared route-axis index (ascending route-block offset, then C-order
    flat index).

``IRG_PROPOSAL_ROUTE_CAP`` (a positive integer | ``none``; unset = ``none``)
    With a cap, every nominal rung interval of the ladder is divided into the
    fewest equal sub-steps such that no single proposal removes more than the
    cap in any partition (a grain control, floor(gamma/tau)).  The
    nominal rungs stay on the executed ladder, so the capped ladder refines the
    nominal one. The cap applies to either diagnostic scope and is not a
    registered arm. ``CAP_SEMANTICS`` states the rule exactly; every receipt
    carries it.

``IRG_PROPOSAL_GRAIN`` (``layerwise_matched`` | ``nominal``; unset = ``nominal``)
    The registered global comparison uses ``layerwise_matched``. Each
    partition's layerwise rung removals are sorted by nominal size largest first, with ties
    resolved by declared layer order and then rung index. The pooled queue
    loses that many non-floor routes at each position. Nominal and capped
    geometry remain available for tests and diagnostics only.

A variable that is present but blank or whitespace-only is refused:
only an ABSENT variable, or the explicit ``layerwise`` / ``none``,
selects the default, so a launcher row with a missing field cannot silently
become an uncapped layerwise run.

Admission: every non-default setting runs only the
registered comparison of the paper: tau exactly the registered comparison's tau
(``registered_sensitivity_tau()``: 5e-7 by default, or the campaign price set
in ``IRG_REGISTERED_SENSITIVITY_TAU``;
tau unchanged across scopes and not transferred), the architecture family ``alexnet_conv_kernel_weight``, and
exactly AlexNet's seven registered connection layers
(``REGISTERED_ALEXNET_LAYER_GROUPS``).  The generic lowerer and panel below
stay architecture-free, for synthetic tests.

Everything else is the registered machinery, called unchanged: the decision
loop ``trace_bce_precision_frontier`` (stall rule, stop reasons, pricing
per actually removed connection, dBCE / (A * removed) <= tau), the decision
backend, the v6 committed-state and candidate-view types, the walk logs and
the receipts.  Only the proposal geometry (which routes a rung keeps) is new.

Evidence boundary: the layerwise census certifies score order within each
(layer, partition) at the layerwise cut points.  It does not certify the
cross-layer order at global cuts, and no decision reads a cross-layer order
result.  ``cross_layer_order_diagnostic`` reports, after the walk, (a) the
call ``score_order.certify_prune_order_stability(scope="cross_layer")``
on the stored census field with the realized pooled cuts (every committed
cut, capped intermediate cuts included) as its audited fractions; the field
is one census look, so this call's within-look content is the consequence
price, and the block carries that name (``consequence_price``); (b) the
boundary-uncertain activity fraction at every committed cut
(``boundary_uncertain_activity``): the interval-overlap calculation of
``score_order.certify_prune_order_uncertainty``, applied to the non-floor
pooled queue with the per-layer floors pinned; and (c) a descriptive
pointwise band that is labelled as NOT a certificate.  No optimality
certificate over unvisited routes is claimed.  The three parts run as torch
kernels on the device (``cross_layer_order_device``).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from mechanistic_systems.circuits.evidence_bce_precision import (
    BCEPrecisionPolicy,
    V0MarginalConnectionDecisionRule,
    trace_bce_precision_frontier,
)
from mechanistic_systems.circuits.evidence_continuous import NaturalizationContext
from mechanistic_systems.circuits.frontier_rungs import (
    exact_ceil_fraction as _exact_ceil_fraction,
    maximum_decisions_from_retained_counts,
)
from mechanistic_systems.circuits.device_mask_plane import (
    DeviceRemoved,
    OrderCache,
)
from mechanistic_systems.circuits.masking import (
    CandidateView,
    CommittedMaskState,
    HistoricalMaskLoweringResult,
    _LayerSlice,
    _immutable_array,
    _keep_count,
    _v6_stable_descending_order,
    compose_committed_masks,
    historical_density_axis,
)
from mechanistic_systems.circuits.native_continuous import (
    BackendNaturalizationFacts,
    NativeContinuousRun,
    NativeContinuousTraceSignature,
    continuous_group_density_floor,
    frontier_prior_from_history,
    naturalization_facts_from_backend,
    record_frontier_history,
)
from mechanistic_systems.circuits.native_sampled import (
    BCE_PRECISION_ALGORITHM,
    INITIAL_DENSE_ANCHOR_NUMERICAL_TOLERANCE,
)
from mechanistic_systems.circuits.sampled_path import (
    SampledComparisonRequest,
    sampled_path_mask_hash,
)
from mechanistic_systems.inference.sequential_bce import (
    IID_WITH_REPLACEMENT_SAMPLING_ID,
)

PRUNE_SCOPE_ENV = "IRG_PRUNE_SCOPE"
PROPOSAL_ROUTE_CAP_ENV = "IRG_PROPOSAL_ROUTE_CAP"
PROPOSAL_GRAIN_ENV = "IRG_PROPOSAL_GRAIN"
PROPOSAL_GRAIN_NOMINAL = "nominal"
PROPOSAL_GRAIN_LAYERWISE_MATCHED = "layerwise_matched"
PRUNE_SCOPE_LAYERWISE = "layerwise"
PRUNE_SCOPE_GLOBAL = "global"
PRUNE_SCOPES = (PRUNE_SCOPE_LAYERWISE, PRUNE_SCOPE_GLOBAL)
# The walk's group label for the pooled group.  It names no real layer; the
# per-layer removals of every decision live in the walk logs and in the
# ``prune_scope_walk`` receipt block.  No ':' so decision keys stay parseable.
GLOBAL_GROUP_ID = "global_all_eligible_layers"
PRUNE_SCOPE_WALK_SCHEMA = "irg_prune_scope_walk_v2"
KEEP_RULE = "pooled_top_k_union_per_layer_floor_v1"
TIE_RULE = "declared_route_axis_index_ascending_v1"
GRAIN_RULE_NOMINAL = "nominal_geometric_ladder_uncapped_v1"
GRAIN_RULE_CAPPED = "nominal_rung_equal_subdivision_at_most_cap_removed_per_partition_v1"
GRAIN_RULE_LAYERWISE_MATCHED = "layerwise_ladder_steps_pooled_nominal_largest_first_v2"
# The exact meaning of each grain rule, recorded verbatim in every receipt.
# It is the rule ``_refined_ladder`` implements.
CAP_SEMANTICS = {
    GRAIN_RULE_LAYERWISE_MATCHED: (
        "For each partition p and every eligible layer l, nominal rung j "
        "removes r_{l,p,j} = ceil(n_{l,p} * d_{j-1}) - ceil(n_{l,p} * d_j), "
        "using the layerwise ladder d_0 = 1 > ... > d_J = rho_min and the "
        "engine's _keep_count. Sort all L*J steps for p by descending "
        "unrounded nominal size n_{l,p} * (d_{j-1} - d_j), "
        "then ascending declared route-axis layer order, then ascending j. "
        "At shared position k = 0..L*J retain each layer's floor plus the "
        "first N^nf_p minus the sum of the first k step sizes non-floor "
        "routes in pooled score order. The steps sum exactly to N^nf_p, "
        "so the final position is precisely the per-layer floor. Both "
        "partitions share the position index; a zero-size step advances "
        "that partition's cursor without a decision under the frontier's "
        "zero-removal rule. Distinct descending float64 density labels "
        "identify positions; exact retained counts determine the masks."
    ),
    GRAIN_RULE_NOMINAL: (
        "No cap: the executed ladder is the profile's nominal geometric "
        "ladder; one proposal moves one partition from one nominal rung to the "
        "next."
    ),
    GRAIN_RULE_CAPPED: (
        "Cap C: each nominal interval j (rung j-1 to rung j) removes R_{j,p} "
        "routes of partition p on the nominal ladder. It is split into "
        "s_j = max(1, max over partitions p of ceil(R_{j,p} / C)) equal "
        "sub-steps, the same s_j for both partitions; after sub-step i "
        "(1 <= i <= s_j) partition p keeps c_{j-1,p} - floor(R_{j,p} * i / s_j) "
        "of its non-floor routes. So every proposal removes at most C routes "
        "in its partition, the last sub-step lands exactly on nominal rung j "
        "(the capped ladder refines the nominal one), and a partition with "
        "R_{j,p} < s_j has sub-steps that remove nothing. At such a sub-step "
        "the registered frontier moves that partition's cursor on without a "
        "decision and passes the turn to the other partition (no proposal "
        "index is spent and no stalled turn is counted), so in a two-partition "
        "arm the partition with more routes to remove proposes more often. "
        "With one live partition s_j = ceil(R_j / C) <= R_j, so the cap adds "
        "no zero-removal position there. The same rule applies to global "
        "and layerwise diagnostic walks. Capped ladders are not registered "
        "Paper 1 arms."
    ),
}
# Tau is unchanged across scopes and
# not transferred, so every non-default setting runs at exactly this value.
REGISTERED_SENSITIVITY_TAU = 5e-7
TAU_PIN_RULE = "tau_pinned_5e-7_T1_note08_WALKER_PLAN_3.2_v1"
# Every arm of a network shares its
# price. The registered comparison's tau may be set explicitly to the campaign
# price; the rule is unchanged (every non-default setting runs at exactly the
# same tau as the layerwise comparison, never a re-derived one). Absent: 5e-7
# and TAU_PIN_RULE, byte-identical to before. Present: a positive finite float.
REGISTERED_SENSITIVITY_TAU_ENV = "IRG_REGISTERED_SENSITIVITY_TAU"
TAU_PIN_RULE_CAMPAIGN = "tau_pinned_campaign_price_T1_note08_WALKER_PLAN_3.2_v2"


def registered_sensitivity_tau(environ: Mapping[str, str] | None = None) -> float:
    """The registered comparison's tau: 5e-7 unless IRG_REGISTERED_SENSITIVITY_TAU sets it.

    Present but blank, non-numeric, non-finite or not positive is refused (as
    for the other settings of this module, only an absent variable selects the
    default)."""
    env = os.environ if environ is None else environ
    if REGISTERED_SENSITIVITY_TAU_ENV not in env:
        return REGISTERED_SENSITIVITY_TAU
    raw = env[REGISTERED_SENSITIVITY_TAU_ENV]
    try:
        value = float(str(raw).strip())
    except ValueError:
        value = math.nan
    if not str(raw).strip() or not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"{REGISTERED_SENSITIVITY_TAU_ENV} must be a positive finite tau when set, got {raw!r}"
        )
    return value


def tau_pin_rule(environ: Mapping[str, str] | None = None) -> str:
    """The receipt's tau-pin rule: the original name unless the campaign price is set."""
    env = os.environ if environ is None else environ
    return TAU_PIN_RULE if REGISTERED_SENSITIVITY_TAU_ENV not in env else TAU_PIN_RULE_CAMPAIGN
# Note 08's reference grain: gamma = 0.05 of the loss span per proposal at
# tau 5e-7 gives floor(gamma / tau) = 100,000 routes per proposal.
NOTE08_REFERENCE_GAMMA = 0.05
NOTE08_REFERENCE_CAP = 100_000
# A non-default setting is admitted only for the registered
# network of the paper, AlexNet at kernel-weight granularity, and only when its
# eligible connection groups are exactly these seven (stable layer ID,
# route kind, route count).  Counts are out * in * kh * kw for the five
# convolutions and out * in for the two hidden fully connected layers; the
# fixed readout (readout.logits) is not an eligible connection group.
REGISTERED_ARCHITECTURE_FAMILY = "alexnet_conv_kernel_weight"
REGISTERED_ALEXNET_LAYER_GROUPS = (
    ("features.stage1.conv", "conv_kernel_weight", 64 * 3 * 11 * 11),
    ("features.stage2.conv", "conv_kernel_weight", 192 * 64 * 5 * 5),
    ("features.stage3.conv", "conv_kernel_weight", 384 * 192 * 3 * 3),
    ("features.stage4.conv", "conv_kernel_weight", 256 * 384 * 3 * 3),
    ("features.stage5.conv", "conv_kernel_weight", 256 * 256 * 3 * 3),
    ("readout.hidden1", "linear_weight", 4096 * 9216),
    ("readout.hidden2", "linear_weight", 4096 * 4096),
)
CROSS_LAYER_ORDER_DIAGNOSTIC_SCHEMA = "prune_scope_cross_layer_order_diagnostic_v2"
POINTWISE_BAND_SCHEMA = "global_cut_pointwise_band_descriptive_v2"
BOUNDARY_UNCERTAINTY_SCHEMA = "prune_scope_boundary_uncertain_activity_v1"
# The declared denominator of the boundary-uncertain activity fraction: the
# ranked activity summed over every eligible route of the partition, pinned
# floor routes included.  It is the denominator
# ``score_order.certify_prune_order_uncertainty`` uses for a cross-layer
# scope ``("global", partition)``.
BOUNDARY_UNCERTAINTY_DENOMINATOR = "partition_activity_all_eligible_routes_floors_included"
# Two-sided 95% normal quantile for the descriptive band only.
ORDER_DIAGNOSTIC_CRITICAL_VALUE = 1.959963984540054
# The two partition families whose registered layerwise arms (SA-H and P) have
# matched global counterparts (SAG and SG). Every other family is refused.
PRUNE_SCOPE_FAMILIES = ("single_population", "local_attribution_sign")
_PLAIN_ATTRIBUTION_ORDER = "plain_global_attribution"
_ELIGIBLE_ROUTE_KINDS = frozenset(
    {"linear_weight", "conv_channel_pair", "conv_kernel_weight"}
)
_PARTITIONS = ("positive", "negative")
_PARTITION_CODE = {"positive": 0, "negative": 1}
_CAP_PATTERN = re.compile(r"[0-9]+")


# ---------------------------------------------------------------------------
# The setting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PruneScopeSetting:
    """The resolved scope and proposal grain of one walk."""

    scope: str = PRUNE_SCOPE_LAYERWISE
    proposal_route_cap: int | None = None
    proposal_grain: str = PROPOSAL_GRAIN_NOMINAL

    def __post_init__(self) -> None:
        if self.scope not in PRUNE_SCOPES:
            raise ValueError(f"unknown prune scope {self.scope!r}")
        cap = self.proposal_route_cap
        if cap is not None and (
            isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0
        ):
            raise ValueError("the proposal route cap must be a positive integer")
        if self.proposal_grain not in {PROPOSAL_GRAIN_NOMINAL, PROPOSAL_GRAIN_LAYERWISE_MATCHED}:
            raise ValueError(f"unknown proposal grain {self.proposal_grain!r}")
        if cap is not None and self.proposal_grain == PROPOSAL_GRAIN_LAYERWISE_MATCHED:
            raise ValueError("layerwise-matched grain cannot also have a route cap")
        if self.proposal_grain == PROPOSAL_GRAIN_LAYERWISE_MATCHED and self.scope != PRUNE_SCOPE_GLOBAL:
            raise ValueError("layerwise-matched grain requires global scope")

    @property
    def is_default(self) -> bool:
        """True for the registered layerwise walk on the nominal ladder."""

        return (
            self.scope == PRUNE_SCOPE_LAYERWISE
            and self.proposal_route_cap is None
            and self.proposal_grain == PROPOSAL_GRAIN_NOMINAL
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "prune_scope": self.scope,
            "proposal_route_cap": self.proposal_route_cap,
        }
        if self.proposal_grain != PROPOSAL_GRAIN_NOMINAL:
            result["proposal_grain"] = self.proposal_grain
        return result


def resolve_prune_scope_setting(
    environ: Mapping[str, str] | None = None,
) -> PruneScopeSetting:
    """Read scope, route cap and grain variables; refuse unknowns.

    An ABSENT variable gives the registered default (layerwise, no cap).  A
    variable that is present but blank or whitespace-only is refused: a
    launcher that exports an empty field has lost the arm's setting, and must
    not run the default arm in its place.  The scope accepts only
    ``layerwise`` or ``global`` (case-insensitive, surrounding spaces
    ignored).  The cap accepts only ``none`` or a positive decimal integer
    (no sign, no exponent, no fraction).
    """

    source = os.environ if environ is None else environ
    raw_scope = source.get(PRUNE_SCOPE_ENV)
    if raw_scope is not None and not str(raw_scope).strip():
        raise ValueError(
            f"{PRUNE_SCOPE_ENV} is set but blank ({raw_scope!r}); unset it for "
            "the default layerwise walk or set 'layerwise' or 'global'"
        )
    scope = (
        PRUNE_SCOPE_LAYERWISE if raw_scope is None else str(raw_scope).strip().lower()
    )
    if scope not in PRUNE_SCOPES:
        raise ValueError(
            f"{PRUNE_SCOPE_ENV} must be 'layerwise' or 'global', got {raw_scope!r}"
        )
    raw_cap = source.get(PROPOSAL_ROUTE_CAP_ENV)
    if raw_cap is not None and not str(raw_cap).strip():
        raise ValueError(
            f"{PROPOSAL_ROUTE_CAP_ENV} is set but blank ({raw_cap!r}); unset it "
            "for no cap or set 'none' or a positive integer"
        )
    cap: int | None
    if raw_cap is None or str(raw_cap).strip().lower() == "none":
        cap = None
    else:
        text = str(raw_cap).strip()
        if _CAP_PATTERN.fullmatch(text) is None or int(text) <= 0:
            raise ValueError(
                f"{PROPOSAL_ROUTE_CAP_ENV} must be 'none' or a positive integer, "
                f"got {raw_cap!r}"
            )
        cap = int(text)
    raw_grain = source.get(PROPOSAL_GRAIN_ENV)
    if raw_grain is not None and not str(raw_grain).strip():
        raise ValueError(f"{PROPOSAL_GRAIN_ENV} is set but blank ({raw_grain!r})")
    grain = PROPOSAL_GRAIN_NOMINAL if raw_grain is None else str(raw_grain).strip().lower()
    if grain not in {PROPOSAL_GRAIN_NOMINAL, PROPOSAL_GRAIN_LAYERWISE_MATCHED}:
        raise ValueError(f"{PROPOSAL_GRAIN_ENV} must be 'nominal' or 'layerwise_matched', got {raw_grain!r}")
    return PruneScopeSetting(scope=scope, proposal_route_cap=cap, proposal_grain=grain)


def validate_prune_scope_admission(
    setting: PruneScopeSetting,
    *,
    version: str,
    family: str,
    schedule_replay_requested: bool,
    layer_execution_order: str,
    v0_attribution_order: str,
    direct_candidate_comparison: bool,
    architecture_family: str,
    sensitivity_tau: float,
    diagnostic_only: bool = False,
) -> None:
    """Admit only the registered comparison, with an explicit diagnostic escape.

    The default setting is admitted everywhere (it takes the registered path
    and this check returns before reading anything).  Every non-default
    setting is checked against the paper's execution context: the v0 walk on
    the v6 control plane, the plain
    attribution order, the SA-H or P partition family, tau exactly
    ``registered_sensitivity_tau()`` and the AlexNet kernel-weight family.  The
    seven layer groups themselves are checked once the runtime exists
    (``validate_registered_prune_scope_route_groups``).
    Registered global walks require the matched grain; nominal and capped
    walks are selectable only when ``diagnostic_only`` is true or by direct
    use of the lowerer and panel in tests.
    """

    if setting.is_default:
        return
    label = (
        f"{PRUNE_SCOPE_ENV}={setting.scope} "
        f"{PROPOSAL_ROUTE_CAP_ENV}={setting.proposal_route_cap} "
        f"{PROPOSAL_GRAIN_ENV}={setting.proposal_grain}"
    )
    tau = sensitivity_tau
    registered_tau = registered_sensitivity_tau()
    if (
        isinstance(tau, bool)
        or not isinstance(tau, (int, float))
        or float(tau) != registered_tau
    ):
        raise ValueError(
            f"{label} runs only at the registered tau {registered_tau!r} "
            f"(T1 note 08, WALKER_PLAN 3.2: tau unchanged and not transferred), "
            f"got {sensitivity_tau!r}; pass the registered tau explicitly when "
            "the profile's value differs"
        )
    if str(architecture_family).lower() != REGISTERED_ARCHITECTURE_FAMILY:
        raise ValueError(
            f"{label} is registered only for the architecture family "
            f"{REGISTERED_ARCHITECTURE_FAMILY!r}, got {architecture_family!r}"
        )
    if str(version) != "v0":
        raise ValueError(f"{label} requires the registered v0 walk, got {version!r}")
    if schedule_replay_requested:
        raise ValueError(f"{label} is undefined for the schedule-replay walker")
    if direct_candidate_comparison:
        raise ValueError(
            f"{label} is undefined for the modern-v1 direct-candidate adapter"
        )
    if str(v0_attribution_order) != _PLAIN_ATTRIBUTION_ORDER:
        raise ValueError(
            f"{label} is defined only for the plain attribution order, got "
            f"{v0_attribution_order!r}"
        )
    if str(family) not in PRUNE_SCOPE_FAMILIES:
        raise ValueError(
            f"{label} is defined only for the families {PRUNE_SCOPE_FAMILIES!r}, "
            f"got {family!r}"
        )
    order = str(layer_execution_order)
    if setting.scope == PRUNE_SCOPE_GLOBAL and order != "forward":
        raise ValueError(
            f"{label}: a global walk has one group, so only the default layer "
            f"execution order 'forward' is admitted, got {order!r}"
        )
    if order not in {"forward", "reverse"}:
        raise ValueError(f"unknown layer execution order {order!r}")
    if not diagnostic_only:
        if setting.proposal_route_cap is not None:
            raise ValueError("a route cap is available only in tests and diagnostics, not a registered arm")
        if (
            setting.scope == PRUNE_SCOPE_GLOBAL
            and setting.proposal_grain != PROPOSAL_GRAIN_LAYERWISE_MATCHED
        ):
            raise ValueError(
                "a registered global walk requires "
                f"{PROPOSAL_GRAIN_ENV}={PROPOSAL_GRAIN_LAYERWISE_MATCHED}"
            )


def validate_registered_prune_scope_route_groups(
    setting: PruneScopeSetting,
    route_groups: Sequence[Any],
    *,
    architecture_family: str,
) -> dict[str, Any]:
    """Refuse a non-default setting unless the runtime is AlexNet's seven groups.

    Called by the pilot once the runtime's route groups exist, before the
    census and the walk.  The eligible connection groups (the route kinds the
    walk prunes) must be exactly ``REGISTERED_ALEXNET_LAYER_GROUPS``: the same
    stable layer IDs, route kinds and route counts, no more and no fewer.
    Groups of other kinds (never pruned by any v0 walk) are listed in the
    returned record and otherwise ignored.  The default setting is not
    checked.  Returns the record the receipt carries.
    """

    if setting.is_default:
        return {"checked": False, "reason": "default_setting"}
    family = str(architecture_family).lower()
    if family != REGISTERED_ARCHITECTURE_FAMILY:
        raise ValueError(
            "a non-default prune scope or grain is registered only for "
            f"{REGISTERED_ARCHITECTURE_FAMILY!r}, got {architecture_family!r}"
        )
    observed: list[tuple[str, str, int]] = []
    other: list[str] = []
    for group in route_groups:
        kind = str(group.route_kind)
        if kind not in _ELIGIBLE_ROUTE_KINDS:
            other.append(str(group.layer_id))
            continue
        observed.append(
            (str(group.layer_id), kind, int(np.prod(tuple(group.shape), dtype=np.int64)))
        )
    expected = sorted(REGISTERED_ALEXNET_LAYER_GROUPS)
    if sorted(observed) != expected or len(observed) != len(expected):
        raise ValueError(
            "a non-default prune scope or grain is registered only for AlexNet's "
            "seven connection layers "
            f"{[layer for layer, _, _ in REGISTERED_ALEXNET_LAYER_GROUPS]!r} with "
            f"their registered kinds and route counts; the runtime's eligible "
            f"groups are {observed!r}"
        )
    return {
        "checked": True,
        "architecture_family": family,
        "registered_layer_groups": [
            {"layer_id": layer, "route_kind": kind, "route_count": int(count)}
            for layer, kind, count in REGISTERED_ALEXNET_LAYER_GROUPS
        ],
        "eligible_route_count": int(sum(count for _, _, count in observed)),
        "non_eligible_group_ids": other,
    }


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------


def executable_ladder_for_kinds(
    route_group_kinds: Sequence[str],
    profile: Any,
) -> tuple[tuple[float, ...], float]:
    """The registered executable ladder and floor, common to every kind given.

    Each kind's ladder is computed exactly as the layerwise group runner does
    (``native_continuous.run_profiled_native_bce_precision_group``).  A pooled
    group needs one ladder, so kinds whose ladders differ are refused.
    """

    kinds = tuple(dict.fromkeys(str(kind) for kind in route_group_kinds))
    if not kinds:
        raise ValueError("a ladder needs at least one route-group kind")
    ladders: dict[str, tuple[tuple[float, ...], float]] = {}
    for kind in kinds:
        minimum_density = continuous_group_density_floor(kind, profile)
        schedule = tuple(
            value
            for value in historical_density_axis(
                kind,
                conv_decay=float(profile.target_preservation_conv_decay),
                conv_target_final_density=float(
                    profile.target_preservation_conv_target_final_density
                ),
                linear_decay=float(profile.target_preservation_linear_decay),
                linear_target_final_density=float(
                    profile.target_preservation_linear_target_final_density
                ),
            )
            if float(value) >= float(minimum_density) - 1e-12 and float(value) > 0.0
        )
        ladders[kind] = (schedule, float(minimum_density))
    distinct = set(ladders.values())
    if len(distinct) != 1:
        raise ValueError(
            "a pooled group needs one ladder, but its route-group kinds have "
            "different ladders (conv and linear decay or target differ): "
            + ", ".join(
                f"{kind}: {len(value[0])} rungs, floor {value[1]!r}"
                for kind, value in ladders.items()
            )
        )
    return next(iter(distinct))


def _refined_ladder(
    nominal: tuple[float, ...],
    retained_by_partition: Mapping[str, tuple[int, ...]],
    cap: int,
) -> tuple[tuple[float, ...], dict[str, tuple[int, ...]]]:
    """Divide each nominal interval so no proposal removes more than ``cap``.

    Interval ``j`` (``nominal[j-1] -> nominal[j]``) removes ``R_{j,p}`` routes
    of partition ``p``.  It is split into ``s_j = max(1, max_p ceil(R_{j,p} /
    cap))`` equal sub-steps; sub-step ``i`` retains ``c_{j-1,p} -
    floor(R_{j,p} * i / s_j)`` routes, so every sub-step removes at most
    ``ceil(R_{j,p} / s_j) <= cap`` routes and the last one lands exactly on
    the nominal rung.  The intermediate density values are linear ladder
    positions between the two nominal rungs; the kept counts are exact.
    """

    schedule: list[float] = [float(nominal[0])]
    counts: dict[str, list[int]] = {
        partition: [int(values[0])] for partition, values in retained_by_partition.items()
    }
    for j in range(1, len(nominal)):
        upper = float(nominal[j - 1])
        lower = float(nominal[j])
        removals = {
            partition: int(values[j - 1]) - int(values[j])
            for partition, values in retained_by_partition.items()
        }
        if any(value < 0 for value in removals.values()):
            raise ValueError("the nominal ladder is not nested")
        steps = max(
            [1] + [int(math.ceil(value / cap)) for value in removals.values()]
        )
        for index in range(1, steps + 1):
            position = (
                lower if index == steps else upper - (upper - lower) * index / steps
            )
            if not position < schedule[-1]:
                raise ValueError(
                    "the capped ladder cannot place distinct density positions "
                    "between two nominal rungs at float64 resolution"
                )
            schedule.append(float(position))
            for partition, values in retained_by_partition.items():
                counts[partition].append(
                    int(values[j - 1]) - (removals[partition] * index) // steps
                )
    return tuple(schedule), {key: tuple(value) for key, value in counts.items()}


def _layerwise_matched_ladder(
    nominal: tuple[float, ...],
    layers: tuple[str, ...],
    layer_partition_route_counts: Mapping[tuple[str, str], int],
    floor_counts: Mapping[tuple[str, str], int],
    non_floor_counts: Mapping[str, int],
) -> tuple[tuple[float, ...], dict[str, tuple[int, ...]], dict[str, list[dict[str, Any]]]]:
    """Pool each partition's exact layerwise rung sizes, largest first."""

    steps_per_partition: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, tuple[int, ...]] = {}
    step_count = len(layers) * (len(nominal) - 1)
    for partition in _PARTITIONS:
        steps: list[tuple[int, int, int]] = []
        for layer_index, layer in enumerate(layers):
            size = int(layer_partition_route_counts[(layer, partition)])
            for rung in range(1, len(nominal)):
                removed = _keep_count(size, nominal[rung - 1]) - _keep_count(
                    size, nominal[rung]
                )
                if removed < 0:
                    raise AssertionError("the layerwise ladder must be nested")
                steps.append((int(removed), layer_index, rung))
        steps.sort(
            key=lambda item: (
                -float(layer_partition_route_counts[(layers[item[1]], partition)])
                * (nominal[item[2] - 1] - nominal[item[2]]),
                item[1],
                item[2],
            )
        )
        non_floor = int(non_floor_counts[partition])
        expected = sum(
            int(layer_partition_route_counts[(layer, partition)])
            - int(floor_counts[(layer, partition)])
            for layer in layers
        )
        if sum(item[0] for item in steps) != non_floor or non_floor != expected:
            raise AssertionError("layerwise steps must sum exactly to pooled non-floor routes")
        retained = [non_floor]
        table: list[dict[str, Any]] = []
        for removed, layer_index, rung in steps:
            retained.append(retained[-1] - removed)
            table.append(
                {"size": removed, "source_layer": layers[layer_index], "source_rung": rung}
            )
        if len(steps) != step_count or retained[-1] != 0:
            raise AssertionError("matched ladder must end exactly at the floor")
        steps_per_partition[partition] = table
        counts[partition] = tuple(retained)
    # Keep density labels in [rho_min, 1] for the unchanged frontier policy.
    floor = float(nominal[-1])
    schedule = tuple(
        floor if index == step_count else float(1.0 - (1.0 - floor) * index / step_count)
        for index in range(step_count + 1)
    )
    if schedule[0] != 1.0 or schedule[-1] != floor or any(
        later >= earlier for earlier, later in zip(schedule, schedule[1:])
    ):
        raise ValueError("matched ladder needs distinct descending float64 positions")
    return schedule, counts, steps_per_partition


# ---------------------------------------------------------------------------
# The pooled lowerer
# ---------------------------------------------------------------------------


class PooledCandidateLowerer:
    """Lower ladder positions to masks for one group of one or more layers.

    The kept set of partition ``p`` at a ladder position is its floor routes
    plus the first ``c`` of its non-floor routes in pooled descending score
    order; ``c`` is a pure function of the ladder position (see
    ``retained_non_floor``).  With one layer and no cap this is exactly
    ``masking.HistoricalPartitionedCandidateLowerer``.

    It offers the v6 lowerer interface the frontier uses: ``build_candidate``
    (legacy materialization, also ``__call__``), ``dense_group``,
    ``initial_state`` and ``build_candidate_view``.
    """

    def __init__(
        self,
        scores: Mapping[str, Any],
        partition_labels: Mapping[str, Any],
        *,
        nominal_schedule: Sequence[float],
        floor_fraction: float,
        previous_group_masks: Mapping[str, Any] | None = None,
        proposal_route_cap: int | None = None,
        proposal_grain: str = PROPOSAL_GRAIN_NOMINAL,
    ) -> None:
        if not scores:
            raise ValueError("a pooled group needs at least one layer")
        if {str(key) for key in scores} != {str(key) for key in partition_labels}:
            raise ValueError("score and partition-label layers must match exactly")
        floor_fraction = float(floor_fraction)
        if not 0.0 < floor_fraction <= 1.0:
            raise ValueError("the floor fraction must lie in (0, 1]")
        nominal = tuple(float(value) for value in nominal_schedule)
        if (
            len(nominal) < 2
            or nominal[0] != 1.0
            or any(later >= earlier for earlier, later in zip(nominal, nominal[1:]))
            or nominal[-1] < floor_fraction - 1e-12
        ):
            raise ValueError(
                "the nominal ladder must descend from 1.0 to no lower than the floor"
            )
        if proposal_route_cap is not None and (
            isinstance(proposal_route_cap, bool)
            or not isinstance(proposal_route_cap, int)
            or proposal_route_cap <= 0
        ):
            raise ValueError("the proposal route cap must be a positive integer")
        if proposal_grain not in {PROPOSAL_GRAIN_NOMINAL, PROPOSAL_GRAIN_LAYERWISE_MATCHED}:
            raise ValueError(f"unknown proposal grain {proposal_grain!r}")
        if proposal_grain == PROPOSAL_GRAIN_LAYERWISE_MATCHED and proposal_route_cap is not None:
            raise ValueError("layerwise-matched grain cannot also have a route cap")
        if proposal_grain == PROPOSAL_GRAIN_LAYERWISE_MATCHED and nominal[-1] != floor_fraction:
            raise ValueError("the matched ladder's last nominal rung must be its floor fraction")
        self.scores = scores
        self.partition_labels = partition_labels
        self.previous_group_masks = previous_group_masks
        self.floor_fraction = floor_fraction
        self.proposal_route_cap = proposal_route_cap
        self.proposal_grain = proposal_grain
        self.nominal_schedule = nominal
        self.group_layers = tuple(str(layer) for layer in scores)
        labels_by_layer = {str(key): value for key, value in partition_labels.items()}
        self._shapes: dict[str, tuple[int, ...]] = {}
        self._sizes: dict[str, int] = {}
        flat_scores: list[np.ndarray] = []
        flat_labels: list[np.ndarray] = []
        for layer in self.group_layers:
            raw_score = scores[layer]
            if hasattr(raw_score, "detach"):
                raw_score = raw_score.detach().cpu().numpy()
            score = np.asarray(raw_score)
            label = np.asarray(labels_by_layer[layer])
            if score.dtype == object or label.dtype == object:
                raise TypeError("route score and partition arrays must be numeric")
            if score.shape != label.shape:
                raise ValueError(f"score/partition-label shape mismatch for {layer!r}")
            if not np.isfinite(score).all():
                raise ValueError(f"score arrays must be finite for layer {layer!r}")
            if label.dtype.kind not in {"i", "u", "b"} or not np.isin(label, (0, 1)).all():
                raise ValueError(
                    f"partition labels must be integer zero/one for layer {layer!r}"
                )
            self._shapes[layer] = tuple(int(dim) for dim in score.shape)
            self._sizes[layer] = int(score.size)
            flat_scores.append(np.asarray(score, dtype=np.float64).reshape(-1))
            flat_labels.append(label.reshape(-1))
        if previous_group_masks is not None:
            for layer in self.group_layers:
                if layer in previous_group_masks and not bool(
                    np.all(np.asarray(previous_group_masks[layer], dtype=bool))
                ):
                    raise ValueError(
                        "a pooled group is walked once: its own layers must be "
                        f"dense in the previous groups' masks ({layer!r} is not)"
                    )
        # Concatenation in group order is the declared route-axis order: the
        # caller passes the group's layers by ascending route-block offset.
        self._offsets = np.zeros(len(self.group_layers) + 1, dtype=np.int64)
        for index, layer in enumerate(self.group_layers):
            self._offsets[index + 1] = self._offsets[index] + self._sizes[layer]
        self.total_route_count = int(self._offsets[-1])
        self._flat_scores = np.concatenate(flat_scores)
        labels_flat = np.concatenate(flat_labels)
        del flat_scores, flat_labels
        self.partition_activity: dict[str, float] = {}
        self._floor_members: dict[str, np.ndarray] = {}
        self._floor_pooled_positions: dict[str, np.ndarray] = {}
        self._non_floor: dict[str, np.ndarray] = {}
        self.partition_route_counts: dict[str, int] = {}
        self.layer_partition_route_counts: dict[tuple[str, str], int] = {}
        self.floor_counts: dict[tuple[str, str], int] = {}
        pooled_keep: dict[str, Callable[[float], int]] = {}
        for partition in _PARTITIONS:
            members = np.flatnonzero(labels_flat == _PARTITION_CODE[partition]).astype(
                np.int64, copy=False
            )
            self.partition_activity[partition] = float(np.sum(self._flat_scores[members]))
            size = int(members.size)
            self.partition_route_counts[partition] = size
            layer_of_member = (
                np.searchsorted(self._offsets, members, side="right") - 1
            ).astype(np.int64, copy=False)
            for index, layer in enumerate(self.group_layers):
                count = int(np.count_nonzero(layer_of_member == index))
                self.layer_partition_route_counts[(layer, partition)] = count
                self.floor_counts[(layer, partition)] = _keep_count(count, floor_fraction)
            if size == 0:
                self._floor_members[partition] = np.empty(0, dtype=np.int64)
                self._non_floor[partition] = np.empty(0, dtype=np.int64)
                self._floor_pooled_positions[partition] = np.empty(0, dtype=np.int64)
                pooled_keep[partition] = lambda _density: 0
                continue
            # One verified stable descending order over the pooled partition:
            # ties keep the lower concatenated (declared route-axis) index.
            order = _v6_stable_descending_order(self._flat_scores[members])
            ordered_members = members[order]
            ordered_layers = layer_of_member[order]
            del order
            # Restricted to one layer, a stable order is that layer's own
            # stable order, so its first floor_l entries are the layer's
            # best-scored floor routes.
            is_floor = np.zeros(size, dtype=bool)
            for index, layer in enumerate(self.group_layers):
                floor = self.floor_counts[(layer, partition)]
                if floor:
                    positions = np.flatnonzero(ordered_layers == index)[:floor]
                    is_floor[positions] = True
            non_floor_prefix = np.cumsum(~is_floor, dtype=np.int64)
            # Pooled-order positions of the floor routes (a few per layer);
            # reporting only, for ``pooled_prefix_length``.
            self._floor_pooled_positions[partition] = _immutable_array(
                np.flatnonzero(is_floor), dtype=np.int64
            )
            self._floor_members[partition] = _immutable_array(
                np.sort(ordered_members[is_floor]), dtype=np.int64
            )
            self._non_floor[partition] = _immutable_array(
                ordered_members[~is_floor], dtype=np.int64
            )
            del ordered_members, ordered_layers, is_floor, members, layer_of_member

            def keep(density: float, *, _prefix=non_floor_prefix, _size=size) -> int:
                pooled = _keep_count(_size, density)
                return 0 if pooled <= 0 else int(_prefix[pooled - 1])

            pooled_keep[partition] = keep
        # Kept non-floor count per nominal rung: c(d) = number of non-floor
        # routes among the pooled top ceil(N_p * d).  Retained set =
        # Floor_p union first-c(d) non-floor routes, i.e. TopK union Floor.
        nominal_counts = {
            partition: tuple(int(pooled_keep[partition](value)) for value in nominal)
            for partition in _PARTITIONS
        }
        del pooled_keep
        self.step_table: dict[str, list[dict[str, Any]]] | None = None
        if proposal_grain == PROPOSAL_GRAIN_LAYERWISE_MATCHED:
            self.schedule, retained_counts, self.step_table = _layerwise_matched_ladder(
                nominal,
                self.group_layers,
                self.layer_partition_route_counts,
                self.floor_counts,
                {partition: int(self._non_floor[partition].size) for partition in _PARTITIONS},
            )
            self.grain_rule = GRAIN_RULE_LAYERWISE_MATCHED
        elif proposal_route_cap is None:
            self.schedule = nominal
            retained_counts = nominal_counts
            self.grain_rule = GRAIN_RULE_NOMINAL
        else:
            self.schedule, retained_counts = _refined_ladder(
                nominal, nominal_counts, int(proposal_route_cap)
            )
            self.grain_rule = GRAIN_RULE_CAPPED
        self._retained_by_position = {
            partition: {
                float(position): int(count)
                for position, count in zip(self.schedule, retained_counts[partition])
            }
            for partition in _PARTITIONS
        }
        self.retained_counts_by_rung = retained_counts
        for partition in _PARTITIONS:
            if retained_counts[partition][0] != int(self._non_floor[partition].size):
                raise AssertionError("density 1 must keep every non-floor route")
        # The pooled orders resident on the walk's device plane, and the
        # per-layer counts of each rung's kept prefix (exact integers), so a
        # candidate view is formed with no host gather, sort or device read.
        self._device_orders = OrderCache()
        self._rung_layer_counts: dict[str, dict[int, np.ndarray]] = {}

    # -- ladder arithmetic ----------------------------------------------------
    def retained_non_floor(self, partition: str, density: float) -> int:
        """Kept non-floor routes of ``partition`` at a position of this ladder."""

        table = self._retained_by_position[partition]
        key = float(density)
        if key not in table:
            raise ValueError(
                f"density {density!r} is not a position of this group's ladder"
            )
        return table[key]

    def decision_limit(self) -> int:
        """A frontier decision limit this group's walk cannot reach.

        Under the stall rule each refusal is followed by an opposing commit
        or ends the group. Thus the bound is 2 * sum_p J_p + 2, where J_p
        counts only this partition's count-changing steps under the current
        rule.
        """

        return maximum_decisions_from_retained_counts(self.retained_counts_by_rung)

    def maximum_proposal_removed_routes(self) -> int:
        """The largest removal of any single ladder step, over both partitions."""

        largest = 0
        for counts in self.retained_counts_by_rung.values():
            for earlier, later in zip(counts, counts[1:]):
                largest = max(largest, int(earlier) - int(later))
        return largest

    def zero_removal_steps(self) -> dict[str, int]:
        """Ladder steps that remove nothing, per partition with routes.

        On the capped ladder a partition whose nominal interval removes fewer
        routes than the interval's sub-step count has such steps (see
        ``CAP_SEMANTICS``); the frontier skips them without a proposal.
        """

        return {
            partition: int(
                sum(
                    1
                    for earlier, later in zip(counts, counts[1:])
                    if int(earlier) == int(later)
                )
            )
            for partition, counts in self.retained_counts_by_rung.items()
            if self.partition_route_counts[partition] > 0
        }

    def pooled_prefix_length(self, partition: str, density: float) -> int:
        """The shortest pooled score prefix that, with the floors, is the kept set.

        At a ladder position the partition keeps its floor routes plus its
        first ``c`` non-floor routes in pooled order.  That set equals
        ``TopK_p(k) union Floor_p`` exactly for ``k`` = the pooled position of
        the ``c``-th non-floor route plus one (``k = 0`` when ``c = 0``: the
        floors alone, a within-layer cut).  This is how a realized pooled cut
        is translated into the within-partition retained fraction a
        cross-layer order certificate audits.
        """

        c = self.retained_non_floor(partition, density)
        if c <= 0:
            return 0
        floor_positions = self._floor_pooled_positions[partition]
        # The c-th non-floor route comes after floor route i exactly when the
        # number of non-floor routes before it, F_i - i, is at most c - 1.
        before = int(
            np.searchsorted(
                floor_positions - np.arange(floor_positions.size, dtype=np.int64),
                c - 1,
                side="right",
            )
        )
        return int(c + before)

    def _layer_counts_at(self, partition: str, count: int) -> np.ndarray:
        """Per-layer counts of ``_non_floor[partition][:count]`` (a rung's kept prefix)."""

        width = len(self.group_layers)
        members = self._non_floor[partition]
        table = self._rung_layer_counts.get(partition)
        if table is None:
            rungs = sorted({0, *(int(value) for value in self.retained_counts_by_rung[partition])})
            running = np.zeros(width, dtype=np.int64)
            table = {0: running.copy()}
            for low, high in zip(rungs, rungs[1:]):
                running = running + np.bincount(
                    self._layer_index(members[low:high]), minlength=width
                )
                table[high] = running.copy()
            self._rung_layer_counts[partition] = table
        counts = table.get(int(count))
        if counts is None:
            counts = table[int(count)] = np.bincount(
                self._layer_index(members[: int(count)]), minlength=width
            ).astype(np.int64)
        return counts

    def _layer_index(self, flat: np.ndarray) -> np.ndarray:
        return (np.searchsorted(self._offsets, flat, side="right") - 1).astype(
            np.int64, copy=False
        )

    def removed_by_layer(
        self, partition: str, previous_density: float, proposed_density: float
    ) -> dict[str, int]:
        """Per-layer removal counts of one ladder step (reporting)."""

        c_old = self.retained_non_floor(partition, previous_density)
        c_new = self.retained_non_floor(partition, proposed_density)
        if c_new > c_old:
            raise ValueError("a ladder step may only lower a kept count")
        chosen = self._non_floor[partition][c_new:c_old]
        counts = np.bincount(self._layer_index(chosen), minlength=len(self.group_layers))
        return {
            layer: int(counts[index])
            for index, layer in enumerate(self.group_layers)
            if int(counts[index])
        }

    # -- lowering -------------------------------------------------------------
    def _lowered_flat(self, positive_density: float, negative_density: float) -> tuple[np.ndarray, dict[str, int]]:
        densities = {"positive": float(positive_density), "negative": float(negative_density)}
        flat = np.zeros(self.total_route_count, dtype=bool)
        retained: dict[str, int] = {}
        for partition in _PARTITIONS:
            if self.partition_route_counts[partition] == 0:
                retained[partition] = 0
                continue
            count = self.retained_non_floor(partition, densities[partition])
            flat[self._floor_members[partition]] = True
            flat[self._non_floor[partition][:count]] = True
            retained[partition] = int(self._floor_members[partition].size) + int(count)
        return flat, retained

    def _layer_masks(self, flat: np.ndarray) -> dict[str, np.ndarray]:
        return {
            layer: flat[self._offsets[index] : self._offsets[index + 1]]
            .copy()
            .reshape(self._shapes[layer])
            for index, layer in enumerate(self.group_layers)
        }

    def lower(self, positive_density: float, negative_density: float) -> HistoricalMaskLoweringResult:
        flat, retained = self._lowered_flat(positive_density, negative_density)
        return HistoricalMaskLoweringResult(
            masks=self._layer_masks(flat),
            positive_route_count=self.partition_route_counts["positive"],
            negative_route_count=self.partition_route_counts["negative"],
            retained_positive_count=retained["positive"],
            retained_negative_count=retained["negative"],
        )

    def dense_group(self) -> HistoricalMaskLoweringResult:
        return self.lower(1.0, 1.0)

    def build_candidate(
        self,
        _committed_masks: Mapping[str, np.ndarray],
        *,
        partition_id: str,
        positive_density: float,
        negative_density: float,
    ) -> dict[str, np.ndarray]:
        """Materialize the candidate as complete masks for callers that need arrays."""

        if partition_id not in _PARTITIONS:
            raise ValueError("partition_id must be positive or negative")
        lowered = self.lower(positive_density, negative_density)
        return compose_committed_masks(lowered.masks, self.previous_group_masks)

    __call__ = build_candidate

    def _kept_counts(self, positive_density: float, negative_density: float) -> dict[tuple[str, str], int]:
        densities = {"positive": float(positive_density), "negative": float(negative_density)}
        kept: dict[tuple[str, str], int] = {}
        for partition in _PARTITIONS:
            count = (
                0
                if self.partition_route_counts[partition] == 0
                else self.retained_non_floor(partition, densities[partition])
            )
            by_layer = np.bincount(
                self._layer_index(self._non_floor[partition][:count]),
                minlength=len(self.group_layers),
            )
            for index, layer in enumerate(self.group_layers):
                kept[(layer, partition)] = int(
                    self.floor_counts[(layer, partition)]
                ) + int(by_layer[index])
        return kept

    def initial_state(
        self,
        committed_masks: Mapping[str, Any],
        *,
        positive_density: float = 1.0,
        negative_density: float = 1.0,
    ) -> CommittedMaskState:
        """Construct the committed state at the supplied densities."""

        arrays = {
            str(key): _immutable_array(value, dtype=bool)
            for key, value in committed_masks.items()
        }
        return CommittedMaskState(
            arrays=arrays,
            layer_ids=tuple(arrays),
            group_layers=self.group_layers,
            densities={
                "positive": float(positive_density),
                "negative": float(negative_density),
            },
            kept_counts=self._kept_counts(positive_density, negative_density),
            _owned_arrays=True,
        )

    def build_candidate_view(
        self,
        state: CommittedMaskState,
        *,
        partition_id: str,
        positive_density: float,
        negative_density: float,
    ) -> CandidateView:
        if partition_id not in _PARTITIONS:
            raise ValueError("partition_id must be positive or negative")
        if state.group_layers != self.group_layers:
            raise ValueError("committed state belongs to another route group")
        densities = {"positive": float(positive_density), "negative": float(negative_density)}
        parent_densities = state.densities
        kept = dict(state.kept_counts)
        slices: list[_LayerSlice] = []
        plane = state._device_plane()
        for partition in _PARTITIONS:
            if self.partition_route_counts[partition] == 0:
                continue
            c_old = self.retained_non_floor(partition, parent_densities[partition])
            c_new = self.retained_non_floor(partition, densities[partition])
            if c_new > c_old:
                raise ValueError("a candidate view may only lower a partition's kept count")
            if c_new == c_old:
                continue
            if partition != partition_id:
                raise ValueError("a candidate view may only change the turn partition")
            # The slice of the resident pooled order, sorted on the device.
            # Ascending global indices group by layer, and each layer's
            # segment length is an exact rung-table difference.
            resident = self._device_orders.get(
                plane, (partition,),
                lambda plane, partition=partition: plane.upload_order(self._non_floor[partition]),
            )
            removed = plane.sorted_slice(resident, c_new, c_old)
            counts = self._layer_counts_at(partition, c_old) - self._layer_counts_at(partition, c_new)
            cursor = 0
            for index, layer in enumerate(self.group_layers):
                count = int(counts[index])
                if count == 0:
                    continue
                k_old = int(kept[(layer, partition)])
                k_new = k_old - count
                if k_new < self.floor_counts[(layer, partition)]:
                    raise AssertionError("a pooled proposal crossed a layer floor")
                kept[(layer, partition)] = k_new
                slices.append(
                    _LayerSlice(
                        layer=layer,
                        partition=partition,
                        k_old=k_old,
                        k_new=k_new,
                        removed=DeviceRemoved(
                            plane,
                            plane.segment(removed, cursor, cursor + count, int(self._offsets[index])),
                            count,
                        ),
                    )
                )
                cursor += count
            if cursor != c_old - c_new:
                raise AssertionError("pooled rung counts do not cover the slice")
        view = CandidateView(
            parent=state,
            partition_id=partition_id,
            densities=densities,
            kept_counts=kept,
            slices=tuple(slices),
        )
        return view


# ---------------------------------------------------------------------------
# One group run (the v0 branch of run_profiled_native_bce_precision_group)
# ---------------------------------------------------------------------------


def run_profiled_native_bce_precision_pooled_group(
    *,
    backend: Any,
    profile: Any,
    scores: Mapping[str, Any],
    partition_labels: Mapping[str, Any],
    query_id: str,
    group_id: str,
    route_group_kinds: Sequence[str],
    decision_rule: V0MarginalConnectionDecisionRule,
    loss_span_certificate: Any,
    route_semantics_by_layer: Mapping[str, str],
    previous_group_masks: Mapping[str, Any] | None = None,
    proposal_route_cap: int | None = None,
    proposal_grain: str = PROPOSAL_GRAIN_NOMINAL,
    frontier_prior: Mapping[str, float] | None = None,
    image_budget: float | None = None,
    naturalization_facts: BackendNaturalizationFacts | None = None,
    gpu_batch_size: int | None = None,
    reference_anchor_loss: float | None = None,
    reference_anchor_evidence_hash: str | None = None,
    reference_anchor_mask_hash: str | None = None,
    reference_anchor_panel_hash: str | None = None,
    maximum_accepted_decisions: int | None = None,
    progress_callback: Any | None = None,
    step_context_delta_sink: Any | None = None,
    accept_replication_gate: bool = False,
    oracle_sampling_law: str = IID_WITH_REPLACEMENT_SAMPLING_ID,
    oracle_acquisition_envelope_seconds: float | None = None,
    decision_law_id: str = "sequential_calibrated_bce_resolution_v5",
) -> tuple[NativeContinuousRun, PooledCandidateLowerer]:
    """Run one group through the unchanged frontier with the pooled lowerer.

    This is the modern-v0 branch of
    ``native_continuous.run_profiled_native_bce_precision_group`` with two
    substitutions: the lowerer (``PooledCandidateLowerer``) and the ladder
    (common to the group's kinds, optionally refined by the cap).  The policy
    fields are the v0 branch's, value for value.  Other decision rules are
    refused.
    """

    if not isinstance(decision_rule, V0MarginalConnectionDecisionRule):
        raise ValueError("the pooled group runner is defined for the modern-v0 rule only")
    profile.validate()
    nominal, minimum_density = executable_ladder_for_kinds(route_group_kinds, profile)
    lowerer = PooledCandidateLowerer(
        scores,
        partition_labels,
        nominal_schedule=nominal,
        floor_fraction=minimum_density,
        previous_group_masks=previous_group_masks,
        proposal_route_cap=proposal_route_cap,
        proposal_grain=proposal_grain,
    )
    dense_group = lowerer.dense_group()
    committed = compose_committed_masks(dense_group.masks, previous_group_masks)
    partition_counts = {
        name: int(lowerer.partition_route_counts[name])
        for name in _PARTITIONS
        if int(lowerer.partition_route_counts[name]) > 0
    }
    if not partition_counts:
        raise ValueError("BCE precision execution needs at least one route")
    active_partitions = tuple(partition_counts)
    facts = naturalization_facts or naturalization_facts_from_backend(
        backend,
        gpu_batch_size=gpu_batch_size,
    )
    facts.validate()
    decision_rule.validate()
    policy = BCEPrecisionPolicy(
        rule=decision_rule,
        image_budget=(None if image_budget is None else float(image_budget)),
        fold_count=int(profile.crossfit_fold_count),
        minimum_density=minimum_density,
        finite_population_correction=False,
        random_seed=int(profile.statistical_sampling_seed),
        maximum_planned_looks=None,
        precision_only_iid=True,
        runtime_deadline_monotonic=None,
        accept_replication_gate=bool(accept_replication_gate),
        oracle_sampling_law=str(oracle_sampling_law),
        decision_law_id=str(decision_law_id),
        oracle_acquisition_envelope_seconds=(
            None
            if oracle_acquisition_envelope_seconds is None
            else float(oracle_acquisition_envelope_seconds)
        ),
    )
    context = NaturalizationContext(
        class_count=int(facts.class_count),
        bank_depth_per_class=int(facts.bank_depth_per_class),
        gpu_batch_size=int(facts.gpu_batch_size),
        partition_route_counts=partition_counts,
        frontier_prior=(
            {
                key: value
                for key, value in dict(frontier_prior).items()
                if key in partition_counts
            }
            if frontier_prior
            else None
        ),
    )
    maximum_decisions = lowerer.decision_limit()
    result = trace_bce_precision_frontier(
        backend=backend,
        committed_masks=committed,
        build_candidate=lowerer,
        maximum_decisions=maximum_decisions,
        query_id=str(query_id),
        layer_id=str(group_id),
        total_layer_route_count=int(dense_group.total_route_count),
        policy=policy.resolve(context),
        class_count=int(facts.class_count),
        initial_turn=(
            str(profile.initial_partition)
            if str(profile.initial_partition) in active_partitions
            else active_partitions[0]
        ),
        active_partitions=active_partitions,
        executable_density_schedule=lowerer.schedule,
        maximum_accepted_decisions=maximum_accepted_decisions,
        progress_callback=progress_callback,
        step_context_delta_sink=step_context_delta_sink,
        loss_span_certificate=loss_span_certificate,
        route_semantics_by_layer=route_semantics_by_layer,
    )
    signature = NativeContinuousTraceSignature(
        profile_hash=str(profile.content_hash),
        query_id=str(query_id),
        layer_id=str(group_id),
        final_positive_density=float(result.positive_density),
        final_negative_density=float(result.negative_density),
        final_mask_hash=sampled_path_mask_hash(result.committed_masks),
        decision_count=len(result.decisions),
        images_consumed=int(result.images_consumed),
        stopping_reason=str(result.stopping_reason),
        delta_budget=float(result.delta_budget),
        delta_spent=float(result.delta_spent),
        algorithm_version=result.algorithm_version,
    )
    run = NativeContinuousRun(
        result=result,
        signature=signature,
        maturity="exploratory_bce_precision_default",
        analysis_status="eligible",
        reference_anchor_loss=(
            None if reference_anchor_loss is None else float(reference_anchor_loss)
        ),
        reference_anchor_evidence_hash=reference_anchor_evidence_hash,
        reference_anchor_mask_hash=reference_anchor_mask_hash,
        reference_anchor_panel_hash=reference_anchor_panel_hash,
    )
    return run, lowerer


# ---------------------------------------------------------------------------
# The panel (the modern-v0 subset of run_profiled_native_sampled_path_panel)
# ---------------------------------------------------------------------------


@dataclass
class PruneScopeWalk:
    """The runs of one prune-scope panel plus what its receipts need."""

    setting: PruneScopeSetting
    runs_by_query: dict[str, tuple[Any, ...]]
    lowerers_by_query: dict[str, tuple[PooledCandidateLowerer, ...]]
    eligible_layers: tuple[str, ...]
    ladders_by_kind: dict[str, tuple[tuple[float, ...], float]]
    labels_by_layer: dict[str, np.ndarray]
    profile_ladder: dict[str, float]
    scores_overridden: bool
    extra: dict[str, Any] = field(default_factory=dict)

    def traversal_complete(self, query_id: str) -> bool:
        """Scope-aware completeness.

        Layerwise: one run per eligible group, in execution order (the
        registered rule).  Global: exactly one run, labelled as the global
        group, whose lowerer covers every eligible layer.
        """

        runs = self.runs_by_query[str(query_id)]
        lowerers = self.lowerers_by_query[str(query_id)]
        if self.setting.scope == PRUNE_SCOPE_GLOBAL:
            return (
                len(runs) == 1
                and len(lowerers) == 1
                and str(runs[0].signature.layer_id) == GLOBAL_GROUP_ID
                and set(lowerers[0].group_layers) == set(self.eligible_layers)
                and len(lowerers[0].group_layers) == len(self.eligible_layers)
            )
        return (
            len(runs) == len(self.eligible_layers)
            and tuple(str(run.signature.layer_id) for run in runs) == self.eligible_layers
            and tuple(lowerer.group_layers for lowerer in lowerers)
            == tuple((layer,) for layer in self.eligible_layers)
        )

    def receipt(
        self,
        query_id: str,
        *,
        score_source: str,
        census: Mapping[str, Any] | None = None,
        route_group_registration: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """The ``prune_scope_walk`` block of the walk's ``workflow`` receipt."""

        query = str(query_id)
        tau = self.extra.get("sensitivity_tau_by_query", {}).get(query)
        runs = self.runs_by_query[query]
        lowerers = self.lowerers_by_query[query]
        groups: list[dict[str, Any]] = []
        retained_by_layer: dict[str, dict[str, int]] = {}
        total_by_layer: dict[str, dict[str, int]] = {}
        final_masks = runs[-1].result.committed_masks if runs else {}
        for run, lowerer in zip(runs, lowerers, strict=True):
            decisions = []
            matched = lowerer.step_table is not None
            current_labels = {partition: 1.0 for partition in _PARTITIONS}
            committed_positions = []
            def pooled_kept_at(labels: Mapping[str, float]) -> dict[str, dict[str, float | int | None]]:
                result = {}
                for partition in _PARTITIONS:
                    total = int(lowerer.partition_route_counts[partition])
                    floor = sum(int(lowerer.floor_counts[(layer, partition)]) for layer in lowerer.group_layers)
                    kept = floor + lowerer.retained_non_floor(partition, labels[partition])
                    result[partition] = {
                        "position_label": float(labels[partition]),
                        "kept_route_count": int(kept),
                        "route_count": total,
                        "kept_density": float(kept / total) if total else None,
                    }
                return result
            if matched:
                committed_positions.append({"proposal_index": None, "partition": pooled_kept_at(current_labels)})
            for index, decision in enumerate(run.result.decisions):
                if matched:
                    # A zero-removal rung advances its cursor without a decision.
                    current_labels[str(decision.partition_id)] = float(decision.previous_density)
                by_layer = lowerer.removed_by_layer(
                    str(decision.partition_id),
                    float(decision.previous_density),
                    float(decision.proposed_density),
                )
                if sum(by_layer.values()) != int(decision.removed_route_count):
                    raise AssertionError(
                        "a decision's removed count differs from its ladder step"
                    )
                decisions.append(
                    {
                        "proposal_index": int(index),
                        "partition_id": str(decision.partition_id),
                        "action": str(decision.action),
                        "reason": str(decision.reason),
                        "previous_density": float(decision.previous_density),
                        "proposed_density": float(decision.proposed_density),
                        "removed_route_count": int(decision.removed_route_count),
                        "removed_by_layer": by_layer,
                    }
                )
                if matched and str(decision.action) == "accept":
                    current_labels[str(decision.partition_id)] = float(decision.proposed_density)
                    committed_positions.append({"proposal_index": int(index), "partition": pooled_kept_at(current_labels)})
            groups.append(
                {
                    "group_id": str(run.signature.layer_id),
                    "layers": list(lowerer.group_layers),
                    "executed_rung_count": len(lowerer.schedule),
                    "grain_rule": lowerer.grain_rule,
                    "partition_route_counts": dict(lowerer.partition_route_counts),
                    "layer_partition_route_counts": {
                        layer: {
                            partition: int(
                                lowerer.layer_partition_route_counts[(layer, partition)]
                            )
                            for partition in _PARTITIONS
                        }
                        for layer in lowerer.group_layers
                    },
                    "floor_route_counts": {
                        layer: {
                            partition: int(lowerer.floor_counts[(layer, partition)])
                            for partition in _PARTITIONS
                        }
                        for layer in lowerer.group_layers
                    },
                    "maximum_ladder_step_removed_routes": int(
                        lowerer.maximum_proposal_removed_routes()
                    ),
                    "zero_removal_steps_by_partition": lowerer.zero_removal_steps(),
                    "maximum_decisions": int(lowerer.decision_limit()),
                    "decision_count": len(decisions),
                    "accepted_count": sum(
                        1 for item in decisions if item["action"] == "accept"
                    ),
                    "stopping_reason": str(run.result.stopping_reason),
                    "final_positive_density": float(run.result.positive_density),
                    "final_negative_density": float(run.result.negative_density),
                    "decisions": decisions,
                }
            )
            if lowerer.step_table is not None:
                groups[-1]["step_table"] = lowerer.step_table
                groups[-1]["density_field_semantics"] = (
                    "The previous_density, proposed_density, final_positive_density, "
                    "final_negative_density, group_final_densities, and signature final "
                    "densities are ladder position labels; they are not pooled kept densities."
                )
                final_labels = {
                    "positive": float(run.result.positive_density),
                    "negative": float(run.result.negative_density),
                }
                groups[-1]["committed_pooled_kept_density_by_partition"] = committed_positions
                groups[-1]["final_pooled_kept_density_by_partition"] = pooled_kept_at(final_labels)
        for layer in self.eligible_layers:
            labels = np.asarray(self.labels_by_layer[layer]).reshape(-1)
            flat = (
                np.asarray(final_masks[layer], dtype=bool).reshape(-1)
                if layer in final_masks
                else np.ones(labels.size, dtype=bool)
            )
            retained_by_layer[layer] = {
                partition: int(
                    np.count_nonzero(flat & (labels == _PARTITION_CODE[partition]))
                )
                for partition in _PARTITIONS
            }
            total_by_layer[layer] = {
                partition: int(np.count_nonzero(labels == _PARTITION_CODE[partition]))
                for partition in _PARTITIONS
            }
        realized = {}
        for layer in self.eligible_layers:
            kept = sum(retained_by_layer[layer].values())
            total = sum(total_by_layer[layer].values())
            realized[layer] = {
                "retained_route_count": int(kept),
                "route_count": int(total),
                "retained_fraction": float(kept / total) if total else None,
                "retained_by_partition": dict(retained_by_layer[layer]),
                "route_count_by_partition": dict(total_by_layer[layer]),
            }
        kept_total = sum(item["retained_route_count"] for item in realized.values())
        route_total = sum(item["route_count"] for item in realized.values())
        grain_rule = (
            GRAIN_RULE_LAYERWISE_MATCHED
            if self.setting.proposal_grain == PROPOSAL_GRAIN_LAYERWISE_MATCHED
            else (
                GRAIN_RULE_NOMINAL
                if self.setting.proposal_route_cap is None
                else GRAIN_RULE_CAPPED
            )
        )
        receipt = {
            "schema": PRUNE_SCOPE_WALK_SCHEMA,
            "setting": self.setting.to_dict(),
            "environment": {
                PRUNE_SCOPE_ENV: self.setting.scope,
                PROPOSAL_ROUTE_CAP_ENV: (
                    "none"
                    if self.setting.proposal_route_cap is None
                    else str(self.setting.proposal_route_cap)
                ),
            },
            "keep_rule": KEEP_RULE,
            "tie_rule": TIE_RULE,
            "grain_rule": grain_rule,
            "grain": {
                "rule": grain_rule,
                "definition": CAP_SEMANTICS[grain_rule],
                "proposal_route_cap": self.setting.proposal_route_cap,
                # Note 08 defines the cap as floor(gamma / tau); this is the
                # gamma a cap corresponds to at the walk's tau.
                "cap_times_tau": (
                    None
                    if self.setting.proposal_route_cap is None or tau is None
                    else float(self.setting.proposal_route_cap) * float(tau)
                ),
                "note08_reference": {
                    "gamma": NOTE08_REFERENCE_GAMMA,
                    "tau": REGISTERED_SENSITIVITY_TAU,
                    "proposal_route_cap": NOTE08_REFERENCE_CAP,
                },
                "matched_comparator": (
                    "the diagnostic layerwise walk capped at the same "
                    "proposal_route_cap by the same rule"
                    if self.setting.proposal_route_cap is not None
                    else "the diagnostic layerwise walk on the same nominal ladder"
                ),
            },
            "sensitivity_tau": None if tau is None else float(tau),
            "sensitivity_tau_rule": tau_pin_rule(),
            "route_group_registration": dict(route_group_registration or {}),
            "price": "bce_increment_over_loss_span_per_actually_removed_connection",
            "ladder": {
                **dict(self.profile_ladder),
                "by_route_group_kind": {
                    kind: {
                        "nominal_rung_count": len(schedule),
                        "nominal_schedule": [float(value) for value in schedule],
                        "floor_fraction": float(floor),
                    }
                    for kind, (schedule, floor) in self.ladders_by_kind.items()
                },
                "floor_rule": "ceil(floor_fraction * n_layer_partition) per layer and partition",
            },
            "score_source": str(score_source),
            "scores_pooled_without_rescaling": True,
            "census": dict(census or {}),
            "eligible_layers": list(self.eligible_layers),
            "groups": groups,
            "realized_density_by_layer": realized,
            "retained_route_count": int(kept_total),
            "eligible_route_count": int(route_total),
            "retained_fraction": float(kept_total / route_total) if route_total else None,
            "traversal_complete": bool(self.traversal_complete(query)),
            "traversal_rule": (
                "one_run_covering_every_eligible_layer"
                if self.setting.scope == PRUNE_SCOPE_GLOBAL
                else "one_run_per_eligible_layer_group"
            ),
            # The order certificate that governs the walk's proposals.  A
            # global walk's cross-layer order is reported, never used, in
            # ``cross_layer_order_diagnostic`` (added by the pilot after the
            # traversal clock stops).
            "order_certificate": (
                "layerwise_census_within_layer_partition_only"
                if self.setting.scope == PRUNE_SCOPE_GLOBAL
                else "layerwise_census"
            ),
            "cross_layer_order_certificate_governs_decisions": False,
            "optimality_certificate": "none",
        }
        if grain_rule == GRAIN_RULE_LAYERWISE_MATCHED:
            if len(lowerers) != 1 or lowerers[0].step_table is None:
                raise AssertionError("a matched global receipt needs its realized step table")
            receipt["environment"][PROPOSAL_GRAIN_ENV] = PROPOSAL_GRAIN_LAYERWISE_MATCHED
            receipt["grain"]["matched_comparator"] = (
                "the registered layerwise arm on the same nominal ladder, "
                "with the same partition step-size multiset"
            )
            receipt["grain"]["step_table"] = lowerers[0].step_table
            receipt["grain"]["density_field_semantics"] = groups[0]["density_field_semantics"]
            receipt["committed_pooled_kept_density_by_partition"] = groups[0]["committed_pooled_kept_density_by_partition"]
            receipt["final_pooled_kept_density_by_partition"] = groups[0]["final_pooled_kept_density_by_partition"]
        return receipt


def _eligible_groups_in_route_axis_order(
    groups: Sequence[Any], measured_blocks: Mapping[str, Any]
) -> tuple[Any, ...]:
    ordered = sorted(groups, key=lambda group: int(measured_blocks[group.layer_id].offset))
    offsets = [int(measured_blocks[group.layer_id].offset) for group in ordered]
    if len(set(offsets)) != len(offsets):
        raise ValueError("route blocks share an offset; the pooled tie rule is undefined")
    return tuple(ordered)


def run_prune_scope_panel(
    *,
    setting: PruneScopeSetting,
    backend: Any,
    profile: Any,
    measurement: Any,
    query_ids: Sequence[str],
    route_groups: Sequence[Any],
    partition_labels_by_layer: Mapping[str, Any],
    dense_anchor_loss_by_query: Mapping[str, float],
    bce_decision_rule_by_query: Mapping[str, Any],
    loss_span_certificate_by_query: Mapping[str, Any],
    route_activity_scores_by_layer: Mapping[str, Any] | None = None,
    promotion_receipt: Any | None = None,
    claim_bearing: bool = False,
    layer_execution_order: str = "forward",
    algorithm: str = BCE_PRECISION_ALGORITHM,
    image_budget_per_group: float | None = None,
    naturalization_facts: Any = None,
    maximum_commits: int | None = None,
    accept_replication_gate: bool = False,
    progress_callback: Any | None = None,
    step_context_delta_sink: Any | None = None,
    group_transition_callback: Any | None = None,
    oracle_sampling_law: str = IID_WITH_REPLACEMENT_SAMPLING_ID,
    oracle_acquisition_envelope_seconds: float | None = None,
    decision_law_id: str = "sequential_calibrated_bce_resolution_v5",
) -> PruneScopeWalk:
    """Run the modern-v0 walk under a non-default prune-scope setting.

    Its inputs, validation and per-group protocol (dense reference, first-group
    anchor check, carried committed masks, frontier-prior history,
    campaign-wide commit limit, group-transition callback) are those of
    ``run_profiled_native_sampled_path_panel`` on the modern-v0 path.  What
    differs is only the group structure (one pooled group for ``global``) and
    the lowerer and ladder of each group.  The default setting is refused here:
    it runs the registered panel.
    """

    del promotion_receipt  # accepted for call parity; never claim-bearing here
    if not isinstance(setting, PruneScopeSetting):
        raise TypeError("setting must be a PruneScopeSetting")
    if setting.is_default:
        raise ValueError(
            "the default setting (layerwise, no cap) runs the registered "
            "layerwise panel, not this walk"
        )
    if claim_bearing:
        raise ValueError("prune-scope walks are exploratory and never claim-bearing")
    if str(algorithm) != BCE_PRECISION_ALGORITHM:
        raise ValueError(
            f"prune-scope walks run the BCE-precision v0 engine only, got {algorithm!r}"
        )
    if maximum_commits is not None and int(maximum_commits) <= 0:
        raise ValueError("sampled-path maximum_commits must be positive")
    order = str(layer_execution_order)
    if order not in {"forward", "reverse"}:
        raise ValueError(
            f"layer_execution_order must be 'forward' or 'reverse', got {order!r}"
        )
    if setting.scope == PRUNE_SCOPE_GLOBAL and order != "forward":
        raise ValueError("a global walk admits only the layer execution order 'forward'")
    profile.validate()
    queries = tuple(str(value) for value in query_ids)
    if not queries or len(set(queries)) != len(queries):
        raise ValueError("sampled-path panel requires unique query IDs")
    measurement_queries = tuple(str(value) for value in measurement.query_ids)
    if not measurement_queries or len(set(measurement_queries)) != len(
        measurement_queries
    ):
        raise ValueError("sampled-path measurement requires unique query IDs")
    missing = tuple(query for query in queries if query not in measurement_queries)
    if missing:
        raise ValueError(
            "sampled-path panel query axis is outside its measurement: " + ",".join(missing)
        )
    positions = tuple(measurement_queries.index(query) for query in queries)
    if tuple(sorted(positions)) != positions:
        raise ValueError("sampled-path panel queries must preserve the measurement query order")
    groups = tuple(route_groups)
    if not groups:
        raise ValueError("sampled-path panel requires route groups")
    for group in groups:
        group.validate()
    group_layers = tuple(group.layer_id for group in groups)
    if len(set(group_layers)) != len(group_layers):
        raise ValueError("sampled-path panel route groups repeat stable layer IDs")
    measured_blocks = {str(block.layer_id): block for block in measurement.route_blocks}
    if not set(group_layers).issubset(measured_blocks):
        raise ValueError("sampled-path panel groups do not cover the prunable route axis")
    for group in groups:
        if tuple(measured_blocks[group.layer_id].shape) != tuple(group.shape):
            raise ValueError(f"sampled-path group shape differs for {group.layer_id!r}")
    anchors = {str(key): float(value) for key, value in dense_anchor_loss_by_query.items()}
    if set(anchors) != set(queries) or any(not math.isfinite(value) for value in anchors.values()):
        raise ValueError(
            "sampled-path dense anchors must be finite and cover the exact query axis"
        )
    rules = {str(key): value for key, value in dict(bce_decision_rule_by_query).items()}
    if set(rules) != set(queries) or any(
        not isinstance(rule, V0MarginalConnectionDecisionRule) for rule in rules.values()
    ):
        raise ValueError(
            "prune-scope walks require a modern-v0 decision rule for every query"
        )
    certificates = {
        str(key): value for key, value in dict(loss_span_certificate_by_query).items()
    }
    if set(certificates) != set(queries):
        raise ValueError(
            "modern v0 decision rules require a loss-span certificate for every query"
        )
    ordered_groups = groups if order == "forward" else tuple(reversed(groups))
    execution_groups = tuple(
        group for group in ordered_groups if group.route_kind in _ELIGIBLE_ROUTE_KINDS
    )
    if not execution_groups:
        raise ValueError("modern v0 route axis contains no eligible circuit connections")
    route_semantics_by_layer = {
        group.layer_id: (
            "fc_scalar_weight"
            if group.route_kind == "linear_weight"
            else (
                "conv_channel_pair"
                if group.route_kind == "conv_channel_pair"
                else (
                    "conv_kernel_tap"
                    if group.route_kind == "conv_kernel_weight"
                    else "protected_nonconnection_route"
                )
            )
        )
        for group in groups
    }
    if set(str(key) for key in partition_labels_by_layer) != set(group_layers):
        raise ValueError(
            "sampled-path structural partition labels must cover the exact route axis"
        )
    scores_overridden = route_activity_scores_by_layer is not None
    if scores_overridden and set(map(str, route_activity_scores_by_layer)) != set(
        group_layers
    ):
        raise ValueError("sampled-path activity-score override must cover the route groups")
    query_index = {query: index for index, query in enumerate(measurement_queries)}
    expected_head_rows = len(queries) if scores_overridden else len(measurement_queries)
    head_row_by_query = {
        query: (queries.index(query) if scores_overridden else query_index[query])
        for query in queries
    }
    score_heads: dict[str, np.ndarray] = {}
    structural_labels: dict[str, np.ndarray] = {}
    for group in groups:
        absolute = np.asarray(
            (
                route_activity_scores_by_layer[group.layer_id]
                if scores_overridden
                else measurement.source_image_aggregate_block(
                    group.layer_id, score_head="route_abs_scores"
                )
            ),
            dtype=np.float64,
        )
        expected = (expected_head_rows, int(np.prod(group.shape, dtype=np.int64)))
        if absolute.shape != expected:
            raise ValueError(
                "sampled-path source heads have the wrong shape for "
                f"{group.layer_id!r}: observed={absolute.shape!r} expected={expected!r}"
            )
        if not np.isfinite(absolute).all() or np.any(absolute < 0.0):
            raise ValueError("sampled-path source score heads must be finite")
        labels = np.asarray(partition_labels_by_layer[group.layer_id])
        if labels.shape != group.shape:
            raise ValueError(
                f"sampled-path partition labels have the wrong shape for {group.layer_id!r}"
            )
        if labels.dtype.kind not in {"i", "u", "b"} or not np.isin(labels, (0, 1)).all():
            raise ValueError("sampled-path partition labels must be integer zero/one")
        score_heads[group.layer_id] = absolute
        structural_labels[group.layer_id] = labels.astype(np.int8, copy=False)
    if setting.scope == PRUNE_SCOPE_GLOBAL:
        pooled_groups = _eligible_groups_in_route_axis_order(execution_groups, measured_blocks)
        eligible_layers = tuple(group.layer_id for group in pooled_groups)
    else:
        pooled_groups = ()
        eligible_layers = tuple(group.layer_id for group in execution_groups)
    # Each kind's registered ladder; a global walk needs them to coincide.
    ladders_by_kind = {
        kind: executable_ladder_for_kinds([kind], profile)
        for kind in dict.fromkeys(group.route_group_kind for group in execution_groups)
    }
    if setting.scope == PRUNE_SCOPE_GLOBAL:
        executable_ladder_for_kinds(list(ladders_by_kind), profile)
    dense_route_axis = {group.layer_id: np.ones(group.shape, dtype=bool) for group in groups}
    history: dict[str, list[float]] = {}
    runs_by_query: dict[str, tuple[Any, ...]] = {}
    lowerers_by_query: dict[str, tuple[PooledCandidateLowerer, ...]] = {}

    def dense_reference(query: str, layer_id: str, masks: Mapping[str, np.ndarray]) -> Any:
        request = SampledComparisonRequest(
            query_id=query,
            layer_id=layer_id,
            partition_id="dense_reference",
            previous_density=1.0,
            proposed_density=1.0,
            stage_index=0,
            stage_per_class=int(profile.sample_stages_per_class[0]),
            proposal_index=0,
            random_seed=int(profile.statistical_sampling_seed),
            fold_count=int(profile.crossfit_fold_count),
        )
        reference = backend.evaluate_dense_reference(masks, request)
        reference.validate(fold_count=int(profile.crossfit_fold_count))
        if reference.query_id != query:
            raise ValueError("sampled-path group reference query identity differs")
        if reference.mask_hash != sampled_path_mask_hash(masks):
            raise ValueError("sampled-path group reference mask identity differs")
        return reference

    for query in queries:
        index = head_row_by_query[query]
        prior_masks = {layer: mask.copy() for layer, mask in dense_route_axis.items()}
        query_runs: list[Any] = []
        query_lowerers: list[PooledCandidateLowerer] = []
        accepted_commit_count = 0
        plan: tuple[tuple[str, tuple[Any, ...]], ...] = (
            ((GLOBAL_GROUP_ID, pooled_groups),)
            if setting.scope == PRUNE_SCOPE_GLOBAL
            else tuple((group.layer_id, (group,)) for group in execution_groups)
        )
        for group_index, (group_id, members) in enumerate(plan):
            if maximum_commits is not None and accepted_commit_count >= int(maximum_commits):
                break
            if group_transition_callback is not None:
                group_transition_callback(
                    query_id=str(query), group_index=int(group_index), layer_id=str(group_id)
                )
            reference = dense_reference(query, group_id, prior_masks)
            group_anchor_loss = reference.canonical_loss
            if group_index == 0 and not math.isclose(
                group_anchor_loss,
                anchors[query],
                rel_tol=INITIAL_DENSE_ANCHOR_NUMERICAL_TOLERANCE,
                abs_tol=INITIAL_DENSE_ANCHOR_NUMERICAL_TOLERANCE,
            ):
                raise ValueError(
                    "native first-group reference differs from the sealed globally "
                    f"dense anchor for query {query!r}: "
                    f"observed={group_anchor_loss:.17g} expected={anchors[query]:.17g}"
                )
            run, lowerer = run_profiled_native_bce_precision_pooled_group(
                backend=backend,
                profile=profile,
                scores={
                    group.layer_id: score_heads[group.layer_id][index].reshape(
                        group.shape, order="C"
                    )
                    for group in members
                },
                partition_labels={
                    group.layer_id: structural_labels[group.layer_id] for group in members
                },
                query_id=query,
                group_id=group_id,
                route_group_kinds=[group.route_group_kind for group in members],
                decision_rule=rules[query],
                loss_span_certificate=certificates[query],
                route_semantics_by_layer=route_semantics_by_layer,
                previous_group_masks=prior_masks,
                proposal_route_cap=setting.proposal_route_cap,
                proposal_grain=setting.proposal_grain,
                frontier_prior=frontier_prior_from_history(history),
                image_budget=image_budget_per_group,
                naturalization_facts=naturalization_facts,
                reference_anchor_loss=group_anchor_loss,
                reference_anchor_evidence_hash=reference.content_hash,
                reference_anchor_mask_hash=reference.mask_hash,
                reference_anchor_panel_hash=reference.panel.panel_hash,
                maximum_accepted_decisions=(
                    None
                    if maximum_commits is None
                    else int(maximum_commits) - accepted_commit_count
                ),
                progress_callback=progress_callback,
                step_context_delta_sink=step_context_delta_sink,
                accept_replication_gate=accept_replication_gate,
                oracle_sampling_law=str(oracle_sampling_law),
                decision_law_id=str(decision_law_id),
                oracle_acquisition_envelope_seconds=oracle_acquisition_envelope_seconds,
            )
            record_frontier_history(history, run.result)
            prior_masks = {
                str(layer): np.asarray(mask, dtype=bool).copy()
                for layer, mask in run.result.committed_masks.items()
            }
            query_runs.append(run)
            query_lowerers.append(lowerer)
            accepted_commit_count += sum(
                decision.action == "accept" for decision in run.result.decisions
            )
        runs_by_query[query] = tuple(query_runs)
        lowerers_by_query[query] = tuple(query_lowerers)
    return PruneScopeWalk(
        setting=setting,
        runs_by_query=runs_by_query,
        lowerers_by_query=lowerers_by_query,
        eligible_layers=eligible_layers,
        ladders_by_kind={
            kind: (tuple(float(value) for value in schedule), float(floor))
            for kind, (schedule, floor) in ladders_by_kind.items()
        },
        labels_by_layer={layer: structural_labels[layer] for layer in eligible_layers},
        profile_ladder={
            "conv_decay": float(profile.target_preservation_conv_decay),
            "conv_target_final_density": float(
                profile.target_preservation_conv_target_final_density
            ),
            "linear_decay": float(profile.target_preservation_linear_decay),
            "linear_target_final_density": float(
                profile.target_preservation_linear_target_final_density
            ),
        },
        scores_overridden=scores_overridden,
        extra={
            # The tau each query's decision rule actually priced with; the
            # pilot pins it at entry (``validate_prune_scope_admission``).
            "sensitivity_tau_by_query": {
                query: float(rules[query].sensitivity_tau) for query in queries
            },
        },
    )


# ---------------------------------------------------------------------------
# Reporting-only cross-layer order diagnostic
# ---------------------------------------------------------------------------


def _committed_cuts(run: Any) -> dict[str, list[float]]:
    """Every committed ladder position of each partition, in walk order.

    With a cap these include the intermediate (non-nominal) positions.
    """

    committed: dict[str, list[float]] = {partition: [] for partition in _PARTITIONS}
    for decision in run.result.decisions:
        if str(decision.action) == "accept":
            committed[str(decision.partition_id)].append(float(decision.proposed_density))
    return committed


class _DeviceField:
    """A global walk's pooled census field on one device, each array uploaded once.

    The cross-layer diagnostic's three parts share one instance, so the walk's
    scores, labels and non-floor queues and each stored census field cross to
    the device once (a tensor already on the device is used as it is).
    Nothing is uploaded until a part asks for it.
    """

    def __init__(self, walk: "PruneScopeWalk", query_id: str, device: Any = None) -> None:
        from .cross_layer_order_device import resolve_device

        self.walk = walk
        self.query_id = str(query_id)
        self.device = resolve_device(device)
        self._arrays: dict[Any, Any] = {}

    @classmethod
    def of(cls, walk: "PruneScopeWalk", query_id: str, device: Any) -> "_DeviceField":
        """``device`` itself when it is this walk's field, else a new field on ``device``."""

        if isinstance(device, cls) and device.walk is walk and device.query_id == str(query_id):
            return device
        return cls(walk, query_id, device)

    @property
    def lowerer(self) -> "PooledCandidateLowerer":
        (lowerer,) = self.walk.lowerers_by_query[self.query_id]
        return lowerer

    def _vector(self, values: Any, dtype: torch.dtype) -> torch.Tensor:
        from .cross_layer_order_device import as_device_vector

        return as_device_vector(values, dtype=dtype, device=self.device)

    def _once(self, key: Any, build: Callable[[], Any]) -> Any:
        if key not in self._arrays:
            self._arrays[key] = build()
        return self._arrays[key]

    def scores(self) -> torch.Tensor:
        """The walk's ranked scores on the pooled axis (``lowerer._flat_scores``)."""

        return self._once("scores", lambda: self._vector(self.lowerer._flat_scores, torch.float64))

    def labels(self) -> torch.Tensor:
        """The walk's partition labels on the pooled axis, int8."""

        return self._once(
            "labels",
            lambda: torch.cat(
                [
                    self._vector(self.walk.labels_by_layer[layer], torch.int8)
                    for layer in self.lowerer.group_layers
                ]
            ),
        )

    def queue(self, partition: str) -> torch.Tensor:
        """A partition's non-floor pooled queue (flat indices, descending score)."""

        return self._once(
            ("queue", partition),
            lambda: self._vector(self.lowerer._non_floor[partition], torch.int64),
        )

    def layer_index(self, flat: torch.Tensor) -> torch.Tensor:
        """``lowerer._layer_index`` on the device."""

        offsets = self._once("offsets", lambda: self._vector(self.lowerer._offsets, torch.int64))
        return torch.searchsorted(offsets, flat, right=True) - 1

    def pooled(
        self, by_layer: Mapping[str, Any] | None, *, name: str
    ) -> tuple[torch.Tensor | None, str | None]:
        """One stored per-layer field, flat on the pooled axis, or a reason."""

        if by_layer is None:
            return None, f"no_{name}"
        # The parts read the same mapping; holding it keeps its id its own.
        key = ("field", id(by_layer))
        if key not in self._arrays:
            self._arrays[key] = (by_layer, self._upload(by_layer))
        flat, problem = self._arrays[key][1]
        if problem is None:
            return flat, None
        kind, layer = problem
        if kind == "missing":
            return None, f"no_{name}_for_{layer}"
        if kind == "shape":
            return None, f"{name}_shape_{layer}"
        return None, f"nonfinite_{name}"

    def _upload(self, by_layer: Mapping[str, Any]) -> tuple[torch.Tensor | None, tuple[str, Any] | None]:
        lowerer = self.lowerer
        pieces = []
        for layer in lowerer.group_layers:
            if layer not in by_layer:
                return None, ("missing", layer)
            values = self._vector(by_layer[layer], torch.float64)
            if values.numel() != lowerer._sizes[layer]:
                return None, ("shape", layer)
            pieces.append(values)
        flat = torch.cat(pieces)
        if not bool(torch.isfinite(flat).all()):
            return None, ("nonfinite", None)
        return flat, None


def _any_negative(*fields: torch.Tensor) -> list[bool]:
    """Whether each device field holds a negative value, in one read."""

    return [bool(value) for value in torch.stack([(field < 0.0).any() for field in fields]).tolist()]


def cross_layer_order_certificate(
    walk: PruneScopeWalk,
    query_id: str,
    *,
    score_standard_error_by_layer: Mapping[str, Any] | None,
    route_damage_by_layer: Mapping[str, Any] | None,
    route_damage_standard_error_by_layer: Mapping[str, Any] | None,
    critical_value: float | None,
    maximum_mask_disagreement_activity_fraction: float,
    device: Any = None,
) -> dict[str, Any]:
    """The cross-layer order certificate, reporting only.

    Computes ``score_order.certify_prune_order_stability`` with
    ``scope="cross_layer"`` (scopes ``global:partition:0`` and
    ``global:partition:1``: every eligible route of a partition, all layers
    pooled) on the stored census field of the walk: the ranked activity
    (the walk's own scores), its standard error, and the global-head mean and
    its standard error as the route damage, exactly the arguments the
    layerwise census passes for these quantities.  The audited retained
    fractions are the realized pooled cuts: every committed ladder position
    of each partition, capped intermediate positions included, translated to
    the within-partition fraction whose top-k plus the floors is the committed
    kept set (``PooledCandidateLowerer.pooled_prefix_length``).  A cut with no
    pooled boundary (floors only, a within-layer cut) is listed and not
    audited.

    The stored field is ONE census look, so the certificate's look-to-look
    mask-disagreement measure has no previous look (its receipt says
    ``previous_look_available: false`` and carries the function's
    no-previous-look value 1.0, which is not a measurement).  Its within-look
    content is the consequence price per scope: the band of routes whose
    membership at an audited cut the intervals cannot decide, and the damage
    swing over it.  The diagnostic therefore files this block under
    ``consequence_price``.  It is not the unresolved activity
    fraction; that is ``boundary_uncertain_activity``.  No decision reads
    this block.

    The call runs on the device through its port
    ``cross_layer_order_device.certify_cross_layer_order_stability``, which
    returns the same certificate (sums differ only by summation order).
    ``device`` is a torch device or the diagnostic's shared ``_DeviceField``.
    """

    from .cross_layer_order_device import certify_cross_layer_order_stability

    started = time.perf_counter()
    base: dict[str, Any] = {
        "function": "mechanistic_systems.circuits.irg.score_order.certify_prune_order_stability",
        "scope": "cross_layer",
        "quantity": "consequence_price",
        "is_certificate": True,
        "decision_role": "reporting_only",
        "governs_decisions": False,
    }
    if walk.setting.scope != PRUNE_SCOPE_GLOBAL:
        return {**base, "available": False, "reason": "layerwise_scope"}
    if not walk.scores_overridden:
        return {**base, "available": False, "reason": "scores_not_from_the_census_field"}
    if critical_value is None or not math.isfinite(float(critical_value)) or float(critical_value) <= 0.0:
        return {**base, "available": False, "reason": "no_census_order_critical_value"}
    (run,) = walk.runs_by_query[str(query_id)]
    (lowerer,) = walk.lowerers_by_query[str(query_id)]
    field = _DeviceField.of(walk, query_id, device)
    fields: dict[str, torch.Tensor] = {}
    for name, by_layer in (
        ("loss_activity_standard_error", score_standard_error_by_layer),
        ("loss_mean", route_damage_by_layer),
        ("loss_mean_standard_error", route_damage_standard_error_by_layer),
    ):
        flat, reason = field.pooled(by_layer, name=name)
        if flat is None:
            return {**base, "available": False, "reason": reason}
        fields[name] = flat
    if any(
        _any_negative(
            fields["loss_activity_standard_error"], fields["loss_mean_standard_error"]
        )
    ):
        return {**base, "available": False, "reason": "negative_standard_error"}
    nominal = {float(value) for value in lowerer.nominal_schedule}
    committed = _committed_cuts(run)
    audited_by_scope: dict[str, tuple[float, ...]] = {}
    translation: list[dict[str, Any]] = []
    for partition in _PARTITIONS:
        code = _PARTITION_CODE[partition]
        size = int(lowerer.partition_route_counts[partition])
        fractions: list[float] = []
        for density in committed[partition]:
            prefix = lowerer.pooled_prefix_length(partition, density)
            row: dict[str, Any] = {
                "scope_id": f"global:partition:{code}",
                "partition_id": partition,
                "ladder_position": float(density),
                "nominal_rung": bool(float(density) in nominal),
                "retained_non_floor_count": int(
                    lowerer.retained_non_floor(partition, density)
                ),
                "pooled_prefix_length": int(prefix),
                "partition_route_count": size,
            }
            if 1 <= prefix <= size - 1:
                fraction = _exact_ceil_fraction(prefix, size)
                fractions.append(fraction)
                row.update({"audited": True, "audited_retained_fraction": fraction})
            else:
                row.update(
                    {
                        "audited": False,
                        "reason": (
                            "floors_only_within_layer_cut"
                            if prefix == 0
                            else "no_pooled_boundary"
                        ),
                    }
                )
            translation.append(row)
        if fractions:
            audited_by_scope[f"global:partition:{code}"] = tuple(fractions)
    translation_rule = (
        "committed position -> k = shortest pooled score prefix whose union "
        "with the per-layer floors is the committed kept set -> fraction f "
        "with ceil(f * partition_route_count) == k"
    )
    if not audited_by_scope:
        return {
            **base,
            "available": False,
            "reason": "no_committed_pooled_cut",
            "audited_cut_translation": translation,
            "translation_rule": translation_rule,
        }
    axis_hash = "prune-scope-pooled-axis-" + hashlib.sha256(
        json.dumps(
            [[layer, int(lowerer._sizes[layer])] for layer in lowerer.group_layers]
        ).encode("utf-8")
    ).hexdigest()
    # The call: scope cross_layer, no previous look, global_route_field
    # normalization, the three census fields, the census critical value.
    certificate = certify_cross_layer_order_stability(
        current_global_activity=field.scores(),
        current_partition_labels=field.labels(),
        score_standard_error=fields["loss_activity_standard_error"],
        route_damage=fields["loss_mean"],
        route_damage_standard_error=fields["loss_mean_standard_error"],
        critical_value=float(critical_value),
        audited_retained_fractions_by_scope=audited_by_scope,
        maximum_mask_disagreement_activity_fraction=float(
            maximum_mask_disagreement_activity_fraction
        ),
        route_axis_hash=axis_hash,
    )
    return {
        **base,
        "available": True,
        "field": {
            "current_global_activity": "census loss_activity (the walk's ranked scores)",
            "score_standard_error": "census loss_activity_standard_error",
            "route_damage": "census loss-head mean",
            "route_damage_standard_error": "census loss_mean_standard_error",
            "looks": 1,
        },
        "critical_value": float(critical_value),
        "critical_value_source": "census order_normal_critical_value",
        "audited_cut_count": int(sum(len(value) for value in audited_by_scope.values())),
        "audited_cut_translation": translation,
        "translation_rule": translation_rule,
        "reading": (
            "one stored look: previous_look_available is false, so the "
            "mask-disagreement fields hold the function's no-previous-look "
            "value (1.0) and are not a measurement; the consequence-price "
            "fields are the within-look result at the audited pooled cuts. "
            "The unresolved (boundary-uncertain) activity fraction at each "
            "committed cut is boundary_uncertain_activity, not this block. "
            "Reporting only; the walk's decisions are governed by the "
            "layerwise census, which certifies order only within each layer "
            "and partition."
        ),
        "result": certificate.to_receipt(),
        "content_hash": certificate.content_hash,
        "seconds": float(time.perf_counter() - started),
    }


def boundary_uncertain_activity(
    walk: PruneScopeWalk,
    query_id: str,
    *,
    standard_error_by_layer: Mapping[str, Any] | None,
    critical_value: float | None,
    device: Any = None,
) -> dict[str, Any]:
    """The boundary-uncertain activity fraction at every committed pooled cut.

    Reporting only; no decision reads it and it applies no threshold.  This
    is the "unresolved activity fraction" for
    a walk whose stored field is one census look.

    The calculation is ``score_order.certify_prune_order_uncertainty``'s at
    an audited count (``_scope_receipt``: interval overlap with strict
    comparisons, ambiguous routes weighted by activity), adapted to the
    walk's keep rule.  At a committed position partition ``p`` keeps its
    pinned floor routes plus the first ``c`` routes of its non-floor pooled
    queue.  The floors are kept whatever the scores, so they are never free
    to exchange: they are never counted as uncertain, and their intervals do
    not enter the extremes.  The exchange question is asked on the non-floor
    queue alone, with ``c`` held at its committed value: a kept queue route
    is uncertain when ``max(a - z*se, 0)`` is strictly below the largest
    ``a + z*se`` of a removed queue route, and a removed queue route is
    uncertain when ``a + z*se`` is strictly above the smallest
    ``max(a - z*se, 0)`` of a kept queue route.  ``a`` is the ranked census
    activity (the walk's own scores), ``se`` its census standard error, and
    ``z`` the census's ``order_normal_critical_value``.  Partition labels and
    floor membership are taken as committed.

    The fraction's declared denominator (``BOUNDARY_UNCERTAINTY_DENOMINATOR``)
    is the partition's activity over every eligible route, floors included.
    Each row also gives the numerator and the fraction of the whole field's
    activity.  A cut that keeps no non-floor route (floors only) or removes
    none has no exchange to make and reports zero.

    The queue passes run on the device
    (``cross_layer_order_device.queue_boundary_uncertainty``, the same linear
    algorithm); ``device`` is a torch device or the diagnostic's shared
    ``_DeviceField``.
    """

    from .cross_layer_order_device import queue_boundary_uncertainty

    started = time.perf_counter()
    base: dict[str, Any] = {
        "schema": BOUNDARY_UNCERTAINTY_SCHEMA,
        "statistic": "boundary_uncertain_activity_fraction",
        "adapted_from": "mechanistic_systems.circuits.irg.score_order.certify_prune_order_uncertainty",
        "is_certificate": False,
        "decision_role": "reporting_only",
        "governs_decisions": False,
        "applies_threshold": False,
    }
    if walk.setting.scope != PRUNE_SCOPE_GLOBAL:
        return {**base, "available": False, "reason": "layerwise_scope"}
    if not walk.scores_overridden:
        return {**base, "available": False, "reason": "scores_not_from_the_census_field"}
    if (
        critical_value is None
        or isinstance(critical_value, bool)
        or not math.isfinite(float(critical_value))
        or float(critical_value) <= 0.0
    ):
        return {**base, "available": False, "reason": "no_census_order_critical_value"}
    (run,) = walk.runs_by_query[str(query_id)]
    (lowerer,) = walk.lowerers_by_query[str(query_id)]
    field = _DeviceField.of(walk, query_id, device)
    flat_se, reason = field.pooled(standard_error_by_layer, name="loss_activity_standard_error")
    if flat_se is None:
        return {**base, "available": False, "reason": reason}
    scores = field.scores()
    negative_error, negative_activity = _any_negative(flat_se, scores)
    if negative_error:
        return {**base, "available": False, "reason": "negative_standard_error"}
    if negative_activity:
        # ``_scope_receipt`` refuses negative activity; so does this.
        return {**base, "available": False, "reason": "negative_activity"}
    z = float(critical_value)
    field_activity = float(scores.sum())
    nominal = {float(value) for value in lowerer.nominal_schedule}
    committed = _committed_cuts(run)
    layer_count = len(lowerer.group_layers)
    partitions: dict[str, Any] = {}
    cuts: list[dict[str, Any]] = []
    for partition in _PARTITIONS:
        positions = committed[partition]
        if not positions:
            continue
        queue = lowerer._non_floor[partition]
        size = int(queue.size)
        partition_activity = float(lowerer.partition_activity[partition])
        floor_members = lowerer._floor_members[partition]
        retained = [int(lowerer.retained_non_floor(partition, value)) for value in positions]
        interior = np.asarray(
            sorted({value for value in retained if 1 <= value <= size - 1}), dtype=np.int64
        )
        measured = None
        if interior.size:
            members = field.queue(partition)
            measured = queue_boundary_uncertainty(
                activity=scores[members],
                standard_error=flat_se[members],
                layers=field.layer_index(members),
                cuts=interior,
                critical_value=z,
                layer_count=layer_count,
            )
            del members
        index_of = {int(value): index for index, value in enumerate(interior)}
        fractions: list[float] = []
        for density, kept_count in zip(positions, retained):
            row: dict[str, Any] = {
                "partition_id": partition,
                "ladder_position": float(density),
                "nominal_rung": bool(float(density) in nominal),
                "retained_non_floor_count": int(kept_count),
                "removed_non_floor_count": int(size - kept_count),
            }
            if measured is None or kept_count not in index_of:
                row.update(
                    {
                        "exchange_possible": False,
                        "reason": (
                            "floors_only_no_kept_non_floor_route"
                            if kept_count <= 0
                            else "no_removed_non_floor_route"
                        ),
                        "boundary_uncertain_route_count": 0,
                        "boundary_uncertain_kept_route_count": 0,
                        "boundary_uncertain_removed_route_count": 0,
                        "boundary_uncertain_activity": 0.0,
                        "boundary_uncertain_activity_fraction": 0.0,
                        "boundary_uncertain_activity_fraction_of_field": 0.0,
                        "boundary_uncertain_route_count_by_layer": {},
                    }
                )
                fractions.append(0.0)
                cuts.append(row)
                continue
            index = index_of[kept_count]
            kept_layers = measured["kept_counts_by_layer"][index]
            removed_layers = measured["removed_counts_by_layer"][index]
            activity = float(
                measured["kept_activity"][index] + measured["removed_activity"][index]
            )
            fraction = float(min(activity / partition_activity, 1.0)) if partition_activity > 0 else None
            if fraction is not None:
                fractions.append(fraction)
            by_layer = kept_layers + removed_layers
            row.update(
                {
                    "exchange_possible": True,
                    "largest_removed_upper_end": float(measured["largest_removed_upper"][index]),
                    "smallest_kept_lower_end": float(measured["smallest_kept_lower"][index]),
                    "boundary_uncertain_route_count": int(by_layer.sum()),
                    "boundary_uncertain_kept_route_count": int(kept_layers.sum()),
                    "boundary_uncertain_removed_route_count": int(removed_layers.sum()),
                    "boundary_uncertain_activity": activity,
                    "boundary_uncertain_activity_fraction": fraction,
                    "boundary_uncertain_activity_fraction_of_field": (
                        float(min(activity / field_activity, 1.0)) if field_activity > 0 else None
                    ),
                    "boundary_uncertain_route_count_by_layer": {
                        layer: int(by_layer[position])
                        for position, layer in enumerate(lowerer.group_layers)
                        if int(by_layer[position])
                    },
                }
            )
            cuts.append(row)
        queue_activity = (
            measured["queue_activity"]
            if measured is not None
            else float(scores[field.queue(partition)].sum()) if size else 0.0
        )
        partitions[partition] = {
            "partition_route_count": int(lowerer.partition_route_counts[partition]),
            "non_floor_queue_route_count": size,
            "pinned_floor_route_count": int(floor_members.size),
            "denominator_activity": partition_activity,
            "non_floor_queue_activity": float(queue_activity),
            "pinned_floor_activity": (
                float(np.sum(lowerer._flat_scores[floor_members])) if floor_members.size else 0.0
            ),
            "committed_cut_count": len(positions),
            "maximum_boundary_uncertain_activity_fraction": (
                float(max(fractions)) if fractions else None
            ),
        }
        del measured
    maxima = [
        item["maximum_boundary_uncertain_activity_fraction"]
        for item in partitions.values()
        if item["maximum_boundary_uncertain_activity_fraction"] is not None
    ]
    return {
        **base,
        "available": True,
        "score_field": "loss_activity (the walk's ranked scores; also the activity weight)",
        "standard_error_field": "loss_activity_standard_error",
        "critical_value": z,
        "critical_value_source": "census order_normal_critical_value",
        "denominator": BOUNDARY_UNCERTAINTY_DENOMINATOR,
        "field_activity": field_activity,
        "exchange_set": "the partition's non-floor pooled queue; pinned floor routes are never free to exchange",
        "held_fixed": "the committed kept count c, the partition labels and the floor membership",
        "rule": (
            "at cut c: kept queue route uncertain iff max(a - z*se, 0) < max over removed queue "
            "routes of (a + z*se); removed queue route uncertain iff a + z*se > min over kept "
            "queue routes of max(a - z*se, 0); strict comparisons; activity-weighted"
        ),
        "audited_cuts": "every_committed_position_per_partition",
        "partitions": partitions,
        "maximum_boundary_uncertain_activity_fraction": float(max(maxima)) if maxima else None,
        "cuts": cuts,
        "seconds": float(time.perf_counter() - started),
    }


def descriptive_pointwise_band(
    walk: PruneScopeWalk,
    query_id: str,
    *,
    standard_error_by_layer: Mapping[str, Any] | None,
    critical_value: float = ORDER_DIAGNOSTIC_CRITICAL_VALUE,
    device: Any = None,
) -> dict[str, Any]:
    """A descriptive pointwise measure at each committed global cut: NOT a certificate.

    For a committed cut of partition ``p`` that keeps the first ``c``
    non-floor routes of the pooled order, the threshold ``t`` is the midpoint
    between the last kept and the first removed non-floor score.  A kept
    non-floor route is counted when ``score - z * se <= t``, a removed one
    when ``score + z * se >= t`` (``se`` = the census standard error of the
    ranked activity; ``z`` a fixed pointwise normal quantile, with no
    simultaneous correction across routes or cuts).  It reports those counts
    (total and per layer) and their activity as a fraction of the partition's
    and of the whole field's activity, at every committed position (capped
    intermediate positions included).  It certifies nothing and no decision
    reads it; the certificate call is ``cross_layer_order_certificate``.

    Every cut of a partition is counted in one device pass
    (``cross_layer_order_device.pointwise_band``; the same window, threshold
    and comparisons); ``device`` is a torch device or the diagnostic's shared
    ``_DeviceField``.
    """

    from .cross_layer_order_device import pointwise_band

    started = time.perf_counter()
    base = {
        "schema": POINTWISE_BAND_SCHEMA,
        "is_certificate": False,
        "label": (
            "descriptive pointwise band at each committed global cut; not a "
            "certificate (pointwise normal quantile, no simultaneous coverage)"
        ),
        "decision_role": "reporting_only",
        "critical_value": float(critical_value),
    }
    if walk.setting.scope != PRUNE_SCOPE_GLOBAL:
        return {**base, "available": False, "reason": "layerwise_scope"}
    if standard_error_by_layer is None:
        return {**base, "available": False, "reason": "no_census_standard_errors"}
    if not walk.scores_overridden:
        return {**base, "available": False, "reason": "scores_not_from_the_census_field"}
    (run,) = walk.runs_by_query[str(query_id)]
    (lowerer,) = walk.lowerers_by_query[str(query_id)]
    field = _DeviceField.of(walk, query_id, device)
    flat_se, reason = field.pooled(standard_error_by_layer, name="standard_error")
    if flat_se is None:
        return {**base, "available": False, "reason": reason}
    (negative_error,) = _any_negative(flat_se)
    if negative_error:
        return {**base, "available": False, "reason": "negative_standard_error"}
    scores = field.scores()
    field_activity = float(scores.sum())
    nominal = {float(value) for value in lowerer.nominal_schedule}
    committed = _committed_cuts(run)
    z = float(critical_value)
    layer_count = len(lowerer.group_layers)
    cuts = []
    for partition in _PARTITIONS:
        positions = committed[partition]
        if not positions:
            continue
        size = int(lowerer._non_floor[partition].size)
        partition_activity = float(lowerer.partition_activity[partition])
        retained = [int(lowerer.retained_non_floor(partition, density)) for density in positions]
        interior = np.asarray(sorted({c for c in retained if 0 < c < size}), dtype=np.int64)
        band = None
        if interior.size:
            members = field.queue(partition)
            band = pointwise_band(
                scores=scores[members],
                standard_error=flat_se[members],
                layers=field.layer_index(members),
                cuts=interior,
                critical_value=z,
                layer_count=layer_count,
            )
            del members
        index_of = {int(value): index for index, value in enumerate(interior)}
        for density, c in zip(positions, retained):
            row: dict[str, Any] = {
                "partition_id": partition,
                "ladder_position": float(density),
                "nominal_rung": bool(density in nominal),
                "retained_non_floor_count": int(c),
            }
            if c <= 0 or c >= size:
                row.update({"threshold": None, "reason": "no_boundary_inside_the_partition"})
                cuts.append(row)
                continue
            index = index_of[c]
            kept_layers = band["kept_counts_by_layer"][index]
            removed_layers = band["removed_counts_by_layer"][index]
            layer_counts = kept_layers + removed_layers
            activity = float(band["activity"][index])
            row.update(
                {
                    "threshold": float(band["threshold"][index]),
                    "band_route_count": int(layer_counts.sum()),
                    "band_kept_route_count": int(kept_layers.sum()),
                    "band_removed_route_count": int(removed_layers.sum()),
                    "band_activity_fraction_of_partition": (
                        float(activity / partition_activity) if partition_activity > 0 else None
                    ),
                    "band_activity_fraction_of_field": (
                        float(activity / field_activity) if field_activity > 0 else None
                    ),
                    "band_route_count_by_layer": {
                        layer: int(layer_counts[position])
                        for position, layer in enumerate(lowerer.group_layers)
                        if int(layer_counts[position])
                    },
                }
            )
            cuts.append(row)
        del band
    return {
        **base,
        "available": True,
        "score_field": "loss_activity",
        "standard_error_field": "loss_activity_standard_error",
        "audited_cuts": "every_committed_position_per_partition",
        "cuts": cuts,
        "seconds": float(time.perf_counter() - started),
    }


def census_order_critical_value(measurement_receipt: Mapping[str, Any] | None) -> float | None:
    """The census's simultaneous order critical value, from its field receipt."""

    block = (measurement_receipt or {}).get("analytic_sign_partition")
    if not isinstance(block, Mapping):
        return None
    value = block.get("order_normal_critical_value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def cross_layer_order_diagnostic(
    walk: PruneScopeWalk,
    query_id: str,
    *,
    ordering_field: Mapping[str, Mapping[str, Any] | None] | None,
    critical_value: float | None,
    maximum_mask_disagreement_activity_fraction: float,
    device: Any = None,
) -> dict[str, Any]:
    """The reporting-only cross-layer order block of a global walk's receipt.

    ``ordering_field`` holds the stored census field by layer:
    ``loss_activity_standard_error``, ``loss_mean`` and
    ``loss_mean_standard_error``.  The block carries three quantities,
    each under its own name: ``consequence_price`` (the certificate
    call, ``cross_layer_order_certificate``; on one stored look its content
    is the consequence price), ``boundary_uncertain_activity`` (the
    unresolved activity fraction at each committed cut, with its declared
    denominator) and ``descriptive_pointwise_band``, which is not a
    certificate.  None governs any decision.

    The three parts run as torch kernels on one device (``device``; by
    default the current CUDA device, else the CPU) and share one upload of the
    field; only the per-cut rows come back.  ``device`` names where it ran.
    """

    field_by_name = dict(ordering_field or {})
    field = _DeviceField(walk, query_id, device)
    return {
        "schema": CROSS_LAYER_ORDER_DIAGNOSTIC_SCHEMA,
        "decision_role": "reporting_only",
        "order_field": "loss_activity",
        "damage_field": "loss_mean",
        "governs_decisions": False,
        "governing_order_certificate": "layerwise_census_within_layer_partition_only",
        "device": str(field.device),
        "quantities": {
            "consequence_price": (
                "certify_prune_order_stability(scope=cross_layer) on one stored "
                "look: consequence price at the audited pooled cuts; its "
                "mask-disagreement fields hold the no-previous-look value"
            ),
            "boundary_uncertain_activity": (
                "unresolved (boundary-uncertain) activity fraction at every "
                "committed pooled cut, non-floor queue, floors pinned"
            ),
            "descriptive_pointwise_band": "descriptive only, not a certificate",
        },
        "consequence_price": cross_layer_order_certificate(
            walk,
            query_id,
            score_standard_error_by_layer=field_by_name.get("loss_activity_standard_error"),
            route_damage_by_layer=field_by_name.get("loss_mean"),
            route_damage_standard_error_by_layer=field_by_name.get(
                "loss_mean_standard_error"
            ),
            critical_value=critical_value,
            maximum_mask_disagreement_activity_fraction=(
                maximum_mask_disagreement_activity_fraction
            ),
            device=field,
        ),
        "boundary_uncertain_activity": boundary_uncertain_activity(
            walk,
            query_id,
            standard_error_by_layer=field_by_name.get("loss_activity_standard_error"),
            critical_value=critical_value,
            device=field,
        ),
        "descriptive_pointwise_band": descriptive_pointwise_band(
            walk,
            query_id,
            standard_error_by_layer=field_by_name.get("loss_activity_standard_error"),
            device=field,
        ),
        "optimality_certificate": "none",
    }


__all__ = [
    "GLOBAL_GROUP_ID",
    "GRAIN_RULE_CAPPED",
    "GRAIN_RULE_LAYERWISE_MATCHED",
    "GRAIN_RULE_NOMINAL",
    "KEEP_RULE",
    "BOUNDARY_UNCERTAINTY_DENOMINATOR",
    "BOUNDARY_UNCERTAINTY_SCHEMA",
    "CAP_SEMANTICS",
    "CROSS_LAYER_ORDER_DIAGNOSTIC_SCHEMA",
    "NOTE08_REFERENCE_CAP",
    "NOTE08_REFERENCE_GAMMA",
    "ORDER_DIAGNOSTIC_CRITICAL_VALUE",
    "POINTWISE_BAND_SCHEMA",
    "REGISTERED_ALEXNET_LAYER_GROUPS",
    "REGISTERED_ARCHITECTURE_FAMILY",
    "REGISTERED_SENSITIVITY_TAU",
    "REGISTERED_SENSITIVITY_TAU_ENV",
    "PROPOSAL_ROUTE_CAP_ENV",
    "PROPOSAL_GRAIN_ENV",
    "PROPOSAL_GRAIN_LAYERWISE_MATCHED",
    "PROPOSAL_GRAIN_NOMINAL",
    "PRUNE_SCOPES",
    "PRUNE_SCOPE_ENV",
    "PRUNE_SCOPE_FAMILIES",
    "PRUNE_SCOPE_GLOBAL",
    "PRUNE_SCOPE_LAYERWISE",
    "PRUNE_SCOPE_WALK_SCHEMA",
    "PooledCandidateLowerer",
    "PruneScopeSetting",
    "PruneScopeWalk",
    "TAU_PIN_RULE",
    "TAU_PIN_RULE_CAMPAIGN",
    "TIE_RULE",
    "boundary_uncertain_activity",
    "census_order_critical_value",
    "cross_layer_order_certificate",
    "cross_layer_order_diagnostic",
    "descriptive_pointwise_band",
    "executable_ladder_for_kinds",
    "registered_sensitivity_tau",
    "resolve_prune_scope_setting",
    "run_prune_scope_panel",
    "run_profiled_native_bce_precision_pooled_group",
    "tau_pin_rule",
    "validate_prune_scope_admission",
    "validate_registered_prune_scope_route_groups",
]
