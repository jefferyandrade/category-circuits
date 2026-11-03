"""Historical schedule-replay walker for the pre-registered v0 probe.

Implements the pre-registered schedule-replay probe under the
verified ``historical_symmetric_frontier_replay_schedule_v2`` data.  The
walker is ADDITIVE: it replaces only the tau-traversal panel call when the
``v0_schedule_replay_spec`` knob is set, and reuses the same decision
provider, calibration fit path, and paired frozen-mask audit machinery as
the baseline v0 arm.

Binding obligations:

* exactly ONE fresh-draw paired look per replayed step; a failed step
  is never retried; the candidate sequence is the recorded branch data.
* per-layer ``stop_reason`` in {ladder_exhausted, budget_tripped,
  replay_cap_reached}; the run receipt discloses the fraction of layers
  stopped by budget (``budget_dominated_stopping``) without adjudicating it.
* Census-loss campaign — replay rank-source labels both consume the dense
  calibrated pruning-loss activity, ranking within the registered sign groups.
* acceptance uses RAW relative cumulative damage (no span-times-
  connections normalization); candidate fits go through the same provider
  entry the decision backend uses for candidate states; the policy knobs
  survive ``resolve()`` under a pinned regression test.
* v3 power design (binding) — the
  per-step look pairs CANDIDATE vs DENSE on the same fresh draw; the
  acceptance statistic is UCB(paired candidate-minus-dense BCE) divided by
  the LOCKED optimization dense certificate point estimate ``b``; the dense
  lane's measured mean is an audit disclosure, never a denominator.  The
  dense extraction fit is held by hard reference across group transitions,
  every drawn panel must cover the full oracle stratum axis (the per-look
  weight renormalization over present strata is the risk this removes), and
  v3 receipts are renamed so v2/v3 arms can never be pooled.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..evidence_continuous import ContinuousDecision, ContinuousEvidenceResult
from ..native_continuous import (
    NativeContinuousRun,
    NativeContinuousTraceSignature,
)
from ..sampled_path import sampled_path_mask_hash
from .contracts import content_hash

SCHEDULE_REPLAY_ALGORITHM_VERSION = "historical_schedule_replay_walker_v3"
SCHEDULE_REPLAY_SCHEDULE_SCHEMA = (
    "historical_symmetric_frontier_replay_schedule_v2"
)
SCHEDULE_REPLAY_RUN_RECEIPT_SCHEMA = "v0_schedule_replay_run_receipt_v3"
SCHEDULE_REPLAY_STEP_RECEIPT_SCHEMA = "v0_schedule_replay_step_receipt_v3"
SCHEDULE_REPLAY_ACCEPTANCE_IDENTITY_V3 = (
    "paired_candidate_minus_dense_relative_cumulative_damage_ucb_vs_"
    "locked_dense_v3"
)
SCHEDULE_REPLAY_RANK_SOURCES = ("family", "local_attribution")
# Modern v0's cost coordinate: only these kinds are prunable connections.
_ELIGIBLE_ROUTE_KINDS = frozenset({"linear_weight", "conv_channel_pair"})
_BRANCHES = ("positive", "negative")
# Historical label polarity contract (masking.py): label 0 is the positive
# branch, label 1 is the negative branch.
_BRANCH_LABEL = {"positive": 0, "negative": 1}


def load_schedule_replay_spec(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate one sealed replay schedule; return (data, receipt)."""

    source = Path(path)
    raw = source.read_bytes()
    schedule = json.loads(raw.decode("utf-8"))
    validate_schedule_replay_spec(schedule)
    receipt = {
        "path": str(source),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "schema_name": str(schedule["schema_name"]),
        "provenance": str(schedule.get("provenance", "")),
        "layer_count": len(schedule["layers"]),
        "total_recorded_steps": int(
            sum(int(layer["recorded_step_count"]) for layer in schedule["layers"])
        ),
    }
    return schedule, receipt


def validate_schedule_replay_spec(schedule: Mapping[str, Any]) -> None:
    if schedule.get("schema_name") != SCHEDULE_REPLAY_SCHEDULE_SCHEMA:
        raise ValueError(
            "schedule replay requires schema "
            f"{SCHEDULE_REPLAY_SCHEDULE_SCHEMA!r}, got "
            f"{schedule.get('schema_name')!r}"
        )
    order = tuple(str(value) for value in schedule.get("executed_layer_order", ()))
    layers = tuple(schedule.get("layers", ()))
    if not order or len(set(order)) != len(order):
        raise ValueError("executed_layer_order must be nonempty and unique")
    if tuple(str(layer.get("layer")) for layer in layers) != order:
        raise ValueError(
            "schedule layers must list exactly executed_layer_order, in order"
        )
    for layer in layers:
        branches = tuple(str(value) for value in layer.get("branch_sequence", ()))
        # A zero-step layer is legal recorded data (history never stepped
        # it); the replay walks it as an immediately exhausted ladder.
        if any(branch not in _BRANCHES for branch in branches):
            raise ValueError(
                f"layer {layer.get('layer')!r} has an illegal branch sequence"
            )
        if int(layer.get("recorded_step_count", -1)) != len(branches):
            raise ValueError(
                f"layer {layer.get('layer')!r} recorded_step_count differs "
                "from its branch sequence"
            )


def _schedule_replay_route_binding(
    schedule: Mapping[str, Any], route_groups: Sequence[Any],
) -> tuple[tuple[Any, ...], dict[str, str], tuple[str, ...]]:
    """Use the walk's eligible axis and module/layer identities for replay cuts."""

    eligible = tuple(
        group for group in route_groups
        if str(group.route_kind) in _ELIGIBLE_ROUTE_KINDS
    )
    # Replay records name historical module paths; modern groups use layer ids.
    spec_identity: dict[str, Any] = {}
    for group in eligible:
        identities = {
            str(group.layer_id), str(getattr(group, "module_path", group.layer_id)),
        }
        for key in identities:
            existing = spec_identity.setdefault(key, group)
            if existing is not group:
                raise ValueError(
                    f"replay layer identity {key!r} is ambiguous across the "
                    "eligible route axis"
                )
    executed_order_spec = tuple(str(value) for value in schedule["executed_layer_order"])
    missing = tuple(layer for layer in executed_order_spec if layer not in spec_identity)
    if missing:
        raise ValueError(
            "schedule layers are outside the eligible route axis: " + ", ".join(missing)
        )
    spec_to_layer_id = {
        spec_name: str(spec_identity[spec_name].layer_id)
        for spec_name in executed_order_spec
    }
    if len(set(spec_to_layer_id.values())) != len(executed_order_spec):
        raise ValueError("schedule layers collapse onto a shared route group")
    return eligible, spec_to_layer_id, tuple(
        spec_to_layer_id[name] for name in executed_order_spec
    )


def schedule_replay_executable_retained_counts(
    schedule: Mapping[str, Any],
    route_groups: Sequence[Any],
    partition_route_counts_by_layer: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, tuple[int, ...]]]:
    """Every possible accepted replay prefix, from the walk's integer rule.

    First refusal ends a layer, so the all-accepted branch sequence contains
    every candidate cutoff that its walk can execute. Empty branches remain
    zero; a last full removal is recorded although it needs no order cut.
    Eligible layers absent from the replay stay dense and have no cuts.
    """

    validate_schedule_replay_spec(schedule)
    eligible, spec_to_layer_id, _executed_order = _schedule_replay_route_binding(
        schedule, route_groups,
    )
    retained: dict[str, dict[str, list[int]]] = {}
    for group in eligible:
        layer = str(group.layer_id)
        counts = partition_route_counts_by_layer[layer]
        retained[layer] = {branch: [int(counts[branch])] for branch in _BRANCHES}
        if any(values[0] < 0 for values in retained[layer].values()):
            raise ValueError("replay partition route counts must be nonnegative")
    for record in schedule["layers"]:
        layer = spec_to_layer_id[str(record["layer"])]
        for branch in record["branch_sequence"]:
            counts = retained[layer][str(branch)]
            remaining = counts[-1]
            counts.append(remaining - int(math.ceil(0.1 * remaining)))
    return {
        layer: {branch: tuple(counts) for branch, counts in partitions.items()}
        for layer, partitions in retained.items()
    }


@dataclass(frozen=True)
class ScheduleReplayPolicy:
    """Replay knobs plus the naturalized per-step draw allocation.

    ``resolve()`` rebuilds the policy field by field against provider facts;
    a field omitted from such a rebuild is silently dropped, so both replay
    knobs must survive ``resolve()``.
    """

    schedule_replay_spec: str
    schedule_replay_rank_source: str = "family"
    maximum_relative_bce_increase: float = 0.20
    confidence_level: float = 0.95
    oracle_initial_packet: int = 32
    random_seed: int = 0
    maximum_commits: int | None = None
    stratum_draw_counts: Mapping[str, int] | None = None

    def validate(self) -> None:
        if not str(self.schedule_replay_spec):
            raise ValueError("schedule replay requires a sealed spec identity")
        if self.schedule_replay_rank_source not in SCHEDULE_REPLAY_RANK_SOURCES:
            raise ValueError(
                "schedule replay rank source must be one of "
                f"{SCHEDULE_REPLAY_RANK_SOURCES!r}"
            )
        budget = float(self.maximum_relative_bce_increase)
        if not math.isfinite(budget) or budget < 0.0:
            raise ValueError(
                "maximum_relative_bce_increase must be finite and nonnegative"
            )
        if not 0.0 < float(self.confidence_level) < 1.0:
            raise ValueError("confidence_level must lie in (0,1)")
        if int(self.oracle_initial_packet) <= 0:
            raise ValueError("oracle_initial_packet must be positive")
        if self.maximum_commits is not None and int(self.maximum_commits) <= 0:
            raise ValueError("maximum_commits must be positive when declared")
        if self.stratum_draw_counts is not None and (
            not self.stratum_draw_counts
            or any(int(value) < 2 for value in self.stratum_draw_counts.values())
        ):
            raise ValueError(
                "resolved stratum draw counts require at least two per stratum"
            )

    def resolve(
        self,
        *,
        oracle_statistical_weights: Mapping[str, float],
        oracle_minimum_per_stratum: int,
    ) -> "ScheduleReplayPolicy":
        """Naturalize the per-step logical draw counts from provider facts.

        The per-step packet is the baseline per-decision oracle packet
        (``oracle_initial_packet``) split across the provider's declared
        oracle strata by their Neyman estimand weights, floored at the
        provider's per-stratum minimum — the confirm-path weight contract.
        """

        self.validate()
        weights = {
            str(key): float(value)
            for key, value in oracle_statistical_weights.items()
        }
        if not weights or any(
            not math.isfinite(value) or value <= 0.0
            for value in weights.values()
        ):
            raise ValueError(
                "schedule replay requires positive oracle stratum weights"
            )
        minimum = max(2, int(oracle_minimum_per_stratum))
        total_weight = sum(weights.values())
        packet = int(self.oracle_initial_packet)
        counts = {
            stratum: max(
                minimum,
                int(math.ceil(packet * weight / total_weight)),
            )
            for stratum, weight in weights.items()
        }
        # resolve() rebuilds the policy field by field; omitting a field here
        # silently reverts it to its default in every production run while
        # direct-policy unit tests keep passing.
        resolved = ScheduleReplayPolicy(
            schedule_replay_spec=str(self.schedule_replay_spec),
            schedule_replay_rank_source=str(self.schedule_replay_rank_source),
            maximum_relative_bce_increase=float(
                self.maximum_relative_bce_increase
            ),
            confidence_level=float(self.confidence_level),
            oracle_initial_packet=int(self.oracle_initial_packet),
            random_seed=int(self.random_seed),
            maximum_commits=(
                None if self.maximum_commits is None else int(self.maximum_commits)
            ),
            stratum_draw_counts=dict(counts),
        )
        resolved.validate()
        return resolved


def resolve_replay_rank_scores(
    *,
    rank_source: str,
    route_groups: Sequence[Any],
    partition_loss_activity: Mapping[str, Any] | None,
) -> dict[str, np.ndarray]:
    """Read dense pruning-loss activity inside the registered sign groups.

    Historical rank-source labels remain replay metadata. Every label reads
    the same loss-gradient census field for this frozen campaign.
    """
    if rank_source not in SCHEDULE_REPLAY_RANK_SOURCES:
        raise ValueError(f"unknown schedule replay rank source {rank_source!r}")
    if partition_loss_activity is None:
        raise ValueError("schedule replay requires the dense census loss_activity field")
    scores: dict[str, np.ndarray] = {}
    for group in route_groups:
        if str(group.route_kind) not in _ELIGIBLE_ROUTE_KINDS:
            continue
        layer_id = str(group.layer_id)
        size = int(np.prod(group.shape, dtype=np.int64))
        if layer_id not in partition_loss_activity:
            raise ValueError(f"census loss_activity is missing replay layer {layer_id!r}")
        values = np.asarray(partition_loss_activity[layer_id], dtype=np.float64).reshape(-1)
        if values.shape != (size,):
            raise ValueError(f"replay rank scores have the wrong size for {layer_id!r}")
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError(f"replay loss activity must be finite and nonnegative for {layer_id!r}")
        scores[layer_id] = values
    if not scores:
        raise ValueError("schedule replay found no eligible route groups")
    return scores


def _sign_member_indices(
    labels: np.ndarray,
    branch: str,
) -> np.ndarray:
    return np.flatnonzero(labels == _BRANCH_LABEL[branch]).astype(
        np.int64, copy=False
    )


def _branch_density(retained: np.ndarray, members: np.ndarray) -> float:
    # Branch-relative retention; an empty structural branch is by convention
    # fully retained (there is nothing to prune).
    if members.size == 0:
        return 1.0
    return float(np.count_nonzero(retained[members]) / members.size)


def _fresh_step_panel(
    provider: Any,
    *,
    stream_id: str,
    counts: Mapping[str, int],
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    dict[str, tuple[str, ...]],
    dict[str, float],
]:
    """Draw one class-resolved fresh panel from the step's logical stream.

    Copies the accept-confirm stream machinery exactly: logical draws via the
    provider's IID prefix, physical resolution via ``_source_context_ids``,
    per-stratum dedupe in draw order, and deterministic top-up (doubling a
    deficient stratum extends the same prefix, leaving earlier draws
    identical) until every stratum holds two unique physical rows.
    """

    working = {str(key): int(value) for key, value in counts.items()}
    populations: dict[str, list[str]] = {}
    draws: tuple[str, ...] = ()
    physical_ids: tuple[str, ...] = ()
    for _attempt in range(5):
        draws = provider._iid_draw_prefix(
            role="proposal_adjudication",
            stream_id=stream_id,
            counts=working,
        )
        physical_ids = provider._source_context_ids(
            draws,
            role="proposal_adjudication",
        )
        populations = {}
        for draw_id, context_id in zip(draws, physical_ids):
            stratum = provider._context_stratum(
                draw_id,
                role="proposal_adjudication",
            )
            rows = populations.setdefault(stratum, [])
            if context_id not in rows:
                rows.append(context_id)
        deficient = tuple(
            stratum
            for stratum, rows in populations.items()
            if len(rows) < 2
        )
        if not deficient:
            break
        for stratum in deficient:
            working[stratum] = int(working[stratum]) * 2
    else:
        raise RuntimeError(
            "schedule replay could not assemble two unique rows per stratum"
        )
    sealed_populations = {
        stratum: tuple(rows)
        for stratum, rows in populations.items()
        if rows
    }
    # v3 condition: the sealed panel must cover EXACTLY the full oracle
    # stratum axis.  Renormalizing the weights over merely-present strata
    # would silently change the measured population against the locked
    # dense certificate b — the adjudicated deciding risk.
    missing_strata = tuple(
        stratum
        for stratum in provider._oracle_statistical_strata
        if stratum not in sealed_populations
    )
    if missing_strata or set(sealed_populations) != set(
        provider._oracle_statistical_strata
    ):
        raise RuntimeError(
            "schedule replay panel lost oracle stratum coverage; missing: "
            + ", ".join(missing_strata)
        )
    raw_weights = {
        stratum: float(provider._oracle_statistical_weights[stratum])
        for stratum in sealed_populations
    }
    weight_total = sum(raw_weights.values())
    weights = {
        stratum: value / weight_total
        for stratum, value in raw_weights.items()
    }
    return draws, physical_ids, sealed_populations, weights


def run_schedule_replay_walker(
    *,
    schedule: Mapping[str, Any],
    policy: ScheduleReplayPolicy,
    backend: Any,
    query_id: str,
    route_groups: Sequence[Any],
    partition_labels_by_layer: Mapping[str, Any],
    rank_scores_by_layer: Mapping[str, Any],
    dense_baseline_bce: float,
    positive_loss_span_estimate: float,
    profile_hash: str,
    dense_certificate_bounds: tuple[float, float] | None = None,
    group_transition_callback: Callable[..., Any] | None = None,
    progress_callback: Callable[..., Any] | None = None,
) -> tuple[tuple[NativeContinuousRun, ...], dict[str, Any]]:
    """Replay the recorded schedule against the modern paired-oracle stack.

    Returns run items shaped exactly like the tau-panel's per-layer output
    (so ``_hard_mask_from_layers``, receipt assembly, and terminal
    validation run unchanged) plus one walker run receipt for the
    ``v0_schedule_replay`` block.

    v3 estimator: every step audits CANDIDATE vs the walk-initial DENSE
    state on one fresh paired draw; acceptance divides the paired-delta
    upper confidence bound by the locked ``dense_baseline_bce``.
    ``dense_certificate_bounds`` (the dense certificate CI) feeds only the
    pooled dense-lane disclosure, never the decision.
    """

    validate_schedule_replay_spec(schedule)
    baseline = float(dense_baseline_bce)
    span = float(positive_loss_span_estimate)
    if not math.isfinite(baseline) or baseline <= 0.0:
        raise ValueError(
            "schedule replay requires a positive locked dense baseline"
        )
    if not math.isfinite(span) or span <= 0.0:
        raise ValueError("schedule replay requires a positive loss span")
    certificate_bounds: tuple[float, float] | None = None
    if dense_certificate_bounds is not None:
        lower_bound, upper_bound = (
            float(dense_certificate_bounds[0]),
            float(dense_certificate_bounds[1]),
        )
        if (
            not math.isfinite(lower_bound)
            or not math.isfinite(upper_bound)
            or lower_bound > upper_bound
        ):
            raise ValueError(
                "dense certificate bounds must be finite and ordered"
            )
        certificate_bounds = (lower_bound, upper_bound)
    provider = backend.provider
    resolved = policy.resolve(
        oracle_statistical_weights=provider._oracle_statistical_weights,
        oracle_minimum_per_stratum=provider.oracle_minimum_per_stratum,
    )
    budget = float(resolved.maximum_relative_bce_increase)
    confidence = float(resolved.confidence_level)
    if resolved.stratum_draw_counts is None:
        raise RuntimeError(
            "schedule replay policy did not resolve its draw allocation"
        )

    def _progress(event: str, **facts: Any) -> None:
        if progress_callback is not None:
            progress_callback(event, **facts)

    eligible, spec_to_layer_id, executed_order = _schedule_replay_route_binding(
        schedule, route_groups,
    )
    labels_by_layer: dict[str, np.ndarray] = {}
    scores_by_layer: dict[str, np.ndarray] = {}
    for group in eligible:
        layer_id = str(group.layer_id)
        size = int(np.prod(group.shape, dtype=np.int64))
        labels = np.asarray(partition_labels_by_layer[layer_id]).reshape(-1)
        scores = np.asarray(
            rank_scores_by_layer[layer_id], dtype=np.float64
        ).reshape(-1)
        if labels.shape != (size,) or scores.shape != (size,):
            raise ValueError(
                f"replay labels/scores have the wrong size for {layer_id!r}"
            )
        if labels.dtype.kind not in {"i", "u", "b"} or not np.isin(
            labels, (0, 1)
        ).all():
            raise ValueError(
                f"replay partition labels must be integer zero/one for "
                f"{layer_id!r}"
            )
        labels_by_layer[layer_id] = labels.astype(np.int8, copy=False)
        scores_by_layer[layer_id] = scores

    # The committed state is a total function over the declared route axis;
    # unvisited groups (including protected non-connection routes) stay dense.
    committed: dict[str, np.ndarray] = {
        str(group.layer_id): np.ones(group.shape, dtype=bool)
        for group in route_groups
    }
    committed_hard = backend._hard_mask(committed)
    if committed_hard.mask_hash not in provider._calibration_fits:
        # Mirror of the decision backend's parent-fit precondition: the
        # dense trajectory state must already carry its frozen accepted fit.
        raise RuntimeError(
            "schedule replay parent lacks its frozen accepted calibration state"
        )
    # v3 pair reference: the walk-initial all-ones state IS the dense
    # reference for every step, and its extraction fit is held by hard
    # reference — layer-boundary cache retirement must never let the audit
    # see a refit (a silent break would surface at layer 2, not step 1).
    dense_hard = committed_hard
    dense_fit = provider._calibration_fits[dense_hard.mask_hash]
    dense_fit_restorations = 0
    committed_relative_damage = 0.0
    committed_relative_damage_ucb = 0.0
    # Paired candidate-minus-dense LEVEL of the committed state; per-step
    # increments are derived by subtracting it (diagnostic only — the two
    # levels come from different draws, so the shift is a correlated-draw
    # approximation, never a third audit lane).
    committed_level_mean = 0.0
    dense_lane_means: list[float] = []
    accepted_relative_ucbs: list[float] = []
    accepted_total = 0
    rejected_total = 0
    skipped_total = 0
    steps_replayed_total = 0
    connections_removed_total = 0
    images_consumed_total = 0
    cap_tripped = False
    runs: list[NativeContinuousRun] = []
    layer_receipts: list[dict[str, Any]] = []
    schedule_layer_records = {
        spec_to_layer_id[str(entry["layer"])]: entry
        for entry in schedule["layers"]
        if str(entry["layer"]) in spec_to_layer_id
    }
    _progress(
        "v0_schedule_replay_started",
        rank_source=resolved.schedule_replay_rank_source,
        order_field="loss_activity",
        layer_count=len(executed_order),
        total_recorded_steps=int(
            sum(
                int(entry["recorded_step_count"])
                for entry in schedule["layers"]
            )
        ),
    )

    def _layer_flat(layer_id: str) -> np.ndarray:
        return committed[layer_id].reshape(-1)

    def _layer_run(
        *,
        layer_id: str,
        decisions: tuple[ContinuousDecision, ...],
        step_receipts: tuple[Mapping[str, Any], ...],
        stopping_reason: str,
        images_consumed: int,
    ) -> NativeContinuousRun:
        labels = labels_by_layer.get(layer_id)
        flat = _layer_flat(layer_id)
        if labels is None:
            positive_density = negative_density = 1.0
        else:
            positive_density = _branch_density(
                flat, _sign_member_indices(labels, "positive")
            )
            negative_density = _branch_density(
                flat, _sign_member_indices(labels, "negative")
            )
        snapshot = {
            key: value.copy() for key, value in committed.items()
        }
        result = ContinuousEvidenceResult(
            positive_density=positive_density,
            negative_density=negative_density,
            committed_masks=snapshot,
            decisions=decisions,
            stopping_reason=stopping_reason,
            delta_budget=budget,
            delta_spent=float(committed_relative_damage),
            images_consumed=int(images_consumed),
            ledger_receipt={
                "order_field": "loss_activity",
                "sequential_bce_decision_receipts": step_receipts,
            },
            moderation_prior=None,
            resolved_policy={
                "schema": "v0_schedule_replay_resolved_policy_v1",
                "schedule_replay_spec": resolved.schedule_replay_spec,
                "schedule_replay_rank_source": (
                    resolved.schedule_replay_rank_source
                ),
                "order_field": "loss_activity",
                "maximum_relative_bce_increase": budget,
                "confidence_level": confidence,
                "oracle_initial_packet": int(resolved.oracle_initial_packet),
                "stratum_draw_counts": dict(resolved.stratum_draw_counts),
                "random_seed": int(resolved.random_seed),
                "maximum_commits": resolved.maximum_commits,
            },
            algorithm_version=SCHEDULE_REPLAY_ALGORITHM_VERSION,
        )
        signature = NativeContinuousTraceSignature(
            profile_hash=str(profile_hash),
            query_id=str(query_id),
            layer_id=str(layer_id),
            final_positive_density=positive_density,
            final_negative_density=negative_density,
            final_mask_hash=sampled_path_mask_hash(snapshot),
            decision_count=len(decisions),
            images_consumed=int(images_consumed),
            stopping_reason=stopping_reason,
            delta_budget=budget,
            delta_spent=float(committed_relative_damage),
            algorithm_version=SCHEDULE_REPLAY_ALGORITHM_VERSION,
        )
        return NativeContinuousRun(
            result=result,
            signature=signature,
            maturity="schedule_replay_walker",
            analysis_status="eligible",
        )

    for layer_position, layer_id in enumerate(executed_order):
        if cap_tripped:
            break
        record = schedule_layer_records[layer_id]
        branch_sequence = tuple(
            str(value) for value in record["branch_sequence"]
        )
        if group_transition_callback is not None:
            # Runtime-only cache retirement at a layer boundary; the callback
            # cannot alter contexts, statistics, or masks.
            group_transition_callback(
                query_id=str(query_id),
                group_index=int(layer_position),
                layer_id=str(layer_id),
            )
        labels = labels_by_layer[layer_id]
        scores = scores_by_layer[layer_id]
        members = {
            branch: _sign_member_indices(labels, branch)
            for branch in _BRANCHES
        }
        pre_counts = {
            branch: int(members[branch].size) for branch in _BRANCHES
        }
        decisions: list[ContinuousDecision] = []
        step_receipts: list[dict[str, Any]] = []
        stop_reason: str | None = None
        steps_accepted_layer = 0
        first_rejected_step_index: int | None = None
        layer_images = 0
        for step_index, branch in enumerate(branch_sequence):
            if (
                resolved.maximum_commits is not None
                and accepted_total >= int(resolved.maximum_commits)
            ):
                stop_reason = "replay_cap_reached"
                cap_tripped = True
                break
            flat = _layer_flat(layer_id)
            branch_members = members[branch]
            remaining = branch_members[flat[branch_members]]
            remaining_count = int(remaining.size)
            previous_density = _branch_density(flat, branch_members)
            if remaining_count == 0:
                # A structurally empty branch cannot express the recorded
                # multiplicative step; record the skip and keep the fixed
                # candidate sequence intact (no evaluation, no retry).
                skipped_total += 1
                steps_replayed_total += 1
                decisions.append(
                    ContinuousDecision(
                        partition_id=branch,
                        previous_density=previous_density,
                        proposed_density=previous_density,
                        action="skip",
                        reason="empty_sign_group_branch",
                        looks=0,
                        panel_images=0,
                        mean_sensitivity=0.0,
                        certified_upper_bound=0.0,
                        threshold=budget,
                        delta_debit=0.0,
                        alpha_allocated=0.0,
                        panel_hashes=(),
                        removed_route_count=0,
                        certified_lower_bound=0.0,
                        confidence_level=confidence,
                    )
                )
                step_receipts.append(
                    {
                        "schema": SCHEDULE_REPLAY_STEP_RECEIPT_SCHEMA,
                        "order_field": "loss_activity",
                        "layer_id": layer_id,
                        "step_index": int(step_index),
                        "branch": branch,
                        "action": "skip",
                        "reason": "empty_sign_group_branch",
                        "suppressed": False,
                    }
                )
                continue
            prune_count = int(math.ceil(0.1 * remaining_count))
            # Stable descending magnitude order reproduces the panel's
            # lowering contract: ties retain the flattened route-coordinate
            # order, and the pruned set is the lowest-ranked tail.
            order = np.argsort(-scores[remaining], kind="stable")
            pruned_indices = remaining[order[remaining_count - prune_count :]]
            candidate_layer = committed[layer_id].copy()
            candidate_flat = candidate_layer.reshape(-1)
            candidate_flat[pruned_indices] = False
            candidate_masks = dict(committed)
            candidate_masks[layer_id] = candidate_layer
            candidate_hard = backend._hard_mask(candidate_masks)
            # v3: the audit reference is the walk-initial dense state under
            # its HELD extraction fit.  Re-assert membership before ANY fit
            # use — at each layer's first step the committed parent IS the
            # dense state, so a layer-boundary cache retirement would
            # otherwise break the warm-start lookup, not just the audit.
            if (
                provider._calibration_fits.get(dense_hard.mask_hash)
                is not dense_fit
            ):
                provider._calibration_fits[dense_hard.mask_hash] = dense_fit
                dense_fit_restorations += 1
            parent_fit = provider._calibration_fits[committed_hard.mask_hash]
            # Same fit entry the decision backend uses for candidate states:
            # a minimum extraction fit warm-started from the committed
            # parent's frozen accepted calibration.
            candidate_fit = provider.ensure_calibration(
                candidate_hard,
                warm_start_calibration=parent_fit,
            )
            stream_id = (
                "schedule-replay:"
                f"{query_id}:{layer_id}:{step_index}:{branch}:"
                f"{committed_hard.mask_hash}:{candidate_hard.mask_hash}:"
                f"{int(resolved.random_seed)}"
            )
            draws, physical_ids, sealed_populations, weights = (
                _fresh_step_panel(
                    provider,
                    stream_id=stream_id,
                    counts=resolved.stratum_draw_counts,
                )
            )
            # Re-assert once more immediately before the audit: nothing
            # between the panel draw and the paired look may swap the held
            # dense fit for a refit.
            if (
                provider._calibration_fits.get(dense_hard.mask_hash)
                is not dense_fit
            ):
                provider._calibration_fits[dense_hard.mask_hash] = dense_fit
                dense_fit_restorations += 1
            # Exactly one paired look per replayed step, no retry.
            audit = provider.evaluate_frozen_mask_bce_pair(
                dense_hard,
                candidate_hard,
                evaluation_context_ids_by_stratum=sealed_populations,
                evaluation_stratum_weights=weights,
                batch_factory=provider.oracle_batch_factory,
                confidence_level=confidence,
                allow_optimization_row_reuse=True,
            )
            candidate_summary = dict(audit["candidate_bce"])
            dense_reference_summary = dict(audit["reference_bce"])
            delta_summary = dict(audit["candidate_minus_reference_bce"])
            delta_mean = float(delta_summary["mean"])
            delta_lower = float(delta_summary["lower"])
            delta_upper = float(delta_summary["upper"])
            # Acceptance divides the PAIRED delta by the locked b; the
            # dense lane's measured mean is an audit disclosure, never a
            # denominator.
            relative_damage = delta_mean / baseline
            relative_damage_ucb = delta_upper / baseline
            accepted = relative_damage_ucb <= budget
            dense_lane_means.append(float(dense_reference_summary["mean"]))
            removed = int(prune_count)
            unique_rows = int(
                sum(len(rows) for rows in sealed_populations.values())
            )
            layer_images += unique_rows
            images_consumed_total += unique_rows
            steps_replayed_total += 1
            # The paired delta is a cumulative LEVEL vs dense; the step
            # increment subtracts the committed level.  Diagnostic only:
            # the two levels come from different draws.
            step_increment_mean = delta_mean - committed_level_mean
            step_increment_lower = delta_lower - committed_level_mean
            step_increment_upper = delta_upper - committed_level_mean
            per_connection = span * float(removed)
            audit_interval_method = (
                str(delta_summary.get("method", "")) or None
            )
            decision = ContinuousDecision(
                partition_id=branch,
                previous_density=previous_density,
                proposed_density=_branch_density(
                    candidate_flat, branch_members
                ),
                action="accept" if accepted else "reject",
                reason=(
                    "cumulative_damage_ucb_within_relative_budget"
                    if accepted
                    else "cumulative_damage_ucb_exceeded_relative_budget"
                ),
                looks=1,
                panel_images=unique_rows,
                mean_sensitivity=step_increment_mean / per_connection,
                certified_upper_bound=step_increment_upper / per_connection,
                threshold=budget,
                delta_debit=step_increment_mean if accepted else 0.0,
                alpha_allocated=1.0 - confidence,
                panel_hashes=(str(audit["context_design_hash"]),),
                removed_route_count=removed,
                certified_lower_bound=step_increment_lower / per_connection,
                confidence_level=confidence,
                interval_method=(
                    None
                    if audit_interval_method is None
                    else audit_interval_method
                    + "+committed_level_shift_diagnostic"
                ),
            )
            decisions.append(decision)
            step_receipt = {
                "schema": SCHEDULE_REPLAY_STEP_RECEIPT_SCHEMA,
                "order_field": "loss_activity",
                "layer_id": layer_id,
                "step_index": int(step_index),
                "branch": branch,
                "action": decision.action,
                "reason": decision.reason,
                "suppressed": False,
                "pre_branch_member_count": remaining_count,
                "post_branch_member_count": remaining_count - removed,
                "pruned_route_count": removed,
                "pre_sign_densities": {
                    other: _branch_density(flat, members[other])
                    for other in _BRANCHES
                },
                "post_sign_densities": {
                    other: _branch_density(candidate_flat, members[other])
                    for other in _BRANCHES
                },
                "dense_baseline_bce": baseline,
                "reference_mask_role": "dense",
                "dense_reference_bce": dense_reference_summary,
                "candidate_bce": candidate_summary,
                "paired_candidate_minus_dense_bce": delta_summary,
                "step_increment_bce": {
                    "mean": step_increment_mean,
                    "lower": step_increment_lower,
                    "upper": step_increment_upper,
                    "note": (
                        "diagnostic committed-level shift over correlated "
                        "draws; not a third audit lane"
                    ),
                },
                "relative_cumulative_damage": relative_damage,
                "relative_cumulative_damage_ucb": relative_damage_ucb,
                "maximum_relative_bce_increase": budget,
                "accepted": bool(accepted),
                "confidence_level": confidence,
                "one_fresh_draw_look_no_retry": True,
                "stratum_coverage_full": True,
                "stream_id": stream_id,
                "logical_draw_count": len(draws),
                "unique_row_counts_by_stratum": {
                    stratum: len(rows)
                    for stratum, rows in sealed_populations.items()
                },
                "stratum_weights": dict(weights),
                "draw_ids_content_hash": content_hash(
                    {
                        "stream_id": stream_id,
                        "logical_draws": tuple(draws),
                        "physical_context_ids": tuple(physical_ids),
                    },
                    prefix="irg-schedule-replay-draws-",
                ),
                "committed_mask_hash": committed_hard.mask_hash,
                "candidate_mask_hash": candidate_hard.mask_hash,
                "dense_reference_mask_hash": dense_hard.mask_hash,
                "committed_calibration_state_id": str(
                    parent_fit.calibration_state_id
                ),
                "candidate_calibration_state_id": str(
                    candidate_fit.calibration_state_id
                ),
                "dense_reference_calibration_state_id": str(
                    dense_fit.calibration_state_id
                ),
                "committed_fit_receipt_hash": str(
                    parent_fit.fit_receipt_hash
                ),
                "candidate_fit_receipt_hash": str(
                    candidate_fit.fit_receipt_hash
                ),
                "dense_reference_fit_receipt_hash": str(
                    dense_fit.fit_receipt_hash
                ),
                "independence_contract": str(
                    audit["independence_contract"]
                ),
                "pair_audit_context_design_hash": str(
                    audit["context_design_hash"]
                ),
                "audit_rows_used_for_optimization": bool(
                    audit["audit_rows_used_for_optimization"]
                ),
            }
            _progress(
                "v0_schedule_replay_step_decided",
                layer_id=layer_id,
                step_index=int(step_index),
                branch=branch,
                action=decision.action,
                relative_cumulative_damage_ucb=relative_damage_ucb,
            )
            if accepted:
                committed = candidate_masks
                committed_hard = candidate_hard
                committed_relative_damage = relative_damage
                committed_relative_damage_ucb = relative_damage_ucb
                committed_level_mean = delta_mean
                accepted_relative_ucbs.append(relative_damage_ucb)
                steps_accepted_layer += 1
                accepted_total += 1
                connections_removed_total += removed
                step_receipts.append(step_receipt)
            else:
                # First failure ends this layer; the failed step is never
                # retried and the committed predecessor is adopted.
                rejected_total += 1
                first_rejected_step_index = int(step_index)
                stop_reason = "budget_tripped"
                step_receipt["stop_reason"] = stop_reason
                step_receipts.append(step_receipt)
                break
        if stop_reason is None:
            stop_reason = "ladder_exhausted"
        flat = _layer_flat(layer_id)
        post_counts = {
            branch: int(np.count_nonzero(flat[members[branch]]))
            for branch in _BRANCHES
        }
        final_densities = {
            sign: _branch_density(flat, members[sign]) for sign in _BRANCHES
        }
        adopted_densities = {
            "positive": record.get("historical_adopted_positive"),
            "negative": record.get("historical_adopted_negative"),
        }
        layer_receipts.append(
            {
                "layer_id": layer_id,
                "stop_reason": stop_reason,
                "steps_replayed": len(decisions),
                "steps_accepted": int(steps_accepted_layer),
                "recorded_step_count": int(record["recorded_step_count"]),
                "first_rejected_step_index": first_rejected_step_index,
                "pre_sign_member_counts": pre_counts,
                "post_sign_member_counts": post_counts,
                "final_sign_densities": final_densities,
                "historical_adopted_densities": dict(adopted_densities),
                # A coincidental global retained fraction with mismatched
                # per-layer depths must not be over-read (v3 receipt).
                "final_minus_adopted_densities": {
                    sign: (
                        None
                        if adopted_densities[sign] is None
                        else float(
                            final_densities[sign]
                            - float(adopted_densities[sign])
                        )
                    )
                    for sign in _BRANCHES
                },
                # Cumulative paired-vs-dense level at this layer boundary;
                # boundary-to-boundary differences decompose the damage.
                "boundary_relative_cumulative_damage": float(
                    committed_relative_damage
                ),
                "boundary_relative_cumulative_damage_ucb": float(
                    committed_relative_damage_ucb
                ),
            }
        )
        runs.append(
            _layer_run(
                layer_id=layer_id,
                decisions=tuple(decisions),
                step_receipts=tuple(step_receipts),
                stopping_reason=stop_reason,
                images_consumed=layer_images,
            )
        )
        _progress(
            "v0_schedule_replay_layer_completed",
            layer_id=layer_id,
            stop_reason=stop_reason,
            steps_accepted=int(steps_accepted_layer),
        )

    walked_layers = tuple(receipt["layer_id"] for receipt in layer_receipts)
    unwalked_schedule_layers = tuple(
        layer for layer in executed_order if layer not in set(walked_layers)
    )
    non_schedule_layers: tuple[str, ...] = ()
    if not cap_tripped:
        # Layers outside the recorded schedule stay dense but still emit one
        # run item each, so the traversal-completeness accounting and the
        # cumulative committed-mask chain keep their tau-panel shape.
        non_schedule_layers = tuple(
            str(group.layer_id)
            for group in eligible
            if str(group.layer_id) not in set(executed_order)
        )
        for layer_id in non_schedule_layers:
            runs.append(
                _layer_run(
                    layer_id=layer_id,
                    decisions=(),
                    step_receipts=(),
                    stopping_reason="not_in_replay_schedule",
                    images_consumed=0,
                )
            )
    budget_layers = sum(
        1
        for receipt in layer_receipts
        if receipt["stop_reason"] == "budget_tripped"
    )
    stop_histogram: dict[str, int] = {}
    for receipt in layer_receipts:
        key = str(receipt["stop_reason"])
        stop_histogram[key] = stop_histogram.get(key, 0) + 1
    budget_fraction = (
        float(budget_layers / len(layer_receipts)) if layer_receipts else 0.0
    )
    ladder_fraction = (
        float(
            stop_histogram.get("ladder_exhausted", 0) / len(layer_receipts)
        )
        if layer_receipts
        else 0.0
    )
    # Pre-registered v3 failure modes: budget-dominated stopping (as v2) or
    # near-total ladder exhaustion (the probe then tested fixed recorded
    # depths, not an adaptive acceptance rule).  Disclosure, not verdict.
    if budget_fraction >= 0.9:
        replay_isolation_failure_mode = "budget_dominated"
    elif ladder_fraction >= 0.9:
        replay_isolation_failure_mode = "ladder_exhausted_dominated"
    else:
        replay_isolation_failure_mode = "none"
    dense_lane_pooled_mean = (
        float(np.mean(np.asarray(dense_lane_means, dtype=np.float64)))
        if dense_lane_means
        else None
    )
    dense_lane_within_certificate = (
        None
        if dense_lane_pooled_mean is None or certificate_bounds is None
        else bool(
            certificate_bounds[0]
            <= dense_lane_pooled_mean
            <= certificate_bounds[1]
        )
    )
    acceptance_margin_distribution = {
        "max_accepted_ucb": (
            float(max(accepted_relative_ucbs))
            if accepted_relative_ucbs
            else None
        ),
        "median_accepted_ucb": (
            float(np.median(np.asarray(accepted_relative_ucbs)))
            if accepted_relative_ucbs
            else None
        ),
        "accepted_count_ucb_in_0p10_0p20": int(
            sum(0.10 < value <= 0.20 for value in accepted_relative_ucbs)
        ),
    }
    run_receipt = {
        "schema": SCHEDULE_REPLAY_RUN_RECEIPT_SCHEMA,
        "algorithm_version": SCHEDULE_REPLAY_ALGORITHM_VERSION,
        "schedule_schema_name": str(schedule["schema_name"]),
        "schedule_provenance": str(schedule.get("provenance", "")),
        "rank_source": resolved.schedule_replay_rank_source,
        "order_field": "loss_activity",
        "executed_layer_order": list(executed_order),
        "acceptance_rule": {
            "identity": SCHEDULE_REPLAY_ACCEPTANCE_IDENTITY_V3,
            "statistic": (
                "upper(paired candidate_minus_dense_bce) / "
                "dense_baseline_bce <= maximum_relative_bce_increase"
            ),
            "threshold": budget,
            "confidence_level": confidence,
            "reference_mask_role": "dense",
            "span_connections_normalization_used": False,
            "dense_baseline_bce": baseline,
            "dense_baseline_role": "locked_constant_denominator",
            "dense_lane_row_reuse_bias": {
                "contract": (
                    "allow_optimization_row_reuse permits the dense lane "
                    "to score rows used in its own extraction fit"
                ),
                "direction": "conservative_inflates_measured_damage",
                "recorded_not_corrected": True,
            },
        },
        "per_step_look_contract": {
            "looks_per_step": 1,
            "retry_permitted": False,
            "stream_role": "proposal_adjudication",
            "stream_id_fields": [
                "query_id",
                "layer_id",
                "step_index",
                "branch",
                "committed_mask_hash",
                "candidate_mask_hash",
                "random_seed",
            ],
            "oracle_initial_packet": int(resolved.oracle_initial_packet),
            "stratum_draw_counts": dict(resolved.stratum_draw_counts),
            "row_reuse_contract": (
                "logical_stream_independent_physical_reuse_permitted"
            ),
        },
        "candidate_fit_contract": {
            "entry": (
                "provider.ensure_calibration(candidate, "
                "warm_start_calibration=committed_fit)"
            ),
            "matches_decision_backend_candidate_fit_path": True,
            "deployment_refit_handling": (
                "inherited_from_shared_provider_calibration_policy"
            ),
        },
        "layers": layer_receipts,
        "totals": {
            "steps_replayed": int(steps_replayed_total),
            "steps_accepted": int(accepted_total),
            "steps_rejected": int(rejected_total),
            "steps_skipped": int(skipped_total),
            "connections_removed": int(connections_removed_total),
            "images_consumed": int(images_consumed_total),
        },
        "stop_reason_histogram": stop_histogram,
        "budget_dominated_stopping": budget_fraction,
        "budget_dominated_stopping_disclosure": (
            "pre-registered A2 disclosure: if stopping is budget-dominated "
            "the schedule variable was not isolated; recorded, not "
            "adjudicated"
        ),
        "replay_isolation_failure_mode": replay_isolation_failure_mode,
        "acceptance_margin_distribution": acceptance_margin_distribution,
        "dense_lane_pooled_mean": dense_lane_pooled_mean,
        "dense_certificate_bounds": (
            None
            if certificate_bounds is None
            else [certificate_bounds[0], certificate_bounds[1]]
        ),
        # Disclosure only: a pooled dense-lane mean outside the locked
        # certificate CI is reported, never re-tuned against.
        "dense_lane_pooled_mean_within_certificate": (
            dense_lane_within_certificate
        ),
        "dense_fit_restorations": int(dense_fit_restorations),
        "replay_cap": resolved.maximum_commits,
        "replay_cap_tripped": bool(cap_tripped),
        "unwalked_schedule_layers": list(unwalked_schedule_layers),
        "non_schedule_dense_layers": list(non_schedule_layers),
        "final_mask_hash": committed_hard.mask_hash,
        "final_relative_cumulative_damage": float(committed_relative_damage),
        "final_relative_cumulative_damage_ucb": float(
            committed_relative_damage_ucb
        ),
    }
    _progress(
        "v0_schedule_replay_completed",
        steps_replayed=int(steps_replayed_total),
        steps_accepted=int(accepted_total),
        budget_dominated_stopping=run_receipt["budget_dominated_stopping"],
        replay_cap_tripped=bool(cap_tripped),
    )
    return tuple(runs), run_receipt


__all__ = [
    "SCHEDULE_REPLAY_ACCEPTANCE_IDENTITY_V3",
    "SCHEDULE_REPLAY_ALGORITHM_VERSION",
    "SCHEDULE_REPLAY_RANK_SOURCES",
    "SCHEDULE_REPLAY_RUN_RECEIPT_SCHEMA",
    "SCHEDULE_REPLAY_SCHEDULE_SCHEMA",
    "SCHEDULE_REPLAY_STEP_RECEIPT_SCHEMA",
    "ScheduleReplayPolicy",
    "load_schedule_replay_spec",
    "resolve_replay_rank_scores",
    "run_schedule_replay_walker",
    "schedule_replay_executable_retained_counts",
    "validate_schedule_replay_spec",
]
