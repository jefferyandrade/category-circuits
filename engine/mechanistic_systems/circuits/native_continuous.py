"""Continuous-evidence execution of profiled native layer groups.

The continuous engine (``evidence_foundation_continuous_v2``) is the canonical
identification default (see ``versions.canonical_identification_default``).
This module adapts one frozen-profile layer-group turn to
:func:`trace_continuous_evidence_frontier` while keeping the grid engine's
lowering, composition, and committed-mask chain semantics intact:

- the constant sensitivity price is the profile's ``sensitivity_tau`` — the
  group's distortion budget is an OUTPUT (``tau x prunable mass``), never a
  per-layer declaration;
- the density floor is the historical schedule's terminal density for the
  route-group kind, so continuous execution can never prune past the region
  the grid engine could reach;
- look cadence, horizon, and step sizes are naturalized from backend facts
  (class axis, bank depth, saturating GPU batch), not configured.

Claim-bearing execution remains contractually pinned to
``statistical_sequential_symmetric_frontier`` until the continuous engine has
its own promotion boundary; this module refuses ``claim_bearing=True``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from contracts.hashing import stable_digest

from .evidence_bce_precision import (
    BCE_PRECISION_V1,
    BCELossBoundaryDecisionRule,
    BCEPrecisionDecisionRule,
    BCEPrecisionPolicy,
    DirectCandidateIncrementDecisionRule,
    V0MarginalConnectionDecisionRule,
    trace_bce_precision_frontier,
)
from mechanistic_systems.inference.sequential_bce import (
    IID_WITH_REPLACEMENT_SAMPLING_ID,
    LossSpanCertificate,
)
from .evidence_continuous import (
    EVIDENCE_CONTINUOUS_V2,
    ContinuousEvidencePolicy,
    ContinuousEvidenceResult,
    NaturalizationContext,
    trace_continuous_evidence_frontier,
)
from .masking import (
    build_historical_partitioned_candidate_lowerer,
    compose_committed_masks,
    historical_density_axis,
    lower_historical_partitioned_masks,
)
from .sampled_path import sampled_path_mask_hash

NATIVE_CONTINUOUS_SIGNATURE_SCHEMA = "native_continuous_trace_signature_v1"

# Fallback look granularity when the backend exposes no batch-size fact. The
# look batch only sets the evidence cadence (how often spending is checked),
# never validity, so a conservative constant is safe.
DEFAULT_CONTINUOUS_LOOK_BATCH = 256


@dataclass(frozen=True)
class BackendNaturalizationFacts:
    """Runtime facts the continuous policy naturalizes its knobs from."""

    class_count: int
    bank_depth_per_class: int
    gpu_batch_size: int

    def validate(self) -> None:
        if self.class_count < 2:
            raise ValueError("continuous execution needs at least two classes")
        if self.bank_depth_per_class < 1:
            raise ValueError("continuous execution needs a positive bank depth")
        if self.gpu_batch_size < 1:
            raise ValueError("continuous execution needs a positive batch size")


@dataclass(frozen=True)
class NativeContinuousTraceSignature:
    profile_hash: str
    query_id: str
    layer_id: str
    final_positive_density: float
    final_negative_density: float
    final_mask_hash: str
    decision_count: int
    images_consumed: int
    stopping_reason: str
    delta_budget: float
    delta_spent: float
    algorithm_version: str = EVIDENCE_CONTINUOUS_V2
    schema_name: str = NATIVE_CONTINUOUS_SIGNATURE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @property
    def content_hash(self) -> str:
        return stable_digest(self.to_dict())


@dataclass(frozen=True)
class NativeContinuousRun:
    """One continuous layer-group run, chain-compatible with the panel loop."""

    result: ContinuousEvidenceResult
    signature: NativeContinuousTraceSignature
    maturity: str
    analysis_status: str
    parity_receipt_hash: str | None = None
    reference_anchor_loss: float | None = None
    reference_anchor_evidence_hash: str | None = None
    reference_anchor_mask_hash: str | None = None
    reference_anchor_panel_hash: str | None = None


def naturalization_facts_from_backend(
    backend: Any,
    *,
    gpu_batch_size: int | None = None,
) -> BackendNaturalizationFacts:
    """Read class axis, bank depth, and batch size off a concrete backend."""

    source = getattr(backend, "panel_source", None)
    records_by_label = getattr(source, "records_by_label", None)
    if records_by_label is None:
        axis = getattr(source, "context_axis", None)
        records = getattr(axis, "records", None)
        if records is None:
            raise ValueError(
                "continuous execution could not read the class axis from the "
                "backend's panel source; pass naturalization_facts explicitly"
            )
        grouped: dict[int, int] = {}
        for record in records:
            grouped[int(record.label)] = grouped.get(int(record.label), 0) + 1
        counts = grouped
    else:
        counts = {
            int(label): len(records)
            for label, records in records_by_label.items()
        }
    if not counts:
        raise ValueError("continuous execution needs a non-empty context axis")
    resolved_batch = gpu_batch_size
    if resolved_batch is None:
        evaluator = getattr(backend, "evaluator", None)
        for attribute in (
            "saturating_batch_size",
            "execution_batch_size",
            "batch_size",
        ):
            value = getattr(evaluator, attribute, None)
            if value:
                resolved_batch = int(value)
                break
    if resolved_batch is None:
        resolved_batch = DEFAULT_CONTINUOUS_LOOK_BATCH
    facts = BackendNaturalizationFacts(
        class_count=len(counts),
        bank_depth_per_class=int(min(counts.values())),
        gpu_batch_size=int(resolved_batch),
    )
    facts.validate()
    return facts


def continuous_group_density_floor(route_group_kind: str, profile: Any) -> float:
    """The historical schedule's smallest positive density for this group kind.

    The historical axis terminates in an exact-zero rung (full prune); the
    continuous bisection's ``minimum_density`` is an open lower bound in
    (0, 1), so the floor is the terminal *positive* rung — the profile's
    target-preservation final density region.
    """

    schedule = historical_density_axis(
        route_group_kind,
        conv_decay=float(profile.target_preservation_conv_decay),
        conv_target_final_density=float(
            profile.target_preservation_conv_target_final_density
        ),
        linear_decay=float(profile.target_preservation_linear_decay),
        linear_target_final_density=float(
            profile.target_preservation_linear_target_final_density
        ),
    )
    positive = [float(value) for value in schedule if float(value) > 0.0]
    if not positive:
        raise ValueError(
            f"historical schedule for {route_group_kind!r} has no positive rung"
        )
    return min(positive)


def frontier_prior_from_history(
    history: Mapping[str, list[float]] | None,
) -> dict[str, float] | None:
    """Mean accepted density per partition across finished groups/queries."""

    if not history:
        return None
    prior = {
        str(partition): float(np.mean(values))
        for partition, values in history.items()
        if values
    }
    return prior or None


def record_frontier_history(
    history: dict[str, list[float]],
    result: ContinuousEvidenceResult,
) -> None:
    history.setdefault("positive", []).append(float(result.positive_density))
    history.setdefault("negative", []).append(float(result.negative_density))


def run_profiled_native_continuous_group(
    *,
    backend: Any,
    profile: Any,
    scores: Mapping[str, Any],
    partition_labels: Mapping[str, Any],
    query_id: str,
    layer_id: str,
    route_group_kind: str,
    dense_anchor_loss: float,
    previous_group_masks: Mapping[str, Any] | None = None,
    frontier_prior: Mapping[str, float] | None = None,
    image_budget: float | None = None,
    claim_bearing: bool = False,
    naturalization_facts: BackendNaturalizationFacts | None = None,
    gpu_batch_size: int | None = None,
    reference_anchor_loss: float | None = None,
    reference_anchor_evidence_hash: str | None = None,
    reference_anchor_mask_hash: str | None = None,
    reference_anchor_panel_hash: str | None = None,
) -> NativeContinuousRun:
    """Run one layer-group turn through the continuous canonical engine."""

    if claim_bearing:
        raise ValueError(
            "claim-bearing execution is contractually pinned to "
            "statistical_sequential_symmetric_frontier; the continuous engine "
            "has no promotion boundary yet"
        )
    profile.validate()
    minimum_density = continuous_group_density_floor(route_group_kind, profile)
    dense_group = lower_historical_partitioned_masks(
        scores,
        partition_labels,
        positive_density=1.0,
        negative_density=1.0,
    )
    committed = compose_committed_masks(dense_group.masks, previous_group_masks)
    build_candidate = build_historical_partitioned_candidate_lowerer(
        scores,
        partition_labels,
        previous_group_masks=previous_group_masks,
    )
    labels = np.concatenate(
        [
            np.asarray(value).reshape(-1)
            for value in partition_labels.values()
        ]
    )
    # Canonical lowering defines historical label 0 as the positive
    # partition and label 1 as the negative partition.  Keep that polarity in
    # the controller's route ledger so density proposals and lowered masks
    # name the same physical routes.
    positive_routes = int(np.count_nonzero(labels == 0))
    negative_routes = int(np.count_nonzero(labels == 1))
    # Some layer groups are structurally single-partition (uniform polarity
    # labels); the empty partition is excluded from bisection entirely.
    partition_counts = {
        name: count
        for name, count in (
            ("positive", positive_routes),
            ("negative", negative_routes),
        )
        if count > 0
    }
    if not partition_counts:
        raise ValueError("continuous execution needs at least one route")
    active_partitions = tuple(partition_counts)
    facts = naturalization_facts or naturalization_facts_from_backend(
        backend, gpu_batch_size=gpu_batch_size
    )
    facts.validate()
    total_routes = int(dense_group.total_route_count)
    tau = float(profile.sensitivity_tau)
    delta_budget = tau * (1.0 - minimum_density) * float(total_routes)
    policy = ContinuousEvidencePolicy(
        distortion_budget=delta_budget,
        image_budget=(None if image_budget is None else float(image_budget)),
        voi_images_per_route=None,
        futility_horizon_images=None,
        look_batch_images=None,
        initial_density_step=None,
        minimum_density_step=None,
        minimum_density=minimum_density,
        random_seed=int(profile.statistical_sampling_seed),
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
    resolved = policy.resolve(context)
    result = trace_continuous_evidence_frontier(
        backend=backend,
        committed_masks=committed,
        build_candidate=build_candidate,
        query_id=str(query_id),
        layer_id=str(layer_id),
        total_layer_route_count=total_routes,
        dense_anchor_loss=float(dense_anchor_loss),
        fully_pruned_anchor_loss=float(profile.fully_pruned_anchor_loss),
        policy=resolved,
        class_count=int(facts.class_count),
        initial_turn=(
            str(profile.initial_partition)
            if str(profile.initial_partition) in active_partitions
            else active_partitions[0]
        ),
        active_partitions=active_partitions,
    )
    signature = NativeContinuousTraceSignature(
        profile_hash=str(profile.content_hash),
        query_id=str(query_id),
        layer_id=str(layer_id),
        final_positive_density=float(result.positive_density),
        final_negative_density=float(result.negative_density),
        final_mask_hash=sampled_path_mask_hash(result.committed_masks),
        decision_count=len(result.decisions),
        images_consumed=int(result.images_consumed),
        stopping_reason=str(result.stopping_reason),
        delta_budget=float(result.delta_budget),
        delta_spent=float(result.delta_spent),
    )
    return NativeContinuousRun(
        result=result,
        signature=signature,
        maturity="continuous_canonical",
        analysis_status="eligible",
        reference_anchor_loss=(
            None if reference_anchor_loss is None else float(reference_anchor_loss)
        ),
        reference_anchor_evidence_hash=reference_anchor_evidence_hash,
        reference_anchor_mask_hash=reference_anchor_mask_hash,
        reference_anchor_panel_hash=reference_anchor_panel_hash,
    )


def run_profiled_native_bce_precision_group(
    *,
    backend: Any,
    profile: Any,
    scores: Mapping[str, Any],
    partition_labels: Mapping[str, Any],
    query_id: str,
    layer_id: str,
    route_group_kind: str,
    previous_group_masks: Mapping[str, Any] | None = None,
    frontier_prior: Mapping[str, float] | None = None,
    image_budget: float | None = None,
    claim_bearing: bool = False,
    naturalization_facts: BackendNaturalizationFacts | None = None,
    gpu_batch_size: int | None = None,
    reference_anchor_loss: float | None = None,
    reference_anchor_evidence_hash: str | None = None,
    reference_anchor_mask_hash: str | None = None,
    reference_anchor_panel_hash: str | None = None,
    maximum_accepted_decisions: int | None = None,
    decision_rule: (
        BCEPrecisionDecisionRule
        | V0MarginalConnectionDecisionRule
        | BCELossBoundaryDecisionRule
        | DirectCandidateIncrementDecisionRule
        | None
    ) = None,
    loss_span_certificate: LossSpanCertificate | None = None,
    route_semantics_by_layer: Mapping[str, str] | None = None,
    runtime_deadline_monotonic: float | None = None,
    progress_callback: Any | None = None,
    step_context_delta_sink: Any | None = None,
    accept_replication_gate: bool = False,
    oracle_sampling_law: str = IID_WITH_REPLACEMENT_SAMPLING_ID,
    oracle_acquisition_envelope_seconds: float | None = None,
    decision_law_id: str = "sequential_calibrated_bce_resolution_v5",
) -> NativeContinuousRun:
    """Run adaptive v0 under raw paired-BCE threshold precision.

    The historical score ordering, sign partitions, reverse layer-group chain,
    and accept-expand/reject-bisect dynamics are retained.  Only the
    candidate-adjudication rule changes to the two-input BCE precision
    contract.

    ``oracle_sampling_law`` and ``oracle_acquisition_envelope_seconds`` are
    copied verbatim from ``IndependentSampleSizePolicy`` by the caller; the
    defaults reproduce the pre-v4 policy exactly.
    """

    if claim_bearing:
        raise ValueError(
            "raw-BCE precision is exploratory because BCE is unbounded; "
            "select a promoted bounded mode for claim-bearing execution"
        )
    profile.validate()
    minimum_density = continuous_group_density_floor(route_group_kind, profile)
    build_candidate = build_historical_partitioned_candidate_lowerer(
        scores,
        partition_labels,
        previous_group_masks=previous_group_masks,
    )
    dense_group = build_candidate.dense_group()
    committed = compose_committed_masks(dense_group.masks, previous_group_masks)
    labels = np.concatenate(
        [np.asarray(value).reshape(-1) for value in partition_labels.values()]
    )
    partition_counts = {
        name: count
        for name, count in (
            ("positive", int(np.count_nonzero(labels == 0))),
            ("negative", int(np.count_nonzero(labels == 1))),
        )
        if count > 0
    }
    if not partition_counts:
        raise ValueError("BCE precision execution needs at least one route")
    active_partitions = tuple(partition_counts)
    facts = naturalization_facts or naturalization_facts_from_backend(
        backend,
        gpu_batch_size=gpu_batch_size,
    )
    facts.validate()
    confidence = float(
        getattr(
            profile,
            "bce_precision_confidence_level",
            getattr(profile, "posterior_accept_probability", 0.95),
        )
    )
    resolved_decision_rule = decision_rule or BCEPrecisionDecisionRule(
        confidence_level=confidence,
        sensitivity_tau=float(profile.sensitivity_tau),
    )
    resolved_decision_rule.validate()
    policy = BCEPrecisionPolicy(
        rule=resolved_decision_rule,
        image_budget=(None if image_budget is None else float(image_budget)),
        fold_count=int(profile.crossfit_fold_count),
        minimum_density=minimum_density,
        # Modern-v0 calibration and oracle rows are logical IID draws with
        # replacement. A profile-level finite-population preference belongs
        # only to sealed without-replacement evidence identities.
        finite_population_correction=(
            False
            if isinstance(
                resolved_decision_rule,
                (V0MarginalConnectionDecisionRule, DirectCandidateIncrementDecisionRule),
            )
            else bool(profile.finite_population_correction)
        ),
        random_seed=int(profile.statistical_sampling_seed),
        # Modern v0 has exactly two scientific inputs: confidence and tau.
        # IID calibration/oracle acquisition therefore has no image, look, or
        # wall-clock stopping horizon.  Runtime remains measured, but cannot
        # change a pruning decision.
        maximum_planned_looks=(
            None
            if isinstance(
                resolved_decision_rule,
                (V0MarginalConnectionDecisionRule, DirectCandidateIncrementDecisionRule),
            )
            else 32
        ),
        precision_only_iid=isinstance(
            resolved_decision_rule,
            (V0MarginalConnectionDecisionRule, DirectCandidateIncrementDecisionRule),
        ),
        runtime_deadline_monotonic=(
            None
            if isinstance(
                resolved_decision_rule,
                (V0MarginalConnectionDecisionRule, DirectCandidateIncrementDecisionRule),
            )
            else runtime_deadline_monotonic
        ),
        accept_replication_gate=bool(accept_replication_gate)
        and isinstance(
            resolved_decision_rule,
            V0MarginalConnectionDecisionRule,
        ),
        # Phase-1 v4 decision law: the paired-oracle sampling law and the
        # operational acquisition envelope arrive from the sample-size policy
        # (``IndependentSampleSizePolicy.oracle_sampling_law`` /
        # ``.oracle_acquisition_envelope_seconds``).  The frontier refuses a
        # backend whose ``oracle_sampling_id`` disagrees with the law.
        oracle_sampling_law=str(oracle_sampling_law),
        decision_law_id=(
            str(decision_law_id)
            if isinstance(resolved_decision_rule, V0MarginalConnectionDecisionRule)
            else "sequential_calibrated_bce_resolution_v4"
        ),
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
    executable_density_schedule = None
    if isinstance(
        resolved_decision_rule,
        (V0MarginalConnectionDecisionRule, DirectCandidateIncrementDecisionRule),
    ):
        # Modern v0 keeps the historical layerwise/intermixed density
        # topology as its comparison surface, but replaces its noisy
        # fixed-panel verdict with confidence-bearing paired BCE.  The first
        # unsafe or unresolved rung retains the last accepted parent and
        # freezes that partition; it never launches a route-resolution
        # bisection or commits an ambiguous mask.
        executable_density_schedule = tuple(
            value
            for value in historical_density_axis(
                route_group_kind,
                conv_decay=float(profile.target_preservation_conv_decay),
                conv_target_final_density=float(
                    profile.target_preservation_conv_target_final_density
                ),
                linear_decay=float(profile.target_preservation_linear_decay),
                linear_target_final_density=float(
                    profile.target_preservation_linear_target_final_density
                ),
            )
            if float(value) >= float(minimum_density) - 1e-12
            and float(value) > 0.0
        )
    result = trace_bce_precision_frontier(
        backend=backend,
        committed_masks=committed,
        build_candidate=build_candidate,
        query_id=str(query_id),
        layer_id=str(layer_id),
        total_layer_route_count=int(dense_group.total_route_count),
        policy=policy.resolve(context),
        class_count=int(facts.class_count),
        initial_turn=(
            str(profile.initial_partition)
            if str(profile.initial_partition) in active_partitions
            else active_partitions[0]
        ),
        active_partitions=active_partitions,
        executable_density_schedule=executable_density_schedule,
        maximum_accepted_decisions=maximum_accepted_decisions,
        progress_callback=progress_callback,
        step_context_delta_sink=step_context_delta_sink,
        loss_span_certificate=loss_span_certificate,
        route_semantics_by_layer=route_semantics_by_layer,
    )
    signature = NativeContinuousTraceSignature(
        profile_hash=str(profile.content_hash),
        query_id=str(query_id),
        layer_id=str(layer_id),
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
    return NativeContinuousRun(
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


def continuous_frontier_rows(
    query_id: str,
    layer_id: str,
    group_index: int,
    run: NativeContinuousRun,
) -> list[dict[str, Any]]:
    """Materialization rows for one continuous group, one row per decision."""

    rows: list[dict[str, Any]] = []
    for index, decision in enumerate(run.result.decisions):
        rows.append(
            {
                "query_id": str(query_id),
                "layer_id": str(layer_id),
                "group_index": int(group_index),
                "proposal_index": int(index),
                "partition_id": str(decision.partition_id),
                "previous_density": float(decision.previous_density),
                "proposed_density": float(decision.proposed_density),
                "accepted": decision.action == "accept",
                "action": str(decision.action),
                "reason": str(decision.reason),
                "looks": int(decision.looks),
                "panel_images": int(decision.panel_images),
                "mean_sensitivity": float(decision.mean_sensitivity),
                "certified_lower_bound": float(
                    decision.certified_lower_bound
                ),
                "certified_upper_bound": float(decision.certified_upper_bound),
                "threshold": float(decision.threshold),
                "confidence_level": decision.confidence_level,
                "interval_method": decision.interval_method,
                "delta_debit": float(decision.delta_debit),
                "alpha_allocated": float(decision.alpha_allocated),
                "group_reference_anchor_loss": (
                    None
                    if run.reference_anchor_loss is None
                    else float(run.reference_anchor_loss)
                ),
                "group_reference_anchor_evidence_hash": (
                    run.reference_anchor_evidence_hash or ""
                ),
            }
        )
    return rows


__all__ = [
    "BackendNaturalizationFacts",
    "DEFAULT_CONTINUOUS_LOOK_BATCH",
    "NATIVE_CONTINUOUS_SIGNATURE_SCHEMA",
    "NativeContinuousRun",
    "NativeContinuousTraceSignature",
    "continuous_frontier_rows",
    "continuous_group_density_floor",
    "frontier_prior_from_history",
    "naturalization_facts_from_backend",
    "record_frontier_history",
    "run_profiled_native_bce_precision_group",
    "run_profiled_native_continuous_group",
]
