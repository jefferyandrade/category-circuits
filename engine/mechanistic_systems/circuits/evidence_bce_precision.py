"""Sequential calibrated-BCE frontier used by modern v0.

The shared measurement foundation lives in :mod:`.sequential_bce`. This
module supplies the modern-v0 adapter:

* fit and freeze state-specific affine BCE calibration;
* execute exactly the parent and one candidate on paired logical IID draws;
* project unequal on/off-target Neyman panels with a fixed-weight,
  Satterthwaite Student-t interval;
* test the paired increment against tau times the positive dense-to-empty
  span and exact connection count, and report its Fieller ratio set;
* accept only below ``sensitivity_tau``; reject/bisect only above it; and
  never commit unresolved evidence.

The prior aggregate-increment and per-removed-route policies remain explicit
compatibility identities. v1--v4 use the candidate-only dense-baseline adapter
in :mod:`mechanistic_systems.circuits.irg.runner`, not this paired frontier.

Raw BCE is unbounded. Predictable alpha-spent Student-t intervals are therefore
assumption-dependent exploratory inference, not distribution-free confidence
sequences or a campaign-global safety theorem.
"""

from __future__ import annotations

import math
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from mechanistic_systems.evidence import student_t_interval
from mechanistic_systems.contracts.bce_evidence import (
    CalibrationPrecisionHorizonExhausted,
    FixedCalibrationPairedBCEEvidence,
    OracleFinitePopulationCensusExhausted,
)
from mechanistic_systems.contracts.sampled_evidence import PANEL_AXIS_CHAIN_KEY
from mechanistic_systems.measurement.host_preplanning import host_preplanning_enabled, plan_host_branches

from .evidence_continuous import (
    ContinuousDecision,
    ContinuousEvidenceResult,
    NaturalizationContext,
    _mask_hamming_distance,
)
from .evidence_sampled_path import _weighted_sensitivity_moments
from .frontier_rungs import (
    maximum_decisions_from_retained_counts,
    retained_counts_for_density_schedule,
)
from .glocal_sampled_path import (
    GlocalSampledComparisonRequest,
    GlocalSampledPathEvaluationBackend,
)
from .sampled_path import CrossfitPanel, fold_warm_start, sampled_path_mask_hash
from .step_observables import step_observables_enabled
from .v6_records import (
    FLAG_ACCEPT_REPLICATION,
    FLAG_FORCED,
    FLAG_SUPPRESSED,
    V6_DECISION_REF_SCHEMA,
    V6_LEDGER_SCHEMA,
    V6_PANEL_RECEIPT_SCHEMA,
    ForcedAction,
    TraceWalkLog,
    TraceWalkLogHandle,
    decision_key,
    forced_actions_from_env,
)
from mechanistic_systems.inference.sequential_bce import (
    AnalyticChanceRiskCertificate,
    IID_WITH_REPLACEMENT_SAMPLING_ID,
    JOINT_DECISION_WALD_METHOD,
    ORACLE_SAMPLING_LAWS,
    PLANNED_LOOK_SCOPE,
    PRECISION_ONLY_IID_SCOPE,
    SEALED_WOR_CLASS_STRATIFIED_SAMPLING_ID,
    SEQUENTIAL_BCE_DECISION_RECEIPT_V4,
    SEQUENTIAL_BCE_DECISION_RECEIPT_V5,
    SEQUENTIAL_CALIBRATED_BCE_V3,
    SEQUENTIAL_CALIBRATED_BCE_V4,
    SEQUENTIAL_CALIBRATED_BCE_V5,
    SCALAR_DECISION_FIELLER_METHOD,
    ScalarV0DecisionEnvelope,
    STRATIFIED_T_ALPHA_SPENDING_METHOD,
    STRATIFIED_T_FPC_ALPHA_SPENDING_METHOD,
    STRATIFIED_T_FPC_METHOD,
    STRATIFIED_T_METHOD,
    V0_NORMALIZED_LOSS_SPAN_PER_CONNECTION_V1,
    CalibrationImprovementProfile,
    CircuitConnectionDeltaReceipt,
    ConnectionDelta,
    JointDecisionConfidenceEnvelope,
    LossSpanCertificate,
    OptimizedCalibrationDecisionEnvelope,
    RunningMoment,
    SequentialCalibratedBCEResolutionSpec,
    SequentialBCEDecisionReceipt,
    conservative_ratio_interval,
    forecast_cumulative_target,
    guard_unvalidated_cumulative_forecast,
    joint_v0_decision_interval,
    scalar_v0_decision_interval,
    stratified_student_t_summary,
    stratified_student_t_summary_from_moment_arrays,
    stratified_student_t_summary_from_moments,
)
from mechanistic_systems.measurement import evidence_plane_v6 as _v6sb

BCE_PRECISION_V1 = "evidence_foundation_bce_precision_v1"
BCE_PRECISION_INTERVAL_METHOD = (
    "paired_crossfit_bce_student_t_bonferroni_planned_looks_v1"
)
BCE_PRECISION_BOUNDED_INTERVAL_METHOD = (
    "stratified_bounded_bce_empirical_bernstein_" "bonferroni_planned_looks_v1"
)

from mechanistic_systems.contracts.bce_evidence import (
    ORACLE_METADATA_SAMPLING_WITH_REPLACEMENT,
    ORACLE_METADATA_FPC_APPLIED,
    ORACLE_METADATA_SAMPLING_LAW,
    ORACLE_METADATA_POPULATION_BY_STRATUM,
    ORACLE_METADATA_DRAWN_BY_STRATUM,
    ORACLE_METADATA_EXHAUSTED_STRATA,
    ORACLE_METADATA_REMAINING_POPULATION,
    ORACLE_METADATA_CENSUS_TERMINAL_REACHED,
    ORACLE_METADATA_CENSUS_COMPLETE,
    ORACLE_METADATA_CENSUS_STRATA_COUNT,
    ORACLE_METADATA_CENSUS_EXACT_WEIGHT,
    ORACLE_METADATA_PANEL_AXIS_CHAIN,
    ORACLE_METADATA_KEYS,
)

ORACLE_ACQUISITION_ENVELOPE_RECEIPT_V1 = "oracle_acquisition_envelope_receipt_v1"
# Forced-action hook. The acceptance harness installs this object.
FORCED_ACTIONS_HOOK_VERSION = "v6acc_forced_actions_hook_v1"
FORCED_ACTIONS_HOOK: Any | None = None


def _v6_forced_hook() -> Any | None:
    hook = FORCED_ACTIONS_HOOK
    if hook is None:
        return None
    version = getattr(hook, "FORCED_ACTIONS_HOOK_VERSION", getattr(hook, "version", None))
    if version != FORCED_ACTIONS_HOOK_VERSION:
        raise RuntimeError("forced-action hook has no matching version")
    for method in ("look", "exit", "replication"):
        if not callable(getattr(hook, method, None)):
            raise RuntimeError(f"forced-action hook lacks {method}")
    return hook


def _v6_hook_result(result: Any, *, natural: Mapping[str, Any], point: str) -> tuple[str, str, int | None]:
    if not isinstance(result, Mapping):
        raise RuntimeError(f"forced-action {point} returned no mapping")
    outcome = str(result.get("outcome", result.get("action", natural["outcome"])))
    if outcome not in {"accept", "reject", "conservative_no_commit", "unresolved"}:
        raise RuntimeError(f"forced-action {point} returned invalid outcome {outcome!r}")
    reason = str(result.get("reason", natural["reason"]))
    target_value = result.get("next_cumulative_target", natural.get("next_cumulative_target"))
    target = None if target_value is None else int(target_value)
    if point == "look" and outcome == "unresolved" and (target is None or target <= int(natural.get("actual_images", 0))):
        raise RuntimeError(f"forced-action {point} returned invalid next target")
    return outcome, reason, target


class _V6DecisionLogHook:
    """Versioned compatibility provider for the lane's decision-only logs.

    It waits for the engine's terminal statistical look, so the existing
    decision-log replay still acquires every natural look. A per-look
    reference provider can replace it through ``FORCED_ACTIONS_HOOK``.
    """

    FORCED_ACTIONS_HOOK_VERSION = FORCED_ACTIONS_HOOK_VERSION

    def __init__(self, forced: ForcedAction) -> None:
        self.forced = forced

    def look(self, query_id, layer_id, partition_id, proposal_index, look_index, natural):
        if natural["outcome"] == "unresolved":
            return dict(natural)
        return {"outcome": self.forced.action,
                "reason": self.forced.reason or natural["reason"],
                "next_cumulative_target": natural["next_cumulative_target"],
                "terminal": True}

    def exit(self, query_id, layer_id, partition_id, proposal_index, point, completed_looks, natural):
        return {"action": self.forced.action,
                "reason": self.forced.reason or natural["reason"]}

    def replication(self, query_id, layer_id, partition_id, proposal_index, completed_looks, natural):
        return {"replicated": natural["replicated"]}
BCE_PRECISION_DECISION_LAW_V1 = "bce_precision_decision_law_v1"
CENSUS_TERMINAL_NO_COMMIT_REASON = (
    "census_terminal_interval_straddles_boundary_no_commit"
)
CENSUS_EXHAUSTED_UNRESOLVED_REASON = "finite_population_census_exhausted_unresolved"
ORACLE_ENVELOPE_UNRESOLVED_REASON = (
    "oracle_acquisition_envelope_exhausted_unresolved"
)
# Terminal vocabulary of ``_decide_bce_precision_step`` by decision law.  The
# verdict ladder (accept / reject / v0 point-estimate no-commit) and the
# forecast escapes are shared; the sealed without-replacement law adds the
# census terminals and an armed envelope adds its typed refusal.
_LADDER_TERMINAL_REASONS = (
    "confidence_interval_wholly_at_or_below_tau",
    "confidence_interval_wholly_at_or_below_boundary",
    "confidence_interval_wholly_above_tau",
    "confidence_interval_wholly_above_boundary",
    "point_estimate_at_or_above_boundary_no_commit",
    "panel_growth_exhausted_unresolved",
    "predicted_sample_horizon_exhausted_unresolved",
    "calibration_precision_horizon_exhausted_no_commit",
    "accept_failed_confirm_replication",
    "evidence_horizon_exhausted_unresolved",
    "image_budget_exhausted_unresolved",
    "image_budget_minimum_packet_unavailable",
    "planned_look_horizon_exhausted_unresolved",
)
_SEALED_WOR_TERMINAL_REASONS = (
    CENSUS_TERMINAL_NO_COMMIT_REASON,
    CENSUS_EXHAUSTED_UNRESOLVED_REASON,
)
V0_DECISION_TERMINAL_REASONS: Mapping[str, tuple[str, ...]] = {
    IID_WITH_REPLACEMENT_SAMPLING_ID: _LADDER_TERMINAL_REASONS,
    SEALED_WOR_CLASS_STRATIFIED_SAMPLING_ID: (
        _LADDER_TERMINAL_REASONS + _SEALED_WOR_TERMINAL_REASONS
    ),
}


def _string_tuple(values: Any) -> tuple[str, ...]:
    """Return ``values`` as ``tuple[str, ...]`` without copying when it already is.

    The panel axis can hold millions of ids; when the container is already an
    exact ``tuple`` of exact ``str`` (the ``CrossfitPanel`` contract) the same
    object is returned, otherwise a ``tuple(str(v) ...)`` copy is
    made.  Values are identical either way (``str(s) is s`` for exact str).
    """

    if type(values) is tuple and set(map(type, values)) <= {str}:
        return values
    return tuple(str(value) for value in values)


@dataclass(frozen=True)
class _OracleCensusState:
    """Per-look view of the provider's finite-population disclosure.

    ``legacy()`` is the state of a panel without the contract keys: no
    populations, no census, unbounded remaining population.
    """

    sampling_with_replacement: bool
    sampling_law: str | None
    population_by_stratum: Mapping[str, int] | None
    remaining_population: int | None
    census_terminal_reached: bool
    census_complete: bool
    census_strata_count: int
    census_exact_weight: float
    exhausted_strata: tuple[str, ...]
    contract_present: bool

    @classmethod
    def legacy(cls, *, sampling_with_replacement: bool = True) -> "_OracleCensusState":
        return cls(
            sampling_with_replacement=bool(sampling_with_replacement),
            sampling_law=None,
            population_by_stratum=None,
            remaining_population=None,
            census_terminal_reached=False,
            census_complete=False,
            census_strata_count=0,
            census_exact_weight=0.0,
            exhausted_strata=(),
            contract_present=False,
        )

    def receipt_fields(self) -> dict[str, Any]:
        """Disclosure keys for panel receipts; empty for legacy panels."""

        if not self.contract_present:
            return {}
        return {
            "oracle_sampling_law": self.sampling_law,
            "oracle_census_terminal_reached": bool(self.census_terminal_reached),
            "oracle_census_complete": bool(self.census_complete),
            "oracle_census_strata_count": int(self.census_strata_count),
            "oracle_census_exact_weight": float(self.census_exact_weight),
            "oracle_remaining_population": self.remaining_population,
            "oracle_exhausted_strata_count": len(self.exhausted_strata),
        }


def _oracle_census_state(
    metadata: Mapping[str, Any],
    *,
    sampling_with_replacement: bool,
) -> _OracleCensusState:
    """Parse the provider contract from one panel's ``allocation_metadata``."""

    contract_present = any(
        key in metadata
        for key in (
            ORACLE_METADATA_SAMPLING_LAW,
            ORACLE_METADATA_POPULATION_BY_STRATUM,
            ORACLE_METADATA_CENSUS_COMPLETE,
            ORACLE_METADATA_CENSUS_TERMINAL_REACHED,
            ORACLE_METADATA_REMAINING_POPULATION,
        )
    )
    if not contract_present:
        return _OracleCensusState.legacy(
            sampling_with_replacement=sampling_with_replacement
        )
    raw_populations = metadata.get(ORACLE_METADATA_POPULATION_BY_STRATUM)
    populations = (
        None
        if sampling_with_replacement or not isinstance(raw_populations, Mapping)
        else {str(key): int(value) for key, value in raw_populations.items()}
    )
    raw_remaining = metadata.get(ORACLE_METADATA_REMAINING_POPULATION)
    remaining = None if raw_remaining is None else int(raw_remaining)
    if remaining is not None and remaining < 0:
        raise ValueError("oracle_remaining_population must be nonnegative")
    exact_weight = float(metadata.get(ORACLE_METADATA_CENSUS_EXACT_WEIGHT, 0.0))
    if not math.isfinite(exact_weight) or not 0.0 <= exact_weight <= 1.0 + 1e-9:
        raise ValueError("oracle_census_exact_weight must lie in [0, 1]")
    census_complete = bool(metadata.get(ORACLE_METADATA_CENSUS_COMPLETE, False))
    terminal = bool(metadata.get(ORACLE_METADATA_CENSUS_TERMINAL_REACHED, False))
    raw_law = metadata.get(ORACLE_METADATA_SAMPLING_LAW)
    return _OracleCensusState(
        sampling_with_replacement=bool(sampling_with_replacement),
        sampling_law=None if raw_law is None else str(raw_law),
        population_by_stratum=populations,
        remaining_population=remaining,
        census_terminal_reached=terminal,
        census_complete=census_complete,
        census_strata_count=int(metadata.get(ORACLE_METADATA_CENSUS_STRATA_COUNT, 0)),
        census_exact_weight=exact_weight,
        exhausted_strata=tuple(
            str(value) for value in metadata.get(ORACLE_METADATA_EXHAUSTED_STRATA, ())
        ),
        contract_present=True,
    )


def _oracle_acquisition_envelope_exceeded(
    *,
    elapsed_seconds: float,
    envelope_seconds: float | None,
    look_seconds_per_row: Any,
    requested_increment: int,
) -> dict[str, float] | None:
    """Decline the next oracle look when the envelope cannot cover it.

    Mirror of the pilot's response-field predicate: affordability is checked
    BEFORE the purchase, the per-row cost is the MEDIAN over completed looks
    (one slow look cannot end an acquisition), and ``None`` is returned until
    at least one look has been measured, so the first look is never refused
    on a guess.  Operational disclosure only; it enters no statistic.
    """

    if envelope_seconds is None or not len(look_seconds_per_row):
        return None
    increment = max(int(requested_increment), 0)
    predicted_rate = float(
        np.median(np.asarray(look_seconds_per_row, dtype=np.float64))
    )
    predicted_seconds = predicted_rate * float(increment)
    if float(elapsed_seconds) + predicted_seconds <= float(envelope_seconds):
        return None
    return {
        "elapsed_seconds": float(elapsed_seconds),
        "envelope_seconds": float(envelope_seconds),
        "predicted_next_look_seconds": predicted_seconds,
        "predicted_seconds_per_row": predicted_rate,
        "declined_increment": int(increment),
    }


def _resolved_interval_method(
    policy: "BCEPrecisionPolicy",
    *,
    modern_v0_mode: bool,
    direct_candidate_mode: bool,
) -> str:
    if not (modern_v0_mode or direct_candidate_mode):
        return BCE_PRECISION_INTERVAL_METHOD
    if modern_v0_mode and policy.resolved_decision_law_id == SEQUENTIAL_CALIBRATED_BCE_V5:
        return SCALAR_DECISION_FIELLER_METHOD
    sealed = (
        str(policy.oracle_sampling_law) == SEALED_WOR_CLASS_STRATIFIED_SAMPLING_ID
    )
    if policy.precision_only_iid:
        return (
            STRATIFIED_T_FPC_ALPHA_SPENDING_METHOD
            if sealed
            else STRATIFIED_T_ALPHA_SPENDING_METHOD
        )
    return STRATIFIED_T_FPC_METHOD if sealed else STRATIFIED_T_METHOD


def _decision_law_block(policy: "BCEPrecisionPolicy") -> dict[str, Any]:
    law = str(policy.oracle_sampling_law)
    envelope = policy.oracle_acquisition_envelope_seconds
    reasons = tuple(V0_DECISION_TERMINAL_REASONS[law])
    if envelope is not None:
        reasons = reasons + (ORACLE_ENVELOPE_UNRESOLVED_REASON,)
    block = {
        "schema": BCE_PRECISION_DECISION_LAW_V1,
        "oracle_sampling_law": law,
        "oracle_acquisition_envelope_seconds": (
            None if envelope is None else float(envelope)
        ),
        "terminal_reasons": reasons,
    }
    if policy.resolved_decision_law_id == SEQUENTIAL_CALIBRATED_BCE_V5:
        block["decision_law_id"] = SEQUENTIAL_CALIBRATED_BCE_V5
        block["decision_interval_method"] = SCALAR_DECISION_FIELLER_METHOD
        block["v5_sidedness"] = policy.v5_sidedness
    return block


def _v0_joint_interval(*, policy: "BCEPrecisionPolicy", threshold: float,
                       **inputs: Any) -> JointDecisionConfidenceEnvelope | Any:
    if policy.resolved_decision_law_id == SEQUENTIAL_CALIBRATED_BCE_V4:
        return joint_v0_decision_interval(**inputs)
    return scalar_v0_decision_interval(
        threshold=float(threshold), v5_sidedness=policy.v5_sidedness, **inputs
    )


def _v5_anchor_inputs(span: LossSpanCertificate) -> dict[str, float]:
    anchor = span.fully_pruned
    if isinstance(anchor, AnalyticChanceRiskCertificate):
        return {"fully_pruned_standard_error": 0.0, "fully_pruned_bias_bound": 0.0}
    return {
        "fully_pruned_standard_error": float(anchor.standard_error),
        "fully_pruned_bias_bound": max(
            float(anchor.estimate) - float(anchor.lower),
            float(anchor.upper) - float(anchor.estimate),
        ),
    }


def _decision_resource_usage(backend: Any) -> Mapping[str, Any]:
    """Return one bounded decision delta instead of a cumulative ledger copy.

    Modern-v0 providers expose a content-addressed checkpoint/delta surface.
    Historical and third-party backends retain the legacy mapping fallback so
    replay remains parseable, but production receipts do not embed the
    provider's ever-growing physical-execution history once per decision.
    """

    delta = getattr(backend, "decision_resource_usage_delta", None)
    if callable(delta):
        value = delta()
        if not isinstance(value, Mapping):
            raise TypeError("decision resource delta must be a mapping")
        return dict(value)
    value = getattr(backend, "resource_usage", None)
    return dict(value) if isinstance(value, Mapping) else {}


def _compact_panel_allocation_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Retain inferential identities without serializing the logical axis.

    Native IID panels can contain hundreds of thousands of unique logical
    draw IDs even when repeated physical images are deduplicated.  The
    allocation receipt already binds that ordered draw stream through its
    ``draw_ids_hash`` and ``source_context_ids_hash``.  Repeating every source
    ID and every stratum label in each progress/final panel receipt adds no
    authority, can produce tens of megabytes per look, and makes receipt I/O
    part of the pruning critical path.

    The in-memory panel retains its full axis for nested-prefix validation.
    Only the persisted timing/progress view is compacted.
    """

    compact = {
        str(key): value
        for key, value in metadata.items()
        if key not in {"oracle_source_context_ids", "oracle_strata"}
    }
    strata = _string_tuple(metadata.get("oracle_strata", ()))
    source_ids = _string_tuple(metadata.get("oracle_source_context_ids", ()))
    # One linear pass; ``sorted(counts)`` reproduces the sorted
    # key order and ``Counter`` the same per-label values.
    strata_counts = Counter(strata)
    allocation = metadata.get("oracle_allocation_receipt")
    iid_receipt = (
        allocation.get("iid_stream_receipt")
        if isinstance(allocation, Mapping)
        else None
    )
    compact.update(
        {
            "oracle_strata_count": len(strata),
            "oracle_strata_counts": {
                label: strata_counts[label] for label in sorted(strata_counts)
            },
            "oracle_strata_inline": False,
            "oracle_source_context_count": len(source_ids),
            "oracle_source_context_ids_inline": False,
            "oracle_draw_ids_hash": (
                iid_receipt.get("draw_ids_hash")
                if isinstance(iid_receipt, Mapping)
                else None
            ),
            "oracle_source_context_ids_hash": (
                iid_receipt.get("source_context_ids_hash")
                if isinstance(iid_receipt, Mapping)
                else None
            ),
        }
    )
    return compact


@dataclass(frozen=True)
class BCEPrecisionDecisionRule:
    """The complete public scientific interface for one pruning decision."""

    confidence_level: float
    sensitivity_tau: float

    def validate(self) -> None:
        if not 0.5 < float(self.confidence_level) < 1.0:
            raise ValueError("confidence_level must lie in (0.5, 1)")
        if (
            not math.isfinite(float(self.sensitivity_tau))
            or float(self.sensitivity_tau) < 0.0
        ):
            raise ValueError("sensitivity_tau must be finite and nonnegative")

    @property
    def error_probability(self) -> float:
        self.validate()
        return 1.0 - float(self.confidence_level)


@dataclass(frozen=True)
class V0MarginalConnectionDecisionRule:
    """Modern v0 loss-span fraction consumed per ablated connection."""

    confidence_level: float
    sensitivity_tau: float

    def validate(self) -> None:
        if not 0.5 < float(self.confidence_level) < 1.0:
            raise ValueError("confidence_level must lie in (0.5, 1)")
        if (
            not math.isfinite(float(self.sensitivity_tau))
            or float(self.sensitivity_tau) < 0.0
        ):
            raise ValueError("sensitivity_tau must be finite and nonnegative")

    @property
    def error_probability(self) -> float:
        self.validate()
        return 1.0 - float(self.confidence_level)


@dataclass(frozen=True)
class BCELossBoundaryDecisionRule:
    """v1+ confidence rule for a complete-mask BCE loss increment."""

    confidence_level: float
    maximum_bce_increment: float

    def validate(self) -> None:
        if not 0.5 < float(self.confidence_level) < 1.0:
            raise ValueError("confidence_level must lie in (0.5, 1)")
        if (
            not math.isfinite(float(self.maximum_bce_increment))
            or float(self.maximum_bce_increment) < 0.0
        ):
            raise ValueError("maximum_bce_increment must be finite and nonnegative")

    @property
    def error_probability(self) -> float:
        self.validate()
        return 1.0 - float(self.confidence_level)


@dataclass(frozen=True)
class DirectCandidateIncrementDecisionRule:
    """Modern-v1 per-candidate genuine increment confidence rule.

    The same threshold is applied independently to every proposed candidate.
    Accepted upper endpoints are not debited from later thresholds and do not
    form a trajectory confidence bound.
    """

    confidence_level: float
    maximum_candidate_increment: float

    def validate(self) -> None:
        if not 0.5 < float(self.confidence_level) < 1.0:
            raise ValueError("confidence_level must lie in (0.5, 1)")
        if (
            not math.isfinite(float(self.maximum_candidate_increment))
            or float(self.maximum_candidate_increment) < 0.0
        ):
            raise ValueError(
                "maximum_candidate_increment must be finite and nonnegative"
            )

    @property
    def error_probability(self) -> float:
        self.validate()
        return 1.0 - float(self.confidence_level)


@dataclass(frozen=True)
class RelativeBCELossBoundary:
    """Compile a relative v1+ threshold against one locked dense baseline."""

    confidence_level: float
    maximum_relative_increase: float = 0.20

    def compile(
        self,
        *,
        locked_dense_baseline_bce: float,
    ) -> BCELossBoundaryDecisionRule:
        baseline = float(locked_dense_baseline_bce)
        relative = float(self.maximum_relative_increase)
        if not math.isfinite(baseline) or baseline <= 0.0:
            raise ValueError("locked dense baseline BCE must be finite and positive")
        if not math.isfinite(relative) or relative < 0.0:
            raise ValueError("maximum_relative_increase must be finite and nonnegative")
        rule = BCELossBoundaryDecisionRule(
            confidence_level=float(self.confidence_level),
            maximum_bce_increment=relative * baseline,
        )
        rule.validate()
        return rule


@dataclass(frozen=True)
class BCEPrecisionPolicy:
    """Decision rule plus operational facts; only ``rule`` is scientific."""

    rule: (
        BCEPrecisionDecisionRule
        | V0MarginalConnectionDecisionRule
        | BCELossBoundaryDecisionRule
        | DirectCandidateIncrementDecisionRule
    )
    image_budget: float | None = None
    futility_horizon_images: int | None = None
    look_batch_images: int | None = None
    fold_count: int = 2
    initial_density_step: float | None = None
    minimum_density_step: float | None = None
    minimum_density: float = 0.05
    finite_population_correction: bool = False
    maximum_planned_looks: int | None = 32
    precision_only_iid: bool = False
    runtime_deadline_monotonic: float | None = None
    unresolved_action: str = "no_commit"
    random_seed: int = 0
    # Select/confirm replication: a provisional accept must replicate on a
    # disjoint confirm stream (one capped confirm query per proposal, only
    # on accepts) before it commits. Targets winner's-curse accepts at the
    # threshold; operational, not part of the scientific rule identity.
    accept_replication_gate: bool = False
    # Proposal-adjudication oracle sampling law (mirrors
    # ``IndependentSampleSizePolicy.oracle_sampling_law``); the backend must
    # expose the same id as ``oracle_sampling_id``.
    oracle_sampling_law: str = IID_WITH_REPLACEMENT_SAMPLING_ID
    # Operational fail-safe for ONE proposal's paired oracle acquisition; never
    # refuses the first look, enters no statistic, ``None`` reproduces prior
    # behaviour exactly (mirrors the response-field envelope).
    oracle_acquisition_envelope_seconds: float | None = None
    # None preserves the historical with-replacement API as v4. Registered
    # runners pass v5 explicitly; sealed low-level runs also resolve to v5.
    decision_law_id: str | None = None
    v5_sidedness: str = "one_sided_alpha_over_two"

    @property
    def resolved_decision_law_id(self) -> str:
        if self.decision_law_id is not None:
            return str(self.decision_law_id)
        if not isinstance(self.rule, V0MarginalConnectionDecisionRule):
            return SEQUENTIAL_CALIBRATED_BCE_V4
        return (
            SEQUENTIAL_CALIBRATED_BCE_V5
            if self.oracle_sampling_law == SEALED_WOR_CLASS_STRATIFIED_SAMPLING_ID
            else SEQUENTIAL_CALIBRATED_BCE_V4
        )

    def validate(self) -> None:
        self.rule.validate()
        if self.resolved_decision_law_id not in {
            SEQUENTIAL_CALIBRATED_BCE_V4,
            SEQUENTIAL_CALIBRATED_BCE_V5,
        }:
            raise ValueError("unknown v0 decision law")
        if self.v5_sidedness != "one_sided_alpha_over_two":
            raise ValueError("unsupported v5 sidedness")
        if str(self.oracle_sampling_law) not in ORACLE_SAMPLING_LAWS:
            raise ValueError(
                "oracle_sampling_law must be one of "
                f"{ORACLE_SAMPLING_LAWS!r}, got {self.oracle_sampling_law!r}"
            )
        if self.oracle_acquisition_envelope_seconds is not None and (
            not math.isfinite(float(self.oracle_acquisition_envelope_seconds))
            or float(self.oracle_acquisition_envelope_seconds) <= 0.0
        ):
            raise ValueError(
                "oracle_acquisition_envelope_seconds must be positive and "
                "finite when declared"
            )
        if self.image_budget is not None and float(self.image_budget) <= 0.0:
            raise ValueError("image_budget must be positive when declared")
        if self.fold_count < 2:
            raise ValueError("cross-fitting requires at least two folds")
        if self.look_batch_images is not None and int(self.look_batch_images) <= 0:
            raise ValueError("look_batch_images must be positive")
        if (
            self.futility_horizon_images is not None
            and self.look_batch_images is not None
            and int(self.futility_horizon_images) < int(self.look_batch_images)
        ):
            raise ValueError("the evidence horizon must contain at least one look")
        if (
            self.initial_density_step is not None
            and self.minimum_density_step is not None
            and not (
                0.0
                < float(self.minimum_density_step)
                <= float(self.initial_density_step)
                <= 0.95
            )
        ):
            raise ValueError("density steps must satisfy 0 < min <= initial")
        if not 0.0 < float(self.minimum_density) < 1.0:
            raise ValueError("minimum_density must lie in (0,1)")
        if self.unresolved_action != "no_commit":
            raise ValueError("BCE precision exhaustion must never commit")
        if self.precision_only_iid:
            if self.maximum_planned_looks is not None:
                raise ValueError(
                    "precision-only IID execution cannot declare a look horizon"
                )
            if self.image_budget is not None:
                raise ValueError(
                    "precision-only IID execution cannot declare an image budget"
                )
            if self.runtime_deadline_monotonic is not None:
                raise ValueError(
                    "precision-only IID execution cannot declare a runtime deadline"
                )
        elif self.maximum_planned_looks is None or int(self.maximum_planned_looks) <= 0:
            raise ValueError("maximum_planned_looks must be positive")
        if self.runtime_deadline_monotonic is not None and (
            not math.isfinite(float(self.runtime_deadline_monotonic))
            or float(self.runtime_deadline_monotonic) <= 0.0
        ):
            raise ValueError("runtime deadline must be finite and positive")

    def resolve(self, context: NaturalizationContext) -> "BCEPrecisionPolicy":
        """Compile cadence, horizon, and bisection resolution from runtime facts."""

        context.validate()
        look = int(self.look_batch_images or context.gpu_batch_size)
        horizon = None if self.precision_only_iid else self.futility_horizon_images
        if horizon is None and not self.precision_only_iid:
            horizon = int(context.class_count) * int(context.bank_depth_per_class)
            if self.image_budget is not None:
                horizon = min(int(horizon), max(look, int(self.image_budget)))
        minimum_step = self.minimum_density_step
        if minimum_step is None:
            minimum_step = 1.0 / float(min(context.partition_route_counts.values()))
        initial_step = self.initial_density_step
        if initial_step is None:
            if context.frontier_prior:
                prior = float(np.mean(tuple(context.frontier_prior.values())))
                # The prior is the mean FINAL density of finished groups; a
                # deep predecessor (loose tau walked to its grid floor)
                # drives 1 - prior above validate()'s 0.95 step ceiling and
                # killed the arm on a quantity schedule-mode walks never
                # use. Derived values must satisfy their own validator.
                initial_step = min(
                    max(1.0 - prior, float(minimum_step), 0.1),
                    1.0 - float(self.minimum_density),
                    0.95,
                )
            else:
                initial_step = max(0.25, float(minimum_step))
        resolved = BCEPrecisionPolicy(
            rule=self.rule,
            image_budget=self.image_budget,
            futility_horizon_images=(None if horizon is None else int(horizon)),
            look_batch_images=int(look),
            fold_count=int(self.fold_count),
            initial_density_step=float(initial_step),
            minimum_density_step=float(minimum_step),
            minimum_density=float(self.minimum_density),
            finite_population_correction=bool(self.finite_population_correction),
            maximum_planned_looks=(
                None
                if self.maximum_planned_looks is None
                else int(self.maximum_planned_looks)
            ),
            precision_only_iid=bool(self.precision_only_iid),
            runtime_deadline_monotonic=self.runtime_deadline_monotonic,
            unresolved_action="no_commit",
            random_seed=int(self.random_seed),
            # resolve() rebuilds the policy field by field; omitting a field
            # here silently reverts it to its default in every production
            # run while direct-policy unit tests keep passing.
            accept_replication_gate=bool(self.accept_replication_gate),
            oracle_sampling_law=str(self.oracle_sampling_law),
            decision_law_id=self.decision_law_id,
            v5_sidedness=self.v5_sidedness,
            oracle_acquisition_envelope_seconds=(
                None
                if self.oracle_acquisition_envelope_seconds is None
                else float(self.oracle_acquisition_envelope_seconds)
            ),
        )
        resolved.validate()
        return resolved


@dataclass(frozen=True)
class _BCEStepEvidence:
    action: str
    reason: str
    looks: int
    panel_images: int
    mean: float
    lower: float
    upper: float
    panel_hashes: tuple[str, ...]
    panel_receipts: tuple[Mapping[str, Any], ...]
    calibration_precision_receipts: tuple[Mapping[str, Any], ...] = ()
    connection_delta_receipt: Mapping[str, Any] | None = None
    decision_receipt: Mapping[str, Any] | None = None
    oracle_census_receipts: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class _V6BCEStepEvidence(_BCEStepEvidence):
    """The step evidence plus its compact kind-3 decision record.

    A subclass, so the legacy evidence keeps exactly its v5.2 fields.
    """

    decision_record: Mapping[str, Any] | None = None


def _stratified_bounded_bce_interval(
    values: np.ndarray,
    strata: tuple[str, ...],
    *,
    stratum_weights: Mapping[str, float],
    alpha: float,
    loss_upper_bound: float,
) -> tuple[float, float, float, float]:
    """Two-sided empirical-Bernstein projection for predictable IID strata."""

    observations = np.asarray(values, dtype=np.float64).reshape(-1)
    labels = tuple(str(value) for value in strata)
    keys = tuple(sorted(str(key) for key in stratum_weights))
    if (
        observations.size != len(labels)
        or not keys
        or set(labels) != set(keys)
        or not 0.0 < float(alpha) < 1.0
        or not math.isfinite(float(loss_upper_bound))
        or float(loss_upper_bound) <= 0.0
    ):
        raise ValueError("bounded stratified BCE interval design is invalid")
    weight_sum = sum(float(stratum_weights[key]) for key in keys)
    if not math.isclose(weight_sum, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("bounded BCE stratum weights must sum to one")
    per_stratum_alpha = float(alpha) / float(len(keys))
    logarithm = math.log(3.0 / per_stratum_alpha)
    increment_range = 2.0 * float(loss_upper_bound)
    mean = 0.0
    lower = 0.0
    upper = 0.0
    aggregate_variance = 0.0
    for key in keys:
        sample = observations[
            np.asarray(tuple(value == key for value in labels), dtype=bool)
        ]
        if sample.size < 2:
            raise ValueError(
                "bounded BCE interval requires two IID draws in every stratum"
            )
        sample_mean = float(np.mean(sample))
        sample_variance = max(float(np.var(sample, ddof=1)), 0.0)
        radius = math.sqrt(2.0 * sample_variance * logarithm / float(sample.size)) + (
            3.0 * increment_range * logarithm / float(sample.size)
        )
        weight = float(stratum_weights[key])
        mean += weight * sample_mean
        lower += weight * (sample_mean - radius)
        upper += weight * (sample_mean + radius)
        aggregate_variance += weight * weight * sample_variance / float(sample.size)
    return mean, lower, upper, aggregate_variance


def _population_size(panel: CrossfitPanel, sample_size: int) -> float:
    metadata = dict(panel.allocation_metadata)
    explicit = sum(
        int(metadata.get(name, 0) or 0)
        for name in ("positive_population", "negative_population")
    )
    if explicit > 0:
        return float(max(explicit, sample_size))
    if panel.sampling_weights:
        return float(max(sample_size, sum(panel.sampling_weights)))
    return float(sample_size)


def _v6_panel_receipt(
    *,
    query_id: str,
    layer_id: str,
    partition_id: str,
    proposal_index: int,
    look_index: int,
    target_images: int,
    actual_images: int,
    previous_actual: int,
    sampling_with_replacement: bool,
    fpc_applied: bool,
    census: "_OracleCensusState",
    policy: "BCEPrecisionPolicy",
    look_started: float,
    interval_method: str,
) -> dict[str, Any]:
    """The per-look timing receipt with scalar fields only.

    The panel digest and the compacted allocation metadata (O(cumulative)
    per look) are kind 4 and are dropped.  ``elapsed_seconds``
    is kept exactly: the acquisition envelope reads it.
    """

    return {
        "schema": V6_PANEL_RECEIPT_SCHEMA,
        "query_id": str(query_id),
        "layer_id": str(layer_id),
        "partition_id": str(partition_id),
        "proposal_index": int(proposal_index),
        "look_index": int(look_index),
        "requested_cumulative_images": int(target_images),
        "delivered_cumulative_images": int(actual_images),
        "new_context_count": int(actual_images - previous_actual),
        "sampling_with_replacement": bool(sampling_with_replacement),
        "finite_population_correction_applied": bool(fpc_applied),
        "interval_method": str(interval_method),
        **(
            {
                "finite_population_correction_policy": bool(
                    policy.finite_population_correction
                ),
                **census.receipt_fields(),
            }
            if census.contract_present
            else {}
        ),
        "elapsed_seconds": float(time.perf_counter() - look_started),
    }


# Sentinel. ``_look_statistics`` returns it for a raw-numerator local
# that the look left unassigned (the bounded-IID interval assigns none of
# the four), and the loop then keeps its prior binding, exactly as the
# inline block did.
_LOOK_UNASSIGNED: Any = object()


def _look_statistics_v6(
    *,
    census,
    estimand,
    evidence,
    joint_precision_allocation_receipts,
    latest_step_observables,
    look_alpha,
    looks,
    loss_span_certificate,
    observables_enabled,
    policy,
    raw_stratum_statistics,
    removed,
    sampling_with_replacement,
    threshold,
) -> tuple[Any, ...] | None:
    """The summary-only paired look over the moment arrays.

    Compute the normalized-loss-span interval directly from the supplied
    moment arrays. Other evidence and interval forms use the ordinary
    moments interface over those same arrays.
    """

    moments_view = evidence.stratified_moments
    if (
        threshold is None
        or estimand != V0_NORMALIZED_LOSS_SPAN_PER_CONNECTION_V1
        or evidence.bce_loss_upper_bound is not None
        or loss_span_certificate is None
    ):
        return None
    paired_bce_increase = np.asarray(
        evidence.paired_bce_increment,
        dtype=np.float64,
    )
    if paired_bce_increase.size:
        return None
    calibration_state_ids = (
        evidence.committed_calibration_state_id,
        evidence.candidate_calibration_state_id,
    )
    observed_step = latest_step_observables
    if observables_enabled:
        candidate_percentiles = evidence.candidate_preactivation_percentiles
        observed_step = {
            **(
                {
                    "activation_flip_counts": dict(
                        evidence.activation_flip_counts
                    ),
                    "activation_flip_context_count": int(
                        evidence.activation_flip_context_count
                    ),
                }
                if evidence.activation_flip_counts
                and evidence.activation_flip_context_count is not None
                else {}
            ),
            **(
                {"activation_coverage": dict(evidence.activation_coverage)}
                if evidence.activation_coverage
                else {}
            ),
            **(
                {
                    "candidate_preactivation_percentiles": {
                        str(layer): dict(values)
                        for layer, values in candidate_percentiles.items()
                    }
                }
                if evidence.candidate_preactivation_percentiles
                else {}
            ),
            **(
                {
                    "preactivation_percentile_coverage": dict(
                        evidence.preactivation_percentile_coverage
                    )
                }
                if evidence.preactivation_percentile_coverage
                else {}
            ),
            **(
                {
                    "dense_anchored_readings": dict(
                        evidence.dense_anchored_readings
                    )
                }
                if evidence.dense_anchored_readings
                else {}
            ),
        }
    raw_paired_bce_increase = paired_bce_increase
    final_context_ids = _string_tuple(evidence.panel.context_ids)
    observation_rows = evidence.paired_bce_increment_observations
    final_raw_paired_bce_increase = np.asarray(
        raw_paired_bce_increase if observation_rows is None else observation_rows,
        dtype=np.float64,
    ).copy()
    strata = _string_tuple(
        evidence.panel.allocation_metadata.get("oracle_strata", ())
    )
    stratum_weights = {
        str(key): float(value)
        for key, value in evidence.panel.allocation_metadata.get(
            "oracle_stratum_weights",
            {},
        ).items()
    }
    oracle_population_counts = (
        None if sampling_with_replacement else census.population_by_stratum
    )
    raw_summary = stratified_student_t_summary_from_moment_arrays(
        moments_view.names,
        moments_view.counts,
        moments_view.means,
        moments_view.m2,
        stratum_weights=stratum_weights,
        alpha=look_alpha,
        population_counts=oracle_population_counts,
    )
    raw_mean = raw_summary.mean
    raw_lower = raw_summary.lower
    raw_upper = raw_summary.upper
    raw_standard_error = raw_summary.standard_error
    joint_decision_envelope = _v0_joint_interval(
        policy=policy,
        threshold=float(threshold),
        **({"degrees_of_freedom": raw_summary.degrees_of_freedom}
           if policy.resolved_decision_law_id == SEQUENTIAL_CALIBRATED_BCE_V5 else {}),
        **(_v5_anchor_inputs(loss_span_certificate)
           if policy.resolved_decision_law_id == SEQUENTIAL_CALIBRATED_BCE_V5 else {}),
        numerator_estimate=raw_mean,
        numerator_standard_error=raw_standard_error,
        dense_estimate=(loss_span_certificate.dense.estimate),
        dense_standard_error=(loss_span_certificate.dense.standard_error),
        fully_pruned_risk=(loss_span_certificate.fully_pruned.estimate),
        connection_delta=removed,
        parent_calibration=(evidence.committed_calibration_profile),
        candidate_calibration=(evidence.candidate_calibration_profile),
        dense_calibration=(loss_span_certificate.dense.calibration_profile),
        alpha=look_alpha,
        look_index=looks - 1,
    )
    mean = joint_decision_envelope.estimate
    lower = joint_decision_envelope.lower
    upper = joint_decision_envelope.upper
    # Committed: from here on the look is the v6 look (side effects once).
    stratified_fpc_applied = bool(
        raw_summary.finite_population_correction_applied
    )
    raw_current_interval_width = max(
        0.0,
        float(raw_upper) - float(raw_lower),
    )
    stratum_counts = dict(raw_summary.counts_by_stratum)
    stratum_means = dict(raw_summary.means_by_stratum)
    stratum_variances = dict(raw_summary.variances_by_stratum)
    raw_stratum_statistics = {
        key: {
            "count": stratum_counts[key],
            "mean": stratum_means[key],
            "variance": stratum_variances[key],
        }
        for key in stratum_counts
    }
    optimized_calibration_envelope = (
        OptimizedCalibrationDecisionEnvelope.paired_increment(
            oracle_estimate=raw_mean,
            oracle_lower=raw_lower,
            oracle_upper=raw_upper,
            parent_improvement_bound=(
                evidence.committed_calibration_improvement_bound
            ),
            candidate_improvement_bound=(
                evidence.candidate_calibration_improvement_bound
            ),
        )
    )
    optimized_numerator_lower = float(
        joint_decision_envelope.component_bounds["paired_oracle_radius"]
    )
    optimized_numerator_lower = (
        raw_mean
        - optimized_numerator_lower
        - joint_decision_envelope.component_bounds[
            "candidate_calibration_improvement"
        ]
    )
    optimized_numerator_upper = (
        raw_mean
        + joint_decision_envelope.component_bounds["paired_oracle_radius"]
        + joint_decision_envelope.component_bounds[
            "parent_calibration_improvement"
        ]
    )
    current_standard_error = raw_standard_error / (
        loss_span_certificate.estimate * float(removed)
    )
    if isinstance(joint_decision_envelope, ScalarV0DecisionEnvelope):
        current_standard_error = math.sqrt(
            joint_decision_envelope.statistic_variance
        ) / (loss_span_certificate.estimate * float(removed))
    joint_precision_allocation_receipts.extend(
        dict(item) for item in evidence.joint_precision_allocation_receipts
    )
    return (
        False,
        calibration_state_ids,
        current_standard_error,
        final_context_ids,
        final_raw_paired_bce_increase,
        joint_decision_envelope,
        observed_step,
        lower,
        mean,
        optimized_calibration_envelope,
        optimized_numerator_lower,
        optimized_numerator_upper,
        raw_current_interval_width,
        raw_lower,
        raw_mean,
        raw_standard_error,
        raw_stratum_statistics,
        raw_upper,
        strata,
        stratified_fpc_applied,
        stratum_weights,
        upper,
    )


def _look_statistics(
    *,
    candidate_hash,
    census,
    committed_hash,
    estimand,
    evidence,
    joint_decision_envelope,
    joint_precision_allocation_receipts,
    latest_step_observables,
    look_alpha,
    looks,
    loss_span_certificate,
    observables_enabled,
    optimized_calibration_envelope,
    optimized_numerator_lower,
    optimized_numerator_upper,
    policy,
    raw_current_interval_width,
    raw_stratum_statistics,
    removed,
    sampling_with_replacement,
    stratified_fpc_applied,
    warm_starts,
    threshold=None,
) -> tuple[Any, ...]:
    """Turn one look's evidence into moments, the interval and forecast inputs.

    The statements are the original per-look block of
    ``_decide_bce_precision_step``, unchanged and in the same order. Each local
    the block reads arrives as a keyword argument, and each local the loop reads
    afterwards is returned, in the order the call site unpacks. The four
    raw-numerator locals start as ``_LOOK_UNASSIGNED``: the bounded-IID branch
    assigns none of them, and the caller then keeps its prior binding.
    """

    if isinstance(evidence, FixedCalibrationPairedBCEEvidence) and isinstance(
        evidence.stratified_moments, _v6sb.V6StratumMoments
    ):
        # The specialized array interval and the ordinary moments interface
        # both consume the same supplied packet; neither recomputes its rows.
        v6_look = _look_statistics_v6(
            census=census,
            estimand=estimand,
            evidence=evidence,
            joint_precision_allocation_receipts=joint_precision_allocation_receipts,
            latest_step_observables=latest_step_observables,
            look_alpha=look_alpha,
            looks=looks,
            loss_span_certificate=loss_span_certificate,
            observables_enabled=observables_enabled,
            policy=policy,
            raw_stratum_statistics=raw_stratum_statistics,
            removed=removed,
            sampling_with_replacement=sampling_with_replacement,
            threshold=(
                float(threshold if threshold is not None else policy.rule.sensitivity_tau)
            ),
        )
        if v6_look is not None:
            return v6_look
    raw_lower = raw_mean = raw_standard_error = raw_upper = _LOOK_UNASSIGNED
    if isinstance(evidence, FixedCalibrationPairedBCEEvidence):
        paired_bce_increase = np.asarray(
            evidence.paired_bce_increment,
            dtype=np.float64,
        )
        paired_bce_moments = {
            str(key): RunningMoment.from_mapping(value)
            for key, value in evidence.stratified_moments.items()
        }
        calibration_state_ids = (
            evidence.committed_calibration_state_id,
            evidence.candidate_calibration_state_id,
        )
        joint_precision_allocation_receipts.extend(
            dict(item) for item in evidence.joint_precision_allocation_receipts
        )
        if observables_enabled:
            candidate_percentiles = evidence.candidate_preactivation_percentiles
            latest_step_observables = {
                **(
                    {
                        "activation_flip_counts": dict(
                            evidence.activation_flip_counts
                        ),
                        "activation_flip_context_count": int(
                            evidence.activation_flip_context_count
                        ),
                    }
                    if evidence.activation_flip_counts
                    and evidence.activation_flip_context_count is not None
                    else {}
                ),
                **(
                    {"activation_coverage": dict(evidence.activation_coverage)}
                    if evidence.activation_coverage
                    else {}
                ),
                **(
                    {
                        "candidate_preactivation_percentiles": {
                            str(layer): dict(values)
                            for layer, values in candidate_percentiles.items()
                        }
                    }
                    if evidence.candidate_preactivation_percentiles
                    else {}
                ),
                **(
                    {
                        "preactivation_percentile_coverage": dict(
                            evidence.preactivation_percentile_coverage
                        )
                    }
                    if evidence.preactivation_percentile_coverage
                    else {}
                ),
                **(
                    {
                        "dense_anchored_readings": dict(
                            evidence.dense_anchored_readings
                        )
                    }
                    if evidence.dense_anchored_readings
                    else {}
                ),
            }
    else:
        warm_starts[committed_hash] = fold_warm_start(
            evidence.committed_calibration_by_fold
        )
        warm_starts[candidate_hash] = fold_warm_start(
            evidence.candidate_calibration_by_fold
        )
        paired_bce_increase = np.asarray(
            evidence.candidate_loss, dtype=np.float64
        ) - np.asarray(evidence.committed_loss, dtype=np.float64)
        paired_bce_moments = {}
        calibration_state_ids = (
            f"crossfit-parent:{committed_hash}",
            f"crossfit-candidate:{candidate_hash}",
        )
    raw_paired_bce_increase = np.asarray(
        paired_bce_increase,
        dtype=np.float64,
    )
    final_context_ids = _string_tuple(evidence.panel.context_ids)
    observation_rows = (
        evidence.paired_bce_increment_observations
        if isinstance(evidence, FixedCalibrationPairedBCEEvidence)
        else None
    )
    final_raw_paired_bce_increase = np.asarray(
        raw_paired_bce_increase if observation_rows is None else observation_rows,
        dtype=np.float64,
    ).copy()
    if estimand == "bce_increment_per_removed_route":
        paired_bce_increase = paired_bce_increase / float(removed)
        paired_bce_moments = {
            key: value.scaled(1.0 / float(removed))
            for key, value in paired_bce_moments.items()
        }
    weights = (
        np.asarray(evidence.panel.sampling_weights, dtype=np.float64)
        if evidence.panel.sampling_weights
        else np.ones(paired_bce_increase.size, dtype=np.float64)
    )
    bounded_iid = (
        isinstance(evidence, FixedCalibrationPairedBCEEvidence)
        and evidence.bce_loss_upper_bound is not None
        and sampling_with_replacement
        and estimand
        not in {
            V0_NORMALIZED_LOSS_SPAN_PER_CONNECTION_V1,
            "direct_candidate_bce_increment",
        }
    )
    if bounded_iid:
        strata = _string_tuple(
            evidence.panel.allocation_metadata.get("oracle_strata", ())
        )
        stratum_weights = {
            str(key): float(value)
            for key, value in evidence.panel.allocation_metadata.get(
                "oracle_stratum_weights",
                {},
            ).items()
        }
        loss_upper_bound = float(evidence.bce_loss_upper_bound)
        if estimand == "bce_increment_per_removed_route":
            loss_upper_bound /= float(removed)
        (
            mean,
            lower,
            upper,
            observation_variance,
        ) = _stratified_bounded_bce_interval(
            paired_bce_increase,
            strata,
            stratum_weights=stratum_weights,
            alpha=look_alpha,
            loss_upper_bound=loss_upper_bound,
        )
        effective_n = float(paired_bce_increase.size)
        current_standard_error = math.sqrt(max(0.0, float(observation_variance)))
    else:
        strata = _string_tuple(
            evidence.panel.allocation_metadata.get("oracle_strata", ())
        )
        stratum_weights = {
            str(key): float(value)
            for key, value in evidence.panel.allocation_metadata.get(
                "oracle_stratum_weights",
                {},
            ).items()
        }
        # Sealed without-replacement panels carry their finite
        # populations; ``None`` keeps the with-replacement arithmetic
        # bitwise identical.
        oracle_population_counts = (
            None if sampling_with_replacement else census.population_by_stratum
        )
        if (strata and stratum_weights) or paired_bce_moments:
            raw_summary = (
                stratified_student_t_summary_from_moments(
                    paired_bce_moments,
                    stratum_weights=stratum_weights,
                    alpha=look_alpha,
                    population_counts=oracle_population_counts,
                )
                if paired_bce_moments
                else stratified_student_t_summary(
                    raw_paired_bce_increase,
                    strata,
                    stratum_weights=stratum_weights,
                    alpha=look_alpha,
                    population_counts=oracle_population_counts,
                )
            )
            stratified_fpc_applied = bool(
                raw_summary.finite_population_correction_applied
            )
            raw_mean = raw_summary.mean
            raw_lower = raw_summary.lower
            raw_upper = raw_summary.upper
            raw_standard_error = raw_summary.standard_error
            raw_current_interval_width = max(
                0.0,
                float(raw_upper) - float(raw_lower),
            )
            # Build each stratum lookup once (not all three per stratum);
            # same keys, order and values.
            stratum_counts = dict(raw_summary.counts_by_stratum)
            stratum_means = dict(raw_summary.means_by_stratum)
            stratum_variances = dict(raw_summary.variances_by_stratum)
            raw_stratum_statistics = {
                key: {
                    "count": stratum_counts[key],
                    "mean": stratum_means[key],
                    "variance": stratum_variances[key],
                }
                for key in stratum_counts
            }
            observation_variance = raw_standard_error * raw_standard_error
            effective_n = float(
                sum(moment.count for moment in paired_bce_moments.values())
                if paired_bce_moments
                else raw_paired_bce_increase.size
            )
            if estimand == V0_NORMALIZED_LOSS_SPAN_PER_CONNECTION_V1:
                assert loss_span_certificate is not None
                if isinstance(
                    evidence,
                    FixedCalibrationPairedBCEEvidence,
                ):
                    joint_decision_envelope = _v0_joint_interval(
                        policy=policy,
                        threshold=float(
                            threshold if threshold is not None
                            else policy.rule.sensitivity_tau
                        ),
                        **({"degrees_of_freedom": raw_summary.degrees_of_freedom}
                           if policy.resolved_decision_law_id == SEQUENTIAL_CALIBRATED_BCE_V5 else {}),
                        **(_v5_anchor_inputs(loss_span_certificate)
                           if policy.resolved_decision_law_id == SEQUENTIAL_CALIBRATED_BCE_V5 else {}),
                        numerator_estimate=raw_mean,
                        numerator_standard_error=raw_standard_error,
                        dense_estimate=(loss_span_certificate.dense.estimate),
                        dense_standard_error=(
                            loss_span_certificate.dense.standard_error
                        ),
                        fully_pruned_risk=(
                            loss_span_certificate.fully_pruned.estimate
                        ),
                        connection_delta=removed,
                        parent_calibration=(evidence.committed_calibration_profile),
                        candidate_calibration=(
                            evidence.candidate_calibration_profile
                        ),
                        dense_calibration=(
                            loss_span_certificate.dense.calibration_profile
                        ),
                        alpha=look_alpha,
                        look_index=looks - 1,
                    )
                    optimized_calibration_envelope = (
                        OptimizedCalibrationDecisionEnvelope.paired_increment(
                            oracle_estimate=raw_mean,
                            oracle_lower=raw_lower,
                            oracle_upper=raw_upper,
                            parent_improvement_bound=(
                                evidence.committed_calibration_improvement_bound
                            ),
                            candidate_improvement_bound=(
                                evidence.candidate_calibration_improvement_bound
                            ),
                        )
                    )
                    optimized_numerator_lower = float(
                        joint_decision_envelope.component_bounds[
                            "paired_oracle_radius"
                        ]
                    )
                    optimized_numerator_lower = (
                        raw_mean
                        - optimized_numerator_lower
                        - joint_decision_envelope.component_bounds[
                            "candidate_calibration_improvement"
                        ]
                    )
                    optimized_numerator_upper = (
                        raw_mean
                        + joint_decision_envelope.component_bounds[
                            "paired_oracle_radius"
                        ]
                        + joint_decision_envelope.component_bounds[
                            "parent_calibration_improvement"
                        ]
                    )
                    mean = joint_decision_envelope.estimate
                    lower = joint_decision_envelope.lower
                    upper = joint_decision_envelope.upper
                else:
                    optimized_numerator_lower = float(raw_lower)
                    optimized_numerator_upper = float(raw_upper)
                    if policy.resolved_decision_law_id == SEQUENTIAL_CALIBRATED_BCE_V5:
                        zero_profile = CalibrationImprovementProfile(parameter_dimension=0)
                        joint_decision_envelope = scalar_v0_decision_interval(
                            **_v5_anchor_inputs(loss_span_certificate),
                            numerator_estimate=raw_mean,
                            numerator_standard_error=raw_standard_error,
                            dense_estimate=loss_span_certificate.dense.estimate,
                            dense_standard_error=loss_span_certificate.dense.standard_error,
                            fully_pruned_risk=loss_span_certificate.fully_pruned.estimate,
                            connection_delta=removed,
                            parent_calibration=zero_profile,
                            candidate_calibration=zero_profile,
                            dense_calibration=loss_span_certificate.dense.calibration_profile,
                            threshold=float(threshold if threshold is not None else policy.rule.sensitivity_tau),
                            alpha=look_alpha,
                            look_index=looks - 1,
                            degrees_of_freedom=raw_summary.degrees_of_freedom,
                        )
                        mean, lower, upper = (
                            joint_decision_envelope.estimate,
                            joint_decision_envelope.lower,
                            joint_decision_envelope.upper,
                        )
                    else:
                        mean, lower, upper = conservative_ratio_interval(
                            numerator_estimate=raw_mean,
                            numerator_lower=raw_lower,
                            numerator_upper=raw_upper,
                            positive_denominator_estimate=loss_span_certificate.estimate,
                            positive_denominator_lower=loss_span_certificate.lower,
                            positive_denominator_upper=loss_span_certificate.upper,
                            connection_delta=removed,
                        )
                current_standard_error = raw_standard_error / (
                    loss_span_certificate.estimate * float(removed)
                )
                if isinstance(joint_decision_envelope, ScalarV0DecisionEnvelope):
                    current_standard_error = math.sqrt(
                        joint_decision_envelope.statistic_variance
                    ) / (loss_span_certificate.estimate * float(removed))
            else:
                scale = (
                    float(removed)
                    if estimand == "bce_increment_per_removed_route"
                    else 1.0
                )
                mean = raw_mean / scale
                lower = raw_lower / scale
                upper = raw_upper / scale
                current_standard_error = raw_standard_error / scale
        else:
            mean, observation_variance, effective_n = _weighted_sensitivity_moments(
                paired_bce_increase,
                weights,
            )
            population = _population_size(
                evidence.panel,
                paired_bce_increase.size,
            )
            if (
                policy.finite_population_correction
                and not sampling_with_replacement
            ):
                fpc = max(
                    0.0,
                    1.0 - effective_n / max(population, effective_n),
                )
                observation_variance *= fpc
            interval = student_t_interval(
                mean,
                observation_variance,
                max(effective_n, 1.0),
                alpha=look_alpha,
                dof=max(effective_n - 1.0, 0.0),
            )
            lower = float(interval.lower[()])
            upper = float(interval.upper[()])
            current_standard_error = float(
                np.asarray(interval.standard_error).reshape(-1)[0]
            )
            # The unstratified fallback must still bind the raw
            # numerator statistics the sequential receipt and the
            # per-look progress event require, and the modern-v0
            # statistic is a loss-span fraction per connection -- the
            # raw paired reading has to pass through the same
            # conservative ratio normalization as the stratified path
            # or it is compared against sensitivity_tau in raw BCE
            # units.
            raw_scale = (
                float(removed)
                if estimand == "bce_increment_per_removed_route"
                else 1.0
            )
            raw_mean = float(mean) * raw_scale
            raw_lower = float(lower) * raw_scale
            raw_upper = float(upper) * raw_scale
            raw_standard_error = float(current_standard_error) * raw_scale
            raw_current_interval_width = max(
                0.0,
                raw_upper - raw_lower,
            )
            if estimand == V0_NORMALIZED_LOSS_SPAN_PER_CONNECTION_V1:
                assert loss_span_certificate is not None
                if policy.resolved_decision_law_id == SEQUENTIAL_CALIBRATED_BCE_V5:
                    zero_profile = CalibrationImprovementProfile(parameter_dimension=0)
                    joint_decision_envelope = scalar_v0_decision_interval(
                        **_v5_anchor_inputs(loss_span_certificate),
                        numerator_estimate=raw_mean,
                        numerator_standard_error=raw_standard_error,
                        dense_estimate=loss_span_certificate.dense.estimate,
                        dense_standard_error=loss_span_certificate.dense.standard_error,
                        fully_pruned_risk=loss_span_certificate.fully_pruned.estimate,
                        connection_delta=removed,
                        parent_calibration=zero_profile,
                        candidate_calibration=zero_profile,
                        dense_calibration=loss_span_certificate.dense.calibration_profile,
                        threshold=float(threshold if threshold is not None else policy.rule.sensitivity_tau),
                        alpha=look_alpha,
                        look_index=looks - 1,
                        degrees_of_freedom=max(effective_n - 1.0, 1.0),
                    )
                    mean, lower, upper = (
                        joint_decision_envelope.estimate,
                        joint_decision_envelope.lower,
                        joint_decision_envelope.upper,
                    )
                else:
                    mean, lower, upper = conservative_ratio_interval(
                        numerator_estimate=raw_mean,
                        numerator_lower=raw_lower,
                        numerator_upper=raw_upper,
                        positive_denominator_estimate=loss_span_certificate.estimate,
                        positive_denominator_lower=loss_span_certificate.lower,
                        positive_denominator_upper=loss_span_certificate.upper,
                        connection_delta=removed,
                    )
                current_standard_error = raw_standard_error / (
                    loss_span_certificate.estimate * float(removed)
                )
                if isinstance(joint_decision_envelope, ScalarV0DecisionEnvelope):
                    current_standard_error = math.sqrt(
                        joint_decision_envelope.statistic_variance
                    ) / (loss_span_certificate.estimate * float(removed))
    return (
        bounded_iid,
        calibration_state_ids,
        current_standard_error,
        final_context_ids,
        final_raw_paired_bce_increase,
        joint_decision_envelope,
        latest_step_observables,
        lower,
        mean,
        optimized_calibration_envelope,
        optimized_numerator_lower,
        optimized_numerator_upper,
        raw_current_interval_width,
        raw_lower,
        raw_mean,
        raw_standard_error,
        raw_stratum_statistics,
        raw_upper,
        strata,
        stratified_fpc_applied,
        stratum_weights,
        upper,
    )


def _build_decision_receipt(
    *,
    accept_replication_audit,
    action,
    allocation_receipts,
    calibration_state_ids,
    candidate_hash,
    census,
    connection_delta_receipt,
    decision_resource_usage,
    envelope_receipt,
    joint_decision_envelope,
    joint_precision_allocation_receipts,
    latest_step_observables,
    layer_id,
    loss_span_certificate,
    lower,
    mean,
    observables_enabled,
    optimized_calibration_envelope,
    partition_id,
    proposal_index,
    query_id,
    raw_lower,
    raw_mean,
    raw_stratum_statistics,
    raw_upper,
    refinement_rounds,
    removed,
    resolved_threshold,
    sequential_spec,
    upper,
) -> Mapping[str, Any]:
    """Build the modern-v0 ``SequentialBCEDecisionReceipt`` of one decision.

    The statements are the original receipt block of
    ``_decide_bce_precision_step``, unchanged and in the same order. The caller
    invokes it only for the modern-v0 estimand when no suppression condition
    failed.
    """

    spec = sequential_spec
    receipt_action = (
        "accept"
        if action == "accept"
        else (
            "reject"
            if (
                joint_decision_envelope.statistic_lower > 0.0
                if isinstance(joint_decision_envelope, ScalarV0DecisionEnvelope)
                else lower > resolved_threshold
            )
            else "unresolved"
        )
    )
    sequential_receipt = SequentialBCEDecisionReceipt(
        spec_hash=spec.content_hash,
        proposal_id=(
            f"{query_id}:{layer_id}:{partition_id}:{proposal_index}:"
            f"{candidate_hash}"
        ),
        estimand_policy=V0_NORMALIZED_LOSS_SPAN_PER_CONNECTION_V1,
        aggregate_estimate=float(mean),
        lower=float(lower),
        upper=float(upper),
        accept_boundary=resolved_threshold,
        reject_boundary=resolved_threshold,
        action=receipt_action,
        raw_stratum_statistics=raw_stratum_statistics,
        allocation_receipts=tuple(allocation_receipts),
        calibration_state_ids=calibration_state_ids,
        resource_usage=decision_resource_usage,
        connection_delta_receipt_hash=(connection_delta_receipt.content_hash),
        loss_span_certificate_hash=loss_span_certificate.content_hash,
        confidence_scope=spec.confidence_scope,
        optimized_calibration_envelope=(
            {
                **(
                    {}
                    if optimized_calibration_envelope is None
                    else {
                        "legacy_component_envelope": {
                            **asdict(optimized_calibration_envelope),
                            "content_hash": (
                                optimized_calibration_envelope.content_hash
                            ),
                        }
                    }
                ),
                **(
                    {}
                    if joint_decision_envelope is None
                    else {
                        "joint_decision_envelope": {
                            **asdict(joint_decision_envelope),
                            "content_hash": (joint_decision_envelope.content_hash),
                        }
                    }
                ),
            }
            if (
                optimized_calibration_envelope is not None
                or joint_decision_envelope is not None
            )
            else {
                "confidence_bearing_object": "joint_accept_reject_statistic",
            }
        ),
        joint_precision_allocation_receipts=tuple(
            joint_precision_allocation_receipts
        ),
        raw_numerator_estimate=float(raw_mean),
        raw_numerator_lower=float(raw_lower),
        raw_numerator_upper=float(raw_upper),
        positive_loss_span_estimate=float(loss_span_certificate.estimate),
        positive_loss_span_lower=float(loss_span_certificate.lower),
        positive_loss_span_upper=float(loss_span_certificate.upper),
        connection_delta=int(removed),
        accept_replication=accept_replication_audit,
        step_observables=latest_step_observables,
        step_observables_enabled=(None if observables_enabled else False),
        oracle_sampling_law=str(spec.oracle_sampling_id),
        oracle_census_strata_count=int(census.census_strata_count),
        oracle_census_exact_weight=float(census.census_exact_weight),
        oracle_census_complete=bool(census.census_complete),
        refinement_rounds=int(refinement_rounds),
        oracle_acquisition_envelope=envelope_receipt,
        fieller_interval=(
            None if spec.decision_interval_method != SCALAR_DECISION_FIELLER_METHOD
            else joint_decision_envelope.fieller_interval
        ),
        schema=(
            SEQUENTIAL_BCE_DECISION_RECEIPT_V5
            if spec.decision_interval_method == SCALAR_DECISION_FIELLER_METHOD
            else SEQUENTIAL_BCE_DECISION_RECEIPT_V4
        ),
    ).receipt
    return sequential_receipt


def _v6_decision_record(
    *,
    query_id: str,
    layer_id: str,
    partition_id: str,
    proposal_index: int,
    previous_density: float,
    proposed_density: float,
    action: str,
    natural_action: str,
    engine_reason: str | None = None,
    engine_forecast_target: int | None = None,
    reason: str,
    looks: int,
    panel_images: int,
    mean: float,
    lower: float,
    upper: float,
    threshold: float,
    removed: int,
    alpha_allocated: float,
    suppressed: bool,
    forced: bool,
    replication: bool,
    per_look: tuple[tuple[int, int, int, float, float, float], ...],
    candidate_masks: Any,
) -> dict[str, Any]:
    """The compact kind-3 record of one decision.

    It replaces ``SequentialBCEDecisionReceipt`` in production.  Its
    join key is (query, layer, partition, proposal_index); the candidate's
    exact ``HardMask.mask_hash`` rides along when the backend built it.
    """

    cached = getattr(candidate_masks, "cached_mask_hash", None)
    slices = getattr(candidate_masks, "slices", ())
    first = slices[0] if slices else None
    return {
        "schema": "irg_v6_decision_record_v2",
        "query_id": str(query_id),
        "layer_id": str(layer_id),
        "partition_id": str(partition_id),
        "proposal_index": int(proposal_index),
        "previous_density": float(previous_density),
        "proposed_density": float(proposed_density),
        "action": str(action),
        "natural_action": str(natural_action),
        "engine_outcome": str(natural_action),
        "engine_reason": str(reason if engine_reason is None else engine_reason),
        "engine_forecast_target": engine_forecast_target,
        "reason": str(reason),
        "looks": int(looks),
        "panel_images": int(panel_images),
        "mean": float(mean),
        "lower": float(lower),
        "upper": float(upper),
        "threshold": float(threshold),
        "removed_count": int(removed),
        "alpha_allocated": float(alpha_allocated),
        "flags": (
            (FLAG_SUPPRESSED if suppressed else 0)
            | (FLAG_FORCED if forced else 0)
            | (FLAG_ACCEPT_REPLICATION if replication else 0)
        ),
        "k_old": -1 if first is None else int(first.k_old),
        "k_new": -1 if first is None else int(first.k_new),
        "candidate_mask_hash": cached() if callable(cached) else None,
        "per_look": tuple(per_look),
    }


def _v6_decision_reference(record: Mapping[str, Any]) -> dict[str, Any]:
    """The small ``decision_receipt`` of a v6 decision (for the recorder)."""

    return {
        "schema": V6_DECISION_REF_SCHEMA,
        "decision_key": decision_key(
            record["query_id"],
            record["layer_id"],
            record["partition_id"],
            record["proposal_index"],
        ),
        "action": str(record["action"]),
        "suppressed": bool(int(record["flags"]) & FLAG_SUPPRESSED),
        "candidate_mask_hash": record["candidate_mask_hash"],
    }


def _bce_comparison_request(*, query_id, layer_id, partition_id, previous_density, proposed_density, looks, target_images, class_count, proposal_index, policy, prior_panel, warm_starts, committed_hash, candidate_hash, allocation_evidence, precision_only_iid, remaining_group, maximum_target, calibration_target_loss_precision):
    """The same request construction for live looks and first-look descriptions."""

    return GlocalSampledComparisonRequest(
        query_id=str(query_id),
        layer_id=str(layer_id),
        partition_id=str(partition_id),
        previous_density=float(previous_density),
        proposed_density=float(proposed_density),
        stage_index=looks - 1,
        stage_per_class=max(1, math.ceil(target_images / max(int(class_count), 1))),
        proposal_index=int(proposal_index),
        random_seed=int(policy.random_seed),
        fold_count=int(policy.fold_count),
        prior_panel=prior_panel,
        committed_warm_start=warm_starts.get(committed_hash),
        candidate_warm_start=warm_starts.get(candidate_hash),
        allocation_evidence=allocation_evidence,
        cumulative_target_images=target_images,
        maximum_new_contexts=(
            None
            if precision_only_iid or math.isinf(remaining_group)
            else int(remaining_group)
        ),
        maximum_cumulative_images=(
            None if maximum_target is None else int(maximum_target)
        ),
        calibration_target_loss_precision=(calibration_target_loss_precision),
        runtime_deadline_monotonic=(
            None if precision_only_iid else policy.runtime_deadline_monotonic
        ),
    )


def _decide_bce_precision_step(
    *,
    backend: GlocalSampledPathEvaluationBackend,
    committed_masks: Mapping[str, np.ndarray],
    candidate_masks: Mapping[str, np.ndarray],
    query_id: str,
    layer_id: str,
    partition_id: str,
    previous_density: float,
    proposed_density: float,
    proposal_index: int,
    removed_route_count: int,
    threshold: float,
    estimand: str,
    connection_delta_receipt: CircuitConnectionDeltaReceipt | None,
    loss_span_certificate: LossSpanCertificate | None,
    policy: BCEPrecisionPolicy,
    warm_starts: dict[str, Mapping[str, Any]],
    class_count: int,
    progress_callback: Callable[..., None] | None = None,
    step_context_delta_sink: Callable[..., None] | None = None,
    forced_action: ForcedAction | None = None,
    v6_record: bool = False,
    receipt_executor: ThreadPoolExecutor | None = None,
) -> _BCEStepEvidence:
    """Acquire nested paired BCE rows until the threshold relation resolves.

    ``v6_record`` (set by the v6 trace for candidate views): the
    kind-4 digests and JSON receipts are not built; the decision leaves a
    compact ``decision_record`` and a small reference as ``decision_receipt``.
    ``forced_action`` (acceptance only, ``IRG_V6_FORCED_ACTIONS``) replaces the
    action after every statistic of every look has been computed.
    """

    # Forced hook: resolve the version before evidence or a receipt is emitted.
    forced_hook = _v6_forced_hook()
    if forced_hook is not None and not v6_record:
        raise RuntimeError(
            "forced-action hook requires the v6 candidate-view decision path"
        )
    if v6_record and forced_hook is None and forced_action is not None:
        forced_hook = _V6DecisionLogHook(forced_action)
    hook_terminal = False
    engine_outcome = "unresolved"
    engine_reason = "evidence_horizon_exhausted_unresolved"
    engine_forecast_target: int | None = None

    def hook_look(natural_outcome: str, natural_reason: str, target: int | None) -> tuple[str, str, int | None]:
        nonlocal engine_outcome, engine_reason, engine_forecast_target, hook_terminal
        engine_outcome, engine_reason = natural_outcome, natural_reason
        engine_forecast_target = target
        hook_terminal = natural_outcome != "unresolved"
        if forced_hook is None:
            return natural_outcome, natural_reason, target
        natural = {
            "outcome": natural_outcome,
            "reason": natural_reason,
            "next_cumulative_target": target,
            "actual_images": actual_images,
            "target_images": target_images,
            "mean": float(mean),
            "lower": float(lower),
            "upper": float(upper),
            "threshold": float(resolved_threshold),
        }
        result = forced_hook.look(
            str(query_id), str(layer_id), str(partition_id),
            int(proposal_index), int(looks - 1), natural,
        )
        if not isinstance(result, Mapping):
            raise RuntimeError("forced-action look returned no mapping")
        hook_terminal = bool(result.get(
            "terminal",
            result.get("outcome", result.get("action", natural_outcome)) != "unresolved",
        ))
        outcome, forced_reason, forced_target = _v6_hook_result(
            result, natural=natural,
            point="look_terminal" if hook_terminal else "look"
        )
        return outcome, forced_reason, forced_target

    def hook_exit(point: str, natural_outcome: str, natural_reason: str) -> tuple[str, str]:
        nonlocal engine_outcome, engine_reason
        engine_outcome, engine_reason = natural_outcome, natural_reason
        if forced_hook is None:
            return natural_outcome, natural_reason
        natural = {"outcome": natural_outcome, "reason": natural_reason,
                   "next_cumulative_target": None, "actual_images": actual_images}
        result = forced_hook.exit(
            str(query_id), str(layer_id), str(partition_id),
            int(proposal_index), str(point), int(looks), natural,
        )
        outcome, forced_reason, _ = _v6_hook_result(result, natural=natural, point="exit")
        return outcome, forced_reason

    removed = int(removed_route_count)
    if removed <= 0:
        raise ValueError("a BCE precision proposal must remove at least one route")
    precision_only_iid = bool(policy.precision_only_iid)
    horizon = None if precision_only_iid else int(policy.futility_horizon_images)
    packet = int(policy.look_batch_images)
    planned_looks = (
        None
        if policy.maximum_planned_looks is None
        else int(policy.maximum_planned_looks)
    )
    backend_oracle_sampling_id = str(
        getattr(backend, "oracle_sampling_id", IID_WITH_REPLACEMENT_SAMPLING_ID)
    )
    if backend_oracle_sampling_id != str(policy.oracle_sampling_law):
        # The policy declares the law the receipts certify; the backend
        # states the law it actually samples under.  A disagreement is a
        # wiring fault, never something to reconcile silently.
        raise ValueError(
            "oracle sampling law mismatch: policy declares "
            f"{policy.oracle_sampling_law!r} but the backend samples under "
            f"{backend_oracle_sampling_id!r}"
        )
    sequential_spec = SequentialCalibratedBCEResolutionSpec(
        confidence_level=float(policy.rule.confidence_level),
        estimand_policy=V0_NORMALIZED_LOSS_SPAN_PER_CONNECTION_V1,
        maximum_planned_looks=planned_looks,
        precision_only_iid=precision_only_iid,
        oracle_sampling_id=backend_oracle_sampling_id,
        decision_interval_method=(
            SCALAR_DECISION_FIELLER_METHOD
            if policy.resolved_decision_law_id == SEQUENTIAL_CALIBRATED_BCE_V5
            else JOINT_DECISION_WALD_METHOD
        ),
    )
    proposal_error_probability = (
        sequential_spec.proposal_error_probability
        if estimand == V0_NORMALIZED_LOSS_SPAN_PER_CONNECTION_V1
        else float(policy.rule.error_probability)
    )
    resolved_threshold = float(threshold)
    if not math.isfinite(resolved_threshold) or resolved_threshold < 0.0:
        raise ValueError("the BCE decision threshold must be finite and nonnegative")
    if estimand not in {
        "aggregate_complete_mask_bce_increment",
        "direct_candidate_bce_increment",
        "bce_increment_per_removed_route",
        V0_NORMALIZED_LOSS_SPAN_PER_CONNECTION_V1,
    }:
        raise ValueError(f"unsupported BCE precision estimand {estimand!r}")
    if estimand == V0_NORMALIZED_LOSS_SPAN_PER_CONNECTION_V1:
        if connection_delta_receipt is None or loss_span_certificate is None:
            raise ValueError(
                "modern v0 requires connection-delta and loss-span certificates"
            )
        loss_span_certificate.validate_positive()
    if estimand == V0_NORMALIZED_LOSS_SPAN_PER_CONNECTION_V1 and isinstance(
        connection_delta_receipt, ConnectionDelta
    ):
        # The connection counts are exact integers from the
        # order slices; the receipt digests they used to re-check are kind 4.
        if int(connection_delta_receipt.total_removed_connections) != removed:
            raise ValueError(
                "modern v0 connection delta differs from the exact removed count"
            )
        checked_committed_hash = checked_candidate_hash = None
    elif estimand == V0_NORMALIZED_LOSS_SPAN_PER_CONNECTION_V1:
        assert connection_delta_receipt is not None
        # The digests checked here are the ones bound below; the
        # short-circuit order is that of a single ``or`` condition, and no
        # code between the check and the binding can touch the masks.
        receipt_mismatch = (
            connection_delta_receipt.total_removed_connections != removed
        )
        if not receipt_mismatch:
            checked_committed_hash = sampled_path_mask_hash(committed_masks)
            receipt_mismatch = (
                connection_delta_receipt.parent_mask_hash != checked_committed_hash
            )
        if not receipt_mismatch:
            checked_candidate_hash = sampled_path_mask_hash(candidate_masks)
            receipt_mismatch = (
                connection_delta_receipt.candidate_mask_hash != checked_candidate_hash
            )
        if receipt_mismatch:
            raise ValueError(
                "modern v0 connection receipt differs from the exact masks"
            )
    else:
        checked_committed_hash = checked_candidate_hash = None
    # A v5 decision on a backend whose evidence moments live on the
    # acquisition device runs its looks, statistics and control there.
    if (policy.resolved_decision_law_id == SEQUENTIAL_CALIBRATED_BCE_V5
            and forced_hook is None
            and estimand == V0_NORMALIZED_LOSS_SPAN_PER_CONNECTION_V1
            and getattr(backend, "device_bce_statistics_supported", False)
            and (callable(getattr(backend, "device_precision_options", None))
                 or not callable(getattr(backend, "maybe_refine_joint_precision", None)))):
        from .device_bce_decision import decide
        # Execution errors propagate with decision/look details. Consumed
        # draws and provider histories must never be replayed on the host.
        return decide(backend=backend,committed_masks=committed_masks,candidate_masks=candidate_masks,
            query_id=query_id,layer_id=layer_id,partition_id=partition_id,previous_density=previous_density,
            proposed_density=proposed_density,proposal_index=proposal_index,removed=removed,
            threshold=resolved_threshold,connection_delta_receipt=connection_delta_receipt,
            loss_span_certificate=loss_span_certificate,policy=policy,class_count=class_count,
            sequential_spec=sequential_spec,progress_callback=progress_callback,
            step_context_delta_sink=step_context_delta_sink,forced_action=forced_action,
            v6_record=v6_record,receipt_executor=receipt_executor)
    prior_panel: CrossfitPanel | None = None
    allocation_evidence: Mapping[str, Any] = {}
    panel_hashes: list[str] = []
    panel_receipts: list[Mapping[str, Any]] = []
    if v6_record and callable(
        getattr(backend, "evaluate_fixed_calibration_pair", None)
    ):
        # v6: a fixed-calibration backend's evidence never reads these (they
        # key the warm starts and ids of the non-fixed branch only), so the
        # kind-4 sampled-path digests are not computed.
        committed_hash = candidate_hash = None
    else:
        committed_hash = (
            sampled_path_mask_hash(committed_masks)
            if checked_committed_hash is None
            else checked_committed_hash
        )
        candidate_hash = (
            sampled_path_mask_hash(candidate_masks)
            if checked_candidate_hash is None
            else checked_candidate_hash
        )
    look_records: list[tuple[int, int, int, float, float, float]] = []
    target_images = 0
    actual_images = 0
    stagnant_looks = 0
    mean = 0.0
    lower, upper = -math.inf, math.inf
    action, reason = "unresolved", "evidence_horizon_exhausted_unresolved"
    looks = 0
    next_cumulative_target = packet if horizon is None else min(packet, horizon)
    current_standard_error = math.inf
    raw_current_interval_width = math.inf
    forecast_validation_count = 0
    pending_forecast_validation: tuple[int, float, int] | None = None
    latest_forecast_validation_ratio: float | None = None
    raw_stratum_statistics: dict[str, dict[str, float | int]] = {}
    allocation_receipts: list[Mapping[str, Any]] = []
    calibration_state_ids: tuple[str, ...] = ()
    calibration_target_loss_precision = None
    optimized_calibration_envelope: OptimizedCalibrationDecisionEnvelope | None = None
    joint_decision_envelope: JointDecisionConfidenceEnvelope | None = None
    optimized_numerator_lower = -math.inf
    optimized_numerator_upper = math.inf
    joint_precision_allocation_receipts: list[Mapping[str, Any]] = []
    observables_enabled = step_observables_enabled()
    latest_step_observables: dict[str, Any] = {}
    final_context_ids: tuple[str, ...] = ()
    final_raw_paired_bce_increase = np.asarray((), dtype=np.float64)
    # v4 decision-law state: refinement accounting, the operational
    # acquisition envelope, and the provider's finite-population disclosure.
    refinement_rounds = 0
    refined_since_last_look = False
    look_seconds_per_row: list[float] = []
    decision_started = time.perf_counter()
    envelope_seconds = (
        None
        if policy.oracle_acquisition_envelope_seconds is None
        else float(policy.oracle_acquisition_envelope_seconds)
    )
    envelope_receipt: dict[str, Any] | None = None
    census = _OracleCensusState.legacy()
    oracle_census_receipts: list[Mapping[str, Any]] = []
    sampling_with_replacement = True

    while precision_only_iid or (
        horizon is not None
        and planned_looks is not None
        and target_images < horizon
        and looks < planned_looks
    ):
        if envelope_seconds is not None:
            declined = _oracle_acquisition_envelope_exceeded(
                elapsed_seconds=time.perf_counter() - decision_started,
                envelope_seconds=envelope_seconds,
                look_seconds_per_row=look_seconds_per_row,
                requested_increment=max(
                    1,
                    int(next_cumulative_target) - int(actual_images),
                ),
            )
            if declined is not None:
                envelope_receipt = {
                    "schema": ORACLE_ACQUISITION_ENVELOPE_RECEIPT_V1,
                    **declined,
                    "look_index": int(looks),
                    "refinement_rounds": int(refinement_rounds),
                }
                action, reason = "unresolved", ORACLE_ENVELOPE_UNRESOLVED_REASON
                if progress_callback is not None:
                    progress_callback(
                        "v0_oracle_acquisition_envelope_declined",
                        query_id=str(query_id),
                        layer_id=str(layer_id),
                        partition_id=str(partition_id),
                        proposal_index=int(proposal_index),
                        **envelope_receipt,
                    )
                break
        joint_refiner = getattr(
            backend,
            "maybe_refine_joint_precision",
            None,
        )
        if callable(joint_refiner) and looks > 0:
            if loss_span_certificate is not None:
                latest_loss_span = getattr(
                    backend,
                    "current_loss_span_certificate",
                    None,
                )
                if isinstance(latest_loss_span, LossSpanCertificate):
                    latest_loss_span.validate_positive()
                    loss_span_certificate = latest_loss_span
            requested_increment = max(
                1,
                int(next_cumulative_target) - int(actual_images),
            )
            dense_reference_multiplier = 1.0
            if (
                estimand == V0_NORMALIZED_LOSS_SPAN_PER_CONNECTION_V1
                and loss_span_certificate is not None
                and connection_delta_receipt is not None
            ):
                # All candidate/parent/oracle reductions are forecast in raw
                # numerator-BCE units.  Convert a possible dense-span
                # contraction to the equivalent numerator-width contraction:
                # d(mu/A) / dA divided by d(mu/A) / dmu = |mu|/A.
                # Include the boundary-equivalent numerator so the forecast
                # remains conservative when the current estimate is near
                # zero but denominator uncertainty can still change the
                # threshold relation.
                boundary_numerator = (
                    float(resolved_threshold)
                    * float(loss_span_certificate.upper)
                    * float(connection_delta_receipt.total_removed_connections)
                )
                numerator_scale = max(
                    abs(float(optimized_numerator_lower)),
                    abs(float(optimized_numerator_upper)),
                    abs(boundary_numerator),
                    float(np.finfo(np.float64).eps),
                )
                dense_reference_multiplier = numerator_scale / float(
                    loss_span_certificate.lower
                )
            refinement = joint_refiner(
                committed_masks,
                candidate_masks,
                query_id=str(query_id),
                layer_id=str(layer_id),
                partition_id=str(partition_id),
                proposal_index=int(proposal_index),
                oracle_count=int(actual_images),
                oracle_interval_width=float(raw_current_interval_width),
                requested_oracle_increment=requested_increment,
                dense_reference_multiplier=dense_reference_multiplier,
                joint_radius_squared=(
                    None
                    if joint_decision_envelope is None
                    else float(joint_decision_envelope.radius_squared)
                ),
                **(
                    {"scalar_width_components": dict(
                        joint_decision_envelope.component_bounds
                    )}
                    if policy.resolved_decision_law_id == SEQUENTIAL_CALIBRATED_BCE_V5
                    and joint_decision_envelope is not None
                    else {}
                ),
            )
            if refinement is not None and bool(
                refinement.get(
                    "executed_precision_action",
                    refinement.get("executed_calibration", False),
                )
            ):
                joint_precision_allocation_receipts.append(dict(refinement))
                if progress_callback is not None:
                    progress_callback(
                        "joint_precision_acquisition",
                        **dict(refinement),
                    )
                latest_loss_span = getattr(
                    backend,
                    "current_loss_span_certificate",
                    None,
                )
                if isinstance(latest_loss_span, LossSpanCertificate):
                    latest_loss_span.validate_positive()
                    loss_span_certificate = latest_loss_span
                refinement_rounds += 1
                refined_since_last_look = True
                continue
        if (
            looks > 0
            and (census.census_complete or census.census_terminal_reached)
            and not refined_since_last_look
        ):
            # Every stratum is at its exact census and the ladder did not
            # fire on that evidence: the interval still straddles the
            # boundary with zero oracle sampling variance left to buy.  The
            # rule is never to point-estimate accept, so this is a
            # typed conservative no-commit.  (A refinement round since the
            # census look earns one re-projection look first; that look adds
            # no rows and is not stagnation.)
            action = "conservative_no_commit"
            reason = CENSUS_TERMINAL_NO_COMMIT_REASON
            if progress_callback is not None:
                progress_callback(
                    "v0_oracle_census_terminal_no_commit",
                    query_id=str(query_id),
                    layer_id=str(layer_id),
                    partition_id=str(partition_id),
                    proposal_index=int(proposal_index),
                    look_index=int(looks - 1),
                    panel_images=int(actual_images),
                    estimate=float(mean),
                    lower=float(lower),
                    upper=float(upper),
                    threshold=float(resolved_threshold),
                    oracle_census_strata_count=int(census.census_strata_count),
                    oracle_census_exact_weight=float(census.census_exact_weight),
                    refinement_rounds=int(refinement_rounds),
                )
            break
        remaining_group = math.inf
        if not precision_only_iid and policy.image_budget is not None:
            remaining_group = max(
                0.0,
                float(policy.image_budget)
                - float(getattr(backend, "decision_images_consumed", 0)),
            )
        maximum_target = (
            None
            if horizon is None
            else min(
                horizon,
                actual_images
                + (horizon if math.isinf(remaining_group) else int(remaining_group)),
            )
        )
        next_target = max(int(next_cumulative_target), target_images + 1)
        if maximum_target is not None:
            next_target = min(next_target, maximum_target)
        if next_target <= target_images:
            reason = "image_budget_exhausted_unresolved"
            break
        if next_target < 2 * int(policy.fold_count):
            reason = "image_budget_minimum_packet_unavailable"
            break
        target_images = int(next_target)
        looks += 1
        request = _bce_comparison_request(
            query_id=query_id,
            layer_id=layer_id,
            partition_id=partition_id,
            previous_density=previous_density,
            proposed_density=proposed_density,
            looks=looks,
            target_images=target_images,
            class_count=class_count,
            proposal_index=proposal_index,
            policy=policy,
            prior_panel=prior_panel,
            warm_starts=warm_starts,
            committed_hash=committed_hash,
            candidate_hash=candidate_hash,
            allocation_evidence=allocation_evidence,
            precision_only_iid=precision_only_iid,
            remaining_group=remaining_group,
            maximum_target=maximum_target,
            calibration_target_loss_precision=calibration_target_loss_precision,
        )
        look_started = __import__("time").perf_counter()
        if progress_callback is not None:
            progress_callback(
                "bce_oracle_panel_started",
                query_id=str(query_id),
                layer_id=str(layer_id),
                partition_id=str(partition_id),
                proposal_index=int(proposal_index),
                look_index=looks - 1,
                requested_cumulative_images=target_images,
                previous_panel_images=actual_images,
            )
        fixed_evaluator = getattr(
            backend,
            "evaluate_fixed_calibration_pair",
            None,
        )
        try:
            if callable(fixed_evaluator):
                evidence = fixed_evaluator(
                    committed_masks,
                    candidate_masks,
                    request,
                )
                if not isinstance(evidence, FixedCalibrationPairedBCEEvidence):
                    raise TypeError(
                        "fixed-calibration backend returned an unsupported evidence type"
                    )
            else:
                evidence = backend.evaluate_pair(
                    committed_masks,
                    candidate_masks,
                    request,
                )
        except OracleFinitePopulationCensusExhausted as error:
            census_receipt = {
                **error.receipt,
                "query_id": str(query_id),
                "layer_id": str(layer_id),
                "partition_id": str(partition_id),
                "proposal_index": int(proposal_index),
                "previous_density": float(previous_density),
                "proposed_density": float(proposed_density),
                "removed_route_count": int(removed),
                "look_index": int(looks - 1),
                "decision_action": "unresolved_no_commit",
            }
            oracle_census_receipts.append(census_receipt)
            # The refused look delivered no evidence and spent no alpha.
            looks -= 1
            action, reason = "unresolved", CENSUS_EXHAUSTED_UNRESOLVED_REASON
            if progress_callback is not None:
                progress_callback(
                    "v0_oracle_finite_population_census_exhausted",
                    **census_receipt,
                )
            break
        except CalibrationPrecisionHorizonExhausted as error:
            receipt = {
                **error.receipt,
                "query_id": str(query_id),
                "layer_id": str(layer_id),
                "partition_id": str(partition_id),
                "proposal_index": int(proposal_index),
                "previous_density": float(previous_density),
                "proposed_density": float(proposed_density),
                "removed_route_count": int(removed),
                "decision_action": "no_commit_and_bisect",
            }
            if progress_callback is not None:
                progress_callback(
                    "calibration_precision_horizon_exhausted_no_commit",
                    **receipt,
                )
            if v6_record:
                horizon_action = "unresolved"
                horizon_reason = "calibration_precision_horizon_exhausted_no_commit"
                natural_horizon_action = horizon_action
                if forced_hook is not None:
                    horizon_action, horizon_reason = hook_exit(
                        "calibration_precision_horizon", horizon_action, horizon_reason
                    )
                elif forced_action is not None:
                    horizon_action = forced_action.action
                    horizon_reason = forced_action.reason or horizon_reason
                v6_decision = _v6_decision_record(
                    query_id=query_id,
                    layer_id=layer_id,
                    partition_id=partition_id,
                    proposal_index=proposal_index,
                    previous_density=previous_density,
                    proposed_density=proposed_density,
                    action=horizon_action,
                    natural_action=natural_horizon_action,
                    engine_reason="calibration_precision_horizon_exhausted_no_commit",
                    engine_forecast_target=None,
                    reason=horizon_reason,
                    looks=0,
                    panel_images=0,
                    mean=0.0,
                    lower=-math.inf,
                    upper=math.inf,
                    threshold=resolved_threshold,
                    removed=removed,
                    alpha_allocated=float(policy.rule.error_probability),
                    suppressed=True,
                    forced=forced_hook is not None or forced_action is not None,
                    replication=False,
                    per_look=(),
                    candidate_masks=candidate_masks,
                )
                return _V6BCEStepEvidence(
                    action=horizon_action,
                    reason=horizon_reason,
                    looks=0,
                    panel_images=0,
                    mean=0.0,
                    lower=-math.inf,
                    upper=math.inf,
                    panel_hashes=(),
                    panel_receipts=(),
                    calibration_precision_receipts=(receipt,),
                    connection_delta_receipt=(
                        None
                        if connection_delta_receipt is None
                        else connection_delta_receipt.receipt
                    ),
                    decision_receipt=_v6_decision_reference(v6_decision),
                    decision_record=v6_decision,
                )
            return _BCEStepEvidence(
                action="unresolved",
                reason="calibration_precision_horizon_exhausted_no_commit",
                looks=0,
                panel_images=0,
                mean=0.0,
                lower=-math.inf,
                upper=math.inf,
                panel_hashes=(),
                panel_receipts=(),
                calibration_precision_receipts=(receipt,),
                connection_delta_receipt=(
                    None
                    if connection_delta_receipt is None
                    else connection_delta_receipt.receipt
                ),
                decision_receipt=(
                    {
                        "schema": ("sequential_bce_decision_receipt_suppressed_v1"),
                        "suppressed": True,
                        "proposal_id": (
                            f"{query_id}:{layer_id}:{partition_id}:"
                            f"{proposal_index}:{candidate_hash}"
                        ),
                        "estimand_policy": estimand,
                        "action": "unresolved",
                        "reason": ("calibration_precision_horizon_exhausted_no_commit"),
                        "failed_conditions": ("at_least_one_usable_look",),
                        "looks": 0,
                        **(
                            {}
                            if observables_enabled
                            else {"step_observables_enabled": False}
                        ),
                    }
                    if estimand == V0_NORMALIZED_LOSS_SPAN_PER_CONNECTION_V1
                    else None
                ),
            )
        look_alpha = sequential_spec.look_alpha(
            looks - 1,
            error_probability=proposal_error_probability,
        )
        evidence.validate(fold_count=policy.fold_count)
        if prior_panel is not None:
            evidence.panel.assert_nested_extension(prior_panel)
        if not v6_record:
            # v6: the panel digest is kind 4 and is not requested.
            panel_hashes.append(
                str(
                    (evidence.panel.panel_axis_chain or {}).get("hash")
                    or evidence.panel.draw_id
                )
            )
        previous_actual = actual_images
        actual_images = len(evidence.panel.context_ids)
        sampling_with_replacement = bool(
            getattr(backend, "sampling_with_replacement", False)
            or evidence.panel.allocation_metadata.get(
                "sampling_with_replacement",
                False,
            )
        )
        census = _oracle_census_state(
            evidence.panel.allocation_metadata,
            sampling_with_replacement=sampling_with_replacement,
        )
        refined_since_last_look = False
        stratified_fpc_applied = False
        if not census.census_terminal_reached:
            stagnant_looks = (
                stagnant_looks + 1 if actual_images <= previous_actual else 0
            )
        # else: a census-terminal look adds no rows by construction; it is the
        # exact terminal state of the finite population, not stagnation.
        (
            bounded_iid,
            calibration_state_ids,
            current_standard_error,
            final_context_ids,
            final_raw_paired_bce_increase,
            joint_decision_envelope,
            latest_step_observables,
            lower,
            mean,
            optimized_calibration_envelope,
            optimized_numerator_lower,
            optimized_numerator_upper,
            raw_current_interval_width,
            look_raw_lower,
            look_raw_mean,
            look_raw_standard_error,
            raw_stratum_statistics,
            look_raw_upper,
            strata,
            stratified_fpc_applied,
            stratum_weights,
            upper,
        ) = _look_statistics(
            candidate_hash=candidate_hash,
            census=census,
            committed_hash=committed_hash,
            estimand=estimand,
            evidence=evidence,
            joint_decision_envelope=joint_decision_envelope,
            joint_precision_allocation_receipts=joint_precision_allocation_receipts,
            latest_step_observables=latest_step_observables,
            look_alpha=look_alpha,
            looks=looks,
            loss_span_certificate=loss_span_certificate,
            observables_enabled=observables_enabled,
            optimized_calibration_envelope=optimized_calibration_envelope,
            optimized_numerator_lower=optimized_numerator_lower,
            optimized_numerator_upper=optimized_numerator_upper,
            policy=policy,
            raw_current_interval_width=raw_current_interval_width,
            raw_stratum_statistics=raw_stratum_statistics,
            removed=removed,
            sampling_with_replacement=sampling_with_replacement,
            stratified_fpc_applied=stratified_fpc_applied,
            warm_starts=warm_starts,
            threshold=resolved_threshold,
        )
        if look_raw_lower is not _LOOK_UNASSIGNED:
            raw_lower = look_raw_lower
        if look_raw_mean is not _LOOK_UNASSIGNED:
            raw_mean = look_raw_mean
        if look_raw_standard_error is not _LOOK_UNASSIGNED:
            raw_standard_error = look_raw_standard_error
        if look_raw_upper is not _LOOK_UNASSIGNED:
            raw_upper = look_raw_upper
        allocation_rule = str(
            evidence.panel.allocation_metadata.get("allocation_rule", "")
        )
        if allocation_rule == "cost_optimal_two_stage_neyman":
            from .neyman import summarize_hierarchical_paired_risk

            allocation_evidence = summarize_hierarchical_paired_risk(
                evidence
            ).to_allocation_state()
        prior_panel = evidence.panel
        look_records.append(
            (
                int(looks - 1),
                int(target_images),
                int(actual_images),
                float(mean),
                float(lower),
                float(upper),
            )
        )
        panel_receipt = (
            _v6_panel_receipt(
                query_id=query_id,
                layer_id=layer_id,
                partition_id=partition_id,
                proposal_index=proposal_index,
                look_index=looks - 1,
                target_images=target_images,
                actual_images=actual_images,
                previous_actual=previous_actual,
                sampling_with_replacement=sampling_with_replacement,
                fpc_applied=bool(
                    (policy.finite_population_correction and not sampling_with_replacement)
                    or stratified_fpc_applied
                ),
                census=census,
                policy=policy,
                look_started=look_started,
                interval_method=(
                    BCE_PRECISION_BOUNDED_INTERVAL_METHOD
                    if bounded_iid
                    else (
                        (
                            STRATIFIED_T_FPC_ALPHA_SPENDING_METHOD
                            if precision_only_iid
                            else STRATIFIED_T_FPC_METHOD
                        )
                        if stratified_fpc_applied
                        else (
                            STRATIFIED_T_METHOD
                            if strata and stratum_weights
                            else BCE_PRECISION_INTERVAL_METHOD
                        )
                    )
                ),
            )
            if v6_record
            else None
        )
        if panel_receipt is None:
            panel_receipt = {
                "schema": "bce_oracle_sampled_panel_timing_receipt_v1",
                "query_id": str(query_id),
                "layer_id": str(layer_id),
                "partition_id": str(partition_id),
                "proposal_index": int(proposal_index),
                "look_index": looks - 1,
                "requested_cumulative_images": target_images,
                "delivered_cumulative_images": actual_images,
                "new_context_count": actual_images - previous_actual,
                "panel_hash": panel_hashes[-1],
                "allocation_metadata": _compact_panel_allocation_metadata(
                    evidence.panel.allocation_metadata
                ),
                "sampling_with_replacement": sampling_with_replacement,
                # Applied FPC follows the sampling law (exact per-stratum
                # correction on sealed without-replacement panels); the legacy
                # unstratified knob keeps its historical meaning and is
                # disclosed separately below on contract panels.
                "finite_population_correction_applied": bool(
                    (policy.finite_population_correction and not sampling_with_replacement)
                    or stratified_fpc_applied
                ),
                "interval_method": (
                    BCE_PRECISION_BOUNDED_INTERVAL_METHOD
                    if bounded_iid
                    else (
                        (
                            STRATIFIED_T_FPC_ALPHA_SPENDING_METHOD
                            if precision_only_iid
                            else STRATIFIED_T_FPC_METHOD
                        )
                        if stratified_fpc_applied
                        else (
                            STRATIFIED_T_METHOD
                            if strata and stratum_weights
                            else BCE_PRECISION_INTERVAL_METHOD
                        )
                    )
                ),
                **(
                    {
                        "finite_population_correction_policy": bool(
                            policy.finite_population_correction
                        ),
                        **census.receipt_fields(),
                    }
                    if census.contract_present
                    else {}
                ),
                "elapsed_seconds": float(__import__("time").perf_counter() - look_started),
            }
        panel_receipts.append(panel_receipt)
        if actual_images > previous_actual:
            look_seconds_per_row.append(
                float(panel_receipt["elapsed_seconds"])
                / float(actual_images - previous_actual)
            )
        allocation_receipt = (
            None
            if v6_record
            else evidence.panel.allocation_metadata.get("oracle_allocation_receipt")
        )
        if isinstance(allocation_receipt, Mapping):
            allocation_receipts.append(dict(allocation_receipt))
        if progress_callback is not None:
            progress_callback(
                "bce_oracle_panel_completed",
                **panel_receipt,
            )

        if (
            # Base v5 scalar rule: the signed upper bound must be strictly
            # negative; an exactly zero bound does not accept.
            joint_decision_envelope.statistic_upper < 0.0
            if isinstance(joint_decision_envelope, ScalarV0DecisionEnvelope)
            else upper <= resolved_threshold
        ):
            action = "accept"
            reason = (
                "confidence_interval_wholly_at_or_below_tau"
                if estimand == "bce_increment_per_removed_route"
                else "confidence_interval_wholly_at_or_below_boundary"
            )
            action, reason, forced_target = hook_look(action, reason, None)
            if hook_terminal:
                break
            next_cumulative_target = int(forced_target)
            continue
        if (
            joint_decision_envelope.statistic_lower > 0.0
            if isinstance(joint_decision_envelope, ScalarV0DecisionEnvelope)
            else lower > resolved_threshold
        ):
            action = "reject"
            reason = (
                "confidence_interval_wholly_above_tau"
                if estimand == "bce_increment_per_removed_route"
                else "confidence_interval_wholly_above_boundary"
            )
            action, reason, forced_target = hook_look(action, reason, None)
            if hook_terminal:
                break
            next_cumulative_target = int(forced_target)
            continue
        if (
            estimand == V0_NORMALIZED_LOSS_SPAN_PER_CONNECTION_V1
            and (
                joint_decision_envelope.statistic_estimate >= 0.0
                if isinstance(joint_decision_envelope, ScalarV0DecisionEnvelope)
                else mean >= resolved_threshold
            )
        ):
            # Modern v0 needs confidence to *commit* a destructive action,
            # not to retain the parent mask.  When the current best estimate
            # is already on the unsafe side, launching an arbitrarily large
            # look merely to prove a conservative no-commit decision has no
            # scientific benefit.  Return a typed non-claiming outcome so
            # the topology may bisect the connection action.  Accepted masks
            # remain protected by the unchanged one-sided confidence bound.
            action = "conservative_no_commit"
            reason = "point_estimate_at_or_above_boundary_no_commit"
            action, reason, forced_target = hook_look(action, reason, None)
            if hook_terminal:
                break
            next_cumulative_target = int(forced_target)
            continue
        if stagnant_looks >= 2:
            reason = "panel_growth_exhausted_unresolved"
            action, reason, forced_target = hook_look(action, reason, None)
            if not hook_terminal and action == "unresolved" and forced_target is not None:
                next_cumulative_target = forced_target
                continue
            break
        radius = max(float(mean) - float(lower), float(upper) - float(mean))
        if pending_forecast_validation is not None:
            (
                forecast_origin_count,
                forecast_origin_radius,
                forecast_target_count,
            ) = pending_forecast_validation
            predicted_radius = float(forecast_origin_radius) * math.sqrt(
                float(forecast_origin_count) / float(max(actual_images, 1))
            )
            latest_forecast_validation_ratio = (
                math.inf
                if predicted_radius <= 0.0
                else float(radius) / predicted_radius
            )
            if (
                actual_images >= forecast_target_count
                and math.isfinite(latest_forecast_validation_ratio)
                and 0.5 <= latest_forecast_validation_ratio <= 2.0
            ):
                forecast_validation_count += 1
            else:
                forecast_validation_count = 0
            pending_forecast_validation = None
        analytic_cumulative_target = forecast_cumulative_target(
            current_count=max(actual_images, 1),
            current_standard_error=float(current_standard_error),
            estimate=float(mean),
            accept_boundary=resolved_threshold,
            # One boundary: accept == reject, so the forecast's indifference
            # terminal is live exactly when the two ever differ.
            reject_boundary=resolved_threshold,
            current_radius=float(radius),
            minimum_increment=packet,
            maximum_count=maximum_target,
        )
        if analytic_cumulative_target <= actual_images:
            reason = "predicted_sample_horizon_exhausted_unresolved"
            action, reason, forced_target = hook_look(action, reason, None)
            if not hook_terminal and action == "unresolved" and forced_target is not None:
                next_cumulative_target = forced_target
                continue
            break
        (
            next_cumulative_target,
            forecast_growth_guard_applied,
        ) = guard_unvalidated_cumulative_forecast(
            current_count=max(actual_images, 1),
            analytic_target=int(analytic_cumulative_target),
            minimum_increment=packet,
            # A validated 1/sqrt(n) radius contraction does not validate an
            # orders-of-magnitude extrapolation of the *distance to the
            # boundary*.  Keep every physical acquisition at most fourfold
            # and re-evaluate the actual estimate between looks.  This is a
            # growth safeguard only: it imposes neither a sample ceiling nor
            # a runtime horizon.
            #
            # This hardcoded False is DELIBERATE and is the same-quantity rule
            # stated on ``validate_forecast_ratio``: the instrument below
            # measures the realized interval RADIUS, but the analytic target
            # extrapolates the distance to the boundary, which is random and
            # sits squared in the denominator. The sibling calibration
            # forecast does feed its flag back precisely because there the
            # predicted and realized quantities coincide. The ratio computed
            # below is telemetry, not a control input -- do not wire it here.
            forecast_validated=False,
        )
        forecast_population_clamp_applied = False
        if census.remaining_population is not None:
            # Sealed without-replacement: the finite population is the hard
            # ceiling on what the next look can deliver.
            population_ceiling = int(actual_images) + int(census.remaining_population)
            if int(next_cumulative_target) > population_ceiling:
                next_cumulative_target = population_ceiling
                forecast_population_clamp_applied = True
        pending_forecast_validation = (
            int(actual_images),
            float(radius),
            int(next_cumulative_target),
        )
        action, reason, forced_target = hook_look(
            "unresolved", reason, int(next_cumulative_target)
        )
        if hook_terminal:
            break
        next_cumulative_target = int(forced_target)
        if progress_callback is not None:
            progress_callback(
                "v0_decision_precision_forecast",
                query_id=str(query_id),
                layer_id=str(layer_id),
                partition_id=str(partition_id),
                proposal_index=int(proposal_index),
                look_index=int(looks - 1),
                current_context_count=int(actual_images),
                forecast_cumulative_context_count=int(next_cumulative_target),
                analytic_forecast_cumulative_context_count=int(
                    analytic_cumulative_target
                ),
                forecast_incremental_context_count=int(
                    next_cumulative_target - actual_images
                ),
                forecast_growth_guard_applied=bool(forecast_growth_guard_applied),
                forecast_validation_count=int(forecast_validation_count),
                latest_forecast_validation_ratio=(
                    None
                    if latest_forecast_validation_ratio is None
                    else float(latest_forecast_validation_ratio)
                ),
                estimate=float(mean),
                lower=float(lower),
                upper=float(upper),
                threshold=float(resolved_threshold),
                standard_error=float(current_standard_error),
                raw_paired_bce_estimate=float(raw_mean),
                raw_paired_bce_lower=float(raw_lower),
                raw_paired_bce_upper=float(raw_upper),
                raw_paired_bce_standard_error=float(raw_standard_error),
                raw_paired_bce_interval_width=float(raw_current_interval_width),
                connection_delta=int(removed),
                loss_span_estimate=(
                    None
                    if loss_span_certificate is None
                    else float(loss_span_certificate.estimate)
                ),
                joint_component_bounds=(
                    None
                    if joint_decision_envelope is None
                    else dict(joint_decision_envelope.component_bounds)
                ),
                **(
                    {
                        "forecast_population_clamp_applied": bool(
                            forecast_population_clamp_applied
                        ),
                        "oracle_remaining_population": int(
                            census.remaining_population
                        ),
                    }
                    if census.remaining_population is not None
                    else {}
                ),
            )
    if (
        planned_looks is not None
        and looks >= planned_looks
        and action == "unresolved"
        and lower <= resolved_threshold
    ):
        reason = "planned_look_horizon_exhausted_unresolved"

    # Forced hook: each non-look terminal passes through exit before a receipt.
    if forced_hook is not None and not hook_terminal:
        action, reason = hook_exit("decision_terminal", action, reason)

    accept_replication_audit: Mapping[str, Any] | None = None
    if (
        action == "accept"
        and estimand == V0_NORMALIZED_LOSS_SPAN_PER_CONNECTION_V1
        and bool(getattr(policy, "accept_replication_gate", False))
    ):
        confirm_hook = getattr(
            backend,
            "confirm_accept_replication",
            None,
        )
        if callable(confirm_hook):
            confirmation = confirm_hook(
                committed_masks,
                candidate_masks,
                query_id=str(query_id),
                layer_id=str(layer_id),
                partition_id=str(partition_id),
                proposal_index=int(proposal_index),
                confidence_level=float(policy.rule.confidence_level),
            )
            confirm_sensitivity = float(confirmation["confirm_mean_sensitivity"])
            replicated = confirm_sensitivity <= resolved_threshold
            accept_replication_audit = {
                **dict(confirmation),
                "select_action": "accept",
                "replicated": bool(replicated),
                "accept_boundary": float(resolved_threshold),
            }
            if not replicated:
                # The select stream cleared the boundary but the disjoint
                # confirm stream did not: demote the accept to an
                # uncommitted step. Winner's-curse accepts are filtered
                # here; the density machinery then bisects or freezes
                # exactly as for any uncleared interval.
                action = "unresolved"
                reason = "accept_failed_confirm_replication"
            # Confirmation changes the natural reference only when the
            # unforced decision itself was an accept. A forced accept can
            # enter this gate, but cannot rewrite a natural reject or reason.
            if engine_outcome == "accept":
                if not replicated:
                    engine_outcome = "unresolved"
                    engine_reason = "accept_failed_confirm_replication"
            if forced_hook is not None:
                natural_replication = {
                    "replicated": bool(replicated),
                    "outcome": str(action),
                    "reason": str(reason),
                    "next_cumulative_target": None,
                    "actual_images": int(actual_images),
                }
                forced_replication = forced_hook.replication(
                    str(query_id), str(layer_id), str(partition_id),
                    int(proposal_index), int(looks), natural_replication,
                )
                if not isinstance(forced_replication, Mapping) or not isinstance(
                    forced_replication.get("replicated"), bool
                ):
                    raise RuntimeError("forced-action replication returned no boolean")
                if bool(forced_replication["replicated"]) != bool(replicated):
                    action = "accept" if forced_replication["replicated"] else "unresolved"
                    reason = str(forced_replication.get("reason", reason))
            if progress_callback is not None:
                progress_callback(
                    "v0_accept_confirm_replication_completed",
                    query_id=str(query_id),
                    layer_id=str(layer_id),
                    partition_id=str(partition_id),
                    proposal_index=int(proposal_index),
                    replicated=bool(replicated),
                    confirm_mean_sensitivity=confirm_sensitivity,
                    accept_boundary=float(resolved_threshold),
                    confirm_row_count=int(confirmation["confirm_row_count"]),
                )
    natural_action = engine_outcome if forced_hook is not None else action
    if forced_action is not None and forced_hook is None:
        # Acceptance hook: every look's statistics are recorded above;
        # only the action (and its reason when logged) is replaced.
        action = str(forced_action.action)
        reason = str(forced_action.reason) if forced_action.reason else reason
    if v6_record:
        v6_decision = _v6_decision_record(
            query_id=query_id,
            layer_id=layer_id,
            partition_id=partition_id,
            proposal_index=proposal_index,
            previous_density=previous_density,
            proposed_density=proposed_density,
            action=action,
            natural_action=natural_action,
            engine_reason=(engine_reason if forced_hook is not None else reason),
            engine_forecast_target=(engine_forecast_target if forced_hook is not None else None),
            reason=reason,
            looks=looks,
            panel_images=actual_images,
            mean=mean,
            lower=lower,
            upper=upper,
            threshold=resolved_threshold,
            removed=removed,
            alpha_allocated=float(policy.rule.error_probability),
            suppressed=bool(
                estimand == V0_NORMALIZED_LOSS_SPAN_PER_CONNECTION_V1
                and (
                    connection_delta_receipt is None
                    or loss_span_certificate is None
                    or looks <= 0
                    or not all(math.isfinite(value) for value in (mean, lower, upper))
                )
            ),
            forced=forced_hook is not None or forced_action is not None,
            replication=accept_replication_audit is not None,
            per_look=tuple(look_records),
            candidate_masks=candidate_masks,
        )
        if (
            observables_enabled
            and step_context_delta_sink is not None
            and final_context_ids
        ):
            rows_available = int(final_raw_paired_bce_increase.size) == len(
                final_context_ids
            )
            step_context_delta_sink(
                proposal_index=int(proposal_index),
                layer_id=str(layer_id),
                partition_id=str(partition_id),
                action=str(action),
                context_ids=(final_context_ids if rows_available else ()),
                paired_bce_delta=np.asarray(
                    (final_raw_paired_bce_increase if rows_available else ()),
                    dtype=np.float32,
                ),
                unavailable_context_count=(
                    0 if rows_available else len(final_context_ids)
                ),
            )
        # The per-decision resource-usage delta (two content-addressed
        # checkpoints) fed only the JSON receipt: kind 4, not computed.
        return _V6BCEStepEvidence(
            action=action,
            reason=reason,
            looks=looks,
            panel_images=actual_images,
            mean=float(mean),
            lower=float(lower),
            upper=float(upper),
            panel_hashes=(),
            panel_receipts=tuple(panel_receipts),
            # The exact connection counts stay (a statistic input); only the
            # digests are gone (``ConnectionDelta.receipt``).
            connection_delta_receipt=(
                None
                if connection_delta_receipt is None
                else connection_delta_receipt.receipt
            ),
            decision_receipt=_v6_decision_reference(v6_decision),
            oracle_census_receipts=tuple(oracle_census_receipts),
            decision_record=v6_decision,
        )
    sequential_receipt = None
    receipt_suppression_conditions: tuple[str, ...] = ()
    decision_resource_usage: Mapping[str, Any] = {}
    if estimand == V0_NORMALIZED_LOSS_SPAN_PER_CONNECTION_V1:
        # Advance exactly once for every decision, including typed suppressed
        # receipts.  Otherwise the next successful receipt would silently
        # absorb the suppressed decision's executions into its delta.
        decision_resource_usage = _decision_resource_usage(backend)
        receipt_suppression_conditions = tuple(
            name
            for name, passed in (
                (
                    "connection_delta_receipt_present",
                    connection_delta_receipt is not None,
                ),
                (
                    "loss_span_certificate_present",
                    loss_span_certificate is not None,
                ),
                ("at_least_one_usable_look", looks > 0),
                (
                    "finite_decision_statistics",
                    all(math.isfinite(value) for value in (mean, lower, upper)),
                ),
            )
            if not passed
        )
        if receipt_suppression_conditions:
            # A silent None here once erased the entire per-decision receipt
            # record of a completed 145-decision traversal. Every modern-v0
            # decision now emits either a full receipt or this typed
            # suppression naming exactly which precondition failed.
            sequential_receipt = {
                "schema": "sequential_bce_decision_receipt_suppressed_v1",
                "suppressed": True,
                "proposal_id": (
                    f"{query_id}:{layer_id}:{partition_id}:"
                    f"{proposal_index}:{candidate_hash}"
                ),
                "estimand_policy": V0_NORMALIZED_LOSS_SPAN_PER_CONNECTION_V1,
                "action": str(action),
                "reason": str(reason),
                "failed_conditions": receipt_suppression_conditions,
                "looks": int(looks),
                "resource_usage": decision_resource_usage,
                **({} if observables_enabled else {"step_observables_enabled": False}),
            }
    if (
        estimand == V0_NORMALIZED_LOSS_SPAN_PER_CONNECTION_V1
        and not receipt_suppression_conditions
    ):
        sequential_receipt = _build_decision_receipt(
            accept_replication_audit=accept_replication_audit,
            action=action,
            allocation_receipts=allocation_receipts,
            calibration_state_ids=calibration_state_ids,
            candidate_hash=candidate_hash,
            census=census,
            connection_delta_receipt=connection_delta_receipt,
            decision_resource_usage=decision_resource_usage,
            envelope_receipt=envelope_receipt,
            joint_decision_envelope=joint_decision_envelope,
            joint_precision_allocation_receipts=joint_precision_allocation_receipts,
            latest_step_observables=latest_step_observables,
            layer_id=layer_id,
            loss_span_certificate=loss_span_certificate,
            lower=lower,
            mean=mean,
            observables_enabled=observables_enabled,
            optimized_calibration_envelope=optimized_calibration_envelope,
            partition_id=partition_id,
            proposal_index=proposal_index,
            query_id=query_id,
            raw_lower=raw_lower,
            raw_mean=raw_mean,
            raw_stratum_statistics=raw_stratum_statistics,
            raw_upper=raw_upper,
            refinement_rounds=refinement_rounds,
            removed=removed,
            resolved_threshold=resolved_threshold,
            sequential_spec=sequential_spec,
            upper=upper,
        )
    if (
        observables_enabled
        and step_context_delta_sink is not None
        and final_context_ids
    ):
        rows_available = int(final_raw_paired_bce_increase.size) == len(
            final_context_ids
        )
        step_context_delta_sink(
            proposal_index=int(proposal_index),
            layer_id=str(layer_id),
            partition_id=str(partition_id),
            action=str(action),
            context_ids=(final_context_ids if rows_available else ()),
            paired_bce_delta=np.asarray(
                (final_raw_paired_bce_increase if rows_available else ()),
                dtype=np.float32,
            ),
            unavailable_context_count=(0 if rows_available else len(final_context_ids)),
        )
    return _BCEStepEvidence(
        action=action,
        reason=reason,
        looks=looks,
        panel_images=actual_images,
        mean=float(mean),
        lower=float(lower),
        upper=float(upper),
        panel_hashes=tuple(panel_hashes),
        panel_receipts=tuple(panel_receipts),
        connection_delta_receipt=(
            None
            if connection_delta_receipt is None
            else connection_delta_receipt.receipt
        ),
        decision_receipt=sequential_receipt,
        oracle_census_receipts=tuple(oracle_census_receipts),
    )


@dataclass(frozen=True)
class _FrontierCursor:
    masks: Any
    densities: Mapping[str, float]
    steps: Mapping[str, float]
    schedule_indices: Mapping[str, int]
    frozen_partitions: frozenset[str]
    turn: str
    proposal_index: int = 0
    accepted_count: int = 0
    stalled_turns: int = 0
    increment_spent: float = 0.0
    stopping_reason: str = "frontier_resolved_in_both_partitions"

    def key(self):
        return (
            self.masks.kept_key, self.masks.cached_mask_hash(),
            tuple(self.densities.items()), tuple(self.steps.items()),
            tuple(self.schedule_indices.items()), self.frozen_partitions,
            self.turn, self.proposal_index, self.accepted_count, self.stalled_turns,
            self.increment_spent, self.stopping_reason,
        )


@dataclass(frozen=True)
class _FrontierProposal:
    cursor: _FrontierCursor
    candidate: Any
    proposed_density: float
    removed: int


def _frontier_step_to_next_proposal(
    cursor: _FrontierCursor, *, build_candidate, policy, active,
    density_schedule, v6_views, modern_v0_mode, aggregate_boundary_mode,
    maximum_decisions, maximum_accepted_decisions, images_consumed,
    completed: _FrontierProposal | None = None, action: str | None = None,
    upper: float = 0.0, preferred: _FrontierProposal | None = None,
) -> tuple[_FrontierCursor, _FrontierProposal | None]:
    """Pure frontier transition and proposal selection, including every skip.

    The serial loop and both host descriptions use this function. Memo tables
    belong to the returned masks and views; filling them cannot alter inputs.
    Providers, draws, outcome records, alpha spending and images are absent.
    """

    masks = cursor.masks.fork() if v6_views else {
        str(k): np.asarray(v, dtype=bool).copy() for k, v in cursor.masks.items()
    }
    densities, steps = dict(cursor.densities), dict(cursor.steps)
    indices, frozen = dict(cursor.schedule_indices), set(cursor.frozen_partitions)
    turn, proposal_index = cursor.turn, cursor.proposal_index
    accepted, stalled = cursor.accepted_count, cursor.stalled_turns
    spent, reason = cursor.increment_spent, cursor.stopping_reason
    if completed is not None:
        if action is None:
            raise ValueError("a completed frontier proposal requires its action")
        proposal_index += 1
        if action == "accept":
            masks = masks.accept(completed.candidate.fork(parent=masks)) if v6_views else {
                str(k): np.asarray(v, dtype=bool).copy() for k, v in completed.candidate.items()
            }
            densities[turn] = completed.proposed_density
            if density_schedule is not None:
                indices[turn] += 1
            else:
                steps[turn] = min(2.0 * steps[turn], max(densities[turn] - float(policy.minimum_density), 0.0))
            accepted += 1
            stalled = 0
            if aggregate_boundary_mode:
                spent += max(float(upper), 0.0)
            if maximum_accepted_decisions is not None and accepted >= int(maximum_accepted_decisions):
                reason = "maximum_accepted_decisions_reached"
        elif action in {"reject", "conservative_no_commit"}:
            if density_schedule is not None:
                stalled += 1
            else:
                steps[turn] /= 2.0
        elif modern_v0_mode:
            frozen.add(turn)
            steps[turn] = 0.0
            reason = "unresolved_partition_frozen_no_commit"
        else:
            steps[turn] /= 2.0
        turn = "negative" if turn == "positive" else "positive"

    def snapshot():
        return _FrontierCursor(masks, dict(densities), dict(steps), dict(indices),
                               frozenset(frozen), turn, proposal_index, accepted, stalled, spent, reason)

    def done(partition):
        return (
            partition not in active or partition in frozen
            or (density_schedule is not None and indices[partition] >= len(density_schedule) - 1)
            or steps[partition] < float(policy.minimum_density_step)
            or densities[partition] <= float(policy.minimum_density) + 1e-12
        )

    while reason != "maximum_accepted_decisions_reached":
        if maximum_decisions is not None and proposal_index >= int(maximum_decisions):
            reason = "maximum_decisions_reached"
            break
        if all(done(partition) for partition in densities):
            if density_schedule is not None and reason == "frontier_resolved_in_both_partitions":
                reason = "density_schedule_exhausted"
            break
        if density_schedule is not None and stalled >= 2:
            reason = "both_partitions_stalled"
            break
        if policy.image_budget is not None and float(images_consumed) >= float(policy.image_budget):
            reason = "image_budget_exhausted"
            break
        if done(turn):
            if density_schedule is not None:
                stalled += 1
            turn = "negative" if turn == "positive" else "positive"
            continue
        proposed = (
            float(density_schedule[indices[turn] + 1]) if density_schedule is not None
            else max(densities[turn] - steps[turn], float(policy.minimum_density))
        )
        current = snapshot()
        if preferred is not None and v6_views and preferred.cursor.key() == current.key():
            return preferred.cursor, preferred
        kwargs = dict(
            partition_id=turn,
            positive_density=proposed if turn == "positive" else densities["positive"],
            negative_density=proposed if turn == "negative" else densities["negative"],
        )
        candidate = build_candidate.build_candidate_view(masks, **kwargs) if v6_views else build_candidate(masks, **kwargs)
        removed = int(candidate.removed_count) if v6_views else _mask_hamming_distance(masks, candidate)
        if removed > 0:
            return current, _FrontierProposal(current, candidate, proposed, removed)
        if density_schedule is not None:
            indices[turn] += 1
            densities[turn] = proposed
        else:
            steps[turn] = 0.0
        turn = "negative" if turn == "positive" else "positive"
    return snapshot(), None


def trace_bce_precision_frontier(
    *,
    backend: GlocalSampledPathEvaluationBackend,
    committed_masks: Mapping[str, np.ndarray],
    build_candidate: Any,
    query_id: str,
    layer_id: str,
    total_layer_route_count: int,
    policy: BCEPrecisionPolicy,
    class_count: int,
    context: NaturalizationContext | None = None,
    initial_turn: str = "negative",
    maximum_decisions: int | None = None,
    maximum_accepted_decisions: int | None = None,
    active_partitions: tuple[str, ...] = ("positive", "negative"),
    executable_density_schedule: tuple[float, ...] | None = None,
    loss_span_certificate: LossSpanCertificate | None = None,
    route_semantics_by_layer: Mapping[str, str] | None = None,
    progress_callback: Callable[..., None] | None = None,
    step_context_delta_sink: Callable[..., None] | None = None,
) -> ContinuousEvidenceResult:
    """Run sign-partitioned adaptive bisection under the BCE precision rule."""

    if context is not None:
        policy = policy.resolve(context)
    policy.validate()
    aggregate_boundary_mode = isinstance(policy.rule, BCELossBoundaryDecisionRule)
    direct_candidate_mode = isinstance(
        policy.rule,
        DirectCandidateIncrementDecisionRule,
    )
    modern_v0_mode = isinstance(policy.rule, V0MarginalConnectionDecisionRule)
    if aggregate_boundary_mode:
        estimand = "aggregate_complete_mask_bce_increment"
        total_increment_budget = float(policy.rule.maximum_bce_increment)
    elif direct_candidate_mode:
        estimand = "direct_candidate_bce_increment"
        total_increment_budget = 0.0
    elif modern_v0_mode:
        estimand = V0_NORMALIZED_LOSS_SPAN_PER_CONNECTION_V1
        total_increment_budget = 0.0
        if loss_span_certificate is None:
            raise ValueError("modern v0 requires a loss-span certificate")
        loss_span_certificate.validate_positive()
        sequential_spec = SequentialCalibratedBCEResolutionSpec(
            confidence_level=float(policy.rule.confidence_level),
            estimand_policy=V0_NORMALIZED_LOSS_SPAN_PER_CONNECTION_V1,
            maximum_planned_looks=policy.maximum_planned_looks,
            precision_only_iid=bool(policy.precision_only_iid),
            decision_interval_method=(
                SCALAR_DECISION_FIELLER_METHOD
                if policy.resolved_decision_law_id == SEQUENTIAL_CALIBRATED_BCE_V5
                else JOINT_DECISION_WALD_METHOD
            ),
        )
        if (
            min(
                float(loss_span_certificate.dense.confidence_level),
                float(loss_span_certificate.fully_pruned.confidence_level),
            )
            + 1e-15
            < sequential_spec.v0_anchor_confidence_level
        ):
            raise ValueError(
                "loss-span anchors have insufficient confidence for the "
                "declared combined v0 decision"
            )
        if route_semantics_by_layer is None:
            raise ValueError(
                "modern v0 requires explicit per-layer connection semantics"
            )
    else:
        estimand = "bce_increment_per_removed_route"
        total_increment_budget = 0.0
    increment_spent = 0.0
    required_fields = [
        "look_batch_images",
        "initial_density_step",
        "minimum_density_step",
    ]
    if not policy.precision_only_iid:
        required_fields.insert(0, "futility_horizon_images")
    for field_name in required_fields:
        if getattr(policy, field_name) is None:
            raise ValueError(f"{field_name} is underived; pass a NaturalizationContext")
    active = tuple(dict.fromkeys(str(value) for value in active_partitions))
    if not active or not set(active) <= {"positive", "negative"}:
        raise ValueError("active partitions must be positive and/or negative")
    if int(total_layer_route_count) <= 0:
        raise ValueError("total_layer_route_count must be positive")
    if maximum_accepted_decisions is not None and int(maximum_accepted_decisions) <= 0:
        raise ValueError("maximum accepted BCE decisions must be positive")
    density_schedule: tuple[float, ...] | None = None
    if executable_density_schedule is not None:
        values = tuple(float(value) for value in executable_density_schedule)
        if (
            len(values) < 2
            or not math.isclose(values[0], 1.0, rel_tol=0.0, abs_tol=1e-12)
            or any(
                not math.isfinite(value) or not 0.0 < value <= 1.0 for value in values
            )
            or any(later >= earlier for earlier, later in zip(values, values[1:]))
            or values[-1] < float(policy.minimum_density) - 1e-12
        ):
            raise ValueError(
                "executable density schedule must descend from one through "
                "positive densities no lower than the declared floor"
            )
        density_schedule = values

    if maximum_decisions is None and density_schedule is not None:
        retained_counts = getattr(build_candidate, "retained_counts_by_rung", None)
        if retained_counts is None:
            prepared = getattr(build_candidate, "prepared", None)
            if isinstance(prepared, Mapping):
                # Historical layerwise lowering retains ceil(n_l,p * d)
                # within each layer; summing those counts also handles a
                # caller that supplies several layers to the same lowerer.
                layer_counts = tuple(
                    retained_counts_for_density_schedule(
                        {partition: int(getattr(layer, partition).size) for partition in active},
                        density_schedule,
                    )
                    for layer in prepared.values()
                )
                retained_counts = {
                    partition: tuple(
                        sum(counts[partition][index] for counts in layer_counts)
                        for index in range(len(density_schedule))
                    )
                    for partition in active
                }
            elif context is not None and context.partition_route_counts:
                retained_counts = retained_counts_for_density_schedule(
                    {partition: int(context.partition_route_counts.get(partition, 0)) for partition in active},
                    density_schedule,
                )
        if retained_counts is not None:
            maximum_decisions = maximum_decisions_from_retained_counts(
                {partition: retained_counts[partition] for partition in active}
            )
        else:
            # Legacy callable lowerers expose no integer counts. Their
            # declared schedule still bounds J_p by its rung positions;
            # deriving that bound does not execute a stateful builder.
            maximum_decisions = 2 * len(active) * (len(density_schedule) - 1) + 2
    elif maximum_decisions is None:
        # Preserve the historical unscheduled bisection API's horizon.
        # Current v0 always supplies its rung rule and uses the bound above.
        maximum_decisions = 512
    # With the v6 group lowerer, the committed masks are
    # a CommittedMaskState and every candidate an order-slice CandidateView.
    # ``initial_state`` returns None unless the committed masks are exactly
    # the density-1 lowering over the fixed earlier groups, which is the
    # invariant the slice arithmetic relies on; on None, dense masks are used.
    state = None
    if (
        callable(getattr(build_candidate, "build_candidate_view", None))
        and callable(getattr(build_candidate, "initial_state", None))
    ):
        state = build_candidate.initial_state(committed_masks)
    v6_views = state is not None
    # Legacy receipts independently rederive the statistical action. They
    # cannot bind a forced flip, so refuse before the first decision receipt.
    if FORCED_ACTIONS_HOOK is not None and not v6_views:
        _v6_forced_hook()  # validate the provider even on a refused path
        raise RuntimeError(
            "forced-action hook requires the v6 candidate-view decision path"
        )
    walk_log = (
        TraceWalkLog(query_id=str(query_id), layer_id=str(layer_id))
        if v6_views
        else None
    )
    # Forced hook: the installed versioned provider owns its reference log.
    forced_actions = (
        None if FORCED_ACTIONS_HOOK is not None else forced_actions_from_env()
    )
    if v6_views:
        masks = state
    else:
        masks = {
            str(key): np.asarray(value, dtype=bool).copy()
            for key, value in committed_masks.items()
        }
    densities = {"positive": 1.0, "negative": 1.0}
    steps = {
        "positive": float(policy.initial_density_step),
        "negative": float(policy.initial_density_step),
    }
    warm_starts: dict[str, Mapping[str, Any]] = {}
    decisions: list[ContinuousDecision] = []
    sampled_panel_receipts: list[Mapping[str, Any]] = []
    calibration_precision_receipts: list[Mapping[str, Any]] = []
    oracle_census_receipts: list[Mapping[str, Any]] = []
    connection_delta_receipts: list[Mapping[str, Any]] = []
    sequential_decision_receipts: list[Mapping[str, Any]] = []
    proposal_index = 0
    accepted_count = 0
    turn = initial_turn if initial_turn in active else active[0]
    stopping_reason = "frontier_resolved_in_both_partitions"
    schedule_indices = {"positive": 0, "negative": 0}
    frozen_partitions: set[str] = set()
    # Classic-v0 stall semantics: on a registered schedule a refused or
    # unproved rung is a stalled TURN, not a terminal state for the
    # partition. The cursor stays on the rung and the partition re-proposes
    # it on its next turn, after the opposing partition's commits have
    # changed the committed landscape. Two consecutive zero-commit turns
    # stop the group; a partition at its ladder floor (or inactive) counts
    # its turns as stalls, so a lone live partition gets exactly one
    # unproved rung before the group stops.
    stalled_turns = 0

    transition_parameters = dict(
        build_candidate=build_candidate, policy=policy, active=active,
        density_schedule=density_schedule, v6_views=v6_views,
        modern_v0_mode=modern_v0_mode, aggregate_boundary_mode=aggregate_boundary_mode,
        maximum_decisions=maximum_decisions,
        maximum_accepted_decisions=maximum_accepted_decisions,
    )
    cursor = _FrontierCursor(
        masks, densities, steps, schedule_indices, frozenset(frozen_partitions), turn,
    )
    prewarm_statistics = getattr(backend, "prewarm_device_statistics", None)
    if callable(prewarm_statistics):
        prewarm_statistics(policy)
    cursor, proposal = _frontier_step_to_next_proposal(
        cursor, **transition_parameters,
        images_consumed=getattr(backend, "decision_images_consumed", 0),
    )
    preplanning = (
        (host_preplanning_enabled() or getattr(getattr(getattr(backend, "provider", None), "_device", None), "type", None) == "cuda")
        and modern_v0_mode and v6_views
        and bool(policy.precision_only_iid)
        and callable(getattr(backend, "prepare_host_branch", None))
        and bool(getattr(getattr(backend, "provider", None), "iid_design_mode", False))
    )
    if preplanning:
        diagnostics = backend.host_preplanning_diagnostics
    defer_receipts=(v6_views and modern_v0_mode and policy.resolved_decision_law_id == SEQUENTIAL_CALIBRATED_BCE_V5
        and getattr(backend,"device_bce_statistics_supported",False)
        and not policy.accept_replication_gate
        and forced_actions is None and FORCED_ACTIONS_HOOK is None)
    step_accumulator = getattr(getattr(backend, "provider", None), "step_category_accumulator", None)
    cuda_step_records = (step_accumulator is not None and
        getattr(getattr(getattr(backend, "provider", None), "_device", None), "type", None) == "cuda")
    if defer_receipts or cuda_step_records:
        from mechanistic_systems.circuits.device_bce_decision import receipt_worker
        receipt_executor = receipt_worker()
    else:
        receipt_executor = None
    if cuda_step_records:
        step_accumulator.set_receipt_executor(receipt_executor)
    pending_receipts=[]

    def drain_receipts(*,block):
        while pending_receipts and (block or pending_receipts[0][0].done()):
            future,index,ordinal,completion=pending_receipts.pop(0)
            resolved=future.result()
            walk_log.records[ordinal]=dict(resolved.decision_record)
            walk_log.looks.extend((ordinal,*look) for look in resolved.decision_record.get("per_look",()))
            decisions[index]=replace(decisions[index],mean_sensitivity=resolved.mean,
                certified_lower_bound=resolved.lower,certified_upper_bound=resolved.upper)
            if progress_callback is not None:
                progress_callback("v0_proposal_decision_completed",**completion,
                    estimate=resolved.mean,lower=resolved.lower,upper=resolved.upper)

    pending_bookkeeping = None
    pending_planning = None

    def flush_bookkeeping():
        nonlocal pending_bookkeeping
        if pending_bookkeeping is not None:
            record, pending_bookkeeping = pending_bookkeeping, None
            started = time.perf_counter()
            record()
            diagnostics["bookkeeping_host_seconds"] += time.perf_counter() - started

    def record_decision(completed, evidence, proposal_threshold, masks_after):
        # This receives snapshots, never the loop's subsequently changed locals.
        turn = completed.cursor.turn
        densities = completed.cursor.densities
        proposed, removed = completed.proposed_density, completed.removed
        proposal_index = completed.cursor.proposal_index
        masks = masks_after
        decision_ordinal = (
            walk_log.add_decision(evidence.decision_record)
            if walk_log is not None
            and getattr(evidence, "decision_record", None) is not None
            else None
        )
        sampled_panel_receipts.extend(evidence.panel_receipts)
        calibration_precision_receipts.extend(evidence.calibration_precision_receipts)
        oracle_census_receipts.extend(evidence.oracle_census_receipts)
        if evidence.connection_delta_receipt is not None:
            connection_delta_receipts.append(evidence.connection_delta_receipt)
        if evidence.decision_receipt is not None:
            sequential_decision_receipts.append(evidence.decision_receipt)
        decisions.append(
            ContinuousDecision(
                partition_id=turn,
                previous_density=densities[turn],
                proposed_density=proposed,
                action=evidence.action,
                reason=evidence.reason,
                looks=evidence.looks,
                panel_images=evidence.panel_images,
                mean_sensitivity=evidence.mean,
                certified_lower_bound=evidence.lower,
                certified_upper_bound=evidence.upper,
                threshold=proposal_threshold,
                delta_debit=0.0,
                alpha_allocated=float(policy.rule.error_probability),
                panel_hashes=evidence.panel_hashes,
                removed_route_count=int(removed),
                confidence_level=float(policy.rule.confidence_level),
                interval_method=(
                    STRATIFIED_T_METHOD
                    if modern_v0_mode
                    else BCE_PRECISION_INTERVAL_METHOD
                ),
            )
        )
        future=getattr(evidence,"receipt_future",None)
        if future is not None:
            pending_receipts.append((future,len(decisions)-1,decision_ordinal,
                dict(query_id=str(query_id),layer_id=str(layer_id),partition_id=str(turn),
                    proposal_index=int(proposal_index),previous_density=float(densities[turn]),
                    proposed_density=float(proposed),density_step=float(densities[turn]-proposed),
                    removed_connection_count=int(removed),action=str(evidence.action),reason=str(evidence.reason),
                    looks=int(evidence.looks),panel_images=int(evidence.panel_images),threshold=float(proposal_threshold))))
        elif progress_callback is not None:
            progress_callback(
                "v0_proposal_decision_completed",
                query_id=str(query_id),
                layer_id=str(layer_id),
                partition_id=str(turn),
                proposal_index=int(proposal_index),
                previous_density=float(densities[turn]),
                proposed_density=float(proposed),
                density_step=float(densities[turn] - proposed),
                removed_connection_count=int(removed),
                action=str(evidence.action),
                reason=str(evidence.reason),
                looks=int(evidence.looks),
                panel_images=int(evidence.panel_images),
                estimate=float(evidence.mean),
                lower=float(evidence.lower),
                upper=float(evidence.upper),
                threshold=float(proposal_threshold),
            )
        if evidence.action == "accept" and walk_log is not None:
            walk_log.add_accept(decision_ordinal=decision_ordinal, view=completed.candidate)
        proposal_index += 1
        category_outcome_recorder = getattr(
            backend, "record_step_category_outcome", None
        )
        if callable(category_outcome_recorder):
            category_outcome_recorder(
                query_id=str(query_id),
                layer_id=str(layer_id),
                proposal_index=int(proposal_index - 1),
                partition_id=str(turn),
                action=str(evidence.action),
                reason=str(evidence.reason),
                committed_masks_after=masks,
                decision_receipt=evidence.decision_receipt,
                    **(
                        {
                            "engine_outcome": evidence.decision_record["engine_outcome"],
                            "engine_reason": evidence.decision_record["engine_reason"],
                            "engine_forecast_target": evidence.decision_record["engine_forecast_target"],
                        }
                        if getattr(evidence, "decision_record", None)
                        and int(evidence.decision_record["flags"]) & FLAG_FORCED
                        else {}
                    ),
            )

    try:
        while proposal is not None:
            drain_receipts(block=False)
            cursor = proposal.cursor
            masks, candidate = cursor.masks, proposal.candidate
            turn, proposed, removed = cursor.turn, proposal.proposed_density, proposal.removed
            densities, proposal_index = cursor.densities, cursor.proposal_index
            def prepare_branches():
                plans = {}
                for branch_action in ("accept", "reject"):
                    started = time.perf_counter()
                    try:
                        _branch_cursor, upcoming = _frontier_step_to_next_proposal(
                            planning_cursor, **transition_parameters, completed=planning_proposal,
                            action=branch_action,
                            images_consumed=planning_images,
                        )
                        if upcoming is None:
                            plans[branch_action] = (None, None)
                            continue
                        request = _bce_comparison_request(
                            query_id=str(query_id), layer_id=str(layer_id),
                            partition_id=upcoming.cursor.turn,
                            previous_density=upcoming.cursor.densities[upcoming.cursor.turn],
                            proposed_density=upcoming.proposed_density,
                            looks=1, target_images=int(policy.look_batch_images),
                            class_count=class_count, proposal_index=upcoming.cursor.proposal_index,
                            policy=policy, prior_panel=None, warm_starts={},
                            committed_hash=None, candidate_hash=None, allocation_evidence={},
                            precision_only_iid=True, remaining_group=math.inf, maximum_target=None,
                            calibration_target_loss_precision=None,
                        )
                        prepared = backend.prepare_host_branch(upcoming.cursor.masks, upcoming.candidate, request)
                        # The planner waits for its independent draw worker,
                        # leaving the main thread free to enqueue the look.
                        for packet in prepared.draws:
                            drawn = packet.result()
                            for name, seconds in (
                                ("planning_draw_worker_seconds", drawn.preparation_wall_seconds),
                                ("planning_draw_worker_thread_cpu_seconds", drawn.preparation_thread_cpu_seconds),
                                ("planning_draw_sha_packet_seconds", drawn.sha_packet_wall_seconds),
                                ("planning_draw_native_sha_seconds", drawn.sha_native_wall_seconds),
                                ("planning_draw_native_sha_thread_cpu_seconds", drawn.sha_native_thread_cpu_seconds),
                                ("planning_draw_gil_cpu_upper_bound_seconds", drawn.preparation_gil_cpu_upper_bound_seconds),
                                ("planning_draw_host_objects_thread_cpu_seconds", max(
                                    drawn.preparation_thread_cpu_seconds - drawn.sha_packet_thread_cpu_seconds, 0.0)),
                            ):
                                diagnostics[name] = diagnostics.get(name, 0.0) + seconds
                        plans[branch_action] = (upcoming, prepared)
                        # Accept images start arriving as soon as their rows
                        # are ready, while the reject description is built.
                        planning_window.wait()
                        backend.post_host_branch_predictions(tuple(
                            plans[action][1] for action in ("accept", "reject")
                            if plans.get(action, (None, None))[1] is not None
                        ))
                    finally:
                        diagnostics["preparation_host_seconds"] += time.perf_counter() - started
                return plans

            def prepare_next(ready):
                # The complete planner task was submitted before enqueue.
                # The before-read hook performs no candidate or row work.
                planning_window.set()

            connection_receipt = (
                ConnectionDelta.from_layer_counts(
                    layer_ids=tuple(masks),
                    removed_by_layer=candidate.removed_counts_by_layer(),
                    route_semantics_by_layer=route_semantics_by_layer,
                )
                if v6_views and modern_v0_mode and route_semantics_by_layer is not None
                else CircuitConnectionDeltaReceipt.from_masks(
                    parent_masks=masks, candidate_masks=candidate,
                    route_semantics_by_layer=route_semantics_by_layer,
                    parent_mask_hash=sampled_path_mask_hash(masks),
                    candidate_mask_hash=sampled_path_mask_hash(candidate),
                )
                if modern_v0_mode and route_semantics_by_layer is not None else None
            )
            proposal_threshold = (
                max(total_increment_budget - cursor.increment_spent, 0.0)
                if aggregate_boundary_mode else
                float(policy.rule.maximum_candidate_increment) if direct_candidate_mode
                else float(policy.rule.sensitivity_tau)
            )
            if preplanning:
                begin = getattr(backend, "begin_host_preplanning", None)
                if callable(begin):
                    begin()
                planning_cursor = replace(cursor, masks=cursor.masks.fork())
                planning_proposal = replace(proposal, cursor=planning_cursor,
                    candidate=proposal.candidate.fork(parent=planning_cursor.masks))
                planning_images = getattr(backend, "decision_images_consumed", 0)
                planning_window = threading.Event()
                pending_planning = plan_host_branches(prepare_branches, diagnostics)
            window = backend.host_preplanning_window(prepare_next, flush_bookkeeping) if preplanning else nullcontext()
            with window:
                evidence = _decide_bce_precision_step(
                    backend=backend, committed_masks=masks, candidate_masks=candidate,
                    query_id=str(query_id), layer_id=str(layer_id), partition_id=turn,
                    previous_density=densities[turn], proposed_density=proposed,
                    proposal_index=proposal_index, removed_route_count=removed,
                    threshold=proposal_threshold, estimand=estimand,
                    connection_delta_receipt=connection_receipt, loss_span_certificate=loss_span_certificate,
                    policy=policy, warm_starts=warm_starts, class_count=class_count,
                    progress_callback=progress_callback, step_context_delta_sink=step_context_delta_sink,
                    forced_action=(None if forced_actions is None else forced_actions.lookup(
                        query_id=str(query_id), layer_id=str(layer_id), partition_id=str(turn),
                        proposal_index=int(proposal_index),
                    )), v6_record=v6_views,
                    **({"receipt_executor":receipt_executor} if defer_receipts else {}),
                )
            # A census re-projection or an early typed exit can contain no
            # device enqueue. Finish the preceding record on that path too.
            flush_bookkeeping()
            if preplanning:
                planning_window.set()
            plans = pending_planning.result() if pending_planning is not None else {}
            pending_planning = None
            # Delivery checks of the looks just decided: their flags were
            # copied before the decision's read, so this never waits.
            from mechanistic_systems.measurement.device_deferred import settle_device_faults

            settle_device_faults()
            branch = "accept" if evidence.action == "accept" else "reject"
            preferred, prepared = plans.get(branch, (None, None))
            next_cursor, upcoming = _frontier_step_to_next_proposal(
                cursor, **transition_parameters, completed=proposal, action=evidence.action,
                upper=evidence.upper, images_consumed=getattr(backend, "decision_images_consumed", 0),
                preferred=preferred,
            )
            if preplanning:
                backend.select_host_prepared_branch(prepared if upcoming is preferred and upcoming is not None else None)
                if upcoming is preferred and upcoming is not None:
                    backend.post_host_prepared_branch(prepared)
            completed = proposal
            record = lambda completed=completed, evidence=evidence, threshold=proposal_threshold, after=next_cursor.masks: record_decision(completed, evidence, threshold, after)
            if preplanning and upcoming is not None:
                pending_bookkeeping = record
                diagnostics["deferred_records"] += 1
            else:
                record()
            cursor, proposal = next_cursor, upcoming
    finally:
        try:
            if pending_planning is not None:
                planning_window.set()
                pending_planning.result()
        finally:
            flush_bookkeeping()
            try:
                drain_receipts(block=True)
                if cuda_step_records:
                    step_accumulator.drain_receipts()
            finally:
                if receipt_executor is not None:
                    receipt_executor.shutdown(wait=True)
                if preplanning:
                    backend.select_host_prepared_branch(None)
                    backend.provider._host_prepared_draws = {}
    # The group's remaining receipts and delivery flags, in bulk (waits once).
    settle_deferred = getattr(backend, "settle_deferred_device_work", None)
    if callable(settle_deferred):
        settle_deferred()
    masks = cursor.masks
    state = masks if v6_views else None
    densities, steps = cursor.densities, cursor.steps
    schedule_indices, frozen_partitions = cursor.schedule_indices, cursor.frozen_partitions
    proposal_index, accepted_count = cursor.proposal_index, cursor.accepted_count
    stalled_turns, increment_spent = cursor.stalled_turns, cursor.increment_spent
    stopping_reason = cursor.stopping_reason

    if progress_callback is not None:
        progress_callback(
            "v0_group_frontier_completed",
            query_id=str(query_id),
            layer_id=str(layer_id),
            proposal_count=int(len(decisions)),
            accepted_count=int(accepted_count),
            positive_density=float(densities["positive"]),
            negative_density=float(densities["negative"]),
            retained_connection_count=int(
                sum(
                    np.count_nonzero(np.asarray(mask, dtype=bool))
                    for mask in masks.values()
                )
            ),
            stopping_reason=str(stopping_reason),
        )

    confidence = float(policy.rule.confidence_level)
    walk_log_handle = None
    if v6_views:
        walk_log.final_mask_hashes = {
            str(route_axis_hash): str(mask.mask_hash)
            for (route_axis_hash, _layout), mask in state._hard_masks.items()
        }
        walk_log_handle = TraceWalkLogHandle(walk_log)
        # The result carries plain arrays for every downstream consumer.
        masks = state.as_dict()
        sequential_decision_receipts = []
    receipt = {
        "schema_name": (
            V6_LEDGER_SCHEMA if v6_views else "bce_precision_decision_ledger_v1"
        ),
        "algorithm_version": (
            policy.resolved_decision_law_id
            if modern_v0_mode
            else (
                SEQUENTIAL_CALIBRATED_BCE_V3
                if direct_candidate_mode
                else BCE_PRECISION_V1
            )
        ),
        "decision_law": _decision_law_block(policy),
        "confidence_scope": (
            (
                PRECISION_ONLY_IID_SCOPE
                if policy.precision_only_iid
                else PLANNED_LOOK_SCOPE
            )
            if modern_v0_mode or direct_candidate_mode
            else "per_proposal_simultaneous_over_planned_looks"
        ),
        "confidence_level": confidence,
        "estimand": estimand,
        "decision_boundary": (
            float(policy.rule.maximum_bce_increment)
            if aggregate_boundary_mode
            else (
                float(policy.rule.maximum_candidate_increment)
                if direct_candidate_mode
                else float(policy.rule.sensitivity_tau)
            )
        ),
        "decision_boundary_name": (
            "maximum_bce_increment"
            if aggregate_boundary_mode
            else (
                "maximum_candidate_increment"
                if direct_candidate_mode
                else "sensitivity_tau"
            )
        ),
        "proposal_count": len(decisions),
        "accepted_count": accepted_count,
        "maximum_accepted_decisions": maximum_accepted_decisions,
        "sampled_panel_timing_receipts": tuple(sampled_panel_receipts),
        "calibration_precision_horizon_receipts": tuple(calibration_precision_receipts),
        "oracle_finite_population_census_receipts": tuple(oracle_census_receipts),
        "connection_delta_receipts": tuple(connection_delta_receipts),
        "sequential_bce_decision_receipts": tuple(sequential_decision_receipts),
        "unresolved_action": "no_commit",
        "claim_status": "exploratory_raw_bce_unbounded",
        "loss_span_certificate_hash": (
            None
            if loss_span_certificate is None
            else loss_span_certificate.content_hash
        ),
    }
    if walk_log_handle is not None:
        # The decision log, mask-delta log and removal log replace the
        # JSON decision and connection receipts (kind 3, compact binary).
        receipt["v6_walk_log"] = walk_log_handle
        receipt["decision_record_count"] = int(walk_log_handle.decision_count)
    return ContinuousEvidenceResult(
        positive_density=float(densities["positive"]),
        negative_density=float(densities["negative"]),
        committed_masks=masks,
        decisions=tuple(decisions),
        stopping_reason=stopping_reason,
        delta_budget=total_increment_budget,
        delta_spent=increment_spent,
        images_consumed=int(getattr(backend, "decision_images_consumed", 0)),
        ledger_receipt=receipt,
        moderation_prior=None,
        resolved_policy={
            "confidence_level": confidence,
            "estimand": estimand,
            "decision_boundary": receipt["decision_boundary"],
            "decision_boundary_name": receipt["decision_boundary_name"],
            "confidence_scope": receipt["confidence_scope"],
            "interval_method": _resolved_interval_method(
                policy,
                modern_v0_mode=modern_v0_mode,
                direct_candidate_mode=direct_candidate_mode,
            ),
            "futility_horizon_images": (
                None
                if policy.futility_horizon_images is None
                else int(policy.futility_horizon_images)
            ),
            "look_batch_images": int(policy.look_batch_images),
            "initial_density_step": float(policy.initial_density_step),
            "minimum_density_step": float(policy.minimum_density_step),
            "minimum_density": float(policy.minimum_density),
            "executable_density_schedule": density_schedule,
            "maximum_decisions": maximum_decisions,
            "density_topology": (
                "registered_multiplicative_schedule_retry_until_both_stalled"
                if density_schedule is not None
                else "accept_expand_reject_bisect"
            ),
            "finite_population_correction": bool(policy.finite_population_correction),
            "maximum_planned_looks": policy.maximum_planned_looks,
            "precision_only_iid": bool(policy.precision_only_iid),
            "image_budget": policy.image_budget,
            "runtime_deadline_monotonic": (policy.runtime_deadline_monotonic),
            "decision_law": receipt["decision_law"],
        },
        resource_usage=(
            dict(backend.resource_usage)
            if isinstance(getattr(backend, "resource_usage", None), Mapping)
            else None
        ),
        algorithm_version=(
            policy.resolved_decision_law_id
            if modern_v0_mode
            else (
                SEQUENTIAL_CALIBRATED_BCE_V3
                if direct_candidate_mode
                else BCE_PRECISION_V1
            )
        ),
    )


__all__ = [
    "BCE_PRECISION_DECISION_LAW_V1",
    "BCE_PRECISION_INTERVAL_METHOD",
    "BCE_PRECISION_V1",
    "BCEPrecisionDecisionRule",
    "BCEPrecisionPolicy",
    "BCELossBoundaryDecisionRule",
    "CENSUS_EXHAUSTED_UNRESOLVED_REASON",
    "CENSUS_TERMINAL_NO_COMMIT_REASON",
    "DirectCandidateIncrementDecisionRule",
    "CalibrationPrecisionHorizonExhausted",
    "FixedCalibrationPairedBCEEvidence",
    "ORACLE_ACQUISITION_ENVELOPE_RECEIPT_V1",
    "ORACLE_ENVELOPE_UNRESOLVED_REASON",
    "ORACLE_METADATA_CENSUS_COMPLETE",
    "ORACLE_METADATA_CENSUS_EXACT_WEIGHT",
    "ORACLE_METADATA_CENSUS_STRATA_COUNT",
    "ORACLE_METADATA_CENSUS_TERMINAL_REACHED",
    "ORACLE_METADATA_DRAWN_BY_STRATUM",
    "ORACLE_METADATA_EXHAUSTED_STRATA",
    "ORACLE_METADATA_FPC_APPLIED",
    "ORACLE_METADATA_KEYS",
    "ORACLE_METADATA_PANEL_AXIS_CHAIN",
    "ORACLE_METADATA_POPULATION_BY_STRATUM",
    "ORACLE_METADATA_REMAINING_POPULATION",
    "ORACLE_METADATA_SAMPLING_LAW",
    "ORACLE_METADATA_SAMPLING_WITH_REPLACEMENT",
    "OracleFinitePopulationCensusExhausted",
    "RelativeBCELossBoundary",
    "V0MarginalConnectionDecisionRule",
    "V0_DECISION_TERMINAL_REASONS",
    "trace_bce_precision_frontier",
]
